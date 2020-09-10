class Ray
{
    private:
        Vector position;
        Vector velocity;

    public:
        Ray( Vector Pi, Vector Vi )
        {
            position = Pi;
            velocity = Vi;
        }
};
