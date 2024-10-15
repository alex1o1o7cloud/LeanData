import Mathlib

namespace NUMINAMATH_CALUDE_no_integer_points_between_l3090_309098

def point_C : ℤ × ℤ := (2, 3)
def point_D : ℤ × ℤ := (101, 200)

def is_between (a b c : ℤ) : Prop := a < b ∧ b < c

theorem no_integer_points_between :
  ¬ ∃ (x y : ℤ), 
    (is_between point_C.1 x point_D.1) ∧ 
    (is_between point_C.2 y point_D.2) ∧ 
    (y - point_C.2) * (point_D.1 - point_C.1) = (point_D.2 - point_C.2) * (x - point_C.1) :=
sorry

end NUMINAMATH_CALUDE_no_integer_points_between_l3090_309098


namespace NUMINAMATH_CALUDE_no_two_digit_number_satisfies_conditions_l3090_309036

theorem no_two_digit_number_satisfies_conditions : ¬∃ n : ℕ,
  10 ≤ n ∧ n < 100 ∧  -- two-digit number
  Even n ∧            -- even
  n % 13 = 0 ∧        -- multiple of 13
  ∃ a b : ℕ,          -- digits a and b
    n = 10 * a + b ∧
    0 ≤ a ∧ a ≤ 9 ∧
    0 ≤ b ∧ b ≤ 9 ∧
    ∃ k : ℕ, a * b = k * k  -- product of digits is a perfect square
  := by sorry

end NUMINAMATH_CALUDE_no_two_digit_number_satisfies_conditions_l3090_309036


namespace NUMINAMATH_CALUDE_triangle_segment_length_l3090_309089

theorem triangle_segment_length (a b c h x : ℝ) : 
  a = 24 → b = 45 → c = 51 → 
  a^2 = x^2 + h^2 → 
  b^2 = (c - x)^2 + h^2 → 
  c - x = 40 := by
  sorry

end NUMINAMATH_CALUDE_triangle_segment_length_l3090_309089


namespace NUMINAMATH_CALUDE_ellipse_circle_tangent_perpendicular_l3090_309050

/-- Ellipse M with focal length 2√3 and eccentricity √3/2 -/
def ellipse_M (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1

/-- Circle N with radius r -/
def circle_N (x y r : ℝ) : Prop :=
  x^2 + y^2 = r^2

/-- Tangent line l with slope k -/
def line_l (x y k m : ℝ) : Prop :=
  y = k * x + m

/-- P and Q are intersection points of line l and ellipse M -/
def intersection_points (P Q : ℝ × ℝ) (k m : ℝ) : Prop :=
  let (x₁, y₁) := P
  let (x₂, y₂) := Q
  ellipse_M x₁ y₁ ∧ ellipse_M x₂ y₂ ∧
  line_l x₁ y₁ k m ∧ line_l x₂ y₂ k m

/-- OP and OQ are perpendicular -/
def perpendicular (P Q : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := P
  let (x₂, y₂) := Q
  x₁ * x₂ + y₁ * y₂ = 0

theorem ellipse_circle_tangent_perpendicular (k m r : ℝ) (P Q : ℝ × ℝ) :
  m^2 = r^2 * (k^2 + 1) →
  intersection_points P Q k m →
  (perpendicular P Q ↔ r = 2 * Real.sqrt 5 / 5) :=
sorry

end NUMINAMATH_CALUDE_ellipse_circle_tangent_perpendicular_l3090_309050


namespace NUMINAMATH_CALUDE_rectangle_division_theorem_l3090_309060

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle in 2D space -/
structure Rectangle where
  bottomLeft : Point
  topRight : Point

/-- Checks if a point is inside a rectangle -/
def pointInRectangle (p : Point) (r : Rectangle) : Prop :=
  r.bottomLeft.x ≤ p.x ∧ p.x ≤ r.topRight.x ∧
  r.bottomLeft.y ≤ p.y ∧ p.y ≤ r.topRight.y

/-- Theorem: Given a rectangle with 4 points, it can be divided into 4 equal rectangles, each containing one point -/
theorem rectangle_division_theorem 
  (r : Rectangle) 
  (p1 p2 p3 p4 : Point) 
  (h1 : pointInRectangle p1 r)
  (h2 : pointInRectangle p2 r)
  (h3 : pointInRectangle p3 r)
  (h4 : pointInRectangle p4 r)
  (h_distinct : p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4) :
  ∃ (r1 r2 r3 r4 : Rectangle),
    -- The four rectangles are equal in area
    (r1.topRight.x - r1.bottomLeft.x) * (r1.topRight.y - r1.bottomLeft.y) =
    (r2.topRight.x - r2.bottomLeft.x) * (r2.topRight.y - r2.bottomLeft.y) ∧
    (r1.topRight.x - r1.bottomLeft.x) * (r1.topRight.y - r1.bottomLeft.y) =
    (r3.topRight.x - r3.bottomLeft.x) * (r3.topRight.y - r3.bottomLeft.y) ∧
    (r1.topRight.x - r1.bottomLeft.x) * (r1.topRight.y - r1.bottomLeft.y) =
    (r4.topRight.x - r4.bottomLeft.x) * (r4.topRight.y - r4.bottomLeft.y) ∧
    -- Each smaller rectangle contains exactly one point
    (pointInRectangle p1 r1 ∧ pointInRectangle p2 r2 ∧ pointInRectangle p3 r3 ∧ pointInRectangle p4 r4) ∧
    -- The union of the smaller rectangles is the original rectangle
    (r1.bottomLeft = r.bottomLeft) ∧ (r4.topRight = r.topRight) ∧
    (r1.topRight.x = r2.bottomLeft.x) ∧ (r2.topRight.x = r.topRight.x) ∧
    (r1.topRight.y = r3.bottomLeft.y) ∧ (r3.topRight.y = r.topRight.y) :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_division_theorem_l3090_309060


namespace NUMINAMATH_CALUDE_tracy_candies_problem_l3090_309082

theorem tracy_candies_problem (x : ℕ) : 
  (x % 4 = 0) →  -- x is divisible by 4
  (x % 2 = 0) →  -- x is divisible by 2
  (∃ y : ℕ, 2 ≤ y ∧ y ≤ 6 ∧ x / 2 - 20 - y = 5) →  -- sister took between 2 to 6 candies, leaving 5
  x = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_tracy_candies_problem_l3090_309082


namespace NUMINAMATH_CALUDE_range_of_a_l3090_309049

/-- The function g(x) = ax + 2 where a > 0 -/
def g (a : ℝ) (x : ℝ) : ℝ := a * x + 2

/-- The function f(x) = x^2 + 2x -/
def f (x : ℝ) : ℝ := x^2 + 2*x

theorem range_of_a (a : ℝ) :
  (a > 0) →
  (∀ x₁ ∈ Set.Icc (-1 : ℝ) 1, ∃ x₀ ∈ Set.Icc (-2 : ℝ) 1, g a x₁ = f x₀) →
  a ∈ Set.Ioo 0 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3090_309049


namespace NUMINAMATH_CALUDE_vector_b_calculation_l3090_309015

theorem vector_b_calculation (a b : ℝ × ℝ) : 
  a = (1, 2) → (2 • a + b = (3, 2)) → b = (1, -2) := by sorry

end NUMINAMATH_CALUDE_vector_b_calculation_l3090_309015


namespace NUMINAMATH_CALUDE_perpendicular_lines_parallel_l3090_309097

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the parallel relation between two lines
variable (parallel : Line → Line → Prop)

-- Theorem statement
theorem perpendicular_lines_parallel (a b : Line) (α : Plane) :
  perpendicular a α → perpendicular b α → parallel a b :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_parallel_l3090_309097


namespace NUMINAMATH_CALUDE_exponential_inequality_l3090_309090

theorem exponential_inequality (a x : ℝ) : 
  a > Real.log 2 - 1 → x > 0 → Real.exp x > x^2 - 2*a*x + 1 := by sorry

end NUMINAMATH_CALUDE_exponential_inequality_l3090_309090


namespace NUMINAMATH_CALUDE_total_saltwater_animals_l3090_309018

/-- The number of aquariums -/
def num_aquariums : ℕ := 20

/-- The number of animals per aquarium -/
def animals_per_aquarium : ℕ := 2

/-- Theorem stating the total number of saltwater animals -/
theorem total_saltwater_animals : 
  num_aquariums * animals_per_aquarium = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_saltwater_animals_l3090_309018


namespace NUMINAMATH_CALUDE_root_sum_fraction_l3090_309004

theorem root_sum_fraction (α β γ : ℂ) : 
  α^3 - α - 1 = 0 → β^3 - β - 1 = 0 → γ^3 - γ - 1 = 0 →
  (1 + α) / (1 - α) + (1 + β) / (1 - β) + (1 + γ) / (1 - γ) = -7 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_fraction_l3090_309004


namespace NUMINAMATH_CALUDE_sqrt_abs_sum_eq_one_l3090_309091

theorem sqrt_abs_sum_eq_one (a : ℝ) (h : 1 < a ∧ a < 2) :
  Real.sqrt ((a - 2)^2) + |a - 1| = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_abs_sum_eq_one_l3090_309091


namespace NUMINAMATH_CALUDE_five_balls_three_boxes_l3090_309052

/-- Represents the number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that there are 18 ways to distribute 5 indistinguishable balls into 3 distinguishable boxes -/
theorem five_balls_three_boxes : distribute_balls 5 3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_three_boxes_l3090_309052


namespace NUMINAMATH_CALUDE_unfair_coin_probability_l3090_309013

/-- The probability of getting exactly k heads in n tosses of a coin with probability p of heads -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

/-- The main theorem -/
theorem unfair_coin_probability (p : ℝ) (h1 : 0 ≤ p) (h2 : p ≤ 1) :
  binomial_probability 7 4 p = 210 / 1024 → p = 4 / 7 := by
  sorry

#check unfair_coin_probability

end NUMINAMATH_CALUDE_unfair_coin_probability_l3090_309013


namespace NUMINAMATH_CALUDE_essay_competition_probability_l3090_309093

/-- The number of topics in the essay competition -/
def num_topics : ℕ := 6

/-- The probability that two students select different topics -/
def prob_different_topics : ℚ := 5/6

/-- Theorem stating that the probability of two students selecting different topics
    from a pool of 6 topics is 5/6 -/
theorem essay_competition_probability :
  (num_topics : ℚ) * (num_topics - 1) / (num_topics * num_topics) = prob_different_topics :=
sorry

end NUMINAMATH_CALUDE_essay_competition_probability_l3090_309093


namespace NUMINAMATH_CALUDE_negative_two_x_plus_two_positive_l3090_309075

theorem negative_two_x_plus_two_positive (x : ℝ) : x < 1 → -2*x + 2 > 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_two_x_plus_two_positive_l3090_309075


namespace NUMINAMATH_CALUDE_exam_mean_score_l3090_309001

theorem exam_mean_score (morning_mean : ℝ) (afternoon_mean : ℝ) (ratio : ℚ) 
  (h1 : morning_mean = 90)
  (h2 : afternoon_mean = 75)
  (h3 : ratio = 5 / 7) : 
  ∃ (overall_mean : ℝ), 
    (overall_mean ≥ 81 ∧ overall_mean < 82) ∧ 
    (∀ (m a : ℕ), m / a = ratio → 
      (m * morning_mean + a * afternoon_mean) / (m + a) = overall_mean) :=
sorry

end NUMINAMATH_CALUDE_exam_mean_score_l3090_309001


namespace NUMINAMATH_CALUDE_octal_to_decimal_l3090_309044

theorem octal_to_decimal (octal_num : ℕ) : octal_num = 362 → 
  (3 * 8^2 + 6 * 8^1 + 2 * 8^0) = 242 := by
  sorry

end NUMINAMATH_CALUDE_octal_to_decimal_l3090_309044


namespace NUMINAMATH_CALUDE_digit_150_is_6_l3090_309016

/-- The decimal representation of 17/270 as a sequence of digits -/
def decimalRepresentation : ℕ → Fin 10 := sorry

/-- The decimal representation of 17/270 is periodic with period 5 -/
axiom period_five : ∀ n : ℕ, decimalRepresentation n = decimalRepresentation (n + 5)

/-- The first period of the decimal representation -/
axiom first_period : 
  (decimalRepresentation 0 = 0) ∧
  (decimalRepresentation 1 = 6) ∧
  (decimalRepresentation 2 = 2) ∧
  (decimalRepresentation 3 = 9) ∧
  (decimalRepresentation 4 = 6)

/-- The 150th digit after the decimal point in 17/270 is 6 -/
theorem digit_150_is_6 : decimalRepresentation 149 = 6 := by sorry

end NUMINAMATH_CALUDE_digit_150_is_6_l3090_309016


namespace NUMINAMATH_CALUDE_stating_interest_rate_calculation_l3090_309028

/-- Represents the annual interest rate as a percentage -/
def annual_rate : ℝ := 15

/-- Represents the principal amount in rupees -/
def principal : ℝ := 147.69

/-- Represents the time period for the first deposit in years -/
def time1 : ℝ := 3.5

/-- Represents the time period for the second deposit in years -/
def time2 : ℝ := 10

/-- Represents the difference in interests in rupees -/
def interest_diff : ℝ := 144

/-- 
Theorem stating that given the conditions, the annual interest rate is approximately 15%.
-/
theorem interest_rate_calculation : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |annual_rate - (interest_diff * 100) / (principal * (time2 - time1))| < ε :=
sorry

end NUMINAMATH_CALUDE_stating_interest_rate_calculation_l3090_309028


namespace NUMINAMATH_CALUDE_garden_spaces_per_row_l3090_309011

/-- Represents a vegetable garden with given properties --/
structure Garden where
  tomatoes : Nat
  cucumbers : Nat
  potatoes : Nat
  rows : Nat
  additional_capacity : Nat

/-- Calculates the number of spaces in each row of the garden --/
def spaces_per_row (g : Garden) : Nat :=
  ((g.tomatoes + g.cucumbers + g.potatoes + g.additional_capacity) / g.rows)

/-- Theorem stating that for the given garden configuration, there are 15 spaces per row --/
theorem garden_spaces_per_row :
  let g : Garden := {
    tomatoes := 3 * 5,
    cucumbers := 5 * 4,
    potatoes := 30,
    rows := 10,
    additional_capacity := 85
  }
  spaces_per_row g = 15 := by
  sorry

end NUMINAMATH_CALUDE_garden_spaces_per_row_l3090_309011


namespace NUMINAMATH_CALUDE_x_squared_eq_neg_one_is_quadratic_l3090_309077

/-- A quadratic equation in one variable -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  h : a ≠ 0

/-- Check if an equation is in the form ax² + bx + c = 0 -/
def isQuadraticForm (f : ℝ → ℝ) : Prop :=
  ∃ (q : QuadraticEquation), ∀ x, f x = q.a * x^2 + q.b * x + q.c

/-- The specific equation x² = -1 -/
def f (x : ℝ) : ℝ := x^2 + 1

/-- Theorem: The equation x² = -1 is a quadratic equation in one variable -/
theorem x_squared_eq_neg_one_is_quadratic : isQuadraticForm f := by sorry

end NUMINAMATH_CALUDE_x_squared_eq_neg_one_is_quadratic_l3090_309077


namespace NUMINAMATH_CALUDE_matts_climbing_speed_l3090_309057

/-- Prove Matt's climbing speed given Jason's speed and their height difference after 7 minutes -/
theorem matts_climbing_speed 
  (jason_speed : ℝ) 
  (time : ℝ) 
  (height_diff : ℝ) 
  (h1 : jason_speed = 12)
  (h2 : time = 7)
  (h3 : height_diff = 42) :
  ∃ (matt_speed : ℝ), 
    matt_speed = 6 ∧ 
    jason_speed * time = matt_speed * time + height_diff :=
by sorry

end NUMINAMATH_CALUDE_matts_climbing_speed_l3090_309057


namespace NUMINAMATH_CALUDE_a_fourth_minus_b_fourth_l3090_309088

theorem a_fourth_minus_b_fourth (a b : ℝ) 
  (h1 : a - b = 1) 
  (h2 : a^2 - b^2 = -1) : 
  a^4 - b^4 = -1 := by
sorry

end NUMINAMATH_CALUDE_a_fourth_minus_b_fourth_l3090_309088


namespace NUMINAMATH_CALUDE_inscribed_square_properties_l3090_309032

theorem inscribed_square_properties (circle_area : ℝ) (h : circle_area = 324 * Real.pi) :
  let r : ℝ := Real.sqrt (circle_area / Real.pi)
  let d : ℝ := 2 * r
  let s : ℝ := d / Real.sqrt 2
  let square_area : ℝ := s ^ 2
  let total_diagonal_length : ℝ := 2 * d
  (square_area = 648) ∧ (total_diagonal_length = 72) := by
  sorry

#check inscribed_square_properties

end NUMINAMATH_CALUDE_inscribed_square_properties_l3090_309032


namespace NUMINAMATH_CALUDE_no_bounded_integral_exists_l3090_309099

/-- Base 2 representation of x in [0, 1) -/
def base2Rep (x : ℝ) : ℕ → Fin 2 :=
  sorry

/-- Function f_n as defined in the problem -/
def f_n (n : ℕ) (x : ℝ) : ℤ :=
  sorry

/-- The main theorem -/
theorem no_bounded_integral_exists :
  ∀ (φ : ℝ → ℝ),
    (∀ y, 0 ≤ φ y) →
    (∀ M, ∃ N, ∀ x, N ≤ x → M ≤ φ x) →
    (∀ B, ∃ n : ℕ, B < ∫ x in (0 : ℝ)..1, φ (|f_n n x|)) :=
  sorry

end NUMINAMATH_CALUDE_no_bounded_integral_exists_l3090_309099


namespace NUMINAMATH_CALUDE_multiples_4_or_9_less_than_201_l3090_309043

def is_multiple (n m : ℕ) : Prop := ∃ k, n = m * k

def count_multiples (max divisor : ℕ) : ℕ :=
  (max - 1) / divisor

def count_either_not_both (max a b : ℕ) : ℕ :=
  count_multiples max a + count_multiples max b - 2 * count_multiples max (lcm a b)

theorem multiples_4_or_9_less_than_201 :
  count_either_not_both 201 4 9 = 62 := by sorry

end NUMINAMATH_CALUDE_multiples_4_or_9_less_than_201_l3090_309043


namespace NUMINAMATH_CALUDE_cube_volume_problem_l3090_309095

theorem cube_volume_problem (a : ℝ) (h : a > 0) :
  (3 * a) * (a / 2) * a - a^3 = 2 * a^2 → a^3 = 64 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l3090_309095


namespace NUMINAMATH_CALUDE_equation_solution_l3090_309076

theorem equation_solution : ∃ x : ℚ, 3 * (x - 2) = x - (2 * x - 1) ∧ x = 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3090_309076


namespace NUMINAMATH_CALUDE_line_circle_intersection_range_l3090_309081

/-- The range of m for a line intersecting a circle under specific conditions -/
theorem line_circle_intersection_range (m : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    A ≠ B ∧ 
    A.1 + A.2 + m = 0 ∧ 
    B.1 + B.2 + m = 0 ∧ 
    A.1^2 + A.2^2 = 2 ∧ 
    B.1^2 + B.2^2 = 2 ∧ 
    ‖(A.1, A.2)‖ + ‖(B.1, B.2)‖ ≥ ‖(A.1 - B.1, A.2 - B.2)‖) →
  m ∈ Set.Ioo (-2 : ℝ) (-Real.sqrt 2) ∪ Set.Ioo (Real.sqrt 2) 2 :=
by sorry

end NUMINAMATH_CALUDE_line_circle_intersection_range_l3090_309081


namespace NUMINAMATH_CALUDE_six_balls_four_boxes_l3090_309084

/-- Represents a distribution of balls into boxes -/
def Distribution := List Nat

/-- Checks if a distribution is valid for the given number of balls and boxes -/
def is_valid_distribution (d : Distribution) (num_balls num_boxes : Nat) : Prop :=
  d.length ≤ num_boxes ∧ d.sum = num_balls ∧ d.all (· ≥ 0)

/-- Counts the number of distinct ways to distribute indistinguishable balls into indistinguishable boxes -/
def count_distributions (num_balls num_boxes : Nat) : Nat :=
  sorry

theorem six_balls_four_boxes :
  count_distributions 6 4 = 9 := by sorry

end NUMINAMATH_CALUDE_six_balls_four_boxes_l3090_309084


namespace NUMINAMATH_CALUDE_volume_maximized_at_height_1_2_l3090_309030

/-- Represents the dimensions of a rectangular container frame -/
structure ContainerFrame where
  shortSide : ℝ
  longSide : ℝ
  height : ℝ

/-- Calculates the volume of a container given its dimensions -/
def volume (frame : ContainerFrame) : ℝ :=
  frame.shortSide * frame.longSide * frame.height

/-- Calculates the perimeter of a container given its dimensions -/
def perimeter (frame : ContainerFrame) : ℝ :=
  2 * (frame.shortSide + frame.longSide + frame.height)

/-- Theorem: The volume of the container is maximized when the height is 1.2 m -/
theorem volume_maximized_at_height_1_2 :
  ∃ (frame : ContainerFrame),
    frame.longSide = frame.shortSide + 0.5 ∧
    perimeter frame = 14.8 ∧
    ∀ (other : ContainerFrame),
      other.longSide = other.shortSide + 0.5 →
      perimeter other = 14.8 →
      volume other ≤ volume frame ∧
      frame.height = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_volume_maximized_at_height_1_2_l3090_309030


namespace NUMINAMATH_CALUDE_extremum_at_zero_l3090_309092

/-- Given a function f(x) = e^x - ax with an extremum at x = 0, prove that a = 1 -/
theorem extremum_at_zero (a : ℝ) : 
  (∃ f : ℝ → ℝ, (∀ x, f x = Real.exp x - a * x) ∧ 
   (∃ ε > 0, ∀ x ∈ Set.Ioo (-ε) ε, f x ≤ f 0 ∨ f x ≥ f 0)) → 
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_extremum_at_zero_l3090_309092


namespace NUMINAMATH_CALUDE_decimal_insertion_sum_l3090_309041

-- Define a function to represent the possible ways to insert a decimal point in 2016
def insert_decimal (n : ℕ) : List ℝ :=
  [2.016, 20.16, 201.6]

-- Define the problem statement
theorem decimal_insertion_sum :
  ∃ (a b c d e f : ℝ),
    (a ∈ insert_decimal 2016) ∧
    (b ∈ insert_decimal 2016) ∧
    (c ∈ insert_decimal 2016) ∧
    (d ∈ insert_decimal 2016) ∧
    (e ∈ insert_decimal 2016) ∧
    (f ∈ insert_decimal 2016) ∧
    (a + b + c + d + e + f = 46368 / 100) :=
by
  sorry

end NUMINAMATH_CALUDE_decimal_insertion_sum_l3090_309041


namespace NUMINAMATH_CALUDE_train_crossing_time_l3090_309024

/-- Proves that a train with the given length and speed takes the calculated time to cross an electric pole -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) : 
  train_length = 120 ∧ train_speed_kmh = 72 → 
  (train_length / (train_speed_kmh * 1000 / 3600)) = 6 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l3090_309024


namespace NUMINAMATH_CALUDE_x_cos_x_necessary_not_sufficient_l3090_309010

theorem x_cos_x_necessary_not_sufficient (x : ℝ) (h : 0 < x ∧ x < Real.pi / 2) :
  (∀ y : ℝ, 0 < y ∧ y < Real.pi / 2 → (y < 1 → y * Real.cos y < 1)) ∧
  (∃ z : ℝ, 0 < z ∧ z < Real.pi / 2 ∧ z * Real.cos z < 1 ∧ z ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_x_cos_x_necessary_not_sufficient_l3090_309010


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3090_309087

theorem arithmetic_sequence_sum (a b c : ℝ) : 
  (∃ d : ℝ, a = 3 + d ∧ b = a + d ∧ c = b + d ∧ 15 = c + d) → 
  a + b + c = 27 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3090_309087


namespace NUMINAMATH_CALUDE_lines_intersection_l3090_309055

-- Define the two lines
def line1 (s : ℚ) : ℚ × ℚ := (1 + 2*s, 4 - 3*s)
def line2 (v : ℚ) : ℚ × ℚ := (-2 + 3*v, 6 - v)

-- Define the intersection point
def intersection_point : ℚ × ℚ := (17/11, 35/11)

-- Theorem statement
theorem lines_intersection :
  ∃ (s v : ℚ), line1 s = line2 v ∧ line1 s = intersection_point :=
by sorry

end NUMINAMATH_CALUDE_lines_intersection_l3090_309055


namespace NUMINAMATH_CALUDE_extremum_point_implies_a_zero_b_range_for_real_roots_l3090_309054

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x + 1) + x^3 - x^2 - a * x

theorem extremum_point_implies_a_zero :
  (∀ a : ℝ, (∃ ε > 0, ∀ x ∈ Set.Ioo ((2/3) - ε) ((2/3) + ε), f a x ≤ f a (2/3) ∨ f a x ≥ f a (2/3))) →
  (∃ a : ℝ, ∀ x : ℝ, f a x = f a (2/3)) :=
sorry

theorem b_range_for_real_roots :
  ∀ b : ℝ, (∃ x : ℝ, f (-1) (1 - x) - (1 - x)^3 = b) → b ∈ Set.Iic 0 :=
sorry

end NUMINAMATH_CALUDE_extremum_point_implies_a_zero_b_range_for_real_roots_l3090_309054


namespace NUMINAMATH_CALUDE_fuel_a_amount_proof_l3090_309063

-- Define the tank capacity
def tank_capacity : ℝ := 200

-- Define the ethanol content percentages
def ethanol_content_a : ℝ := 0.12
def ethanol_content_b : ℝ := 0.16

-- Define the total ethanol in the full tank
def total_ethanol : ℝ := 28

-- Define the amount of fuel A added (to be proved)
def fuel_a_added : ℝ := 100

-- Theorem statement
theorem fuel_a_amount_proof :
  ∃ (x : ℝ), 
    x ≥ 0 ∧ 
    x ≤ tank_capacity ∧
    ethanol_content_a * x + ethanol_content_b * (tank_capacity - x) = total_ethanol ∧
    x = fuel_a_added :=
by
  sorry


end NUMINAMATH_CALUDE_fuel_a_amount_proof_l3090_309063


namespace NUMINAMATH_CALUDE_shekar_average_marks_l3090_309078

def shekar_scores : List ℝ := [92, 78, 85, 67, 89, 74, 81, 95, 70, 88]

theorem shekar_average_marks :
  (shekar_scores.sum / shekar_scores.length : ℝ) = 81.9 := by
  sorry

end NUMINAMATH_CALUDE_shekar_average_marks_l3090_309078


namespace NUMINAMATH_CALUDE_ellipse_focus_x_axis_l3090_309025

/-- 
Given an ellipse with equation x²/(1-k) + y²/(2+k) = 1,
if its focus lies on the x-axis, then k ∈ (-2, -1/2)
-/
theorem ellipse_focus_x_axis (k : ℝ) : 
  (∃ (x y : ℝ), x^2 / (1 - k) + y^2 / (2 + k) = 1) →
  (∃ (c : ℝ), c > 0 ∧ c^2 = (1 - k) - (2 + k)) →
  k ∈ Set.Ioo (-2 : ℝ) (-1/2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_focus_x_axis_l3090_309025


namespace NUMINAMATH_CALUDE_correct_operation_l3090_309009

theorem correct_operation (x : ℝ) : x - 2*x = -x := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l3090_309009


namespace NUMINAMATH_CALUDE_sqrt_inequality_l3090_309051

theorem sqrt_inequality (a : ℝ) (h : a > 1) : Real.sqrt (a + 1) + Real.sqrt (a - 1) < 2 * Real.sqrt a := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l3090_309051


namespace NUMINAMATH_CALUDE_otimes_two_three_l3090_309053

-- Define the new operation ⊗
def otimes (a b : ℝ) : ℝ := 4 * a + 5 * b

-- Theorem to prove
theorem otimes_two_three : otimes 2 3 = 23 := by
  sorry

end NUMINAMATH_CALUDE_otimes_two_three_l3090_309053


namespace NUMINAMATH_CALUDE_largest_perfect_square_factor_882_l3090_309071

def largest_perfect_square_factor (n : ℕ) : ℕ := sorry

theorem largest_perfect_square_factor_882 :
  largest_perfect_square_factor 882 = 441 := by sorry

end NUMINAMATH_CALUDE_largest_perfect_square_factor_882_l3090_309071


namespace NUMINAMATH_CALUDE_units_digit_of_2_pow_2015_l3090_309027

theorem units_digit_of_2_pow_2015 (h : ∀ n : ℕ, n > 0 → (2^n : ℕ) % 10 = (2^(n % 4) : ℕ) % 10) :
  (2^2015 : ℕ) % 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_2_pow_2015_l3090_309027


namespace NUMINAMATH_CALUDE_min_sum_squares_l3090_309079

def S : Finset Int := {-8, -6, -4, -1, 1, 3, 5, 7, 9}

theorem min_sum_squares (p q r s t u v w x : Int) 
  (hp : p ∈ S) (hq : q ∈ S) (hr : r ∈ S) (hs : s ∈ S) 
  (ht : t ∈ S) (hu : u ∈ S) (hv : v ∈ S) (hw : w ∈ S) (hx : x ∈ S)
  (hdistinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ p ≠ v ∧ p ≠ w ∧ p ≠ x ∧
               q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ q ≠ v ∧ q ≠ w ∧ q ≠ x ∧
               r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ r ≠ v ∧ r ≠ w ∧ r ≠ x ∧
               s ≠ t ∧ s ≠ u ∧ s ≠ v ∧ s ≠ w ∧ s ≠ x ∧
               t ≠ u ∧ t ≠ v ∧ t ≠ w ∧ t ≠ x ∧
               u ≠ v ∧ u ≠ w ∧ u ≠ x ∧
               v ≠ w ∧ v ≠ x ∧
               w ≠ x) :
  (p + q + r + s)^2 + (t + u + v + w + x)^2 ≥ 18 := by
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l3090_309079


namespace NUMINAMATH_CALUDE_cost_price_is_40_l3090_309037

/-- Calculates the cost price per metre of cloth given the total length, 
    total selling price, and loss per metre. -/
def cost_price_per_metre (total_length : ℕ) (total_selling_price : ℕ) (loss_per_metre : ℕ) : ℕ :=
  (total_selling_price + total_length * loss_per_metre) / total_length

/-- Proves that the cost price per metre is 40 given the specified conditions. -/
theorem cost_price_is_40 :
  cost_price_per_metre 500 15000 10 = 40 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_is_40_l3090_309037


namespace NUMINAMATH_CALUDE_pool_capacity_l3090_309035

theorem pool_capacity (additional_water : ℝ) (final_percentage : ℝ) (increase_percentage : ℝ) :
  additional_water = 300 →
  final_percentage = 0.7 →
  increase_percentage = 0.3 →
  ∃ (total_capacity : ℝ),
    total_capacity = 1000 ∧
    additional_water = (final_percentage - increase_percentage) * total_capacity :=
by sorry

end NUMINAMATH_CALUDE_pool_capacity_l3090_309035


namespace NUMINAMATH_CALUDE_parade_tricycles_l3090_309074

theorem parade_tricycles :
  ∀ (w b t : ℕ),
    w + b + t = 10 →
    2 * b + 3 * t = 25 →
    t = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_parade_tricycles_l3090_309074


namespace NUMINAMATH_CALUDE_problem_solving_probability_l3090_309017

theorem problem_solving_probability 
  (arthur_prob : ℚ) 
  (bella_prob : ℚ) 
  (xavier_prob : ℚ) 
  (yvonne_prob : ℚ) 
  (zelda_prob : ℚ) 
  (h_arthur : arthur_prob = 1/4)
  (h_bella : bella_prob = 3/10)
  (h_xavier : xavier_prob = 1/6)
  (h_yvonne : yvonne_prob = 1/2)
  (h_zelda : zelda_prob = 5/8)
  (h_independent : True)  -- Assumption of independence
  : arthur_prob * bella_prob * xavier_prob * yvonne_prob * (1 - zelda_prob) = 9/3840 :=
by
  sorry

end NUMINAMATH_CALUDE_problem_solving_probability_l3090_309017


namespace NUMINAMATH_CALUDE_rhombus_60_min_rotation_l3090_309019

/-- A rhombus with a 60° angle -/
structure Rhombus60 where
  /-- The rhombus has a 60° angle -/
  angle_60 : ∃ θ, θ = 60

/-- Minimum rotation for a Rhombus60 to coincide with its original position -/
def min_rotation (r : Rhombus60) : ℝ :=
  180

/-- Theorem: The minimum rotation for a Rhombus60 to coincide with its original position is 180° -/
theorem rhombus_60_min_rotation (r : Rhombus60) :
  min_rotation r = 180 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_60_min_rotation_l3090_309019


namespace NUMINAMATH_CALUDE_bisection_termination_condition_l3090_309039

/-- The bisection method termination condition -/
def is_termination_condition (x₁ x₂ e : ℝ) : Prop :=
  |x₁ - x₂| < e

/-- Theorem stating that the correct termination condition for the bisection method is |x₁ - x₂| < e -/
theorem bisection_termination_condition (x₁ x₂ e : ℝ) (h : e > 0) :
  is_termination_condition x₁ x₂ e ↔ |x₁ - x₂| < e := by sorry

end NUMINAMATH_CALUDE_bisection_termination_condition_l3090_309039


namespace NUMINAMATH_CALUDE_ten_point_circle_triangles_l3090_309094

/-- The number of ways to choose 3 points from n points to form a triangle -/
def triangles_from_points (n : ℕ) : ℕ := n.choose 3

/-- Given 10 points on a circle, the number of inscribed triangles is 360 -/
theorem ten_point_circle_triangles :
  triangles_from_points 10 = 360 := by
  sorry

end NUMINAMATH_CALUDE_ten_point_circle_triangles_l3090_309094


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l3090_309083

/-- An isosceles triangle with perimeter 13 and one side length 3 -/
structure IsoscelesTriangle where
  -- The length of two equal sides
  side : ℝ
  -- The length of the base
  base : ℝ
  -- The triangle is isosceles
  isIsosceles : side ≠ base
  -- The perimeter is 13
  perimeterIs13 : side + side + base = 13
  -- One side length is 3
  oneSideIs3 : side = 3 ∨ base = 3

/-- The base of an isosceles triangle with perimeter 13 and one side 3 must be 3 -/
theorem isosceles_triangle_base_length (t : IsoscelesTriangle) : t.base = 3 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l3090_309083


namespace NUMINAMATH_CALUDE_complex_fraction_ratio_l3090_309085

theorem complex_fraction_ratio (x : ℝ) : x = 200 → x / 10 = 20 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_ratio_l3090_309085


namespace NUMINAMATH_CALUDE_oak_trees_cut_down_problem_l3090_309073

/-- The number of oak trees cut down in a park --/
def oak_trees_cut_down (initial : ℕ) (final : ℕ) : ℕ :=
  initial - final

/-- Theorem: Given 9 initial oak trees and 7 remaining after cutting, 2 oak trees were cut down --/
theorem oak_trees_cut_down_problem : oak_trees_cut_down 9 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_oak_trees_cut_down_problem_l3090_309073


namespace NUMINAMATH_CALUDE_equation_has_four_solutions_l3090_309031

/-- The number of integer solutions to the equation 6y² + 3xy + x + 2y - 72 = 0 -/
def num_solutions : ℕ := 4

/-- The equation 6y² + 3xy + x + 2y - 72 = 0 -/
def equation (x y : ℤ) : Prop :=
  6 * y^2 + 3 * x * y + x + 2 * y - 72 = 0

theorem equation_has_four_solutions :
  ∃! (s : Finset (ℤ × ℤ)), s.card = num_solutions ∧ 
  ∀ (p : ℤ × ℤ), p ∈ s ↔ equation p.1 p.2 := by
  sorry

end NUMINAMATH_CALUDE_equation_has_four_solutions_l3090_309031


namespace NUMINAMATH_CALUDE_equation_one_solution_equation_two_no_solution_l3090_309072

-- Problem 1
theorem equation_one_solution (x : ℝ) : 
  (9 / x = 8 / (x - 1)) ↔ x = 9 :=
sorry

-- Problem 2
theorem equation_two_no_solution : 
  ¬∃ (x : ℝ), ((x - 8) / (x - 7) - 8 = 1 / (7 - x)) :=
sorry

end NUMINAMATH_CALUDE_equation_one_solution_equation_two_no_solution_l3090_309072


namespace NUMINAMATH_CALUDE_not_right_triangle_l3090_309000

theorem not_right_triangle (A B C : ℝ) (h1 : A + B + C = 180) (h2 : A = 2 * B) (h3 : A = 3 * C) :
  A ≠ 90 ∧ B ≠ 90 ∧ C ≠ 90 := by
  sorry

end NUMINAMATH_CALUDE_not_right_triangle_l3090_309000


namespace NUMINAMATH_CALUDE_probability_of_white_ball_l3090_309065

theorem probability_of_white_ball (P_red P_black P_yellow P_white : ℚ) : 
  P_red = 1/3 →
  P_black + P_yellow = 5/12 →
  P_yellow + P_white = 5/12 →
  P_red + P_black + P_yellow + P_white = 1 →
  P_white = 1/4 := by
sorry

end NUMINAMATH_CALUDE_probability_of_white_ball_l3090_309065


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l3090_309021

theorem x_squared_minus_y_squared (x y : ℝ) 
  (h1 : x + y = 15) 
  (h2 : 3 * x + y = 20) : 
  x^2 - y^2 = -150 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l3090_309021


namespace NUMINAMATH_CALUDE_negation_of_all_exponential_monotonic_l3090_309020

-- Define the set of exponential functions
def ExponentialFunction : Type := ℝ → ℝ

-- Define the property of being monotonic
def Monotonic (f : ℝ → ℝ) : Prop := ∀ x y, x ≤ y → f x ≤ f y

-- State the theorem
theorem negation_of_all_exponential_monotonic :
  (¬ ∀ f : ExponentialFunction, Monotonic f) ↔ (∃ f : ExponentialFunction, ¬ Monotonic f) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_all_exponential_monotonic_l3090_309020


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l3090_309023

theorem root_sum_reciprocal (x₁ x₂ : ℝ) : 
  x₁^2 - 3*x₁ + 2 = 0 → 
  x₂^2 - 3*x₂ + 2 = 0 → 
  x₁ ≠ x₂ →
  (1/x₁) + (1/x₂) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l3090_309023


namespace NUMINAMATH_CALUDE_base12_addition_l3090_309070

/-- Represents a digit in base 12 --/
inductive Digit12 : Type
| D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | A | B | C

/-- Converts a Digit12 to its decimal (base 10) value --/
def toDecimal (d : Digit12) : Nat :=
  match d with
  | Digit12.D0 => 0
  | Digit12.D1 => 1
  | Digit12.D2 => 2
  | Digit12.D3 => 3
  | Digit12.D4 => 4
  | Digit12.D5 => 5
  | Digit12.D6 => 6
  | Digit12.D7 => 7
  | Digit12.D8 => 8
  | Digit12.D9 => 9
  | Digit12.A => 10
  | Digit12.B => 11
  | Digit12.C => 12

/-- Represents a number in base 12 --/
def Base12 := List Digit12

/-- Converts a Base12 number to its decimal (base 10) value --/
def base12ToDecimal (n : Base12) : Nat :=
  n.foldr (fun d acc => toDecimal d + 12 * acc) 0

/-- The main theorem to prove --/
theorem base12_addition :
  base12ToDecimal [Digit12.D3, Digit12.C, Digit12.D5] +
  base12ToDecimal [Digit12.D2, Digit12.A, Digit12.B] =
  base12ToDecimal [Digit12.D6, Digit12.D3, Digit12.D4] := by
  sorry


end NUMINAMATH_CALUDE_base12_addition_l3090_309070


namespace NUMINAMATH_CALUDE_intercept_sum_mod_17_l3090_309061

theorem intercept_sum_mod_17 :
  ∃! (x₀ y₀ : ℕ), x₀ < 17 ∧ y₀ < 17 ∧
  (5 * x₀ ≡ 2 [MOD 17]) ∧
  (3 * y₀ + 2 ≡ 0 [MOD 17]) ∧
  x₀ + y₀ = 19 :=
by sorry

end NUMINAMATH_CALUDE_intercept_sum_mod_17_l3090_309061


namespace NUMINAMATH_CALUDE_shares_ratio_l3090_309062

/-- Represents the shares of money for three individuals -/
structure Shares where
  a : ℚ
  b : ℚ
  c : ℚ

/-- The problem setup -/
def problem_setup (s : Shares) : Prop :=
  s.a + s.b + s.c = 700 ∧  -- Total amount
  s.a = 280 ∧              -- A's share
  ∃ x, s.a = x * (s.b + s.c) ∧  -- A's share as a fraction of B and C
  s.b = (6/9) * (s.a + s.c)     -- B's share as 6/9 of A and C

/-- The theorem to prove -/
theorem shares_ratio (s : Shares) (h : problem_setup s) : 
  s.a / (s.b + s.c) = 2/3 := by
  sorry


end NUMINAMATH_CALUDE_shares_ratio_l3090_309062


namespace NUMINAMATH_CALUDE_sixth_term_is_twelve_l3090_309069

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  -- First term
  a : ℝ
  -- Common difference
  d : ℝ
  -- Sum of first four terms is 20
  sum_first_four : a + (a + d) + (a + 2*d) + (a + 3*d) = 20
  -- Fifth term is 10
  fifth_term : a + 4*d = 10

/-- The sixth term of the arithmetic sequence is 12 -/
theorem sixth_term_is_twelve (seq : ArithmeticSequence) : seq.a + 5*seq.d = 12 := by
  sorry

end NUMINAMATH_CALUDE_sixth_term_is_twelve_l3090_309069


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l3090_309056

theorem contrapositive_equivalence (f : ℝ → ℝ) (a : ℝ) :
  (a ≥ (1/2) → ∀ x ≥ 0, f x ≥ 0) ↔
  (∃ x ≥ 0, f x < 0 → a < (1/2)) :=
sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l3090_309056


namespace NUMINAMATH_CALUDE_sum_reciprocal_inequality_l3090_309046

theorem sum_reciprocal_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a + b + c + d) * (1/a + 1/b + 1/c + 1/d) ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocal_inequality_l3090_309046


namespace NUMINAMATH_CALUDE_simplest_fraction_sum_l3090_309068

theorem simplest_fraction_sum (a b : ℕ+) : 
  (a : ℚ) / b = 0.4375 ∧ 
  ∀ (c d : ℕ+), (c : ℚ) / d = 0.4375 → a ≤ c ∧ b ≤ d →
  a + b = 23 := by
sorry

end NUMINAMATH_CALUDE_simplest_fraction_sum_l3090_309068


namespace NUMINAMATH_CALUDE_only_prime_square_difference_pair_l3090_309005

theorem only_prime_square_difference_pair : 
  ∀ p q : ℕ, 
    Prime p → 
    Prime q → 
    p > q → 
    Prime (p^2 - q^2) → 
    p = 3 ∧ q = 2 :=
by sorry

end NUMINAMATH_CALUDE_only_prime_square_difference_pair_l3090_309005


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3090_309045

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℤ := {x | x^2 + x = 0}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3090_309045


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3090_309086

-- Define an arithmetic sequence
def isArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h1 : isArithmeticSequence a) 
  (h2 : a 7 + a 9 = 8) : 
  a 8 = 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3090_309086


namespace NUMINAMATH_CALUDE_georgia_buttons_l3090_309012

/-- Georgia's button problem -/
theorem georgia_buttons (yellow black green given_away remaining : ℕ) :
  yellow + black + green = given_away + remaining →
  remaining = 5 →
  yellow = 4 →
  black = 2 →
  green = 3 →
  given_away = 4 :=
by sorry

end NUMINAMATH_CALUDE_georgia_buttons_l3090_309012


namespace NUMINAMATH_CALUDE_certain_number_divisibility_l3090_309026

theorem certain_number_divisibility (z x : ℕ) (h1 : z > 0) (h2 : 4 ∣ z) : 
  (z + x + 4 + z + 3) % 2 = 1 ↔ Even x := by sorry

end NUMINAMATH_CALUDE_certain_number_divisibility_l3090_309026


namespace NUMINAMATH_CALUDE_pumpkin_weight_sum_total_pumpkin_weight_l3090_309066

theorem pumpkin_weight_sum : ℝ → ℝ → ℝ
  | weight1, weight2 => weight1 + weight2

theorem total_pumpkin_weight :
  let weight1 : ℝ := 4
  let weight2 : ℝ := 8.7
  pumpkin_weight_sum weight1 weight2 = 12.7 := by
  sorry

end NUMINAMATH_CALUDE_pumpkin_weight_sum_total_pumpkin_weight_l3090_309066


namespace NUMINAMATH_CALUDE_min_distance_curve_to_line_l3090_309034

/-- The minimum distance from a point on y = e^(2x) to the line 2x - y - 4 = 0 -/
theorem min_distance_curve_to_line : 
  let f : ℝ → ℝ := fun x ↦ Real.exp (2 * x)
  let l : ℝ → ℝ → ℝ := fun x y ↦ 2 * x - y - 4
  let d : ℝ → ℝ := fun x ↦ |l x (f x)| / Real.sqrt 5
  ∃ (x_min : ℝ), ∀ (x : ℝ), d x_min ≤ d x ∧ d x_min = 4 * Real.sqrt 5 / 5 := by
sorry

end NUMINAMATH_CALUDE_min_distance_curve_to_line_l3090_309034


namespace NUMINAMATH_CALUDE_tree_planting_problem_l3090_309008

theorem tree_planting_problem (total_trees : ℕ) 
  (h1 : 205000 ≤ total_trees ∧ total_trees ≤ 205300) 
  (h2 : ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ 7 * (x - 1) = total_trees ∧ 13 * (y - 1) = total_trees) : 
  ∃ (students : ℕ), students = 62 ∧ 
    ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x + y = students ∧ 
      7 * (x - 1) = total_trees ∧ 13 * (y - 1) = total_trees :=
by sorry

end NUMINAMATH_CALUDE_tree_planting_problem_l3090_309008


namespace NUMINAMATH_CALUDE_percentage_of_cat_owners_l3090_309006

theorem percentage_of_cat_owners (total_students : ℕ) (cat_owners : ℕ) 
  (h1 : total_students = 400) (h2 : cat_owners = 80) : 
  (cat_owners : ℝ) / total_students * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_cat_owners_l3090_309006


namespace NUMINAMATH_CALUDE_sum_of_roots_l3090_309048

-- Define the quadratic equation
def quadratic_equation (x m n : ℝ) : Prop := 2 * x^2 + m * x + n = 0

-- State the theorem
theorem sum_of_roots (m n : ℝ) 
  (hm : quadratic_equation m m n) 
  (hn : quadratic_equation n m n) : 
  m + n = -m / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3090_309048


namespace NUMINAMATH_CALUDE_triangle_area_rational_l3090_309080

theorem triangle_area_rational (x₁ x₂ x₃ y₁ y₂ y₃ : ℤ) :
  ∃ (q : ℚ), q = (1/2) * |((x₁ + (1/2 : ℚ)) * (y₂ - y₃) + 
                           (x₂ + (1/2 : ℚ)) * (y₃ - y₁) + 
                           (x₃ + (1/2 : ℚ)) * (y₁ - y₂))| := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_rational_l3090_309080


namespace NUMINAMATH_CALUDE_probability_sum_15_l3090_309029

def is_valid_roll (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 6 ∧ 1 ≤ b ∧ b ≤ 6 ∧ 1 ≤ c ∧ c ≤ 6

def sum_is_15 (a b c : ℕ) : Prop :=
  a + b + c = 15

def count_valid_rolls : ℕ := 216

def count_sum_15_rolls : ℕ := 10

theorem probability_sum_15 :
  (count_sum_15_rolls : ℚ) / count_valid_rolls = 5 / 108 :=
sorry

end NUMINAMATH_CALUDE_probability_sum_15_l3090_309029


namespace NUMINAMATH_CALUDE_bridge_length_l3090_309059

/-- The length of a bridge given train parameters -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : 
  train_length = 135 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  ∃ (bridge_length : ℝ), bridge_length = 240 :=
by
  sorry

#check bridge_length

end NUMINAMATH_CALUDE_bridge_length_l3090_309059


namespace NUMINAMATH_CALUDE_negation_equivalence_l3090_309042

variable (a : ℝ)

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 - a*x + 1 < 0) ↔ (∀ x : ℝ, x^2 - a*x + 1 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3090_309042


namespace NUMINAMATH_CALUDE_exponential_inequality_l3090_309003

theorem exponential_inequality : 
  (2/5 : ℝ)^(3/5) < (2/5 : ℝ)^(2/5) ∧ (2/5 : ℝ)^(2/5) < (3/5 : ℝ)^(3/5) := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l3090_309003


namespace NUMINAMATH_CALUDE_min_value_expression_l3090_309040

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 1) :
  x^2 + 4*x*y + 9*y^2 + 8*y*z + 3*z^2 ≥ 18 ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 1 ∧
    x₀^2 + 4*x₀*y₀ + 9*y₀^2 + 8*y₀*z₀ + 3*z₀^2 = 18 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3090_309040


namespace NUMINAMATH_CALUDE_base4_to_base10_3201_l3090_309002

/-- Converts a base 4 number to base 10 -/
def base4_to_base10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

/-- The base 4 representation of the number -/
def base4_num : List Nat := [1, 0, 2, 3]

theorem base4_to_base10_3201 :
  base4_to_base10 base4_num = 225 := by
  sorry

end NUMINAMATH_CALUDE_base4_to_base10_3201_l3090_309002


namespace NUMINAMATH_CALUDE_journey_distance_l3090_309033

/-- Given a journey that takes 3 hours and can be completed in half the time
    at a speed of 293.3333333333333 kmph, prove that the distance traveled is 440 km. -/
theorem journey_distance (original_time : ℝ) (new_speed : ℝ) (distance : ℝ) : 
  original_time = 3 →
  new_speed = 293.3333333333333 →
  distance = new_speed * (original_time / 2) →
  distance = 440 := by
  sorry

end NUMINAMATH_CALUDE_journey_distance_l3090_309033


namespace NUMINAMATH_CALUDE_dog_treat_expenditure_l3090_309007

/-- Calculate John's total expenditure on dog treats for a month --/
theorem dog_treat_expenditure :
  let treats_first_half : ℕ := 3 * 15
  let treats_second_half : ℕ := 4 * 15
  let total_treats : ℕ := treats_first_half + treats_second_half
  let original_price : ℚ := 0.1
  let discount_threshold : ℕ := 50
  let discount_rate : ℚ := 0.1
  let discounted_price : ℚ := original_price * (1 - discount_rate)
  total_treats > discount_threshold →
  (total_treats : ℚ) * discounted_price = 9.45 :=
by sorry

end NUMINAMATH_CALUDE_dog_treat_expenditure_l3090_309007


namespace NUMINAMATH_CALUDE_opposite_numbers_theorem_l3090_309067

theorem opposite_numbers_theorem (a : ℚ) : (4 * a + 9) + (3 * a + 5) = 0 → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_numbers_theorem_l3090_309067


namespace NUMINAMATH_CALUDE_number_of_ways_to_turn_off_lights_l3090_309038

/-- The number of streetlights --/
def total_lights : ℕ := 12

/-- The number of lights that can be turned off --/
def lights_off : ℕ := 3

/-- The number of positions where lights can be turned off --/
def eligible_positions : ℕ := total_lights - 2 - lights_off + 1

/-- Theorem stating the number of ways to turn off lights --/
theorem number_of_ways_to_turn_off_lights : 
  Nat.choose eligible_positions lights_off = 56 := by sorry

end NUMINAMATH_CALUDE_number_of_ways_to_turn_off_lights_l3090_309038


namespace NUMINAMATH_CALUDE_sticker_distribution_l3090_309064

/-- The number of stickers Mary bought initially -/
def total_stickers : ℕ := 1500

/-- Susan's share of stickers -/
def susan_share : ℕ := 300

/-- Andrew's initial share of stickers -/
def andrew_initial_share : ℕ := 300

/-- Sam's initial share of stickers -/
def sam_initial_share : ℕ := 900

/-- The amount of stickers Sam gave to Andrew -/
def sam_to_andrew : ℕ := 600

/-- Andrew's final share of stickers -/
def andrew_final_share : ℕ := 900

theorem sticker_distribution :
  -- The total is the sum of all initial shares
  total_stickers = susan_share + andrew_initial_share + sam_initial_share ∧
  -- The ratio of shares is 1:1:3
  susan_share = andrew_initial_share ∧
  sam_initial_share = 3 * andrew_initial_share ∧
  -- Sam gave Andrew two-thirds of his share
  sam_to_andrew = 2 * sam_initial_share / 3 ∧
  -- Andrew's final share is his initial plus what Sam gave him
  andrew_final_share = andrew_initial_share + sam_to_andrew :=
by sorry

end NUMINAMATH_CALUDE_sticker_distribution_l3090_309064


namespace NUMINAMATH_CALUDE_third_person_weight_is_131_l3090_309096

/-- Calculates the true weight of the third person (C) entering an elevator given the following conditions:
    - There are initially 6 people in the elevator with an average weight of 156 lbs.
    - Three people (A, B, C) enter the elevator one by one.
    - The weights of their clothing and backpacks are 18 lbs, 20 lbs, and 22 lbs respectively.
    - After each person enters, the average weight changes to 159 lbs, 162 lbs, and 161 lbs respectively. -/
def calculate_third_person_weight (initial_people : Nat) (initial_avg : Nat)
  (a_extra_weight : Nat) (b_extra_weight : Nat) (c_extra_weight : Nat)
  (avg_after_a : Nat) (avg_after_b : Nat) (avg_after_c : Nat) : Nat :=
  let total_initial := initial_people * initial_avg
  let total_after_a := (initial_people + 1) * avg_after_a
  let total_after_b := (initial_people + 2) * avg_after_b
  let total_after_c := (initial_people + 3) * avg_after_c
  total_after_c - total_after_b - c_extra_weight

/-- Theorem stating that given the conditions in the problem, 
    the true weight of the third person (C) is 131 lbs. -/
theorem third_person_weight_is_131 :
  calculate_third_person_weight 6 156 18 20 22 159 162 161 = 131 := by
  sorry

end NUMINAMATH_CALUDE_third_person_weight_is_131_l3090_309096


namespace NUMINAMATH_CALUDE_solution_implies_a_value_l3090_309014

theorem solution_implies_a_value (x a : ℝ) : 
  x = 5 → 2 * x + 3 * a = 4 → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_solution_implies_a_value_l3090_309014


namespace NUMINAMATH_CALUDE_inverse_variation_cube_fourth_l3090_309022

/-- Given that a³ varies inversely with b⁴, and a = 2 when b = 4,
    prove that a = 1/∛2 when b = 8 -/
theorem inverse_variation_cube_fourth (a b : ℝ) (k : ℝ) :
  (∀ a b, a^3 * b^4 = k) →  -- a³ varies inversely with b⁴
  (2^3 * 4^4 = k) →         -- a = 2 when b = 4
  (a^3 * 8^4 = k) →         -- condition for b = 8
  a = 1 / (2^(1/3)) :=      -- a = 1/∛2 when b = 8
by sorry

end NUMINAMATH_CALUDE_inverse_variation_cube_fourth_l3090_309022


namespace NUMINAMATH_CALUDE_problem_solution_l3090_309058

theorem problem_solution : 
  |Real.sqrt 3 - 2| + (27 : ℝ) ^ (1/3 : ℝ) - Real.sqrt 16 + (-1) ^ 2023 = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3090_309058


namespace NUMINAMATH_CALUDE_angle_B_value_max_perimeter_max_perimeter_achievable_l3090_309047

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

/-- The given condition (2a-c)cos B = b cos C -/
def triangle_condition (t : Triangle) : Prop :=
  (2 * t.a - t.c) * Real.cos t.B = t.b * Real.cos t.C

/-- Theorem stating that B = π/3 -/
theorem angle_B_value (t : Triangle) (h : triangle_condition t) : t.B = π / 3 :=
sorry

/-- Theorem stating that when b = 2, the maximum perimeter is 6 -/
theorem max_perimeter (t : Triangle) (h : triangle_condition t) (hb : t.b = 2) :
  t.a + t.b + t.c ≤ 6 :=
sorry

/-- Theorem stating that the maximum perimeter of 6 is achievable -/
theorem max_perimeter_achievable : ∃ t : Triangle, triangle_condition t ∧ t.b = 2 ∧ t.a + t.b + t.c = 6 :=
sorry

end NUMINAMATH_CALUDE_angle_B_value_max_perimeter_max_perimeter_achievable_l3090_309047
