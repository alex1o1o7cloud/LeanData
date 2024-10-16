import Mathlib

namespace NUMINAMATH_CALUDE_square_perimeter_l2087_208703

theorem square_perimeter (total_area common_area circle_area : ℝ) 
  (h1 : total_area = 329)
  (h2 : common_area = 101)
  (h3 : circle_area = 234) :
  let square_area := total_area + common_area - circle_area
  let side_length := Real.sqrt square_area
  4 * side_length = 56 := by sorry

end NUMINAMATH_CALUDE_square_perimeter_l2087_208703


namespace NUMINAMATH_CALUDE_expansion_coefficient_l2087_208716

theorem expansion_coefficient (a : ℝ) (h1 : a > 0) 
  (h2 : (1 + 1) * (a + 1)^6 = 1458) : 
  (1 + 6 * 4) = 61 := by sorry

end NUMINAMATH_CALUDE_expansion_coefficient_l2087_208716


namespace NUMINAMATH_CALUDE_endpoint_sum_is_twelve_l2087_208765

/-- Given a line segment with one endpoint (6, -2) and midpoint (3, 5),
    the sum of the coordinates of the other endpoint is 12. -/
theorem endpoint_sum_is_twelve (x y : ℝ) : 
  (6 + x) / 2 = 3 → (-2 + y) / 2 = 5 → x + y = 12 := by
  sorry

end NUMINAMATH_CALUDE_endpoint_sum_is_twelve_l2087_208765


namespace NUMINAMATH_CALUDE_negative_division_l2087_208786

theorem negative_division (a b : ℤ) (ha : a = -300) (hb : b = -50) :
  a / b = 6 := by
  sorry

end NUMINAMATH_CALUDE_negative_division_l2087_208786


namespace NUMINAMATH_CALUDE_call_duration_is_60_minutes_l2087_208734

/-- Represents the duration of a single customer call in minutes. -/
def call_duration (cost_per_minute : ℚ) (monthly_bill : ℚ) (customers_per_week : ℕ) (weeks_per_month : ℕ) : ℚ :=
  (monthly_bill / cost_per_minute) / (customers_per_week * weeks_per_month)

/-- Theorem stating that under the given conditions, each call lasts 60 minutes. -/
theorem call_duration_is_60_minutes :
  call_duration (5 / 100) 600 50 4 = 60 := by
  sorry

end NUMINAMATH_CALUDE_call_duration_is_60_minutes_l2087_208734


namespace NUMINAMATH_CALUDE_sum_F_equals_250_l2087_208727

-- Define the function F
def F (n : ℕ) : ℕ := sorry

-- Define the sum of F from 1 to 50
def sum_F : ℕ := (List.range 50).map (fun i => F (i + 1)) |>.sum

-- Theorem statement
theorem sum_F_equals_250 : sum_F = 250 := by sorry

end NUMINAMATH_CALUDE_sum_F_equals_250_l2087_208727


namespace NUMINAMATH_CALUDE_solution_range_l2087_208745

theorem solution_range (x y z : ℝ) 
  (sum_eq : x + y + z = 5) 
  (prod_eq : x*y + y*z + z*x = 3) : 
  -1 ≤ x ∧ x ≤ 13/3 ∧ 
  -1 ≤ y ∧ y ≤ 13/3 ∧ 
  -1 ≤ z ∧ z ≤ 13/3 := by
sorry

end NUMINAMATH_CALUDE_solution_range_l2087_208745


namespace NUMINAMATH_CALUDE_rad_polynomial_characterization_l2087_208708

/-- rad(n) is the product of all distinct prime factors of n -/
def rad (n : ℕ+) : ℕ+ := sorry

/-- A number is square-free if it's not divisible by any perfect square other than 1 -/
def IsSquareFree (n : ℕ+) : Prop := sorry

/-- Polynomial with rational coefficients -/
def RationalPolynomial := Polynomial ℚ

theorem rad_polynomial_characterization (P : RationalPolynomial) :
  (∃ (s : Set ℕ+), Set.Infinite s ∧ ∀ n ∈ s, (P.eval n : ℚ) = (rad n : ℚ)) ↔
  (∃ b : ℕ+, P = Polynomial.monomial 1 (1 / (b : ℚ))) ∨
  (∃ k : ℕ+, IsSquareFree k ∧ P = Polynomial.C (k : ℚ)) := by sorry

end NUMINAMATH_CALUDE_rad_polynomial_characterization_l2087_208708


namespace NUMINAMATH_CALUDE_sqrt_sum_diff_zero_sqrt_minus_squared_l2087_208764

-- Problem 1
theorem sqrt_sum_diff_zero : Real.sqrt 8 + Real.sqrt 2 - Real.sqrt 18 = 0 := by sorry

-- Problem 2
theorem sqrt_minus_squared : (Real.sqrt 3 - 2)^2 = 7 - 4 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_sqrt_sum_diff_zero_sqrt_minus_squared_l2087_208764


namespace NUMINAMATH_CALUDE_tangent_circle_circumference_is_36_l2087_208747

/-- Represents a geometric setup with two circular arcs and a tangent circle -/
structure GeometricSetup where
  -- The length of arc BC
  arc_length : ℝ
  -- Predicate that the arcs subtend 90° angles
  subtend_right_angle : Prop
  -- Predicate that the circle is tangent to both arcs and line segment AB
  circle_tangent : Prop

/-- The circumference of the tangent circle in the given geometric setup -/
def tangent_circle_circumference (setup : GeometricSetup) : ℝ :=
  sorry

/-- Theorem stating that the circumference of the tangent circle is 36 -/
theorem tangent_circle_circumference_is_36 (setup : GeometricSetup) 
  (h1 : setup.arc_length = 18)
  (h2 : setup.subtend_right_angle)
  (h3 : setup.circle_tangent) :
  tangent_circle_circumference setup = 36 :=
sorry

end NUMINAMATH_CALUDE_tangent_circle_circumference_is_36_l2087_208747


namespace NUMINAMATH_CALUDE_grid_square_triangle_count_l2087_208791

/-- Represents a square divided into a 4x4 grid with diagonals --/
structure GridSquare :=
  (size : ℕ)
  (has_diagonals : Bool)
  (has_small_square_diagonals : Bool)

/-- Counts the number of triangles in a GridSquare --/
def count_triangles (sq : GridSquare) : ℕ :=
  sorry

/-- The main theorem stating that a 4x4 GridSquare with all diagonals has 42 triangles --/
theorem grid_square_triangle_count :
  ∀ (sq : GridSquare), 
    sq.size = 4 ∧ 
    sq.has_diagonals = true ∧ 
    sq.has_small_square_diagonals = true → 
    count_triangles sq = 42 :=
  sorry

end NUMINAMATH_CALUDE_grid_square_triangle_count_l2087_208791


namespace NUMINAMATH_CALUDE_brick_length_is_20_l2087_208701

/-- The length of a brick in centimeters -/
def brick_length : ℝ := 20

/-- The width of a brick in centimeters -/
def brick_width : ℝ := 10

/-- The height of a brick in centimeters -/
def brick_height : ℝ := 7.5

/-- The length of the wall in meters -/
def wall_length : ℝ := 29

/-- The width of the wall in meters -/
def wall_width : ℝ := 2

/-- The height of the wall in meters -/
def wall_height : ℝ := 0.75

/-- The number of bricks in the wall -/
def number_of_bricks : ℕ := 29000

/-- Conversion factor from meters to centimeters -/
def m_to_cm : ℝ := 100

theorem brick_length_is_20 :
  brick_length = 20 :=
by
  have h1 : brick_length * brick_width * brick_height = 
    (wall_length * wall_width * wall_height * m_to_cm^3) / number_of_bricks :=
    sorry
  sorry

end NUMINAMATH_CALUDE_brick_length_is_20_l2087_208701


namespace NUMINAMATH_CALUDE_system_solution_is_correct_l2087_208707

/-- The solution set of the system of inequalities {2x + 3 ≤ x + 2, (x + 1) / 3 > x - 1} -/
def solution_set : Set ℝ := {x : ℝ | x ≤ -1}

/-- The first inequality of the system -/
def inequality1 (x : ℝ) : Prop := 2 * x + 3 ≤ x + 2

/-- The second inequality of the system -/
def inequality2 (x : ℝ) : Prop := (x + 1) / 3 > x - 1

theorem system_solution_is_correct :
  ∀ x : ℝ, x ∈ solution_set ↔ (inequality1 x ∧ inequality2 x) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_is_correct_l2087_208707


namespace NUMINAMATH_CALUDE_intersection_point_correct_l2087_208781

/-- Represents a 2D vector --/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Represents a parametric line in 2D --/
structure ParametricLine where
  origin : Vector2D
  direction : Vector2D

/-- The first line --/
def line1 : ParametricLine :=
  { origin := { x := 1, y := 2 },
    direction := { x := -2, y := 4 } }

/-- The second line --/
def line2 : ParametricLine :=
  { origin := { x := 3, y := 5 },
    direction := { x := 1, y := 3 } }

/-- Calculates a point on a parametric line given a parameter t --/
def pointOnLine (line : ParametricLine) (t : ℝ) : Vector2D :=
  { x := line.origin.x + t * line.direction.x,
    y := line.origin.y + t * line.direction.y }

/-- The intersection point of the two lines --/
def intersectionPoint : Vector2D :=
  { x := 1.2, y := 1.6 }

/-- Theorem stating that the calculated intersection point is correct --/
theorem intersection_point_correct :
  ∃ t u : ℝ, pointOnLine line1 t = intersectionPoint ∧ pointOnLine line2 u = intersectionPoint :=
sorry


end NUMINAMATH_CALUDE_intersection_point_correct_l2087_208781


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2087_208785

def z : ℂ := 2 + Complex.I

theorem imaginary_part_of_z : z.im = 1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2087_208785


namespace NUMINAMATH_CALUDE_proportional_calculation_l2087_208795

/-- Given that 2994 ã · 14.5 = 171, prove that 29.94 ã · 1.45 = 1.71 -/
theorem proportional_calculation (h : 2994 * 14.5 = 171) : 29.94 * 1.45 = 1.71 := by
  sorry

end NUMINAMATH_CALUDE_proportional_calculation_l2087_208795


namespace NUMINAMATH_CALUDE_fraction_inequality_l2087_208731

theorem fraction_inequality (a b : ℝ) : ¬(∀ a b, a / b = (a + 1) / (b + 1)) :=
sorry

end NUMINAMATH_CALUDE_fraction_inequality_l2087_208731


namespace NUMINAMATH_CALUDE_first_house_price_correct_l2087_208733

/-- Represents the price of Tommy's first house in dollars -/
def first_house_price : ℝ := 400000

/-- Represents the price of Tommy's new house in dollars -/
def new_house_price : ℝ := 500000

/-- Represents the loan percentage for the new house -/
def loan_percentage : ℝ := 0.75

/-- Represents the annual interest rate for the loan -/
def annual_interest_rate : ℝ := 0.035

/-- Represents the loan term in years -/
def loan_term : ℕ := 15

/-- Represents the annual property tax rate -/
def property_tax_rate : ℝ := 0.015

/-- Represents the annual home insurance cost in dollars -/
def annual_insurance_cost : ℝ := 7500

/-- Theorem stating that the first house price is correct given the conditions -/
theorem first_house_price_correct :
  first_house_price = new_house_price / 1.25 ∧
  new_house_price = first_house_price * 1.25 ∧
  loan_percentage * new_house_price * annual_interest_rate +
  property_tax_rate * new_house_price +
  annual_insurance_cost =
  28125 :=
sorry

end NUMINAMATH_CALUDE_first_house_price_correct_l2087_208733


namespace NUMINAMATH_CALUDE_selection_ways_eq_55_l2087_208722

/-- The number of ways to select 5 students out of 5 male and 3 female students,
    ensuring both male and female students are included. -/
def selection_ways : ℕ :=
  Nat.choose 8 5 - Nat.choose 5 5

/-- Theorem stating that the number of ways to select 5 students
    out of 5 male and 3 female students, ensuring both male and
    female students are included, is equal to 55. -/
theorem selection_ways_eq_55 : selection_ways = 55 := by
  sorry

end NUMINAMATH_CALUDE_selection_ways_eq_55_l2087_208722


namespace NUMINAMATH_CALUDE_correct_answer_probability_l2087_208739

/-- The probability of correctly answering one MCQ question with 3 options -/
def mcq_prob : ℚ := 1 / 3

/-- The probability of correctly answering a true/false question -/
def tf_prob : ℚ := 1 / 2

/-- The number of true/false questions -/
def num_tf_questions : ℕ := 2

/-- The probability of correctly answering all questions -/
def total_prob : ℚ := mcq_prob * tf_prob ^ num_tf_questions

theorem correct_answer_probability :
  total_prob = 1 / 12 := by sorry

end NUMINAMATH_CALUDE_correct_answer_probability_l2087_208739


namespace NUMINAMATH_CALUDE_jerry_showers_l2087_208724

/-- Represents the water usage scenario for Jerry's household --/
structure WaterUsage where
  total_allowance : ℕ
  drinking_cooking : ℕ
  shower_usage : ℕ
  pool_length : ℕ
  pool_width : ℕ
  pool_height : ℕ
  gallon_to_cubic_foot : ℕ

/-- Calculates the number of showers Jerry can take in July --/
def calculate_showers (w : WaterUsage) : ℕ :=
  let pool_volume := w.pool_length * w.pool_width * w.pool_height
  let remaining_water := w.total_allowance - w.drinking_cooking - pool_volume
  remaining_water / w.shower_usage

/-- Theorem stating that Jerry can take 15 showers in July --/
theorem jerry_showers :
  let w : WaterUsage := {
    total_allowance := 1000,
    drinking_cooking := 100,
    shower_usage := 20,
    pool_length := 10,
    pool_width := 10,
    pool_height := 6,
    gallon_to_cubic_foot := 1
  }
  calculate_showers w = 15 := by
  sorry

#eval calculate_showers {
  total_allowance := 1000,
  drinking_cooking := 100,
  shower_usage := 20,
  pool_length := 10,
  pool_width := 10,
  pool_height := 6,
  gallon_to_cubic_foot := 1
}

end NUMINAMATH_CALUDE_jerry_showers_l2087_208724


namespace NUMINAMATH_CALUDE_doughnut_cost_l2087_208773

/-- The cost of one dozen doughnuts -/
def cost_one_dozen : ℝ := sorry

/-- The cost of two dozen doughnuts -/
def cost_two_dozen : ℝ := 14

theorem doughnut_cost : cost_one_dozen = 7 := by
  sorry

end NUMINAMATH_CALUDE_doughnut_cost_l2087_208773


namespace NUMINAMATH_CALUDE_min_distance_point_to_tangent_l2087_208780

/-- The minimum distance between a point on the line x - y - 6 = 0 and 
    its tangent point on the circle (x-1)^2 + (y-1)^2 = 4 is √14 -/
theorem min_distance_point_to_tangent (x y : ℝ) : 
  let circle := {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 1)^2 = 4}
  let line := {p : ℝ × ℝ | p.1 - p.2 - 6 = 0}
  ∃ (M N : ℝ × ℝ), M ∈ line ∧ N ∈ circle ∧ 
    (∀ (M' N' : ℝ × ℝ), M' ∈ line → N' ∈ circle → 
      (M'.1 - N'.1)^2 + (M'.2 - N'.2)^2 ≥ (M.1 - N.1)^2 + (M.2 - N.2)^2) ∧
    (M.1 - N.1)^2 + (M.2 - N.2)^2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_min_distance_point_to_tangent_l2087_208780


namespace NUMINAMATH_CALUDE_star_two_neg_three_l2087_208798

-- Define the ★ operation
def star (a b : ℤ) : ℤ := a * b^3 - 2*b + 2

-- Theorem statement
theorem star_two_neg_three : star 2 (-3) = -46 := by
  sorry

end NUMINAMATH_CALUDE_star_two_neg_three_l2087_208798


namespace NUMINAMATH_CALUDE_basketball_win_rate_l2087_208760

theorem basketball_win_rate (total_games : ℕ) (first_part_games : ℕ) (games_won : ℕ) 
  (remaining_games : ℕ) (target_percentage : ℚ) :
  total_games = first_part_games + remaining_games →
  games_won ≤ first_part_games →
  target_percentage = 3 / 4 →
  ∃ (x : ℕ), x ≤ remaining_games ∧ 
    (games_won + x : ℚ) / total_games = target_percentage ∧
    x = 38 :=
by
  sorry

#check basketball_win_rate 130 80 60 50 (3/4)

end NUMINAMATH_CALUDE_basketball_win_rate_l2087_208760


namespace NUMINAMATH_CALUDE_exists_a_min_value_3_l2087_208797

open Real

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * x - log x

theorem exists_a_min_value_3 :
  ∃ a : ℝ, ∀ x : ℝ, 0 < x → x ≤ exp 1 → g a x ≥ 3 ∧
  ∃ x₀ : ℝ, 0 < x₀ ∧ x₀ ≤ exp 1 ∧ g a x₀ = 3 :=
by sorry

end NUMINAMATH_CALUDE_exists_a_min_value_3_l2087_208797


namespace NUMINAMATH_CALUDE_fraction_simplification_l2087_208796

theorem fraction_simplification (x y : ℝ) (hx : -x ≥ 0) (hy : -y ≥ 0) :
  (Real.sqrt (-x) - Real.sqrt (-3 * y)) / (x + 3 * y + 2 * Real.sqrt (3 * x * y)) =
  1 / (Real.sqrt (-3 * y) - Real.sqrt (-x)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2087_208796


namespace NUMINAMATH_CALUDE_unique_distribution_l2087_208753

structure Desserts where
  coconut : Nat
  meringue : Nat
  caramel : Nat

def total_desserts (d : Desserts) : Nat :=
  d.coconut + d.meringue + d.caramel

def is_valid_distribution (d : Desserts) : Prop :=
  total_desserts d = 10 ∧
  d.coconut < d.meringue ∧
  d.meringue < d.caramel ∧
  d.caramel ≥ 6 ∧
  (d.coconut + d.meringue ≥ 3)

theorem unique_distribution :
  ∃! d : Desserts, is_valid_distribution d ∧ d.coconut = 1 ∧ d.meringue = 2 ∧ d.caramel = 7 :=
by sorry

end NUMINAMATH_CALUDE_unique_distribution_l2087_208753


namespace NUMINAMATH_CALUDE_papaya_problem_l2087_208770

/-- The number of fruits that turned yellow on Friday -/
def friday_yellow : ℕ := 2

theorem papaya_problem (initial_green : ℕ) (final_green : ℕ) :
  initial_green = 14 →
  final_green = 8 →
  initial_green - final_green = friday_yellow + 2 * friday_yellow →
  friday_yellow = 2 := by
  sorry

#check papaya_problem

end NUMINAMATH_CALUDE_papaya_problem_l2087_208770


namespace NUMINAMATH_CALUDE_jose_age_l2087_208742

/-- Given the ages of Jose, Zack, and Inez, prove that Jose is 21 years old -/
theorem jose_age (jose zack inez : ℕ) 
  (h1 : jose = zack + 5) 
  (h2 : zack = inez + 4) 
  (h3 : inez = 12) : 
  jose = 21 := by
  sorry

end NUMINAMATH_CALUDE_jose_age_l2087_208742


namespace NUMINAMATH_CALUDE_weight_difference_is_35_l2087_208746

def labrador_start : ℝ := 40
def dachshund_start : ℝ := 12
def weight_gain_percentage : ℝ := 0.25

def weight_difference : ℝ :=
  (labrador_start + labrador_start * weight_gain_percentage) -
  (dachshund_start + dachshund_start * weight_gain_percentage)

theorem weight_difference_is_35 : weight_difference = 35 := by
  sorry

end NUMINAMATH_CALUDE_weight_difference_is_35_l2087_208746


namespace NUMINAMATH_CALUDE_nth_equation_l2087_208771

/-- The product of consecutive integers from n+1 to n+n -/
def leftSide (n : ℕ) : ℕ := (n + 1).factorial / n.factorial

/-- The product of odd numbers from 1 to 2n-1 -/
def oddProduct (n : ℕ) : ℕ := 
  Finset.prod (Finset.range n) (fun i => 2 * i + 1)

/-- The statement of the equality to be proved -/
theorem nth_equation (n : ℕ) : 
  leftSide n = 2^n * oddProduct n := by
  sorry

end NUMINAMATH_CALUDE_nth_equation_l2087_208771


namespace NUMINAMATH_CALUDE_special_triangle_bisecting_lines_angle_l2087_208756

/-- Triangle with specific side lengths -/
structure SpecialTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  a_eq : a = 13
  b_eq : b = 14
  c_eq : c = 15

/-- A line that bisects both perimeter and area of the triangle -/
structure BisectingLine (t : SpecialTriangle) where
  bisects_perimeter : Bool
  bisects_area : Bool

/-- The acute angle between two bisecting lines -/
def acute_angle (t : SpecialTriangle) (l1 l2 : BisectingLine t) : ℝ := sorry

theorem special_triangle_bisecting_lines_angle 
  (t : SpecialTriangle) 
  (l1 l2 : BisectingLine t) 
  (h_unique : ∀ (l : BisectingLine t), l = l1 ∨ l = l2) :
  Real.tan (acute_angle t l1 l2) = Real.sqrt 6 / 12 := by sorry

end NUMINAMATH_CALUDE_special_triangle_bisecting_lines_angle_l2087_208756


namespace NUMINAMATH_CALUDE_container_weight_problem_l2087_208738

theorem container_weight_problem (x y z : ℝ) 
  (h1 : x + y = 234)
  (h2 : y + z = 241)
  (h3 : z + x = 255) :
  x + y + z = 365 := by
sorry

end NUMINAMATH_CALUDE_container_weight_problem_l2087_208738


namespace NUMINAMATH_CALUDE_characterize_positive_product_set_l2087_208728

def positive_product_set : Set ℤ :=
  {a : ℤ | (5 + a) * (3 - a) > 0}

theorem characterize_positive_product_set :
  positive_product_set = {-4, -3, -2, -1, 0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_characterize_positive_product_set_l2087_208728


namespace NUMINAMATH_CALUDE_inequality_preservation_l2087_208720

theorem inequality_preservation (a b c : ℝ) (h : a > b) : 
  a / (c^2 + 1) > b / (c^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l2087_208720


namespace NUMINAMATH_CALUDE_circle_within_circle_l2087_208718

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- A point is inside a circle if its distance from the center is less than the radius -/
def is_inside (p : ℝ × ℝ) (c : Circle) : Prop :=
  Real.sqrt ((p.1 - c.center.1)^2 + (p.2 - c.center.2)^2) < c.radius

/-- A circle is contained within another circle if all its points are inside the larger circle -/
def is_contained (inner outer : Circle) : Prop :=
  ∀ p : ℝ × ℝ, is_inside p inner → is_inside p outer

theorem circle_within_circle (C : Circle) (A B : ℝ × ℝ) 
    (hA : is_inside A C) (hB : is_inside B C) :
  ∃ D : Circle, is_inside A D ∧ is_inside B D ∧ is_contained D C := by
  sorry

end NUMINAMATH_CALUDE_circle_within_circle_l2087_208718


namespace NUMINAMATH_CALUDE_decimal_multiplication_l2087_208741

theorem decimal_multiplication : (0.7 : ℝ) * 0.8 = 0.56 := by
  sorry

end NUMINAMATH_CALUDE_decimal_multiplication_l2087_208741


namespace NUMINAMATH_CALUDE_probability_at_least_two_same_l2087_208721

/-- The number of sides on each die -/
def num_sides : ℕ := 8

/-- The number of dice rolled -/
def num_dice : ℕ := 8

/-- The probability of at least two dice showing the same number when rolling 8 fair 8-sided dice -/
theorem probability_at_least_two_same : 
  (1 - (Nat.factorial num_dice : ℚ) / (num_sides ^ num_dice : ℚ)) = 16736996 / 16777216 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_two_same_l2087_208721


namespace NUMINAMATH_CALUDE_fraction_subtraction_division_l2087_208744

theorem fraction_subtraction_division : 
  (10 : ℚ) / 5 - (10 : ℚ) / 2 / ((2 : ℚ) / 5) = -21 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_division_l2087_208744


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l2087_208766

theorem pure_imaginary_condition (θ : ℝ) : 
  let z : ℂ := (Complex.exp (Complex.I * -θ)) * (1 + Complex.I)
  θ = 3 * Real.pi / 4 → Complex.re z = 0 ∧ Complex.im z ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l2087_208766


namespace NUMINAMATH_CALUDE_quadratic_root_l2087_208769

theorem quadratic_root (b : ℝ) : 
  (1 : ℝ) ^ 2 + b * 1 + 2 = 0 → ∃ x : ℝ, x ≠ 1 ∧ x ^ 2 + b * x + 2 = 0 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_l2087_208769


namespace NUMINAMATH_CALUDE_meaningful_expression_range_l2087_208787

-- Define the set of real numbers where the expression is meaningful
def meaningfulSet : Set ℝ :=
  {x : ℝ | 3 - x ≥ 0 ∧ x + 1 ≠ 0}

-- Theorem statement
theorem meaningful_expression_range :
  meaningfulSet = {x : ℝ | x ≤ 3 ∧ x ≠ -1} := by
  sorry

end NUMINAMATH_CALUDE_meaningful_expression_range_l2087_208787


namespace NUMINAMATH_CALUDE_seventh_term_is_eleven_l2087_208777

/-- An arithmetic sequence with specified properties -/
structure ArithmeticSequence where
  /-- The first term of the sequence -/
  a : ℝ
  /-- The common difference of the sequence -/
  d : ℝ
  /-- The sum of the first five terms is 35 -/
  sum_first_five : a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) = 35
  /-- The sixth term is 10 -/
  sixth_term : a + 5*d = 10

/-- The seventh term of the arithmetic sequence is 11 -/
theorem seventh_term_is_eleven (seq : ArithmeticSequence) : seq.a + 6*seq.d = 11 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_is_eleven_l2087_208777


namespace NUMINAMATH_CALUDE_brick_length_l2087_208775

/-- The surface area of a rectangular prism -/
def surface_area (l w h : ℝ) : ℝ := 2 * l * w + 2 * l * h + 2 * w * h

/-- Theorem: For a brick with width 4 cm, height 2 cm, and surface area 136 square centimeters, the length of the brick is 10 cm -/
theorem brick_length : 
  ∃ (l : ℝ), surface_area l 4 2 = 136 ∧ l = 10 :=
sorry

end NUMINAMATH_CALUDE_brick_length_l2087_208775


namespace NUMINAMATH_CALUDE_fixed_order_queue_arrangement_l2087_208778

def queue_arrangements (n : ℕ) (k : ℕ) : Prop :=
  n ≥ k ∧ (n - k).factorial * k.factorial * (n.choose k) = 20

theorem fixed_order_queue_arrangement :
  queue_arrangements 5 3 :=
sorry

end NUMINAMATH_CALUDE_fixed_order_queue_arrangement_l2087_208778


namespace NUMINAMATH_CALUDE_als_original_portion_l2087_208757

theorem als_original_portion (total_initial : ℕ) (total_final : ℕ) 
  (al_loss : ℕ) (al betty clare : ℕ) :
  total_initial = 1500 →
  total_final = 2250 →
  al_loss = 150 →
  al + betty + clare = total_initial →
  (al - al_loss) + 3 * betty + 3 * clare = total_final →
  al = 1050 :=
by sorry

end NUMINAMATH_CALUDE_als_original_portion_l2087_208757


namespace NUMINAMATH_CALUDE_two_unique_pairs_for_15_l2087_208740

/-- The number of unique pairs of nonnegative integers (a, b) satisfying a^2 - b^2 = n, for n = 15 -/
def uniquePairsCount (n : Nat) : Nat :=
  (Finset.filter (fun p : Nat × Nat => 
    p.1 ≥ p.2 ∧ p.1^2 - p.2^2 = n) (Finset.product (Finset.range (n+1)) (Finset.range (n+1)))).card

/-- Theorem stating that there are exactly 2 unique pairs for n = 15 -/
theorem two_unique_pairs_for_15 : uniquePairsCount 15 = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_unique_pairs_for_15_l2087_208740


namespace NUMINAMATH_CALUDE_find_a_l2087_208714

def U : Finset ℕ := {1, 3, 5, 7}

theorem find_a (a : ℕ) : 
  let M : Finset ℕ := {1, a - 5}
  M ⊆ U ∧ 
  (U \ M : Finset ℕ) = {5, 7} →
  a = 8 := by
sorry

end NUMINAMATH_CALUDE_find_a_l2087_208714


namespace NUMINAMATH_CALUDE_no_integer_solution_l2087_208761

theorem no_integer_solution :
  ¬ ∃ (m n k : ℕ+), ∀ (x y : ℝ),
    (x + 1)^2 + y^2 = (m : ℝ)^2 ∧
    (x - 1)^2 + y^2 = (n : ℝ)^2 ∧
    x^2 + (y - Real.sqrt 3)^2 = (k : ℝ)^2 :=
by sorry

end NUMINAMATH_CALUDE_no_integer_solution_l2087_208761


namespace NUMINAMATH_CALUDE_three_digit_repeating_decimal_cube_l2087_208750

theorem three_digit_repeating_decimal_cube (n : ℕ) : 
  (n < 1000 ∧ n > 0) →
  (∃ (a b : ℕ), b > a ∧ a > 0 ∧ b > 0 ∧ (n : ℚ) / 999 = (a : ℚ) / b ^ 3) →
  (n = 037 ∨ n = 296) :=
sorry

end NUMINAMATH_CALUDE_three_digit_repeating_decimal_cube_l2087_208750


namespace NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l2087_208704

def C : Set Nat := {64, 66, 67, 68, 71}

def has_smallest_prime_factor (n : Nat) (s : Set Nat) : Prop :=
  n ∈ s ∧ ∀ m ∈ s, ∃ p : Nat, Prime p ∧ p ∣ n ∧ ∀ q : Nat, Prime q → q ∣ m → p ≤ q

theorem smallest_prime_factor_in_C :
  has_smallest_prime_factor 64 C ∧
  has_smallest_prime_factor 66 C ∧
  has_smallest_prime_factor 68 C :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l2087_208704


namespace NUMINAMATH_CALUDE_indeterminate_product_at_opposite_points_l2087_208726

-- Define a continuous function on an open interval
def ContinuousOnOpenInterval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, a < x ∧ x < b → ContinuousAt f x

-- Define the property of having a single root at 0 in the interval (-2, 2)
def SingleRootAtZero (f : ℝ → ℝ) : Prop :=
  (∀ x, -2 < x ∧ x < 2 ∧ f x = 0 → x = 0) ∧
  (f 0 = 0)

-- Theorem statement
theorem indeterminate_product_at_opposite_points
  (f : ℝ → ℝ)
  (h_cont : ContinuousOnOpenInterval f (-2) 2)
  (h_root : SingleRootAtZero f) :
  ∃ (f₁ f₂ f₃ : ℝ → ℝ),
    (ContinuousOnOpenInterval f₁ (-2) 2 ∧ SingleRootAtZero f₁ ∧ f₁ (-1) * f₁ 1 > 0) ∧
    (ContinuousOnOpenInterval f₂ (-2) 2 ∧ SingleRootAtZero f₂ ∧ f₂ (-1) * f₂ 1 < 0) ∧
    (ContinuousOnOpenInterval f₃ (-2) 2 ∧ SingleRootAtZero f₃ ∧ f₃ (-1) * f₃ 1 = 0) :=
  sorry

end NUMINAMATH_CALUDE_indeterminate_product_at_opposite_points_l2087_208726


namespace NUMINAMATH_CALUDE_expand_expression_l2087_208762

theorem expand_expression (x y z : ℝ) : 
  (2*x + 5) * (3*y + 15 + 4*z) = 6*x*y + 30*x + 8*x*z + 15*y + 20*z + 75 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2087_208762


namespace NUMINAMATH_CALUDE_union_P_complement_Q_l2087_208752

open Set

-- Define the sets P and Q
def P : Set ℝ := {x | x^2 - 4*x + 3 ≤ 0}
def Q : Set ℝ := {x | x^2 - 4 < 0}

-- State the theorem
theorem union_P_complement_Q : P ∪ (univ \ Q) = Iic (-2) ∪ Ici 1 := by sorry

end NUMINAMATH_CALUDE_union_P_complement_Q_l2087_208752


namespace NUMINAMATH_CALUDE_min_chord_length_l2087_208768

/-- The minimum length of a chord passing through (1,1) in the circle (x-2)^2 + (y-3)^2 = 9 is 4 -/
theorem min_chord_length (x y : ℝ) : 
  let circle := fun (x y : ℝ) => (x - 2)^2 + (y - 3)^2 = 9
  let point := (1, 1)
  let chord_length := fun (a b c d : ℝ) => Real.sqrt ((a - c)^2 + (b - d)^2)
  ∃ (a b c d : ℝ), 
    circle a b ∧ circle c d ∧ 
    (1 - a) * (d - b) = (1 - c) * (b - 1) ∧ 
    (∀ (e f g h : ℝ), circle e f ∧ circle g h ∧ 
      (1 - e) * (h - f) = (1 - g) * (f - 1) → 
      chord_length a b c d ≤ chord_length e f g h) ∧
    chord_length a b c d = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_chord_length_l2087_208768


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2087_208790

theorem polynomial_factorization (a x : ℝ) : 2*a*x^2 - 12*a*x + 18*a = 2*a*(x-3)^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2087_208790


namespace NUMINAMATH_CALUDE_roots_of_quadratic_equation_l2087_208783

theorem roots_of_quadratic_equation (a b : ℝ) : 
  (a^2 + a - 5 = 0) → 
  (b^2 + b - 5 = 0) → 
  (a + b = -1) → 
  (a * b = -5) → 
  (2 * a^2 + a + b^2 = 16) := by
sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_equation_l2087_208783


namespace NUMINAMATH_CALUDE_tutor_schedule_lcm_l2087_208759

theorem tutor_schedule_lcm : Nat.lcm 5 (Nat.lcm 9 (Nat.lcm 8 11)) = 3960 := by
  sorry

end NUMINAMATH_CALUDE_tutor_schedule_lcm_l2087_208759


namespace NUMINAMATH_CALUDE_square_area_12m_l2087_208788

/-- The area of a square with side length 12 meters is 144 square meters. -/
theorem square_area_12m : 
  let side_length : ℝ := 12
  let area : ℝ := side_length ^ 2
  area = 144 := by sorry

end NUMINAMATH_CALUDE_square_area_12m_l2087_208788


namespace NUMINAMATH_CALUDE_number_difference_l2087_208709

theorem number_difference (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a / b = 2 / 3) (h4 : a^3 + b^3 = 945) : b - a = 3 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l2087_208709


namespace NUMINAMATH_CALUDE_pizza_toppings_l2087_208706

theorem pizza_toppings (total_slices : ℕ) (pepperoni_slices : ℕ) (mushroom_slices : ℕ) 
  (h1 : total_slices = 24)
  (h2 : pepperoni_slices = 14)
  (h3 : mushroom_slices = 16)
  (h4 : ∀ s, s ≤ total_slices → (s ≤ pepperoni_slices ∨ s ≤ mushroom_slices)) :
  ∃ both_toppings : ℕ, 
    both_toppings = pepperoni_slices + mushroom_slices - total_slices ∧ 
    both_toppings = 6 :=
by sorry

end NUMINAMATH_CALUDE_pizza_toppings_l2087_208706


namespace NUMINAMATH_CALUDE_rectangular_solid_diagonal_angles_sum_l2087_208754

/-- A rectangular solid with a diagonal forming angles with edges. -/
structure RectangularSolid where
  /-- Length of the diagonal -/
  diagonal : ℝ
  /-- Angle between diagonal and first edge -/
  α : ℝ
  /-- Angle between diagonal and second edge -/
  β : ℝ
  /-- Angle between diagonal and third edge -/
  γ : ℝ
  /-- The angles are formed by the diagonal and edges of the rectangular solid -/
  angles_from_edges : True

/-- 
In a rectangular solid, if one of the diagonals forms angles α, β, and γ 
with the three edges emanating from one of the vertices, 
then cos²α + cos²β + cos²γ = 1.
-/
theorem rectangular_solid_diagonal_angles_sum 
  (rs : RectangularSolid) : Real.cos rs.α ^ 2 + Real.cos rs.β ^ 2 + Real.cos rs.γ ^ 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_diagonal_angles_sum_l2087_208754


namespace NUMINAMATH_CALUDE_inequalities_proof_l2087_208712

theorem inequalities_proof :
  (∀ x > 0, Real.log x ≥ 1 - 1/x) ∧
  (∃ x > 0, Real.sin (2*x) ≥ x) ∧
  ((1 + Real.tan (π/12)) / (1 - Real.tan (π/12)) > π/3) ∧
  (∀ x > 0, Real.exp x > 2 * Real.sin x) := by
sorry

end NUMINAMATH_CALUDE_inequalities_proof_l2087_208712


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2087_208789

theorem complex_fraction_equality (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (hab : a + b ≠ 0) (h : a^3 + a^2*b + a*b^2 + b^3 = 0) : 
  (a^12 + b^12) / (a + b)^12 = 2/81 := by
sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2087_208789


namespace NUMINAMATH_CALUDE_divisiblity_by_2018_l2087_208713

theorem divisiblity_by_2018 (k : ℕ) (h : k > 0) (hodd : Odd k) :
  let n := 252 * k
  2018 ∣ (1 + 2^n + 3^n + 4^n) := by
  sorry

end NUMINAMATH_CALUDE_divisiblity_by_2018_l2087_208713


namespace NUMINAMATH_CALUDE_rosie_pies_l2087_208782

/-- The number of apples required to make one pie -/
def apples_per_pie : ℕ := 5

/-- The total number of apples Rosie has -/
def total_apples : ℕ := 32

/-- The maximum number of whole pies that can be made -/
def max_pies : ℕ := total_apples / apples_per_pie

theorem rosie_pies :
  max_pies = 6 :=
sorry

end NUMINAMATH_CALUDE_rosie_pies_l2087_208782


namespace NUMINAMATH_CALUDE_b_77_mod_40_l2087_208749

def b (n : ℕ) : ℕ := 5^n + 9^n

theorem b_77_mod_40 : b 77 ≡ 14 [MOD 40] := by
  sorry

end NUMINAMATH_CALUDE_b_77_mod_40_l2087_208749


namespace NUMINAMATH_CALUDE_complex_quadrant_l2087_208732

theorem complex_quadrant (z : ℂ) (h : (1 - Complex.I) * z = 2 * Complex.I) :
  z.re < 0 ∧ z.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_quadrant_l2087_208732


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2087_208792

/-- Given a geometric sequence with sum of first n terms Sn = 24 and sum of first 3n terms S3n = 42,
    prove that the sum of first 2n terms S2n = 36 -/
theorem geometric_sequence_sum (n : ℕ) (Sn S2n S3n : ℝ) : 
  Sn = 24 → S3n = 42 → (S2n - Sn)^2 = Sn * (S3n - S2n) → S2n = 36 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2087_208792


namespace NUMINAMATH_CALUDE_olivias_wallet_after_supermarket_l2087_208737

/-- The amount of money left in Olivia's wallet after visiting the supermarket -/
def money_left (initial_amount spent : ℕ) : ℕ :=
  initial_amount - spent

/-- Theorem stating the amount of money left in Olivia's wallet -/
theorem olivias_wallet_after_supermarket :
  money_left 94 16 = 78 := by sorry

end NUMINAMATH_CALUDE_olivias_wallet_after_supermarket_l2087_208737


namespace NUMINAMATH_CALUDE_average_hiring_per_week_l2087_208702

def employee_hiring (week1 week2 week3 week4 : ℕ) : Prop :=
  (week1 = week2 + 200) ∧
  (week2 + 150 = week3) ∧
  (week4 = 2 * week3) ∧
  (week4 = 400)

theorem average_hiring_per_week 
  (week1 week2 week3 week4 : ℕ) 
  (h : employee_hiring week1 week2 week3 week4) : 
  (week1 + week2 + week3 + week4) / 4 = 225 := by
  sorry

end NUMINAMATH_CALUDE_average_hiring_per_week_l2087_208702


namespace NUMINAMATH_CALUDE_matchstick_pattern_l2087_208774

/-- 
Given a sequence where:
- The first term is 5
- Each subsequent term increases by 3
Prove that the 20th term is 62
-/
theorem matchstick_pattern (a : ℕ → ℕ) 
  (h1 : a 1 = 5)
  (h2 : ∀ n : ℕ, n ≥ 2 → a n = a (n-1) + 3) :
  a 20 = 62 := by
  sorry

end NUMINAMATH_CALUDE_matchstick_pattern_l2087_208774


namespace NUMINAMATH_CALUDE_intersection_line_ellipse_l2087_208779

/-- Prove that if a line y = kx intersects the ellipse x^2/4 + y^2/3 = 1 at points A and B, 
    and the perpendiculars from A and B to the x-axis have their feet at ±1 
    (which are the foci of the ellipse), then k = ± 3/2. -/
theorem intersection_line_ellipse (k : ℝ) : 
  (∀ x y : ℝ, y = k * x → x^2 / 4 + y^2 / 3 = 1 → 
    (x = 1 ∨ x = -1) → k = 3/2 ∨ k = -3/2) := by
  sorry


end NUMINAMATH_CALUDE_intersection_line_ellipse_l2087_208779


namespace NUMINAMATH_CALUDE_right_triangle_integer_area_l2087_208736

theorem right_triangle_integer_area 
  (a b c : ℕ) 
  (h_right_angle : a^2 + b^2 = c^2) 
  (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) : 
  ∃ A : ℕ, 2 * A = a * b :=
sorry

end NUMINAMATH_CALUDE_right_triangle_integer_area_l2087_208736


namespace NUMINAMATH_CALUDE_exponent_division_l2087_208719

theorem exponent_division (a : ℝ) (m n : ℕ) :
  a ^ m / a ^ n = a ^ (m - n) :=
sorry

end NUMINAMATH_CALUDE_exponent_division_l2087_208719


namespace NUMINAMATH_CALUDE_triangle_inequality_l2087_208784

variable (A B C : ℝ) -- Angles of the triangle
variable (da db dc : ℝ) -- Distances from P to sides
variable (Ra Rb Rc : ℝ) -- Distances from P to vertices

-- Assume all variables are non-negative
variable (h1 : 0 ≤ A) (h2 : 0 ≤ B) (h3 : 0 ≤ C)
variable (h4 : 0 ≤ da) (h5 : 0 ≤ db) (h6 : 0 ≤ dc)
variable (h7 : 0 ≤ Ra) (h8 : 0 ≤ Rb) (h9 : 0 ≤ Rc)

-- Assume A, B, C form a valid triangle
variable (h10 : A + B + C = Real.pi)

theorem triangle_inequality (A B C da db dc Ra Rb Rc : ℝ)
  (h1 : 0 ≤ A) (h2 : 0 ≤ B) (h3 : 0 ≤ C)
  (h4 : 0 ≤ da) (h5 : 0 ≤ db) (h6 : 0 ≤ dc)
  (h7 : 0 ≤ Ra) (h8 : 0 ≤ Rb) (h9 : 0 ≤ Rc)
  (h10 : A + B + C = Real.pi) :
  3 * (da^2 + db^2 + dc^2) ≥ (Ra * Real.sin A)^2 + (Rb * Real.sin B)^2 + (Rc * Real.sin C)^2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2087_208784


namespace NUMINAMATH_CALUDE_tan_x_axis_intersection_l2087_208725

theorem tan_x_axis_intersection :
  ∀ (x : ℝ), (∃ (n : ℤ), x = -π/8 + n*π/2) ↔ Real.tan (2*x + π/4) = 0 :=
by sorry

end NUMINAMATH_CALUDE_tan_x_axis_intersection_l2087_208725


namespace NUMINAMATH_CALUDE_grape_rate_proof_l2087_208793

/-- The rate per kg for grapes -/
def grape_rate : ℝ := 70

/-- The rate per kg for mangoes -/
def mango_rate : ℝ := 55

/-- The quantity of grapes and mangoes in kg -/
def quantity : ℝ := 8

/-- The total cost paid to the shopkeeper -/
def total_cost : ℝ := 1000

theorem grape_rate_proof :
  grape_rate * quantity + mango_rate * quantity = total_cost :=
by sorry

end NUMINAMATH_CALUDE_grape_rate_proof_l2087_208793


namespace NUMINAMATH_CALUDE_hyperbola_intersection_line_l2087_208763

theorem hyperbola_intersection_line (θ : Real) : 
  let ρ := λ θ : Real => 3 / (1 - 2 * Real.cos θ)
  let A := (ρ θ, θ)
  let B := (ρ (θ + π), θ + π)
  let distance := |ρ θ + ρ (θ + π)|
  distance = 6 → 
    θ = π/2 ∨ θ = π/4 ∨ θ = 3*π/4 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_intersection_line_l2087_208763


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l2087_208715

/-- A geometric sequence with first term 1/3 and the property that 2a_2 = a_4 -/
def geometric_sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 1/3 ∧
  (∃ q : ℚ, ∀ n : ℕ, a n = 1/3 * q^(n-1)) ∧
  2 * (a 2) = a 4

theorem geometric_sequence_fifth_term
  (a : ℕ → ℚ)
  (h : geometric_sequence a) :
  a 5 = 4/3 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l2087_208715


namespace NUMINAMATH_CALUDE_equation_solution_l2087_208751

theorem equation_solution : ∃ x : ℝ, 
  (216 + Real.sqrt 41472 - 18 * x - Real.sqrt (648 * x^2) = 0) ∧ 
  (x = (140 * Real.sqrt 2 - 140) / 9) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2087_208751


namespace NUMINAMATH_CALUDE_triangle_side_b_value_l2087_208729

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if the area is 2√3, B = π/3, and a² + c² = 3ac, then b = 4. -/
theorem triangle_side_b_value (a b c : ℝ) (A B C : ℝ) :
  (1/2 * a * c * Real.sin B = 2 * Real.sqrt 3) →  -- Area condition
  (B = π/3) →                                     -- Angle B condition
  (a^2 + c^2 = 3*a*c) →                           -- Relation between a, c
  (b^2 = a^2 + c^2 - 2*a*c*Real.cos B) →          -- Law of cosines
  (b = 4) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_b_value_l2087_208729


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2087_208767

theorem necessary_but_not_sufficient (x : ℝ) :
  (∀ x, (abs x = -x → x^2 ≥ -x)) ∧
  (∃ x, x^2 ≥ -x ∧ abs x ≠ -x) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2087_208767


namespace NUMINAMATH_CALUDE_min_value_expression_l2087_208700

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  ∃ m : ℝ, m = -2031948.5 ∧ 
  ∀ x y : ℝ, x > 0 → y > 0 → 
    (x + 1/y) * (x + 1/y - 2023) + (y + 1/x) * (y + 1/x - 2023) ≥ m ∧
    (a + 1/b) * (a + 1/b - 2023) + (b + 1/a) * (b + 1/a - 2023) = m :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2087_208700


namespace NUMINAMATH_CALUDE_smallest_number_with_conditions_l2087_208705

def alice_number : ℕ := 30

def has_all_prime_factors_except_7 (n : ℕ) : Prop :=
  ∀ p : ℕ, p.Prime → p ∣ alice_number → p ≠ 7 → p ∣ n

theorem smallest_number_with_conditions :
  ∃ (bob_number : ℕ), bob_number > 0 ∧
  has_all_prime_factors_except_7 bob_number ∧
  ∀ m : ℕ, m > 0 → has_all_prime_factors_except_7 m → bob_number ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_conditions_l2087_208705


namespace NUMINAMATH_CALUDE_tom_flashlight_batteries_l2087_208799

/-- The number of batteries Tom used on his flashlights -/
def batteries_on_flashlights : ℕ := 28

/-- The number of batteries Tom used in his toys -/
def batteries_in_toys : ℕ := 15

/-- The number of batteries Tom used in his controllers -/
def batteries_in_controllers : ℕ := 2

/-- The difference between the number of batteries on flashlights and in toys -/
def battery_difference : ℕ := 13

theorem tom_flashlight_batteries :
  batteries_on_flashlights = batteries_in_toys + battery_difference := by
  sorry

end NUMINAMATH_CALUDE_tom_flashlight_batteries_l2087_208799


namespace NUMINAMATH_CALUDE_neil_cookies_l2087_208730

theorem neil_cookies (total : ℕ) (first_fraction second_fraction third_fraction : ℚ) : 
  total = 60 ∧ 
  first_fraction = 1/3 ∧ 
  second_fraction = 1/4 ∧ 
  third_fraction = 2/5 →
  total - 
    (total * first_fraction).floor - 
    ((total - (total * first_fraction).floor) * second_fraction).floor - 
    ((total - (total * first_fraction).floor - ((total - (total * first_fraction).floor) * second_fraction).floor) * third_fraction).floor = 18 :=
by sorry

end NUMINAMATH_CALUDE_neil_cookies_l2087_208730


namespace NUMINAMATH_CALUDE_fixed_distance_point_l2087_208758

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- Given vectors a and b, and a vector p satisfying ||p - b|| = 3||p - a||,
    p is at a fixed distance from (9/8)a + (-1/8)b. -/
theorem fixed_distance_point (a b p : V) 
    (h : ‖p - b‖ = 3 * ‖p - a‖) : 
    ∃ (c : ℝ), ∀ (q : V), ‖p - q‖ = c ↔ q = (9/8 : ℝ) • a + (-1/8 : ℝ) • b :=
sorry

end NUMINAMATH_CALUDE_fixed_distance_point_l2087_208758


namespace NUMINAMATH_CALUDE_integer_solution_exists_iff_n_eq_one_l2087_208711

theorem integer_solution_exists_iff_n_eq_one (n : ℕ+) :
  (∃ x : ℤ, x^(n : ℕ) + (2 + x)^(n : ℕ) + (2 - x)^(n : ℕ) = 0) ↔ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_integer_solution_exists_iff_n_eq_one_l2087_208711


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l2087_208743

theorem completing_square_equivalence :
  ∀ x : ℝ, 3 * x^2 + 4 * x + 1 = 0 ↔ (x + 2/3)^2 = 1/9 :=
by sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l2087_208743


namespace NUMINAMATH_CALUDE_equilateral_triangle_to_square_l2087_208776

/-- Given an equilateral triangle with area 121√3 cm², prove that decreasing each side by 6 cm
    and transforming it into a square results in a square with area 256 cm². -/
theorem equilateral_triangle_to_square (s : ℝ) : 
  (s^2 * Real.sqrt 3 / 4 = 121 * Real.sqrt 3) →
  ((s - 6)^2 = 256) := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_to_square_l2087_208776


namespace NUMINAMATH_CALUDE_point_outside_circle_if_line_intersects_l2087_208748

/-- A line intersects a circle at two distinct points if and only if
    the distance from the center of the circle to the line is less than the radius -/
axiom line_intersects_circle_iff_distance_lt_radius {a b : ℝ} :
  (∃ x₁ y₁ x₂ y₂ : ℝ, (x₁, y₁) ≠ (x₂, y₂) ∧
    a * x₁ + b * y₁ = 1 ∧ x₁^2 + y₁^2 = 1 ∧
    a * x₂ + b * y₂ = 1 ∧ x₂^2 + y₂^2 = 1) ↔
  (1 / (a^2 + b^2).sqrt < 1)

theorem point_outside_circle_if_line_intersects
  (a b : ℝ)
  (h_intersect : ∃ x₁ y₁ x₂ y₂ : ℝ, (x₁, y₁) ≠ (x₂, y₂) ∧
    a * x₁ + b * y₁ = 1 ∧ x₁^2 + y₁^2 = 1 ∧
    a * x₂ + b * y₂ = 1 ∧ x₂^2 + y₂^2 = 1) :
  a^2 + b^2 > 1 :=
sorry

end NUMINAMATH_CALUDE_point_outside_circle_if_line_intersects_l2087_208748


namespace NUMINAMATH_CALUDE_last_two_digits_product_l2087_208723

theorem last_two_digits_product (A B : ℕ) : 
  A < 10 → B < 10 → A + B = 12 → (10 * A + B) % 3 = 0 → A * B = 35 :=
by sorry

end NUMINAMATH_CALUDE_last_two_digits_product_l2087_208723


namespace NUMINAMATH_CALUDE_solution_set_reciprocal_inequality_l2087_208755

theorem solution_set_reciprocal_inequality (x : ℝ) :
  (∃ y ∈ Set.Ioo (0 : ℝ) (1/3 : ℝ), x = y) ↔ 1/x > 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_reciprocal_inequality_l2087_208755


namespace NUMINAMATH_CALUDE_planes_parallel_from_skew_lines_l2087_208710

-- Define the basic types
variable (Point Line Plane : Type)

-- Define the parallelism relation
variable (parallel : Plane → Plane → Prop)
variable (lineParallelToPlane : Line → Plane → Prop)

-- Define skew lines
variable (skew : Line → Line → Prop)

-- Theorem statement
theorem planes_parallel_from_skew_lines 
  (α β : Plane) (l m : Line) :
  skew l m →
  lineParallelToPlane l α →
  lineParallelToPlane l β →
  lineParallelToPlane m α →
  lineParallelToPlane m β →
  parallel α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_from_skew_lines_l2087_208710


namespace NUMINAMATH_CALUDE_right_triangle_trig_l2087_208794

/-- Given a right-angled triangle ABC where ∠A = 90° and tan C = 2,
    prove that cos C = √5/5 and sin C = 2√5/5 -/
theorem right_triangle_trig (A B C : ℝ) (h1 : A = Real.pi / 2) (h2 : Real.tan C = 2) :
  Real.cos C = Real.sqrt 5 / 5 ∧ Real.sin C = 2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_trig_l2087_208794


namespace NUMINAMATH_CALUDE_james_letter_frequency_l2087_208717

/-- Calculates how many times per week James writes letters to his friends -/
def letters_per_week (pages_per_year : ℕ) (weeks_per_year : ℕ) (pages_per_letter : ℕ) (num_friends : ℕ) : ℕ :=
  (pages_per_year / weeks_per_year) / (pages_per_letter * num_friends)

/-- Theorem stating that James writes letters 2 times per week -/
theorem james_letter_frequency :
  letters_per_week 624 52 3 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_james_letter_frequency_l2087_208717


namespace NUMINAMATH_CALUDE_joan_seashells_l2087_208772

theorem joan_seashells (total : ℝ) (percentage : ℝ) (remaining : ℝ) : 
  total = 79.5 → 
  percentage = 45 → 
  remaining = total - (percentage / 100) * total → 
  remaining = 43.725 := by
sorry

end NUMINAMATH_CALUDE_joan_seashells_l2087_208772


namespace NUMINAMATH_CALUDE_roots_of_product_equation_l2087_208735

theorem roots_of_product_equation (p r : ℝ) (f g : ℝ → ℝ) 
  (hp : p > 0) (hr : r > 0)
  (hf : ∀ x, f x = 0 ↔ x = p)
  (hg : ∀ x, g x = 0 ↔ x = r)
  (hlin_f : ∃ a b, ∀ x, f x = a * x + b)
  (hlin_g : ∃ c d, ∀ x, g x = c * x + d) :
  ∀ x, f x * g x = f 0 * g 0 ↔ x = 0 ∨ x = p + r :=
sorry

end NUMINAMATH_CALUDE_roots_of_product_equation_l2087_208735
