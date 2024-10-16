import Mathlib

namespace NUMINAMATH_CALUDE_tree_distribution_l2210_221021

/-- The number of ways to distribute n indistinguishable objects into k distinct groups,
    with each group containing at least one object -/
def distribute (n k : ℕ) : ℕ :=
  Nat.choose (n + k - 1) (k - 1)

/-- The number of ways to distribute 10 trees over 3 days with at least one tree per day -/
theorem tree_distribution : distribute 10 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_tree_distribution_l2210_221021


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_2_to_1000_l2210_221026

def arithmetic_sequence_sum (a₁ : ℕ) (aₙ : ℕ) (d : ℕ) : ℕ :=
  let n : ℕ := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

theorem arithmetic_sequence_sum_2_to_1000 :
  arithmetic_sequence_sum 2 1000 2 = 250500 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_2_to_1000_l2210_221026


namespace NUMINAMATH_CALUDE_cylinder_ellipse_intersection_l2210_221006

/-- Given a right circular cylinder with radius 2 and a plane intersecting it to form an ellipse,
    if the major axis of the ellipse is 25% longer than its minor axis,
    then the length of the major axis is 5. -/
theorem cylinder_ellipse_intersection (cylinder_radius : ℝ) (minor_axis major_axis : ℝ) :
  cylinder_radius = 2 →
  minor_axis = 2 * cylinder_radius →
  major_axis = 1.25 * minor_axis →
  major_axis = 5 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_ellipse_intersection_l2210_221006


namespace NUMINAMATH_CALUDE_equilateral_parallelogram_diagonal_l2210_221053

/-- A parallelogram composed of four equilateral triangles -/
structure EquilateralParallelogram where
  -- Define the vertices of the parallelogram
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  -- Ensure the parallelogram is made up of four equilateral triangles
  is_equilateral : 
    (dist A B = 2) ∧ 
    (dist B C = 2) ∧ 
    (dist C D = 2) ∧ 
    (dist D A = 2) ∧
    (dist A C = dist B D)
  -- Ensure each equilateral triangle has side length 1
  triangle_side_length : dist A B / 2 = 1

/-- The length of the diagonal in an equilateral parallelogram is √7 -/
theorem equilateral_parallelogram_diagonal 
  (p : EquilateralParallelogram) : dist p.A p.C = Real.sqrt 7 :=
sorry

end NUMINAMATH_CALUDE_equilateral_parallelogram_diagonal_l2210_221053


namespace NUMINAMATH_CALUDE_train_length_l2210_221096

/-- Calculates the length of a train given its speed and the time and distance it takes to cross a bridge -/
theorem train_length (train_speed : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) : 
  train_speed = 72 * (1000 / 3600) →
  bridge_length = 150 →
  crossing_time = 20 →
  (train_speed * crossing_time) - bridge_length = 250 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2210_221096


namespace NUMINAMATH_CALUDE_least_number_divisibility_l2210_221013

theorem least_number_divisibility (n : ℕ) : 
  (∃ k : ℕ, n = 9 * k) ∧ 
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, m = 9 * k)) ∧
  (∃ r : ℕ, r < 5 ∧ r < 6 ∧ r < 7 ∧ r < 8 ∧
    n % 5 = r ∧ n % 6 = r ∧ n % 7 = r ∧ n % 8 = r) ∧
  n = 1680 →
  n % 5 = 0 ∧ n % 6 = 0 ∧ n % 7 = 0 ∧ n % 8 = 0 :=
by sorry

end NUMINAMATH_CALUDE_least_number_divisibility_l2210_221013


namespace NUMINAMATH_CALUDE_total_shirts_produced_l2210_221020

/-- Represents the number of shirts produced per minute -/
def shirts_per_minute : ℕ := 6

/-- Represents the number of minutes the machine operates -/
def operation_time : ℕ := 6

/-- Theorem stating that the total number of shirts produced is 36 -/
theorem total_shirts_produced :
  shirts_per_minute * operation_time = 36 := by
  sorry

end NUMINAMATH_CALUDE_total_shirts_produced_l2210_221020


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l2210_221098

/-- Proves that a train 165 meters long, running at 54 kmph, takes 59 seconds to cross a bridge 720 meters in length. -/
theorem train_bridge_crossing_time :
  let train_length : ℝ := 165
  let bridge_length : ℝ := 720
  let train_speed_kmph : ℝ := 54
  let train_speed_mps : ℝ := train_speed_kmph * 1000 / 3600
  let total_distance : ℝ := train_length + bridge_length
  let crossing_time : ℝ := total_distance / train_speed_mps
  crossing_time = 59 := by sorry

end NUMINAMATH_CALUDE_train_bridge_crossing_time_l2210_221098


namespace NUMINAMATH_CALUDE_route_length_l2210_221094

/-- Given two trains traveling on a route, prove the length of the route. -/
theorem route_length : 
  ∀ (route_length : ℝ) (train_x_speed : ℝ) (train_y_speed : ℝ),
  train_x_speed > 0 →
  train_y_speed > 0 →
  route_length / train_x_speed = 5 →
  route_length / train_y_speed = 4 →
  80 / train_x_speed + (route_length - 80) / train_y_speed = route_length / train_y_speed →
  route_length = 180 := by
  sorry

end NUMINAMATH_CALUDE_route_length_l2210_221094


namespace NUMINAMATH_CALUDE_sum_of_squares_of_coefficients_l2210_221051

-- Define the polynomial
def p (x : ℝ) : ℝ := 6 * (x^5 + 2*x^3 + x^2 + 3)

-- Define a function to get the coefficients of the expanded polynomial
def coefficients (p : ℝ → ℝ) : List ℝ := sorry

-- Define a function to calculate the sum of squares of a list of numbers
def sum_of_squares (l : List ℝ) : ℝ := sorry

-- Theorem statement
theorem sum_of_squares_of_coefficients :
  sum_of_squares (coefficients p) = 540 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_coefficients_l2210_221051


namespace NUMINAMATH_CALUDE_zero_in_interval_l2210_221067

theorem zero_in_interval :
  ∃ c : ℝ, 2 < c ∧ c < 3 ∧ Real.log c - 6 + 2 * c = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_in_interval_l2210_221067


namespace NUMINAMATH_CALUDE_unique_number_property_l2210_221007

theorem unique_number_property : ∃! x : ℝ, x / 3 = x - 3 := by sorry

end NUMINAMATH_CALUDE_unique_number_property_l2210_221007


namespace NUMINAMATH_CALUDE_max_gcd_lcm_product_l2210_221029

theorem max_gcd_lcm_product (a b c : ℕ) 
  (h : Nat.gcd (Nat.lcm a b) c * Nat.lcm (Nat.gcd a b) c = 200) :
  Nat.gcd (Nat.lcm a b) c ≤ 10 ∧ 
  ∃ (a₀ b₀ c₀ : ℕ), Nat.gcd (Nat.lcm a₀ b₀) c₀ = 10 ∧
    Nat.gcd (Nat.lcm a₀ b₀) c₀ * Nat.lcm (Nat.gcd a₀ b₀) c₀ = 200 :=
by sorry

end NUMINAMATH_CALUDE_max_gcd_lcm_product_l2210_221029


namespace NUMINAMATH_CALUDE_product_digit_count_l2210_221035

theorem product_digit_count (k n : ℕ) (a b : ℕ) :
  (10^(k-1) ≤ a ∧ a < 10^k) →
  (10^(n-1) ≤ b ∧ b < 10^n) →
  (10^(k+n-1) ≤ a * b ∧ a * b < 10^(k+n+1)) :=
sorry

end NUMINAMATH_CALUDE_product_digit_count_l2210_221035


namespace NUMINAMATH_CALUDE_sum_medial_areas_is_one_third_l2210_221091

/-- Definition of a medial triangle -/
def medialTriangle (T : Set ℝ × Set ℝ) : Set ℝ × Set ℝ := sorry

/-- Area of a triangle -/
def area (T : Set ℝ × Set ℝ) : ℝ := sorry

/-- Sequence of medial triangles -/
def medialSequence (T : Set ℝ × Set ℝ) : ℕ → Set ℝ × Set ℝ
  | 0 => T
  | n + 1 => medialTriangle (medialSequence T n)

/-- Sum of areas of medial triangles -/
def sumMedialAreas (T : Set ℝ × Set ℝ) : ℝ := sorry

theorem sum_medial_areas_is_one_third (T : Set ℝ × Set ℝ) 
  (h : area T = 1) : sumMedialAreas T = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_sum_medial_areas_is_one_third_l2210_221091


namespace NUMINAMATH_CALUDE_third_year_afforestation_l2210_221023

/-- Represents the yearly afforestation area -/
def afforestation (n : ℕ) : ℝ :=
  match n with
  | 0 => 10000  -- Initial afforestation
  | m + 1 => afforestation m * 1.2  -- 20% increase each year

/-- Theorem stating the area afforested in the third year -/
theorem third_year_afforestation :
  afforestation 2 = 14400 := by
  sorry

end NUMINAMATH_CALUDE_third_year_afforestation_l2210_221023


namespace NUMINAMATH_CALUDE_interest_calculation_l2210_221058

/-- Simple interest calculation -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem interest_calculation (principal rate interest : ℝ) :
  principal = 26 →
  rate = 7 / 100 →
  interest = 10.92 →
  ∃ (time : ℝ), simple_interest principal rate time = interest ∧ time = 6 :=
by sorry

end NUMINAMATH_CALUDE_interest_calculation_l2210_221058


namespace NUMINAMATH_CALUDE_percentage_relation_l2210_221062

theorem percentage_relation (x y z : ℝ) (h1 : y = 0.6 * z) (h2 : x = 0.78 * z) :
  x = y * (1 + 0.3) :=
by sorry

end NUMINAMATH_CALUDE_percentage_relation_l2210_221062


namespace NUMINAMATH_CALUDE_imaginary_unit_seventh_power_l2210_221014

theorem imaginary_unit_seventh_power :
  ∀ i : ℂ, i^2 = -1 → i^7 = -i :=
by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_seventh_power_l2210_221014


namespace NUMINAMATH_CALUDE_polynomial_real_root_l2210_221032

/-- The polynomial p(x) = x^6 + bx^4 - x^3 + bx^2 + 1 -/
def p (b : ℝ) (x : ℝ) : ℝ := x^6 + b*x^4 - x^3 + b*x^2 + 1

/-- The theorem stating the condition for the polynomial to have at least one real root -/
theorem polynomial_real_root (b : ℝ) :
  (∃ x : ℝ, p b x = 0) ↔ b ≤ -3/2 := by sorry

end NUMINAMATH_CALUDE_polynomial_real_root_l2210_221032


namespace NUMINAMATH_CALUDE_fixed_points_of_f_composition_l2210_221045

def f (x : ℝ) : ℝ := x^2 - 4*x

theorem fixed_points_of_f_composition (x : ℝ) : 
  f (f x) = f x ↔ x ∈ ({-1, 0, 4, 5} : Set ℝ) := by
  sorry

end NUMINAMATH_CALUDE_fixed_points_of_f_composition_l2210_221045


namespace NUMINAMATH_CALUDE_std_dev_and_range_invariance_l2210_221082

variable {n : ℕ} (c : ℝ)
variable (X Y : Fin n → ℝ)

def add_constant (X : Fin n → ℝ) (c : ℝ) : Fin n → ℝ :=
  fun i => X i + c

def sample_std_dev (X : Fin n → ℝ) : ℝ := sorry

def sample_range (X : Fin n → ℝ) : ℝ := sorry

theorem std_dev_and_range_invariance
  (h_nonzero : c ≠ 0)
  (h_Y : Y = add_constant X c) :
  sample_std_dev X = sample_std_dev Y ∧
  sample_range X = sample_range Y := by sorry

end NUMINAMATH_CALUDE_std_dev_and_range_invariance_l2210_221082


namespace NUMINAMATH_CALUDE_inscribed_parallelogram_sides_l2210_221095

/-- Triangle ABC with inscribed parallelogram BKLM -/
structure InscribedParallelogram where
  -- Side lengths of triangle ABC
  AB : ℝ
  BC : ℝ
  -- Sides of parallelogram BKLM
  BM : ℝ
  BK : ℝ
  -- Condition that BKLM is inscribed in ABC
  inscribed : BM ≤ BC ∧ BK ≤ AB

/-- The theorem stating the possible side lengths of the inscribed parallelogram -/
theorem inscribed_parallelogram_sides
  (T : InscribedParallelogram)
  (h_AB : T.AB = 18)
  (h_BC : T.BC = 12)
  (h_area : T.BM * T.BK = 48) :
  (T.BM = 8 ∧ T.BK = 6) ∨ (T.BM = 4 ∧ T.BK = 12) := by
  sorry

#check inscribed_parallelogram_sides

end NUMINAMATH_CALUDE_inscribed_parallelogram_sides_l2210_221095


namespace NUMINAMATH_CALUDE_alster_frogs_l2210_221040

theorem alster_frogs (alster quinn bret : ℕ) 
  (h1 : quinn = 2 * alster)
  (h2 : bret = 3 * quinn)
  (h3 : bret = 12) :
  alster = 2 := by
sorry

end NUMINAMATH_CALUDE_alster_frogs_l2210_221040


namespace NUMINAMATH_CALUDE_trevors_age_l2210_221016

theorem trevors_age (T : ℕ) : 
  (20 + (24 - T) = 3 * T) → T = 11 := by
  sorry

end NUMINAMATH_CALUDE_trevors_age_l2210_221016


namespace NUMINAMATH_CALUDE_integer_subset_condition_l2210_221090

theorem integer_subset_condition (a b : ℤ) : 
  (a * b * (a - b) ≠ 0) →
  (∃ (Z₀ : Set ℤ), ∀ (n : ℤ), (n ∈ Z₀ ∨ (n + a) ∈ Z₀ ∨ (n + b) ∈ Z₀) ∧ 
    ¬(n ∈ Z₀ ∧ (n + a) ∈ Z₀) ∧ ¬(n ∈ Z₀ ∧ (n + b) ∈ Z₀) ∧ ¬((n + a) ∈ Z₀ ∧ (n + b) ∈ Z₀)) ↔
  (∃ (k y z : ℤ), a = k * y ∧ b = k * z ∧ y % 3 ≠ 0 ∧ z % 3 ≠ 0 ∧ (y - z) % 3 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_integer_subset_condition_l2210_221090


namespace NUMINAMATH_CALUDE_ways_to_top_teaching_building_l2210_221001

/-- A building with multiple floors and staircases -/
structure Building where
  floors : ℕ
  staircases_per_floor : ℕ

/-- The number of ways to go from the bottom floor to the top floor -/
def ways_to_top (b : Building) : ℕ :=
  b.staircases_per_floor ^ (b.floors - 1)

/-- The specific building in the problem -/
def teaching_building : Building :=
  { floors := 5, staircases_per_floor := 2 }

theorem ways_to_top_teaching_building :
  ways_to_top teaching_building = 2^4 := by
  sorry

#eval ways_to_top teaching_building

end NUMINAMATH_CALUDE_ways_to_top_teaching_building_l2210_221001


namespace NUMINAMATH_CALUDE_triangle_formation_l2210_221025

-- Define a function to check if three lengths can form a triangle
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Theorem statement
theorem triangle_formation :
  can_form_triangle 4 4 7 ∧
  ¬ can_form_triangle 2 3 6 ∧
  ¬ can_form_triangle 3 4 8 ∧
  ¬ can_form_triangle 5 8 13 :=
by sorry

end NUMINAMATH_CALUDE_triangle_formation_l2210_221025


namespace NUMINAMATH_CALUDE_negation_equivalence_l2210_221028

theorem negation_equivalence (a b : ℝ) : 
  (¬(a + b = 1 → a^2 + b^2 > 1)) ↔ (∃ a b : ℝ, a + b = 1 ∧ a^2 + b^2 ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2210_221028


namespace NUMINAMATH_CALUDE_simplify_fraction_expression_l2210_221064

theorem simplify_fraction_expression :
  1 / (1 / (1/3)^1 + 1 / (1/3)^2 + 1 / (1/3)^3) = 1 / 39 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_expression_l2210_221064


namespace NUMINAMATH_CALUDE_employment_percentage_l2210_221009

theorem employment_percentage (total_population : ℝ) 
  (employed_males_percentage : ℝ) (employed_females_percentage : ℝ) :
  employed_males_percentage = 48 →
  employed_females_percentage = 20 →
  (employed_males_percentage / (100 - employed_females_percentage)) * 100 = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_employment_percentage_l2210_221009


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2210_221073

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n ∧ a n > 0

theorem geometric_sequence_property (a : ℕ → ℝ) (h : GeometricSequence a) 
    (h1 : a 4 * a 8 = 4) : a 5 * a 6 * a 7 = 8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2210_221073


namespace NUMINAMATH_CALUDE_temperature_rise_l2210_221011

theorem temperature_rise (initial_temp final_temp rise : ℤ) : 
  initial_temp = -2 → rise = 3 → final_temp = initial_temp + rise → final_temp = 1 :=
by sorry

end NUMINAMATH_CALUDE_temperature_rise_l2210_221011


namespace NUMINAMATH_CALUDE_power_two_greater_than_sum_of_powers_l2210_221033

theorem power_two_greater_than_sum_of_powers (n : ℕ) (x : ℝ) 
  (h1 : n ≥ 2) (h2 : |x| < 1) : 
  2^n > (1 - x)^n + (1 + x)^n := by
  sorry

end NUMINAMATH_CALUDE_power_two_greater_than_sum_of_powers_l2210_221033


namespace NUMINAMATH_CALUDE_nancy_mexican_antacids_l2210_221010

/-- Represents the number of antacids Nancy takes per day when eating Mexican food -/
def mexican_antacids : ℕ := sorry

/-- Represents the number of antacids Nancy takes per day when eating Indian food -/
def indian_antacids : ℕ := 3

/-- Represents the number of antacids Nancy takes per day when eating other food -/
def other_antacids : ℕ := 1

/-- Represents the number of times Nancy eats Indian food per week -/
def indian_meals_per_week : ℕ := 3

/-- Represents the number of times Nancy eats Mexican food per week -/
def mexican_meals_per_week : ℕ := 2

/-- Represents the number of antacids Nancy takes per month -/
def antacids_per_month : ℕ := 60

/-- Represents the number of weeks in a month (approximated) -/
def weeks_per_month : ℕ := 4

theorem nancy_mexican_antacids : 
  mexican_antacids = 2 :=
by sorry

end NUMINAMATH_CALUDE_nancy_mexican_antacids_l2210_221010


namespace NUMINAMATH_CALUDE_sum_factorial_units_digit_2023_l2210_221041

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def unitsDigit (n : ℕ) : ℕ := n % 10

def factorialUnitsDigit (n : ℕ) : ℕ :=
  if n > 4 then 0 else unitsDigit (factorial n)

def sumFactorialUnitsDigits (n : ℕ) : ℕ :=
  (List.range n).foldl (λ acc i => (acc + factorialUnitsDigit (i + 1)) % 10) 0

theorem sum_factorial_units_digit_2023 :
  sumFactorialUnitsDigits 2023 = 3 := by sorry

end NUMINAMATH_CALUDE_sum_factorial_units_digit_2023_l2210_221041


namespace NUMINAMATH_CALUDE_number_equation_solution_l2210_221070

theorem number_equation_solution : ∃ x : ℝ, (0.75 * x + 2 = 8) ∧ (x = 8) := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l2210_221070


namespace NUMINAMATH_CALUDE_arbelos_external_tangent_l2210_221099

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents an arbelos configuration -/
structure Arbelos where
  A : Point
  B : Point
  C : Point
  D : Point
  M : Point
  N : Point
  O₁ : Point
  O₂ : Point
  smallCircle1 : Circle
  smallCircle2 : Circle
  largeCircle : Circle

/-- Checks if a line is tangent to a circle -/
def isTangent (p1 p2 : Point) (c : Circle) : Prop :=
  sorry

/-- Angle between three points -/
def angle (p1 p2 p3 : Point) : ℝ :=
  sorry

/-- Main theorem: MN is a common external tangent to the small circles of the arbelos -/
theorem arbelos_external_tangent (arb : Arbelos) (α : ℝ) 
    (h1 : angle arb.B arb.A arb.D = α)
    (h2 : arb.smallCircle1.center = arb.O₁)
    (h3 : arb.smallCircle2.center = arb.O₂) :
  isTangent arb.M arb.N arb.smallCircle1 ∧ isTangent arb.M arb.N arb.smallCircle2 :=
by sorry

end NUMINAMATH_CALUDE_arbelos_external_tangent_l2210_221099


namespace NUMINAMATH_CALUDE_smallest_winning_points_l2210_221018

/-- Represents the possible placings in a race -/
inductive Placing
| First
| Second
| Third
| Other

/-- Calculates the points for a given placing -/
def points_for_placing (p : Placing) : ℕ :=
  match p with
  | Placing.First => 7
  | Placing.Second => 4
  | Placing.Third => 2
  | Placing.Other => 0

/-- Calculates the total points for a list of placings -/
def total_points (placings : List Placing) : ℕ :=
  placings.map points_for_placing |>.sum

/-- Represents the results of four races -/
def RaceResults := List Placing

/-- Checks if a given point total guarantees winning -/
def guarantees_win (points : ℕ) : Prop :=
  ∀ (other_results : RaceResults), 
    other_results.length = 4 → total_points other_results < points

theorem smallest_winning_points : 
  (guarantees_win 25) ∧ (∀ p : ℕ, p < 25 → ¬guarantees_win p) := by
  sorry

end NUMINAMATH_CALUDE_smallest_winning_points_l2210_221018


namespace NUMINAMATH_CALUDE_sqrt_expression_value_l2210_221092

theorem sqrt_expression_value (x y : ℝ) 
  (h : Real.sqrt (x + 5) + (2 * x - y)^2 = 0) : 
  Real.sqrt (x^2 - 2*x*y + y^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_value_l2210_221092


namespace NUMINAMATH_CALUDE_baking_distribution_problem_l2210_221068

/-- Calculates the number of leftover items when distributing a total number of items into containers of a specific capacity -/
def leftovers (total : ℕ) (capacity : ℕ) : ℕ :=
  total % capacity

/-- Represents the baking and distribution problem -/
theorem baking_distribution_problem 
  (gingerbread_batches : ℕ) (gingerbread_per_batch : ℕ) (gingerbread_per_jar : ℕ)
  (sugar_batches : ℕ) (sugar_per_batch : ℕ) (sugar_per_box : ℕ)
  (tart_batches : ℕ) (tarts_per_batch : ℕ) (tarts_per_box : ℕ)
  (h_gingerbread : gingerbread_batches = 3 ∧ gingerbread_per_batch = 47 ∧ gingerbread_per_jar = 6)
  (h_sugar : sugar_batches = 2 ∧ sugar_per_batch = 78 ∧ sugar_per_box = 9)
  (h_tart : tart_batches = 4 ∧ tarts_per_batch = 36 ∧ tarts_per_box = 4) :
  leftovers (gingerbread_batches * gingerbread_per_batch) gingerbread_per_jar = 3 ∧
  leftovers (sugar_batches * sugar_per_batch) sugar_per_box = 3 ∧
  leftovers (tart_batches * tarts_per_batch) tarts_per_box = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_baking_distribution_problem_l2210_221068


namespace NUMINAMATH_CALUDE_rockets_win_in_7_l2210_221065

/-- Probability of Warriors winning a single game -/
def p_warriors : ℚ := 3/4

/-- Probability of Rockets winning a single game -/
def p_rockets : ℚ := 1 - p_warriors

/-- Number of games needed to win the series -/
def games_to_win : ℕ := 4

/-- Maximum number of games in the series -/
def max_games : ℕ := 7

/-- Probability of Rockets winning the series in exactly 7 games -/
def p_rockets_win_in_7 : ℚ := 135/4096

theorem rockets_win_in_7 :
  p_rockets_win_in_7 = (Nat.choose 6 3 : ℚ) * p_rockets^3 * p_warriors^3 * p_rockets :=
by sorry

end NUMINAMATH_CALUDE_rockets_win_in_7_l2210_221065


namespace NUMINAMATH_CALUDE_second_discount_percentage_l2210_221078

def initial_price : ℝ := 400
def first_discount : ℝ := 20
def final_price : ℝ := 272

theorem second_discount_percentage :
  ∃ (second_discount : ℝ),
    initial_price * (1 - first_discount / 100) * (1 - second_discount / 100) = final_price ∧
    second_discount = 15 := by
  sorry

end NUMINAMATH_CALUDE_second_discount_percentage_l2210_221078


namespace NUMINAMATH_CALUDE_product_of_integers_l2210_221081

theorem product_of_integers (x y : ℕ+) 
  (sum_eq : x + y = 18) 
  (diff_squares_eq : x^2 - y^2 = 36) : 
  x * y = 80 := by
  sorry

end NUMINAMATH_CALUDE_product_of_integers_l2210_221081


namespace NUMINAMATH_CALUDE_max_symmetry_axes_is_2k_l2210_221071

/-- The maximum number of axes of symmetry for the union of k line segments on a plane -/
def max_symmetry_axes (k : ℕ) : ℕ := 2 * k

/-- Theorem: The maximum number of axes of symmetry for the union of k line segments on a plane is 2k -/
theorem max_symmetry_axes_is_2k (k : ℕ) :
  max_symmetry_axes k = 2 * k :=
by sorry

end NUMINAMATH_CALUDE_max_symmetry_axes_is_2k_l2210_221071


namespace NUMINAMATH_CALUDE_tank_fill_time_l2210_221015

/-- Represents a pipe with a flow rate (positive for filling, negative for draining) -/
structure Pipe where
  rate : Int

/-- Represents a tank with a capacity and a list of pipes -/
structure Tank where
  capacity : Nat
  pipes : List Pipe

def cycleTime : Nat := 3

def cycleVolume (tank : Tank) : Int :=
  tank.pipes.foldl (fun acc pipe => acc + pipe.rate) 0

theorem tank_fill_time (tank : Tank) (h1 : tank.capacity = 750)
    (h2 : tank.pipes = [⟨40⟩, ⟨30⟩, ⟨-20⟩])
    (h3 : cycleVolume tank = 50)
    (h4 : tank.capacity / cycleVolume tank * cycleTime = 45) :
  ∃ (t : Nat), t = 45 ∧ t * cycleVolume tank ≥ tank.capacity := by
  sorry

end NUMINAMATH_CALUDE_tank_fill_time_l2210_221015


namespace NUMINAMATH_CALUDE_divisibility_by_power_of_two_l2210_221084

theorem divisibility_by_power_of_two (a b : ℕ) (ha : Odd a) (hb : Odd b) :
  ∃ k : ℕ, (2^2018 ∣ b^k - a^2) ∨ (2^2018 ∣ a^k - b^2) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_power_of_two_l2210_221084


namespace NUMINAMATH_CALUDE_circle_C_equation_l2210_221083

/-- A circle C with center on the x-axis passing through points A(-1,1) and B(1,3) -/
structure CircleC where
  center : ℝ × ℝ
  center_on_x_axis : center.2 = 0
  passes_through_A : (center.1 + 1)^2 + (center.2 - 1)^2 = (center.1 - 1)^2 + (center.2 - 3)^2

/-- The equation of circle C is (x-2)²+y²=10 -/
theorem circle_C_equation (C : CircleC) :
  ∀ (x y : ℝ), (x - 2)^2 + y^2 = 10 ↔ (x - C.center.1)^2 + (y - C.center.2)^2 = (C.center.1 + 1)^2 + (C.center.2 - 1)^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_C_equation_l2210_221083


namespace NUMINAMATH_CALUDE_abs_diff_opposite_l2210_221066

theorem abs_diff_opposite (x : ℝ) (h : x < 0) : |x - (-x)| = -2*x := by
  sorry

end NUMINAMATH_CALUDE_abs_diff_opposite_l2210_221066


namespace NUMINAMATH_CALUDE_petya_win_probability_l2210_221076

/-- The game "Pile of Stones" --/
structure PileOfStones where
  initialStones : Nat
  minTake : Nat
  maxTake : Nat

/-- The optimal strategy for the game --/
def optimalStrategy (game : PileOfStones) : Nat → Nat :=
  sorry

/-- The probability of winning when playing randomly --/
def randomWinProbability (game : PileOfStones) : ℚ :=
  sorry

/-- The theorem stating the probability of Petya winning --/
theorem petya_win_probability :
  let game : PileOfStones := {
    initialStones := 16,
    minTake := 1,
    maxTake := 4
  }
  randomWinProbability game = 1 / 256 := by
  sorry

end NUMINAMATH_CALUDE_petya_win_probability_l2210_221076


namespace NUMINAMATH_CALUDE_special_pizza_all_toppings_l2210_221044

/-- Represents a pizza with various toppings -/
structure Pizza where
  total_slices : ℕ
  pepperoni_slices : ℕ
  mushroom_slices : ℕ
  olive_slices : ℕ
  all_toppings_slices : ℕ

/-- Conditions for our specific pizza -/
def special_pizza : Pizza := {
  total_slices := 24,
  pepperoni_slices := 15,
  mushroom_slices := 16,
  olive_slices := 10,
  all_toppings_slices := 2
}

/-- Every slice has at least one topping -/
def has_at_least_one_topping (p : Pizza) : Prop :=
  p.pepperoni_slices + p.mushroom_slices + p.olive_slices - p.all_toppings_slices ≥ p.total_slices

/-- The theorem to prove -/
theorem special_pizza_all_toppings :
  has_at_least_one_topping special_pizza ∧
  special_pizza.all_toppings_slices = 2 :=
sorry


end NUMINAMATH_CALUDE_special_pizza_all_toppings_l2210_221044


namespace NUMINAMATH_CALUDE_equation_solution_l2210_221069

theorem equation_solution :
  ∃ x : ℝ, (x + 6) / (x - 3) = 4 ∧ x = 6 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2210_221069


namespace NUMINAMATH_CALUDE_min_value_on_circle_l2210_221087

theorem min_value_on_circle (x y : ℝ) (h : (x - 2)^2 + (y - 1)^2 = 1) :
  ∃ (min : ℝ), (∀ (a b : ℝ), (a - 2)^2 + (b - 1)^2 = 1 → a^2 + b^2 ≥ min) ∧ min = 6 - 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_on_circle_l2210_221087


namespace NUMINAMATH_CALUDE_first_hour_coins_is_20_l2210_221077

/-- The number of coins Tina put in the jar during the first hour -/
def first_hour_coins : ℕ := sorry

/-- The number of coins Tina put in the jar during the second hour -/
def second_hour_coins : ℕ := 30

/-- The number of coins Tina put in the jar during the third hour -/
def third_hour_coins : ℕ := 30

/-- The number of coins Tina put in the jar during the fourth hour -/
def fourth_hour_coins : ℕ := 40

/-- The number of coins Tina took out of the jar during the fifth hour -/
def fifth_hour_coins : ℕ := 20

/-- The total number of coins in the jar after the fifth hour -/
def total_coins : ℕ := 100

/-- Theorem stating that the number of coins Tina put in during the first hour is 20 -/
theorem first_hour_coins_is_20 :
  first_hour_coins = 20 :=
by
  have h : first_hour_coins + second_hour_coins + third_hour_coins + fourth_hour_coins - fifth_hour_coins = total_coins := sorry
  sorry


end NUMINAMATH_CALUDE_first_hour_coins_is_20_l2210_221077


namespace NUMINAMATH_CALUDE_rational_roots_of_p_l2210_221086

def p (x : ℚ) : ℚ := x^4 - 3*x^3 - 8*x^2 + 12*x + 16

theorem rational_roots_of_p :
  {x : ℚ | p x = 0} = {-1, -2, 2, 4} := by sorry

end NUMINAMATH_CALUDE_rational_roots_of_p_l2210_221086


namespace NUMINAMATH_CALUDE_prime_square_sum_not_perfect_square_l2210_221072

theorem prime_square_sum_not_perfect_square
  (p q : ℕ) (hp : Prime p) (hq : Prime q)
  (h_perfect_square : ∃ a : ℕ, a > 0 ∧ p + q^2 = a^2) :
  ∀ n : ℕ, n > 0 → ¬∃ b : ℕ, b > 0 ∧ p^2 + q^n = b^2 :=
by sorry

end NUMINAMATH_CALUDE_prime_square_sum_not_perfect_square_l2210_221072


namespace NUMINAMATH_CALUDE_parabola_intersection_theorem_l2210_221017

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through the focus
def line_through_focus (m : ℝ) (x y : ℝ) : Prop := x = m*y + 1

-- Define the intersection points
def intersection_points (m : ℝ) : ℝ × ℝ × ℝ × ℝ := sorry

-- Define the condition |MD| = 2|NF|
def length_condition (M N : ℝ × ℝ) : Prop := 
  let (x₁, y₁) := M
  let (x₂, y₂) := N
  x₁ = 2*x₂ + 2

-- Main theorem
theorem parabola_intersection_theorem (m : ℝ) : 
  let (x₁, y₁, x₂, y₂) := intersection_points m
  let M := (x₁, y₁)
  let N := (x₂, y₂)
  parabola x₁ y₁ ∧ 
  parabola x₂ y₂ ∧
  line_through_focus m x₁ y₁ ∧
  line_through_focus m x₂ y₂ ∧
  length_condition M N →
  Real.sqrt ((x₁ - 1)^2 + y₁^2) = 2 + Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_parabola_intersection_theorem_l2210_221017


namespace NUMINAMATH_CALUDE_jerry_money_duration_l2210_221038

/-- The number of weeks Jerry's money will last -/
def weeks_money_lasts (lawn_mowing_earnings weed_eating_earnings weekly_spending : ℕ) : ℕ :=
  (lawn_mowing_earnings + weed_eating_earnings) / weekly_spending

/-- Theorem: Given Jerry's earnings and weekly spending, his money will last 9 weeks -/
theorem jerry_money_duration :
  weeks_money_lasts 14 31 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_jerry_money_duration_l2210_221038


namespace NUMINAMATH_CALUDE_percentage_of_defective_meters_l2210_221027

theorem percentage_of_defective_meters
  (total_meters : ℕ)
  (rejected_meters : ℕ)
  (h1 : total_meters = 8000)
  (h2 : rejected_meters = 4) :
  (rejected_meters : ℝ) / (total_meters : ℝ) * 100 = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_defective_meters_l2210_221027


namespace NUMINAMATH_CALUDE_octagon_diagonals_l2210_221042

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- An octagon has 8 sides -/
def octagon_sides : ℕ := 8

theorem octagon_diagonals : num_diagonals octagon_sides = 20 := by
  sorry

end NUMINAMATH_CALUDE_octagon_diagonals_l2210_221042


namespace NUMINAMATH_CALUDE_circle_in_second_quadrant_implies_a_range_l2210_221031

/-- Definition of the circle equation -/
def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 + 6*x - 4*a*y + 3*a^2 + 9 = 0

/-- Definition of a point being in the second quadrant -/
def in_second_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

/-- Theorem stating that if all points on the circle are in the second quadrant,
    then a is between 0 and 3 -/
theorem circle_in_second_quadrant_implies_a_range :
  (∀ x y : ℝ, circle_equation x y a → in_second_quadrant x y) →
  0 < a ∧ a < 3 :=
sorry

end NUMINAMATH_CALUDE_circle_in_second_quadrant_implies_a_range_l2210_221031


namespace NUMINAMATH_CALUDE_laura_shopping_cost_l2210_221030

/-- Calculates the total cost of Laura's shopping trip given the prices and quantities of items. -/
def shopping_cost (salad_price : ℚ) (juice_price : ℚ) : ℚ :=
  let beef_price := 2 * salad_price
  let potato_price := salad_price / 3
  let mixed_veg_price := beef_price / 2 + 0.5
  let tomato_sauce_price := salad_price * 3 / 4
  let pasta_price := juice_price + mixed_veg_price
  2 * salad_price +
  2 * beef_price +
  1 * potato_price +
  2 * juice_price +
  3 * mixed_veg_price +
  5 * tomato_sauce_price +
  4 * pasta_price

theorem laura_shopping_cost :
  shopping_cost 3 1.5 = 63.75 := by
  sorry

end NUMINAMATH_CALUDE_laura_shopping_cost_l2210_221030


namespace NUMINAMATH_CALUDE_sin_equality_problem_l2210_221088

theorem sin_equality_problem (m : ℤ) (h1 : -90 ≤ m) (h2 : m ≤ 90) :
  Real.sin (m * Real.pi / 180) = Real.sin (710 * Real.pi / 180) → m = -10 := by
  sorry

end NUMINAMATH_CALUDE_sin_equality_problem_l2210_221088


namespace NUMINAMATH_CALUDE_horner_v1_equals_22_l2210_221075

/-- Horner's Method for polynomial evaluation -/
def horner_step (coeff : ℝ) (x : ℝ) (prev : ℝ) : ℝ :=
  prev * x + coeff

/-- The polynomial f(x) = 4x⁵ + 2x⁴ + 3.5x³ - 2.6x² + 1.7x - 0.8 -/
def f (x : ℝ) : ℝ :=
  4 * x^5 + 2 * x^4 + 3.5 * x^3 - 2.6 * x^2 + 1.7 * x - 0.8

/-- Theorem: The value of V₁ when calculating f(5) using Horner's Method is 22 -/
theorem horner_v1_equals_22 :
  let v0 := 4  -- Initialize V₀ with the coefficient of the highest degree term
  let v1 := horner_step 2 5 v0  -- Calculate V₁
  v1 = 22 := by sorry

end NUMINAMATH_CALUDE_horner_v1_equals_22_l2210_221075


namespace NUMINAMATH_CALUDE_direct_proportion_point_value_l2210_221055

/-- A directly proportional function passing through points (-2, 3) and (a, -3) has a = 2 -/
theorem direct_proportion_point_value (k a : ℝ) : 
  (∃ k : ℝ, k * (-2) = 3 ∧ k * a = -3) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_direct_proportion_point_value_l2210_221055


namespace NUMINAMATH_CALUDE_wage_increase_for_unit_productivity_increase_l2210_221093

/-- Regression line equation for workers' wages as a function of labor productivity -/
def regression_line (x : ℝ) : ℝ := 80 * x + 50

/-- Theorem: The average increase in wage when labor productivity increases by 1 unit -/
theorem wage_increase_for_unit_productivity_increase :
  ∀ x : ℝ, regression_line (x + 1) - regression_line x = 80 := by
sorry

end NUMINAMATH_CALUDE_wage_increase_for_unit_productivity_increase_l2210_221093


namespace NUMINAMATH_CALUDE_claires_remaining_balance_l2210_221004

/-- Calculates the remaining balance on Claire's gift card after a week of purchases --/
def remaining_balance (gift_card latte_price croissant_price bagel_price holiday_drink_price cookie_price : ℚ)
  (days bagel_occasions cookies : ℕ) : ℚ :=
  let daily_total := latte_price + croissant_price
  let weekly_total := daily_total * days
  let bagel_total := bagel_price * bagel_occasions
  let friday_treats := holiday_drink_price + cookie_price * cookies
  let friday_adjustment := friday_treats - latte_price
  let total_expenses := weekly_total + bagel_total + friday_adjustment
  gift_card - total_expenses

/-- Theorem stating that Claire's remaining balance is $35.50 --/
theorem claires_remaining_balance :
  remaining_balance 100 3.75 3.50 2.25 4.50 1.25 7 3 5 = 35.50 := by
  sorry

end NUMINAMATH_CALUDE_claires_remaining_balance_l2210_221004


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l2210_221054

theorem diophantine_equation_solution : ∃ (a b c d : ℕ+), 
  (a^3 + b^4 + c^5 = d^11) ∧ (a * b * c < 10^5) := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l2210_221054


namespace NUMINAMATH_CALUDE_parallel_vectors_l2210_221060

/-- Given two vectors a and b in R², prove that ka + b is parallel to a - 3b iff k = -1/3 -/
theorem parallel_vectors (a b : Fin 2 → ℝ) (h1 : a 0 = 1) (h2 : a 1 = 2) (h3 : b 0 = -3) (h4 : b 1 = 2) :
  (∃ k : ℝ, ∀ i : Fin 2, k * (a i) + (b i) = c * ((a i) - 3 * (b i)) ∧ c ≠ 0) ↔ k = -1/3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_l2210_221060


namespace NUMINAMATH_CALUDE_line_moved_down_l2210_221037

/-- Given a line with equation y = -3x + 5, prove that moving it down 3 units
    results in the line with equation y = -3x + 2 -/
theorem line_moved_down (x y : ℝ) :
  (y = -3 * x + 5) → (y - 3 = -3 * x + 2) := by sorry

end NUMINAMATH_CALUDE_line_moved_down_l2210_221037


namespace NUMINAMATH_CALUDE_dye_per_dot_l2210_221024

/-- The amount of dye per dot given the number of dots per blouse, 
    total amount of dye, and number of blouses -/
theorem dye_per_dot 
  (dots_per_blouse : ℕ) 
  (total_dye : ℕ) 
  (num_blouses : ℕ) 
  (h1 : dots_per_blouse = 20)
  (h2 : total_dye = 50 * 400)
  (h3 : num_blouses = 100) :
  total_dye / (dots_per_blouse * num_blouses) = 10 := by
  sorry

#check dye_per_dot

end NUMINAMATH_CALUDE_dye_per_dot_l2210_221024


namespace NUMINAMATH_CALUDE_all_pairs_product_48_l2210_221043

theorem all_pairs_product_48 : 
  ((-6) * (-8) = 48) ∧
  ((-4) * (-12) = 48) ∧
  ((3/2 : ℚ) * 32 = 48) ∧
  (2 * 24 = 48) ∧
  ((4/3 : ℚ) * 36 = 48) := by
  sorry

end NUMINAMATH_CALUDE_all_pairs_product_48_l2210_221043


namespace NUMINAMATH_CALUDE_adam_apple_purchase_l2210_221034

/-- The quantity of apples Adam bought on Monday -/
def monday_apples : ℕ := 15

/-- The quantity of apples Adam bought on Tuesday -/
def tuesday_apples : ℕ := 3 * monday_apples

/-- The quantity of apples Adam bought on Wednesday -/
def wednesday_apples : ℕ := 4 * tuesday_apples

/-- The total quantity of apples Adam bought on these three days -/
def total_apples : ℕ := monday_apples + tuesday_apples + wednesday_apples

theorem adam_apple_purchase : total_apples = 240 := by
  sorry

end NUMINAMATH_CALUDE_adam_apple_purchase_l2210_221034


namespace NUMINAMATH_CALUDE_average_age_when_youngest_born_l2210_221063

/-- Proves that for a group of 7 people with an average age of 30 and the youngest being 8 years old,
    the average age of the group when the youngest was born was 22 years. -/
theorem average_age_when_youngest_born
  (num_people : ℕ)
  (current_average_age : ℝ)
  (youngest_age : ℕ)
  (h_num_people : num_people = 7)
  (h_current_average : current_average_age = 30)
  (h_youngest : youngest_age = 8) :
  (num_people * current_average_age - num_people * youngest_age) / num_people = 22 :=
sorry

end NUMINAMATH_CALUDE_average_age_when_youngest_born_l2210_221063


namespace NUMINAMATH_CALUDE_regular_triangular_pyramid_properties_l2210_221049

structure RegularTriangularPyramid where
  PA : ℝ
  θ : ℝ

def distance_to_base (p : RegularTriangularPyramid) : ℝ :=
  sorry

def surface_area (p : RegularTriangularPyramid) : ℝ :=
  sorry

theorem regular_triangular_pyramid_properties
  (p : RegularTriangularPyramid)
  (h1 : p.PA = 2)
  (h2 : 0 < p.θ ∧ p.θ ≤ π / 2) :
  (distance_to_base { PA := 2, θ := π / 2 } = 2 * Real.sqrt 3 / 3) ∧
  (∀ θ₁ θ₂, 0 < θ₁ ∧ θ₁ < θ₂ ∧ θ₂ ≤ π / 2 →
    surface_area { PA := 2, θ := θ₁ } < surface_area { PA := 2, θ := θ₂ }) :=
sorry

end NUMINAMATH_CALUDE_regular_triangular_pyramid_properties_l2210_221049


namespace NUMINAMATH_CALUDE_range_of_f_l2210_221003

-- Define the function f
def f (x : ℝ) : ℝ := |x + 3| - |x - 5|

-- State the theorem about the range of f
theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ -8 ≤ y ∧ y ≤ 8 :=
by sorry

end NUMINAMATH_CALUDE_range_of_f_l2210_221003


namespace NUMINAMATH_CALUDE_at_least_one_nonnegative_l2210_221079

theorem at_least_one_nonnegative (a b c d e f g h : ℝ) :
  (ac + bd ≥ 0) ∨ (ae + bf ≥ 0) ∨ (ag + bh ≥ 0) ∨ 
  (ce + df ≥ 0) ∨ (cg + dh ≥ 0) ∨ (eg + fh ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_at_least_one_nonnegative_l2210_221079


namespace NUMINAMATH_CALUDE_polynomial_uniqueness_l2210_221022

/-- Given a polynomial P(x) = P(0) + P(1)x + P(2)x^2 where P(-2) = 4,
    prove that P(x) = (4x^2 - 6x) / 7 -/
theorem polynomial_uniqueness (P : ℝ → ℝ) (h1 : ∀ x, P x = P 0 + P 1 * x + P 2 * x^2) 
    (h2 : P (-2) = 4) : 
  ∀ x, P x = (4 * x^2 - 6 * x) / 7 := by
sorry

end NUMINAMATH_CALUDE_polynomial_uniqueness_l2210_221022


namespace NUMINAMATH_CALUDE_figure_50_squares_l2210_221052

/-- The number of nonoverlapping unit squares in figure n -/
def g (n : ℕ) : ℕ := 2 * n^2 + 4 * n + 2

/-- The sequence of nonoverlapping unit squares follows the pattern -/
axiom pattern_holds : g 0 = 2 ∧ g 1 = 8 ∧ g 2 = 18 ∧ g 3 = 32

theorem figure_50_squares : g 50 = 5202 := by sorry

end NUMINAMATH_CALUDE_figure_50_squares_l2210_221052


namespace NUMINAMATH_CALUDE_divisor_cube_eq_four_n_l2210_221046

/-- The number of positive divisors of a natural number -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- The set of solutions to the equation d(n)^3 = 4n -/
def solution_set : Set ℕ := {2, 128, 2000}

/-- Theorem stating that n is a solution if and only if it's in the solution set -/
theorem divisor_cube_eq_four_n (n : ℕ) : 
  (num_divisors n)^3 = 4 * n ↔ n ∈ solution_set := by sorry

end NUMINAMATH_CALUDE_divisor_cube_eq_four_n_l2210_221046


namespace NUMINAMATH_CALUDE_gcd_polynomial_and_multiple_l2210_221039

theorem gcd_polynomial_and_multiple (x : ℤ) : 
  (∃ k : ℤ, x = 32515 * k) →
  Int.gcd ((3*x+5)*(5*x+3)*(11*x+7)*(x+17)) x = 35 := by
  sorry

end NUMINAMATH_CALUDE_gcd_polynomial_and_multiple_l2210_221039


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_l2210_221000

theorem rectangular_prism_volume (l w h : ℝ) : 
  l > 0 → w > 0 → h > 0 →
  l * w = 15 → w * h = 10 → l * h = 6 →
  l * w * h = 30 := by
sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_l2210_221000


namespace NUMINAMATH_CALUDE_odd_function_negative_x_l2210_221089

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_negative_x
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_pos : ∀ x > 0, f x = 2 * x - 3) :
  ∀ x < 0, f x = 2 * x + 3 :=
by sorry

end NUMINAMATH_CALUDE_odd_function_negative_x_l2210_221089


namespace NUMINAMATH_CALUDE_carries_mom_payment_ratio_l2210_221050

/-- The ratio of the amount Carrie's mom pays to the total cost of all clothes -/
theorem carries_mom_payment_ratio :
  let shirt_count : ℕ := 4
  let pants_count : ℕ := 2
  let jacket_count : ℕ := 2
  let shirt_price : ℕ := 8
  let pants_price : ℕ := 18
  let jacket_price : ℕ := 60
  let carries_payment : ℕ := 94
  let total_cost : ℕ := shirt_count * shirt_price + pants_count * pants_price + jacket_count * jacket_price
  let moms_payment : ℕ := total_cost - carries_payment
  moms_payment * 2 = total_cost :=
by sorry

end NUMINAMATH_CALUDE_carries_mom_payment_ratio_l2210_221050


namespace NUMINAMATH_CALUDE_pythagorean_triple_check_l2210_221012

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

theorem pythagorean_triple_check : 
  is_pythagorean_triple 6 8 10 ∧
  ¬ is_pythagorean_triple 3 4 5 ∧
  ¬ is_pythagorean_triple 5 11 12 :=
by sorry

end NUMINAMATH_CALUDE_pythagorean_triple_check_l2210_221012


namespace NUMINAMATH_CALUDE_equation_solution_l2210_221080

theorem equation_solution : 
  ∃ x : ℝ, (3*x - 5) / (x^2 - 7*x + 12) + (5*x - 1) / (x^2 - 5*x + 6) = (8*x - 13) / (x^2 - 6*x + 8) ∧ x = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2210_221080


namespace NUMINAMATH_CALUDE_simplify_expression_l2210_221002

theorem simplify_expression (r : ℝ) : 150 * r - 70 * r + 25 = 80 * r + 25 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2210_221002


namespace NUMINAMATH_CALUDE_machine_work_time_equation_l2210_221048

theorem machine_work_time_equation (x : ℝ) (hx : x > 0) : 
  (1 / (x + 6) + 1 / (x + 1) + 1 / (2 * x) = 1 / x) → x = 2 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_machine_work_time_equation_l2210_221048


namespace NUMINAMATH_CALUDE_arithmetic_sequence_inequality_l2210_221008

theorem arithmetic_sequence_inequality (a : ℕ → ℝ) (d : ℝ) :
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence definition
  0 < a 1 → a 1 < a 2 →
  a 2 > Real.sqrt (a 1 * a 3) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_inequality_l2210_221008


namespace NUMINAMATH_CALUDE_cylinder_packing_l2210_221085

theorem cylinder_packing (n : ℕ) (d : ℝ) (h : d > 0) :
  let rectangular_width := 8 * d
  let hexagonal_width := n * d * (Real.sqrt 3 / 2) + d
  40 < n → n < 42 →
  hexagonal_width < rectangular_width ∧
  hexagonal_width > rectangular_width - d :=
by sorry

end NUMINAMATH_CALUDE_cylinder_packing_l2210_221085


namespace NUMINAMATH_CALUDE_complement_M_intersect_N_l2210_221097

-- Define the sets M and N
def M : Set ℝ := {x | 2/x < 1}
def N : Set ℝ := {y | ∃ x, y = Real.sqrt (x - 1)}

-- State the theorem
theorem complement_M_intersect_N :
  (Set.univ \ M) ∩ N = Set.Icc 0 2 := by sorry

end NUMINAMATH_CALUDE_complement_M_intersect_N_l2210_221097


namespace NUMINAMATH_CALUDE_paper_clip_distribution_l2210_221057

theorem paper_clip_distribution (total_clips : ℕ) (clips_per_box : ℕ) (boxes : ℕ) : 
  total_clips = 81 → 
  clips_per_box = 9 → 
  total_clips = boxes * clips_per_box → 
  boxes = 9 := by
sorry

end NUMINAMATH_CALUDE_paper_clip_distribution_l2210_221057


namespace NUMINAMATH_CALUDE_acute_triangle_sine_cosine_inequality_l2210_221056

theorem acute_triangle_sine_cosine_inequality 
  (A B C : Real) 
  (h_acute : A ∈ Set.Ioo 0 (π/2) ∧ B ∈ Set.Ioo 0 (π/2) ∧ C ∈ Set.Ioo 0 (π/2)) 
  (h_sum : A + B + C = π) : 
  Real.sin A + Real.sin B > Real.cos A + Real.cos B + Real.cos C := by
  sorry

end NUMINAMATH_CALUDE_acute_triangle_sine_cosine_inequality_l2210_221056


namespace NUMINAMATH_CALUDE_time_equation_l2210_221061

/-- Given the equations V = 2gt + V₀ and S = (1/3)gt² + V₀t + Ct³, where C is a constant,
    prove that the time t can be expressed as t = (V - V₀) / (2g). -/
theorem time_equation (g V V₀ S t : ℝ) (C : ℝ) :
  V = 2 * g * t + V₀ ∧ S = (1/3) * g * t^2 + V₀ * t + C * t^3 →
  t = (V - V₀) / (2 * g) := by
  sorry

end NUMINAMATH_CALUDE_time_equation_l2210_221061


namespace NUMINAMATH_CALUDE_science_quiz_participation_l2210_221059

theorem science_quiz_participation (j s : ℕ) : 
  j > 0 → s > 0 → (3 * j) / 4 = s / 2 → s = 2 * j := by
  sorry

end NUMINAMATH_CALUDE_science_quiz_participation_l2210_221059


namespace NUMINAMATH_CALUDE_percentage_of_mathematicians_in_it_l2210_221047

theorem percentage_of_mathematicians_in_it (total : ℝ) (mathematicians : ℝ) 
  (h1 : mathematicians > 0) 
  (h2 : total > mathematicians) 
  (h3 : 0.7 * mathematicians = 0.07 * total) : 
  mathematicians / total = 0.1 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_mathematicians_in_it_l2210_221047


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2210_221074

theorem arithmetic_calculation : 2 + 3 * 4 - 5 + 6 - (1 + 2) = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2210_221074


namespace NUMINAMATH_CALUDE_chef_apples_used_l2210_221036

/-- The number of apples the chef used to make pies -/
def applesUsed (initialApples remainingApples : ℕ) : ℕ :=
  initialApples - remainingApples

theorem chef_apples_used :
  let initialApples : ℕ := 43
  let remainingApples : ℕ := 2
  applesUsed initialApples remainingApples = 41 := by
  sorry

end NUMINAMATH_CALUDE_chef_apples_used_l2210_221036


namespace NUMINAMATH_CALUDE_set_cardinality_lower_bound_l2210_221005

theorem set_cardinality_lower_bound (A : Finset ℤ) (m : ℕ) (hm : m ≥ 2) 
  (B : Fin m → Finset ℤ) (hB : ∀ i, B i ⊆ A) (hB_nonempty : ∀ i, (B i).Nonempty) 
  (hsum : ∀ i, (B i).sum id = m ^ (i : ℕ).succ) : 
  A.card ≥ m / 2 := by
  sorry

end NUMINAMATH_CALUDE_set_cardinality_lower_bound_l2210_221005


namespace NUMINAMATH_CALUDE_intersection_M_N_l2210_221019

def M : Set ℝ := {x | (x + 2) * (x - 2) > 0}
def N : Set ℝ := {-3, -2, 2, 3, 4}

theorem intersection_M_N : M ∩ N = {-3, 3, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2210_221019
