import Mathlib

namespace NUMINAMATH_CALUDE_min_area_sum_l2879_287901

def point := ℝ × ℝ

def triangle_area (p1 p2 p3 : point) : ℝ := sorry

def is_integer (x : ℝ) : Prop := ∃ n : ℤ, x = n

theorem min_area_sum (m : ℝ) :
  let p1 : point := (2, 8)
  let p2 : point := (12, 20)
  let p3 : point := (8, m)
  is_integer m →
  (∀ k : ℝ, is_integer k → 
    k ≠ 15.2 → 
    triangle_area p1 p2 (8, k) ≥ triangle_area p1 p2 p3) →
  m ≠ 15.2 →
  (∃ n : ℝ, is_integer n ∧ 
    n ≠ 15.2 ∧
    triangle_area p1 p2 (8, n) = triangle_area p1 p2 p3 ∧
    |m - 15.2| + |n - 15.2| = |14 - 15.2| + |16 - 15.2|) →
  m + (30 - m) = 30 := by sorry

end NUMINAMATH_CALUDE_min_area_sum_l2879_287901


namespace NUMINAMATH_CALUDE_triangle_side_calculation_l2879_287970

theorem triangle_side_calculation (A B C : Real) (a b c : Real) :
  a * Real.cos B = b * Real.sin A →
  C = π / 6 →
  c = 2 →
  b = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_calculation_l2879_287970


namespace NUMINAMATH_CALUDE_rectangular_solid_volume_l2879_287988

theorem rectangular_solid_volume (x y z : ℝ) 
  (h1 : x * y = 3) 
  (h2 : x * z = 5) 
  (h3 : y * z = 15) : 
  x * y * z = 15 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_volume_l2879_287988


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2879_287980

theorem min_value_sum_reciprocals (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 2 → 1/x + 4/y ≥ 1/a + 4/b) →
  1/a + 4/b = 9/2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2879_287980


namespace NUMINAMATH_CALUDE_solve_daily_wage_l2879_287943

def daily_wage_problem (a b c : ℕ) : Prop :=
  -- Define the ratio of daily wages
  a * 4 = b * 3 ∧ b * 5 = c * 4 ∧
  -- Define the total earnings
  6 * a + 9 * b + 4 * c = 1406 ∧
  -- The daily wage of c is 95
  c = 95

theorem solve_daily_wage : ∃ a b c : ℕ, daily_wage_problem a b c :=
sorry

end NUMINAMATH_CALUDE_solve_daily_wage_l2879_287943


namespace NUMINAMATH_CALUDE_no_natural_solutions_l2879_287905

theorem no_natural_solutions : ∀ x y : ℕ, x^2 + x*y + y^2 ≠ x^2 * y^2 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_solutions_l2879_287905


namespace NUMINAMATH_CALUDE_function_properties_l2879_287972

def f (a b x : ℝ) : ℝ := x^2 - (a + 1) * x + b

theorem function_properties (a b : ℝ) :
  (∀ x, f a b x < 0 ↔ -5 < x ∧ x < 2) →
  (a = -4 ∧ b = -10) ∧
  (∀ x, f a a x > 0 ↔
    (a > 1 ∧ (x < 1 ∨ x > a)) ∨
    (a = 1 ∧ x ≠ 1) ∨
    (a < 1 ∧ (x < a ∨ x > 1))) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l2879_287972


namespace NUMINAMATH_CALUDE_french_speakers_l2879_287929

theorem french_speakers (total : ℕ) (latin : ℕ) (neither : ℕ) (both : ℕ) :
  total = 25 →
  latin = 13 →
  neither = 6 →
  both = 9 →
  ∃ french : ℕ, french = 15 ∧ total = latin + french - both + neither :=
by sorry

end NUMINAMATH_CALUDE_french_speakers_l2879_287929


namespace NUMINAMATH_CALUDE_incorrect_average_calculation_l2879_287921

theorem incorrect_average_calculation (n : ℕ) (correct_sum incorrect_sum : ℝ) 
  (h1 : n = 10)
  (h2 : correct_sum / n = 15)
  (h3 : incorrect_sum = correct_sum - 10) :
  incorrect_sum / n = 14 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_average_calculation_l2879_287921


namespace NUMINAMATH_CALUDE_expected_value_is_thirteen_eighths_l2879_287951

/-- Represents the outcome of rolling an 8-sided die -/
inductive DieRoll
  | one
  | two
  | three
  | four
  | five
  | six
  | seven
  | eight

/-- Determines if a DieRoll is prime -/
def isPrime (roll : DieRoll) : Bool :=
  match roll with
  | DieRoll.two | DieRoll.three | DieRoll.five | DieRoll.seven => true
  | _ => false

/-- Calculates the winnings for a given DieRoll -/
def winnings (roll : DieRoll) : Int :=
  match roll with
  | DieRoll.two => 2
  | DieRoll.three => 3
  | DieRoll.five => 5
  | DieRoll.seven => 7
  | DieRoll.eight => -4
  | _ => 0

/-- The probability of each DieRoll -/
def probability : DieRoll → Rat
  | _ => 1/8

/-- The expected value of the winnings -/
def expectedValue : Rat :=
  (winnings DieRoll.one   * probability DieRoll.one)   +
  (winnings DieRoll.two   * probability DieRoll.two)   +
  (winnings DieRoll.three * probability DieRoll.three) +
  (winnings DieRoll.four  * probability DieRoll.four)  +
  (winnings DieRoll.five  * probability DieRoll.five)  +
  (winnings DieRoll.six   * probability DieRoll.six)   +
  (winnings DieRoll.seven * probability DieRoll.seven) +
  (winnings DieRoll.eight * probability DieRoll.eight)

theorem expected_value_is_thirteen_eighths :
  expectedValue = 13/8 := by sorry

end NUMINAMATH_CALUDE_expected_value_is_thirteen_eighths_l2879_287951


namespace NUMINAMATH_CALUDE_race_finish_times_l2879_287906

/-- Race problem statement -/
theorem race_finish_times 
  (malcolm_speed : ℝ) 
  (joshua_speed : ℝ) 
  (lila_speed : ℝ) 
  (race_distance : ℝ) 
  (h1 : malcolm_speed = 6)
  (h2 : joshua_speed = 8)
  (h3 : lila_speed = 7)
  (h4 : race_distance = 12) :
  let malcolm_time := malcolm_speed * race_distance
  let joshua_time := joshua_speed * race_distance
  let lila_time := lila_speed * race_distance
  (joshua_time - malcolm_time = 24 ∧ lila_time - malcolm_time = 12) := by
  sorry


end NUMINAMATH_CALUDE_race_finish_times_l2879_287906


namespace NUMINAMATH_CALUDE_polynomial_division_l2879_287978

def p (x : ℝ) : ℝ := x^5 + 3*x^4 - 28*x^3 + 45*x^2 - 58*x + 24
def d (x : ℝ) : ℝ := x - 3
def q (x : ℝ) : ℝ := x^4 + 6*x^3 - 10*x^2 + 15*x - 13
def r : ℝ := -15

theorem polynomial_division :
  ∀ x : ℝ, p x = d x * q x + r :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_l2879_287978


namespace NUMINAMATH_CALUDE_count_with_zero_3017_l2879_287903

/-- A function that counts the number of integers from 1 to n that contain at least one digit '0' in their base-ten representation. -/
def count_with_zero (n : ℕ) : ℕ :=
  sorry

/-- The theorem stating that the count of positive integers less than or equal to 3017 that contain at least one digit '0' in their base-ten representation is 740. -/
theorem count_with_zero_3017 : count_with_zero 3017 = 740 :=
  sorry

end NUMINAMATH_CALUDE_count_with_zero_3017_l2879_287903


namespace NUMINAMATH_CALUDE_omega_sequence_monotone_increasing_l2879_287968

/-- Definition of an Ω sequence -/
def is_omega_sequence (a : ℕ+ → ℝ) : Prop :=
  (∀ n : ℕ+, (a n + a (n + 2)) / 2 ≤ a (n + 1)) ∧
  (∃ M : ℝ, ∀ n : ℕ+, a n ≤ M)

/-- Theorem: For any Ω sequence of positive integers, each term is less than or equal to the next term -/
theorem omega_sequence_monotone_increasing
  (d : ℕ+ → ℕ+)
  (h_omega : is_omega_sequence (λ n => (d n : ℝ))) :
  ∀ n : ℕ+, d n ≤ d (n + 1) := by
sorry

end NUMINAMATH_CALUDE_omega_sequence_monotone_increasing_l2879_287968


namespace NUMINAMATH_CALUDE_f_min_bound_l2879_287912

noncomputable def f (a x : ℝ) : ℝ := a * (Real.exp x + a) - x

theorem f_min_bound (a : ℝ) (ha : a > 0) :
  ∀ x : ℝ, f a x > 2 * Real.log a + 3/2 := by sorry

end NUMINAMATH_CALUDE_f_min_bound_l2879_287912


namespace NUMINAMATH_CALUDE_verna_sherry_combined_weight_l2879_287977

def haley_weight : ℕ := 103
def verna_weight : ℕ := haley_weight + 17

theorem verna_sherry_combined_weight : 
  ∃ (sherry_weight : ℕ), 
    verna_weight = haley_weight + 17 ∧ 
    verna_weight * 2 = sherry_weight ∧ 
    verna_weight + sherry_weight = 360 :=
by
  sorry

end NUMINAMATH_CALUDE_verna_sherry_combined_weight_l2879_287977


namespace NUMINAMATH_CALUDE_digit_pair_sum_l2879_287926

/-- Two different digits that form two-digit numbers whose sum is 202 -/
structure DigitPair where
  a : ℕ
  b : ℕ
  a_is_digit : a ≥ 1 ∧ a ≤ 9
  b_is_digit : b ≥ 0 ∧ b ≤ 9
  a_ne_b : a ≠ b
  sum_eq_202 : 10 * a + b + 10 * b + a = 202

/-- The sum of the digits in a DigitPair is 12 -/
theorem digit_pair_sum (p : DigitPair) : p.a + p.b = 12 := by
  sorry

end NUMINAMATH_CALUDE_digit_pair_sum_l2879_287926


namespace NUMINAMATH_CALUDE_equation_solutions_l2879_287945

/-- The equation has solutions for all real a, and these solutions are as specified -/
theorem equation_solutions (a : ℝ) :
  let f := fun x : ℝ => (2 - 2*a*(x + 1)) / (|x| - x) = Real.sqrt (1 - a - a*x)
  (∃ x, f x) ∧
  (a < 0 → (f ((1-a)/a) ∧ f (-1))) ∧
  (0 ≤ a ∧ a ≤ 1 → f (-1)) ∧
  (1 < a ∧ a < 2 → (f ((1-a)/a) ∧ f (-1))) ∧
  (a = 2 → (f (-1) ∧ f (-1/2))) ∧
  (a > 2 → (f ((1-a)/a) ∧ f (-1) ∧ f (1-a))) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2879_287945


namespace NUMINAMATH_CALUDE_male_teacher_classes_proof_l2879_287923

/-- Represents the number of classes taught by male teachers when only male teachers are teaching. -/
def male_teacher_classes : ℕ := 10

/-- Represents the number of classes taught by female teachers. -/
def female_teacher_classes : ℕ := 15

/-- Represents the average number of tutoring classes per month. -/
def average_classes : ℕ := 6

theorem male_teacher_classes_proof (x y : ℕ) :
  female_teacher_classes * x = average_classes * (x + y) →
  male_teacher_classes * y = average_classes * (x + y) :=
by sorry

end NUMINAMATH_CALUDE_male_teacher_classes_proof_l2879_287923


namespace NUMINAMATH_CALUDE_gcd_65_169_l2879_287936

theorem gcd_65_169 : Nat.gcd 65 169 = 13 := by
  sorry

end NUMINAMATH_CALUDE_gcd_65_169_l2879_287936


namespace NUMINAMATH_CALUDE_range_when_p_true_range_when_p_and_q_true_l2879_287976

-- Define proposition p
def has_real_roots (m : ℝ) : Prop :=
  ∃ x : ℝ, x^2 - 3*x + m = 0

-- Define proposition q
def is_ellipse_with_x_foci (m : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 / (9 - m) + y^2 / (m - 2) = 1 ∧ 
  9 - m > 0 ∧ m - 2 > 0 ∧ 9 - m > m - 2

-- Theorem 1
theorem range_when_p_true (m : ℝ) :
  has_real_roots m → m ≤ 9/4 := by sorry

-- Theorem 2
theorem range_when_p_and_q_true (m : ℝ) :
  has_real_roots m ∧ is_ellipse_with_x_foci m → 2 < m ∧ m ≤ 9/4 := by sorry

end NUMINAMATH_CALUDE_range_when_p_true_range_when_p_and_q_true_l2879_287976


namespace NUMINAMATH_CALUDE_bug_travel_distance_l2879_287916

/-- The total distance traveled by a bug on a number line -/
def bugDistance (start end1 end2 end3 : ℤ) : ℝ :=
  |end1 - start| + |end2 - end1| + |end3 - end2|

/-- Theorem: The bug's total travel distance is 25 units -/
theorem bug_travel_distance :
  bugDistance (-3) (-7) 8 2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_bug_travel_distance_l2879_287916


namespace NUMINAMATH_CALUDE_sum_of_squares_representation_l2879_287913

theorem sum_of_squares_representation : 
  (((17 ^ 2 + 19 ^ 2) / 2) ^ 2 : ℕ) = 260 ^ 2 + 195 ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_representation_l2879_287913


namespace NUMINAMATH_CALUDE_roller_coaster_tickets_l2879_287985

theorem roller_coaster_tickets : ∃ x : ℕ, 
  (∀ y : ℕ, y = 3 → 7 * x + 4 * y = 47) → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_roller_coaster_tickets_l2879_287985


namespace NUMINAMATH_CALUDE_stream_speed_l2879_287940

/-- The speed of a stream given downstream and upstream speeds -/
theorem stream_speed (downstream_speed upstream_speed : ℝ) 
  (h1 : downstream_speed = 14)
  (h2 : upstream_speed = 8) :
  (downstream_speed - upstream_speed) / 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l2879_287940


namespace NUMINAMATH_CALUDE_point_C_coordinates_l2879_287987

/-- Given points A(-1,2) and B(2,8), if vector AB = 3 * vector AC, 
    then C has coordinates (0,4) -/
theorem point_C_coordinates 
  (A B C : ℝ × ℝ) 
  (h1 : A = (-1, 2)) 
  (h2 : B = (2, 8)) 
  (h3 : B - A = 3 • (C - A)) : 
  C = (0, 4) := by
sorry

end NUMINAMATH_CALUDE_point_C_coordinates_l2879_287987


namespace NUMINAMATH_CALUDE_complex_magnitude_l2879_287994

theorem complex_magnitude (z : ℂ) : z = (2 - I) / (1 + I) → Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l2879_287994


namespace NUMINAMATH_CALUDE_salary_decrease_percentage_l2879_287996

def original_salary : ℝ := 4000.0000000000005
def initial_increase_rate : ℝ := 0.1
def final_salary : ℝ := 4180

theorem salary_decrease_percentage :
  ∃ (decrease_rate : ℝ),
    final_salary = original_salary * (1 + initial_increase_rate) * (1 - decrease_rate) ∧
    decrease_rate = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_salary_decrease_percentage_l2879_287996


namespace NUMINAMATH_CALUDE_basketball_team_probability_l2879_287900

def team_size : ℕ := 12
def main_players : ℕ := 6
def classes_with_two_students : ℕ := 2
def classes_with_one_student : ℕ := 8

theorem basketball_team_probability :
  (Nat.choose classes_with_two_students 1 * Nat.choose classes_with_two_students 1 * Nat.choose classes_with_one_student 4) / 
  Nat.choose team_size main_players = 10 / 33 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_probability_l2879_287900


namespace NUMINAMATH_CALUDE_meet_once_l2879_287924

/-- Represents the movement of Michael and the garbage truck -/
structure Movement where
  michael_speed : ℝ
  truck_speed : ℝ
  pail_distance : ℝ
  truck_stop_time : ℝ

/-- Calculates the number of times Michael and the truck meet -/
def number_of_meetings (m : Movement) : ℕ :=
  sorry

/-- The specific movement scenario described in the problem -/
def problem_scenario : Movement where
  michael_speed := 6
  truck_speed := 12
  pail_distance := 300
  truck_stop_time := 20

/-- Theorem stating that Michael and the truck meet exactly once -/
theorem meet_once : number_of_meetings problem_scenario = 1 := by
  sorry

end NUMINAMATH_CALUDE_meet_once_l2879_287924


namespace NUMINAMATH_CALUDE_multiply_a_equals_four_l2879_287958

theorem multiply_a_equals_four (a b x : ℝ) 
  (h1 : x * a = 5 * b) 
  (h2 : a * b ≠ 0) 
  (h3 : a / 5 = b / 4) : 
  x = 4 := by
  sorry

end NUMINAMATH_CALUDE_multiply_a_equals_four_l2879_287958


namespace NUMINAMATH_CALUDE_squirrel_nuts_problem_l2879_287954

theorem squirrel_nuts_problem 
  (a b c d : ℕ) 
  (h1 : a + b + c + d = 2020)
  (h2 : a ≥ 103 ∧ b ≥ 103 ∧ c ≥ 103 ∧ d ≥ 103)
  (h3 : a > b ∧ a > c ∧ a > d)
  (h4 : b + c = 1277) :
  a = 640 := by
  sorry

end NUMINAMATH_CALUDE_squirrel_nuts_problem_l2879_287954


namespace NUMINAMATH_CALUDE_no_solution_equation_simplify_fraction_l2879_287902

-- Problem 1
theorem no_solution_equation :
  ¬ ∃ x : ℝ, (3 - x) / (x - 4) - 1 / (4 - x) = 1 := by sorry

-- Problem 2
theorem simplify_fraction (x : ℝ) (h : x ≠ 2 ∧ x ≠ -2) :
  2 * x / (x^2 - 4) - 1 / (x + 2) = 1 / (x - 2) := by sorry

end NUMINAMATH_CALUDE_no_solution_equation_simplify_fraction_l2879_287902


namespace NUMINAMATH_CALUDE_complex_sum_of_parts_l2879_287950

theorem complex_sum_of_parts (z : ℂ) (h : z * Complex.I = 1 + Complex.I) :
  z.re + z.im = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_of_parts_l2879_287950


namespace NUMINAMATH_CALUDE_min_value_problem_l2879_287975

theorem min_value_problem (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h1 : a^2 + b^2 = 4) (h2 : c * d = 1) :
  (a^2 * c^2 + b^2 * d^2) * (b^2 * c^2 + a^2 * d^2) ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_problem_l2879_287975


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2879_287904

theorem quadratic_inequality_solution_set :
  let f : ℝ → ℝ := λ x => x^2 + 2*x - 3
  {x : ℝ | f x ≥ 0} = {x : ℝ | x ≤ -3 ∨ x ≥ 1} := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2879_287904


namespace NUMINAMATH_CALUDE_sum_of_divisors_360_l2879_287971

def sum_of_divisors (n : ℕ) : ℕ := (Finset.filter (·∣n) (Finset.range (n + 1))).sum id

theorem sum_of_divisors_360 : sum_of_divisors 360 = 1170 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_divisors_360_l2879_287971


namespace NUMINAMATH_CALUDE_rural_school_absence_percentage_l2879_287955

theorem rural_school_absence_percentage :
  let total_students : ℕ := 120
  let boys : ℕ := 70
  let girls : ℕ := 50
  let absent_boys : ℕ := boys / 5
  let absent_girls : ℕ := girls / 4
  let total_absent : ℕ := absent_boys + absent_girls
  (total_absent : ℚ) / total_students * 100 = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_rural_school_absence_percentage_l2879_287955


namespace NUMINAMATH_CALUDE_min_sum_reciprocal_constraint_l2879_287910

theorem min_sum_reciprocal_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : (2 / x) + (2 / y) = 1) : 
  x + y ≥ 8 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ (2 / x₀) + (2 / y₀) = 1 ∧ x₀ + y₀ = 8 :=
sorry

end NUMINAMATH_CALUDE_min_sum_reciprocal_constraint_l2879_287910


namespace NUMINAMATH_CALUDE_largest_prime_factors_difference_l2879_287909

/-- The positive difference between the two largest prime factors of 204204 is 16 -/
theorem largest_prime_factors_difference : ∃ (p q : Nat), 
  Nat.Prime p ∧ Nat.Prime q ∧ p > q ∧
  p ∣ 204204 ∧ q ∣ 204204 ∧
  (∀ r : Nat, Nat.Prime r → r ∣ 204204 → r ≤ p) ∧
  (∀ r : Nat, Nat.Prime r → r ∣ 204204 → r ≠ p → r ≤ q) ∧
  p - q = 16 := by
sorry

end NUMINAMATH_CALUDE_largest_prime_factors_difference_l2879_287909


namespace NUMINAMATH_CALUDE_triangular_front_view_solids_l2879_287931

/-- A type representing different types of solids -/
inductive Solid
  | TriangularPyramid
  | SquarePyramid
  | TriangularPrism
  | SquarePrism
  | Cone
  | Cylinder

/-- A predicate that determines if a solid can have a triangular front view -/
def has_triangular_front_view (s : Solid) : Prop :=
  match s with
  | Solid.TriangularPyramid => True
  | Solid.SquarePyramid => True
  | Solid.TriangularPrism => True
  | Solid.Cone => True
  | _ => False

/-- Theorem stating which solids can have a triangular front view -/
theorem triangular_front_view_solids :
  ∀ s : Solid, has_triangular_front_view s ↔ 
    (s = Solid.TriangularPyramid ∨ 
     s = Solid.SquarePyramid ∨ 
     s = Solid.TriangularPrism ∨ 
     s = Solid.Cone) :=
by sorry

end NUMINAMATH_CALUDE_triangular_front_view_solids_l2879_287931


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2879_287989

theorem trigonometric_identity (α β : ℝ) 
  (h : Real.cos α ^ 2 * Real.sin β ^ 2 + Real.sin α ^ 2 * Real.cos β ^ 2 = 
       Real.cos α * Real.sin α * Real.cos β * Real.sin β) : 
  (Real.sin β ^ 2 * Real.cos α ^ 2) / Real.sin α ^ 2 + 
  (Real.cos β ^ 2 * Real.sin α ^ 2) / Real.cos α ^ 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2879_287989


namespace NUMINAMATH_CALUDE_jony_walking_speed_l2879_287939

/-- Calculates the walking speed given distance and time -/
def walking_speed (distance : ℕ) (time : ℕ) : ℕ :=
  distance / time

/-- The number of blocks Jony walks -/
def blocks_walked : ℕ := (90 - 10) + (90 - 70)

/-- The length of each block in meters -/
def block_length : ℕ := 40

/-- The total distance Jony walks in meters -/
def total_distance : ℕ := blocks_walked * block_length

/-- The total time Jony spends walking in minutes -/
def total_time : ℕ := 40

theorem jony_walking_speed :
  walking_speed total_distance total_time = 100 := by
  sorry

end NUMINAMATH_CALUDE_jony_walking_speed_l2879_287939


namespace NUMINAMATH_CALUDE_lucas_payment_l2879_287938

/-- Calculates the payment for window cleaning based on given conditions -/
def calculate_payment (windows_per_floor : ℕ) (num_floors : ℕ) (pay_per_window : ℕ) 
                      (deduction_per_period : ℕ) (days_per_period : ℕ) (days_taken : ℕ) : ℕ :=
  let total_windows := windows_per_floor * num_floors
  let gross_pay := total_windows * pay_per_window
  let num_periods := days_taken / days_per_period
  let total_deduction := num_periods * deduction_per_period
  gross_pay - total_deduction

/-- Theorem stating that Lucas will be paid $16 for cleaning windows -/
theorem lucas_payment : 
  calculate_payment 3 3 2 1 3 6 = 16 := by
  sorry

end NUMINAMATH_CALUDE_lucas_payment_l2879_287938


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2879_287992

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) = q * a n) →  -- geometric sequence condition
  q ≠ 1 →  -- common ratio is not 1
  (∀ n, S n = (a 1) * (1 - q^n) / (1 - q)) →  -- sum formula for geometric sequence
  (2 * a 5 = (a 2 + 3 * a 8) / 2) →  -- arithmetic sequence condition
  (3 * S 3) / (S 6) = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2879_287992


namespace NUMINAMATH_CALUDE_japanese_study_fraction_l2879_287932

theorem japanese_study_fraction (J S : ℕ) (x : ℚ) : 
  S = 2 * J →
  (3 / 8 : ℚ) * S + x * J = (1 / 3 : ℚ) * (J + S) →
  x = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_japanese_study_fraction_l2879_287932


namespace NUMINAMATH_CALUDE_wood_measurement_equations_l2879_287984

/-- Represents the wood measurement problem from "The Mathematical Classic of Sunzi" -/
def wood_measurement_problem (x y : ℝ) : Prop :=
  (y - x = 4.5) ∧ (y / 2 = x - 1)

/-- The correct system of equations for the wood measurement problem -/
theorem wood_measurement_equations :
  ∃ x y : ℝ,
    (x > 0) ∧  -- Length of wood is positive
    (y > 0) ∧  -- Length of rope is positive
    (y > x) ∧  -- Rope is longer than wood
    wood_measurement_problem x y :=
sorry

end NUMINAMATH_CALUDE_wood_measurement_equations_l2879_287984


namespace NUMINAMATH_CALUDE_polynomial_divisibility_double_divisibility_not_triple_divisible_l2879_287948

/-- Definition of the polynomial P_n(x) -/
def P (n : ℕ) (x : ℝ) : ℝ := (x + 1)^n - x^n - 1

/-- Definition of divisibility for polynomials -/
def divisible (p q : ℝ → ℝ) : Prop := ∃ r : ℝ → ℝ, ∀ x, p x = q x * r x

theorem polynomial_divisibility (n : ℕ) :
  (∃ k : ℤ, n = 6 * k + 1 ∨ n = 6 * k - 1) ↔ 
  divisible (P n) (fun x ↦ x^2 + x + 1) :=
sorry

theorem double_divisibility (n : ℕ) :
  (∃ k : ℤ, n = 6 * k + 1) ↔ 
  divisible (P n) (fun x ↦ (x^2 + x + 1)^2) :=
sorry

theorem not_triple_divisible (n : ℕ) :
  ¬(divisible (P n) (fun x ↦ (x^2 + x + 1)^3)) :=
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_double_divisibility_not_triple_divisible_l2879_287948


namespace NUMINAMATH_CALUDE_sock_selection_theorem_l2879_287919

/-- The number of ways to select two socks of different colors -/
def differentColorPairs (white brown blue : ℕ) : ℕ :=
  white * brown + brown * blue + white * blue

/-- Theorem: The number of ways to select two socks of different colors
    from a drawer containing 5 white socks, 3 brown socks, and 4 blue socks
    is equal to 47. -/
theorem sock_selection_theorem :
  differentColorPairs 5 3 4 = 47 := by
  sorry

end NUMINAMATH_CALUDE_sock_selection_theorem_l2879_287919


namespace NUMINAMATH_CALUDE_max_leap_years_150_years_l2879_287922

/-- A calendrical system where leap years occur every four years -/
structure CalendarSystem where
  leap_year_frequency : ℕ
  leap_year_frequency_is_four : leap_year_frequency = 4

/-- The maximum number of leap years in a given period -/
def max_leap_years (c : CalendarSystem) (period : ℕ) : ℕ :=
  (period / c.leap_year_frequency) + min 1 (period % c.leap_year_frequency)

/-- Theorem stating that the maximum number of leap years in a 150-year period is 38 -/
theorem max_leap_years_150_years (c : CalendarSystem) :
  max_leap_years c 150 = 38 := by
  sorry

#eval max_leap_years ⟨4, rfl⟩ 150

end NUMINAMATH_CALUDE_max_leap_years_150_years_l2879_287922


namespace NUMINAMATH_CALUDE_lemon_orange_ratio_decrease_l2879_287947

/-- Calculates the percentage decrease in the ratio of lemons to oranges --/
theorem lemon_orange_ratio_decrease 
  (initial_lemons : ℕ) 
  (initial_oranges : ℕ) 
  (final_lemons : ℕ) 
  (final_oranges : ℕ) 
  (h1 : initial_lemons = 50) 
  (h2 : initial_oranges = 60) 
  (h3 : final_lemons = 20) 
  (h4 : final_oranges = 40) :
  (1 - (final_lemons * initial_oranges) / (initial_lemons * final_oranges : ℚ)) * 100 = 40 := by
  sorry


end NUMINAMATH_CALUDE_lemon_orange_ratio_decrease_l2879_287947


namespace NUMINAMATH_CALUDE_spongebob_earnings_l2879_287993

/-- Represents the sales data for a single item --/
structure ItemSales where
  quantity : ℕ
  price : ℚ

/-- Calculates the total earnings for a single item --/
def itemEarnings (item : ItemSales) : ℚ :=
  item.quantity * item.price

/-- Represents all sales data for the day --/
structure DailySales where
  burgers : ItemSales
  largeFries : ItemSales
  smallFries : ItemSales
  sodas : ItemSales
  milkshakes : ItemSales
  softServeCones : ItemSales

/-- Calculates the total earnings for the day --/
def totalEarnings (sales : DailySales) : ℚ :=
  itemEarnings sales.burgers +
  itemEarnings sales.largeFries +
  itemEarnings sales.smallFries +
  itemEarnings sales.sodas +
  itemEarnings sales.milkshakes +
  itemEarnings sales.softServeCones

/-- Spongebob's sales data for the day --/
def spongebobSales : DailySales :=
  { burgers := { quantity := 30, price := 2.5 }
  , largeFries := { quantity := 12, price := 1.75 }
  , smallFries := { quantity := 20, price := 1.25 }
  , sodas := { quantity := 50, price := 1 }
  , milkshakes := { quantity := 18, price := 2.85 }
  , softServeCones := { quantity := 40, price := 1.3 }
  }

theorem spongebob_earnings :
  totalEarnings spongebobSales = 274.3 := by
  sorry

end NUMINAMATH_CALUDE_spongebob_earnings_l2879_287993


namespace NUMINAMATH_CALUDE_product_of_roots_quadratic_l2879_287907

theorem product_of_roots_quadratic (x₁ x₂ : ℝ) : 
  x₁^2 - 2*x₁ - 3 = 0 → 
  x₂^2 - 2*x₂ - 3 = 0 → 
  x₁ * x₂ = -3 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_quadratic_l2879_287907


namespace NUMINAMATH_CALUDE_vector_sum_proof_l2879_287935

/-- Given vectors a = (2,3) and b = (-1,2), prove their sum is (1,5) -/
theorem vector_sum_proof :
  let a : Fin 2 → ℝ := ![2, 3]
  let b : Fin 2 → ℝ := ![-1, 2]
  (a + b) = ![1, 5] := by sorry

end NUMINAMATH_CALUDE_vector_sum_proof_l2879_287935


namespace NUMINAMATH_CALUDE_remainder_of_sum_of_powers_l2879_287949

theorem remainder_of_sum_of_powers (n : ℕ) : (20^16 + 201^6) % 9 = 7 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_sum_of_powers_l2879_287949


namespace NUMINAMATH_CALUDE_factor_expression_l2879_287979

theorem factor_expression (x : ℝ) : x * (x - 4) + 2 * (x - 4) = (x + 2) * (x - 4) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2879_287979


namespace NUMINAMATH_CALUDE_second_half_speed_l2879_287930

theorem second_half_speed (total_distance : ℝ) (first_half_speed : ℝ) (total_time : ℝ)
  (h1 : total_distance = 3600)
  (h2 : first_half_speed = 90)
  (h3 : total_time = 30) :
  (total_distance / 2) / (total_time - (total_distance / 2) / first_half_speed) = 180 :=
by sorry

end NUMINAMATH_CALUDE_second_half_speed_l2879_287930


namespace NUMINAMATH_CALUDE_parabola_focal_distance_l2879_287959

/-- Given a parabola y^2 = 2px with focus F and a point A(1, 2) on the parabola, |AF| = 2 -/
theorem parabola_focal_distance (p : ℝ) (F : ℝ × ℝ) :
  (∀ x y, y^2 = 2*p*x → (x, y) = (1, 2)) →  -- point A(1, 2) satisfies the parabola equation
  F.1 = p/2 →  -- x-coordinate of focus
  F.2 = 0 →  -- y-coordinate of focus
  let A := (1, 2)
  ((A.1 - F.1)^2 + (A.2 - F.2)^2).sqrt = 2 := by
sorry

end NUMINAMATH_CALUDE_parabola_focal_distance_l2879_287959


namespace NUMINAMATH_CALUDE_work_completion_time_l2879_287999

/-- The number of days it takes for A to complete the work alone -/
def days_A : ℝ := 30

/-- The number of days it takes for B to complete the work alone -/
def days_B : ℝ := 55

/-- The number of days it takes for A and B to complete the work together -/
def days_AB : ℝ := 19.411764705882355

theorem work_completion_time :
  (1 / days_A) + (1 / days_B) = (1 / days_AB) := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l2879_287999


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2879_287957

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B : Set ℝ := {x | x > 1}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x | 1 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2879_287957


namespace NUMINAMATH_CALUDE_campaign_fundraising_l2879_287928

theorem campaign_fundraising (max_donation : ℕ) (max_donors : ℕ) (half_donors : ℕ) (percentage : ℚ) : 
  max_donation = 1200 →
  max_donors = 500 →
  half_donors = 3 * max_donors →
  percentage = 40 / 100 →
  (max_donation * max_donors + (max_donation / 2) * half_donors) / percentage = 3750000 := by
sorry

end NUMINAMATH_CALUDE_campaign_fundraising_l2879_287928


namespace NUMINAMATH_CALUDE_total_produce_yield_l2879_287990

def garden_length_steps : ℕ := 18
def garden_width_steps : ℕ := 25
def feet_per_step : ℕ := 3
def carrot_yield_per_sqft : ℚ := 0.4
def potato_yield_per_sqft : ℚ := 0.5

theorem total_produce_yield :
  let garden_length_feet := garden_length_steps * feet_per_step
  let garden_width_feet := garden_width_steps * feet_per_step
  let garden_area := garden_length_feet * garden_width_feet
  let carrot_yield := garden_area * carrot_yield_per_sqft
  let potato_yield := garden_area * potato_yield_per_sqft
  carrot_yield + potato_yield = 3645 := by
  sorry

end NUMINAMATH_CALUDE_total_produce_yield_l2879_287990


namespace NUMINAMATH_CALUDE_haley_initial_marbles_l2879_287998

/-- The number of marbles Haley gives to each boy -/
def marbles_per_boy : ℕ := 2

/-- The number of boys in Haley's class who receive marbles -/
def number_of_boys : ℕ := 14

/-- The total number of marbles Haley had initially -/
def total_marbles : ℕ := marbles_per_boy * number_of_boys

theorem haley_initial_marbles : total_marbles = 28 := by
  sorry

end NUMINAMATH_CALUDE_haley_initial_marbles_l2879_287998


namespace NUMINAMATH_CALUDE_point_not_in_transformed_plane_l2879_287933

/-- A plane in 3D space -/
structure Plane where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ

/-- A point in 3D space -/
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Similarity transformation of a plane -/
def transformPlane (p : Plane) (k : ℝ) : Plane :=
  { A := p.A, B := p.B, C := p.C, D := k * p.D }

/-- Check if a point satisfies a plane equation -/
def satisfiesPlane (point : Point) (plane : Plane) : Prop :=
  plane.A * point.x + plane.B * point.y + plane.C * point.z + plane.D = 0

theorem point_not_in_transformed_plane :
  let originalPlane : Plane := { A := 7, B := -6, C := 1, D := -5 }
  let k : ℝ := -2
  let A : Point := { x := 1, y := 1, z := 1 }
  let transformedPlane := transformPlane originalPlane k
  ¬ satisfiesPlane A transformedPlane := by
  sorry

end NUMINAMATH_CALUDE_point_not_in_transformed_plane_l2879_287933


namespace NUMINAMATH_CALUDE_spinner_probability_l2879_287937

/-- Represents the outcomes of the spinner -/
inductive SpinnerOutcome
| one
| two
| three
| five

/-- Represents a three-digit number formed by three spins -/
structure ThreeDigitNumber where
  hundreds : SpinnerOutcome
  tens : SpinnerOutcome
  units : SpinnerOutcome

def isDivisibleByFive (n : ThreeDigitNumber) : Prop :=
  n.units = SpinnerOutcome.five

def totalOutcomes : ℕ := 4^3

def favorableOutcomes : ℕ := 4 * 4

theorem spinner_probability :
  (favorableOutcomes : ℚ) / totalOutcomes = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_spinner_probability_l2879_287937


namespace NUMINAMATH_CALUDE_rectangle_circle_equality_l2879_287986

/-- Given a rectangle with sides a and b, where a = 2b, and a circle with radius 3,
    if the perimeter of the rectangle equals the circumference of the circle,
    then a = 2π and b = π. -/
theorem rectangle_circle_equality (a b : ℝ) :
  a = 2 * b →
  2 * (a + b) = 2 * π * 3 →
  a = 2 * π ∧ b = π := by
sorry

end NUMINAMATH_CALUDE_rectangle_circle_equality_l2879_287986


namespace NUMINAMATH_CALUDE_phil_wins_n_12_ellie_wins_n_2012_l2879_287927

/-- Represents a move on the chessboard -/
structure Move where
  x : Nat
  y : Nat
  shape : Fin 4 -- 4 possible L-shapes

/-- Represents the state of the chessboard -/
def Board (n : Nat) := Fin n → Fin n → Fin n

/-- Applies a move to the board -/
def applyMove (n : Nat) (board : Board n) (move : Move) : Board n :=
  sorry

/-- Checks if all numbers on the board are zero -/
def allZero (n : Nat) (board : Board n) : Prop :=
  sorry

/-- Sum of all numbers on the board -/
def boardSum (n : Nat) (board : Board n) : Nat :=
  sorry

theorem phil_wins_n_12 :
  ∃ (initial : Board 12),
    ∀ (moves : List Move),
      ¬(boardSum 12 (moves.foldl (applyMove 12) initial) % 3 = 0) :=
sorry

theorem ellie_wins_n_2012 :
  ∀ (initial : Board 2012),
    ∃ (moves : List Move),
      allZero 2012 (moves.foldl (applyMove 2012) initial) :=
sorry

end NUMINAMATH_CALUDE_phil_wins_n_12_ellie_wins_n_2012_l2879_287927


namespace NUMINAMATH_CALUDE_odd_sum_of_squares_implies_odd_sum_l2879_287963

theorem odd_sum_of_squares_implies_odd_sum (n m : ℤ) (h : Odd (n^2 + m^2)) : Odd (n + m) := by
  sorry

end NUMINAMATH_CALUDE_odd_sum_of_squares_implies_odd_sum_l2879_287963


namespace NUMINAMATH_CALUDE_milk_distribution_l2879_287966

/-- Given a total number of milk bottles, number of cartons, and number of bags per carton,
    calculate the number of bottles in one bag. -/
def bottles_per_bag (total_bottles : ℕ) (num_cartons : ℕ) (bags_per_carton : ℕ) : ℕ :=
  total_bottles / (num_cartons * bags_per_carton)

/-- Prove that given 180 total bottles, 3 cartons, and 4 bags per carton,
    the number of bottles in one bag is 15. -/
theorem milk_distribution :
  bottles_per_bag 180 3 4 = 15 := by
  sorry

end NUMINAMATH_CALUDE_milk_distribution_l2879_287966


namespace NUMINAMATH_CALUDE_pharmacy_masks_problem_l2879_287911

theorem pharmacy_masks_problem (first_batch_cost second_batch_cost : ℕ)
  (h1 : first_batch_cost = 1600)
  (h2 : second_batch_cost = 6000)
  (h3 : ∃ (x : ℕ), x > 0 ∧ 
    (second_batch_cost : ℚ) / (3 * x) - (first_batch_cost : ℚ) / x = 2) :
  ∃ (x : ℕ), x = 200 ∧ 
    (second_batch_cost : ℚ) / (3 * x) - (first_batch_cost : ℚ) / x = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_pharmacy_masks_problem_l2879_287911


namespace NUMINAMATH_CALUDE_max_consecutive_sum_less_than_500_l2879_287995

/-- The sum of the first n positive integers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- 31 is the largest positive integer n such that the sum of the first n positive integers is less than 500 -/
theorem max_consecutive_sum_less_than_500 :
  ∀ n : ℕ, sum_first_n n < 500 ↔ n ≤ 31 :=
by sorry

end NUMINAMATH_CALUDE_max_consecutive_sum_less_than_500_l2879_287995


namespace NUMINAMATH_CALUDE_school_play_tickets_l2879_287953

theorem school_play_tickets (total_money : ℕ) (adult_price child_price : ℕ) 
  (child_tickets : ℕ) :
  total_money = 104 →
  adult_price = 6 →
  child_price = 4 →
  child_tickets = 11 →
  ∃ (adult_tickets : ℕ), 
    adult_price * adult_tickets + child_price * child_tickets = total_money ∧
    adult_tickets + child_tickets = 21 := by
  sorry

end NUMINAMATH_CALUDE_school_play_tickets_l2879_287953


namespace NUMINAMATH_CALUDE_pink_highlighters_count_l2879_287944

def total_highlighters : ℕ := 33
def yellow_highlighters : ℕ := 15
def blue_highlighters : ℕ := 8

theorem pink_highlighters_count :
  total_highlighters - yellow_highlighters - blue_highlighters = 10 := by
  sorry

end NUMINAMATH_CALUDE_pink_highlighters_count_l2879_287944


namespace NUMINAMATH_CALUDE_convex_polygon_23_sides_diagonals_l2879_287917

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A convex polygon with 23 sides has 230 diagonals -/
theorem convex_polygon_23_sides_diagonals :
  num_diagonals 23 = 230 := by sorry

end NUMINAMATH_CALUDE_convex_polygon_23_sides_diagonals_l2879_287917


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2879_287941

theorem inequality_solution_set (a : ℝ) :
  let S := {x : ℝ | x^2 - (a + 1) * x + a < 0}
  (a > 1 → S = {x : ℝ | 1 < x ∧ x < a}) ∧
  (a = 1 → S = ∅) ∧
  (a < 1 → S = {x : ℝ | a < x ∧ x < 1}) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2879_287941


namespace NUMINAMATH_CALUDE_artist_paintings_l2879_287965

/-- Calculates the number of paintings an artist can make in a given number of weeks -/
def paintings_made (hours_per_week : ℕ) (hours_per_painting : ℕ) (num_weeks : ℕ) : ℕ :=
  (hours_per_week / hours_per_painting) * num_weeks

/-- Proves that an artist spending 30 hours per week painting, taking 3 hours per painting, can make 40 paintings in 4 weeks -/
theorem artist_paintings : paintings_made 30 3 4 = 40 := by
  sorry

end NUMINAMATH_CALUDE_artist_paintings_l2879_287965


namespace NUMINAMATH_CALUDE_square_expression_l2879_287983

theorem square_expression (x y : ℝ) (square : ℝ) :
  4 * x^2 * square = 81 * x^3 * y → square = (81/4) * x * y := by
  sorry

end NUMINAMATH_CALUDE_square_expression_l2879_287983


namespace NUMINAMATH_CALUDE_third_root_of_polynomial_l2879_287960

theorem third_root_of_polynomial (a b : ℚ) : 
  (∀ x : ℚ, a * x^3 + (a + 3*b) * x^2 + (2*b - 4*a) * x + (10 - a) = 0 ↔ x = -1 ∨ x = 4 ∨ x = -24/19) :=
by sorry

end NUMINAMATH_CALUDE_third_root_of_polynomial_l2879_287960


namespace NUMINAMATH_CALUDE_shopkeeper_loss_l2879_287925

/-- Represents the shopkeeper's fruit inventory and sales data --/
structure FruitShop where
  total_fruit : ℝ
  apples : ℝ
  oranges : ℝ
  bananas : ℝ
  apple_price : ℝ
  orange_price : ℝ
  banana_price : ℝ
  apple_increase : ℝ
  orange_increase : ℝ
  banana_increase : ℝ
  overhead : ℝ
  apple_morning_sales : ℝ
  orange_morning_sales : ℝ
  banana_morning_sales : ℝ

/-- Calculates the profit of the fruit shop --/
def calculate_profit (shop : FruitShop) : ℝ :=
  let morning_revenue := 
    shop.apple_price * shop.apples * shop.apple_morning_sales +
    shop.orange_price * shop.oranges * shop.orange_morning_sales +
    shop.banana_price * shop.bananas * shop.banana_morning_sales
  let afternoon_revenue := 
    shop.apple_price * (1 + shop.apple_increase) * shop.apples * (1 - shop.apple_morning_sales) +
    shop.orange_price * (1 + shop.orange_increase) * shop.oranges * (1 - shop.orange_morning_sales) +
    shop.banana_price * (1 + shop.banana_increase) * shop.bananas * (1 - shop.banana_morning_sales)
  let total_revenue := morning_revenue + afternoon_revenue
  let total_cost := 
    shop.apple_price * shop.apples +
    shop.orange_price * shop.oranges +
    shop.banana_price * shop.bananas +
    shop.overhead
  total_revenue - total_cost

/-- Theorem stating that the shopkeeper incurs a loss of $178.88 --/
theorem shopkeeper_loss (shop : FruitShop) 
  (h1 : shop.total_fruit = 700)
  (h2 : shop.apples = 280)
  (h3 : shop.oranges = 210)
  (h4 : shop.bananas = shop.total_fruit - shop.apples - shop.oranges)
  (h5 : shop.apple_price = 5)
  (h6 : shop.orange_price = 4)
  (h7 : shop.banana_price = 2)
  (h8 : shop.apple_increase = 0.12)
  (h9 : shop.orange_increase = 0.15)
  (h10 : shop.banana_increase = 0.08)
  (h11 : shop.overhead = 320)
  (h12 : shop.apple_morning_sales = 0.5)
  (h13 : shop.orange_morning_sales = 0.6)
  (h14 : shop.banana_morning_sales = 0.8) :
  calculate_profit shop = -178.88 := by
  sorry


end NUMINAMATH_CALUDE_shopkeeper_loss_l2879_287925


namespace NUMINAMATH_CALUDE_complex_z_in_first_quadrant_l2879_287956

theorem complex_z_in_first_quadrant (z : ℂ) (h : (1 : ℂ) + Complex.I = Complex.I / z) :
  0 < z.re ∧ 0 < z.im := by
  sorry

end NUMINAMATH_CALUDE_complex_z_in_first_quadrant_l2879_287956


namespace NUMINAMATH_CALUDE_horner_method_f_3_l2879_287946

/-- Horner's method for polynomial evaluation -/
def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 5x^5 + 4x^4 + 3x^3 + 2x^2 + x -/
def f (x : ℝ) : ℝ := 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x

theorem horner_method_f_3 :
  f 3 = horner_eval [5, 4, 3, 2, 1, 0] 3 ∧ horner_eval [5, 4, 3, 2, 1, 0] 3 = 1641 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_f_3_l2879_287946


namespace NUMINAMATH_CALUDE_three_lines_intersection_l2879_287997

/-- Represents a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- The three lines in the problem -/
def line1 : Line := ⟨1, 1, 1⟩
def line2 : Line := ⟨2, -1, 8⟩
def line3 (a : ℝ) : Line := ⟨a, 3, -5⟩

/-- The theorem to be proved -/
theorem three_lines_intersection (a : ℝ) : 
  (parallel line1 (line3 a) ∨ parallel line2 (line3 a)) → 
  a = 3 ∨ a = -6 :=
sorry

end NUMINAMATH_CALUDE_three_lines_intersection_l2879_287997


namespace NUMINAMATH_CALUDE_product_closure_l2879_287982

def A : Set ℤ := {n | ∃ t s : ℤ, n = t^2 + s^2}

theorem product_closure (x y : ℤ) (hx : x ∈ A) (hy : y ∈ A) : x * y ∈ A := by
  sorry

end NUMINAMATH_CALUDE_product_closure_l2879_287982


namespace NUMINAMATH_CALUDE_initial_cloth_length_l2879_287973

/-- Given that 4 men can colour an initial length of cloth in 2 days,
    and 8 men can colour 36 meters of cloth in 0.75 days,
    prove that the initial length of cloth is 48 meters. -/
theorem initial_cloth_length (initial_length : ℝ) : 
  (4 * initial_length / 2 = 8 * 36 / 0.75) → initial_length = 48 := by
  sorry

end NUMINAMATH_CALUDE_initial_cloth_length_l2879_287973


namespace NUMINAMATH_CALUDE_hilt_fountain_distance_l2879_287942

/-- The total distance Mrs. Hilt walks to and from the water fountain -/
def total_distance (distance_to_fountain : ℕ) (number_of_trips : ℕ) : ℕ :=
  2 * distance_to_fountain * number_of_trips

/-- Theorem: Mrs. Hilt walks 240 feet in total -/
theorem hilt_fountain_distance :
  total_distance 30 4 = 240 := by
  sorry

end NUMINAMATH_CALUDE_hilt_fountain_distance_l2879_287942


namespace NUMINAMATH_CALUDE_green_toad_count_l2879_287908

/-- Represents the number of toads per acre -/
structure ToadPopulation where
  green : ℕ
  brown : ℕ
  spotted_brown : ℕ

/-- The conditions of the toad population -/
def valid_population (p : ToadPopulation) : Prop :=
  p.brown = 25 * p.green ∧
  p.spotted_brown = p.brown / 4 ∧
  p.spotted_brown = 50

/-- Theorem stating that in a valid toad population, there are 8 green toads per acre -/
theorem green_toad_count (p : ToadPopulation) (h : valid_population p) : p.green = 8 := by
  sorry


end NUMINAMATH_CALUDE_green_toad_count_l2879_287908


namespace NUMINAMATH_CALUDE_tangent_lines_to_circle_tangent_lines_correct_l2879_287969

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space using the general form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Checks if a line is tangent to a circle centered at the origin -/
def isTangentToCircle (l : Line) (r : ℝ) : Prop :=
  (l.a ^ 2 + l.b ^ 2) * r ^ 2 = l.c ^ 2

theorem tangent_lines_to_circle (p : Point) (r : ℝ) :
  p.x ^ 2 + p.y ^ 2 > r ^ 2 →
  (∃ l₁ l₂ : Line,
    (pointOnLine p l₁ ∧ isTangentToCircle l₁ r) ∧
    (pointOnLine p l₂ ∧ isTangentToCircle l₂ r) ∧
    ((l₁.a = 1 ∧ l₁.b = 0 ∧ l₁.c = -p.x) ∨
     (l₂.a = 5 ∧ l₂.b = -12 ∧ l₂.c = 26))) :=
by sorry

/-- The main theorem that proves the given tangent lines are correct -/
theorem tangent_lines_correct : 
  ∃ l₁ l₂ : Line,
    (pointOnLine ⟨2, 3⟩ l₁ ∧ isTangentToCircle l₁ 2) ∧
    (pointOnLine ⟨2, 3⟩ l₂ ∧ isTangentToCircle l₂ 2) ∧
    ((l₁.a = 1 ∧ l₁.b = 0 ∧ l₁.c = -2) ∨
     (l₂.a = 5 ∧ l₂.b = -12 ∧ l₂.c = 26)) :=
by
  apply tangent_lines_to_circle ⟨2, 3⟩ 2
  norm_num

end NUMINAMATH_CALUDE_tangent_lines_to_circle_tangent_lines_correct_l2879_287969


namespace NUMINAMATH_CALUDE_union_determines_x_l2879_287961

def A : Set ℕ := {1, 3}
def B (x : ℕ) : Set ℕ := {2, x}

theorem union_determines_x (x : ℕ) : A ∪ B x = {1, 2, 3, 4} → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_union_determines_x_l2879_287961


namespace NUMINAMATH_CALUDE_angle_equality_l2879_287991

theorem angle_equality (θ : Real) (h1 : Real.cos (60 * π / 180) = Real.cos (45 * π / 180) * Real.cos θ) 
  (h2 : 0 ≤ θ) (h3 : θ ≤ π / 2) : θ = 45 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_angle_equality_l2879_287991


namespace NUMINAMATH_CALUDE_product_equality_l2879_287914

theorem product_equality : 1500 * 2987 * 0.2987 * 15 = 2989502.987 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l2879_287914


namespace NUMINAMATH_CALUDE_g_9_l2879_287920

/-- A function g satisfying g(x + y) = g(x) * g(y) for all real x and y, and g(3) = 4 -/
def g : ℝ → ℝ :=
  fun x => sorry

/-- The functional equation for g -/
axiom g_mul (x y : ℝ) : g (x + y) = g x * g y

/-- The initial condition for g -/
axiom g_3 : g 3 = 4

/-- Theorem stating that g(9) = 64 -/
theorem g_9 : g 9 = 64 := by
  sorry

end NUMINAMATH_CALUDE_g_9_l2879_287920


namespace NUMINAMATH_CALUDE_collinear_points_imply_a_eq_two_l2879_287964

/-- Three points are collinear if the slope between any two pairs of points is the same. -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

/-- Given three points A(a,2), B(5,1), and C(-4,2a) are collinear, prove that a = 2. -/
theorem collinear_points_imply_a_eq_two (a : ℝ) :
  collinear a 2 5 1 (-4) (2*a) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_imply_a_eq_two_l2879_287964


namespace NUMINAMATH_CALUDE_least_odd_prime_factor_of_2047_4_plus_1_l2879_287967

theorem least_odd_prime_factor_of_2047_4_plus_1 (p : Nat) : 
  p = 41 ↔ 
    Prime p ∧ 
    Odd p ∧ 
    p ∣ (2047^4 + 1) ∧ 
    ∀ q : Nat, Prime q → Odd q → q ∣ (2047^4 + 1) → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_least_odd_prime_factor_of_2047_4_plus_1_l2879_287967


namespace NUMINAMATH_CALUDE_sum_of_unit_complex_squares_l2879_287974

theorem sum_of_unit_complex_squares (a b c : ℂ) 
  (h1 : Complex.abs a = 1) 
  (h2 : Complex.abs b = 1) 
  (h3 : Complex.abs c = 1)
  (h4 : a^3 / (b*c) + b^3 / (a*c) + c^3 / (a*b) = 3) : 
  Complex.abs (a + b + c)^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_unit_complex_squares_l2879_287974


namespace NUMINAMATH_CALUDE_function_composition_property_l2879_287981

/-- Given a function f(x) = (ax + b) / (cx + d), prove that if f(f(f(1))) = 1 and f(f(f(2))) = 3, then f(1) = 1. -/
theorem function_composition_property (a b c d : ℝ) :
  let f (x : ℝ) := (a * x + b) / (c * x + d)
  (f (f (f 1)) = 1) → (f (f (f 2)) = 3) → (f 1 = 1) := by
  sorry

end NUMINAMATH_CALUDE_function_composition_property_l2879_287981


namespace NUMINAMATH_CALUDE_largest_solution_reciprocal_power_l2879_287962

noncomputable def largest_solution (x : ℝ) : Prop :=
  (Real.log 5 / Real.log (5 * x^2) + Real.log 5 / Real.log (25 * x^3) = -1) ∧
  ∀ y, (Real.log 5 / Real.log (5 * y^2) + Real.log 5 / Real.log (25 * y^3) = -1) → y ≤ x

theorem largest_solution_reciprocal_power (x : ℝ) :
  largest_solution x → 1 / x^10 = 0.00001 :=
by sorry

end NUMINAMATH_CALUDE_largest_solution_reciprocal_power_l2879_287962


namespace NUMINAMATH_CALUDE_lowest_sum_due_bank_a_l2879_287915

structure Bank where
  name : String
  bankers_discount : ℕ
  true_discount : ℕ

def sum_due (b : Bank) : ℕ := b.bankers_discount - (b.bankers_discount - b.true_discount)

def bank_a : Bank := { name := "A", bankers_discount := 42, true_discount := 36 }
def bank_b : Bank := { name := "B", bankers_discount := 48, true_discount := 41 }
def bank_c : Bank := { name := "C", bankers_discount := 54, true_discount := 47 }

theorem lowest_sum_due_bank_a :
  (sum_due bank_a < sum_due bank_b) ∧
  (sum_due bank_a < sum_due bank_c) ∧
  (sum_due bank_a = 36) :=
by sorry

end NUMINAMATH_CALUDE_lowest_sum_due_bank_a_l2879_287915


namespace NUMINAMATH_CALUDE_supermarket_sales_l2879_287918

-- Define the sales volume function
def P (k b t : ℕ) : ℕ := k * t + b

-- Define the unit price function
def Q (t : ℕ) : ℕ :=
  if t < 25 then t + 20 else 80 - t

-- Define the daily sales revenue function
def Y (k b t : ℕ) : ℕ := P k b t * Q t

theorem supermarket_sales (k b : ℕ) :
  (∀ t : ℕ, 1 ≤ t ∧ t ≤ 30) →
  P k b 5 = 55 →
  P k b 10 = 50 →
  (P k b 20 = 40) ∧
  (∀ t : ℕ, 1 ≤ t ∧ t ≤ 30 → Y k b t ≤ 2395) ∧
  (∃ t : ℕ, 1 ≤ t ∧ t ≤ 30 ∧ Y k b t = 2395) :=
by sorry

end NUMINAMATH_CALUDE_supermarket_sales_l2879_287918


namespace NUMINAMATH_CALUDE_fraction_of_single_men_l2879_287934

theorem fraction_of_single_men (total : ℕ) (h1 : total > 0) :
  let women := (60 : ℚ) / 100 * total
  let men := total - women
  let married := (60 : ℚ) / 100 * total
  let married_men := (1 : ℚ) / 4 * men
  (men - married_men) / men = (3 : ℚ) / 4 :=
by sorry

end NUMINAMATH_CALUDE_fraction_of_single_men_l2879_287934


namespace NUMINAMATH_CALUDE_circle_center_l2879_287952

/-- The center of a circle given by the equation x^2 - 8x + y^2 - 4y = 5 is (4, 2) -/
theorem circle_center (x y : ℝ) : x^2 - 8*x + y^2 - 4*y = 5 → (4, 2) = (x, y) := by
  sorry

end NUMINAMATH_CALUDE_circle_center_l2879_287952
