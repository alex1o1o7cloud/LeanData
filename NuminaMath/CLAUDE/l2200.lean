import Mathlib

namespace NUMINAMATH_CALUDE_additional_charge_per_segment_l2200_220039

/-- Proves that the additional charge per 2/5 of a mile is $0.40 --/
theorem additional_charge_per_segment (initial_fee : ℚ) (trip_distance : ℚ) (total_charge : ℚ) 
  (h1 : initial_fee = 9/4)  -- $2.25
  (h2 : trip_distance = 18/5)  -- 3.6 miles
  (h3 : total_charge = 117/20)  -- $5.85
  : (total_charge - initial_fee) / (trip_distance / (2/5)) = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_additional_charge_per_segment_l2200_220039


namespace NUMINAMATH_CALUDE_z_in_third_quadrant_l2200_220037

/-- Given that i is the imaginary unit and i · z = 1 - 2i, 
    prove that z is located in the third quadrant of the complex plane. -/
theorem z_in_third_quadrant (i z : ℂ) : 
  i * i = -1 →  -- i is the imaginary unit
  i * z = 1 - 2*i →  -- given equation
  z.re < 0 ∧ z.im < 0  -- z is in the third quadrant
  := by sorry

end NUMINAMATH_CALUDE_z_in_third_quadrant_l2200_220037


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l2200_220058

theorem diophantine_equation_solutions :
  {(a, b) : ℕ × ℕ | 12 * a + 11 * b = 2002} =
    {(11, 170), (22, 158), (33, 146), (44, 134), (55, 122), (66, 110),
     (77, 98), (88, 86), (99, 74), (110, 62), (121, 50), (132, 38),
     (143, 26), (154, 14), (165, 2)} := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l2200_220058


namespace NUMINAMATH_CALUDE_percentage_calculations_l2200_220011

theorem percentage_calculations (M N : ℝ) (h : M < N) :
  (100 * (N - M) / M = (N - M) / M * 100) ∧
  (100 * M / N = M / N * 100) :=
by sorry

end NUMINAMATH_CALUDE_percentage_calculations_l2200_220011


namespace NUMINAMATH_CALUDE_equalize_buses_l2200_220088

def students_first_bus : ℕ := 57
def students_second_bus : ℕ := 31

def students_to_move : ℕ := 13

theorem equalize_buses :
  (students_first_bus - students_to_move = students_second_bus + students_to_move) ∧
  (students_first_bus - students_to_move > 0) ∧
  (students_second_bus + students_to_move > 0) :=
by sorry

end NUMINAMATH_CALUDE_equalize_buses_l2200_220088


namespace NUMINAMATH_CALUDE_circle_radius_is_13_main_result_l2200_220064

/-- Represents a circle with tangents -/
structure CircleWithTangents where
  r : ℝ  -- radius of the circle
  ab : ℝ  -- length of tangent AB
  ac : ℝ  -- length of tangent AC
  de : ℝ  -- length of tangent DE perpendicular to BC

/-- Theorem: Given the conditions, the radius of the circle is 13 -/
theorem circle_radius_is_13 (c : CircleWithTangents) 
  (h1 : c.ab = 5) 
  (h2 : c.ac = 12) 
  (h3 : c.de = 13) : 
  c.r = 13 := by
  sorry

/-- The main result -/
theorem main_result : ∃ c : CircleWithTangents, 
  c.ab = 5 ∧ c.ac = 12 ∧ c.de = 13 ∧ c.r = 13 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_is_13_main_result_l2200_220064


namespace NUMINAMATH_CALUDE_max_value_cos_theta_l2200_220015

noncomputable def f (x : ℝ) : ℝ := 3 * Real.cos x - Real.sin x

theorem max_value_cos_theta (θ : ℝ) 
  (h : ∀ x, f x ≤ f θ) : 
  Real.cos θ = 3 * Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_max_value_cos_theta_l2200_220015


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l2200_220082

-- Define the universe U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.log (1 - x)}

-- Define set B
def B : Set ℝ := {x | x^2 - x - 2 < 0}

-- State the theorem
theorem intersection_complement_theorem : 
  B ∩ (Set.compl A) = {x : ℝ | 1 ≤ x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l2200_220082


namespace NUMINAMATH_CALUDE_fourth_number_in_first_set_l2200_220048

theorem fourth_number_in_first_set (x : ℝ) (y : ℝ) : 
  (28 + x + 70 + y + 104) / 5 = 67 →
  (50 + 62 + 97 + 124 + x) / 5 = 75.6 →
  y = 88 := by
sorry

end NUMINAMATH_CALUDE_fourth_number_in_first_set_l2200_220048


namespace NUMINAMATH_CALUDE_unique_set_A_l2200_220066

def M : Set ℤ := {1, 3, 5, 7, 9}

theorem unique_set_A : ∃! A : Set ℤ, A.Nonempty ∧ 
  (∀ a ∈ A, a + 4 ∈ M) ∧ 
  (∀ a ∈ A, a - 4 ∈ M) ∧
  A = {5} := by sorry

end NUMINAMATH_CALUDE_unique_set_A_l2200_220066


namespace NUMINAMATH_CALUDE_triangle_area_from_squares_l2200_220023

theorem triangle_area_from_squares (a b c : ℝ) (ha : a^2 = 36) (hb : b^2 = 225) (hc : c^2 = 324) :
  (1/2) * a * b = 45 :=
sorry

end NUMINAMATH_CALUDE_triangle_area_from_squares_l2200_220023


namespace NUMINAMATH_CALUDE_sequence_properties_l2200_220007

def arithmetic_sequence (n : ℕ) : ℕ := n

def geometric_sequence (n : ℕ) : ℕ := 2^n

def S (n : ℕ) : ℚ := (n^2 + n) / 2

def T (n : ℕ) : ℕ := 2 * (2^n - 1)

theorem sequence_properties :
  (∀ n : ℕ, n ≥ 1 → arithmetic_sequence n = n) ∧
  (∀ n : ℕ, n ≥ 1 → geometric_sequence n = 2^n) ∧
  (∀ n : ℕ, n < 8 → T n + arithmetic_sequence n ≤ 300) ∧
  (T 8 + arithmetic_sequence 8 > 300) :=
sorry

end NUMINAMATH_CALUDE_sequence_properties_l2200_220007


namespace NUMINAMATH_CALUDE_system_solution_l2200_220026

theorem system_solution :
  let f (x y : ℝ) := x^2 - 5*x*y + 6*y^2
  let g (x y : ℝ) := x^2 + y^2
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ),
    (f x₁ y₁ = 0 ∧ g x₁ y₁ = 40) ∧
    (f x₂ y₂ = 0 ∧ g x₂ y₂ = 40) ∧
    (f x₃ y₃ = 0 ∧ g x₃ y₃ = 40) ∧
    (f x₄ y₄ = 0 ∧ g x₄ y₄ = 40) ∧
    x₁ = 4 * Real.sqrt 2 ∧ y₁ = 2 * Real.sqrt 2 ∧
    x₂ = -4 * Real.sqrt 2 ∧ y₂ = -2 * Real.sqrt 2 ∧
    x₃ = 6 ∧ y₃ = 2 ∧
    x₄ = -6 ∧ y₄ = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2200_220026


namespace NUMINAMATH_CALUDE_rectangle_area_and_range_l2200_220028

/-- Represents the area of a rectangle formed by a rope of length 10cm -/
def area (x : ℝ) : ℝ := -x^2 + 5*x

/-- The length of the rope forming the rectangle -/
def ropeLength : ℝ := 10

theorem rectangle_area_and_range :
  ∀ x : ℝ, 0 < x ∧ x < 5 →
  (2 * (x + (ropeLength / 2 - x)) = ropeLength) ∧
  (area x = x * (ropeLength / 2 - x)) :=
sorry

end NUMINAMATH_CALUDE_rectangle_area_and_range_l2200_220028


namespace NUMINAMATH_CALUDE_machine_values_after_two_years_l2200_220072

def machineValue (initialValue : ℝ) (depreciationRate : ℝ) (years : ℕ) : ℝ :=
  initialValue - (initialValue * depreciationRate * years)

def combinedValue (valueA valueB valueC : ℝ) : ℝ :=
  valueA + valueB + valueC

theorem machine_values_after_two_years :
  let machineA := machineValue 8000 0.20 2
  let machineB := machineValue 10000 0.15 2
  let machineC := machineValue 12000 0.10 2
  combinedValue machineA machineB machineC = 21400 := by
  sorry

end NUMINAMATH_CALUDE_machine_values_after_two_years_l2200_220072


namespace NUMINAMATH_CALUDE_sum_of_solutions_squared_equation_l2200_220006

theorem sum_of_solutions_squared_equation (x₁ x₂ : ℝ) :
  (x₁ - 8)^2 = 49 → (x₂ - 8)^2 = 49 → x₁ + x₂ = 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_squared_equation_l2200_220006


namespace NUMINAMATH_CALUDE_industrial_machine_output_l2200_220070

theorem industrial_machine_output (shirts_per_minute : ℕ) 
  (yesterday_minutes : ℕ) (today_shirts : ℕ) (total_shirts : ℕ) :
  yesterday_minutes = 12 →
  today_shirts = 14 →
  total_shirts = 156 →
  yesterday_minutes * shirts_per_minute + today_shirts = total_shirts →
  shirts_per_minute = 11 :=
by sorry

end NUMINAMATH_CALUDE_industrial_machine_output_l2200_220070


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2200_220029

theorem inequality_equivalence (x : ℝ) : 
  (x / (x + 1) + 2 * x / ((x + 1) * (2 * x + 1)) + 3 * x / ((x + 1) * (2 * x + 1) * (3 * x + 1)) > 1) ↔ 
  (x < -1 ∨ (-1/2 < x ∧ x < -1/3)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2200_220029


namespace NUMINAMATH_CALUDE_complex_pure_imaginary_l2200_220056

theorem complex_pure_imaginary (a : ℝ) : 
  (Complex.I * (Complex.I * (a + 1) - 2 * a) / 5 = (Complex.I * (a + Complex.I) / (1 + 2 * Complex.I))) → 
  a = -2 := by
sorry

end NUMINAMATH_CALUDE_complex_pure_imaginary_l2200_220056


namespace NUMINAMATH_CALUDE_journey_time_l2200_220013

/-- Given a journey where:
  * The distance is 320 miles
  * The speed is 50 miles per hour
  * There is a 30-minute stopover
Prove that the total trip time is 6.9 hours -/
theorem journey_time (distance : ℝ) (speed : ℝ) (stopover : ℝ) :
  distance = 320 →
  speed = 50 →
  stopover = 0.5 →
  distance / speed + stopover = 6.9 :=
by sorry

end NUMINAMATH_CALUDE_journey_time_l2200_220013


namespace NUMINAMATH_CALUDE_income_difference_after_raise_l2200_220093

-- Define the annual raise percentage
def annual_raise_percent : ℚ := 8 / 100

-- Define Don's raise amount
def don_raise : ℕ := 800

-- Define Don's wife's raise amount
def wife_raise : ℕ := 840

-- Define function to calculate original salary given the raise amount
def original_salary (raise : ℕ) : ℚ := (raise : ℚ) / annual_raise_percent

-- Define function to calculate new salary after raise
def new_salary (raise : ℕ) : ℚ := original_salary raise + raise

-- Theorem statement
theorem income_difference_after_raise :
  new_salary wife_raise - new_salary don_raise = 540 := by
  sorry

end NUMINAMATH_CALUDE_income_difference_after_raise_l2200_220093


namespace NUMINAMATH_CALUDE_divisibility_by_240_l2200_220096

theorem divisibility_by_240 (p : ℕ) (hp : Nat.Prime p) (hp_ge_7 : p ≥ 7) :
  (240 : ℕ) ∣ (p^4 - 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_240_l2200_220096


namespace NUMINAMATH_CALUDE_dice_cube_properties_l2200_220083

/-- Represents a cube formed from 27 dice in a 3x3x3 configuration -/
structure DiceCube where
  size : Nat
  visible_dice : Nat
  faces_per_die : Nat

/-- Calculates the probability of exactly 25 sixes on the surface of the cube -/
def prob_25_sixes (cube : DiceCube) : ℚ :=
  31 / (2^13 * 3^18)

/-- Calculates the probability of at least one "one" on the surface of the cube -/
def prob_at_least_one_one (cube : DiceCube) : ℚ :=
  1 - (5^6 / (2^2 * 3^18))

/-- Calculates the expected number of sixes showing on the surface of the cube -/
def expected_sixes (cube : DiceCube) : ℚ :=
  9

/-- Calculates the expected sum of the numbers on the surface of the cube -/
def expected_sum (cube : DiceCube) : ℚ :=
  6 - (5^6 / (2 * 3^17))

/-- Main theorem stating the properties of the dice cube -/
theorem dice_cube_properties (cube : DiceCube) 
    (h1 : cube.size = 27) 
    (h2 : cube.visible_dice = 26) 
    (h3 : cube.faces_per_die = 6) : 
  (prob_25_sixes cube = 31 / (2^13 * 3^18)) ∧ 
  (prob_at_least_one_one cube = 1 - 5^6 / (2^2 * 3^18)) ∧ 
  (expected_sixes cube = 9) ∧ 
  (expected_sum cube = 6 - 5^6 / (2 * 3^17)) := by
  sorry

#check dice_cube_properties

end NUMINAMATH_CALUDE_dice_cube_properties_l2200_220083


namespace NUMINAMATH_CALUDE_adam_cat_food_packages_l2200_220095

theorem adam_cat_food_packages : 
  ∀ (c : ℕ), -- c represents the number of packages of cat food
  (10 * c = 7 * 5 + 55) → c = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_adam_cat_food_packages_l2200_220095


namespace NUMINAMATH_CALUDE_otimes_composition_l2200_220073

-- Define the binary operation ⊗
def otimes (x y : ℝ) : ℝ := x^3 + y^3 - x - y

-- Theorem statement
theorem otimes_composition (a b : ℝ) : 
  otimes a (otimes b a) = a^3 + (b^3 + a^3 - b - a)^3 - a - (b^3 + a^3 - b - a) := by
  sorry

end NUMINAMATH_CALUDE_otimes_composition_l2200_220073


namespace NUMINAMATH_CALUDE_death_rate_calculation_l2200_220024

/-- Given a birth rate, net growth rate, and initial population, 
    calculate the death rate. -/
def calculate_death_rate (birth_rate : ℝ) (net_growth_rate : ℝ) 
                          (initial_population : ℝ) : ℝ :=
  birth_rate - net_growth_rate * initial_population

/-- Theorem stating that under the given conditions, 
    the death rate is 16 per certain number of people. -/
theorem death_rate_calculation :
  let birth_rate : ℝ := 52
  let net_growth_rate : ℝ := 0.012
  let initial_population : ℝ := 3000
  calculate_death_rate birth_rate net_growth_rate initial_population = 16 := by
  sorry

#eval calculate_death_rate 52 0.012 3000

end NUMINAMATH_CALUDE_death_rate_calculation_l2200_220024


namespace NUMINAMATH_CALUDE_positive_number_square_sum_l2200_220069

theorem positive_number_square_sum (x : ℝ) : 
  0 < x → x < 15 → x^2 + x = 210 → x = 14 := by sorry

end NUMINAMATH_CALUDE_positive_number_square_sum_l2200_220069


namespace NUMINAMATH_CALUDE_certain_number_problem_l2200_220084

theorem certain_number_problem (x : ℝ) : 
  (10 + x + 60) / 3 = (10 + 40 + 25) / 3 + 5 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l2200_220084


namespace NUMINAMATH_CALUDE_max_value_theorem_l2200_220018

theorem max_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_eq : x^2 - 3*x*y + 4*y^2 - z = 0) :
  (∃ (x' y' z' : ℝ), x' > 0 ∧ y' > 0 ∧ z' > 0 ∧
    x'^2 - 3*x'*y' + 4*y'^2 - z' = 0 ∧
    z'/(x'*y') ≤ z/(x*y) ∧
    x + 2*y - z ≤ x' + 2*y' - z') ∧
  (∀ (x' y' z' : ℝ), x' > 0 → y' > 0 → z' > 0 →
    x'^2 - 3*x'*y' + 4*y'^2 - z' = 0 →
    z'/(x'*y') ≤ z/(x*y) →
    x' + 2*y' - z' ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2200_220018


namespace NUMINAMATH_CALUDE_nth_equation_l2200_220092

theorem nth_equation (n : ℕ) : Real.sqrt ((n + 1) * (n + 3) + 1) = n + 2 := by
  sorry

end NUMINAMATH_CALUDE_nth_equation_l2200_220092


namespace NUMINAMATH_CALUDE_arithmetic_expression_equals_24_l2200_220008

theorem arithmetic_expression_equals_24 : (2 + 4 / 10) * 10 = 24 := by
  sorry

#check arithmetic_expression_equals_24

end NUMINAMATH_CALUDE_arithmetic_expression_equals_24_l2200_220008


namespace NUMINAMATH_CALUDE_inequality_proof_l2200_220054

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 1) : 
  Real.sqrt (a * b / (c + a * b)) + 
  Real.sqrt (b * c / (a + b * c)) + 
  Real.sqrt (a * c / (b + a * c)) ≤ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2200_220054


namespace NUMINAMATH_CALUDE_overtime_rate_calculation_l2200_220009

/-- Calculate the overtime rate given the following conditions:
  * Regular hourly rate
  * Total weekly pay
  * Total hours worked
  * Overtime hours worked
-/
def calculate_overtime_rate (regular_rate : ℚ) (total_pay : ℚ) (total_hours : ℕ) (overtime_hours : ℕ) : ℚ :=
  let regular_hours := total_hours - overtime_hours
  let regular_pay := regular_rate * regular_hours
  let overtime_pay := total_pay - regular_pay
  overtime_pay / overtime_hours

theorem overtime_rate_calculation :
  let regular_rate : ℚ := 60 / 100  -- 60 cents per hour
  let total_pay : ℚ := 3240 / 100   -- $32.40
  let total_hours : ℕ := 50
  let overtime_hours : ℕ := 8
  calculate_overtime_rate regular_rate total_pay total_hours overtime_hours = 90 / 100 := by
    sorry

#eval calculate_overtime_rate (60 / 100) (3240 / 100) 50 8

end NUMINAMATH_CALUDE_overtime_rate_calculation_l2200_220009


namespace NUMINAMATH_CALUDE_assignment_count_assignment_count_proof_l2200_220010

theorem assignment_count : ℕ → Prop :=
  fun total_assignments =>
    ∃ (initial_hours : ℕ),
      -- Initial plan: 6 assignments per hour for initial_hours
      6 * initial_hours = total_assignments ∧
      -- New plan: 2 hours at 6 per hour, then 8 per hour for (initial_hours - 5) hours
      2 * 6 + 8 * (initial_hours - 5) = total_assignments ∧
      -- Total assignments is 84
      total_assignments = 84

-- The proof of this theorem would show that the conditions are satisfied
-- and the total number of assignments is indeed 84
theorem assignment_count_proof : assignment_count 84 := by
  sorry

#check assignment_count_proof

end NUMINAMATH_CALUDE_assignment_count_assignment_count_proof_l2200_220010


namespace NUMINAMATH_CALUDE_combination_problem_l2200_220042

theorem combination_problem (m : ℕ) : 
  (1 : ℚ) / Nat.choose 5 m - (1 : ℚ) / Nat.choose 6 m = (7 : ℚ) / (10 * Nat.choose 7 m) → 
  Nat.choose 8 m = 28 := by
  sorry

end NUMINAMATH_CALUDE_combination_problem_l2200_220042


namespace NUMINAMATH_CALUDE_volume_of_T_l2200_220030

/-- The solid T in ℝ³ -/
def T : Set (ℝ × ℝ × ℝ) :=
  {p : ℝ × ℝ × ℝ | let (x, y, z) := p
                   (|x| + |y| ≤ 2) ∧ (|x| + |z| ≤ 2) ∧ (|y| + |z| ≤ 2)}

/-- The volume of a set in ℝ³ -/
noncomputable def volume (S : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

/-- The volume of T is 1664/81 -/
theorem volume_of_T : volume T = 1664 / 81 := by sorry

end NUMINAMATH_CALUDE_volume_of_T_l2200_220030


namespace NUMINAMATH_CALUDE_road_trip_speed_l2200_220001

/-- Road trip problem -/
theorem road_trip_speed (total_distance : ℝ) (jenna_distance : ℝ) (friend_distance : ℝ)
  (jenna_speed : ℝ) (total_time : ℝ) (num_breaks : ℕ) (break_duration : ℝ) :
  total_distance = jenna_distance + friend_distance →
  jenna_distance = 200 →
  friend_distance = 100 →
  jenna_speed = 50 →
  total_time = 10 →
  num_breaks = 2 →
  break_duration = 0.5 →
  ∃ (friend_speed : ℝ), friend_speed = 20 ∧ 
    total_time = jenna_distance / jenna_speed + friend_distance / friend_speed + num_breaks * break_duration :=
by sorry


end NUMINAMATH_CALUDE_road_trip_speed_l2200_220001


namespace NUMINAMATH_CALUDE_smallest_solution_abs_equation_l2200_220012

theorem smallest_solution_abs_equation :
  ∀ x : ℝ, |x - 3| = 8 → x ≥ -5 ∧ |-5 - 3| = 8 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_abs_equation_l2200_220012


namespace NUMINAMATH_CALUDE_solution_bounds_and_expression_l2200_220089

def system_of_equations (x y m : ℝ) : Prop :=
  3 * (x + 1) / 2 + y = 2 ∧ 3 * x - m = 2 * y

theorem solution_bounds_and_expression (x y m : ℝ) 
  (h_system : system_of_equations x y m) 
  (h_x_bound : x ≤ 1) 
  (h_y_bound : y ≤ 1) : 
  (-3 ≤ m ∧ m ≤ 5) ∧ 
  |x - 1| + |y - 1| + |m + 3| + |m - 5| - |x + y - 2| = 8 := by
  sorry

end NUMINAMATH_CALUDE_solution_bounds_and_expression_l2200_220089


namespace NUMINAMATH_CALUDE_correct_operation_l2200_220025

theorem correct_operation (a b : ℝ) : 3 * a^2 * b - b * a^2 = 2 * a^2 * b := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l2200_220025


namespace NUMINAMATH_CALUDE_hispanic_west_percentage_l2200_220050

/-- Represents the population data for a specific ethnic group across regions -/
structure PopulationData :=
  (ne : ℕ) (mw : ℕ) (south : ℕ) (west : ℕ)

/-- Calculates the total population across all regions -/
def total_population (data : PopulationData) : ℕ :=
  data.ne + data.mw + data.south + data.west

/-- Calculates the percentage of population in the West, rounded to the nearest percent -/
def west_percentage (data : PopulationData) : ℕ :=
  (data.west * 100 + (total_population data) / 2) / (total_population data)

/-- The given Hispanic population data for 1990 in millions -/
def hispanic_data : PopulationData :=
  { ne := 4, mw := 5, south := 12, west := 20 }

theorem hispanic_west_percentage :
  west_percentage hispanic_data = 49 := by sorry

end NUMINAMATH_CALUDE_hispanic_west_percentage_l2200_220050


namespace NUMINAMATH_CALUDE_integer_fraction_sum_l2200_220076

theorem integer_fraction_sum (n : ℕ) : n > 0 →
  (∃ (x y z : ℤ), x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ 
    x + y + z = 0 ∧ 
    (1 : ℚ) / x + (1 : ℚ) / y + (1 : ℚ) / z = (1 : ℚ) / n) ↔ 
  ∃ (k : ℕ), n = 2 * k ∧ k > 0 :=
by sorry

end NUMINAMATH_CALUDE_integer_fraction_sum_l2200_220076


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l2200_220032

theorem solution_set_equivalence :
  ∀ x : ℝ, (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) := by sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l2200_220032


namespace NUMINAMATH_CALUDE_solution_set_correct_l2200_220090

/-- The set of solutions to the equation 1/(x^2 + 13x - 12) + 1/(x^2 + 4x - 12) + 1/(x^2 - 11x - 12) = 0 -/
def solution_set : Set ℝ := {1, -12, 4, -3}

/-- The equation to be solved -/
def equation (x : ℝ) : Prop :=
  1 / (x^2 + 13*x - 12) + 1 / (x^2 + 4*x - 12) + 1 / (x^2 - 11*x - 12) = 0

theorem solution_set_correct :
  ∀ x : ℝ, equation x ↔ x ∈ solution_set := by sorry

end NUMINAMATH_CALUDE_solution_set_correct_l2200_220090


namespace NUMINAMATH_CALUDE_triangle_altitude_l2200_220086

theorem triangle_altitude (area : ℝ) (base : ℝ) (altitude : ℝ) : 
  area = 500 → base = 50 → area = (1/2) * base * altitude → altitude = 20 := by
  sorry

end NUMINAMATH_CALUDE_triangle_altitude_l2200_220086


namespace NUMINAMATH_CALUDE_measure_six_with_special_ruler_l2200_220053

/-- A ruler with marks at specific positions -/
structure Ruler :=
  (marks : List ℝ)

/-- Definition of a ruler with marks at 0, 2, and 5 -/
def specialRuler : Ruler :=
  { marks := [0, 2, 5] }

/-- A function to check if a length can be measured using a given ruler -/
def canMeasure (r : Ruler) (length : ℝ) : Prop :=
  ∃ (a b : ℝ), a ∈ r.marks ∧ b ∈ r.marks ∧ (b - a = length ∨ a - b = length)

/-- Theorem stating that the special ruler can measure a segment of length 6 -/
theorem measure_six_with_special_ruler :
  canMeasure specialRuler 6 := by
  sorry


end NUMINAMATH_CALUDE_measure_six_with_special_ruler_l2200_220053


namespace NUMINAMATH_CALUDE_starting_number_proof_l2200_220016

theorem starting_number_proof (x : ℕ) : 
  (∃ (l : List ℕ), l.length = 12 ∧ 
    (∀ n ∈ l, x ≤ n ∧ n ≤ 47 ∧ n % 3 = 0) ∧
    (∀ m, x ≤ m ∧ m ≤ 47 ∧ m % 3 = 0 → m ∈ l)) ↔ 
  x = 12 := by
  sorry

#check starting_number_proof

end NUMINAMATH_CALUDE_starting_number_proof_l2200_220016


namespace NUMINAMATH_CALUDE_jason_pokemon_cards_l2200_220094

theorem jason_pokemon_cards 
  (cards_given_away : ℕ) 
  (cards_remaining : ℕ) 
  (h1 : cards_given_away = 9) 
  (h2 : cards_remaining = 4) : 
  cards_given_away + cards_remaining = 13 := by
sorry

end NUMINAMATH_CALUDE_jason_pokemon_cards_l2200_220094


namespace NUMINAMATH_CALUDE_rectangular_garden_length_l2200_220000

/-- Theorem: For a rectangular garden with a perimeter of 600 m and a breadth of 95 m, the length is 205 m. -/
theorem rectangular_garden_length 
  (perimeter : ℝ) 
  (breadth : ℝ) 
  (h1 : perimeter = 600) 
  (h2 : breadth = 95) :
  2 * (breadth + 205) = perimeter := by
  sorry

end NUMINAMATH_CALUDE_rectangular_garden_length_l2200_220000


namespace NUMINAMATH_CALUDE_janet_total_cost_l2200_220059

/-- Calculates the total cost for Janet's group at the waterpark -/
def waterpark_cost (adult_price : ℚ) (group_size : ℕ) (child_count : ℕ) (soda_price : ℚ) : ℚ :=
  let child_price := adult_price / 2
  let adult_count := group_size - child_count
  let total_ticket_cost := adult_price * adult_count + child_price * child_count
  let discount := total_ticket_cost * (1/5)
  (total_ticket_cost - discount) + soda_price

/-- Proves that Janet's total cost is $197 -/
theorem janet_total_cost : 
  waterpark_cost 30 10 4 5 = 197 := by
  sorry

end NUMINAMATH_CALUDE_janet_total_cost_l2200_220059


namespace NUMINAMATH_CALUDE_smallest_third_term_of_geometric_progression_l2200_220071

/-- Given an arithmetic progression with first term 7, prove that the smallest
    possible value for the third term of the resulting geometric progression is 3.752 -/
theorem smallest_third_term_of_geometric_progression
  (a b c : ℝ)
  (h_arithmetic : ∃ (d : ℝ), a = 7 ∧ b = 7 + d ∧ c = 7 + 2*d)
  (h_geometric : ∃ (r : ℝ), (7 : ℝ) * r = b - 3 ∧ (b - 3) * r = c + 15) :
  ∃ (x : ℝ), (∀ (y : ℝ), (7 : ℝ) * (b - 3) = (b - 3) * (c + 15) → c + 15 ≥ x) ∧ c + 15 ≥ 3.752 :=
sorry

end NUMINAMATH_CALUDE_smallest_third_term_of_geometric_progression_l2200_220071


namespace NUMINAMATH_CALUDE_expression_evaluation_l2200_220081

theorem expression_evaluation :
  let x : ℝ := 3 + Real.sqrt 2
  (1 - 1 / (x + 3)) / ((x + 2) / (x^2 - 9)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2200_220081


namespace NUMINAMATH_CALUDE_degree_three_polynomial_l2200_220035

/-- The polynomial f(x) -/
def f (x : ℝ) : ℝ := 2 - 15*x + 4*x^2 - 5*x^3 + 6*x^4

/-- The polynomial g(x) -/
def g (x : ℝ) : ℝ := 4 - 3*x - 7*x^3 + 10*x^4

/-- The combined polynomial h(x) = f(x) + c*g(x) -/
def h (c : ℝ) (x : ℝ) : ℝ := f x + c * g x

/-- The theorem stating that c = -3/5 makes h(x) a polynomial of degree 3 -/
theorem degree_three_polynomial :
  ∃ (c : ℝ), c = -3/5 ∧ 
  (∀ (x : ℝ), h c x = 2 + (-15 - 3*c)*x + (4 - 0*c)*x^2 + (-5 - 7*c)*x^3) :=
sorry

end NUMINAMATH_CALUDE_degree_three_polynomial_l2200_220035


namespace NUMINAMATH_CALUDE_flower_expense_proof_l2200_220065

/-- Calculates the total expense for flowers given the quantities and price per flower -/
def totalExpense (tulips carnations roses : ℕ) (pricePerFlower : ℕ) : ℕ :=
  (tulips + carnations + roses) * pricePerFlower

/-- Proves that the total expense for the given flower quantities and price is 1890 -/
theorem flower_expense_proof :
  totalExpense 250 375 320 2 = 1890 := by
  sorry

end NUMINAMATH_CALUDE_flower_expense_proof_l2200_220065


namespace NUMINAMATH_CALUDE_three_digit_sum_property_l2200_220074

def is_valid_number (N : ℕ) : Prop :=
  ∃ (a b c : ℕ),
    N = 100 * a + 10 * b + c ∧
    a < 10 ∧ b < 10 ∧ c < 10 ∧
    (a + b + 1 = (a + b + c) / 3 ∨
     a + (b + 1) + 1 = (a + b + c) / 3)

theorem three_digit_sum_property :
  ∀ N : ℕ, is_valid_number N → (N = 207 ∨ N = 117 ∨ N = 108) :=
by sorry

end NUMINAMATH_CALUDE_three_digit_sum_property_l2200_220074


namespace NUMINAMATH_CALUDE_midpoint_sum_midpoint_sum_specific_l2200_220078

/-- Given a line segment with endpoints (3, 4) and (9, 18), 
    the sum of the coordinates of its midpoint is 17. -/
theorem midpoint_sum : ℝ → ℝ → ℝ → ℝ → ℝ := fun x₁ y₁ x₂ y₂ =>
  let midpoint_x := (x₁ + x₂) / 2
  let midpoint_y := (y₁ + y₂) / 2
  midpoint_x + midpoint_y

#check midpoint_sum 3 4 9 18 = 17

theorem midpoint_sum_specific : midpoint_sum 3 4 9 18 = 17 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_sum_midpoint_sum_specific_l2200_220078


namespace NUMINAMATH_CALUDE_incorrect_proposition_l2200_220077

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)

-- State the theorem
theorem incorrect_proposition
  (m n : Line) (α β : Plane)
  (h1 : parallel m α)
  (h2 : perpendicular n β)
  (h3 : perpendicular_planes α β) :
  ¬ (parallel_lines m n) :=
sorry

end NUMINAMATH_CALUDE_incorrect_proposition_l2200_220077


namespace NUMINAMATH_CALUDE_fifth_root_of_unity_l2200_220047

theorem fifth_root_of_unity (p q r s t m : ℂ) :
  p ≠ 0 →
  p * m^4 + q * m^3 + r * m^2 + s * m + t = 0 →
  q * m^4 + r * m^3 + s * m^2 + t * m + p = 0 →
  m^5 = 1 :=
by sorry

end NUMINAMATH_CALUDE_fifth_root_of_unity_l2200_220047


namespace NUMINAMATH_CALUDE_f_equals_negative_two_iff_b_equals_negative_one_l2200_220005

def f (x : ℝ) : ℝ := 5 * x + 3

theorem f_equals_negative_two_iff_b_equals_negative_one :
  ∀ b : ℝ, f b = -2 ↔ b = -1 := by sorry

end NUMINAMATH_CALUDE_f_equals_negative_two_iff_b_equals_negative_one_l2200_220005


namespace NUMINAMATH_CALUDE_f_range_l2200_220060

noncomputable def f (x : ℝ) : ℝ :=
  (Real.sin x ^ 3 + 5 * Real.sin x ^ 2 + 4 * Real.sin x + 2 * Real.cos x ^ 2 - 9) / (Real.sin x - 1)

theorem f_range :
  Set.range (fun (x : ℝ) => f x) = Set.Icc (-12) 0 :=
by sorry

end NUMINAMATH_CALUDE_f_range_l2200_220060


namespace NUMINAMATH_CALUDE_min_teams_for_employees_l2200_220004

theorem min_teams_for_employees (total_employees : ℕ) (max_team_size : ℕ) (h1 : total_employees = 36) (h2 : max_team_size = 12) : 
  (total_employees + max_team_size - 1) / max_team_size = 3 :=
by sorry

end NUMINAMATH_CALUDE_min_teams_for_employees_l2200_220004


namespace NUMINAMATH_CALUDE_missing_sale_is_correct_l2200_220041

/-- Calculates the missing sale amount given the other 5 sales and the target average -/
def calculate_missing_sale (sale1 sale3 sale4 sale5 sale6 average : ℝ) : ℝ :=
  6 * average - (sale1 + sale3 + sale4 + sale5 + sale6)

/-- Proves that the calculated missing sale is correct given the problem conditions -/
theorem missing_sale_is_correct (sale1 sale3 sale4 sale5 sale6 average : ℝ) 
  (h1 : sale1 = 5420)
  (h3 : sale3 = 6200)
  (h4 : sale4 = 6350)
  (h5 : sale5 = 6500)
  (h6 : sale6 = 7070)
  (havg : average = 6200) :
  calculate_missing_sale sale1 sale3 sale4 sale5 sale6 average = 5660 := by
  sorry

#eval calculate_missing_sale 5420 6200 6350 6500 7070 6200

end NUMINAMATH_CALUDE_missing_sale_is_correct_l2200_220041


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2200_220068

-- Define the complex number z
variable (z : ℂ)

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_equation_solution :
  (1 + i) * z = Complex.abs (1 + Real.sqrt 3 * i) →
  z = 1 - i :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2200_220068


namespace NUMINAMATH_CALUDE_absolute_value_sum_equality_l2200_220097

theorem absolute_value_sum_equality (x y : ℝ) : 
  (|x + y| = |x| + |y|) ↔ x * y ≥ 0 := by sorry

end NUMINAMATH_CALUDE_absolute_value_sum_equality_l2200_220097


namespace NUMINAMATH_CALUDE_tan_ratio_sum_l2200_220051

theorem tan_ratio_sum (x y : ℝ) 
  (h1 : (Real.sin x / Real.cos y) + (Real.sin y / Real.cos x) = 2)
  (h2 : (Real.cos x / Real.sin y) + (Real.cos y / Real.sin x) = 3) :
  (Real.tan x / Real.tan y) + (Real.tan y / Real.tan x) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_tan_ratio_sum_l2200_220051


namespace NUMINAMATH_CALUDE_greatest_lower_bound_l2200_220091

theorem greatest_lower_bound (x y : ℝ) (h1 : x ≠ y) (h2 : x * y = 2) :
  ((x + y)^2 - 6) * ((x - y)^2 + 8) / (x - y)^2 ≥ 18 ∧
  ∀ C > 18, ∃ x y : ℝ, x ≠ y ∧ x * y = 2 ∧
    ((x + y)^2 - 6) * ((x - y)^2 + 8) / (x - y)^2 < C :=
by sorry

end NUMINAMATH_CALUDE_greatest_lower_bound_l2200_220091


namespace NUMINAMATH_CALUDE_simplify_fraction_division_l2200_220044

theorem simplify_fraction_division (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) :
  ((1 - x) / x) / ((1 - x) / x^2) = x := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_division_l2200_220044


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2200_220080

-- Define the function f
noncomputable def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_function_properties
  (a b c : ℝ)
  (ha : a ≠ 0)
  (hno_roots : ∀ x : ℝ, f a b c x ≠ x) :
  (a > 0 → ∀ x : ℝ, f a b c (f a b c x) > x) ∧
  (¬ (a < 0 → ∃ x : ℝ, f a b c (f a b c x) > x)) ∧
  (∀ x : ℝ, f a b c (f a b c x) ≠ x) ∧
  (a + b + c = 0 → ∀ x : ℝ, f a b c (f a b c x) < x) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l2200_220080


namespace NUMINAMATH_CALUDE_non_multiples_count_is_412_l2200_220031

/-- The count of three-digit numbers that are not multiples of 3, 5, or 7 -/
def non_multiples_count : ℕ :=
  let total_three_digit := 999 - 100 + 1
  let multiples_3 := (999 - 100) / 3 + 1
  let multiples_5 := (995 - 100) / 5 + 1
  let multiples_7 := (994 - 105) / 7 + 1
  let multiples_15 := (990 - 105) / 15 + 1
  let multiples_21 := (987 - 105) / 21 + 1
  let multiples_35 := (980 - 105) / 35 + 1
  let multiples_105 := (945 - 105) / 105 + 1
  let total_multiples := multiples_3 + multiples_5 + multiples_7 - multiples_15 - multiples_21 - multiples_35 + multiples_105
  total_three_digit - total_multiples

theorem non_multiples_count_is_412 : non_multiples_count = 412 := by
  sorry

end NUMINAMATH_CALUDE_non_multiples_count_is_412_l2200_220031


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fourth_term_l2200_220027

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_fourth_term
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_a1 : a 1 = 2)
  (h_a2 : a 2 = 4) :
  a 4 = 8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fourth_term_l2200_220027


namespace NUMINAMATH_CALUDE_cake_piece_volume_l2200_220038

/-- The volume of a piece of cake -/
theorem cake_piece_volume (diameter : ℝ) (thickness : ℝ) (num_pieces : ℕ) 
  (h1 : diameter = 16)
  (h2 : thickness = 1/2)
  (h3 : num_pieces = 8) :
  (π * (diameter/2)^2 * thickness) / num_pieces = 4 * π := by
  sorry

end NUMINAMATH_CALUDE_cake_piece_volume_l2200_220038


namespace NUMINAMATH_CALUDE_prob_at_least_one_red_l2200_220079

theorem prob_at_least_one_red (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ) :
  total_balls = 5 →
  red_balls = 2 →
  white_balls = 3 →
  (red_balls + white_balls = total_balls) →
  (probability_at_least_one_red : ℚ) =
    1 - (white_balls / total_balls * (white_balls - 1) / (total_balls - 1)) →
  probability_at_least_one_red = 7 / 10 := by
  sorry

#check prob_at_least_one_red

end NUMINAMATH_CALUDE_prob_at_least_one_red_l2200_220079


namespace NUMINAMATH_CALUDE_ac_negative_l2200_220022

theorem ac_negative (a b c d : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0)
  (h5 : a / b + c / d = (a + c) / (b + d)) : a * c < 0 := by
  sorry

end NUMINAMATH_CALUDE_ac_negative_l2200_220022


namespace NUMINAMATH_CALUDE_size_relationship_l2200_220055

theorem size_relationship (a b c : ℝ) 
  (ha : a = Real.sqrt 3)
  (hb : b = Real.sqrt 15 - Real.sqrt 7)
  (hc : c = Real.sqrt 11 - Real.sqrt 3) :
  a > c ∧ c > b := by
  sorry

end NUMINAMATH_CALUDE_size_relationship_l2200_220055


namespace NUMINAMATH_CALUDE_girls_percentage_l2200_220062

/-- The percentage of girls in a school with 150 total students and 60 boys is 60%. -/
theorem girls_percentage (total : ℕ) (boys : ℕ) (h1 : total = 150) (h2 : boys = 60) :
  (total - boys : ℚ) / total * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_girls_percentage_l2200_220062


namespace NUMINAMATH_CALUDE_initial_milk_amount_l2200_220002

/-- Proves that the initial amount of milk is 10 liters given the conditions of the problem -/
theorem initial_milk_amount (initial_water_content : Real) 
                            (final_water_content : Real)
                            (pure_milk_added : Real) :
  initial_water_content = 0.05 →
  final_water_content = 0.02 →
  pure_milk_added = 15 →
  ∃ (initial_milk : Real),
    initial_milk = 10 ∧
    initial_water_content * initial_milk = 
    final_water_content * (initial_milk + pure_milk_added) :=
by
  sorry


end NUMINAMATH_CALUDE_initial_milk_amount_l2200_220002


namespace NUMINAMATH_CALUDE_car_speed_second_hour_l2200_220067

/-- Proves that given a car's average speed and first hour speed, we can determine the second hour speed -/
theorem car_speed_second_hour 
  (average_speed : ℝ) 
  (first_hour_speed : ℝ) 
  (h1 : average_speed = 55) 
  (h2 : first_hour_speed = 65) : 
  ∃ (second_hour_speed : ℝ), 
    second_hour_speed = 45 ∧ 
    average_speed = (first_hour_speed + second_hour_speed) / 2 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_second_hour_l2200_220067


namespace NUMINAMATH_CALUDE_certain_number_is_fifteen_l2200_220017

/-- The number of Doberman puppies -/
def doberman_puppies : ℕ := 20

/-- The number of Schnauzers -/
def schnauzers : ℕ := 55

/-- The certain number calculated from the given expression -/
def certain_number : ℤ := 3 * doberman_puppies - 5 + (doberman_puppies - schnauzers)

/-- Theorem stating that the certain number equals 15 -/
theorem certain_number_is_fifteen : certain_number = 15 := by sorry

end NUMINAMATH_CALUDE_certain_number_is_fifteen_l2200_220017


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2200_220085

/-- The function f(x) = -x³ + 2ax -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + 2*a*x

/-- f is monotonically decreasing on (-∞, 1] -/
def is_monotone_decreasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, x ≤ y → y ≤ 1 → f a x ≥ f a y

theorem necessary_but_not_sufficient :
  (∀ a : ℝ, is_monotone_decreasing_on_interval a → a < 3/2) ∧
  (∃ a : ℝ, a < 3/2 ∧ ¬is_monotone_decreasing_on_interval a) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2200_220085


namespace NUMINAMATH_CALUDE_sum_of_roots_l2200_220045

/-- Given distinct real numbers p, q, r, s such that
    x^2 - 12px - 13q = 0 has roots r and s, and
    x^2 - 12rx - 13s = 0 has roots p and q,
    prove that p + q + r + s = 2028 -/
theorem sum_of_roots (p q r s : ℝ) 
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s)
  (h_roots1 : ∀ x, x^2 - 12*p*x - 13*q = 0 ↔ x = r ∨ x = s)
  (h_roots2 : ∀ x, x^2 - 12*r*x - 13*s = 0 ↔ x = p ∨ x = q) :
  p + q + r + s = 2028 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l2200_220045


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2200_220033

theorem inequality_solution_set (x : ℝ) :
  (-6 * x^2 - x + 2 < 0) ↔ (x < -2/3 ∨ x > 1/2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2200_220033


namespace NUMINAMATH_CALUDE_sum_of_cubes_consecutive_integers_l2200_220098

theorem sum_of_cubes_consecutive_integers :
  ∃ n : ℤ, n^3 + (n + 1)^3 = 9 :=
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_consecutive_integers_l2200_220098


namespace NUMINAMATH_CALUDE_four_digit_harmonious_divisible_by_11_l2200_220052

/-- A four-digit harmonious number with 'a' as the first and last digit, and 'b' as the second and third digit. -/
def four_digit_harmonious (a b : ℕ) : ℕ := 1000 * a + 100 * b + 10 * b + a

/-- Proposition: All four-digit harmonious numbers are divisible by 11. -/
theorem four_digit_harmonious_divisible_by_11 (a b : ℕ) :
  ∃ k : ℕ, four_digit_harmonious a b = 11 * k := by
  sorry

end NUMINAMATH_CALUDE_four_digit_harmonious_divisible_by_11_l2200_220052


namespace NUMINAMATH_CALUDE_factorial_equality_l2200_220036

theorem factorial_equality : 7 * 6 * 4 * 2160 = Nat.factorial 9 := by
  sorry

end NUMINAMATH_CALUDE_factorial_equality_l2200_220036


namespace NUMINAMATH_CALUDE_g_value_at_five_sixths_l2200_220063

/-- Given a function f and g with specific properties, prove that g(5/6) = -√3/2 -/
theorem g_value_at_five_sixths 
  (f : ℝ → ℝ) 
  (g : ℝ → ℝ) 
  (a : ℝ) 
  (h1 : a > 0)
  (h2 : ∀ x, f x = Real.sqrt 2 * Real.sin (a * x + π / 4))
  (h3 : ∀ x, x ≥ 0 → g x = g (x - 1))
  (h4 : ∀ x, x < 0 → g x = Real.sin (a * x))
  (h5 : ∃ T, T > 0 ∧ T = 1 ∧ ∀ x, f (x + T) = f x) :
  g (5/6) = -Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_g_value_at_five_sixths_l2200_220063


namespace NUMINAMATH_CALUDE_union_of_A_and_B_complement_of_intersection_A_and_B_l2200_220014

-- Define the sets A and B
def A : Set ℝ := {x | -5 ≤ x ∧ x ≤ -1}
def B : Set ℝ := {x | x + 4 ≥ 0}

-- Theorem for A ∪ B
theorem union_of_A_and_B : A ∪ B = {x : ℝ | x ≥ -5} := by sorry

-- Theorem for ∁ᵤ(A ∩ B)
theorem complement_of_intersection_A_and_B : (A ∩ B)ᶜ = {x : ℝ | x < -4 ∨ x > -1} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_complement_of_intersection_A_and_B_l2200_220014


namespace NUMINAMATH_CALUDE_last_segment_speed_l2200_220057

theorem last_segment_speed (total_distance : ℝ) (total_time : ℝ) 
  (first_segment_speed : ℝ) (second_segment_speed : ℝ) :
  total_distance = 120 →
  total_time = 120 →
  first_segment_speed = 50 →
  second_segment_speed = 70 →
  ∃ (last_segment_speed : ℝ),
    last_segment_speed = 60 ∧
    (first_segment_speed * (total_time / 3) + 
     second_segment_speed * (total_time / 3) + 
     last_segment_speed * (total_time / 3)) = total_distance :=
by sorry

end NUMINAMATH_CALUDE_last_segment_speed_l2200_220057


namespace NUMINAMATH_CALUDE_complex_number_location_l2200_220021

theorem complex_number_location :
  ∀ z : ℂ, z * (2 + Complex.I) = 1 + 3 * Complex.I →
  Complex.re z > 0 ∧ Complex.im z > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_location_l2200_220021


namespace NUMINAMATH_CALUDE_solve_equation_1_solve_equation_2_solve_equation_3_l2200_220040

-- Equation 1: (x-2)^2 = 25
theorem solve_equation_1 : 
  ∃ x₁ x₂ : ℝ, (x₁ - 2)^2 = 25 ∧ (x₂ - 2)^2 = 25 ∧ x₁ = 7 ∧ x₂ = -3 :=
sorry

-- Equation 2: x^2 + 4x + 3 = 0
theorem solve_equation_2 : 
  ∃ x₁ x₂ : ℝ, x₁^2 + 4*x₁ + 3 = 0 ∧ x₂^2 + 4*x₂ + 3 = 0 ∧ x₁ = -3 ∧ x₂ = -1 :=
sorry

-- Equation 3: 2x^2 + 4x - 1 = 0
theorem solve_equation_3 : 
  ∃ x₁ x₂ : ℝ, 2*x₁^2 + 4*x₁ - 1 = 0 ∧ 2*x₂^2 + 4*x₂ - 1 = 0 ∧ 
  x₁ = (-2 + Real.sqrt 6) / 2 ∧ x₂ = (-2 - Real.sqrt 6) / 2 :=
sorry

end NUMINAMATH_CALUDE_solve_equation_1_solve_equation_2_solve_equation_3_l2200_220040


namespace NUMINAMATH_CALUDE_triangle_max_perimeter_l2200_220049

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x ^ 2 + Real.sin x * Real.cos x - Real.sqrt 3 / 2

theorem triangle_max_perimeter 
  (A : ℝ) 
  (h_acute : 0 < A ∧ A < π / 2)
  (h_f_A : f A = Real.sqrt 3 / 2)
  (h_a : ∀ (a b c : ℝ), a = 2 → a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A) :
  ∃ (b c : ℝ), 2 + b + c ≤ 6 ∧ 
    ∀ (b' c' : ℝ), 2 + b' + c' ≤ 2 + b + c := by
  sorry

end NUMINAMATH_CALUDE_triangle_max_perimeter_l2200_220049


namespace NUMINAMATH_CALUDE_quadratic_minimum_l2200_220019

def f (x : ℝ) := 5 * x^2 - 15 * x + 2

theorem quadratic_minimum :
  ∃ (x : ℝ), ∀ (y : ℝ), f y ≥ f x ∧ f x = -9.25 :=
sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l2200_220019


namespace NUMINAMATH_CALUDE_certain_amount_proof_l2200_220020

theorem certain_amount_proof : 
  let x : ℝ := 10
  let percentage_of_500 : ℝ := 0.05 * 500
  let percentage_of_x : ℝ := 0.5 * x
  percentage_of_500 - percentage_of_x = 20 := by
sorry

end NUMINAMATH_CALUDE_certain_amount_proof_l2200_220020


namespace NUMINAMATH_CALUDE_student_line_count_l2200_220099

theorem student_line_count :
  ∀ (n : ℕ),
    n > 0 →
    (∃ (eunjung_pos yoojung_pos : ℕ),
      eunjung_pos = 5 ∧
      yoojung_pos = n ∧
      yoojung_pos - eunjung_pos - 1 = 8) →
    n = 14 := by
  sorry

end NUMINAMATH_CALUDE_student_line_count_l2200_220099


namespace NUMINAMATH_CALUDE_sum_equals_250_l2200_220061

theorem sum_equals_250 : 157 + 18 + 32 + 43 = 250 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_250_l2200_220061


namespace NUMINAMATH_CALUDE_average_speed_multi_segment_l2200_220003

/-- Calculates the average speed of a journey with multiple segments --/
theorem average_speed_multi_segment 
  (t1 t2 t3 : ℝ) 
  (v1 v2 v3 : ℝ) 
  (h1 : t1 = 5)
  (h2 : t2 = 3)
  (h3 : t3 = 2)
  (h4 : v1 = 40)
  (h5 : v2 = 80)
  (h6 : v3 = 60) :
  (t1 * v1 + t2 * v2 + t3 * v3) / (t1 + t2 + t3) = 56 := by
  sorry

#check average_speed_multi_segment

end NUMINAMATH_CALUDE_average_speed_multi_segment_l2200_220003


namespace NUMINAMATH_CALUDE_sin_alpha_value_l2200_220043

-- Define the angle α
def α : Real := sorry

-- Define the point P on the terminal side of α
def P : ℝ × ℝ := (-2, 1)

-- Theorem statement
theorem sin_alpha_value :
  (α.sin = -2 * Real.sqrt 5 / 5) ∧
  (α.cos ≥ 0) ∧
  (α.sin * 2 + α.cos * (-2) = 0) := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l2200_220043


namespace NUMINAMATH_CALUDE_equation_solution_l2200_220075

theorem equation_solution :
  ∃ x : ℝ, (4 / 7) * (1 / 9) * x = 14 ∧ x = 220.5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2200_220075


namespace NUMINAMATH_CALUDE_student_travel_distance_l2200_220034

/-- Proves that given a total distance of 105.00000000000003 km, where 1/5 is traveled by foot
    and 2/3 is traveled by bus, the remaining distance traveled by car is 14.000000000000002 km. -/
theorem student_travel_distance (total_distance : ℝ) 
    (h1 : total_distance = 105.00000000000003) 
    (foot_fraction : ℝ) (h2 : foot_fraction = 1/5)
    (bus_fraction : ℝ) (h3 : bus_fraction = 2/3) : 
    total_distance - (foot_fraction * total_distance + bus_fraction * total_distance) = 14.000000000000002 := by
  sorry

end NUMINAMATH_CALUDE_student_travel_distance_l2200_220034


namespace NUMINAMATH_CALUDE_regular_polygons_covering_plane_l2200_220087

/-- A function that returns true if a regular n-gon can completely and tightly cover a plane without gaps -/
def can_cover_plane (n : ℕ) : Prop :=
  n ≥ 3 ∧ ∃ k : ℕ, k ≥ 3 ∧ k * (1 - 2 / n) = 2

/-- The theorem stating which regular polygons can completely and tightly cover a plane without gaps -/
theorem regular_polygons_covering_plane :
  ∀ n : ℕ, can_cover_plane n ↔ n = 3 ∨ n = 4 ∨ n = 6 :=
sorry

end NUMINAMATH_CALUDE_regular_polygons_covering_plane_l2200_220087


namespace NUMINAMATH_CALUDE_house_cleaning_time_l2200_220046

/-- Proves that given John cleans the entire house in 6 hours and Nick takes 3 times as long as John to clean half the house, the time it takes for them to clean the entire house together is 3.6 hours. -/
theorem house_cleaning_time (john_time nick_time combined_time : ℝ) : 
  john_time = 6 → 
  nick_time = 3 * (john_time / 2) → 
  combined_time = 18 / 5 → 
  combined_time = 3.6 :=
by sorry

end NUMINAMATH_CALUDE_house_cleaning_time_l2200_220046
