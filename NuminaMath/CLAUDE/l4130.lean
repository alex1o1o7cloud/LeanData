import Mathlib

namespace NUMINAMATH_CALUDE_polynomial_inequality_conditions_l4130_413085

/-- A polynomial function f(x) = x^3 + ax^2 + bx + c satisfying f(x+y) ≥ f(x) + f(y) for non-negative x and y -/
def PolynomialFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ x^3 + a*x^2 + b*x + c

/-- The inequality condition for the polynomial function -/
def SatisfiesInequality (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≥ 0 → y ≥ 0 → f (x + y) ≥ f x + f y

theorem polynomial_inequality_conditions
    (a b c : ℝ)
    (h : SatisfiesInequality (PolynomialFunction a b c)) :
    a ≥ (3/2) * (9*c)^(1/3) ∧ c ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_inequality_conditions_l4130_413085


namespace NUMINAMATH_CALUDE_inequality_problem_l4130_413038

theorem inequality_problem (a b : ℝ) (h : a < b ∧ b < 0) :
  (1/a > 1/b) ∧ (abs a > -b) ∧ (Real.sqrt (-a) > Real.sqrt (-b)) ∧ ¬(1/(a-b) > 1/a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_problem_l4130_413038


namespace NUMINAMATH_CALUDE_sum_remainder_three_l4130_413080

theorem sum_remainder_three (m : ℤ) : (9 - m + (m + 4)) % 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_three_l4130_413080


namespace NUMINAMATH_CALUDE_missing_fraction_sum_l4130_413062

theorem missing_fraction_sum (x : ℚ) : 
  (1/2 : ℚ) + (-5/6 : ℚ) + (1/5 : ℚ) + (1/4 : ℚ) + (-9/20 : ℚ) + (-2/15 : ℚ) + (3/5 : ℚ) = (8/60 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_missing_fraction_sum_l4130_413062


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l4130_413097

-- Define the sets M and N
def M : Set ℝ := {x | -3 < x ∧ x < 1}
def N : Set ℝ := {x | x ≤ -3}

-- State the theorem
theorem union_of_M_and_N : M ∪ N = {x | x < 1} := by
  sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l4130_413097


namespace NUMINAMATH_CALUDE_f_is_quadratic_l4130_413056

/-- Definition of a quadratic equation in standard form -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing x^2 - 1 = 0 -/
def f (x : ℝ) : ℝ := x^2 - 1

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry


end NUMINAMATH_CALUDE_f_is_quadratic_l4130_413056


namespace NUMINAMATH_CALUDE_shop_markup_problem_l4130_413098

/-- A shop owner purchases goods at a discount and wants to mark them up for profit. -/
theorem shop_markup_problem (L : ℝ) (C : ℝ) (M : ℝ) (S : ℝ) :
  C = 0.75 * L →           -- Cost price is 75% of list price
  S = 1.3 * C →            -- Selling price is 130% of cost price
  S = 0.75 * M →           -- Selling price is 75% of marked price
  M = 1.3 * L              -- Marked price is 130% of list price
:= by sorry

end NUMINAMATH_CALUDE_shop_markup_problem_l4130_413098


namespace NUMINAMATH_CALUDE_ratio_comparison_l4130_413034

theorem ratio_comparison (y : ℕ) (h : y > 4) : (3 : ℚ) / 4 < 3 / y :=
sorry

end NUMINAMATH_CALUDE_ratio_comparison_l4130_413034


namespace NUMINAMATH_CALUDE_base8_246_equals_base10_166_l4130_413022

/-- Converts a base 8 number to base 10 --/
def base8_to_base10 (d2 d1 d0 : ℕ) : ℕ :=
  d2 * 8^2 + d1 * 8^1 + d0 * 8^0

/-- The base 8 number 246₈ is equal to 166 in base 10 --/
theorem base8_246_equals_base10_166 : base8_to_base10 2 4 6 = 166 := by
  sorry

end NUMINAMATH_CALUDE_base8_246_equals_base10_166_l4130_413022


namespace NUMINAMATH_CALUDE_annual_lesson_cost_difference_l4130_413011

/-- The number of weeks in a year -/
def weeks_per_year : ℕ := 52

/-- The hourly rate for clarinet lessons in dollars -/
def clarinet_rate : ℕ := 40

/-- The number of hours per week of clarinet lessons -/
def clarinet_hours : ℕ := 3

/-- The hourly rate for piano lessons in dollars -/
def piano_rate : ℕ := 28

/-- The number of hours per week of piano lessons -/
def piano_hours : ℕ := 5

/-- The difference in annual spending between piano and clarinet lessons -/
theorem annual_lesson_cost_difference :
  (piano_rate * piano_hours - clarinet_rate * clarinet_hours) * weeks_per_year = 1040 := by
  sorry

end NUMINAMATH_CALUDE_annual_lesson_cost_difference_l4130_413011


namespace NUMINAMATH_CALUDE_max_cubes_in_box_l4130_413020

/-- The volume of a rectangular box -/
def box_volume (length width height : ℕ) : ℕ :=
  length * width * height

/-- The volume of a cube -/
def cube_volume (side : ℕ) : ℕ :=
  side ^ 3

/-- The maximum number of cubes that can fit in a box -/
def max_cubes (box_length box_width box_height cube_side : ℕ) : ℕ :=
  (box_volume box_length box_width box_height) / (cube_volume cube_side)

theorem max_cubes_in_box :
  max_cubes 8 9 12 3 = 32 :=
by sorry

end NUMINAMATH_CALUDE_max_cubes_in_box_l4130_413020


namespace NUMINAMATH_CALUDE_f_symmetry_l4130_413029

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

-- State the theorem
theorem f_symmetry (a b : ℝ) : f a b (-2) = 10 → f a b 2 = -26 := by
  sorry

end NUMINAMATH_CALUDE_f_symmetry_l4130_413029


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l4130_413026

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_a2 : a 2 = 2) 
  (h_a4 : a 4 = 8) : 
  ∃ d : ℝ, (∀ n : ℕ, a (n + 1) - a n = d) ∧ d = 3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l4130_413026


namespace NUMINAMATH_CALUDE_fuel_station_problem_l4130_413070

/-- Represents the problem of determining the number of mini-vans filled up at a fuel station. -/
theorem fuel_station_problem (service_cost truck_count mini_van_tank truck_tank_ratio total_cost fuel_cost : ℚ) :
  service_cost = 210/100 →
  fuel_cost = 70/100 →
  truck_count = 2 →
  mini_van_tank = 65 →
  truck_tank_ratio = 220/100 →
  total_cost = 3472/10 →
  ∃ (mini_van_count : ℚ),
    mini_van_count = 3 ∧
    mini_van_count * (service_cost + mini_van_tank * fuel_cost) +
    truck_count * (service_cost + (mini_van_tank * truck_tank_ratio) * fuel_cost) = total_cost :=
by sorry

end NUMINAMATH_CALUDE_fuel_station_problem_l4130_413070


namespace NUMINAMATH_CALUDE_am_gm_positive_condition_l4130_413025

theorem am_gm_positive_condition (a b : ℝ) (h : a * b ≠ 0) :
  (a > 0 ∧ b > 0) ↔ ((a + b) / 2 ≥ Real.sqrt (a * b)) :=
sorry

end NUMINAMATH_CALUDE_am_gm_positive_condition_l4130_413025


namespace NUMINAMATH_CALUDE_pokemon_cards_total_l4130_413067

def jenny_cards : ℕ := 6

def orlando_cards (jenny : ℕ) : ℕ := jenny + 2

def richard_cards (orlando : ℕ) : ℕ := orlando * 3

def total_cards (jenny orlando richard : ℕ) : ℕ := jenny + orlando + richard

theorem pokemon_cards_total :
  total_cards jenny_cards (orlando_cards jenny_cards) (richard_cards (orlando_cards jenny_cards)) = 38 := by
  sorry

end NUMINAMATH_CALUDE_pokemon_cards_total_l4130_413067


namespace NUMINAMATH_CALUDE_min_cos_plus_sin_l4130_413084

theorem min_cos_plus_sin (A : Real) :
  let f := λ A : Real => Real.cos (A / 2) + Real.sin (A / 2)
  ∃ (min_value : Real), 
    (∀ A, f A ≥ min_value) ∧ 
    (min_value = -Real.sqrt 2) ∧
    (f (π / 2) = min_value) :=
by sorry

end NUMINAMATH_CALUDE_min_cos_plus_sin_l4130_413084


namespace NUMINAMATH_CALUDE_sqrt_expression_equality_l4130_413076

theorem sqrt_expression_equality : 
  (Real.sqrt 3 + Real.sqrt 2 - 1) * (Real.sqrt 3 - Real.sqrt 2 + 1) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equality_l4130_413076


namespace NUMINAMATH_CALUDE_milk_cans_problem_l4130_413041

theorem milk_cans_problem (x y : ℕ) : 
  x = 2 * y ∧ 
  x - 30 = 3 * (y - 20) → 
  x = 60 ∧ y = 30 := by
sorry

end NUMINAMATH_CALUDE_milk_cans_problem_l4130_413041


namespace NUMINAMATH_CALUDE_sector_central_angle_l4130_413048

theorem sector_central_angle (arc_length : Real) (area : Real) :
  arc_length = 2 * Real.pi ∧ area = 5 * Real.pi →
  ∃ (central_angle : Real),
    central_angle = 72 ∧
    central_angle * Real.pi / 180 = 2 * Real.pi * Real.pi / (5 * Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_sector_central_angle_l4130_413048


namespace NUMINAMATH_CALUDE_passengers_ratio_l4130_413060

/-- Proves that the ratio of first class to second class passengers is 1:50 given the problem conditions -/
theorem passengers_ratio (fare_ratio : ℚ) (total_amount : ℕ) (second_class_amount : ℕ) :
  fare_ratio = 3 / 1 →
  total_amount = 1325 →
  second_class_amount = 1250 →
  ∃ (x y : ℕ), x ≠ 0 ∧ y ≠ 0 ∧ (x : ℚ) / y = 1 / 50 ∧
    fare_ratio * x * (second_class_amount : ℚ) / y = (total_amount - second_class_amount : ℚ) := by
  sorry

#check passengers_ratio

end NUMINAMATH_CALUDE_passengers_ratio_l4130_413060


namespace NUMINAMATH_CALUDE_spherical_to_rectangular_conversion_l4130_413053

theorem spherical_to_rectangular_conversion :
  let ρ : ℝ := 10
  let θ : ℝ := 3 * π / 4
  let φ : ℝ := π / 4
  let x : ℝ := ρ * Real.sin φ * Real.cos θ
  let y : ℝ := ρ * Real.sin φ * Real.sin θ
  let z : ℝ := ρ * Real.cos φ
  (x = -5 ∧ y = 5 ∧ z = 5 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_spherical_to_rectangular_conversion_l4130_413053


namespace NUMINAMATH_CALUDE_arithmetic_mean_inequality_and_minimum_t_l4130_413059

theorem arithmetic_mean_inequality_and_minimum_t :
  (∀ a b c : ℝ, (((a + b + c) / 3) ^ 2 ≤ (a ^ 2 + b ^ 2 + c ^ 2) / 3) ∧
    (((a + b + c) / 3) ^ 2 = (a ^ 2 + b ^ 2 + c ^ 2) / 3 ↔ a = b ∧ b = c)) ∧
  (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 →
    (∀ t : ℝ, Real.sqrt x + Real.sqrt y + Real.sqrt z ≤ t * Real.sqrt (x + y + z) →
      t ≥ Real.sqrt 3) ∧
    ∃ t : ℝ, t = Real.sqrt 3 ∧
      Real.sqrt x + Real.sqrt y + Real.sqrt z ≤ t * Real.sqrt (x + y + z)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_inequality_and_minimum_t_l4130_413059


namespace NUMINAMATH_CALUDE_used_cd_price_l4130_413005

theorem used_cd_price (n u : ℝ) 
  (eq1 : 6 * n + 2 * u = 127.92)
  (eq2 : 3 * n + 8 * u = 133.89) :
  u = 9.99 := by
sorry

end NUMINAMATH_CALUDE_used_cd_price_l4130_413005


namespace NUMINAMATH_CALUDE_algorithm_output_is_36_l4130_413090

def algorithm_result : ℕ := 
  let s := (List.range 3).foldl (fun acc i => acc + (i + 1)) 0
  let t := (List.range 3).foldl (fun acc i => acc * (i + 1)) 1
  s * t

theorem algorithm_output_is_36 : algorithm_result = 36 := by
  sorry

end NUMINAMATH_CALUDE_algorithm_output_is_36_l4130_413090


namespace NUMINAMATH_CALUDE_percentage_equality_l4130_413089

theorem percentage_equality : ∃ P : ℝ, (P / 100) * 400 = (20 / 100) * 700 ∧ P = 35 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equality_l4130_413089


namespace NUMINAMATH_CALUDE_nurse_distribution_count_l4130_413008

/-- The number of hospitals --/
def num_hospitals : ℕ := 6

/-- The number of nurses --/
def num_nurses : ℕ := 3

/-- The maximum number of nurses allowed per hospital --/
def max_nurses_per_hospital : ℕ := 2

/-- The total number of possible nurse distributions --/
def total_distributions : ℕ := num_hospitals ^ num_nurses

/-- The number of invalid distributions (all nurses in one hospital) --/
def invalid_distributions : ℕ := num_hospitals

/-- The number of valid nurse distribution plans --/
def valid_distribution_plans : ℕ := total_distributions - invalid_distributions

theorem nurse_distribution_count :
  valid_distribution_plans = 210 :=
sorry

end NUMINAMATH_CALUDE_nurse_distribution_count_l4130_413008


namespace NUMINAMATH_CALUDE_clay_cost_calculation_l4130_413028

/-- The price of clay in won per gram -/
def clay_price : ℝ := 17.25

/-- The weight of the first clay piece in grams -/
def clay_weight_1 : ℝ := 1000

/-- The weight of the second clay piece in grams -/
def clay_weight_2 : ℝ := 10

/-- The total cost of clay for Seungjun -/
def total_cost : ℝ := clay_price * (clay_weight_1 + clay_weight_2)

theorem clay_cost_calculation :
  total_cost = 17422.5 := by sorry

end NUMINAMATH_CALUDE_clay_cost_calculation_l4130_413028


namespace NUMINAMATH_CALUDE_exists_committees_with_four_common_members_l4130_413033

/-- Represents a committee -/
structure Committee where
  members : Finset Nat
  size_eq : members.card = 80

/-- Represents the parliament -/
structure Parliament where
  deputies : Finset Nat
  committees : Finset Committee
  deputy_count : deputies.card = 1600
  committee_count : committees.card = 16000
  valid_committees : ∀ c ∈ committees, c.members ⊆ deputies

theorem exists_committees_with_four_common_members (p : Parliament) :
  ∃ c1 c2 : Committee, c1 ∈ p.committees ∧ c2 ∈ p.committees ∧ c1 ≠ c2 ∧
    (c1.members ∩ c2.members).card ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_exists_committees_with_four_common_members_l4130_413033


namespace NUMINAMATH_CALUDE_equal_angles_necessary_not_sufficient_l4130_413001

-- Define a quadrilateral
structure Quadrilateral :=
  (vertices : Fin 4 → ℝ × ℝ)

-- Define a square
def is_square (q : Quadrilateral) : Prop :=
  sorry -- Definition of a square

-- Define the property of having equal interior angles
def has_equal_interior_angles (q : Quadrilateral) : Prop :=
  sorry -- Definition of equal interior angles

theorem equal_angles_necessary_not_sufficient :
  (∀ q : Quadrilateral, is_square q → has_equal_interior_angles q) ∧
  (∃ q : Quadrilateral, has_equal_interior_angles q ∧ ¬is_square q) :=
sorry

end NUMINAMATH_CALUDE_equal_angles_necessary_not_sufficient_l4130_413001


namespace NUMINAMATH_CALUDE_eleventh_term_of_arithmetic_sequence_l4130_413061

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem eleventh_term_of_arithmetic_sequence 
  (a : ℕ → ℤ) 
  (h_arith : arithmetic_sequence a) 
  (h_1 : a 1 = 100) 
  (h_10 : a 10 = 10) : 
  a 11 = 0 := by
  sorry

end NUMINAMATH_CALUDE_eleventh_term_of_arithmetic_sequence_l4130_413061


namespace NUMINAMATH_CALUDE_root_sum_square_theorem_l4130_413075

theorem root_sum_square_theorem (m n : ℝ) : 
  (m^2 + 2*m - 2025 = 0) → 
  (n^2 + 2*n - 2025 = 0) → 
  (m ≠ n) →
  (m^2 + 3*m + n = 2023) := by
sorry

end NUMINAMATH_CALUDE_root_sum_square_theorem_l4130_413075


namespace NUMINAMATH_CALUDE_martin_bell_rings_l4130_413049

theorem martin_bell_rings (s b : ℕ) : s = 4 + b / 3 → s + b = 52 → b = 36 := by sorry

end NUMINAMATH_CALUDE_martin_bell_rings_l4130_413049


namespace NUMINAMATH_CALUDE_scientific_notation_of_18860000_l4130_413015

theorem scientific_notation_of_18860000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 18860000 = a * (10 : ℝ) ^ n ∧ a = 1.886 ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_18860000_l4130_413015


namespace NUMINAMATH_CALUDE_arithmetic_sequence_tenth_term_l4130_413087

theorem arithmetic_sequence_tenth_term :
  let a : ℚ := 1/4  -- First term
  let d : ℚ := 1/2  -- Common difference
  let n : ℕ := 10   -- Term number we're looking for
  let a_n : ℚ := a + (n - 1) * d  -- Formula for nth term of arithmetic sequence
  a_n = 19/4 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_tenth_term_l4130_413087


namespace NUMINAMATH_CALUDE_determinant_of_specific_matrix_l4130_413050

theorem determinant_of_specific_matrix :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![7, -2; -3, 6]
  Matrix.det A = 36 := by
sorry

end NUMINAMATH_CALUDE_determinant_of_specific_matrix_l4130_413050


namespace NUMINAMATH_CALUDE_power_multiplication_problem_solution_l4130_413031

theorem power_multiplication (a : ℕ) (m n : ℕ) :
  a * (a ^ n) = a ^ (n + 1) :=
by sorry

theorem problem_solution : 
  2000 * (2000 ^ 2000) = 2000 ^ 2001 :=
by sorry

end NUMINAMATH_CALUDE_power_multiplication_problem_solution_l4130_413031


namespace NUMINAMATH_CALUDE_flare_problem_l4130_413099

-- Define the height function
def h (v : ℝ) (t : ℝ) : ℝ := v * t - 4.9 * t^2

-- State the theorem
theorem flare_problem (v : ℝ) :
  h v 5 = 245 →
  v = 73.5 ∧
  ∃ t1 t2 : ℝ, t1 = 5 ∧ t2 = 10 ∧ ∀ t, t1 < t ∧ t < t2 → h v t > 245 :=
by sorry

end NUMINAMATH_CALUDE_flare_problem_l4130_413099


namespace NUMINAMATH_CALUDE_cos_pi_minus_2alpha_l4130_413052

theorem cos_pi_minus_2alpha (α : ℝ) (h : Real.sin (3 * Real.pi / 2 + α) = 1 / 3) :
  Real.cos (Real.pi - 2 * α) = - 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_minus_2alpha_l4130_413052


namespace NUMINAMATH_CALUDE_twice_as_frequent_l4130_413039

/-- Represents the direction of travel -/
inductive Direction
| East
| West

/-- Represents a train schedule -/
structure TrainSchedule where
  interval : ℕ  -- Time interval between trains
  offset : ℕ    -- Time offset from zero

/-- Represents the metro system -/
structure MetroSystem where
  eastSchedule : TrainSchedule
  westSchedule : TrainSchedule

/-- Represents a passenger's travel pattern -/
structure TravelPattern where
  eastCount : ℕ
  westCount : ℕ

/-- Function to determine which train arrives first -/
def firstTrain (metro : MetroSystem) (arrivalTime : ℕ) : Direction :=
  sorry

/-- Function to simulate passenger travel over a period -/
def simulateTravel (metro : MetroSystem) (period : ℕ) : TravelPattern :=
  sorry

/-- Theorem stating that under certain conditions, travel in one direction will be twice as frequent -/
theorem twice_as_frequent (metro : MetroSystem) 
  (h1 : metro.eastSchedule.interval = metro.westSchedule.interval)
  (h2 : metro.eastSchedule.offset = 1)
  (h3 : metro.westSchedule.offset = 9)
  (h4 : metro.eastSchedule.interval = 10) :
  ∃ (period : ℕ), 
    let pattern := simulateTravel metro period
    pattern.eastCount = 2 * pattern.westCount ∨ pattern.westCount = 2 * pattern.eastCount :=
  sorry

end NUMINAMATH_CALUDE_twice_as_frequent_l4130_413039


namespace NUMINAMATH_CALUDE_coffee_blend_price_l4130_413073

/-- Proves the price of the second blend of coffee given the conditions of the problem -/
theorem coffee_blend_price
  (total_blend : ℝ)
  (target_price : ℝ)
  (first_blend_price : ℝ)
  (first_blend_amount : ℝ)
  (h1 : total_blend = 20)
  (h2 : target_price = 8.4)
  (h3 : first_blend_price = 9)
  (h4 : first_blend_amount = 8)
  : ∃ (second_blend_price : ℝ),
    second_blend_price = 8 ∧
    total_blend * target_price =
      first_blend_amount * first_blend_price +
      (total_blend - first_blend_amount) * second_blend_price :=
by sorry

end NUMINAMATH_CALUDE_coffee_blend_price_l4130_413073


namespace NUMINAMATH_CALUDE_ellipse_condition_l4130_413064

/-- The equation of the curve -/
def curve_equation (x y k : ℝ) : Prop :=
  x^2 + 2*y^2 - 6*x + 24*y = k

/-- The condition for a non-degenerate ellipse -/
def is_non_degenerate_ellipse (k : ℝ) : Prop :=
  ∃ (a b c d e : ℝ), a > 0 ∧ b > 0 ∧
    ∀ (x y : ℝ), curve_equation x y k ↔ ((x - c)^2 / a + (y - d)^2 / b = e)

/-- The theorem stating the condition for the curve to be a non-degenerate ellipse -/
theorem ellipse_condition :
  ∀ k : ℝ, is_non_degenerate_ellipse k ↔ k > -81 :=
sorry

end NUMINAMATH_CALUDE_ellipse_condition_l4130_413064


namespace NUMINAMATH_CALUDE_quilt_shaded_fraction_l4130_413095

/-- Represents a square quilt block -/
structure QuiltBlock where
  size : Nat
  total_squares : Nat
  fully_shaded : Nat
  half_shaded : Nat

/-- Calculates the fraction of shaded area in a quilt block -/
def shaded_fraction (q : QuiltBlock) : Rat :=
  let total_area : Rat := q.total_squares
  let shaded_area : Rat := q.fully_shaded + q.half_shaded / 2
  shaded_area / total_area

/-- Theorem stating the shaded fraction of the specific quilt block -/
theorem quilt_shaded_fraction :
  let q : QuiltBlock := ⟨4, 16, 2, 4⟩
  shaded_fraction q = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_quilt_shaded_fraction_l4130_413095


namespace NUMINAMATH_CALUDE_color_selection_ways_l4130_413063

def total_colors : ℕ := 10
def colors_to_choose : ℕ := 3
def remaining_colors : ℕ := total_colors - 1  -- Subtracting blue

theorem color_selection_ways :
  (total_colors.choose colors_to_choose) - (remaining_colors.choose (colors_to_choose - 1)) =
  remaining_colors.choose (colors_to_choose - 1) := by
  sorry

end NUMINAMATH_CALUDE_color_selection_ways_l4130_413063


namespace NUMINAMATH_CALUDE_tree_height_difference_l4130_413037

/-- The height difference between two trees -/
theorem tree_height_difference (maple_height spruce_height : ℚ) 
  (h_maple : maple_height = 10 + 1/4)
  (h_spruce : spruce_height = 14 + 1/2) :
  spruce_height - maple_height = 19 + 3/4 := by
  sorry

end NUMINAMATH_CALUDE_tree_height_difference_l4130_413037


namespace NUMINAMATH_CALUDE_three_Y_five_l4130_413072

def Y (a b : ℤ) : ℤ := b + 10*a - a^2 - b^2

theorem three_Y_five : Y 3 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_three_Y_five_l4130_413072


namespace NUMINAMATH_CALUDE_friends_recycled_pounds_l4130_413040

/-- The number of pounds of paper required to earn one point -/
def pounds_per_point : ℕ := 8

/-- The number of pounds Zoe recycled -/
def zoe_pounds : ℕ := 25

/-- The total number of points earned by Zoe and her friends -/
def total_points : ℕ := 6

/-- The number of pounds Zoe's friends recycled -/
def friends_pounds : ℕ := total_points * pounds_per_point - zoe_pounds

theorem friends_recycled_pounds : friends_pounds = 23 := by sorry

end NUMINAMATH_CALUDE_friends_recycled_pounds_l4130_413040


namespace NUMINAMATH_CALUDE_fraction_addition_l4130_413091

theorem fraction_addition : (3 / 4) / (5 / 8) + 1 / 8 = 53 / 40 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l4130_413091


namespace NUMINAMATH_CALUDE_plains_total_area_l4130_413003

/-- The total area of two plains given their individual areas -/
def total_area (area_A area_B : ℝ) : ℝ := area_A + area_B

/-- Theorem: Given the conditions, the total area of both plains is 350 square miles -/
theorem plains_total_area :
  ∀ (area_A area_B : ℝ),
  area_B = 200 →
  area_A = area_B - 50 →
  total_area area_A area_B = 350 := by
sorry

end NUMINAMATH_CALUDE_plains_total_area_l4130_413003


namespace NUMINAMATH_CALUDE_simplify_expression_l4130_413086

theorem simplify_expression (x y : ℝ) :
  (15 * x + 45 * y) + (7 * x + 18 * y) - (6 * x + 35 * y) = 16 * x + 28 * y :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l4130_413086


namespace NUMINAMATH_CALUDE_solution_is_correct_l4130_413074

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- Define the equation
def equation (y : ℝ) : Prop :=
  log 3 ((4*y + 16) / (6*y - 9)) + log 3 ((6*y - 9) / (2*y - 5)) = 3

-- Theorem statement
theorem solution_is_correct :
  equation (151/50) := by sorry

end NUMINAMATH_CALUDE_solution_is_correct_l4130_413074


namespace NUMINAMATH_CALUDE_max_sequence_length_l4130_413004

theorem max_sequence_length (a : ℕ → ℤ) (n : ℕ) : 
  (∀ i : ℕ, i + 6 < n → (a i + a (i+1) + a (i+2) + a (i+3) + a (i+4) + a (i+5) + a (i+6) > 0)) →
  (∀ i : ℕ, i + 10 < n → (a i + a (i+1) + a (i+2) + a (i+3) + a (i+4) + a (i+5) + a (i+6) + a (i+7) + a (i+8) + a (i+9) + a (i+10) < 0)) →
  n ≤ 18 :=
by sorry

end NUMINAMATH_CALUDE_max_sequence_length_l4130_413004


namespace NUMINAMATH_CALUDE_unique_perfect_square_P_l4130_413023

def P (x : ℤ) : ℤ := x^4 + 6*x^3 + 11*x^2 + 3*x + 31

theorem unique_perfect_square_P :
  ∃! x : ℤ, ∃ y : ℤ, P x = y^2 :=
sorry

end NUMINAMATH_CALUDE_unique_perfect_square_P_l4130_413023


namespace NUMINAMATH_CALUDE_pages_per_chapter_l4130_413017

theorem pages_per_chapter 
  (total_chapters : ℕ) 
  (total_pages : ℕ) 
  (h1 : total_chapters = 31) 
  (h2 : total_pages = 1891) :
  total_pages / total_chapters = 61 := by
sorry

end NUMINAMATH_CALUDE_pages_per_chapter_l4130_413017


namespace NUMINAMATH_CALUDE_hyperbola_condition_l4130_413016

/-- The equation represents a hyperbola -/
def is_hyperbola (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (m - 2) + y^2 / (m - 6) = 1 ∧ (m - 2) * (m - 6) < 0

/-- The theorem stating the condition for the equation to represent a hyperbola -/
theorem hyperbola_condition (m : ℝ) : is_hyperbola m ↔ 2 < m ∧ m < 6 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_condition_l4130_413016


namespace NUMINAMATH_CALUDE_jury_stabilization_jury_stabilization_30_l4130_413044

/-- Represents a jury member -/
structure JuryMember where
  id : Nat

/-- Represents the state of the jury after a voting session -/
structure JuryState where
  members : List JuryMember
  sessionCount : Nat

/-- Represents a voting process -/
def votingProcess (state : JuryState) : JuryState :=
  sorry

/-- Theorem: For a jury with 2n members (n ≥ 2), the jury stabilizes after at most n sessions -/
theorem jury_stabilization (n : Nat) (h : n ≥ 2) :
  ∀ (initialState : JuryState),
    initialState.members.length = 2 * n →
    ∃ (finalState : JuryState),
      finalState = (votingProcess^[n]) initialState ∧
      finalState.members = ((votingProcess^[n + 1]) initialState).members :=
by
  sorry

/-- Corollary: A jury with 30 members stabilizes after at most 15 sessions -/
theorem jury_stabilization_30 :
  ∀ (initialState : JuryState),
    initialState.members.length = 30 →
    ∃ (finalState : JuryState),
      finalState = (votingProcess^[15]) initialState ∧
      finalState.members = ((votingProcess^[16]) initialState).members :=
by
  sorry

end NUMINAMATH_CALUDE_jury_stabilization_jury_stabilization_30_l4130_413044


namespace NUMINAMATH_CALUDE_small_cube_volume_ratio_l4130_413012

/-- Given a larger cube composed of smaller cubes, this theorem proves
    the relationship between the volumes of the larger cube and each smaller cube. -/
theorem small_cube_volume_ratio (V_L V_S : ℝ) (h : V_L > 0) (h_cube : V_L = 125 * V_S) :
  V_S = V_L / 125 := by
  sorry

end NUMINAMATH_CALUDE_small_cube_volume_ratio_l4130_413012


namespace NUMINAMATH_CALUDE_aria_apple_weeks_l4130_413007

theorem aria_apple_weeks (total_apples : ℕ) (apples_per_day : ℕ) (days_per_week : ℕ) : 
  total_apples = 14 →
  apples_per_day = 1 →
  days_per_week = 7 →
  (total_apples / (apples_per_day * days_per_week) : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_aria_apple_weeks_l4130_413007


namespace NUMINAMATH_CALUDE_egg_count_proof_l4130_413018

def initial_eggs : ℕ := 47
def eggs_added : ℝ := 5.0
def final_eggs : ℕ := 52

theorem egg_count_proof : 
  (initial_eggs : ℝ) + eggs_added = final_eggs := by sorry

end NUMINAMATH_CALUDE_egg_count_proof_l4130_413018


namespace NUMINAMATH_CALUDE_max_b_for_inequality_solution_l4130_413093

theorem max_b_for_inequality_solution (b : ℝ) : 
  (∃ x : ℝ, b * (b ^ (1/2)) * (x^2 - 10*x + 25) + (b ^ (1/2)) / (x^2 - 10*x + 25) ≤ 
    (1/5) * (b ^ (3/4)) * |Real.sin (π * x / 10)|) 
  → b ≤ (1/10000) :=
sorry

end NUMINAMATH_CALUDE_max_b_for_inequality_solution_l4130_413093


namespace NUMINAMATH_CALUDE_max_score_is_31_l4130_413081

/-- Represents a problem-solving robot with a limited IQ balance. -/
structure Robot where
  iq : ℕ

/-- Represents a problem with a score. -/
structure Problem where
  score : ℕ

/-- Calculates the maximum achievable score for a robot solving a set of problems. -/
def maxAchievableScore (initialIQ : ℕ) (problems : List Problem) : ℕ :=
  sorry

/-- The theorem stating the maximum achievable score for the given conditions. -/
theorem max_score_is_31 :
  let initialIQ := 25
  let problems := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map Problem.mk
  maxAchievableScore initialIQ problems = 31 := by
  sorry

end NUMINAMATH_CALUDE_max_score_is_31_l4130_413081


namespace NUMINAMATH_CALUDE_parabola_directrix_l4130_413024

-- Define the parabola
def parabola (x y : ℝ) : Prop := y = 16 * x^2

-- Define the directrix
def directrix (y : ℝ) : Prop := y = -1/64

-- Theorem statement
theorem parabola_directrix :
  ∀ x y : ℝ, parabola x y → ∃ d : ℝ, directrix d ∧ d = -1/64 :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l4130_413024


namespace NUMINAMATH_CALUDE_bottom_right_figure_impossible_l4130_413051

/-- Represents a rhombus with a fixed white and gray pattern -/
structure Rhombus :=
  (pattern : ℕ → ℕ → Bool)

/-- Represents a rotation of a rhombus -/
def rotate (r : Rhombus) (angle : ℕ) : Rhombus :=
  sorry

/-- Represents a larger figure composed of rhombuses -/
structure LargeFigure :=
  (shape : List (Rhombus × ℕ × ℕ))

/-- The specific larger figure that cannot be assembled (bottom right) -/
def bottomRightFigure : LargeFigure :=
  sorry

/-- Predicate to check if a larger figure can be assembled using only rotations of the given rhombus -/
def canAssemble (r : Rhombus) (lf : LargeFigure) : Prop :=
  sorry

/-- Theorem stating that the bottom right figure cannot be assembled -/
theorem bottom_right_figure_impossible (r : Rhombus) :
  ¬ (canAssemble r bottomRightFigure) :=
sorry

end NUMINAMATH_CALUDE_bottom_right_figure_impossible_l4130_413051


namespace NUMINAMATH_CALUDE_coconut_grove_problem_l4130_413066

theorem coconut_grove_problem (x : ℝ) 
  (yield_1 : (x + 2) * 40 = (x + 2) * 40)
  (yield_2 : x * 120 = x * 120)
  (yield_3 : (x - 2) * 180 = (x - 2) * 180)
  (average_yield : ((x + 2) * 40 + x * 120 + (x - 2) * 180) / (3 * x) = 100) :
  x = 7 := by
sorry

end NUMINAMATH_CALUDE_coconut_grove_problem_l4130_413066


namespace NUMINAMATH_CALUDE_property_characterization_l4130_413006

/-- The sum of digits of a natural number -/
def sum_of_digits (m : ℕ) : ℕ := sorry

/-- Predicate for numbers with the desired property -/
def has_property (n : ℕ) : Prop :=
  n > 1 ∧ ∀ k, 0 ≤ k ∧ k < n → ∃ m : ℕ, n ∣ m ∧ sum_of_digits m % n = k

theorem property_characterization (n : ℕ) :
  has_property n ↔ (n > 1 ∧ ¬(3 ∣ n)) :=
sorry

end NUMINAMATH_CALUDE_property_characterization_l4130_413006


namespace NUMINAMATH_CALUDE_min_operations_to_256_l4130_413035

/-- Represents the allowed operations -/
inductive Operation
  | AddOne
  | MultiplyTwo

/-- Defines a sequence of operations -/
def OperationSequence := List Operation

/-- Applies a sequence of operations to a number -/
def applyOperations (start : ℕ) (ops : OperationSequence) : ℕ :=
  ops.foldl (fun n op => match op with
    | Operation.AddOne => n + 1
    | Operation.MultiplyTwo => n * 2) start

/-- Checks if a sequence of operations transforms start into target -/
def isValidSequence (start target : ℕ) (ops : OperationSequence) : Prop :=
  applyOperations start ops = target

/-- The main theorem to be proved -/
theorem min_operations_to_256 :
  ∃ (ops : OperationSequence), isValidSequence 1 256 ops ∧ 
    ops.length = 8 ∧
    (∀ (other_ops : OperationSequence), isValidSequence 1 256 other_ops → 
      ops.length ≤ other_ops.length) :=
  sorry

end NUMINAMATH_CALUDE_min_operations_to_256_l4130_413035


namespace NUMINAMATH_CALUDE_smallest_bound_inequality_l4130_413077

theorem smallest_bound_inequality (a b c : ℝ) : 
  let M : ℝ := (9 * Real.sqrt 2) / 32
  ∀ ε > 0, ∃ a b c : ℝ, 
    |a*b*(a^2 - b^2) + b*c*(b^2 - c^2) + c*a*(c^2 - a^2)| > (M - ε)*(a^2 + b^2 + c^2)^2 ∧
    |a*b*(a^2 - b^2) + b*c*(b^2 - c^2) + c*a*(c^2 - a^2)| ≤ M*(a^2 + b^2 + c^2)^2 :=
sorry

end NUMINAMATH_CALUDE_smallest_bound_inequality_l4130_413077


namespace NUMINAMATH_CALUDE_pipe_filling_time_l4130_413043

theorem pipe_filling_time (pipe_a_rate pipe_b_rate total_time : ℚ) 
  (h1 : pipe_a_rate = 1 / 12)
  (h2 : pipe_b_rate = 1 / 20)
  (h3 : total_time = 10) :
  ∃ (x : ℚ), 
    x * (pipe_a_rate + pipe_b_rate) + (total_time - x) * pipe_b_rate = 1 ∧ 
    x = 6 := by
  sorry

end NUMINAMATH_CALUDE_pipe_filling_time_l4130_413043


namespace NUMINAMATH_CALUDE_curve_family_point_condition_l4130_413021

/-- A point (x, y) lies on at least one curve of the family y = p^2 + (2p - 1)x + 2x^2 
    if and only if y ≥ x^2 - x -/
theorem curve_family_point_condition (x y : ℝ) : 
  (∃ p : ℝ, y = p^2 + (2*p - 1)*x + 2*x^2) ↔ y ≥ x^2 - x := by
sorry

end NUMINAMATH_CALUDE_curve_family_point_condition_l4130_413021


namespace NUMINAMATH_CALUDE_waiter_new_customers_l4130_413042

theorem waiter_new_customers 
  (initial_customers : ℕ) 
  (customers_left : ℕ) 
  (final_customers : ℕ) 
  (h1 : initial_customers = 33) 
  (h2 : customers_left = 31) 
  (h3 : final_customers = 28) : 
  final_customers - (initial_customers - customers_left) = 26 := by
  sorry

end NUMINAMATH_CALUDE_waiter_new_customers_l4130_413042


namespace NUMINAMATH_CALUDE_violets_family_size_l4130_413019

/-- Proves the number of children in Violet's family given ticket prices and total cost -/
theorem violets_family_size (adult_ticket : ℕ) (child_ticket : ℕ) (total_cost : ℕ) :
  adult_ticket = 35 →
  child_ticket = 20 →
  total_cost = 155 →
  ∃ (num_children : ℕ), adult_ticket + num_children * child_ticket = total_cost ∧ num_children = 6 :=
by sorry

end NUMINAMATH_CALUDE_violets_family_size_l4130_413019


namespace NUMINAMATH_CALUDE_commute_time_difference_l4130_413079

/-- Given a set of 5 numbers {x, y, 10, 11, 9} with a mean of 10 and variance of 2, prove that |x-y| = 4 -/
theorem commute_time_difference (x y : ℝ) 
  (mean_eq : (x + y + 10 + 11 + 9) / 5 = 10) 
  (variance_eq : ((x - 10)^2 + (y - 10)^2 + 0^2 + 1^2 + (-1)^2) / 5 = 2) : 
  |x - y| = 4 := by
sorry

end NUMINAMATH_CALUDE_commute_time_difference_l4130_413079


namespace NUMINAMATH_CALUDE_equation_root_sum_l4130_413058

theorem equation_root_sum (m n : ℝ) : 
  n > 0 → 
  (1 + n * Complex.I) ^ 2 + m * (1 + n * Complex.I) + 2 = 0 → 
  m + n = -1 := by sorry

end NUMINAMATH_CALUDE_equation_root_sum_l4130_413058


namespace NUMINAMATH_CALUDE_min_distance_to_line_l4130_413032

/-- The minimum value of (x-2)^2 + (y-2)^2 given that x - y - 1 = 0 -/
theorem min_distance_to_line : 
  (∃ (m : ℝ), ∀ (x y : ℝ), x - y - 1 = 0 → (x - 2)^2 + (y - 2)^2 ≥ m) ∧ 
  (∃ (x y : ℝ), x - y - 1 = 0 ∧ (x - 2)^2 + (y - 2)^2 = 1/2) :=
sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l4130_413032


namespace NUMINAMATH_CALUDE_first_orphanage_donation_l4130_413071

/-- Given a total donation and donations to two orphanages, 
    calculate the donation to the first orphanage -/
def donation_to_first_orphanage (total : ℚ) (second : ℚ) (third : ℚ) : ℚ :=
  total - (second + third)

theorem first_orphanage_donation 
  (total : ℚ) (second : ℚ) (third : ℚ)
  (h_total : total = 650)
  (h_second : second = 225)
  (h_third : third = 250) :
  donation_to_first_orphanage total second third = 175 := by
  sorry

end NUMINAMATH_CALUDE_first_orphanage_donation_l4130_413071


namespace NUMINAMATH_CALUDE_brothers_ages_product_l4130_413096

theorem brothers_ages_product (O Y : ℕ) 
  (h1 : O > Y)
  (h2 : O - Y = 12)
  (h3 : O + Y = (O - Y) + 40) : 
  O * Y = 640 := by
  sorry

end NUMINAMATH_CALUDE_brothers_ages_product_l4130_413096


namespace NUMINAMATH_CALUDE_product_of_distinct_roots_l4130_413088

theorem product_of_distinct_roots (x y : ℝ) : 
  x ≠ 0 → y ≠ 0 → x ≠ y → (x + 6 / x = y + 6 / y) → x * y = 6 := by
  sorry

end NUMINAMATH_CALUDE_product_of_distinct_roots_l4130_413088


namespace NUMINAMATH_CALUDE_fractional_equation_solution_exists_l4130_413083

theorem fractional_equation_solution_exists : ∃ m : ℝ, ∃ x : ℝ, x ≠ 1 ∧ (x + 2) / (x - 1) = m / (1 - x) := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_exists_l4130_413083


namespace NUMINAMATH_CALUDE_correct_num_schools_l4130_413014

/-- The number of schools receiving soccer ball donations -/
def num_schools : ℕ := 2

/-- The number of classes per school -/
def classes_per_school : ℕ := 9

/-- The number of soccer balls per class -/
def balls_per_class : ℕ := 5

/-- The total number of soccer balls donated -/
def total_balls : ℕ := 90

/-- Theorem stating that the number of schools is correct -/
theorem correct_num_schools : 
  num_schools * classes_per_school * balls_per_class = total_balls :=
by sorry

end NUMINAMATH_CALUDE_correct_num_schools_l4130_413014


namespace NUMINAMATH_CALUDE_smallest_triangle_perimeter_smallest_triangle_perimeter_proof_l4130_413094

/-- The smallest possible perimeter of a triangle with consecutive integer side lengths,
    where the smallest side is at least 4. -/
theorem smallest_triangle_perimeter : ℕ → Prop :=
  fun p => (∃ n : ℕ, n ≥ 4 ∧ p = n + (n + 1) + (n + 2)) ∧
           (∀ m : ℕ, m ≥ 4 → m + (m + 1) + (m + 2) ≥ p) →
           p = 15

/-- Proof of the smallest_triangle_perimeter theorem -/
theorem smallest_triangle_perimeter_proof : smallest_triangle_perimeter 15 := by
  sorry

end NUMINAMATH_CALUDE_smallest_triangle_perimeter_smallest_triangle_perimeter_proof_l4130_413094


namespace NUMINAMATH_CALUDE_cubic_properties_l4130_413036

theorem cubic_properties :
  (∀ x : ℝ, x^3 > 0 → x > 0) ∧
  (∀ x : ℝ, x < 1 → x^3 < x) :=
by sorry

end NUMINAMATH_CALUDE_cubic_properties_l4130_413036


namespace NUMINAMATH_CALUDE_power_tower_mod_2000_l4130_413009

theorem power_tower_mod_2000 : 7^(7^(7^7)) ≡ 343 [ZMOD 2000] := by sorry

end NUMINAMATH_CALUDE_power_tower_mod_2000_l4130_413009


namespace NUMINAMATH_CALUDE_inequality_transformation_l4130_413082

theorem inequality_transformation (a b : ℝ) (h : a > b) : -3 * a < -3 * b := by
  sorry

end NUMINAMATH_CALUDE_inequality_transformation_l4130_413082


namespace NUMINAMATH_CALUDE_min_value_expression_l4130_413057

theorem min_value_expression (x y : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : x + y = 1) :
  (4 / (x + 3*y)) + (1 / (x - y)) ≥ 9/2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l4130_413057


namespace NUMINAMATH_CALUDE_angle_BXY_is_30_degrees_l4130_413054

-- Define the points and angles
variable (A B C D X Y E : Point)
variable (angle_AXE angle_CYX angle_BXY : ℝ)

-- Define the parallel lines condition
variable (h1 : Parallel (Line.mk A B) (Line.mk C D))

-- Define the angle relationship
variable (h2 : angle_AXE = 4 * angle_CYX - 90)

-- Define the equality of alternate interior angles
variable (h3 : angle_AXE = angle_CYX)

-- Define the relationship between BXY and AXE due to parallel lines
variable (h4 : angle_BXY = angle_AXE)

-- State the theorem
theorem angle_BXY_is_30_degrees :
  angle_BXY = 30 := by sorry

end NUMINAMATH_CALUDE_angle_BXY_is_30_degrees_l4130_413054


namespace NUMINAMATH_CALUDE_gcd_7920_13230_l4130_413069

theorem gcd_7920_13230 : Nat.gcd 7920 13230 = 30 := by
  sorry

end NUMINAMATH_CALUDE_gcd_7920_13230_l4130_413069


namespace NUMINAMATH_CALUDE_hyperbola_equation_l4130_413030

/-- Given a hyperbola with the equation (x²/a² - y²/b² = 1) where a > 0 and b > 0,
    if one of its asymptotes is y = (√3/2)x and one of its foci is on the directrix
    of the parabola y² = 4√7x, then a² = 4 and b² = 3. -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ (x y : ℝ), y = (Real.sqrt 3 / 2) * x) →
  (∃ (x : ℝ), x = -Real.sqrt 7) →
  a^2 = 4 ∧ b^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l4130_413030


namespace NUMINAMATH_CALUDE_largest_valid_number_l4130_413092

def is_valid_number (n : ℕ) : Prop :=
  (n ≥ 100000000 ∧ n < 1000000000) ∧
  (∀ d : ℕ, d ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9] → (∃! p, 10^p * d ≤ n ∧ n < 10^(p+1) * d)) ∧
  n % 11 = 0

theorem largest_valid_number : 
  (∀ n : ℕ, is_valid_number n → n ≤ 987652413) ∧ 
  is_valid_number 987652413 := by sorry

end NUMINAMATH_CALUDE_largest_valid_number_l4130_413092


namespace NUMINAMATH_CALUDE_min_value_of_x_plus_y_l4130_413045

theorem min_value_of_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + 8*y - x*y = 0) :
  ∀ z w : ℝ, z > 0 → w > 0 → 2*z + 8*w - z*w = 0 → x + y ≤ z + w ∧ ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 2*a + 8*b - a*b = 0 ∧ a + b = 18 := by
sorry

end NUMINAMATH_CALUDE_min_value_of_x_plus_y_l4130_413045


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l4130_413010

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

def arithmetic_sequence (a b c : ℝ) : Prop :=
  b - a = c - b

theorem geometric_sequence_problem (a : ℕ → ℝ) (b : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n : ℕ, a (n + 1) = 2 * a n) →
  arithmetic_sequence (a 2) (a 3 + 1) (a 4) →
  (∀ n : ℕ, b n = a n + n) →
  a 1 = 1 ∧
  (∀ n : ℕ, a n = 2^(n - 1)) ∧
  (b 1 + b 2 + b 3 + b 4 + b 5 = 46) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l4130_413010


namespace NUMINAMATH_CALUDE_solve_for_y_l4130_413002

theorem solve_for_y (x y : ℝ) (h1 : x^2 + 3*x + 7 = y - 5) (h2 : x = -4) : y = 16 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l4130_413002


namespace NUMINAMATH_CALUDE_first_prime_in_special_product_l4130_413047

theorem first_prime_in_special_product (x y z : Nat) : 
  Nat.Prime x ∧ Nat.Prime y ∧ Nat.Prime z ∧  -- x, y, z are prime
  x ≠ y ∧ x ≠ z ∧ y ≠ z ∧  -- x, y, z are different
  (∃ (divisors : Finset Nat), divisors.card = 12 ∧ 
    ∀ d ∈ divisors, (x^2 * y * z) % d = 0) →  -- x^2 * y * z has 12 divisors
  x = 2 :=
by sorry

end NUMINAMATH_CALUDE_first_prime_in_special_product_l4130_413047


namespace NUMINAMATH_CALUDE_box_filling_proof_l4130_413065

theorem box_filling_proof (length width depth : ℕ) (num_cubes : ℕ) : 
  length = 49 → 
  width = 42 → 
  depth = 14 → 
  num_cubes = 84 → 
  ∃ (cube_side : ℕ), 
    cube_side > 0 ∧ 
    length % cube_side = 0 ∧ 
    width % cube_side = 0 ∧ 
    depth % cube_side = 0 ∧ 
    (length / cube_side) * (width / cube_side) * (depth / cube_side) = num_cubes :=
by
  sorry

#check box_filling_proof

end NUMINAMATH_CALUDE_box_filling_proof_l4130_413065


namespace NUMINAMATH_CALUDE_pipe_stack_height_l4130_413027

/-- The height of a stack of three pipes in an isosceles triangular configuration -/
theorem pipe_stack_height (d : ℝ) (h : d = 12) : 
  let r := d / 2
  let base_center_distance := 2 * r
  let triangle_height := Real.sqrt (base_center_distance ^ 2 - (base_center_distance / 2) ^ 2)
  let total_height := triangle_height + 2 * r
  total_height = 12 + 6 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_pipe_stack_height_l4130_413027


namespace NUMINAMATH_CALUDE_common_face_sum_l4130_413068

/-- Represents a cube with numbers at its vertices -/
structure NumberedCube where
  vertices : Fin 8 → Nat
  additional : Fin 8 → Nat

/-- The sum of numbers from 1 to n -/
def sum_to_n (n : Nat) : Nat := n * (n + 1) / 2

/-- The sum of numbers on a face of the cube -/
def face_sum (cube : NumberedCube) (face : Fin 6) : Nat :=
  sorry -- Definition of face sum

/-- The theorem stating the common sum on each face -/
theorem common_face_sum (cube : NumberedCube) : 
  (∀ (i j : Fin 6), face_sum cube i = face_sum cube j) → 
  (∀ (i : Fin 8), cube.vertices i ∈ Finset.range 9) →
  (∀ (i : Fin 6), face_sum cube i = 9) :=
sorry

end NUMINAMATH_CALUDE_common_face_sum_l4130_413068


namespace NUMINAMATH_CALUDE_inequality_proof_equality_condition_l4130_413046

theorem inequality_proof (a b c d : ℝ) 
  (non_neg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) 
  (sum_squares : a^2 + b^2 + c^2 + d^2 = 4) : 
  (a + b + c + d) / 2 ≥ 1 + Real.sqrt (a * b * c * d) := by
  sorry

theorem equality_condition (a b c d : ℝ) 
  (non_neg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) 
  (sum_squares : a^2 + b^2 + c^2 + d^2 = 4) :
  (a + b + c + d) / 2 = 1 + Real.sqrt (a * b * c * d) ↔ a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_equality_condition_l4130_413046


namespace NUMINAMATH_CALUDE_least_sum_of_bases_l4130_413000

theorem least_sum_of_bases (a b : ℕ+) : 
  (7 * a.val + 8 = 8 * b.val + 7) → 
  (∀ c d : ℕ+, (7 * c.val + 8 = 8 * d.val + 7) → (c.val + d.val ≥ a.val + b.val)) →
  a.val + b.val = 17 := by
sorry

end NUMINAMATH_CALUDE_least_sum_of_bases_l4130_413000


namespace NUMINAMATH_CALUDE_division_of_decimals_l4130_413055

theorem division_of_decimals : (0.05 : ℝ) / 0.01 = 5 := by
  sorry

end NUMINAMATH_CALUDE_division_of_decimals_l4130_413055


namespace NUMINAMATH_CALUDE_concave_hexagon_guard_theorem_l4130_413013

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A concave hexagon represented by its vertices -/
structure ConcaveHexagon where
  vertices : Fin 6 → Point
  is_concave : Bool

/-- Represents visibility between two points -/
def visible (p1 p2 : Point) (h : ConcaveHexagon) : Prop :=
  sorry

/-- A guard's position -/
structure Guard where
  position : Point

/-- Checks if a point is visible to at least one guard -/
def visible_to_guards (p : Point) (guards : List Guard) (h : ConcaveHexagon) : Prop :=
  ∃ g ∈ guards, visible g.position p h

theorem concave_hexagon_guard_theorem (h : ConcaveHexagon) :
  ∃ (guards : List Guard), guards.length ≤ 2 ∧
    ∀ (p : Point), (∃ i : Fin 6, p = h.vertices i) → visible_to_guards p guards h :=
  sorry

end NUMINAMATH_CALUDE_concave_hexagon_guard_theorem_l4130_413013


namespace NUMINAMATH_CALUDE_gcd_1729_1337_l4130_413078

theorem gcd_1729_1337 : Nat.gcd 1729 1337 = 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1729_1337_l4130_413078
