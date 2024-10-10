import Mathlib

namespace cat_catches_rat_l1600_160029

/-- The time (in hours) it takes for the cat to catch the rat after it starts chasing -/
def catchTime : ℝ := 4

/-- The average speed of the cat in km/h -/
def catSpeed : ℝ := 90

/-- The average speed of the rat in km/h -/
def ratSpeed : ℝ := 36

/-- The time (in hours) the cat waits before chasing the rat -/
def waitTime : ℝ := 6

theorem cat_catches_rat : 
  catchTime * catSpeed = (catchTime + waitTime) * ratSpeed :=
by sorry

end cat_catches_rat_l1600_160029


namespace absolute_value_sum_zero_l1600_160076

theorem absolute_value_sum_zero (a b : ℝ) (h : |a - 3| + |b + 5| = 0) : 
  (a + b = -2) ∧ (|a| + |b| = 8) := by
  sorry

end absolute_value_sum_zero_l1600_160076


namespace intersection_of_A_and_B_l1600_160027

-- Define set A
def A : Set ℝ := {x | 2 < x ∧ x < 4}

-- Define set B
def B : Set ℝ := {x | (x - 1) * (x - 3) < 0}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x | 2 < x ∧ x < 3} := by
  sorry

end intersection_of_A_and_B_l1600_160027


namespace unique_linear_function_l1600_160056

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b

-- State the theorem
theorem unique_linear_function :
  ∀ a b : ℝ,
  (∀ x y : ℝ, x ∈ [0, 1] → y ∈ [0, 1] → |f a b x + f a b y - x * y| ≤ 1/4) →
  f a b = f (1/2) (-1/8) := by
sorry

end unique_linear_function_l1600_160056


namespace exists_identical_triangles_l1600_160058

-- Define a triangle type
structure Triangle :=
  (side1 : ℝ)
  (side2 : ℝ)
  (hypotenuse : ℝ)

-- Define a function to represent a cut operation
def cut (t : Triangle) : (Triangle × Triangle) := sorry

-- Define a function to check if two triangles are identical
def are_identical (t1 t2 : Triangle) : Prop := sorry

-- Define the initial set of triangles
def initial_triangles : Finset Triangle := sorry

-- Define the set of triangles after n cuts
def triangles_after_cuts (n : ℕ) : Finset Triangle := sorry

-- The main theorem
theorem exists_identical_triangles (n : ℕ) :
  ∃ t1 t2 : Triangle, t1 ∈ triangles_after_cuts n ∧ t2 ∈ triangles_after_cuts n ∧ t1 ≠ t2 ∧ are_identical t1 t2 :=
sorry

end exists_identical_triangles_l1600_160058


namespace compound_interest_duration_l1600_160054

theorem compound_interest_duration (P A r : ℝ) (h_P : P = 979.0209790209791) (h_A : A = 1120) (h_r : r = 0.06) :
  ∃ t : ℝ, A = P * (1 + r) ^ t := by
  sorry

end compound_interest_duration_l1600_160054


namespace root_sum_reciprocal_l1600_160038

theorem root_sum_reciprocal (p q r A B C : ℝ) : 
  p ≠ q ∧ q ≠ r ∧ p ≠ r →
  (∀ x : ℝ, x^3 - 14*x^2 + 49*x - 24 = 0 ↔ x = p ∨ x = q ∨ x = r) →
  (∀ s : ℝ, s ≠ p ∧ s ≠ q ∧ s ≠ r → 
    1 / (s^3 - 14*s^2 + 49*s - 24) = A / (s - p) + B / (s - q) + C / (s - r)) →
  1 / A + 1 / B + 1 / C = 123 := by
sorry

end root_sum_reciprocal_l1600_160038


namespace twopirsquared_is_standard_l1600_160072

/-- Represents a mathematical expression -/
inductive MathExpression
  | Constant (c : ℝ)
  | Variable (v : String)
  | Multiplication (e1 e2 : MathExpression)
  | Exponentiation (base : MathExpression) (exponent : ℕ)

/-- Checks if an expression follows standard mathematical notation -/
def isStandardNotation : MathExpression → Bool
  | MathExpression.Constant _ => true
  | MathExpression.Variable _ => true
  | MathExpression.Multiplication e1 e2 => 
      match e1, e2 with
      | MathExpression.Constant _, _ => isStandardNotation e2
      | _, _ => false
  | MathExpression.Exponentiation base _ => isStandardNotation base

/-- Represents the expression 2πr² -/
def twopirsquared : MathExpression :=
  MathExpression.Multiplication
    (MathExpression.Constant 2)
    (MathExpression.Multiplication
      (MathExpression.Variable "π")
      (MathExpression.Exponentiation (MathExpression.Variable "r") 2))

/-- Theorem stating that 2πr² follows standard mathematical notation -/
theorem twopirsquared_is_standard : isStandardNotation twopirsquared = true := by
  sorry

end twopirsquared_is_standard_l1600_160072


namespace sufficient_not_necessary_l1600_160044

theorem sufficient_not_necessary : 
  (∃ a : ℝ, a = 1 → (a - 1) * (a - 2) = 0) ∧ 
  (∃ a : ℝ, (a - 1) * (a - 2) = 0 ∧ a ≠ 1) := by
  sorry

end sufficient_not_necessary_l1600_160044


namespace product_mod_seven_l1600_160015

theorem product_mod_seven : (2023 * 2024 * 2025 * 2026) % 7 = 0 := by
  sorry

end product_mod_seven_l1600_160015


namespace equal_probability_sums_l1600_160091

/-- The number of dice rolled -/
def num_dice : ℕ := 8

/-- The minimum face value on each die -/
def min_face : ℕ := 1

/-- The maximum face value on each die -/
def max_face : ℕ := 6

/-- The sum we're comparing against -/
def sum1 : ℕ := 12

/-- The sum that should have the same probability as sum1 -/
def sum2 : ℕ := 44

/-- The probability of obtaining a specific sum when rolling num_dice dice -/
noncomputable def prob_sum (s : ℕ) : ℝ := sorry

theorem equal_probability_sums : prob_sum sum1 = prob_sum sum2 := by sorry

end equal_probability_sums_l1600_160091


namespace circles_externally_separate_l1600_160051

theorem circles_externally_separate (m n : ℝ) : 
  2 > 0 ∧ m > 0 ∧ 
  (2 : ℝ)^2 - 10*2 + n = 0 ∧ 
  m^2 - 10*m + n = 0 → 
  n > 2 + m :=
by sorry

end circles_externally_separate_l1600_160051


namespace power_of_power_three_l1600_160071

theorem power_of_power_three : (3^3)^(3^3) = 27^27 := by sorry

end power_of_power_three_l1600_160071


namespace nearest_integer_to_3_plus_sqrt2_pow6_l1600_160008

theorem nearest_integer_to_3_plus_sqrt2_pow6 :
  ∃ n : ℤ, n = 7414 ∧ ∀ m : ℤ, |((3 : ℝ) + Real.sqrt 2)^6 - (n : ℝ)| ≤ |((3 : ℝ) + Real.sqrt 2)^6 - (m : ℝ)| :=
sorry

end nearest_integer_to_3_plus_sqrt2_pow6_l1600_160008


namespace polynomial_equality_l1600_160036

theorem polynomial_equality (x : ℝ) (g : ℝ → ℝ) : 
  (4 * x^5 + 3 * x^3 - 2 * x + 5 + g x = 7 * x^3 - 4 * x^2 + x + 2) → 
  (g x = -4 * x^5 + 4 * x^3 - 4 * x^2 + 3 * x - 3) := by
  sorry

end polynomial_equality_l1600_160036


namespace choose_materials_eq_120_l1600_160049

/-- The number of ways two students can choose 2 out of 6 materials each, 
    such that they have exactly 1 material in common -/
def choose_materials : ℕ :=
  let total_materials : ℕ := 6
  let materials_per_student : ℕ := 2
  let common_materials : ℕ := 1
  Nat.choose total_materials common_materials *
  (total_materials - common_materials) * (total_materials - common_materials - 1)

theorem choose_materials_eq_120 : choose_materials = 120 := by
  sorry

end choose_materials_eq_120_l1600_160049


namespace decimal_difference_l1600_160002

def repeating_decimal : ℚ := 9 / 11
def terminating_decimal : ℚ := 81 / 100

theorem decimal_difference : repeating_decimal - terminating_decimal = 9 / 1100 := by
  sorry

end decimal_difference_l1600_160002


namespace sqrt_x_div_sqrt_y_l1600_160003

theorem sqrt_x_div_sqrt_y (x y : ℝ) (h : (1/3)^2 + (1/4)^2 = ((1/5)^2 + (1/6)^2) * (25*x)/(61*y)) :
  Real.sqrt x / Real.sqrt y = 5/2 := by
  sorry

end sqrt_x_div_sqrt_y_l1600_160003


namespace zero_in_interval_l1600_160033

def f (x : ℝ) := -x^3 - 3*x + 5

theorem zero_in_interval :
  (∀ x y, x < y → f x > f y) →  -- f is monotonically decreasing
  Continuous f →               -- f is continuous
  f 1 > 0 →                    -- f(1) > 0
  f 2 < 0 →                    -- f(2) < 0
  ∃ z, z ∈ Set.Ioo 1 2 ∧ f z = 0 := by sorry

end zero_in_interval_l1600_160033


namespace jill_jack_time_ratio_l1600_160096

/-- The ratio of Jill's time to Jack's time for a given route -/
theorem jill_jack_time_ratio (d : ℝ) (x y : ℝ) : 
  (x = d / (2 * 6) + d / (2 * 12)) →
  (y = d / (3 * 5) + 2 * d / (3 * 15)) →
  x / y = 9 / 8 := by
  sorry

end jill_jack_time_ratio_l1600_160096


namespace bus_ride_cost_l1600_160017

theorem bus_ride_cost (bus_cost train_cost : ℝ) : 
  train_cost = bus_cost + 6.85 →
  bus_cost + train_cost = 9.65 →
  bus_cost = 1.40 := by
sorry

end bus_ride_cost_l1600_160017


namespace log_inequality_l1600_160020

theorem log_inequality (x : ℝ) : 
  (Real.log (1 + 8 * x^5) / Real.log (1 + x^2) + 
   Real.log (1 + x^2) / Real.log (1 - 3 * x^2 + 16 * x^4) ≤ 
   1 + Real.log (1 + 8 * x^5) / Real.log (1 - 3 * x^2 + 16 * x^4)) ↔ 
  (x ∈ Set.Ioc (-((1/8)^(1/5))) (-1/2) ∪ 
       Set.Ioo (-Real.sqrt 3 / 4) 0 ∪ 
       Set.Ioo 0 (Real.sqrt 3 / 4) ∪ 
       {1/2}) := by sorry

end log_inequality_l1600_160020


namespace matrix_determinant_l1600_160035

def matrix : Matrix (Fin 3) (Fin 3) ℤ := !![3, 1, 0; 8, 5, -2; 3, -1, 6]

theorem matrix_determinant :
  Matrix.det matrix = 138 := by sorry

end matrix_determinant_l1600_160035


namespace geometric_sequence_product_l1600_160030

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_product (a : ℕ → ℝ) :
  GeometricSequence a →
  (a 1 * a 19 = 16) →
  (a 1 + a 19 = 10) →
  a 8 * a 10 * a 12 = 64 := by
  sorry

end geometric_sequence_product_l1600_160030


namespace smallest_norm_v_l1600_160023

theorem smallest_norm_v (v : ℝ × ℝ) (h : ‖v + (4, 2)‖ = 10) :
  ∃ (w : ℝ × ℝ), ‖w‖ = 10 - 2 * Real.sqrt 5 ∧ ∀ (u : ℝ × ℝ), ‖u + (4, 2)‖ = 10 → ‖w‖ ≤ ‖u‖ := by
  sorry

end smallest_norm_v_l1600_160023


namespace min_value_sqrt_sum_l1600_160057

theorem min_value_sqrt_sum (a b c : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0)
  (h4 : a * b + b * c + c * a = a + b + c) (h5 : a + b + c > 0) :
  Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a) ≥ 2 := by
  sorry

#check min_value_sqrt_sum

end min_value_sqrt_sum_l1600_160057


namespace unique_quadratic_function_l1600_160079

/-- A quadratic function satisfying specific conditions -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  (∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c) ∧
  f 0 = 1 ∧
  ∀ x, f (x + 1) - f x = 2 * x

/-- The unique quadratic function satisfying the given conditions -/
theorem unique_quadratic_function (f : ℝ → ℝ) (hf : QuadraticFunction f) :
  ∀ x, f x = x^2 - x + 1 := by
  sorry

end unique_quadratic_function_l1600_160079


namespace tangent_perpendicular_line_l1600_160018

-- Define the curve
def C : ℝ → ℝ := fun x ↦ x^2

-- Define the point P
def P : ℝ × ℝ := (1, 1)

-- Define the slope of the tangent line at P
def tangent_slope : ℝ := 2

-- Define the perpendicular line
def perpendicular_line (a : ℝ) : ℝ → ℝ := fun x ↦ -a * x - 1

-- State the theorem
theorem tangent_perpendicular_line :
  ∀ a : ℝ, (C P.1 = P.2) →
  (tangent_slope * (-1/a) = -1) →
  a = 1/2 := by sorry

end tangent_perpendicular_line_l1600_160018


namespace pipe_filling_time_l1600_160097

theorem pipe_filling_time (rate_A rate_B : ℝ) (time_B : ℝ) : 
  rate_A = 1 / 12 →
  rate_B = 1 / 36 →
  time_B = 12 →
  ∃ time_A : ℝ, time_A * rate_A + time_B * rate_B = 1 ∧ time_A = 8 :=
by sorry

end pipe_filling_time_l1600_160097


namespace one_third_minus_zero_point_three_three_three_l1600_160052

theorem one_third_minus_zero_point_three_three_three :
  (1 : ℚ) / 3 - (333 : ℚ) / 1000 = 1 / (3 * 1000) := by sorry

end one_third_minus_zero_point_three_three_three_l1600_160052


namespace michaels_fish_count_l1600_160031

theorem michaels_fish_count 
  (initial_fish : ℝ) 
  (fish_from_ben : ℝ) 
  (fish_from_maria : ℝ) 
  (h1 : initial_fish = 49.5)
  (h2 : fish_from_ben = 18.25)
  (h3 : fish_from_maria = 23.75) :
  initial_fish + fish_from_ben + fish_from_maria = 91.5 := by
  sorry

end michaels_fish_count_l1600_160031


namespace beth_class_size_l1600_160006

/-- Calculates the sum of an arithmetic sequence -/
def arithmeticSum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- Calculates the final number of students in Beth's class after n years -/
def finalStudents (initialStudents : ℕ) (joiningStart : ℕ) (joiningDiff : ℕ) 
                  (leavingStart : ℕ) (leavingDiff : ℕ) (years : ℕ) : ℕ :=
  initialStudents + 
  (arithmeticSum joiningStart joiningDiff years) - 
  (arithmeticSum leavingStart leavingDiff years)

theorem beth_class_size :
  finalStudents 150 30 5 15 3 4 = 222 := by
  sorry

end beth_class_size_l1600_160006


namespace shadow_length_l1600_160095

/-- Given a flagpole and a building under similar conditions, this theorem calculates
    the length of the shadow cast by the building. -/
theorem shadow_length
  (flagpole_height : ℝ)
  (flagpole_shadow : ℝ)
  (building_height : ℝ)
  (h_flagpole_height : flagpole_height = 18)
  (h_flagpole_shadow : flagpole_shadow = 45)
  (h_building_height : building_height = 24)
  : (building_height * flagpole_shadow) / flagpole_height = 60 := by
  sorry


end shadow_length_l1600_160095


namespace wills_initial_money_l1600_160093

/-- Will's initial amount of money -/
def initial_money : ℕ := 57

/-- Cost of the game Will bought -/
def game_cost : ℕ := 27

/-- Number of toys Will can buy with the remaining money -/
def num_toys : ℕ := 5

/-- Cost of each toy -/
def toy_cost : ℕ := 6

/-- Theorem stating that Will's initial money is correct given the conditions -/
theorem wills_initial_money :
  initial_money = game_cost + num_toys * toy_cost :=
by sorry

end wills_initial_money_l1600_160093


namespace count_squares_specific_grid_l1600_160053

/-- Represents a grid with a diagonal line --/
structure DiagonalGrid :=
  (width : Nat)
  (height : Nat)
  (diagonalLength : Nat)

/-- Counts the number of squares in a diagonal grid --/
def countSquares (g : DiagonalGrid) : Nat :=
  sorry

/-- The specific 6x5 grid with a diagonal in the top-left 3x3 square --/
def specificGrid : DiagonalGrid :=
  { width := 6, height := 5, diagonalLength := 3 }

/-- Theorem stating that the number of squares in the specific grid is 64 --/
theorem count_squares_specific_grid :
  countSquares specificGrid = 64 := by sorry

end count_squares_specific_grid_l1600_160053


namespace max_value_of_expression_l1600_160082

theorem max_value_of_expression (x y z : ℕ) : 
  (10 ≤ x ∧ x ≤ 99) → 
  (10 ≤ y ∧ y ≤ 99) → 
  (10 ≤ z ∧ z ≤ 99) → 
  ((x + y + z) / 3 = 60) → 
  ((x + y) / z ≤ 17) ∧ (∃ x' y' z' : ℕ, (10 ≤ x' ∧ x' ≤ 99) ∧ (10 ≤ y' ∧ y' ≤ 99) ∧ (10 ≤ z' ∧ z' ≤ 99) ∧ ((x' + y' + z') / 3 = 60) ∧ ((x' + y') / z' = 17)) :=
by sorry

end max_value_of_expression_l1600_160082


namespace product_digit_sum_l1600_160084

/-- The number of 9's in the factor that, when multiplied by 9, 
    produces a number whose digits sum to 1111 -/
def k : ℕ := 124

/-- Function to calculate the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

/-- The factor consisting of k 9's -/
def factor (k : ℕ) : ℕ :=
  10^k - 1

theorem product_digit_sum :
  sum_of_digits (9 * factor k) = 1111 := by
  sorry

end product_digit_sum_l1600_160084


namespace car_speed_conversion_l1600_160013

/-- Converts speed from m/s to km/h -/
def speed_ms_to_kmh (speed_ms : ℝ) : ℝ := speed_ms * 3.6

/-- Given a car's speed of 10 m/s, its speed in km/h is 36 km/h -/
theorem car_speed_conversion :
  let speed_ms : ℝ := 10
  speed_ms_to_kmh speed_ms = 36 := by sorry

end car_speed_conversion_l1600_160013


namespace arithmetic_sequence_property_l1600_160046

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  d : ℤ      -- Common difference
  first_term : a 1 = 3
  arithmetic : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  n * (2 * seq.a 1 + (n - 1) * seq.d) / 2

/-- Main theorem -/
theorem arithmetic_sequence_property (seq : ArithmeticSequence) 
    (h : S seq 8 = seq.a 8) : seq.a 19 = -15 := by
  sorry

end arithmetic_sequence_property_l1600_160046


namespace new_average_weight_with_D_l1600_160025

/-- Given the weights of individuals A, B, C, D, and E, prove that the average weight
    of the group when D joins is 82 kg. -/
theorem new_average_weight_with_D (w_A w_B w_C w_D : ℝ) : 
  w_A = 95 →
  (w_A + w_B + w_C) / 3 = 80 →
  ∃ w_E : ℝ, w_E = w_D + 3 ∧ (w_B + w_C + w_D + w_E) / 4 = 81 →
  (w_A + w_B + w_C + w_D) / 4 = 82 := by
  sorry


end new_average_weight_with_D_l1600_160025


namespace least_common_period_l1600_160004

-- Define the property that f satisfies the given functional equation
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 6) + f (x - 6) = f x

-- Define the property of being periodic with period p
def IsPeriodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f (x + p) = f x

-- State the theorem
theorem least_common_period :
  ∀ f : ℝ → ℝ, SatisfiesFunctionalEquation f →
    (∃ p : ℝ, p > 0 ∧ IsPeriodic f p) →
    (∀ q : ℝ, q > 0 → IsPeriodic f q → q ≥ 36) :=
  sorry

end least_common_period_l1600_160004


namespace no_five_digit_perfect_square_with_all_even_or_odd_digits_l1600_160009

theorem no_five_digit_perfect_square_with_all_even_or_odd_digits : 
  ¬ ∃ (n : ℕ), 
    (∃ (k : ℕ), n = k^2) ∧ 
    (10000 ≤ n ∧ n < 100000) ∧
    (∀ (d₁ d₂ : ℕ), d₁ < 5 → d₂ < 5 → d₁ ≠ d₂ → 
      (n / 10^d₁ % 10) ≠ (n / 10^d₂ % 10)) ∧
    ((∀ (d : ℕ), d < 5 → Even (n / 10^d % 10)) ∨ 
     (∀ (d : ℕ), d < 5 → Odd (n / 10^d % 10))) :=
by sorry

end no_five_digit_perfect_square_with_all_even_or_odd_digits_l1600_160009


namespace smallest_prime_factor_of_2939_l1600_160048

theorem smallest_prime_factor_of_2939 :
  (Nat.minFac 2939 = 13) := by sorry

end smallest_prime_factor_of_2939_l1600_160048


namespace existence_of_unfactorable_number_l1600_160069

theorem existence_of_unfactorable_number (p : ℕ) (hp : p.Prime) (hp_gt_3 : p > 3) :
  ∃ y : ℕ, y < p / 2 ∧ ¬∃ (a b : ℕ), a > y ∧ b > y ∧ p * y + 1 = a * b := by
  sorry

end existence_of_unfactorable_number_l1600_160069


namespace quadratic_real_root_condition_l1600_160010

theorem quadratic_real_root_condition (b : ℝ) :
  (∃ x : ℝ, x^2 + b*x + 25 = 0) ↔ b ≤ -10 ∨ b ≥ 10 := by
sorry

end quadratic_real_root_condition_l1600_160010


namespace point_range_theorem_l1600_160001

-- Define the line equation
def line_equation (x y a : ℝ) : Prop := 3 * x - 2 * y + a = 0

-- Define the condition for two points being on the same side of the line
def same_side (x1 y1 x2 y2 a : ℝ) : Prop :=
  (3 * x1 - 2 * y1 + a) * (3 * x2 - 2 * y2 + a) > 0

-- Theorem statement
theorem point_range_theorem (a : ℝ) :
  line_equation 3 (-1) a ∧ line_equation (-4) (-3) a ∧ same_side 3 (-1) (-4) (-3) a
  ↔ a < -11 ∨ a > 6 :=
sorry

end point_range_theorem_l1600_160001


namespace dolphins_score_l1600_160086

theorem dolphins_score (total_points sharks_points dolphins_points : ℕ) : 
  total_points = 72 →
  sharks_points - dolphins_points = 20 →
  sharks_points ≥ 2 * dolphins_points →
  sharks_points + dolphins_points = total_points →
  dolphins_points = 26 := by
sorry

end dolphins_score_l1600_160086


namespace max_gangsters_is_35_l1600_160078

/-- Represents a gang in Chicago -/
structure Gang :=
  (id : Nat)

/-- Represents a gangster in Chicago -/
structure Gangster :=
  (id : Nat)

/-- The total number of gangs in Chicago -/
def totalGangs : Nat := 36

/-- Represents the conflict relation between gangs -/
def inConflict : Gang → Gang → Prop := sorry

/-- Represents the membership of a gangster in a gang -/
def isMember : Gangster → Gang → Prop := sorry

/-- All gangsters belong to multiple gangs -/
axiom multiple_membership (g : Gangster) : ∃ (g1 g2 : Gang), g1 ≠ g2 ∧ isMember g g1 ∧ isMember g g2

/-- Any two gangsters belong to different sets of gangs -/
axiom different_memberships (g1 g2 : Gangster) : g1 ≠ g2 → ∃ (gang : Gang), (isMember g1 gang ∧ ¬isMember g2 gang) ∨ (isMember g2 gang ∧ ¬isMember g1 gang)

/-- No gangster belongs to two gangs that are in conflict -/
axiom no_conflict_membership (g : Gangster) (gang1 gang2 : Gang) : isMember g gang1 → isMember g gang2 → ¬inConflict gang1 gang2

/-- Each gang not including a gangster is in conflict with some gang including that gangster -/
axiom conflict_with_member_gang (g : Gangster) (gang1 : Gang) : ¬isMember g gang1 → ∃ (gang2 : Gang), isMember g gang2 ∧ inConflict gang1 gang2

/-- The maximum number of gangsters in Chicago -/
def maxGangsters : Nat := 35

/-- Theorem: The maximum number of gangsters in Chicago is 35 -/
theorem max_gangsters_is_35 : ∀ (gangsters : Finset Gangster), gangsters.card ≤ maxGangsters :=
  sorry

end max_gangsters_is_35_l1600_160078


namespace problem_statement_l1600_160000

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a * b + 2 * a + b = 16) :
  (∀ x y : ℝ, x > 0 → y > 0 → x * y + 2 * x + y = 16 → a * b ≥ x * y) ∧
  (∀ x y : ℝ, x > 0 → y > 0 → x * y + 2 * x + y = 16 → 2 * a + b ≤ 2 * x + y) ∧
  (∀ x y : ℝ, x > 0 → y > 0 → x * y + 2 * x + y = 16 → 1 / (a + 1) + 1 / (b + 2) ≤ 1 / (x + 1) + 1 / (y + 2)) ∧
  (a * b = 8 ∨ 2 * a + b = 8 ∨ 1 / (a + 1) + 1 / (b + 2) = Real.sqrt 2 / 3) :=
by sorry

end problem_statement_l1600_160000


namespace john_widget_production_rate_l1600_160014

/-- Represents the number of widgets John can make in an hour -/
def widgets_per_hour : ℕ := 20

/-- Represents the number of hours John works per day -/
def hours_per_day : ℕ := 8

/-- Represents the number of days John works per week -/
def days_per_week : ℕ := 5

/-- Represents the total number of widgets John makes in a week -/
def widgets_per_week : ℕ := 800

/-- Proves that the number of widgets John can make in an hour is 20 -/
theorem john_widget_production_rate : 
  widgets_per_hour * (hours_per_day * days_per_week) = widgets_per_week :=
by sorry

end john_widget_production_rate_l1600_160014


namespace system_solution_l1600_160092

theorem system_solution :
  ∃ (x y z : ℚ),
    (x + (1/3)*y + (1/3)*z = 14) ∧
    (y + (1/4)*x + (1/4)*z = 8) ∧
    (z + (1/5)*x + (1/5)*y = 8) ∧
    (x = 11) ∧ (y = 4) ∧ (z = 5) := by
  sorry

end system_solution_l1600_160092


namespace hermia_election_probability_l1600_160011

theorem hermia_election_probability (n : ℕ) (hodd : Odd n) (hpos : 0 < n) :
  let p := (2^n - 1) / (n * 2^(n-1) : ℝ)
  ∃ (probability_hermia_elected : ℝ),
    probability_hermia_elected = p ∧
    0 ≤ probability_hermia_elected ∧
    probability_hermia_elected ≤ 1 :=
by sorry

end hermia_election_probability_l1600_160011


namespace equal_sums_iff_odd_l1600_160098

def is_valid_seating (n : ℕ) : Prop :=
  n ≥ 3 ∧ ∀ (boy : ℕ) (girl1 : ℕ) (girl2 : ℕ),
    boy ≤ n ∧ n < girl1 ∧ girl1 ≤ 2*n ∧ n < girl2 ∧ girl2 ≤ 2*n →
    boy + girl1 + girl2 = 4*n + (3*n + 3)/2

theorem equal_sums_iff_odd (n : ℕ) :
  is_valid_seating n ↔ Odd n :=
sorry

end equal_sums_iff_odd_l1600_160098


namespace limit_x_cubed_minus_eight_over_x_minus_two_l1600_160050

theorem limit_x_cubed_minus_eight_over_x_minus_two : 
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 2| ∧ |x - 2| < δ → |((x^3 - 8) / (x - 2)) - 12| < ε :=
by sorry

end limit_x_cubed_minus_eight_over_x_minus_two_l1600_160050


namespace sams_remaining_money_l1600_160034

/-- Given an initial amount of money, the cost per book, and the number of books bought,
    calculate the remaining money after the purchase. -/
def remaining_money (initial_amount cost_per_book num_books : ℕ) : ℕ :=
  initial_amount - cost_per_book * num_books

/-- Theorem stating that given the specific conditions of Sam's book purchase,
    the remaining money is 16 dollars. -/
theorem sams_remaining_money :
  remaining_money 79 7 9 = 16 := by
  sorry

end sams_remaining_money_l1600_160034


namespace line_mb_value_l1600_160065

/-- A line in the 2D plane -/
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- Definition of a point on the line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  y = l.m * x + l.b

theorem line_mb_value (l : Line) :
  l.contains 0 (-1) → l.contains 1 1 → l.m * l.b = -2 := by
  sorry

end line_mb_value_l1600_160065


namespace system_solution_l1600_160087

theorem system_solution (x y z : ℝ) : 
  (x * y = 1 ∧ y * z = 2 ∧ z * x = 8) ↔ 
  ((x = 2 ∧ y = (1/2) ∧ z = 4) ∨ (x = -2 ∧ y = -(1/2) ∧ z = -4)) :=
by sorry

end system_solution_l1600_160087


namespace cone_base_circumference_l1600_160047

/-- The circumference of the base of a cone formed from a 180° sector of a circle with radius 6 inches is equal to 6π. -/
theorem cone_base_circumference (r : ℝ) (θ : ℝ) : 
  r = 6 → θ = π → 2 * π * r * (θ / (2 * π)) = 6 * π :=
by sorry

end cone_base_circumference_l1600_160047


namespace m_range_l1600_160075

def p (m : ℝ) : Prop := ∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

def q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 > 0

theorem m_range (m : ℝ) (h1 : ¬(p m)) (h2 : p m ∨ q m) : 1 < m ∧ m ≤ 2 := by
  sorry

end m_range_l1600_160075


namespace eccentricity_properties_l1600_160040

-- Define the points A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (2, 0)

-- Define the line y = x + 4
def line (x : ℝ) : ℝ := x + 4

-- Define the eccentricity function
noncomputable def eccentricity (x₀ : ℝ) : ℝ :=
  let P : ℝ × ℝ := (x₀, line x₀)
  let PA := Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2)
  let PB := Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2)
  let a := (PA + PB) / 2
  let c := 2  -- half the distance between foci
  c / a

-- Theorem statement
theorem eccentricity_properties :
  (∀ ε > 0, ∃ x₀ : ℝ, eccentricity x₀ < ε) ∧
  (∃ M : ℝ, ∀ x₀ : ℝ, eccentricity x₀ ≤ M) :=
sorry

end eccentricity_properties_l1600_160040


namespace point_on_x_axis_l1600_160089

/-- A point P with coordinates (2m-6, m-1) lies on the x-axis if and only if its coordinates are (-4, 0) -/
theorem point_on_x_axis (m : ℝ) :
  (∃ P : ℝ × ℝ, P = (2*m - 6, m - 1) ∧ P.2 = 0) ↔ (∃ P : ℝ × ℝ, P = (-4, 0)) :=
by sorry

end point_on_x_axis_l1600_160089


namespace license_plate_difference_l1600_160061

theorem license_plate_difference : 
  let georgia_plates := 26^5 * 10^2
  let texas_plates := 26^4 * 10^3
  georgia_plates - texas_plates = 731161600 := by
sorry

end license_plate_difference_l1600_160061


namespace football_club_player_selling_price_l1600_160064

/-- Calculates the selling price of each player given the financial transactions of a football club. -/
theorem football_club_player_selling_price 
  (initial_balance : ℝ) 
  (players_sold : ℕ) 
  (players_bought : ℕ) 
  (buying_price : ℝ) 
  (final_balance : ℝ) : 
  initial_balance + players_sold * ((initial_balance - final_balance + players_bought * buying_price) / players_sold) - players_bought * buying_price = final_balance → 
  (initial_balance - final_balance + players_bought * buying_price) / players_sold = 10 :=
by sorry

end football_club_player_selling_price_l1600_160064


namespace coin_coverage_probability_l1600_160070

/-- The probability of a coin covering part of the black region on a square -/
theorem coin_coverage_probability (square_side : ℝ) (triangle_leg : ℝ) (diamond_side : ℝ) (coin_diameter : ℝ) : 
  square_side = 10 →
  triangle_leg = 3 →
  diamond_side = 3 * Real.sqrt 2 →
  coin_diameter = 2 →
  (78 + 5 * Real.pi + 12 * Real.sqrt 2) / 64 = 
    (4 * (triangle_leg^2 / 2 + Real.pi + 2 * triangle_leg) + 
     2 * diamond_side^2 + Real.pi + 4 * diamond_side) / 
    ((square_side - coin_diameter)^2) := by
  sorry

end coin_coverage_probability_l1600_160070


namespace parallelogram_diagonal_intersection_l1600_160045

/-- A parallelogram with opposite vertices at (2, -3) and (10, 9) has its diagonals intersecting at (6, 3). -/
theorem parallelogram_diagonal_intersection :
  let v1 : ℝ × ℝ := (2, -3)
  let v2 : ℝ × ℝ := (10, 9)
  let midpoint : ℝ × ℝ := ((v1.1 + v2.1) / 2, (v1.2 + v2.2) / 2)
  midpoint = (6, 3) := by sorry

end parallelogram_diagonal_intersection_l1600_160045


namespace quadratic_root_transformation_l1600_160022

/-- Given a quadratic equation ax² + bx + 3 = 0 with roots -2 and 3,
    prove that the equation a(x+2)² + b(x+2) + 3 = 0 has roots -4 and 1 -/
theorem quadratic_root_transformation (a b : ℝ) :
  (∃ x, a * x^2 + b * x + 3 = 0) →
  ((-2 : ℝ) * (-2 : ℝ) * a + (-2 : ℝ) * b + 3 = 0) →
  ((3 : ℝ) * (3 : ℝ) * a + (3 : ℝ) * b + 3 = 0) →
  (a * ((-4 : ℝ) + 2)^2 + b * ((-4 : ℝ) + 2) + 3 = 0) ∧
  (a * ((1 : ℝ) + 2)^2 + b * ((1 : ℝ) + 2) + 3 = 0) :=
by sorry


end quadratic_root_transformation_l1600_160022


namespace candy_eaten_count_l1600_160077

/-- Represents the number of candy pieces collected and eaten by Travis and his brother -/
structure CandyCount where
  initial : ℕ
  remaining : ℕ
  eaten : ℕ

/-- Theorem stating that the difference between initial and remaining candy count equals the eaten count -/
theorem candy_eaten_count (c : CandyCount) (h1 : c.initial = 68) (h2 : c.remaining = 60) :
  c.eaten = 8 := by
  sorry

end candy_eaten_count_l1600_160077


namespace vector_BC_coordinates_l1600_160019

theorem vector_BC_coordinates :
  let A : ℝ × ℝ := (0, 1)
  let B : ℝ × ℝ := (3, 2)
  let AC : ℝ × ℝ := (4, 3)
  let BC : ℝ × ℝ := (B.1 - A.1 + AC.1, B.2 - A.2 + AC.2)
  BC = (1, 2) := by sorry

end vector_BC_coordinates_l1600_160019


namespace cistern_fill_time_l1600_160090

def fill_cistern (problem : ℝ → Prop) : Prop :=
  ∃ t : ℝ,
    -- Tap A fills 1/12 of the cistern per minute
    let rate_A := 1 / 12
    -- Tap B fills 1/t of the cistern per minute
    let rate_B := 1 / t
    -- Both taps run for 4 minutes
    let combined_fill := 4 * (rate_A + rate_B)
    -- Tap B runs for 8 more minutes
    let remaining_fill := 8 * rate_B
    -- The total fill is 1 (complete cistern)
    combined_fill + remaining_fill = 1 ∧
    -- The solution satisfies the original problem
    problem t

theorem cistern_fill_time :
  fill_cistern (λ t ↦ t = 18) :=
sorry

end cistern_fill_time_l1600_160090


namespace carrot_sticks_after_dinner_l1600_160067

/-- Given that James ate 22 carrot sticks before dinner and 37 carrot sticks in total,
    prove that he ate 15 carrot sticks after dinner. -/
theorem carrot_sticks_after_dinner
  (before_dinner : ℕ)
  (total : ℕ)
  (h1 : before_dinner = 22)
  (h2 : total = 37) :
  total - before_dinner = 15 := by
  sorry

end carrot_sticks_after_dinner_l1600_160067


namespace decagon_perimeter_l1600_160039

/-- The number of sides in a decagon -/
def decagon_sides : ℕ := 10

/-- The length of each side of the regular decagon -/
def side_length : ℝ := 3

/-- The perimeter of a regular polygon -/
def perimeter (n : ℕ) (s : ℝ) : ℝ := n * s

/-- Theorem: The perimeter of a regular decagon with side length 3 units is 30 units -/
theorem decagon_perimeter : 
  perimeter decagon_sides side_length = 30 := by sorry

end decagon_perimeter_l1600_160039


namespace least_period_is_36_l1600_160026

-- Define the property that f must satisfy
def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 6) + f (x - 6) = f x

-- Define what it means for a function to have a period
def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f (x + p) = f x

-- Define the least positive period
def is_least_positive_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ has_period f p ∧ ∀ q : ℝ, 0 < q ∧ q < p → ¬(has_period f q)

-- The main theorem
theorem least_period_is_36 (f : ℝ → ℝ) (h : satisfies_condition f) :
  is_least_positive_period f 36 := by
  sorry

end least_period_is_36_l1600_160026


namespace annieka_free_throws_l1600_160021

def free_throws_problem (deshawn kayla annieka : ℕ) : Prop :=
  deshawn = 12 ∧
  kayla = deshawn + (deshawn / 2) ∧
  annieka = kayla - 4

theorem annieka_free_throws :
  ∀ deshawn kayla annieka : ℕ,
    free_throws_problem deshawn kayla annieka →
    annieka = 14 :=
by
  sorry

end annieka_free_throws_l1600_160021


namespace debate_school_ratio_l1600_160028

/-- The number of students in the third school -/
def third_school : ℕ := 200

/-- The number of students in the second school -/
def second_school : ℕ := third_school + 40

/-- The total number of students who shook the mayor's hand -/
def total_students : ℕ := 920

/-- The number of students in the first school -/
def first_school : ℕ := total_students - second_school - third_school

/-- The ratio of students in the first school to students in the second school -/
def school_ratio : ℚ := first_school / second_school

theorem debate_school_ratio : school_ratio = 2 := by
  sorry

end debate_school_ratio_l1600_160028


namespace right_triangle_sin_value_l1600_160083

-- Define a right triangle ABC with angle B = 90°
def RightTriangle (A B C : Real) : Prop :=
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi ∧ B = Real.pi / 2

-- State the theorem
theorem right_triangle_sin_value
  (A B C : Real)
  (h_right_triangle : RightTriangle A B C)
  (h_sin_cos_relation : 4 * Real.sin A = 5 * Real.cos A) :
  Real.sin A = 5 * Real.sqrt 41 / 41 := by
    sorry


end right_triangle_sin_value_l1600_160083


namespace reading_ratio_l1600_160068

/-- Given the reading speeds of Carter, Oliver, and Lucy, prove that the ratio of pages
    Carter can read to pages Lucy can read in 1 hour is 1/2. -/
theorem reading_ratio (carter_pages oliver_pages lucy_extra : ℕ) 
  (h1 : carter_pages = 30)
  (h2 : oliver_pages = 40)
  (h3 : lucy_extra = 20) :
  (carter_pages : ℚ) / ((oliver_pages : ℚ) + lucy_extra) = 1 / 2 := by
  sorry

end reading_ratio_l1600_160068


namespace smallest_number_divisible_by_2022_starting_with_2023_l1600_160016

def starts_with (n m : ℕ) : Prop :=
  ∃ k : ℕ, n = m * 10^k + (n % 10^k) ∧ m * 10^k > n / 10

theorem smallest_number_divisible_by_2022_starting_with_2023 :
  ∀ n : ℕ, (n % 2022 = 0 ∧ starts_with n 2023) → n ≥ 20230110 := by
  sorry

end smallest_number_divisible_by_2022_starting_with_2023_l1600_160016


namespace function_inequality_l1600_160005

/-- Given a function f: ℝ → ℝ satisfying certain conditions, prove that f(-x₁) > f(-x₂) -/
theorem function_inequality (f : ℝ → ℝ) (x₁ x₂ : ℝ)
  (h1 : ∀ x, f (x + 1) = f (-x - 1))
  (h2 : ∀ x₁ x₂, x₁ ≥ 1 ∧ x₂ ≥ 1 ∧ x₁ < x₂ → f x₁ < f x₂)
  (h3 : x₁ < 0)
  (h4 : x₂ > 0)
  (h5 : x₁ + x₂ < -2) :
  f (-x₁) > f (-x₂) := by
  sorry

end function_inequality_l1600_160005


namespace circle_area_ratio_l1600_160032

theorem circle_area_ratio (r : ℝ) (h : r > 0) : 
  (π * r^2) / (π * (3*r)^2) = 1/9 := by sorry

end circle_area_ratio_l1600_160032


namespace f_monotonic_intervals_fixed_and_extremum_point_condition_no_two_distinct_extrema_fixed_points_l1600_160085

/-- The function f(x) = x³ + ax² + bx + 3 -/
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 3

/-- A point x₀ is a fixed point of f if f(x₀) = x₀ -/
def is_fixed_point (a b x₀ : ℝ) : Prop := f a b x₀ = x₀

/-- A point x₀ is an extremum point of f if f'(x₀) = 0 -/
def is_extremum_point (a b x₀ : ℝ) : Prop :=
  3*x₀^2 + 2*a*x₀ + b = 0

theorem f_monotonic_intervals (b : ℝ) :
  (b ≥ 0 → StrictMono (f 0 b)) ∧
  (b < 0 → StrictMonoOn (f 0 b) {x | x < -Real.sqrt (-b/3) ∨ x > Real.sqrt (-b/3)}) :=
sorry

theorem fixed_and_extremum_point_condition :
  ∃ x₀ : ℝ, is_fixed_point 0 (-3) x₀ ∧ is_extremum_point 0 (-3) x₀ :=
sorry

theorem no_two_distinct_extrema_fixed_points :
  ¬∃ a b x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    is_fixed_point a b x₁ ∧ is_extremum_point a b x₁ ∧
    is_fixed_point a b x₂ ∧ is_extremum_point a b x₂ :=
sorry

end f_monotonic_intervals_fixed_and_extremum_point_condition_no_two_distinct_extrema_fixed_points_l1600_160085


namespace magic_8_ball_probability_l1600_160042

def n : ℕ := 5
def k : ℕ := 2
def p : ℚ := 2/5

theorem magic_8_ball_probability :
  (n.choose k) * p^k * (1 - p)^(n - k) = 216/625 := by
  sorry

end magic_8_ball_probability_l1600_160042


namespace evaluate_expression_l1600_160099

theorem evaluate_expression : (18 ^ 36) / (54 ^ 18) = 6 ^ 18 := by sorry

end evaluate_expression_l1600_160099


namespace remainder_of_product_mod_17_l1600_160094

theorem remainder_of_product_mod_17 : (157^3 * 193^4) % 17 = 4 := by
  sorry

end remainder_of_product_mod_17_l1600_160094


namespace integer_root_of_polynomial_l1600_160062

-- Define the polynomial
def polynomial (d e f g x : ℚ) : ℚ := x^4 + d*x^3 + e*x^2 + f*x + g

-- State the theorem
theorem integer_root_of_polynomial (d e f g : ℚ) :
  (∃ (x : ℚ), x = 3 + Real.sqrt 5 ∧ polynomial d e f g x = 0) →
  (∃ (n : ℤ), polynomial d e f g (↑n) = 0 ∧ 
    (∀ (m : ℤ), m ≠ n → polynomial d e f g (↑m) ≠ 0)) →
  polynomial d e f g (-3) = 0 :=
sorry

end integer_root_of_polynomial_l1600_160062


namespace inequality_proof_l1600_160007

theorem inequality_proof (x y z : ℝ) 
  (h1 : y > 2*z) 
  (h2 : 2*z > 4*x) 
  (h3 : 2*(x^3 + y^3 + z^3) + 15*(x*y^2 + y*z^2 + z*x^2) > 16*(x^2*y + y^2*z + z^2*x) + 2*x*y*z) : 
  4*x + y > 4*z := by
  sorry

end inequality_proof_l1600_160007


namespace product_of_numbers_with_given_sum_and_difference_l1600_160059

theorem product_of_numbers_with_given_sum_and_difference :
  ∀ x y : ℝ, x + y = 25 ∧ x - y = 7 → x * y = 144 := by
  sorry

end product_of_numbers_with_given_sum_and_difference_l1600_160059


namespace diana_candies_l1600_160081

/-- The number of candies Diana took out of a box -/
def candies_taken (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

theorem diana_candies :
  let initial_candies : ℕ := 88
  let remaining_candies : ℕ := 82
  candies_taken initial_candies remaining_candies = 6 := by
sorry

end diana_candies_l1600_160081


namespace intersection_of_A_and_B_l1600_160037

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {1, 2, 3, 4}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by
  sorry

end intersection_of_A_and_B_l1600_160037


namespace shelby_gold_stars_l1600_160063

def gold_stars_problem (yesterday : ℕ) (total : ℕ) : Prop :=
  ∃ today : ℕ, yesterday + today = total

theorem shelby_gold_stars :
  gold_stars_problem 4 7 → ∃ today : ℕ, today = 3 :=
by
  sorry

end shelby_gold_stars_l1600_160063


namespace total_amount_is_156_l1600_160060

-- Define the ratio of shares
def x_share : ℚ := 1
def y_share : ℚ := 45 / 100
def z_share : ℚ := 50 / 100

-- Define y's actual share
def y_actual_share : ℚ := 36

-- Theorem to prove
theorem total_amount_is_156 :
  let x_actual_share := y_actual_share / y_share
  let total_amount := x_actual_share * (x_share + y_share + z_share)
  total_amount = 156 := by
sorry


end total_amount_is_156_l1600_160060


namespace multiplication_result_l1600_160043

theorem multiplication_result : 9995 * 82519 = 824777405 := by sorry

end multiplication_result_l1600_160043


namespace ellipse_triangle_area_l1600_160066

/-- The ellipse with equation x²/49 + y²/24 = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / 49) + (p.2^2 / 24) = 1}

/-- The foci of the ellipse -/
def F₁ : ℝ × ℝ := sorry
def F₂ : ℝ × ℝ := sorry

/-- A point on the ellipse -/
def P : ℝ × ℝ := sorry

/-- The distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- The area of a triangle given three points -/
def triangleArea (p q r : ℝ × ℝ) : ℝ := sorry

theorem ellipse_triangle_area :
  P ∈ Ellipse ∧ 
  (distance P F₁) / (distance P F₂) = 4 / 3 →
  triangleArea P F₁ F₂ = 24 := by
  sorry

end ellipse_triangle_area_l1600_160066


namespace addition_of_decimals_l1600_160024

theorem addition_of_decimals : (0.3 : ℝ) + 0.03 = 0.33 := by
  sorry

end addition_of_decimals_l1600_160024


namespace campground_distance_l1600_160073

/-- Calculates the total distance traveled given multiple segments of driving at different speeds. -/
def total_distance (segments : List (ℝ × ℝ)) : ℝ :=
  segments.map (fun (speed, time) => speed * time) |>.sum

/-- The driving segments for Sue's family vacation. -/
def vacation_segments : List (ℝ × ℝ) :=
  [(50, 3), (60, 2), (55, 1), (65, 2)]

/-- Theorem stating that the total distance to the campground is 455 miles. -/
theorem campground_distance :
  total_distance vacation_segments = 455 := by
  sorry

#eval total_distance vacation_segments

end campground_distance_l1600_160073


namespace cookie_radius_cookie_is_circle_l1600_160088

theorem cookie_radius (x y : ℝ) :
  (x^2 + y^2 + 17 = 6*x + 10*y) ↔ ((x - 3)^2 + (y - 5)^2 = 17) :=
by sorry

theorem cookie_is_circle (x y : ℝ) :
  (x^2 + y^2 + 17 = 6*x + 10*y) → ∃ (center_x center_y radius : ℝ),
    ((x - center_x)^2 + (y - center_y)^2 = radius^2) ∧ (radius = Real.sqrt 17) :=
by sorry

end cookie_radius_cookie_is_circle_l1600_160088


namespace trailing_zeros_500_factorial_l1600_160012

/-- Definition of factorial -/
def factorial (n : ℕ) : ℕ := (Finset.range n).prod (λ i => i + 1)

/-- Definition to count trailing zeros -/
def trailingZeros (n : ℕ) : ℕ :=
  Nat.log 10 (Nat.gcd n (10^(Nat.log 2 n + 1)))

/-- Theorem: The number of trailing zeros in 500! is 124 -/
theorem trailing_zeros_500_factorial :
  trailingZeros (factorial 500) = 124 := by sorry

end trailing_zeros_500_factorial_l1600_160012


namespace complex_number_coordinates_l1600_160041

theorem complex_number_coordinates : 
  let i : ℂ := Complex.I
  let z : ℂ := (1 + 2 * i^3) / (2 + i)
  Complex.re z = 0 ∧ Complex.im z = -1 := by sorry

end complex_number_coordinates_l1600_160041


namespace seed_germination_problem_l1600_160055

theorem seed_germination_problem (x : ℝ) : 
  x > 0 ∧ 
  0.15 * x + 0.35 * 200 = 0.23 * (x + 200) → 
  x = 300 := by
sorry

end seed_germination_problem_l1600_160055


namespace probability_theorem_l1600_160074

def harmonic_number (n : ℕ) : ℚ :=
  Finset.sum (Finset.range n) (λ i => 1 / (i + 1 : ℚ))

def probability_all_own_hats (n : ℕ) : ℚ :=
  (Finset.prod (Finset.range n) (λ i => harmonic_number (i + 1))) / (n.factorial : ℚ)

theorem probability_theorem (n : ℕ) :
  probability_all_own_hats n =
    (Finset.prod (Finset.range n) (λ i => harmonic_number (i + 1))) / (n.factorial : ℚ) :=
by sorry

#eval probability_all_own_hats 10

end probability_theorem_l1600_160074


namespace part_I_part_II_l1600_160080

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The cosine law for triangle ABC -/
def cosineLaw (t : Triangle) : Prop :=
  2 * t.b - t.c = 2 * t.a * Real.cos t.C

/-- Additional condition for part II -/
def additionalCondition (t : Triangle) : Prop :=
  4 * (t.b + t.c) = 3 * t.b * t.c

/-- Theorem for part I -/
theorem part_I (t : Triangle) (h : cosineLaw t) : t.A = 2 * Real.pi / 3 := by sorry

/-- Theorem for part II -/
theorem part_II (t : Triangle) (h1 : cosineLaw t) (h2 : additionalCondition t) (h3 : t.a = 2 * Real.sqrt 3) :
  (1/2) * t.b * t.c * Real.sin t.A = 4 * Real.sqrt 3 / 3 := by sorry

end part_I_part_II_l1600_160080
