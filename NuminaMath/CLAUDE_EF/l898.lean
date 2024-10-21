import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoidal_field_bases_l898_89827

theorem trapezoidal_field_bases :
  ∃! n : ℕ, n > 0 ∧ 
  (∃ s : Finset (ℕ × ℕ), s.card = n ∧ 
    (∀ p ∈ s, 
      let (b₁, b₂) := p
      b₁ > 0 ∧ b₂ > 0 ∧
      b₁ % 10 = 0 ∧ b₂ % 10 = 0 ∧
      (b₁ + b₂) / 2 * 40 = 1600) ∧
    (∀ p : ℕ × ℕ, 
      let (b₁, b₂) := p
      b₁ > 0 ∧ b₂ > 0 ∧
      b₁ % 10 = 0 ∧ b₂ % 10 = 0 ∧
      (b₁ + b₂) / 2 * 40 = 1600 → p ∈ s)) ∧
  n = 4 :=
by
  sorry

#eval "Trapezoidal field bases theorem has been stated."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoidal_field_bases_l898_89827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_production_equivalence_l898_89817

noncomputable section

variable (y : ℝ)

/-- The number of days it takes for y+4 cows to produce y+7 cans of milk,
    given that y cows produce y+2 cans of milk in y+4 days -/
noncomputable def days_to_produce (y : ℝ) : ℝ :=
  y * (y + 7) / (y + 2)

/-- The daily milk production per cow -/
noncomputable def daily_production_per_cow (y : ℝ) : ℝ :=
  (y + 2) / (y * (y + 4))

theorem milk_production_equivalence (y : ℝ) (h : y > 0) :
  (y + 4) * daily_production_per_cow y * days_to_produce y = y + 7 := by
  -- Proof steps would go here
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_production_equivalence_l898_89817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_max_value_l898_89856

theorem cos_sum_max_value (a b c : ℝ) 
  (h : Real.cos (a + c + b + c) = Real.cos (a + c) + Real.cos (b + c)) :
  ∃ (x : ℝ), Real.cos (a + c) ≤ x ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_max_value_l898_89856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_permutations_count_l898_89853

open Finset

def is_valid_permutation (a : Fin 8 → ℕ) : Prop :=
  (∀ i j : Fin 8, i ≠ j → a i ≠ a j) ∧ 
  (∀ i : Fin 8, a i ∈ range 9 \ {0})

def satisfies_condition (a : Fin 8 → ℕ) : Prop :=
  a 0 - a 1 + a 2 - a 3 + a 4 - a 5 + a 6 - a 7 = 0

theorem valid_permutations_count :
  ∃ s : Finset (Fin 8 → ℕ), 
    (∀ a ∈ s, is_valid_permutation a ∧ satisfies_condition a) ∧
    s.card = 4608 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_permutations_count_l898_89853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_valid_numbers_l898_89879

-- Define the function to count trailing zeros in n!
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

-- Define the set of numbers whose factorial has exactly 57 trailing zeros
def validNumbers : Finset ℕ :=
  Finset.filter (fun n => trailingZeros n = 57) (Finset.range 240)

-- State the theorem
theorem sum_of_valid_numbers : (Finset.sum validNumbers id) = 1185 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_valid_numbers_l898_89879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_surface_area_increase_l898_89818

theorem cube_surface_area_increase (a : ℝ) (h : a > 0) : 
  (6 * (1.5 * a)^2 - 6 * a^2) / (6 * a^2) * 100 = 125 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_surface_area_increase_l898_89818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hex_B1F_eq_2847_l898_89821

/-- Conversion function from hexadecimal digit to decimal --/
def hexToDec (c : Char) : ℕ :=
  match c with
  | 'B' => 11
  | '1' => 1
  | 'F' => 15
  | _ => 0  -- Default case, should not be reached for valid input

/-- Function to calculate the decimal value of a hexadecimal number --/
def hexToDecimal (s : String) : ℕ :=
  s.toList.reverse.enum.foldl (fun acc (i, c) => acc + (hexToDec c) * (16 ^ i)) 0

/-- Theorem stating that B1F in hexadecimal is equal to 2847 in decimal --/
theorem hex_B1F_eq_2847 : hexToDecimal "B1F" = 2847 := by
  sorry

#eval hexToDecimal "B1F"  -- This should output 2847

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hex_B1F_eq_2847_l898_89821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_a4_range_l898_89805

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ+ → ℝ
  d : ℝ
  is_arithmetic : ∀ n : ℕ+, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def sum_n (seq : ArithmeticSequence) (n : ℕ+) : ℝ :=
  n * (seq.a 1 + seq.a n) / 2

/-- The theorem statement -/
theorem arithmetic_sequence_a4_range (seq : ArithmeticSequence) 
  (monotone_inc : ∀ n : ℕ+, seq.a n < seq.a (n + 1))
  (sum4_lower : sum_n seq 4 ≥ 10)
  (sum5_upper : sum_n seq 5 ≤ 15) :
  5/2 < seq.a 4 ∧ seq.a 4 ≤ 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_a4_range_l898_89805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_area_l898_89840

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * (Real.cos (ω * x / 2))^2 + Real.cos (ω * x + Real.pi / 3)

theorem circumcircle_area 
  (ω : ℝ) 
  (h_period : ∀ x, f ω (x + Real.pi / ω) = f ω x) 
  (h_smallest_period : ∀ T, (0 < T) → (T < Real.pi / ω) → ∃ x, f ω (x + T) ≠ f ω x)
  (A B C : ℝ) 
  (h_acute : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi)
  (h_fA : f ω A = -1/2)
  (c : ℝ)
  (h_c : c = 3)
  (S : ℝ)
  (h_area : S = 6 * Real.sqrt 3) :
  (Real.pi * (c / (2 * Real.sin A))^2) = 49 * Real.pi / 3 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_area_l898_89840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_l898_89878

/-- The x-coordinate of the intersection point of two curves -/
noncomputable def intersection_x : ℝ := 0

/-- The first curve: y = 9 / (x^2 + 3) -/
noncomputable def curve1 (x : ℝ) : ℝ := 9 / (x^2 + 3)

/-- The second curve: x + y = 3 -/
def curve2 (x y : ℝ) : Prop := x + y = 3

theorem intersection_point :
  curve2 intersection_x (curve1 intersection_x) ∧
  ∀ x : ℝ, x ≠ intersection_x → ¬(curve2 x (curve1 x)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_l898_89878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_product_l898_89820

theorem cube_root_of_product : (2^9 * 5^3 * 7^3 : ℝ) ^ (1/3) = 280 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_product_l898_89820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_real_sqrt_negative_square_l898_89845

def IsRealSqrt (y : ℝ) : Prop := ∃ z : ℝ, z^2 = y

theorem unique_real_sqrt_negative_square : ∃! x : ℝ, IsRealSqrt (-(x + 2)^2) := by
  -- We'll prove this later
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_real_sqrt_negative_square_l898_89845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_sum_l898_89896

theorem parabola_intersection_sum :
  ∃ (x₁ x₂ x₃ x₄ y₁ y₂ y₃ y₄ : ℝ),
  (∀ i, i ∈ ({1, 2, 3, 4} : Finset ℕ) →
    y_i = (x_i - 1)^2 ∧ x_i - 2 = (y_i + 1)^2) →
  x₁ + x₂ + x₃ + x₄ + y₁ + y₂ + y₃ + y₄ = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_sum_l898_89896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_weeknight_sleep_l898_89860

/-- Represents the number of hours Tom sleeps each weeknight -/
def weeknight_sleep : ℝ := 5

/-- Number of weeknights in a week -/
def weeknights : ℕ := 5

/-- Number of weekend nights in a week -/
def weekend_nights : ℕ := 2

/-- Hours of sleep Tom gets each weekend night -/
def weekend_sleep : ℝ := 6

/-- Ideal hours of sleep per night -/
def ideal_sleep : ℝ := 8

/-- Hours of sleep Tom is behind in a week -/
def sleep_deficit : ℝ := 19

theorem tom_weeknight_sleep :
  weeknight_sleep = 5 :=
by
  have h1 : (weeknights : ℝ) * weeknight_sleep + (weekend_nights : ℝ) * weekend_sleep + sleep_deficit = 7 * ideal_sleep := by
    -- Proof goes here
    sorry
  -- Rest of the proof
  sorry

#check tom_weeknight_sleep

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_weeknight_sleep_l898_89860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l898_89867

noncomputable def f (x a : ℝ) : ℝ := Real.sin (x + Real.pi / 6) + Real.sin (x - Real.pi / 6) + Real.cos x + a

theorem f_properties :
  ∀ a : ℝ,
  (∀ x : ℝ, f x a = f (x + 2 * Real.pi) a) ∧
  (∃ x : ℝ, f x a ≤ 1 → a = -1) ∧
  (∃ x : ℝ, x ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2) ∧ f x a ≤ 1 → a = -1) := by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l898_89867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l898_89834

theorem solve_exponential_equation :
  ∃ y : ℝ, 5 * (2 : ℝ) ^ y = 320 ∧ y = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l898_89834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pet_food_difference_sum_l898_89863

theorem pet_food_difference_sum (dog cat bird fish : ℕ) 
  (h_dog : dog = 600) 
  (h_cat : cat = 327) 
  (h_bird : bird = 415) 
  (h_fish : fish = 248) : 
  (dog - cat) + (bird - cat) + (bird - fish) = 528 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pet_food_difference_sum_l898_89863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_problem_l898_89803

/-- Given that the solution set of ax² - 3x + 2 > 0 is {x | x < 1 or x > b}, 
    prove the values of a and b, and the solution set of ax² - (2b-a)x - 2b < 0 -/
theorem quadratic_inequality_problem 
  (a b : ℝ) 
  (h : Set.Ioi b ∪ Set.Iic 1 = {x | a * x^2 - 3 * x + 2 > 0}) :
  (a = 1 ∧ b = 2) ∧ 
  {x : ℝ | a * x^2 - (2 * b - a) * x - 2 * b < 0} = Set.Ioo (-1 : ℝ) 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_problem_l898_89803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l898_89809

/-- The Focus of a parabola is a point. -/
def Focus (f : ℝ → ℝ) : ℝ × ℝ := sorry

/-- A parabola is defined by the equation y = 4x^2. Its focus has coordinates (0, 1/16). -/
theorem parabola_focus : 
  ∃ (x y : ℝ), y = 4 * x^2 ∧ Focus (λ x => 4 * x^2) = (0, 1/16) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l898_89809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_iff_l898_89804

noncomputable def f (x : ℝ) : ℝ := (2*x - 15*x^2 + 56*x^3) / (9 - x^3)

theorem f_nonnegative_iff (x : ℝ) (h : 9 - x^3 ≠ 0) :
  f x ≥ 0 ↔ 0 ≤ x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_iff_l898_89804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clubsuit_calculation_l898_89877

-- Define the binary operation as noncomputable
noncomputable def clubsuit (a b : ℝ) : ℝ := 3 - (2 * a) / b

-- State the theorem
theorem clubsuit_calculation : 
  clubsuit (clubsuit 5 (clubsuit 3 6)) 1 = 7 := by
  -- Expand the definition of clubsuit
  unfold clubsuit
  -- Simplify the expression
  simp [div_eq_mul_inv]
  -- Perform numerical calculations
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clubsuit_calculation_l898_89877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_intercept_l898_89833

def Circle (O : ℝ × ℝ → Prop) : Prop :=
  ∀ x y, O (x, y) ↔ x^2 + y^2 = 16

def Trajectory (C : ℝ × ℝ → Prop) (O : ℝ × ℝ → Prop) : Prop :=
  ∀ x y, C (x, y) ↔ 
    ∃ x_P y_P, O (x_P, y_P) ∧ x = x_P ∧ y = 3/4 * y_P

noncomputable def InterceptLength (C : ℝ × ℝ → Prop) : ℝ :=
  let x1 := (2 + Real.sqrt 28) / 2
  let x2 := (2 - Real.sqrt 28) / 2
  let y1 := 3/4 * (x1 - 2)
  let y2 := 3/4 * (x2 - 2)
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

theorem trajectory_and_intercept 
  (O : ℝ × ℝ → Prop) (C : ℝ × ℝ → Prop)
  (h_circle : Circle O) (h_traj : Trajectory C O) :
  (∀ x y, C (x, y) ↔ x^2/16 + y^2/9 = 1) ∧
  InterceptLength C = 5 * Real.sqrt 7 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_intercept_l898_89833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_closed_form_l898_89894

def sequence_a : ℕ → ℤ
  | 0 => 0  -- Add this case for n = 0
  | 1 => 0
  | n+2 => 2 * sequence_a (n+1) + (n+2)^2 - 3

theorem sequence_a_closed_form (n : ℕ) :
  sequence_a n = 4 * 2^n - n^2 - 4*n - 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_closed_form_l898_89894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_R_depends_on_a_d_m_l898_89800

/-- Sum of first k terms of an arithmetic progression with first term a and common difference d -/
noncomputable def AP_sum (a d k : ℝ) : ℝ := k / 2 * (2 * a + (k - 1) * d)

/-- Definition of R based on sums of arithmetic progression terms -/
noncomputable def R (a d m : ℝ) : ℝ :=
  AP_sum a d (4 * m) - 2 * AP_sum a d (2 * m) + AP_sum a d m

/-- Theorem stating that R depends on a, d, and m -/
theorem R_depends_on_a_d_m (a d m : ℝ) :
  ∃ f : ℝ → ℝ → ℝ → ℝ, R a d m = f a d m ∧
  (∀ a₁ d₁ m₁ a₂ d₂ m₂, (a₁ ≠ a₂ ∨ d₁ ≠ d₂ ∨ m₁ ≠ m₂) → f a₁ d₁ m₁ ≠ f a₂ d₂ m₂) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_R_depends_on_a_d_m_l898_89800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_time_no_wind_l898_89899

/-- Represents the speed of the cyclist in km/min -/
def cyclist_speed (time_with_wind time_against_wind : ℚ) : ℚ :=
  (1 / time_with_wind + 1 / time_against_wind) / 2

/-- Theorem: The cyclist's time to travel 1 km without wind -/
theorem cyclist_time_no_wind 
  (time_with_wind : ℚ) 
  (time_against_wind : ℚ) 
  (h1 : time_with_wind = 3)
  (h2 : time_against_wind = 4) : 
  1 / (cyclist_speed time_with_wind time_against_wind) = 24 / 7 := by
  -- Unfold the definition of cyclist_speed
  unfold cyclist_speed
  -- Substitute the given values
  rw [h1, h2]
  -- Simplify the rational expressions
  norm_num
  -- The proof is complete
  done

-- Example usage (this will not execute the proof, just check the type)
#check cyclist_time_no_wind 3 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_time_no_wind_l898_89899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_line_to_curve_l898_89893

/-- The minimum distance between a point on the line √3x - y + 1 = 0 and a point on the curve y = ln x -/
theorem min_distance_line_to_curve :
  let line := fun x y : ℝ => Real.sqrt 3 * x - y + 1 = 0
  let curve := fun x : ℝ => Real.log x
  let min_distance := 1 + (1/4) * Real.log 3
  ∀ (x₁ y₁ x₂ : ℝ), line x₁ y₁ → 
    Real.sqrt ((x₁ - x₂)^2 + (y₁ - curve x₂)^2) ≥ min_distance :=
by
  -- Introduce the local definitions
  intro line curve min_distance
  -- Introduce the variables and hypothesis
  intro x₁ y₁ x₂ h_line
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_line_to_curve_l898_89893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_pole_time_l898_89814

/-- The time (in seconds) for a train to cross an electric pole -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  train_length / train_speed_ms

theorem train_crossing_pole_time :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |train_crossing_time 100 162 - 2.22| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_pole_time_l898_89814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_not_satisfied_l898_89838

noncomputable def log (base x : ℝ) : ℝ := Real.log x / Real.log base

theorem condition_not_satisfied (x : ℝ) 
  (h_pos : x > 0)
  (h_cond : ∃ (a b c : Prop),
    (a = (|x - 2.5| < 1.5)) ∧
    (b = (¬ ∃ (n : ℕ), (n : ℝ)^2 = x^2 + x + 1)) ∧
    (c = (∃ (n : ℕ), x = n)) ∧
    (log x 10 > 2) ∧
    (a ∧ b ∧ ¬c) ∨ (a ∧ ¬b ∧ c) ∨ (¬a ∧ b ∧ c)) :
  ¬ (∃ (n : ℕ), x = n) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_not_satisfied_l898_89838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_one_l898_89802

-- Define the function f as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (-a * x) / Real.log 10

-- State the theorem
theorem odd_function_implies_a_equals_one (a : ℝ) :
  (∀ x, f a x = -(f a (-x))) → a = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_one_l898_89802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_3_of_0_216_approx_l898_89837

-- Define the value we want to prove
def target_value : ℝ := -1.3947

-- State the theorem
theorem log_3_of_0_216_approx (ε : ℝ) (hε : ε > 0) : 
  |Real.log 0.216 / Real.log 3 - target_value| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_3_of_0_216_approx_l898_89837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_multiple_of_2_to_7_l898_89801

theorem smallest_multiple_of_2_to_7 : ∃ n : ℕ, n > 0 ∧ 
  (∀ k : ℕ, k > 0 → k < n → ¬(∀ i ∈ ({2, 3, 4, 5, 6, 7} : Finset ℕ), k % i = 0)) ∧ 
  (∀ i ∈ ({2, 3, 4, 5, 6, 7} : Finset ℕ), n % i = 0) :=
by
  use 420
  constructor
  · exact Nat.zero_lt_succ 419
  constructor
  · sorry -- Proof for the second condition
  · sorry -- Proof for the third condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_multiple_of_2_to_7_l898_89801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_back_wheel_revs_for_specific_bicycle_l898_89854

/-- Represents a bicycle with front and back wheel radii -/
structure Bicycle where
  front_radius : ℝ
  back_radius : ℝ

/-- Calculates the number of revolutions made by the back wheel -/
noncomputable def back_wheel_revolutions (b : Bicycle) (front_revs : ℝ) : ℝ :=
  (b.front_radius * front_revs) / b.back_radius

/-- Theorem: For a bicycle with front wheel radius 4 feet and back wheel radius 0.5 feet,
    when the front wheel makes 150 revolutions, the back wheel makes 1200 revolutions -/
theorem back_wheel_revs_for_specific_bicycle :
  let b : Bicycle := { front_radius := 4, back_radius := 0.5 }
  back_wheel_revolutions b 150 = 1200 := by
  -- Unfold the definition of back_wheel_revolutions
  unfold back_wheel_revolutions
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_back_wheel_revs_for_specific_bicycle_l898_89854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_arithmetic_geometric_progression_l898_89876

/-- Checks if a list of real numbers forms an arithmetic sequence -/
def IsArithmeticSeq (l : List ℝ) : Prop :=
  l.length ≥ 3 ∧ ∀ i : Fin (l.length - 2), l[i.1 + 1] - l[i.1] = l[i.1 + 2] - l[i.1 + 1]

/-- Checks if a list of real numbers forms a geometric sequence -/
def IsGeometricSeq (l : List ℝ) : Prop :=
  l.length ≥ 3 ∧ ∀ i : Fin (l.length - 2), l[i.1] ≠ 0 ∧ l[i.1 + 1] / l[i.1] = l[i.1 + 2] / l[i.1 + 1]

theorem unique_arithmetic_geometric_progression :
  ∃! (a b : ℝ), 
    (IsArithmeticSeq [12, a, b, a*b] ∧ IsGeometricSeq [12, a, b]) ∧
    a = 12 ∧ b = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_arithmetic_geometric_progression_l898_89876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_consumption_properties_l898_89866

/-- Represents the power consumption of a new energy vehicle -/
structure PowerConsumption where
  initialPower : ℝ
  consumptionRate : ℝ

/-- Calculates the remaining power after driving for a given time -/
noncomputable def remainingPower (pc : PowerConsumption) (t : ℝ) : ℝ :=
  pc.initialPower - pc.consumptionRate * t

/-- Calculates the distance that can be traveled with given remaining power and speed -/
noncomputable def travelableDistance (remainingPower speed : ℝ) : ℝ :=
  remainingPower / 15 * speed

theorem power_consumption_properties (pc : PowerConsumption) :
  pc.initialPower = 80 ∧ pc.consumptionRate = 15 →
  (∀ t, remainingPower pc t = 80 - 15 * t) ∧
  remainingPower pc 5 = 5 ∧
  travelableDistance 40 90 = 240 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_consumption_properties_l898_89866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_inside_angle_l898_89882

/-- Represents an angle BAC in a circle. -/
structure AngleBAC where
  angle : ℝ

/-- Indicates that an angle is acute (less than 90 degrees or π/2 radians). -/
def AngleBAC.IsAcute (a : AngleBAC) : Prop :=
  0 < a.angle ∧ a.angle < Real.pi / 2

/-- The area of the part of a circle that lies inside an angle ABC, given specific chord lengths and conditions. -/
noncomputable def AreaInsideAngleABC (r AB BC : ℝ) : ℝ := 
  1 / 2 + 10 * Real.sqrt 6 / 49 + 3 * Real.pi / 4 - Real.arcsin (5 / 7)

theorem circle_area_inside_angle (r : ℝ) (AB BC : ℝ) (h_r : r = 1) (h_AB : AB = Real.sqrt 2) (h_BC : BC = 10 / 7) (angle_BAC : AngleBAC) (h_acute : AngleBAC.IsAcute angle_BAC) :
  AreaInsideAngleABC r AB BC = 1 / 2 + 10 * Real.sqrt 6 / 49 + 3 * Real.pi / 4 - Real.arcsin (5 / 7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_inside_angle_l898_89882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amusement_park_recovery_time_l898_89871

/-- Calculates the number of days required to recover the initial investment in an amusement park. -/
theorem amusement_park_recovery_time 
  (initial_cost : ℝ) 
  (daily_cost_percentage : ℝ) 
  (daily_ticket_sales : ℕ) 
  (ticket_price : ℝ) 
  (h1 : initial_cost = 100000)
  (h2 : daily_cost_percentage = 0.01)
  (h3 : daily_ticket_sales = 150)
  (h4 : ticket_price = 10) : 
  ⌈initial_cost / (daily_ticket_sales * ticket_price - initial_cost * daily_cost_percentage)⌉ = 200 := by
  sorry

#check amusement_park_recovery_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_amusement_park_recovery_time_l898_89871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_liquefied_gas_more_economical_savings_equal_retrofitting_cost_l898_89898

/-- Represents the daily cost of fuel for a taxi --/
structure DailyCost where
  gasoline : ℚ
  liquefied_gas_min : ℚ
  liquefied_gas_max : ℚ

/-- Calculates the daily cost based on given parameters --/
def calculate_daily_cost (
  gasoline_price : ℚ)
  (gasoline_efficiency : ℚ)
  (gas_price : ℚ)
  (gas_efficiency_min : ℚ)
  (gas_efficiency_max : ℚ)
  (daily_distance : ℚ) : DailyCost :=
{
  gasoline := daily_distance / gasoline_efficiency * gasoline_price,
  liquefied_gas_min := daily_distance / gas_efficiency_max * gas_price,
  liquefied_gas_max := daily_distance / gas_efficiency_min * gas_price
}

/-- Theorem: Liquefied gas is more economical than gasoline --/
theorem liquefied_gas_more_economical (cost : DailyCost) : 
  cost.liquefied_gas_max < cost.gasoline := by
  sorry

/-- Theorem: There exists a range of days where savings equal retrofitting cost --/
theorem savings_equal_retrofitting_cost 
  (daily_savings_min : ℚ)
  (daily_savings_max : ℚ)
  (retrofitting_cost : ℚ) :
  ∃ (t1 t2 : ℕ), 
    t1 ≥ 546 ∧ 
    t2 ≤ 750 ∧ 
    (t1 : ℚ) * daily_savings_min ≤ retrofitting_cost ∧ 
    (t2 : ℚ) * daily_savings_max ≥ retrofitting_cost := by
  sorry

#check liquefied_gas_more_economical
#check savings_equal_retrofitting_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_liquefied_gas_more_economical_savings_equal_retrofitting_cost_l898_89898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l898_89832

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3)

-- State the theorem
theorem smallest_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T' ≥ T) ∧
  T = Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l898_89832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_camp_kids_count_l898_89891

theorem camp_kids_count (total_kids : ℕ) : total_kids = 2000 :=
  by
  -- Half of the kids are going to soccer camp
  have half_to_soccer : total_kids / 2 = total_kids - total_kids / 2 := by sorry
  
  -- 1/4 of the kids going to soccer camp are going in the morning
  have morning_soccer : (total_kids / 2) / 4 = total_kids / 2 - 750 := by sorry
  
  -- 750 kids are going to soccer camp in the afternoon
  have afternoon_soccer : 750 = total_kids / 2 - (total_kids / 2) / 4 := by sorry
  
  -- The proof (skipped with sorry)
  sorry

#check camp_kids_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_camp_kids_count_l898_89891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_equals_one_l898_89850

theorem sin_plus_cos_equals_one (x : ℝ) : 
  0 ≤ x ∧ x < 2 * Real.pi → (Real.sin x + Real.cos x = 1 ↔ x = 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_equals_one_l898_89850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_of_A_l898_89836

-- Define the set A
noncomputable def A : Set ℝ := {x | x ≤ Real.sqrt 13}

-- Define a
noncomputable def a : ℝ := Real.sqrt 11

-- Theorem statement
theorem subset_of_A : {a} ⊆ A := by
  -- We'll use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_of_A_l898_89836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_is_26_27ths_l898_89859

-- Define the original pyramid
noncomputable def base_edge : ℝ := 40
noncomputable def altitude : ℝ := 15

-- Define the smaller pyramid
noncomputable def small_altitude : ℝ := altitude / 3

-- Define the volume of a pyramid
noncomputable def pyramid_volume (base : ℝ) (height : ℝ) : ℝ :=
  (1 / 3) * (base^2) * height

-- Define the volume ratio of the smaller pyramid to the original
noncomputable def volume_ratio : ℝ := (small_altitude / altitude)^3

-- Define the volume of the frustum as a fraction of the original pyramid
noncomputable def frustum_volume_ratio : ℝ := 1 - volume_ratio

-- Theorem statement
theorem frustum_volume_is_26_27ths :
  frustum_volume_ratio = 26 / 27 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_is_26_27ths_l898_89859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subsets_of_intersection_l898_89826

def A : Finset ℕ := {1, 2, 3}
def B : Finset ℕ := {1, 3, 4}

theorem subsets_of_intersection : Finset.card (Finset.powerset (A ∩ B)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subsets_of_intersection_l898_89826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_satisfying_number_product_l898_89889

/-- Represents a positive integer as a list of its digits in ascending order -/
def AscendingDigits : Type := List Nat

/-- Check if the digits are in strictly ascending order -/
def is_strictly_ascending (digits : AscendingDigits) : Prop :=
  List.Pairwise (·<·) digits

/-- Calculate the sum of squares of digits -/
def sum_of_squares (digits : AscendingDigits) : Nat :=
  digits.map (fun d => d * d) |> List.sum

/-- Convert a list of digits to a natural number -/
def to_nat (digits : AscendingDigits) : Nat :=
  digits.foldl (fun acc d => acc * 10 + d) 0

/-- Find the largest number satisfying the conditions -/
def largest_satisfying_number (digits : AscendingDigits) : Prop :=
  sum_of_squares digits = 65 ∧
  is_strictly_ascending digits ∧
  ∀ other : AscendingDigits, sum_of_squares other = 65 → is_strictly_ascending other →
    to_nat other ≤ to_nat digits

/-- The product of digits of the largest satisfying number -/
def product_of_digits (digits : AscendingDigits) : Nat :=
  digits.foldl (· * ·) 1

theorem largest_satisfying_number_product :
  ∃ digits : AscendingDigits, largest_satisfying_number digits ∧ product_of_digits digits = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_satisfying_number_product_l898_89889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_tangents_range_l898_89886

/-- The function f(x) = x + ln x -/
noncomputable def f (x : ℝ) : ℝ := x + Real.log x

/-- The function g(x) = ax - 2sin x -/
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * x - 2 * Real.sin x

/-- The derivative of f(x) -/
noncomputable def f_deriv (x : ℝ) : ℝ := 1 + 1 / x

/-- The derivative of g(x) -/
noncomputable def g_deriv (a : ℝ) (x : ℝ) : ℝ := a - 2 * Real.cos x

/-- The theorem stating the range of a -/
theorem perpendicular_tangents_range (a : ℝ) : 
  (∀ x₁ > 0, ∃ x₂, f_deriv x₁ * g_deriv a x₂ = -1) → 
  -2 ≤ a ∧ a ≤ 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_tangents_range_l898_89886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relay_team_arrangements_l898_89864

theorem relay_team_arrangements (n : ℕ) (fixed_positions : ℕ) (h1 : n = 5) (h2 : fixed_positions = 2) :
  Nat.factorial (n - fixed_positions) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relay_team_arrangements_l898_89864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_characterization_l898_89887

/-- The functional equation that f must satisfy for all real a, b, c -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ a b c : ℝ, (f a - f b) * (f b - f c) * (f c - f a) = f (a*b^2 + b*c^2 + c*a^2) - f (a^2*b + b^2*c + c^2*a)

/-- The set of all functions that satisfy the functional equation -/
def SolutionSet : Set (ℝ → ℝ) :=
  {f | FunctionalEquation f}

/-- The characterization of the solution set -/
theorem solution_characterization :
  ∀ f ∈ SolutionSet, ∃ c : ℝ,
    f = (λ x ↦ c) ∨
    f = (λ x ↦ x + c) ∨
    f = (λ x ↦ -x + c) ∨
    f = (λ x ↦ x^3 + c) ∨
    f = (λ x ↦ -x^3 + c) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_characterization_l898_89887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_approx_l898_89813

/-- The area of a rhombus given its diagonals and the angle between them -/
noncomputable def rhombusArea (d1 d2 angle : ℝ) : ℝ :=
  (d1 * d2 * Real.sin (angle * Real.pi / 180)) / 2

/-- Theorem: The area of a rhombus with diagonals 80m and 120m, and angle 40° between them, is approximately 3085.44 m² -/
theorem rhombus_area_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |rhombusArea 80 120 40 - 3085.44| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_approx_l898_89813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_property_l898_89862

noncomputable def f (m : ℤ) (x : ℝ) := x^(-m^2 + 2*m + 3)

theorem power_function_property (m : ℤ) :
  (∀ x y : ℝ, 0 < x ∧ x < y → f m x < f m y) ∧
  (∀ x : ℝ, x ≠ 0 → f m x = f m (-x)) →
  m = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_property_l898_89862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_root_nearest_integer_l898_89852

theorem largest_root_nearest_integer : ∃ x : ℝ,
  (∀ y : ℝ, y^4 - 2009*y + 1 = 0 → y ≤ x) ∧
  x^4 - 2009*x + 1 = 0 ∧
  ∃ n : ℤ, n = -13 ∧ ∀ m : ℤ, |1 / (x^3 - 2009) - (n : ℝ)| ≤ |1 / (x^3 - 2009) - (m : ℝ)| :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_root_nearest_integer_l898_89852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_satisfies_projections_l898_89830

noncomputable def proj (a : ℝ × ℝ) (u : ℝ × ℝ) : ℝ × ℝ :=
  let scalar := ((u.1 * a.1 + u.2 * a.2) / (a.1^2 + a.2^2))
  (scalar * a.1, scalar * a.2)

theorem vector_satisfies_projections :
  let u : ℝ × ℝ := (3, 2)
  proj (3, 2) u = (45/13, 30/13) ∧
  proj (1, 4) u = (14/17, 56/17) :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_satisfies_projections_l898_89830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_a_l898_89842

open Real

/-- The function f parameterized by a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a * log x - a * x

/-- f is increasing on (0, +∞) -/
def is_increasing (a : ℝ) : Prop :=
  ∀ x y, 0 < x → 0 < y → x < y → f a x < f a y

theorem max_value_of_a :
  ∃ (max_a : ℝ), max_a = 8 ∧
    (∀ a : ℝ, a > 0 → is_increasing a → a ≤ max_a) ∧
    (∃ a : ℝ, a > 0 ∧ is_increasing a ∧ a = max_a) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_a_l898_89842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_construction_l898_89835

/-- Two lines in a plane -/
structure Line where

/-- A point in a plane -/
structure Point where

/-- A circle in a plane -/
structure Circle where

/-- Predicate to check if a point lies on a line -/
def Point.on_line (p : Point) (l : Line) : Prop := sorry

/-- Predicate to check if two lines intersect -/
def Line.intersect (l1 l2 : Line) : Prop := sorry

/-- Predicate to check if a circle passes through a point -/
def Circle.passes_through (c : Circle) (p : Point) : Prop := sorry

/-- Predicate to check if a circle is tangent to a line -/
def Circle.tangent_to (c : Circle) (l : Line) : Prop := sorry

/-- Get the center of a circle -/
def Circle.center (c : Circle) : Point := sorry

theorem circle_construction (a b : Line) (P : Point) 
  (h1 : Line.intersect a b) 
  (h2 : Point.on_line P b) :
  ∃ O₁ O₂ : Point, 
    Point.on_line O₁ b ∧ 
    Point.on_line O₂ b ∧ 
    ∃ c₁ c₂ : Circle, 
      Circle.center c₁ = O₁ ∧ 
      Circle.center c₂ = O₂ ∧ 
      Circle.passes_through c₁ P ∧ 
      Circle.passes_through c₂ P ∧ 
      Circle.tangent_to c₁ a ∧ 
      Circle.tangent_to c₂ a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_construction_l898_89835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_points_10th_game_l898_89824

/-- Represents the total points scored in the first 6 games -/
def X : ℕ := sorry

/-- The points scored in the 7th game -/
def points_7th_game : ℕ := 24

/-- The points scored in the 8th game -/
def points_8th_game : ℕ := 17

/-- The points scored in the 9th game -/
def points_9th_game : ℕ := 25

/-- The condition that the mean after 9 games is higher than after 6 games -/
axiom mean_condition : (X : ℚ) / 6 < (X + points_7th_game + points_8th_game + points_9th_game : ℚ) / 9

/-- The theorem to prove -/
theorem smallest_points_10th_game :
  ∀ y : ℕ, 
    ((X + points_7th_game + points_8th_game + points_9th_game + y : ℚ) / 10 > 22 ↔ y ≥ 24) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_points_10th_game_l898_89824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_top_block_value_l898_89883

/-- Represents the pyramid structure --/
structure Pyramid where
  layer1 : List Nat
  layer2 : List Nat
  layer3 : List Nat
  top : Nat

/-- Checks if a list of numbers is valid for the bottom layer --/
def validBottomLayer (l : List Nat) : Prop :=
  l.length = 10 ∧ l.all (λ n => 5 ≤ n ∧ n ≤ 14) ∧ l.toFinset.card = 10

/-- Calculates the value of a block based on the three blocks beneath it --/
def blockValue (a b c : Nat) : Nat := a + b + c

/-- Checks if the pyramid structure is valid according to the rules --/
def validPyramid (p : Pyramid) : Prop :=
  validBottomLayer p.layer1 ∧
  p.layer2.length = 6 ∧
  p.layer3.length = 3 ∧
  (∀ i : Fin 6, i.val + 2 < p.layer1.length → 
    p.layer2[i]! = blockValue p.layer1[i]! p.layer1[i.val + 1]! p.layer1[i.val + 2]!) ∧
  (∀ i : Fin 3, i.val + 2 < p.layer2.length → 
    p.layer3[i]! = blockValue p.layer2[i]! p.layer2[i.val + 1]! p.layer2[i.val + 2]!) ∧
  p.top = blockValue p.layer3[0]! p.layer3[1]! p.layer3[2]!

/-- The main theorem stating the minimum possible value for the top block --/
theorem min_top_block_value :
  ∀ p : Pyramid, validPyramid p → p.top ≥ 222 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_top_block_value_l898_89883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_perimeter_is_32_l898_89897

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

/-- Calculates the perimeter of a trapezoid given its four vertices -/
noncomputable def trapezoidPerimeter (j k l m : Point) : ℝ :=
  distance j k + distance k l + distance l m + distance m j

/-- The main theorem stating that the perimeter of the given trapezoid is 32 -/
theorem trapezoid_perimeter_is_32 :
  let j : Point := ⟨-2, -3⟩
  let k : Point := ⟨-2, 1⟩
  let l : Point := ⟨6, 7⟩
  let m : Point := ⟨6, -3⟩
  trapezoidPerimeter j k l m = 32 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_perimeter_is_32_l898_89897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_chord_and_tangents_l898_89873

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y + 3 = 0

-- Define the circle C
def circle_C (x y a : ℝ) : Prop := (x - a)^2 + (y - 2)^2 = 4

-- Define the chord length condition
def chord_length_condition (a : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    line_l x₁ y₁ ∧ line_l x₂ y₂ ∧
    circle_C x₁ y₁ a ∧ circle_C x₂ y₂ a ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 8

-- Define the tangent line equations
def tangent_line_1 (x y : ℝ) : Prop := 5*x - 12*y + 45 = 0
def tangent_line_2 (x : ℝ) : Prop := x = 3

theorem circle_chord_and_tangents
  (a : ℝ)
  (h_a_pos : a > 0)
  (h_chord : chord_length_condition a) :
  a = 1 ∧
  (∀ x y, circle_C x y 1 → ((x - 3)^2 + (y - 5)^2 = 4 ↔ (tangent_line_1 x y ∨ tangent_line_2 x))) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_chord_and_tangents_l898_89873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplification_proof_l898_89815

noncomputable def original_expression : ℝ := Real.rpow (8 + 27) (1/3) * Real.rpow (8 + Real.rpow 27 (1/3)) (1/3)

noncomputable def simplified_result : ℝ := Real.rpow 385 (1/3)

theorem simplification_proof : original_expression = simplified_result := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplification_proof_l898_89815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_for_f_eq_3_l898_89869

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ -1 then x + 2
  else if x < 2 then x^2
  else 2*x

-- Theorem statement
theorem unique_solution_for_f_eq_3 :
  ∃! x, f x = 3 ∧ x = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_for_f_eq_3_l898_89869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_correct_statements_l898_89868

-- Define the decimal equality
def decimal_equality : Prop := 4.2302300 = 4.23

-- Define the ratio simplification
def ratio_simplification : Prop := (1 : ℚ) / 0.125 = 8

-- Theorem statement
theorem two_correct_statements : 
  decimal_equality ∧ ratio_simplification :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_correct_statements_l898_89868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sailors_guessing_strategy_exists_l898_89872

/-- Represents a sailor with a number on their forehead -/
structure Sailor :=
  (number : Nat)

/-- The strategy function that each sailor uses to guess their number -/
def strategy (sailors : List Sailor) (index : Nat) : Nat :=
  sorry

/-- Theorem stating that there exists a strategy guaranteeing at least one correct guess -/
theorem sailors_guessing_strategy_exists :
  ∃ (strategy : List Sailor → Nat → Nat),
    ∀ (sailors : List Sailor),
      sailors.length = 11 →
      (∀ s ∈ sailors, s.number ≥ 1 ∧ s.number ≤ 11) →
      ∃ i, i < sailors.length ∧ strategy sailors i = (sailors.get ⟨i, sorry⟩).number :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sailors_guessing_strategy_exists_l898_89872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_rate_is_14_percent_l898_89816

-- Define the tax structure
structure TaxSystem where
  lowRate : ℚ  -- Rate for first $40,000
  highRate : ℚ  -- Rate for income above $40,000
  threshold : ℚ  -- Threshold for higher rate

-- Define the income and tax paid
def income : ℚ := 51999.99
def taxPaid : ℚ := 8000

-- Define the tax calculation function
def calculateTax (sys : TaxSystem) (inc : ℚ) : ℚ :=
  if inc ≤ sys.threshold then
    inc * sys.lowRate
  else
    sys.threshold * sys.lowRate + (inc - sys.threshold) * sys.highRate

-- Theorem statement
theorem tax_rate_is_14_percent :
  ∃ (sys : TaxSystem),
    sys.threshold = 40000 ∧
    sys.highRate = 1/5 ∧
    sys.lowRate = 7/50 ∧
    calculateTax sys income = taxPaid := by
  -- Construct the TaxSystem
  let sys : TaxSystem := {
    threshold := 40000
    highRate := 1/5
    lowRate := 7/50
  }
  -- Prove that this system satisfies all conditions
  exists sys
  constructor
  · rfl
  constructor
  · rfl
  constructor
  · rfl
  -- Prove that the calculated tax matches the given tax
  · sorry  -- The actual calculation would go here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_rate_is_14_percent_l898_89816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l898_89855

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt x

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (f (a + 1) < f (10 - 2 * a)) ↔ (a ≥ -1 ∧ a < 3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l898_89855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_difference_factors_1620_l898_89865

theorem min_difference_factors_1620 :
  ∃ (a b : ℕ), 
    a * b = 1620 ∧ 
    (∀ (x y : ℕ), x * y = 1620 → |Int.ofNat a - Int.ofNat b| ≤ |Int.ofNat x - Int.ofNat y|) ∧
    |Int.ofNat a - Int.ofNat b| = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_difference_factors_1620_l898_89865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_sphere_volume_ratio_l898_89880

/-- Given a sphere and a cone where the height and base diameter of the cone
    are both equal to the diameter of the sphere, the ratio of the volume of
    the cone to the volume of the sphere is 1/2. -/
theorem cone_sphere_volume_ratio (r : ℝ) (hr : r > 0) : 
  (1/3 * Real.pi * (2*r)^2 * (2*r)) / ((4/3) * Real.pi * r^3) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_sphere_volume_ratio_l898_89880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intercepted_segment_length_l898_89828

/-- Line l in parametric form -/
def line (t : ℝ) : ℝ × ℝ := (1 + t, 1 + 2*t)

/-- Curve C in polar form -/
noncomputable def curve (θ : ℝ) : ℝ := 2 * Real.cos θ

/-- Length of the intercepted line segment -/
noncomputable def intercepted_length : ℝ := 4 * Real.sqrt 5 / 5

theorem intercepted_segment_length :
  ∃ (t₁ t₂ : ℝ), t₁ ≠ t₂ ∧
  (∃ (θ₁ θ₂ : ℝ),
    line t₁ = (curve θ₁ * Real.cos θ₁, curve θ₁ * Real.sin θ₁) ∧
    line t₂ = (curve θ₂ * Real.cos θ₂, curve θ₂ * Real.sin θ₂) ∧
    Real.sqrt ((line t₂).1 - (line t₁).1)^2 + ((line t₂).2 - (line t₁).2)^2 = intercepted_length) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intercepted_segment_length_l898_89828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_arithmetic_sequence_not_in_range_l898_89846

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

noncomputable def frac (x : ℝ) : ℝ := x - floor x

noncomputable def f (x : ℝ) : ℝ := floor (x^2) + frac x

def arith_seq (a₀ d : ℚ) (n : ℕ) : ℚ := a₀ + n • d

theorem existence_of_arithmetic_sequence_not_in_range :
  ∃ (a₀ d : ℚ),
    (∀ n : ℕ, (arith_seq a₀ d n).den = 3) ∧
    (∀ n m : ℕ, n ≠ m → arith_seq a₀ d n ≠ arith_seq a₀ d m) ∧
    (∀ n : ℕ, ∀ x : ℝ, x > 0 → f x ≠ arith_seq a₀ d n) := by
  sorry

#check existence_of_arithmetic_sequence_not_in_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_arithmetic_sequence_not_in_range_l898_89846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_with_120_degree_angle_has_one_obtuse_angle_l898_89808

/-- Definition of an angle -/
structure Angle where
  value : ℝ
  property : 0 < value ∧ value < 180

/-- Definition of a side -/
structure Side where
  length : ℝ
  property : length > 0

/-- Definition of a triangle -/
structure Triangle where
  angles : Finset Angle
  sides : Finset Side
  angleSum : (angles.toList.map (λ a => a.value)).sum = 180
  sideCount : sides.card = 3
  angleCount : angles.card = 3

/-- Definition of an isosceles triangle -/
def IsoscelesTriangle (t : Triangle) : Prop :=
  ∃ (s1 s2 : Side), s1 ∈ t.sides ∧ s2 ∈ t.sides ∧ s1 ≠ s2 ∧ s1.length = s2.length

/-- An isosceles triangle with one 120-degree angle has exactly one obtuse angle. -/
theorem isosceles_triangle_with_120_degree_angle_has_one_obtuse_angle :
  ∀ (t : Triangle),
    IsoscelesTriangle t →
    (∃ (a : Angle), a ∈ t.angles ∧ a.value = 120) →
    (t.angles.filter (λ a => a.value > 90)).card = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_with_120_degree_angle_has_one_obtuse_angle_l898_89808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_and_simplify_l898_89843

noncomputable def original_expression : ℝ := (Real.sqrt 12 + Real.sqrt 5) / (Real.sqrt 3 + Real.sqrt 5)

noncomputable def simplified_expression : ℝ := (Real.sqrt 15 - 1) / 2

theorem rationalize_and_simplify : original_expression = simplified_expression := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_and_simplify_l898_89843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reachable_arrangement_no_two_counter_swap_l898_89849

/-- A counter on the grid -/
structure Counter where
  x : ℕ
  y : ℕ

/-- The grid configuration -/
def GridConfig (n : ℕ) := Fin (n * n) → Counter

/-- A move on the grid -/
inductive Move (n : ℕ)
| horizontal : ℕ → Bool → Move n  -- row number, direction (true for right, false for left)
| vertical : ℕ → Bool → Move n    -- column number, direction (true for down, false for up)

/-- A sequence of moves -/
def MoveSequence (n : ℕ) := List (Move n)

/-- Predicate to check if a move sequence is returning -/
def is_returning (n : ℕ) (seq : MoveSequence n) (initial : GridConfig n) : Prop := sorry

/-- Predicate to check if two configurations are distinguishably equivalent -/
def is_distinguishably_equivalent (n : ℕ) (config1 config2 : GridConfig n) : Prop := sorry

/-- Apply a single move to a grid configuration -/
def apply_move (n : ℕ) (move : Move n) (config : GridConfig n) : GridConfig n := sorry

/-- Apply a sequence of moves to a grid configuration -/
def apply_sequence (n : ℕ) : MoveSequence n → GridConfig n → GridConfig n
| [], config => config
| (move :: rest), config => apply_sequence n rest (apply_move n move config)

/-- Theorem for part (a) -/
theorem reachable_arrangement (n : ℕ) (h : n > 1) :
  ∀ (initial target : GridConfig n),
  is_distinguishably_equivalent n initial target →
  ∃ (seq : MoveSequence n), is_returning n seq initial ∧
  is_distinguishably_equivalent n (apply_sequence n seq initial) target := sorry

/-- Theorem for part (b) -/
theorem no_two_counter_swap (n : ℕ) (h : n > 1) :
  ∀ (initial : GridConfig n) (i j : Fin (n * n)),
  i ≠ j →
  ¬∃ (seq : MoveSequence n),
    is_returning n seq initial ∧
    (∀ (k : Fin (n * n)), k ≠ i ∧ k ≠ j →
      (apply_sequence n seq initial k) = (initial k)) ∧
    (apply_sequence n seq initial i) = (initial j) ∧
    (apply_sequence n seq initial j) = (initial i) := sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reachable_arrangement_no_two_counter_swap_l898_89849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l898_89892

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 1) / (x - 2)

-- Define the domain
def domain (x : ℝ) : Prop := x ≥ 1 ∧ x ≠ 2

-- Theorem statement
theorem f_domain : ∀ x : ℝ, (∃ y : ℝ, f x = y) ↔ domain x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l898_89892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_refracted_rays_angle_l898_89812

noncomputable def air_refractive_index : ℝ := 1

noncomputable def glass_refractive_index : ℝ := 1.6

noncomputable def first_refraction_angle : ℝ := 30 * Real.pi / 180

def perpendicular_rays (α β : ℝ) : Prop :=
  α + β = Real.pi / 2

theorem refracted_rays_angle (α β γ : ℝ) 
  (h1 : perpendicular_rays α β)
  (h2 : air_refractive_index * Real.sin α = glass_refractive_index * Real.sin first_refraction_angle)
  (h3 : air_refractive_index * Real.sin β = glass_refractive_index * Real.sin γ) :
  first_refraction_angle + γ = 52 * Real.pi / 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_refracted_rays_angle_l898_89812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_cosine_l898_89822

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 4 + (Real.log (x + 4)) / (Real.log a)

-- Define the fixed point P
def P : ℝ × ℝ := (-3, 4)

-- Theorem statement
theorem fixed_point_cosine (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let (x, y) := P
  Real.cos (Real.arctan (y / x)) = -3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_cosine_l898_89822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_hexagon_semicircles_l898_89881

/-- The area of the shaded region in a regular hexagon with side length 2 and six inscribed semicircles --/
theorem shaded_area_hexagon_semicircles : 
  ∀ (hexagon_side_length : ℝ) (semicircle_radius : ℝ),
    hexagon_side_length = 2 →
    semicircle_radius = 1 →
    ∃ (shaded_area : ℝ),
      shaded_area = (3 * Real.sqrt 3 * hexagon_side_length^2 / 2) - (6 * π * semicircle_radius^2 / 2) ∧
      shaded_area = 6 * Real.sqrt 3 - 3 * π := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_hexagon_semicircles_l898_89881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_target_scores_theorem_l898_89844

/-- Represents the scores of a target with three fields -/
structure TargetScores where
  small : ℕ
  middle : ℕ
  large : ℕ
  small_lt_middle : small < middle
  middle_lt_large : middle < large

/-- Checks if a given set of target scores satisfies the problem conditions -/
def satisfiesConditions (scores : TargetScores) : Prop :=
  let j := scores.small
  let m := scores.middle
  let p := scores.large
  (m = j + 5) ∧ (p = m + 5) ∧ (12 = j ∨ 12 = m ∨ 12 = p)

/-- The theorem stating the possible target scores -/
theorem target_scores_theorem :
  ∀ scores : TargetScores,
    satisfiesConditions scores →
    (scores.small, scores.middle, scores.large) = (2, 7, 12) ∨
    (scores.small, scores.middle, scores.large) = (7, 12, 17) ∨
    (scores.small, scores.middle, scores.large) = (12, 17, 22) :=
by
  sorry

#check target_scores_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_target_scores_theorem_l898_89844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cheerleader_ratio_l898_89884

/-- Represents the ratio of two numbers -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ
deriving Repr

/-- Represents a group of cheerleaders -/
structure CheerleaderGroup where
  males : ℕ
  females : ℕ
  malesMalt : ℕ
  femalesMalt : ℕ

def totalCheerleaders (group : CheerleaderGroup) : ℕ :=
  group.males + group.females

def totalMalt (group : CheerleaderGroup) : ℕ :=
  group.malesMalt + group.femalesMalt

def totalCoke (group : CheerleaderGroup) : ℕ :=
  totalCheerleaders group - totalMalt group

def maltToCokeRatio (group : CheerleaderGroup) : Ratio :=
  { numerator := totalMalt group,
    denominator := totalCoke group }

theorem cheerleader_ratio (group : CheerleaderGroup)
  (h1 : group.males = 10)
  (h2 : group.females = 16)
  (h3 : group.malesMalt = 6)
  (h4 : group.femalesMalt = 8) :
  maltToCokeRatio group = { numerator := 7, denominator := 6 } := by
  sorry

#eval maltToCokeRatio { males := 10, females := 16, malesMalt := 6, femalesMalt := 8 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cheerleader_ratio_l898_89884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l898_89848

noncomputable section

def a (x : ℝ) : Fin 2 → ℝ := ![2, x]
def b (y : ℝ) : Fin 2 → ℝ := ![y, 1]
def c : Fin 2 → ℝ := ![3, -3]

def perpendicular (u v : Fin 2 → ℝ) : Prop :=
  u 0 * v 0 + u 1 * v 1 = 0

def parallel (u v : Fin 2 → ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ u 0 = k * v 0 ∧ u 1 = k * v 1

noncomputable def magnitude (v : Fin 2 → ℝ) : ℝ :=
  Real.sqrt (v 0 ^ 2 + v 1 ^ 2)

def dot_product (u v : Fin 2 → ℝ) : ℝ :=
  u 0 * v 0 + u 1 * v 1

noncomputable def cosine_angle (u v : Fin 2 → ℝ) : ℝ :=
  dot_product u v / (magnitude u * magnitude v)

theorem vector_problem (x y : ℝ) 
  (h1 : perpendicular (a x) (b y))
  (h2 : parallel (b y) c) :
  magnitude (λ i => a x i + b y i) = Real.sqrt 10 ∧
  cosine_angle 
    (λ i => a x i + b y i) 
    (λ i => a x i + 2 * b y i + c i) = 3/5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l898_89848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conditional_probability_20_to_25_l898_89811

/-- Given:
  - p_20: Probability of living to 20 years old from birth
  - p_25: Probability of living to 25 years old from birth
  
  Prove: The conditional probability of a 20-year-old animal living to 25 years old is 0.5
-/
theorem conditional_probability_20_to_25 (p_20 p_25 : ℝ) 
  (h1 : p_20 = 0.8) 
  (h2 : p_25 = 0.4) : 
  p_25 / p_20 = 0.5 := by
  rw [h1, h2]
  norm_num

#check conditional_probability_20_to_25

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conditional_probability_20_to_25_l898_89811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_odd_sum_two_darts_l898_89839

/-- Represents a region on the dartboard -/
structure Region where
  inner : Bool
  value : Nat

/-- Represents the dartboard -/
def Dartboard : Type := List Region

/-- The dartboard configuration -/
def dartboard : Dartboard := [
  {inner := true,  value := 0},
  {inner := true,  value := 3},
  {inner := true,  value := 5},
  {inner := false, value := 4},
  {inner := false, value := 6},
  {inner := false, value := 3}
]

/-- The radius of the outer circle -/
noncomputable def outerRadius : ℝ := 10

/-- The radius of the inner circle -/
noncomputable def innerRadius : ℝ := 5

/-- Calculates the area of a region -/
noncomputable def regionArea (r : Region) : ℝ :=
  if r.inner then
    Real.pi * innerRadius^2 / 3
  else
    Real.pi * (outerRadius^2 - innerRadius^2) / 3

/-- Calculates the total area of the dartboard -/
noncomputable def totalArea : ℝ := Real.pi * outerRadius^2

/-- Calculates the probability of a dart landing in a specific region -/
noncomputable def regionProbability (r : Region) : ℝ := regionArea r / totalArea

/-- Determines if a score is odd -/
def isOdd (n : Nat) : Bool := n % 2 = 1

/-- The main theorem to prove -/
theorem probability_odd_sum_two_darts :
  (let probOdd := (dartboard.filter (fun r => isOdd r.value)).map regionProbability |>.sum
   let probEven := 1 - probOdd
   2 * probOdd * probEven) = 4/9 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_odd_sum_two_darts_l898_89839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_for_single_zero_l898_89841

-- Define the piecewise function f(x)
noncomputable def f (a : ℤ) : ℝ → ℝ := fun x =>
  if x ≤ 0 then x + Real.exp (x * Real.log 3) else (1/3) * x^3 - 4*x + a

-- State the theorem
theorem min_a_for_single_zero : 
  (∃ (a : ℤ), (∃! x : ℝ, f a x = 0) ∧ 
   (∀ b : ℤ, b < a → ¬(∃! x : ℝ, f b x = 0))) → 
  (∃ (a : ℤ), a = 6 ∧ (∃! x : ℝ, f a x = 0) ∧ 
   (∀ b : ℤ, b < a → ¬(∃! x : ℝ, f b x = 0))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_for_single_zero_l898_89841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_comparison_l898_89890

noncomputable def f (x : ℝ) : ℝ := 5 / x

theorem inverse_proportion_comparison : f 2 > f 3 := by
  -- Unfold the definition of f
  unfold f
  -- Simplify the fractions
  simp
  -- Use the fact that 5/2 > 5/3
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_comparison_l898_89890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l898_89874

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 2) + 1 / (x^2 - 4)

-- Define the domain of f
def domain_f : Set ℝ := {x | x > -2 ∧ x ≠ 2}

-- Theorem stating that the domain of f is (-2, 2) ∪ (2, +∞)
theorem domain_of_f : domain_f = Set.Ioo (-2 : ℝ) 2 ∪ Set.Ioi 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l898_89874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_corresponding_angles_equivalence_l898_89810

/-- Two angles are corresponding if they are in the same relative position when a line intersects two other lines. -/
def are_corresponding_angles (α β : Real) : Prop := sorry

/-- The statement "corresponding angles are equal" -/
def corresponding_angles_are_equal : Prop :=
  ∀ α β : Real, are_corresponding_angles α β → α = β

/-- The if-then form of the statement -/
def if_corresponding_then_equal : Prop :=
  ∀ α β : Real, are_corresponding_angles α β → α = β

/-- Theorem stating the equivalence of the two forms -/
theorem corresponding_angles_equivalence :
  corresponding_angles_are_equal ↔ if_corresponding_then_equal := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_corresponding_angles_equivalence_l898_89810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_digits_sum_divisible_by_ten_l898_89819

/-- The frequency of a digit in a natural number -/
def digit_frequency (n : ℕ) (d : Fin 10) : ℕ :=
  sorry

/-- Given two natural numbers with the same digit frequency, if their sum is 10^1000, then both are divisible by 10 -/
theorem same_digits_sum_divisible_by_ten (a b : ℕ) 
  (h_same_digits : ∀ d : Fin 10, (digit_frequency a d) = (digit_frequency b d))
  (h_sum : a + b = 10^1000) : 
  10 ∣ a ∧ 10 ∣ b := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_digits_sum_divisible_by_ten_l898_89819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_a1_value_l898_89875

-- Define a geometric sequence
noncomputable def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a₁ * q ^ (n - 1)

-- Define the partial sum of a geometric sequence
noncomputable def partial_sum (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a₁
  else a₁ * (1 - q^n) / (1 - q)

-- Theorem statement
theorem geometric_sequence_a1_value
  (a₁ : ℝ)
  (q : ℝ)
  (h_q_pos : q > 0)
  (h_S2 : partial_sum a₁ q 2 = 3 * (geometric_sequence a₁ q 2) + 2)
  (h_S4 : partial_sum a₁ q 4 = 3 * (geometric_sequence a₁ q 4) + 2) :
  a₁ = -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_a1_value_l898_89875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Cr_is_21_31_l898_89829

/-- Molar mass of Hydrogen -/
noncomputable def molar_mass_H : ℝ := 1.01

/-- Molar mass of Chromium -/
noncomputable def molar_mass_Cr : ℝ := 52.00

/-- Molar mass of Oxygen -/
noncomputable def molar_mass_O : ℝ := 16.00

/-- Molar mass of Potassium -/
noncomputable def molar_mass_K : ℝ := 39.10

/-- Molar mass of Manganese -/
noncomputable def molar_mass_Mn : ℝ := 54.94

/-- Molar mass of H2CrO4 -/
noncomputable def molar_mass_H2CrO4 : ℝ := 2 * molar_mass_H + molar_mass_Cr + 4 * molar_mass_O

/-- Molar mass of KMnO4 -/
noncomputable def molar_mass_KMnO4 : ℝ := molar_mass_K + molar_mass_Mn + 4 * molar_mass_O

/-- Molar mass of the resulting compound -/
noncomputable def molar_mass_result : ℝ := molar_mass_H2CrO4 + molar_mass_KMnO4 - 2 * molar_mass_O

/-- Mass percentage of Cr in the resulting compound -/
noncomputable def mass_percentage_Cr : ℝ := (molar_mass_Cr / molar_mass_result) * 100

theorem mass_percentage_Cr_is_21_31 : 
  ∃ ε > 0, |mass_percentage_Cr - 21.31| < ε :=
by
  sorry -- Skipping the proof for now

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Cr_is_21_31_l898_89829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_concyclic_l898_89858

/-- An ellipse in a 2D plane -/
structure Ellipse where
  a : ℝ
  b : ℝ
  (positive_a : 0 < a)
  (positive_b : 0 < b)

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in a 2D plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Definition: A point is on an ellipse -/
def Point.on_ellipse (p : Point) (e : Ellipse) : Prop :=
  (p.x^2 / e.a^2) + (p.y^2 / e.b^2) = 1

/-- Definition: A line passes through a point -/
def Line.passes_through (l : Line) (p : Point) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- Definition: Two lines intersect an ellipse -/
def intersect_ellipse (l₁ l₂ : Line) (e : Ellipse) (A B C D : Point) : Prop :=
  A.on_ellipse e ∧ B.on_ellipse e ∧ C.on_ellipse e ∧ D.on_ellipse e ∧
  l₁.passes_through A ∧ l₁.passes_through B ∧
  l₂.passes_through C ∧ l₂.passes_through D

/-- Definition: Points are concyclic -/
def concyclic (A B C D : Point) : Prop :=
  ∃ (center : Point) (radius : ℝ),
    (A.x - center.x)^2 + (A.y - center.y)^2 = radius^2 ∧
    (B.x - center.x)^2 + (B.y - center.y)^2 = radius^2 ∧
    (C.x - center.x)^2 + (C.y - center.y)^2 = radius^2 ∧
    (D.x - center.x)^2 + (D.y - center.y)^2 = radius^2

theorem ellipse_intersection_concyclic 
  (e : Ellipse) (P : Point) (l₁ l₂ : Line) (A B C D : Point) (α β : ℝ) :
  ¬P.on_ellipse e →
  intersect_ellipse l₁ l₂ e A B C D →
  l₁.slope = Real.tan α →
  l₂.slope = Real.tan β →
  α + β = π →
  concyclic A B C D :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_concyclic_l898_89858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_of_cones_l898_89861

noncomputable section

/-- The volume of a cone with radius r and height h -/
def cone_volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

/-- Cone C with height 16 and radius 8 -/
def cone_C : ℝ × ℝ := (16, 8)

/-- Cone D with height 8 and radius 16 -/
def cone_D : ℝ × ℝ := (8, 16)

theorem volume_ratio_of_cones :
  let vc := cone_volume cone_C.2 cone_C.1
  let vd := cone_volume cone_D.2 cone_D.1
  vc / vd = 1 / 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_of_cones_l898_89861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_travel_distance_proof_l898_89895

/-- The distance traveled by the center of a ball rolling along a track with three arcs -/
noncomputable def ball_travel_distance (ball_diameter : ℝ) (R₁ R₂ R₃ : ℝ) : ℝ :=
  let r := ball_diameter / 2
  let R₁' := R₁ - r
  let R₂' := R₂ + r
  let R₃' := R₃ - r
  Real.pi * (R₁' + R₂' + R₃')

/-- Theorem stating that the distance traveled by the center of the ball is 467π inches -/
theorem ball_travel_distance_proof :
  ball_travel_distance 6 150 200 120 = 467 * Real.pi :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_travel_distance_proof_l898_89895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_sum_l898_89851

theorem min_value_of_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_log : Real.log b / Real.log a = -1) :
  a + 4 * b ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_sum_l898_89851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_10_eq_110_l898_89807

/-- A sequence of integers satisfying the given recurrence relation -/
def b : ℕ → ℤ
  | 0 => 2  -- We use 0 to represent the first term
  | (n + 1) => b (n / 2) + b ((n + 1) / 2) + 2 * ((n / 2) + 1) * (((n + 1) / 2) + 1)

/-- The 10th term of the sequence is 110 -/
theorem b_10_eq_110 : b 9 = 110 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_10_eq_110_l898_89807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_antelope_gnu_max_distance_antelope_gnu_switch_point_l898_89870

/-- Represents the wear rate of a tire in a specific position --/
structure TireWear where
  front : ℕ
  rear : ℕ

/-- Calculates the maximum distance a car can travel given tire wear rates --/
noncomputable def maxDistance (wear : TireWear) : ℚ :=
  (wear.front * wear.rear : ℚ) / (wear.front + wear.rear)

/-- Theorem: The maximum distance for the "Antelope Gnu" car --/
theorem antelope_gnu_max_distance :
  let wear := TireWear.mk 25000 15000
  maxDistance wear = 18750 := by
  sorry

/-- Theorem: The optimal switch point for the "Antelope Gnu" car --/
theorem antelope_gnu_switch_point :
  let wear := TireWear.mk 25000 15000
  maxDistance wear / 2 = 9375 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_antelope_gnu_max_distance_antelope_gnu_switch_point_l898_89870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_box_volume_increase_l898_89831

def box_volume (l w h : ℝ) : ℝ := l * w * h

def box_surface_area (l w h : ℝ) : ℝ := 2 * (l * w + w * h + l * h)

def box_edge_sum (l w h : ℝ) : ℝ := 4 * (l + w + h)

theorem box_volume_increase 
  (l w h : ℝ) 
  (hv : box_volume l w h = 4320)
  (hs : box_surface_area l w h = 1704)
  (he : box_edge_sum l w h = 208) :
  box_volume (l + 2) (w + 2) (h + 2) = 6240 := by
  sorry

#check box_volume_increase

end NUMINAMATH_CALUDE_ERRORFEEDBACK_box_volume_increase_l898_89831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reach_boundary_l898_89823

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a half-space in 3D -/
structure HalfSpace where
  normal : Point3D  -- Unit normal vector of the boundary plane
  offset : ℝ        -- Signed distance from origin to boundary plane

/-- Checks if a point is inside the half-space -/
def isInside (hs : HalfSpace) (p : Point3D) : Prop :=
  hs.normal.x * p.x + hs.normal.y * p.y + hs.normal.z * p.z < hs.offset

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

/-- Radius of the sphere inscribed around a regular octahedron -/
noncomputable def R : ℝ := 1  -- Simplified for demonstration; actual value would depend on octahedron properties

/-- Main theorem: For any point in the half-space, there exists a direction to reach the boundary within distance R -/
theorem reach_boundary (hs : HalfSpace) (p : Point3D) (h : isInside hs p) :
  ∃ (direction : Point3D), 
    let boundary_point := Point3D.mk (p.x + direction.x * R) (p.y + direction.y * R) (p.z + direction.z * R)
    ¬(isInside hs boundary_point) ∧ 
    distance p boundary_point ≤ R := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reach_boundary_l898_89823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_is_30_degrees_l898_89847

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x - Real.sqrt 3 * y - 1 = 0

-- Define the slope of the line
noncomputable def line_slope : ℝ := Real.sqrt 3 / 1

-- Define the slope angle in radians
noncomputable def slope_angle : ℝ := Real.arctan (Real.sqrt 3 / 3)

-- Theorem statement
theorem slope_angle_is_30_degrees :
  slope_angle = 30 * Real.pi / 180 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_is_30_degrees_l898_89847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_squared_range_l898_89888

theorem y_squared_range (y : ℝ) (h : (y + 25) ^ (1/3) - (y - 25) ^ (1/3) = 4) :
  615 < y^2 ∧ y^2 < 635 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_squared_range_l898_89888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_mold_radius_l898_89885

noncomputable def hemisphere_volume (r : ℝ) : ℝ := (2 / 3) * Real.pi * r^3

theorem smaller_mold_radius :
  let large_bowl_radius : ℝ := 2
  let number_of_molds : ℕ := 64
  let large_bowl_volume : ℝ := hemisphere_volume large_bowl_radius
  ∀ small_mold_radius : ℝ,
    (number_of_molds : ℝ) * hemisphere_volume small_mold_radius = large_bowl_volume →
    small_mold_radius = 1 / 2 := by
  intro small_mold_radius hypothesis
  -- The proof steps would go here
  sorry

#check smaller_mold_radius

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_mold_radius_l898_89885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zoo_exhibits_count_l898_89806

-- Define the number of animals in each exhibit
def rain_forest : ℕ := sorry
def reptile_house : ℕ := sorry
def aquarium : ℕ := sorry
def aviary : ℕ := sorry

-- Define the conditions
axiom reptile_house_def : reptile_house = 3 * rain_forest - 5
axiom reptile_house_count : reptile_house = 16
axiom aquarium_def : aquarium = 2 * reptile_house
axiom aviary_def : aviary = (aquarium - rain_forest) + 3

-- Theorem to prove
theorem zoo_exhibits_count :
  rain_forest = 7 ∧
  reptile_house = 16 ∧
  aquarium = 32 ∧
  aviary = 28 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zoo_exhibits_count_l898_89806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_in_xz_plane_l898_89825

theorem equidistant_point_in_xz_plane :
  let p : ℝ × ℝ × ℝ := (19/6, 0, 5/6)
  let a : ℝ × ℝ × ℝ := (1, 0, 0)
  let b : ℝ × ℝ × ℝ := (2, 2, 2)
  let c : ℝ × ℝ × ℝ := (3, 3, -1)
  let dist (x y : ℝ × ℝ × ℝ) := Real.sqrt ((x.1 - y.1)^2 + (x.2.1 - y.2.1)^2 + (x.2.2 - y.2.2)^2)
  dist p a = dist p b ∧ dist p b = dist p c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_in_xz_plane_l898_89825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_area_in_circle_l898_89857

/-- The area of a regular octagon inscribed in a circle with radius 3 units -/
noncomputable def octagonArea : ℝ := 18 * Real.sqrt 2

/-- The radius of the circle in which the octagon is inscribed -/
def circleRadius : ℝ := 3

theorem octagon_area_in_circle (r : ℝ) (h : r = circleRadius) :
  octagonArea = 8 * (1/2 * r^2 * Real.sin (135 * π / 180)) := by
  sorry

#check octagon_area_in_circle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_area_in_circle_l898_89857
