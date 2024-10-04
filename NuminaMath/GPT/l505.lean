import Mathlib
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.ContinuedFractions.Computation.Basic
import Mathlib.Algebra.Factorial
import Mathlib.Algebra.Functional
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.GroupPower
import Mathlib.Analysis.Calculus.Continuous
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Data.Finset
import Mathlib.Data.Int.Basic
import Mathlib.Data.Int.ModEq
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Nat.Primes
import Mathlib.Data.Real.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Init.Data.Nat.Basic
import Mathlib.Probability.Basic
import Mathlib.ProbabilityTheory.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

namespace find_K_l505_505146

noncomputable def Z (K : ℕ) : ℕ := K^3

def is_perfect_square (n : ℕ) : Prop :=
  ∃ (m : ℕ), m * m = n

def K_values : Finset ℕ := {4, 9, 16}

theorem find_K (K : ℕ) (hK : K > 1) (hZ : 50 < Z K ∧ Z K < 5000) :
  is_perfect_square (Z K) ↔ K ∈ K_values :=
by
  sorry

end find_K_l505_505146


namespace least_positive_integer_with_12_factors_l505_505295

def has_exactly_12_factors (n : ℕ) : Prop :=
  (finset.range(n + 1).filter (λ d, n % d = 0)).card = 12

theorem least_positive_integer_with_12_factors : ∀ n : ℕ, has_exactly_12_factors n → n ≥ 150 :=
sorry

end least_positive_integer_with_12_factors_l505_505295


namespace i_603_sum_l505_505485

-- Given conditions
def i : ℂ := complex.i -- Define i as the imaginary unit, ℂ is the set of complex numbers
theorem i_603_sum :
  (∑ k in (range 604), i^(603 - k)) = 0 := by
{
  -- Proof would go here
  sorry
}

end i_603_sum_l505_505485


namespace least_positive_integer_with_12_factors_is_96_l505_505273

theorem least_positive_integer_with_12_factors_is_96 :
  ∃ n : ℕ, (n = 96 ∧ (∀ m : ℕ, (m > 0 ∧ ∀ d : ℕ, d ∣ m → d > 0 → d < m → (∃ k : ℕ, d = k ∨ k * d = m) → k > 1 → m > n))) ∧ (∃ m : ℕ, m = 12 ∧ (∀ d : ℕ, d ∣ m → d ∈ {1, m} ∨ ∀ k : ℕ, k * d = m → d = k ∨ d < k)).
sorry

end least_positive_integer_with_12_factors_is_96_l505_505273


namespace find_c_l505_505755

theorem find_c
  (m b d c : ℝ)
  (h : m = b * d * c / (d + c)) :
  c = m * d / (b * d - m) :=
sorry

end find_c_l505_505755


namespace least_positive_integer_with_12_factors_l505_505375

theorem least_positive_integer_with_12_factors : 
  ∃ (n : ℕ), (n > 0) ∧ (nat.factors_count n = 12) ∧ (∀ m, (m > 0) → (nat.factors_count m = 12) → n ≤ m) ∧ n = 108 :=
sorry

end least_positive_integer_with_12_factors_l505_505375


namespace fraction_less_than_mode_l505_505712

def lst : List ℕ := [3, 4, 5, 5, 5, 5, 7, 11, 21]

def mode (l : List ℕ) : ℕ :=
  l.foldr (λ x m, if l.count x > l.count m then x else m) l.head!

def count_less_than_mode (l : List ℕ) (m : ℕ) : ℕ :=
  l.filter (λ x, x < m).length

theorem fraction_less_than_mode :
  count_less_than_mode lst (mode lst) = 2 → lst.length = 9 → (2 / 9 : ℚ) = 2 / 9 :=
by
  intro h1 h2
  sorry

end fraction_less_than_mode_l505_505712


namespace length_of_boat_l505_505727

-- Define Josie's jogging variables and problem conditions
variables (L J B : ℝ)
axiom eqn1 : 130 * J = L + 130 * B
axiom eqn2 : 70 * J = L - 70 * B

-- The theorem to prove that the length of the boat L equals 91 steps (i.e., 91 * J)
theorem length_of_boat : L = 91 * J :=
by
  sorry

end length_of_boat_l505_505727


namespace lydia_eats_apple_age_l505_505722

-- Define the conditions
def years_to_bear_fruit : ℕ := 7
def age_when_planted : ℕ := 4
def current_age : ℕ := 9

-- Define the theorem statement
theorem lydia_eats_apple_age : 
  (age_when_planted + years_to_bear_fruit = 11) :=
by
  sorry

end lydia_eats_apple_age_l505_505722


namespace least_positive_integer_with_12_factors_is_972_l505_505357

theorem least_positive_integer_with_12_factors_is_972 : ∃ k : ℕ, (∀ n : ℕ, (∃ (d : ℕ), d * k = n) ↔ n = 12) ∧ k = 972 := sorry

end least_positive_integer_with_12_factors_is_972_l505_505357


namespace range_of_f_l505_505954

def range_sqrt_sub_exp (f : ℝ → ℝ) : Set ℝ :=
  {y : ℝ | ∃ x : ℝ, f x = y}

noncomputable def f (x : ℝ) : ℝ := real.sqrt (16 - real.exp x (real.log 4))

theorem range_of_f :
  range_sqrt_sub_exp f = set.Ico 0 4 :=
sorry

end range_of_f_l505_505954


namespace cylinder_volume_l505_505820

noncomputable def volume_of_cylinder (d h : ℝ) : ℝ :=
  let r := d / 2
  in Real.pi * r^2 * h

theorem cylinder_volume :
  volume_of_cylinder 14 5 = 769.36 :=
by sorry

end cylinder_volume_l505_505820


namespace practice_other_days_l505_505467

-- Defining the total practice time for the week and the practice time for two days 
variable (total_minutes_week : ℤ) (total_minutes_two_days : ℤ)

-- Given conditions
axiom total_minutes_week_eq : total_minutes_week = 450
axiom total_minutes_two_days_eq : total_minutes_two_days = 172

-- The proof goal
theorem practice_other_days : (total_minutes_week - total_minutes_two_days) = 278 :=
by
  rw [total_minutes_week_eq, total_minutes_two_days_eq]
  show 450 - 172 = 278
  -- The proof goes here
  sorry

end practice_other_days_l505_505467


namespace least_positive_integer_with_12_factors_is_96_l505_505261

theorem least_positive_integer_with_12_factors_is_96 :
  ∃ n : ℕ, (n = 96 ∧ (∀ m : ℕ, (m > 0 ∧ ∀ d : ℕ, d ∣ m → d > 0 → d < m → (∃ k : ℕ, d = k ∨ k * d = m) → k > 1 → m > n))) ∧ (∃ m : ℕ, m = 12 ∧ (∀ d : ℕ, d ∣ m → d ∈ {1, m} ∨ ∀ k : ℕ, k * d = m → d = k ∨ d < k)).
sorry

end least_positive_integer_with_12_factors_is_96_l505_505261


namespace four_digit_multiples_of_5_count_l505_505622

theorem four_digit_multiples_of_5_count :
  let lower_bound := 200
  let upper_bound := 1999
  (upper_bound - lower_bound + 1) = 1800 :=
by
  let lower_bound := 200
  let upper_bound := 1999
  show (upper_bound - lower_bound + 1) = 1800,
  from sorry

end four_digit_multiples_of_5_count_l505_505622


namespace bird_difference_l505_505706

-- Variables representing given conditions
def num_migrating_families : Nat := 86
def num_remaining_families : Nat := 45
def avg_birds_per_migrating_family : Nat := 12
def avg_birds_per_remaining_family : Nat := 8

-- Definition to calculate total number of birds for migrating families
def total_birds_migrating : Nat := num_migrating_families * avg_birds_per_migrating_family

-- Definition to calculate total number of birds for remaining families
def total_birds_remaining : Nat := num_remaining_families * avg_birds_per_remaining_family

-- The statement that we need to prove
theorem bird_difference (h : total_birds_migrating - total_birds_remaining = 672) : 
  total_birds_migrating - total_birds_remaining = 672 := 
sorry

end bird_difference_l505_505706


namespace small_angle_9_15_l505_505613

theorem small_angle_9_15 : 
  let hours_to_angle := 30
  let minute_to_angle := 6
  let minutes := 15
  let hour := 9
  let hour_position := hour * hours_to_angle
  let additional_hour_angle := (hours_to_angle * (minutes / 60.0))
  let final_hour_position := hour_position + additional_hour_angle
  let minute_position := (minutes / 5.0) * hours_to_angle
  let angle := abs (final_hour_position - minute_position)
  let small_angle := if angle > 180 then 360 - angle else angle
  in small_angle = 187.5 := by 
  sorry

end small_angle_9_15_l505_505613


namespace limit_leq_l505_505077

variables {α : Type*} [LinearOrder α] [TopologicalSpace α] [OrderTopology α]

theorem limit_leq {a_n b_n : ℕ → α} {a b : α}
  (ha : Filter.Tendsto a_n Filter.atTop (nhds a))
  (hb : Filter.Tendsto b_n Filter.atTop (nhds b))
  (h_leq : ∀ n, a_n n ≤ b_n n)
  : a ≤ b :=
by
  -- Proof will be constructed here
  sorry

end limit_leq_l505_505077


namespace complex_fraction_value_l505_505570

noncomputable def z (a : ℝ) : ℂ := (a^2 - 1) + (a - 1) * complex.I

theorem complex_fraction_value (a : ℝ) (h : (a^2 - 1) = 0) (h1: a ≠ 1): 
  (a + complex.I ^ 2024) / (1 - complex.I) = 0 := by
  sorry

end complex_fraction_value_l505_505570


namespace four_digit_multiples_of_5_count_l505_505625

theorem four_digit_multiples_of_5_count :
  let lower_bound := 200
  let upper_bound := 1999
  (upper_bound - lower_bound + 1) = 1800 :=
by
  let lower_bound := 200
  let upper_bound := 1999
  show (upper_bound - lower_bound + 1) = 1800,
  from sorry

end four_digit_multiples_of_5_count_l505_505625


namespace rectangle_area_at_stage_8_l505_505684

-- Definitions based on conditions
def area_of_square (side_length : ℕ) : ℕ := side_length * side_length
def number_of_squares_in_stage (stage : ℕ) : ℕ := stage

-- The main theorem to prove
theorem rectangle_area_at_stage_8 : 
  area_of_square 4 * number_of_squares_in_stage 8 = 128 := by
  sorry

end rectangle_area_at_stage_8_l505_505684


namespace least_positive_integer_with_12_factors_l505_505208

/--
Prove that the smallest positive integer with exactly 12 positive factors is 60.
-/
theorem least_positive_integer_with_12_factors : 
  ∃ (n : ℕ), (0 < n) ∧ (nat.factors_count n = 12) ∧ (∀ m : ℕ, (0 < m) ∧ (nat.factors_count m = 12) → n ≤ m) → n = 60 :=
sorry

end least_positive_integer_with_12_factors_l505_505208


namespace inflating_time_l505_505448

theorem inflating_time:
  let t : ℕ := 20 in
  let time_alexia := 20 * t in
  let time_ermias := 25 * t in
  time_alexia + time_ermias = 900 → t = 20 :=
by sorry

end inflating_time_l505_505448


namespace complex_magnitude_and_imaginary_part_l505_505043

variable {a : ℝ} {b : ℝ} {z : ℂ}

def complexNumber (a b : ℝ) : ℂ := a + b * Complex.i
def magnitude (z : ℂ) : ℝ := Complex.abs z

theorem complex_magnitude_and_imaginary_part (a : ℝ) (b : ℝ) (z : ℂ) (h1: z = complexNumber a b) (h2: magnitude z = 15) (h3: a = 9) :
  (z * Complex.conj z = 225) ∧ (b = 12 ∨ b = -12) :=
by
  sorry

end complex_magnitude_and_imaginary_part_l505_505043


namespace least_positive_integer_with_12_factors_is_96_l505_505397

def has_exactly_12_factors (n : ℕ) : Prop :=
  (12 : ℕ) = (List.range (n + 1)).filter (λ x, n % x = 0).length

theorem least_positive_integer_with_12_factors_is_96 :
  n ∈ (ℕ) ∧ has_exactly_12_factors n → n = 96 :=
by
  sorry

end least_positive_integer_with_12_factors_is_96_l505_505397


namespace least_positive_integer_with_12_factors_l505_505205

/--
Prove that the smallest positive integer with exactly 12 positive factors is 60.
-/
theorem least_positive_integer_with_12_factors : 
  ∃ (n : ℕ), (0 < n) ∧ (nat.factors_count n = 12) ∧ (∀ m : ℕ, (0 < m) ∧ (nat.factors_count m = 12) → n ≤ m) → n = 60 :=
sorry

end least_positive_integer_with_12_factors_l505_505205


namespace expression_equals_two_l505_505754

noncomputable def expression (a b c : ℝ) : ℝ :=
  (1 + a) / (1 + a + a * b) + (1 + b) / (1 + b + b * c) + (1 + c) / (1 + c + c * a)

theorem expression_equals_two (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) :
  expression a b c = 2 := by
  sorry

end expression_equals_two_l505_505754


namespace least_positive_integer_with_12_factors_l505_505287

def num_factors (n : ℕ) : ℕ := 
  (finset.range (n + 1)).filter (λ d, n % d = 0).card

theorem least_positive_integer_with_12_factors : 
  ∃ n : ℕ, num_factors n = 12 ∧ n = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_l505_505287


namespace find_certain_number_l505_505862

theorem find_certain_number :
  ∃ C, ∃ A B, (A + B = 15) ∧ (A = 7) ∧ (C * B = 5 * A - 11) ∧ (C = 3) :=
by
  sorry

end find_certain_number_l505_505862


namespace least_positive_integer_with_12_factors_l505_505337

def has_exactly_12_factors (n : ℕ) : Prop :=
  n.factors.length = 12

theorem least_positive_integer_with_12_factors : ∃ k : ℕ, has_exactly_12_factors k ∧ (∀ n : ℕ, has_exactly_12_factors n → n ≥ k) ∧ k = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_l505_505337


namespace least_positive_integer_with_12_factors_l505_505179

theorem least_positive_integer_with_12_factors :
  ∃ m : ℕ, 0 < m ∧ (∀ n : ℕ, 0 < n → (n ≠ m →  m ≤ n)) ∧ (12 = (m.factors.toFinset.card)) := sorry

end least_positive_integer_with_12_factors_l505_505179


namespace major_axis_length_l505_505003

def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  real.sqrt ((x1 - x2) ^ 2 + (y1 - y2) ^ 2)

noncomputable def length_major_axis_of_ellipse (f1_x f1_y f2_x f2_y : ℝ) : ℝ :=
  distance f1_x f1_y f2_x (-f2_y)

theorem major_axis_length (f1_x f1_y f2_x f2_y : ℝ) (h : f1_x = 9 ∧ f1_y = 20 ∧ f2_x = 49 ∧ f2_y = 55) :
  length_major_axis_of_ellipse f1_x f1_y f2_x f2_y = 85 :=
by
  sorry

end major_axis_length_l505_505003


namespace smallest_value_of_expression_l505_505676

theorem smallest_value_of_expression (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a^2 - b^2 = 16) : 
  (∃ k : ℚ, k = (a + b) / (a - b) + (a - b) / (a + b) ∧ (∀ x : ℚ, x = (a + b) / (a - b) + (a - b) / (a + b) → x ≥ 9/4)) :=
sorry

end smallest_value_of_expression_l505_505676


namespace compound_interest_correct_l505_505899

-- define the problem conditions
def P : ℝ := 3000
def r : ℝ := 0.07
def n : ℕ := 25

-- the compound interest formula
def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r) ^ n

-- state the theorem we want to prove
theorem compound_interest_correct :
  compound_interest P r n = 16281 := 
by
  sorry

end compound_interest_correct_l505_505899


namespace least_pos_int_with_12_pos_factors_is_72_l505_505236

def least_positive_integer_with_12_factors (n : ℕ) : Prop :=
  ∃ (m : ℕ), m > 0 ∧ m ≠ n ∧ (∀ d : ℕ, d ∣ m → (d = n ∨ d = 1)) ∧
  ((∀ d : ℕ, d ∣ m → ∃ e : ℕ, e = 1) → n = 72)

theorem least_pos_int_with_12_pos_factors_is_72 (n : ℕ) :
  least_positive_integer_with_12_factors n → n = 72 := by
  sorry

end least_pos_int_with_12_pos_factors_is_72_l505_505236


namespace rachel_picked_4_apples_l505_505078

theorem rachel_picked_4_apples (initial_apples : ℕ) (remaining_apples : ℕ) (initial_apples_eq : initial_apples = 7) (remaining_apples_eq : remaining_apples = 3) :
  initial_apples - remaining_apples = 4 :=
by
  rw [initial_apples_eq, remaining_apples_eq]
  exact rfl

end rachel_picked_4_apples_l505_505078


namespace total_fruits_sum_l505_505851

-- Definitions based on the conditions
def bonnies_third_dog : ℕ := 60
def blueberries_second_dog : ℕ := (3/4 : ℚ) * bonnies_third_dog
def apples_first_dog : ℕ := 3 * blueberries_second_dog

-- Total fruits calculation
def total_fruits : ℕ := apples_first_dog + blueberries_second_dog + bonnies_third_dog

-- The statement to be proved
theorem total_fruits_sum :
  let bonnies := bonnies_third_dog,
      blueberries := blueberries_second_dog,
      apples := apples_first_dog
  in apples + blueberries + bonnies = 240 :=
by
  sorry

end total_fruits_sum_l505_505851


namespace ratio_of_water_to_milk_l505_505012

theorem ratio_of_water_to_milk (V : ℝ) (hV : V > 0) :
  let milk1 := (3 / 5) * V;
      milk2 := (4 / 5) * V;
      water1 := V - milk1;
      water2 := V - milk2;
      total_milk := milk1 + milk2;
      total_water := water1 + water2;
  total_water / total_milk = 3 / 7 :=
by
  have milk1 := (3 / 5) * V;
  have milk2 := (4 / 5) * V;
  have water1 := V - milk1;
  have water2 := V - milk2;
  have total_milk := milk1 + milk2;
  have total_water := water1 + water2;
  calc
    total_water / total_milk
      = ((2 / 5) * V + (1 / 5) * V) / ((3 / 5) * V + (4 / 5) * V) : by sorry
  ... = (3 / 5 * V) / (7 / 5 * V) : by sorry
  ... = (3 / 5) * (5 / 7) : by sorry
  ... = 3 / 7 : by sorry

end ratio_of_water_to_milk_l505_505012


namespace min_different_weights_l505_505122

theorem min_different_weights : ∃ (weights : Finset ℕ), (∀ n ∈ (Finset.range 20).map (λ n, n+1), ∃ a b ∈ weights, a + b = n) ∧ weights.card = 6 :=
by
  sorry

end min_different_weights_l505_505122


namespace least_integer_with_twelve_factors_l505_505194

open Classical

theorem least_integer_with_twelve_factors : ∃ k : ℕ, (∀ n : ℕ, (count_factors n = 12 → k ≤ n)) ∧ k = 72 :=
by
  sorry

-- Here count_factors is a function that returns the number of positive divisors of a given integer.

end least_integer_with_twelve_factors_l505_505194


namespace line_cannot_pass_through_third_quadrant_l505_505105

theorem line_cannot_pass_through_third_quadrant :
  ∀ (x y : ℝ), x + y - 1 = 0 → ¬(x < 0 ∧ y < 0) :=
by
  sorry

end line_cannot_pass_through_third_quadrant_l505_505105


namespace maggie_kept_bouncy_balls_l505_505056

def packs_bought_yellow : ℝ := 8.0
def packs_given_away_green : ℝ := 4.0
def packs_bought_green : ℝ := 4.0
def balls_per_pack : ℝ := 10.0

theorem maggie_kept_bouncy_balls :
  packs_bought_yellow * balls_per_pack + (packs_bought_green - packs_given_away_green) * balls_per_pack = 80.0 :=
by sorry

end maggie_kept_bouncy_balls_l505_505056


namespace man_born_in_1892_l505_505443

-- Define the conditions and question
def man_birth_year (x : ℕ) : ℕ :=
x^2 - x

-- Conditions:
variable (x : ℕ)
-- 1. The man was born in the first half of the 20th century
variable (h1 : man_birth_year x < 1950)
-- 2. The man's age x and the conditions in the problem
variable (h2 : x^2 - x < 1950)

-- The statement we aim to prove
theorem man_born_in_1892 (x : ℕ) (h1 : man_birth_year x < 1950) (h2 : x = 44) : man_birth_year x = 1892 := by
  sorry

end man_born_in_1892_l505_505443


namespace find_x_values_l505_505974

theorem find_x_values (x : ℝ) :
  (3 * x + 2 < (x - 1) ^ 2 ∧ (x - 1) ^ 2 < 9 * x + 1) ↔
  (x > (5 + Real.sqrt 29) / 2 ∧ x < 11) := 
by
  sorry

end find_x_values_l505_505974


namespace sum_of_sines_greater_than_third_angle_l505_505081

theorem sum_of_sines_greater_than_third_angle (α β γ : ℝ) (hα : 0 < α) (hβ : 0 < β) (hγ : 0 < γ)
(h_sum_angles : α + β + γ = π) : 
sin α + sin β > sin γ :=
sorry

end sum_of_sines_greater_than_third_angle_l505_505081


namespace count_four_digit_multiples_of_5_l505_505659

theorem count_four_digit_multiples_of_5 : 
  let first_4_digit := 1000
  let last_4_digit := 9999
  let first_multiple_of_5 := 1000
  let last_multiple_of_5 := 9995
  let total_multiples_of_5 := (1999 - 200 + 1)
  first_multiple_of_5 % 5 = 0 ∧ last_multiple_of_5 % 5 = 0 ∧ first_4_digit ≤ first_multiple_of_5 ∧ last_multiple_of_5 ≤ last_4_digit
  → total_multiples_of_5 = 1800 :=
by
  sorry

end count_four_digit_multiples_of_5_l505_505659


namespace shaded_fraction_equiv_l505_505452

noncomputable def geometric_series_sum (a r : ℝ) : ℝ :=
a / (1 - r)

theorem shaded_fraction_equiv :
  let initial_shaded := (4 : ℝ) * (1 / 16)
      r := 1 / 16
      total_shaded := geometric_series_sum initial_shaded r
  in total_shaded = 1 / 15 :=
by
  let initial_shaded := (4 : ℝ) * (1 / 16)
  let r := 1 / 16
  let total_shaded := geometric_series_sum initial_shaded r
  exact sorry -- Proof is left as an exercise.

end shaded_fraction_equiv_l505_505452


namespace least_positive_integer_with_12_factors_l505_505160

theorem least_positive_integer_with_12_factors : ∃ n : ℕ, (nat.factors_count n = 12 ∧ ∀ m : ℕ, (nat.factors_count m = 12) → n ≤ m) ∧ n = 72 :=
begin
  sorry
end

end least_positive_integer_with_12_factors_l505_505160


namespace least_positive_integer_with_12_factors_l505_505343

theorem least_positive_integer_with_12_factors : ∃ k : ℕ, k > 0 ∧ (∀ n : ℕ, n > 0 → (factors_count n = 12 → k ≤ n)) ∧ k = 72 :=
by {
  sorry,
}

end least_positive_integer_with_12_factors_l505_505343


namespace cost_of_each_shirt_l505_505775

theorem cost_of_each_shirt (initial_money : ℕ) (cost_pants : ℕ) (money_left : ℕ) (shirt_cost : ℕ)
  (h1 : initial_money = 109)
  (h2 : cost_pants = 13)
  (h3 : money_left = 74)
  (h4 : initial_money - (2 * shirt_cost + cost_pants) = money_left) :
  shirt_cost = 11 :=
by
  sorry

end cost_of_each_shirt_l505_505775


namespace least_integer_with_twelve_factors_l505_505187

open Classical

theorem least_integer_with_twelve_factors : ∃ k : ℕ, (∀ n : ℕ, (count_factors n = 12 → k ≤ n)) ∧ k = 72 :=
by
  sorry

-- Here count_factors is a function that returns the number of positive divisors of a given integer.

end least_integer_with_twelve_factors_l505_505187


namespace least_positive_integer_with_12_factors_l505_505285

def num_factors (n : ℕ) : ℕ := 
  (finset.range (n + 1)).filter (λ d, n % d = 0).card

theorem least_positive_integer_with_12_factors : 
  ∃ n : ℕ, num_factors n = 12 ∧ n = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_l505_505285


namespace least_positive_integer_with_12_factors_l505_505156

theorem least_positive_integer_with_12_factors : ∃ n : ℕ, (nat.factors_count n = 12 ∧ ∀ m : ℕ, (nat.factors_count m = 12) → n ≤ m) ∧ n = 72 :=
begin
  sorry
end

end least_positive_integer_with_12_factors_l505_505156


namespace none_satisfied_l505_505579

-- Define the conditions
variables {a b c x y z : ℝ}
  
-- Theorem that states that none of the given inequalities are satisfied strictly
theorem none_satisfied (h1 : x ≤ a) (h2 : y ≤ b) (h3 : z ≤ c) :
  ¬(x^2 * y + y^2 * z + z^2 * x < a^2 * b + b^2 * c + c^2 * a) ∧
  ¬(x^3 + y^3 + z^3 < a^3 + b^3 + c^3) :=
  by
    sorry

end none_satisfied_l505_505579


namespace least_positive_integer_with_12_factors_l505_505336

def has_exactly_12_factors (n : ℕ) : Prop :=
  n.factors.length = 12

theorem least_positive_integer_with_12_factors : ∃ k : ℕ, has_exactly_12_factors k ∧ (∀ n : ℕ, has_exactly_12_factors n → n ≥ k) ∧ k = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_l505_505336


namespace find_x_l505_505507

theorem find_x (x : ℤ) : 2^3 * 2^x = 16 → x = 1 := 
by sorry

end find_x_l505_505507


namespace least_positive_integer_with_12_factors_l505_505384

theorem least_positive_integer_with_12_factors : 
  ∃ (n : ℕ), (n > 0) ∧ (nat.factors_count n = 12) ∧ (∀ m, (m > 0) → (nat.factors_count m = 12) → n ≤ m) ∧ n = 108 :=
sorry

end least_positive_integer_with_12_factors_l505_505384


namespace least_positive_integer_with_12_factors_l505_505246

theorem least_positive_integer_with_12_factors :
  ∃ k : ℕ, (1 ≤ k) ∧ (nat.factors k).length = 12 ∧
    ∀ n : ℕ, (1 ≤ n) ∧ (nat.factors n).length = 12 → k ≤ n := 
begin
  sorry
end

end least_positive_integer_with_12_factors_l505_505246


namespace least_positive_integer_with_12_factors_l505_505342

theorem least_positive_integer_with_12_factors : ∃ k : ℕ, k > 0 ∧ (∀ n : ℕ, n > 0 → (factors_count n = 12 → k ≤ n)) ∧ k = 72 :=
by {
  sorry,
}

end least_positive_integer_with_12_factors_l505_505342


namespace least_integer_with_twelve_factors_l505_505182

open Classical

theorem least_integer_with_twelve_factors : ∃ k : ℕ, (∀ n : ℕ, (count_factors n = 12 → k ≤ n)) ∧ k = 72 :=
by
  sorry

-- Here count_factors is a function that returns the number of positive divisors of a given integer.

end least_integer_with_twelve_factors_l505_505182


namespace four_digit_multiples_of_five_count_l505_505668

theorem four_digit_multiples_of_five_count : 
  {n : ℕ | 1000 ≤ n ∧ n ≤ 9999 ∧ n % 5 = 0}.toFinset.card = 1800 :=
by
  sorry

end four_digit_multiples_of_five_count_l505_505668


namespace least_positive_integer_with_12_factors_l505_505348

theorem least_positive_integer_with_12_factors : ∃ k : ℕ, k > 0 ∧ (∀ n : ℕ, n > 0 → (factors_count n = 12 → k ≤ n)) ∧ k = 72 :=
by {
  sorry,
}

end least_positive_integer_with_12_factors_l505_505348


namespace least_pos_int_with_12_pos_factors_is_72_l505_505235

def least_positive_integer_with_12_factors (n : ℕ) : Prop :=
  ∃ (m : ℕ), m > 0 ∧ m ≠ n ∧ (∀ d : ℕ, d ∣ m → (d = n ∨ d = 1)) ∧
  ((∀ d : ℕ, d ∣ m → ∃ e : ℕ, e = 1) → n = 72)

theorem least_pos_int_with_12_pos_factors_is_72 (n : ℕ) :
  least_positive_integer_with_12_factors n → n = 72 := by
  sorry

end least_pos_int_with_12_pos_factors_is_72_l505_505235


namespace ff_zero_requiem_gg_zero_requiem_l505_505898

-- Definition of zero-requiem function
def zero_requiem (ψ : ℤ → ℤ) : Prop :=
  ∀ n : ℕ, n > 0 → ∀ (a : Fin n → ℤ), 
  let s := (Finset.univ : Finset (Fin n)).sum (λ i, a i) in
  s ≠ 0 ∨ (Finset.univ : Finset (Fin n)).sum (λ i, ψ (a i)) ≠ 0

-- Given functions f and g
variables (f g : ℤ → ℤ)

-- Conditions of the problem
axiom f_zero_requiem : zero_requiem f
axiom g_zero_requiem : zero_requiem g
axiom fg_id : ∀ x : ℤ, f (g x) = x
axiom gf_id : ∀ x : ℤ, g (f x) = x
axiom fg_not_zero_requiem : ¬ zero_requiem (λ x, f x + g x)

-- Theorem to be proved
theorem ff_zero_requiem : zero_requiem (λ x, f (f x)) :=
sorry

theorem gg_zero_requiem : zero_requiem (λ x, g (g x)) :=
sorry

end ff_zero_requiem_gg_zero_requiem_l505_505898


namespace ratio_of_x_intercepts_l505_505140

theorem ratio_of_x_intercepts (r q c : ℝ) (hc : c ≠ 0)
  (h1 : ∀ x y, y = 8 * x + 2 * c → (r, 0) = (x, y))
  (h2 : ∀ x y, y = 4 * x + c → (q, 0) = (x, y)) :
  r / q = 1 :=
by
  have hr : r = -c / 4, from (h1 r 0 (by simp [r]= -c / 4)),
  have hq : q = -c / 4, from (h2 q 0 (by simp [q]=-c / 4)),
  sorry

end ratio_of_x_intercepts_l505_505140


namespace find_x2_y2_z2_l505_505107

def matrix_M (x y z : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![0, 3 * y, -z],
    ![-x, 2 * y, z],
    ![2 * x, -y, 2 * z]
  ]

def matrix_I : Matrix (Fin 3) (Fin 3) ℝ :=
  Matrix.fromFunction (λ i j => if i = j then 1 else 0)

theorem find_x2_y2_z2 (x y z : ℝ)
    (h : (matrix_M x y z)ᵀ.mul (matrix_M x y z) = matrix_I) :
  x^2 + y^2 + z^2 = 46 / 105 := by
  sorry

end find_x2_y2_z2_l505_505107


namespace towels_folded_in_one_hour_l505_505017

theorem towels_folded_in_one_hour 
    (jane_rate : ℕ)
    (kyla_rate : ℕ)
    (anthony_rate : ℕ) 
    (h1 : jane_rate = 36)
    (h2 : kyla_rate = 30)
    (h3 : anthony_rate = 21) 
    : jane_rate + kyla_rate + anthony_rate = 87 := 
by
  simp [h1, h2, h3]
  sorry

end towels_folded_in_one_hour_l505_505017


namespace total_distance_l505_505959

theorem total_distance (D : ℝ) 
  (h₁ : 60 * (D / 2 / 60) = D / 2) 
  (h₂ : 40 * ((D / 2) / 4 / 40) = D / 8) 
  (h₃ : 50 * (105 / 50) = 105)
  (h₄ : D = D / 2 + D / 8 + 105) : 
  D = 280 :=
by sorry

end total_distance_l505_505959


namespace playground_ball_cost_l505_505492

-- Define the given conditions
def cost_jump_rope : ℕ := 7
def cost_board_game : ℕ := 12
def saved_by_dalton : ℕ := 6
def given_by_uncle : ℕ := 13
def additional_needed : ℕ := 4

-- Total money Dalton has
def total_money : ℕ := saved_by_dalton + given_by_uncle

-- Total cost needed to buy all three items
def total_cost_needed : ℕ := total_money + additional_needed

-- Combined cost of the jump rope and the board game
def combined_cost : ℕ := cost_jump_rope + cost_board_game

-- Prove the cost of the playground ball
theorem playground_ball_cost : ℕ := total_cost_needed - combined_cost

-- Expected result
example : playground_ball_cost = 4 := by
  sorry

end playground_ball_cost_l505_505492


namespace multiples_of_5_in_4_digit_range_l505_505656

theorem multiples_of_5_in_4_digit_range : 
  let count_multiples := (9995 - 1000) / 5 + 1
  in count_multiples = 1800 :=
by
  sorry

end multiples_of_5_in_4_digit_range_l505_505656


namespace set_difference_equality_l505_505764

theorem set_difference_equality :
  let A := {x : ℝ | x < 4}
  let B := {x : ℝ | x^2 - 4 * x + 3 > 0}
  let S := {x : ℝ | x ∈ A ∧ x ∉ (set.Inter {x ∈ A | x ∈ B})}
  S = {x : ℝ | 1 ≤ x ∧ x ≤ 3} :=
by {
  let A := {x : ℝ | x < 4},
  let B := {x : ℝ | x^2 - 4 * x + 3 > 0},
  let S := {x : ℝ | x ∈ A ∧ x ∉ (A ∩ B)},
  sorry
}

end set_difference_equality_l505_505764


namespace bottles_purchased_l505_505086

/-- Given P bottles can be bought for R dollars, determine how many bottles can be bought for M euros
    if 1 euro is worth 1.2 dollars and there is a 10% discount when buying with euros. -/
theorem bottles_purchased (P R M : ℝ) (hR : R > 0) (hP : P > 0) :
  let euro_to_dollars := 1.2
  let discount := 0.9
  let dollars := euro_to_dollars * M * discount
  (P / R) * dollars = (1.32 * P * M) / R :=
by
  sorry

end bottles_purchased_l505_505086


namespace initial_men_l505_505795

variable (P M : ℕ) -- P represents the provisions and M represents the initial number of men.

-- Conditons
def provision_lasts_20_days : Prop := P / (M * 20) = P / ((M + 200) * 15)

-- The proof problem
theorem initial_men (h : provision_lasts_20_days P M) : M = 600 :=
sorry

end initial_men_l505_505795


namespace count_four_digit_multiples_of_5_l505_505661

theorem count_four_digit_multiples_of_5 : 
  let first_4_digit := 1000
  let last_4_digit := 9999
  let first_multiple_of_5 := 1000
  let last_multiple_of_5 := 9995
  let total_multiples_of_5 := (1999 - 200 + 1)
  first_multiple_of_5 % 5 = 0 ∧ last_multiple_of_5 % 5 = 0 ∧ first_4_digit ≤ first_multiple_of_5 ∧ last_multiple_of_5 ≤ last_4_digit
  → total_multiples_of_5 = 1800 :=
by
  sorry

end count_four_digit_multiples_of_5_l505_505661


namespace ff_even_of_f_even_l505_505040

-- Define what it means for a function f to be even.
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f (x)

-- Theorem to prove that f(f(x)) is even if f is even.
theorem ff_even_of_f_even (f : ℝ → ℝ) (hf : is_even_function f) : is_even_function (f ∘ f) :=
by
  intros x,
  specialize hf x,
  specialize hf (-x),
  rw [←hf, hf],
  sorry

end ff_even_of_f_even_l505_505040


namespace longest_segment_CD_l505_505004

theorem longest_segment_CD :
  ∀ (A B C D : Type)
    (angle_ABD : ℝ) (angle_ADB : ℝ) (angle_CBD : ℝ) (angle_BDC : ℝ)
    (AB AD BD BC CD : ℝ),
  angle_ABD = 30 ∧ angle_ADB = 65 ∧ angle_CBD = 85 ∧ angle_BDC = 65 →
  ∠BAD = 180 - angle_ABD - angle_ADB ∧
  ∠BCD = 180 - angle_CBD - angle_BDC →
  ∠ABD < ∠ADB ∧ ∠ADB < ∠BAD ∧ ∠BCD < ∠BDC ∧ ∠BDC < ∠CBD →
  AD < AB ∧ AB < BD ∧ BD < BC ∧ BC < CD →
  CD > AB ∧ CD > AD ∧ CD > BD ∧ CD > BC :=
  by
    intros A B C D angle_ABD angle_ADB angle_CBD angle_BDC AB AD BD BC CD conditions angles_in_triangles angle_inequalities side_inequalities
    sorry

end longest_segment_CD_l505_505004


namespace sufficient_but_not_necessary_l505_505536

theorem sufficient_but_not_necessary (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ((1 / 2)^a < (1 / 2)^b → log (a + 1) > log b) ∧ ¬(log (a + 1) > log b → (1 / 2)^a < (1 / 2)^b) :=
by
  sorry

end sufficient_but_not_necessary_l505_505536


namespace least_positive_integer_with_12_factors_l505_505244

theorem least_positive_integer_with_12_factors :
  ∃ k : ℕ, (1 ≤ k) ∧ (nat.factors k).length = 12 ∧
    ∀ n : ℕ, (1 ≤ n) ∧ (nat.factors n).length = 12 → k ≤ n := 
begin
  sorry
end

end least_positive_integer_with_12_factors_l505_505244


namespace find_a_l505_505002

open Real

-- Define conditions in Lean
def parametric_line (t : ℝ) : ℝ × ℝ :=
  ( - 3 / 5 * t, 4 / 5 * t )

def polar_circle (θ : ℝ) (a : ℝ) : ℝ :=
  a * sin θ

-- Lean statement of the proof problem
theorem find_a (a : ℝ) (t θ : ℝ) (c : ℝ) :
  (a ≠ 0) →
  -- Parametric form of the line
  (parametric_line t = (-3 / 5 * t, 4 / 5 * t)) →
  -- Polar form of the circle
  (polar_circle θ a = a * sin θ) →
  -- Chord cut by the line is sqrt(3) times the radius
  let line_eq := 4 * (-3 / 5 * t) + 3 * (4 / 5 * t) - 8 = 0,
      circle_eq := (λ (x y : ℝ), x^2 + y^2 - a * y = 0),
      center := (0, a / 2),
      radius := a / 2,
      d := abs (3 / 2 * a - 8) / 5,
      chord_length := 2 * sqrt (radius^2 - d^2)
  in (sqrt 3 * radius = chord_length) →
  (a = 32 ∨ a = 32 / 11) :=
by
  intro ha_param hline_eq hp_circle_eq hchord_length_eq
  -- Omitting the proofs
  sorry

end find_a_l505_505002


namespace Justine_colored_sheets_l505_505855

theorem Justine_colored_sheets :
  (∃ sheets total_binders sheets_per_binder :
    nat, total_binders > 0 ∧ total_binders = 5 ∧ sheets = 2450 ∧ sheets_per_binder = sheets / total_binders 
    ∧ Justine_colored = sheets_per_binder / 2) → Justine_colored = 245 :=
by 
  intro h,
  rcases h with ⟨sheets, total_binders, sheets_per_binder, h1, h2, h3, h4, h5⟩,
  have sheets_per_binder_calc : sheets_per_binder = 490 := nat.div_eq_of_eq_mul_left h1 h2.symm h3,
  have Justine_colored_calc : Justine_colored = 245 := nat.div_eq_of_eq_mul_left (nat.zero_lt_of_lt h1 (nat.succ_pos 4)) h4.symm h5,
  exact Justine_colored_calc,
  simp [Justine_colored_calc],
  sorry

end Justine_colored_sheets_l505_505855


namespace second_month_interest_l505_505022

def compounded_interest (initial_loan : ℝ) (rate_per_month : ℝ) : ℝ :=
  initial_loan * rate_per_month

theorem second_month_interest :
  let initial_loan := 200
  let rate_per_month := 0.10
  compounded_interest (initial_loan + compounded_interest initial_loan rate_per_month) rate_per_month = 22 :=
by
  sorry

end second_month_interest_l505_505022


namespace artemon_distance_covered_l505_505810

-- Define the condition of the rectangle
def rectangle_side1 : ℝ := 6
def rectangle_side2 : ℝ := 2.5

-- Define the speeds of Malvina, Buratino, and Artemon
def speed_malvina : ℝ := 4
def speed_buratino : ℝ := 6
def speed_artemon : ℝ := 12

-- Using the given conditions, we need to show that Artemon runs 7.8 km before Malvina and Buratino meet
theorem artemon_distance_covered :
  let d := real.sqrt (rectangle_side1^2 + rectangle_side2^2) in
  let t := d / (speed_malvina + speed_buratino) in
  speed_artemon * t = 7.8 :=
by
  let d := real.sqrt (rectangle_side1^2 + rectangle_side2^2)
  let t := d / (speed_malvina + speed_buratino)
  have h: speed_artemon * t = 7.8 := sorry
  exact h

end artemon_distance_covered_l505_505810


namespace least_positive_integer_with_12_factors_is_96_l505_505394

def has_exactly_12_factors (n : ℕ) : Prop :=
  (12 : ℕ) = (List.range (n + 1)).filter (λ x, n % x = 0).length

theorem least_positive_integer_with_12_factors_is_96 :
  n ∈ (ℕ) ∧ has_exactly_12_factors n → n = 96 :=
by
  sorry

end least_positive_integer_with_12_factors_is_96_l505_505394


namespace least_positive_integer_with_12_factors_is_96_l505_505392

def has_exactly_12_factors (n : ℕ) : Prop :=
  (12 : ℕ) = (List.range (n + 1)).filter (λ x, n % x = 0).length

theorem least_positive_integer_with_12_factors_is_96 :
  n ∈ (ℕ) ∧ has_exactly_12_factors n → n = 96 :=
by
  sorry

end least_positive_integer_with_12_factors_is_96_l505_505392


namespace least_positive_integer_with_12_factors_l505_505254

theorem least_positive_integer_with_12_factors :
  ∃ k : ℕ, (1 ≤ k) ∧ (nat.factors k).length = 12 ∧
    ∀ n : ℕ, (1 ≤ n) ∧ (nat.factors n).length = 12 → k ≤ n := 
begin
  sorry
end

end least_positive_integer_with_12_factors_l505_505254


namespace distance_from_A_to_B_l505_505840

-- Define the conditions
def smaller_square_perimeter := 8 -- cm
def larger_square_area := 49 -- cm²

-- Define side lengths based on the conditions
def smaller_square_side := smaller_square_perimeter / 4
def larger_square_side := Real.sqrt larger_square_area

-- Define the horizontal and vertical segments of the right triangle
def horizontal_segment := smaller_square_side + larger_square_side
def vertical_segment := larger_square_side - smaller_square_side

-- Define the distance AB using the Pythagorean theorem
def distance_AB := Real.sqrt (horizontal_segment ^ 2 + vertical_segment ^ 2)

-- Theorem to prove the distance AB is approximately 10.3 cm
theorem distance_from_A_to_B : distance_AB ≈ 10.3 := by
  sorry

end distance_from_A_to_B_l505_505840


namespace find_PR_l505_505036

-- Defining the problem with given conditions
variables (P Q R A B : Type) [point : EUCL (Type)] 
variables [triangle : TRIANGLE P Q R] [right_angle : ANGLE B P Q 90] 
variables (QB AR PR : ℝ)
variables (midpointA : MIDPOINT A P Q) 
variables (midpointB : MIDPOINT B P R)

-- Given conditions
def condition1 : QB = 25 := sorry
def condition2 : AR = 15 := sorry

theorem find_PR : PR = (sqrt 170) := 
by 
  sorry

end find_PR_l505_505036


namespace right_triangle_ratio_l505_505902

theorem right_triangle_ratio (ABC : Triangle) (AC BC AB : ℝ) (K N M : Point) 
  (h_parallel : Parallel K AC) (h_intersect_BC : K ∈ Segment BC) (h_intersect_AB : N ∈ Segment AB)
  (h_isosceles : distance M K = distance M N) 
  (h_ratio : distance BK / distance BC = 1/14) :
  distance AM / distance MC = 27 := by
  sorry

end right_triangle_ratio_l505_505902


namespace part_a_part_b_impossible_l505_505071

noncomputable def phi (x y : ℝ) := x * y + x + y + 1
noncomputable def theta (x y : ℝ) := x * y + x + y

-- Define sequence of polynomials for phi
noncomputable def sequence_phi (n : ℕ) : ℝ → ℝ
| 0       => 1
| 1       => id
| (n + 1) => λ x, phi (sequence_phi n x) x

-- Definition of the 1982 polynomial
noncomputable def target_poly (x : ℝ) : ℝ := ∑ i in finset.range 1983, x^i

-- Main statements
theorem part_a : sequence_phi 1982 = target_poly := sorry

theorem part_b_impossible :
  ¬ (∃ (sequence_theta : ℕ → (ℝ → ℝ)), sequence_theta 0 = 1 ∧
    sequence_theta 1 = id ∧
    (∀ n, sequence_theta (n + 1) = λ x, theta (sequence_theta n x) x) ∧
    (sequence_theta 1982 = target_poly)) := sorry

end part_a_part_b_impossible_l505_505071


namespace percent_employed_females_l505_505883

variables (P : Type) (population : P → ℝ) (employed : P → Prop)
variables [fintype P]

def percent_employed (p : P) : ℝ := 0.60
def percent_employed_males (p : P) : ℝ := 0.45

theorem percent_employed_females (p : P) :
  (0.15 / percent_employed p) * 100 = 25 :=
sorry

end percent_employed_females_l505_505883


namespace least_positive_integer_with_12_factors_is_96_l505_505388

def has_exactly_12_factors (n : ℕ) : Prop :=
  (12 : ℕ) = (List.range (n + 1)).filter (λ x, n % x = 0).length

theorem least_positive_integer_with_12_factors_is_96 :
  n ∈ (ℕ) ∧ has_exactly_12_factors n → n = 96 :=
by
  sorry

end least_positive_integer_with_12_factors_is_96_l505_505388


namespace cricket_team_players_l505_505126

theorem cricket_team_players (P N : ℕ) (h1 : 37 = 37) 
  (h2 : (57 - 37) = 20) 
  (h3 : ∀ N, (2 / 3 : ℚ) * N = 20 → N = 30) 
  (h4 : P = 37 + 30) : P = 67 := 
by
  -- Proof steps will go here
  sorry

end cricket_team_players_l505_505126


namespace least_positive_integer_with_12_factors_is_972_l505_505370

theorem least_positive_integer_with_12_factors_is_972 : ∃ k : ℕ, (∀ n : ℕ, (∃ (d : ℕ), d * k = n) ↔ n = 12) ∧ k = 972 := sorry

end least_positive_integer_with_12_factors_is_972_l505_505370


namespace column_proportion_l505_505431

def table : Type := 
  {M : matrix (fin 100) (fin 100) ℕ // ∀ i : fin 100, 
    (M i).to_list.sort = list.range' 1 (i + 1 : ℕ) ++ (list.repeat 0 (100 - (i + 1)))}

theorem column_proportion (M : table) :
  ∃ (c1 c2 : fin 100), ∑ i, M.val i c2 ≥ 19 * ∑ i, M.val i c1 :=
by
  sorry

end column_proportion_l505_505431


namespace max_value_f_l505_505108

def f (x : ℝ) : ℝ := 2 * Real.sin x

theorem max_value_f : ∃ x : ℝ,  f(x) = 2 := by
  sorry

end max_value_f_l505_505108


namespace sum_of_non_domain_elements_l505_505488

def g (x : ℝ) : ℝ := 1 / (2 + 1 / (1 + 1 / x^2))

theorem sum_of_non_domain_elements : (∑ x in {0}.to_finset, x) = 0 :=
by
  sorry

end sum_of_non_domain_elements_l505_505488


namespace multiples_of_5_in_4_digit_range_l505_505651

theorem multiples_of_5_in_4_digit_range : 
  let count_multiples := (9995 - 1000) / 5 + 1
  in count_multiples = 1800 :=
by
  sorry

end multiples_of_5_in_4_digit_range_l505_505651


namespace least_positive_integer_with_12_factors_is_972_l505_505356

theorem least_positive_integer_with_12_factors_is_972 : ∃ k : ℕ, (∀ n : ℕ, (∃ (d : ℕ), d * k = n) ↔ n = 12) ∧ k = 972 := sorry

end least_positive_integer_with_12_factors_is_972_l505_505356


namespace sphere_partition_regions_l505_505947

theorem sphere_partition_regions (n : ℕ) (h : n = 6) :
  let regions := (3 * n - 3) * (2 * 3^n - 1)
  regions = 128 :=
by
  have n_eq_6 : n = 6 := h
  have regions_calc : regions = 128 := sorry
  exact regions_calc

end sphere_partition_regions_l505_505947


namespace probability_all_quit_same_tribe_l505_505114

-- Define the number of participants and the number of tribes
def numParticipants : ℕ := 18
def numTribes : ℕ := 2
def tribeSize : ℕ := 9 -- Each tribe has 9 members

-- Define the problem statement
theorem probability_all_quit_same_tribe : 
  (numParticipants.choose 3) = 816 ∧
  ((tribeSize.choose 3) * numTribes) = 168 ∧
  ((tribeSize.choose 3) * numTribes) / (numParticipants.choose 3) = 7 / 34 :=
by
  sorry

end probability_all_quit_same_tribe_l505_505114


namespace least_positive_integer_with_12_factors_l505_505340

theorem least_positive_integer_with_12_factors : ∃ k : ℕ, k > 0 ∧ (∀ n : ℕ, n > 0 → (factors_count n = 12 → k ≤ n)) ∧ k = 72 :=
by {
  sorry,
}

end least_positive_integer_with_12_factors_l505_505340


namespace least_positive_integer_with_12_factors_is_72_l505_505309

-- Definition of having exactly 12 factors for a positive integer
def has_exactly_12_factors (n : ℕ) : Prop := n > 0 ∧ (finset.filter (λ d, n % d = 0) (finset.range (n + 1))).card = 12

-- Statement of the problem in Lean 4
theorem least_positive_integer_with_12_factors_is_72 :
  ∃ n : ℕ, has_exactly_12_factors n ∧ ∀ m : ℕ, has_exactly_12_factors m → n ≤ m :=
begin
  use 72,
  split,
  { -- Prove that 72 has exactly 12 factors
    sorry },
  { -- Prove that 72 is the least such integer
    sorry }
end

end least_positive_integer_with_12_factors_is_72_l505_505309


namespace relationship_between_a_b_m_n_l505_505589

noncomputable def g (x : ℝ) (m : ℝ) : ℝ := x^2 - m * x

noncomputable def f (x : ℝ) (n : ℝ) (g : ℝ → ℝ) : ℝ := (x - n) * g x

theorem relationship_between_a_b_m_n (m n a b : ℝ) 
  (h1 : m > 0)
  (h2 : n > 0)
  (h3 : m > n)
  (h4 : b < a)
  (h5 : g 0 m = 0)
  (h6 : g m m = 0)
  (h7 : g (m + 1) m = m + 1)
  (h8 : ∀ x, f x n (g x m) = x^3 - (m + n) * x^2 + m * n * x)
  (h9 : f' a n (g x m) = 0)
  (h10 : f' b n (g x m) = 0) : b < n ∧ n < a ∧ a < m :=
sorry

end relationship_between_a_b_m_n_l505_505589


namespace false_statement_divisibility_l505_505945

-- Definitions for the divisibility conditions
def divisible_by (a b : ℕ) : Prop := ∃ k, b = a * k

-- The problem statement
theorem false_statement_divisibility (N : ℕ) :
  (divisible_by 2 N ∧ divisible_by 4 N ∧ divisible_by 12 N ∧ ¬ divisible_by 24 N) →
  (¬ divisible_by 24 N) :=
by
  -- The proof will need to be filled in here
  sorry

end false_statement_divisibility_l505_505945


namespace bruce_will_be_3_times_as_old_in_6_years_l505_505930

variables (x : ℕ)

-- Definitions from conditions
def bruce_age_now := 36
def son_age_now := 8

-- Equivalent Lean 4 statement
theorem bruce_will_be_3_times_as_old_in_6_years :
  (bruce_age_now + x = 3 * (son_age_now + x)) → x = 6 :=
sorry

end bruce_will_be_3_times_as_old_in_6_years_l505_505930


namespace least_integer_with_twelve_factors_l505_505195

open Classical

theorem least_integer_with_twelve_factors : ∃ k : ℕ, (∀ n : ℕ, (count_factors n = 12 → k ≤ n)) ∧ k = 72 :=
by
  sorry

-- Here count_factors is a function that returns the number of positive divisors of a given integer.

end least_integer_with_twelve_factors_l505_505195


namespace least_positive_integer_with_12_factors_l505_505210

/--
Prove that the smallest positive integer with exactly 12 positive factors is 60.
-/
theorem least_positive_integer_with_12_factors : 
  ∃ (n : ℕ), (0 < n) ∧ (nat.factors_count n = 12) ∧ (∀ m : ℕ, (0 < m) ∧ (nat.factors_count m = 12) → n ≤ m) → n = 60 :=
sorry

end least_positive_integer_with_12_factors_l505_505210


namespace no_subset_with_total_age_at_least_225_l505_505847

variables (students : ℕ) (total_age : ℝ)
variable (subset_age : ℝ)

-- Given conditions
def class_total_students := 35
def class_total_age := 280

-- Prove that there does not exist a subset of 25 students whose total age is at least 225 years
theorem no_subset_with_total_age_at_least_225:
  students = class_total_students →
  total_age = class_total_age →
  ∀ (subset : finset ℕ), (subset.card = 25 → ∑ i in subset, subset_age < 225) :=
by
  sorry

end no_subset_with_total_age_at_least_225_l505_505847


namespace rectangular_base_dimensions_l505_505957

theorem rectangular_base_dimensions :
  ∀ (total_volume parts height : ℝ) (ratio : ℝ), 
  total_volume = 120 ∧ parts = 10 ∧ height = 1 ∧ ratio = 3 / 4 →
  ∃ (a b : ℝ), 
  a = 4 ∧ b = 3 ∧ 
  (total_volume / parts / height = a * b) ∧ 
  (b = (ratio) * a) :=
by
  intro total_volume parts height ratio
  intro h
  cases h with h_volume rest,
  cases rest with h_parts rest,
  cases rest with h_height h_ratio,
  have volume_one_part := h_volume / h_parts,
  have base_area := volume_one_part / h_height,
  have ratio_squared := ratio * ratio,
  have x_squared := base_area / ratio_squared,
  have x := real.sqrt x_squared,
  have a := x,
  have b := ratio * x,
  use [a, b],
  sorry

end rectangular_base_dimensions_l505_505957


namespace decreasing_interval_of_log_composed_quadratic_l505_505546

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log a (x^2 - 4 * x + 3)

theorem decreasing_interval_of_log_composed_quadratic {a : ℝ} (h_a : 0 < a ∧ a < 1) :
  (∀ x y : ℝ, 3 < x ∧ x < y → f a y < f a x) :=
by
  sorry

end decreasing_interval_of_log_composed_quadratic_l505_505546


namespace nm_value_l505_505675

theorem nm_value (n m : ℤ) (h1 : n + 8 = 6) (h2 : 2 * m = 6) : n^m = -8 :=
by
  sorry

end nm_value_l505_505675


namespace least_integer_with_twelve_factors_l505_505185

open Classical

theorem least_integer_with_twelve_factors : ∃ k : ℕ, (∀ n : ℕ, (count_factors n = 12 → k ≤ n)) ∧ k = 72 :=
by
  sorry

-- Here count_factors is a function that returns the number of positive divisors of a given integer.

end least_integer_with_twelve_factors_l505_505185


namespace geometric_sequence_q_S_4_l505_505555

theorem geometric_sequence_q_S_4 :
  ∃ {a : ℕ → ℝ} {q : ℝ}, 
  (∀ n, a (n+1) = a n * q) ∧ 
  (a 2 * a 4 = a 5) ∧ 
  (a 4 = 8) ∧ 
  (q = 2) ∧ 
  (a 0 = 1) ∧ 
  (a 0 + a 1 + a 2 + a 3 = 15) :=
sorry

end geometric_sequence_q_S_4_l505_505555


namespace least_positive_integer_with_12_factors_is_96_l505_505260

theorem least_positive_integer_with_12_factors_is_96 :
  ∃ n : ℕ, (n = 96 ∧ (∀ m : ℕ, (m > 0 ∧ ∀ d : ℕ, d ∣ m → d > 0 → d < m → (∃ k : ℕ, d = k ∨ k * d = m) → k > 1 → m > n))) ∧ (∃ m : ℕ, m = 12 ∧ (∀ d : ℕ, d ∣ m → d ∈ {1, m} ∨ ∀ k : ℕ, k * d = m → d = k ∨ d < k)).
sorry

end least_positive_integer_with_12_factors_is_96_l505_505260


namespace interest_second_month_l505_505023

theorem interest_second_month {P r n : ℝ} (hP : P = 200) (hr : r = 0.10) (hn : n = 12) :
  (P * (1 + r / n) ^ (n * (1/12)) - P) * r / n = 1.68 :=
by
  sorry

end interest_second_month_l505_505023


namespace least_positive_integer_with_12_factors_is_72_l505_505222

open Nat

def hasExactly12Factors (n : ℕ) : Prop :=
  (divisors n).length = 12

theorem least_positive_integer_with_12_factors_is_72 :
  ∃ n : ℕ, hasExactly12Factors n ∧ ∀ m : ℕ, hasExactly12Factors m → n ≤ m → n = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_is_72_l505_505222


namespace least_positive_integer_with_12_factors_l505_505296

def has_exactly_12_factors (n : ℕ) : Prop :=
  (finset.range(n + 1).filter (λ d, n % d = 0)).card = 12

theorem least_positive_integer_with_12_factors : ∀ n : ℕ, has_exactly_12_factors n → n ≥ 150 :=
sorry

end least_positive_integer_with_12_factors_l505_505296


namespace area_at_stage_8_l505_505682

-- Defining the constants and initial settings
def first_term : ℕ := 1
def common_difference : ℕ := 1
def stage : ℕ := 8
def square_side_length : ℕ := 4

-- Calculating the number of squares at the given stage
def num_squares : ℕ := first_term + (stage - 1) * common_difference

--Calculating the area of one square
def area_one_square : ℕ := square_side_length * square_side_length

-- Calculating the total area at the given stage
def total_area : ℕ := num_squares * area_one_square

-- Proving the total area equals 128 at Stage 8
theorem area_at_stage_8 : total_area = 128 := 
by
  sorry

end area_at_stage_8_l505_505682


namespace scientific_notation_l505_505929

-- Definition for the diameter of pollen in millimeters
def pollen_diameter : ℝ := 0.0000021

-- Statement that we need to prove
theorem scientific_notation (d : ℝ) (h : d = pollen_diameter) : d = 2.1 * 10^(-6) := 
by {
  -- prove here
  sorry
}

end scientific_notation_l505_505929


namespace min_d_value_l505_505518

theorem min_d_value (x : Fin 51 → ℝ) (h_sum : (∑ i, x i) = 100) :
  (∑ i, (x i)^2) ≥ 50 * (100 / 51)^2 :=
by
  sorry

end min_d_value_l505_505518


namespace passes_through_fixed_point_l505_505101

theorem passes_through_fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : 
  ∃ x y : ℝ, (x = 2 ∧ y = 3) ∧ y = a^(x - 2) + 2 :=
by 
  use 2
  use 3
  split
  { exact ⟨rfl, rfl⟩ }
  { sorry }

end passes_through_fixed_point_l505_505101


namespace area_of_triangle_is_correct_l505_505766

noncomputable def area_of_triangle {P Q R : ℝ × ℝ} (pqr_right_angle_at_R P_median Q_median hypotenuse_length) : ℝ :=
  if pqr_right_angle_at_R
     ∧ (∃ (f g : ℝ → ℝ), f = λ x, x + 5 ∧ g = λ x, 2x + 5
                ∧ median P f ∧ median Q g)
     ∧ hypotenuse_length = 50 then
    250
  else 0

theorem area_of_triangle_is_correct : 
  ∀ {P Q R : ℝ × ℝ} (pqr_right_triangle_at_R : is_right_triangle P Q R)
  (median_through_P : ∃ (f : ℝ → ℝ), f = λ x, x + 5 ∧ median P f)
  (median_through_Q : ∃ (g : ℝ → ℝ), g = λ x, 2x + 5 ∧ median Q g)
  (hypotenuse_length : dist P Q = 50), 
  area_of_triangle pqr_right_triangle_at_R median_through_P median_through_Q hypotenuse_length = 250 :=
sorry

end area_of_triangle_is_correct_l505_505766


namespace least_positive_integer_with_12_factors_l505_505202

/--
Prove that the smallest positive integer with exactly 12 positive factors is 60.
-/
theorem least_positive_integer_with_12_factors : 
  ∃ (n : ℕ), (0 < n) ∧ (nat.factors_count n = 12) ∧ (∀ m : ℕ, (0 < m) ∧ (nat.factors_count m = 12) → n ≤ m) → n = 60 :=
sorry

end least_positive_integer_with_12_factors_l505_505202


namespace square_area_l505_505867

theorem square_area (x y : ℝ)
    (h1 : x = 20 ∨ x = x)
    (h2 : y = 20 ∨ y = 5) :
    (let side_length : ℝ := abs (20 - 5) in side_length * side_length) = 225 := by
  -- side_length calculation
  let side_length : ℝ := abs (20 - 5)
  have h3 : side_length = 15 := by
    simp [side_length]
  -- Area calculation
  have area : ℝ := side_length * side_length
  have h4 : area = 225 := by
    simp [h3]
  exact h4

-- Proof is omitted here as instructed.

end square_area_l505_505867


namespace find_expression_for_3f_l505_505677

theorem find_expression_for_3f (f : ℝ → ℝ) (h : ∀ x > 0, f (3 * x) = 3 / (3 + 2 * x)) : 
  ∀ x > 0, 3 * f x = 27 / (9 + 2 * x) :=
by
  intro x hx
  have hx' : 3 * (x / 3) > 0 := by linarith
  rw [← (h (x / 3) hx')]
  sorry

end find_expression_for_3f_l505_505677


namespace base8_difference_divisible_by_7_l505_505096

theorem base8_difference_divisible_by_7 (A B : ℕ) (h₁ : A < 8) (h₂ : B < 8) (h₃ : A ≠ B) : 
  ∃ k : ℕ, k * 7 = (if 8 * A + B > 8 * B + A then 8 * A + B - (8 * B + A) else 8 * B + A - (8 * A + B)) :=
by
  sorry

end base8_difference_divisible_by_7_l505_505096


namespace interest_rate_is_10_percent_l505_505094

theorem interest_rate_is_10_percent (P : ℝ) (t : ℝ) (d : ℝ) (r : ℝ) 
  (hP : P = 9999.99999999988) 
  (ht : t = 1) 
  (hd : d = 25)
  : P * (1 + r / 2)^(2 * t) - P - (P * r * t) = d → r = 0.1 :=
by
  intros h
  rw [hP, ht, hd] at h
  sorry

end interest_rate_is_10_percent_l505_505094


namespace min_positive_period_of_f_l505_505109

noncomputable def f (x : ℝ) : ℝ := cos (2 * x) ^ 2 - sin (2 * x) ^ 2

theorem min_positive_period_of_f : ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = π / 2 :=
begin
  sorry
end

end min_positive_period_of_f_l505_505109


namespace decrease_in_ratio_of_royalties_l505_505909

noncomputable def calculate_decrease_in_ratio (Royalties₁ Sales₁ Royalties₂ Sales₂ : ℝ) : ℝ :=
  let ratio₁ := Royalties₁ / Sales₁
  let ratio₂ := Royalties₂ / Sales₂
  (ratio₁ - ratio₂) * 100

theorem decrease_in_ratio_of_royalties 
  (Royalties₁ Sales₁ : ℝ) (h₁ : Royalties₁ = 4) (h₂ : Sales₁ = 20)
  (Royalties₂ Sales₂ : ℝ) (h₃ : Royalties₂ = 9) (h₄ : Sales₂ = 108) : 
  calculate_decrease_in_ratio Royalties₁ Sales₁ Royalties₂ Sales₂ ≈ 11.67 :=
by
  sorry

end decrease_in_ratio_of_royalties_l505_505909


namespace solution_to_system_l505_505085

theorem solution_to_system : ∃ x y : ℤ, (2 * x + 3 * y = -11 ∧ 6 * x - 5 * y = 9) ↔ (x = -1 ∧ y = -3) :=
by
  sorry

end solution_to_system_l505_505085


namespace estevan_polka_dot_blankets_l505_505964

theorem estevan_polka_dot_blankets (total_blankets : ℕ) (polka_dot_fraction_numerator : ℕ) (polka_dot_fraction_denominator : ℕ) (additional_polka_dot_blankets : ℕ) 
(h_total : total_blankets = 156) 
(h_fraction_num : polka_dot_fraction_numerator = 3) 
(h_fraction_den : polka_dot_fraction_denominator = 7) 
(h_additional : additional_polka_dot_blankets = 9) 
: (polka_dot_fraction_numerator * (total_blankets / polka_dot_fraction_denominator)) + additional_polka_dot_blankets = 75 := 
by
  rw [h_total, h_fraction_num, h_fraction_den, h_additional]
  simp
  norm_num
  sorry

end estevan_polka_dot_blankets_l505_505964


namespace least_positive_integer_with_12_factors_l505_505328

def has_exactly_12_factors (n : ℕ) : Prop :=
  n.factors.length = 12

theorem least_positive_integer_with_12_factors : ∃ k : ℕ, has_exactly_12_factors k ∧ (∀ n : ℕ, has_exactly_12_factors n → n ≥ k) ∧ k = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_l505_505328


namespace density_carat_cubic_inch_l505_505093

noncomputable def density_emerald_g_cm3 := 2.7
noncomputable def gram_to_carat := 0.2
noncomputable def inch_to_cm := 2.54

theorem density_carat_cubic_inch :
  let density_emerald_carat_cm3 := density_emerald_g_cm3 / gram_to_carat
  let cubic_inch_to_cm3 := inch_to_cm ^ 3
  let density_emerald_carat_inch3 := density_emerald_carat_cm3 * cubic_inch_to_cm3
  abs (density_emerald_carat_inch3 - 221) < 1 :=
by
  sorry

end density_carat_cubic_inch_l505_505093


namespace least_positive_integer_with_12_factors_l505_505338

def has_exactly_12_factors (n : ℕ) : Prop :=
  n.factors.length = 12

theorem least_positive_integer_with_12_factors : ∃ k : ℕ, has_exactly_12_factors k ∧ (∀ n : ℕ, has_exactly_12_factors n → n ≥ k) ∧ k = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_l505_505338


namespace always_two_real_roots_find_m_l505_505598

-- Given quadratic equation: x^2 - 4mx + 3m^2 = 0
-- Definitions for the problem
def quadratic_eq (m x : ℝ) : Prop := x^2 - 4 * m * x + 3 * m^2 = 0

-- Q1: Prove that this equation always has two real roots.
theorem always_two_real_roots (m : ℝ) : ∃ x₁ x₂ : ℝ, quadratic_eq m x₁ ∧ quadratic_eq m x₂ :=
by
  sorry

-- Q2: If m > 0 and the difference between the two real roots is 2, find the value of m.
theorem find_m (m : ℝ) (h₁ : m > 0) (h₂ : ∃ x₁ x₂ : ℝ, quadratic_eq m x₁ ∧ quadratic_eq m x₂ ∧ |x₁ - x₂| = 2) : m = 1 :=
by
  sorry

end always_two_real_roots_find_m_l505_505598


namespace solve_inequality_l505_505082

theorem solve_inequality (x : ℝ) :
  (x - 4) / (x^2 + 3 * x + 10) ≥ 0 ↔ x ∈ set.Ici 4 :=
by
  have denom_pos : ∀ x : ℝ, x^2 + 3 * x + 10 > 0 := by
    intro x
    calc
      x^2 + 3 * x + 10
        = (x + 3 / 2)^2 + 31 / 4 : by ring
      _ > 0                       : by norm_num; apply sq_nonneg
  sorry

end solve_inequality_l505_505082


namespace isosceles_triangle_perimeter_l505_505104

theorem isosceles_triangle_perimeter (a b c : ℝ) (h1 : a = b) (h2 : a = 9) (h3 : c = 4) (h4 : a + b > c) :
  a + b + c = 22 := 
by
  rw [h1, h2, h3]
  linarith

end isosceles_triangle_perimeter_l505_505104


namespace solve_for_x_l505_505083

theorem solve_for_x (x : ℝ) : 3^(9^x) = 27^(3^x) ↔ x = 1 :=
by {
  sorry
}

end solve_for_x_l505_505083


namespace area_of_region_l505_505713

open Set Real

def K1 := {p : ℝ × ℝ | |p.1| + |3 * p.2| ≤ 6}
def K2 := {p : ℝ × ℝ | |3 * p.1| + |p.2| ≤ 6}

def K := {p : ℝ × ℝ | ( |p.1| + |3 * p.2| - 6 ) * ( |3 * p.1| + |p.2| - 6 ) ≤ 0 }

theorem area_of_region : MeasureTheory.measureSpace.volume (K) = 24 := 
  sorry

end area_of_region_l505_505713


namespace least_positive_integer_with_12_factors_l505_505382

theorem least_positive_integer_with_12_factors : 
  ∃ (n : ℕ), (n > 0) ∧ (nat.factors_count n = 12) ∧ (∀ m, (m > 0) → (nat.factors_count m = 12) → n ≤ m) ∧ n = 108 :=
sorry

end least_positive_integer_with_12_factors_l505_505382


namespace clock_angle_915_pm_l505_505618

theorem clock_angle_915_pm :
  let minute_hand := 90
  let hour_hand := 277.5
  abs (hour_hand - minute_hand) = 187.5 :=
by
  sorry

end clock_angle_915_pm_l505_505618


namespace books_from_library_l505_505069

def initial_books : ℝ := 54.5
def additional_books_1 : ℝ := 23.7
def returned_books_1 : ℝ := 12.3
def additional_books_2 : ℝ := 15.6
def returned_books_2 : ℝ := 9.1
def additional_books_3 : ℝ := 7.2

def total_books : ℝ :=
  initial_books + additional_books_1 - returned_books_1 + additional_books_2 - returned_books_2 + additional_books_3

theorem books_from_library : total_books = 79.6 := by
  sorry

end books_from_library_l505_505069


namespace log_calculation_l505_505481

theorem log_calculation :
  let (a : ℝ) := log 5 25
      (b : ℝ) := log 10 (1 / 100)
      (c : ℝ) := log (Real.exp 1) (Real.sqrt (Real.exp 1))
      (d : ℝ) := 2 ^ log 2 1 in
  a + b + c + d = 3 / 2 := by
  sorry

end log_calculation_l505_505481


namespace volume_cone_l505_505817

-- Conditions
variables (A B C D : ℝ) (h : ℝ)
variables (angle_ACB : ∠ A C B = real.pi / 2) -- Right angle condition
variables (height_D : D = h) -- Height of the pyramid
variables (lateral_edges_equal : True) -- Lateral edges are equal (no direct translation needed)
variables (angle_DF_base : ∠ D ⟮h sqrt(3)⟯ = real.pi / 3) -- 60 degrees in radians
variables (angle_DE_base : ∠ D ⟮h / sqrt(3)⟯ = real.pi / 6) -- 30 degrees in radians

-- Target volume of the circumscribed cone
noncomputable def volume_of_circumscribed_cone := (10 * real.pi * h^3) / 9

-- Statement to prove
theorem volume_cone (h : ℝ) :
  volume_of_circumscribed_cone h = (10 * real.pi * h^3) / 9 := sorry

end volume_cone_l505_505817


namespace inverse_g_undefined_at_2_l505_505521

def g (x : ℝ) : ℝ := (2 * x - 5) / (x - 6)

theorem inverse_g_undefined_at_2 : ¬∃ y : ℝ, g y = 2 :=
by
  intro h
  sorry

end inverse_g_undefined_at_2_l505_505521


namespace least_positive_integer_with_12_factors_is_972_l505_505366

theorem least_positive_integer_with_12_factors_is_972 : ∃ k : ℕ, (∀ n : ℕ, (∃ (d : ℕ), d * k = n) ↔ n = 12) ∧ k = 972 := sorry

end least_positive_integer_with_12_factors_is_972_l505_505366


namespace linear_function_through_point_l505_505074

theorem linear_function_through_point (m : ℝ) :
  ∃ b, (∀ x y : ℝ, (y = m * x + b) ↔ ((0, 3) = (x, y))) → b = 3 :=
by {
  intro m,
  use 3,
  intro h,
  specialize h 0 3,
  simp at h,
  exact h.mp rfl,
}

end linear_function_through_point_l505_505074


namespace least_positive_integer_with_12_factors_l505_505252

theorem least_positive_integer_with_12_factors :
  ∃ k : ℕ, (1 ≤ k) ∧ (nat.factors k).length = 12 ∧
    ∀ n : ℕ, (1 ≤ n) ∧ (nat.factors n).length = 12 → k ≤ n := 
begin
  sorry
end

end least_positive_integer_with_12_factors_l505_505252


namespace least_positive_integer_with_12_factors_l505_505300

def has_exactly_12_factors (n : ℕ) : Prop :=
  (finset.range(n + 1).filter (λ d, n % d = 0)).card = 12

theorem least_positive_integer_with_12_factors : ∀ n : ℕ, has_exactly_12_factors n → n ≥ 150 :=
sorry

end least_positive_integer_with_12_factors_l505_505300


namespace product_of_possible_a_values_l505_505828

noncomputable def segment_length_eq : Prop :=
  ∀ a : ℝ, (sqrt ((2 * a - 4) ^ 2 + (a - 3) ^ 2)) = 2 * sqrt 10

theorem product_of_possible_a_values : ∀ (a1 a2 : ℝ), 
  (a1 ≠ a2) → segment_length_eq → ((a1 = 5 ∨ a1 = -3 / 5) ∧ (a2 = 5 ∨ a2 = -3 / 5)) 
  → a1 * a2 = -3 :=
by
  intros a1 a2 hneq hseg heq
  cases heq with ha1 ha2
  cases ha1; 
  cases ha2; 
  simp [ha1, ha2]
sorry

end product_of_possible_a_values_l505_505828


namespace quadratic_has_real_roots_find_value_of_m_l505_505595

theorem quadratic_has_real_roots (m : ℝ) : 
  ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (x1 - x2) ≠ 0 ∧ (x^2 - 4 * m * x + 3 * m^2 = 0) := 
by 
  sorry

theorem find_value_of_m (m : ℝ) (h1 : m > 0) (h2 : ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (x1 - x2 = 2)) :
  m = 1 := 
by 
  sorry

end quadratic_has_real_roots_find_value_of_m_l505_505595


namespace sum_mod_13_remainder_l505_505054

-- Definitions and given conditions 
variables (a b c d e : ℕ)

hypothesis a_mod : a % 13 = 3
hypothesis b_mod : b % 13 = 5
hypothesis c_mod : c % 13 = 7
hypothesis d_mod : d % 13 = 9
hypothesis e_mod : e % 13 = 11
hypothesis c_square : ∃ k : ℕ, c = k * k

-- Statement to prove
theorem sum_mod_13_remainder : 
  (a + b + c + d + e) % 13 = 9 :=
sorry

end sum_mod_13_remainder_l505_505054


namespace multiples_of_5_in_4_digit_range_l505_505653

theorem multiples_of_5_in_4_digit_range : 
  let count_multiples := (9995 - 1000) / 5 + 1
  in count_multiples = 1800 :=
by
  sorry

end multiples_of_5_in_4_digit_range_l505_505653


namespace repeating_decimal_to_fraction_l505_505502

theorem repeating_decimal_to_fraction :
  let x := 0.5 + 0.\overline{01} in
  x = 5 / 10 + 1 / 99 →
  x = 101 / 198 :=
by
  intro h
  rw [←h]
  linarith

end repeating_decimal_to_fraction_l505_505502


namespace least_positive_integer_with_12_factors_is_72_l505_505316

-- Definition of having exactly 12 factors for a positive integer
def has_exactly_12_factors (n : ℕ) : Prop := n > 0 ∧ (finset.filter (λ d, n % d = 0) (finset.range (n + 1))).card = 12

-- Statement of the problem in Lean 4
theorem least_positive_integer_with_12_factors_is_72 :
  ∃ n : ℕ, has_exactly_12_factors n ∧ ∀ m : ℕ, has_exactly_12_factors m → n ≤ m :=
begin
  use 72,
  split,
  { -- Prove that 72 has exactly 12 factors
    sorry },
  { -- Prove that 72 is the least such integer
    sorry }
end

end least_positive_integer_with_12_factors_is_72_l505_505316


namespace sin_cos_expression_l505_505575

noncomputable def point_P : ℝ × ℝ := (-3, 4)

def x := (point_P.1 : ℝ)
def y := (point_P.2 : ℝ)
def r := Real.sqrt (x^2 + y^2)

def cos_alpha := x / r
def sin_alpha := y / r

theorem sin_cos_expression : sin_alpha + 2 * cos_alpha = -2 / 5 :=
by
  sorry

end sin_cos_expression_l505_505575


namespace sum_sequence_l505_505542

theorem sum_sequence (a : Fin 101 → ℝ) 
    (h1 : a 1 + a 2 = 1)
    (h2 : a 2 + a 3 = 2)
    (h3 : a 3 + a 4 = 3)
    -- ... Similar conditions for a_4 + a_5 = 4 to a_99 + a_100 = 99
    (h98 : a 98 + a 99 = 98)
    (h99 : a 99 + a 100 = 99)
    (h100 : a 100 + a 1 = 100) : 
    (a 1 + a 2 + a 3 + ... + a 100) = 2525 :=
sorry

end sum_sequence_l505_505542


namespace least_positive_integer_with_12_factors_is_72_l505_505221

open Nat

def hasExactly12Factors (n : ℕ) : Prop :=
  (divisors n).length = 12

theorem least_positive_integer_with_12_factors_is_72 :
  ∃ n : ℕ, hasExactly12Factors n ∧ ∀ m : ℕ, hasExactly12Factors m → n ≤ m → n = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_is_72_l505_505221


namespace first_fisherman_fish_per_day_l505_505702

theorem first_fisherman_fish_per_day :
  let total_days := 213
  let first_30_days := 30
  let next_60_days := 60
  let remaining_days := total_days - first_30_days - next_60_days 
  let fish_by_second_30_days := first_30_days * 1
  let fish_by_second_60_days := next_60_days * 2
  let fish_by_second_remaining_days := remaining_days * 4
  let total_fish_by_second := fish_by_second_30_days + fish_by_second_60_days + fish_by_second_remaining_days
  let total_fish_by_first := total_fish_by_second + 3
  total_fish_by_first / total_days = 3 :=
by
  let total_days := 213
  let first_30_days := 30
  let next_60_days := 60
  let remaining_days := total_days - first_30_days - next_60_days 
  have fish_by_second_30_days : ℕ := first_30_days * 1
  have fish_by_second_60_days : ℕ := next_60_days * 2
  have fish_by_second_remaining_days : ℕ := remaining_days * 4
  have total_fish_by_second : ℕ := fish_by_second_30_days + fish_by_second_60_days + fish_by_second_remaining_days
  have total_fish_by_first : ℕ := total_fish_by_second + 3
  have final_result : total_fish_by_first / total_days = 3 := sorry
  exact final_result

end first_fisherman_fish_per_day_l505_505702


namespace least_integer_with_twelve_factors_l505_505183

open Classical

theorem least_integer_with_twelve_factors : ∃ k : ℕ, (∀ n : ℕ, (count_factors n = 12 → k ≤ n)) ∧ k = 72 :=
by
  sorry

-- Here count_factors is a function that returns the number of positive divisors of a given integer.

end least_integer_with_twelve_factors_l505_505183


namespace concurrency_or_parallel_l505_505067

open Geometry

variables {P₀ P₁ P₂ : Type} [Points P₀] [Points P₁] [Points P₂]
variables {A B C A₁ B₁ C₁ A₂ B₂ C₂ : P₀}

-- Definitions of the conditions in the problem
axiom on_sides : (∃ A₁ ∈ line(B, C), ∃ B₁ ∈ line(C, A), ∃ C₁ ∈ line(A, B))
axiom on_sides_A1B1C1 : (∃ A₂ ∈ line(B₁, C₁), ∃ B₂ ∈ line(C₁, A₁), ∃ C₂ ∈ line(A₁, B₁))
axiom concur_ABC : concurrent (line(A, A₁)) (line(B, B₁)) (line(C, C₁))
axiom concur_A1B1C1 : concurrent (line(A₁, A₂)) (line(B₁, B₂)) (line(C₁, C₂))

-- Proof goal to be stated
theorem concurrency_or_parallel :
concurrent (line(A, A₂)) (line(B, B₂)) (line(C, C₂)) ∨
parallel (line(A, A₂)) (line(B, B₂)) (line(A, A₂)) (line(C, C₂)) :=
by
  sorry

end concurrency_or_parallel_l505_505067


namespace find_n_l505_505822

-- Define the hyperbola and its properties
def hyperbola (m n : ℝ) : Prop :=
  m > 0 ∧ n > 0 ∧ 2 = (m / (m / 2)) ∧ ∃ f : ℝ × ℝ, f = (m, 0)

-- Define the parabola and its properties
def parabola_focus (m : ℝ) : Prop :=
  (m, 0) = (m, 0)

-- The statement we want to prove
theorem find_n (m : ℝ) (n : ℝ) (H_hyperbola : hyperbola m n) (H_parabola : parabola_focus m) : n = 12 :=
sorry

end find_n_l505_505822


namespace smallest_w_for_factors_l505_505680

def factors (n : ℕ) : List (ℕ × ℕ) :=
  match n with
  | 1 => []
  | n => 
    let p := Nat.min_fac n
    let m := Nat.multiplicity p n
    (p, m.get) :: factors ((n / p ^ m.get).factorial)

theorem smallest_w_for_factors (w : ℕ) (hw_pos : 0 < w) 
  (h2 : 2^7 ∣ 936 * w) (h3 : 3^4 ∣ 936 * w) 
  (h5 : 5^3 ∣ 936 * w) (h7 : 7^2 ∣ 936 * w) 
  (h11 : 11^2 ∣ 936 * w) : 
  w = 320166000 :=
by sorry

end smallest_w_for_factors_l505_505680


namespace max_min_nested_eq_q_l505_505988

open Real

theorem max_min_nested_eq_q 
  (p q r s t : ℝ) 
  (h1 : p < q) 
  (h2 : q < r) 
  (h3 : r < s) 
  (h4 : s < t) 
  (hpqrs : set.pairwise {p, q, r, s, t} (≠)) :
  let M := λ (x y : ℝ), max x y
  let m := λ (x y : ℝ), min x y
  M (M p (m q r)) (m s (m p t)) = q := by
sorry

end max_min_nested_eq_q_l505_505988


namespace least_positive_integer_with_12_factors_is_72_l505_505315

-- Definition of having exactly 12 factors for a positive integer
def has_exactly_12_factors (n : ℕ) : Prop := n > 0 ∧ (finset.filter (λ d, n % d = 0) (finset.range (n + 1))).card = 12

-- Statement of the problem in Lean 4
theorem least_positive_integer_with_12_factors_is_72 :
  ∃ n : ℕ, has_exactly_12_factors n ∧ ∀ m : ℕ, has_exactly_12_factors m → n ≤ m :=
begin
  use 72,
  split,
  { -- Prove that 72 has exactly 12 factors
    sorry },
  { -- Prove that 72 is the least such integer
    sorry }
end

end least_positive_integer_with_12_factors_is_72_l505_505315


namespace infinite_geometric_series_second_term_l505_505923

theorem infinite_geometric_series_second_term (a r S : ℝ) (h1 : r = 1 / 4) (h2 : S = 16) (h3 : S = a / (1 - r)) : a * r = 3 := 
sorry

end infinite_geometric_series_second_term_l505_505923


namespace four_digit_multiples_of_5_count_l505_505642

-- Define the range of four-digit numbers
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

-- Define a multiple of 5
def is_multiple_of_five (n : ℕ) : Prop := n % 5 = 0

-- Define the required proof problem
theorem four_digit_multiples_of_5_count : 
  ∃ n : ℕ, (n = 1800) ∧ (∀ k : ℕ, is_four_digit k ∧ is_multiple_of_five k → n = 1800) :=
sorry

end four_digit_multiples_of_5_count_l505_505642


namespace minutes_practiced_other_days_l505_505462

theorem minutes_practiced_other_days (total_hours : ℕ) (minutes_per_day : ℕ) (num_days : ℕ) :
  total_hours = 450 ∧ minutes_per_day = 86 ∧ num_days = 2 → (total_hours - num_days * minutes_per_day) = 278 := by
  sorry

end minutes_practiced_other_days_l505_505462


namespace percentage_of_mara_pink_crayons_l505_505769

-- Define variables and constants
variables (P : ℝ) (mara_crayons : ℝ := 40) (luna_crayons : ℝ := 50) (luna_pink_percentage: ℝ := 0.2)
variables (total_pink_crayons : ℝ := 14)

-- Define conditions given in the problem
def mara_pink_crayons (P : ℝ) := (P / 100) * mara_crayons
def luna_pink_crayons := luna_pink_percentage * luna_crayons
def total_pink := mara_pink_crayons P + luna_pink_crayons

-- The proof goal
theorem percentage_of_mara_pink_crayons : mara_pink_crayons P + luna_pink_crayons = total_pink_crayons -> P = 10 :=
by
  sorry

end percentage_of_mara_pink_crayons_l505_505769


namespace find_f_neg_a_l505_505550

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin x - 4 * Real.tan x + 1

theorem find_f_neg_a (a : ℝ) (h : f a = 2) : f (-a) = 0 :=
by
  sorry

end find_f_neg_a_l505_505550


namespace trigonometric_relation_for_given_sum_l505_505428

theorem trigonometric_relation_for_given_sum (a : ℕ → ℕ) (h : ∀ i j, i ≠ j → a i ≠ a j)
  (sum_eq : (∑ i in finset.range 7, 2 ^ (a i)) = 2008):
  let m := ∑ i in finset.range 7, a i in
  sin (m : ℝ) > tan (m : ℝ) ∧ tan (m : ℝ) > cos (m : ℝ) :=
by
  -- We can assume the values based on the binary representation of 2008
  have a_values : (a 0 = 10 ∧ a 1 = 9 ∧ a 2 = 8 ∧ a 3 = 7 ∧ a 4 = 6 ∧ a 5 = 4 ∧ a 6 = 3) := sorry,
  let m := 10 + 9 + 8 + 7 + 6 + 4 + 3,
  have m_val : m = 47 := by norm_num,
  have sin_val := real.sin_pos_of_pos_of_lt_pi (by norm_num : 0 < 47) (by norm_num : 47 < 3.14 * 15),
  have cos_val := real.cos_neg_of_pi_div_two_lt_of_lt (by norm_num : 3.14 * 14.5 < 47) (by norm_num : 47 < 3.14 * 14.7),
  have tan_val := real.tan_pos_of_div_two_pi_lt_of_lt (by norm_num : 3.14 * 14 + 5 * 3.14/6 < 47) (by norm_num : 47 < 3.14 *15),
  exact ⟨sin_val, tan_val, cos_val⟩

end trigonometric_relation_for_given_sum_l505_505428


namespace dealer_profit_percent_l505_505439

theorem dealer_profit_percent (CP SP : ℝ) (h1 : CP = SP) (h2 : SP > 0) : (SP - 0.7 * CP) / (0.7 * CP) * 100 = 42.857 :=
by
  assume CP SP : ℝ
  assume h1 : CP = SP
  assume h2 : SP > 0
  show (SP - 0.7 * CP) / (0.7 * CP) * 100 = 42.857
  sorry

end dealer_profit_percent_l505_505439


namespace angle_BHC_in_degrees_l505_505009

theorem angle_BHC_in_degrees {A B C H D E F : Type}
  (h_triangle : triangle A B C)
  (h_altitudes : are_altitudes A B C D E F)
  (H_orthocenter : is_orthocenter D E F H)
  (angle_ABC : ∠ B A C = 41)
  (angle_ACB : ∠ B C A = 27) :
  ∠ B H C = 68 :=
sorry

end angle_BHC_in_degrees_l505_505009


namespace arithmetic_sequence_common_difference_l505_505562

theorem arithmetic_sequence_common_difference :
  ∃ d : ℤ, 
    (∀ n, n ≤ 6 → 23 + (n - 1) * d > 0) ∧ 
    (∀ n, n ≥ 7 → 23 + (n - 1) * d < 0) ∧
    d = -4 :=
by
  sorry

end arithmetic_sequence_common_difference_l505_505562


namespace wildlife_sanctuary_l505_505475

theorem wildlife_sanctuary : 
  ∀ (b i : ℕ), 
  b + i = 300 ∧ 
  2 * b + 6 * i = 780 → 
  b = 255 := 
by
  intros b i
  assume h
  cases h with h1 h2
  sorry

end wildlife_sanctuary_l505_505475


namespace least_positive_integer_with_12_factors_l505_505245

theorem least_positive_integer_with_12_factors :
  ∃ k : ℕ, (1 ≤ k) ∧ (nat.factors k).length = 12 ∧
    ∀ n : ℕ, (1 ≤ n) ∧ (nat.factors n).length = 12 → k ≤ n := 
begin
  sorry
end

end least_positive_integer_with_12_factors_l505_505245


namespace correct_option_D_l505_505470

def A : Prop :=
  ∀ (R_squared : ℝ), R_squared = 0.80 → false

def B : Prop :=
  ∀ (r : ℝ), r = 0.852 → false

def C : Prop :=
  ∀ (R_squared : ℝ), R_squared < 1 → false

def D : Prop :=
  ∀ (R_squared : ℝ), R_squared ≥ 0 → true

theorem correct_option_D : D ∧ ¬A ∧ ¬B ∧ ¬C :=
by
  split
  · intro R_squared
    trivial
  · sorry
  · sorry
  · sorry

end correct_option_D_l505_505470


namespace least_positive_integer_with_12_factors_is_96_l505_505391

def has_exactly_12_factors (n : ℕ) : Prop :=
  (12 : ℕ) = (List.range (n + 1)).filter (λ x, n % x = 0).length

theorem least_positive_integer_with_12_factors_is_96 :
  n ∈ (ℕ) ∧ has_exactly_12_factors n → n = 96 :=
by
  sorry

end least_positive_integer_with_12_factors_is_96_l505_505391


namespace least_positive_integer_with_12_factors_is_96_l505_505398

def has_exactly_12_factors (n : ℕ) : Prop :=
  (12 : ℕ) = (List.range (n + 1)).filter (λ x, n % x = 0).length

theorem least_positive_integer_with_12_factors_is_96 :
  n ∈ (ℕ) ∧ has_exactly_12_factors n → n = 96 :=
by
  sorry

end least_positive_integer_with_12_factors_is_96_l505_505398


namespace total_weight_of_dumbbell_system_l505_505483

-- Definitions from the given conditions
def weight_pair1 : ℕ := 3
def weight_pair2 : ℕ := 5
def weight_pair3 : ℕ := 8

-- Goal: Prove that the total weight of the dumbbell system is 32 lbs
theorem total_weight_of_dumbbell_system :
  2 * weight_pair1 + 2 * weight_pair2 + 2 * weight_pair3 = 32 :=
by sorry

end total_weight_of_dumbbell_system_l505_505483


namespace smaller_angle_at_9_15_is_172_5_l505_505620

-- Defining the problem in Lean
def hour_hand_angle (hours minutes : ℕ) : ℝ :=
  hours * 30 + minutes * 0.5

def minute_hand_angle (minutes : ℕ) : ℝ :=
  minutes * 6

def angle_between_hands (hours minutes : ℕ) : ℝ :=
  let hour_angle := hour_hand_angle hours minutes
  let minute_angle := minute_hand_angle minutes
  let angle := real.abs (hour_angle - minute_angle)
  if angle < 180 then angle else 360 - angle

-- Definition of the smaller angle at 9:15
def smaller_angle_9_15 := angle_between_hands 9 15

-- The problem statement in Lean 4
theorem smaller_angle_at_9_15_is_172_5 :
  smaller_angle_9_15 = 172.5 := by
  sorry

end smaller_angle_at_9_15_is_172_5_l505_505620


namespace least_positive_integer_with_12_factors_l505_505175

theorem least_positive_integer_with_12_factors :
  ∃ m : ℕ, 0 < m ∧ (∀ n : ℕ, 0 < n → (n ≠ m →  m ≤ n)) ∧ (12 = (m.factors.toFinset.card)) := sorry

end least_positive_integer_with_12_factors_l505_505175


namespace least_positive_integer_with_12_factors_is_72_l505_505220

open Nat

def hasExactly12Factors (n : ℕ) : Prop :=
  (divisors n).length = 12

theorem least_positive_integer_with_12_factors_is_72 :
  ∃ n : ℕ, hasExactly12Factors n ∧ ∀ m : ℕ, hasExactly12Factors m → n ≤ m → n = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_is_72_l505_505220


namespace contrapositive_statement_l505_505092

theorem contrapositive_statement (x y : ℤ) : ¬ (x + y) % 2 = 1 → ¬ (x % 2 = 1 ∧ y % 2 = 1) :=
sorry

end contrapositive_statement_l505_505092


namespace interest_second_month_l505_505024

theorem interest_second_month {P r n : ℝ} (hP : P = 200) (hr : r = 0.10) (hn : n = 12) :
  (P * (1 + r / n) ^ (n * (1/12)) - P) * r / n = 1.68 :=
by
  sorry

end interest_second_month_l505_505024


namespace least_positive_integer_with_12_factors_l505_505378

theorem least_positive_integer_with_12_factors : 
  ∃ (n : ℕ), (n > 0) ∧ (nat.factors_count n = 12) ∧ (∀ m, (m > 0) → (nat.factors_count m = 12) → n ≤ m) ∧ n = 108 :=
sorry

end least_positive_integer_with_12_factors_l505_505378


namespace point_symmetric_y_axis_l505_505716

theorem point_symmetric_y_axis (P : ℝ × ℝ) (hx : P = (2, 5)) : reflection_y P = (-2, 5) :=
by sorry

end point_symmetric_y_axis_l505_505716


namespace harriet_speed_l505_505419

-- Conditions as definitions
def speed_return : ℝ := 160 -- km/hr
def total_time : ℝ := 5 -- hours
def time_to_b : ℝ := 192 / 60 -- hours

-- Theorem to prove
theorem harriet_speed : 
  let time_return := total_time - time_to_b in
  let distance := speed_return * time_return in
  let speed_to_b := distance / time_to_b in
  speed_to_b = 90 :=
by
  sorry


end harriet_speed_l505_505419


namespace multiples_of_5_in_4_digit_range_l505_505652

theorem multiples_of_5_in_4_digit_range : 
  let count_multiples := (9995 - 1000) / 5 + 1
  in count_multiples = 1800 :=
by
  sorry

end multiples_of_5_in_4_digit_range_l505_505652


namespace least_positive_integer_with_12_factors_is_72_l505_505321

-- Definition of having exactly 12 factors for a positive integer
def has_exactly_12_factors (n : ℕ) : Prop := n > 0 ∧ (finset.filter (λ d, n % d = 0) (finset.range (n + 1))).card = 12

-- Statement of the problem in Lean 4
theorem least_positive_integer_with_12_factors_is_72 :
  ∃ n : ℕ, has_exactly_12_factors n ∧ ∀ m : ℕ, has_exactly_12_factors m → n ≤ m :=
begin
  use 72,
  split,
  { -- Prove that 72 has exactly 12 factors
    sorry },
  { -- Prove that 72 is the least such integer
    sorry }
end

end least_positive_integer_with_12_factors_is_72_l505_505321


namespace tree_height_on_slope_l505_505724

theorem tree_height_on_slope (H H_adjusted : ℝ) : 
  (Jane_shadow tree_shadow : ℝ) 
  (Jane_height : ℝ) 
  (slope_angle : ℝ) 
  (sin15 : ℝ) 
  (shadow_ratio : ℝ) :

  Jane_shadow = 0.5 ∧
  tree_shadow = 10 ∧
  Jane_height = 1.5 ∧
  slope_angle = 15 ∧
  sin15 = Real.sin (15 * Real.pi / 180) ∧
  shadow_ratio = Jane_height / Jane_shadow →
  H = tree_shadow * shadow_ratio ∧ 
  H_adjusted = H * sin15 →
  H_adjusted ≈ 7.764 :=
by
  sorry

end tree_height_on_slope_l505_505724


namespace trapezoid_area_bisection_l505_505099

/-- Given a trapezoid ABCD with extensions of AB and CD intersecting at E,
  M1 as the midpoint of AB, a perpendicular M1N1 such that M1N1 = 1/2 * AB,
  and segment EN on ray EA, where EN = EN1.
  Prove that the line passing through point N parallel to the bases of the trapezoid bisects its area. --/
theorem trapezoid_area_bisection
  {A B C D E M1 N N1 : Point}
  {AB CD AD BC : Line}
  (h_trapezoid : Trapezoid ABCD AB CD AD BC)
  (h_intersect : extensions AB CD E)
  (h_midpoint : midpoint M1 A B)
  (h_perpendicular : perpendicular M1N1 AB)
  (h_length : length M1N1 = (1 / 2) * length AB)
  (h_ray : segment_on_ray EN E A)
  (h_equal : length EN = length EN1) :
  bisects_area (parallel_line_through N AD BC) ABCD :=
sorry

end trapezoid_area_bisection_l505_505099


namespace last_digit_of_3_pow_2012_l505_505501

-- Theorem: The last digit of 3^2012 is 1 given the cyclic pattern of last digits for powers of 3.
theorem last_digit_of_3_pow_2012 : (3 ^ 2012) % 10 = 1 :=
by
  sorry

end last_digit_of_3_pow_2012_l505_505501


namespace difference_of_largest_and_smallest_l505_505497

-- Defining the digits
def digits := [6, 2, 5]

-- Largest number function
def largest_number (l : List ℕ) : ℕ :=
  l.permutations.map (λ p, p.foldl (λ acc x, 10 * acc + x) 0).maximum'.getOrElse 0

-- Smallest number function
def smallest_number (l : List ℕ) : ℕ :=
  l.permutations.map (λ p, p.foldl (λ acc x, 10 * acc + x) 0).minimum'.getOrElse 0

-- Define them for the specific digits
def largest := largest_number digits
def smallest := smallest_number digits

-- Prove the difference
theorem difference_of_largest_and_smallest :
  largest - smallest = 396 := by
  sorry

end difference_of_largest_and_smallest_l505_505497


namespace least_positive_integer_with_12_factors_is_96_l505_505393

def has_exactly_12_factors (n : ℕ) : Prop :=
  (12 : ℕ) = (List.range (n + 1)).filter (λ x, n % x = 0).length

theorem least_positive_integer_with_12_factors_is_96 :
  n ∈ (ℕ) ∧ has_exactly_12_factors n → n = 96 :=
by
  sorry

end least_positive_integer_with_12_factors_is_96_l505_505393


namespace count_real_root_quadratics_l505_505953

theorem count_real_root_quadratics : 
  let valid_as := {1, 2}
  let valid_bc := {1, 2, 3, 4}
  let real_root_count := λ a b c : ℕ, b^2 ≥ 4*a*c
  let quadratics_with_real_roots := (∑ a in valid_as, ∑ b in valid_bc, ∑ c in valid_bc, if real_root_count a b c then 1 else 0)
  quadratics_with_real_roots = 10 :=
by
  -- Here we would provide the proof details, but it is marked with sorry to skip.
  sorry

end count_real_root_quadratics_l505_505953


namespace infinite_series_sum_l505_505942

theorem infinite_series_sum :
  (∑' n : ℕ, (2 * (n + 1) * (n + 1) + (n + 1) + 1) / ((n + 1) * ((n + 1) + 1) * ((n + 1) + 2))) = 5 / 6 := by
  sorry

end infinite_series_sum_l505_505942


namespace x_coordinate_of_tangent_point_l505_505584

noncomputable def f (x a : ℝ) : ℝ := Real.exp x + a * Real.exp (-x)

theorem x_coordinate_of_tangent_point 
  (a : ℝ) 
  (h_even : ∀ x : ℝ, f x a = f (-x) a)
  (h_slope : ∃ m : ℝ, Real.exp m - Real.exp (-m) = 3 / 2) : 
  ∃ m : ℝ, m = Real.log 2 := 
by
  sorry

end x_coordinate_of_tangent_point_l505_505584


namespace coordinates_of_point_M_l505_505116

noncomputable def curve (x : ℝ) : ℝ := x^2 + x - 2

def slope_of_tangent (x : ℝ) : ℝ := 2 * x + 1

theorem coordinates_of_point_M :
  ∃ x y : ℝ, curve x = y ∧ slope_of_tangent x = 3 ∧ x = 1 ∧ y = 0 :=
begin
  sorry
end

end coordinates_of_point_M_l505_505116


namespace greatest_negative_value_x_minus_y_l505_505512

theorem greatest_negative_value_x_minus_y :
  (∃ x y : ℝ, (1 - Real.cot x) * (1 + Real.cot y) = 2 ∧ x - y = -3 * Real.pi / 4) :=
sorry

end greatest_negative_value_x_minus_y_l505_505512


namespace minutes_practiced_other_days_l505_505460

theorem minutes_practiced_other_days (total_hours : ℕ) (minutes_per_day : ℕ) (num_days : ℕ) :
  total_hours = 450 ∧ minutes_per_day = 86 ∧ num_days = 2 → (total_hours - num_days * minutes_per_day) = 278 := by
  sorry

end minutes_practiced_other_days_l505_505460


namespace tan_alpha_l505_505534

theorem tan_alpha {α : ℝ} (h : 3 * Real.sin α + 4 * Real.cos α = 5) : Real.tan α = 3 / 4 :=
by
  -- Proof goes here
  sorry

end tan_alpha_l505_505534


namespace unique_solution_of_pair_of_equations_l505_505991

-- Definitions and conditions
def pair_of_equations (x k : ℝ) : Prop :=
  (x^2 + 1 = 4 * x + k)

-- Theorem to prove
theorem unique_solution_of_pair_of_equations :
  ∃ k : ℝ, (∀ x : ℝ, pair_of_equations x k -> x = 2) ∧ k = 0 :=
by
  -- Proof omitted
  sorry

end unique_solution_of_pair_of_equations_l505_505991


namespace least_positive_integer_with_12_factors_is_72_l505_505219

open Nat

def hasExactly12Factors (n : ℕ) : Prop :=
  (divisors n).length = 12

theorem least_positive_integer_with_12_factors_is_72 :
  ∃ n : ℕ, hasExactly12Factors n ∧ ∀ m : ℕ, hasExactly12Factors m → n ≤ m → n = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_is_72_l505_505219


namespace least_positive_integer_with_12_factors_l505_505161

theorem least_positive_integer_with_12_factors : ∃ n : ℕ, (nat.factors_count n = 12 ∧ ∀ m : ℕ, (nat.factors_count m = 12) → n ≤ m) ∧ n = 72 :=
begin
  sorry
end

end least_positive_integer_with_12_factors_l505_505161


namespace least_positive_integer_with_12_factors_l505_505201

/--
Prove that the smallest positive integer with exactly 12 positive factors is 60.
-/
theorem least_positive_integer_with_12_factors : 
  ∃ (n : ℕ), (0 < n) ∧ (nat.factors_count n = 12) ∧ (∀ m : ℕ, (0 < m) ∧ (nat.factors_count m = 12) → n ≤ m) → n = 60 :=
sorry

end least_positive_integer_with_12_factors_l505_505201


namespace cricket_team_members_l505_505132

theorem cricket_team_members (n : ℕ)
    (captain_age : ℕ) (wicket_keeper_age : ℕ) (average_age : ℕ)
    (remaining_average_age : ℕ) (total_age : ℕ) (remaining_players : ℕ) :
    captain_age = 27 →
    wicket_keeper_age = captain_age + 3 →
    average_age = 24 →
    remaining_average_age = average_age - 1 →
    total_age = average_age * n →
    remaining_players = n - 2 →
    total_age = captain_age + wicket_keeper_age + remaining_average_age * remaining_players →
    n = 11 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end cricket_team_members_l505_505132


namespace set_difference_equality_l505_505765

theorem set_difference_equality :
  let A := {x : ℝ | x < 4}
  let B := {x : ℝ | x^2 - 4 * x + 3 > 0}
  let S := {x : ℝ | x ∈ A ∧ x ∉ (set.Inter {x ∈ A | x ∈ B})}
  S = {x : ℝ | 1 ≤ x ∧ x ≤ 3} :=
by {
  let A := {x : ℝ | x < 4},
  let B := {x : ℝ | x^2 - 4 * x + 3 > 0},
  let S := {x : ℝ | x ∈ A ∧ x ∉ (A ∩ B)},
  sorry
}

end set_difference_equality_l505_505765


namespace least_positive_integer_with_12_factors_is_972_l505_505361

theorem least_positive_integer_with_12_factors_is_972 : ∃ k : ℕ, (∀ n : ℕ, (∃ (d : ℕ), d * k = n) ↔ n = 12) ∧ k = 972 := sorry

end least_positive_integer_with_12_factors_is_972_l505_505361


namespace towels_folded_together_l505_505018

/-
Jane, Kyla, and Anthony's towel-folding rates.
-/
def Jane_rate := 3 * 60 / 5
def Kyla_rate := 5 * 60 / 10
def Anthony_rate := 7 * 60 / 20

/-- The total number of towels they can fold together in one hour. -/
theorem towels_folded_together : Jane_rate + Kyla_rate + Anthony_rate = 87 :=
by
  -- Jane folds 3 towels in 5 minutes, thus in 1 hour she can fold 36 towels.
  have hJane : Jane_rate = 36 := by sorry
  -- Kyla folds 5 towels in 10 minutes, thus in 1 hour she can fold 30 towels.
  have hKyla : Kyla_rate = 30 := by sorry
  -- Anthony folds 7 towels in 20 minutes, thus in 1 hour he can fold 21 towels.
  have hAnthony : Anthony_rate = 21 := by sorry
  -- Therefore, the total number of towels folded together in one hour is 87.
  calc
    Jane_rate + Kyla_rate + Anthony_rate
      = 36 + 30 + 21 : by sorry
      = 87 : by rfl

end towels_folded_together_l505_505018


namespace hypotenuse_intersection_incircle_diameter_l505_505499

/-- Let \( a \) and \( b \) be the legs of a right triangle with hypotenuse \( c \). 
    Let two circles be centered at the endpoints of the hypotenuse, with radii \( a \) and \( b \). 
    Prove that the segment of the hypotenuse that lies in the intersection of the two circles is equal in length to the diameter of the incircle of the triangle. -/
theorem hypotenuse_intersection_incircle_diameter (a b : ℝ) :
    let c := Real.sqrt (a^2 + b^2)
    let x := a + b - c
    let r := (a + b - c) / 2
    x = 2 * r :=
by
  let c := Real.sqrt (a^2 + b^2)
  let x := a + b - c
  let r := (a + b - c) / 2
  show x = 2 * r
  sorry

end hypotenuse_intersection_incircle_diameter_l505_505499


namespace least_positive_integer_with_12_factors_l505_505197

/--
Prove that the smallest positive integer with exactly 12 positive factors is 60.
-/
theorem least_positive_integer_with_12_factors : 
  ∃ (n : ℕ), (0 < n) ∧ (nat.factors_count n = 12) ∧ (∀ m : ℕ, (0 < m) ∧ (nat.factors_count m = 12) → n ≤ m) → n = 60 :=
sorry

end least_positive_integer_with_12_factors_l505_505197


namespace cos_2theta_identity_l505_505567

theorem cos_2theta_identity {θ : ℝ} 
  (h : 2^(-5/4 + 3 * Real.cos θ) + 1 = 2^(3/4 + 2 * Real.cos θ)) :
  Real.cos (2 * θ) = -7 / 32 := 
by
  sorry

end cos_2theta_identity_l505_505567


namespace area_enclosed_by_S_l505_505796

open Complex

def five_presentable (v : ℂ) : Prop := abs v = 5

def S : Set ℂ := {u | ∃ v : ℂ, five_presentable v ∧ u = v - (1 / v)}

theorem area_enclosed_by_S : 
  ∃ (area : ℝ), area = 624 / 25 * Real.pi :=
by
  sorry

end area_enclosed_by_S_l505_505796


namespace each_girl_gets_2_dollars_after_debt_l505_505088

variable (Lulu_saved : ℕ)
variable (Nora_saved : ℕ)
variable (Tamara_saved : ℕ)
variable (debt : ℕ)
variable (remaining : ℕ)
variable (each_girl_share : ℕ)

-- Conditions
axiom Lulu_saved_cond : Lulu_saved = 6
axiom Nora_saved_cond : Nora_saved = 5 * Lulu_saved
axiom Nora_Tamara_relation : Nora_saved = 3 * Tamara_saved
axiom debt_cond : debt = 40

-- Question == Answer to prove
theorem each_girl_gets_2_dollars_after_debt (total_saved : ℕ) (remaining: ℕ) (each_girl_share: ℕ) :
  total_saved = Tamara_saved + Nora_saved + Lulu_saved →
  remaining = total_saved - debt →
  each_girl_share = remaining / 3 →
  each_girl_share = 2 := 
sorry

end each_girl_gets_2_dollars_after_debt_l505_505088


namespace least_integer_with_twelve_factors_l505_505181

open Classical

theorem least_integer_with_twelve_factors : ∃ k : ℕ, (∀ n : ℕ, (count_factors n = 12 → k ≤ n)) ∧ k = 72 :=
by
  sorry

-- Here count_factors is a function that returns the number of positive divisors of a given integer.

end least_integer_with_twelve_factors_l505_505181


namespace sequence_probability_mn_l505_505447

theorem sequence_probability_mn :
  let a : ℕ → ℕ := λ n, if n = 1 then 1 else if n = 2 then 2 else a (n-1) + a (n-2)
  let probability := 233 / 2048
  let m := 233
  let n := 2048
  m + n = 2281 :=
by {
  have a1 : ∀ n, 1 ≤ n → a n = nat.fib (n + 1) := sorry,
  have prob : probability = (a 12 : ℚ) / (2^11 : ℚ) := sorry,
  rw [a1, add_comm],
  have hmn : m + n = 233 + 2048 := rfl,
  exact hmn
}

end sequence_probability_mn_l505_505447


namespace least_positive_integer_with_12_factors_l505_505253

theorem least_positive_integer_with_12_factors :
  ∃ k : ℕ, (1 ≤ k) ∧ (nat.factors k).length = 12 ∧
    ∀ n : ℕ, (1 ≤ n) ∧ (nat.factors n).length = 12 → k ≤ n := 
begin
  sorry
end

end least_positive_integer_with_12_factors_l505_505253


namespace parabola_focus_and_directrix_line_chord_length_through_focus_l505_505606

theorem parabola_focus_and_directrix :
  (focus (parabola y^2 = 6x) = (3 / 2, 0)) ∧ (directrix (parabola y^2 = 6x) = x = -3 / 2) := 
sorry

theorem line_chord_length_through_focus :
  let parabola := y^2 = 6x 
  let focus := (3 / 2, 0)
  let line_L := y = x - 3 / 2 
  in
  chord_length_parabola_intersect line_L parabola focus 45 = 12 :=
sorry

end parabola_focus_and_directrix_line_chord_length_through_focus_l505_505606


namespace least_positive_integer_with_12_factors_is_96_l505_505389

def has_exactly_12_factors (n : ℕ) : Prop :=
  (12 : ℕ) = (List.range (n + 1)).filter (λ x, n % x = 0).length

theorem least_positive_integer_with_12_factors_is_96 :
  n ∈ (ℕ) ∧ has_exactly_12_factors n → n = 96 :=
by
  sorry

end least_positive_integer_with_12_factors_is_96_l505_505389


namespace necessary_but_not_sufficient_for_hyperbola_l505_505005

noncomputable def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def propA (F1 F2 M : ℝ × ℝ) (k : ℝ) : Prop :=
  abs (dist M F1 - dist M F2) = k

def propB (F1 F2 : ℝ × ℝ) (traj : ℝ × ℝ → Prop) (k : ℝ) : Prop :=
  ∀ M, traj M → abs (dist M F1 - dist M F2) = k

theorem necessary_but_not_sufficient_for_hyperbola (F1 F2 : ℝ × ℝ) (traj : ℝ × ℝ → Prop) (k : ℝ) :
  (∀ M, abs (dist M F1 - dist M F2) = k → traj M) ∧ (∀ M, traj M → abs (dist M F1 - dist M F2) = k) :=
sorry

end necessary_but_not_sufficient_for_hyperbola_l505_505005


namespace least_positive_integer_with_12_factors_l505_505297

def has_exactly_12_factors (n : ℕ) : Prop :=
  (finset.range(n + 1).filter (λ d, n % d = 0)).card = 12

theorem least_positive_integer_with_12_factors : ∀ n : ℕ, has_exactly_12_factors n → n ≥ 150 :=
sorry

end least_positive_integer_with_12_factors_l505_505297


namespace complete_square_l505_505824

theorem complete_square (x : ℝ) : ∃ c d : ℝ, (x^2 - 6*x + 20 = (x + c)^2 + d) ∧ d = 11 :=
by
  let c := -3
  let d := 11
  use [c, d]
  constructor
  { sorry }
  { rfl }

end complete_square_l505_505824


namespace four_digit_multiples_of_five_count_l505_505664

theorem four_digit_multiples_of_five_count : 
  {n : ℕ | 1000 ≤ n ∧ n ≤ 9999 ∧ n % 5 = 0}.toFinset.card = 1800 :=
by
  sorry

end four_digit_multiples_of_five_count_l505_505664


namespace correct_calculation_l505_505415

-- Define the statements
def option_A (a : ℕ) := (a^2 + a^3 ≠ a^5)
def option_B (a : ℕ) := (a^2 * a^3 = a^5)
def option_C (a : ℕ) := ((a^2)^3 ≠ a^5)
def option_D (a : ℕ) := (a^5 / a^3 = a^2)

-- Prove that only option D is correct
theorem correct_calculation (a : ℕ) : option_A a → option_B a → option_C a → option_D a :=
by
  intros hA hB hC hD
  exact hD

end correct_calculation_l505_505415


namespace solve_equation_l505_505842

theorem solve_equation (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 1) :
  (3 / (x - 2) = 2 / (x - 1)) ↔ (x = -1) :=
sorry

end solve_equation_l505_505842


namespace quadratic_has_two_real_roots_find_m_l505_505591

def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem quadratic_has_two_real_roots (m : ℝ) :
  discriminant 1 (-4 * m) (3 * m^2) ≥ 0 :=
by
  unfold discriminant
  have h : (-4 * m)^2 - 4 * 1 * (3 * m^2) = 4 * m^2
  ring
  exact ge_of_eq h

theorem find_m (h : 0 < m) (root_diff : ℝ) 
  (diff_eq_two : root_diff = 2) : m = 1 :=
by
  -- Let the roots be x1 and x2
  let x1 := (4 * m + root_diff) / 2
  let x2 := (4 * m - root_diff) / 2
  have : x1 - x2 = root_diff :=
    by
      field_simp
      exact diff_eq_two
  have sum_eq := (x1 - x2) * (x1 + x2) - (x1 + x2) * (x1 - x2) = 4
  ring
  have h_m_eq_1 : 4 * m = 4,
  by field_simp
  exact h_m_eq_1

  have h_m_1 : m = 1,
  sorry
  exact ge_of_eq h_m_1

end quadratic_has_two_real_roots_find_m_l505_505591


namespace identity_of_brothers_l505_505774

theorem identity_of_brothers
  (first_brother_speaks : Prop)
  (second_brother_speaks : Prop)
  (one_tells_truth : first_brother_speaks → ¬ second_brother_speaks)
  (other_tells_truth : ¬first_brother_speaks → second_brother_speaks) :
  first_brother_speaks = false ∧ second_brother_speaks = true :=
by
  sorry

end identity_of_brothers_l505_505774


namespace product_fraction_eq_l505_505479

theorem product_fraction_eq :
  (\(n) → ∀ (f : ℕ → ℚ), (∀ k : ℕ, k > 0 ∧ k ≤ n → f k = (3 * k) / (5 * k + 2)) → ∏ k in finset.range(n).map (nat.succ),
  f k) = \(\frac{1}{2009}),
begin
  sorry
end

end product_fraction_eq_l505_505479


namespace find_intervals_l505_505548

def f (x : ℝ) : ℝ := x ^ 3 - 6 * x ^ 2 + 9 * x + 2

def f_derivative (x : ℝ) : ℝ := 3 * x ^ 2 - 12 * x + 9

def same_monotonicity_intervals (f f' : ℝ → ℝ) : set (set ℝ) :=
  { S | ∀ x ∈ S, ∀ y ∈ S, (f' x > 0 → f' y > 0) ∧ (f' x < 0 → f' y < 0) }

theorem find_intervals : 
  same_monotonicity_intervals f f_derivative = { I | I = set.Icc 1 2 ∨ I = set.Ici 3 } :=
sorry

end find_intervals_l505_505548


namespace solve_equation_l505_505791

theorem solve_equation (x y z : ℕ) (h1 : 2^x + 5^y + 63 = z!) (h2 : z ≥ 5) : 
  (x = 5 ∧ y = 2 ∧ z = 5) ∨ (x = 4 ∧ y = 4 ∧ z = 6) :=
sorry

end solve_equation_l505_505791


namespace four_digit_multiples_of_5_count_l505_505637

-- Define the range of four-digit numbers
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

-- Define a multiple of 5
def is_multiple_of_five (n : ℕ) : Prop := n % 5 = 0

-- Define the required proof problem
theorem four_digit_multiples_of_5_count : 
  ∃ n : ℕ, (n = 1800) ∧ (∀ k : ℕ, is_four_digit k ∧ is_multiple_of_five k → n = 1800) :=
sorry

end four_digit_multiples_of_5_count_l505_505637


namespace find_angle_A_determine_triangle_shape_l505_505700

noncomputable def sides := ℝ
noncomputable def angles := ℝ

variables {a b c : sides} {A B C : angles}
variables (h1 : 2 * a * cos C = 2 * b - c)
variables (h2 : a^2 ≤ b * (b + c))

theorem find_angle_A (h1 : 2 * a * cos C = 2 * b - c) : A = π / 3 := sorry

theorem determine_triangle_shape (h2 : a^2 ≤ b * (b + c)) (hA : A = π / 3) : (c = 2 * b) ∧ (a^2 + b^2 = c^2) := sorry

end find_angle_A_determine_triangle_shape_l505_505700


namespace jerry_recycling_time_l505_505725

-- Define all the conditions and computations
def cans_per_trip := 4
def drain_time_per_trip := 30 -- seconds
def walk_time_one_way := 10 -- seconds
def total_cans := 28

-- The proof statement in Lean 4
theorem jerry_recycling_time:
  let trips_needed := total_cans / cans_per_trip in
  let round_trip_time := 2 * walk_time_one_way in
  let time_per_trip := round_trip_time + drain_time_per_trip in
  let total_time := time_per_trip * trips_needed in
  total_time = 350 := by
  sorry

end jerry_recycling_time_l505_505725


namespace four_digit_multiples_of_5_count_l505_505623

theorem four_digit_multiples_of_5_count :
  let lower_bound := 200
  let upper_bound := 1999
  (upper_bound - lower_bound + 1) = 1800 :=
by
  let lower_bound := 200
  let upper_bound := 1999
  show (upper_bound - lower_bound + 1) = 1800,
  from sorry

end four_digit_multiples_of_5_count_l505_505623


namespace inequality_proof_l505_505048

open Real

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy: 0 < y) (hz : 0 < z):
  ( ( (x + y + z) / 3 ) ^ (x + y + z) ) ≤ x^x * y^y * z^z ∧ x^x * y^y * z^z ≤ ( (x^2 + y^2 + z^2) / (x + y + z) ) ^ (x + y + z) :=
by
  sorry

end inequality_proof_l505_505048


namespace part1_part2_l505_505581

def f (a : ℝ) (x : ℝ) : ℝ := (2 - a) * (x - 1) - 2 * Real.log x
def g (a : ℝ) (x : ℝ) : ℝ := f(a, x) + x

theorem part1 (a : ℝ) (h : a = 2) : ∀ x, 0 < x ∧ x < 2 → Derivative (g(2, x)) < 0 := 
by
  sorry

theorem part2
  (a : ℝ) 
  (h : ∀ x, 0 < x ∧ x < 1/2 → ((2 - a) * (x - 1) - 2 * Real.log x) > 0) : 
  a ≥ 2 - 4 * Real.log 2 := 
by 
  sorry

end part1_part2_l505_505581


namespace least_positive_integer_with_12_factors_l505_505166

theorem least_positive_integer_with_12_factors :
  ∃ m : ℕ, 0 < m ∧ (∀ n : ℕ, 0 < n → (n ≠ m →  m ≤ n)) ∧ (12 = (m.factors.toFinset.card)) := sorry

end least_positive_integer_with_12_factors_l505_505166


namespace sum_a_i_eq_2525_l505_505538

theorem sum_a_i_eq_2525 (a : ℕ → ℝ)
  (h1 : a 1 + a 2 = 1)
  (h2 : a 2 + a 3 = 2)
  (h3 : a 3 + a 4 = 3)
  -- include all intermediate conditions up to
  (h99 : a 99 + a 100 = 99)
  (h100 : a 100 + a 1 = 100) :
  (Finset.sum (Finset.range 100) (λ k, a (k + 1))) = 2525 :=
sorry

end sum_a_i_eq_2525_l505_505538


namespace least_positive_integer_with_12_factors_is_72_l505_505319

-- Definition of having exactly 12 factors for a positive integer
def has_exactly_12_factors (n : ℕ) : Prop := n > 0 ∧ (finset.filter (λ d, n % d = 0) (finset.range (n + 1))).card = 12

-- Statement of the problem in Lean 4
theorem least_positive_integer_with_12_factors_is_72 :
  ∃ n : ℕ, has_exactly_12_factors n ∧ ∀ m : ℕ, has_exactly_12_factors m → n ≤ m :=
begin
  use 72,
  split,
  { -- Prove that 72 has exactly 12 factors
    sorry },
  { -- Prove that 72 is the least such integer
    sorry }
end

end least_positive_integer_with_12_factors_is_72_l505_505319


namespace least_positive_integer_with_12_factors_is_96_l505_505263

theorem least_positive_integer_with_12_factors_is_96 :
  ∃ n : ℕ, (n = 96 ∧ (∀ m : ℕ, (m > 0 ∧ ∀ d : ℕ, d ∣ m → d > 0 → d < m → (∃ k : ℕ, d = k ∨ k * d = m) → k > 1 → m > n))) ∧ (∃ m : ℕ, m = 12 ∧ (∀ d : ℕ, d ∣ m → d ∈ {1, m} ∨ ∀ k : ℕ, k * d = m → d = k ∨ d < k)).
sorry

end least_positive_integer_with_12_factors_is_96_l505_505263


namespace james_coffee_problem_l505_505723

variable (x : ℕ)

/-- James decides to start making his own coffee. He buys a coffee machine for $200 and gets a $20 
discount. He figures it will cost him $3 a day to make his coffee. He previously bought a certain 
number of coffees a day for $4 each. The machine pays for itself in 36 days. Prove that he used 
to buy 2 coffees per day before he started making his own coffee. -/
theorem james_coffee_problem
    (coffee_machine_cost : ℕ)
    (discount : ℕ)
    (cost_per_day_to_make : ℕ)
    (days_to_pay_off : ℕ)
    (cost_per_coffee_before : ℕ)
    (x : ℕ)
    (H1 : coffee_machine_cost = 200)
    (H2 : discount = 20)
    (H3 : cost_per_day_to_make = 3)
    (H4 : days_to_pay_off = 36)
    (H5 : cost_per_coffee_before = 4)
    (H6 : (days_to_pay_off * (cost_per_coffee_before * x - cost_per_day_to_make) = coffee_machine_cost - discount)) :
    x = 2 :=
by {
  sorry,
}

end james_coffee_problem_l505_505723


namespace function_decreasing_on_interval_l505_505782

noncomputable def g (x : ℝ) := -(1 / 3) * Real.sin (4 * x - Real.pi / 3)
noncomputable def f (x : ℝ) := -(1 / 3) * Real.sin (4 * x)

theorem function_decreasing_on_interval :
  ∀ x y : ℝ, (-Real.pi / 8 ≤ x ∧ x ≤ Real.pi / 8) → (-Real.pi / 8 ≤ y ∧ y ≤ Real.pi / 8) → x < y → f x > f y :=
sorry

end function_decreasing_on_interval_l505_505782


namespace least_positive_integer_with_12_factors_l505_505374

theorem least_positive_integer_with_12_factors : 
  ∃ (n : ℕ), (n > 0) ∧ (nat.factors_count n = 12) ∧ (∀ m, (m > 0) → (nat.factors_count m = 12) → n ≤ m) ∧ n = 108 :=
sorry

end least_positive_integer_with_12_factors_l505_505374


namespace ratio_of_intercepts_l505_505141

theorem ratio_of_intercepts (b s t : ℝ) (h1 : s = -2 * b / 5) (h2 : t = -3 * b / 7) :
  s / t = 14 / 15 :=
by
  sorry

end ratio_of_intercepts_l505_505141


namespace figure100_l505_505970

-- Define the sequence function g(n)
def g (n : ℕ) : ℕ := 3 * n^2 + n + 1

-- Define the conditions using the given sequences
def figure0 := g 0 = 1
def figure1 := g 1 = 5
def figure2 := g 2 = 15
def figure3 := g 3 = 29
def figure4 := g 4 = 49

-- Main theorem to prove
theorem figure100 : g 100 = 30101 := 
by {
  -- Use the conditions provided (sorry for the actual proof)
  have h0 : g 0 = 1 := by sorry,
  have h1 : g 1 = 5 := by sorry,
  have h2 : g 2 = 15 := by sorry,
  have h3 : g 3 = 29 := by sorry,
  have h4 : g 4 = 49 := by sorry,
  sorry
}

end figure100_l505_505970


namespace ellipse_condition_necessary_but_not_sufficient_l505_505568

-- Define the conditions and proof statement in Lean 4
theorem ellipse_condition (m : ℝ) (h₁ : 2 < m) (h₂ : m < 6) : 
  (6 - m ≠ m - 2) -> 
  (∃ x y : ℝ, (x^2) / (m - 2) + (y^2) / (6 - m)= 1) :=
by
  sorry

theorem necessary_but_not_sufficient : (2 < m ∧ m < 6) ↔ (2 < m ∧ m < 6 ∧ m ≠ 4) :=
by
  sorry

end ellipse_condition_necessary_but_not_sufficient_l505_505568


namespace least_integer_with_twelve_factors_l505_505191

open Classical

theorem least_integer_with_twelve_factors : ∃ k : ℕ, (∀ n : ℕ, (count_factors n = 12 → k ≤ n)) ∧ k = 72 :=
by
  sorry

-- Here count_factors is a function that returns the number of positive divisors of a given integer.

end least_integer_with_twelve_factors_l505_505191


namespace sum_of_n_natural_numbers_l505_505117

theorem sum_of_n_natural_numbers (n : ℕ) (h : n * (n + 1) / 2 = 1035) : n = 46 :=
sorry

end sum_of_n_natural_numbers_l505_505117


namespace average_speed_trip_l505_505880

theorem average_speed_trip 
  (total_distance : ℕ)
  (first_distance : ℕ)
  (first_speed : ℕ)
  (second_distance : ℕ)
  (second_speed : ℕ)
  (h1 : total_distance = 60)
  (h2 : first_distance = 30)
  (h3 : first_speed = 60)
  (h4 : second_distance = 30)
  (h5 : second_speed = 30) :
  40 = total_distance / ((first_distance / first_speed) + (second_distance / second_speed)) :=
by sorry

end average_speed_trip_l505_505880


namespace least_positive_integer_with_12_factors_l505_505385

theorem least_positive_integer_with_12_factors : 
  ∃ (n : ℕ), (n > 0) ∧ (nat.factors_count n = 12) ∧ (∀ m, (m > 0) → (nat.factors_count m = 12) → n ≤ m) ∧ n = 108 :=
sorry

end least_positive_integer_with_12_factors_l505_505385


namespace least_positive_integer_with_12_factors_l505_505169

theorem least_positive_integer_with_12_factors :
  ∃ m : ℕ, 0 < m ∧ (∀ n : ℕ, 0 < n → (n ≠ m →  m ≤ n)) ∧ (12 = (m.factors.toFinset.card)) := sorry

end least_positive_integer_with_12_factors_l505_505169


namespace quadratic_two_real_roots_find_m_l505_505602

theorem quadratic_two_real_roots (m : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁ * x₂ = 3 * m^2 ∧ x₁ + x₂ = 4 * m :=
by
  sorry

theorem find_m (m : ℝ) (h : m > 0) (h_diff : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ - x₂ = 2) : m = 1 :=
by
  sorry

end quadratic_two_real_roots_find_m_l505_505602


namespace factorize_expression_l505_505504

variable (a b : ℝ)

theorem factorize_expression : a^2 - 4 * b^2 - 2 * a + 4 * b = (a + 2 * b - 2) * (a - 2 * b) := 
  sorry

end factorize_expression_l505_505504


namespace union_M_N_l505_505051

def M : set ℝ := {x | -1/2 < x ∧ x < 1/2}
def N : set ℝ := {x | 0 ≤ x ∧ x ≤ 1}

theorem union_M_N : M ∪ N = {x | -1/2 < x ∧ x ≤ 1} :=
by {
  sorry
}

end union_M_N_l505_505051


namespace least_positive_integer_with_12_factors_l505_505172

theorem least_positive_integer_with_12_factors :
  ∃ m : ℕ, 0 < m ∧ (∀ n : ℕ, 0 < n → (n ≠ m →  m ≤ n)) ∧ (12 = (m.factors.toFinset.card)) := sorry

end least_positive_integer_with_12_factors_l505_505172


namespace least_positive_integer_with_12_factors_l505_505257

theorem least_positive_integer_with_12_factors :
  ∃ k : ℕ, (1 ≤ k) ∧ (nat.factors k).length = 12 ∧
    ∀ n : ℕ, (1 ≤ n) ∧ (nat.factors n).length = 12 → k ≤ n := 
begin
  sorry
end

end least_positive_integer_with_12_factors_l505_505257


namespace least_positive_integer_with_12_factors_l505_505163

theorem least_positive_integer_with_12_factors : ∃ n : ℕ, (nat.factors_count n = 12 ∧ ∀ m : ℕ, (nat.factors_count m = 12) → n ≤ m) ∧ n = 72 :=
begin
  sorry
end

end least_positive_integer_with_12_factors_l505_505163


namespace least_positive_integer_with_12_factors_l505_505339

def has_exactly_12_factors (n : ℕ) : Prop :=
  n.factors.length = 12

theorem least_positive_integer_with_12_factors : ∃ k : ℕ, has_exactly_12_factors k ∧ (∀ n : ℕ, has_exactly_12_factors n → n ≥ k) ∧ k = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_l505_505339


namespace find_c_l505_505547

theorem find_c (c : ℝ) (f : ℝ → ℝ) (f_inv : ℝ → ℝ)
  (hf : ∀ x, f x = 2 / (3 * x + c))
  (hfinv : ∀ x, f_inv x = (2 - 3 * x) / (3 * x)) :
  c = 3 :=
by
  sorry

end find_c_l505_505547


namespace length_FQ_l505_505806

-- Define the right triangle and its properties
structure RightTriangle (DF DE EF : ℝ) :=
(right_angle_at_E : (DE^2 + EF^2 = DF^2))

-- Define the existence of a circle tangent to sides DF and EF
structure TangentCircle (DE DF EF : ℝ) :=
(center_on_DE : ∃ center : ℝ, center ∈ set.Icc 0 DE) -- Center is on segment DE
(tangent_to_DF : DF = √(center^2 + EF^2)) -- Tangency condition with side DF
(tangent_to_EF : EF = √(center^2 + DF^2)) -- Tangency condition with side EF

-- Problem statement: The length of FQ
theorem length_FQ
  (DE DF : ℝ)
  (DE_eq_7 : DE = 7)
  (DF_eq_sqrt85 : DF = real.sqrt 85)
  (EF : ℝ)
  (right_triangle : RightTriangle DF DE EF)
  (tangent_circle : TangentCircle DE DF EF) :
  (∃ FQ : ℝ, FQ = EF ∧ FQ = 6) := by
  -- Proof not included, just the statement
  sorry

end length_FQ_l505_505806


namespace length_of_platform_l505_505432

/-- Variables and parameters defining the problem conditions --/
variables (T : ℕ) (t_pole t_platform V L : ℕ)
  (h1 : T = 600)
  (h2 : t_pole = 36)
  (h3 : t_platform = 54)
  (h4 : V = T / t_pole)
  (h5 : V = (T + L) / t_platform)

/-- Statement to be proved: the length of the platform is 300 meters --/
theorem length_of_platform (T t_pole t_platform V L : ℕ)
  (h1 : T = 600)
  (h2 : t_pole = 36)
  (h3 : t_platform = 54)
  (h4 : V = T / t_pole)
  (h5 : V = (T + L) / t_platform) :
  L = 300 :=
sorry

end length_of_platform_l505_505432


namespace least_positive_integer_with_12_factors_is_72_l505_505215

open Nat

def hasExactly12Factors (n : ℕ) : Prop :=
  (divisors n).length = 12

theorem least_positive_integer_with_12_factors_is_72 :
  ∃ n : ℕ, hasExactly12Factors n ∧ ∀ m : ℕ, hasExactly12Factors m → n ≤ m → n = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_is_72_l505_505215


namespace percentage_decrease_l505_505836

theorem percentage_decrease (original_price new_price : ℕ) (h₁ : original_price = 250) (h₂ : new_price = 200) :
  (\( (original_price - new_price) / original_price \) * 100 = 20) :=
by
  sorry

end percentage_decrease_l505_505836


namespace least_positive_integer_with_12_factors_l505_505151

theorem least_positive_integer_with_12_factors : ∃ n : ℕ, (nat.factors_count n = 12 ∧ ∀ m : ℕ, (nat.factors_count m = 12) → n ≤ m) ∧ n = 72 :=
begin
  sorry
end

end least_positive_integer_with_12_factors_l505_505151


namespace pills_left_l505_505062

theorem pills_left (total_days : ℕ) (dose_per_day : ℕ) (elapsed_fraction : ℚ) :
  (total_days = 30) →
  (dose_per_day = 2) →
  (elapsed_fraction = 4 / 5) →
  let total_pills := total_days * dose_per_day in
  let days_elapsed := (elapsed_fraction * total_days) in
  let pills_taken := days_elapsed * dose_per_day in
  (total_pills - pills_taken) = 12 :=
by
  intros total_days_eq dose_per_day_eq elapsed_fraction_eq
  let total_pills := 30 * 2
  let days_elapsed := (4 / 5) * 30
  let pills_taken := days_elapsed * 2
  have h1 : total_pills = 60 := by norm_num
  have h2 : days_elapsed = 24 := by norm_num
  have h3 : pills_taken = 48 := by norm_num
  show 60 - 48 = 12
  
  sorry

end pills_left_l505_505062


namespace parabola_shift_units_l505_505834

theorem parabola_shift_units (m x1 x2 : ℝ) :
  let y := λ x : ℝ, x^2 - (2 * m - 1) * x - 6 * m in
  (y x1 = 0) ∧ (y x2 = 0) ∧ (x1 * x2 = x1 + x2 + 49) → abs x1 = 4 :=
by
  sorry

end parabola_shift_units_l505_505834


namespace company_profit_on_6000_sales_l505_505896

def profit_rate_first : ℝ := 0.06
def profit_rate_exceed : ℝ := 0.05
def threshold : ℝ := 1000
def total_sales : ℝ := 6000

def profit (sales : ℝ) : ℝ :=
  if sales <= threshold then
    profit_rate_first * sales
  else 
    profit_rate_first * threshold + profit_rate_exceed * (sales - threshold)

theorem company_profit_on_6000_sales :
  profit total_sales = 310 :=
by
  sorry

end company_profit_on_6000_sales_l505_505896


namespace quadrilateral_cyclic_iff_perpendicular_AC_BD_P_l505_505734

noncomputable section

open Classical 

namespace Geometry

structure Point (α : Type*) := (x : α) (y : α)
structure Line (α : Type*) := (A : Point α) (B : Point α)

variables {α : Type*} [Field α]

def midpoint (A B : Point α) : Point α :=
  ⟨ (A.x + B.x) / 2, (A.y + B.y) / 2 ⟩

def perpendicular (l1 l2 : Line α) : Prop :=
  let s1 := (l1.B.y - l1.A.y)/(l1.B.x - l1.A.x)
  let s2 := (l2.B.y - l2.A.y)/(l2.B.x - l2.A.x)
  s1 * s2 = -1 

def cyclic (A B C D : Point α) : Prop :=
  ∃ O : Point α, ∀ (P : Point α), P ∈ {A, B, C, D} → (O.x - P.x)^2 + (O.y - P.y)^2 = (O.x - A.x)^2 + (O.y - A.y)^2

variables (A B C D P M : Point α) 

def is_convex_quadrilateral (A B C D : Point α) : Prop := sorry -- Placeholder for convex quadrilateral

def intersection (l1 l2 : Line α) : Point α := sorry -- Placeholder for intersection of two lines

def AC := Line.mk A C
def BD := Line.mk B D
def PM := Line.mk P M
def DC := Line.mk D C

theorem quadrilateral_cyclic_iff_perpendicular_AC_BD_P : 
  is_convex_quadrilateral A B C D ∧ 
  intersection AC BD = P ∧ 
  perpendicular AC BD ∧ 
  M = midpoint A B → 
  (cyclic A B C D ↔ perpendicular PM DC) := 
by
  sorry

end quadrilateral_cyclic_iff_perpendicular_AC_BD_P_l505_505734


namespace geometric_locus_intersection_l505_505556

theorem geometric_locus_intersection :
  ∀ (M : ℝ × ℝ), 
  (∃ (m : ℝ), M = (4 / m^2, 4 / m) ∧ M.2^2 = 4 * M.1) →
  let P := (4 / (M.2 / M.1)^2, -4 / (M.2 / M.1)^3) in
  (P.1 = -4 ∧ (P ≠ (-4, 0))) :=
begin
  -- Proof omitted, only statement provided
  sorry
end

end geometric_locus_intersection_l505_505556


namespace machine_C_works_in_6_hours_l505_505768

theorem machine_C_works_in_6_hours :
  ∃ C : ℝ, (0 < C ∧ (1/4 + 1/12 + 1/C = 1/2)) → C = 6 :=
by
  sorry

end machine_C_works_in_6_hours_l505_505768


namespace least_positive_integer_with_12_factors_is_96_l505_505390

def has_exactly_12_factors (n : ℕ) : Prop :=
  (12 : ℕ) = (List.range (n + 1)).filter (λ x, n % x = 0).length

theorem least_positive_integer_with_12_factors_is_96 :
  n ∈ (ℕ) ∧ has_exactly_12_factors n → n = 96 :=
by
  sorry

end least_positive_integer_with_12_factors_is_96_l505_505390


namespace least_positive_integer_with_12_factors_is_72_l505_505310

-- Definition of having exactly 12 factors for a positive integer
def has_exactly_12_factors (n : ℕ) : Prop := n > 0 ∧ (finset.filter (λ d, n % d = 0) (finset.range (n + 1))).card = 12

-- Statement of the problem in Lean 4
theorem least_positive_integer_with_12_factors_is_72 :
  ∃ n : ℕ, has_exactly_12_factors n ∧ ∀ m : ℕ, has_exactly_12_factors m → n ≤ m :=
begin
  use 72,
  split,
  { -- Prove that 72 has exactly 12 factors
    sorry },
  { -- Prove that 72 is the least such integer
    sorry }
end

end least_positive_integer_with_12_factors_is_72_l505_505310


namespace correct_choice_l505_505471

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_increasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x < f y

def fA : ℝ → ℝ := λ x, x + 1
def fB : ℝ → ℝ := λ x, -x^2
def fC : ℝ → ℝ := λ x, -1/x
def fD : ℝ → ℝ := λ x, x^3 + x

theorem correct_choice : (is_odd fD ∧ is_increasing fD) ∧
  ¬ (is_odd fA ∧ is_increasing fA) ∧
  ¬ (is_odd fB ∧ is_increasing fB) ∧
  ¬ (is_odd fC ∧ is_increasing fC) :=
by {
  -- Prove is_odd for fD
  have h1 : is_odd fD,
  sorry,
  -- Prove is_increasing for fD
  have h2 : is_increasing fD,
  sorry,
  -- Prove ¬(is_odd fA ∧ is_increasing fA)
  have h3: ¬ (is_odd fA ∧ is_increasing fA),
  sorry,
  -- Prove ¬(is_odd fB ∧ is_increasing fB)
  have h4: ¬ (is_odd fB ∧ is_increasing fB),
  sorry,
  -- Prove ¬(is_odd fC ∧ is_increasing fC)
  have h5: ¬ (is_odd fC ∧ is_increasing fC),
  sorry,
  -- Conclude the theorem
  exact ⟨⟨h1, h2⟩, ⟨h3, ⟨h4, h5⟩⟩⟩
}

end correct_choice_l505_505471


namespace train_crossing_time_l505_505917

noncomputable def train_speed_km_hr : ℝ := 45
noncomputable def train_length_meters : ℝ := 200
noncomputable def conversion_factor : ℝ := 1000 / 3600

theorem train_crossing_time : 
  let speed_m_s := train_speed_km_hr * conversion_factor in
  let time_seconds := train_length_meters / speed_m_s in
  time_seconds = 16 :=
by
  let speed_m_s := train_speed_km_hr * conversion_factor
  let time_seconds := train_length_meters / speed_m_s
  sorry

end train_crossing_time_l505_505917


namespace how_many_unanswered_l505_505080

theorem how_many_unanswered (c w u : ℕ) (h1 : 25 + 5 * c - 2 * w = 95)
                            (h2 : 6 * c + u = 110) (h3 : c + w + u = 30) : u = 10 :=
by
  sorry

end how_many_unanswered_l505_505080


namespace perimeter_of_triangle_l505_505904

/-- Given a line passing through the origin, a vertical line x = 2, and a line y = 2 - (sqrt 5 / 5) * x,
    which form a right triangle. The perimeter of the triangle is 2 + (12 * sqrt 5 - 10) / 5 + 2 * sqrt 6. -/
theorem perimeter_of_triangle :
  let m : ℝ := - (real.sqrt 5 / 5), 
      line1 : ℝ → ℝ := λ (x : ℝ), real.sqrt 5 * x,
      vertex1 : ℝ × ℝ := (2, 2 * real.sqrt 5),
      line2 : ℝ → ℝ := λ (x : ℝ), 2 - m * x,
      vertex2 : ℝ × ℝ := (2, 2 - 2 * real.sqrt 5 / 5),
      vertical_side_length : ℝ := abs (2 * real.sqrt 5 - (2 - 2 * real.sqrt 5 / 5)),
      horizontal_side_length : ℝ := abs (0 - 2),
      hypotenuse_length : ℝ := real.sqrt (2^2 + (2 * real.sqrt 5)^2)
  in vertical_side_length + horizontal_side_length + hypotenuse_length = 2 + (12 * real.sqrt 5 - 10) / 5 + 2 * real.sqrt 6 := 
begin
  let m := -(real.sqrt 5 / 5),
  let line1 := λ (x : ℝ), real.sqrt 5 * x,
  let vertex1 := (2, 2 * real.sqrt 5),
  let line2 := λ (x : ℝ), 2 - m * x,
  let vertex2 := (2, 2 - 2 * real.sqrt 5 / 5),
  let vertical_side_length := abs (2 * real.sqrt 5 - (2 - 2 * real.sqrt 5 / 5)),
  let horizontal_side_length := abs (0 - 2),
  let hypotenuse_length := real.sqrt (2^2 + (2 * real.sqrt 5)^2),
  have h1 : vertical_side_length = (12 * real.sqrt 5 - 10) / 5, sorry,
  have h2 : horizontal_side_length = 2, sorry,
  have h3 : hypotenuse_length = 2 * real.sqrt 6, sorry,
  rw [h1, h2, h3],
  exact calc
    (12 * real.sqrt 5 - 10) / 5 + 2 + 2 * real.sqrt 6 = 2 + (12 * real.sqrt 5 - 10) / 5 + 2 * real.sqrt 6 : by ring,
end

end perimeter_of_triangle_l505_505904


namespace distance_traveled_by_center_l505_505889

noncomputable def ball_diameter := 8
noncomputable def ball_radius := ball_diameter / 2
noncomputable def r1 := 110
noncomputable def r2 := 70
noncomputable def r3 := 90

noncomputable def adjusted_r1 := r1 - ball_radius
noncomputable def adjusted_r2 := r2 + ball_radius
noncomputable def adjusted_r3 := r3 - ball_radius

noncomputable def distance_arc (radius : ℝ) := π * radius

noncomputable def total_distance := 
  distance_arc adjusted_r1 + 
  distance_arc adjusted_r2 + 
  distance_arc adjusted_r3

theorem distance_traveled_by_center (d_a b_r r1 r2 r3 : ℝ) (b_r = d_a / 2) :
  let ar1 := r1 - b_r,
      ar2 := r2 + b_r,
      ar3 := r3 - b_r in
  distance_arc ar1 + distance_arc ar2 + distance_arc ar3 = 266 * π :=
by
  have ball_diameter := 8
  have ball_radius := ball_diameter / 2
  have r1 := 110
  have r2 := 70
  have r3 := 90
  have adjusted_r1 := r1 - ball_radius
  have adjusted_r2 := r2 + ball_radius
  have adjusted_r3 := r3 - ball_radius
  have d1 := distance_arc adjusted_r1
  have d2 := distance_arc adjusted_r2
  have d3 := distance_arc adjusted_r3
  show distances = 266 * π from 
    calc 
    distance_arc adjusted_r1 + distance_arc adjusted_r2 + distance_arc adjusted_r3 
    = 266 * π
  sorry


end distance_traveled_by_center_l505_505889


namespace diophantine_eq_solutions_l505_505510

theorem diophantine_eq_solutions (p q r k : ℕ) (hp : p > 1) (hq : q > 1) (hr : r > 1) 
  (hp_prime : Prime p) (hq_prime : Prime q) (hr_prime : Prime r) (hk : k > 0) :
  p^2 + q^2 + 49*r^2 = 9*k^2 - 101 ↔ 
  (p = 3 ∧ q = 5 ∧ r = 3 ∧ k = 8) ∨ (p = 5 ∧ q = 3 ∧ r = 3 ∧ k = 8) :=
by sorry

end diophantine_eq_solutions_l505_505510


namespace ducks_in_marsh_l505_505134

theorem ducks_in_marsh 
  (num_geese : ℕ) 
  (total_birds : ℕ) 
  (num_ducks : ℕ)
  (h1 : num_geese = 58) 
  (h2 : total_birds = 95) 
  (h3 : total_birds = num_geese + num_ducks) : 
  num_ducks = 37 :=
by
  sorry

end ducks_in_marsh_l505_505134


namespace julia_money_left_l505_505731

def initial_money : ℕ := 40
def spent_on_game : ℕ := initial_money / 2
def money_left_after_game : ℕ := initial_money - spent_on_game
def spent_on_in_game_purchases : ℕ := money_left_after_game / 4
def final_money_left : ℕ := money_left_after_game - spent_on_in_game_purchases

theorem julia_money_left : final_money_left = 15 := by
  sorry

end julia_money_left_l505_505731


namespace four_digit_multiples_of_5_count_l505_505641

-- Define the range of four-digit numbers
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

-- Define a multiple of 5
def is_multiple_of_five (n : ℕ) : Prop := n % 5 = 0

-- Define the required proof problem
theorem four_digit_multiples_of_5_count : 
  ∃ n : ℕ, (n = 1800) ∧ (∀ k : ℕ, is_four_digit k ∧ is_multiple_of_five k → n = 1800) :=
sorry

end four_digit_multiples_of_5_count_l505_505641


namespace area_of_rectangle_at_stage_8_l505_505688

-- Define the conditions given in the problem
def square_side_length : ℝ := 4
def square_area : ℝ := square_side_length * square_side_length
def stages : ℕ := 8

-- Define the statement to be proved
theorem area_of_rectangle_at_stage_8 : (stages * square_area) = 128 := 
by 
  have h1 : square_area = 16 := by
    unfold square_area
    norm_num
  have h2 : (stages * square_area) = 8 * 16 := by
    unfold stages
    rw h1
  rw h2
  norm_num
  sorry

end area_of_rectangle_at_stage_8_l505_505688


namespace balloon_height_is_correct_l505_505530

noncomputable def height_of_balloon : ℝ :=
  let a : ℝ := 100
  let ha_length : ℝ := 180
  let hb_length : ℝ := 156
  let b : ℝ := a
  let h_square := ha_length^2 - a^2
  real.sqrt h_square

theorem balloon_height_is_correct :
  height_of_balloon = 50 * real.sqrt 10 :=
by
  let a : ℝ := 100
  let ha_length : ℝ := 180
  let hb_length : ℝ := 156
  let h_square := ha_length^2 - a^2
  exact real.sqrt h_square = 50 * real.sqrt 10
  sorry

end balloon_height_is_correct_l505_505530


namespace find_general_term_sum_first_n_terms_l505_505587

variable (a : ℕ → ℝ)

-- Conditions
axiom a_n (n : ℕ) : a n = 2^n
axiom a_arith_seq (a1 a2 a3 : ℝ) : (a1, a2 + 1, a3) forms arithmetic_sequence  
axiom a4_eq_8a1 : a 4 = 8 * a 1

-- Questions
theorem find_general_term :
  ∀ n, a n = 2^n := by
sorry

theorem sum_first_n_terms (n : ℕ) :
  ∑ i in finset.range n.succ, abs (a i - 4) = 2 ^ (n + 1) - 4 * n + 2 := by
sorry

end find_general_term_sum_first_n_terms_l505_505587


namespace count_four_digit_multiples_of_5_l505_505631

def is_multiple_of_5 (n : ℕ) : Prop :=
  n % 5 = 0

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def four_digit_multiples_of_5_count (lower upper : ℕ) :=
  (upper - lower + 1)

theorem count_four_digit_multiples_of_5 : 
  four_digit_multiples_of_5_count 200 1999 = 1800 :=
by
  sorry

end count_four_digit_multiples_of_5_l505_505631


namespace least_positive_integer_with_12_factors_l505_505344

theorem least_positive_integer_with_12_factors : ∃ k : ℕ, k > 0 ∧ (∀ n : ℕ, n > 0 → (factors_count n = 12 → k ≤ n)) ∧ k = 72 :=
by {
  sorry,
}

end least_positive_integer_with_12_factors_l505_505344


namespace least_positive_integer_with_12_factors_l505_505351

theorem least_positive_integer_with_12_factors : ∃ k : ℕ, k > 0 ∧ (∀ n : ℕ, n > 0 → (factors_count n = 12 → k ≤ n)) ∧ k = 72 :=
by {
  sorry,
}

end least_positive_integer_with_12_factors_l505_505351


namespace marcus_earnings_correct_l505_505961

noncomputable def marcus_earnings : ℝ :=
  let x := 65.20 / 10 in
  let total_hours := 30 + 20 in
  let total_earnings_before_tax := total_hours * x in
  let earnings_after_tax := 0.9 * total_earnings_before_tax in
  earnings_after_tax

theorem marcus_earnings_correct : marcus_earnings = 293.40 := by
  sorry

end marcus_earnings_correct_l505_505961


namespace least_integer_N_l505_505984

theorem least_integer_N (N : ℕ) (h : N = 6097392) :
  ∀ (S : Finset ℕ), S.card = 2016 → (∃ (T : Finset ℕ), T.card = 2016 ∧ T.sum = N) := sorry

end least_integer_N_l505_505984


namespace only_f2_symmetric_and_increasing_l505_505919

def f1 (x : ℝ) : ℝ := x^3
def f2 (x : ℝ) : ℝ := |x| - 1
def f3 (x : ℝ) : ℝ := -x^2 + 1
def f4 (x : ℝ) : ℝ := 3^x

theorem only_f2_symmetric_and_increasing : 
  (∀ x : ℝ, f2 x = f2 (-x))
  ∧ (∀ x : ℝ, x > 0 → f2 (x + 1) > f2 x)
  ∧ (∀ x : ℝ, f1 x ≠ f1 (-x) ∨ ∃ x : ℝ, x > 0 ∧ f1 (x + 1) ≤ f1 x)
  ∧ (∀ x : ℝ, f3 x = f3 (-x) ∧ ∃ x : ℝ, x > 0 ∧ f3 (x + 1) ≤ f3 x)
  ∧ (∀ x : ℝ, f4 x ≠ f4 (-x) ∨ ∃ x : ℝ, x > 0 ∧ f4 (x + 1) ≤ f4 x) :=
by
  sorry

end only_f2_symmetric_and_increasing_l505_505919


namespace least_integer_with_twelve_factors_l505_505192

open Classical

theorem least_integer_with_twelve_factors : ∃ k : ℕ, (∀ n : ℕ, (count_factors n = 12 → k ≤ n)) ∧ k = 72 :=
by
  sorry

-- Here count_factors is a function that returns the number of positive divisors of a given integer.

end least_integer_with_twelve_factors_l505_505192


namespace infinite_geometric_series_second_term_l505_505921

theorem infinite_geometric_series_second_term (a r S : ℝ) (h1 : r = 1 / 4) (h2 : S = 16) (h3 : S = a / (1 - r)) : a * r = 3 := 
sorry

end infinite_geometric_series_second_term_l505_505921


namespace rectangle_area_at_stage_8_l505_505686

-- Definitions based on conditions
def area_of_square (side_length : ℕ) : ℕ := side_length * side_length
def number_of_squares_in_stage (stage : ℕ) : ℕ := stage

-- The main theorem to prove
theorem rectangle_area_at_stage_8 : 
  area_of_square 4 * number_of_squares_in_stage 8 = 128 := by
  sorry

end rectangle_area_at_stage_8_l505_505686


namespace problem1_problem2_l505_505935

open Real

-- Proof problem for the first expression
theorem problem1 : 
  (-2^2 * (1 / 4) + 4 / (4/9) + (-1) ^ 2023 = 7) :=
by 
  sorry

-- Proof problem for the second expression
theorem problem2 : 
  (-1 ^ 4 + abs (2 - (-3)^2) + (1/2) / (-3/2) = 17/3) :=
by 
  sorry

end problem1_problem2_l505_505935


namespace sequence_pattern_l505_505516

theorem sequence_pattern (a b c : ℝ) (h1 : a = 19.8) (h2 : b = 18.6) (h3 : c = 17.4) 
  (h4 : ∀ n, n = a ∨ n = b ∨ n = c ∨ n = 16.2 ∨ n = 15) 
  (H : ∀ x y, (y = x - 1.2) → 
    (x = a ∨ x = b ∨ x = c ∨ y = 16.2 ∨ y = 15)) :
  (16.2 = c - 1.2) ∧ (15 = (c - 1.2) - 1.2) :=
by
  sorry

end sequence_pattern_l505_505516


namespace four_digit_multiples_of_5_l505_505644

theorem four_digit_multiples_of_5 : 
  let f := fun n => 1000 <= n ∧ n <= 9999 ∧ n % 5 = 0
  in finset.card (finset.filter f (finset.range 10000)) = 1800 :=
by
  sorry

end four_digit_multiples_of_5_l505_505644


namespace ellipse_equation_line_equation_l505_505098

-- Define the conditions for the ellipse
def ellipse (x y : ℝ) (a b : ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1
def point_on_ellipse (x y a b : ℝ) : Prop := ellipse x y a b

-- Define the coordinates of the foci and point P
def f1 : ℝ × ℝ := (0, -√53/3)
def f2 : ℝ × ℝ := (0, √53/3)
def p : ℝ × ℝ := (2, 4)

-- Define distances
def dist (p q : ℝ × ℝ) : ℝ := (p.1 - q.1)^2 + (p.2 - q.2)^2
def pf1 := dist p f1 = (4/3)^2
def pf2 := dist p f2 = (14/3)^2

-- Prove the equation of the ellipse
theorem ellipse_equation : ∃ (a b : ℝ), a = 3 ∧ b^2 = 28/9 ∧ ellipse 2 4 a b := by
  sorry

-- Define the line equation L passing through the center M of the circle
def center_circle : ℝ × ℝ := (-2, 1)
def line_through_M (x y : ℝ) := 56 * x - 81 * y + 193 = 0

-- Prove the equation of the line L
theorem line_equation : ∃ (k : ℝ), k = 56 / 81 ∧ line_through_M 56 (-193) := by
  sorry

end ellipse_equation_line_equation_l505_505098


namespace lisa_minimum_fifth_term_score_l505_505860

theorem lisa_minimum_fifth_term_score :
  ∀ (score1 score2 score3 score4 average_needed total_terms : ℕ),
  score1 = 84 →
  score2 = 80 →
  score3 = 82 →
  score4 = 87 →
  average_needed = 85 →
  total_terms = 5 →
  (∃ (score5 : ℕ), 
     (score1 + score2 + score3 + score4 + score5) / total_terms ≥ average_needed ∧ 
     score5 = 92) :=
by
  sorry

end lisa_minimum_fifth_term_score_l505_505860


namespace non_coplanar_points_parallel_lines_l505_505938

theorem non_coplanar_points_parallel_lines :
  ∃ (points : Finset (ℝ × ℝ × ℝ)), 
  (∃ (p1 p2 : points), p1 ≠ p2) ∧ 
  (∃ (p3 p4 : points), p3 ≠ p4 ∧ p3 ∉ {p1, p2} ∧ p4 ∉ {p1, p2}) ∧ 
  (∀ (A B : points), ∃ (C D : points), C ≠ D ∧ C ≠ A ∧ C ≠ B ∧ D ≠ A ∧ D ≠ B ∧
    ∃ (v1 v2 : ℝ × ℝ × ℝ), v1 = (A.1 - B.1, A.2 - B.2, A.3 - B.3) ∧ 
    v2 = (C.1 - D.1, C.2 - D.2, C.3 - D.3) ∧ 
    (v1.1 * v2.2 - v1.2 * v2.1 = 0 ∧ v1.1 * v2.3 - v1.3 * v2.1 = 0 ∧ v1.2 * v2.3 - v1.3 * v2.2 = 0)) :=
sorry

end non_coplanar_points_parallel_lines_l505_505938


namespace sum_a_i_eq_2525_l505_505539

theorem sum_a_i_eq_2525 (a : ℕ → ℝ)
  (h1 : a 1 + a 2 = 1)
  (h2 : a 2 + a 3 = 2)
  (h3 : a 3 + a 4 = 3)
  -- include all intermediate conditions up to
  (h99 : a 99 + a 100 = 99)
  (h100 : a 100 + a 1 = 100) :
  (Finset.sum (Finset.range 100) (λ k, a (k + 1))) = 2525 :=
sorry

end sum_a_i_eq_2525_l505_505539


namespace least_positive_integer_with_12_factors_l505_505333

def has_exactly_12_factors (n : ℕ) : Prop :=
  n.factors.length = 12

theorem least_positive_integer_with_12_factors : ∃ k : ℕ, has_exactly_12_factors k ∧ (∀ n : ℕ, has_exactly_12_factors n → n ≥ k) ∧ k = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_l505_505333


namespace log4_16_l505_505967

namespace LogarithmsProof

-- Define the conditions first
def log_b (b x : ℝ) : ℝ := sorry -- log_b function for ℝ numbers
def cond1 : 16 = 4^2 := by rfl -- condition 1: 16 = 4^2
def power_rule (b c : ℝ) (a : ℝ) : log_b b (a^c) = c * log_b b a := sorry -- power rule of logarithms
def cond3 : log_b 4 4 = 1 := by rfl -- condition 3: log4(4) = 1

-- Formalize the theorem statement
theorem log4_16 : log_b 4 16 = 2 := 
by
  unfold log_b
  apply sorry -- proof will go here
end LogarithmsProof

end log4_16_l505_505967


namespace minimum_combinations_required_l505_505558

noncomputable def safe_min_combinations : ℕ :=
  32

theorem minimum_combinations_required 
  (safe : Type) 
  (dials : ℕ) 
  (positions : ℕ) 
  (correct_open : ℕ → ℕ → ℕ → Prop)
  (H1 : dials = 3)
  (H2 : positions = 8) :
  ∃ min_combinations, min_combinations = 32 :=
begin
  use 32,
  exact eq.refl 32,
end

end minimum_combinations_required_l505_505558


namespace max_value_of_function_in_interval_l505_505514

open Real

noncomputable def my_function : ℝ → ℝ := λ x, 2 * x ^ 3 - 3 * x ^ 2

theorem max_value_of_function_in_interval :
  ∃ x ∈ Icc (-1 : ℝ) (2 : ℝ), ∀ y ∈ Icc (-1 : ℝ) (2 : ℝ), my_function x ≥ my_function y ∧ my_function x = 4 :=
by
  sorry

end max_value_of_function_in_interval_l505_505514


namespace least_positive_integer_with_12_factors_l505_505173

theorem least_positive_integer_with_12_factors :
  ∃ m : ℕ, 0 < m ∧ (∀ n : ℕ, 0 < n → (n ≠ m →  m ≤ n)) ∧ (12 = (m.factors.toFinset.card)) := sorry

end least_positive_integer_with_12_factors_l505_505173


namespace least_positive_integer_with_12_factors_l505_505290

def num_factors (n : ℕ) : ℕ := 
  (finset.range (n + 1)).filter (λ d, n % d = 0).card

theorem least_positive_integer_with_12_factors : 
  ∃ n : ℕ, num_factors n = 12 ∧ n = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_l505_505290


namespace least_pos_int_with_12_pos_factors_is_72_l505_505241

def least_positive_integer_with_12_factors (n : ℕ) : Prop :=
  ∃ (m : ℕ), m > 0 ∧ m ≠ n ∧ (∀ d : ℕ, d ∣ m → (d = n ∨ d = 1)) ∧
  ((∀ d : ℕ, d ∣ m → ∃ e : ℕ, e = 1) → n = 72)

theorem least_pos_int_with_12_pos_factors_is_72 (n : ℕ) :
  least_positive_integer_with_12_factors n → n = 72 := by
  sorry

end least_pos_int_with_12_pos_factors_is_72_l505_505241


namespace four_digit_multiples_of_five_count_l505_505667

theorem four_digit_multiples_of_five_count : 
  {n : ℕ | 1000 ≤ n ∧ n ≤ 9999 ∧ n % 5 = 0}.toFinset.card = 1800 :=
by
  sorry

end four_digit_multiples_of_five_count_l505_505667


namespace simplify_expression_l505_505783

variable (x y : ℝ)

theorem simplify_expression (h : x ≠ y ∧ x ≠ -y) : 
  ((1 / (x - y) - 1 / (x + y)) / (x * y / (x^2 - y^2)) = 2 / x) :=
by sorry

end simplify_expression_l505_505783


namespace intersection_points_l505_505833

-- Define the exponential function
def f (a x : ℝ) : ℝ := a^x

-- Define the logarithmic function
def g (a x : ℝ) : ℝ := log a x

-- Define the condition for the number of intersection points
theorem intersection_points (a : ℝ) (h : a > 1) : ℕ :=
if h1 : a = Real.exp (1 / Real.exp 1) then
  1
else if h2 : a > Real.exp (1 / Real.exp 1) then
  0
else if h3 : 1 < a ∧ a < Real.exp (1 / Real.exp 1) then
  2
else
  0

end intersection_points_l505_505833


namespace practice_other_days_l505_505468

-- Defining the total practice time for the week and the practice time for two days 
variable (total_minutes_week : ℤ) (total_minutes_two_days : ℤ)

-- Given conditions
axiom total_minutes_week_eq : total_minutes_week = 450
axiom total_minutes_two_days_eq : total_minutes_two_days = 172

-- The proof goal
theorem practice_other_days : (total_minutes_week - total_minutes_two_days) = 278 :=
by
  rw [total_minutes_week_eq, total_minutes_two_days_eq]
  show 450 - 172 = 278
  -- The proof goes here
  sorry

end practice_other_days_l505_505468


namespace root_interval_l505_505826

noncomputable def log2 (x : ℝ) := log x / log 2

def f (x : ℝ) : ℝ := log2 x + x - 4

theorem root_interval : ∃ c ∈ Ioo 2 3, f c = 0 :=
by
  have f_cont : ContinuousOn f (Icc 2 3) := sorry
  have f_2 : f 2 = -1 := sorry
  have f_3 : f 3 = log2 3 - 1 := sorry
  have f_sign_change : f 2 < 0 ∧ f 3 > 0 := sorry
  exact exists_root_Ioo f_cont f_2 f_3 f_sign_change

end root_interval_l505_505826


namespace radius_of_sphere_tangent_to_truncated_cone_l505_505458

-- Define the given conditions
def radius_bottom := 20 -- cm
def radius_top := 5 -- cm
def height_truncated_cone := 15 -- cm

noncomputable def sphere_radius := (15 * Real.sqrt 2) / 2 -- cm

-- The theorem to prove the radius of the sphere
theorem radius_of_sphere_tangent_to_truncated_cone :
  (∃ r : ℝ, r = sphere_radius) ∧
  (sphere_radius = (height_truncated_cone * Real.sqrt 2) / 2) :=
by 
  intros 
  use sphere_radius
  split
  . sorry
  . sorry

end radius_of_sphere_tangent_to_truncated_cone_l505_505458


namespace taxi_speed_is_60_l505_505453

theorem taxi_speed_is_60 (v_b v_t : ℝ) (h1 : v_b = v_t - 30) (h2 : 3 * v_t = 6 * v_b) : v_t = 60 := 
by 
  sorry

end taxi_speed_is_60_l505_505453


namespace least_positive_integer_with_12_factors_l505_505255

theorem least_positive_integer_with_12_factors :
  ∃ k : ℕ, (1 ≤ k) ∧ (nat.factors k).length = 12 ∧
    ∀ n : ℕ, (1 ≤ n) ∧ (nat.factors n).length = 12 → k ≤ n := 
begin
  sorry
end

end least_positive_integer_with_12_factors_l505_505255


namespace linear_function_passing_through_0_3_l505_505073

theorem linear_function_passing_through_0_3 (m : ℝ) : ∃ f : ℝ → ℝ, f(0) = 3 ∧ ∀ x, f(x) = m * x + 3 :=
by
  existsi (λ x => m * x + 3)
  split
  { simp }
  sorry

end linear_function_passing_through_0_3_l505_505073


namespace kids_wearing_shoes_l505_505846

-- Definitions based on the problem's conditions
def total_kids := 22
def kids_with_socks := 12
def kids_with_both := 6
def barefoot_kids := 8

-- Theorem statement
theorem kids_wearing_shoes :
  (∃ (kids_with_shoes : ℕ), 
     (kids_with_shoes = (total_kids - barefoot_kids) - (kids_with_socks - kids_with_both) + kids_with_both) ∧ 
     kids_with_shoes = 8) :=
by
  sorry

end kids_wearing_shoes_l505_505846


namespace f_f_is_even_l505_505038

-- Let f be a function from reals to reals
variables {f : ℝ → ℝ}

-- Given that f is an even function
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

-- Theorem to prove
theorem f_f_is_even (h : is_even f) : is_even (fun x => f (f x)) :=
by
  intros
  unfold is_even at *
  -- at this point, we assume the function f is even,
  -- follow from the assumption, we can prove the result
  sorry

end f_f_is_even_l505_505038


namespace isosceles_point_equidistant_from_vertices_l505_505776

variable {α : Type*} [MetricSpace α] [NormedSpace ℝ α]

structure Polygon (n : ℕ) :=
(vertices : Fin n → α)
(convex : ConvexHull ℝ (Set.range vertices))

theorem isosceles_point_equidistant_from_vertices {n : ℕ} (P : Polygon n) (O : α) :
  (∀ i j, i ≠ j → Dist O (P.vertices i) = Dist O (P.vertices j)) →
  (∀ i j k, i ≠ j → j ≠ k → k ≠ i → 
    (Dist (P.vertices i) O = Dist (P.vertices j) O ∧ 
     Dist (P.vertices j) O = Dist (P.vertices k) O ∧ 
     Dist (P.vertices k) O = Dist (P.vertices i) O)) →
  (∀ i, Dist O (P.vertices i) = Dist O (P.vertices 0)) :=
by
  intros h1 h2 i
  sorry

end isosceles_point_equidistant_from_vertices_l505_505776


namespace vertical_asymptote_x_value_l505_505990

theorem vertical_asymptote_x_value (x : ℝ) : 
  4 * x - 6 = 0 ↔ x = 3 / 2 :=
by
  sorry

end vertical_asymptote_x_value_l505_505990


namespace least_positive_integer_with_12_factors_is_972_l505_505359

theorem least_positive_integer_with_12_factors_is_972 : ∃ k : ℕ, (∀ n : ℕ, (∃ (d : ℕ), d * k = n) ↔ n = 12) ∧ k = 972 := sorry

end least_positive_integer_with_12_factors_is_972_l505_505359


namespace wider_can_radius_l505_505139

theorem wider_can_radius
  (h : ℝ)  -- the height of the wider can
  (r_narrow : ℝ)  -- the radius of the narrower can
  (r_wide : ℝ)  -- the radius of the wider can
  (V1 V2 : ℝ)  -- the volumes of the two cans
  (h_narrow : ℝ)  -- the height of the narrower can
  (H1 : r_narrow = 10)
  (H2 : h_narrow = 4 * h)
  (H3 : V1 = V2)
  (H4 : V1 = π * r_narrow^2 * h_narrow)
  (H5 : V2 = π * r_wide^2 * h) : 
  r_wide = 20 :=
begin
  sorry
end

end wider_can_radius_l505_505139


namespace quadratic_has_real_roots_find_value_of_m_l505_505594

theorem quadratic_has_real_roots (m : ℝ) : 
  ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (x1 - x2) ≠ 0 ∧ (x^2 - 4 * m * x + 3 * m^2 = 0) := 
by 
  sorry

theorem find_value_of_m (m : ℝ) (h1 : m > 0) (h2 : ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (x1 - x2 = 2)) :
  m = 1 := 
by 
  sorry

end quadratic_has_real_roots_find_value_of_m_l505_505594


namespace heads_between_4_and_6_times_heads_at_least_once_l505_505436

open ProbabilityTheory

/-- Theorem for the probability of getting heads between 4 and 6 times in 10 coin tosses --/
theorem heads_between_4_and_6_times {p : ℝ} (hp : p = 0.5) :
  ∑ k in finset.range 7, if k < 4 then 0 else nat.choose 10 k * (real.pow p k) * (real.pow (1 - p) (10 - k)) = 21 / 32 :=
by {
  sorry
}

/-- Theorem for the probability of getting heads at least once in 10 coin tosses --/
theorem heads_at_least_once {p : ℝ} (hp : p = 0.5) :
  1 - (real.pow (1 - p) 10) = 1023 / 1024 :=
by {
  sorry
}

end heads_between_4_and_6_times_heads_at_least_once_l505_505436


namespace books_and_exercise_books_count_l505_505850

noncomputable def number_of_students_first_scenario := 13
noncomputable def number_of_books := number_of_students_first_scenario + 2
noncomputable def number_of_exercise_books := 2 * number_of_students_first_scenario

theorem books_and_exercise_books_count :
  (∀ n, (books_distributed_1 n = 1) ∧ (exercise_books_distributed_1 n = 2 * n) ∧ (books_remaining_1 = 2)) →
  (∀ n, (books_distributed_2 n = 3 * n) ∧ (exercise_books_distributed_2 n = 5 * n + 1) ∧ (exercise_remaining_2 = 1)) →
  number_of_books = 15 ∧ number_of_exercise_books = 26 :=
by
  intro h1 h2
  sorry -- Proof is skipped as it is not required

end books_and_exercise_books_count_l505_505850


namespace count_four_digit_multiples_of_5_l505_505663

theorem count_four_digit_multiples_of_5 : 
  let first_4_digit := 1000
  let last_4_digit := 9999
  let first_multiple_of_5 := 1000
  let last_multiple_of_5 := 9995
  let total_multiples_of_5 := (1999 - 200 + 1)
  first_multiple_of_5 % 5 = 0 ∧ last_multiple_of_5 % 5 = 0 ∧ first_4_digit ≤ first_multiple_of_5 ∧ last_multiple_of_5 ≤ last_4_digit
  → total_multiples_of_5 = 1800 :=
by
  sorry

end count_four_digit_multiples_of_5_l505_505663


namespace solve_inequality_l505_505794

theorem solve_inequality (x : ℝ) (h₁ : x ≠ -1) (h₂ : x ≠ 0) :
  (2 * x / (x + 1) + (x - 3) / (3 * x) ≤ 4) ↔ (x ∈ (-∞, -1) ∪ (-1/2, ∞)) :=
sorry

end solve_inequality_l505_505794


namespace value_m2_3m_n_l505_505678

-- Define the quadratic equation
def quadratic (x : ℝ) : ℝ := x^2 + 2*x - 8

-- Define m and n as roots of the quadratic equation
def is_root (x : ℝ) : Prop := quadratic x = 0

-- Translate the conditions into Lean
variables {m n : ℝ}
hypothesis root_m : is_root m
hypothesis root_n : is_root n

-- Prove: m^2 + 3*m + n = 6
theorem value_m2_3m_n : m^2 + 3*m + n = 6 := 
sorry  -- Proof goes here

end value_m2_3m_n_l505_505678


namespace count_uphill_integers_divisible_by_eight_l505_505053

def is_uphill_integer (n : ℕ) : Prop :=
  let digits := n.digits 10
  list.pairwise (<) digits

def divisible_by_eight (n : ℕ) : Prop := n % 8 = 0

theorem count_uphill_integers_divisible_by_eight :
  (finset.filter divisible_by_eight ((finset.range 1000).filter is_uphill_integer)).card = 17 :=
sorry

end count_uphill_integers_divisible_by_eight_l505_505053


namespace four_digit_multiples_of_5_count_l505_505640

-- Define the range of four-digit numbers
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

-- Define a multiple of 5
def is_multiple_of_five (n : ℕ) : Prop := n % 5 = 0

-- Define the required proof problem
theorem four_digit_multiples_of_5_count : 
  ∃ n : ℕ, (n = 1800) ∧ (∀ k : ℕ, is_four_digit k ∧ is_multiple_of_five k → n = 1800) :=
sorry

end four_digit_multiples_of_5_count_l505_505640


namespace sum_of_digits_of_d_l505_505066

noncomputable def exchange_rate := (8:ℝ) / 5
noncomputable def spent_pesos := 72
noncomputable def remaining_pesos (d : ℝ) : ℝ :=
  exchange_rate * d - spent_pesos

theorem sum_of_digits_of_d (d : ℝ) (h : exchange_rate * d - spent_pesos = d) :
  d = 120 ∧ (1 + 2 + 0 = 3) :=
begin
  sorry
end

end sum_of_digits_of_d_l505_505066


namespace sum_of_diagonals_l505_505703

def valid_grid (grid : Matrix ℤ 6 6) : Prop :=
  ∀ i j : Fin 6,
    ∃ (numbers : Multiset ℤ) (blanks : ℕ),
      grid i j = numbers.sum ∧
      (numbers.card + blanks) = 6 ∧
      numbers = {2, 0, 1, 5} ∧
      ∀ d ∈ numbers, d ≠ 0 ∨ d = 0 ∧ numbers.card > 1 → ¬(d.head!.1 = 0)

theorem sum_of_diagonals (grid : Matrix ℤ 6 6) (h : valid_grid grid) :
    (∑ i : Fin 6, grid i i) + (∑ i : Fin 6, grid i (5 - i)) = 18 :=
sorry

end sum_of_diagonals_l505_505703


namespace tangent_line_exists_l505_505498

-- Define the circle and its properties
def circle_center : ℝ × ℝ := (1, 1)
def circle_radius : ℝ := 1

-- Define the conditions for the line to be tangent to the circle and pass through point (2,0)
def is_tangent (l : ℝ → ℝ) : Prop :=
  ∀ p : ℝ × ℝ, p = (1, 1) → abs (l 1 - p.2) / sqrt ((1 - 1) ^ 2 + (1 - (1 + 1/(l 0 - 2)*2 - 1))^2) = 1

def passes_through (l : ℝ → ℝ) : Prop := l 2 = 0

-- Statement of the problem
theorem tangent_line_exists :
  ∃ l : ℝ → ℝ, passes_through l ∧ is_tangent l → (l = λ x, 0) ∨ (l = λ x, x - 2) :=
sorry

end tangent_line_exists_l505_505498


namespace least_positive_integer_with_12_factors_is_72_l505_505218

open Nat

def hasExactly12Factors (n : ℕ) : Prop :=
  (divisors n).length = 12

theorem least_positive_integer_with_12_factors_is_72 :
  ∃ n : ℕ, hasExactly12Factors n ∧ ∀ m : ℕ, hasExactly12Factors m → n ≤ m → n = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_is_72_l505_505218


namespace abs_non_positive_eq_zero_l505_505407

theorem abs_non_positive_eq_zero (y : ℚ) (h : |4 * y - 7| ≤ 0) : y = 7 / 4 :=
by
  sorry

end abs_non_positive_eq_zero_l505_505407


namespace fruit_usage_in_pies_l505_505892

theorem fruit_usage_in_pies (initial_apples initial_peaches initial_pears initial_plums : ℕ)
  (remaining_apples : ℕ) :
  initial_apples = 40 →
  initial_peaches = 54 →
  initial_pears = 60 →
  initial_plums = 48 →
  remaining_apples = 39 →
  ∃ (apples_used peaches_used pears_used plums_used : ℕ), 
    apples_used = 1 ∧
    peaches_used = 2 ∧
    pears_used = 3 ∧
    plums_used = 4 :=
begin
  assume h0 h1 h2 h3 h4,
  use [1, 2, 3, 4],
  split, refl,
  split, refl,
  split, refl,
  refl,
end

end fruit_usage_in_pies_l505_505892


namespace least_positive_integer_with_12_factors_l505_505281

def num_factors (n : ℕ) : ℕ := 
  (finset.range (n + 1)).filter (λ d, n % d = 0).card

theorem least_positive_integer_with_12_factors : 
  ∃ n : ℕ, num_factors n = 12 ∧ n = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_l505_505281


namespace representative_arrangements_l505_505848

theorem representative_arrangements
  (boys girls : ℕ) 
  (subjects : ℕ) 
  (pe_rep : ∀ n, n = 1) 
  (eng_rep : ∀ n, n = 1) 
  : boys = 4 → girls = 3 → subjects = 7 → (4 * 3 * (Nat.factorial 5) = 1440) :=
by
  intros hb hg hs
  rw [hb, hg, hs]
  exact Nat.mul_assoc 4 3 (Nat.factorial 5) = 1440
  sorry

end representative_arrangements_l505_505848


namespace total_spokes_in_garage_l505_505928

-- Definitions based on the problem conditions
def num_bicycles : ℕ := 4
def spokes_per_wheel : ℕ := 10
def wheels_per_bicycle : ℕ := 2

-- The goal is to prove the total number of spokes
theorem total_spokes_in_garage : (num_bicycles * wheels_per_bicycle * spokes_per_wheel) = 80 :=
by
    sorry

end total_spokes_in_garage_l505_505928


namespace least_positive_integer_with_12_factors_l505_505159

theorem least_positive_integer_with_12_factors : ∃ n : ℕ, (nat.factors_count n = 12 ∧ ∀ m : ℕ, (nat.factors_count m = 12) → n ≤ m) ∧ n = 72 :=
begin
  sorry
end

end least_positive_integer_with_12_factors_l505_505159


namespace least_positive_integer_with_12_factors_is_72_l505_505322

-- Definition of having exactly 12 factors for a positive integer
def has_exactly_12_factors (n : ℕ) : Prop := n > 0 ∧ (finset.filter (λ d, n % d = 0) (finset.range (n + 1))).card = 12

-- Statement of the problem in Lean 4
theorem least_positive_integer_with_12_factors_is_72 :
  ∃ n : ℕ, has_exactly_12_factors n ∧ ∀ m : ℕ, has_exactly_12_factors m → n ≤ m :=
begin
  use 72,
  split,
  { -- Prove that 72 has exactly 12 factors
    sorry },
  { -- Prove that 72 is the least such integer
    sorry }
end

end least_positive_integer_with_12_factors_is_72_l505_505322


namespace sum_of_powers_of_neg2_l505_505506

theorem sum_of_powers_of_neg2 :
  ∑ n in Finset.range 21, (-2)^(n - 10) = 0 :=
by
  sorry

end sum_of_powers_of_neg2_l505_505506


namespace exists_sol_in_naturals_l505_505779

theorem exists_sol_in_naturals : ∃ (x y : ℕ), x^2 + y^2 = 61^3 := 
sorry

end exists_sol_in_naturals_l505_505779


namespace roots_nature_l505_505404

noncomputable def nature_of_roots (a b c : ℝ) : Prop :=
  let Δ := b^2 - 4*a*c in
  if Δ > 0 then 
    ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (a * x₁^2 + b * x₁ + c = 0) ∧ (a * x₂^2 + b * x₂ + c = 0)
  else if Δ = 0 then 
    ∃ x : ℝ, (a * x^2 + b * x + c = 0)
  else 
    ∃ x₁ x₂ : ℂ, x₁ ≠ x₂ ∧ (a * x₁^2 + b * x₁ + c = 0) ∧ (a * x₂^2 + b * x₂ + c = 0)

theorem roots_nature (a b c : ℝ) : nature_of_roots a b c :=
by
  sorry

end roots_nature_l505_505404


namespace perimeter_of_irregular_pentagonal_picture_frame_l505_505063

theorem perimeter_of_irregular_pentagonal_picture_frame 
  (base : ℕ) (left_side : ℕ) (right_side : ℕ) (top_left_diagonal_side : ℕ) (top_right_diagonal_side : ℕ)
  (h_base : base = 10) (h_left_side : left_side = 12) (h_right_side : right_side = 11)
  (h_top_left_diagonal_side : top_left_diagonal_side = 6) (h_top_right_diagonal_side : top_right_diagonal_side = 7) :
  base + left_side + right_side + top_left_diagonal_side + top_right_diagonal_side = 46 :=
by {
  sorry
}

end perimeter_of_irregular_pentagonal_picture_frame_l505_505063


namespace incenter_vector_sum_right_angle_l505_505045

theorem incenter_vector_sum_right_angle 
  (ABC : Type*) [metric_space ABC] [finite_dimensional ℝ ABC]
  (A B C I : ABC)
  (hI : ∃ I, I = incenter_triangle A B C)
  (hvec : 3 • (I -ᵥ A : ABC) + 4 • (I -ᵥ B) + 5 • (I -ᵥ C) = (0 : ABC)) :
  angle A B C = π / 2 := 
sorry

end incenter_vector_sum_right_angle_l505_505045


namespace modified_octagon_perimeter_l505_505944

variable (S : ℕ → ℝ)
hypothesis h1 : ∀ n, S n = if n < 5 then 3 else 3 / 2

theorem modified_octagon_perimeter : (S 0 + S 1 + S 2 + S 3 + S 4 + S 5 + S 6 + S 7) = 19.5 := by
  sorry

end modified_octagon_perimeter_l505_505944


namespace number_of_solutions_l505_505980

open Real

-- Define the system of equations
def system_equations (x y z : ℝ) : Prop :=
  (x + y + z = 3 * x * y) ∧ 
  (x^2 + y^2 + z^2 = 3 * x * z) ∧ 
  (x^3 + y^3 + z^3 = 3 * y * z)

-- Statement of the problem: Prove that the system has 4 sets of real solutions.
theorem number_of_solutions : 
  (∃ (sol : Finset (ℝ × ℝ × ℝ)), 
    (∀ t ∈ sol, let ⟨x, y, z⟩ := t in system_equations x y z) ∧ 
    sol.card = 4) := 
sorry

end number_of_solutions_l505_505980


namespace player_one_wins_if_bottom_right_is_blue_l505_505144

noncomputable def winning_strategy_for_player_one (n : ℕ) (initial_board : fin n → fin n → bool) : Prop :=
  initial_board (fin.last n) (fin.last n) = tt → 
  ∃ strategy : (fin n → fin n → bool) → (fin n → fin n → bool), 
  ∀ board, 
    (initial_board (fin.last n) (fin.last n) = tt → 
      ∃ k < n * n, 
        (∀ i j, i ≤ fin.last n ∧ j ≤ fin.last n → 
          (strategy board i j = initial_board i j)))
 
theorem player_one_wins_if_bottom_right_is_blue (n : ℕ) (initial_board : fin n → fin n → bool) 
  (h : initial_board (fin.last n) (fin.last n) = tt) : 
  winning_strategy_for_player_one n initial_board := 
sorry

end player_one_wins_if_bottom_right_is_blue_l505_505144


namespace pairs_count_l505_505952

theorem pairs_count : 
  let valid_pairs (a b : ℕ) := 1 ≤ b ∧ b < a ∧ a ≤ 200 ∧ 
    ∃ n : ℕ, n ≥ 1 ∧ (a = b * n^2) ∧ 
    (a + b) + (a - b) + a * b + a / b = (m : ℕ) ^ 2
  in (Finset.card ((Finset.filter (λ (p : ℕ × ℕ), valid_pairs p.1 p.2) 
    (Finset.Icc 1 200).product (Finset.Icc 1 200))) = 112)
:= sorry

end pairs_count_l505_505952


namespace least_positive_integer_with_12_factors_l505_505331

def has_exactly_12_factors (n : ℕ) : Prop :=
  n.factors.length = 12

theorem least_positive_integer_with_12_factors : ∃ k : ℕ, has_exactly_12_factors k ∧ (∀ n : ℕ, has_exactly_12_factors n → n ≥ k) ∧ k = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_l505_505331


namespace condition1_condition2_condition3_l505_505435

-- Condition 1 statement
theorem condition1: (number_of_ways_condition1 : ℕ) = 5520 := by
  -- Expected proof that number_of_ways_condition1 = 5520
  sorry

-- Condition 2 statement
theorem condition2: (number_of_ways_condition2 : ℕ) = 3360 := by
  -- Expected proof that number_of_ways_condition2 = 3360
  sorry

-- Condition 3 statement
theorem condition3: (number_of_ways_condition3 : ℕ) = 360 := by
  -- Expected proof that number_of_ways_condition3 = 360
  sorry

end condition1_condition2_condition3_l505_505435


namespace simplify_expression_l505_505933

theorem simplify_expression:
    (sqrt (12) * sqrt (2) / sqrt (3) - 2 * real.sin (real.pi / 4)) = 1 := by
  have h1: sqrt (12) = 2 * sqrt (3), by sorry
  have h2: real.sin (real.pi / 4) = sqrt (2) / 2, by sorry
  rw [h1, h2]
  sorry

end simplify_expression_l505_505933


namespace coefficients_sum_l505_505573

theorem coefficients_sum :
  let T := (x^2 - 2)^6
  in T = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + ⋯ + a_{12} * x^{12} → 
  a_3 + a_4 = 240 :=
sorry

end coefficients_sum_l505_505573


namespace interval_length_l505_505739

-- Define the conditions for the set of points and the line
def S (x y : ℤ) : Prop := 1 ≤ x ∧ x ≤ 20 ∧ 1 ≤ y ∧ y ≤ 20
def line_below (m : ℚ) (x y : ℤ) : Prop := y ≤ m * x

-- The proof problem
theorem interval_length (m : ℚ) :
  (∃ (m_min m_max : ℚ), (∀ x y, S x y → line_below m x y 
      ↔ 1 ≤ x ∧ x ≤ 20 ∧ 1 ≤ y ∧ y ≤ 20 → y ≤ m * x) 
      ∧ ((m_max - m_min).num = 1 ∧ (m_max - m_min).denom = 120)) 
      →  (let a := (m_max - m_min).num, b := (m_max - m_min).denom in a + b = 121) :=
sorry

end interval_length_l505_505739


namespace smallest_period_sin_pi_l505_505841

def smallest_positive_period (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

theorem smallest_period_sin_pi (T : ℝ) :
  (smallest_positive_period (λ x, Real.sin (Real.pi * x)) T) ↔ T = 2 :=
by
  -- This is where the proof steps would go, but we'll use 'sorry' to skip the proof
  sorry

end smallest_period_sin_pi_l505_505841


namespace total_weight_of_dumbbell_system_l505_505482

-- Definitions from the given conditions
def weight_pair1 : ℕ := 3
def weight_pair2 : ℕ := 5
def weight_pair3 : ℕ := 8

-- Goal: Prove that the total weight of the dumbbell system is 32 lbs
theorem total_weight_of_dumbbell_system :
  2 * weight_pair1 + 2 * weight_pair2 + 2 * weight_pair3 = 32 :=
by sorry

end total_weight_of_dumbbell_system_l505_505482


namespace second_term_is_three_l505_505925

-- Given conditions
variables (r : ℝ) (S : ℝ)
hypothesis hr : r = 1 / 4
hypothesis hS : S = 16

-- Definition of the first term a
noncomputable def first_term (r : ℝ) (S : ℝ) : ℝ :=
  S * (1 - r)

-- Definition of the second term
noncomputable def second_term (r : ℝ) (a : ℝ) : ℝ :=
  a * r

-- Prove that the second term is 3
theorem second_term_is_three : second_term r (first_term r S) = 3 :=
by
  rw [first_term, second_term]
  sorry

end second_term_is_three_l505_505925


namespace trajectory_of_E_max_area_and_line_equation_l505_505554

-- Definitions based on conditions
noncomputable def point_A : ℝ × ℝ := (-real.sqrt 3, 0)
noncomputable def circle_C (x y : ℝ) : Prop := (x - real.sqrt 3)^2 + y^2 = 16

-- The trajectory of the point E is an ellipse.
theorem trajectory_of_E : 
  ∀ (x y : ℝ), 
  (∀ (B : ℝ × ℝ), circle_C B.1 B.2 → false) →
  (∃ (E : ℝ × ℝ), (E.1/2)^2 + E.2^2 = 1) :=
sorry

-- Maximum area of triangle OPQ and equation of line l
noncomputable def line_l (k m : ℝ) (x y : ℝ) : Prop := y = k * x + m

theorem max_area_and_line_equation (k m : ℝ) (P Q : ℝ × ℝ) 
  (h1 : k ≠ 0) (h2 : m > 0) (h3 : line_l k m P.1 P.2) (h4 : line_l k m Q.1 Q.2)
  (h5 : (P.2 - Q.2 = (-1) / k)) : 
  (∃ S : ℝ, S = 1) ∧ (∀ (x y : ℝ), line_l (real.sqrt 2) (3 * real.sqrt 2 / 2) x y → true) :=
sorry

end trajectory_of_E_max_area_and_line_equation_l505_505554


namespace construct_circle_l505_505948

-- Define the given points and tangent lengths
variables (A B C : Point) (a b c : ℝ)

-- Define the main theorem statement
theorem construct_circle (exists_circle : ∃ (O : Point), 
  is_radical_axis A B a b ∧ 
  is_radical_axis B C b c ∧ 
  ∀ (P : Point), dist P O = dist O A ∨ dist O A = a ∨ 
                   dist O B = b ∨ dist O C = c) : 
  ∃ (O : Point), (O ∈ intersect (radical_axis A B a b) (radical_axis B C b c)) ∧
  (∀ (P : Point), dist P O = a ∨ dist P O = b ∨ dist P O = c) := sorry

end construct_circle_l505_505948


namespace sufficient_but_not_necessary_condition_l505_505552

theorem sufficient_but_not_necessary_condition (m : ℝ) :
  (∀ x > 0, (m^2 - m - 1) * x^(m - 1) > 0 → m = 2) →
  (|m - 2| < 1) :=
by
  sorry

end sufficient_but_not_necessary_condition_l505_505552


namespace least_positive_integer_with_12_factors_l505_505293

def has_exactly_12_factors (n : ℕ) : Prop :=
  (finset.range(n + 1).filter (λ d, n % d = 0)).card = 12

theorem least_positive_integer_with_12_factors : ∀ n : ℕ, has_exactly_12_factors n → n ≥ 150 :=
sorry

end least_positive_integer_with_12_factors_l505_505293


namespace count_four_digit_multiples_of_5_l505_505630

def is_multiple_of_5 (n : ℕ) : Prop :=
  n % 5 = 0

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def four_digit_multiples_of_5_count (lower upper : ℕ) :=
  (upper - lower + 1)

theorem count_four_digit_multiples_of_5 : 
  four_digit_multiples_of_5_count 200 1999 = 1800 :=
by
  sorry

end count_four_digit_multiples_of_5_l505_505630


namespace least_pos_int_with_12_pos_factors_is_72_l505_505242

def least_positive_integer_with_12_factors (n : ℕ) : Prop :=
  ∃ (m : ℕ), m > 0 ∧ m ≠ n ∧ (∀ d : ℕ, d ∣ m → (d = n ∨ d = 1)) ∧
  ((∀ d : ℕ, d ∣ m → ∃ e : ℕ, e = 1) → n = 72)

theorem least_pos_int_with_12_pos_factors_is_72 (n : ℕ) :
  least_positive_integer_with_12_factors n → n = 72 := by
  sorry

end least_pos_int_with_12_pos_factors_is_72_l505_505242


namespace least_positive_integer_with_12_factors_is_72_l505_505214

open Nat

def hasExactly12Factors (n : ℕ) : Prop :=
  (divisors n).length = 12

theorem least_positive_integer_with_12_factors_is_72 :
  ∃ n : ℕ, hasExactly12Factors n ∧ ∀ m : ℕ, hasExactly12Factors m → n ≤ m → n = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_is_72_l505_505214


namespace percentage_error_in_area_l505_505425

theorem percentage_error_in_area (s : ℝ) (h : s > 0) :
  let s' := 1.02 * s
  let A := s ^ 2
  let A' := s' ^ 2
  let error := A' - A
  let percent_error := (error / A) * 100
  percent_error = 4.04 := by
  sorry

end percentage_error_in_area_l505_505425


namespace least_pos_int_with_12_pos_factors_is_72_l505_505240

def least_positive_integer_with_12_factors (n : ℕ) : Prop :=
  ∃ (m : ℕ), m > 0 ∧ m ≠ n ∧ (∀ d : ℕ, d ∣ m → (d = n ∨ d = 1)) ∧
  ((∀ d : ℕ, d ∣ m → ∃ e : ℕ, e = 1) → n = 72)

theorem least_pos_int_with_12_pos_factors_is_72 (n : ℕ) :
  least_positive_integer_with_12_factors n → n = 72 := by
  sorry

end least_pos_int_with_12_pos_factors_is_72_l505_505240


namespace count_four_digit_multiples_of_5_l505_505658

theorem count_four_digit_multiples_of_5 : 
  let first_4_digit := 1000
  let last_4_digit := 9999
  let first_multiple_of_5 := 1000
  let last_multiple_of_5 := 9995
  let total_multiples_of_5 := (1999 - 200 + 1)
  first_multiple_of_5 % 5 = 0 ∧ last_multiple_of_5 % 5 = 0 ∧ first_4_digit ≤ first_multiple_of_5 ∧ last_multiple_of_5 ≤ last_4_digit
  → total_multiples_of_5 = 1800 :=
by
  sorry

end count_four_digit_multiples_of_5_l505_505658


namespace multiples_of_5_in_4_digit_range_l505_505654

theorem multiples_of_5_in_4_digit_range : 
  let count_multiples := (9995 - 1000) / 5 + 1
  in count_multiples = 1800 :=
by
  sorry

end multiples_of_5_in_4_digit_range_l505_505654


namespace inequality_proof_l505_505744

open Real -- This gives us access to Real number operations and properties directly

noncomputable def a := 2 ^ (1 / 2)
noncomputable def b := 3 ^ (1 / 3)
noncomputable def c := log 3 2

theorem inequality_proof : c < a ∧ a < b :=
by
  -- Conditions
  have h1 : a = 2 ^ (1 / 2), by rfl
  have h2 : b = 3 ^ (1 / 3), by rfl
  have h3 : c = log 3 2, by rfl

  -- Evaluations following solution steps
  have ha_gt_1 : 1 < a, from Real.sqrt_lt (by norm_num) (by norm_num : 0 < 2),
  
  -- Comparisons
  suffices h4 : c < a ∧ a < b, from h4, 
  
  -- Calculate bounds for c
  have h_c_lt_1 : c < 1, 
    from log_lt_log (by norm_num) (by norm_num : 1 < 2) (by norm_num),

  -- Concluding proof using calculated information
  exact ⟨h_c_lt_1, ha_gt_1⟩

-- Proof is intentionally incomplete for illustrative purposes:
sorry -- This indicates skipping the comprehensive proof details.


end inequality_proof_l505_505744


namespace additional_pots_last_hour_l505_505412

theorem additional_pots_last_hour (h1 : 60 / 6 = 10) (h2 : 60 / 5 = 12) : 12 - 10 = 2 :=
by
  sorry

end additional_pots_last_hour_l505_505412


namespace hypotenuse_length_l505_505426

def is_isosceles_right_triangle (a c : ℝ) : Prop :=
  c = real.sqrt 2 * a

def perimeter (a c : ℝ) : ℝ :=
  2 * a + c

theorem hypotenuse_length (a c : ℝ) (h1 : perimeter a c = 4 + 4 * real.sqrt 2) (h2 : is_isosceles_right_triangle a c) : 
  c = 4 := by
  sorry

end hypotenuse_length_l505_505426


namespace proof_problem_l505_505763

def A : Set ℝ := {x | x < 4}
def B : Set ℝ := {x | x^2 - 4 * x + 3 > 0}

theorem proof_problem : {x | x ∈ A ∧ x ∉ (A ∩ B)} = {x | 1 ≤ x ∧ x ≤ 3} :=
by {
  sorry
}

end proof_problem_l505_505763


namespace count_four_digit_multiples_of_5_l505_505657

theorem count_four_digit_multiples_of_5 : 
  let first_4_digit := 1000
  let last_4_digit := 9999
  let first_multiple_of_5 := 1000
  let last_multiple_of_5 := 9995
  let total_multiples_of_5 := (1999 - 200 + 1)
  first_multiple_of_5 % 5 = 0 ∧ last_multiple_of_5 % 5 = 0 ∧ first_4_digit ≤ first_multiple_of_5 ∧ last_multiple_of_5 ≤ last_4_digit
  → total_multiples_of_5 = 1800 :=
by
  sorry

end count_four_digit_multiples_of_5_l505_505657


namespace sum_of_m_for_common_root_l505_505405

theorem sum_of_m_for_common_root :
  let p1 := λ x : ℝ, x^2 - 4 * x + 3
  let p2 := λ x (m : ℝ), x^2 - 6 * x + m
  let r1 := {x | p1 x = 0}
  let m1 := {m | ∃ x ∈ r1, p2 x m = 0}
  (m1.sum) = 14 := sorry

end sum_of_m_for_common_root_l505_505405


namespace part_I_part_II_l505_505997

theorem part_I (a b m : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a^2 + b^2 = 9/2) (h4 : a + b ≤ m) : m ≥ 3 := by
  sorry

theorem part_II (a b x : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a^2 + b^2 = 9/2)
  (h4 : 2 * |x - 1| + |x| ≥ a + b) : (x ≤ -1 / 3 ∨ x ≥ 5 / 3) := by
  sorry

end part_I_part_II_l505_505997


namespace four_digit_multiples_of_5_l505_505649

theorem four_digit_multiples_of_5 : 
  let f := fun n => 1000 <= n ∧ n <= 9999 ∧ n % 5 = 0
  in finset.card (finset.filter f (finset.range 10000)) = 1800 :=
by
  sorry

end four_digit_multiples_of_5_l505_505649


namespace area_of_rectangle_at_stage_8_l505_505689

-- Define the conditions given in the problem
def square_side_length : ℝ := 4
def square_area : ℝ := square_side_length * square_side_length
def stages : ℕ := 8

-- Define the statement to be proved
theorem area_of_rectangle_at_stage_8 : (stages * square_area) = 128 := 
by 
  have h1 : square_area = 16 := by
    unfold square_area
    norm_num
  have h2 : (stages * square_area) = 8 * 16 := by
    unfold stages
    rw h1
  rw h2
  norm_num
  sorry

end area_of_rectangle_at_stage_8_l505_505689


namespace least_positive_integer_with_12_factors_l505_505249

theorem least_positive_integer_with_12_factors :
  ∃ k : ℕ, (1 ≤ k) ∧ (nat.factors k).length = 12 ∧
    ∀ n : ℕ, (1 ≤ n) ∧ (nat.factors n).length = 12 → k ≤ n := 
begin
  sorry
end

end least_positive_integer_with_12_factors_l505_505249


namespace always_two_real_roots_find_m_l505_505599

-- Given quadratic equation: x^2 - 4mx + 3m^2 = 0
-- Definitions for the problem
def quadratic_eq (m x : ℝ) : Prop := x^2 - 4 * m * x + 3 * m^2 = 0

-- Q1: Prove that this equation always has two real roots.
theorem always_two_real_roots (m : ℝ) : ∃ x₁ x₂ : ℝ, quadratic_eq m x₁ ∧ quadratic_eq m x₂ :=
by
  sorry

-- Q2: If m > 0 and the difference between the two real roots is 2, find the value of m.
theorem find_m (m : ℝ) (h₁ : m > 0) (h₂ : ∃ x₁ x₂ : ℝ, quadratic_eq m x₁ ∧ quadratic_eq m x₂ ∧ |x₁ - x₂| = 2) : m = 1 :=
by
  sorry

end always_two_real_roots_find_m_l505_505599


namespace triangles_perpendicular_concurrency_l505_505610

theorem triangles_perpendicular_concurrency 
  (ABC A'B'C' : Triangle) 
  (H : ∃ M', lines_through_pts_perpendicular_intersect A'B'C' ABC M') :
  ∃ M, lines_through_pts_perpendicular_intersect ABC A'B'C' M :=
sorry

end triangles_perpendicular_concurrency_l505_505610


namespace movies_left_to_watch_l505_505844

theorem movies_left_to_watch (total_movies watched_movies : Nat) (h_total : total_movies = 12) (h_watched : watched_movies = 6) : total_movies - watched_movies = 6 :=
by
  sorry

end movies_left_to_watch_l505_505844


namespace four_digit_div_by_14_l505_505529

theorem four_digit_div_by_14 (n : ℕ) (h₁ : 9450 + n < 10000) :
  (∃ k : ℕ, 9450 + n = 14 * k) ↔ (n = 8) := by
  sorry

end four_digit_div_by_14_l505_505529


namespace least_positive_integer_with_12_factors_is_72_l505_505323

-- Definition of having exactly 12 factors for a positive integer
def has_exactly_12_factors (n : ℕ) : Prop := n > 0 ∧ (finset.filter (λ d, n % d = 0) (finset.range (n + 1))).card = 12

-- Statement of the problem in Lean 4
theorem least_positive_integer_with_12_factors_is_72 :
  ∃ n : ℕ, has_exactly_12_factors n ∧ ∀ m : ℕ, has_exactly_12_factors m → n ≤ m :=
begin
  use 72,
  split,
  { -- Prove that 72 has exactly 12 factors
    sorry },
  { -- Prove that 72 is the least such integer
    sorry }
end

end least_positive_integer_with_12_factors_is_72_l505_505323


namespace largest_possible_s_l505_505042

noncomputable def max_value_of_s (p q r s : ℝ) (h1 : p + q + r + s = 8) (h2 : pq + pr + ps + qr + qs + rs = 12) : ℝ :=
  2 + 3 * Real.sqrt 2

theorem largest_possible_s (p q r s : ℝ) (h1 : p + q + r + s = 8) (h2 : pq + pr + ps + qr + qs + rs = 12) :
  s ≤ max_value_of_s p q r s h1 h2 := 
sorry

end largest_possible_s_l505_505042


namespace intersection_point_exists_l505_505489

-- Define the first line parameterization
def line1 (t : ℝ) : ℝ × ℝ :=
  (1 + 2 * t, 2 - 3 * t)

-- Define the second line parameterization
def line2 (u : ℝ) : ℝ × ℝ :=
  (4 + u, 5 - u)

-- The theorem statement asserting the intersection point of the two lines
theorem intersection_point_exists :
  ∃ (t u : ℝ), line1 t = line2 u ∧ line1 t = (-11, 20) :=
by {
  use (-6),
  use (-15),
  simp [line1, line2],
  split;
  simp  -- Split both equalities and simplify
  sorry  -- Completing the proof is skipped.
}

end intersection_point_exists_l505_505489


namespace range_of_a_l505_505493

def tensor (x y : ℝ) : ℝ := x * (1 - y)

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → tensor (x - a) (x + a) < 2) → -1 < a ∧ a < 2 := by
  sorry

end range_of_a_l505_505493


namespace least_positive_integer_with_12_factors_l505_505299

def has_exactly_12_factors (n : ℕ) : Prop :=
  (finset.range(n + 1).filter (λ d, n % d = 0)).card = 12

theorem least_positive_integer_with_12_factors : ∀ n : ℕ, has_exactly_12_factors n → n ≥ 150 :=
sorry

end least_positive_integer_with_12_factors_l505_505299


namespace area_region_inside_circle_outside_triangle_l505_505434

-- Define the problem conditions
def radius_circle := 3
def side_triangle := 6
def area_circle := π * radius_circle ^ 2
def area_triangle := (sqrt 3 / 4) * side_triangle ^ 2

-- Proof statement
theorem area_region_inside_circle_outside_triangle :
  (area_circle - area_triangle) = 9 * (π - sqrt 3) := 
  sorry

end area_region_inside_circle_outside_triangle_l505_505434


namespace distinct_row_col_sums_l505_505993

theorem distinct_row_col_sums (n : ℕ) (h : n = 1 ∨ n = 2 ∨ n = 4) :
  ∃ (A : matrix (fin 4) (fin 4) ℕ),
  (∀ i : fin 4, distinct ((λ j, matrix (fin 4) (fin 4) ℕ A i j).sum_row)) ∧
  (∀ j : fin 4, distinct ((λ i, matrix (fin 4) (fin 4) ℕ A i j).sum_col)) ∧
  (∀ i : fin 4, (λ j, matrix (fin 4) (fin 4) ℕ A i j).sum_row % n = 0) ∧
  (∀ j : fin 4, (λ i, matrix (fin 4) (fin 4) ℕ A i j).sum_col % n = 0)
:= sorry

end distinct_row_col_sums_l505_505993


namespace find_primes_adding_3_eq_multiple_4_l505_505920

open Nat

def is_prime (n : Nat) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def primes_less_than_30 := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}
def primes_become_multiple_of_4 := {p | p ∈ primes_less_than_30 ∧ (p + 3) % 4 = 0}

theorem find_primes_adding_3_eq_multiple_4 :
  primes_become_multiple_of_4 = {5, 13, 17, 29} :=
by
  sorry

end find_primes_adding_3_eq_multiple_4_l505_505920


namespace least_positive_integer_with_12_factors_is_96_l505_505275

theorem least_positive_integer_with_12_factors_is_96 :
  ∃ n : ℕ, (n = 96 ∧ (∀ m : ℕ, (m > 0 ∧ ∀ d : ℕ, d ∣ m → d > 0 → d < m → (∃ k : ℕ, d = k ∨ k * d = m) → k > 1 → m > n))) ∧ (∃ m : ℕ, m = 12 ∧ (∀ d : ℕ, d ∣ m → d ∈ {1, m} ∨ ∀ k : ℕ, k * d = m → d = k ∨ d < k)).
sorry

end least_positive_integer_with_12_factors_is_96_l505_505275


namespace sum_tens_units_digit_8_pow_2003_l505_505406

theorem sum_tens_units_digit_8_pow_2003 :
  let n := 8^2003 in
  ((n % 100) / 10 + (n % 10)) = 5 :=
by
  sorry

end sum_tens_units_digit_8_pow_2003_l505_505406


namespace parallelogram_sides_l505_505719

variables (A B C K L M : Type) [metric_space A] [metric_space B] [metric_space C]
variables [metric_space K] [metric_space L] [metric_space M]

-- Definitions of points and variables
variables (AB BC : ℝ)
variables (S : ℝ)
variables (x y : ℝ)

-- Conditions
def triangle_ABC (A B C : Type) := metric_space.distance A B = 18 ∧ metric_space.distance B C = 12
def parallelogram_area (S : ℝ) := S = 4 / 9 * 1 / 2 * 18 * 12
def xy_product := x * y = 48
def xy_sum := 3 * x + 2 * y = 36

-- Proposition in Lean
theorem parallelogram_sides (A B C K L M : Type) [metric_space A] [metric_space B] [metric_space C]
  [metric_space K] [metric_space L] [metric_space M] 
  (hABC : triangle_ABC A B C) (hArea : parallelogram_area S) (hxy_prod : xy_product x y) (hxy_sum : xy_sum x y) :
  (x = 8 ∧ y = 6) ∨ (x = 4 ∧ y = 12) :=
sorry

end parallelogram_sides_l505_505719


namespace octagon_area_ratio_l505_505032

theorem octagon_area_ratio (m n: ℕ) (rel_prime: Nat.coprime m n) :
  let ABCDEFGH : Type := {A B C D E F G H : Point | regular_octagon A B C D E F G H}
  be the regular octagon with sides where I J K L M N O P are midpoints of sides:
  let smaller_octagon : Type :=
    {Q R S T U V W X : Point | regular_octagon Q R S T U V W X} where vertices meet intersections of segments:
  (∀ (AI BJ CK DL EM FN GO HP : Segment),
    intersects AI BJ ∧ intersects CK DL ∧ intersects EM FN ∧ intersects GO HP) →
  area_ratio(smaller_octagon, ABCDEFGH) = (m, n) →
  m = 1 ∧ n = 4 →
  m + n = 5 :=
begin
  intros,
  sorry
end

end octagon_area_ratio_l505_505032


namespace sum_sequence_l505_505541

theorem sum_sequence (a : Fin 101 → ℝ) 
    (h1 : a 1 + a 2 = 1)
    (h2 : a 2 + a 3 = 2)
    (h3 : a 3 + a 4 = 3)
    -- ... Similar conditions for a_4 + a_5 = 4 to a_99 + a_100 = 99
    (h98 : a 98 + a 99 = 98)
    (h99 : a 99 + a 100 = 99)
    (h100 : a 100 + a 1 = 100) : 
    (a 1 + a 2 + a 3 + ... + a 100) = 2525 :=
sorry

end sum_sequence_l505_505541


namespace least_positive_integer_with_12_factors_l505_505157

theorem least_positive_integer_with_12_factors : ∃ n : ℕ, (nat.factors_count n = 12 ∧ ∀ m : ℕ, (nat.factors_count m = 12) → n ≤ m) ∧ n = 72 :=
begin
  sorry
end

end least_positive_integer_with_12_factors_l505_505157


namespace exists_formula_always_zero_no_formula_always_one_l505_505487

-- Part (a): Prove that there exists a formula that always evaluates to 0.
theorem exists_formula_always_zero {a : ℤ} :
  ∃ (perp bowtie : ℤ → ℤ → ℤ), (∀ a, perp = (λ x y, x - y) ∨ perp = (λ x y, x * y)) ∧ 
  (∀ a, bowtie = (λ x y, x - y) ∨ bowtie = (λ x y, x * y)) ∧ 
  ((perp a a) = 0 ∨ (bowtie a a) = 0) → (perp a a) bowtie (perp a a) = 0 :=
by
  sorry

-- Part (b): Prove that there does not exist a formula that always evaluates to 1.
theorem no_formula_always_one {a b : ℤ} :
  ¬ ∃ (perp bowtie : ℤ → ℤ → ℤ), (∀ a, perp = (λ x y, x - y) ∨ perp = (λ x y, x * y)) ∧
  (∀ a, bowtie = (λ x y, x - y) ∨ bowtie = (λ x y, x * y)) ∧
  ((perp a a) = 1 ∨ (bowtie a a) = 1) :=
by
  sorry

end exists_formula_always_zero_no_formula_always_one_l505_505487


namespace exist_side_not_shorter_than_sqrt_a2_b2_l505_505691

noncomputable def convex_quadrilateral_side_length_condition (a b : ℝ) : Prop :=
  ∀ (ABCD : Type) [convex_quadrilateral ABCD] 
    (AC BD : ℝ),
    (AC = 2 * a) → (BD = 2 * b) →
    ∃ (side_length : ℝ), side_length ≥ real.sqrt (a^2 + b^2)

theorem exist_side_not_shorter_than_sqrt_a2_b2 (2a 2b : ℝ) :
  convex_quadrilateral_side_length_condition a b :=
sorry

end exist_side_not_shorter_than_sqrt_a2_b2_l505_505691


namespace lana_extra_flowers_l505_505732

theorem lana_extra_flowers :
  ∀ (tulips roses used total_extra : ℕ),
    tulips = 36 →
    roses = 37 →
    used = 70 →
    total_extra = (tulips + roses - used) →
    total_extra = 3 :=
by
  intros tulips roses used total_extra ht hr hu hte
  rw [ht, hr, hu] at hte
  sorry

end lana_extra_flowers_l505_505732


namespace median_divides_triangle_l505_505780

-- Defining the structure and properties of a triangle
structure Triangle (α : Type*) :=
(A B C : α)

-- Defining points and lines
structure Point (α : Type*) :=
(x y : α)

structure LineSegment (α : Type*) :=
(start end : Point α)

structure MidPoint (α : Type*) := 
(m : Point α)
(prop : ∀ (a b : Point α), m.x = (a.x + b.x) / 2 ∧ m.y = (a.y + b.y) / 2)

structure Height (α : Type*) :=
(h : α)

-- Given conditions
variables {α : Type*}

variables (A B C : Point α)
variable M : Point α
variable CM : LineSegment α
variable CH : Height α

noncomputable def area (t : Triangle α) [has_div α] [has_mul α] [has_one α] : α :=
  1 / 2 * (t.B.x - t.A.x) * (t.C.y - t.A.y) - (t.C.x - t.A.x) * (t.B.y - t.A.y)

theorem median_divides_triangle (hM : MidPoint (α) (M) (A) (B))
  (hEqualSegs : M.x - A.x = B.x - M.x ∧ M.y - A.y = B.y - M.y) 
  (hHeight : CH)
  (t : Triangle α)
  : area (Triangle.mk A C M) = area (Triangle.mk B C M) :=
sorry

end median_divides_triangle_l505_505780


namespace sphere_inscribed_in_cone_l505_505449

theorem sphere_inscribed_in_cone :
  ∃ (a c : ℝ), 
  (∀ (r : ℝ), 
    (let coneHeight := 20 in
     let coneRadius := 10 in
     let CD := coneRadius in
     let AC := Real.sqrt (coneRadius^2 + coneHeight^2) in
     let AO := coneHeight - r in
     let ratio := r / AO in
     10 * r * AC = 10 * coneHeight - 10 * r) →
    r = a * Real.sqrt c - a) ∧ a + c = 55 :=
by
  sorry

end sphere_inscribed_in_cone_l505_505449


namespace consumption_increase_percentage_l505_505118

theorem consumption_increase_percentage
  (T C : ℝ)
  (H1 : 0.90 * (1 + X/100) = 0.9999999999999858) :
  X = 11.11111111110953 :=
by
  sorry

end consumption_increase_percentage_l505_505118


namespace least_positive_integer_with_12_factors_is_972_l505_505367

theorem least_positive_integer_with_12_factors_is_972 : ∃ k : ℕ, (∀ n : ℕ, (∃ (d : ℕ), d * k = n) ↔ n = 12) ∧ k = 972 := sorry

end least_positive_integer_with_12_factors_is_972_l505_505367


namespace problem_solution_l505_505784

theorem problem_solution (n : Real) (h : 0.04 * n + 0.1 * (30 + n) = 15.2) : n = 89.09 := 
sorry

end problem_solution_l505_505784


namespace multinomial_coefficient_range_integer_l505_505989

theorem multinomial_coefficient_range_integer :
  (finset.Icc 1 40).card = 40 :=
by
  sorry

end multinomial_coefficient_range_integer_l505_505989


namespace find_twentieth_special_number_l505_505976

theorem find_twentieth_special_number :
  ∃ n : ℕ, (n ≡ 2 [MOD 3]) ∧ (n ≡ 5 [MOD 8]) ∧ (∀ k < 20, ∃ m : ℕ, (m ≡ 2 [MOD 3]) ∧ (m ≡ 5 [MOD 8]) ∧ m < n) ∧ (n = 461) := 
sorry

end find_twentieth_special_number_l505_505976


namespace original_price_eq_2000_l505_505424

-- Define the conditions in Lean
variable (P : ℝ)
variable h1 : 0.7 * P * 0.8 = 1120

-- State the theorem to solve for P
theorem original_price_eq_2000 : P = 2000 :=
by
  have h2 : 0.56 * P = 1120 := by
    calc
      0.56 * P = 0.7 * P * 0.8 : by rw [mul_assoc]
            ... = 1120         : by exact h1
  sorry

end original_price_eq_2000_l505_505424


namespace problem_proof_l505_505006

-- Given Definitions
def seq (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ (n : ℕ), n > 0 → a (n + 1) = a n / (2 * a n + 1)

def geo_seq (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 1 ∧ (a 2 = a 1 * r) ∧ (a 5 = a 2 * r ^ 3)

def arith_seq (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ (n : ℕ), n > 0 → b (n + 1) = b n + d

def b_seq (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  ∀ n, n > 0 → b n = a n * a (n + 1)

def S_n (a : ℕ → ℝ) : ℕ → ℝ :=
  λ n, (1/2) * (1 - 1/(2 * n + 1))

-- Proof Statement
theorem problem_proof :
    ∃ (a : ℕ → ℝ), seq a ∧ geo_seq a ∧ 
    (∃ (b : ℕ → ℝ), arith_seq (λ n, 1 / a n) ∧ 
                     (∀ n, n > 0 → b n = a n * a (n + 1)) ∧ 
                     (∀ n, n > 0 → (S_n a) n = (1/2) * (1 - 1/(2 * n + 1)))) :=
by 
  sorry

end problem_proof_l505_505006


namespace minutes_practiced_other_days_l505_505461

theorem minutes_practiced_other_days (total_hours : ℕ) (minutes_per_day : ℕ) (num_days : ℕ) :
  total_hours = 450 ∧ minutes_per_day = 86 ∧ num_days = 2 → (total_hours - num_days * minutes_per_day) = 278 := by
  sorry

end minutes_practiced_other_days_l505_505461


namespace more_time_running_than_skipping_l505_505877

def time_running : ℚ := 17 / 20
def time_skipping_rope : ℚ := 83 / 100

theorem more_time_running_than_skipping :
  time_running > time_skipping_rope :=
by
  -- sorry skips the proof
  sorry

end more_time_running_than_skipping_l505_505877


namespace least_positive_integer_with_12_factors_is_72_l505_505223

open Nat

def hasExactly12Factors (n : ℕ) : Prop :=
  (divisors n).length = 12

theorem least_positive_integer_with_12_factors_is_72 :
  ∃ n : ℕ, hasExactly12Factors n ∧ ∀ m : ℕ, hasExactly12Factors m → n ≤ m → n = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_is_72_l505_505223


namespace election_winner_votes_l505_505884

theorem election_winner_votes (V : ℝ) : (0.62 * V = 806) → (0.62 * V) - (0.38 * V) = 312 → 0.62 * V = 806 :=
by
  intro hWin hDiff
  exact hWin

end election_winner_votes_l505_505884


namespace least_positive_integer_with_12_factors_l505_505279

def num_factors (n : ℕ) : ℕ := 
  (finset.range (n + 1)).filter (λ d, n % d = 0).card

theorem least_positive_integer_with_12_factors : 
  ∃ n : ℕ, num_factors n = 12 ∧ n = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_l505_505279


namespace total_fat_served_l505_505900

theorem total_fat_served (fat_herring fat_eel fat_pike : ℕ) (num_fish_each_type : ℕ)
  (h_herring : fat_herring = 40)
  (h_eel : fat_eel = 20)
  (h_pike : fat_pike = fat_eel + 10)
  (h_num_fish : num_fish_each_type = 40) :
  40 * fat_herring + 40 * fat_eel + 40 * fat_pike = 3600 :=
by
  -- Given the values, prove the total fat served is 3600 oz.
  rw [h_herring, h_eel, h_pike, h_num_fish]
  sorry

end total_fat_served_l505_505900


namespace problem1_problem2_l505_505934

open Real

-- Proof problem for the first expression
theorem problem1 : 
  (-2^2 * (1 / 4) + 4 / (4/9) + (-1) ^ 2023 = 7) :=
by 
  sorry

-- Proof problem for the second expression
theorem problem2 : 
  (-1 ^ 4 + abs (2 - (-3)^2) + (1/2) / (-3/2) = 17/3) :=
by 
  sorry

end problem1_problem2_l505_505934


namespace cube_edge_length_l505_505133

theorem cube_edge_length (total_edge_length : ℕ) (num_edges : ℕ) : total_edge_length = 72 → num_edges = 12 → total_edge_length / num_edges = 6 :=
by
  assume h1 : total_edge_length = 72,
  assume h2 : num_edges = 12,
  sorry

end cube_edge_length_l505_505133


namespace least_positive_integer_with_12_factors_l505_505376

theorem least_positive_integer_with_12_factors : 
  ∃ (n : ℕ), (n > 0) ∧ (nat.factors_count n = 12) ∧ (∀ m, (m > 0) → (nat.factors_count m = 12) → n ≤ m) ∧ n = 108 :=
sorry

end least_positive_integer_with_12_factors_l505_505376


namespace least_positive_integer_with_12_factors_is_96_l505_505271

theorem least_positive_integer_with_12_factors_is_96 :
  ∃ n : ℕ, (n = 96 ∧ (∀ m : ℕ, (m > 0 ∧ ∀ d : ℕ, d ∣ m → d > 0 → d < m → (∃ k : ℕ, d = k ∨ k * d = m) → k > 1 → m > n))) ∧ (∃ m : ℕ, m = 12 ∧ (∀ d : ℕ, d ∣ m → d ∈ {1, m} ∨ ∀ k : ℕ, k * d = m → d = k ∨ d < k)).
sorry

end least_positive_integer_with_12_factors_is_96_l505_505271


namespace smallest_total_area_l505_505061

open Real Geometry

/-- Given 12 match sticks of length 1, we want to find the smallest total area of 
    regular polygons that can be formed without overlapping or crossing matches. -/
theorem smallest_total_area (matches : ℕ) (length : ℝ) (h1 : matches = 12) (h2 : length = 1) :
  ∃ A, (∀ (P : RegularPolygon) (n : ℕ), 
      (3 ≤ n) → (length = side_length P) → (matches = n * P.num_sides) → P.area = A) ∧ A = sqrt 3 := by
  sorry

end smallest_total_area_l505_505061


namespace least_positive_integer_with_12_factors_l505_505167

theorem least_positive_integer_with_12_factors :
  ∃ m : ℕ, 0 < m ∧ (∀ n : ℕ, 0 < n → (n ≠ m →  m ≤ n)) ∧ (12 = (m.factors.toFinset.card)) := sorry

end least_positive_integer_with_12_factors_l505_505167


namespace christine_stickers_l505_505939

-- Define the conditions from the problem
def total_stickers : ℕ := 30
def needed_stickers : ℕ := 19
def current_stickers : ℕ := total_stickers - needed_stickers

-- Prove the statement
theorem christine_stickers : current_stickers = 11 :=
by
  unfold current_stickers
  rw [total_stickers, needed_stickers]
  norm_num
  sorry

end christine_stickers_l505_505939


namespace least_positive_integer_with_12_factors_l505_505278

def num_factors (n : ℕ) : ℕ := 
  (finset.range (n + 1)).filter (λ d, n % d = 0).card

theorem least_positive_integer_with_12_factors : 
  ∃ n : ℕ, num_factors n = 12 ∧ n = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_l505_505278


namespace first_discount_percentage_l505_505839

theorem first_discount_percentage (original_price final_price : ℝ) (additional_discount : ℝ) (x : ℝ) 
  (h1 : original_price = 400) 
  (h2 : additional_discount = 0.05) 
  (h3 : final_price = 342) 
  (hx : (original_price * (100 - x) / 100) * (1 - additional_discount) = final_price) :
  x = 10 := 
sorry

end first_discount_percentage_l505_505839


namespace find_b_perpendicular_lines_l505_505520

theorem find_b_perpendicular_lines 
  (b : ℝ) 
  (f1 : ℝ → ℝ := λ x, 3 * x - 7) 
  (f2 : ℝ → ℝ := λ x, -b / 4 * x + 3) 
  (perpendicular: (∀ x1 x2, f1 x1 = f2 x2 → (3 * (-b / 4) = -1))) :
  b = 4 / 3 :=
begin
  sorry
end

end find_b_perpendicular_lines_l505_505520


namespace least_positive_integer_with_12_factors_l505_505158

theorem least_positive_integer_with_12_factors : ∃ n : ℕ, (nat.factors_count n = 12 ∧ ∀ m : ℕ, (nat.factors_count m = 12) → n ≤ m) ∧ n = 72 :=
begin
  sorry
end

end least_positive_integer_with_12_factors_l505_505158


namespace count_four_digit_multiples_of_5_l505_505629

def is_multiple_of_5 (n : ℕ) : Prop :=
  n % 5 = 0

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def four_digit_multiples_of_5_count (lower upper : ℕ) :=
  (upper - lower + 1)

theorem count_four_digit_multiples_of_5 : 
  four_digit_multiples_of_5_count 200 1999 = 1800 :=
by
  sorry

end count_four_digit_multiples_of_5_l505_505629


namespace least_pos_int_with_12_pos_factors_is_72_l505_505230

def least_positive_integer_with_12_factors (n : ℕ) : Prop :=
  ∃ (m : ℕ), m > 0 ∧ m ≠ n ∧ (∀ d : ℕ, d ∣ m → (d = n ∨ d = 1)) ∧
  ((∀ d : ℕ, d ∣ m → ∃ e : ℕ, e = 1) → n = 72)

theorem least_pos_int_with_12_pos_factors_is_72 (n : ℕ) :
  least_positive_integer_with_12_factors n → n = 72 := by
  sorry

end least_pos_int_with_12_pos_factors_is_72_l505_505230


namespace least_positive_integer_with_12_factors_l505_505383

theorem least_positive_integer_with_12_factors : 
  ∃ (n : ℕ), (n > 0) ∧ (nat.factors_count n = 12) ∧ (∀ m, (m > 0) → (nat.factors_count m = 12) → n ≤ m) ∧ n = 108 :=
sorry

end least_positive_integer_with_12_factors_l505_505383


namespace circle_area_eq_pi_div_4_l505_505709

theorem circle_area_eq_pi_div_4 :
  ∀ (x y : ℝ), 3*x^2 + 3*y^2 - 9*x + 12*y + 27 = 0 -> (π * (1 / 2)^2 = π / 4) :=
by
  sorry

end circle_area_eq_pi_div_4_l505_505709


namespace larger_number_correct_l505_505089

theorem larger_number_correct {a b : ℕ} (hcf_ab : Nat.gcd a b = 37) 
  (lcm_factors : ∀ p ∈ [17, 23, 29, 31], Nat.Prime p) :
  ∃ c, c = 37 * 17 * 23 * 29 * 31 ∧ a = 37 ∨ b = 37 ∧ (a * b) / 37 = c :=
by
  use (37 * 17 * 23 * 29 * 31)
  split
  · refl
  · left
    sorry

end larger_number_correct_l505_505089


namespace shadow_boundary_l505_505450

theorem shadow_boundary (r : ℝ) (O P : ℝ × ℝ × ℝ) :
  r = 2 → O = (0, 0, 2) → P = (0, -2, 4) → ∀ x : ℝ, ∃ y : ℝ, y = -10 :=
by sorry

end shadow_boundary_l505_505450


namespace cinematic_academy_members_l505_505857

theorem cinematic_academy_members (h1 : ∀ x, x / 4 ≥ 196.25 → x ≥ 785) : 
  ∃ n : ℝ, 1 / 4 * n = 196.25 ∧ n = 785 :=
by
  sorry

end cinematic_academy_members_l505_505857


namespace polynomial_root_sum_bound_l505_505112

noncomputable def polynomial_roots_in_interval (p q r : ℝ) : Prop :=
  ∃ a b c : ℝ, (∀ x, x ∈ set.Ioo 0 2 → a = x ∨ b = x ∨ c = x) ∧
  a + b + c = -p ∧ a * b + b * c + c * a = q ∧  a * b * c = -r
  
theorem polynomial_root_sum_bound (p q r : ℝ) 
  (h : polynomial_roots_in_interval p q r) : -2 < p + q + r ∧ p + q + r < 0 :=
sorry

end polynomial_root_sum_bound_l505_505112


namespace least_positive_integer_with_12_factors_is_96_l505_505267

theorem least_positive_integer_with_12_factors_is_96 :
  ∃ n : ℕ, (n = 96 ∧ (∀ m : ℕ, (m > 0 ∧ ∀ d : ℕ, d ∣ m → d > 0 → d < m → (∃ k : ℕ, d = k ∨ k * d = m) → k > 1 → m > n))) ∧ (∃ m : ℕ, m = 12 ∧ (∀ d : ℕ, d ∣ m → d ∈ {1, m} ∨ ∀ k : ℕ, k * d = m → d = k ∨ d < k)).
sorry

end least_positive_integer_with_12_factors_is_96_l505_505267


namespace carolyn_sum_is_28_l505_505087

def initial_list : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def game_rules {l : List ℕ} : List ℕ → List ℕ
| []           := []
| (x :: xs)    := 
  let carolyn_removes := if l.any (x / _) else x
  let paul_removes := List.filter (λ y, y ∣ carolyn_removes) xs
  game_rules (xs.filter (! paul_removes.contains))

theorem carolyn_sum_is_28 :
  sum (game_rules initial_list) = 28 :=
sorry

end carolyn_sum_is_28_l505_505087


namespace hexagon_triangle_areas_l505_505911

def area_F : ℝ := 1.8  -- Area of quadrilateral F in cm^2
def area_basic_triangle : ℝ := area_F / 3  -- Area of one basic triangle in cm^2

theorem hexagon_triangle_areas :
  let A := 2 * area_basic_triangle,
      B := 2 * area_basic_triangle,
      C := area_basic_triangle,
      D := area_basic_triangle,
      E := 2 * area_basic_triangle,
      G := area_basic_triangle
  in
  A = 1.2 ∧ B = 1.2 ∧ C = 0.6 ∧ D = 0.6 ∧ E = 1.2 ∧ G = 0.6 :=
by
  sorry

end hexagon_triangle_areas_l505_505911


namespace volume_of_cuboid_l505_505882

theorem volume_of_cuboid (l w h : ℝ) (hl_pos : 0 < l) (hw_pos : 0 < w) (hh_pos : 0 < h) 
  (h1 : l * w = 120) (h2 : w * h = 72) (h3 : h * l = 60) : l * w * h = 4320 :=
by
  sorry

end volume_of_cuboid_l505_505882


namespace courses_choice_l505_505849

theorem courses_choice (total_courses : ℕ) (chosen_courses : ℕ)
  (h_total_courses : total_courses = 5)
  (h_chosen_courses : chosen_courses = 2) :
  ∃ (ways : ℕ), ways = 60 ∧
    (ways = ((Nat.choose total_courses chosen_courses)^2) - 
            (Nat.choose total_courses chosen_courses) - 
            ((Nat.choose total_courses chosen_courses) * 
             (Nat.choose (total_courses - chosen_courses) chosen_courses))) :=
by
  sorry

end courses_choice_l505_505849


namespace area_triangle_BEC_l505_505008

def is_trapezoid (A B C D : Type) [add_torsor ℝ A] (AD DC : ℝ) [has_dist A] :=
AD = 4 ∧ DC = 8 ∧ ∀ E F : Type, is_rectangle A B D E ∧ is_rectangle A F D E

theorem area_triangle_BEC (A B C D E F : Type) [add_torsor ℝ A] [has_dist A] :
  is_trapezoid A B C D 4 8 ∧ 
  (EF : ℝ) = 1 ∧ 
  (BE : ℝ) = 4 ∧ 
  (EC : ℝ) = 3 → 
  (area : ℝ) = 6 :=
begin
  sorry
end

end area_triangle_BEC_l505_505008


namespace find_dividend_l505_505885

-- Define the conditions
def divisor : ℕ := 20
def quotient : ℕ := 8
def remainder : ℕ := 6

-- Lean 4 statement to prove the dividend
theorem find_dividend : (divisor * quotient + remainder) = 166 := by
  sorry

end find_dividend_l505_505885


namespace problem1_problem2_l505_505936

-- Problem 1
theorem problem1 : (-2)^2 * (1 / 4) + 4 / (4 / 9) + (-1)^2023 = 7 :=
by
  sorry

-- Problem 2
theorem problem2 : -1^4 + abs (2 - (-3)^2) + (1 / 2) / (-3 / 2) = 5 + 2 / 3 :=
by
  sorry

end problem1_problem2_l505_505936


namespace correct_choice_l505_505875

-- Definitions for conditions in the problem
def condition_A (p q : Prop) (hp : p) (hq : ¬q) : ¬(p ∧ q) :=
by simp [hp, hq]

def condition_B (x y : ℝ) : ¬ (∀ y : ℝ, (x * y = 0) → (x = 0)) :=
by simp

def condition_C (α : ℝ) : (sin α = 1 / 2) → (α = π / 6) :=
by simp

def condition_D : ¬ (∀ x : ℝ, 2 ^ x > 0) ↔ ∃ x₀ : ℝ, 2 ^ x₀ ≤ 0 :=
by simp

-- The main theorem indicating that D is the correct condition
theorem correct_choice : (∃ x₀ : ℝ, 2 ^ x₀ ≤ 0) :=
sorry -- proof omitted

end correct_choice_l505_505875


namespace additional_pots_produced_l505_505411

theorem additional_pots_produced (first_hour_time_per_pot last_hour_time_per_pot : ℕ) :
  first_hour_time_per_pot = 6 →
  last_hour_time_per_pot = 5 →
  60 / last_hour_time_per_pot - 60 / first_hour_time_per_pot = 2 :=
by
  intros
  sorry

end additional_pots_produced_l505_505411


namespace solve_equation_l505_505972

noncomputable def equation (x : ℝ) : ℝ :=
  real.cbrt (18 * x - 1) - real.cbrt (10 * x + 1) - 3 * real.cbrt x

theorem solve_equation (x : ℝ) :
  (x = 0 ∨ x = -5 / 8317 ∨ x = -60 / 1614) ↔ equation x = 0 :=
by
  sorry

end solve_equation_l505_505972


namespace least_positive_integer_with_12_factors_l505_505168

theorem least_positive_integer_with_12_factors :
  ∃ m : ℕ, 0 < m ∧ (∀ n : ℕ, 0 < n → (n ≠ m →  m ≤ n)) ∧ (12 = (m.factors.toFinset.card)) := sorry

end least_positive_integer_with_12_factors_l505_505168


namespace quadratic_has_two_real_roots_find_m_l505_505593

def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem quadratic_has_two_real_roots (m : ℝ) :
  discriminant 1 (-4 * m) (3 * m^2) ≥ 0 :=
by
  unfold discriminant
  have h : (-4 * m)^2 - 4 * 1 * (3 * m^2) = 4 * m^2
  ring
  exact ge_of_eq h

theorem find_m (h : 0 < m) (root_diff : ℝ) 
  (diff_eq_two : root_diff = 2) : m = 1 :=
by
  -- Let the roots be x1 and x2
  let x1 := (4 * m + root_diff) / 2
  let x2 := (4 * m - root_diff) / 2
  have : x1 - x2 = root_diff :=
    by
      field_simp
      exact diff_eq_two
  have sum_eq := (x1 - x2) * (x1 + x2) - (x1 + x2) * (x1 - x2) = 4
  ring
  have h_m_eq_1 : 4 * m = 4,
  by field_simp
  exact h_m_eq_1

  have h_m_1 : m = 1,
  sorry
  exact ge_of_eq h_m_1

end quadratic_has_two_real_roots_find_m_l505_505593


namespace length_of_CX_l505_505076

namespace Geometry

open Classical

variables {A B C D X : Type} [metric_space A] 
  (AC AX AD CD CB DB XB CX : ℝ)
  (hAC : dist A C = 2)
  (hAX : dist A X = 5)
  (hAD : dist A D = 11)
  (hCD : dist C D = 9)
  (hCB : dist C B = 10)
  (hDB : dist D B = 1)
  (hXB : dist X B = 7)

theorem length_of_CX :
  AC = 2 → AX = 5 → AD = 11 → CD = 9 → CB = 10 → DB = 1 → XB = 7 → CX = 3 :=
  by
  intros hAC hAX hAD hCD hCB hDB hXB
  sorry

end Geometry

end length_of_CX_l505_505076


namespace least_positive_integer_with_12_factors_is_72_l505_505317

-- Definition of having exactly 12 factors for a positive integer
def has_exactly_12_factors (n : ℕ) : Prop := n > 0 ∧ (finset.filter (λ d, n % d = 0) (finset.range (n + 1))).card = 12

-- Statement of the problem in Lean 4
theorem least_positive_integer_with_12_factors_is_72 :
  ∃ n : ℕ, has_exactly_12_factors n ∧ ∀ m : ℕ, has_exactly_12_factors m → n ≤ m :=
begin
  use 72,
  split,
  { -- Prove that 72 has exactly 12 factors
    sorry },
  { -- Prove that 72 is the least such integer
    sorry }
end

end least_positive_integer_with_12_factors_is_72_l505_505317


namespace ratio_A_B_l505_505808

-- Define constants for non-zero numbers A and B
variables {A B : ℕ} (h1 : A ≠ 0) (h2 : B ≠ 0)

-- Define the given condition
theorem ratio_A_B (h : (2 * A) * 7 = (3 * B) * 3) : A / B = 9 / 14 := by
  sorry

end ratio_A_B_l505_505808


namespace standard_eq_minimal_circle_l505_505749

-- Definitions
variables {x y : ℝ}
variables (h₀ : 0 < x) (h₁ : 0 < y)
variables (h₂ : 3 / (2 + x) + 3 / (2 + y) = 1)

-- Theorem statement
theorem standard_eq_minimal_circle : (x - 4)^2 + (y - 4)^2 = 16^2 :=
sorry

end standard_eq_minimal_circle_l505_505749


namespace four_digit_multiples_of_5_count_l505_505639

-- Define the range of four-digit numbers
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

-- Define a multiple of 5
def is_multiple_of_five (n : ℕ) : Prop := n % 5 = 0

-- Define the required proof problem
theorem four_digit_multiples_of_5_count : 
  ∃ n : ℕ, (n = 1800) ∧ (∀ k : ℕ, is_four_digit k ∧ is_multiple_of_five k → n = 1800) :=
sorry

end four_digit_multiples_of_5_count_l505_505639


namespace length_of_PA_l505_505560

theorem length_of_PA
  (P A B C : Type)
  (PA_perp : ∀ (p : P), p ∈ A → p ∈ B → p /→ ⟂ p)
  (ABC_equilateral : ∀ (a b c : A), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ ∀ (x : A), dist x a = dist x b ∧ dist x b = dist x c)
  (side_length_ABC : ∀ (a b : A), dist a b = 6)
  (sphere_surface_area : ∀ (S : Type) (R : ℝ), (c : P) → S = 4 * 4 * R^2 * π)
  (sphere_radius : ∀ (R : ℝ), R = 4) :
  ∃ (PA : ℝ), PA = 4 := 
  sorry

end length_of_PA_l505_505560


namespace no_four_distinct_real_roots_l505_505496

theorem no_four_distinct_real_roots (a b : ℝ) :
  ¬ ∃ x1 x2 x3 x4 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 ∧ 
  (x1^4 - 4*x1^3 + 6*x1^2 + a*x1 + b = 0) ∧
  (x2^4 - 4*x2^3 + 6*x2^2 + a*x2 + b = 0) ∧
  (x3^4 - 4*x3^3 + 6*x3^2 + a*x3 + b = 0) ∧
  (x4^4 - 4*x4^3 + 6*x4^2 + a*x4 + b = 0) := 
by {
  sorry
}

end no_four_distinct_real_roots_l505_505496


namespace divide_rect_into_9_satisfying_conditions_l505_505956

-- Define the conditions for the rectangles
def unequal_sides (r₁ r₂ : Rectangle) : Prop :=
  r₁.width ≠ r₂.width ∧ r₁.height ≠ r₂.height

def non_combinable (rects : List Rectangle) : Prop :=
  ∀ r₁ r₂ (h₁ : r₁ ∈ rects) (h₂ : r₂ ∈ rects), r₁ ≠ r₂ → ¬(r₁.width = r₂.width ∧ r₁.height = r₂.height)

-- Define what it means to divide a rectangle into smaller rectangles meeting the criteria
def divide_into_nine (r : Rectangle) : List Rectangle :=
  sorry -- the actual implementation of the function is complex and skipped

-- The theorem that needs proof
theorem divide_rect_into_9_satisfying_conditions (r : Rectangle) :
  ∃ rects : List Rectangle, rects.length = 9 ∧ non_combinable rects ∧ ∀ r₁ r₂, r₁ ∈ rects → r₂ ∈ rects → unequal_sides r₁ r₂ := 
begin 
  sorry -- proof is omitted
end

end divide_rect_into_9_satisfying_conditions_l505_505956


namespace area_at_stage_8_l505_505683

-- Defining the constants and initial settings
def first_term : ℕ := 1
def common_difference : ℕ := 1
def stage : ℕ := 8
def square_side_length : ℕ := 4

-- Calculating the number of squares at the given stage
def num_squares : ℕ := first_term + (stage - 1) * common_difference

--Calculating the area of one square
def area_one_square : ℕ := square_side_length * square_side_length

-- Calculating the total area at the given stage
def total_area : ℕ := num_squares * area_one_square

-- Proving the total area equals 128 at Stage 8
theorem area_at_stage_8 : total_area = 128 := 
by
  sorry

end area_at_stage_8_l505_505683


namespace cylinder_volume_l505_505121

-- Defining the volume calculation for a cylinder
theorem cylinder_volume (d h : ℝ) (π : ℝ) : d = 4 → h = 4 → π > 0 → 
  let r := d / 2,
      S := π * r^2,
      V := S * h
  in V = 16 * π :=
by
  intros hd hh hπ
  let r := 4 / 2
  let S := π * r^2
  let V := S * 4
  have : r = 2 := by norm_num
  rw this at S
  have : S = 4 * π := by norm_num
  rw this at V
  norm_num at V
  rw mul_comm
  rw this
  sorry

end cylinder_volume_l505_505121


namespace FQ_span_of_tangent_circle_right_triangle_l505_505801

noncomputable def length_FQ : ℝ :=
  let DE := 7
  let DF := real.sqrt 85
  let EF := real.sqrt (DF^2 - DE^2) in
  EF

theorem FQ_span_of_tangent_circle_right_triangle (DE DF EF FQ : ℝ) (h_DE : DE = 7) (h_DF : DF = real.sqrt 85)
(h_EF : EF = real.sqrt (DF^2 - DE^2)) (h_FQ : FQ = EF) : FQ = 6 :=
by
  sorry

end FQ_span_of_tangent_circle_right_triangle_l505_505801


namespace base5_division_quotient_remainder_l505_505969

def base5_to_base10 (digits : List ℕ) : ℕ :=
  digits.reverse.foldl (λ acc d, acc * 5 + d) 0

def base10_to_base5 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else List.unfold (λ x, if x = 0 then none else some (x % 5, x / 5)) n

theorem base5_division_quotient_remainder :
  let num1 := base5_to_base10 [1, 3, 2, 4]
  let num2 := base5_to_base10 [2, 1]
  let (q, r) := (num1 / num2, num1 % num2)
  base10_to_base5 q = [3, 4] ∧ base10_to_base5 r = [1, 0] :=
by
  sorry

end base5_division_quotient_remainder_l505_505969


namespace magnitude_of_product_l505_505500

theorem magnitude_of_product :
  complex.abs ((3 * real.sqrt 2 - complex.I * 3) * (2 * real.sqrt 5 + complex.I * 5)) = 9 * real.sqrt 15 :=
by
  sorry

end magnitude_of_product_l505_505500


namespace ratio_red_to_green_apple_l505_505473

def total_apples : ℕ := 496
def green_apples : ℕ := 124
def red_apples : ℕ := total_apples - green_apples

theorem ratio_red_to_green_apple :
  red_apples / green_apples = 93 / 31 :=
by
  sorry

end ratio_red_to_green_apple_l505_505473


namespace rectangle_ratio_l505_505907

-- Define the problem
variables (x y : ℝ) (h1 : x > 0) (h2 : y > 0)
-- Given: The diagonal distance is sqrt(x^2 + y^2)
-- Given: Shortcut saves distance of x / 2 compared to walking along two sides

def shortcut_saves_distance : Prop :=
  real.sqrt (x^2 + y^2) + (x / 2) = x + y

-- Prove the ratio of the shorter side to the longer side of the rectangle
theorem rectangle_ratio (h : shortcut_saves_distance x y) : y / x = 3 / 4 :=
by 
  sorry

end rectangle_ratio_l505_505907


namespace carpet_area_l505_505025

def width : ℝ := 8
def length : ℝ := 1.5

theorem carpet_area : width * length = 12 := by
  sorry

end carpet_area_l505_505025


namespace round_robin_matches_12_players_l505_505951

theorem round_robin_matches_12_players : 
  (∑ i in (finset.range 12), i) / 2 = 66 :=
by sorry

end round_robin_matches_12_players_l505_505951


namespace same_terminal_side_l505_505472

theorem same_terminal_side (θ : ℝ) : (∃ k : ℤ, θ = 2 * k * π - π / 6) → θ = 11 * π / 6 :=
sorry

end same_terminal_side_l505_505472


namespace least_positive_integer_with_12_factors_l505_505149

theorem least_positive_integer_with_12_factors : ∃ n : ℕ, (nat.factors_count n = 12 ∧ ∀ m : ℕ, (nat.factors_count m = 12) → n ≤ m) ∧ n = 72 :=
begin
  sorry
end

end least_positive_integer_with_12_factors_l505_505149


namespace length_FQ_l505_505805

-- Define the right triangle and its properties
structure RightTriangle (DF DE EF : ℝ) :=
(right_angle_at_E : (DE^2 + EF^2 = DF^2))

-- Define the existence of a circle tangent to sides DF and EF
structure TangentCircle (DE DF EF : ℝ) :=
(center_on_DE : ∃ center : ℝ, center ∈ set.Icc 0 DE) -- Center is on segment DE
(tangent_to_DF : DF = √(center^2 + EF^2)) -- Tangency condition with side DF
(tangent_to_EF : EF = √(center^2 + DF^2)) -- Tangency condition with side EF

-- Problem statement: The length of FQ
theorem length_FQ
  (DE DF : ℝ)
  (DE_eq_7 : DE = 7)
  (DF_eq_sqrt85 : DF = real.sqrt 85)
  (EF : ℝ)
  (right_triangle : RightTriangle DF DE EF)
  (tangent_circle : TangentCircle DE DF EF) :
  (∃ FQ : ℝ, FQ = EF ∧ FQ = 6) := by
  -- Proof not included, just the statement
  sorry

end length_FQ_l505_505805


namespace least_positive_integer_with_12_factors_l505_505206

/--
Prove that the smallest positive integer with exactly 12 positive factors is 60.
-/
theorem least_positive_integer_with_12_factors : 
  ∃ (n : ℕ), (0 < n) ∧ (nat.factors_count n = 12) ∧ (∀ m : ℕ, (0 < m) ∧ (nat.factors_count m = 12) → n ≤ m) → n = 60 :=
sorry

end least_positive_integer_with_12_factors_l505_505206


namespace find_k_for_perpendicular_lines_l505_505608

theorem find_k_for_perpendicular_lines (k : ℝ) :
  (∀ x y : ℝ, (k-3) * x + (5 - k) * y + 1 = 0) →
  (∀ x y : ℝ, 2 * (k-3) * x - 2 * y + 3 = 0) →
  (k = 1 ∨ k = 4) :=
by
  sorry

end find_k_for_perpendicular_lines_l505_505608


namespace tyler_age_l505_505864

theorem tyler_age (T B : ℕ) (h1 : T = B - 3) (h2 : T + B = 11) : T = 4 :=
  sorry

end tyler_age_l505_505864


namespace additional_pots_last_hour_l505_505413

theorem additional_pots_last_hour (h1 : 60 / 6 = 10) (h2 : 60 / 5 = 12) : 12 - 10 = 2 :=
by
  sorry

end additional_pots_last_hour_l505_505413


namespace smaller_angle_at_9_15_is_172_5_l505_505619

-- Defining the problem in Lean
def hour_hand_angle (hours minutes : ℕ) : ℝ :=
  hours * 30 + minutes * 0.5

def minute_hand_angle (minutes : ℕ) : ℝ :=
  minutes * 6

def angle_between_hands (hours minutes : ℕ) : ℝ :=
  let hour_angle := hour_hand_angle hours minutes
  let minute_angle := minute_hand_angle minutes
  let angle := real.abs (hour_angle - minute_angle)
  if angle < 180 then angle else 360 - angle

-- Definition of the smaller angle at 9:15
def smaller_angle_9_15 := angle_between_hands 9 15

-- The problem statement in Lean 4
theorem smaller_angle_at_9_15_is_172_5 :
  smaller_angle_9_15 = 172.5 := by
  sorry

end smaller_angle_at_9_15_is_172_5_l505_505619


namespace rice_in_first_5_days_l505_505878

-- Define the arithmetic sequence for number of workers dispatched each day
def num_workers (n : ℕ) : ℕ := 64 + (n - 1) * 7

-- Function to compute the sum of the first n terms of the arithmetic sequence
def sum_workers (n : ℕ) : ℕ := n * 64 + (n * (n - 1)) / 2 * 7

-- Given the rice distribution conditions
def rice_per_worker : ℕ := 3

-- Given the problem specific conditions
def total_rice_distributed_first_5_days : ℕ := 
  rice_per_worker * (sum_workers 1 + sum_workers 2 + sum_workers 3 + sum_workers 4 + sum_workers 5)
  
-- Proof goal
theorem rice_in_first_5_days : total_rice_distributed_first_5_days = 3300 :=
  by
  sorry

end rice_in_first_5_days_l505_505878


namespace second_train_cross_time_l505_505863

variables (L1 L2 t2 : ℝ)
variables (v : ℝ) -- Speed of both trains

-- Time for the first train to cross the man
def time_first_train_cross_man := 27
-- Time for the trains to cross each other
def time_trains_cross_each_other := 22
-- Equality of speeds
def speeds_equal : Prop := v = (L1 / time_first_train_cross_man)

theorem second_train_cross_time (h1 : time_first_train_cross_man = 27)
                                (h2 : time_trains_cross_each_other = 22)
                                (h3 : speeds_equal) :
  t2 = 17 :=
sorry

end second_train_cross_time_l505_505863


namespace count_four_digit_multiples_of_5_l505_505662

theorem count_four_digit_multiples_of_5 : 
  let first_4_digit := 1000
  let last_4_digit := 9999
  let first_multiple_of_5 := 1000
  let last_multiple_of_5 := 9995
  let total_multiples_of_5 := (1999 - 200 + 1)
  first_multiple_of_5 % 5 = 0 ∧ last_multiple_of_5 % 5 = 0 ∧ first_4_digit ≤ first_multiple_of_5 ∧ last_multiple_of_5 ≤ last_4_digit
  → total_multiples_of_5 = 1800 :=
by
  sorry

end count_four_digit_multiples_of_5_l505_505662


namespace practice_minutes_other_days_l505_505465

-- Definitions based on given conditions
def total_hours_practiced : ℕ := 7.5 * 60 -- converting hours to minutes
def minutes_per_day := 86
def days_practiced := 2

-- Lean 4 statement for the proof problem
theorem practice_minutes_other_days :
  let total_minutes := total_hours_practiced
  let minutes_2_days := minutes_per_day * days_practiced
  total_minutes - minutes_2_days = 278 := by
  sorry

end practice_minutes_other_days_l505_505465


namespace geometric_sequence_third_term_l505_505519

theorem geometric_sequence_third_term (q : ℝ) (b1 : ℝ) (h1 : abs q < 1)
    (h2 : b1 / (1 - q) = 8 / 5) (h3 : b1 * q = -1 / 2) :
    b1 * q^2 / 2 = 1 / 8 := by
  sorry

end geometric_sequence_third_term_l505_505519


namespace least_positive_integer_with_12_factors_l505_505284

def num_factors (n : ℕ) : ℕ := 
  (finset.range (n + 1)).filter (λ d, n % d = 0).card

theorem least_positive_integer_with_12_factors : 
  ∃ n : ℕ, num_factors n = 12 ∧ n = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_l505_505284


namespace quadratic_two_real_roots_find_m_l505_505604

theorem quadratic_two_real_roots (m : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁ * x₂ = 3 * m^2 ∧ x₁ + x₂ = 4 * m :=
by
  sorry

theorem find_m (m : ℝ) (h : m > 0) (h_diff : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ - x₂ = 2) : m = 1 :=
by
  sorry

end quadratic_two_real_roots_find_m_l505_505604


namespace zhang_hong_weight_l505_505065

theorem zhang_hong_weight :
  ∀ (x : ℝ), x = 178 → 0.72 * x - 58.2 = 69.96 :=
by {
  intros x hx,
  rw hx,
  norm_num,
}

end zhang_hong_weight_l505_505065


namespace fraction_sequence_product_l505_505478

theorem fraction_sequence_product:
  (list.foldr (*) 1 (list.map (λ i: ℕ, (5 + i) / (8 + i)) (list.range 50))) = (7 / 4213440) :=
by
  sorry

end fraction_sequence_product_l505_505478


namespace HCF_of_numbers_l505_505090

/-- Define the highest common factor of two numbers -/
def HCF (a b : ℕ) : ℕ := nat.gcd a b

/-- Define the least common multiple of two numbers -/
def LCM (a b : ℕ) : ℕ := nat.lcm a b

/-- Condition 1: The HCF of the two numbers is a certain value H -/
variable {H : ℕ}

/-- Condition 2: The other two factors of their LCM are 11 and 12 -/
variable {a b : ℕ}

/-- Condition 3: The largest number is 480 -/
variable (largest : ℕ)
#check largest = max (H * a) (H * b)

theorem HCF_of_numbers :
  ∃ H : ℕ, H = 40 ∧ HCF (H * a) (H * b) = H ∧ (LCM (H * a) (H * b) = H * 11 * 12) :=
begin  
  sorry
end

end HCF_of_numbers_l505_505090


namespace log4_16_l505_505968

namespace LogarithmsProof

-- Define the conditions first
def log_b (b x : ℝ) : ℝ := sorry -- log_b function for ℝ numbers
def cond1 : 16 = 4^2 := by rfl -- condition 1: 16 = 4^2
def power_rule (b c : ℝ) (a : ℝ) : log_b b (a^c) = c * log_b b a := sorry -- power rule of logarithms
def cond3 : log_b 4 4 = 1 := by rfl -- condition 3: log4(4) = 1

-- Formalize the theorem statement
theorem log4_16 : log_b 4 16 = 2 := 
by
  unfold log_b
  apply sorry -- proof will go here
end LogarithmsProof

end log4_16_l505_505968


namespace FQ_span_of_tangent_circle_right_triangle_l505_505803

noncomputable def length_FQ : ℝ :=
  let DE := 7
  let DF := real.sqrt 85
  let EF := real.sqrt (DF^2 - DE^2) in
  EF

theorem FQ_span_of_tangent_circle_right_triangle (DE DF EF FQ : ℝ) (h_DE : DE = 7) (h_DF : DF = real.sqrt 85)
(h_EF : EF = real.sqrt (DF^2 - DE^2)) (h_FQ : FQ = EF) : FQ = 6 :=
by
  sorry

end FQ_span_of_tangent_circle_right_triangle_l505_505803


namespace count_four_digit_multiples_of_5_l505_505634

def is_multiple_of_5 (n : ℕ) : Prop :=
  n % 5 = 0

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def four_digit_multiples_of_5_count (lower upper : ℕ) :=
  (upper - lower + 1)

theorem count_four_digit_multiples_of_5 : 
  four_digit_multiples_of_5_count 200 1999 = 1800 :=
by
  sorry

end count_four_digit_multiples_of_5_l505_505634


namespace tank_capacity_l505_505459

theorem tank_capacity (C : ℝ) : 
  (0.5 * C = 0.9 * C - 45) → C = 112.5 :=
by
  intro h
  sorry

end tank_capacity_l505_505459


namespace four_digit_multiples_of_five_count_l505_505670

theorem four_digit_multiples_of_five_count : 
  {n : ℕ | 1000 ≤ n ∧ n ≤ 9999 ∧ n % 5 = 0}.toFinset.card = 1800 :=
by
  sorry

end four_digit_multiples_of_five_count_l505_505670


namespace parabola_ratio_l505_505035

-- Define the conditions and question as a theorem statement
theorem parabola_ratio
  (V₁ V₃ : ℝ × ℝ)
  (F₁ F₃ : ℝ × ℝ)
  (hV₁ : V₁ = (0, 0))
  (hF₁ : F₁ = (0, 1/8))
  (hV₃ : V₃ = (0, -1/2))
  (hF₃ : F₃ = (0, -1/4)) :
  dist F₁ F₃ / dist V₁ V₃ = 3 / 4 :=
  by
  sorry

end parabola_ratio_l505_505035


namespace least_positive_integer_with_12_factors_l505_505155

theorem least_positive_integer_with_12_factors : ∃ n : ℕ, (nat.factors_count n = 12 ∧ ∀ m : ℕ, (nat.factors_count m = 12) → n ≤ m) ∧ n = 72 :=
begin
  sorry
end

end least_positive_integer_with_12_factors_l505_505155


namespace permutations_fixed_points_sum_eq_factorial_l505_505046

noncomputable def num_permutations_with_k_fixed_points (n k : ℕ) : ℕ :=
  -- Here we assume num_permutations_with_k_fixed_points is defined based on the specific
  -- mathematical definition from the problem statement.
  sorry

theorem permutations_fixed_points_sum_eq_factorial (n : ℕ) :
  (∑ k in Finset.range (n+1), k * num_permutations_with_k_fixed_points n k) = Nat.factorial n :=
by
  sorry

end permutations_fixed_points_sum_eq_factorial_l505_505046


namespace linear_function_through_point_l505_505075

theorem linear_function_through_point (m : ℝ) :
  ∃ b, (∀ x y : ℝ, (y = m * x + b) ↔ ((0, 3) = (x, y))) → b = 3 :=
by {
  intro m,
  use 3,
  intro h,
  specialize h 0 3,
  simp at h,
  exact h.mp rfl,
}

end linear_function_through_point_l505_505075


namespace least_positive_integer_with_12_factors_is_96_l505_505264

theorem least_positive_integer_with_12_factors_is_96 :
  ∃ n : ℕ, (n = 96 ∧ (∀ m : ℕ, (m > 0 ∧ ∀ d : ℕ, d ∣ m → d > 0 → d < m → (∃ k : ℕ, d = k ∨ k * d = m) → k > 1 → m > n))) ∧ (∃ m : ℕ, m = 12 ∧ (∀ d : ℕ, d ∣ m → d ∈ {1, m} ∨ ∀ k : ℕ, k * d = m → d = k ∨ d < k)).
sorry

end least_positive_integer_with_12_factors_is_96_l505_505264


namespace max_digit_d_for_number_divisible_by_33_l505_505508

theorem max_digit_d_for_number_divisible_by_33 : ∃ d e : ℕ, d ≤ 9 ∧ e ≤ 9 ∧ 8 * 100000 + d * 10000 + 8 * 1000 + 3 * 100 + 3 * 10 + e % 33 = 0 ∧  d = 8 :=
by {
  sorry
}

end max_digit_d_for_number_divisible_by_33_l505_505508


namespace sum_of_sequence_l505_505543

theorem sum_of_sequence :
  ∀ (a : ℕ → ℕ), 
  (a 1 + a 2 = 1) →
  (a 2 + a 3 = 2) →
  (a 3 + a 4 = 3) →
  -- (conditions continue for all up to)
  (a 99 + a 100 = 99) →
  (a 100 + a 1 = 100) →
  (∑ i in Finset.range 101, a i = 2525) :=
by
  intros a h1 h2 h3 h99 h100
  sorry

end sum_of_sequence_l505_505543


namespace FGH_supermarkets_total_l505_505131

theorem FGH_supermarkets_total (US Canada : ℕ) 
  (h1 : US = 49) 
  (h2 : US = Canada + 14) : 
  US + Canada = 84 := 
by 
  sorry

end FGH_supermarkets_total_l505_505131


namespace least_positive_integer_with_12_factors_l505_505324

def has_exactly_12_factors (n : ℕ) : Prop :=
  n.factors.length = 12

theorem least_positive_integer_with_12_factors : ∃ k : ℕ, has_exactly_12_factors k ∧ (∀ n : ℕ, has_exactly_12_factors n → n ≥ k) ∧ k = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_l505_505324


namespace second_month_interest_l505_505021

def compounded_interest (initial_loan : ℝ) (rate_per_month : ℝ) : ℝ :=
  initial_loan * rate_per_month

theorem second_month_interest :
  let initial_loan := 200
  let rate_per_month := 0.10
  compounded_interest (initial_loan + compounded_interest initial_loan rate_per_month) rate_per_month = 22 :=
by
  sorry

end second_month_interest_l505_505021


namespace smaller_angle_at_9_15_is_172_5_l505_505621

-- Defining the problem in Lean
def hour_hand_angle (hours minutes : ℕ) : ℝ :=
  hours * 30 + minutes * 0.5

def minute_hand_angle (minutes : ℕ) : ℝ :=
  minutes * 6

def angle_between_hands (hours minutes : ℕ) : ℝ :=
  let hour_angle := hour_hand_angle hours minutes
  let minute_angle := minute_hand_angle minutes
  let angle := real.abs (hour_angle - minute_angle)
  if angle < 180 then angle else 360 - angle

-- Definition of the smaller angle at 9:15
def smaller_angle_9_15 := angle_between_hands 9 15

-- The problem statement in Lean 4
theorem smaller_angle_at_9_15_is_172_5 :
  smaller_angle_9_15 = 172.5 := by
  sorry

end smaller_angle_at_9_15_is_172_5_l505_505621


namespace time_canoe_from_P_to_Q_l505_505445

variables {t p r : ℝ}

theorem time_canoe_from_P_to_Q (h1 : ∃ t : ℝ, t > 0)
                               (h2 : ∃ p : ℝ, p > 0)
                               (h3 : ∃ r : ℝ, r > 0)
                               (h4 : (p - r) * t + (p + r) * (8 - t) = 8 * r) :
  t = (4 * p) / r :=
begin
  sorry,
end

end time_canoe_from_P_to_Q_l505_505445


namespace midpoint_PQ_on_circumcircle_of_PF1F2_l505_505049

-- Definitions of the geometric elements
variables {E : Type*} [metric_space E] [normed_group E] [normed_space ℝ E]
variables (ellipse : set E) (F1 F2 P A B Q : E)

-- Conditions
axiom ellipse_with_foci (hfoci : ellipse = { x : E | dist x F1 + dist x F2 = dist P F1 + dist P F2 })
axiom P_on_ellipse (hP : P ∈ ellipse)
axiom PA_intersects_ellipse (hA : A ∈ ellipse ∧ A ≠ P ∧ A ∈ line_through P F1)
axiom PB_intersects_ellipse (hB : B ∈ ellipse ∧ B ≠ P ∧ B ∈ line_through P F2)
axiom tangents_intersect_at_Q (hQ : tangent_at A ellipse ∩ tangent_at B ellipse = {Q})

-- Question to prove
theorem midpoint_PQ_on_circumcircle_of_PF1F2 (midpoint_PQ : E)
  (h_midpoint : midpoint_PQ = (P + Q) / 2) :
  (∃ ω, is_circumcircle_of_triangle ω P F1 F2 ∧ midpoint_PQ ∈ ω) :=
sorry

end midpoint_PQ_on_circumcircle_of_PF1F2_l505_505049


namespace four_digit_multiples_of_5_count_l505_505628

theorem four_digit_multiples_of_5_count :
  let lower_bound := 200
  let upper_bound := 1999
  (upper_bound - lower_bound + 1) = 1800 :=
by
  let lower_bound := 200
  let upper_bound := 1999
  show (upper_bound - lower_bound + 1) = 1800,
  from sorry

end four_digit_multiples_of_5_count_l505_505628


namespace least_positive_integer_with_12_factors_l505_505380

theorem least_positive_integer_with_12_factors : 
  ∃ (n : ℕ), (n > 0) ∧ (nat.factors_count n = 12) ∧ (∀ m, (m > 0) → (nat.factors_count m = 12) → n ≤ m) ∧ n = 108 :=
sorry

end least_positive_integer_with_12_factors_l505_505380


namespace quadratic_has_two_real_roots_find_m_l505_505592

def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem quadratic_has_two_real_roots (m : ℝ) :
  discriminant 1 (-4 * m) (3 * m^2) ≥ 0 :=
by
  unfold discriminant
  have h : (-4 * m)^2 - 4 * 1 * (3 * m^2) = 4 * m^2
  ring
  exact ge_of_eq h

theorem find_m (h : 0 < m) (root_diff : ℝ) 
  (diff_eq_two : root_diff = 2) : m = 1 :=
by
  -- Let the roots be x1 and x2
  let x1 := (4 * m + root_diff) / 2
  let x2 := (4 * m - root_diff) / 2
  have : x1 - x2 = root_diff :=
    by
      field_simp
      exact diff_eq_two
  have sum_eq := (x1 - x2) * (x1 + x2) - (x1 + x2) * (x1 - x2) = 4
  ring
  have h_m_eq_1 : 4 * m = 4,
  by field_simp
  exact h_m_eq_1

  have h_m_1 : m = 1,
  sorry
  exact ge_of_eq h_m_1

end quadratic_has_two_real_roots_find_m_l505_505592


namespace point_symmetric_y_axis_l505_505717

theorem point_symmetric_y_axis (P : ℝ × ℝ) (hx : P = (2, 5)) : reflection_y P = (-2, 5) :=
by sorry

end point_symmetric_y_axis_l505_505717


namespace proof_problem_l505_505762

def A : Set ℝ := {x | x < 4}
def B : Set ℝ := {x | x^2 - 4 * x + 3 > 0}

theorem proof_problem : {x | x ∈ A ∧ x ∉ (A ∩ B)} = {x | 1 ≤ x ∧ x ≤ 3} :=
by {
  sorry
}

end proof_problem_l505_505762


namespace no_intersection_points_l505_505672

theorem no_intersection_points :
  ∀ x y : ℝ, y = abs (3 * x + 6) ∧ y = -2 * abs (2 * x - 1) → false :=
by
  intros x y h
  cases h
  sorry

end no_intersection_points_l505_505672


namespace correct_calculation_l505_505414

-- Define the statements
def option_A (a : ℕ) := (a^2 + a^3 ≠ a^5)
def option_B (a : ℕ) := (a^2 * a^3 = a^5)
def option_C (a : ℕ) := ((a^2)^3 ≠ a^5)
def option_D (a : ℕ) := (a^5 / a^3 = a^2)

-- Prove that only option D is correct
theorem correct_calculation (a : ℕ) : option_A a → option_B a → option_C a → option_D a :=
by
  intros hA hB hC hD
  exact hD

end correct_calculation_l505_505414


namespace least_positive_integer_with_12_factors_l505_505248

theorem least_positive_integer_with_12_factors :
  ∃ k : ℕ, (1 ≤ k) ∧ (nat.factors k).length = 12 ∧
    ∀ n : ℕ, (1 ≤ n) ∧ (nat.factors n).length = 12 → k ≤ n := 
begin
  sorry
end

end least_positive_integer_with_12_factors_l505_505248


namespace petr_son_is_nikolai_l505_505064

theorem petr_son_is_nikolai (x y : ℕ) (h1 : x + x + y + 3*y = 25) : Petr.Son.Name = "Nikolai" :=
sorry

end petr_son_is_nikolai_l505_505064


namespace soccer_most_students_l505_505477

def sports := ["hockey", "basketball", "soccer", "volleyball", "badminton"]
def num_students (sport : String) : Nat :=
  match sport with
  | "hockey" => 30
  | "basketball" => 35
  | "soccer" => 50
  | "volleyball" => 20
  | "badminton" => 25
  | _ => 0

theorem soccer_most_students : ∀ sport ∈ sports, num_students "soccer" ≥ num_students sport := by
  sorry

end soccer_most_students_l505_505477


namespace find_special_vectors_l505_505029

structure Vector (α : Type) :=
(a : α)
(b : α)

def P : Set (Vector ℤ) := 
  {v | 0 ≤ v.a ∧ v.a ≤ 2 ∧ 0 ≤ v.b ∧ v.b ≤ 100 ∧ Int ∈ {v.a, v.b}}

theorem find_special_vectors : 
  {v | v ∈ P ∧
    (∀ n, (∃ m, P \ v = (finset.range n).sum (λ i, v) ↔ (finset.range m).sum (λ j, v)) 
    → v = Vector.mk 1 (2 * ℕ))} :=
sorry

end find_special_vectors_l505_505029


namespace manufacturing_cost_is_190_l505_505106

theorem manufacturing_cost_is_190:
  ∃ M: ℝ, 
  let transport_cost := 5 in
  let selling_price := 234 in
  let gain := 0.20 in
  selling_price = (1 + gain) * (M + transport_cost) →
  M = 190 :=
by
  sorry

end manufacturing_cost_is_190_l505_505106


namespace least_positive_integer_with_12_factors_l505_505307

def has_exactly_12_factors (n : ℕ) : Prop :=
  (finset.range(n + 1).filter (λ d, n % d = 0)).card = 12

theorem least_positive_integer_with_12_factors : ∀ n : ℕ, has_exactly_12_factors n → n ≥ 150 :=
sorry

end least_positive_integer_with_12_factors_l505_505307


namespace least_integer_with_twelve_factors_l505_505190

open Classical

theorem least_integer_with_twelve_factors : ∃ k : ℕ, (∀ n : ℕ, (count_factors n = 12 → k ≤ n)) ∧ k = 72 :=
by
  sorry

-- Here count_factors is a function that returns the number of positive divisors of a given integer.

end least_integer_with_twelve_factors_l505_505190


namespace ned_initial_lives_l505_505421

-- Define the initial number of lives Ned had
def initial_lives (start_lives current_lives lost_lives : ℕ) : ℕ :=
  current_lives + lost_lives

-- Define the conditions
def current_lives := 70
def lost_lives := 13

-- State the theorem
theorem ned_initial_lives : initial_lives current_lives current_lives lost_lives = 83 := by
  sorry

end ned_initial_lives_l505_505421


namespace least_positive_integer_with_12_factors_l505_505247

theorem least_positive_integer_with_12_factors :
  ∃ k : ℕ, (1 ≤ k) ∧ (nat.factors k).length = 12 ∧
    ∀ n : ℕ, (1 ≤ n) ∧ (nat.factors n).length = 12 → k ≤ n := 
begin
  sorry
end

end least_positive_integer_with_12_factors_l505_505247


namespace polynomial_divisibility_l505_505986

noncomputable def iterated_p (p : ℂ[X]) (n : ℕ) : ℂ[X] :=
  if n = 0 then X else (λ q, p.comp q)^[n] X

theorem polynomial_divisibility {p : ℂ[X]} (h_deg : 1 < p.natDegree)
  (h_div : ∀ n : ℕ, 1 < n → (p - X)^2 ∣ iterated_p p n - X) :
  ∃ (c : ℂ) (r : list ℂ), p = X + c * (r.map (λ ri, X - polynomial.C ri)).prod :=
begin
  sorry
end

end polynomial_divisibility_l505_505986


namespace least_positive_integer_with_12_factors_l505_505303

def has_exactly_12_factors (n : ℕ) : Prop :=
  (finset.range(n + 1).filter (λ d, n % d = 0)).card = 12

theorem least_positive_integer_with_12_factors : ∀ n : ℕ, has_exactly_12_factors n → n ≥ 150 :=
sorry

end least_positive_integer_with_12_factors_l505_505303


namespace Justine_colored_sheets_l505_505854

theorem Justine_colored_sheets :
  (∃ sheets total_binders sheets_per_binder :
    nat, total_binders > 0 ∧ total_binders = 5 ∧ sheets = 2450 ∧ sheets_per_binder = sheets / total_binders 
    ∧ Justine_colored = sheets_per_binder / 2) → Justine_colored = 245 :=
by 
  intro h,
  rcases h with ⟨sheets, total_binders, sheets_per_binder, h1, h2, h3, h4, h5⟩,
  have sheets_per_binder_calc : sheets_per_binder = 490 := nat.div_eq_of_eq_mul_left h1 h2.symm h3,
  have Justine_colored_calc : Justine_colored = 245 := nat.div_eq_of_eq_mul_left (nat.zero_lt_of_lt h1 (nat.succ_pos 4)) h4.symm h5,
  exact Justine_colored_calc,
  simp [Justine_colored_calc],
  sorry

end Justine_colored_sheets_l505_505854


namespace range_of_a_l505_505998

noncomputable def real_condition (a : ℝ) (x : ℝ) : Prop :=
  x^2 - 4*a*x + 3*a^2 < 0

noncomputable def A : Set ℝ :=
  {x | x^2 - x - 6 < 0}

noncomputable def B : Set ℝ :=
  {x | x^2 + 2*x - 8 ≥ 0}

lemma neg_R_B_eq : (Set.univ \ B) = {x | -4 < x ∧ x < 2} :=
sorry

lemma A_eq : A = { x | -2 < x ∧ x < 3 } :=
sorry

lemma subset_condition (a : ℝ) : set_of (real_condition a) ⊆ ({ x | -2 < x ∧ x < 2 }) :=
sorry

theorem range_of_a (a : ℝ) : 
  a ≠ 0 → 
  (real_condition a) ⊆ ({ x | -2 < x ∧ x < 2 }) →
  (0 < a ∧ a ≤ 2 / 3) ∨ (-2 / 3 ≤ a ∧ a < 0) :=
sorry

end range_of_a_l505_505998


namespace number_of_solutions_l505_505745

noncomputable def g (x : ℝ) : ℝ := -3 * Real.sin (2 * Real.pi * x)

theorem number_of_solutions (h : -1 ≤ x ∧ x ≤ 1) : 
  (∃ s : ℕ, s = 21 ∧ ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → g (g (g x)) = g x) :=
sorry

end number_of_solutions_l505_505745


namespace linear_function_passing_through_0_3_l505_505072

theorem linear_function_passing_through_0_3 (m : ℝ) : ∃ f : ℝ → ℝ, f(0) = 3 ∧ ∀ x, f(x) = m * x + 3 :=
by
  existsi (λ x => m * x + 3)
  split
  { simp }
  sorry

end linear_function_passing_through_0_3_l505_505072


namespace least_positive_integer_with_12_factors_is_72_l505_505311

-- Definition of having exactly 12 factors for a positive integer
def has_exactly_12_factors (n : ℕ) : Prop := n > 0 ∧ (finset.filter (λ d, n % d = 0) (finset.range (n + 1))).card = 12

-- Statement of the problem in Lean 4
theorem least_positive_integer_with_12_factors_is_72 :
  ∃ n : ℕ, has_exactly_12_factors n ∧ ∀ m : ℕ, has_exactly_12_factors m → n ≤ m :=
begin
  use 72,
  split,
  { -- Prove that 72 has exactly 12 factors
    sorry },
  { -- Prove that 72 is the least such integer
    sorry }
end

end least_positive_integer_with_12_factors_is_72_l505_505311


namespace students_making_stars_l505_505128

theorem students_making_stars (total_stars stars_per_student : ℕ) (h1 : total_stars = 372) (h2 : stars_per_student = 3) : 
  total_stars / stars_per_student = 124 :=
by
  sorry

end students_making_stars_l505_505128


namespace stratified_sampling_sophomores_l505_505441

variable (total_students freshmen sophomores juniors sample_size : ℕ)
variable (ratio : ℚ)

def total_students := 400 + 320 + 280
def freshmen := 400
def sophomores := 320
def juniors := 280
def sample_size := 50
def ratio := sample_size / total_students

theorem stratified_sampling_sophomores :
  sophomores * ratio = 16 :=
by 
  sorry

end stratified_sampling_sophomores_l505_505441


namespace find_x_equidistant_l505_505910

noncomputable def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem find_x_equidistant (x : ℝ) :
  dist (x, 2) (1, 3) = dist (x, 2) (5, -1) ↔ x = 4 :=
by
  sorry

end find_x_equidistant_l505_505910


namespace area_of_closed_region_l505_505816

theorem area_of_closed_region :
  ∃ (n : ℕ), (binomial n 2 = binomial n 3) ∧
  ∃ (a b : ℝ), (y = λ x => n * x) ∧
  ∃ (f g : ℝ → ℝ), f = (λ x => 5 * x) ∧
  g = (λ x => x ^ 2) ∧
  ∃ (x1 x2 : ℝ), ((5 * x1 = x1 ^ 2) ∧ (5 * x2 = x2 ^ 2)) ∧
  ∫ x1 to x2, (5 * x - x^2) dx = 125 / 6 := 
by 
  sorry

end area_of_closed_region_l505_505816


namespace a_33_equals_3_l505_505999

def seq : ℕ → ℤ
| 0 := 3
| 1 := 6
| n+2 := seq (n+1) - seq n

theorem a_33_equals_3 : seq 32 = 3 := 
by {
  sorry
}

end a_33_equals_3_l505_505999


namespace least_positive_integer_with_12_factors_l505_505178

theorem least_positive_integer_with_12_factors :
  ∃ m : ℕ, 0 < m ∧ (∀ n : ℕ, 0 < n → (n ≠ m →  m ≤ n)) ∧ (12 = (m.factors.toFinset.card)) := sorry

end least_positive_integer_with_12_factors_l505_505178


namespace polynomial_zero_iff_divisibility_l505_505886

theorem polynomial_zero_iff_divisibility (P : Polynomial ℤ) :
  (∀ n : ℕ, n > 0 → ∃ k : ℤ, P.eval (2^n) = n * k) ↔ P = 0 :=
by sorry

end polynomial_zero_iff_divisibility_l505_505886


namespace T_sum_example_l505_505740

def T (n : ℕ) : ℤ :=
  if n % 2 = 0 then
    -n / 2
  else
    (n + 1) / 2

theorem T_sum_example : T 20 + T 36 + T 45 = -5 := by
  have T20 : T 20 = -10 := by
    unfold T
    simp [if_pos (by exact Mod.eq_zero_of_dvd (show 2 ∣ (20 : ℕ) by norm_num))]
  have T36 : T 36 = -18 := by
    unfold T
    simp [if_pos (by exact Mod.eq_zero_of_dvd (show 2 ∣ (36 : ℕ) by norm_num))]
  have T45 : T 45 = 23 := by
    unfold T
    simp [if_neg (by exact Mod.ne_zero_of_dvd ((show 2 ∣ (45 : ℕ) by dec_trivial)))]
  -- Compute the final sum
  calc
    T 20 + T 36 + T 45
    _ = -10 + -18 + 23 := by rw [T20, T36, T45]
    _ = -5 := by norm_num

end T_sum_example_l505_505740


namespace train_length_correct_l505_505916

-- Define the conditions
def time_to_cross_bridge : ℝ := 29.997600191984642
def bridge_length : ℝ := 150
def train_speed_kmph : ℝ := 36

-- Convert speed from kmph to m/s
def train_speed_mps : ℝ := (train_speed_kmph * 1000) / 3600

-- Define the correct answer for the length of the train
def length_of_train : ℝ := train_speed_mps * time_to_cross_bridge - bridge_length

-- Lean theorem statement to prove the length of the train
theorem train_length_correct : length_of_train = 149.97600191984642 := by
  sorry

end train_length_correct_l505_505916


namespace multiples_of_5_in_4_digit_range_l505_505655

theorem multiples_of_5_in_4_digit_range : 
  let count_multiples := (9995 - 1000) / 5 + 1
  in count_multiples = 1800 :=
by
  sorry

end multiples_of_5_in_4_digit_range_l505_505655


namespace find_sum_of_squares_l505_505742

def orthogonal_matrix (B : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  B.transpose = B⁻¹

def B := ![![x, y], ![z, w]]

theorem find_sum_of_squares (x y z w : ℝ) 
  (h_orthogonal : orthogonal_matrix B) 
  (h_sum_zero : x + y + z + w = 0) : 
  x^2 + y^2 + z^2 + w^2 = 2 :=
  sorry

end find_sum_of_squares_l505_505742


namespace max_average_numbers_l505_505044

-- Definitions based on conditions from the problem
def total_weight (weights : list ℕ) : ℕ := weights.sum

def is_average (weights : list ℕ) (S : ℕ) (k : ℕ) : Prop := 
  ∃ (subset : list ℕ), subset ⊆ weights ∧ subset.length = k ∧ total_weight subset = S

-- Lean statement of the proof problem
theorem max_average_numbers (weights : list ℕ) (S : ℕ) (h1 : weights.length = 100) (h2 : total_weight weights = 2 * S) :
  (∃ (avg_count : ℕ), avg_count = 97 ∧ ∀ k, is_average weights S k ↔ (k = 2 ∨ k = 3 ∨ k = 4 ∨ ... ∨ k = 98)) :=
sorry

end max_average_numbers_l505_505044


namespace towels_folded_together_l505_505019

/-
Jane, Kyla, and Anthony's towel-folding rates.
-/
def Jane_rate := 3 * 60 / 5
def Kyla_rate := 5 * 60 / 10
def Anthony_rate := 7 * 60 / 20

/-- The total number of towels they can fold together in one hour. -/
theorem towels_folded_together : Jane_rate + Kyla_rate + Anthony_rate = 87 :=
by
  -- Jane folds 3 towels in 5 minutes, thus in 1 hour she can fold 36 towels.
  have hJane : Jane_rate = 36 := by sorry
  -- Kyla folds 5 towels in 10 minutes, thus in 1 hour she can fold 30 towels.
  have hKyla : Kyla_rate = 30 := by sorry
  -- Anthony folds 7 towels in 20 minutes, thus in 1 hour he can fold 21 towels.
  have hAnthony : Anthony_rate = 21 := by sorry
  -- Therefore, the total number of towels folded together in one hour is 87.
  calc
    Jane_rate + Kyla_rate + Anthony_rate
      = 36 + 30 + 21 : by sorry
      = 87 : by rfl

end towels_folded_together_l505_505019


namespace angle_between_hour_minute_hands_8PM_l505_505119

-- Definitions of given conditions
def time_on_clock := 8 -- representing 8 PM
def hour_hand_position (time: Nat) := if time = 8 then 8 else 0
def minute_hand_position (time: Nat) := if time = 8 then 12 else 0
def major_divisions := 12 -- clock has 12 major divisions
def degrees_per_division := 30 -- each division is 30 degrees

-- Proposition statement to prove the angle less than 180° between the hour and minute hands is 120°
theorem angle_between_hour_minute_hands_8PM : 
  let hour_hand := hour_hand_position time_on_clock,
      minute_hand := minute_hand_position time_on_clock,
      divisions_between_hands := (if minute_hand >= hour_hand then minute_hand - hour_hand else major_divisions - (hour_hand - minute_hand)),
      angle := divisions_between_hands * degrees_per_division
  in divisions_between_hands = 4 ∧ angle = 120° := 
by 
  sorry

end angle_between_hour_minute_hands_8PM_l505_505119


namespace euler_totient_sum_eq_m_l505_505987

open Nat

theorem euler_totient_sum_eq_m (m : ℕ) (hm : 0 < m) : ∑ d in Nat.divisors m, Nat.totient d = m :=
sorry

end euler_totient_sum_eq_m_l505_505987


namespace unit_vector_orthogonal_l505_505971

open Real EuclideanSpace

-- Define the two vectors
def v1 : EuclideanSpace ℝ (Fin 3) := ![2, 1, 1]
def v2 : EuclideanSpace ℝ (Fin 3) := ![0, 1, 2]

-- Define the cross product
def cross_prod (a b : EuclideanSpace ℝ (Fin 3)) : EuclideanSpace ℝ (Fin 3) :=
  ![
    a 1 * b 2 - a 2 * b 1,
    a 2 * b 0 - a 0 * b 2,
    a 0 * b 1 - a 1 * b 0
  ]

-- Calculate the vector orthogonal to both v1 and v2
def orthogonal_vector : EuclideanSpace ℝ (Fin 3) := cross_prod v1 v2

-- Define the magnitude of a vector
def magnitude (v : EuclideanSpace ℝ (Fin 3)) : ℝ :=
  sqrt (v 0 * v 0 + v 1 * v 1 + v 2 * v 2)

-- Define the unit vector function
def unit_vector (v : EuclideanSpace ℝ (Fin 3)) : EuclideanSpace ℝ (Fin 3) :=
  let m := magnitude v in
  ![
    v 0 / m,
    v 1 / m,
    v 2 / m
  ]

-- Theorem statement
theorem unit_vector_orthogonal : 
  unit_vector orthogonal_vector = ![
    1 / sqrt 21,
    -4 / sqrt 21,
    2 / sqrt 21
  ] :=
sorry

end unit_vector_orthogonal_l505_505971


namespace ambulance_ride_cost_is_correct_l505_505491

-- Define all the constants and conditions
def daily_bed_cost : ℝ := 900
def bed_days : ℕ := 3
def specialist_rate_per_hour : ℝ := 250
def specialist_minutes_per_day : ℕ := 15
def specialists_count : ℕ := 2
def total_bill : ℝ := 4625

noncomputable def ambulance_cost : ℝ :=
  total_bill - ((daily_bed_cost * bed_days) + (specialist_rate_per_hour * (specialist_minutes_per_day / 60) * specialists_count))

-- The proof statement
theorem ambulance_ride_cost_is_correct : ambulance_cost = 1675 := by
  sorry

end ambulance_ride_cost_is_correct_l505_505491


namespace four_digit_multiples_of_5_l505_505643

theorem four_digit_multiples_of_5 : 
  let f := fun n => 1000 <= n ∧ n <= 9999 ∧ n % 5 = 0
  in finset.card (finset.filter f (finset.range 10000)) = 1800 :=
by
  sorry

end four_digit_multiples_of_5_l505_505643


namespace recurring_decimal_to_rational_l505_505872

theorem recurring_decimal_to_rational : ∀ (x : ℚ), x = 0.125125125125... → x = 125 / 999 := by
  intro x
  assume h : x = 0.125125125125...
  have h₁ : 1000 * x = 125.125125125...
  sorry

  have h₂ : 999 * x = 125
  sorry

  have h₃ : x = 125 / 999
  sorry

  exact h₃

end recurring_decimal_to_rational_l505_505872


namespace length_of_rectangle_l505_505815

-- Given conditions as per the problem statement
variables {s l : ℝ} -- side length of the square, length of the rectangle
def width_rectangle : ℝ := 10 -- width of the rectangle

-- Conditions
axiom sq_perimeter : 4 * s = 200
axiom area_relation : s^2 = 5 * (l * width_rectangle)

-- Goal to prove
theorem length_of_rectangle : l = 50 :=
by
  sorry

end length_of_rectangle_l505_505815


namespace length_LM_l505_505079

def RectanglePQRS (PQ QR : ℝ) (PQ_value : PQ = 5) (QR_value : QR = 7) :=
  True  -- Placeholder for representing the rectangle conditions

theorem length_LM (PQ QR : ℝ) (PQ_value : PQ = 5) (QR_value : QR = 7)
    (LM_perpendicular : LM ⊥ diagonal PR)
    (P_on_DL : P_P') (S_on_DM : S_S') :
    LM = 2 * Real.sqrt 74 :=
by 
  sorry

end length_LM_l505_505079


namespace general_term_formula_l505_505050

noncomputable def sequence (n : ℕ) : ℕ :=
  if n = 0 ∨ n = 1 then 1 else ∏ k in finset.range (n + 1), if k ≥ 2 then (2^k - 1)^2 else 1

theorem general_term_formula :
  ∀ (a : ℕ → ℕ),
    (a 0 = 1) →
    (a 1 = 1) →
    (∀ n, n ≥ 2 → sqrt (a n * a (n - 2)) - sqrt (a (n - 1) * a (n - 2)) = 2 * a (n-1)) →
    (∀ n, a n = (sequence n)) :=
by sorry

end general_term_formula_l505_505050


namespace least_positive_integer_with_12_factors_l505_505349

theorem least_positive_integer_with_12_factors : ∃ k : ℕ, k > 0 ∧ (∀ n : ℕ, n > 0 → (factors_count n = 12 → k ≤ n)) ∧ k = 72 :=
by {
  sorry,
}

end least_positive_integer_with_12_factors_l505_505349


namespace problem_f2017_equals_sin_cos_l505_505551

noncomputable def f : ℕ → (ℝ → ℝ)
| 0     := λ x, 0
| (n+1) := λ x, if n = 0 then sin x + cos x else deriv (f n)

theorem problem_f2017_equals_sin_cos : f 2017 = λ x, sin x + cos x :=
by sorry

end problem_f2017_equals_sin_cos_l505_505551


namespace pineapple_cost_l505_505876

theorem pineapple_cost (
  pineapple_cost : ℝ,
  num_pineapples : ℕ, 
  shipping_cost : ℝ
) (h₀ : pineapple_cost = 1.25)
  (h₁ : num_pineapples = 12)
  (h₂ : shipping_cost = 21.00) :
  (pineapple_cost * num_pineapples + shipping_cost) / num_pineapples = 3.00 := 
by {
  sorry
}

end pineapple_cost_l505_505876


namespace sectors_with_frogs_occupied_l505_505433

theorem sectors_with_frogs_occupied (N : ℕ) (N_pos : 0 < N) (frogs : Finset (Fin N)) (h : frogs.card = N + 1) :
∃ t : ℕ, (frogs_at_time t).size ≥ (N + 1) / 2 :=
sorry

end sectors_with_frogs_occupied_l505_505433


namespace four_digit_multiples_of_5_count_l505_505624

theorem four_digit_multiples_of_5_count :
  let lower_bound := 200
  let upper_bound := 1999
  (upper_bound - lower_bound + 1) = 1800 :=
by
  let lower_bound := 200
  let upper_bound := 1999
  show (upper_bound - lower_bound + 1) = 1800,
  from sorry

end four_digit_multiples_of_5_count_l505_505624


namespace MathMattersRankings_l505_505813

theorem MathMattersRankings : 
  let x : ℕ → ℕ := λ n, Nat.pow 2 (n - 1)
  in x 10 = 512 := 
by
  -- x(10) = 2^(10-1) = 2^9 = 512
  sorry

end MathMattersRankings_l505_505813


namespace least_pos_int_with_12_pos_factors_is_72_l505_505229

def least_positive_integer_with_12_factors (n : ℕ) : Prop :=
  ∃ (m : ℕ), m > 0 ∧ m ≠ n ∧ (∀ d : ℕ, d ∣ m → (d = n ∨ d = 1)) ∧
  ((∀ d : ℕ, d ∣ m → ∃ e : ℕ, e = 1) → n = 72)

theorem least_pos_int_with_12_pos_factors_is_72 (n : ℕ) :
  least_positive_integer_with_12_factors n → n = 72 := by
  sorry

end least_pos_int_with_12_pos_factors_is_72_l505_505229


namespace quadrilateral_OBEC_area_l505_505442

noncomputable def quadrilateral_area : ℝ := 
  let A := (20 / 3, 0)
  let B := (0, 20)
  let C := (10, 0)
  let O := (0, 0)
  let E := (5, 5)
  let area_triangle := λ (p1 p2 p3 : ℝ × ℝ), 
    0.5 * abs (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2))
  area_triangle O B E + area_triangle O E C

theorem quadrilateral_OBEC_area :
  quadrilateral_area = 275 / 3 :=
by
  sorry

end quadrilateral_OBEC_area_l505_505442


namespace complex_conjugate_magnitude_l505_505576

theorem complex_conjugate_magnitude (z : ℂ) (h : Complex.abs (Complex.conj z) = 4) : z * Complex.conj z = 16 := 
by
  sorry

end complex_conjugate_magnitude_l505_505576


namespace equal_common_points_k_half_values_of_k_for_common_points_l505_505733

-- Define the initial conditions: the triangle, the circumcenter, and the angle bisectors
variables (A B C O : Point) (k : ℝ)
axiom acute_triangle_not_isosceles : acute_triangle A B C ∧ ¬ isosceles_triangle A B C
axiom circumcenter_O : circumcenter O A B C
axiom internal_bisectors : internal_bisector A D ∧ internal_bisector B E ∧ internal_bisector C F
axiom points_on_rays : 
  ∀ L M N, 
    (∃ k, 0 < k ∧ identical_ratios (AL / AD = BM / BE = CN / CF = k))

-- Denote the circles passing through those points
def circle_passing_through (O_i : Point) (P : Point) (Q : Point) : Prop := 
  ∃ O_i P Q, circle_circ (passes_through_point O_i P) (tangent_at_point O_i Q)

axiom circles_defined : 
  ∀ O_1 O_2 O_3, 
    circle_passing_through O_1 L A O ∧
    circle_passing_through O_2 M B O ∧
    circle_passing_through O_3 N C O

-- Prove the condition for k = 1/2
theorem equal_common_points_k_half (G : Point) :
  ∀ k = 1/2, 
    (common_points (O_1, O_2, O_3) = 2) ∧ 
    (centroid_of G A B C) ∧ 
    lies_on_common_chord G (O_1, O_2, O_3) := sorry

-- General proof for all values of k such that the circles have exactly two common points
theorem values_of_k_for_common_points :
  ∀ k, 
    (common_points (O_1, O_2, O_3) = 2) ↔ (k = 1 ∨ k = 1/2) := sorry

end equal_common_points_k_half_values_of_k_for_common_points_l505_505733


namespace least_positive_integer_with_12_factors_l505_505289

def num_factors (n : ℕ) : ℕ := 
  (finset.range (n + 1)).filter (λ d, n % d = 0).card

theorem least_positive_integer_with_12_factors : 
  ∃ n : ℕ, num_factors n = 12 ∧ n = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_l505_505289


namespace only_value_of_k_l505_505975

def A (k a b : ℕ) : ℚ := (a + b : ℚ) / (a^2 + k^2 * b^2 - k^2 * a * b : ℚ)

theorem only_value_of_k : (∀ a b : ℕ, 0 < a → 0 < b → ¬ (∃ c d : ℕ, 1 < c ∧ A 1 a b = (c : ℚ) / (d : ℚ))) → k = 1 := 
    by sorry  -- proof omitted

-- Note: 'only_value_of_k' states that given the conditions, there is no k > 1 that makes A(k, a, b) a composite number, hence k must be 1.

end only_value_of_k_l505_505975


namespace area_at_stage_8_l505_505681

-- Defining the constants and initial settings
def first_term : ℕ := 1
def common_difference : ℕ := 1
def stage : ℕ := 8
def square_side_length : ℕ := 4

-- Calculating the number of squares at the given stage
def num_squares : ℕ := first_term + (stage - 1) * common_difference

--Calculating the area of one square
def area_one_square : ℕ := square_side_length * square_side_length

-- Calculating the total area at the given stage
def total_area : ℕ := num_squares * area_one_square

-- Proving the total area equals 128 at Stage 8
theorem area_at_stage_8 : total_area = 128 := 
by
  sorry

end area_at_stage_8_l505_505681


namespace coeff_x3_l505_505868

def P : Polynomial ℤ := x^5 - 4*x^4 + 7*x^3 - 5*x^2 + 3*x - 2
def Q : Polynomial ℤ := 3*x^2 - 5*x + 6

theorem coeff_x3 (P Q : Polynomial ℤ) : (P * Q).coeff 3 = 42 := 
by
  sorry

end coeff_x3_l505_505868


namespace no_polynomial_with_all_prime_values_l505_505781

open Polynomial

noncomputable def polynomial_is_prime_for_all (P : ℤ[X]) : Prop :=
  ∀ (n : ℕ), nat.prime (eval n P)

theorem no_polynomial_with_all_prime_values (P : ℤ[X]) :
  ¬ polynomial_is_prime_for_all P :=
sorry

end no_polynomial_with_all_prime_values_l505_505781


namespace minimum_cards_ensuring_pair_sum_l505_505438

def total_cards : Nat := 54

def card_points (card : Nat) : Nat :=
  if card = 1 then 1
  else if card = 11 then 11
  else if card = 12 then 12
  else if card = 13 then 13
  else if card = 0 then 0
  else card

def pairs_to_fourteen : List (Nat × Nat) := 
  [ (1, 13), (2, 12), (3, 11), (4, 10), (5, 9), (6, 8) ]

def minimum_draw_for_certain_pair_sum : Nat := 28

theorem minimum_cards_ensuring_pair_sum : 
  ∀ (drawn_cards : List Nat), (List.length drawn_cards = minimum_draw_for_certain_pair_sum) →
    ∃ (pair : Nat × Nat), pair ∈ pairs_to_fourteen ∧ 
    List.mem (fst pair) drawn_cards ∧ List.mem (snd pair) drawn_cards :=
sorry

end minimum_cards_ensuring_pair_sum_l505_505438


namespace xy_in_A_l505_505027

def A : Set ℤ :=
  {z | ∃ (a b k n : ℤ), z = a^2 + k * a * b + n * b^2}

theorem xy_in_A (x y : ℤ) (hx : x ∈ A) (hy : y ∈ A) : x * y ∈ A := sorry

end xy_in_A_l505_505027


namespace problem_l505_505679

theorem problem (a b : ℤ) (h : {a^2, 0, -1} = {a, b, 0}) : a^2018 + b^2018 = 2 :=
sorry

end problem_l505_505679


namespace least_number_divisible_by_12_leaves_remainder_4_is_40_l505_505147

theorem least_number_divisible_by_12_leaves_remainder_4_is_40 :
  ∃ n : ℕ, (∀ k : ℕ, n = 12 * k + 4) ∧ (∀ m : ℕ, (∀ k : ℕ, m = 12 * k + 4) → n ≤ m) ∧ n = 40 :=
by
  sorry

end least_number_divisible_by_12_leaves_remainder_4_is_40_l505_505147


namespace exponent_rule_division_l505_505417

theorem exponent_rule_division (a : ℝ) (h₁ : a ≠ 0) : (a^5 / a^3 = a^2) := 
by
  -- Since (a^5 / a^3) = a^(5-3) from the exponent division rule
  have h₂ : a^5 / a^3 = a^(5 - 3), by sorry,
  -- Substitute the exponent subtraction
  have h₃ : 5 - 3 = 2, by linarith,
  -- Final substitution
  rw [h₃] at h₂,
  exact h₂

end exponent_rule_division_l505_505417


namespace sum_intercepts_eq_2003_over_2004_l505_505525

noncomputable def sum_of_intercepted_segments : ℚ :=
  ∑ n in finset.range 2003, 1 / (n.succ * (n.succ + 1))

theorem sum_intercepts_eq_2003_over_2004 : 
  sum_of_intercepted_segments = 2003 / 2004 := 
by
  sorry

end sum_intercepts_eq_2003_over_2004_l505_505525


namespace probability_of_selecting_quarter_l505_505901

theorem probability_of_selecting_quarter :
  let value_of_quarters := 10.00 in
  let value_of_nickels := 5.00 in
  let value_of_dimes := 5.00 in
  let value_of_pennies := 15.00 in
  let value_of_quarter := 0.25 in
  let value_of_nickel := 0.05 in
  let value_of_dime := 0.10 in
  let value_of_penny := 0.01 in
  let num_quarters := value_of_quarters / value_of_quarter in
  let num_nickels := value_of_nickels / value_of_nickel in
  let num_dimes := value_of_dimes / value_of_dime in
  let num_pennies := value_of_pennies / value_of_penny in
  let total_coins := num_quarters + num_nickels + num_dimes + num_pennies in
  num_quarters / total_coins = 40 / 1690 := by
  sorry

end probability_of_selecting_quarter_l505_505901


namespace holes_remaining_unfilled_l505_505058

def total_holes : ℕ := 8
def filled_percentage : ℝ := 0.75

theorem holes_remaining_unfilled : total_holes - (filled_percentage * total_holes).to_nat = 2 :=
by
  sorry

end holes_remaining_unfilled_l505_505058


namespace problem_proof_l505_505611

def A (n : ℕ) : ℕ := sorry  -- the definition of A must be provided, currently placeholder

theorem problem_proof (n : ℕ) (a : ℕ → ℕ) (h₁ : n ≠ 0)
(h₂ : A n ∈ ℕ) (h₃ : A n ^ 3 / 6 = n)
(h₄ : (2 - 1) ^ n = 81)
(h₅ : (2 - x)^n = ∑ i in range (n + 1), a i * x^i) :
a 0 - a 1 + a 2 - a 3 + (-1)^n * a n = 81 := 
by
  sorry

end problem_proof_l505_505611


namespace fulfill_customer_order_in_nights_l505_505865

structure JerkyCompany where
  batch_size : ℕ
  nightly_batches : ℕ

def customerOrder (ordered : ℕ) (current_stock : ℕ) : ℕ :=
  ordered - current_stock

def batchesNeeded (required : ℕ) (batch_size : ℕ) : ℕ :=
  required / batch_size

def daysNeeded (batches_needed : ℕ) (nightly_batches : ℕ) : ℕ :=
  batches_needed / nightly_batches

theorem fulfill_customer_order_in_nights :
  ∀ (ordered current_stock : ℕ) (jc : JerkyCompany),
    jc.batch_size = 10 →
    jc.nightly_batches = 1 →
    ordered = 60 →
    current_stock = 20 →
    daysNeeded (batchesNeeded (customerOrder ordered current_stock) jc.batch_size) jc.nightly_batches = 4 :=
by
  intros ordered current_stock jc h1 h2 h3 h4
  sorry

end fulfill_customer_order_in_nights_l505_505865


namespace citizen_income_l505_505881

noncomputable def tax_income (income first_threshold remaining_tax_rate total_tax first_threshold_tax_rate : ℝ) : ℝ :=
  let first_tax := income.min first_threshold * first_threshold_tax_rate
  let remaining_tax := (income - first_threshold).max 0 * remaining_tax_rate
  first_tax + remaining_tax

theorem citizen_income {I : ℝ} :
  tax_income I 40000 0.20 8000 0.14 = 8000 → I = 52000 := 
begin
  intros h,
  sorry
end

end citizen_income_l505_505881


namespace prime_factor_count_l505_505125

theorem prime_factor_count
  (a b p q : ℕ)
  (h1 : 2 * log 10 a + log 10 (Nat.gcd a b) = 100)
  (h2 : log 10 b + 2 * log 10 (Nat.lcm a b) = 450)
  (hp : p = PrimeFactorsCount a)
  (hq : q = PrimeFactorsCount b) :
  2 * p + 3 * q = 976 :=
sorry

end prime_factor_count_l505_505125


namespace circle_equation_proof_l505_505563

/-
Given:
1. Circle C: (x+2)^2 + y^2 = 4
2. Two perpendicular lines l_1 and l_2 both pass through point A(2, 0).
3. Circle with center M(1, m) (m > 0) is tangent to circle C and also tangent to lines l_1 and l_2.
Prove:
The equation of circle M is (x-1)^2 + (y-\sqrt{7})^2 = 4.
-/

theorem circle_equation_proof : 
  ∀ (m : ℝ), (m > 0) → 
  (1 - 2) ^ 2 + m ^ 2 = 2 * ((2 + r)^2) ∧
  (1 + 2) ^ 2 + m ^ 2 = (2 + r) ^ 2 →
  (∃ r : ℝ, (r = 2) ∧ m = real.sqrt 7 ∧ 
  ∀ x y : ℝ, ((x - 1) ^ 2 + (y - real.sqrt 7) ^ 2 = 4)) := 
begin
  sorry
end

end circle_equation_proof_l505_505563


namespace least_positive_integer_with_12_factors_l505_505335

def has_exactly_12_factors (n : ℕ) : Prop :=
  n.factors.length = 12

theorem least_positive_integer_with_12_factors : ∃ k : ℕ, has_exactly_12_factors k ∧ (∀ n : ℕ, has_exactly_12_factors n → n ≥ k) ∧ k = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_l505_505335


namespace ff_even_of_f_even_l505_505041

-- Define what it means for a function f to be even.
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f (x)

-- Theorem to prove that f(f(x)) is even if f is even.
theorem ff_even_of_f_even (f : ℝ → ℝ) (hf : is_even_function f) : is_even_function (f ∘ f) :=
by
  intros x,
  specialize hf x,
  specialize hf (-x),
  rw [←hf, hf],
  sorry

end ff_even_of_f_even_l505_505041


namespace union_of_sets_eq_l505_505995

variable (M N : Set ℕ)

theorem union_of_sets_eq (h1 : M = {1, 2}) (h2 : N = {2, 3}) : M ∪ N = {1, 2, 3} := by
  sorry

end union_of_sets_eq_l505_505995


namespace least_positive_integer_with_12_factors_l505_505301

def has_exactly_12_factors (n : ℕ) : Prop :=
  (finset.range(n + 1).filter (λ d, n % d = 0)).card = 12

theorem least_positive_integer_with_12_factors : ∀ n : ℕ, has_exactly_12_factors n → n ≥ 150 :=
sorry

end least_positive_integer_with_12_factors_l505_505301


namespace least_positive_integer_with_12_factors_is_96_l505_505399

def has_exactly_12_factors (n : ℕ) : Prop :=
  (12 : ℕ) = (List.range (n + 1)).filter (λ x, n % x = 0).length

theorem least_positive_integer_with_12_factors_is_96 :
  n ∈ (ℕ) ∧ has_exactly_12_factors n → n = 96 :=
by
  sorry

end least_positive_integer_with_12_factors_is_96_l505_505399


namespace find_a_l505_505549

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 + a / ((Real.exp (2 * x)) - 1)

theorem find_a : ∃ a : ℝ, (∀ x : ℝ, f a x = -f a (-x)) → a = 2 :=
by
  sorry

end find_a_l505_505549


namespace four_digit_multiples_of_5_l505_505648

theorem four_digit_multiples_of_5 : 
  let f := fun n => 1000 <= n ∧ n <= 9999 ∧ n % 5 = 0
  in finset.card (finset.filter f (finset.range 10000)) = 1800 :=
by
  sorry

end four_digit_multiples_of_5_l505_505648


namespace largest_angle_of_triangle_l505_505718

-- Define the conditions of the problem
variables {x y z : ℝ}

-- Conditions given in the problem
def condition1 (x y z : ℝ) : Prop := x + 3 * y + 4 * z = x^2
def condition2 (x y z : ℝ) : Prop := x + 3 * y - 4 * z = -7

-- Define the theorem to prove
theorem largest_angle_of_triangle (x y z : ℝ) (h1 : condition1 x y z) (h2 : condition2 x y z) : 
  ∃ (Z : ℝ), Z = 120 ∧ is_largest_angle_of_triangle_z x y z Z :=
sorry   -- Placeholder for the actual proof

-- Predicate to check if Z is the largest angle
def is_largest_angle_of_triangle_z (x y z angle : ℝ) : Prop := 
  ∃ (cosZ : ℝ), cosZ = -1 / 2 ∧ angle = real.acos cosZ

end largest_angle_of_triangle_l505_505718


namespace part_a_part_b_l505_505832

-- Mathematical definitions and conditions
def is_rearranged (a b : ℕ) : Prop :=
  ∃ (σ : (List ℕ) → (List ℕ)), a.digits 10 = σ (b.digits 10) ∧ 
    ∀ (x : ℕ), x ∈ a.digits 10 ↔ x ∈ b.digits 10

def sum_of_digits (n : ℕ) : ℕ := (n.digits 10).sum

-- Prove Part (a)
theorem part_a (a b : ℕ) (h : is_rearranged a b) : 
  sum_of_digits (2 * a) = sum_of_digits (2 * b) :=
sorry

-- Prove Part (b)
theorem part_b (a b : ℕ) (h : is_rearranged a b) (ha : even a) (hb : even b) : 
  sum_of_digits (a / 2) = sum_of_digits (b / 2) :=
sorry

end part_a_part_b_l505_505832


namespace always_two_real_roots_find_m_l505_505601

-- Given quadratic equation: x^2 - 4mx + 3m^2 = 0
-- Definitions for the problem
def quadratic_eq (m x : ℝ) : Prop := x^2 - 4 * m * x + 3 * m^2 = 0

-- Q1: Prove that this equation always has two real roots.
theorem always_two_real_roots (m : ℝ) : ∃ x₁ x₂ : ℝ, quadratic_eq m x₁ ∧ quadratic_eq m x₂ :=
by
  sorry

-- Q2: If m > 0 and the difference between the two real roots is 2, find the value of m.
theorem find_m (m : ℝ) (h₁ : m > 0) (h₂ : ∃ x₁ x₂ : ℝ, quadratic_eq m x₁ ∧ quadratic_eq m x₂ ∧ |x₁ - x₂| = 2) : m = 1 :=
by
  sorry

end always_two_real_roots_find_m_l505_505601


namespace vacant_seats_l505_505707

theorem vacant_seats (total_seats filled_percentage : ℕ) (h_filled_percentage : filled_percentage = 62) (h_total_seats : total_seats = 600) : 
  (total_seats - total_seats * filled_percentage / 100) = 228 :=
by
  sorry

end vacant_seats_l505_505707


namespace quadratic_inequality_l505_505697

theorem quadratic_inequality (a : ℝ) (h : ∀ x : ℝ, a * x^2 + 2 * x + 1 > 0) : a ≥ 1 :=
sorry

end quadratic_inequality_l505_505697


namespace ratio_of_squares_l505_505914

def square_inscribed_triangle_1 (x : ℝ) : Prop :=
  ∃ (a b c : ℝ), 
  a = 6 ∧ b = 8 ∧ c = 10 ∧
  x = 24 / 7

def square_inscribed_triangle_2 (y : ℝ) : Prop :=
  ∃ (a b c : ℝ), 
  a = 6 ∧ b = 8 ∧ c = 10 ∧
  y = 10 / 3

theorem ratio_of_squares (x y : ℝ) 
  (hx : square_inscribed_triangle_1 x) 
  (hy : square_inscribed_triangle_2 y) : 
  x / y = 36 / 35 := 
by sorry

end ratio_of_squares_l505_505914


namespace least_positive_integer_with_12_factors_is_96_l505_505403

def has_exactly_12_factors (n : ℕ) : Prop :=
  (12 : ℕ) = (List.range (n + 1)).filter (λ x, n % x = 0).length

theorem least_positive_integer_with_12_factors_is_96 :
  n ∈ (ℕ) ∧ has_exactly_12_factors n → n = 96 :=
by
  sorry

end least_positive_integer_with_12_factors_is_96_l505_505403


namespace compute_sum_l505_505940

/-- A lemma to simplify the given sum. -/
lemma sum_simplify (n : ℕ) (h : n ≥ 2) : 
  let a := n^3.root (n - 1)
      b := (n - 1)^3.root n in
  (1 / (n * a + (n - 1) * b)) =
  (1 / a - 1 / b) := 
by
  sorry

/-- The sum from 2 to 100 of 1 / (n * (n-1)^(1/3) + (n-1) * n^(1/3)) equals 9/10. -/
theorem compute_sum : 
  ∑ n in Finset.range (99) + 2, 
    (1 / (n * Real.cbrt (n - 1 : ℝ) + (n - 1) * Real.cbrt n)) = (9 / 10) :=
by
  sorry

end compute_sum_l505_505940


namespace probability_greater_than_5_l505_505409

noncomputable def X : finset ℕ := {1, 2, 3, 4, 5, 6}

def P (A : set ℕ) : ℝ :=
  (A.to_finset.card : ℝ) / X.card

theorem probability_greater_than_5 : P {n : ℕ | n > 5} = 1 / 6 :=
by
  sorry

end probability_greater_than_5_l505_505409


namespace find_z_coordinate_l505_505903

-- Definitions of the initial points and direction vector
def p1 : ℝ × ℝ × ℝ := (2, 2, 1)
def p2 : ℝ × ℝ × ℝ := (5, 1, -2)
def dir : ℝ × ℝ × ℝ := (5 - 2, 1 - 2, -2 - 1)
def param_line (t : ℝ) : ℝ × ℝ × ℝ := (2 + 3 * t, 2 - t, 1 - 3 * t)

-- The main theorem
theorem find_z_coordinate (x_val : ℝ) (z_val : ℝ) (t_val : ℝ) :
  (param_line t_val).1 = x_val → -- x-coordinate condition
  x_val = 4 →                  -- given x-coordinate of the point is 4
  (param_line t_val).3 = z_val → -- verifying z-coordinate
  z_val = -1 := 
by {
  intros h1 hx h2,
  rw [h1, hx],
  unfold param_line,
  sorry
}

end find_z_coordinate_l505_505903


namespace longest_side_of_enclosure_l505_505866

theorem longest_side_of_enclosure
  (l w : ℝ)
  (h1 : 2 * l + 2 * w = 180)
  (h2 : l * w = 1440) :
  l = 72 ∨ w = 72 :=
by {
  sorry
}

end longest_side_of_enclosure_l505_505866


namespace min_value_expression_l505_505743

variable (a b c : ℝ)
variable (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
variable (h_eq : a * b * c = 64)

theorem min_value_expression :
  a^2 + 6 * a * b + 9 * b^2 + 3 * c^2 ≥ 192 :=
by {
  sorry
}

end min_value_expression_l505_505743


namespace sum_of_valid_three_digit_numbers_l505_505955

theorem sum_of_valid_three_digit_numbers :
  let valid_sum := ∑ n in finset.filter (λ n,
    let a := n / 100,
        b := (n / 10) % 10,
        c := n % 10,
        S := a + b + c,
        P := a * b * c
    in n % S = 0 ∧ n % P = 0) (finset.range 900).filter (λ n, 100 ≤ n ∧ n < 1000 ∧ (n / 100 ≠ 0) ∧ ((n / 10) % 10 ≠ 0) ∧ (n % 10 ≠ 0)),
  true :=
by { sorry }

end sum_of_valid_three_digit_numbers_l505_505955


namespace Justine_colored_sheets_l505_505852

theorem Justine_colored_sheets :
  ∀ (total_sheets : ℕ) (binders : ℕ) (colored_fraction : ℚ),
  total_sheets = 2450 →
  binders = 5 →
  colored_fraction = 1 / 2 →
  total_sheets / binders * colored_fraction = 245 := by
  intros total_sheets binders colored_fraction
  intros h1 h2 h3
  have h_total_binders : total_sheets / binders = 490 := by
    rw [h1, h2]
    exact Nat.div_eq_of_eq_mul_right (by norm_num) (by norm_num)
  have h_colored_sheets : 490 * colored_fraction = 245 := by
    rw h3
    exact (by norm_num)
  rw [h_total_binders, h_colored_sheets]
  exact (by norm_num)

end Justine_colored_sheets_l505_505852


namespace sequence_x_y_sum_l505_505962

theorem sequence_x_y_sum :
  ∃ (r x y : ℝ), 
    (r * 3125 = 625) ∧ 
    (r * 625 = 125) ∧ 
    (r * 125 = x) ∧ 
    (r * x = y) ∧ 
    (r * y = 1) ∧
    (r * 1 = 1/5) ∧ 
    (r * (1/5) = 1/25) ∧ 
    x + y = 30 := 
by
  -- A placeholder for the actual proof
  sorry

end sequence_x_y_sum_l505_505962


namespace least_positive_integer_with_12_factors_l505_505341

theorem least_positive_integer_with_12_factors : ∃ k : ℕ, k > 0 ∧ (∀ n : ℕ, n > 0 → (factors_count n = 12 → k ≤ n)) ∧ k = 72 :=
by {
  sorry,
}

end least_positive_integer_with_12_factors_l505_505341


namespace least_positive_integer_with_12_factors_l505_505250

theorem least_positive_integer_with_12_factors :
  ∃ k : ℕ, (1 ≤ k) ∧ (nat.factors k).length = 12 ∧
    ∀ n : ℕ, (1 ≤ n) ∧ (nat.factors n).length = 12 → k ≤ n := 
begin
  sorry
end

end least_positive_integer_with_12_factors_l505_505250


namespace hands_with_four_identical_cards_l505_505145

theorem hands_with_four_identical_cards (cards : Finset (Fin 52)) 
  (values : Finset (Fin 13)) 
  (suits : Finset (Fin 4)) 
  (hand : Finset (Fin 5)) 
  (h_distinct : ∀ (v : Fin 13), ∀ (s : Finset (Fin 4)), s.card = 4) :
  (∃ c : Finset (Fin 52), c.card = 5 ∧
    ∃ v : Fin 13, ∃ s1 s2 s3 s4 : Fin 4, c = {⟨v.1, s1⟩, ⟨v.1, s2⟩, ⟨v.1, s3⟩, ⟨v.1, s4⟩} ∧ ∃ k, k ≠ v ∧ k ∈ values) →
  ∃ n : ℕ, n = 13 * 48 →
  n = 624 :=
by {
  sorry
}

end hands_with_four_identical_cards_l505_505145


namespace least_positive_integer_with_12_factors_is_72_l505_505318

-- Definition of having exactly 12 factors for a positive integer
def has_exactly_12_factors (n : ℕ) : Prop := n > 0 ∧ (finset.filter (λ d, n % d = 0) (finset.range (n + 1))).card = 12

-- Statement of the problem in Lean 4
theorem least_positive_integer_with_12_factors_is_72 :
  ∃ n : ℕ, has_exactly_12_factors n ∧ ∀ m : ℕ, has_exactly_12_factors m → n ≤ m :=
begin
  use 72,
  split,
  { -- Prove that 72 has exactly 12 factors
    sorry },
  { -- Prove that 72 is the least such integer
    sorry }
end

end least_positive_integer_with_12_factors_is_72_l505_505318


namespace tangents_distance_property_l505_505994

noncomputable def distance_from_point_to_line
(M A B C : Point) (a b : ℝ)
(h_M : is_outside_circle M circle)
(h_A_tangent : is_tangent MA circle A)
(h_B_tangent : is_tangent MB circle B)
(h_C_on_circle : C ∈ circle)
(h_C_MA_dist : perpendicular_distance C MA = a)
(h_C_MB_dist : perpendicular_distance C MB = b) : ℝ :=
sqrt (a * b)

theorem tangents_distance_property
(M A B C : Point) (a b : ℝ)
(h_M : is_outside_circle M circle)
(h_A_tangent : is_tangent MA circle A)
(h_B_tangent : is_tangent MB circle B)
(h_C_on_circle : C ∈ circle)
(h_C_MA_dist : perpendicular_distance C MA = a)
(h_C_MB_dist : perpendicular_distance C MB = b) :
distance_from_point_to_line M A B C a b = sqrt (a * b) := by sorry

end tangents_distance_property_l505_505994


namespace range_of_m_l505_505031

noncomputable def set_A := { x : ℝ | x^2 + x - 6 = 0 }
noncomputable def set_B (m : ℝ) := { x : ℝ | m * x + 1 = 0 }

theorem range_of_m (m : ℝ) : set_A ∪ set_B m = set_A → m = 0 ∨ m = -1 / 2 ∨ m = 1 / 3 :=
by
  sorry

end range_of_m_l505_505031


namespace least_positive_integer_with_12_factors_l505_505355

theorem least_positive_integer_with_12_factors : ∃ k : ℕ, k > 0 ∧ (∀ n : ℕ, n > 0 → (factors_count n = 12 → k ≤ n)) ∧ k = 72 :=
by {
  sorry,
}

end least_positive_integer_with_12_factors_l505_505355


namespace fifth_flower_is_e_l505_505408

def flowers : List String := ["a", "b", "c", "d", "e", "f", "g"]

theorem fifth_flower_is_e : flowers.get! 4 = "e" := sorry

end fifth_flower_is_e_l505_505408


namespace max_side_range_of_triangle_l505_505561

-- Define the requirement on the sides a and b
def side_condition (a b : ℝ) : Prop :=
  |a - 3| + (b - 7)^2 = 0

-- Prove the range of side c
theorem max_side_range_of_triangle (a b c : ℝ) (h : side_condition a b) (hc : c = max a (max b c)) :
  7 ≤ c ∧ c < 10 :=
sorry

end max_side_range_of_triangle_l505_505561


namespace meet_at_35_l505_505142

def walking_distance_A (t : ℕ) := 5 * t

def walking_distance_B (t : ℕ) := (t * (7 + t)) / 2

def total_distance (t : ℕ) := walking_distance_A t + walking_distance_B t

theorem meet_at_35 : ∃ (t : ℕ), total_distance t = 100 ∧ walking_distance_A t - walking_distance_B t = 35 := by
  sorry

end meet_at_35_l505_505142


namespace least_positive_integer_with_12_factors_l505_505386

theorem least_positive_integer_with_12_factors : 
  ∃ (n : ℕ), (n > 0) ∧ (nat.factors_count n = 12) ∧ (∀ m, (m > 0) → (nat.factors_count m = 12) → n ≤ m) ∧ n = 108 :=
sorry

end least_positive_integer_with_12_factors_l505_505386


namespace P_2021_squared_l505_505752

noncomputable def P (x : ℕ) : ℕ := 
  -- Definition ignored for now as focus is on the final theorem statement
sorry

theorem P_2021_squared :
  -- Main theorem statement capturing the mathematical problem and solution
  -- Given conditions on the polynomial P
  (∀ k : ℕ, k ≤ 2020 → P(k^2) = k) ∧ degree P ≤ 2020 → 
  P(2021^2) = 2021 - Nat.binom 4040 2020 :=
sorry

end P_2021_squared_l505_505752


namespace melody_sequence_count_correct_l505_505110

def melody_sequence_count : ℕ :=
  let N := 3 * 2^18 in
  100 * 3 + 18

theorem melody_sequence_count_correct :
  melody_sequence_count = 318 :=
by
  sorry

end melody_sequence_count_correct_l505_505110


namespace least_positive_integer_with_12_factors_l505_505199

/--
Prove that the smallest positive integer with exactly 12 positive factors is 60.
-/
theorem least_positive_integer_with_12_factors : 
  ∃ (n : ℕ), (0 < n) ∧ (nat.factors_count n = 12) ∧ (∀ m : ℕ, (0 < m) ∧ (nat.factors_count m = 12) → n ≤ m) → n = 60 :=
sorry

end least_positive_integer_with_12_factors_l505_505199


namespace least_positive_integer_with_12_factors_l505_505286

def num_factors (n : ℕ) : ℕ := 
  (finset.range (n + 1)).filter (λ d, n % d = 0).card

theorem least_positive_integer_with_12_factors : 
  ∃ n : ℕ, num_factors n = 12 ∧ n = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_l505_505286


namespace production_relationship_l505_505897

noncomputable def production_function (a : ℕ) (p : ℝ) (x : ℕ) : ℝ := a * (1 + p / 100)^x

theorem production_relationship (a : ℕ) (p : ℝ) (m : ℕ) (x : ℕ) (hx : 0 ≤ x ∧ x ≤ m) :
  production_function a p x = a * (1 + p / 100)^x := by
  sorry

end production_relationship_l505_505897


namespace least_positive_integer_with_12_factors_is_96_l505_505396

def has_exactly_12_factors (n : ℕ) : Prop :=
  (12 : ℕ) = (List.range (n + 1)).filter (λ x, n % x = 0).length

theorem least_positive_integer_with_12_factors_is_96 :
  n ∈ (ℕ) ∧ has_exactly_12_factors n → n = 96 :=
by
  sorry

end least_positive_integer_with_12_factors_is_96_l505_505396


namespace binomial_product_9_2_7_2_l505_505941

theorem binomial_product_9_2_7_2 : Nat.choose 9 2 * Nat.choose 7 2 = 756 := by
  sorry

end binomial_product_9_2_7_2_l505_505941


namespace son_age_is_8_l505_505135

-- Definitions based on the conditions
def father_age (S : ℕ) : ℕ := 6 * S

def age_sum (F S : ℕ) : ℕ := (F + 6) + (S + 6)

-- The main theorem: proving the age of the son is 8 years
theorem son_age_is_8 : ∃ S : ℕ, father_age S = 6 * S ∧ age_sum (father_age S) S = 68 ∧ S = 8 := 
by 
  have hF: ∀ S, father_age S = 6 * S := by intro S; rfl
  have h_sum: ∀ S, age_sum (father_age S) S = (father_age S + 6) + (S + 6) := by intro S; rfl
  exists 8
  apply And.intro
  apply hF
  apply And.intro
  calc
    age_sum (father_age 8) 8
        = (father_age 8 + 6) + (8 + 6) : by apply h_sum 
    ... = (6 * 8 + 6) + (8 + 6) : by rw hF 
    ... = 48 + 6 + 8 + 6 : by norm_num
    ... = 68 : by norm_num
  rfl

end son_age_is_8_l505_505135


namespace least_positive_integer_with_12_factors_is_96_l505_505269

theorem least_positive_integer_with_12_factors_is_96 :
  ∃ n : ℕ, (n = 96 ∧ (∀ m : ℕ, (m > 0 ∧ ∀ d : ℕ, d ∣ m → d > 0 → d < m → (∃ k : ℕ, d = k ∨ k * d = m) → k > 1 → m > n))) ∧ (∃ m : ℕ, m = 12 ∧ (∀ d : ℕ, d ∣ m → d ∈ {1, m} ∨ ∀ k : ℕ, k * d = m → d = k ∨ d < k)).
sorry

end least_positive_integer_with_12_factors_is_96_l505_505269


namespace sum_of_squares_of_chords_l505_505913

noncomputable def surface_area_sphere (r : ℝ) : ℝ := 4 * π * r^2

theorem sum_of_squares_of_chords (r MA MB MC : ℝ) (h1 : surface_area_sphere r = 4 * π) 
  (h2 : r ^ 2 = 1) (h3 : MA = r) (h4 : MB = r) (h5 : MC = r) : 
  MA^2 + MB^2 + MC^2 = 4 := 
by 
  sorry

end sum_of_squares_of_chords_l505_505913


namespace least_positive_integer_with_12_factors_l505_505170

theorem least_positive_integer_with_12_factors :
  ∃ m : ℕ, 0 < m ∧ (∀ n : ℕ, 0 < n → (n ≠ m →  m ≤ n)) ∧ (12 = (m.factors.toFinset.card)) := sorry

end least_positive_integer_with_12_factors_l505_505170


namespace quadratic_has_real_roots_find_value_of_m_l505_505596

theorem quadratic_has_real_roots (m : ℝ) : 
  ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (x1 - x2) ≠ 0 ∧ (x^2 - 4 * m * x + 3 * m^2 = 0) := 
by 
  sorry

theorem find_value_of_m (m : ℝ) (h1 : m > 0) (h2 : ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (x1 - x2 = 2)) :
  m = 1 := 
by 
  sorry

end quadratic_has_real_roots_find_value_of_m_l505_505596


namespace least_positive_integer_with_12_factors_l505_505207

/--
Prove that the smallest positive integer with exactly 12 positive factors is 60.
-/
theorem least_positive_integer_with_12_factors : 
  ∃ (n : ℕ), (0 < n) ∧ (nat.factors_count n = 12) ∧ (∀ m : ℕ, (0 < m) ∧ (nat.factors_count m = 12) → n ≤ m) → n = 60 :=
sorry

end least_positive_integer_with_12_factors_l505_505207


namespace rectangle_area_at_stage_8_l505_505685

-- Definitions based on conditions
def area_of_square (side_length : ℕ) : ℕ := side_length * side_length
def number_of_squares_in_stage (stage : ℕ) : ℕ := stage

-- The main theorem to prove
theorem rectangle_area_at_stage_8 : 
  area_of_square 4 * number_of_squares_in_stage 8 = 128 := by
  sorry

end rectangle_area_at_stage_8_l505_505685


namespace minimum_number_of_wizards_l505_505123

-- Define our individuals and their respective statuses
def Individual := { coins : ℕ // 1 ≤ coins ∧ coins ≤ 10 }
def is_wizard (x : Individual) : Prop := sorry  -- Define property to check if the individual is wizard
def is_elf (x : Individual) : Prop := sorry  -- Define property to check if the individual is elf

-- Define the total number of individuals
def individuals : List Individual := sorry  -- Define our list of individuals

-- Sum of reported coins must be equal to 36
def sum_reported_coins (l : List Individual) : ℕ := List.sum (l.map (fun x => x.val))
axiom sum_coins_condition : sum_reported_coins individuals = 36

-- There are exactly 10 individuals
axiom total_individuals : individuals.length = 10

-- Prove that there are at least 5 wizards
theorem minimum_number_of_wizards : ∃ W : ℕ, W ≥ 5 ∧ ∀ S : ℕ, (S + W = 10) → 
  let coins_truth_sum := 55 in
  coins_truth_sum - (55 - 36) <= 36 :=
by sorry

end minimum_number_of_wizards_l505_505123


namespace least_positive_integer_with_12_factors_is_72_l505_505217

open Nat

def hasExactly12Factors (n : ℕ) : Prop :=
  (divisors n).length = 12

theorem least_positive_integer_with_12_factors_is_72 :
  ∃ n : ℕ, hasExactly12Factors n ∧ ∀ m : ℕ, hasExactly12Factors m → n ≤ m → n = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_is_72_l505_505217


namespace convert_micrometers_to_meters_l505_505819

theorem convert_micrometers_to_meters :
  ∃ (a : ℝ), (32 * 10 ^ (-6)) = a * 10 ^ (-6) :=
begin
  use 32,
  sorry,
end

end convert_micrometers_to_meters_l505_505819


namespace percentage_of_first_relative_to_second_l505_505695

theorem percentage_of_first_relative_to_second (X : ℝ) 
  (first_number : ℝ := 8/100 * X) 
  (second_number : ℝ := 16/100 * X) :
  (first_number / second_number) * 100 = 50 := 
sorry

end percentage_of_first_relative_to_second_l505_505695


namespace no_calls_in_year_2017_l505_505770

theorem no_calls_in_year_2017 :
  let days_in_year := 365
  let calls_every_2_days := List.range (days_in_year // 2 + 1) |>.map (· * 2)
  let calls_every_5_days := List.range (days_in_year // 5 + 1) |>.map (· * 5)
  let calls_every_7_days := List.range (days_in_year // 7 + 1) |>.map (· * 7)
  let union_calls := calls_every_2_days.to_finset ∪ calls_every_5_days.to_finset ∪ calls_every_7_days.to_finset
  days_in_year - union_calls.card = 125 := 
by
  sorry

end no_calls_in_year_2017_l505_505770


namespace sqrt_distances_form_triangle_and_area_l505_505474

variable {ABC : Type} [triangle ABC]
variable (I : ABC) [incenter I]
variable {r : ℝ} [inradius I r]
variable {P : Point ABC} [in_circle I P]
variables {d1 d2 d3 : ℝ} [distance_to_line P d1 d2 d3]

theorem sqrt_distances_form_triangle_and_area 
  (h1 : ∀ (ABC : Type) [triangle ABC], incenter I) 
  (h2 : ∀ (r : ℝ) [inradius I r])
  (h3 : ∀ (P : Point ABC) [in_circle I P])
  (h4 : ∀ {d1 d2 d3 : ℝ} [distance_to_line P d1 d2 d3]): 
  (∃ (a b c : ℝ), (a = sqrt d1) ∧ (b = sqrt d2) ∧ (c = sqrt d3) ∧ 
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b) ∧ 
  area_triangle a b c = (sqrt 3 / 4) * sqrt (r^2 - (dist I P)^2)) :=
sorry

end sqrt_distances_form_triangle_and_area_l505_505474


namespace least_positive_integer_with_12_factors_l505_505325

def has_exactly_12_factors (n : ℕ) : Prop :=
  n.factors.length = 12

theorem least_positive_integer_with_12_factors : ∃ k : ℕ, has_exactly_12_factors k ∧ (∀ n : ℕ, has_exactly_12_factors n → n ≥ k) ∧ k = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_l505_505325


namespace trajectory_of_P_equal_area_triangles_l505_505001

-- Conditions provided in problem
def symmetric (A B O : Point) : Prop :=
  O.x = (A.x + B.x) / 2 ∧ O.y = (A.y + B.y) / 2

def slopes_condition (A B P : Point) : Prop :=
  let slope_AP := (P.y - A.y) / (P.x - A.x)
  let slope_BP := (P.y - B.y) / (P.x - B.x)
  slope_AP * slope_BP = -1 / 3

-- Definitions for points and their symmetric relationships
structure Point where
  x : ℝ
  y : ℝ

noncomputable def O : Point := ⟨0, 0⟩
noncomputable def A : Point := ⟨-1, 1⟩
noncomputable def B : Point := ⟨1, -1⟩ 

-- The trajectory equation of point P
def trajectory (P : Point) : Prop :=
  P.x ^ 2 + 3 * P.y ^ 2 = 4 ∧ P.x ≠ 1 ∧ P.x ≠ -1 

-- Proposition to be proven in Lean 4.
theorem trajectory_of_P (P : Point) (h1 : symmetric A B O) (h2 : slopes_condition A B P) : 
  trajectory P :=
sorry

-- Existence of P such that the areas of triangles PAB and PMN are equal
-- This excerpt will ensure the existence of point P with the given coordinates
theorem equal_area_triangles (x0 y0 : ℝ) (hx0 : x0 = 5 / 3) (hy0 : y0 = sqrt 33 / 9 ∨ y0 = - sqrt 33 / 9)
  (hx : P.x = x0) (hy : P.y = y0) : 
  ∃ P : Point, trajectory P ∧ triangle_area_equal_condition A B P :=
sorry

end trajectory_of_P_equal_area_triangles_l505_505001


namespace cube_tower_remainder_l505_505895

theorem cube_tower_remainder :
  let T := 64
  in T % 100 = 64 := 
by
  -- Given conditions
  -- let k ∈ {2, 3, 4, 5, 6, 7, 8}
  -- Each cube with edge-length \( k \) can be used to build towers adhering to the rule:
  -- The cube immediately on top of a cube with edge-length \( k \) must have edge-length at most \( k + 1 \)
  -- Calculation of T follows the recursive pattern:
  -- T_m = 2 * T_{m-1} for m from 3 to 8, starting with T_2 = 1
  sorry

end cube_tower_remainder_l505_505895


namespace least_positive_integer_with_12_factors_is_72_l505_505213

open Nat

def hasExactly12Factors (n : ℕ) : Prop :=
  (divisors n).length = 12

theorem least_positive_integer_with_12_factors_is_72 :
  ∃ n : ℕ, hasExactly12Factors n ∧ ∀ m : ℕ, hasExactly12Factors m → n ≤ m → n = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_is_72_l505_505213


namespace least_positive_integer_with_12_factors_l505_505387

theorem least_positive_integer_with_12_factors : 
  ∃ (n : ℕ), (n > 0) ∧ (nat.factors_count n = 12) ∧ (∀ m, (m > 0) → (nat.factors_count m = 12) → n ≤ m) ∧ n = 108 :=
sorry

end least_positive_integer_with_12_factors_l505_505387


namespace julia_money_left_l505_505729

def initial_amount : ℕ := 40

def amount_spent_on_game (initial : ℕ) : ℕ := initial / 2

def amount_left_after_game (initial : ℕ) (spent_game : ℕ) : ℕ := initial - spent_game

def amount_spent_on_in_game (left_after_game : ℕ) : ℕ := left_after_game / 4

def final_amount (left_after_game : ℕ) (spent_in_game : ℕ) : ℕ := left_after_game - spent_in_game

theorem julia_money_left (initial : ℕ) 
  (h_init : initial = initial_amount)
  (spent_game : ℕ)
  (h_spent_game : spent_game = amount_spent_on_game initial)
  (left_after_game : ℕ)
  (h_left_after_game : left_after_game = amount_left_after_game initial spent_game)
  (spent_in_game : ℕ)
  (h_spent_in_game : spent_in_game = amount_spent_on_in_game left_after_game)
  : final_amount left_after_game spent_in_game = 15 := by 
  sorry

end julia_money_left_l505_505729


namespace multiples_of_5_in_4_digit_range_l505_505650

theorem multiples_of_5_in_4_digit_range : 
  let count_multiples := (9995 - 1000) / 5 + 1
  in count_multiples = 1800 :=
by
  sorry

end multiples_of_5_in_4_digit_range_l505_505650


namespace area_triangle_ADE_l505_505720

-- Define the triangle with its properties
variable (A B C D E : Type) [add_comm_group A] [module ℝ A]

-- Define points and distances
variables {a b c d e : A}
variable (h_eq_sides : dist a b = dist b c)
variable (h_ac : dist a c = 2)
variable (h_angle : ∠ a c b = pi / 6) -- 30 degrees in radians

-- Define bisector and median
variable (ae : line a e)
variable (ad : line a d)
variable (h_bisector : is_angle_bisector ae)
variable (h_median : is_median ad)

-- Prove the area of triangle ADE
theorem area_triangle_ADE :
  area (triangle a d e) = sqrt 3 / 6 :=
sorry

end area_triangle_ADE_l505_505720


namespace least_positive_integer_with_12_factors_is_72_l505_505224

open Nat

def hasExactly12Factors (n : ℕ) : Prop :=
  (divisors n).length = 12

theorem least_positive_integer_with_12_factors_is_72 :
  ∃ n : ℕ, hasExactly12Factors n ∧ ∀ m : ℕ, hasExactly12Factors m → n ≤ m → n = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_is_72_l505_505224


namespace solve_eq_integers_l505_505788

theorem solve_eq_integers (x y : ℤ) : 
    x^2 - x * y - 6 * y^2 + 2 * x + 19 * y = 18 ↔ (x = 2 ∧ y = 2) ∨ (x = -2 ∧ y = 2) := by
    sorry

end solve_eq_integers_l505_505788


namespace polynomial_condition_degree_n_l505_505494

open Polynomial

theorem polynomial_condition_degree_n 
  (P_n : ℤ[X]) (n : ℕ) (hn_pos : 0 < n) (hn_deg : P_n.degree = n) 
  (hx0 : P_n.eval 0 = 0)
  (hx_conditions : ∃ (a : ℤ) (b : Fin n → ℤ), ∀ i, P_n.eval (b i) = n) : 
  n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 := 
sorry

end polynomial_condition_degree_n_l505_505494


namespace unique_solution_of_system_l505_505509

theorem unique_solution_of_system (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z)
  (h1 : x * (x + y + z) = 26) (h2 : y * (x + y + z) = 27) (h3 : z * (x + y + z) = 28) :
  x = 26 / 9 ∧ y = 3 ∧ z = 28 / 9 :=
by
  sorry

end unique_solution_of_system_l505_505509


namespace least_positive_integer_with_12_factors_is_96_l505_505274

theorem least_positive_integer_with_12_factors_is_96 :
  ∃ n : ℕ, (n = 96 ∧ (∀ m : ℕ, (m > 0 ∧ ∀ d : ℕ, d ∣ m → d > 0 → d < m → (∃ k : ℕ, d = k ∨ k * d = m) → k > 1 → m > n))) ∧ (∃ m : ℕ, m = 12 ∧ (∀ d : ℕ, d ∣ m → d ∈ {1, m} ∨ ∀ k : ℕ, k * d = m → d = k ∨ d < k)).
sorry

end least_positive_integer_with_12_factors_is_96_l505_505274


namespace analytical_expression_solve_inequality_l505_505572

def f_analytical_expression (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = x^2 + 3 * x + 2 ∨ f x = x^2 - 3 * x + 2)

noncomputable def f_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

noncomputable def f_defined_positive (f : ℝ → ℝ) : Prop :=
  ∀ x, x ≥ 0 → f x = x^2 + 3 * x + 2

theorem analytical_expression (f : ℝ → ℝ) 
  (h_even : f_even f) 
  (h_defined_positive : f_defined_positive f) :
  f_analytical_expression f :=
sorry

noncomputable def f_expression (x : ℝ) : ℝ :=
if x ≥ 0 then x^2 + 3 * x + 2 else x^2 - 3 * x + 2

theorem solve_inequality :
  (∀ x, f_expression x = x^2 + 3 * x + 2 ∨ f_expression x = x^2 - 3 * x + 2) →
  set_of (λ x, f_expression (2 * x - 1) < 20) = (set.Ioo (-1 : ℝ) (2 : ℝ)) :=
sorry

end analytical_expression_solve_inequality_l505_505572


namespace artemon_distance_covered_l505_505812

-- Definition of the problem conditions
def rectangle_side_a : ℝ := 6
def rectangle_side_b : ℝ := 2.5
def malvina_speed : ℝ := 4
def buratino_speed : ℝ := 6
def artemon_speed : ℝ := 12

-- Computation of the diagonal distance
def diagonal_distance : ℝ :=
  real.sqrt (rectangle_side_a ^ 2 + rectangle_side_b ^ 2)

-- Computation of the relative speed and the time taken for Malvina and Buratino to meet
def relative_speed : ℝ :=
  malvina_speed + buratino_speed

def meeting_time : ℝ :=
  diagonal_distance / relative_speed

-- Computation of the distance covered by Artemon
def artemon_distance : ℝ :=
  artemon_speed * meeting_time

-- The theorem that we need to prove
theorem artemon_distance_covered : artemon_distance = 7.8 := 
sorry

end artemon_distance_covered_l505_505812


namespace least_positive_integer_with_12_factors_l505_505347

theorem least_positive_integer_with_12_factors : ∃ k : ℕ, k > 0 ∧ (∀ n : ℕ, n > 0 → (factors_count n = 12 → k ≤ n)) ∧ k = 72 :=
by {
  sorry,
}

end least_positive_integer_with_12_factors_l505_505347


namespace pentagon_operation_terminates_l505_505858

theorem pentagon_operation_terminates
  (a b c d e : ℤ)
  (h_sum : a + b + c + d + e > 0)
  (h_op : ∀ x y z, y < 0 → ∃ a' b' c', (a', b', c') = (x + y, -y, z + y)) :
  ∃ n : ℕ, ∀ a b c d e : ℤ, (a + b + c + d + e > 0) → 
    (∃ k ≤ n, ∀ m ≥ k, ∀ a b c d e : ℤ, a + b + c + d + e > 0 → ¬ (a < 0 ∨ b < 0 ∨ c < 0 ∨ d < 0 ∨ e < 0)) :=
begin
  sorry
end

end pentagon_operation_terminates_l505_505858


namespace four_digit_multiples_of_5_l505_505645

theorem four_digit_multiples_of_5 : 
  let f := fun n => 1000 <= n ∧ n <= 9999 ∧ n % 5 = 0
  in finset.card (finset.filter f (finset.range 10000)) = 1800 :=
by
  sorry

end four_digit_multiples_of_5_l505_505645


namespace clock_angle_915_pm_l505_505616

theorem clock_angle_915_pm :
  let minute_hand := 90
  let hour_hand := 277.5
  abs (hour_hand - minute_hand) = 187.5 :=
by
  sorry

end clock_angle_915_pm_l505_505616


namespace solve_diophantine_equation_l505_505786

theorem solve_diophantine_equation :
  ∃ (x y : ℤ), x^2 - x * y - 6 * y^2 + 2 * x + 19 * y = 18 ∧ (x = 2 ∧ y = 2 ∨ x = -2 ∧ y = 2) :=
  sorry

end solve_diophantine_equation_l505_505786


namespace proof_of_arithmetic_sequence_l505_505843

theorem proof_of_arithmetic_sequence 
  (x y z : ℕ) 
  (h1 : x + y + z = 15) 
  (h2 : x < y) 
  (h3 : y < z)
  (h4 : (x + 1) * (z + 9) = (y + 3) ^ 2) : 
  (x, y, z) = (3, 5, 7) :=
sorry

end proof_of_arithmetic_sequence_l505_505843


namespace probability_leq_one_l505_505692

variable (η : ℤ → ℝ)

-- Conditions from the table
def prob_eta : ℤ → ℝ
| -1 := 0.1
| 0 := 0.1
| 1 := 0.2
| 2 := 0.3
| 3 := 0.25
| 4 := 0.05
| _ := 0   -- For other values of η, probability is 0

-- Question statement
theorem probability_leq_one : (prob_eta η (-1) + prob_eta η 0 + prob_eta η 1) = 0.4 :=
by sorry

end probability_leq_one_l505_505692


namespace quadratic_has_two_real_roots_find_m_l505_505590

def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem quadratic_has_two_real_roots (m : ℝ) :
  discriminant 1 (-4 * m) (3 * m^2) ≥ 0 :=
by
  unfold discriminant
  have h : (-4 * m)^2 - 4 * 1 * (3 * m^2) = 4 * m^2
  ring
  exact ge_of_eq h

theorem find_m (h : 0 < m) (root_diff : ℝ) 
  (diff_eq_two : root_diff = 2) : m = 1 :=
by
  -- Let the roots be x1 and x2
  let x1 := (4 * m + root_diff) / 2
  let x2 := (4 * m - root_diff) / 2
  have : x1 - x2 = root_diff :=
    by
      field_simp
      exact diff_eq_two
  have sum_eq := (x1 - x2) * (x1 + x2) - (x1 + x2) * (x1 - x2) = 4
  ring
  have h_m_eq_1 : 4 * m = 4,
  by field_simp
  exact h_m_eq_1

  have h_m_1 : m = 1,
  sorry
  exact ge_of_eq h_m_1

end quadratic_has_two_real_roots_find_m_l505_505590


namespace four_digit_multiples_of_5_count_l505_505626

theorem four_digit_multiples_of_5_count :
  let lower_bound := 200
  let upper_bound := 1999
  (upper_bound - lower_bound + 1) = 1800 :=
by
  let lower_bound := 200
  let upper_bound := 1999
  show (upper_bound - lower_bound + 1) = 1800,
  from sorry

end four_digit_multiples_of_5_count_l505_505626


namespace ratio_of_ages_l505_505905

-- Definitions of the conditions
def son_current_age : ℕ := 24
def man_current_age : ℕ := son_current_age + 26

-- Function to compute the ages in two years
def son_age_in_two_years (S : ℕ) := S + 2
def man_age_in_two_years (M : ℕ) := M + 2

-- The proof statement
theorem ratio_of_ages (S : ℕ) (M : ℕ) (h1 : S = son_current_age) (h2 : M = man_current_age) :
  man_age_in_two_years M / son_age_in_two_years S = 2 :=
by
  intros
  rw [h1, h2]
  dsimp [son_current_age, man_current_age, son_age_in_two_years, man_age_in_two_years]
  norm_num
  -- This would be expanded into a proof step by step
  sorry

end ratio_of_ages_l505_505905


namespace least_positive_integer_with_12_factors_is_96_l505_505272

theorem least_positive_integer_with_12_factors_is_96 :
  ∃ n : ℕ, (n = 96 ∧ (∀ m : ℕ, (m > 0 ∧ ∀ d : ℕ, d ∣ m → d > 0 → d < m → (∃ k : ℕ, d = k ∨ k * d = m) → k > 1 → m > n))) ∧ (∃ m : ℕ, m = 12 ∧ (∀ d : ℕ, d ∣ m → d ∈ {1, m} ∨ ∀ k : ℕ, k * d = m → d = k ∨ d < k)).
sorry

end least_positive_integer_with_12_factors_is_96_l505_505272


namespace least_positive_integer_with_12_factors_is_72_l505_505312

-- Definition of having exactly 12 factors for a positive integer
def has_exactly_12_factors (n : ℕ) : Prop := n > 0 ∧ (finset.filter (λ d, n % d = 0) (finset.range (n + 1))).card = 12

-- Statement of the problem in Lean 4
theorem least_positive_integer_with_12_factors_is_72 :
  ∃ n : ℕ, has_exactly_12_factors n ∧ ∀ m : ℕ, has_exactly_12_factors m → n ≤ m :=
begin
  use 72,
  split,
  { -- Prove that 72 has exactly 12 factors
    sorry },
  { -- Prove that 72 is the least such integer
    sorry }
end

end least_positive_integer_with_12_factors_is_72_l505_505312


namespace herons_formula_proof_l505_505091

theorem herons_formula_proof :
  ∀ (a b c : ℝ), a = 2 → b = 3 → c = 4 → 
  let p := (a + b + c) / 2 in 
  let S := Real.sqrt (p * (p - a) * (p - b) * (p - c)) in
  p = 9 / 2 ∧ S = 3 * Real.sqrt 15 / 4 := by
  intros a b c ha hb hc
  simp [ha, hb, hc]
  let p := (2 + 3 + 4) / 2
  have hp : p = 9 / 2 := by norm_num
  let S := Real.sqrt (p * (p - 2) * (p - 3) * (p - 4))
  have hS : S = 3 * Real.sqrt 15 / 4 := by 
    rw [hp]
    have A : p - 2 = 5 / 2 := by norm_num
    have B : p - 3 = 3 / 2 := by norm_num
    have C : p - 4 = 1 / 2 := by norm_num
    rw [A, B, C]
    norm_num
    sorry -- Completing the proof is skipped
  exact ⟨hp, hS⟩

end herons_formula_proof_l505_505091


namespace correct_statement_is_one_l505_505747

-- Define the conditions:
variables (m n l : Type) [Line m] [Line n] [Line l]
variables (α β γ : Type) [Plane α] [Plane β] [Plane γ]

-- Define the perpendicular and parallel relationships:
variable (perp : Plane → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (in_plane : Line → Plane → Prop)
variable (intersects : Plane → Plane → Line)

-- State that α is perpendicular to β:
axiom alpha_perp_beta : perp α β

-- Hypothesize the statements:
def statement1 : Prop := ∃ l : Line, in_plane l α ∧ parallel l (intersects α β)
def statement2 : Prop := (perp γ α) → (parallel γ β)
def statement3 : Prop := (angle_with_plane m α = 30 ∧ angle_with_plane n α = 30) → (parallel m n)
def statement4 : Prop := ∀ A : Point, (in_plane A α ∧ intersects α β = l) → (perp m l) → (perp m β)

-- The proof problem:
theorem correct_statement_is_one : (statement1 ∧ ¬statement2 ∧ ¬statement3 ∧ ¬statement4) :=
sorry

end correct_statement_is_one_l505_505747


namespace least_pos_int_with_12_pos_factors_is_72_l505_505233

def least_positive_integer_with_12_factors (n : ℕ) : Prop :=
  ∃ (m : ℕ), m > 0 ∧ m ≠ n ∧ (∀ d : ℕ, d ∣ m → (d = n ∨ d = 1)) ∧
  ((∀ d : ℕ, d ∣ m → ∃ e : ℕ, e = 1) → n = 72)

theorem least_pos_int_with_12_pos_factors_is_72 (n : ℕ) :
  least_positive_integer_with_12_factors n → n = 72 := by
  sorry

end least_pos_int_with_12_pos_factors_is_72_l505_505233


namespace range_log2_sqrt_sin_cos_l505_505870

noncomputable def log2 (x : ℝ) := Real.log x / Real.log 2

theorem range_log2_sqrt_sin_cos :
  (Set.range (λ x : ℝ, log2 (Real.sqrt (Real.sin x * Real.cos x)))) ∩ Set.Icc 0 (Real.pi / 2) = Set.Iio (-1 / 2) :=
by
  sorry

end range_log2_sqrt_sin_cos_l505_505870


namespace four_digit_multiples_of_5_count_l505_505627

theorem four_digit_multiples_of_5_count :
  let lower_bound := 200
  let upper_bound := 1999
  (upper_bound - lower_bound + 1) = 1800 :=
by
  let lower_bound := 200
  let upper_bound := 1999
  show (upper_bound - lower_bound + 1) = 1800,
  from sorry

end four_digit_multiples_of_5_count_l505_505627


namespace count_leftmost_digit_7_l505_505738

noncomputable def num_of_digits (n : ℕ) : ℕ := (Real.log10 n).floor + 1

def has_leftmost_digit_7 (n : ℕ) : Prop :=
  (Real.log10 n - (Real.log10 n).floor).toNat = 7

def S := {k ∈ Finset.range 5001 | has_leftmost_digit_7 (5 ^ k)}

theorem count_leftmost_digit_7 :
  ∃! n, n = S.card ∧ n = 1501 := 
by
  sorry

end count_leftmost_digit_7_l505_505738


namespace range_of_a_l505_505694

theorem range_of_a (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) : 1 < a ∧ a < 2 :=
by
  -- Insert the proof here
  sorry

end range_of_a_l505_505694


namespace least_positive_integer_with_12_factors_l505_505381

theorem least_positive_integer_with_12_factors : 
  ∃ (n : ℕ), (n > 0) ∧ (nat.factors_count n = 12) ∧ (∀ m, (m > 0) → (nat.factors_count m = 12) → n ≤ m) ∧ n = 108 :=
sorry

end least_positive_integer_with_12_factors_l505_505381


namespace least_positive_integer_with_12_factors_l505_505352

theorem least_positive_integer_with_12_factors : ∃ k : ℕ, k > 0 ∧ (∀ n : ℕ, n > 0 → (factors_count n = 12 → k ≤ n)) ∧ k = 72 :=
by {
  sorry,
}

end least_positive_integer_with_12_factors_l505_505352


namespace solve_equation_in_natural_numbers_l505_505792

-- Define the main theorem
theorem solve_equation_in_natural_numbers :
  (∃ (x y z : ℕ), 2^x + 5^y + 63 = z! ∧ ((x = 5 ∧ y = 2 ∧ z = 5) ∨ (x = 4 ∧ y = 4 ∧ z = 6))) :=
sorry

end solve_equation_in_natural_numbers_l505_505792


namespace length_FQ_l505_505804

-- Define the right triangle and its properties
structure RightTriangle (DF DE EF : ℝ) :=
(right_angle_at_E : (DE^2 + EF^2 = DF^2))

-- Define the existence of a circle tangent to sides DF and EF
structure TangentCircle (DE DF EF : ℝ) :=
(center_on_DE : ∃ center : ℝ, center ∈ set.Icc 0 DE) -- Center is on segment DE
(tangent_to_DF : DF = √(center^2 + EF^2)) -- Tangency condition with side DF
(tangent_to_EF : EF = √(center^2 + DF^2)) -- Tangency condition with side EF

-- Problem statement: The length of FQ
theorem length_FQ
  (DE DF : ℝ)
  (DE_eq_7 : DE = 7)
  (DF_eq_sqrt85 : DF = real.sqrt 85)
  (EF : ℝ)
  (right_triangle : RightTriangle DF DE EF)
  (tangent_circle : TangentCircle DE DF EF) :
  (∃ FQ : ℝ, FQ = EF ∧ FQ = 6) := by
  -- Proof not included, just the statement
  sorry

end length_FQ_l505_505804


namespace ratio_of_x_to_y_l505_505871

variable (x y : Rational)

theorem ratio_of_x_to_y (h : (8 * x + 5 * y) / (10 * x + 3 * y) = 4 / 7) : x / y = -23 / 16 :=
by
  sorry

end ratio_of_x_to_y_l505_505871


namespace perfect_square_difference_l505_505821

def lastDigit (n : ℕ) : ℕ :=
  n % 10

theorem perfect_square_difference :
  ∃ a b : ℕ, ∃ x y : ℕ,
    a = x^2 ∧ b = y^2 ∧
    lastDigit a = 6 ∧
    lastDigit b = 4 ∧
    lastDigit (a - b) = 2 ∧
    lastDigit a > lastDigit b :=
by
  sorry

end perfect_square_difference_l505_505821


namespace clock_angle_915_pm_l505_505617

theorem clock_angle_915_pm :
  let minute_hand := 90
  let hour_hand := 277.5
  abs (hour_hand - minute_hand) = 187.5 :=
by
  sorry

end clock_angle_915_pm_l505_505617


namespace least_integer_with_twelve_factors_l505_505180

open Classical

theorem least_integer_with_twelve_factors : ∃ k : ℕ, (∀ n : ℕ, (count_factors n = 12 → k ≤ n)) ∧ k = 72 :=
by
  sorry

-- Here count_factors is a function that returns the number of positive divisors of a given integer.

end least_integer_with_twelve_factors_l505_505180


namespace pool_capacity_l505_505422

variable (C : ℕ)

-- Conditions
def rate_first_valve := C / 120
def rate_second_valve := C / 120 + 50
def combined_rate := C / 48

-- Proof statement
theorem pool_capacity (C_pos : 0 < C) (h1 : rate_first_valve C + rate_second_valve C = combined_rate C) : C = 12000 := by
  sorry

end pool_capacity_l505_505422


namespace return_trip_time_approx_l505_505446

variable (d p w t : ℝ)

theorem return_trip_time_approx (h1 : d = 120 * (p - w)) 
  (h2 : d = 108 * p)
  (h3 : 120 + t - 6 = 222)
  (h4 : t = 108) 
  : (d / (p + w) ≈ 98) := by
sorry

end return_trip_time_approx_l505_505446


namespace symmetric_point_example_l505_505714

def symmetric_point_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

theorem symmetric_point_example : symmetric_point_y_axis (2, 5) = (-2, 5) :=
by
  simp [symmetric_point_y_axis]
  sorry

end symmetric_point_example_l505_505714


namespace total_area_of_farm_l505_505918

-- Define the number of sections and area of each section
def number_of_sections : ℕ := 5
def area_of_each_section : ℕ := 60

-- State the problem as proving the total area of the farm
theorem total_area_of_farm : number_of_sections * area_of_each_section = 300 :=
by sorry

end total_area_of_farm_l505_505918


namespace proposition_C_correct_l505_505526

theorem proposition_C_correct (a b c : ℝ) (h : a * c ^ 2 > b * c ^ 2) : a > b :=
sorry

end proposition_C_correct_l505_505526


namespace line_ellipse_intersection_l505_505588

theorem line_ellipse_intersection (m : ℝ) : 
  (∀ x y : ℝ, y = 2 * x + m ∧ (x^2 / 4 + y^2 / 2 = 1)) →
  (-3 * Real.sqrt 2 < m ∧ m < 3 * Real.sqrt 2) ∨
  (m = 3 * Real.sqrt 2 ∨ m = -3 * Real.sqrt 2) ∨ 
  (m < -3 * Real.sqrt 2 ∨ m > 3 * Real.sqrt 2) :=
sorry

end line_ellipse_intersection_l505_505588


namespace least_positive_integer_with_12_factors_is_72_l505_505308

-- Definition of having exactly 12 factors for a positive integer
def has_exactly_12_factors (n : ℕ) : Prop := n > 0 ∧ (finset.filter (λ d, n % d = 0) (finset.range (n + 1))).card = 12

-- Statement of the problem in Lean 4
theorem least_positive_integer_with_12_factors_is_72 :
  ∃ n : ℕ, has_exactly_12_factors n ∧ ∀ m : ℕ, has_exactly_12_factors m → n ≤ m :=
begin
  use 72,
  split,
  { -- Prove that 72 has exactly 12 factors
    sorry },
  { -- Prove that 72 is the least such integer
    sorry }
end

end least_positive_integer_with_12_factors_is_72_l505_505308


namespace max_profit_is_45_6_l505_505437

noncomputable def profit_A (x : ℝ) : ℝ := 5.06 * x - 0.15 * x^2
noncomputable def profit_B (x : ℝ) : ℝ := 2 * x

noncomputable def total_profit (x : ℝ) : ℝ :=
  profit_A x + profit_B (15 - x)

theorem max_profit_is_45_6 : 
  ∃ x, 0 ≤ x ∧ x ≤ 15 ∧ total_profit x = 45.6 :=
by
  sorry

end max_profit_is_45_6_l505_505437


namespace savings_on_discounted_milk_l505_505026

theorem savings_on_discounted_milk :
  let num_gallons := 8
  let price_per_gallon := 3.20
  let discount_rate := 0.25
  let discount_per_gallon := price_per_gallon * discount_rate
  let discounted_price_per_gallon := price_per_gallon - discount_per_gallon
  let total_cost_without_discount := num_gallons * price_per_gallon
  let total_cost_with_discount := num_gallons * discounted_price_per_gallon
  let savings := total_cost_without_discount - total_cost_with_discount
  savings = 6.40 :=
by
  sorry

end savings_on_discounted_milk_l505_505026


namespace least_pos_int_with_12_pos_factors_is_72_l505_505234

def least_positive_integer_with_12_factors (n : ℕ) : Prop :=
  ∃ (m : ℕ), m > 0 ∧ m ≠ n ∧ (∀ d : ℕ, d ∣ m → (d = n ∨ d = 1)) ∧
  ((∀ d : ℕ, d ∣ m → ∃ e : ℕ, e = 1) → n = 72)

theorem least_pos_int_with_12_pos_factors_is_72 (n : ℕ) :
  least_positive_integer_with_12_factors n → n = 72 := by
  sorry

end least_pos_int_with_12_pos_factors_is_72_l505_505234


namespace least_integer_with_twelve_factors_l505_505184

open Classical

theorem least_integer_with_twelve_factors : ∃ k : ℕ, (∀ n : ℕ, (count_factors n = 12 → k ≤ n)) ∧ k = 72 :=
by
  sorry

-- Here count_factors is a function that returns the number of positive divisors of a given integer.

end least_integer_with_twelve_factors_l505_505184


namespace smallest_m_has_36_divisors_not_multiple_of_15_l505_505746

theorem smallest_m_has_36_divisors_not_multiple_of_15 :
  ∃ m : ℕ, (∃ k1 : ℕ, m = 2^k1 ∧ k1 % 3 = 0) ∧
           (∃ k2 : ℕ, m = 3^k2 ∧ k2 % 5 = 1) ∧
           (∃ k3 : ℕ, m = 5^k3 ∧ k3 % 2 = 0) ∧
           (∀ n : ℕ, (∃ k4 : ℕ, n = 2^k4 ∧ k4 % 3 = 0) ∧
                      (∃ k5 : ℕ, n = 3^k5 ∧ k5 % 5 = 1) ∧
                      (∃ k6 : ℕ, n = 5^k6 ∧ k6 % 2 = 0) → m ≤ n) ∧
           ((∏ p in (finset.divisors m).filter (λ d, ¬ (15 ∣ d)), 1) = 36) :=
by sorry

end smallest_m_has_36_divisors_not_multiple_of_15_l505_505746


namespace least_positive_integer_with_12_factors_l505_505327

def has_exactly_12_factors (n : ℕ) : Prop :=
  n.factors.length = 12

theorem least_positive_integer_with_12_factors : ∃ k : ℕ, has_exactly_12_factors k ∧ (∀ n : ℕ, has_exactly_12_factors n → n ≥ k) ∧ k = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_l505_505327


namespace cars_in_first_store_l505_505420

theorem cars_in_first_store (
  cars_store_2 : ℕ := 14,
  cars_store_3 : ℕ := 14,
  cars_store_4 : ℕ := 21,
  cars_store_5 : ℕ := 25,
  mean_cars : ℚ := 20.8,
  total_stores : ℕ := 5) :
  ∃ (X : ℕ), X = 30 :=
by
  sorry

end cars_in_first_store_l505_505420


namespace four_digit_multiples_of_five_count_l505_505665

theorem four_digit_multiples_of_five_count : 
  {n : ℕ | 1000 ≤ n ∧ n ≤ 9999 ∧ n % 5 = 0}.toFinset.card = 1800 :=
by
  sorry

end four_digit_multiples_of_five_count_l505_505665


namespace totalMilkConsumption_l505_505701

-- Conditions
def regularMilk (week: ℕ) : ℝ := 0.5
def soyMilk (week: ℕ) : ℝ := 0.1

-- Theorem statement
theorem totalMilkConsumption : regularMilk 1 + soyMilk 1 = 0.6 := 
by 
  sorry

end totalMilkConsumption_l505_505701


namespace least_positive_integer_with_12_factors_l505_505162

theorem least_positive_integer_with_12_factors : ∃ n : ℕ, (nat.factors_count n = 12 ∧ ∀ m : ℕ, (nat.factors_count m = 12) → n ≤ m) ∧ n = 72 :=
begin
  sorry
end

end least_positive_integer_with_12_factors_l505_505162


namespace least_positive_integer_with_12_factors_l505_505211

/--
Prove that the smallest positive integer with exactly 12 positive factors is 60.
-/
theorem least_positive_integer_with_12_factors : 
  ∃ (n : ℕ), (0 < n) ∧ (nat.factors_count n = 12) ∧ (∀ m : ℕ, (0 < m) ∧ (nat.factors_count m = 12) → n ≤ m) → n = 60 :=
sorry

end least_positive_integer_with_12_factors_l505_505211


namespace find_b_l505_505527

theorem find_b (a b c d : ℝ) (x : ℂ) :
  (∀ x : ℂ, x^4 + (a : ℂ) * x^3 + (b : ℂ) * x^2 + (c : ℂ) * x + (d : ℂ) = 0 →
    x = z ∨ x = w ∨ x = conj z ∨ x = conj w) →
  (z * w = 7 + 2 * I) →
  (conj z * conj w = 7 - 2 * I) →
  (conj z + conj w = 5 - 3 * I) →
  (z + w = 5 + 3 * I) →
  b = 48 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end find_b_l505_505527


namespace log4_16_eq_2_l505_505966

theorem log4_16_eq_2 : log 4 16 = 2 := sorry

end log4_16_eq_2_l505_505966


namespace least_positive_integer_with_12_factors_l505_505282

def num_factors (n : ℕ) : ℕ := 
  (finset.range (n + 1)).filter (λ d, n % d = 0).card

theorem least_positive_integer_with_12_factors : 
  ∃ n : ℕ, num_factors n = 12 ∧ n = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_l505_505282


namespace arithmetic_geometric_difference_l505_505943

-- Define a three-digit number as arithmetic-geometric.
def is_arithmetic_geometric (n : ℕ) : Prop :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  (d1 < 10 ∧ d2 < 10 ∧ d3 < 10) ∧ (d1 != d2 ∧ d2 != d3 ∧ d1 != d3) ∧
  (d2 - d1 = d3 - d2 ∨ ((d2 * d2 = d1 * d3) ∧ (d1 ≠ 0) ∧ (d2 ≠ 0)))

-- Prove the arithmetic-geometric numbers problem statement.
theorem arithmetic_geometric_difference : 
  let candidates := [816, 832, 848, 879, 916, 932, 948, 987]
  let largest := 987
  let smallest := 816
  largest - smallest = 171 := 
by
  let candidates := [816, 832, 848, 879, 916, 932, 948, 987]
  let largest := 987
  let smallest := 816
  have h1 : largest = List.maximum candidates := by sorry
  have h2 : smallest = List.minimum candidates := by sorry
  show largest - smallest = 171 from
    calc
      largest - smallest
        = 987 - 816 : by congr; exact h1; exact h2
    ... = 171 : by rfl

end arithmetic_geometric_difference_l505_505943


namespace factorization_solution_1_factorization_solution_2_factorization_solution_3_l505_505505

noncomputable def factorization_problem_1 (m : ℝ) : Prop :=
  -3 * m^3 + 12 * m = -3 * m * (m + 2) * (m - 2)

noncomputable def factorization_problem_2 (x y : ℝ) : Prop :=
  2 * x^2 * y - 8 * x * y + 8 * y = 2 * y * (x - 2)^2

noncomputable def factorization_problem_3 (a : ℝ) : Prop :=
  a^4 + 3 * a^2 - 4 = (a^2 + 4) * (a + 1) * (a - 1)

-- Lean statements for the proofs
theorem factorization_solution_1 (m : ℝ) : factorization_problem_1 m :=
  by sorry

theorem factorization_solution_2 (x y : ℝ) : factorization_problem_2 x y :=
  by sorry

theorem factorization_solution_3 (a : ℝ) : factorization_problem_3 a :=
  by sorry

end factorization_solution_1_factorization_solution_2_factorization_solution_3_l505_505505


namespace abs_inequality_solution_l505_505429

theorem abs_inequality_solution (x : ℝ) : |x + 2| + |x - 1| ≥ 5 ↔ x ≤ -3 ∨ x ≥ 2 :=
sorry

end abs_inequality_solution_l505_505429


namespace expected_value_and_variance_of_X1_l505_505741

noncomputable def X : Type → ℝ := sorry -- a discrete random variable
axiom E_of_X (Ω : Type) : ℝ -- expected value of X
axiom D_of_X (Ω : Type) : ℝ -- variance of X

-- Given conditions
axiom ex_X (Ω : Type) : E_of_X Ω = 6 -- E(X) = 6
axiom var_X (Ω : Type) : D_of_X Ω = 0.5 -- D(X) = 0.5

-- Definitions
noncomputable def X1 (Ω : Type) : ℝ := 2 * X Ω - 5

-- Proof goal
theorem expected_value_and_variance_of_X1 (Ω : Type) : 
  E_of_X1 Ω = 7 ∧ D_of_X1 Ω = 2 :=
by
  -- use the linearity properties of expectation and variance
  sorry

end expected_value_and_variance_of_X1_l505_505741


namespace village_population_l505_505835
noncomputable def current_population (P : ℝ) : Prop :=
  P * (1 + 18 / 100) ^ 2 = 10860.72

theorem village_population : ∃ P : ℝ, current_population P ∧ P ≈ 7800 :=
begin
  sorry
end

end village_population_l505_505835


namespace least_positive_integer_with_12_factors_l505_505176

theorem least_positive_integer_with_12_factors :
  ∃ m : ℕ, 0 < m ∧ (∀ n : ℕ, 0 < n → (n ≠ m →  m ≤ n)) ∧ (12 = (m.factors.toFinset.card)) := sorry

end least_positive_integer_with_12_factors_l505_505176


namespace triangle_AE_eq_CF_l505_505756

theorem triangle_AE_eq_CF (A B C D E F : Type)
  [InTriangle A B C]
  (h1 : FootAngleBisector B D)
  (h2 : SecondIntersectionCircumcircle B C D E AB)
  (h3 : SecondIntersectionCircumcircle A B D F BC) :
  AE = CF :=
by
  sorry

end triangle_AE_eq_CF_l505_505756


namespace small_angle_9_15_l505_505615

theorem small_angle_9_15 : 
  let hours_to_angle := 30
  let minute_to_angle := 6
  let minutes := 15
  let hour := 9
  let hour_position := hour * hours_to_angle
  let additional_hour_angle := (hours_to_angle * (minutes / 60.0))
  let final_hour_position := hour_position + additional_hour_angle
  let minute_position := (minutes / 5.0) * hours_to_angle
  let angle := abs (final_hour_position - minute_position)
  let small_angle := if angle > 180 then 360 - angle else angle
  in small_angle = 187.5 := by 
  sorry

end small_angle_9_15_l505_505615


namespace inequality_proof_l505_505838

def a : ℝ := 0.9^2
def b : ℝ := Real.log 0.9
def c : ℝ := 2^0.9

theorem inequality_proof : b < a ∧ a < c :=
by 
  -- a = 0.9^2
  -- b = ln(0.9)
  -- c = 2^0.9
  sorry

end inequality_proof_l505_505838


namespace min_trig_sum_abs_l505_505515

-- Statement for proving the minimum value of absolute trigonometric function sum
theorem min_trig_sum_abs (x : ℝ) :
  ∃ u : ℝ, u = sin x + cos x ∧ ∀ u ∈ [-real.sqrt 2, real.sqrt 2], 
    abs (sin x + cos x + tan x + cot x + sec x + csc x) = 2 * real.sqrt 2 - 1 := 
begin
  sorry
end

end min_trig_sum_abs_l505_505515


namespace number_of_legs_twice_heads_diff_eq_22_l505_505705

theorem number_of_legs_twice_heads_diff_eq_22 (P H : ℕ) (L : ℤ) (Heads : ℕ) (X : ℤ) (h1 : P = 11)
  (h2 : L = 4 * P + 2 * H) (h3 : Heads = P + H) (h4 : L = 2 * Heads + X) : X = 22 :=
by
  sorry

end number_of_legs_twice_heads_diff_eq_22_l505_505705


namespace least_positive_integer_with_12_factors_l505_505345

theorem least_positive_integer_with_12_factors : ∃ k : ℕ, k > 0 ∧ (∀ n : ℕ, n > 0 → (factors_count n = 12 → k ≤ n)) ∧ k = 72 :=
by {
  sorry,
}

end least_positive_integer_with_12_factors_l505_505345


namespace least_pos_int_with_12_pos_factors_is_72_l505_505237

def least_positive_integer_with_12_factors (n : ℕ) : Prop :=
  ∃ (m : ℕ), m > 0 ∧ m ≠ n ∧ (∀ d : ℕ, d ∣ m → (d = n ∨ d = 1)) ∧
  ((∀ d : ℕ, d ∣ m → ∃ e : ℕ, e = 1) → n = 72)

theorem least_pos_int_with_12_pos_factors_is_72 (n : ℕ) :
  least_positive_integer_with_12_factors n → n = 72 := by
  sorry

end least_pos_int_with_12_pos_factors_is_72_l505_505237


namespace least_positive_integer_with_12_factors_l505_505204

/--
Prove that the smallest positive integer with exactly 12 positive factors is 60.
-/
theorem least_positive_integer_with_12_factors : 
  ∃ (n : ℕ), (0 < n) ∧ (nat.factors_count n = 12) ∧ (∀ m : ℕ, (0 < m) ∧ (nat.factors_count m = 12) → n ≤ m) → n = 60 :=
sorry

end least_positive_integer_with_12_factors_l505_505204


namespace last_ball_is_green_l505_505060

theorem last_ball_is_green :
  let R : ℕ := 2020 in
  let G : ℕ := 2021 in
  (∃ p : ℝ, p = 1 ∧ ⌊ (2021 : ℕ) * p ⌋ = 2021)
:=
by
  let R := 2020
  let G := 2021
  have h : (G - R) % 4 = 1 % 4,
  { rw [nat.sub_self, nat.mod_self, nat.mod_eq_of_lt (nat.zero_lt_4), nat.succ_zero] },
  use 1
  split
  { refl }
  { rw [nat.floor_eq_iff, nat.cast_mul, nat.cast_to_real], split
    { linarith },
    { linarith }
  }
  sorry

end last_ball_is_green_l505_505060


namespace train_length_correct_l505_505454

def speed_kmh : ℝ := 42
def time_seconds : ℝ := 42.34285714285714
def bridge_length_meters : ℝ := 137

-- Convert speed to meters per second
def speed_ms : ℝ := (speed_kmh * 1000) / 3600

-- Total distance covered while passing the bridge
def total_distance : ℝ := speed_ms * time_seconds

-- Length of the train
def train_length : ℝ := total_distance - bridge_length_meters

theorem train_length_correct : train_length ≈ 356.7142857142857 := by
  sorry

end train_length_correct_l505_505454


namespace practice_minutes_other_days_l505_505464

-- Definitions based on given conditions
def total_hours_practiced : ℕ := 7.5 * 60 -- converting hours to minutes
def minutes_per_day := 86
def days_practiced := 2

-- Lean 4 statement for the proof problem
theorem practice_minutes_other_days :
  let total_minutes := total_hours_practiced
  let minutes_2_days := minutes_per_day * days_practiced
  total_minutes - minutes_2_days = 278 := by
  sorry

end practice_minutes_other_days_l505_505464


namespace least_positive_integer_with_12_factors_l505_505305

def has_exactly_12_factors (n : ℕ) : Prop :=
  (finset.range(n + 1).filter (λ d, n % d = 0)).card = 12

theorem least_positive_integer_with_12_factors : ∀ n : ℕ, has_exactly_12_factors n → n ≥ 150 :=
sorry

end least_positive_integer_with_12_factors_l505_505305


namespace least_positive_integer_with_12_factors_is_72_l505_505313

-- Definition of having exactly 12 factors for a positive integer
def has_exactly_12_factors (n : ℕ) : Prop := n > 0 ∧ (finset.filter (λ d, n % d = 0) (finset.range (n + 1))).card = 12

-- Statement of the problem in Lean 4
theorem least_positive_integer_with_12_factors_is_72 :
  ∃ n : ℕ, has_exactly_12_factors n ∧ ∀ m : ℕ, has_exactly_12_factors m → n ≤ m :=
begin
  use 72,
  split,
  { -- Prove that 72 has exactly 12 factors
    sorry },
  { -- Prove that 72 is the least such integer
    sorry }
end

end least_positive_integer_with_12_factors_is_72_l505_505313


namespace units_digit_of_expression_l505_505982

theorem units_digit_of_expression :
  (8 * 18 * 1988 - 8^4) % 10 = 6 := 
by
  sorry

end units_digit_of_expression_l505_505982


namespace second_term_is_three_l505_505926

-- Given conditions
variables (r : ℝ) (S : ℝ)
hypothesis hr : r = 1 / 4
hypothesis hS : S = 16

-- Definition of the first term a
noncomputable def first_term (r : ℝ) (S : ℝ) : ℝ :=
  S * (1 - r)

-- Definition of the second term
noncomputable def second_term (r : ℝ) (a : ℝ) : ℝ :=
  a * r

-- Prove that the second term is 3
theorem second_term_is_three : second_term r (first_term r S) = 3 :=
by
  rw [first_term, second_term]
  sorry

end second_term_is_three_l505_505926


namespace complex_div_example_l505_505564

open Complex

theorem complex_div_example (z1 z2 : ℂ) (h1 : z1 = -1 + 3 * Complex.i) (h2 : z2 = 1 + Complex.i) :
  (z1 + z2) / (z1 - z2) = 1 - Complex.i := 
sorry

end complex_div_example_l505_505564


namespace least_positive_integer_with_12_factors_l505_505276

def num_factors (n : ℕ) : ℕ := 
  (finset.range (n + 1)).filter (λ d, n % d = 0).card

theorem least_positive_integer_with_12_factors : 
  ∃ n : ℕ, num_factors n = 12 ∧ n = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_l505_505276


namespace least_positive_integer_with_12_factors_is_72_l505_505216

open Nat

def hasExactly12Factors (n : ℕ) : Prop :=
  (divisors n).length = 12

theorem least_positive_integer_with_12_factors_is_72 :
  ∃ n : ℕ, hasExactly12Factors n ∧ ∀ m : ℕ, hasExactly12Factors m → n ≤ m → n = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_is_72_l505_505216


namespace preservation_time_at_33_degrees_l505_505113

theorem preservation_time_at_33_degrees (k b : ℝ) 
  (h1 : ∀ x, (192 = Real.exp (k * 0 + b)))
  (h2 : ∀ x, (48 = Real.exp (k * 22 + b)))
  : Real.exp (k * 33 + b) = 24 :=
by 
  -- Problem given the conditions
  sorry

end preservation_time_at_33_degrees_l505_505113


namespace imo_2007_p6_l505_505879

theorem imo_2007_p6 (x y : ℕ) (hx : 0 < x) (hy : 0 < y) :
  ∃ k : ℕ, (x = 11 * k^2) ∧ (y = 11 * k) ↔
  ∃ k : ℕ, (∃ k₁ : ℤ, k₁ = (x^2 * y + x + y) / (x * y^2 + y + 11)) :=
sorry

end imo_2007_p6_l505_505879


namespace graph_shift_l505_505859

/-- 
Theorem: To obtain the graph of the function f(x) = sin(2x - π/6) from 
the graph of the function g(x) = sin(2x), the necessary shift is to the 
right by π/12 units.
-/
theorem graph_shift {π : ℝ} (x : ℝ) (h1 : f x = Real.sin (2 * x - π / 6))
  (h2 : g x = Real.sin (2 * x)) : 
  ∃ d : ℝ, d = π / 12 ∧ ∀ x, f x = g (x - d) :=
by
  sorry

end graph_shift_l505_505859


namespace f_f_is_even_l505_505039

-- Let f be a function from reals to reals
variables {f : ℝ → ℝ}

-- Given that f is an even function
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

-- Theorem to prove
theorem f_f_is_even (h : is_even f) : is_even (fun x => f (f x)) :=
by
  intros
  unfold is_even at *
  -- at this point, we assume the function f is even,
  -- follow from the assumption, we can prove the result
  sorry

end f_f_is_even_l505_505039


namespace least_positive_integer_with_12_factors_l505_505288

def num_factors (n : ℕ) : ℕ := 
  (finset.range (n + 1)).filter (λ d, n % d = 0).card

theorem least_positive_integer_with_12_factors : 
  ∃ n : ℕ, num_factors n = 12 ∧ n = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_l505_505288


namespace least_positive_integer_with_12_factors_l505_505154

theorem least_positive_integer_with_12_factors : ∃ n : ℕ, (nat.factors_count n = 12 ∧ ∀ m : ℕ, (nat.factors_count m = 12) → n ≤ m) ∧ n = 72 :=
begin
  sorry
end

end least_positive_integer_with_12_factors_l505_505154


namespace least_integer_with_twelve_factors_l505_505189

open Classical

theorem least_integer_with_twelve_factors : ∃ k : ℕ, (∀ n : ℕ, (count_factors n = 12 → k ≤ n)) ∧ k = 72 :=
by
  sorry

-- Here count_factors is a function that returns the number of positive divisors of a given integer.

end least_integer_with_twelve_factors_l505_505189


namespace count_four_digit_multiples_of_5_l505_505635

def is_multiple_of_5 (n : ℕ) : Prop :=
  n % 5 = 0

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def four_digit_multiples_of_5_count (lower upper : ℕ) :=
  (upper - lower + 1)

theorem count_four_digit_multiples_of_5 : 
  four_digit_multiples_of_5_count 200 1999 = 1800 :=
by
  sorry

end count_four_digit_multiples_of_5_l505_505635


namespace six_digit_phone_numbers_possible_l505_505721

theorem six_digit_phone_numbers_possible (S : Set (Fin 10 × Fin 10 × Fin 10 × Fin 10 × Fin 10 × Fin 10)) :
  (∃ S : Set (Fin 10 × Fin 10 × Fin 10 × Fin 10 × Fin 10 × Fin 10), 
    S.card = 100000 ∧ ∀ (k : Fin 6), 
      ∀ (n : Fin 10 × Fin 10 × Fin 10 × Fin 10 × Fin 10), 
        ∃ s ∈ S, f_k k s = n) :=
sorry

noncomputable def f_k (k : Fin 6) 
  (s : Fin 10 × Fin 10 × Fin 10 × Fin 10 × Fin 10 × Fin 10) : Fin 100000 :=
  let (a, b, c, d, e, f) := s in
  match k with
  | 0 => (b, c, d, e, f)
  | 1 => (a, c, d, e, f)
  | 2 => (a, b, d, e, f)
  | 3 => (a, b, c, e, f)
  | 4 => (a, b, c, d, f)
  | 5 => (a, b, c, d, e)
  end

end six_digit_phone_numbers_possible_l505_505721


namespace number_of_eggs_in_one_unit_l505_505057

theorem number_of_eggs_in_one_unit 
    (supplies_per_day_in_units : ℕ)
    (supplies_per_day_in_eggs : ℕ)
    (total_supplies_per_week : ℕ) :
    supplies_per_day_in_units = 5 → 
    supplies_per_day_in_eggs = 30 → 
    total_supplies_per_week = 630 → 
    ∃ x : ℕ, 7 * (5 * x + 30) = 630 ∧ x = 12 :=
by
  intros h1 h2 h3
  use 12
  split
  {
    calc 
      7 * (5 * 12 + 30) = 7 * (60 + 30) : by sorry
      ... = 7 * 90 : by sorry
      ... = 630 : by sorry
  }
  {
    exact rfl
  }
  sorry

end number_of_eggs_in_one_unit_l505_505057


namespace least_positive_integer_with_12_factors_l505_505304

def has_exactly_12_factors (n : ℕ) : Prop :=
  (finset.range(n + 1).filter (λ d, n % d = 0)).card = 12

theorem least_positive_integer_with_12_factors : ∀ n : ℕ, has_exactly_12_factors n → n ≥ 150 :=
sorry

end least_positive_integer_with_12_factors_l505_505304


namespace A_intersection_B_eq_C_l505_505566

def A := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def B := {x : ℝ | 0 < x ∧ x < 3}
def C := {x : ℝ | 0 < x ∧ x ≤ 2}

theorem A_intersection_B_eq_C : A ∩ B = C := 
by sorry

end A_intersection_B_eq_C_l505_505566


namespace least_positive_integer_with_12_factors_l505_505346

theorem least_positive_integer_with_12_factors : ∃ k : ℕ, k > 0 ∧ (∀ n : ℕ, n > 0 → (factors_count n = 12 → k ≤ n)) ∧ k = 72 :=
by {
  sorry,
}

end least_positive_integer_with_12_factors_l505_505346


namespace perfect_square_for_n_l505_505772

theorem perfect_square_for_n 
  (a b : ℕ)
  (h1 : ∃ x : ℕ, ab = x^2)
  (h2 : ∃ y : ℕ, (a + 1) * (b + 1) = y^2) 
  : ∃ n : ℕ, n > 1 ∧ ∃ z : ℕ, (a + n) * (b + n) = z^2 :=
by
  let n := ab
  have h3 : n > 1 := sorry
  have h4 : ∃ z : ℕ, (a + n) * (b + n) = z^2 := sorry
  exact ⟨n, h3, h4⟩

end perfect_square_for_n_l505_505772


namespace sum_2n_sum_2n1_l505_505708

-- condition: common difference of the sequence {a_n} is zero, and a_1 = 2
def a_n (n : ℕ) : ℝ := 2

-- condition: a_1 and 2 a_4 are in geometric progression, which we've verified
example (n : ℕ) : a_n 1 * 2 * a_n n = a_n (n + 1) * a_n (n - 1) :=
by sorry

-- definition of b_n given a_n = 2 for all n
def b_n (n : ℕ) : ℝ := 
  (-1)^(n+1) * (1 / n + 1 / (n + 1))

-- sum of the first 2n terms of the sequence {b_n}
def sum_first_2n_terms (n : ℕ) : ℝ :=
  ∑ i in finset.range (2 * n), b_n i

-- sum of the first 2n+1 terms of the sequence {b_n}
def sum_first_2n1_terms (n : ℕ) : ℝ :=
  ∑ i in finset.range (2 * n + 1), b_n i

-- theorem to prove sum of first 2n terms is 1 + 1/2n
theorem sum_2n (n : ℕ) : sum_first_2n_terms n = 1 + 1 / (2 * n) :=
by sorry

-- theorem to prove sum of first 2n+1 terms is 1 - 1/2n+1
theorem sum_2n1 (n : ℕ) : sum_first_2n1_terms n = 1 - 1 / (2 * n + 1) :=
by sorry

end sum_2n_sum_2n1_l505_505708


namespace four_digit_multiples_of_5_l505_505647

theorem four_digit_multiples_of_5 : 
  let f := fun n => 1000 <= n ∧ n <= 9999 ∧ n % 5 = 0
  in finset.card (finset.filter f (finset.range 10000)) = 1800 :=
by
  sorry

end four_digit_multiples_of_5_l505_505647


namespace least_positive_integer_with_12_factors_l505_505350

theorem least_positive_integer_with_12_factors : ∃ k : ℕ, k > 0 ∧ (∀ n : ℕ, n > 0 → (factors_count n = 12 → k ≤ n)) ∧ k = 72 :=
by {
  sorry,
}

end least_positive_integer_with_12_factors_l505_505350


namespace symmetric_point_example_l505_505715

def symmetric_point_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

theorem symmetric_point_example : symmetric_point_y_axis (2, 5) = (-2, 5) :=
by
  simp [symmetric_point_y_axis]
  sorry

end symmetric_point_example_l505_505715


namespace find_number_of_girls_in_class_l505_505124

variable (G : ℕ)

def number_of_ways_to_select_two_boys (n : ℕ) : ℕ := Nat.choose n 2

theorem find_number_of_girls_in_class 
  (boys : ℕ := 13) 
  (ways_to_select_students : ℕ := 780) 
  (ways_to_select_two_boys : ℕ := number_of_ways_to_select_two_boys boys) :
  G * ways_to_select_two_boys = ways_to_select_students → G = 10 := 
by
  sorry

end find_number_of_girls_in_class_l505_505124


namespace zero_if_and_only_if_m_eq_3_imaginary_if_and_only_if_m_neq_0_and_3_pure_imaginary_if_and_only_if_m_eq_2_l505_505992

variable (m : ℝ)
def z : ℂ := (m^2 - 5*m + 6) + (m^2 - 3*m) * complex.I

-- Question 1: z is zero if and only if m = 3
theorem zero_if_and_only_if_m_eq_3 : z m = 0 ↔ m = 3 :=
sorry

-- Question 2: z is an imaginary number if and only if m ≠ 0 and m ≠ 3
theorem imaginary_if_and_only_if_m_neq_0_and_3 : (∃ im : ℝ, z m = im * complex.I) ↔ (m ≠ 0 ∧ m ≠ 3) :=
sorry

-- Question 3: z is a pure imaginary number if and only if m = 2
theorem pure_imaginary_if_and_only_if_m_eq_2 : (∃ im : ℝ, z m = im * complex.I ∧ real_part m = 0) ↔ m = 2 :=
sorry

end zero_if_and_only_if_m_eq_3_imaginary_if_and_only_if_m_neq_0_and_3_pure_imaginary_if_and_only_if_m_eq_2_l505_505992


namespace number_of_students_l505_505129

theorem number_of_students (total_stars : ℕ) (stars_per_student : ℕ) (h1 : total_stars = 372) (h2 : stars_per_student = 3) : total_stars / stars_per_student = 124 :=
by
  sorry

end number_of_students_l505_505129


namespace car_travel_time_l505_505891

theorem car_travel_time (speed distance : ℝ) (h₁ : speed = 65) (h₂ : distance = 455) :
  distance / speed = 7 :=
by
  -- We will invoke the conditions h₁ and h₂ to conclude the theorem
  sorry

end car_travel_time_l505_505891


namespace distance_between_bicyclists_l505_505020

theorem distance_between_bicyclists (t : ℝ) :
  (10 * t) ^ 2 + (15 * t) ^ 2 = 325 →
  t = 100 / real.sqrt 325 →
  abs(t - (20 * real.sqrt 13 / 13)) < 0.0001 := 
by
  intros h1 h2
  sorry

end distance_between_bicyclists_l505_505020


namespace least_positive_integer_with_12_factors_l505_505353

theorem least_positive_integer_with_12_factors : ∃ k : ℕ, k > 0 ∧ (∀ n : ℕ, n > 0 → (factors_count n = 12 → k ≤ n)) ∧ k = 72 :=
by {
  sorry,
}

end least_positive_integer_with_12_factors_l505_505353


namespace minor_axis_length_l505_505915

noncomputable def length_of_minor_axis : ℝ :=
  let p1 := (1, 0) in
  let p2 := (1, 3) in
  let p3 := (4, 0) in
  let p4 := (4, 3) in
  let p5 := (-1, 1.5) in
  -- Calculation of b
  21 / 4 / Real.sqrt 10

theorem minor_axis_length :
  let p1 := (1, 0) in
  let p2 := (1, 3) in
  let p3 := (4, 0) in
  let p4 := (4, 3) in
  let p5 := (-1, 1.5) in
  ∃ b : ℝ, (2 * b = length_of_minor_axis) := by
  -- The actual proof is omitted here
  sorry

end minor_axis_length_l505_505915


namespace alpha_values_l505_505973

theorem alpha_values (α : ℝ) (k : ℤ) :
  (∃ (H : ∀ n : ℕ, (real.cos ((2 : ℕ) ^ n * α) ≤ - 1 / 4)), α = 2 * k * real.pi + 2 * real.pi / 3 ∨ α = 2 * k * real.pi - 2 * real.pi / 3) :=
sorry

end alpha_values_l505_505973


namespace least_positive_integer_with_12_factors_l505_505372

theorem least_positive_integer_with_12_factors : 
  ∃ (n : ℕ), (n > 0) ∧ (nat.factors_count n = 12) ∧ (∀ m, (m > 0) → (nat.factors_count m = 12) → n ≤ m) ∧ n = 108 :=
sorry

end least_positive_integer_with_12_factors_l505_505372


namespace solve_eq_proof_l505_505084

noncomputable def solve_equation : List ℚ := [-4, 1, 3 / 2, 2]

theorem solve_eq_proof :
  (∀ x : ℚ, 
    ((x^2 + 3 * x - 4)^2 + (2 * x^2 - 7 * x + 6)^2 = (3 * x^2 - 4 * x + 2)^2) ↔ 
    (x ∈ solve_equation)) :=
by
  sorry

end solve_eq_proof_l505_505084


namespace projectile_height_l505_505823

theorem projectile_height (t : ℝ) : 
  (∃ t : ℝ, (-4.9 * t^2 + 30.4 * t = 35)) → 
  (0 < t ∧ t ≤ 5) → 
  t = 10 / 7 :=
by
  sorry

end projectile_height_l505_505823


namespace find_x_for_divisibility_18_l505_505522

theorem find_x_for_divisibility_18 (x : ℕ) (h_digits : x < 10) :
  (1001 * x + 150) % 18 = 0 ↔ x = 6 :=
by
  sorry

end find_x_for_divisibility_18_l505_505522


namespace triangle_area_l505_505010

theorem triangle_area
  (a b c : ℝ)
  (h1 : c^2 = (a - b)^2 + 6)
  (h2 : C = π / 3) :
  let area := (1 / 2) * a * b * sin C in
  area = 3 * (real.sqrt 3) / 2 :=
by {
  sorry
}

end triangle_area_l505_505010


namespace tan_difference_sum_of_angles_l505_505887

-- Problem (1) Lean 4 Statement
theorem tan_difference (A B : ℝ) (h : 2 * tan A = 3 * tan B) :
  tan (A - B) = sin (2 * B) / (5 - cos (2 * B)) :=
sorry

-- Problem (2) Lean 4 Statement
theorem sum_of_angles (α β : ℝ) (h1 : α < π / 2) (h2 : β < π / 2)
  (h3 : tan α = 1 / 7) (h4 : sin β = sqrt 10 / 10) :
  α + 2 * β = π / 4 :=
sorry

end tan_difference_sum_of_angles_l505_505887


namespace evaluate_expression_l505_505569

variable (a b c : ℝ)

theorem evaluate_expression 
  (h : a / (20 - a) + b / (75 - b) + c / (55 - c) = 8) :
  4 / (20 - a) + 15 / (75 - b) + 11 / (55 - c) = 8.8 :=
sorry

end evaluate_expression_l505_505569


namespace least_positive_integer_with_12_factors_l505_505177

theorem least_positive_integer_with_12_factors :
  ∃ m : ℕ, 0 < m ∧ (∀ n : ℕ, 0 < n → (n ≠ m →  m ≤ n)) ∧ (12 = (m.factors.toFinset.card)) := sorry

end least_positive_integer_with_12_factors_l505_505177


namespace domain_h_correct_l505_505932

def h (x : ℝ) := (x^4 - 3 * x^3 + 5 * x^2 - 2 * x + 7) / (x^3 - 5 * x^2 + 8 * x - 4)

def domain_h : Set ℝ := {x : ℝ | x ≠ 1 ∧ x ≠ 2}

theorem domain_h_correct : {x : ℝ | ∃ y : ℝ, h x = y} = domain_h := by
  sorry

end domain_h_correct_l505_505932


namespace ring_area_l505_505138

theorem ring_area (r1 r2 : ℝ) (h1 : r1 = 12) (h2 : r2 = 7) (h_pos1 : 0 < r1) (h_pos2 : 0 < r2) :
  (r1^2 * Real.pi) - (r2^2 * Real.pi) = 95 * Real.pi :=
by
  rw [h1, h2]
  norm_num
  rw [pow_two, pow_two]
  rfl

end ring_area_l505_505138


namespace trip_time_difference_l505_505444

-- Define the speed of the motorcycle
def speed : ℤ := 60

-- Define the distances for the two trips
def distance1 : ℤ := 360
def distance2 : ℤ := 420

-- Define the time calculation function
def time (distance speed : ℤ) : ℤ := distance / speed

-- Prove the problem statement
theorem trip_time_difference : (time distance2 speed - time distance1 speed) * 60 = 60 := by
  -- Provide the proof here
  sorry

end trip_time_difference_l505_505444


namespace intersect_pentachoron_with_plane_desargues_l505_505486

-- Define the structure of a pentachoron
structure Pentachoron :=
(points : Finset Point)
(no_four_points_coplanar : ∀ (p1 p2 p3 p4 : Point) (hp1 : p1 ∈ points) (hp2 : p2 ∈ points) (hp3 : p3 ∈ points) (hp4 : p4 ∈ points), 
  ¬Coplanar p1 p2 p3 p4)

-- Assume some basic definitions for points, lines, planes, and their intersections
axiom Point : Type
axiom Line : Type
axiom Plane : Type
axiom Coplanar : Point → Point → Point → Point → Prop
axiom intersect_lines : Plane → Finset Line → Finset Point
axiom defines_Desargues_Configuration : Finset Point → Finset Line → Prop

-- Define the lines and planes given the pentachoron
def pentachoron_lines (p : Pentachoron) : Finset Line := sorry
def pentachoron_planes (p : Pentachoron) : Finset Plane := sorry

-- Assume we have a secant plane intersecting all lines
axiom secant_plane : Plane
axiom plane_intersects_all_lines : ∀ l ∈ pentachoron_lines p, intersects secant_plane l

noncomputable def intersection_of_pentachoron_with_plane (p : Pentachoron) : Finset Point :=
intersect_lines secant_plane (pentachoron_lines p)

theorem intersect_pentachoron_with_plane_desargues (p : Pentachoron) :
  defines_Desargues_Configuration (intersection_of_pentachoron_with_plane p) (pentachoron_lines p) :=
sorry

end intersect_pentachoron_with_plane_desargues_l505_505486


namespace interesting_numbers_perfect_square_l505_505052

/-- A natural number is defined as interesting if all its prime factors are less than 30. -/
def is_interesting (n : ℕ) : Prop :=
  ∀ p ∈ nat.factors n, p < 30

/-- From any set of 10000 interesting numbers, we can always pick out two whose product is a perfect square. -/
theorem interesting_numbers_perfect_square (S : set ℕ) (hS : S.card = 10000) :
  (∀ n ∈ S, is_interesting n) →
  ∃ a b ∈ S, a ≠ b ∧ ∃ k : ℕ, a * b = k^2 :=
begin
  sorry
end

end interesting_numbers_perfect_square_l505_505052


namespace julia_money_left_l505_505728

def initial_amount : ℕ := 40

def amount_spent_on_game (initial : ℕ) : ℕ := initial / 2

def amount_left_after_game (initial : ℕ) (spent_game : ℕ) : ℕ := initial - spent_game

def amount_spent_on_in_game (left_after_game : ℕ) : ℕ := left_after_game / 4

def final_amount (left_after_game : ℕ) (spent_in_game : ℕ) : ℕ := left_after_game - spent_in_game

theorem julia_money_left (initial : ℕ) 
  (h_init : initial = initial_amount)
  (spent_game : ℕ)
  (h_spent_game : spent_game = amount_spent_on_game initial)
  (left_after_game : ℕ)
  (h_left_after_game : left_after_game = amount_left_after_game initial spent_game)
  (spent_in_game : ℕ)
  (h_spent_in_game : spent_in_game = amount_spent_on_in_game left_after_game)
  : final_amount left_after_game spent_in_game = 15 := by 
  sorry

end julia_money_left_l505_505728


namespace artemon_distance_covered_l505_505809

-- Define the condition of the rectangle
def rectangle_side1 : ℝ := 6
def rectangle_side2 : ℝ := 2.5

-- Define the speeds of Malvina, Buratino, and Artemon
def speed_malvina : ℝ := 4
def speed_buratino : ℝ := 6
def speed_artemon : ℝ := 12

-- Using the given conditions, we need to show that Artemon runs 7.8 km before Malvina and Buratino meet
theorem artemon_distance_covered :
  let d := real.sqrt (rectangle_side1^2 + rectangle_side2^2) in
  let t := d / (speed_malvina + speed_buratino) in
  speed_artemon * t = 7.8 :=
by
  let d := real.sqrt (rectangle_side1^2 + rectangle_side2^2)
  let t := d / (speed_malvina + speed_buratino)
  have h: speed_artemon * t = 7.8 := sorry
  exact h

end artemon_distance_covered_l505_505809


namespace f_monotonically_decreasing_f_max_min_interval_l505_505585

def f (x : ℝ) : ℝ := 3 * x^3 - 9 * x + 5

-- Prove that the function f is monotonically decreasing on the interval (-1, 1)
theorem f_monotonically_decreasing : ∀ x, (-1 < x ∧ x < 1) → (f' x < 0) :=
begin
  -- Define the derivative of f
  let f' (x : ℝ) : ℝ := 9 * x^2 - 9,
  sorry -- proof of monotonicity condition
end

-- Prove the maximum and minimum values of f on the interval [-3, 3]
theorem f_max_min_interval : 
  ∃ (x_max x_min : ℝ), 
    x_max ∈ Icc (-3 : ℝ) 3 ∧ 
    x_min ∈ Icc (-3 : ℝ) 3 ∧ 
    (∀ x ∈ Icc (-3 : ℝ) 3, f x ≤ f x_max) ∧ 
    (∀ x ∈ Icc (-3 : ℝ) 3, f x_min ≤ f x) ∧ 
    f x_max = 59 ∧ 
    f x_min = -49 :=
begin
  sorry -- proof of maximum and minimum values
end

end f_monotonically_decreasing_f_max_min_interval_l505_505585


namespace least_integer_with_twelve_factors_l505_505188

open Classical

theorem least_integer_with_twelve_factors : ∃ k : ℕ, (∀ n : ℕ, (count_factors n = 12 → k ≤ n)) ∧ k = 72 :=
by
  sorry

-- Here count_factors is a function that returns the number of positive divisors of a given integer.

end least_integer_with_twelve_factors_l505_505188


namespace least_positive_integer_with_12_factors_l505_505209

/--
Prove that the smallest positive integer with exactly 12 positive factors is 60.
-/
theorem least_positive_integer_with_12_factors : 
  ∃ (n : ℕ), (0 < n) ∧ (nat.factors_count n = 12) ∧ (∀ m : ℕ, (0 < m) ∧ (nat.factors_count m = 12) → n ≤ m) → n = 60 :=
sorry

end least_positive_integer_with_12_factors_l505_505209


namespace julia_money_left_l505_505730

def initial_money : ℕ := 40
def spent_on_game : ℕ := initial_money / 2
def money_left_after_game : ℕ := initial_money - spent_on_game
def spent_on_in_game_purchases : ℕ := money_left_after_game / 4
def final_money_left : ℕ := money_left_after_game - spent_on_in_game_purchases

theorem julia_money_left : final_money_left = 15 := by
  sorry

end julia_money_left_l505_505730


namespace find_f_2008_l505_505578

variable (f : ℝ → ℝ) 
variable (g : ℝ → ℝ) -- g is the inverse of f

def satisfies_conditions (f : ℝ → ℝ) (g : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (g x) = x) ∧ (∀ y : ℝ, g (f y) = y) ∧ 
  (f 9 = 18) ∧ (∀ x : ℝ, g (x + 1) = (f (x + 1)))

theorem find_f_2008 (f : ℝ → ℝ) (g : ℝ → ℝ) 
  (h : satisfies_conditions f g) : f 2008 = -1981 :=
sorry

end find_f_2008_l505_505578


namespace work_done_correct_l505_505427

noncomputable def work_done_by_gas_isothermal_compression : ℝ :=
  let p0 := 103.3 * 10 ^ 3 in  -- initial pressure in Pa
  let H := 0.8 in              -- height of the cylinder in meters
  let h := 0.4 in              -- displacement of the piston in meters
  let R := 0.2 in              -- radius of the cylinder in meters
  let S := Real.pi * R ^ 2 in  -- area of the piston
  p0 * S * H * Real.log(H / (H - h))

theorem work_done_correct:
  work_done_by_gas_isothermal_compression = 7200 :=
sorry

end work_done_correct_l505_505427


namespace baker_sold_more_cakes_than_pastries_l505_505476

theorem baker_sold_more_cakes_than_pastries (cakes_sold pastries_sold : ℕ) 
  (h_cakes_sold : cakes_sold = 158) (h_pastries_sold : pastries_sold = 147) : 
  (cakes_sold - pastries_sold) = 11 := by
  sorry

end baker_sold_more_cakes_than_pastries_l505_505476


namespace least_positive_integer_with_12_factors_l505_505164

theorem least_positive_integer_with_12_factors :
  ∃ m : ℕ, 0 < m ∧ (∀ n : ℕ, 0 < n → (n ≠ m →  m ≤ n)) ∧ (12 = (m.factors.toFinset.card)) := sorry

end least_positive_integer_with_12_factors_l505_505164


namespace middle_cars_occupied_by_l505_505963

-- Constants representing the people.
constant Ellen : Type
constant Maren : Type
constant Aaron : Type
constant Sharon : Type
constant Darren : Type
constant Karen : Type

-- Positions of the cars (1 through 6)
constant P : Type
constant pos1 : P
constant pos6 : P

-- Relation indicating who is in which position.
constant sits_at : Type → P → Prop

-- Relation indicating one person sits directly behind another.
constant directly_behind : Type → Type → Prop

-- The conditions provided as axioms.
axiom Ellen_in_first: sits_at Ellen pos1
axiom Maren_in_last: sits_at Maren pos6
axiom Aaron_behind_Sharon: directly_behind Aaron Sharon
axiom Darren_in_front_of_Aaron: ∃ p1 p2 : P, sits_at Darren p1 ∧ sits_at Aaron p2 ∧ p1 < p2
axiom Darren_Karen_distance: ∃ p1 p2 : P, sits_at Darren p1 ∧ sits_at Karen p2 ∧ |p1 - p2| ≥ 3

-- Final proof statement.
theorem middle_cars_occupied_by : (∃ p3 p4 : P, sits_at Darren p3 ∧ sits_at Aaron p4 ∧ (p3 = 3 ∨ p3 = 4) ∧ (p4 = 3 ∨ p4 = 4)) :=
sorry

end middle_cars_occupied_by_l505_505963


namespace practice_other_days_l505_505466

-- Defining the total practice time for the week and the practice time for two days 
variable (total_minutes_week : ℤ) (total_minutes_two_days : ℤ)

-- Given conditions
axiom total_minutes_week_eq : total_minutes_week = 450
axiom total_minutes_two_days_eq : total_minutes_two_days = 172

-- The proof goal
theorem practice_other_days : (total_minutes_week - total_minutes_two_days) = 278 :=
by
  rw [total_minutes_week_eq, total_minutes_two_days_eq]
  show 450 - 172 = 278
  -- The proof goes here
  sorry

end practice_other_days_l505_505466


namespace sum_sequence_l505_505540

theorem sum_sequence (a : Fin 101 → ℝ) 
    (h1 : a 1 + a 2 = 1)
    (h2 : a 2 + a 3 = 2)
    (h3 : a 3 + a 4 = 3)
    -- ... Similar conditions for a_4 + a_5 = 4 to a_99 + a_100 = 99
    (h98 : a 98 + a 99 = 98)
    (h99 : a 99 + a 100 = 99)
    (h100 : a 100 + a 1 = 100) : 
    (a 1 + a 2 + a 3 + ... + a 100) = 2525 :=
sorry

end sum_sequence_l505_505540


namespace YZ_length_l505_505011

theorem YZ_length : 
  ∀ (X Y Z : Type) 
  (angle_Y angle_Z angle_X : ℝ)
  (XZ YZ : ℝ),
  angle_Y = 45 ∧ angle_Z = 60 ∧ XZ = 6 →
  angle_X = 180 - angle_Y - angle_Z →
  YZ = XZ * (Real.sin angle_X / Real.sin angle_Y) →
  YZ = 3 * (Real.sqrt 6 + Real.sqrt 2) :=
by
  intros X Y Z angle_Y angle_Z angle_X XZ YZ
  intro h1 h2 h3
  sorry

end YZ_length_l505_505011


namespace maximum_w_value_l505_505565

theorem maximum_w_value (x y : ℝ) (h : 2 * x^2 - 6 * x + y^2 = 0) : 
  ∃ w, w = x^2 + y^2 - 8 * x ∧ ∀ x' y' h', 2 * x'^2 - 6 * x' + y'^2 = 0 → (x'^2 + y'^2 - 8 * x' ≤ w) :=
sorry

end maximum_w_value_l505_505565


namespace part_a_l505_505778

theorem part_a : (2^41 + 1) % 83 = 0 :=
  sorry

end part_a_l505_505778


namespace thirty_times_multiple_of_every_integer_is_zero_l505_505873

theorem thirty_times_multiple_of_every_integer_is_zero (n : ℤ) (h : ∀ x : ℤ, n = 30 * x ∧ x = 0 → n = 0) : n = 0 :=
by
  sorry

end thirty_times_multiple_of_every_integer_is_zero_l505_505873


namespace least_positive_integer_with_12_factors_is_96_l505_505270

theorem least_positive_integer_with_12_factors_is_96 :
  ∃ n : ℕ, (n = 96 ∧ (∀ m : ℕ, (m > 0 ∧ ∀ d : ℕ, d ∣ m → d > 0 → d < m → (∃ k : ℕ, d = k ∨ k * d = m) → k > 1 → m > n))) ∧ (∃ m : ℕ, m = 12 ∧ (∀ d : ℕ, d ∣ m → d ∈ {1, m} ∨ ∀ k : ℕ, k * d = m → d = k ∨ d < k)).
sorry

end least_positive_integer_with_12_factors_is_96_l505_505270


namespace first_term_arithmetic_sequence_l505_505034

theorem first_term_arithmetic_sequence
    (a: ℚ)
    (S_n S_2n: ℕ → ℚ)
    (n: ℕ) 
    (h1: ∀ n > 0, S_n n = (n * (2 * a + (n - 1) * 5)) / 2)
    (h2: ∀ n > 0, S_2n (2 * n) = ((2 * n) * (2 * a + ((2 * n) - 1) * 5)) / 2)
    (h3: ∀ n > 0, (S_2n (2 * n)) / (S_n n) = 4) :
  a = 5 / 2 :=
by
  sorry

end first_term_arithmetic_sequence_l505_505034


namespace notepad_last_duration_l505_505469

def note_duration (folds_per_paper : ℕ) (pieces_of_paper : ℕ) (notes_per_day : ℕ) : ℕ :=
  let note_size_papers_per_letter_paper := 2 ^ folds_per_paper
  let total_note_size_papers := pieces_of_paper * note_size_papers_per_letter_paper
  total_note_size_papers / notes_per_day

theorem notepad_last_duration :
  note_duration 3 5 10 = 4 := by
  sorry

end notepad_last_duration_l505_505469


namespace least_positive_integer_with_12_factors_l505_505150

theorem least_positive_integer_with_12_factors : ∃ n : ℕ, (nat.factors_count n = 12 ∧ ∀ m : ℕ, (nat.factors_count m = 12) → n ≤ m) ∧ n = 72 :=
begin
  sorry
end

end least_positive_integer_with_12_factors_l505_505150


namespace unique_polynomial_l505_505983

-- Define the conditions
def valid_polynomial (P : ℝ → ℝ) : Prop :=
  ∃ (p : Polynomial ℝ), Polynomial.degree p > 0 ∧ ∀ (z : ℝ), z ≠ 0 → P z = Polynomial.eval z p

-- The main theorem
theorem unique_polynomial (P : ℝ → ℝ) (hP : valid_polynomial P) :
  (∀ (z : ℝ), z ≠ 0 → P z ≠ 0 → P (1/z) ≠ 0 → 
  1 / P z + 1 / P (1 / z) = z + 1 / z) → ∀ x, P x = x :=
by
  sorry

end unique_polynomial_l505_505983


namespace small_angle_9_15_l505_505614

theorem small_angle_9_15 : 
  let hours_to_angle := 30
  let minute_to_angle := 6
  let minutes := 15
  let hour := 9
  let hour_position := hour * hours_to_angle
  let additional_hour_angle := (hours_to_angle * (minutes / 60.0))
  let final_hour_position := hour_position + additional_hour_angle
  let minute_position := (minutes / 5.0) * hours_to_angle
  let angle := abs (final_hour_position - minute_position)
  let small_angle := if angle > 180 then 360 - angle else angle
  in small_angle = 187.5 := by 
  sorry

end small_angle_9_15_l505_505614


namespace parallel_lines_l505_505418

theorem parallel_lines {L1 L2 L3 : Set Point} (h1: Perpendicular L1 L3) (h2: Perpendicular L2 L3) (h3: CoPlanar L1 L2 L3) : Parallel L1 L2 :=
sorry

end parallel_lines_l505_505418


namespace least_positive_integer_with_12_factors_l505_505148

theorem least_positive_integer_with_12_factors : ∃ n : ℕ, (nat.factors_count n = 12 ∧ ∀ m : ℕ, (nat.factors_count m = 12) → n ≤ m) ∧ n = 72 :=
begin
  sorry
end

end least_positive_integer_with_12_factors_l505_505148


namespace units_digit_of_square_l505_505528

theorem units_digit_of_square (n : ℕ) (h : 1 ≤ n ∧ n ≤ 50) : 
  (∃ t, (units_digit (n^2) = t) ∧ (t = 1 ∨ t = 9)) → 
  (count (λ x, ∃ t, (units_digit (x^2) = t) ∧ (t = 1 ∨ t = 9)) (range 1 51) = 25) :=
by sorry

def units_digit (n : ℕ) : ℕ := n % 10

def range (a b : ℕ) : list ℕ := list.range' a (b - a + 1)

def count {α : Type*} (p : α → Prop) [decidable_pred p] (l : list α) : ℕ :=
l.countp (λ x, p x)

axiom decidable_pred (p : Prop) : decidable p

end units_digit_of_square_l505_505528


namespace four_digit_multiples_of_five_count_l505_505666

theorem four_digit_multiples_of_five_count : 
  {n : ℕ | 1000 ≤ n ∧ n ≤ 9999 ∧ n % 5 = 0}.toFinset.card = 1800 :=
by
  sorry

end four_digit_multiples_of_five_count_l505_505666


namespace midpoint_product_l505_505869

theorem midpoint_product (x1 y1 x2 y2 : ℤ) (hx1 : x1 = 5) (hy1 : y1 = -2) (hx2 : x2 = -3) (hy2 : y2 = 6) :
  let mx := (x1 + x2) / 2
      my := (y1 + y2) / 2
  in mx * my = 2 :=
by
  apply sorry

end midpoint_product_l505_505869


namespace least_positive_integer_with_12_factors_is_972_l505_505368

theorem least_positive_integer_with_12_factors_is_972 : ∃ k : ℕ, (∀ n : ℕ, (∃ (d : ℕ), d * k = n) ↔ n = 12) ∧ k = 972 := sorry

end least_positive_integer_with_12_factors_is_972_l505_505368


namespace least_positive_integer_with_12_factors_l505_505153

theorem least_positive_integer_with_12_factors : ∃ n : ℕ, (nat.factors_count n = 12 ∧ ∀ m : ℕ, (nat.factors_count m = 12) → n ≤ m) ∧ n = 72 :=
begin
  sorry
end

end least_positive_integer_with_12_factors_l505_505153


namespace least_positive_integer_with_12_factors_l505_505174

theorem least_positive_integer_with_12_factors :
  ∃ m : ℕ, 0 < m ∧ (∀ n : ℕ, 0 < n → (n ≠ m →  m ≤ n)) ∧ (12 = (m.factors.toFinset.card)) := sorry

end least_positive_integer_with_12_factors_l505_505174


namespace solve_expression_l505_505874

noncomputable def expression : ℝ :=
  3 + 3 / (1 + 5 / (2 + 3 / (1 + 5 / (2 + 3 / (1 + ... )))))

theorem solve_expression : expression = (1 + Real.sqrt 61) / 2 :=
sorry

end solve_expression_l505_505874


namespace quadratic_no_real_roots_range_l505_505696

theorem quadratic_no_real_roots_range (k : ℝ) : 
  (∀ x : ℝ, ¬ (x^2 + 2 * x - k = 0)) ↔ k < -1 :=
by
  sorry

end quadratic_no_real_roots_range_l505_505696


namespace least_positive_integer_with_12_factors_is_72_l505_505320

-- Definition of having exactly 12 factors for a positive integer
def has_exactly_12_factors (n : ℕ) : Prop := n > 0 ∧ (finset.filter (λ d, n % d = 0) (finset.range (n + 1))).card = 12

-- Statement of the problem in Lean 4
theorem least_positive_integer_with_12_factors_is_72 :
  ∃ n : ℕ, has_exactly_12_factors n ∧ ∀ m : ℕ, has_exactly_12_factors m → n ≤ m :=
begin
  use 72,
  split,
  { -- Prove that 72 has exactly 12 factors
    sorry },
  { -- Prove that 72 is the least such integer
    sorry }
end

end least_positive_integer_with_12_factors_is_72_l505_505320


namespace average_interest_rate_equal_4_09_percent_l505_505908

-- Define the given conditions
def investment_total : ℝ := 5000
def interest_rate_at_3_percent : ℝ := 0.03
def interest_rate_at_5_percent : ℝ := 0.05
def return_relationship (x : ℝ) : Prop := 
  interest_rate_at_5_percent * x = 2 * interest_rate_at_3_percent * (investment_total - x)

-- Define the final statement
theorem average_interest_rate_equal_4_09_percent :
  ∃ x : ℝ, return_relationship x ∧ 
  ((interest_rate_at_5_percent * x + interest_rate_at_3_percent * (investment_total - x)) / investment_total) = 0.04091 := 
by
  sorry

end average_interest_rate_equal_4_09_percent_l505_505908


namespace ratio_green_to_orange_straws_l505_505533

theorem ratio_green_to_orange_straws 
  (red_per_mat orange_per_mat total_straws mats : ℕ)
  (h_red : red_per_mat = 20)
  (h_orange : orange_per_mat = 30)
  (h_total : total_straws = 650)
  (h_mats : mats = 10) :
  let green_per_mat := (total_straws - red_per_mat * mats - orange_per_mat * mats) / mats in
  (green_per_mat : ℚ) / orange_per_mat = 1 / 2 :=
by
  sorry

end ratio_green_to_orange_straws_l505_505533


namespace isosceles_triangle_at_most_one_obtuse_l505_505671

-- Define isosceles triangle
structure IsoscelesTriangle where
  a b c : ℝ
  angleA angleB angleC : ℝ
  equalSides : a = b ∨ b = c ∨ c = a
  sumOfAngles : angleA + angleB + angleC = 180

-- Define obtuse angle
def is_obtuse (angle : ℝ) : Prop :=
  angle > 90

-- Prove an isosceles triangle can have at most one obtuse angle
theorem isosceles_triangle_at_most_one_obtuse (T : IsoscelesTriangle) :
  (is_obtuse T.angleA → ¬ is_obtuse T.angleB ∧ ¬ is_obtuse T.angleC) ∧
  (is_obtuse T.angleB → ¬ is_obtuse T.angleA ∧ ¬ is_obtuse T.angleC) ∧
  (is_obtuse T.angleC → ¬ is_obtuse T.angleA ∧ ¬ is_obtuse T.angleB) :=
sorry

end isosceles_triangle_at_most_one_obtuse_l505_505671


namespace purely_imaginary_solution_l505_505760

noncomputable def z (m : ℝ) : ℂ := log 2 (m^2 - 3*m - 3) + complex.I * log 2 (3 - m)

theorem purely_imaginary_solution (m : ℝ) (h : z m ∈ {z : ℂ | ∃ b : ℝ, z = complex.I * b}) : m = -1 :=
by
  sorry

end purely_imaginary_solution_l505_505760


namespace least_positive_integer_with_12_factors_l505_505203

/--
Prove that the smallest positive integer with exactly 12 positive factors is 60.
-/
theorem least_positive_integer_with_12_factors : 
  ∃ (n : ℕ), (0 < n) ∧ (nat.factors_count n = 12) ∧ (∀ m : ℕ, (0 < m) ∧ (nat.factors_count m = 12) → n ≤ m) → n = 60 :=
sorry

end least_positive_integer_with_12_factors_l505_505203


namespace interchanged_digits_subtraction_l505_505690

theorem interchanged_digits_subtraction (a b k : ℤ) (h1 : 10 * a + b = 2 * k * (a + b)) :
  10 * b + a - 3 * (a + b) = (9 - 4 * k) * (a + b) :=
by sorry

end interchanged_digits_subtraction_l505_505690


namespace solve_equation_in_natural_numbers_l505_505793

-- Define the main theorem
theorem solve_equation_in_natural_numbers :
  (∃ (x y z : ℕ), 2^x + 5^y + 63 = z! ∧ ((x = 5 ∧ y = 2 ∧ z = 5) ∨ (x = 4 ∧ y = 4 ∧ z = 6))) :=
sorry

end solve_equation_in_natural_numbers_l505_505793


namespace length_of_XY_in_terms_of_R_l505_505767

variables {A B C : Type*}
variable [RealGeometry A B C]

noncomputable def triangleXYLength (R : ℝ) : ℝ :=
  let triangle_ABC_is_acute : Prop := True
  let circumcircle_Gamma : Circle := Circle A B C
  let X_is_midpoint_minor_arc_AB : Prop := True
  let Y_is_midpoint_minor_arc_AC : Prop := True
  let XY_tangent_to_incircle : Prop := True
  R * sqrt 3

theorem length_of_XY_in_terms_of_R (R : ℝ)
  (triangle_ABC_is_acute : True)
  (circumcircle_Gamma : Circle)
  (X_is_midpoint_minor_arc_AB : True)
  (Y_is_midpoint_minor_arc_AC : True)
  (XY_tangent_to_incircle : True) :
  triangleXYLength R = R * sqrt 3 := 
  sorry

end length_of_XY_in_terms_of_R_l505_505767


namespace Justine_colored_sheets_l505_505853

theorem Justine_colored_sheets :
  ∀ (total_sheets : ℕ) (binders : ℕ) (colored_fraction : ℚ),
  total_sheets = 2450 →
  binders = 5 →
  colored_fraction = 1 / 2 →
  total_sheets / binders * colored_fraction = 245 := by
  intros total_sheets binders colored_fraction
  intros h1 h2 h3
  have h_total_binders : total_sheets / binders = 490 := by
    rw [h1, h2]
    exact Nat.div_eq_of_eq_mul_right (by norm_num) (by norm_num)
  have h_colored_sheets : 490 * colored_fraction = 245 := by
    rw h3
    exact (by norm_num)
  rw [h_total_binders, h_colored_sheets]
  exact (by norm_num)

end Justine_colored_sheets_l505_505853


namespace least_pos_int_with_12_pos_factors_is_72_l505_505231

def least_positive_integer_with_12_factors (n : ℕ) : Prop :=
  ∃ (m : ℕ), m > 0 ∧ m ≠ n ∧ (∀ d : ℕ, d ∣ m → (d = n ∨ d = 1)) ∧
  ((∀ d : ℕ, d ∣ m → ∃ e : ℕ, e = 1) → n = 72)

theorem least_pos_int_with_12_pos_factors_is_72 (n : ℕ) :
  least_positive_integer_with_12_factors n → n = 72 := by
  sorry

end least_pos_int_with_12_pos_factors_is_72_l505_505231


namespace find_f_2_l505_505100

noncomputable theory

def f (x : ℝ) : ℝ := 
-- the function f, defined according to some function

axiom f_eq : ∀ (x : ℝ), x ≠ 0 → f(x) - 3 * f(1 / x) = 3^x

theorem find_f_2 :
  f 2 = - (9 + 3 * real.sqrt 3) / 8 :=
by
  sorry

end find_f_2_l505_505100


namespace ten_times_hundred_eq_thousand_ten_times_thousand_eq_ten_thousand_hundreds_in_ten_thousand_tens_in_one_thousand_l505_505430

theorem ten_times_hundred_eq_thousand : 10 * 100 = 1000 := 
by sorry

theorem ten_times_thousand_eq_ten_thousand : 10 * 1000 = 10000 := 
by sorry

theorem hundreds_in_ten_thousand : 10000 / 100 = 100 := 
by sorry

theorem tens_in_one_thousand : 1000 / 10 = 100 := 
by sorry

end ten_times_hundred_eq_thousand_ten_times_thousand_eq_ten_thousand_hundreds_in_ten_thousand_tens_in_one_thousand_l505_505430


namespace three_digit_numbers_no_679_l505_505673

theorem three_digit_numbers_no_679 : 
  ∃ (count : ℕ), count = 294 ∧ 
    (∀ n : ℕ, 100 ≤ n ∧ n < 1000 → 
      (n / 100 ≠ 6 ∧ n / 100 ≠ 7 ∧ n / 100 ≠ 9) ∧ 
      ((n / 10 % 10 ≠ 6 ∧ n / 10 % 10 ≠ 7 ∧ n / 10 % 10 ≠ 9) ∧ 
      (n % 10 ≠ 6 ∧ n % 10 ≠ 7 ∧ n % 10 ≠ 9)) → count > 0) := 
begin
  sorry
end

end three_digit_numbers_no_679_l505_505673


namespace polynomial_condition_l505_505511

noncomputable def polynomial_of_degree_le (n : ℕ) (P : Polynomial ℝ) :=
  P.degree ≤ n

noncomputable def has_nonneg_coeff (P : Polynomial ℝ) :=
  ∀ i, 0 ≤ P.coeff i

theorem polynomial_condition
  (n : ℕ) (P : Polynomial ℝ)
  (h1 : polynomial_of_degree_le n P)
  (h2 : has_nonneg_coeff P)
  (h3 : ∀ x : ℝ, x > 0 → P.eval x * P.eval (1 / x) ≤ (P.eval 1) ^ 2) : 
  ∃ a_n : ℝ, 0 ≤ a_n ∧ P = Polynomial.C a_n * Polynomial.X^n :=
sorry

end polynomial_condition_l505_505511


namespace quadratic_two_real_roots_find_m_l505_505603

theorem quadratic_two_real_roots (m : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁ * x₂ = 3 * m^2 ∧ x₁ + x₂ = 4 * m :=
by
  sorry

theorem find_m (m : ℝ) (h : m > 0) (h_diff : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ - x₂ = 2) : m = 1 :=
by
  sorry

end quadratic_two_real_roots_find_m_l505_505603


namespace six_div_one_minus_three_div_ten_equals_twenty_four_l505_505070

theorem six_div_one_minus_three_div_ten_equals_twenty_four :
  (6 : ℤ) / (1 - (3 : ℤ) / (10 : ℤ)) = 24 := 
by
  sorry

end six_div_one_minus_three_div_ten_equals_twenty_four_l505_505070


namespace least_positive_integer_with_12_factors_l505_505379

theorem least_positive_integer_with_12_factors : 
  ∃ (n : ℕ), (n > 0) ∧ (nat.factors_count n = 12) ∧ (∀ m, (m > 0) → (nat.factors_count m = 12) → n ≤ m) ∧ n = 108 :=
sorry

end least_positive_integer_with_12_factors_l505_505379


namespace least_positive_integer_with_12_factors_l505_505377

theorem least_positive_integer_with_12_factors : 
  ∃ (n : ℕ), (n > 0) ∧ (nat.factors_count n = 12) ∧ (∀ m, (m > 0) → (nat.factors_count m = 12) → n ≤ m) ∧ n = 108 :=
sorry

end least_positive_integer_with_12_factors_l505_505377


namespace max_min_value_ratio_l505_505761

noncomputable def f (x : ℝ) : ℝ := (2 * x) / (x - 2)

theorem max_min_value_ratio :
  ∃ M m, M = f 3 ∧ m = f 4 ∧ (∀ x ∈ (Icc 3 4), f(x) ≤ M ∧ f(x) ≥ m) ∧ (m^2 / M = 8 / 3) :=
by
  let M := f 3
  let m := f 4
  use M, m
  have hM : M = f 3 := rfl
  have hm : m = f 4 := rfl
  split
  · exact hM
  split
  · exact hm
  split
  · intro x hx
    split
    · sorry -- Proof that f(x) ≤ M
    · sorry -- Proof that f(x) ≥ m
  · sorry -- Proof that m^2 / M = 8 / 3

end max_min_value_ratio_l505_505761


namespace linear_dependence_over_rationals_l505_505736

variable {n : ℕ}
variable {v : Fin (n + 2) → Fin n → ℝ}

theorem linear_dependence_over_rationals
  (h0 : v 0 = 0)
  (h1 : ∀ i j : Fin (n + 2), ∃ q : ℚ, ‖v i - v j‖₂ = q) :
  ∃ (coeff : Fin (n + 2) → ℚ), (coeff ≠ 0) ∧ (∑ i, coeff i • v i = 0) := 
sorry

end linear_dependence_over_rationals_l505_505736


namespace least_positive_integer_with_12_factors_l505_505329

def has_exactly_12_factors (n : ℕ) : Prop :=
  n.factors.length = 12

theorem least_positive_integer_with_12_factors : ∃ k : ℕ, has_exactly_12_factors k ∧ (∀ n : ℕ, has_exactly_12_factors n → n ≥ k) ∧ k = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_l505_505329


namespace range_of_a_l505_505582

noncomputable def f (a x : ℝ) : ℝ := Real.logBase a (8 - a * x)

theorem range_of_a (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : ∀ x, 1 ≤ x ∧ x ≤ 2 → f a x > 1) :
  1 < a ∧ a < 8 / 3 :=
by
  sorry

end range_of_a_l505_505582


namespace min_value_of_2x_plus_y_l505_505759

open Real

def cauchy_schwarz_ineq (x y : ℝ) (h₀ : 0 < x) (h₁ : 0 < y) (h₂ : 1 / (x + 3) + 1 / (y + 4) = 1 / 2) : Prop :=
  2 * x + y ≥ 1 + 4 * sqrt 2

theorem min_value_of_2x_plus_y (x y : ℝ) (h₀ : 0 < x) (h₁ : 0 < y) (h₂ : 1 / (x + 3) + 1 / (y + 4) = 1 / 2) :
  2 * x + y ≥ 1 + 4 * sqrt 2 :=
cauchy_schwarz_ineq x y h₀ h₁ h₂

end min_value_of_2x_plus_y_l505_505759


namespace unoccupied_volume_of_tank_l505_505807

theorem unoccupied_volume_of_tank (length width height : ℝ) (num_marbles : ℕ) (marble_radius : ℝ) (fill_fraction : ℝ) :
    length = 12 → width = 12 → height = 15 → num_marbles = 5 → marble_radius = 1.5 → fill_fraction = 1/3 →
    (length * width * height * (1 - fill_fraction) - num_marbles * (4 / 3 * Real.pi * marble_radius^3) = 1440 - 22.5 * Real.pi) :=
by
  intros
  sorry

end unoccupied_volume_of_tank_l505_505807


namespace flower_exhibition_arrangements_l505_505960

theorem flower_exhibition_arrangements :
  let volunteers := 6
  let areas := 4
  let cond_a_b : (A B : Nat) := (A = 1) ∧ (B = 1)
  let cond_c_d : (C D : Nat) := (C = 2) ∧ (D = 2)
  let cond_no_together : Prop := ∀ (li wang : Nat), li ≠ wang → ¬(C = li.abs) ∧ ¬(D = wang.abs)
  let total_arrangements := (ucount (x : volunteers × areas), x.fst ≠ x.snd 
    ∧ cond_a_b ∧ cond_c_d ∧ cond_no_together)
  in total_arrangements = 156 := by sorry

end flower_exhibition_arrangements_l505_505960


namespace collinear_points_l505_505137

open EuclideanGeometry

-- Definitions and conditions based on the problem statement
variables {K M T : Point} {r : ℝ}
variables {O1 O2 O3 : Point}
variable {circleLarger : Circle} (radiusLarger : circleLarger.radius = 2 * r)
variable {circleSmaller : Circle} (radiusSmaller : circleSmaller.radius = r)
variable {circleThird : Circle} (radiusThird : circleThird.radius = r)
variable (touches1 : circleLarger.touches circleSmaller K)
variable (touches2 : circleThird.touches circleSmaller M)
variable (intersects : circleThird.intersects circleLarger T)

theorem collinear_points (h1 : touches circleLarger circleSmaller K)
                         (h2 : touches circleThird circleSmaller M)
                         (h3 : intersects circleThird circleLarger T) :
  collinear K M T :=
  sorry

end collinear_points_l505_505137


namespace exists_ratios_leq_and_geq_two_l505_505711

theorem exists_ratios_leq_and_geq_two
  (P P1 P2 P3 : Point)
  (Q1 Q2 Q3 : Point)
  (hP : is_in_triangle P P1 P2 P3)
  (hQ1 : is_intersection P P1 (P2, P3) Q1)
  (hQ2 : is_intersection P P2 (P1, P3) Q2)
  (hQ3 : is_intersection P P3 (P1, P2) Q3) :
  ∃ i j : ℕ, i ∈ {1, 2, 3} ∧ j ∈ {1, 2, 3} ∧ (PP_ratio P P1 Q1 ≤ 2 ∨ PP_ratio P P2 Q2 ≤ 2 ∨ PP_ratio P P3 Q3 ≤ 2) ∧ (PP_ratio P P1 Q1 ≥ 2 ∨ PP_ratio P P2 Q2 ≥ 2 ∨ PP_ratio P P3 Q3 ≥ 2) :=
begin
  sorry
end

end exists_ratios_leq_and_geq_two_l505_505711


namespace coeff_x99_zero_l505_505557

theorem coeff_x99_zero (P Q : ℝ[X]) (h₁ : P.coeff 0 = 1) (h₂ : P^2 = 1 + X + X^100 * Q) :
  ((P + 1)^100).coeff 99 = 0 :=
by
  sorry

end coeff_x99_zero_l505_505557


namespace right_triangle_tangent_circle_l505_505800

theorem right_triangle_tangent_circle 
  (D E F Q : Type) [metric_space D] [has_dist E] [has_dist F] [has_dist Q]
  (h1: right_triangle D E F)
  (h2: dist D F = real.sqrt 85)
  (h3: dist D E = 7)
  (h4: ∃ C : E, circle_tangent_to D E F C)
  (h5: point_where_circle_and_side_meet Q D F) :
  dist F Q = 6 := sorry

end right_triangle_tangent_circle_l505_505800


namespace least_positive_integer_with_12_factors_l505_505330

def has_exactly_12_factors (n : ℕ) : Prop :=
  n.factors.length = 12

theorem least_positive_integer_with_12_factors : ∃ k : ℕ, has_exactly_12_factors k ∧ (∀ n : ℕ, has_exactly_12_factors n → n ≥ k) ∧ k = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_l505_505330


namespace tangent_line_l505_505751

open_locale classical
noncomputable theory

variables (A B C : Point) (I : Point) (E F : Point)

-- conditions
def incenter (I : Point) (A B C : Point) : Prop :=
  -- definition of incenter can go here
  sorry

def circumcircle_intersects (P Q R : Point) (I : Point) : Prop :=
  -- definition of circumcircle intersections can go here
  sorry

-- statement of the problem
theorem tangent_line (A B C I E F : Point)
  (h1 : incenter I A B C)
  (h2 : circumcircle_intersects I B C E F A) :
  tangent (line_through E F) (incircle A B C) :=
sorry

end tangent_line_l505_505751


namespace elevator_probability_diff_floors_l505_505143

theorem elevator_probability_diff_floors :
  let n := 2 -- number of people
  let k := 5 -- number of options from the second to sixth floor
  (∑ i in finset.Ico 1 (k + 1), 1 / (k ^ 2) * (1 - 1 / k)) = 4 / 5 :=
by 
  sorry

end elevator_probability_diff_floors_l505_505143


namespace solve_diophantine_equation_l505_505787

theorem solve_diophantine_equation :
  ∃ (x y : ℤ), x^2 - x * y - 6 * y^2 + 2 * x + 19 * y = 18 ∧ (x = 2 ∧ y = 2 ∨ x = -2 ∧ y = 2) :=
  sorry

end solve_diophantine_equation_l505_505787


namespace scoops_distribution_l505_505674

theorem scoops_distribution : 
  let n := 4 -- Number of scoops
  let k := 4 -- Number of flavors
  in (Nat.choose (n + k - 1) (k - 1) = 35) := 
begin
  -- We define n and k as given in the problem and use the binomial coefficient
  let n := 4,
  let k := 4,
  suffices h : Nat.choose (n + k - 1) (k - 1) = 35, from h,
  sorry
end

end scoops_distribution_l505_505674


namespace least_positive_integer_with_12_factors_l505_505334

def has_exactly_12_factors (n : ℕ) : Prop :=
  n.factors.length = 12

theorem least_positive_integer_with_12_factors : ∃ k : ℕ, has_exactly_12_factors k ∧ (∀ n : ℕ, has_exactly_12_factors n → n ≥ k) ∧ k = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_l505_505334


namespace gloria_pencils_total_l505_505612

-- Define the number of pencils Gloria initially has.
def pencils_gloria_initial : ℕ := 2

-- Define the number of pencils Lisa initially has.
def pencils_lisa_initial : ℕ := 99

-- Define the final number of pencils Gloria will have after receiving all of Lisa's pencils.
def pencils_gloria_final : ℕ := pencils_gloria_initial + pencils_lisa_initial

-- Prove that the final number of pencils Gloria will have is 101.
theorem gloria_pencils_total : pencils_gloria_final = 101 :=
by sorry

end gloria_pencils_total_l505_505612


namespace number_of_episodes_l505_505861

def episode_length : ℕ := 20
def hours_per_day : ℕ := 2
def days : ℕ := 15

theorem number_of_episodes : (days * hours_per_day * 60) / episode_length = 90 :=
by
  sorry

end number_of_episodes_l505_505861


namespace number_of_people_for_cheaper_second_caterer_l505_505068

theorem number_of_people_for_cheaper_second_caterer : 
  ∃ (x : ℕ), (150 + 20 * x > 250 + 15 * x + 50) ∧ 
  ∀ (y : ℕ), (y < x → ¬ (150 + 20 * y > 250 + 15 * y + 50)) :=
by
  sorry

end number_of_people_for_cheaper_second_caterer_l505_505068


namespace least_positive_integer_with_12_factors_is_972_l505_505371

theorem least_positive_integer_with_12_factors_is_972 : ∃ k : ℕ, (∀ n : ℕ, (∃ (d : ℕ), d * k = n) ↔ n = 12) ∧ k = 972 := sorry

end least_positive_integer_with_12_factors_is_972_l505_505371


namespace four_digit_multiples_of_5_count_l505_505636

-- Define the range of four-digit numbers
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

-- Define a multiple of 5
def is_multiple_of_five (n : ℕ) : Prop := n % 5 = 0

-- Define the required proof problem
theorem four_digit_multiples_of_5_count : 
  ∃ n : ℕ, (n = 1800) ∧ (∀ k : ℕ, is_four_digit k ∧ is_multiple_of_five k → n = 1800) :=
sorry

end four_digit_multiples_of_5_count_l505_505636


namespace num_ways_choose_five_distinct_l505_505532

theorem num_ways_choose_five_distinct (S : Finset ℕ) (h : S.card = 20) :
  (∃ (l : List ℕ), l.length = 5 ∧ (∀ i j, i ≠ j → (i ∈ l → j ∈ l) → |i - j| ≥ 4)) →
  (Finset.card (Finset.filter (λ l : List ℕ, l.length = 5 ∧ (∀ i j, i ≠ j → (i ∈ l → j ∈ l) → |i - j| ≥ 4)) (Finset.powersetLen 5 S)) = 56) :=
sorry

end num_ways_choose_five_distinct_l505_505532


namespace semicircle_area_ratio_l505_505894

theorem semicircle_area_ratio (r : ℝ) (h₁ : r = 10) (h₂ : ∀ (d₁ d₂ : ℝ), d₁ = d₂ ∧ d₁ ⊥ d₂) :
  ( 2 * (π * (r/2)^2 / 2) ) / (π * r^2) = 1/4 :=
by
  rw [h₁, ← h₂, π, rfl]
  sorry

end semicircle_area_ratio_l505_505894


namespace Math_Proof_Problem_l505_505996

open Real

-- Definitions for the vectors a and b
def a (x : ℝ) : ℝ × ℝ := (2 * cos x, cos x - sqrt 3 * sin x)
def b (x : ℝ) : ℝ × ℝ := (sin (x + π / 3), sin x)

-- Definition for function f
def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

-- Statement for the proof problem
theorem Math_Proof_Problem :
  (∀ x, f (x + π) = f x) ∧ 
  (∀ k : ℤ, ∀ x, x ∈ set.Icc (k * π + π / 12) (k * π + 7 * π / 12) → deriv f x < 0) ∧
  (∀ φ ∈ set.Icc 0 π, (∀ x, f (x + φ) = f (-x + φ)) ↔ (φ = π / 12 ∨ φ = 7 * π / 12)) :=
by sorry

end Math_Proof_Problem_l505_505996


namespace least_positive_integer_with_12_factors_is_972_l505_505365

theorem least_positive_integer_with_12_factors_is_972 : ∃ k : ℕ, (∀ n : ℕ, (∃ (d : ℕ), d * k = n) ↔ n = 12) ∧ k = 972 := sorry

end least_positive_integer_with_12_factors_is_972_l505_505365


namespace largest_interval_invertible_l505_505946

def g (x : ℝ) : ℝ := 3 * x^2 + 6 * x - 9

theorem largest_interval_invertible :
  ∃ I : set ℝ, I = {x | x ≤ -1} ∧ (∀ x y ∈ I, g x = g y → x = y) :=
by
  -- Proof goes here
  sorry

end largest_interval_invertible_l505_505946


namespace b_investment_calculation_l505_505423

noncomputable def total_profit : ℝ := 9600
noncomputable def A_investment : ℝ := 2000
noncomputable def A_management_fee : ℝ := 0.10 * total_profit
noncomputable def remaining_profit : ℝ := total_profit - A_management_fee
noncomputable def A_total_received : ℝ := 4416
noncomputable def B_investment : ℝ := 1000

theorem b_investment_calculation (B: ℝ) 
  (h_total_profit: total_profit = 9600)
  (h_A_investment: A_investment = 2000)
  (h_A_management_fee: A_management_fee = 0.10 * total_profit)
  (h_remaining_profit: remaining_profit = total_profit - A_management_fee)
  (h_A_total_received: A_total_received = 4416)
  (h_A_total_formula : A_total_received = A_management_fee + (A_investment / (A_investment + B)) * remaining_profit) :
  B = 1000 :=
by
  have h1 : total_profit = 9600 := h_total_profit
  have h2 : A_investment = 2000 := h_A_investment
  have h3 : A_management_fee = 0.10 * total_profit := h_A_management_fee
  have h4 : remaining_profit = total_profit - A_management_fee := h_remaining_profit
  have h5 : A_total_received = 4416 := h_A_total_received
  have h6 : A_total_received = A_management_fee + (A_investment / (A_investment + B)) * remaining_profit := h_A_total_formula
  
  sorry

end b_investment_calculation_l505_505423


namespace roots_of_polynomial_l505_505981

noncomputable def polynomial (x : ℝ) : ℝ := 3 * x^4 + 17 * x^3 - 32 * x^2 - 12 * x

theorem roots_of_polynomial : 
  (0 ∈ set_of (λ x, polynomial x = 0)) ∧ 
  (-1/2 ∈ set_of (λ x, polynomial x = 0)) ∧ 
  (4/3 ∈ set_of (λ x, polynomial x = 0)) ∧ 
  (-3 ∈ set_of (λ x, polynomial x = 0)) :=
by sorry

end roots_of_polynomial_l505_505981


namespace least_positive_integer_with_12_factors_l505_505251

theorem least_positive_integer_with_12_factors :
  ∃ k : ℕ, (1 ≤ k) ∧ (nat.factors k).length = 12 ∧
    ∀ n : ℕ, (1 ≤ n) ∧ (nat.factors n).length = 12 → k ≤ n := 
begin
  sorry
end

end least_positive_integer_with_12_factors_l505_505251


namespace quadratic_has_real_roots_find_value_of_m_l505_505597

theorem quadratic_has_real_roots (m : ℝ) : 
  ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (x1 - x2) ≠ 0 ∧ (x^2 - 4 * m * x + 3 * m^2 = 0) := 
by 
  sorry

theorem find_value_of_m (m : ℝ) (h1 : m > 0) (h2 : ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (x1 - x2 = 2)) :
  m = 1 := 
by 
  sorry

end quadratic_has_real_roots_find_value_of_m_l505_505597


namespace infinite_geometric_series_second_term_l505_505922

theorem infinite_geometric_series_second_term (a r S : ℝ) (h1 : r = 1 / 4) (h2 : S = 16) (h3 : S = a / (1 - r)) : a * r = 3 := 
sorry

end infinite_geometric_series_second_term_l505_505922


namespace min_value_expr_l505_505978

theorem min_value_expr (x y : ℝ) : 
  ∃ x y : ℝ, (x, y) = (4, 0) ∧ (∀ x y : ℝ, x^2 + 4 * x * y + 5 * y^2 - 8 * x - 6 * y ≥ -22) :=
by
  sorry

end min_value_expr_l505_505978


namespace f_odd_function_l505_505609

def operation_1 (a b : ℝ) := sqrt (a^2 - b^2)
def operation_2 (a b : ℝ) := sqrt ((a - b)^2)

def f (x : ℝ) := (operation_1 2 x) / (2 - (operation_2 x 2))

theorem f_odd_function : ∀ x : ℝ, f (-x) = -f x :=
by
  sorry

end f_odd_function_l505_505609


namespace part1_l505_505007

noncomputable def a : ℕ → ℝ
| 1 => 1
| (n + 1) => a n / (a n + 1)

def seq_is_arithmetic (a : ℕ → ℝ) : Prop := ∀ n, 1 / a (n + 1) - 1 / a n = 1

noncomputable def a_general_term (n : ℕ) : ℝ := 1 / n

theorem part1 (n : ℕ) : seq_is_arithmetic a ∧ (∀ n, a n = a_general_term n) := sorry

end part1_l505_505007


namespace log4_16_eq_2_l505_505965

theorem log4_16_eq_2 : log 4 16 = 2 := sorry

end log4_16_eq_2_l505_505965


namespace sum_geometric_sequence_l505_505033

theorem sum_geometric_sequence (a_1 r : ℝ) (n : ℕ) (h: 8 * a_1 * r + a_1 * r ^ 4 = 0) :
  let S_n := a_1 * (1 - r ^ n) / (1 - r) in
  r = -2 → S_n = a_1 * (1 + 2 ^ n) / 3 :=
by 
  sorry

end sum_geometric_sequence_l505_505033


namespace least_positive_integer_with_12_factors_l505_505291

def num_factors (n : ℕ) : ℕ := 
  (finset.range (n + 1)).filter (λ d, n % d = 0).card

theorem least_positive_integer_with_12_factors : 
  ∃ n : ℕ, num_factors n = 12 ∧ n = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_l505_505291


namespace least_positive_integer_with_12_factors_l505_505292

def has_exactly_12_factors (n : ℕ) : Prop :=
  (finset.range(n + 1).filter (λ d, n % d = 0)).card = 12

theorem least_positive_integer_with_12_factors : ∀ n : ℕ, has_exactly_12_factors n → n ≥ 150 :=
sorry

end least_positive_integer_with_12_factors_l505_505292


namespace practice_minutes_other_days_l505_505463

-- Definitions based on given conditions
def total_hours_practiced : ℕ := 7.5 * 60 -- converting hours to minutes
def minutes_per_day := 86
def days_practiced := 2

-- Lean 4 statement for the proof problem
theorem practice_minutes_other_days :
  let total_minutes := total_hours_practiced
  let minutes_2_days := minutes_per_day * days_practiced
  total_minutes - minutes_2_days = 278 := by
  sorry

end practice_minutes_other_days_l505_505463


namespace least_positive_integer_with_12_factors_is_972_l505_505360

theorem least_positive_integer_with_12_factors_is_972 : ∃ k : ℕ, (∀ n : ℕ, (∃ (d : ℕ), d * k = n) ↔ n = 12) ∧ k = 972 := sorry

end least_positive_integer_with_12_factors_is_972_l505_505360


namespace count_four_digit_multiples_of_5_l505_505633

def is_multiple_of_5 (n : ℕ) : Prop :=
  n % 5 = 0

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def four_digit_multiples_of_5_count (lower upper : ℕ) :=
  (upper - lower + 1)

theorem count_four_digit_multiples_of_5 : 
  four_digit_multiples_of_5_count 200 1999 = 1800 :=
by
  sorry

end count_four_digit_multiples_of_5_l505_505633


namespace largest_order_of_permutation_of_size_11_l505_505047

-- Definition of the permutation order
def permutation_order {α : Type*} [Fintype α] [DecidableEq α] (p : Equiv.Perm α) : ℕ :=
  Classical.find (exists_pow_eq_one p)

-- Problem Statement
theorem largest_order_of_permutation_of_size_11 :
  ∃ p : Equiv.Perm (Fin 11), permutation_order p = 30 :=
sorry

end largest_order_of_permutation_of_size_11_l505_505047


namespace count_circles_tangent_to_C3_C4_l505_505757

def circle (center : ℝ × ℝ) (radius : ℝ) : Prop := 
  ∃ (x y : ℝ), center = (x, y) ∧ radius ≥ 0

variables (C3_center C4_center : ℝ × ℝ)
variables (r3 r4 : ℝ)
variable (distance_between_C3_C4_centers : ℝ)
variable is_tangent : ∃ (p : ℝ × ℝ), 
  (circle C3_center r3) ∧ 
  (circle C4_center r4) ∧ 
  r3 = 2 ∧ 
  r4 = 2 ∧ 
  p ∈ { c3 | circle c3 2 } ∧ 
  p ∈ { c4 | circle c4 2 } ∧ 
  distance_between_C3_C4_centers = 4

theorem count_circles_tangent_to_C3_C4 : ∃ n : ℕ, 
  (∀ C : ℝ × ℝ, ∀ r : ℝ, circle C r ∧ r = 3 → 
  (C ∈ { z | dist z (C3_center) = r + 2 ∨ dist z (C3_center) = 2 - r } ∧ 
   C ∈ { z | dist z (C4_center) = r + 2 ∨ dist z (C4_center) = 2 - r })) ∧ 
   n = 6 :=
sorry

end count_circles_tangent_to_C3_C4_l505_505757


namespace least_positive_integer_with_12_factors_l505_505258

theorem least_positive_integer_with_12_factors :
  ∃ k : ℕ, (1 ≤ k) ∧ (nat.factors k).length = 12 ∧
    ∀ n : ℕ, (1 ≤ n) ∧ (nat.factors n).length = 12 → k ≤ n := 
begin
  sorry
end

end least_positive_integer_with_12_factors_l505_505258


namespace triangle_inequality_l505_505758

variables {a b c x y z : ℝ}

theorem triangle_inequality 
  (h1 : ∀ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a)
  (h2 : x + y + z = 0) :
  a^2 * y * z + b^2 * z * x + c^2 * x * y ≤ 0 :=
sorry

end triangle_inequality_l505_505758


namespace artemon_distance_covered_l505_505811

-- Definition of the problem conditions
def rectangle_side_a : ℝ := 6
def rectangle_side_b : ℝ := 2.5
def malvina_speed : ℝ := 4
def buratino_speed : ℝ := 6
def artemon_speed : ℝ := 12

-- Computation of the diagonal distance
def diagonal_distance : ℝ :=
  real.sqrt (rectangle_side_a ^ 2 + rectangle_side_b ^ 2)

-- Computation of the relative speed and the time taken for Malvina and Buratino to meet
def relative_speed : ℝ :=
  malvina_speed + buratino_speed

def meeting_time : ℝ :=
  diagonal_distance / relative_speed

-- Computation of the distance covered by Artemon
def artemon_distance : ℝ :=
  artemon_speed * meeting_time

-- The theorem that we need to prove
theorem artemon_distance_covered : artemon_distance = 7.8 := 
sorry

end artemon_distance_covered_l505_505811


namespace least_positive_integer_with_12_factors_l505_505200

/--
Prove that the smallest positive integer with exactly 12 positive factors is 60.
-/
theorem least_positive_integer_with_12_factors : 
  ∃ (n : ℕ), (0 < n) ∧ (nat.factors_count n = 12) ∧ (∀ m : ℕ, (0 < m) ∧ (nat.factors_count m = 12) → n ≤ m) → n = 60 :=
sorry

end least_positive_integer_with_12_factors_l505_505200


namespace always_two_real_roots_find_m_l505_505600

-- Given quadratic equation: x^2 - 4mx + 3m^2 = 0
-- Definitions for the problem
def quadratic_eq (m x : ℝ) : Prop := x^2 - 4 * m * x + 3 * m^2 = 0

-- Q1: Prove that this equation always has two real roots.
theorem always_two_real_roots (m : ℝ) : ∃ x₁ x₂ : ℝ, quadratic_eq m x₁ ∧ quadratic_eq m x₂ :=
by
  sorry

-- Q2: If m > 0 and the difference between the two real roots is 2, find the value of m.
theorem find_m (m : ℝ) (h₁ : m > 0) (h₂ : ∃ x₁ x₂ : ℝ, quadratic_eq m x₁ ∧ quadratic_eq m x₂ ∧ |x₁ - x₂| = 2) : m = 1 :=
by
  sorry

end always_two_real_roots_find_m_l505_505600


namespace least_positive_integer_with_12_factors_is_972_l505_505363

theorem least_positive_integer_with_12_factors_is_972 : ∃ k : ℕ, (∀ n : ℕ, (∃ (d : ℕ), d * k = n) ↔ n = 12) ∧ k = 972 := sorry

end least_positive_integer_with_12_factors_is_972_l505_505363


namespace least_positive_integer_with_12_factors_l505_505298

def has_exactly_12_factors (n : ℕ) : Prop :=
  (finset.range(n + 1).filter (λ d, n % d = 0)).card = 12

theorem least_positive_integer_with_12_factors : ∀ n : ℕ, has_exactly_12_factors n → n ≥ 150 :=
sorry

end least_positive_integer_with_12_factors_l505_505298


namespace least_positive_integer_with_12_factors_l505_505354

theorem least_positive_integer_with_12_factors : ∃ k : ℕ, k > 0 ∧ (∀ n : ℕ, n > 0 → (factors_count n = 12 → k ≤ n)) ∧ k = 72 :=
by {
  sorry,
}

end least_positive_integer_with_12_factors_l505_505354


namespace AP_is_angle_bisector_BPM_l505_505737

variables {A B C P M N : Point}
variables [triangle : Triangle ABC]
variables [interior_point : InteriorPointOfTriangle P ABC]
variables [midpoint_M : Midpoint M A C]
variables [midpoint_N : Midpoint N B C]

-- Declaring the conditions
axioms (angle_BPC_90 : angle B P C = 90)
       (angle_BAP_eq_BCP : angle B A P = angle B C P)
       (BP_eq_2PN : dist B P = 2 * dist P N)

-- The required proof of angle bisector property
theorem AP_is_angle_bisector_BPM :
  is_angle_bisector (line_through A P) (angle B P M) :=
sorry

end AP_is_angle_bisector_BPM_l505_505737


namespace least_positive_integer_with_12_factors_l505_505294

def has_exactly_12_factors (n : ℕ) : Prop :=
  (finset.range(n + 1).filter (λ d, n % d = 0)).card = 12

theorem least_positive_integer_with_12_factors : ∀ n : ℕ, has_exactly_12_factors n → n ≥ 150 :=
sorry

end least_positive_integer_with_12_factors_l505_505294


namespace least_positive_integer_with_12_factors_is_72_l505_505314

-- Definition of having exactly 12 factors for a positive integer
def has_exactly_12_factors (n : ℕ) : Prop := n > 0 ∧ (finset.filter (λ d, n % d = 0) (finset.range (n + 1))).card = 12

-- Statement of the problem in Lean 4
theorem least_positive_integer_with_12_factors_is_72 :
  ∃ n : ℕ, has_exactly_12_factors n ∧ ∀ m : ℕ, has_exactly_12_factors m → n ≤ m :=
begin
  use 72,
  split,
  { -- Prove that 72 has exactly 12 factors
    sorry },
  { -- Prove that 72 is the least such integer
    sorry }
end

end least_positive_integer_with_12_factors_is_72_l505_505314


namespace solution_set_inequality_l505_505950

noncomputable def f : ℝ → ℝ := sorry  -- Define f according to the problem condition.

def g (f : ℝ → ℝ) : ℝ → ℝ := λ x, exp x * f x - exp x

theorem solution_set_inequality {f : ℝ → ℝ} (h₁ : ∀ x, f x + deriv f x > 1) 
    (h₂ : f 0 = 2017) : 
    {x : ℝ | g f x > 2016} = Ioi 0 := 
  sorry

end solution_set_inequality_l505_505950


namespace no_integer_pairs_satisfy_equation_l505_505979

theorem no_integer_pairs_satisfy_equation :
  ∀ (m n : ℤ), ¬(m^3 + 10 * m^2 + 11 * m + 2 = 81 * n^3 + 27 * n^2 + 3 * n - 8) :=
by
  sorry

end no_integer_pairs_satisfy_equation_l505_505979


namespace least_pos_int_with_12_pos_factors_is_72_l505_505228

def least_positive_integer_with_12_factors (n : ℕ) : Prop :=
  ∃ (m : ℕ), m > 0 ∧ m ≠ n ∧ (∀ d : ℕ, d ∣ m → (d = n ∨ d = 1)) ∧
  ((∀ d : ℕ, d ∣ m → ∃ e : ℕ, e = 1) → n = 72)

theorem least_pos_int_with_12_pos_factors_is_72 (n : ℕ) :
  least_positive_integer_with_12_factors n → n = 72 := by
  sorry

end least_pos_int_with_12_pos_factors_is_72_l505_505228


namespace sum_of_rectangles_areas_l505_505985

theorem sum_of_rectangles_areas :
  let widths := [3, 3, 3, 3, 3]
  let lengths := [1, 9, 25, 49, 81]
  let areas := list.map2 (*) widths lengths
  list.sum areas = 495 :=
by
  let widths := [3, 3, 3, 3, 3]
  let lengths := [1, 9, 25, 49, 81]
  let areas := list.map2 (*) widths lengths
  have hareas : areas = [3 * 1, 3 * 9, 3 * 25, 3 * 49, 3 * 81] := by rfl
  have hsum : list.sum areas = 3 + 27 + 75 + 147 + 243 := by
    rw hareas
    refl
  have : 3 + 27 + 75 + 147 + 243 = 495 := by 
    sorry  -- the proof that 3 + 27 + 75 + 147 + 243 = 495
  rw this at hsum
  exact hsum

end sum_of_rectangles_areas_l505_505985


namespace area_not_less_of_acute_triangles_l505_505607

noncomputable def heron_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  in real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem area_not_less_of_acute_triangles
  {A B C D E F : ℝ}
  (h1 : 0 < A ∧ 0 < B ∧ 0 < C ∧ (heron_area A B C) > 0)
  (h2 : 0 < D ∧ 0 < E ∧ 0 < F ∧ (heron_area D E F) > 0)
  (h3 : C ≥ F) (h4 : A ≥ D) (h5 : B ≥ E) :
  heron_area A B C ≥ heron_area D E F :=
by sorry

end area_not_less_of_acute_triangles_l505_505607


namespace jackson_maximum_usd_l505_505015

-- Define the rates for chores in various currencies
def usd_per_hour : ℝ := 5
def gbp_per_hour : ℝ := 3
def jpy_per_hour : ℝ := 400
def eur_per_hour : ℝ := 4

-- Define the hours Jackson worked for each task
def usd_hours_vacuuming : ℝ := 2 * 2
def gbp_hours_washing_dishes : ℝ := 0.5
def jpy_hours_cleaning_bathroom : ℝ := 1.5
def eur_hours_sweeping_yard : ℝ := 1

-- Define the exchange rates over three days
def exchange_rates_day1 := (1.35, 0.009, 1.18)  -- (GBP to USD, JPY to USD, EUR to USD)
def exchange_rates_day2 := (1.38, 0.0085, 1.20)
def exchange_rates_day3 := (1.33, 0.0095, 1.21)

-- Define a function to convert currency to USD based on best exchange rates
noncomputable def max_usd (gbp_to_usd jpy_to_usd eur_to_usd : ℝ) : ℝ :=
  (usd_hours_vacuuming * usd_per_hour) +
  (gbp_hours_washing_dishes * gbp_per_hour * gbp_to_usd) +
  (jpy_hours_cleaning_bathroom * jpy_per_hour * jpy_to_usd) +
  (eur_hours_sweeping_yard * eur_per_hour * eur_to_usd)

-- Prove the maximum USD Jackson can have by choosing optimal rates is $32.61
theorem jackson_maximum_usd : max_usd 1.38 0.0095 1.21 = 32.61 :=
by
  sorry

end jackson_maximum_usd_l505_505015


namespace part1_part2_l505_505777

-- Define the function f(x) and the conditions for propositions P and Q
def f (a x : ℝ) : ℝ := x^2 - 2*a*x + 2

-- Proposition P: Function f has exactly one zero point in the interval [0, 1]
def prop_P (a : ℝ) : Prop := ∃! x ∈ Icc (0 : ℝ) 1, f a x = 0

-- Proposition Q: Function y = a^x (a > 0, a ≠ 1) is increasing on ℝ
def prop_Q (a : ℝ) : Prop := 0 < a ∧ a ≠ 1 ∧ ∀ x y : ℝ, x < y → real.exp (log a * x) < real.exp (log a * y)

-- Part (1): If f(1) = 0, then a = 3/2
theorem part1 (a : ℝ) : f a 1 = 0 → a = 3 / 2 :=
by
  sorry

-- Part (2): If "P or Q" is true and "P and Q" is false, the range of values for a is 1 < a < 3/2
theorem part2 (a : ℝ) : (prop_P a ∨ prop_Q a) ∧ ¬ (prop_P a ∧ prop_Q a) → 1 < a ∧ a < 3 / 2 :=
by
  sorry

end part1_part2_l505_505777


namespace least_positive_integer_with_12_factors_is_96_l505_505400

def has_exactly_12_factors (n : ℕ) : Prop :=
  (12 : ℕ) = (List.range (n + 1)).filter (λ x, n % x = 0).length

theorem least_positive_integer_with_12_factors_is_96 :
  n ∈ (ℕ) ∧ has_exactly_12_factors n → n = 96 :=
by
  sorry

end least_positive_integer_with_12_factors_is_96_l505_505400


namespace largest_solution_l505_505977

-- Define the largest solution to the equation |5x - 3| = 28 as 31/5.
theorem largest_solution (x : ℝ) (h : |5 * x - 3| = 28) : x ≤ 31 / 5 := 
  sorry

end largest_solution_l505_505977


namespace domain_of_log_sqrt_l505_505097

theorem domain_of_log_sqrt (x : ℝ) : (-1 < x ∧ x ≤ 3) ↔ (0 < x + 1 ∧ 3 - x ≥ 0) :=
by
  sorry

end domain_of_log_sqrt_l505_505097


namespace total_books_l505_505856

-- Conditions
def TimsBooks : Nat := 44
def SamsBooks : Nat := 52
def AlexsBooks : Nat := 65
def KatiesBooks : Nat := 37

-- Theorem Statement
theorem total_books :
  TimsBooks + SamsBooks + AlexsBooks + KatiesBooks = 198 :=
by
  sorry

end total_books_l505_505856


namespace second_term_is_three_l505_505924

-- Given conditions
variables (r : ℝ) (S : ℝ)
hypothesis hr : r = 1 / 4
hypothesis hS : S = 16

-- Definition of the first term a
noncomputable def first_term (r : ℝ) (S : ℝ) : ℝ :=
  S * (1 - r)

-- Definition of the second term
noncomputable def second_term (r : ℝ) (a : ℝ) : ℝ :=
  a * r

-- Prove that the second term is 3
theorem second_term_is_three : second_term r (first_term r S) = 3 :=
by
  rw [first_term, second_term]
  sorry

end second_term_is_three_l505_505924


namespace number_of_students_l505_505130

theorem number_of_students (total_stars : ℕ) (stars_per_student : ℕ) (h1 : total_stars = 372) (h2 : stars_per_student = 3) : total_stars / stars_per_student = 124 :=
by
  sorry

end number_of_students_l505_505130


namespace count_four_digit_multiples_of_5_l505_505632

def is_multiple_of_5 (n : ℕ) : Prop :=
  n % 5 = 0

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def four_digit_multiples_of_5_count (lower upper : ℕ) :=
  (upper - lower + 1)

theorem count_four_digit_multiples_of_5 : 
  four_digit_multiples_of_5_count 200 1999 = 1800 :=
by
  sorry

end count_four_digit_multiples_of_5_l505_505632


namespace repeating_decimal_as_fraction_l505_505503

theorem repeating_decimal_as_fraction : 
  let x := 0.474747... in x = 47 / 99 :=
by
  sorry

end repeating_decimal_as_fraction_l505_505503


namespace max_projection_area_l505_505103

theorem max_projection_area (a b c : ℝ) : 
  (∃ (P : set (ℝ × ℝ × ℝ)), is_rectangular_parallelepiped P a b c ∧ area_of_rectangular_projection P = √(a^2 * b^2 + b^2 * c^2 + c^2 * a^2)) :=
sorry

end max_projection_area_l505_505103


namespace least_positive_integer_with_12_factors_is_96_l505_505262

theorem least_positive_integer_with_12_factors_is_96 :
  ∃ n : ℕ, (n = 96 ∧ (∀ m : ℕ, (m > 0 ∧ ∀ d : ℕ, d ∣ m → d > 0 → d < m → (∃ k : ℕ, d = k ∨ k * d = m) → k > 1 → m > n))) ∧ (∃ m : ℕ, m = 12 ∧ (∀ d : ℕ, d ∣ m → d ∈ {1, m} ∨ ∀ k : ℕ, k * d = m → d = k ∨ d < k)).
sorry

end least_positive_integer_with_12_factors_is_96_l505_505262


namespace binomial_coeff_sum_abs_l505_505030

theorem binomial_coeff_sum_abs (a_0 a_1 a_2 a_3 a_4 a_5 a_6 : ℤ)
  (h : (2 * x - 1)^6 = a_6 * x^6 + a_5 * x^5 + a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a_0):
  |a_0| + |a_1| + |a_2| + |a_3| + |a_4| + |a_5| + |a_6| = 729 :=
by
  sorry

end binomial_coeff_sum_abs_l505_505030


namespace least_positive_integer_with_12_factors_l505_505171

theorem least_positive_integer_with_12_factors :
  ∃ m : ℕ, 0 < m ∧ (∀ n : ℕ, 0 < n → (n ≠ m →  m ≤ n)) ∧ (12 = (m.factors.toFinset.card)) := sorry

end least_positive_integer_with_12_factors_l505_505171


namespace solve_eq_integers_l505_505789

theorem solve_eq_integers (x y : ℤ) : 
    x^2 - x * y - 6 * y^2 + 2 * x + 19 * y = 18 ↔ (x = 2 ∧ y = 2) ∨ (x = -2 ∧ y = 2) := by
    sorry

end solve_eq_integers_l505_505789


namespace least_integer_with_twelve_factors_l505_505186

open Classical

theorem least_integer_with_twelve_factors : ∃ k : ℕ, (∀ n : ℕ, (count_factors n = 12 → k ≤ n)) ∧ k = 72 :=
by
  sorry

-- Here count_factors is a function that returns the number of positive divisors of a given integer.

end least_integer_with_twelve_factors_l505_505186


namespace right_triangle_tangent_circle_l505_505798

theorem right_triangle_tangent_circle 
  (D E F Q : Type) [metric_space D] [has_dist E] [has_dist F] [has_dist Q]
  (h1: right_triangle D E F)
  (h2: dist D F = real.sqrt 85)
  (h3: dist D E = 7)
  (h4: ∃ C : E, circle_tangent_to D E F C)
  (h5: point_where_circle_and_side_meet Q D F) :
  dist F Q = 6 := sorry

end right_triangle_tangent_circle_l505_505798


namespace discount_percentage_in_february_l505_505906

variable (C D : ℝ)

-- Conditions
def initial_price := 1.20 * C
def new_year_price := 1.50 * C
def final_price := new_year_price * (1 - D)
def profit := 0.38 * C

-- Goal
theorem discount_percentage_in_february (h : final_price = C + profit) : D = 0.08 := by
  sorry

end discount_percentage_in_february_l505_505906


namespace solution_set_of_inequality_l505_505577

noncomputable def f : ℝ → ℝ := sorry

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

def zero_at_one (f : ℝ → ℝ) : Prop :=
  f 1 = 0

def inequality_holds (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → (x * (deriv^[2] f x) - f x) / (x^2) > 0

theorem solution_set_of_inequality 
  (h1 : odd_function f)
  (h2 : zero_at_one f)
  (h3 : inequality_holds f) :
  { x : ℝ | f x > 0 } = set.Ioo (-1) 0 ∪ set.Ioi 1 := 
sorry

end solution_set_of_inequality_l505_505577


namespace part1_part2_l505_505559

variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}

def sequence_def (a : ℕ → ℕ) : Prop :=
  a 1 = 2 ∧ ∀ n, a (n + 1) = a n + 2 ^ n + 2

def arithmetic_sequence (b : ℕ → ℕ) : Prop :=
  ∃ d, ∀ n, b (n + 1) - b n = d

theorem part1 (a : ℕ → ℕ) (h : sequence_def a) : 
  arithmetic_sequence (λ n, a n - 2 ^ n) :=
sorry

theorem part2 {a : ℕ → ℕ} (h : sequence_def a) :
  (∃ n, S n > n ^ 2 - n + 31 ∧ (∀ m, m < n → S m ≤ m ^ 2 - m + 31)) →
  ∃ n, S n = ∑ i in range n, a i ∧ n = 5 :=
sorry

end part1_part2_l505_505559


namespace min_value_of_polynomial_l505_505831

theorem min_value_of_polynomial : ∃ x : ℝ, (x^2 + x + 1) = 3 / 4 :=
by {
  -- Solution steps are omitted
  sorry
}

end min_value_of_polynomial_l505_505831


namespace holes_remaining_unfilled_l505_505059

def total_holes : ℕ := 8
def filled_percentage : ℝ := 0.75

theorem holes_remaining_unfilled : total_holes - (filled_percentage * total_holes).to_nat = 2 :=
by
  sorry

end holes_remaining_unfilled_l505_505059


namespace hyperbola_sum_l505_505704

noncomputable def h : ℝ := 3
noncomputable def k : ℝ := -4
noncomputable def a : ℝ := 4
noncomputable def c : ℝ := Real.sqrt 53
noncomputable def b : ℝ := Real.sqrt (c^2 - a^2)

theorem hyperbola_sum : h + k + a + b = 3 + Real.sqrt 37 :=
by
  -- sorry is used to skip the proof as per the instruction
  sorry
  -- exact calc
  --   h + k + a + b = 3 + (-4) + 4 + Real.sqrt 37 : by simp
  --             ... = 3 + Real.sqrt 37 : by simp

end hyperbola_sum_l505_505704


namespace sin_angle_585_l505_505495

theorem sin_angle_585 :
  let a := 585
  let b := 360
  let c := 225
  let d := 45
  let e := 180
  ∀ (sin : ℝ → ℝ),
  (sin a = sin (a - b)) →
  (sin c = sin (d + e)) →
  (sin (d + e) = - sin d) →
  (sin d = (Real.sqrt 2) / 2) →
  sin a = - (Real.sqrt 2) / 2 :=
by
  intros sin h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end sin_angle_585_l505_505495


namespace product_of_first_four_consecutive_primes_l505_505931

theorem product_of_first_four_consecutive_primes : 
  (2 * 3 * 5 * 7) = 210 :=
by
  sorry

end product_of_first_four_consecutive_primes_l505_505931


namespace angle_sum_outside_triangle_l505_505457

-- Define the inscribed angle property and the sum of all angles properties
noncomputable def angle_inscribed_circle (a b c : ℝ) : ℝ := (180 - a) + (180 - b) + (180 - c)

theorem angle_sum_outside_triangle :
  ∀ (a b c : ℝ), (0 < a ∧ a < 180) ∧ (0 < b ∧ b < 180) ∧ (0 < c ∧ c < 180) ∧ (a + b + c = 180) →
  angle_inscribed_circle a b c = 360 :=
by
  intros a b c h,
  sorry

end angle_sum_outside_triangle_l505_505457


namespace perfect_square_for_n_l505_505771

theorem perfect_square_for_n 
  (a b : ℕ)
  (h1 : ∃ x : ℕ, ab = x^2)
  (h2 : ∃ y : ℕ, (a + 1) * (b + 1) = y^2) 
  : ∃ n : ℕ, n > 1 ∧ ∃ z : ℕ, (a + n) * (b + n) = z^2 :=
by
  let n := ab
  have h3 : n > 1 := sorry
  have h4 : ∃ z : ℕ, (a + n) * (b + n) = z^2 := sorry
  exact ⟨n, h3, h4⟩

end perfect_square_for_n_l505_505771


namespace towels_folded_in_one_hour_l505_505016

theorem towels_folded_in_one_hour 
    (jane_rate : ℕ)
    (kyla_rate : ℕ)
    (anthony_rate : ℕ) 
    (h1 : jane_rate = 36)
    (h2 : kyla_rate = 30)
    (h3 : anthony_rate = 21) 
    : jane_rate + kyla_rate + anthony_rate = 87 := 
by
  simp [h1, h2, h3]
  sorry

end towels_folded_in_one_hour_l505_505016


namespace least_positive_integer_with_12_factors_l505_505277

def num_factors (n : ℕ) : ℕ := 
  (finset.range (n + 1)).filter (λ d, n % d = 0).card

theorem least_positive_integer_with_12_factors : 
  ∃ n : ℕ, num_factors n = 12 ∧ n = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_l505_505277


namespace least_positive_integer_with_12_factors_l505_505196

/--
Prove that the smallest positive integer with exactly 12 positive factors is 60.
-/
theorem least_positive_integer_with_12_factors : 
  ∃ (n : ℕ), (0 < n) ∧ (nat.factors_count n = 12) ∧ (∀ m : ℕ, (0 < m) ∧ (nat.factors_count m = 12) → n ≤ m) → n = 60 :=
sorry

end least_positive_integer_with_12_factors_l505_505196


namespace least_pos_int_with_12_pos_factors_is_72_l505_505239

def least_positive_integer_with_12_factors (n : ℕ) : Prop :=
  ∃ (m : ℕ), m > 0 ∧ m ≠ n ∧ (∀ d : ℕ, d ∣ m → (d = n ∨ d = 1)) ∧
  ((∀ d : ℕ, d ∣ m → ∃ e : ℕ, e = 1) → n = 72)

theorem least_pos_int_with_12_pos_factors_is_72 (n : ℕ) :
  least_positive_integer_with_12_factors n → n = 72 := by
  sorry

end least_pos_int_with_12_pos_factors_is_72_l505_505239


namespace incenter_of_triangle_xyz_l505_505028

-- Define the geometric setting of the problem
variables {A B C I D E F X Y Z : Type*}

-- Conditions in the problem
-- Triangle and inequality
variables (h_triangle : triangle A B C)
variables (h_ab_lt_ac : AB < AC)

-- Incircle touches sides
variables (h_incircle_bc : incircle_touch D B C)
variables (h_incircle_ca : incircle_touch E C A)
variables (h_incircle_ab : incircle_touch F A B)

-- Angle bisector AI and intersections
variables (h_ai_bisector : angle_bisector A I)
variables (h_ai_intersects_de : intersects AI DE X)
variables (h_ai_intersects_df : intersects AI DF Y)

-- Altitude foot Z
variables (h_altitude_foot : altitude_foot A B C Z)

-- The final proof goal
theorem incenter_of_triangle_xyz : incenter D X Y Z :=
sorry

end incenter_of_triangle_xyz_l505_505028


namespace percentage_decrease_of_original_number_is_30_l505_505095

theorem percentage_decrease_of_original_number_is_30 :
  ∀ (original_number : ℕ) (difference : ℕ) (percent_increase : ℚ) (percent_decrease : ℚ),
  original_number = 40 →
  percent_increase = 0.25 →
  difference = 22 →
  original_number + percent_increase * original_number - (original_number - percent_decrease * original_number) = difference →
  percent_decrease = 0.30 :=
by
  intros original_number difference percent_increase percent_decrease h_original h_increase h_diff h_eq
  sorry

end percentage_decrease_of_original_number_is_30_l505_505095


namespace cartesian_eq_line_l_std_eq_curve_C_max_distance_P_to_line_l_l505_505710

noncomputable def curve_C (α : ℝ) : ℝ × ℝ :=
  ⟨3 * Real.cos α, Real.sqrt 3 * Real.sin α⟩

def line_l (ρ θ : ℝ) : Prop :=
  ρ * Real.cos (θ + Real.pi / 4) = Real.sqrt 6 / 2

theorem cartesian_eq_line_l (x y : ℝ) : x - y - Real.sqrt 3 = 0 ↔ ∃ (ρ θ : ℝ), ρ * Real.cos (θ + Real.pi / 4) = Real.sqrt 6 / 2 ∧ ρ * Real.cos θ = x ∧ ρ * Real.sin θ = y := 
begin
  sorry
end

theorem std_eq_curve_C (x y : ℝ) : (∃ α : ℝ, x = 3 * Real.cos α ∧ y = Real.sqrt 3 * Real.sin α) ↔ (x^2 / 9 + y^2 / 3 = 1) :=
begin
  sorry
end

theorem max_distance_P_to_line_l : ∃ α : ℝ, ∀ α : ℝ, 
  let P := curve_C α in
  let d := abs ((P.1 - Real.sqrt 3 * P.2 - Real.sqrt 3) / Real.sqrt 2) in
  d ≤ Real.sqrt 6 / 2 ∧ (d = Real.sqrt 6 / 2 ↔ α = 2 * Real.pi / 3 - π) :=
begin
  sorry
end

end cartesian_eq_line_l_std_eq_curve_C_max_distance_P_to_line_l_l505_505710


namespace pages_left_to_write_l505_505949

theorem pages_left_to_write : 
  let first_day := 25
  let second_day := 2 * first_day
  let third_day := 2 * second_day
  let fourth_day := 10
  let total_written := first_day + second_day + third_day + fourth_day
  let total_pages := 500
  let remaining_pages := total_pages - total_written
  remaining_pages = 315 :=
by
  let first_day := 25
  let second_day := 2 * first_day
  let third_day := 2 * second_day
  let fourth_day := 10
  let total_written := first_day + second_day + third_day + fourth_day
  let total_pages := 500
  let remaining_pages := total_pages - total_written
  show remaining_pages = 315
  sorry

end pages_left_to_write_l505_505949


namespace time_to_cross_pole_correct_l505_505455

-- Definitions of the conditions
def trainSpeed_kmh : ℝ := 120 -- km/hr
def trainLength_m : ℝ := 300 -- meters

-- Assumed conversions
def kmToMeters : ℝ := 1000 -- meters in a km
def hoursToSeconds : ℝ := 3600 -- seconds in an hour

-- Conversion of speed from km/hr to m/s
noncomputable def trainSpeed_ms := (trainSpeed_kmh * kmToMeters) / hoursToSeconds

-- Time to cross the pole
noncomputable def timeToCrossPole := trainLength_m / trainSpeed_ms

-- The theorem stating the proof problem
theorem time_to_cross_pole_correct : timeToCrossPole = 9 := by
  sorry

end time_to_cross_pole_correct_l505_505455


namespace least_positive_integer_with_12_factors_is_96_l505_505266

theorem least_positive_integer_with_12_factors_is_96 :
  ∃ n : ℕ, (n = 96 ∧ (∀ m : ℕ, (m > 0 ∧ ∀ d : ℕ, d ∣ m → d > 0 → d < m → (∃ k : ℕ, d = k ∨ k * d = m) → k > 1 → m > n))) ∧ (∃ m : ℕ, m = 12 ∧ (∀ d : ℕ, d ∣ m → d ∈ {1, m} ∨ ∀ k : ℕ, k * d = m → d = k ∨ d < k)).
sorry

end least_positive_integer_with_12_factors_is_96_l505_505266


namespace least_positive_integer_with_12_factors_is_96_l505_505402

def has_exactly_12_factors (n : ℕ) : Prop :=
  (12 : ℕ) = (List.range (n + 1)).filter (λ x, n % x = 0).length

theorem least_positive_integer_with_12_factors_is_96 :
  n ∈ (ℕ) ∧ has_exactly_12_factors n → n = 96 :=
by
  sorry

end least_positive_integer_with_12_factors_is_96_l505_505402


namespace main_theorem_l505_505735

variable (n : ℕ)
variable (f : ℕ → ℕ → ℝ)

-- Define the conditions
def condition1 : Prop := ∀ i, 0 ≤ i ∧ i ≤ n → f i i = 0
def condition2 : Prop := ∀ i j k l, 
  0 ≤ i ∧ i ≤ j ∧ j ≤ k ∧ k ≤ l ∧ l ≤ n → 0 ≤ f i l ∧ f i l ≤ 2 * max (f i j) (max (f j k) (f k l))

-- Define the main proof statement
theorem main_theorem 
  (hcond1 : condition1 n f) 
  (hcond2 : condition2 n f) 
  : f 0 n ≤ 2 * (Finset.sum (Finset.range n) (λ k, f k (k + 1))) :=
sorry

end main_theorem_l505_505735


namespace sum_u_t_values_l505_505586

open BigOperators

def t : Fin 4 → Fin 4
| ⟨0, _⟩ => ⟨1, by decide⟩
| ⟨1, _⟩ => ⟨3, by decide⟩
| ⟨2, _⟩ => ⟨5, by decide⟩
| ⟨3, _⟩ => ⟨7, by decide⟩

def u : Fin 5 → Fin 5
| ⟨x, h⟩ => ⟨x - 1, by linarith [h]⟩

theorem sum_u_t_values : (∑ x in (Finset.filter (λ x => x.val ∈ {3, 5}) (Finset.image t Finset.univ)), u x).val = 6 := 
sorry

end sum_u_t_values_l505_505586


namespace problem1_problem2_l505_505480

noncomputable def expression1 : ℝ := (9/4)^(1/2) + (-2017)^0 + (27/8)^(-2/3)
noncomputable def expression2 : ℝ := real.sqrt ((real.log (1/3))^2 - 4 * real.log 3 + 4) + real.log 6 - real.log 0.02

theorem problem1 : expression1 = 53 / 18 := 
by 
  sorry

theorem problem2 : expression2 = 4 :=
by 
  sorry

end problem1_problem2_l505_505480


namespace FQ_span_of_tangent_circle_right_triangle_l505_505802

noncomputable def length_FQ : ℝ :=
  let DE := 7
  let DF := real.sqrt 85
  let EF := real.sqrt (DF^2 - DE^2) in
  EF

theorem FQ_span_of_tangent_circle_right_triangle (DE DF EF FQ : ℝ) (h_DE : DE = 7) (h_DF : DF = real.sqrt 85)
(h_EF : EF = real.sqrt (DF^2 - DE^2)) (h_FQ : FQ = EF) : FQ = 6 :=
by
  sorry

end FQ_span_of_tangent_circle_right_triangle_l505_505802


namespace problem1_problem2_l505_505937

-- Problem 1
theorem problem1 : (-2)^2 * (1 / 4) + 4 / (4 / 9) + (-1)^2023 = 7 :=
by
  sorry

-- Problem 2
theorem problem2 : -1^4 + abs (2 - (-3)^2) + (1 / 2) / (-3 / 2) = 5 + 2 / 3 :=
by
  sorry

end problem1_problem2_l505_505937


namespace perpendicular_vectors_k_value_l505_505535

open Real

theorem perpendicular_vectors_k_value :
  let a := (1 : ℝ, 2 : ℝ)
  let b := (-2 : ℝ, 3 : ℝ)
  (∀ k : ℝ, 
    let v1 := (k * a.1 + b.1, k * a.2 + b.2)
    let v2 := (a.1 - k * b.1, a.2 - k * b.2)
    (v1.1 * v2.1 + v1.2 * v2.2 = 0) → 
    k = -1 + sqrt 2 ∨ k = -1 - sqrt 2) :=
by
  intro a b h k v1 v2 hp 
  sorry

end perpendicular_vectors_k_value_l505_505535


namespace least_positive_integer_with_12_factors_is_972_l505_505369

theorem least_positive_integer_with_12_factors_is_972 : ∃ k : ℕ, (∀ n : ℕ, (∃ (d : ℕ), d * k = n) ↔ n = 12) ∧ k = 972 := sorry

end least_positive_integer_with_12_factors_is_972_l505_505369


namespace max_xyz_l505_505748

theorem max_xyz (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) 
(h4 : (x * y) + 3 * z = (x + 3 * z) * (y + 3 * z)) 
: ∀ x y z, ∃ (a : ℝ), a = (x * y * z) ∧ a ≤ (1/81) :=
sorry

end max_xyz_l505_505748


namespace spider_returns_prob_l505_505451

/-- Definition of the recurrence relation for the spider's movement probability. -/
def P : ℕ → ℚ
| 0     := 1
| (n+1) := (1/3) * (1 - P n)

/-- The main statement of the proof problem:
    Proving the probability that the spider returns to its starting corner after 8 moves is 547/2187,
    and that 547 + 2187 = 2734. -/
theorem spider_returns_prob :
  P 8 = 547 / 2187 ∧ 547 + 2187 = 2734 :=
  sorry

end spider_returns_prob_l505_505451


namespace solve_for_x_l505_505785

theorem solve_for_x (x : ℂ) (h : 5 - 2 * complex.I * x = 7 - 5 * complex.I * x) : 
  x = (2 * complex.I) / 3 := by
  sorry

end solve_for_x_l505_505785


namespace least_positive_integer_with_12_factors_l505_505259

theorem least_positive_integer_with_12_factors :
  ∃ k : ℕ, (1 ≤ k) ∧ (nat.factors k).length = 12 ∧
    ∀ n : ℕ, (1 ≤ n) ∧ (nat.factors n).length = 12 → k ≤ n := 
begin
  sorry
end

end least_positive_integer_with_12_factors_l505_505259


namespace least_positive_integer_with_12_factors_l505_505152

theorem least_positive_integer_with_12_factors : ∃ n : ℕ, (nat.factors_count n = 12 ∧ ∀ m : ℕ, (nat.factors_count m = 12) → n ≤ m) ∧ n = 72 :=
begin
  sorry
end

end least_positive_integer_with_12_factors_l505_505152


namespace monotonic_decreasing_interval_l505_505693

theorem monotonic_decreasing_interval (f : ℝ → ℝ) (h : ∀ x ∈ Ioo (2 * m) (m + 1), deriv f x ≤ 0) :
  -1 ≤ m ∧ m < 1 :=
by
  -- defining the function f
  let f := λ x : ℝ, x^3 - 12 * x
  -- taking the derivative of the function
  have h_deriv : ∀ x : ℝ, deriv f x = 3 * x^2 - 12 := by sorry
  -- applying the condition of the function being monotonically decreasing to the derivative
  have h_ineq : ∀ x ∈ Ioo (2 * m) (m + 1), 3 * x^2 - 12 ≤ 0 := by
    intros x hx
    exact h x hx
  -- solving the inequalities
  have h_sol : -1 ≤ m ∧ m < 1 := by sorry
  exact h_sol

end monotonic_decreasing_interval_l505_505693


namespace product_of_real_values_eq_4_l505_505517

theorem product_of_real_values_eq_4 : ∀ s : ℝ, 
  (∃ x : ℝ, x ≠ 0 ∧ (1/(3*x) = (s - x)/9) → 
  (∀ x : ℝ, x ≠ 0 → (1/(3*x) = (s - x)/9 → x = s - 3))) → s = 4 :=
by
  sorry

end product_of_real_values_eq_4_l505_505517


namespace least_positive_integer_with_12_factors_l505_505326

def has_exactly_12_factors (n : ℕ) : Prop :=
  n.factors.length = 12

theorem least_positive_integer_with_12_factors : ∃ k : ℕ, has_exactly_12_factors k ∧ (∀ n : ℕ, has_exactly_12_factors n → n ≥ k) ∧ k = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_l505_505326


namespace largest_num_consecutive_integers_sum_45_l505_505513

theorem largest_num_consecutive_integers_sum_45 : 
  ∃ n : ℕ, (0 < n) ∧ (n * (n + 1) / 2 = 45) ∧ (∀ m : ℕ, (0 < m) → m * (m + 1) / 2 = 45 → m ≤ n) :=
by {
  sorry
}

end largest_num_consecutive_integers_sum_45_l505_505513


namespace least_positive_integer_with_12_factors_l505_505373

theorem least_positive_integer_with_12_factors : 
  ∃ (n : ℕ), (n > 0) ∧ (nat.factors_count n = 12) ∧ (∀ m, (m > 0) → (nat.factors_count m = 12) → n ≤ m) ∧ n = 108 :=
sorry

end least_positive_integer_with_12_factors_l505_505373


namespace shaded_area_between_circles_l505_505136

theorem shaded_area_between_circles :
  let r1 := 4 in
  let r2 := 5 in
  let R := r1 + r2 in
  let A_large := Real.pi * R^2 in
  let A_inner := Real.pi * r1^2 in
  let A_outer := Real.pi * r2^2 in
  let A_shaded := A_large - (A_inner + A_outer) in
  A_shaded = 40 * Real.pi :=
by
  sorry

end shaded_area_between_circles_l505_505136


namespace find_numerator_of_A_l505_505797

noncomputable def sum_arith_series (a d l : ℕ) : ℕ :=
  let n := (l - a) / d + 1
  n * (a + l) / 2

def A : ℚ :=
  sum_arith_series 2 2 2014 / sum_arith_series 1 2 2013 - sum_arith_series 1 2 2013 / sum_arith_series 2 2 2014

theorem find_numerator_of_A : ∃ m n : ℕ, Nat.gcd m n = 1 ∧ A = m / n ∧ m = 2015 := by
  sorry

end find_numerator_of_A_l505_505797


namespace least_pos_int_with_12_pos_factors_is_72_l505_505243

def least_positive_integer_with_12_factors (n : ℕ) : Prop :=
  ∃ (m : ℕ), m > 0 ∧ m ≠ n ∧ (∀ d : ℕ, d ∣ m → (d = n ∨ d = 1)) ∧
  ((∀ d : ℕ, d ∣ m → ∃ e : ℕ, e = 1) → n = 72)

theorem least_pos_int_with_12_pos_factors_is_72 (n : ℕ) :
  least_positive_integer_with_12_factors n → n = 72 := by
  sorry

end least_pos_int_with_12_pos_factors_is_72_l505_505243


namespace least_positive_integer_with_12_factors_is_972_l505_505364

theorem least_positive_integer_with_12_factors_is_972 : ∃ k : ℕ, (∀ n : ℕ, (∃ (d : ℕ), d * k = n) ↔ n = 12) ∧ k = 972 := sorry

end least_positive_integer_with_12_factors_is_972_l505_505364


namespace problem1_problem2_l505_505888

-- Problem (1): Maximum value of (a + 1/a)(b + 1/b)
theorem problem1 {a b : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) :
  (a + 1/a) * (b + 1/b) ≤ 25 / 4 := 
sorry

-- Problem (2): Minimum value of u = (a + 1/a)^3 + (b + 1/b)^3 + (c + 1/c)^3
theorem problem2 {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) :
  (a + 1/a)^3 + (b + 1/b)^3 + (c + 1/c)^3 ≥ 1000 / 9 :=
sorry

end problem1_problem2_l505_505888


namespace area_of_rectangle_at_stage_8_l505_505687

-- Define the conditions given in the problem
def square_side_length : ℝ := 4
def square_area : ℝ := square_side_length * square_side_length
def stages : ℕ := 8

-- Define the statement to be proved
theorem area_of_rectangle_at_stage_8 : (stages * square_area) = 128 := 
by 
  have h1 : square_area = 16 := by
    unfold square_area
    norm_num
  have h2 : (stages * square_area) = 8 * 16 := by
    unfold stages
    rw h1
  rw h2
  norm_num
  sorry

end area_of_rectangle_at_stage_8_l505_505687


namespace team_selection_with_captain_l505_505115

theorem team_selection_with_captain 
  (girls : Finset ℕ)
  (boys : Finset ℕ)
  (choose : ∀ {n k : ℕ}, n.choose k)
  (h_girls_length : girls.card = 4)
  (h_boys_length : boys.card = 6) :
  (choose 4 3 * choose 6 3) * 6 = 480 := 
sorry

end team_selection_with_captain_l505_505115


namespace probability_of_drawing_red_ball_from_bag_B_l505_505927

-- Define events A1, A2, A3 in terms of probabilities
def P_A1 : ℚ := 2/5
def P_A2 : ℚ := 2/5
def P_A3 : ℚ := 1/5

-- Define conditional probabilities for drawing a red ball from bag B
def P_B_given_A1 : ℚ := 4/6
def P_B_given_A2 : ℚ := 3/6
def P_B_given_A3 : ℚ := 3/6

-- Define the total probability of drawing a red ball from bag B after transferring any ball from bag A.
def P_B : ℚ := P_A1 * P_B_given_A1 + P_A2 * P_B_given_A2 + P_A3 * P_B_given_A3

-- Proof that P(B) = 17/30
theorem probability_of_drawing_red_ball_from_bag_B :
  P_B = 17/30 :=
by
  -- Calculation of probability
  sorry

end probability_of_drawing_red_ball_from_bag_B_l505_505927


namespace number_of_obtuse_triangles_l505_505120

-- Definition of the conditions of the problem
def vertices : ℕ := 120

def is_obtuse_triangle (k l m : ℕ) : Prop :=
  0 < m - k ∧ m - k < 60

def choose_3_vertices (n : ℕ) : ℕ :=
  n * (n - 1) * (n - 2) / 6  -- Number of ways to choose 3 vertices out of n

theorem number_of_obtuse_triangles : 
  ∃ (n : ℕ), n = vertices → 
  ∑ (k l m : ℕ) in ({k, l, m} : finset ℕ), is_obtuse_triangle k l m = 205320 :=
by
  sorry

end number_of_obtuse_triangles_l505_505120


namespace least_positive_integer_with_12_factors_is_72_l505_505227

open Nat

def hasExactly12Factors (n : ℕ) : Prop :=
  (divisors n).length = 12

theorem least_positive_integer_with_12_factors_is_72 :
  ∃ n : ℕ, hasExactly12Factors n ∧ ∀ m : ℕ, hasExactly12Factors m → n ≤ m → n = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_is_72_l505_505227


namespace least_positive_integer_with_12_factors_l505_505256

theorem least_positive_integer_with_12_factors :
  ∃ k : ℕ, (1 ≤ k) ∧ (nat.factors k).length = 12 ∧
    ∀ n : ℕ, (1 ≤ n) ∧ (nat.factors n).length = 12 → k ≤ n := 
begin
  sorry
end

end least_positive_integer_with_12_factors_l505_505256


namespace least_positive_integer_with_12_factors_is_72_l505_505212

open Nat

def hasExactly12Factors (n : ℕ) : Prop :=
  (divisors n).length = 12

theorem least_positive_integer_with_12_factors_is_72 :
  ∃ n : ℕ, hasExactly12Factors n ∧ ∀ m : ℕ, hasExactly12Factors m → n ≤ m → n = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_is_72_l505_505212


namespace least_positive_integer_with_12_factors_l505_505306

def has_exactly_12_factors (n : ℕ) : Prop :=
  (finset.range(n + 1).filter (λ d, n % d = 0)).card = 12

theorem least_positive_integer_with_12_factors : ∀ n : ℕ, has_exactly_12_factors n → n ≥ 150 :=
sorry

end least_positive_integer_with_12_factors_l505_505306


namespace find_total_distance_l505_505456

variables (s D : ℝ)

-- Conditions from the problem
def condition_1 : Prop :=
  let distance_before_accident := 2 * s in
  let remaining_distance := D - distance_before_accident in
  let time_at_reduced_speed := 3 * remaining_distance / (2 * s) in
  2 + 1 + time_at_reduced_speed = t1

def condition_2 : Prop :=
  let distance_before_accident := 2 * s + 60 in
  let remaining_distance := D - distance_before_accident in
  let time_at_reduced_speed := 3 * remaining_distance / (2 * s) in
  (distance_before_accident / s) + 1 + time_at_reduced_speed = t2

-- Theorem to be proved: The total distance D is 720 miles
theorem find_total_distance 
  (h1 : condition_1 s D)
  (h2 : condition_2 s D)
  (t1 : ℝ := 7.5)  -- 2 + 1 + 3 / 2 = 6.5 -> 6.5 + 1 = 7.5 hours total travel time
  (t2 : ℝ := 7)  -- same calculation for t2 = 7 hours total travel time
  (hs : s = 60) -- Assuming the given speed
  : D = 720 :=
sorry

end find_total_distance_l505_505456


namespace smallest_n_multiple_of_24_l505_505037

def f (n : ℕ) : ℕ := if h : n > 0 then 
  Inf { k | nat.factorial k % n = 0 }
else 0

theorem smallest_n_multiple_of_24 (n : ℕ) (h₁ : n > 0) (h₂ : 24 ∣ n) (h₃ : f n > 15) : n = 408 := 
sorry

end smallest_n_multiple_of_24_l505_505037


namespace least_integer_with_twelve_factors_l505_505193

open Classical

theorem least_integer_with_twelve_factors : ∃ k : ℕ, (∀ n : ℕ, (count_factors n = 12 → k ≤ n)) ∧ k = 72 :=
by
  sorry

-- Here count_factors is a function that returns the number of positive divisors of a given integer.

end least_integer_with_twelve_factors_l505_505193


namespace fifty_third_number_is_sixty_one_l505_505726

-- Define the sequence
def sequence (n : ℕ) : ℕ :=
  2 + n + n / 4  -- Skipping every 5th number adjusts the position

-- Problem statement: Prove that the 53rd number in the sequence is 61
theorem fifty_third_number_is_sixty_one : sequence 52 = 61 :=
by sorry

end fifty_third_number_is_sixty_one_l505_505726


namespace right_triangle_tangent_circle_l505_505799

theorem right_triangle_tangent_circle 
  (D E F Q : Type) [metric_space D] [has_dist E] [has_dist F] [has_dist Q]
  (h1: right_triangle D E F)
  (h2: dist D F = real.sqrt 85)
  (h3: dist D E = 7)
  (h4: ∃ C : E, circle_tangent_to D E F C)
  (h5: point_where_circle_and_side_meet Q D F) :
  dist F Q = 6 := sorry

end right_triangle_tangent_circle_l505_505799


namespace solution_for_b_l505_505698

theorem solution_for_b (x y b : ℚ) (h1 : 4 * x + 3 * y = b) (h2 : 3 * x + 4 * y = 3 * b) (hx : x = 3) : b = -21 / 5 := by
  sorry

end solution_for_b_l505_505698


namespace least_positive_integer_with_12_factors_l505_505332

def has_exactly_12_factors (n : ℕ) : Prop :=
  n.factors.length = 12

theorem least_positive_integer_with_12_factors : ∃ k : ℕ, has_exactly_12_factors k ∧ (∀ n : ℕ, has_exactly_12_factors n → n ≥ k) ∧ k = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_l505_505332


namespace least_positive_integer_with_12_factors_is_72_l505_505225

open Nat

def hasExactly12Factors (n : ℕ) : Prop :=
  (divisors n).length = 12

theorem least_positive_integer_with_12_factors_is_72 :
  ∃ n : ℕ, hasExactly12Factors n ∧ ∀ m : ℕ, hasExactly12Factors m → n ≤ m → n = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_is_72_l505_505225


namespace value_of_b_l505_505523

theorem value_of_b (b : ℝ) (h : 4 * ((3.6 * b * 2.50) / (0.12 * 0.09 * 0.5)) = 3200.0000000000005) : b = 0.48 :=
by {
  sorry
}

end value_of_b_l505_505523


namespace solve_equation_l505_505790

theorem solve_equation (x y z : ℕ) (h1 : 2^x + 5^y + 63 = z!) (h2 : z ≥ 5) : 
  (x = 5 ∧ y = 2 ∧ z = 5) ∨ (x = 4 ∧ y = 4 ∧ z = 6) :=
sorry

end solve_equation_l505_505790


namespace least_pos_int_with_12_pos_factors_is_72_l505_505232

def least_positive_integer_with_12_factors (n : ℕ) : Prop :=
  ∃ (m : ℕ), m > 0 ∧ m ≠ n ∧ (∀ d : ℕ, d ∣ m → (d = n ∨ d = 1)) ∧
  ((∀ d : ℕ, d ∣ m → ∃ e : ℕ, e = 1) → n = 72)

theorem least_pos_int_with_12_pos_factors_is_72 (n : ℕ) :
  least_positive_integer_with_12_factors n → n = 72 := by
  sorry

end least_pos_int_with_12_pos_factors_is_72_l505_505232


namespace least_positive_integer_with_12_factors_is_96_l505_505401

def has_exactly_12_factors (n : ℕ) : Prop :=
  (12 : ℕ) = (List.range (n + 1)).filter (λ x, n % x = 0).length

theorem least_positive_integer_with_12_factors_is_96 :
  n ∈ (ℕ) ∧ has_exactly_12_factors n → n = 96 :=
by
  sorry

end least_positive_integer_with_12_factors_is_96_l505_505401


namespace least_positive_integer_with_12_factors_is_972_l505_505358

theorem least_positive_integer_with_12_factors_is_972 : ∃ k : ℕ, (∀ n : ℕ, (∃ (d : ℕ), d * k = n) ↔ n = 12) ∧ k = 972 := sorry

end least_positive_integer_with_12_factors_is_972_l505_505358


namespace percentage_of_managers_l505_505845

theorem percentage_of_managers (P : ℝ) :
  (200 : ℝ) * (P / 100) - 99.99999999999991 = 0.98 * (200 - 99.99999999999991) →
  P = 99 := 
sorry

end percentage_of_managers_l505_505845


namespace dinner_potatoes_l505_505893

def lunch_potatoes : ℕ := 5
def total_potatoes : ℕ := 7

theorem dinner_potatoes : total_potatoes - lunch_potatoes = 2 :=
by
  sorry

end dinner_potatoes_l505_505893


namespace find_width_of_room_eq_l505_505827

noncomputable def total_cost : ℝ := 20625
noncomputable def rate_per_sqm : ℝ := 1000
noncomputable def length_of_room : ℝ := 5.5
noncomputable def area_paved : ℝ := total_cost / rate_per_sqm
noncomputable def width_of_room : ℝ := area_paved / length_of_room

theorem find_width_of_room_eq :
  width_of_room = 3.75 :=
sorry

end find_width_of_room_eq_l505_505827


namespace distinct_elements_in_subset_sum_104_l505_505753

theorem distinct_elements_in_subset_sum_104 :
  ∀ (S : Set ℕ), 
  (∀ n, n ∈ S ↔ ∃ k, k ∈ Finset.range 34 ∧ n = 3 * k + 1) →
  ∀ T : Set ℕ, T ⊆ S → Finset.card T = 20 →
  ∃ x y ∈ T, x ≠ y ∧ x + y = 104 :=
by
  intros S hS T hT cardT
  sorry

end distinct_elements_in_subset_sum_104_l505_505753


namespace parallelepiped_volume_l505_505524

noncomputable def volume_of_parallelepiped (a : ℝ) : ℝ :=
  (a^3 * Real.sqrt 2) / 2

theorem parallelepiped_volume (a : ℝ) (h_pos : 0 < a) :
  volume_of_parallelepiped a = (a^3 * Real.sqrt 2) / 2 :=
by
  sorry

end parallelepiped_volume_l505_505524


namespace least_positive_integer_with_12_factors_l505_505302

def has_exactly_12_factors (n : ℕ) : Prop :=
  (finset.range(n + 1).filter (λ d, n % d = 0)).card = 12

theorem least_positive_integer_with_12_factors : ∀ n : ℕ, has_exactly_12_factors n → n ≥ 150 :=
sorry

end least_positive_integer_with_12_factors_l505_505302


namespace least_positive_integer_with_12_factors_l505_505280

def num_factors (n : ℕ) : ℕ := 
  (finset.range (n + 1)).filter (λ d, n % d = 0).card

theorem least_positive_integer_with_12_factors : 
  ∃ n : ℕ, num_factors n = 12 ∧ n = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_l505_505280


namespace increasing_interval_of_cubic_plus_linear_l505_505825

theorem increasing_interval_of_cubic_plus_linear :
  ∀ x : ℝ, ∃ I : set ℝ, (I = set.univ) ∧ (is_increasing_on (λ x, x^3 + x) I) :=
by
  intros x
  use set.univ
  split
  { refl }
  {
    sorry    -- Proof goes here
  }

end increasing_interval_of_cubic_plus_linear_l505_505825


namespace sum_of_fifth_powers_divisible_by_n_l505_505102

theorem sum_of_fifth_powers_divisible_by_n (a b c d e n : ℤ) (hn : odd n) 
  (h1 : (a + b + c + d + e) % n = 0) 
  (h2 : (a^2 + b^2 + c^2 + d^2 + e^2) % n = 0) : 
  ((a^5 + b^5 + c^5 + d^5 + e^5 - 5 * a * b * c * d * e) % n = 0) :=
  by
  sorry

end sum_of_fifth_powers_divisible_by_n_l505_505102


namespace sum_of_sequence_l505_505544

theorem sum_of_sequence :
  ∀ (a : ℕ → ℕ), 
  (a 1 + a 2 = 1) →
  (a 2 + a 3 = 2) →
  (a 3 + a 4 = 3) →
  -- (conditions continue for all up to)
  (a 99 + a 100 = 99) →
  (a 100 + a 1 = 100) →
  (∑ i in Finset.range 101, a i = 2525) :=
by
  intros a h1 h2 h3 h99 h100
  sorry

end sum_of_sequence_l505_505544


namespace find_d_l505_505830

-- Given conditions
def line_eq (x y : ℚ) : Prop := y = (3 * x - 4) / 4

def parametrized_eq (v d : ℚ × ℚ) (t x y : ℚ) : Prop :=
  (x, y) = (v.1 + t * d.1, v.2 + t * d.2)

def distance_eq (x y : ℚ) (t : ℚ) : Prop :=
  (x - 3) * (x - 3) + (y - 1) * (y - 1) = t * t

-- The proof problem statement
theorem find_d (d : ℚ × ℚ) 
  (h_d : d = (7/2, 5/2)) :
  ∀ (x y t : ℚ) (v : ℚ × ℚ) (h_v : v = (3, 1)),
    (x ≥ 3) → 
    line_eq x y → 
    parametrized_eq v d t x y → 
    distance_eq x y t → 
    d = (7/2, 5/2) := 
by 
  intros;
  sorry


end find_d_l505_505830


namespace least_positive_integer_with_12_factors_is_96_l505_505268

theorem least_positive_integer_with_12_factors_is_96 :
  ∃ n : ℕ, (n = 96 ∧ (∀ m : ℕ, (m > 0 ∧ ∀ d : ℕ, d ∣ m → d > 0 → d < m → (∃ k : ℕ, d = k ∨ k * d = m) → k > 1 → m > n))) ∧ (∃ m : ℕ, m = 12 ∧ (∀ d : ℕ, d ∣ m → d ∈ {1, m} ∨ ∀ k : ℕ, k * d = m → d = k ∨ d < k)).
sorry

end least_positive_integer_with_12_factors_is_96_l505_505268


namespace angle_ACP_eq_angle_ABQ_l505_505699

-- A point in the Euclidean space
structure Point :=
  (x y z : ℝ)

-- Definition of a triangle
structure Triangle :=
  (A B C : Point)

-- Midpoint of two points
noncomputable def midpoint (P Q : Point) : Point :=
  { x := (P.x + Q.x) / 2,
    y := (P.y + Q.y) / 2,
    z := (P.z + Q.z) / 2 }

-- Definition of circumcircle intersection and other necessary constructions
axiom circumcircle_intersects (T1 T2 : Triangle) (common_chord : Set Point) : Prop

axiom lines_intersect (l1 l2 : Set Point) (intersection_point : Point) : Prop

-- Define the problem as a theorem in lean

theorem angle_ACP_eq_angle_ABQ (ABC : Triangle) :
  let D := midpoint ABC.C ABC.A in
  let E := midpoint ABC.A ABC.B in
  let circum1 := {p: Point | circumcircle_intersects ⟨ABC.A, ABC.B, D⟩ ⟨ABC.A, D, ⟫} in
  let circum2 := {p: Point | circumcircle_intersects ⟨ABC.A, ABC.C, E⟩ ⟨ABC.A, E, ⟫} in
  ∃ T P Q, (lines_intersect {p: Point | p ∈ circum1} {p: Point | p ∈ circum2} T) ∧
           (lines_intersect {p: Point | p ∉ circum1} { p: Point | p ∉ circum2} P) ∧
           (lines_intersect {p: Point | p ∉ circum1} { p: Point | p ∉ circum2} Q) ∧
           ∠ (ABC.A) (ABC.C) (P) = ∠ (ABC.A) (ABC.B) (Q) :=
begin
  sorry
end

end angle_ACP_eq_angle_ABQ_l505_505699


namespace quadratic_two_real_roots_find_m_l505_505605

theorem quadratic_two_real_roots (m : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁ * x₂ = 3 * m^2 ∧ x₁ + x₂ = 4 * m :=
by
  sorry

theorem find_m (m : ℝ) (h : m > 0) (h_diff : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ - x₂ = 2) : m = 1 :=
by
  sorry

end quadratic_two_real_roots_find_m_l505_505605


namespace four_digit_multiples_of_5_count_l505_505638

-- Define the range of four-digit numbers
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

-- Define a multiple of 5
def is_multiple_of_five (n : ℕ) : Prop := n % 5 = 0

-- Define the required proof problem
theorem four_digit_multiples_of_5_count : 
  ∃ n : ℕ, (n = 1800) ∧ (∀ k : ℕ, is_four_digit k ∧ is_multiple_of_five k → n = 1800) :=
sorry

end four_digit_multiples_of_5_count_l505_505638


namespace sum_a_i_eq_2525_l505_505537

theorem sum_a_i_eq_2525 (a : ℕ → ℝ)
  (h1 : a 1 + a 2 = 1)
  (h2 : a 2 + a 3 = 2)
  (h3 : a 3 + a 4 = 3)
  -- include all intermediate conditions up to
  (h99 : a 99 + a 100 = 99)
  (h100 : a 100 + a 1 = 100) :
  (Finset.sum (Finset.range 100) (λ k, a (k + 1))) = 2525 :=
sorry

end sum_a_i_eq_2525_l505_505537


namespace least_positive_integer_with_12_factors_is_72_l505_505226

open Nat

def hasExactly12Factors (n : ℕ) : Prop :=
  (divisors n).length = 12

theorem least_positive_integer_with_12_factors_is_72 :
  ∃ n : ℕ, hasExactly12Factors n ∧ ∀ m : ℕ, hasExactly12Factors m → n ≤ m → n = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_is_72_l505_505226


namespace least_positive_integer_with_12_factors_is_96_l505_505395

def has_exactly_12_factors (n : ℕ) : Prop :=
  (12 : ℕ) = (List.range (n + 1)).filter (λ x, n % x = 0).length

theorem least_positive_integer_with_12_factors_is_96 :
  n ∈ (ℕ) ∧ has_exactly_12_factors n → n = 96 :=
by
  sorry

end least_positive_integer_with_12_factors_is_96_l505_505395


namespace determine_words_per_page_l505_505890

noncomputable def wordsPerPage (totalPages : ℕ) (wordsPerPage : ℕ) (totalWordsMod : ℕ) : ℕ :=
if totalPages * wordsPerPage % 250 = totalWordsMod ∧ wordsPerPage <= 200 then wordsPerPage else 0

theorem determine_words_per_page :
  wordsPerPage 150 198 137 = 198 :=
by 
  sorry

end determine_words_per_page_l505_505890


namespace ratio_HO_OT_l505_505490

/-- Given a convex quadrilateral \(MATH\) with the properties:
1. \( \frac{HM}{MT} = \frac{3}{4} \)
2. \( \angle ATM = \angle MAT = \angle AHM = 60^\circ \)
3. \( N \) is the midpoint of \( MA \)
4. \( O \) is a point on \( TH \) such that lines \( MT, AH, NO \) are concurrent,
show that the ratio \( \frac{HO}{OT} \) is \( \frac{9}{16} \).
-/
theorem ratio_HO_OT {M A T H N O : Type*}
  (HM MT : ℝ) (h_ratio : HM / MT = 3 / 4)
  (angle_ATM : ∠ A T M = 60) (angle_MAT : ∠ M A T = 60) (angle_AHM : ∠ A H M = 60)
  (h_midpoint : midpoint N A = M) 
  (h_concurrent : concurrent [MT, AH, NO]) : 
  HO / OT = 9 / 16 :=
sorry

end ratio_HO_OT_l505_505490


namespace least_pos_int_with_12_pos_factors_is_72_l505_505238

def least_positive_integer_with_12_factors (n : ℕ) : Prop :=
  ∃ (m : ℕ), m > 0 ∧ m ≠ n ∧ (∀ d : ℕ, d ∣ m → (d = n ∨ d = 1)) ∧
  ((∀ d : ℕ, d ∣ m → ∃ e : ℕ, e = 1) → n = 72)

theorem least_pos_int_with_12_pos_factors_is_72 (n : ℕ) :
  least_positive_integer_with_12_factors n → n = 72 := by
  sorry

end least_pos_int_with_12_pos_factors_is_72_l505_505238


namespace students_making_stars_l505_505127

theorem students_making_stars (total_stars stars_per_student : ℕ) (h1 : total_stars = 372) (h2 : stars_per_student = 3) : 
  total_stars / stars_per_student = 124 :=
by
  sorry

end students_making_stars_l505_505127


namespace time_after_9999_seconds_l505_505013

def secondsToTime (s : ℕ) : (ℕ × ℕ × ℕ) :=
  let (minutes, sec) := Nat.divMod s 60
  let (hours, min) := Nat.divMod minutes 60
  (hours, min, sec)

def addTime (h_start m_start s_start h_add m_add s_add : ℕ) : (ℕ × ℕ × ℕ) :=
  let total_seconds_start := h_start * 3600 + m_start * 60 + s_start
  let total_seconds_add := h_add * 3600 + m_add * 60 + s_add
  let total_seconds := total_seconds_start + total_seconds_add
  secondsToTime total_seconds

theorem time_after_9999_seconds :
  addTime 8 0 0 2 46 39 = (10, 46, 39) :=
by
  unfold addTime
  unfold secondsToTime
  sorry

end time_after_9999_seconds_l505_505013


namespace addition_proof_l505_505484

theorem addition_proof : 157 + 18 + 32 + 43 = 250 := by
  calc
    157 + 18 + 32 + 43
        = 157 + 43 + 18 + 32 : by rw [add_assoc, add_comm (18 + 32)]
    ... = 200 + 50              : by rw [add_assoc, add_assoc, add_comm (157 + 43)]
    ... = 250                   : by
      rw add_comm (157 + 43)  -- to preciseily show the second step
      sorry

end addition_proof_l505_505484


namespace area_of_rectangle_area_of_triangle_l505_505773

variables {AB AD AN NC AM MB : ℝ}
variables (h1 : AN = 9) (h2 : NC = 39) (h3 : AM = 10) (h4 : MB = 5)
def AB := AM + MB
def AD := AN + NC
def area_rectangle := AB * AD
def MN := real.sqrt (AM^2 + AN^2)
def area_triangle := 0.5 * MN * NC

theorem area_of_rectangle : area_rectangle = 720 :=
by {
  have hAB : AB = 15 := by { unfold AB, linarith },
  have hAD : AD = 48 := by { unfold AD, linarith },
  unfold area_rectangle,
  rw [hAB, hAD],
  norm_num,
}

theorem area_of_triangle : area_triangle = 247.5 :=
by {
  have hMN : MN = 13 := by {
    sorry, -- This step involves some intermediary calculation that normally would require proving MN = sqrt(10^2 + 9^2)
  },
  unfold area_triangle,
  rw [hMN],
  norm_num,
}

end area_of_rectangle_area_of_triangle_l505_505773


namespace four_digit_multiples_of_five_count_l505_505669

theorem four_digit_multiples_of_five_count : 
  {n : ℕ | 1000 ≤ n ∧ n ≤ 9999 ∧ n % 5 = 0}.toFinset.card = 1800 :=
by
  sorry

end four_digit_multiples_of_five_count_l505_505669


namespace B_k_n_closed_form_l505_505750

theorem B_k_n_closed_form {n k : ℕ} (hn : 2 ≤ n) (hk : 2 ≤ k) (hkn : k ≤ n) :
  let B_k (n k : ℕ) := Nat.choose (2 * n) k - 2 * Nat.choose n k in
  B_k n k = Nat.choose (2 * n) k - 2 * Nat.choose n k :=
by
  let B_k (n k : ℕ) := Nat.choose (2 * n) k - 2 * Nat.choose n k
  sorry

end B_k_n_closed_form_l505_505750


namespace part1_part2_l505_505571

open Set

variable {α : Type*} [LinearOrderedField α]

def f (a b c : α) : α → α := λ x, a * x^2 + b * x + c

def A_set (a b c : α) : Set α := {x | f a b c x = f 0 b c x}
def B_set (a b c : α) : Set α := {x | f a b c x = f c 0 a x}
def C_set (a b c : α) : Set α := {x | f a b c x = f a x b}

theorem part1 (a b c : α) (hab : (A_set a b c ∩ B_set a b c).Nonempty) : a = c := by
  sorry

theorem part2 (a b : α) (hac : a ≠ 0) (H : (A_set a b 1 ∪ B_set a b 1 ∪ C_set a b 1).card = 3) : 2 * a + b = -1 := by
  sorry

end part1_part2_l505_505571


namespace line_parallel_plane_no_common_points_l505_505574

-- Definitions of lines and planes
structure Line :=
(to_set : set ℝ³) -- Representing a line as a subset in a 3D space

structure Plane :=
(to_set : set ℝ³) -- Representing a plane as a subset in a 3D space

-- Parallel relationship between a line and a plane
def parallel_line_plane (l : Line) (α : Plane) : Prop :=
∀ p ∈ α.to_set, ∀ q ∈ l.to_set, ¬ (p = q)

-- Subset relationship between a line and a plane
def subset_line_plane (a : Line) (α : Plane) : Prop :=
∀ p ∈ a.to_set, p ∈ α.to_set

-- No common points relationship between two lines
def no_common_points (l a : Line) : Prop :=
∀ p ∈ l.to_set, p ∉ a.to_set

-- Main theorem statement
theorem line_parallel_plane_no_common_points
  (l a : Line) (α : Plane)
  (h1 : parallel_line_plane l α)
  (h2 : subset_line_plane a α) :
  no_common_points l a :=
sorry

end line_parallel_plane_no_common_points_l505_505574


namespace point_exists_in_multiple_polygons_l505_505553

open Convex Set

-- Define the problem conditions
variables {P : Type} [MetricSpace P] [NormedAddCommGroup P] [NormedSpace ℝ P] {n k : ℕ}
variables (polygons : Fin n → ConvexPolygon ℝ P) (k_sides : ∀ i, (polygons i).sides = k)
variable (intersect_pairwise : ∀ i j, i ≠ j → (polygons i).intersects (polygons j))
variable (homothety_transform : ∀ i j, ∃ (c : ℝ), c > 0 ∧ Homothety P (c • (polygons i).vertices) = (polygons j).vertices)

-- The proof statement
theorem point_exists_in_multiple_polygons :
  ∃ p : P, ∃ m : ℕ, (1 + (n - 1) / (2 * k) : ℝ) ≤ m ∧ m ≤ n ∧ (∃ S : Finset (Fin n), S.card = m ∧ (∀ i ∈ S, ConvexPolygon.contains (polygons i) p)) :=
sorry

end point_exists_in_multiple_polygons_l505_505553


namespace exponent_rule_division_l505_505416

theorem exponent_rule_division (a : ℝ) (h₁ : a ≠ 0) : (a^5 / a^3 = a^2) := 
by
  -- Since (a^5 / a^3) = a^(5-3) from the exponent division rule
  have h₂ : a^5 / a^3 = a^(5 - 3), by sorry,
  -- Substitute the exponent subtraction
  have h₃ : 5 - 3 = 2, by linarith,
  -- Final substitution
  rw [h₃] at h₂,
  exact h₂

end exponent_rule_division_l505_505416


namespace additional_pots_produced_l505_505410

theorem additional_pots_produced (first_hour_time_per_pot last_hour_time_per_pot : ℕ) :
  first_hour_time_per_pot = 6 →
  last_hour_time_per_pot = 5 →
  60 / last_hour_time_per_pot - 60 / first_hour_time_per_pot = 2 :=
by
  intros
  sorry

end additional_pots_produced_l505_505410


namespace least_positive_integer_with_12_factors_l505_505198

/--
Prove that the smallest positive integer with exactly 12 positive factors is 60.
-/
theorem least_positive_integer_with_12_factors : 
  ∃ (n : ℕ), (0 < n) ∧ (nat.factors_count n = 12) ∧ (∀ m : ℕ, (0 < m) ∧ (nat.factors_count m = 12) → n ≤ m) → n = 60 :=
sorry

end least_positive_integer_with_12_factors_l505_505198


namespace abs_sqrt3_minus_2_l505_505814

theorem abs_sqrt3_minus_2 : |sqrt 3 - 2| = 2 - sqrt 3 :=
by sorry

end abs_sqrt3_minus_2_l505_505814


namespace sum_of_consecutive_page_numbers_l505_505837

def consecutive_page_numbers_product_and_sum (n m : ℤ) :=
  n * m = 20412

theorem sum_of_consecutive_page_numbers (n : ℤ) (h1 : consecutive_page_numbers_product_and_sum n (n + 1)) : n + (n + 1) = 285 :=
by
  sorry

end sum_of_consecutive_page_numbers_l505_505837


namespace number_of_bad_arrangements_l505_505111

def is_bad_arrangement (arrangement : List ℕ) : Prop :=
  let n := arrangement.length
  ∀ sum ∈ Finset.range (1, 22), ∀ k : ℕ, k > 0 → k ≤ n → 
  ∀ i : ℕ, i < n →
  sum ≠ List.sum (List.take k (List.drop i (List.cycle arrangement)))

theorem number_of_bad_arrangements : 
  Finset.card { arrangement | arrangement.perm List.range 1 7 ∧ is_bad_arrangement arrangement} = 2 :=
sorry

end number_of_bad_arrangements_l505_505111


namespace least_positive_integer_with_12_factors_is_96_l505_505265

theorem least_positive_integer_with_12_factors_is_96 :
  ∃ n : ℕ, (n = 96 ∧ (∀ m : ℕ, (m > 0 ∧ ∀ d : ℕ, d ∣ m → d > 0 → d < m → (∃ k : ℕ, d = k ∨ k * d = m) → k > 1 → m > n))) ∧ (∃ m : ℕ, m = 12 ∧ (∀ d : ℕ, d ∣ m → d ∈ {1, m} ∨ ∀ k : ℕ, k * d = m → d = k ∨ d < k)).
sorry

end least_positive_integer_with_12_factors_is_96_l505_505265


namespace susan_cars_fewer_than_carol_l505_505055

theorem susan_cars_fewer_than_carol 
  (Lindsey_cars Carol_cars Susan_cars Cathy_cars : ℕ)
  (h1 : Lindsey_cars = Cathy_cars + 4)
  (h2 : Susan_cars < Carol_cars)
  (h3 : Carol_cars = 2 * Cathy_cars)
  (h4 : Cathy_cars = 5)
  (h5 : Cathy_cars + Carol_cars + Lindsey_cars + Susan_cars = 32) :
  Carol_cars - Susan_cars = 2 :=
sorry

end susan_cars_fewer_than_carol_l505_505055


namespace count_four_digit_multiples_of_5_l505_505660

theorem count_four_digit_multiples_of_5 : 
  let first_4_digit := 1000
  let last_4_digit := 9999
  let first_multiple_of_5 := 1000
  let last_multiple_of_5 := 9995
  let total_multiples_of_5 := (1999 - 200 + 1)
  first_multiple_of_5 % 5 = 0 ∧ last_multiple_of_5 % 5 = 0 ∧ first_4_digit ≤ first_multiple_of_5 ∧ last_multiple_of_5 ≤ last_4_digit
  → total_multiples_of_5 = 1800 :=
by
  sorry

end count_four_digit_multiples_of_5_l505_505660


namespace area_larger_rectangle_l505_505818

theorem area_larger_rectangle
  (area_shaded : ℝ)
  (width_shaded height_shaded : ℝ)
  (h1 : area_shaded = 2)
  (h2 : width_shaded = 1)
  (h3 : height_shaded = 2) :
  let width_larger := width_shaded * 6
  let height_larger := height_shaded
  let area_larger := width_larger * height_larger
  in area_larger = 6 :=
by
  sorry

end area_larger_rectangle_l505_505818


namespace four_digit_multiples_of_5_l505_505646

theorem four_digit_multiples_of_5 : 
  let f := fun n => 1000 <= n ∧ n <= 9999 ∧ n % 5 = 0
  in finset.card (finset.filter f (finset.range 10000)) = 1800 :=
by
  sorry

end four_digit_multiples_of_5_l505_505646


namespace sum_of_sequence_l505_505545

theorem sum_of_sequence :
  ∀ (a : ℕ → ℕ), 
  (a 1 + a 2 = 1) →
  (a 2 + a 3 = 2) →
  (a 3 + a 4 = 3) →
  -- (conditions continue for all up to)
  (a 99 + a 100 = 99) →
  (a 100 + a 1 = 100) →
  (∑ i in Finset.range 101, a i = 2525) :=
by
  intros a h1 h2 h3 h99 h100
  sorry

end sum_of_sequence_l505_505545


namespace no_such_function_exists_l505_505958

theorem no_such_function_exists :
  ¬(∃ (f : ℝ → ℝ), ∀ x y : ℝ, |f (x + y) + Real.sin x + Real.sin y| < 2) :=
sorry

end no_such_function_exists_l505_505958


namespace percent_of_flowers_are_daisies_l505_505440

-- Definitions for the problem
def total_flowers (F : ℕ) := F
def blue_flowers (F : ℕ) := (7/10) * F
def red_flowers (F : ℕ) := (3/10) * F
def blue_tulips (F : ℕ) := (1/2) * (7/10) * F
def blue_daisies (F : ℕ) := (7/10) * F - (1/2) * (7/10) * F
def red_daisies (F : ℕ) := (2/3) * (3/10) * F
def total_daisies (F : ℕ) := blue_daisies F + red_daisies F
def percentage_of_daisies (F : ℕ) := (total_daisies F / F) * 100

-- The statement to prove
theorem percent_of_flowers_are_daisies (F : ℕ) (hF : F > 0) :
  percentage_of_daisies F = 55 := by
  sorry

end percent_of_flowers_are_daisies_l505_505440


namespace perpendicular_lines_a_eq_0_or_neg1_l505_505829

theorem perpendicular_lines_a_eq_0_or_neg1 (a : ℝ) :
  (∃ (k₁ k₂: ℝ), (k₁ = a ∧ k₂ = (2 * a - 1)) ∧ ∃ (k₃ k₄: ℝ), (k₃ = 3 ∧ k₄ = a) ∧ k₁ * k₃ + k₂ * k₄ = 0) →
  (a = 0 ∨ a = -1) := 
sorry

end perpendicular_lines_a_eq_0_or_neg1_l505_505829


namespace f_monotonically_increasing_range_of_fx1_fx2_l505_505583

noncomputable def f (x m : ℝ) : ℝ := 2 * Real.log x + x^2 - m * x

theorem f_monotonically_increasing (m : ℝ) : (∀ x ε : ℝ, 0 < ε → 0 < x → (ε + (2 / x + 2 * x - m)) =x - 2 / x - 2 x + m ≥ 0 ↔ m ≤ 4 :=
by
    sorry

theorem range_of_fx1_fx2 (m : ℝ) (x₁ x₂ : ℝ) 
  (h_m_range : 5 < m ∧ m < 17 / 2) 
  (h_extreme_pts : ∀x1 x2: ℝ, x1<x2) 
  (hx1_condition : x₁ * x₂ = 1 ∧ x₁ + x₂ = m / 2 ∧ 0 < x₁ ∧ 0 < x₂ ∧ x₁ < 1 < x₂) :
  (15 / 4 - 4 * Real.log 2 < f(x₁, m) - f(x₂, m) ∧ f(x₁, m) - f(x₂, m) < 255 / 16 - 8 * Real.log 2 := by
  sorry

end f_monotonically_increasing_range_of_fx1_fx2_l505_505583


namespace least_positive_integer_with_12_factors_is_972_l505_505362

theorem least_positive_integer_with_12_factors_is_972 : ∃ k : ℕ, (∀ n : ℕ, (∃ (d : ℕ), d * k = n) ↔ n = 12) ∧ k = 972 := sorry

end least_positive_integer_with_12_factors_is_972_l505_505362


namespace ivans_board_problem_l505_505014

-- Definitions of the conditions
def board (n : ℕ) := fin n × fin n

/-- Ivan's board condition: a black square has exactly two black neighbors --/
def valid_coloring (n : ℕ) (coloring : board n → bool) : Prop :=
  ∀ (x y : fin n), coloring (x, y) = tt → 
  (((x + 1, y) < n ∧ coloring (x + 1, y) = tt) ∨ 
   ((x - 1, y) < n ∧ coloring (x - 1, y) = tt) ∨ 
   ((x, y + 1) < n ∧ coloring (x, y + 1) = tt) ∨ 
   ((x, y - 1) < n ∧ coloring (x, y - 1) = tt))

noncomputable def d_n (n : ℕ) : ℕ :=
  max {sq | ∃ coloring : board n → bool, valid_coloring n coloring ∧
                (sq = (finset.filter (λ p, coloring p) (finset.univ : finset (board n))).card)}

-- The main theorem statement
theorem ivans_board_problem (n : ℕ) : ∃ (a b c : ℝ) (a_eq b_eq c_eq : unit), 
  a = 2/3 ∧ b = 8 ∧ c = 4 ∧ (∀ n : ℕ, (a * (n : ℝ)^2 - b * (n : ℝ) ≤ (d_n n : ℝ) ∧ (d_n n : ℝ) ≤ a * (n : ℝ)^2 + c * (n : ℝ))) :=
by
  sorry

end ivans_board_problem_l505_505014


namespace packs_of_juice_bought_l505_505912

-- Define the costs of individual items
def cost_candy_bar := 25
def cost_chocolate := 75
def cost_juice_pack := 50

-- Define the quantities of items to be bought
def num_candy_bars := 3
def num_chocolates := 2
def total_quarters := 11

-- Calculate the total cost of candy bars and chocolates in quarters
def total_cost_candy_bars := num_candy_bars * cost_candy_bar
def total_cost_chocolates := num_chocolates * cost_chocolate
def total_cost_candy_chocolates := total_cost_candy_bars + total_cost_chocolates

-- Convert total cost to quarters
def quarters_for_candy_chocolates := total_cost_candy_chocolates / 25

-- Define the theorem for the proof problem
theorem packs_of_juice_bought : 
  total_quarters - quarters_for_candy_chocolates = cost_juice_pack / 25 :=
begin
  sorry
end

end packs_of_juice_bought_l505_505912


namespace ellipse_eccentricity_range_l505_505580

theorem ellipse_eccentricity_range (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (∃ x y c : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1 ∧ x^2 + y^2 = c^2) →
  (let e := Real.sqrt (1 - (b^2 / a^2)) in (Real.sqrt 2 / 2) ≤ e ∧ e < 1) :=
by
  sorry

end ellipse_eccentricity_range_l505_505580


namespace least_positive_integer_with_12_factors_l505_505283

def num_factors (n : ℕ) : ℕ := 
  (finset.range (n + 1)).filter (λ d, n % d = 0).card

theorem least_positive_integer_with_12_factors : 
  ∃ n : ℕ, num_factors n = 12 ∧ n = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_l505_505283


namespace cantaloupe_total_l505_505531

theorem cantaloupe_total (Fred Tim Alicia : ℝ) 
  (hFred : Fred = 38.5) 
  (hTim : Tim = 44.2)
  (hAlicia : Alicia = 29.7) : 
  Fred + Tim + Alicia = 112.4 :=
by
  sorry

end cantaloupe_total_l505_505531


namespace calculate_FI_squared_l505_505000

noncomputable def square_area {A B C D : Point} (s : ℝ) (sq : square A B C D s) : ℝ := s^2
noncomputable def triangle_area {A B C : Point} (A B C) : ℝ := sorry -- formula needed here
noncomputable def quad_area {A B C D : Point}(quad : quadrilateral A B C D) : ℝ := sorry -- formula needed here
noncomputable def pent_area {A B C D E Point}(pent: pentagon A B C D E) : ℝ := sorry -- formula needed here

theorem calculate_FI_squared 
  (A B C D E F G H I J : Point)
  (s : ℝ)
  (x y : ℝ)
  (sq : square A B C D s)
  (hebb : E.AB = x ∧ H.DA = x)
  (hfi1 : F.BC = x ∧ G.CD = x)
  (hfi2 : FI⊥ EH ∧ GJ⊥ EH)
  (area1 : triangle_area A E H = 1)
  (area2 : quad_area B F I E = 1)
  (area3 : quad_area D H J G = 1)
  (area4 : pent_area F C G J I = 1) : 
  (FI_squared A B C D E F G H I J s x y sq hebb hfi1 hfi2 area1 area2 area3 area4 = 8 - 4 * sqrt 2) :=
 sorry

end calculate_FI_squared_l505_505000


namespace least_positive_integer_with_12_factors_l505_505165

theorem least_positive_integer_with_12_factors :
  ∃ m : ℕ, 0 < m ∧ (∀ n : ℕ, 0 < n → (n ≠ m →  m ≤ n)) ∧ (12 = (m.factors.toFinset.card)) := sorry

end least_positive_integer_with_12_factors_l505_505165
