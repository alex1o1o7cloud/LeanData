import Mathlib
import Mathlib.Algebra.AbsoluteValue
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Field
import Mathlib.Algebra.Field.Defs
import Mathlib.Algebra.GcdMonoid.Basic
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Ring.Basic
import Mathlib.Analysis.Calculus.Fderiv.Basic
import Mathlib.Analysis.SpecialFunctions.Abs
import Mathlib.Analysis.Special_Functions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Int.Mod
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Set.Basic
import Mathlib.Data.Zmod.Basic
import Mathlib.Geometry
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Geometry.Euclidean.CircumscribedCircle
import Mathlib.MeasureTheory.Integral.IntervalIntegral
import Mathlib.NumberTheory.ArithmeticFunction
import Mathlib.NumberTheory.Basic
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Tactic.Basic

namespace price_of_second_tea_l389_389033

theorem price_of_second_tea (P : ℝ) (H1 : 62 ≤ P) (H2 : P ≤ 77) : 
  let tea1_price := 62
  let mixture_price := 67
  let ratio_1_1 := 1 / (1 + 1) : ℝ
  mixture_price = 67 → (P = 67 + 5) :=
by sorry

end price_of_second_tea_l389_389033


namespace range_of_f_l389_389348

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := x^k

theorem range_of_f (k : ℝ) (hk : k < 0) : 
  set.range (λ x, f x k) = set.Ioo (0 : ℝ) 1 :=
sorry

end range_of_f_l389_389348


namespace range_of_a_l389_389775

theorem range_of_a (a : ℝ) :
  (∃ (f : ℝ → ℝ), ∀ x ∈ set.Icc 1 3, f (x^2 - 2 * a * x + a) = x) ∧
  (|a - 1| + |a - 3| ≤ 4) →
  a ∈ set.Icc 0 1 ∪ set.Icc 3 4 :=
by
  sorry

end range_of_a_l389_389775


namespace bottles_remaining_correct_l389_389075

def bottles_left (initial : ℝ) (maria_drank : ℝ) (sister_drank : ℝ) : ℝ :=
  initial - maria_drank - sister_drank

theorem bottles_remaining_correct :
  ∀ (initial maria_drank sister_drank : ℝ), 
    initial = 45.0 → 
    maria_drank = 14.0 → 
    sister_drank = 8.0 → 
    bottles_left initial maria_drank sister_drank = 23.0 :=
by
  intros initial maria_drank sister_drank h_initial h_maria h_sister
  rw [h_initial, h_maria, h_sister]
  calc
    45.0 - 14.0 - 8.0 = 31.0 - 8.0 : by simp
    ...               = 23.0      : by simp

end bottles_remaining_correct_l389_389075


namespace rhombuses_in_grid_l389_389121

def number_of_rhombuses (n : ℕ) : ℕ :=
(n - 1) * n + (n - 1) * n

theorem rhombuses_in_grid :
  number_of_rhombuses 5 = 30 :=
by
  sorry

end rhombuses_in_grid_l389_389121


namespace road_completion_l389_389110

/- 
  The company "Roga and Kopyta" undertook a project to build a road 100 km long. 
  The construction plan is: 
  - In the first month, 1 km of the road will be built.
  - Subsequently, if by the beginning of some month A km is already completed, then during that month an additional 1 / A^10 km of road will be constructed.
  Prove that the road will be completed within 100^11 months.
-/

theorem road_completion (L : ℕ → ℝ) (h1 : L 1 = 1)
  (h2 : ∀ n ≥ 1, L (n + 1) = L n + 1 / (L n) ^ 10) :
  ∃ m ≤ 100 ^ 11, L m ≥ 100 := 
  sorry

end road_completion_l389_389110


namespace volume_of_5th_section_l389_389106

noncomputable def volume_of_section (a1 : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  a1 + (n - 1) * d

theorem volume_of_5th_section
  (a1 d : ℚ)
  (h1 : 4 * a1 + (4 * 3 / 2) * d = 3)
  (h2 : 3 * a1 + (3 * 5 / 2) * d = 4) :
  volume_of_section a1 d 5 = 67 / 66 :=
by
  -- declare the hypothesis corresponding to the system of equations solutions
  have h3 : a1 = 13 / 22 := sorry,
  have h4 : d = 7 / 66 := sorry,
  rw [h3, h4],
  -- compute the volume of the 5th section
  unfold volume_of_section,
  norm_num, -- simplify the expression
  sorry

end volume_of_5th_section_l389_389106


namespace intersection_of_A_and_B_l389_389728

def A : Set ℤ := {0, 1, 2}
def B : Set ℤ := {-2, -1, 0, 1}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} :=
by
  sorry

end intersection_of_A_and_B_l389_389728


namespace total_highlighters_correct_l389_389780

variable (y p b : ℕ)
variable (total_highlighters : ℕ)

def num_yellow_highlighters := 7
def num_pink_highlighters := num_yellow_highlighters + 7
def num_blue_highlighters := num_pink_highlighters + 5
def total_highlighters_in_drawer := num_yellow_highlighters + num_pink_highlighters + num_blue_highlighters

theorem total_highlighters_correct : 
  total_highlighters_in_drawer = 40 :=
sorry

end total_highlighters_correct_l389_389780


namespace polyhedron_vertex_assignment_l389_389879

theorem polyhedron_vertex_assignment (V : Type) (E : V → V → Prop) [finite V] :
  (∃ f : V → ℕ, (∀ v1 v2 : V, E v1 v2 → Nat.gcd (f v1) (f v2) > 1) ∧
                (∀ v1 v2 : V, ¬E v1 v2 → Nat.gcd (f v1) (f v2) = 1)) :=
begin
  sorry
end

end polyhedron_vertex_assignment_l389_389879


namespace two_buckets_have_40_liters_l389_389194

def liters_in_jug := 5
def jugs_in_bucket := 4
def liters_in_bucket := liters_in_jug * jugs_in_bucket
def buckets := 2

theorem two_buckets_have_40_liters :
  buckets * liters_in_bucket = 40 :=
by
  sorry

end two_buckets_have_40_liters_l389_389194


namespace contains_zero_l389_389255

open Nat

theorem contains_zero 
  (x y : ℕ)
  (hx : 10000 ≤ x ∧ x < 100000)
  (hy : 10000 ≤ y ∧ y < 100000)
  (h_swap : ∃ (i j : ℕ), i ≠ j ∧ swap_digits x i j = y)
  (h_sum : x + y = 111111) :
  (∃ i, digit x i = 0) ∨ (∃ j, digit y j = 0) :=
sorry

-- Swap two digits in a number
def swap_digits (n : ℕ) (i j : ℕ) : ℕ := 
  -- Dummy definition for the proof placeholder
  n -- replace with actual implementation

-- Extract the digit at position i
def digit (n : ℕ) (i : ℕ) : ℕ :=
  -- Dummy definition for the proof placeholder
  0 -- replace with actual implementation

end contains_zero_l389_389255


namespace a_values_l389_389053

def A (a : ℤ) : Set ℤ := {2, a^2 - a + 2, 1 - a}

theorem a_values (a : ℤ) (h : 4 ∈ A a) : a = 2 ∨ a = -3 :=
sorry

end a_values_l389_389053


namespace probability_of_sum_4_or_16_l389_389580

open Finset

def dice_rolls : Finset (ℕ × ℕ) :=
  (range 6).product (range 6)

def successful_outcomes : Finset (ℕ × ℕ) :=
  filter (λ (roll : ℕ × ℕ), roll.1 + roll.2 + 2 = 4) dice_rolls

theorem probability_of_sum_4_or_16 : 
  (card successful_outcomes : ℚ) / (card dice_rolls) = 1 / 12 :=
by
  sorry

end probability_of_sum_4_or_16_l389_389580


namespace sixty_different_numerators_l389_389837

theorem sixty_different_numerators : 
  (Finset.card {ab : ℕ | 1 ≤ ab ∧ ab ≤ 99 ∧ Nat.gcd ab 99 = 1} = 60) :=
sorry

end sixty_different_numerators_l389_389837


namespace ratio_of_seventh_terms_l389_389906

theorem ratio_of_seventh_terms 
  (a_1 b_1 d1 d2 : ℝ) (n : ℕ) 
  (hn : n > 0)
  (h_ratio : 2 * (5 * (n:ℝ) * (2 * a_1 + (n - 1) * d1)) = 7 * (2 * (n:ℝ) * (2 * b_1 + (n - 1) * d2))) 
  :
  (a_1 + 6 * d1) / (b_1 + 6 * d2) = 5 / 7 := 
by {
  sorry,
}

end ratio_of_seventh_terms_l389_389906


namespace neg_p_iff_exists_ge_zero_l389_389709

variable (x : ℝ)

def p : Prop := ∀ x : ℝ, x^2 + x + 1 < 0

theorem neg_p_iff_exists_ge_zero : ¬ p ↔ ∃ x : ℝ, x^2 + x + 1 ≥ 0 :=
by 
   sorry

end neg_p_iff_exists_ge_zero_l389_389709


namespace triangle_shape_l389_389757

theorem triangle_shape 
  (a b : ℕ) 
  (h1 : a = 3) 
  (h2 : b = 5) 
  (x : ℝ) 
  (hx : 3 * x^2 - 10 * x = 8) 
  (hx_nonneg : x > 0) 
  (hx_int : x ∈ {3, 4, 5}) :
  (a^2 + b^2 = 4^2 ∨ b^2 + 4^2 = a^2 ∨ a^2 + 4^2 = b^2) :=
by
  sorry

end triangle_shape_l389_389757


namespace sum_of_num_denom_eq_52_l389_389712

-- Definitions of conditions
def set_a := {a : ℕ | 1 ≤ a ∧ a ≤ 1000}
def set_b (a_values : set ℕ) := {b : ℕ | b ∉ a_values ∧ 1 ≤ b ∧ b ≤ 1000}

-- Probability q as a fraction in lowest terms
def probability_q (a_values b_values : set ℕ) :=
  let q := (20 + 10 + 4) / 70
  let numerator_denominator_sum := 17 + 35
  numerator_denominator_sum

theorem sum_of_num_denom_eq_52 (a_values b_values : set ℕ) :
  a_values ⊆ set_a ∧ b_values ⊆ set_b a_values ∧ a_values.card = 4 ∧ b_values.card = 4 →
  probability_q a_values b_values = 52 :=
by sorry

end sum_of_num_denom_eq_52_l389_389712


namespace quadratic_inequality_solution_l389_389685

theorem quadratic_inequality_solution (x : ℝ) : (x^2 + 3 * x - 18 > 0) ↔ (x < -6 ∨ x > 3) := 
sorry

end quadratic_inequality_solution_l389_389685


namespace complex_expression_proof_l389_389673

noncomputable def compute_expression : ℂ :=
  (3 * complex.cos (real.pi / 6) + 3 * complex.sin (real.pi / 6) * complex.I) ^ 8

theorem complex_expression_proof :
  compute_expression = -3280.5 - 3280.5 * real.sqrt 3 * complex.I :=
by
  sorry

end complex_expression_proof_l389_389673


namespace average_visitors_per_day_l389_389195

theorem average_visitors_per_day (avg_sunday : ℕ) (avg_other_day : ℕ) (days_in_month : ℕ) (starts_on_sunday : Bool) :
  avg_sunday = 570 →
  avg_other_day = 240 →
  days_in_month = 30 →
  starts_on_sunday = true →
  (5 * avg_sunday + 25 * avg_other_day) / days_in_month = 295 :=
by
  intros
  sorry

end average_visitors_per_day_l389_389195


namespace contains_zero_l389_389249

-- Define a five-digit number in terms of its digits.
def five_digit_number (A B C D E : ℕ) : ℕ :=
  10^4 * A + 10^3 * B + 10^2 * C + 10 * D + E

-- Define the conditions:
noncomputable def switch_two_digits (A B C D E F : ℕ) : Prop :=
  (10^4 * A + 10^3 * B + 10^2 * C + 10 * D + E ≠ 10^4 * A + 10^3 * B + 10^2 * F + 10 * D + E)

-- Lean statement for the proof problem.
theorem contains_zero (A B C D E F : ℕ) :
  let ABCDE := five_digit_number A B C D E,
      ABFDE := five_digit_number A B F D E in
  switch_two_digits A B C D E F ∧ (ABCDE + ABFDE = 111111) → (A = 0 ∨ B = 0 ∨ C = 0 ∨ D = 0 ∨ E = 0) :=
by
  -- This is where the proof would go.
  sorry

end contains_zero_l389_389249


namespace at_least_one_closed_under_mul_l389_389055

-- Definitions based on the given conditions
def is_closed_under_mul (S : Set ℤ) : Prop :=
  ∀ a b ∈ S, a * b ∈ S

def disjoint (S T : Set ℤ) : Prop :=
  ∀ a, a ∈ S → a ∉ T

def covers_universe (T V : Set ℤ) : Prop :=
  ∀ a : ℤ, a ∈ T ∨ a ∈ V

def closed_under_triplet_mul (S : Set ℤ) : Prop :=
  ∀ a b c ∈ S, a * b * c ∈ S

-- Main theorem to be proved
theorem at_least_one_closed_under_mul
  (T V : Set ℤ)
  (hT_nonempty : ∃ t, t ∈ T)
  (hV_nonempty : ∃ v, v ∈ V)
  (h_disjoint : disjoint T V)
  (h_cover : covers_universe T V)
  (hT_triplet : closed_under_triplet_mul T)
  (hV_triplet : closed_under_triplet_mul V) :
  is_closed_under_mul T ∨ is_closed_under_mul V :=
sorry

end at_least_one_closed_under_mul_l389_389055


namespace binomial_coefficient_largest_term_constant_term_polynomial_l389_389359

noncomputable def polynomial_expansion (x a : ℝ) (n : ℕ) : ℝ :=
(a * x^(-1/2) + x)^n

noncomputable def expanded_expression (x a : ℝ) (n : ℕ) : ℝ :=
  polynomial_expansion (x^1.5) a n - polynomial_expansion x (-1 / x)

theorem binomial_coefficient_largest_term :
  ∀ (n : ℕ) (a : ℝ),
    (∃ (C : ℕ → ℝ), C 1 / C 2 = 1/4) → -- Condition 1
    (polynomial_expansion 1 a n = 512) →  -- Condition 2
    (∃ (k : ℕ), 7 = k + 1) → -- Condition 3
    a = 1 ∧
    (let binom_coeff := λ (k : ℕ), choose n k * (a : ℝ)^k in
    (∃ (m₁ m₂ : ℕ), (binom_coeff m₁ = 126) ∨ (binom_coeff m₂ = 126))) := sorry

theorem constant_term_polynomial :
  ∀ (n : ℕ) (a : ℝ),
    (∃ (C : ℕ → ℝ), C 1 / C 2 = 1/4) → -- Condition 1
    (polynomial_expansion 1 a n = 512) →  -- Condition 2
    (∃ (k : ℕ), 7 = k + 1) → -- Condition 3
    expanded_expression x a n = -48 := sorry

end binomial_coefficient_largest_term_constant_term_polynomial_l389_389359


namespace three_digit_number_units_digit_condition_l389_389390

theorem three_digit_number_units_digit_condition :
  let valid_numbers := (λ (n : ℕ), (100 ≤ n ∧ n ≤ 999) ∧ ((n % 10) ≥ 3 * ((n / 10) % 10))) in
  (Finset.filter valid_numbers (Finset.range 900)) = 198 :=
by
  sorry

end three_digit_number_units_digit_condition_l389_389390


namespace max_dwarves_l389_389148

theorem max_dwarves (d : Type) [finite d]
  (has_hat : d → Set ℕ)
  (hats_correct : ∀ x : d, ∃ (a b c : ℕ), has_hat x = {a, b, c} ∧ 1 ≤ a ∧ a < b ∧ b < c ∧ c ≤ 28)
  (no_overlap : ∀ x y : d, x ≠ y → (has_hat x ∩ has_hat y).card ≤ 1) :
  finite.card d ≤ 182 :=
by
  sorry

end max_dwarves_l389_389148


namespace distinct_complex_number_count_l389_389532

noncomputable def num_distinct_complex_numbers : Nat :=
  let S := {0, 1, 2, 3, 4, 5, 6}
  S.erase 0.size * S.size

theorem distinct_complex_number_count :
  num_distinct_complex_numbers = 36 :=
by
  sorry

end distinct_complex_number_count_l389_389532


namespace intersection_of_sets_l389_389372

open Set

def A : Set ℕ := \{0, 1, 2, 3\}
def B : Set ℕ := \{2, 3, 4, 5\}

theorem intersection_of_sets : A ∩ B = \{2, 3\} := by
  sorry

end intersection_of_sets_l389_389372


namespace melinda_textbooks_problem_l389_389515

theorem melinda_textbooks_problem :
  let total_ways := (nat.choose 15 4) * (nat.choose 11 5) * (nat.choose 6 6),
      ways_all_math_same_box := (nat.choose 11 5) + (nat.choose 11 1) * (nat.choose 6 6) + (nat.choose 11 2),
      probability := ways_all_math_same_box.to_nat_gcd (total_ways / ways_all_math_same_box),
      m := probability.1,
      n := probability.2
  in m + n = 9551 := sorry

end melinda_textbooks_problem_l389_389515


namespace number_of_valid_3_digit_numbers_l389_389412

theorem number_of_valid_3_digit_numbers : 
  ∃ (n : ℕ), 
    (∀ (h t u : ℕ), 
      (n = h * 100 + t * 10 + u) ∧ 
      (1 ≤ h ∧ h ≤ 9) ∧ 
      (0 ≤ t ∧ t ≤ 9) ∧ 
      (0 ≤ u ∧ u ≤ 9) ∧ 
      (u ≥ 3 * t)) →
      n = 198 := 
by
  sorry

end number_of_valid_3_digit_numbers_l389_389412


namespace contains_zero_l389_389257

open Nat

theorem contains_zero 
  (x y : ℕ)
  (hx : 10000 ≤ x ∧ x < 100000)
  (hy : 10000 ≤ y ∧ y < 100000)
  (h_swap : ∃ (i j : ℕ), i ≠ j ∧ swap_digits x i j = y)
  (h_sum : x + y = 111111) :
  (∃ i, digit x i = 0) ∨ (∃ j, digit y j = 0) :=
sorry

-- Swap two digits in a number
def swap_digits (n : ℕ) (i j : ℕ) : ℕ := 
  -- Dummy definition for the proof placeholder
  n -- replace with actual implementation

-- Extract the digit at position i
def digit (n : ℕ) (i : ℕ) : ℕ :=
  -- Dummy definition for the proof placeholder
  0 -- replace with actual implementation

end contains_zero_l389_389257


namespace days_after_monday_l389_389668

theorem days_after_monday (n : ℕ) (h₀ : n % 7 = 3) : n = 45 → day_of_week_after "Monday" 45 = "Thursday" :=
by
  intros h₁
  rw [h₁]
  sorry
  
noncomputable def day_of_week_after (d : string) (n : ℕ) : string :=
  let days := ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
  days[(days.index_of d + n % 7) % days.length]

example : day_of_week_after "Monday" 45 = "Thursday" :=
  by sorry

end days_after_monday_l389_389668


namespace logarithmic_function_l389_389211

variable {R : Type*} [OrderedRing R] [Nontrivial R]

-- Given function f : R → R, with the property:
def satisfies_property (f : R → R) : Prop :=
  ∀ x y : R, 0 < x → 0 < y → f (x * y) = f x + f y

-- We need to prove that this function is a logarithmic function
theorem logarithmic_function (f : R → R) :
  satisfies_property f → ∃ log_const : R, ∀ x : R, 0 < x → f x = log_const * log x :=
sorry

end logarithmic_function_l389_389211


namespace number_of_valid_3_digit_numbers_l389_389409

theorem number_of_valid_3_digit_numbers : 
  ∃ (n : ℕ), 
    (∀ (h t u : ℕ), 
      (n = h * 100 + t * 10 + u) ∧ 
      (1 ≤ h ∧ h ≤ 9) ∧ 
      (0 ≤ t ∧ t ≤ 9) ∧ 
      (0 ≤ u ∧ u ≤ 9) ∧ 
      (u ≥ 3 * t)) →
      n = 198 := 
by
  sorry

end number_of_valid_3_digit_numbers_l389_389409


namespace store_hours_open_per_day_l389_389037

theorem store_hours_open_per_day
  (rent_per_week : ℝ)
  (utility_percentage : ℝ)
  (employees_per_shift : ℕ)
  (hourly_wage : ℝ)
  (days_per_week_open : ℕ)
  (weekly_expenses : ℝ)
  (H_rent : rent_per_week = 1200)
  (H_utility_percentage : utility_percentage = 0.20)
  (H_employees_per_shift : employees_per_shift = 2)
  (H_hourly_wage : hourly_wage = 12.50)
  (H_days_open : days_per_week_open = 5)
  (H_weekly_expenses : weekly_expenses = 3440) :
  (16 : ℝ) = weekly_expenses / ((rent_per_week * (1 + utility_percentage)) + (employees_per_shift * hourly_wage * days_per_week_open)) :=
by
  sorry

end store_hours_open_per_day_l389_389037


namespace triangle_sides_are_6_8_10_l389_389909

theorem triangle_sides_are_6_8_10 (a b c r r1 r2 r3 : ℕ) (hr_even : Even r) (hr1_even : Even r1) 
(hr2_even : Even r2) (hr3_even : Even r3) (relationship : r * r1 * r2 + r * r2 * r3 + r * r3 * r1 + r1 * r2 * r3 = r * r1 * r2 * r3) :
  (a, b, c) = (6, 8, 10) :=
sorry

end triangle_sides_are_6_8_10_l389_389909


namespace log_3100_nearest_int_l389_389940

theorem log_3100_nearest_int :
  ∀ (log : ℝ → ℝ), 
    (∀ x, strict_mono log) → 
    log 3125 = 5 → 
    log 625 = 4 → 
    625 < 3100 ∧ 3100 < 3125 → 
    Nat.round (log 3100) = 5 :=
by
  intros log h_mono h1 h2 h3
  sorry

end log_3100_nearest_int_l389_389940


namespace part1_part2_l389_389760

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define the set A dependent on parameter b
def A (b : ℝ) : Set ℝ := {x | x^2 - 3*x + b = 0}

-- Define the set B
def B : Set ℝ := {x | (x-2)*(x^2 + 3*x - 4) = 0}

-- Define the complement of B in U
def complement_B_U : Set ℝ := U \ B

-- Part (1): Find the sets M such that A ⊊ M ⊊ B, for b = 4
theorem part1 (b : ℝ) (h : b = 4) : 
  Finset.card {M : Set ℝ | A b ⊂ M ∧ M ⊂ B} = 6 := 
sorry

-- Part (2): Prove the range of b for which (complement_B_U ∩ A = ∅)
theorem part2 : 
  {b : ℝ | (complement_B_U ∩ A b = ∅)} = 
  Set.Ioi (9/4) ∪ {2} := 
sorry

end part1_part2_l389_389760


namespace parabola_standard_equation_l389_389354

theorem parabola_standard_equation (vertex_origin : (0, 0) ∈ parabola) 
    (axis_of_symmetry_coord_axes : axis_of_symmetry parabola ∈ {x_axis, y_axis}) 
    (focus_on_line : ∃ p : ℝ × ℝ, p ∈ focus parabola ∧ (2 * p.1 - p.2 - 4 = 0)) : 
    (∃ a : ℝ, parabola = { p | p.2 ^ 2 = 8 * p.1 } ∨ parabola = { p | p.1 ^ 2 = -16 * p.2 }) :=
by
  sorry

end parabola_standard_equation_l389_389354


namespace normal_level_shortage_l389_389651

variable (T : ℝ) (normal_capacity : ℝ) (end_of_month_reservoir : ℝ)
variable (h1 : end_of_month_reservoir = 6)
variable (h2 : end_of_month_reservoir = 2 * normal_capacity)
variable (h3 : end_of_month_reservoir = 0.60 * T)

theorem normal_level_shortage :
  normal_capacity = 7 :=
by
  sorry

end normal_level_shortage_l389_389651


namespace carl_garden_area_l389_389666

theorem carl_garden_area (x : ℕ) (longer_side_post_count : ℕ) (total_posts : ℕ) 
  (shorter_side_length : ℕ) (longer_side_length : ℕ) 
  (posts_per_gap : ℕ) (spacing : ℕ) :
  -- Conditions
  total_posts = 20 → 
  posts_per_gap = 4 → 
  spacing = 4 → 
  longer_side_post_count = 2 * x → 
  2 * x + 2 * (2 * x) - 4 = total_posts →
  shorter_side_length = (x - 1) * spacing → 
  longer_side_length = (longer_side_post_count - 1) * spacing →
  -- Conclusion
  shorter_side_length * longer_side_length = 336 :=
by
  sorry

end carl_garden_area_l389_389666


namespace renee_allergic_probability_l389_389040

theorem renee_allergic_probability :
  let peanut_butter_from_jenny := 40
  let chocolate_chip_from_jenny := 50
  let peanut_butter_from_marcus := 30
  let lemon_from_marcus := 20
  let total_cookies := peanut_butter_from_jenny + chocolate_chip_from_jenny + peanut_butter_from_marcus + lemon_from_marcus
  let total_peanut_butter := peanut_butter_from_jenny + peanut_butter_from_marcus
  let p := (total_peanut_butter : ℝ) / (total_cookies : ℝ) * 100
  in p = 50 := by sorry

end renee_allergic_probability_l389_389040


namespace jacob_three_heads_probability_l389_389806

noncomputable section

def probability_three_heads_after_two_tails : ℚ := 1 / 96

theorem jacob_three_heads_probability :
  let p := (1 / 2) ^ 4 * (1 / 6)
  p = probability_three_heads_after_two_tails := by
sorry

end jacob_three_heads_probability_l389_389806


namespace largest_exterior_angle_l389_389566

theorem largest_exterior_angle (x : ℝ) (h1 : 3 * x + 4 * x + 5 * x = 180) : 180 - 3 * (180 / 12) = 135 :=
by {
  -- Sorry is a placeholder for the actual proof
  sorry
}

end largest_exterior_angle_l389_389566


namespace shorter_piece_length_l389_389607

def wireLength := 150
def ratioLongerToShorter := 5 / 8

theorem shorter_piece_length : ∃ x : ℤ, x + (5 / 8) * x = wireLength ∧ x = 92 := by
  sorry

end shorter_piece_length_l389_389607


namespace reciprocal_of_repeating_decimal_l389_389978

theorem reciprocal_of_repeating_decimal :
  let x : ℚ := 36 / 99 in
  1 / x = 11 / 4 :=
by
  sorry

end reciprocal_of_repeating_decimal_l389_389978


namespace mildred_weight_l389_389078

theorem mildred_weight (carol_weight mildred_is_heavier : ℕ) (h1 : carol_weight = 9) (h2 : mildred_is_heavier = 50) :
  carol_weight + mildred_is_heavier = 59 :=
by
  sorry

end mildred_weight_l389_389078


namespace water_volume_per_minute_l389_389200

theorem water_volume_per_minute (depth width flow_rate_kmph : ℕ) 
  (h_depth : depth = 4) 
  (h_width : width = 65) 
  (h_flow_rate : flow_rate_kmph = 6): 
  (depth * width * (flow_rate_kmph * 1000 / 60)) = 26000 :=
by
  -- Given depth and width of the river
  have h_area: depth * width = 260, by { rw [h_depth, h_width], norm_num }

  -- Given flow rate in m/min
  have h_flow_rate: flow_rate_kmph * 1000 / 60 = 100, by { rw h_flow_rate, norm_num }

  -- Calculate the volume per minute
  rw [h_area, h_flow_rate]
  norm_num
  sorry

end water_volume_per_minute_l389_389200


namespace range_of_m_l389_389010

-- Defining the quadratic function with the given condition
def quadratic (m : ℝ) (x : ℝ) : ℝ := (m-1)*x^2 + (m-1)*x + 2

-- Stating the problem
theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, quadratic m x > 0) ↔ 1 ≤ m ∧ m < 9 :=
by
  sorry

end range_of_m_l389_389010


namespace dodgeballs_purchasable_l389_389138

-- Definitions for the given conditions
def original_budget (B : ℝ) := B
def new_budget (B : ℝ) := 1.2 * B
def cost_per_dodgeball : ℝ := 5
def cost_per_softball : ℝ := 9
def softballs_purchased (B : ℝ) := 10

-- Theorem statement
theorem dodgeballs_purchasable {B : ℝ} (h : new_budget B = 90) : original_budget B / cost_per_dodgeball = 15 := 
by 
  sorry

end dodgeballs_purchasable_l389_389138


namespace three_digit_numbers_count_l389_389432

theorem three_digit_numbers_count :
  let t_range := finset.range 10
  let h_range := finset.range 9
  let valid_u t := if t * 3 ≤ 9 then 9 - (t * 3) + 1 else 0
  let total_count := h_range.card * t_range.sum (λ t, valid_u t)
  total_count = 198 :=
by
  sorry

end three_digit_numbers_count_l389_389432


namespace find_m_values_l389_389304

noncomputable def s (n m : ℕ) : ℕ :=
  Nat.card { k : ℕ | n ≤ k ∧ k ≤ m ∧ Nat.coprime k m ∧ k > 0 }

theorem find_m_values :
  ∀ m: ℕ, m ≥ 2 →
  (∀ n : ℕ, 1 ≤ n ∧ n < m → (s n m) / (m - n) ≥ (s 1 m) / m) →
  (m * m ∣ 2022 ^ m + 1) →
  m = 7 ∨ m = 17 :=
by
  sorry

end find_m_values_l389_389304


namespace simplify_expression_l389_389883

variables (a b : ℝ)
noncomputable def x := (1 / 2) * (Real.sqrt (a / b) - Real.sqrt (b / a))

theorem simplify_expression (ha : a > 0) (hb : b > 0) :
  (2 * a * Real.sqrt (1 + x a b ^ 2)) / (x a b + Real.sqrt (1 + x a b ^ 2)) = a + b :=
sorry

end simplify_expression_l389_389883


namespace fraction_of_remaining_water_used_is_half_l389_389809

theorem fraction_of_remaining_water_used_is_half :
  ∀ (initial_water water_per_car num_cars water_plates_clothes : ℕ), 
  initial_water = 65 →
  water_per_car = 7 →
  num_cars = 2 →
  water_plates_clothes = 24 →
  let water_for_cars := water_per_car * num_cars in
  let water_for_plants := water_for_cars - 11 in
  let total_water_used := water_for_cars + water_for_plants in
  let remaining_water := initial_water - total_water_used in
  remaining_water = 48 →
  (water_plates_clothes : ℚ) / (remaining_water : ℚ) = 1 / 2 :=
by
  intros; sorry

end fraction_of_remaining_water_used_is_half_l389_389809


namespace closest_percentage_change_is_21_l389_389483

noncomputable def item_prices : List ℚ := [13.24, 7.95, 3.75, 10.99, 3.45]
def total_payment : ℚ := 50.00

def total_cost (prices : List ℚ) : ℚ := prices.sum
def change (payment cost : ℚ) : ℚ := payment - cost
def percentage (part whole : ℚ) : ℚ := (part / whole) * 100

theorem closest_percentage_change_is_21 :
  percentage (change total_payment (total_cost item_prices)) total_payment ≈ 21 :=
sorry

end closest_percentage_change_is_21_l389_389483


namespace digit_contains_zero_l389_389229

theorem digit_contains_zero 
  (n₁ n₂ : ℕ) 
  (h₁ : 10000 ≤ n₁ ∧ n₁ ≤ 99999)
  (h₂ : 10000 ≤ n₂ ∧ n₂ ≤ 99999)
  (h₃ : ∃ (i j : ℕ), i ≠ j ∧ n₂ = n₁.swap_digits i j)
  (h₄ : n₁ + n₂ = 111111) 
  : ∃ (d : ℕ), d = 0 :=
sorry

end digit_contains_zero_l389_389229


namespace initial_games_l389_389489

def games_given_away : ℕ := 91
def games_left : ℕ := 92

theorem initial_games :
  games_given_away + games_left = 183 :=
by
  sorry

end initial_games_l389_389489


namespace Noah_age_in_10_years_is_22_l389_389083

def Joe_age : Nat := 6
def Noah_age := 2 * Joe_age
def Noah_age_after_10_years := Noah_age + 10

theorem Noah_age_in_10_years_is_22 : Noah_age_after_10_years = 22 := by
  sorry

end Noah_age_in_10_years_is_22_l389_389083


namespace number_of_correct_propositions_l389_389511

def f (x b c : ℝ) := x * |x| + b * x + c

def proposition1 (b : ℝ) : Prop :=
  ∀ (x : ℝ), f x b 0 = -f (-x) b 0

def proposition2 (c : ℝ) : Prop :=
  c > 0 → ∃ (x : ℝ), ∀ (y : ℝ), f y 0 c = 0 → y = x

def proposition3 (b c : ℝ) : Prop :=
  ∀ (x : ℝ), f x b c = f (-x) b c + 2 * c

def proposition4 (b c : ℝ) : Prop :=
  ∀ (x₁ x₂ x₃ : ℝ), f x₁ b c = 0 → f x₂ b c = 0 → f x₃ b c = 0 → x₁ = x₂ ∨ x₂ = x₃ ∨ x₁ = x₃

theorem number_of_correct_propositions (b c : ℝ) : 
  1 + (if c > 0 then 1 else 0) + 1 + 0 = 3 :=
  sorry

end number_of_correct_propositions_l389_389511


namespace initial_average_mark_l389_389889

theorem initial_average_mark (A : ℝ) (n_total n_excluded remaining_students_avg : ℝ) 
  (h1 : n_total = 25) 
  (h2 : n_excluded = 5) 
  (h3 : remaining_students_avg = 90)
  (excluded_students_avg : ℝ)
  (h_excluded_avg : excluded_students_avg = 40)
  (A_def : (n_total * A) = (n_excluded * excluded_students_avg + (n_total - n_excluded) * remaining_students_avg)) :
  A = 80 := 
by
  sorry

end initial_average_mark_l389_389889


namespace MNPQ_cyclic_l389_389049

open EuclideanGeometry

variable {A B C M N P Q : Point}
variable {angleA angleB angleC : Angle}
variable {ABC : Triangle}

-- given conditions
axiom angle_condition : angleA < angleB ∧ angleB ≤ angleC
axiom M_midpoint_CA : Midpoint M C A
axiom N_midpoint_AB : Midpoint N A B
axiom P_projection_B_CN : Projection P B (Median A B C N)
axiom Q_projection_C_BM : Projection Q C (Median A C B M)

theorem MNPQ_cyclic (h1 : angleA < angleB) (h2 : angleB ≤ angleC)
    (h3 : Midpoint M C A) (h4 : Midpoint N A B)
    (h5 : Projection P B (Median A B C N)) (h6 : Projection Q C (Median A C B M)) :
    CyclicQuadrilateral M N P Q :=
sorry

end MNPQ_cyclic_l389_389049


namespace count_valid_three_digit_numbers_l389_389423

-- Define the property of a number being three digits
def isThreeDigit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

-- Define the property of the units digit being at least three times the tens digit
def validUnitsTimesTens (n : ℕ) : Prop :=
  let units := n % 10
  let tens := (n / 10) % 10
  units ≥ 3 * tens

-- Definition combining both conditions
def validThreeDigitNumber (n : ℕ) : Prop :=
  isThreeDigit n ∧ validUnitsTimesTens n

-- The main theorem stating the number of valid three-digit numbers
theorem count_valid_three_digit_numbers :
  Finset.card (Finset.filter validThreeDigitNumber (Finset.range 1000)) = 198 :=
by
  sorry

end count_valid_three_digit_numbers_l389_389423


namespace digit_contains_zero_l389_389223

theorem digit_contains_zero 
  (n₁ n₂ : ℕ) 
  (h₁ : 10000 ≤ n₁ ∧ n₁ ≤ 99999)
  (h₂ : 10000 ≤ n₂ ∧ n₂ ≤ 99999)
  (h₃ : ∃ (i j : ℕ), i ≠ j ∧ n₂ = n₁.swap_digits i j)
  (h₄ : n₁ + n₂ = 111111) 
  : ∃ (d : ℕ), d = 0 :=
sorry

end digit_contains_zero_l389_389223


namespace contains_zero_l389_389231

-- Define what constitutes a five-digit number
def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

-- Define the main proof statement
theorem contains_zero (n1 n2 : ℕ) (c1 c2 : Prop) :
  is_five_digit n1 →
  is_five_digit n2 →
  c1 ∧ c2 →
  n1 + n2 = 111111 →
  ∃ (d : ℕ), d < 10 ∧ (d = 0 ∧ (∃ p1 p2 : ℕ, n1 = p1 * 10 + d ∧ n2 = p2 * 10 + d) ∨ 
             ∃ p1 p2 : ℕ, n1 = d * (10^4) + p1 ∧ n2 = d * (10^4) + p2) :=
begin
  sorry
end

end contains_zero_l389_389231


namespace train_speed_correct_l389_389636

def length_of_train : ℕ := 700
def time_to_cross_pole : ℕ := 20
def expected_speed : ℕ := 35

theorem train_speed_correct : (length_of_train / time_to_cross_pole) = expected_speed := by
  sorry

end train_speed_correct_l389_389636


namespace basketball_player_scores_mode_median_l389_389613

theorem basketball_player_scores_mode_median :
  let scores := [20, 18, 23, 17, 20, 20, 18]
  let ordered_scores := List.sort scores
  let mode := 20
  let median := 20
  (mode = List.maximum (List.frequency ordered_scores)) ∧ 
  (median = List.nthLe ordered_scores (List.length ordered_scores / 2) sorry) :=
by
  sorry

end basketball_player_scores_mode_median_l389_389613


namespace inequality_proof_l389_389494

variable (a b c d : ℝ)

theorem inequality_proof (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) (h5 : a + b + c + d = 1) : 
  (1 / (4 * a + 3 * b + c) + 1 / (3 * a + b + 4 * d) + 1 / (a + 4 * c + 3 * d) + 1 / (4 * b + 3 * c + d)) ≥ 2 :=
by
  sorry

end inequality_proof_l389_389494


namespace contains_zero_l389_389242

theorem contains_zero (a b c d e f: ℕ) (h1: 1 ≤ a ∧ a ≤ 9) (h2: 1 ≤ b ∧ b ≤ 9) (h3: 0 ≤ c ∧ c ≤ 9) 
  (h4: 1 ≤ d ∧ d ≤ 9) (h5: 0 ≤ e ∧ e ≤ 9) (h6: 0 ≤ f ∧ f ≤ 9) 
  (h7: c ≠ f) (h8: 10^4*a + 10^3*b + 10^2*c + 10^1*d + e + 10^4*a + 10^3*b + 10^2*f + 10^1*d + e = 111111) :
  c = 0 ∨ f = 0 := 
sorry

end contains_zero_l389_389242


namespace angle_GH_at_F_is_perpendicular_l389_389020

structure RightIsoscelesTriangle :=
(a b c p g h f : Point)
(right_angle_C : angle a c b = 90°)
(isosceles : dist a c = dist b c)
(on_line_bc : on_line_segment c b p)
(perpendicular_projection : is_perpendicular (g - c) (ap_segment a p))
(equal_segments : dist a h = dist c g)
(midpoint_of_ab : midpoint a b f)
 
theorem angle_GH_at_F_is_perpendicular (T : RightIsoscelesTriangle)
  : angle (T.f) (T.g) (T.h) = 90° :=
sorry

end angle_GH_at_F_is_perpendicular_l389_389020


namespace contains_zero_l389_389239

theorem contains_zero (a b c d e f: ℕ) (h1: 1 ≤ a ∧ a ≤ 9) (h2: 1 ≤ b ∧ b ≤ 9) (h3: 0 ≤ c ∧ c ≤ 9) 
  (h4: 1 ≤ d ∧ d ≤ 9) (h5: 0 ≤ e ∧ e ≤ 9) (h6: 0 ≤ f ∧ f ≤ 9) 
  (h7: c ≠ f) (h8: 10^4*a + 10^3*b + 10^2*c + 10^1*d + e + 10^4*a + 10^3*b + 10^2*f + 10^1*d + e = 111111) :
  c = 0 ∨ f = 0 := 
sorry

end contains_zero_l389_389239


namespace reported_length_correct_l389_389128

def length_in_yards := 80
def conversion_factor := 3 -- 1 yard is 3 feet
def length_in_feet := 240

theorem reported_length_correct :
  length_in_feet = length_in_yards * conversion_factor :=
by rfl

end reported_length_correct_l389_389128


namespace quadratic_equation_solution_l389_389536

theorem quadratic_equation_solution :
  ∃ x1 x2 : ℝ, (x1 = (-1 + Real.sqrt 13) / 2 ∧ x2 = (-1 - Real.sqrt 13) / 2 
  ∧ (∀ x : ℝ, x^2 + x - 3 = 0 → x = x1 ∨ x = x2)) :=
sorry

end quadratic_equation_solution_l389_389536


namespace sandwich_cost_l389_389303

theorem sandwich_cost 
  (loaf_sandwiches : ℕ) (target_sandwiches : ℕ) 
  (bread_cost : ℝ) (meat_cost : ℝ) (cheese_cost : ℝ) 
  (cheese_coupon : ℝ) (meat_coupon : ℝ) (total_threshold : ℝ) 
  (discount_rate : ℝ)
  (h1 : loaf_sandwiches = 10) 
  (h2 : target_sandwiches = 50) 
  (h3 : bread_cost = 4) 
  (h4 : meat_cost = 5) 
  (h5 : cheese_cost = 4) 
  (h6 : cheese_coupon = 1) 
  (h7 : meat_coupon = 1) 
  (h8 : total_threshold = 60) 
  (h9 : discount_rate = 0.1) :
  ( ∃ cost_per_sandwich : ℝ, 
      cost_per_sandwich = 1.944 ) :=
  sorry

end sandwich_cost_l389_389303


namespace count_valid_numbers_l389_389408

def is_valid_number (n : Nat) : Prop := 
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let units := n % 10
  1 ≤ hundreds ∧ hundreds ≤ 9 ∧
  0 ≤ tens ∧ tens ≤ 9 ∧
  units ≥ 3 * tens

theorem count_valid_numbers : 
  (Finset.filter is_valid_number (Finset.range 1000)).card = 198 :=
by sorry

end count_valid_numbers_l389_389408


namespace abs_inequality_const_l389_389145

theorem abs_inequality_const {k : ℕ} (h : 8 = (finset.filter (λ x, |x + 4| < k) (finset.range 100)).card) : k = 13 :=
sorry

end abs_inequality_const_l389_389145


namespace constant_term_expansion_l389_389952

-- auxiliary definitions and facts
def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def term_constant (n k : ℕ) (a b x : ℂ) : ℂ :=
  binomial_coeff n k * (a * x)^(n-k) * (b / x)^k

-- main theorem statement
theorem constant_term_expansion : ∀ (x : ℂ), (term_constant 8 4 (5 : ℂ) (2 : ℂ) x).re = 1120 :=
by
  intro x
  sorry

end constant_term_expansion_l389_389952


namespace set_difference_A_B_l389_389860

-- Defining the sets A and B
def setA : Set ℝ := { x : ℝ | abs (4 * x - 1) > 9 }
def setB : Set ℝ := { x : ℝ | x >= 0 }

-- The theorem stating the result of set difference A - B
theorem set_difference_A_B : (setA \ setB) = { x : ℝ | x > 5/2 } :=
by
  -- Proof omitted
  sorry

end set_difference_A_B_l389_389860


namespace area_inside_circle_but_outside_rectangle_l389_389627

theorem area_inside_circle_but_outside_rectangle
  (length : ℝ) (width : ℝ) (radius : ℝ)
  (center : ℝ × ℝ)
  (h_length : length = 3)
  (h_width : width = 1.5)
  (h_radius : radius = 1 / 3) :
  let area_circle := π * (radius ^ 2),
      area_rectangle := length * width
  in area_circle - area_rectangle = (π / 9) :=
by
  -- Proof is intentionally skipped
  sorry

end area_inside_circle_but_outside_rectangle_l389_389627


namespace b_in_terms_of_a_axis_of_symmetry_and_a_value_range_of_a_l389_389321

-- Definitions based on problem conditions
def parabola (a b x : ℝ) : ℝ := a * x^2 + b * x
def passes_through_A (a b : ℝ) : Prop := parabola a b 3 = 3
def points_on_parabola (a b x1 x2 : ℝ) : Prop := x1 < x2 ∧ x1 + x2 = 2
def equal_y_values (a b x1 x2 : ℝ) : Prop := parabola a b x1 = parabola a b x2
def less_than_y_values (a b x1 x2 : ℝ) : Prop := parabola a b x1 < parabola a b x2

-- 1) Express b in terms of a
theorem b_in_terms_of_a (a : ℝ) (h : passes_through_A a (1 - 3 * a)) : True := sorry

-- 2) Axis of symmetry and the value of a when y1 = y2
theorem axis_of_symmetry_and_a_value (a : ℝ) (x1 x2 : ℝ) 
    (h1 : points_on_parabola a (1 - 3 * a) x1 x2)
    (h2 : equal_y_values a (1 - 3 * a) x1 x2) 
    : a = 1 ∧ -1 / 2 * (1 - 3 * a) / a = 1 := sorry

-- 3) Range of values for a when y1 < y2
theorem range_of_a (a x1 x2 : ℝ) 
    (h1 : points_on_parabola a (1 - 3 * a) x1 x2)
    (h2 : less_than_y_values a (1 - 3 * a) x1 x2) 
    (h3 : a ≠ 0) : 0 < a ∧ a < 1 := sorry

end b_in_terms_of_a_axis_of_symmetry_and_a_value_range_of_a_l389_389321


namespace count_valid_3_digit_numbers_l389_389383

def valid_units_digit (tens_digit units_digit : ℕ) : Prop :=
  units_digit ≥ 3 * tens_digit

def count_valid_numbers : ℕ :=
  (∑ tens_digit in {0, 1, 2, 3}, (finset.filter (valid_units_digit tens_digit) (finset.range 10)).card) * 9

theorem count_valid_3_digit_numbers : count_valid_numbers = 198 :=
by
  sorry

end count_valid_3_digit_numbers_l389_389383


namespace max_multiples_l389_389308

noncomputable def m_max (n p : ℕ) : ℕ :=
  ∑ i in Finset.range (n+1),
    if n / p ^ i = 0 then 0 else (-1) ^ i * (n / p ^ i)

theorem max_multiples (n p : ℕ) : 
  (∃ m, m = m_max n p) ∧ (∀ m, m ≤ m_max n p) :=
sorry

end max_multiples_l389_389308


namespace lucas_annual_income_l389_389139

variable (p A : ℝ)

def income_tax (p A : ℝ) : ℝ :=
  if A ≤ 35000 then
    A * (p / 100)
  else
    35000 * (p / 100) + (A - 35000) * ((p + 4) / 100)

theorem lucas_annual_income (h : (income_tax p A) = A * ((p + 0.5) / 100)) :
  A = 40000 :=
by
  sorry

end lucas_annual_income_l389_389139


namespace inequality_ab_l389_389841

theorem inequality_ab (a b : ℝ) (h : a * b < 0) : |a + b| < |a - b| := 
sorry

end inequality_ab_l389_389841


namespace abs_diff_roots_eq_zero_l389_389604

theorem abs_diff_roots_eq_zero :
  let p : Polynomial ℝ := Polynomial.C 9 + Polynomial.C (-6) * Polynomial.X + Polynomial.X^2
  ∃ x1 x2, Polynomial.isRoot p x1 ∧ Polynomial.isRoot p x2 ∧ |x1 - x2| = 0 :=
by
  let p : Polynomial ℝ := Polynomial.C 9 + Polynomial.C (-6) * Polynomial.X + Polynomial.X^2
  have hroot : Polynomial.roots p = {3, 3} := sorry
  use 3
  use 3
  have h : |3 - 3| = 0 := sorry
  show Polynomial.isRoot p 3 ∧ Polynomial.isRoot p 3 ∧ |3 - 3| = 0 from sorry

end abs_diff_roots_eq_zero_l389_389604


namespace count_valid_3_digit_numbers_l389_389378

def valid_units_digit (tens_digit units_digit : ℕ) : Prop :=
  units_digit ≥ 3 * tens_digit

def count_valid_numbers : ℕ :=
  (∑ tens_digit in {0, 1, 2, 3}, (finset.filter (valid_units_digit tens_digit) (finset.range 10)).card) * 9

theorem count_valid_3_digit_numbers : count_valid_numbers = 198 :=
by
  sorry

end count_valid_3_digit_numbers_l389_389378


namespace square_area_example_l389_389133

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

noncomputable def square_area (x1 y1 x2 y2 : ℝ) : ℝ :=
  (distance x1 y1 x2 y2)^2

theorem square_area_example : square_area 1 3 5 6 = 25 :=
by
  sorry

end square_area_example_l389_389133


namespace det_D_l389_389839

-- Define the 2x2 dilation matrix D with scale factor 7
def D : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![7, 0], ![0, 7]]

-- State the theorem about the determinant of D
theorem det_D : det D = 49 :=
by
  sorry

end det_D_l389_389839


namespace reciprocal_of_repeating_decimal_l389_389983

theorem reciprocal_of_repeating_decimal :
  let x := (36 : ℚ) / 99 in
  x⁻¹ = 11 / 4 :=
by
  have h_simplify : x = 4 / 11 := by sorry
  rw [h_simplify, inv_div]
  norm_num
  exact eq.refl (11 / 4)

end reciprocal_of_repeating_decimal_l389_389983


namespace reciprocal_of_repeating_decimal_l389_389962

theorem reciprocal_of_repeating_decimal :
  let x := 0.36363636 -- simplified as .\overline{36}
  ∃ y : ℚ, x = 4 / 11 ∧ y = 1 / x ∧ y = 11 / 4 :=
by
  sorry

end reciprocal_of_repeating_decimal_l389_389962


namespace changing_quantities_is_3_l389_389875

-- Define the setup
variables {α : Type*} [ordered_field α]  -- assuming coordinates in a field

-- Conditions
variables (A B P : α × α)  -- Points A, B, and P in the plane
variables (M N : α × α)    -- Midpoints M and N
variable (hAB : A ≠ B)     -- A and B are distinct
variable (line_perpendicular_AB : ∀ P, ∃ d : α, P = (d, P.2))  -- P moves perpendicular to AB

-- Definitions of M and N
def midpoint (X Y : α × α) : α × α := ((X.1 + Y.1) / 2, (X.2 + Y.2) / 2)
def M_def : M = midpoint P A := sorry
def N_def : N = midpoint P B := sorry

-- Key result: Number of changing quantities
def changing_quantities : ℕ := 
  let length_MN_changes := false in
  let perimeter_PAB_changes := true in
  let area_PAB_changes := true in
  let area_trapezoid_ABNM_changes := true in
    (if length_MN_changes then 1 else 0) +
    (if perimeter_PAB_changes then 1 else 0) +
    (if area_PAB_changes then 1 else 0) +
    (if area_trapezoid_ABNM_changes then 1 else 0)

-- The main theorem stating the number of changing quantities is 3
theorem changing_quantities_is_3 : changing_quantities = 3 := 
  by 
    sorry

end changing_quantities_is_3_l389_389875


namespace factorial_ratio_eq_zero_l389_389162

theorem factorial_ratio_eq_zero :
  (∏ i in Finset.range 10, i) / (Nat.factorial (Finset.sum (Finset.range 10))) = 0 := by
  sorry

end factorial_ratio_eq_zero_l389_389162


namespace count_valid_3_digit_numbers_l389_389381

def valid_units_digit (tens_digit units_digit : ℕ) : Prop :=
  units_digit ≥ 3 * tens_digit

def count_valid_numbers : ℕ :=
  (∑ tens_digit in {0, 1, 2, 3}, (finset.filter (valid_units_digit tens_digit) (finset.range 10)).card) * 9

theorem count_valid_3_digit_numbers : count_valid_numbers = 198 :=
by
  sorry

end count_valid_3_digit_numbers_l389_389381


namespace max_marks_eq_300_l389_389892

-- Problem Statement in Lean 4

theorem max_marks_eq_300 (m_score p_score c_score : ℝ) 
    (m_percent p_percent c_percent : ℝ)
    (h1 : m_score = 285) (h2 : m_percent = 95) 
    (h3 : p_score = 270) (h4 : p_percent = 90) 
    (h5 : c_score = 255) (h6 : c_percent = 85) :
    (m_score / (m_percent / 100) = 300) ∧ 
    (p_score / (p_percent / 100) = 300) ∧ 
    (c_score / (c_percent / 100) = 300) :=
by
  sorry

end max_marks_eq_300_l389_389892


namespace maximize_angle_points_l389_389493

-- Assume some definitions for points, circles, etc.
variable {Point : Type} [inhabited Point]
variable (O A : Point)
variable (ω : circle O)

-- Definition of line through two points
def line_through (P Q : Point) : set Point := sorry

-- Definition of points that maximize the angle at OPA
def maximizing_points (O A : Point) (ω : circle O) : set Point := sorry

theorem maximize_angle_points : 
  maximizing_points O A ω = {P | P ∈ ω ∧ P ∈ line_through O A} :=
sorry

end maximize_angle_points_l389_389493


namespace squats_on_day_4_l389_389689

def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

theorem squats_on_day_4 : 
  ∀ (initial_squats : ℕ), (initial_squats = 30) →
  (initial_squats + factorial 1 + factorial 2 + factorial 3 = 39) :=
by
  intros initial_squats h_init
  rw h_init
  have h1 : factorial 1 = 1 := rfl
  have h2 : factorial 2 = 2 := rfl
  have h3 : factorial 3 = 6 := rfl
  simp [h1, h2, h3]
  sorry

end squats_on_day_4_l389_389689


namespace mary_initial_sheep_l389_389514

variable (M : ℕ) -- Representing the number of sheep Mary initially has

-- Conditions from the problem:
def condition1 := (2 * M + 35 : ℕ) -- Bob has double the number of sheep as Mary plus 35

def condition2 := (M + 266 = (2 * M + 35) - 69 : Prop) -- Mary must buy 266 sheep to have 69 fewer than Bob

-- The proof goal is to show that M = 300 given the conditions:
theorem mary_initial_sheep (h1 : condition1) (h2 : condition2) : M = 300 :=
sorry

end mary_initial_sheep_l389_389514


namespace magnitude_of_c_l389_389376

open Real

noncomputable def a := (1 : ℝ, 0 : ℝ)
noncomputable def b := (1 : ℝ, 2 : ℝ)

def projection (v w : ℝ × ℝ) : ℝ :=
  let dot_product := (v.1 * w.1 + v.2 * w.2)
  let magnitude_w := sqrt (w.1 * w.1 + w.2 * w.2)
  dot_product / magnitude_w

theorem magnitude_of_c (c : ℝ × ℝ)
  (h1 : c.1 * 1 + c.2 * 0 = 2)
  (h2 : ∃ k : ℝ, c = (k * 1, k * 2)) :
  let magnitude_c := sqrt (c.1 * c.1 + c.2 * c.2)
  magnitude_c = 2 * sqrt 5 :=
by
  sorry

end magnitude_of_c_l389_389376


namespace good_points_count_l389_389784

-- Define the vertices of the square OABC
def O := (0, 0)
def A := (100, 0)
def B := (100, 100)
def C := (0, 100)

-- Define what it means to be a "good point" P inside the square OABC excluding the boundary and vertices
def is_good_point (P : (ℕ × ℕ)) : Prop :=
  let x := P.1
  let y := P.2 in
  1 <= x ∧ x < 100 ∧ 1 <= y ∧ y < 100 ∧ y * (100 - y) = x * (100 - x)

-- State the theorem to be proved using the conditions provided
theorem good_points_count : 
  (finset.filter is_good_point (finset.product (finset.range 100) (finset.range 100))).card = 197 := 
sorry

end good_points_count_l389_389784


namespace difference_between_numbers_l389_389895

noncomputable def L : ℕ := 1614
noncomputable def Q : ℕ := 6
noncomputable def R : ℕ := 15

theorem difference_between_numbers (S : ℕ) (h : L = Q * S + R) : L - S = 1348 :=
by {
  -- proof skipped
  sorry
}

end difference_between_numbers_l389_389895


namespace m_range_l389_389752

noncomputable def f (x : ℝ) : ℝ :=
  Real.sqrt 3 * Real.sin (2018 * Real.pi - x) * Real.sin (3 * Real.pi / 2 + x) 
  - Real.cos x ^ 2 + 1

def valid_m (m : ℝ) : Prop := 
  ∀ x ∈ Set.Icc (-Real.pi / 12) (Real.pi / 2), abs (f x - m) ≤ 1

theorem m_range : 
  ∀ m : ℝ, valid_m m ↔ (m ∈ Set.Icc (1 / 2) ((3 - Real.sqrt 3) / 2)) :=
by sorry

end m_range_l389_389752


namespace reciprocal_of_repeating_decimal_l389_389974

theorem reciprocal_of_repeating_decimal :
  let x : ℚ := 36 / 99 in
  1 / x = 11 / 4 :=
by
  sorry

end reciprocal_of_repeating_decimal_l389_389974


namespace problem_statement_l389_389997

theorem problem_statement (a b : ℝ) (h : a ≠ b) : (a - b) ^ 2 > 0 := sorry

end problem_statement_l389_389997


namespace shaded_unshaded_area_ratio_l389_389894

-- Define the given conditions: arrangement of circles and their common radius
variables (r : ℝ) (C : ℝ → Type) [∀ r, TopologicalSpace (C r)] [MetricSpace (C r)] [NormedSpace ℝ (C r)]
variables (c₁ c₂ c₃ c₄ c₅ : C r) -- the five circles

-- Definition of the centers forming a square
def centers_form_square (c₁ c₂ c₃ c₄ : C r) : Prop :=
  dist c₁ c₂ = r * √2 ∧ dist c₂ c₃ = r * √2 ∧ dist c₃ c₄ = r * √2 ∧ dist c₄ c₁ = r * √2

-- Proof statement for the area ratio of shaded to unshaded parts
theorem shaded_unshaded_area_ratio
  (h1 : centers_form_square c₁ c₂ c₃ c₄) : (2 / 3 : ℝ) = 2 / 3 :=
by
  sorry

end shaded_unshaded_area_ratio_l389_389894


namespace percentage_discount_is_75_l389_389667

-- Conditions
def num_bags : ℕ := 2
def original_price_per_bag : ℝ := 6.0
def total_spent : ℝ := 3.0

-- Original total price
def original_total_price : ℝ := num_bags * original_price_per_bag

-- Total discount
def total_discount : ℝ := original_total_price - total_spent

-- Discount per bag
def discount_per_bag : ℝ := total_discount / num_bags

-- Percentage discount
def percentage_discount : ℝ := (discount_per_bag / original_price_per_bag) * 100

-- Proof statement
theorem percentage_discount_is_75 :
  percentage_discount = 75 := by
  sorry

end percentage_discount_is_75_l389_389667


namespace ellipse_focus_distance_l389_389724

theorem ellipse_focus_distance :
  ∀ {x y : ℝ},
    (x^2) / 25 + (y^2) / 16 = 1 →
    (dist (x, y) (3, 0) = 8) →
    dist (x, y) (-3, 0) = 2 :=
by
  intro x y h₁ h₂
  sorry

end ellipse_focus_distance_l389_389724


namespace roots_quadratic_square_diff_10_l389_389356

-- Definition and theorem statement in Lean 4
theorem roots_quadratic_square_diff_10 :
  ∀ x1 x2 : ℝ, (2 * x1^2 + 4 * x1 - 3 = 0) ∧ (2 * x2^2 + 4 * x2 - 3 = 0) →
  (x1 - x2)^2 = 10 :=
by
  sorry

end roots_quadratic_square_diff_10_l389_389356


namespace smallest_n_eq_180_l389_389862

noncomputable def alpha : ℝ := Real.pi / 36
noncomputable def beta : ℝ := Real.pi / 30
noncomputable def L_angle : ℝ := Real.arctan (14 / 75)

-- Condition: Transformation T reflects line L in L_1 and then in L_2
def T (θ : ℝ) : ℝ := θ + (beta - alpha)

-- Define the nth transformation T^(n)
def T_n (θ : ℝ) (n : ℕ) : ℝ := θ + n * (beta - alpha)

-- Main Statement: proving the smallest n such that T^n(L) = L
theorem smallest_n_eq_180 :
  ∃ n : ℕ, n > 0 ∧ T_n L_angle n = L_angle ∧ ∀ m : ℕ, m > 0 → T_n L_angle m = L_angle → m ≥ 180 :=
begin
  sorry -- Proof not required
end

end smallest_n_eq_180_l389_389862


namespace complement_union_covers_until_1_l389_389175

open Set

noncomputable def S := {x : ℝ | x > -2}
noncomputable def T := {x : ℝ | x^2 + 3*x - 4 ≤ 0}
noncomputable def complement_R_S := {x : ℝ | x ≤ -2}
noncomputable def union := complement_R_S ∪ T

theorem complement_union_covers_until_1 : union = {x : ℝ | x ≤ 1} := by
  sorry

end complement_union_covers_until_1_l389_389175


namespace shift_sin_to_cos_l389_389208

/-- Shifting the graph of y = sin(2x) right by φ units results in the graph of y = cos(2x + π/6). Prove the value of φ. -/
theorem shift_sin_to_cos : ∃ φ : ℝ, 
  (∀ x : ℝ, sin (2 * (x - φ)) = cos (2 * x + π / 6)) → φ = 2 * π / 3 :=
begin
  sorry
end

end shift_sin_to_cos_l389_389208


namespace sum_modulo_five_remainder_is_one_l389_389585

theorem sum_modulo_five_remainder_is_one : 
  (finset.sum (finset.range 124) id) % 5 = 1 := 
begin
  sorry
end

end sum_modulo_five_remainder_is_one_l389_389585


namespace interval_of_monotonicity_solutions_range_exists_m_area_l389_389070

section problem

def f (x : ℝ) : ℝ := x - (x + 1) * Real.log (x + 1)

theorem interval_of_monotonicity :
  (∀ x, -1 < x ∧ x < 0 → f' x > 0) ∧
  (∀ x, 0 < x → f' x < 0) := 
sorry

theorem solutions_range (t : ℝ) :
  (∀ a b ∈ Icc (-1/2 : ℝ) (1 : ℝ), f a = t ∧ f b = t → a ≠ b) →
  t ∈ Ioc (-1/2 + 1/2 * Real.log 2) 0 :=
sorry

theorem exists_m_area :
  ∃ m ∈ Icc (0 : ℝ) (1/2 : ℝ), 
    ∫ (x in -Real.log 6..Real.log (2/3)), (Real.exp x - 1/6) dx +
    ∫ (x in Real.log (2/3)..0), (Real.exp (-x) - 1) dx = 1 + 2/3 * Real.log 2 - Real.log 3 ∧ m = 0 :=
sorry

end problem

end interval_of_monotonicity_solutions_range_exists_m_area_l389_389070


namespace contains_zero_l389_389241

theorem contains_zero (a b c d e f: ℕ) (h1: 1 ≤ a ∧ a ≤ 9) (h2: 1 ≤ b ∧ b ≤ 9) (h3: 0 ≤ c ∧ c ≤ 9) 
  (h4: 1 ≤ d ∧ d ≤ 9) (h5: 0 ≤ e ∧ e ≤ 9) (h6: 0 ≤ f ∧ f ≤ 9) 
  (h7: c ≠ f) (h8: 10^4*a + 10^3*b + 10^2*c + 10^1*d + e + 10^4*a + 10^3*b + 10^2*f + 10^1*d + e = 111111) :
  c = 0 ∨ f = 0 := 
sorry

end contains_zero_l389_389241


namespace complex_exponentiation_l389_389671

noncomputable def cos30 : ℂ := complex.cos (real.pi / 6)
noncomputable def sin30 : ℂ := complex.sin (real.pi / 6)
noncomputable def cos240 : ℂ := complex.cos (4 * real.pi / 3)
noncomputable def sin240 : ℂ := complex.sin (4 * real.pi / 3)

theorem complex_exponentiation :
  (3 * (cos30 + complex.I * sin30))^8 = -3281 - 3281 * complex.I * real.sqrt 3 :=
by sorry

end complex_exponentiation_l389_389671


namespace contains_zero_l389_389251

open Nat

theorem contains_zero 
  (x y : ℕ)
  (hx : 10000 ≤ x ∧ x < 100000)
  (hy : 10000 ≤ y ∧ y < 100000)
  (h_swap : ∃ (i j : ℕ), i ≠ j ∧ swap_digits x i j = y)
  (h_sum : x + y = 111111) :
  (∃ i, digit x i = 0) ∨ (∃ j, digit y j = 0) :=
sorry

-- Swap two digits in a number
def swap_digits (n : ℕ) (i j : ℕ) : ℕ := 
  -- Dummy definition for the proof placeholder
  n -- replace with actual implementation

-- Extract the digit at position i
def digit (n : ℕ) (i : ℕ) : ℕ :=
  -- Dummy definition for the proof placeholder
  0 -- replace with actual implementation

end contains_zero_l389_389251


namespace skew_lines_in_hexagonal_pyramid_l389_389451

-- Define the concept of a hexagonal pyramid with 12 edges
def hexagonal_pyramid.edges : Finset (Fin 12) := Finset.finRange 12

-- Define a function that checks if two edges are skew
def are_skew (e1 e2 : Fin 12) : Prop :=
  -- Implement the conditions for skewness, assuming pyramid structure
  sorry

-- Define the set of pairs of edges
def pairs_of_edges : Finset (Fin 12 × Fin 12) :=
  Finset.univ.filter (λ p, are_skew p.1 p.2)

-- The theorem to be proved
theorem skew_lines_in_hexagonal_pyramid : pairs_of_edges.card = 24 :=
  sorry

end skew_lines_in_hexagonal_pyramid_l389_389451


namespace side_length_of_equilateral_triangle_l389_389524

noncomputable def point_on_parabola (x : ℝ) : ℝ × ℝ :=
(x, -2 * x^2)

theorem side_length_of_equilateral_triangle
  (x : ℝ)
  (hx : x ≠ 0)
  (P := point_on_parabola x)
  (Q := point_on_parabola (-x)) :
  let O := (0, 0) in
  let d := λ A B : ℝ × ℝ, real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) in
  d O P = d O Q ∧ d O P = d P Q ∧ d O P = real.sqrt 3 :=
sorry

end side_length_of_equilateral_triangle_l389_389524


namespace complex_expression_proof_l389_389672

noncomputable def compute_expression : ℂ :=
  (3 * complex.cos (real.pi / 6) + 3 * complex.sin (real.pi / 6) * complex.I) ^ 8

theorem complex_expression_proof :
  compute_expression = -3280.5 - 3280.5 * real.sqrt 3 * complex.I :=
by
  sorry

end complex_expression_proof_l389_389672


namespace trig_limit_value_l389_389660

noncomputable def trig_limit (a b : ℝ) : ℝ :=
  limit (fun x => (sin (b * x) - sin (a * x)) / log (tan (π / 4 + a * x))) 0

theorem trig_limit_value (a b : ℝ) : trig_limit a b = (b - a) / (2 * a) :=
  sorry

end trig_limit_value_l389_389660


namespace largest_a_condition_l389_389861

def isGoodSet (A : Set ℕ) (X : Set ℕ) :=
  ∃ x y ∈ X, x < y ∧ x ∣ y

theorem largest_a_condition (A : Set ℕ) (hA : A = {1, 2, ..., 2016}) :
  ∃ a ∈ A, (∀ X ⊆ A, |X| = 1008 → a ∈ X → isGoodSet A X) ∧ a = 671 := 
sorry

end largest_a_condition_l389_389861


namespace calculate_interest_rate_l389_389214

theorem calculate_interest_rate
  (total_investment : ℝ)
  (invested_at_eleven_percent : ℝ)
  (total_interest : ℝ)
  (interest_rate_first_type : ℝ) :
  total_investment = 100000 ∧ 
  invested_at_eleven_percent = 30000 ∧ 
  total_interest = 9.6 → 
  interest_rate_first_type = 9 :=
by
  intros
  sorry

end calculate_interest_rate_l389_389214


namespace three_digit_numbers_count_l389_389428

theorem three_digit_numbers_count :
  let t_range := finset.range 10
  let h_range := finset.range 9
  let valid_u t := if t * 3 ≤ 9 then 9 - (t * 3) + 1 else 0
  let total_count := h_range.card * t_range.sum (λ t, valid_u t)
  total_count = 198 :=
by
  sorry

end three_digit_numbers_count_l389_389428


namespace sqrt_sub_sqrt_plus_sqrt_eq_zero_div_sqrt_diff_eq_one_sqrt_frac_add_sqrt_sub_sqrt_eq_neg_frac_sqrt_sqrt_sum_mul_sqrt_diff_eq_two_l389_389656

-- Theorem (1)
theorem sqrt_sub_sqrt_plus_sqrt_eq_zero : sqrt 18 - sqrt 32 + sqrt 2 = 0 := 
sorry

-- Theorem (2)
theorem div_sqrt_diff_eq_one : (sqrt 27 - sqrt 12) / sqrt 3 = 1 := 
sorry

-- Theorem (3)
theorem sqrt_frac_add_sqrt_sub_sqrt_eq_neg_frac_sqrt :
  sqrt (1 / 6) + sqrt 24 - sqrt 600 = - (43 / 6) * sqrt 6 := 
sorry

-- Theorem (4)
theorem sqrt_sum_mul_sqrt_diff_eq_two : (sqrt 3 + 1) * (sqrt 3 - 1) = 2 := 
sorry

end sqrt_sub_sqrt_plus_sqrt_eq_zero_div_sqrt_diff_eq_one_sqrt_frac_add_sqrt_sub_sqrt_eq_neg_frac_sqrt_sqrt_sum_mul_sqrt_diff_eq_two_l389_389656


namespace cube_volume_l389_389189

theorem cube_volume
  (s : ℝ) 
  (surface_area_eq : 6 * s^2 = 54) :
  s^3 = 27 := 
by 
  sorry

end cube_volume_l389_389189


namespace contains_zero_l389_389243

theorem contains_zero (a b c d e f: ℕ) (h1: 1 ≤ a ∧ a ≤ 9) (h2: 1 ≤ b ∧ b ≤ 9) (h3: 0 ≤ c ∧ c ≤ 9) 
  (h4: 1 ≤ d ∧ d ≤ 9) (h5: 0 ≤ e ∧ e ≤ 9) (h6: 0 ≤ f ∧ f ≤ 9) 
  (h7: c ≠ f) (h8: 10^4*a + 10^3*b + 10^2*c + 10^1*d + e + 10^4*a + 10^3*b + 10^2*f + 10^1*d + e = 111111) :
  c = 0 ∨ f = 0 := 
sorry

end contains_zero_l389_389243


namespace range_of_m_l389_389681

def custom_op (x y : ℝ) : ℝ := if x ≤ y then x else y

theorem range_of_m (m : ℝ) (h : custom_op (|m + 1|) (|m|) = |m + 1|) : m ≤ -1 / 2 :=
by
  sorry

end range_of_m_l389_389681


namespace reciprocal_of_repeating_decimal_l389_389985

theorem reciprocal_of_repeating_decimal :
  let x := (36 : ℚ) / 99 in 1 / x = 11 / 4 :=
by
  trivial -- This is a placeholder, the actual proof is not required and hence replaced by trivial.

end reciprocal_of_repeating_decimal_l389_389985


namespace total_earnings_correct_l389_389170

noncomputable def total_earnings (a_days b_days c_days b_share : ℝ) : ℝ :=
  let a_work_per_day := 1 / a_days
  let b_work_per_day := 1 / b_days
  let c_work_per_day := 1 / c_days
  let combined_work_per_day := a_work_per_day + b_work_per_day + c_work_per_day
  let b_fraction_of_total_work := b_work_per_day / combined_work_per_day
  let total_earnings := b_share / b_fraction_of_total_work
  total_earnings

theorem total_earnings_correct :
  total_earnings 6 8 12 780.0000000000001 = 2340 :=
by
  sorry

end total_earnings_correct_l389_389170


namespace solution_set_l389_389741

noncomputable def f : ℝ → ℝ := sorry -- The even function f
noncomputable def f' : ℝ → ℝ := sorry -- The derivative f'

axiom f_even : ∀ x, f x = f (-x)
axiom domain : ∀ x, x ∈ (-∞, 0) ∪ (0, +∞)
axiom f_deriv_exists : ∀ x, f' x = derivative f x
axiom inequality_condition : ∀ x, 2 * f x + x * f' x > 0
axiom f_at_2 : f 2 = 1

theorem solution_set : { x : ℝ | x^2 * f x < 4 } = { x : ℝ | -2 < x ∧ x < 0 ∨ 0 < x ∧ x < 2 } :=
sorry

end solution_set_l389_389741


namespace probability_of_winning_more_than_12000_l389_389517

-- Define the conditions
def boxes : List ℕ := [100, 500, 1000, 5000, 10000]

def num_keys : ℕ := 5

def win_amount_more_than : ℕ := 12000

-- Define the correct answer
def probability : ℕ := 1 / 20

-- Theorem statement
theorem probability_of_winning_more_than_12000 :
  ∃ (k1 k2 : Fin 5), boxes[Fin.val k1] = 5000 ∧ boxes[Fin.val k2] = 10000 →
  finset.card (Finset.filter (λ k, boxes[Fin.val k] > win_amount_more_than - 15000) (Finset.range num_keys)) = 3 →
  Real.weighing_of_lists (list.range num_keys) = probability := 
sorry

end probability_of_winning_more_than_12000_l389_389517


namespace people_in_room_l389_389702

/-- 
   Problem: Five-sixths of the people in a room are seated in five-sixths of the chairs.
   The rest of the people are standing. If there are 10 empty chairs, 
   prove that there are 60 people in the room.
-/
theorem people_in_room (people chairs : ℕ) 
  (h_condition1 : 5 / 6 * people = 5 / 6 * chairs) 
  (h_condition2 : chairs = 60) :
  people = 60 :=
by
  sorry

end people_in_room_l389_389702


namespace S_8_arithmetic_sequence_l389_389332

theorem S_8_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n, S n = (n * (a 1 + a n)) / 2)
  (h2 : a 4 = 18 - a 5):
  S 8 = 72 :=
by
  sorry

end S_8_arithmetic_sequence_l389_389332


namespace snail_distance_round_100_l389_389593

def snail_distance (n : ℕ) : ℕ :=
  if n = 0 then 100 else (100 * (n + 2)) / (n + 1)

theorem snail_distance_round_100 : snail_distance 100 = 5050 :=
  sorry

end snail_distance_round_100_l389_389593


namespace total_cost_is_correct_l389_389016

noncomputable def service_cost : ℚ := 2.20
noncomputable def mini_van_fuel_cost_per_liter : ℚ := 0.70
noncomputable def pickup_truck_fuel_cost_per_liter : ℚ := 0.85
noncomputable def semi_truck_fuel_cost_per_liter : ℚ := 0.95

noncomputable def mini_van_tank_capacity : ℚ := 65
noncomputable def pickup_truck_tank_capacity : ℚ := 100
noncomputable def semi_truck_tank_capacity : ℚ := 220

noncomputable def mini_van_total_cost : ℚ :=
  (service_cost + mini_van_fuel_cost_per_liter * mini_van_tank_capacity) * 4

noncomputable def pickup_truck_total_cost : ℚ :=
  (service_cost + pickup_truck_fuel_cost_per_liter * pickup_truck_tank_capacity) * 2

noncomputable def semi_truck_total_cost : ℚ :=
  (service_cost + semi_truck_fuel_cost_per_liter * semi_truck_tank_capacity) * 3

noncomputable def total_cost : ℚ :=
  mini_van_total_cost + pickup_truck_total_cost + semi_truck_total_cost

theorem total_cost_is_correct : total_cost = 998.80 := by
  unfold total_cost mini_van_total_cost pickup_truck_total_cost semi_truck_total_cost
  unfold service_cost mini_van_fuel_cost_per_liter mini_van_tank_capacity
  unfold pickup_truck_fuel_cost_per_liter pickup_truck_tank_capacity semi_truck_fuel_cost_per_liter semi_truck_tank_capacity
  norm_num
  sorry

end total_cost_is_correct_l389_389016


namespace intersection_M_N_l389_389858

def f (x : ℝ) : ℝ := x^2 - 4 * x + 3

def g (x : ℝ) : ℝ := 3^x - 2

def M : Set ℝ := {x | f (g x) > 0}

def N : Set ℝ := {x | g x < 2}

theorem intersection_M_N : M ∩ N = {x | x < 1} :=
by
  sorry

end intersection_M_N_l389_389858


namespace minimum_value_ineq_l389_389843

theorem minimum_value_ineq (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_sum : a + b + c = 1) :
    (1 / (a + 2 * b)) + (1 / (b + 2 * c)) + (1 / (c + 2 * a)) ≥ 3 := 
by
  sorry

end minimum_value_ineq_l389_389843


namespace log_3100_nearest_int_l389_389939

theorem log_3100_nearest_int :
  ∀ (log : ℝ → ℝ), 
    (∀ x, strict_mono log) → 
    log 3125 = 5 → 
    log 625 = 4 → 
    625 < 3100 ∧ 3100 < 3125 → 
    Nat.round (log 3100) = 5 :=
by
  intros log h_mono h1 h2 h3
  sorry

end log_3100_nearest_int_l389_389939


namespace emma_uniform_number_correct_l389_389711

def is_two_digit_prime (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ Prime n

noncomputable def dan : ℕ := 11  -- Example value, but needs to satisfy all conditions
noncomputable def emma : ℕ := 19  -- This is what we need to prove
noncomputable def fiona : ℕ := 13  -- Example value, but needs to satisfy all conditions
noncomputable def george : ℕ := 11  -- Example value, but needs to satisfy all conditions

theorem emma_uniform_number_correct :
  is_two_digit_prime dan ∧
  is_two_digit_prime emma ∧
  is_two_digit_prime fiona ∧
  is_two_digit_prime george ∧
  dan ≠ emma ∧ dan ≠ fiona ∧ dan ≠ george ∧
  emma ≠ fiona ∧ emma ≠ george ∧
  fiona ≠ george ∧
  dan + fiona = 23 ∧
  george + emma = 9 ∧
  dan + fiona + george + emma = 32
  → emma = 19 :=
sorry

end emma_uniform_number_correct_l389_389711


namespace problem_proof_l389_389496

variables {α : Type*} [plane α]
variables (a b : line α) (l : line (↥(complement α)))

-- Assume conditions
variables (h_diff: a ≠ b)
variables (h_perp_a : l.perpendicular a)
variables (h_perp_b : l.perpendicular b)

-- The statement we need to prove
def necessary_but_not_sufficient_for_perpendicular_plane : Prop :=
  (∀ l : line (↥(complement α)), (l.perpendicular α) → (l.perpendicular a ∧ l.perpendicular b)) ∧ 
  ¬(∀ l : line (↥(complement α)), (l.perpendicular a ∧ l.perpendicular b) → (l.perpendicular α))

theorem problem_proof :
  necessary_but_not_sufficient_for_perpendicular_plane a b l h_diff h_perp_a h_perp_b :=
sorry

end problem_proof_l389_389496


namespace three_digit_numbers_count_l389_389436

def is_3_digit_number (n : Nat) : Prop :=
  100 ≤ n ∧ n ≤ 999

def units_digit (n : Nat) : Nat :=
  n % 10

def tens_digit (n : Nat) : Nat :=
  (n / 10) % 10

def hundreds_digit (n : Nat) : Nat :=
  (n / 100) % 10

def valid_number (n : Nat) : Prop :=
  is_3_digit_number n ∧ units_digit n ≥ 3 * tens_digit n

theorem three_digit_numbers_count : 
  (Finset.card (Finset.filter valid_number (Finset.range 1000))) = 198 :=
by
  sorry

end three_digit_numbers_count_l389_389436


namespace positive_difference_of_diagonals_is_zero_l389_389271

def original_matrix : Matrix (Fin 4) (Fin 4) ℕ := 
  ![
    ![5, 6, 7, 8],
    ![12, 13, 14, 15],
    ![19, 20, 21, 22],
    ![26, 27, 28, 29]
  ]

def reversed_matrix : Matrix (Fin 4) (Fin 4) ℕ := 
  ![
    ![26, 27, 28, 29],
    ![19, 20, 21, 22],
    ![12, 13, 14, 15],
    ![5, 6, 7, 8]
  ]

def main_diagonal_sum (m : Matrix (Fin 4) (Fin 4) ℕ) : ℕ :=
  ∑ i, m i i

def anti_diagonal_sum (m : Matrix (Fin 4) (Fin 4) ℕ) : ℕ :=
  ∑ i, m i (3 - i)

theorem positive_difference_of_diagonals_is_zero :
  abs (main_diagonal_sum reversed_matrix - anti_diagonal_sum reversed_matrix) = 0 :=
by
  sorry

end positive_difference_of_diagonals_is_zero_l389_389271


namespace segment_bisected_by_perpendicular_l389_389017

variables {A B C D E A' : Point} {O : Circle}

-- Given conditions
axiom (right_triangle_ABC : ∠ A B C = 90)
axiom (circumcircle_O : Circumcircle O (Triangle A B C))
axiom (diameter_B : Diameter O B)
axiom (diametrically_opposite_A : A' ∈ O ∧ A' ≠ A)
axiom (segment_DE_intersects_BC : E ∈ Segment B C ∧ D ∈ Segment B C)

-- Proof goal
theorem segment_bisected_by_perpendicular (h1 : B ∈ O) (h2 : C ∈ O) :
  LineSegment A' (Midpoint D E) ∈ Line A' B :=
sorry

end segment_bisected_by_perpendicular_l389_389017


namespace find_least_multiple_of_50_l389_389953

def digits (n : ℕ) : List ℕ := n.digits 10

def product_of_digits (n : ℕ) : ℕ := (digits n).prod

theorem find_least_multiple_of_50 :
  ∃ n, (n % 50 = 0) ∧ ((product_of_digits n) % 50 = 0) ∧ (∀ m, (m % 50 = 0) ∧ ((product_of_digits m) % 50 = 0) → n ≤ m) ↔ n = 5550 :=
by sorry

end find_least_multiple_of_50_l389_389953


namespace value_of_y_l389_389601

theorem value_of_y : exists y : ℝ, (∀ k : ℝ, (∀ x y : ℝ, x = k / y^2 → (x = 1 → y = 2 → k = 4)) ∧ (x = 0.1111111111111111 → k = 4 → y = 6)) := by
  sorry

end value_of_y_l389_389601


namespace contains_zero_if_sum_is_111111_l389_389222

variables (x y : ℕ)
variables (digits_x digits_y : Fin 5 → ℕ)

-- Conditions
def is_five_digit (n : ℕ) : Prop := 
  10000 ≤ n ∧ n < 100000

def differs_by_two_digit_swap (dx dy : Fin 5 → ℕ) : Prop :=
  ∃ i j : Fin 5, i ≠ j ∧ 
  (∀ k : Fin 5, (k = i → dy k = dx j) ∧ 
                (k = j → dy k = dx i) ∧ 
                (k ≠ i ∧ k ≠ j → dy k = dx k))

theorem contains_zero_if_sum_is_111111 
  (hx : is_five_digit x) (hy : is_five_digit y)
  (h_digits_x : digits_x = fun i : Fin 5 => (x / 10^(4-i) % 10))
  (h_digits_y : digits_y = fun i : Fin 5 => (y / 10^(4-i) % 10))
  (h_swap : differs_by_two_digit_swap digits_x digits_y)
  (h_sum : x + y = 111111) :
  ∃ i : Fin 5, digits_x i = 0 :=
begin
  -- Proof omitted
  sorry,
end

end contains_zero_if_sum_is_111111_l389_389222


namespace count_valid_numbers_l389_389404

def is_valid_number (n : Nat) : Prop := 
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let units := n % 10
  1 ≤ hundreds ∧ hundreds ≤ 9 ∧
  0 ≤ tens ∧ tens ≤ 9 ∧
  units ≥ 3 * tens

theorem count_valid_numbers : 
  (Finset.filter is_valid_number (Finset.range 1000)).card = 198 :=
by sorry

end count_valid_numbers_l389_389404


namespace net_emails_received_l389_389035

-- Define the conditions
def emails_received_morning : ℕ := 3
def emails_sent_morning : ℕ := 2
def emails_received_afternoon : ℕ := 5
def emails_sent_afternoon : ℕ := 1

-- Define the problem statement
theorem net_emails_received :
  emails_received_morning - emails_sent_morning + emails_received_afternoon - emails_sent_afternoon = 5 := by
  sorry

end net_emails_received_l389_389035


namespace smallest_b_problem_l389_389091

noncomputable def smallest_b : ℝ :=
  let b := (7 + Real.sqrt 17) / 4 in b

theorem smallest_b_problem (a b : ℝ) (h1 : 2 < a) (h2 : a < b)
  (h3 : 2 + a ≤ b) (h4 : (1 / a) + (1 / b) ≤ 1 / 2) :
  b = smallest_b :=
begin
  sorry,
end

end smallest_b_problem_l389_389091


namespace museum_admission_ratio_l389_389547

theorem museum_admission_ratio (a c : ℕ) (h1 : 30 * a + 15 * c = 2700) (h2 : 2 ≤ a) (h3 : 2 ≤ c) :
  a / (180 - 2 * a) = 2 :=
by
  sorry

end museum_admission_ratio_l389_389547


namespace couplet_distribution_l389_389650

theorem couplet_distribution :
  let long_couplets : ℕ := 4 in
  let short_couplets : ℕ := 7 in
  let total_couplets : ℕ := 11 in
  let ways_to_distribute := (Nat.choose 4 1) * (Nat.choose 7 2) *
                            (Nat.choose 3 1) * (Nat.choose 5 1) *
                            (Nat.choose 2 1) * (Nat.choose 4 2) *
                            (Nat.choose 1 1) * (Nat.choose 2 2) in
  total_couplets = long_couplets + short_couplets ∧
  ways_to_distribute = 15120 := sorry

end couplet_distribution_l389_389650


namespace problem_1_problem_2_l389_389538

theorem problem_1 (x : ℝ) : (2 * x + 3)^2 = 16 ↔ x = 1/2 ∨ x = -7/2 := by
  sorry

theorem problem_2 (x : ℝ) : x^2 - 4 * x - 3 = 0 ↔ x = 2 + Real.sqrt 7 ∨ x = 2 - Real.sqrt 7 := by
  sorry

end problem_1_problem_2_l389_389538


namespace eight_pyramids_touch_by_faces_l389_389662

theorem eight_pyramids_touch_by_faces :
  ∃ (pyramids : list ℝ), 
  (∀ (i j : ℝ), i ≠ j → ∃ (face : ℝ), face ∈ pyramids i ∧ face ∈ pyramids j) →
  true := 
begin
  sorry
end

end eight_pyramids_touch_by_faces_l389_389662


namespace log_3100_nearest_int_l389_389938

theorem log_3100_nearest_int :
  ∀ (log : ℝ → ℝ), 
    (∀ x, strict_mono log) → 
    log 3125 = 5 → 
    log 625 = 4 → 
    625 < 3100 ∧ 3100 < 3125 → 
    Nat.round (log 3100) = 5 :=
by
  intros log h_mono h1 h2 h3
  sorry

end log_3100_nearest_int_l389_389938


namespace area_ratio_ge_four_l389_389088

theorem area_ratio_ge_four
  (A B C M N : Point)
  (h₁ : on_line_segment M A B)
  (h₂ : on_line_segment N A C)
  (h₃ : AM = CN)
  (h₄ : AN = BM) :
  let S (X Y Z : Point) := (1 / 2) * abs (det (Y - X) (Z - X))
  S (triangle A B C) / S (triangle A M N) ≥ 4 := 
sorry

end area_ratio_ge_four_l389_389088


namespace count_different_numerators_l389_389832

def hasRepeatingDecimalForm (r : ℚ) : Prop :=
  ∃ a b : ℕ, 0 ≤ a ∧ a < 10 ∧ 0 ≤ b ∧ b < 10 ∧
    r = a * (1 / 100 + 1 / 100^2 + 1 / 100^3 + ...) + b * (1 / 10 + 1 / 10^3 + 1 / 10^5 + ...)

def setT : set ℚ := {r | r > 0 ∧ r < 1 ∧ hasRepeatingDecimalForm r}

theorem count_different_numerators : ∃ n : ℕ, n = 60 ∧
  ∀ r ∈ setT, r = (numerator r) / (99 * gcd (numerator r) 99) :=
sorry

end count_different_numerators_l389_389832


namespace factorize_expression_l389_389288

theorem factorize_expression (a b : ℝ) :
  ab^(3 : ℕ) - 4 * ab = ab * (b + 2) * (b - 2) :=
by
  -- proof to be provided
  sorry

end factorize_expression_l389_389288


namespace dana_hours_sunday_l389_389679

-- Define the constants given in the problem
def hourly_rate : ℝ := 13
def hours_worked_friday : ℝ := 9
def hours_worked_saturday : ℝ := 10
def total_earnings : ℝ := 286

-- Define the function to compute total earnings from worked hours and hourly rate
def earnings (hours : ℝ) (rate : ℝ) : ℝ := hours * rate

-- Define the proof problem to show the number of hours worked on Sunday
theorem dana_hours_sunday (hours_sunday : ℝ) :
  earnings hours_worked_friday hourly_rate
  + earnings hours_worked_saturday hourly_rate
  + earnings hours_sunday hourly_rate = total_earnings ->
  hours_sunday = 3 :=
by
  sorry -- proof to be filled in

end dana_hours_sunday_l389_389679


namespace represent_1917_as_sum_diff_of_squares_l389_389529

theorem represent_1917_as_sum_diff_of_squares : ∃ a b c : ℤ, 1917 = a^2 - b^2 + c^2 :=
by
  use 480, 478, 1
  sorry

end represent_1917_as_sum_diff_of_squares_l389_389529


namespace three_digit_number_units_digit_condition_l389_389386

theorem three_digit_number_units_digit_condition :
  let valid_numbers := (λ (n : ℕ), (100 ≤ n ∧ n ≤ 999) ∧ ((n % 10) ≥ 3 * ((n / 10) % 10))) in
  (Finset.filter valid_numbers (Finset.range 900)) = 198 :=
by
  sorry

end three_digit_number_units_digit_condition_l389_389386


namespace geometric_sequence_proof_l389_389342

noncomputable def geom_seq (a1 q : ℝ) (n : ℕ) : ℝ := a1 * q^n

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ a1 q, ∀ n, a n = geom_seq a1 q n

def is_arithmetic_sequence (a b c : ℝ) : Prop :=
  2 * b = a + c

def sum_of_first_n_terms (a : ℕ → ℝ) : ℕ → ℝ
| 0     := 0
| (n+1) := (a n) + (sum_of_first_n_terms a n)

def q_values (a : ℕ → ℝ) (m : ℕ) (q : ℝ) : Prop := 
  q = 1 ∨ q = -1/2

def arithmetic_sum_check (a : ℕ → ℝ) (m : ℕ) (q : ℝ) : Prop :=
  q = -1/2 → 
  is_arithmetic_sequence (sum_of_first_n_terms a m)
                         (sum_of_first_n_terms a (m + 1))
                         (sum_of_first_n_terms a (m + 2))

theorem geometric_sequence_proof (a : ℕ → ℝ) (m : ℕ) :
  is_geometric_sequence a →
  is_arithmetic_sequence (a m) (a (m + 1)) (a (m + 2)) →
  ∃ q , q_values a m q ∧ arithmetic_sum_check a m q :=
by
  intro h_geom h_arith
  sorry

end geometric_sequence_proof_l389_389342


namespace sin_gamma_isosceles_l389_389462

theorem sin_gamma_isosceles (a c m_a m_c s_1 s_2 : ℝ) (γ : ℝ) 
  (h1 : a + m_c = s_1) (h2 : c + m_a = s_2) :
  Real.sin γ = (s_2 / (2 * s_1)) * Real.sqrt ((4 * s_1^2) - s_2^2) :=
sorry

end sin_gamma_isosceles_l389_389462


namespace ratio_of_longer_side_to_square_l389_389632

theorem ratio_of_longer_side_to_square (s a b : ℝ) (h1 : a * b = 2 * s^2) (h2 : a = 2 * b) : a / s = 2 :=
by
  sorry

end ratio_of_longer_side_to_square_l389_389632


namespace largest_average_l389_389149

def average_multiples (k n : ℕ) : ℚ :=
  let a1 := k
  let an := n / k * k
  (a1 + an) / 2

theorem largest_average :
  let avg_11 := average_multiples 11 100810 
  let avg_13 := average_multiples 13 100810
  let avg_17 := average_multiples 17 100810
  let avg_19 := average_multiples 19 100810
  avg_17 = 50413.5 ∧ avg_17 > avg_11 ∧ avg_17 > avg_13 ∧ avg_17 > avg_19 :=
by {
  let avg_11 := average_multiples 11 100810 
  let avg_13 := average_multiples 13 100810
  let avg_17 := average_multiples 17 100810
  let avg_19 := average_multiples 19 100810
  simp [average_multiples] at *,
  -- Add the necessary steps to verify the solution
  -- but proof is not required as per instructions.
  sorry
}

end largest_average_l389_389149


namespace contains_zero_l389_389247

-- Define a five-digit number in terms of its digits.
def five_digit_number (A B C D E : ℕ) : ℕ :=
  10^4 * A + 10^3 * B + 10^2 * C + 10 * D + E

-- Define the conditions:
noncomputable def switch_two_digits (A B C D E F : ℕ) : Prop :=
  (10^4 * A + 10^3 * B + 10^2 * C + 10 * D + E ≠ 10^4 * A + 10^3 * B + 10^2 * F + 10 * D + E)

-- Lean statement for the proof problem.
theorem contains_zero (A B C D E F : ℕ) :
  let ABCDE := five_digit_number A B C D E,
      ABFDE := five_digit_number A B F D E in
  switch_two_digits A B C D E F ∧ (ABCDE + ABFDE = 111111) → (A = 0 ∨ B = 0 ∨ C = 0 ∨ D = 0 ∨ E = 0) :=
by
  -- This is where the proof would go.
  sorry

end contains_zero_l389_389247


namespace reciprocal_of_repeating_decimal_l389_389981

theorem reciprocal_of_repeating_decimal :
  let x := (36 : ℚ) / 99 in
  x⁻¹ = 11 / 4 :=
by
  have h_simplify : x = 4 / 11 := by sorry
  rw [h_simplify, inv_div]
  norm_num
  exact eq.refl (11 / 4)

end reciprocal_of_repeating_decimal_l389_389981


namespace absolute_difference_probability_l389_389456

-- Define the conditions
def num_red_marbles : ℕ := 1500
def num_black_marbles : ℕ := 2000
def total_marbles : ℕ := num_red_marbles + num_black_marbles

def P_s : ℚ :=
  let ways_to_choose_2_red := (num_red_marbles * (num_red_marbles - 1)) / 2
  let ways_to_choose_2_black := (num_black_marbles * (num_black_marbles - 1)) / 2
  let total_favorable_outcomes := ways_to_choose_2_red + ways_to_choose_2_black
  total_favorable_outcomes / (total_marbles * (total_marbles - 1) / 2)

def P_d : ℚ :=
  (num_red_marbles * num_black_marbles) / (total_marbles * (total_marbles - 1) / 2)

-- Prove the statement
theorem absolute_difference_probability : |P_s - P_d| = 1 / 50 := by
  sorry

end absolute_difference_probability_l389_389456


namespace contains_zero_l389_389248

-- Define a five-digit number in terms of its digits.
def five_digit_number (A B C D E : ℕ) : ℕ :=
  10^4 * A + 10^3 * B + 10^2 * C + 10 * D + E

-- Define the conditions:
noncomputable def switch_two_digits (A B C D E F : ℕ) : Prop :=
  (10^4 * A + 10^3 * B + 10^2 * C + 10 * D + E ≠ 10^4 * A + 10^3 * B + 10^2 * F + 10 * D + E)

-- Lean statement for the proof problem.
theorem contains_zero (A B C D E F : ℕ) :
  let ABCDE := five_digit_number A B C D E,
      ABFDE := five_digit_number A B F D E in
  switch_two_digits A B C D E F ∧ (ABCDE + ABFDE = 111111) → (A = 0 ∨ B = 0 ∨ C = 0 ∨ D = 0 ∨ E = 0) :=
by
  -- This is where the proof would go.
  sorry

end contains_zero_l389_389248


namespace deer_households_eq_l389_389026

theorem deer_households_eq (x : ℕ) (h_deer_eq : 100 = x + ⅓ * x) : 
  (x + ⅓ * x = 100) :=
by
  sorry

end deer_households_eq_l389_389026


namespace mohan_cookies_l389_389867

theorem mohan_cookies :
  ∃ (a : ℕ), 
    (a % 6 = 5) ∧ 
    (a % 7 = 3) ∧ 
    (a % 9 = 7) ∧ 
    (a % 11 = 10) ∧ 
    (a = 1817) :=
sorry

end mohan_cookies_l389_389867


namespace sector_area_eq_13pi_l389_389793

theorem sector_area_eq_13pi
    (O A B C : Type)
    (r : ℝ)
    (θ : ℝ)
    (h1 : θ = 130)
    (h2 : r = 6) :
    (θ / 360) * (π * r^2) = 13 * π := by
  sorry

end sector_area_eq_13pi_l389_389793


namespace no_integer_a_divides_gx_l389_389805

theorem no_integer_a_divides_gx : ∀ (a : ℤ), ¬ (∃ p : ℤ[X], (X^2 + X + polynomial.C a) * p = X^10 + X^2 + 50) :=
by
  sorry -- proof skipped

end no_integer_a_divides_gx_l389_389805


namespace modular_inverse_sum_l389_389498

theorem modular_inverse_sum (b : ℤ) 
  (h : b ≡ (3^(-1 : ℤ) + 4^(-1 : ℤ) + 5^(-1 : ℤ))^(-1 : ℤ) [MOD 13]) :
  b ≡ 1 [MOD 13] :=
sorry

end modular_inverse_sum_l389_389498


namespace triangle_bisector_angle_AX_length_l389_389032

theorem triangle_bisector_angle_AX_length 
(A B C X : Type) 
[h1 : Segment AB X] 
(h2 : Bisector (Angle ACB) CX) 
(h3 : AC = 18) 
(h4 : BC = 36) 
(h5 : BX = 15) 
: AX = 7.5 := 
  sorry

end triangle_bisector_angle_AX_length_l389_389032


namespace bug_eighth_move_probability_bug_eighth_move_result_l389_389184

noncomputable def Q : ℕ → ℚ
| 0     := 1
| (n+1) := 1/3 * (1 - Q n)

theorem bug_eighth_move_probability : Q 8 = 547 / 2187 := by
  -- proof would go here
  sorry

theorem bug_eighth_move_result : 547 + 2187 = 2734 := by
  -- proof would go here
  sorry

end bug_eighth_move_probability_bug_eighth_move_result_l389_389184


namespace minimum_degree_monic_poly_l389_389542

noncomputable def minimum_degree (P : ℤ[X]) :=
  degree P

theorem minimum_degree_monic_poly (P : ℤ[X])
  (h_monic : monic P)
  (h_degree : ∀ r ∈ finset.range 2014, (P.eval r) % 2013 = 0 → r ∈ {r : ℕ | r % 3 = 0 ∧ r % 11 = 0 ∧ r % 61 = 0})
  (h_card : {r ∈ finset.range 2014 | (P.eval r) % 2013 = 0}.card = 1000) :
  minimum_degree P = 50 :=
sorry

end minimum_degree_monic_poly_l389_389542


namespace ratio_of_pictures_hung_horizontally_l389_389864

theorem ratio_of_pictures_hung_horizontally 
  (total_pictures : ℕ) 
  (pictures_vertically : ℕ) 
  (pictures_haphazardly : ℕ) 
  (pictures_horizontally : ℕ) 
  (total_pictures_eq : total_pictures = 30)
  (pictures_vertically_eq : pictures_vertically = 10)
  (pictures_haphazardly_eq : pictures_haphazardly = 5)
  (pictures_horizontally_eq : pictures_horizontally = total_pictures - pictures_vertically - pictures_haphazardly) :
  (pictures_horizontally / nat.gcd pictures_horizontally total_pictures) =
  (total_pictures / nat.gcd pictures_horizontally total_pictures) :=
by
  -- Proof to be filled in
  sorry

end ratio_of_pictures_hung_horizontally_l389_389864


namespace range_of_x_l389_389370

def p (x : ℝ) : Prop := x^2 + 2*x - 3 > 0
def q (x : ℝ) : Prop := 1 / (3 - x) > 1
def compound_proposition (x : ℝ) : Prop := p x ∧ ¬q x

theorem range_of_x (x : ℝ) (h : compound_proposition x) : x ∈ set.Ioo (-∞) (-3) ∪ set.Ioc 1 2 ∪ set.Icc 3 ∞ :=
sorry

end range_of_x_l389_389370


namespace digit_contains_zero_l389_389225

theorem digit_contains_zero 
  (n₁ n₂ : ℕ) 
  (h₁ : 10000 ≤ n₁ ∧ n₁ ≤ 99999)
  (h₂ : 10000 ≤ n₂ ∧ n₂ ≤ 99999)
  (h₃ : ∃ (i j : ℕ), i ≠ j ∧ n₂ = n₁.swap_digits i j)
  (h₄ : n₁ + n₂ = 111111) 
  : ∃ (d : ℕ), d = 0 :=
sorry

end digit_contains_zero_l389_389225


namespace reciprocal_of_repeating_decimal_l389_389968

noncomputable def repeating_decimal_to_fraction (x : ℚ) : ℚ :=
  if x = .\overline{36} then 4/11 else 0

theorem reciprocal_of_repeating_decimal :
  ∀ (x : ℚ), repeating_decimal_to_fraction (.\overline{36}) = 4/11 →
  1 / (repeating_decimal_to_fraction x) = 11/4 :=
by
  intros x hx
  have h : repeating_decimal_to_fraction x = 4/11 := hx
  rw h
  norm_num
  done
  sorry

end reciprocal_of_repeating_decimal_l389_389968


namespace check_correct_props_l389_389059

/-- 
Propositions:
① If α ∩ β = m, n ⊆ α, and n ⊥ m, then α ⊥ β.
② If α ⊥ β, α ∩ γ = m, and β ∩ γ = n, then m ⊥ n.
③ If α ⊥ β, α ⊥ γ, and β ∩ γ = m, then m ⊥ α.
④ If m ⊥ α, n ⊥ β, and m ⊥ n, then α ⊥ β.
-/
universe u
variables {α β γ : Type u} [plane α] [plane β] [plane γ]
variables {m n : Type u} [line m] [line n]

/-- The statement of the correctness of the propositions. -/
theorem check_correct_props
  (h1 : α ∩ β = m) (h2 : n ⊆ α) (h3 : perpendicular n m) :
  ((h1 ∧ h2 ∧ h3) → perpendicular α β) ∧
  ((perpendicular α β ∧ (α ∩ γ = m) ∧ (β ∩ γ = n)) → perpendicular m n) ∧
  ((perpendicular α β ∧ perpendicular α γ ∧ (β ∩ γ = m)) → perpendicular m α) ∧
  ((perpendicular m α ∧ perpendicular n β ∧ perpendicular m n) → perpendicular α β) :=
sorry

end check_correct_props_l389_389059


namespace arithmetic_mean_adjustment_l389_389072

theorem arithmetic_mean_adjustment (mean : ℝ) (n : ℕ) (a b c : ℝ) :
  n = 60 → mean = 42 → a = 40 → b = 50 → c = 60 →
  (let new_mean := ((mean * n) - (a + b + c)) / (n - 3) in
  new_mean = 41.6) :=
by
  intros h_n h_mean h_a h_b h_c
  let total_sum := mean * n
  let new_sum := total_sum - (a + b + c)
  let new_mean := new_sum / (n - 3)
  have : new_mean = 41.6, from sorry
  exact this

end arithmetic_mean_adjustment_l389_389072


namespace contains_zero_if_sum_is_111111_l389_389218

variables (x y : ℕ)
variables (digits_x digits_y : Fin 5 → ℕ)

-- Conditions
def is_five_digit (n : ℕ) : Prop := 
  10000 ≤ n ∧ n < 100000

def differs_by_two_digit_swap (dx dy : Fin 5 → ℕ) : Prop :=
  ∃ i j : Fin 5, i ≠ j ∧ 
  (∀ k : Fin 5, (k = i → dy k = dx j) ∧ 
                (k = j → dy k = dx i) ∧ 
                (k ≠ i ∧ k ≠ j → dy k = dx k))

theorem contains_zero_if_sum_is_111111 
  (hx : is_five_digit x) (hy : is_five_digit y)
  (h_digits_x : digits_x = fun i : Fin 5 => (x / 10^(4-i) % 10))
  (h_digits_y : digits_y = fun i : Fin 5 => (y / 10^(4-i) % 10))
  (h_swap : differs_by_two_digit_swap digits_x digits_y)
  (h_sum : x + y = 111111) :
  ∃ i : Fin 5, digits_x i = 0 :=
begin
  -- Proof omitted
  sorry,
end

end contains_zero_if_sum_is_111111_l389_389218


namespace three_digit_numbers_count_l389_389434

def is_3_digit_number (n : Nat) : Prop :=
  100 ≤ n ∧ n ≤ 999

def units_digit (n : Nat) : Nat :=
  n % 10

def tens_digit (n : Nat) : Nat :=
  (n / 10) % 10

def hundreds_digit (n : Nat) : Nat :=
  (n / 100) % 10

def valid_number (n : Nat) : Prop :=
  is_3_digit_number n ∧ units_digit n ≥ 3 * tens_digit n

theorem three_digit_numbers_count : 
  (Finset.card (Finset.filter valid_number (Finset.range 1000))) = 198 :=
by
  sorry

end three_digit_numbers_count_l389_389434


namespace red_sequence_57_eq_103_l389_389787

-- Definitions based on conditions described in the problem
def red_sequence : Nat → Nat
| 0 => 1  -- First number is 1
| 1 => 2  -- Next even number
| 2 => 4  -- Next even number
-- Continue defining based on patterns from problem
| (n+3) => -- Each element recursively following the pattern
 sorry  -- Detailed pattern definition is skipped

-- Main theorem: the 57th number in the red subsequence is 103
theorem red_sequence_57_eq_103 : red_sequence 56 = 103 :=
 sorry

end red_sequence_57_eq_103_l389_389787


namespace expression_maximum_value_l389_389293

noncomputable def expression (seq : Fin 1990 → ℕ) : ℕ :=
  abs (seq 0 - seq 1) - seq[2] - seq[3] - ... - seq[1989]

theorem expression_maximum_value :
  (∃ seq : Fin 1990 → ℕ, (∀ i j, i ≠ j → seq i ≠ seq j) ∧ (∀ i, seq i ∈ {1, 2, ..., 1990})) →
    (∃ seq : Fin 1990 → ℕ, ∀ i j, expression seq ≤ 1989) :=
by
  sorry

end expression_maximum_value_l389_389293


namespace fraction_meaningful_l389_389153

theorem fraction_meaningful (x : ℝ) : (x ≠ 2) ↔ (x - 2 ≠ 0) :=
by
  sorry

end fraction_meaningful_l389_389153


namespace contains_zero_if_sum_is_111111_l389_389220

variables (x y : ℕ)
variables (digits_x digits_y : Fin 5 → ℕ)

-- Conditions
def is_five_digit (n : ℕ) : Prop := 
  10000 ≤ n ∧ n < 100000

def differs_by_two_digit_swap (dx dy : Fin 5 → ℕ) : Prop :=
  ∃ i j : Fin 5, i ≠ j ∧ 
  (∀ k : Fin 5, (k = i → dy k = dx j) ∧ 
                (k = j → dy k = dx i) ∧ 
                (k ≠ i ∧ k ≠ j → dy k = dx k))

theorem contains_zero_if_sum_is_111111 
  (hx : is_five_digit x) (hy : is_five_digit y)
  (h_digits_x : digits_x = fun i : Fin 5 => (x / 10^(4-i) % 10))
  (h_digits_y : digits_y = fun i : Fin 5 => (y / 10^(4-i) % 10))
  (h_swap : differs_by_two_digit_swap digits_x digits_y)
  (h_sum : x + y = 111111) :
  ∃ i : Fin 5, digits_x i = 0 :=
begin
  -- Proof omitted
  sorry,
end

end contains_zero_if_sum_is_111111_l389_389220


namespace digit_contains_zero_l389_389224

theorem digit_contains_zero 
  (n₁ n₂ : ℕ) 
  (h₁ : 10000 ≤ n₁ ∧ n₁ ≤ 99999)
  (h₂ : 10000 ≤ n₂ ∧ n₂ ≤ 99999)
  (h₃ : ∃ (i j : ℕ), i ≠ j ∧ n₂ = n₁.swap_digits i j)
  (h₄ : n₁ + n₂ = 111111) 
  : ∃ (d : ℕ), d = 0 :=
sorry

end digit_contains_zero_l389_389224


namespace total_onions_l389_389531

theorem total_onions (S SA F J : ℕ) (h1 : S = 4) (h2 : SA = 5) (h3 : F = 9) (h4 : J = 7) : S + SA + F + J = 25 :=
by {
  sorry
}

end total_onions_l389_389531


namespace hyperbola_eccentricity_l389_389755

-- Definitions and conditions
variables {a b : ℝ} (h_a : a > 0) (h_b : b > 0)
def hyperbola (x y : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1
def inSecondQuadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0
def perpendicularBisectorCondition (x y : ℝ) : Prop := (y = (b / a) * x)

-- Theorem
theorem hyperbola_eccentricity (h : ∃ (x y : ℝ), hyperbola x y ∧ inSecondQuadrant x y ∧ perpendicularBisectorCondition x y) : 
  let c := sqrt (a^2 + b^2) in
  let e := c / a in
  e = sqrt 5 :=
by
  sorry

end hyperbola_eccentricity_l389_389755


namespace bounded_sum_l389_389064

variables {n : ℕ} {b : ℕ → ℝ} {a : ℕ → ℝ} {m M : ℝ}

-- Conditions:
-- 1. Monotonic decreasing sequence b.
def mon_decr (b : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ n → b i ≥ b j ∧ b j > 0

-- 2. Sum bounds for sequence a.
def sum_bounds (a : ℕ → ℝ) (m M : ℝ) (n : ℕ) : Prop :=
  ∀ t, 1 ≤ t ∧ t ≤ n → m ≤ (∑ i in (finset.range t), a i) ∧ (∑ i in (finset.range t), a i) ≤ M

-- Theorem statement:
theorem bounded_sum (h_decr: mon_decr b n) (h_sum: sum_bounds a m M n) :
  b 1 * m ≤ (∑ k in (finset.range n), a k * b k) ∧ (∑ k in (finset.range n), a k * b k) ≤ b 1 * M :=
sorry

end bounded_sum_l389_389064


namespace contains_zero_l389_389256

open Nat

theorem contains_zero 
  (x y : ℕ)
  (hx : 10000 ≤ x ∧ x < 100000)
  (hy : 10000 ≤ y ∧ y < 100000)
  (h_swap : ∃ (i j : ℕ), i ≠ j ∧ swap_digits x i j = y)
  (h_sum : x + y = 111111) :
  (∃ i, digit x i = 0) ∨ (∃ j, digit y j = 0) :=
sorry

-- Swap two digits in a number
def swap_digits (n : ℕ) (i j : ℕ) : ℕ := 
  -- Dummy definition for the proof placeholder
  n -- replace with actual implementation

-- Extract the digit at position i
def digit (n : ℕ) (i : ℕ) : ℕ :=
  -- Dummy definition for the proof placeholder
  0 -- replace with actual implementation

end contains_zero_l389_389256


namespace polygon_has_five_sides_l389_389141

theorem polygon_has_five_sides (S : ℕ) (h : S = 540) : (n : ℕ) (h_eq : (n - 2) * 180 = 540) := 
  n = 5 :=
by 
  intro n h_eq
  sorry

end polygon_has_five_sides_l389_389141


namespace right_triangle_ratio_l389_389505

theorem right_triangle_ratio
  (A B C D E F : Type)
  [is_right_triangle ABC]
  (h1 : ∠A = 90)
  (h2 : midpoint D A B)
  (h3 : point_on_segment E A C)
  (h4 : AD = AE)
  (h5 : intersects BE CD F)
  (h6 : ∠BFC = 135) :
  BC / AB = sqrt 5 / 2 :=
sorry

end right_triangle_ratio_l389_389505


namespace largest_class_students_l389_389785

theorem largest_class_students (x : ℕ)
  (h1 : x + (x - 2) + (x - 4) + (x - 6) + (x - 8) = 115) : x = 27 := 
by 
  sorry

end largest_class_students_l389_389785


namespace ratio_new_time_to_previous_time_l389_389206

theorem ratio_new_time_to_previous_time (distance : ℝ) (time_previous : ℝ) (speed_new : ℝ) :
  distance = 252 → time_previous = 6 → speed_new = 28 →
  (distance / speed_new) / time_previous = 3 / 2 :=
by
  intros h_distance h_time_previous h_speed_new
  rw [h_distance, h_time_previous, h_speed_new]
  calc
    (252 / 28) / 6 = 9 / 6  : by norm_num
                ... = 3 / 2 : by norm_num

end ratio_new_time_to_previous_time_l389_389206


namespace smallest_N_l389_389297

theorem smallest_N (N : ℕ) : 
  (N = 484) ∧ 
  (∃ k : ℕ, 484 = 4 * k) ∧
  (∃ k : ℕ, 485 = 25 * k) ∧
  (∃ k : ℕ, 486 = 9 * k) ∧
  (∃ k : ℕ, 487 = 121 * k) :=
by
  -- Proof omitted (replaced by sorry)
  sorry

end smallest_N_l389_389297


namespace correct_drawings_count_l389_389644

def suspect := Type
def link (A B : suspect) := Prop

def suspects : list suspect := [Ali, Bob, Cai, Dee, Eve, Fay]

def links : list (suspect × suspect) := [
  (Ali, Bob),
  (Bob, Cai),
  (Cai, Dee),
  (Dee, Eve),
  (Eve, Fay),
  (Fay, Ali),
  (Ali, Dee),
  (Bob, Eve)
]

theorem correct_drawings_count : 
  ∃ d : ℕ, (d = 1) ∧ 
  (∀ A B, link A B → (A, B) ∈ links) := 
sorry

end correct_drawings_count_l389_389644


namespace translated_and_stretched_function_l389_389154

theorem translated_and_stretched_function :
  ∀ x : ℝ, ∃ y : ℝ, y = sin (x - π / 3) →
  (y = sin (x + π / 6 - π / 3) → y = sin (x - π / 6)) →
  (y = sin (½ * x - π / 6)) :=
by
  intros x y h1 h2 h3
  sorry

end translated_and_stretched_function_l389_389154


namespace david_grade_students_l389_389278

def total_students_in_grade (better worse : ℕ) : ℕ := better + worse + 1

theorem david_grade_students (better worse : ℕ) (h_best : better = 74) (h_worst : worse = 74) : total_students_in_grade better worse = 149 :=
  by
  unfold total_students_in_grade
  rw [h_best, h_worst]
  -- result follows after simplification
  sorry

end david_grade_students_l389_389278


namespace sum_mod_7_remainder_l389_389586

def sum_to (n : ℕ) : ℕ := (n * (n + 1)) / 2

theorem sum_mod_7_remainder : (sum_to 140) % 7 = 0 :=
by
  sorry

end sum_mod_7_remainder_l389_389586


namespace tan_A_over_tan_C_a_squared_minus_b_squared_minus_c_squared_over_bc_l389_389730

-- Problem 1
theorem tan_A_over_tan_C 
  (a b c A B C : ℝ) 
  (h : c * cos A - a * cos C = (2 / 3) * b) 
  (h1 : ∠ABC = A)
  (h2 : ∠BAC = B)
  (h3 : ∠BCA = C) :
  tan A / tan C = 1 / 5 :=
by sorry

-- Problem 2
theorem a_squared_minus_b_squared_minus_c_squared_over_bc 
  (a b c A B C : ℝ) 
  (h : c * cos A - a * cos C = (2 / 3) * b) 
  (h_arith : 2 * tan B = tan A + tan C)
  (h1 : ∠ABC = A)
  (h2 : ∠BAC = B)
  (h3 : ∠BCA = C) :
  (a^2 - b^2 - c^2) / (b * c) = - (√10) / 2 :=
by sorry

end tan_A_over_tan_C_a_squared_minus_b_squared_minus_c_squared_over_bc_l389_389730


namespace range_of_a_l389_389009

open Real

theorem range_of_a (a : ℝ) 
  (h : ¬ ∃ x₀ : ℝ, 2 ^ x₀ - 2 ≤ a ^ 2 - 3 * a) : 1 ≤ a ∧ a ≤ 2 := 
by
  sorry

end range_of_a_l389_389009


namespace squares_in_ap_l389_389130

theorem squares_in_ap (a b c : ℝ) (h : (1 / (a + b) + 1 / (b + c)) / 2 = 1 / (a + c)) : 
  a^2 + c^2 = 2 * b^2 :=
by
  sorry

end squares_in_ap_l389_389130


namespace count_valid_three_digit_numbers_l389_389424

-- Define the property of a number being three digits
def isThreeDigit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

-- Define the property of the units digit being at least three times the tens digit
def validUnitsTimesTens (n : ℕ) : Prop :=
  let units := n % 10
  let tens := (n / 10) % 10
  units ≥ 3 * tens

-- Definition combining both conditions
def validThreeDigitNumber (n : ℕ) : Prop :=
  isThreeDigit n ∧ validUnitsTimesTens n

-- The main theorem stating the number of valid three-digit numbers
theorem count_valid_three_digit_numbers :
  Finset.card (Finset.filter validThreeDigitNumber (Finset.range 1000)) = 198 :=
by
  sorry

end count_valid_three_digit_numbers_l389_389424


namespace distinct_x_intercepts_l389_389441

theorem distinct_x_intercepts : 
  ∃ (s : Finset ℝ), s.card = 3 ∧ ∀ x, (x + 5) * (x^2 + 5 * x - 6) = 0 ↔ x ∈ s :=
by { 
  sorry 
}

end distinct_x_intercepts_l389_389441


namespace median_length_of_right_triangle_l389_389023

theorem median_length_of_right_triangle 
  (a b c : ℝ)
  (h1 : ∠C = 90)
  (h2 : a^2 + b^2 = c^2)
  (h3 : ∀ x : ℝ, x^2 - 7 * x + c + 7 = 0 → x = a ∨ x = b) :
  median_length a b = 5 / 2 := by
  sorry

end median_length_of_right_triangle_l389_389023


namespace Noah_age_in_10_years_is_22_l389_389082

def Joe_age : Nat := 6
def Noah_age := 2 * Joe_age
def Noah_age_after_10_years := Noah_age + 10

theorem Noah_age_in_10_years_is_22 : Noah_age_after_10_years = 22 := by
  sorry

end Noah_age_in_10_years_is_22_l389_389082


namespace correct_calculation_result_l389_389913

theorem correct_calculation_result :
  ∀ (A B D : ℝ),
  C = 6 →
  E = 5 →
  (A * 10 + B) * 6 + D * E = 39.6 ∨ (A * 10 + B) * 6 * D * E = 36.9 →
  (A * 10 + B) * 6 + D * E = 26.1 :=
by
  intros A B D C_eq E_eq errors
  sorry

end correct_calculation_result_l389_389913


namespace polynomial_root_b_value_l389_389729

noncomputable def b_from_roots (a b : ℚ) (r1 r2 r3 : ℚ) : ℚ := 
  -(r1 + r2 + r3)

theorem polynomial_root_b_value (a b : ℚ) :
  let r1 := 2 + Real.sqrt 5
  let r2 := 2 - Real.sqrt 5
  let r3 := 20
  (r1 * r2 * r3 = -20) → 
  b_from_roots a b r1 r2 r3 = -24 := 
by {
    have fact1 : (r1 * r2 = -1), by {
        have fact1_irr : (2^2 - (Real.sqrt 5)^2 = -1), by {
            sorry
        },
        exact fact1_irr,
    },
    have fact2 : (r1 * r2 * r3 = -20), by {
        sorry
    },
    sorry
}

end polynomial_root_b_value_l389_389729


namespace mean_score_is_94_5_l389_389459
-- Adding necessary imports

-- Definitions based on the conditions in a)
def scores : List ℕ := [120, 110, 100, 90, 75, 65, 50]
def frequencies : List ℕ := [12, 19, 33, 30, 15, 9, 2]
def total_students := 120

-- Lean 4 statement for the mean score proof
theorem mean_score_is_94_5 :
  ((List.zipWith (*) scores frequencies).sum / total_students : ℚ) = 94.5 := by
  sorry

end mean_score_is_94_5_l389_389459


namespace logical_form_equivalences_l389_389289

def divisible_by (a b : Nat) : Prop := b ∣ a

def is_sqrt (n a : Nat) : Prop := a * a = n

def is_high_school_freshman (person : Type) (p : person) : Prop := true  -- Placeholder for actual property definition
def is_youth_league_member (person : Type) (p : person) : Prop := true  -- Placeholder for actual property definition

variable (LiQiang : Type)
variable (liqiang : LiQiang)

theorem logical_form_equivalences :
  (divisible_by 15 3 ∧ divisible_by 15 5) ∧
  (is_sqrt 16 4 ∨ is_sqrt 16 (neg 4)) ∧
  (is_high_school_freshman LiQiang liqiang ∧ is_youth_league_member LiQiang liqiang) :=
by sorry

end logical_form_equivalences_l389_389289


namespace sequence_property_l389_389720

theorem sequence_property (a : ℕ+ → ℚ)
  (h1 : ∀ p q : ℕ+, a p + a q = a (p + q))
  (h2 : a 1 = 1 / 9) :
  a 36 = 4 :=
sorry

end sequence_property_l389_389720


namespace find_length_DE_l389_389477

open Real

theorem find_length_DE
  (A B C D E : Point ℝ)
  (A B C : Point ℝ)
  (hBC : dist B C = 40)
  (h_angle_C : ∠BAC = 45)
  (hD_midpoint : midpoint B C = D)
  (hDE_perpendicular : is_perpendicular DE BC) :
  dist D E = 10 * sqrt 2 :=
by
  sorry

end find_length_DE_l389_389477


namespace remainder_91_pow_91_mod_100_l389_389296

theorem remainder_91_pow_91_mod_100 : Nat.mod (91 ^ 91) 100 = 91 :=
by
  sorry

end remainder_91_pow_91_mod_100_l389_389296


namespace Rachel_drinks_correct_glasses_l389_389090

def glasses_Sunday : ℕ := 2
def glasses_Monday : ℕ := 4
def glasses_TuesdayToFriday : ℕ := 3
def days_TuesdayToFriday : ℕ := 4
def ounces_per_glass : ℕ := 10
def total_goal : ℕ := 220
def glasses_Saturday : ℕ := 4

theorem Rachel_drinks_correct_glasses :
  ounces_per_glass * (glasses_Sunday + glasses_Monday + days_TuesdayToFriday * glasses_TuesdayToFriday + glasses_Saturday) = total_goal :=
sorry

end Rachel_drinks_correct_glasses_l389_389090


namespace four_common_tangents_intersect_or_tangent_to_circle_l389_389109

theorem four_common_tangents_intersect_or_tangent_to_circle
  (S1 S2 S3 S4 : Circle)
  (external_touch_12 : ∃ P1, P1 ∈ S1 ∧ P1 ∈ S2)
  (external_touch_23 : ∃ P2, P2 ∈ S2 ∧ P2 ∈ S3)
  (external_touch_34 : ∃ P3, P3 ∈ S3 ∧ P3 ∈ S4)
  (external_touch_41 : ∃ P4, P4 ∈ S4 ∧ P4 ∈ S1) :
  ∃ O : Point, (∃ l, is_tangent_line_of_circle l O ∧ 
  (∀ i, l.1 = O ∨ l.2 = O ∨ (l.1 + l.2 + O = i))) ∨
  (∀ P, is_tangent_together S1 S2 S3 S4 P) := sorry

end four_common_tangents_intersect_or_tangent_to_circle_l389_389109


namespace tree_planting_campaign_l389_389931

theorem tree_planting_campaign
  (P : ℝ)
  (h1 : 456 = P * (1 - 1/20))
  (h2 : P ≥ 0)
  : (P * (1 + 0.1)) = (456 / (1 - 1/20) * 1.1) :=
by
  sorry

end tree_planting_campaign_l389_389931


namespace general_formula_a_minimum_m_l389_389567

noncomputable def a (n : ℕ) : ℕ :=
  match n with
  | 0       => 0
  | succ n' => 2 * (n' + 1) - 1

def b (n : ℕ) : ℚ :=
  (a n + 1) / 2^(a n)

def S (n : ℕ) : ℚ :=
  (finset.range n).sum (λ k => b (k+1))

def satisfies_inequality (n m : ℚ) : Prop :=
  (3 * n + 4) * m ≥ (2 * n - 5) * (16 / 9 - S n) * 2 ^ n

theorem general_formula_a (n : ℕ) :
  a n = 2 * n - 1 :=
sorry

theorem minimum_m : ∃ m : ℚ, (∀ n : ℕ, satisfies_inequality n m) ∧ m = 1 / 12 :=
sorry

end general_formula_a_minimum_m_l389_389567


namespace smallest_n_l389_389588

theorem smallest_n (n : ℕ) : ∃ n : ℕ, n > 0 ∧ (∀ m : ℕ, m > 0 → (sqrt m.to_real - sqrt (m-1).to_real < 0.02 → m >= 626)) :=
by
  sorry

end smallest_n_l389_389588


namespace remainder_m_n_mod_1000_l389_389500

noncomputable def m : ℕ :=
  Nat.card {p : ℕ × ℕ × ℕ // 4 * p.1 + 3 * p.2 + 2 * p.2.1 = 2009 ∧ 0 < p.1 ∧ 0 < p.2 ∧ 0 < p.2.1}

noncomputable def n : ℕ :=
  Nat.card {p : ℕ × ℕ × ℕ // 4 * p.1 + 3 * p.2 + 2 * p.2.1 = 2000 ∧ 0 < p.1 ∧ 0 < p.2 ∧ 0 < p.2.1}

theorem remainder_m_n_mod_1000 : (m - n) % 1000 = 0 :=
by
  sorry

end remainder_m_n_mod_1000_l389_389500


namespace f_divisible_by_27_l389_389876

theorem f_divisible_by_27 (n : ℕ) : 27 ∣ (2^(2*n - 1) - 9 * n^2 + 21 * n - 14) :=
sorry

end f_divisible_by_27_l389_389876


namespace oxen_eat_as_much_as_buffaloes_or_cows_l389_389178

theorem oxen_eat_as_much_as_buffaloes_or_cows
  (B C O : ℝ)
  (h1 : 3 * B = 4 * C)
  (h2 : (15 * B + 8 * O + 24 * C) * 36 = (30 * B + 8 * O + 64 * C) * 18) :
  3 * B = 4 * O :=
by sorry

end oxen_eat_as_much_as_buffaloes_or_cows_l389_389178


namespace sum_of_digits_is_15_l389_389044

-- Define the initial sequence generation
def initial_sequence : List ℕ := List.repeat [1, 2, 3, 4, 5, 6] (12000 / 6)

-- Define a function to perform the erasures
def erase_every_nth (seq : List ℕ) (n : ℕ) : List ℕ :=
  seq.enum.filter (λ x, (x.1 + 1) % n ≠ 0).map Prod.snd

def resulting_sequence_after_erasure : List ℕ :=
  let after_first_erasure := erase_every_nth initial_sequence 4
  let after_second_erasure := erase_every_nth after_first_erasure 5
  erase_every_nth after_second_erasure 3

def get_digits_at_positions (seq : List ℕ) (positions : List ℕ) : List ℕ :=
  positions.map (λ pos, seq.get! (pos % seq.length))

def positions := [3031, 3032, 3033]

def digits_at_positions := get_digits_at_positions resulting_sequence_after_erasure positions

def sum_digits := digits_at_positions.sum

theorem sum_of_digits_is_15 :
  sum_digits = 15 := by
  sorry

end sum_of_digits_is_15_l389_389044


namespace julia_paint_area_l389_389486

noncomputable def area_to_paint (bedroom_length: ℕ) (bedroom_width: ℕ) (bedroom_height: ℕ) (non_paint_area: ℕ) (num_bedrooms: ℕ) : ℕ :=
  let wall_area_one_bedroom := 2 * (bedroom_length * bedroom_height) + 2 * (bedroom_width * bedroom_height)
  let paintable_area_one_bedroom := wall_area_one_bedroom - non_paint_area
  num_bedrooms * paintable_area_one_bedroom

theorem julia_paint_area :
  area_to_paint 14 11 9 70 4 = 1520 :=
by
  sorry

end julia_paint_area_l389_389486


namespace calculate_area_bounded_figure_l389_389658

noncomputable def area_of_bounded_figure : ℝ :=
  let x := λ (t : ℝ), 4 * real.sqrt 2 * real.cos t^3
  let y := λ (t : ℝ), 2 * real.sqrt 2 * real.sin t^3
  let integral := 2 * ∫ t in (0 : ℝ)..(real.pi / 4), y t * (deriv x t)
  (integral).val

theorem calculate_area_bounded_figure :
  area_of_bounded_figure = (3 / 2) * real.pi - 2 :=
by
  sorry

end calculate_area_bounded_figure_l389_389658


namespace last_digit_appears_at_6_l389_389272

noncomputable def modified_fib (n : ℕ) : ℕ :=
  if n = 1 then 2
  else if n = 2 then 3
  else modified_fib (n - 1) + modified_fib (n - 2)

def last_digit (n : ℕ) : ℕ :=
  modified_fib n % 10

def appears (digit : ℕ) : ℕ → Prop
| 0 := last_digit 0 = digit
| (n + 1) := last_digit (n + 1) = digit ∨ appears digit n

theorem last_digit_appears_at_6 : ∃ n, last_digit n = 6 :=
  sorry

end last_digit_appears_at_6_l389_389272


namespace range_of_f_l389_389367

-- Define the interval
def interval : Set ℝ := set.Icc (Real.pi / 4) (Real.pi / 2)

-- Define the function
def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 4)

-- Define the target range
def target_range : Set ℝ := set.Icc (-Real.sqrt 2 / 2) (Real.sqrt 2 / 2)

-- Statement to be proved: range of 'f' on the specified interval equals target_range
theorem range_of_f : Real.range (f ∘ (λ x, x ∈ interval)) = target_range :=
sorry

end range_of_f_l389_389367


namespace green_peaches_sum_l389_389919

theorem green_peaches_sum (G1 G2 G3 : ℕ) : 
  (4 + G1) + (4 + G2) + (3 + G3) = 20 → G1 + G2 + G3 = 9 :=
by
  intro h
  sorry

end green_peaches_sum_l389_389919


namespace lateral_area_of_cylinder_l389_389633

theorem lateral_area_of_cylinder (side_length : ℝ) (h_square : side_length = 1) :
    let r := side_length
    let h := side_length
    let A := 2 * Real.pi * r * h
    A = 2 * Real.pi :=
by
  -- By the conditions of the problem
  rw [h_square]
  -- Simplify the formula for lateral area
  rw [mul_one, mul_one]

  -- A simple step to show the answer:
  sorry

end lateral_area_of_cylinder_l389_389633


namespace b_20_value_l389_389122

-- Definitions based on conditions
def a (n : ℕ) : ℕ := 2 * n - 1

def b (n : ℕ) : ℕ := a n  -- Given that \( b_n = a_n \)

-- The theorem stating that \( b_{20} = 39 \)
theorem b_20_value : b 20 = 39 :=
by
  -- Skipping the proof
  sorry

end b_20_value_l389_389122


namespace fruit_store_initial_quantities_l389_389618

-- Definitions from conditions:
def total_fruit (a b c : ℕ) := a + b + c = 275
def sold_apples (a : ℕ) := a - 30
def added_peaches (b : ℕ) := b + 45
def sold_pears (c : ℕ) := c - c / 4
def final_ratio (a b c : ℕ) := (sold_apples a) / 4 = (added_peaches b) / 3 ∧ (added_peaches b) / 3 = (sold_pears c) / 2

-- The proof problem:
theorem fruit_store_initial_quantities (a b c : ℕ) (h1 : total_fruit a b c) 
  (h2 : final_ratio a b c) : a = 150 ∧ b = 45 ∧ c = 80 :=
sorry

end fruit_store_initial_quantities_l389_389618


namespace area_between_sqrt_and_x_l389_389108

noncomputable def area_enclosed_by_sqrt_x_and_x : ℝ :=
∫ x in 0..1, real.sqrt x - x

theorem area_between_sqrt_and_x :
  area_enclosed_by_sqrt_x_and_x = 1 / 6 :=
by
  sorry

end area_between_sqrt_and_x_l389_389108


namespace part1_monotonic_intervals_part2_range_of_a_l389_389844

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x * Real.log x - x + 0.5

theorem part1_monotonic_intervals (x : ℝ) : 
  (f 1 x < (f 1 (x + 1)) ↔ x < 1) ∧ 
  (f 1 x > (f 1 (x - 1)) ↔ x > 1) :=
by sorry

theorem part2_range_of_a (a : ℝ) (x : ℝ) (hx : 1 < x ∧ x ≤ Real.exp 1) 
  (h : (f a x / x) + (1 / (2 * x)) < 0) : 
  a < 1 - (1 / Real.exp 1) :=
by sorry

end part1_monotonic_intervals_part2_range_of_a_l389_389844


namespace min_P_equals_34_over_67_l389_389705

def floor_div (a b : ℕ) : ℕ := a / b

noncomputable def P (k : ℕ) : ℚ :=
(1 - (2 * (floor_div 200 k).to_nat - 1) / k)

noncomputable def minimum_possible_value_of_P : ℚ :=
if h : (k % 2 = 1) ∧ (1 ≤ k) ∧ (k ≤ 199) then 
  min (P k)
else 
  0

theorem min_P_equals_34_over_67 : minimum_possible_value_of_P = 34 / 67 := sorry

end min_P_equals_34_over_67_l389_389705


namespace mildred_weight_is_correct_l389_389077

noncomputable def carol_weight := 9
noncomputable def mildred_weight := carol_weight + 50

theorem mildred_weight_is_correct : mildred_weight = 59 :=
by 
  -- the proof is omitted
  sorry

end mildred_weight_is_correct_l389_389077


namespace find_m_l389_389768

theorem find_m (m : ℕ) (h : 7^(2 * m) = (1 / 7)^(m - 30)) : m = 10 :=
by {
  have h1 : 7^(2 * m) = 7^(-m + 30),
  {
    rw [←pow_neg],
    exact h,
  },
  have h2 : 2 * m = -m + 30,
  {
    apply pow_inj,
    exact nat.prime_two,
    exact nat.prime_odd_one,
    exact h1,
  },
  have h3 : 3 * m = 30,
  {
    linarith,
  },
  exact (eq_of_mul_eq_mul_right (nat.zero_lt_succ 2) h3),
}

end find_m_l389_389768


namespace rectangular_plot_dimensions_l389_389628

theorem rectangular_plot_dimensions (a b : ℝ) 
  (h_area : a * b = 800) 
  (h_perimeter_fencing : 2 * a + b = 100) :
  (a = 40 ∧ b = 20) ∨ (a = 10 ∧ b = 80) := 
sorry

end rectangular_plot_dimensions_l389_389628


namespace line_AB_eq_line_circle_intersection_l389_389025

-- Definitions based on the conditions
def polar_to_cartesian (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

def circle_parametric_eq (θ : ℝ) : ℝ × ℝ :=
  (1 + 2 * Real.cos θ, 2 * Real.sin θ)

def line_eq (A B : ℝ × ℝ) : ℝ → ℝ := λ x, -x + 2

-- Points A and B in Cartesian coordinates
def point_A : ℝ × ℝ := polar_to_cartesian 2 (Real.pi / 2)
def point_B : ℝ × ℝ := polar_to_cartesian (Real.sqrt 2) (Real.pi / 4)

-- Line AB in Cartesian coordinates
def line_AB (x : ℝ) : ℝ := line_eq point_A point_B x

-- The standard form of line AB
theorem line_AB_eq : ∀ x y, x + y - 2 = 0 ↔ y = line_AB x := sorry

-- Determining the intersection of line AB and circle C
def center_C : ℝ × ℝ := (1, 0)
def radius_C : ℝ := 2

theorem line_circle_intersection : 
  let line_distance := λ (C : ℝ × ℝ), (|C.1 - 2| / Real.sqrt 2)
  line_distance center_C < radius_C ↔ line_AB 1 = 1 := sorry

end line_AB_eq_line_circle_intersection_l389_389025


namespace contains_zero_l389_389238

theorem contains_zero (a b c d e f: ℕ) (h1: 1 ≤ a ∧ a ≤ 9) (h2: 1 ≤ b ∧ b ≤ 9) (h3: 0 ≤ c ∧ c ≤ 9) 
  (h4: 1 ≤ d ∧ d ≤ 9) (h5: 0 ≤ e ∧ e ≤ 9) (h6: 0 ≤ f ∧ f ≤ 9) 
  (h7: c ≠ f) (h8: 10^4*a + 10^3*b + 10^2*c + 10^1*d + e + 10^4*a + 10^3*b + 10^2*f + 10^1*d + e = 111111) :
  c = 0 ∨ f = 0 := 
sorry

end contains_zero_l389_389238


namespace find_f_500_l389_389899

noncomputable def f : ℝ → ℝ := sorry

axiom f_continuous : continuous f
axiom f_property : ∀ x : ℝ, f(x) * f(f(x)) = 1
axiom f_at_1000 : f 1000 = 999

theorem find_f_500 : f 500 = 1 / 500 :=
by sorry

end find_f_500_l389_389899


namespace contains_zero_l389_389234

-- Define what constitutes a five-digit number
def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

-- Define the main proof statement
theorem contains_zero (n1 n2 : ℕ) (c1 c2 : Prop) :
  is_five_digit n1 →
  is_five_digit n2 →
  c1 ∧ c2 →
  n1 + n2 = 111111 →
  ∃ (d : ℕ), d < 10 ∧ (d = 0 ∧ (∃ p1 p2 : ℕ, n1 = p1 * 10 + d ∧ n2 = p2 * 10 + d) ∨ 
             ∃ p1 p2 : ℕ, n1 = d * (10^4) + p1 ∧ n2 = d * (10^4) + p2) :=
begin
  sorry
end

end contains_zero_l389_389234


namespace three_digit_numbers_count_l389_389437

def is_3_digit_number (n : Nat) : Prop :=
  100 ≤ n ∧ n ≤ 999

def units_digit (n : Nat) : Nat :=
  n % 10

def tens_digit (n : Nat) : Nat :=
  (n / 10) % 10

def hundreds_digit (n : Nat) : Nat :=
  (n / 100) % 10

def valid_number (n : Nat) : Prop :=
  is_3_digit_number n ∧ units_digit n ≥ 3 * tens_digit n

theorem three_digit_numbers_count : 
  (Finset.card (Finset.filter valid_number (Finset.range 1000))) = 198 :=
by
  sorry

end three_digit_numbers_count_l389_389437


namespace find_single_digit_number_l389_389623

-- Define the given conditions:
def single_digit (A : ℕ) := A < 10
def rounded_down_tens (x : ℕ) (result: ℕ) := (x / 10) * 10 = result

-- Lean statement of the problem:
theorem find_single_digit_number (A : ℕ) (H1 : single_digit A) (H2 : rounded_down_tens (A * 1000 + 567) 2560) : A = 2 :=
sorry

end find_single_digit_number_l389_389623


namespace average_calculation_l389_389545

def average (a b c : ℚ) : ℚ := (a + b + c) / 3
def pairAverage (a b : ℚ) : ℚ := (a + b) / 2

theorem average_calculation :
  average (average (pairAverage 2 2) 3 1) (pairAverage 1 2) 1 = 3 / 2 := sorry

end average_calculation_l389_389545


namespace sum_of_cubes_l389_389011

theorem sum_of_cubes (x y : ℝ) (h_sum : x + y = 3) (h_prod : x * y = 2) : x^3 + y^3 = 9 :=
by
  sorry

end sum_of_cubes_l389_389011


namespace probability_of_drawing_1_red_1_white_l389_389782

-- Definitions
def total_balls : ℕ := 5
def red_balls : ℕ := 2
def white_balls : ℕ := 3

-- Probabilities
def p_red_first_white_second : ℚ := (red_balls / total_balls : ℚ) * (white_balls / total_balls : ℚ)
def p_white_first_red_second : ℚ := (white_balls / total_balls : ℚ) * (red_balls / total_balls : ℚ)

-- Total probability
def total_probability : ℚ := p_red_first_white_second + p_white_first_red_second

theorem probability_of_drawing_1_red_1_white :
  total_probability = 12 / 25 := by
  sorry

end probability_of_drawing_1_red_1_white_l389_389782


namespace product_without_x3_and_x2_l389_389008

theorem product_without_x3_and_x2 (m n : ℤ) :
  let p := (λ x : ℤ, x^2 + m * x) * (λ x : ℤ, x^2 - 2 * x + n)
  (∀ x : ℤ, x ^ 3 * (λ x : ℤ, p x) = 0) ∧ (∀ x : ℤ, x ^ 2 * (λ x : ℤ, p x) = 0)
  → (m = 2) ∧ (n = 4) :=
by {sorry}

end product_without_x3_and_x2_l389_389008


namespace reciprocal_of_repeating_decimal_l389_389964

theorem reciprocal_of_repeating_decimal :
  let x := 0.36363636 -- simplified as .\overline{36}
  ∃ y : ℚ, x = 4 / 11 ∧ y = 1 / x ∧ y = 11 / 4 :=
by
  sorry

end reciprocal_of_repeating_decimal_l389_389964


namespace triangle_properties_l389_389374

variable {A B C a b c : ℝ}

theorem triangle_properties
  (hA : A < π / 2)
  (h_sin : Real.sin (A - π / 4) = sqrt 2 / 10)
  (area : ℝ := 24)
  (h_b : b = 10) :
  Real.sin A = 4 / 5 ∧ 
  a = 8 := 
by 
  sorry

end triangle_properties_l389_389374


namespace multiples_of_5_in_range_l389_389144

theorem multiples_of_5_in_range {n : ℕ} (h : n = 100) : (Finset.card (Finset.filter (λ x, x % 5 = 0) (Finset.range (n + 1)))) = 20 :=
by
  rw h
  -- rewriting the proof would be continued here
  sorry

end multiples_of_5_in_range_l389_389144


namespace min_value_frac_sum_l389_389733

theorem min_value_frac_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + y = 1) : 
  ∃ m, ∀ x y, 0 < x → 0 < y → 2 * x + y = 1 → m ≤ (1/x + 1/y) ∧ (1/x + 1/y) = 3 + 2 * Real.sqrt 2 :=
sorry

end min_value_frac_sum_l389_389733


namespace area_enclosed_by_abs_inequality_l389_389291

noncomputable def enclosedArea : ℝ := 36

theorem area_enclosed_by_abs_inequality : 
  (∃ (A : ℝ), (∀ (x y : ℝ), abs (x + y) + abs (x - y) ≤ 6 → A = enclosedArea)) :=
begin
  -- The proof will be constructed here.
  sorry
end

end area_enclosed_by_abs_inequality_l389_389291


namespace angle_BFC_l389_389050

/-- 
Let \(ABC\) be an acute triangle and \(AD\) a height. The angle bisector of \(\angle DAC\) intersects \(DC\) at \(E\). 
Let \(F\) be a point on \(AE\) such that \(BF \perp AE\). If \(\angle BAE = 45^\circ\), then \(\angle BFC = 135^\circ\).
-/
theorem angle_BFC (A B C D E F : Type*)
  (acute_triangle_ABC : triangle A B C ∧ ∀ (A B C : Type*), acute_triangle A B C)
  (height_AD : is_height A D B C)
  (angle_bisector_DAC_E: is_angle_bisector (angle D A C) E D C)
  (F_on_AE : is_on_line_segment F A E)
  (BF_perp_AE : ∀ (B F E : Type*), perp B F A E)
  (angle_BAE_45 : angle B A E = 45) :
  angle B F C = 135 :=
sorry

end angle_BFC_l389_389050


namespace angle_eq_in_triangle_l389_389803

-- Variables representing points and triangles
variables {A B C I I_a A' A₁ : Type} [MetricSpace A]

-- Definitions to establish conditions from the problem statement
def is_incenter (I A B C : Type) := sorry
def is_excenter (I_a A B C : Type) := sorry
def is_circumcircle_opposite (A' A B C : Type) := sorry
def is_foot_of_altitude (A₁ A B C : Type) := sorry

-- The main theorem statement
theorem angle_eq_in_triangle 
  {A B C I I_a A' A₁ : Type} [MetricSpace A]
  (h1 : is_incenter I A B C)
  (h2 : is_excenter I_a A B C)
  (h3 : is_circumcircle_opposite A' A B C)
  (h4 : is_foot_of_altitude A₁ A B C) : 
  ∠ I A' I_a = ∠ I A₁ I_a :=
sorry

end angle_eq_in_triangle_l389_389803


namespace tax_diminished_percentage_l389_389912

theorem tax_diminished_percentage (T C : ℝ) (hT : T > 0) (hC : C > 0) (X : ℝ) 
  (h : T * (1 - X / 100) * C * 1.15 = T * C * 0.9315) : X = 19 :=
by 
  sorry

end tax_diminished_percentage_l389_389912


namespace problem_1_problem_2_l389_389539

theorem problem_1 (x : ℝ) : (2 * x + 3)^2 = 16 ↔ x = 1/2 ∨ x = -7/2 := by
  sorry

theorem problem_2 (x : ℝ) : x^2 - 4 * x - 3 = 0 ↔ x = 2 + Real.sqrt 7 ∨ x = 2 - Real.sqrt 7 := by
  sorry

end problem_1_problem_2_l389_389539


namespace shaded_squares_sum_l389_389472

theorem shaded_squares_sum :
  let shaded_squares := [1, 3, 5, 4, 2] in
  (List.sum shaded_squares) = 15 :=
by
  sorry

end shaded_squares_sum_l389_389472


namespace actual_distance_between_towns_l389_389553

def map_distance := 20 -- distance between towns on the map in inches
def scale := 10 -- scale: 1 inch = 10 miles

theorem actual_distance_between_towns : map_distance * scale = 200 := by
  sorry

end actual_distance_between_towns_l389_389553


namespace edward_garage_sale_games_l389_389690

variables
  (G_total : ℕ) -- total number of games
  (G_good : ℕ) -- number of good games
  (G_bad : ℕ) -- number of bad games
  (G_friend : ℕ) -- number of games bought from a friend
  (G_garage : ℕ) -- number of games bought at the garage sale

-- The conditions
def total_games (G_total : ℕ) (G_good : ℕ) (G_bad : ℕ) : Prop :=
  G_total = G_good + G_bad

def garage_sale_games (G_total : ℕ) (G_friend : ℕ) (G_garage : ℕ) : Prop :=
  G_total = G_friend + G_garage

-- The theorem to be proved
theorem edward_garage_sale_games
  (G_total : ℕ) 
  (G_good : ℕ) 
  (G_bad : ℕ)
  (G_friend : ℕ) 
  (G_garage : ℕ) 
  (h1 : total_games G_total G_good G_bad)
  (h2 : G_good = 24)
  (h3 : G_bad = 31)
  (h4 : G_friend = 41) :
  G_garage = 14 :=
by
  sorry

end edward_garage_sale_games_l389_389690


namespace third_twenty_third_wise_superior_number_l389_389281

def wise_superior_number (x : ℕ) : Prop :=
  ∃ m n : ℕ, m > n ∧ m - n > 1 ∧ x = m^2 - n^2

theorem third_twenty_third_wise_superior_number :
  ∃ T_3 T_23 : ℕ, wise_superior_number T_3 ∧ wise_superior_number T_23 ∧ T_3 = 15 ∧ T_23 = 57 :=
by
  sorry

end third_twenty_third_wise_superior_number_l389_389281


namespace digit_contains_zero_l389_389227

theorem digit_contains_zero 
  (n₁ n₂ : ℕ) 
  (h₁ : 10000 ≤ n₁ ∧ n₁ ≤ 99999)
  (h₂ : 10000 ≤ n₂ ∧ n₂ ≤ 99999)
  (h₃ : ∃ (i j : ℕ), i ≠ j ∧ n₂ = n₁.swap_digits i j)
  (h₄ : n₁ + n₂ = 111111) 
  : ∃ (d : ℕ), d = 0 :=
sorry

end digit_contains_zero_l389_389227


namespace number_of_smaller_triangles_filling_larger_triangle_l389_389019

theorem number_of_smaller_triangles_filling_larger_triangle :
  let side_large := 12
  let side_small := 2
  let area_large := (Math.sqrt 3 / 4) * side_large^2
  let area_small := (Math.sqrt 3 / 4) * side_small^2
  ∃ n : ℕ, n * area_small = area_large ∧ n = 36 :=
by
  sorry

end number_of_smaller_triangles_filling_larger_triangle_l389_389019


namespace solution_set_condition_l389_389569

theorem solution_set_condition (a : ℝ) :
  (∀ x : ℝ, x * (x - a + 1) > a ↔ (x < -1 ∨ x > a)) → a > -1 :=
sorry

end solution_set_condition_l389_389569


namespace find_L_l389_389043

-- Definitions based on the given conditions
def side_length_cube : ℝ := 3
def surface_area_cube : ℝ := 6 * side_length_cube^2
def radius_sphere (surface_area : ℝ) : ℝ := (surface_area / (4 * Real.pi))^.5
def volume_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3
def sphere_volume_correct_form (L : ℝ) : ℝ := (L * Real.sqrt 15) / Real.sqrt Real.pi

-- Theorem to prove
theorem find_L : ∃ (L : ℝ), sphere_volume_correct_form L = volume_sphere (radius_sphere surface_area_cube) ∧ L = 108 :=
by
  sorry

end find_L_l389_389043


namespace lucas_earnings_l389_389074

-- Declare constants and definitions given in the problem
def dollars_per_window : ℕ := 3
def windows_per_floor : ℕ := 5
def floors : ℕ := 4
def penalty_amount : ℕ := 2
def days_per_period : ℕ := 4
def total_days : ℕ := 12

-- Definition of the number of total windows
def total_windows : ℕ := windows_per_floor * floors

-- Initial earnings before penalties
def initial_earnings : ℕ := total_windows * dollars_per_window

-- Number of penalty periods
def penalty_periods : ℕ := total_days / days_per_period

-- Total penalty amount
def total_penalty : ℕ := penalty_periods * penalty_amount

-- Final earnings after penalties
def final_earnings : ℕ := initial_earnings - total_penalty

-- Proof problem: correct amount Lucas' father will pay
theorem lucas_earnings : final_earnings = 54 :=
by
  sorry

end lucas_earnings_l389_389074


namespace circle_configuration_problem_l389_389691

theorem circle_configuration_problem
  (U P Q R S F : Type)
  (radius_P : ℝ)
  (radius_Q : ℝ)
  (radius_R : ℝ)
  (radius_S : ℝ)
  (radius_F : ℝ)
  (m n : ℕ)
  (h1 : ∀ x, (x ∈ U) → isEquilateralTriangle x P)
  (h2 : radius_P = 12)
  (h3 : radius_Q = 4)
  (h4 : radius_R = 3)
  (h5 : radius_S = 3)
  (h6 : ∀ x, (x ∈ Q) → isInternallyTangent x P)
  (h7 : ∀ x, (x ∈ R ∨ x ∈ S) → isInternallyTangent x P)
  (h8 : ∀ x, (x ∈ Q ∨ x ∈ R ∨ x ∈ S) → isExternallyTangent x F)
  (h9 : coprime m n)
  (h10 : radius_F = (m : ℝ) / (n : ℝ)) :
  m + n = 111 := 
sorry

end circle_configuration_problem_l389_389691


namespace hexagon_diagonal_sum_l389_389622

noncomputable def diagonal_sum (x y z : ℕ) := x + y + z

theorem hexagon_diagonal_sum :
  (inscribed_hexagon : (A B C D E F : ℕ) (AB BC CD DE EF FA : ℕ) 
  (eq₁ : AB = 70) 
  (eq₂ : BC = 90) 
  (eq₃ : CD = 90) 
  (eq₄ : DE = 90) 
  (eq₅ : EF = 90) 
  (eq₆ : FA = 50) ⊢ 
  x = AC ∧ y = AD ∧ z = AE :=
  (diagonal_sum x y z = 376) sorry

end hexagon_diagonal_sum_l389_389622


namespace contains_zero_l389_389233

-- Define what constitutes a five-digit number
def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

-- Define the main proof statement
theorem contains_zero (n1 n2 : ℕ) (c1 c2 : Prop) :
  is_five_digit n1 →
  is_five_digit n2 →
  c1 ∧ c2 →
  n1 + n2 = 111111 →
  ∃ (d : ℕ), d < 10 ∧ (d = 0 ∧ (∃ p1 p2 : ℕ, n1 = p1 * 10 + d ∧ n2 = p2 * 10 + d) ∨ 
             ∃ p1 p2 : ℕ, n1 = d * (10^4) + p1 ∧ n2 = d * (10^4) + p2) :=
begin
  sorry
end

end contains_zero_l389_389233


namespace infinitely_many_n_l389_389282

def largest_odd_divisor (n : Nat) : Nat :=
  Nat.least_divisor n (λ m, m % 2 = 1 ∧ m > 1)

def D (n : Nat) : Nat :=
  (Finset.range n).sum (λ k, largest_odd_divisor (k + 1))

def T (n : Nat) : Nat :=
  (Finset.range n).sum (λ k, k + 1)

theorem infinitely_many_n :
  ∃ᶠ n in Finset.univ, 3 * D n = 2 * T n :=
sorry

end infinitely_many_n_l389_389282


namespace mouse_meeting_impossible_l389_389616

theorem mouse_meeting_impossible (mice : ℕ) (nights : ℕ) (group_size : ℕ)
  (h_mice : mice = 24)
  (h_group_size : group_size = 4)
  (h_total_nights : ∀ (x : ℕ), x = nights → ∀ (mouse : ℕ), mouse < 24 → (nights * 3) < (24 * 23)) : 
  ¬∃ (f : ℕ → set ℕ), (∀ n, # (f n) = 4 ∧ f n ⊆ finset.range 24) ∧ (∀ i j, i ≠ j → ∃! n, {i, j} ⊆ f n) := 
sorry

end mouse_meeting_impossible_l389_389616


namespace part1_part2_l389_389736

theorem part1 (n : ℕ) (a : Real) (a_i : Fin n → Real): 
  (Σ i, a_i i)^2 ≤ n * (Σ i, (a_i i)^2) :=
sorry

theorem part2 (x : Real) (h : 3 / 2 ≤ x ∧ x ≤ 5) : 
  2 * Real.sqrt (x + 1) + Real.sqrt (2 * x - 3) + Real.sqrt (15 - 3 * x) < 2 * Real.sqrt 19 :=
sorry

end part1_part2_l389_389736


namespace difference_of_fractions_l389_389994

-- The conditions
def one_twentieth_percent (x : ℝ) : ℝ := (x * 0.05) / 100
def one_tenth (x : ℝ) : ℝ := x / 10

-- The theorem statement
theorem difference_of_fractions (y : ℝ) (h₁ : one_twentieth_percent 8000 = 4) (h₂ : one_tenth 8000 = 800) :
  one_tenth 8000 - one_twentieth_percent 8000 = 796 :=
by
  rw [one_tenth, one_twentieth_percent]
  exact (h₂.sub h₁).trans (by norm_num)

-- Proof omitted

end difference_of_fractions_l389_389994


namespace value_satisfying_x_cubed_less_than_x_squared_l389_389710

theorem value_satisfying_x_cubed_less_than_x_squared :
  ∃ x ∈ ({5 / 3, 3 / 4, 1, 3 / 2, 21 / 20} : Set ℚ), x^3 < x^2 :=
by
  use 3 / 4
  split
  · right
    left
    refl
  · norm_num
  · norm_num

end value_satisfying_x_cubed_less_than_x_squared_l389_389710


namespace triangle_length_identity_l389_389676

theorem triangle_length_identity (a b : ℝ) (h : a = a) (hBC : 3 = 3) (hBD : 2 = 2) : 
  let AB := (0 - b)^2 + a^2 in
  let AC := (0 - (3 - b))^2 + a^2 in
  let AD := (0 - (2 - b))^2 + a^2 in
  AB + 2 * AC - 3 * AD = 6 := 
by
  -- Assume a and b are given and satisfy the given conditions.
  sorry

end triangle_length_identity_l389_389676


namespace integral_value_l389_389686

def integral1 : ℝ := ∫ x in (0 : ℝ)..1, real.sqrt(1 - (x - 1)^2)
def integral2 : ℝ := ∫ x in (0 : ℝ)..1, x^2

theorem integral_value : ∫ x in (0 : ℝ)..1, (real.sqrt (1 - (x - 1)^2) - x^2) = (π / 4) - (1 / 3) :=
by
  have h1 : integral1 = π / 4 := sorry
  have h2 : integral2 = 1 / 3 := sorry
  calc
    ∫ x in (0 : ℝ)..1, (real.sqrt(1 - (x - 1)^2) - x^2)
        = integral1 - integral2 : by sorry
    ... = (π / 4) - (1 / 3) : by rw [h1, h2]
  sorry

end integral_value_l389_389686


namespace x_is_half_l389_389000

theorem x_is_half (w x y : ℝ) 
  (h1 : 6 / w + 6 / x = 6 / y) 
  (h2 : w * x = y) 
  (h3 : (w + x) / 2 = 0.5) : x = 0.5 :=
sorry

end x_is_half_l389_389000


namespace fraction_equation_l389_389116

theorem fraction_equation (a : ℕ) (h : a > 0) (eq : (a : ℚ) / (a + 35) = 0.875) : a = 245 :=
by
  sorry

end fraction_equation_l389_389116


namespace area_of_set_M_l389_389797

theorem area_of_set_M :
  let M := {(x, y) | ∃ (α β : ℝ), x = Math.sin α + Math.cos β ∧ y = Math.cos α - Math.sin β} in
  ∃ (a : ℝ), set.area {p : ℝ × ℝ | ∃ q ∈ M, p = q} = 4 * π :=
by sorry

end area_of_set_M_l389_389797


namespace max_marks_l389_389204

theorem max_marks (marks_obtained marks_failed : ℕ) (pass_percent : ℕ) : ℕ :=
  let marks_required := marks_obtained + marks_failed in
  let max_marks := marks_required * 100 / pass_percent in
  nat.ceil (max_marks : ℝ)

example : max_marks 245 53 45 = 663 :=
by
  let marks_required := 245 + 53
  let max_marks := (marks_required * 100) / 45
  have : (θ : ℝ) ceil (max_marks : ℝ) = 663 := sorry
  exact ceiling_eq 663

end max_marks_l389_389204


namespace solve_for_x_l389_389884

theorem solve_for_x : ∀ x : ℝ, 2^(2*x - 6) = 8^(x + 2) → x = -12 :=
begin
  intro x,
  intro h,
  sorry,
end

end solve_for_x_l389_389884


namespace height_at_2_years_l389_389639

variable (height : ℕ → ℕ) -- height function representing the height of the tree at the end of n years
variable (triples_height : ∀ n, height (n + 1) = 3 * height n) -- tree triples its height every year
variable (height_4 : height 4 = 81) -- height at the end of 4 years is 81 feet

-- We need the height at the end of 2 years
theorem height_at_2_years : height 2 = 9 :=
by {
  sorry
}

end height_at_2_years_l389_389639


namespace tim_books_l389_389924

def has_some_books (Tim Sam : ℕ) : Prop :=
  Sam = 52 ∧ Tim + Sam = 96

theorem tim_books (Tim : ℕ) :
  has_some_books Tim 52 → Tim = 44 := 
by
  intro h
  obtain ⟨hSam, hTogether⟩ := h
  sorry

end tim_books_l389_389924


namespace best_selling_price_70_l389_389614

-- Definitions for the conditions in the problem
def purchase_price : ℕ := 40
def initial_selling_price : ℕ := 50
def initial_sales_volume : ℕ := 50

-- The profit function
def profit (x : ℕ) : ℕ :=
(50 + x - purchase_price) * (initial_sales_volume - x)

-- The problem statement to be proved
theorem best_selling_price_70 :
  ∃ x : ℕ, 0 < x ∧ x < 50 ∧ profit x = 900 ∧ (initial_selling_price + x) = 70 :=
by
  sorry

end best_selling_price_70_l389_389614


namespace perpendicular_line_through_point_l389_389555

open Real

theorem perpendicular_line_through_point (B : ℝ × ℝ) (x y : ℝ) (c : ℝ)
  (hB : B = (3, 0)) (h_perpendicular : 2 * x + y - 5 = 0) :
  x - 2 * y + 3 = 0 :=
sorry

end perpendicular_line_through_point_l389_389555


namespace count_valid_three_digit_numbers_l389_389419

-- Define the property of a number being three digits
def isThreeDigit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

-- Define the property of the units digit being at least three times the tens digit
def validUnitsTimesTens (n : ℕ) : Prop :=
  let units := n % 10
  let tens := (n / 10) % 10
  units ≥ 3 * tens

-- Definition combining both conditions
def validThreeDigitNumber (n : ℕ) : Prop :=
  isThreeDigit n ∧ validUnitsTimesTens n

-- The main theorem stating the number of valid three-digit numbers
theorem count_valid_three_digit_numbers :
  Finset.card (Finset.filter validThreeDigitNumber (Finset.range 1000)) = 198 :=
by
  sorry

end count_valid_three_digit_numbers_l389_389419


namespace postman_pete_mileage_l389_389526

theorem postman_pete_mileage :
  let initial_steps := 30000
  let resets := 72
  let final_steps := 45000
  let steps_per_mile := 1500
  let steps_per_full_cycle := 99999 + 1
  let total_steps := initial_steps + resets * steps_per_full_cycle + final_steps
  total_steps / steps_per_mile = 4850 := 
by 
  sorry

end postman_pete_mileage_l389_389526


namespace count_valid_numbers_l389_389402

def is_valid_number (n : Nat) : Prop := 
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let units := n % 10
  1 ≤ hundreds ∧ hundreds ≤ 9 ∧
  0 ≤ tens ∧ tens ≤ 9 ∧
  units ≥ 3 * tens

theorem count_valid_numbers : 
  (Finset.filter is_valid_number (Finset.range 1000)).card = 198 :=
by sorry

end count_valid_numbers_l389_389402


namespace luke_first_load_clothes_l389_389865

theorem luke_first_load_clothes (total_pieces first_load_pieces pieces_per_load num_small_loads : ℕ)
  (h1 : total_pieces = 47)
  (h2 : pieces_per_load = 6)
  (h3 : num_small_loads = 5)
  (h4 : ∑ i in finset.range num_small_loads, pieces_per_load = 30) :
  first_load_pieces = total_pieces - ∑ i in finset.range num_small_loads, pieces_per_load :=
by {
  simp [h1, h2, h3, h4],
  exact 17, 
}

#eval luke_first_load_clothes 47 17 6 5 (by norm_num) (by norm_num) (by norm_num) (by norm_num) -- Output should be 17

end luke_first_load_clothes_l389_389865


namespace contains_zero_l389_389236

-- Define what constitutes a five-digit number
def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

-- Define the main proof statement
theorem contains_zero (n1 n2 : ℕ) (c1 c2 : Prop) :
  is_five_digit n1 →
  is_five_digit n2 →
  c1 ∧ c2 →
  n1 + n2 = 111111 →
  ∃ (d : ℕ), d < 10 ∧ (d = 0 ∧ (∃ p1 p2 : ℕ, n1 = p1 * 10 + d ∧ n2 = p2 * 10 + d) ∨ 
             ∃ p1 p2 : ℕ, n1 = d * (10^4) + p1 ∧ n2 = d * (10^4) + p2) :=
begin
  sorry
end

end contains_zero_l389_389236


namespace angle_adb_l389_389475

-- Define the problem conditions
variables {A B C D : Type} [Tetrahedron ABCD]

def perpendicular_base (DA ABC: PlanarRegion) :=
  DA ⊥ ABC

def perpendicular_faces (ABD BCD: PlanarRegion) :=
  ABD ⊥ BCD

def equal_lengths (BD BC : Length) :=
  BD = 2 ∧ BC = 2

def sum_of_squares_of_areas (areaDAB areaDBC areaDCA: Real) :=
  areaDAB^2 + areaDBC^2 + areaDCA^2 = 8

-- Define the theorem to prove the angle ∠ ADB is π/4
theorem angle_adb (DA ABC ABD BCD: PlanarRegion) (BD BC : Length) (areaDAB areaDBC areaDCA: Real) :
  perpendicular_base DA ABC ∧
  perpendicular_faces ABD BCD ∧
  equal_lengths BD BC ∧
  sum_of_squares_of_areas areaDAB areaDBC areaDCA → 
  ∠ ADB = π/4 :=
sorry

end angle_adb_l389_389475


namespace rectangular_plot_breadth_l389_389886

theorem rectangular_plot_breadth:
  ∀ (b l : ℝ), (l = b + 10) → (24 * b = l * b) → b = 14 :=
by
  intros b l hl hs
  sorry

end rectangular_plot_breadth_l389_389886


namespace eunji_evening_jumps_l389_389795

theorem eunji_evening_jumps (jimin_morning jimin_evening eunji_morning : ℕ) (h1 : jimin_morning = 20) (h2 : eunji_morning = 18) (h3 : jimin_evening = 14) :
  (∃ (eunji_evening : ℕ), eunji_evening > jimin_morning + jimin_evening - eunji_morning) :=
begin
  -- Prove that Eunji should jump at least 17 times in the evening.
  use 17,
  rw [h1, h2, h3],
  norm_num,
end

end eunji_evening_jumps_l389_389795


namespace count_valid_numbers_l389_389406

def is_valid_number (n : Nat) : Prop := 
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let units := n % 10
  1 ≤ hundreds ∧ hundreds ≤ 9 ∧
  0 ≤ tens ∧ tens ≤ 9 ∧
  units ≥ 3 * tens

theorem count_valid_numbers : 
  (Finset.filter is_valid_number (Finset.range 1000)).card = 198 :=
by sorry

end count_valid_numbers_l389_389406


namespace exists_distinct_integers_l389_389001

theorem exists_distinct_integers (n : ℕ) (h : n ≥ 2) :
  ∃ (a b : Fin n → ℕ), (∀ i j, i ≠ j → a i ≠ a j ∧ b i ≠ b j ∧ a i ≠ b j) ∧
                       (∑ i, a i = ∑ i, b i) ∧ 
                       (n - 1 - 1 / 1998 < ∑ i, (a i - b i) / (a i + b i) ∧ 
                        ∑ i, (a i - b i) / (a i + b i) < n - 1) :=
sorry

end exists_distinct_integers_l389_389001


namespace find_a_when_b_10_l389_389544

-- Condition: a and b are inversely proportional
def inversely_proportional (a b : ℝ) : Prop :=
  ∃ k : ℝ, a * b = k

-- Given conditions
constant a₀ b₀ : ℝ
constant h₀ : a₀ = 40
constant h₀_b : b₀ = 5
constant a b : ℝ
constant hb : b = 10
constant h_inv_prop : inversely_proportional a₀ b₀

-- Proof problem statement in Lean 4
theorem find_a_when_b_10 : a = 20 :=
  sorry

end find_a_when_b_10_l389_389544


namespace compute_expression_l389_389674

theorem compute_expression:
  let a := 3
  let b := 7
  (a + b) ^ 2 + Real.sqrt (a^2 + b^2) = 100 + Real.sqrt 58 :=
by
  sorry

end compute_expression_l389_389674


namespace ink_length_figure_4_ink_length_difference_9_8_ink_length_figure_100_l389_389626

-- Define the basic conditions of the figures
def regular_pentagon (side_length : ℕ) : ℝ := 5 * side_length

-- Define ink length of a figure n
def ink_length (n : ℕ) : ℝ :=
  if n = 1 then regular_pentagon 1 else
  regular_pentagon (n-1) + (3 * (n - 1) + 2)

-- Part (a): Ink length of Figure 4
theorem ink_length_figure_4 : ink_length 4 = 38 := 
  by sorry

-- Part (b): Difference between ink length of Figure 9 and Figure 8
theorem ink_length_difference_9_8 : ink_length 9 - ink_length 8 = 29 :=
  by sorry

-- Part (c): Ink length of Figure 100
theorem ink_length_figure_100 : ink_length 100 = 15350 :=
  by sorry

end ink_length_figure_4_ink_length_difference_9_8_ink_length_figure_100_l389_389626


namespace circle_center_coordinates_l389_389695

theorem circle_center_coordinates :
  ∃ (h k : ℝ), (∀ x y : ℝ, x^2 + y^2 - 4*x - 4*y - 10 = 0 → (x - h)^2 + (y - k)^2 = 18) ∧ h = 2 ∧ k = 2 :=
begin
  sorry
end

end circle_center_coordinates_l389_389695


namespace percentage_reduction_25_percent_l389_389193

-- Define the constants
def housewife_can_obtain_more_oil (P_o P_r : ℝ) (K : ℝ) (delta_q : ℝ) : Prop :=
  δP_r := 35
  K / P_r - K / P_o = delta_q

theorem percentage_reduction_25_percent (P_o P_r : ℝ) (K : ℝ) (delta_q : ℝ) :
  P_r = δP_r →
  δP_r = 35 →
  housewife_can_obtain_more_oil P_o P_r K delta_q →
  (K = 700) →
  (delta_q = 5) →
  abs (((P_o - P_r) / P_o) * 100 - 25) < 0.01 := 
sorry

end percentage_reduction_25_percent_l389_389193


namespace contains_zero_if_sum_is_111111_l389_389221

variables (x y : ℕ)
variables (digits_x digits_y : Fin 5 → ℕ)

-- Conditions
def is_five_digit (n : ℕ) : Prop := 
  10000 ≤ n ∧ n < 100000

def differs_by_two_digit_swap (dx dy : Fin 5 → ℕ) : Prop :=
  ∃ i j : Fin 5, i ≠ j ∧ 
  (∀ k : Fin 5, (k = i → dy k = dx j) ∧ 
                (k = j → dy k = dx i) ∧ 
                (k ≠ i ∧ k ≠ j → dy k = dx k))

theorem contains_zero_if_sum_is_111111 
  (hx : is_five_digit x) (hy : is_five_digit y)
  (h_digits_x : digits_x = fun i : Fin 5 => (x / 10^(4-i) % 10))
  (h_digits_y : digits_y = fun i : Fin 5 => (y / 10^(4-i) % 10))
  (h_swap : differs_by_two_digit_swap digits_x digits_y)
  (h_sum : x + y = 111111) :
  ∃ i : Fin 5, digits_x i = 0 :=
begin
  -- Proof omitted
  sorry,
end

end contains_zero_if_sum_is_111111_l389_389221


namespace jared_annual_salary_l389_389302

def monthly_salary_diploma_holder : ℕ := 4000
def factor_degree_to_diploma : ℕ := 3
def months_in_year : ℕ := 12

theorem jared_annual_salary :
  (factor_degree_to_diploma * monthly_salary_diploma_holder) * months_in_year = 144000 :=
by
  sorry

end jared_annual_salary_l389_389302


namespace number_of_valid_3_digit_numbers_l389_389416

theorem number_of_valid_3_digit_numbers : 
  ∃ (n : ℕ), 
    (∀ (h t u : ℕ), 
      (n = h * 100 + t * 10 + u) ∧ 
      (1 ≤ h ∧ h ≤ 9) ∧ 
      (0 ≤ t ∧ t ≤ 9) ∧ 
      (0 ≤ u ∧ u ≤ 9) ∧ 
      (u ≥ 3 * t)) →
      n = 198 := 
by
  sorry

end number_of_valid_3_digit_numbers_l389_389416


namespace sum_of_triangle_perimeters_const_l389_389642

def unit_square : ℝ ∧ ℝ := (1, 1)

variables (L L' : ℝ → ℝ → Prop) (unit_distance : ∀ p ∈ L, ∀ q ∈ L', (p - q).abs = 1)

theorem sum_of_triangle_perimeters_const (A B A' B' X Y X' Y' : ℝ → ℝ → Prop)
  (square : (A = (0, 0) ∧ B = (1, 0) ∧ A' = (1, 1) ∧ B' = (0, 1)))
  (triangles_outside: (L A ∧ L X ∧ L Y ∧ L' A' ∧ L' X' ∧ L' Y'))
  : (perimeter A X Y + perimeter A' X' Y') = 2 :=
sorry

end sum_of_triangle_perimeters_const_l389_389642


namespace three_digit_number_units_digit_condition_l389_389391

theorem three_digit_number_units_digit_condition :
  let valid_numbers := (λ (n : ℕ), (100 ≤ n ∧ n ≤ 999) ∧ ((n % 10) ≥ 3 * ((n / 10) % 10))) in
  (Finset.filter valid_numbers (Finset.range 900)) = 198 :=
by
  sorry

end three_digit_number_units_digit_condition_l389_389391


namespace original_polygon_sides_l389_389777

theorem original_polygon_sides (angles_sum : ℕ) :
    angles_sum = 1620 →
    (∃ n : ℕ, n ∈ {10, 11, 12} ∧ 180 * (n - 2) = 1620 ∨ 180 * ((n+1) - 2) = 1620 ∨ 180 * ((n-1) - 2) = 1620) :=
begin
  intro h,
  use 10,
  left,
  split,
  exact 10,
  exact h,
  use 11,
  left,
  split,
  exact 11,
  exact h,
  use 12,
  left,
  split,
  exact 12,
  exact h,
  sorry
end

end original_polygon_sides_l389_389777


namespace count_different_numerators_l389_389831

def hasRepeatingDecimalForm (r : ℚ) : Prop :=
  ∃ a b : ℕ, 0 ≤ a ∧ a < 10 ∧ 0 ≤ b ∧ b < 10 ∧
    r = a * (1 / 100 + 1 / 100^2 + 1 / 100^3 + ...) + b * (1 / 10 + 1 / 10^3 + 1 / 10^5 + ...)

def setT : set ℚ := {r | r > 0 ∧ r < 1 ∧ hasRepeatingDecimalForm r}

theorem count_different_numerators : ∃ n : ℕ, n = 60 ∧
  ∀ r ∈ setT, r = (numerator r) / (99 * gcd (numerator r) 99) :=
sorry

end count_different_numerators_l389_389831


namespace find_lambda_l389_389821

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (A B C D : V) (λ : ℝ)
variables (hD: D = - (1/3 : ℝ) • (B - A) + (4/3 : ℝ) • (C - A))
variables (hλ : B - C = λ • (D - C))

theorem find_lambda (A B C D : V) (λ : ℝ) 
  (hD : D = - (1/3 : ℝ) • (B - A) + (4/3 : ℝ) • (C - A)) 
  (hλ : B - C = λ • (D - C)) 
  : λ = -3 := 
sorry

end find_lambda_l389_389821


namespace min_triangle_perimeter_l389_389322

/-- Given a point (a, b) with 0 < b < a,
    determine the minimum perimeter of a triangle with one vertex at (a, b),
    one on the x-axis, and one on the line y = x. 
    The minimum perimeter is √(2(a^2 + b^2)).
-/
theorem min_triangle_perimeter (a b : ℝ) (h : 0 < b ∧ b < a) 
  : ∃ c d : ℝ, c^2 + d^2 = 2 * (a^2 + b^2) := sorry

end min_triangle_perimeter_l389_389322


namespace three_digit_numbers_count_l389_389438

def is_3_digit_number (n : Nat) : Prop :=
  100 ≤ n ∧ n ≤ 999

def units_digit (n : Nat) : Nat :=
  n % 10

def tens_digit (n : Nat) : Nat :=
  (n / 10) % 10

def hundreds_digit (n : Nat) : Nat :=
  (n / 100) % 10

def valid_number (n : Nat) : Prop :=
  is_3_digit_number n ∧ units_digit n ≥ 3 * tens_digit n

theorem three_digit_numbers_count : 
  (Finset.card (Finset.filter valid_number (Finset.range 1000))) = 198 :=
by
  sorry

end three_digit_numbers_count_l389_389438


namespace meeting_cannot_have_65_people_l389_389572

namespace MeetingProblem

-- Define the conditions as per the problem statement
def conditions (V : Type) [Fintype V] (G : SimpleGraph V) : Prop :=
  ∀ v : V, G.degree v ≤ 8 ∧
  ∀ (u v : V), ∃ w : V, u ≠ w ∧ v ≠ w ∧ G.Adj u w ∧ G.Adj v w

-- Define the proposition to be proven: there cannot be 65 vertices under these conditions
theorem meeting_cannot_have_65_people : ∀ (V : Type) [Fintype V] (G : SimpleGraph V),
  (conditions V G) → Fintype.card V ≠ 65 :=
begin
  intros V instV G h,
  -- Detailed proof omitted
  sorry,
end

end MeetingProblem

end meeting_cannot_have_65_people_l389_389572


namespace equilateral_triangle_parabola_perimeter_l389_389464

/-- Let A, B, and C be vertices of an equilateral triangle ABC with side length 2, 
    and let A(0, √3), B(-1, 0), C(1, 0). Consider the parabolas with foci A, B, C 
    and directrices BC, CA, AB respectively. 
    Let A1, A2 be intersection points of the parabola with focus A and directrix BC with sides AB and AC.
    Let B1, B2 be intersection points of the parabola with focus B and directrix CA with sides BC and BA.
    Let C1, C2 be intersection points of the parabola with focus C and directrix AB with sides CA and CB.
    The perimeter of the triangle formed by lines A1A2, B1B2, C1C2 is 12√3 - 18.
 -/
theorem equilateral_triangle_parabola_perimeter : 
  ∀ A B C A₁ A₂ B₁ B₂ C₁ C₂ : ℝ × ℝ,
  (A = (0, sqrt 3) ∧ B = (-1, 0) ∧ C = (1, 0)) →
  -- Here we would define the parabolas' properties, intersections, etc., formally
  -- but we skip to the perimeter assertion
  -- (sum of lengths of A₁A₂, B₁B₂, and C₁C₂) = 12 * sqrt 3 - 18 :=
  sorry

end equilateral_triangle_parabola_perimeter_l389_389464


namespace reciprocal_of_36_recurring_decimal_l389_389958

-- Definitions and conditions
def recurring_decimal (x : ℚ) : Prop := x = 36 / 99

-- Theorem statement
theorem reciprocal_of_36_recurring_decimal :
  recurring_decimal (36 / 99) → (1 / (36 / 99) = 11 / 4) :=
sorry

end reciprocal_of_36_recurring_decimal_l389_389958


namespace garden_area_l389_389663

theorem garden_area (posts : Nat) (distance : Nat) (n_corners : Nat) (a b : Nat)
  (h_posts : posts = 20)
  (h_distance : distance = 4)
  (h_corners : n_corners = 4)
  (h_total_posts : 2 * (a + b) = posts)
  (h_side_relation : b + 1 = 2 * (a + 1)) :
  (distance * (a + 1 - 1)) * (distance * (b + 1 - 1)) = 336 := 
by 
  sorry

end garden_area_l389_389663


namespace Total_Cookies_is_135_l389_389085

-- Define the number of cookies in each pack
def PackA_Cookies : ℕ := 15
def PackB_Cookies : ℕ := 30
def PackC_Cookies : ℕ := 45

-- Define the number of packs bought by Paul and Paula
def Paul_PackA_Count : ℕ := 1
def Paul_PackB_Count : ℕ := 2
def Paula_PackA_Count : ℕ := 1
def Paula_PackC_Count : ℕ := 1

-- Calculate total cookies for Paul
def Paul_Cookies : ℕ := (Paul_PackA_Count * PackA_Cookies) + (Paul_PackB_Count * PackB_Cookies)

-- Calculate total cookies for Paula
def Paula_Cookies : ℕ := (Paula_PackA_Count * PackA_Cookies) + (Paula_PackC_Count * PackC_Cookies)

-- Calculate total cookies for Paul and Paula together
def Total_Cookies : ℕ := Paul_Cookies + Paula_Cookies

theorem Total_Cookies_is_135 : Total_Cookies = 135 := by
  sorry

end Total_Cookies_is_135_l389_389085


namespace three_digit_numbers_count_l389_389427

theorem three_digit_numbers_count :
  let t_range := finset.range 10
  let h_range := finset.range 9
  let valid_u t := if t * 3 ≤ 9 then 9 - (t * 3) + 1 else 0
  let total_count := h_range.card * t_range.sum (λ t, valid_u t)
  total_count = 198 :=
by
  sorry

end three_digit_numbers_count_l389_389427


namespace math_theorem_l389_389450

open Real

noncomputable def problem_statement : Prop :=
  ∀ (m : ℝ), m > 0 →
  (∃ (k : ℝ), (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ 
    (x1 - k * (x1 + m - 2 * exp 1 * x1) * (log (x1 + m) - log x1) = 0) ∧ 
    (x2 - k * (x2 + m - 2 * exp 1 * x2) * (log (x2 + m) - log x2) = 0)) →
    k ∈ Iio (-(1 / exp 1)))

theorem math_theorem : problem_statement :=
sorry

end math_theorem_l389_389450


namespace different_numerators_count_l389_389822

noncomputable def isRepeatingDecimalInTheForm_ab (r : ℚ) : Prop :=
∃ (a b : ℕ), 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 
(∃ (k : ℕ), 10 + k = 100 * r ∧ (100 * r - 10) = k ∧ (k < 100) ∧ (100 * k) % 99 = 0)

def T : set ℚ := { r : ℚ | 0 < r ∧ r < 1 ∧ isRepeatingDecimalInTheForm_ab r }

theorem different_numerators_count : 
  ∃ n, n = 63 ∧ ∀ r ∈ T, ((r.denom) % 99 ≠ 0 → n = 63) :=
sorry

end different_numerators_count_l389_389822


namespace rectangle_ratio_l389_389922

theorem rectangle_ratio (s : ℝ) (w h : ℝ) (h_cond : h = 3 * s) (w_cond : w = 2 * s) :
  h / w = 3 / 2 :=
by
  sorry

end rectangle_ratio_l389_389922


namespace perpendicular_lines_k_value_l389_389772

theorem perpendicular_lines_k_value :
  ∀ (k : ℝ), (∀ (x y : ℝ), x + 4 * y - 1 = 0) →
             (∀ (x y : ℝ), k * x + y + 2 = 0) →
             (-1 / 4 * -k = -1) →
             k = -4 :=
by
  intros k h1 h2 h3
  sorry

end perpendicular_lines_k_value_l389_389772


namespace even_condition_l389_389127

-- Definitions based on conditions
def even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

-- The function f(x) = x^2 + ax + b
def quadratic_function (a b : ℝ) : ℝ → ℝ := λ x, x^2 + a * x + b

-- The theorem statement
theorem even_condition (a b : ℝ) : even_function (quadratic_function a b) ↔ a = 0 :=
by
  sorry

end even_condition_l389_389127


namespace incenter_CEF_l389_389549

-- Definitions of geometric entities
variables {S : Type*} [metric_space S] [normed_group S] [normed_space ℝ S]
variables {O A B C D I E F : S}
variables (S : set S) (circle_S : is_circle S O)
variables (O_center : O ∈ interior S)
variables (BC_diameter : ∀ x ∈ S, x = B ∨ x = C → dist O x = radius S)
variables (A_on_S : A ∈ S)
variables (angle_AOB_lt_120 : angle O A B < π / 3)
variables (D_midpoint_arc : D ∈ S ∧ ∀ t ∈ S, t ≠ B ∧ t ≠ A ∧ dist O t = dist O D → dist A t = dist B t)
variables (line_OD_parallel_DA : ∃ line l, l ⊥ (DA) ∧ l ⊥ (OD))
variables (I_intersection : ∀ line l1 l2, (l1 || l2) → l1 ∩ l2 = {I})
variables (perpendicular_bisector_OA : ∃ line l, l ⊥ (OA) ∧ ∃ P Q, P ∈ S ∧ Q ∈ S ∧ P ≠ Q ∧ l ∩ S = {E, F})

-- Theorem statement
theorem incenter_CEF :
  incenter I (triangle C E F) :=
sorry

end incenter_CEF_l389_389549


namespace tangent_asymptotes_of_hyperbola_l389_389369

noncomputable def tangent_value_asymptotes (m n : ℝ) (h1 : m > n) (h2 : n > 0) : ℝ :=
  2 * (n / m) / (1 - (n^2 / m^2))

theorem tangent_asymptotes_of_hyperbola :
  ∀ (m n : ℝ), 
    m > n -> 
    n > 0 -> 
    (∀ (x : ℝ), ∃ y : ℝ, y = x + 1 ∧ mx^2 + ny^2 = 1) -> 
    (∀ (x1 x2 : ℝ), x1 + x2 = -(2 * n) / (m + n) -> - (x1 + x2) / 2 = -1 / 3) -> 
    tangent_value_asymptotes m n = 4 / 3 :=
by
  intros m n h1 h2 h3 h4
  sorry

end tangent_asymptotes_of_hyperbola_l389_389369


namespace point_line_plane_relationship_l389_389004

variable {P : Type}  -- P recognized as a point
variable {m : Type}  -- m recognized as a line
variable {α : Type}  -- α recognized as a plane

variable (P_m : P ∈ m)  -- P ∈ m
variable (m_α : m ⊆ α)  -- m ⊆ α

theorem point_line_plane_relationship : P ∈ m ∧ m ⊆ α := by
  sorry

end point_line_plane_relationship_l389_389004


namespace three_digit_numbers_count_l389_389425

theorem three_digit_numbers_count :
  let t_range := finset.range 10
  let h_range := finset.range 9
  let valid_u t := if t * 3 ≤ 9 then 9 - (t * 3) + 1 else 0
  let total_count := h_range.card * t_range.sum (λ t, valid_u t)
  total_count = 198 :=
by
  sorry

end three_digit_numbers_count_l389_389425


namespace similar_triangle_side_length_l389_389311

theorem similar_triangle_side_length
  (ABC_sim_DEF : Similarity ABC DEF)
  (AB_DE_ratio : AB / DE = 1 / 2)
  (BC_length : BC = 5) :
  EF = 10 :=
by 
  -- Our goal statement that EF should be 10 given the conditions
  sorry

end similar_triangle_side_length_l389_389311


namespace sum_of_series_l389_389261

open BigOperators

-- Define the sequence a(n) = 2 / (n * (n + 3))
def a (n : ℕ) : ℚ := 2 / (n * (n + 3))

-- Prove the sum of the first 20 terms of sequence a equals 10 / 9.
theorem sum_of_series : (∑ n in Finset.range 20, a (n + 1)) = 10 / 9 := by
  sorry

end sum_of_series_l389_389261


namespace triangle_BE_value_l389_389799

theorem triangle_BE_value (A B C D E : Point) (AB BC CA CD : ℝ) (h1 : AB = 12) (h2 : BC = 17) (h3 : CA = 15) (h4 : CD = 7)
  (h5 : ∠BAE = ∠CAD) : 
  BE = 952 / 181 := by 
  sorry

end triangle_BE_value_l389_389799


namespace contains_zero_l389_389230

-- Define what constitutes a five-digit number
def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

-- Define the main proof statement
theorem contains_zero (n1 n2 : ℕ) (c1 c2 : Prop) :
  is_five_digit n1 →
  is_five_digit n2 →
  c1 ∧ c2 →
  n1 + n2 = 111111 →
  ∃ (d : ℕ), d < 10 ∧ (d = 0 ∧ (∃ p1 p2 : ℕ, n1 = p1 * 10 + d ∧ n2 = p2 * 10 + d) ∨ 
             ∃ p1 p2 : ℕ, n1 = d * (10^4) + p1 ∧ n2 = d * (10^4) + p2) :=
begin
  sorry
end

end contains_zero_l389_389230


namespace probability_same_color_opposite_feet_l389_389046

/-- Define the initial conditions: number of pairs of each color. -/
def num_black_pairs : ℕ := 8
def num_brown_pairs : ℕ := 4
def num_gray_pairs : ℕ := 3
def num_red_pairs : ℕ := 1

/-- The total number of shoes. -/
def total_shoes : ℕ := 2 * (num_black_pairs + num_brown_pairs + num_gray_pairs + num_red_pairs)

theorem probability_same_color_opposite_feet :
  ((num_black_pairs * (num_black_pairs - 1)) + 
   (num_brown_pairs * (num_brown_pairs - 1)) + 
   (num_gray_pairs * (num_gray_pairs - 1)) + 
   (num_red_pairs * (num_red_pairs - 1))) * 2 / (total_shoes * (total_shoes - 1)) = 45 / 248 :=
by sorry

end probability_same_color_opposite_feet_l389_389046


namespace reciprocal_of_repeating_decimal_l389_389989

theorem reciprocal_of_repeating_decimal :
  let x := (36 : ℚ) / 99 in 1 / x = 11 / 4 :=
by
  trivial -- This is a placeholder, the actual proof is not required and hence replaced by trivial.

end reciprocal_of_repeating_decimal_l389_389989


namespace find_correct_grades_l389_389602

structure StudentGrades := 
  (Volodya: ℕ) 
  (Sasha: ℕ) 
  (Petya: ℕ)

def isCorrectGrades (grades : StudentGrades) : Prop :=
  grades.Volodya = 5 ∧ grades.Sasha = 4 ∧ grades.Petya = 3

theorem find_correct_grades (grades : StudentGrades)
  (h1 : grades.Volodya = 5 ∨ grades.Volodya ≠ 5)
  (h2 : grades.Sasha = 3 ∨ grades.Sasha ≠ 3)
  (h3 : grades.Petya ≠ 5 ∨ grades.Petya = 5)
  (unique_h1: grades.Volodya = 5 ∨ grades.Sasha = 5 ∨ grades.Petya = 5) 
  (unique_h2: grades.Volodya = 4 ∨ grades.Sasha = 4 ∨ grades.Petya = 4)
  (unique_h3: grades.Volodya = 3 ∨ grades.Sasha = 3 ∨ grades.Petya = 3) 
  (lyingCount: (grades.Volodya ≠ 5 ∧ grades.Sasha ≠ 3 ∧ grades.Petya = 5)
              ∨ (grades.Volodya = 5 ∧ grades.Sasha ≠ 3 ∧ grades.Petya ≠ 5)
              ∨ (grades.Volodya ≠ 5 ∧ grades.Sasha = 3 ∧ grades.Petya ≠ 5)) :
  isCorrectGrades grades :=
sorry

end find_correct_grades_l389_389602


namespace convert_to_rectangular_and_find_line_l389_389564

noncomputable def circle_eq1 (x y : ℝ) : Prop := x^2 + y^2 = 4 * x
noncomputable def circle_eq2 (x y : ℝ) : Prop := x^2 + y^2 + 4 * y = 0
noncomputable def line_eq (x y : ℝ) : Prop := y = -x

theorem convert_to_rectangular_and_find_line :
  (∀ x y : ℝ, circle_eq1 x y → x^2 + y^2 = 4 * x) →
  (∀ x y : ℝ, circle_eq2 x y → x^2 + y^2 + 4 * y = 0) →
  (∀ x y : ℝ, circle_eq1 x y ∧ circle_eq2 x y → line_eq x y)
:=
sorry

end convert_to_rectangular_and_find_line_l389_389564


namespace arc_length_ln_cos_l389_389659

noncomputable def arcLength (f : ℝ → ℝ) (a b : ℝ) := ∫ x in a..b, sqrt(1 + (deriv f x) ^ 2)

theorem arc_length_ln_cos :
  arcLength (λ x, - Real.log (Real.cos x)) 0 (Real.pi / 6) = Real.log (Real.sqrt 3) := by
  sorry

end arc_length_ln_cos_l389_389659


namespace boards_nailing_l389_389521

variables {x y a b : ℕ} 

theorem boards_nailing :
  (2 * x + 3 * y = 87) ∧
  (3 * a + 5 * b = 94) →
  (x + y = 30) ∧ (a + b = 30) :=
by
  sorry

end boards_nailing_l389_389521


namespace infinite_primes_in_sequence_l389_389528

def arithmetic_sequence (n : ℕ) : ℕ := 6 * n + 5

theorem infinite_primes_in_sequence : ∀ N : ℕ, ∃ n ≥ N, Nat.Prime (arithmetic_sequence n) :=
begin
  sorry
end

end infinite_primes_in_sequence_l389_389528


namespace triangle_BC_squared_l389_389802

theorem triangle_BC_squared (A B C I X Y : Point)
    (h_triangle : Triangle ABC)
    (h_AB : distance A B = 8)
    (h_AC : distance A C = 10)
    (h_incenter : Incenter I ABC)
    (h_reflect_I_AB : Reflects I AB X)
    (h_reflect_I_AC : Reflects I AC Y)
    (h_bisect_XY_AI : Bisects (Segment XY) (Segment AI)) :
    distance B C ^ 2 = 84 := 
sorry

end triangle_BC_squared_l389_389802


namespace meal_serving_ways_l389_389081

-- Define the problem conditions in Lean
def num_people := 9
def num_meals := 3
def meal_orders := {beef: 3, chicken: 3, fish: 3} -- This is a high-level overview of meal counts

-- The theorem we need to prove based on the conditions
theorem meal_serving_ways :
  let beef_meals := 3 in
  let chicken_meals := 3 in
  let fish_meals := 3 in
  -- Compute total ways to serve meals so exactly one person gets their ordered meal correctly
  compute_ways(beef_meals, chicken_meals, fish_meals, num_people) = 216 :=
begin
  sorry
end

end meal_serving_ways_l389_389081


namespace cost_of_slices_eaten_by_dog_is_correct_l389_389490

noncomputable def total_cost_before_tax : ℝ :=
  2 * 3 + 1 * 2 + 1 * 5 + 3 * 0.5 + 0.25 + 1.5 + 1.25

noncomputable def sales_tax_rate : ℝ := 0.06

noncomputable def sales_tax : ℝ := total_cost_before_tax * sales_tax_rate

noncomputable def total_cost_after_tax : ℝ := total_cost_before_tax + sales_tax

noncomputable def slices : ℝ := 8

noncomputable def cost_per_slice : ℝ := total_cost_after_tax / slices

noncomputable def slices_eaten_by_dog : ℝ := 8 - 3

noncomputable def cost_of_slices_eaten_by_dog : ℝ := cost_per_slice * slices_eaten_by_dog

theorem cost_of_slices_eaten_by_dog_is_correct : 
  cost_of_slices_eaten_by_dog = 11.59 := by
    sorry

end cost_of_slices_eaten_by_dog_is_correct_l389_389490


namespace parallel_vectors_l389_389714

variables (λ μ : ℝ)
def a := (λ + 1, 0, 2)
def b := (6, 2 * μ - 1, 2 * λ)

theorem parallel_vectors (h : a λ = b λ μ → a = (λ + 1, 0, 2) ∧ b = (6, 2 * μ - 1, 2 * λ)) :
  (λ = 2 ∨ λ = -1/3) ∧ μ = 1/2 :=
by { sorry }

end parallel_vectors_l389_389714


namespace tangent_line_a_1_monotonicity_g_slope_k_l389_389748

noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := f x + a * x ^ 2 -  (2 * a + 1) * x

-- Proof 1: Equation of tangent line at x=1 when a=1 is y=-2
theorem tangent_line_a_1 : ∀ (x : ℝ),
  g x 1 = Real.log x + x ^ 2 - 3 * x →
  ∀ x = 1, tangent line at x = 1 should be y = -2 := sorry

-- Proof 2: Monotonicity of g(x) for a > 0
theorem monotonicity_g : ∀ (a : ℝ), a > 0 →
  ∀ x : ℝ, g_derivative_sign for a > 0, discuss increase and decrease intervals := sorry
  
-- Proof 3: For a line with slope k intersecting the graph of f at points A(x1, y1) and B(x2, y2), x1 < x2 
theorem slope_k : ∀ (x1 x2 : ℝ), (x1 < x2) →
  ∀ (k : ℝ), (k = (f x2 - f x1)/(x2 - x1)) →
  (1 / x2) < k ∧ k < (1 / x1) := sorry

end tangent_line_a_1_monotonicity_g_slope_k_l389_389748


namespace determine_value_of_a_l389_389842

noncomputable def complex_number (a : ℝ) : ℂ :=
  (a / (complex.I : ℝ) + (1 - complex.I) / 2 * complex.I)

theorem determine_value_of_a (a : ℝ) :
  (∃ (y : ℝ), y = (a + 1/2) ∧ 1/2 + y = 0) → a = 0 :=
by
  intro h
  cases h with y hy
  rw [hy.right] at hy
  have h_expression := half_eq_zero_eq_snow
  linarith

end determine_value_of_a_l389_389842


namespace axes_of_symmetry_intersect_at_one_point_l389_389878

-- Statement to declare the existence of a point where all axes of symmetry intersect
theorem axes_of_symmetry_intersect_at_one_point
  (M : Type)
  [polygon M]
  (AB CD : M → M → Prop)
  (hAB_symmetry : is_axis_of_symmetry M AB)
  (hCD_symmetry : is_axis_of_symmetry M CD)
  (hAB_CD_intersect_within : ∃ P, AB P ∧ CD P) :
  ∀ (EF : M → M → Prop), is_axis_of_symmetry M EF → (∃ P, AB P ∧ CD P ∧ EF P) :=
sorry

end axes_of_symmetry_intersect_at_one_point_l389_389878


namespace total_notebooks_distributed_l389_389713

variable (S : ℕ)

theorem total_notebooks_distributed (h_half_students : (16 * (S / 2)) = (S / 2) * 16)
  (h_each_student_gets : (S / 8) * S = 16 * (S / 2)) :
  S = 64 → (S * (S / 8)) = 512 := by
  intro hS
  rw hS
  have h1 : 64 * (64 / 8) = 512 := by norm_num
  exact h1

end total_notebooks_distributed_l389_389713


namespace solve_floor_equation_l389_389099

theorem solve_floor_equation :
  exists x : ℝ, 
  let t := (15 * x - 7) / 5 in 
  t ∈ ℤ ∧ (⌊(5 + 6 * x) / 8⌋ = (15 * x - 7) / 5) ∧ 
  (x = 7 / 15 ∨ x = 4 / 5) :=
by 
  sorry

end solve_floor_equation_l389_389099


namespace sigma_condition_iff_form_l389_389305

-- Conditions: sigma function and the structure of n
def sigma (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ i, i > 0 ∧ n % i = 0).sum id

def has_at_most_two_distinct_prime_factors (n : ℕ) : Prop :=
  (n.factorization.filter (λ p_exp, p_exp > 0)).keys.length ≤ 2

-- Question converted into Lean statement
theorem sigma_condition_iff_form (n : ℕ) :
  has_at_most_two_distinct_prime_factors(n) ∧ sigma n = 2 * n - 2 ↔ ∃ k : ℕ, n = 2^k * (2^(k + 1) + 1) ∧ Nat.prime (2^(k + 1) + 1) := 
sorry

end sigma_condition_iff_form_l389_389305


namespace smallest_N_l389_389818

/--
Let r be a positive integer,
and let N be the smallest positive integer such that the numbers
  (N / (n + r)) * (fact (2 * n) / (fact n * fact n))
For n = 0, 1, 2,... are all integers. Show that
N = (r / 2) * (fact (2 * r) / (fact r * fact r)) -/
theorem smallest_N (r : ℕ) (hr : 0 < r) :
  ∃ N : ℕ, (∀ n : ℕ, (N / (n + r)) * (fact (2 * n) / (fact n * fact n)) ∈ ℕ) ∧ 
          (N = (r / 2) * (fact (2 * r) / (fact r * fact r)) :=
begin
  sorry
end

end smallest_N_l389_389818


namespace range_of_g_l389_389315

noncomputable def g (x : ℝ) := x + real.sqrt (1 - 3 * x)

theorem range_of_g : set.range g = set.Icc (1 / 3 : ℝ) 1 :=
begin
  sorry
end

end range_of_g_l389_389315


namespace circle_theorem_l389_389264

noncomputable def circle_problem (O A B C D E F M P Q : Type) (diameters : AB) (diameters2 : CD) (chord : AM) (chord2 : EF)
  (intersect_CD : ∃ P: Type, intersects P CD) (parallel_CD : ∃ Q: Type, intersects Q CD) 
  (Q_on_AM : Q ∈ AM) : Prop :=
  let AB := AB.diameter
  let AM := AM.chord
  let AO := AO.radius
  AP * AQ = (AB^3 / (4 * AM^2))

-- Circle Problem Lean Statement
theorem circle_theorem 
  (O A B C D E F M P Q : Type) 
  (diameters : AB) 
  (diameters2 : CD) 
  (chord : AM) 
  (chord2 : EF)
  (intersect_CD : ∃ P: Type, intersects P CD) 
  (parallel_CD : ∃ Q: Type, intersects Q CD) 
  (Q_on_AM : Q ∈ AM) : 
  AP * AQ = (AB^3 / (4 * AM^2)) :=
begin
  sorry
end

end circle_theorem_l389_389264


namespace boards_nailing_l389_389522

variables {x y a b : ℕ} 

theorem boards_nailing :
  (2 * x + 3 * y = 87) ∧
  (3 * a + 5 * b = 94) →
  (x + y = 30) ∧ (a + b = 30) :=
by
  sorry

end boards_nailing_l389_389522


namespace reciprocal_of_repeating_decimal_l389_389980

theorem reciprocal_of_repeating_decimal :
  let x := (36 : ℚ) / 99 in
  x⁻¹ = 11 / 4 :=
by
  have h_simplify : x = 4 / 11 := by sorry
  rw [h_simplify, inv_div]
  norm_num
  exact eq.refl (11 / 4)

end reciprocal_of_repeating_decimal_l389_389980


namespace count_valid_3_digit_numbers_l389_389377

def valid_units_digit (tens_digit units_digit : ℕ) : Prop :=
  units_digit ≥ 3 * tens_digit

def count_valid_numbers : ℕ :=
  (∑ tens_digit in {0, 1, 2, 3}, (finset.filter (valid_units_digit tens_digit) (finset.range 10)).card) * 9

theorem count_valid_3_digit_numbers : count_valid_numbers = 198 :=
by
  sorry

end count_valid_3_digit_numbers_l389_389377


namespace constant_term_expansion_l389_389948

theorem constant_term_expansion : 
  (constant_term_of_expansion (5 * x + 2 / (5 * x)) 8) = 1120 := 
by sorry

end constant_term_expansion_l389_389948


namespace find_functions_l389_389856

noncomputable def verify_function_equation (f : ℝ → ℝ) : Prop :=
∀ x y : ℝ, f(x^2) + f(x * y) = f(x) * f(y) + y * f(x) + x * f(x + y)

theorem find_functions (f : ℝ → ℝ) :
  verify_function_equation f → (f = (λ x, 0) ∨ f = (λ x, -x)) :=
by
  sorry

end find_functions_l389_389856


namespace expand_non_integer_terms_count_l389_389769

noncomputable theory

def f (x : ℝ) : ℝ := x + |x|

theorem expand_non_integer_terms_count :
  let a := 2 * ∫ x in -3..3, f x
  a = 18 →
  ∃ n : ℕ, 
    (∀ (r : ℕ), n = (∑ k in finset.range 19, 
      if (9 - (5 : ℝ) * ↑r / 6).denom ≠ 1 then 1 else 0)) ∧ 
    n = 15 := 
sorry

end expand_non_integer_terms_count_l389_389769


namespace area_of_square_adjacent_vertices_l389_389131

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem area_of_square_adjacent_vertices : 
  distance 1 3 5 6 ^ 2 = 25 :=
by
  let side_length := distance 1 3 5 6
  show side_length ^ 2 = 25
  sorry

end area_of_square_adjacent_vertices_l389_389131


namespace max_sum_arithmetic_sequence_l389_389470

noncomputable def arith_max_sum := ∀ (a₁ d : ℤ) (S : ℕ → ℤ),
  (∀ n : ℕ, S n = n * (2 * a₁ + (n - 1) * d) / 2) →
  a₁ = 50 →
  S 9 = S 17 →
  ∃ n : ℕ, S n = 91

theorem max_sum_arithmetic_sequence : arith_max_sum :=
begin
  intros a₁ d S hS ha1 hS17,
  have ha1_50 : a₁ = 50 := ha1,
  sorry
end

end max_sum_arithmetic_sequence_l389_389470


namespace vector_dot_product_l389_389309

variables (AB AC : ℝ × ℝ)
variables (BC : ℝ × ℝ)

def vec_neg (v : ℝ × ℝ) : ℝ × ℝ := (-v.1, -v.2)
def vec_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem vector_dot_product :
  AB = (4, 2) →
  AC = (1, 4) →
  BC = vec_add (vec_neg AB) AC →
  dot_product AB BC = -8 :=
by
  intros hAB hAC hBC
  rw [hAB, hAC, hBC]
  sorry

end vector_dot_product_l389_389309


namespace jill_walking_time_l389_389680

/-- 
Problem statement: Given the following conditions,
1. Dave's walking rate is 80 steps per minute.
2. Each of Dave's steps is 65 cm long.
3. Dave takes 20 minutes to reach school.
4. Jill's walking rate is 120 steps per minute.
5. Each of Jill's steps is 50 cm long.
Prove that Jill takes 17 1/3 minutes or 52/3 minutes to walk to school.
-/
theorem jill_walking_time :
  let dave_steps_per_min := 80
  let dave_step_length := 65 -- in cm
  let dave_time_to_school := 20 -- in minutes
  let jill_steps_per_min := 120
  let jill_step_length := 50 -- in cm
  let dave_speed := dave_steps_per_min * dave_step_length -- in cm/min
  let distance_to_school := dave_speed * dave_time_to_school -- in cm
  let jill_speed := jill_steps_per_min * jill_step_length -- in cm/min
  in (distance_to_school.to_rat / jill_speed.to_rat) = (52 / 3 : ℚ) :=
  sorry

end jill_walking_time_l389_389680


namespace solve_circle_chord_l389_389068

noncomputable def circle_chord_problem (ω : Type) [metric_space ω] 
  (A B P M C : ω) (x : ℝ) : Prop :=
  let chord_AB := dist A B,
      segment_MP := dist M P,
      segment_AP := dist A P,
      radius := dist M C in
  ¬ midpoint A B P → 
  segment_AP = x + 2 ∧ 
  segment_MP = x + 1 ∧ 
  dist A C = x →
  (midpoint M P M) →
  dist P B = x + 2

theorem solve_circle_chord (ω : Type) [metric_space ω] (A B P M C : ω) (x : ℝ) :
  circle_chord_problem ω A B P M C x :=
by 
  sorry

end solve_circle_chord_l389_389068


namespace trig_fraction_simplification_l389_389853

theorem trig_fraction_simplification :
  let c := 2 * Real.pi / 13 in
  (sin (4 * c) * sin (8 * c) * sin (12 * c) * sin (16 * c) * sin (20 * c)) / 
  (sin (2 * c) * sin (4 * c) * sin (6 * c) * sin (8 * c) * sin (10 * c)) = 1 :=
by
  sorry

end trig_fraction_simplification_l389_389853


namespace min_cut_length_l389_389156

theorem min_cut_length (x : ℝ) (h_longer : 23 - x ≥ 0) (h_shorter : 15 - x ≥ 0) :
  23 - x ≥ 2 * (15 - x) → x ≥ 7 :=
by
  sorry

end min_cut_length_l389_389156


namespace shifted_function_is_g_l389_389364

noncomputable def f (ω x : ℝ) := sqrt 3 * sin (ω * x) + cos (ω * x)

noncomputable def min_period (f : ℝ → ℝ) := π

noncomputable def g (x : ℝ) := 2 * cos (2 * x)

theorem shifted_function_is_g :
  ∀ (ω : ℝ), ω > 0 → min_period (f ω) = π →
  (∀ x, f 1 (x + π / 6) = g x) :=
by
  intros ω hω hT
  sorry

end shifted_function_is_g_l389_389364


namespace constant_term_expansion_l389_389950

-- auxiliary definitions and facts
def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def term_constant (n k : ℕ) (a b x : ℂ) : ℂ :=
  binomial_coeff n k * (a * x)^(n-k) * (b / x)^k

-- main theorem statement
theorem constant_term_expansion : ∀ (x : ℂ), (term_constant 8 4 (5 : ℂ) (2 : ℂ) x).re = 1120 :=
by
  intro x
  sorry

end constant_term_expansion_l389_389950


namespace original_soldiers_eq_136_l389_389921

-- Conditions
def original_soldiers (n : ℕ) : ℕ := 8 * n
def after_adding_120 (n : ℕ) : ℕ := original_soldiers n + 120
def after_removing_120 (n : ℕ) : ℕ := original_soldiers n - 120

-- Given that both after_adding_120 n and after_removing_120 n are perfect squares.
def is_square (m : ℕ) : Prop := ∃ k : ℕ, k * k = m

-- Theorem statement
theorem original_soldiers_eq_136 : ∃ n : ℕ, original_soldiers n = 136 ∧ 
                                   is_square (after_adding_120 n) ∧ 
                                   is_square (after_removing_120 n) :=
sorry

end original_soldiers_eq_136_l389_389921


namespace total_bushels_needed_l389_389275

def cows := 5
def sheep := 4
def chickens := 8
def pigs := 6
def horses := 2

def cow_bushels := 3.5
def sheep_bushels := 1.75
def chicken_bushels := 1.25
def pig_bushels := 4.5
def horse_bushels := 5.75

theorem total_bushels_needed
  (cows : ℕ) (sheep : ℕ) (chickens : ℕ) (pigs : ℕ) (horses : ℕ)
  (cow_bushels: ℝ) (sheep_bushels: ℝ) (chicken_bushels: ℝ) (pig_bushels: ℝ) (horse_bushels: ℝ) :
  cows * cow_bushels + sheep * sheep_bushels + chickens * chicken_bushels + pigs * pig_bushels + horses * horse_bushels = 73 :=
by
  -- Skipping the proof
  sorry

end total_bushels_needed_l389_389275


namespace largest_angle_in_triangle_l389_389013

open Real

theorem largest_angle_in_triangle
  (A B C : ℝ)
  (h : sin A / sin B / sin C = 1 / sqrt 2 / sqrt 5) :
  A ≤ B ∧ B ≤ C → C = 3 * π / 4 :=
by
  sorry

end largest_angle_in_triangle_l389_389013


namespace main_theorem_l389_389337

-- Definitions based on conditions
variable {Ω : Type} [ProbabilitySpace Ω]

-- X is a binomial random variable.
def X : Ω → ℕ := sorry

-- Y is a random variable such that X + Y = 8.
def Y : Ω → ℕ := λ ω => 8 - X ω

axiom X_binomial : ∀ ω, ∃ m : ℕ, X ω ~ binomial m 0.6
axiom X_binomial_params : ∀ ω, X ω ~ binomial 10 0.6

-- Expected value and variance properties of X.
noncomputable def E_X : ℝ := 6
noncomputable def D_X : ℝ := 2.4

-- Expected value of Y.
noncomputable def E_Y : ℝ := 8 - E_X

-- The final statement to prove
theorem main_theorem : D_X + E_Y = 4.4 := by
  sorry

end main_theorem_l389_389337


namespace equal_utilities_l389_389048

-- Conditions
def utility (juggling coding : ℕ) : ℕ := juggling * coding

def wednesday_utility (s : ℕ) : ℕ := utility s (12 - s)
def thursday_utility (s : ℕ) : ℕ := utility (6 - s) (s + 4)

-- Theorem
theorem equal_utilities (s : ℕ) (h : wednesday_utility s = thursday_utility s) : s = 12 / 5 := 
by sorry

end equal_utilities_l389_389048


namespace general_term_sequence_l389_389798

theorem general_term_sequence (a : ℕ → ℕ) (h₁ : a 1 = 1) (h₂ : ∀ n, a (n + 1) = a n + 3^n) :
  ∀ n, a n = (3^n - 1) / 2 := 
by
  sorry

end general_term_sequence_l389_389798


namespace time_to_cross_pole_l389_389478

-- Define necessary units and operations
def km_per_hr_to_m_per_s (v : ℝ) : ℝ :=
  v * 1000 / 3600

-- The given conditions
def train_length : ℝ := 500  -- in meters

def train_speed_km_per_hr : ℝ := 350  -- in km/hr

-- Convert the speed to m/s
def train_speed_m_per_s : ℝ := km_per_hr_to_m_per_s train_speed_km_per_hr

-- Prove the required time
theorem time_to_cross_pole :
  train_length / train_speed_m_per_s ≈ 5.14 :=
by
  unfold train_length train_speed_m_per_s
  sorry

end time_to_cross_pole_l389_389478


namespace lake_glacial_monoliths_count_l389_389649

theorem lake_glacial_monoliths_count :
  ∀ (total_monoliths sandy_loam_proportion marine_loam_proportion : ℚ)
  (sandy_loam_not_marine total_monoliths_near_300 : bool),
  total_monoliths_near_300 = tt →
  total_monoliths = 296 →
  sandy_loam_proportion = 1/8 →
  marine_loam_proportion = 22/37 →
  sandy_loam_not_marine = tt →
  let sandy_loams := total_monoliths * sandy_loam_proportion,
      loams := total_monoliths - sandy_loams,
      marine_loams := loams * marine_loam_proportion,
      lake_glacial_clays := loams - marine_loams,
      total_lake_glacial := sandy_loams + lake_glacial_clays
  in total_lake_glacial = 142 :=
by
  intros
  sorry

end lake_glacial_monoliths_count_l389_389649


namespace different_numerators_count_l389_389825

noncomputable def isRepeatingDecimalInTheForm_ab (r : ℚ) : Prop :=
∃ (a b : ℕ), 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 
(∃ (k : ℕ), 10 + k = 100 * r ∧ (100 * r - 10) = k ∧ (k < 100) ∧ (100 * k) % 99 = 0)

def T : set ℚ := { r : ℚ | 0 < r ∧ r < 1 ∧ isRepeatingDecimalInTheForm_ab r }

theorem different_numerators_count : 
  ∃ n, n = 63 ∧ ∀ r ∈ T, ((r.denom) % 99 ≠ 0 → n = 63) :=
sorry

end different_numerators_count_l389_389825


namespace part_I_part_II_solution_set_l389_389366

noncomputable section

open Set

def f (a b x : ℝ) := a * x^2 - (a + 1) * x + 1 - b

theorem part_I {b : ℝ} :
  (∀ x ∈ Icc 1 3, (f 1 b x) / x ≥ 6) ↔ b ≤ -14 :=
by
  sorry

theorem part_II_solution_set (a : ℝ) :
  let s := {x : ℝ | f a 0 x < 0}
  if h₀ : a < 0 then
    s = Ioo (-(⊤ : ℝ)) (1/a) ∪ Ioo 1 ⊤
  else if h₁ : a = 0 then
    s = Ioc 1 ⊤
  else if h₂ : 0 < a ∧ a < 1 then
    s = Ioo 1 (1/a)
  else if h₃ : a = 1 then
    s = ∅
  else if h₄ : a > 1 then
    s = Ioo (1/a) 1
  else
    False :=
by
  sorry

end part_I_part_II_solution_set_l389_389366


namespace garden_area_l389_389664

theorem garden_area (posts : Nat) (distance : Nat) (n_corners : Nat) (a b : Nat)
  (h_posts : posts = 20)
  (h_distance : distance = 4)
  (h_corners : n_corners = 4)
  (h_total_posts : 2 * (a + b) = posts)
  (h_side_relation : b + 1 = 2 * (a + 1)) :
  (distance * (a + 1 - 1)) * (distance * (b + 1 - 1)) = 336 := 
by 
  sorry

end garden_area_l389_389664


namespace count_valid_3_digit_numbers_l389_389380

def valid_units_digit (tens_digit units_digit : ℕ) : Prop :=
  units_digit ≥ 3 * tens_digit

def count_valid_numbers : ℕ :=
  (∑ tens_digit in {0, 1, 2, 3}, (finset.filter (valid_units_digit tens_digit) (finset.range 10)).card) * 9

theorem count_valid_3_digit_numbers : count_valid_numbers = 198 :=
by
  sorry

end count_valid_3_digit_numbers_l389_389380


namespace oldest_sister_clothing_l389_389080

-- Define the initial conditions
def Nicole_initial := 10
def First_sister := Nicole_initial / 2
def Next_sister := Nicole_initial + 2
def Nicole_end := 36

-- Define the proof statement
theorem oldest_sister_clothing : 
    (First_sister + Next_sister + Nicole_initial + x = Nicole_end) → x = 9 :=
by
  sorry

end oldest_sister_clothing_l389_389080


namespace numerators_required_l389_389829

def is_valid_numerator (n : Nat) : Prop :=
  (n % 3 ≠ 0) ∧ (n % 11 ≠ 0)

def numerators_count : Nat :=
  (Finset.range 100).filter (λ n, n > 0 ∧ is_valid_numerator n).card

theorem numerators_required : numerators_count = 60 :=
  sorry

end numerators_required_l389_389829


namespace log_base_5_of_3100_l389_389941

theorem log_base_5_of_3100 (h1 : 5^4 < 3100) (h2 : 3100 < 5^5) (h3 : 5^5 = 3125) : Int.round (Real.logb 5 3100) = 5 :=
by
  sorry

end log_base_5_of_3100_l389_389941


namespace olivia_grocery_cost_l389_389868

theorem olivia_grocery_cost :
  let cost_bananas := 12
  let cost_bread := 9
  let cost_milk := 7
  let cost_apples := 14
  cost_bananas + cost_bread + cost_milk + cost_apples = 42 :=
by
  rfl

end olivia_grocery_cost_l389_389868


namespace three_digit_number_units_digit_condition_l389_389392

theorem three_digit_number_units_digit_condition :
  let valid_numbers := (λ (n : ℕ), (100 ≤ n ∧ n ≤ 999) ∧ ((n % 10) ≥ 3 * ((n / 10) % 10))) in
  (Finset.filter valid_numbers (Finset.range 900)) = 198 :=
by
  sorry

end three_digit_number_units_digit_condition_l389_389392


namespace pos_relationships_lines_l389_389030

theorem pos_relationships_lines (L1 L2 : ℝ → ℝ → Prop) (hL1_plane : ∀ x y, L1 x y → plane x y)
    (hL2_plane : ∀ x y, L2 x y → plane x y) : 
    (∃ x y, L1 x y ∧ L2 x y) ∨ (∀ x y, (L1 x y → ¬ L2 x y) ∧ (L2 x y → ¬ L1 x y)) :=
by 
  sorry

end pos_relationships_lines_l389_389030


namespace three_digit_numbers_count_l389_389433

def is_3_digit_number (n : Nat) : Prop :=
  100 ≤ n ∧ n ≤ 999

def units_digit (n : Nat) : Nat :=
  n % 10

def tens_digit (n : Nat) : Nat :=
  (n / 10) % 10

def hundreds_digit (n : Nat) : Nat :=
  (n / 100) % 10

def valid_number (n : Nat) : Prop :=
  is_3_digit_number n ∧ units_digit n ≥ 3 * tens_digit n

theorem three_digit_numbers_count : 
  (Finset.card (Finset.filter valid_number (Finset.range 1000))) = 198 :=
by
  sorry

end three_digit_numbers_count_l389_389433


namespace domain_of_function_l389_389896

open Real

noncomputable def domain (f : ℝ → ℝ) : Set ℝ :=
  {x : ℝ | ∃ y : ℝ, f x = y}

theorem domain_of_function :
  domain (λ x : ℝ, 1 / (1 - sin x)) = {x : ℝ | ∀ k : ℤ, x ≠ (π / 2 + 2 * k * π)} :=
by
  sorry

end domain_of_function_l389_389896


namespace determine_k_l389_389901

theorem determine_k (k : ℝ) (h1 : ∃ x y : ℝ, y = 4 * x + 3 ∧ y = -2 * x - 25 ∧ y = 3 * x + k) : k = -5 / 3 := by
  sorry

end determine_k_l389_389901


namespace work_completion_time_l389_389168

theorem work_completion_time (B_rate A_rate Combined_rate : ℝ) (B_time : ℝ) :
  (B_rate = 1 / 60) →
  (A_rate = 4 * B_rate) →
  (Combined_rate = A_rate + B_rate) →
  (B_time = 1 / Combined_rate) →
  B_time = 12 :=
by sorry

end work_completion_time_l389_389168


namespace quadratic_function_unique_l389_389324

noncomputable def f (x : ℝ) : ℝ := (1/4 : ℝ) * x^2 + (1/2 : ℝ) * x + (1/4 : ℝ)

theorem quadratic_function_unique (f : ℝ → ℝ)
  (h1 : f (-1) = 0)
  (h2 : ∀ x, x ≤ f x ∧ f x ≤ (1 + x^2) / 2) :
  f = λ x, (1/4 : ℝ) * x^2 + (1/2 : ℝ) * x + (1/4 : ℝ) :=
by
  sorry

end quadratic_function_unique_l389_389324


namespace allergic_reaction_probability_is_50_percent_l389_389039

def can_have_allergic_reaction (choice : String) : Prop :=
  choice = "peanut_butter"

def percentage_of_allergic_reaction :=
  let total_peanut_butter := 40 + 30
  let total_cookies := 40 + 50 + 30 + 20
  (total_peanut_butter : Float) / (total_cookies : Float) * 100

theorem allergic_reaction_probability_is_50_percent :
  percentage_of_allergic_reaction = 50 := sorry

end allergic_reaction_probability_is_50_percent_l389_389039


namespace probability_5_6_l389_389353

noncomputable def X : ℝ → ℝ := sorry -- Define the random variable X

theorem probability_5_6 (μ σ : ℝ) (h1 : μ = 4) (h2 : σ = 1)
  (h3 : ∀ a b : ℝ, P (a < X ≤ b) = (1 / (σ * sqrt (2 * π))) * ∫ c in a..b, exp (-(c - μ)^2 / (2 * σ^2)) )
  (h4 : P((μ - 2 * σ) < X ≤ (μ + 2 * σ)) = 0.9544)
  (h5 : P((μ - σ) < X ≤ (μ + σ)) = 0.6826) :
  P(5 < X < 6) = 0.1359 :=
sorry

end probability_5_6_l389_389353


namespace xy_zero_l389_389579

theorem xy_zero (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 162) : x * y = 0 :=
by
  sorry

end xy_zero_l389_389579


namespace circumscribed_cone_volume_l389_389890

-- Define the given conditions for the pyramid
variables (a : ℝ) (α β : ℝ)

-- Define the circumcircle radius R and the height SO in terms of a, α, β
def circumcircle_radius (a α : ℝ) : ℝ :=
  a / (2 * real.cos (α / 2))

def pyramid_height (a α β : ℝ) : ℝ :=
  (a * real.cos α * real.tan β) / (2 * real.cos (α / 2))

-- Definition of the cone volume using R and height
def cone_volume (a α β : ℝ) : ℝ :=
  let R := circumcircle_radius a α in
  let h := pyramid_height a α β in
  (1 / 3) * real.pi * R^2 * h

-- The main theorem to state the volume of the circumscribed cone
theorem circumscribed_cone_volume : cone_volume a α β = (real.pi * a^3 * real.cos α * real.tan β) / (24 * (real.cos (α / 2))^3) :=
sorry

end circumscribed_cone_volume_l389_389890


namespace rectangle_area_l389_389600

theorem rectangle_area (b l : ℕ) (h1: l = 3 * b) (h2: 2 * (l + b) = 120) : l * b = 675 := by
  sorry

end rectangle_area_l389_389600


namespace reciprocal_of_repeating_decimal_l389_389961

theorem reciprocal_of_repeating_decimal :
  let x := 0.36363636 -- simplified as .\overline{36}
  ∃ y : ℚ, x = 4 / 11 ∧ y = 1 / x ∧ y = 11 / 4 :=
by
  sorry

end reciprocal_of_repeating_decimal_l389_389961


namespace reciprocal_of_repeating_decimal_l389_389973

theorem reciprocal_of_repeating_decimal :
  let x : ℚ := 36 / 99 in
  1 / x = 11 / 4 :=
by
  sorry

end reciprocal_of_repeating_decimal_l389_389973


namespace shape_is_cone_l389_389781

variables {x y z a b c d θ : ℝ}

def equation (x y z a b c d θ : ℝ) : Prop :=
  (x - a)^2 + (y - b)^2 + (z - c)^2 = (d * cos θ)^2

theorem shape_is_cone (d : ℝ) (h_d : 0 < d) :
  ∀ a b c, 
  equation x y z a b c d θ → 
  a = 0 → 
  b = 0 → 
  c = 0 → 
  is_cone (x y z θ) :=
by
  sorry

end shape_is_cone_l389_389781


namespace two_planes_perpendicular_to_same_plane_are_parallel_l389_389646

-- Statement of the proposition 
theorem two_planes_perpendicular_to_same_plane_are_parallel
    (planes_parallel : ∀ (P Q : Plane), (P || Q) ↔ ∀ L, (proj_L P = proj_L Q))  -- condition (a)
    (line_plane_rel : ∀ (L : Line) (P Q : Plane), ((L ⊥ P ∧ L ⊥ Q) → (P || Q)))  -- condition (b)
    (perpendicular_to_same_plane : ∀ (P Q R : Plane), (P ⊥ R ∧ Q ⊥ R) → (P || Q))  -- condition (c)
    : (∀ (P Q : Plane) (R : Plane), (P ⊥ R ∧ Q ⊥ R) → (P || Q)) :=  -- prove statement D
begin
  sorry
end

end two_planes_perpendicular_to_same_plane_are_parallel_l389_389646


namespace sqrt_a_add_sqrt_b_max_value_l389_389773

theorem sqrt_a_add_sqrt_b_max_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + b = 2) :
  sqrt a + sqrt b ≤ 2 :=
by
  sorry

end sqrt_a_add_sqrt_b_max_value_l389_389773


namespace contains_zero_l389_389253

open Nat

theorem contains_zero 
  (x y : ℕ)
  (hx : 10000 ≤ x ∧ x < 100000)
  (hy : 10000 ≤ y ∧ y < 100000)
  (h_swap : ∃ (i j : ℕ), i ≠ j ∧ swap_digits x i j = y)
  (h_sum : x + y = 111111) :
  (∃ i, digit x i = 0) ∨ (∃ j, digit y j = 0) :=
sorry

-- Swap two digits in a number
def swap_digits (n : ℕ) (i j : ℕ) : ℕ := 
  -- Dummy definition for the proof placeholder
  n -- replace with actual implementation

-- Extract the digit at position i
def digit (n : ℕ) (i : ℕ) : ℕ :=
  -- Dummy definition for the proof placeholder
  0 -- replace with actual implementation

end contains_zero_l389_389253


namespace cos_theta_plus_5π_div_6_l389_389739

theorem cos_theta_plus_5π_div_6 (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < π / 2)
  (hcond : Real.sin (θ / 2 + π / 6) = 3 / 5) :
  Real.cos (θ + 5 * π / 6) = -24 / 25 :=
by
  sorry -- Proof is skipped as instructed

end cos_theta_plus_5π_div_6_l389_389739


namespace find_r_from_tan_cosine_tangent_l389_389771

theorem find_r_from_tan_cosine_tangent 
  (θ : ℝ) 
  (r : ℝ) 
  (htan : Real.tan θ = -7 / 24) 
  (hquadrant : π / 2 < θ ∧ θ < π) 
  (hr : 100 * Real.cos θ = r) : 
  r = -96 := 
sorry

end find_r_from_tan_cosine_tangent_l389_389771


namespace solve_for_k_l389_389316

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)
variables (k : ℝ)

-- given conditions
def condition1 : norm a = 2 := sorry
def condition2 : norm b = 4 := sorry
def condition3 : real.angle a b = real.pi / 3 := sorry -- 60 degrees

-- the dot product of a and b using angle
def dot_product : a ⬝ b = 4 :=
begin
  rw [inner_product_space.norm_eq_norm, 
      inner_product_space.norm_eq_norm, 
      ← real_inner_eq_norm_mul_cos_angle],
  norm_num,
  rw real.cos_pi_div_three,
  norm_num,
end

-- the condition for perpendicular vectors
def perpendicular_condition : (a + 3 • b) ⬝ (k • a - b) = 0 :=
begin
  calc (a + 3 • b) ⬝ (k • a - b)
     = a ⬝ (k • a) + a ⬝ (-b) + (3 • b) ⬝ (k • a) + (3 • b) ⬝ (-b) : by simp [inner_add_left, inner_add_right]
...  = k * (a ⬝ a) + 3 * k * (b ⬝ a) + -1 * (a ⬝ b) + 3 * (b ⬝ b) * (-1) : by norm_num
...  = sorry, -- This will use dot_product, expanding a ⬝ a and b ⬝ b as well.

-- prove k = 13 / 4
theorem solve_for_k : k = 13 / 4 :=
begin
  have h1 : dot_product k a b = 4,
  rw [condition1, condition2, condition3],

  -- Expand and solve for k
  have h := perpendicular_condition k a b,
  sorry -- Fill in the detailed arithmetic steps.
end

end solve_for_k_l389_389316


namespace compute_t_f_4_l389_389501

def t (x : ℝ) : ℝ := real.sqrt (4 * x + 4)
def f (x : ℝ) : ℝ := 4 - t(x)

theorem compute_t_f_4 : t(f(4)) = real.sqrt (20 - 8 * real.sqrt 5) :=
by sorry

end compute_t_f_4_l389_389501


namespace sixty_different_numerators_l389_389834

theorem sixty_different_numerators : 
  (Finset.card {ab : ℕ | 1 ≤ ab ∧ ab ≤ 99 ∧ Nat.gcd ab 99 = 1} = 60) :=
sorry

end sixty_different_numerators_l389_389834


namespace melanie_correct_coins_and_value_l389_389866

def melanie_coins_problem : Prop :=
let dimes_initial := 19
let dimes_dad := 39
let dimes_sister := 15
let dimes_mother := 25
let total_dimes := dimes_initial + dimes_dad + dimes_sister + dimes_mother

let nickels_initial := 12
let nickels_dad := 22
let nickels_sister := 7
let nickels_mother := 10
let nickels_grandmother := 30
let total_nickels := nickels_initial + nickels_dad + nickels_sister + nickels_mother + nickels_grandmother

let quarters_initial := 8
let quarters_dad := 15
let quarters_sister := 12
let quarters_grandmother := 3
let total_quarters := quarters_initial + quarters_dad + quarters_sister + quarters_grandmother

let dimes_value := total_dimes * 0.10
let nickels_value := total_nickels * 0.05
let quarters_value := total_quarters * 0.25
let total_value := dimes_value + nickels_value + quarters_value

total_dimes = 98 ∧ total_nickels = 81 ∧ total_quarters = 38 ∧ total_value = 23.35

theorem melanie_correct_coins_and_value : melanie_coins_problem :=
by sorry

end melanie_correct_coins_and_value_l389_389866


namespace constant_term_expansion_l389_389949

theorem constant_term_expansion : 
  (constant_term_of_expansion (5 * x + 2 / (5 * x)) 8) = 1120 := 
by sorry

end constant_term_expansion_l389_389949


namespace CauchySchwarz_inequality_inequality_sqrt_l389_389735

variable {n : ℕ}
variable {a : Fin n → ℝ}
variable {x : ℝ}

theorem CauchySchwarz_inequality (h1 : ∀ i, a i ∈ ℝ) : (∑ i, a i)^2 ≤ n * (∑ i, (a i)^2) := 
by
  sorry
  
theorem inequality_sqrt (h2 : 3/2 ≤ x ∧ x ≤ 5) : 2 * Real.sqrt (x + 1) + Real.sqrt (2 * x - 3) + Real.sqrt (15 - 3 * x) < 2 * Real.sqrt (19) := 
by
  sorry

end CauchySchwarz_inequality_inequality_sqrt_l389_389735


namespace perimeter_quadrilateral_ABCD_l389_389028

-- The conditions in Lean
noncomputable def is_right_angle (θ : ℝ) := θ = 90
noncomputable def angle_60_deg (θ : ℝ) := θ = 60

variables (A B C D E : Type)
-- Angles:
variable (angle_AEB : ℝ)
variable (angle_BEC : ℝ)
variable (angle_CED : ℝ)
-- Length:
variable (length_AE : ℝ)
-- Functions to calculate the lengths based on hypotenuse properties
noncomputable def length_side_60 (hypo : ℝ) : ℝ := hypo * (√3 / 2)
noncomputable def length_side_30 (hypo : ℝ) : ℝ := hypo / 2

-- Hypotheses based on problem statement
axiom h1 : is_right_angle angle_AEB
axiom h2 : is_right_angle angle_BEC
axiom h3 : is_right_angle angle_CED
axiom h4 : angle_60_deg angle_AEB
axiom h5 : angle_60_deg angle_BEC
axiom h6 : angle_60_deg angle_CED
axiom h7 : length_AE = 36

-- Proving the perimeter
theorem perimeter_quadrilateral_ABCD :
  let AE := length_AE,
      AB := length_side_60 AE,
      BE := length_side_30 AE,
      BC := length_side_60 BE,
      CE := length_side_30 BE,
      CD := length_side_60 CE,
      DE := length_side_30 CE,
      DA := DE + AE
  in AB + BC + CD + DA = 40.5 + 31.5 * √3 := by
        sorry

end perimeter_quadrilateral_ABCD_l389_389028


namespace proof_of_solveForAlpha_proof_of_all_steps_l389_389358

noncomputable def solveForAlpha (α : ℝ) : Prop :=
  (1 - 2 * Real.cos α ^ 2) / 
  (2 * Real.tan (2 * α - Real.pi / 4) * Real.sin (Real.pi / 4 + 2 * α) ^ 2) = 1

-- Additional lean4 mathematics
def solveStep1 (α : ℝ) : Prop :=
  (1 - 2 * Real.cos α ^ 2) = -Real.cos (2 * α)

def solveStep2 (α : ℝ) : Prop :=
  Real.tan(2 * α - Real.pi / 4) = Real.sin(2 * α - Real.pi / 4) / Real.cos(2 * α - Real.pi / 4)

def solveStep3 (α : ℝ) : Prop :=
  Real.sin (2 * α - Real.pi / 4) = (Real.sqrt 2 / 2) * (Real.sin 2 * α - Real.cos 2 * α)

def solveStep4 (α : ℝ) : Prop :=
  Real.cos (2 * α - Real.pi / 4) = (Real.sqrt 2 / 2) * (Real.cos 2 * α + Real.sin 2 * α)

def solveStep5 (α : ℝ) : Prop :=
  Real.sin(Real.pi / 4 + 2 * α) ^ 2 = (1 / 2) * (Real.cos 2 * α + Real.sin 2 * α) ^ 2

theorem proof_of_solveForAlpha (α : ℝ) : solveForAlpha α :=
by
  sorry

-- Define the proof step based on simplifications
theorem proof_of_all_steps (α : ℝ) :
  solveStep1 α ∧ solveStep2 α ∧ solveStep3 α ∧ solveStep4 α ∧ solveStep5 α → solveForAlpha α :=
by
  intros hstep1 hstep2 hstep3 hstep4 hstep5
  sorry

end proof_of_solveForAlpha_proof_of_all_steps_l389_389358


namespace friends_in_group_l389_389621

theorem friends_in_group (n : ℕ) 
  (avg_before_increase : ℝ := 800) 
  (avg_after_increase : ℝ := 850) 
  (individual_rent_increase : ℝ := 800 * 0.25) 
  (original_rent : ℝ := 800) 
  (new_rent : ℝ := 1000)
  (original_total : ℝ := avg_before_increase * n) 
  (new_total : ℝ := original_total + individual_rent_increase):
  new_total = avg_after_increase * n → 
  n = 4 :=
by
  sorry

end friends_in_group_l389_389621


namespace position_of_point_l389_389007

theorem position_of_point (a b : ℝ) (h_tangent: (a ≠ 0 ∨ b ≠ 0) ∧ (a^2 + b^2 = 1)) : a^2 + b^2 = 1 :=
by
  sorry

end position_of_point_l389_389007


namespace complex_quadrant_l389_389550

noncomputable def z : ℂ := 5 / (complex.i + 2)

theorem complex_quadrant :
  (z.re > 0) ∧ (z.im < 0) := by
sorry

end complex_quadrant_l389_389550


namespace variance_combined_classes_l389_389669

-- Define the conditions as parameters/variables
variables (nA nB : ℕ) (meanA meanB varA varB : ℝ)
-- Specify the given values
#check (h₀ : nA = 50)
#check (h₁ : nB = 40)
#check (h₂ : meanA = 76)
#check (h₃ : meanB = 85)
#check (h₄ : varA = 96)
#check (h₅ : varB = 60)

-- Define combined counts, means, and the final variance to be proven
def combinedMean (nA nB : ℕ) (meanA meanB : ℝ) : ℝ :=
  (nA * meanA + nB * meanB) / (nA + nB)

noncomputable def combinedVariance (nA nB : ℕ) (meanA meanB : ℝ) (varA varB : ℝ) : ℝ :=
  let meanC := combinedMean nA nB meanA meanB in
  (nA * (varA + (meanA - meanC)^2) + nB * (varB + (meanB - meanC)^2)) / (nA + nB)

theorem variance_combined_classes (h₀ : nA = 50)
                                  (h₁ : nB = 40)
                                  (h₂ : meanA = 76)
                                  (h₃ : meanB = 85)
                                  (h₄ : varA = 96)
                                  (h₅ : varB = 60) :
  combinedVariance 50 40 76 85 96 60 = 100 := 
sorry

end variance_combined_classes_l389_389669


namespace square_and_circle_area_l389_389330

open Real

/-- Proof problem: given four circles each with a radius of 3 inches arranged
    in a square such that there is no gap between them, prove that the 
    area of the square is 144 square inches and the total area covered
    by the circles inside the square is 36π square inches. -/
theorem square_and_circle_area :
  ∀ (r : ℝ) (num_circles : ℕ),
  r = 3 ∧ num_circles = 4 →
  let d := 2 * r in
  let side_length := 2 * d in
  let area_square := side_length^2 in
  let area_circle := π * r^2 in
  let total_circle_area := num_circles * area_circle in
  area_square = 144 ∧ total_circle_area = 36 * π :=
by
  intros r num_circles h
  have h_radius : r = 3 := h.1
  have h_circles : num_circles = 4 := h.2
  let d := 2 * r
  let side_length := 2 * d
  let area_square := side_length^2
  let area_circle := π * r^2
  let total_circle_area := num_circles * area_circle
  sorry

end square_and_circle_area_l389_389330


namespace fraction_of_constants_l389_389093

theorem fraction_of_constants :
  ∃ a b c : ℤ, (4 : ℤ) * a * (k + b)^2 + c = 4 * k^2 - 8 * k + 16 ∧
             4 * -1 * (k + (-1))^2 + 12 = 4 * k^2 - 8 * k + 16 ∧
             a = 4 ∧ b = -1 ∧ c = 12 ∧ c / b = -12 :=
by
  sorry

end fraction_of_constants_l389_389093


namespace frog_safe_jumps_l389_389065

noncomputable def largest_safe_jump (n : ℤ) : ℤ :=
  (n - 1) / 2

theorem frog_safe_jumps (n : ℤ) (h : n ≥ 3) : ∀ k, k ≤ (largest_safe_jump n) → 
  ∃ sequence : list ℤ, list.sum sequence = 0 ∧ 
  (∀ (m : ℤ), m ∈ sequence → m ≠ 1 ∧ m ≠ 2 ∧ ... ∧ m ≠ k) :=
sorry

end frog_safe_jumps_l389_389065


namespace digit_contains_zero_l389_389226

theorem digit_contains_zero 
  (n₁ n₂ : ℕ) 
  (h₁ : 10000 ≤ n₁ ∧ n₁ ≤ 99999)
  (h₂ : 10000 ≤ n₂ ∧ n₂ ≤ 99999)
  (h₃ : ∃ (i j : ℕ), i ≠ j ∧ n₂ = n₁.swap_digits i j)
  (h₄ : n₁ + n₂ = 111111) 
  : ∃ (d : ℕ), d = 0 :=
sorry

end digit_contains_zero_l389_389226


namespace mandarin_ducks_total_l389_389920

theorem mandarin_ducks_total : (3 * 2) = 6 := by
  sorry

end mandarin_ducks_total_l389_389920


namespace necessary_but_not_sufficient_l389_389510

noncomputable def geometric_sum (a₁ q : ℝ) (n : ℕ) : ℝ :=
  a₁ * (1 - q^n) / (1 - q)

def is_nec_suff_cond (a₁ q : ℝ) : Prop :=
  (∀ (n : ℕ), q < 1 → ∃ (S : ℕ → ℝ), S n = geometric_sum a₁ q n) ∧
  (a₁ + q = 1 → lim (λn, geometric_sum a₁ q n) = 1) ∧
  ((¬ (a₁ + q = 1 → True → lim (λn, geometric_sum a₁ q n) = 1))) → False

theorem necessary_but_not_sufficient :
  ∀ a₁ q : ℝ, (a₁ + q = 1 → lim (λ n, geometric_sum a₁ q n) = 1 ∧ 
  (∃ a₁ q : ℝ, a₁ + q = 1 ∧ lim (λ n, geometric_sum a₁ q n) ≠ 1)) → 
  is_nec_suff_cond a₁ q :=
by
  sorry

end necessary_but_not_sufficient_l389_389510


namespace chord_length_progression_impossible_l389_389198

theorem chord_length_progression_impossible (k d : ℝ) (a1 ak : ℝ) 
  (h_eq_circle : ∃ (x y : ℝ), (x - 5)^2 + y^2 = 5^2) -- Circle equation rewritten
  (h_point_inside : x * x + y * y = 10 * x) -- Confirm the original circle equation form
  (h_common_diff : d ∈ set.Icc (1/3) (1/2)) -- Common difference interval
  (h_arith_seq : ∀ n : ℕ, a1 + (n - 1) * d ≤ ak) -- Arithmetic sequence constraint
  (k_eq : k = 5) : false := sorry

end chord_length_progression_impossible_l389_389198


namespace range_of_f_l389_389697

def f (x : ℝ) := x - 1 + Real.sqrt (6 * x - x ^ 2)

theorem range_of_f :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → f x ∈ Icc (-1 : ℝ) 2 :=
by 
  sorry

end range_of_f_l389_389697


namespace total_price_correct_l389_389571

-- Definitions based on given conditions
def basic_computer_price : ℝ := 2125
def enhanced_computer_price : ℝ := 2125 + 500
def printer_price (P : ℝ) := P = 1/8 * (enhanced_computer_price + P)

-- Statement to prove the total price of the basic computer and printer
theorem total_price_correct (P : ℝ) (h : printer_price P) : 
  basic_computer_price + P = 2500 :=
by
  sorry

end total_price_correct_l389_389571


namespace first_group_number_l389_389927

theorem first_group_number (n_students sample_size interval drawn_18th : ℕ) (h1 : n_students = 1000) (h2 : sample_size = 40) (h3 : interval = n_students / sample_size) (h4 : drawn_18th = 443) : 
  ∃ (x : ℕ), 443 = x + 17 * interval :=
by
  have h_interval : interval = 25 := by rw [h1, h2, Nat.div_eq_of_lt]; norm_num
  use 18
  rw [h_interval, mul_comm, Nat.mul_add, Nat.add_sub_cancel_left]
  norm_num
  refl

end first_group_number_l389_389927


namespace area_of_square_adjacent_vertices_l389_389132

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem area_of_square_adjacent_vertices : 
  distance 1 3 5 6 ^ 2 = 25 :=
by
  let side_length := distance 1 3 5 6
  show side_length ^ 2 = 25
  sorry

end area_of_square_adjacent_vertices_l389_389132


namespace jordan_rectangle_width_l389_389171

theorem jordan_rectangle_width
  (carol_length : ℕ) (carol_width : ℕ) (jordan_length : ℕ) (jordan_width : ℕ)
  (h_carol_dims : carol_length = 12) (h_carol_dims2 : carol_width = 15)
  (h_jordan_length : jordan_length = 6)
  (h_area_eq : carol_length * carol_width = jordan_length * jordan_width) :
  jordan_width = 30 := 
sorry

end jordan_rectangle_width_l389_389171


namespace three_digit_numbers_count_l389_389429

theorem three_digit_numbers_count :
  let t_range := finset.range 10
  let h_range := finset.range 9
  let valid_u t := if t * 3 ≤ 9 then 9 - (t * 3) + 1 else 0
  let total_count := h_range.card * t_range.sum (λ t, valid_u t)
  total_count = 198 :=
by
  sorry

end three_digit_numbers_count_l389_389429


namespace max_m_for_roots_difference_l389_389684

theorem max_m_for_roots_difference (m : ℝ) : 
  (∃ x1 x2 : ℝ, (x1 ≠ x2) ∧ (x1^2 + m*x1 + 6 = 0) ∧ (x2^2 + m*x2 + 6 = 0) ∧ (|x1 - x2| = √85)) → m = √109 :=
by
  sorry

end max_m_for_roots_difference_l389_389684


namespace find_all_m_l389_389817

theorem find_all_m (m : ℕ) (n : ℕ) (p : ℕ) : 
  (1000 ≤ m ∧ m ≤ 2021) ∧ 
  (mn_perfect_sq : ∃ n : ℕ, m * n = (nat.sqrt (m * n))^2) ∧ 
  (prime_diff : ∃ n : ℕ, nat.prime (m - n)) ↔ 
  (m = 1156 ∨ m = 1296 ∨ m = 1369 ∨ m = 1600 ∨ m = 1764) :=
sorry

end find_all_m_l389_389817


namespace log_5_3100_nearest_int_l389_389937

theorem log_5_3100_nearest_int :
  (4 : ℝ) < real.log 3100 / real.log 5 ∧ real.log 3100 / real.log 5 < (5 : ℝ) ∧
  abs (3100 - 3125) < abs (3100 - 625) →
  real.log 3100 / real.log 5 = 5 :=
sorry

end log_5_3100_nearest_int_l389_389937


namespace equal_perimeters_of_quadrilaterals_l389_389066

-- Definitions based on the conditions in the problem:
variables {A B C D I X Y Z T : Type*}
variables [AddCommGroup I] [Circle I]
variables (ABCD : convex_quadrilateral) (circumscribed : circumscribed_around_circle ABCD I)
variables (omega : circumcircle (triangle A C I))
variables (BA_extends_to : extends_to BA A ω X)
variables (BC_extends_to : extends_to BC C ω Z)
variables (AD_extends_to : extends_to AD D ω Y)
variables (CD_extends_to : extends_to CD D ω T)

-- Statement of the theorem:
theorem equal_perimeters_of_quadrilaterals :
  perimeter (quadrilateral A D T X) = perimeter (quadrilateral C D Y Z) :=
sorry

end equal_perimeters_of_quadrilaterals_l389_389066


namespace tangent_line_to_curve_l389_389556

-- Define the function representing the curve.
def curve (x : ℝ) : ℝ := x^3 - 2 * x

-- The point of tangency on the curve.
def point_of_tangency : ℝ × ℝ := (1, -1)

-- Define the derivative of the curve.
def derivative (x : ℝ) : ℝ := 3 * x^2 - 2

-- Define the equation of the tangent line at a given point (x_1, y_1) with slope m.
def tangent_line_equation (m x_1 y_1 x : ℝ) : ℝ := m * (x - x_1) + y_1

theorem tangent_line_to_curve :
  tangent_line_equation (derivative 1) 1 (-1) = λ x, x - 2 :=
by
  sorry

end tangent_line_to_curve_l389_389556


namespace absolute_difference_equation_l389_389107

theorem absolute_difference_equation :
  ∃ x : ℝ, (|16 - x| - |x - 12| = 4) ∧ x = 12 :=
by
  sorry

end absolute_difference_equation_l389_389107


namespace multiplication_incorrect_mod_3_l389_389034

theorem multiplication_incorrect_mod_3 :
  let a := 79133
  let b := 111107
  let c := 8794230231
  a % 3 = 2 ∧ b % 3 = 2 ∧ c % 3 = 0 → a * b ≠ c :=
by
  intro h
  let ⟨ha, hb, hc⟩ := h
  sorry

end multiplication_incorrect_mod_3_l389_389034


namespace sum_x_Q3_Q4_l389_389606

-- Define the structure of a 50-gon in the Cartesian plane
structure Polygon :=
  (vertices : Fin 50 → ℝ × ℝ)

-- Define the sum of x-coordinates of a polygon
def sum_x_coordinates (p : Polygon) : ℝ :=
  (Finset.univ).sum (λ i => (p.vertices i).1)

-- Define the midpoint polygon given a polygon
def midpoint_polygon (p : Polygon) : Polygon :=
  {
    vertices := λ i => 
      let v1 := p.vertices i
      let v2 := p.vertices ((i + 1) % 50)
      ((v1.1 + v2.1) / 2, (v1.2 + v2.2) / 2)
  }

-- Given condition of initial polygon Q1
variable {Q1 : Polygon}
axiom sum_x_Q1 : sum_x_coordinates Q1 = 2010

-- Define Q2, Q3, Q4 based on the midpoint polygon definition
def Q2 : Polygon := midpoint_polygon Q1
def Q3 : Polygon := midpoint_polygon Q2
def Q4 : Polygon := midpoint_polygon Q3

-- Proof goal: sum of x-coordinates of vertices of Q3 and Q4 is 2010 each
theorem sum_x_Q3_Q4 : sum_x_coordinates Q3 = 2010 ∧ sum_x_coordinates Q4 = 2010 :=
  sorry

end sum_x_Q3_Q4_l389_389606


namespace three_digit_number_units_digit_condition_l389_389388

theorem three_digit_number_units_digit_condition :
  let valid_numbers := (λ (n : ℕ), (100 ≤ n ∧ n ≤ 999) ∧ ((n % 10) ≥ 3 * ((n / 10) % 10))) in
  (Finset.filter valid_numbers (Finset.range 900)) = 198 :=
by
  sorry

end three_digit_number_units_digit_condition_l389_389388


namespace det_of_dilation_matrix_l389_389056

open Matrix

-- Define the 3x3 dilation matrix E
def E : Matrix (Fin 3) (Fin 3) ℝ := ![
  [3, 0, 0],
  [0, 3, 0],
  [0, 0, 3]
]

-- State the theorem
theorem det_of_dilation_matrix :
  det E = 27 :=
  by
  sorry

end det_of_dilation_matrix_l389_389056


namespace find_b_in_triangle_l389_389453

theorem find_b_in_triangle (c : ℝ) (B C : ℝ) (h1 : c = Real.sqrt 3)
  (h2 : B = Real.pi / 4) (h3 : C = Real.pi / 3) : ∃ b : ℝ, b = Real.sqrt 2 :=
by
  sorry

end find_b_in_triangle_l389_389453


namespace different_numerators_count_l389_389824

noncomputable def isRepeatingDecimalInTheForm_ab (r : ℚ) : Prop :=
∃ (a b : ℕ), 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 
(∃ (k : ℕ), 10 + k = 100 * r ∧ (100 * r - 10) = k ∧ (k < 100) ∧ (100 * k) % 99 = 0)

def T : set ℚ := { r : ℚ | 0 < r ∧ r < 1 ∧ isRepeatingDecimalInTheForm_ab r }

theorem different_numerators_count : 
  ∃ n, n = 63 ∧ ∀ r ∈ T, ((r.denom) % 99 ≠ 0 → n = 63) :=
sorry

end different_numerators_count_l389_389824


namespace ratio_of_AB_BC_l389_389457

section CircleRatio

variable {r : ℝ} (h1 : AB = 2 * r) (h2 : AC = 2 * r) (h3 : length (BC.arcminor) = r)

theorem ratio_of_AB_BC (r : ℝ) (A B C : Point) [Circle A B C r] (h1 : AB = 2 * r) (h2 : AC = 2 * r) (h3 : length (BC.arcminor) = r) :
  (AB / BC) = 2 :=
by
  sorry

end CircleRatio

end ratio_of_AB_BC_l389_389457


namespace P_union_Q_eq_Q_l389_389069

noncomputable def P : Set ℝ := {x : ℝ | x > 1}
noncomputable def Q : Set ℝ := {x : ℝ | x^2 - x > 0}

theorem P_union_Q_eq_Q : P ∪ Q = Q := by
  sorry

end P_union_Q_eq_Q_l389_389069


namespace problem_1_problem_2_l389_389750

def f (a x : ℝ) : ℝ := |a - 3 * x| - |2 + x|

theorem problem_1 (x : ℝ) : f 2 x ≤ 3 ↔ -3 / 4 ≤ x ∧ x ≤ 7 / 2 := by
  sorry

theorem problem_2 (a x : ℝ) : f a x ≥ 1 - a + 2 * |2 + x| → a ≥ -5 / 2 := by
  sorry

end problem_1_problem_2_l389_389750


namespace sum_P_n_eq_zero_l389_389063

noncomputable def P_n (n : ℕ) (x z : ℂ) : ℂ :=
  ∏ i in finset.range (n + 1), (1 - z * x^(i - 1)) / (z - x^i)

theorem sum_P_n_eq_zero {x z : ℂ} (hx : |x| < 1) (hz : 1 < |z|) :
  1 + ∑' n : ℕ, (1 + x^n) * P_n n x z = 0 :=
begin
  sorry
end

end sum_P_n_eq_zero_l389_389063


namespace initial_kittens_count_l389_389811

-- Let's define the initial conditions first.
def kittens_given_away : ℕ := 2
def kittens_remaining : ℕ := 6

-- The main theorem to prove the initial number of kittens.
theorem initial_kittens_count : (kittens_given_away + kittens_remaining) = 8 :=
by sorry

end initial_kittens_count_l389_389811


namespace shortest_side_cyclic_quadrilateral_l389_389096

theorem shortest_side_cyclic_quadrilateral (q : Type) (h_cyclic : IsCyclicQuadrilateral q) (h_radius : Circumradius q = 1) : 
  ∃ s, s ∈ sides q ∧ s ≤ Real.sqrt 2 := 
sorry

end shortest_side_cyclic_quadrilateral_l389_389096


namespace work_done_is_halved_l389_389898

theorem work_done_is_halved
  (A₁₂ A₃₄ : ℝ)
  (isothermal_process : ∀ (p V₁₂ V₃₄ : ℝ), V₁₂ = 2 * V₃₄ → p * V₁₂ = A₁₂ → p * V₃₄ = A₃₄) :
  A₃₄ = (1 / 2) * A₁₂ :=
sorry

end work_done_is_halved_l389_389898


namespace three_digit_numbers_count_l389_389430

theorem three_digit_numbers_count :
  let t_range := finset.range 10
  let h_range := finset.range 9
  let valid_u t := if t * 3 ≤ 9 then 9 - (t * 3) + 1 else 0
  let total_count := h_range.card * t_range.sum (λ t, valid_u t)
  total_count = 198 :=
by
  sorry

end three_digit_numbers_count_l389_389430


namespace no_tree_for_positive_c_l389_389637

noncomputable def expected_remaining_edges (n : ℕ) (c : ℝ) (T : Type) [Fintype T] [DecidableEq T] := sorry

-- Statement: For any tree T and any positive constant c, 
-- the expected number of edges remaining in the complete graph on n vertices
-- after erasing subgraphs isomorphic to T is not at least cn^2 for all n.
theorem no_tree_for_positive_c (T : Type) [tree : Fintype T] [DecidableEq T] :
  ¬ ∃ (c : ℝ) (h : c > 0), ∀ (n : ℕ), expected_remaining_edges n c T ≥ c * (n : ℝ) ^ 2 := sorry

end no_tree_for_positive_c_l389_389637


namespace count_valid_three_digit_numbers_l389_389421

-- Define the property of a number being three digits
def isThreeDigit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

-- Define the property of the units digit being at least three times the tens digit
def validUnitsTimesTens (n : ℕ) : Prop :=
  let units := n % 10
  let tens := (n / 10) % 10
  units ≥ 3 * tens

-- Definition combining both conditions
def validThreeDigitNumber (n : ℕ) : Prop :=
  isThreeDigit n ∧ validUnitsTimesTens n

-- The main theorem stating the number of valid three-digit numbers
theorem count_valid_three_digit_numbers :
  Finset.card (Finset.filter validThreeDigitNumber (Finset.range 1000)) = 198 :=
by
  sorry

end count_valid_three_digit_numbers_l389_389421


namespace count_valid_3_digit_numbers_l389_389384

def valid_units_digit (tens_digit units_digit : ℕ) : Prop :=
  units_digit ≥ 3 * tens_digit

def count_valid_numbers : ℕ :=
  (∑ tens_digit in {0, 1, 2, 3}, (finset.filter (valid_units_digit tens_digit) (finset.range 10)).card) * 9

theorem count_valid_3_digit_numbers : count_valid_numbers = 198 :=
by
  sorry

end count_valid_3_digit_numbers_l389_389384


namespace range_of_a_l389_389726

theorem range_of_a :
  (∀ a : ℝ,
    (¬ ((∃ r1 r2 : ℝ, r1 * r1 - a * r1 + 1 = 0 ∧ r2 * r2 - a * r2 + 1 = 0) ∧
       (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → a^2 - 3 * a - x + 1 ≤ 0))) ∧
    (¬ (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → ¬ (a^2 - 3 * a - x + 1 ≤ 0)))) ↔ (1 ≤ a ∧ a < 2)) :=
begin
  sorry,
end

end range_of_a_l389_389726


namespace subway_station_route_or_cycle_l389_389147

variables (V : Type*) [fintype V] [decidable_eq V] (E : V → V → Prop)

theorem subway_station_route_or_cycle (h1 : ∃ (v0 v1 v2 : V), v0 ≠ v1 ∧ v0 ≠ v2 ∧ v1 ≠ v2)
  (h2 : ∃ (P : list V), (∀ (a b : V), E a b → E b a) ∧ (nodup P) ∧ (length P > 1) ∧
        (∀(i : ℕ), i + 1 < length P → E (P.nth_le i sorry) (P.nth_le (i+1) sorry))) :
  (∃ (A B C : V), ∀ (P' : list V), (∀ (a b : V), E a b → E b a) → (nodup P') → (A ∈ P') → (B ∈ P') → 
     ¬ A = B → ¬ C ∈ P' → ∃ (a b : V), E A a → E B b → a = C → ¬ E a c) ∨
  (∃ (cycle : list V), nodup cycle ∧ length cycle ≥ ⌊√(2 * length P)⌋) :=
sorry

end subway_station_route_or_cycle_l389_389147


namespace probability_of_selection_same_l389_389152

variable {Population : Type*} [Fintype Population]

-- Definitions for the problem
variable (sample_size : ℕ) 
variable (sampling_method_reasonable : Prop)

-- Hypotheses based on conditions
axiom same_sample_size : ∀ (d1 d2 : Population → Bool), 
  (card (filter d1 univ) = sample_size) ∧ (card (filter d2 univ) = sample_size)

axiom reasonable_sampling : sampling_method_reasonable

-- Define probability of being selected
def prob_selected (p : Population) : ℝ :=
  1 / (Fintype.card Population)

-- The proof statement
theorem probability_of_selection_same :
  (∀ p : Population, prob_selected p = 1 / (Fintype.card Population)) :=
by
  sorry

end probability_of_selection_same_l389_389152


namespace find_a_for_perpendicular_lines_l389_389763

-- Define the slopes of the given lines
def slope_l1 (a : ℝ) : ℝ := (a + 1) / 2
def slope_l2 (a : ℝ) : ℝ := -1 / a

-- Define the condition for perpendicularity
def perpendicular (slope1 slope2 : ℝ) : Prop := slope1 * slope2 = -1

-- The main theorem we need to prove
theorem find_a_for_perpendicular_lines : 
  ∀ (a : ℝ), perpendicular (slope_l1 a) (slope_l2 a) → a = 1 :=
by
  intro a perp_condition
  sorry

end find_a_for_perpendicular_lines_l389_389763


namespace count_valid_three_digit_numbers_l389_389422

-- Define the property of a number being three digits
def isThreeDigit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

-- Define the property of the units digit being at least three times the tens digit
def validUnitsTimesTens (n : ℕ) : Prop :=
  let units := n % 10
  let tens := (n / 10) % 10
  units ≥ 3 * tens

-- Definition combining both conditions
def validThreeDigitNumber (n : ℕ) : Prop :=
  isThreeDigit n ∧ validUnitsTimesTens n

-- The main theorem stating the number of valid three-digit numbers
theorem count_valid_three_digit_numbers :
  Finset.card (Finset.filter validThreeDigitNumber (Finset.range 1000)) = 198 :=
by
  sorry

end count_valid_three_digit_numbers_l389_389422


namespace coefficient_of_x_three_halves_l389_389259

theorem coefficient_of_x_three_halves :
  let a := (λ x : ℂ, 2 * x^2)
  let b := (λ x : ℂ, -3 / x)
  ∀ x : ℂ, is_coeff_zero : 
    ∂x : ℤ, (x^2 = 2x^2 - 3 / x) - (2x^2 - 3 / x) = 0 :=
begin
  sorry -- Proof omitted
end

end coefficient_of_x_three_halves_l389_389259


namespace num_triangles_with_longest_side_eleven_l389_389563

theorem num_triangles_with_longest_side_eleven : 
    let a b c : ℕ := 
    ∃ (a : ℕ) (b : ℕ) (c : ℕ), a > 0 ∧ b > 0 ∧ c = 11 ∧ a + b > 11 ∧ a ≤ 11 ∧ b ≤ 11 ∧ a + b + 11 = 36 :=
sorry

end num_triangles_with_longest_side_eleven_l389_389563


namespace extremum_necessary_but_not_sufficient_l389_389350

variables {ℝ : Type*} [linear_ordered_field ℝ] [topological_space ℝ] [topological_ring ℝ] [differentiable ℝ ℝ]

noncomputable def f (x : ℝ) : ℝ := sorry -- Function definition will be deferred

def p (x : ℝ) := deriv f x = 0
def q (x : ℝ) := ∃ x₀, (∃ ε > 0, ∀ x, abs (x - x₀) < ε → f x ≤ f x₀) ∨ (∃ ε > 0, ∀ x, abs (x - x₀) < ε → f x ≥ f x₀)

-- Theorem statement
theorem extremum_necessary_but_not_sufficient (x : ℝ) :
  (q x → p x) ∧ ¬(p x → q x) :=
by 
  sorry

end extremum_necessary_but_not_sufficient_l389_389350


namespace fold_equilateral_triangle_k_l389_389596

-- Definitions for the triangle and the folding properties
def equilateral_triangle (A B C : Type) (side_length : ℝ) :=
  ∀ (AB BC CA : ℝ), AB = side_length ∧ BC = side_length ∧ CA = side_length

def point_on_segment (X : Type) (B C : Type) (BX : ℝ) :=
  BX = 6

def fold_creaselength_sqrt_k (k : ℝ) :=
  sqrt k

-- The problem statement: given the above conditions, the fold length squared is 343
theorem fold_equilateral_triangle_k (A B C X : Type) (k : ℝ) :
            equilateral_triangle A B C 30 →
            point_on_segment X B 6 →
            fold_creaselength_sqrt_k k → 
            k = 343 :=
by {
  sorry
}

end fold_equilateral_triangle_k_l389_389596


namespace seating_probability_l389_389146

/-- There are 9 representatives from 3 different countries, 
    each country having 3 representatives. They are seated randomly 
    around a round table with 9 chairs. The probability that each 
    representative has at least one representative from another country 
    sitting next to them is 41/56. -/
theorem seating_probability (total_representatives : ℕ) (countries : ℕ) (reps_per_country : ℕ) :
  total_representatives = 9 → 
  countries = 3 → 
  reps_per_country = 3 → 
  let total_arrangements := (Nat.factorial total_representatives) / (Nat.factorial reps_per_country * Nat.factorial reps_per_country * Nat.factorial reps_per_country) in
  let favorable_arrangements := total_arrangements - 450 in
  (favorable_arrangements / total_arrangements : ℚ) = 41 / 56 :=
by
  intros htotal hcountries hreps
  simp [total_arrangements, favorable_arrangements, Nat.factorial]
  sorry

end seating_probability_l389_389146


namespace fixed_point_of_f_l389_389900

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x-1) + 4

theorem fixed_point_of_f (a : ℝ) (h_pos : a > 0) (h_neq_one : a ≠ 1) : f a 1 = 5 :=
by
  unfold f
  -- Skip the proof; it will be filled in the subsequent steps
  sorry

end fixed_point_of_f_l389_389900


namespace selling_price_when_profit_equals_loss_l389_389136

theorem selling_price_when_profit_equals_loss (CP SP Rs_57 : ℕ) (h1: CP = 50) (h2: Rs_57 = 57) (h3: Rs_57 - CP = CP - SP) : 
  SP = 43 := by
  sorry

end selling_price_when_profit_equals_loss_l389_389136


namespace distance_BC_example_l389_389203

def distance_between_B_and_C (d_AB : ℝ) (v_b : ℝ) (v_r : ℝ) (t_total : ℝ) : ℝ := 
  let v_down := v_b + v_r
  let v_up := v_b - v_r
  have eq1 : (d_AB + distance_between_B_and_C d_AB v_b v_r t_total) / v_down + distance_between_B_and_C d_AB v_b v_r t_total / v_up = t_total := sorry
  distance_between_B_and_C d_AB v_b v_r t_total

theorem distance_BC_example : distance_between_B_and_C 20 40 10 10 = 180 := 
  sorry

end distance_BC_example_l389_389203


namespace range_of_g_l389_389732

noncomputable def g (x : ℝ) (k : ℝ) : ℝ := x^(k + 1)

theorem range_of_g (k : ℝ) (hk : k > 0) : 
  set.range (g (·) k) ∩ set.Ici 2 = set.Ici (2^(k + 1)) :=
sorry

end range_of_g_l389_389732


namespace length_of_EF_l389_389015

theorem length_of_EF (AB CD EF : ℝ) (h1 : AB ∥ CD ∥ EF) (h2 : CD = 120) (h3 : AB = 180) (h4 : EF = (2 / 3) * CD) : EF = 80 :=
by
  sorry

end length_of_EF_l389_389015


namespace parking_cost_l389_389111

/-- 
Theorem: The cost for each hour in excess of 2 hours is $1.75.
-/
theorem parking_cost (total_cost_up_to_2_hours : ℝ) (average_cost_per_hour : ℝ) (total_hours : ℝ) :
  total_cost_up_to_2_hours = 10 ∧ average_cost_per_hour = 2.4722222222222223 ∧ total_hours = 9  →
  let x := (average_cost_per_hour * total_hours - total_cost_up_to_2_hours) / (total_hours - 2) in
  x = 1.75 := 
by
  intros h
  cases h with h1 h2
  cases h2 with h2 h3
  simp only [h1, h2, h3]
  sorry

end parking_cost_l389_389111


namespace find_F_l389_389444

-- Define the condition and the equation
def C (F : ℤ) : ℤ := (5 * (F - 30)) / 9

-- Define the assumption that C = 25
def C_condition : ℤ := 25

-- The theorem to prove that F = 75 given the conditions
theorem find_F (F : ℤ) (h : C F = C_condition) : F = 75 :=
sorry

end find_F_l389_389444


namespace eq_satisfied_for_all_y_l389_389701

theorem eq_satisfied_for_all_y (x : ℝ) : 
  (∀ y: ℝ, 10 * x * y - 15 * y + 3 * x - 4.5 = 0) ↔ x = 3 / 2 :=
by
  sorry

end eq_satisfied_for_all_y_l389_389701


namespace proposition_C_l389_389335

-- Definitions of terms used in the propositions
variable {l m : Line}
variable {α β : Plane}

-- Prove the correct proposition (Proposition C)
theorem proposition_C (h1 : Perpendicular l α) (h2 : Parallel α β) : Perpendicular l β :=
sorry

end proposition_C_l389_389335


namespace child_running_speed_on_still_sidewalk_l389_389615

theorem child_running_speed_on_still_sidewalk (c s : ℕ) 
  (h1 : c + s = 93) 
  (h2 : c - s = 55) : c = 74 :=
sorry

end child_running_speed_on_still_sidewalk_l389_389615


namespace numerators_required_l389_389827

def is_valid_numerator (n : Nat) : Prop :=
  (n % 3 ≠ 0) ∧ (n % 11 ≠ 0)

def numerators_count : Nat :=
  (Finset.range 100).filter (λ n, n > 0 ∧ is_valid_numerator n).card

theorem numerators_required : numerators_count = 60 :=
  sorry

end numerators_required_l389_389827


namespace cone_section_volume_ratio_l389_389629

theorem cone_section_volume_ratio :
  ∀ (r h : ℝ), (h > 0 ∧ r > 0) →
  let V1 := ((75 / 3) * π * r^2 * h - (64 / 3) * π * r^2 * h)
  let V2 := ((64 / 3) * π * r^2 * h - (27 / 3) * π * r^2 * h)
  V2 / V1 = 37 / 11 :=
by
  intros r h h_pos
  sorry

end cone_section_volume_ratio_l389_389629


namespace fifth_score_l389_389301

theorem fifth_score (r : ℕ) 
  (h1 : r % 5 = 0)
  (h2 : (60 + 75 + 85 + 95 + r) / 5 = 80) : 
  r = 85 := by 
  sorry

end fifth_score_l389_389301


namespace correct_operation_C_l389_389996

theorem correct_operation_C (m : ℕ) : m^7 / m^3 = m^4 := by
  sorry

end correct_operation_C_l389_389996


namespace find_abc_sum_l389_389338

theorem find_abc_sum :
  ∀ (a b c : ℝ),
    2 * |a + 3| + 4 - b = 0 →
    c^2 + 4 * b - 4 * c - 12 = 0 →
    a + b + c = 5 :=
by
  intros a b c h1 h2
  sorry

end find_abc_sum_l389_389338


namespace set_A_is_all_real_numbers_set_B_is_real_numbers_ge_1_set_C_is_parabola_l389_389287

noncomputable def A : set ℝ := {x | ∃ y, y = x^2 + 1}
noncomputable def B : set ℝ := {y | ∃ x, y = x^2 + 1}
noncomputable def C : set (ℝ × ℝ) := {(x, y) | y = x^2 + 1}

theorem set_A_is_all_real_numbers : A = set.univ :=
by {
  sorry
}

theorem set_B_is_real_numbers_ge_1 : B = {y | y ≥ 1} :=
by {
  sorry
}

theorem set_C_is_parabola : C = {p | ∃ x, p = (x, x^2 + 1)} :=
by {
  sorry
}

end set_A_is_all_real_numbers_set_B_is_real_numbers_ge_1_set_C_is_parabola_l389_389287


namespace probability_product_multiple_of_4_l389_389047

noncomputable def probability_multiple_of_4 (p_j : ℚ) (p_a : ℚ) : ℚ :=
let prob_juan := 2 / 10 in
let prob_amal := 2 / 8 in
let prob_not_juan := (1 - prob_juan) * prob_amal in
prob_juan + prob_not_juan

theorem probability_product_multiple_of_4 :
  let p_j := (2 : ℚ) / 10 in -- Probability that Juan rolls a multiple of 4
  let p_a := (2 : ℚ) / 8 in  -- Probability that Amal rolls a multiple of 4 if Juan does not
  probability_multiple_of_4 p_j p_a = 2 / 5 :=
by
  sorry

end probability_product_multiple_of_4_l389_389047


namespace sum_of_distinct_integers_l389_389061

noncomputable def distinct_integers (p q r s t : ℤ) : Prop :=
  (p ≠ q) ∧ (p ≠ r) ∧ (p ≠ s) ∧ (p ≠ t) ∧ 
  (q ≠ r) ∧ (q ≠ s) ∧ (q ≠ t) ∧ 
  (r ≠ s) ∧ (r ≠ t) ∧ 
  (s ≠ t)

theorem sum_of_distinct_integers
  (p q r s t : ℤ)
  (h_distinct : distinct_integers p q r s t)
  (h_product : (7 - p) * (7 - q) * (7 - r) * (7 - s) * (7 - t) = -120) :
  p + q + r + s + t = 22 :=
  sorry

end sum_of_distinct_integers_l389_389061


namespace number_of_valid_3_digit_numbers_l389_389413

theorem number_of_valid_3_digit_numbers : 
  ∃ (n : ℕ), 
    (∀ (h t u : ℕ), 
      (n = h * 100 + t * 10 + u) ∧ 
      (1 ≤ h ∧ h ≤ 9) ∧ 
      (0 ≤ t ∧ t ≤ 9) ∧ 
      (0 ≤ u ∧ u ≤ 9) ∧ 
      (u ≥ 3 * t)) →
      n = 198 := 
by
  sorry

end number_of_valid_3_digit_numbers_l389_389413


namespace segments_are_equal_l389_389124

variables {A B C D E F M K P L N S : Type*}

-- Define midpoints 
variables (mid_AB : M = midpoint A B)
variables (mid_BC : K = midpoint B C)
variables (mid_CD : P = midpoint C D)
variables (mid_DE : L = midpoint D E)
variables (mid_EF : N = midpoint E F)
variables (mid_FA : S = midpoint F A)
variables (H_eq_triangle : equilateral_triangle (intersection P S S K) 
                    (intersection K N N L) 
                    (intersection L M M P))

theorem segments_are_equal 
  (H : equilateral_triangle (intersection P S S K)
            (intersection K N N L)
            (intersection L M M P)) : 
  (segment_length SP = segment_length KN) ∧ 
  (segment_length KN = segment_length LM) ∧ 
  (segment_length LM = segment_length SP) :=
sorry

end segments_are_equal_l389_389124


namespace inverse_variation_solution_l389_389103

noncomputable def const_k (x y : ℝ) := (x^2) * (y^4)

theorem inverse_variation_solution (x y : ℝ) (k : ℝ) (h1 : x = 8) (h2 : y = 2) (h3 : k = const_k x y) :
  ∀ y' : ℝ, y' = 4 → const_k x y' = 1024 → x^2 = 4 := by
  intros
  sorry

end inverse_variation_solution_l389_389103


namespace ship_passage_time_l389_389201

theorem ship_passage_time (ship_length : ℝ) (ship_speed_kmph : ℝ) (bridge_length : ℝ) :
  ship_length = 450 → ship_speed_kmph = 24 → bridge_length = 900 →
  let total_distance := ship_length + bridge_length in
  let ship_speed_mps := ship_speed_kmph * 1000 / 3600 in
  abs ((total_distance / ship_speed_mps) - 202.4) < 0.1 :=
by {
  intros h1 h2 h3,
  let total_distance := ship_length + bridge_length,
  let ship_speed_mps := ship_speed_kmph * 1000 / 3600,
  have ht : total_distance / ship_speed_mps ≈ 202.4,
  {
    calc total_distance / ship_speed_mps
      = (450 + 900) / (24 * 1000 / 3600) : by rw [h1, h2, h3]
  },
  sorry
}

end ship_passage_time_l389_389201


namespace scientific_notation_0_000003_l389_389466

noncomputable def scientific_notation (x : ℝ) : ℝ × ℤ :=
if h : x ≠ 0 then
  let abs_x := |x|
  let exp := abs_x.log10.floor.cast
  let base := abs_x / 10^exp
  (base, exp)
else (0, 0)

theorem scientific_notation_0_000003 : scientific_notation 0.000003 = (3, -6) :=
by {
  sorry
}

end scientific_notation_0_000003_l389_389466


namespace steve_book_sales_l389_389885

theorem steve_book_sales
  (copies_price : ℝ)
  (agent_rate : ℝ)
  (total_earnings : ℝ)
  (net_per_copy : ℝ := copies_price * (1 - agent_rate))
  (total_copies_sold : ℝ := total_earnings / net_per_copy) :
  copies_price = 2 → agent_rate = 0.10 → total_earnings = 1620000 → total_copies_sold = 900000 :=
by
  intros
  sorry

end steve_book_sales_l389_389885


namespace number_of_ways_to_place_coins_l389_389874

theorem number_of_ways_to_place_coins :
  ∃ (ways : ℕ), ways = 396 ∧ 
    ∀ (board : array (2 * 100) (option bool)), 
      (∀ i j, (board[i][j] = some tt → 
        (i = 0 ∨ board[i-1][j] ≠ some tt) ∧
        (i = 1 ∨ board[i+1][j] ≠ some tt) ∧
        (j = 0 ∨ board[i][j-1] ≠ some tt) ∧
        (j = 99 ∨ board[i][j+1] ≠ some tt)))) ∧
      ((list.count (some tt) (list.join (array.to_list board))) = 99) :=
sorry

end number_of_ways_to_place_coins_l389_389874


namespace sum_of_two_is_zero_l389_389852

variable (a b c d : ℝ) 

theorem sum_of_two_is_zero 
  (h1 : a^3 + b^3 + c^3 + d^3 = 0) 
  (h2 : a + b + c + d = 0) : 
  ∃ x y ∈ {a, b, c, d}, x + y = 0 :=
by 
  sorry

end sum_of_two_is_zero_l389_389852


namespace reciprocal_of_repeating_decimal_l389_389976

theorem reciprocal_of_repeating_decimal :
  let x : ℚ := 36 / 99 in
  1 / x = 11 / 4 :=
by
  sorry

end reciprocal_of_repeating_decimal_l389_389976


namespace students_remaining_l389_389914

-- Define the initial conditions
def groups : Nat := 5
def students_per_group : Nat := 12
def students_left : Nat := 7

-- Define the total initial number of students
def total_students : Nat := groups * students_per_group

-- Define the number of remaining students
def remaining_students : Nat := total_students - students_left

-- Theorem stating that the remaining number of students is 53
theorem students_remaining : remaining_students = 53 := by
  -- Definitions based on conditions
  have initial_students : total_students = 5 * 12 := rfl
  have calculate_remaining : remaining_students = 60 - 7 := by
    rw [←initial_students]
  show remaining_students = 53
  rw [calculate_remaining]
  rfl

end students_remaining_l389_389914


namespace find_quadratic_function_l389_389325

theorem find_quadratic_function (f : ℝ → ℝ) 
  (h1 : f(-1) = 0)
  (h2 : ∀ x : ℝ, x ≤ f(x) ∧ f(x) ≤ (1 + x^2) / 2) :
  f = (λ x : ℝ, (1/4) * x^2 + (1/2) * x + (1/4)) :=
sorry

end find_quadratic_function_l389_389325


namespace range_of_f_l389_389698

def f (x : ℝ) := x - 1 + Real.sqrt (6 * x - x ^ 2)

theorem range_of_f :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → f x ∈ Icc (-1 : ℝ) 2 :=
by 
  sorry

end range_of_f_l389_389698


namespace find_b_perpendicular_l389_389700

def line_1_direction : ℝ × ℝ × ℝ := (1, -2, 1)
def line_2_direction (b : ℝ) : ℝ × ℝ × ℝ := (b, 1, 3)

theorem find_b_perpendicular (b : ℝ) : (line_1_direction.1 * line_2_direction b.1
  + line_1_direction.2 * line_2_direction b.2
  + line_1_direction.3 * line_2_direction b.3 = 0) ↔ b = -1 :=
by
  sorry

end find_b_perpendicular_l389_389700


namespace count_valid_numbers_l389_389403

def is_valid_number (n : Nat) : Prop := 
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let units := n % 10
  1 ≤ hundreds ∧ hundreds ≤ 9 ∧
  0 ≤ tens ∧ tens ≤ 9 ∧
  units ≥ 3 * tens

theorem count_valid_numbers : 
  (Finset.filter is_valid_number (Finset.range 1000)).card = 198 :=
by sorry

end count_valid_numbers_l389_389403


namespace count_random_events_l389_389645

theorem count_random_events : 
  (is_random (throw_dice_twice_get_two_both_times) + 
   is_random (rain_tomorrow) + 
   is_random (win_lottery) + 
   is_random (sum_of_two_elements_greater_than_two) + 
   is_random (water_boiling_at_90_Celsius)) = 3 :=
sorry

-- Let's define what it means for each event to be random in Lean.

def throw_dice_twice_get_two_both_times := true  -- This event can happen (it is random)
def rain_tomorrow := true  -- This event can happen (it is random)
def win_lottery := true  -- This event can happen (it is random)
def sum_of_two_elements_greater_than_two : Prop := ∀ x y : ℕ, x ∈ {1, 2, 3} → y ∈ {1, 2, 3} → x + y > 2  -- This event always happens (it is certain)
def water_boiling_at_90_Celsius := false  -- This event cannot happen (it is impossible)

def is_random (event : Prop) : ℕ := if event then 1 else 0

end count_random_events_l389_389645


namespace david_older_than_rosy_l389_389277

theorem david_older_than_rosy
  (R D : ℕ) 
  (h1 : R = 12) 
  (h2 : D + 6 = 2 * (R + 6)) : 
  D - R = 18 := 
by
  sorry

end david_older_than_rosy_l389_389277


namespace unit_vector_v_conditions_l389_389299

-- Given the necessary vectors and conditions, define them first.
def v : Vector3 ℝ := ⟨(3 - Real.sqrt 6) / 4, 0, -Real.sqrt 6 / 2⟩
def w1 : Vector3 ℝ := ⟨2, 2, -1⟩
def w2 : Vector3 ℝ := ⟨0, 1, -1⟩

-- Definitions of angle conditions
def cos60 : ℝ := Real.cos (Real.pi / 3) -- cos(60°) = 1/2
def cos30 : ℝ := Real.cos (Real.pi / 6) -- cos(30°) = sqrt(3)/2

-- Proving the conditions are satisfied for the given vector
theorem unit_vector_v_conditions : 
  (v.norm = 1) ∧ 
  (angle v w1 = Real.pi / 3) ∧ 
  (angle v w2 = Real.pi / 6) :=
by
  sorry

end unit_vector_v_conditions_l389_389299


namespace inequality_proof_l389_389851

variable (a b c : ℝ)
variable (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
variable (h_eq : a + b + c + a * b + b * c + c * a + a * b * c = 7)

theorem inequality_proof : 
  (Real.sqrt (a ^ 2 + b ^ 2 + 2) + Real.sqrt (b ^ 2 + c ^ 2 + 2) + Real.sqrt (c ^ 2 + a ^ 2 + 2)) ≥ 6 := by
  sorry

end inequality_proof_l389_389851


namespace contains_zero_l389_389244

-- Define a five-digit number in terms of its digits.
def five_digit_number (A B C D E : ℕ) : ℕ :=
  10^4 * A + 10^3 * B + 10^2 * C + 10 * D + E

-- Define the conditions:
noncomputable def switch_two_digits (A B C D E F : ℕ) : Prop :=
  (10^4 * A + 10^3 * B + 10^2 * C + 10 * D + E ≠ 10^4 * A + 10^3 * B + 10^2 * F + 10 * D + E)

-- Lean statement for the proof problem.
theorem contains_zero (A B C D E F : ℕ) :
  let ABCDE := five_digit_number A B C D E,
      ABFDE := five_digit_number A B F D E in
  switch_two_digits A B C D E F ∧ (ABCDE + ABFDE = 111111) → (A = 0 ∨ B = 0 ∨ C = 0 ∨ D = 0 ∨ E = 0) :=
by
  -- This is where the proof would go.
  sorry

end contains_zero_l389_389244


namespace max_balloons_l389_389863

theorem max_balloons (regular_price : ℝ) (discount : ℝ) (num_balloons : ℕ) 
  (total_money : ℝ) (pair_price : ℝ) :
  regular_price = 4 → 
  discount = 0.25 → 
  num_balloons = 40 → 
  total_money = regular_price * num_balloons → 
  pair_price = regular_price + (regular_price * (1 - discount)) → 
  2 * real.floor (total_money / pair_price) = 44 :=
by
  sorry

end max_balloons_l389_389863


namespace boards_nailing_l389_389520

variables {x y a b : ℕ}

theorem boards_nailing (h1 : 2 * x + 3 * y = 87)
                       (h2 : 3 * a + 5 * b = 94) :
                       x + y = 30 ∧ a + b = 30 :=
sorry

end boards_nailing_l389_389520


namespace tangent_line_SB_circumcircle_ABD_l389_389849

open EuclideanGeometry

variables (A B C D S : Point)
variables (hABC : IsTriangle A B C)
variables (hBisector : IsAngleBisector (Segment A S) (Angle B A C))
variables (hDS : SegmentIntersect S (Circumcircle TriangleABC))

theorem tangent_line_SB_circumcircle_ABD 
  (hD_on_BC : OnSegment D (Segment B C))
  (hS_on_circumcircleABC : OnCircumcircle S (CircumcircleABC A B C)) :
  Tangent (Line S B) (CircumcircleABD A B D) := sorry

end tangent_line_SB_circumcircle_ABD_l389_389849


namespace contains_zero_l389_389237

theorem contains_zero (a b c d e f: ℕ) (h1: 1 ≤ a ∧ a ≤ 9) (h2: 1 ≤ b ∧ b ≤ 9) (h3: 0 ≤ c ∧ c ≤ 9) 
  (h4: 1 ≤ d ∧ d ≤ 9) (h5: 0 ≤ e ∧ e ≤ 9) (h6: 0 ≤ f ∧ f ≤ 9) 
  (h7: c ≠ f) (h8: 10^4*a + 10^3*b + 10^2*c + 10^1*d + e + 10^4*a + 10^3*b + 10^2*f + 10^1*d + e = 111111) :
  c = 0 ∨ f = 0 := 
sorry

end contains_zero_l389_389237


namespace minimum_area_of_triangle_OMN_l389_389471

open Real

noncomputable def min_triangle_area : ℝ :=
  let P := some (x, y : ℝ) (hP : (x^2 / 25 + y^2 / 16 = 1) ∧ x < 0 ∧ y < 0)
  let A := some (x1, y1 : ℝ) (hA : x1^2 + y1^2 = 9)
  let B := some (x2, y2 : ℝ) (hB : x2^2 + y2^2 = 9)
  let (M_x, _) := (y : ℝ) (M : ((P.1 * M_x = 9) ∧ x ∈ ℝ))
  let (_, N_y) := (x : ℝ) (N : ((P.2 * N_y = 9) ∧ y ∈ ℝ))
  (9^2 / (10 * (sin (2 * θ))))

theorem minimum_area_of_triangle_OMN : min_triangle_area = 81 / 20 :=
  sorry

end minimum_area_of_triangle_OMN_l389_389471


namespace log_5_3100_nearest_int_l389_389935

theorem log_5_3100_nearest_int :
  (4 : ℝ) < real.log 3100 / real.log 5 ∧ real.log 3100 / real.log 5 < (5 : ℝ) ∧
  abs (3100 - 3125) < abs (3100 - 625) →
  real.log 3100 / real.log 5 = 5 :=
sorry

end log_5_3100_nearest_int_l389_389935


namespace range_g_l389_389954

def g (x : ℝ) : ℝ := 1 / (x^2 + x)

theorem range_g :
  ∀ y > 0, ∃ x : ℝ, x ≠ 0 ∧ x ≠ -1 ∧ g x = y :=
by
  sorry

end range_g_l389_389954


namespace contains_zero_if_sum_is_111111_l389_389217

variables (x y : ℕ)
variables (digits_x digits_y : Fin 5 → ℕ)

-- Conditions
def is_five_digit (n : ℕ) : Prop := 
  10000 ≤ n ∧ n < 100000

def differs_by_two_digit_swap (dx dy : Fin 5 → ℕ) : Prop :=
  ∃ i j : Fin 5, i ≠ j ∧ 
  (∀ k : Fin 5, (k = i → dy k = dx j) ∧ 
                (k = j → dy k = dx i) ∧ 
                (k ≠ i ∧ k ≠ j → dy k = dx k))

theorem contains_zero_if_sum_is_111111 
  (hx : is_five_digit x) (hy : is_five_digit y)
  (h_digits_x : digits_x = fun i : Fin 5 => (x / 10^(4-i) % 10))
  (h_digits_y : digits_y = fun i : Fin 5 => (y / 10^(4-i) % 10))
  (h_swap : differs_by_two_digit_swap digits_x digits_y)
  (h_sum : x + y = 111111) :
  ∃ i : Fin 5, digits_x i = 0 :=
begin
  -- Proof omitted
  sorry,
end

end contains_zero_if_sum_is_111111_l389_389217


namespace triangle_1_triangle_2_l389_389779

theorem triangle_1
  (A B C : ℝ) (a b c S : ℝ)
  (h1 : a = c + b) -- Dealing with sides opposite to angles A, B, and C
  (h2 : 2 * a * (cos (C/2))^2 + 2 * c * (cos (A / 2))^2 = (5 / 2) * b) :
  2 * (a + c) = 3 * b := 
sorry

theorem triangle_2
  (A B C : ℝ) (a b c S : ℝ)
  (cos_B_cond : real.cos B = 1 / 4)
  (S_cond : S = real.sqrt 15)
  (ac_cond : a * c = 8)
  (h1 : 2 * (a + c) = 3 * b)
  (h2 : b^2 = (a + c)^2 - 2 * a * c * cos (B)) : 
  b = 4 :=
sorry

end triangle_1_triangle_2_l389_389779


namespace tangency_implies_concyclic_l389_389788

variables (A B C D K L M N E F S T U V : Type*) [H0 : quadrilateral A B C D]
variables [is_point_on_line K A B] [is_point_on_line L B C] [is_point_on_line M C D] [is_point_on_line N D A]

-- given ratios
variables (H1 : AK / KB = DA / BC) (H2 : BL / LC = AB / CD) (H3 : CM / MD = BC / DA) (H4 : DN / NA = CD / AB)

-- Extended points E and F
variables (H5 : intersection_point E A B C D) (H6 : intersection_point F A D B C)

-- Tangency points of incircles
variables (H7 : incircle_tangent_point S A E F) (H8 : incircle_tangent_point T A F E)
variables (H9 : incircle_tangent_point U C E F) (H10 : incircle_tangent_point V C F E)

-- K, L, M, N are concyclic
variables (H11 : concyclic [K, L, M, N])

theorem tangency_implies_concyclic :
  concyclic [S, T, U, V] :=
sorry

end tangency_implies_concyclic_l389_389788


namespace sets_coincide_l389_389051

theorem sets_coincide (a b c : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
                      (h1 : ∃ k, c * (c^2 - c + 1) = k * (a * b))
                      (h2 : ∃ m, a + b = m * (c^2 + 1)) : 
                      ({a, b} = {c, c^2 - c + 1}) :=
by
  sorry

end sets_coincide_l389_389051


namespace sixty_different_numerators_l389_389835

theorem sixty_different_numerators : 
  (Finset.card {ab : ℕ | 1 ≤ ab ∧ ab ≤ 99 ∧ Nat.gcd ab 99 = 1} = 60) :=
sorry

end sixty_different_numerators_l389_389835


namespace shaded_area_of_logo_l389_389197

theorem shaded_area_of_logo 
  (side_length_of_square : ℝ)
  (side_length_of_square_eq : side_length_of_square = 30)
  (radius_of_circle : ℝ)
  (radius_eq : radius_of_circle = side_length_of_square / 4)
  (number_of_circles : ℕ)
  (number_of_circles_eq : number_of_circles = 4)
  : (side_length_of_square^2) - (number_of_circles * Real.pi * (radius_of_circle^2)) = 900 - 225 * Real.pi := by
    sorry

end shaded_area_of_logo_l389_389197


namespace no_x4_term_in_expansion_l389_389575

theorem no_x4_term_in_expansion (a : ℝ) : 
  let expansion := (x ^ 2 + a * x + 1) * (-6 * x ^ 3) in 
  (expansion.coeff 4 = 0) ↔ (a = 0) :=
by
  sorry

end no_x4_term_in_expansion_l389_389575


namespace people_got_off_at_second_stop_l389_389205

theorem people_got_off_at_second_stop (x : ℕ) :
  (10 - x) + 20 - 18 + 2 = 12 → x = 2 :=
  by sorry

end people_got_off_at_second_stop_l389_389205


namespace height_at_end_of_2_years_l389_389640

-- Step d): Define the conditions and state the theorem

-- Define a function modeling the height of the tree each year
def tree_height (initial_height : ℕ) (years : ℕ) : ℕ :=
  initial_height * 3^years

-- Given conditions as definitions
def year_4_height := 81 -- height at the end of 4 years

-- Theorem that we need to prove
theorem height_at_end_of_2_years (initial_height : ℕ) (h : tree_height initial_height 4 = year_4_height) :
  tree_height initial_height 2 = 9 :=
sorry

end height_at_end_of_2_years_l389_389640


namespace reciprocal_of_36_recurring_decimal_l389_389956

-- Definitions and conditions
def recurring_decimal (x : ℚ) : Prop := x = 36 / 99

-- Theorem statement
theorem reciprocal_of_36_recurring_decimal :
  recurring_decimal (36 / 99) → (1 / (36 / 99) = 11 / 4) :=
sorry

end reciprocal_of_36_recurring_decimal_l389_389956


namespace max_area_ABC_l389_389027

noncomputable def PA := 3
noncomputable def PB := 4
noncomputable def PC := 5
noncomputable def BC := 6
noncomputable def maximum_area_ABC := Real.sqrt 98.4375 + 9

theorem max_area_ABC
  (PA_eq : PA = 3)
  (PB_eq : PB = 4)
  (PC_eq : PC = 5)
  (BC_eq : BC = 6)
  (angle_APB_right : ∠' \(PA_eq) \(PB_eq) = 90) :
  (area_of_triangle_ABC PA PB PC BC ∠_APB) = maximum_area_ABC := 
  sorry

end max_area_ABC_l389_389027


namespace reciprocal_of_36_recurring_decimal_l389_389959

-- Definitions and conditions
def recurring_decimal (x : ℚ) : Prop := x = 36 / 99

-- Theorem statement
theorem reciprocal_of_36_recurring_decimal :
  recurring_decimal (36 / 99) → (1 / (36 / 99) = 11 / 4) :=
sorry

end reciprocal_of_36_recurring_decimal_l389_389959


namespace find_x_l389_389329

-- Define the set h
def h (x : ℝ) : set ℝ := {1, x, 18, 20, 29, 33}

-- Helper function to compute the mean of a set of 6 elements
def mean (s : set ℝ) [fintype s] : ℝ := (∑ x in s, x) / fintype.card s

-- Helper function to compute the median of a set of 6 elements
def median (s : set ℝ) [fintype s] : ℝ :=
  let sorted_list := (finset.sort (≤) s.to_finset).val in
  (sorted_list.nth_le 2 (by norm_num) + sorted_list.nth_le 3 (by norm_num)) / 2

theorem find_x (x : ℝ) (h : set ℝ) (mean h = (median h) - 1) : x = 7 :=
by
  -- Definition of the set h
  have h_eq : h = {1, x, 18, 20, 29, 33} := sorry
  -- Given condition on mean and median
  have mean_eq : mean h = 19 - 1 := sorry
  -- Solve for x
  sorry

end find_x_l389_389329


namespace boards_nailing_l389_389519

variables {x y a b : ℕ}

theorem boards_nailing (h1 : 2 * x + 3 * y = 87)
                       (h2 : 3 * a + 5 * b = 94) :
                       x + y = 30 ∧ a + b = 30 :=
sorry

end boards_nailing_l389_389519


namespace greatest_percentage_zero_l389_389513

theorem greatest_percentage_zero (n : ℕ) (p_A : ℕ → ℕ) (percentage_A : ℕ → ℕ) 
  (h_min_planes : ∀ i, n ≥ 5)
  (h_fleet_compositions : ∀ i, percentage_A i ∈ {10, 20, 30, 40, 50, 60, 70, 80, 90})
  (h_amenities_type_A : ∀ (i : ℕ), p_A i offers_wireless_internet ∧ p_A i offers_free_snacks)
  (h_amenities_type_B : ∀ (i : ℕ), ¬(p_B i offers_wireless_internet ∧ p_B i offers_free_snacks)) :
  (∀ i, p_A i ∈ percentage_A i) → (∃ (i : ℕ), percentage_A i = 100) → false := 
sorry

end greatest_percentage_zero_l389_389513


namespace count_true_propositions_l389_389442

theorem count_true_propositions :
  let P := ∀ a : ℝ, a > 2 → a > 1
  let converseP := ∀ a : ℝ, a > 1 → a > 2
  let inverseP := ∀ a : ℝ, a ≤ 2 → a ≤ 1
  let contrapositiveP := ∀ a : ℝ, a ≤ 1 → a ≤ 2
  2 =
    (if P then 1 else 0) +
    (if converseP then 1 else 0) +
    (if inverseP then 1 else 0) +
    (if contrapositiveP then 1 else 0) := by
  sorry

end count_true_propositions_l389_389442


namespace find_a_if_perpendicular_and_nonzero_l389_389012

variable (a : ℝ)

-- Conditions
def MN := (a, a + 4)
def PQ := (-5, a)
def perpendicular (u v : ℝ × ℝ) := u.1 * v.1 + u.2 * v.2 = 0

-- Theorem statement
theorem find_a_if_perpendicular_and_nonzero (h : perpendicular (MN a) (PQ a)) (hnz : a ≠ 0) : a = 1 := by
  sorry

end find_a_if_perpendicular_and_nonzero_l389_389012


namespace part1_3_neg5_is_pair_part1_neg2_4_is_not_pair_part2_find_n_part3_find_k_l389_389360

def is_equation_number_pair (a b : ℝ) : Prop :=
  ∀ x : ℝ, (x = 1 / (a + b) ↔ a / x + 1 = b)

theorem part1_3_neg5_is_pair : is_equation_number_pair 3 (-5) :=
sorry

theorem part1_neg2_4_is_not_pair : ¬ is_equation_number_pair (-2) 4 :=
sorry

theorem part2_find_n (n : ℝ) : is_equation_number_pair n (3 - n) ↔ n = 1 / 2 :=
sorry

theorem part3_find_k (m k : ℝ) (hm : m ≠ -1) (hm0 : m ≠ 0) (hk1 : k ≠ 1) :
  is_equation_number_pair (m - k) k → k = (m^2 + 1) / (m + 1) :=
sorry

end part1_3_neg5_is_pair_part1_neg2_4_is_not_pair_part2_find_n_part3_find_k_l389_389360


namespace sin_cos_unique_solution_l389_389317

theorem sin_cos_unique_solution (α : ℝ) (hα1 : 0 < α) (hα2 : α < (π / 2)) :
  ∃! x : ℝ, (Real.sin α) ^ x + (Real.cos α) ^ x = 1 :=
sorry

end sin_cos_unique_solution_l389_389317


namespace maximize_lower_houses_l389_389791

theorem maximize_lower_houses (x y : ℕ) 
    (h1 : x + 2 * y = 30)
    (h2 : 0 < y)
    (h3 : (∃ k, k = 112)) :
  ∃ x y, (x + 2 * y = 30) ∧ ((x * y)) = 112 :=
by
  sorry

end maximize_lower_houses_l389_389791


namespace barry_magic_wand_l389_389871

theorem barry_magic_wand (n : ℕ) : 
  (∏ i in finset.range n, (2 * i + 2) / (2 * i + 1)) = 50 :=
sorry

end barry_magic_wand_l389_389871


namespace graph_S_line_segment_l389_389928

theorem graph_S_line_segment (a b t : ℝ) (h1 : a + b = 2) (h2 : a * b = t - 1) : 
  1 < t ∧ t < 2 → (∃ S, S = (a - b)^2 ∧ S = 8 - 4t) :=
by
  sorry

end graph_S_line_segment_l389_389928


namespace largest_inscribed_square_side_length_l389_389804

/-- 
  Inside a square with side length 10, two congruent equilateral triangles are drawn 
  such that they share one side and each has one vertex on a vertex of the square.
  Prove that the side length of the largest square that can be inscribed in the 
  space inside the square and outside of the triangles is 5 - 5*sqrt(3)/3.
-/
theorem largest_inscribed_square_side_length :
  let side_length_of_square := 10 
  and equilateral_triangle_side := (10 * √2) / (√3) 
  in 
  let max_inscribed_square_side := 5 - 5 * √3 / 3 
  in 
  ∀ (side_length : ℝ), side_length = max_inscribed_square_side := sorry

end largest_inscribed_square_side_length_l389_389804


namespace numerators_required_l389_389828

def is_valid_numerator (n : Nat) : Prop :=
  (n % 3 ≠ 0) ∧ (n % 11 ≠ 0)

def numerators_count : Nat :=
  (Finset.range 100).filter (λ n, n > 0 ∧ is_valid_numerator n).card

theorem numerators_required : numerators_count = 60 :=
  sorry

end numerators_required_l389_389828


namespace three_digit_numbers_with_units_at_least_three_times_tens_l389_389393

open Nat

noncomputable def valid_units_and_tens_digit (tens units : ℕ) : Prop :=
  (tens = 0 ∧ units ∈ range 10) ∨
  (tens = 1 ∧ units ∈ range' 3 10) ∨
  (tens = 2 ∧ units ∈ range' 6 10)

noncomputable def valid_hundreds_digit (hundreds : ℕ) : Prop :=
  hundreds ∈ range' 1 10

theorem three_digit_numbers_with_units_at_least_three_times_tens :
  let total_numbers := 189
  ∃ total_numbers : ℕ, total_numbers = ∑ h in range' 1 10, ∑ t in range 10, ∑ u in range 10, if valid_hundreds_digit h ∧ valid_units_and_tens_digit t u then 1 else 0 :=
begin
  sorry
end

end three_digit_numbers_with_units_at_least_three_times_tens_l389_389393


namespace find_angle_A_max_area_l389_389352

noncomputable theory

open Real

variables {a b c A B : ℝ} -- sides and angles of the triangle
-- radius of circumcircle
def radius : ℝ := 2

-- Given condition
def condition1 : Prop := c^2 - a^2 = 4 * (sqrt 3 * c - b) * sin B

-- Angle A solution
theorem find_angle_A (h1 : radius = 2) (h2 : condition1) (h3 : sin B = b / 4) : A = π / 6 :=
sorry

-- Maximum area solution
theorem max_area (h1 : radius = 2) (h2 : condition1) (h3 : sin B = b / 4) (h4 : cos A = sqrt 3 / 2) : 
  ∃ S, S = 1/4 * b * c ∧ S ≤ 2 + sqrt 3 :=
sorry

end find_angle_A_max_area_l389_389352


namespace general_term_l389_389031

theorem general_term (a : ℕ → ℝ) (S : ℕ → ℝ) (h_pos : ∀ n, 0 < a n) (h_sum : ∀ n, S n = ∑ i in finset.range (n + 1), a i) (h_eq : ∀ n, a n + 1 / a n = 2 * S n) :
  ∀ n, a n = Real.sqrt n - Real.sqrt (n - 1) :=
by
  sorry

end general_term_l389_389031


namespace bisectors_of_adjacent_supplementary_angles_are_perpendicular_l389_389891

theorem bisectors_of_adjacent_supplementary_angles_are_perpendicular
  {α β : ℝ} (hadj : ∃ p : ℝ, α = p ∧ β = 180 - p) :
  (let θ := α / 2 in let φ := β / 2 in θ + φ = 90) :=
by
  sorry

end bisectors_of_adjacent_supplementary_angles_are_perpendicular_l389_389891


namespace last_two_digits_l389_389583

theorem last_two_digits :
  (2 * 5^2 * 2^2 * 13 * 2 * 27 * 2^3 * 7 * 2 * 29 * 2^2 * 3 * 5 / (2^6 * 10^3)) % 100 = 22 :=
by sorry

end last_two_digits_l389_389583


namespace simplify_fraction_subtraction_l389_389097

theorem simplify_fraction_subtraction : (1 / 210) - (17 / 35) = -101 / 210 := by
  sorry

end simplify_fraction_subtraction_l389_389097


namespace three_digit_numbers_count_l389_389440

def is_3_digit_number (n : Nat) : Prop :=
  100 ≤ n ∧ n ≤ 999

def units_digit (n : Nat) : Nat :=
  n % 10

def tens_digit (n : Nat) : Nat :=
  (n / 10) % 10

def hundreds_digit (n : Nat) : Nat :=
  (n / 100) % 10

def valid_number (n : Nat) : Prop :=
  is_3_digit_number n ∧ units_digit n ≥ 3 * tens_digit n

theorem three_digit_numbers_count : 
  (Finset.card (Finset.filter valid_number (Finset.range 1000))) = 198 :=
by
  sorry

end three_digit_numbers_count_l389_389440


namespace surface_area_three_dimensional_shape_l389_389929

-- Define the edge length of the largest cube
def edge_length_large : ℕ := 5

-- Define the condition for dividing the edge of the attachment face of the large cube into five equal parts
def divided_into_parts (edge_length : ℕ) (parts : ℕ) : Prop :=
  parts = 5

-- Define the condition that the edge lengths of all three blocks are different
def edge_lengths_different (e1 e2 e3 : ℕ) : Prop :=
  e1 ≠ e2 ∧ e1 ≠ e3 ∧ e2 ≠ e3

-- Define the surface area formula for a cube
def surface_area (s : ℕ) : ℕ :=
  6 * s^2

-- State the problem as a theorem
theorem surface_area_three_dimensional_shape (e1 e2 e3 : ℕ) (h1 : e1 = edge_length_large)
    (h2 : divided_into_parts e1 5) (h3 : edge_lengths_different e1 e2 e3) : 
    surface_area e1 + (surface_area e2 + surface_area e3 - 4 * (e2 * e3)) = 270 :=
sorry

end surface_area_three_dimensional_shape_l389_389929


namespace sum_first_101_terms_l389_389328

noncomputable theory
open_locale big_operators

def a_seq (n : ℕ) : ℕ := n
def S (n : ℕ) := ∑ i in finset.range(n+1), a_seq i

theorem sum_first_101_terms :
  (2 * S 101 = (a_seq 101) * (a_seq 101 + 1)) →
  (∑ i in finset.range(101), (-1) ^ i * a_seq i = -51) :=
begin
  intros h,
  sorry
end

end sum_first_101_terms_l389_389328


namespace velocity_at_t2_is_6_l389_389552

-- Definition of the displacement function S
def displacement (t : ℝ) : ℝ := 10 * t - t^2

-- Definition of the velocity as the derivative of displacement
def velocity (t : ℝ) : ℝ := (deriv displacement) t

-- Theorem stating that the velocity at t = 2 is 6 m/s
theorem velocity_at_t2_is_6 : velocity 2 = 6 := by
  sorry

end velocity_at_t2_is_6_l389_389552


namespace find_BD_l389_389476

variables {A B C D : Type*} [metric_space A] [metric_space B] [metric_space C] [metric_space D]

def right_triangle (A B C : Type*) (C : Type*) : Prop :=
  ∃ (B₁ : A), (dist A B = dist B₁ C) ∧ (dist C A = sqrt (dist A B ^ 2 - dist B₁ C ^ 2))

def similar_triangle (A B C D : Type*) (E F G : Type*) : Prop :=
  (dist A B / dist B C = dist D E / dist F G) ∧ (dist B A / dist C B = dist E D / dist G F)

theorem find_BD {A B C D : Type*} [metric_space A] [metric_space B] [metric_space C] [metric_space D]
    (h1: right_triangle A B C)
    (h2: dist B C = 1/2)
    (h3: dist A C = b/2)
    (h4: dist A D = 1)
    (h5: ∠BAD = ∠BCA) :
  dist B D = 1 / b := by {
    sorry
}

end find_BD_l389_389476


namespace reciprocal_of_repeating_decimal_l389_389990

theorem reciprocal_of_repeating_decimal :
  let x := (36 : ℚ) / 99 in 1 / x = 11 / 4 :=
by
  trivial -- This is a placeholder, the actual proof is not required and hence replaced by trivial.

end reciprocal_of_repeating_decimal_l389_389990


namespace proof_problem_l389_389857

-- Definitions
def F : Point := (whatever the coordinates of F are)
def ellipse_C (x y : ℝ) (m : ℝ) := (x^2 / (4 * m) + y^2 / (3 * m) = 1)
def chord_length (x1 y1 x2 y2 : ℝ) := 2 * sqrt ((x2 - x1)^2 + (y2 - y1)^2)
def circle_P (x y : ℝ) (r : ℝ) := ((x + (4 * √3) / 7)^2 + (y - (3 * √3) / 7)^2 = r^2)

-- Conditions
variable (m : ℝ) (hm : m > 0)
variable (x y : ℝ) (hC : ellipse_C x y m)
variable (hx : y = x)
variable (hchord : chord_length x x y y = (4 * √42) / 7)


-- Proposition: Part 1
def ellipse_equation : Prop :=
  (∃ m > 0, ∀ (x y : ℝ), ellipse_C x y m → ellipse_C x y 1)
-- Proposition: Part 2
def range_PQF (k : ℝ) : Prop :=
  (k ≥ √3 → (∀ (x3 x4 : ℝ), (|PF| * |QF|) = (1 + k^2) | x3 * x4 + (x3 + x4) + 1 | 
    → |PF| * |QF| ∈ (9 / 4, 12 / 5]))

-- Main theorem to be proved
theorem proof_problem : (ellipse_equation ∧ range_PQF) := by 
  sorry

end proof_problem_l389_389857


namespace mildred_weight_is_correct_l389_389076

noncomputable def carol_weight := 9
noncomputable def mildred_weight := carol_weight + 50

theorem mildred_weight_is_correct : mildred_weight = 59 :=
by 
  -- the proof is omitted
  sorry

end mildred_weight_is_correct_l389_389076


namespace correct_propositions_l389_389557

-- Define each proposition
def proposition1 := ∀ (r : ℝ), |r| ≠ r → "Larger r implies stronger correlation."
def proposition2 := ∀ (rss : ℝ), rss > 0 → "Smaller RSS implies better fitting."
def proposition3 := ∀ (R2 : ℝ), R2 > 0 → "Smaller R2 implies worse fitting."
def proposition4 := ∀ (ᾰ ᾱ ᾰ' ᾱ' : ℝ), (ᾰ + ᾱ' = ᾱ + ᾰ') → "Regression line passes through sample centroid (x̄, ȳ)."

-- State the theorem
theorem correct_propositions : proposition2 ∧ proposition4 :=
by
  sorry

end correct_propositions_l389_389557


namespace arithmetic_sequence_proof_l389_389320

theorem arithmetic_sequence_proof (a : ℕ → ℤ) (b : ℕ → ℤ) (d : ℤ) (n : ℕ) 
  (h1 : a(1) + a(2) + a(3) = 12) 
  (h2 : (a(4))^2 = a(2) * a(8))
  (h3 : ∀ n, a(n) = a(1) + (n - 1) * d)
  (h4 : d ≠ 0) 
  : 
  (∀ n, a(n) = 2 * n) ∧ 
  (∀ n, b n = 2 ^ (a n)) ∧ 
  (S n = ∑ i in finset.range n, b(i) = (4 / 3) * (4^n - 1)) :=
  sorry

end arithmetic_sequence_proof_l389_389320


namespace solve_equation_l389_389101

theorem solve_equation (x y z : ℕ) : (3 ^ x + 5 ^ y + 14 = z!) ↔ ((x = 4 ∧ y = 2 ∧ z = 5) ∨ (x = 4 ∧ y = 4 ∧ z = 6)) :=
by
  sorry

end solve_equation_l389_389101


namespace ratio_AD_BC_l389_389331

variables (A B C D M N K : Type)
variables [linear_ordered_field ℚ]
variables (AB CD AD BC : ℚ)
variables (CM MD CN NA : ℚ)

theorem ratio_AD_BC (h0 : CM / MD = 4 / 3) (h1 : CN / NA = 4 / 3) :
  (AD / BC = 7 / 12) :=
begin
  sorry,
end

end ratio_AD_BC_l389_389331


namespace find_g_function_l389_389280

variable {R : Type*} [LinearOrderedField R]

noncomputable def g (x : R) : R := (5^x - 3^x) / 4

theorem find_g_function (x y : R) :
  g 2 = 4 ∧ g (x + y) = 5^y * g x + 3^x * g y := 
sorry

end find_g_function_l389_389280


namespace smallest_x_l389_389447

theorem smallest_x (x y : ℝ) (h1 : 4 < x) (h2 : x < 6) (h3 : 6 < y) (h4 : y < 10) (h5 : y - x = 5) :
  x = 4 :=
sorry

end smallest_x_l389_389447


namespace olympiad_permutations_l389_389021

theorem olympiad_permutations : 
  let total_permutations := Nat.factorial 9 / (Nat.factorial 2 * Nat.factorial 2) 
  let invalid_permutations := 5 * (Nat.factorial 4 / Nat.factorial 2)
  total_permutations - invalid_permutations = 90660 :=
by
  let total_permutations : ℕ := Nat.factorial 9 / (Nat.factorial 2 * Nat.factorial 2)
  let invalid_permutations : ℕ := 5 * (Nat.factorial 4 / Nat.factorial 2)
  show total_permutations - invalid_permutations = 90660
  sorry

end olympiad_permutations_l389_389021


namespace length_of_JK_l389_389157

def triangle_area (a b c : ℝ) (angle_c : ℝ) : ℝ :=
  (1 / 2) * a * b * Real.sin angle_c

variables (FG FH : ℝ) (angleGFH : ℝ) (areaFGH : ℝ)
variables (IJ JK : ℝ) (angleIJK : ℝ) (areaIJK : ℝ)

noncomputable theory

def FGH := (FG = 5) ∧ (FH = 4) ∧ (angleGFH = π / 6) ∧ (areaFGH = triangle_area 5 4 (π / 6))

def IJK := (IJ = 2) ∧ (angleIJK = π / 6) ∧ (areaIJK = triangle_area 2 JK (π / 6))

theorem length_of_JK :
  (FGH ∧ IJK ∧ (areaFGH = areaIJK)) → (JK = 10) :=
by
  sorry

end length_of_JK_l389_389157


namespace shopkeeper_total_cards_l389_389202

-- Conditions
def num_standard_decks := 3
def cards_per_standard_deck := 52
def num_tarot_decks := 2
def cards_per_tarot_deck := 72
def num_trading_sets := 5
def cards_per_trading_set := 100
def additional_random_cards := 27

-- Calculate total cards
def total_standard_cards := num_standard_decks * cards_per_standard_deck
def total_tarot_cards := num_tarot_decks * cards_per_tarot_deck
def total_trading_cards := num_trading_sets * cards_per_trading_set
def total_cards := total_standard_cards + total_tarot_cards + total_trading_cards + additional_random_cards

-- Proof statement
theorem shopkeeper_total_cards : total_cards = 827 := by
    sorry

end shopkeeper_total_cards_l389_389202


namespace eq1_eq2_eq3_eq4_l389_389537

theorem eq1 (x : ℝ) : x^2 + 12*x + 27 = 0 → x = -3 ∨ x = -9 := sorry

theorem eq2 (x : ℝ) : 3*x^2 + 10*x + 5 = 0 → x = (-5 + real.sqrt 10) / 3 ∨ x = (-5 - real.sqrt 10) / 3 := sorry

theorem eq3 (x : ℝ) : 3*x*(x - 1) = 2 - 2*x → x = 1 ∨ x = 2 / 3 := sorry

theorem eq4 (x : ℝ) : (3*x + 1)^2 - 9 = 0 → x = -4 / 3 ∨ x = 2 / 3 := sorry

end eq1_eq2_eq3_eq4_l389_389537


namespace find_modulus_z_l389_389503

open Complex

noncomputable def z_w_condition1 (z w : ℂ) : Prop := abs (3 * z - w) = 17
noncomputable def z_w_condition2 (z w : ℂ) : Prop := abs (z + 3 * w) = 4
noncomputable def z_w_condition3 (z w : ℂ) : Prop := abs (z + w) = 6

theorem find_modulus_z (z w : ℂ) (h1 : z_w_condition1 z w) (h2 : z_w_condition2 z w) (h3 : z_w_condition3 z w) :
  abs z = 5 :=
by
  sorry

end find_modulus_z_l389_389503


namespace reciprocal_of_repeating_decimal_l389_389975

theorem reciprocal_of_repeating_decimal :
  let x : ℚ := 36 / 99 in
  1 / x = 11 / 4 :=
by
  sorry

end reciprocal_of_repeating_decimal_l389_389975


namespace major_premise_incorrect_conclusion_incorrect_l389_389558

def continuous_on_real (f : ℝ → ℝ) : Prop := Continuous f

def differentiable_on_real (f : ℝ → ℝ) : Prop := Differentiable ℝ f

def critical_point_correct (f : ℝ → ℝ) (x₀ : ℝ) : Prop := 
  Differentiable ℝ f ∧ f' x₀ = 0 ∧
  (∀ x, x < x₀ → f' x < 0) ∧ (∀ x, x > x₀ → f' x > 0)

noncomputable def major_premise (f : ℝ → ℝ) (x₀ : ℝ) : Prop := 
  Differentiable ℝ f ∧ f' x₀ = 0 ∧ x₀ ∈ {x | f x < 0}

noncomputable def minor_premise (f : ℝ → ℝ) : Prop := 
  continuous_on_real f ∧ Differentiable ℝ f ∧ f' 0 = 0 ∧ f = λ x, x^3

theorem major_premise_incorrect : 
  ¬ (∀ (f : ℝ → ℝ) (x₀ : ℝ), major_premise f x₀ → f x₀ < 0) := sorry

theorem conclusion_incorrect :
  ∀ (f : ℝ → ℝ), minor_premise f → ¬ (f = λ x, x^3) := sorry

end major_premise_incorrect_conclusion_incorrect_l389_389558


namespace symmedian_bisects_antiparallel_segment_l389_389907

theorem symmedian_bisects_antiparallel_segment (A B C B1 C1 S : Point)
  (h_B1 : B1 ∈ ray A C) (h_C1 : C1 ∈ ray A B) 
  (h_antiparallel : ∠A B1 C1 = ∠A B C ∧ ∠A C1 B1 = ∠A C B)
  (h_symmedian : is_symmedian_of_triangle A B C S) :
  midpoint_segment S B1 C1 :=
sorry

end symmedian_bisects_antiparallel_segment_l389_389907


namespace reciprocal_of_36_recurring_decimal_l389_389955

-- Definitions and conditions
def recurring_decimal (x : ℚ) : Prop := x = 36 / 99

-- Theorem statement
theorem reciprocal_of_36_recurring_decimal :
  recurring_decimal (36 / 99) → (1 / (36 / 99) = 11 / 4) :=
sorry

end reciprocal_of_36_recurring_decimal_l389_389955


namespace incorrect_statement_among_five_notes_l389_389463

theorem incorrect_statement_among_five_notes (a : ℕ) :
  ¬ (let zhi_len := (2 : ℚ) / 3 * a in
     let shang_len := (4 : ℚ) / 3 * zhi_len in
     let yu_len := (2 : ℚ) / 3 * shang_len in
     let jue_len := (4 : ℚ) / 3 * yu_len in
     shang_len > a ∧ shang_len > zhi_len ∧ shang_len > yu_len ∧ shang_len > jue_len) :=
by sorry

end incorrect_statement_among_five_notes_l389_389463


namespace prime_sequence_bounded_l389_389932

noncomputable def sequence_bound (p : ℕ → ℕ) : Prop :=
  (Prime p 0) ∧ (Prime p 1) ∧
  (∀ (n : ℕ), n ≥ 1 → 
    p (n + 1) = max_prime_factor (p (n - 1) + p n + 100))

theorem prime_sequence_bounded (p : ℕ → ℕ) (hp : sequence_bound p) : 
  ∃ (P : ℕ), ∀ (n : ℕ), p n ≤ P :=
sorry

end prime_sequence_bounded_l389_389932


namespace cone_vertex_angle_l389_389112

theorem cone_vertex_angle (r h : ℝ) (h_gt_0 : 0 < h) (r_sq_eq : r^2 = 4 * h^2 / 3) :
  let α := 2 * Real.arctan(2 / Real.sqrt 3) in
  α = 2 * Real.arctan(2 / Real.sqrt 3) :=
by
  let α := 2 * Real.arctan(2 / Real.sqrt 3)
  exact rfl

end cone_vertex_angle_l389_389112


namespace complex_number_B_height_in_triangle_OAB_l389_389893
noncomputable def complex_add (z1 z2 : ℂ) : ℂ := z1 + z2
noncomputable def norm (z : ℂ) : ℝ := complex.abs z

theorem complex_number_B (zO zA zC zB : ℂ) (h1 : zO = 0) (h2 : zA = 3 + 2 * complex.I) (h3 : zC = -2 + 4 * complex.I) :
  complex_add zA zC = 1 + 6 * complex.I := by
  sorry

theorem height_in_triangle_OAB (zO zA zB : ℂ) (h1 : zO = 0) (h2 : zA = 3 + 2 * complex.I) (h3 : zB = 1 + 6 * complex.I) :
  let OA := zA - zO in
  let OB := zB - zO in
  let cos_angle := (OA.re * OB.re + OA.im * OB.im) / (norm OA * norm OB) in
  let sin_angle := real.sqrt (1 - cos_angle ^ 2) in
  (norm OA * sin_angle) = 16 * real.sqrt 37 / 37 := by
  sorry

end complex_number_B_height_in_triangle_OAB_l389_389893


namespace solve_trig_system_l389_389540

theorem solve_trig_system (x y z : ℝ) (k1 k2 k3 : ℤ)
  (h1 : cos x + cos y + cos z = (3 * sqrt 3) / 2)
  (h2 : sin x + sin y + sin z = 3 / 2) :
  ∃ (k1 k2 k3 : ℤ), x = π / 6 + 2 * k3 * π ∧ y = π / 6 + 2 * k1 * π ∧ z = π / 6 + 2 * k2 * π := 
sorry

end solve_trig_system_l389_389540


namespace three_digit_numbers_count_l389_389439

def is_3_digit_number (n : Nat) : Prop :=
  100 ≤ n ∧ n ≤ 999

def units_digit (n : Nat) : Nat :=
  n % 10

def tens_digit (n : Nat) : Nat :=
  (n / 10) % 10

def hundreds_digit (n : Nat) : Nat :=
  (n / 100) % 10

def valid_number (n : Nat) : Prop :=
  is_3_digit_number n ∧ units_digit n ≥ 3 * tens_digit n

theorem three_digit_numbers_count : 
  (Finset.card (Finset.filter valid_number (Finset.range 1000))) = 198 :=
by
  sorry

end three_digit_numbers_count_l389_389439


namespace log_base_5_of_3100_l389_389943

theorem log_base_5_of_3100 (h1 : 5^4 < 3100) (h2 : 3100 < 5^5) (h3 : 5^5 = 3125) : Int.round (Real.logb 5 3100) = 5 :=
by
  sorry

end log_base_5_of_3100_l389_389943


namespace three_digit_numbers_with_units_at_least_three_times_tens_l389_389396

open Nat

noncomputable def valid_units_and_tens_digit (tens units : ℕ) : Prop :=
  (tens = 0 ∧ units ∈ range 10) ∨
  (tens = 1 ∧ units ∈ range' 3 10) ∨
  (tens = 2 ∧ units ∈ range' 6 10)

noncomputable def valid_hundreds_digit (hundreds : ℕ) : Prop :=
  hundreds ∈ range' 1 10

theorem three_digit_numbers_with_units_at_least_three_times_tens :
  let total_numbers := 189
  ∃ total_numbers : ℕ, total_numbers = ∑ h in range' 1 10, ∑ t in range 10, ∑ u in range 10, if valid_hundreds_digit h ∧ valid_units_and_tens_digit t u then 1 else 0 :=
begin
  sorry
end

end three_digit_numbers_with_units_at_least_three_times_tens_l389_389396


namespace probability_A_and_B_occurs_l389_389905

variable (P : Set α → ℝ) [ProbabilitySpace α]

variable {A B : Set α}

-- Conditions given in the problem
def P_A_occurs : Prop := P A = 0.4
def P_B_occurs : Prop := P B = 0.65
def P_A_or_B_occurs : Prop := P (A ∪ B) = 0.8

-- The theorem to prove
theorem probability_A_and_B_occurs (h_A : P_A_occurs P) (h_B : P_B_occurs P) (h_A_or_B : P_A_or_B_occurs P) : 
  P (A ∩ B) = 0.25 := 
by 
  sorry

end probability_A_and_B_occurs_l389_389905


namespace moveWest_is_negative_sixty_l389_389003

-- The given condition
def moveEast (d: Int) : Int := d -- Moving east by d meters is +d

-- The problem statement to prove
theorem moveWest_is_negative_sixty (d: Int) (h : d = 80) : moveEast d = 80 → moveEast (-60) = -60 :=
by
  intro h1
  rw [moveEast]
  sorry

end moveWest_is_negative_sixty_l389_389003


namespace ratio_of_areas_eq_one_l389_389084

noncomputable def triangle (A B C : Type*) := sorry
noncomputable def point (K L M P Q : Type*) := sorry
noncomputable def circle (circumcircle_abc : Type*) := sorry

-- Definitions from conditions
variables {A B C K L M P Q : Type*}
variables {BKLC_AMB_AMC_cyclic_AM_M_AMC_AMB_P_Q : Prop}
variables {Δ : triangle A B C}
variables {K_pnt : point K}
variables {L_pnt : point L}
variables {P_pnt : point P}
variables {Q_pnt : point Q}
variables {circumcircle_abc : circle Δ}

-- Problem equivalent to finding the ratio of areas
theorem ratio_of_areas_eq_one (h1 : acute Δ) 
  (h2 : points_on_AB_and_AC K_pnt L_pnt)
  (h3 : cyclic_quadrilateral BKLC)
  (h4 : angle_bisector AM)
  (h5 : intersection_with_circumcircle BM circumcircle_abc P_pnt)
  (h6 : intersection_with_circumcircle CM circumcircle_abc Q_pnt) :
  area (triangle A L P) = area (triangle A K Q) :=
sorry

end ratio_of_areas_eq_one_l389_389084


namespace count_perfect_squares_l389_389765

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

def E1 : ℕ := 1^3 + 2^3
def E2 : ℕ := 1^3 + 2^3 + 3^3
def E3 : ℕ := 1^3 + 2^3 + 3^3 + 4^3
def E4 : ℕ := 1^3 + 2^3 + 3^3 + 4^3 + 5^3

theorem count_perfect_squares :
  (is_perfect_square E1 → true) ∧
  (is_perfect_square E2 → true) ∧
  (is_perfect_square E3 → true) ∧
  (is_perfect_square E4 → true) →
  (∀ n : ℕ, (n = 4) ↔
    ∃ E1 E2 E3 E4, is_perfect_square E1 ∧ is_perfect_square E2 ∧ is_perfect_square E3 ∧ is_perfect_square E4) :=
by
  sorry

end count_perfect_squares_l389_389765


namespace expression_eq_one_l389_389508

theorem expression_eq_one (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a + b - c = 0) : 
  (a^2 * b^2 / ((a^2 + b * c) * (b^2 + a * c)) + a^2 * c^2 / ((a^2 + b * c) * (c^2 + a * b)) + b^2 * c^2 / ((b^2 + a * c) * (c^2 + a * b))) = 1 :=
by
  have h5 : c = a + b, by sorry
  sorry

end expression_eq_one_l389_389508


namespace angle_BDC_l389_389543

theorem angle_BDC (A B C D : Type) 
  [triangle ABC] [triangle ACD] 
  (h1 : AB = AC) 
  (h2 : AD > AC) 
  (h3 : ∠ BAC = 30) 
  (h4 : AD = 2 * AC) 
  : ∠ BDC = 15 :=
by
  sorry

end angle_BDC_l389_389543


namespace find_pq_l389_389135

theorem find_pq (p q : ℤ) : (∀ m : ℤ, (9 * m ^ 2 - 2 * m + p) * (4 * m ^ 2 + q * m - 5) = 36 * m ^ 4 - 23 * m ^ 3 - 31 * m ^ 2 + 6 * m - 10) → p + q = 0 :=
by 
  intros h,
  sorry

end find_pq_l389_389135


namespace right_triangle_height_on_hypotenuse_l389_389448

theorem right_triangle_height_on_hypotenuse
  (a b : ℕ) (h : ℝ)
  (ha : a = 3) (hb : b = 6)
  (hypotenuse : ℝ) (hhypotenuse : hypotenuse = real.sqrt (a^2 + b^2))
  (area_legs : ℝ) (harea_legs : area_legs = 1/2 * a * b)
  (area_hypotenuse : ℝ) (harea_hypotenuse : area_hypotenuse = 1/2 * hypotenuse * h) :
  h = 6 * real.sqrt 5 / 5 := 
sorry

end right_triangle_height_on_hypotenuse_l389_389448


namespace distance_between_intersections_polar_equations_l389_389467

noncomputable def C1_cartesian (x y : ℝ) : Prop := (x - real.sqrt 3)^2 + (y - 1)^2 = 4

noncomputable def C1_polar (ρ θ : ℝ) : Prop := ρ = 2 * real.sqrt 3 * real.cos θ + 2 * real.sin θ

noncomputable def C2_polar (ρ θ : ℝ) : Prop := ρ = 2 * real.cos θ + 2 * real.sqrt 3 * real.sin θ

noncomputable def ray_theta (θ : ℝ) : Prop := θ = real.pi / 3

theorem distance_between_intersections :
  let A_θ := real.pi / 3,
      A_ρ := 2 * real.sqrt 3 * real.cos A_θ + 2 * real.sin A_θ,
      B_ρ := 2 * real.cos A_θ + 2 * real.sqrt 3 * real.sin A_θ in
  |A_ρ - B_ρ| = 4 - 2 * real.sqrt 3 :=
by
  sorry

theorem polar_equations :
  ∀ (ρ θ : ℝ),
  C1_cartesian ρ θ ↔ C1_polar ρ θ ∧ C2_polar ρ θ :=
by
  sorry

end distance_between_intersections_polar_equations_l389_389467


namespace probability_of_prime_and_multiple_of_3_l389_389678

-- Definitions based on the conditions
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def is_multiple_of_3 (n : ℕ) : Prop :=
  n % 3 = 0

-- Definition of the probability function to compute the desired probability
noncomputable def probability_prime_multiple_of_3 : ℚ :=
  let primes := {n ∈ finset.range 101 | is_prime n}
  let multiples_of_3 := {n ∈ finset.range 101 | is_multiple_of_3 n}
  let prime_multiples_of_3 := primes ∩ multiples_of_3
  prime_multiples_of_3.card / 100

-- The problem statement: Prove the probability is 1/100
theorem probability_of_prime_and_multiple_of_3 : probability_prime_multiple_of_3 = 1 / 100 := by
  sorry

end probability_of_prime_and_multiple_of_3_l389_389678


namespace clothing_discounted_to_fraction_of_original_price_l389_389611

-- Given conditions
variable (P : ℝ) (f : ℝ)

-- Price during first sale is fP, price during second sale is 0.5P
-- Price decreased by 40% from first sale to second sale
def price_decrease_condition : Prop :=
  f * P - (1/2) * P = 0.4 * (f * P)

-- The main theorem to prove
theorem clothing_discounted_to_fraction_of_original_price (h : price_decrease_condition P f) :
  f = 5/6 :=
sorry

end clothing_discounted_to_fraction_of_original_price_l389_389611


namespace quadratic_function_unique_l389_389323

noncomputable def f (x : ℝ) : ℝ := (1/4 : ℝ) * x^2 + (1/2 : ℝ) * x + (1/4 : ℝ)

theorem quadratic_function_unique (f : ℝ → ℝ)
  (h1 : f (-1) = 0)
  (h2 : ∀ x, x ≤ f x ∧ f x ≤ (1 + x^2) / 2) :
  f = λ x, (1/4 : ℝ) * x^2 + (1/2 : ℝ) * x + (1/4 : ℝ) :=
by
  sorry

end quadratic_function_unique_l389_389323


namespace tangent_line_at_2_number_of_solutions_l389_389361

def f (a x : ℝ) : ℝ := 1/2 * x^2 - a * Real.log x

theorem tangent_line_at_2 (a b : ℝ) :
  (deriv (f a) 2 = 1) → (f a 2 = 2 + b) → a = 2 ∧ b = -2 * Real.log 2 :=
by {
  sorry
}

theorem number_of_solutions (a : ℝ) :
  (if a = 0 then ∀ x, f a x ≠ 0 else
  if a < 0 then ∃! x, f a x = 0 else
  if 0 < a ∧ a < Real.exp 1 then ∀ x, f a x ≠ 0 else
  if a = Real.exp 1 then ∃! x, f a x = 0 else
  ∃ x y, x ≠ y ∧ f a x = 0 ∧ f a y = 0) :=
by {
  sorry
}

end tangent_line_at_2_number_of_solutions_l389_389361


namespace vector_expression_l389_389340

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (i j k a b : V)
variables (h_i_j_k_non_coplanar : ∃ (l m n : ℝ), l • i + m • j + n • k = 0 → l = 0 ∧ m = 0 ∧ n = 0)
variables (h_a : a = (1 / 2 : ℝ) • i - j + k)
variables (h_b : b = 5 • i - 2 • j - k)

theorem vector_expression :
  4 • a - 3 • b = -13 • i + 2 • j + 7 • k :=
by
  sorry

end vector_expression_l389_389340


namespace least_number_added_divisible_by_23_and_17_l389_389584

theorem least_number_added_divisible_by_23_and_17 : 
  ∃ x, (x + 1077) % (Nat.lcm 23 17) = 0 ∧ x = 96 :=
by
  have lcm_23_17 : Nat.lcm 23 17 = 391 := by norm_num
  use 96
  split
  · norm_num
  · sorry

end least_number_added_divisible_by_23_and_17_l389_389584


namespace log_cubert_27_l389_389286

theorem log_cubert_27 : log 3 (3 ^ (27 ^ (1/3))) = 1 :=
by sorry

end log_cubert_27_l389_389286


namespace tan_A_of_triangle_l389_389800
-- Import Mathlib to bring in the necessary libraries

-- Define the problem conditions and translate it to Lean
theorem tan_A_of_triangle (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (angle_C : angle B A C = 60) (AB : dist A B = 12) (BC : dist B C = 2 * Real.sqrt 21) :
  Real.tan A = 2 * Real.sqrt 21 / Real.sqrt (228 - 24 * Real.sqrt 21) := 
sorry

end tan_A_of_triangle_l389_389800


namespace eulerian_path_exists_variant1_variant2_variant3_variant4_variant5_l389_389624

-- Define the graphs for each variant with respective degrees
structure Graph :=
  (V : Type)
  (E : V → ℕ)

-- Variant 1
def G1 : Graph := {
  V := ℕ,
  E := λ n, if n = 1 ∨ n = 2 ∨ n = 5 ∨ n = 7 ∨ n = 8 then 3 else 2
}

-- Variant 2
def G2 : Graph := {
  V := ℕ,
  E := λ n, if n = 1 ∨ n = 5 ∨ n = 6 ∨ n = 7 then 3 else 2
}

-- Variant 3
def G3 : Graph := {
  V := ℕ,
  E := λ n, if n = 4 ∨ n = 7 ∨ n = 8 then 3 else 2
}

-- Variant 4
def G4 : Graph := {
  V := ℕ,
  E := λ n, if n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 5 ∨ n = 7 then 3 else 2
}

-- Variant 5
def G5 : Graph := {
  V := ℕ,
  E := λ n, if n = 2 ∨ n = 6 then 3 else 2
}

-- Main theorem to determine starting/ending vertices with Euler path

theorem eulerian_path_exists (G : Graph) (odd_vertices : set G.V) :
  (∀ v ∈ odd_vertices, G.E v % 2 = 1) ∧ set.card odd_vertices = 2 → 
  ∃ start end : G.V, start ∈ odd_vertices ∧ end ∈ odd_vertices :=
begin
  intros h,
  sorry -- proof omitted
end

-- Applying the theorem for each variant to find valid starting points

theorem variant1 : ∃ start end : G1.V, (start = 2 ∨ start = 5) ∧ (end = 2 ∨ end = 5) :=
  eulerian_path_exists G1 {2, 5} by {
    -- Verifying the degrees and cardinality conditions
    split,
    { intros v h,
      simp [G1.E],
      -- Each vertex in {2, 5} has an odd degree of 3
      sorry -- proof omitted
    },
    -- There are exactly 2 vertices with odd degrees in set {2, 5}
    exact rfl
  }

theorem variant2 : ∃ start end : G2.V, (start = 1 ∨ start = 6) ∧ (end = 1 ∨ end = 6) :=
  eulerian_path_exists G2 {1, 6} by {
    split,
    { intros v h,
      simp [G2.E],
      sorry -- proof omitted
    },
    exact rfl
  }

theorem variant3 : ∃ start end : G3.V, (start = 7 ∨ start = 8) ∧ (end = 7 ∨ end = 8) :=
  eulerian_path_exists G3 {7, 8} by {
    split,
    { intros v h,
      simp [G3.E],
      sorry -- proof omitted
    },
    exact rfl
  }

theorem variant4 : ∃ start end : G4.V, (start = 2 ∨ start = 4) ∧ (end = 2 ∨ end = 4) :=
  eulerian_path_exists G4 {2, 4} by {
    split,
    { intros v h,
      simp [G4.E],
      sorry -- proof omitted
    },
    exact rfl
  }

theorem variant5 : ∃ start end : G5.V, (start = 2 ∨ start = 6) ∧ (end = 2 ∨ end = 6) :=
  eulerian_path_exists G5 {2, 6} by {
    split,
    { intros v h,
      simp [G5.E],
      sorry -- proof omitted
    },
    exact rfl
  }

end eulerian_path_exists_variant1_variant2_variant3_variant4_variant5_l389_389624


namespace largest_divisor_expression_l389_389446

theorem largest_divisor_expression (y : ℤ) (h : y % 2 = 1) : 
  4320 ∣ (15 * y + 3) * (15 * y + 9) * (10 * y + 10) :=
sorry  

end largest_divisor_expression_l389_389446


namespace bees_direction_when_12_feet_apart_l389_389155

-- Define bee travel patterns
def bee_A_pattern : List (Int × Int × Int) := [(0, 1, 0), (1, 0, 0), (0, 0, 1), (0, -1, 0)]
def bee_B_pattern : List (Int × Int × Int) := [(0, -2, 0), (-2, 0, 0), (0, 0, 2)]

-- Define the position calculation
def calc_position (pattern : List (Int × Int × Int)) (steps : Nat) : Int × Int × Int :=
  let full_cycles := steps / pattern.length
  let remaining_steps := steps % pattern.length
  let basic_position := pattern.take remaining_steps |>.foldl (λ acc val => (acc.1 + val.1, acc.2 + val.2, acc.3 + val.3)) (0, 0, 0)
  let cycle_position := pattern.foldl (λ acc val => (acc.1 + val.1, acc.2 + val.2, acc.3 + val.3)) (0, 0, 0)
  (full_cycles * cycle_position.1 + basic_position.1,
   full_cycles * cycle_position.2 + basic_position.2,
   full_cycles * cycle_position.3 + basic_position.3)

-- Define the distance function
def distance (p1 p2 : Int × Int × Int) : Real :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2 + (p1.3 - p2.3) ^ 2)

-- Prove the problem statement
theorem bees_direction_when_12_feet_apart (n m : Nat) :
  distance (calc_position bee_A_pattern n) (calc_position bee_B_pattern m) = 12 →
  (List.nth bee_A_pattern (n % bee_A_pattern.length) = some (0, -1, 0)) ∧
  (List.nth bee_B_pattern (m % bee_B_pattern.length) = some (0, 0, 2)) :=
sorry

end bees_direction_when_12_feet_apart_l389_389155


namespace max_int_with_10_divisors_and_8th_divisor_is_n_div_3_l389_389560

theorem max_int_with_10_divisors_and_8th_divisor_is_n_div_3 : 
  ∃ n : ℕ, (finset.card (finset.divisors n) = 10) ∧
  (finset.divisors n).sort (≤).nth 7 = some (n / 3) ∧
  ∀ m : ℕ, (finset.card (finset.divisors m) = 10) ∧
  (finset.divisors m).sort (≤).nth 7 = some (m / 3) → m ≤ 162 :=
begin
  sorry
end

end max_int_with_10_divisors_and_8th_divisor_is_n_div_3_l389_389560


namespace ratio_correct_l389_389576

-- Define the initial conditions
def initial_stamps := 3000
def stamps_from_mike := 17
def final_stamps := 3061

-- Define the number of stamps given by Harry
def stamps_from_harry (k : ℕ) := k * stamps_from_mike + 10

-- Main definition to calculate the ratio and prove it
def ratio_of_harry_to_mike (k : ℕ) : (ℚ × ℚ) :=
  let h := stamps_from_harry k in
  if (initial_stamps + h + stamps_from_mike = final_stamps) then
    (h, stamps_from_mike)
  else
    (0, 0)

-- Proof that the ratio is 44:17 given the conditions
theorem ratio_correct (k : ℕ) : ratio_of_harry_to_mike k = (44, 17) :=
by
  sorry

end ratio_correct_l389_389576


namespace positional_relationship_planes_l389_389449

-- Given conditions
def a : ℝ × ℝ × ℝ := (1, 0, -2)
def b : ℝ × ℝ × ℝ := (-1, 0, 2)

-- A ∥ B means the planes are parallel.
def planes_parallel (a b : ℝ × ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ (a.1 = k * b.1) ∧ (a.2 = k * b.2) ∧ (a.3 = k * b.3)

theorem positional_relationship_planes : planes_parallel a b :=
  sorry

end positional_relationship_planes_l389_389449


namespace correct_statements_l389_389314

-- Define a complex number
noncomputable theory
def complex : Type := ℂ

-- Define problem conditions and properties

def z_property_A (z : ℂ) : Prop := z = conj(z) → (z.im = 0)

def z_property_B (z : ℂ) : Prop := z = I * (1 - 2 * I) → (z.im = 1)

def z_property_C (z : ℂ) (a : ℝ) : Prop := z = a + I → ∥z∥ = real.sqrt 2 → (a = 1 ∨ a = -1)

def z_property_D (z : ℂ) : Prop := ∥z∥ = 1 → (∥z + 1∥ ≤ 2)

-- The goal to prove
theorem correct_statements :
  ∀ z : ℂ, z_property_B z ∧ z_property_D z := by
  intros
  -- Specific proof steps would go here
  sorry

end correct_statements_l389_389314


namespace geometric_series_sum_l389_389675

theorem geometric_series_sum :
  let a := -2
  let r := 4
  let n := 10
  let S := (a * (r^n - 1)) / (r - 1)
  S = -699050 :=
by
  sorry

end geometric_series_sum_l389_389675


namespace sum_of_real_values_l389_389699

theorem sum_of_real_values (y : ℝ) (h : y = 1 + y - y^2 + y^3 - y^4 + y^5 - ∞) : y = 1 := 
by 
  have h_series : y = 1 + y - y^2 + y^3 - y^4 + y^5 - ∞,
  from h,
  have h_eq : y = 1 / (1 - y + y^2),
  sorry,
  have h_real_solutions : y = 1,
  sorry,
  sorry

end sum_of_real_values_l389_389699


namespace remi_spilled_second_time_l389_389092

-- Defining the conditions from the problem
def bottle_capacity : ℕ := 20
def daily_refills : ℕ := 3
def total_days : ℕ := 7
def total_water_consumed : ℕ := 407
def first_spill : ℕ := 5

-- Using the conditions to define the total amount of water that Remi would have drunk without spilling.
def no_spill_total : ℕ := bottle_capacity * daily_refills * total_days

-- Defining the second spill
def second_spill : ℕ := no_spill_total - first_spill - total_water_consumed

-- Stating the theorem that we need to prove
theorem remi_spilled_second_time : second_spill = 8 :=
by
  sorry

end remi_spilled_second_time_l389_389092


namespace problem_def_l389_389740

def g (f : ℝ → ℝ) (x : ℝ) := exp x * f x - exp x - 7

theorem problem_def (f : ℝ → ℝ) 
  (h_domain : ∀ x, f x ∈ ℝ)
  (h_ineq : ∀ x, 2 * f x * 2 * (deriv f x) > 2)
  (h_init : f 0 = 8):
  {x : ℝ | (f x - 1) / exp (log 7 - x) > 1} = set.Ioi 0 :=
sorry

end problem_def_l389_389740


namespace contains_zero_l389_389235

-- Define what constitutes a five-digit number
def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

-- Define the main proof statement
theorem contains_zero (n1 n2 : ℕ) (c1 c2 : Prop) :
  is_five_digit n1 →
  is_five_digit n2 →
  c1 ∧ c2 →
  n1 + n2 = 111111 →
  ∃ (d : ℕ), d < 10 ∧ (d = 0 ∧ (∃ p1 p2 : ℕ, n1 = p1 * 10 + d ∧ n2 = p2 * 10 + d) ∨ 
             ∃ p1 p2 : ℕ, n1 = d * (10^4) + p1 ∧ n2 = d * (10^4) + p2) :=
begin
  sorry
end

end contains_zero_l389_389235


namespace pow_zero_eq_one_l389_389657

theorem pow_zero_eq_one : (-2023)^0 = 1 :=
by
  -- The proof of this theorem will go here.
  sorry

end pow_zero_eq_one_l389_389657


namespace joe_initial_cars_l389_389042

theorem joe_initial_cars (x : ℕ) (h : x + 12 = 62) : x = 50 :=
by {
  sorry
}

end joe_initial_cars_l389_389042


namespace problem_solution_l389_389497

theorem problem_solution
  (a b c : ℝ)
  (habc_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_condition : a / (b - c) + b / (c - a) + c / (a - b) = 0) :
  a^2 / (b - c)^2 + b^2 / (c - a)^2 + c^2 / (a - b)^2 = 0 :=
by
  sorry

end problem_solution_l389_389497


namespace equilibrium_asymptotically_stable_l389_389298

theorem equilibrium_asymptotically_stable :
  ∀ (x y : ℝ), 
    (dx dt == y - (x/2) - (x * y^3)/2) ∧ 
    (dy dt == -y - 2*x + (x^2)*y^2) →
    ∃ V : ℝ → ℝ → ℝ, (V x y = 2*x^2 + y^2) ∧
      (∀ (dx dt dy dt x y : ℝ),
        x ≠ 0 ∨ y ≠ 0 →
        dv/dt ≤ 0 ∧ (dv/dt == 0 ↔ (x, y) = (0, 0))) :=
by
  sorry

end equilibrium_asymptotically_stable_l389_389298


namespace cubes_divisible_by_nine_l389_389930

theorem cubes_divisible_by_nine (n : ℕ) (hn : n > 0) : 
    (n^3 + (n + 1)^3 + (n + 2)^3) % 9 = 0 := by
  sorry

end cubes_divisible_by_nine_l389_389930


namespace angle_ABC_sixty_l389_389067

universe u

variable {I A B C A' B' C' : Point}

def is_incenter (I : Point) (A B C : Point) : Prop :=
  I.dist A = I.dist B ∧ I.dist B = I.dist C

def is_reflection (P Q R : Point) (I P' : Point) : Prop :=
  I.dist P' = 2 * I.dist P ∧ I.dist P' = I.dist Q ∧ I.dist P' = I.dist R

def circumcircle_passes_through (A' B' C' B : Point) : Prop :=
  ∃ O r, is_center_of_circle O ∧ is_radius_of_circle r ∧ 
  (A'.dist O = r ∧ B'.dist O = r ∧ C'.dist O = r ∧ B.dist O = r)

theorem angle_ABC_sixty {I A B C A' B' C' : Point}
  (H1 : is_incenter I A B C)
  (H2 : is_reflection I B C A A')
  (H3 : is_reflection I A C B B')
  (H4 : is_reflection I A B C C')
  (H5 : circumcircle_passes_through A' B' C' B) :
  angle_ABC A B C = 60 := sorry

end angle_ABC_sixty_l389_389067


namespace prob_same_color_is_correct_l389_389443

-- Define the sides of one die
def blue_sides := 6
def yellow_sides := 8
def green_sides := 10
def purple_sides := 6
def total_sides := 30

-- Define the probability each die shows a specific color
def prob_blue := blue_sides / total_sides
def prob_yellow := yellow_sides / total_sides
def prob_green := green_sides / total_sides
def prob_purple := purple_sides / total_sides

-- The probability that both dice show the same color
def prob_same_color :=
  (prob_blue * prob_blue) + 
  (prob_yellow * prob_yellow) + 
  (prob_green * prob_green) + 
  (prob_purple * prob_purple)

-- We should prove that the computed probability is equal to the given answer
theorem prob_same_color_is_correct :
  prob_same_color = 59 / 225 := 
sorry

end prob_same_color_is_correct_l389_389443


namespace polynomial_divisible_by_5040_l389_389533

theorem polynomial_divisible_by_5040 (n : ℤ) (hn : n > 3) :
  5040 ∣ (n^7 - 14 * n^5 + 49 * n^3 - 36 * n) :=
sorry

end polynomial_divisible_by_5040_l389_389533


namespace units_digit_sum_seq_l389_389993

theorem units_digit_sum_seq : 
  let seq := (λ n : ℕ, (n ! * n)) in
  let units_digit := (λ x : ℕ, x % 10) in
  let sum_units_digit := ∑ n in range 1 11, units_digit (seq n) in
  units_digit sum_units_digit = 9 :=
by sorry

end units_digit_sum_seq_l389_389993


namespace ratio_of_peters_french_bulldogs_l389_389530

theorem ratio_of_peters_french_bulldogs (n_gs_sam n_fb_sam : ℕ) (n_dogs_peter n_gs_peter : ℕ) (ratio : ℕ) :
  n_gs_sam = 3 →
  n_fb_sam = 4 →
  n_dogs_peter = 17 →
  n_gs_peter = 3 * n_gs_sam →
  n_dogs_peter = n_gs_peter + n_fb_sam * ratio →
  ratio = 2 :=
by
  intros h1 h2 h3 h4 h5
  have h6 : n_gs_peter = 9 := by rw [h1, mul_comm, mul_assoc, Nat.mul_assoc, Nat.add_mul_mod_self, Nat.add_mod_right, Nat.one_mul, Nat.gcd_eq_mod]
  have h7 : 17 - n_gs_peter = 8 := by rw [h3, h6, Nat.sub_eq_sub_min_eq, Nat.sub_eq_its_add_self, Nat.add_eq_mod_self, Nat.add_right_eq_mod_self, Nat.add_comm]
  have h8 : 8 / 4 = 2 := by rw [Nat.div_eq_val_eq, Nat.add, Nat.neg_mod_not]
  rw [h5] at h8
  exact h8

end ratio_of_peters_french_bulldogs_l389_389530


namespace quadratic_equation_only_option_B_l389_389164

theorem quadratic_equation_only_option_B (a b c : ℝ) (x : ℝ):
  (a ≠ 0 → (a * x^2 + b * x + c = 0)) ∧              -- Option A
  (3 * (x + 1)^2 = 2 * (x - 2) ↔ 3 * x^2 + 4 * x + 7 = 0) ∧  -- Option B
  (1 / x^2 + 1 = x^2 + 1 → False) ∧         -- Option C
  (1 / x^2 + 1 / x - 2 = 0 → False) →       -- Option D
  -- Option B is the only quadratic equation.
  (3 * (x + 1)^2 = 2 * (x - 2)) :=
sorry

end quadratic_equation_only_option_B_l389_389164


namespace initial_roses_l389_389617

theorem initial_roses (x : ℕ) (h1 : x - 3 + 34 = 36) : x = 5 :=
by 
  sorry

end initial_roses_l389_389617


namespace solution_proof_l389_389052

variable (A B C : ℕ+) (x y : ℚ)
variable (h1 : A > B) (h2 : B > C) (h3 : A = B * (1 + x / 100)) (h4 : B = C * (1 + y / 100))

theorem solution_proof : x = 100 * ((A / (C * (1 + y / 100))) - 1) :=
by
  sorry

end solution_proof_l389_389052


namespace basketball_player_scores_mode_median_l389_389612

theorem basketball_player_scores_mode_median :
  let scores := [20, 18, 23, 17, 20, 20, 18]
  let ordered_scores := List.sort scores
  let mode := 20
  let median := 20
  (mode = List.maximum (List.frequency ordered_scores)) ∧ 
  (median = List.nthLe ordered_scores (List.length ordered_scores / 2) sorry) :=
by
  sorry

end basketball_player_scores_mode_median_l389_389612


namespace plane_equation_l389_389696

theorem plane_equation 
    (P: ℝ × ℝ × ℝ := (1, 4, -3))
    (line : ℝ → ℝ × ℝ × ℝ := λ t, (4 * t + 2, -t - 1, 2 * t + 3)) : 
    ∃ A B C D : ℤ, 
        (A > 0) ∧ 
        (Int.gcd4 (Int.natAbs A) (Int.natAbs B) (Int.natAbs C) (Int.natAbs D) = 1) ∧ 
        (∀ (x y z : ℝ), (x, y, z) = P ∨ (∃ t : ℝ, (x, y, z) = line t) → A * x + B * y + C * z + D = 0) := 
begin
    use [5, -22, -21, 41],
    split,
    { exact dec_trivial },
    split,
    { exact dec_trivial },
    intros x y z h,
    cases h,
    { subst h,
      simp, },
    { rcases h with ⟨t, rfl⟩,
      simp, }
end

end plane_equation_l389_389696


namespace max_a_for_integer_roots_l389_389745

theorem max_a_for_integer_roots (a : ℕ) :
  (∀ x : ℤ, x^2 - 2 * (a : ℤ) * x + 64 = 0 → (∃ y : ℤ, x = y)) →
  (∀ x1 x2 : ℤ, x1 * x2 = 64 ∧ x1 + x2 = 2 * (a : ℤ)) →
  a ≤ 17 := 
sorry

end max_a_for_integer_roots_l389_389745


namespace find_f_of_one_l389_389715

def f (x : ℝ) : ℝ := 3 * x - 1

theorem find_f_of_one : f 1 = 2 := 
by
  sorry

end find_f_of_one_l389_389715


namespace perpendicular_case_parallel_case_l389_389375

variables {x : ℝ}
def a := (1 : ℝ, 3 : ℝ)
def b := (x^2 - 1, x + 1)

-- Perpendicular case
theorem perpendicular_case (h_ab_perp : (1 * (x^2 - 1) + 3 * (x + 1)) = 0) : x = -1 ∨ x = -2 :=
sorry

-- Parallel case
theorem parallel_case (h_ab_parallel : (x + 1) = 3 * (x^2 - 1)) : 
  (|((1 - (x^2 - 1), 3 - (x + 1)) : ℝ × ℝ)| = ∥(1, 3)∥ ∨
   ∥((1 - (x^2 - 1), 3 - (x + 1)) : ℝ × ℝ)∥ = (2 * real.sqrt 10) / 9) :=
sorry

end perpendicular_case_parallel_case_l389_389375


namespace cubic_roots_sum_of_products_l389_389845

theorem cubic_roots_sum_of_products :
  let p q r : ℝ := sorry in
  (∀ x : ℝ, 6 * x^3 - 4 * x^2 + 7 * x - 3 = 0 → (x = p ∨ x = q ∨ x = r)) →
  (pq + qr + rp = 7 / 6) := by
  sorry

end cubic_roots_sum_of_products_l389_389845


namespace find_f_40_l389_389117

variable {f : ℝ → ℝ}

axiom f_eq (x y : ℝ) (hx : x > 0) (hy : y > 0) : f (x * y) = f x / (y ^ 2)
axiom f_30 : f 30 = 10

theorem find_f_40 : f 40 = 45 / 8 :=
by
  have h_pos : 4 / 3 > 0 := by norm_num
  have h_eq := f_eq 30 (4 / 3) (by norm_num) h_pos
  rw [mul_div_cancel''] at h_eq
  rw [f_30] at h_eq
  norm_num at h_eq
  exact h_eq

end find_f_40_l389_389117


namespace mod_1237_17_l389_389284

theorem mod_1237_17 : 1237 % 17 = 13 := by
  sorry

end mod_1237_17_l389_389284


namespace daniel_fraction_l389_389643

theorem daniel_fraction (A B C D : Type) (money : A → ℝ) 
  (adriano bruno cesar daniel : A)
  (h1 : money daniel = 0)
  (given_amount : ℝ)
  (h2 : money adriano = 5 * given_amount)
  (h3 : money bruno = 4 * given_amount)
  (h4 : money cesar = 3 * given_amount)
  (h5 : money daniel = (1 / 5) * money adriano + (1 / 4) * money bruno + (1 / 3) * money cesar) :
  money daniel / (money adriano + money bruno + money cesar) = 1 / 4 := 
by
  sorry

end daniel_fraction_l389_389643


namespace largest_integer_N_exists_l389_389682

/-- The largest integer N for which there exists a table T of integers with 
    N rows and 100 columns that satisfies the following properties: 
    1. Every row contains the numbers 1, 2, ..., 100 in some order. 
    2. For any two distinct rows r and s, there is a column c such that 
       |T(r, c) - T(s, c)| ≥ 2 
    is exactly 50. -/
theorem largest_integer_N_exists (N : ℕ) :
  (∃ T : matrix (fin N) (fin 100) ℕ, 
    (∀ r : fin N, (∃ σ : fin 100 → fin 100, 
      ∀ c : fin 100, T r c = σ c + 1)) ∧ 
    (∀ (r s : fin N), r ≠ s → ∃ c : fin 100, |T r c - T s c| ≥ 2)) ↔ N = 50 :=
sorry

end largest_integer_N_exists_l389_389682


namespace discount_percentage_is_correct_l389_389137

-- Define regular price per can
def regular_price_per_can : ℝ := 0.60

-- Define the discounted price for 72 cans
def discounted_price_for_72_cans : ℝ := 34.56

-- Define the total regular price for 72 cans
def total_regular_price_for_72_cans : ℝ := 72 * regular_price_per_can

-- Define the discount amount
def discount_amount : ℝ := total_regular_price_for_72_cans - discounted_price_for_72_cans

-- Define the discount percentage
def discount_percentage : ℝ := (discount_amount / total_regular_price_for_72_cans) * 100

-- The proof goal
theorem discount_percentage_is_correct : discount_percentage = 20 :=
by
  sorry

end discount_percentage_is_correct_l389_389137


namespace subset_set_count_l389_389904

theorem subset_set_count :
  {A : set ℕ // {1,2} ⊆ A ∧ A ⊆ {1,2,3,4,5} ∧ A ≠ {1, 2, 3, 4, 5}}.card = 7 :=
by
  sorry

end subset_set_count_l389_389904


namespace total_alternating_sum_eq_1024_l389_389307

def subset_alternating_sum (s : Finset ℕ) : ℤ :=
  s.sort (· > ·).foldr (λ x (acc: ℤ) i, if i % 2 = 0 then acc + x else acc - x) (0 : ℤ)

theorem total_alternating_sum_eq_1024 : 
  let n := 8,
  let S := (Finset.range (n + 1)).erase 0, -- {1, 2, ..., 8}
  (Finset.powerset S).val.sum (λ s, subset_alternating_sum s) = 1024 :=
by
  sorry

end total_alternating_sum_eq_1024_l389_389307


namespace mildred_weight_l389_389079

theorem mildred_weight (carol_weight mildred_is_heavier : ℕ) (h1 : carol_weight = 9) (h2 : mildred_is_heavier = 50) :
  carol_weight + mildred_is_heavier = 59 :=
by
  sorry

end mildred_weight_l389_389079


namespace inequality_X_Y_l389_389816

noncomputable def X (a : ℕ → ℝ) (k : ℕ) : ℝ :=
∀ k, ∑ i in finset.range (2^k + 1), a i

noncomputable def Y (a : ℕ → ℝ) (k : ℕ) : ℝ :=
∀ k, ∑ i in finset.range (2^k + 1), ⌊(2^k : ℝ) / i⌋ * a i

theorem inequality_X_Y (a : ℕ → ℝ) [∀ n, 0 ≤ a n] (n : ℕ) (hn : n > 0) : 
  X a n ≤ Y a n - ∑ i in finset.range n, Y a i ∧ Y a n - ∑ i in finset.range n, Y a i ≤ ∑ i in finset.range (n+1), X a i :=
sorry

end inequality_X_Y_l389_389816


namespace CauchySchwarz_inequality_inequality_sqrt_l389_389734

variable {n : ℕ}
variable {a : Fin n → ℝ}
variable {x : ℝ}

theorem CauchySchwarz_inequality (h1 : ∀ i, a i ∈ ℝ) : (∑ i, a i)^2 ≤ n * (∑ i, (a i)^2) := 
by
  sorry
  
theorem inequality_sqrt (h2 : 3/2 ≤ x ∧ x ≤ 5) : 2 * Real.sqrt (x + 1) + Real.sqrt (2 * x - 3) + Real.sqrt (15 - 3 * x) < 2 * Real.sqrt (19) := 
by
  sorry

end CauchySchwarz_inequality_inequality_sqrt_l389_389734


namespace evaluate_expression_l389_389262

theorem evaluate_expression :
  -1^2 + (1 / 2)⁻² - (3.14 - Real.pi)^0 = 2 :=
by sorry

end evaluate_expression_l389_389262


namespace average_books_collected_per_day_l389_389484

theorem average_books_collected_per_day :
  let n := 7
  let a := 12
  let d := 12
  let S_n := (n * (2 * a + (n - 1) * d)) / 2
  S_n / n = 48 :=
by
  let n := 7
  let a := 12
  let d := 12
  let S_n := (n * (2 * a + (n - 1) * d)) / 2
  show S_n / n = 48
  sorry

end average_books_collected_per_day_l389_389484


namespace three_digit_numbers_with_units_at_least_three_times_tens_l389_389399

open Nat

noncomputable def valid_units_and_tens_digit (tens units : ℕ) : Prop :=
  (tens = 0 ∧ units ∈ range 10) ∨
  (tens = 1 ∧ units ∈ range' 3 10) ∨
  (tens = 2 ∧ units ∈ range' 6 10)

noncomputable def valid_hundreds_digit (hundreds : ℕ) : Prop :=
  hundreds ∈ range' 1 10

theorem three_digit_numbers_with_units_at_least_three_times_tens :
  let total_numbers := 189
  ∃ total_numbers : ℕ, total_numbers = ∑ h in range' 1 10, ∑ t in range 10, ∑ u in range 10, if valid_hundreds_digit h ∧ valid_units_and_tens_digit t u then 1 else 0 :=
begin
  sorry
end

end three_digit_numbers_with_units_at_least_three_times_tens_l389_389399


namespace best_fitting_model_is_model_2_l389_389926

-- Variables representing the correlation coefficients of the four models
def R2_model_1 : ℝ := 0.86
def R2_model_2 : ℝ := 0.96
def R2_model_3 : ℝ := 0.73
def R2_model_4 : ℝ := 0.66

-- Statement asserting that Model 2 has the best fitting effect
theorem best_fitting_model_is_model_2 :
  R2_model_2 = 0.96 ∧ R2_model_2 > R2_model_1 ∧ R2_model_2 > R2_model_3 ∧ R2_model_2 > R2_model_4 :=
by {
  sorry
}

end best_fitting_model_is_model_2_l389_389926


namespace sum_first_n_natural_numbers_l389_389882

theorem sum_first_n_natural_numbers (n : ℕ) :
  (∑ k in Finset.range (n + 1), k) = (n * (n + 1)) / 2 :=
by
  sorry

end sum_first_n_natural_numbers_l389_389882


namespace sum_of_quotients_geq_n_over_n_minus_1_proof_l389_389877

noncomputable def sum_of_quotients_geq_n_over_n_minus_1 (n : ℕ) (a : ℕ → ℝ) :=
  (∀ i, 1 ≤ i ∧ i ≤ n → a i > 0) →
  n ≥ 2 →
  let s := ∑ i in Finset.range n, a i in
  ∑ i in Finset.range n, a i / (s - a i) ≥ n / (n - 1)

theorem sum_of_quotients_geq_n_over_n_minus_1_proof (a : ℕ → ℝ) (n : ℕ) :
  sum_of_quotients_geq_n_over_n_minus_1 n a :=
begin
  -- assuming all conditions
  intros h_pos hn_ge2,
  let s := ∑ i in Finset.range n, a i,
  sorry
end

end sum_of_quotients_geq_n_over_n_minus_1_proof_l389_389877


namespace min_draw_three_colors_l389_389183

-- Given conditions
def total_balls : ℕ := 111
def colors : ℕ := 4
def min_four_colors_draw : ℕ := 100

-- The question to be proved
def min_three_colors_draw : ℕ := 88

theorem min_draw_three_colors {total_balls colors min_four_colors_draw : ℕ} :
  total_balls = 111 →
  colors = 4 →
  min_four_colors_draw = 100 →
  min_draw_three_colors = 88 →
  ∃ n, n ≤ min_draw_three_colors ∧ at_least_three_colors n :=
sorry

end min_draw_three_colors_l389_389183


namespace maximize_profit_l389_389998

noncomputable def profit_function (x : ℝ) : ℝ :=
if 0 < x ∧ x ≤ 40 then
  -2 * x^2 + 120 * x - 300
else if 40 < x ∧ x ≤ 100 then
  -x - 3600 / x + 1800
else
  0

theorem maximize_profit :
  profit_function 60 = 1680 ∧
  ∀ x, 0 < x ∧ x ≤ 100 → profit_function x ≤ 1680 := 
sorry

end maximize_profit_l389_389998


namespace three_digit_numbers_with_units_at_least_three_times_tens_l389_389398

open Nat

noncomputable def valid_units_and_tens_digit (tens units : ℕ) : Prop :=
  (tens = 0 ∧ units ∈ range 10) ∨
  (tens = 1 ∧ units ∈ range' 3 10) ∨
  (tens = 2 ∧ units ∈ range' 6 10)

noncomputable def valid_hundreds_digit (hundreds : ℕ) : Prop :=
  hundreds ∈ range' 1 10

theorem three_digit_numbers_with_units_at_least_three_times_tens :
  let total_numbers := 189
  ∃ total_numbers : ℕ, total_numbers = ∑ h in range' 1 10, ∑ t in range 10, ∑ u in range 10, if valid_hundreds_digit h ∧ valid_units_and_tens_digit t u then 1 else 0 :=
begin
  sorry
end

end three_digit_numbers_with_units_at_least_three_times_tens_l389_389398


namespace shaded_quadrilateral_area_correct_l389_389151

noncomputable def area_of_shaded_quadrilateral (s1 s2 s3 : ℝ) (h_squares_aligned : s1 + s2 + s3 = 15) (height_ratio : (7 : ℝ) / 15 = 7 / (s1 + s2 + s3)) : ℝ :=
  let h1 := s1 * (7 / (s1 + s2 + s3)) in
  let h2 := (s1 + s2) * (7 / (s1 + s2 + s3)) in
  let b1 := h1 in
  let b2 := h2 in
  let height := s2 in
  1/2 * (b1 + b2) * height

theorem shaded_quadrilateral_area_correct :
  area_of_shaded_quadrilateral 3 5 7 (by norm_num [add]) (by norm_num [div_eq_div_iff]) = 12.825 :=
by sorry

end shaded_quadrilateral_area_correct_l389_389151


namespace divisible_by_sum_of_digits_in_18_consecutive_numbers_l389_389880

theorem divisible_by_sum_of_digits_in_18_consecutive_numbers :
  ∀ n : ℕ, (100 ≤ n ∧ n + 17 ≤ 999) →
  ∃ k, (k ≥ n ∧ k ≤ n + 17) ∧ (k % (nat.digits 10 k).sum = 0) :=
begin
  intros n h_n,
  sorry
end

end divisible_by_sum_of_digits_in_18_consecutive_numbers_l389_389880


namespace sum_of_mixed_numbers_in_interval_l389_389260

theorem sum_of_mixed_numbers_in_interval :
  let a := 4 + (5 / 9)
  let b := 5 + (3 / 4)
  let c := 7 + (8 / 17)
  (17.5 < a + b + c) ∧ (a + b + c < 18) :=
by
  let a := 4 + (5 / 9)
  let b := 5 + (3 / 4)
  let c := 7 + (8 / 17)
  have h : 17.5 < a + b + c ∧ a + b + c < 18 := sorry
  exact h

end sum_of_mixed_numbers_in_interval_l389_389260


namespace min_alpha_gamma_value_l389_389191

noncomputable 
def f (z : ℂ) (α γ : ℂ) : ℂ := (3 - 2 * complex.I) * z^2 + α * z + γ

theorem min_alpha_gamma_value {α γ : ℂ} 
  (h1 : f 1 α γ ∈ ℝ) 
  (h2 : f complex.I α γ ∈ ℝ) : 
  |α| + |γ| = 2 * real.sqrt 2 :=
sorry

end min_alpha_gamma_value_l389_389191


namespace value_of_r_when_n_is_one_l389_389847

def s (n : ℕ) : ℕ := 4^n + 2

def r (s : ℕ) : ℕ := 2 * 3^s + s

theorem value_of_r_when_n_is_one : r (s 1) = 1464 :=
by
  -- substitute n = 1 into s(n) ≡ 4^n + 2, resulting in s(1) = 6
  have : s 1 = 6 := by norm_num
  
  -- substitute s = 6 into r(s) ≡ 2 * 3^s + s, so r(6) = 2 * 3^6 + 6
  have : r (s 1) = r 6 := by rw this
  
  -- verify r(6) = 1464
  calc r 6 = 2 * 3^6 + 6 : by rfl
      ... = 2 * 729 + 6 : by norm_num
      ... = 1458 + 6 : by norm_num
      ... = 1464 : by norm_num

end value_of_r_when_n_is_one_l389_389847


namespace complex_exponentiation_l389_389670

noncomputable def cos30 : ℂ := complex.cos (real.pi / 6)
noncomputable def sin30 : ℂ := complex.sin (real.pi / 6)
noncomputable def cos240 : ℂ := complex.cos (4 * real.pi / 3)
noncomputable def sin240 : ℂ := complex.sin (4 * real.pi / 3)

theorem complex_exponentiation :
  (3 * (cos30 + complex.I * sin30))^8 = -3281 - 3281 * complex.I * real.sqrt 3 :=
by sorry

end complex_exponentiation_l389_389670


namespace height_at_2_years_l389_389638

variable (height : ℕ → ℕ) -- height function representing the height of the tree at the end of n years
variable (triples_height : ∀ n, height (n + 1) = 3 * height n) -- tree triples its height every year
variable (height_4 : height 4 = 81) -- height at the end of 4 years is 81 feet

-- We need the height at the end of 2 years
theorem height_at_2_years : height 2 = 9 :=
by {
  sorry
}

end height_at_2_years_l389_389638


namespace part1_part2_l389_389737

theorem part1 (n : ℕ) (a : Real) (a_i : Fin n → Real): 
  (Σ i, a_i i)^2 ≤ n * (Σ i, (a_i i)^2) :=
sorry

theorem part2 (x : Real) (h : 3 / 2 ≤ x ∧ x ≤ 5) : 
  2 * Real.sqrt (x + 1) + Real.sqrt (2 * x - 3) + Real.sqrt (15 - 3 * x) < 2 * Real.sqrt 19 :=
sorry

end part1_part2_l389_389737


namespace general_formula_sum_bn_l389_389723

variable {a : ℕ → ℕ} {b : ℕ → ℝ} 

axiom h_arith_seq : ∀ n, a (n + 1) = a 1 + n * d
axiom d_pos : d > 0
axiom h_sum : a 2 + a 4 = 8
axiom h_geom_seq : (a 3, a 5, a 8).1 = (a 1 + 2 * d)
                 ∧ (a_3, a_5, a_8).2 = (a 1 + 4 * d)
                 ∧ (a 5)² = (a 1 + 2 * d) * (a 1 + 7 * d)

theorem general_formula :
  a n = n + 1 := by
  sorry

noncomputable def b (n : ℕ) : ℝ := (1 : ℝ) / (a (2 * n - 1) * a (2 * n + 1)) + n

theorem sum_bn (n : ℕ) :
  ∑ i in range n, b i = (n : ℝ) / (4 * (n + 1)) + (n * (n + 1) / 2) := by
  sorry

end general_formula_sum_bn_l389_389723


namespace S_p_plus_q_gt_4_l389_389910

-- Define the arithmetic sequence sum
def S (n : ℕ) (a d : ℝ) : ℝ := n * a + (n * (n - 1) / 2) * d

-- Define the conditions as given in the problem
variables {p q : ℕ} (a d : ℝ)
hypothesis h1 : S p a d = p / q
hypothesis h2 : S q a d = q / p
hypothesis h3 : p ≠ q

-- Define the result we want to prove
theorem S_p_plus_q_gt_4 : S (p + q) (a : ℝ) (d : ℝ) ≥ 4 :=
by
  sorry

end S_p_plus_q_gt_4_l389_389910


namespace find_length_AB_and_cos_A_minus_pi_over_6_l389_389455

open Real

open_locale real

theorem find_length_AB_and_cos_A_minus_pi_over_6 (AC : ℝ) (B C : ℝ)
  (h1: AC = 6) (h2: cos B = 4/5) (h3: C = π/4) :
  ∃ (AB : ℝ) (cos_A_minus_pi_over_6 : ℝ), AB = 5 * sqrt 2 ∧ cos_A_minus_pi_over_6 = (7 * sqrt 2 - sqrt 6) / 20 :=
by
  sorry

end find_length_AB_and_cos_A_minus_pi_over_6_l389_389455


namespace range_of_m_l389_389339

theorem range_of_m (m : ℝ) (h1 : m > 0) (C : {p : ℝ × ℝ | (p.1^2 + p.2^2 - 2 * p.1 + 2 * m * p.2 + m^2 - 7 = 0)}) 
  (P : (2,0) ∈ C) (max_area_ABC : ∃ A B : ℝ × ℝ, (line_through (2,0) A ∧ line_through (2,0) B) ∧ 
  A ∈ C ∧ B ∈ C ∧ max_area_ABC (2,0) A B = 4) : sqrt 3 ≤ m ∧ m < sqrt 7 :=
sorry

end range_of_m_l389_389339


namespace choose_three_of_1400_l389_389177

open Real

theorem choose_three_of_1400 
  (a : ℕ → ℝ) (h : ∀ n, n < 1400 → ∃ x y z, 
  x = a n ∧ y = a (n+1) ∧ z = a (n+2) ∧ 
  ∣(x - y) * (y - z) * (z - x) / (x^4 + y^4 + z^4 + 1)∣ < 0.009) :
  ∃ x y z, ∃ i j k, i < 1400 ∧ j < 1400 ∧ k < 1400 ∧ x = a i ∧ y = a j ∧ z = a k ∧ 
  ∣(x - y) * (y - z) * (z - x) / (x^4 + y^4 + z^4 + 1)∣ < 0.009 := 
begin
  sorry
end

end choose_three_of_1400_l389_389177


namespace total_students_l389_389014

variables (T : ℕ)

-- Conditions from the problem statement
def students_below_8 := 0.20 * T
def students_age_8 := 48
def students_age_9_to_11 := 96
def students_above_11 := 80

theorem total_students (h1 : students_below_8 T = 0.20 * T)
                      (h2 : students_age_8 = 48)
                      (h3 : students_age_9_to_11 = 2 * students_age_8)
                      (h4 : students_above_11 = 5 / 6 * students_age_9_to_11):
  T = 280 :=
by
  sorry

end total_students_l389_389014


namespace graph_passes_through_point_l389_389559

theorem graph_passes_through_point (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) :
  ∃ p : ℝ × ℝ, p = (2, 0) ∧ ∀ x, (x = 2 → a ^ (x - 2) - 1 = 0) :=
by
  sorry

end graph_passes_through_point_l389_389559


namespace problem_statement_l389_389731

noncomputable def f (a b c x : ℝ) : ℝ := a * x^7 - b * x^5 + c * x^3 + 2

theorem problem_statement (a b c : ℝ) (h : f a b c (-5) = 3) : f a b c 5 + f a b c (-5) = 4 :=
by
  sorry

end problem_statement_l389_389731


namespace euler_totient_solution_l389_389290

theorem euler_totient_solution:
  ∀ (a n : ℕ), φ (a ^ n + n) = 2 ^ n ↔ (a = 2 ∧ n = 1) :=
sorry

end euler_totient_solution_l389_389290


namespace probability_of_drawing_3_numbered_ball_l389_389761

theorem probability_of_drawing_3_numbered_ball :
  let box_1 := [(1, 2), (2, 1), (3, 1)],
      box_2 := [(1, 2), (3, 1)],
      box_3 := [(1, 3), (2, 2)] in
  let p1 := 2/4 * 1/5,
      p2 := 1/4 * 1/4,
      p3 := 1/4 * 1/6 in
  box_1 = [(1, 2), (2, 1), (3, 1)] →
  box_2 = [(1, 2), (3, 1)] →
  box_3 = [(1, 3), (2, 2)] →
  (p1 + p2 + p3) = 11/48 :=
by
  let box_1 := [(1, 2), (2, 1), (3, 1)],
      box_2 := [(1, 2), (3, 1)],
      box_3 := [(1, 3), (2, 2)]
  let p1 := 2/4 * 1/5,
      p2 := 1/4 * 1/4,
      p3 := 1/4 * 1/6
  sorry 

end probability_of_drawing_3_numbered_ball_l389_389761


namespace cars_15th_time_l389_389574

noncomputable def minutes_since_8am (hour : ℕ) (minute : ℕ) : ℕ :=
  hour * 60 + minute

theorem cars_15th_time :
  let initial_time := minutes_since_8am 8 0
  let interval := 5
  let obstacles_time := 3 * 10
  let minutes_passed := (15 - 1) * interval + obstacles_time
  let total_time := initial_time + minutes_passed
  let expected_time := minutes_since_8am 9 40
  total_time = expected_time :=
by
  let initial_time := minutes_since_8am 8 0
  let interval := 5
  let obstacles_time := 3 * 10
  let minutes_passed := (15 - 1) * interval + obstacles_time
  let total_time := initial_time + minutes_passed
  let expected_time := minutes_since_8am 9 40
  show total_time = expected_time
  sorry

end cars_15th_time_l389_389574


namespace triangle_inequality_max_l389_389534

theorem triangle_inequality_max (a b c : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  a^2 + b^2 + c^2 > real.sqrt 3 * max (|a^2 - b^2|) (max (|b^2 - c^2|) (|c^2 - a^2|)) := 
sorry

end triangle_inequality_max_l389_389534


namespace a_minus_b_eq_minus_2_l389_389258

-- Define the function f is invertible
def f (x : ℝ) : ℝ := ((x - 3) * x * (x + 2)) / 7 + x + 3
def f_inv (y : ℝ) : ℝ := sorry  -- Inverse function should be defined based on invertibility
axiom invertible_f : ∀ x y : ℝ, f x = y ↔ f_inv y = x

-- Given conditions
axiom h1 : f a = b
axiom h2 : f b = 5
axiom h3 : f 3 = 5
axiom h4 : f 1 = 3

-- We need to prove a - b = -2
theorem a_minus_b_eq_minus_2 (a b : ℝ) : a - b = -2 := by
  sorry

end a_minus_b_eq_minus_2_l389_389258


namespace angles_equal_l389_389492

-- Axiom statements representing the conditions of the problem.
axiom parallelogram (A B C D : Point) : IsParallelogram A B C D
axiom points_on_segments (A B C D K L P : Point) : 
  OnSegment K B C ∧ OnSegment L C D ∧ (DK = line D K) ∧ (BL = line B L) ∧ (P = DK ∩ BL)

-- Given the condition
axiom condition (A B C D K L : Point) :
  BK * AD = DL * AB

-- Proof goal
theorem angles_equal (A B C D K L P : Point) 
  [parallelogram A B C D]
  [points_on_segments A B C D K L P]
  [condition A B C D K L] :
  angle DAP = angle BAC := sorry

end angles_equal_l389_389492


namespace angle_between_vectors_l389_389355

noncomputable def findAngle (a b: ℝ × ℝ) : ℝ :=
  let dot_product (u v: ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
  let norm (v: ℝ × ℝ) : ℝ := real.sqrt (dot_product v v)
  let theta := real.acos ((-1) / 2)
  if dot_product a (a + (2, 2) * b) = 0 ∧ norm a = 2 ∧ norm b = 2
    then theta
    else 0 -- placeholder, it should not be executed if conditions are met

theorem angle_between_vectors (a b: ℝ × ℝ):
  (a.1^2 + a.2^2 = 4) ∧ (b.1^2 + b.2^2 = 4) ∧
  (a.1 * (a.1 + 2 * b.1) + a.2 * (a.2 + 2 * b.2) = 0) →
  findAngle a b = real.pi * (2 / 3) := 
by
  intros
  sorry

end angle_between_vectors_l389_389355


namespace contains_zero_if_sum_is_111111_l389_389219

variables (x y : ℕ)
variables (digits_x digits_y : Fin 5 → ℕ)

-- Conditions
def is_five_digit (n : ℕ) : Prop := 
  10000 ≤ n ∧ n < 100000

def differs_by_two_digit_swap (dx dy : Fin 5 → ℕ) : Prop :=
  ∃ i j : Fin 5, i ≠ j ∧ 
  (∀ k : Fin 5, (k = i → dy k = dx j) ∧ 
                (k = j → dy k = dx i) ∧ 
                (k ≠ i ∧ k ≠ j → dy k = dx k))

theorem contains_zero_if_sum_is_111111 
  (hx : is_five_digit x) (hy : is_five_digit y)
  (h_digits_x : digits_x = fun i : Fin 5 => (x / 10^(4-i) % 10))
  (h_digits_y : digits_y = fun i : Fin 5 => (y / 10^(4-i) % 10))
  (h_swap : differs_by_two_digit_swap digits_x digits_y)
  (h_sum : x + y = 111111) :
  ∃ i : Fin 5, digits_x i = 0 :=
begin
  -- Proof omitted
  sorry,
end

end contains_zero_if_sum_is_111111_l389_389219


namespace jessy_jokes_l389_389810

variable (J : ℕ)
variable (A : ℕ := 7)
variable (T : ℕ := 54)

theorem jessy_jokes (h : 2 * J + 2 * A = T) : J = 20 :=
by simp at h; linarith

end jessy_jokes_l389_389810


namespace find_a_l389_389312

variable (a : ℝ)
def p : Prop := ∀ x ∈ set.Icc 1 2, x^2 - a ≥ 0
def q : Prop := ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0

theorem find_a (h : p a ∨ q a) (hn : ¬ (p a ∧ q a)) : a > 1 ∨ -2 < a ∧ a < 1 := by
  sorry

end find_a_l389_389312


namespace find_min_value_expression_l389_389294

noncomputable def minValueExpression (θ : ℝ) : ℝ :=
  2 * Real.cos θ + 1 / Real.sin θ + Real.sqrt 2 * Real.tan θ

theorem find_min_value_expression :
  ∃ θ : ℝ, 0 < θ ∧ θ < Real.pi / 2 ∧ minValueExpression θ = 3 * Real.sqrt 2 :=
sorry

end find_min_value_expression_l389_389294


namespace contains_zero_l389_389245

-- Define a five-digit number in terms of its digits.
def five_digit_number (A B C D E : ℕ) : ℕ :=
  10^4 * A + 10^3 * B + 10^2 * C + 10 * D + E

-- Define the conditions:
noncomputable def switch_two_digits (A B C D E F : ℕ) : Prop :=
  (10^4 * A + 10^3 * B + 10^2 * C + 10 * D + E ≠ 10^4 * A + 10^3 * B + 10^2 * F + 10 * D + E)

-- Lean statement for the proof problem.
theorem contains_zero (A B C D E F : ℕ) :
  let ABCDE := five_digit_number A B C D E,
      ABFDE := five_digit_number A B F D E in
  switch_two_digits A B C D E F ∧ (ABCDE + ABFDE = 111111) → (A = 0 ∨ B = 0 ∨ C = 0 ∨ D = 0 ∨ E = 0) :=
by
  -- This is where the proof would go.
  sorry

end contains_zero_l389_389245


namespace contains_zero_l389_389250

-- Define a five-digit number in terms of its digits.
def five_digit_number (A B C D E : ℕ) : ℕ :=
  10^4 * A + 10^3 * B + 10^2 * C + 10 * D + E

-- Define the conditions:
noncomputable def switch_two_digits (A B C D E F : ℕ) : Prop :=
  (10^4 * A + 10^3 * B + 10^2 * C + 10 * D + E ≠ 10^4 * A + 10^3 * B + 10^2 * F + 10 * D + E)

-- Lean statement for the proof problem.
theorem contains_zero (A B C D E F : ℕ) :
  let ABCDE := five_digit_number A B C D E,
      ABFDE := five_digit_number A B F D E in
  switch_two_digits A B C D E F ∧ (ABCDE + ABFDE = 111111) → (A = 0 ∨ B = 0 ∨ C = 0 ∨ D = 0 ∨ E = 0) :=
by
  -- This is where the proof would go.
  sorry

end contains_zero_l389_389250


namespace smallest_solution_x4_50x2_576_eq_0_l389_389589

theorem smallest_solution_x4_50x2_576_eq_0 :
  ∃ x : ℝ, (x^4 - 50*x^2 + 576 = 0) ∧ ∀ y : ℝ, y^4 - 50*y^2 + 576 = 0 → x ≤ y :=
sorry

end smallest_solution_x4_50x2_576_eq_0_l389_389589


namespace value_of_m_if_f_is_power_function_l389_389006

theorem value_of_m_if_f_is_power_function (m : ℤ) :
  (2 * m + 3 = 1) → m = -1 :=
by
  sorry

end value_of_m_if_f_is_power_function_l389_389006


namespace February1IsThursday_l389_389770

theorem February1IsThursday (day_of_week : ℕ → string)
    (H : day_of_week 1 = "Monday") : day_of_week 32 = "Thursday" := 
sorry

end February1IsThursday_l389_389770


namespace option_c_evaluates_to_9_l389_389995

theorem option_c_evaluates_to_9 : 3 * 3 - 3 + 3 = 9 := 
by 
soriyorum 

end option_c_evaluates_to_9_l389_389995


namespace intersection_volume_l389_389274

-- Definitions for edge length, points, and pyramids
def edge_length : ℝ := 6

def point_A : ℝ × ℝ × ℝ := (0, edge_length, edge_length)
def point_B : ℝ × ℝ × ℝ := (edge_length, edge_length, edge_length)
def point_C : ℝ × ℝ × ℝ := (edge_length, 0, edge_length)
def point_D : ℝ × ℝ × ℝ := (0, 0, edge_length)
def point_E : ℝ × ℝ × ℝ := (0, edge_length, 0)
def point_F : ℝ × ℝ × ℝ := (edge_length, edge_length, 0)
def point_G : ℝ × ℝ × ℝ := (edge_length, 0, 0)
def point_H : ℝ × ℝ × ℝ := (0, 0, 0)
def point_P : ℝ × ℝ × ℝ := (edge_length, edge_length / 2, edge_length / 2)

-- Volume of pyramids intersected region
def volume_of_intersection : ℝ := 4

theorem intersection_volume :
  let pyramid1 := (point_E, point_F, point_G, point_H, point_P)
  let pyramid2 := (point_A, point_B, point_C, point_D, point_G)
  volume_of_intersection = 4 :=
sorry

end intersection_volume_l389_389274


namespace points_on_line_relation_l389_389351

theorem points_on_line_relation :
  (∀ x, y = -3 * x + 5 → 
    (∃ y_1, (x = -6 → y = y_1)) ∧ 
    (∃ y_2, (x = 3 → y = y_2)) → 
    y_1 > y_2) :=
by
  assume h
  sorry

end points_on_line_relation_l389_389351


namespace find_b_value_l389_389454

theorem find_b_value (a b c A B C : ℝ) 
  (h1 : a = 1)
  (h2 : B = 120 * (π / 180))
  (h3 : c = b * Real.cos C + c * Real.cos B)
  (h4 : c = 1) : 
  b = Real.sqrt 3 :=
by
  sorry

end find_b_value_l389_389454


namespace count_different_numerators_l389_389833

def hasRepeatingDecimalForm (r : ℚ) : Prop :=
  ∃ a b : ℕ, 0 ≤ a ∧ a < 10 ∧ 0 ≤ b ∧ b < 10 ∧
    r = a * (1 / 100 + 1 / 100^2 + 1 / 100^3 + ...) + b * (1 / 10 + 1 / 10^3 + 1 / 10^5 + ...)

def setT : set ℚ := {r | r > 0 ∧ r < 1 ∧ hasRepeatingDecimalForm r}

theorem count_different_numerators : ∃ n : ℕ, n = 60 ∧
  ∀ r ∈ setT, r = (numerator r) / (99 * gcd (numerator r) 99) :=
sorry

end count_different_numerators_l389_389833


namespace lowest_number_in_range_l389_389873

theorem lowest_number_in_range (y : ℕ) (h : ∀ x y : ℕ, 0 < x ∧ x < y) : ∃ x : ℕ, x = 999 :=
by
  existsi 999
  sorry

end lowest_number_in_range_l389_389873


namespace alpha_quadrant_fourth_alpha_smallest_positive_angle_l389_389744

/-- Define a point as it passes through the terminal side --/
def point (α : ℝ) : ℝ × ℝ := (Real.sin (5 * Real.pi / 6), Real.cos (5 * Real.pi / 6))

/-- First Question: Prove α lies in the fourth quadrant --/
theorem alpha_quadrant_fourth (α : ℝ) (h : point α = (1/2, - Real.sqrt 3 / 2)) : 
  α ∈ Set.Ioo (-3 * Real.pi / 2) (- Real.pi / 2) := 
sorry

/-- Second Question: Prove the smallest positive angle with the same terminal side as 
    α is 5π/3 --/
theorem alpha_smallest_positive_angle (α : ℝ) (h : point α = (1/2, - Real.sqrt 3 / 2)) :
  ∃ k : ℤ, α = - Real.pi / 3 + 2 * k * Real.pi ∧ α = 5 * Real.pi / 3 :=
sorry

end alpha_quadrant_fourth_alpha_smallest_positive_angle_l389_389744


namespace z_purely_imaginary_z_div_1_add_i_l389_389357

-- Definitions
def z (m : ℝ) : ℂ := m * (m - 1) + (m - 1 : ℂ) * complex.I

-- Proof that z is purely imaginary for m = 0
theorem z_purely_imaginary (m : ℝ) (hm : m = 0) : ∃ (c : ℝ), z m = c * complex.I :=
by {
  sorry
}

-- Calculation of z / (1 + i) for m = 2
theorem z_div_1_add_i (hm2 : z 2 = 2 + complex.I) : (2 + complex.I) / (1 + complex.I) = 3 / 2 - (1 / 2) * complex.I :=
by {
  sorry
}

end z_purely_imaginary_z_div_1_add_i_l389_389357


namespace largest_m_equal_nonzero_digits_l389_389129

theorem largest_m_equal_nonzero_digits (n : ℕ) :
  ∃ m : ℕ, ∀ k : ℕ, k > 0 → k <= m → ∃ b : ℕ, n % (10^k) = b ∧ 
  (b.to_digits.count (b % 10) = k) :=
begin
  sorry
end

end largest_m_equal_nonzero_digits_l389_389129


namespace three_digit_number_units_digit_condition_l389_389385

theorem three_digit_number_units_digit_condition :
  let valid_numbers := (λ (n : ℕ), (100 ≤ n ∧ n ≤ 999) ∧ ((n % 10) ≥ 3 * ((n / 10) % 10))) in
  (Finset.filter valid_numbers (Finset.range 900)) = 198 :=
by
  sorry

end three_digit_number_units_digit_condition_l389_389385


namespace smallest_positive_four_digit_div_by_11_with_even_odd_digits_l389_389587

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

def count_even_and_odd_digits (n : ℕ) : Prop :=
  let digits := [n / 1000, (n / 100) % 10, (n / 10) % 10, n % 10] in
  (digits.filter (λ d => d % 2 = 0)).length = 2 ∧
  (digits.filter (λ d => d % 2 = 1)).length = 2

def smallest_valid_number : ℕ := 1065

theorem smallest_positive_four_digit_div_by_11_with_even_odd_digits :
  is_four_digit smallest_valid_number ∧
  is_divisible_by_11 smallest_valid_number ∧
  count_even_and_odd_digits smallest_valid_number :=
by
  sorry

end smallest_positive_four_digit_div_by_11_with_even_odd_digits_l389_389587


namespace rectangle_diagonal_eq_distances_l389_389796

theorem rectangle_diagonal_eq_distances 
  (a b x y : ℝ) : 
  let A := (0, 0)
      B := (a, 0)
      C := (a, b)
      D := (0, b)
      E := (x, y)
  in (dist A E)^2 + (dist C E)^2 = (dist B E)^2 + (dist D E)^2 := 
by sorry

end rectangle_diagonal_eq_distances_l389_389796


namespace sum_of_distinct_digits_l389_389334

theorem sum_of_distinct_digits
  (w x y z : ℕ)
  (h1 : y + w = 10)
  (h2 : x + y = 9)
  (h3 : w + z = 10)
  (h4 : w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z)
  (hw : w < 10) (hx : x < 10) (hy : y < 10) (hz : z < 10) :
  w + x + y + z = 20 := sorry

end sum_of_distinct_digits_l389_389334


namespace count_valid_numbers_l389_389401

def is_valid_number (n : Nat) : Prop := 
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let units := n % 10
  1 ≤ hundreds ∧ hundreds ≤ 9 ∧
  0 ≤ tens ∧ tens ≤ 9 ∧
  units ≥ 3 * tens

theorem count_valid_numbers : 
  (Finset.filter is_valid_number (Finset.range 1000)).card = 198 :=
by sorry

end count_valid_numbers_l389_389401


namespace condition_1_condition_2_condition_3_l389_389373

-- Define the sets A and B in terms of a condition on x.
def A : Set ℝ := {x | x^2 + 2 * x - 3 = 0}
def B (a : ℝ) : Set ℝ := if a = 0 then ∅ else {-1 / (2 * a)}

-- Condition ①: A ∩ B = B
theorem condition_1 (a : ℝ) : (A ∩ B a) = B a ↔ a ∈ {0, -1 / 2, 1 / 6} :=
sorry

-- Condition ②: (complement_R B) ∩ A = {1}
theorem condition_2 (a : ℝ) : (B aᶜ ∩ A) = {1} ↔ a = 1 / 6 :=
sorry

-- Condition ③: A ∩ B = ∅
theorem condition_3 (a : ℝ) : (A ∩ B a) = ∅ ↔ a ≠ 1 / 6 ∧ a ≠ -1 / 2 :=
sorry

end condition_1_condition_2_condition_3_l389_389373


namespace number_of_meetings_excluding_start_finish_l389_389578

-- Define the conditions
def forward_speed : ℕ := 6 -- ft/s
def backward_speed : ℕ := 3 -- ft/s
def circumference : ℕ := 270 -- ft

-- Define the effective speed and the relative speed
def effective_speed (forward backward : ℕ) : ℕ := forward + backward

def time_for_meeting (circumference effective_speed : ℕ) : ℕ :=
  circumference / effective_speed

-- Prove that they meet 0 times excluding start and finish
theorem number_of_meetings_excluding_start_finish :
  ∀ (forward_speed backward_speed circumference : ℕ),
    effective_speed forward_speed backward_speed = 9 →
    time_for_meeting circumference 9 = 30 →
    (circumference / 9 - 1) = 0 :=
by intros
   calc
     (circumference / 9 - 1) = 0 := sorry

end number_of_meetings_excluding_start_finish_l389_389578


namespace meiosis_fertilization_correct_l389_389166

theorem meiosis_fertilization_correct :
  (∀ (half_nuclear_sperm half_nuclear_egg mitochondrial_egg : Prop)
     (recognition_basis_clycoproteins : Prop)
     (fusion_basis_nuclei : Prop)
     (meiosis_eukaryotes : Prop)
     (random_fertilization : Prop),
    (half_nuclear_sperm ∧ half_nuclear_egg ∧ mitochondrial_egg ∧ recognition_basis_clycoproteins ∧ fusion_basis_nuclei ∧ meiosis_eukaryotes ∧ random_fertilization) →
    (D : Prop) ) := 
sorry

end meiosis_fertilization_correct_l389_389166


namespace reciprocal_of_repeating_decimal_l389_389967

noncomputable def repeating_decimal_to_fraction (x : ℚ) : ℚ :=
  if x = .\overline{36} then 4/11 else 0

theorem reciprocal_of_repeating_decimal :
  ∀ (x : ℚ), repeating_decimal_to_fraction (.\overline{36}) = 4/11 →
  1 / (repeating_decimal_to_fraction x) = 11/4 :=
by
  intros x hx
  have h : repeating_decimal_to_fraction x = 4/11 := hx
  rw h
  norm_num
  done
  sorry

end reciprocal_of_repeating_decimal_l389_389967


namespace train_speed_l389_389169

/-- Given:
  Length of the train: 320 meters
  Time to cross a telegraph post: 16 seconds

  Prove:
  The speed of the train is 20 meters per second.
-/
theorem train_speed (length_of_train : ℕ) (time_to_cross : ℕ) (h1 : length_of_train = 320) (h2 : time_to_cross = 16) : length_of_train / time_to_cross = 20 :=
by
  rw [h1, h2]
  sorry

end train_speed_l389_389169


namespace ratio_of_group_average_l389_389887

theorem ratio_of_group_average
  (d l e : ℕ)
  (avg_group_age : ℕ := 45) 
  (avg_doctors_age : ℕ := 40) 
  (avg_lawyers_age : ℕ := 55) 
  (avg_engineers_age : ℕ := 35)
  (h : (40 * d + 55 * l + 35 * e) / (d + l + e) = avg_group_age)
  : d = 2 * l - e ∧ l = 2 * e :=
sorry

end ratio_of_group_average_l389_389887


namespace B_alone_can_do_work_in_9_days_l389_389167

-- Define the conditions
def A_completes_work_in : ℕ := 15
def A_completes_portion_in (days : ℕ) : ℚ := days / 15
def portion_of_work_left (days : ℕ) : ℚ := 1 - A_completes_portion_in days
def B_completes_remaining_work_in_left_days (days_left : ℕ) : ℕ := 6
def B_completes_work_in (days_left : ℕ) : ℚ := B_completes_remaining_work_in_left_days days_left / (portion_of_work_left 5)

-- Define the theorem to be proven
theorem B_alone_can_do_work_in_9_days (days_left : ℕ) : B_completes_work_in days_left = 9 := by
  sorry

end B_alone_can_do_work_in_9_days_l389_389167


namespace claudia_can_fill_four_ounce_glasses_l389_389267

def claudia_problem : Prop :=
  ∀ (initial_water : ℕ) (num_five_ounce_glasses : ℕ) (num_eight_ounce_glasses : ℕ) (four_ounce_glass_capacity : ℕ) (remaining_water_per_four_ounce_glass : ℕ),
    initial_water = 122 →
    num_five_ounce_glasses = 6 →
    num_eight_ounce_glasses = 4 →
    four_ounce_glass_capacity = 4 →
    remaining_water_per_four_ounce_glass = 15 →
    let total_used_water := (num_five_ounce_glasses * 5) + (num_eight_ounce_glasses * 8) in
    let remaining_water := initial_water - total_used_water in
    remaining_water / four_ounce_glass_capacity = remaining_water_per_four_ounce_glass

theorem claudia_can_fill_four_ounce_glasses : claudia_problem := by
  intros initial_water num_five_ounce_glasses num_eight_ounce_glasses four_ounce_glass_capacity remaining_water_per_four_ounce_glass
  intros h_initial_water h_num_five_ounce_glasses h_num_eight_ounce_glasses h_four_ounce_glass_capacity h_remaining_water_per_four_ounce_glass
  have total_used_water := (num_five_ounce_glasses * 5) + (num_eight_ounce_glasses * 8)
  have remaining_water := initial_water - total_used_water
  calc
    remaining_water / four_ounce_glass_capacity = remaining_water_per_four_ounce_glass : by sorry

end claudia_can_fill_four_ounce_glasses_l389_389267


namespace solution_set_of_inequality_l389_389568

open Set

theorem solution_set_of_inequality :
  {x : ℝ | x^2 - x - 6 < 0} = Ioo (-2 : ℝ) 3 := 
sorry

end solution_set_of_inequality_l389_389568


namespace radius_squared_l389_389186

theorem radius_squared (r : ℝ) (AB_len CD_len BP_len : ℝ) (angle_APD : ℝ) (r_squared : ℝ) :
  AB_len = 10 →
  CD_len = 7 →
  BP_len = 8 →
  angle_APD = 60 →
  r_squared = r^2 →
  r_squared = 73 :=
by
  intros h₁ h₂ h₃ h₄ h₅
  sorry

end radius_squared_l389_389186


namespace three_digit_number_units_digit_condition_l389_389387

theorem three_digit_number_units_digit_condition :
  let valid_numbers := (λ (n : ℕ), (100 ≤ n ∧ n ≤ 999) ∧ ((n % 10) ≥ 3 * ((n / 10) % 10))) in
  (Finset.filter valid_numbers (Finset.range 900)) = 198 :=
by
  sorry

end three_digit_number_units_digit_condition_l389_389387


namespace arithmetic_geometric_sequence_l389_389722

theorem arithmetic_geometric_sequence
  (a : ℕ → ℝ)
  (b : ℕ → ℝ)
  (S : ℕ → ℝ)
  (f : ℕ → ℝ)
  (h₁ : a 1 = 3)
  (h₂ : b 1 = 1)
  (h₃ : b 2 * S 2 = 64)
  (h₄ : b 3 * S 3 = 960)
  : (∀ n, a n = 2 * n + 1) ∧ (∀ n, b n = 8^(n - 1)) ∧ 
    (∀ n, f n = (a n - 1) / (S n + 100)) ∧ 
    (∃ n, f n = 1 / 11 ∧ n = 10) := 
sorry

end arithmetic_geometric_sequence_l389_389722


namespace max_x_plus_y_l389_389630

/-- Define four numbers {a, b, c, d} in a set. -/
variables {a b c d x y : ℕ}

-- Pairwise sums of the set elements
variables (h1 : a + b = 172) (h2 : a + c = 305) (h3 : a + d = 271) 
variables (h4 : b + c = 229) (h5 : b + d = x) (h6 : c + d = y)

-- Given condition
variables (h7 : a + d = 299)

/-- Statement to prove the maximum possible value of x + y. -/
theorem max_x_plus_y : x + y = 835 :=
by
  sorry

end max_x_plus_y_l389_389630


namespace sphere_surface_area_l389_389349

-- Define the problem conditions
variables {R : ℝ}  -- Radius of the sphere

axiom h1 (H A B : ℝ) (O : Type) (α : Type) :
  ∃ H A B O α,
  (∃ (ratio : ℝ), ratio = 1/2) ∧
  (∃ (right_angle : Prop), right_angle) ∧
  (∃ (cross_section_area : ℝ), cross_section_area = 4 * Real.pi)

-- Define the sphere's surface area based on the given conditions
theorem sphere_surface_area (H A B O α : ℝ) :
  (∃ H A B O α,
    (AH_to_HB : ℝ) = 1 / 2 ∧                   -- AH:HB = 1:2
    AB_perpendicular_to_plane_at_H H A B O α ∧  -- AB is perpendicular to plane α at point H
    cross_section_area (4 * Real.pi))           -- The cross-section area is 4π
  → 
  ∃ S, S = 4 * Real.pi * (9 / 2) := sorry

end sphere_surface_area_l389_389349


namespace circle_through_fixed_point_l389_389333

/- Define the ellipse parameters -/
def a : ℝ := 2
def b : ℝ := sqrt 2
def e : ℝ := (sqrt 2) / 2

/- Define the vertices of the ellipse -/
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (2, 0)
def l : ℝ := -3 -- Line x = -3

/- Define the equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

/- Define the property to be proved about the circle with diameter PQ -/
theorem circle_through_fixed_point (M : ℝ × ℝ) (P Q : ℝ × ℝ)
    (hM : ellipse_equation M.1 M.2)
    (hAM : P = (-3, -M.2 / (M.1 + 2)))
    (hBM : Q = (-3, -5 * M.2 / (M.1 - 2))) :
    ∃ x0 : ℝ, ∃ y0 : ℝ, ((x0 = -3 - sqrt 10 / 2 ∨ x0 = -3 + sqrt 10 / 2) ∧ y0 = 0) ∧
    let center := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2) in
    let radius := sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) / 2 in
    ∀ (x y : ℝ), (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
sorry

end circle_through_fixed_point_l389_389333


namespace measure_angle5_is_60_l389_389073

-- Conditions
variable (m n k : Type) [ParallelLines m n] (angle1 angle2 angle3 angle4 angle5 : ℝ)
variable (h_parallel : m ∥ n)
variable (h_angle1_angle2 : angle1 = 1 / 2 * angle2)
variable (h_angle1_angle3 : angle1 + angle3 = 180)
variable (h_angle2_angle4 : angle2 + angle4 = 180)

-- Prove the measure of angle5 is 60 degrees
theorem measure_angle5_is_60 :
  angle5 = 60 :=
by
  sorry

end measure_angle5_is_60_l389_389073


namespace aborigines_can_satisfy_order_minimum_colors_is_clique_number_l389_389143

/-- Part a: Prove that for any graph G with vertices representing aborigines and edges representing friendships, 
there exists a color assignment such that:
1. For every pair of friends, there is at least one shared color.
2. For every pair of enemies, there is no shared color.
-/
theorem aborigines_can_satisfy_order {G : Type*} [graph G] {n : ℕ} : 
  ∃ (color_assignment : G → ℕ → Prop), 
  (∀ {u v : G}, G.edge u v → ∃ c, color_assignment u c ∧ color_assignment v c) ∧ 
  (∀ {u v : G}, ¬ G.edge u v → ∀ c, ¬ (color_assignment u c ∧ color_assignment v c)) :=
sorry

/-- Part b: Prove that the minimum number of colors required to satisfy the chieftain’s order for any graph G 
is the clique number ω(G).
-/
theorem minimum_colors_is_clique_number {G : Type*} [graph G] (ω : ℕ) : 
  ∃ (color_assignment : G → ℕ → Prop), 
  (∀ {u v : G}, G.edge u v → ∃ c, color_assignment u c ∧ color_assignment v c) ∧ 
  (∀ {u v : G}, ¬ G.edge u v → ∀ c, ¬ (color_assignment u c ∧ color_assignment v c)) → 
  (∀ (v : G), ∃ c, color_assignment v c) ↔ ω = clique_number G :=
sorry

end aborigines_can_satisfy_order_minimum_colors_is_clique_number_l389_389143


namespace gcd_of_987654_and_123456_l389_389158

theorem gcd_of_987654_and_123456 : Nat.gcd 987654 123456 = 6 := by
  sorry

end gcd_of_987654_and_123456_l389_389158


namespace example_problem_l389_389269

theorem example_problem : 
  ⌊(1007^3 / (1005 * 1006) - 1005^3 / (1006 * 1007) + 5)⌋ = 12 := 
by 
sorry

end example_problem_l389_389269


namespace three_digit_numbers_with_units_at_least_three_times_tens_l389_389395

open Nat

noncomputable def valid_units_and_tens_digit (tens units : ℕ) : Prop :=
  (tens = 0 ∧ units ∈ range 10) ∨
  (tens = 1 ∧ units ∈ range' 3 10) ∨
  (tens = 2 ∧ units ∈ range' 6 10)

noncomputable def valid_hundreds_digit (hundreds : ℕ) : Prop :=
  hundreds ∈ range' 1 10

theorem three_digit_numbers_with_units_at_least_three_times_tens :
  let total_numbers := 189
  ∃ total_numbers : ℕ, total_numbers = ∑ h in range' 1 10, ∑ t in range 10, ∑ u in range 10, if valid_hundreds_digit h ∧ valid_units_and_tens_digit t u then 1 else 0 :=
begin
  sorry
end

end three_digit_numbers_with_units_at_least_three_times_tens_l389_389395


namespace trainers_average_age_l389_389888

theorem trainers_average_age (total_members : ℕ) (total_average_age : ℕ) (num_women : ℕ) 
  (women_average_age : ℕ) (num_men : ℕ) (men_average_age : ℕ) (num_trainers : ℕ) 
  (total_sum_ages : ℕ) :-
  total_members = 70 -> 
  total_average_age = 23 -> 
  num_women = 30 -> 
  women_average_age = 20 -> 
  num_men = 25 -> 
  men_average_age = 25 -> 
  num_trainers = 15 -> 
  total_sum_ages = 1610 -> 
  (total_sum_ages - (num_women * women_average_age) - (num_men * men_average_age)) / num_trainers = 25 + 2/3 
:= 
sorry

end trainers_average_age_l389_389888


namespace total_apartments_in_building_l389_389087

theorem total_apartments_in_building (A k m n : ℕ)
  (cond1 : 5 = A)
  (cond2 : 636 = (m-1) * k + n)
  (cond3 : 242 = (A-m) * k + n) :
  A * k = 985 :=
by
  sorry

end total_apartments_in_building_l389_389087


namespace min_distance_S_origin_l389_389054

noncomputable def z : ℂ := sorry
noncomputable def P := z
noncomputable def Q := (2 + complex.i) * z
noncomputable def R := 3 * complex.conj z
noncomputable def S := Q + R - P

-- impose the condition that |z| = 1
axiom z_abs_eq_one : complex.abs z = 1

-- S is calculated based on the parallelogram PQRS
theorem min_distance_S_origin : complex.abs S = real.sqrt 10 := 
  sorry

end min_distance_S_origin_l389_389054


namespace solution_interval_l389_389283

theorem solution_interval (x : ℝ) (h1 : x / 2 ≤ 5 - x) (h2 : 5 - x < -3 * (2 + x)) :
  x < -11 / 2 := 
sorry

end solution_interval_l389_389283


namespace find_f_of_neg_1_l389_389704

-- Define the conditions
variables (a b c : ℝ)
variables (g f : ℝ → ℝ)
axiom g_definition : ∀ x, g x = x^3 + a*x^2 + 2*x + 15
axiom f_definition : ∀ x, f x = x^4 + x^3 + b*x^2 + 150*x + c

axiom g_has_distinct_roots : ∃ r1 r2 r3 : ℝ, r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3 ∧ ∀ x, g x = 0 ↔ (x = r1 ∨ x = r2 ∨ x = r3)
axiom roots_of_g_are_roots_of_f : ∀ x, g x = 0 → f x = 0

-- Prove the value of f(-1) given the conditions
theorem find_f_of_neg_1 (a : ℝ) (b : ℝ) (c : ℝ) (g f : ℝ → ℝ)
  (h_g_def : ∀ x, g x = x^3 + a*x^2 + 2*x + 15)
  (h_f_def : ∀ x, f x = x^4 + x^3 + b*x^2 + 150*x + c)
  (h_g_has_distinct_roots : ∃ r1 r2 r3 : ℝ, r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3 ∧ ∀ x, g x = 0 ↔ (x = r1 ∨ x = r2 ∨ x = r3))
  (h_roots : ∀ x, g x = 0 → f x = 0) :
  f (-1) = 3733.25 := 
by {
  sorry
}

end find_f_of_neg_1_l389_389704


namespace min_losses_max_loses_l389_389605

theorem min_losses_max_loses (n : ℕ) (h_n : n = 12)
  (h_players : ∀ (i j : ℕ), i ≠ j → ∃ (m: ℕ), m ≤ n - 1) :
  ∃ (LB : ℕ), LB ≥ ⌈((n - 1)/ 2: ℝ)⌉.to_nat := 
by
  have h_n_spec : n = 12 := rfl
  have h_matches : ∀ (i j : ℕ), i ≠ j → ∃ (m: ℕ), m ≤ n - 1 := λ _ _ _, sorry
  use(⌈((n - 1)/ 2: ℝ)⌉.to_nat)
  have h_lb : ⌈((n - 1)/ 2: ℝ)⌉.to_nat = 6 := sorry
  linarith

end min_losses_max_loses_l389_389605


namespace tangent_line_circle_l389_389002

theorem tangent_line_circle (a : ℝ) :
  (∃ t : ℝ, let x := t, y := t^2 / 4 in
    (x^2 + y^2 - 2 * x - 4 * y + a = 0) ∧
    (y = t^2 / 4) ∧
    (∃ m b : ℝ, y = m * x + b ∧ (forall z : ℝ, y = m * z + b → (z^2 / 4 = m*z + b)))
  ) → a = 3 := 
begin
  sorry
end

end tangent_line_circle_l389_389002


namespace sequence_equality_l389_389917

theorem sequence_equality (a : Fin 1973 → ℝ) (hpos : ∀ n, a n > 0)
  (heq : a 0 ^ a 0 = a 1 ^ a 2 ∧ a 1 ^ a 2 = a 2 ^ a 3 ∧ 
         a 2 ^ a 3 = a 3 ^ a 4 ∧ 
         -- etc., continued for all indices, 
         -- ensuring last index correctly refers back to a 0
         a 1971 ^ a 1972 = a 1972 ^ a 0) :
  a 0 = a 1972 :=
sorry

end sequence_equality_l389_389917


namespace functional_equation_solution_l389_389692

theorem functional_equation_solution {f : ℝ → ℝ}
  (h : ∀ x y : ℝ, f (x^2 + f x * f y) = x * f (x + y)) :
  (f = fun x => 0) ∨ (f = id) ∨ (f = fun x => -x) :=
sorry

end functional_equation_solution_l389_389692


namespace total_nuts_correct_l389_389487

-- Definitions for conditions
def w : ℝ := 0.25
def a : ℝ := 0.25
def p : ℝ := 0.15
def c : ℝ := 0.40

-- The theorem to be proven
theorem total_nuts_correct : w + a + p + c = 1.05 := by
  sorry

end total_nuts_correct_l389_389487


namespace one_proposition_false_l389_389708

variables (a b : ℝ)

def quadratic_eq (x a b : ℝ) := x^2 + a * x + b = 0

theorem one_proposition_false (hA : ¬ quadratic_eq 1 a b)
                              (hB : quadratic_eq 3 a b)
                              (hC : 2 = 4 * b - a^2)
                              (hD : ∃ r₁ r₂ : ℝ, r₁ * r₂ = b ∧ r₁ ≠ 0 ∧ r₂ ≠ 0 ∧ r₁ * (r₁ + a) = -b) :
                              ¬ quadratic_eq 3 a b ∧ hC ∧ hD :=
sorry

end one_proposition_false_l389_389708


namespace dice_cd_divisible_by_2_probability_l389_389163

theorem dice_cd_divisible_by_2_probability :
  let D := {n : ℕ | n ∈ {1, 2, 3, 4, 5, 6}} in
  let even := {n : ℕ | n ∈ {2, 4, 6}} in
  let prob_even d := if d ∈ even then 1 / 2 else 0 in
  ∑ c in D, ∑ d in D, (if c ∈ even ∧ d ∈ even then 1 else 0) / (|D| * |D|) = 1 / 4 :=
by sorry -- Proof goes here

end dice_cd_divisible_by_2_probability_l389_389163


namespace claudia_can_fill_four_ounce_glasses_l389_389268

def claudia_problem : Prop :=
  ∀ (initial_water : ℕ) (num_five_ounce_glasses : ℕ) (num_eight_ounce_glasses : ℕ) (four_ounce_glass_capacity : ℕ) (remaining_water_per_four_ounce_glass : ℕ),
    initial_water = 122 →
    num_five_ounce_glasses = 6 →
    num_eight_ounce_glasses = 4 →
    four_ounce_glass_capacity = 4 →
    remaining_water_per_four_ounce_glass = 15 →
    let total_used_water := (num_five_ounce_glasses * 5) + (num_eight_ounce_glasses * 8) in
    let remaining_water := initial_water - total_used_water in
    remaining_water / four_ounce_glass_capacity = remaining_water_per_four_ounce_glass

theorem claudia_can_fill_four_ounce_glasses : claudia_problem := by
  intros initial_water num_five_ounce_glasses num_eight_ounce_glasses four_ounce_glass_capacity remaining_water_per_four_ounce_glass
  intros h_initial_water h_num_five_ounce_glasses h_num_eight_ounce_glasses h_four_ounce_glass_capacity h_remaining_water_per_four_ounce_glass
  have total_used_water := (num_five_ounce_glasses * 5) + (num_eight_ounce_glasses * 8)
  have remaining_water := initial_water - total_used_water
  calc
    remaining_water / four_ounce_glass_capacity = remaining_water_per_four_ounce_glass : by sorry

end claudia_can_fill_four_ounce_glasses_l389_389268


namespace graphs_symmetric_y_axis_l389_389165

theorem graphs_symmetric_y_axis : ∀ (x : ℝ), (-x) ∈ { y | y = 3^(-x) } ↔ x ∈ { y | y = 3^x } :=
by
  intro x
  sorry

end graphs_symmetric_y_axis_l389_389165


namespace employee_reduction_l389_389631

noncomputable def percentage_reduction (O R : ℝ) : ℝ :=
  ((O - R) / O) * 100

theorem employee_reduction :
  let O := 227
  let R := 195
  percentage_reduction O R ≈ 14.1 :=
sorry

end employee_reduction_l389_389631


namespace side_length_a_l389_389801

theorem side_length_a (a b c : ℝ) (B : ℝ) (h1 : a = c - 2 * a * Real.cos B) (h2 : c = 5) (h3 : 3 * a = 2 * b) :
  a = 4 := by
  sorry

end side_length_a_l389_389801


namespace number_of_towers_l389_389634

theorem number_of_towers :
  let r := 2
  let b := 3
  let g := 4
  let n := 8
  (fact n / (fact 1 * fact 3 * fact 4)) + (fact n / (fact 2 * fact 2 * fact 4)) + (fact n / (fact 2 * fact 3 * fact 3)) = 1260 :=
by
  let r := 2
  let b := 3
  let g := 4
  let n := 8
  have h1 : fact n / (fact 1 * fact 3 * fact 4) = 280 := by exact sorry
  have h2 : fact n / (fact 2 * fact 2 * fact 4) = 420 := by exact sorry
  have h3 : fact n / (fact 2 * fact 3 * fact 3) = 560 := by exact sorry
  rw [← add_assoc, ← add_assoc, h1, h2, h3]
  exact sorry

end number_of_towers_l389_389634


namespace constant_term_expansion_l389_389947

theorem constant_term_expansion : 
  (constant_term_of_expansion (5 * x + 2 / (5 * x)) 8) = 1120 := 
by sorry

end constant_term_expansion_l389_389947


namespace find_f_of_2_l389_389717

theorem find_f_of_2 
  (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, f (x - 1/x) = x^2 + 1/x^2) : f 2 = 6 :=
sorry

end find_f_of_2_l389_389717


namespace triangle_area_l389_389113

theorem triangle_area (A B C D : Point)
  (h_right_angle : right_angle A C D)
  (h_B_on_AC : B ∈ segment A C)
  (h_ABD : (distance A B) = 3 ∧ (distance B D) = 7 ∧ (distance A D) = 8) :
  area (triangle B C D) = 2 * sqrt 3 := 
sorry

end triangle_area_l389_389113


namespace hyperbola_distance_three_lines_l389_389306

theorem hyperbola_distance_three_lines :
  ∃ λ : ℝ,
    (∀ l : Set (ℝ × ℝ), l ∈ {l | l passes through (sqrt 3, 0) ∧ l intersects {p | p.1^2 - (p.2^2 / 2) = 1} at points A B ∧ |distance A B| = λ}) ∧
    (card {l : Set (ℝ × ℝ) | l passes through (sqrt 3, 0) ∧ l intersects {p : ℝ × ℝ | p.1^2 - (p.2^2 / 2) = 1} at points A B ∧ |distance A B| = λ} = 3) ∧
    λ = 4 :=
  sorry

end hyperbola_distance_three_lines_l389_389306


namespace parallel_chords_equal_and_CD_is_diameter_l389_389174

-- Define the circle and its geometric properties
variables {C : Type} [metric_space C] {O A B C D : C}
variable {r : ℝ}  -- radius

-- Assume points A and B, and diameter AB of circle with center O
def is_diameter_of_circle (O A B : C) (r : ℝ) : Prop :=
  dist O A = r ∧ dist O B = r ∧ dist A B = 2 * r

-- Assume points C and D and parallel chords AC and BD
def are_parallel_chords (A B C D : C) : Prop :=
  AC_parallel : (C - A) = (D - B) -- This represents the parallelism of vectors AC and BD

-- Define the theorem to prove
theorem parallel_chords_equal_and_CD_is_diameter
  (h_diameter : is_diameter_of_circle O A B r)
  (h_parallel : are_parallel_chords A B C D) :
  dist A C = dist B D ∧ AC_perpendicular_BO ∧ CD_perpendicular_DO ∧ line_proj O C D = line O :=
sorry

end parallel_chords_equal_and_CD_is_diameter_l389_389174


namespace find_integer_triples_l389_389693

theorem find_integer_triples (x y z : ℤ) : 
  x^3 + y^3 + z^3 - 3 * x * y * z = 2003 ↔ 
  (x = 668 ∧ y = 668 ∧ z = 667) ∨ 
  (x = 668 ∧ y = 667 ∧ z = 668) ∨ 
  (x = 667 ∧ y = 668 ∧ z = 668) :=
by sorry

end find_integer_triples_l389_389693


namespace exists_positive_integer_n_l389_389846

noncomputable def P : Polynomial ℤ := sorry

theorem exists_positive_integer_n (q : ℕ) (hq : 0 < q) :
  ∃ n : ℕ, 0 < n ∧ n = q^2 ∧ q ∣ ∑ i in Finset.range (n+1), P.eval i :=
begin
  sorry
end

end exists_positive_integer_n_l389_389846


namespace find_direction_vector_l389_389561

noncomputable def projection_matrix : Matrix (Fin 3) (Fin 3) ℚ :=
  ![![2/15, -1/15, -1/3],
    ![-1/15, 1/30, 1/6],
    ![-1/3, 1/6, 5/6]]

def valid_direction_vector (v : Vector ℤ 3) : Prop :=
  v.head > 0 ∧ (v.to_list.map Int.natAbs).gcd = 1

theorem find_direction_vector :
  ∃ (v : Vector ℤ 3), projection_matrix.mulVec ![1, 0, 0] = (1 / 15 : ℚ) • v ∧ valid_direction_vector v :=
begin
  let v := Vector.of [2, -1, -5],
  use v,
  simp only [Matrix.mulVec, Matrix.dotProduct, Pi.smul_apply, Matrix.smulVec, Matrix.cons_val', Matrix.head_cons, Matrix.fin_zero_eq_zero, Matrix.row_vec_lin_equiv_apply],
  split,
  { simp, },
  { split,
    { norm_num, },
    { simp [valid_direction_vector],
      norm_num, } },
end

end find_direction_vector_l389_389561


namespace smallest_positive_period_of_f_minimum_value_of_f_on_interval_l389_389754

noncomputable def f (x : ℝ) : ℝ :=
  (Real.sqrt 2) * (Real.sin (x / 2)) * (Real.cos (x / 2)) - (Real.sqrt 2) * (Real.sin (x / 2)) ^ 2

theorem smallest_positive_period_of_f : ∀ x : ℝ, f (x + 2 * Real.pi) = f x :=
by sorry

theorem minimum_value_of_f_on_interval : 
  ∃ x ∈ Set.Icc (-Real.pi) 0, 
  f x = -1 - Real.sqrt 2 / 2 :=
by sorry

end smallest_positive_period_of_f_minimum_value_of_f_on_interval_l389_389754


namespace james_earnings_l389_389807

def amount_per_inch := 15
def rain_monday := 4
def rain_tuesday := 3
def price_per_gallon := 1.2
def total_made := 126

theorem james_earnings :
  (rain_monday * amount_per_inch + rain_tuesday * amount_per_inch) * price_per_gallon = total_made :=
by
  sorry

end james_earnings_l389_389807


namespace james_jane_same_speed_l389_389808

noncomputable def james_speed (x : ℝ) : ℝ := x^2 - 13 * x - 30

noncomputable def jane_speed (x : ℝ) : ℝ := (x^2 - 5 * x - 66) / (x + 6)

theorem james_jane_same_speed (x : ℝ) (h : james_speed x = jane_speed x) : james_speed x = -4 + 2 * real.sqrt 17 :=
by
  -- Proof omitted
  sorry

end james_jane_same_speed_l389_389808


namespace speed_of_second_train_l389_389181

theorem speed_of_second_train (l1 l2 : ℝ) (s1 t : ℝ) 
  (cross_in : t > 0) (length1 : l1 = 150) (length2 : l2 = 350.04) 
  (speed1 : s1 = 120) : 
  (∃ s2 : ℝ, s2 = 79.83) :=
by
  -- The given length of the first train
  have h1 : l1 = 150 := length1,

  -- The given length of the second train
  have h2 : l2 = 350.04 := length2,

  -- The given speed of the first train in kmph
  have h3 : s1 = 120 := speed1,

  -- Convert speed of the first train from kmph to m/s
  let speed1_m_s := (s1 * 5 / 18),

  -- Calculate total length of both trains
  let total_length := l1 + l2,

  -- Calculate their relative speed
  let v_r := total_length / t,

  -- Calculate speed of the second train in m/s
  let speed2_m_s := v_r - speed1_m_s,

  -- Convert speed of the second train back to kmph
  let s2 := speed2_m_s * 18 / 5,

  -- The answer we're looking for
  existsi 79.83,
  sorry

end speed_of_second_train_l389_389181


namespace cole_drive_time_to_work_cole_drive_time_to_work_in_minutes_l389_389597

-- Definition of the conditions
def average_speed_to_work : ℝ := 50
def average_speed_home : ℝ := 110
def total_round_trip_time : ℝ := 2

-- Statement of the problem to be proven
theorem cole_drive_time_to_work :
  ∃ T_work : ℝ, (T_work = total_round_trip_time * (average_speed_home / (average_speed_to_work * average_speed_home / (average_speed_to_work + average_speed_home)))) / 2 :=
by
  sorry

-- Assertion that the time taken to drive to work in minutes matches 82.5 from the conditions.
theorem cole_drive_time_to_work_in_minutes :
  ∃ T_work_minutes : ℕ, T_work_minutes = 82.5 * 60 :=
by
  sorry

end cole_drive_time_to_work_cole_drive_time_to_work_in_minutes_l389_389597


namespace two_people_200_meters_apart_times_l389_389173

theorem two_people_200_meters_apart_times :
  ∀ (T : ℤ), (T = 10 ∨ T = 15) ↔ ∃ t, t = T ∧ (A_dist(t) - B_dist(t) = 200 ∨ B_dist(t) - A_dist(t) = 200) :=
by
  let A_speed := 120
  let B_speed := 200
  let A_depart := 0
  let B_depart := 5
  let A_dist (x: ℤ) := A_speed * x
  let B_dist (x: ℤ) := B_speed * (x - B_depart)
  sorry

end two_people_200_meters_apart_times_l389_389173


namespace Scruffy_weight_l389_389655

variable {Muffy Puffy Scruffy : ℝ}

def Puffy_weight_condition (Muffy Puffy : ℝ) : Prop := Puffy = Muffy + 5
def Scruffy_weight_condition (Muffy Scruffy : ℝ) : Prop := Scruffy = Muffy + 3
def Combined_weight_condition (Muffy Puffy : ℝ) : Prop := Muffy + Puffy = 23

theorem Scruffy_weight (h1 : Puffy_weight_condition Muffy Puffy) (h2 : Scruffy_weight_condition Muffy Scruffy) (h3 : Combined_weight_condition Muffy Puffy) : Scruffy = 12 := by
  sorry

end Scruffy_weight_l389_389655


namespace find_quadratic_function_l389_389326

theorem find_quadratic_function (f : ℝ → ℝ) 
  (h1 : f(-1) = 0)
  (h2 : ∀ x : ℝ, x ≤ f(x) ∧ f(x) ≤ (1 + x^2) / 2) :
  f = (λ x : ℝ, (1/4) * x^2 + (1/2) * x + (1/4)) :=
sorry

end find_quadratic_function_l389_389326


namespace pirate_treasure_l389_389523

theorem pirate_treasure (x : ℕ) (h1 : ∀ n, PeteCoins n = 2 * (n * (n + 1) / 2)) (h2 : PeteCoins x = 3 * PaulCoins x) : 
  PirateCoinsTotal =
  8 :=
by
  sorry

end pirate_treasure_l389_389523


namespace arithmetic_progression_only_digit_5_l389_389694

theorem arithmetic_progression_only_digit_5 (n k : ℕ) : 
a_n = 19n - 20 → 
(∃ k, 19n - 20 = 5 * ((10^k - 1) / 9)) :=
by
  sorry

end arithmetic_progression_only_digit_5_l389_389694


namespace largest_part_of_proportional_division_l389_389766

theorem largest_part_of_proportional_division :
  ∀ (x y z : ℝ),
    x + y + z = 120 ∧
    x / (1 / 2) = y / (1 / 4) ∧
    x / (1 / 2) = z / (1 / 6) →
    max x (max y z) = 60 :=
by sorry

end largest_part_of_proportional_division_l389_389766


namespace count_valid_numbers_l389_389405

def is_valid_number (n : Nat) : Prop := 
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let units := n % 10
  1 ≤ hundreds ∧ hundreds ≤ 9 ∧
  0 ≤ tens ∧ tens ≤ 9 ∧
  units ≥ 3 * tens

theorem count_valid_numbers : 
  (Finset.filter is_valid_number (Finset.range 1000)).card = 198 :=
by sorry

end count_valid_numbers_l389_389405


namespace part1_part2_l389_389473

-- Definitions related to geometry and the problem
variables {A B C I D E F G M : Type} 
variables [Incenter I A B C] [Excenter D A B C]
          [TangentPoint E (Circle I)] [TangentPoint F (Circle D)]
          [Intersection G (Line A D) (Segment B C)]
          [Midpoint M E F]

-- Proof of the first statement: (AI / AD) = (GE / GF)
theorem part1 (h1: Tangent E (Circle I) (Line B C)) 
              (h2: Tangent F (Circle D) (Line B C)) 
              (h3: Intersection G (Line A D) (Line B C)) :
  (AI / AD) = (GE / GF) := 
sorry

-- Proof of the second statement: AE || DM
theorem part2 (hM: Midpoint M E F) :
  AE || DM := 
sorry

end part1_part2_l389_389473


namespace max_expression_eq_min_value_l389_389762

variables (m n : ℕ) (r s : ℝ)
variables (a : ℕ → ℕ → ℝ)

def expression (m n : ℕ) (r s : ℝ) (a : ℕ → ℕ → ℝ) :=
  (∑ j in Finset.range n, (∑ i in Finset.range m, (a i j)^s)^(r/s))^(1/r) /
  (∑ i in Finset.range m, ∑ j in Finset.range n, (a i j)^r)^(1/s)

theorem max_expression_eq_min_value
  (hm : 1 < m) (hn : 1 < n) (hr : 0 < r) (hs : 0 < s) (hrs : r < s)
  (a_nonneg : ∀ i j, 0 ≤ a i j) (a_nonzero : ∃ i j, 0 < a i j) :
  expression m n r s a =
  min m n ^ (1/r - 1/s) :=
sorry

end max_expression_eq_min_value_l389_389762


namespace dihedral_angle_greater_than_60_l389_389527

structure RegularTriangularPyramid (A B C S : Type) :=
  (is_regular : True) -- Assuming regular triangular pyramid with A, B, C as the base and S as the apex
  (height_BH_in_ABS : ∃ H, is_height B H S ∧ is_in_face H ABS)
  (height_CH_in_ACS : ∃ H, is_height C H S ∧ is_in_face H ACS)
  (CH_perpendicular_SA : ∃ H, is_height C H S ∧ is_orthogonal CH SA)

theorem dihedral_angle_greater_than_60 (A B C S : Type)
  [RegularTriangularPyramid A B C S] :
  ∃ θ, dihedral_angle A B C S θ ∧ θ > 60 := by
  sorry

end dihedral_angle_greater_than_60_l389_389527


namespace sam_packs_books_l389_389094

-- Definitions based on given conditions
def books := 15
def novels := 4
def box1_capacity := 3
def box2_capacity := 4
def box3_capacity := 4
def box4_capacity := 2
def box5_capacity := 2

def total_ways := (Nat.C.choose 15 box1_capacity) * (Nat.C.choose (books - box1_capacity) box2_capacity) * 
                  (Nat.C.choose (books - box1_capacity - box2_capacity) box3_capacity) *
                  (Nat.C.choose (books - box1_capacity - box2_capacity - box3_capacity) box4_capacity) *
                  (Nat.C.choose (books - box1_capacity - box2_capacity - box3_capacity - box4_capacity) box5_capacity)

def favorable_ways := 2

def probability := favorable_ways // total_ways

-- The simplified probability fraction and the final answer
def final_fraction := Nat.gcdExt favorable_ways total_ways |>.1

def answer := final_fraction + (total_ways // final_fraction)

theorem sam_packs_books : answer = 46905751 := by
  sorry

end sam_packs_books_l389_389094


namespace xy_product_l389_389923

theorem xy_product (x y z : ℝ) (h : x^2 + y^2 = x * y * (z + 1 / z)) :
  x = y * z ∨ y = x * z := 
by
  sorry

end xy_product_l389_389923


namespace pond_field_area_ratio_l389_389123

theorem pond_field_area_ratio (w l s A_field A_pond : ℕ) (h1 : l = 2 * w) (h2 : l = 96) (h3 : s = 8) (h4 : A_field = l * w) (h5 : A_pond = s * s) :
  A_pond.toFloat / A_field.toFloat = 1 / 72 := 
by
  sorry

end pond_field_area_ratio_l389_389123


namespace three_digit_numbers_with_units_at_least_three_times_tens_l389_389397

open Nat

noncomputable def valid_units_and_tens_digit (tens units : ℕ) : Prop :=
  (tens = 0 ∧ units ∈ range 10) ∨
  (tens = 1 ∧ units ∈ range' 3 10) ∨
  (tens = 2 ∧ units ∈ range' 6 10)

noncomputable def valid_hundreds_digit (hundreds : ℕ) : Prop :=
  hundreds ∈ range' 1 10

theorem three_digit_numbers_with_units_at_least_three_times_tens :
  let total_numbers := 189
  ∃ total_numbers : ℕ, total_numbers = ∑ h in range' 1 10, ∑ t in range 10, ∑ u in range 10, if valid_hundreds_digit h ∧ valid_units_and_tens_digit t u then 1 else 0 :=
begin
  sorry
end

end three_digit_numbers_with_units_at_least_three_times_tens_l389_389397


namespace increasing_function_range_a_l389_389118

theorem increasing_function_range_a (a : ℝ) :
  (∀ x y : ℝ, 1 ≤ x → x ≤ y → f x ≤ f y) → a ≤ 1 :=
by
  let f := λ x : ℝ, x^2 - 2 * a * x
  sorry

end increasing_function_range_a_l389_389118


namespace third_smallest_palindromic_prime_proof_l389_389902

def is_palindromic_prime (n : ℕ) : Prop :=
  let s := n.digits 10 in
  s = s.reverse ∧ Prime n

def third_smallest_palindromic_prime : ℕ :=
  151

theorem third_smallest_palindromic_prime_proof :
  131 < third_smallest_palindromic_prime ∧
  is_palindromic_prime third_smallest_palindromic_prime ∧
  ∃ x1 x2 : ℕ, x1 < 131 ∧ is_palindromic_prime x1 ∧ 
               x2 < third_smallest_palindromic_prime ∧ is_palindromic_prime x2 ∧ x1 < x2 ∧ 
               (¬ ∃ x3 : ℕ, x2 < x3 ∧ x3 < third_smallest_palindromic_prime ∧ is_palindromic_prime x3)
:=
  sorry

end third_smallest_palindromic_prime_proof_l389_389902


namespace reciprocal_of_repeating_decimal_l389_389988

theorem reciprocal_of_repeating_decimal :
  let x := (36 : ℚ) / 99 in 1 / x = 11 / 4 :=
by
  trivial -- This is a placeholder, the actual proof is not required and hence replaced by trivial.

end reciprocal_of_repeating_decimal_l389_389988


namespace remainder_1501_1505_mod_13_l389_389991

theorem remainder_1501_1505_mod_13 :
  (1501 * 1502 * 1503 * 1504 * 1505) % 13 = 5 :=
by
  have h1 : 1501 % 13 = 2 := by norm_num
  have h2 : 1502 % 13 = 3 := by norm_num
  have h3 : 1503 % 13 = 4 := by norm_num
  have h4 : 1504 % 13 = 5 := by norm_num
  have h5 : 1505 % 13 = 6 := by norm_num
  sorry

end remainder_1501_1505_mod_13_l389_389991


namespace evaluate_expression_l389_389992

theorem evaluate_expression : 40 + 5 * 12 / (180 / 3) = 41 :=
by
  -- Proof goes here
  sorry

end evaluate_expression_l389_389992


namespace forty_ab_l389_389598

theorem forty_ab (a b : ℝ) (h₁ : 4 * a = 30) (h₂ : 5 * b = 30) : 40 * a * b = 1800 :=
by
  sorry

end forty_ab_l389_389598


namespace reciprocal_of_repeating_decimal_l389_389972

noncomputable def repeating_decimal_to_fraction (x : ℚ) : ℚ :=
  if x = .\overline{36} then 4/11 else 0

theorem reciprocal_of_repeating_decimal :
  ∀ (x : ℚ), repeating_decimal_to_fraction (.\overline{36}) = 4/11 →
  1 / (repeating_decimal_to_fraction x) = 11/4 :=
by
  intros x hx
  have h : repeating_decimal_to_fraction x = 4/11 := hx
  rw h
  norm_num
  done
  sorry

end reciprocal_of_repeating_decimal_l389_389972


namespace binomial_coefficient_equality_x_square_term_coefficient_find_value_l389_389738

theorem binomial_coefficient_equality (n : ℕ) (h : (Nat.choose n 3) = (Nat.choose n 5)) : n = 8 :=
by sorry

theorem x_square_term_coefficient (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℤ)
  (h : (3 + -1) = 2) : 
  (8.choose 6) * 9 = 252 :=
by sorry

theorem find_value (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℤ)
  (n : ℕ) (h : n = 8)
  (exp : (3*(1/2) - 1)^8 = ∑ i in Finset.range(n+1), (a_ i / 2^i)) :
  2^8*a_0 + 2^7*a_1 + 2^6*a_2 + 2^5*a_3 + 2^4*a_4 + 2^3*a_5 + 2^2*a_6 + 2^1*a_7 + 2^0*a_8 = 1 :=
by sorry

end binomial_coefficient_equality_x_square_term_coefficient_find_value_l389_389738


namespace three_digit_numbers_count_l389_389431

theorem three_digit_numbers_count :
  let t_range := finset.range 10
  let h_range := finset.range 9
  let valid_u t := if t * 3 ≤ 9 then 9 - (t * 3) + 1 else 0
  let total_count := h_range.card * t_range.sum (λ t, valid_u t)
  total_count = 198 :=
by
  sorry

end three_digit_numbers_count_l389_389431


namespace find_perpendicular_line_through_point_perpendicular_line_equation_l389_389554

theorem find_perpendicular_line_through_point
    (A : ℝ × ℝ) (a b c : ℝ) (x y m : ℝ)
    (hA : A = (2, 3))
    (hLine : 2 * x + y - 5 = 0)
    (hPerpendicular : x - 2 * y + m = 0) :
    m = 4 := 
by
  cases hA with xt yt
  subst xt yt
  let eq1 := 2 - 2 * 3 + m = 0
  have h_eq1 : eq1 = 2 - 6 + m = 0 := by
    rfl
  have h_m : m = 4 := by
    linarith
  exact h_m

-- Use a theorem to state the resultant line equation.
theorem perpendicular_line_equation (A : ℝ × ℝ) (a b c : ℝ)
  (hA : A = (2, 3)) (hLine : 2 * x + y - 5 = 0) :
  ∃ (x y m : ℝ), (m = 4) :=
exists.intro 2 (exists.intro 3 (exists.intro 4 (find_perpendicular_line_through_point (2, 3) 2 1 (-5) 1 2 4 rfl)))

end find_perpendicular_line_through_point_perpendicular_line_equation_l389_389554


namespace distance_from_A_to_BE_l389_389086

variables (A B C D E : Point)
variables (a b c : ℝ)

/-- Pentagon \(ABCDE\) is inscribed in a circle. -/
def pentagon_inscribed_in_circle (A B C D E : Point) : Prop := sorry

/-- Distance from point \(A\) to the line \(BC\) -/
def distance_from_A_to_BC (A B C : Point) : ℝ := a

/-- Distance from point \(A\) to the line \(CD\) -/
def distance_from_A_to_CD (A C D : Point) : ℝ := b

/-- Distance from point \(A\) to the line \(DE\) -/
def distance_from_A_to_DE (A D E : Point) : ℝ := c

/-- The distance from vertex \(A\) to the line \(BE\) is \(\frac{ac}{b}\) -/
theorem distance_from_A_to_BE (h1 : pentagon_inscribed_in_circle A B C D E)
  (h2 : distance_from_A_to_BC A B C = a)
  (h3 : distance_from_A_to_CD A C D = b)
  (h4 : distance_from_A_to_DE A D E = c) :
  distance_from_A_to_BE A B E = (a * c) / b :=
sorry

end distance_from_A_to_BE_l389_389086


namespace count_valid_numbers_l389_389407

def is_valid_number (n : Nat) : Prop := 
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let units := n % 10
  1 ≤ hundreds ∧ hundreds ≤ 9 ∧
  0 ≤ tens ∧ tens ≤ 9 ∧
  units ≥ 3 * tens

theorem count_valid_numbers : 
  (Finset.filter is_valid_number (Finset.range 1000)).card = 198 :=
by sorry

end count_valid_numbers_l389_389407


namespace constant_term_in_binomial_expansion_l389_389057

-- Conditions
def a : ℝ := ∫ x in 0..π, Real.sin x

-- Theorem statement
theorem constant_term_in_binomial_expansion : 
  (∫ x in 0..π, Real.sin x) = 2 → 
  ((2 * Real.sqrt x - 1 / Real.sqrt x) ^ 6).eval 1 = -160 :=
by 
  -- skipping proof, add sorry
  sorry

end constant_term_in_binomial_expansion_l389_389057


namespace plane_through_midpoints_divides_tetrahedron_equally_l389_389881

noncomputable theory

variables {A B C D M K : Type} [Tetrahedron A B C D] [Midpoint M A B] [Midpoint K C D]

theorem plane_through_midpoints_divides_tetrahedron_equally (h1 : Midpoint M A B)
                                                          (h2 : Midpoint K C D)
                                                          (h3 : ∃ (P : Plane), P.contains M ∧ P.contains K) :
  divides_into_equal_volumes (Plane_through M K) A B C D :=
sorry

end plane_through_midpoints_divides_tetrahedron_equally_l389_389881


namespace vertex_coordinates_l389_389916

theorem vertex_coordinates (x : ℝ) : 
  let y := 3 * x^2 - 6 * x + 2 in
  ∃ (vx vy : ℝ), 
    (∀ x : ℝ, y = 3 * (x - vx) ^ 2 + vy) ∧ 
    (vx = 1) ∧ 
    (vy = -7) :=
by {
  sorry
}

end vertex_coordinates_l389_389916


namespace constant_term_in_binomial_expansion_max_coef_sixth_term_l389_389548

theorem constant_term_in_binomial_expansion_max_coef_sixth_term 
  (n : ℕ) (h : n = 10) : 
  (∃ C : ℕ → ℕ → ℕ, C 10 2 * (Nat.sqrt 2) ^ 8 = 720) :=
sorry

end constant_term_in_binomial_expansion_max_coef_sixth_term_l389_389548


namespace simplify_complex_expression_l389_389347

theorem simplify_complex_expression (i : ℂ) (h_i : i * i = -1) : 
  (11 - 3 * i) / (1 + 2 * i) = 3 - 5 * i :=
sorry

end simplify_complex_expression_l389_389347


namespace sum_of_first_six_primes_gt_ten_l389_389591

theorem sum_of_first_six_primes_gt_ten :
  let primes : List ℕ := [11, 13, 17, 19, 23, 29] in
  primes.sum = 112 :=
by
  sorry

end sum_of_first_six_primes_gt_ten_l389_389591


namespace smallest_solution_x4_50x2_576_eq_0_l389_389590

theorem smallest_solution_x4_50x2_576_eq_0 :
  ∃ x : ℝ, (x^4 - 50*x^2 + 576 = 0) ∧ ∀ y : ℝ, y^4 - 50*y^2 + 576 = 0 → x ≤ y :=
sorry

end smallest_solution_x4_50x2_576_eq_0_l389_389590


namespace reciprocal_of_repeating_decimal_l389_389982

theorem reciprocal_of_repeating_decimal :
  let x := (36 : ℚ) / 99 in
  x⁻¹ = 11 / 4 :=
by
  have h_simplify : x = 4 / 11 := by sorry
  rw [h_simplify, inv_div]
  norm_num
  exact eq.refl (11 / 4)

end reciprocal_of_repeating_decimal_l389_389982


namespace exists_two_points_distance_le_half_not_guaranteed_two_points_distance_le_049_l389_389933

-- Define an equilateral triangle and five points within it
structure EquilateralTriangle (α : Type) [MetricSpace α] :=
  (vertices : Set α)
  (side_length : ℝ)
  (is_equilateral : ∀ (x y : α), (x ∈ vertices → y ∈ vertices → dist x y = side_length ∨ dist x y = 0))

def points_in_triangle {α : Type} [MetricSpace α] (Δ : EquilateralTriangle α) : Set α := sorry

-- Assume there are 5 points inside or on the boundary of the triangle
axiom five_points_in_triangle (α : Type) [MetricSpace α] (Δ : EquilateralTriangle α) :
  ∃ (points : Set α), (points ⊆ points_in_triangle Δ) ∧ (points.card = 5)

-- First statement: There exist at least two points that are at most 1/2 units apart
theorem exists_two_points_distance_le_half {α : Type} [MetricSpace α] (Δ : EquilateralTriangle α)
  (h : Δ.side_length = 1) (h5 : ∃ (points : Set α), (points ⊆ points_in_triangle Δ) ∧ (points.card = 5)) :
  ∃ x y ∈ points_in_triangle Δ, dist x y ≤ 1 / 2 :=
sorry

-- Second statement: It is not guaranteed to have two points at most 0.49 units apart
theorem not_guaranteed_two_points_distance_le_049 {α : Type} [MetricSpace α] (Δ : EquilateralTriangle α)
  (h : Δ.side_length = 1) (h5 : ∃ (points : Set α), (points ⊆ points_in_triangle Δ) ∧ (points.card = 5)) :
  ¬ (∀ x y ∈ points_in_triangle Δ, dist x y ≤ 0.49) :=
sorry

end exists_two_points_distance_le_half_not_guaranteed_two_points_distance_le_049_l389_389933


namespace units_digit_18_power_18_7_power_7_l389_389300

theorem units_digit_18_power_18_7_power_7 :
  ∃ (u : ℕ), (u ⟨mod⟩ 10) = 4 ∧ ∃ (k : ℕ), 18 ^ (18 * 7 ^ 7) = k * 10 + u := by
  sorry

end units_digit_18_power_18_7_power_7_l389_389300


namespace find_even_and_increasing_function_l389_389212

-- Define the functions given in the problem
def f1 (x : ℝ) := x ^ 3
def f2 (x : ℝ) := Real.log x
def f3 (x : ℝ) := abs x
def f4 (x : ℝ) := 1 - x ^ 2

-- State the theorem
theorem find_even_and_increasing_function :
  (∀ x, x > 0 → (f3 x ≥ f3 0 ∧ ∀ y > x, f3 y > f3 x)) ∧
  (∀ x, abs (f3 x) = f3 x) ∧
  ¬ ((∀ x, x > 0 → (f1 x ≥ f1 0 ∧ ∀ y > x, f1 y > f1 x)) ∧ (∀ x, abs (f1 x) = f1 x)) ∧
  ¬ ((∀ x, x > 0 → (f2 x ≥ f2 0 ∧ ∀ y > x, f2 y > f2 x)) ∧ (∀ x, abs (f2 x) = f2 x)) ∧
  ¬ ((∀ x, x > 0 → (f4 x ≥ f4 0 ∧ ∀ y > x, f4 y > f4 x)) ∧ (∀ x, abs (f4 x) = f4 x)) := by
  sorry

end find_even_and_increasing_function_l389_389212


namespace alice_wins_2011_pow_2012_alice_wins_2012_pow_2011_l389_389934

-- Part a: Proof that Alice has a winning strategy if the game exceeds 2011^2012
theorem alice_wins_2011_pow_2012 (n : ℕ) (d : ℕ) (h : 0 < d ∧ d < n) :
  ∀ (n : ℕ), n ≤ 2011 ^ 2012 → Alice has a winning strategy :=
sorry

-- Part b: Proof that Alice has a winning strategy if the game exceeds 2012^2011
theorem alice_wins_2012_pow_2011 (n : ℕ) (d : ℕ) (h : 0 < d ∧ d < n) :
  ∀ (n : ℕ), n ≤ 2012 ^ 2011 → Alice has a winning strategy :=
sorry

end alice_wins_2011_pow_2012_alice_wins_2012_pow_2011_l389_389934


namespace allergic_reaction_probability_is_50_percent_l389_389038

def can_have_allergic_reaction (choice : String) : Prop :=
  choice = "peanut_butter"

def percentage_of_allergic_reaction :=
  let total_peanut_butter := 40 + 30
  let total_cookies := 40 + 50 + 30 + 20
  (total_peanut_butter : Float) / (total_cookies : Float) * 100

theorem allergic_reaction_probability_is_50_percent :
  percentage_of_allergic_reaction = 50 := sorry

end allergic_reaction_probability_is_50_percent_l389_389038


namespace zu_rate_number_count_l389_389024

def zu_rate : ℝ := 3.1415926

theorem zu_rate_number_count :
  ∀ (d1 d2 d3 d4 d5 d6 d7 : ℕ),
    {d1, d2, d3, d4, d5, d6, d7} = {1, 4, 1, 5, 9, 2, 6} ∧
    (3 + d1 * 0.1 + d2 * 0.01 < 3.14) →
    (3.111 < pi) ∧ (pi < 3.1415927) →
    240 := by
  sorry

end zu_rate_number_count_l389_389024


namespace proof_solution_l389_389104

noncomputable def proof_problem (x : ℂ) : Prop :=
  x ^ 2018 - 3 * x + 2 = 0 ∧ x ≠ 1 → x ^ 2017 + x ^ 2016 + ⋯ + x + 1 = 1

-- Note: The ellipsis (⋯) in the statement x ^ 2017 + x ^ 2016 + ⋯ + x + 1 represents the sum,
-- and needs to be rewritten using Lean's summation notation, for clarity.

noncomputable def proof_problem (x : ℂ) : Prop :=
  (x ^ 2018 - 3 * x + 2 = 0 ∧ x ≠ 1) → (Finset.range 2018).sum (λ i, x ^ i) = 1

theorem proof_solution (x : ℂ) : proof_problem x :=
sorry

end proof_solution_l389_389104


namespace find_first_4_hours_speed_l389_389610

noncomputable def average_speed_first_4_hours
  (total_avg_speed : ℝ)
  (first_4_hours_avg_speed : ℝ)
  (remaining_hours_avg_speed : ℝ)
  (total_time : ℕ)
  (first_4_hours : ℕ)
  (remaining_hours : ℕ) : Prop :=
  total_avg_speed * total_time = first_4_hours_avg_speed * first_4_hours + remaining_hours * remaining_hours_avg_speed

theorem find_first_4_hours_speed :
  average_speed_first_4_hours 50 35 53 24 4 20 :=
by
  sorry

end find_first_4_hours_speed_l389_389610


namespace weighted_sum_nonzero_l389_389719

theorem weighted_sum_nonzero (x : Fin 102 → ℤ) (h : ∀ i, x i = -1 ∨ x i = 1) : 
  ∑ i in Finset.range 101.succ, (i+1) * x ⟨i, Fin.is_lt i 101.succ_pos⟩ ≠ 0 :=
sorry

end weighted_sum_nonzero_l389_389719


namespace count_valid_three_digit_numbers_l389_389418

-- Define the property of a number being three digits
def isThreeDigit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

-- Define the property of the units digit being at least three times the tens digit
def validUnitsTimesTens (n : ℕ) : Prop :=
  let units := n % 10
  let tens := (n / 10) % 10
  units ≥ 3 * tens

-- Definition combining both conditions
def validThreeDigitNumber (n : ℕ) : Prop :=
  isThreeDigit n ∧ validUnitsTimesTens n

-- The main theorem stating the number of valid three-digit numbers
theorem count_valid_three_digit_numbers :
  Finset.card (Finset.filter validThreeDigitNumber (Finset.range 1000)) = 198 :=
by
  sorry

end count_valid_three_digit_numbers_l389_389418


namespace initial_crayons_count_l389_389150

theorem initial_crayons_count (C : ℕ) :
  (3 / 8) * C = 18 → C = 48 :=
by
  sorry

end initial_crayons_count_l389_389150


namespace intersection_M_N_l389_389371

def M : Set ℝ := {x | x^2 - 3 * x - 4 < 0}
def N : Set ℤ := {-2, -1, 0, 1, 2}

theorem intersection_M_N : (M ∩ N : Set ℤ) = {0, 1, 2} :=
by
  sorry

end intersection_M_N_l389_389371


namespace carl_garden_area_l389_389665

theorem carl_garden_area (x : ℕ) (longer_side_post_count : ℕ) (total_posts : ℕ) 
  (shorter_side_length : ℕ) (longer_side_length : ℕ) 
  (posts_per_gap : ℕ) (spacing : ℕ) :
  -- Conditions
  total_posts = 20 → 
  posts_per_gap = 4 → 
  spacing = 4 → 
  longer_side_post_count = 2 * x → 
  2 * x + 2 * (2 * x) - 4 = total_posts →
  shorter_side_length = (x - 1) * spacing → 
  longer_side_length = (longer_side_post_count - 1) * spacing →
  -- Conclusion
  shorter_side_length * longer_side_length = 336 :=
by
  sorry

end carl_garden_area_l389_389665


namespace min_orders_to_minimize_spent_l389_389176

-- Definitions for the given conditions
def original_price (n p : ℕ) : ℕ := n * p
def discounted_price (T : ℕ) : ℕ := (3 * T) / 5  -- Equivalent to 0.6 * T, using integer math

-- Define the conditions
theorem min_orders_to_minimize_spent 
  (n p : ℕ)
  (h1 : n = 42)
  (h2 : p = 48)
  : ∃ m : ℕ, m = 3 :=
by 
  sorry

end min_orders_to_minimize_spent_l389_389176


namespace range_of_x_in_function_l389_389794

theorem range_of_x_in_function :
  ∀ (x : ℝ), (x - 1 ≥ 0) ∧ (x ≠ 2) ↔ (x ≥ 1) ∧ (x ≠ 2) :=
by
  intro x
  split
  . intro h
    cases h
    split
    . exact h_left
    . exact h_right
  . intro h
    cases h
    split
    . exact h_left
    . exact h_right
  sorry

end range_of_x_in_function_l389_389794


namespace magnitude_diff_is_correct_l389_389318

def vector_a : ℝ × ℝ × ℝ := (2, -3, 1)
def vector_b : ℝ × ℝ × ℝ := (-1, 1, -4)

theorem magnitude_diff_is_correct : 
  ‖(2, -3, 1) - (-1, 1, -4)‖ = 5 * Real.sqrt 2 := 
by
  sorry

end magnitude_diff_is_correct_l389_389318


namespace sum_coefficients_l389_389603

noncomputable def f (x : ℝ) : ℝ := (x - 5)^7 + (x - 8)^5

theorem sum_coefficients : 
  let a := (λ x => (x - 5)^7 + (x - 8)^5) in
  let a_series := Polynomial.C a :=
  let a₀ : ℝ ... := Polynomial.a_Coeff a_series 0
  let a₁ ... := Polynomial.a_Coeff a_series 1
  let a₂ ... := Polynomial.a_Coeff a_series 2
  let a₃ ... := Polynomial.a_Coeff a_series 3
  let a₄ ... := Polynomial.a_Coeff a_series 4
  let a₅ ... := Polynomial.a_Coeff a_series 5
  let a₆ ... := Polynomial.a_Coeff a_series 6
  let a₇ ... := Polynomial.a_Coeff a_series 7
  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 127 :=
sorry

end sum_coefficients_l389_389603


namespace slope_of_best_fit_line_l389_389594

noncomputable def mean (xs : List ℝ) : ℝ :=
  xs.sum / xs.length

noncomputable def best_fit_slope (points : List (ℝ × ℝ)) : ℝ :=
  let xs := points.map Prod.fst
  let ys := points.map Prod.snd
  let x̄ := mean xs
  let ȳ := mean ys
  let numerator := (points.map (λ ⟨x, y⟩, (x - x̄) * (y - ȳ))).sum
  let denominator := (points.map (λ ⟨x, _⟩, (x - x̄)^2)).sum
  numerator / denominator

def problem_conditions (p1 p2 p3 : ℝ × ℝ) : Prop :=
  p1 = (150, 50) ∧ p2 = (160, 55) ∧ p3 = (170, 60.5)

theorem slope_of_best_fit_line : ∀ (p1 p2 p3 : ℝ × ℝ),
  problem_conditions p1 p2 p3 →
  best_fit_slope [p1, p2, p3] = 0.525 :=
by
  intros
  sorry

end slope_of_best_fit_line_l389_389594


namespace mouse_jump_distance_l389_389120

theorem mouse_jump_distance :
  (G : ℕ) (F : ℕ) (M : ℕ)
  (h1 : G = 39)
  (h2 : F = G - 19)
  (h3 : M = F - 12)
  : M = 8 :=
by
  sorry -- placeholder for the proof

end mouse_jump_distance_l389_389120


namespace kth_term_in_sequence_l389_389516

theorem kth_term_in_sequence (k : ℕ) (hk : 0 < k) : ℚ :=
  (2 * k) / (2 * k + 1)

end kth_term_in_sequence_l389_389516


namespace bags_of_bags_10_l389_389573

def Pi : ℕ → ℕ
| 0     := 1
| 1     := 1
| 2     := 1
| 3     := 2
| 4     := 4
| 5     := 9
| 6     := 20
| 7     := 48
| 8     := 115
| 9     := 286
| 10    := 719
| _     := sorry

theorem bags_of_bags_10 : Pi 10 = 719 := 
by 
  unfold Pi 
  sorry

end bags_of_bags_10_l389_389573


namespace constant_term_expansion_l389_389951

-- auxiliary definitions and facts
def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def term_constant (n k : ℕ) (a b x : ℂ) : ℂ :=
  binomial_coeff n k * (a * x)^(n-k) * (b / x)^k

-- main theorem statement
theorem constant_term_expansion : ∀ (x : ℂ), (term_constant 8 4 (5 : ℂ) (2 : ℂ) x).re = 1120 :=
by
  intro x
  sorry

end constant_term_expansion_l389_389951


namespace line_perpendicular_trapezoid_is_perpendicular_to_plane_l389_389565

variables {α : Type*} [EuclideanSpace α]

-- Define the trapezoid with its legs and bases
structure Trapezoid (α : Type*) [EuclideanSpace α] :=
(leg1 : Line α)
(leg2 : Line α)
(base1 : Line α)
(base2 : Line α)
(intersect_legs : ∃ p : α, p ∈ leg1 ∧ p ∈ leg2)
(in_plane : ∃ plane : Plane α, ∀ l ∈ {leg1, leg2, base1, base2}, ∀ p : α, p ∈ l → p ∈ plane)

-- Define perpendicularity relation
def is_perpendicular_to_legs (l : Line α) (tr : Trapezoid α) :=
Perpendicular l tr.leg1 ∧ Perpendicular l tr.leg2

def is_perpendicular_to_plane (l : Line α) (pl : Plane α) := ∀ p1 p2 : α, p1 ≠ p2 → p1 ∈ l → p2 ∈ l → Perpendicular (@vector_between _ _ _ p1 p2) pl

theorem line_perpendicular_trapezoid_is_perpendicular_to_plane (tr : Trapezoid α) (l : Line α) :
  is_perpendicular_to_legs l tr → (∃ plane : Plane α, ∀ l ∈ {tr.base1, tr.base2}, ∀ p : α, p ∈ l → p ∈ plane) →
  ∃ plane : Plane α, is_perpendicular_to_plane l plane :=
by
  sorry

end line_perpendicular_trapezoid_is_perpendicular_to_plane_l389_389565


namespace scalene_triangle_height_ratio_l389_389460

theorem scalene_triangle_height_ratio {a b c : ℝ} (h1 : a > b ∧ b > c ∧ a > c)
  (h2 : a + c = 2 * b) : 
  1 / 3 < c / a ∧ c / a < 1 :=
by sorry

end scalene_triangle_height_ratio_l389_389460


namespace total_students_sampled_l389_389789

theorem total_students_sampled (freq_ratio : ℕ → ℕ → ℕ) (second_group_freq : ℕ) 
  (ratio_condition : freq_ratio 2 1 = 2 ∧ freq_ratio 2 3 = 3) : 
  (6 + second_group_freq + 18) = 48 := 
by 
  sorry

end total_students_sampled_l389_389789


namespace log_base_5_of_3100_l389_389942

theorem log_base_5_of_3100 (h1 : 5^4 < 3100) (h2 : 3100 < 5^5) (h3 : 5^5 = 3125) : Int.round (Real.logb 5 3100) = 5 :=
by
  sorry

end log_base_5_of_3100_l389_389942


namespace count_positive_rationals_l389_389210

theorem count_positive_rationals :
  let numbers := [-2023, 0.01, (3 : ℚ) / 2, 0, 20 / 100]
  in list.countp (λ x, 0 < x) numbers = 3 := by
  sorry

end count_positive_rationals_l389_389210


namespace expression_evaluation_l389_389161

theorem expression_evaluation : 4 * (9 - 6) / 2 - 3 = 3 := 
by
  sorry

end expression_evaluation_l389_389161


namespace propositions_true_l389_389648

variables {A B : Point} {K : Real}
variable (p : Real)

def roots_of_quadratic (a b c : ℝ) : set ℝ :=
  { x | a * x ^ 2 + b * x + c = 0 }

def hyperbola_foci_eq_ellipse_foci : Prop :=
  (foci (hyperbola (1 / 25) (1 / 9))) = (foci (ellipse (1 / 35) (1)))

def chord_circle_tangent_to_directrix : Prop :=
  ∀ A B : Point, ∀ F l : Line, is_chord AB F → passes_through_directrix AB l → tangent_chord_circle_directrix A B F l

theorem propositions_true :
  (∀ A B : Point, ∀ K : Real, is_hyperbola_trajectory A B K ↔ (K ≠ dist A B)) ∧
  (roots_of_quadratic 2 (-5) 2 = {1/2, 2}) ∧
  hyperbola_foci_eq_ellipse_foci ∧
  chord_circle_tangent_to_directrix
:=
begin
  sorry
end

end propositions_true_l389_389648


namespace toms_age_is_16_l389_389485

variable (J T : ℕ) -- John's current age is J and Tom's current age is T

-- Condition 1: John was thrice as old as Tom 6 years ago
axiom h1 : J - 6 = 3 * (T - 6)

-- Condition 2: John will be 2 times as old as Tom in 4 years
axiom h2 : J + 4 = 2 * (T + 4)

-- Proving Tom's current age is 16
theorem toms_age_is_16 : T = 16 := by
  sorry

end toms_age_is_16_l389_389485


namespace reciprocal_of_repeating_decimal_l389_389979

theorem reciprocal_of_repeating_decimal :
  let x := (36 : ℚ) / 99 in
  x⁻¹ = 11 / 4 :=
by
  have h_simplify : x = 4 / 11 := by sorry
  rw [h_simplify, inv_div]
  norm_num
  exact eq.refl (11 / 4)

end reciprocal_of_repeating_decimal_l389_389979


namespace energy_difference_l389_389677

noncomputable def initial_energy (k q d : ℝ) : ℝ :=
  2 * (k * q^2 / d) + (k * q^2 / (2 * d))

noncomputable def new_energy (k q d : ℝ) : ℝ :=
  3 * (k * q^2 / d)

theorem energy_difference (k q d : ℝ) (h₁ : initial_energy k q d = 18) :
  new_energy k q d - initial_energy k q d = 3.6 :=
by
  sorry

end energy_difference_l389_389677


namespace triangle_BCD_area_l389_389820

theorem triangle_BCD_area (A B C D : EuclideanSpace ℝ (Fin 3))
  (hAB : A = ![0, 0, 0])
  (hAC : B = ![b, 0, 0])
  (hAD : C = ![0, c, 0])
  (hBD : D = ![0, 0, d])
  (hABC_area : ∥ cross_product (B - A) (C - A) ∥ / 2 = x)
  (hACD_area : ∥ cross_product (C - A) (D - A) ∥ / 2 = y)
  (hADB_area : ∥ cross_product (A - C) (D - B) ∥ / 2 = z) :
  ∥ cross_product (C - B) (D - B) ∥ / 2 = sqrt (x^2 + y^2 + z^2) := 
sorry

end triangle_BCD_area_l389_389820


namespace focus_of_parabola_l389_389551

theorem focus_of_parabola (y x p : ℝ) (h : y^2 = 20 * x) : ∃ p, (x, y) = (5, 0) :=
by
  use 5
  sorry

end focus_of_parabola_l389_389551


namespace max_interval_difference_l389_389776

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

theorem max_interval_difference (a b : ℝ) 
  (h1 : ∀ x ∈ Icc a b, f x ∈ Icc (-3 : ℝ) 1)
  (h2 : f a = -3)
  (h3 : f b = 1) :
  b - a = 4 :=
by
  -- Proof omitted
  sorry

end max_interval_difference_l389_389776


namespace number_of_valid_3_digit_numbers_l389_389415

theorem number_of_valid_3_digit_numbers : 
  ∃ (n : ℕ), 
    (∀ (h t u : ℕ), 
      (n = h * 100 + t * 10 + u) ∧ 
      (1 ≤ h ∧ h ≤ 9) ∧ 
      (0 ≤ t ∧ t ≤ 9) ∧ 
      (0 ≤ u ∧ u ≤ 9) ∧ 
      (u ≥ 3 * t)) →
      n = 198 := 
by
  sorry

end number_of_valid_3_digit_numbers_l389_389415


namespace Apollonius_Circle_radius_l389_389897

theorem Apollonius_Circle_radius (a λ : ℝ) (h_pos_a : 0 < a) (h_pos_λ : 0 < λ) (h_neq_1 : λ ≠ 1) :
  ∃ r : ℝ, r = (2 * a * λ) / |1 - λ^2| :=
sorry

end Apollonius_Circle_radius_l389_389897


namespace distance_to_airport_l389_389276

variable (d : ℝ) (t : ℝ)

-- Conditions as per the problem statement
def condition1 : Prop := t > 0
def condition2 : Prop := d = 45 * (t + 0.75)
def condition3 : Prop := d - 45 = 65 * (t - 1.75)

-- Target statement to be proved
theorem distance_to_airport (h1 : condition1) (h2 : condition2) (h3 : condition3) : d = 264 :=
by
  sorry

end distance_to_airport_l389_389276


namespace problem_4_5_part_I_problem_4_5_part_II_l389_389095

-- Definitions used in Lean 4 are directly from the conditions in a)
def f (x m : ℝ) : ℝ := abs (x - m) - abs (x + 3 * m)

-- Definition of the first proof
theorem problem_4_5_part_I (x : ℝ) : f x 1 ≥ 1 → x ≤ -3 / 2 := 
by
  sorry

-- Definition of the second proof
theorem problem_4_5_part_II (x t m : ℝ) (h : 0 < m) : (∀ x t : ℝ, f x m < abs (2 + t) + abs (t - 1)) → m < 3 / 4 := 
by
  sorry

end problem_4_5_part_I_problem_4_5_part_II_l389_389095


namespace identify_odd_and_decreasing_function_l389_389592

theorem identify_odd_and_decreasing_function :
  (∀ f, (f = (λ x : ℝ, -3 * x) ∧ (∀ x, f (-x) = -f x) ∧ (∀ x₁ x₂, x₁ < x₂ → f x₁ > f x₂)) ∨
        (f = (λ x : ℝ, x^3) ∧ (∀ x, f (-x) = -f x) ∧ ¬(∀ x₁ x₂, x₁ < x₂ → f x₁ > f x₂)) ∨
        (f = (λ x : ℝ, real.log (3:ℝ) x) ∧ ¬(∀ x, f (-x) = -f x)) ∨
        (f = (λ x : ℝ, 3^x) ∧ ¬(∀ x, f (-x) = -f x))) →
  (λ x : ℝ, -3 * x) = (λ x : ℝ, -3 * x) ∧ ∀ x, (λ x : ℝ, -3 * x) (-x) = -(λ x : ℝ, -3 * x x) ∧ ∀ x₁ x₂, x₁ < x₂ → (λ x : ℝ, -3 * x x₁) > (λ x : ℝ, -3 * x x₂) :=
begin
  sorry
end

end identify_odd_and_decreasing_function_l389_389592


namespace reciprocal_of_repeating_decimal_l389_389977

theorem reciprocal_of_repeating_decimal :
  let x : ℚ := 36 / 99 in
  1 / x = 11 / 4 :=
by
  sorry

end reciprocal_of_repeating_decimal_l389_389977


namespace reciprocal_of_repeating_decimal_l389_389969

noncomputable def repeating_decimal_to_fraction (x : ℚ) : ℚ :=
  if x = .\overline{36} then 4/11 else 0

theorem reciprocal_of_repeating_decimal :
  ∀ (x : ℚ), repeating_decimal_to_fraction (.\overline{36}) = 4/11 →
  1 / (repeating_decimal_to_fraction x) = 11/4 :=
by
  intros x hx
  have h : repeating_decimal_to_fraction x = 4/11 := hx
  rw h
  norm_num
  done
  sorry

end reciprocal_of_repeating_decimal_l389_389969


namespace smallest_positive_period_of_f_monotonically_increasing_intervals_of_f_l389_389747

noncomputable def f (x : ℝ) : ℝ := 2 * sin x * cos x + 2 * cos x ^ 2 - 1

theorem smallest_positive_period_of_f : ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = π := sorry

theorem monotonically_increasing_intervals_of_f : 
  ∀ k : ℤ, ∃ a b, a = k * π - 3 * π / 8 ∧ b = k * π + π / 8 ∧ ∀ {x : ℝ}, a ≤ x ∧ x ≤ b → ∀ {y : ℝ}, x ≤ y → f x ≤ f y := sorry

end smallest_positive_period_of_f_monotonically_increasing_intervals_of_f_l389_389747


namespace extreme_points_of_f_range_of_m_for_one_intersection_with_x_axis_l389_389499

noncomputable def f (x m : ℝ) : ℝ :=
  x^3 - x^2 - x + m

theorem extreme_points_of_f (m : ℝ) :
  (∃ x, f x m = x^3 - x^2 - x + m ∧ (f' x m = 0 ∧ f'' x m < 0 ∧ x = -1/3) ∨ (f' x m = 0 ∧ f'' x m > 0 ∧ x = 1)) := by
  sorry

theorem range_of_m_for_one_intersection_with_x_axis (m : ℝ) :
  (∃ x, f x m = 0 ∧ ∀ y, f y m = 0 → y = x) → (m < -5/27 ∨ m > 0) := by
  sorry

end extreme_points_of_f_range_of_m_for_one_intersection_with_x_axis_l389_389499


namespace problem1_problem2_l389_389341

-- Given conditions
variable (θ : ℝ)
variable (a : ℝ)
variable (h1 : Polynomial.X^2 - C a * Polynomial.X + C a = 0)

-- Correct Answers
theorem problem1 : a = 1 - Real.sqrt 2 → cos (π / 2 + θ) + sin (3 * π / 2 + θ) = Real.sqrt 2 - 1 := 
sorry

theorem problem2 : a = 1 - Real.sqrt 2 → (tan (π - θ)) - (1 / (tan θ)) = Real.sqrt 2 + 1 :=
sorry

end problem1_problem2_l389_389341


namespace monotonically_decreasing_intervals_l389_389126
open Real

noncomputable def f (x : ℝ) : ℝ := (1 / 3) ^ abs (cos x)

theorem monotonically_decreasing_intervals :
  MonotoneDecreasingOn f (Icc (-π/2) 0) ∧ MonotoneDecreasingOn f (Icc (π/2) π) :=
by
  sorry

end monotonically_decreasing_intervals_l389_389126


namespace reciprocal_of_repeating_decimal_l389_389987

theorem reciprocal_of_repeating_decimal :
  let x := (36 : ℚ) / 99 in 1 / x = 11 / 4 :=
by
  trivial -- This is a placeholder, the actual proof is not required and hence replaced by trivial.

end reciprocal_of_repeating_decimal_l389_389987


namespace number_of_valid_3_digit_numbers_l389_389411

theorem number_of_valid_3_digit_numbers : 
  ∃ (n : ℕ), 
    (∀ (h t u : ℕ), 
      (n = h * 100 + t * 10 + u) ∧ 
      (1 ≤ h ∧ h ≤ 9) ∧ 
      (0 ≤ t ∧ t ≤ 9) ∧ 
      (0 ≤ u ∧ u ≤ 9) ∧ 
      (u ≥ 3 * t)) →
      n = 198 := 
by
  sorry

end number_of_valid_3_digit_numbers_l389_389411


namespace three_digit_numbers_count_l389_389435

def is_3_digit_number (n : Nat) : Prop :=
  100 ≤ n ∧ n ≤ 999

def units_digit (n : Nat) : Nat :=
  n % 10

def tens_digit (n : Nat) : Nat :=
  (n / 10) % 10

def hundreds_digit (n : Nat) : Nat :=
  (n / 100) % 10

def valid_number (n : Nat) : Prop :=
  is_3_digit_number n ∧ units_digit n ≥ 3 * tens_digit n

theorem three_digit_numbers_count : 
  (Finset.card (Finset.filter valid_number (Finset.range 1000))) = 198 :=
by
  sorry

end three_digit_numbers_count_l389_389435


namespace three_digit_numbers_with_units_at_least_three_times_tens_l389_389394

open Nat

noncomputable def valid_units_and_tens_digit (tens units : ℕ) : Prop :=
  (tens = 0 ∧ units ∈ range 10) ∨
  (tens = 1 ∧ units ∈ range' 3 10) ∨
  (tens = 2 ∧ units ∈ range' 6 10)

noncomputable def valid_hundreds_digit (hundreds : ℕ) : Prop :=
  hundreds ∈ range' 1 10

theorem three_digit_numbers_with_units_at_least_three_times_tens :
  let total_numbers := 189
  ∃ total_numbers : ℕ, total_numbers = ∑ h in range' 1 10, ∑ t in range 10, ∑ u in range 10, if valid_hundreds_digit h ∧ valid_units_and_tens_digit t u then 1 else 0 :=
begin
  sorry
end

end three_digit_numbers_with_units_at_least_three_times_tens_l389_389394


namespace math_problem_l389_389362

variable {f : ℝ → ℝ}

theorem math_problem (h1 : ∀ x y : ℝ, f (x + y) = f x + f y)
                     (h2 : ∀ x : ℝ, x > 0 → f x > 0)
                     (h3 : f 1 = 2) :
                     f 0 = 0 ∧
                     (∀ x : ℝ, f (-x) = -f x) ∧
                     (∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2) ∧
                     (∃ a : ℝ, f (2 - a) = 6 ∧ a = -1) := 
by
  sorry

end math_problem_l389_389362


namespace tan_angle_sum_l389_389512

noncomputable def tan_sum (θ : ℝ) : ℝ := Real.tan (θ + (Real.pi / 4))

theorem tan_angle_sum :
  let x := 1
  let y := 2
  let θ := Real.arctan (y / x)
  tan_sum θ = -3 := by
  sorry

end tan_angle_sum_l389_389512


namespace length_of_chord_AB_l389_389625

-- Define the parabola equation
def parabola (x y : ℝ) : Prop := y^2 = 3*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (3/4, 0)

-- Define the line passing through the focus with an inclination of 30 degrees
def line_through_focus (θ : ℝ) (x y : ℝ) : Prop :=
  y = tan θ * (x - (focus.1))

-- Inclination of 30 degrees in radians
def inclination : ℝ := π / 6

-- The main problem statement
theorem length_of_chord_AB :
  ∀ A B : ℝ × ℝ,
  parabola A.1 A.2 →
  parabola B.1 B.2 →
  A ≠ B →
  line_through_focus inclination A.1 A.2 →
  line_through_focus inclination B.1 B.2 →
  (A.1 + B.1 + 3/2 = 21/2) →
  (ℝ.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 12) :=
by
  sorry

end length_of_chord_AB_l389_389625


namespace circle_radius_integer_l389_389172

noncomputable def radius_of_circle_centered_at (x1 y1 : ℝ) (r : ℝ) (x2 y2 : ℝ) := 
  (x2 - x1)^2 + (y2 - y1)^2 = r^2

theorem circle_radius_integer (r : ℕ) 
  (h_center : ∀ x y : ℝ, (x, y) = (-2, -3))
  (h_inside : (radius_of_circle_centered_at (-2) (-3) _ (-2) 3) < (6)^2)
  (h_outside : (radius_of_circle_centered_at (-2) (-3) _ 6 (-3)) > (8)^2) :
  r = 7 :=
by sorry

end circle_radius_integer_l389_389172


namespace cos_two_beta_sin_alpha_l389_389718

-- Given conditions
variables (α β : Real) (h₀ : 0 < α) (h₁ : α < π/2) (h₂ : 0 < β) (h₃ : β < π/2)
variables (h₄ : sin β = 1/3)
variables (h₅ : sin (α - β) = 3/5)

-- Problem statements

-- 1. Prove that cos 2β = 7/9
theorem cos_two_beta : cos (2 * β) = 7 / 9 :=
by sorry

-- 2. Prove that sin α = (6 * sqrt 2 + 4) / 15
theorem sin_alpha : sin α = (6 * sqrt 2 + 4) / 15 :=
by sorry

end cos_two_beta_sin_alpha_l389_389718


namespace analytical_synthetic_correct_l389_389480

-- Let us define the predicates for analytical and synthetic methods
def analytical_method_ideas : Prop := 
"It is often the case that the analytical method is used to find the ideas and methods for solving problems"

def synthetic_method_demonstration : Prop := 
"the synthetic method is used to demonstrate the process of solving problems"

-- The main theorem we need to prove
theorem analytical_synthetic_correct (h1 : analytical_method_ideas)
                                     (h2: synthetic_method_demonstration) :
  (analytical_method_ideas ∧ synthetic_method_demonstration) :=
by {
  -- skipping the actual proof
  sorry
}

end analytical_synthetic_correct_l389_389480


namespace problem_1_problem_2_l389_389058

def f (x : ℝ) : ℝ := |x - 1| - |x + 3|

-- Part 1: Prove the solution set of f(x) > 2 is {x | x < -2}.
theorem problem_1 : {x : ℝ | f x > 2} = {x : ℝ | x < -2} :=
by 
  sorry

-- Part 2: Prove the range of k if f(x) <= kx + 1 for x in [-3, -1].
theorem problem_2 : ∀ k : ℝ, (∀ x : ℝ, x ∈ Icc (-3 : ℝ) (-1 : ℝ) → f x ≤ k * x + 1) → k ≤ -1 :=
by 
  sorry

end problem_1_problem_2_l389_389058


namespace count_valid_3_digit_numbers_l389_389382

def valid_units_digit (tens_digit units_digit : ℕ) : Prop :=
  units_digit ≥ 3 * tens_digit

def count_valid_numbers : ℕ :=
  (∑ tens_digit in {0, 1, 2, 3}, (finset.filter (valid_units_digit tens_digit) (finset.range 10)).card) * 9

theorem count_valid_3_digit_numbers : count_valid_numbers = 198 :=
by
  sorry

end count_valid_3_digit_numbers_l389_389382


namespace find_a_100_l389_389721

-- Define the sequence and the sum of the sequence
def a (n : ℕ) : ℕ := sorry -- We will define this later
def S (n : ℕ) : ℕ := ∑ i in finset.range n, a i

-- Given conditions
axiom a1 : a 1 = 1

axiom a_n (n : ℕ) (h : 2 ≤ n): a n = n * (2 * S n ^ 2) / (2 * S n - 1)

-- The problem is to find a_100
theorem find_a_100 : a 100 = 100 * (2 * S 100 ^ 2) / (2 * S 100 - 1) :=
sorry

end find_a_100_l389_389721


namespace Keanu_refills_14_times_l389_389488

theorem Keanu_refills_14_times
  (gas_storage : ℕ)
  (distance_one_way : ℕ)
  (consumption_per_40_miles : ℕ)
  (round_trip_distance := 2 * distance_one_way)
  (gas_consumption_per_mile := consumption_per_40_miles / 40)
  (total_gas_consumption := round_trip_distance * gas_consumption_per_mile)
  (refills := total_gas_consumption / gas_storage) :
  gas_storage = 8 →
  distance_one_way = 280 →
  consumption_per_40_miles = 8 →
  refills = 14 := by
    intros hs dist cp
    simp only [hs, dist, cp]
    sorry

end Keanu_refills_14_times_l389_389488


namespace largest_c_value_l389_389292

theorem largest_c_value (c : ℝ) :
  (∃ x : ℝ, x^2 + 5 * x + c = -3) → c ≤ 13 / 4 :=
sorry

end largest_c_value_l389_389292


namespace contains_zero_if_sum_is_111111_l389_389216

variables (x y : ℕ)
variables (digits_x digits_y : Fin 5 → ℕ)

-- Conditions
def is_five_digit (n : ℕ) : Prop := 
  10000 ≤ n ∧ n < 100000

def differs_by_two_digit_swap (dx dy : Fin 5 → ℕ) : Prop :=
  ∃ i j : Fin 5, i ≠ j ∧ 
  (∀ k : Fin 5, (k = i → dy k = dx j) ∧ 
                (k = j → dy k = dx i) ∧ 
                (k ≠ i ∧ k ≠ j → dy k = dx k))

theorem contains_zero_if_sum_is_111111 
  (hx : is_five_digit x) (hy : is_five_digit y)
  (h_digits_x : digits_x = fun i : Fin 5 => (x / 10^(4-i) % 10))
  (h_digits_y : digits_y = fun i : Fin 5 => (y / 10^(4-i) % 10))
  (h_swap : differs_by_two_digit_swap digits_x digits_y)
  (h_sum : x + y = 111111) :
  ∃ i : Fin 5, digits_x i = 0 :=
begin
  -- Proof omitted
  sorry,
end

end contains_zero_if_sum_is_111111_l389_389216


namespace three_digit_number_units_digit_condition_l389_389389

theorem three_digit_number_units_digit_condition :
  let valid_numbers := (λ (n : ℕ), (100 ≤ n ∧ n ≤ 999) ∧ ((n % 10) ≥ 3 * ((n / 10) % 10))) in
  (Finset.filter valid_numbers (Finset.range 900)) = 198 :=
by
  sorry

end three_digit_number_units_digit_condition_l389_389389


namespace scientific_notation_of_grainOutput_l389_389595

-- Define the condition and the problem statement
def grainOutput : ℕ := 130000000000

theorem scientific_notation_of_grainOutput :
  grainOutput = 1.3 * 10 ^ 11 := by
  sorry

end scientific_notation_of_grainOutput_l389_389595


namespace dot_product_is_one_l389_389778

variable (a : ℝ × ℝ := (1, 1))
variable (b : ℝ × ℝ := (-1, 2))

theorem dot_product_is_one : (a.1 * b.1 + a.2 * b.2) = 1 := by
  sorry

end dot_product_is_one_l389_389778


namespace inequalities_in_acute_triangle_l389_389461

open Real -- Open the real number namespace

variables {A B : ℝ} -- Declare variables A and B as real numbers

-- Assert the conditions as definitions
def acute_angle (θ : ℝ) : Prop := 0 < θ ∧ θ < π / 2
def A_gt_B : Prop := A > B
def A_and_B_are_acute : Prop := acute_angle A ∧ acute_angle B

-- Define the proof problem
theorem inequalities_in_acute_triangle (h1 : A_and_B_are_acute) (h2 : A_gt_B) :
  (sin A > sin B) ∧ (cos A < cos B) ∧ (cos 2 * A < cos 2 * B) :=
by 
  sorry

end inequalities_in_acute_triangle_l389_389461


namespace general_term_seq_l389_389758

theorem general_term_seq (a : ℕ → ℚ) 
  (h₁ : a 1 = 1/2) 
  (h₂ : ∀ n : ℕ, 1 ≤ n → (∑ i in Finset.range n.succ, a i) = n^2 * a n) : 
  ∀ n, a n = 1 / (n * (n + 1)) := 
sorry

end general_term_seq_l389_389758


namespace balance_balls_l389_389869

variable {R Y B W : ℕ}

theorem balance_balls (h1 : 4 * R = 8 * B) 
                      (h2 : 3 * Y = 9 * B) 
                      (h3 : 5 * B = 3 * W) : 
    (2 * R + 4 * Y + 3 * W) = 21 * B :=
by 
  sorry

end balance_balls_l389_389869


namespace count_valid_three_digit_numbers_l389_389417

-- Define the property of a number being three digits
def isThreeDigit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

-- Define the property of the units digit being at least three times the tens digit
def validUnitsTimesTens (n : ℕ) : Prop :=
  let units := n % 10
  let tens := (n / 10) % 10
  units ≥ 3 * tens

-- Definition combining both conditions
def validThreeDigitNumber (n : ℕ) : Prop :=
  isThreeDigit n ∧ validUnitsTimesTens n

-- The main theorem stating the number of valid three-digit numbers
theorem count_valid_three_digit_numbers :
  Finset.card (Finset.filter validThreeDigitNumber (Finset.range 1000)) = 198 :=
by
  sorry

end count_valid_three_digit_numbers_l389_389417


namespace price_of_book_l389_389518

variables (D B : ℝ)

def younger_brother : ℝ := 10

theorem price_of_book 
  (h1 : D = 1/2 * (B + younger_brother))
  (h2 : B = 1/3 * (D + younger_brother)) : 
  D + B + younger_brother = 24 := 
sorry

end price_of_book_l389_389518


namespace minimum_positive_period_minimum_value_l389_389125

noncomputable def f (x : Real) : Real :=
  Real.sin (x / 5) - Real.cos (x / 5)

theorem minimum_positive_period (T : Real) : (∀ x, f (x + T) = f x) ∧ T > 0 → T = 10 * Real.pi :=
  sorry

theorem minimum_value : ∃ x, f x = -Real.sqrt 2 :=
  sorry

end minimum_positive_period_minimum_value_l389_389125


namespace problem_I_problem_II_problem_III_l389_389716

noncomputable def f (x t : ℝ) : ℝ := Real.exp x - t * (x + 1)

noncomputable def g (x t : ℝ) : ℝ := f x t + t / Real.exp x

/- Proof Problem I -/
theorem problem_I (x : ℝ) (h : 0 < x) (hx : ∀ x > 0, f x t ≥ 0) : t ≤ 1 := sorry

/- Proof Problem II -/
theorem problem_II (x1 x2 : ℝ) (h : x1 ≠ x2) (ht : t ≤ -1) 
    (hx : ∀ x1 x2 : ℝ, x1 < x2 → (g x2 t - g x1 t > m * (x2 - x1))) :
    m < 3 := sorry

/- Proof Problem III -/
theorem problem_III (n : ℕ) : Real.log (1 + n) < (Finset.range n).sum (λ i, 1 / (i + 1)) ∧
    (Finset.range n).sum (λ i, 1 / (i + 1)) ≤ 1 + Real.log n := sorry

end problem_I_problem_II_problem_III_l389_389716


namespace reciprocal_of_repeating_decimal_l389_389966

theorem reciprocal_of_repeating_decimal :
  let x := 0.36363636 -- simplified as .\overline{36}
  ∃ y : ℚ, x = 4 / 11 ∧ y = 1 / x ∧ y = 11 / 4 :=
by
  sorry

end reciprocal_of_repeating_decimal_l389_389966


namespace point_A_path_length_l389_389199

-- Definitions of the problem conditions
structure Rectangle where
  A B C D : Point
  AB CD : ℝ
  BC DA : ℝ
  deriving Inhabited

noncomputable def distance (p1 p2 : Point) : ℝ := 
  real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Definitions of rotations and distances
noncomputable def rotated_distance (rectangle : Rectangle) : ℝ :=
  let AD := distance rectangle.A rectangle.D
  in 3 * (π * AD / 2)

-- Statement of the proof problem
theorem point_A_path_length (rectangle : Rectangle) (h_AB : rectangle.AB = 3) (h_BC : rectangle.BC = 5) :
  rotated_distance rectangle = (3 * π * sqrt 34) / 2 :=
sorry

end point_A_path_length_l389_389199


namespace min_subset_size_digit_condition_l389_389815

noncomputable def smallest_subset_size (S : set ℕ) : ℕ :=
  if h : ∃ N > 0, ∀ n > N, ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ n = a + b ∧ (∀ d ∈ (digits 10 a) ++ (digits 10 b), d ∈ S) then
    S.card
  else
    0

theorem min_subset_size_digit_condition (S : set ℕ) :
  (∃ N > 0, ∀ n > N, ∃ a b : ℕ,
    a > 0 ∧ b > 0 ∧ n = a + b ∧
    (∀ d ∈ (digits 10 a) ++ (digits 10 b), d ∈ S))
  → S.card ≥ 5 :=
sorry

end min_subset_size_digit_condition_l389_389815


namespace constant_term_expansion_l389_389945

theorem constant_term_expansion (x : ℝ) :
  (constant_term ((5 * x + 2 / (5 * x)) ^ 8) = 1120) := by
  sorry

end constant_term_expansion_l389_389945


namespace paint_gallons_needed_l389_389577

theorem paint_gallons_needed (n : ℕ) (h : n = 16) (h_col_height : ℝ) (h_col_height_val : h_col_height = 24)
  (h_col_diameter : ℝ) (h_col_diameter_val : h_col_diameter = 8) (cover_area : ℝ) 
  (cover_area_val : cover_area = 350) : 
  ∃ (gallons : ℤ), gallons = 33 := 
by
  sorry

end paint_gallons_needed_l389_389577


namespace find_first_term_l389_389142

def geom_seq (a r : ℝ) (n : ℕ) : ℝ := a * r^(n - 1)

theorem find_first_term (a r : ℝ) (h1 : r = 2/3) (h2 : geom_seq a r 3 = 18) (h3 : geom_seq a r 4 = 12) : a = 40.5 := 
by sorry

end find_first_term_l389_389142


namespace suresh_average_after_13th_and_minimum_score_in_11th_l389_389105

theorem suresh_average_after_13th_and_minimum_score_in_11th (
  A : ℝ) -- Average runs for the first 9 innings
  (h1 : ∀ i ∈ {10, 11, 12}, 70 ≤ 390 / 3) -- Maintain at least 70 runs in the 10th, 11th, and 12th innings
  (h2 : 200) -- Runs scored in the 13th inning
  (h3 : 8) -- Overall average increases by 8 runs
  :
  let B := A + 8 in 
  let total_runs_after_13th := 9 * A + 390 + 200 in 
  (B = total_runs_after_13th / 13) ∧ 
  (min_score_11th : ℝ := 250) (100 + min_score_11th + 70 = 390) :=
  sorry

end suresh_average_after_13th_and_minimum_score_in_11th_l389_389105


namespace probability_convex_quadrilateral_l389_389098

theorem probability_convex_quadrilateral (pts : Fin 6 → circle) :
  let total_chords := Nat.choose 6 2,
      total_ways_to_select_4_chords := Nat.choose total_chords 4,
      favorable_outcomes := Nat.choose 6 4
  in total_chords = 15 ∧ total_ways_to_select_4_chords = 1365 ∧ favorable_outcomes = 15 →
     (favorable_outcomes : ℚ) / total_ways_to_select_4_chords = 1 / 91 :=
by
  intros total_chords total_ways_to_select_4_chords favorable_outcomes h
  cases h with h1 h2
  cases h2 with h3 h4
  have h5 : (favorable_outcomes : ℚ) / total_ways_to_select_4_chords = 15 / 1365, from sorry,
  have h6 : 15 / 1365 = 1 / 91, from sorry,
  exact eq.trans h5 h6

end probability_convex_quadrilateral_l389_389098


namespace silver_cube_selling_price_l389_389654

noncomputable def side_length : ℝ := 3
noncomputable def weight_per_cubic_inch : ℝ := 6
noncomputable def price_per_ounce : ℝ := 25
noncomputable def selling_percentage : ℝ := 1.10

theorem silver_cube_selling_price :
  let volume := side_length ^ 3 in
  let weight := volume * weight_per_cubic_inch in
  let value := weight * price_per_ounce in
  let selling_price := value * selling_percentage in
  selling_price = 4455 := 
by
  sorry

end silver_cube_selling_price_l389_389654


namespace triangle_division_l389_389479

theorem triangle_division (n : ℕ) (h : n = 1997) : 
  let a_0 := 1 in 
  let a_n := a_0 + n * 2 in 
  a_n = 3995 :=
by 
  sorry

end triangle_division_l389_389479


namespace three_digit_numbers_count_l389_389426

theorem three_digit_numbers_count :
  let t_range := finset.range 10
  let h_range := finset.range 9
  let valid_u t := if t * 3 ≤ 9 then 9 - (t * 3) + 1 else 0
  let total_count := h_range.card * t_range.sum (λ t, valid_u t)
  total_count = 198 :=
by
  sorry

end three_digit_numbers_count_l389_389426


namespace height_at_end_of_2_years_l389_389641

-- Step d): Define the conditions and state the theorem

-- Define a function modeling the height of the tree each year
def tree_height (initial_height : ℕ) (years : ℕ) : ℕ :=
  initial_height * 3^years

-- Given conditions as definitions
def year_4_height := 81 -- height at the end of 4 years

-- Theorem that we need to prove
theorem height_at_end_of_2_years (initial_height : ℕ) (h : tree_height initial_height 4 = year_4_height) :
  tree_height initial_height 2 = 9 :=
sorry

end height_at_end_of_2_years_l389_389641


namespace problem_1_problem_2_l389_389756

theorem problem_1 (a b : ℝ) 
  (h_a : a ∈ Ioo (-1/2) (1/2))
  (h_b : b ∈ Ioo (-1/2) (1/2)) :
  abs (1/3 * a + 1/6 * b) < 1/4 :=
sorry

theorem problem_2 (a b : ℝ) 
  (h_a : a ∈ Ioo (-1/2) (1/2))
  (h_b : b ∈ Ioo (-1/2) (1/2)) :
  abs (1 - 4 * a * b) > 2 * abs (a - b) :=
sorry

end problem_1_problem_2_l389_389756


namespace reciprocal_of_repeating_decimal_l389_389965

theorem reciprocal_of_repeating_decimal :
  let x := 0.36363636 -- simplified as .\overline{36}
  ∃ y : ℚ, x = 4 / 11 ∧ y = 1 / x ∧ y = 11 / 4 :=
by
  sorry

end reciprocal_of_repeating_decimal_l389_389965


namespace number_of_valid_3_digit_numbers_l389_389414

theorem number_of_valid_3_digit_numbers : 
  ∃ (n : ℕ), 
    (∀ (h t u : ℕ), 
      (n = h * 100 + t * 10 + u) ∧ 
      (1 ≤ h ∧ h ≤ 9) ∧ 
      (0 ≤ t ∧ t ≤ 9) ∧ 
      (0 ≤ u ∧ u ≤ 9) ∧ 
      (u ≥ 3 * t)) →
      n = 198 := 
by
  sorry

end number_of_valid_3_digit_numbers_l389_389414


namespace convex_polyhedron_edge_sum_leq_3div8_l389_389814

theorem convex_polyhedron_edge_sum_leq_3div8
  (P : Type)
  [fintype P]
  (V : finset P)
  (x : P → ℝ)
  (E : finset (P × P))
  (hV : ∀ v ∈ V, 0 ≤ x v)
  (h_sum : ∑ v in V, x v = 1)
  (edge_weight : (P × P) → ℝ := λ edge, x edge.fst * x edge.snd) :
  ∑ e in E, edge_weight e ≤ 3 / 8 :=
by
  sorry

end convex_polyhedron_edge_sum_leq_3div8_l389_389814


namespace complex_quadrant_l389_389774

theorem complex_quadrant (z : ℂ) (h1 : (1 + I) * z = complex.abs (√3 + I)) : (z.re > 0) ∧ (z.im < 0) := by
  sorry

end complex_quadrant_l389_389774


namespace log_equation_solution_l389_389445

theorem log_equation_solution (x : ℝ) (hx : log 2 (2 * x^2 - 7 * x + 12) = 3) : 
  x = 2 ∨ x = 1 / 2 := 
sorry

end log_equation_solution_l389_389445


namespace minimize_expression_l389_389502

theorem minimize_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (cond1 : x + y > z) (cond2 : y + z > x) (cond3 : z + x > y) :
  (x + y + z) * (1 / (x + y - z) + 1 / (y + z - x) + 1 / (z + x - y)) ≥ 9 :=
by
  sorry

end minimize_expression_l389_389502


namespace cone_and_cylinder_volumes_not_equal_min_volume_ratio_l389_389188

noncomputable def volume_cone (r : ℝ) (theta : ℝ) : ℝ :=
  (1 / 3) * real.pi * (r * (1 + (1 / real.sin theta)) * real.tan theta)^2 * (r * (1 + (1 / real.sin theta)))

def volume_cylinder (r : ℝ) : ℝ :=
  real.pi * r^2 * (2 * r)

theorem cone_and_cylinder_volumes_not_equal (r : ℝ) (theta : ℝ) :
  volume_cone r theta ≠ volume_cylinder r := sorry

theorem min_volume_ratio (r : ℝ) (theta : ℝ) (h : real.sin theta = 1 / 3) :
  (volume_cone r theta) / (volume_cylinder r) = 4 / 3 := sorry

end cone_and_cylinder_volumes_not_equal_min_volume_ratio_l389_389188


namespace min_colors_complete_graph_l389_389582

open Fin

noncomputable def min_colors_for_kcomplete_graph (n : ℕ) : ℕ :=
  n

theorem min_colors_complete_graph (n : ℕ) :
  ∀ (G : SimpleGraph (Fin n)),
  (G = SimpleGraph.complete (Fin n)) →
  (∀ (v : Fin n) (i j : Fin n), i ≠ j → G.edge (v, i) ≠ G.edge (v, j)) →
  (∀ (v : Fin n) (i : Fin n), G.edge (v, i) ≠ n) →
  min_colors_for_kcomplete_graph n = n :=
by sorry

end min_colors_complete_graph_l389_389582


namespace number_of_valid_3_digit_numbers_l389_389410

theorem number_of_valid_3_digit_numbers : 
  ∃ (n : ℕ), 
    (∀ (h t u : ℕ), 
      (n = h * 100 + t * 10 + u) ∧ 
      (1 ≤ h ∧ h ≤ 9) ∧ 
      (0 ≤ t ∧ t ≤ 9) ∧ 
      (0 ≤ u ∧ u ≤ 9) ∧ 
      (u ≥ 3 * t)) →
      n = 198 := 
by
  sorry

end number_of_valid_3_digit_numbers_l389_389410


namespace ratio_diamond_brace_ring_l389_389812

theorem ratio_diamond_brace_ring
  (cost_ring : ℤ) (cost_car : ℤ) (total_worth : ℤ) (cost_diamond_brace : ℤ)
  (h1 : cost_ring = 4000) (h2 : cost_car = 2000) (h3 : total_worth = 14000)
  (h4 : cost_diamond_brace = total_worth - (cost_ring + cost_car)) :
  cost_diamond_brace / cost_ring = 2 :=
by
  sorry

end ratio_diamond_brace_ring_l389_389812


namespace num_valid_arithmetic_sequences_l389_389850

theorem num_valid_arithmetic_sequences (n : ℕ) :
  ∃ (A : finset ℕ), (∀ d > 0, 
    (∀ a b ∈ A, b - a = d)) ∧ 
    (∀ x ∈ range (n + 1), x ∉ A → ∀ y ∈ A, ∀ d > 0, 
    ((∃ k : ℕ, y = x + k * d) → x ∈ A) → False) ∧ 
    (A.card = ⌊n^2 / 4⌋) :=
sorry

end num_valid_arithmetic_sequences_l389_389850


namespace general_term_formula_l389_389742

def seq (n : ℕ) : ℤ :=
  if n = 0 then -1 
  else 2 * seq (n - 1) + 2

theorem general_term_formula (a : ℕ → ℤ) (h₀ : a 0 = -1) (h₁ : ∀ n, a (n + 1) = 2 * a n + 2) :
  ∀ n, a n = 2^(n-1).natAbs - 2 :=
by
  sorry

end general_term_formula_l389_389742


namespace nails_count_l389_389608

theorem nails_count (side_length : ℕ) (nails_per_side : ℕ) (equal_distance : Π (i : ℕ), i ≤ nails_per_side) : 
  side_length = 24 → nails_per_side = 25 → 
  ∀ i, (1 ≤ i ∧ i < nails_per_side) → (equal_distance i - equal_distance (i-1) = equal_distance (i+1) - equal_distance i) →
  ∑ i in (range (4 * nails_per_side)).erase [0, nails_per_side-1, (nails_per_side*(2)-1), (nails_per_side*(3)-1)], 1 = 96 :=
by
  intros
  sorry

end nails_count_l389_389608


namespace intersection_points_on_single_circle_l389_389908

-- Define the conditions
variables (A B C D E X H K L M N : Type) -- Vertex types
variables (ABCDEX : convex_polygon A B C D E X)
variables (star : points_extended_to_star A H B K C L D M E N)
variables (circumscribed_circles : circumscribed_circles_around_triangles star)

-- Problem Statement in Lean 4
theorem intersection_points_on_single_circle :
  ∃ (circle : Type) (points : List Type),
  points.length = 5 ∧ 
  ∀ P ∈ points, circumscribed_circles_intersect P ∧
  points_distinct points A B C D E ∧
  points_on_circle points circle := 
sorry

end intersection_points_on_single_circle_l389_389908


namespace sal_less_than_phil_l389_389813

variables (S P : ℝ)

def percentage_less (P S : ℝ) : ℝ :=
  ((P - S) / P) * 100

theorem sal_less_than_phil :
  1.40 * S = 1.12 ∧
  S + P = 1.80 →
  percentage_less P S = 20 :=
by
  sorry

end sal_less_than_phil_l389_389813


namespace tan_alpha_minus_pi_div_4_l389_389764

open Real

theorem tan_alpha_minus_pi_div_4 (α : ℝ) (h : (cos α * 2 + (-1) * sin α = 0)) : 
  tan (α - π / 4) = 1 / 3 :=
sorry

end tan_alpha_minus_pi_div_4_l389_389764


namespace fish_population_estimate_l389_389925

theorem fish_population_estimate :
  (∀ (x : ℕ),
    ∃ (m n k : ℕ), 
      m = 30 ∧
      k = 2 ∧
      n = 30 ∧
      ((k : ℚ) / n = m / x) → x = 450) :=
by
  sorry

end fish_population_estimate_l389_389925


namespace three_digit_numbers_with_units_at_least_three_times_tens_l389_389400

open Nat

noncomputable def valid_units_and_tens_digit (tens units : ℕ) : Prop :=
  (tens = 0 ∧ units ∈ range 10) ∨
  (tens = 1 ∧ units ∈ range' 3 10) ∨
  (tens = 2 ∧ units ∈ range' 6 10)

noncomputable def valid_hundreds_digit (hundreds : ℕ) : Prop :=
  hundreds ∈ range' 1 10

theorem three_digit_numbers_with_units_at_least_three_times_tens :
  let total_numbers := 189
  ∃ total_numbers : ℕ, total_numbers = ∑ h in range' 1 10, ∑ t in range 10, ∑ u in range 10, if valid_hundreds_digit h ∧ valid_units_and_tens_digit t u then 1 else 0 :=
begin
  sorry
end

end three_digit_numbers_with_units_at_least_three_times_tens_l389_389400


namespace max_price_per_notebook_l389_389652

theorem max_price_per_notebook (n B S : ℕ) (T : ℝ) (h_n : n = 18) (h_B : B = 160) (h_S : S = 5) (h_T : T = 0.07) :
  let available_funds := B - S,
      funds_after_tax := available_funds / (1 + T),
      max_price := funds_after_tax / n 
  in floor max_price = 8 :=
by {
  -- Here we calculate the available_funds, funds_after_tax, and max_price based on the given conditions
  let available_funds := B - S,
  have h_available_funds : available_funds = 155 := by simp [h_B, h_S],
  
  let funds_after_tax := available_funds / (1 + T),
  have h_funds_after_tax : funds_after_tax = 155 / 1.07 := by simp [h_available_funds, h_T],
  
  let max_price := funds_after_tax / n,
  have h_max_price : max_price = (155 / 1.07) / 18 := by simp [h_funds_after_tax, h_n],
  
  -- Assert the final value of max_price approximates to 8 with integer rounding to the nearest whole number.
  -- Using floor function to get the greatest integer less than or equal to the computed max_price.
  have h_floor_max_price : floor ((155 / 1.07) / 18) = 8 := 
    by { exact sorry } -- Placeholder for further numeric simplification steps
}

end max_price_per_notebook_l389_389652


namespace orient_cameras_to_cover_plane_l389_389060

-- Definitions from the conditions
def n (n : ℕ) : Prop := n ≥ 1
def S (n : ℕ) : set (ℝ × ℝ) := { p | True } -- Placeholders, we'll just consider it as a set of n points
def θ (n : ℕ) : ℝ := 2 * real.pi / n
def camera θ (pos : ℝ × ℝ) : set ((ℝ × ℝ) → Prop) := { orient | True } -- Dummy definition for illustration

-- The proof problem
theorem orient_cameras_to_cover_plane (n : ℕ) (hn : n ≥ 1) (S : set (ℝ × ℝ)) (hS : ∃ (S_size : S.size = n))
  (θ := 2 * real.pi / n)
  (hcam : ∀ p ∈ S, ∃ orient, camera θ p orient)
: ∃ orientations : (ℝ × ℝ) → (ℝ × ℝ) → Prop, (∀ q : ℝ × ℝ, ∃ p ∈ S, orientations p q) :=
sorry

end orient_cameras_to_cover_plane_l389_389060


namespace acute_angle_APB_rhombus_ABPQ_l389_389336

variables {A B P Q : ℝ × ℝ} (λ : ℝ)
def A := (1,0)
def B := (0,-1)
def P := (λ, λ + 1)
def AB := (1, 1)
def BA := (-1, -1)
// PA and PB vectors
def PA := (1 - λ, - λ - 1)
def PB := (- λ, -2 - λ)

-- Proof that angle APB is always acute
theorem acute_angle_APB : PA λ • PB λ > 0 → ¬ collinear ℝ {A, P λ, B} → acute_angle A (P λ) B := 
by sorry

-- Proof that BQ • AQ = 2 when ABPQ is a rhombus
theorem rhombus_ABPQ (h : ∀ λ, AB • PB λ = λ^2 + (λ + 2)^2) : 
(\| λ \| ≠ 0 → B = (0, 1) ∧ Q = (λ*1, λ * (λ + 2 + 1)) → (λ ≠ -1 → λ^2 + 2λ + 1 = 0 → λ = -1) → 
(∃ Q, Q = (0, 1) → \| BQ \| * \| AQ \| = 2 ) := 
by sorry

end acute_angle_APB_rhombus_ABPQ_l389_389336


namespace computer_cost_l389_389481

theorem computer_cost (C : ℝ) (h1 : 0.10 * C = a) (h2 : 3 * C = b) (h3 : b - 1.10 * C = 2700) : 
  C = 2700 / 2.90 :=
by
  sorry

end computer_cost_l389_389481


namespace cubic_polynomial_coefficients_l389_389859

theorem cubic_polynomial_coefficients (f g : Polynomial ℂ) (b c d : ℂ) :
  f = Polynomial.C 4 + Polynomial.X * (Polynomial.C 3 + Polynomial.X * (Polynomial.C 2 + Polynomial.X)) →
  (∀ x, Polynomial.eval x f = 0 → Polynomial.eval (x^2) g = 0) →
  g = Polynomial.C d + Polynomial.X * (Polynomial.C c + Polynomial.X * (Polynomial.C b + Polynomial.X)) →
  (b, c, d) = (4, -15, -32) :=
by
  intro h1 h2 h3
  sorry

end cubic_polynomial_coefficients_l389_389859


namespace binomial_coefficient_square_sum_l389_389089
  
theorem binomial_coefficient_square_sum (n : ℕ) : 
  (Finset.range (n + 1)).sum (λ k, (Nat.choose n k) ^ 2) = Nat.choose (2 * n) n := 
sorry

end binomial_coefficient_square_sum_l389_389089


namespace count_different_numerators_l389_389830

def hasRepeatingDecimalForm (r : ℚ) : Prop :=
  ∃ a b : ℕ, 0 ≤ a ∧ a < 10 ∧ 0 ≤ b ∧ b < 10 ∧
    r = a * (1 / 100 + 1 / 100^2 + 1 / 100^3 + ...) + b * (1 / 10 + 1 / 10^3 + 1 / 10^5 + ...)

def setT : set ℚ := {r | r > 0 ∧ r < 1 ∧ hasRepeatingDecimalForm r}

theorem count_different_numerators : ∃ n : ℕ, n = 60 ∧
  ∀ r ∈ setT, r = (numerator r) / (99 * gcd (numerator r) 99) :=
sorry

end count_different_numerators_l389_389830


namespace cost_efficiency_ranking_l389_389187

noncomputable def small_pack_cost : ℝ := 1
noncomputable def medium_pack_cost : ℝ := small_pack_cost * 1.25
noncomputable def large_pack_cost : ℝ := medium_pack_cost * 1.4

noncomputable def small_pack_wipes : ℝ := 10
noncomputable def medium_pack_wipes : ℝ := small_pack_wipes * 1.5
noncomputable def large_pack_wipes : ℝ := small_pack_wipes * 2.5

noncomputable def small_pack_cost_per_wipe : ℝ := small_pack_cost / small_pack_wipes
noncomputable def medium_pack_cost_per_wipe : ℝ := medium_pack_cost / medium_pack_wipes
noncomputable def large_pack_cost_per_wipe : ℝ := large_pack_cost / large_pack_wipes

theorem cost_efficiency_ranking : [large_pack_cost_per_wipe, medium_pack_cost_per_wipe, small_pack_cost_per_wipe] = ([
    (large_pack_cost / large_pack_wipes),
    (medium_pack_cost / medium_pack_wipes),
    (small_pack_cost / small_pack_wipes)])
by
  simp [large_pack_cost_per_wipe, medium_pack_cost_per_wipe, small_pack_cost_per_wipe]
  sorry

end cost_efficiency_ranking_l389_389187


namespace angle_bisectors_result_l389_389474

variables {A B C O A' B' : Type} [metric_space A] [metric_space B] [metric_space C]
variables [metric_space O] [metric_space A'] [metric_space B']

-- Conditions
variables (ABC : Prop) (AngleBisectorsIntersect : Prop) 
variables (A'_on_CB : Prop) (B'_on_CA : Prop)
variables (H1 : ABC = (∠C = 90))
variables (H2 : AngleBisectorsIntersect = (O.is_intersection_point_of_angle_bisectors A B))
variables (H3 : A'_on_CB = (A'.is_on_segment C B))
variables (H4 : B'_on_CA = (B'.is_on_segment C A))

theorem angle_bisectors_result
  (H1 : ABC) (H2 : AngleBisectorsIntersect) (H3 : A'_on_CB) (H4 : B'_on_CA) :
  (distance A A') * (distance B B') = 2 * (distance A O) * (distance B O) :=
  sorry

end angle_bisectors_result_l389_389474


namespace final_S_is_correct_l389_389207

/-- Define a function to compute the final value of S --/
def final_value_of_S : ℕ :=
  let S := 0
  let I_values := List.range' 1 27 3 -- generate list [1, 4, 7, ..., 28]
  I_values.foldl (fun S I => S + I) 0  -- compute the sum of the list

/-- Theorem stating the final value of S is 145 --/
theorem final_S_is_correct : final_value_of_S = 145 := by
  sorry

end final_S_is_correct_l389_389207


namespace TeresaTotalMarks_l389_389546

/-- Teresa's scores in various subjects as given conditions -/
def ScienceScore := 70
def MusicScore := 80
def SocialStudiesScore := 85
def PhysicsScore := 1 / 2 * MusicScore

/-- Total marks Teresa scored in all the subjects -/
def TotalMarks := ScienceScore + MusicScore + SocialStudiesScore + PhysicsScore

/-- Proof statement: The total marks scored by Teresa in all subjects is 275. -/
theorem TeresaTotalMarks : TotalMarks = 275 := by
  sorry

end TeresaTotalMarks_l389_389546


namespace proof_x_y_3_l389_389819

noncomputable def prime (n : ℤ) : Prop := 2 <= n ∧ ∀ m : ℤ, 1 ≤ m → m < n → n % m ≠ 0

theorem proof_x_y_3 (x y : ℝ) (p q r : ℤ) (h1 : x - y = p) (hp : prime p) 
  (h2 : x^2 - y^2 = q) (hq : prime q)
  (h3 : x^3 - y^3 = r) (hr : prime r) : p = 3 :=
sorry

end proof_x_y_3_l389_389819


namespace different_numerators_count_l389_389823

noncomputable def isRepeatingDecimalInTheForm_ab (r : ℚ) : Prop :=
∃ (a b : ℕ), 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 
(∃ (k : ℕ), 10 + k = 100 * r ∧ (100 * r - 10) = k ∧ (k < 100) ∧ (100 * k) % 99 = 0)

def T : set ℚ := { r : ℚ | 0 < r ∧ r < 1 ∧ isRepeatingDecimalInTheForm_ab r }

theorem different_numerators_count : 
  ∃ n, n = 63 ∧ ∀ r ∈ T, ((r.denom) % 99 ≠ 0 → n = 63) :=
sorry

end different_numerators_count_l389_389823


namespace interest_rate_for_loan_l389_389196

def calculate_interest (principal : ℕ) (rate : ℚ) (time : ℚ) : ℚ :=
  principal * rate * time

theorem interest_rate_for_loan (interest_total : ℚ) 
  (principal1 : ℕ) (rate1 : ℚ) (time1 : ℚ) 
  (principal2 : ℕ) (time2 : ℚ) (rate2 : ℚ) : 
  let interest1 := calculate_interest principal1 rate1 time1 in
  let interest2 := calculate_interest principal2 rate2 time2 in
  interest1 + interest2 = interest_total → 
  rate2 * 100 = 5 := 
sorry

# Example usage where Rs. is omitted for simplicity
# interest_total = 390
# principal1 = 1000
# rate1 = 0.03
# time1 = 3.9
# principal2 = 1400
# time2 = 3.9
# rate2 = 0.05
example : interest_rate_for_loan 390 1000 0.03 3.9 1400 3.9 0.05 :=
  sorry

end interest_rate_for_loan_l389_389196


namespace quadruple_solution_count_l389_389903

theorem quadruple_solution_count :
  { (x, y, z, w) : ℝ × ℝ × ℝ × ℝ |
      x^3 + 2 = 3 * y ∧
      y^3 + 2 = 3 * z ∧
      z^3 + 2 = 3 * w ∧
      w^3 + 2 = 3 * x }.finite ∧
  ({ (x, y, z, w) : ℝ × ℝ × ℝ × ℝ |
      x^3 + 2 = 3 * y ∧
      y^3 + 2 = 3 * z ∧
      z^3 + 2 = 3 * w ∧
      w^3 + 2 = 3 * x }.count ≠ 0) :=
sorry

end quadruple_solution_count_l389_389903


namespace sum_totient_eq_n_l389_389535

open Nat

theorem sum_totient_eq_n (n : ℕ) (h : n > 0) :
  ∑ d in divisors n, φ d = n :=
sorry

end sum_totient_eq_n_l389_389535


namespace total_units_per_day_all_work_together_l389_389918

-- Conditions
def men := 250
def women := 150
def units_per_day_by_men := 15
def units_per_day_by_women := 3

-- Problem statement and proof
theorem total_units_per_day_all_work_together :
  units_per_day_by_men + units_per_day_by_women = 18 :=
sorry

end total_units_per_day_all_work_together_l389_389918


namespace garage_pump_time_l389_389619

-- Defining the conditions and required calculations formally in Lean

noncomputable def depth_in_feet (depth_in_inches : ℕ) : ℝ := depth_in_inches / 12
noncomputable def volume_in_cubic_feet (length width height : ℝ) : ℝ := length * width * height
noncomputable def volume_in_gallons (volume_ft3 : ℝ) (gallons_per_ft3 : ℝ) : ℝ := volume_ft3 * gallons_per_ft3
noncomputable def pumping_rate (gallons_per_minute_per_pump : ℝ) (number_of_pumps : ℕ) : ℝ := gallons_per_minute_per_pump * number_of_pumps
noncomputable def time_to_pump_out_water (total_gallons : ℝ) (total_pumping_rate : ℝ) : ℝ := total_gallons / total_pumping_rate

-- Stating the final proof problem
theorem garage_pump_time :
  let length := 20
  let width := 40
  let depth := 24
  let gallons_per_ft3 := 7.5
  let pumps := 4
  let gallons_per_minute_per_pump := 10
  depth_in_feet depth = 2 ∧
  volume_in_cubic_feet length width (depth_in_feet depth) = 1600 ∧
  volume_in_gallons 1600 gallons_per_ft3 = 12000 ∧
  pumping_rate gallons_per_minute_per_pump pumps = 40 ∧
  time_to_pump_out_water 12000 40 = 300
sorry

end garage_pump_time_l389_389619


namespace sum_b_first_50_l389_389570

-- Definitions for conditions
def S (n : ℕ) : ℕ := n^2 + n + 1 -- Definition for the sum of the first n terms of the sequence {a_n}

def a (n : ℕ) : ℕ :=
if n = 1 then 3
else 2 * n -- General formula for the sequence {a_n} based on given condition

def b (n : ℕ) : ℤ :=
if n = 1 then -3
else (-1 : ℤ)^n * a n -- General formula for the sequence {b_n}

-- The theorem to be proved
theorem sum_b_first_50 : (Finset.sum (Finset.range 50) (λ n, b (n + 1))) = 49 := 
sorry

end sum_b_first_50_l389_389570


namespace find_a_for_even_function_l389_389751

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := e^x + a / e^x

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (g a x) * x^3

theorem find_a_for_even_function (a : ℝ) : 
  (∀ x : ℝ, f a x = f a (-x)) ↔ a = -1 :=
by
  sorry

end find_a_for_even_function_l389_389751


namespace constant_term_expansion_l389_389946

theorem constant_term_expansion (x : ℝ) :
  (constant_term ((5 * x + 2 / (5 * x)) ^ 8) = 1120) := by
  sorry

end constant_term_expansion_l389_389946


namespace contains_zero_l389_389254

open Nat

theorem contains_zero 
  (x y : ℕ)
  (hx : 10000 ≤ x ∧ x < 100000)
  (hy : 10000 ≤ y ∧ y < 100000)
  (h_swap : ∃ (i j : ℕ), i ≠ j ∧ swap_digits x i j = y)
  (h_sum : x + y = 111111) :
  (∃ i, digit x i = 0) ∨ (∃ j, digit y j = 0) :=
sorry

-- Swap two digits in a number
def swap_digits (n : ℕ) (i j : ℕ) : ℕ := 
  -- Dummy definition for the proof placeholder
  n -- replace with actual implementation

-- Extract the digit at position i
def digit (n : ℕ) (i : ℕ) : ℕ :=
  -- Dummy definition for the proof placeholder
  0 -- replace with actual implementation

end contains_zero_l389_389254


namespace sin_double_angle_rotated_l389_389468

theorem sin_double_angle_rotated {
  α : ℝ
  (hcos : Real.cos (α + (π / 6)) = -3 / 5)
  (hsin : Real.sin (α + (π / 6)) = 4 / 5)
} : Real.sin (2 * α - (π / 6)) = 7 / 25 :=
by
  sorry

end sin_double_angle_rotated_l389_389468


namespace cows_in_group_l389_389458

variable (c h : ℕ)

theorem cows_in_group (hcow : 4 * c + 2 * h = 2 * (c + h) + 18) : c = 9 := 
by 
  sorry

end cows_in_group_l389_389458


namespace problem_statement_l389_389848

theorem problem_statement (x y z : ℝ) (h₀ : x ≠ 0) (h₁ : y ≠ 0) (h₂ : z ≠ 0) (h₃ : x + y + z = 0) (h₄ : xy + xz + yz ≠ 0) : 
  (x^7 + y^7 + z^7) / (x * y * z * (x * y + x * z + y * z)) = -7 :=
by
  sorry

end problem_statement_l389_389848


namespace sum_n_k_given_ratio_l389_389114

theorem sum_n_k_given_ratio :
  (∃ n k : ℕ, (nchoosek_ratio (n) (k) = 1 / 3) ∧ (nchoosek_ratio (n) (k + 1) = 3 / 5)) → 
  (∑ n k : ℕ, n + k = 8) := 
by sorry

/-- Utility definitions to avoid opaque details -/
def nchoosek_ratio (n k : ℕ) : ℚ :=
  (nat.choose n k : ℚ) / (nat.choose n (k + 1) : ℚ)

end sum_n_k_given_ratio_l389_389114


namespace cosine_period_l389_389159

theorem cosine_period (k : ℝ) (h : k = 6) : 
  ∀ x : ℝ, ∃ T : ℝ, T = 2 * π / k ∧ ∀ x : ℝ, cos (k * (x + T)) = cos (k * x + π) :=
by
  sorry

end cosine_period_l389_389159


namespace max_sequence_length_l389_389786

theorem max_sequence_length (a : ℕ → ℕ) 
  (h_seq : ∀ n, a (n + 2) = (a (n + 1) - a n).abs) 
  (h_bound : ∀ n, a n ≤ 2022) : 
  ∃ k, k = 3034 ∧ a.length ≤ k :=
by
  sorry

end max_sequence_length_l389_389786


namespace sum_of_squares_inequality_l389_389344

theorem sum_of_squares_inequality (a b c : ℝ) : a^2 + b^2 + c^2 ≥ (1/3)*(a + b + c)^2 := sorry

end sum_of_squares_inequality_l389_389344


namespace houses_with_both_garage_and_pool_l389_389783

-- Define the given values
def G := 50
def P := 40
def N := 15
def TotalHouses := 70

-- Define the condition based on the principle of inclusion-exclusion
def Total := TotalHouses - N
def B := G + P - Total

-- The theorem we need to prove
theorem houses_with_both_garage_and_pool : B = 35 := by
  -- Applying the given values and simplifying the expressions
  unfold G P N TotalHouses Total B
  sorry

end houses_with_both_garage_and_pool_l389_389783


namespace average_speed_for_trip_l389_389045

theorem average_speed_for_trip :
  ∀ (walk_dist bike_dist drive_dist tot_dist walk_speed bike_speed drive_speed : ℝ)
  (h1 : walk_dist = 5) (h2 : bike_dist = 35) (h3 : drive_dist = 80)
  (h4 : tot_dist = 120) (h5 : walk_speed = 5) (h6 : bike_speed = 15)
  (h7 : drive_speed = 120),
  (tot_dist / (walk_dist / walk_speed + bike_dist / bike_speed + drive_dist / drive_speed)) = 30 :=
by
  intros
  sorry

end average_speed_for_trip_l389_389045


namespace digit_contains_zero_l389_389228

theorem digit_contains_zero 
  (n₁ n₂ : ℕ) 
  (h₁ : 10000 ≤ n₁ ∧ n₁ ≤ 99999)
  (h₂ : 10000 ≤ n₂ ∧ n₂ ≤ 99999)
  (h₃ : ∃ (i j : ℕ), i ≠ j ∧ n₂ = n₁.swap_digits i j)
  (h₄ : n₁ + n₂ = 111111) 
  : ∃ (d : ℕ), d = 0 :=
sorry

end digit_contains_zero_l389_389228


namespace reciprocal_of_36_recurring_decimal_l389_389957

-- Definitions and conditions
def recurring_decimal (x : ℚ) : Prop := x = 36 / 99

-- Theorem statement
theorem reciprocal_of_36_recurring_decimal :
  recurring_decimal (36 / 99) → (1 / (36 / 99) = 11 / 4) :=
sorry

end reciprocal_of_36_recurring_decimal_l389_389957


namespace extreme_point_property_l389_389749

variables (f : ℝ → ℝ) (a b x x₀ x₁ : ℝ) 

-- Define the function f
def func (x : ℝ) := x^3 - a * x - b

-- The main theorem
theorem extreme_point_property (h₀ : ∃ x₀, ∃ x₁, (x₀ ≠ 0) ∧ (x₀^2 = a / 3) ∧ (x₁ ≠ x₀) ∧ (func a b x₀ = func a b x₁)) :
  x₁ + 2 * x₀ = 0 :=
sorry

end extreme_point_property_l389_389749


namespace count_valid_three_digit_numbers_l389_389420

-- Define the property of a number being three digits
def isThreeDigit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

-- Define the property of the units digit being at least three times the tens digit
def validUnitsTimesTens (n : ℕ) : Prop :=
  let units := n % 10
  let tens := (n / 10) % 10
  units ≥ 3 * tens

-- Definition combining both conditions
def validThreeDigitNumber (n : ℕ) : Prop :=
  isThreeDigit n ∧ validUnitsTimesTens n

-- The main theorem stating the number of valid three-digit numbers
theorem count_valid_three_digit_numbers :
  Finset.card (Finset.filter validThreeDigitNumber (Finset.range 1000)) = 198 :=
by
  sorry

end count_valid_three_digit_numbers_l389_389420


namespace range_of_m_l389_389119

def f (x : ℝ) : ℝ := x ^ 2 + 2 * x + 3

theorem range_of_m (m : ℝ) (h_max : ∀ x ∈ set.Icc m 0, f x ≤ 3) (h_min : ∀ x ∈ set.Icc m 0, f x ≥ 2) : 
  m ∈ set.Icc (-2) (-1) :=
by 
  sorry

end range_of_m_l389_389119


namespace reciprocal_of_repeating_decimal_l389_389986

theorem reciprocal_of_repeating_decimal :
  let x := (36 : ℚ) / 99 in 1 / x = 11 / 4 :=
by
  trivial -- This is a placeholder, the actual proof is not required and hence replaced by trivial.

end reciprocal_of_repeating_decimal_l389_389986


namespace renee_allergic_probability_l389_389041

theorem renee_allergic_probability :
  let peanut_butter_from_jenny := 40
  let chocolate_chip_from_jenny := 50
  let peanut_butter_from_marcus := 30
  let lemon_from_marcus := 20
  let total_cookies := peanut_butter_from_jenny + chocolate_chip_from_jenny + peanut_butter_from_marcus + lemon_from_marcus
  let total_peanut_butter := peanut_butter_from_jenny + peanut_butter_from_marcus
  let p := (total_peanut_butter : ℝ) / (total_cookies : ℝ) * 100
  in p = 50 := by sorry

end renee_allergic_probability_l389_389041


namespace contains_zero_l389_389246

-- Define a five-digit number in terms of its digits.
def five_digit_number (A B C D E : ℕ) : ℕ :=
  10^4 * A + 10^3 * B + 10^2 * C + 10 * D + E

-- Define the conditions:
noncomputable def switch_two_digits (A B C D E F : ℕ) : Prop :=
  (10^4 * A + 10^3 * B + 10^2 * C + 10 * D + E ≠ 10^4 * A + 10^3 * B + 10^2 * F + 10 * D + E)

-- Lean statement for the proof problem.
theorem contains_zero (A B C D E F : ℕ) :
  let ABCDE := five_digit_number A B C D E,
      ABFDE := five_digit_number A B F D E in
  switch_two_digits A B C D E F ∧ (ABCDE + ABFDE = 111111) → (A = 0 ∨ B = 0 ∨ C = 0 ∨ D = 0 ∨ E = 0) :=
by
  -- This is where the proof would go.
  sorry

end contains_zero_l389_389246


namespace partition_inequality_l389_389491

variable (n : ℕ)
variable (f : ℕ → ℕ)
variable (P : Type) [is_square P]

-- Assume n is a nonzero positive integer
-- Assume f(n) is the maximum number of elements in a partition of P into rectangles
-- such that each line parallel to some side of P intersects at most n interiors
axiom max_partition_rectangles (h : n > 0) (H : ∃ P : Type, is_square P) :
  3 * 2 ^ (n - 1) - 2 ≤ f(n) ∧ f(n) ≤ 3 ^ n - 2

-- The Lean statement we want to prove
theorem partition_inequality : 3 * 2 ^ (n - 1) - 2 ≤ f(n) ∧ f(n) ≤ 3 ^ n - 2 :=
  max_partition_rectangles sorry sorry

end partition_inequality_l389_389491


namespace multiple_of_kids_finishing_early_l389_389179

-- Definitions based on conditions
def num_10_percent_kids (total_kids : ℕ) : ℕ := (total_kids * 10) / 100

def num_remaining_kids (total_kids kids_less_6 kids_more_14 : ℕ) : ℕ := total_kids - kids_less_6 - kids_more_14

def num_multiple_finishing_less_8 (total_kids : ℕ) (multiple : ℕ) : ℕ := multiple * num_10_percent_kids total_kids

-- Main theorem statement
theorem multiple_of_kids_finishing_early 
  (total_kids : ℕ)
  (h_total_kids : total_kids = 40)
  (kids_more_14 : ℕ)
  (h_kids_more_14 : kids_more_14 = 4)
  (h_1_6_remaining : kids_more_14 = num_remaining_kids total_kids (num_10_percent_kids total_kids) kids_more_14 / 6)
  : (num_multiple_finishing_less_8 total_kids 3) = (total_kids - num_10_percent_kids total_kids - kids_more_14) := 
by 
  sorry

end multiple_of_kids_finishing_early_l389_389179


namespace paint_area_correct_l389_389182

-- Define the dimensions
def width := 11
def length := 14
def height := 6

-- Calculate individual areas
def area_walls_first_pair := 2 * (width * height)
def area_walls_second_pair := 2 * (length * height)
def area_ceiling := width * length

-- Sum the total area with the conditions specified
def total_area_to_be_painted := 2 * (area_walls_first_pair + area_walls_second_pair) + area_ceiling

-- The proof statement
theorem paint_area_correct : total_area_to_be_painted = 654 :=
by
  sorry

end paint_area_correct_l389_389182


namespace find_sets_of_odd_numbers_l389_389140

-- Define the main variables and conditions
def four_consecutive_odd_numbers (a : ℤ) : set ℤ :=
  {a - 3, a - 1, a + 1, a + 3}

def is_cube_of_single_digit (n : ℤ) : Prop :=
  (n ∈ {2, 4, 6, 8}) ∧ (∃ a, 4 * a = n ^ 3)

def valid_odd_sets (s : set ℤ) : Prop :=
  s = {13, 15, 17, 19} ∨ s = {51, 53, 55, 57} ∨ s = {125, 127, 129, 131}

theorem find_sets_of_odd_numbers :
  ∃ (a n : ℤ), is_cube_of_single_digit n ∧ four_consecutive_odd_numbers a = {13, 15, 17, 19} ∨ 
                four_consecutive_odd_numbers a = {51, 53, 55, 57} ∨ 
                four_consecutive_odd_numbers a = {125, 127, 129, 131} :=
by
  sorry

end find_sets_of_odd_numbers_l389_389140


namespace domain_of_function_l389_389270

def domain_of_f (x: ℝ) : Prop :=
x >= -1 ∧ x <= 48

theorem domain_of_function :
  ∀ x, (x + 1 >= 0 ∧ 7 - Real.sqrt (x + 1) >= 0 ∧ 4 - Real.sqrt (7 - Real.sqrt (x + 1)) >= 0)
  ↔ domain_of_f x := by
  sorry

end domain_of_function_l389_389270


namespace max_intersections_circle_cos_curve_correct_answer_l389_389683

open Real

-- Definitions of circle and curve
def circle_eq (x y : ℝ) := (x - 1)^2 + (y - 2)^2 = 4
def cos_curve (x y : ℝ) := y = cos x

-- Main Statement: Proof that the maximum number of intersection points within the interval [0, 2π] is 4
theorem max_intersections_circle_cos_curve :
  ∃ (P : ℝ → ℝ → Prop), (P = circle_eq) ∧ 
    ∃ (Q : ℝ → ℝ → Prop), (Q = cos_curve) ∧ 
    ∀ (a b c d : ℝ), 
      (0 ≤ a ∧ a ≤ 2 * π) ∧ 
      (0 ≤ b ∧ b ≤ 2 * π) ∧ 
      (0 ≤ c ∧ c ≤ 2 * π) ∧ 
      (0 ≤ d ∧ d ≤ 2 * π) →
      (P a (cos a) ∧ P b (cos b) ∧ P c (cos c) ∧ P d (cos d)) →
      (cos_curve a (cos a) ∧ cos_curve b (cos b) ∧ cos_curve c (cos c) ∧ cos_curve d (cos d)) →
      a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem correct_answer : max_intersections_circle_cos_curve = 4 := sorry

end max_intersections_circle_cos_curve_correct_answer_l389_389683


namespace inequality_solution_set_l389_389346

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def f' (x : ℝ) : ℝ := sorry

axiom deriv_cond : ∀ (x : ℝ), x ≠ 0 → f' x < (2 * f x) / x
axiom zero_points : f (-2) = 0 ∧ f 1 = 0

theorem inequality_solution_set :
  {x : ℝ | x * f x < 0} = { x : ℝ | (-2 ≤ x ∧ x < 0) ∨ (0 < x ∧ x ≤ 1) } :=
sorry

end inequality_solution_set_l389_389346


namespace midpoint_proof_l389_389022

noncomputable def quadrilateral_midpoint (A B C D E F G : Type*) [add_comm_group A] [module ℝ A] :=
  ∃ (AB DC AD BC : ℝ),
  (AB > 0) ∧ (DC > 0) ∧ (AD > 0) ∧ (BC > 0) ∧
  (∃ (E F : A), 
    (A - B) ∥ (D - C) ∧ (A - D) ∥ (B - C) ∧ 
    (BD - EF) ∥ 0 ∧
    (G ∈ line (AC) ∧ G ∈ line (EF)) ∧
    EG = GF)

theorem midpoint_proof (A B C D E F G : Type*) [add_comm_group A] [module ℝ A] :
  quadrilateral_midpoint A B C D E F G → (EG = GF) :=
sorry

end midpoint_proof_l389_389022


namespace no_valid_choice_l389_389688

-- Definitions based on our conditions
def is_sum_of_consecutive_150 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 150 * k + 11175

-- List of given integers to check
def given_integers : List ℕ := [1625999850, 2344293800, 3578726150, 4691196050, 5815552000]

-- The main theorem statement: None of the given integers can be expressed as the sum of 150 consecutive positive integers.
theorem no_valid_choice : ∀ n ∈ given_integers, ¬ is_sum_of_consecutive_150 n :=
by
  intro n h
  cases h with
  | head 1625999850_tail =>
    intro h
    obtain ⟨k, hk⟩ := h
    sorry -- Proof to be completed

  | head 2344293800_tail =>
    intro h
    obtain ⟨k, hk⟩ := h
    sorry -- Proof to be completed

  | head 3578726150_tail =>
    intro h
    obtain ⟨k, hk⟩ := h
    sorry -- Proof to be completed

  | head 4691196050_tail =>
    intro h
    obtain ⟨k, hk⟩ := h
    sorry -- Proof to be completed

  | head 5815552000_tail =>
    intro h
    obtain ⟨k, hk⟩ := h
    sorry -- Proof to be completed

end no_valid_choice_l389_389688


namespace hyperbola_asymptote_eq_l389_389368

-- Define the given hyperbola equation and its asymptote
def hyperbola_eq (a : ℝ) : Prop :=
  ∃ x y : ℝ, (x^2 / a^2) - (y^2 / 4) = 1

def asymptote_eq (a : ℝ) : Prop :=
  ∀ x : ℝ, ∃ y : ℝ, y = (1/2) * x

-- State the main theorem
theorem hyperbola_asymptote_eq :
  (∃ a : ℝ, hyperbola_eq a ∧ asymptote_eq a) →
  (∃ x y : ℝ, (x^2 / 16) - (y^2 / 4) = 1) := 
by
  sorry

end hyperbola_asymptote_eq_l389_389368


namespace problem_divisibility_by_3_l389_389525

/-- Positive integers a, b, and c are randomly and independently selected with replacement 
    from the set {1, 2, 3, ..., 2011}. Prove that the probability that abc + ab + a is divisible 
    by 3 is (671 * 27 + 5360) / 54297. -/
theorem problem_divisibility_by_3:
  ∀ (a b c : ℕ), 
  (a ∈ Finset.range 2012) →
  (b ∈ Finset.range 2012) →
  (c ∈ Finset.range 2012) →
  let prob_zero_mod_3 := 671 / 2011
  let prob_a_non_zero := (2 / 3) * (2 / 9)
  let total_prob := prob_zero_mod_3 + (5360 / 54297)
  total_prob = (671 * 27 + 5360) / 54297 :=
begin
  sorry
end

end problem_divisibility_by_3_l389_389525


namespace curvilinear_triangle_area_difference_l389_389872

variable (A B C : Type) [MetricSpace A]
variable [HasDist A B] [HasDist B C] [HasDist C A]
variable [AcuteAngledTriangle A B C]
variable (x y z u : ℝ)

-- Definition of Area using given sides of triangle ABC
noncomputable def area_of_triangle (A B C : Type) [MetricSpace A] := sorry

-- We assume the area of the external curvilinear triangles and the internal curvilinear triangle are given
variable (external_area1 external_area2 external_area3 internal_area : ℝ)

-- Given areas for the problem statement
axiom h1 : external_area1 = x
axiom h2 : external_area2 = y
axiom h3 : external_area3 = z
axiom h4 : internal_area = u

-- Problem statement proof
theorem curvilinear_triangle_area_difference (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
  [AcuteAngledTriangle A B C] (x y z u : ℝ)
  (external_area1 external_area2 external_area3 internal_area: ℝ)
  (h1 : external_area1 = x)
  (h2 : external_area2 = y)
  (h3 : external_area3 = z)
  (h4 : internal_area = u) :
  (x + y + z) - u = 2 * (area_of_triangle A B C) := by
  sorry

end curvilinear_triangle_area_difference_l389_389872


namespace team_total_games_l389_389635

theorem team_total_games (R : ℕ) (G : ℕ)
  (h1 : G = 30 + R)
  (h2 : 0.5 * G = 12 + 0.8 * R) :
  G = 40 :=
by
  sorry

end team_total_games_l389_389635


namespace square_area_example_l389_389134

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

noncomputable def square_area (x1 y1 x2 y2 : ℝ) : ℝ :=
  (distance x1 y1 x2 y2)^2

theorem square_area_example : square_area 1 3 5 6 = 25 :=
by
  sorry

end square_area_example_l389_389134


namespace five_b_value_l389_389767

theorem five_b_value (a b : ℚ) (h1 : 3 * a + 4 * b = 2) (h2 : a = 2 * b - 3) : 5 * b = 5.5 := 
by
  sorry

end five_b_value_l389_389767


namespace prob_courses_chosen_different_l389_389192

def total_combinations (choices : ℕ) : ℕ := choices * choices
def different_combinations (choices : ℕ) : ℕ := choices * (choices - 1)
def prob_different (choices : ℕ) : ℚ := different_combinations choices / total_combinations choices

theorem prob_courses_chosen_different (choices : ℕ) (h_choices : choices = 4) :
  prob_different choices = 3 / 4 :=
by
  have h_total : total_combinations choices = 16 := by
    rw [h_choices]
    norm_num
  have h_diff : different_combinations choices = 12 := by
    rw [h_choices]
    norm_num
  rw [prob_different, h_total, h_diff]
  norm_num
  sorry

end prob_courses_chosen_different_l389_389192


namespace find_a_and_b_l389_389363

noncomputable def f (a b : ℝ) (x : ℝ) := (a * Real.log x) / (x + 1) + b / x

theorem find_a_and_b (a b : ℝ) (h_tangent: ∀ x y : ℝ, (x + 2 * y = 3) → y = f a b x)
  (h_f1 : f a b 1 = 1)
  (h_f1_deriv : (∂ (λ x, f a b x) / ∂ x) 1 = -1/2)
  : a = 1 ∧ b = 1 := 
sorry

end find_a_and_b_l389_389363


namespace mod7_remainder_problem_l389_389507

theorem mod7_remainder_problem (a b c : ℤ) (h1 : 0 ≤ a ∧ a < 7) (h2 : 0 ≤ b ∧ b < 7) (h3 : 0 ≤ c ∧ c < 7)
    (hc1 : a + 3 * b + 2 * c ≡ 2 [ZMOD 7])
    (hc2 : 2 * a + b + 3 * c ≡ 3 [ZMOD 7])
    (hc3 : 3 * a + 2 * b + c ≡ 5 [ZMOD 7]) : a * b * c ≡ 1 [ZMOD 7] := by
  sorry

end mod7_remainder_problem_l389_389507


namespace concyclic_S_P_Q_R_l389_389506

variables {A B C D H P Q R S : Type*}
variables [ordered_geometry A B C D] -- assuming some ordered geometry structure to define relationships.

-- Assume A, B, C, D form a parallelogram
variables (AB CD : A ≃ B) (BC AD : B ≃ C) (CD AD : C ≃ D)

-- H is the orthocenter of ∆ABC
variables (orthocenter H A B C)

-- Lines parallel to AB through H and its intersections
variables (parallel1 : parallel (line A B) (line H P))
variables (intersect1 : intersects (line H P) (line B C) P)
variables (intersect2 : intersects (line H P) (line A D) Q)

-- Lines parallel to BC through H and its intersections
variables (parallel2 : parallel (line B C) (line H R))
variables (intersect3 : intersects (line H R) (line A B) R)
variables (intersect4 : intersects (line H R) (line C D) S)

-- The goal
theorem concyclic_S_P_Q_R :
  concyclic S P Q R :=
sorry

end concyclic_S_P_Q_R_l389_389506


namespace tiles_needed_l389_389036

def hallway_length : ℕ := 14
def hallway_width : ℕ := 20
def border_tile_side : ℕ := 2
def interior_tile_side : ℕ := 3

theorem tiles_needed :
  let border_length_tiles := ((hallway_length - 2 * border_tile_side) / border_tile_side) * 2
  let border_width_tiles := ((hallway_width - 2 * border_tile_side) / border_tile_side) * 2
  let corner_tiles := 4
  let total_border_tiles := border_length_tiles + border_width_tiles + corner_tiles
  let interior_length := hallway_length - 2 * border_tile_side
  let interior_width := hallway_width - 2 * border_tile_side
  let interior_area := interior_length * interior_width
  let interior_tiles_needed := (interior_area + interior_tile_side * interior_tile_side - 1) / (interior_tile_side * interior_tile_side)
  total_border_tiles + interior_tiles_needed = 48 := 
by {
  sorry
}

end tiles_needed_l389_389036


namespace claudia_can_fill_glasses_l389_389265

theorem claudia_can_fill_glasses : 
  ∀ (initial_water ounces_per_glass_five ounces_per_glass_eight ounces_per_glass_four num_glasses_five num_glasses_eight) 
  (remaining_water : ℕ),
  initial_water = 122 →
  ounces_per_glass_five = 5 →
  ounces_per_glass_eight = 8 →
  ounces_per_glass_four = 4 →
  num_glasses_five = 6 →
  num_glasses_eight = 4 →
  remaining_water = initial_water - (num_glasses_five * ounces_per_glass_five) - (num_glasses_eight * ounces_per_glass_eight) →
  remaining_water / ounces_per_glass_four = 15 :=
by
  assume initial_water ounces_per_glass_five ounces_per_glass_eight ounces_per_glass_four num_glasses_five num_glasses_eight
  assume remaining_water :
  initial_water = 122 →
  ounces_per_glass_five = 5 →
  ounces_per_glass_eight = 8 →
  ounces_per_glass_four = 4 →
  num_glasses_five = 6 →
  num_glasses_eight = 4 →
  remaining_water = initial_water - (num_glasses_five * ounces_per_glass_five) - (num_glasses_eight * ounces_per_glass_eight) →
  remaining_water / ounces_per_glass_four = 15 := 
sorry

end claudia_can_fill_glasses_l389_389265


namespace airplane_seat_count_l389_389213

theorem airplane_seat_count (F C T : ℕ) (hF : F = 77) (hC : C = 4 * F + 2) (hT : T = F + C) : T = 387 :=
by
  rw [hF, hC, hT]
  sorry

end airplane_seat_count_l389_389213


namespace dot_product_evaluate_n_l389_389279

theorem dot_product (a b c p q r n : ℝ) (h : (3, 4, 5) • (p, q, r) = n) :
    n = 3 * p + 4 * q + 5 * r :=
sorry

theorem evaluate_n (y n : ℝ) (h : (3, 4, 5) • (y, -2, 1) = 3 * y - 3) : 
    n = 34.5 :=
by
  have h1 : n = 3 * 12.5 - 3 := by sorry
  exact h1

end

end dot_product_evaluate_n_l389_389279


namespace median_pull_ups_l389_389620

-- Defining the proof problem with conditions
theorem median_pull_ups (student_counts : List ℕ) (h1 : student_counts = [2, 3, 2, 2, 1]) :
  median (expand_list student_counts) = 5.5 :=
by
  sorry

-- Helper function to generate the list of pull-up counts from the grouped counts
def expand_list (counts : List ℕ) : List ℕ :=
  [4, 4] ++ [5, 5, 5] ++ [6, 6] ++ [7, 7] ++ [8]

-- Function to compute the median of a list
noncomputable def median (l : List ℕ) : ℝ :=
  let sorted_list (irr : l ≠ []) := list.sort (<=) l
  if h : l.length % 2 = 1 then
    sorted_list h).nth_le (l.length / 2) (by simp [h])
  else
    let mid := l.length / 2
    (sorted_list h).nth_le (mid - 1) (by simp [h])
    +
    (sorted_list h).nth_le mid (by simp [h])
    / 2

end median_pull_ups_l389_389620


namespace number_of_integers_not_in_lowest_terms_l389_389706

theorem number_of_integers_not_in_lowest_terms :
  {N : ℕ | 1 ≤ N ∧ N ≤ 2000 ∧ ¬ is_coprime (N ^ 2 + 11) (N + 5)}.card = 55 :=
sorry

end number_of_integers_not_in_lowest_terms_l389_389706


namespace part1_solution_part2_solution_l389_389661

theorem part1_solution (x : ℝ) (h1 : (2 * x) / (x - 2) + 3 / (2 - x) = 1) : x = 1 := by
  sorry

theorem part2_solution (x : ℝ) 
  (h1 : 2 * x - 1 ≥ 3 * (x - 1)) 
  (h2 : (5 - x) / 2 < x + 3) : -1 / 3 < x ∧ x ≤ 2 := by
  sorry

end part1_solution_part2_solution_l389_389661


namespace find_vector_c_l389_389310

def angle_equal_coordinates (c : ℝ × ℝ) : Prop :=
  let a : ℝ × ℝ := (1, 0)
  let b : ℝ × ℝ := (1, -Real.sqrt 3)
  let cos_angle_ab (u v : ℝ × ℝ) : ℝ :=
    (u.1 * v.1 + u.2 * v.2) / (Real.sqrt (u.1^2 + u.2^2) * Real.sqrt (v.1^2 + v.2^2))
  cos_angle_ab c a = cos_angle_ab c b

theorem find_vector_c :
  angle_equal_coordinates (Real.sqrt 3, -1) :=
sorry

end find_vector_c_l389_389310


namespace range_of_a_one_zero_a_eq_1_l389_389365

-- Part (1)
def fx (a x : ℝ) : ℝ := a * Real.log(x + 1) - Real.sin x

def is_monotonically_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≥ f y

theorem range_of_a (a : ℝ) :
  is_monotonically_decreasing_on (fx a) (Real.pi / 4) (Real.pi / 2) → a ≤ 0 :=
sorry

-- Part (2)
def f (x : ℝ) : ℝ := Real.log(x + 1) - Real.sin x

theorem one_zero_a_eq_1 :
  ∃! x, x ∈ (Real.pi / 2, +∞) ∧ f x = 0 :=
sorry

end range_of_a_one_zero_a_eq_1_l389_389365


namespace problem_solution_l389_389495

def f (n : ℕ) : ℕ :=
if n % 2 = 1 then n^2 + 1 else n / 2

theorem problem_solution :
  (finset.filter (λ n, ∃ k, (f^[k]) n = 1) (finset.range 100).succ_card = 7 :=
by
  sorry

end problem_solution_l389_389495


namespace problem_g_divisors_1005_l389_389707

noncomputable def g (n : ℕ) : ℕ :=
if n = 0 then 1 else 2^n

theorem problem_g_divisors_1005 :
  let k := g 1005 in
  ∃ d : Finset ℕ, (∀ x ∈ d, x ∣ k) ∧ (∀ x : ℕ, x ∣ k → x ∈ d) ∧ d.card = 1006 :=
by
  let k := g 1005
  use (Finset.range (1006)).image (λ x, 2^x)
  split
  {
    intro x,
    simp only [Finset.mem_image, Finset.mem_range, exists_prop],
    rintro ⟨y, hy, rfl⟩,
    apply nat.dvd_pow,
    apply nat.dvd_refl,
  },
  split
  {
    intros x hx,
    simp only [Finset.mem_image, Finset.mem_range, exists_prop],
    use (nat.log 2 x),
    split
    {
      apply nat.lt_succ_of_le,
      apply nat.log_le_log_of_le,
      apply hx,
      norm_num,
    },
    apply nat.pow_log,
    apply hx,
  },
  simp only [← nat.succ_eq_add_one],
  rw [nat.succ_eq_add_one, nat_range_card']
  sorry

end problem_g_divisors_1005_l389_389707


namespace sum_first_10_terms_l389_389703

variable {a : ℕ → ℝ} -- Declare the arithmetic sequence
variable (d : ℝ)     -- Declare the common difference

-- Conditions
axiom a_pos (n : ℕ) : a n > 0
axiom arithmetic_sequence : ∀ n m, a m = a 1 + (m-1)*d
axiom condition : (a 4)^2 + (a 7)^2 + 2 * (a 4) * (a 7) = 9

-- Theorem to prove
theorem sum_first_10_terms : (∑ i in (finset.range 10).map (λ i => i + 1), a i) = 15 :=
by sorry

end sum_first_10_terms_l389_389703


namespace convert_rectangular_to_cylindrical_example_l389_389273

noncomputable def rectangular_to_cylindrical (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := Real.atan2 y x  -- atan2 handles the quadrants correctly
  (r, θ, z)

theorem convert_rectangular_to_cylindrical_example :
  rectangular_to_cylindrical (-3) 4 5 = (5, Real.atan2 4 (-3), 5) := by
  -- Proof that ensures the conversion logic and expected result
  sorry

end convert_rectangular_to_cylindrical_example_l389_389273


namespace reciprocal_of_repeating_decimal_l389_389971

noncomputable def repeating_decimal_to_fraction (x : ℚ) : ℚ :=
  if x = .\overline{36} then 4/11 else 0

theorem reciprocal_of_repeating_decimal :
  ∀ (x : ℚ), repeating_decimal_to_fraction (.\overline{36}) = 4/11 →
  1 / (repeating_decimal_to_fraction x) = 11/4 :=
by
  intros x hx
  have h : repeating_decimal_to_fraction x = 4/11 := hx
  rw h
  norm_num
  done
  sorry

end reciprocal_of_repeating_decimal_l389_389971


namespace claudia_can_fill_glasses_l389_389266

theorem claudia_can_fill_glasses : 
  ∀ (initial_water ounces_per_glass_five ounces_per_glass_eight ounces_per_glass_four num_glasses_five num_glasses_eight) 
  (remaining_water : ℕ),
  initial_water = 122 →
  ounces_per_glass_five = 5 →
  ounces_per_glass_eight = 8 →
  ounces_per_glass_four = 4 →
  num_glasses_five = 6 →
  num_glasses_eight = 4 →
  remaining_water = initial_water - (num_glasses_five * ounces_per_glass_five) - (num_glasses_eight * ounces_per_glass_eight) →
  remaining_water / ounces_per_glass_four = 15 :=
by
  assume initial_water ounces_per_glass_five ounces_per_glass_eight ounces_per_glass_four num_glasses_five num_glasses_eight
  assume remaining_water :
  initial_water = 122 →
  ounces_per_glass_five = 5 →
  ounces_per_glass_eight = 8 →
  ounces_per_glass_four = 4 →
  num_glasses_five = 6 →
  num_glasses_eight = 4 →
  remaining_water = initial_water - (num_glasses_five * ounces_per_glass_five) - (num_glasses_eight * ounces_per_glass_eight) →
  remaining_water / ounces_per_glass_four = 15 := 
sorry

end claudia_can_fill_glasses_l389_389266


namespace zero_of_f_implies_a_le_bound_l389_389071

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * real.exp(1) * x - real.log x / x

theorem zero_of_f_implies_a_le_bound (a : ℝ) (h : ∃ x > 0, f x + a = 0) :
  a ≤ real.exp(2) + 1 / real.exp(1) :=
by
  sorry

end zero_of_f_implies_a_le_bound_l389_389071


namespace sequence_term_formula_l389_389911

open Real

def sequence_sum_condition (S : ℕ → ℝ) (a : ℕ → ℝ) :=
  ∀ n : ℕ, n > 0 → S n + a n = 4 - 1 / (2 ^ (n - 2))

theorem sequence_term_formula 
  (S : ℕ → ℝ) (a : ℕ → ℝ) 
  (h : sequence_sum_condition S a) :
  ∀ n : ℕ, n > 0 → a n = n / 2 ^ (n - 1) :=
sorry

end sequence_term_formula_l389_389911


namespace contains_zero_l389_389252

open Nat

theorem contains_zero 
  (x y : ℕ)
  (hx : 10000 ≤ x ∧ x < 100000)
  (hy : 10000 ≤ y ∧ y < 100000)
  (h_swap : ∃ (i j : ℕ), i ≠ j ∧ swap_digits x i j = y)
  (h_sum : x + y = 111111) :
  (∃ i, digit x i = 0) ∨ (∃ j, digit y j = 0) :=
sorry

-- Swap two digits in a number
def swap_digits (n : ℕ) (i j : ℕ) : ℕ := 
  -- Dummy definition for the proof placeholder
  n -- replace with actual implementation

-- Extract the digit at position i
def digit (n : ℕ) (i : ℕ) : ℕ :=
  -- Dummy definition for the proof placeholder
  0 -- replace with actual implementation

end contains_zero_l389_389252


namespace right_triangle_sin_C_eq_l389_389465

variables (A B C : Type) [Nonempty A] [Nonempty B] [Nonempty C]

noncomputable def sin_of_triangle (A B C : Type) [metric_space A] [metric_space B] [metric_space C]: ℝ :=
  let AB := 8
  let BC := 17
  let AC := Real.sqrt (BC^2 - AB^2)
  AB / BC

-- Statement of the problem
theorem right_triangle_sin_C_eq (h₁ : ∠ A = 90) (h₂ : AB = 8) (h₃ : BC = 17) :
  sin_of_triangle A B C = 8 / 17 :=
sorry

end right_triangle_sin_C_eq_l389_389465


namespace initial_volume_of_mixture_l389_389185

theorem initial_volume_of_mixture 
  (V : ℝ)
  (h1 : 0 < V) 
  (h2 : 0.20 * V = 0.15 * (V + 5)) :
  V = 15 :=
by 
  -- proof steps 
  sorry

end initial_volume_of_mixture_l389_389185


namespace area_difference_l389_389792

structure RightAngleTriangle (A B C : Type) :=
  (right_angle_at: A = B ∧ B = C)

noncomputable def area   -- defining a noncomputable function for area calculation
  : ℝ × ℝ → ℝ
  | (b, h) := (1 / 2) * b * h

def problem_conditions : Prop :=
  ∃ {α β γ δ ε ζ η : Type}, RightAngleTriangle α β ζ ∧ RightAngleTriangle β δ η ∧
    α = 5 ∧ β = 7 ∧ γ = 10

theorem area_difference:
  problem_conditions →
  let A, B, F, D, C : Type in
  let area_ABF := area(5, 10)
  let area_BAC := area(5, 7)
  let area_ADF : ℝ := 7  -- Placeholder for the actual calculation of triangle ADF
  let area_BDC : ℝ := 7 -- Placeholder for the actual calculation of triangle BDC
  in 
  (area_ADF - area_BDC) = 7.5 :=
by
  intros
  let x := (7:ℝ) -- Placeholder for the actual \( \triangle ADF \) area calculation
  let y := (7:ℝ) -- Placeholder for the actual \( \triangle BDC \) area calculation
  let z := area_ABF
  let t := area_BAC
  have h1 : x + z = 25 := sorry  -- Derived from the problem
  have h2 : y + z = 17.5 := sorry  -- Derived from the problem
  linarith [h1, h2] using [x, y, z, t]  
  sorry

end area_difference_l389_389792


namespace jill_tax_percentage_l389_389599

theorem jill_tax_percentage (total_excl_tax : ℕ)
  (h_clothing : total_excl_tax * 50 / 100)
  (h_food : total_excl_tax * 25 / 100)
  (h_other : total_excl_tax * 25 / 100)
  (tax_clothing : total_excl_tax * 50 / 100 * 10 / 100)
  (tax_food : total_excl_tax * 25 / 100 * 0 / 100)
  (tax_other : total_excl_tax * 25 / 100 * 20 / 100) 
  : (tax_clothing + tax_food + tax_other) / total_excl_tax = 10 / 100 :=
by
  sorry

end jill_tax_percentage_l389_389599


namespace reciprocal_of_repeating_decimal_l389_389984

theorem reciprocal_of_repeating_decimal :
  let x := (36 : ℚ) / 99 in
  x⁻¹ = 11 / 4 :=
by
  have h_simplify : x = 4 / 11 := by sorry
  rw [h_simplify, inv_div]
  norm_num
  exact eq.refl (11 / 4)

end reciprocal_of_repeating_decimal_l389_389984


namespace at_least_one_not_less_than_two_l389_389343

theorem at_least_one_not_less_than_two (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a + 1/b ≥ 2) ∨ (b + 1/c ≥ 2) ∨ (c + 1/a ≥ 2) :=
sorry

end at_least_one_not_less_than_two_l389_389343


namespace relationship_m_n_l389_389509

variable (a b : ℝ)
variable (m n : ℝ)

theorem relationship_m_n (h1 : a > b) (h2 : b > 0) (hm : m = Real.sqrt a - Real.sqrt b) (hn : n = Real.sqrt (a - b)) : m < n := sorry

end relationship_m_n_l389_389509


namespace sqrt_193_interval_l389_389285

theorem sqrt_193_interval :
  13 < Real.sqrt 193 ∧ Real.sqrt 193 < 14 :=
by
  have h1 : 169 < 193 := by linarith
  have h2 : 193 < 196 := by linarith
  have h3 : (13:ℝ)^2 = 169 := by norm_num
  have h4 : (14:ℝ)^2 = 196 := by norm_num
  rw [←Real.sqrt_lt, h3] at h1
  rw [←Real.sqrt_lt, h4] at h2
  norm_num
  exact ⟨h1, h2⟩

end sqrt_193_interval_l389_389285


namespace auditorium_max_students_l389_389215

noncomputable def max_students_seated : ℕ :=
  let seats_in_row (i : ℕ) := 6 + 2 * i
  let max_students_in_row (i : ℕ) := (6 + 2 * i + 1) / 2 -- Math.ceil((6 + 2 * i).toFloat / 2).toInt
  (list.range 25).sum (λ i, max_students_in_row (i + 1))

theorem auditorium_max_students : max_students_seated = 400 :=
  by
    -- Proof of the theorem goes here
    sorry

end auditorium_max_students_l389_389215


namespace intersection_minimum_value_l389_389790

theorem intersection_minimum_value :
  ∀ (α : ℝ), ∃ (t1 t2 : ℝ),
  let x_l := λ t, 1 + t * cos α,
      y_l := λ t, 2 + t * sin α,
      x_c := (λ x y, x^2 + y^2 = 6 * y),
      satisfies_eq := λ t, x_c (x_l t) (y_l t),
      Δ := ((2 * cos α - 2 * sin α)^2 + 4 * 7) > 0 in
  (Δ ∧ satisfies_eq t1 ∧ satisfies_eq t2) ∧ (t1 + t2 = -2 * (cos α - sin α) ∧ t1 * t2 = -7) ∧
  ( ∃ (P : ℝ × ℝ), P = (1, 2) ∧ 
    abs ( ( (x_l t1 - 1)^2 + (y_l t1 - 2)^2 )^0.5  ) + abs ( ( (x_l t2 - 1)^2 + (y_l t2 - 2)^2 )^0.5 ) = 2 * sqrt 7 )
:= sorry

end intersection_minimum_value_l389_389790


namespace count_valid_3_digit_numbers_l389_389379

def valid_units_digit (tens_digit units_digit : ℕ) : Prop :=
  units_digit ≥ 3 * tens_digit

def count_valid_numbers : ℕ :=
  (∑ tens_digit in {0, 1, 2, 3}, (finset.filter (valid_units_digit tens_digit) (finset.range 10)).card) * 9

theorem count_valid_3_digit_numbers : count_valid_numbers = 198 :=
by
  sorry

end count_valid_3_digit_numbers_l389_389379


namespace number_of_zeros_of_g_l389_389746

def f (x : ℝ) : ℝ :=
  if x ≤ 2 then abs (x + 1) else -x^2 + 3 * x

def g (x : ℝ) : ℝ := f (f x - 1)

theorem number_of_zeros_of_g : (∃ x : ℝ, g x = 0) ∧
                              (∃ y : ℝ, y ≠ x ∧ g y = 0) ∧
                              (∃ z : ℝ, z ≠ x ∧ z ≠ y ∧ g z = 0) ∧
                              (¬∃ w : ℝ, w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ g w = 0) :=
by
  sorry

end number_of_zeros_of_g_l389_389746


namespace election_percentage_l389_389018

-- Define the total number of votes (V), winner's votes, and the vote difference
def total_votes (V : ℕ) : Prop := V = 1944 + (1944 - 288)

-- Define the percentage calculation from the problem
def percentage_of_votes (votes_received total_votes : ℕ) : ℕ := (votes_received * 100) / total_votes

-- State the core theorem to prove the winner received 54 percent of the total votes
theorem election_percentage (V : ℕ) (h : total_votes V) : percentage_of_votes 1944 V = 54 := by
  sorry

end election_percentage_l389_389018


namespace fahrenheit_to_celsius_fixed_points_l389_389115

theorem fahrenheit_to_celsius_fixed_points :
  (finset.Icc 30 300).filter
    (λ F, let C := (5 * (F - 32)) / 9 in
          let C' := Int.round C in
          let F' := ceil ((9 * C') / 5 + 32) in
          F = F').card = 30 := sorry

end fahrenheit_to_celsius_fixed_points_l389_389115


namespace reciprocal_of_repeating_decimal_l389_389970

noncomputable def repeating_decimal_to_fraction (x : ℚ) : ℚ :=
  if x = .\overline{36} then 4/11 else 0

theorem reciprocal_of_repeating_decimal :
  ∀ (x : ℚ), repeating_decimal_to_fraction (.\overline{36}) = 4/11 →
  1 / (repeating_decimal_to_fraction x) = 11/4 :=
by
  intros x hx
  have h : repeating_decimal_to_fraction x = 4/11 := hx
  rw h
  norm_num
  done
  sorry

end reciprocal_of_repeating_decimal_l389_389970


namespace S_n_pattern_l389_389838

def S (n : ℕ) : ℕ := ∑ i in finset.range (2*n+1), ∥ real.sqrt (↑(n^2 + i)) ∥

theorem S_n_pattern (n : ℕ) : S n = n * (2 * n + 1) := sorry

lemma S_10 : S 10 = 210 := by
  rw [S_n_pattern]
  norm_num
  sorry

end S_n_pattern_l389_389838


namespace part_a_part_b_l389_389915

theorem part_a (total_matches : ℕ) (first_third_matches_won : ℕ) :
  55 ≤ 100 → 
  (\frac{first_third_matches_won}{total_matches / 3} = 55 / 100) →
  (let remaining_matches_won := total_matches * 85 / 100 in
  (\frac{first_third_matches_won + remaining_matches_won}{total_matches} = 3 / 4)) :=
sorry

theorem part_b (total_matches : ℕ) (first_third_matches_won : ℕ) :
  55 ≤ 100 → 
  (\frac{first_third_matches_won}{total_matches / 3} = 55 / 100) →
  (let remaining_matches_won := total_matches * 2 / 3 in
  (\frac{first_third_matches_won + remaining_matches_won}{total_matches} = 85 / 100)) :=
sorry

end part_a_part_b_l389_389915


namespace cyclic_sum_inequality_l389_389840

theorem cyclic_sum_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x * y / real.sqrt ((x^2 + x * z + z^2) * (y^2 + y * z + z^2))) +
  (y * z / real.sqrt ((y^2 + y * x + x^2) * (z^2 + z * x + x^2))) +
  (z * x / real.sqrt ((z^2 + z * y + y^2) * (x^2 + x * y + y^2))) ≥ 1 :=
sorry

end cyclic_sum_inequality_l389_389840


namespace monotonicity_f_a1_range_of_m_l389_389753

-- Decreasing and increasing intervals
theorem monotonicity_f_a1 : 
  (∀ x ∈ (0 : ℝ) .. 1, ∀ y ∈ (0 : ℝ) .. 1, x < y → (f y - f x) < 0) ∧
  (∀ x ∈ (1 : ℝ) .. ∞, ∀ y ∈ (1 : ℝ) .. ∞, x < y → (f y - f x) > 0) :=
sorry

-- Range of m
theorem range_of_m (m : ℝ) : 
  (∀ x ∈ [3 : ℝ, ∞), x + 1 / x ≥ |m - 5 / 3| + |m + 5 / 3|) → 
  -5 / 3 ≤ m ∧ m ≤ 5 / 3 :=
sorry

end monotonicity_f_a1_range_of_m_l389_389753


namespace contains_zero_l389_389232

-- Define what constitutes a five-digit number
def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

-- Define the main proof statement
theorem contains_zero (n1 n2 : ℕ) (c1 c2 : Prop) :
  is_five_digit n1 →
  is_five_digit n2 →
  c1 ∧ c2 →
  n1 + n2 = 111111 →
  ∃ (d : ℕ), d < 10 ∧ (d = 0 ∧ (∃ p1 p2 : ℕ, n1 = p1 * 10 + d ∧ n2 = p2 * 10 + d) ∨ 
             ∃ p1 p2 : ℕ, n1 = d * (10^4) + p1 ∧ n2 = d * (10^4) + p2) :=
begin
  sorry
end

end contains_zero_l389_389232


namespace area_triangle_PF1F2_l389_389504

-- Define the ellipse C1 and its foci
def ellipse_C1 (x y : ℝ) : Prop :=
  (x^2 / 6) + (y^2 / 2) = 1

def focus_F1 : (ℝ × ℝ) := (-2, 0)
def focus_F2 : (ℝ × ℝ) := (2, 0)

-- Define the hyperbola C2
def hyperbola_C2 (x y : ℝ) : Prop :=
  (x^2 / 3) - (y^2) = 1

-- Define the point P which is an intersection of ellipse C1 and hyperbola C2
def P (x y : ℝ) : Prop :=
  ellipse_C1 x y ∧ hyperbola_C2 x y

-- The proof statement
theorem area_triangle_PF1F2 (x y : ℝ) (hP : P x y) :
  let PF1 := real.sqrt 6 + real.sqrt 3,
      PF2 := real.sqrt 6 - real.sqrt 3,
      sin_angle := (2 * real.sqrt 2) / 3 in
  (1 / 2) * PF1 * PF2 * sin_angle = real.sqrt 2 :=
sorry

end area_triangle_PF1F2_l389_389504


namespace reciprocal_of_36_recurring_decimal_l389_389960

-- Definitions and conditions
def recurring_decimal (x : ℚ) : Prop := x = 36 / 99

-- Theorem statement
theorem reciprocal_of_36_recurring_decimal :
  recurring_decimal (36 / 99) → (1 / (36 / 99) = 11 / 4) :=
sorry

end reciprocal_of_36_recurring_decimal_l389_389960


namespace Q_difference_l389_389319

def Q (n : ℕ) (x : ℕ) : ℕ :=
  (List.range (10 ^ n)).sum (λ k, Nat.floor (x / (k+1)))

theorem Q_difference (n : ℕ) : Q n (10^n) - Q n (10^n - 1) = n + 1 := by
  sorry

end Q_difference_l389_389319


namespace tenth_term_is_10_l389_389482

def next_term (n : ℕ) : ℕ :=
  if n ≤ 5 then n * 8
  else if n % 2 = 0 then n / 2
  else n + 3

def sequence (a : ℕ) : ℕ → ℕ
| 0     := a
| (n+1) := next_term (sequence n)

theorem tenth_term_is_10 : sequence 100 10 = 10 :=
by sorry

end tenth_term_is_10_l389_389482


namespace contains_zero_l389_389240

theorem contains_zero (a b c d e f: ℕ) (h1: 1 ≤ a ∧ a ≤ 9) (h2: 1 ≤ b ∧ b ≤ 9) (h3: 0 ≤ c ∧ c ≤ 9) 
  (h4: 1 ≤ d ∧ d ≤ 9) (h5: 0 ≤ e ∧ e ≤ 9) (h6: 0 ≤ f ∧ f ≤ 9) 
  (h7: c ≠ f) (h8: 10^4*a + 10^3*b + 10^2*c + 10^1*d + e + 10^4*a + 10^3*b + 10^2*f + 10^1*d + e = 111111) :
  c = 0 ∨ f = 0 := 
sorry

end contains_zero_l389_389240


namespace negation_of_exists_sin_gt_one_l389_389562

theorem negation_of_exists_sin_gt_one : 
  (¬ ∃ x : ℝ, Real.sin x > 1) ↔ (∀ x : ℝ, Real.sin x ≤ 1) := 
by
  sorry

end negation_of_exists_sin_gt_one_l389_389562


namespace solve_equation_l389_389100

theorem solve_equation (x y z : ℕ) : (3 ^ x + 5 ^ y + 14 = z!) ↔ ((x = 4 ∧ y = 2 ∧ z = 5) ∨ (x = 4 ∧ y = 4 ∧ z = 6)) :=
by
  sorry

end solve_equation_l389_389100


namespace reciprocal_of_repeating_decimal_l389_389963

theorem reciprocal_of_repeating_decimal :
  let x := 0.36363636 -- simplified as .\overline{36}
  ∃ y : ℚ, x = 4 / 11 ∧ y = 1 / x ∧ y = 11 / 4 :=
by
  sorry

end reciprocal_of_repeating_decimal_l389_389963


namespace volume_of_sand_pile_l389_389190

theorem volume_of_sand_pile (d h : ℝ) (π : ℝ) (r : ℝ) (vol : ℝ) :
  d = 8 →
  h = (3 / 4) * d →
  r = d / 2 →
  vol = (1 / 3) * π * r^2 * h →
  vol = 32 * π :=
by
  intros hd hh hr hv
  subst hd
  subst hh
  subst hr
  subst hv
  sorry

end volume_of_sand_pile_l389_389190


namespace sequence_general_term_l389_389999

theorem sequence_general_term (n : ℕ) : 
  let a : ℕ → ℚ := λ n, (-1 : ℚ)^n * (n^2 / (n + 1)) in
  a n = (-1)^n * (n^2 / (n + 1)) := 
by
  sorry

end sequence_general_term_l389_389999


namespace largest_value_of_y_l389_389541

theorem largest_value_of_y :
  (∃ x y : ℝ, x^2 + 3 * x * y - y^2 = 27 ∧ 3 * x^2 - x * y + y^2 = 27 ∧ y ≤ 3) → (∃ y : ℝ, y = 3) :=
by
  intro h
  obtain ⟨x, y, h1, h2, h3⟩ := h
  -- proof steps go here
  sorry

end largest_value_of_y_l389_389541


namespace constant_term_expansion_l389_389944

theorem constant_term_expansion (x : ℝ) :
  (constant_term ((5 * x + 2 / (5 * x)) ^ 8) = 1120) := by
  sorry

end constant_term_expansion_l389_389944


namespace sum_of_products_inequality_l389_389581

theorem sum_of_products_inequality (n : ℕ) (x y : Fin n → ℝ) 
  (h1 : ∀ i j, i < j → x i < x j) 
  (h2 : ∀ i, -1 < x i ∧ x i < 1) 
  (h3 : ∀ i j, i < j → y i < y j) 
  (h_sum : ∑ i, x i = ∑ i, x i ^ 13): 
  (∑ i, (x i ^ 13) * (y i) < ∑ i, (x i) * (y i)) :=
sorry

end sum_of_products_inequality_l389_389581


namespace numerators_required_l389_389826

def is_valid_numerator (n : Nat) : Prop :=
  (n % 3 ≠ 0) ∧ (n % 11 ≠ 0)

def numerators_count : Nat :=
  (Finset.range 100).filter (λ n, n > 0 ∧ is_valid_numerator n).card

theorem numerators_required : numerators_count = 60 :=
  sorry

end numerators_required_l389_389826


namespace minimum_value_of_a_l389_389725

def is_prime (n : ℕ) : Prop := sorry  -- Provide the definition of a prime number

def is_perfect_square (n : ℕ) : Prop := sorry  -- Provide the definition of a perfect square

theorem minimum_value_of_a 
  (a b : ℕ) 
  (h1 : is_prime (a - b)) 
  (h2 : is_perfect_square (a * b)) 
  (h3 : a ≥ 2012) : 
  a = 2025 := 
sorry

end minimum_value_of_a_l389_389725


namespace roots_are_real_and_examined_l389_389855

-- Define positive numbers and the condition ps = qr
variables {p q r s : ℝ} (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) (hs : 0 < s) (h : p * s = q * r)

-- Define the polynomial
def polynomial (x : ℝ) : ℝ := p * x^3 - q * x^2 - r * x + s

-- Statement to prove that the roots of the polynomial are real and whether any roots can be equal
theorem roots_are_real_and_examined (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) (hs : 0 < s) (h : p * s = q * r) : 
  ∃ x1 x2 x3 : ℝ, (polynomial p q r s x1 = 0) ∧ (polynomial p q r s x2 = 0) ∧ (polynomial p q r s x3 = 0) ∧ 
  (x1 = sqrt (r / p) ∨ x1 = -sqrt (r / p) ∨ x1 = q / p) ∧ 
  (x2 = sqrt (r / p) ∨ x2 = -sqrt (r / p) ∨ x2 = q / p) ∧ 
  (x3 = sqrt (r / p) ∨ x3 = -sqrt (r / p) ∨ x3 = q / p) ∧ 
  ((∀ x, polynomial p q r s x = 0 → x = sqrt (r / p) ∨ x = -sqrt (r / p) ∨ x = q / p) ∧ 
  (x1 = x2 → x1 = sqrt (r / p) ∧ q^2 = r * p) ∧ 
  (x2 = x3 → x2 = sqrt (r / p) ∧ q^2 = r * p) ∧ 
  (x3 = x1 → x3 = sqrt (r / p) ∧ q^2 = r * p)) :=
sorry

end roots_are_real_and_examined_l389_389855


namespace count_collinear_sets_l389_389209

def num_sets_of_collinear_points (vertices midpoints centers : Set (ℝ × ℝ × ℝ)) (center_of_cube : ℝ × ℝ × ℝ) : ℕ :=
  let points := vertices ∪ midpoints ∪ centers ∪ {center_of_cube}
  -- Assume 'collinear_sets' function that counts sets of three collinear points among the given points
  collinear_sets points

theorem count_collinear_sets :
  let vertices := { -- 8 vertices of the cube
    (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
    (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)
  }
  let midpoints := { -- 12 edge midpoints
    (0.5, 0, 0), (0.5, 1, 0), (0.5, 0, 1), (0.5, 1, 1),
    (0, 0.5, 0), (1, 0.5, 0), (0, 0.5, 1), (1, 0.5, 1),
    (0, 0, 0.5), (1, 0, 0.5), (0, 1, 0.5), (1, 1, 0.5)
  }
  let centers := { -- 6 face centers
    (0.5, 0.5, 0), (0.5, 0.5, 1),
    (0.5, 0, 0.5), (0.5, 1, 0.5),
    (0, 0.5, 0.5), (1, 0.5, 0.5)
  }
  let center_of_cube := (0.5, 0.5, 0.5)
  num_sets_of_collinear_points vertices midpoints centers center_of_cube = 49 := 
sorry

end count_collinear_sets_l389_389209


namespace log_5_3100_nearest_int_l389_389936

theorem log_5_3100_nearest_int :
  (4 : ℝ) < real.log 3100 / real.log 5 ∧ real.log 3100 / real.log 5 < (5 : ℝ) ∧
  abs (3100 - 3125) < abs (3100 - 625) →
  real.log 3100 / real.log 5 = 5 :=
sorry

end log_5_3100_nearest_int_l389_389936


namespace sector_ratio_of_EOF_l389_389870

namespace CircleProblem

variable {O : Type} [circle : Circle O]
variable (A B E F : Point O)
variable (h_diameter : diameter A B)
variable (h_opposite : OppositeSides E F A B)
variable (h_angle_AOE : Angle A O E = 60)
variable (h_angle_BOF : Angle B O F = 30)

theorem sector_ratio_of_EOF :
  sector_area_ratio E O F = 3 / 4 :=
by
  sorry

end CircleProblem

end sector_ratio_of_EOF_l389_389870


namespace transformed_samples_avg_var_l389_389005

noncomputable def is_average (l : List ℝ) (μ : ℝ) : Prop :=
  (l.sum / l.length) = μ

noncomputable def is_variance (l : List ℝ) (σ² : ℝ) : Prop :=
  (l.map (λ x => (x - l.sum / l.length)^2)).sum / l.length = σ²

theorem transformed_samples_avg_var (n : ℕ) (x : Fin n → ℝ) 
(h_avg : is_average (List.ofFn (λ i => x i + 1)) 9)
(h_var : is_variance (List.ofFn (λ i => x i + 1)) 3) : 
  is_average (List.ofFn (λ i => 2 * x i + 3)) 19 ∧
  is_variance (List.ofFn (λ i => 2 * x i + 3)) 12 :=
by
  sorry

end transformed_samples_avg_var_l389_389005


namespace not_necessarily_divisor_of_44_l389_389102

theorem not_necessarily_divisor_of_44 {k : ℤ} (h1 : ∃ k, n = k * (k + 1) * (k + 2)) (h2 : 11 ∣ n) :
  ¬(44 ∣ n) :=
sorry

end not_necessarily_divisor_of_44_l389_389102


namespace sixty_different_numerators_l389_389836

theorem sixty_different_numerators : 
  (Finset.card {ab : ℕ | 1 ≤ ab ∧ ab ≤ 99 ∧ Nat.gcd ab 99 = 1} = 60) :=
sorry

end sixty_different_numerators_l389_389836


namespace find_a2018_l389_389759

-- Defining the sequence {a_n}
def seq (n : ℕ) : ℕ → ℝ
| 0         := 1
| (n + 1)   := 2^(n / 2)

-- Problem statement
theorem find_a2018 : seq (2018) = 2^1009 :=
by begin
  sorry
end

end find_a2018_l389_389759


namespace find_value_of_question_mark_l389_389180

theorem find_value_of_question_mark (q : ℕ) : q * 40 = 173 * 240 → q = 1036 :=
by
  intro h
  sorry

end find_value_of_question_mark_l389_389180


namespace max_distance_from_origin_to_circle_l389_389062

theorem max_distance_from_origin_to_circle :
  let O := (0, 0)
  let C := (3, 4)
  let radius := 1
  let M_condition := ∀ M : ℝ × ℝ, (M.fst - 3)^2 + (M.snd - 4)^2 = 1
  (∀ M : ℝ × ℝ, M_condition M → real.sqrt ((M.fst - O.fst)^2 + (M.snd - O.snd)^2) ≤ 6) := sorry

end max_distance_from_origin_to_circle_l389_389062


namespace simplification_of_expression_l389_389263

theorem simplification_of_expression (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) : 
  ( (x - 2) / (x^2 - 2 * x + 1) / (x / (x - 1)) + 1 / (x^2 - x) ) = 1 / x := 
by 
  sorry

end simplification_of_expression_l389_389263


namespace list_all_E_sequences_A5_correct_E_sequence_increasing_iff_l389_389327

def is_E_sequence (n : ℕ) (a : ℕ → ℤ) : Prop :=
  n ≥ 2 ∧ ∀ k, k ∈ Finset.range (n - 1) → |a (k + 2) - a (k + 1)| = 1

def list_E_sequences_A5 : Finset (ℕ → ℤ) :=
  {a | 
    is_E_sequence 5 a ∧ a 1 = 0 ∧ a 5 = 0
  }

theorem list_all_E_sequences_A5_correct :
  list_E_sequences_A5 = 
    {{x | x 1 = 0 ∧ x 2 = 1 ∧ x 3 = 0 ∧ x 4 = 1 ∧ x 5 = 0},
     {x | x 1 = 0 ∧ x 2 = -1 ∧ x 3 = 0 ∧ x 4 = -1 ∧ x 5 = 0},
     {x | x 1 = 0 ∧ x 2 = -1 ∧ x 3 = 0 ∧ x 4 = 1 ∧ x 5 = 0},
     {x | x 1 = 0 ∧ x 2 = 1 ∧ x 3 = 0 ∧ x 4 = -1 ∧ x 5 = 0}} := sorry

theorem E_sequence_increasing_iff (a : ℕ → ℤ) :
  (∀ k, k ∈ Finset.range (2000 - 1) → a (k + 2) > a (k + 1))
  ↔ a 1 = 13 ∧ a 2000 = 2012 := sorry

end list_all_E_sequences_A5_correct_E_sequence_increasing_iff_l389_389327


namespace find_f1_plus_g1_l389_389345

variables (f g : ℝ → ℝ)

-- Conditions
def even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x
def odd_function (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = -g x
def function_equation (f g : ℝ → ℝ) : Prop := ∀ x : ℝ, f x - g x = x^3 - 2*x^2 + 1

theorem find_f1_plus_g1 
  (hf : even_function f)
  (hg : odd_function g)
  (hfg : function_equation f g):
  f 1 + g 1 = -2 :=
by {
  sorry
}

end find_f1_plus_g1_l389_389345


namespace smallest_four_digit_divisible_by_57_l389_389160

theorem smallest_four_digit_divisible_by_57 : 
  ∃ (n : ℕ), (1000 ≤ n) ∧ (n < 10000) ∧ (n % 57 = 0) ∧ ∀ m : ℕ, (1000 ≤ m) ∧ (m < 10000) ∧ (m % 57 = 0) → m ≥ n :=
begin
  use 1026,
  split, { linarith },
  split, { linarith },
  split, { norm_num },
  intros m hm1 hm2 hm3,
  have h : m ≥ 18 * 57,
  { apply nat.mul_le_mul_right,
    apply nat.floor_le,
    norm_num,
    linarith },
  norm_num at h,
  exact h,
end

end smallest_four_digit_divisible_by_57_l389_389160


namespace find_seven_m_squared_minus_one_l389_389743

theorem find_seven_m_squared_minus_one (m : ℝ)
  (h1 : ∃ x₁, 5 * m + 3 * x₁ = 1 + x₁)
  (h2 : ∃ x₂, 2 * x₂ + m = 3 * m)
  (h3 : ∀ x₁ x₂, (5 * m + 3 * x₁ = 1 + x₁) → (2 * x₂ + m = 3 * m) → x₁ = x₂ + 2) :
  7 * m^2 - 1 = 2 / 7 :=
by
  let m := -3/7
  sorry

end find_seven_m_squared_minus_one_l389_389743


namespace arithmetic_sequence_an_l389_389469

-- Define the arithmetic sequence \{a_n\} and the sum of the first n terms S_n
def Sn (n : ℕ) : ℕ := 5 * n^2 + 3 * n

-- Define the nth term a_n
def an (n : ℕ) : ℕ := an(1) + (n - 1) * 10

-- Prove that a_n equals 10n - 2
theorem arithmetic_sequence_an (n : ℕ) : an(n) = 10 * n - 2 := by
  sorry

end arithmetic_sequence_an_l389_389469


namespace geometric_sequence_sum_identity_l389_389029

noncomputable theory

open_locale big_operators

-- Definitions established from the conditions
def a (n : ℕ) : ℝ := 2^(n-1)
def b (n : ℕ) : ℝ := real.logb 4 (a n)
def S (n : ℕ) : ℝ := ∑ i in finset.range n, b (i + 1)
def partial_sum (n : ℕ) : ℝ := ∑ k in finset.range (n - 1), 1 / (S (k + 2))

-- Main problem statement
theorem geometric_sequence_sum_identity (n : ℕ) (hn : n > 1) :
  partial_sum n = 4 * (1 - 1 / n) :=
sorry

end geometric_sequence_sum_identity_l389_389029


namespace balcony_height_l389_389609

-- Definitions for conditions given in the problem

def final_position := 0 -- y, since the ball hits the ground
def initial_velocity := 5 -- v₀ in m/s
def time_elapsed := 3 -- t in seconds
def gravity := 10 -- g in m/s²

theorem balcony_height : 
  ∃ h₀ : ℝ, final_position = h₀ + initial_velocity * time_elapsed - (1/2) * gravity * time_elapsed^2 ∧ h₀ = 30 := 
by 
  sorry

end balcony_height_l389_389609


namespace number_of_irrationals_l389_389647

def is_irrational (x : ℝ) : Prop := ¬ ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

def sqrt_2_irrational : is_irrational (Real.sqrt 2) := sorry
def pi_over_3_irrational : is_irrational (Real.pi / 3) := sorry
def frac_22_over_7_rational : ¬ is_irrational (22 / 7 : ℝ) := sorry
def non_terminating_decimal_irrational : is_irrational 0.1010010001 := sorry
def cube_root_5_irrational : is_irrational (Real.cbrt 5) := sorry
def sqrt_16_rational : ¬ is_irrational (Real.sqrt 16) := sorry

theorem number_of_irrationals :
  (∃ (S : Finset ℝ), S.card = 4 ∧
    S = {Real.sqrt 2, Real.pi / 3, 0.1010010001, Real.cbrt 5}) := sorry

end number_of_irrationals_l389_389647


namespace proof_problem_l389_389727

-- Given conditions:
def p : Prop := ∀ x : ℝ, y = sin x → sin (-x) = -sin x
def q : Prop := ∀ a b : ℝ, (a^2 ≠ b^2) → (a = b)

-- Goal: Prove that \( p \land \lnot q \).
theorem proof_problem : p ∧ ¬q :=
by
  sorry

end proof_problem_l389_389727


namespace area_of_ABPQ_l389_389452

-- Define the entities in the problem: segment lengths and area calculation
variables (A B X Y P Q : Type*) [EuclideanGeometry A B X Y P Q]

-- Given conditions as variable assignments
def AB : ℝ := 10
def BP : ℝ := 2 * AB
def AP : ℝ := 0.5 * AB

-- Goal statement to prove
theorem area_of_ABPQ : (20 : ℝ) * (5 : ℝ) = 100 :=
by
  -- Given data
  have h1 : AB = 10, by sorry
  have h2 : BP = 2 * AB, by sorry
  have h3 : AP = 0.5 * AB, by sorry

  -- Calculation
  show (20 : ℝ) * (5 : ℝ) = 100, by norm_num

end area_of_ABPQ_l389_389452


namespace minimum_value_l389_389295

theorem minimum_value (x : ℝ) : 
  ∃ (x₀ : ℝ), x₀ = -1 / 2 ∧ 16^x₀ - 4^x₀ + 1 = 3 / 4 
  ∧ ∀ (x : ℝ), 16^x - 4^x + 1 ≥ 3 / 4 :=
begin
  sorry
end

end minimum_value_l389_389295


namespace quadratic_has_roots_l389_389687

theorem quadratic_has_roots :
  ∀ (c : ℝ), (∀ x : ℝ, 2 * x^2 + 8 * x + c = 0 ↔ x = -2 + sqrt 3 ∨ x = -2 - sqrt 3) → c = 6.5 :=
by
  -- Proof to be filled
  sorry

end quadratic_has_roots_l389_389687


namespace vyshny_connections_route_count_l389_389653

theorem vyshny_connections_route_count:
  ∃ (x : ℕ) (routes : ℕ),
  (x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5) ∧
  (routes = 9 ∨ routes = 10 ∨ routes = 15) ∧
  let Moscow_connections : ℕ := 7,
      SPB_connections : ℕ := 5,
      Tver_connections : ℕ := 4,
      Yaroslavl_connections : ℕ := 2,
      Bologoe_connections : ℕ := 2,
      Shestikhino_connections : ℕ := 2,
      Zavidovo_connections : ℕ := 1 in
  23 + x - 1 - 1 = 23 ∧
  (2, 3, 4, 5).contains x ∧
  (9, 10, 15).contains routes := sorry

end vyshny_connections_route_count_l389_389653


namespace odd_log_floor_sum_l389_389854

theorem odd_log_floor_sum (n : ℕ) (h_odd : n % 2 = 1) (h_pos : 0 < n) :
  (∑ k in (finset.filter (λ k, k % 2 = 1) (n.divisors)), int.floor (real.log (n / k) / real.log 2))
  = ((n - 1) / 2 : ℤ) :=
by sorry

end odd_log_floor_sum_l389_389854


namespace find_x_squared_minus_y_squared_l389_389313

theorem find_x_squared_minus_y_squared 
  (x y : ℝ)
  (h1 : x + y = 5)
  (h2 : x - y = 1) :
  x^2 - y^2 = 5 := 
by
  sorry

end find_x_squared_minus_y_squared_l389_389313
