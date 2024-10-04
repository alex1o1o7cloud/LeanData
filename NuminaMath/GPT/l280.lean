import Mathlib
import Mathlib.Algebra.BigOperators.Finprod
import Mathlib.Algebra.Field.Defs
import Mathlib.Algebra.GeomSum
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.GroupPower
import Mathlib.Algebra.GroupWithZero.Defs
import Mathlib.Algebra.Order.SquareRoot
import Mathlib.Algebra.Series
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.NormedSpace.Basic
import Mathlib.Analysis.SpecialFunctions.ExpLog
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Polynomials
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Digits
import Mathlib.Data.Probability
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Logic.Basic
import Mathlib.Nat.Gcd
import Mathlib.Probability.Basic
import Mathlib.ProbabilityTheory.Basic
import Mathlib.Tactic
import Mathlib.Topology.MetricSpace.Basic
import data.list.basic
import data.nat.basic

namespace kolacky_bounds_l280_280509

theorem kolacky_bounds (x y : ℕ) (h : 9 * x + 4 * y = 219) :
  294 ≤ 12 * x + 6 * y ∧ 12 * x + 6 * y ≤ 324 :=
sorry

end kolacky_bounds_l280_280509


namespace probability_not_all_same_l280_280869

-- Definitions of conditions from the problem
def six_sided_die_faces : ℕ := 6
def number_of_dice : ℕ := 5

-- Lean statement to prove the probability calculation
theorem probability_not_all_same : 
  let total_outcomes := six_sided_die_faces ^ number_of_dice in
  let all_same_outcomes := six_sided_die_faces in
  (1 - ((all_same_outcomes : ℚ) / total_outcomes)) = (1295 / 1296) := 
by
  sorry

end probability_not_all_same_l280_280869


namespace find_a1_l280_280766

variable (a : ℕ → ℤ) (S : ℕ → ℤ)

def is_arithmetic_seq (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def sum_n_terms (S : ℕ → ℤ) (a : ℕ → ℤ) : Prop :=
  ∀ n, S n = (n * (a 1 + a n)) / 2

theorem find_a1 (h1 : is_arithmetic_seq a (-2)) 
               (h2 : sum_n_terms S a) 
               (h3 : S 10 = S 11) : 
  a 1 = 20 :=
sorry

end find_a1_l280_280766


namespace exists_at_least_n_charming_7_digit_integers_l280_280926

-- Define what it means for a 7-digit number to be charming
def is_charming (num : ℕ) : Prop :=
  let digits := [1, 2, 3, 4, 5, 6, 7] in
  let permutation := list.perm ([1, 2, 3, 4, 5, 6, 7]) (nat.digits 10 num) in
  permutation ∧ 
  ∀ k, 1 ≤ k ∧ k ≤ 7 → nat.mod (nat.of_digits 10 (take k (nat.digits 10 num))) k = 0

-- The number of charming 7-digit integers, 'n' value is a placeholder
noncomputable def num_charming_7_digit_integers : ℕ := sorry
  
-- The proof problem statement
theorem exists_at_least_n_charming_7_digit_integers (n : ℕ) : 
  ∃ l : list ℕ, l.length ≥ n ∧ ∀ num ∈ l, is_charming num := 
sorry

end exists_at_least_n_charming_7_digit_integers_l280_280926


namespace average_temperature_week_l280_280603

theorem average_temperature_week :
  let sunday := 99.1
  let monday := 98.2
  let tuesday := 98.7
  let wednesday := 99.3
  let thursday := 99.8
  let friday := 99.0
  let saturday := 98.9
  (sunday + monday + tuesday + wednesday + thursday + friday + saturday) / 7 = 99.0 :=
by
  sorry

end average_temperature_week_l280_280603


namespace age_relation_l280_280041

/--
Given that a woman is 42 years old and her daughter is 8 years old,
prove that in 9 years, the mother will be three times as old as her daughter.
-/
theorem age_relation (x : ℕ) (mother_age daughter_age : ℕ) 
  (h1 : mother_age = 42) (h2 : daughter_age = 8) 
  (h3 : 42 + x = 3 * (8 + x)) : 
  x = 9 :=
by
  sorry

end age_relation_l280_280041


namespace cos_inequality_in_triangle_l280_280747

theorem cos_inequality_in_triangle 
  {A B C : ℝ} (hA : A > 0) (hB : B > 0) (hC : C > 0) (h_sum : A + B + C = π) :
  cos (A / 2) + cos (B / 2) + cos (C / 2) ≥ (√3 / 2) * (cos ((B - C) / 2) + cos ((C - A) / 2) + cos ((A - B) / 2)) :=
sorry

end cos_inequality_in_triangle_l280_280747


namespace company_p_employees_in_december_l280_280058

theorem company_p_employees_in_december :
  ∀ (jan_employees : ℝ), jan_employees = 391.304347826087 → 
  (jan_employees + 0.15 * jan_employees = 450) :=
by
  intros jan_employees h1
  rw h1
  sorry

end company_p_employees_in_december_l280_280058


namespace good_sets_count_l280_280024

def card_deck := {n : ℕ // n ≤ 10} ⊕ {0 : ℕ} -- Cards numbered 0 and 1 to 10 in 3 colors

def card_score (k : ℕ) : ℕ :=
  2^k

def good_set_score : ℕ :=
  2004

def number_of_good_sets (deck : multiset (card_deck)) :=
  let scores := deck.map (λ card, match card with
                                 | sum.inl ⟨n, _⟩ => 2^n
                                 | sum.inr 0 => 2^0
                                 end)
  in if scores.sum = good_set_score then 1 else 0

def total_good_sets (deck : multiset (card_deck)) : ℕ :=
  multiset.sum (deck.powerset.map number_of_good_sets)

theorem good_sets_count : total_good_sets 32 = 1006009 :=
sorry

end good_sets_count_l280_280024


namespace max_beads_identified_in_two_weighings_l280_280421

theorem max_beads_identified_in_two_weighings
  (pile : List ℝ) (h_same_shape_size : ∀ b1 b2 ∈ pile, b1 = b2 ∨ b1 ≠ b2)
  (h_one_lighter : ∃ b ∈ pile, ∀ other ∈ pile, b ≠ other → b < other)
  (h_balance_no_weights : True) :
  pile.length ≤ 9 →
  ∃ b ∈ pile, ∀ other ∈ pile, b ≠ other → b < other :=
by
  intro h_len
  sorry

end max_beads_identified_in_two_weighings_l280_280421


namespace no_zeros_implies_a_lt_neg_one_l280_280200

open Real

noncomputable def f (a x : ℝ) : ℝ := 4^x - 2^(x + 1) - a

theorem no_zeros_implies_a_lt_neg_one {
  (h : ∀ x : ℝ, f a x ≠ 0) :
  a < -1 :=
by
  sorry

end no_zeros_implies_a_lt_neg_one_l280_280200


namespace triangle_ratio_l280_280621

theorem triangle_ratio
  (D E F X : Type)
  [DecidableEq D] [DecidableEq E] [DecidableEq F] [DecidableEq X]
  (DE DF : ℝ)
  (hDE : DE = 36)
  (hDF : DF = 40)
  (DX_bisects_EDF : ∀ EX FX, (DE * FX = DF * EX)) :
  ∃ (EX FX : ℝ), EX / FX = 9 / 10 :=
sorry

end triangle_ratio_l280_280621


namespace calculate_monthly_rent_l280_280585

def monthly_earnings_needed (investment: ℝ) (return_rate: ℝ) (taxes_per_year: ℝ) : ℝ :=
  let desired_return := return_rate * investment
  (desired_return + taxes_per_year) / 12

def monthly_rent (earnings_needed: ℝ) (maintenance_rate: ℝ) : ℝ :=
  earnings_needed / (1 - maintenance_rate)

theorem calculate_monthly_rent :
  monthly_rent (monthly_earnings_needed 20000 0.06 400) 0.15 = 156.86 :=
  sorry

end calculate_monthly_rent_l280_280585


namespace subway_train_distance_difference_l280_280948

theorem subway_train_distance_difference :
  let d (s : ℝ) := 0.5 * s^3 + s^2
  in d 7 - d 4 = 172.5 :=
by
  let d := λ s : ℝ, 0.5 * s^3 + s^2
  have h_d7 : d 7 = 220.5 := by
    calc d 7 = 0.5 * 7^3 + 7^2 : by rfl
         ... = 0.5 * 343 + 49 : by norm_num
         ... = 171.5 + 49 : by norm_num
         ... = 220.5 : by norm_num
  have h_d4 : d 4 = 48 := by
    calc d 4 = 0.5 * 4^3 + 4^2 : by rfl
         ... = 0.5 * 64 + 16 : by norm_num
         ... = 32 + 16 : by norm_num
         ... = 48 : by norm_num
  calc d 7 - d 4 = 220.5 - 48 : by rw [h_d7, h_d4]
             ... = 172.5 : by norm_num

end subway_train_distance_difference_l280_280948


namespace product_of_constants_l280_280126

theorem product_of_constants (x t a b : ℤ) (h1 : x^2 + t * x - 12 = (x + a) * (x + b)) :
  ∃ ts : Finset ℤ, ∏ t in ts, t = 1936 :=
by
  sorry

end product_of_constants_l280_280126


namespace expected_lone_cars_l280_280556
-- Import Lean's math library to ensure necessary functions and theorems are available.

-- Define a theorem to prove the expected number of lone cars is 1.
theorem expected_lone_cars (n : ℕ) : 
  -- n must be greater than or equal to 1, since there must be at least one car.
  n ≥ 1 -> 
  -- Expected number of lone cars is 1.
  (∑ k in finset.range n, (1 : ℝ) / (k + 1)) = 1 := 
begin
  intro hn, -- Assume n ≥ 1
  sorry,    -- The proof of this theorem is to be provided, but the statement is correct.
end

end expected_lone_cars_l280_280556


namespace uniq_seq_l280_280642

noncomputable def sequence : ℕ → ℝ 
| 0       := 1
| (n + 1) := sorry -- will be defined in the proof

theorem uniq_seq :
  (∀ n : ℕ, sequence n - sequence (n + 1) = sequence (n + 2)) →
  (sequence 0 = 1) →
  ∀ n : ℕ, sequence n = (Real.sqrt 5 - 1) / 2 ^ n :=
begin
  sorry -- The proof will be inserted here
end

end uniq_seq_l280_280642


namespace student_correct_answers_l280_280916

theorem student_correct_answers 
  (c w : ℕ) 
  (h1 : c + w = 60) 
  (h2 : 4 * c - w = 130) : 
  c = 38 :=
by
  sorry

end student_correct_answers_l280_280916


namespace α_eq_β_plus_two_l280_280167

-- Definitions based on the given conditions:
-- α(n): number of ways n can be expressed as a sum of the integers 1 and 2, considering different orders as distinct ways.
-- β(n): number of ways n can be expressed as a sum of integers greater than 1, considering different orders as distinct ways.

def α (n : ℕ) : ℕ := sorry
def β (n : ℕ) : ℕ := sorry

-- The proof statement that needs to be proved.
theorem α_eq_β_plus_two (n : ℕ) (h : 0 < n) : α n = β (n + 2) := 
  sorry

end α_eq_β_plus_two_l280_280167


namespace probability_not_all_same_l280_280874

/-- What is the probability that when we roll five fair 6-sided dice, they won't all show the same number? -/
theorem probability_not_all_same :
  let total_outcomes := 6^5 in
  let same_number_outcomes := 6 in
  let probability_all_same := same_number_outcomes / total_outcomes.to_real in
  1 - probability_all_same = (1295 : ℝ) / 1296 :=
by
  let total_outcomes := 6^5
  let same_number_outcomes := 6
  have probability_all_same := (same_number_outcomes : ℝ) / total_outcomes.to_real
  show 1 - probability_all_same = (1295 : ℝ) / 1296       
  sorry

end probability_not_all_same_l280_280874


namespace alex_minus_sam_eq_negative_2_50_l280_280247

def original_price : ℝ := 120.00
def tax_rate : ℝ := 0.07
def discount_rate : ℝ := 0.15
def packaging_fee : ℝ := 2.50

def alex_total (original_price tax_rate discount_rate : ℝ) : ℝ :=
  let price_with_tax := original_price * (1 + tax_rate)
  let final_price := price_with_tax * (1 - discount_rate)
  final_price

def sam_total (original_price tax_rate discount_rate packaging_fee : ℝ) : ℝ :=
  let price_with_discount := original_price * (1 - discount_rate)
  let price_with_tax := price_with_discount * (1 + tax_rate)
  let final_price := price_with_tax + packaging_fee
  final_price

theorem alex_minus_sam_eq_negative_2_50 :
  alex_total original_price tax_rate discount_rate - sam_total original_price tax_rate discount_rate packaging_fee = -2.50 := by
  sorry

end alex_minus_sam_eq_negative_2_50_l280_280247


namespace number_of_ostriches_l280_280079

theorem number_of_ostriches
    (x y : ℕ)
    (h1 : x + y = 150)
    (h2 : 2 * x + 6 * y = 624) :
    x = 69 :=
by
  -- Proof omitted
  sorry

end number_of_ostriches_l280_280079


namespace sequence_general_formula_max_S_n_value_l280_280743

noncomputable def a_n : ℕ → ℤ
| n := if n % 2 = 1 then 21 - 3 * n else 30 - 3 * n

def S_n (n : ℕ) : ℤ :=
(nat.range (n + 1)).sum a_n

theorem sequence_general_formula :
  ∀ n, (a_n n = if n % 2 = 1 then 21 - 3 * n else 30 - 3 * n)
    ∧ (a_n 1 = 18)
    ∧ (a_n 2 = 24)
    ∧ ∀ n, a_n (n + 2) - a_n n = -6 := 
begin
  sorry
end

theorem max_S_n_value :
  ∃ n, S_n n = 96 := 
begin
  sorry
end

end sequence_general_formula_max_S_n_value_l280_280743


namespace equal_angles_MKO_MLO_l280_280617

variables {K L A P Q M O : Point} (Γ : Circle)

-- Definitions of conditions
def tangent_points_of_A_on_Γ := (is_tangent_point A P Γ) ∧ (is_tangent_point A Q Γ)
def midpoint_of_PQ (M P Q : Point) := dist M P = dist M Q
def points_on_circle (K L : Point) (Γ : Circle) := Γ.mem K ∧ Γ.mem L

-- Given conditions
axiom K_L_on_circle : points_on_circle K L Γ
axiom A_on_line_KL  : is_on_line A K L
axiom P_Q_tangent_points : tangent_points_of_A_on_Γ A P Q Γ
axiom M_midpoint_PQ : midpoint_of_PQ M P Q

-- Theorem to prove
theorem equal_angles_MKO_MLO :
  ∠MKO = ∠MLO :=
by sorry

end equal_angles_MKO_MLO_l280_280617


namespace eval_complex_powers_l280_280631

theorem eval_complex_powers (i : ℂ) (h1 : i^2 = -1) (h2 : i^4 = 1) : i^8 + i^{24} + i^{-32} = 3 :=
by
  sorry

end eval_complex_powers_l280_280631


namespace non_empty_odd_subsets_l280_280222

theorem non_empty_odd_subsets (S : Finset ℕ) :
  S = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} →
  let odds := {1, 3, 5, 7, 9}
  let all_odd_non_empty_subsets := (2^odds.card - 1) 
  let subsets_with_1_and_5 := (2^(odds.card - 2))
  all_odd_non_empty_subsets - subsets_with_1_and_5 = 23 :=
by
  sorry

end non_empty_odd_subsets_l280_280222


namespace find_principal_amount_l280_280949

theorem find_principal_amount :
  ∃ P : ℝ, (P * (1 + 0.07 / 4) ^ (4 * 2) = 4410) ∧ (P ≈ 3837.97) :=
by
  let r : ℝ := 0.07
  let n : ℕ := 4
  let t : ℕ := 2
  let A : ℝ := 4410
  let factor := (1 + r / n) ^ (n * t)
  have h1 : 1 + r / n = 1.0175 := by norm_num
  have h2 : factor = 1.148882 := by
    calc (1 + r / n) ^ (n * t)
      = (1.0175) ^ 8 : by rw [h1]; norm_num
      ≈ 1.148882 : by norm_num
  use P,
  have h3 : A / factor ≈ 3837.97 := by
    calc A / factor
      = 4410 / 1.148882 : by rw [h2]; norm_num
      ≈ 3837.97 : by norm_num
  exact ⟨P, (P * factor = 4410), h3⟩,
  sorry

end find_principal_amount_l280_280949


namespace increasing_or_decreasing_subsequence_l280_280341

theorem increasing_or_decreasing_subsequence {m n : ℕ} (a : Fin (m * n + 1) → ℝ) :
  (∃ i : Fin (m * n + 2) → ℕ, StrictMono (λ k, a (i k))) ∨
  (∃ i : Fin (n + 1) → ℕ, StrictAnti (λ k, a (i k))) :=
sorry

end increasing_or_decreasing_subsequence_l280_280341


namespace five_dice_not_all_same_number_l280_280891
open Classical

noncomputable def probability_not_all_same (n : ℕ) : ℚ :=
  1 - 1 / (6^n)

theorem five_dice_not_all_same_number :
  probability_not_all_same 5 = 1295 / 1296 :=
by
  sorry

end five_dice_not_all_same_number_l280_280891


namespace james_weekly_expenses_l280_280297

noncomputable def utility_cost (rent: ℝ):  ℝ := 0.2 * rent
noncomputable def weekly_hours_open (hours_per_day: ℕ) (days_per_week: ℕ): ℕ := hours_per_day * days_per_week
noncomputable def employee_weekly_wages (wage_per_hour: ℝ) (weekly_hours: ℕ): ℝ := wage_per_hour * weekly_hours
noncomputable def total_employee_wages (employees: ℕ) (weekly_wages: ℝ): ℝ := employees * weekly_wages
noncomputable def total_weekly_expenses (rent: ℝ) (utilities: ℝ) (employee_wages: ℝ): ℝ := rent + utilities + employee_wages

theorem james_weekly_expenses : 
  let rent := 1200
  let utility_percentage := 0.2
  let hours_per_day := 16
  let days_per_week := 5
  let employees := 2
  let wage_per_hour := 12.5
  let weekly_hours := weekly_hours_open hours_per_day days_per_week
  let utilities := utility_cost rent
  let employee_wages_per_week := employee_weekly_wages wage_per_hour weekly_hours
  let total_employee_wages_per_week := total_employee_wages employees employee_wages_per_week
  total_weekly_expenses rent utilities total_employee_wages_per_week = 3440 := 
by
  sorry

end james_weekly_expenses_l280_280297


namespace smallest_value_fraction_l280_280687

theorem smallest_value_fraction (x y : ℝ) (hx : -6 ≤ x ∧ x ≤ -3) (hy : 3 ≤ y ∧ y ≤ 6) :
  ∃ k : ℝ, (∀ (x y : ℝ), (-6 ≤ x ∧ x ≤ -3) → (3 ≤ y ∧ y ≤ 6) → k ≤ (x + y) / x) ∧ k = 0 :=
by
  sorry

end smallest_value_fraction_l280_280687


namespace gcd_fact8_fact6_squared_l280_280118

-- Definition of 8! and (6!)²
def fact8 : ℕ := 8!
def fact6_squared : ℕ := (6!)^2

-- The theorem statement to be proved
theorem gcd_fact8_fact6_squared : Nat.gcd fact8 fact6_squared = 11520 := 
by
    sorry

end gcd_fact8_fact6_squared_l280_280118


namespace max_value_is_5_l280_280073

def max_value_expr (x y : ℝ) : ℝ :=
  (3 * x + 4 * y + 5) / Real.sqrt (x^2 + y^2 + 4)

theorem max_value_is_5 :
  ∃ (x y : ℝ), max_value_expr x y = 5 :=
by
  sorry

end max_value_is_5_l280_280073


namespace jane_earnings_l280_280753

variables (payment_per_bulb : ℝ) (tulip daffodil : ℕ)
variable iris := tulip / 2
variable hyacinth := iris + 2
variable crocus := daffodil * 3
variable gladiolus := 2 * (crocus - daffodil)

def total_bulbs := tulip + iris + hyacinth + daffodil + crocus + gladiolus
def earnings := total_bulbs * payment_per_bulb

theorem jane_earnings :
  payment_per_bulb = 0.5 →
  tulip = 20 →
  daffodil = 30 →
  earnings = 141 :=
by
  intros h1 h2 h3
  -- iris computation
  have h_iris : iris = 10 := by
    simp [iris, h2]
  -- hyacinth computation
  have h_hyacinth : hyacinth = 12 := by
    simp [hyacinth, h_iris]
  -- crocus computation
  have h_crocus : crocus = 90 := by
    simp [crocus, h3]
  -- gladiolus computation
  have h_gladiolus : gladiolus = 120 := by
    simp [gladiolus, h_crocus, h3]
  -- total_bulbs computation
  have h_total_bulbs : total_bulbs = 282 := by
    simp [total_bulbs, h2, h_iris, h_hyacinth, h_crocus, h_gladiolus, h3]
  -- earnings computation
  simp [earnings, h_total_bulbs, h1]
  sorry

end jane_earnings_l280_280753


namespace probability_sum_greater_than_four_l280_280479

theorem probability_sum_greater_than_four :
  let all_outcomes := (Fin 6) × (Fin 6)
  let favorable_outcomes := {p : Fin 6 × Fin 6 | (p.1.val + 1) + (p.2.val + 1) > 4}
  (favorable_outcomes.card : ℚ) / ((Fin 6 × Fin 6).card : ℚ) = 5 / 6 :=
by
  sorry

end probability_sum_greater_than_four_l280_280479


namespace directrix_of_parabola_l280_280100

theorem directrix_of_parabola :
  (∀ x : ℝ, y = 3 * x^2 - 6 * x + 1) → 
  ∃ c : ℝ, c = -25/12 ∧ (∀ x, y + c = 0) := 
sorry

end directrix_of_parabola_l280_280100


namespace simplify_expression_l280_280972

theorem simplify_expression :
  2 * sqrt 12 * (3 * sqrt 48 - 4 * sqrt (1 / 8) - 3 * sqrt 27) = 36 - 4 * sqrt 6 := by
  sorry

end simplify_expression_l280_280972


namespace find_point_P_l280_280176

-- Define the points M and N
def M : ℝ × ℝ := (3, -2)
def N : ℝ × ℝ := (-5, -1)

-- We state the main theorem
theorem find_point_P : ∃ P : ℝ × ℝ, P = (-1, -3/2) ∧ 
  (λ M N P : ℝ × ℝ, (P.1 - M.1, P.2 - M.2) = (1/2) • (N.1 - M.1, N.2 - M.2)) M N P :=
sorry

end find_point_P_l280_280176


namespace problem1_problem2_problem3_problem4_l280_280925

-- Problem 1
theorem problem1 (a b x y : ℝ) (h : x^2 / a^2 - y^2 / b^2 = 1) : 
  deriv (λ x, y) x = (b^2 * x) / (a^2 * y) :=
sorry

-- Problem 2
theorem problem2 (ϕ r : ℝ) (h : real.exp (ϕ - 2) + r * ϕ - 3 * r - 2 = 0) :
  deriv (λ ϕ, r) 2 = 0 :=
sorry

-- Problem 3
theorem problem3 (x y : ℝ) (h : x^y = y^x) : 
  deriv (λ y, x) y = (x * (x - y * real.log x)) / (y * (y - x * real.log y)) :=
sorry

-- Problem 4
theorem problem4 (x y : ℝ) (h : x^2 + y^2 - 4 * x - 10 * y + 4 = 0) (hx : x = 8) : 
  deriv (λ x, y) x = (4 - 2 * x) / (2 * y - 10) :=
sorry

end problem1_problem2_problem3_problem4_l280_280925


namespace max_min_angle_is_36_l280_280659

noncomputable def max_min_angle_is_36_degrees : Prop :=
  let points := (Fin 5 → ℝ × ℝ)
  ∃ (A : points), 
    (∀ i j k : Fin 5, i ≠ j ∧ j ≠ k ∧ k ≠ i → (A i, A j, A k).not_collinear) ∧
    (∀ m : ℝ, (∃ i j k : Fin 5, i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ ∠ (A i) (A j) (A k) = m) → 
      36 ≤ m)

theorem max_min_angle_is_36 : max_min_angle_is_36_degrees :=
  sorry

end max_min_angle_is_36_l280_280659


namespace highest_power_of_prime_dividing_factorial_l280_280776

open Nat

theorem highest_power_of_prime_dividing_factorial (p : ℕ) (hp : Prime p) 
 (a : Fin k → ℕ) (h_bound : ∀ i : Fin k, a i < p) 
 (n : ℕ) (h_representation : n = ∑ i in range k, a i * p^i) :
   ∃ α_p : ℕ, α_p = (n - (∑ i in range k, a i)) / (p - 1) :=
by
  sorry

end highest_power_of_prime_dividing_factorial_l280_280776


namespace point_D_on_segment_in_triangle_l280_280745

def point := (ℝ × ℝ)

def triangle_vertices (A B C : point) : Prop := 
  A = (1, 2) ∧ B = (4, 6) ∧ C = (6, 3)

def on_segment (A B : point) (D : point) : Prop := 
  ∃ t ∈ set.Icc (1 : ℝ) 4, D = (t, (4/3 : ℝ) * t - (2/3 : ℝ))

theorem point_D_on_segment_in_triangle :
  ∀ (A B C : point) (D : point),
    triangle_vertices A B C → 
    on_segment A B D → 
    let (x, y) := D in y = (4/3 : ℝ) * x - (2/3 : ℝ) :=
by 
  intros A B C D H_vertices H_on_segment
  obtain ⟨t, ht, hd⟩ := H_on_segment
  simp [hd]
  sorry

end point_D_on_segment_in_triangle_l280_280745


namespace slope_angle_45_l280_280409

theorem slope_angle_45 (a : ℝ) : 
  (∃ k, k = Real.tan (Real.pi / 4) ∧ k = -a) → a = -1 :=
  begin
    intro h,
    cases h with k hk,
    cases hk with hk1 hk2,
    sorry
  end

end slope_angle_45_l280_280409


namespace probability_not_all_dice_same_l280_280889

theorem probability_not_all_dice_same :
  let total_outcomes := 6^5
  let same_number_outcomes := 6
  let probability_same_number := same_number_outcomes / total_outcomes
  let probability_not_same_number := 1 - probability_same_number
  probability_not_same_number = (1295 : ℚ) / 1296 :=
by
  sorry

end probability_not_all_dice_same_l280_280889


namespace average_temperature_week_l280_280602

theorem average_temperature_week :
  let sunday := 99.1
  let monday := 98.2
  let tuesday := 98.7
  let wednesday := 99.3
  let thursday := 99.8
  let friday := 99.0
  let saturday := 98.9
  (sunday + monday + tuesday + wednesday + thursday + friday + saturday) / 7 = 99.0 :=
by
  sorry

end average_temperature_week_l280_280602


namespace locus_of_excircles_l280_280641

-- Define the right triangle type with properties
structure RightTriangle :=
(A B C : Point)
(hypotenuse : Segment)
(is_right : rightTriangle A B C)
(AB_is_hypotenuse : hypotenuse = ⟨A, B⟩)

-- Define excenters corresponding to the vertices of the triangle
def excenter_a (T : RightTriangle) : Point := sorry
def excenter_b (T : RightTriangle) : Point := sorry
def excenter_c (T : RightTriangle) : Point := sorry

-- Define the circle arcs corresponding to excenters loci
def Omega1_arc (T : RightTriangle) : Arc := sorry
def Omega2_arc (T : RightTriangle) : Arc := sorry

-- Main theorem statement: the locus of excenters I_a, I_b, and I_c
theorem locus_of_excircles (T : RightTriangle) :
  ∀I, (I = excenter_a T ∨ I = excenter_b T ∨ I = excenter_c T) →
  (I ∈ Omega1_arc T ∨ I ∈ Omega2_arc T) :=
  sorry

end locus_of_excircles_l280_280641


namespace domain_of_f_half_x_l280_280185

-- Define a function f over real numbers
variable {f : ℝ → ℝ}

-- Define the domain of f(log10(x)) as [0.1, 100]
def domain_f_log10 := {x | 0.1 ≤ (log 10 x) ∧ (log 10 x) ≤ 2}

-- Define the domain of f(x) as [-1, 2] inferred from domain_f_log10
def domain_f := {x | -1 ≤ x ∧ x ≤ 2}

-- Define the domain of f(x/2) based on the domain of f(x)
def domain_f_half_x := {x | -2 ≤ x ∧ x ≤ 4}

-- The statement we need to prove
theorem domain_of_f_half_x :
  domain_f_half_x = {x | -2 ≤ x ∧ x ≤ 4} :=
sorry

end domain_of_f_half_x_l280_280185


namespace derivative_of_y_integral_value_l280_280551

/-- We want to prove the derivative of the given function -/
theorem derivative_of_y :
  let y x := (3 * x ^ 2 - x * sqrt x + 5 * sqrt x - 9) / sqrt x
  in deriv y = λ x, (9 / 2) * x ^ (1 / 2) - 1 - (9 / 2) * x ^ (- 3 / 2) :=
by sorry

/-- We want to prove the value of the definite integral -/
theorem integral_value :
  ∫ x in 1..2, 1 / (x ^ 2 + 2 * x) = (1 / 2) * log (3 / 2) :=
by sorry

end derivative_of_y_integral_value_l280_280551


namespace angle_ACB_eq_40_l280_280426

theorem angle_ACB_eq_40 (A B C D E F : Point) (x : ℝ)
  (h1 : dist A B = 3 * dist A C)
  (h2 : F ∈ line_through A E)
  (h3 : F ∈ line_through C D)
  (h4 : angle B A E = x)
  (h5 : angle A C D = x)
  (h6 : equilateral (triangle C F E)) :
  angle A C B = 40 :=
sorry

end angle_ACB_eq_40_l280_280426


namespace fraction_representation_of_3_36_l280_280524

theorem fraction_representation_of_3_36 : (336 : ℚ) / 100 = 84 / 25 := 
by sorry

end fraction_representation_of_3_36_l280_280524


namespace probability_sum_greater_than_four_l280_280456

theorem probability_sum_greater_than_four : 
  let num_dice := 2
  let sides_per_die := 6
  let favorable_outcomes := { (a, b) | a > 0 ∧ a ≤ sides_per_die ∧ b > 0 ∧ b ≤ sides_per_die ∧ a + b > 4 }
  let total_outcomes := sides_per_die * sides_per_die
  let probability := (favorable_outcomes.card : ℚ) / total_outcomes
  probability = 5 / 6 :=
by
  sorry

end probability_sum_greater_than_four_l280_280456


namespace arithmetic_sequence_common_difference_l280_280277

theorem arithmetic_sequence_common_difference :
  ∃ d : ℚ,
  let a := 7 in
  let l := 88 in
  let S := 570 in
  ∃ n : ℕ,
  l = a + (n - 1) * d ∧ S = n * (a + l) / 2 ∧ d = 81 / 11 :=
by
  -- an outline of high-level steps, actual proof is omitted here
  sorry

end arithmetic_sequence_common_difference_l280_280277


namespace hexagon_largest_angle_measure_l280_280824

theorem hexagon_largest_angle_measure (x : ℝ) (a b c d e f : ℝ)
  (h_ratio: a = 2 * x) (h_ratio2: b = 3 * x)
  (h_ratio3: c = 3 * x) (h_ratio4: d = 4 * x)
  (h_ratio5: e = 4 * x) (h_ratio6: f = 6 * x)
  (h_sum: a + b + c + d + e + f = 720) :
  f = 2160 / 11 :=
by
  -- Proof is not required
  sorry

end hexagon_largest_angle_measure_l280_280824


namespace inequality_proof_l280_280339

theorem inequality_proof (a : ℝ) : (3 * a - 6) * (2 * a^2 - a^3) ≤ 0 := 
by 
  sorry

end inequality_proof_l280_280339


namespace perpendicular_and_bisecting_diagonals_implies_rhombus_l280_280512

structure Quadrilateral :=
(vertices : Fin 4 → Point)
-- Assume some function that checks if the diagonals are perpendicular
(def quadrilateral_perpendicular_diagonals (q : Quadrilateral) : Prop := sorry)
-- Assume some function that checks if the diagonals bisect each other
(def quadrilateral_bisecting_diagonals (q : Quadrilateral) : Prop := sorry)
-- Assume some function that checks if a quadrilateral is a rhombus
(def is_rhombus (q : Quadrilateral) : Prop := sorry)

theorem perpendicular_and_bisecting_diagonals_implies_rhombus (q : Quadrilateral) :
  quadrilateral_perpendicular_diagonals q ∧ quadrilateral_bisecting_diagonals q → is_rhombus q := sorry

end perpendicular_and_bisecting_diagonals_implies_rhombus_l280_280512


namespace length_of_platform_is_correct_l280_280006

-- Given conditions:
def length_of_train : ℕ := 250
def speed_of_train_kmph : ℕ := 72
def time_to_cross_platform : ℕ := 20

-- Convert speed from kmph to m/s
def speed_of_train_mps : ℕ := speed_of_train_kmph * 1000 / 3600

-- Distance covered in 20 seconds
def distance_covered : ℕ := speed_of_train_mps * time_to_cross_platform

-- Length of the platform
def length_of_platform : ℕ := distance_covered - length_of_train

-- The proof statement
theorem length_of_platform_is_correct :
  length_of_platform = 150 := by
  -- This proof would involve the detailed calculations and verifications as laid out in the solution steps.
  sorry

end length_of_platform_is_correct_l280_280006


namespace problem_statement_l280_280780

noncomputable def f (x : ℝ) : ℝ := sorry

axiom even_function : ∀ x : ℝ, f (-x) = f (x)
axiom functional_eq : ∀ x : ℝ, f (x + 6) = f (x) + f 3
axiom initial_condition : f 2 = 3

theorem problem_statement : f 2006 + f 2007 = 3 :=
by
  sorry

end problem_statement_l280_280780


namespace probability_sum_greater_than_four_is_5_over_6_l280_280431

-- Define the sample space for two dice.
def sample_space : Finset (ℕ × ℕ) :=
  Finset.product (Finset.range 6) (Finset.range 6) 

-- Define the event where the sum is greater than four.
def event_sum_greater_than_four : Finset (ℕ × ℕ) :=
  sample_space.filter (λ p, p.1 + p.2 > 4)

-- Compute the probability of the event.
def probability_sum_greater_than_four : ℚ :=
  (event_sum_greater_than_four.card : ℚ) / (sample_space.card : ℚ)

-- Theorem to be proven
theorem probability_sum_greater_than_four_is_5_over_6 :
  probability_sum_greater_than_four = 5 / 6 :=
by
  -- Proof would go here
  sorry

end probability_sum_greater_than_four_is_5_over_6_l280_280431


namespace find_interval_for_inequality_l280_280636

theorem find_interval_for_inequality :
  ∀ x ∈ Icc 0 (2 * Real.pi),
    (2 * Real.cos x ≤ abs (Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))) ∧
    abs (Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))) ≤ Real.sqrt 2)
    ↔ x ∈ Icc (Real.pi / 4) ((7 : ℝ) * Real.pi / 4) :=
by
  sorry

end find_interval_for_inequality_l280_280636


namespace poodle_barks_proof_l280_280574

-- Definitions based on our conditions
def terrier_barks (hushes : Nat) : Nat := hushes * 2
def poodle_barks (terrier_barks : Nat) : Nat := terrier_barks * 2

-- Given that the terrier's owner says "hush" six times
def hushes : Nat := 6
def terrier_barks_total : Nat := terrier_barks hushes

-- The final statement that we need to prove
theorem poodle_barks_proof : 
    ∃ P, P = poodle_barks terrier_barks_total ∧ P = 24 := 
by
  -- The proof goes here
  sorry

end poodle_barks_proof_l280_280574


namespace problem_statement_l280_280969

noncomputable def z : ℂ := complex.of_real_angle (125 * real.pi / 180) -- z = cos 125° + i sin 125°

theorem problem_statement :
  z^48 = -1 / 2 + (real.sqrt 3 / 2) * complex.I := by
  sorry

end problem_statement_l280_280969


namespace correct_propositions_l280_280905

-- Define the propositions
def prop1 : Prop := ¬∃ S : set ℝ, ∀ x ∈ S, x is "very small" -- Note: "very small" is informal
def prop2 : Prop := {y : ℝ | ∃ x : ℝ, y = x^2 - 1} ≠ {(x, y) : ℝ × ℝ | y = x^2 - 1}
def prop3 : Prop := ({1, 3/2, 6/4, -|1/2|, 0.5} : finset ℚ).card ≠ 5
def prop4 : Prop := {p : ℝ × ℝ | p.1 * p.2 ≤ 0} ≠ {p : ℝ × ℝ | (p.1 ≤ 0 ∧ p.2 ≥ 0) ∨ (p.1 ≥ 0 ∧ p.2 ≤ 0)}

-- Stating the problem equivalent to proving the answer is 0 correct propositions
theorem correct_propositions : 
  ¬prop1 ∧ ¬prop2 ∧ ¬prop3 ∧ ¬prop4 ↔ true := 
begin
  -- Proof starts here
  sorry
end

end correct_propositions_l280_280905


namespace cube_traversal_impossible_l280_280566

theorem cube_traversal_impossible :
  let vertices : ℕ := 8
  let face_centers : ℕ := 6
  ∃ G : Graph, G.vertices = vertices ∧ G.face_centers = face_centers ∧
  ¬ (∃ path : List G.V, path.nodup ∧ ∀ edge ∈ G.edges, edge ∈ path) := by
  sorry

end cube_traversal_impossible_l280_280566


namespace inequality_inequality_always_holds_l280_280232

theorem inequality_inequality_always_holds (x y : ℝ) (h : x > y) : |x| > y :=
sorry

end inequality_inequality_always_holds_l280_280232


namespace probability_sum_greater_than_four_l280_280470

theorem probability_sum_greater_than_four : 
  let total_outcomes := 36 in
  let favorable_outcomes := total_outcomes - 6 in
  let probability := favorable_outcomes / total_outcomes in
  probability = 5 / 6 :=
by
  sorry

end probability_sum_greater_than_four_l280_280470


namespace solution_set_of_inequality_l280_280154

theorem solution_set_of_inequality (a : ℝ) (x : ℝ) (h : 4^a = 2^(a + 2)) :
  a = 2 → (a^(2 * x + 1) > a^(x - 1) ↔ x > -2) := by
  sorry

end solution_set_of_inequality_l280_280154


namespace probability_not_all_same_l280_280856

theorem probability_not_all_same :
    let total_outcomes := 6 ^ 5 in
    let same_number_outcomes := 6 in
    let p_all_same := same_number_outcomes / total_outcomes in
    let p_not_all_same := 1 - p_all_same in
    p_not_all_same = 1295 / 1296 :=
by
  sorry

end probability_not_all_same_l280_280856


namespace initial_black_pens_correct_l280_280936

-- Define the conditions
def initial_blue_pens : ℕ := 9
def removed_blue_pens : ℕ := 4
def remaining_blue_pens : ℕ := initial_blue_pens - removed_blue_pens

def initial_red_pens : ℕ := 6
def removed_red_pens : ℕ := 0
def remaining_red_pens : ℕ := initial_red_pens - removed_red_pens

def total_remaining_pens : ℕ := 25
def removed_black_pens : ℕ := 7

-- Assume B is the initial number of black pens
def B : ℕ := 21

-- Prove the initial number of black pens condition
theorem initial_black_pens_correct : 
  (initial_blue_pens + B + initial_red_pens) - (removed_blue_pens + removed_black_pens) = total_remaining_pens :=
by 
  have h1 : initial_blue_pens - removed_blue_pens = remaining_blue_pens := rfl
  have h2 : initial_red_pens - removed_red_pens = remaining_red_pens := rfl
  have h3 : remaining_blue_pens + (B - removed_black_pens) + remaining_red_pens = total_remaining_pens := sorry
  exact h3

end initial_black_pens_correct_l280_280936


namespace relatively_prime_divisibility_l280_280635

theorem relatively_prime_divisibility (x y : ℕ) (h1 : Nat.gcd x y = 1) (h2 : y^2 * (y - x)^2 ∣ x^2 * (x + y)) :
  (x = 2 ∧ y = 1) ∨ (x = 3 ∧ y = 1) :=
sorry

end relatively_prime_divisibility_l280_280635


namespace product_of_constants_t_l280_280130

theorem product_of_constants_t (a b : ℤ) (t : ℤ) (h1 : a * b = -12) (h2 : t = a + b) :
  ∃ (ts : Finset ℤ), (ts = {11, 4, 1, -1, -4, -11}) ∧ ts.prod (λ x, x) = -1936 :=
by sorry

end product_of_constants_t_l280_280130


namespace count_integers_200_to_250_increasing_digits_l280_280712

theorem count_integers_200_to_250_increasing_digits : 
  (∃ n, 200 ≤ n ∧ n < 250 ∧ (∀ i j k : ℕ, n = 200 + i*10 + j*1 + k*1 ∧ 2 < i ∧ i < j ∧ j < k) → false) = 15 :=
sorry

end count_integers_200_to_250_increasing_digits_l280_280712


namespace mondays_equals_fridays_in_30_days_l280_280939

theorem mondays_equals_fridays_in_30_days (days_in_month : ℕ) (days_in_week : ℕ) 
(starting_days : Finset ℤ) (days_with_equal_mondays_fridays : ℕ) :
  days_in_month = 30 →
  days_in_week = 7 →
  ∃ count_days_with_equal_mondays_fridays, 
  count_days_with_equal_mondays_fridays = days_with_equal_mondays_fridays ∧ 
  days_with_equal_mondays_fridays = starting_days.filter 
  (λ start_day, let (mondays, fridays) := ((30 / days_in_week) + if start_day < 2 then 1 else 0,
                                          (30 / days_in_week) + if 4 - start_day < 2 then 1 else 0)
  in mondays = fridays) .card ∧ 
  days_with_equal_mondays_fridays = 3 :=
begin
  intros h_month h_week,
  use (starting_days.filter 
  (λ start_day, let (mondays, fridays) := ((30 / 7) + if start_day < 2 then 1 else 0,
                                          (30 / 7) + if 4 - start_day < 2 then 1 else 0) 
  in mondays = fridays).card,
  split,
  exact rfl,
  sorry -- Proof part skipped
end

end mondays_equals_fridays_in_30_days_l280_280939


namespace prob_correct_l280_280940

def P : ℕ × ℕ → ℚ
-- Base cases
| (0, 0) := 1
| (x+1, 0) := 0
| (0, y+1) := 0
-- Recursive case
| (x+1, y+1) := (P (x, y+1) + P (x+1, y) + P (x, y)) / 3

theorem prob_correct : ∃ (m n : ℕ), P (5, 5) = m / 3 ^ n ∧ m + n = 251 :=
by
  existsi 245
  existsi 6
  -- Assuming P(5,5) equals the computed fraction
  have h1 : P (5, 5) = 245 / 729 := sorry
  -- Verify that 729 = 3^6
  have h2 : 729 = 3 ^ 6 := by norm_num
  rw [← h1, h2]
  norm_num
  sorry

end prob_correct_l280_280940


namespace quadratic_expression_range_of_b_l280_280707

-- Define the context and conditions for the first part of the problem
variable (a b c x : ℝ)

-- Part (1): Analytic expression of f(x) given the conditions
theorem quadratic_expression (h1 : ∀ x, f x = a * x^2 + b * x + c)
  (h2 : ∀ x, f (-1 - x) = f (-1 + x))
  (h3 : f 0 = 1)
  (h4 : ∃ x, f x = 0) :
  f x = (x + 1)^2 := 
sorry

-- Define the context and conditions for the second part of the problem.
variables (a : ℝ) {c : ℝ} (b : ℝ)

-- Part (2): Range of b
theorem range_of_b (h5 : a = 1) 
  (h6 : c = 0)
  (h7 : x ∈ Ioo 0 1) :
  -2 ≤ b ∧ b ≤ 0 := 
sorry

end quadratic_expression_range_of_b_l280_280707


namespace probability_correct_l280_280445

def outcome (s₁ s₂ : ℕ) : Prop := s₁ ≥ 1 ∧ s₁ ≤ 6 ∧ s₂ ≥ 1 ∧ s₂ ≤ 6

def sum_outcome_greater_than_four (s₁ s₂ : ℕ) : Prop := outcome s₁ s₂ ∧ s₁ + s₂ > 4

def total_outcomes : ℕ := 36

def favorable_outcomes : ℕ := 30 -- As derived from 36 - 6

def probability_sum_greater_than_four : ℚ := favorable_outcomes / total_outcomes

theorem probability_correct : probability_sum_greater_than_four = 5 / 6 := 
by 
  sorry

end probability_correct_l280_280445


namespace probability_sum_greater_than_four_l280_280466

def sum_greater_than_four_probability :=
  (5 / 6: ℚ)

theorem probability_sum_greater_than_four :
  let outcomes := { (a, b) | a in (Finset.range 1 7) ∧ b in (Finset.range 1 7) }
  let favorable_outcomes := outcomes.filter (λ pair, (pair.1 + pair.2) > 4)
  let probability := favorable_outcomes.card.to_rat / outcomes.card.to_rat
  probability = sum_greater_than_four_probability :=
by
  sorry

end probability_sum_greater_than_four_l280_280466


namespace meet_time_approx_l280_280013

noncomputable def length_of_track : ℝ := 1800 -- in meters
noncomputable def speed_first_woman : ℝ := 10 * 1000 / 3600 -- in meters per second
noncomputable def speed_second_woman : ℝ := 20 * 1000 / 3600 -- in meters per second
noncomputable def relative_speed : ℝ := speed_first_woman + speed_second_woman

theorem meet_time_approx (ε : ℝ) (hε : ε = 216.048) :
  ∃ t : ℝ, t = length_of_track / relative_speed ∧ abs (t - ε) < 0.001 :=
by
  sorry

end meet_time_approx_l280_280013


namespace student_departments_l280_280423

variable {Student : Type}
variable (Anna Vika Masha : Student)

-- Let Department be an enumeration type representing the three departments
inductive Department
| Literature : Department
| History : Department
| Biology : Department

open Department

variables (isLit : Student → Prop) (isHist : Student → Prop) (isBio : Student → Prop)

-- Conditions
axiom cond1 : isLit Anna → ¬isHist Masha
axiom cond2 : ¬isHist Vika → isLit Anna
axiom cond3 : ¬isLit Masha → isBio Vika

-- Target conclusion
theorem student_departments :
  isHist Vika ∧ isLit Masha ∧ isBio Anna :=
sorry

end student_departments_l280_280423


namespace product_of_t_l280_280135

theorem product_of_t (a b : ℤ) (t : ℤ) (h1 : a * b = -12) (h2 : t = a + b) :
  ∃ (t_values : Finset ℤ), 
  (∀ x ∈ t_values, ∃ a b : ℤ, a * b = -12 ∧ x = a + b) ∧ 
  (t_values.product = -1936) :=
by
  sorry

end product_of_t_l280_280135


namespace problem_b_problem_d_l280_280237

variable (x y t : ℝ)

def condition_curve (t : ℝ) : Prop :=
  ∃ C : ℝ × ℝ → Prop, ∀ x y : ℝ, C (x, y) ↔ (x^2 / (5 - t) + y^2 / (t - 1) = 1)

theorem problem_b (h1 : t < 1) : condition_curve t → ∃ (C : ℝ × ℝ → Prop), (∀ x y, C (x, y) ↔ x^2 / (5 - t) + y^2 / (t - 1) = 1) → ¬(5 - t) < 0 ∧ (t - 1) < 0 := 
sorry

theorem problem_d (h1 : 3 < t) (h2 : t < 5) (h3 : condition_curve t) : ∃ (C : ℝ × ℝ → Prop), (∀ x y, C (x, y) ↔ x^2 / (5 - t) + y^2 / (t - 1) = 1) → 0 < (t - 1) ∧ (t - 1) > (5 - t) := 
sorry

end problem_b_problem_d_l280_280237


namespace number_of_nickels_is_three_l280_280538

def coin_problem : Prop :=
  ∃ p n d q : ℕ,
    p + n + d + q = 12 ∧
    p + 5 * n + 10 * d + 25 * q = 128 ∧
    p ≥ 1 ∧ n ≥ 1 ∧ d ≥ 1 ∧ q ≥ 1 ∧
    q = 2 * d ∧
    n = 3

theorem number_of_nickels_is_three : coin_problem := 
by 
  sorry

end number_of_nickels_is_three_l280_280538


namespace probability_not_all_same_l280_280866

-- Definitions of conditions from the problem
def six_sided_die_faces : ℕ := 6
def number_of_dice : ℕ := 5

-- Lean statement to prove the probability calculation
theorem probability_not_all_same : 
  let total_outcomes := six_sided_die_faces ^ number_of_dice in
  let all_same_outcomes := six_sided_die_faces in
  (1 - ((all_same_outcomes : ℚ) / total_outcomes)) = (1295 / 1296) := 
by
  sorry

end probability_not_all_same_l280_280866


namespace guide_is_knight_l280_280583

-- Definitions
def knight (p : Prop) : Prop := p
def liar (p : Prop) : Prop := ¬p

-- Conditions
variable (GuideClaimsKnight : Prop)
variable (SecondResidentClaimsKnight : Prop)
variable (GuideReportsAccurately : Prop)

-- Proof problem
theorem guide_is_knight
  (GuideClaimsKnight : Prop)
  (SecondResidentClaimsKnight : Prop)
  (GuideReportsAccurately : (GuideClaimsKnight ↔ SecondResidentClaimsKnight)) :
  GuideClaimsKnight := 
sorry

end guide_is_knight_l280_280583


namespace smallest_constant_n_l280_280137

theorem smallest_constant_n :
  ∀ (a b c d e : ℝ), 
    (0 < a) → (0 < b) → (0 < c) → (0 < d) → (0 < e) →
    (sqrt (a / (b + c + d + e)) + sqrt (b / (a + c + d + e)) +
     sqrt (c / (a + b + d + e)) + sqrt (d / (a + b + c + e)) +
     sqrt (e / (a + b + c + d))) > 2 :=
by
  intros a b c d e ha hb hc hd he
  sorry

end smallest_constant_n_l280_280137


namespace max_lambda_mu_of_triangle_ABO_l280_280746

/-- In triangle ABO with OA = OB = 1 and ∠AOB = π/3, given OC = λ⃗OA + μ⃗OB and |⃗OC| = √3,
    the maximum value of λ + μ is 2. -/
theorem max_lambda_mu_of_triangle_ABO 
  (A B O C : Type) [InnerProductSpace ℝ O] (OA OB OC : O)
  (lambda mu : ℝ)
  (h1 : ‖OA‖ = 1)
  (h2 : ‖OB‖ = 1)
  (h3 : inner_product O OA OB = 1/2)
  (h4 : OC = λ • OA + μ • OB)
  (h5 : ‖OC‖ = sqrt 3) : 
  λ + μ ≤ 2 :=
sorry

end max_lambda_mu_of_triangle_ABO_l280_280746


namespace roots_of_quadratic_l280_280318

theorem roots_of_quadratic (m n : ℝ) (h₁ : m + n = -2) (h₂ : m * n = -2022) (h₃ : ∀ x, x^2 + 2 * x - 2022 = 0 → x = m ∨ x = n) :
  m^2 + 3 * m + n = 2020 :=
sorry

end roots_of_quadratic_l280_280318


namespace division_then_multiplication_l280_280501

theorem division_then_multiplication : (180 / 6) * 3 = 90 := 
by
  have step1 : 180 / 6 = 30 := sorry
  have step2 : 30 * 3 = 90 := sorry
  sorry

end division_then_multiplication_l280_280501


namespace rosie_pies_l280_280801

def apples_to_pies (total_apples usable_fraction apples_per_pie: ℝ) : ℝ :=
  (total_apples * usable_fraction) / apples_per_pie

theorem rosie_pies :
  apples_to_pies 48 0.75 (12 / 3) = 9 := 
sorry

end rosie_pies_l280_280801


namespace complex_conjugate_solution_l280_280671

noncomputable def Z : ℂ := 1 - I
def eq_condition := Z * (I - 1) = 2 * I
def conjugate_Z := conj Z = 1 + I

theorem complex_conjugate_solution (h : eq_condition) : conjugate_Z :=
by
  sorry

end complex_conjugate_solution_l280_280671


namespace arithmetic_expression_evaluation_l280_280017

theorem arithmetic_expression_evaluation :
  4 * (7 * 24) / 3 + 5 * (13 * 15) - 2 * (6 * 28) + 7 * (3 * 19) / 2 = 1062.5 := 
by
  -- Skipping the proof.
  sorry

end arithmetic_expression_evaluation_l280_280017


namespace find_divisor_l280_280921

theorem find_divisor (D : ℕ) : 
  (242 % D = 15) ∧ 
  (698 % D = 27) ∧ 
  ((242 + 698) % D = 5) → 
  D = 42 := 
by 
  sorry

end find_divisor_l280_280921


namespace abs_inequality_no_solution_l280_280240

theorem abs_inequality_no_solution (a : ℝ) : (∀ x : ℝ, |x - 5| + |x + 3| ≥ a) ↔ a ≤ 8 :=
by sorry

end abs_inequality_no_solution_l280_280240


namespace mn_sum_l280_280720

theorem mn_sum (M N : ℚ) (h1 : (4 : ℚ) / 7 = M / 63) (h2 : (4 : ℚ) / 7 = 84 / N) : M + N = 183 := sorry

end mn_sum_l280_280720


namespace cos_alpha_minus_beta_sin_alpha_l280_280711

noncomputable section

open Real

variables (α β : ℝ)
variables (a b : ℝ × ℝ)
hypothesis (h0 : a = (cos α, sin α))
hypothesis (h1 : b = (cos β, sin β))
hypothesis (h2 : dist a b = 4 * sqrt 13)
hypothesis (h3 : 0 < α ∧ α < pi / 2)
hypothesis (h4 : -pi / 2 < β ∧ β < 0)
hypothesis (h5 : sin β = -4/5)

theorem cos_alpha_minus_beta : cos (α - β) = 5/13 := 
by
  sorry

theorem sin_alpha : sin α = 12/13 :=
by
  sorry

end cos_alpha_minus_beta_sin_alpha_l280_280711


namespace sum_of_ages_l280_280301

theorem sum_of_ages (age1 age2 age3 : ℕ) (h : age1 * age2 * age3 = 128) : age1 + age2 + age3 = 18 :=
sorry

end sum_of_ages_l280_280301


namespace equal_integral_functions_3_pairs_l280_280689

def integrals_equal (f g : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∫ x in a..b, f x = ∫ x in a..b, g x

theorem equal_integral_functions_3_pairs:
  let f1 := λ x : ℝ, 2 * |x|
  let g1 := λ x : ℝ, x + 1
  let f2 := λ x : ℝ, sin x
  let g2 := λ x : ℝ, cos x
  let f3 := λ x : ℝ, sqrt (1 - x^2)
  let g3 := λ x : ℝ, (3 / 4) * π * x^2
  let f4 := λ x : ℝ, x
  let g4 := λ x : ℝ, -x
  (integrals_equal f1 g1 (-1) 1) ∧
  ¬(integrals_equal f2 g2 (-1) 1) ∧
  (integrals_equal f3 g3 (-1) 1) ∧
  (integrals_equal f4 g4 (-1) 1) :=
by
  -- Proof goes here
  sorry

end equal_integral_functions_3_pairs_l280_280689


namespace avg_salary_rest_l280_280815

def total_average_salary (n total_salary : ℝ) : ℝ := total_salary / n

def total_salary (n avg_salary : ℝ) : ℝ := n * avg_salary

theorem avg_salary_rest (w : ℝ) (w_avg : ℝ := 8000) (t : ℝ := 7) (t_avg : ℝ := 12000) (n : ℝ := 21) : 
  total_average_salary (n - t) ((w_avg * n) - (total_salary t t_avg)) = 6000 :=
by
  have w_total_salary : ℝ := total_salary t t_avg
  have total_rest_salary : ℝ := (w_avg * n) - w_total_salary
  have rest_count : ℝ := n - t
  show total_rest_salary / rest_count = 6000
  sorry

end avg_salary_rest_l280_280815


namespace every_natural_gt_2_is_sum_of_distinct_fibonacci_l280_280210

-- Definition of the Fibonacci sequence
def fibonacci : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fibonacci (n+1) + fibonacci n

-- The theorem to state
theorem every_natural_gt_2_is_sum_of_distinct_fibonacci :
  ∀ n : ℕ, n > 2 → ∃ (S : Finset ℕ), (∀ m ∈ S, m ≠ 0 ∧ ∃ k, fibonacci k = m) ∧ S.sum id = n :=
by
  sorry

end every_natural_gt_2_is_sum_of_distinct_fibonacci_l280_280210


namespace log_base_2_of_81_eq_4m_l280_280716

theorem log_base_2_of_81_eq_4m (m : ℝ) (h : log 2 3 = m) : log 2 81 = 4 * m :=
by
  sorry

end log_base_2_of_81_eq_4m_l280_280716


namespace probability_sum_greater_than_four_l280_280484

theorem probability_sum_greater_than_four :
  let all_outcomes := (Fin 6) × (Fin 6)
  let favorable_outcomes := {p : Fin 6 × Fin 6 | (p.1.val + 1) + (p.2.val + 1) > 4}
  (favorable_outcomes.card : ℚ) / ((Fin 6 × Fin 6).card : ℚ) = 5 / 6 :=
by
  sorry

end probability_sum_greater_than_four_l280_280484


namespace range_of_a8_l280_280287

variable {a_n : ℕ → ℝ}
variable d : ℝ
variable a_1 a_2 a_3 a_4 a_5 : ℝ

axiom arithmetic_sequence_def : ∀ n : ℕ, a_n = a_1 + (n - 1) * d
axiom condition_1 : a_3 + a_6 = a_4 + 5
axiom condition_2 : a_2 ≤ 1

theorem range_of_a8 : a_8 ≥ 9 :=
by
  have h_arithmetic := λ n, arithmetic_sequence_def n
  have h3 := h_arithmetic 3
  have h4 := h_arithmetic 4
  have h6 := h_arithmetic 6
  have h5 := h_arithmetic 5
  rw [h3, h4, h6, h5] at condition_1
  sorry

end range_of_a8_l280_280287


namespace probability_correct_l280_280449

def outcome (s₁ s₂ : ℕ) : Prop := s₁ ≥ 1 ∧ s₁ ≤ 6 ∧ s₂ ≥ 1 ∧ s₂ ≤ 6

def sum_outcome_greater_than_four (s₁ s₂ : ℕ) : Prop := outcome s₁ s₂ ∧ s₁ + s₂ > 4

def total_outcomes : ℕ := 36

def favorable_outcomes : ℕ := 30 -- As derived from 36 - 6

def probability_sum_greater_than_four : ℚ := favorable_outcomes / total_outcomes

theorem probability_correct : probability_sum_greater_than_four = 5 / 6 := 
by 
  sorry

end probability_correct_l280_280449


namespace inscribed_circle_radius_l280_280259

-- Define the given conditions
def radius_large : ℝ := 18
def radius_small : ℝ := 9
def radius_inscribed : ℝ := 8

-- Define tangency conditions and relationships based on the problem statement
def large_semicircle (R : ℝ) := { x : ℝ // 0 <= x ∧ x <= R }
def small_semicircle (r : ℝ) := { x : ℝ // 0 <= x ∧ x <= r }

-- Prove the radius of the circle inscribed between the two semicircles
theorem inscribed_circle_radius :
  large_semicircle radius_large ∧ small_semicircle radius_small →
  ∃ (x : ℝ), x = radius_inscribed := 
by
  intro h;  -- Assume the hypothesis h
  exists radius_inscribed;  -- Show the existence of the radius of the inscribed circle
  have hp1 : sqrt (324 - 36 * radius_inscribed) + radius_small = sqrt (81 + 18 * radius_inscribed) := sorry,
  have hp2 : sqrt (324 - 36 * radius_inscribed) = sqrt (81 + 18 * radius_inscribed) - 9 := sorry,
  have h_sqr : (324 - 36 * radius_inscribed) = (sqrt (81 + 18 * radius_inscribed) - 9)^2 := sorry,
  sorry  -- Proof skipped for simplicity of problem setup

end inscribed_circle_radius_l280_280259


namespace rec_seq1_rec_seq2_l280_280101

-- First recurrence sequence
theorem rec_seq1 (a : ℕ → ℚ) (n : ℕ) :
  a 1 = 1/2 ∧ (∀ n ≥ 1, a (n + 1) = (a n + 3) / (2 * a n - 4)) →
  a n = (- 5 ^ n + (-1) ^ (n-1) * 3 * 2 ^ (n + 1)) / (2 * 5 ^ n + (-1) ^ (n-1) * 2 ^ (n + 1)) :=
by {
  assume h,
  sorry
}

-- Second recurrence sequence
theorem rec_seq2 (a : ℕ → ℚ) (n : ℕ) :
  a 1 = 5 ∧ (∀ n ≥ 1, a (n + 1) = (a n - 4) / (a n - 3)) →
  a n = (6 * n - 11) / (3 * n - 4) :=
by {
  assume h,
  sorry
}

end rec_seq1_rec_seq2_l280_280101


namespace part_I_min_value_m_l280_280702

noncomputable def f_part_I (x : ℝ) : ℝ := abs (2 * x^2 - 1) + x

theorem part_I_min_value_m :
  ∃ (m : ℝ), m = -real.sqrt 2 / 2 ∧ ∃ (x : ℝ), f_part_I x - m ≤ 0 := 
sorry

end part_I_min_value_m_l280_280702


namespace area_of_triangle_is_39_l280_280099

def point := ℝ × ℝ

def A : point := (-2, 3)
def B : point := (8, -1)
def C : point := (10, 6)

def vector (p q : point) : point :=
  (q.1 - p.1, q.2 - p.2)

def cross_product (v w : point) : ℝ :=
  v.1 * w.2 - v.2 * w.1

def area_of_triangle (A B C : point) : ℝ :=
  |cross_product (vector C A) (vector C B)| / 2

theorem area_of_triangle_is_39 : area_of_triangle A B C = 39 := 
by {
  sorry
}

end area_of_triangle_is_39_l280_280099


namespace quadratic_form_2b_3c_l280_280405

theorem quadratic_form_2b_3c : 
    ∃ (a b c : ℝ), (∀ x : ℝ, 4*x^2 - 40*x + 100 = (a*x + b)^2 + c) 
        ∧ a = 2 
        ∧ b = -10 
        ∧ c = 0 
        ∧ (2*b - 3*c = -20) := 
by 
    use 2, -10, 0 
    split 
    { intros x 
      calc
        4*x^2 - 40*x + 100
          = 4*((x-5)^2) : by { ring }
          = (2*x - 10)^2 + 0 : by { ring }
    }
    split 
    { refl }
    split 
    { refl }
    split 
    { refl }
    exact (by linarith : 2 * (-10) - 3 * 0 = -20)

end quadratic_form_2b_3c_l280_280405


namespace area_of_quadrilateral_l280_280419

theorem area_of_quadrilateral :
  ∀ (O₁ O₂ A B C D : Point) (r : Real) (one_cm : r = 1),
  -- Conditions
  dist O₁ A = r ∧ dist O₁ O₂ = 2 * r ∧
  circle O₁ r ∩ line_segment O₁ O₂ = {C, D} ∧
  circle O₂ r ∩ line_segment O₁ O₂ = {C, D} ∧
  quadrilateral A B C D ∧ on_line C (line_segment O₁ O₂) ∧ on_line D (line_segment O₁ O₂) ∧
  -- Conditions specific to area calculation
  (∃ E : Point, perpendicular DE AB ∧ dist DE = r) →
  -- Result: the area of quadrilateral ABCD is 1 cm²
  quadrilateral_area A B C D = r * r := by
  sorry

end area_of_quadrilateral_l280_280419


namespace sum_of_integer_solutions_is_8_l280_280807

theorem sum_of_integer_solutions_is_8 :
  let f := λ x: ℝ, 8 * ((|x+3| - |x-5|) / (|2*x-11| - |2*x+7|)) - 9 * ((|x+3| + |x-5|) / (|2*x-11| + |2*x+7|)) in
  (∑ i in Finset.filter (λ x, (f x ≥ -8) ∧ (|x| < 90)) (Finset.range 180).filter_map (λ x, if x%2 = 0 then some (x / 2 : ℤ) else none)) = 8 := 
sorry

end sum_of_integer_solutions_is_8_l280_280807


namespace black_spot_shape_square_l280_280727

theorem black_spot_shape_square (O : Point) (P : ℚ) 
  (rectangles : set (Rectangle))
  (h1 : ∀ r ∈ rectangles, r.center = O)
  (h2 : ∀ r ∈ rectangles, r.perimeter = P)
  (h3 : ∀ r1 r2 ∈ rectangles, r1.sides_parallel_to r2) :
  (spot_shape rectangles).shape = Shape.square :=
by
  sorry

end black_spot_shape_square_l280_280727


namespace max_value_l280_280189

noncomputable def satisfies_equation (x y : ℝ) : Prop :=
  x + 4 * y - x * y = 0

theorem max_value (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : satisfies_equation x y) :
  ∃ m, m = (4 / (x + y)) ∧ m ≤ (4 / 9) :=
by
  sorry

end max_value_l280_280189


namespace one_divides_the_other_l280_280308

theorem one_divides_the_other (x y : ℕ) (hx : 0 < x) (hy : 0 < y) 
(h : Nat.lcm (x+2) (y+2) - Nat.lcm (x+1) (y+1) = Nat.lcm (x+1) (y+1) - Nat.lcm x y) :
  ∃ m n : ℕ, (x = m * y) ∨ (y = n * x) :=
by 
  -- Proof goes here
  sorry

end one_divides_the_other_l280_280308


namespace minimum_50_boxes_required_l280_280851

-- Define the concept of cards and boxes
def is_valid_box (card : ℕ) (box : ℕ) : Prop :=
  let digits := (card / 100, (card / 10) % 10, card % 10)
  in (digits.2 + digits.3).even ∧ box = 10 * (card / 10 % 10) + card % 10
     ∨ (digits.1 + digits.3).even ∧ box = 10 * (card / 100) + card % 10
     ∨ (digits.1 + digits.2).even ∧ box = 10 * (card / 100) + (card / 10 % 10)

-- Prove that it is possible to place 1000 cards into at least 50 boxes
theorem minimum_50_boxes_required : 
  ∃ (boxes : finset ℕ), boxes.card = 50 ∧ 
  ∀ card : ℕ, card ≥ 0 ∧ card < 1000 → 
    ∃ box ∈ boxes, is_valid_box card box :=
by
  sorry

end minimum_50_boxes_required_l280_280851


namespace probability_sum_greater_than_four_l280_280457

theorem probability_sum_greater_than_four : 
  let num_dice := 2
  let sides_per_die := 6
  let favorable_outcomes := { (a, b) | a > 0 ∧ a ≤ sides_per_die ∧ b > 0 ∧ b ≤ sides_per_die ∧ a + b > 4 }
  let total_outcomes := sides_per_die * sides_per_die
  let probability := (favorable_outcomes.card : ℚ) / total_outcomes
  probability = 5 / 6 :=
by
  sorry

end probability_sum_greater_than_four_l280_280457


namespace min_xy_l280_280698

theorem min_xy (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 2 * x + 8 * y - x * y = 0) : xy ≥ 64 :=
by sorry

end min_xy_l280_280698


namespace shaded_area_of_circumscribed_circles_l280_280612

theorem shaded_area_of_circumscribed_circles :
  let r1 := 3 in
  let r2 := 4 in
  let r3 := r1 + r2 in
  let area_larger_circle := π * r3^2 in
  let area_smaller_circle1 := π * r1^2 in
  let area_smaller_circle2 := π * r2^2 in
  (area_larger_circle - (area_smaller_circle1 + area_smaller_circle2)) = 24 * π :=
by
  let r1 := 3
  let r2 := 4
  let r3 := r1 + r2
  let area_larger_circle := π * r3^2
  let area_smaller_circle1 := π * r1^2
  let area_smaller_circle2 := π * r2^2
  show (area_larger_circle - (area_smaller_circle1 + area_smaller_circle2)) = 24 * π, from sorry

end shaded_area_of_circumscribed_circles_l280_280612


namespace final_result_l280_280036

/-- A student chose a number, multiplied it by 5, then subtracted 138 
from the result. The number he chose was 48. What was the final result 
after subtracting 138? -/
theorem final_result (x : ℕ) (h1 : x = 48) : (x * 5) - 138 = 102 := by
  sorry

end final_result_l280_280036


namespace probability_not_all_same_l280_280864

-- Definitions of conditions from the problem
def six_sided_die_faces : ℕ := 6
def number_of_dice : ℕ := 5

-- Lean statement to prove the probability calculation
theorem probability_not_all_same : 
  let total_outcomes := six_sided_die_faces ^ number_of_dice in
  let all_same_outcomes := six_sided_die_faces in
  (1 - ((all_same_outcomes : ℚ) / total_outcomes)) = (1295 / 1296) := 
by
  sorry

end probability_not_all_same_l280_280864


namespace worker_bees_in_hive_l280_280249

variable (initialWorkerBees leavingWorkerBees returningWorkerBees : ℕ)

def finalWorkerBees (initialWorkerBees leavingWorkerBees returningWorkerBees : ℕ) : ℕ :=
  initialWorkerBees - leavingWorkerBees + returningWorkerBees

theorem worker_bees_in_hive
  (initialWorkerBees : ℕ := 400)
  (leavingWorkerBees : ℕ := 28)
  (returningWorkerBees : ℕ := 15) :
  finalWorkerBees initialWorkerBees leavingWorkerBees returningWorkerBees = 387 := by
  sorry

end worker_bees_in_hive_l280_280249


namespace graph_shift_correct_l280_280845

theorem graph_shift_correct :
  ∀ (x : ℝ), 3 * cos (2 * x - π / 3) = 3 * cos (2 * (x - π / 6)) :=
by
  intro x
  sorry

end graph_shift_correct_l280_280845


namespace probability_sum_greater_than_four_l280_280486

theorem probability_sum_greater_than_four : 
  let total_outcomes := 36 in
  let outcomes_sum_less_or_equal_4 := 6 in
  let prob_sum_less_or_equal_4 := outcomes_sum_less_or_equal_4 / total_outcomes in
  prob_sum_less_or_equal_4 = (1 : ℝ) / 6 → 
  (1 - prob_sum_less_or_equal_4) = (5 : ℝ) / 6 := 
by 
  intros total_outcomes outcomes_sum_less_or_equal_4 prob_sum_less_or_equal_4 h1,
  sorry

end probability_sum_greater_than_four_l280_280486


namespace probability_sum_greater_than_four_l280_280476

theorem probability_sum_greater_than_four : 
  let total_outcomes := 36 in
  let favorable_outcomes := total_outcomes - 6 in
  let probability := favorable_outcomes / total_outcomes in
  probability = 5 / 6 :=
by
  sorry

end probability_sum_greater_than_four_l280_280476


namespace average_last_three_l280_280374

/-- The average of the last three numbers is 65, given that the average of six numbers is 60
  and the average of the first three numbers is 55. -/
theorem average_last_three (a b c d e f : ℝ) (h1 : (a + b + c + d + e + f) / 6 = 60) (h2 : (a + b + c) / 3 = 55) :
  (d + e + f) / 3 = 65 :=
by
  sorry

end average_last_three_l280_280374


namespace parallelogram_ABCD_l280_280427

variables 
  (Γ1 Γ2 : Circle)
  (M N A B C D : Point)
  (ℓ : Line)
  (h_Γ1Γ2_intersect : Intersect Γ1 Γ2 = {M, N})
  (h_tangent_Γ1 : Tangent ℓ Γ1 A)
  (h_tangent_Γ2 : Tangent ℓ Γ2 B)
  (h_AC_perp_ℓ : Perpendicular (LineThrough A C) ℓ)
  (h_BD_perp_ℓ : Perpendicular (LineThrough B D) ℓ)
  (h_CMN : C ∈ LineThrough M N)
  (h_DMN : D ∈ LineThrough M N)

theorem parallelogram_ABCD : Parallelogram A B C D :=
  sorry

end parallelogram_ABCD_l280_280427


namespace min_cone_volume_l280_280035

theorem min_cone_volume (h : ℝ) (r : ℝ) (hsphere : sphere_radius = 1) (cone_condition : touches_base) :
  volume_of_cone h r = (8 * π / 3) :=
by
  sorry

end min_cone_volume_l280_280035


namespace smallest_four_digit_number_l280_280504

def is_digit_even (d : ℕ) : Prop := d % 2 = 0

def is_digit_odd (d : ℕ) : Prop := d % 2 = 1

def sum_of_digits (n : ℕ) : ℕ :=
let digits := [n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10]
in digits.sum

def is_smallest_valid_number (n : ℕ) : Prop :=
  n >= 1000 ∧ 
  n < 10000 ∧ 
  n % 6 = 0 ∧ 
  (sum_of_digits n) % 3 = 0 ∧ 
  let d := [n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10] in
  (d.filter is_digit_even).length = 3 ∧ 
  (d.filter is_digit_odd).length = 1 ∧ 
  ∀ m, (m >= 1000 ∧ m < 10000 ∧ m % 6 = 0 ∧ 
    (sum_of_digits m) % 3 = 0 ∧ 
    let dm := [m / 1000 % 10, m / 100 % 10, m / 10 % 10, m % 10] in
    (dm.filter is_digit_even).length = 3 ∧ 
    (dm.filter is_digit_odd).length = 1) -> 
  n ≤ m

theorem smallest_four_digit_number : is_smallest_valid_number 1002 := 
  sorry

end smallest_four_digit_number_l280_280504


namespace count_terminating_decimals_l280_280662

def terminates_as_decimal (n: ℕ) : Prop :=
  ∃ k: ℕ, n = k * 21

theorem count_terminating_decimals :
  (finset.range 300).filter (λ n, terminates_as_decimal n).card = 14 :=
sorry

end count_terminating_decimals_l280_280662


namespace project_completion_days_l280_280003

-- Define the work rates and the total number of days to complete the project
variables (a_rate b_rate : ℝ) (days_to_complete : ℝ)
variable (a_quit_before_completion : ℝ)

-- Define the conditions
def A_rate := 1 / 20
def B_rate := 1 / 20
def quit_before_completion := 10 

-- The total work done in the project as 1 project 
def total_work := 1

-- Define the equation representing the amount of work done by A and B
def total_days := 
  A_rate * (days_to_complete - a_quit_before_completion) + B_rate * days_to_complete

-- The theorem statement
theorem project_completion_days :
  A_rate = a_rate → 
  B_rate = b_rate → 
  quit_before_completion = a_quit_before_completion → 
  total_days = total_work → 
  days_to_complete = 15 :=
by 
  -- placeholders for the conditions
  intros h1 h2 h3 h4
  sorry

end project_completion_days_l280_280003


namespace simplify_336_to_fraction_l280_280520

theorem simplify_336_to_fraction : (336 / 100) = (84 / 25) :=
by sorry

end simplify_336_to_fraction_l280_280520


namespace probability_sum_greater_than_four_l280_280492

theorem probability_sum_greater_than_four : 
  let total_outcomes := 36 in
  let outcomes_sum_less_or_equal_4 := 6 in
  let prob_sum_less_or_equal_4 := outcomes_sum_less_or_equal_4 / total_outcomes in
  prob_sum_less_or_equal_4 = (1 : ℝ) / 6 → 
  (1 - prob_sum_less_or_equal_4) = (5 : ℝ) / 6 := 
by 
  intros total_outcomes outcomes_sum_less_or_equal_4 prob_sum_less_or_equal_4 h1,
  sorry

end probability_sum_greater_than_four_l280_280492


namespace gcd_factorial_8_6_squared_l280_280103

theorem gcd_factorial_8_6_squared : Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2) = 5760 := by
  sorry

end gcd_factorial_8_6_squared_l280_280103


namespace at_least_one_machine_requires_maintenance_l280_280586

variable (P_A : ℝ) (P_B : ℝ) (hA : P_A = 0.9) (hB : P_B = 0.8)
variable (independent_AB : Independent {A} {B})

theorem at_least_one_machine_requires_maintenance :
  P(A ∪ B) = 0.98 := by
  sorry

end at_least_one_machine_requires_maintenance_l280_280586


namespace intersection_complement_range_m_l280_280214

open Set

variable (A : Set ℝ) (B : ℝ → Set ℝ) (m : ℝ)

def setA : Set ℝ := Icc (-1 : ℝ) (3 : ℝ)
def setB (m : ℝ) : Set ℝ := Icc m (m + 6)

theorem intersection_complement (m : ℝ) (h : m = 2) : 
  (setA ∩ (setB 2)ᶜ) = Ico (-1 : ℝ) (2 : ℝ) :=
by
  sorry

theorem range_m (m : ℝ) : 
  A ∪ B m = B m ↔ -3 ≤ m ∧ m ≤ -1 :=
by
  sorry

end intersection_complement_range_m_l280_280214


namespace chloe_treasures_first_level_l280_280968

def chloe_treasures_score (T : ℕ) (score_per_treasure : ℕ) (treasures_second_level : ℕ) (total_score : ℕ) :=
  T * score_per_treasure + treasures_second_level * score_per_treasure = total_score

theorem chloe_treasures_first_level :
  chloe_treasures_score T 9 3 81 → T = 6 :=
by
  intro h
  sorry

end chloe_treasures_first_level_l280_280968


namespace inscribed_circle_radius_l280_280260

theorem inscribed_circle_radius (R r : ℝ) (hR : R = 18) (hr : r = 9) : ∃ x : ℝ, x = 8 := 
by
  use 8
  sorry


end inscribed_circle_radius_l280_280260


namespace probability_correct_l280_280447

def outcome (s₁ s₂ : ℕ) : Prop := s₁ ≥ 1 ∧ s₁ ≤ 6 ∧ s₂ ≥ 1 ∧ s₂ ≤ 6

def sum_outcome_greater_than_four (s₁ s₂ : ℕ) : Prop := outcome s₁ s₂ ∧ s₁ + s₂ > 4

def total_outcomes : ℕ := 36

def favorable_outcomes : ℕ := 30 -- As derived from 36 - 6

def probability_sum_greater_than_four : ℚ := favorable_outcomes / total_outcomes

theorem probability_correct : probability_sum_greater_than_four = 5 / 6 := 
by 
  sorry

end probability_correct_l280_280447


namespace dining_bill_split_l280_280545

theorem dining_bill_split (original_bill : ℝ) (num_people : ℕ) (tip_percent : ℝ) (total_bill_with_tip : ℝ) (amount_per_person : ℝ)
  (h1 : original_bill = 139.00)
  (h2 : num_people = 3)
  (h3 : tip_percent = 0.10)
  (h4 : total_bill_with_tip = original_bill + (tip_percent * original_bill))
  (h5 : amount_per_person = total_bill_with_tip / num_people) :
  amount_per_person = 50.97 :=
by 
  sorry

end dining_bill_split_l280_280545


namespace handshake_count_l280_280052

def total_handshakes (men women : ℕ) := 
  (men * (men - 1)) / 2 + men * (women - 1)

theorem handshake_count :
  let men := 13
  let women := 13
  total_handshakes men women = 234 :=
by
  sorry

end handshake_count_l280_280052


namespace evaluate_expression_l280_280087

theorem evaluate_expression : (8 : ℝ) ^ (-2 / 3) + (81 : ℝ) ^ (-1 / 2) = 13 / 36 := by
  let a := (8 : ℝ)
  let b := (81 : ℝ)
  have h1 : a = 2 ^ 3 := by sorry
  have h2 : b = 3 ^ 4 := by sorry
  have h3 : a⁻¹ * (a^(2 / 3)) = (a ^ (-2 / 3)) := by sorry
  have h4 : b⁻¹ * (b^(1 / 2)) = (b ^ (-1 / 2)) := by sorry
  have h5 : (2 ^ 3) ^ (2 / 3) = 4 := by sorry
  have h6 : (3 ^ 4) ^ (1 / 2) = 9 := by sorry
  have h7 : 1 / 4 + 1 / 9 = 13 / 36 := by sorry
  sorry

end evaluate_expression_l280_280087


namespace part1_minimum_value_of_f_part2_range_of_a_l280_280774

open Real

-- Define the functions f(x) and g(x)
def f (x : ℝ) := (x + 1) * log (x + 1)
def g (a x : ℝ) := a * x^2 + x

-- Problem (1): Prove the minimum value of f(x)
theorem part1_minimum_value_of_f :
  ∃ x : ℝ, f x = -(1 / exp 1) :=
sorry

-- Problem (2): Find the range of real number a
theorem part2_range_of_a (a : ℝ) :
  (∀ x : ℝ, x ≥ 0 → f x ≤ g a x) → a ≥ 1/2 :=
sorry

end part1_minimum_value_of_f_part2_range_of_a_l280_280774


namespace frustum_sphere_radius_l280_280935

noncomputable def frustum_volume (r1 r2 h : ℝ) : ℝ :=
  (1 / 3) * real.pi * h * (r1^2 + r1 * r2 + r2^2)

noncomputable def sphere_radius (volume : ℝ) : ℝ :=
  real.cbrt (volume * 3 / (4 * real.pi))

theorem frustum_sphere_radius (r1 r2 h : ℝ) (r1_eq : r1 = 2) (r2_eq : r2 = 3) (h_eq : h = 5) :
  sphere_radius (frustum_volume r1 r2 h) = real.cbrt (95 / 4) :=
by
  rw [←r1_eq, ←r2_eq, ←h_eq]
  sorry

end frustum_sphere_radius_l280_280935


namespace dollar_op_5_neg2_l280_280070

def dollar_op (x y : Int) : Int := x * (2 * y - 1) + 2 * x * y

theorem dollar_op_5_neg2 :
  dollar_op 5 (-2) = -45 := by
  sorry

end dollar_op_5_neg2_l280_280070


namespace graph_equiv_l280_280394

theorem graph_equiv {x y : ℝ} :
  (x^3 - 2 * x^2 * y + x * y^2 - 2 * y^3 = 0) ↔ (x = 2 * y) :=
sorry

end graph_equiv_l280_280394


namespace unique_solution_l280_280071

noncomputable def fulfills_conditions (f : ℝ → ℝ → ℝ) : Prop :=
  (∀ x y z : ℝ, f(x,y) + f(y,z) + f(z,x) = Real.max (Real.max x y) z - Real.min (Real.min x y) z) ∧ 
  (∃ a : ℝ, ∀ x : ℝ, f(x, a) = f(a, x))

theorem unique_solution (f : ℝ → ℝ → ℝ) (h : fulfills_conditions f) :
  ∀ x y : ℝ, f(x,y) = |(x - y) / 2| :=
sorry

end unique_solution_l280_280071


namespace intersection_of_squares_perimeter_l280_280395

noncomputable def perimeter_of_rectangle (side1 side2 : ℝ) : ℝ :=
2 * (side1 + side2)

theorem intersection_of_squares_perimeter
  (side_length : ℝ)
  (diagonal : ℝ)
  (distance_between_centers : ℝ)
  (h1 : 4 * side_length = 8) 
  (h2 : (side1^2 + side2^2) = diagonal^2)
  (h3 : (2 - side1)^2 + (2 - side2)^2 = distance_between_centers^2) : 
10 * (perimeter_of_rectangle side1 side2) = 25 :=
sorry

end intersection_of_squares_perimeter_l280_280395


namespace find_a_l280_280637

noncomputable def sequence_converges (a : ℝ) : Prop :=
∃ L : ℝ, (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, abs (x n a - L) < ε)

def recurrence_relation (x : ℕ → ℝ) (a : ℝ) : Prop :=
x 0 = 1996 ∧ ∀ n : ℕ, x (n+1) = a / (1 + (x n)^2)

theorem find_a (a : ℝ) : 
  (∃ x : ℕ → ℝ, recurrence_relation x a ∧ sequence_converges x a) ↔ (|a| ≤ 2) :=
sorry

end find_a_l280_280637


namespace difference_between_numbers_l280_280841

theorem difference_between_numbers :
  ∃ X Y : ℕ, 
    100 ≤ X ∧ X < 1000 ∧
    100 ≤ Y ∧ Y < 1000 ∧
    X + Y = 999 ∧
    1000 * X + Y = 6 * (1000 * Y + X) ∧
    (X - Y = 715 ∨ Y - X = 715) :=
by
  sorry

end difference_between_numbers_l280_280841


namespace probability_sum_greater_than_four_l280_280489

theorem probability_sum_greater_than_four : 
  let total_outcomes := 36 in
  let outcomes_sum_less_or_equal_4 := 6 in
  let prob_sum_less_or_equal_4 := outcomes_sum_less_or_equal_4 / total_outcomes in
  prob_sum_less_or_equal_4 = (1 : ℝ) / 6 → 
  (1 - prob_sum_less_or_equal_4) = (5 : ℝ) / 6 := 
by 
  intros total_outcomes outcomes_sum_less_or_equal_4 prob_sum_less_or_equal_4 h1,
  sorry

end probability_sum_greater_than_four_l280_280489


namespace gcd_fact_8_fact_6_sq_l280_280108

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def gcd (a b : ℕ) : ℕ := Nat.gcd a b

theorem gcd_fact_8_fact_6_sq : gcd (factorial 8) ((factorial 6)^2) = 11520 := by
  sorry

end gcd_fact_8_fact_6_sq_l280_280108


namespace inscribed_circle_radius_l280_280263

theorem inscribed_circle_radius (R r : ℝ) (hR : R = 18) (hr : r = 9) : ∃ x : ℝ, x = 8 := 
by
  use 8
  sorry


end inscribed_circle_radius_l280_280263


namespace arithmetic_sum_property_l280_280411

variable (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Condition: a_4 = 7
def condition := a 4 = 7

-- Definition of sum of the first n terms of an arithmetic sequence
def sum_first_n_terms (n : ℕ) := n * (a 1 + a n) / 2

-- Theorem statement: If a_4 = 7, then S_7 = 49
theorem arithmetic_sum_property (h : a 4 = 7) : S 7 = 49 := by
  sorry

end arithmetic_sum_property_l280_280411


namespace a_2_pow_100_value_l280_280313

theorem a_2_pow_100_value
  (a : ℕ → ℕ)
  (h1 : a 1 = 2)
  (h2 : ∀ n : ℕ, a (2 * n) = 3 * n * a n) :
  a (2^100) = 2^4852 * 3^4950 :=
by
  sorry

end a_2_pow_100_value_l280_280313


namespace probability_not_all_dice_same_l280_280886

theorem probability_not_all_dice_same :
  let total_outcomes := 6^5
  let same_number_outcomes := 6
  let probability_same_number := same_number_outcomes / total_outcomes
  let probability_not_same_number := 1 - probability_same_number
  probability_not_same_number = (1295 : ℚ) / 1296 :=
by
  sorry

end probability_not_all_dice_same_l280_280886


namespace length_of_train_l280_280038

noncomputable theory

-- Define the conditions
def speed_of_train_kmh : ℝ := 60
def speed_of_man_kmh : ℝ := 6
def time_seconds : ℝ := 18

-- Convert speeds from km/h to m/s
def speed_of_train_ms : ℝ := speed_of_train_kmh * 1000 / 3600
def speed_of_man_ms : ℝ := speed_of_man_kmh * 1000 / 3600

-- Calculate the relative speed of the train and the man
def relative_speed_ms : ℝ := speed_of_train_ms + speed_of_man_ms

-- The theorem to prove the length of the train
theorem length_of_train : relative_speed_ms * time_seconds = 330.12 :=
sorry

end length_of_train_l280_280038


namespace ordering_of_exponentiations_l280_280312

def a : ℕ := 3 ^ 34
def b : ℕ := 2 ^ 51
def c : ℕ := 4 ^ 25

theorem ordering_of_exponentiations : c < b ∧ b < a := by
  sorry

end ordering_of_exponentiations_l280_280312


namespace circumradius_inradius_inequality_l280_280320

theorem circumradius_inradius_inequality {ABC : Triangle} (R r : ℝ) 
  (hR : R = circumradius ABC) (hr : r = inradius ABC) :
  R ≥ 2 * r ∧ (R = 2 * r ↔ is_equilateral ABC) := 
sorry

end circumradius_inradius_inequality_l280_280320


namespace find_n_l280_280122

theorem find_n (n : ℤ) (h1 : 3 ≤ n) (h2 : n ≤ 11) (h3 : n ≡ 2023 [MOD 7]) : n = 7 :=
by {
  -- Proof by integer properties and modular arithmetic
  sorry
}

end find_n_l280_280122


namespace simplify_fraction_l280_280529

theorem simplify_fraction (h1 : 3.36 = 3 + 0.36) 
                          (h2 : 0.36 = (36 : ℚ) / 100) 
                          (h3 : (36 : ℚ) / 100 = 9 / 25) 
                          : 3.36 = 84 / 25 := 
by 
  rw [h1, h2, h3]
  norm_num
  rw [←Rat.add_div, show 3 = 75 / 25 by norm_num]
  norm_num
  
  sorry  -- This line can be safely removed when the proof is complete.

end simplify_fraction_l280_280529


namespace foci_coordinates_PA_max_min_l280_280195

-- Part 1
theorem foci_coordinates (m : ℝ) (hm : m > 1) (M A : ℝ × ℝ) (hM : M = (2, 0)) (hA : A = (2, 0)) :
  ∃ c : ℝ, c = real.sqrt 3 ∧ M = A ∧ 
  ((real.sqrt c, 0) ∨ (-real.sqrt c, 0) ∈ ({p : ℝ × ℝ | (p.1 / m)^2 + p.2^2 = 1})) :=
sorry

-- Part 2
theorem PA_max_min (m : ℝ) (hm : m = 3) (A : ℝ × ℝ) (hA : A = (2, 0)) :
  let PA := (λ x y, real.sqrt ((x - 2)^2 + y^2)) in
  ∀ x : ℝ, -3 ≤ x ∧ x ≤ 3 →
  ((x = -3 → PA x (real.sqrt (1 - (x^2 / 9))) = 5) ∧ 
   (x = 9 / 4 → PA x (real.sqrt (1 - (x^2 / 9))) = real.sqrt (1 / 2))) :=
sorry

end foci_coordinates_PA_max_min_l280_280195


namespace finite_geometric_mean_no_infinite_geometric_mean_l280_280393

-- Definition of geometric mean for a set of positive integers
def geometric_mean (s : Finset ℕ) : ℕ :=
  if h : s.nonempty then Nat.root (s.card * ∏ x in s, x) s.card else 0

-- First problem: There exists a finite set S_n such that the geometric mean of any finite subset is an integer
theorem finite_geometric_mean (n : ℕ) (hn : 0 < n) : 
  ∃ (S_n : Finset ℕ), S_n.card = n ∧ ∀ (s : Finset ℕ), s ⊆ S_n → ∃ k : ℕ, geometric_mean s = k :=
sorry

-- Second problem: There does not exist an infinite set such that the geometric mean of any finite subset is an integer
theorem no_infinite_geometric_mean : 
  ¬ ∃ (S : Set ℕ), S.infinite ∧ ∀ (s : Finset ℕ), s ⊆ S.to_finset → ∃ k : ℕ, geometric_mean s = k :=
sorry

end finite_geometric_mean_no_infinite_geometric_mean_l280_280393


namespace union_eq_universal_set_l280_280666

-- Define the sets U, M, and N
def U : Set ℕ := {2, 3, 4, 5, 6}
def M : Set ℕ := {3, 4, 5}
def N : Set ℕ := {2, 4, 6}

-- The theorem stating the desired equality
theorem union_eq_universal_set : M ∪ N = U := 
sorry

end union_eq_universal_set_l280_280666


namespace find_number_l280_280947

-- Define the necessary variables and constants
variables (N : ℝ) (h1 : (5 / 4) * N = (4 / 5) * N + 18)

-- State the problem as a theorem to be proved
theorem find_number : N = 40 :=
by
  sorry

end find_number_l280_280947


namespace find_meat_options_l280_280980

theorem find_meat_options :
  ∃ m : ℕ, 3 * (m - 1) * 5 + 3 * 1 * 4 = 57 ∧ m = 4 :=
by
  exists 4
  split
  { sorry }
  { rfl }

end find_meat_options_l280_280980


namespace bookstore_budget_problem_l280_280928

open_locale big_operators

def num_ways_to_spend_10_yuan (total_magazines : ℕ) (price2_magazines : ℕ) (price1_magazines : ℕ) (budget : ℕ) : ℕ :=
  if total_magazines = 11 ∧ price2_magazines = 8 ∧ price1_magazines = 3 ∧ budget = 10 then
    (nat.choose 8 5) + (nat.choose 8 4 * nat.choose 3 2)
  else 0

theorem bookstore_budget_problem :
  num_ways_to_spend_10_yuan 11 8 3 10 = 266 :=
sorry

end bookstore_budget_problem_l280_280928


namespace range_of_k_l280_280206

noncomputable theory

theorem range_of_k 
  (k : ℝ)
  (h1 : k > 0)
  (h2 : ∃ A B : ℝ × ℝ, A ≠ B ∧ (A.1^2 + A.2^2 = 4 ∧ B.1^2 + B.2^2 = 4)
    ∧ (A.1 - A.2 - k = 0 ∧ B.1 - B.2 - k = 0))
  (h3 : ∀ (A B : ℝ × ℝ), ∃ D : ℝ × ℝ, (D = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
    → ‖(A.1 + A.2, A.2 + A.1)‖ ≥ (√3 / 3) * ‖(A.1 - B.1, A.2 - B.2)‖) :
  sqrt 2 ≤ k ∧ k < 2 * sqrt 2 :=
by
  sorry

end range_of_k_l280_280206


namespace projections_cyclic_l280_280360

open EuclideanGeometry

theorem projections_cyclic
  (A B C P Q : Point)
  (hP : interior A B C P)
  (hReflections : ∀ (L M N : Line), reflection_over_bisector A P L → reflection_over_bisector B P M → reflection_over_bisector C P N → intersection_of_lines L M N Q) :
  cyclic_quadrilateral (projection P (side A B)) (projection P (side B C)) (projection P (side C A)) (projection Q (side A B)) (projection Q (side B C)) (projection Q (side C A)) :=
sorry

end projections_cyclic_l280_280360


namespace jett_profit_l280_280756

def purchase_price : ℕ := 600
def cost_per_day_food : ℕ := 20
def days : ℕ := 40
def cost_vaccination_deworming : ℕ := 500
def selling_price : ℕ := 2500

def total_cost : ℕ := purchase_price + days * cost_per_day_food + cost_vaccination_deworming
def profit : ℕ := selling_price - total_cost

theorem jett_profit : profit = 600 := by
  rw [profit, total_cost, purchase_price, cost_per_day_food, days, cost_vaccination_deworming, selling_price]
  sorry

end jett_profit_l280_280756


namespace base7_subtraction_l280_280639

noncomputable def base7_to_decimal (x : ℕ) : ℕ :=
  let digits := [5, 5, 2, 1] -- representing 1255_7
  in digits.enum_from 0 |>.map (λ (i, d), d * 7^i) |>.sum

noncomputable def base7_1255 : ℕ := base7_to_decimal 1255

noncomputable def base7_to_decimal_b (x : ℕ) : ℕ :=
  let digits := [2, 3, 4] -- representing 432_7
  in digits.enum_from 0 |>.map (λ (i, d), d * 7^i) |>.sum

noncomputable def base7_432 : ℕ := base7_to_decimal_b 432

noncomputable def base10_to_base7 (x : ℕ) : ℕ :=
  let digits := [1, 2, 5] -- representing 521_7
  in digits.enum_from 0 |>.map (λ (i, d), d * 7^i) |>.sum

theorem base7_subtraction : base7_to_decimal 1255 - base7_to_decimal_b 432 = base10_to_base7 262 := by
  sorry

end base7_subtraction_l280_280639


namespace tiffany_uploaded_7_pics_from_her_phone_l280_280424

theorem tiffany_uploaded_7_pics_from_her_phone
  (camera_pics : ℕ)
  (albums : ℕ)
  (pics_per_album : ℕ)
  (total_pics : ℕ)
  (h_camera_pics : camera_pics = 13)
  (h_albums : albums = 5)
  (h_pics_per_album : pics_per_album = 4)
  (h_total_pics : total_pics = albums * pics_per_album) :
  total_pics - camera_pics = 7 := by
  sorry

end tiffany_uploaded_7_pics_from_her_phone_l280_280424


namespace probability_not_all_same_l280_280873

/-- What is the probability that when we roll five fair 6-sided dice, they won't all show the same number? -/
theorem probability_not_all_same :
  let total_outcomes := 6^5 in
  let same_number_outcomes := 6 in
  let probability_all_same := same_number_outcomes / total_outcomes.to_real in
  1 - probability_all_same = (1295 : ℝ) / 1296 :=
by
  let total_outcomes := 6^5
  let same_number_outcomes := 6
  have probability_all_same := (same_number_outcomes : ℝ) / total_outcomes.to_real
  show 1 - probability_all_same = (1295 : ℝ) / 1296       
  sorry

end probability_not_all_same_l280_280873


namespace probability_of_root_l280_280342

noncomputable theory

open set real

def interval_a := Icc (-real.pi) real.pi
def interval_b := Icc (-real.pi) real.pi

def prob_has_root (a b : ℝ) : Prop :=
  a^2 + b^2 ≥ real.pi

theorem probability_of_root : 
  ∫∫ (a ∈ interval_a) (b ∈ interval_b), indicator (λ x, x ∈ {p : ℝ×ℝ | prob_has_root p.1 p.1 }) 1 (a, b)
  = (3 / 4) * (4 * real.pi^2) := 
  sorry

end probability_of_root_l280_280342


namespace minimize_sum_of_squares_of_roots_l280_280209

theorem minimize_sum_of_squares_of_roots :
  ∃ m : ℝ, (∀ x : ℝ, 6 * x^2 - 8 * x + m = 0 → real.discriminant 6 (-8) m > 0) 
    ∧ (∀ a b : ℝ, a + b = 8/6 ∧ a * b = m/6 → a^2 + b^2 = 8/9) 
    ∧ m = 8 / 3 := 
by sorry

end minimize_sum_of_squares_of_roots_l280_280209


namespace xyz_expression_l280_280072

theorem xyz_expression (x y z : ℝ) 
  (h1 : x^2 - y * z = 2)
  (h2 : y^2 - z * x = 2)
  (h3 : z^2 - x * y = 2) :
  x * y + y * z + z * x = -2 :=
sorry

end xyz_expression_l280_280072


namespace area_of_triangle_KBC_l280_280741

noncomputable def length_FE := 7
noncomputable def length_BC := 7
noncomputable def length_JB := 5
noncomputable def length_BK := 5

theorem area_of_triangle_KBC : (1 / 2 : ℝ) * length_BC * length_BK = 17.5 := by
  -- conditions: 
  -- 1. Hexagon ABCDEF is equilateral with each side of length s.
  -- 2. Squares ABJI and FEHG are formed outside the hexagon with areas 25 and 49 respectively.
  -- 3. Triangle JBK is equilateral.
  -- 4. FE = BC.
  sorry

end area_of_triangle_KBC_l280_280741


namespace algebraic_expression_zero_l280_280158

theorem algebraic_expression_zero (a b : ℝ) (h : a^2 + 2 * a * b + b^2 = 0) : 
  a * (a + 4 * b) - (a + 2 * b) * (a - 2 * b) = 0 :=
by
  sorry

end algebraic_expression_zero_l280_280158


namespace part_I_solution_part_II_solution_l280_280159

-- Define the functions f(x) and g(x)
def f (x : ℝ) : ℝ := |x + 3| + |x - 1|
def g (x : ℝ) (m : ℝ) : ℝ := -x^2 + 2 * m * x

-- Ⅰ: Prove the solution set of the inequality f(x) > 4
theorem part_I_solution (x : ℝ) : f(x) > 4 ↔ x < -3 ∨ x > 1 :=
  sorry

-- Ⅱ: Prove the range of m for which f(x1) ≥ g(x2) for any x1, x2
theorem part_II_solution (m : ℝ) : (∀ (x1 x2 : ℝ), f(x1) ≥ g(x2, m)) ↔ (m > -2 ∧ m < 2) :=
  sorry

end part_I_solution_part_II_solution_l280_280159


namespace axis_of_symmetry_l280_280384

theorem axis_of_symmetry (x : ℝ) : IsAxisOfSymmetry (λ x, Real.sin(2 * Real.pi * x - Real.pi / 3)) x ↔ x = 5 / 12 :=
by
  sorry

end axis_of_symmetry_l280_280384


namespace smallest_possible_median_l280_280507

-- Define the set and the conditions
def set_elements (x : Int) : List Int := [x, 3 * x, 4, 3, 7]

-- Function to calculate the median of a sorted list
def median (l : List Int) : Int :=
  let sorted_l := List.sort (· ≤ ·) l
  sorted_l.get (sorted_l.length / 2)

-- Define the statement to be proved
theorem smallest_possible_median (x : Int) : 
  median (set_elements x) = 3 := 
  sorry

end smallest_possible_median_l280_280507


namespace exists_zero_of_f_l280_280986

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x - 2

theorem exists_zero_of_f : ∃ c ∈ Ioo 0 1, f c = 0 :=
by
  -- proof will go here
  sorry

end exists_zero_of_f_l280_280986


namespace inscribed_circle_radius_l280_280261

theorem inscribed_circle_radius (R r : ℝ) (hR : R = 18) (hr : r = 9) : ∃ x : ℝ, x = 8 := 
by
  use 8
  sorry


end inscribed_circle_radius_l280_280261


namespace days_in_first_quarter_2010_l280_280387

theorem days_in_first_quarter_2010 : 
  let not_leap_year := ¬ (2010 % 4 = 0)
  let days_in_february := 28
  let days_in_january_and_march := 31
  not_leap_year → days_in_february = 28 → days_in_january_and_march = 31 → (31 + 28 + 31 = 90)
:= 
sorry

end days_in_first_quarter_2010_l280_280387


namespace problem_statement_l280_280910

noncomputable def question := ∀ x : ℝ, sin(x)^9 * cos(x) - cos(x)^9 * sin(x) = sin(4 * x)

theorem problem_statement : question := 
by
  sorry

end problem_statement_l280_280910


namespace find_n_l280_280985

theorem find_n (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 6) (h3 : n ≡ -2890 [MOD 7]) : n = 6 :=
by
  sorry

end find_n_l280_280985


namespace chess_tournament_games_l280_280564

theorem chess_tournament_games (n : ℕ) (h : n = 10) : ∃ g : ℕ, g = n * (n - 1) / 2 ∧ g = 45 :=
by {
  use n * (n - 1) / 2,
  split,
  { refl },
  { rw h,
    norm_num, }
}

end chess_tournament_games_l280_280564


namespace election_vote_ratio_l280_280784

theorem election_vote_ratio : 
  (let voters_total_area : ℕ := 100000 in
   let percentage_votes_won : ℝ := 0.70 in
   let total_votes_won : ℕ := 210000 in
   let votes_first_area : ℕ := (percentage_votes_won * voters_total_area).toNat in
   let votes_remaining_area : ℕ := total_votes_won - votes_first_area in
   (votes_remaining_area / votes_first_area) = 2) :=
by 
  sorry

end election_vote_ratio_l280_280784


namespace average_age_of_women_is_29_l280_280918

-- Let A be the average age of the 8 men
variable (A : ℝ)

-- The two men replaced have ages 20 and 22 respectively
variable (age_man1 age_man2 : ℝ)
-- The ages of the two women replacing the men
variable (age_woman1 age_woman2 : ℝ)

-- The average age increase condition
variable (average_increase : ℝ)
-- The combined age of the two women is 58
variable (combined_age_women : ℝ := age_woman1 + age_woman2)
-- The average age of the two women
def average_age_women := combined_age_women / 2

-- The proof problem
theorem average_age_of_women_is_29
    (h1 : age_man1 = 20)
    (h2 : age_man2 = 22)
    (h3 : average_increase = 2)
    (h4 : combined_age_women = 58) :
    average_age_women = 29 :=
by
  sorry  -- Proof is skipped here

end average_age_of_women_is_29_l280_280918


namespace slope_of_bisecting_line_l280_280973

theorem slope_of_bisecting_line (m n : ℕ) (hmn : Int.gcd m n = 1) : 
  let p1 := (20, 90)
  let p2 := (20, 228)
  let p3 := (56, 306)
  let p4 := (56, 168)
  -- Define conditions for line through origin (x = 0, y = 0) bisecting the parallelogram
  let b := 135 / 19
  let slope := (90 + b) / 20
  -- The slope must be equal to 369/76 (m = 369, n = 76)
  m = 369 ∧ n = 76 → m + n = 445 := by
  intro m n hmn
  let p1 := (20, 90)
  let p2 := (20, 228)
  let p3 := (56, 306)
  let p4 := (56, 168)
  let b := 135 / 19
  let slope := (90 + b) / 20
  sorry

end slope_of_bisecting_line_l280_280973


namespace sequence_sum_l280_280622

theorem sequence_sum (n : ℕ) : 
  let a := λ k, 2 + 3 * (k * (k - 1) / 2)
  in (∑ k in finset.range n.succ, a k) = n * (n + 1) * (n + 1) / 4 + 2 * n :=
by 
  let a := λ k, 2 + 3 * (k * (k - 1) / 2)
  sorry


end sequence_sum_l280_280622


namespace AO_BO_parallel_MN_l280_280306

open EuclideanGeometry

variables {A B C D M N P O_A O_B : Point ℝ}

def is_midpoint (X Y Z : Point ℝ) : Prop := dist X Y = dist X Z

def is_circumcenter (P X Y Z : Point ℝ) : Prop :=
  ∃ r, ∀ Q, dist P Q = r ↔ (dist P Q = dist P X ∧ dist P Q = dist P Y ∧ dist P Q = dist P Z)

axiom geom_conditions :
  (∃ (semicircle : Set (Point ℝ)),
     A ∈ semicircle ∧
     B ∈ semicircle ∧ 
     C ∈ semicircle ∧ 
     D ∈ semicircle ∧  
     ¬(B = C ∨ (∃ line, A, D ∈ line ∧ B, C ∈ line))) ∧
  is_midpoint A C M ∧ 
  is_midpoint B D N ∧ 
  is_midpoint C D P ∧
  is_circumcenter O_A A C P ∧
  is_circumcenter O_B B D P

theorem AO_BO_parallel_MN : Parallel (Line.mk O_A O_B) (Line.mk M N) :=
by
  sorry

end AO_BO_parallel_MN_l280_280306


namespace value_of_second_type_coin_l280_280018

-- Define the conditions
variables (x : ℝ)

-- Define the individual values of the coins
def one_rupee_value : ℝ := 20 * 1
def second_type_value : ℝ := 20 * x
def twenty_five_paise_value : ℝ := 20 * 0.25

-- Proof goal: verifying the value of "x"
theorem value_of_second_type_coin :
  (one_rupee_value + second_type_value + twenty_five_paise_value = 35) -> x = 0.5 :=
by
  intro h
  sorry

end value_of_second_type_coin_l280_280018


namespace tangent_ratio_l280_280709

open Set

variables {O_1 O_2 : Type}  -- The circles
variables {A X P : O_1} {B Y Q : O_2} -- Points on the circles
variables {r_1 r_2 : ℝ} -- Radii of the circles
variables {AB XY : Set (O_1 × O_2)} -- Common internal tangents
variables (h_tangent : externally_tangent O_1 O_2)
variables (h_radius : r_1 < r_2)
variables (h_tangent_points_A : A ∈ circle O_1 r_1)
variables (h_tangent_points_X : X ∈ circle O_1 r_1)
variables (h_tangent_points_B : B ∈ circle O_2 r_2)
variables (h_tangent_points_Y : Y ∈ circle O_2 r_2)
variables (h_circle_diameter_A : is_diameter AB A B)
variables (h_circle_diameter_P : intersects_circle O_1 P AB)
variables (h_circle_diameter_Q : intersects_circle O_2 Q AB)
variables (h_angle : ∠O_1 A P + ∠O_2 B Q = 180°)

theorem tangent_ratio (PX QY : ℝ) :
  PX / QY = Real.sqrt (r_1 / r_2) :=
sorry

end tangent_ratio_l280_280709


namespace smallest_positive_period_min_max_values_in_interval_l280_280701

noncomputable def f (x : ℝ) : ℝ := (1/2) * Real.sin (2 * x) + Real.sqrt 3 * (Real.cos x) ^ 2 - Real.sqrt 3 / 2

theorem smallest_positive_period :
  ∃ T > 0, ∀ x : ℝ, f (x + T) = f x ∧ T = π :=
by sorry

theorem min_max_values_in_interval :
  let interval := Set.Icc (- (π / 12)) (5 * (π / 12)) in
  let min_val := -(1/2) in
  let max_val := 1 in
  (∀ x ∈ interval, f x ≥ min_val) ∧
  (∀ x ∈ interval, f x ≤ max_val) :=
by sorry

end smallest_positive_period_min_max_values_in_interval_l280_280701


namespace ellipse_equation_correct_distance_to_line_correct_l280_280688

noncomputable def ellipse_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_ab : a > b) := 
    (4 : ℝ) * (a^2) = a^2 ∧ a^2 = b^2 + (a / 2)^2

noncomputable def distance_to_line (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_perpendicular : a * b = 0) :=
    (sqrt (21) * 2) / 7

theorem ellipse_equation_correct (a b : ℝ) (ha : a = 2) (hb : b = sqrt 3) : 
    ellipse_equation a b (by norm_num) (by norm_num) (by norm_num) :=
by
  sorry

theorem distance_to_line_correct (a b : ℝ) (h : a > b) :
    distance_to_line a b h (by norm_num) = (2 * sqrt 21) / 7 :=
by
  sorry

end ellipse_equation_correct_distance_to_line_correct_l280_280688


namespace tetrahedra_overlap_volume_greater_than_half_l280_280283

noncomputable def tetrahedron_volume (a : ℝ) : ℝ :=
  a^3 / (6 * real.sqrt 2)

theorem tetrahedra_overlap_volume_greater_than_half (a : ℝ) (V_overlap : ℝ) : 
  a = real.sqrt 6 →
  V_overlap > (1 / 2) * tetrahedron_volume a :=
by
  intro ha
  let V := tetrahedron_volume a
  have ha' : a^3 = (real.sqrt 6)^3, from ha ▸ rfl
  let V_correct := ((real.sqrt 6)^3) / (6 * real.sqrt 2)
  sorry

end tetrahedra_overlap_volume_greater_than_half_l280_280283


namespace f_neg_one_eq_three_l280_280151

noncomputable def f : ℝ → ℝ 
| x => if x < 6 then f(x + 3) else Real.log x / Real.log 2

theorem f_neg_one_eq_three : f (-1) = 3 := 
by
  sorry

end f_neg_one_eq_three_l280_280151


namespace Bill_main_project_hours_l280_280965

noncomputable def main_project_hours (work_days total_daily_hours task1_hours task2_hours 
                                      naps_day1_hours naps_day2_hours naps_day3_hours naps_day4_hours : ℕ) : ℕ :=
  let total_available_hours := work_days * total_daily_hours
  let total_small_tasks_hours := task1_hours + task2_hours
  let total_naps_hours := naps_day1_hours + naps_day2_hours + naps_day3_hours + naps_day4_hours
  in total_available_hours - total_small_tasks_hours - total_naps_hours

theorem Bill_main_project_hours :
  main_project_hours 4 8 6 3 (2 + 1) (1.5 + 2 + 1) 3 (1.5 + 1.5) = 9.5 := by
  sorry

end Bill_main_project_hours_l280_280965


namespace units_digit_of_factorial_sum_l280_280650

theorem units_digit_of_factorial_sum : 
  (1 + 2 + 6 + 4) % 10 = 3 := sorry

end units_digit_of_factorial_sum_l280_280650


namespace a_and_b_together_30_days_l280_280722

variable (R_a R_b : ℝ)

-- Conditions
axiom condition1 : R_a = 3 * R_b
axiom condition2 : R_a * 40 = (R_a + R_b) * 30

-- Question: prove that a and b together can complete the work in 30 days.
theorem a_and_b_together_30_days (R_a R_b : ℝ) (condition1 : R_a = 3 * R_b) (condition2 : R_a * 40 = (R_a + R_b) * 30) : true :=
by
  sorry

end a_and_b_together_30_days_l280_280722


namespace DE_perp_EF_l280_280305

open EuclideanGeometry

variables {A B C D E F : Point}

-- Define conditions
variables (h₁ : ∃ (D : Point), ∠ A D C = 30 ∧ ∠ D C A = 30)
variables (h₂ : ∠ D B A = 60)
variables (h₃ : Midpoint E B C)
variables (h₄ : Collinear A F C ∧ dist A F = 2 * dist F C)

theorem DE_perp_EF (h₁ : ∃ (D : Point), ∠ A D C = 30 ∧ ∠ D C A = 30)
                   (h₂ : ∠ D B A = 60)
                   (h₃ : Midpoint E B C)
                   (h₄ : Collinear A F C ∧ dist A F = 2 * dist F C)
                   : ∠ D E F = 90 := 
by
  sorry

end DE_perp_EF_l280_280305


namespace number_of_rows_in_wall_l280_280728

theorem number_of_rows_in_wall :
    ∃ n : ℕ, (∃ a d s : ℕ, a = 18 ∧ d = -1 ∧ s = 100 ∧ (s = n * (2 * a + (n - 1) * d) / 2)) ∧ n = 8 :=
by
    sorry

end number_of_rows_in_wall_l280_280728


namespace triangle_altitude_l280_280367

theorem triangle_altitude (A b h : ℝ) (hA : A = 900) (hb : b = 45) : h = 40 :=
by
  -- Definitions: A = 1/2 * b * h and given conditions
  have h_area_formula : A = (1 / 2) * b * h,
  { sorry },
  -- Substitute given values into the area formula
  have h_substitute : 900 = (1 / 2) * 45 * h,
  { sorry },
  -- Solve for h
  sorry

end triangle_altitude_l280_280367


namespace slips_with_3_l280_280810

-- Definitions of the conditions
def num_slips : ℕ := 15
def expected_value : ℚ := 5.4

-- Theorem statement
theorem slips_with_3 (y : ℕ) (t : ℕ := num_slips) (E : ℚ := expected_value) :
  E = (3 * y + 8 * (t - y)) / t → y = 8 :=
by
  sorry

end slips_with_3_l280_280810


namespace desired_average_grade_l280_280300

-- Define the grades for the three tests
def grade1 : ℝ := 95
def grade2 : ℝ := 80
def grade3 : ℝ := 95

-- Calculate the desired average grade
def avg_grade : ℝ := (grade1 + grade2 + grade3) / 3

-- The Lean statement to prove the desired average grade
theorem desired_average_grade : avg_grade = 90 := by
  simp [avg_grade]
  sorry

end desired_average_grade_l280_280300


namespace inscribed_circle_radius_l280_280273

theorem inscribed_circle_radius (R r : ℝ) (hR : R = 18) (hr : r = 9) :
    ∃ x : ℝ, (∀ P Px OP O1P : ℝ, Px = sqrt((R - x) ^ 2 - x ^ 2) ∧ O1P = sqrt((r + x) ^ 2 - x ^ 2)
                 ∧ Px + r = O1P) ∧ x = 8 :=
begin
  sorry
end

end inscribed_circle_radius_l280_280273


namespace correct_systematic_sampling_l280_280165

-- Defining the parameters of the problem
def num_bottles : ℕ := 60
def num_to_select : ℕ := 6
def interval : ℕ := num_bottles / num_to_select

-- Defining the selection according to systematic sampling
def systematic_sampling (start : ℕ) (interval : ℕ) (num : ℕ) : list ℕ :=
  (list.range num).map (λ i, start + i * interval)

-- Given values for this problem
def start_bottle : ℕ := 5
def selected_bottles := [5, 10, 15, 20, 25, 30]

-- Lean statement to show that given selection matches systematic sampling criteria
theorem correct_systematic_sampling:
  systematic_sampling start_bottle interval num_to_select = selected_bottles :=
by simp [systematic_sampling, list.range, map, interval, num_to_select, start_bottle]


end correct_systematic_sampling_l280_280165


namespace duty_schedule_impossibility_l280_280252

theorem duty_schedule_impossibility :
  ∀ (n : ℕ), n = 100 → (∃ a b c : Fin 100, (a ≠ b ∧ a ≠ c ∧ b ≠ c)) →
  ¬ (∀ p : Fin 100, ∀ q : Fin 100, p ≠ q → (∃! r : Fin 100, r ≠ p ∧ r ≠ q ∧ duty_tohether p q r)) :=
by
  intro n hn h3
  simp [hn] at h3
  sorry

end duty_schedule_impossibility_l280_280252


namespace exists_equilateral_triangle_same_color_l280_280084

-- Define a type for colors
inductive Color
| red : Color
| blue : Color

-- Define our statement
-- Given each point in the plane is colored either red or blue,
-- there exists an equilateral triangle with vertices of the same color.
theorem exists_equilateral_triangle_same_color (coloring : ℝ × ℝ → Color) : 
  ∃ (p₁ p₂ p₃ : ℝ × ℝ), 
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₂ ≠ p₃ ∧ 
    dist p₁ p₂ = dist p₂ p₃ ∧ dist p₂ p₃ = dist p₃ p₁ ∧ 
    (coloring p₁ = coloring p₂ ∧ coloring p₂ = coloring p₃) :=
by
  sorry

end exists_equilateral_triangle_same_color_l280_280084


namespace equation_of_line_C_parallel_AB_area_of_triangle_OMN_l280_280174

-- Definitions of points
structure Point where
  x : ℚ
  y : ℚ

def A : Point := {x := 1, y := 4}
def B : Point := {x := 3, y := 2}
def C : Point := {x := 1, y := 1}

-- Slope calculation
def slope (P Q : Point) : ℚ := (P.y - Q.y) / (P.x - Q.x)

-- Line equation in form Ax + By + C = 0 for line passing through point with given slope
def line_eq (P : Point) (m : ℚ) : ℚ → ℚ → Prop := fun x y => m * (x - P.x) = y - P.y

-- Midpoint calculation
def midpoint (P Q : Point) : Point := {x := (P.x + Q.x) / 2, y := (P.y + Q.y) / 2}

-- Check perpendicular slope
def perpendicular_slope (m : ℚ) := -1 / m

-- Intersection points
def M : Point := {x := -1, y := 0}
def N : Point := {x := 0, y := 1}

-- Triangle area calculation
def triangle_area (O M N : Point) : ℚ := (1 / 2) * (1) * (1)  -- OM and ON lengths are both 1 and form a right triangle

theorem equation_of_line_C_parallel_AB : 
  ∃ k, ∃ a b c : ℚ, a * (C.x := 1) + b * (C.y := 1) + c = 0 :=
begin
  -- This is the formal statement in Lean, proof will follow steps to show x + y - 2 = 0
  sorry
end

theorem area_of_triangle_OMN :
  triangle_area {x := 0, y := 0} M N = 1 / 2 :=
begin
  -- This is the formal statement in Lean, proof will follow steps to calculate area
  sorry
end

end equation_of_line_C_parallel_AB_area_of_triangle_OMN_l280_280174


namespace total_thread_needed_l280_280352

def keychain_length : Nat := 12
def friends_in_classes : Nat := 10
def multiplier_for_club_friends : Nat := 2
def thread_per_class_friend : Nat := 16
def thread_per_club_friend : Nat := 20

theorem total_thread_needed :
  10 * thread_per_class_friend + (10 * multiplier_for_club_friends) * thread_per_club_friend = 560 := by
  sorry

end total_thread_needed_l280_280352


namespace conjugate_of_z_l280_280325

def i : ℂ := complex.I

def z : ℂ := (i - 1) * -i

theorem conjugate_of_z : complex.conj z = 1 - i := by
  sorry

end conjugate_of_z_l280_280325


namespace averageTemperature_is_99_l280_280601

-- Define the daily temperatures
def tempSunday : ℝ := 99.1
def tempMonday : ℝ := 98.2
def tempTuesday : ℝ := 98.7
def tempWednesday : ℝ := 99.3
def tempThursday : ℝ := 99.8
def tempFriday : ℝ := 99
def tempSaturday : ℝ := 98.9

-- Define the number of days
def numDays : ℝ := 7

-- Define the total temperature
def totalTemp : ℝ := tempSunday + tempMonday + tempTuesday + tempWednesday + tempThursday + tempFriday + tempSaturday

-- Define the average temperature
def averageTemp : ℝ := totalTemp / numDays

-- The theorem to prove
theorem averageTemperature_is_99 : averageTemp = 99 := by
  sorry

end averageTemperature_is_99_l280_280601


namespace solution_set_log_inequality_l280_280829

theorem solution_set_log_inequality (x : ℝ) : 
  (-1 < x ∧ x ≤ 0) ↔ (lg (x + 1) ≤ 0) :=
sorry

-- The following ensures the conditions needed for the domain:
def log_domain (x : ℝ) : Prop := x + 1 > 0

end solution_set_log_inequality_l280_280829


namespace num_solutions_in_S_l280_280321

open Rat

theorem num_solutions_in_S :
  let S := {x : ℚ | 0 < x ∧ x < 5 / 8}
  let f (qp : ℚ) := (qp.num.add 1) / qp.denom in
  (∃ p q : ℕ, p ≠ 0 ∧ coprime p q ∧ q / p ∈ S ∧ f (q / p) = 2 / 3) → 
  {qp : ℚ | ∃ p q : ℕ, p ≠ 0 ∧ coprime p q ∧ q / p ∈ S ∧ f (q / p) = 2 / 3}.size = 5 :=
by
  sorry

end num_solutions_in_S_l280_280321


namespace rectangular_field_area_l280_280337

-- Given a rectangle with one side 4 meters and diagonal 5 meters, prove that its area is 12 square meters.
theorem rectangular_field_area
  (w l d : ℝ)
  (h_w : w = 4)
  (h_d : d = 5)
  (h_pythagoras : w^2 + l^2 = d^2) :
  w * l = 12 := 
by
  sorry

end rectangular_field_area_l280_280337


namespace find_x_l280_280718

theorem find_x (x : ℕ) (h : real.cbrt (5 + real.sqrt x) = 4) : x = 3481 :=
sorry

end find_x_l280_280718


namespace find_a_plus_b_l280_280390

noncomputable theory

variable {a b : ℝ}

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

def f (x : ℝ) : ℝ := a * x^2 + b * x + 1

def domain_condition : Prop := ∃ x y : ℝ, x = 2 * a ∧ y = 1 - a

theorem find_a_plus_b (hf : is_even_function f) (hd : domain_condition) : a + b = -1 :=
by
  sorry

end find_a_plus_b_l280_280390


namespace positive_whole_numbers_with_cube_root_less_than_15_l280_280225

theorem positive_whole_numbers_with_cube_root_less_than_15 :
  { n : ℕ // 1 ≤ n ∧ n < 3375 }.card = 3374 :=
by
  -- Introduction of the natural number and the required conditions
  sorry

end positive_whole_numbers_with_cube_root_less_than_15_l280_280225


namespace max_ab_l280_280198

noncomputable def f (a x : ℝ) : ℝ := -a * Real.log x + (a + 1) * x - (1/2) * x^2

theorem max_ab (a b : ℝ) (h₁ : 0 < a)
  (h₂ : ∀ x, f a x ≥ - (1/2) * x^2 + a * x + b) : 
  ab ≤ ((Real.exp 1) / 2) :=
sorry

end max_ab_l280_280198


namespace allen_gave_delivery_man_100_dollars_l280_280046

theorem allen_gave_delivery_man_100_dollars
  (pizza_cost_per_box : ℝ)
  (num_boxes : ℕ)
  (tip_fraction : ℝ)
  (change_received : ℝ)
  (h1 : pizza_cost_per_box = 7)
  (h2 : num_boxes = 5)
  (h3 : tip_fraction = 1/7)
  (h4 : change_received = 60) :
  let total_pizza_cost := num_boxes * pizza_cost_per_box,
      tip := tip_fraction * total_pizza_cost,
      total_paid := total_pizza_cost + tip in
  total_paid + change_received = 100 :=
by
  -- The proof will be skipped.
  sorry

end allen_gave_delivery_man_100_dollars_l280_280046


namespace rays_dog_daily_walk_l280_280347

theorem rays_dog_daily_walk :
  ∀ (walks_to_park walks_to_school walks_home trips_per_day : ℕ),
    walks_to_park = 4 →
    walks_to_school = 7 →
    walks_home = 11 →
    trips_per_day = 3 →
    trips_per_day * (walks_to_park + walks_to_school + walks_home) = 66 :=
by
  intros walks_to_park walks_to_school walks_home trips_per_day
  intro h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  exact rfl

end rays_dog_daily_walk_l280_280347


namespace sum_of_seven_step_palindromes_l280_280663

def is_palindrome (n : ℕ) : Prop :=
  let digits := (toString n).data;
  digits = digits.reverse

def reverse_and_add (n : ℕ) : ℕ :=
  let rev := String.toNat! ((toString n).data.reverse.asString);
  n + rev

def steps_to_palindrome (n : ℕ) : ℕ :=
  let rec steps_aux (m : ℕ) (count : ℕ) : ℕ :=
    if is_palindrome m then count
    else steps_aux (reverse_and_add m) (count + 1)
  steps_aux n 0

def is_valid_n (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 200 ∧ ¬is_palindrome n ∧ steps_to_palindrome n = 7

theorem sum_of_seven_step_palindromes : 
  (Finset.filter is_valid_n (Finset.range 200)) 
  .sum = 1304 :=
by
  sorry

end sum_of_seven_step_palindromes_l280_280663


namespace inscribed_circle_radius_l280_280266

theorem inscribed_circle_radius (R r x : ℝ) (hR : R = 18) (hr : r = 9) :
  x = 8 :=
sorry

end inscribed_circle_radius_l280_280266


namespace one_minus_repeating_three_l280_280089

theorem one_minus_repeating_three : ∀ b : ℚ, b = 1 / 3 → 1 - b = 2 / 3 :=
by
  intro b hb
  rw [hb]
  norm_num

end one_minus_repeating_three_l280_280089


namespace inverse_proportion_inequality_l280_280173

variable (x1 x2 k : ℝ)
variable (y1 y2 : ℝ)

theorem inverse_proportion_inequality (h1 : x1 < 0) (h2 : 0 < x2) (hk : k < 0)
  (hy1 : y1 = k / x1) (hy2 : y2 = k / x2) : y2 < 0 ∧ 0 < y1 := 
by sorry

end inverse_proportion_inequality_l280_280173


namespace probability_of_one_of_each_color_l280_280148

-- Definitions based on the conditions
def total_marbles : ℕ := 12
def marbles_of_each_color : ℕ := 3
def number_of_selected_marbles : ℕ := 4

-- Calculation based on problem requirements
def total_ways_to_choose_marbles : ℕ := Nat.choose total_marbles number_of_selected_marbles
def favorable_ways_to_choose : ℕ := marbles_of_each_color ^ number_of_selected_marbles

-- The main theorem to prove the probability
theorem probability_of_one_of_each_color :
  (favorable_ways_to_choose : ℚ) / total_ways_to_choose = 9 / 55 := by
  sorry

end probability_of_one_of_each_color_l280_280148


namespace stephanie_remaining_payment_l280_280356

open Real

def electricity_bill_total : ℝ := 60
def electricity_bill_paid : ℝ := 60

def gas_bill_total : ℝ := 40
def gas_bill_paid : ℝ := (3/4) * gas_bill_total
def additional_gas_payment : ℝ := 5

def water_bill_total : ℝ := 40
def water_bill_paid : ℝ := (1/2) * water_bill_total

def internet_bill_total : ℝ := 25
def internet_bill_payment : ℝ := 5
def internet_bill_payments_number : ℝ := 4

def remaining_payment (total paid additional_paid : ℝ) : ℝ :=
  total - (paid + additional_paid)

def total_remaining_payment : ℝ :=
  remaining_payment electricity_bill_total electricity_bill_paid 0 +
  remaining_payment gas_bill_total gas_bill_paid additional_gas_payment +
  remaining_payment water_bill_total water_bill_paid 0 +
  remaining_payment internet_bill_total (internet_bill_payment * internet_bill_payments_number) 0

theorem stephanie_remaining_payment : total_remaining_payment = 30 := by
  sorry

end stephanie_remaining_payment_l280_280356


namespace five_dice_not_all_same_probability_l280_280882

noncomputable def probability_not_all_same : ℚ :=
  let total_outcomes := (6:ℚ) ^ 5
  let favorable_outcomes := (6:ℚ)
  1 - (favorable_outcomes / total_outcomes)

theorem five_dice_not_all_same_probability :
  probability_not_all_same = 1295 / 1296 :=
by
  unfold probability_not_all_same
  norm_cast
  simp
  sorry

end five_dice_not_all_same_probability_l280_280882


namespace rays_dog_daily_walk_l280_280346

theorem rays_dog_daily_walk :
  ∀ (walks_to_park walks_to_school walks_home trips_per_day : ℕ),
    walks_to_park = 4 →
    walks_to_school = 7 →
    walks_home = 11 →
    trips_per_day = 3 →
    trips_per_day * (walks_to_park + walks_to_school + walks_home) = 66 :=
by
  intros walks_to_park walks_to_school walks_home trips_per_day
  intro h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  exact rfl

end rays_dog_daily_walk_l280_280346


namespace tree_in_large_graph_l280_280231
open Finset

/-- A tree is defined as a connected acyclic graph --/
def is_tree (T : SimpleGraph V) := T.is_connected ∧ T.no_cycles

/-- The minimum degree of a graph G is at least |T| - 1 --/
def min_degree_at_least (G : SimpleGraph V) (k : ℕ) := 
  ∀ v, G.degree v ≥ k

/-- Main theorem statement --/
theorem tree_in_large_graph {V : Type} {T G : SimpleGraph V} (hT : is_tree T)
  (hG : min_degree_at_least G (T.vertex_count - 1)) :
  T.subgraph_isomorphic G := 
sorry

end tree_in_large_graph_l280_280231


namespace probability_sum_greater_than_four_is_5_over_6_l280_280436

-- Define the sample space for two dice.
def sample_space : Finset (ℕ × ℕ) :=
  Finset.product (Finset.range 6) (Finset.range 6) 

-- Define the event where the sum is greater than four.
def event_sum_greater_than_four : Finset (ℕ × ℕ) :=
  sample_space.filter (λ p, p.1 + p.2 > 4)

-- Compute the probability of the event.
def probability_sum_greater_than_four : ℚ :=
  (event_sum_greater_than_four.card : ℚ) / (sample_space.card : ℚ)

-- Theorem to be proven
theorem probability_sum_greater_than_four_is_5_over_6 :
  probability_sum_greater_than_four = 5 / 6 :=
by
  -- Proof would go here
  sorry

end probability_sum_greater_than_four_is_5_over_6_l280_280436


namespace acute_angles_of_right_triangle_l280_280623

noncomputable def acute_angles_right_triangle (a b c : ℝ) (α β : ℝ) : Prop :=
  (a^2 + b^2 = c^2) ∧
  (a = c * sin α) ∧ (b = c * cos β) ∧
  (sin (α + β) = 1) ∧ 
  (2 * (α + β) = π / 3) 

theorem acute_angles_of_right_triangle (a b c : ℝ) (α β : ℝ) :
  acute_angles_right_triangle a b c α β → 
  α = π / 6 ∧ β = π / 3 :=
begin
  sorry
end

end acute_angles_of_right_triangle_l280_280623


namespace area_of_triangle_is_39_l280_280098

def point := ℝ × ℝ

def A : point := (-2, 3)
def B : point := (8, -1)
def C : point := (10, 6)

def vector (p q : point) : point :=
  (q.1 - p.1, q.2 - p.2)

def cross_product (v w : point) : ℝ :=
  v.1 * w.2 - v.2 * w.1

def area_of_triangle (A B C : point) : ℝ :=
  |cross_product (vector C A) (vector C B)| / 2

theorem area_of_triangle_is_39 : area_of_triangle A B C = 39 := 
by {
  sorry
}

end area_of_triangle_is_39_l280_280098


namespace tan_4x_eq_cos_x_has_9_solutions_l280_280227

theorem tan_4x_eq_cos_x_has_9_solutions :
  ∃ (s : Finset ℝ), s.card = 9 ∧ ∀ x ∈ s, (0 ≤ x ∧ x ≤ 2 * Real.pi) ∧ (Real.tan (4 * x) = Real.cos x) :=
sorry

end tan_4x_eq_cos_x_has_9_solutions_l280_280227


namespace sequence_problem_l280_280742

theorem sequence_problem 
  (a : ℕ → ℕ) 
  (h1 : a 1 = 5) 
  (h2 : ∀ n : ℕ, a (n + 1) - a n = 3 + 4 * (n - 1)) : 
  a 50 = 4856 :=
sorry

end sequence_problem_l280_280742


namespace probability_not_all_dice_same_l280_280885

theorem probability_not_all_dice_same :
  let total_outcomes := 6^5
  let same_number_outcomes := 6
  let probability_same_number := same_number_outcomes / total_outcomes
  let probability_not_same_number := 1 - probability_same_number
  probability_not_same_number = (1295 : ℚ) / 1296 :=
by
  sorry

end probability_not_all_dice_same_l280_280885


namespace smaller_wheel_rotation_time_l280_280849

-- Definitions and given conditions
def larger_wheel_rotations_per_minute : ℕ := 10
def larger_wheel_radius (R_B : ℝ) : ℝ := 3 * R_B

-- The time for one rotation of the larger wheel in seconds
def time_per_revolution_larger_wheel : ℝ := 60 / larger_wheel_rotations_per_minute

-- The time for one rotation of the smaller wheel
def time_per_revolution_smaller_wheel (R_B : ℝ) : ℝ := time_per_revolution_larger_wheel / 3

-- Theorem statement
theorem smaller_wheel_rotation_time (R_B : ℝ) : time_per_revolution_smaller_wheel R_B = 2 :=
by
  -- The proof is omitted
  sorry

end smaller_wheel_rotation_time_l280_280849


namespace inequality_solution_set_l280_280152

theorem inequality_solution_set (a x : ℝ) (h : 4 ^ a = 2 ^ (a + 2)) : (a^(2*x + 1) > a^(x - 1)) ↔ (x > -2) :=
by
  sorry

end inequality_solution_set_l280_280152


namespace area_of_triangle_def_equals_42_78_l280_280957

noncomputable def triangle_def_area (side_abc : ℝ) (distance_ad : ℝ) : ℝ :=
let altitude_abc := (Real.sqrt 3 / 2) * side_abc in
let circumradius_abc := (2 / 3) * altitude_abc in
let theta := 2 * Real.pi * (5 / 6) in
let theta' := theta / 3 in
let side_def' := 2 * circumradius_abc * Real.sin (theta' / 2) in
(Real.sqrt 3 / 4) * side_def' ^ 2

theorem area_of_triangle_def_equals_42_78 :
  triangle_def_area 6 1 ≈ 42.78 :=
by
  sorry

end area_of_triangle_def_equals_42_78_l280_280957


namespace average_of_all_results_l280_280813
open_locale big_operators

noncomputable def avg_all_results (n1 n2 s1 s2 : ℝ) :=
  (s1 * n1 + s2 * n2) / (n1 + n2)

theorem average_of_all_results :
  avg_all_results 45 25 25 45 = 32.14 :=
by
  -- conditions
  let avg1 := 25
  let avg2 := 45
  let n1 := 45
  let n2 := 25
  -- calculations
  let total1 := n1 * avg1
  let total2 := n2 * avg2
  let combined_total := total1 + total2
  let total_results := n1 + n2
  let combined_avg := combined_total / total_results
  -- proof
  have h : combined_avg = 2250 / 70 := by
    calc
      combined_avg = (total1 + total2) / total_results : rfl
      ... = 2250 / 70 : by simp [total1, total2, combined_total, total_results]
  exact h.symm.trans (by norm_num)

end average_of_all_results_l280_280813


namespace probability_sum_greater_than_four_l280_280482

theorem probability_sum_greater_than_four :
  let all_outcomes := (Fin 6) × (Fin 6)
  let favorable_outcomes := {p : Fin 6 × Fin 6 | (p.1.val + 1) + (p.2.val + 1) > 4}
  (favorable_outcomes.card : ℚ) / ((Fin 6 × Fin 6).card : ℚ) = 5 / 6 :=
by
  sorry

end probability_sum_greater_than_four_l280_280482


namespace average_of_last_three_numbers_l280_280378

theorem average_of_last_three_numbers (nums : List ℝ) (h_len : nums.length = 6) 
    (h_avg6 : nums.sum / 6 = 60) (h_avg3 : (nums.take 3).sum / 3 = 55) : 
    ((nums.drop 3).sum) / 3 = 65 := 
sorry

end average_of_last_three_numbers_l280_280378


namespace number_of_men_in_first_group_l280_280932

-- Define the conditions
def condition1 (M : ℕ) : Prop := M * 80 = 20 * 40

-- State the main theorem to be proved
theorem number_of_men_in_first_group (M : ℕ) (h : condition1 M) : M = 10 := by
  sorry

end number_of_men_in_first_group_l280_280932


namespace sum_of_coordinates_of_transformed_graph_point_l280_280188

theorem sum_of_coordinates_of_transformed_graph_point (g : ℝ → ℝ) 
  (h : g 3 = 5) : 
  let x := 1 in
  let y := (4 * g (3 * x) - 6) / 3 in
  x + y = 17 / 3 :=
by
  -- Define x and y based on transformation
  let x := 1
  have hg3 : g (3 * x) = 5, from by simp [h]
  let y := (4 * g (3 * x) - 6) / 3
  -- Simplify y using g(3) = 5
  have hy : y = 14 / 3, from by simp [hg3]
  -- Compute the sum of coordinates
  show 1 + y = 17 / 3
  rw [hy]
  norm_num
  sorry

end sum_of_coordinates_of_transformed_graph_point_l280_280188


namespace probability_sum_greater_than_four_l280_280483

theorem probability_sum_greater_than_four :
  let all_outcomes := (Fin 6) × (Fin 6)
  let favorable_outcomes := {p : Fin 6 × Fin 6 | (p.1.val + 1) + (p.2.val + 1) > 4}
  (favorable_outcomes.card : ℚ) / ((Fin 6 × Fin 6).card : ℚ) = 5 / 6 :=
by
  sorry

end probability_sum_greater_than_four_l280_280483


namespace transformed_cosine_function_l280_280140

theorem transformed_cosine_function :
  ∀ (x : ℝ), 
    let f1 := λ x, cos x,
        f2 := λ x, cos (x - π / 3),
        f3 := λ x, cos (2 * x - π / 3) in
    f3 x = (λ x, cos (2 * x - π / 3)) x :=
by
  -- Proof steps
  sorry

end transformed_cosine_function_l280_280140


namespace domain_of_f_l280_280984

noncomputable def f (x : ℝ) : ℝ := (4 * x - 2) / (Real.sqrt (x - 7))

theorem domain_of_f : {x : ℝ | ∃ y : ℝ, f y = f x } = {x : ℝ | x > 7} :=
by
  sorry

end domain_of_f_l280_280984


namespace example_rook_configuration_number_of_configurations_l280_280335

-- Definitions representing the chessboard and characteristics
def is_black_square : ℕ × ℕ → Prop := 
  λ (r c : ℕ), (r + c) % 2 = 1

def is_white_square : ℕ × ℕ → Prop := 
  λ (r c : ℕ), (r + c) % 2 = 0

-- A rook attacking a square (r, c) will attack all squares in the same row or same column
def rook_attacks : (ℕ × ℕ) → (ℕ × ℕ) → Prop :=
  λ (r1 c1) (r2 c2), r1 = r2 ∨ c1 = c2

-- The main statement: part (a) proof for existence of required arrangement.
theorem example_rook_configuration :
  ∃ (rooks : list (ℕ × ℕ)),
    rooks.length = 4 ∧ 
    (∀ (rook : ℕ × ℕ), rook ∈ rooks → is_black_square rook) ∧
    (∀ (row col : ℕ), is_white_square (row, col) → ∃ (rook : ℕ × ℕ), rook ∈ rooks ∧ rook_attacks rook (row, col)) :=
sorry

-- The main statement: part (b) proof of the number of possible configurations.
theorem number_of_configurations :
  ∃ n : ℕ, n = 384 ∧ 
  (∀ (configuration : list (ℕ × ℕ)),
    configuration.length = 4 ∧
    (∀ (rook : ℕ × ℕ), rook ∈ configuration → is_black_square rook) ∧
    (∀ (row col : ℕ), is_white_square (row, col) → ∃ (rook : ℕ × ℕ), rook ∈ configuration ∧ rook_attacks rook (row, col))
  → 
  (∃ (configurations : finset (list (ℕ × ℕ))), configurations.card = n))
:= 
sorry

end example_rook_configuration_number_of_configurations_l280_280335


namespace hockey_players_count_l280_280253

theorem hockey_players_count (cricket_players : ℕ) (football_players : ℕ) (softball_players : ℕ) (total_players : ℕ) 
(h_cricket : cricket_players = 16) 
(h_football : football_players = 18) 
(h_softball : softball_players = 13) 
(h_total : total_players = 59) : 
  total_players - (cricket_players + football_players + softball_players) = 12 := 
by sorry

end hockey_players_count_l280_280253


namespace not_enough_money_to_buy_airplane_l280_280587

-- Define the given conditions
def airplane_cost_eur : ℝ := 3.80
def exchange_rate_usd_to_eur : ℝ := 0.82
def sales_tax_rate : ℝ := 0.075
def credit_card_surcharge_rate : ℝ := 0.035
def processing_fee_usd : ℝ := 0.25
def adam_money_usd : ℝ := 5.00

-- Main theorem
theorem not_enough_money_to_buy_airplane :
  let airplane_cost_usd := airplane_cost_eur / exchange_rate_usd_to_eur
      sales_tax_usd := airplane_cost_usd * sales_tax_rate
      credit_card_surcharge_usd := airplane_cost_usd * credit_card_surcharge_rate
      total_cost_usd := airplane_cost_usd + sales_tax_usd + credit_card_surcharge_usd + processing_fee_usd
  in total_cost_usd > adam_money_usd :=
by
  sorry

end not_enough_money_to_buy_airplane_l280_280587


namespace centroid_path_area_correct_l280_280063

noncomputable def centroid_path_area (D E : ℝ) (r : ℝ) (h1 : E - D = 36) (h2 : r = 18) : ℝ :=
  let centroid_radius := r / 3
  in π * centroid_radius^2

theorem centroid_path_area_correct:
  centroid_path_area D E 18 (by norm_num) (by norm_num) = 36 * π :=
sorry

end centroid_path_area_correct_l280_280063


namespace probability_not_all_same_l280_280868

-- Definitions of conditions from the problem
def six_sided_die_faces : ℕ := 6
def number_of_dice : ℕ := 5

-- Lean statement to prove the probability calculation
theorem probability_not_all_same : 
  let total_outcomes := six_sided_die_faces ^ number_of_dice in
  let all_same_outcomes := six_sided_die_faces in
  (1 - ((all_same_outcomes : ℚ) / total_outcomes)) = (1295 / 1296) := 
by
  sorry

end probability_not_all_same_l280_280868


namespace volume_of_right_tetrahedron_proof_l280_280796

noncomputable def volume_of_right_tetrahedron (a b c : ℝ) : ℝ :=
  let S_squared := (a^2 + b^2 + c^2) / 2
  in (1 / 6) * real.sqrt ((S_squared - a^2) * (S_squared - b^2) * (S_squared - c^2))

theorem volume_of_right_tetrahedron_proof (a b c : ℝ) :
  let S_squared := (a^2 + b^2 + c^2) / 2
  in volume_of_right_tetrahedron a b c = (1 / 6) * real.sqrt ((S_squared - a^2) * (S_squared - b^2) * (S_squared - c^2)) :=
by sorry

end volume_of_right_tetrahedron_proof_l280_280796


namespace probability_sum_greater_than_four_l280_280485

theorem probability_sum_greater_than_four : 
  let total_outcomes := 36 in
  let outcomes_sum_less_or_equal_4 := 6 in
  let prob_sum_less_or_equal_4 := outcomes_sum_less_or_equal_4 / total_outcomes in
  prob_sum_less_or_equal_4 = (1 : ℝ) / 6 → 
  (1 - prob_sum_less_or_equal_4) = (5 : ℝ) / 6 := 
by 
  intros total_outcomes outcomes_sum_less_or_equal_4 prob_sum_less_or_equal_4 h1,
  sorry

end probability_sum_greater_than_four_l280_280485


namespace minimum_value_expression_l280_280500

-- Defining the expression
def expression (x y : ℝ) : ℝ := x^2 + y^2 - 8 * x + 6 * y + x * y + 20

-- Defining the function representing the minimum value
def minimumValue : ℝ := -88 / 3

-- The theorem statement asserting the minimum value for the given expression
theorem minimum_value_expression :
  ∃ x y : ℝ, expression x y = minimumValue ∧ ∀ a b : ℝ, expression a b ≥ minimumValue :=
begin
  sorry
end

end minimum_value_expression_l280_280500


namespace smallest_prime_divisible_l280_280643

theorem smallest_prime_divisible (p : ℕ) (prime_p : Nat.Prime p) :
  (∃ n : ℤ, p ∣ n^2 + n + 11) ↔ p = 11 :=
by
  sorry

end smallest_prime_divisible_l280_280643


namespace common_tangents_count_l280_280219

def circle1 (x y : ℝ) := x^2 + y^2 + 2 * x + 4 * y + 1 = 0
def circle2 (x y : ℝ) := x^2 + y^2 - 4 * x - 4 * y - 1 = 0
theorem common_tangents_count : (∃ t:int, t = 3) := 
  sorry

end common_tangents_count_l280_280219


namespace no_five_pairwise_coprime_two_digit_composites_l280_280989

def is_composite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

def two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def pairwise_coprime (s : List ℕ) : Prop :=
  ∀ (x ∈ s) (y ∈ s) (h : x ≠ y), Nat.gcd x y = 1

def is_composite_and_two_digit (n : ℕ) : Prop :=
  is_composite n ∧ two_digit n

theorem no_five_pairwise_coprime_two_digit_composites:
  ¬∃ (s : List ℕ), s.length = 5 ∧
                    (∀ x ∈ s, is_composite_and_two_digit x) ∧
                    pairwise_coprime s := by
  sorry

end no_five_pairwise_coprime_two_digit_composites_l280_280989


namespace oldest_child_age_l280_280368

theorem oldest_child_age (ages : Fin 5 → ℕ) 
  (average_age : (∑ i, ages i) / 5 = 6) 
  (distinct_ages : Function.Injective ages) 
  (consecutive_diff : ∀ i : Fin 4, ages i.succ = ages i + 2) : 
  ages 4 = 10 := 
sorry

end oldest_child_age_l280_280368


namespace find_positive_number_l280_280028

-- The definition to state the given condition
def condition1 (n : ℝ) : Prop := n > 0 ∧ n^2 + n = 245

-- The theorem stating the problem and its solution
theorem find_positive_number (n : ℝ) (h : condition1 n) : n = 14 :=
by sorry

end find_positive_number_l280_280028


namespace real_part_of_z_is_neg_3_div_2_l280_280192

-- Define the imaginary unit i
def i : ℂ := complex.I

-- Define the complex number z
def z : ℂ := (1 + 4 * i) / (1 - i)

-- Prove that the real part of z is -3/2
theorem real_part_of_z_is_neg_3_div_2 : complex.re z = -3 / 2 :=
by 
  sorry

end real_part_of_z_is_neg_3_div_2_l280_280192


namespace probability_correct_l280_280446

def outcome (s₁ s₂ : ℕ) : Prop := s₁ ≥ 1 ∧ s₁ ≤ 6 ∧ s₂ ≥ 1 ∧ s₂ ≤ 6

def sum_outcome_greater_than_four (s₁ s₂ : ℕ) : Prop := outcome s₁ s₂ ∧ s₁ + s₂ > 4

def total_outcomes : ℕ := 36

def favorable_outcomes : ℕ := 30 -- As derived from 36 - 6

def probability_sum_greater_than_four : ℚ := favorable_outcomes / total_outcomes

theorem probability_correct : probability_sum_greater_than_four = 5 / 6 := 
by 
  sorry

end probability_correct_l280_280446


namespace isosceles_triangle_angle_diff_l280_280065

theorem isosceles_triangle_angle_diff (A B C : Point) 
  (h_isosceles: dist A B = dist A C) (h_angle: ∠ABC + ∠ACB = 100) :
  let M := midpoint B C in 
  let C1 := angle_bisector C A M in
  let C2 := angle_bisector C B M in
  C1 - C2 = 0 :=
sorry

end isosceles_triangle_angle_diff_l280_280065


namespace circle_center_l280_280638

theorem circle_center (x y : ℝ) : (x^2 - 2*x + y^2 - 4*y - 28 = 0) → ((x - 1)^2 + (y - 2)^2 = 33) ∧ (1, 2) :=
by
  assume h : (x^2 - 2*x + y^2 - 4*y - 28 = 0)
  sorry

end circle_center_l280_280638


namespace probability_sum_greater_than_four_l280_280478

theorem probability_sum_greater_than_four :
  let all_outcomes := (Fin 6) × (Fin 6)
  let favorable_outcomes := {p : Fin 6 × Fin 6 | (p.1.val + 1) + (p.2.val + 1) > 4}
  (favorable_outcomes.card : ℚ) / ((Fin 6 × Fin 6).card : ℚ) = 5 / 6 :=
by
  sorry

end probability_sum_greater_than_four_l280_280478


namespace find_smallest_positive_e_l280_280404

noncomputable theory

open Polynomial

def smallest_e_with_roots : ℕ :=
  let f := (X + 2) * (X - 5) * (X - 9) * (C 2 * X + 1) in
  (f.coeff 0).natAbs

theorem find_smallest_positive_e :
  smallest_e_with_roots = 90 :=
  sorry

end find_smallest_positive_e_l280_280404


namespace probability_sum_greater_than_four_l280_280474

theorem probability_sum_greater_than_four : 
  let total_outcomes := 36 in
  let favorable_outcomes := total_outcomes - 6 in
  let probability := favorable_outcomes / total_outcomes in
  probability = 5 / 6 :=
by
  sorry

end probability_sum_greater_than_four_l280_280474


namespace common_area_of_overlapping_triangles_l280_280848

theorem common_area_of_overlapping_triangles:
  let t1 := (5, 5 * Real.sqrt 3, 10)
  let t2 := (5, 5 * Real.sqrt 3, 10)
  let hypotenuse_overlap := 5
  let common_height := 5 * (Real.sqrt 3 / 2)
  let common_area := (1 / 2) * hypotenuse_overlap * common_height
  common_area = 25 * Real.sqrt 3 / 4 := by
sory

end common_area_of_overlapping_triangles_l280_280848


namespace AplusBplusC_4_l280_280761

theorem AplusBplusC_4 :
  ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 1 ∧ Nat.gcd a c = 1 ∧ (a^2 + a * b + b^2 = c^2) ∧ (a + b + c = 4) :=
by
  sorry

end AplusBplusC_4_l280_280761


namespace triangle_side_b_l280_280725

theorem triangle_side_b (a c : ℝ) (B : ℝ) (h1 : a = 5) (h2 : c = 8) (h3 : B = Real.pi / 3) : 
  let b := Real.sqrt(a^2 + c^2 - 2 * a * c * Real.cos B) in b = 7 := by
{
  sorry
}

end triangle_side_b_l280_280725


namespace gcd_factorial_8_6_squared_l280_280102

theorem gcd_factorial_8_6_squared : Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2) = 5760 := by
  sorry

end gcd_factorial_8_6_squared_l280_280102


namespace exists_unique_zero_in_interval_l280_280835

def f (x : ℝ) : ℝ := Real.log x + x^3 - 9

theorem exists_unique_zero_in_interval :
  ∃! x ∈ Set.Ioo 2 3, f x = 0 :=
sorry

end exists_unique_zero_in_interval_l280_280835


namespace eval_frac_expr_l280_280993

theorem eval_frac_expr :
  (Real.ceil (19 / 6 - Real.ceil (34 / 21))) / (Real.ceil (34 / 6 + Real.ceil (6 * 19 / 34))) = 1 / 5 :=
by
  have h1 : Real.ceil (34 / 21) = 2 := sorry
  have h2 : Real.ceil (6 * 19 / 34) = 4 := sorry
  have h3 : Real.ceil (19 / 6 - 2) = 2 := sorry
  have h4 : Real.ceil (34 / 6 + 4) = 10 := sorry
  sorry

end eval_frac_expr_l280_280993


namespace cone_radius_of_surface_area_and_lateral_surface_unfolds_to_semicircle_l280_280696

theorem cone_radius_of_surface_area_and_lateral_surface_unfolds_to_semicircle
  (surface_area : ℝ) (lateral_surface_unfolds_to_semicircle : Prop) :
  surface_area = 12 * Real.pi → lateral_surface_unfolds_to_semicircle → ∃ r : ℝ, r = 2 := by
  sorry

end cone_radius_of_surface_area_and_lateral_surface_unfolds_to_semicircle_l280_280696


namespace cashier_correction_l280_280022

theorem cashier_correction (y : ℕ) :
  let quarter_value := 25
  let nickel_value := 5
  let penny_value := 1
  let dime_value := 10
  let quarters_as_nickels_value := y * (quarter_value - nickel_value)
  let pennies_as_dimes_value := y * (dime_value - penny_value)
  let total_correction := quarters_as_nickels_value - pennies_as_dimes_value
  total_correction = 11 * y := by
  sorry

end cashier_correction_l280_280022


namespace exists_equal_mod_p_l280_280062

theorem exists_equal_mod_p (p : ℕ) [hp_prime : Fact p.Prime] 
  (m : Fin p → ℕ) 
  (h_consecutive : ∀ i j : Fin p, (i : ℕ) < j → m i + 1 = m j) 
  (sigma : Equiv (Fin p) (Fin p)) :
  ∃ (k l : Fin p), k ≠ l ∧ (m k * m (sigma k) - m l * m (sigma l)) % p = 0 :=
by
  sorry

end exists_equal_mod_p_l280_280062


namespace product_of_constants_l280_280128

theorem product_of_constants (x t a b : ℤ) (h1 : x^2 + t * x - 12 = (x + a) * (x + b)) :
  ∃ ts : Finset ℤ, ∏ t in ts, t = 1936 :=
by
  sorry

end product_of_constants_l280_280128


namespace necessary_but_not_sufficient_condition_l280_280233

theorem necessary_but_not_sufficient_condition (x : ℝ) : (|x - 1| < 1 → x^2 - 5 * x < 0) ∧ (¬(x^2 - 5 * x < 0 → |x - 1| < 1)) :=
by
  sorry

end necessary_but_not_sufficient_condition_l280_280233


namespace ratio_AF_FB_l280_280290

-- Definitions and conditions
variables (A B C D F P : Type) [affine_space ℝ Type]
variables (a b c d f p : point3d)
variables (λ : ℝ)
variables (on_A_C : ∃ λ : ℝ, d = λ • star a + (1 - λ) • star c)
variables (on_A_B : ∃ μ : ℝ, f = μ • star a + (1 - μ) • star b)
variables (AP_PD_ratio : (AP_ratio : ℝ) = 3/2)
variables (FP_PC_ratio : (FP_ratio : ℝ) = 3/1)

-- Statement to prove
theorem ratio_AF_FB : ∃λ, (af_to_fb_ratio : ℝ) = (1 - 3*λ) / (2 + 3*λ) :=
sorry

end ratio_AF_FB_l280_280290


namespace find_a_l280_280138

noncomputable def point_line_distance (x1 y1 a : ℝ) : ℝ :=
  (abs (x1 + a * y1 - 1)) / (Real.sqrt (1 + a^2))

theorem find_a 
  (a : ℝ) 
  (h1 : point_line_distance 3 1 a = 1) 
  (h2 : (x - 3)^2 + (y - 1)^2 = 7) 
  (h3 : x + a * y - 1 = 0) : 
  a = -3 / 4 := 
begin
  sorry,
end

end find_a_l280_280138


namespace limit_sequence_l280_280014

theorem limit_sequence :
  (λ n : ℕ, (Real.sqrt (n^7 + 5) - Real.sqrt (n - 5)) / (Real.root 7 (n^7 + 5) + Real.sqrt (n - 5))) ⟶ 1 :=
by {
  -- Proving the limit 
  sorry
}

end limit_sequence_l280_280014


namespace product_of_constants_l280_280125

theorem product_of_constants (x t a b : ℤ) (h1 : x^2 + t * x - 12 = (x + a) * (x + b)) :
  ∃ ts : Finset ℤ, ∏ t in ts, t = 1936 :=
by
  sorry

end product_of_constants_l280_280125


namespace probability_sum_greater_than_four_l280_280480

theorem probability_sum_greater_than_four :
  let all_outcomes := (Fin 6) × (Fin 6)
  let favorable_outcomes := {p : Fin 6 × Fin 6 | (p.1.val + 1) + (p.2.val + 1) > 4}
  (favorable_outcomes.card : ℚ) / ((Fin 6 × Fin 6).card : ℚ) = 5 / 6 :=
by
  sorry

end probability_sum_greater_than_four_l280_280480


namespace f_evaluation_l280_280199

noncomputable def f : ℝ → ℝ
| x := if x ≥ 3 then (1/2)^x else f (x + 1)

theorem f_evaluation : f (1 - real.log 3 / real.log (1/2)) = 1 / 24 :=
by
  sorry

end f_evaluation_l280_280199


namespace max_cube_volume_l280_280954

-- Define the conditions
def card_stock_length : ℝ := 60
def card_stock_width : ℝ := 25
def card_stock_area : ℝ := card_stock_length * card_stock_width
def max_cube_edge_length : ℝ := Real.sqrt (card_stock_area / 6)

-- Theorem stating the largest volume of the cube
theorem max_cube_volume : ∃ a : ℝ, a = max_cube_edge_length ∧ a^3 = 3375 :=
by
  sorry

end max_cube_volume_l280_280954


namespace exists_bounded_diff_l280_280146

theorem exists_bounded_diff {f : ℝ → ℝ} (c1 c2 : ℝ) (h1 : 0 < c1) (h2 : 0 < c2)
  (h3 : ∀ x y : ℝ, f(x) + f(y) - c1 ≤ f(x + y) ∧ f(x + y) ≤ f(x) + f(y) + c2) :
  ∃ k : ℝ, ∃ M : ℝ, ∀ x : ℝ, |f(x) - k * x| < M :=
sorry

end exists_bounded_diff_l280_280146


namespace probability_sum_greater_than_four_l280_280461

def sum_greater_than_four_probability :=
  (5 / 6: ℚ)

theorem probability_sum_greater_than_four :
  let outcomes := { (a, b) | a in (Finset.range 1 7) ∧ b in (Finset.range 1 7) }
  let favorable_outcomes := outcomes.filter (λ pair, (pair.1 + pair.2) > 4)
  let probability := favorable_outcomes.card.to_rat / outcomes.card.to_rat
  probability = sum_greater_than_four_probability :=
by
  sorry

end probability_sum_greater_than_four_l280_280461


namespace hydrogen_moles_formed_l280_280123

/-- A type representing a number of moles of a substance. -/
structure Moles :=
(value : ℕ)

def Fe : Moles := ⟨2⟩
def H₂SO₄ : Moles := ⟨2⟩

theorem hydrogen_moles_formed (fe_moles : Moles) (h2so4_moles : Moles) : 
  (fe_moles.value = 2) → (h2so4_moles.value = 2) → (∃ h2_moles : Moles, h2_moles.value = 2) :=
by
  intros hFe hH2SO4
  use 2
  sorry

end hydrogen_moles_formed_l280_280123


namespace arithmetic_mean_of_fractions_l280_280075

def mean (a b : ℚ) : ℚ := (a + b) / 2

theorem arithmetic_mean_of_fractions (a b c : ℚ) (h₁ : a = 8/11)
                                      (h₂ : b = 5/6) (h₃ : c = 19/22) :
  mean a c = b :=
by
  sorry

end arithmetic_mean_of_fractions_l280_280075


namespace cristina_catches_nicky_after_12_seconds_l280_280917

variable (t : ℝ) -- the time in seconds

-- Conditions
axiom head_start : ℝ := 36
axiom cristina_pace : ℝ := 6
axiom nicky_pace : ℝ := 3

-- Problem statement
theorem cristina_catches_nicky_after_12_seconds
    (distance_cristina : ℝ := cristina_pace * t)
    (distance_nicky : ℝ := nicky_pace * t + head_start) :
  distance_cristina = distance_nicky → t = 12 := 
sorry

end cristina_catches_nicky_after_12_seconds_l280_280917


namespace uniqueness_f_l280_280777

-- Define the sum of the digits of a natural number
def s (n : ℕ) : ℕ := n.digits 10 |>.sum

-- Define the function f
def f (n : ℕ) : ℕ :=
  let q := n / 9
  let r := n % 9
  r * 10^q + (10^q - 1)

-- Main theorem statement
theorem uniqueness_f {n : ℕ} (h1 : n > 1) (h2 : n ≠ 10) :
  ∃! (fn : ℕ), fn ≥ 2 ∧ ∀ k : ℕ, 0 < k ∧ k < fn → s(k) + s(fn - k) = n :=
sorry

end uniqueness_f_l280_280777


namespace find_c_l280_280397

theorem find_c (a b c : ℝ) (h1 : a * 2 = 3 * b / 2) (h2 : a * 2 + 9 = c) (h3 : 4 - 3 * b = -c) : 
  c = 12 :=
by
  sorry

end find_c_l280_280397


namespace a1_max_changes_sign_l280_280547

-- n is a positive integer
variables {n : ℕ} (hn : n > 0)

-- a sequence of n real numbers
variables (a : ℕ → ℝ)

-- definition of neighbors (a_1 and a_n are not neighbors)
def neighbors (i : ℕ) : ℕ × ℕ := 
  if i = 1 then (1, 2)
  else if i = n then (n, n - 1)
  else (i - 1, i + 1)

-- each move replaces a number with the average of itself and its neighbors
def average_move (i : ℕ) (a : ℕ → ℝ) : ℝ :=
  let (left, right) := neighbors n i in
  (a left + a i + a right) / 3

-- sign change definition
def changes_sign (x y : ℝ) : bool :=
  (x >= 0 ∧ y < 0) ∨ (x < 0 ∧ y >= 0)

-- Prove the maximum number of times that a₁ can change sign is n-1
noncomputable def max_sign_changes_a1 : ℕ :=
  n - 1

theorem a1_max_changes_sign {n : ℕ} (hn : n > 0)  (a : ℕ → ℝ) :
  ∃ sequence_moves : (ℕ → ℝ) → (ℕ → ℝ),
    -- sequence_moves is a function describing the sequence of moves Alice makes
    (forall_moves (moves : ℕ → ℝ), 
      (λ t i, (changes_sign t i))) = 
      (max n - 1) := sorry

end a1_max_changes_sign_l280_280547


namespace divisible_by_17_l280_280664

theorem divisible_by_17 (k : ℕ) : 17 ∣ (2^(2*k+3) + 3^(k+2) * 7^k) :=
  sorry

end divisible_by_17_l280_280664


namespace find_k_l280_280770

variable {α : Type*} [LinearOrderedField α]
variable (a : ℕ → α) (k : ℕ)

theorem find_k
  (h_seq : ∀ n, a n = a 1 + (n-1) * (a 2 - a 1))
  (h_condition1 : a 5 + a 8 + a 11 = 22)
  (h_condition2 : (∑ i in Finset.range (16) \ Finset.range (5), a i.succ) = 100)
  (h_condition3 : a k = 15) : 
  k = 14 := 
sorry

end find_k_l280_280770


namespace B_project_completion_l280_280929

-- Definitions for conditions
def rate_A : ℝ := 1 / 20
def rate_B (B : ℝ) : ℝ := 1 / B
def rate_A_and_B (B : ℝ) : ℝ := rate_A + rate_B B
def work_done_together (B : ℝ) : ℝ := 10 * rate_A_and_B B
def work_done_alone (B : ℝ) : ℝ := 5 * rate_B B

theorem B_project_completion (B : ℝ) : work_done_together B + work_done_alone B = 1 → B = 30 :=
sorry

end B_project_completion_l280_280929


namespace minimum_distance_l280_280289

noncomputable def curve1_polar (ρ θ : ℝ) : Prop :=
  ρ = 2 * sin (θ + π / 3)

noncomputable def curve2_polar (ρ θ : ℝ) : Prop :=
  ρ * sin (θ + π / 3) = 4

noncomputable def curve1_cartesian (x y : ℝ) : Prop :=
  x^2 + y^2 - (√3) * x - y = 0

noncomputable def curve2_cartesian (x y : ℝ) : Prop :=
  (√3) * x + y = 8

theorem minimum_distance (A B : (ℝ × ℝ)) (hA : curve1_cartesian A.1 A.2) (hB : curve2_cartesian B.1 B.2) :
  |(A.1 - B.1)^2 + (A.2 - B.2)^2|.sqrt = 2 :=
sorry

end minimum_distance_l280_280289


namespace expression_value_l280_280054

theorem expression_value :
  3 + Real.sqrt 3 + 1 / (3 + Real.sqrt 3) + 1 / (Real.sqrt 3 - 3) = 3 + 2 * Real.sqrt 3 / 3 :=
by
  sorry

end expression_value_l280_280054


namespace PTAFinalAmount_l280_280362

theorem PTAFinalAmount (initial_amount : ℝ) (spent_on_supplies_fraction : ℝ) (spent_on_food_fraction : ℝ) : 
  initial_amount = 400 → 
  spent_on_supplies_fraction = 1 / 4 → 
  spent_on_food_fraction = 1 / 2 → 
  (initial_amount - (initial_amount * spent_on_supplies_fraction)) / 2 = 150 := 
by
  intros h_initial h_supplies h_food
  rw [h_initial, h_supplies, h_food]
  norm_num
  sorry

end PTAFinalAmount_l280_280362


namespace collinear_XYZ_l280_280763

-- Definitions for the problem conditions
variables (Γ : Type*) [circumference Γ]
variables (A B C D : Γ)
variables (r s : tangent_line Γ)
variable [tangent_to r Γ A]
variable [tangent_to s Γ D]
variables (BC : line Γ) (AB CD DE : line Γ)
variables (E : Γ) (F : Γ) (X : Γ) (Y : Γ) (Z : Γ)

-- Intersections as defined by the problem:
variable [intersects E r BC]
variable [intersects F s BC]
variable [intersects X r s]
variable [intersects Y (line_through A F) (line_through D E)]
variable [intersects Z (line_through A B) (line_through C D)]

-- The theorem to prove:
theorem collinear_XYZ : collinear X Y Z :=
begin
  sorry
end

end collinear_XYZ_l280_280763


namespace halt_duration_l280_280544

-- Define the given conditions
def avg_speed (speed : ℝ) := speed = 87
def total_distance (distance : ℝ) := distance = 348
def start_time (start : ℝ) := start = 9
def end_time (end : ℝ) := end = 13.75 -- 1:45 pm is 13.75 in 24-hour format

-- Define the proof problem
theorem halt_duration (speed distance start end halt : ℝ) 
  (h_speed : avg_speed speed) 
  (h_distance : total_distance distance) 
  (h_start : start_time start) 
  (h_end : end_time end) : 
  halt = 45 :=
by
  -- The proof steps would be completed here
  sorry

end halt_duration_l280_280544


namespace exists_invertible_int_matrix_l280_280764

theorem exists_invertible_int_matrix (m : ℕ) (k : Fin m → ℤ) : 
  ∃ A : Matrix (Fin m) (Fin m) ℤ,
    (∀ j, IsUnit (A + k j • (1 : Matrix (Fin m) (Fin m) ℤ))) :=
sorry

end exists_invertible_int_matrix_l280_280764


namespace behind_schedule_by_5_days_l280_280912

-- Definitions
def r : ℝ := sorry  -- rate at which each man works, assumed to be positive
def W : ℝ := (100 * r * 35) + (200 * r * 5)  -- total work done

-- without additional men
def t : ℝ := W / (100 * r)

-- Prove the contractor would be behind schedule by 5 days if additional men weren't engaged
theorem behind_schedule_by_5_days :  t - 40 = 5 := by
  have h1 : W = (100 * r * 35) + (200 * r * 5) := by
    sorry
  have h2 : t = W / (100 * r) := by
    sorry
  show t - 40 = 5 from sorry

end behind_schedule_by_5_days_l280_280912


namespace prob_a_prob_b_l280_280211

def A (a : ℝ) := {x : ℝ | 0 < x + a ∧ x + a ≤ 5}
def B := {x : ℝ | -1/2 ≤ x ∧ x < 6}

theorem prob_a (a : ℝ) : (A a ⊆ B) → (-1 < a ∧ a ≤ 1/2) :=
sorry

theorem prob_b (a : ℝ) : (∃ x, A a ∩ B = {x}) → a = 11/2 :=
sorry

end prob_a_prob_b_l280_280211


namespace max_product_of_sequence_l280_280359

open BigOperators

theorem max_product_of_sequence :
  ∃ (a : Fin 2010 → ℝ),
    (∀ i j : Fin 2010, i ≠ j → (a i) * (a j) ≤ (i.val + 1) + (j.val + 1)) ∧
    (∀ a' : Fin 2010 → ℝ, (∀ i j : Fin 2010, i ≠ j → (a' i) * (a' j) ≤ (i.val + 1) + (j.val + 1)) → 
      finprod (λ i, a i) ≥ finprod (λ i, a' i))
:=
  sorry

end max_product_of_sequence_l280_280359


namespace day284_is_Saturday_l280_280236

-- Define the conditions
def dayOfWeek (day: Nat) (year: Nat): Nat := day % 7

-- Given that the 25th day of the year 2003 is a Saturday (day 6 of the week where 0 is Sunday)
def day25_is_Saturday : dayOfWeek 25 2003 = 6 := rfl

-- The main statement to be proved
theorem day284_is_Saturday : dayOfWeek 284 2003 = 6 :=
by
  have h25 := day25_is_Saturday
  sorry

end day284_is_Saturday_l280_280236


namespace number_of_odd_sum_subsets_l280_280068

def integers : finset ℕ := {42, 55, 61, 78, 102, 117}

def is_subset_odd_sum (s : finset ℕ) : Prop := 
  s.card = 3 ∧ (s.sum % 2 = 1)

theorem number_of_odd_sum_subsets : 
  (finset.filter is_subset_odd_sum (finset.subsets_of_size 3 integers)).card = 9 := 
sorry

end number_of_odd_sum_subsets_l280_280068


namespace sum_of_divisors_product_even_l280_280142

noncomputable def sum_of_divisors (n : ℕ) : ℕ :=
  if n = 0 then 0 else ∑ i in (Finset.range (n + 1)).filter (λ d, n % d = 0), i

theorem sum_of_divisors_product_even {n : ℕ} (h : n > 1) :
  (sum_of_divisors (n - 1)) * (sum_of_divisors n) * (sum_of_divisors (n + 1)) % 2 = 0 :=
begin
  sorry
end

end sum_of_divisors_product_even_l280_280142


namespace total_cost_at_discount_l280_280581

-- Definitions for conditions
def original_price_notebook : ℕ := 15
def original_price_planner : ℕ := 10
def discount_rate : ℕ := 20
def number_of_notebooks : ℕ := 4
def number_of_planners : ℕ := 8

-- Theorem statement for the proof
theorem total_cost_at_discount :
  let discounted_price_notebook := original_price_notebook - (original_price_notebook * discount_rate / 100)
  let discounted_price_planner := original_price_planner - (original_price_planner * discount_rate / 100)
  let total_cost := (number_of_notebooks * discounted_price_notebook) + (number_of_planners * discounted_price_planner)
  total_cost = 112 :=
by
  sorry

end total_cost_at_discount_l280_280581


namespace probability_not_all_same_l280_280863

-- Definitions of conditions from the problem
def six_sided_die_faces : ℕ := 6
def number_of_dice : ℕ := 5

-- Lean statement to prove the probability calculation
theorem probability_not_all_same : 
  let total_outcomes := six_sided_die_faces ^ number_of_dice in
  let all_same_outcomes := six_sided_die_faces in
  (1 - ((all_same_outcomes : ℚ) / total_outcomes)) = (1295 / 1296) := 
by
  sorry

end probability_not_all_same_l280_280863


namespace find_specific_poly_l280_280847

theorem find_specific_poly :
  ∀ P : ℝ → ℝ, (∀ a b c : ℝ, ab + bc + ca = 0 → P(a - b) + P(b - c) + P(c - a) = 2P(a + b + c)) →
  ∃ α β : ℝ, ∀ x : ℝ, P(x) = α * x ^ 4 + β * x ^ 2 :=
by
  intro P hP
  -- formal proofs go here
  sorry

end find_specific_poly_l280_280847


namespace find_added_amount_l280_280573

theorem find_added_amount (x y : ℕ) (h1 : x = 18) (h2 : 3 * (2 * x + y) = 123) : y = 5 :=
by
  sorry

end find_added_amount_l280_280573


namespace part1_part2_l280_280163

-- Part (1)
theorem part1 (x y : ℝ) (h1 : abs x = 3) (h2 : abs y = 7) (hx : x > 0) (hy : y < 0) : x + y = -4 :=
sorry

-- Part (2)
theorem part2 (x y : ℝ) (h1 : abs x = 3) (h2 : abs y = 7) (hxy : x < y) : x - y = -10 ∨ x - y = -4 :=
sorry

end part1_part2_l280_280163


namespace part_i_l280_280915

theorem part_i (a : ℕ → ℝ) (h1: ∀ n, a n > 0) (h2: ∑' n, a n < ∞) :
  ∑' n, (∏ i in finset.range (n+1), (a i))^(1/(n+1):ℝ) < real.exp 1 * ∑' n, a n :=
sorry

end part_i_l280_280915


namespace five_dice_not_all_same_probability_l280_280881

noncomputable def probability_not_all_same : ℚ :=
  let total_outcomes := (6:ℚ) ^ 5
  let favorable_outcomes := (6:ℚ)
  1 - (favorable_outcomes / total_outcomes)

theorem five_dice_not_all_same_probability :
  probability_not_all_same = 1295 / 1296 :=
by
  unfold probability_not_all_same
  norm_cast
  simp
  sorry

end five_dice_not_all_same_probability_l280_280881


namespace solve_prob_problem_l280_280900

open ProbabilityTheory

noncomputable def prob_problem (Ω : Type*) [MeasureSpace Ω] : Prop :=
  let rolls : Ω → ℕ × ℕ := sorry
  let is_event (p : (ℕ × ℕ) → Prop) : Event Ω := sorry
  let A : Event Ω := is_event (λ (x : ℕ × ℕ), x.1 + x.2 = 4)
  let B : Event Ω := is_event (λ (x : ℕ × ℕ), x.2 % 2 = 0)
  let C : Event Ω := is_event (λ (x : ℕ × ℕ), x.1 = x.2)
  let D : Event Ω := is_event (λ (x : ℕ × ℕ), (x.1 % 2 ≠ 0) ∨ (x.2 % 2 ≠ 0))
  (probability[D] = 3/4) ∧ (probability[B ⊓ D] = 1/4) ∧ (independent B C)

theorem solve_prob_problem : prob_problem := sorry

end solve_prob_problem_l280_280900


namespace distance_to_point_from_circle_center_l280_280498

theorem distance_to_point_from_circle_center : 
  let (x1, y1) := (2 : ℝ, 3 : ℝ)
  let (x2, y2) := (10 : ℝ, 10 : ℝ)
  let distance := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)
  distance = Real.sqrt 113 :=
by
  let (cx, cy) := (2 : ℝ, 3 : ℝ)
  have center_eq : (x1, y1) = (cx, cy) := by
    sorry -- proving center of circle from given equation
  rw [center_eq, Real.sqrt] -- use computed center coordinates
  sorry -- rest of the proof for distance formula

end distance_to_point_from_circle_center_l280_280498


namespace arithmetic_sequence_condition_l280_280679

noncomputable def a_seq (n : ℕ) : ℚ := (2 * n + 1) / 3

def S_n {a : ℕ → ℚ} (n : ℕ) := (Finset.range (n + 1)).sum a

def b_seq (a : ℕ → ℚ) (n : ℕ) : ℚ :=
if n = 1 then 3
else 1 / (a (n - 1) * a n)

def T_n (b : ℕ → ℚ) (n : ℕ) := (Finset.range n).sum b

theorem arithmetic_sequence_condition
(a : ℕ → ℚ)
(S : ℕ → ℚ)
(a_2_6_eq_6 : a 2 + a 6 = 6)
(S_3_eq_5 : S 3 = 5) :
(∀ n : ℕ, a n = (2 * n + 1) / 3) ∧
(∀ (b : ℕ → ℚ) (T : ℕ → ℚ),
(∀ (n : ℕ), b n = if n = 1 then 3 else 1 / ((a (n - 1)) * (a n))) →
(∀ n : ℕ, T n = (Finset.range n).sum b) →
∃ m : ℕ, m = 5 ∧ ∀ n : ℕ, T n < m) :=
begin
  sorry
end

end arithmetic_sequence_condition_l280_280679


namespace circumcenter_on_angle_bisector_l280_280675

noncomputable def circumcenter (A B C : Point) : Point := sorry

theorem circumcenter_on_angle_bisector (A B C P Q O : Point)
  (h₁ : P ∈ line_through B C ∧ ¬ collinear B C P)
  (h₂ : dist B P = dist B A)
  (h₃ : Q ∈ line_through B C ∧ ¬ collinear B C Q)
  (h₄ : dist C Q = dist C A)
  (h₅ : circumcenter A P Q = O) :
  lies_on_angle_bisector O A B C :=
sorry

end circumcenter_on_angle_bisector_l280_280675


namespace bacteria_increase_pattern_l280_280963

theorem bacteria_increase_pattern (t1 t2 t3 t4 : ℝ) 
  (b1 b2 b3 b4 : ℝ)
  (h1 : t1 = 1)
  (h2 : t2 = 4)
  (h3 : t3 = 7)
  (h4 : t4 = 10)
  (hb1 : b1 = 10)
  (hb2 : b2 = 13)
  (hb3 : b3 = 16.9)
  (increase1 : b2 - b1 = 3)
  (increase2 : b3 - b2 = 3.9)
  (increase_pattern : increase2 - increase1 = 0.9) :
  b4 = b3 + (increase2 + 0.9) :=
by
  sorry

end bacteria_increase_pattern_l280_280963


namespace selection_methods_ensuring_each_type_l280_280164

-- Defining the problem conditions
variables (A B : Type) -- Types representing Type-A and Type-B televisions

def select_ways : ℕ :=
  let type_A_count := 4
  let type_B_count := 5
  let select_type_A := nat.choose type_A_count 1 -- Choose 1 Type-A television from 4
  let select_type_B := nat.choose type_B_count 1 -- Choose 1 Type-B television from 5
  let remaining_televisions := type_A_count - 1 + type_B_count - 1 -- Rest of the televisions
  let select_remaining := nat.choose remaining_televisions 1 -- Choose 1 more television from remaining
  select_type_A * select_type_B * select_remaining

-- The statement of the problem
theorem selection_methods_ensuring_each_type :
  select_ways A B = 140 := 
sorry -- Proof to be filled in

end selection_methods_ensuring_each_type_l280_280164


namespace calculate_expr_l280_280605

noncomputable def expr : ℕ := (5^8 - 3^7) * (1^6 + (-1)^5)^11

theorem calculate_expr : expr = 0 := 
by
  sorry

end calculate_expr_l280_280605


namespace james_packs_of_sodas_l280_280752

theorem james_packs_of_sodas (sodas_per_pack : ℕ) (initial_sodas : ℕ) (sodas_per_day : ℕ) (days_in_week : ℕ) :
  sodas_per_pack = 12 → initial_sodas = 10 → sodas_per_day = 10 → days_in_week = 7 → (days_in_week * sodas_per_day - initial_sodas) / sodas_per_pack = 5 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end james_packs_of_sodas_l280_280752


namespace triangle_perimeter_l280_280749

variable (A B C D : Type)
variable [Triangle A B C]
variable [Bisector B D (Angle A B C)]
variable [Length AB 21]
variable [Length BD (8 * Real.sqrt 7)]
variable [Length DC 8]

theorem triangle_perimeter (A B C D : Type)
  [Triangle A B C] 
  [Bisector B D (Angle A B C)] 
  [Length AB 21] 
  [Length BD (8 * Real.sqrt 7)] 
  [Length DC 8] : 
  perimeter A B C = 60 := 
sorry

end triangle_perimeter_l280_280749


namespace rectangle_area_l280_280000

def length : ℝ := 15
def width : ℝ := 0.9 * length
def area : ℝ := length * width

theorem rectangle_area : area = 202.5 := by
  sorry

end rectangle_area_l280_280000


namespace inscribed_circle_radius_l280_280262

theorem inscribed_circle_radius (R r : ℝ) (hR : R = 18) (hr : r = 9) : ∃ x : ℝ, x = 8 := 
by
  use 8
  sorry


end inscribed_circle_radius_l280_280262


namespace no_polygon_with_1974_diagonals_l280_280077

theorem no_polygon_with_1974_diagonals :
  ¬ ∃ N : ℕ, N * (N - 3) / 2 = 1974 :=
sorry

end no_polygon_with_1974_diagonals_l280_280077


namespace smallest_square_area_l280_280561

theorem smallest_square_area (a b c d : ℕ) (square_side : ℕ) (h1 : a = 2) (h2 : b = 4) (h3 : c = 4) (h4 : d = 5) (h5 : ¬ ∃ x y w z : ℕ, x + w ≤ square_side ∧ y + z ≤ square_side ∧ x = 2 ∧ y = 4 ∧ w = 4 ∧ z = 5) : square_side * square_side = 49 :=
by
  -- Necessary for defining the noncomputable nature of the side of the square
  noncomputable theory
  -- Provide proof here
  sorry

end smallest_square_area_l280_280561


namespace area_OBEC_is_19_5_l280_280938

-- Definitions for the points and lines from the conditions
structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨5, 0⟩
def B : Point := ⟨0, 15⟩
def C : Point := ⟨6, 0⟩
def E : Point := ⟨3, 6⟩

-- Function to calculate the area of a triangle given its vertices
def triangle_area (P1 P2 P3 : Point) : ℝ :=
  0.5 * |(P1.x * P2.y + P2.x * P3.y + P3.x * P1.y) - (P1.y * P2.x + P2.y * P3.x + P3.y * P1.x)|

-- Definitions of the vertices of the quadrilateral
def O : Point := ⟨0, 0⟩

-- Calculating the area of triangles OCE and OBE
def OCE_area : ℝ := triangle_area O C E
def OBE_area : ℝ := triangle_area O B E

-- Total area of quadrilateral OBEC
def OBEC_area : ℝ := OCE_area + OBE_area

-- Proof statement: The area of quadrilateral OBEC is 19.5
theorem area_OBEC_is_19_5 : OBEC_area = 19.5 := sorry

end area_OBEC_is_19_5_l280_280938


namespace tree_iff_unique_path_l280_280767

structure Graph (V : Type) :=
(adj : V → V → Prop)
(connected : ∀ u v, adj u v → adj v u) -- Assuming undirected graph

def AcyclicGraph (V : Type) :=
{ g : Graph V // ¬∃ (u v w : V), g.adj u v ∧ g.adj v w ∧ g.adj w u }

noncomputable def unique_path (V : Type) (g : Graph V) :=
∀ u v, u ≠ v → ∃! p, (p : list V) → (∀ i, i < p.length - 1 → g.adj (p.nth i) (p.nth (i + 1))) ∧ (p.head = u) ∧ (p.tail.head = v)

theorem tree_iff_unique_path (V : Type) (T : Graph V) :
  (∃ (t : AcyclicGraph V), t.val = T) ↔ unique_path V T := sorry

end tree_iff_unique_path_l280_280767


namespace smallest_f_n_gt_15_l280_280719

def sum_of_digits (n : ℕ) : ℕ := n.digits.sum

def f (n : ℕ) : ℕ := sum_of_digits (Real.ceil ((10 / 3 : ℝ) ^ n))

theorem smallest_f_n_gt_15 : (∀ m < 7, f m ≤ 15) ∧ f 7 > 15 :=
by
  sorry

end smallest_f_n_gt_15_l280_280719


namespace line_through_point_l280_280640

noncomputable
def is_line_with_equal_intercepts (a b : ℝ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ a = b

theorem line_through_point (a b : ℝ) :
  (is_line_with_equal_intercepts a b ∧ (2, 3) ∈ [(x : ℝ) × (y : ℝ)], x + y = a) →
    (∀ x y : ℝ, y = (3 / 2) * x ∨ x + y = 5) :=
by
  sorry

end line_through_point_l280_280640


namespace find_y_l280_280994

theorem find_y :
  (∀ (y : ℝ), 2 * Real.arctan (1 / 5) + Real.arctan (1 / 25) + Real.arctan (1 / y) = π / 3 → y = 1005 / 97) :=
by
  assume y,
  intro h,
  sorry

end find_y_l280_280994


namespace simplify_336_to_fraction_l280_280517

theorem simplify_336_to_fraction :
  let gcd_36_100 := Nat.gcd 36 100
  3.36 = (84 : ℚ) / 25 := 
by
  let g := Nat.gcd 36 100
  have h1 : 3.36 = 3 + 0.36 := by norm_num
  have h2 : 0.36 = 36 / 100 := by norm_num
  have h3 : g = 4 := by norm_num [Nat.gcd, Nat.gcd_def, Nat.gcd_rec]
  have h4 : (36 : ℚ) / 100 = 9 / 25 := by norm_num; field_simp [h3];
  have h5 : (3 : ℚ) + (9 / 25) = 84 / 25 := by norm_num; field_simp;
  rw [h1, h2, h4, h5]

end simplify_336_to_fraction_l280_280517


namespace three_point_three_six_as_fraction_l280_280534

theorem three_point_three_six_as_fraction : 3.36 = (84 : ℚ) / 25 := 
by
  sorry

end three_point_three_six_as_fraction_l280_280534


namespace find_k_m_l280_280768

noncomputable def cross_vec (v1 v2 : Vector3 ℝ) : Vector3 ℝ := sorry
noncomputable def dot_vec (v1 v2 : Vector3 ℝ) : ℝ := sorry
noncomputable def det_matrix (col1 col2 col3 : Vector3 ℝ) : ℝ := sorry

variables (a b c d : Vector3 ℝ) (E : ℝ)
hypothesis (hE: E = dot_vec a (cross_vec b c))
def matrix_new := (cross_vec a b, cross_vec b c, cross_vec (cross_vec c a) d)
def E' := det_matrix (cross_vec a b) (cross_vec b c) (cross_vec (cross_vec c a) d)

theorem find_k_m (hE' : E' = 0) : ∃ (k m : ℤ), E' = k * E^m := 
begin
  use [0, 1],
  sorry
end

end find_k_m_l280_280768


namespace value_calculation_l280_280410

theorem value_calculation (x : ℕ) (hx : x = 43) : 38 + 2 * x = 124 := by
  rw [hx]
  norm_num

end value_calculation_l280_280410


namespace probability_sum_greater_than_four_l280_280468

def sum_greater_than_four_probability :=
  (5 / 6: ℚ)

theorem probability_sum_greater_than_four :
  let outcomes := { (a, b) | a in (Finset.range 1 7) ∧ b in (Finset.range 1 7) }
  let favorable_outcomes := outcomes.filter (λ pair, (pair.1 + pair.2) > 4)
  let probability := favorable_outcomes.card.to_rat / outcomes.card.to_rat
  probability = sum_greater_than_four_probability :=
by
  sorry

end probability_sum_greater_than_four_l280_280468


namespace ship_path_graph_l280_280943

-- Definitions according to the conditions
def point : Type := ℝ × ℝ  -- representing coordinates
def distance (p1 p2 : point) : ℝ := ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2).sqrt

variable (A B C D X : point)
variable (r : ℝ)
variable (semicircular_path : A ≠ B ∧ ∀ t ∈ Icc (0 : ℝ) 1, distance (lerp t A B) X = r)
variable (straight_path_1 : B ≠ C ∧ ∃ M : point, distance M X = 2 * r / 2 ∧ ∀ t ∈ Icc (0 : ℝ) 1, distance (lerp t B M) X = (t * (2 * r - r) + r) ∧ distance (lerp t M C) X = ((1 - t) * (2 * r - r) + r))
variable (straight_path_2 : C ≠ D ∧ ∃ N : point, distance N X < r ∧ ∀ t ∈ Icc (0 : ℝ) 1, distance (lerp t C N) X = ((1 - t) * (r - N) + N) ∧ distance (lerp t N D) X = (t * (r - N) + N))

-- Final theorem statement
theorem ship_path_graph :
  semicircular_path ∧ straight_path_1 ∧ straight_path_2 →
  ∃ f : ℝ → ℝ, (∀ t ∈ Icc (0 : ℝ) (1/3), f t = r) ∧
               (∀ t ∈ Icc (1/3) (2/3), f t = t * (2 * r - r) + r) ∧
               (∀ t ∈ Icc (2/3) 1, f t = ((1 - t) * (r - some_point) + some_point) :=
by
  sorry

end ship_path_graph_l280_280943


namespace shaded_area_of_circumscribed_circles_l280_280611

theorem shaded_area_of_circumscribed_circles :
  let r1 := 3 in
  let r2 := 4 in
  let r3 := r1 + r2 in
  let area_larger_circle := π * r3^2 in
  let area_smaller_circle1 := π * r1^2 in
  let area_smaller_circle2 := π * r2^2 in
  (area_larger_circle - (area_smaller_circle1 + area_smaller_circle2)) = 24 * π :=
by
  let r1 := 3
  let r2 := 4
  let r3 := r1 + r2
  let area_larger_circle := π * r3^2
  let area_smaller_circle1 := π * r1^2
  let area_smaller_circle2 := π * r2^2
  show (area_larger_circle - (area_smaller_circle1 + area_smaller_circle2)) = 24 * π, from sorry

end shaded_area_of_circumscribed_circles_l280_280611


namespace quad_area_is_one_l280_280416

-- Define the rectangle and the circles
variable (O1 O2 A B C D : ℝ) -- Points on the plane
variable (r : ℝ) -- Radius of the circles
variable (width : ℝ) -- Width of the rectangle 

-- Given conditions
variable (h_width : width = 1) -- The width of the rectangle is 1 cm
variable (h_radius : r = 1) -- The radius of both circles is 1 cm
variable (h_intersection : (C, D) ∈ lineSegment O1 O2) -- C and D are points of intersection on the segment O1 O2

-- The proof goal
theorem quad_area_is_one :
  area ℚ (ABCD) = 1 :=
by sorry

end quad_area_is_one_l280_280416


namespace gcd_fact_8_fact_6_sq_l280_280110

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def gcd (a b : ℕ) : ℕ := Nat.gcd a b

theorem gcd_fact_8_fact_6_sq : gcd (factorial 8) ((factorial 6)^2) = 11520 := by
  sorry

end gcd_fact_8_fact_6_sq_l280_280110


namespace find_students_l280_280572

theorem find_students (n : ℕ) (h1 : n % 8 = 5) (h2 : n % 6 = 1) (h3 : n < 50) : n = 13 :=
sorry

end find_students_l280_280572


namespace one_minus_repeating_decimal_three_equals_two_thirds_l280_280091

-- Define the repeating decimal as a fraction
def repeating_decimal_three : ℚ := 1 / 3

-- Prove the desired equality
theorem one_minus_repeating_decimal_three_equals_two_thirds :
  1 - repeating_decimal_three = 2 / 3 :=
by
  sorry

end one_minus_repeating_decimal_three_equals_two_thirds_l280_280091


namespace number_of_students_l280_280251

theorem number_of_students (S G : ℕ) (h1 : G = 2 * S / 3) (h2 : 8 = 2 * G / 5) : S = 30 :=
by
  sorry

end number_of_students_l280_280251


namespace find_m_l280_280959

theorem find_m (m : ℝ) : 
  let r := (5 / 15 : ℝ),
    S1 := 15 / (1 - r),
    s := (5 + m) / 15,
    S2 := 15 / (1 - s)
  in
    r = 1/3 ∧ S1 = 22.5 ∧ S2 = 3 * S1 → m = 6.67 :=
by
  sorry

end find_m_l280_280959


namespace shaded_area_of_circumscribed_circles_l280_280614

theorem shaded_area_of_circumscribed_circles (r₁ r₂ : ℝ) (h₁ : r₁ = 3) (h₂ : r₂ = 4) :
  let R := r₁ + r₂
  let A := π * R^2
  let A₁ := π * r₁^2
  let A₂ := π * r₂^2
  A - A₁ - A₂ = 24 * π :=
by
  -- Define the radius of the large circle.
  let R := r₁ + r₂
  -- Define the areas of the large circle and the two smaller circles.
  let A := π * R^2
  let A₁ := π * r₁^2
  let A₂ := π * r₂^2
  -- Compute shaded area.
  have h₃ : R = 7, from calc
    R = r₁ + r₂ : rfl
    ... = 3 + 4 : by rw [h₁, h₂]
    ... = 7 : by norm_num,
  have h₄ : A = 49 * π, from calc
    A = π * R^2 : rfl
    ... = π * 7^2 : by rw [h₃]
    ... = 49 * π : by norm_num,
  have h₅ : A₁ = 9 * π, from calc
    A₁ = π * r₁^2 : rfl
    ... = π * 3^2 : by rw [h₁]
    ... = 9 * π : by norm_num,
  have h₆ : A₂ = 16 * π, from calc
    A₂ = π * r₂^2 : rfl
    ... = π * 4^2 : by rw [h₂]
    ... = 16 * π : by norm_num,
  calc
    A - A₁ - A₂
      = 49 * π - 9 * π - 16 * π : by rw [h₄, h₅, h₆]
      ... = 24 * π : by norm_num

end shaded_area_of_circumscribed_circles_l280_280614


namespace cot_identity_1_cot_sum_identity_cot_arccot_sum_l280_280658

noncomputable def cot (x : ℝ) : ℝ := 1 / Real.tan x
noncomputable def arccot (x : ℝ) : ℝ := Real.atan (1 / x)

theorem cot_identity_1 (x : ℝ) : cot (arccot x) = x := by sorry

theorem cot_sum_identity (α β : ℝ) :
    cot (α + β) = (cot α * cot β - 1) / (cot α + cot β) := by sorry

theorem cot_arccot_sum :
  cot (arccot 2 + arccot 5 + arccot 11 + arccot 17) = 1478 / 219 := by sorry

end cot_identity_1_cot_sum_identity_cot_arccot_sum_l280_280658


namespace least_n_froods_l280_280740

def froods_score (n : ℕ) : ℕ := n * (n + 1) / 2
def eating_score (n : ℕ) : ℕ := n ^ 2

theorem least_n_froods :
    ∃ n : ℕ, 0 < n ∧ (froods_score n > eating_score n) ∧ (∀ m : ℕ, 0 < m ∧ m < n → froods_score m ≤ eating_score m) :=
  sorry

end least_n_froods_l280_280740


namespace area_of_quadrilateral_l280_280418

theorem area_of_quadrilateral :
  ∀ (O₁ O₂ A B C D : Point) (r : Real) (one_cm : r = 1),
  -- Conditions
  dist O₁ A = r ∧ dist O₁ O₂ = 2 * r ∧
  circle O₁ r ∩ line_segment O₁ O₂ = {C, D} ∧
  circle O₂ r ∩ line_segment O₁ O₂ = {C, D} ∧
  quadrilateral A B C D ∧ on_line C (line_segment O₁ O₂) ∧ on_line D (line_segment O₁ O₂) ∧
  -- Conditions specific to area calculation
  (∃ E : Point, perpendicular DE AB ∧ dist DE = r) →
  -- Result: the area of quadrilateral ABCD is 1 cm²
  quadrilateral_area A B C D = r * r := by
  sorry

end area_of_quadrilateral_l280_280418


namespace arithmetic_geometric_sequence_problem_l280_280412

variable {a_n : ℕ → ℝ} {S : ℕ → ℝ}

-- Define the conditions
def is_arithmetic_sequence (a_n : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, a_n n = a_n 0 + n * d

def sum_of_first_n_terms (a_n : ℕ → ℝ) (S : ℕ → ℝ) :=
  ∀ n : ℕ, S n = (n * (a_n 0 + a_n (n-1))) / 2

def forms_geometric_sequence (a1 a3 a4 : ℝ) :=
  a3^2 = a1 * a4

-- The main proof statement
theorem arithmetic_geometric_sequence_problem
        (h_arith : is_arithmetic_sequence a_n)
        (h_sum : sum_of_first_n_terms a_n S)
        (h_geom : forms_geometric_sequence (a_n 0) (a_n 2) (a_n 3)) :
        (S 3 - S 2) / (S 5 - S 3) = 2 ∨ (S 3 - S 2) / (S 5 - S 3) = 1 / 2 :=
  sorry

end arithmetic_geometric_sequence_problem_l280_280412


namespace cube_volume_l280_280833

theorem cube_volume (SA : ℝ) (hSA : SA = 1350) : 
  let a := real.sqrt (SA / 6) in 
  let V := a^3 in 
  V = 3375 :=
by
  sorry

end cube_volume_l280_280833


namespace inscribed_circle_radius_l280_280270

theorem inscribed_circle_radius (R r : ℝ) (hR : R = 18) (hr : r = 9) :
    ∃ x : ℝ, (∀ P Px OP O1P : ℝ, Px = sqrt((R - x) ^ 2 - x ^ 2) ∧ O1P = sqrt((r + x) ^ 2 - x ^ 2)
                 ∧ Px + r = O1P) ∧ x = 8 :=
begin
  sorry
end

end inscribed_circle_radius_l280_280270


namespace right_triangle_third_side_l280_280241

theorem right_triangle_third_side (a b : ℝ) (h : a = 6 ∧ b = 8 ∨ a = 8 ∧ b = 6) :
  (∃ c : ℝ, c^2 = a^2 + b^2 ∨ c^2 = a^2 - b^2) :=
begin
  use [10, 2 * real.sqrt 7],
  cases h,
  { cases h with ha hb,
    left,
    simp [ha, hb],
    norm_num, },
  { cases h with ha hb,
    right,
    simp [ha, hb],
    norm_num,
    field_simp,
    ring, },
end

end right_triangle_third_side_l280_280241


namespace PTA_money_left_l280_280364

theorem PTA_money_left (initial_savings : ℝ) (spent_on_supplies : ℝ) (spent_on_food : ℝ) :
  initial_savings = 400 →
  spent_on_supplies = initial_savings / 4 →
  spent_on_food = (initial_savings - spent_on_supplies) / 2 →
  (initial_savings - spent_on_supplies - spent_on_food) = 150 :=
by
  intro initial_savings_eq
  intro spent_on_supplies_eq
  intro spent_on_food_eq
  sorry

end PTA_money_left_l280_280364


namespace total_blocks_per_day_l280_280344

def blocks_to_park : ℕ := 4
def blocks_to_hs : ℕ := 7
def blocks_to_home : ℕ := 11
def walks_per_day : ℕ := 3

theorem total_blocks_per_day :
  (blocks_to_park + blocks_to_hs + blocks_to_home) * walks_per_day = 66 :=
by
  sorry

end total_blocks_per_day_l280_280344


namespace rectangular_area_length_width_l280_280946

open Nat

theorem rectangular_area_length_width (lengthInMeters widthInMeters : ℕ) (h1 : lengthInMeters = 500) (h2 : widthInMeters = 60) :
  (lengthInMeters * widthInMeters = 30000) ∧ ((lengthInMeters * widthInMeters) / 10000 = 3) :=
by
  sorry

end rectangular_area_length_width_l280_280946


namespace least_max_subset_count_l280_280067

theorem least_max_subset_count :
  let S := {1, 2, ..., 30} in
  ∀ (A : Finset (Finset ℕ)),
  (A.card = 10) →
  (∀ a ∈ A, a.card = 3) →
  (∀ x y ∈ A, x ≠ y → (x ∩ y).nonempty) →
  ∃ n : ℕ, (∀ i ∈ S, n_i i = {j | j ∈ A ∧ i ∈ j}.card) ∧ (∀ i ∈ S, n_i i ≤ 5) :=
by
  sorry

end least_max_subset_count_l280_280067


namespace find_pairs_l280_280999

theorem find_pairs (a b : ℝ) :
  (∀ x ∈ set.Icc (0 : ℝ) 1, |real.sqrt (1 - x^2) - a * x - b| ≤ (real.sqrt 2 - 1) / 2) ↔ (a = 0 ∧ b = 0) :=
by
  -- Proof goes here
  sorry

end find_pairs_l280_280999


namespace delta_k_not_zero_l280_280143

def u (n : ℕ) : ℕ := n^4 + n^2

def delta_1 (u : ℕ → ℕ) (n : ℕ) : ℕ := u (n + 1) - u n

-- Define delta^k in terms of delta_1
def delta (u : ℕ → ℕ) : ℕ → ℕ → ℕ
| 0, n := u n
| (k + 1), n := delta_1 (delta u k) n

theorem delta_k_not_zero :
  ∀ k n : ℕ, delta u k n ≠ 0 := by
  sorry

end delta_k_not_zero_l280_280143


namespace sum_of_squares_l280_280645

theorem sum_of_squares (a b : ℝ) (h1 : a + b = 16) (h2 : a * b = 20) : a^2 + b^2 = 216 :=
by
  sorry

end sum_of_squares_l280_280645


namespace contractor_absent_days_l280_280539

variable (x y : ℕ)

theorem contractor_absent_days :
  (x + y = 30) ∧ (25 * x - 7.5 * y = 425) → y = 10 := by
  sorry

end contractor_absent_days_l280_280539


namespace correct_order_of_syllogism_l280_280051

-- Definitions based on conditions
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f(x)

def symmetric_about_origin (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(x) = f(-x) ↔ x = 0

def function_example (x : ℝ) : ℝ := x * Real.cos x

-- Major premise
def major_premise : Prop := symmetric_about_origin function_example

-- Minor premise
def minor_premise : Prop := is_odd_function function_example

-- Conclusion
def conclusion : Prop := ∀ x : ℝ, function_example(-x) = -function_example(x)

-- Theorem: correct syllogism order is ② major premise, ① minor premise, ③ conclusion forming (major_premise → minor_premise → conclusion).
theorem correct_order_of_syllogism : major_premise ∧ minor_premise → conclusion := by
  sorry

end correct_order_of_syllogism_l280_280051


namespace book_arrangement_l280_280837

theorem book_arrangement :
  let books : ℕ := 8,
      math_books : ℕ := 3,
      foreign_language_books : ℕ := 3,
      literature_books : ℕ := 2 in
  (math_books + foreign_language_books + literature_books = books) →
  ∃ ways, ways = 864 :=
by
  sorry

end book_arrangement_l280_280837


namespace find_distance_city_A_B_l280_280628

variable (D_AB : ℝ)    -- Distance between city A and city B
variable (D_AC : ℝ)    -- Distance between city A and city C
variable (T_Eddy : ℝ)  -- Time taken by Eddy to travel from city A to city B
variable (T_Freddy : ℝ) -- Time taken by Freddy to travel from city A to city C
variable (speed_ratio : ℝ) -- Ratio of Eddy's speed to Freddy's speed

-- Define the basic conditions given in the problem
def problem_conditions (D_AC = 300) (T_Eddy = 3) (T_Freddy = 4) (speed_ratio = 2.2666666666666666) : Prop :=
  let VE := D_AB / T_Eddy
  let VF := D_AC / T_Freddy
  VE / VF = speed_ratio

-- Prove the distance D_AB given the conditions
theorem find_distance_city_A_B (h : problem_conditions D_AC 3 4 2.2666666666666666) : D_AB = 510 := 
sorry

end find_distance_city_A_B_l280_280628


namespace find_length_of_AC_l280_280282

theorem find_length_of_AC 
  (A B C : Type) [RightTriangle A B C] (hC : angle C = 90)
  (hSinA : sin A = sqrt 5 / 3)
  (hBC : length BC = 2 * sqrt 5) :
  length AC = 4 := sorry

end find_length_of_AC_l280_280282


namespace sum_of_integers_l280_280380

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 8) (h3 : x * y = 144) : x + y = 24 :=
sorry

end sum_of_integers_l280_280380


namespace indeterminate_eq_solutions_l280_280172

theorem indeterminate_eq_solutions (k : ℕ) (h : k > 1) :
  ∃ (S : set (ℕ × ℕ × ℕ)), 
    (∀ s ∈ S, let (m, n, r) := s in mn + nr + mr = k * (m + n + r)) ∧ 
    S.card ≥ 3 * k + 1 :=
sorry

end indeterminate_eq_solutions_l280_280172


namespace compound_interest_calculation_l280_280010

noncomputable def compoundInterest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  let A := P * ((1 + r / (n : ℝ)) ^ (n * t))
  A - P

theorem compound_interest_calculation :
  compoundInterest 500 0.05 1 5 = 138.14 := by
  sorry

end compound_interest_calculation_l280_280010


namespace probability_sum_greater_than_four_is_5_over_6_l280_280433

-- Define the sample space for two dice.
def sample_space : Finset (ℕ × ℕ) :=
  Finset.product (Finset.range 6) (Finset.range 6) 

-- Define the event where the sum is greater than four.
def event_sum_greater_than_four : Finset (ℕ × ℕ) :=
  sample_space.filter (λ p, p.1 + p.2 > 4)

-- Compute the probability of the event.
def probability_sum_greater_than_four : ℚ :=
  (event_sum_greater_than_four.card : ℚ) / (sample_space.card : ℚ)

-- Theorem to be proven
theorem probability_sum_greater_than_four_is_5_over_6 :
  probability_sum_greater_than_four = 5 / 6 :=
by
  -- Proof would go here
  sorry

end probability_sum_greater_than_four_is_5_over_6_l280_280433


namespace probability_not_all_dice_same_l280_280887

theorem probability_not_all_dice_same :
  let total_outcomes := 6^5
  let same_number_outcomes := 6
  let probability_same_number := same_number_outcomes / total_outcomes
  let probability_not_same_number := 1 - probability_same_number
  probability_not_same_number = (1295 : ℚ) / 1296 :=
by
  sorry

end probability_not_all_dice_same_l280_280887


namespace probability_not_all_dice_same_l280_280890

theorem probability_not_all_dice_same :
  let total_outcomes := 6^5
  let same_number_outcomes := 6
  let probability_same_number := same_number_outcomes / total_outcomes
  let probability_not_same_number := 1 - probability_same_number
  probability_not_same_number = (1295 : ℚ) / 1296 :=
by
  sorry

end probability_not_all_dice_same_l280_280890


namespace intersecting_curves_count_l280_280609

-- Define the set of coefficients
def coefficient_set := {1, 2, 3, 4, 5, 6, 7, 8}

-- State the main theorem
theorem intersecting_curves_count :
  ∃ A B C D E F ∈ coefficient_set, 
  (A ≠ D) ∧ (B ≠ E) ∧ (C ≠ F) ∧ 
  (Amy.distinct {A, B, C, D, E, F}) = True ∧ 
  (∃ x ∈ ℝ, Ax^2 + Bx + C = Dx^2 + Ex + F) :=
sorry

end intersecting_curves_count_l280_280609


namespace tangent_line_equality_l280_280697

theorem tangent_line_equality (x₁ x₂ : ℝ) 
  (h : e^x₁ = 1 / x₂) 
  (h_tangent_lines : e^x₁ * (x - x₁) + e^x₁ * (1 - x₁) = (1 / x₂) * (x - x₂) + ln x₂ - 1) : 
  (x₁ + 1) * (x₂ - 1) = -2 :=
sorry

end tangent_line_equality_l280_280697


namespace ellipse_equation_reciprocal_square_distances_triangle_area_range_l280_280171

-- Conditions for the Ellipse
variables {a b c : ℝ} (O F1 F2 M : ℝ × ℝ)
def ellipse (x y : ℝ) := (x^2 / (a^2) + y^2 / (b^2) = 1)
def is_origin (O : ℝ × ℝ) := O = (0, 0)
def foci_distance (F1 F2 : ℝ × ℝ) := (real.dist F1 F2 = 2 * real.sqrt 3)
def max_angle (F1 M F2 : ℝ × ℝ) := ( ∠ (F1,M,F2) = (2/3) * real.pi)
def orthogonal_points (P Q : ℝ × ℝ) (O : ℝ × ℝ) := ( ( (P.2 - O.2) * (Q.2 - O.2) ) + (( Q.1 - O.1 ) * (P.1 - O.1)) = 0 )

-- Prove 1: Ellipse equation
theorem ellipse_equation (h : foci_distance F1 F2) (hmax : max_angle F1 M F2) : (a = 2) ∧ (b = 1) ∧ (ellipse a b) := by sorry

-- Prove 2: Constant value of reciprocal square distances
theorem reciprocal_square_distances (P Q : ℝ × ℝ) (h1 : orthogonal_points P Q O) : (1/(real.norm (P - O))^2) + (1/(real.norm Q - O)^2) = 5/4 := by sorry

-- Prove 3: Range for area of triangle OPQ
theorem triangle_area_range (P Q : ℝ × ℝ) (h1 : orthogonal_points P Q O) : ( ( 4 / 5 ) ≤ ( 1/2 * real.sqrt ( (real.dist O P)^2 * ( real.dist O Q)^2 ))) ∧ (1/2 * real.sqrt ( (real.dist O P)^2 * ( real.dist O Q)^2 ) ≤ 1 ) := by sorry


end ellipse_equation_reciprocal_square_distances_triangle_area_range_l280_280171


namespace part1_part2_l280_280765

-- Definitions and conditions
def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {0, 1, 2, 3}

-- Part 1: Proving bijective mappings
theorem part1 : (Finset.perm {1, 2, 3, 4}).card = 24 := by
  sorry

-- Part 2: Proving the number of mappings satisfying the sum condition
theorem part2 : (Finset.filter (λ f : A → B, f 1 + f 2 + f 3 + f 4 = 4)
  (Finset.univ A → B)).card = 31 := by
  sorry

end part1_part2_l280_280765


namespace simplify_336_to_fraction_l280_280515

theorem simplify_336_to_fraction :
  let gcd_36_100 := Nat.gcd 36 100
  3.36 = (84 : ℚ) / 25 := 
by
  let g := Nat.gcd 36 100
  have h1 : 3.36 = 3 + 0.36 := by norm_num
  have h2 : 0.36 = 36 / 100 := by norm_num
  have h3 : g = 4 := by norm_num [Nat.gcd, Nat.gcd_def, Nat.gcd_rec]
  have h4 : (36 : ℚ) / 100 = 9 / 25 := by norm_num; field_simp [h3];
  have h5 : (3 : ℚ) + (9 / 25) = 84 / 25 := by norm_num; field_simp;
  rw [h1, h2, h4, h5]

end simplify_336_to_fraction_l280_280515


namespace expected_lone_cars_l280_280555
-- Import Lean's math library to ensure necessary functions and theorems are available.

-- Define a theorem to prove the expected number of lone cars is 1.
theorem expected_lone_cars (n : ℕ) : 
  -- n must be greater than or equal to 1, since there must be at least one car.
  n ≥ 1 -> 
  -- Expected number of lone cars is 1.
  (∑ k in finset.range n, (1 : ℝ) / (k + 1)) = 1 := 
begin
  intro hn, -- Assume n ≥ 1
  sorry,    -- The proof of this theorem is to be provided, but the statement is correct.
end

end expected_lone_cars_l280_280555


namespace elder_brother_birth_year_l280_280790

-- Define a fortunate year as a year with all unique digits
def is_fortunate_year (y : Nat) : Prop :=
  let digits := List.ofString (y.repr)
  List.nodup digits

-- Define the problem statement as a Lean theorem
theorem elder_brother_birth_year :
  (∃ y, is_fortunate_year y ∧ y = 2013) ∧ 
  (∃ y, is_fortunate_year y ∧ y < 2013 ∧ y = 1987) →
  ∃ y, y = 1987 :=
by
  intro h
  existsi 1987
  sorry

end elder_brother_birth_year_l280_280790


namespace sqrt_sum_eq_l280_280550

theorem sqrt_sum_eq : sqrt 12 + sqrt (1 / 3) = (7 * sqrt 3) / 3 := sorry

end sqrt_sum_eq_l280_280550


namespace five_dice_not_all_same_probability_l280_280883

noncomputable def probability_not_all_same : ℚ :=
  let total_outcomes := (6:ℚ) ^ 5
  let favorable_outcomes := (6:ℚ)
  1 - (favorable_outcomes / total_outcomes)

theorem five_dice_not_all_same_probability :
  probability_not_all_same = 1295 / 1296 :=
by
  unfold probability_not_all_same
  norm_cast
  simp
  sorry

end five_dice_not_all_same_probability_l280_280883


namespace probability_sum_greater_than_four_l280_280454

theorem probability_sum_greater_than_four : 
  let num_dice := 2
  let sides_per_die := 6
  let favorable_outcomes := { (a, b) | a > 0 ∧ a ≤ sides_per_die ∧ b > 0 ∧ b ≤ sides_per_die ∧ a + b > 4 }
  let total_outcomes := sides_per_die * sides_per_die
  let probability := (favorable_outcomes.card : ℚ) / total_outcomes
  probability = 5 / 6 :=
by
  sorry

end probability_sum_greater_than_four_l280_280454


namespace initial_stickers_correct_l280_280329

-- Definitions based on the conditions
def initial_stickers (X : ℕ) : ℕ := X
def after_buying (X : ℕ) : ℕ := X + 26
def after_birthday (X : ℕ) : ℕ := after_buying X + 20
def after_giving (X : ℕ) : ℕ := after_birthday X - 6
def after_decorating (X : ℕ) : ℕ := after_giving X - 58

-- Theorem stating the problem and the expected answer
theorem initial_stickers_correct (X : ℕ) (h : after_decorating X = 2) : initial_stickers X = 26 :=
by {
  sorry
}

end initial_stickers_correct_l280_280329


namespace tangent_line_eq_l280_280819

/-- The equation of the tangent line to the curve y = 2x * tan x at the point x = π/4 is 
    (2 + π/2) * x - y - π^2/4 = 0. -/
theorem tangent_line_eq : ∀ x y : ℝ, 
  (y = 2 * x * Real.tan x) →
  (x = Real.pi / 4) →
  ((2 + Real.pi / 2) * x - y - Real.pi^2 / 4 = 0) :=
by
  intros x y h_curve h_point
  sorry

end tangent_line_eq_l280_280819


namespace sum_of_digits_base_31_for_primes_lt_20000_l280_280661

-- Define the sum of the digits of n in base k
def S (k n : ℕ) : ℕ :=
  let digits := n.digits k in
  digits.foldl (fun acc d => acc + d) 0

-- Define the problem as a theorem statement
theorem sum_of_digits_base_31_for_primes_lt_20000 :
  ∀ p : ℕ, prime p ∧ p < 20000 → 
    S 31 p ∈ {v | ¬ prime v ∧ (v = 49 ∨ v = 77)} ∨ prime (S 31 p) := 
  by 
  intro p,
  intro hp,
  have prime_p := hp.left,
  have p_lt_20000 := hp.right,
  sorry

end sum_of_digits_base_31_for_primes_lt_20000_l280_280661


namespace product_of_t_l280_280136

theorem product_of_t (a b : ℤ) (t : ℤ) (h1 : a * b = -12) (h2 : t = a + b) :
  ∃ (t_values : Finset ℤ), 
  (∀ x ∈ t_values, ∃ a b : ℤ, a * b = -12 ∧ x = a + b) ∧ 
  (t_values.product = -1936) :=
by
  sorry

end product_of_t_l280_280136


namespace CMO_2018_Q6_l280_280967

theorem CMO_2018_Q6 
  (n k : ℕ)
  (h : n > k)
  (a : fin n → ℝ)
  (ha : ∀ i, a i ∈ Ioo (k-1 : ℝ) k)
  (x : fin n → ℝ)
  (hx : ∀ (I : finset (fin n)), I.card = k → ∑ i in I, x i ≤ ∑ i in I, a i) : 
  ∏ i, x i ≤ ∏ i, a i :=
sorry

end CMO_2018_Q6_l280_280967


namespace dimes_in_piggy_bank_l280_280575

variable (q d : ℕ)

def total_coins := q + d = 100
def total_amount := 25 * q + 10 * d = 1975

theorem dimes_in_piggy_bank (h1 : total_coins q d) (h2 : total_amount q d) : d = 35 := by
  sorry

end dimes_in_piggy_bank_l280_280575


namespace bee_speed_l280_280332

-- Define distances and speeds
def distance_A_B : ℝ := 120
def speed_A : ℝ := 30
def speed_B : ℝ := 10
def distance_bee : ℝ := 180

-- Define the expected speed of the bee
def expected_speed_bee : ℝ := 60

-- The statement to be proven
theorem bee_speed :
  (let relative_speed := speed_A + speed_B in
   let time_to_meet := distance_A_B / relative_speed in
   let speed_bee := distance_bee / time_to_meet in
   speed_bee = expected_speed_bee) :=
by
  -- Skip the proof
  sorry

end bee_speed_l280_280332


namespace length_of_jordans_rectangle_l280_280608

theorem length_of_jordans_rectangle 
  (h1 : ∃ (length width : ℕ), length = 5 ∧ width = 24) 
  (h2 : ∃ (width_area : ℕ), width_area = 30 ∧ ∃ (area : ℕ), area = 5 * 24 ∧ ∃ (L : ℕ), area = L * width_area) :
  ∃ L, L = 4 := by 
  sorry

end length_of_jordans_rectangle_l280_280608


namespace second_term_of_geometric_series_l280_280958

theorem second_term_of_geometric_series (a r S: ℝ) (h_r : r = 1/4) (h_S : S = 40) (h_geom_sum : S = a / (1 - r)) : a * r = 7.5 :=
by
  sorry

end second_term_of_geometric_series_l280_280958


namespace dihedral_angle_is_correct_l280_280593

noncomputable def dihedral_angle_BAC_E (A B C D E F : ℝ × ℝ × ℝ) 
  (ABCD_square : (B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2 + (B.3 - A.3) ^ 2 = 4 ∧
    (C.1 - B.1) ^ 2 + (C.2 - B.2) ^ 2 + (C.3 - B.3) ^ 2 = 4 ∧
    (D.1 - C.1) ^ 2 + (D.2 - C.2) ^ 2 + (D.3 - C.3) ^ 2 = 4 ∧
    (A.1 - D.1) ^ 2 + (A.2 - D.2) ^ 2 + (A.3 - D.3) ^ 2 = 4) 
  (E_midpoint : (E.1 - A.1) = (B.1 - E.1) ∧ E.2 = 0 ∧ E.3 = 0)
  (BF_perp_ACE : F.1 = E.1 ∧ F.2 = 1 * E.1 ∧ F.3 = E.3 ∧ 
    1 * (F.2 - A.2)* (B.2 - A.2) = 0) : ℝ :=
by
  let n1 := (1, 0, 0)
  let n2 := (1, -1, 1)
  let inner_product := n1.1 * n2.1 + n1.2 * n2.2 + n1.3 * n2.3
  let magnitude1 := Math.sqrt(n1.1 ^ 2 + n1.2 ^ 2 + n1.3 ^ 2)
  let magnitude2 := Math.sqrt(n2.1 ^ 2 + n2.2 ^ 2 + n2.3 ^ 2)
  let cos_theta := inner_product / (magnitude1 * magnitude2)
  exact Real.arccos cos_theta

theorem dihedral_angle_is_correct :
  ∀ (A B C D E F : ℝ × ℝ × ℝ),
  ((B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2 + (B.3 - A.3) ^ 2 = 4) ∧
  ((C.1 - B.1) ^ 2 + (C.2 - B.2) ^ 2 + (C.3 - B.3) ^ 2 = 4) ∧
  ((D.1 - C.1) ^ 2 + (D.2 - C.2) ^ 2 + (D.3 - C.3) ^ 2 = 4) ∧
  ((A.1 - D.1) ^ 2 + (A.2 - D.2) ^ 2 + (A.3 - D.3) ^ 2 = 4) →
  ((E.1 - A.1) = (B.1 - E.1) ∧ E.2 = 0 ∧ E.3 = 0) →
  F.1 = E.1 ∧ F.2 = 1 * E.1 ∧ F.3 = E.3 ∧ 
  1 * (F.2 - A.2)* (B.2 - A.2) = 0 →
  dihedral_angle_BAC_E A B C D E F = Real.arccos (Math.sqrt 3 / 3) :=
sorry

end dihedral_angle_is_correct_l280_280593


namespace equation_of_ellipse_existence_line_l_l280_280170

-- Definitions based on Conditions
variable (a b c : ℝ)
variable (ellipse_eq : (x y : ℝ) → x^2 / a^2 + y^2 / b^2 = 1)
variable (eccentricity : c / a = sqrt 2 / 2)
variable (max_distance : (point : ℝ × ℝ) → point.snd^2 + (point.fst - 1)^2 = (sqrt 2 + 1)^2)
variable (m : ℝ)
variable (C : ℝ × ℝ := (m, 0))
variable (F : ℝ × ℝ := (1, 0))
variable (k : ℝ)

-- Proof problems
theorem equation_of_ellipse (hpos : a > b) (hbpos : b > 0)
  : ∃ a b, (canonical_eq : ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1) :=
  sorry

theorem existence_line_l (h_pos : 0 < m ∧ m < 1)
    (h_line : k ≠ 0) 
    (intersects_A_B : ∀ k, ∃ (A B : ℝ × ℝ), ellipse_eq A.fst A.snd ∧ ellipse_eq B.fst B.snd ∧ |(A.fst - C.fst)^2 + (A.snd - C.snd)^2| = |(B.fst - C.fst)^2 + (B.snd - C.snd)^2|)
    : (0 < m ∧ m < 1/2 → ∃ (l : affine Slope), intersects_A_B k)
    ∧ (1/2 ≤ m ∧ m < 1 → ¬ ∃ (l : affine Slope), intersects_A_B k) :=
  sorry

end equation_of_ellipse_existence_line_l_l280_280170


namespace paint_replacement_l280_280354

theorem paint_replacement :
  ∀ (original_paint new_paint : ℝ), 
  original_paint = 100 →
  new_paint = 0.10 * (original_paint - 0.5 * original_paint) + 0.20 * (0.5 * original_paint) →
  new_paint / original_paint = 0.15 :=
by
  intros original_paint new_paint h_orig h_new
  sorry

end paint_replacement_l280_280354


namespace magnitude_of_a_l280_280244

-- Describe the conditions given in the problem
def vector_a (m : ℝ) : ℝ × ℝ := (2^m, -1)
def vector_b (m : ℝ) : ℝ × ℝ := (2^m - 1, 2^(m + 1))

-- Define the condition that vectors a and b are perpendicular
def perpendicular (a b : ℝ × ℝ) : Prop := a.1 * b.1 + a.2 * b.2 = 0

-- The proof problem: prove that the magnitude of vector a is √10 given the conditions
theorem magnitude_of_a (m : ℝ) (h : perpendicular (vector_a m) (vector_b m)) : real.sqrt ((2^m)^2 + 1) = real.sqrt 10 :=
by sorry

end magnitude_of_a_l280_280244


namespace sum_of_data_l280_280012

theorem sum_of_data (a b c : ℕ) (h1 : a + b = c) (h2 : b = 3 * a) (h3 : a = 12) : a + b + c = 96 :=
by
  sorry

end sum_of_data_l280_280012


namespace prob1_line_m_prob2_line_l_l280_280553

-- Problem 1
theorem prob1_line_m (x y : ℝ) :
  (∃ m : ℝ, (2, 4) = (x, y) ∧ (m = -1) ∧ (x - y + 1 = 0 ∨ x - y + 2 = 0)) →
  (x + y - 6 = 0) := 
by
  sorry

-- Problem 2
theorem prob2_line_l (x y : ℝ) :
  (∃ l : ℝ, (2, 4) = (x, y) ∧ 
  (∃ x0 y0 : ℝ, midpoint ((x0, y0), (2, 4)) = ((x + 2 * y - 3) / 2, (y + 4) / 2) ∧ 
  (x + y + 4 = 0 ∧ x0 - y0 + 1 = 0))) → 
  (5 * x - 4 * y + 6 = 0) := 
by
  sorry

end prob1_line_m_prob2_line_l_l280_280553


namespace bus_stop_time_per_hour_l280_280914

theorem bus_stop_time_per_hour (speed_without_stops : ℝ) (speed_with_stops : ℝ) (stoppage_time : ℝ) : 
  speed_without_stops = 54 ∧ speed_with_stops = 41 → stoppage_time = 14.44 :=
begin
  intro h,
  sorry
end

end bus_stop_time_per_hour_l280_280914


namespace simplify_336_to_fraction_l280_280519

theorem simplify_336_to_fraction : (336 / 100) = (84 / 25) :=
by sorry

end simplify_336_to_fraction_l280_280519


namespace cheaper_price_difference_is_75_cents_l280_280059

noncomputable def list_price := 42.50
noncomputable def store_a_discount := 12.00
noncomputable def store_b_discount_percent := 0.30

noncomputable def store_a_price := list_price - store_a_discount
noncomputable def store_b_price := (1 - store_b_discount_percent) * list_price
noncomputable def price_difference_in_dollars := store_a_price - store_b_price
noncomputable def price_difference_in_cents := price_difference_in_dollars * 100

theorem cheaper_price_difference_is_75_cents :
  price_difference_in_cents = 75 := by
  sorry

end cheaper_price_difference_is_75_cents_l280_280059


namespace num_digits_product_l280_280220

noncomputable def num_digits (n : ℕ) : ℕ := 
  if n = 0 then 1 else 1 + Nat.log10 n

theorem num_digits_product : num_digits (3^7 * 7^4) = 7 := by
  sorry

end num_digits_product_l280_280220


namespace find_principal_amount_l280_280919

theorem find_principal_amount 
  (P₁ : ℝ) (r₁ t₁ : ℝ) (S₁ : ℝ)
  (P₂ : ℝ) (r₂ t₂ : ℝ) (C₂ : ℝ) :
  S₁ = (P₁ * r₁ * t₁) / 100 →
  C₂ = P₂ * ( (1 + r₂) ^ t₂ - 1) →
  S₁ = C₂ / 2 →
  P₁ = 2800 :=
by
  sorry

end find_principal_amount_l280_280919


namespace hens_count_l280_280571

theorem hens_count (H C : ℕ) (h1 : H + C = 48) (h2 : 2 * H + 4 * C = 144) 
  (h3 : H ≥ 10) (h4 : C ≥ 5) : H = 24 :=
by
  sorry

end hens_count_l280_280571


namespace probability_not_all_same_l280_280862

theorem probability_not_all_same :
    let total_outcomes := 6 ^ 5 in
    let same_number_outcomes := 6 in
    let p_all_same := same_number_outcomes / total_outcomes in
    let p_not_all_same := 1 - p_all_same in
    p_not_all_same = 1295 / 1296 :=
by
  sorry

end probability_not_all_same_l280_280862


namespace train_crossing_time_l280_280011

theorem train_crossing_time :
  ∀ (L_train L_bridge : ℕ) (S_train_kmph : ℕ),
    L_train = 110 →
    L_bridge = 290 →
    S_train_kmph = 60 →
    (L_train + L_bridge : ℕ) / (((S_train_kmph : ℚ) * 1000 / 3600) : ℚ) ≈ 24 :=
by
  sorry

end train_crossing_time_l280_280011


namespace find_a_value_l280_280694

theorem find_a_value (a : ℝ) : (∀ x : ℝ, (a + 3) * x^(|a| - 2) + 5 = 0 → |a| - 2 = 1) → a = 3 :=
by
  intro h
  have h1 : |a| - 2 = 1 := sorry  -- based on the condition that the equation is linear
  have h2 : |a| = 3 := by linarith
  have h3 : a ≠ -3 := by linarith
  have h4 : a = 3 := sorry  -- considering h2 and h3, we conclude a = 3
  exact h4

end find_a_value_l280_280694


namespace probability_of_A_l280_280386

variable (Ω : Type) [ProbabilitySpace Ω]
variables (A B : Event Ω)

theorem probability_of_A :
  independent A B →
  0 < P(A) →
  P(A) = 2 * P(B) →
  P(A ∪ B) = 8 * P(A ∩ B) →
  P(A) = 1 / 3 := by
sory

end probability_of_A_l280_280386


namespace count_positive_whole_numbers_cuberoot_lt_15_l280_280223

theorem count_positive_whole_numbers_cuberoot_lt_15 :
  set.count {x : ℕ | x > 0 ∧ real.cbrt x < 15} = 3374 :=
by
  sorry

end count_positive_whole_numbers_cuberoot_lt_15_l280_280223


namespace hose_rate_l280_280665

theorem hose_rate (V : ℝ) (T : ℝ) (r_fixed : ℝ) (total_rate : ℝ) (R : ℝ) :
  V = 15000 ∧ T = 25 ∧ r_fixed = 3 ∧ total_rate = 10 ∧
  (2 * R + 2 * r_fixed = total_rate) → R = 2 :=
by
  -- Given conditions:
  -- Volume V = 15000 gallons
  -- Time T = 25 hours
  -- Rate of fixed hoses r_fixed = 3 gallons per minute each
  -- Total rate of filling the pool total_rate = 10 gallons per minute
  -- Relationship: 2 * rate of first two hoses + 2 * rate of fixed hoses = total rate
  
  sorry

end hose_rate_l280_280665


namespace parabola_standard_equation_intersection_point_fixed_line_l280_280208

-- Given the parabola y^2 = 2px with p > 0, the equation of the parabola with the condition p = 1 is y^2 = 2x
theorem parabola_standard_equation (p : ℝ) (hp : p > 0) :  -- conditions and identification of variables
  (∀ x y : ℝ, y^2 = 2 * p * x ↔ y^2 = 2 * x) :=           -- proof that this equation matches our expected result
sorry

-- Given certain lines intersecting the parabola and defining points, the intersection point G lies on the line x = 1
theorem intersection_point_fixed_line (xG : ℝ) (yG : ℝ) 
  (h1 : ∀ x y : ℝ, y^2 = 2 * x) -- defining parabolic relation
  (h2 : xG = 1) :               -- intersection point prediction
  (∃ G : ℝ × ℝ, G = (xG, yG ∧ xG = 1)) :=   -- statement to prove existence of such G
sorry

end parabola_standard_equation_intersection_point_fixed_line_l280_280208


namespace biased_coin_probability_l280_280565

noncomputable def correct_probability : ℝ := (1 - real.sqrt(1 - 4 / (real.cbrt 400))) / 2

theorem biased_coin_probability (p : ℝ) (h1 : p < 1/2)
  (h2 : (20 * p^3 * (1 - p)^3) = (1 / 20)) : p = correct_probability := 
sorry

end biased_coin_probability_l280_280565


namespace two_dice_sum_greater_than_four_l280_280437
open Classical

def probability_sum_greater_than_four : ℚ := by sorry

theorem two_dice_sum_greater_than_four :
  probability_sum_greater_than_four = 5 / 6 :=
sorry

end two_dice_sum_greater_than_four_l280_280437


namespace smallest_integer_n_condition_l280_280074

theorem smallest_integer_n_condition (a b c d : ℤ) (n : ℕ) (hn : n ≥ 4) (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) (satisfactory : ∀ (s : Finset ℤ), s.card = n → ∃ (a b c d ∈ s), a + b - c - d % 20 = 0) :
  n = 7 :=
by
  sorry

end smallest_integer_n_condition_l280_280074


namespace D_72_is_22_l280_280309

def D (n : ℕ) : ℕ :=
   -- function definition for D that satisfies the problem's conditions
   sorry

theorem D_72_is_22 : D 72 = 22 :=
by sorry

end D_72_is_22_l280_280309


namespace right_triangle_l280_280908

-- Definitions for each set of segments
def setA := (2 : ℕ, 3 : ℕ, 4 : ℕ)
def setB := (Real.sqrt 7, 3 : ℕ, 5 : ℕ)
def setC := (6 : ℕ, 8 : ℕ, 10 : ℕ)
def setD := (5 : ℕ, 12 : ℕ, 12 : ℕ)

-- Main statement that needs to be proven
theorem right_triangle : 
  ¬((setA.1^2 + setA.2^2 = setA.3^2)) ∧
  ¬((setB.1^2 + setB.2^2 = setB.3^2)) ∧
   (setC.1^2 + setC.2^2 = setC.3^2) ∧
  ¬((setD.1^2 + setD.2^2 = setD.3^2)) :=
by 
  sorry

end right_triangle_l280_280908


namespace find_smallest_n_l280_280762

noncomputable def polynomial_P (x : ℤ) : polynomial ℤ :=
  3 * ∑ k in finset.range 10, x ^ k +
  2 * ∑ k in finset.range (1200 - 10 + 1), x ^ (k + 10) +
  ∑ k in finset.range (146200 - 1200 + 1), x ^ (k + 1200)

theorem find_smallest_n :
  ∃ (f g : polynomial ℤ), x^35431200 - 1 = (x^16 + 1) * polynomial_P x * f + 11 * g :=
sorry

end find_smallest_n_l280_280762


namespace one_minus_repeating_decimal_three_equals_two_thirds_l280_280092

-- Define the repeating decimal as a fraction
def repeating_decimal_three : ℚ := 1 / 3

-- Prove the desired equality
theorem one_minus_repeating_decimal_three_equals_two_thirds :
  1 - repeating_decimal_three = 2 / 3 :=
by
  sorry

end one_minus_repeating_decimal_three_equals_two_thirds_l280_280092


namespace probability_earning_1600_l280_280336

/-- On the game show "Wheel of Fraction", a contestant spins a spinner three times.
    The spinner has six regions, each with a monetary value or function: 
    "Bankrupt," "$1000," "$200," "$700," "$500," and "$400."
    What is the probability that the contestant earns exactly $1600 
    in their first three spins, assuming all regions are equally likely? -/
theorem probability_earning_1600 :
  let outcomes := ["Bankrupt", "$1000", "$200", "$700", "$500", "$400"]
  let values : list ℕ := [0, 1000, 200, 700, 500, 400]
  (∀ region ∈ outcomes, region ≠ "Bankrupt" → ∃ val ∈ values,
    region = to_string val) →
  (∃ f : fin 6 → ℕ, (values^[f 0] + values^[f 1] + values^[f 2] = 1600) ∧ 
     probability_is (f 0, f 1, f 2)) →
  ((9 : ℕ) / (216 : ℕ) = (1 : ℕ) / (24 : ℕ)) := sorry

end probability_earning_1600_l280_280336


namespace impossible_transform_l280_280040

theorem impossible_transform : 
  ¬(∃ a b c : ℕ, 
    a + b + c = 45 ∧ 
    ((a = 0 ∧ b = 0) ∨ (a = 0 ∧ c = 0) ∨ (b = 0 ∧ c = 0)) ∧ 
    (∃ f : ℕ → ℕ × ℕ × ℕ, f 0 = (13, 15, 17) ∧ 
      (∀ n, let (x, y, z) := f n in (f (n + 1) = (x + 2, y - 1, z - 1) ∨ f (n + 1) = (x - 1, y + 2, z - 1) ∨ f (n + 1) = (x - 1, y - 1, z + 2))))
  ) := 
sorry

end impossible_transform_l280_280040


namespace total_price_of_order_l280_280563

-- Define the price of each item
def price_ice_cream_bar : ℝ := 0.60
def price_sundae : ℝ := 1.40

-- Define the quantity of each item
def quantity_ice_cream_bar : ℕ := 125
def quantity_sundae : ℕ := 125

-- Calculate the costs
def cost_ice_cream_bar := quantity_ice_cream_bar * price_ice_cream_bar
def cost_sundae := quantity_sundae * price_sundae

-- Calculate the total cost
def total_cost := cost_ice_cream_bar + cost_sundae

-- Statement of the theorem
theorem total_price_of_order : total_cost = 250 := 
by {
  sorry
}

end total_price_of_order_l280_280563


namespace tangent_circle_distance_proof_l280_280710

noncomputable def tangent_circle_distance (R r : ℝ) (tangent_type : String) : ℝ :=
  if tangent_type = "external" then R + r else R - r

theorem tangent_circle_distance_proof (R r : ℝ) (tangent_type : String) (hR : R = 4) (hr : r = 3) :
  tangent_circle_distance R r tangent_type = 7 ∨ tangent_circle_distance R r tangent_type = 1 := by
  sorry

end tangent_circle_distance_proof_l280_280710


namespace limit_of_sequence_l280_280978

def binom (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def a (p : ℕ) : ℝ :=
  (List.range (p + 1)).sum (λ k, (-1)^k * binom p k / ((k + 2) * (k + 4)))

theorem limit_of_sequence :
  filter.tendsto (λ n, (List.range (n + 1)).sum a) filter.at_top (𝓝 (1 / 3)) :=
by
  sorry

end limit_of_sequence_l280_280978


namespace evaluate_81_pow_3_div_4_l280_280086

theorem evaluate_81_pow_3_div_4 : (81 : ℝ)^(3/4) = 27 := by
  have h1 : (81 : ℝ) = (3 : ℝ)^4 := by norm_num
  have h2 : ((3 : ℝ)^4)^(3/4) = (3 : ℝ)^(4 * (3/4)) := by rw [Real.rpow_mul]
  have h3 : (3 : ℝ)^(4 * (3/4)) = (3 : ℝ)^3 := by norm_num
  rw [h1, h2, h3]
  norm_num

end evaluate_81_pow_3_div_4_l280_280086


namespace descendants_of_one_are_fibonacci_quotients_l280_280852

def fib : ℕ → ℕ
| 0 := 0
| 1 := 1
| (n+2) := fib n + fib (n+1)

theorem descendants_of_one_are_fibonacci_quotients :
  ∀ x: ℕ, x > 1 → ∃ n: ℕ, x = fib n / fib (n-1) ∨ x = fib n / fib (n+1) :=
by
  sorry

end descendants_of_one_are_fibonacci_quotients_l280_280852


namespace probability_sum_greater_than_four_l280_280472

theorem probability_sum_greater_than_four : 
  let total_outcomes := 36 in
  let favorable_outcomes := total_outcomes - 6 in
  let probability := favorable_outcomes / total_outcomes in
  probability = 5 / 6 :=
by
  sorry

end probability_sum_greater_than_four_l280_280472


namespace probability_not_all_dice_same_l280_280888

theorem probability_not_all_dice_same :
  let total_outcomes := 6^5
  let same_number_outcomes := 6
  let probability_same_number := same_number_outcomes / total_outcomes
  let probability_not_same_number := 1 - probability_same_number
  probability_not_same_number = (1295 : ℚ) / 1296 :=
by
  sorry

end probability_not_all_dice_same_l280_280888


namespace polynomial_condition_l280_280094

def is_relatively_prime (a b : ℕ) : Prop := Nat.gcd a b = 1

def divides (m n : ℤ) : Prop := ∃ k : ℤ, n = m * k

theorem polynomial_condition (f : ℤ → ℤ) :
  (∃ (a b : ℕ), is_relatively_prime a b ∧ divides (a + b) (f a + f b)) →
  (∀ (g h : ℤ → ℤ), (f = λ x, g x + h x) ∧ (∀ x, g (-x) = -g x) ∧ (∀ x, h (-x) = h x) →
                     (∃ (n : ℕ), f = λ x, ∑ i in Finset.range (2 * n + 1), (C i) * x^i)) :=
by
  sorry

end polynomial_condition_l280_280094


namespace part_a_part_b_part_c_l280_280620

def op (a b : ℕ) : ℕ := a ^ b + b ^ a

theorem part_a (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : op a b = op b a :=
by
  dsimp [op]
  rw [add_comm]

theorem part_b (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : ¬ (op a (op b c) = op (op a b) c) :=
by
  -- example counter: a = 2, b = 2, c = 2 
  -- 2 ^ (2^2 + 2^2) + (2^2 + 2^2) ^ 2 ≠ (2^2 + 2 ^ 2) ^ 2 + 8 ^ 2
  sorry

theorem part_c (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : ¬ (op (op a b) (op b c) = op (op b a) (op c b)) :=
by
  -- example counter: a = 2, b = 3, c = 2 
  -- This will involve specific calculations showing the inequality.
  sorry

end part_a_part_b_part_c_l280_280620


namespace gcd_fact_8_fact_6_sq_l280_280111

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def gcd (a b : ℕ) : ℕ := Nat.gcd a b

theorem gcd_fact_8_fact_6_sq : gcd (factorial 8) ((factorial 6)^2) = 11520 := by
  sorry

end gcd_fact_8_fact_6_sq_l280_280111


namespace sum_of_ages_is_18_l280_280304

-- Define the conditions
def product_of_ages (kiana twin : ℕ) := kiana * twin^2 = 128

-- Define the proof problem statement
theorem sum_of_ages_is_18 : ∃ (kiana twin : ℕ), product_of_ages kiana twin ∧ twin > kiana ∧ kiana + twin + twin = 18 :=
by
  sorry

end sum_of_ages_is_18_l280_280304


namespace incenter_coincidence_l280_280338

noncomputable def circumscribed_circle_center (A B C : Point) : Point := sorry
-- Assume this function calculates the center of the circumscribed circle of triangle ABC

noncomputable def incenter (A B C : Point) : Point := sorry
-- Assume this function calculates the incenter of triangle ABC

theorem incenter_coincidence {A B C A1 B1 C1 : Point}
  (hA1 : A1 ∈ segment B C)
  (hB1 : B1 ∈ segment C A)
  (hC1 : C1 ∈ segment A B)
  (h_condition : AB - AC1 = CA1 - CB1 ∧ CA1 - CB1 = BC1 - BA1) :
  incenter (circumscribed_circle_center A B1 C1)
           (circumscribed_circle_center A1 B C1)
           (circumscribed_circle_center A1 B1 C) 
  = incenter A B C :=
sorry

end incenter_coincidence_l280_280338


namespace g_60_eq_10_l280_280391

noncomputable def g : ℝ+ → ℝ := sorry 

theorem g_60_eq_10
  (h1 : ∀ (x y : ℝ+), g (x * y) = g x / y)
  (h2 : g 40 = 15) :
  g 60 = 10 :=
sorry

end g_60_eq_10_l280_280391


namespace tetrahedron_ratio_l280_280955

theorem tetrahedron_ratio (a b c d : ℝ) (h₁ : a^2 = b^2 + c^2) (h₂ : b^2 = a^2 + d^2) (h₃ : c^2 = a^2 + b^2) : 
  a / d = Real.sqrt ((1 + Real.sqrt 5) / 2) :=
sorry

end tetrahedron_ratio_l280_280955


namespace train_speed_l280_280913

theorem train_speed (train_length : ℕ) (bridge_length : ℕ) (crossing_time : ℕ) 
  (h₁ : train_length = 150) (h₂ : bridge_length = 320) (h₃ : crossing_time = 40) : 
  ((train_length + bridge_length) / crossing_time) = 11.75 :=
by
  have total_distance := train_length + bridge_length
  rw [h₁, h₂] at total_distance
  have speed := total_distance / crossing_time
  rw [h₃] at speed
  -- Perform the actual division and verify it's equal to 11.75
  sorry

end train_speed_l280_280913


namespace work_together_days_l280_280931

theorem work_together_days (A_rate B_rate : ℝ) (x B_alone_days : ℝ)
  (hA : A_rate = 1 / 5)
  (hB : B_rate = 1 / 15)
  (h_total_work : (A_rate + B_rate) * x + B_rate * B_alone_days = 1) :
  x = 2 :=
by
  -- Set up the equation based on given rates and solving for x.
  sorry

end work_together_days_l280_280931


namespace inscribed_circle_radius_l280_280274

theorem inscribed_circle_radius (R r : ℝ) (hR : R = 18) (hr : r = 9) :
    ∃ x : ℝ, (∀ P Px OP O1P : ℝ, Px = sqrt((R - x) ^ 2 - x ^ 2) ∧ O1P = sqrt((r + x) ^ 2 - x ^ 2)
                 ∧ Px + r = O1P) ∧ x = 8 :=
begin
  sorry
end

end inscribed_circle_radius_l280_280274


namespace average_of_last_three_numbers_l280_280373

theorem average_of_last_three_numbers (a b c d e f : ℝ) 
  (h1 : (a + b + c + d + e + f) / 6 = 60) 
  (h2 : (a + b + c) / 3 = 55) : 
  (d + e + f) / 3 = 65 :=
sorry

end average_of_last_three_numbers_l280_280373


namespace MN_length_angle_cosine_l280_280748

-- Lean definitions to match the given conditions
variables {a b c : ℝ}

-- Given conditions
def DE := c / 2

theorem MN_length_angle_cosine (cos_angle_C : ℝ) (h_cos : cos_angle_C = (a^2 + b^2 - c^2) / (2 * a * b)) :
  ∃ MN : ℝ, MN = (c * (a^2 + b^2 - c^2)) / (4 * a * b) :=
begin
  use (DE * cos_angle_C),
  rw [h_cos, DE],
  field_simp,
  ring,
  sorry -- proof to be completed
end

-- Ensuring DE is defined correctly as half of AB
lemma DE_is_half_c : DE = c / 2 :=
by refl

end MN_length_angle_cosine_l280_280748


namespace probability_sum_greater_than_four_is_5_over_6_l280_280429

-- Define the sample space for two dice.
def sample_space : Finset (ℕ × ℕ) :=
  Finset.product (Finset.range 6) (Finset.range 6) 

-- Define the event where the sum is greater than four.
def event_sum_greater_than_four : Finset (ℕ × ℕ) :=
  sample_space.filter (λ p, p.1 + p.2 > 4)

-- Compute the probability of the event.
def probability_sum_greater_than_four : ℚ :=
  (event_sum_greater_than_four.card : ℚ) / (sample_space.card : ℚ)

-- Theorem to be proven
theorem probability_sum_greater_than_four_is_5_over_6 :
  probability_sum_greater_than_four = 5 / 6 :=
by
  -- Proof would go here
  sorry

end probability_sum_greater_than_four_is_5_over_6_l280_280429


namespace hyperbola_chord_line_eq_l280_280160

theorem hyperbola_chord_line_eq (m n s t : ℝ) (h_mn_pos : m > 0 ∧ n > 0 ∧ s > 0 ∧ t > 0)
  (h_mn_sum : m + n = 2)
  (h_m_n_s_t : m / s + n / t = 9)
  (h_s_t_min : s + t = 4 / 9)
  (h_midpoint : (2 : ℝ) = (m + n)) :
  ∃ (c : ℝ), (∀ (x1 y1 x2 y2 : ℝ), 
    (x1 + x2) / 2 = m ∧ (y1 + y2) / 2 = n ∧ 
    (x1 ^ 2 / 4 - y1 ^ 2 / 2 = 1 ∧ x2 ^ 2 / 4 - y2 ^ 2 / 2 = 1) → 
    y2 - y1 = c * (x2 - x1)) ∧ (c = 1 / 2) →
  ∀ (x y : ℝ), x - 2 * y + 1 = 0 :=
by sorry

end hyperbola_chord_line_eq_l280_280160


namespace fraction_representation_of_3_36_l280_280525

theorem fraction_representation_of_3_36 : (336 : ℚ) / 100 = 84 / 25 := 
by sorry

end fraction_representation_of_3_36_l280_280525


namespace probability_not_all_same_l280_280875

/-- What is the probability that when we roll five fair 6-sided dice, they won't all show the same number? -/
theorem probability_not_all_same :
  let total_outcomes := 6^5 in
  let same_number_outcomes := 6 in
  let probability_all_same := same_number_outcomes / total_outcomes.to_real in
  1 - probability_all_same = (1295 : ℝ) / 1296 :=
by
  let total_outcomes := 6^5
  let same_number_outcomes := 6
  have probability_all_same := (same_number_outcomes : ℝ) / total_outcomes.to_real
  show 1 - probability_all_same = (1295 : ℝ) / 1296       
  sorry

end probability_not_all_same_l280_280875


namespace area_of_gray_region_l280_280428

variables (r : ℝ) (hi : 0 < r)

/-- Two concentric circles, one with radius r and the other with radius 3r, where the width of the grey region is 4 feet. Prove the area of the grey region -/
theorem area_of_gray_region (h₁ : 3 * r - r = 4) : 
  let A_o := π * (3 * r)^2,
      A_i := π * r^2 in
  (A_o - A_i) = 8 * π * r^2 :=
by
  let A_o := π * (3 * r)^2,
      A_i := π * r^2
  sorry

end area_of_gray_region_l280_280428


namespace rectangle_perimeter_l280_280577

theorem rectangle_perimeter (a b : ℚ) (ha : ¬ a.den = 1) (hb : ¬ b.den = 1) (hab : a ≠ b) (h : (a - 2) * (b - 2) = -7) : 2 * (a + b) = 20 :=
by
  sorry

end rectangle_perimeter_l280_280577


namespace probability_not_all_same_l280_280876

/-- What is the probability that when we roll five fair 6-sided dice, they won't all show the same number? -/
theorem probability_not_all_same :
  let total_outcomes := 6^5 in
  let same_number_outcomes := 6 in
  let probability_all_same := same_number_outcomes / total_outcomes.to_real in
  1 - probability_all_same = (1295 : ℝ) / 1296 :=
by
  let total_outcomes := 6^5
  let same_number_outcomes := 6
  have probability_all_same := (same_number_outcomes : ℝ) / total_outcomes.to_real
  show 1 - probability_all_same = (1295 : ℝ) / 1296       
  sorry

end probability_not_all_same_l280_280876


namespace smallest_positive_angle_l280_280502

def coterminal_angle (θ : ℤ) : ℤ := θ % 360

theorem smallest_positive_angle (θ : ℤ) (hθ : θ % 360 ≠ 0) : 
  0 < coterminal_angle θ ∧ coterminal_angle θ = 158 :=
by
  sorry

end smallest_positive_angle_l280_280502


namespace find_r_l280_280975

theorem find_r (r s : ℝ)
  (h1 : ∀ α β : ℝ, (α + β = -r) ∧ (α * β = s) → 
         ∃ t : ℝ, (t^2 - (α^2 + β^2) * t + (α^2 * β^2) = 0) ∧ |α^2 - β^2| = 8)
  (h_sum : ∃ α β : ℝ, α + β = 10) :
  r = -10 := by
  sorry

end find_r_l280_280975


namespace gcd_factorial_8_6_squared_l280_280104

theorem gcd_factorial_8_6_squared : Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2) = 5760 := by
  sorry

end gcd_factorial_8_6_squared_l280_280104


namespace regular12gon_diagonals_concurrent_l280_280626

-- Define the vertices of the regular 12-gon
structure Regular12Gon :=
(vertices : Fin 12 → ℝ × ℝ)
(regular : ∀ i j, dist (vertices i) (vertices j) = dist (vertices 0) (vertices 1))
-- Define the diagonals
def diagonal (g : Regular12Gon) (i j : Fin 12) : ℝ × ℝ := 
  ((g.vertices i).1 + (g.vertices j).1) / 2, ((g.vertices i).2 + (g.vertices j).2) / 2

-- Prove the concurrency
theorem regular12gon_diagonals_concurrent (g : Regular12Gon) :
  diagonal g 1 8 = diagonal g 11 3 ∧ diagonal g 0 3 = diagonal g 11 3 ∧ diagonal g 1 10 = diagonal g 2 10 :=
sorry

end regular12gon_diagonals_concurrent_l280_280626


namespace max_sum_in_center_color_squares_l280_280760

theorem max_sum_in_center_color_squares : 
  let grid := List.range 25
  let center_color := (3 + 3) % 2
  let same_color_indices := List.filter (λ (i : ℕ), ((i / 5) + (i % 5)) % 2 = center_color) grid
  same_color_indices.length = 13 → 
  (let nums := List.range (25 + 1)
  let colored_nums := same_color_indices.map (λ (i : ℕ), nums.get? i).filter_map id
  (colored_nums.sum = 169)) :=
by
  intro grid center_color same_color_indices h_len nums colored_nums
  sorry

end max_sum_in_center_color_squares_l280_280760


namespace exists_valid_arrangement_l280_280607

-- Definition of the given numbers and the requirement on their arrangement
def numbers : List ℕ := [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]

-- Definition of the triangular arrangement
structure triangle (α : Type) :=
(v1 v2 v3 v4 v5 v6 v7 v8 v9 : α)

-- Condition that the sum of numbers on each side of the triangle must be the same
def valid_triangle_arrangement (t : triangle ℕ) : Prop :=
  let side1_sum := t.v1 + t.v2 + t.v3 in
  let side2_sum := t.v4 + t.v5 + t.v6 in
  let side3_sum := t.v7 + t.v8 + t.v9 in
  list.perm t.toList numbers ∧ 
  side1_sum = side2_sum ∧ side2_sum = side3_sum

-- Statement to prove the existence of a valid arrangement
theorem exists_valid_arrangement :
  ∃ t : triangle ℕ, valid_triangle_arrangement t :=
sorry

end exists_valid_arrangement_l280_280607


namespace sum_of_interior_angles_of_remaining_polygon_l280_280043

theorem sum_of_interior_angles_of_remaining_polygon (n : ℕ) (h1 : 3 ≤ n) (h2 : n ≤ 5) :
  (n - 2) * 180 ≠ 270 :=
by 
  sorry

end sum_of_interior_angles_of_remaining_polygon_l280_280043


namespace simplify_fraction_l280_280528

theorem simplify_fraction (h1 : 3.36 = 3 + 0.36) 
                          (h2 : 0.36 = (36 : ℚ) / 100) 
                          (h3 : (36 : ℚ) / 100 = 9 / 25) 
                          : 3.36 = 84 / 25 := 
by 
  rw [h1, h2, h3]
  norm_num
  rw [←Rat.add_div, show 3 = 75 / 25 by norm_num]
  norm_num
  
  sorry  -- This line can be safely removed when the proof is complete.

end simplify_fraction_l280_280528


namespace number_relatively_prime_to_2001_l280_280713

/-- The Euler's Totient Function, φ(n), counts the number of integers up to n that are relatively prime to n. -/
def euler_totient (n : ℕ) : ℕ :=
  (List.range n).filter (Nat.coprime n) |>.length

def phi_of_2001_calculated : ℕ := euler_totient 2001

theorem number_relatively_prime_to_2001 (n : ℕ) (h : n = 2000) :
  phi_of_2001_calculated = 1232 :=
  sorry

end number_relatively_prime_to_2001_l280_280713


namespace find_a_b_l280_280314

def g (a b x : ℝ) : ℝ := a * x^2 + b * x + sqrt 3

theorem find_a_b : 
  ∃ a b : ℝ, 
  g a b (g a b 1) = 1 ∧ g a b 1 = 2 := 
by
  sorry

end find_a_b_l280_280314


namespace twelfth_nine_position_l280_280275

-- Define sequence of natural numbers written without spaces
def seq := String.join (List.map toString (List.range 1000))

-- Function to count occurrences of '9' up to a given position
def count_nines (str : String) (pos : Nat) : Nat :=
  (str.take pos).toList.count (λ c => c = '9')

-- Proposition that states the position of the 12th '9' is 174
theorem twelfth_nine_position : count_nines seq 174 = 12 :=
  sorry

end twelfth_nine_position_l280_280275


namespace probability_sum_greater_than_four_is_5_over_6_l280_280435

-- Define the sample space for two dice.
def sample_space : Finset (ℕ × ℕ) :=
  Finset.product (Finset.range 6) (Finset.range 6) 

-- Define the event where the sum is greater than four.
def event_sum_greater_than_four : Finset (ℕ × ℕ) :=
  sample_space.filter (λ p, p.1 + p.2 > 4)

-- Compute the probability of the event.
def probability_sum_greater_than_four : ℚ :=
  (event_sum_greater_than_four.card : ℚ) / (sample_space.card : ℚ)

-- Theorem to be proven
theorem probability_sum_greater_than_four_is_5_over_6 :
  probability_sum_greater_than_four = 5 / 6 :=
by
  -- Proof would go here
  sorry

end probability_sum_greater_than_four_is_5_over_6_l280_280435


namespace multiples_count_two_l280_280836

theorem multiples_count_two (n : ℕ) (h : 46 = (finset.Icc 10 100).filter (λ k, k % n = 0).card) : n = 2 := 
by
  have : (finset.Icc 10 100).filter (λ k, k % 2 = 0).card = 46 := 
    by
      simp
  exact eq_of_heq (eq_drec (λ h, h) (this.symm))
  sorry

end multiples_count_two_l280_280836


namespace inscribed_circle_radius_l280_280257

-- Define the given conditions
def radius_large : ℝ := 18
def radius_small : ℝ := 9
def radius_inscribed : ℝ := 8

-- Define tangency conditions and relationships based on the problem statement
def large_semicircle (R : ℝ) := { x : ℝ // 0 <= x ∧ x <= R }
def small_semicircle (r : ℝ) := { x : ℝ // 0 <= x ∧ x <= r }

-- Prove the radius of the circle inscribed between the two semicircles
theorem inscribed_circle_radius :
  large_semicircle radius_large ∧ small_semicircle radius_small →
  ∃ (x : ℝ), x = radius_inscribed := 
by
  intro h;  -- Assume the hypothesis h
  exists radius_inscribed;  -- Show the existence of the radius of the inscribed circle
  have hp1 : sqrt (324 - 36 * radius_inscribed) + radius_small = sqrt (81 + 18 * radius_inscribed) := sorry,
  have hp2 : sqrt (324 - 36 * radius_inscribed) = sqrt (81 + 18 * radius_inscribed) - 9 := sorry,
  have h_sqr : (324 - 36 * radius_inscribed) = (sqrt (81 + 18 * radius_inscribed) - 9)^2 := sorry,
  sorry  -- Proof skipped for simplicity of problem setup

end inscribed_circle_radius_l280_280257


namespace right_triangle_l280_280907

-- Definitions for each set of segments
def setA := (2 : ℕ, 3 : ℕ, 4 : ℕ)
def setB := (Real.sqrt 7, 3 : ℕ, 5 : ℕ)
def setC := (6 : ℕ, 8 : ℕ, 10 : ℕ)
def setD := (5 : ℕ, 12 : ℕ, 12 : ℕ)

-- Main statement that needs to be proven
theorem right_triangle : 
  ¬((setA.1^2 + setA.2^2 = setA.3^2)) ∧
  ¬((setB.1^2 + setB.2^2 = setB.3^2)) ∧
   (setC.1^2 + setC.2^2 = setC.3^2) ∧
  ¬((setD.1^2 + setD.2^2 = setD.3^2)) :=
by 
  sorry

end right_triangle_l280_280907


namespace length_of_AD_in_cyclic_quadrilateral_l280_280735

-- Definitions and conditions
noncomputable def cyclic_quadrilateral {A B C D : Type}
  (cyclic : Prop) (AB_parallel_CD : Prop) (AB : ℝ) (CD : ℝ) (AE : ℝ) (EC : ℝ) (BE : ℝ) (ED : ℝ)
  (ratio_AE_EC : AE / EC = 2) (ratio_BE_ED : BE / ED = 3) 
  (goal : ℝ) : Prop :=
  AB = 3 ∧ CD = 5 ∧ cyclic ∧ AB_parallel_CD ∧ ratio_AE_EC ∧ ratio_BE_ED ∧ goal = 4

-- Main theorem statement
theorem length_of_AD_in_cyclic_quadrilateral :
  ∃ (A B C D : Type), cyclic_quadrilateral true (A, B ∥ D, C) 3 5 2 1 3 1 4 := sorry

end length_of_AD_in_cyclic_quadrilateral_l280_280735


namespace sequentially_decreasing_travel_l280_280922

theorem sequentially_decreasing_travel (n : ℕ) (E : Finset (ℕ × ℕ)) (price : ℕ × ℕ → ℕ)
  (symm : ∀ {a b}, price (a, b) = price (b, a))
  (diff_prices : ∀ {a b c d}, a ≠ b → c ≠ d → (price (a, b) ≠ price (c, d))) :
  ∃ (path : list (ℕ × ℕ)),
  path.length = n - 1 ∧
  ∀ (i : ℕ), i < path.length - 1 → price (path.nth_le i sorry) > price (path.nth_le (i + 1) sorry) := sorry

end sequentially_decreasing_travel_l280_280922


namespace solve_for_x_l280_280230

theorem solve_for_x (x : ℝ) (h : 9 / x^3 = x / 27) : x = 3 * real.root 4 3 :=
by
  sorry

end solve_for_x_l280_280230


namespace alexander_bought_apples_l280_280588

theorem alexander_bought_apples :
  ∀ (A : ℕ), (1 * A + 2 * 2 = 9) → A = 5 :=
begin
  sorry
end

end alexander_bought_apples_l280_280588


namespace find_n_eq_4_l280_280124

theorem find_n_eq_4 (n : ℕ) (h : sin (Real.pi / (3 * n)) + cos (Real.pi / (3 * n)) = Real.sqrt (2 * n) / 2) : n = 4 :=
sorry

end find_n_eq_4_l280_280124


namespace max_intersection_points_for_10_circles_l280_280991

def I (n : ℕ) : ℕ := n * (n - 1)

theorem max_intersection_points_for_10_circles :
  I 10 = 90 :=
by
  -- Definitions of conditions for n = 2 and n = 3
  have hI2 : I 2 = 2 := by sorry,
  have hI3 : I 3 = 6 := by sorry,
  -- Goal statement
  exact calc
    I 10 = 10 * (10 - 1) := by rfl
    ... = 10 * 9 := by rfl
    ... = 90 := by norm_num

end max_intersection_points_for_10_circles_l280_280991


namespace find_f_half_l280_280182

noncomputable def g (x : ℝ) : ℝ := 1 - 2 * x
noncomputable def f (y : ℝ) : ℝ := if y ≠ 0 then (1 - y^2) / y^2 else 0

theorem find_f_half :
  f (g (1 / 4)) = 15 :=
by
  have g_eq : g (1 / 4) = 1 / 2 := sorry
  rw [g_eq]
  have f_eq : f (1 / 2) = 15 := sorry
  exact f_eq

end find_f_half_l280_280182


namespace problem_part1_problem_part2_problem_part3_l280_280670

def f (x : ℝ) : ℝ := 1 / (1 + x)
def g (x : ℝ) : ℝ := x^2 + 2

theorem problem_part1 :
  f 2 = 1 / 3 := 
by 
  unfold f 
  norm_num

theorem problem_part2 :
  g 2 = 6 := 
by 
  unfold g 
  norm_num

theorem problem_part3 :
  f (g 3) = 1 / 12 := 
by 
  unfold f g 
  norm_num

end problem_part1_problem_part2_problem_part3_l280_280670


namespace jill_action_units_digit_l280_280081

def units_digit_frequencies_equal : Prop :=
  ∀ n : ℕ, n < 10 → 
  (let l := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] in
  (∑ J in l, ∑ K in l, if (J + K) % 10 = n then 1 else 0) +
  (∑ J in l, ∑ K in l, if (J - K + 10) % 10 = n then 1 else 0)) =
  2 * (10 * 10 / 10)

theorem jill_action_units_digit : units_digit_frequencies_equal := 
by sorry

end jill_action_units_digit_l280_280081


namespace validate_grid_l280_280243

-- Definitions for primes and the grid
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_grid (grid : List (List ℕ)) : Prop :=
  (∀ row : List ℕ, row ∈ grid → is_prime (row.foldl (.+.) 0)) ∧
  (∀ col_index : ℕ, col_index < 3 →
   is_prime (grid.foldl (λ acc row, acc + row.getD col_index 0) 0))

-- Our specific grid
def grid : List (List ℕ) := [
  [1, 7, 9],
  [2, 6, 3],
  [8, 4, 5]
]

-- The theorem we want to prove
theorem validate_grid : prime_grid grid :=
by
  sorry

end validate_grid_l280_280243


namespace volume_is_44076_l280_280150

noncomputable def central_spherical_layer_thickness : ℝ := Real.sqrt (30^2 - 28^2)
def cylinder_height : ℝ := 20 * Real.sqrt 2
def volume_of_part : ℝ := 
  let V1 := (π * central_spherical_layer_thickness / 6) * (3 * 900 + 3 * 784 + 116)
  let V2 := π * 10^2 * (2 * cylinder_height - central_spherical_layer_thickness)
  let V3 := 2 * π * (30 - cylinder_height) / 6 * (3 * 10^2 + (30 - cylinder_height)^2)
  V1 + V2 + V3

theorem volume_is_44076 : volume_of_part = 44076 := by
  sorry

end volume_is_44076_l280_280150


namespace hall_length_width_difference_l280_280834

theorem hall_length_width_difference (L W : ℝ) 
  (h1 : W = 1 / 2 * L) 
  (h2 : L * W = 128) : 
  L - W = 8 :=
by
  sorry

end hall_length_width_difference_l280_280834


namespace probability_of_event_A_l280_280562

def total_balls : ℕ := 10
def white_balls : ℕ := 7
def black_balls : ℕ := 3

def event_A : Prop := (black_balls / total_balls) * (white_balls / (total_balls - 1)) = 7 / 30

theorem probability_of_event_A : event_A := by
  sorry

end probability_of_event_A_l280_280562


namespace log_xy_eq_5_over_11_l280_280715

-- Definitions of the conditions
axiom log_xy4_eq_one {x y : ℝ} : Real.log (x * y^4) = 1
axiom log_x3y_eq_one {x y : ℝ} : Real.log (x^3 * y) = 1

-- The statement to be proven
theorem log_xy_eq_5_over_11 {x y : ℝ} (h1 : Real.log (x * y^4) = 1) (h2 : Real.log (x^3 * y) = 1) :
  Real.log (x * y) = 5 / 11 :=
by
  sorry

end log_xy_eq_5_over_11_l280_280715


namespace observations_decrement_l280_280839

theorem observations_decrement (n : ℤ) (h_n_pos : n > 0) : 200 - 15 = 185 :=
by
  sorry

end observations_decrement_l280_280839


namespace two_dice_sum_greater_than_four_l280_280439
open Classical

def probability_sum_greater_than_four : ℚ := by sorry

theorem two_dice_sum_greater_than_four :
  probability_sum_greater_than_four = 5 / 6 :=
sorry

end two_dice_sum_greater_than_four_l280_280439


namespace n_is_900_l280_280082

theorem n_is_900 
  (m n : ℕ) 
  (h1 : ∃ x y : ℤ, m = x^2 ∧ n = y^2) 
  (h2 : Prime (m - n)) : n = 900 := 
sorry

end n_is_900_l280_280082


namespace lattice_points_midpoint_l280_280682

-- Define lattice points in ℤ²
structure LatticePoint :=
  (x : ℤ)
  (y : ℤ)

-- Define the main theorem statement
theorem lattice_points_midpoint
  (P1 P2 P3 P4 P5 : LatticePoint) :
  ∃ (i j : ℕ), 1 ≤ i ∧ i < j ∧ j ≤ 5 ∧
  let Pi := [P1, P2, P3, P4, P5].nth (i - 1) (by sorry),
      Pj := [P1, P2, P3, P4, P5].nth (j - 1) (by sorry) in
  (Pi.x + Pj.x) % 2 = 0 ∧ (Pi.y + Pj.y) % 2 = 0 :=
sorry

end lattice_points_midpoint_l280_280682


namespace sq_sum_extension_l280_280495

theorem sq_sum_extension (k : ℕ) (h : 0 < k) :
  1^2 + 2^2 + ⋯ + k^2 + ⋯ + 2^2 + 1^2 + (k + 1)^2 + (k + 1)^2 = 
  1^2 + 2^2 + ⋯ + k^2 + ⋯ + 2^2 + 1^2 + k^2 + (k + 1)^2 :=
sorry

end sq_sum_extension_l280_280495


namespace nico_books_pages_l280_280787

/-- Nico's reading problem: conditions and proof of the number of pages in the fifth book. -/
theorem nico_books_pages :
  let pages_read := 
    (20) +                   -- First book on Monday
    (45 / 2) +               -- Half of the second book on Monday
    (45 / 2) +               -- Remaining half of the second book on Tuesday
    (32) +                   -- Third book on Tuesday
    (60 * (2/5)) +           -- Two-fifths of the fourth book on Tuesday
    (60 * (3/5)) +           -- Three-fifths of the fourth book on Wednesday
    (80 * (1/4) : ℝ)         -- 25% of the sixth book on Wednesday
  in pages_read + (pages_read + (X : ℝ)) = 234 → X = 57 :=
begin
  sorry
end

end nico_books_pages_l280_280787


namespace circle_equation_l280_280693

-- Define the necessary entities
def center : ℝ × ℝ := (2, -3)
def point_on_circle : ℝ × ℝ := (0, 0)

-- Define the equation of a circle
def is_circle_equation (h k r : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r

-- The goal is to prove that the equation is indeed of the form provided
theorem circle_equation :
  ∃ (r : ℝ), r = 13 ∧ is_circle_equation 2 (-3) r 0 0 :=
begin
  -- Construct the radius
  let r := real.sqrt ((2 - 0)^2 + ((-3) - 0)^2),
  -- Show that r^2 = 13 through computation
  have h_r : r = real.sqrt 13 := by {
    unfold r,
    simp,
  },
  -- Now state the full equation
  use 13,
  split,
  { exact h_r },
  { -- Check the point (0, 0) satisfies the equation
    unfold is_circle_equation,
    simp,
    norm_num
  }
end

end circle_equation_l280_280693


namespace closest_number_l280_280903

theorem closest_number (x y : ℝ) (options : set ℝ) (closest : ℝ) : 
  x = 2.015 ∧ y = 510.2 ∧ options = {0.1, 1, 10, 100, 1000} ∧ closest = 1000 →
  ∃ z ∈ options, (abs ((x * y) - closest) ≤ abs ((x * y) - z)) :=
by
  intros h
  have hxy : x = 2.015 ∧ y = 510.2 ∧ options = {0.1, 1, 10, 100, 1000} ∧ closest = 1000 := h
  sorry

end closest_number_l280_280903


namespace quadrilateral_unit_square_bounds_l280_280276

theorem quadrilateral_unit_square_bounds :
  let a b c d : ℝ := ... -- define lengths of the sides of the quadrilateral formed by the points on the sides of unit square
  (2 ≤ a^2 + b^2 + c^2 + d^2) ∧ (a^2 + b^2 + c^2 + d^2 ≤ 4)
  ∧ (2 * (2).sqrt ≤ a + b + c + d) ∧ (a + b + c + d ≤ 4) := 
begin
  sorry
end

end quadrilateral_unit_square_bounds_l280_280276


namespace sally_credit_card_balance_l280_280349

theorem sally_credit_card_balance (G P : ℝ) (X : ℝ)  
  (h1 : P = 2 * G)  
  (h2 : XP = X * P)  
  (h3 : G / 3 + XP = (5 / 12) * P) : 
  X = 1 / 4 :=
by
  sorry

end sally_credit_card_balance_l280_280349


namespace xiaoying_final_score_l280_280909

def speech_competition_score (score_content score_expression score_demeanor : ℕ) 
                             (weight_content weight_expression weight_demeanor : ℝ) : ℝ :=
  score_content * weight_content + score_expression * weight_expression + score_demeanor * weight_demeanor

theorem xiaoying_final_score :
  speech_competition_score 86 90 80 0.5 0.4 0.1 = 87 :=
by 
  sorry

end xiaoying_final_score_l280_280909


namespace average_stoppage_time_is_correct_l280_280840

-- Definitions for bus speeds
def speed_A_excluding := 60 -- kmph
def speed_A_including := 40 -- kmph

def speed_B_excluding := 50 -- kmph
def speed_B_including := 35 -- kmph

def speed_C_excluding := 70 -- kmph
def speed_C_including := 50 -- kmph

-- Stoppage time calculations in minutes
def stoppage_time (speed_excluding speed_including : ℕ) : ℕ :=
  ((speed_excluding - speed_including) * 60) / speed_excluding

def stoppage_time_A := stoppage_time speed_A_excluding speed_A_including -- in minutes
def stoppage_time_B := stoppage_time speed_B_excluding speed_B_including -- in minutes
def stoppage_time_C := stoppage_time speed_C_excluding speed_C_including -- in minutes

-- Average stoppage time calculation
def average_stoppage_time := (stoppage_time_A + stoppage_time_B + stoppage_time_C) / 3

-- The theorem to prove
theorem average_stoppage_time_is_correct : average_stoppage_time = 18.38 := by
  -- Proof goes here
  sorry

end average_stoppage_time_is_correct_l280_280840


namespace probability_not_all_same_l280_280858

theorem probability_not_all_same :
    let total_outcomes := 6 ^ 5 in
    let same_number_outcomes := 6 in
    let p_all_same := same_number_outcomes / total_outcomes in
    let p_not_all_same := 1 - p_all_same in
    p_not_all_same = 1295 / 1296 :=
by
  sorry

end probability_not_all_same_l280_280858


namespace circles_packed_line_equation_l280_280803

/-- Six circles of diameter 2 are packed in the first quadrant with centers
at (1,1), (1,3), (1,5), (3,1), (3,3), and (3,5). A line l with slope 2
passes through these circles and divides the region into two equal areas.
The equation of line l can be written in the form ax = by + c where a, b, and c
are positive integers and gcd(a, b, c) = 1. Prove a^2 + b^2 + c^2 = 6. -/
theorem circles_packed_line_equation :
  ∃ (a b c : ℕ), (a^2 + b^2 + c^2 = 6) ∧ (Nat.gcd a (Nat.gcd b c) = 1) ∧ (a = 2) ∧ (b = 1) ∧ (c = 1) :=
by
  existsi 2, 1, 1
  simp [Nat.gcd, pow_two]
  sorry

end circles_packed_line_equation_l280_280803


namespace probability_sum_greater_than_four_l280_280467

def sum_greater_than_four_probability :=
  (5 / 6: ℚ)

theorem probability_sum_greater_than_four :
  let outcomes := { (a, b) | a in (Finset.range 1 7) ∧ b in (Finset.range 1 7) }
  let favorable_outcomes := outcomes.filter (λ pair, (pair.1 + pair.2) > 4)
  let probability := favorable_outcomes.card.to_rat / outcomes.card.to_rat
  probability = sum_greater_than_four_probability :=
by
  sorry

end probability_sum_greater_than_four_l280_280467


namespace product_of_constants_t_l280_280131

theorem product_of_constants_t (a b : ℤ) (t : ℤ) (h1 : a * b = -12) (h2 : t = a + b) :
  ∃ (ts : Finset ℤ), (ts = {11, 4, 1, -1, -4, -11}) ∧ ts.prod (λ x, x) = -1936 :=
by sorry

end product_of_constants_t_l280_280131


namespace matrix_condition_and_value_l280_280619

variables {a b c : ℝ}

def N : ℝ × ℝ × ℝ × ℝ × ℝ × ℝ × ℝ × ℝ × ℝ :=
  (a,  0, c,
   b, 2c, -a,
   2b, -2c, a)

noncomputable def N_T : ℝ × ℝ × ℝ × ℝ × ℝ × ℝ × ℝ × ℝ × ℝ :=
  (a, b, 2b,
   0, 2c, -2c,
   c, -a, a)

noncomputable def N_T_mul_N : ℝ × ℝ × ℝ × ℝ × ℝ × ℝ × ℝ × ℝ × ℝ :=
  let (a₁₁, a₁₂, a₁₃, a₂₁, a₂₂, a₂₃, a₃₁, a₃₂, a₃₃) := N in
  let (b₁₁, b₁₂, b₁₃, b₂₁, b₂₂, b₂₃, b₃₁, b₃₂, b₃₃) := N_T in
  (a₁₁ * b₁₁ + a₁₂ * b₂₁ + a₁₃ * b₃₁, a₁₁ * b₁₂ + a₁₂ * b₂₂ + a₁₃ * b₃₂, a₁₁ * b₁₃ + a₁₂ * b₂₃ + a₁₃ * b₃₃,
   a₂₁ * b₁₁ + a₂₂ * b₂₁ + a₂₃ * b₃₁, a₂₁ * b₁₂ + a₂₂ * b₂₂ + a₂₃ * b₃₂, a₂₁ * b₁₃ + a₂₂ * b₂₃ + a₂₃ * b₃₃,
   a₃₁ * b₁₁ + a₃₂ * b₂₁ + a₃₃ * b₃₁, a₃₁ * b₁₂ + a₃₂ * b₂₂ + a₃₃ * b₃₂, a₃₁ * b₁₃ + a₃₂ * b₂₃ + a₃₃ * b₃₃)

noncomputable def I : ℝ × ℝ × ℝ × ℝ × ℝ × ℝ × ℝ × ℝ × ℝ :=
  (1, 0, 0,
   0, 1, 0,
   0, 0, 1)

theorem matrix_condition_and_value (h: N_T_mul_N = I) : a^2 + b^2 + c^2 = 27 / 40 := by
  sorry

end matrix_condition_and_value_l280_280619


namespace average_of_last_three_numbers_l280_280379

theorem average_of_last_three_numbers (nums : List ℝ) (h_len : nums.length = 6) 
    (h_avg6 : nums.sum / 6 = 60) (h_avg3 : (nums.take 3).sum / 3 = 55) : 
    ((nums.drop 3).sum) / 3 = 65 := 
sorry

end average_of_last_three_numbers_l280_280379


namespace continuous_function_proof_l280_280923

noncomputable def exists_special_x (f : ℝ → ℝ) (n : ℕ) : Prop :=
  n > 1 → ∃ x, 0 ≤ x ∧ x + (1 : ℝ) / n ≤ 1 ∧ f x = f (x + (1 : ℝ) / n)

theorem continuous_function_proof (f : ℝ → ℝ) (h_cont : ContinuousOn f (Icc 0 1)) (h_boundary : f 0 = 0 ∧ f 1 = 0) : 
  ∀ n, exists_special_x f n :=
  by
  sorry

end continuous_function_proof_l280_280923


namespace average_speed_is_3_l280_280358

def TabbysSwimSpeed : ℝ := 1
def TabbysRunSpeed : ℝ := 6

theorem average_speed_is_3.5 :
  (TabbysSwimSpeed + TabbysRunSpeed) / 2 = 3.5 := by
  sorry

end average_speed_is_3_l280_280358


namespace train_length_calc_l280_280007

/-!
# Problem Statement
Prove that the length of the train is 119.97 meters, given the train runs
at a speed of 48 km/hr and crosses a pole in 9 seconds.
-/

noncomputable def convert_speed_to_m_per_s (speed_kmh : ℝ) : ℝ :=
  speed_kmh * (1000 / 3600)

noncomputable def train_length (speed_kmh : ℝ) (time_seconds : ℝ) : ℝ :=
  convert_speed_to_m_per_s(speed_kmh) * time_seconds

theorem train_length_calc (speed_kmh : ℝ) (time_seconds : ℝ) (expected_length : ℝ) :
  speed_kmh = 48 → time_seconds = 9 → expected_length = 119.97 →
  train_length speed_kmh time_seconds = expected_length :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end train_length_calc_l280_280007


namespace pastries_made_l280_280595

theorem pastries_made (P cakes_sold pastries_sold extra_pastries : ℕ)
  (h1 : cakes_sold = 78)
  (h2 : pastries_sold = 154)
  (h3 : extra_pastries = 76)
  (h4 : pastries_sold = cakes_sold + extra_pastries) :
  P = 154 := sorry

end pastries_made_l280_280595


namespace total_time_spent_l280_280750

theorem total_time_spent (woody_time_years : ℝ) (ivanka_extra_months : ℕ) (months_in_year : ℕ) :
  woody_time_years = 1.5 →
  ivanka_extra_months = 3 →
  months_in_year = 12 →
  let woody_time_months := woody_time_years * months_in_year,
      ivanka_time_months := woody_time_months + ivanka_extra_months,
      alice_time_months := woody_time_months / 2,
      tom_time_months := alice_time_months * 2,
      total_time := ivanka_time_months + woody_time_months + alice_time_months + tom_time_months
  in total_time = 66 :=
by
  intros
  sorry

end total_time_spent_l280_280750


namespace probability_sum_greater_than_four_l280_280477

theorem probability_sum_greater_than_four :
  let all_outcomes := (Fin 6) × (Fin 6)
  let favorable_outcomes := {p : Fin 6 × Fin 6 | (p.1.val + 1) + (p.2.val + 1) > 4}
  (favorable_outcomes.card : ℚ) / ((Fin 6 × Fin 6).card : ℚ) = 5 / 6 :=
by
  sorry

end probability_sum_greater_than_four_l280_280477


namespace min_value_proof_l280_280179

noncomputable def min_value (a b : ℝ) : ℝ :=
  if ha : a > 1 ∧ b > 0 ∧ a + b = 2 then
    min_point (λ a b, (1 / (a - 1) + 2 / b))
  else
    0

theorem min_value_proof :
  ∃ a b : ℝ, a > 1 ∧ b > 0 ∧ a + b = 2 ∧ (1 / (a - 1) + 2 / b) = 3 + 2 * sqrt 2 :=
sorry

end min_value_proof_l280_280179


namespace find_integers_satisfying_conditions_l280_280996

-- Define the quadratic function
def quadratic_fn (x : ℤ) : ℤ := 2 * x * x + x - 6

-- Define what it means to be a power of a prime integer
def is_prime_power (n : ℤ) : Prop :=
  ∃ (p : ℤ) (k : ℕ), nat.prime p ∧ (n = p ^ k)

-- Main theorem statement
theorem find_integers_satisfying_conditions :
  { x : ℤ | is_prime_power (quadratic_fn x) } = {-3, 2, 5} :=
by
  sorry

end find_integers_satisfying_conditions_l280_280996


namespace dealer_profit_percent_l280_280933

-- Define the actual weight in grams.
def actual_weight : ℕ := 1000

-- Define the weight used by the dealer in grams.
def dealer_weight : ℕ := 880

-- Define the profit in grams.
def profit : ℕ := actual_weight - dealer_weight

-- Define the profit percent as a real number.
def profit_percent : ℝ := (profit / actual_weight.toReal) * 100

-- Prove that the dealer's profit percent is 12%.
theorem dealer_profit_percent : profit_percent = 12 := by
  sorry

end dealer_profit_percent_l280_280933


namespace num_ways_second_defective_third_non_defective_num_ways_end_testing_400_cost_l280_280590

variable (products : Finset ℕ)
variable (defective non_defective : ℕ)
variable (select_second_defective select_third_non_defective : Prop)
variable (testing_cost : ℕ)

-- Given conditions
axiom conditions : 
  products.card = 6 ∧
  defective = 2 ∧
  non_defective = 4 ∧
  select_second_defective ∧
  select_third_non_defective ∧
  testing_cost = 100

-- Questions as proof statements

-- Part 1: Number of ways to select products in the given condition
theorem num_ways_second_defective_third_non_defective :
  conditions products defective non_defective select_second_defective select_third_non_defective testing_cost →
  ∃ (ways : ℕ), ways = 120 := 
  by
  intro h
  sorry

-- Part 2: Number of ways to end the testing with total cost $400
theorem num_ways_end_testing_400_cost :
  conditions products defective non_defective select_second_defective select_third_non_defective testing_cost →
  ∃ (ways : ℕ), ways = 96 :=
  by
  intro h
  sorry

end num_ways_second_defective_third_non_defective_num_ways_end_testing_400_cost_l280_280590


namespace gcd_factorial_l280_280114

theorem gcd_factorial (a b : ℕ) : 
    ∃ (g : ℕ), nat.gcd (nat.factorial 8) ((nat.factorial 6) ^ 2) = g ∧ g = 5760 := 
by 
  let g := nat.gcd (nat.factorial 8) ((nat.factorial 6) ^ 2)
  existsi (5760 : ℕ)
  split
  · sorry
  · rfl

end gcd_factorial_l280_280114


namespace min_bottles_needed_l280_280057

open Real

-- Conditions as definitions in Lean
def fluidOuncesToLiters (oz : ℝ) : ℝ := oz / 33.8
def litersToMilliLiters (liters : ℝ) : ℝ := liters * 1000
def minBottles (mL : ℝ) (bottleSize : ℝ) : ℕ := (ceil (mL / bottleSize)).toNat

-- Theorem statement
theorem min_bottles_needed :
  (minBottles (litersToMilliLiters (fluidOuncesToLiters 60)) 250) = 8 :=
  by
    sorry

end min_bottles_needed_l280_280057


namespace trajectory_of_point_M_is_circle_l280_280166

theorem trajectory_of_point_M_is_circle :
  ∀ (x y : ℝ), (sqrt (x^2 + y^2) / sqrt ((x - 3)^2 + y^2) = 1 / 2) →
  (∃ (center : ℝ × ℝ) (radius : ℝ), center = (1, 0) ∧ radius = 2 ∧ (x - center.1)^2 + y^2 = radius^2) :=
by
  intros x y h
  sorry

end trajectory_of_point_M_is_circle_l280_280166


namespace probability_sum_greater_than_four_is_5_over_6_l280_280434

-- Define the sample space for two dice.
def sample_space : Finset (ℕ × ℕ) :=
  Finset.product (Finset.range 6) (Finset.range 6) 

-- Define the event where the sum is greater than four.
def event_sum_greater_than_four : Finset (ℕ × ℕ) :=
  sample_space.filter (λ p, p.1 + p.2 > 4)

-- Compute the probability of the event.
def probability_sum_greater_than_four : ℚ :=
  (event_sum_greater_than_four.card : ℚ) / (sample_space.card : ℚ)

-- Theorem to be proven
theorem probability_sum_greater_than_four_is_5_over_6 :
  probability_sum_greater_than_four = 5 / 6 :=
by
  -- Proof would go here
  sorry

end probability_sum_greater_than_four_is_5_over_6_l280_280434


namespace product_of_t_l280_280133

theorem product_of_t (a b : ℤ) (t : ℤ) (h1 : a * b = -12) (h2 : t = a + b) :
  ∃ (t_values : Finset ℤ), 
  (∀ x ∈ t_values, ∃ a b : ℤ, a * b = -12 ∧ x = a + b) ∧ 
  (t_values.product = -1936) :=
by
  sorry

end product_of_t_l280_280133


namespace continuous_tape_length_l280_280634

theorem continuous_tape_length :
  let num_sheets := 15
  let sheet_length_cm := 25
  let overlap_cm := 0.5 
  let total_length_without_overlap := num_sheets * sheet_length_cm
  let num_overlaps := num_sheets - 1
  let total_overlap_length := num_overlaps * overlap_cm
  let total_length_cm := total_length_without_overlap - total_overlap_length
  let total_length_m := total_length_cm / 100
  total_length_m = 3.68 := 
by {
  sorry
}

end continuous_tape_length_l280_280634


namespace axis_of_symmetry_of_g_l280_280953
noncomputable theory

def f (x : ℝ) : ℝ := Real.sin (2 * x - π / 6)

def g (x : ℝ) : ℝ := Real.sin (2 * (x + π / 3) - π / 6)

theorem axis_of_symmetry_of_g : ∃ k : ℤ, 2 * (k * π / 2) = π / 2 :=
by sorry

end axis_of_symmetry_of_g_l280_280953


namespace ratio_area_circle_trapezoid_inscribed_l280_280951

theorem ratio_area_circle_trapezoid_inscribed 
    (α β : ℝ) 
    (hα : 0 < α ∧ α < π) 
    (hβ : 0 < β ∧ β < π) 
    : ∃ S_C S_T, S_C / S_T = π / (2 * (sin α)^2 * sin (2 * β)) :=
by
  sorry

end ratio_area_circle_trapezoid_inscribed_l280_280951


namespace expansion_constant_term_l280_280983

theorem expansion_constant_term : 
  let expr := (x - 1 / x) ^ 4 in
  let constant_term := 6 in
  is_constant_term expr constant_term :=
sorry

end expansion_constant_term_l280_280983


namespace simplify_336_to_fraction_l280_280522

theorem simplify_336_to_fraction : (336 / 100) = (84 / 25) :=
by sorry

end simplify_336_to_fraction_l280_280522


namespace min_value_f_range_of_a_l280_280772
open Real

noncomputable def f (x : ℝ) : ℝ := (x + 1) * log (x + 1)
noncomputable def g (a x : ℝ) : ℝ := a * x^2 + x

theorem min_value_f : infimum (range f) = -1 / exp 1 := 
sorry

theorem range_of_a (a : ℝ) : (∀ x ≥ 0, f x ≤ g a x) ↔ a ≥ 1 / 2 := 
sorry

end min_value_f_range_of_a_l280_280772


namespace marble_problem_l280_280083

theorem marble_problem 
  (a b r1 r2 b1 b2 p q : ℕ)
  (ha : a + b = 30)
  (hp_red : r1 * r2 * 9 = a * b * 4)
  (hr1 : r1 + b1 = a)
  (hr2 : r2 + b2 = b)
  (hpq : p + q = 10)
  : ∃ (p q : ℕ), nat.gcd p q = 1 ∧ b1 * b2 * q = a * b * p :=
begin
  sorry
end

end marble_problem_l280_280083


namespace find_integers_satisfying_conditions_l280_280995

-- Define the quadratic function
def quadratic_fn (x : ℤ) : ℤ := 2 * x * x + x - 6

-- Define what it means to be a power of a prime integer
def is_prime_power (n : ℤ) : Prop :=
  ∃ (p : ℤ) (k : ℕ), nat.prime p ∧ (n = p ^ k)

-- Main theorem statement
theorem find_integers_satisfying_conditions :
  { x : ℤ | is_prime_power (quadratic_fn x) } = {-3, 2, 5} :=
by
  sorry

end find_integers_satisfying_conditions_l280_280995


namespace solution_sets_and_range_l280_280552

theorem solution_sets_and_range 
    (x a : ℝ) 
    (A : Set ℝ)
    (M : Set ℝ) :
    (∀ x, x ∈ A ↔ 1 ≤ x ∧ x ≤ 4) ∧
    (M = {x | (x - a) * (x - 2) ≤ 0} ) ∧
    (M ⊆ A) → (1 ≤ a ∧ a ≤ 4) :=
by
  sorry

end solution_sets_and_range_l280_280552


namespace product_of_t_l280_280134

theorem product_of_t (a b : ℤ) (t : ℤ) (h1 : a * b = -12) (h2 : t = a + b) :
  ∃ (t_values : Finset ℤ), 
  (∀ x ∈ t_values, ∃ a b : ℤ, a * b = -12 ∧ x = a + b) ∧ 
  (t_values.product = -1936) :=
by
  sorry

end product_of_t_l280_280134


namespace number_of_red_circles_l280_280078

-- Define the problem setup in Lean
def sequence_of_circles : Type := list char
def is_valid_sequence (s : sequence_of_circles) : Prop :=
  (∀ (i : ℕ), i < s.length - 2 → ('R' ∈ s[i:i+3])) ∧ -- At least one red in every triplet
  (∀ (i : ℕ), i < s.length - 3 → ('B' ∈ s[i:i+4])) ∧ -- At least one blue in every quadruplet
  (s.count 'G' > s.length / 2) -- More than half are green circles

def solution : {s : sequence_of_circles // s.length = 11 ∧ is_valid_sequence s} :=
  { val := ['G', 'G', 'R', 'B', 'G', 'R', 'G', 'B', 'R', 'G', 'G'],
    property := sorry } -- Fill in property proof (is_valid_sequence evidence)

-- The proof statement that there are exactly 3 red circles
theorem number_of_red_circles : ∀ s : {s : sequence_of_circles // s.length = 11 ∧ is_valid_sequence s},
  s.val.count 'R' = 3 :=
by
  intros s,
  -- Proof content to be filled here
  sorry

end number_of_red_circles_l280_280078


namespace sum_of_common_ratios_l280_280315

theorem sum_of_common_ratios (k p r a2 a3 b2 b3 : ℝ)
  (h1 : a3 = k * p^2) (h2 : a2 = k * p) 
  (h3 : b3 = k * r^2) (h4 : b2 = k * r)
  (h5 : p ≠ r)
  (h6 : 3 * a3 - 4 * b3 = 5 * (3 * a2 - 4 * b2)) :
  p + r = 5 :=
by {
  sorry
}

end sum_of_common_ratios_l280_280315


namespace tangent_line_at_1_min_value_of_f_l280_280700

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := -1/x + 2 * x - 3 * Real.log x

-- The derivative of the function f
noncomputable def f' (x : ℝ) : ℝ := 1 / x^2 + 2 - 3 / x

-- Simplify computing f(1)
noncomputable def f_at_1 : ℝ := f 1

-- Prove that the equation of the tangent line to f(x) at the point (1, f(1)) is y = 1.
theorem tangent_line_at_1 :
  ∃ k b : ℝ, k = f' 1 ∧ b = f 1 ∧ ∀ x, f x = k * (x - 1) + b := sorry

-- Prove that the minimum value of f(x) for x in [1/e, 2] is 1.
theorem min_value_of_f :
  ∃ x_min ∈ Set.Icc (1 / Real.e) 2, (∀ x ∈ Set.Icc (1 / Real.e) 2, f x_min ≤ f x) ∧ f x_min = 1 := sorry

end tangent_line_at_1_min_value_of_f_l280_280700


namespace bricks_in_chimney_l280_280053

noncomputable def totalBricks : ℕ := 150

-- Defining individual rates
def brendaRate (h : ℕ) : ℚ := h / 9
def brandonRate (h : ℕ) : ℚ := h / 10
def bradyRate (h : ℕ) : ℚ := h / 18

-- Defining combined rate with decreased output
def combinedRate (h : ℕ) : ℚ := brendaRate h + brandonRate h + bradyRate h - 15

-- Final theorem statement
theorem bricks_in_chimney (h : ℕ) : combinedRate h * 6 = h → h = totalBricks :=
by
  intro h
  have : combinedRate h * 6 = h
  sorry

end bricks_in_chimney_l280_280053


namespace surface_area_of_prism_is_40_l280_280945

noncomputable def side_length_of_base (r : ℝ) (h : ℝ) : ℝ :=
  let a := Real.sqrt ((r^2 - h^2/4) / 2) in
  a

def surface_area_of_prism (a h : ℝ) : ℝ :=
  2 * a^2 + 4 * a * h

theorem surface_area_of_prism_is_40 :
  let r := Real.sqrt 6
  let h := 4
  let a := side_length_of_base r h
  a = 2 →
  surface_area_of_prism a h = 40 :=
by
  intros r h a a_eq
  rw [side_length_of_base, a_eq, surface_area_of_prism]
  sorry

end surface_area_of_prism_is_40_l280_280945


namespace total_floor_area_is_correct_l280_280422

-- Define the combined area of the three rugs
def combined_area_of_rugs : ℕ := 212

-- Define the area covered by exactly two layers of rug
def area_covered_by_two_layers : ℕ := 24

-- Define the area covered by exactly three layers of rug
def area_covered_by_three_layers : ℕ := 24

-- Define the total floor area covered by the rugs
def total_floor_area_covered : ℕ :=
  combined_area_of_rugs - area_covered_by_two_layers - 2 * area_covered_by_three_layers

-- The theorem stating the total floor area covered
theorem total_floor_area_is_correct : total_floor_area_covered = 140 := by
  sorry

end total_floor_area_is_correct_l280_280422


namespace base_555_ABAB_eq_6_l280_280899

def base_in_ABAB_form (n b : ℕ) : Prop :=
  let digits := (n / b^3, (n % b^3) / b^2, (n % b^2) / b, n % b) in
  digits.1 = digits.3 ∧ digits.2 = digits.4 ∧ digits.1 ≠ digits.2

theorem base_555_ABAB_eq_6 :
  ∃ b : ℕ, (5 ≤ b ∧ b ≤ 8) ∧ (b^4 > 555 ∧ 555 ≥ b^3) ∧ base_in_ABAB_form 555 b :=
by {
  use 6,
  split,
  exact ⟨by norm_num, by norm_num⟩,
  split,
  norm_num,
  split,
  sorry -- This part would involve computational checks similar to what was manually done
}

end base_555_ABAB_eq_6_l280_280899


namespace quadratic_equation_root_conditions_quadratic_equation_distinct_real_roots_l280_280706

theorem quadratic_equation_root_conditions
  (k : ℝ)
  (h_discriminant : 4 * k - 3 > 0)
  (h_sum_product : ∀ (x1 x2 : ℝ),
    x1 + x2 = -(2 * k + 1) ∧ 
    x1 * x2 = k^2 + 1 →
    x1 + x2 + 2 * (x1 * x2) = 1) :
  k = 1 :=
by
  sorry

theorem quadratic_equation_distinct_real_roots
  (k : ℝ) :
  (∃ (x1 x2 : ℝ),
    x1 ≠ x2 ∧
    x1^2 + (2 * k + 1) * x1 + (k^2 + 1) = 0 ∧
    x2^2 + (2 * k + 1) * x2 + (k^2 + 1) = 0) ↔
  k > 3 / 4 :=
by
  sorry

end quadratic_equation_root_conditions_quadratic_equation_distinct_real_roots_l280_280706


namespace max_regions_by_four_lines_l280_280292

theorem max_regions_by_four_lines : ∀ (n : ℕ), n = 4 → (n * (n + 1) / 2) + 1 = 11 :=
by
  intros n hn
  rw hn
  norm_num
  sorry

end max_regions_by_four_lines_l280_280292


namespace total_blocks_per_day_l280_280345

def blocks_to_park : ℕ := 4
def blocks_to_hs : ℕ := 7
def blocks_to_home : ℕ := 11
def walks_per_day : ℕ := 3

theorem total_blocks_per_day :
  (blocks_to_park + blocks_to_hs + blocks_to_home) * walks_per_day = 66 :=
by
  sorry

end total_blocks_per_day_l280_280345


namespace probability_red_buttons_l280_280754

/-- 
Initial condition: Jar A contains 6 red buttons and 10 blue buttons.
Carla removes the same number of red buttons as blue buttons from Jar A and places them in Jar B.
Jar A's state after action: Jar A retains 3/4 of its original number of buttons.
Question: What is the probability that both selected buttons are red? Express your answer as a common fraction.
-/
theorem probability_red_buttons :
  let initial_red_a := 6
  let initial_blue_a := 10
  let total_buttons_a := initial_red_a + initial_blue_a
  
  -- Jar A after removing buttons
  let retained_fraction := 3 / 4
  let remaining_buttons_a := retained_fraction * total_buttons_a
  let removed_buttons := total_buttons_a - remaining_buttons_a
  let removed_red_buttons := removed_buttons / 2
  let removed_blue_buttons := removed_buttons / 2
  
  -- Remaining red and blue buttons in Jar A
  let remaining_red_a := initial_red_a - removed_red_buttons
  let remaining_blue_a := initial_blue_a - removed_blue_buttons

  -- Total remaining buttons in Jar A
  let total_remaining_a := remaining_red_a + remaining_blue_a

  -- Jar B contains the removed buttons
  let total_buttons_b := removed_buttons
  
  -- Probability calculations
  let probability_red_a := remaining_red_a / total_remaining_a
  let probability_red_b := removed_red_buttons / total_buttons_b

  -- Combined probability of selecting red button from both jars
  probability_red_a * probability_red_b = 1 / 6 :=
by
  sorry

end probability_red_buttons_l280_280754


namespace greatest_perimeter_10_pieces_iso_triangle_l280_280357

def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := 
  real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

def perimeter (k : ℕ) : ℝ :=
  1 + distance 0 12 k 0 + distance 0 12 (k + 1) 0

theorem greatest_perimeter_10_pieces_iso_triangle :
  let P_max := (1 + real.sqrt (12^2 + 9^2) + real.sqrt (12^2 + 10^2)) 
  P_max = 31.62 :=
by
  sorry

end greatest_perimeter_10_pieces_iso_triangle_l280_280357


namespace third_stick_length_l280_280730

theorem third_stick_length (x : ℝ) (h1 : 2 > 0) (h2 : 5 > 0) (h3 : 3 < x) (h4 : x < 7) : x = 4 :=
by
  sorry

end third_stick_length_l280_280730


namespace inscribed_circle_radius_l280_280264

theorem inscribed_circle_radius (R r : ℝ) (hR : R = 18) (hr : r = 9) : ∃ x : ℝ, x = 8 := 
by
  use 8
  sorry


end inscribed_circle_radius_l280_280264


namespace M_is_listed_correctly_l280_280213

noncomputable def M : Set ℕ := { m | ∃ n : ℕ+, 3 / (5 - m : ℝ) = n }

theorem M_is_listed_correctly : M = { 2, 4 } :=
by
  sorry

end M_is_listed_correctly_l280_280213


namespace inscribed_circle_radius_l280_280267

theorem inscribed_circle_radius (R r x : ℝ) (hR : R = 18) (hr : r = 9) :
  x = 8 :=
sorry

end inscribed_circle_radius_l280_280267


namespace center_of_circumcircle_lies_on_PQ_l280_280015

open EuclideanGeometry

variables {A B C B1 B2 C1 C2 P Q : Point}

-- We define the triangle ABC
noncomputable def triangleABC (A B C : Point) (h : ¬(A = B ∨ B = C ∨ C = A)) (h2 : ∠(B, A, C) = ∠(A, C, B) = ∠(C, B, A) = 90) : Triangle := Triangle.mk A B C

-- We define the perpendicular bisectors intersections
noncomputable def bisectorIntersections (A B C B1 B2 C1 C2 : Point) (h : ¬is_isosceles (triangleABC A B C h)) : intersections := 
{ B1 := perpendicular_bisector_intersection (segment A C) (line B C)
, B2 := perpendicular_bisector_intersection (segment A C) (line B C)
, C1 := perpendicular_bisector_intersection (segment A B) (line B C)
, C2 := perpendicular_bisector_intersection (segment A B) (line B C)
}

-- We define the circumcircles and their intersections
variables (ωb ωc : Circle)
noncomputable def circumcircles (B B1 B2 C C1 C2 : Point) : circles :=
{ ωb := circumscribed_circle_triangle B B1 B2
, ωc := circumscribed_circle_triangle C C1 C2 }

noncomputable def circumcircle_intersections (ωb ωc : Circle) : intersections :=
{ P := circle_intersection ωb ωc
, Q := second_circle_intersection ωb ωc P }

-- Finally, prove that the center of the circumcircle of triangle ABC lies on line PQ
theorem center_of_circumcircle_lies_on_PQ 
  {A B C B1 B2 C1 C2 P Q : Point}
  (h1 : acute_triangle A B C)
  (h2 : perpendicular_bisector_intersects_lines A B C B1 B2 C1 C2)
  (h3 : circumcircles_intersect_at_two_points B B1 B2 C C1 C2 P Q) :
  ∃ O : Point, is_circumcenter O (triangleABC A B C h1) ∧ lies_on_line O (line_through P Q)
:= sorry

end center_of_circumcircle_lies_on_PQ_l280_280015


namespace original_price_of_petrol_l280_280540

variable (P : ℝ)

theorem original_price_of_petrol (h : 0.9 * (200 / P - 200 / (0.9 * P)) = 5) : 
  (P = 20 / 4.5) :=
sorry

end original_price_of_petrol_l280_280540


namespace cauliflower_production_proof_l280_280004

theorem cauliflower_production_proof (x y : ℕ) 
  (h1 : y^2 - x^2 = 401)
  (hx : x > 0)
  (hy : y > 0) :
  y^2 = 40401 :=
by
  sorry

end cauliflower_production_proof_l280_280004


namespace correct_statements_about_sample_l280_280797

def sample_data : List ℕ := [92, 93, 95, 95, 97, 98]

def mean (data : List ℕ) : ℝ := (data.sum : ℝ) / data.length

def range (data : List ℕ) : ℝ := (data.maximum' - data.minimum')

def variance (data : List ℕ) : ℝ :=
  let μ := mean data
  (data.map (fun x => (x - μ) ^ 2)).sum / data.length

def percentile (data : List ℕ) (p : ℝ) : ℕ :=
  let sorted := data.sort
  sorted.get ⌈p * sorted.length⌉.to_nat.pred

theorem correct_statements_about_sample :
    mean sample_data = 95
  ∧ range sample_data = 6
  ∧ variance sample_data ≠ 26
  ∧ percentile sample_data 0.80 = 97 :=
by
  sorry

end correct_statements_about_sample_l280_280797


namespace translate_symmetric_y_axis_l280_280846

theorem translate_symmetric_y_axis (m : ℝ) (h : m > 0) :
  ∀ (x : ℝ), sqrt 3 * cos (x + m) + sin (x + m) = sqrt 3 * cos (-x + m) + sin (-x + m) → m = π / 6 :=
by
    sorry

end translate_symmetric_y_axis_l280_280846


namespace angle_sum_180_l280_280738

def convex_quadrilateral := {ABCD : Type*} -- definition simplification placeholder

theorem angle_sum_180
  (ABCD: convex_quadrilateral)
  (A B C D X : Type) -- points type placeholders
  (h1: AB * CD = BC * DA)
  (h2: ∠XAB = ∠XCD)
  (h3: ∠XBC = ∠XDA)
  : ∠BXA + ∠DXC = 180 :=
by
  sorry

end angle_sum_180_l280_280738


namespace reciprocal_F_is_D_l280_280817

-- Define the complex number F inside the unit circle in the second quadrant
def F : ℂ := sorry

-- Assumptions
axiom F_in_unit_circle : ∥F∥ < 1 
axiom F_in_second_quadrant : realPart F < 0 ∧ imaginaryPart F < 0

-- Statement of the problem
theorem reciprocal_F_is_D :
  let reciprocal_F := 1 / F
  in (realPart reciprocal_F < 0 ∧ imaginaryPart reciprocal_F > 0 ∧ ∥reciprocal_F∥ > 1) →
  reciprocal_F = D := sorry

end reciprocal_F_is_D_l280_280817


namespace train_passes_jogger_in_36_seconds_l280_280937

-- Define the speeds in kmph
def speed_jogger_kmph : ℝ := 9
def speed_train_kmph : ℝ := 45

-- Convert speeds from kmph to m/s
def speed_jogger_mps : ℝ := speed_jogger_kmph * (1000 / 3600)
def speed_train_mps : ℝ := speed_train_kmph * (1000 / 3600)

-- Define the head start and train length in meters
def head_start_meters : ℝ := 240
def train_length_meters : ℝ := 120

-- Calculate the relative speed
def relative_speed_mps : ℝ := speed_train_mps - speed_jogger_mps

-- Calculate the total distance to be covered
def total_distance_meters : ℝ := head_start_meters + train_length_meters

-- Calculate the time taken to pass the jogger
def time_to_pass_seconds : ℝ := total_distance_meters / relative_speed_mps

-- The statement to be proven
theorem train_passes_jogger_in_36_seconds 
: time_to_pass_seconds = 36 := by
  sorry

end train_passes_jogger_in_36_seconds_l280_280937


namespace solve_arcsin_sin_l280_280805

theorem solve_arcsin_sin (x : ℝ) (h : -Real.pi / 2 ≤ x ∧ x ≤ Real.pi / 2) :
  Real.arcsin (Real.sin (2 * x)) = x ↔ x = 0 ∨ x = Real.pi / 3 ∨ x = -Real.pi / 3 :=
by
  sorry

end solve_arcsin_sin_l280_280805


namespace gcd_fact8_fact6_squared_l280_280120

-- Definition of 8! and (6!)²
def fact8 : ℕ := 8!
def fact6_squared : ℕ := (6!)^2

-- The theorem statement to be proved
theorem gcd_fact8_fact6_squared : Nat.gcd fact8 fact6_squared = 11520 := 
by
    sorry

end gcd_fact8_fact6_squared_l280_280120


namespace gcd_factorial_l280_280115

theorem gcd_factorial (a b : ℕ) : 
    ∃ (g : ℕ), nat.gcd (nat.factorial 8) ((nat.factorial 6) ^ 2) = g ∧ g = 5760 := 
by 
  let g := nat.gcd (nat.factorial 8) ((nat.factorial 6) ^ 2)
  existsi (5760 : ℕ)
  split
  · sorry
  · rfl

end gcd_factorial_l280_280115


namespace simplify_fraction_l280_280532

theorem simplify_fraction (h1 : 3.36 = 3 + 0.36) 
                          (h2 : 0.36 = (36 : ℚ) / 100) 
                          (h3 : (36 : ℚ) / 100 = 9 / 25) 
                          : 3.36 = 84 / 25 := 
by 
  rw [h1, h2, h3]
  norm_num
  rw [←Rat.add_div, show 3 = 75 / 25 by norm_num]
  norm_num
  
  sorry  -- This line can be safely removed when the proof is complete.

end simplify_fraction_l280_280532


namespace pow_units_digit_six_probability_l280_280779

noncomputable def units_digit_of_pow_eq_six_prob : ℚ :=
  let ms := {12, 14, 16, 18, 20} in
  let ns := {2005, 2006, ..., 2024} in
  if ∀ m ∈ ms, ∀ n ∈ ns, (m^n % 10 = 6) then 
    (1 : ℚ) / 5 
  else 
    0

theorem pow_units_digit_six_probability :
  units_digit_of_pow_eq_six_prob = (1 : ℚ) / 5 :=
sorry

end pow_units_digit_six_probability_l280_280779


namespace harmonic_series_inequality_l280_280850

theorem harmonic_series_inequality (n : ℕ) (h : 2 ≤ n) : 
  ∑ k in Finset.range n, (1 / ((2^k) - 1) : ℝ) < n :=
sorry

end harmonic_series_inequality_l280_280850


namespace roses_difference_l280_280842

def roses_in_vase (n_initial n_thrown n_current n_cut : ℕ) : Prop :=
  n_initial = 21 ∧ n_thrown = 34 ∧ n_current = 15 ∧ 
  n_current = n_cut - (n_thrown - n_initial)

theorem roses_difference (n_cut : ℕ) : 
  roses_in_vase 21 34 15 n_cut → (34 - n_cut = 19) :=
by
  intro h,
  unfold roses_in_vase at h,
  cases h with h1 htmp,
  cases htmp with h2 htmp',
  cases htmp' with h3 h4,
  sorry

end roses_difference_l280_280842


namespace altitude_of_triangle_l280_280673

theorem altitude_of_triangle (a b : ℝ) (h : ℝ) :
  let d := real.sqrt(a^2 + b^2)
  ab * 2 = 0.5 * d * h ↔ h = 4 * a * b / d :=
by sorry

end altitude_of_triangle_l280_280673


namespace clock_angle_at_550_l280_280853

def hour : ℕ := 5
def minute : ℕ := 50
def calculate_angle (h: ℕ) (m: ℕ) : ℕ := (abs (60 * h - 11 * m)) / 2

theorem clock_angle_at_550 :
  calculate_angle hour minute = 125 :=
by
  sorry

end clock_angle_at_550_l280_280853


namespace gcd_factorial_8_6_squared_l280_280106

theorem gcd_factorial_8_6_squared : Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2) = 5760 := by
  sorry

end gcd_factorial_8_6_squared_l280_280106


namespace correct_statements_about_sample_l280_280798

def sample_data : List ℕ := [92, 93, 95, 95, 97, 98]

def mean (data : List ℕ) : ℝ := (data.sum : ℝ) / data.length

def range (data : List ℕ) : ℝ := (data.maximum' - data.minimum')

def variance (data : List ℕ) : ℝ :=
  let μ := mean data
  (data.map (fun x => (x - μ) ^ 2)).sum / data.length

def percentile (data : List ℕ) (p : ℝ) : ℕ :=
  let sorted := data.sort
  sorted.get ⌈p * sorted.length⌉.to_nat.pred

theorem correct_statements_about_sample :
    mean sample_data = 95
  ∧ range sample_data = 6
  ∧ variance sample_data ≠ 26
  ∧ percentile sample_data 0.80 = 97 :=
by
  sorry

end correct_statements_about_sample_l280_280798


namespace convex_polyhedron_formula_l280_280629

theorem convex_polyhedron_formula
  (V E F t h T H : ℕ)
  (hF : F = 40)
  (hFaces : F = t + h)
  (hVertex : 2 * T + H = 7)
  (hEdges : E = (3 * t + 6 * h) / 2)
  (hEuler : V - E + F = 2)
  : 100 * H + 10 * T + V = 367 := 
sorry

end convex_polyhedron_formula_l280_280629


namespace gcd_factorial_8_6_squared_l280_280105

theorem gcd_factorial_8_6_squared : Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2) = 5760 := by
  sorry

end gcd_factorial_8_6_squared_l280_280105


namespace simplify_336_to_fraction_l280_280518

theorem simplify_336_to_fraction : (336 / 100) = (84 / 25) :=
by sorry

end simplify_336_to_fraction_l280_280518


namespace upper_lower_inequality_l280_280717

-- Define what an upper family and a lower family are for an n-element set
variables {X : Type} [Fintype X] {n : ℕ} (U D: Set (Set X))

def is_upper_family (U : Set (Set X)) : Prop :=
∀ (A B : Set X), A ⊆ B → A ∈ U → B ∈ U

def is_lower_family (D : Set (Set X)) : Prop :=
∀ (A B : Set X), B ⊆ A → A ∈ D → B ∈ D

-- Define the main inequality statement
theorem upper_lower_inequality (hU : is_upper_family U) (hD : is_lower_family D) :
  (Fintype.card U) * (Fintype.card D) ≥ 2^n * Fintype.card (U ∩ D) :=
sorry

end upper_lower_inequality_l280_280717


namespace james_weekly_expenses_l280_280298

noncomputable def utility_cost (rent: ℝ):  ℝ := 0.2 * rent
noncomputable def weekly_hours_open (hours_per_day: ℕ) (days_per_week: ℕ): ℕ := hours_per_day * days_per_week
noncomputable def employee_weekly_wages (wage_per_hour: ℝ) (weekly_hours: ℕ): ℝ := wage_per_hour * weekly_hours
noncomputable def total_employee_wages (employees: ℕ) (weekly_wages: ℝ): ℝ := employees * weekly_wages
noncomputable def total_weekly_expenses (rent: ℝ) (utilities: ℝ) (employee_wages: ℝ): ℝ := rent + utilities + employee_wages

theorem james_weekly_expenses : 
  let rent := 1200
  let utility_percentage := 0.2
  let hours_per_day := 16
  let days_per_week := 5
  let employees := 2
  let wage_per_hour := 12.5
  let weekly_hours := weekly_hours_open hours_per_day days_per_week
  let utilities := utility_cost rent
  let employee_wages_per_week := employee_weekly_wages wage_per_hour weekly_hours
  let total_employee_wages_per_week := total_employee_wages employees employee_wages_per_week
  total_weekly_expenses rent utilities total_employee_wages_per_week = 3440 := 
by
  sorry

end james_weekly_expenses_l280_280298


namespace find_C_find_a_plus_b_l280_280668

-- Define the setup for the triangle and the vectors
variables {a b c : ℝ}
variables (C : ℝ)
variables (area : ℝ := 4 * real.sqrt 3 / 3)
variables (m : EuclideanSpace ℝ (fin 2) := ![(real.cos (C / 2)), (real.sin (C / 2))])
variables (n : EuclideanSpace ℝ (fin 2) := ![(real.cos (C / 2)), -(real.sin (C / 2))])

-- Conditions
axiom h_vec_angle : inner m n = (1 / 2)

-- Given conditions for part (2)
axiom h_c : c = 3
axiom h_area : area = (1 / 2 * a * b * (real.sin C))

-- Questions posed as proof goals
theorem find_C : C = π / 3 :=
by
  sorry

noncomputable def area_triangle_s : ℝ := (1/2) * a * b * real.sin C
theorem find_a_plus_b (h1 : C = π / 3) (h2 : area_triangle_s = 4 * real.sqrt 3 / 3) (h3 : c = 3) : a + b = 5 :=
by
  sorry

end find_C_find_a_plus_b_l280_280668


namespace find_integers_l280_280998

noncomputable def f (x : ℤ) : ℤ := 2 * x^2 + x - 6

def is_prime_power (n : ℤ) : Prop :=
  ∃ p k : ℕ, p.prime ∧ k > 0 ∧ n = (p : ℤ)^k

theorem find_integers (x : ℤ) :
  is_prime_power (f x) ↔ x = -3 ∨ x = 2 ∨ x = 5 := 
sorry

end find_integers_l280_280998


namespace Nell_initial_cards_l280_280786

theorem Nell_initial_cards (given_away : ℕ) (now_has : ℕ) : 
  given_away = 276 → now_has = 252 → (now_has + given_away) = 528 :=
by
  intros h_given_away h_now_has
  sorry

end Nell_initial_cards_l280_280786


namespace probability_sum_greater_than_four_l280_280465

def sum_greater_than_four_probability :=
  (5 / 6: ℚ)

theorem probability_sum_greater_than_four :
  let outcomes := { (a, b) | a in (Finset.range 1 7) ∧ b in (Finset.range 1 7) }
  let favorable_outcomes := outcomes.filter (λ pair, (pair.1 + pair.2) > 4)
  let probability := favorable_outcomes.card.to_rat / outcomes.card.to_rat
  probability = sum_greater_than_four_probability :=
by
  sorry

end probability_sum_greater_than_four_l280_280465


namespace randy_piggy_bank_balance_l280_280343

def initial_amount : ℕ := 200
def store_trip_cost : ℕ := 2
def trips_per_month : ℕ := 4
def extra_cost_trip : ℕ := 1
def extra_trip_interval : ℕ := 3
def months_in_year : ℕ := 12
def weekly_income : ℕ := 15
def internet_bill_per_month : ℕ := 20
def birthday_gift : ℕ := 100
def weeks_in_year : ℕ := 52

-- To be proved
theorem randy_piggy_bank_balance : 
  initial_amount 
  + (weekly_income * weeks_in_year) 
  + birthday_gift 
  - ((store_trip_cost * trips_per_month * months_in_year)
  + (months_in_year / extra_trip_interval) * extra_cost_trip
  + (internet_bill_per_month * months_in_year))
  = 740 :=
by
  sorry

end randy_piggy_bank_balance_l280_280343


namespace chocolate_bars_l280_280005

theorem chocolate_bars (num_small_boxes : ℕ) (num_bars_per_box : ℕ) (total_bars : ℕ) (h1 : num_small_boxes = 20) (h2 : num_bars_per_box = 32) (h3 : total_bars = num_small_boxes * num_bars_per_box) :
  total_bars = 640 :=
by
  sorry

end chocolate_bars_l280_280005


namespace solve_linear_equation_l280_280388

theorem solve_linear_equation :
  ∀ x : ℝ, (2 * x + 1) / 3 - (5 * x - 1) / 6 = 1 → x = -3 :=
by
  sorry

end solve_linear_equation_l280_280388


namespace number_of_arrangements_l280_280681

open Function

-- Defining the problem in Lean
def exactly_two_hits (balls boxes : Fin 5 → Fin 5) : Prop :=
  (Finset.filter (λ i, balls i = boxes i) (Finset.univ : Finset (Fin 5))).card = 2

def valid_arrangement_count : ℕ :=
  Finset.card (Finset.filter exactly_two_hits (Finset.allFunctionsOfFin 5))

theorem number_of_arrangements (balls boxes : Fin 5 → Fin 5) :
  valid_arrangement_count = 60 := sorry

end number_of_arrangements_l280_280681


namespace units_digit_of_factorial_sum_l280_280649

theorem units_digit_of_factorial_sum : 
  (1 + 2 + 6 + 4) % 10 = 3 := sorry

end units_digit_of_factorial_sum_l280_280649


namespace probability_correct_l280_280452

def outcome (s₁ s₂ : ℕ) : Prop := s₁ ≥ 1 ∧ s₁ ≤ 6 ∧ s₂ ≥ 1 ∧ s₂ ≤ 6

def sum_outcome_greater_than_four (s₁ s₂ : ℕ) : Prop := outcome s₁ s₂ ∧ s₁ + s₂ > 4

def total_outcomes : ℕ := 36

def favorable_outcomes : ℕ := 30 -- As derived from 36 - 6

def probability_sum_greater_than_four : ℚ := favorable_outcomes / total_outcomes

theorem probability_correct : probability_sum_greater_than_four = 5 / 6 := 
by 
  sorry

end probability_correct_l280_280452


namespace inscribed_circles_radius_l280_280732

theorem inscribed_circles_radius (R r : ℝ) (h1 : R > 0) (h2 : r > 0)
  (h3 : ∀ x, 0 ≤ x ∧ x ≤ 2 * R) 
  (h4 : r = (ℝ.sqrt 2 - 1) * R) : r = (ℝ.sqrt 2 - 1) * R :=
by
  sorry

end inscribed_circles_radius_l280_280732


namespace cube_sufficient_but_not_necessary_l280_280162

theorem cube_sufficient_but_not_necessary (x : ℝ) : (x^3 > 27 → |x| > 3) ∧ (¬(|x| > 3 → x^3 > 27)) :=
by
  sorry

end cube_sufficient_but_not_necessary_l280_280162


namespace construct_triangle_from_excircles_l280_280069

variable (O1 O2 O3 : Point)

noncomputable def excircle_triangle (O1 O2 O3 : Point) : Triangle :=
  let A := altitude_foot O1 O2 O3
  let B := altitude_foot O2 O3 O1
  let C := altitude_foot O3 O1 O2
  Triangle.mk A B C

theorem construct_triangle_from_excircles :
  ∃ (A B C : Point), 
  let Δ := Triangle.mk A B C,
  is_excircle_center Δ O1 ∧ is_excircle_center Δ O2 ∧ is_excircle_center Δ O3 :=
sorry

end construct_triangle_from_excircles_l280_280069


namespace alices_number_in_possible_numbers_l280_280045

-- Definitions based on conditions
def is_factor (m n : ℕ) : Prop := ∃ k, m = n * k
def is_multiple (m n : ℕ) : Prop := ∃ k, m = k * n

-- Define the number that Alice is thinking of, satisfying given conditions
def alices_number (n : ℕ) : Prop :=
  is_factor n 40 ∧    -- Condition 1: Has 40 as a factor
  is_multiple n 72 ∧  -- Condition 2: Is a multiple of 72
  1000 ≤ n ∧ n ≤ 3000 -- Condition 3: Is between 1000 and 3000

-- The set of possible numbers Alice could be thinking of
def possible_numbers : set ℕ :=
  {1080, 1440, 1800, 2160, 2520, 2880}

-- The theorem statement
theorem alices_number_in_possible_numbers (n : ℕ) :
  alices_number n → n ∈ possible_numbers :=
sorry

end alices_number_in_possible_numbers_l280_280045


namespace probability_sum_greater_than_four_l280_280469

theorem probability_sum_greater_than_four : 
  let total_outcomes := 36 in
  let favorable_outcomes := total_outcomes - 6 in
  let probability := favorable_outcomes / total_outcomes in
  probability = 5 / 6 :=
by
  sorry

end probability_sum_greater_than_four_l280_280469


namespace g_of_2_equals_5_l280_280392

noncomputable def g : ℝ → ℝ := sorry

theorem g_of_2_equals_5 (g_property : ∀ x y : ℝ, x * g(y) = y * g(x))
                        (g_at_10 : g(10) = 25) :
  g(2) = 5 :=
begin
  sorry
end

end g_of_2_equals_5_l280_280392


namespace smallest_number_of_pencils_l280_280351

theorem smallest_number_of_pencils 
  (p : ℕ) 
  (h1 : p % 6 = 5)
  (h2 : p % 7 = 3)
  (h3 : p % 8 = 7) :
  p = 35 := 
sorry

end smallest_number_of_pencils_l280_280351


namespace solution_for_a_l280_280186

theorem solution_for_a : 
  ∀ (a : ℝ), 
  (∀ x, (x + a)^2 * (2 * x - 1 / x)^5 ∉ has_term x^3) → a = 1 ∨ a = -1 :=
by
  sorry

end solution_for_a_l280_280186


namespace ratio_BE_EC_l280_280246

theorem ratio_BE_EC
  (A B C F G E : Type)
  [IsTriangle A B C]
  (F_divides_AC : divides_ratio F A C (2, 3))
  (G_midpoint_BF : is_midpoint G B F)
  (E_intersection_BC_AG : E ∈ intersection (line B C) (line A G)) :
  divides_ratio E B C (2, 5) := sorry

end ratio_BE_EC_l280_280246


namespace gcd_bc_minimum_l280_280310

theorem gcd_bc_minimum
  (a b c : ℕ)
  (h1 : Nat.gcd a b = 360)
  (h2 : Nat.gcd a c = 1170)
  (h3 : ∃ k1 : ℕ, b = 5 * k1)
  (h4 : ∃ k2 : ℕ, c = 13 * k2) : Nat.gcd b c = 90 :=
by
  sorry

end gcd_bc_minimum_l280_280310


namespace angle_between_a_and_b_is_2pi_over_3_l280_280218

noncomputable def angle_between_vectors (a b : EuclideanSpace ℝ (Fin 3)) (h1 : ‖a‖ = 2) (h2 : ‖b‖ = 2)
  (h3 : dot_product b (2 • a + b) = 0) : Real :=
  Real.arccos (dot_product a b / (‖a‖ * ‖b‖))

theorem angle_between_a_and_b_is_2pi_over_3 {a b : EuclideanSpace ℝ (Fin 3)}
  (h1 : ‖a‖ = 2) (h2 : ‖b‖ = 2) (h3 : dot_product b (2 • a + b) = 0) :
  angle_between_vectors a b h1 h2 h3 = 2 * Real.pi / 3 :=
by
  sorry

end angle_between_a_and_b_is_2pi_over_3_l280_280218


namespace garden_area_eq_450_l280_280569

theorem garden_area_eq_450
  (width length : ℝ)
  (fencing : ℝ := 60) 
  (length_eq_twice_width : length = 2 * width)
  (fencing_eq : 2 * width + length = fencing) :
  width * length = 450 := by
  sorry

end garden_area_eq_450_l280_280569


namespace fractions_are_integers_l280_280778

theorem fractions_are_integers
  (a b c : ℤ)
  (h : (a * b) / c + (a * c) / b + (b * c) / a ∈ ℤ) :
  ((a * b) / c ∈ ℤ) ∧ ((a * c) / b ∈ ℤ) ∧ ((b * c) / a ∈ ℤ) := 
sorry

end fractions_are_integers_l280_280778


namespace berry_average_temperature_l280_280598

def sunday_temp : ℝ := 99.1
def monday_temp : ℝ := 98.2
def tuesday_temp : ℝ := 98.7
def wednesday_temp : ℝ := 99.3
def thursday_temp : ℝ := 99.8
def friday_temp : ℝ := 99.0
def saturday_temp : ℝ := 98.9

def total_temp : ℝ := sunday_temp + monday_temp + tuesday_temp + wednesday_temp + thursday_temp + friday_temp + saturday_temp
def average_temp : ℝ := total_temp / 7

theorem berry_average_temperature : average_temp = 99 := by
  sorry

end berry_average_temperature_l280_280598


namespace train_cross_time_20_seconds_l280_280582

noncomputable def train_cross_pole (L : ℝ) (v : ℝ) : ℝ :=
  L / v

theorem train_cross_time_20_seconds :
  train_cross_pole 400 20 = 20 :=
by
  unfold train_cross_pole
  norm_num
  sorry

end train_cross_time_20_seconds_l280_280582


namespace isosceles_triangle_perimeter_correct_l280_280826

-- Definitions based on conditions
def equilateral_triangle_side_length (perimeter : ℕ) : ℕ :=
  perimeter / 3

def isosceles_triangle_perimeter (side1 side2 base : ℕ) : ℕ :=
  side1 + side2 + base

-- Given conditions
def equilateral_triangle_perimeter : ℕ := 45
def equilateral_triangle_side : ℕ := equilateral_triangle_side_length equilateral_triangle_perimeter

-- The side of the equilateral triangle is also a leg of the isosceles triangle
def isosceles_triangle_leg : ℕ := equilateral_triangle_side
def isosceles_triangle_base : ℕ := 10

-- The problem to prove
theorem isosceles_triangle_perimeter_correct : 
  isosceles_triangle_perimeter isosceles_triangle_leg isosceles_triangle_leg isosceles_triangle_base = 40 :=
by
  sorry

end isosceles_triangle_perimeter_correct_l280_280826


namespace distance_between_Andrey_and_Gleb_l280_280822

def Andrey_Borya_distance : ℕ := 600
def Vova_Gleb_distance : ℕ := 600
def AG_is_3_times_BV : Prop :=
  ∃ (BV : ℕ), (600 + BV + 600 = 3 * BV)

theorem distance_between_Andrey_and_Gleb :
  ∃ (AG : ℕ), (AG = 1800 ∨ AG = 1500) :=
by
  use 1800
  use 1500
  sorry

end distance_between_Andrey_and_Gleb_l280_280822


namespace team_b_wins_probability_l280_280407

theorem team_b_wins_probability :
  let p_A := 2 / 3 -- probability of Team A winning a set
  let p_B := 1 / 3 -- probability of Team B winning a set
  let match_probability := 
    p_B + p_A * p_B + (p_A)^2 * p_B -- compute the probability of Team B winning the match
  in match_probability = 19 / 27 :=
by
  -- conditions
  let p_A := (2 : ℚ) / 3
  let p_B := (1 : ℚ) / 3
  -- set calculation
  let match_probability := p_B + p_A * p_B + (p_A)^2 * p_B
  -- claim
  exact Eq.refl _
  -- the actual proof goes here
  sorry

end team_b_wins_probability_l280_280407


namespace angle_ACD_measure_l280_280739

theorem angle_ACD_measure {ABD BAE ABC ACD : ℕ} 
  (h1 : ABD = 125) 
  (h2 : BAE = 95) 
  (h3 : ABC = 180 - ABD) 
  (h4 : ABD + ABC = 180 ) : 
  ACD = 180 - (BAE + ABC) :=
by 
  sorry

end angle_ACD_measure_l280_280739


namespace sum_of_possible_values_of_B_l280_280942

theorem sum_of_possible_values_of_B 
    (h1: ∀ n: ℕ, (∀ d: Nat, d ∈ (Nat.digits 10 n) -> Nat.mod (Nat.sum (Nat.digits 10 n)) 9 = 0 → Nat.mod n 9 = 0)) 
    (h2: ∀ n: ℕ, n = 567403B05 -> Nat.mod n 9 = 0) :
    ∑ x in ({0, 1, 2, 3, 4, 5, 6, 7, 8, 9} : Finset Nat).filter (λ B, Nat.mod (30 + B) 9 = 0), x = 18 :=
by
    sorry

end sum_of_possible_values_of_B_l280_280942


namespace sum_integer_solutions_l280_280809

def condition_inequality (x : ℝ) :=
  8 * ((|x + 3| - |x - 5|) / (|2 * x - 11| - |2 * x + 7|)) -
  9 * ((|x + 3| + |x - 5|) / (|2 * x - 11| + |2 * x + 7|)) ≥ -8

def condition_interval (x : ℝ) :=
  |x| < 90

theorem sum_integer_solutions :
  let S := {x : ℤ | condition_inequality x ∧ condition_interval x} in
  (∑ x in S.to_finset, x) = 8 :=
by
  sorry

end sum_integer_solutions_l280_280809


namespace gcd_factorial_l280_280113

theorem gcd_factorial (a b : ℕ) : 
    ∃ (g : ℕ), nat.gcd (nat.factorial 8) ((nat.factorial 6) ^ 2) = g ∧ g = 5760 := 
by 
  let g := nat.gcd (nat.factorial 8) ((nat.factorial 6) ^ 2)
  existsi (5760 : ℕ)
  split
  · sorry
  · rfl

end gcd_factorial_l280_280113


namespace units_digit_sum_factorials_l280_280654

theorem units_digit_sum_factorials : 
  (∑ n in finset.range 2011, (n.factorial % 10)) % 10 = 3 := 
by
  sorry

end units_digit_sum_factorials_l280_280654


namespace inscribed_circle_radius_l280_280258

-- Define the given conditions
def radius_large : ℝ := 18
def radius_small : ℝ := 9
def radius_inscribed : ℝ := 8

-- Define tangency conditions and relationships based on the problem statement
def large_semicircle (R : ℝ) := { x : ℝ // 0 <= x ∧ x <= R }
def small_semicircle (r : ℝ) := { x : ℝ // 0 <= x ∧ x <= r }

-- Prove the radius of the circle inscribed between the two semicircles
theorem inscribed_circle_radius :
  large_semicircle radius_large ∧ small_semicircle radius_small →
  ∃ (x : ℝ), x = radius_inscribed := 
by
  intro h;  -- Assume the hypothesis h
  exists radius_inscribed;  -- Show the existence of the radius of the inscribed circle
  have hp1 : sqrt (324 - 36 * radius_inscribed) + radius_small = sqrt (81 + 18 * radius_inscribed) := sorry,
  have hp2 : sqrt (324 - 36 * radius_inscribed) = sqrt (81 + 18 * radius_inscribed) - 9 := sorry,
  have h_sqr : (324 - 36 * radius_inscribed) = (sqrt (81 + 18 * radius_inscribed) - 9)^2 := sorry,
  sorry  -- Proof skipped for simplicity of problem setup

end inscribed_circle_radius_l280_280258


namespace probability_of_perfect_square_is_correct_l280_280576

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def probability_perfect_square (p : ℚ) : ℚ :=
  let less_than_equal_60 := 7 * p
  let greater_than_60 := 4 * 4 * p
  less_than_equal_60 + greater_than_60

theorem probability_of_perfect_square_is_correct :
  let p : ℚ := 1 / 300
  probability_perfect_square p = 23 / 300 :=
sorry

end probability_of_perfect_square_is_correct_l280_280576


namespace two_dice_sum_greater_than_four_l280_280443
open Classical

def probability_sum_greater_than_four : ℚ := by sorry

theorem two_dice_sum_greater_than_four :
  probability_sum_greater_than_four = 5 / 6 :=
sorry

end two_dice_sum_greater_than_four_l280_280443


namespace probability_not_all_same_l280_280861

theorem probability_not_all_same :
    let total_outcomes := 6 ^ 5 in
    let same_number_outcomes := 6 in
    let p_all_same := same_number_outcomes / total_outcomes in
    let p_not_all_same := 1 - p_all_same in
    p_not_all_same = 1295 / 1296 :=
by
  sorry

end probability_not_all_same_l280_280861


namespace kolya_play_time_l280_280758

theorem kolya_play_time (t : ℕ) (h1 : t < 720) 
  (h2 : ∃ k : ℕ, 5.5 * t = 180 + 360 * k) : 
  t = 360 := 
sorry

end kolya_play_time_l280_280758


namespace price_of_sundae_l280_280911

theorem price_of_sundae (total_ice_cream_bars : ℕ) (total_sundae_price : ℝ)
                        (total_price : ℝ) (price_per_ice_cream_bar : ℝ) (num_ice_cream_bars : ℕ) (num_sundaes : ℕ)
                        (h1 : total_ice_cream_bars = num_ice_cream_bars)
                        (h2 : total_price = 200)
                        (h3 : price_per_ice_cream_bar = 0.40)
                        (h4 : num_ice_cream_bars = 200)
                        (h5 : num_sundaes = 200)
                        (h6 : total_ice_cream_bars * price_per_ice_cream_bar + total_sundae_price = total_price) :
  total_sundae_price / num_sundaes = 0.60 :=
sorry

end price_of_sundae_l280_280911


namespace quad_area_is_one_l280_280417

-- Define the rectangle and the circles
variable (O1 O2 A B C D : ℝ) -- Points on the plane
variable (r : ℝ) -- Radius of the circles
variable (width : ℝ) -- Width of the rectangle 

-- Given conditions
variable (h_width : width = 1) -- The width of the rectangle is 1 cm
variable (h_radius : r = 1) -- The radius of both circles is 1 cm
variable (h_intersection : (C, D) ∈ lineSegment O1 O2) -- C and D are points of intersection on the segment O1 O2

-- The proof goal
theorem quad_area_is_one :
  area ℚ (ABCD) = 1 :=
by sorry

end quad_area_is_one_l280_280417


namespace probability_sum_greater_than_four_l280_280475

theorem probability_sum_greater_than_four : 
  let total_outcomes := 36 in
  let favorable_outcomes := total_outcomes - 6 in
  let probability := favorable_outcomes / total_outcomes in
  probability = 5 / 6 :=
by
  sorry

end probability_sum_greater_than_four_l280_280475


namespace proof_problem_l280_280497

def a : ℕ := 5^2
def b : ℕ := a^4

theorem proof_problem : b = 390625 := 
by 
  sorry

end proof_problem_l280_280497


namespace weight_of_second_piece_l280_280029

-- Given conditions
def area (length : ℕ) (width : ℕ) : ℕ := length * width

def weight (density : ℚ) (area : ℕ) : ℚ := density * area

-- Given dimensions and weight of the first piece
def length1 : ℕ := 4
def width1 : ℕ := 3
def area1 : ℕ := area length1 width1
def weight1 : ℚ := 18

-- Given dimensions of the second piece
def length2 : ℕ := 6
def width2 : ℕ := 4
def area2 : ℕ := area length2 width2

-- Uniform density implies a proportional relationship between area and weight
def density1 : ℚ := weight1 / area1

-- The main theorem to prove
theorem weight_of_second_piece :
  weight density1 area2 = 36 :=
by
  -- use sorry to skip the proof
  sorry

end weight_of_second_piece_l280_280029


namespace sum_of_ages_is_18_l280_280303

-- Define the conditions
def product_of_ages (kiana twin : ℕ) := kiana * twin^2 = 128

-- Define the proof problem statement
theorem sum_of_ages_is_18 : ∃ (kiana twin : ℕ), product_of_ages kiana twin ∧ twin > kiana ∧ kiana + twin + twin = 18 :=
by
  sorry

end sum_of_ages_is_18_l280_280303


namespace estimate_total_fish_l280_280398

theorem estimate_total_fish 
  (caught_first : ℕ) (marked : ℕ)
  (caught_second : ℕ) (marked_second : ℕ)
  (proportion : (marked_second : ℚ) / caught_second = marked / (caught_first : ℚ))
  (caught_first = 30) (marked = 30)
  (caught_second = 50) (marked_second = 2) : 
  (∃ N : ℕ, (marked : ℚ) / N = proportion) ∧ N = 750 :=
by
  sorry

end estimate_total_fish_l280_280398


namespace ellipse_dot_product_inequality_l280_280194

theorem ellipse_dot_product_inequality :
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ (2 * a = √2 * 2 * b)
  ∧ (4 / a^2 + 2 / b^2 = 1)
  ∧ (∃ x y : ℝ, (x, y) = (2, √2) ∧  x^2 / 8 + y^2 / 4 = 1)
  ∧ (∀ A B : ℝ × ℝ, (A.1^2) / 8 + (A.2^2) / 4 = 1
  ∧ (B.1^2) / 8 + (B.2^2) / 4 = 1
  ∧ (∃ C D : ℝ × ℝ, (C.1^2) / 8 + (C.2^2) / 4 = 1
  ∧ (D.1^2) / 8 + (D.2^2) / 4 = 1
  ∧ ∃ k1 k2 : ℝ, k1 * k2 = -1/2
  ∧ (0, 0) = (0, 0)
  ∧ -2 ≤ (A.1 * B.1 + A.2 * B.2) ∧ (A.1 * B.1 + A.2 * B.2)< 2)) 
: sorry

end ellipse_dot_product_inequality_l280_280194


namespace eccentricity_of_hyperbola_l280_280704

def hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) : ℝ := 
  let c := 4
  let e := c / a
  e

theorem eccentricity_of_hyperbola :
  ∀ (a b : ℝ) (ha : a > 0) (hb : b > 0),
  (∀ (A F B : ℝ × ℝ),
    A = (-a, 0) →
    F = (4, 0) →
    B = (0, b) →
    let BA := (-(a : ℝ), -b)
    let BF := (4, -b)
    ( ∀ (dot_product : ℝ),
      dot_product = BA.1 * BF.1 + BA.2 * BF.2 →
      dot_product = 2 * a →
      b^2 = 6 * a
    ) →
    a = 2 → 
    hyperbola_eccentricity a b ha hb = 2
  )
| a b ha hb := 
  assume A F B,
  assume hA : A = (-a, 0),
  assume hF : F = (4, 0),
  assume hB : B = (0, b),
  let BA := (-(a : ℝ), -b),
  let BF := (4, -b),
  assume dot_product,
  assume hdot : dot_product = BA.1 * BF.1 + BA.2 * BF.2,
  assume hdot_eq : dot_product = 2 * a,
  show (b^2 = 6 * a) → (a = 2) → hyperbola_eccentricity a b ha hb = 2, by sorry

end eccentricity_of_hyperbola_l280_280704


namespace smallest_N_for_m_l280_280729

theorem smallest_N_for_m (N : ℕ) (numbers : Fin N → ℕ) :
  (∀ i, numbers i ≤ 100) →  -- Each number is between 0 and 100
  (∃ i, numbers i = 66 ∧ 66 = (2 / 3 : ℚ) * ((∑ j, numbers j) / N : ℚ)) → -- Condition for winning
  N = 34 :=
by
  sorry

end smallest_N_for_m_l280_280729


namespace multiply_exponents_l280_280966

theorem multiply_exponents (a : ℝ) : (6 * a^2) * (1/2 * a^3) = 3 * a^5 := by
  sorry

end multiply_exponents_l280_280966


namespace popsicle_sticks_ratio_l280_280802

/-- Sam, Sid, and Steve brought popsicle sticks for their group activity in their Art class. Sid has twice as many popsicle sticks as Steve. If Steve has 12 popsicle sticks and they can use 108 popsicle sticks for their Art class activity, prove that the ratio of the number of popsicle sticks Sam has to the number Sid has is 3:1. -/
theorem popsicle_sticks_ratio (Sid Sam Steve : ℕ) 
    (h1 : Sid = 2 * Steve) 
    (h2 : Steve = 12) 
    (h3 : Sam + Sid + Steve = 108) : 
    Sam / Sid = 3 :=
by 
    -- Proof steps go here
    sorry

end popsicle_sticks_ratio_l280_280802


namespace initial_number_of_girls_l280_280355

theorem initial_number_of_girls (p g : ℕ) (h1 : g = 0.5 * p) (h2 : (g - 3) / p = 0.4) : g = 15 :=
by sorry

end initial_number_of_girls_l280_280355


namespace triangle_area_is_39_l280_280096

noncomputable def vec2 (x y : ℝ) : Matrix (Fin 2) (Fin 1) ℝ :=
  ![![x], ![y]]

def vertices := (vec2 (-2) 3, vec2 8 (-1), vec2 10 6)

def vector_CA := vertices.1 - vertices.3
def vector_CB := vertices.2 - vertices.3

def determinant_2x2 (v w : Matrix (Fin 2) (Fin 1) ℝ) : ℝ :=
  v[0,0] * w[1,0] - v[1,0] * w[0,0]

def area_of_parallelogram := determinant_2x2 vector_CA vector_CB

def triangle_area := area_of_parallelogram / 2

theorem triangle_area_is_39 : triangle_area = 39 := by
  sorry

end triangle_area_is_39_l280_280096


namespace probability_not_all_same_l280_280857

theorem probability_not_all_same :
    let total_outcomes := 6 ^ 5 in
    let same_number_outcomes := 6 in
    let p_all_same := same_number_outcomes / total_outcomes in
    let p_not_all_same := 1 - p_all_same in
    p_not_all_same = 1295 / 1296 :=
by
  sorry

end probability_not_all_same_l280_280857


namespace variance_of_scores_is_correct_l280_280944

theorem variance_of_scores_is_correct : 
  let scores := [9, 10, 8, 10, 8] in 
  let mean := (scores.sum / scores.length : ℝ) in 
  let variance := (scores.map (λ x, (x - mean) ^ 2)).sum / scores.length in
  variance = 4 / 5 :=
sorry

end variance_of_scores_is_correct_l280_280944


namespace percentage_concentration_acid_l280_280420

-- Definitions based on the given conditions
def volume_acid : ℝ := 1.6
def total_volume : ℝ := 8.0

-- Lean statement to prove the percentage concentration is 20%
theorem percentage_concentration_acid : (volume_acid / total_volume) * 100 = 20 := by
  sorry

end percentage_concentration_acid_l280_280420


namespace average_last_three_l280_280376

/-- The average of the last three numbers is 65, given that the average of six numbers is 60
  and the average of the first three numbers is 55. -/
theorem average_last_three (a b c d e f : ℝ) (h1 : (a + b + c + d + e + f) / 6 = 60) (h2 : (a + b + c) / 3 = 55) :
  (d + e + f) / 3 = 65 :=
by
  sorry

end average_last_three_l280_280376


namespace mark_weekly_reading_time_l280_280782

-- Define the conditions
def hours_per_day : ℕ := 2
def days_per_week : ℕ := 7
def additional_hours : ℕ := 4

-- State the main theorem to prove
theorem mark_weekly_reading_time : (hours_per_day * days_per_week) + additional_hours = 18 := 
by
  -- The proof steps are omitted as per instructions
  sorry

end mark_weekly_reading_time_l280_280782


namespace problem_1_problem_2_problem_3_l280_280217

variable (x k y : ℝ)

def a : ℝ × ℝ := (4^x + 1, 2^x)
def b : ℝ × ℝ := (y - 1, y - k)
def f (x k : ℝ) : ℝ := (4^x + k * 2^x + 1) / (4^x + 2^x + 1)

theorem problem_1 (h : (a x) ⬝ (b y k) = 0) : y = f x k := sorry

theorem problem_2 (h : ∀ x, f x k ≥ -3) : k = -11 := sorry

theorem problem_3 (h : ∀ x1 x2 x3, ∃ a b c, a = f x1 k ∧ b = f x2 k ∧ c = f x3 k ∧ a + b > c ∧ a + c > b ∧ b + c > a) : -1 / 2 ≤ k ∧ k ≤ 4 := sorry

end problem_1_problem_2_problem_3_l280_280217


namespace find_clock_angle_l280_280366

def angle_at_hour (hour : Int) : Float := hour * 30.0
def minute_hand_movement (minutes : Int) : Float := minutes * 6.0
def hour_hand_movement (hour : Int) (minutes : Int) : Float :=
  (hour * 30.0) + (minutes * 0.5)

theorem find_clock_angle (a : Float) :
  let angle_2_00 := angle_at_hour 2
  let minute_angle := minute_hand_movement 15
  let hour_angle := hour_hand_movement 2 15
  let acute_angle := abs (minute_angle - hour_angle)
  acute_angle = 22.5 → 18.5 + a = 22.5 → a = 4 := sorry

end find_clock_angle_l280_280366


namespace shifted_graph_sum_l280_280510

theorem shifted_graph_sum:
  let shift_left := fun f x => f (x + 2)
  let original := fun x => 3 * x^2 + 2 * x + 4
  let shifted := shift_left original
  let a := 3
  let b := 14
  let c := 20
  in a + b + c = 37 :=
by
  sorry

end shifted_graph_sum_l280_280510


namespace coins_in_bag_l280_280002

theorem coins_in_bag (x : ℝ) (h : x + 0.5 * x + 0.25 * x = 140) : x = 80 :=
by sorry

end coins_in_bag_l280_280002


namespace units_digit_of_factorial_sum_l280_280651

theorem units_digit_of_factorial_sum : 
  (1 + 2 + 6 + 4) % 10 = 3 := sorry

end units_digit_of_factorial_sum_l280_280651


namespace probability_multiple_of_6_or_8_l280_280804

theorem probability_multiple_of_6_or_8
  (n : ℕ)
  (h_n : n = 60)
  (multiples_6_count : ℕ)
  (h_multiples_6 : multiples_6_count = 10)
  (multiples_8_count : ℕ)
  (h_multiples_8 : multiples_8_count = 7)
  (multiples_6_and_8_count : ℕ)
  (h_multiples_6_and_8 : multiples_6_and_8_count = 2) :
  (15 / 60 : ℚ) = 1 / 4 := sorry

end probability_multiple_of_6_or_8_l280_280804


namespace simplify_fraction_l280_280530

theorem simplify_fraction (h1 : 3.36 = 3 + 0.36) 
                          (h2 : 0.36 = (36 : ℚ) / 100) 
                          (h3 : (36 : ℚ) / 100 = 9 / 25) 
                          : 3.36 = 84 / 25 := 
by 
  rw [h1, h2, h3]
  norm_num
  rw [←Rat.add_div, show 3 = 75 / 25 by norm_num]
  norm_num
  
  sorry  -- This line can be safely removed when the proof is complete.

end simplify_fraction_l280_280530


namespace min_steps_to_return_l280_280048

open_locale classical

-- Define the main variables and conditions of the problem.
noncomputable def problem_statement : Prop :=
  ∃ (n : ℕ), 
  (∀ (S : Type) [metric_space S] (P : S) (step_length : ℝ), 
    let unit_sphere : set S := { Q | dist Q P = 1 }
    in S = metric.ball P 1 →
       step_length = 1.99 →
       (∀ (seq : ℕ → S), seq 0 = P → 
        (∀ m, dist (seq (m+1)) (seq m) = step_length) →
        (∀ k, dist (seq k) (seq (k+1)) ≠ 0) →
        dist (seq n) P = 0 → n ≥ 4))

-- State the theorem
theorem min_steps_to_return : problem_statement :=
sorry

end min_steps_to_return_l280_280048


namespace averageTemperature_is_99_l280_280599

-- Define the daily temperatures
def tempSunday : ℝ := 99.1
def tempMonday : ℝ := 98.2
def tempTuesday : ℝ := 98.7
def tempWednesday : ℝ := 99.3
def tempThursday : ℝ := 99.8
def tempFriday : ℝ := 99
def tempSaturday : ℝ := 98.9

-- Define the number of days
def numDays : ℝ := 7

-- Define the total temperature
def totalTemp : ℝ := tempSunday + tempMonday + tempTuesday + tempWednesday + tempThursday + tempFriday + tempSaturday

-- Define the average temperature
def averageTemp : ℝ := totalTemp / numDays

-- The theorem to prove
theorem averageTemperature_is_99 : averageTemp = 99 := by
  sorry

end averageTemperature_is_99_l280_280599


namespace frac_eval_eq_l280_280630

theorem frac_eval_eq :
  let a := 19
  let b := 8
  let c := 35
  let d := 19 * 8 / 35
  ( (⌈a / b - ⌈c / d⌉⌉) / ⌈c / b + ⌈d⌉⌉) = (1 / 10) := by
  sorry

end frac_eval_eq_l280_280630


namespace sum_first_4_terms_l280_280191

-- Define the sequence and its properties
def a (n : ℕ) : ℝ := sorry   -- The actual definition will be derived based on n, a_1, and q
def S (n : ℕ) : ℝ := sorry   -- The sum of the first n terms, also will be derived

-- Define the initial sequence properties based on the given conditions
axiom h1 : 0 < a 1  -- The sequence is positive
axiom h2 : a 4 * a 6 = 1 / 4
axiom h3 : a 7 = 1 / 8

-- The goal is to prove the sum of the first 4 terms equals 15
theorem sum_first_4_terms : S 4 = 15 := by
  sorry

end sum_first_4_terms_l280_280191


namespace monomials_like_terms_l280_280242

theorem monomials_like_terms (n m : ℕ) (h1 : n = 4) (h2 : m = 1) : m + n = 5 := by
  rw [h1, h2]
  rfl

end monomials_like_terms_l280_280242


namespace gcd_fact_8_fact_6_sq_l280_280107

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def gcd (a b : ℕ) : ℕ := Nat.gcd a b

theorem gcd_fact_8_fact_6_sq : gcd (factorial 8) ((factorial 6)^2) = 11520 := by
  sorry

end gcd_fact_8_fact_6_sq_l280_280107


namespace lengthMoreThanBreadth_l280_280396

variable (b : ℝ) 

def length := 58

noncomputable def costPerMeter := 26.50
noncomputable def totalCost := 5300

def perimeter := length + length + b + b
def cost := costPerMeter * perimeter

theorem lengthMoreThanBreadth (h1 : cost = totalCost) : (length - b) = 37 :=
by
  sorry

end lengthMoreThanBreadth_l280_280396


namespace part_a_part_b_l280_280549

-- Part (a): Proving that 91 divides n^37 - n for all integers n
theorem part_a (n : ℤ) : 91 ∣ (n ^ 37 - n) := 
sorry

-- Part (b): Finding the largest k that divides n^37 - n for all integers n is 3276
theorem part_b (n : ℤ) : ∀ k : ℤ, (k > 0) → (∀ n : ℤ, k ∣ (n ^ 37 - n)) → k ≤ 3276 :=
sorry

end part_a_part_b_l280_280549


namespace least_n_for_multiple_of_8_l280_280542

def is_positive_integer (n : ℕ) : Prop := n > 0

def is_multiple_of_8 (k : ℕ) : Prop := ∃ m : ℕ, k = 8 * m

theorem least_n_for_multiple_of_8 :
  ∀ n : ℕ, (is_positive_integer n → is_multiple_of_8 (Nat.factorial n)) → n ≥ 6 :=
by
  sorry

end least_n_for_multiple_of_8_l280_280542


namespace solution_set_l280_280977

def op (a b : ℝ) : ℝ := -2 * a + b

theorem solution_set (x : ℝ) : (op x 4 > 0) ↔ (x < 2) :=
by {
  -- proof required here
  sorry
}

end solution_set_l280_280977


namespace lcm_of_two_numbers_l280_280546

theorem lcm_of_two_numbers (a b : ℕ) (h1 : Nat.gcd a b = 9) (h2 : a * b = 1800) : Nat.lcm a b = 200 :=
by
  sorry

end lcm_of_two_numbers_l280_280546


namespace size_T_le_n_div_2_l280_280676

noncomputable def isValidMatrix (n m : ℕ) (A : matrix (fin n) (fin n) ℕ) : Prop :=
  n ≥ 2 ∧ 
  (∀ i j, A i j ∈ {0, 1}) ∧ 
  (∀ i, finset.sum finset.univ (λ k, A i k) = m) ∧ 
  (0 < m < n) ∧
  (∀ i j, i ≠ j → finset.sum finset.univ (λ k, abs (A i k - A j k)) > 0)

noncomputable def T (n m : ℕ) (A : matrix (fin n) (fin n) ℕ) : finset ℕ :=
  finset.image (λ ⟨i, j⟩, finset.sum finset.univ (λ k, A i k * A j k))
    (finset.filter (λ ⟨i, j⟩, i.val < j.val)
      (finset.product finset.univ finset.univ))

theorem size_T_le_n_div_2 {n m : ℕ} {A : matrix (fin n) (fin n) ℕ}
  (h : isValidMatrix n m A) : (T n m A).card ≤ n / 2 :=
  sorry

end size_T_le_n_div_2_l280_280676


namespace infinite_number_of_triangles_with_side_8_l280_280180

theorem infinite_number_of_triangles_with_side_8 (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b > 8) (h4 : a - b < 8) : 
  ∃’ (a b : ℕ), a > 0 ∧ b > 0 ∧ a + b > 8 ∧ a - b < 8 :=
by sorry

end infinite_number_of_triangles_with_side_8_l280_280180


namespace perpendicular_mn_ik_l280_280677

open EuclideanGeometry

noncomputable def semi_perimeter (a b c : ℝ) : ℝ :=
  (a + b + c) / 2

noncomputable def points_on_circumcircle (A B C O : Point) : Prop :=
  -- dummy definition to indicate points A, B, C are on circumcircle centered at O
  sorry

noncomputable def K_from_BO (B O : Point) (circumcircle : Circle) : Point :=
  -- dummy definition to indicate K is the point where BO extended meets circumcircle
  sorry

noncomputable def N_and_M (A C B : Point) (semi_perim : ℝ) : Point × Point :=
  -- dummy definition to indicate N and M are points on extended AB and CB respectively
  sorry

noncomputable def is_perpendicular (M N I K : Point) : Prop :=
  -- dummy definition to check if line segments MN and IK are perpendicular
  sorry

theorem perpendicular_mn_ik 
  (A B C O I : Point) 
  (cond1 : AcuteAngleTriangle A B C) -- Condition 1
  (cond2 : circumcenter O A B C) -- Condition 2
  (cond3 : incenter I A B C) -- Condition 3
  (circumcircle : Circle) 
  (cond4 : points_on_circumcircle A B C circumcircle.center) -- Condition 4
  (K : Point) 
  (cond5 : K = K_from_BO B O circumcircle) -- Condition 5
  (s : ℝ) 
  (cond6 : s = semi_perimeter (dist A B) (dist B C) (dist C A)) -- Condition 6
  (NM : Point × Point) 
  (cond7 : NM = N_and_M A C B s) -- Condition 7 
: is_perpendicular NM.fst NM.snd I K :=
  sorry

end perpendicular_mn_ik_l280_280677


namespace berry_average_temperature_l280_280596

def sunday_temp : ℝ := 99.1
def monday_temp : ℝ := 98.2
def tuesday_temp : ℝ := 98.7
def wednesday_temp : ℝ := 99.3
def thursday_temp : ℝ := 99.8
def friday_temp : ℝ := 99.0
def saturday_temp : ℝ := 98.9

def total_temp : ℝ := sunday_temp + monday_temp + tuesday_temp + wednesday_temp + thursday_temp + friday_temp + saturday_temp
def average_temp : ℝ := total_temp / 7

theorem berry_average_temperature : average_temp = 99 := by
  sorry

end berry_average_temperature_l280_280596


namespace total_paintings_l280_280327

theorem total_paintings (paintings_per_room : ℕ) (number_of_rooms : ℕ) (h₁ : paintings_per_room = 8) (h₂ : number_of_rooms = 4) : paintings_per_room * number_of_rooms = 32 :=
by {
  rw [h₁, h₂],
  norm_num,
}

end total_paintings_l280_280327


namespace three_point_three_six_as_fraction_l280_280536

theorem three_point_three_six_as_fraction : 3.36 = (84 : ℚ) / 25 := 
by
  sorry

end three_point_three_six_as_fraction_l280_280536


namespace double_root_condition_l280_280307

theorem double_root_condition (a : ℝ) : 
  (∃! x : ℝ, (x+2)^2 * (x+7)^2 + a = 0) ↔ a = -625 / 16 :=
sorry

end double_root_condition_l280_280307


namespace radicals_equality_l280_280001

noncomputable def sqrt6 (x : ℝ) := x ^ (1/6)
noncomputable def sqrt3 (x : ℝ) := x ^ (1/3)

theorem radicals_equality (x : ℝ) (h : x ≥ 0) : 
  sqrt6 (4 * x * (11 + 4 * real.sqrt 6)) * sqrt3 (4 * real.sqrt (2 * x) - 2 * real.sqrt (3 * x)) = sqrt3 (20 * x) :=
  sorry

end radicals_equality_l280_280001


namespace polynomial_roots_l280_280095

theorem polynomial_roots :
  (∀ x : ℤ, (x^3 - 4*x^2 - 11*x + 24 = 0) ↔ (x = 4 ∨ x = 3 ∨ x = -1)) :=
sorry

end polynomial_roots_l280_280095


namespace train_length_l280_280950

theorem train_length (L V : ℝ) (h1 : V = L / 15) (h2 : V = (L + 100) / 40) : L = 60 := by
  sorry

end train_length_l280_280950


namespace probability_sum_greater_than_four_l280_280462

def sum_greater_than_four_probability :=
  (5 / 6: ℚ)

theorem probability_sum_greater_than_four :
  let outcomes := { (a, b) | a in (Finset.range 1 7) ∧ b in (Finset.range 1 7) }
  let favorable_outcomes := outcomes.filter (λ pair, (pair.1 + pair.2) > 4)
  let probability := favorable_outcomes.card.to_rat / outcomes.card.to_rat
  probability = sum_greater_than_four_probability :=
by
  sorry

end probability_sum_greater_than_four_l280_280462


namespace octahedron_vertices_sum_l280_280992

noncomputable def octahedron_faces_sum (a b c d e f : ℕ) : ℕ :=
  a + b + c + d + e + f

theorem octahedron_vertices_sum (a b c d e f : ℕ) 
  (h : 8 * (octahedron_faces_sum a b c d e f) = 440) : 
  octahedron_faces_sum a b c d e f = 147 :=
by
  sorry

end octahedron_vertices_sum_l280_280992


namespace prove_projection_minimum_value_l280_280692

variable {ℝ : Type} [LinearOrderedField ℝ]

noncomputable def projection_minimum_value (a b : EuclideanSpace ℝ (Fin 2)) 
  (h1 : ∥a∥ = Real.sqrt 6) 
  (h2 : ∥a + (2 : ℝ) • b∥ = ∥(3 : ℝ) • a - (4 : ℝ) • b∥) : 
  ℝ := 12 / 7

theorem prove_projection_minimum_value (a b : EuclideanSpace ℝ (Fin 2)) 
  (h1 : ∥a∥ = Real.sqrt 6) 
  (h2 : ∥a + (2 : ℝ) • b∥ = ∥(3 : ℝ) • a - (4 : ℝ) • b∥) : 
  ∃ c : ℝ, c = projection_minimum_value a b h1 h2 :=
begin
  use (12 / 7),
  sorry
end

end prove_projection_minimum_value_l280_280692


namespace father_weight_loss_l280_280299

theorem father_weight_loss (calories_burned_per_day : ℕ)
  (pounds_per_calories : ℕ)
  (calories_intake_per_day : ℕ)
  (days : ℕ)
  (h_burned : calories_burned_per_day = 2500)
  (h_pound_calories : pounds_per_calories = 3500)
  (h_intake : calories_intake_per_day = 2000)
  (h_days : days = 35) :
  (days * (calories_burned_per_day - calories_intake_per_day)) / pounds_per_calories = 5 :=
by {
  have daily_deficit := calories_burned_per_day - calories_intake_per_day,
  rw [h_burned, h_intake, h_pound_calories, h_days],
  norm_num,
  rw div_eq_of_eq_mul,
  norm_num,
}


end father_weight_loss_l280_280299


namespace count_positive_whole_numbers_cuberoot_lt_15_l280_280224

theorem count_positive_whole_numbers_cuberoot_lt_15 :
  set.count {x : ℕ | x > 0 ∧ real.cbrt x < 15} = 3374 :=
by
  sorry

end count_positive_whole_numbers_cuberoot_lt_15_l280_280224


namespace pete_and_raymond_spent_together_l280_280794

    def value_nickel : ℕ := 5
    def value_dime : ℕ := 10
    def value_quarter : ℕ := 25

    def pete_nickels_spent : ℕ := 4
    def pete_dimes_spent : ℕ := 3
    def pete_quarters_spent : ℕ := 2

    def raymond_initial : ℕ := 250
    def raymond_nickels_left : ℕ := 5
    def raymond_dimes_left : ℕ := 7
    def raymond_quarters_left : ℕ := 4
    
    def total_spent : ℕ := 155

    theorem pete_and_raymond_spent_together :
      (pete_nickels_spent * value_nickel + pete_dimes_spent * value_dime + pete_quarters_spent * value_quarter)
      + (raymond_initial - (raymond_nickels_left * value_nickel + raymond_dimes_left * value_dime + raymond_quarters_left * value_quarter))
      = total_spent :=
      by
        sorry
    
end pete_and_raymond_spent_together_l280_280794


namespace inequality_solution_range_l280_280239

theorem inequality_solution_range (a : ℝ) :
  (∃ x : ℝ, 1 < x ∧ x < 4 ∧ 2 * x ^ 2 - 8 * x - 4 - a > 0) → a < -4 :=
by
  sorry

end inequality_solution_range_l280_280239


namespace investment_total_amount_l280_280050

def P : ℝ := 12000
def r : ℝ := 0.05
def t : ℝ := 7

noncomputable def compound_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P * ((1 + r) ^ t)

theorem investment_total_amount : 
  real.to_nat (compound_interest P r t) = 16885 :=
by
  sorry

end investment_total_amount_l280_280050


namespace pentagon_ratio_l280_280402

-- Definitions and conditions
variables {α R : ℝ}
variables {A B C D E : Type} [InscribeCircle A B C D E]
-- Angle conditions
variables (⦃CAB_eq_2nd ⦄ : ∠CAB = 2 * ∠CEA)
variables (⦃CBD_CDB_eq_alpha ⦄ : ∠CBD - ∠CDB = α)

-- Main theorem statement
theorem pentagon_ratio
  (BD_parallel_AE : BD ∥ AE)
  {P Q : Type} (hP : P = intersection BD CE)
  (hQ : Q = intersection BD AC) :
  (perimeter_triangle ACE) / R = 2 * (sin α + sin (2 * α) + sin (3 * α)) := sorry

end pentagon_ratio_l280_280402


namespace units_digit_sum_factorials_l280_280652

theorem units_digit_sum_factorials : 
  (∑ n in finset.range 2011, (n.factorial % 10)) % 10 = 3 := 
by
  sorry

end units_digit_sum_factorials_l280_280652


namespace combined_tax_rate_l280_280541

-- Definitions:
def mork_income (X : ℝ) : ℝ := X
def mindy_income (X : ℝ) : ℝ := 4 * X
def mork_tax_rate : ℝ := 0.45
def mindy_tax_rate : ℝ := 0.20

-- Proof that combined tax rate is 25%.
theorem combined_tax_rate (X : ℝ) (mork_income mindy_income : ℝ) (mork_tax_rate mindy_tax_rate : ℝ):
  (mork_tax_rate = 0.45) → 
  (mindy_tax_rate = 0.20) → 
  (mindy_income = 4 * mork_income) → 
  ((0.45 * mork_income + 0.20 * mindy_income) / (mork_income + mindy_income) * 100 = 25) :=
by {
  intros h1 h2 h3,
  sorry
}

end combined_tax_rate_l280_280541


namespace expected_lone_cars_is_one_l280_280559

def indicator_var (n k : ℕ) : ℚ :=
  if k < n then 1 / (k + 1) else 1 / n

def lone_car_expec (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ k, indicator_var (n) (k + 1))

theorem expected_lone_cars_is_one (n : ℕ) (hn : 1 ≤ n) : lone_car_expec n = 1 :=
by
  sorry

end expected_lone_cars_is_one_l280_280559


namespace part1_exists_rectangle_B_part2_no_rectangle_B_general_exists_rectangle_B_l280_280811

-- Part 1: Prove existence of rectangle B with sides 2 + sqrt(2)/2 and 2 - sqrt(2)/2
theorem part1_exists_rectangle_B : 
  ∃ (x y : ℝ), (x + y = 4) ∧ (x * y = 7 / 2) :=
by
  sorry

-- Part 2: Prove non-existence of rectangle B for given sides of the known rectangle
theorem part2_no_rectangle_B : 
  ¬ ∃ (x y : ℝ), (x + y = 5 / 2) ∧ (x * y = 2) :=
by
  sorry

-- Part 3: General proof for any given sides of the known rectangle
theorem general_exists_rectangle_B (m n : ℝ) : 
  ∃ (x y : ℝ), (x + y = 3 * (m + n)) ∧ (x * y = 3 * m * n) :=
by
  sorry

end part1_exists_rectangle_B_part2_no_rectangle_B_general_exists_rectangle_B_l280_280811


namespace residue_mod_13_l280_280987

theorem residue_mod_13 : 
  (156 % 13 = 0) ∧ (52 % 13 = 0) ∧ (182 % 13 = 0) ∧ (26 % 13 = 0) →
  (156 + 3 * 52 + 4 * 182 + 6 * 26) % 13 = 0 :=
by
  intros h
  sorry

end residue_mod_13_l280_280987


namespace calc_result_l280_280548

theorem calc_result (initial_number : ℕ) (square : ℕ → ℕ) (subtract_five : ℕ → ℕ) : 
  initial_number = 7 ∧ (square 7 = 49) ∧ (subtract_five 49 = 44) → 
  subtract_five (square initial_number) = 44 := 
by
  sorry

end calc_result_l280_280548


namespace sum_x_i_less_8n_div_15_l280_280323

theorem sum_x_i_less_8n_div_15 (n : ℕ) (x : Fin n → ℝ) (h1 : n ≥ 3)
    (h2 : ∀ i, x i ≥ -1) (h3 : (∑ i, (x i)^5) = 0) : 
    (∑ i, x i) < (8 * n / 15) :=
by
  sorry

end sum_x_i_less_8n_div_15_l280_280323


namespace simplify_336_to_fraction_l280_280513

theorem simplify_336_to_fraction :
  let gcd_36_100 := Nat.gcd 36 100
  3.36 = (84 : ℚ) / 25 := 
by
  let g := Nat.gcd 36 100
  have h1 : 3.36 = 3 + 0.36 := by norm_num
  have h2 : 0.36 = 36 / 100 := by norm_num
  have h3 : g = 4 := by norm_num [Nat.gcd, Nat.gcd_def, Nat.gcd_rec]
  have h4 : (36 : ℚ) / 100 = 9 / 25 := by norm_num; field_simp [h3];
  have h5 : (3 : ℚ) + (9 / 25) = 84 / 25 := by norm_num; field_simp;
  rw [h1, h2, h4, h5]

end simplify_336_to_fraction_l280_280513


namespace sum_of_ages_l280_280302

theorem sum_of_ages (age1 age2 age3 : ℕ) (h : age1 * age2 * age3 = 128) : age1 + age2 + age3 = 18 :=
sorry

end sum_of_ages_l280_280302


namespace probability_of_at_least_75_cents_coins_l280_280020

open_locale big_operators

noncomputable def prob_at_least_75_cents : ℚ :=
  let total_outcomes := (nat.choose 15 7 : ℕ) in
  let successful_outcomes := 597 in
  successful_outcomes / total_outcomes

theorem probability_of_at_least_75_cents_coins :
  prob_at_least_75_cents = 597 / 6435 := by
  sorry

end probability_of_at_least_75_cents_coins_l280_280020


namespace simplify_336_to_fraction_l280_280521

theorem simplify_336_to_fraction : (336 / 100) = (84 / 25) :=
by sorry

end simplify_336_to_fraction_l280_280521


namespace fraction_representation_of_3_36_l280_280526

theorem fraction_representation_of_3_36 : (336 : ℚ) / 100 = 84 / 25 := 
by sorry

end fraction_representation_of_3_36_l280_280526


namespace nth_letter_258_is_X_l280_280812

-- Define the English alphabet as a sequence
def english_alphabet : List Char := ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

-- Define a function that returns the nth letter in the infinite, repeated sequence of the English alphabet
def nth_letter (n : Nat) : Char :=
  english_alphabet[(n - 1) % 26]

-- The theorem to prove
theorem nth_letter_258_is_X : nth_letter 258 = 'X' := 
by
  -- The proof is omitted
  sorry

end nth_letter_258_is_X_l280_280812


namespace range_of_a_l280_280721

-- Definitions of the given conditions
def z (a : ℝ) : ℂ := 2 + (a + 1) * complex.i
def condition (a : ℝ) : Prop := complex.abs (z a) < 2 * real.sqrt 2

-- The proof problem statement
theorem range_of_a (a : ℝ) (h : condition a) : -3 < a ∧ a < 1 :=
sorry  -- Proof not required

end range_of_a_l280_280721


namespace expected_lone_cars_is_one_l280_280560

def indicator_var (n k : ℕ) : ℚ :=
  if k < n then 1 / (k + 1) else 1 / n

def lone_car_expec (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ k, indicator_var (n) (k + 1))

theorem expected_lone_cars_is_one (n : ℕ) (hn : 1 ≤ n) : lone_car_expec n = 1 :=
by
  sorry

end expected_lone_cars_is_one_l280_280560


namespace petya_wins_against_sasha_l280_280795

theorem petya_wins_against_sasha
    (petya_games : ℕ)
    (sasha_games : ℕ)
    (misha_games : ℕ)
    (total_games := 15) -- derived implicitly in the solution steps
    (petya_games = 12)
    (sasha_games = 7)
    (misha_games = 11) :
  (∀ (games_p_s: ℕ), games_p_s = 4) :=
by
  sorry

end petya_wins_against_sasha_l280_280795


namespace simplify_fraction_l280_280531

theorem simplify_fraction (h1 : 3.36 = 3 + 0.36) 
                          (h2 : 0.36 = (36 : ℚ) / 100) 
                          (h3 : (36 : ℚ) / 100 = 9 / 25) 
                          : 3.36 = 84 / 25 := 
by 
  rw [h1, h2, h3]
  norm_num
  rw [←Rat.add_div, show 3 = 75 / 25 by norm_num]
  norm_num
  
  sorry  -- This line can be safely removed when the proof is complete.

end simplify_fraction_l280_280531


namespace five_dice_not_all_same_number_l280_280892
open Classical

noncomputable def probability_not_all_same (n : ℕ) : ℚ :=
  1 - 1 / (6^n)

theorem five_dice_not_all_same_number :
  probability_not_all_same 5 = 1295 / 1296 :=
by
  sorry

end five_dice_not_all_same_number_l280_280892


namespace nonnegative_integer_with_divisors_is_multiple_of_6_l280_280981

-- Definitions as per conditions in (a)
def has_two_distinct_divisors_with_distance (n : ℕ) : Prop := ∃ d1 d2 : ℕ,
  d1 ≠ d2 ∧ d1 ∣ n ∧ d2 ∣ n ∧
  (d1:ℚ) - n / 3 = n / 3 - (d2:ℚ)

-- Main statement to prove as derived in (c)
theorem nonnegative_integer_with_divisors_is_multiple_of_6 (n : ℕ) :
  n > 0 ∧ has_two_distinct_divisors_with_distance n → ∃ k : ℕ, n = 6 * k :=
by
  sorry

end nonnegative_integer_with_divisors_is_multiple_of_6_l280_280981


namespace probability_not_all_same_l280_280860

theorem probability_not_all_same :
    let total_outcomes := 6 ^ 5 in
    let same_number_outcomes := 6 in
    let p_all_same := same_number_outcomes / total_outcomes in
    let p_not_all_same := 1 - p_all_same in
    p_not_all_same = 1295 / 1296 :=
by
  sorry

end probability_not_all_same_l280_280860


namespace five_dice_not_all_same_number_l280_280893
open Classical

noncomputable def probability_not_all_same (n : ℕ) : ℚ :=
  1 - 1 / (6^n)

theorem five_dice_not_all_same_number :
  probability_not_all_same 5 = 1295 / 1296 :=
by
  sorry

end five_dice_not_all_same_number_l280_280893


namespace sum_integer_solutions_l280_280808

def condition_inequality (x : ℝ) :=
  8 * ((|x + 3| - |x - 5|) / (|2 * x - 11| - |2 * x + 7|)) -
  9 * ((|x + 3| + |x - 5|) / (|2 * x - 11| + |2 * x + 7|)) ≥ -8

def condition_interval (x : ℝ) :=
  |x| < 90

theorem sum_integer_solutions :
  let S := {x : ℤ | condition_inequality x ∧ condition_interval x} in
  (∑ x in S.to_finset, x) = 8 :=
by
  sorry

end sum_integer_solutions_l280_280808


namespace planes_to_buy_l280_280591

theorem planes_to_buy (current_planes : ℕ) (n : ℕ) : (current_planes = 29) → (n = 3) ↔ (∃ k : ℕ, 8 * k - 29 = n) := by
  intros h1 h2
  split
  {
    intro h3
    use 4
    rw [←h1, h2]
    exact h3
  }
  {
    intro h3
    rcases h3 with ⟨k, hk⟩
    rw [←h1] at hk
    exact hk
  }

end planes_to_buy_l280_280591


namespace find_number_l280_280988

theorem find_number (x : ℝ) (h : 4 * (3 * x / 5 - 220) = 320) : x = 500 :=
sorry

end find_number_l280_280988


namespace point_in_first_quadrant_l280_280235

theorem point_in_first_quadrant (x y : ℝ) (h1 : x < 0) (h2 : y > 0) : (-2 * x > 0) ∧ (1 / 3 * y > 0) :=
by 
  split
  sorry

end point_in_first_quadrant_l280_280235


namespace probability_sum_greater_than_four_l280_280487

theorem probability_sum_greater_than_four : 
  let total_outcomes := 36 in
  let outcomes_sum_less_or_equal_4 := 6 in
  let prob_sum_less_or_equal_4 := outcomes_sum_less_or_equal_4 / total_outcomes in
  prob_sum_less_or_equal_4 = (1 : ℝ) / 6 → 
  (1 - prob_sum_less_or_equal_4) = (5 : ℝ) / 6 := 
by 
  intros total_outcomes outcomes_sum_less_or_equal_4 prob_sum_less_or_equal_4 h1,
  sorry

end probability_sum_greater_than_four_l280_280487


namespace abs_sq_lt_self_iff_l280_280644

theorem abs_sq_lt_self_iff {x : ℝ} : abs x * abs x < x ↔ (0 < x ∧ x < 1) ∨ (x < -1) :=
by
  sorry

end abs_sq_lt_self_iff_l280_280644


namespace trapezoid_area_eq_30_l280_280744

def point := (ℝ × ℝ)

def is_trapezoid (A B C D : point) : Prop :=
  ∃ h b1 b2, 
    A = (2, -3) ∧ B = (2, 2) ∧ C = (7, 10) ∧ D = (7, 3) ∧
    h = 7 - 2 ∧ b1 = |2 - (-3)| ∧ b2 = |10 - 3| 

theorem trapezoid_area_eq_30 : 
  ∃ (A B C D : point), is_trapezoid A B C D ∧ 
  let h := 5 in let b1 := 5 in let b2 := 7 in 
  (1/2 : ℝ) * (b1 + b2) * h = 30 :=
sorry

end trapezoid_area_eq_30_l280_280744


namespace inscribed_circle_radius_l280_280271

theorem inscribed_circle_radius (R r : ℝ) (hR : R = 18) (hr : r = 9) :
    ∃ x : ℝ, (∀ P Px OP O1P : ℝ, Px = sqrt((R - x) ^ 2 - x ^ 2) ∧ O1P = sqrt((r + x) ^ 2 - x ^ 2)
                 ∧ Px + r = O1P) ∧ x = 8 :=
begin
  sorry
end

end inscribed_circle_radius_l280_280271


namespace find_integers_l280_280997

noncomputable def f (x : ℤ) : ℤ := 2 * x^2 + x - 6

def is_prime_power (n : ℤ) : Prop :=
  ∃ p k : ℕ, p.prime ∧ k > 0 ∧ n = (p : ℤ)^k

theorem find_integers (x : ℤ) :
  is_prime_power (f x) ↔ x = -3 ∨ x = 2 ∨ x = 5 := 
sorry

end find_integers_l280_280997


namespace problem_statement_l280_280825

def modified_hash (a b : ℝ) : ℝ := a + (a^2 / b)

theorem problem_statement : modified_hash 4 3 - 10 = -2 / 3 :=
by
  sorry

end problem_statement_l280_280825


namespace sheila_will_attend_picnic_l280_280403

noncomputable def prob_sheila_attends_picnic (P_Rain P_Attend_if_Rain P_Attend_if_Sunny P_Special : ℝ) : ℝ :=
  let P_Sunny := 1 - P_Rain
  let P_Rain_and_Attend := P_Rain * P_Attend_if_Rain
  let P_Sunny_and_Attend := P_Sunny * P_Attend_if_Sunny
  let P_Attends := P_Rain_and_Attend + P_Sunny_and_Attend + P_Special - P_Rain_and_Attend * P_Special - P_Sunny_and_Attend * P_Special
  P_Attends

theorem sheila_will_attend_picnic :
  prob_sheila_attends_picnic 0.3 0.25 0.7 0.15 = 0.63025 :=
by
  sorry

end sheila_will_attend_picnic_l280_280403


namespace smallest_valid_number_is_1002_l280_280506

noncomputable def is_valid_four_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧ (n % 6 = 0) ∧
  ((n / 1000 % 2 = 1) ∨ (n / 100 % 2 = 1) ∨ (n / 10 % 2 = 1) ∨ (n % 2 = 1)) ∧
  (if n / 1000 % 2 = 1 then 
     (((n / 100 % 2 = 0) + (n / 10 % 2 = 0) + (n % 2 = 0)) = 3)
   else if n / 100 % 2 = 1 then 
     (((n / 1000 % 2 = 0) + (n / 10 % 2 = 0) + (n % 2 = 0)) = 3)
   else if n / 10 % 2 = 1 then 
     (((n / 1000 % 2 = 0) + (n / 100 % 2 = 0) + (n % 2 = 0)) = 3)
   else 
     (((n / 1000 % 2 = 0) + (n / 100 % 2 = 0) + (n / 10 % 2 = 0)) = 3))

theorem smallest_valid_number_is_1002 : ∃ (n : ℕ), is_valid_four_digit_number n ∧ n = 1002 :=
by {
  use 1002,
  sorry
}

end smallest_valid_number_is_1002_l280_280506


namespace largest_distance_between_spheres_l280_280854

theorem largest_distance_between_spheres :
  let O1 := (3, -14, 8)
  let O2 := (-9, 5, -12)
  let d := Real.sqrt ((3 + 9)^2 + (-14 - 5)^2 + (8 + 12)^2)
  let r1 := 24
  let r2 := 50
  r1 + d + r2 = Real.sqrt 905 + 74 :=
by
  intro O1 O2 d r1 r2
  sorry

end largest_distance_between_spheres_l280_280854


namespace hyperbola_eccentricity_l280_280705

theorem hyperbola_eccentricity (a : ℝ) (h : a > 0) 
  (h_ecc : real.sqrt (1 + (4*a - 2) / a^2) = real.sqrt 3) : a = 1 :=
by
  sorry

end hyperbola_eccentricity_l280_280705


namespace five_dice_not_all_same_number_l280_280896
open Classical

noncomputable def probability_not_all_same (n : ℕ) : ℚ :=
  1 - 1 / (6^n)

theorem five_dice_not_all_same_number :
  probability_not_all_same 5 = 1295 / 1296 :=
by
  sorry

end five_dice_not_all_same_number_l280_280896


namespace gcd_fact8_fact6_squared_l280_280121

-- Definition of 8! and (6!)²
def fact8 : ℕ := 8!
def fact6_squared : ℕ := (6!)^2

-- The theorem statement to be proved
theorem gcd_fact8_fact6_squared : Nat.gcd fact8 fact6_squared = 11520 := 
by
    sorry

end gcd_fact8_fact6_squared_l280_280121


namespace inscribed_circle_radius_l280_280269

theorem inscribed_circle_radius (R r x : ℝ) (hR : R = 18) (hr : r = 9) :
  x = 8 :=
sorry

end inscribed_circle_radius_l280_280269


namespace facebook_total_employees_l280_280088

theorem facebook_total_employees 
  (annual_earnings : ℕ)
  (men_ratio : ℚ)
  (women_non_mothers : ℕ)
  (bonus_ratio : ℚ)
  (individual_bonus : ℕ)
  (total_bonus : ℕ)
  (E : ℕ) :
  annual_earnings = 5000000 →
  men_ratio = 1 / 3 →
  women_non_mothers = 1200 →
  bonus_ratio = 1 / 4 →
  individual_bonus = 1250 →
  total_bonus = (bonus_ratio * annual_earnings).toNat →
  total_bonus / individual_bonus = (2 / 3 * E - women_non_mothers) →
  E = 3300 :=
by
  sorry

end facebook_total_employees_l280_280088


namespace find_m_value_l280_280026

theorem find_m_value (m : Real) (h : (3 * m + 8) * (m - 3) = 72) : m = (1 + Real.sqrt 1153) / 6 :=
by
  sorry

end find_m_value_l280_280026


namespace factorization_proof_l280_280633

theorem factorization_proof (a : ℝ) : 2 * a^2 + 4 * a + 2 = 2 * (a + 1)^2 :=
by { sorry }

end factorization_proof_l280_280633


namespace probability_area_condition_l280_280960

noncomputable def area (a b c : ℝ) : ℝ := (1/2) * a * b * c

theorem probability_area_condition :
  let AB := 10,
      AC := 10,
      height := 10,
      BC := 10 * Real.sqrt 2,
      total_area := (1/2) * (10 * Real.sqrt 2) * 10,
      area_condition := (1/4) * total_area in
  (∃ (P : Point), triangle PBC.area < area_condition) → 
  probability (conditions_satisfied_by_P P) = 1 / 16 :=
sorry

end probability_area_condition_l280_280960


namespace find_carl_age_l280_280814

variables (Alice Bob Carl : ℝ)

-- Conditions
def average_age : Prop := (Alice + Bob + Carl) / 3 = 15
def carl_twice_alice : Prop := Carl - 5 = 2 * Alice
def bob_fraction_alice : Prop := Bob + 4 = (3 / 4) * (Alice + 4)

-- Conjecture
theorem find_carl_age : average_age Alice Bob Carl ∧ carl_twice_alice Alice Carl ∧ bob_fraction_alice Alice Bob → Carl = 34.818 :=
by
  sorry

end find_carl_age_l280_280814


namespace students_after_last_stop_on_mondays_and_wednesdays_students_after_last_stop_on_tuesdays_and_thursdays_students_after_last_stop_on_fridays_l280_280294

structure BusRoute where
  first_stop : Nat
  second_stop_on : Nat
  second_stop_off : Nat
  third_stop_on : Nat
  third_stop_off : Nat
  fourth_stop_on : Nat
  fourth_stop_off : Nat

def mondays_and_wednesdays := BusRoute.mk 39 29 12 35 18 27 15
def tuesdays_and_thursdays := BusRoute.mk 39 33 10 5 0 8 4
def fridays := BusRoute.mk 39 25 10 40 20 10 5

def students_after_last_stop (route : BusRoute) : Nat :=
  let stop1 := route.first_stop
  let stop2 := stop1 + route.second_stop_on - route.second_stop_off
  let stop3 := stop2 + route.third_stop_on - route.third_stop_off
  stop3 + route.fourth_stop_on - route.fourth_stop_off

theorem students_after_last_stop_on_mondays_and_wednesdays :
  students_after_last_stop mondays_and_wednesdays = 85 := by
  sorry

theorem students_after_last_stop_on_tuesdays_and_thursdays :
  students_after_last_stop tuesdays_and_thursdays = 71 := by
  sorry

theorem students_after_last_stop_on_fridays :
  students_after_last_stop fridays = 79 := by
  sorry

end students_after_last_stop_on_mondays_and_wednesdays_students_after_last_stop_on_tuesdays_and_thursdays_students_after_last_stop_on_fridays_l280_280294


namespace product_of_constants_l280_280127

theorem product_of_constants (x t a b : ℤ) (h1 : x^2 + t * x - 12 = (x + a) * (x + b)) :
  ∃ ts : Finset ℤ, ∏ t in ts, t = 1936 :=
by
  sorry

end product_of_constants_l280_280127


namespace base3_last_two_digits_l280_280499

open Nat

theorem base3_last_two_digits (a b c : ℕ) (h1 : a = 2005) (h2 : b = 2003) (h3 : c = 2004) :
  (2005 ^ (2003 ^ 2004 + 3) % 81) = 11 :=
by
  sorry

end base3_last_two_digits_l280_280499


namespace vocabia_words_count_l280_280334

theorem vocabia_words_count : 
  let letter_count := 6 in
  let max_word_length := 4 in
  ∑ i in finset.range (max_word_length + 1), (letter_count ^ i) = 1554 := 
by
  sorry

end vocabia_words_count_l280_280334


namespace probability_correct_l280_280451

def outcome (s₁ s₂ : ℕ) : Prop := s₁ ≥ 1 ∧ s₁ ≤ 6 ∧ s₂ ≥ 1 ∧ s₂ ≤ 6

def sum_outcome_greater_than_four (s₁ s₂ : ℕ) : Prop := outcome s₁ s₂ ∧ s₁ + s₂ > 4

def total_outcomes : ℕ := 36

def favorable_outcomes : ℕ := 30 -- As derived from 36 - 6

def probability_sum_greater_than_four : ℚ := favorable_outcomes / total_outcomes

theorem probability_correct : probability_sum_greater_than_four = 5 / 6 := 
by 
  sorry

end probability_correct_l280_280451


namespace trajectory_of_M_ellipse_trajectory_l280_280683

variable {x y : ℝ}

theorem trajectory_of_M (hx : x ≠ 5) (hnx : x ≠ -5)
  (h : (y / (x + 5)) * (y / (x - 5)) = -2) : 
  (2 * x^2 + y^2 = 50) :=
by
  -- Proof is omitted.
  sorry

theorem ellipse_trajectory (hx : x ≠ 5) (hnx : x ≠ -5) 
  (h : (y / (x + 5)) * (y / (x - 5)) = -2) : 
  (x^2 / 25 + y^2 / 50 = 1) :=
by
  -- Using the previous theorem to derive.
  have h1 : (2 * x^2 + y^2 = 50) := trajectory_of_M hx hnx h
  -- Proof of transformation is omitted.
  sorry

end trajectory_of_M_ellipse_trajectory_l280_280683


namespace area_of_circular_cross_section_l280_280175

theorem area_of_circular_cross_section {A B C O : Type} (r : ℝ) (h_eq_angle: ∀ {x y z : Type}, angle x y z = angle y z x) (O_radius : r = 2) :
  (Δ O A B C ∧ (angle (O A) (O B) = angle (O B) (O C) = angle (O A) (O C))) → 
  let cross_section_area = (π * (√3 / 3 * 2)^2) in
  cross_section_area = (8 * π / 3) :=
begin
  sorry
end

end area_of_circular_cross_section_l280_280175


namespace sales_decrease_percentage_l280_280828

theorem sales_decrease_percentage 
  (P S : ℝ) 
  (P_new : ℝ := 1.30 * P) 
  (R : ℝ := P * S) 
  (R_new : ℝ := 1.04 * R) 
  (x : ℝ) 
  (S_new : ℝ := S * (1 - x/100)) 
  (h1 : 1.30 * P * S * (1 - x/100) = 1.04 * P * S) : 
  x = 20 :=
by
  sorry

end sales_decrease_percentage_l280_280828


namespace averageTemperature_is_99_l280_280600

-- Define the daily temperatures
def tempSunday : ℝ := 99.1
def tempMonday : ℝ := 98.2
def tempTuesday : ℝ := 98.7
def tempWednesday : ℝ := 99.3
def tempThursday : ℝ := 99.8
def tempFriday : ℝ := 99
def tempSaturday : ℝ := 98.9

-- Define the number of days
def numDays : ℝ := 7

-- Define the total temperature
def totalTemp : ℝ := tempSunday + tempMonday + tempTuesday + tempWednesday + tempThursday + tempFriday + tempSaturday

-- Define the average temperature
def averageTemp : ℝ := totalTemp / numDays

-- The theorem to prove
theorem averageTemperature_is_99 : averageTemp = 99 := by
  sorry

end averageTemperature_is_99_l280_280600


namespace distinct_common_points_l280_280737

theorem distinct_common_points :
  ∃! (p : ℝ × ℝ), 
  (let (x, y) := p in (x + y - 5) = 0 ∨ (2 * x - 3 * y + 5) = 0) ∧ 
  (let (x, y) := p in (x - y + 1) = 0 ∨ (3 * x + 2 * y - 12) = 0) :=
sorry

end distinct_common_points_l280_280737


namespace symmetry_axis_sine_curve_l280_280382

theorem symmetry_axis_sine_curve :
  ∀ (x : ℝ), axis_of_symmetry (λ x, Real.sin (2 * Real.pi * x - Real.pi / 3)) x ↔ x = 5 / 12 :=
sorry

end symmetry_axis_sine_curve_l280_280382


namespace vector_condition_l280_280216

noncomputable def find_vector (l_param : ℝ → ℝ × ℝ) (m_param : ℝ → ℝ × ℝ) : ℤ × ℤ :=
  let A := λ t, (3 + 5 * t, 2 + 4 * t)
  let B := λ s, (-7 + 5 * s, 3 + 4 * s)
  let BA := λ (t s : ℝ), (10 + 5 * t - 5 * s, -1 + 4 * t - 4 * s)
  let direction_m := (5, 4)
  (6, 2)

theorem vector_condition : find_vector (λ t, (3 + 5 * t, 2 + 4 * t)) (λ s, (-7 + 5 * s, 3 + 4 * s)) = (6, 2) → (6 * 2 = 12) :=
by
  intro v
  simp [v]
  sorry

end vector_condition_l280_280216


namespace sum_of_first_10_terms_l280_280413

-- Mathematical definition of the sequence term
def sequence (n : ℕ) : ℚ := 1 / ((3 * n - 2) * (3 * n + 1))

-- Proves that the sum of the first 10 terms of the sequence equals 10/31
theorem sum_of_first_10_terms : ∑ i in Finset.range 10, sequence i = 10 / 31 :=
by
  sorry

end sum_of_first_10_terms_l280_280413


namespace find_c_l280_280147

noncomputable def inequality_holds_for_all (c : ℝ) : Prop :=
  ∀ x : ℝ, (e ^ x + e ^ (-x)) / 2 ≤ e ^ (c * x ^ 2)

theorem find_c (c : ℝ) : inequality_holds_for_all c ↔ c ≥ 1 / 2 :=
sorry

end find_c_l280_280147


namespace probability_sum_greater_than_four_l280_280458

theorem probability_sum_greater_than_four : 
  let num_dice := 2
  let sides_per_die := 6
  let favorable_outcomes := { (a, b) | a > 0 ∧ a ≤ sides_per_die ∧ b > 0 ∧ b ≤ sides_per_die ∧ a + b > 4 }
  let total_outcomes := sides_per_die * sides_per_die
  let probability := (favorable_outcomes.card : ℚ) / total_outcomes
  probability = 5 / 6 :=
by
  sorry

end probability_sum_greater_than_four_l280_280458


namespace days_passed_before_cows_ran_away_l280_280934

def initial_cows := 1000
def initial_days := 50
def cows_left := 800
def cows_run_away := initial_cows - cows_left
def total_food := initial_cows * initial_days
def remaining_food (x : ℕ) := total_food - initial_cows * x
def food_needed := cows_left * initial_days

theorem days_passed_before_cows_ran_away (x : ℕ) :
  (remaining_food x = food_needed) → (x = 10) :=
by
  sorry

end days_passed_before_cows_ran_away_l280_280934


namespace five_dice_not_all_same_probability_l280_280877

noncomputable def probability_not_all_same : ℚ :=
  let total_outcomes := (6:ℚ) ^ 5
  let favorable_outcomes := (6:ℚ)
  1 - (favorable_outcomes / total_outcomes)

theorem five_dice_not_all_same_probability :
  probability_not_all_same = 1295 / 1296 :=
by
  unfold probability_not_all_same
  norm_cast
  simp
  sorry

end five_dice_not_all_same_probability_l280_280877


namespace sum_of_first_17_terms_l280_280183

theorem sum_of_first_17_terms (a : ℕ → ℤ) (h_arith : ∀ n, a (n + 1) - a n = a 1 - a 0 )
  (h_condition : a 3 + a 9 + a 15 = 15) :
  (∑ i in finset.range 17, a i) = 85 := 
  sorry

end sum_of_first_17_terms_l280_280183


namespace gcd_factorial_l280_280116

theorem gcd_factorial (a b : ℕ) : 
    ∃ (g : ℕ), nat.gcd (nat.factorial 8) ((nat.factorial 6) ^ 2) = g ∧ g = 5760 := 
by 
  let g := nat.gcd (nat.factorial 8) ((nat.factorial 6) ^ 2)
  existsi (5760 : ℕ)
  split
  · sorry
  · rfl

end gcd_factorial_l280_280116


namespace inequality_solution_set_l280_280830

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem inequality_solution_set (x : ℝ) :
    (x > 0) →
    (sqrt (log_base 2 x - 1) + 1 / 2 * log_base (1 / 2) (x ^ 3) + 2 > 0) ↔
    (2 ≤ x ∧ x < 4) :=
by
  -- This is the statement translation. The proof is omitted.
  sorry

end inequality_solution_set_l280_280830


namespace selection_prob_l280_280149

noncomputable def comb (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem selection_prob : 
  let n := 10
  let k := 3
  ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧
  (∀ (x : ℕ), x ∈ {a, b, c} → x ≤ n) →
  ¬ c = 3 →
  (∃ (x y : ℕ), (x = a ∨ x = b) ∧ x ≤ n ∧ y ≤ n) →
  comb 9 3 - comb 7 3 = 49 :=
by
  intros
  sorry

end selection_prob_l280_280149


namespace probability_greater_than_half_l280_280799

noncomputable def x : ℝ → ℝ
noncomputable def y : ℝ → ℝ

def probability_cond (x: ℝ) (y: ℝ) : set ℝ := 
  {ω | |x ω - y ω| > 1/2}

theorem probability_greater_than_half :
  ∀ x y : ℝ, P(probability_cond x y) = 3 / 8 := 
sorry

end probability_greater_than_half_l280_280799


namespace margaret_more_points_than_marco_l280_280816

def average_score : ℕ := 90
def marco_percentage_less : ℕ := 10
def margaret_score : ℕ := 86

theorem margaret_more_points_than_marco :
  let marco_score := average_score - (average_score * marco_percentage_less / 100) in
  margaret_score - marco_score = 5 :=
by
  sorry

end margaret_more_points_than_marco_l280_280816


namespace gcd_fact_8_fact_6_sq_l280_280109

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def gcd (a b : ℕ) : ℕ := Nat.gcd a b

theorem gcd_fact_8_fact_6_sq : gcd (factorial 8) ((factorial 6)^2) = 11520 := by
  sorry

end gcd_fact_8_fact_6_sq_l280_280109


namespace team_A_wins_set_team_A_wins_entire_match_possible_x_values_and_probabilities_l280_280284

noncomputable theory

open ProbabilityTheory

variables {A B : Type}

-- Definition of the probability space for the volleyball match scenario
-- Assuming all events are equally likely.
variables (sample_space : ℕ → ProbState A B) (prob : ℕ → Prob)

-- Part (1): Probability of Team A winning
def prob_team_A_wins_set : ℕ → ℝ := 1 / 2 -- The probability that Team A wins a set.
def prob_team_A_wins : ℝ := 3 / 4 -- The probability that Team A wins the entire match.

-- Part (2): Possible values of x and corresponding probabilities
def prob_team_A_scores_serving : ℝ := 2 / 5 -- Probability of Team A scoring when they are serving
def prob_team_A_scores_receiving : ℝ := 3 / 5 -- Probability of Team A scoring when they are receiving
def prob_x_less_than_eq_4 : ℝ := 172 / 625 -- The combined probability of Team A winning in 2 or 4 rallies.

theorem team_A_wins_set (prob_A_wins_step : ℝ) : 
  prob_team_A_wins_set ∘ sample_space = prob ∘ prob_team_A_wins_set :=
begin
  sorry
end

theorem team_A_wins_entire_match : 
  prob_team_A_wins (sample_space 1) = 3 / 4 :=
begin
  sorry
end

theorem possible_x_values_and_probabilities :
  possible_x_values ∘ sample_space = 
    ∃ (x : ℕ), (x ≤ 4 ∧ (P(x = 2) = 4 / 25) ∧ (P(x = 4) = 72 / 625)) :=
begin
  sorry
end

end team_A_wins_set_team_A_wins_entire_match_possible_x_values_and_probabilities_l280_280284


namespace diagonal_length_square_l280_280855

theorem diagonal_length_square (a : ℝ) (h : a = 50 * real.sqrt 2) : 
  let d := a * real.sqrt 2 in d = 100 :=
by {
  -- This is a placeholder for the actual proof.
  sorry
}

end diagonal_length_square_l280_280855


namespace five_circles_intersection_l280_280788

-- Define the basic setup and assumptions
variables (Plane : Type) [plane : Geometry Plane] 
variables (Circle : Plane → Type) 
variables (intersection : ∀ (c1 c2 : Circle) (pt : Plane), Prop) -- intersection relation
variables (common_point : ∀ (c1 c2 c3 c4 : Circle), Plane → Prop) 
  -- common_point c1 c2 c3 c4 p means p is the intersection point of c1, c2, c3, and c4

-- Assume five distinct circles C1, C2, C3, C4, and C5
variable (C1 C2 C3 C4 C5 : Circle)

-- Condition: Every subset of four circles have a common intersection point
axiom h1 : ∃ P1 : Plane, common_point C1 C2 C3 C4 P1
axiom h2 : ∃ P2 : Plane, common_point C1 C2 C3 C5 P2
axiom h3 : ∃ P3 : Plane, common_point C1 C2 C4 C5 P3
axiom h4 : ∃ P4 : Plane, common_point C1 C3 C4 C5 P4
axiom h5 : ∃ P5 : Plane, common_point C2 C3 C4 C5 P5

-- Goal: Prove that there exists a point P such that all five circles intersect at this point
theorem five_circles_intersection : ∃ P : Plane, 
  intersection C1 C2 P ∧ intersection C1 C3 P ∧ intersection C1 C4 P ∧ intersection C1 C5 P ∧ 
  intersection C2 C3 P ∧ intersection C2 C4 P ∧ intersection C2 C5 P ∧ 
  intersection C3 C4 P ∧ intersection C3 C5 P ∧ 
  intersection C4 C5 P := sorry

end five_circles_intersection_l280_280788


namespace correct_option_l280_280904

theorem correct_option (a b : ℝ) : (ab) ^ 2 = a ^ 2 * b ^ 2 :=
by sorry

end correct_option_l280_280904


namespace functional_relationship_max_daily_profit_price_reduction_1200_profit_l280_280034

noncomputable def y : ℝ → ℝ := λ x => -2 * x^2 + 60 * x + 800

theorem functional_relationship :
  ∀ x : ℝ, y x = (40 - x) * (20 + 2 * x) := 
by
  intro x
  sorry

theorem max_daily_profit :
  y 15 = 1250 :=
by
  sorry

theorem price_reduction_1200_profit :
  ∀ x : ℝ, y x = 1200 → x = 10 ∨ x = 20 :=
by
  intro x
  sorry

end functional_relationship_max_daily_profit_price_reduction_1200_profit_l280_280034


namespace other_group_of_grapes_l280_280570

theorem other_group_of_grapes (n : ℕ) (h1 : n > 105) (h2 : n % 3 = 1) (h3 : n % 5 = 1) : ∃ x, x > 5 ∧ (n-1) % x = 0 ∧ x = 7 := 
by {
  use 7,
  split,
  { exact dec_trivial, },
  split,
  { sorry, },
  { refl, }
}

end other_group_of_grapes_l280_280570


namespace original_mixture_percentage_l280_280901

def mixture_percentage_acid (a w : ℕ) : ℚ :=
  a / (a + w)

theorem original_mixture_percentage (a w : ℕ) :
  (a / (a + w+2) = 1 / 4) ∧ ((a + 2) / (a + w + 4) = 2 / 5) → 
  mixture_percentage_acid a w = 1 / 3 :=
by
  sorry

end original_mixture_percentage_l280_280901


namespace series_value_l280_280971

noncomputable def infinite_series := ∑' (n : ℕ) in (finset.range 2).compl, 
  (n^5 + 4 * n^3 + 12 * n^2 + 20 * n + 12) / (2^n * (n^5 + 5))

theorem series_value : infinite_series = 1 / 2 := sorry

end series_value_l280_280971


namespace find_p_8_l280_280324

noncomputable def p (x : ℝ) : ℝ := Sorry

theorem find_p_8 :
  (∀ x : ℝ, p x = x^7 - 7*x^6 + 21*x^5 - 35*x^4 + 35*x^3 - 21*x^2 + 7*x - 14 + 2*x) ∧
  p 1 = 2 ∧
  p 2 = 4 ∧
  p 3 = 6 ∧
  p 4 = 8 ∧
  p 5 = 10 ∧
  p 6 = 12 ∧
  p 7 = 14 →
  p 8 = 5056 :=
by
  sorry

end find_p_8_l280_280324


namespace part1_minimum_value_of_f_part2_range_of_a_l280_280773

open Real

-- Define the functions f(x) and g(x)
def f (x : ℝ) := (x + 1) * log (x + 1)
def g (a x : ℝ) := a * x^2 + x

-- Problem (1): Prove the minimum value of f(x)
theorem part1_minimum_value_of_f :
  ∃ x : ℝ, f x = -(1 / exp 1) :=
sorry

-- Problem (2): Find the range of real number a
theorem part2_range_of_a (a : ℝ) :
  (∀ x : ℝ, x ≥ 0 → f x ≤ g a x) → a ≥ 1/2 :=
sorry

end part1_minimum_value_of_f_part2_range_of_a_l280_280773


namespace expr1_val_expr2_val_l280_280055

noncomputable def expr1 : ℝ :=
  (1 / Real.sin (10 * Real.pi / 180)) - (Real.sqrt 3 / Real.cos (10 * Real.pi / 180))

theorem expr1_val : expr1 = 4 :=
  sorry

noncomputable def expr2 : ℝ :=
  (Real.sin (50 * Real.pi / 180) * (1 + Real.sqrt 3 * Real.tan (10 * Real.pi / 180)) - Real.cos (20 * Real.pi / 180)) /
  (Real.cos (80 * Real.pi / 180) * Real.sqrt (1 - Real.cos (20 * Real.pi / 180)))

theorem expr2_val : expr2 = Real.sqrt 2 :=
  sorry

end expr1_val_expr2_val_l280_280055


namespace profit_percentage_B_is_25_l280_280032

variable (selling_price_final : ℝ := 225)
variable (cost_price_A : ℝ := 120)
variable (profit_percentage_A : ℝ := 50)

def calculate_profit_A : ℝ := (profit_percentage_A / 100) * cost_price_A
def selling_price_A : ℝ := cost_price_A + calculate_profit_A
def profit_B : ℝ := selling_price_final - selling_price_A
def profit_percentage_B : ℝ := (profit_B / selling_price_A) * 100

-- The Lean 4 statement to prove
theorem profit_percentage_B_is_25 : profit_percentage_B = 25 := by
  sorry

end profit_percentage_B_is_25_l280_280032


namespace even_integers_count_9_2_24_1_l280_280221

noncomputable def count_even_integers (a b : ℚ) : ℕ :=
array.filter (λ n, n % 2 = 0) (array.range (b.toNat - a.toNat + 1)).length

theorem even_integers_count_9_2_24_1 :
  count_even_integers (9/2) (24/1) = 10 :=
by
  sorry

end even_integers_count_9_2_24_1_l280_280221


namespace jack_has_euros_l280_280751

theorem jack_has_euros :
  ∀ (yen_per_pound pounds jack_yen total_yen pounds_per_euro : ℕ),
    (yen_per_pound = 100) →
    (pounds = 42) →
    (jack_yen = 3000) →
    (total_yen = 9400) →
    (pounds_per_euro = 2) →
    let total_pounds_in_yen := pounds * yen_per_pound in
    let total_current_yen := jack_yen + total_pounds_in_yen in
    let yen_from_euros := total_yen - total_current_yen in
    let pounds_from_euros := yen_from_euros / yen_per_pound in
    let euros := pounds_from_euros / pounds_per_euro in
    euros = 11 :=
by
  intros yen_per_pound pounds jack_yen total_yen pounds_per_euro 
         yen_per_pound_eq yen_pounds_eq jack_yen_eq total_yen_eq pounds_per_euro_eq
  let total_pounds_in_yen := pounds * yen_per_pound
  let total_current_yen := jack_yen + total_pounds_in_yen
  let yen_from_euros := total_yen - total_current_yen
  let pounds_from_euros := yen_from_euros / yen_per_pound
  let euros := pounds_from_euros / pounds_per_euro
  have : total_pounds_in_yen = 4200 := by rw [yen_pounds_eq, yen_per_pound_eq]; norm_num
  have : total_current_yen = 7200 := by rw [jack_yen_eq, this]; norm_num
  have : yen_from_euros = 2200 := by rw [total_yen_eq, this]; norm_num
  have : pounds_from_euros = 22 := by rw [this, yen_per_pound_eq]; norm_num
  have : euros = 11 := by rw [this, pounds_per_euro_eq]; norm_num
  exact this

end jack_has_euros_l280_280751


namespace problem_statement_l280_280316

-- Define lines l1 and l2 with conditions on m
def line_l1 (m : ℝ) : ℝ × ℝ → Prop := λ (P : ℝ × ℝ), P.1 + m * P.2 = 0
def line_l2 (m : ℝ) : ℝ × ℝ → Prop := λ (P : ℝ × ℝ), m * P.1 - P.2 - m + 3 = 0

-- Define points A and B
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (1, 3)

-- Define distance function
def dist (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Define conditions for perpendicular lines
def perpendicular_lines (m : ℝ) : Prop :=
  (m = 0 ∧ line_l1 m (0, 3) ∧ line_l2 m (0, 3))
  ∨ (m ≠ 0 ∧ (∀ P : ℝ × ℝ, line_l1 m P ∧ line_l2 m P → (m * - (1/m)) = -1))

-- Define the maximum value condition
def maximum_PA_PB (m : ℝ) : ℝ :=
  if m = 0 then dist (0, 3) A * dist (0, 3) B else 5

-- The main theorem statement
theorem problem_statement (m : ℝ) : 
  perpendicular_lines m ∧ maximum_PA_PB m ≤ 5 :=
by
  sorry

end problem_statement_l280_280316


namespace relationship_abc_l280_280669

theorem relationship_abc (a b c : ℝ) (ha : a = 2^0.3) (hb : b = 2^0.1) (hc : c = 0.2^1.3) :
  c < b ∧ b < a :=
by {
  sorry
}

end relationship_abc_l280_280669


namespace probability_sum_greater_than_four_is_5_over_6_l280_280432

-- Define the sample space for two dice.
def sample_space : Finset (ℕ × ℕ) :=
  Finset.product (Finset.range 6) (Finset.range 6) 

-- Define the event where the sum is greater than four.
def event_sum_greater_than_four : Finset (ℕ × ℕ) :=
  sample_space.filter (λ p, p.1 + p.2 > 4)

-- Compute the probability of the event.
def probability_sum_greater_than_four : ℚ :=
  (event_sum_greater_than_four.card : ℚ) / (sample_space.card : ℚ)

-- Theorem to be proven
theorem probability_sum_greater_than_four_is_5_over_6 :
  probability_sum_greater_than_four = 5 / 6 :=
by
  -- Proof would go here
  sorry

end probability_sum_greater_than_four_is_5_over_6_l280_280432


namespace five_dice_not_all_same_probability_l280_280880

noncomputable def probability_not_all_same : ℚ :=
  let total_outcomes := (6:ℚ) ^ 5
  let favorable_outcomes := (6:ℚ)
  1 - (favorable_outcomes / total_outcomes)

theorem five_dice_not_all_same_probability :
  probability_not_all_same = 1295 / 1296 :=
by
  unfold probability_not_all_same
  norm_cast
  simp
  sorry

end five_dice_not_all_same_probability_l280_280880


namespace two_dice_sum_greater_than_four_l280_280442
open Classical

def probability_sum_greater_than_four : ℚ := by sorry

theorem two_dice_sum_greater_than_four :
  probability_sum_greater_than_four = 5 / 6 :=
sorry

end two_dice_sum_greater_than_four_l280_280442


namespace domain_of_f_l280_280238

theorem domain_of_f {f : ℝ → ℝ} :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → 3 ≤ 2 * x + 1 ∧ 2 * x + 1 ≤ 5) →
  (∀ y : ℝ, (∃ x : ℝ, (1 ≤ x ∧ x ≤ 2) ∧ y = 2 * x + 1) → 3 ≤ y ∧ y ≤ 5) :=
by
  assume h,
  intro y,
  assume hy,
  obtain ⟨x, hx, hy_eq⟩ := hy,
  cases hx with hx1 hx2,
  rw hy_eq,
  exact h x ⟨hx1, hx2⟩,
  done

end domain_of_f_l280_280238


namespace range_of_a_l280_280685

noncomputable def A : Set ℝ := {x | x ≥ abs (x^2 - 2 * x)}
noncomputable def B (a : ℝ) : Set ℝ := {x | x^2 - 2 * a * x + a ≤ 0}

theorem range_of_a (a : ℝ) : (A ∩ B a = B a) → (0 ≤ a ∧ a ≤ 1) :=
by
  sorry

end range_of_a_l280_280685


namespace solve_inequality_l280_280660

theorem solve_inequality (x : ℝ) : (2 ≤ |3 * x - 6| ∧ |3 * x - 6| ≤ 12) ↔ (x ∈ Set.Icc (-2 : ℝ) (4 / 3) ∨ x ∈ Set.Icc (8 / 3) (6 : ℝ)) :=
sorry

end solve_inequality_l280_280660


namespace graph_shift_for_cosine_l280_280821

theorem graph_shift_for_cosine {ω : ℝ} (hω : ω > 0)
  (h_intercepts : ∀ n : ℤ, ∃ k : ℤ, (n * (π / 2) = (k * π + π / 6) / ω)) :
  ∃ δ : ℝ, ∀ x : ℝ, cos (ω * x + π / 6) = sin (ω * (x + δ) + π / 6) ∧ δ = -π / 4 :=
by
  sorry

end graph_shift_for_cosine_l280_280821


namespace sum_of_squares_l280_280647

theorem sum_of_squares (a b : ℝ) (h1 : (a + b) / 2 = 8) (h2 : Real.sqrt (a * b) = 2 * Real.sqrt 5) :
  a^2 + b^2 = 216 :=
by
  sorry

end sum_of_squares_l280_280647


namespace two_dice_sum_greater_than_four_l280_280444
open Classical

def probability_sum_greater_than_four : ℚ := by sorry

theorem two_dice_sum_greater_than_four :
  probability_sum_greater_than_four = 5 / 6 :=
sorry

end two_dice_sum_greater_than_four_l280_280444


namespace AL_perpendicular_DK_l280_280288

theorem AL_perpendicular_DK
  (A B C D K L M : Point)
  (h_square: square A B C D)
  (h_KAB: lies_on K A B)
  (h_LBC: lies_on L B C)
  (h_MCD: lies_on M C D)
  (h_right_isosceles: right_isosceles_triangle_at L K L M) :
  perpendicular (line_through A L) (line_through D K) := 
begin
  sorry
end

end AL_perpendicular_DK_l280_280288


namespace find_length_of_AC_l280_280281

theorem find_length_of_AC 
  (A B C : Type) [RightTriangle A B C] (hC : angle C = 90)
  (hSinA : sin A = sqrt 5 / 3)
  (hBC : length BC = 2 * sqrt 5) :
  length AC = 4 := sorry

end find_length_of_AC_l280_280281


namespace probability_sum_greater_than_four_l280_280491

theorem probability_sum_greater_than_four : 
  let total_outcomes := 36 in
  let outcomes_sum_less_or_equal_4 := 6 in
  let prob_sum_less_or_equal_4 := outcomes_sum_less_or_equal_4 / total_outcomes in
  prob_sum_less_or_equal_4 = (1 : ℝ) / 6 → 
  (1 - prob_sum_less_or_equal_4) = (5 : ℝ) / 6 := 
by 
  intros total_outcomes outcomes_sum_less_or_equal_4 prob_sum_less_or_equal_4 h1,
  sorry

end probability_sum_greater_than_four_l280_280491


namespace ab_value_l280_280348

theorem ab_value (a b : ℝ) (h1 : 3^a = 81^(b + 2)) (h2 : 125^b = 5^(a - 3)) : a * b = 60 := by
  sorry

end ab_value_l280_280348


namespace tourists_both_languages_l280_280793

def total_tourists : ℕ := 100
def tourists_neither_language : ℕ := 10
def tourists_german : ℕ := 76
def tourists_french : ℕ := 83
def tourists_at_least_one_language : ℕ := total_tourists - tourists_neither_language

theorem tourists_both_languages :
  ∃ n : ℕ, n = tourists_german + tourists_french - tourists_at_least_one_language ∧ n = 69 :=
begin
  have h1 : tourists_at_least_one_language = 90,
  { exact (total_tourists - tourists_neither_language).symm },

  have h2 : 90 = tourists_german + tourists_french - (tourists_german + tourists_french - tourists_at_least_one_language),
  { rw [← h1, add_comm], ring },

  use 69,
  split,
  { exact (tourists_german + tourists_french - tourists_at_least_one_language).symm },
  { exact (159 - tourists_at_least_one_language).symm }
end

end tourists_both_languages_l280_280793


namespace no_other_integer_solutions_l280_280496

theorem no_other_integer_solutions :
  (∀ (x : ℤ), (x + 1) ^ 3 + (x + 2) ^ 3 + (x + 3) ^ 3 = (x + 4) ^ 3 → x = 2) := 
by sorry

end no_other_integer_solutions_l280_280496


namespace red_targets_count_l280_280734

theorem red_targets_count (total_targets red_points green_points : ℕ) 
  (h_total : total_targets = 100)
  (h_red_less_than_third_green : red_points < (100 - red_points) / 3)
  (h_score_equal : 10 * (x : ℕ) = 8.5 * red_points) : red_points = 20 := 
by 
  sorry

end red_targets_count_l280_280734


namespace sandcastle_height_differences_l280_280964

theorem sandcastle_height_differences :
  let Miki_height := 0.83
  let Sister_height := 0.5
  let Sam_height := 1.2
  (Miki_height - Sister_height = 0.33) ∧
  (Sam_height - Miki_height = 0.37) ∧
  (Sam_height - Sister_height = 0.7) :=
by
  let Miki_height := 0.83
  let Sister_height := 0.5
  let Sam_height := 1.2
  have h1 : Miki_height - Sister_height = 0.33 := by linarith
  have h2 : Sam_height - Miki_height = 0.37 := by linarith
  have h3 : Sam_height - Sister_height = 0.7 := by linarith
  exact ⟨h1, h2, h3⟩

end sandcastle_height_differences_l280_280964


namespace distance_between_trees_l280_280248

theorem distance_between_trees (yard_length : ℕ) (num_trees : ℕ) (yard_length_eq : yard_length = 325) (num_trees_eq : num_trees = 26) :
  (yard_length / (num_trees - 1)) = 13 := by
  sorry

end distance_between_trees_l280_280248


namespace range_of_g_minus_2x_l280_280703

def g (x : ℝ) : ℝ :=
if x = -4 then -2
else if x = -2 then -1
else if x = 0 then 0
else if x = 2 then 3
else if x = 4 then 4
else if x < -2 then ((-1 + 2) / (-2 + 4)) * (x + 2) + (-1)
else if x < 0 then ((0 + 1) / (-0 + 2)) * (x + 0) + (0)
else if x < 2 then ((3 + 0) / (2 + 0)) * (x - 0) + (0)
else ((4 - 3) / (4 - 2)) * (x - 2) + (3)

theorem range_of_g_minus_2x :
  ∀ x : ℝ, -4 ≤ x ∧ x ≤ 4 → ∃ y : ℝ, y ∈ set.range (λ x, g x - 2 * x) ∧ y = -4 ∨ y = 6 :=
sorry

end range_of_g_minus_2x_l280_280703


namespace probability_sum_greater_than_four_l280_280481

theorem probability_sum_greater_than_four :
  let all_outcomes := (Fin 6) × (Fin 6)
  let favorable_outcomes := {p : Fin 6 × Fin 6 | (p.1.val + 1) + (p.2.val + 1) > 4}
  (favorable_outcomes.card : ℚ) / ((Fin 6 × Fin 6).card : ℚ) = 5 / 6 :=
by
  sorry

end probability_sum_greater_than_four_l280_280481


namespace alex_silver_tokens_l280_280044

theorem alex_silver_tokens :
  let x := 31 in
  let y := 35 in
  let red_tokens (x y : ℕ) := 60 - 3 * x + y in
  let blue_tokens (x y : ℕ) := 90 + 2 * x - 4 * y in
  let silver_tokens (x y : ℕ) := x + 2 * y in
  red_tokens x y < 3 ∧ blue_tokens x y < 4 → silver_tokens x y = 101 :=
by
  sorry

end alex_silver_tokens_l280_280044


namespace no_rational_roots_of_odd_coeffs_l280_280962

theorem no_rational_roots_of_odd_coeffs (a b c : ℤ) (h_a_odd : a % 2 = 1) (h_b_odd : b % 2 = 1) (h_c_odd : c % 2 = 1)
  (h_rational_root : ∃ (p q : ℤ), q ≠ 0 ∧ (a * (p / q : ℚ)^2 + b * (p / q : ℚ) + c = 0)) : false :=
sorry

end no_rational_roots_of_odd_coeffs_l280_280962


namespace probability_not_all_same_l280_280859

theorem probability_not_all_same :
    let total_outcomes := 6 ^ 5 in
    let same_number_outcomes := 6 in
    let p_all_same := same_number_outcomes / total_outcomes in
    let p_not_all_same := 1 - p_all_same in
    p_not_all_same = 1295 / 1296 :=
by
  sorry

end probability_not_all_same_l280_280859


namespace maximize_profit_constraints_l280_280568

variable (a1 a2 b1 b2 d1 d2 c1 c2 x y z : ℝ)

theorem maximize_profit_constraints (a1 a2 b1 b2 d1 d2 c1 c2 x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  (a1 * x + a2 * y ≤ c1) ∧ (b1 * x + b2 * y ≤ c2) :=
sorry

end maximize_profit_constraints_l280_280568


namespace find_point_on_ellipse_l280_280791

/-
Mathematical proof problem statement in Lean 4:

Given:
1. The ellipse equation: \(\frac{x^2}{4} + y^2 = 1\).
2. A focus \(F = (\sqrt{3}, 0)\) of this ellipse.
3. A point \(P = (p, 0)\) where \(p > 0\).

Prove:
There exists \(p = 2\) such that for any chord \(\overline{AB}\) passing through \(F\), the angles \(\angle APF\) and \(\angle BPF\) are equal.
-/

theorem find_point_on_ellipse (p : ℝ) (hp : p > 0) :
  (∀ A B : (ℝ × ℝ), 
    let abs := (λ x y : ℝ, (x^2 + y^2).sqrt),
        ellipse := (λ x y, x^2 / 4 + y^2 = 1), 
        focus := (F : ℝ × ℝ) := (Real.sqrt 3, 0), 
        chord_through_focus := ∀ (m : ℝ), 
          let line := (λ x, m * x - m * Real.sqrt 3),
          tangent_condition := (∀ x_a x_b : ℝ, 
            (m^2 + 1/4) * x^2 - 2 * m^2 * Real.sqrt 3 * x + 3 * m^2 - 1 = 0)
    in
    (angles_equal : ∀ A B : ℝ × ℝ, tangent_condition A.1 B.1 → 
      (abs A.1 B.1 = abs p F.1) →
      angle A (p, 0) F = angle B (p, 0) F) → 
    ∃ p = 2)
:=
sorry

end find_point_on_ellipse_l280_280791


namespace setB_right_triangle_l280_280047

theorem setB_right_triangle : 
    (6^2 + 8^2 = 10^2) :=
by
  calc
    6^2 + 8^2 = 36 + 64 : by rfl
    ... = 100 : by rfl
    ... = 10^2 : by rfl

end setB_right_triangle_l280_280047


namespace find_angle_ABD_l280_280039

def angle_ABC := 135

def is_circle (ω : Circle) := True

def tangent_line_at (ω : Circle) (P : Point) := Line

def intersect (l₁ l₂ : Line) : Point := Point

def bisect (l₁ l₂ : Line) := True

def angle (A B C : Point) : ℝ := 0

theorem find_angle_ABD (ω : Circle) (A B C D : Point)
  (h1 : ∠ABC = angle_ABC) 
  (h2 : is_circle ω)
  (h3 : tangent_line_at ω A = tangent_line_at ω A)
  (h4 : tangent_line_at ω C = tangent_line_at ω C)
  (h5 : D = intersect (tangent_line_at ω A) (tangent_line_at ω C))
  (h6 : bisect AB CD) :
  ∠ABD = 90 :=
by
  sorry

end find_angle_ABD_l280_280039


namespace liam_walked_distance_l280_280781

theorem liam_walked_distance :
  ∀ (y : ℝ),
  (let bike_speed := 20
   let walk_speed := 4
   let total_time := 39 / 60
   let bike_time := (2 * y) / bike_speed
   let walk_time := y / walk_speed
   let actual_total_time := bike_time + walk_time in
	actual_total_time = total_time) →
  y = 1.9 :=
by
  intros y h
  sorry

end liam_walked_distance_l280_280781


namespace probability_not_all_same_l280_280870

/-- What is the probability that when we roll five fair 6-sided dice, they won't all show the same number? -/
theorem probability_not_all_same :
  let total_outcomes := 6^5 in
  let same_number_outcomes := 6 in
  let probability_all_same := same_number_outcomes / total_outcomes.to_real in
  1 - probability_all_same = (1295 : ℝ) / 1296 :=
by
  let total_outcomes := 6^5
  let same_number_outcomes := 6
  have probability_all_same := (same_number_outcomes : ℝ) / total_outcomes.to_real
  show 1 - probability_all_same = (1295 : ℝ) / 1296       
  sorry

end probability_not_all_same_l280_280870


namespace calculate_speed_l280_280724

theorem calculate_speed :
  ∀ (distance_ft time_sec miles_per_ft hours_per_sec approx_speed_mph : ℚ),
  distance_ft = 200 →
  time_sec = 2 →
  miles_per_ft = 1 / 5280 →
  hours_per_sec = 1 / 3600 →
  approx_speed_mph = 68.1818181818 →
  (distance_ft * miles_per_ft) / (time_sec * hours_per_sec) = approx_speed_mph :=
by
  intros distance_ft time_sec miles_per_ft hours_per_sec approx_speed_mph
  intro h_distance_eq h_time_eq h_miles_eq h_hours_eq h_speed_eq
  sorry

end calculate_speed_l280_280724


namespace kyungsoo_came_second_l280_280759

theorem kyungsoo_came_second
  (kyungsoo_jump : ℝ) (younghee_jump : ℝ) (jinju_jump : ℝ) (chanho_jump : ℝ)
  (h_kyungsoo : kyungsoo_jump = 2.3)
  (h_younghee : younghee_jump = 0.9)
  (h_jinju : jinju_jump = 1.8)
  (h_chanho : chanho_jump = 2.5) :
  kyungsoo_jump = 2.3 := 
by
  sorry

end kyungsoo_came_second_l280_280759


namespace area_of_parallelogram_is_sqrt130_l280_280827

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def A : Point3D := ⟨2, -3, 1⟩
def B : Point3D := ⟨4, -6, 4⟩
def C : Point3D := ⟨5, -2, 1⟩
def D : Point3D := ⟨7, -5, 4⟩

def vector_sub (p1 p2 : Point3D) : Point3D :=
  ⟨p1.x - p2.x, p1.y - p2.y, p1.z - p2.z⟩

def cross_product (u v : Point3D) : Point3D :=
  ⟨u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x⟩

def vector_norm (v : Point3D) : ℝ :=
  real.sqrt (v.x^2 + v.y^2 + v.z^2)

def is_parallelogram (A B C D : Point3D) : Prop :=
  vector_sub B A = vector_sub D C

theorem area_of_parallelogram_is_sqrt130 (A B C D : Point3D) (h : is_parallelogram A B C D) :
  vector_norm (cross_product (vector_sub B A) (vector_sub C A)) = real.sqrt 130 :=
by {
  sorry
}

end area_of_parallelogram_is_sqrt130_l280_280827


namespace probability_not_all_same_l280_280872

/-- What is the probability that when we roll five fair 6-sided dice, they won't all show the same number? -/
theorem probability_not_all_same :
  let total_outcomes := 6^5 in
  let same_number_outcomes := 6 in
  let probability_all_same := same_number_outcomes / total_outcomes.to_real in
  1 - probability_all_same = (1295 : ℝ) / 1296 :=
by
  let total_outcomes := 6^5
  let same_number_outcomes := 6
  have probability_all_same := (same_number_outcomes : ℝ) / total_outcomes.to_real
  show 1 - probability_all_same = (1295 : ℝ) / 1296       
  sorry

end probability_not_all_same_l280_280872


namespace find_AC_l280_280280

-- Definitions: right angle, sine, given length
variables {A B C : Type} [Real A] [Real B] [Real C]

-- Assuming \(\angle C = 90^\circ\)
def angle_C_right : Real := 90

-- Assuming \(\sin A = \frac{\sqrt{5}}{3}\)
def sin_A : Real := Real.sqrt 5 / 3

-- Assuming \(BC = 2\sqrt{5}\)
def BC_length : Real := 2 * Real.sqrt 5

-- Prove \(AC = 4\)
theorem find_AC (angle_C_right : ∠ C = 90) (sin_A : Real := Real.sqrt 5 / 3) (BC_length : Real := 2 * Real.sqrt 5) :
  ∃ (AC : Real), AC = 4 :=
by
  sorry

end find_AC_l280_280280


namespace cubic_has_one_real_root_iff_l280_280695

theorem cubic_has_one_real_root_iff (a : ℝ) :
  (∃! x : ℝ, x^3 + (1 - a) * x^2 - 2 * a * x + a^2 = 0) ↔ a < -1/4 := by
  sorry

end cubic_has_one_real_root_iff_l280_280695


namespace no_zero_in_interval_2_to_16_l280_280187

variable {α : Type*} 
variable {f : α → ℝ} 

-- Conditions
def unique_zero_in_intervals (f : ℝ → ℝ) : Prop := 
  ∃! c, c ∈ (0, 2) ∧ 
        f c = 0 ∧ 
        c ∈ (0, 4) ∧ 
        c ∈ (0, 8) ∧ 
        c ∈ (0, 16)

-- Theorem
theorem no_zero_in_interval_2_to_16 (h : unique_zero_in_intervals f) : 
  ∀ x ∈ [2, 16), f x ≠ 0 :=
sorry

end no_zero_in_interval_2_to_16_l280_280187


namespace arithmetic_sequence_ninth_term_l280_280832

theorem arithmetic_sequence_ninth_term (a d : ℤ)
  (h1 : a + 2 * d = 5)
  (h2 : a + 5 * d = 11) :
  a + 8 * d = 17 := by
  sorry

end arithmetic_sequence_ninth_term_l280_280832


namespace alloy_problem_l280_280736

theorem alloy_problem (x : ℝ) (h1 : 0.12 * x + 0.08 * 30 = 0.09333333333333334 * (x + 30)) : x = 15 :=
by
  sorry

end alloy_problem_l280_280736


namespace B_C_work_time_l280_280930

-- Define the individual work rates based on the problem conditions
def A_rate : ℝ := 1/4
def B_rate : ℝ := 1/12
def A_C_rate : ℝ := 1/2

-- Define the combined work rate of B and C
def C_rate : ℝ := A_C_rate - A_rate
def B_C_rate : ℝ := B_rate + C_rate

-- Define the time B and C need to complete the work together
def B_C_time : ℝ := 1 / B_C_rate

-- Theorem statement to prove that B and C need 3 hours to complete the work together
theorem B_C_work_time : B_C_time = 3 := 
by 
  -- Skip the proof
  sorry

end B_C_work_time_l280_280930


namespace net_salary_change_l280_280408

variable (S : ℝ)

theorem net_salary_change (h1 : S > 0) : 
  (1.3 * S - 0.3 * (1.3 * S)) - S = -0.09 * S := by
  sorry

end net_salary_change_l280_280408


namespace exist_points_with_three_nearest_neighbors_l280_280293

theorem exist_points_with_three_nearest_neighbors :
  ∃ (points : Finset (EuclideanSpace ℝ (Fin 2))),
    (∀ (p ∈ points),
      (Finset.card (Finset.filter (λ q, dist p q = min_dist points p) points) = 3)) :=
sorry

end exist_points_with_three_nearest_neighbors_l280_280293


namespace product_equals_one_l280_280979

-- Define the sequence
def a : ℕ → ℕ
| 0       := 1
| 1       := 2
| (n + 2) := a n + a (n + 1)

-- Conditions and terms in the product
def term1 := a 2 / a 1
def term2 : ℕ → ℚ := λ n, (a n) / (a (n + 1))
def term_last := a 98 / a 2

-- The proposition to be proved
theorem product_equals_one :
  (⌊term1⌋ : ℤ) * (((finset.range 97).sum (λ i, term2 (i + 1))) : ℚ) * (⌊term_last⌋ : ℤ) = 1 :=
by
  sorry

end product_equals_one_l280_280979


namespace probability_drawing_two_black_two_white_l280_280019

theorem probability_drawing_two_black_two_white :
  let total_balls := 20
  let total_black_balls := 10
  let total_white_balls := 10
  let balls_drawn := 4
  -- Total ways to draw 4 balls from 20
  let total_ways := Nat.choose 20 4
  -- Ways to choose 2 black balls from 10 black balls
  let black_ways := Nat.choose 10 2
  -- Ways to choose 2 white balls from 10 white balls
  let white_ways := Nat.choose 10 2
  -- Successful ways to draw 2 black and 2 white balls
  let successful_ways := black_ways * white_ways
  -- The expected probability
  let probability := successful_ways / total_ways
  -- Simplify probability to its reduced form
  (num, denom) = (135, 323)
in probability = num / denom :=
by
  sorry

end probability_drawing_two_black_two_white_l280_280019


namespace streetlights_needed_l280_280361

theorem streetlights_needed (squares : ℕ) (streetlights_per_square : ℕ) 
  (streetlights_needed_repair : ℕ) (streetlights_bought : ℕ) :
  squares = 15 →
  streetlights_per_square = 12 →
  streetlights_needed_repair = 35 →
  streetlights_bought = 200 →
  streetlights_per_square * squares + streetlights_needed_repair - streetlights_bought = 15 :=
by
  intros hsquares hstreetlights_per_square hstreetlights_needed_repair hstreetlights_bought
  rw [hsquares, hstreetlights_per_square, hstreetlights_needed_repair, hstreetlights_bought]
  calc
    12 * 15 + 35 - 200 = 180 + 35 - 200 : by ring
    ... = 215 - 200 : by ring
    ... = 15 : by ring

end streetlights_needed_l280_280361


namespace smallest_four_digit_number_l280_280503

def is_digit_even (d : ℕ) : Prop := d % 2 = 0

def is_digit_odd (d : ℕ) : Prop := d % 2 = 1

def sum_of_digits (n : ℕ) : ℕ :=
let digits := [n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10]
in digits.sum

def is_smallest_valid_number (n : ℕ) : Prop :=
  n >= 1000 ∧ 
  n < 10000 ∧ 
  n % 6 = 0 ∧ 
  (sum_of_digits n) % 3 = 0 ∧ 
  let d := [n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10] in
  (d.filter is_digit_even).length = 3 ∧ 
  (d.filter is_digit_odd).length = 1 ∧ 
  ∀ m, (m >= 1000 ∧ m < 10000 ∧ m % 6 = 0 ∧ 
    (sum_of_digits m) % 3 = 0 ∧ 
    let dm := [m / 1000 % 10, m / 100 % 10, m / 10 % 10, m % 10] in
    (dm.filter is_digit_even).length = 3 ∧ 
    (dm.filter is_digit_odd).length = 1) -> 
  n ≤ m

theorem smallest_four_digit_number : is_smallest_valid_number 1002 := 
  sorry

end smallest_four_digit_number_l280_280503


namespace find_y_l280_280941

variable {R : Type} [Field R] (y : R)

-- The condition: y = (1/y) * (-y) + 3
def condition (y : R) : Prop :=
  y = (1 / y) * (-y) + 3

-- The theorem to prove: under the condition, y = 2
theorem find_y (y : R) (h : condition y) : y = 2 := 
sorry

end find_y_l280_280941


namespace problem_solution_l280_280493

noncomputable def probability_2A_plus_2B_eq_C : ℝ :=
  let a_domain := Set.Icc 0 1
  let b_domain := Set.Icc 0 1
  let f : ℝ × ℝ → ℝ := λ (x : ℝ × ℝ), 2 * x.1 + 2 * x.2
  let round_to_int (x : ℝ) : ℤ := if ((x - x.floor) ≥ 0.5) then x.ceil else x.floor
  let f1 : ℝ × ℝ → ℤ := λ (x : ℝ × ℝ), round_to_int x.1 
  let f2 : ℝ × ℝ → ℤ := λ (x : ℝ × ℝ), round_to_int x.2 
  let f3 : ℝ × ℝ → ℤ := λ (x : ℝ × ℝ), round_to_int (f x)
  let pred (x : ℝ × ℝ) : Prop := 2 * (f1 x) + 2 * (f2 x) = f3 x
  let volume : Real := ∫⁻ x in Set.univ.restrict (Set.prod a_domain b_domain), indicator pred x
  let full_space_volume : Real := (Set.prod a_domain b_domain).measure
  volume / full_space_volume

theorem problem_solution :
  probability_2A_plus_2B_eq_C = 7 / 16 := by
  sorry

end problem_solution_l280_280493


namespace sum_b_n_eq_2101_l280_280286

noncomputable def a_n (n : ℕ) : ℕ :=
  if n = 0 then 0
  else n + 1

def b_n (n : ℕ) : ℕ :=
  2^(a_n n - n) + n

theorem sum_b_n_eq_2101 :
  ∑ i in Finset.range 10, b_n (i+1) = 2101 :=
by
  sorry

end sum_b_n_eq_2101_l280_280286


namespace probability_sum_greater_than_four_l280_280473

theorem probability_sum_greater_than_four : 
  let total_outcomes := 36 in
  let favorable_outcomes := total_outcomes - 6 in
  let probability := favorable_outcomes / total_outcomes in
  probability = 5 / 6 :=
by
  sorry

end probability_sum_greater_than_four_l280_280473


namespace PTAFinalAmount_l280_280363

theorem PTAFinalAmount (initial_amount : ℝ) (spent_on_supplies_fraction : ℝ) (spent_on_food_fraction : ℝ) : 
  initial_amount = 400 → 
  spent_on_supplies_fraction = 1 / 4 → 
  spent_on_food_fraction = 1 / 2 → 
  (initial_amount - (initial_amount * spent_on_supplies_fraction)) / 2 = 150 := 
by
  intros h_initial h_supplies h_food
  rw [h_initial, h_supplies, h_food]
  norm_num
  sorry

end PTAFinalAmount_l280_280363


namespace estimate_pi_l280_280844

theorem estimate_pi (m : ℝ) (n : ℝ) (a : ℝ) (b : ℝ) (h1 : m = 56) (h2 : n = 200) (h3 : a = 1/2) (h4 : b = 1/4) :
  (m / n) = (π / 4 - 1 / 2) ↔ π = 78 / 25 :=
by
  sorry

end estimate_pi_l280_280844


namespace dana_worked_saturday_l280_280976

def hourly_rate : ℝ := 13

def hours_worked_friday : ℝ := 9

def hours_worked_sunday : ℝ := 3

def total_earnings : ℝ := 286

def earned_friday : ℝ := hourly_rate * hours_worked_friday := by simp [hourly_rate, hours_worked_friday]

def earned_sunday : ℝ := hourly_rate * hours_worked_sunday := by simp [hourly_rate, hours_worked_sunday]

def earnings_friday_sunday : ℝ := earned_friday + earned_sunday := by simp [earned_friday, earned_sunday]

def to_find_hours_saturday : ℝ := 10

theorem dana_worked_saturday (h : hourly_rate * (9 + to_find_hours_saturday + 3) = total_earnings) :
  to_find_hours_saturday = 10 := by 
{
    simp [hourly_rate, hours_worked_friday, hours_worked_sunday, total_earnings] at * 
    sorry
}

end dana_worked_saturday_l280_280976


namespace orthocentric_tetrahedron_12_point_sphere_l280_280340

-- Define the mathematical objects involved
structure Tetrahedron :=
(vertices : Fin 4 → ℝ³)
-- Additional properties and definitions for orthocentric, centroids, etc., should be specified here.

-- Define centroids, intersections of altitudes and specific points
def centroids_of_faces (T : Tetrahedron) : Fin 4 → ℝ³ := sorry
def intersections_of_altitudes (T : Tetrahedron) : Fin 4 → ℝ³ := sorry
def points_dividing_segments (T : Tetrahedron) (ratio : ℝ) : Fin 4 → ℝ³ := sorry

-- Main theorem statement
theorem orthocentric_tetrahedron_12_point_sphere (T : Tetrahedron) (orthocentric_T : orthocentric T) :
  ∃ S : Sphere ℝ³,
    (∀ i, centroids_of_faces T i ∈ S) ∧
    (∀ i, intersections_of_altitudes T i ∈ S) ∧
    (∀ i, points_dividing_segments T (2 / 1) i ∈ S) :=
sorry

end orthocentric_tetrahedron_12_point_sphere_l280_280340


namespace average_of_last_three_numbers_l280_280377

theorem average_of_last_three_numbers (nums : List ℝ) (h_len : nums.length = 6) 
    (h_avg6 : nums.sum / 6 = 60) (h_avg3 : (nums.take 3).sum / 3 = 55) : 
    ((nums.drop 3).sum) / 3 = 65 := 
sorry

end average_of_last_three_numbers_l280_280377


namespace min_value_reciprocal_l280_280156

theorem min_value_reciprocal (a b : ℝ) (h_a : a > 0) (h_b : b > 0) (h_eq : 2 * a + b = 4) : 
  (∀ (x : ℝ), (∀ (a b : ℝ), a > 0 ∧ b > 0 ∧ 2 * a + b = 4 -> x ≥ 1 / (2 * a * b)) -> x ≥ 1 / 2) := 
by
  sorry

end min_value_reciprocal_l280_280156


namespace probability_not_all_same_l280_280871

/-- What is the probability that when we roll five fair 6-sided dice, they won't all show the same number? -/
theorem probability_not_all_same :
  let total_outcomes := 6^5 in
  let same_number_outcomes := 6 in
  let probability_all_same := same_number_outcomes / total_outcomes.to_real in
  1 - probability_all_same = (1295 : ℝ) / 1296 :=
by
  let total_outcomes := 6^5
  let same_number_outcomes := 6
  have probability_all_same := (same_number_outcomes : ℝ) / total_outcomes.to_real
  show 1 - probability_all_same = (1295 : ℝ) / 1296       
  sorry

end probability_not_all_same_l280_280871


namespace average_of_last_three_numbers_l280_280371

theorem average_of_last_three_numbers (a b c d e f : ℝ) 
  (h1 : (a + b + c + d + e + f) / 6 = 60) 
  (h2 : (a + b + c) / 3 = 55) : 
  (d + e + f) / 3 = 65 :=
sorry

end average_of_last_three_numbers_l280_280371


namespace two_dice_sum_greater_than_four_l280_280438
open Classical

def probability_sum_greater_than_four : ℚ := by sorry

theorem two_dice_sum_greater_than_four :
  probability_sum_greater_than_four = 5 / 6 :=
sorry

end two_dice_sum_greater_than_four_l280_280438


namespace speed_in_km_per_hr_l280_280021

noncomputable def side : ℝ := 40
noncomputable def time : ℝ := 64

-- Theorem statement
theorem speed_in_km_per_hr (side : ℝ) (time : ℝ) (h₁ : side = 40) (h₂ : time = 64) : 
  (4 * side * 3600) / (time * 1000) = 9 := by
  rw [h₁, h₂]
  sorry

end speed_in_km_per_hr_l280_280021


namespace average_actions_for_search_l280_280818

def sequential_search_average_actions (n : ℕ) (not_found : Prop) (unordered : Prop) : ℕ :=
  if n = 100 ∧ not_found ∧ unordered then
    100
  else
    0

theorem average_actions_for_search (n : ℕ) (not_found : Prop) (unordered : Prop) :
  n = 100 → not_found → unordered → sequential_search_average_actions n not_found unordered = 100 :=
by
  intros h_n h_not_found h_unordered
  rw [sequential_search_average_actions]
  rw [if_pos]
  rfl
  exact ⟨h_n, h_not_found, h_unordered⟩

sorry

end average_actions_for_search_l280_280818


namespace units_digit_sum_factorials_l280_280657

-- Definitions
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def units_digit (n : ℕ) : ℕ := n % 10

-- Theorem to prove
theorem units_digit_sum_factorials : 
  units_digit (∑ i in Finset.range 2011, factorial i) = 3 :=
by
  sorry

end units_digit_sum_factorials_l280_280657


namespace overall_percentage_change_l280_280584

-- Defining the initial membership and the percentage changes
variables (M : ℝ)
def fall_increase := 0.08
def winter_increase := 0.15
def spring_decrease := 0.19

-- Calculating the membership at each stage
def M_fall := M * (1 + fall_increase)
def M_winter := M_fall * (1 + winter_increase)
def M_spring := M_winter * (1 - spring_decrease)

-- Calculating the overall percentage change
def percentage_change := ((M_spring - M) / M) * 100

-- Theorem asserting the overall percentage change is 24.38%
theorem overall_percentage_change : percentage_change M = 24.38 := by
  unfold percentage_change M_spring M_winter M_fall
  rw [←mul_assoc, ←mul_assoc, mul_comm M_fall, mul_assoc, ←mul_assoc, mul_assoc, div_mul_eq_div_mul_one_div (M * (1 + 0.08) * (1 + 0.15)), ←mul_assoc, mul_div_cancel' _ (ne_of_gt (by norm_num : ((1 + 0.08) * (1 + 0.15)) > 0))]
  simp [fall_increase, winter_increase, spring_decrease]
  norm_num
  done


end overall_percentage_change_l280_280584


namespace berry_average_temperature_l280_280597

def sunday_temp : ℝ := 99.1
def monday_temp : ℝ := 98.2
def tuesday_temp : ℝ := 98.7
def wednesday_temp : ℝ := 99.3
def thursday_temp : ℝ := 99.8
def friday_temp : ℝ := 99.0
def saturday_temp : ℝ := 98.9

def total_temp : ℝ := sunday_temp + monday_temp + tuesday_temp + wednesday_temp + thursday_temp + friday_temp + saturday_temp
def average_temp : ℝ := total_temp / 7

theorem berry_average_temperature : average_temp = 99 := by
  sorry

end berry_average_temperature_l280_280597


namespace product_of_constants_t_l280_280129

theorem product_of_constants_t (a b : ℤ) (t : ℤ) (h1 : a * b = -12) (h2 : t = a + b) :
  ∃ (ts : Finset ℤ), (ts = {11, 4, 1, -1, -4, -11}) ∧ ts.prod (λ x, x) = -1936 :=
by sorry

end product_of_constants_t_l280_280129


namespace length_of_X_l280_280254

theorem length_of_X
  {X : ℝ}
  (h1 : 2 + 2 + X = 4 + X)
  (h2 : 3 + 4 + 1 = 8)
  (h3 : ∃ y : ℝ, y * (4 + X) = 29) : 
  X = 4 := sorry

end length_of_X_l280_280254


namespace min_value_f_range_of_a_l280_280771
open Real

noncomputable def f (x : ℝ) : ℝ := (x + 1) * log (x + 1)
noncomputable def g (a x : ℝ) : ℝ := a * x^2 + x

theorem min_value_f : infimum (range f) = -1 / exp 1 := 
sorry

theorem range_of_a (a : ℝ) : (∀ x ≥ 0, f x ≤ g a x) ↔ a ≥ 1 / 2 := 
sorry

end min_value_f_range_of_a_l280_280771


namespace cos_double_angle_l280_280161

theorem cos_double_angle (θ : ℝ) 
  (h : tan(θ + π/4) = (1/2) * tan θ - 7/2) : 
  cos (2*θ) = -4/5 := 
sorry

end cos_double_angle_l280_280161


namespace negation_correct_l280_280399

def original_statement (x : ℝ) : Prop := x > 0 → x^2 + 3 * x - 2 > 0

def negated_statement (x : ℝ) : Prop := x > 0 ∧ x^2 + 3 * x - 2 ≤ 0

theorem negation_correct : (¬ ∀ x, original_statement x) ↔ ∃ x, negated_statement x := by
  sorry

end negation_correct_l280_280399


namespace answer_still_E_after_16_rounds_l280_280567

-- Cube faces labels
inductive FaceLabel
| A | B | C | D | E | F
deriving DecidableEq

-- Define the pair of operations
def rotateRightClockwise (face : FaceLabel) : FaceLabel :=
  match face with
  | FaceLabel.A => FaceLabel.A -- Simplified for illustration; should define actual rotation effect
  | FaceLabel.B => FaceLabel.B
  | FaceLabel.C => FaceLabel.C
  | FaceLabel.D => FaceLabel.D
  | _, _ => face -- Simplified for illustration; should define actual rotation effect

def rollRight (face : FaceLabel) : FaceLabel :=
  match face with
  | FaceLabel.A => FaceLabel.A -- Simplified for illustration; should define actual roll effect
  | FaceLabel.B => FaceLabel.B
  | FaceLabel.C => FaceLabel.C
  | FaceLabel.D => FaceLabel.D
  | _, _ => face -- Simplified for illustration; should define actual roll effect

-- One complete round of operations
def oneRound (face : FaceLabel) : FaceLabel :=
  rollRight (rotateRightClockwise face)

-- Prove that after 16 rounds, the top face is still E
theorem answer_still_E_after_16_rounds 
  (initialFace : FaceLabel) 
  (h_init : initialFace = FaceLabel.E) : 
  (iterate oneRound 16 initialFace) = FaceLabel.E :=
by
  sorry

end answer_still_E_after_16_rounds_l280_280567


namespace gcd_factorial_l280_280112

theorem gcd_factorial (a b : ℕ) : 
    ∃ (g : ℕ), nat.gcd (nat.factorial 8) ((nat.factorial 6) ^ 2) = g ∧ g = 5760 := 
by 
  let g := nat.gcd (nat.factorial 8) ((nat.factorial 6) ^ 2)
  existsi (5760 : ℕ)
  split
  · sorry
  · rfl

end gcd_factorial_l280_280112


namespace negation_statement_l280_280400

variable {α : Type} (teacher generous : α → Prop)

theorem negation_statement :
  ¬ ∀ x, teacher x → generous x ↔ ∃ x, teacher x ∧ ¬ generous x := by
sorry

end negation_statement_l280_280400


namespace length_of_segment_cutoff_l280_280723

-- Define the parabola equation
def parabola (x y : ℝ) := y^2 = 4 * (x + 1)

-- Define the line passing through the focus and perpendicular to the x-axis
def line_through_focus_perp_x_axis (x y : ℝ) := x = 0

-- The actual length calculation lemma
lemma segment_length : 
  ∀ (x y : ℝ), parabola x y → line_through_focus_perp_x_axis x y → y = 2 ∨ y = -2 :=
by sorry

-- The final theorem which gives the length of the segment
theorem length_of_segment_cutoff (y1 y2 : ℝ) :
  ∀ (x : ℝ), parabola x y1 → parabola x y2 → line_through_focus_perp_x_axis x y1 → line_through_focus_perp_x_axis x y2 → (y1 = 2 ∨ y1 = -2) ∧ (y2 = 2 ∨ y2 = -2) → abs (y2 - y1) = 4 :=
by sorry

end length_of_segment_cutoff_l280_280723


namespace circle_tangent_lines_l280_280023

theorem circle_tangent_lines (h k : ℝ) (r : ℝ) (h_gt_10 : h > 10) (k_gt_10 : k > 10)
  (tangent_y_eq_10 : k - 10 = r)
  (tangent_y_eq_x : r = (|h - k| / Real.sqrt 2)) :
  (h, k) = (10 + (1 + Real.sqrt 2) * r, 10 + r) :=
by
  sorry

end circle_tangent_lines_l280_280023


namespace three_point_three_six_as_fraction_l280_280535

theorem three_point_three_six_as_fraction : 3.36 = (84 : ℚ) / 25 := 
by
  sorry

end three_point_three_six_as_fraction_l280_280535


namespace five_digit_num_properties_l280_280228

open Nat

def five_digit_num := 54321

theorem five_digit_num_properties :
  (five_digit_num % 3 ≠ 0) ∧ 
  (five_digit_num % 4 ≠ 0) ∧ 
  ((54322 % 3 = 0) ∧ (54320 % 4 = 0)) ∧
  (∀ (a b : ℕ), a ≠ b → swap a b five_digit_num < five_digit_num) := 
by
  sorry

-- Utility function for swapping two digits in a number
def swap (a b : ℕ) (n : ℕ) : ℕ :=
  -- Implementation to swap the digits a and b in number n
  sorry

end five_digit_num_properties_l280_280228


namespace find_k_l280_280066

def f (x : ℝ) : ℝ := 5 * x^2 - 3 * x + 6
def g (k x : ℝ) : ℝ := 2 * x^2 - k * x + 2

theorem find_k (k : ℝ) : 
  f 5 - g k 5 = 15 -> k = -15.8 :=
by
  intro h
  sorry

end find_k_l280_280066


namespace total_students_l280_280085

variables {n m: ℕ}

def rows_horizontal (left right : ℕ) : ℕ := left + right - 1

def rows_vertical (front back : ℕ) : ℕ := front + back - 1

theorem total_students (left right front back : ℕ) (h1 : left = 10) (h2 : right = 8) (h3 : front = 3) (h4 : back = 12)
  : rows_horizontal left right * rows_vertical front back = 238 :=
by
  have h5 : rows_horizontal 10 8 = 17 := by sorry
  have h6 : rows_vertical 3 12 = 14 := by sorry
  rw [h1, h2, h3, h4, h5, h6]
  exact Nat.mul_comm 17 14

end total_students_l280_280085


namespace smallest_palindrome_in_both_bases_l280_280616

-- Define the condition: number is greater than 8
def greater_than_8 (n : ℕ) : Prop :=
  n > 8

-- Define the function to check if a number is a palindrome in a given base
def is_palindrome_in_base (n : ℕ) (b : ℕ) : Prop :=
  let digits := Nat.digits b n in
  digits = digits.reverse

-- Define the condition: number is a palindrome in base 3
def palindrome_base_3 (n : ℕ) : Prop :=
  is_palindrome_in_base n 3

-- Define the condition: number is a palindrome in base 5
def palindrome_base_5 (n : ℕ) : Prop :=
  is_palindrome_in_base n 5

-- Main theorem: the smallest number greater than 8 which is a palindrome in both base 3 and base 5 is 26
theorem smallest_palindrome_in_both_bases : ∃ n, greater_than_8 n ∧ palindrome_base_3 n ∧ palindrome_base_5 n ∧ n = 26 :=
by
  sorry

end smallest_palindrome_in_both_bases_l280_280616


namespace number_of_zeros_of_f_l280_280401

def f (x : ℝ) : ℝ :=
if x ≤ 0 then x^2 + 2*x - 3 else -2 + log x

theorem number_of_zeros_of_f :
  {x : ℝ | f x = 0}.finite.to_finset.card = 2 := sorry

end number_of_zeros_of_f_l280_280401


namespace probability_sum_greater_than_four_l280_280488

theorem probability_sum_greater_than_four : 
  let total_outcomes := 36 in
  let outcomes_sum_less_or_equal_4 := 6 in
  let prob_sum_less_or_equal_4 := outcomes_sum_less_or_equal_4 / total_outcomes in
  prob_sum_less_or_equal_4 = (1 : ℝ) / 6 → 
  (1 - prob_sum_less_or_equal_4) = (5 : ℝ) / 6 := 
by 
  intros total_outcomes outcomes_sum_less_or_equal_4 prob_sum_less_or_equal_4 h1,
  sorry

end probability_sum_greater_than_four_l280_280488


namespace correct_solution_l280_280494

variable (x y : ℤ) (a b : ℤ) (h1 : 2 * x + a * y = 6) (h2 : b * x - 7 * y = 16)

theorem correct_solution : 
  (∃ x y : ℤ, 2 * x - 3 * y = 6 ∧ 5 * x - 7 * y = 16 ∧ x = 6 ∧ y = 2) :=
by
  use 6, 2
  constructor
  · exact sorry -- 2 * 6 - 3 * 2 = 6
  constructor
  · exact sorry -- 5 * 6 - 7 * 2 = 16
  constructor
  · exact rfl
  · exact rfl

end correct_solution_l280_280494


namespace positive_whole_numbers_with_cube_root_less_than_15_l280_280226

theorem positive_whole_numbers_with_cube_root_less_than_15 :
  { n : ℕ // 1 ≤ n ∧ n < 3375 }.card = 3374 :=
by
  -- Introduction of the natural number and the required conditions
  sorry

end positive_whole_numbers_with_cube_root_less_than_15_l280_280226


namespace probability_sum_greater_than_four_is_5_over_6_l280_280430

-- Define the sample space for two dice.
def sample_space : Finset (ℕ × ℕ) :=
  Finset.product (Finset.range 6) (Finset.range 6) 

-- Define the event where the sum is greater than four.
def event_sum_greater_than_four : Finset (ℕ × ℕ) :=
  sample_space.filter (λ p, p.1 + p.2 > 4)

-- Compute the probability of the event.
def probability_sum_greater_than_four : ℚ :=
  (event_sum_greater_than_four.card : ℚ) / (sample_space.card : ℚ)

-- Theorem to be proven
theorem probability_sum_greater_than_four_is_5_over_6 :
  probability_sum_greater_than_four = 5 / 6 :=
by
  -- Proof would go here
  sorry

end probability_sum_greater_than_four_is_5_over_6_l280_280430


namespace find_AC_l280_280279

-- Definitions: right angle, sine, given length
variables {A B C : Type} [Real A] [Real B] [Real C]

-- Assuming \(\angle C = 90^\circ\)
def angle_C_right : Real := 90

-- Assuming \(\sin A = \frac{\sqrt{5}}{3}\)
def sin_A : Real := Real.sqrt 5 / 3

-- Assuming \(BC = 2\sqrt{5}\)
def BC_length : Real := 2 * Real.sqrt 5

-- Prove \(AC = 4\)
theorem find_AC (angle_C_right : ∠ C = 90) (sin_A : Real := Real.sqrt 5 / 3) (BC_length : Real := 2 * Real.sqrt 5) :
  ∃ (AC : Real), AC = 4 :=
by
  sorry

end find_AC_l280_280279


namespace value_of_S_l280_280234

theorem value_of_S (x R S : ℝ) (h1 : x + 1/x = R) (h2 : R = 6) : x^3 + 1/x^3 = 198 :=
by
  sorry

end value_of_S_l280_280234


namespace point_outside_circle_l280_280406

theorem point_outside_circle :
  let A := (1 : ℝ, 2 : ℝ)
  let center := (-1 : ℝ, 2 : ℝ)
  let radius := 1
  dist A center > radius := by
  sorry

end point_outside_circle_l280_280406


namespace five_dice_not_all_same_number_l280_280894
open Classical

noncomputable def probability_not_all_same (n : ℕ) : ℚ :=
  1 - 1 / (6^n)

theorem five_dice_not_all_same_number :
  probability_not_all_same 5 = 1295 / 1296 :=
by
  sorry

end five_dice_not_all_same_number_l280_280894


namespace female_democrats_l280_280920

theorem female_democrats (F M : ℕ) (h1 : F + M = 840) (h2 : F / 2 + M / 4 = 280) : F / 2 = 140 :=
by 
  sorry

end female_democrats_l280_280920


namespace great_wall_scientific_notation_l280_280414

theorem great_wall_scientific_notation :
  6700000 = 6.7 * 10^6 :=
sorry

end great_wall_scientific_notation_l280_280414


namespace river_width_is_500_l280_280927

open Real

noncomputable theory

-- Define the boatman's velocity
def boatman_velocity : ℝ := 10

-- Define the time taken to cross the river
def crossing_time : ℝ := 50

-- Define the drift (unused in the final calculation)
def drift : ℝ := 300

-- Define the width of the river
def river_width : ℝ := boatman_velocity * crossing_time

-- Statement of the problem
theorem river_width_is_500 : river_width = 500 := by
  sorry

end river_width_is_500_l280_280927


namespace sin_120_eq_sqrt3_div_2_l280_280060

theorem sin_120_eq_sqrt3_div_2
  (h1 : 120 = 180 - 60)
  (h2 : ∀ θ, Real.sin (180 - θ) = Real.sin θ)
  (h3 : Real.sin 60 = Real.sqrt 3 / 2) :
  Real.sin 120 = Real.sqrt 3 / 2 :=
sorry

end sin_120_eq_sqrt3_div_2_l280_280060


namespace quadratic_root_value_d_l280_280245

theorem quadratic_root_value_d (d : ℚ) :
  (∀ x : ℝ, 2 * x^2 + 14 * x + d = 0 ↔ x = -7 + sqrt 15 ∨ x = -7 - sqrt 15) →
  d = 181 / 8 :=
by
  intro h
  sorry

end quadratic_root_value_d_l280_280245


namespace sum_50th_row_l280_280618

-- Define the function representing the special sum f(n)
def f : ℕ → ℕ
| 0 := 0
| n := 2 * f (n - 1) + 2 * n

-- Main statement that we want to prove : sum of the 50th row is 2^50 - 100
theorem sum_50th_row : f 50 = 2^50 - 100 := 
sorry

end sum_50th_row_l280_280618


namespace rational_square_l280_280823

variable (r : ℕ → ℚ)
variable (H : ∀ (k : ℕ), (∏ i in Finset.range k, r i) = ∑ i in Finset.range k, r i)

theorem rational_square (n : ℕ) (hn : n ≥ 3) : 
  ∃ q : ℚ, (1 / r n - 3 / 4) = q^2 := 
by
  sorry

end rational_square_l280_280823


namespace johns_sixth_quiz_score_l280_280757

theorem johns_sixth_quiz_score
  (score1 score2 score3 score4 score5 : ℕ)
  (h1 : score1 = 85)
  (h2 : score2 = 90)
  (h3 : score3 = 88)
  (h4 : score4 = 92)
  (h5 : score5 = 95)
  : (∃ score6 : ℕ, (score1 + score2 + score3 + score4 + score5 + score6) / 6 = 90) :=
by
  use 90
  sorry

end johns_sixth_quiz_score_l280_280757


namespace ellipse_eqn_exists_k_l280_280680

noncomputable def eccentricity : ℝ := real.sqrt 3 / 2
noncomputable def major_axis : ℝ := 4
noncomputable def ellipse_equation := λ (x y : ℝ) (a b : ℝ), (x^2 / b^2) + (y^2 / a^2) = 1
noncomputable def line_eq := λ (k x : ℝ), k * x + real.sqrt 3

theorem ellipse_eqn (a b : ℝ) (h1 : a > b) (h2 : a > 0) (h3 : b > 0) (eccentricity := real.sqrt 3 / 2) (major_axis := 4) :
  a = 2 ∧ b = 1 → ellipse_equation x y a b ↔ (y^2 / 4 + x^2 = 1) :=
sorry

theorem exists_k (x1 x2 y1 y2 k : ℝ) (h_line : ∀ l, line_eq k = λ x, l) :
  ellipse_equation x y 2 1 →
  (∃ k, k = real.sqrt 11 / 2 ∨ k = -real.sqrt 11 / 2 ∧
         ∀ A B, circle_with_diameter A B (x1, y1) (x2, y2) passes_through_origin (line_eq k)):
sorry

end ellipse_eqn_exists_k_l280_280680


namespace smallest_s_plus_d_l280_280229

theorem smallest_s_plus_d (s d : ℕ) (h_pos_s : s > 0) (h_pos_d : d > 0)
  (h_eq : 1 / s + 1 / (2 * s) + 1 / (3 * s) = 1 / (d^2 - 2 * d)) :
  s + d = 50 :=
sorry

end smallest_s_plus_d_l280_280229


namespace not_a_factorization_l280_280902

open Nat

theorem not_a_factorization : ¬ (∃ (f g : ℝ → ℝ), (∀ (x : ℝ), x^2 + 6*x - 9 = f x * g x)) :=
by
  sorry

end not_a_factorization_l280_280902


namespace carrie_saves_90_l280_280056

-- Define the original prices
def delta_price := 850
def united_price := 1100
def american_price := 950
def southwest_price := 900
def jetblue_price := 1200

-- Define the discount rates
def delta_discount := 0.20
def united_discount := 0.30
def american_discount := 0.25
def southwest_discount := 0.15
def jetblue_discount := 0.40

-- Calculate the discounted prices
def delta_final_price := delta_price - (delta_price * delta_discount)
def united_final_price := united_price - (united_price * united_discount)
def american_final_price := american_price - (american_price * american_discount)
def southwest_final_price := southwest_price - (southwest_price * southwest_discount)
def jetblue_final_price := jetblue_price - (jetblue_price * jetblue_discount)

-- Identify the minimum final price
def min_price := min delta_final_price 
                    (min united_final_price 
                         (min american_final_price 
                              (min southwest_final_price jetblue_final_price)))

-- Identify the maximum final price for comparison
def max_price := max delta_final_price 
                    (max united_final_price 
                         (max american_final_price 
                              (max southwest_final_price jetblue_final_price)))

-- Calculate savings
def savings := max_price - min_price

-- Prove that the savings amount to $90
theorem carrie_saves_90 : savings = 90 :=
by sorry

end carrie_saves_90_l280_280056


namespace arithmetic_seq_general_term_l280_280190

variable {a : ℕ → ℤ} (d : ℤ)
variables (a1 a2 a3 a4 : ℤ)

-- Definitions to represent the conditions
def isArithmetic (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def isGeometric (b1 b2 b3 : ℤ) : Prop :=
  (b2 * b2) = (b1 * b3)

axiom a2_eq_a1_add_d : a 2 = a 1 + d
axiom a3_eq_a1_add_2d : a 3 = a 1 + 2 * d
axiom a4_eq_a1_add_3d : a 4 = a 1 + 3 * d

-- Arithmetic sequence with non-zero common difference
axiom non_zero_d : d ≠ 0

-- Given conditions
axiom geom_seq_cond : isGeometric (a 1 + 1) (a 2 + 1) (a 4 + 1)
axiom sum_cond : a 2 + a 3 = -12

-- Goal: Prove the general term formula for the arithmetic sequence
theorem arithmetic_seq_general_term :
  (a : ℕ → ℤ) → (isArithmetic a d) → (geom_seq_cond) → (sum_cond) → (∀ n : ℕ, a n = -2 * n - 1) :=
by
  intros,
  sorry

end arithmetic_seq_general_term_l280_280190


namespace identify_minor_premise_l280_280800

theorem identify_minor_premise :
  (∀ (R P : Type) (square : R -> P -> Prop), 
    (∀ r, r → P) →
    (∀ s, square s → R) →
    ∃ minor_premise, minor_premise = ∀ r, square r → R) :=
by
  sorry

end identify_minor_premise_l280_280800


namespace sine_curve_transformation_l280_280193

theorem sine_curve_transformation (x y x' y' : ℝ) 
  (h1 : x' = (1 / 2) * x) 
  (h2 : y' = 3 * y) :
  (y = Real.sin x) ↔ (y' = 3 * Real.sin (2 * x')) := by 
  sorry

end sine_curve_transformation_l280_280193


namespace field_trip_least_fuel_l280_280328

def CarCapacity := 4
def MinivanCapacity := 6
def BusCapacity := 20

def CarsAvailable := 3
def MinivansAvailable := 2
def BusesAvailable := 1

def TotalPeople := 33

def CarFuelEfficiency := 30 -- miles per gallon
def MinivanFuelEfficiency := 20 -- miles per gallon
def BusFuelEfficiency := 10 -- miles per gallon

def RoundTripMiles := 50

def leastFuelUsage : Prop :=
  let carsFuel := (RoundTripMiles / CarFuelEfficiency : Float) * CarsAvailable
  let minivansFuel := (RoundTripMiles / MinivanFuelEfficiency : Float) * MinivansAvailable
  let busFuel := (RoundTripMiles / BusFuelEfficiency : Float) * BusesAvailable
  ∃ busUsage minivanUsage carUsage,
    (busUsage * BusCapacity +
     minivanUsage * MinivanCapacity +
     carUsage * CarCapacity) ≥ TotalPeople ∧
    (busUsage * busFuel +
     minivanUsage * minivansFuel +
     carUsage * carsFuel) = 9.17

theorem field_trip_least_fuel : leastFuelUsage :=
by {
  -- The proof follows.
  sorry
}

end field_trip_least_fuel_l280_280328


namespace parabola_arc_length_correct_l280_280615

noncomputable def parabola_arc_length : ℝ :=
  let f : ℝ → ℝ := λ y, (y ^ 2) / 4
  let f' : ℝ → ℝ := λ y, y / 2
  (1 / 2) * ∫ y in 0..2, sqrt (4 + y ^ 2)

theorem parabola_arc_length_correct :
  parabola_arc_length = sqrt(2) + log(1 + sqrt(2)) :=
by
  sorry

end parabola_arc_length_correct_l280_280615


namespace compute_x_l280_280970

theorem compute_x :
  (∑' n : ℕ, (1 / (3^n)) * (1 / (3^n) * (-1)^n)) = (∑' n : ℕ, 1 / (9^n)) →
  (∑' n : ℕ, (1 / (3^n)) * (1 / (3^n) * (-1)^n)) = 1 / (1 - (1 / 9)) →
  9 = 9 :=
by
  sorry

end compute_x_l280_280970


namespace line_intersects_curve_l280_280625

theorem line_intersects_curve (k : ℝ) : 
  (∃ x y : ℝ, y = k * (x - 2) + 4 ∧ y = sqrt (4 - x^2)) 
  ↔ k ∈ set.Ici (3 / 4) :=
by sorry

end line_intersects_curve_l280_280625


namespace axis_of_symmetry_l280_280385

theorem axis_of_symmetry (x : ℝ) : IsAxisOfSymmetry (λ x, Real.sin(2 * Real.pi * x - Real.pi / 3)) x ↔ x = 5 / 12 :=
by
  sorry

end axis_of_symmetry_l280_280385


namespace probability_not_all_dice_same_l280_280884

theorem probability_not_all_dice_same :
  let total_outcomes := 6^5
  let same_number_outcomes := 6
  let probability_same_number := same_number_outcomes / total_outcomes
  let probability_not_same_number := 1 - probability_same_number
  probability_not_same_number = (1295 : ℚ) / 1296 :=
by
  sorry

end probability_not_all_dice_same_l280_280884


namespace area_of_triangle_PQS_l280_280291

-- Definitions based on conditions
def PQR (P Q R S : Type) (PQ QR PR PS : ℝ) (PQS_area : ℝ) : Prop :=
  (PQ = 120) ∧ (PS bisects_angle PQR) ∧ (QR = y) ∧ (PR = 3 * QR - 10) ∧ 
  (PQS_area = (1 / 2) * PQ * (QR - PS_length_segment))

-- Hypothesis combining all conditions
theorem area_of_triangle_PQS
  (P Q R S : Type) (PQ QR PR PS : ℝ) (y : ℝ) (PQS_area : ℝ) :
  PQ = 120 → 
  PS bisects_angle PQR → 
  QR = y → 
  PR = 3 * y - 10 → 
  PQS_area ≈ 1578 := 
sorry

end area_of_triangle_PQS_l280_280291


namespace probability_not_all_same_l280_280865

-- Definitions of conditions from the problem
def six_sided_die_faces : ℕ := 6
def number_of_dice : ℕ := 5

-- Lean statement to prove the probability calculation
theorem probability_not_all_same : 
  let total_outcomes := six_sided_die_faces ^ number_of_dice in
  let all_same_outcomes := six_sided_die_faces in
  (1 - ((all_same_outcomes : ℚ) / total_outcomes)) = (1295 / 1296) := 
by
  sorry

end probability_not_all_same_l280_280865


namespace shorter_side_length_l280_280578

theorem shorter_side_length (a b : ℕ) (h1 : 2 * a + 2 * b = 50) (h2 : a * b = 126) : b = 9 :=
sorry

end shorter_side_length_l280_280578


namespace james_weekly_expenses_l280_280296

theorem james_weekly_expenses :
  let rent := 1200 in
  let utilities := rent * 0.20 in
  let hours_per_week_per_employee := 16 * 5 in
  let total_hours := 2 * hours_per_week_per_employee in
  let employee_wages := total_hours * 12.50 in
  let total_expenses := rent + utilities + employee_wages in
  total_expenses = 3440 :=
by
  let rent := 1200
  let utilities := rent * 0.20
  let hours_per_week_per_employee := 16 * 5
  let total_hours := 2 * hours_per_week_per_employee
  let employee_wages := total_hours * 12.50
  let total_expenses := rent + utilities + employee_wages
  sorry

end james_weekly_expenses_l280_280296


namespace solve_equation_l280_280831

theorem solve_equation (x : ℝ) (h : 3 * x ≠ 0) (h2 : x + 2 ≠ 0) : (2 / (3 * x) = 1 / (x + 2)) ↔ x = 4 := by
  sorry

end solve_equation_l280_280831


namespace distance_between_parallel_lines_l280_280381

theorem distance_between_parallel_lines :
  ∀ (A B : ℝ) (C1 C2 : ℝ), 
  (∀ (x y : ℝ), 6 * x + 8 * y + C1 = 0) → 
  (∀ (x y : ℝ), 6 * x + 8 * y + C2 = 0) → 
  C1 = -1 → 
  C2 = -9 → 
  ∃ (d : ℝ), d = |C1 - C2| / Real.sqrt (6^2 + 8^2) ∧ d = 4/5 := 
by 
  intros A B C1 C2 h1 h2 hC1 hC2
  use |C1 - C2| / Real.sqrt (A^2 + B^2)
  split
  {
    sorry
  }
  {
    sorry
  }

end distance_between_parallel_lines_l280_280381


namespace smallest_valid_number_is_1002_l280_280505

noncomputable def is_valid_four_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧ (n % 6 = 0) ∧
  ((n / 1000 % 2 = 1) ∨ (n / 100 % 2 = 1) ∨ (n / 10 % 2 = 1) ∨ (n % 2 = 1)) ∧
  (if n / 1000 % 2 = 1 then 
     (((n / 100 % 2 = 0) + (n / 10 % 2 = 0) + (n % 2 = 0)) = 3)
   else if n / 100 % 2 = 1 then 
     (((n / 1000 % 2 = 0) + (n / 10 % 2 = 0) + (n % 2 = 0)) = 3)
   else if n / 10 % 2 = 1 then 
     (((n / 1000 % 2 = 0) + (n / 100 % 2 = 0) + (n % 2 = 0)) = 3)
   else 
     (((n / 1000 % 2 = 0) + (n / 100 % 2 = 0) + (n / 10 % 2 = 0)) = 3))

theorem smallest_valid_number_is_1002 : ∃ (n : ℕ), is_valid_four_digit_number n ∧ n = 1002 :=
by {
  use 1002,
  sorry
}

end smallest_valid_number_is_1002_l280_280505


namespace range_of_a_l280_280181

variable (f : ℝ → ℝ)
variable (a : ℝ)

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y

def holds_on_interval (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, (1/2) ≤ x ∧ x ≤ 1 → f (a*x + 1) ≤ f (x - 2)

theorem range_of_a (h1 : is_even f)
                   (h2 : is_increasing_on_nonneg f)
                   (h3 : holds_on_interval f a) :
  -2 ≤ a ∧ a ≤ 0 := 
sorry

end range_of_a_l280_280181


namespace inscribed_circle_radius_l280_280256

-- Define the given conditions
def radius_large : ℝ := 18
def radius_small : ℝ := 9
def radius_inscribed : ℝ := 8

-- Define tangency conditions and relationships based on the problem statement
def large_semicircle (R : ℝ) := { x : ℝ // 0 <= x ∧ x <= R }
def small_semicircle (r : ℝ) := { x : ℝ // 0 <= x ∧ x <= r }

-- Prove the radius of the circle inscribed between the two semicircles
theorem inscribed_circle_radius :
  large_semicircle radius_large ∧ small_semicircle radius_small →
  ∃ (x : ℝ), x = radius_inscribed := 
by
  intro h;  -- Assume the hypothesis h
  exists radius_inscribed;  -- Show the existence of the radius of the inscribed circle
  have hp1 : sqrt (324 - 36 * radius_inscribed) + radius_small = sqrt (81 + 18 * radius_inscribed) := sorry,
  have hp2 : sqrt (324 - 36 * radius_inscribed) = sqrt (81 + 18 * radius_inscribed) - 9 := sorry,
  have h_sqr : (324 - 36 * radius_inscribed) = (sqrt (81 + 18 * radius_inscribed) - 9)^2 := sorry,
  sorry  -- Proof skipped for simplicity of problem setup

end inscribed_circle_radius_l280_280256


namespace units_digit_of_7_power_19_l280_280898

theorem units_digit_of_7_power_19 : (7^19) % 10 = 3 := by
  sorry

end units_digit_of_7_power_19_l280_280898


namespace ratio_of_female_officers_on_duty_l280_280792

theorem ratio_of_female_officers_on_duty (F T : Nat) (h1 : F = 0.15 * 1000) (h2 : T = 300) : F / T = 1 / 2 :=
by
  sorry

end ratio_of_female_officers_on_duty_l280_280792


namespace solution_set_of_inequality_l280_280155

theorem solution_set_of_inequality (a : ℝ) (x : ℝ) (h : 4^a = 2^(a + 2)) :
  a = 2 → (a^(2 * x + 1) > a^(x - 1) ↔ x > -2) := by
  sorry

end solution_set_of_inequality_l280_280155


namespace range_of_minequality_l280_280202

noncomputable def f (x : ℝ) := x^3 + x

theorem range_of_minequality (m : ℝ) : (∀ θ, 0 ≤ θ ∧ θ ≤ π / 2 → f (m * sin θ) + f (1 - m) > 0) → m < 1 :=
by {
  assume H,
  sorry
}

end range_of_minequality_l280_280202


namespace horner_value_at_2_l280_280554

noncomputable def f (x : ℝ) := 2 * x^5 - 3 * x^3 + 2 * x^2 + x - 3

theorem horner_value_at_2 : f 2 = 12 := sorry

end horner_value_at_2_l280_280554


namespace daily_savings_in_dollars_l280_280785

-- Define the total savings and the number of days
def total_savings_in_dimes : ℕ := 3
def number_of_days : ℕ := 30

-- Define the conversion factor from dimes to dollars
def dime_to_dollar : ℝ := 0.10

-- Prove that the daily savings in dollars is $0.01
theorem daily_savings_in_dollars : total_savings_in_dimes / number_of_days * dime_to_dollar = 0.01 :=
by sorry

end daily_savings_in_dollars_l280_280785


namespace five_dice_not_all_same_number_l280_280895
open Classical

noncomputable def probability_not_all_same (n : ℕ) : ℚ :=
  1 - 1 / (6^n)

theorem five_dice_not_all_same_number :
  probability_not_all_same 5 = 1295 / 1296 :=
by
  sorry

end five_dice_not_all_same_number_l280_280895


namespace sum_reciprocals_eq_l280_280157

theorem sum_reciprocals_eq : ∀ (n : ℕ), (∑ k in Finset.range n.succ, 1 / (k * (k + 1))) = n / (n + 1 : ℝ) :=
by sorry

end sum_reciprocals_eq_l280_280157


namespace inscribed_circle_radius_l280_280255

-- Define the given conditions
def radius_large : ℝ := 18
def radius_small : ℝ := 9
def radius_inscribed : ℝ := 8

-- Define tangency conditions and relationships based on the problem statement
def large_semicircle (R : ℝ) := { x : ℝ // 0 <= x ∧ x <= R }
def small_semicircle (r : ℝ) := { x : ℝ // 0 <= x ∧ x <= r }

-- Prove the radius of the circle inscribed between the two semicircles
theorem inscribed_circle_radius :
  large_semicircle radius_large ∧ small_semicircle radius_small →
  ∃ (x : ℝ), x = radius_inscribed := 
by
  intro h;  -- Assume the hypothesis h
  exists radius_inscribed;  -- Show the existence of the radius of the inscribed circle
  have hp1 : sqrt (324 - 36 * radius_inscribed) + radius_small = sqrt (81 + 18 * radius_inscribed) := sorry,
  have hp2 : sqrt (324 - 36 * radius_inscribed) = sqrt (81 + 18 * radius_inscribed) - 9 := sorry,
  have h_sqr : (324 - 36 * radius_inscribed) = (sqrt (81 + 18 * radius_inscribed) - 9)^2 := sorry,
  sorry  -- Proof skipped for simplicity of problem setup

end inscribed_circle_radius_l280_280255


namespace sum_of_squares_l280_280648

theorem sum_of_squares (a b : ℝ) (h1 : (a + b) / 2 = 8) (h2 : Real.sqrt (a * b) = 2 * Real.sqrt 5) :
  a^2 + b^2 = 216 :=
by
  sorry

end sum_of_squares_l280_280648


namespace remaining_fruits_is_96_l280_280049

def numberOfApplesBefore : ℕ := 180
def numberOfPlumsBefore : ℕ := numberOfApplesBefore / 3
def fractionPicked : ℝ := 3 / 5

def applesPicked : ℕ := fractionPicked * numberOfApplesBefore
def applesRemaining : ℕ := numberOfApplesBefore - applesPicked

def plumsPicked : ℕ := fractionPicked * numberOfPlumsBefore
def plumsRemaining : ℕ := numberOfPlumsBefore - plumsPicked

def totalFruitsRemaining : ℕ := applesRemaining + plumsRemaining

theorem remaining_fruits_is_96 : totalFruitsRemaining = 96 := by
  sorry

end remaining_fruits_is_96_l280_280049


namespace max_value_expression_l280_280311

theorem max_value_expression (a b c : ℝ) 
  (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) 
  (h : a^2 + b^2 + c^2 = 1) : 
  2 * a * b * Real.sqrt 3 + 2 * b * c ≤ 2 :=
sorry

end max_value_expression_l280_280311


namespace zero_follows_two_count_l280_280589

theorem zero_follows_two_count : 
  let numbers := list.range' 100 (999 - 100 + 1)
  let condition (n : ℕ) : bool := 
    let digits := n.digits 10
    (2 ∈ digits) ∧ list.nth digits (list.index_of 2 digits + 1) = some 0
  in
  ((list.filter condition numbers).length = 19) :=
by
  let numbers := list.range' 100 (999 - 100 + 1)
  let condition (n : ℕ) : bool := 
    let digits := n.digits 10
    (2 ∈ digits) ∧ list.nth digits (list.index_of 2 digits + 1) = some 0
  exact ((list.filter condition numbers).length = 19)
  sorry

end zero_follows_two_count_l280_280589


namespace general_term_l280_280820

noncomputable def sequence (n : ℕ) : ℚ :=
match n with
| 0 => -1
| 1 => 1
| 2 => -9/5
| 3 => 27/7
-- Add more as required by the sequence pattern, if necessary

theorem general_term (a : ℕ → ℚ) :
  (∀ n, a n = (-1) ^ (n - 1) * (3^ (n - 1)) / (2 * n - 1)) → 
  a 0 = -1 ∧ 
  a 1 = 1 ∧ 
  a 2 = -9 / 5 ∧
  a 3 = 27 / 7
:= sorry

end general_term_l280_280820


namespace minimum_room_side_for_table_movement_l280_280037

-- Define the dimensions of the table
def table_length : ℤ := 9
def table_width : ℤ := 12

-- Define the calculation for the diagonal of the table
noncomputable def table_diagonal : ℤ :=
  (table_length ^ 2 + table_width ^ 2).sqrt

-- Define a room with side S
def room_side (S : ℤ) : Prop :=
  S ≥ table_diagonal

-- The proof goal
theorem minimum_room_side_for_table_movement : ∃ S, room_side S ∧ S = 15 :=
by
  use 15
  split
  sorry

end minimum_room_side_for_table_movement_l280_280037


namespace negation_even_l280_280511

open Nat

theorem negation_even (x : ℕ) (h : 0 < x) :
  (∀ x : ℕ, 0 < x → Even x) ↔ ¬ (∃ x : ℕ, 0 < x ∧ Odd x) :=
by
  sorry

end negation_even_l280_280511


namespace concurrency_of_reflected_lines_l280_280169

variables {α : Type*} [CommRing α] {A B C G A' B' C' A'' B'' C'' : α}

open EuclideanGeometry

noncomputable def centroid (A B C : α) : α := sorry  -- define centroid function

-- Conditions
axiom cond1 : foot_of_perpendicular G B C = A'
axiom cond2 : foot_of_perpendicular G C A = B'
axiom cond3 : foot_of_perpendicular G A B = C'

axiom cond4 : reflection G A' = A''
axiom cond5 : reflection G B' = B''
axiom cond6 : reflection G C' = C''

-- Statement to prove
theorem concurrency_of_reflected_lines (ABC : triangle α) (G : α) :
  concurrent_lines (line_through A A'') (line_through B B'') (line_through C C'') :=
sorry

end concurrency_of_reflected_lines_l280_280169


namespace discount_percentage_l280_280025

theorem discount_percentage (original_price new_price : ℕ) (h₁ : original_price = 120) (h₂ : new_price = 96) : 
  ((original_price - new_price) * 100 / original_price) = 20 := 
by
  -- sorry is used here to skip the proof
  sorry

end discount_percentage_l280_280025


namespace sum_of_remaining_digit_is_correct_l280_280508

-- Define the local value calculation function for a particular digit with its place value
def local_value (digit place_value : ℕ) : ℕ := digit * place_value

-- Define the number in question
def number : ℕ := 2345

-- Define the local values for each digit in their respective place values
def local_value_2 : ℕ := local_value 2 1000
def local_value_3 : ℕ := local_value 3 100
def local_value_4 : ℕ := local_value 4 10
def local_value_5 : ℕ := local_value 5 1

-- Define the given sum of the local values
def given_sum : ℕ := 2345

-- Define the sum of the local values of the digits 2, 3, and 5
def sum_of_other_digits : ℕ := local_value_2 + local_value_3 + local_value_5

-- Define the target sum which is the sum of the local value of the remaining digit
def target_sum : ℕ := given_sum - sum_of_other_digits

-- Prove that the sum of the local value of the remaining digit is equal to 40
theorem sum_of_remaining_digit_is_correct : target_sum = 40 := 
by
  -- The proof will be provided here
  sorry

end sum_of_remaining_digit_is_correct_l280_280508


namespace part1_part2_l280_280684

-- Given these definitions for real numbers a and b satisfying a + b = 1
variables (a b : ℝ)

-- First part
theorem part1 (h : a + b = 1) : a^3 + b^3 ≥ 1/4 :=
by {
  sorry
}

-- Second part
theorem part2 (h : a + b = 1) (hx : ∃ x : ℝ, |x - a| + |x - b| ≤ 5) : 
  ∀ y, 2a + 3b = y → 0 ≤ y ∧ y ≤ 5 :=
by {
  sorry
}

end part1_part2_l280_280684


namespace five_dice_not_all_same_number_l280_280897
open Classical

noncomputable def probability_not_all_same (n : ℕ) : ℚ :=
  1 - 1 / (6^n)

theorem five_dice_not_all_same_number :
  probability_not_all_same 5 = 1295 / 1296 :=
by
  sorry

end five_dice_not_all_same_number_l280_280897


namespace problem_1_problem_2_l280_280203

def f (x : ℝ) : ℝ := |(1 - 2 * x)| - |(1 + x)|

theorem problem_1 :
  {x | f x ≥ 4} = {x | x ≤ -2 ∨ x ≥ 6} :=
sorry

theorem problem_2 (a : ℝ) : 
  (∀ x : ℝ, a^2 + 2 * a + |(1 + x)| > f x) → (a < -3 ∨ a > 1) :=
sorry

end problem_1_problem_2_l280_280203


namespace det_B_squared_minus_3_B_l280_280714

theorem det_B_squared_minus_3_B :
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 3, 2]
  det (B * B - 3 • B) = 88 := by
  sorry

end det_B_squared_minus_3_B_l280_280714


namespace statement_A_statement_B_main_theorem_l280_280144

variables (a b c d : ℝ)

theorem statement_A (h : a * c^2 > b * c^2) (hc : c ≠ 0) : a > b :=
by sorry

theorem statement_B (h₁ : a > b) (h₂ : c > d) : a + c > b + d :=
by sorry

-- Optional: could also define the main theorem for proving the correctness of statements A and B.
theorem main_theorem (ha : a * c^2 > b * c^2) (hc : c ≠ 0) (hd₁ : a > b) (hd₂ : c > d) :
  (a > b) ∧ (a + c > b + d) :=
  ⟨statement_A _ _ ha hc, statement_B _ _ hd₁ hd₂⟩

end statement_A_statement_B_main_theorem_l280_280144


namespace min_value_abs_plus_one_l280_280990

theorem min_value_abs_plus_one : ∃ x : ℝ, |x| + 1 = 1 :=
by
  use 0
  sorry

end min_value_abs_plus_one_l280_280990


namespace expression_conversion_l280_280632

def base_convert (n : Nat) (base : Nat) : Nat :=
-- This is a placeholder for the actual base conversion function.
sorry

def val_263_8 := base_convert 263 8
def val_13_3 := base_convert 13 3
def val_243_7 := base_convert 243 7
def val_35_6 := base_convert 35 6

theorem expression_conversion :
  val_263_8 / val_13_3 + val_243_7 / val_35_6 ≈ 35.442 :=
by
  have h1 : val_263_8 = 179 := sorry
  have h2 : val_13_3 = 6 := sorry
  have h3 : val_243_7 = 129 := sorry
  have h4 : val_35_6 = 23 := sorry
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end expression_conversion_l280_280632


namespace preimage_of_43_is_21_l280_280207

def f (x y : ℝ) : ℝ × ℝ := (x + 2 * y, 2 * x - y)

theorem preimage_of_43_is_21 : f 2 1 = (4, 3) :=
by {
  -- Proof omitted
  sorry
}

end preimage_of_43_is_21_l280_280207


namespace two_dice_sum_greater_than_four_l280_280440
open Classical

def probability_sum_greater_than_four : ℚ := by sorry

theorem two_dice_sum_greater_than_four :
  probability_sum_greater_than_four = 5 / 6 :=
sorry

end two_dice_sum_greater_than_four_l280_280440


namespace max_intersection_points_l280_280843

theorem max_intersection_points (circle1 circle2 circle3 : set (ℝ × ℝ)) (line : set (ℝ × ℝ)) :
  (∀ i ∈ {circle1, circle2, circle3}, ∃ a b : ℝ, a ≠ b ∧ {p ∈ i | p ∈ line}.card ≤ 2) →
  (∀ i j ∈ {circle1, circle2, circle3}, i ≠ j → disjoint i j) →
  ({p ∈ circle1 | p ∈ line}.card + {p ∈ circle2 | p ∈ line}.card + {p ∈ circle3 | p ∈ line}.card) ≤ 6 := 
by
  intro h1 h2
  sorry

end max_intersection_points_l280_280843


namespace probability_sum_greater_than_four_l280_280455

theorem probability_sum_greater_than_four : 
  let num_dice := 2
  let sides_per_die := 6
  let favorable_outcomes := { (a, b) | a > 0 ∧ a ≤ sides_per_die ∧ b > 0 ∧ b ≤ sides_per_die ∧ a + b > 4 }
  let total_outcomes := sides_per_die * sides_per_die
  let probability := (favorable_outcomes.card : ℚ) / total_outcomes
  probability = 5 / 6 :=
by
  sorry

end probability_sum_greater_than_four_l280_280455


namespace extrema_of_f_l280_280389

noncomputable def f (x : ℝ) : ℝ := -x^3 + 3 * x + 1

theorem extrema_of_f :
  (∃ x : ℝ, f x = -1 ∧ ∀ y : ℝ, f y ≥ f x) ∧
  (∃ x : ℝ, f x = 3 ∧ ∀ y : ℝ, f y ≤ f x) :=
by
  sorry

end extrema_of_f_l280_280389


namespace pyramid_surface_area_is_correct_l280_280579

-- Definitions of the geometrical elements
def side_length : ℝ := 6 -- Given side length of the pentagonal base
def height : ℝ := 15 -- Given vertical height above the center of the base

-- Function to calculate the total surface area of the pyramid
noncomputable def total_surface_area_of_pyramid (s h : ℝ) : ℝ :=
  let R := s / (2 * Real.sin (Real.pi / 5))
  let slant_height := Real.sqrt (h^2 + R^2)
  let area_triangle := (1 / 2) * s * slant_height
  let area_pentagon := (5 * s^2) / (4 * Real.tan (Real.pi / 5))
  area_pentagon + 5 * area_triangle

-- The statement to prove
theorem pyramid_surface_area_is_correct :
  total_surface_area_of_pyramid side_length height = 299.16 :=
by
  sorry

end pyramid_surface_area_is_correct_l280_280579


namespace flux_face_ABC_flux_outer_surface_pyramid_circulation_contour_ABC_pyramid_l280_280197

noncomputable def field (x y z : ℝ) : ℝ × ℝ × ℝ := ((3 * y^2 - 5 * z^2), (5 * x^2 - 3 * z^2), 8 * x * y)

structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

def O : Point := {x := 0, y := 0, z := 0}
def A : Point := {x := 3, y := 0, z := 0}
def B : Point := {x := 0, y := 5, z := 0}
def C : Point := {x := 0, y := 0, z := 8}

-- Proof problem for part (a): Flux through face ABC
def flux_through_ABC : ℝ := -3245 / 49

theorem flux_face_ABC :
  ∃ (Φ : ℝ), Φ = flux_through_ABC := 
sorry

-- Proof problem for part (b): Flux through outer surface using Gauss theorem
def flux_outer_surface : ℝ := 0

theorem flux_outer_surface_pyramid :
  ∃ (Φ_S : ℝ), Φ_S = flux_outer_surface := 
sorry

-- Proof problem for part (c): Circulation along contour ABC
def circulation_contour_ABC : ℝ := 0

theorem circulation_contour_ABC_pyramid :
  ∃ (Γ : ℝ), Γ = circulation_contour_ABC := 
sorry

end flux_face_ABC_flux_outer_surface_pyramid_circulation_contour_ABC_pyramid_l280_280197


namespace choice_related_to_age_probability_B_second_day_maximum_flash_sale_probability_l280_280080

-- Part 1
def contingency_table : Type := 
  {a b c d : ℕ // a + b + c + d = 100 ∧ a + b = 60 ∧ c + d = 40 ∧ a + c = 50 ∧ b + d = 50 ∧ a = 40 ∧ b = 20 ∧ c = 10 ∧ d = 30}

theorem choice_related_to_age (t : contingency_table) : 
  let χ2 := (t.1 * (t.2 * t.3 - t.4 * t.2)) ^ 2 / (t.1 * (t.2 + t.3) * (t.2 + t.4) * (t.4 + t.3)) in 
  χ2 > 10.828 :=
by sorry

-- Part 2
def shopping_probabilities (pA pB : ℝ → ℝ) (initial_prob : ℝ) (second_day_prob_A_given_A second_day_prob_A_given_B : ℝ) : Type :=
  initial_prob = 0.5 ∧ second_day_prob_A_given_A = 0.7 ∧ second_day_prob_A_given_B = 0.8

theorem probability_B_second_day (prob : shopping_probabilities 0.5 0.5 0.7 0.8) : 
  (0.5 * (1 - 0.7) + 0.5 * (1 - 0.8)) = 0.25 :=
by sorry

-- Part 3
def flash_sale_probability (p : ℝ) : ℝ := 10 * (1 - p) ^ 3 * p ^ 2

theorem maximum_flash_sale_probability : 
  ∃ p₀ ∈ (Ioo 0 1), is_local_max flash_sale_probability p₀ ∧ p₀ = 2/5 :=
by sorry

end choice_related_to_age_probability_B_second_day_maximum_flash_sale_probability_l280_280080


namespace units_digit_sum_factorials_l280_280656

-- Definitions
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def units_digit (n : ℕ) : ℕ := n % 10

-- Theorem to prove
theorem units_digit_sum_factorials : 
  units_digit (∑ i in Finset.range 2011, factorial i) = 3 :=
by
  sorry

end units_digit_sum_factorials_l280_280656


namespace boys_neither_happy_nor_sad_l280_280543

theorem boys_neither_happy_nor_sad:
  ∀ (total_children happy_children sad_children neither_happy_nor_sad boys girls happy_boys sad_girls : ℕ),
    total_children = 60 →
    happy_children = 30 →
    sad_children = 10 →
    neither_happy_nor_sad = 20 →
    boys = 19 →
    girls = 41 →
    happy_boys = 6 →
    sad_girls = 4 →
    (boys - happy_boys) - (sad_children - sad_girls) = 7 :=
by
  intros total_children happy_children sad_children neither_happy_nor_sad boys girls happy_boys sad_girls
  assume h_total_children h_happy_children h_sad_children h_neither_happy_nor_sad h_boys h_girls h_happy_boys h_sad_girls
  sorry

end boys_neither_happy_nor_sad_l280_280543


namespace complement_of_25_l280_280667

theorem complement_of_25 (A : ℝ) (h : A = 25) : 90 - A = 65 :=
by
  rw [h]
  norm_num
  -- This concludes the proof
  sorry

end complement_of_25_l280_280667


namespace five_dice_not_all_same_probability_l280_280878

noncomputable def probability_not_all_same : ℚ :=
  let total_outcomes := (6:ℚ) ^ 5
  let favorable_outcomes := (6:ℚ)
  1 - (favorable_outcomes / total_outcomes)

theorem five_dice_not_all_same_probability :
  probability_not_all_same = 1295 / 1296 :=
by
  unfold probability_not_all_same
  norm_cast
  simp
  sorry

end five_dice_not_all_same_probability_l280_280878


namespace distance_to_other_focus_l280_280789

def ellipse : set (ℝ × ℝ) := {p | let ⟨x, y⟩ := p in (x^2 / 25 + y^2 = 1)}

def is_focus (f : ℝ × ℝ) : Prop := let ⟨a, b⟩ := f in b = 0 ∧ (a = 5 ∨ a = -5)

theorem distance_to_other_focus 
  (P : ℝ × ℝ) 
  (hP : P ∈ ellipse) 
  (focus1 : ℝ × ℝ) 
  (focus2 : ℝ × ℝ) 
  (hfocus1 : is_focus focus1) 
  (hfocus2 : is_focus focus2) 
  (hneq : focus1 ≠ focus2) 
  (d1 : ℝ) 
  (hd1 : d1 = dist P focus1) 
  (hwant : d1 = 2) 
:
  ∃ d2, d2 = dist P focus2 ∧ d2 = 8 := 
sorry

end distance_to_other_focus_l280_280789


namespace total_games_played_l280_280350

-- Define the conditions as Lean 4 definitions
def games_won : Nat := 12
def games_lost : Nat := 4

-- Prove the total number of games played is 16
theorem total_games_played : games_won + games_lost = 16 := 
by
  -- Place a proof placeholder
  sorry

end total_games_played_l280_280350


namespace total_amount_spent_l280_280145

def speakers : ℝ := 118.54
def new_tires : ℝ := 106.33
def window_tints : ℝ := 85.27
def seat_covers : ℝ := 79.99
def scheduled_maintenance : ℝ := 199.75
def steering_wheel_cover : ℝ := 15.63
def air_fresheners_set : ℝ := 12.96
def car_wash : ℝ := 25.0

theorem total_amount_spent :
  speakers + new_tires + window_tints + seat_covers + scheduled_maintenance + steering_wheel_cover + air_fresheners_set + car_wash = 643.47 :=
by
  sorry

end total_amount_spent_l280_280145


namespace maximum_integer_value_of_a_l280_280204

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x^2 + (2 - a) * x - a * Real.log x

theorem maximum_integer_value_of_a (a : ℝ) (h : ∀ x ≥ 1, f x a > 0) : a ≤ 2 :=
sorry

end maximum_integer_value_of_a_l280_280204


namespace fifth_inequality_proof_l280_280333

theorem fifth_inequality_proof : 
  1 + (1 / (2:ℝ)^2) + (1 / (3:ℝ)^2) + (1 / (4:ℝ)^2) + (1 / (5:ℝ)^2) + (1 / (6:ℝ)^2) < (11 / 6) :=
by {
  sorry
}

end fifth_inequality_proof_l280_280333


namespace find_m_for_line_passing_through_circle_center_l280_280139

theorem find_m_for_line_passing_through_circle_center :
  ∀ (m : ℝ), (∀ (x y : ℝ), 2 * x + y + m = 0 ↔ (x - 1)^2 + (y + 2)^2 = 5) → m = 0 :=
by
  intro m
  intro h
  -- Here we construct that the center (1, -2) must lie on the line 2x + y + m = 0
  -- using the given condition of the circle center.
  have center := h 1 (-2)
  -- solving for the equation at the point (1, -2) must yield m = 0
  sorry

end find_m_for_line_passing_through_circle_center_l280_280139


namespace color_dots_l280_280627

-- Define the vertices and the edges of the graph representing the figure
inductive Color : Type
| red : Color
| white : Color
| blue : Color

structure Dot :=
  (color : Color)

structure Edge :=
  (u : Dot)
  (v : Dot)

def valid_coloring (dots : List Dot) (edges : List Edge) : Prop :=
  ∀ e ∈ edges, e.u.color ≠ e.v.color

def count_colorings : Nat :=
  6 * 2

theorem color_dots (dots : List Dot) (edges : List Edge)
  (h1 : ∀ d ∈ dots, d.color = Color.red ∨ d.color = Color.white ∨ d.color = Color.blue)
  (h2 : valid_coloring dots edges) :
  count_colorings = 12 :=
by
  sorry

end color_dots_l280_280627


namespace probability_sum_greater_than_four_l280_280490

theorem probability_sum_greater_than_four : 
  let total_outcomes := 36 in
  let outcomes_sum_less_or_equal_4 := 6 in
  let prob_sum_less_or_equal_4 := outcomes_sum_less_or_equal_4 / total_outcomes in
  prob_sum_less_or_equal_4 = (1 : ℝ) / 6 → 
  (1 - prob_sum_less_or_equal_4) = (5 : ℝ) / 6 := 
by 
  intros total_outcomes outcomes_sum_less_or_equal_4 prob_sum_less_or_equal_4 h1,
  sorry

end probability_sum_greater_than_four_l280_280490


namespace james_weekly_expenses_l280_280295

theorem james_weekly_expenses :
  let rent := 1200 in
  let utilities := rent * 0.20 in
  let hours_per_week_per_employee := 16 * 5 in
  let total_hours := 2 * hours_per_week_per_employee in
  let employee_wages := total_hours * 12.50 in
  let total_expenses := rent + utilities + employee_wages in
  total_expenses = 3440 :=
by
  let rent := 1200
  let utilities := rent * 0.20
  let hours_per_week_per_employee := 16 * 5
  let total_hours := 2 * hours_per_week_per_employee
  let employee_wages := total_hours * 12.50
  let total_expenses := rent + utilities + employee_wages
  sorry

end james_weekly_expenses_l280_280295


namespace part1_69_part1_97_not_part2_difference_numbers_in_range_l280_280278

def is_difference_number (n : ℕ) : Prop :=
  (n % 7 = 6) ∧ (n % 5 = 4)

theorem part1_69 : is_difference_number 69 :=
sorry

theorem part1_97_not : ¬is_difference_number 97 :=
sorry

theorem part2_difference_numbers_in_range :
  {n : ℕ | is_difference_number n ∧ 500 < n ∧ n < 600} = {524, 559, 594} :=
sorry

end part1_69_part1_97_not_part2_difference_numbers_in_range_l280_280278


namespace average_decrease_l280_280370

theorem average_decrease (avg_6 : ℝ) (obs_7 : ℝ) (new_avg : ℝ) (decrease : ℝ) :
  avg_6 = 11 → obs_7 = 4 → (6 * avg_6 + obs_7) / 7 = new_avg → avg_6 - new_avg = decrease → decrease = 1 :=
  by
    intros h1 h2 h3 h4
    rw [h1, h2] at *
    sorry

end average_decrease_l280_280370


namespace quadrilateral_AM_lt_MC_l280_280168

theorem quadrilateral_AM_lt_MC
  {A B C D M : Point}
  (hABltBC : dist A B < dist B C)
  (hADltDC : dist A D < dist D C)
  (hM_on_BD : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = Point_on_line_segment B D t) :
  dist A M < dist M C := 
sorry

end quadrilateral_AM_lt_MC_l280_280168


namespace P_and_S_cannot_finish_third_l280_280141

open Function

-- Define the runners
inductive Runner
| P | Q | R | S | T
deriving DecidableEq

open Runner

-- Define the condition that one runner beats another
axiom beats : Runner → Runner → Prop

-- Given conditions
axiom P_beats_Q : beats P Q
axiom P_beats_R : beats P R
axiom Q_beats_S : beats Q S
axiom T_finishe_after_P_and_before_Q : beats P T ∧ beats T Q

-- Define a function that indicates the position of a runner
def position : Runner → ℕ

-- Proof statement
theorem P_and_S_cannot_finish_third :
  ∀ pos, (∀ r1 r2, beats r1 r2 → position r1 < position r2) →
  (position P ≠ 3) ∧ (position S ≠ 3) :=
by intros; sorry

end P_and_S_cannot_finish_third_l280_280141


namespace log_pq_eq_log_p_add_q_iff_p_eq_q_div_q_add_one_l280_280982

theorem log_pq_eq_log_p_add_q_iff_p_eq_q_div_q_add_one (p q : ℝ) (hpq : p * (q + 1) = q) :
  log p + log q = log (p + q) ↔ p = q / (q + 1) :=
by
  sorry

end log_pq_eq_log_p_add_q_iff_p_eq_q_div_q_add_one_l280_280982


namespace original_population_before_changes_l280_280042

open Nat

def halved_population (p: ℕ) (years: ℕ) : ℕ := p / (2^years)

theorem original_population_before_changes (P_init P_final : ℕ)
    (new_people : ℕ) (people_moved_out : ℕ) :
    new_people = 100 →
    people_moved_out = 400 →
    ∀ years, (years = 4 → halved_population P_final years = 60) →
    ∃ P_before_change, P_before_change = 780 ∧
    P_init = P_before_change + new_people - people_moved_out ∧
    halved_population P_init years = P_final := 
by
  intros
  sorry

end original_population_before_changes_l280_280042


namespace sum_of_integer_solutions_is_8_l280_280806

theorem sum_of_integer_solutions_is_8 :
  let f := λ x: ℝ, 8 * ((|x+3| - |x-5|) / (|2*x-11| - |2*x+7|)) - 9 * ((|x+3| + |x-5|) / (|2*x-11| + |2*x+7|)) in
  (∑ i in Finset.filter (λ x, (f x ≥ -8) ∧ (|x| < 90)) (Finset.range 180).filter_map (λ x, if x%2 = 0 then some (x / 2 : ℤ) else none)) = 8 := 
sorry

end sum_of_integer_solutions_is_8_l280_280806


namespace inscribed_circle_radius_l280_280265

theorem inscribed_circle_radius (R r x : ℝ) (hR : R = 18) (hr : r = 9) :
  x = 8 :=
sorry

end inscribed_circle_radius_l280_280265


namespace three_point_three_six_as_fraction_l280_280537

theorem three_point_three_six_as_fraction : 3.36 = (84 : ℚ) / 25 := 
by
  sorry

end three_point_three_six_as_fraction_l280_280537


namespace antichain_inequality_l280_280319

noncomputable def P (E : Type*) [Fintype E] := Set (Set E)

variable (E : Type*) [Fintype E] (n : ℕ) (cardE : Fintype.card E = n) 
variable (A : Set (Set E))

definition isAntichain (A : Set (Set E)) : Prop := ∀ B₁ B₂ ∈ A, B₁ ⊆ B₂ → B₁ = B₂

theorem antichain_inequality (A_is_antichain : isAntichain A)
  (hA : A ⊆ (P E)) :
  (∑ B in A, 1 / (Nat.choose n (Fintype.card B : ℕ)) : ℝ) ≤ 1 ∧ 
  (A.card : ℕ) ≤ Nat.choose n (Nat.floor (n/2 : ℝ)) :=
by 
  sorry

end antichain_inequality_l280_280319


namespace simplify_expr1_simplify_expr2_l280_280353

variables (x y a b : ℝ)

-- Problem 1
theorem simplify_expr1 : 3 * (4 * x - 2 * y) - 3 * (-y + 8 * x) = -12 * x - 3 * y := 
by sorry

-- Problem 2
theorem simplify_expr2 : 3 * a^2 - 2 * (2 * a^2 - (2 * a * b - a^2) + 4 * a * b) = -3 * a^2 - 4 * a * b := 
by sorry

end simplify_expr1_simplify_expr2_l280_280353


namespace median_of_sequence_is_1976_point_5_l280_280956

/-- Definition of the sequence containing integers from 1 to 2020 and their squares -/
def numbers_sequence : List ℕ :=
  (List.range 2020).map (λ n, n + 1) ++ (List.range 2020).map (λ n, (n + 1) * (n + 1))

/-- Proof that the median of the sequence is 1976.5 -/
theorem median_of_sequence_is_1976_point_5 :
  (List.sort numbers_sequence).get? 2019 ≠ none ∧
  (List.sort numbers_sequence).get? 2020 ≠ none ∧
  ((List.sort numbers_sequence).get! 2019 + (List.sort numbers_sequence).get! 2020) / 2 = 1976.5 :=
by
  sorry

end median_of_sequence_is_1976_point_5_l280_280956


namespace difference_between_blue_and_red_balls_l280_280838

-- Definitions and conditions
def number_of_blue_balls := ℕ
def number_of_red_balls := ℕ
def difference_between_balls (m n : ℕ) := m - n

-- Problem statement: Prove that the difference between number_of_blue_balls and number_of_red_balls
-- can be any natural number greater than 1.
theorem difference_between_blue_and_red_balls (m n : ℕ) (h1 : m > n) (h2 : 
  let P_same := (n * (n - 1) + m * (m - 1)) / ((n + m) * (n + m - 1))
  let P_diff := 2 * (n * m) / ((n + m) * (n + m - 1))
  P_same = P_diff
  ) : ∃ a : ℕ, a > 1 ∧ a = m - n :=
by
  sorry

end difference_between_blue_and_red_balls_l280_280838


namespace sum_of_squares_l280_280646

theorem sum_of_squares (a b : ℝ) (h1 : a + b = 16) (h2 : a * b = 20) : a^2 + b^2 = 216 :=
by
  sorry

end sum_of_squares_l280_280646


namespace find_lambda_l280_280178

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables {A B C D P : V}

-- D is the midpoint of side BC
def midpoint (D B C : V) : Prop := D = (B + C) / 2

-- Conditions
variables (h₁ : midpoint D B C)
variables (h₂ : (P - A) + (B - P) + (C - P) = 0)
variables (h₃ : (A - P) = λ * (D - P))

-- The proof goal
theorem find_lambda : λ = -2 :=
by
  sorry

end find_lambda_l280_280178


namespace trig_expression_evaluation_l280_280678

theorem trig_expression_evaluation (a : ℝ) (P : ℝ × ℝ) (hP : P = (-4, 3)) :
  let α := real.arctan (-3/4)
  in (cos (π/2 + a) * sin (-π - a)) / (cos (11*π/2 - a) * sin (9*π/2 + a)) = -3/4 :=
by
  sorry

end trig_expression_evaluation_l280_280678


namespace units_digit_sum_factorials_l280_280655

-- Definitions
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def units_digit (n : ℕ) : ℕ := n % 10

-- Theorem to prove
theorem units_digit_sum_factorials : 
  units_digit (∑ i in Finset.range 2011, factorial i) = 3 :=
by
  sorry

end units_digit_sum_factorials_l280_280655


namespace probability_sum_greater_than_four_l280_280453

theorem probability_sum_greater_than_four : 
  let num_dice := 2
  let sides_per_die := 6
  let favorable_outcomes := { (a, b) | a > 0 ∧ a ≤ sides_per_die ∧ b > 0 ∧ b ≤ sides_per_die ∧ a + b > 4 }
  let total_outcomes := sides_per_die * sides_per_die
  let probability := (favorable_outcomes.card : ℚ) / total_outcomes
  probability = 5 / 6 :=
by
  sorry

end probability_sum_greater_than_four_l280_280453


namespace denominator_exceeds_numerator_by_10_l280_280775

-- Define G as the repeating decimal 0.729729729...
def G : ℝ := 0.729.repeating

-- Define the fraction representation of G in lowest terms
def G_fraction : ℚ := 27 / 37

-- Theorem statement: The difference between the denominator and the numerator is 10
theorem denominator_exceeds_numerator_by_10 : (G_fraction.denom : ℤ) - (G_fraction.num : ℤ) = 10 :=
sorry

end denominator_exceeds_numerator_by_10_l280_280775


namespace inequality_solution_set_l280_280153

theorem inequality_solution_set (a x : ℝ) (h : 4 ^ a = 2 ^ (a + 2)) : (a^(2*x + 1) > a^(x - 1)) ↔ (x > -2) :=
by
  sorry

end inequality_solution_set_l280_280153


namespace equation_D_has_two_distinct_real_roots_l280_280906

def quadratic_has_two_distinct_real_roots (a b c : ℝ) : Prop :=
  b^2 - 4 * a * c > 0

theorem equation_D_has_two_distinct_real_roots : quadratic_has_two_distinct_real_roots 1 2 (-1) :=
by {
  sorry
}

end equation_D_has_two_distinct_real_roots_l280_280906


namespace simplify_336_to_fraction_l280_280516

theorem simplify_336_to_fraction :
  let gcd_36_100 := Nat.gcd 36 100
  3.36 = (84 : ℚ) / 25 := 
by
  let g := Nat.gcd 36 100
  have h1 : 3.36 = 3 + 0.36 := by norm_num
  have h2 : 0.36 = 36 / 100 := by norm_num
  have h3 : g = 4 := by norm_num [Nat.gcd, Nat.gcd_def, Nat.gcd_rec]
  have h4 : (36 : ℚ) / 100 = 9 / 25 := by norm_num; field_simp [h3];
  have h5 : (3 : ℚ) + (9 / 25) = 84 / 25 := by norm_num; field_simp;
  rw [h1, h2, h4, h5]

end simplify_336_to_fraction_l280_280516


namespace three_point_three_six_as_fraction_l280_280533

theorem three_point_three_six_as_fraction : 3.36 = (84 : ℚ) / 25 := 
by
  sorry

end three_point_three_six_as_fraction_l280_280533


namespace range_of_a_l280_280592

open Real

-- Definition of the functions f and g
noncomputable def f (x : ℝ) : ℝ := x / (4 * x - 1)

def g (a x : ℝ) : ℝ := x^3 + 3 * a^2 * x + 2 * a

-- Main theorem proving the given problem
theorem range_of_a (a : ℝ) :
  (∀ x1 ∈ Icc (1 / 3) 1, ∃ x2 ∈ Icc 0 1, f x1 = g a x2) →
  a ∈ Icc (-real.sqrt (4 / (3:ℝ)) / 2) (-sqrt (4 / (3:ℝ)) / 2) :=
sorry

end range_of_a_l280_280592


namespace simplify_336_to_fraction_l280_280514

theorem simplify_336_to_fraction :
  let gcd_36_100 := Nat.gcd 36 100
  3.36 = (84 : ℚ) / 25 := 
by
  let g := Nat.gcd 36 100
  have h1 : 3.36 = 3 + 0.36 := by norm_num
  have h2 : 0.36 = 36 / 100 := by norm_num
  have h3 : g = 4 := by norm_num [Nat.gcd, Nat.gcd_def, Nat.gcd_rec]
  have h4 : (36 : ℚ) / 100 = 9 / 25 := by norm_num; field_simp [h3];
  have h5 : (3 : ℚ) + (9 / 25) = 84 / 25 := by norm_num; field_simp;
  rw [h1, h2, h4, h5]

end simplify_336_to_fraction_l280_280514


namespace triangle_area_is_39_l280_280097

noncomputable def vec2 (x y : ℝ) : Matrix (Fin 2) (Fin 1) ℝ :=
  ![![x], ![y]]

def vertices := (vec2 (-2) 3, vec2 8 (-1), vec2 10 6)

def vector_CA := vertices.1 - vertices.3
def vector_CB := vertices.2 - vertices.3

def determinant_2x2 (v w : Matrix (Fin 2) (Fin 1) ℝ) : ℝ :=
  v[0,0] * w[1,0] - v[1,0] * w[0,0]

def area_of_parallelogram := determinant_2x2 vector_CA vector_CB

def triangle_area := area_of_parallelogram / 2

theorem triangle_area_is_39 : triangle_area = 39 := by
  sorry

end triangle_area_is_39_l280_280097


namespace child_ticket_cost_l280_280594

def total_cost_for_group := 54.50
def adult_ticket_cost := 9.50
def number_of_adults := 3
def number_of_total_people := 7

theorem child_ticket_cost :
  ∃ C : ℝ, (number_of_adults * adult_ticket_cost + (number_of_total_people - number_of_adults) * C = total_cost_for_group) ∧
         C = 6.50 :=
by
  sorry

end child_ticket_cost_l280_280594


namespace unique_pair_15_a_b_ab_arithmetic_progression_l280_280624

theorem unique_pair_15_a_b_ab_arithmetic_progression (a b : ℝ) 
  (h1 : a - b = 5) 
  (h2 : 2 * a = 15 + b)
  (h3 : 2 * ab = 2b + a): 
  a = 10 ∧ b = 5 := 
by 
  sorry

end unique_pair_15_a_b_ab_arithmetic_progression_l280_280624


namespace units_digit_sum_factorials_l280_280653

theorem units_digit_sum_factorials : 
  (∑ n in finset.range 2011, (n.factorial % 10)) % 10 = 3 := 
by
  sorry

end units_digit_sum_factorials_l280_280653


namespace probability_correct_l280_280450

def outcome (s₁ s₂ : ℕ) : Prop := s₁ ≥ 1 ∧ s₁ ≤ 6 ∧ s₂ ≥ 1 ∧ s₂ ≤ 6

def sum_outcome_greater_than_four (s₁ s₂ : ℕ) : Prop := outcome s₁ s₂ ∧ s₁ + s₂ > 4

def total_outcomes : ℕ := 36

def favorable_outcomes : ℕ := 30 -- As derived from 36 - 6

def probability_sum_greater_than_four : ℚ := favorable_outcomes / total_outcomes

theorem probability_correct : probability_sum_greater_than_four = 5 / 6 := 
by 
  sorry

end probability_correct_l280_280450


namespace correct_statements_l280_280974

theorem correct_statements
  (a : ℝ) (ha : 0 < a ∧ a ≠ 1)
  (y_exp : ℝ → ℝ)
  (y_square : ℝ → ℝ)
  (y_exp_domain : ∀ x : ℝ, y_exp x = a^x)
  (y_square_domain : ∀ x : ℝ, y_square x = x^2)
  (not_inverse : ¬ ∀ x : ℝ, 2^x = log 3 x)
  (log_monotonic_not_incr : ¬ monotonically_increasing (λ x : ℝ, log 3 (x^2 - 2*x - 3)) (Ici 1))
  (abs_exp_range : set.range (λ x : ℝ, 3^|x|) = Icc 1 ⊤) :
  (y_exp_domain = y_square_domain ∧ abs_exp_range = Icc 1 ⊤) :=
sorry

end correct_statements_l280_280974


namespace range_of_a_l280_280212

noncomputable def A : Set ℝ := {x : ℝ | ((x^2) - x - 2) ≤ 0}

theorem range_of_a (a : ℝ) : (∀ x ∈ A, (x^2 - a*x - a - 2) ≤ 0) → a ≥ (2/3) :=
by
  intro h
  sorry

end range_of_a_l280_280212


namespace cost_of_ingredients_l280_280961

theorem cost_of_ingredients (C : ℝ) 
  (packaging_cost_per_cake : ℝ) (selling_price_per_cake : ℝ) (profit_per_cake : ℝ)
  (h1 : packaging_cost_per_cake = 1)
  (h2 : selling_price_per_cake = 15)
  (h3 : profit_per_cake = 8)
  (h4 : 2 * selling_price_per_cake - 2 * profit_per_cake = C + 2 * packaging_cost_per_cake) :
  C = 12 :=
begin
  sorry
end

end cost_of_ingredients_l280_280961


namespace volume_frustum_l280_280952

open Real

-- Definition of the base edge and height of the original pyramid
def base_edge := 15
def original_height := 12

-- Similar smaller pyramid definitions
def smaller_base_edge := 7.5
def smaller_height := 6

-- Calculate the volume of a pyramid
noncomputable def volume_pyramid (base_edge: ℝ) (height: ℝ): ℝ :=
  (1 / 3) * (base_edge ^ 2) * height

-- Volume of the original pyramid
noncomputable def volume_original_pyramid := volume_pyramid base_edge original_height

-- Volume of the smaller pyramid
noncomputable def volume_smaller_pyramid := volume_pyramid (smaller_base_edge) (smaller_height)

-- Proving the volume of the frustum is 787.5 cm^3
theorem volume_frustum : volume_original_pyramid - volume_smaller_pyramid = 787.5 :=
by
  -- These values should be calculation steps, hence marked as sorry for now.
  sorry

end volume_frustum_l280_280952


namespace average_temperature_week_l280_280604

theorem average_temperature_week :
  let sunday := 99.1
  let monday := 98.2
  let tuesday := 98.7
  let wednesday := 99.3
  let thursday := 99.8
  let friday := 99.0
  let saturday := 98.9
  (sunday + monday + tuesday + wednesday + thursday + friday + saturday) / 7 = 99.0 :=
by
  sorry

end average_temperature_week_l280_280604


namespace area_of_new_section_l280_280030

theorem area_of_new_section (length width : ℕ) (h1 : length = 5) (h2 : width = 7) : length * width = 35 :=
by
  rw [h1, h2]
  exact Nat.mul_comm 5 7 ▸ rfl

end area_of_new_section_l280_280030


namespace probability_sum_greater_than_four_l280_280463

def sum_greater_than_four_probability :=
  (5 / 6: ℚ)

theorem probability_sum_greater_than_four :
  let outcomes := { (a, b) | a in (Finset.range 1 7) ∧ b in (Finset.range 1 7) }
  let favorable_outcomes := outcomes.filter (λ pair, (pair.1 + pair.2) > 4)
  let probability := favorable_outcomes.card.to_rat / outcomes.card.to_rat
  probability = sum_greater_than_four_probability :=
by
  sorry

end probability_sum_greater_than_four_l280_280463


namespace volume_space_inside_sphere_outside_cylinder_cone_l280_280031

noncomputable def volume_cylinder (r h : ℝ) : ℝ := π * r^2 * h
noncomputable def volume_cone (r h : ℝ) : ℝ := (1/3) * π * r^2 * h
noncomputable def volume_sphere (r : ℝ) : ℝ := (4/3) * π * r^3

theorem volume_space_inside_sphere_outside_cylinder_cone :
  let r_sphere : ℝ := 7
  let r_cylinder : ℝ := 4
  let h_cylinder := 2 * Real.sqrt (r_sphere^2 - r_cylinder^2)
  let V_sphere := volume_sphere r_sphere
  let V_cylinder := volume_cylinder r_cylinder h_cylinder
  let V_cone := volume_cone r_cylinder h_cylinder
  V_sphere - V_cylinder - V_cone = π * ((1372 - 128 * Real.sqrt 33) / 3) :=
by
  sorry

end volume_space_inside_sphere_outside_cylinder_cone_l280_280031


namespace pushups_fri_is_39_l280_280331

/-- Defining the number of pushups done by Miriam -/
def pushups_mon := 5
def pushups_tue := 7
def pushups_wed := pushups_tue * 2
def pushups_total_mon_to_wed := pushups_mon + pushups_tue + pushups_wed
def pushups_thu := pushups_total_mon_to_wed / 2
def pushups_total_mon_to_thu := pushups_mon + pushups_tue + pushups_wed + pushups_thu
def pushups_fri := pushups_total_mon_to_thu

/-- Prove the number of pushups Miriam does on Friday equals 39 -/
theorem pushups_fri_is_39 : pushups_fri = 39 := by 
  sorry

end pushups_fri_is_39_l280_280331


namespace parabola_properties_l280_280184

theorem parabola_properties (p : ℝ) (h : p > 0) (F : ℝ × ℝ) (l : ℝ → ℝ) (A B : ℝ × ℝ) (M : ℝ × ℝ)
  (hp : p = 4) 
  (hF : F = (p / 2, 0)) 
  (hA : A.2^2 = 2 * p * A.1) 
  (hB : B.2^2 = 2 * p * B.1) 
  (hM : M = ((A.1 + B.1) / 2, 2)) 
  (hl : ∀ x, l x = 2 * x - 4) 
  : (p = 4) ∧ (l 0 = -4) ∧ (A ≠ B) → 
    (p = 4) ∧ (l 0 = -4) ∧ (A ≠ B) ∧ (|A.1 - B.1| + |A.2 - B.2| = 10) :=
by 
  sorry

end parabola_properties_l280_280184


namespace measles_cases_in_1990_l280_280726

variable (cases1950 cases2000 numberOfYears : ℕ) (cases1990 : ℕ)

axiom h_cases1950 : cases1950 = 600000
axiom h_cases2000 : cases2000 = 200
axiom h_years_between : numberOfYears = 50 
axiom h_year1990 : yearsSince1950_1990 = 40

theorem measles_cases_in_1990 : cases1990 = 120160 := by
  sorry

end measles_cases_in_1990_l280_280726


namespace probability_b_l280_280691

variable {Ω : Type} [ProbabilitySpace Ω]

variable {a b : Event Ω}

-- Given conditions
def p_a := (4 : ℝ) / 5
def p_b := (2 : ℝ) / 5
def p_a_and_b := 0.32
def independent (a b : Event Ω) : Prop := ProbabilityTheory.IndepEvents a b

-- The problem to prove 
theorem probability_b (h_ind : independent a b) (ha : probability a = p_a) (ha_and_b : probability (a ∩ b) = p_a_and_b) :
  probability b = 0.4 :=
by
  sorry

end probability_b_l280_280691


namespace question1_intersection_condition_question2_range_condition_l280_280686

-- Definitions for sets A, B, and C given the condition a > 0
def set_A : set ℝ := {x | x^2 - x - 12 < 0}
def set_B : set ℝ := {x | x^2 + 2x - 8 > 0}
def set_C (a : ℝ) : set ℝ := {x | x^2 - 4*a*x + 3*(a^2) < 0}

-- Proposition for Question 1
theorem question1_intersection_condition (a : ℝ) (ha : a > 0) :
  set_A ∩ ((set_C a) \ set_B) = {x : ℝ | -3 < x ∧ x ≤ 2} :=
sorry

-- Proposition for Question 2
theorem question2_range_condition (a : ℝ) (ha : a > 0) (h_subset : (set_C a) ⊆ (set_A ∩ set_B)) :
  (4/3 : ℝ) ≤ a ∧ a ≤ 2 :=
sorry

end question1_intersection_condition_question2_range_condition_l280_280686


namespace number_of_members_l280_280285

-- Definitions for the conditions
def cost_of_cleats : ℝ := 6
def cost_of_jersey : ℝ := cost_of_cleats + 7
def cost_per_member : ℝ := 2 * (cost_of_cleats + cost_of_jersey)
def total_cost : ℝ := 3360

-- Goal: Prove the number of members is 88
theorem number_of_members :
  ∃ (n : ℕ), n * cost_per_member = total_cost ∧ n = 88 :=
by
  use 88
  split
  calc
    88 * cost_per_member
      = 88 * (2 * (cost_of_cleats + cost_of_jersey)) : rfl
    ... = 88 * (2 * (6 + (6 + 7))) : rfl
    ... = 88 * 38 : rfl
    ... = 3360 : by norm_num,
  rfl

end number_of_members_l280_280285


namespace median_length_l280_280205

noncomputable def triangle_ABC : Type := 
{a b c : ℝ // a^2 + b^2 = c^2 ∧ a = 10 ∧ b = 24 ∧ c = 26}

theorem median_length (t : triangle_ABC) : 
  let ⟨a, b, c, h1, h2, h3⟩ := t in
  (10^2 + 24^2 = 26^2) → 
  (a = 10) →
  (b = 24) → 
  (c = 26) → 
  median_length_to_longest_side t = 13 :=
by
  sorry

end median_length_l280_280205


namespace tan_alpha_plus_15_over_2_pi_l280_280690

theorem tan_alpha_plus_15_over_2_pi 
  (α : ℝ) 
  (h1 : α ∈ Ioo (π / 2) π)
  (h2 : sin α = 1 / 4) 
  : tan (α + 15 * π / 2) = sqrt 15 := 
  sorry

end tan_alpha_plus_15_over_2_pi_l280_280690


namespace probability_sum_greater_than_four_l280_280471

theorem probability_sum_greater_than_four : 
  let total_outcomes := 36 in
  let favorable_outcomes := total_outcomes - 6 in
  let probability := favorable_outcomes / total_outcomes in
  probability = 5 / 6 :=
by
  sorry

end probability_sum_greater_than_four_l280_280471


namespace trigonometric_identity_l280_280196

theorem trigonometric_identity (α : ℝ) : 
  (cot (π / 4 + α) * (1 + cos (2 * α - π / 2)) * cos (2 * α)⁻¹ + 2 * cos (4 * α - 2 * π) = (sin (6 * α) / sin (2 * α))) :=
by 
  sorry

end trigonometric_identity_l280_280196


namespace monotonicity_f_range_of_a_l280_280201

noncomputable def f (x : ℝ) (a : ℝ) := x^2 + (a - 2) * x - a * Real.log x

theorem monotonicity_f (a x : ℝ) (h : a < 0) :
  (a > -2 → (f' x < 0 → x < -a/2) ∧ (f' x > 0 → x > -a/2)) ∧
  (a = -2 → ∀ x, f' x ≥ 0) ∧
  (a < -2 → (f' x < 0 → 1 < x ∧ x < -a/2) ∧ (f' x > 0 → (0 < x ∧ x < 1) ∨ x > -a/2)) := by
  sorry

noncomputable def g (x : ℝ) (a : ℝ) := x^2 + a * x + a * Real.log x

theorem range_of_a (a : ℝ) (h : a < 0) :
  (∀ x, g x > (e + 1) / 2 * a → (0 < x0 ∧ x0 < e) ∧ a = -2 * x0^2 / (x0 + 1) → (a > -2 * e^2 / (e + 1) ∧ a < 0)) := by
  sorry

end monotonicity_f_range_of_a_l280_280201


namespace sin_double_pi_minus_theta_eq_l280_280177

variable {θ : ℝ}
variable {k : ℤ}
variable (h1 : 3 * (Real.cos θ) ^ 2 = Real.tan θ + 3)
variable (h2 : θ ≠ k * Real.pi)

theorem sin_double_pi_minus_theta_eq :
  Real.sin (2 * (Real.pi - θ)) = 2 / 3 :=
sorry

end sin_double_pi_minus_theta_eq_l280_280177


namespace five_dice_not_all_same_probability_l280_280879

noncomputable def probability_not_all_same : ℚ :=
  let total_outcomes := (6:ℚ) ^ 5
  let favorable_outcomes := (6:ℚ)
  1 - (favorable_outcomes / total_outcomes)

theorem five_dice_not_all_same_probability :
  probability_not_all_same = 1295 / 1296 :=
by
  unfold probability_not_all_same
  norm_cast
  simp
  sorry

end five_dice_not_all_same_probability_l280_280879


namespace expected_lone_cars_l280_280557
-- Import Lean's math library to ensure necessary functions and theorems are available.

-- Define a theorem to prove the expected number of lone cars is 1.
theorem expected_lone_cars (n : ℕ) : 
  -- n must be greater than or equal to 1, since there must be at least one car.
  n ≥ 1 -> 
  -- Expected number of lone cars is 1.
  (∑ k in finset.range n, (1 : ℝ) / (k + 1)) = 1 := 
begin
  intro hn, -- Assume n ≥ 1
  sorry,    -- The proof of this theorem is to be provided, but the statement is correct.
end

end expected_lone_cars_l280_280557


namespace part1_part2_l280_280708

def A (t : ℝ) : Prop :=
  ∀ x : ℝ, (t+2)*x^2 + 2*x + 1 > 0

def B (a x : ℝ) : Prop :=
  (a*x - 1)*(x + a) > 0

theorem part1 (t : ℝ) : A t ↔ t < -1 :=
sorry

theorem part2 (a : ℝ) : (∀ t : ℝ, t < -1 → ∀ x : ℝ, B a x) → (0 ≤ a ∧ a ≤ 1) :=
sorry

end part1_part2_l280_280708


namespace shooting_training_part_I_shooting_training_part_II_l280_280580

-- Part I: Probability of |a - b| > 1
theorem shooting_training_part_I (x1 x2 x3 y1 y2 y3 : ℝ)
  (hx1 : 7.5 ≤ x1 ∧ x1 < 8.5) (hx2 : 7.5 ≤ x2 ∧ x2 < 8.5) (hx3 : 7.5 ≤ x3 ∧ x3 < 8.5)
  (hy1 : 9.5 ≤ y1 ∧ y1 < 10.5) (hy2 : 9.5 ≤ y2 ∧ y2 < 10.5) (hy3 : 9.5 ≤ y3 ∧ y3 < 10.5) :
  let scores := [x1, x2, x3, y1, y2, y3] in
  -- There are 15 ways to choose 2 scores from these 6
  -- 9 out of the 15 choices |a - b| > 1
  -- Therefore the probability is 3/5
  (∃ (a b : ℝ) (ha : a ∈ scores) (hb : b ∈ scores) (hab : a ≠ b), |a - b| > 1) →
  (∑ (a b : ℝ) (ha : a ∈ scores) (hb : b ∈ scores) (hab : a ≠ b ∧ |a - b| > 1), 1) / 15 = 3 / 5 :=
sorry

-- Part II: Probability of being more than 1cm away from A, B, and C
theorem shooting_training_part_II (A B C : ℝ)
  (hAC : dist A C = 5) (hAB : dist A B = 6) (hBC : dist B C = sqrt 13) :
  let area_triangle_ABC := 1 / 2 * 5 * 6 * (3 / 5) in
  let area_sector := 1 / 2 * π in
  let area_S := area_triangle_ABC - area_sector in
  area_S / area_triangle_ABC = 1 - (π / 18) :=
sorry

end shooting_training_part_I_shooting_training_part_II_l280_280580


namespace probability_sum_greater_than_four_l280_280459

theorem probability_sum_greater_than_four : 
  let num_dice := 2
  let sides_per_die := 6
  let favorable_outcomes := { (a, b) | a > 0 ∧ a ≤ sides_per_die ∧ b > 0 ∧ b ≤ sides_per_die ∧ a + b > 4 }
  let total_outcomes := sides_per_die * sides_per_die
  let probability := (favorable_outcomes.card : ℚ) / total_outcomes
  probability = 5 / 6 :=
by
  sorry

end probability_sum_greater_than_four_l280_280459


namespace equation_b_has_no_solution_l280_280076

theorem equation_b_has_no_solution : ¬ ∃ x : ℝ, | -2 * x + 1 | + 4 = 0 := by
  sorry

end equation_b_has_no_solution_l280_280076


namespace paths_via_checkpoint_l280_280064

/-- Define the grid configuration -/
structure Point :=
  (x : ℕ) (y : ℕ)

/-- Calculate the binomial coefficient -/
def binomial (n k : ℕ) : ℕ :=
  n.choose k

/-- Define points A, B, C -/
def A : Point := ⟨0, 0⟩
def B : Point := ⟨5, 4⟩
def C : Point := ⟨3, 2⟩

/-- Calculate number of paths from A to C -/
def paths_A_to_C : ℕ :=
  binomial (3 + 2) 2

/-- Calculate number of paths from C to B -/
def paths_C_to_B : ℕ :=
  binomial (2 + 2) 2

/-- Calculate total number of paths from A to B via C -/
def total_paths_A_to_B_via_C : ℕ :=
  (paths_A_to_C * paths_C_to_B)

theorem paths_via_checkpoint :
  total_paths_A_to_B_via_C = 60 :=
by
  -- The proof is skipped as per the instruction
  sorry

end paths_via_checkpoint_l280_280064


namespace blu_ray_price_l280_280610

variable (x : ℝ)

def price_per_movie (dvd_cost blu_ray_cost : ℝ) (total_cost num_movies : ℝ) : ℝ :=
  total_cost / num_movies

def total_dvd_cost (num_dvds dvd_price : ℝ) : ℝ :=
  num_dvds * dvd_price

def total_blu_ray_cost (num_blu_rays blu_ray_price : ℝ) : ℝ :=
  num_blu_rays * blu_ray_price

theorem blu_ray_price :
  let num_dvds := 8
  let dvd_price := 12
  let num_blu_rays := 4
  let average_movie_price := 14
  let num_movies := num_dvds + num_blu_rays
  let total_cost := average_movie_price * num_movies

  total_dvd_cost num_dvds dvd_price + total_blu_ray_cost num_blu_rays x = total_cost →
  x = 18 := by
    sorry

end blu_ray_price_l280_280610


namespace probability_sum_greater_than_four_l280_280460

theorem probability_sum_greater_than_four : 
  let num_dice := 2
  let sides_per_die := 6
  let favorable_outcomes := { (a, b) | a > 0 ∧ a ≤ sides_per_die ∧ b > 0 ∧ b ≤ sides_per_die ∧ a + b > 4 }
  let total_outcomes := sides_per_die * sides_per_die
  let probability := (favorable_outcomes.card : ℚ) / total_outcomes
  probability = 5 / 6 :=
by
  sorry

end probability_sum_greater_than_four_l280_280460


namespace pushups_fri_is_39_l280_280330

/-- Defining the number of pushups done by Miriam -/
def pushups_mon := 5
def pushups_tue := 7
def pushups_wed := pushups_tue * 2
def pushups_total_mon_to_wed := pushups_mon + pushups_tue + pushups_wed
def pushups_thu := pushups_total_mon_to_wed / 2
def pushups_total_mon_to_thu := pushups_mon + pushups_tue + pushups_wed + pushups_thu
def pushups_fri := pushups_total_mon_to_thu

/-- Prove the number of pushups Miriam does on Friday equals 39 -/
theorem pushups_fri_is_39 : pushups_fri = 39 := by 
  sorry

end pushups_fri_is_39_l280_280330


namespace palindrome_exists_l280_280924

theorem palindrome_exists (n : ℕ) : ∃ (N : ℕ), ∃ (x : ℕ), (x = 9 * 5^n * N) ∧ (x.toString = String.reverse x.toString) :=
sorry

end palindrome_exists_l280_280924


namespace probability_correct_l280_280448

def outcome (s₁ s₂ : ℕ) : Prop := s₁ ≥ 1 ∧ s₁ ≤ 6 ∧ s₂ ≥ 1 ∧ s₂ ≤ 6

def sum_outcome_greater_than_four (s₁ s₂ : ℕ) : Prop := outcome s₁ s₂ ∧ s₁ + s₂ > 4

def total_outcomes : ℕ := 36

def favorable_outcomes : ℕ := 30 -- As derived from 36 - 6

def probability_sum_greater_than_four : ℚ := favorable_outcomes / total_outcomes

theorem probability_correct : probability_sum_greater_than_four = 5 / 6 := 
by 
  sorry

end probability_correct_l280_280448


namespace inequality_holds_l280_280322

variable {α β : ℝ}
variable {a b : ℝ} (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : |α| > a)

theorem inequality_holds (θ : ℝ) :
  (α * β - real.sqrt (a^2 * β^2 + b^2 * α^2 - a^2 * b^2)) / (α^2 - a^2) ≤ 
  (β + b * real.sin θ) / (α + a * real.cos θ) ∧ 
  (β + b * real.sin θ) / (α + a * real.cos θ) ≤ 
  (α * β + real.sqrt (a^2 * β^2 + b^2 * α^2 - a^2 * b^2)) / (α^2 - a^2)
:= sorry

end inequality_holds_l280_280322


namespace probability_not_all_same_l280_280867

-- Definitions of conditions from the problem
def six_sided_die_faces : ℕ := 6
def number_of_dice : ℕ := 5

-- Lean statement to prove the probability calculation
theorem probability_not_all_same : 
  let total_outcomes := six_sided_die_faces ^ number_of_dice in
  let all_same_outcomes := six_sided_die_faces in
  (1 - ((all_same_outcomes : ℚ) / total_outcomes)) = (1295 / 1296) := 
by
  sorry

end probability_not_all_same_l280_280867


namespace fraction_representation_of_3_36_l280_280523

theorem fraction_representation_of_3_36 : (336 : ℚ) / 100 = 84 / 25 := 
by sorry

end fraction_representation_of_3_36_l280_280523


namespace fraction_representation_of_3_36_l280_280527

theorem fraction_representation_of_3_36 : (336 : ℚ) / 100 = 84 / 25 := 
by sorry

end fraction_representation_of_3_36_l280_280527


namespace integral_of_20x_squared_l280_280415

theorem integral_of_20x_squared :
  ∫ x in 0..2, 20 * x^2 = 160 / 3 :=
by
  sorry

end integral_of_20x_squared_l280_280415


namespace symmetry_axis_sine_curve_l280_280383

theorem symmetry_axis_sine_curve :
  ∀ (x : ℝ), axis_of_symmetry (λ x, Real.sin (2 * Real.pi * x - Real.pi / 3)) x ↔ x = 5 / 12 :=
sorry

end symmetry_axis_sine_curve_l280_280383


namespace connie_total_markers_l280_280061

theorem connie_total_markers :
  let red_markers := 5230
  let blue_markers := 4052
  let green_markers := 3180
  let purple_markers := 2763
  red_markers + blue_markers + green_markers + purple_markers = 15225 :=
by
  let red_markers := 5230
  let blue_markers := 4052
  let green_markers := 3180
  let purple_markers := 2763
  -- Proof would go here, but we use sorry to skip it for now
  sorry

end connie_total_markers_l280_280061


namespace two_digit_number_property_l280_280093

theorem two_digit_number_property : ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ (10 * a + b = 2 * a * b) ∧ (10 * a + b = 36) :=
by
  use [3, 6]
  sorry

end two_digit_number_property_l280_280093


namespace product_of_constants_t_l280_280132

theorem product_of_constants_t (a b : ℤ) (t : ℤ) (h1 : a * b = -12) (h2 : t = a + b) :
  ∃ (ts : Finset ℤ), (ts = {11, 4, 1, -1, -4, -11}) ∧ ts.prod (λ x, x) = -1936 :=
by sorry

end product_of_constants_t_l280_280132


namespace gcd_fact8_fact6_squared_l280_280117

-- Definition of 8! and (6!)²
def fact8 : ℕ := 8!
def fact6_squared : ℕ := (6!)^2

-- The theorem statement to be proved
theorem gcd_fact8_fact6_squared : Nat.gcd fact8 fact6_squared = 11520 := 
by
    sorry

end gcd_fact8_fact6_squared_l280_280117


namespace oldest_child_age_l280_280369

-- Definitions based on the conditions
def avg_age (a b c d : ℕ) : ℕ := (a + b + c + d) / 4

def three_younger_children_ages : List ℕ := [6, 8, 10]

-- Problem statement
theorem oldest_child_age :
  ∃ x : ℕ, avg_age 6 8 10 x = 9 ∧ x = 12 :=
by
  exists 12
  constructor
  { -- avg_age 6 8 10 12 = 9
    sorry }
  { -- x = 12
    refl }

end oldest_child_age_l280_280369


namespace one_minus_repeating_three_l280_280090

theorem one_minus_repeating_three : ∀ b : ℚ, b = 1 / 3 → 1 - b = 2 / 3 :=
by
  intro b hb
  rw [hb]
  norm_num

end one_minus_repeating_three_l280_280090


namespace republicans_voting_for_A_l280_280250

theorem republicans_voting_for_A (V : ℝ) :
    (∀ D R A, D = 0.6 * V ∧ R = 0.4 * V ∧ A = 0.53 * V ∧
    (0.75 * D + (R * ?unknown) = A)) → 
    ?unknown = 0.2 :=
by 
  intros D R A h
  cases h with hD h
  cases h with hR h
  cases h with hA h
  simp at h
  exact sorry

end republicans_voting_for_A_l280_280250


namespace inscribed_circle_radius_l280_280268

theorem inscribed_circle_radius (R r x : ℝ) (hR : R = 18) (hr : r = 9) :
  x = 8 :=
sorry

end inscribed_circle_radius_l280_280268


namespace min_value_abs_2x1_plus_x2_l280_280425

theorem min_value_abs_2x1_plus_x2 (f g : ℝ → ℝ)
  (h₁ : ∀ x, f x = 2 * sin x * cos x)
  (h₂ : ∀ x, g x = sin (2 * x + π / 6) + 1)
  (h₃ : ∃ x1 x2, f x1 * g x2 = 2):
  ∃ x1 x2, |2 * x1 + x2| = π / 3 :=
by
  sorry

end min_value_abs_2x1_plus_x2_l280_280425


namespace probability_sum_greater_than_four_l280_280464

def sum_greater_than_four_probability :=
  (5 / 6: ℚ)

theorem probability_sum_greater_than_four :
  let outcomes := { (a, b) | a in (Finset.range 1 7) ∧ b in (Finset.range 1 7) }
  let favorable_outcomes := outcomes.filter (λ pair, (pair.1 + pair.2) > 4)
  let probability := favorable_outcomes.card.to_rat / outcomes.card.to_rat
  probability = sum_greater_than_four_probability :=
by
  sorry

end probability_sum_greater_than_four_l280_280464


namespace part_a_part_b_part_c_l280_280009

-- Part (a)
theorem part_a (n : ℕ) (hn : n ≥ 1) : 
  (∑ k in finset.range n, (1 : ℝ) / (k + 1)^2) ≤ 2 - (1 : ℝ) / n :=
by sorry

-- Part (b)
theorem part_b (n : ℕ) (hn : n ≥ 2) : 
  (1 / (n^2 - n) - 1 / (n^2 + n)) > (2 / n^3) :=
by sorry

-- Part (c)
theorem part_c (n : ℕ) (hn : n ≥ 1) : 
  (∑ k in finset.range n, (1 : ℝ) / (k + 1)^3) ≤ 5 / 4 :=
by sorry

end part_a_part_b_part_c_l280_280009


namespace pyramid_sphere_surface_area_l280_280674

-- Define a right triangular rectangular pyramid with side length a
-- All sides are right triangles, and it fits into a cube of side length a

theorem pyramid_sphere_surface_area (a : ℝ) (P A B C : Type) 
  (h1: ∀ P A B, ∃ (x : ℝ), right_triangle P A B)
  (h2: ∀ P A C, ∃ (y : ℝ), right_triangle P A C)
  (h3: ∀ P B C, ∃ (z : ℝ), right_triangle P B C)
  (h4: ∀ A B C, ∃ (w : ℝ), right_triangle A B C) 
  (h5: ∀ P A B C, ∃ sphere, ∀ point ∈ {P, A, B, C}, point ∈ sphere) :
  sphere_surface_area a = 3 * pi * a ^ 2 := sorry

-- Added helper definitions
def right_triangle (P A B : Type) : Prop := sorry    -- Assumed to be a helper definition showing that the triangle is a right triangle
def sphere_surface_area (r : ℝ) : ℝ := 4 * pi * r ^ 2    -- Helper definition for the surface area of a sphere

end pyramid_sphere_surface_area_l280_280674


namespace profit_without_discount_l280_280033

theorem profit_without_discount (CP : ℝ) (SP_with_discount : ℝ) (discount_percent profit_percent : ℝ) :
  discount_percent = 5 → profit_percent = 23.5 →
  SP_with_discount = CP - (discount_percent / 100 * CP) →
  CP > 0 →
  let SP_without_discount := CP + (profit_percent / 100 * CP)
  in (SP_without_discount - CP) / CP * 100 = 23.5 :=
begin
  sorry
end

end profit_without_discount_l280_280033


namespace average_last_three_l280_280375

/-- The average of the last three numbers is 65, given that the average of six numbers is 60
  and the average of the first three numbers is 55. -/
theorem average_last_three (a b c d e f : ℝ) (h1 : (a + b + c + d + e + f) / 6 = 60) (h2 : (a + b + c) / 3 = 55) :
  (d + e + f) / 3 = 65 :=
by
  sorry

end average_last_three_l280_280375


namespace expected_lone_cars_is_one_l280_280558

def indicator_var (n k : ℕ) : ℚ :=
  if k < n then 1 / (k + 1) else 1 / n

def lone_car_expec (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ k, indicator_var (n) (k + 1))

theorem expected_lone_cars_is_one (n : ℕ) (hn : 1 ≤ n) : lone_car_expec n = 1 :=
by
  sorry

end expected_lone_cars_is_one_l280_280558


namespace complex_division_result_l280_280699

theorem complex_division_result : (3 + complex.i) / (1 - 3 * complex.i) = complex.i := by
  sorry

end complex_division_result_l280_280699


namespace solution_set_f_ge_1_l280_280326

noncomputable def f (x : ℝ) (a : ℝ) :=
  if x >= 0 then |x - 2| + a else -(|-x - 2| + a)

theorem solution_set_f_ge_1 {a : ℝ} (ha : a = -2) :
  {x : ℝ | f x a ≥ 1} = {x : ℝ | x ≤ -1 ∨ x ≥ 5} :=
by sorry

end solution_set_f_ge_1_l280_280326


namespace min_people_same_score_l280_280731

theorem min_people_same_score (participants : ℕ) (nA nB : ℕ) (pointsA pointsB : ℕ) (scores : Finset ℕ) :
  participants = 400 →
  nA = 8 →
  nB = 6 →
  pointsA = 4 →
  pointsB = 7 →
  scores.card = (nA + 1) * (nB + 1) - 6 →
  participants / scores.card < 8 :=
by
  intros h_participants h_nA h_nB h_pointsA h_pointsB h_scores_card
  sorry

end min_people_same_score_l280_280731


namespace PTA_money_left_l280_280365

theorem PTA_money_left (initial_savings : ℝ) (spent_on_supplies : ℝ) (spent_on_food : ℝ) :
  initial_savings = 400 →
  spent_on_supplies = initial_savings / 4 →
  spent_on_food = (initial_savings - spent_on_supplies) / 2 →
  (initial_savings - spent_on_supplies - spent_on_food) = 150 :=
by
  intro initial_savings_eq
  intro spent_on_supplies_eq
  intro spent_on_food_eq
  sorry

end PTA_money_left_l280_280365


namespace equidistant_point_is_incenter_l280_280672

theorem equidistant_point_is_incenter
  {Δ : Type*} [triangle Δ] 
  (P : Δ)
  (h1 : inside_triangle P)
  (h2 : ∀ (A B C : Δ), equidistant_from_sides P A B C) :
  is_incenter P Δ :=
by
  exact sorry

end equidistant_point_is_incenter_l280_280672


namespace treble_of_doubled_and_increased_l280_280027

theorem treble_of_doubled_and_increased (initial_number : ℕ) (result : ℕ) : 
  initial_number = 15 → (initial_number * 2 + 5) * 3 = result → result = 105 := 
by 
  intros h1 h2
  rw [h1] at h2
  linarith

end treble_of_doubled_and_increased_l280_280027


namespace length_of_train_l280_280008

-- Definitions for given conditions
def speed_kmh : ℝ := 58
def time_s : ℝ := 9
def speed_ms : ℝ := speed_kmh * 1000 / 3600 -- converting km/hr to m/s

-- Theorem statement to prove the length of the train
theorem length_of_train : (speed_ms * time_s).round = 145 := by
  let length := speed_ms * time_s
  have h_length : length ≈ 144.99 := by calc
    length = 58 * 1000 / 3600 * 9 : by simp
        _ = 58000 / 3600 * 9 : by simp
        _ ≈ 144.99 : by sorry -- calculation here approximated
  exact eq.symm (Int.round_to_nearest _ 144.99 145 h_length)

sorry

end length_of_train_l280_280008


namespace gcd_fact8_fact6_squared_l280_280119

-- Definition of 8! and (6!)²
def fact8 : ℕ := 8!
def fact6_squared : ℕ := (6!)^2

-- The theorem statement to be proved
theorem gcd_fact8_fact6_squared : Nat.gcd fact8 fact6_squared = 11520 := 
by
    sorry

end gcd_fact8_fact6_squared_l280_280119


namespace simplify_expression_l280_280606

variable (x y : ℝ)

theorem simplify_expression :
  (2 * x + 3 * y) ^ 2 - 2 * x * (2 * x - 3 * y) = 18 * x * y + 9 * y ^ 2 :=
by
  sorry

end simplify_expression_l280_280606


namespace f_neg1_plus_f_2_l280_280215

def f (x : Int) : Int :=
  if x = -3 then -1
  else if x = -2 then -5
  else if x = -1 then -2
  else if x = 0 then 0
  else if x = 1 then 2
  else if x = 2 then 1
  else if x = 3 then 4
  else 0  -- This handles x values not explicitly in the table, although technically unnecessary.

theorem f_neg1_plus_f_2 : f (-1) + f (2) = -1 := by
  sorry

end f_neg1_plus_f_2_l280_280215


namespace sum_of_integers_n_l280_280016

def f (x : ℤ) : ℤ := (x - 5) * (x - 12)
def g (x : ℤ) : ℤ := (x - 6) * (x - 10)

theorem sum_of_integers_n (h : ∀ n : ℤ, (f (g n)) / (f n)^2 ∈ ℤ) : 
  ∑ (n : ℤ) in { n | (f (g n) / (f n)^2 ∈ ℤ) ∧ 
                       f(g n) / (f n)^2 = some_integer }, n = 23 :=
by
  sorry

end sum_of_integers_n_l280_280016


namespace average_of_last_three_numbers_l280_280372

theorem average_of_last_three_numbers (a b c d e f : ℝ) 
  (h1 : (a + b + c + d + e + f) / 6 = 60) 
  (h2 : (a + b + c) / 3 = 55) : 
  (d + e + f) / 3 = 65 :=
sorry

end average_of_last_three_numbers_l280_280372


namespace sequence_terms_proof_l280_280733

theorem sequence_terms_proof (P Q R T U V W : ℤ) (S : ℤ) 
  (h1 : S = 10) 
  (h2 : P + Q + R + S = 40) 
  (h3 : Q + R + S + T = 40) 
  (h4 : R + S + T + U = 40) 
  (h5 : S + T + U + V = 40) 
  (h6 : T + U + V + W = 40) : 
  P + W = 40 := 
by 
  have h7 : P + Q + R + 10 = 40 := by rwa [h1] at h2
  have h8 : Q + R + 10 + T = 40 := by rwa [h1] at h3
  have h9 : R + 10 + T + U = 40 := by rwa [h1] at h4
  have h10 : 10 + T + U + V = 40 := by rwa [h1] at h5
  have h11 : T + U + V + W = 40 := h6
  sorry

end sequence_terms_proof_l280_280733


namespace inscribed_circle_radius_l280_280272

theorem inscribed_circle_radius (R r : ℝ) (hR : R = 18) (hr : r = 9) :
    ∃ x : ℝ, (∀ P Px OP O1P : ℝ, Px = sqrt((R - x) ^ 2 - x ^ 2) ∧ O1P = sqrt((r + x) ^ 2 - x ^ 2)
                 ∧ Px + r = O1P) ∧ x = 8 :=
begin
  sorry
end

end inscribed_circle_radius_l280_280272


namespace roots_of_quadratic_l280_280317

theorem roots_of_quadratic (m n : ℝ) (h₁ : m + n = -2) (h₂ : m * n = -2022) (h₃ : ∀ x, x^2 + 2 * x - 2022 = 0 → x = m ∨ x = n) :
  m^2 + 3 * m + n = 2020 :=
sorry

end roots_of_quadratic_l280_280317


namespace ab_sum_eq_2_l280_280769

theorem ab_sum_eq_2 (a b : ℝ) (M : Set ℝ) (N : Set ℝ) (f : ℝ → ℝ) 
  (hM : M = {b / a, 1})
  (hN : N = {a, 0})
  (hf : ∀ x ∈ M, f x ∈ N)
  (f_def : ∀ x, f x = 2 * x) :
  a + b = 2 :=
by
  -- proof goes here.
  sorry

end ab_sum_eq_2_l280_280769


namespace two_dice_sum_greater_than_four_l280_280441
open Classical

def probability_sum_greater_than_four : ℚ := by sorry

theorem two_dice_sum_greater_than_four :
  probability_sum_greater_than_four = 5 / 6 :=
sorry

end two_dice_sum_greater_than_four_l280_280441


namespace shaded_area_of_circumscribed_circles_l280_280613

theorem shaded_area_of_circumscribed_circles (r₁ r₂ : ℝ) (h₁ : r₁ = 3) (h₂ : r₂ = 4) :
  let R := r₁ + r₂
  let A := π * R^2
  let A₁ := π * r₁^2
  let A₂ := π * r₂^2
  A - A₁ - A₂ = 24 * π :=
by
  -- Define the radius of the large circle.
  let R := r₁ + r₂
  -- Define the areas of the large circle and the two smaller circles.
  let A := π * R^2
  let A₁ := π * r₁^2
  let A₂ := π * r₂^2
  -- Compute shaded area.
  have h₃ : R = 7, from calc
    R = r₁ + r₂ : rfl
    ... = 3 + 4 : by rw [h₁, h₂]
    ... = 7 : by norm_num,
  have h₄ : A = 49 * π, from calc
    A = π * R^2 : rfl
    ... = π * 7^2 : by rw [h₃]
    ... = 49 * π : by norm_num,
  have h₅ : A₁ = 9 * π, from calc
    A₁ = π * r₁^2 : rfl
    ... = π * 3^2 : by rw [h₁]
    ... = 9 * π : by norm_num,
  have h₆ : A₂ = 16 * π, from calc
    A₂ = π * r₂^2 : rfl
    ... = π * 4^2 : by rw [h₂]
    ... = 16 * π : by norm_num,
  calc
    A - A₁ - A₂
      = 49 * π - 9 * π - 16 * π : by rw [h₄, h₅, h₆]
      ... = 24 * π : by norm_num

end shaded_area_of_circumscribed_circles_l280_280613


namespace peanut_butter_cookies_Jenny_brought_l280_280755

variables (J : ℕ)

theorem peanut_butter_cookies_Jenny_brought :
  (J + 30 = 50 + 20) → J = 40 :=
by {
  intro h,
  have : J + 30 = 70,
  { rw h },
  linarith,
}

end peanut_butter_cookies_Jenny_brought_l280_280755


namespace mark_weekly_reading_time_l280_280783

-- Define the conditions
def hours_per_day : ℕ := 2
def days_per_week : ℕ := 7
def additional_hours : ℕ := 4

-- State the main theorem to prove
theorem mark_weekly_reading_time : (hours_per_day * days_per_week) + additional_hours = 18 := 
by
  -- The proof steps are omitted as per instructions
  sorry

end mark_weekly_reading_time_l280_280783
