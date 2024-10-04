import Mathlib
import Mathlib.Algebra.Divisibility
import Mathlib.Algebra.GeomSum
import Mathlib.Algebra.Lcm
import Mathlib.Algebra.Polynomial
import Mathlib.Algebra.Ring.Basic
import Mathlib.Algebra.Trigonometry
import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.Calculus.Limits
import Mathlib.Analysis.Geometry.Ellipsoid
import Mathlib.Analysis.Integral
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.SimpleGraph.Connectivity
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Gcd.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Interval
import Mathlib.NumberTheory.Basic
import Mathlib.Probability.Independence
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Real

namespace rooks_symmetric_arrangement_l386_386019

theorem rooks_symmetric_arrangement : 
  let n := 8
  in let diagonal_combinations := (λ m : ℕ, Nat.choose 8 (2 * m)) 
  in let non_diagonal_combinations := (λ (pairs : ℕ), Nat.choose 28 pairs)
  in let cases := [
        diagonal_combinations 0 * non_diagonal_combinations 4,
        diagonal_combinations 1 * non_diagonal_combinations 3,
        diagonal_combinations 2 * non_diagonal_combinations 2,
        diagonal_combinations 3 * non_diagonal_combinations 1,
        1
       ]
  in cases.sum = 139448 :=
by
  sorry

end rooks_symmetric_arrangement_l386_386019


namespace probability_S_eq_L_plus_1_l386_386958

/-- Define the binomial distribution for n trials and probability p -/
def binomial (n : ℕ) (p : ℚ) : ℕ → ℚ
| 0       := if h : n = 0 then 1 else 0
| (k + 1) := if h : n = 0 then 0 else (p * binomial (n - 1) p k + (1 - p) * binomial (n - 1) p (k + 1))

noncomputable def prob_S_eq_L_plus_1 : ℚ :=
  let S := binomial 2015 (1/2)
  let L := binomial 2015 (1/2)
  (nat.choose 4030 2016) / (2^4030)

theorem probability_S_eq_L_plus_1 :
  prob_S_eq_L_plus_1 = (nat.choose 4030 2016) / (2^4030) :=
sorry

end probability_S_eq_L_plus_1_l386_386958


namespace ascorbic_acid_molecular_weight_l386_386919

theorem ascorbic_acid_molecular_weight (C H O : ℕ → ℝ)
  (C_weight : C 6 = 6 * 12.01)
  (H_weight : H 8 = 8 * 1.008)
  (O_weight : O 6 = 6 * 16.00)
  (total_mass_given : 528 = 6 * 12.01 + 8 * 1.008 + 6 * 16.00) :
  6 * 12.01 + 8 * 1.008 + 6 * 16.00 = 176.124 := 
by 
  sorry

end ascorbic_acid_molecular_weight_l386_386919


namespace area_to_paint_l386_386608

def height_of_wall : ℝ := 10
def length_of_wall : ℝ := 15
def window_height : ℝ := 3
def window_length : ℝ := 3
def door_height : ℝ := 1
def door_length : ℝ := 7

theorem area_to_paint : 
  let total_wall_area := height_of_wall * length_of_wall
  let window_area := window_height * window_length
  let door_area := door_height * door_length
  let area_to_paint := total_wall_area - window_area - door_area
  area_to_paint = 134 := 
by 
  sorry

end area_to_paint_l386_386608


namespace neither_snow_nor_foggy_probability_l386_386478

theorem neither_snow_nor_foggy_probability :
  (let P_snow := 1 / 4
   let P_not_snow := 1 - P_snow
   let P_foggy_given_not_snow := 1 / 3
   let P_not_foggy_given_not_snow := 1 - P_foggy_given_not_snow
   in P_not_snow * P_not_foggy_given_not_snow = 1 / 2) :=
by
  sorry

end neither_snow_nor_foggy_probability_l386_386478


namespace thirtieth_term_of_sequence_l386_386480

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |> List.sum

def is_valid_term (n : ℕ) : Prop :=
  (n % 3 = 0) ∧ (sum_of_digits n = 10)

def sequence_of_valid_terms : List ℕ :=
  List.filter is_valid_term (List.range' 1 (3 * 100))

theorem thirtieth_term_of_sequence :
  sequence_of_valid_terms.get? 29 = some 282 :=
by
  sorry

end thirtieth_term_of_sequence_l386_386480


namespace stratified_sampling_l386_386170

theorem stratified_sampling (total_elderly total_middle_aged total_young sample_size : ℕ)
  (h_total : total_elderly = 28)
  (h_middle : total_middle_aged = 54)
  (h_young : total_young = 81)
  (h_sample : sample_size = 36):
  let total_population := total_elderly + total_middle_aged + total_young
  in (total_elderly * sample_size / total_population = 6) ∧ 
     (total_middle_aged * sample_size / total_population = 12) ∧ 
     (total_young * sample_size / total_population = 18) :=
by 
  sorry

end stratified_sampling_l386_386170


namespace sarah_toads_l386_386906

theorem sarah_toads (tim_toads : ℕ) (jim_toads : ℕ) (sarah_toads : ℕ)
  (h1 : tim_toads = 30)
  (h2 : jim_toads = tim_toads + 20)
  (h3 : sarah_toads = 2 * jim_toads) :
  sarah_toads = 100 :=
by
  sorry

end sarah_toads_l386_386906


namespace sum_of_first_3n_terms_l386_386128

theorem sum_of_first_3n_terms (n : ℕ) (sn s2n s3n : ℕ) 
  (h1 : sn = 48) (h2 : s2n = 60)
  (h3 : s2n - sn = s3n - s2n) (h4 : 2 * (s2n - sn) = sn + (s3n - s2n)) :
  s3n = 36 := 
by {
  sorry
}

end sum_of_first_3n_terms_l386_386128


namespace cubic_subtraction_l386_386930

theorem cubic_subtraction : 3 - (3^2)⁻³ = 2186 / 729 := by
  sorry

end cubic_subtraction_l386_386930


namespace min_value_expression_l386_386258

theorem min_value_expression :
  ∃ x > 0, x^2 + 6 * x + 100 / x^3 = 3 * (50:ℝ)^(2/5) + 6 * (50:ℝ)^(1/5) :=
by
  sorry

end min_value_expression_l386_386258


namespace all_possible_values_of_m_l386_386388

noncomputable def possible_values_of_m (a b d : Vector ℝ 3) (m : ℝ) : Prop :=
  ∥a∥ = 1 ∧ ∥b∥ = 1 ∧ ∥d∥ = 1 ∧ 
  a ⋅ b = 0 ∧ a ⋅ d = 0 ∧ 
  ∃ θ: ℝ, θ = Real.pi / 3 ∧ 
  m * ∥b × d∥ = 1 ∧
  a = m * (b × d) → m = 2 * Real.sqrt 3 / 3 ∨ m = - (2 * Real.sqrt 3 / 3)

theorem all_possible_values_of_m (a b d : Vector ℝ 3) (m : ℝ) :
  possible_values_of_m a b d m :=
sorry

end all_possible_values_of_m_l386_386388


namespace BQ_QD_ratio_l386_386588

variables (A B C D M N P Q : Type) [PlaneGeometry A]
  (m n p : ℝ) (h_AM_MB : AM / MB = m) (h_AN_NC : AN / NC = n) (h_DP_PC : DP / PC = p)

theorem BQ_QD_ratio :
  ∃ (AM MB AN NC DP PC BQ QD : ℝ),
  AM / MB = m ∧ AN / NC = n ∧ DP / PC = p → (BQ / QD = n / (p * m)) :=
begin
  sorry
end

end BQ_QD_ratio_l386_386588


namespace Adam_final_amount_l386_386984

def initial_amount : ℝ := 5.25
def spent_on_game : ℝ := 2.30
def spent_on_snacks : ℝ := 1.75
def found_dollar : ℝ := 1.00
def allowance : ℝ := 5.50

theorem Adam_final_amount :
  (initial_amount - spent_on_game - spent_on_snacks + found_dollar + allowance) = 7.70 :=
by
  sorry

end Adam_final_amount_l386_386984


namespace tetrahedral_ratios_l386_386412

-- Lean definition of the problem conditions and proof requirement

structure Point (α : Type*) := (x y z : α)

structure Tetrahedron (α : Type*) :=
( A B C D : Point α)

def Segment {α : Type*} (p1 p2 : Point α) := p1 ≠ p2

variable {α : Type*} [LinearOrderedField α]

def parallel (seg1 seg2 : Segment) : Prop := sorry -- placeholder for parallel definition

def ratio (p1 p2 : Point α) (q1 q2 : Point α) : α := sorry -- placeholder for ratio definition

theorem tetrahedral_ratios (A B C D O A1 B1 C1 : Point α)
(h1 : O ∈ {face ABC })
(h2 : parallel (Segment O A1) (Segment D A))
(h3 : parallel (Segment O B1) (Segment D B))
(h4 : parallel (Segment O C1) (Segment D C))
:
ratio O A1 D A + ratio O B1 D B + ratio O C1 D C = 1
:= 
sorry

end tetrahedral_ratios_l386_386412


namespace find_non_equivalent_fraction_l386_386150

-- Define the fractions mentioned in the problem
def sevenSixths := 7 / 6
def optionA := 14 / 12
def optionB := 1 + 1 / 6
def optionC := 1 + 5 / 30
def optionD := 1 + 2 / 6
def optionE := 1 + 14 / 42

-- The main problem statement
theorem find_non_equivalent_fraction :
  optionD ≠ sevenSixths := by
  -- We put a 'sorry' here because we are not required to provide the proof
  sorry

end find_non_equivalent_fraction_l386_386150


namespace second_set_length_is_correct_l386_386050

variables (first_set_length second_set_length : ℝ)

theorem second_set_length_is_correct 
  (h1 : first_set_length = 4)
  (h2 : second_set_length = 5 * first_set_length) : 
  second_set_length = 20 := 
by 
  sorry

end second_set_length_is_correct_l386_386050


namespace sum_of_valid_primes_l386_386211

-- Define the predicate for being a prime number
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the predicate for being a perfect square
def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

-- Define the condition that 100q + p is a perfect square given p is a 2-digit prime and q is a prime
def satisfies_conditions (p q : ℕ) : Prop :=
  10 ≤ p ∧ p < 100 ∧ is_prime p ∧ is_prime q ∧ is_perfect_square (100 * q + p)

-- The main problem statement
theorem sum_of_valid_primes :
  (∑ p in {p : ℕ | ∃ q, satisfies_conditions p q}, p) = 179 :=
sorry

end sum_of_valid_primes_l386_386211


namespace percentage_difference_y_less_than_z_l386_386975

-- Define the variables and the conditions
variables (x y z : ℝ)
variables (h₁ : x = 12 * y)
variables (h₂ : z = 1.2 * x)

-- Define the theorem statement
theorem percentage_difference_y_less_than_z (h₁ : x = 12 * y) (h₂ : z = 1.2 * x) :
  ((z - y) / z) * 100 = 93.06 := by
  sorry

end percentage_difference_y_less_than_z_l386_386975


namespace f_is_strictly_decreasing_on_g_range_on_l386_386740

noncomputable def f (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ :=
  sqrt 3 * sin (ω * x + φ) + 2 * sin ((ω * x + φ) / 2) ^ 2 - 1

theorem f_is_strictly_decreasing_on [-\pi/2, π/4] 
  {ω : ℝ} (hω : 0 < ω) {φ : ℝ} (hφ : 0 < φ ∧ φ < π) :
  ∀ x ∈ Icc (-π / 2) (π / 4), 
    strict_mono_decr_on (f x ω φ) :=
sorry

noncomputable def g (x : ℝ) : ℝ := 
  2 * sin (4 * x - π / 3)

theorem g_range_on [-π/12, π/6] : 
  ∀ x ∈ Icc (-π / 12) (π / 6), 
    g x ∈ Icc (-2 : ℝ) (sqrt 3) :=
sorry

end f_is_strictly_decreasing_on_g_range_on_l386_386740


namespace intersection_points_l386_386304

theorem intersection_points (m n : ℕ) : 
  ((2 ≤ m) ∧ (2 ≤ n)) → 
  (number_of_intersections m n = m * (m - 1) * n * (n - 1) / 4) :=
by
  sorry

end intersection_points_l386_386304


namespace net_gain_calculation_l386_386408

variable (selling_price : ℝ) (profit_percent : ℝ) (loss_percent : ℝ)

def cost_price (sp : ℝ) (percent : ℝ) := sp / (1 + percent/100)

def net_gain (sp : ℝ) (profit_perc : ℝ) (loss_perc : ℝ) := 
  let c1 := cost_price sp profit_perc
  let c2 := cost_price sp (-1 * loss_perc)
  (2 * sp) - (c1 + c2)

theorem net_gain_calculation (sp : ℝ) (profit_perc : ℝ) (loss_perc : ℝ) 
  (hsp: sp = 1.20) (hpp : profit_perc = 25) (hlp : loss_perc = 15) :
  net_gain sp profit_perc loss_perc = 0.028 :=
by
  rw [hsp, hpp, hlp]
  sorry

end net_gain_calculation_l386_386408


namespace structure_cube_count_l386_386117

theorem structure_cube_count :
  let middle_layer := 16
  let other_layers := 4 * 24
  middle_layer + other_layers = 112 :=
by
  let middle_layer := 16
  let other_layers := 4 * 24
  have h : middle_layer + other_layers = 112 := by
    sorry
  exact h

end structure_cube_count_l386_386117


namespace conditional_probability_of_B_given_A_l386_386699

noncomputable def P_B_given_A : ℚ :=
  let totalNumbers := 9
  let oddNumbers := 5
  let C_n_k (n k : ℕ) : ℕ := (nat.choose n k)
  let P_A := (C_n_k oddNumbers 1) / (C_n_k totalNumbers 1)
  let P_AB := (C_n_k oddNumbers 2) / (C_n_k totalNumbers 2)
  P_AB / P_A

theorem conditional_probability_of_B_given_A :
  P_B_given_A = 1 / 2 :=
by
  sorry

end conditional_probability_of_B_given_A_l386_386699


namespace pet_selection_combinations_l386_386587

theorem pet_selection_combinations 
  (puppies : ℕ) (kittens : ℕ) (hamsters : ℕ) (people : ℕ) 
  (h_puppies : puppies = 20)
  (h_kittens : kittens = 9)
  (h_hamsters : hamsters = 12)
  (h_people : people = 3) :
  (puppies * kittens * hamsters) * fintype.card (fin people)! = 12960 :=
by
  simp [h_puppies, h_kittens, h_hamsters, h_people, factorial]
  sorry

end pet_selection_combinations_l386_386587


namespace compare_neg_fractions_l386_386209

theorem compare_neg_fractions : (- (2 / 3) < - (1 / 2)) :=
sorry

end compare_neg_fractions_l386_386209


namespace arccos_cos_9_eq_2_717_l386_386619

-- Statement of the proof problem
theorem arccos_cos_9_eq_2_717 : Real.arccos (Real.cos 9) = 2.717 :=
by
  sorry

end arccos_cos_9_eq_2_717_l386_386619


namespace correct_group_l386_386673

def has_exactly_one_common_number (a b : list ℕ) : Prop :=
  (a.inter b).length = 1

theorem correct_group (card1 card2 card3 : list ℕ):
  card1 = [1, 4, 7] ∧ card2 = [2, 3, 4] ∧ card3 = [2, 5, 7] →
  has_exactly_one_common_number card1 card2 ∧
  has_exactly_one_common_number card1 card3 ∧
  has_exactly_one_common_number card2 card3 :=
by sorry

end correct_group_l386_386673


namespace part_I_part_II_l386_386746

noncomputable theory

def f (x m : ℝ) : ℝ := |x - 4 * m| + |x + 1 / m|

theorem part_I (m : ℝ) (hm : 0 < m) (x : ℝ) : f x m ≥ 4 :=
  sorry

theorem part_II (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 4) :
  1 / a + 4 / b ≥ 9 / 4 :=
  sorry

-- Note that we need proofs of part_I and part_II, but they are omitted here.

end part_I_part_II_l386_386746


namespace bob_start_time_l386_386550

-- Define constants for the problem conditions
def yolandaRate : ℝ := 3 -- Yolanda's walking rate in miles per hour
def bobRate : ℝ := 4 -- Bob's walking rate in miles per hour
def distanceXY : ℝ := 10 -- Distance between point X and Y in miles
def bobDistanceWhenMet : ℝ := 4 -- Distance Bob had walked when they met in miles

-- Define the theorem statement
theorem bob_start_time : 
  ∃ T : ℝ, (yolandaRate * T + bobDistanceWhenMet = distanceXY) →
  (T = 2) →
  ∃ tB : ℝ, T - tB = 1 :=
by
  -- Insert proof here
  sorry

end bob_start_time_l386_386550


namespace Megan_total_earnings_two_months_l386_386830

-- Define the conditions
def hours_per_day : ℕ := 8
def wage_per_hour : ℝ := 7.50
def days_per_month : ℕ := 20

-- Define the main question and correct answer
theorem Megan_total_earnings_two_months : 
  (2 * (days_per_month * (hours_per_day * wage_per_hour))) = 2400 := 
by
  -- In the problem statement, we are given conditions so we just state sorry because the focus is on the statement, not the solution steps.
  sorry

end Megan_total_earnings_two_months_l386_386830


namespace value_of_f_neg_log_5_7_l386_386722

noncomputable def f : ℝ → ℝ := sorry

theorem value_of_f_neg_log_5_7 (f_is_odd : ∀ x : ℝ, f (-x) = -f x)
  (f_for_nonneg : ∀ x : ℝ, 0 ≤ x → f x = 5^x + ( -1 : ℝ )) :
  f (-(Real.log 7 / Real.log 5)) = -6 :=
by
  sorry

end value_of_f_neg_log_5_7_l386_386722


namespace minimum_m_value_l386_386222

noncomputable def determinant (a1 a2 a3 a4 : ℝ) : ℝ :=
  a1 * a4 - a2 * a3

def f (x : ℝ) : ℝ :=
  determinant (-real.sin x) (real.cos x) 1 (-real.sqrt 3)

def translated_function (x m : ℝ) : ℝ :=
  2 * real.sin (x + m - real.pi / 6)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = - g (x)

theorem minimum_m_value (m : ℝ) (h_pos : m > 0) :
  is_odd_function (translated_function x m) ↔ m = real.pi / 6 := sorry

end minimum_m_value_l386_386222


namespace words_difference_l386_386407

-- Definitions based on conditions.
def right_hand_speed (words_per_minute : ℕ) := 10
def left_hand_speed (words_per_minute : ℕ) := 7
def time_duration (minutes : ℕ) := 5

-- Problem statement
theorem words_difference :
  let right_hand_words := right_hand_speed 0 * time_duration 0
  let left_hand_words := left_hand_speed 0 * time_duration 0
  (right_hand_words - left_hand_words) = 15 :=
by
  sorry

end words_difference_l386_386407


namespace friend_rental_fee_percentage_l386_386053

theorem friend_rental_fee_percentage :
  ∀ (camera_value rental_fee_per_week weeks john_paid : ℕ) 
  (rental_percentage : ℝ) 
  (h1 : camera_value = 5000)
  (h2 : rental_fee_per_week = (rental_percentage / 100) * camera_value)
  (h3 : rental_percentage = 10)
  (h4 : weeks = 4)
  (h5 : john_paid = 1200),
  let total_rental_fee := rental_fee_per_week * weeks in
  let friend_paid := total_rental_fee - john_paid in
  (friend_paid / total_rental_fee) * 100 = 40 :=
by
  intros camera_value rental_fee_per_week weeks john_paid rental_percentage h1 h2 h3 h4 h5
  let total_rental_fee := rental_fee_per_week * weeks
  let friend_paid := total_rental_fee - john_paid
  have : (friend_paid / total_rental_fee) * 100 = 40:= sorry
  exact this

end friend_rental_fee_percentage_l386_386053


namespace solve_modulo_problem_l386_386512

theorem solve_modulo_problem (n : ℤ) :
  0 ≤ n ∧ n < 19 ∧ 38574 % 19 = n % 19 → n = 4 := by
  sorry

end solve_modulo_problem_l386_386512


namespace identify_solutions_l386_386951

-- Definitions based on the conditions
def forms_precipitate (solution1 solution2 : Type) : Prop := sorry
def soluble_in_excess (sol : Type) (solution : Type) : Prop := sorry

-- Conditions transformed into Lean assertions
def condition1 (solutions : list Type) : Prop :=
  forms_precipitate (solutions.nth 0) (solutions.nth 2) ∧
  forms_precipitate (solutions.nth 0) (solutions.nth 4) ∧
  forms_precipitate (solutions.nth 0) (solutions.nth 3) ∧
  soluble_in_excess (solutions.nth 3) (solutions.nth 4) ∧
  soluble_in_excess (solutions.nth 4) (solutions.nth 4) ∧
  soluble_in_excess (solutions.nth 5) (solutions.nth 4) ∧
  soluble_in_excess (solutions.nth 6) (solutions.nth 4) ∧
  soluble_in_excess (solutions.nth 7) (solutions.nth 4)

def condition2 (solutions : list Type) : Prop :=
  forms_precipitate (solutions.nth 1) (solutions.nth 3) ∧
  forms_precipitate (solutions.nth 1) (solutions.nth 4) ∧
  soluble_in_excess (solutions.nth 3) (solutions.nth 4) ∧
  soluble_in_excess (solutions.nth 5) (solutions.nth 4) ∧
  soluble_in_excess (solutions.nth 6) (solutions.nth 4) ∧
  soluble_in_excess (solutions.nth 7) (solutions.nth 4)

def condition3 (solutions : list Type) : Prop :=
  forms_precipitate (solutions.nth 2) (solutions.nth 0) ∧
  forms_precipitate (solutions.nth 2) (solutions.nth 3) ∧
  forms_precipitate (solutions.nth 2) (solutions.nth 7) ∧
  soluble_in_excess (solutions.nth 4) (solutions.nth 3)

def condition4 (solutions : list Type) : Prop :=
  forms_precipitate (solutions.nth 3) (solutions.nth 1) ∧
  forms_precipitate (solutions.nth 3) (solutions.nth 4) ∧
  forms_precipitate (solutions.nth 3) (solutions.nth 6) ∧
  forms_precipitate (solutions.nth 3) (solutions.nth 0) ∧
  forms_precipitate (solutions.nth 3) (solutions.nth 7) ∧
  soluble_in_excess (solutions.nth 4) (solutions.nth 5)

theorem identify_solutions (solutions : list Type)
  (h1 : condition1 solutions)
  (h2 : condition2 solutions)
  (h3 : condition3 solutions)
  (h4 : condition4 solutions) :
  solutions.nth 0 = CuSO4 ∧
  solutions.nth 1 = CuCl2 ∧
  solutions.nth 2 = BaCl2 ∧
  solutions.nth 3 = AgNO3 ∧
  solutions.nth 4 = NH4OH ∧
  solutions.nth 5 = HNO3 ∧
  solutions.nth 6 = HCl ∧
  solutions.nth 7 = H2SO4 :=
begin
  sorry
end

end identify_solutions_l386_386951


namespace linear_equation_condition_l386_386870

theorem linear_equation_condition (a : ℝ) :
  (∃ x : ℝ, (a - 2) * x ^ (|a|⁻¹ + 3) = 0) ↔ a = -2 := 
by
  sorry

end linear_equation_condition_l386_386870


namespace members_in_both_sets_l386_386130

def U : Nat := 193
def B : Nat := 41
def not_A_or_B : Nat := 59
def A : Nat := 116

theorem members_in_both_sets
  (h1 : 193 = U)
  (h2 : 41 = B)
  (h3 : 59 = not_A_or_B)
  (h4 : 116 = A) :
  (U - not_A_or_B) = A + B - 23 :=
by
  sorry

end members_in_both_sets_l386_386130


namespace option_d_correct_l386_386294

noncomputable theory
open_locale classical

variables {α β γ : Type} [plane_space α] [plane_space β] [plane_space γ]
variables {a b : line_space} {P : point_space}

def are_non_coincident (a b : line_space) : Prop :=
  ¬ (a = b)

def is_perpendicular_to_plane (a : line_space) (α : plane_space) : Prop :=
  -- Definition of the perpendicular relation between a line and a plane
  sorry

def is_parallel_to (a b : line_space) : Prop :=
  -- Definition of the parallel relation between two lines
  sorry

def intersect_at (a b : line_space) (P : point_space) : Prop :=
  -- Definition of the intersection of two lines at a point
  sorry

theorem option_d_correct (α β γ : plane_space) (a b : line_space) (P : point_space)
  (hne_lines : are_non_coincident a b)
  (hne_planes_α : α ≠ β) (hne_planes_β : β ≠ γ) (hne_planes_γ : γ ≠ α)
  (a_perp_alpha : is_perpendicular_to_plane a α)
  (a_intersect_b_at_p : intersect_at a b P) :
  ¬ is_perpendicular_to_plane b α :=
sorry

end option_d_correct_l386_386294


namespace num_coprime_to_15_l386_386641

theorem num_coprime_to_15 : (filter (fun a => Nat.gcd a 15 = 1) (List.range 15)).length = 8 := by
  sorry

end num_coprime_to_15_l386_386641


namespace additional_charge_per_segment_l386_386798

theorem additional_charge_per_segment :
  ∀ (initial_fee total_charge distance : ℝ), 
    initial_fee = 2.35 →
    total_charge = 5.5 →
    distance = 3.6 →
    (total_charge - initial_fee) / (distance / (2 / 5)) = 0.35 :=
by
  intros initial_fee total_charge distance h_initial_fee h_total_charge h_distance
  sorry

end additional_charge_per_segment_l386_386798


namespace arithmetic_sequence_y_value_l386_386519

theorem arithmetic_sequence_y_value :
  ∃ y : ℤ, (∃ a1 a3 : ℤ, a1 = 9 ∧ a3 = 81 ∧ y = (a1 + a3) / 2) → y = 45 :=
by
  sorry

end arithmetic_sequence_y_value_l386_386519


namespace percentage_of_cost_for_overhead_l386_386479

def purchase_price : ℝ := 48
def net_profit : ℝ := 12
def markup : ℝ := 55

def total_selling_price : ℝ := purchase_price + markup
def cost_including_overhead : ℝ := total_selling_price - net_profit
def overhead_cost : ℝ := cost_including_overhead - purchase_price
def percentage_overhead_cost : ℝ := (overhead_cost / purchase_price) * 100

theorem percentage_of_cost_for_overhead :
  percentage_overhead_cost = 89.58 := by
  sorry

end percentage_of_cost_for_overhead_l386_386479


namespace find_angle_A_l386_386769

theorem find_angle_A (A B C : ℝ) (a b c : ℝ) :
  (a * Real.sin A = b * Real.sin B + (c - b) * Real.sin C)
  → (A = π / 3) :=
sorry

end find_angle_A_l386_386769


namespace intersection_line_l386_386208

-- Define the equations of the circles in Cartesian coordinates.
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 4 * x = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + y = 0

-- The theorem to prove.
theorem intersection_line (x y : ℝ) : circle1 x y ∧ circle2 x y → y + 4 * x = 0 :=
by
  sorry

end intersection_line_l386_386208


namespace sum_geometric_terms_l386_386027

noncomputable def geometric_sequence (a q : ℝ) (n : ℕ) : ℝ := a * q^n

theorem sum_geometric_terms (a q : ℝ) :
  a * (1 + q) = 3 → a * (1 + q) * q^2 = 6 → 
  a * (1 + q) * q^6 = 24 :=
by
  intros h1 h2
  -- Proof would go here
  sorry

end sum_geometric_terms_l386_386027


namespace initial_pencils_sold_l386_386583

noncomputable def price_equation_1 (CP : ℝ) (x : ℝ) : ℝ := 0.85 * x * CP
noncomputable def price_equation_2 (CP : ℝ) : ℝ := 1.15 * 7.391304347826086 * CP

theorem initial_pencils_sold (x : ℝ) (CP : ℝ) : 
  price_equation_1 CP x = 1 ∧ price_equation_2 CP = 1 → x = 10 :=
by {
  intro h,
  simp [price_equation_1, price_equation_2] at h,
  sorry
}

end initial_pencils_sold_l386_386583


namespace hexagon_and_circle_projection_l386_386903

-- Define a point in a two-dimensional plane
structure Point :=
(x : ℝ)
(y : ℝ)

-- Define the condition that three points do not lie on the same line
def non_collinear (A B C : Point) : Prop :=
¬(A.x = B.x ∧ B.x = C.x ∨ A.y = B.y ∧ B.y = C.y)

-- Define a regular hexagon
structure Hexagon :=
(vertices : Fin 6 → Point)
(horizontal_sides : ∀ i : Fin 6, (vertices i).y = (vertices (i + 1) % 6).y)
(equal_sides : ∀ i : Fin 6, dist (vertices i) (vertices (i + 1) % 6) = dist (vertices 0) (vertices 1))

-- Define the projection of vertices
def projections (A' B' C' : Point) (h : non_collinear A' B' C') : Fin 6 → Point := sorry

-- Define the construction of the hexagon from the projections
def construct_hexagon (A' B' C' : Point) (h : non_collinear A' B' C') : Hexagon := 
{ vertices := projections A' B' C' h,
  horizontal_sides := sorry,
  equal_sides := sorry }

-- Define a function for the inscribed circle
def inscribed_circle (H : Hexagon) : Prop := sorry -- Need realistic geometric construction

-- The main statement
theorem hexagon_and_circle_projection (A' B' C' : Point) (h : non_collinear A' B' C') : 
  ∃ H : Hexagon, inscribed_circle H :=
sorry

end hexagon_and_circle_projection_l386_386903


namespace probability_of_same_color_l386_386156

-- Definition of the conditions
def total_shoes : ℕ := 14
def pairs : ℕ := 7

-- The main statement
theorem probability_of_same_color :
  let total_ways := Nat.choose total_shoes 2 in
  let successful_ways := pairs in
  (successful_ways : ℚ) / total_ways = 1 / 13 :=
by
  sorry

end probability_of_same_color_l386_386156


namespace middle_term_arithmetic_sequence_l386_386539

-- Definitions of the given conditions
def a := 3^2
def c := 3^4

-- Assertion that y is the middle term of the arithmetic sequence a, y, c
theorem middle_term_arithmetic_sequence : 
  let y := (a + c) / 2 in 
  y = 45 :=
by
  -- Since the final proof steps are not needed
  sorry

end middle_term_arithmetic_sequence_l386_386539


namespace esperanzas_tax_ratio_l386_386411

theorem esperanzas_tax_ratio :
  let rent := 600
  let food_expenses := (3 / 5) * rent
  let mortgage_bill := 3 * food_expenses
  let savings := 2000
  let gross_salary := 4840
  let total_expenses := rent + food_expenses + mortgage_bill + savings
  let taxes := gross_salary - total_expenses
  (taxes / savings) = (2 / 5) := by
  sorry

end esperanzas_tax_ratio_l386_386411


namespace problem_solution_l386_386205

theorem problem_solution : (3127 - 2972) ^ 3 / 343 = 125 := by
  sorry

end problem_solution_l386_386205


namespace primary_school_total_students_l386_386346

theorem primary_school_total_students :
  (∀ g1 b1 g2 b2 : ℕ,
  ratio g1 b1 = 5 / 8 →
  ratio g2 b2 = 7 / 6 →
  g1 = 160 →
  let total_students_in_G1 := g1 + b1,
      percent_G1 := 0.65,
      total_students_in_school := total_students_in_G1 / percent_G1
  in total_students_in_school = 640) :=
sorry

end primary_school_total_students_l386_386346


namespace equal_opposite_edge_sums_l386_386874

noncomputable def tetrahedron_with_inscribed_circles (A B C D : Point) : Prop :=
  ∃ O : Point,
  (∃ FK FL AKF AFL : ℝ,
  FK = FL ∧
  (AKF = AFL ∧ AK = AL)) ∧
  (∃ AM AL BM BN CM CL DN DM : ℝ,
  AM = AL ∧ BM = BN ∧ CM = CL ∧ DN = DM)

theorem equal_opposite_edge_sums (A B C D : Point) (h : tetrahedron_with_inscribed_circles A B C D) :
  dist A B + dist C D = dist A C + dist B D :=
by sorry

end equal_opposite_edge_sums_l386_386874


namespace number_of_hyperbolas_l386_386814

-- Define the binomial coefficient
def C (n k : ℕ) : ℕ := Nat.choose n k

-- Define the problem conditions in Lean
def possible_values := {C 6 0, C 6 1, C 6 2, C 6 3, C 6 4, C 6 5, C 6 6}

theorem number_of_hyperbolas : 
  (finset.card (possible_values ×ˢ possible_values) = 16) :=
sorry

end number_of_hyperbolas_l386_386814


namespace probability_of_integer_division_l386_386111

open Set

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ (∀ m, m ∣ n → m = 1 ∨ m = n)

def within_range_r (r : ℤ) : Prop := -5 < r ∧ r < 8
def within_range_k (k : ℕ) : Prop := 1 < k ∧ k < 10

def primes_in_range (k : ℕ) : Prop := within_range_k k ∧ is_prime k

def valid_pairs : Finset (ℤ × ℕ) := 
  ((Finset.Ico (-4 : ℤ) 8).product (Finset.filter primes_in_range (Finset.range 10)))

def integer_division_pairs : Finset (ℤ × ℕ) :=
  valid_pairs.filter (λ p, p.1 % p.2 = 0)

noncomputable def probability : ℚ :=
  ⟨integer_division_pairs.card, valid_pairs.card⟩

theorem probability_of_integer_division : probability = 5 / 16 := 
  sorry

end probability_of_integer_division_l386_386111


namespace maximum_value_of_d_l386_386812

theorem maximum_value_of_d (a b c d : ℝ) 
  (h₁ : a + b + c + d = 10)
  (h₂ : ab + ac + ad + bc + bd + cd = 20) :
  d ≤ (5 + Real.sqrt 105) / 2 :=
sorry

end maximum_value_of_d_l386_386812


namespace arith_sqrt_abs_neg_nine_l386_386956

theorem arith_sqrt_abs_neg_nine : Real.sqrt (abs (-9)) = 3 := by
  sorry

end arith_sqrt_abs_neg_nine_l386_386956


namespace arithmetic_sequence_y_value_l386_386516

theorem arithmetic_sequence_y_value :
  ∃ y : ℤ, (∃ a1 a3 : ℤ, a1 = 9 ∧ a3 = 81 ∧ y = (a1 + a3) / 2) → y = 45 :=
by
  sorry

end arithmetic_sequence_y_value_l386_386516


namespace power_increased_by_four_l386_386598

-- Definitions from the conditions
variables (F k v : ℝ) (initial_force_eq_resistive : F = k * v)

-- Define the new conditions with double the force
variables (new_force : ℝ) (new_velocity : ℝ) (new_force_eq_resistive : new_force = k * new_velocity)
  (doubled_force : new_force = 2 * F)

-- The theorem statement
theorem power_increased_by_four (initial_force_eq_resistive : F = k * v) 
  (new_force_eq_resistive : new_force = k * new_velocity)
  (doubled_force : new_force = 2 * F) :
  new_velocity = 2 * v → 
  (new_force * new_velocity) = 4 * (F * v) :=
sorry

end power_increased_by_four_l386_386598


namespace right_angle_vertex_trajectory_l386_386292

theorem right_angle_vertex_trajectory (x y : ℝ) :
  let M := (-2, 0)
  let N := (2, 0)
  let P := (x, y)
  (∃ (x y : ℝ), (x + 2)^2 + y^2 + (x - 2)^2 + y^2 = 16) →
  x ≠ 2 ∧ x ≠ -2 →
  x^2 + y^2 = 4 :=
by
  intro h₁ h₂
  sorry

end right_angle_vertex_trajectory_l386_386292


namespace not_valid_mapping_circle_triangle_l386_386934

inductive Point
| mk : ℝ → ℝ → Point

inductive Circle
| mk : ℝ → ℝ → ℝ → Circle

inductive Triangle
| mk : Point → Point → Point → Triangle

open Point (mk)
open Circle (mk)
open Triangle (mk)

def valid_mapping (A B : Type) (f : A → B) := ∀ a₁ a₂ : A, f a₁ = f a₂ → a₁ = a₂

def inscribed_triangle_mapping (c : Circle) : Triangle := sorry -- map a circle to one of its inscribed triangles

theorem not_valid_mapping_circle_triangle :
  ¬ valid_mapping Circle Triangle inscribed_triangle_mapping :=
sorry

end not_valid_mapping_circle_triangle_l386_386934


namespace part_I_part_II_l386_386739

noncomputable def f (x : ℝ) : ℝ := 2 * |x + 1| + |x - 2|

theorem part_I : ∃ m, (∀ x, f x ≥ m) ∧ (∃ x, f x = m) ∧ m = 3 :=
by
  let m := 3
  have h1 : ∀ x, f x ≥ m := sorry
  have h2 : ∃ x, f x = m := sorry
  exact ⟨m, h1, h2, rfl⟩

theorem part_II (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 3) :
  (b^2 / a) + (c^2 / b) + (a^2 / c) ≥ 3 :=
by
  have := sorry
  exact this

end part_I_part_II_l386_386739


namespace two_zeros_implies_a_in_valid_range_a_as_def_with_x3_l386_386309

namespace LeanProof

def f (x a : ℝ) := log x - a * x 

variables {a x₁ x₂ x₃ : ℝ}

/- Problem 1 -/
theorem two_zeros_implies_a_in_valid_range
  (h_fx_zero : f x₁ a = 0 ∧ f x₂ a = 0)
  (h_x_order : x₁ < x₂)
  (h_monotonicity_change : ∃ c, c ∈ Ioo x₁ x₂ ∧ derivative (fun x => log x - a * x) c = 0) :
  0 < a ∧ a < 1 / real.exp 1 :=
sorry

/- Problem 2 -/
theorem a_as_def_with_x3
  (h_a : a = x₃ / real.exp x₃) 
  (h_x3_gt_1 : x₃ > 1)
  (h_x₂ : f x₂ a = 0) :
  real.exp x₃ = x₂ :=
sorry

end LeanProof

end two_zeros_implies_a_in_valid_range_a_as_def_with_x3_l386_386309


namespace jerky_fulfillment_time_l386_386915

-- Defining the conditions
def bags_per_batch : ℕ := 10
def order_quantity : ℕ := 60
def existing_bags : ℕ := 20
def batch_time_in_nights : ℕ := 1

-- Define the proposition to be proved
theorem jerky_fulfillment_time :
  let additional_bags := order_quantity - existing_bags in
  let batches_needed := additional_bags / bags_per_batch in
  let days_needed := batches_needed * batch_time_in_nights in
  days_needed = 4 :=
begin
  sorry
end

end jerky_fulfillment_time_l386_386915


namespace combined_function_period_2pi_l386_386920

noncomputable def sinusoidal_period : ℝ := 2 * Real.pi

noncomputable def secant_period : ℝ := 2 * Real.pi

noncomputable def combined_function (x : ℝ) : ℝ := Real.sin x + Real.sec x

theorem combined_function_period_2pi : 
  ∀ x, combined_function (x + (2 * Real.pi)) = combined_function x :=
by
  sorry

end combined_function_period_2pi_l386_386920


namespace total_revenue_is_980_l386_386949

noncomputable def ticket_price : ℝ := 20
noncomputable def first_discount : ℝ := 0.40
noncomputable def second_discount : ℝ := 0.15
noncomputable def total_people : ℕ := 56
noncomputable def first_group : ℕ := 10
noncomputable def second_group : ℕ := 20
noncomputable def remaining_group : ℕ := total_people - first_group - second_group

theorem total_revenue_is_980 (total_people = 56) : 
  (first_group * (ticket_price - first_discount * ticket_price) 
  + second_group * (ticket_price - second_discount * ticket_price) 
  + remaining_group * ticket_price) = 980 := by
  sorry

end total_revenue_is_980_l386_386949


namespace exists_non_intersecting_segment_l386_386594

-- Defining the rectangle structure and its center
structure Rectangle :=
  (center : ℝ × ℝ) -- Center of the rectangle

-- Statements based on given conditions and the required proof
theorem exists_non_intersecting_segment (square : Rectangle) 
  (rectangles : finset Rectangle) :
  ∃ (R R' : Rectangle), R ∈ rectangles ∧ R' ∈ rectangles ∧
  R ≠ R' ∧
  (∀ (x : ℝ × ℝ), x ∈ line_segment R.center R'.center → 
    (x = R.center ∨ x = R'.center)) :=
by
  sorry

end exists_non_intersecting_segment_l386_386594


namespace arithmetic_sequence_middle_term_l386_386528

theorem arithmetic_sequence_middle_term :
  let a1 := 3^2
  let a3 := 3^4
  let y := (a1 + a3) / 2
  y = 45 :=
by
  let a1 := (3:ℕ)^2
  let a3 := (3:ℕ)^4
  let y := (a1 + a3) / 2
  have : a1 = 9 := by norm_num
  have : a3 = 81 := by norm_num
  have : y = 45 := by norm_num
  exact this

end arithmetic_sequence_middle_term_l386_386528


namespace table_area_l386_386133

theorem table_area (A : ℝ) (runner_total : ℝ) (cover_percentage : ℝ) (double_layer : ℝ) (triple_layer : ℝ) :
  runner_total = 208 ∧
  cover_percentage = 0.80 ∧
  double_layer = 24 ∧
  triple_layer = 22 →
  A = 260 :=
by
  sorry

end table_area_l386_386133


namespace hyperbola_real_axis_length_l386_386310

theorem hyperbola_real_axis_length
    (a b : ℝ) 
    (h_pos_a : a > 0) 
    (h_pos_b : b > 0) 
    (h_eccentricity : a * Real.sqrt 5 = Real.sqrt (a^2 + b^2))
    (h_distance : b * a * Real.sqrt 5 / Real.sqrt (a^2 + b^2) = 8) :
    2 * a = 8 :=
sorry

end hyperbola_real_axis_length_l386_386310


namespace part1_beef_noodles_mix_sauce_purchased_l386_386167

theorem part1_beef_noodles_mix_sauce_purchased (x y : ℕ) (h1 : x + y = 170) (h2 : 15 * x + 20 * y = 3000) :
  x = 80 ∧ y = 90 :=
sorry

end part1_beef_noodles_mix_sauce_purchased_l386_386167


namespace polar_distance_l386_386785

/-
Problem:
In the polar coordinate system, it is known that A(2, π / 6), B(4, 5π / 6). Then, the distance between points A and B is 2√7.

Conditions:
- Point A in polar coordinates: A(2, π / 6)
- Point B in polar coordinates: B(4, 5π / 6)
-/

/-- The distance between two points in the polar coordinate system A(2, π / 6) and B(4, 5π / 6) is 2√7. -/
theorem polar_distance :
  let A_ρ := 2
  let A_θ := π / 6
  let B_ρ := 4
  let B_θ := 5 * π / 6
  let A_x := A_ρ * Real.cos A_θ
  let A_y := A_ρ * Real.sin A_θ
  let B_x := B_ρ * Real.cos B_θ
  let B_y := B_ρ * Real.sin B_θ
  let distance := Real.sqrt ((B_x - A_x)^2 + (B_y - A_y)^2)
  distance = 2 * Real.sqrt 7 := by
  sorry

end polar_distance_l386_386785


namespace det_matrix_l386_386238

theorem det_matrix : ∀ x : ℝ, 
  Matrix.det !![
    [x + 3, x,     x    ],
    [x,     x + 3, x    ],
    [x,     x,     x + 3]
  ] = 27 * x + 27 :=
by
  intro x
  sorry

end det_matrix_l386_386238


namespace negation_of_universal_proposition_l386_386876

theorem negation_of_universal_proposition (f : ℕ → ℕ) :
  (¬(∀ n : ℕ, f n ∈ ℕ ∧ f n > n)) ↔ (∃ n : ℕ, (f n ∉ ℕ) ∨ (f n ≤ n)) :=
by
sorry

end negation_of_universal_proposition_l386_386876


namespace correct_statement_l386_386377

-- Define the scores for the students
def scores_A := [79, 81]
def scores_B := [90, 70]

-- Calculate the average scores
def avg (scores : List ℚ) : ℚ :=
  (scores.foldl (· + ·) 0) / scores.length

-- Calculate the sample variance
def sample_variance (scores : List ℚ) : ℚ :=
  let mean := avg scores
  (scores.foldl (λ acc x, acc + (x - mean) ^ 2) 0) / (scores.length - 1)

theorem correct_statement : sample_variance scores_A < sample_variance scores_B :=
by
  sorry

end correct_statement_l386_386377


namespace middle_term_arithmetic_sequence_l386_386543

-- Definitions of the given conditions
def a := 3^2
def c := 3^4

-- Assertion that y is the middle term of the arithmetic sequence a, y, c
theorem middle_term_arithmetic_sequence : 
  let y := (a + c) / 2 in 
  y = 45 :=
by
  -- Since the final proof steps are not needed
  sorry

end middle_term_arithmetic_sequence_l386_386543


namespace ramu_profit_percent_l386_386422

-- Defining constants for the conditions given
def cost_of_car : ℝ := 42000
def cost_of_repairs : ℝ := 13000
def selling_price : ℝ := 61900

-- Calculate the total cost
def total_cost := cost_of_car + cost_of_repairs

-- Calculate the profit
def profit := selling_price - total_cost

-- Calculate the profit percent
def profit_percent := (profit / total_cost) * 100

-- Statement of the theorem
theorem ramu_profit_percent : round (profit_percent * 100) / 100 = 12.55 :=
by
  sorry

end ramu_profit_percent_l386_386422


namespace max_rectangles_in_M_l386_386215

-- Define the given conditions
def M : Set (ℝ × ℝ) := sorry  -- The set M consisting of 100 points in the Cartesian plane
axiom card_M : M.card = 100  -- The cardinality of M is 100

-- Proposition about the maximum number of rectangles
theorem max_rectangles_in_M (M : Set (ℝ × ℝ)) (hM : M.card = 100) :
  ∃ n, n ≤ 2025 ∧ (∀ rect, ∃ v1 v2 v3 v4, v1 ∈ M ∧ v2 ∈ M ∧ v3 ∈ M ∧ v4 ∈ M ∧ is_rectangle_parallel_to_axes rect (v1, v2, v3, v4) → true) :=
sorry

end max_rectangles_in_M_l386_386215


namespace max_sum_of_12th_powers_l386_386715

noncomputable def x_i (i : Fin 1997) : ℝ := sorry
def sigma_x : ℝ := ∑ i, x_i i
def sigma_x_12 : ℝ := ∑ i, (x_i i)^12

axiom x_bounds : ∀ i, -1/ℝ.sqrt 3 ≤ x_i i ∧ x_i i ≤ ℝ.sqrt 3
axiom x_sum : sigma_x = -318 * ℝ.sqrt 3

theorem max_sum_of_12th_powers : sigma_x_12 = 189548 := sorry

end max_sum_of_12th_powers_l386_386715


namespace rectangle_area_l386_386021

theorem rectangle_area (AB AC : ℝ) (hAB : AB = 15) (hAC : AC = 17) : 
  ∃ (CD : ℝ), (AB * CD = 120) :=
by
  have hBC : BC = sqrt (AC^2 - AB^2),
  sorry

end rectangle_area_l386_386021


namespace julia_probability_correct_guesses_l386_386054

theorem julia_probability_correct_guesses :
  let p_correct := 1 / 4
  let p_wrong := 3 / 4
  let total_prob := (binomial 5 2 * p_correct^2 * p_wrong^3) +
                    (binomial 5 3 * p_correct^3 * p_wrong^2) +
                    (binomial 5 4 * p_correct^4 * p_wrong^1) +
                    (binomial 5 5 * p_correct^5 * p_wrong^0)
  total_prob = 47 / 128 := by
  sorry

end julia_probability_correct_guesses_l386_386054


namespace stone_process_terminates_and_unique_final_config_l386_386838

-- Define the stone configuration and actions
structure StoneConfig :=
  (config : ℤ → ℕ)

structure ActionResult :=
  (newConfig : StoneConfig)

def action1 (n : ℤ) (c : StoneConfig) : Option ActionResult :=
  if c.config(n-1) > 0 ∧ c.config(n) > 0 then
    some { newConfig := 
            { config := λ k, 
              if k = n-1 then c.config(k) - 1 
              else if k = n then c.config(k) - 1 
              else if k = n+1 then c.config(k) + 1 
              else c.config(k) } }
  else none

def action2 (n : ℤ) (c : StoneConfig) : Option ActionResult :=
  if c.config(n) > 1 then
    some { newConfig := 
            { config := λ k,
              if k = n then c.config(k) - 2
              else if k = n+1 then c.config(k) + 1
              else if k = n-2 then c.config(k) + 1
              else c.config(k) } }
  else none

noncomputable def weight (α : ℝ) (c : StoneConfig) : ℝ :=
  ∑ i, (c.config i) * α^i

theorem stone_process_terminates_and_unique_final_config (init : StoneConfig) : 
  ∃ finalConfig : StoneConfig, (∀ n, action1 n finalConfig = none ∧ action2 n finalConfig = none) ∧ 
  (∀ seq1 seq2 : list ℤ, evalActions init seq1 = finalConfig ∧ evalActions init seq2 = finalConfig) := 
sorry

-- Note: evalActions is a placeholder function to evaluate the sequence of actions on the initial StoneConfig

end stone_process_terminates_and_unique_final_config_l386_386838


namespace ab_power_2023_l386_386325

theorem ab_power_2023 (a b : ℤ) (h : |a + 2| + (b - 1) ^ 2 = 0) : (a + b) ^ 2023 = -1 :=
by
  sorry

end ab_power_2023_l386_386325


namespace min_x2_plus_y2_l386_386763

theorem min_x2_plus_y2 (x y : ℝ) (h : (x + 5)^2 + (y - 12)^2 = 14^2) : x^2 + y^2 ≥ 1 :=
sorry

end min_x2_plus_y2_l386_386763


namespace marie_initial_erasers_l386_386076

def erasers_problem : Prop :=
  ∃ initial_erasers : ℝ, initial_erasers + 42.0 = 137

theorem marie_initial_erasers : erasers_problem :=
  sorry

end marie_initial_erasers_l386_386076


namespace ellipse_properties_l386_386711

-- Define the conditions of the problem
def condition1 (E : ℝ → ℝ → Prop) : Prop :=
  (E 2 1) ∧ (E (2 * Real.sqrt 2) 0) ∧ (∀ x y, E x y → (x^2 / 8 + y^2 / 2 = 1))

-- Define the statement to be proved
theorem ellipse_properties :
  ∃ E : ℝ → ℝ → Prop,
  condition1 E ∧
  (∀ (t : ℝ) (A B : ℝ × ℝ), 
    (A = (x, y)) ∧ (B = (x', y')) → y = (1 / 2) * x + t ∧
    y' = (1 / 2) * x' + t →
    let k1 := (y - 1) / (x - 2) in
    let k2 := (y' - 1) / (x' - 2) in
    k1 + k2 = 0) :=
sorry

end ellipse_properties_l386_386711


namespace find_p_l386_386313

-- Definition of the parabola
def parabola (p : ℝ) (x y : ℝ) := y^2 = 2 * p * x

-- Given conditions
variables (p : ℝ) (hp : p > 0) (y₀ : ℝ) (P : ℝ × ℝ) (hP : P = (2, y₀)) 
          (on_parabola : parabola p 2 y₀)
          (l : ℝ → ℝ) (hl : ∀ x, l x = (y₀ - (sqrt (2 * p) * (x - 2) / (2 * sqrt 2))))
          (F : ℝ × ℝ) (hF : F = (p / 2, p / 2))
          (m : ℝ → ℝ) (hm : ∀ x, m x = y₀)
          (M : ℝ × ℝ) (hM : M = ((2 + 5), y₀) → (M = (7, y₀)))
          (P_M_distance : real.sqrt ((7 - 2)^2 + (y₀ - y₀)^2) = 5)

-- We aim to show that the value of p is 6 given the above conditions.
theorem find_p (p : ℝ) (hp : p > 0) (y₀ : ℝ) (h₀ : y₀ = 2 * real.sqrt p):
  p = 6 := by
  sorry

end find_p_l386_386313


namespace power_function_passing_through_point_l386_386298

theorem power_function_passing_through_point (α : ℝ) (f : ℝ → ℝ) (h1 : f = λ x, x^α) (h2 : f 2 = 4) : f = λ x, x^2 :=
sorry

end power_function_passing_through_point_l386_386298


namespace find_slope_of_line_l_l386_386360

open Real

-- Define the curve C in polar coordinates
def curve_C_polar (ρ θ : ℝ) : Prop := ρ = 2 * cos θ

-- Rectangular coordinate system transformation
def curve_C_rectangular (x y : ℝ) : Prop := x^2 + y^2 - 2 * x = 0

-- Line l passing through point A(3,0) with slope k
def line_l (k x y : ℝ) : Prop := y = k * (x - 3)

-- Intersection condition
def intersects_exactly_once (k : ℝ) : Prop :=
  ((0 : ℤ) < 1 ∧ k = 1 ) → k = 3

theorem find_slope_of_line_l (k : ℝ) :
  (∀ x y : ℝ, (x = 3) ∨ (line_l k x y → curve_C_rectangular x y)) → 
  intersects_exactly_once k ↔ (k = sqrt(3)/3 ∨ k = -sqrt(3)/3) :=
by sorry

end find_slope_of_line_l_l386_386360


namespace ratio_AD_div_BC_eq_1_point_5_l386_386356

theorem ratio_AD_div_BC_eq_1_point_5 
  (A B C D : Type) 
  [is_equilateral_triangle A B C]
  [is_right_triangle B C D]
  (h_angle_BCD : angle A B C D = π / 2)
  (h_equal_sides_BC : BC.length = 2 * BD.length) : 
  (AD.length / BC.length) = 1.5 := 
sorry

end ratio_AD_div_BC_eq_1_point_5_l386_386356


namespace problem_statement_l386_386847

noncomputable def regular_14_gon (R : ℝ) :=
  (vertices : list (ℝ × ℝ)) -- A list of coordinates for the 14 vertices
  (circumradius : ℝ = R) -- The circumradius of the regular polygon is R
  (is_regular_polygon : ∀ i j, vertices.nth i ≠ none ∧ vertices.nth j ≠ none →
    dist (vertices.nth i).get_or_else (0,0) (vertices.nth j).get_or_else (0,0) = R*sin(π/7))

theorem problem_statement (R : ℝ) (hexadecagon : regular_14_gon R) :
  dist (hexadecagon.vertices.nth 0).get_or_else (0,0) (hexadecagon.vertices.nth 1).get_or_else (0,0) -
  dist (hexadecagon.vertices.nth 0).get_or_else (0,0) (hexadecagon.vertices.nth 3).get_or_else (0,0) +
  dist (hexadecagon.vertices.nth 0).get_or_else (0,0) (hexadecagon.vertices.nth 5).get_or_else (0,0) = R :=
by
  sorry

end problem_statement_l386_386847


namespace quartic_to_quadratic_l386_386736

-- Defining the statement of the problem
theorem quartic_to_quadratic (a b c x : ℝ) (y : ℝ) :
  a * x^4 + b * x^3 + c * x^2 + b * x + a = 0 →
  y = x + 1 / x →
  ∃ y1 y2, (a * y^2 + b * y + (c - 2 * a) = 0) ∧
           (x^2 - y1 * x + 1 = 0 ∨ x^2 - y2 * x + 1 = 0) :=
by
  sorry

end quartic_to_quadratic_l386_386736


namespace sum_of_coefficients_l386_386129

theorem sum_of_coefficients : 
  (polynomial.eval (1 : ℤ) (polynomial.eval₂ (fun _ => 1) 1 (1 : polynomial (polynomial ℤ)) (₂x - 3y)^20 : polynomial ℤ)) = 1 :=
sorry

end sum_of_coefficients_l386_386129


namespace jack_jog_speed_l386_386038

theorem jack_jog_speed (melt_time_minutes : ℕ) (distance_blocks : ℕ) (block_length_miles : ℚ) 
    (h_melt_time : melt_time_minutes = 10)
    (h_distance : distance_blocks = 16)
    (h_block_length : block_length_miles = 1/8) :
    let time_hours := (melt_time_minutes : ℚ) / 60
    let distance_miles := (distance_blocks : ℚ) * block_length_miles
        12 = distance_miles / time_hours :=
by
  sorry

end jack_jog_speed_l386_386038


namespace min_distance_point_on_line_l386_386288

theorem min_distance_point_on_line (a b : ℝ) (h : 3 * a + 4 * b = 15) :
  ∃ c : ℝ, (∀ a b : ℝ, 3 * a + 4 * b = 15 → c ≤ sqrt (a^2 + b^2)) ∧ c = 3 :=
sorry

end min_distance_point_on_line_l386_386288


namespace rhombus_perimeter_l386_386466

-- Define the rhombus with diagonals of given lengths and prove the perimeter is 52 inches
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) (d1_pos : 0 < d1) (d2_pos : 0 < d2):
  let s := sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2) in 
  let perimeter := 4 * s in
  perimeter = 52 
  :=
by
  sorry  -- Proof is skipped

end rhombus_perimeter_l386_386466


namespace roxanne_change_l386_386095

variable (lemonade_cost sandwich_cost watermelon_cost chips_cost cookies_cost pretzels_cost salad_cost total_cost change : ℝ)

def calculate_change : Prop :=
  lemonade_cost = 4 * 2 ∧
  sandwich_cost = 3 * 2.5 ∧
  watermelon_cost = 2 * 1.25 ∧
  chips_cost = 1 * 1.75 ∧
  cookies_cost = 4 * 0.75 ∧
  pretzels_cost = 5 * 1 ∧
  salad_cost = 1 * 8 ∧
  total_cost = lemonade_cost + sandwich_cost + watermelon_cost + chips_cost + cookies_cost + pretzels_cost + salad_cost ∧
  change = 100 - total_cost ∧
  change = 63.75

theorem roxanne_change : calculate_change :=
sorry

end roxanne_change_l386_386095


namespace arithmetic_sequence_middle_term_l386_386531

theorem arithmetic_sequence_middle_term :
  let a1 := 3^2
  let a3 := 3^4
  let y := (a1 + a3) / 2
  y = 45 :=
by
  let a1 := (3:ℕ)^2
  let a3 := (3:ℕ)^4
  let y := (a1 + a3) / 2
  have : a1 = 9 := by norm_num
  have : a3 = 81 := by norm_num
  have : y = 45 := by norm_num
  exact this

end arithmetic_sequence_middle_term_l386_386531


namespace sum_abc_l386_386808

variable {a b c : ℝ}
variables (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0) (c_nonzero : c ≠ 0)
variables (h1 : a * b = 2 * (a + b)) (h2 : b * c = 3 * (b + c)) (h3 : c * a = 4 * (c + a))

theorem sum_abc (h1 : a * b = 2 * (a + b)) (h2 : b * c = 3 * (b + c)) (h3 : c * a = 4 * (c + a))
   (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0) (c_nonzero : c ≠ 0) :
   a + b + c = 1128 / 35 := 
sorry

end sum_abc_l386_386808


namespace parabola_x_coords_sum_l386_386581

noncomputable def parabola_focus_line_intersects (a : ℝ) : Prop :=
∀ (A B : ℝ × ℝ),
A ≠ B →
A.2^2 = 4 * A.1 →
B.2^2 = 4 * B.1 →
(A.2 - B.2) / (A.1 - B.1) = a →
A.1 + B.1 = a^2 + 2 * a + 3

theorem parabola_x_coords_sum (a : ℝ) :
  parabola_focus_line_intersects 1 2 :=
by 
  sorry

end parabola_x_coords_sum_l386_386581


namespace shirley_balloons_l386_386096

theorem shirley_balloons : 
  (initial_balloons : ℕ) (initial_dollars : ℝ) (discount : ℝ) (expected_price : ℝ) (sale_price : ℝ) (final_balloons : ℕ) :
  initial_balloons = 120 →
  initial_dollars = 10 →
  discount = 0.20 →
  expected_price = initial_dollars / initial_balloons →
  sale_price = (1 - discount) * expected_price →
  final_balloons = initial_dollars / sale_price →
  final_balloons = 150 :=
by
  intros initial_balloons initial_dollars discount expected_price sale_price final_balloons
  assume h1 h2 h3 h4 h5 h6
  have h7 : expected_price = 1 / 12, by sorry
  have h8 : sale_price = 1 / 15, by sorry
  have h9 : final_balloons = 150, by sorry
  exact h9

end shirley_balloons_l386_386096


namespace expand_polynomial_l386_386241

noncomputable def p (x : ℝ) : ℝ := 7 * x ^ 2 + 5
noncomputable def q (x : ℝ) : ℝ := 3 * x ^ 3 + 2 * x + 1

theorem expand_polynomial (x : ℝ) : 
  (p x) * (q x) = 21 * x ^ 5 + 29 * x ^ 3 + 7 * x ^ 2 + 10 * x + 5 := 
by sorry

end expand_polynomial_l386_386241


namespace girls_reading_fraction_l386_386171

definition fraction_of_girls_reading (G : ℚ) : Prop :=
  let girls := 12
  let boys := 10
  let total_students := girls + boys
  let boys_reading := (4 / 5) * boys
  let not_reading := 4
  let students_reading := total_students - not_reading
  let girls_reading := students_reading - boys_reading
  G = girls_reading / girls

theorem girls_reading_fraction : fraction_of_girls_reading (5 / 6) :=
  sorry

end girls_reading_fraction_l386_386171


namespace simplify_sqrt_neg_2_sq_l386_386432

theorem simplify_sqrt_neg_2_sq : ∃ x : ℝ, x = 2 ∧ sqrt ((-2 : ℝ)^2) = x :=
by 
  use 2
  sorry

end simplify_sqrt_neg_2_sq_l386_386432


namespace condition_implies_at_least_one_gt_one_l386_386390

theorem condition_implies_at_least_one_gt_one (a b : ℝ) :
  (a + b > 2 → (a > 1 ∨ b > 1)) ∧ ¬(a^2 + b^2 > 2 → (a > 1 ∨ b > 1)) :=
by
  sorry

end condition_implies_at_least_one_gt_one_l386_386390


namespace arithmetic_seq_middle_term_l386_386520

theorem arithmetic_seq_middle_term (a1 a3 y : ℤ) (h1 : a1 = 3^2) (h2 : a3 = 3^4)
    (h3 : y = (a1 + a3) / 2) : y = 45 :=
by
  rw [h1, h2] at h3
  simp at h3
  exact h3

end arithmetic_seq_middle_term_l386_386520


namespace log_base_4_of_64sqrt2_l386_386236

theorem log_base_4_of_64sqrt2 : log 4 (64 * sqrt 2) = 13 / 4 := by 
  sorry

end log_base_4_of_64sqrt2_l386_386236


namespace find_n_l386_386267

theorem find_n (n : ℕ) (h : n > 2016) (h_not_divisible : ¬ (1^n + 2^n + 3^n + 4^n) % 10 = 0) : n = 2020 :=
sorry

end find_n_l386_386267


namespace combination_2586_1_eq_2586_l386_386623

noncomputable def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

theorem combination_2586_1_eq_2586 : combination 2586 1 = 2586 := by
  sorry

end combination_2586_1_eq_2586_l386_386623


namespace maximum_profit_l386_386442

/-- 
Given:
- The fixed cost is 3000 (in thousand yuan).
- The revenue per hundred vehicles is 500 (in thousand yuan).
- The additional cost y is defined as follows:
  - y = 10*x^2 + 100*x for 0 < x < 40
  - y = 501*x + 10000/x - 4500 for x ≥ 40
  
Prove:
1. The profit S(x) (in thousand yuan) in 2020 is:
   - S(x) = -10*x^2 + 400*x - 3000 for 0 < x < 40
   - S(x) = 1500 - x - 10000/x for x ≥ 40
2. The production volume x (in hundreds of vehicles) to achieve the maximum profit is 100,
   and the maximum profit is 1300 (in thousand yuan).
-/
noncomputable def profit_function (x : ℝ) : ℝ :=
  if (0 < x ∧ x < 40) then
    -10 * x^2 + 400 * x - 3000
  else if (x ≥ 40) then
    1500 - x - 10000 / x
  else
    0 -- Undefined for other values, though our x will always be positive in our case

theorem maximum_profit : ∃ x : ℝ, 0 < x ∧ 
  (profit_function x = 1300 ∧ x = 100) ∧
  ∀ y, 0 < y → profit_function y ≤ 1300 :=
sorry

end maximum_profit_l386_386442


namespace num_coprime_to_15_l386_386639

theorem num_coprime_to_15 :
  {a : ℕ | a < 15 ∧ Nat.gcd 15 a = 1}.toFinset.card = 8 :=
by
  sorry

end num_coprime_to_15_l386_386639


namespace area_BOFG_l386_386199

-- Define the squares and given conditions
def largeSquareSideLength : ℝ := a
def smallSquareSideLength : ℝ := b
def shadedArea : ℝ := 124

-- Define the area relation between the larger and smaller squares
axiom area_relation (a b : ℝ) : a^2 - b^2 = shadedArea

-- Define the midpoint relation
axiom midpoint_O (a b : ℝ) : midpoint_O_is_between_CF

-- Prove the area of the quadrilateral BOGF
theorem area_BOFG (a b : ℝ) (h1: b < a) (h2: a^2 - b^2 = 124) (h3: midpoint_O_is_between_CF) : 
  area_BOGF = 31 := 
by 
  sorry

end area_BOFG_l386_386199


namespace river_width_l386_386593

theorem river_width (depth : ℝ) (flow_rate_kmph : ℝ) (volume_per_minute : ℝ) : depth = 5 → flow_rate_kmph = 2 → volume_per_minute = 5833.333333333333 → 
  (volume_per_minute / ((flow_rate_kmph * 1000 / 60) * depth) = 35) :=
by 
  intros h_depth h_flow_rate h_volume
  sorry

end river_width_l386_386593


namespace circumscribed_circle_radius_l386_386104

theorem circumscribed_circle_radius (a : ℝ) (AC PQ : ℝ) (hAC : AC = a) (hPQ : PQ = (6*a)/5) :
  ∃ R : ℝ, R = (5*a)/8 :=
by
  use (5*a)/8
  split
  · exact rfl
  sorry

end circumscribed_circle_radius_l386_386104


namespace num_coprime_to_15_l386_386644

theorem num_coprime_to_15 : (filter (fun a => Nat.gcd a 15 = 1) (List.range 15)).length = 8 := by
  sorry

end num_coprime_to_15_l386_386644


namespace megan_total_earnings_l386_386829

-- Define the constants
def work_hours_per_day := 8
def earnings_per_hour := 7.50
def work_days_per_month := 20

-- Define Megan's total earnings for two months
def total_earnings (work_hours_per_day : ℕ) (earnings_per_hour : ℝ) (work_days_per_month : ℕ) : ℝ :=
  2 * (work_hours_per_day * earnings_per_hour * work_days_per_month)

-- Prove that the total earnings for two months is $2400
theorem megan_total_earnings : total_earnings work_hours_per_day earnings_per_hour work_days_per_month = 2400 :=
by
  sorry

end megan_total_earnings_l386_386829


namespace diagonal_crosses_700_cubes_l386_386568

noncomputable def num_cubes_crossed (a b c : ℕ) : ℕ :=
  a + b + c - Nat.gcd a b - Nat.gcd b c - Nat.gcd c a + Nat.gcd (Nat.gcd a b) c

theorem diagonal_crosses_700_cubes :
  num_cubes_crossed 200 300 350 = 700 :=
sorry

end diagonal_crosses_700_cubes_l386_386568


namespace square_area_l386_386841

theorem square_area {x : ℝ} (h1 : sqrt (x^2 + (x/3)^2) = 20) : x^2 = 360 :=
by {
  have h2 : x^2 + (x/3)^2 = 400,
    from (real.mul_self_sqrt (x^2 + (x/3)^2)).symm.trans h1,

  have h3 : x^2 + x^2 / 9 = 400,
    by rwa [div_pow, one_pow, add_div, ←mul_div_assoc, mul_div_cancel' _ (by norm_num : (9 : ℝ) ≠ 0)],

  have h4 : (10/9) * x^2 = 400,
    by { rw [add_comm, add_mul, ← one_mul x^2] at h3, linarith },

  have h5 : x^2 = 400 * (9/10),
    by { symmetry, convert eq_div_iff_mul_eq.2 h4, rwa [div_self (by norm_num : 10 ≠ 0)], },

  exact h5,
  sorry
}

end square_area_l386_386841


namespace hajar_score_l386_386343

variables (F H : ℕ)

theorem hajar_score 
  (h1 : F - H = 21)
  (h2 : F + H = 69)
  (h3 : F > H) :
  H = 24 :=
sorry

end hajar_score_l386_386343


namespace technicians_count_l386_386867

-- Variables
variables (T R : ℕ)
-- Conditions from the problem
def avg_salary_all := 8000
def avg_salary_tech := 12000
def avg_salary_rest := 6000
def total_workers := 30
def total_salary := avg_salary_all * total_workers

-- Equations based on conditions
def eq1 : T + R = total_workers := sorry
def eq2 : avg_salary_tech * T + avg_salary_rest * R = total_salary := sorry

-- Proof statement (external conditions are reused for clarity)
theorem technicians_count : T = 10 :=
by sorry

end technicians_count_l386_386867


namespace parabola_vertex_a_value_l386_386690

theorem parabola_vertex_a_value :
  ∃ (a : ℝ), (∀ (x y : ℝ), y = a * (x - 3)^2 + 2 → (y = -48 ∧ x = -2) → a = -2) :=
by
  use -2
  intros x y h1 h2
  have h3 : y = -48 ∧ x = -2, from h2
  sorry

end parabola_vertex_a_value_l386_386690


namespace probability_negative_cosine_l386_386707

noncomputable def arithmetic_sequence : ℕ → ℝ :=
  λ n, (n * Real.pi) / 10

def S (n : ℕ) : ℝ :=
  n * arithmetic_sequence 1 + (n * (n - 1) / 2) * (arithmetic_sequence 2 - arithmetic_sequence 1)

theorem probability_negative_cosine :
  S 4 = Real.pi ∧ arithmetic_sequence 4 = 2 * arithmetic_sequence 2 →
  let negative_cosine_elements := (Finset.range 30).filter (λ n, Real.cos (arithmetic_sequence (n + 1)) < 0) in
  (negative_cosine_elements.card : ℝ) / 30 = 7 / 15 :=
begin
  sorry
end

end probability_negative_cosine_l386_386707


namespace Alice_paid_36_percent_of_SRP_l386_386114

variable (P : ℝ)

theorem Alice_paid_36_percent_of_SRP 
  (h1 : ∀ P, 0 < P)
  (h2 : ∀ MP, MP = 0.60 * P)
  (h3 : ∀ price_Alice_paid, price_Alice_paid = 0.60 * 0.60 * P) :
  ∀ percentage_of_SRP, percentage_of_SRP = (price_Alice_paid / P) * 100 ∧ percentage_of_SRP = 36 :=

by
  assume P P_pos h2 h3 percentage_of_SRP,
  have price_Alice_paid := 0.60 * 0.60 * P,
  rw h3 at price_Alice_paid,
  have percentage_of_SRP := (price_Alice_paid / P) * 100,
  rw [h3, mul_div_mul_left P P_pos] at percentage_of_SRP,
  simp at percentage_of_SRP,
  exact percentage_of_SRP,
  sorry

end Alice_paid_36_percent_of_SRP_l386_386114


namespace game_ends_at_22_rounds_l386_386439

def initial_tokens : X : ℕ := 17
def initial_tokens : Y : ℕ := 16
def initial_tokens : Z : ℕ := 15

def game_round (x y z : ℕ) : (ℕ × ℕ × ℕ) :=
    if x >= y ∧ x >= z then (x - 4, y + 1, z + 1)
    else if y >= x ∧ y >= z then (x + 1, y - 4, z + 1)
    else (x + 1, y + 1, z - 4)

def end_game_when_zero (x y z : ℕ) (rounds : ℕ) : ℕ :=
    if x = 0 ∨ y = 0 ∨ z = 0 then rounds
    else let (nx, ny, nz) := game_round x y z in
            end_game_when_zero nx ny nz (rounds + 1)

/- Defining the theorem that the game will end at 22 rounds given the initial conditions. -/
theorem game_ends_at_22_rounds (X Y Z : ℕ) (hX : X = 17) (hY : Y = 16) (hZ : Z = 15) :
  end_game_when_zero X Y Z 0 = 22 :=
by
  rw [hX, hY, hZ]
  sorry

end game_ends_at_22_rounds_l386_386439


namespace volume_of_solid_cross_section_y_eq_0_evaluate_definite_integral_l386_386767

-- Definitions used directly in conditions
def pointP := (1, 0, 1)
def pointQ := (-1, 1, 0)
def plane1 := λ x, x = 1
def plane2 := λ x, x = -1
def surface := λ P Q, sorry -- Rotation of PQ around x-axis, needs exact definition

-- Volume of the solid bounded by the surface and planes
theorem volume_of_solid (P Q : ℝ × ℝ × ℝ) (S : set (ℝ × ℝ × ℝ)) :
  (P = (1, 0, 1) ∧ Q = (-1, 1, 0) ∧
  S = {p | ∃ t ∈ Icc (0:ℝ) 1, p = (1 - 2 * t, t, 1 - t)}) →
  (∫ x in -1..1, π / 2 * (1 + x^2)) = 4 * π / 3 :=
by sorry

-- Cross-section of the solid in the plane y = 0
theorem cross_section_y_eq_0 (P Q : ℝ × ℝ × ℝ) :
  (P = (1, 0, 1) ∧ Q = (-1, 1, 0)) → 
  (∀ Q ∈ PQ, (let ⟨x, y, z⟩ := Q in (y = 0) → (Q = (1, 0, 1) ∨ Q = (-1, 0, 0)))) :=
by sorry

-- Evaluate the definite integral by substitution
theorem evaluate_definite_integral :
  (∫ t in 0..1, sqrt(t^2 + 1)) = (1/2 + (1/4) * sinh(2)) :=
by sorry

end volume_of_solid_cross_section_y_eq_0_evaluate_definite_integral_l386_386767


namespace sum_n_values_l386_386558

theorem sum_n_values
  (n : ℕ) :
  (∃ n ∈ ℕ,
    (∑ k in range(n + 1), cos (2 * π * k / 9)) = cos (π / 9)
    ∧ (log 2 n)^2 + 45 < log 2 (8 * n^13)) →
  n = 644 :=
by
  sorry

end sum_n_values_l386_386558


namespace sarah_toads_l386_386905

theorem sarah_toads (tim_toads : ℕ) (jim_toads : ℕ) (sarah_toads : ℕ)
  (h1 : tim_toads = 30)
  (h2 : jim_toads = tim_toads + 20)
  (h3 : sarah_toads = 2 * jim_toads) :
  sarah_toads = 100 :=
by
  sorry

end sarah_toads_l386_386905


namespace ratio_of_areas_DFP_EFP_l386_386630

-- Definitions of segments and points
variables (DEF: Type) [plane_segment DEF] 
variables (D E F P : DEF)

-- Given conditions
axiom DF_eq_39 : segment_length D F = 39
axiom EF_eq_42 : segment_length E F = 42
axiom FP_bisects_DFE : bisects_angle D F E P

-- The proof statement
theorem ratio_of_areas_DFP_EFP : 
  area_ratio (triangle D F P) (triangle E F P) = 13 / 14 :=
sorry

end ratio_of_areas_DFP_EFP_l386_386630


namespace not_equal_77_l386_386089

theorem not_equal_77 (x y : ℤ) : x^5 - 4*x^4*y - 5*y^2*x^3 + 20*y^3*x^2 + 4*y^4*x - 16*y^5 ≠ 77 := by
  sorry

end not_equal_77_l386_386089


namespace no_primitive_roots_modulo_m_l386_386419

theorem no_primitive_roots_modulo_m (m : ℕ) (h : ¬ (m = 2 ∨ m = 4 ∨ (∃ (p : ℕ) (α : ℕ), prime p ∧ odd p ∧ m = p^α) ∨ (∃ (p : ℕ) (α : ℕ), prime p ∧ odd p ∧ m = 2 * p^α))) : 
  ¬ ∃ (g : ℕ), ∀ (x : ℕ) (hx : coprime x m), ∃ (k : ℕ), g^k ≡ x [MOD m] :=
sorry

end no_primitive_roots_modulo_m_l386_386419


namespace final_value_of_S_is_10_l386_386852

-- Define the initial value of S
def initial_S : ℕ := 1

-- Define the sequence of I values
def I_values : List ℕ := [1, 3, 5]

-- Define the update operation on S
def update_S (S : ℕ) (I : ℕ) : ℕ := S + I

-- Final value of S after all updates
def final_S : ℕ := (I_values.foldl update_S initial_S)

-- The theorem stating that the final value of S is 10
theorem final_value_of_S_is_10 : final_S = 10 :=
by
  sorry

end final_value_of_S_is_10_l386_386852


namespace count_coprime_with_15_lt_15_l386_386647

theorem count_coprime_with_15_lt_15 :
  {a : ℕ // a < 15 ∧ Nat.coprime 15 a}.to_finset.card = 8 := 
sorry

end count_coprime_with_15_lt_15_l386_386647


namespace sphere_to_hemisphere_ratio_l386_386880

-- Definitions of the radii of the sphere and hemisphere
def r : ℝ := sorry -- We assume r is a positive real number, but not providing a specific value here

-- Volume of the sphere
def volume_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

-- Volume of the hemisphere with radius 3r
def volume_hemisphere (r : ℝ) : ℝ := (1 / 2) * volume_sphere (3 * r)

-- Ratio of volumes
noncomputable def ratio_volumes (r : ℝ) : ℝ := volume_sphere r / volume_hemisphere r

-- Statement to prove
theorem sphere_to_hemisphere_ratio : ratio_volumes r = 1 / 13.5 :=
by
  sorry

end sphere_to_hemisphere_ratio_l386_386880


namespace rhombus_perimeter_l386_386459

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) : 
  let a := d1 / 2 in
  let b := d2 / 2 in
  let c := Real.sqrt (a^2 + b^2) in
  let side := c in
  let perimeter := 4 * side in
  perimeter = 52 := 
by 
  let a := 5 in 
  let b := 12 in 
  have h3 : a = d1 / 2, by rw [h1]; norm_num
  have h4 : b = d2 / 2, by rw [h2]; norm_num
  let c := Real.sqrt (5^2 + 12^2),
  let side := c,
  have h5 : c = 13, by norm_num,
  let perimeter := 4 * 13,
  show perimeter = 52, by norm_num; sorry

end rhombus_perimeter_l386_386459


namespace roots_of_complex_polynomial_l386_386807

theorem roots_of_complex_polynomial (a b : ℝ) (h : (a + 3 * Complex.i, b + 6 * Complex.i).root_of (Complex.poly_of_2 (12 + 9 * Complex.i) (15 + 65 * Complex.i)) = true) :
  a = 7 / 3 ∧ b = 29 / 3 :=
by
  sorry

end roots_of_complex_polynomial_l386_386807


namespace jack_speed_to_beach_12_mph_l386_386044

theorem jack_speed_to_beach_12_mph :
  let distance := 16 * (1 / 8) -- distance in miles
  let time := 10 / 60        -- time in hours
  distance / time = 12 :=    -- speed in miles per hour
by
  let distance := 16 * (1 / 8) -- evaluation of distance
  let time := 10 / 60          -- evaluation of time
  show distance / time = 12    -- final speed calculation
  from sorry

end jack_speed_to_beach_12_mph_l386_386044


namespace greatest_area_is_inscribed_l386_386845

variable (a b c d : ℝ) -- Side lengths of the quadrilateral
variable (B D : ℝ) -- Opposite angles of the quadrilateral

theorem greatest_area_is_inscribed (h_fixed_sides : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) :
  (B + D = 180) → (∀ B' D', (B' + D' = 180) → area(a, b, c, d, B', D') ≤ area(a, b, c, d, B, D)) :=
sorry

end greatest_area_is_inscribed_l386_386845


namespace percentage_students_enrolled_in_bio_l386_386606

-- Problem statement
theorem percentage_students_enrolled_in_bio (total_students : ℕ) (students_not_in_bio : ℕ) 
    (h1 : total_students = 880) (h2 : students_not_in_bio = 462) : 
    ((total_students - students_not_in_bio : ℚ) / total_students) * 100 = 47.5 := by 
  -- Proof is omitted
  sorry

end percentage_students_enrolled_in_bio_l386_386606


namespace middle_term_arithmetic_sequence_l386_386538

-- Definitions of the given conditions
def a := 3^2
def c := 3^4

-- Assertion that y is the middle term of the arithmetic sequence a, y, c
theorem middle_term_arithmetic_sequence : 
  let y := (a + c) / 2 in 
  y = 45 :=
by
  -- Since the final proof steps are not needed
  sorry

end middle_term_arithmetic_sequence_l386_386538


namespace distance_incenter_circumcenter_l386_386191

-- Definitions for the conditions
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

def mid_point (x y z : ℝ) : ℝ :=
  (x + y) / 2

def incenter_to_circumcenter_distance (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  let K := (1/2) * a * b -- Right triangle area formula
  let r := K / s -- Radius of the inscribed circle
  let D := c / 2 -- Distance from the circumcenter to the hypotenuse
  let DO := D - (5 : ℝ) -- Given solved value for DH (HF distance in right triangle)
  sqrt (r^2 + (DO)^2)

-- Main statement to prove in Lean 4
theorem distance_incenter_circumcenter
  (a b c : ℝ) (h_triangle : is_right_triangle a b c)
  : incenter_to_circumcenter_distance a b c = sqrt(85) / 2 := 
  sorry

end distance_incenter_circumcenter_l386_386191


namespace power_function_analytical_expression_l386_386873

theorem power_function_analytical_expression (α : ℝ) (h : (4:ℝ) ^ α = (2:ℝ)) : ∀ x : ℝ, x ^ α = Real.sqrt x :=
by
  sorry

end power_function_analytical_expression_l386_386873


namespace ordered_pairs_subsets_three_elements_l386_386322

theorem ordered_pairs_subsets_three_elements :
  let n := 10
  let S := Finset ℕ
  let T := Finset ℕ
  let set := Finset.range 1 (n + 1)
  let pairs := {p : (S × T) | p.1 ∪ p.2 = 3}
  Finset.card pairs = 3240 := by sorry

end ordered_pairs_subsets_three_elements_l386_386322


namespace count_valid_numbers_l386_386321

def is_even (n : ℕ) : Prop := n % 2 = 0

def valid_digit_set : Set ℕ := {1, 3, 4, 5, 6, 8}

def distinct_digits (n : ℕ) : Prop :=
  let digits := (n / 100, (n / 10) % 10, n % 10)
  digits.1 ≠ digits.2 ∧ digits.1 ≠ digits.3 ∧ digits.2 ≠ digits.3

def is_valid_number (n : ℕ) : Prop :=
  300 ≤ n ∧ n < 800 ∧
  is_even n ∧
  (n / 100 ∈ valid_digit_set) ∧
  ((n / 10) % 10 ∈ valid_digit_set) ∧
  (n % 10 ∈ valid_digit_set) ∧
  distinct_digits n

theorem count_valid_numbers : 
  (Finset.filter is_valid_number (Finset.Ico 300 800)).card = 48 := sorry

end count_valid_numbers_l386_386321


namespace sphere_hemisphere_volume_ratio_l386_386884

theorem sphere_hemisphere_volume_ratio (r : ℝ) (π : ℝ) (hr : π ≠ 0): 
  let V_sphere := (4 / 3) * π * r^3,
      V_hemisphere := (1 / 2) * (4 / 3) * π * (3 * r)^3
  in V_sphere / V_hemisphere = 1 / 13.5 := 
by {
  let V_sphere := (4 / 3) * π * r^3,
      V_hemisphere := (1 / 2) * (4 / 3) * π * (3 * r)^3;
  have : V_hemisphere = (4 / 3) * π * (13.5 * r^3), {
    sorry
  },
  have ratio := V_sphere / V_hemisphere,
  rw this at ratio,
  simp [V_sphere, V_hemisphere],
  field_simp [hr],
  norm_num,
  rw ←mul_assoc,
  field_simp,
  norm_num,
}

end sphere_hemisphere_volume_ratio_l386_386884


namespace smallest_number_is_negative_three_l386_386602

theorem smallest_number_is_negative_three:
  ∀ (S : set ℤ), 
  S = {1, 0, -2, -3} → 
  ∃ m ∈ S, ∀ a ∈ S, m ≤ a :=
  by
  intros S hS
  use -3
  -- proof goes here
  sorry

end smallest_number_is_negative_three_l386_386602


namespace circle_intersection_through_midpoint_of_arc_l386_386446

variable {α : Type} [Nonempty α]

def midpoint_of_arc (A B O : α) : α := sorry -- This should be the construction of the midpoint of an arc on circle center O

theorem circle_intersection_through_midpoint_of_arc
    (A B L C D N P Q M O : α)
    (h1 : Circle (Triangle A B L).circumcircle = Circle (Triangle C D N).circumcircle)
    (h2 : P ∈ Circle (Triangle A B L).circumcircle ∧ P ∈ Circle (Triangle C D N).circumcircle)
    (h3 : Q ∈ Circle (Triangle A B L).circumcircle ∧ Q ∈ Circle (Triangle C D N).circumcircle)
    (hM : M = midpoint_of_arc A D O) :
    ∃ M, PQ_passes_through_M PQ M :=
sorry

end circle_intersection_through_midpoint_of_arc_l386_386446


namespace total_students_l386_386783

theorem total_students (f b : ℕ) (h1 : f = 8) (h2 : b = 6) :
  f + b - 1 = 13 :=
by
  rw [h1, h2]
  norm_num

end total_students_l386_386783


namespace num_coprime_to_15_l386_386646

theorem num_coprime_to_15 : (filter (fun a => Nat.gcd a 15 = 1) (List.range 15)).length = 8 := by
  sorry

end num_coprime_to_15_l386_386646


namespace relationship_x_x2_negx_l386_386290

theorem relationship_x_x2_negx (x : ℝ) (h : x^2 + x < 0) : x < x^2 ∧ x^2 < -x :=
by
  sorry

end relationship_x_x2_negx_l386_386290


namespace fib_sum_eq_2_pow_l386_386428

-- Definition of Fibonacci sequence
def fib : ℕ → ℕ
| 0       := 1
| 1       := 1
| (n + 2) := fib (n + 1) + fib n

theorem fib_sum_eq_2_pow (n : ℕ) : 
  fib n + fib (n - 1) + ∑ k in Finset.range (n - 1), fib k * 2^(n - 2 - k) = 2^n :=
sorry

end fib_sum_eq_2_pow_l386_386428


namespace rectangularCoordinateEquation_polarRadiusOfB_l386_386301

-- Problem 1: Rectangular coordinate equation
theorem rectangularCoordinateEquation (x y : ℝ) (ρ θ : ℝ) 
  (h : ρ^2 = 4*ρ*cos θ - 3) : (x - 2)^2 + y^2 = 1 :=
sorry

-- Problem 2: Polar radius of point B
theorem polarRadiusOfB (ρ θ : ℝ) (ρ2 : ℝ) 
  (hA : ρ^2 = 4*ρ*cos θ - 3) 
  (hB : 4*ρ2^2 = 8*ρ2*cos θ - 3) 
  (hOAOB : ρ2 = 2*ρ) : ρ = sqrt 6 / 2 :=
sorry

end rectangularCoordinateEquation_polarRadiusOfB_l386_386301


namespace f_always_positive_l386_386848

noncomputable def f (x : ℝ) : ℝ := x^8 - x^5 + x^2 - x + 1

theorem f_always_positive : ∀ x : ℝ, 0 < f x := by
  sorry

end f_always_positive_l386_386848


namespace num_coprime_to_15_l386_386640

theorem num_coprime_to_15 :
  {a : ℕ | a < 15 ∧ Nat.gcd 15 a = 1}.toFinset.card = 8 :=
by
  sorry

end num_coprime_to_15_l386_386640


namespace bromine_atoms_in_compound_l386_386576

theorem bromine_atoms_in_compound :
  ∀ (n : ℕ), 1 * 26.98 + n * 79.90 = 267 → n = 3 :=
by
  assume n,
  intro h,
  sorry

end bromine_atoms_in_compound_l386_386576


namespace coprime_count_15_l386_386663

theorem coprime_count_15 :
  {a : ℕ | a < 15 ∧ Nat.gcd 15 a = 1}.card = 8 :=
by
sorry

end coprime_count_15_l386_386663


namespace number_of_integers_between_2500_and_3200_with_distinct_increasing_digits_l386_386225

theorem number_of_integers_between_2500_and_3200_with_distinct_increasing_digits :
  ∃ n : ℕ, n = 12 ∧ ∀ x : ℕ, 2500 ≤ x ∧ x ≤ 3200 ∧ (finset.univ.filter (λ i : ℕ, 2500 ≤ i ∧ i ≤ 3200 ∧ (λ x, ∀ j k l m, x = j*1000 + k*100 + l*10 + m ∧ j < k ∧ k < l ∧ l < m ∧ j ≠ k ∧ k ≠ l ∧ l ≠ m ∧ j ≠ m))).card = 12 := sorry

end number_of_integers_between_2500_and_3200_with_distinct_increasing_digits_l386_386225


namespace correct_probability_statements_l386_386496

-- Definitions based on the conditions
def same_birthday_probability (days : ℕ) : ℚ := 1 / days
def lottery_probability (win_prob : ℚ) (tickets : ℕ) : bool := win_prob * tickets == 1
def fair_decision_method : bool := true
def weather_forecast_correctness (chance_of_rain : ℚ) : bool := true

-- Proven correct conditions
def correct_statements : list ℕ := [1, 3]

-- Theorem stating the correct statements
theorem correct_probability_statements 
  (days : ℕ) (win_prob : ℚ) (tickets : ℕ) (chance_of_rain : ℚ)
  (h1 : days = 365)
  (h2 : win_prob = (1 / 1000))
  (h3 : tickets = 1000)
  (h4 : chance_of_rain = 0.09) :
  same_birthday_probability days = 1 / 365 ∧
  ¬ lottery_probability win_prob tickets ∧
  fair_decision_method ∧
  ¬ weather_forecast_correctness chance_of_rain :=
by {
  sorry
}

end correct_probability_statements_l386_386496


namespace solution_set_inequality_l386_386125

theorem solution_set_inequality (x : ℝ) : 
  (x - 1) / (x + 2) > 0 ↔ x > 1 ∨ x < -2 :=
by {
  sorry -- proof omitted
}

end solution_set_inequality_l386_386125


namespace age_problem_l386_386553

-- Define the ages of a, b, and c
variables (a b c : ℕ)

-- State the conditions
theorem age_problem (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : a + b + c = 22) : b = 8 :=
by
  sorry

end age_problem_l386_386553


namespace find_a_l386_386216

def f (x : ℝ) : ℝ := 5 * x - 6
def g (x : ℝ) : ℝ := 2 * x + 1

theorem find_a : ∃ a : ℝ, f a + g a = 0 ∧ a = 5 / 7 :=
by
  sorry

end find_a_l386_386216


namespace largest_possible_d_l386_386809

theorem largest_possible_d (a b c d : ℝ) 
  (h1 : a + b + c + d = 10) 
  (h2 : ab + ac + ad + bc + bd + cd = 20) :
  d ≤ (5 + Real.sqrt 105) / 2 := 
sorry

end largest_possible_d_l386_386809


namespace ted_age_solution_l386_386997

theorem ted_age_solution (t s : ℝ) (h1 : t = 3 * s - 10) (h2 : t + s = 60) : t = 42.5 :=
by {
  sorry
}

end ted_age_solution_l386_386997


namespace henry_coffee_table_books_l386_386750

theorem henry_coffee_table_books
    (initial_books : ℕ)
    (boxes : ℕ)
    (books_per_box : ℕ)
    (room_books : ℕ)
    (kitchen_books : ℕ)
    (picked_up_books : ℕ)
    (remaining_books : ℕ) :
    initial_books = 99 →
    boxes = 3 →
    books_per_box = 15 →
    room_books = 21 →
    kitchen_books = 18 →
    picked_up_books = 12 →
    remaining_books = 23 →
    initial_books - (boxes * books_per_box + room_books + kitchen_books) - (remaining_books - picked_up_books) = 4 :=
by
  intros h_initial h_boxes h_books_per_box h_room_books h_kitchen_books h_picked_up_books h_remaining_books
  rw [h_initial, h_boxes, h_books_per_box, h_room_books, h_kitchen_books, h_picked_up_books, h_remaining_books]
  norm_num
  sorry

end henry_coffee_table_books_l386_386750


namespace shortest_side_l386_386420

/-- 
Prove that if the lengths of the sides of a triangle satisfy the inequality \( a^2 + b^2 > 5c^2 \), 
then \( c \) is the length of the shortest side.
-/
theorem shortest_side (a b c : ℝ) (h : a^2 + b^2 > 5 * c^2) (triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b) : c ≤ a ∧ c ≤ b :=
by {
  -- Proof will be provided here.
  sorry
}

end shortest_side_l386_386420


namespace marie_sales_revenue_l386_386055

theorem marie_sales_revenue : 
  let magazines_sold := 425
  let newspapers_sold := 275
  let price_per_magazine := 3.5
  let price_per_newspaper := 1.25
  let total_revenue_from_magazines := magazines_sold * price_per_magazine
  let total_revenue_from_newspapers := newspapers_sold * price_per_newspaper
  total_revenue_from_magazines + total_revenue_from_newspapers = 1831.25 :=
by
  have mag_revenue := magazines_sold * price_per_magazine
  have news_revenue := newspapers_sold * price_per_newspaper
  have total_revenue := mag_revenue + news_revenue
  exact (Eq.trans (Eq.symm (by norm_num1 at mag_revenue total_revenue)) rfl).subst sorry

end marie_sales_revenue_l386_386055


namespace abs_diff_gt_half_prob_l386_386853

noncomputable def probability_abs_diff_gt_half : ℝ :=
  ((1 / 4) * (1 / 8) + 
   (1 / 8) * (1 / 2) + 
   (1 / 8) * 1) * 2

theorem abs_diff_gt_half_prob : probability_abs_diff_gt_half = 5 / 16 := by 
  sorry

end abs_diff_gt_half_prob_l386_386853


namespace HJ_perpendicular_AM_l386_386817

variables {A B C H M Y Q J : Type} [euclidean_geometry A B C H M Y Q J]

def acute_triangle (ABC : triangle A B C) : Prop :=
  let α := (angle A B C), 
      β := (angle B C A), 
      γ := (angle C A B) in
  α < π / 2 ∧ β < π / 2 ∧ γ < π / 2

def orthocenter (H : Point) (ABC : triangle A B C) : Prop :=
  H = intersection (altitude A B C) (altitude B C A) (altitude C A B)

def midpoint (M : Point) (A B : Point) : Prop :=
  M = midpoint A B

def perpendicular (l m : Line) : Prop :=
  angle l m = π / 2

-- Given conditions in Lean 4 statement
def given_conditions (ABC : triangle A B C) (H M Y Q J : Point) : Prop :=
  acute_triangle ABC ∧
  orthocenter H ABC ∧
  midpoint M B C ∧
  (∃ (Y : Point), on AC Y ∧ perpendicular (line Y H) (line M H)) ∧
  (∃ (Q : Point), on BH Q ∧ perpendicular (line Q A) (line A M)) ∧
  (∃ (J : Point), second_intersection (line M Q) (circle M Y) J)

-- Main theorem to prove that HJ is perpendicular to AM
theorem HJ_perpendicular_AM (ABC : triangle A B C) (H M Y Q J : Point) :
  given_conditions ABC H M Y Q J →
  perpendicular (line H J) (line A M) :=
sorry

end HJ_perpendicular_AM_l386_386817


namespace additional_charge_per_segment_l386_386797

variable (initial_fee : ℝ := 2.35)
variable (total_charge : ℝ := 5.5)
variable (distance : ℝ := 3.6)
variable (segment_length : ℝ := (2/5 : ℝ))

theorem additional_charge_per_segment :
  let number_of_segments := distance / segment_length
  let charge_for_distance := total_charge - initial_fee
  let additional_charge_per_segment := charge_for_distance / number_of_segments
  additional_charge_per_segment = 0.35 :=
by
  sorry

end additional_charge_per_segment_l386_386797


namespace A_is_infinite_l386_386726

open Set

variable {f : ℝ → ℝ}

-- Condition 1: Function f is defined for all real numbers.
-- This is implicitly satisfied since f : ℝ → ℝ

-- Condition 2: Given functional inequality for f
axiom functional_inequality : ∀ x : ℝ, (f x) ^ 2 ≤ 2 * x ^ 2 * f (x / 2)

-- Condition 3: The set A is not empty
def A : Set ℝ := {a | f a > a ^ 2}

noncomputable def A_is_nonempty : Prop := ∃ a : ℝ, f a > a ^ 2

-- Question: Prove that A is an infinite set.
theorem A_is_infinite (h : A_is_nonempty) : Infinite A := by
  sorry

end A_is_infinite_l386_386726


namespace count_distinct_ways_to_sum_465_l386_386018

def valid_consecutive_sum_seq_count (S : ℕ) : ℕ :=
  ∑ k in finset.range (S + 1), 
    if k ≥ 3 ∧ (∃ n : ℕ, 2 * n * k + k * (k - 1) = 2 * S) then 1 else 0

theorem count_distinct_ways_to_sum_465 : 
  valid_consecutive_sum_seq_count 465 = 4 :=
  by sorry

end count_distinct_ways_to_sum_465_l386_386018


namespace correct_inequality_l386_386548

def a : ℚ := -4 / 5
def b : ℚ := -3 / 4

theorem correct_inequality : a < b := 
by {
  -- Proof here
  sorry
}

end correct_inequality_l386_386548


namespace eggs_left_in_box_l386_386898

theorem eggs_left_in_box (initial_eggs : ℕ) (taken_eggs : ℕ) (remaining_eggs : ℕ) : 
  initial_eggs = 47 → taken_eggs = 5 → remaining_eggs = initial_eggs - taken_eggs → remaining_eggs = 42 :=
by
  sorry

end eggs_left_in_box_l386_386898


namespace min_sum_of_distinct_nonzero_digits_l386_386350

def distinct_nonzero_digits (a b c d e f h i j : ℕ) : Prop :=
  (∀ x ∈ {a, b, c, d, e, f, h, i, j}, x ≠ 0) ∧ 
  (∀ x y ∈ {a, b, c, d, e, f, h, i, j}, x = y → x = y)

theorem min_sum_of_distinct_nonzero_digits (A B C D E F H I J : ℕ) :
  distinct_nonzero_digits A B C D E F H I J →
  (A * 100 + B * 10 + C) + (D * 100 + E * 10 + F) = H * 100 + I * 10 + J →
  H * 100 + I * 10 + J = 459 :=
sorry

end min_sum_of_distinct_nonzero_digits_l386_386350


namespace equal_areas_quadrilateral_l386_386286

variable (A B C D : Point)
variable (convex_ABCD : ConvexQuadrilateral A B C D)

theorem equal_areas_quadrilateral (O : Point) :
  in_quadrilateral O A B C D convex_ABCD →
  area O B C D = area O A B D →
  area O A B C = area O B C D ∧
  area O B C D = area O C D A ∧
  area O C D A = area O D A B :=
sorry

end equal_areas_quadrilateral_l386_386286


namespace company_food_purchase_1_l386_386164

theorem company_food_purchase_1 (x y : ℕ) (h1: x + y = 170) (h2: 15 * x + 20 * y = 3000) : 
  x = 80 ∧ y = 90 := by
  sorry

end company_food_purchase_1_l386_386164


namespace pyramid_volume_l386_386977

theorem pyramid_volume (s h : ℝ) 
  (h1 : (1 / 3) * s^2 * h = 60) : 
  let s' := 3 * s
      h' := 2 * h in 
  (1 / 3) * (s')^2 * h' = 1080 := 
by
  sorry

end pyramid_volume_l386_386977


namespace rhombus_perimeter_l386_386450

-- Let's define the lengths of the diagonals
def d1 := 10
def d2 := 24

-- Half of the lengths of the diagonals
def half_d1 := d1 / 2
def half_d2 := d2 / 2

-- The length of one side of the rhombus, using the Pythagorean theorem
def side_length := Real.sqrt (half_d1^2 + half_d2^2)

-- The perimeter of the rhombus is 4 times the side length
def perimeter := 4 * side_length

-- Now we state the theorem to prove the perimeter is 52 inches
theorem rhombus_perimeter : perimeter = 52 := 
by
  -- Here you would normally provide the proof steps, but we insert 'sorry'
  sorry

end rhombus_perimeter_l386_386450


namespace angle_bisectors_form_cyclic_quadrilateral_l386_386430

theorem angle_bisectors_form_cyclic_quadrilateral 
  {A B C D E F G H : Type} [IsQuadrilateral A B C D] 
  (hE : IsIntersectionPoint (Bisector ∠A) (Bisector ∠B))
  (hF : IsIntersectionPoint (Bisector ∠B) (Bisector ∠C))
  (hG : IsIntersectionPoint (Bisector ∠C) (Bisector ∠D))
  (hH : IsIntersectionPoint (Bisector ∠D) (Bisector ∠A)) :
  IsCyclicQuadrilateral E F G H :=
sorry

end angle_bisectors_form_cyclic_quadrilateral_l386_386430


namespace count_f_compositions_l386_386822

noncomputable def count_special_functions : Nat :=
  let A := Finset.range 6
  let f := (Set.univ : Set (A → A))
  sorry

theorem count_f_compositions (f : Fin 6 → Fin 6) 
  (h : ∀ x : Fin 6, (f ∘ f ∘ f) x = x) :
  count_special_functions = 81 :=
sorry

end count_f_compositions_l386_386822


namespace hall_length_is_correct_l386_386344

noncomputable def hall_length (total_expenditure cost_per_sq_meter width height : ℕ) : ℕ :=
  let total_area := total_expenditure / cost_per_sq_meter
  let area_floor := width * L
  let area_walls := 2 * (width * height) + 2 * (L * height)
  let total_area_covered := area_floor + area_walls
  let L := (total_area - area_walls) / width
  L

theorem hall_length_is_correct (L : ℕ) (total_expenditure : ℕ) (cost_per_sq_meter : ℕ) (width : ℕ) (height : ℕ) :
  total_expenditure = 19000 →
  cost_per_sq_meter = 20 →
  width = 15 →
  height = 5 →
  hall_length total_expenditure cost_per_sq_meter width height = 32 :=
by
  intros
  unfold hall_length
  sorry

end hall_length_is_correct_l386_386344


namespace prime_cubic_solution_l386_386438

theorem prime_cubic_solution :
  ∃ p1 p2 : ℕ, (Nat.Prime p1 ∧ Nat.Prime p2) ∧ p1 ≠ p2 ∧
  (p1^3 + p1^2 - 18*p1 + 26 = 0) ∧ (p2^3 + p2^2 - 18*p2 + 26 = 0) :=
by
  sorry

end prime_cubic_solution_l386_386438


namespace rhombus_perimeter_l386_386461

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) : 
  let a := d1 / 2 in
  let b := d2 / 2 in
  let c := Real.sqrt (a^2 + b^2) in
  let side := c in
  let perimeter := 4 * side in
  perimeter = 52 := 
by 
  let a := 5 in 
  let b := 12 in 
  have h3 : a = d1 / 2, by rw [h1]; norm_num
  have h4 : b = d2 / 2, by rw [h2]; norm_num
  let c := Real.sqrt (5^2 + 12^2),
  let side := c,
  have h5 : c = 13, by norm_num,
  let perimeter := 4 * 13,
  show perimeter = 52, by norm_num; sorry

end rhombus_perimeter_l386_386461


namespace k_value_l386_386787

noncomputable def find_k (AB BC AC BD : ℝ) (h_AB : AB = 3) (h_BC : BC = 4) (h_AC : AC = 5) (h_BD : BD = (12/7) * Real.sqrt 2) : ℝ :=
  12 / 7

theorem k_value (AB BC AC BD : ℝ) (h_AB : AB = 3) (h_BC : BC = 4) (h_AC : AC = 5) (h_BD : BD = (12/7) * Real.sqrt 2) : 
  find_k AB BC AC BD h_AB h_BC h_AC h_BD = 12 / 7 :=
by
  sorry

end k_value_l386_386787


namespace cone_prism_volume_ratio_l386_386592

-- Define the volumes and the ratio proof problem
theorem cone_prism_volume_ratio (r h : ℝ) (h_pos : 0 < r) (h_height : 0 < h) :
    let V_cone := (1 / 12) * π * r^2 * h
    let V_prism := 3 * r^2 * h
    (V_cone / V_prism) = (π / 36) :=
by
    -- Here we define the volumes of the cone and prism as given in the problem
    let V_cone := (1 / 12) * π * r^2 * h
    let V_prism := 3 * r^2 * h
    -- We then assert the ratio condition based on the solution
    sorry

end cone_prism_volume_ratio_l386_386592


namespace sum_of_terms_l386_386326

theorem sum_of_terms (x : ℝ) (h : x = 0.25) : 625^(-x) + 25^(-2 * x) + 5^(-4 * x) = 3 / 5 :=
by
  sorry

end sum_of_terms_l386_386326


namespace find_c_plus_one_over_b_l386_386437

theorem find_c_plus_one_over_b 
  (a b c : ℝ) 
  (a_pos : 0 < a) 
  (b_pos : 0 < b) 
  (c_pos : 0 < c) 
  (h1 : a * b * c = 1) 
  (h2 : a + 1 / c = 8) 
  (h3 : b + 1 / a = 20) : 
  c + 1 / b = 10 / 53 := 
sorry

end find_c_plus_one_over_b_l386_386437


namespace average_contribution_increase_l386_386052

theorem average_contribution_increase
  (average_old : ℝ)
  (num_people_old : ℕ)
  (john_donation : ℝ)
  (increase_percentage : ℝ) :
  average_old = 75 →
  num_people_old = 3 →
  john_donation = 150 →
  increase_percentage = 25 :=
by {
  sorry
}

end average_contribution_increase_l386_386052


namespace area_of_union_of_triangle_and_reflection_l386_386192

-- Define points in ℝ²
structure Point where
  x : ℝ
  y : ℝ

-- Define the vertices of the original triangle
def A : Point := ⟨2, 3⟩
def B : Point := ⟨4, -1⟩
def C : Point := ⟨7, 0⟩

-- Define the vertices of the reflected triangle
def A' : Point := ⟨-2, 3⟩
def B' : Point := ⟨-4, -1⟩
def C' : Point := ⟨-7, 0⟩

-- Calculate the area of a triangle given three points
def triangleArea (P Q R : Point) : ℝ :=
  0.5 * |P.x * (Q.y - R.y) + Q.x * (R.y - P.y) + R.x * (P.y - Q.y)|

-- Statement to prove: the area of the union of the original and reflected triangles
theorem area_of_union_of_triangle_and_reflection :
  triangleArea A B C + triangleArea A' B' C' = 14 := 
sorry

end area_of_union_of_triangle_and_reflection_l386_386192


namespace arithmetic_sequence_y_value_l386_386518

theorem arithmetic_sequence_y_value :
  ∃ y : ℤ, (∃ a1 a3 : ℤ, a1 = 9 ∧ a3 = 81 ∧ y = (a1 + a3) / 2) → y = 45 :=
by
  sorry

end arithmetic_sequence_y_value_l386_386518


namespace problem_solution_l386_386733

def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4

def ray_ends_at (m : ℝ) : Prop := 
  m ∈ Set.Ioo (-∞) (-7 * Real.sqrt 3 / 3) ∪ Set.Ioo (7 * Real.sqrt 3 / 3) ∞

theorem problem_solution (m : ℝ) : ray_ends_at m :=
sorry

end problem_solution_l386_386733


namespace additional_charge_per_segment_l386_386796

variable (initial_fee : ℝ := 2.35)
variable (total_charge : ℝ := 5.5)
variable (distance : ℝ := 3.6)
variable (segment_length : ℝ := (2/5 : ℝ))

theorem additional_charge_per_segment :
  let number_of_segments := distance / segment_length
  let charge_for_distance := total_charge - initial_fee
  let additional_charge_per_segment := charge_for_distance / number_of_segments
  additional_charge_per_segment = 0.35 :=
by
  sorry

end additional_charge_per_segment_l386_386796


namespace area_of_union_of_triangles_l386_386374

/--
In triangle ABC, AB = 10, BC = 11, AC = 12,
P is the midpoint of side BC. Points A', B', and C'
are the images of A, B, and C respectively after a 
90 degree rotation about P. The area of the union 
of the two regions enclosed by the triangles ABC 
and A'B'C' is 2 sqrt(16.5 * 6.5 * 5.5 * 4.5).
-/
theorem area_of_union_of_triangles :
  ∀ (A B C P A' B' C' : Type)
  (dAB dBC dCA : ℝ)
  (hAB : dAB = 10)
  (hBC : dBC = 11)
  (hCA : dCA = 12)
  (hP_midpoint : ∀ (B C : point), midpoint B C = P)
  (h_rotation : ∀ (A B C P : point), rotation_about P 90 (triangle A B C) = triangle A' B' C'),
  area (union (triangle A B C) (triangle A' B' C')) = 2 * (√(16.5 * 6.5 * 5.5 * 4.5)) := 
by {
  sorry
}

end area_of_union_of_triangles_l386_386374


namespace no_constants_c_d_for_trig_identity_l386_386498

theorem no_constants_c_d_for_trig_identity :
  ¬ ∃ (c d : ℝ), ∀ θ : ℝ, sin θ ^ 2 = c * sin (2 * θ) + d * sin θ :=
by
  sorry

end no_constants_c_d_for_trig_identity_l386_386498


namespace sequence_general_term_l386_386122

noncomputable def a_sequence (a : ℕ → ℤ) : Prop :=
∀ n ≥ 1, (n - 1) * a (n + 1) = (n + 1) * a n - 2 * (n - 1)

theorem sequence_general_term (a : ℕ → ℤ) (h_recurrence : a_sequence a) (h_a100 : a 100 = 10098) :
  ∀ n ≥ 1, a n = (n - 1) * (n + 2) :=
begin
  sorry
end

end sequence_general_term_l386_386122


namespace train_crossing_time_l386_386319

def length_of_train : ℝ := 110
def speed_of_train_kmph : ℝ := 72
def length_of_bridge : ℝ := 132

def speed_of_train_mps : ℝ := speed_of_train_kmph * (1000 / 3600)
def total_distance : ℝ := length_of_train + length_of_bridge
def crossing_time : ℝ := total_distance / speed_of_train_mps

theorem train_crossing_time :
  crossing_time = 12.1 := 
  by
    unfold length_of_train speed_of_train_kmph length_of_bridge speed_of_train_mps total_distance crossing_time
    sorry

end train_crossing_time_l386_386319


namespace diameter_ratio_base_to_top_l386_386577

def conical_frustum (R r : ℝ) (h : ℝ) : Prop :=
  (1 / 3) * Math.PI * h * (R^2 + R * r + r^2) = 1000

def half_volume_at_two_thirds_height (R r : ℝ) (h : ℝ) : Prop :=
  (1 / 3) * Math.PI * (2 / 3 * h) * ((2 * R + r) / 3)^2 +>
  ((2 * R + r) / 3) * r + r^2) = 500

def radius_ratio (R r : ℝ) (ratio : ℝ) : Prop := 
  ratio = (r / R)

theorem diameter_ratio_base_to_top (R r : ℝ) (h : ℝ) :
  conical_frustum R r h →
  half_volume_at_two_thirds_height R r h →
  radius_ratio R r (1/2) := 
begin
  sorry
end

end diameter_ratio_base_to_top_l386_386577


namespace water_breaks_vs_sitting_breaks_l386_386046

theorem water_breaks_vs_sitting_breaks :
  (240 / 20) - (240 / 120) = 10 := by
  sorry

end water_breaks_vs_sitting_breaks_l386_386046


namespace solve_for_x_add_y_l386_386279

theorem solve_for_x_add_y (x y : ℤ) 
  (h1 : y = 245) 
  (h2 : x - y = 200) : 
  x + y = 690 :=
by {
  -- Here we would provide the proof if needed
  sorry
}

end solve_for_x_add_y_l386_386279


namespace sum_proper_divisors_243_l386_386922

theorem sum_proper_divisors_243 : (1 + 3 + 9 + 27 + 81) = 121 :=
by
  sorry

end sum_proper_divisors_243_l386_386922


namespace arc_length_y_l386_386611

-- Define the function y in terms of x
noncomputable def y (x : ℝ) : ℝ := -Real.arccos (Real.sqrt x) + Real.sqrt (x - x^2)

-- Define the interval [0, 1/4]
def interval := Set.Icc 0 (1 / 4)

-- Arc length formula
def arc_length (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
  ∫ x in Set.Icc a b, Real.sqrt (1 + (Real.deriv f x)^2)

-- Prove that the length of the curve is 1
theorem arc_length_y : arc_length y 0 (1 / 4) = 1 :=
by
  sorry

end arc_length_y_l386_386611


namespace arithmetic_sequence_middle_term_l386_386529

theorem arithmetic_sequence_middle_term :
  let a1 := 3^2
  let a3 := 3^4
  let y := (a1 + a3) / 2
  y = 45 :=
by
  let a1 := (3:ℕ)^2
  let a3 := (3:ℕ)^4
  let y := (a1 + a3) / 2
  have : a1 = 9 := by norm_num
  have : a3 = 81 := by norm_num
  have : y = 45 := by norm_num
  exact this

end arithmetic_sequence_middle_term_l386_386529


namespace first_term_of_geometric_series_l386_386987

theorem first_term_of_geometric_series :
  ∃ (a : ℝ), let r : ℝ := -1/3, S : ℝ := 18 in S = a / (1 - r) ∧ a = 24 :=
by
  use 24
  let r := -1 / 3
  let S := 18
  have h : S = 24 / (1 - r) := by
    rw [←S, rfl]; sorry
  exact ⟨h, rfl⟩

end first_term_of_geometric_series_l386_386987


namespace all_flowers_bloom_simultaneously_l386_386772

-- Define days of the week
inductive Day : Type
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday
deriving DecidableEq

open Day

-- Define bloom conditions for the flowers
def sunflowers_bloom (d : Day) : Prop :=
  d ≠ Tuesday ∧ d ≠ Thursday ∧ d ≠ Sunday

def lilies_bloom (d : Day) : Prop :=
  d ≠ Thursday ∧ d ≠ Saturday

def peonies_bloom (d : Day) : Prop :=
  d ≠ Sunday

-- Define the main theorem
theorem all_flowers_bloom_simultaneously : ∃ d : Day, 
  sunflowers_bloom d ∧ lilies_bloom d ∧ peonies_bloom d ∧
  (∀ d', d' ≠ d → ¬ (sunflowers_bloom d' ∧ lilies_bloom d' ∧ peonies_bloom d')) :=
by
  sorry

end all_flowers_bloom_simultaneously_l386_386772


namespace value_of_y_in_arithmetic_sequence_l386_386533

theorem value_of_y_in_arithmetic_sequence :
    ∃ y : ℤ, (arithmetic_sequence (3^2) y (3^4)) ∧ y = 45 := by
  -- Here we define the arithmetic sequence condition.
  def arithmetic_sequence (a b c : ℤ) : Prop := b = (a + c) / 2
  sorry

end value_of_y_in_arithmetic_sequence_l386_386533


namespace youngest_child_is_five_l386_386495

-- Define the set of prime numbers
def is_prime (n: ℕ) := n > 1 ∧ ∀ m: ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the ages of the children
def youngest_child_age (x: ℕ) : Prop :=
  is_prime x ∧
  is_prime (x + 2) ∧
  is_prime (x + 6) ∧
  is_prime (x + 8) ∧
  is_prime (x + 12) ∧
  is_prime (x + 14)

-- The main theorem stating the age of the youngest child
theorem youngest_child_is_five : ∃ x: ℕ, youngest_child_age x ∧ x = 5 :=
  sorry

end youngest_child_is_five_l386_386495


namespace part_a_part_b_l386_386155

-- Part (a)
theorem part_a (f : ℚ → ℝ) (h_add : ∀ x y : ℚ, f (x + y) = f x + f y) (h_mul : ∀ x y : ℚ, f (x * y) = f x * f y) :
  (∀ x : ℚ, f x = x) ∨ (∀ x : ℚ, f x = 0) :=
sorry

-- Part (b)
theorem part_b (f : ℝ → ℝ) (h_add : ∀ x y : ℝ, f (x + y) = f x + f y) (h_mul : ∀ x y : ℝ, f (x * y) = f x * f y) :
  (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = 0) :=
sorry

end part_a_part_b_l386_386155


namespace sarah_age_ratio_l386_386426

theorem sarah_age_ratio 
  (S M : ℕ) 
  (h1 : S = 3 * (S / 3))
  (h2 : S - M = 5 * (S / 3 - 2 * M)) : 
  S / M = 27 / 2 := 
sorry

end sarah_age_ratio_l386_386426


namespace solve_sqrt_eq_l386_386244

theorem solve_sqrt_eq (z : ℤ) (h : sqrt (10 + 3 * z) = 8) : z = 18 := 
by {
  sorry
}

end solve_sqrt_eq_l386_386244


namespace solve_for_x_l386_386757

variables {x : ℝ}

/-- If (x - 2 * I) * I = 2 + I, then x = 1. -/
theorem solve_for_x (h : (x - 2 * complex.I) * complex.I = 2 + complex.I) : x = 1 :=
sorry

end solve_for_x_l386_386757


namespace determine_p_q_l386_386229

theorem determine_p_q (r1 r2 p q : ℝ) (h1 : r1 + r2 = 5) (h2 : r1 * r2 = 6) (h3 : r1^2 + r2^2 = -p) (h4 : r1^2 * r2^2 = q) : p = -13 ∧ q = 36 :=
by
  sorry

end determine_p_q_l386_386229


namespace hilt_miles_traveled_l386_386409

theorem hilt_miles_traveled (initial_miles lunch_additional_miles : Real) (h_initial : initial_miles = 212.3) (h_lunch : lunch_additional_miles = 372.0) :
  initial_miles + lunch_additional_miles = 584.3 :=
by
  sorry

end hilt_miles_traveled_l386_386409


namespace number_of_ordered_pairs_l386_386634

theorem number_of_ordered_pairs (h : ∀ (m n : ℕ), 0 < m → 0 < n → 6/m + 3/n = 1 → true) : 
∃! (s : Finset (ℕ × ℕ)), s.card = 4 ∧ ∀ (x : ℕ × ℕ), x ∈ s → 0 < x.1 ∧ 0 < x.2 ∧ 6 / ↑x.1 + 3 / ↑x.2 = 1 :=
by
-- Sorry, skipping the proof
  sorry

end number_of_ordered_pairs_l386_386634


namespace num_coprime_to_15_l386_386638

theorem num_coprime_to_15 :
  {a : ℕ | a < 15 ∧ Nat.gcd 15 a = 1}.toFinset.card = 8 :=
by
  sorry

end num_coprime_to_15_l386_386638


namespace geometric_sequence_ratio_l386_386003

theorem geometric_sequence_ratio (a : ℕ → ℝ) (h : ∀ n, a (n + 1) = 2 * a n) :
  (2 * a 2 + a 3) / (2 * a 4 + a 5) = 1 / 6 :=
by
  sorry

end geometric_sequence_ratio_l386_386003


namespace square_area_l386_386842

theorem square_area {x : ℝ} (h1 : sqrt (x^2 + (x/3)^2) = 20) : x^2 = 360 :=
by {
  have h2 : x^2 + (x/3)^2 = 400,
    from (real.mul_self_sqrt (x^2 + (x/3)^2)).symm.trans h1,

  have h3 : x^2 + x^2 / 9 = 400,
    by rwa [div_pow, one_pow, add_div, ←mul_div_assoc, mul_div_cancel' _ (by norm_num : (9 : ℝ) ≠ 0)],

  have h4 : (10/9) * x^2 = 400,
    by { rw [add_comm, add_mul, ← one_mul x^2] at h3, linarith },

  have h5 : x^2 = 400 * (9/10),
    by { symmetry, convert eq_div_iff_mul_eq.2 h4, rwa [div_self (by norm_num : 10 ≠ 0)], },

  exact h5,
  sorry
}

end square_area_l386_386842


namespace rotate_right_triangle_along_right_angle_produces_cone_l386_386871

-- Define a right triangle and the conditions for its rotation
structure RightTriangle (α β γ : ℝ) :=
  (zero_angle : α = 0)
  (ninety_angle_1 : β = 90)
  (ninety_angle_2 : γ = 90)
  (sum_180 : α + β + γ = 180)

-- Define the theorem for the resulting shape when rotating the right triangle
theorem rotate_right_triangle_along_right_angle_produces_cone
  (T : RightTriangle α β γ) (line_of_rotation_contains_right_angle : α = 90 ∨ β = 90 ∨ γ = 90) :
  ∃ shape, shape = "cone" :=
sorry

end rotate_right_triangle_along_right_angle_produces_cone_l386_386871


namespace overlapping_area_of_rectangular_strips_l386_386507

theorem overlapping_area_of_rectangular_strips (theta : ℝ) (h_theta : theta ≠ 0) :
  let width := 2
  let diag_1 := width
  let diag_2 := width / Real.sin theta
  let area := (diag_1 * diag_2) / 2
  area = 2 / Real.sin theta :=
by
  let width := 2
  let diag_1 := width
  let diag_2 := width / Real.sin theta
  let area := (diag_1 * diag_2) / 2
  sorry

end overlapping_area_of_rectangular_strips_l386_386507


namespace part1_beef_noodles_mix_sauce_purchased_l386_386166

theorem part1_beef_noodles_mix_sauce_purchased (x y : ℕ) (h1 : x + y = 170) (h2 : 15 * x + 20 * y = 3000) :
  x = 80 ∧ y = 90 :=
sorry

end part1_beef_noodles_mix_sauce_purchased_l386_386166


namespace proof_arithmetic_sequence_l386_386708

noncomputable def arithmetic_sequence_proof (a b : ℕ → ℝ) (S T : ℕ → ℝ) : Prop :=
  (∀ n, S n = (n / 2) * (2 * a 1 + (n - 1) * (a 2 - a 1))) ∧
  (∀ n, T n = (n / 2) * (2 * b 1 + (n - 1) * (b 2 - b 1))) ∧
  (∀ n, S n / T n = (n + 1) / (n + 3)) →
  (a 2 / (b 1 + b 5) + a 4 / (b 2 + b 4) = 3 / 4)

theorem proof_arithmetic_sequence (a b : ℕ → ℝ) (S T : ℕ → ℝ) :
  arithmetic_sequence_proof a b S T :=
by {
  unfold arithmetic_sequence_proof,
  intro h,
  sorry
}

end proof_arithmetic_sequence_l386_386708


namespace cos2C_value_l386_386004

variables (A B C : Point) (BC AC : ℝ)
variables (S : ℝ)
variables [BC_eq : BC = 8] [AC_eq : AC = 5] [S_eq : S = 12]

def sin_C (BC AC S : ℝ) : ℝ := (2 * S) / (BC * AC)

def cos2C (sinC : ℝ) : ℝ := 1 - 2 * sinC^2

theorem cos2C_value : cos2C (sin_C 8 5 12) = 7 / 25 := by
  unfold cos2C sin_C
  rw [BC_eq, AC_eq, S_eq]
  unfold sin_C
  have h : sin_C 8 5 12 = 3 / 5 := by simp [sin_C]
  rw [h]
  simp [cos2C]
  sorry

end cos2C_value_l386_386004


namespace michael_brother_initial_money_l386_386834

theorem michael_brother_initial_money :
  let michael_money := 42
  let brother_receive := michael_money / 2
  let candy_cost := 3
  let brother_left_after_candy := 35
  let brother_initial_money := brother_left_after_candy + candy_cost
  in brother_initial_money - brother_receive = 17 :=
by
  intros
  -- definitions
  let michael_money := 42
  let brother_receive := michael_money / 2
  let candy_cost := 3
  let brother_left_after_candy := 35
  let brother_initial_money := brother_left_after_candy + candy_cost
  -- result
  show brother_initial_money - brother_receive = 17
  sorry

end michael_brother_initial_money_l386_386834


namespace prove_central_angle_of_sector_l386_386002

noncomputable def central_angle_of_sector (R α : ℝ) : Prop :=
  (2 * R + R * α = 8) ∧ (1 / 2 * α * R^2 = 4)

theorem prove_central_angle_of_sector :
  ∃ α R : ℝ, central_angle_of_sector R α ∧ α = 2 :=
sorry

end prove_central_angle_of_sector_l386_386002


namespace trapezoid_area_l386_386364

structure Trapezoid :=
  (AB CD : ℝ)
  (S T X Y : Type)

def midpoint (a b : ℝ) : ℝ := (a + b) / 2

def area_quadrilateral (b h : ℝ) : ℝ := b * h

def area_trapezoid (a b h : ℝ) : ℝ := 0.5 * (a + b) * h

theorem trapezoid_area (AB CD : ℝ) (h : ℝ) (quad_area : ℝ)
  (H1 : AB = 7) 
  (H2 : CD = 4) 
  (H3 : quad_area = 12)
  (H4 : quad_area = area_quadrilateral CD h) :
  area_trapezoid AB CD h = 16.5 :=
by
  sorry

end trapezoid_area_l386_386364


namespace triangle_area_comparison_l386_386395

noncomputable def semiperimeter (a b c : ℝ) : ℝ :=
  (a + b + c) / 2

noncomputable def heron_area (a b c : ℝ) : ℝ :=
  let s := semiperimeter a b c in
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_area_comparison :
  let C := heron_area 13 14 15 in
  let D := heron_area 12 13 5 in
  C > D :=
by
  sorry

end triangle_area_comparison_l386_386395


namespace problem_solution_l386_386743

def f (x : ℝ) := log (sqrt (x ^ 2 + 1) + x) + (2 ^ x - 1) / (2 ^ x + 1) + 3

axiom g_condition : ∀ x : ℝ, g (-x) + g x = 6

theorem problem_solution :
    (f (log 2022 10) + f (log (1 / 2022) 10) = 6) ∧
    (∀ x : ℝ, g (-x) + g x = 6 → (∀ y, g y = g (y + 6) - 6)) ∧  -- Since symmetry about (0,3) effectively means g(x) - g(0) = g(-x) - g(0)
    (∀ a b : ℝ, f a + f b > 6 → a + b > 0) :=
by
  sorry

end problem_solution_l386_386743


namespace pizzas_eaten_by_Nunzio_l386_386083

def pieces_per_day : ℕ := 3
def piece_fraction_per_pizza : ℚ := 1 / 8
def days : ℕ := 72

theorem pizzas_eaten_by_Nunzio :
  (pieces_per_day * days) / (piece_fraction_per_pizza.denom) = 27 := 
by
  sorry

end pizzas_eaten_by_Nunzio_l386_386083


namespace percent_increase_combined_cost_l386_386383

theorem percent_increase_combined_cost :
  let laptop_last_year := 500
  let tablet_last_year := 200
  let laptop_increase := 10 / 100
  let tablet_increase := 20 / 100
  let new_laptop_cost := laptop_last_year * (1 + laptop_increase)
  let new_tablet_cost := tablet_last_year * (1 + tablet_increase)
  let total_last_year := laptop_last_year + tablet_last_year
  let total_this_year := new_laptop_cost + new_tablet_cost
  let increase := total_this_year - total_last_year
  let percent_increase := (increase / total_last_year) * 100
  percent_increase = 13 :=
by
  sorry

end percent_increase_combined_cost_l386_386383


namespace largest_even_integer_sum_12000_l386_386892

theorem largest_even_integer_sum_12000 : 
  ∃ y, (∑ k in (Finset.range 30), (2 * y + 2 * k) = 12000) ∧ (y + 29) * 2 + 58 = 429 :=
by
  sorry

end largest_even_integer_sum_12000_l386_386892


namespace bricks_needed_l386_386944

theorem bricks_needed 
    (brick_length : ℝ) (brick_width : ℝ) (brick_height : ℝ) 
    (wall_length_m : ℝ) (wall_height_m : ℝ) (wall_width_cm : ℝ)
    (H1 : brick_length = 25) (H2 : brick_width = 11.25) (H3 : brick_height = 6)
    (H4 : wall_length_m = 7) (H5 : wall_height_m = 6) (H6 : wall_width_cm = 22.5) :
    (wall_length_m * 100 * wall_height_m * 100 * wall_width_cm) / (brick_length * brick_width * brick_height) = 5600 :=
by
    sorry

end bricks_needed_l386_386944


namespace find_a2_a3_sequence_constant_general_formula_l386_386706

-- Definition of the sequence and its sum Sn
variables (a : ℕ → ℕ) (S : ℕ → ℕ)

-- Conditions
axiom a1_eq : a 1 = 2
axiom S_eq : ∀ n, S (n + 1) = 4 * a n - 2

-- Prove that a_2 = 4 and a_3 = 8
theorem find_a2_a3 : a 2 = 4 ∧ a 3 = 8 :=
sorry

-- Prove that the sequence {a_n - 2a_{n-1}} is constant
theorem sequence_constant {n : ℕ} (hn : n ≥ 2) :
  ∃ c, ∀ k ≥ 2, a k - 2 * a (k - 1) = c :=
sorry

-- Find the general formula for the sequence
theorem general_formula :
  ∀ n, a n = 2^n :=
sorry

end find_a2_a3_sequence_constant_general_formula_l386_386706


namespace statement_1_statement_3_l386_386723

-- Define basic entities
variable (m n : Line)
variable (α β : Plane)

-- Define the conditions
axiom m_perp_α : m ⊥ α
axiom n_perp_α : n ⊥ α
axiom m_perp_β : m ⊥ β

-- Define theorems to prove
theorem statement_1 : (m ⊥ α) → (n ⊥ α) → (m ∥ n) :=
by sorry

theorem statement_3 : (m ⊥ α) → (m ⊥ β) → (α ∥ β) :=
by sorry

#check statement_1
#check statement_3

end statement_1_statement_3_l386_386723


namespace angle_BDC_45_l386_386821

open Real EuclideanGeometry

theorem angle_BDC_45 (A B C D : Point) (h : is_right_triangle A B C ∧ ∠A = 90 ∧
  (∃ B_bisector : Line, is_angle_bisector B B_bisector D) ∧ (∃ C_bisector : Line, is_angle_bisector C C_bisector D)) :
  ∠BDC = 45 := sorry

end angle_BDC_45_l386_386821


namespace coprime_count_15_l386_386661

theorem coprime_count_15 :
  {a : ℕ | a < 15 ∧ Nat.gcd 15 a = 1}.card = 8 :=
by
sorry

end coprime_count_15_l386_386661


namespace OB_perp_DF_l386_386033

noncomputable section

open EuclideanGeometry

variables {A B C D E F H O M N I : Point}

-- Conditions
variables (circumcenter_A : IsCircumcenter O (Triangle A B C))
variables (altitudes_concurrent : AltitudesConcurrent A B C D E F H)
variables (intersection_M : Line ED ∩ Line AB = {M})
variables (intersection_N : Line FI ∩ Line AC = {N})

-- Prove OB ⊥ DF
theorem OB_perp_DF (hO : IsCircumcenter O (Triangle A B C))
                   (hH : AltitudesConcurrent A B C D E F H)
                   (hM : Line ED ∩ Line AB = {M})
                   (hN : Line FI ∩ Line AC = {N}) :
  Perpendicular (Line OB) (Line DF) ∧ Perpendicular (Line OC) (Line DE) ∧ Perpendicular (Line OH) (Line MN) := 
  by 
  sorry

end OB_perp_DF_l386_386033


namespace arithmetic_sequence_50th_term_l386_386335

theorem arithmetic_sequence_50th_term :
  let a_1 := 3
  let d := 5
  let n := 50
  let a_n := a_1 + (n - 1) * d
  a_n = 248 :=
by
  let a_1 := 3
  let d := 5
  let n := 50
  let a_n := a_1 + (n - 1) * d
  sorry

end arithmetic_sequence_50th_term_l386_386335


namespace race_completion_times_l386_386774

theorem race_completion_times :
  ∃ (Patrick Manu Amy Olivia Sophie Jack : ℕ),
  Patrick = 60 ∧
  Manu = Patrick + 12 ∧
  Amy = Manu / 2 ∧
  Olivia = (2 * Amy) / 3 ∧
  Sophie = Olivia - 10 ∧
  Jack = Sophie + 8 ∧
  Manu = 72 ∧
  Amy = 36 ∧
  Olivia = 24 ∧
  Sophie = 14 ∧
  Jack = 22 := 
by
  -- proof here
  sorry

end race_completion_times_l386_386774


namespace tan_theta_eq_5_over_12_l386_386806

open Matrix

variable {k θ : ℝ} (h_k : k > 0)
variable R D : Matrix (Fin 2) (Fin 2) ℝ

-- Defining the matrices
def R := !![
  [Real.cos θ, -Real.sin θ],
  [Real.sin θ, Real.cos θ]
]

def D := !![
  [k, 0],
  [0, k]
]

-- Given condition
def RD := !![
  [12, -5],
  [5, 12]
]

-- The goal is to prove that tan(θ) = 5/12
theorem tan_theta_eq_5_over_12 (hrd : R ⬝ D = RD) : Real.tan θ = 5 / 12 := by
  sorry

end tan_theta_eq_5_over_12_l386_386806


namespace no_real_roots_quadratic_l386_386332

theorem no_real_roots_quadratic (m : Real) :
  (∀ (m = -1) (m = 0) (m = 1) (m = sqrt 3), (m > 1) ↔ (m = sqrt 3)) :=
by
  sorry

end no_real_roots_quadratic_l386_386332


namespace net_revenue_per_person_l386_386379

theorem net_revenue_per_person :
  let market_value_A := 500000
  let market_value_B := 700000
  let selling_price_A := market_value_A * 1.2
  let selling_price_B := market_value_B * 0.9
  let tax_A := 0.1
  let tax_B := 0.12
  let net_revenue_A := selling_price_A * (1 - tax_A)
  let net_revenue_B := selling_price_B * (1 - tax_B)
  let persons_A := 4
  let persons_B := 5
  let net_revenue_per_person_A := net_revenue_A / persons_A
  let net_revenue_per_person_B := net_revenue_B / persons_B
  net_revenue_per_person_A = 135000 ∧ net_revenue_per_person_B = 110880
by
  sorry

end net_revenue_per_person_l386_386379


namespace max_area_of_rotating_lines_triangle_l386_386024

noncomputable def triangle_area_problem : ℝ :=
  let A := (0, 0)
  let B := (12, 0)
  let C := (20, 0)
  let l_A := { p : ℝ × ℝ | p.2 = p.1 } -- line through A with slope 1
  let l_B := { p : ℝ × ℝ | p.1 = 12 } -- vertical line through B
  let l_C := { p : ℝ × ℝ | p.2 = -p.1 + 20 } -- line through C with slope -1
  let X := (12, 8)
  let Y := (8.107, 11.893)
  let Z := (0, 12)
  104

theorem max_area_of_rotating_lines_triangle :
  ∀ (A B C : ℝ × ℝ)
    (l_A l_B l_C : set (ℝ × ℝ)),
    A = (0, 0) →
    B = (12, 0) →
    C = (20, 0) →
    l_A = { p : ℝ × ℝ | p.2 = p.1 } →
    l_B = { p : ℝ × ℝ | p.1 = 12 } →
    l_C = { p : ℝ × ℝ | p.2 = -p.1 + 20 } →
    ∃ X Y Z : ℝ × ℝ,
    X = (12, 8) →
    Y = (8.107, 11.893) →
    Z = (0, 12) →
    ∃ A_max : ℝ,
    A_max = 104 := sorry

end max_area_of_rotating_lines_triangle_l386_386024


namespace value_of_y_in_arithmetic_sequence_l386_386537

theorem value_of_y_in_arithmetic_sequence :
    ∃ y : ℤ, (arithmetic_sequence (3^2) y (3^4)) ∧ y = 45 := by
  -- Here we define the arithmetic sequence condition.
  def arithmetic_sequence (a b c : ℤ) : Prop := b = (a + c) / 2
  sorry

end value_of_y_in_arithmetic_sequence_l386_386537


namespace num_satisfying_integers_l386_386696

noncomputable def satisfies_log_condition (x : ℕ) : Prop := 
  log 10 (x - 20) + log 10 (100 - x) < 3

theorem num_satisfying_integers : 
  (finset.filter (λ x, satisfies_log_condition x) (finset.Ico 21 100)).card = 39 :=
by sorry

end num_satisfying_integers_l386_386696


namespace smallest_n_not_divisible_by_10_l386_386270

theorem smallest_n_not_divisible_by_10 :
  ∃ n > 2016, (1^n + 2^n + 3^n + 4^n) % 10 ≠ 0 ∧ n = 2020 := 
sorry

end smallest_n_not_divisible_by_10_l386_386270


namespace votes_for_art_of_the_deal_l386_386416

-- Define the number of original votes for each category
def original_votes_game_of_thrones : ℕ := 10
def original_votes_twilight : ℕ := 12
def original_votes_art_of_the_deal : ℕ := A

-- Define the number of votes left after Fran's actions
def remaining_votes_art_of_the_deal := 0.20 * (original_votes_art_of_the_deal : ℤ)
def remaining_votes_twilight := (original_votes_twilight / 2 : ℤ)
def total_votes_after_altering := 2 * original_votes_game_of_thrones

-- Problem to solve: find the original number of votes for The Art of the Deal
theorem votes_for_art_of_the_deal (A : ℕ) (h : 10 + 6 + (0.20 * (A : ℤ)) = 20) : A = 20 := by
  sorry

end votes_for_art_of_the_deal_l386_386416


namespace eggs_not_eaten_per_week_l386_386092

theorem eggs_not_eaten_per_week : 
  let trays_bought := 2
  let eggs_per_tray := 24
  let days_per_week := 7
  let eggs_eaten_by_children_per_day := 2 * 2 -- 2 eggs each by 2 children
  let eggs_eaten_by_parents_per_day := 4
  let total_eggs_eaten_per_week := (eggs_eaten_by_children_per_day + eggs_eaten_by_parents_per_day) * days_per_week
  let total_eggs_bought := trays_bought * eggs_per_tray * 2  -- Re-calculated trays
  total_eggs_bought - total_eggs_eaten_per_week = 40 :=
by
  let trays_bought := 2
  let eggs_per_tray := 24
  let days_per_week := 7
  let eggs_eaten_by_children_per_day := 2 * 2
  let eggs_eaten_by_parents_per_day := 4
  let total_eggs_eaten_per_week := (eggs_eaten_by_children_per_day + eggs_eaten_by_parents_per_day) * days_per_week
  let total_eggs_bought := trays_bought * eggs_per_tray * 2
  show total_eggs_bought - total_eggs_eaten_per_week = 40
  sorry

end eggs_not_eaten_per_week_l386_386092


namespace chords_and_circle_l386_386010

theorem chords_and_circle (R : ℝ) (A B C D : ℝ) 
  (hAB : 0 < A - B) (hCD : 0 < C - D) (hR : R > 0) 
  (h_perp : (A - B) * (C - D) = 0) 
  (h_radA : A ^ 2 + B ^ 2 = R ^ 2) 
  (h_radC : C ^ 2 + D ^ 2 = R ^ 2) :
  (A - C)^2 + (B - D)^2 = 4 * R^2 :=
by
  sorry

end chords_and_circle_l386_386010


namespace initial_balance_l386_386802

theorem initial_balance (X : ℝ) : 
  (X - 60 - 30 - 0.25 * (X - 60 - 30) - 10 = 100) ↔ (X = 236.67) := 
  by
    sorry

end initial_balance_l386_386802


namespace find_x_l386_386555

theorem find_x (x : ℝ) (h: 0.8 * 90 = 70 / 100 * x + 30) : x = 60 :=
by
  sorry

end find_x_l386_386555


namespace sum_first_8_even_numbers_is_72_l386_386904

theorem sum_first_8_even_numbers_is_72 : (2 + 4 + 6 + 8 + 10 + 12 + 14 + 16) = 72 :=
by
  sorry

end sum_first_8_even_numbers_is_72_l386_386904


namespace exponential_growth_exceeds_power_growth_l386_386358

theorem exponential_growth_exceeds_power_growth
  (a : ℝ) (alpha : ℝ)
  (ha : 1 < a) (h_alpha : 0 < alpha) :
  ∀ x : ℝ, 0 < x → (deriv (λ x, a^x) x > deriv (λ x, x^alpha) x) :=
by
  sorry

end exponential_growth_exceeds_power_growth_l386_386358


namespace triangle_area_ratio_l386_386788

theorem triangle_area_ratio (A B C D E : Type) 
  {a b c : ℝ}
  (h1 : ∠A = 60) 
  (h2 : ∠B = 45)
  (h3 : ∠D = 45)
  (h4 : ∠C = 75)
  (abc_area_eq : Area (triangle A B C) 
                      = 2 * Area (triangle A D E)) 
  : AD / AB = 1 / real.sqrt(2) :=
by
  sorry

end triangle_area_ratio_l386_386788


namespace average_salary_l386_386777

theorem average_salary (T_salary : ℕ) (R_salary : ℕ) (total_salary : ℕ) (T_count : ℕ) (R_count : ℕ) (total_count : ℕ) :
    T_salary = 12000 * T_count →
    R_salary = 6000 * R_count →
    total_salary = T_salary + R_salary →
    T_count = 6 →
    R_count = total_count - T_count →
    total_count = 18 →
    (total_salary / total_count) = 8000 :=
by
  intros
  sorry

end average_salary_l386_386777


namespace find_x_value_l386_386026

def solve_for_x (a b x : ℝ) (rectangle_perimeter triangle_height equated_areas : Prop) :=
  rectangle_perimeter -> triangle_height -> equated_areas -> x = 20 / 3

-- Definitions of the conditions
def rectangle_perimeter (a b : ℝ) : Prop := 2 * (a + b) = 60
def triangle_height : Prop := 60 > 0
def equated_areas (a b x : ℝ) : Prop := a * b = 30 * x

theorem find_x_value :
  ∃ a b x : ℝ, solve_for_x a b x (rectangle_perimeter a b) triangle_height (equated_areas a b x) :=
  sorry

end find_x_value_l386_386026


namespace number_of_correct_propositions_is_0_l386_386119

noncomputable def proposition_1 (zero greater than_minus_i : Bool) := zero > (- Complex.i)
noncomputable def proposition_2 (z1 z2 sum_real : Complex) := z1 + z2 ∈ ℝ ∧ Complex.conj z1 ≠ z2 
noncomputable def proposition_3 (x y one_i : Complex) := x + y * Complex.i = 1 + Complex.i ∧ ¬(x = 1 ∧ y = 1)
noncomputable def proposition_4 (a pure_imaginary : Complex) := pure_imaginary = a * Complex.i ∧ ¬(a ≠ 0)

theorem number_of_correct_propositions_is_0 
  (prop1 : proposition_1 0 (-Complex.i))
  (prop2 : proposition_2 (3 + Complex.i) (4 - Complex.i) 7)
  (prop3 : proposition_3 Real.Real Real.Real (1 + Complex.i))
  (prop4 : proposition_4 0 (0 * Complex.i)) : 
  0 = 0 := by
sorry

end number_of_correct_propositions_is_0_l386_386119


namespace part1_part2_l386_386291

noncomputable theory
open_locale classical

-- Define given conditions for Part 1 (Ellipse E)
def a : ℝ := 2
def b : ℝ := sqrt 3
def ε : ℝ := 1 / 2
def F := (1,0) : ℝ × ℝ -- One focus F

-- Ellipse equation
def ellipse_eq (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Prove that the standard equation of the ellipse E is x^2/4 + y^2/3 = 1
theorem part1 :
  ellipse_eq x y :=
sorry

-- Define given conditions for Part 2 (Quadrilateral vertices and vectors)
variables {A B C D F M N : ℝ × ℝ}

-- Given conditions
def AC_perpendicular_BD : Prop := (C - A) ⬝ (D - B) = 0
def midpoint (p1 p2 m : ℝ × ℝ): Prop := p1 + p2 = (2:ℝ) • m
def slope_cosine_AC : Prop := cos (angle (C - A) (1,0)) = sqrt 5 / 5

-- Let M be the midpoint of AC and N be the midpoint of BD
def M := (A + C) / 2
def N := (B + D) / 2

-- Define the x-axis intersection point
def intersection_x_axis (p : ℝ × ℝ) : Prop := p.2 = 0

-- Prove that the intersection point of line MN and the x-axis is (4/7, 0)
theorem part2 (P : ℝ × ℝ) :
  intersection_x_axis P ∧ P.1 = 4 / 7 :=
sorry

end part1_part2_l386_386291


namespace percent_increase_is_30_percent_l386_386491

/-- Define the initial processing rate -/
def initial_rate : ℝ := 1250 / 10

/-- Define the number of items processed in the first 6 hours -/
def items_processed_in_6_hours : ℝ := initial_rate * 6

/-- Define the remaining items after the first 6 hours -/
def remaining_items_after_6_hours : ℝ := 1250 - items_processed_in_6_hours

/-- Define the additional items given after the first 6 hours -/
def additional_items : ℝ := 150

/-- Define the total remaining items to be processed -/
def total_remaining_items : ℝ := remaining_items_after_6_hours + additional_items

/-- Define the remaining time to process the remaining items -/
def remaining_time : ℝ := 4

/-- Define the new required rate to process the remaining items in the remaining time -/
def new_required_rate : ℝ := total_remaining_items / remaining_time

/-- Calculate the percent increase needed in the rate -/
def percent_increase : ℝ := ((new_required_rate - initial_rate) / initial_rate) * 100

/-- The proof statement, asserting that the percent increase is 30% -/
theorem percent_increase_is_30_percent : percent_increase = 30 := 
by 
  sorry

end percent_increase_is_30_percent_l386_386491


namespace largest_possible_d_l386_386810

theorem largest_possible_d (a b c d : ℝ) 
  (h1 : a + b + c + d = 10) 
  (h2 : ab + ac + ad + bc + bd + cd = 20) :
  d ≤ (5 + Real.sqrt 105) / 2 := 
sorry

end largest_possible_d_l386_386810


namespace equivalent_problem_l386_386057

-- Definitions
variables {A B C A' B' C' : Point} -- Points in the plane
variables (triangle_ABC : acute_angle_triangle A B C)

-- Conditions
axiom cond1 : A' ∈ segment B C
axiom cond2 : B' ∈ segment C A
axiom cond3 : C' ∈ segment A B
axiom cond4 : is_parallel (line A' B') (line A B)
axiom cond5 : is_bisector (line C' C) (angle A' C' B')
axiom cond6 : length (segment A' C') + length (segment B' C') = length (segment A B)

-- Theorem to prove
theorem equivalent_problem 
  (h1 : acute_angle_triangle A B C)
  (h2 : A' ∈ segment B C)
  (h3 : B' ∈ segment C A)
  (h4 : C' ∈ segment A B)
  (h5 : is_parallel (line A' B') (line A B))
  (h6 : is_bisector (line C' C) (angle A' C' B'))
  (h7 : length (segment A' C') + length (segment B' C') = length (segment A B) )
  : is_bisector (line C' C) (angle A' C' B') ∧
    length (segment A' C') + length (segment B' C') = length (segment A B) ∧
    is_parallel (line A' B') (line A B) := 
sorry

end equivalent_problem_l386_386057


namespace miran_has_fewest_papers_l386_386081

def hasFewestColoredPapers (miran junga minsu : ℕ) : ℕ :=
  if miran ≤ junga ∧ miran ≤ minsu then miran
  else if junga ≤ miran ∧ junga ≤ minsu then junga
  else minsu

theorem miran_has_fewest_papers (miran junga minsu : ℕ) 
  (h_miran : miran = 6) (h_junga : junga = 13) (h_minsu : minsu = 10) :
  hasFewestColoredPapers miran junga minsu = 6 :=
by
  rw [h_miran, h_junga, h_minsu]
  unfold hasFewestColoredPapers
  simp
  rfl

end miran_has_fewest_papers_l386_386081


namespace piecewise_function_evaluation_l386_386820

-- Definition of the piecewise function f
def f (x : ℝ) : ℝ :=
  if x < 1 then 1 + Real.log (2 - x) / Real.log 3 else Real.exp ((x - 1) * Real.log 3)

-- Statement that we need to prove
theorem piecewise_function_evaluation : 
  f (-7) + f (Real.log 12 / Real.log 3) = 7 := 
sorry

end piecewise_function_evaluation_l386_386820


namespace num_coprime_to_15_l386_386643

theorem num_coprime_to_15 : (filter (fun a => Nat.gcd a 15 = 1) (List.range 15)).length = 8 := by
  sorry

end num_coprime_to_15_l386_386643


namespace move_point_right_l386_386414

-- Definitions for conditions
def point_on_number_line : ℤ := -3

-- Statement to prove
theorem move_point_right (A : ℤ) (hA : A = point_on_number_line) (units : ℤ) (hunits : units = 7) : A + units = 4 :=
by {
  rw [hA, hunits],
  exact rfl,
}

end move_point_right_l386_386414


namespace f_in_1_3_2_decreasing_f_in_1_3_2_lt_0_l386_386632

noncomputable def f (x : ℝ) : ℝ :=
if x ∈ Ioc 0 (1/2 : ℝ) then
  Real.log2 (x + 1)
else 
  if x ∈ Ioc (-1 : ℝ) (-1/2) then
    -Real.log2 (-x)
  else
    0 -- put the remaining pieces of the definitions as required

-- Define the conditions for odd function and periodicity
lemma f_odd (x : ℝ) : f (-x) = -f x :=
sorry

lemma f_period_2 (x : ℝ) : f (x + 2) = f x :=
sorry

lemma f_in_interval_01 (x : ℝ) (hx : x ∈ Ioc 0 (1/2 : ℝ)) : f x = Real.log2 (x + 1) :=
sorry

lemma f_in_interval_m1m05 (x : ℝ) (hx : x ∈ Ioc (-1 : ℝ) (-1/2 : ℝ)) : f x = -Real.log2 (-x) :=
sorry

-- Prove that f(x) is decreasing and less than 0 in (1, 3/2)
theorem f_in_1_3_2_decreasing : 
  ∀ x1 x2 : ℝ, (1 < x1 ∧ x1 < 3/2) → (1 < x2 ∧ x2 < 3/2) → x1 < x2 → f x1 > f x2 :=
sorry

theorem f_in_1_3_2_lt_0 : 
  ∀ x : ℝ, (1 < x ∧ x < 3/2) → f x < 0 :=
sorry

end f_in_1_3_2_decreasing_f_in_1_3_2_lt_0_l386_386632


namespace jack_speed_to_beach_12_mph_l386_386042

theorem jack_speed_to_beach_12_mph :
  let distance := 16 * (1 / 8) -- distance in miles
  let time := 10 / 60        -- time in hours
  distance / time = 12 :=    -- speed in miles per hour
by
  let distance := 16 * (1 / 8) -- evaluation of distance
  let time := 10 / 60          -- evaluation of time
  show distance / time = 12    -- final speed calculation
  from sorry

end jack_speed_to_beach_12_mph_l386_386042


namespace set_intersection_complement_l386_386318

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | x < -1 ∨ x > 4}
def complement_B : Set ℝ := U \ B
def expected_set : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

theorem set_intersection_complement :
  A ∩ complement_B = expected_set :=
by
  sorry

end set_intersection_complement_l386_386318


namespace arithmetic_sequence_middle_term_l386_386527

theorem arithmetic_sequence_middle_term :
  let a1 := 3^2
  let a3 := 3^4
  let y := (a1 + a3) / 2
  y = 45 :=
by
  let a1 := (3:ℕ)^2
  let a3 := (3:ℕ)^4
  let y := (a1 + a3) / 2
  have : a1 = 9 := by norm_num
  have : a3 = 81 := by norm_num
  have : y = 45 := by norm_num
  exact this

end arithmetic_sequence_middle_term_l386_386527


namespace hyperbola_equation_params_sum_l386_386771

theorem hyperbola_equation_params_sum : 
  let h := 2
      k := 0
      a := 5
      c := 8
      b := Real.sqrt (c^2 - a^2)
  in h + k + a + b = 7 + Real.sqrt 39 :=
by
  sorry

end hyperbola_equation_params_sum_l386_386771


namespace exists_n_divisible_by_p_l386_386815

def S (t : ℤ) (p : ℕ) : ℕ → ℤ :=
  λ n, (3 - 7 * t) * 2^n + (18 * t - 9) * 3^n + (6 - 10 * t) * 4^n

theorem exists_n_divisible_by_p
  (t : ℤ) (p : ℕ) (hp : Nat.Prime p) (hodd : p % 2 = 1) :
  ∃ n : ℕ, n > 0 ∧ p ∣ S t p n :=
by
  sorry

end exists_n_divisible_by_p_l386_386815


namespace started_with_l386_386605

-- Define the conditions
def total_eggs : ℕ := 70
def bought_eggs : ℕ := 62

-- Define the statement to prove
theorem started_with (initial_eggs : ℕ) : initial_eggs = total_eggs - bought_eggs → initial_eggs = 8 := by
  intro h
  sorry

end started_with_l386_386605


namespace area_triangle_AKF_l386_386106

open Real EuclideanGeometry

-- Definition of the focus of the parabola and related geometric entities
def parabola := {p : Point | p.y ^ 2 = 4 * p.x}
def F : Point := ⟨1, 0⟩
def directrix : Line := {p | p.x = -1}

-- Conditions in the problem
axiom A_on_parabola : A ∈ parabola
axiom B_on_directrix : B ∈ directrix
axiom line_through_F : IsLineConnectedThrough F (line_through_points A B)
axiom perp_AK_line_to_directrix : Perpendicular AK directrix 
axiom AF_eq_BF : dist A F = dist B F

-- Proven statement: the area of triangle AKF
theorem area_triangle_AKF : area_of_triangle A K F = 4 * sqrt 3 := 
sorry

end area_triangle_AKF_l386_386106


namespace constant_term_expansion_l386_386101

theorem constant_term_expansion (x : ℝ) : (∑ r in Finset.range 7, (Nat.choose 6 r) * x^(6 - 2*r)) = 20 :=
by
  sorry

end constant_term_expansion_l386_386101


namespace sqrt_of_sum_of_powers_l386_386931

theorem sqrt_of_sum_of_powers : 
  sqrt (5^3 + 5^4 + 5^5) = 5 * sqrt 155 :=
by
  sorry

end sqrt_of_sum_of_powers_l386_386931


namespace sum_proper_divisors_243_l386_386928

theorem sum_proper_divisors_243 : 
  let proper_divisors_243 := [1, 3, 9, 27, 81] in
  proper_divisors_243.sum = 121 := 
by
  sorry

end sum_proper_divisors_243_l386_386928


namespace engineer_formula_updated_l386_386232

theorem engineer_formula_updated (T H : ℕ) (hT : T = 5) (hH : H = 10) :
  (30 * T^5) / (H^3 : ℚ) = 375 / 4 := by
  sorry

end engineer_formula_updated_l386_386232


namespace matrix_determinant_equality_l386_386803

open Complex Matrix

variable {n : Type*} [Fintype n] [DecidableEq n]

theorem matrix_determinant_equality (A B : Matrix n n ℂ) (x : ℂ) 
  (h1 : A ^ 2 + B ^ 2 = 2 * A * B) :
  det (A - x • 1) = det (B - x • 1) :=
  sorry

end matrix_determinant_equality_l386_386803


namespace second_tray_capacity_l386_386557

noncomputable def capacity_first_tray : ℕ := 260

noncomputable def capacity_second_tray (x : ℕ) : ℕ := x - 20

theorem second_tray_capacity :
  let x := capacity_first_tray in
  capacity_second_tray x = 240 :=
by
  sorry

end second_tray_capacity_l386_386557


namespace boat_cannot_complete_round_trip_l386_386963

theorem boat_cannot_complete_round_trip
  (speed_still_water : ℝ)
  (speed_current : ℝ)
  (distance : ℝ)
  (total_time : ℝ)
  (speed_still_water_pos : speed_still_water > 0)
  (speed_current_nonneg : speed_current ≥ 0)
  (distance_pos : distance > 0)
  (total_time_pos : total_time > 0) :
  let speed_downstream := speed_still_water + speed_current
  let speed_upstream := speed_still_water - speed_current
  let time_downstream := distance / speed_downstream
  let time_upstream := distance / speed_upstream
  let total_trip_time := time_downstream + time_upstream
  total_trip_time > total_time :=
by {
  -- Proof goes here
  sorry
}

end boat_cannot_complete_round_trip_l386_386963


namespace minimum_3x_4y_l386_386330

theorem minimum_3x_4y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = 5 * x * y) : 3 * x + 4 * y ≥ 5 :=
by
  sorry

end minimum_3x_4y_l386_386330


namespace people_visited_neither_l386_386556

theorem people_visited_neither (total_people iceland_visitors norway_visitors both_visitors : ℕ)
  (h1 : total_people = 100)
  (h2 : iceland_visitors = 55)
  (h3 : norway_visitors = 43)
  (h4 : both_visitors = 61) :
  total_people - (iceland_visitors + norway_visitors - both_visitors) = 63 :=
by
  sorry

end people_visited_neither_l386_386556


namespace initial_assessed_value_l386_386340

theorem initial_assessed_value (V : ℝ) (tax_rate : ℝ) (new_value : ℝ) (tax_increase : ℝ) 
  (h1 : tax_rate = 0.10) 
  (h2 : new_value = 28000) 
  (h3 : tax_increase = 800) 
  (h4 : tax_rate * new_value = tax_rate * V + tax_increase) : 
  V = 20000 :=
by
  sorry

end initial_assessed_value_l386_386340


namespace trig_ordering_l386_386197

theorem trig_ordering (x : ℝ) (hx1 : 1 < x) (hx2 : x < π / 2)
  (h1 : 0 < Real.sin x) (h2 : Real.sin x < 1)
  (h3 : 0 < Real.cos x) (h4 : Real.cos x < 1)
  (h5 : Real.tan x = Real.sin x / Real.cos x)
  (h6 : Real.cot x = Real.cos x / Real.sin x) :
  Real.cos x < Real.cot x ∧
  Real.cot x < Real.sin x ∧
  Real.sin x < Real.tan x :=
by
  sorry

end trig_ordering_l386_386197


namespace smallest_positive_integer_n_l386_386271

theorem smallest_positive_integer_n (n : ℕ) (cube : Finset (Fin 8)) :
    (∀ (coloring : Finset (Fin 8)), 
      coloring.card = n → 
      ∃ (v : Fin 8), 
        (∀ (adj : Finset (Fin 8)), adj.card = 3 → adj ⊆ cube → v ∈ adj → adj ⊆ coloring)) 
    ↔ n = 5 := 
by
  sorry

end smallest_positive_integer_n_l386_386271


namespace total_boys_in_camp_l386_386008

theorem total_boys_in_camp (T : ℕ) (h1 : 0.20 * T = (number_boys_A : ℕ))
                          (h2 : 0.70 * number_boys_A = 49) : T = 350 :=
by
  sorry

end total_boys_in_camp_l386_386008


namespace dandelion_seeds_percentage_approx_29_27_l386_386206

/-
Mathematical conditions:
- Carla has the following set of plants and seeds per plant:
  - 6 sunflowers with 9 seeds each
  - 8 dandelions with 12 seeds each
  - 4 roses with 7 seeds each
  - 10 tulips with 15 seeds each.
- Calculate:
  - total seeds
  - percentage of seeds from dandelions
-/ 

def num_sunflowers : ℕ := 6
def num_dandelions : ℕ := 8
def num_roses : ℕ := 4
def num_tulips : ℕ := 10

def seeds_per_sunflower : ℕ := 9
def seeds_per_dandelion : ℕ := 12
def seeds_per_rose : ℕ := 7
def seeds_per_tulip : ℕ := 15

def total_sunflower_seeds : ℕ := num_sunflowers * seeds_per_sunflower
def total_dandelion_seeds : ℕ := num_dandelions * seeds_per_dandelion
def total_rose_seeds : ℕ := num_roses * seeds_per_rose
def total_tulip_seeds : ℕ := num_tulips * seeds_per_tulip

def total_seeds : ℕ := total_sunflower_seeds + total_dandelion_seeds + total_rose_seeds + total_tulip_seeds

def percentage_dandelion_seeds : ℚ := (total_dandelion_seeds : ℚ) / total_seeds * 100

theorem dandelion_seeds_percentage_approx_29_27 : abs (percentage_dandelion_seeds - 29.27) < 0.01 :=
sorry

end dandelion_seeds_percentage_approx_29_27_l386_386206


namespace container_volumes_l386_386299

theorem container_volumes (a r : ℝ) (h1 : (2 * a)^3 = (4 / 3) * Real.pi * r^3) :
  ((2 * a + 2)^3 > (4 / 3) * Real.pi * (r + 1)^3) :=
by sorry

end container_volumes_l386_386299


namespace ages_when_john_is_50_l386_386800

variable (age_john age_alice age_mike : ℕ)

-- Given conditions:
-- John is 10 years old
def john_is_10 : age_john = 10 := by sorry

-- Alice is twice John's age
def alice_is_twice_john : age_alice = 2 * age_john := by sorry

-- Mike is 4 years younger than Alice
def mike_is_4_years_younger : age_mike = age_alice - 4 := by sorry

-- Prove that when John is 50 years old, Alice will be 60 years old, and Mike will be 56 years old
theorem ages_when_john_is_50 : age_john = 50 → age_alice = 60 ∧ age_mike = 56 := 
by 
  intro h
  sorry

end ages_when_john_is_50_l386_386800


namespace expression_value_l386_386732

theorem expression_value
  (x y z : ℝ)
  (hx : x = -5 / 4)
  (hy : y = -3 / 2)
  (hz : z = Real.sqrt 2) :
  -2 * x ^ 3 - y ^ 2 + Real.sin z = 53 / 32 + Real.sin (Real.sqrt 2) :=
by
  rw [hx, hy, hz]
  sorry

end expression_value_l386_386732


namespace total_sum_spent_l386_386866

theorem total_sum_spent (b gift : ℝ) (friends tanya : ℕ) (extra_payment : ℝ)
  (h1 : friends = 10)
  (h2 : tanya = 1)
  (h3 : extra_payment = 3)
  (h4 : gift = 15)
  (h5 : b = 270)
  : (b + gift) = 285 :=
by {
  -- Given:
  -- friends = 10 (number of dinner friends),
  -- tanya = 1 (Tanya who forgot to pay),
  -- extra_payment = 3 (extra payment by each of the remaining 9 friends),
  -- gift = 15 (cost of the gift),
  -- b = 270 (total bill for the dinner excluding the gift),

  -- We need to prove:
  -- total sum spent by the group is $285, i.e., (b + gift) = 285

  sorry 
}

end total_sum_spent_l386_386866


namespace max_S_sq_plus_T_sq_l386_386014

/-- Problem Description: In quadrilateral ABCD, AB = sqrt(3), AD = DC = CB = 1. 
    The areas of triangles ABD and BCD are S and T, respectively.
    Prove that the maximum value of S^2 + T^2 is 7/8. -/
theorem max_S_sq_plus_T_sq (AB AD DC CB : ℝ) (S T : ℝ) (h1 : AB = real.sqrt 3) 
  (h2 : AD = 1) (h3 : DC = 1) (h4 : CB = 1) :
  ∃ θ : ℝ, (S = (real.sqrt 3 / 2) * real.sin θ) ∧ 
           (T = 1 / 2 * real.sqrt (4 - 2 * real.sqrt 3 * real.cos θ) * real.sin (real.acos (real.cos θ / 2))) ∧
           (S^2 + T^2 ≤ 7 / 8) :=
sorry

end max_S_sq_plus_T_sq_l386_386014


namespace like_terms_exponent_equality_l386_386297

theorem like_terms_exponent_equality (a b : ℕ) (x y : ℕ) :
  (x : ℝ)^(a+1) * (y : ℝ)^2 = -2 * (x : ℝ)^3 * (y : ℝ)^b → a^b = 4 := 
by
  intros h
  -- Extracting the equality constraints from like terms
  have h1 : a + 1 = 3 := by sorry -- Given by the fact that exponents of x must be equal
  have h2 : b = 2 := by sorry -- Given by the fact that exponents of y must be equal
  -- Actual computation of the final result
  have ha : a = 2 := by linarith
  rw [ha, h2]
  exact pow_lt_pow_of_lt_right (by decide) (@two_pos ℝ _)

end like_terms_exponent_equality_l386_386297


namespace problem_statement_l386_386668

theorem problem_statement :
  (∃ n : ℕ, n = 8 ∧ ∀ a : ℕ, a < 15 → (∃ x : ℤ, a * x ≡ 1 [MOD 15]) ↔ gcd a 15 = 1) :=
by
  use 8
  intro a
  intro ha
  split
  sorry

end problem_statement_l386_386668


namespace range_of_a_l386_386745

noncomputable def f (a x : ℝ) := a * Real.log x + x - 1

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, 1 ≤ x → f a x ≥ 0) : a ≥ -1 := by
  sorry

end range_of_a_l386_386745


namespace part_a_min_bad_chips_part_b_min_bad_chips_l386_386509

/-- Problem (a): Determine the minimum possible number of bad chips. -/
theorem part_a_min_bad_chips :
  let n := 2021 in
  ∀ chips_colors : Fin (n*n) → Fin n,
  ∃ bad_chips : Fin (n*n) → Bool,
  (∀ chip, bad_chips chip = true ↔
    (∃ left_count right_count,
      left_count % 2 = 1 ∧ right_count % 2 = 1)) ∧
  ∀ arrangement : Fin (n*n),
  ∃ bad_count,
  (bad_count = 1010) :=
by
  sorry

/-- Problem (b): Given the additional condition that each chip must have at least one adjacent chip of the same color, determine the minimum possible number of bad chips. -/
theorem part_b_min_bad_chips :
  let n := 2021 in
  ∀ chips_colors : Fin (n*n) → Fin n,
  ∃ bad_chips : Fin (n*n) → Bool,
  (∀ chip, bad_chips chip = true ↔
    (∃ left_count right_count,
      left_count % 2 = 1 ∧ right_count % 2 = 1)) ∧
  (∀ chips_adj : Fin (n*n) → (Fin (n*n)) → Bool,
    (∀ chip, (chips_adj chip (chip + 1) = true) ∧ (chips_adj chip (chip - 1) = true))) ∧
  ∀ arrangement : Fin (n*n),
  ∃ bad_count,
  (bad_count = 3030) :=
by
  sorry

end part_a_min_bad_chips_part_b_min_bad_chips_l386_386509


namespace star_m_eq_33_l386_386065

def digit_sum (x : ℕ) : ℕ := x.digits.sum

def S_set : set ℕ := {n | digit_sum n = 15 ∧ n < 10^8}

def m : ℕ := S_set.size

theorem star_m_eq_33 : digit_sum m = 33 :=
by
  sorry

end star_m_eq_33_l386_386065


namespace contribution_per_person_l386_386153

theorem contribution_per_person (total_amount : ℕ) (num_participants : ℕ) (amount_per_person : ℕ) :
  total_amount = 2400 → num_participants = 9 → amount_per_person = 267 → 
  (total_amount / num_participants) = amount_per_person :=
by
  intro h1 h2 h3
  rw [h1, h2]
  exact h3

end contribution_per_person_l386_386153


namespace fg_of_2_eq_513_l386_386402

def f (x : ℤ) : ℤ := x^3 + 1
def g (x : ℤ) : ℤ := 3*x + 2

theorem fg_of_2_eq_513 : f (g 2) = 513 := by
  sorry

end fg_of_2_eq_513_l386_386402


namespace transformed_mean_variance_l386_386734

variables {n : ℕ} {x : Fin n → ℝ} 

-- Mean of initial data
def mean (x : Fin n → ℝ) : ℝ := (∑ i, x i) / n

-- Variance of initial data
def variance (x : Fin n → ℝ) : ℝ := (∑ i, (x i - mean x) ^ 2) / n

theorem transformed_mean_variance (x : Fin n → ℝ) (mean_x : ℝ) (var_x : ℝ)
  (hmean : mean x = mean_x) (hvar : variance x = var_x) :
  mean (λ i, 2 * x i + 3) = 2 * mean_x + 3 ∧
  variance (λ i, 2 * x i + 3) = 4 * var_x :=
by
  sorry

end transformed_mean_variance_l386_386734


namespace false_statement_not_parallel_l386_386702

-- Definitions of lines m, n and planes α, β 
variables (m n : Line) (α β : Plane)

-- Given conditions
axiom lines_are_distinct : m ≠ n
axiom planes_are_distinct : α ≠ β

-- False statement to be proven
theorem false_statement_not_parallel (h1 : m ∥ α) (h2 : α ∩ β = n) : ¬ (m ∥ n) :=
sorry

end false_statement_not_parallel_l386_386702


namespace geometric_arithmetic_sequence_problem_l386_386287

theorem geometric_arithmetic_sequence_problem :
  ∃ (a n q : ℝ) (b : ℕ → ℝ), q ≠ 1 ∧
    (∀ n, a n = 4 * (-1/2)^(n-1)) ∧
    (b 1 = a 3) ∧
    (∑ i in range 7, b (i + 1) = 49) ∧
    (∀ n, (1 / (b 1 * b 2) + 1 / (b 2 * b 3) + ... + 1 / (b n * b (n+1))) = n / (2 * n + 1)) :=
sorry

end geometric_arithmetic_sequence_problem_l386_386287


namespace polygon_sides_l386_386347

-- Definitions of the conditions
def is_regular_polygon (n : ℕ) (int_angle ext_angle : ℝ) : Prop :=
  int_angle = 5 * ext_angle ∧ (int_angle + ext_angle = 180)

-- Main theorem statement
theorem polygon_sides (n : ℕ) (int_angle ext_angle : ℝ) :
  is_regular_polygon n int_angle ext_angle →
  (ext_angle = 360 / n) →
  n = 12 :=
sorry

end polygon_sides_l386_386347


namespace trig_identity_l386_386719

theorem trig_identity (x : ℝ) (h : Real.cos (x - π / 3) = 1 / 3) :
  Real.cos (2 * x - 5 * π / 3) + Real.sin (π / 3 - x)^2 = 5 / 3 :=
by
  sorry

end trig_identity_l386_386719


namespace effective_weighted_average_yield_l386_386154

-- Definitions for the problem conditions
def yield_A := 0.21
def tax_A := 0.10
def after_tax_yield_A := yield_A * (1 - tax_A)

def yield_B := 0.15
def tax_B := 0.20
def after_tax_yield_B := yield_B * (1 - tax_B)

def investment_A := 10000
def investment_B := 15000
def total_investment := investment_A + investment_B

def weight_A := investment_A / total_investment
def weight_B := investment_B / total_investment

-- The theorem to prove
theorem effective_weighted_average_yield : 
  (weight_A * after_tax_yield_A + weight_B * after_tax_yield_B) = 0.1476 := 
by
  sorry

end effective_weighted_average_yield_l386_386154


namespace triangle_EF_value_l386_386367

variable (D E EF : ℝ)
variable (DE : ℝ)

theorem triangle_EF_value (h₁ : cos (2 * D - E) + sin (D + E) = 2) (h₂ : DE = 6) : EF = 3 :=
sorry

end triangle_EF_value_l386_386367


namespace log_cos2_range_l386_386144

theorem log_cos2_range (x : ℝ) (h : 0 < x ∧ x < π) : 
  ∃ y, y = real.log2 (real.cos x ^ 2) ∧ y ∈ set.Iic (0) := sorry

end log_cos2_range_l386_386144


namespace initial_men_count_l386_386470

theorem initial_men_count {M : ℕ} (h1 : ∃ F, F / (45 * M) = F / (32.73 * (M + 15))) : M = 40 := by
  sorry

end initial_men_count_l386_386470


namespace train_cross_time_l386_386946

noncomputable def time_to_cross (length : ℕ) (speed_kmh : ℕ) : ℕ :=
  let speed_m_s := speed_kmh * 5 / 18
  in length / speed_m_s

theorem train_cross_time (length : ℕ) (speed_kmh : ℕ) (conversion_factor : ℚ) (time : ℕ) (H1 : length = 200) (H2 : speed_kmh = 144) (H3 : conversion_factor = 5 / 18) (H4 : time = length / (speed_kmh * conversion_factor)) :
  time = 5 :=
by
  rw [H1, H2, H3] at H4
  norm_num at H4
  assumption

end train_cross_time_l386_386946


namespace b_55_div_56_rem_0_l386_386071

def b_n (n : ℕ) : ℕ :=
  (List.finRange (n + 1)).foldl (λ acc x => acc * (10 ^ x.digits.length) + (x + 1)) 0

theorem b_55_div_56_rem_0 : (b_n 55) % 56 = 0 :=
by
  sorry

end b_55_div_56_rem_0_l386_386071


namespace chosen_number_is_129_l386_386188

theorem chosen_number_is_129 (x : ℕ) (h : 2 * x - 148 = 110) : x = 129 :=
by
  sorry

end chosen_number_is_129_l386_386188


namespace eggs_leftover_l386_386221

theorem eggs_leftover (d e f : ℕ) (total_eggs_per_carton : ℕ) 
  (h_d : d = 53) (h_e : e = 65) (h_f : f = 26) (h_carton : total_eggs_per_carton = 15) : (d + e + f) % total_eggs_per_carton = 9 :=
by {
  sorry
}

end eggs_leftover_l386_386221


namespace sin_theta_value_l386_386754

theorem sin_theta_value {θ : ℝ} (h₁ : 9 * (Real.tan θ)^2 = 4 * Real.cos θ) (h₂ : 0 < θ ∧ θ < Real.pi) : 
  Real.sin θ = 1 / 3 :=
by
  sorry

end sin_theta_value_l386_386754


namespace differences_form_arithmetic_progression_l386_386980

def is_arithmetic_progression (seq : ℕ → ℤ) : Prop :=
  ∃ A B : ℤ, ∀ n : ℕ, seq n = A * (n : ℤ) + B

def second_difference (g : ℕ → ℤ) (n : ℕ) : ℤ :=
  g (n + 2) - 2 * g (n + 1) + g n

theorem differences_form_arithmetic_progression
  (f ϕ : ℕ → ℤ)
  (h : ∀ n : ℕ, second_difference f n = second_difference ϕ n) :
  is_arithmetic_progression (λ n, f n - ϕ n) :=
sorry

end differences_form_arithmetic_progression_l386_386980


namespace total_sprockets_produced_in_8h_l386_386824

noncomputable def total_sprockets_produced (A : ℕ) : ℕ :=
  let machine_P_rate := A
  let machine_Q_rate := 0.75 * 1.1 * A
  let machine_R_rate := 0.8 * A
  in 8 * machine_P_rate + 8 * machine_Q_rate + 8 * machine_R_rate

theorem total_sprockets_produced_in_8h (A : ℕ) : total_sprockets_produced A = 21 * A :=
by
  sorry

end total_sprockets_produced_in_8h_l386_386824


namespace arc_length_of_sector_l386_386759

noncomputable def central_angle := 36
noncomputable def radius := 15

theorem arc_length_of_sector : (central_angle * Real.pi * radius / 180 = 3 * Real.pi) :=
by
  sorry

end arc_length_of_sector_l386_386759


namespace midpoint_distance_from_line_l386_386953

-- Given condition definitions
def parabola_eq : ℝ → ℝ → Prop := λ x y, y^2 = x
def focal_chord_length : ℝ := 4
def line_eq : ℝ → Prop := λ x, x + (1 / 2) = 0

-- The question restated as a Lean theorem
theorem midpoint_distance_from_line 
  (A B : ℝ × ℝ) 
  (mid_AB : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (hA : parabola_eq A.1 A.2) 
  (hB : parabola_eq B.1 B.2) 
  (hAB : dist A B = focal_chord_length)
  : dist mid_AB (-(1 / 2), 0) = 9 / 4 :=
sorry

end midpoint_distance_from_line_l386_386953


namespace complex_quadrant_l386_386023

theorem complex_quadrant :
  let z := (3 - complex.i) / (1 + complex.i ^ 2023)
  (z.re > 0) ∧ (z.im > 0) :=
by
  let z := (3 - complex.i) / (1 + complex.i ^ 2023)
  have h1 : z = 2 + complex.i :=
    sorry
  show (2 : ℝ) > 0 ∧ (1 : ℝ) > 0 from sorry

end complex_quadrant_l386_386023


namespace simplify_expression_l386_386146

theorem simplify_expression :
  2 + 1 / (2 + 1 / (2 + 1 / 2)) = 29 / 12 :=
by
  sorry  -- Proof will be provided here

end simplify_expression_l386_386146


namespace max_algebraic_expressions_l386_386149

variable {a b : ℝ}

theorem max_algebraic_expressions (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) :
  ¬(∀ x y : ℝ, x = ((a + 1/a)*(b + 1/b)) ∧ y = (sqrt(ab) + 1/sqrt(ab))^2 → x ≥ y ∨ y ≥ x) :=
sorry

end max_algebraic_expressions_l386_386149


namespace sum_of_squares_l386_386063

variable (n : ℕ)

def S (n : ℕ) := 3^n - 1

def a : ℕ → ℕ
| 0       := 2
| (n + 1) := 2 * 3^n

theorem sum_of_squares (n : ℕ) :
  (∑ k in Finset.range (n + 1), (a k)^2) = (1 / 2 : ℝ) * (9^n - 1) := 
sorry

end sum_of_squares_l386_386063


namespace negation_of_existential_l386_386476

def divisible_by (n x : ℤ) := ∃ k : ℤ, x = k * n
def odd (x : ℤ) := ∃ k : ℤ, x = 2 * k + 1

def P (x : ℤ) := divisible_by 7 x ∧ ¬ odd x

theorem negation_of_existential :
  (¬ ∃ x : ℤ, P x) ↔ ∀ x : ℤ, divisible_by 7 x → odd x :=
by
  sorry

end negation_of_existential_l386_386476


namespace min_dot_product_proof_l386_386312

noncomputable def min_dot_product : ℝ :=
3 - 2 * real.sqrt 3

theorem min_dot_product_proof :
  ∀ O P F : ℝ × ℝ,
    (∃ a : ℝ, a > 0 ∧ (4 * O.2 = O.1 ^ 2) ∧ (a^2 + 1 = 4) ∧
    (O = (0, 0)) ∧ (F = (0, 2)) ∧
    (F.2^2 / a^2 - F.1^2 = 1 ∧
     P ∈ {pt : ℝ × ℝ | pt.2^2 / 3 - pt.1^2 = 1} ∧
     pt.1 ≥ 0) ∧
    (n : ℝ, n ≥ real.sqrt 3 →
    ∃ P : ℝ × ℝ, P.1 ^ 2 = (1 / 3) * n^2 - 1 ∧ P.2 = n)) →
  (∃ n : ℝ, n = real.sqrt 3 ∧
  ∀ m n : ℝ, m = 0 ∧ n = real.sqrt 3 →
  (m * (m + n - 2) = 3 - 2 * real.sqrt 3)) := sorry

end min_dot_product_proof_l386_386312


namespace triangle_sides_l386_386368

theorem triangle_sides
  (D E : ℝ) (DE EF : ℝ)
  (h1 : Real.cos (2 * D - E) + Real.sin (D + E) = 2)
  (hDE : DE = 6) :
  EF = 3 :=
by
  -- Proof is omitted
  sorry

end triangle_sides_l386_386368


namespace central_angle_of_chord_eq_radius_l386_386011

theorem central_angle_of_chord_eq_radius (r : ℝ) (h : r > 0) : 
  ∃ θ : ℝ, θ = real.pi / 3 ∧ 
  ∃ (C : Type) [metric_space C] [normed_group C] [normed_space ℝ C] (center : C) (radius : ℝ), 
  radius = r ∧
  ∃ (chord : C) (arc_endpoint1 : C) (arc_endpoint2 : C), 
  dist center arc_endpoint1 = r ∧ 
  dist center arc_endpoint2 = r ∧
  dist arc_endpoint1 arc_endpoint2 = r ∧
  ∃ θ_center : ℝ, θ_center = θ := 
sorry

end central_angle_of_chord_eq_radius_l386_386011


namespace blue_pens_removed_l386_386176

def initial_blue_pens := 9
def initial_black_pens := 21
def initial_red_pens := 6
def removed_black_pens := 7
def pens_left := 25

theorem blue_pens_removed (x : ℕ) :
  initial_blue_pens - x + (initial_black_pens - removed_black_pens) + initial_red_pens = pens_left ↔ x = 4 := 
by 
  sorry

end blue_pens_removed_l386_386176


namespace value_of_y_in_arithmetic_sequence_l386_386532

theorem value_of_y_in_arithmetic_sequence :
    ∃ y : ℤ, (arithmetic_sequence (3^2) y (3^4)) ∧ y = 45 := by
  -- Here we define the arithmetic sequence condition.
  def arithmetic_sequence (a b c : ℤ) : Prop := b = (a + c) / 2
  sorry

end value_of_y_in_arithmetic_sequence_l386_386532


namespace probability_sum_of_squares_in_interval_l386_386140

theorem probability_sum_of_squares_in_interval :
  let S := {(x, y) : ℝ × ℝ | 0 ≤ x ∧ x ≤ 2 ∧ 0 ≤ y ∧ y ≤ 2 ∧ 0 ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ 2} in
  (volume S) / (volume {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2}) = π / 8 :=
sorry

end probability_sum_of_squares_in_interval_l386_386140


namespace measure_of_angle_BCE_is_45_l386_386979

variable (A B C D E : Type)
variable [triangle C D E]
variable [square A B C D]
variable (angle_BCE : ℚ)
variable (angle_DEC : ℚ) (angle_DEE : ℚ)

noncomputable def measure_angle_BCE (angle_DEC angle_DCE : ℚ) : ℚ :=
  let angle_BCD : ℚ := 90
  180 - (angle_DEC + angle_BCD)

theorem measure_of_angle_BCE_is_45 (h1 : angle_DEC = 45)
    (h2 : angle_DEE = 45) (h3 : ∃ (DE CE : ℚ), DE = CE) :
    measure_angle_BCE A B C D E angle_DEC angle_DEC = 45 := by
  sorry

end measure_of_angle_BCE_is_45_l386_386979


namespace kite_area_overlap_l386_386911

theorem kite_area_overlap (beta : Real) (h_beta : beta ≠ 0 ∧ beta ≠ π) : 
  ∃ (A : Real), A = 1 / Real.sin beta := by
  sorry

end kite_area_overlap_l386_386911


namespace waiter_date_trick_l386_386194

theorem waiter_date_trick :
  ∃ d₂ : ℕ, ∃ x : ℝ, 
  (∀ d₁ : ℕ, ∀ x : ℝ, x + d₁ = 168) ∧
  3 * x + d₂ = 486 ∧
  3 * (x + d₂) = 516 ∧
  d₂ = 15 :=
by
  sorry

end waiter_date_trick_l386_386194


namespace solve_inequality_l386_386223

theorem solve_inequality : { x : ℝ | 2 * x^2 + 4 * x - 6 < 0 } = set.Ioo (-3 : ℝ) 1 :=
sorry

end solve_inequality_l386_386223


namespace man_speed_in_still_water_l386_386941

theorem man_speed_in_still_water (upstream_speed downstream_speed : ℝ) (h1 : upstream_speed = 25) (h2 : downstream_speed = 45) :
  (upstream_speed + downstream_speed) / 2 = 35 :=
by
  sorry

end man_speed_in_still_water_l386_386941


namespace students_just_passed_l386_386017

theorem students_just_passed (total_students : ℕ) (perc_first_div : ℝ) (perc_second_div : ℝ) :
    total_students = 300 →
    perc_first_div = 0.27 →
    perc_second_div = 0.54 →
    300 - (total_students * perc_first_div).to_nat - (total_students * perc_second_div).to_nat = 57 :=
begin
  sorry
end

end students_just_passed_l386_386017


namespace value_of_f_neg_two_l386_386813

noncomputable def f : ℝ → ℝ :=
  sorry

axiom odd_func (x : ℝ) : f(-x) = -f(x)

axiom f_pos_def (x : ℝ) (hx : x > 0) : f(x) = Real.log x / Real.log 2

theorem value_of_f_neg_two : f(-2) = -1 := sorry

end value_of_f_neg_two_l386_386813


namespace tangent_circles_distance_l386_386503

/--
Given:
1. Points A and B are centers of two externally tangent circles.
2. The radius of the circle centered at A is 7.
3. The radius of the circle centered at B is 4.
4. A line tangent to both circles intersects the line segment AB at point C.

Prove that the distance from point B to point C (BC) is 44/3.
-/
theorem tangent_circles_distance
  (A B C : Point) -- Points A, B, and C
  (rA rB : ℝ) -- Radii of circles centered at A and B
  (h_tangent : external_tangent A B rA rB) -- A and B are centers of externally tangent circles
  (h_radiusA : rA = 7) -- Radius of the circle centered at A is 7
  (h_radiusB : rB = 4) -- Radius of the circle centered at B is 4
  (h_tangent_line : tangent_line_to_both_circles C A B) -- A line tangent to both circles intersects AB at C
  : dist B C = 44 / 3 :=
sorry

end tangent_circles_distance_l386_386503


namespace frustum_lateral_not_parallel_l386_386151

noncomputable def frustum (P : Type*) [pyramid P] : Prop :=
∀ (F : face P), (F.parallel_to_base → F.is_frustum)

theorem frustum_lateral_not_parallel {P : Type*} [pyramid P]:
  frustum P →
  (∀ B1 B2 : base P, B1 ∼ B2) →
  (∀ F : face P, F.is_trapezoid) →
  (∀ L1 L2 : lateral_edge P, extensions_intersect_at_point L1 L2) →
  ¬ (∀ L1 L2 : lateral_edge P, parallel L1 L2) :=
sorry

end frustum_lateral_not_parallel_l386_386151


namespace ab_value_in_right_triangle_l386_386365

theorem ab_value_in_right_triangle (A B C : Type)
  [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C]
  (hA : ∠A = 90°) (BC : ℝ := 15) (tanC_eq_3sinC : ∀ (C : ℝ), tan C = 3 * sin C) :
  ∃ AB : ℝ, AB = (5 * Real.sqrt 6) / 4 := by
  sorry

end ab_value_in_right_triangle_l386_386365


namespace apples_handed_out_to_students_l386_386445

def initial_apples : ℕ := 47
def apples_per_pie : ℕ := 4
def number_of_pies : ℕ := 5
def apples_for_pies : ℕ := number_of_pies * apples_per_pie

theorem apples_handed_out_to_students : 
  initial_apples - apples_for_pies = 27 := 
by
  -- Since 20 apples are used for pies and there were initially 47 apples,
  -- it follows that 27 apples were handed out to students.
  sorry

end apples_handed_out_to_students_l386_386445


namespace floor_of_factorial_ratio_l386_386624

-- The statement of the problem
theorem floor_of_factorial_ratio :
  (Real.floor ((↑(Nat.factorial 2010) + ↑(Nat.factorial 2007)) / 
               (↑(Nat.factorial 2009) + ↑(Nat.factorial 2008))) = 2009) :=
by
  sorry

end floor_of_factorial_ratio_l386_386624


namespace rhombus_perimeter_l386_386465

-- Define the rhombus with diagonals of given lengths and prove the perimeter is 52 inches
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) (d1_pos : 0 < d1) (d2_pos : 0 < d2):
  let s := sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2) in 
  let perimeter := 4 * s in
  perimeter = 52 
  :=
by
  sorry  -- Proof is skipped

end rhombus_perimeter_l386_386465


namespace larger_solution_of_quadratic_larger_root_is_14_l386_386685

theorem larger_solution_of_quadratic :
  ∀ x : ℝ, (2 * x^2 - 14 * x - 84 = 0) → (x = 14 ∨ x = -3) :=
begin
  intros x h,
  -- Simplicity through factoring
  have h2 : x^2 - 7 * x - 42 = 0,
  { rw ← eq_div_iff_mul_eq (by norm_num : (2 : ℝ) ≠ 0) at h,
    exact h },
  -- Factoring the quadratic equation
  have h3 : (x - 14) * (x + 3) = 0,
  { apply eq_of_sub_eq_zero,
    ring_nf at h2,
    exact h2 },
  -- Solving the factored equation
  cases (mul_eq_zero.mp h3) with h4 h5,
  { exact or.inl h4 },
  { exact or.inr h5 },
  admit
end

theorem larger_root_is_14 :
  ∀ x : ℝ, (2 * x^2 - 14 * x - 84 = 0) → x = 14 :=
begin
  intros x h,
  have h_solution : x = 14 ∨ x = -3 := larger_solution_of_quadratic x h,
  cases h_solution,
  { exact h_solution },
  { exfalso,
    linarith }
end

end larger_solution_of_quadratic_larger_root_is_14_l386_386685


namespace arithmetic_sequence_middle_term_l386_386530

theorem arithmetic_sequence_middle_term :
  let a1 := 3^2
  let a3 := 3^4
  let y := (a1 + a3) / 2
  y = 45 :=
by
  let a1 := (3:ℕ)^2
  let a3 := (3:ℕ)^4
  let y := (a1 + a3) / 2
  have : a1 = 9 := by norm_num
  have : a3 = 81 := by norm_num
  have : y = 45 := by norm_num
  exact this

end arithmetic_sequence_middle_term_l386_386530


namespace log_four_sixtyfour_sqrt_two_l386_386233

theorem log_four_sixtyfour_sqrt_two :
  log 4 (64 * Real.sqrt 2) = 13 / 4 := by
sorry

end log_four_sixtyfour_sqrt_two_l386_386233


namespace final_surface_area_l386_386959

theorem final_surface_area 
  (original_cube_volume : ℕ)
  (small_cube_volume : ℕ)
  (remaining_cubes : ℕ)
  (removed_cubes : ℕ)
  (per_face_expose_area : ℕ)
  (initial_surface_area_per_cube : ℕ)
  (total_cubes : ℕ)
  (shared_internal_faces_area : ℕ)
  (final_surface_area : ℕ) :
  original_cube_volume = 12 * 12 * 12 →
  small_cube_volume = 3 * 3 * 3 →
  total_cubes = 64 →
  removed_cubes = 14 →
  remaining_cubes = total_cubes - removed_cubes →
  initial_surface_area_per_cube = 6 * 3 * 3 →
  per_face_expose_area = 6 * 4 →
  final_surface_area = remaining_cubes * (initial_surface_area_per_cube + per_face_expose_area) - shared_internal_faces_area →
  (remaining_cubes * (initial_surface_area_per_cube + per_face_expose_area) - shared_internal_faces_area) = 2820 :=
sorry

end final_surface_area_l386_386959


namespace power_increased_by_four_l386_386599

-- Definitions from the conditions
variables (F k v : ℝ) (initial_force_eq_resistive : F = k * v)

-- Define the new conditions with double the force
variables (new_force : ℝ) (new_velocity : ℝ) (new_force_eq_resistive : new_force = k * new_velocity)
  (doubled_force : new_force = 2 * F)

-- The theorem statement
theorem power_increased_by_four (initial_force_eq_resistive : F = k * v) 
  (new_force_eq_resistive : new_force = k * new_velocity)
  (doubled_force : new_force = 2 * F) :
  new_velocity = 2 * v → 
  (new_force * new_velocity) = 4 * (F * v) :=
sorry

end power_increased_by_four_l386_386599


namespace sum_g_equals_half_l386_386393

def g (n : ℕ) : ℝ := ∑' k : ℕ, if k ≥ n + 2 then 1 / (k : ℝ) ^ n else 0

theorem sum_g_equals_half : ∑' n, g n = 1 / 2 := by
  sorry

end sum_g_equals_half_l386_386393


namespace melissa_annual_driving_hours_l386_386078

noncomputable theory -- Use noncomputable for handling non-computable definitions

-- Define the conditions
def hours_per_trip : ℕ := 3
def trips_per_month : ℕ := 2
def months_per_year : ℕ := 12

-- Define the monthly driving hours
def monthly_driving_hours := hours_per_trip * trips_per_month

-- Prove the number of hours Melissa spends driving in a year
theorem melissa_annual_driving_hours : monthly_driving_hours * months_per_year = 72 := sorry

end melissa_annual_driving_hours_l386_386078


namespace Exists_Point_Outside_Line_l386_386950

theorem Exists_Point_Outside_Line (A B C : Point) (h1 : A ≠ B) (h2 : B ≠ C) (h3 : C ≠ A) (l : Line) (hA : A ∈ l) (hB : B ∈ l) (hC : C ∈ l) :
  ∃ P : Point, P ∉ l ∧ ∠APB = ∠BPC :=
sorry

end Exists_Point_Outside_Line_l386_386950


namespace complement_M_l386_386823

def U : Set ℕ := {1, 2, 3, 4}

def M : Set ℕ := {x | (x - 1) * (x - 4) = 0}

theorem complement_M :
  (U \ M) = {2, 3} := by
  sorry

end complement_M_l386_386823


namespace number_of_non_empty_subsets_of_M_inter_N_l386_386317

open Set

def M : Set ℤ := {x | -2 < x ∧ x ≤ 2}
def N : Set ℕ := {y | ∃ x ∈ M, y = x * x}
def common_elements : Set ℕ := M ∩ N

axiom count_non_empty_subsets_common_elements_eq_3 : 
  ∃ (n : ℕ), (n = size (common_elements.to_finset) - 1) = 3

theorem number_of_non_empty_subsets_of_M_inter_N : (∃ (n : ℕ), (n = 2^size (common_elements.to_finset) - 1)) := 
by 
  -- The proof will be filled in here
  sorry

end number_of_non_empty_subsets_of_M_inter_N_l386_386317


namespace count_ordered_pairs_l386_386751

theorem count_ordered_pairs : 
  {p : ℝ × ℝ | let x := p.1, let y := p.2 in 
                x + 2 * y = 4 ∧ abs (abs x - 2 * abs y) = 2}.count = 2 := 
by
  sorry

end count_ordered_pairs_l386_386751


namespace limit_T_div_S_l386_386693

open Real

noncomputable def S (a : ℝ) : ℝ :=
  (1/3) * a^3

noncomputable def T (a : ℝ) : ℝ :=
  let b := (2 * sqrt 3 * a - 1) / (2 * a + sqrt 3) - a
  ((a - b)^3) / 6

theorem limit_T_div_S (a : ℝ) (h : 0 < a) :
  tendsto (λ a, T a / S a) at_top (𝓝 4) :=
sorry

end limit_T_div_S_l386_386693


namespace days_to_fulfill_order_l386_386913

theorem days_to_fulfill_order (bags_per_batch : ℕ) (total_order : ℕ) (initial_bags : ℕ) (required_days : ℕ) :
  bags_per_batch = 10 →
  total_order = 60 →
  initial_bags = 20 →
  required_days = (total_order - initial_bags) / bags_per_batch →
  required_days = 4 :=
by
  intros
  sorry

end days_to_fulfill_order_l386_386913


namespace opposite_greater_self_l386_386601

theorem opposite_greater_self :
  ∃ x ∈ ({-2023, 0, 1/2023, 2023} : Set ℝ), -x > x :=
by 
  use -2023
  simp
  linarith

end opposite_greater_self_l386_386601


namespace distance_Atlanta_NewYork_l386_386410

-- Definitions of points on the complex plane
def NewYork : ℂ := 0
def Miami : ℂ := 3200 * complex.I
def Atlanta : ℂ := 960 + 1280 * complex.I

-- Prove that the distance from Atlanta to New York is 3200
theorem distance_Atlanta_NewYork : complex.abs (Atlanta - NewYork) = 3200 :=
by
  -- Using sorry to skip the proof
  sorry

end distance_Atlanta_NewYork_l386_386410


namespace find_m_l386_386753

noncomputable def m := (30 : ℝ) / 7

theorem find_m (m : ℝ) (h : 15 ^ (5 * m) = (1 / 3) ^ (2 * m - 30)) : m = (30 / 7) :=
by
  sorry

end find_m_l386_386753


namespace least_possible_value_of_T_l386_386387

open Set

-- Mathematical definition of the problem conditions
def valid_set (T : Set ℕ) :=
  T ⊆ (range 15).succ ∧
  T.card = 7 ∧
  (∀ a b, a ∈ T → b ∈ T → a < b → ¬ (b % a = 0)) ∧
  (∀ p, p ∈ T → Nat.Prime p → p ≤ 11)

-- Statement of the proof problem
theorem least_possible_value_of_T (T : Set ℕ) (h : valid_set T) : ∃ n ∈ T, ∀ m ∈ T, n ≤ m := by
  use 2
  sorry

end least_possible_value_of_T_l386_386387


namespace correct_equation_after_moving_digit_l386_386354

theorem correct_equation_after_moving_digit :
  (101 - 102 = 1) →
  101 - 10^2 = 1 :=
by
  intro h
  sorry

end correct_equation_after_moving_digit_l386_386354


namespace largest_integer_b_l386_386633

theorem largest_integer_b (b : ℤ) : (b^2 < 60) → b ≤ 7 :=
by sorry

end largest_integer_b_l386_386633


namespace car_mileage_l386_386580

/-- A line of soldiers 1 mile long is jogging. The drill sergeant, in a car, moving at twice their 
speed, repeatedly drives from the back of the line to the front of the line and back again. 
When each soldier has marched 15 miles, the mileage added to the car, to the nearest mile, is 30 miles. --/
theorem car_mileage (v : ℝ) (soldier_distance: ℝ) (line_length: ℝ)
  (hv : v > 0) (hsoldier_distance : soldier_distance = 15) (hline_length: line_length = 1) :
  let car_speed := 2 * v in
  let trip_distance := line_length * 2 in
  let total_trips := soldier_distance / line_length in
  let total_car_distance := total_trips * trip_distance in
  total_car_distance ≈ 30 := 
by 
  sorry

end car_mileage_l386_386580


namespace real_and_equal_roots_condition_l386_386189

theorem real_and_equal_roots_condition (k : ℝ) : 
  ∀ k : ℝ, (∃ (x : ℝ), 3 * x^2 + 6 * k * x + 9 = 0) ↔ (k = Real.sqrt 3 ∨ k = -Real.sqrt 3) :=
by
  sorry

end real_and_equal_roots_condition_l386_386189


namespace even_or_odd_P_representable_l386_386058

def P_representable (P : ℤ → ℤ) (x : ℤ) : Prop :=
  ∃ a b : ℤ, x = P a - P b

theorem even_or_odd_P_representable (P : ℤ → ℤ)
  (hP_deg : ∃ n : ℕ, n ≥ 4 ∧ (P.degree : with_top ℕ) = n)
  (hP_int_coeffs : ∀ k : ℤ, P.coeff k ∈ ℤ)
  (h_half_P_representable : ∀ N : ℕ, 
    (finset.range (N + 1)).filter (λ x, P_representable P x).card >
    (finset.range (N + 1)).card / 2) :
  (∀ x : ℤ, even x → P_representable P x) ∨
  (∀ x : ℤ, odd x → P_representable P x) :=
sorry

end even_or_odd_P_representable_l386_386058


namespace maximum_value_expression_l386_386762

theorem maximum_value_expression (a b c : ℝ) (h : a^2 + b^2 + c^2 = 9) : 
  (a - b)^2 + (b - c)^2 + (c - a)^2 ≤ 27 :=
sorry

end maximum_value_expression_l386_386762


namespace radius_of_sphere_correct_l386_386595

noncomputable def radius_of_sphere (r : ℝ) : Prop :=
  let diagonal_length : ℝ := 1 / √3 
  let equation_rhs := √3 
  let hypothesis : ℝ := r = √3 * (√3 - 1) / 4 in
  (2 * r + √3 * 2 * r = √3) 

theorem radius_of_sphere_correct (r : ℝ) : radius_of_sphere r → r = √3 * (√3 - 1) / 4 :=
by sorry

end radius_of_sphere_correct_l386_386595


namespace min_blocks_to_remove_l386_386551

theorem min_blocks_to_remove (n : ℕ) (h₁ : n = 59) : ∃ k, ∃ m, (m*m*m ≤ n ∧ n < (m+1)*(m+1)*(m+1)) ∧ k = n - m*m*m ∧ k = 32 :=
by {
  sorry
}

end min_blocks_to_remove_l386_386551


namespace proof_v2_eq_u3_over_r_plus_u_l386_386625

-- Define the conditions and the theorem
variable (O : Point) (r : ℝ) (F G H I : Point) (u v : ℝ)

-- Conditions of the problem
-- Center of the circle
-- radius of the circle
-- Chord FG
-- tangent at F meeting at H
-- I is on FH and FI = GH
-- u is the distance from I to the tangent through G
-- v is the distance from I to the line through the chord FG

axiom circle_center : ∀ (P : Point), (dist O P = r) ↔ (P ∈ Circle O r)
axiom chord_FG : dist F G = g
axiom tangent_FH : tangent (Circle O r) F H
axiom point_I_on_FH : I ∈ line_segment F H
axiom FI_equals_GH : dist F I = dist G H
axiom u_distance : dist I (tangent (Circle O r) G) = u
axiom v_distance : dist I (line_through F G) = v

-- The proof statement
theorem proof_v2_eq_u3_over_r_plus_u : 
  v^2 = u^3 / (r + u) := by
  sorry

end proof_v2_eq_u3_over_r_plus_u_l386_386625


namespace ratio_of_areas_of_squares_l386_386891

theorem ratio_of_areas_of_squares (a b : ℕ) (hC : a = 24) (hD : b = 30) :
  (a^2 : ℚ) / (b^2 : ℚ) = 16 / 25 := 
by
  sorry

end ratio_of_areas_of_squares_l386_386891


namespace brads_running_speed_proof_l386_386827

noncomputable def brads_speed (distance_between_homes : ℕ) (maxwells_speed : ℕ) (maxwells_time : ℕ) (brad_start_delay : ℕ) : ℕ :=
  let distance_covered_by_maxwell := maxwells_speed * maxwells_time
  let distance_covered_by_brad := distance_between_homes - distance_covered_by_maxwell
  let brads_time := maxwells_time - brad_start_delay
  distance_covered_by_brad / brads_time

theorem brads_running_speed_proof :
  brads_speed 54 4 6 1 = 6 := 
by
  unfold brads_speed
  rfl

end brads_running_speed_proof_l386_386827


namespace comb_identity_l386_386755

theorem comb_identity (n : Nat) (h : 0 < n) (h_eq : Nat.choose n 2 = Nat.choose (n-1) 2 + Nat.choose (n-1) 3) : n = 5 := by
  sorry

end comb_identity_l386_386755


namespace right_triangle_hypotenuse_l386_386474

def hypotenuse_length
  (x y : ℝ)
  (ma mb : ℝ)
  (hx : (x^2 / 4) + y^2 = 40)
  (hy : (y^2 / 4) + x^2 = 25)
  : ℝ :=
sorry

theorem right_triangle_hypotenuse
  (x y : ℝ)
  (ma mb : ℝ)
  (hx : (x^2 / 4) + y^2 = 40)
  (hy : (y^2 / 4) + x^2 = 25)
  : hypotenuse_length x y ma mb = 2 * Real.sqrt 13 :=
sorry

end right_triangle_hypotenuse_l386_386474


namespace middle_term_arithmetic_sequence_l386_386540

-- Definitions of the given conditions
def a := 3^2
def c := 3^4

-- Assertion that y is the middle term of the arithmetic sequence a, y, c
theorem middle_term_arithmetic_sequence : 
  let y := (a + c) / 2 in 
  y = 45 :=
by
  -- Since the final proof steps are not needed
  sorry

end middle_term_arithmetic_sequence_l386_386540


namespace max_crosses_fit_in_grid_l386_386961

-- Definition of a grid and a cross
def Cross : Type := Fin 6 × Fin 6

def is_cross (c : Cross) (g : Fin 6 × Fin 11 → bool) : Prop :=
  let (x, y) := c in
  g (x, y) &&
  if x > 0 then g (x - 1, y) && g (x + 1, y) else false &&
  if y > 0 then g (x, y - 1) && g (x, y + 1) else false

-- The main theorem stating the maximum number of crosses that can fit
theorem max_crosses_fit_in_grid (g : Fin 6 × Fin 11 → bool) :
  (∀ (x y : Fin 6) (hx : x < 6) (hy : y < 11), g (x, y) → false ∨ true) →
  (∃ S : set Cross, (∀ c ∈ S, is_cross c g) ∧ S.card = 8) :=
sorry

end max_crosses_fit_in_grid_l386_386961


namespace total_flying_days_l386_386792

-- Definitions for the conditions
def days_fly_south_winter := 40
def days_fly_north_summer := 2 * days_fly_south_winter
def days_fly_east_spring := 60

-- Theorem stating the total flying days
theorem total_flying_days : 
  days_fly_south_winter + days_fly_north_summer + days_fly_east_spring = 180 :=
  by {
    -- This is where we would prove the theorem
    sorry
  }

end total_flying_days_l386_386792


namespace people_not_clients_is_11_l386_386585

-- Define the parameters
variables (E C F P : ℕ)

-- Hypotheses based on the conditions
def H1 : E = 200 := rfl
def H2 : C = 20 := rfl
def H3 : F = 210 := rfl
def H4 : P = 10 := rfl

-- Calculate number of clients and total people
def num_clients : ℕ := E / C
def total_people : ℕ := F / P

-- The final statement to prove
theorem people_not_clients_is_11 (H1 : E = 200) (H2 : C = 20) (H3 : F = 210) (H4 : P = 10) :
  total_people - num_clients = 11 := by
    -- Assign values from hypotheses
    rw [H1, H2, H3, H4]
    -- Calculate the intermediate values (omitted here as steps)
    sorry

end people_not_clients_is_11_l386_386585


namespace cos_inequality_sin_inequality_l386_386090

theorem cos_inequality (t : ℝ) : cos t ≤ (3 / 4) + (cos (2 * t) / 4) := sorry

theorem sin_inequality (t : ℝ) : sin t ≤ (3 / 4) - (cos (2 * t) / 4) := sorry

end cos_inequality_sin_inequality_l386_386090


namespace arithmetic_seq_middle_term_l386_386525

theorem arithmetic_seq_middle_term (a1 a3 y : ℤ) (h1 : a1 = 3^2) (h2 : a3 = 3^4)
    (h3 : y = (a1 + a3) / 2) : y = 45 :=
by
  rw [h1, h2] at h3
  simp at h3
  exact h3

end arithmetic_seq_middle_term_l386_386525


namespace simplify_polynomial_l386_386431
-- Lean 4 statement to prove algebraic simplification


open Polynomial

variable (x : ℤ)

theorem simplify_polynomial :
  3 * (3 * (C x ^ 2) + 9 * C x - 4) - 2 * (C x ^ 2 + 7 * C x - 14) = 7 * (C x ^ 2) + 13 * C x + 16 :=
by
  -- The actual proof steps would be here
  sorry

end simplify_polynomial_l386_386431


namespace convert_speed_l386_386242

-- Definitions based on the given condition
def kmh_to_mps (kmh : ℝ) : ℝ := kmh * 0.277778

-- Theorem statement
theorem convert_speed : kmh_to_mps 84 = 23.33 :=
by
  -- Proof omitted
  sorry

end convert_speed_l386_386242


namespace value_of_other_bills_l386_386382

theorem value_of_other_bills (x : ℕ) : 
  (∃ (num_twenty num_x : ℕ), num_twenty = 3 ∧
                           num_x = 2 * num_twenty ∧
                           20 * num_twenty + x * num_x = 120) → 
  x * 6 = 60 :=
by
  intro h
  obtain ⟨num_twenty, num_x, h1, h2, h3⟩ := h
  have : num_twenty = 3 := h1
  have : num_x = 2 * num_twenty := h2
  have : x * 6 = 60 := sorry
  exact this

end value_of_other_bills_l386_386382


namespace fraction_not_shaded_l386_386187

noncomputable def side_length : ℕ := 4

def coordinates_R : ℝ × ℝ := (2, 0)
def coordinates_S : ℝ × ℝ := (4, 2)
def coordinates_upper_left : ℝ × ℝ := (0, 4)

def area_of_square : ℝ := (side_length : ℝ)^2

def area_of_triangle (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  0.5 * |p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2)|

def shaded_area := area_of_square - area_of_triangle coordinates_upper_left coordinates_R coordinates_S

theorem fraction_not_shaded : (area_of_square - shaded_area) / area_of_square = 3 / 8 :=
by
  sorry

end fraction_not_shaded_l386_386187


namespace more_reliable_survey_results_l386_386132

/-- Conditions of the survey results of three students:
    - Xiao Ming: surveyed 100 elderly people in the hospital
    - Xiao Hong: surveyed 100 elderly people dancing square dancing
    - Xiao Liang: surveyed 100 elderly people in the general city area
-/
structure SurveyCondition :=
  (xiaoMing_hospital : Bool)
  (xiaoHong_squareDance : Bool)
  (xiaoLiang_cityArea : Bool)

theorem more_reliable_survey_results (cond : SurveyCondition) :
  cond.xiaoLiang_cityArea = true → cond.xiaoMing_hospital = true → cond.xiaoHong_squareDance = true → ∃ result, result = "Xiao Liang" :=
by
  intros h1 h2 h3
  use "Xiao Liang"
  sorry

end more_reliable_survey_results_l386_386132


namespace sum_geometric_sequence_l386_386486

theorem sum_geometric_sequence {n : ℕ} (S : ℕ → ℝ) (h1 : S n = 10) (h2 : S (2 * n) = 30) : 
  S (3 * n) = 70 := 
by 
  sorry

end sum_geometric_sequence_l386_386486


namespace number_of_zeroes_of_f_l386_386116

def f (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 - 1 else Real.log (x - 1) / Real.log 2

theorem number_of_zeroes_of_f : 
  set.count (set_of (λ x, f x = 0)) = 2 :=
sorry

end number_of_zeroes_of_f_l386_386116


namespace sqrt_6000_approx_l386_386756

theorem sqrt_6000_approx (
  h1 : Real.sqrt 6 ≈ 2.45,
  h2 : Real.sqrt 60 ≈ 7.75
) : Real.sqrt 6000 ≈ 77.5 := 
sorry

end sqrt_6000_approx_l386_386756


namespace sum_of_squares_of_roots_l386_386878

theorem sum_of_squares_of_roots :
  let a := 10
  let b := 15
  let c := -21
  let sum_roots := (-b : ℚ) / a
  let prod_roots := (c : ℚ) / a
  let sum_squares := sum_roots^2 - 2 * prod_roots
  sum_squares = (129 : ℚ) / 20 :=
by {
  let a := 10
  let b := 15
  let c := -21
  let sum_roots := (-b : ℚ) / a
  let prod_roots := (c : ℚ) / a
  let sum_squares := sum_roots^2 - 2 * prod_roots
  have h := calc sum_squares
      = sum_roots^2 - 2 * prod_roots : by rfl
      ... = ((-b) / a)^2 - 2 * (c / a) : by rfl
      ... = ((-15 : ℚ) / 10)^2 - 2 * ((-21 : ℚ) / 10) : by rfl
      ... = 9 / 4 + 42 / 10 : by { norm_num }
      ... = 129 / 20 : by { norm_num },
  exact h,
}

end sum_of_squares_of_roots_l386_386878


namespace quadrilateral_extension_area_l386_386851

-- Define the quadrilateral with given side lengths
structure Quadrilateral (α : Type) [OrderedField α] :=
  (EF FG GH HE : α)

-- Conditions on the extended sides
structure ExtendedSides (α : Type) [OrderedField α] :=
  (EF_eq_2FF' : Quadrilateral α → Prop)
  (FG_eq_3div2GG' : Quadrilateral α → Prop)
  (GH_eq_4div3HH' : Quadrilateral α → Prop)
  (HE_eq_5div4EE' : Quadrilateral α → Prop)

-- Main theorem statement
theorem quadrilateral_extension_area
  (α : Type) [OrderedField α]
  (quad : Quadrilateral α)
  (extend : ExtendedSides α)
  (h1 : quad.EF = 5)
  (h2 : quad.FG = 6)
  (h3 : quad.GH = 7)
  (h4 : quad.HE = 8)
  (h5 : extend.EF_eq_2FF' quad)
  (h6 : extend.FG_eq_3div2GG' quad)
  (h7 : extend.GH_eq_4div3HH' quad)
  (h8 : extend.HE_eq_5div4EE' quad)
  (area_EFGH : α)
  (h_area_EFGH : area_EFGH = 12) :
  ∃ area_E'F'G'H' : α, area_E'F'G'H' = 84 :=
sorry

end quadrilateral_extension_area_l386_386851


namespace probability_at_least_two_same_l386_386425

theorem probability_at_least_two_same (n m : ℕ) (hn : n = 8) (hm : m = 6):
  (probability (λ (ω : vector (fin m) n), ∃ (i j : fin n), i ≠ j ∧ ω.nth i = ω.nth j)) = 1 :=
begin
  sorry
end

end probability_at_least_two_same_l386_386425


namespace count_coprimes_15_l386_386653

def count_coprimes_less_than (n : ℕ) : ℕ :=
  (Finset.range n).filter (λ a => Nat.gcd a n = 1).card

theorem count_coprimes_15 :
  count_coprimes_less_than 15 = 8 :=
by
  sorry

end count_coprimes_15_l386_386653


namespace construct_parabola_focus_directrix_l386_386218

variable (e : ℝ → Prop) -- Tangent line e is a real-valued proposition (predicate for points).
variable (E : ℝ × ℝ) -- Point of tangency (x, y).
variable (p : ℝ) -- Parameter of the parabola.
variable (dir_axis : ℝ × ℝ) -- Direction of the axis as a vector.

theorem construct_parabola_focus_directrix :
  ∃ (F d : ℝ × ℝ), -- Focus F and directrix d (represented as point and line direction)
    E.1 = (F.1 - E.1) ^ 2 / (4 * p) ∧  -- Using parabola property as a constraint
    E.1 ≠ E.2 → -- Ensuring non-triviality in the construction
    ∃ (Q : ℝ × ℝ), -- Point Q on the directrix
      (E.1 - Q.1) = (Q.2 / p) ^ 2 ∧  -- Property involving point on the directrix
      ∀ (t : ℝ), E.1 = t + dir_axis.1 → -- Given direction of the axis
      let F₁ := reflection_across(E, e) in -- Focus reflection construction
      let Q₁ := F₁ reflection_across(E, e) in -- Directrix construction
      True := sorry

end construct_parabola_focus_directrix_l386_386218


namespace cistern_emptying_l386_386976

theorem cistern_emptying (h: (3 / 4) / 12 = 1 / 16) : (8 * (1 / 16) = 1 / 2) :=
by sorry

end cistern_emptying_l386_386976


namespace find_a_50_l386_386481

def a : ℕ → ℤ
| 0 := 2009
| (n + 1) := a n + (n + 1) ^ 2

theorem find_a_50 : a 50 = 44934 := sorry

end find_a_50_l386_386481


namespace number_of_roses_two_days_ago_l386_386910

-- Define the conditions
variables (R : ℕ) 
-- Condition 1: Variable R is the number of roses planted two days ago.
-- Condition 2: The number of roses planted yesterday is R + 20.
-- Condition 3: The number of roses planted today is 2R.
-- Condition 4: The total number of roses planted over three days is 220.
axiom condition_1 : 0 ≤ R
axiom condition_2 : (R + (R + 20) + (2 * R)) = 220

-- Proof goal: Prove that R = 50 
theorem number_of_roses_two_days_ago : R = 50 :=
by sorry

end number_of_roses_two_days_ago_l386_386910


namespace sum_of_powers_of_neg_one_l386_386680

theorem sum_of_powers_of_neg_one :
  ∑ n in finset.range (23), (-1)^(n - 11) = 0 :=
by
  sorry

end sum_of_powers_of_neg_one_l386_386680


namespace rhombus_perimeter_l386_386462

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) : 
  let a := d1 / 2 in
  let b := d2 / 2 in
  let c := Real.sqrt (a^2 + b^2) in
  let side := c in
  let perimeter := 4 * side in
  perimeter = 52 := 
by 
  let a := 5 in 
  let b := 12 in 
  have h3 : a = d1 / 2, by rw [h1]; norm_num
  have h4 : b = d2 / 2, by rw [h2]; norm_num
  let c := Real.sqrt (5^2 + 12^2),
  let side := c,
  have h5 : c = 13, by norm_num,
  let perimeter := 4 * 13,
  show perimeter = 52, by norm_num; sorry

end rhombus_perimeter_l386_386462


namespace minimum_omega_f_eq_cos_phi_l386_386066

noncomputable def f (x ω φ : ℝ) : ℝ := Real.cos (ω * x + φ)

theorem minimum_omega
  (ω : ℝ)
  (φ : ℝ)
  (T : ℝ)
  (h_ω_pos : ω > 0)
  (h_φ_range : 0 < φ ∧ φ < π)
  (h_T : T = 2 * π / ω)
  (h_f_T : f T ω φ = √3 / 2)
  (h_zero : f (π / 9) ω (π / 6) = 0)
  : ω = 3 := 
begin
  sorry
end

-- Helper condition to establish the function equivalence
theorem f_eq_cos_phi 
  (ω : ℝ) 
  (φ T : ℝ) 
  (h_T : T = 2 * π / ω)
  (h_f_T : f T ω φ = √3 / 2) 
  : φ = π / 6 :=
begin
  sorry
end

end minimum_omega_f_eq_cos_phi_l386_386066


namespace rhombus_perimeter_l386_386468

-- Define the rhombus with diagonals of given lengths and prove the perimeter is 52 inches
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) (d1_pos : 0 < d1) (d2_pos : 0 < d2):
  let s := sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2) in 
  let perimeter := 4 * s in
  perimeter = 52 
  :=
by
  sorry  -- Proof is skipped

end rhombus_perimeter_l386_386468


namespace feeding_sequences_count_l386_386983

/- 
  The zookeeper starts feeding with a male lion and needs to alternate genders until all animals are fed.
  There are a total of 5 pairs of animals (each pair consists of one male and one female),
  Determine the total number of ways to sequence the feeding such that genders are alternated.
-/
theorem feeding_sequences_count (pairs : Fin 6 → Fin 5) : 
  (∃ lion : Fin 5, ∃ females : Fin 6 → Fin 5, 
    lion ∈ pairs ∧ ∀ i, lion ≠ females i ∧ ∀ j, j ≠ i → (females i ≠ females j)) →
  (∃ n, n = 1440) :=
by
  have pairs := 5
  have sequences := 5 * 4 * 4 * 3 * 3 * 2 * 2 * 1 * 1
  exact sequences = 1440
  sorry

end feeding_sequences_count_l386_386983


namespace solve_sqrt_eq_l386_386246

theorem solve_sqrt_eq (z : ℤ) (h : sqrt (10 + 3 * z) = 8) : z = 18 := 
by {
  sorry
}

end solve_sqrt_eq_l386_386246


namespace average_age_of_women_l386_386443

variable {A W : ℝ}

theorem average_age_of_women (A : ℝ) (h : 12 * (A + 3) = 12 * A - 90 + W) : 
  W / 3 = 42 := by
  sorry

end average_age_of_women_l386_386443


namespace triangle_shape_l386_386005

theorem triangle_shape (a b c : ℝ) (A B C : ℝ) (h1 : a * Real.cos A = b * Real.cos B) :
  (a = b ∨ c = a ∨ c = b ∨ A = Real.pi / 2 ∨ B = Real.pi / 2 ∨ C = Real.pi / 2) :=
sorry

end triangle_shape_l386_386005


namespace mary_rental_hours_l386_386084

def ocean_bike_fixed_fee := 17
def ocean_bike_hourly_rate := 7
def total_paid := 80

def calculate_hours (fixed_fee : Nat) (hourly_rate : Nat) (total_amount : Nat) : Nat :=
  (total_amount - fixed_fee) / hourly_rate

theorem mary_rental_hours :
  calculate_hours ocean_bike_fixed_fee ocean_bike_hourly_rate total_paid = 9 :=
by
  sorry

end mary_rental_hours_l386_386084


namespace abs_eq_four_l386_386331

theorem abs_eq_four (x : ℝ) (h : |x| = 4) : x = 4 ∨ x = -4 :=
by
  sorry

end abs_eq_four_l386_386331


namespace sum_proper_divisors_243_l386_386923

theorem sum_proper_divisors_243 : (1 + 3 + 9 + 27 + 81) = 121 :=
by
  sorry

end sum_proper_divisors_243_l386_386923


namespace harold_initial_money_l386_386148

theorem harold_initial_money (x : ℕ) (h1 : x = 210) :
  (x / 2 - 5) / 2 - 15 - 5 = 5 :=
by {
  have h_init : x = 210 := h1,
  have h1 := x / 2,
  let y1 := h1 - 5,
  have h2 := y1 / 2,
  let y2 := h2 - 10,
  have h3 := y2 / 2,
  let y3 := h3 - 15,
  get_solution y1 y2 y3 y_final : 5 = sorry,
} 

end harold_initial_money_l386_386148


namespace sum_of_digits_N_l386_386628

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem sum_of_digits_N :
  let N := (List.range 250).sum (λ n, 10^(n+1) - 1)
  sum_of_digits N = 259 :=
by 
  sorry

end sum_of_digits_N_l386_386628


namespace median_of_data_set_l386_386875

def data_set := [8, 12, 7, 7, 10, 12]
def sorted_data_set := [7, 7, 8, 10, 12, 12]
def is_even (n : Nat) : Prop := n % 2 = 0

theorem median_of_data_set :
  sorted_data_set.length = 6 →
  is_even sorted_data_set.length →
  (sorted_data_set.sorted = sorted_data_set) →
  (∑ i in [(sorted_data_set[2]), sorted_data_set[3]], i) / 2 = 9 := 
by
  intro hlen heven hsorted
  simp at hlen
  simp at heven
  simp
  sorry

end median_of_data_set_l386_386875


namespace isosceles_base_length_l386_386158

theorem isosceles_base_length :
  ∀ (equilateral_perimeter isosceles_perimeter side_length base_length : ℕ), 
  equilateral_perimeter = 60 →  -- Condition: Perimeter of the equilateral triangle is 60
  isosceles_perimeter = 45 →    -- Condition: Perimeter of the isosceles triangle is 45
  side_length = equilateral_perimeter / 3 →   -- Condition: Each side of the equilateral triangle
  isosceles_perimeter = side_length + side_length + base_length  -- Condition: Perimeter relation in isosceles triangle
  → base_length = 5  -- Result: The base length of the isosceles triangle is 5
:= 
sorry

end isosceles_base_length_l386_386158


namespace prove_correct_sum_numerator_denominator_l386_386626

noncomputable def probability_at_least_one_palindrome : ℚ :=
  let letter_combinations := 26 ^ 4
  let letter_palindromes := 26 * 26
  let digit_combinations := 10 ^ 4
  let digit_palindromes := 10 * 10
  let P_letter := (letter_palindromes : ℚ) / letter_combinations
  let P_digit := (digit_palindromes : ℚ) / digit_combinations
  let P := P_letter + P_digit - P_letter * P_digit
  P

noncomputable def sum_numerator_denominator_reduced_fraction : ℚ × ℚ × ℕ :=
  let reduced_fraction := Rat.numDenProbability_at_least_one_palindrome
  (reduced_fraction.num, reduced_fraction.den, reduced_fraction.num + reduced_fraction.den)

theorem prove_correct_sum_numerator_denominator :
  sum_numerator_denominator_reduced_fraction = (655, 57122, 57777) :=
sorry

end prove_correct_sum_numerator_denominator_l386_386626


namespace sally_initial_pokemon_cards_l386_386855

theorem sally_initial_pokemon_cards :
  ∃ (x : ℕ), x + 41 + 20 = 88 ∧ x = 27 :=
by {
  use 27,
  split,
  { norm_num }, -- proof that 27 + 61 = 88
  { refl } -- proof that x = 27
  }

end sally_initial_pokemon_cards_l386_386855


namespace total_berries_l386_386861

theorem total_berries (S_stacy S_steve S_skylar : ℕ) 
  (h1 : S_stacy = 800)
  (h2 : S_stacy = 4 * S_steve)
  (h3 : S_steve = 2 * S_skylar) :
  S_stacy + S_steve + S_skylar = 1100 :=
by
  sorry

end total_berries_l386_386861


namespace rhombus_perimeter_l386_386464

-- Define the rhombus with diagonals of given lengths and prove the perimeter is 52 inches
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) (d1_pos : 0 < d1) (d2_pos : 0 < d2):
  let s := sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2) in 
  let perimeter := 4 * s in
  perimeter = 52 
  :=
by
  sorry  -- Proof is skipped

end rhombus_perimeter_l386_386464


namespace regular_k_gons_equiv_l386_386415

variables {k n : ℕ} (A B C D : Fin k → Fin k → Type*)
  [∀ i j, AddCommGroup (A i j)] [∀ i j, AddCommGroup (B i j)]
  [∀ i j, AddCommGroup (C i j)] [∀ i j, AddCommGroup (D i j)]
  [∀ i j, Module ℝ (A i j)] [∀ i j, Module ℝ (B i j)]
  [∀ i j, Module ℝ (C i j)] [∀ i j, Module ℝ (D i j)]

theorem regular_k_gons_equiv (hA : ∀ i, is_regular (A i))
  (hB : ∀ i, is_regular (B i))
  (hC : ∀ i, is_regular (C i))
  (hD : ∀ i, is_regular (D i))
  (hOrC : has_same_orientation (C 1) (C 2))
  (hOrD : has_same_orientation (D 1) (D 2)) :
  (∀ i, is_regular (C i) ∧ ∀ i, is_regular (D i) ∧ has_same_orientation (C 1) (D 1)) ↔
  (∀ i, is_regular (A i) ∧ ∀ i, is_regular (B i) ∧ has_same_orientation (A 1) (B 1)) :=
sorry

end regular_k_gons_equiv_l386_386415


namespace min_value_when_a_eq_1_range_of_a_for_two_zeros_l386_386401

-- Define the piecewise function f
def f (x a : ℝ) : ℝ :=
if x < 1 then 2^x - a else 4 * (x - a) * (x - 2 * a)

-- Problem 1: Prove minimum value when a = 1
theorem min_value_when_a_eq_1 : ∀ x : ℝ, f x 1 ≥ -1 := sorry

-- Problem 2: Determine the range of a for exactly 2 zeros
theorem range_of_a_for_two_zeros (a : ℝ) :
  (∃! x : ℝ, f x a = 0) ↔ (1/2 ≤ a ∧ a < 1) ∨ (2 ≤ a) := sorry

end min_value_when_a_eq_1_range_of_a_for_two_zeros_l386_386401


namespace train_length_is_400_l386_386981

-- Define the conditions
def time := 40 -- seconds
def speed_kmh := 36 -- km/h

-- Conversion factor from km/h to m/s
def kmh_to_ms (v : ℕ) := (v * 5) / 18

def speed_ms := kmh_to_ms speed_kmh -- convert speed to m/s

-- Definition of length of the train using the given conditions
def train_length := speed_ms * time

-- Theorem to prove the length of the train is 400 meters
theorem train_length_is_400 : train_length = 400 := by
  sorry

end train_length_is_400_l386_386981


namespace count_coprime_with_15_lt_15_l386_386652

theorem count_coprime_with_15_lt_15 :
  {a : ℕ // a < 15 ∧ Nat.coprime 15 a}.to_finset.card = 8 := 
sorry

end count_coprime_with_15_lt_15_l386_386652


namespace probability_divisible_by_fourteen_l386_386672

-- Define the problem conditions
def is_uniform_cubic_die_face (n : ℕ) : Prop :=
  n ∈ {1, 2, 3, 4, 5, 6}

def roll_two_dice_sums (n : ℕ) : Prop :=
  ∃ a b, is_uniform_cubic_die_face a ∧ is_uniform_cubic_die_face b ∧ n = a + b

def divisible_by_fourteen (n : ℕ) : Prop :=
  14 ∣ n

-- Define the probability of the event
def probability_event_divisible_by_fourteen (roll_sum_3 : list ℕ) : Prop :=
  divisible_by_fourteen (roll_sum_3.foldr (*) 1) ∧ roll_sum_3.length = 3

-- Define the main theorem
theorem probability_divisible_by_fourteen :
  ∃ (p : ℚ), p = 1 / 3 ∧ 
  ∀ (roll_sum_3 : list ℕ), (∀ n, n ∈ roll_sum_3 → roll_two_dice_sums n) →
  probability_event_divisible_by_fourteen roll_sum_3 → True :=
sorry

end probability_divisible_by_fourteen_l386_386672


namespace ascending_function_k_ge_2_l386_386400

open Real

def is_ascending (f : ℝ → ℝ) (k : ℝ) (M : Set ℝ) : Prop :=
  ∀ x ∈ M, f (x + k) ≥ f x

theorem ascending_function_k_ge_2 :
  ∀ (k : ℝ), (∀ x : ℝ, x ≥ -1 → (x + k) ^ 2 ≥ x ^ 2) → k ≥ 2 :=
by
  intros k h
  sorry

end ascending_function_k_ge_2_l386_386400


namespace no_matrix_M_exists_l386_386226

def M_transform (M : ℝ²²) (A : ℝ²²) : ℝ²² := M * A

theorem no_matrix_M_exists :
    ∀ M : Matrix (Fin 2) (Fin 2) ℝ,
    ¬(∀ (A : Matrix (Fin 2) (Fin 2) ℝ),
        let a := A 0 0,
            b := A 0 1,
            c := A 1 0,
            d := A 1 1 in
        M.mul A = ![![a, b + 3], ![c, d]]) :=
by {
  intro M,
  -- Conditions derived from the problem description
  have h₁ := fun (A : Matrix (Fin 2) (Fin 2) ℝ),
    (M.mul A) 0 0 = A 0 0 → (M 0 0) * (A 0 0) + (M 0 1) * (A 1 0) = A 0 0,
  have h₂ := fun (A : Matrix (Fin 2) (Fin 2) ℝ),
    (M.mul A) 0 1 = A 0 1 + 3 → (M 0 0) * (A 0 1) + (M 0 1) * (A 1 1) = A 0 1 + 3,
  have h₃ := fun (A : Matrix (Fin 2) (Fin 2) ℝ),
    (M.mul A) 1 0 = A 1 0 → (M 1 0) * (A 0 0) + (M 1 1) * (A 1 0) = A 1 0,
  have h₄ := fun (A : Matrix (Fin 2) (Fin 2) ℝ),
    (M.mul A) 1 1 = A 1 1 → (M 1 0) * (A 0 1) + (M 1 1) * (A 1 1) = A 1 1,
  cases A with _ _ _ _,
  sorry -- The proof steps go here
}

end no_matrix_M_exists_l386_386226


namespace journey_duration_l386_386962

theorem journey_duration
  (distance : ℕ) (speed : ℕ) (h1 : distance = 48) (h2 : speed = 8) :
  distance / speed = 6 := 
by
  sorry

end journey_duration_l386_386962


namespace pair1_equivalent_pair2_non_equivalent_pair3_equivalent_pair4_equivalent_pair5_non_equivalent_pair6_equivalent_l386_386995

theorem pair1_equivalent (x : ℝ) : (x^2 + 5 * x < 4) ↔ (x^2 + 5 * x + 3 * x < 4 + 3 * x) :=
sorry

theorem pair2_non_equivalent (x : ℝ) (hx : x ≠ 0) : (x^2 + 5 * x < 4) ↔ (x^2 + 5 * x + 1 / x < 4 + 1 / x) :=
sorry

theorem pair3_equivalent (x : ℝ) (hx : x ≥ 3) : (x ≥ 3) ↔ (x * (x + 5)^2 ≥ 3 * (x + 5)^2) :=
sorry

theorem pair4_equivalent (x : ℝ) (hx : x ≥ 3) : (x ≥ 3) ↔ (x * (x - 5)^2 ≥ 3 * (x - 5)^2) :=
sorry

theorem pair5_non_equivalent (x : ℝ) (hx : x ≠ -1) : (x + 3 > 0) ↔ ( (x + 3) * (x + 1) / (x + 1) > 0) :=
sorry

theorem pair6_equivalent (x : ℝ) (hx : x ≠ -2) : (x - 3 > 0) ↔ ( (x + 2) * (x - 3) / (x + 2) > 0) :=
sorry

end pair1_equivalent_pair2_non_equivalent_pair3_equivalent_pair4_equivalent_pair5_non_equivalent_pair6_equivalent_l386_386995


namespace amount_x_gets_l386_386230

theorem amount_x_gets (total_money : ℝ) (ratio_x ratio_y ratio_z : ℝ) :
  total_money = 5000 ∧ ratio_x = 2 ∧ ratio_y = 5 ∧ ratio_z = 8 →
  (total_money * ratio_x / (ratio_x + ratio_y + ratio_z)) ≈ 666.66 :=
by
  sorry

end amount_x_gets_l386_386230


namespace problem_statement_l386_386665

theorem problem_statement :
  (∃ n : ℕ, n = 8 ∧ ∀ a : ℕ, a < 15 → (∃ x : ℤ, a * x ≡ 1 [MOD 15]) ↔ gcd a 15 = 1) :=
by
  use 8
  intro a
  intro ha
  split
  sorry

end problem_statement_l386_386665


namespace points_2_units_away_l386_386413

theorem points_2_units_away : (∃ x : ℝ, (x = -3 ∨ x = 1) ∧ (abs (x - (-1)) = 2)) :=
by
  sorry

end points_2_units_away_l386_386413


namespace jack_jog_speed_l386_386037

theorem jack_jog_speed (melt_time_minutes : ℕ) (distance_blocks : ℕ) (block_length_miles : ℚ) 
    (h_melt_time : melt_time_minutes = 10)
    (h_distance : distance_blocks = 16)
    (h_block_length : block_length_miles = 1/8) :
    let time_hours := (melt_time_minutes : ℚ) / 60
    let distance_miles := (distance_blocks : ℚ) * block_length_miles
        12 = distance_miles / time_hours :=
by
  sorry

end jack_jog_speed_l386_386037


namespace bacteria_exceeds_200_l386_386345

def bacteria_growth (n : ℕ) : ℕ := 5 * 3^n

theorem bacteria_exceeds_200 : ∃ n : ℕ, bacteria_growth n > 200 ∧ n = 4 :=
by {
    use 4,
    split,
    { simp [bacteria_growth],
      norm_num,
      rw [pow_succ, pow_succ, pow_zero, mul_assoc, mul_assoc, mul_comm 5 3],
      norm_num },
    { refl },
  }

end bacteria_exceeds_200_l386_386345


namespace arc_length_of_sector_l386_386185

-- Define the central angle in radians
def central_angle_deg : ℝ := 30
def central_angle_rad : ℝ := central_angle_deg * (Real.pi / 180)

-- Define the radius of the sector
def radius : ℝ := 1

-- Define the length of the arc
def arc_length : ℝ := central_angle_rad * radius

-- State the theorem
theorem arc_length_of_sector (r : ℝ) (alpha_deg : ℝ) (alpha_rad : ℝ) :
  alpha_rad = alpha_deg * (Real.pi / 180) →
  r = 1 →
  alpha_deg = 30 →
  arc_length = Real.pi / 6 :=
by
  intros h_alpha_rad h_radius h_alpha_deg
  simp [arc_length, *]
  sorry

end arc_length_of_sector_l386_386185


namespace problem_statement_l386_386162

noncomputable def euler_totient (n : ℕ) : ℕ := 
if n = 0 then 0 else n.prod_factors.eraseDup.map (λ p, p - 1).prod

def divides (m n : ℕ) : Prop := ∃ k, n = m * k

theorem problem_statement {a n : ℕ} (h₀ : n > 1) (h₁ : coprime a n) : 
  divides (euler_totient n) (euler_totient (euler_totient (a^n - 1))) :=
sorry

end problem_statement_l386_386162


namespace prob_red_ball_given_first_red_l386_386791

theorem prob_red_ball_given_first_red :
  let bag := ["red", "red", "red", "white", "white", "white"];
  let remaining_balls := bag.erase("red");
  let remaining_reds := remaining_balls.count("red")
  let total_remaining := remaining_balls.length
  let probability := remaining_reds / total_remaining in
  probability = 2 / 5 :=
by sorry

end prob_red_ball_given_first_red_l386_386791


namespace painting_condition_feasibility_l386_386440

theorem painting_condition_feasibility (k : ℕ) (hk : 1 ≤ k ∧ k ≤ 10) :
  let points  := fin 10
      segments := { (i, j) // i < j }
      colored_painting := segments → fin k
      distinct_colors_condition := ∀ (s : fin k → points), ∀ i ≠ j, ∃ c : fin k, ¬ ∃ r s t : points, r ≠ s ∧ s ≠ t ∧ t ≠ r ∧ (c = r.color) ∧ (c = s.color) ∧ (c = t.color) 
  in (colored_painting → distinct_colors_condition ←→ k ≥ 5) :=
sorry

end painting_condition_feasibility_l386_386440


namespace shaded_region_area_l386_386025

noncomputable def area_of_shaded_region : ℝ :=
  let diameter := 3 in
  let small_semicircle_area (d : ℝ) := (π * d^2) / 8 in
  let large_diameter := 5 * diameter in
  let large_semicircle_area := (π * large_diameter^2) / 8 in
  let intersection_area := 2 * (π * diameter^2) / 16 in
  large_semicircle_area - intersection_area

theorem shaded_region_area (diameter : ℝ) (AF : ℝ) (GH : ℝ) :
  AF = 5 * diameter →
  GH = AF / 5 →
  area_of_shaded_region = 27 * π :=
  by
    intros h1 h2
    sorry

end shaded_region_area_l386_386025


namespace baba_yaga_strategy_l386_386035

/-- Babel Yaga can prevent Ivan from identifying all counterfeit coins within 2020 questions --/
theorem baba_yaga_strategy (coins : Fin 10 → Bool) (h₁ : ∀ i, coins i → (i < 5) ∨ ¬ coins i ∧ (i ≥ 5) := Iff.rfl)
(h₂ : ∀ (s : Finset (Fin 3)) (t : Finset (Fin 2)), (coin 체 Fin 2 → Bool ∧  coins.t neg coin)) :
  ∃ (baba_yaga_strategy : Finset (Fin 3) → Bool), 
  ∀ (s : Finset (Fin 3)) , baba_yaga_strategy s = coins t  → (İncoins.find (2020) neg→⊥) := 
begin
  sorry
end

end baba_yaga_strategy_l386_386035


namespace measure_angle_C_value_of_sin_A_value_of_sin_2A_plus_pi_over_4_l386_386371

noncomputable def C (a b c : ℝ) : ℝ :=
  Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))

#eval C (2 * Real.sqrt 2) 5 (Real.sqrt 13) -- Should evaluate to π/4

theorem measure_angle_C : C (2 * Real.sqrt 2) 5 (Real.sqrt 13) = π / 4 :=
sorry

noncomputable def sin_A (a c C : ℝ) : ℝ :=
  a * Real.sin C / c

#eval sin_A (2 * Real.sqrt 2) (Real.sqrt 13) (π / 4) -- Should evaluate to 2 * Real.sqrt 13 / 13

theorem value_of_sin_A : sin_A (2 * Real.sqrt 2) (Real.sqrt 13) (π / 4) = 2 * Real.sqrt 13 / 13 :=
sorry

noncomputable def sin_2A_plus_pi_over_4 (A : ℝ) : ℝ :=
  Real.sin (2 * A + π / 4)

theorem value_of_sin_2A_plus_pi_over_4 (A : ℝ) : sin_2A_plus_pi_over_4 (Real.arcsin (2 * Real.sqrt 13 / 13)) = 17 * Real.sqrt 2 / 26 :=
sorry

end measure_angle_C_value_of_sin_A_value_of_sin_2A_plus_pi_over_4_l386_386371


namespace find_n_l386_386306

theorem find_n (n : ℝ) (h1 : (n ≠ 0)) (h2 : ∃ (n' : ℝ), n = n' ∧ -n' = -9 / n') (h3 : ∀ x : ℝ, x > 0 → -n * x < 0) : n = 3 :=
sorry

end find_n_l386_386306


namespace move_digit_to_make_equation_correct_l386_386353

theorem move_digit_to_make_equation_correct :
  101 - 102 ≠ 1 → (101 - 10^2 = 1) :=
by
  sorry

end move_digit_to_make_equation_correct_l386_386353


namespace area_BFEC_l386_386591

-- Definition of the points and given lengths in the rectangle.
structure Rectangle :=
  (A B C D : Point)
  (AB BC CD DA : ℝ)
  (AB_def : AB = 40)
  (BC_def : BC = 28)
  (CD_def : CD = 40)
  (DA_def : DA = 28)
  (D_mid : D = midpoint A B)
  (E_mid : E = midpoint B C)
  (F_mid : F = midpoint A D)

noncomputable def area_quadrilateral_BFEC (r : Rectangle) : ℝ :=
  let A_FE := 0.5 * 20 * 14
  let A_FEC := 0.5 * 28 * 20
  let A_total := 40 * 28
  in A_total - A_FE - A_FEC

theorem area_BFEC {r : Rectangle} : area_quadrilateral_BFEC r = 700 := by
  sorry

end area_BFEC_l386_386591


namespace num_math_not_science_l386_386016

-- Definitions as conditions
def students_total : ℕ := 30
def both_clubs : ℕ := 2
def math_to_science_ratio : ℕ := 3

-- The proof we need to show
theorem num_math_not_science :
  ∃ x y : ℕ, (x + y + both_clubs = students_total) ∧ (y = math_to_science_ratio * (x + both_clubs) - 2 * (math_to_science_ratio - 1)) ∧ (y - both_clubs = 20) :=
by
  sorry

end num_math_not_science_l386_386016


namespace cannot_cut_squares_l386_386939

variables (length width : ℝ) (ratio1 ratio2 : ℝ) (sum_areas : ℝ)

def can_cut_squares (length width : ℝ) (ratio1 ratio2 : ℝ) (sum_areas : ℝ) : Prop :=
  ∀ x : ℝ, (ratio1 * x + ratio2 * x ≤ length) → (ratio1^2 * x^2 + ratio2^2 * x^2 = sum_areas) → false

theorem cannot_cut_squares (h : can_cut_squares 10 8 4 3 75) : false :=
  h sorry

end cannot_cut_squares_l386_386939


namespace expected_number_of_digits_is_one_l386_386207

-- Definition of the problem conditions
def is_fair_octahedral_die (n : ℕ) : Prop :=
  n >= 1 ∧ n <= 8

-- Expected number of digits calculation
noncomputable def expected_digits : ℝ :=
  if (∀ n, is_fair_octahedral_die n → (Nat.digits 10 n).length = 1) then 1 else 0

-- Theorem statement
theorem expected_number_of_digits_is_one :
  expected_digits = 1 :=
by
  sorry

end expected_number_of_digits_is_one_l386_386207


namespace max_intersections_of_semicircles_l386_386579

theorem max_intersections_of_semicircles (n : ℕ) (h : n ≥ 4) :
  ∃ (max_intersections : ℕ), max_intersections = (finset.choose n 4) := 
begin
  -- Proof omitted
  sorry
end

end max_intersections_of_semicircles_l386_386579


namespace gcd_of_sum_and_squares_l386_386846

theorem gcd_of_sum_and_squares {a b : ℤ} (h : Int.gcd a b = 1) : 
  Int.gcd (a^2 + b^2) (a + b) = 1 ∨ Int.gcd (a^2 + b^2) (a + b) = 2 := 
by
  sorry

end gcd_of_sum_and_squares_l386_386846


namespace probability_of_condition_l386_386028

/--
In the interval [0,4], two real numbers x and y are randomly selected such that x + 2y ≤ 8. 
The probability is 3/4.
-/
theorem probability_of_condition (x y : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 4) (h3 : 0 ≤ y) (h4 : y ≤ 4) (h5 : x + 2 * y ≤ 8) :
  probability (λ p : ℝ × ℝ, 0 ≤ p.1 ∧ p.1 ≤ 4 ∧ 0 ≤ p.2 ∧ p.2 ≤ 4 ∧ p.1 + 2 * p.2 ≤ 8) = 3 / 4 :=
sorry

end probability_of_condition_l386_386028


namespace markup_calculation_l386_386877

theorem markup_calculation
  (purchase_price : ℝ)
  (overhead_percentage : ℝ)
  (net_profit : ℝ)
  (overhead_costs : ℝ := overhead_percentage * purchase_price) :
  overhead_percentage = 0.45 → 
  purchase_price = 75 → 
  net_profit = 20 → 
  overhead_costs + net_profit = 53.75 :=
begin
  intros h1 h2 h3,
  rw [h1, h2, h3],
  simp only [mul_assoc, mul_comm, Real.mul_self_sqrt],
  norm_num,
end

end markup_calculation_l386_386877


namespace quadrilateral_DB_l386_386030

theorem quadrilateral_DB :
  ∀ (A B C D : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D]
  (AB BC CD DA : ℕ)
  (HAB : AB = 5) 
  (HBC : BC = 17) 
  (HCD : CD = 5) 
  (HDA : DA = 9) 
  (DB : ℕ),
  (9 - 5 < DB ∧ DB < 9 + 5) → 
  (17 - 5 < DB ∧ DB < 17 + 5) → 
  DB = 13 :=
by {
  assume A B C D,
  assume metric_spaceA metric_spaceB metric_spaceC metric_spaceD,
  assume AB BC CD DA,
  assume HAB HBC HCD HDA DB,
  assume H1 H2,
  sorry
}

end quadrilateral_DB_l386_386030


namespace digits_right_of_decimal_of_fraction_l386_386320

def num_digits_right_of_decimal (n d : ℕ) : ℕ :=
  String.length (String.takeWhile (λ c => c ≠ '.') (String.dropWhile (λ c => c ≠ '.') (toString (n / d))) - 1)

theorem digits_right_of_decimal_of_fraction :
  num_digits_right_of_decimal 2^10 (4^3 * 125) = 3 :=
by
  sorry

end digits_right_of_decimal_of_fraction_l386_386320


namespace calculate_angles_l386_386775

noncomputable def plane_angle (n : ℕ) (φ : ℝ) : ℝ :=
  2 * Real.arcsin (Real.sin (Real.pi / n) * Real.sin φ)

noncomputable def dihedral_angle (n : ℕ) (φ : ℝ) : ℝ :=
  2 * Real.arcsin ((Real.sin (2 * Real.pi / n) * Real.sin φ) / Real.sin (plane_angle n φ))

theorem calculate_angles (n : ℕ) (φ plane_angle dihedral_angle : ℝ) :
  plane_angle = 2 * Real.arcsin (Real.sin (Real.pi / n) * Real.sin φ) ∧
  dihedral_angle = 2 * Real.arcsin ((Real.sin (2 * Real.pi / n) * Real.sin φ) / Real.sin (plane_angle n φ)) :=
by
  have h1 : plane_angle = 2 * Real.arcsin (Real.sin (Real.pi / n) * Real.sin φ), by sorry
  have h2 : dihedral_angle = 2 * Real.arcsin ((Real.sin (2 * Real.pi / n) * Real.sin φ) / Real.sin (plane_angle n φ)), by sorry
  exact ⟨h1, h2⟩

end calculate_angles_l386_386775


namespace smallest_k_l386_386818

noncomputable def f (a b : ℕ) (M : ℤ) (n : ℤ) : ℤ :=
  if n < M then n + a else n - b

def M_floor (a b : ℕ) : ℤ := Int.floor ((a + b) / 2)

def f_iter (a b : ℕ) (n : ℤ) (k : ℕ) : ℤ :=
  Nat.recOn k n (λ _ r, f a b (M_floor a b) r)

theorem smallest_k (a b : ℕ) (h : 1 ≤ a ∧ a ≤ b) :
  ∃ k : ℕ, f_iter a b 0 k = 0 ∧ (∀ j, j < k → f_iter a b 0 j ≠ 0) :=
sorry

end smallest_k_l386_386818


namespace area_of_equilateral_triangle_l386_386727

open Classical

-- Define the hyperbola
def is_hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

-- Define point A
def point_A := (-1 : ℝ, 0 : ℝ)

-- Define point B on the hyperbola
def point_B (x y : ℝ) : Prop := is_hyperbola x y ∧ x > 0

-- Define point C on the hyperbola
def point_C (x y : ℝ) : Prop := is_hyperbola x y ∧ x > 0

-- Define equilateral triangle condition for points A, B, and C
def is_equilateral_triangle (A B C : ℝ × ℝ) : Prop :=
  let d1 := (B.1 - A.1)^2 + (B.2 - A.2)^2
  let d2 := (C.1 - A.1)^2 + (C.2 - A.2)^2
  let d3 := (C.1 - B.1)^2 + (C.2 - B.2)^2
  d1 = d2 ∧ d2 = d3 ∧ d1 ≠ 0

-- Define area of triangle given three points
def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  0.5 * | A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2) |

-- Define specific points B and C on the right branch of the hyperbola
def specific_point_B := (2 : ℝ, real.sqrt 3)
def specific_point_C := (2 : ℝ, -real.sqrt 3)

-- The final theorem statement
theorem area_of_equilateral_triangle :
  is_hyperbola point_A.1 point_A.2 →
  point_B 2 (real.sqrt 3) →
  point_C 2 (-real.sqrt 3) →
  is_equilateral_triangle point_A specific_point_B specific_point_C →
  area_of_triangle point_A specific_point_B specific_point_C = 3 * real.sqrt 3 :=
by
  sorry

end area_of_equilateral_triangle_l386_386727


namespace least_common_period_l386_386629

theorem least_common_period (f : ℝ → ℝ) (H : ∀ x, f(x + 5) + f(x - 5) = f(x)) : ∃ p, p > 0 ∧ (∀ x, f(x + p) = f(x)) ∧ ∀ q, q > 0 ∧ (∀ x, f(x + q) = f(x)) → p ≤ q := by
  sorry

end least_common_period_l386_386629


namespace rate_of_current_l386_386127

theorem rate_of_current (speed_boat_still_water : ℕ) (time_hours : ℚ) (distance_downstream : ℚ)
    (h_speed_boat_still_water : speed_boat_still_water = 20)
    (h_time_hours : time_hours = 15 / 60)
    (h_distance_downstream : distance_downstream = 6.25) :
    ∃ c : ℚ, distance_downstream = (speed_boat_still_water + c) * time_hours ∧ c = 5 :=
by
    sorry

end rate_of_current_l386_386127


namespace derivative_of_3sinx_minus_4cosx_l386_386448

open Real

theorem derivative_of_3sinx_minus_4cosx (x : ℝ) : 
  deriv (fun x => 3 * sin x - 4 * cos x) x = 3 * cos x + 4 * sin x := 
sorry

end derivative_of_3sinx_minus_4cosx_l386_386448


namespace vector_calculation_l386_386700

def vector_a : ℝ × ℝ × ℝ := (1, 2, real.sqrt 3)
def vector_b : ℝ × ℝ × ℝ := (-1, real.sqrt 3, 0)

def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2 + v.3^2)

theorem vector_calculation :
  dot_product vector_a vector_b + magnitude vector_b = 1 + 2 * real.sqrt 3 :=
by
  sorry

end vector_calculation_l386_386700


namespace rhombus_diagonals_l386_386099

theorem rhombus_diagonals (x y : ℝ) 
  (h1 : x * y = 234)
  (h2 : x + y = 31) :
  (x = 18 ∧ y = 13) ∨ (x = 13 ∧ y = 18) := by
sorry

end rhombus_diagonals_l386_386099


namespace average_rainfall_per_hour_in_June_1882_l386_386770

open Real

theorem average_rainfall_per_hour_in_June_1882 
  (total_rainfall : ℝ) (days_in_June : ℕ) (hours_per_day : ℕ)
  (H1 : total_rainfall = 450) (H2 : days_in_June = 30) (H3 : hours_per_day = 24) :
  total_rainfall / (days_in_June * hours_per_day) = 5 / 8 :=
by
  sorry

end average_rainfall_per_hour_in_June_1882_l386_386770


namespace truck_X_speed_l386_386908

theorem truck_X_speed :
  ∀ (Vx : ℝ),
    (∃ Y_speed distance_ahead time_overtake distance_ahead_when_overtake : ℝ,
      Y_speed = 63 ∧
      distance_ahead = 14 ∧
      distance_ahead_when_overtake = 4 ∧
      time_overtake = 3 ∧
      (Y_speed * time_overtake = Vx * time_overtake + (distance_ahead + distance_ahead_when_overtake)))
    → Vx = 57 :=
begin
  intros Vx h,
  rcases h with ⟨Y_speed, distance_ahead, time_overtake, distance_ahead_when_overtake, hy⟩,
  rcases hy with ⟨rfl, rfl, rfl, rfl, h_eq⟩,
  have h1 : Y_speed * time_overtake = Vx * time_overtake + (14 + 4),
  { exact h_eq },
  have h2 : 63 * 3 = Vx * 3 + 18,
  { rw Y_speed at h1, rw time_overtake at h1, exact h1 },
  have simplify_left : 63 * 3 = 189 := by norm_num,
  have simplify_right : Vx * 3 + 18 = Vx * 3 + 18 := by ring,
  have h3 : 189 = Vx * 3 + 18,
  { rw simplify_left, rw simplify_right, exact h2 },
  linarith,
end

end truck_X_speed_l386_386908


namespace surface_area_after_removal_l386_386960

theorem surface_area_after_removal :
  let cube_side := 4
  let corner_cube_side := 2
  let original_surface_area := 6 * (cube_side * cube_side)
  (original_surface_area = 96) ->
  (6 * (cube_side * cube_side) - 8 * 3 * (corner_cube_side * corner_cube_side) + 8 * 3 * (corner_cube_side * corner_cube_side) = 96) :=
by
  intros
  sorry

end surface_area_after_removal_l386_386960


namespace kite_diagonal_and_angles_l386_386972

-- Assumptions and Definitions
variables (a b : ℝ)
variables (h_a_b : b ≥ a)
variables (h_eq : a = b)

-- Prove the diagonal length d
def diagonal_length (a b : ℝ) (h_eq : a = b) : ℝ :=
  real.sqrt (2 * b^2)

-- Define the angles at vertices
def angle_alpha (d b : ℝ) : ℝ :=
  real.arccos (1 - d^2 / (2 * b^2))

def angle_beta (d a b : ℝ) : ℝ :=
  real.arccos ((a^2 + b^2 - d^2) / (2 * a * b))

-- The Theorem Statement we need to prove
theorem kite_diagonal_and_angles (a b : ℝ) (h_a_b : b ≥ a) (h_eq : a = b) :
  let d := diagonal_length a b h_eq,
  α := angle_alpha d b,
  β := angle_beta d a b in
  d = real.sqrt (2 * b^2) ∧
  α = real.arccos (1 - d^2 / (2 * b^2)) ∧
  β = real.arccos ((a^2 + b^2 - d^2) / (2 * a * b))
:= by sorry

end kite_diagonal_and_angles_l386_386972


namespace students_taking_only_science_l386_386996

theorem students_taking_only_science (total_students : ℕ) (students_science : ℕ) (students_math : ℕ)
  (h1 : total_students = 120) (h2 : students_science = 80) (h3 : students_math = 75) :
  (students_science - (students_science + students_math - total_students)) = 45 :=
by
  sorry

end students_taking_only_science_l386_386996


namespace smallest_n_not_divisible_by_10_l386_386268

theorem smallest_n_not_divisible_by_10 :
  ∃ n > 2016, (1^n + 2^n + 3^n + 4^n) % 10 ≠ 0 ∧ n = 2020 := 
sorry

end smallest_n_not_divisible_by_10_l386_386268


namespace arccos_cos_9_eq_2_717_l386_386620

-- Statement of the proof problem
theorem arccos_cos_9_eq_2_717 : Real.arccos (Real.cos 9) = 2.717 :=
by
  sorry

end arccos_cos_9_eq_2_717_l386_386620


namespace count_coprimes_15_l386_386658

def count_coprimes_less_than (n : ℕ) : ℕ :=
  (Finset.range n).filter (λ a => Nat.gcd a n = 1).card

theorem count_coprimes_15 :
  count_coprimes_less_than 15 = 8 :=
by
  sorry

end count_coprimes_15_l386_386658


namespace find_g2_l386_386471

open Function

variable (g : ℝ → ℝ)

axiom g_condition : ∀ x : ℝ, g x + 2 * g (1 - x) = 5 * x ^ 2

theorem find_g2 : g 2 = -10 / 3 :=
by {
  sorry
}

end find_g2_l386_386471


namespace additional_charge_per_segment_l386_386799

theorem additional_charge_per_segment :
  ∀ (initial_fee total_charge distance : ℝ), 
    initial_fee = 2.35 →
    total_charge = 5.5 →
    distance = 3.6 →
    (total_charge - initial_fee) / (distance / (2 / 5)) = 0.35 :=
by
  intros initial_fee total_charge distance h_initial_fee h_total_charge h_distance
  sorry

end additional_charge_per_segment_l386_386799


namespace arithmetic_seq_middle_term_l386_386524

theorem arithmetic_seq_middle_term (a1 a3 y : ℤ) (h1 : a1 = 3^2) (h2 : a3 = 3^4)
    (h3 : y = (a1 + a3) / 2) : y = 45 :=
by
  rw [h1, h2] at h3
  simp at h3
  exact h3

end arithmetic_seq_middle_term_l386_386524


namespace missing_number_l386_386240

theorem missing_number (x : ℤ) : 1234562 - 12 * x * 2 = 1234490 ↔ x = 3 :=
by
sorry

end missing_number_l386_386240


namespace find_length_of_BC_l386_386789

variable (A B C D : Type) [metric_space A]
variable (a b c d : ℝ)
variable (AB AC AD BD CD BC : ℝ)

axiom AB_eq : AB = 13
axiom AC_eq : AC = 15
axiom AD_eq : AD = 12
axiom BD_eq : BD = real.sqrt (AB^2 - AD^2)
axiom CD_eq : CD = real.sqrt (AC^2 - AD^2)
axiom BC_eq : BC = BD + CD

theorem find_length_of_BC : BC = 14 :=
by
  rw [AB_eq, AC_eq, AD_eq] at BD_eq CD_eq BC_eq
  sorry

end find_length_of_BC_l386_386789


namespace find_a_l386_386160

def curve1 (x : ℝ) : ℝ := x + Real.log x
def curve2 (a x : ℝ) : ℝ := a * x^2 + (a + 2) * x + 1
def tangent_line_at_1 (x : ℝ) : ℝ := 2 * x - 1

theorem find_a (a : ℝ) :
  (∀ x : ℝ, tangent_line_at_1 x = curve2 a x) → a = 8 := by
  sorry

end find_a_l386_386160


namespace ast_in_S_ast_assoc_l386_386384

-- Define the set S of odd integers greater than 1
def S : Set ℤ := { x | x % 2 = 1 ∧ x > 1 }

-- Define δ(x) as the unique integer such that 2^δ(x) < x < 2^(δ(x) + 1)
noncomputable def δ (x : ℤ) : ℕ := Nat.find (⟨0, by
  have h1 : ∃ n : ℕ, 2 ^ n < x ∧ x < 2 ^ (n + 1) := sorry
  exact h1⟩)

-- Define the operation ∗ on S
def ast (a b : ℤ) : ℤ := 2 ^ (δ a - 1) * (b - 3) + a

-- Theorem statements
theorem ast_in_S (a b : ℤ) (ha : a ∈ S) (hb : b ∈ S) : ast a b ∈ S := sorry

theorem ast_assoc (a b c : ℤ) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) : ast (ast a b) c = ast a (ast b c) := sorry

end ast_in_S_ast_assoc_l386_386384


namespace inequality_solution_set_l386_386123

theorem inequality_solution_set (x : ℝ) : 
  ( (x - 1) / (x + 2) > 0 ) ↔ ( x > 1 ∨ x < -2 ) :=
by sorry

end inequality_solution_set_l386_386123


namespace min_value_sin_cos_l386_386475

theorem min_value_sin_cos (x : ℝ) : ∀ y, y = sin 2 * cos (2 * x) → y ≥ - (1 / 2) :=
by
  -- We intend to solve using the properties of sine and cosine functions.
  sorry

end min_value_sin_cos_l386_386475


namespace simplify_sqrt_eight_l386_386483

theorem simplify_sqrt_eight : Real.sqrt 8 = 2 * Real.sqrt 2 := sorry

end simplify_sqrt_eight_l386_386483


namespace arithmetic_sequence_y_value_l386_386514

theorem arithmetic_sequence_y_value :
  ∃ y : ℤ, (∃ a1 a3 : ℤ, a1 = 9 ∧ a3 = 81 ∧ y = (a1 + a3) / 2) → y = 45 :=
by
  sorry

end arithmetic_sequence_y_value_l386_386514


namespace transport_medical_supplies_l386_386139

theorem transport_medical_supplies :
  ∀ (dist_AB speed_truck speed_motorcycle : ℝ) (total_boxes : ℕ) (time_to_meet total_distance : ℝ),
    dist_AB = 360 →
    speed_truck = 40 →
    speed_motorcycle = 80 →
    total_boxes = 6 →
    time_to_meet = 8 + 2 / 3 →
    total_distance = 693 + 1 / 3 →
    (time_to_meet = 8 + 2 / 3 ∧ total_distance = 693 + 1 / 3) :=
by
  intros dist_AB speed_truck speed_motorcycle total_boxes time_to_meet total_distance
  assume h1 h2 h3 h4 h5 h6
  sorry

end transport_medical_supplies_l386_386139


namespace sunflower_tetrads_l386_386575

theorem sunflower_tetrads (chromosomes : ℕ) (tetrads : ℕ) :
  chromosomes = 34 → tetrads = 17 :=
by
  -- Given condition:
  -- If there are 34 chromosomes at the late stage of the second meiotic division
  assume h1: chromosomes = 34,

  -- To prove:
  -- Then the number of tetrads produced during meiosis is 17
  -- We conclude that tetrads = 17 by the inference process in the solution

  -- This will be the proof part which is skipped here with sorry
  sorry

end sunflower_tetrads_l386_386575


namespace moles_NaOH_combined_with_HCl_l386_386967

-- Definitions for given conditions
def NaOH : Type := Unit
def HCl : Type := Unit
def NaCl : Type := Unit
def H2O : Type := Unit

def balanced_reaction (nHCl nNaOH nNaCl nH2O : ℕ) : Prop :=
  nHCl = nNaOH ∧ nNaOH = nNaCl ∧ nNaCl = nH2O

def mole_mass_H2O : ℕ := 18

-- Given: certain amount of NaOH combined with 1 mole of HCl
def initial_moles_HCl : ℕ := 1

-- Given: 18 grams of H2O formed
def grams_H2O : ℕ := 18

-- Molar mass of H2O is approximately 18 g/mol, so 18 grams is 1 mole
def moles_H2O : ℕ := grams_H2O / mole_mass_H2O

-- Prove that number of moles of NaOH combined with HCl is 1 mole
theorem moles_NaOH_combined_with_HCl : 
  balanced_reaction initial_moles_HCl 1 1 moles_H2O →
  moles_H2O = 1 →
  1 = 1 :=
by
  intros h1 h2
  sorry

end moles_NaOH_combined_with_HCl_l386_386967


namespace imo_1977_p6_l386_386725

theorem imo_1977_p6 (a b q r : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_division : a^2 + b^2 = (a + b) * q + r) (h_remainder : r < a + b) (h_qr : q^2 + r = 1977) :
    { (a, b) } = { (50, 37), (50, 7), (37, 50), (7, 50) } := sorry

end imo_1977_p6_l386_386725


namespace brother_initially_had_17_l386_386832

/-
Michael has $42. Michael gives away half the money to his brother.
His brother buys $3 worth of candy. His brother has $35 left.
Prove that Michael's brother initially had $17.
-/

def michael_initial_money : ℕ := 42

def amount_given_to_brother : ℕ := michael_initial_money / 2

def candy_cost : ℕ := 3

def brother_money_left_after_candy : ℕ := 35

def brother_initial_money : ℕ :=
  brother_money_left_after_candy + candy_cost

theorem brother_initially_had_17 :
  brother_initial_money - amount_given_to_brother = 17 :=
by
  unfold michael_initial_money
  unfold amount_given_to_brother
  unfold candy_cost
  unfold brother_money_left_after_candy
  unfold brother_initial_money
  sorry

end brother_initially_had_17_l386_386832


namespace sheets_taken_l386_386826

noncomputable def remaining_sheets_mean (b c : ℕ) : ℚ :=
  (b * (2 * b + 1) + (100 - 2 * (b + c)) * (2 * (b + c) + 101)) / 2 / (100 - 2 * c)

theorem sheets_taken (b c : ℕ) (h1 : 100 = 2 * 50) 
(h2 : ∀ n, n > 0 → 2 * n = n + n) 
(hmean : remaining_sheets_mean b c = 31) : 
  c = 17 := 
sorry

end sheets_taken_l386_386826


namespace gcd_consecutive_term_max_l386_386214

def b (n : ℕ) : ℕ := n.factorial + 2^n + n 

theorem gcd_consecutive_term_max (n : ℕ) (hn : n ≥ 0) :
  ∃ m ≤ (n : ℕ), (m = 2) := sorry

end gcd_consecutive_term_max_l386_386214


namespace area_PQR_is_4_5_l386_386502

noncomputable def point := (ℝ × ℝ)

def P : point := (2, 1)
def Q : point := (1, 4)
def R_line (x: ℝ) : point := (x, 6 - x)

def area_triangle (A B C : point) : ℝ :=
  0.5 * abs ((A.1 * B.2 + B.1 * C.2 + C.1 * A.2) - (A.2 * B.1 + B.2 * C.1 + C.2 * A.1))

theorem area_PQR_is_4_5 (x : ℝ) (h : R_line x ∈ {p : point | p.1 + p.2 = 6}) : 
  area_triangle P Q (R_line x) = 4.5 :=
    sorry

end area_PQR_is_4_5_l386_386502


namespace range_of_quadratic_expression_l386_386329

theorem range_of_quadratic_expression :
  (∀ x : ℝ, (x - 1) * (x - 2) < 2 → (x + 1) * (x - 3) ∈ set.Ioo (-4 : ℝ) 0) :=
by sorry

end range_of_quadratic_expression_l386_386329


namespace choose_3_captains_from_11_l386_386020

theorem choose_3_captains_from_11 : nat.choose 11 3 = 165 := 
by sorry

end choose_3_captains_from_11_l386_386020


namespace line_equation_through_intersection_circle_equation_given_conditions_l386_386255

theorem line_equation_through_intersection
  (x y : ℝ)
  (h1 : 2 * x + y - 8 = 0)
  (h2 : x - 2 * y + 1 = 0)
  (h_perp : ∀ x y C, 6 * x - 8 * y + 3 = 0 → 4 * x + 3 * y + C - 18 = 0) :
  ∃ (C : ℝ), 4 * x + 3 * y - C = 18 :=
begin
  sorry
end

theorem circle_equation_given_conditions
  (a x y : ℝ)
  (h_center : y = 0)
  (h_C : ∀ p, p = (-1,1) ∨ p = (1,3))
  (h_eq : (a + 1)^2 + 1 = (a - 1)^2 + 9) :
  ∃ r, (x - a)^2 + y^2 = r :=
begin
  sorry
end

end line_equation_through_intersection_circle_equation_given_conditions_l386_386255


namespace dot_product_scaled_vec_l386_386203

def vec_a : Fin 3 → ℝ := (![4, -5, 2] : Fin 3 → ℝ)
def vec_b : Fin 3 → ℝ := (![ -3, 6, -4] : Fin 3 → ℝ)
def scaled_vec_a : Fin 3 → ℝ := 3 • vec_a

theorem dot_product_scaled_vec (a b : Fin 3 → ℝ) : 
  (scaled_vec_a ⬝ vec_b) = -150 :=
by sorry

end dot_product_scaled_vec_l386_386203


namespace inequality_proof_l386_386714

variables {n : ℕ} {a : ℕ → ℝ} (d s : ℝ)

-- Definition of d
def max_min_diff (a : ℕ → ℝ) (n : ℕ) : ℝ := 
  a n - a 1

-- Definition of s
def sum_abs_diffs (a : ℕ → ℝ) (n : ℕ) : ℝ := 
  sum (abs (a i - a j)) for all 1 ≤ i < j ≤ n

-- Statement of the problem
theorem inequality_proof (h1 : ∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ n) (h2 : max_min_diff a n = d) (h3 : sum_abs_diffs a n = s) :
  (n-1) * d ≤ s ∧ s ≤ (n^2 / 4) * d := 
sorry

end inequality_proof_l386_386714


namespace smallest_n_not_divisible_by_10_l386_386269

theorem smallest_n_not_divisible_by_10 :
  ∃ n > 2016, (1^n + 2^n + 3^n + 4^n) % 10 ≠ 0 ∧ n = 2020 := 
sorry

end smallest_n_not_divisible_by_10_l386_386269


namespace slowest_bailing_rate_l386_386098

-- Definitions and conditions.
def distance_to_shore : ℝ := 1.5
def water_intake_rate : ℝ := 12
def boat_capacity : ℝ := 45
def rowing_speed : ℝ := 4
def bailing_efficiency : ℝ := 0.9

-- Time to reach shore in minutes
def time_to_shore : ℝ := (distance_to_shore / rowing_speed) * 60

-- Total water intake in the time it takes to reach shore
def total_water_intake : ℝ := water_intake_rate * time_to_shore

-- Define the slowest bailing rate to prove it must be at least 11 gallons per minute
theorem slowest_bailing_rate (r : ℝ) : (total_water_intake - (bailing_efficiency * r * time_to_shore) <= boat_capacity) → r ≥ 11 := by
  sorry

end slowest_bailing_rate_l386_386098


namespace monthly_income_of_B_l386_386940

theorem monthly_income_of_B :
  ∀ (x y : ℕ), 
  (5 * x - 3 * y = 1800) ∧ (6 * x - 4 * y = 1600) →
  6 * x = 7200 :=
by
  intros x y h,
  sorry

end monthly_income_of_B_l386_386940


namespace count_coprime_with_15_lt_15_l386_386648

theorem count_coprime_with_15_lt_15 :
  {a : ℕ // a < 15 ∧ Nat.coprime 15 a}.to_finset.card = 8 := 
sorry

end count_coprime_with_15_lt_15_l386_386648


namespace total_flying_days_l386_386793

-- Definitions for the conditions
def days_fly_south_winter := 40
def days_fly_north_summer := 2 * days_fly_south_winter
def days_fly_east_spring := 60

-- Theorem stating the total flying days
theorem total_flying_days : 
  days_fly_south_winter + days_fly_north_summer + days_fly_east_spring = 180 :=
  by {
    -- This is where we would prove the theorem
    sorry
  }

end total_flying_days_l386_386793


namespace sphere_hemisphere_volume_ratio_l386_386882

theorem sphere_hemisphere_volume_ratio (r : ℝ) (π : ℝ) (hr : π ≠ 0): 
  let V_sphere := (4 / 3) * π * r^3,
      V_hemisphere := (1 / 2) * (4 / 3) * π * (3 * r)^3
  in V_sphere / V_hemisphere = 1 / 13.5 := 
by {
  let V_sphere := (4 / 3) * π * r^3,
      V_hemisphere := (1 / 2) * (4 / 3) * π * (3 * r)^3;
  have : V_hemisphere = (4 / 3) * π * (13.5 * r^3), {
    sorry
  },
  have ratio := V_sphere / V_hemisphere,
  rw this at ratio,
  simp [V_sphere, V_hemisphere],
  field_simp [hr],
  norm_num,
  rw ←mul_assoc,
  field_simp,
  norm_num,
}

end sphere_hemisphere_volume_ratio_l386_386882


namespace total_of_four_numbers_l386_386273

theorem total_of_four_numbers :
  let a := 1/3
  let b := 5/24
  let c := 8.16
  let d := 1/8
  abs (a + b + c + d - 8.83) < 0.01 :=
by
  sorry

end total_of_four_numbers_l386_386273


namespace sum_proper_divisors_243_l386_386926

theorem sum_proper_divisors_243 : (1 + 3 + 9 + 27 + 81) = 121 := by
  sorry

end sum_proper_divisors_243_l386_386926


namespace triangle_ratio_l386_386627

theorem triangle_ratio (α β γ : ℝ) (a b c : ℝ) 
  (h1 : sin α = 3 / 5) 
  (h2 : cos β = 5 / 13) 
  (h3 : a > 0) 
  (h4 : b > 0) 
  (h5 : c > 0) 
  (h6 : α + β + γ = π) 
  (h7 : 0 < α < π / 2)
  (h8 : 0 < β < π / 2)
  (h9 : 0 < γ < π / 2) :
  (a ^ 2 + b ^ 2 - c ^ 2) / (a * b) = 32 / 65 := 
by
  sorry

end triangle_ratio_l386_386627


namespace rhombus_perimeter_l386_386455

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) (h_bisect : ∀ x, rhombus_diagonals_bisect x) :
  ∃ P, P = 52 := by
  sorry

end rhombus_perimeter_l386_386455


namespace lap_time_improvement_l386_386404

theorem lap_time_improvement (initial_laps : ℕ) (initial_time : ℕ) (current_laps : ℕ) (current_time : ℕ)
  (h1 : initial_laps = 15) (h2 : initial_time = 45) (h3 : current_laps = 18) (h4 : current_time = 42) :
  (45 / 15 - 42 / 18 : ℚ) = 2 / 3 :=
by
  sorry

end lap_time_improvement_l386_386404


namespace count_coprimes_15_l386_386654

def count_coprimes_less_than (n : ℕ) : ℕ :=
  (Finset.range n).filter (λ a => Nat.gcd a n = 1).card

theorem count_coprimes_15 :
  count_coprimes_less_than 15 = 8 :=
by
  sorry

end count_coprimes_15_l386_386654


namespace range_of_fraction_l386_386296

theorem range_of_fraction (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a ≤ 2 * b) (h4 : 2 * b ≤ 2 * a + b) :
  ∃ low high : ℝ, low = 4 / 9 ∧ high = real.sqrt 2 / 2 ∧ low ≤ (2 * a * b) / (a^2 + 2 * b^2) ∧ (2 * a * b) / (a^2 + 2 * b^2) ≤ high :=
by 
  sorry

end range_of_fraction_l386_386296


namespace find_1001st_term_l386_386105

def sum_of_squares_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.map (λ d, d * d) |>.sum

def sequence (n : ℕ) : ℕ :=
match n with
| 0 => 3243
| k + 1 => sum_of_squares_of_digits (sequence k) + 2

lemma sequence_periodic_from_6 :
  ∀ n ≥ 6, sequence (n + 3) = sequence n := sorry

theorem find_1001st_term : sequence 1001 = 51 := sorry

end find_1001st_term_l386_386105


namespace find_f_of_8_l386_386108

-- Define the power function passing through (4, 1/2)
def power_function (α : ℝ) (x : ℝ) : ℝ := x ^ α 

theorem find_f_of_8 : ∃ α : ℝ, power_function α 4 = 1 / 2 ∧ power_function α 8 = sqrt 2 / 4 :=
by
  use (-1 / 2)
  split
  { -- Condition that the power function passes through (4, 1/2)
    show power_function (-1 / 2) 4 = 1 / 2
    sorry },
  { -- Condition to prove that f(8) equals sqrt(2)/4 
    show power_function (-1 / 2) 8 = sqrt 2 / 4
    sorry }

end find_f_of_8_l386_386108


namespace minimum_value_of_f_l386_386147

def f (x : ℝ) : ℝ := 3 * x^3 - 9 * x^2 + 6 * x + 1

theorem minimum_value_of_f :
  exists (x : ℝ), x = 1 + 1 / Real.sqrt 3 ∧ ∀ (y : ℝ), f (1 + 1 / Real.sqrt 3) ≤ f y := sorry

end minimum_value_of_f_l386_386147


namespace michael_brother_initial_money_l386_386835

theorem michael_brother_initial_money :
  let michael_money := 42
  let brother_receive := michael_money / 2
  let candy_cost := 3
  let brother_left_after_candy := 35
  let brother_initial_money := brother_left_after_candy + candy_cost
  in brother_initial_money - brother_receive = 17 :=
by
  intros
  -- definitions
  let michael_money := 42
  let brother_receive := michael_money / 2
  let candy_cost := 3
  let brother_left_after_candy := 35
  let brother_initial_money := brother_left_after_candy + candy_cost
  -- result
  show brother_initial_money - brother_receive = 17
  sorry

end michael_brother_initial_money_l386_386835


namespace mod_99_equal_sum_pairs_mod_99_l386_386859

theorem mod_99_equal_sum_pairs_mod_99 (N : ℕ) (digits : list ℕ) 
  (hN_digits : digits.map (λ i, i / 10) = digits.zip_with (λ x y, x * 10 + y) digits.tail digits.tail.tail) :
  let sum_pairs := (list.sum (digits.zip_with (λ x y, x * 10 + y) digits.tail digits.tail.tail) + digits.head) in
  N % 99 = sum_pairs % 99 :=
by {
  sorry
}

end mod_99_equal_sum_pairs_mod_99_l386_386859


namespace isosceles_triangle_vertex_angle_l386_386604

theorem isosceles_triangle_vertex_angle (T : Triangle) (α : ℝ) 
  (h_isosceles : is_isosceles T)
  (exists_splitting_line : ∃ D : Point, ∀ (R S : Triangle), 
    splits_into_isosceles_triangles T D R S ∧ 
    ¬(is_similar R T ∨ is_similar S T)) :
  α = 36 :=
sorry

end isosceles_triangle_vertex_angle_l386_386604


namespace part_1_part_2_l386_386363

variables {a b : ℕ → ℤ}
variables {m n : ℕ}

/-- Conditions --/
def conditions :=
  a 1 = 2 ∧ b 1 = 4 ∧
  (∀ n, 2 * b n = a n + a (n + 1)) ∧
  (∀ n, 2 * a n = b n + b (n + 1))

/-- Part (1): Prove {a_n + b_n} is geometric --/
theorem part_1 (h : conditions) : 
  ∃ r, ∀ n, a n + b n = 6 * r^n :=
sorry

/-- Part (2): Find all pairs (m, n) satisfying the given ratio condition --/
theorem part_2 (h : conditions) (h_m : 0 < m ∧ m ≤ 100) : 
  (∃ m n, m = 8 ∧ n = 9) ∨ (m = 80 ∧ n = 83) :=
sorry

end part_1_part_2_l386_386363


namespace original_wine_in_jug_l386_386022

theorem original_wine_in_jug :
  ∃ x : ℝ, (x = 0.875) ∧ (2 * (2 * (2 * x - 1) - 1) - 1 = 0) :=
begin
  sorry
end

end original_wine_in_jug_l386_386022


namespace same_terminal_side_angle_in_range_0_to_2pi_l386_386361

theorem same_terminal_side_angle_in_range_0_to_2pi :
  ∃ k : ℤ, 0 ≤ 2 * k * π + (-4) * π / 3 ∧ 2 * k * π + (-4) * π / 3 ≤ 2 * π ∧
  2 * k * π + (-4) * π / 3 = 2 * π / 3 :=
by
  use 1
  sorry

end same_terminal_side_angle_in_range_0_to_2pi_l386_386361


namespace tutors_next_together_in_360_days_l386_386692

open Nat

-- Define the intervals for each tutor
def evan_interval := 5
def fiona_interval := 6
def george_interval := 9
def hannah_interval := 8
def ian_interval := 10

-- Statement to prove
theorem tutors_next_together_in_360_days :
  Nat.lcm (Nat.lcm evan_interval fiona_interval) (Nat.lcm george_interval (Nat.lcm hannah_interval ian_interval)) = 360 :=
by
  sorry

end tutors_next_together_in_360_days_l386_386692


namespace frank_final_amount_l386_386698

/-- 
Frank bought the following items for his breakfast: 
 - 15 buns at $0.20 each, 
 - 4 bottles of milk at $2.30 each, 
 - a carton of eggs at 4 times the price of one bottle of milk, 
 - a jar of jam at $3.75, and 
 - a pack of bacon at $5.25.

The shop had these promotions:
- Buy two buns get one free,
- a 2-for-1 deal on bottles of milk, and
- a 10% discount on eggs.

Additionally, the total bill was subject to:
- a 5% sales tax applied to non-dairy items (buns, eggs, jam, bacon), and
- a 1% sales tax applied to dairy items (milk).

This Lean statement proves that the final amount Frank paid for his breakfast shopping is $25.89.
-/
theorem frank_final_amount :
  let bun_price := 0.20
  let buns := 15
  let milk_price := 2.30
  let milk_bottles := 4
  let eggs_price := 4 * milk_price
  let eggs_discount := 0.10
  let jam_price := 3.75
  let bacon_price := 5.25
  let buns_discounted := (buns - buns / 3) * bun_price
  let milk_discounted := (milk_bottles - milk_bottles / 2) * milk_price
  let eggs_discounted := eggs_price * (1 - eggs_discount)
  let non_dairy_cost := buns_discounted + eggs_discounted + jam_price + bacon_price
  let dairy_cost := milk_discounted
  let non_dairy_tax := 0.05
  let dairy_tax := 0.01
  let tax := non_dairy_cost * non_dairy_tax + dairy_cost * dairy_tax
  let total_cost := non_dairy_cost + dairy_cost + tax
 in total_cost = 25.89 := by
  -- Definitions and conditions
  let bun_price := 0.20
  let buns := 15
  let milk_price := 2.30
  let milk_bottles := 4
  let eggs_price := 4 * milk_price
  let eggs_discount := 0.10
  let jam_price := 3.75
  let bacon_price := 5.25
  let buns_discounted := (buns - buns / 3) * bun_price
  let milk_discounted := (milk_bottles - milk_bottles / 2) * milk_price
  let eggs_discounted := eggs_price * (1 - eggs_discount)
  let non_dairy_cost := buns_discounted + eggs_discounted + jam_price + bacon_price
  let dairy_cost := milk_discounted
  let non_dairy_tax := 0.05
  let dairy_tax := 0.01
  let tax := non_dairy_cost * non_dairy_tax + dairy_cost * dairy_tax
  let total_cost := non_dairy_cost + dairy_cost + tax
  -- Proof calculation
  have h1: buns_discounted = 2.00 := by sorry
  have h2: milk_discounted = 4.60 := by sorry
  have h3: eggs_discounted = 8.28 := by sorry
  have h4: total_non_dairy := h1 + h3 + jam_price + bacon_price := by sorry
  have h5: total_dairy := h2 := by sorry
  have h6: tax := total_non_dairy * non_dairy_tax + total_dairy * dairy_tax := by sorry
  have h7: total_cost = total_non_dairy + total_dairy + tax := by sorry
  show total_cost = 25.89 := by simp [h7]

end frank_final_amount_l386_386698


namespace sum_lent_is_3000_l386_386173

noncomputable def principal_sum (P : ℕ) : Prop :=
  let R := 5
  let T := 5
  let SI := (P * R * T) / 100
  SI = P - 2250

theorem sum_lent_is_3000 : ∃ (P : ℕ), principal_sum P ∧ P = 3000 :=
by
  use 3000
  unfold principal_sum
  -- The following are the essential parts
  sorry

end sum_lent_is_3000_l386_386173


namespace total_birds_l386_386563

-- Definitions from conditions
def num_geese : ℕ := 58
def num_ducks : ℕ := 37

-- Proof problem statement
theorem total_birds : num_geese + num_ducks = 95 := by
  sorry

end total_birds_l386_386563


namespace breaks_difference_l386_386048

-- James works for 240 minutes
def total_work_time : ℕ := 240

-- He takes a water break every 20 minutes
def water_break_interval : ℕ := 20

-- He takes a sitting break every 120 minutes
def sitting_break_interval : ℕ := 120

-- Calculate the number of water breaks James takes
def number_of_water_breaks : ℕ := total_work_time / water_break_interval

-- Calculate the number of sitting breaks James takes
def number_of_sitting_breaks : ℕ := total_work_time / sitting_break_interval

-- Prove the difference between the number of water breaks and sitting breaks is 10
theorem breaks_difference :
  number_of_water_breaks - number_of_sitting_breaks = 10 :=
by
  -- calculate number_of_water_breaks = 12
  -- calculate number_of_sitting_breaks = 2
  -- check the difference 12 - 2 = 10
  sorry

end breaks_difference_l386_386048


namespace jerky_fulfillment_time_l386_386914

-- Defining the conditions
def bags_per_batch : ℕ := 10
def order_quantity : ℕ := 60
def existing_bags : ℕ := 20
def batch_time_in_nights : ℕ := 1

-- Define the proposition to be proved
theorem jerky_fulfillment_time :
  let additional_bags := order_quantity - existing_bags in
  let batches_needed := additional_bags / bags_per_batch in
  let days_needed := batches_needed * batch_time_in_nights in
  days_needed = 4 :=
begin
  sorry
end

end jerky_fulfillment_time_l386_386914


namespace arithmetic_sequence_sum_l386_386730

theorem arithmetic_sequence_sum (a_4 a_5 k : ℕ) (S : ℕ → ℕ) 
  (h1 : a_4 + a_5 = 20) 
  (h2 : a_4 * a_5 = 99) 
  (h3 : ∀ n : ℕ, S n ≤ S k) 
  (a_n : ℕ → ℕ) (h4 : ∀ n : ℕ, a_n n = a_4 + (n - 4) * (-2)) :
  k = 9 := 
sorry

end arithmetic_sequence_sum_l386_386730


namespace solution_set_l386_386434

noncomputable def satisfies_equations (x y : ℝ) : Prop :=
  (x^2 + 3 * x * y = 12) ∧ (x * y = 16 + y^2 - x * y - x^2)

theorem solution_set :
  {p : ℝ × ℝ | satisfies_equations p.1 p.2} = {(4, 1), (-4, -1), (-4, 1), (4, -1)} :=
by sorry

end solution_set_l386_386434


namespace coprime_count_15_l386_386662

theorem coprime_count_15 :
  {a : ℕ | a < 15 ∧ Nat.gcd 15 a = 1}.card = 8 :=
by
sorry

end coprime_count_15_l386_386662


namespace consecutive_integer_quadratic_l386_386978

theorem consecutive_integer_quadratic :
  ∃ (a b c : ℤ) (x₁ x₂ : ℤ),
  (a * x₁ ^ 2 + b * x₁ + c = 0 ∧ a * x₂ ^ 2 + b * x₂ + c = 0) ∧
  (a = 2 ∧ b = 0 ∧ c = -2) ∨ (a = -2 ∧ b = 0 ∧ c = 2) := sorry

end consecutive_integer_quadratic_l386_386978


namespace rhombus_perimeter_l386_386457

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) (h_bisect : ∀ x, rhombus_diagonals_bisect x) :
  ∃ P, P = 52 := by
  sorry

end rhombus_perimeter_l386_386457


namespace perimeter_difference_zero_l386_386217

theorem perimeter_difference_zero :
  let shape1_length := 4
  let shape1_width := 3
  let shape2_length := 6
  let shape2_width := 1
  let perimeter (l w : ℕ) := 2 * (l + w)
  perimeter shape1_length shape1_width = perimeter shape2_length shape2_width :=
by
  sorry

end perimeter_difference_zero_l386_386217


namespace part1_part2_l386_386728

def P (f : ℝ → ℝ) (m : ℝ) : Prop :=
  0 < m ∧ m < 2 ∧ ∃ x_0 ∈ set.Icc (0 : ℝ) (2 - m), f x_0 = f (x_0 + m)

theorem part1 :
  P (λ x, real.sqrt (1 - (x - 1)^2)) (1 / 2) :=
sorry

theorem part2 : 
  ∀ m ∈ set.Ioo (0 : ℝ) 2, P (λ x, (x - 1)^2) m :=
sorry

end part1_part2_l386_386728


namespace range_of_f_l386_386261

noncomputable def f (x : ℝ) : ℝ := (3 * x - 5) / (x + 4)

theorem range_of_f : set.range f = {y : ℝ | y < 3 ∨ y > 3} :=
by
  sorry

end range_of_f_l386_386261


namespace trains_crossing_time_l386_386508

-- Define the given conditions
def length_train1 : ℝ := 200 -- meters
def length_train2 : ℝ := 800 -- meters
def speed_train1 : ℝ := 60 -- km/hr
def speed_train2 : ℝ := 40 -- km/hr

-- Conversion factor from km/hr to m/s
def km_per_hr_to_m_per_s : ℝ := 5 / 18

-- Define the relative speed when trains are moving in opposite directions
def relative_speed_m_per_s : ℝ := (speed_train1 + speed_train2) * km_per_hr_to_m_per_s

-- Define the total distance to be covered
def total_distance : ℝ := length_train1 + length_train2

-- Define the expected time to cross each other
def expected_time_to_cross : ℝ := total_distance / relative_speed_m_per_s

-- State the theorem to prove
theorem trains_crossing_time :
  expected_time_to_cross = 36 := 
sorry

end trains_crossing_time_l386_386508


namespace no_valid_values_of_k_for_prime_roots_l386_386999

theorem no_valid_values_of_k_for_prime_roots :
  ¬ ∃ (k : ℕ), ∀ (p q : ℕ), (p + q = 97 ∧ p * q = k) → (nat.prime p ∧ nat.prime q) := by
  sorry

end no_valid_values_of_k_for_prime_roots_l386_386999


namespace smallest_positive_period_pi_not_odd_at_theta_pi_div_4_axis_of_symmetry_at_pi_div_3_max_value_not_1_on_interval_l386_386308

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 6)

-- Statement A: The smallest positive period of f(x) is π.
theorem smallest_positive_period_pi : 
  ∀ x : ℝ, f (x + Real.pi) = f x :=
by sorry

-- Statement B: If f(x + θ) is an odd function, then one possible value of θ is π/4.
theorem not_odd_at_theta_pi_div_4 : 
  ¬ (∀ x : ℝ, f (x + Real.pi / 4) = -f x) :=
by sorry

-- Statement C: A possible axis of symmetry for f(x) is the line x = π / 3.
theorem axis_of_symmetry_at_pi_div_3 :
  ∀ x : ℝ, f (Real.pi / 3 - x) = f (Real.pi / 3 + x) :=
by sorry

-- Statement D: The maximum value of f(x) on [0, π / 4] is 1.
theorem max_value_not_1_on_interval : 
  ¬ (∀ x ∈ Set.Icc 0 (Real.pi / 4), f x ≤ 1) :=
by sorry

end smallest_positive_period_pi_not_odd_at_theta_pi_div_4_axis_of_symmetry_at_pi_div_3_max_value_not_1_on_interval_l386_386308


namespace solve_equation_l386_386250

theorem solve_equation (z : ℤ) (h : sqrt (10 + 3 * z) = 8) : z = 18 :=
by
  sorry

end solve_equation_l386_386250


namespace eggs_left_after_taking_l386_386899

def eggs_in_box_initial : Nat := 47
def eggs_taken_by_Harry : Nat := 5
theorem eggs_left_after_taking : eggs_in_box_initial - eggs_taken_by_Harry = 42 := 
by
  -- Proof placeholder
  sorry

end eggs_left_after_taking_l386_386899


namespace number_of_spacy_subsets_S15_l386_386678

noncomputable def spacy_subsets : ℕ → ℕ
| 1       := 2
| 2       := 3
| 3       := 4
| (n + 1) := spacy_subsets n + (if n ≥ 2 then spacy_subsets (n - 2) else 0)

theorem number_of_spacy_subsets_S15 : spacy_subsets 15 = 406 := by
  sorry

end number_of_spacy_subsets_S15_l386_386678


namespace frog_jump_distance_l386_386109

theorem frog_jump_distance (grasshopper_jump : ℕ) (extra_jump : ℕ) (frog_jump : ℕ) :
  grasshopper_jump = 9 → extra_jump = 3 → frog_jump = grasshopper_jump + extra_jump → frog_jump = 12 :=
by
  intros h_grasshopper h_extra h_frog
  rw [h_grasshopper, h_extra] at h_frog
  exact h_frog

end frog_jump_distance_l386_386109


namespace arithmetic_sequence_ratio_l386_386695

theorem arithmetic_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n, S n = (n * (a 1 + a n)) / 2)
  (h2 : ∀ n, S n / a n = (n + 1) / 2) :
  (a 2 / a 3 = 2 / 3) :=
sorry

end arithmetic_sequence_ratio_l386_386695


namespace find_n_l386_386266

theorem find_n (n : ℕ) (h : n > 2016) (h_not_divisible : ¬ (1^n + 2^n + 3^n + 4^n) % 10 = 0) : n = 2020 :=
sorry

end find_n_l386_386266


namespace polynomial_solution_l386_386386

noncomputable def P : ℝ → ℝ := sorry

theorem polynomial_solution (x : ℝ) :
  (∃ P : ℝ → ℝ, (∀ x, P x = (P 0) + (P 1) * x + (P 2) * x^2) ∧ 
  (P (-2) = 4)) →
  (P x = (4 * x^2 - 6 * x) / 7) :=
by
  sorry

end polynomial_solution_l386_386386


namespace green_weight_is_3_l386_386607

variable (G : ℕ)

axiom blue_weight : G → 2
axiom green_weights_var : G
axiom blue_weights_count : 4
axiom green_weights_count : 5
axiom bar_weight : 2
axiom total_weight : 25

theorem green_weight_is_3 :
  8 + 5 * G + 2 = 25 → G = 3 :=
by
  intros h
  linarith [h]

# check green_weight_is_3

end green_weight_is_3_l386_386607


namespace ball_hits_ground_at_approx_2_9875_l386_386571

-- Define the problem parameters
def initial_velocity := 36  -- feet per second
def initial_height := 250  -- feet
def height_eq (t : ℝ) : ℝ := -16 * t ^ 2 - initial_velocity * t + initial_height

-- Define the time when the ball hits the ground
def hit_ground_time := 2.9875  -- seconds

-- State the theorem
theorem ball_hits_ground_at_approx_2_9875 : 
  ∃ t : ℝ, height_eq t = 0 ∧ abs (t - hit_ground_time) < 0.0001 :=
by
  sorry

end ball_hits_ground_at_approx_2_9875_l386_386571


namespace cersei_ate_candies_l386_386617

variable (initial : ℕ) (gaveBrother gaveSister gaveCousin final : ℕ)
variable (oneFourth : ℕ → ℕ)

def oneFourth (n : ℕ) : ℕ := n / 4

theorem cersei_ate_candies :
  initial = 50 →
  gaveBrother = 5 →
  gaveSister = 5 →
  gaveCousin = oneFourth (initial - (gaveBrother + gaveSister)) →
  final = 18 →
  initial - (gaveBrother + gaveSister + gaveCousin) - final = 12 := 
by
  intros h1 h2 h3 h4 h5
  -- proof steps here omitted
  sorry

end cersei_ate_candies_l386_386617


namespace radius_of_tangent_circle_l386_386675

theorem radius_of_tangent_circle
  (side_length : ℝ)
  (diameter_of_semicircles : ℝ)
  (r : ℝ)
  (h1 : side_length = 2)
  (h2 : diameter_of_semicircles = 1)
  (h3 : r = (Real.sqrt 5 - 1) / 2) :
  let radius_of_semicircles := diameter_of_semicircles / 2 in
  let distance_from_center_square_to_semicircle := Real.sqrt (1^2 + (side_length / 4)^2) in
  let distance_from_center_square_to_inner_circle := distance_from_center_square_to_semicircle - r in
  distance_from_center_square_to_inner_circle = r :=
sorry

end radius_of_tangent_circle_l386_386675


namespace find_angle_B_and_sin_C_l386_386337

theorem find_angle_B_and_sin_C (a b c A B C: ℝ) 
  (h1 : ∀ (a b c A B C: ℝ), a * sin (2 * B) = sqrt 3 * b * sin A) 
  (h2 : ∀ (a b c A B C: ℝ), cos A = 1 / 3)
  (h3 : ∀ (a b c A B C: ℝ), angle A + angle B + angle C = π):
  B = π / 6 ∧ sin C = (2 * sqrt 6 + 1) / 6 :=
by sorry

end find_angle_B_and_sin_C_l386_386337


namespace polynomial_coefficients_sum_l386_386118

theorem polynomial_coefficients_sum (f : ℤ -> ℤ) (a p q r : ℤ) :
  (∀ x : ℤ, f(x) - f(x-2) = (2*x-1)^2) →
  (∀ x : ℤ, f(x) = a*x^3 + p*x^2 + q*x + r) →
  p + q = 5/6 :=
by
  intro h1 h2
  sorry

end polynomial_coefficients_sum_l386_386118


namespace correct_propositions_l386_386424

theorem correct_propositions
  (m n : Type)
  (α β : Type)
  (perpendicular : Π (x y : Type), Prop)
  (parallel : Π (x y : Type), Prop)
  (h1 : perpendicular m α)
  (h2 : perpendicular n β)
  (h3 : perpendicular α β)
  (h4 : parallel m α)
  (h5 : parallel n β)
  (h6 : parallel α β) :
    ({p | (p = "①" ∧ perpendicular m n) ∨ 
          (p = "②" ∧ parallel m n) ∨ 
          (p = "③" ∧ perpendicular m n) ∨ 
          (p = "④" ∧ parallel m n)} = {"①", "③"}) :=
begin
  sorry
end

end correct_propositions_l386_386424


namespace sphere_to_hemisphere_ratio_l386_386881

-- Definitions of the radii of the sphere and hemisphere
def r : ℝ := sorry -- We assume r is a positive real number, but not providing a specific value here

-- Volume of the sphere
def volume_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

-- Volume of the hemisphere with radius 3r
def volume_hemisphere (r : ℝ) : ℝ := (1 / 2) * volume_sphere (3 * r)

-- Ratio of volumes
noncomputable def ratio_volumes (r : ℝ) : ℝ := volume_sphere r / volume_hemisphere r

-- Statement to prove
theorem sphere_to_hemisphere_ratio : ratio_volumes r = 1 / 13.5 :=
by
  sorry

end sphere_to_hemisphere_ratio_l386_386881


namespace digits_divisible_81_l386_386567

theorem digits_divisible_81 (a : Fin 2016 → ℕ) (N : ℕ) (hN : 81 ∣ N) :
  81 ∣ (a 2015) * 10^(2015) + (a 2014) * 10^(2014) + ... + (a 0) * 10^(0) :=
sorry

end digits_divisible_81_l386_386567


namespace rectangle_sides_l386_386183

-- Let's define the variables and premises
variables {x : ℝ}

-- Given conditions in the problem
def triangle_base : ℝ := 48
def triangle_height : ℝ := 16
def longer_side_on_base (x : ℝ) : Prop := 9 * x * 48 = 16 * (16 - 5 * x)

-- The statement of the theorem
theorem rectangle_sides (h : longer_side_on_base x) :
  5 * x = 10 ∧ 9 * x = 18 :=
begin
  sorry
end

end rectangle_sides_l386_386183


namespace preimage_of_neg1_2_l386_386001

def f (x y : ℝ) : ℝ × ℝ := (2 * x, x - y)

theorem preimage_of_neg1_2 : ∃ (x y : ℝ), (f x y) = (-1, 2) ∧ x = -1/2 ∧ y = -5/2 := 
by
  use -1/2, -5/2
  simp [f]
  split
  · ring
  · ring
  sorry

end preimage_of_neg1_2_l386_386001


namespace find_water_needed_l386_386075

def apple_juice := 4
def honey (A : ℕ) := 3 * A
def water (H : ℕ) := 3 * H

theorem find_water_needed : water (honey apple_juice) = 36 :=
  sorry

end find_water_needed_l386_386075


namespace basketball_rim_height_l386_386120

theorem basketball_rim_height
    (height_in_inches : ℕ)
    (reach_in_inches : ℕ)
    (jump_in_inches : ℕ)
    (above_rim_in_inches : ℕ) :
    height_in_inches = 72
    → reach_in_inches = 22
    → jump_in_inches = 32
    → above_rim_in_inches = 6
    → (height_in_inches + reach_in_inches + jump_in_inches - above_rim_in_inches) = 120 :=
by
  intros h1 h2 h3 h4
  sorry

end basketball_rim_height_l386_386120


namespace sum_of_areas_of_squares_l386_386993

theorem sum_of_areas_of_squares (A B E : Point) (h_right_angle : angle A E B = 90)
  (h_BE : dist B E = 9) : area_square (square_ABCD A B) + area_square (square_AEFG A E) = 81 := 
sorry

end sum_of_areas_of_squares_l386_386993


namespace rhombus_perimeter_l386_386463

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) : 
  let a := d1 / 2 in
  let b := d2 / 2 in
  let c := Real.sqrt (a^2 + b^2) in
  let side := c in
  let perimeter := 4 * side in
  perimeter = 52 := 
by 
  let a := 5 in 
  let b := 12 in 
  have h3 : a = d1 / 2, by rw [h1]; norm_num
  have h4 : b = d2 / 2, by rw [h2]; norm_num
  let c := Real.sqrt (5^2 + 12^2),
  let side := c,
  have h5 : c = 13, by norm_num,
  let perimeter := 4 * 13,
  show perimeter = 52, by norm_num; sorry

end rhombus_perimeter_l386_386463


namespace subtraction_problem_solution_l386_386031

noncomputable def subtraction_problem : Prop :=
  ∃ (A B C D E F : ℕ), 
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    A = 9 ∧ B = 8 ∧
    AB = 10 * A + B ∧ AB = 98 ∧
    C = 0 ∧ D = 1 ∧
    F = B - D ∧ E = A - C ∧
    A * B + (C + E) * (D + F) = 144

theorem subtraction_problem_solution :
  subtraction_problem :=
begin
  sorry
end

end subtraction_problem_solution_l386_386031


namespace probability_odd_sum_of_seven_balls_drawn_l386_386573

theorem probability_odd_sum_of_seven_balls_drawn:
  let balls := finset.range 15 in
  let num_ways := (balls.card).choose 7 in
  let num_odd_ways :=
    ∑ k in {1, 3, 5}.to_finset, 
      ((finset.filter (λ x, (x + 1) % 2 = 1) balls).card.choose k) *
      ((finset.filter (λ x, (x + 1) % 2 = 0) balls).card.choose (7 - k)) in
  (num_odd_ways : ℚ) / (num_ways : ℚ) = 1064 / 2145 :=
by
  sorry

end probability_odd_sum_of_seven_balls_drawn_l386_386573


namespace probability_of_one_head_in_two_tosses_l386_386138

-- We define a fair coin toss and the probability calculation scenario
def fair_coin_toss (n : ℕ) (k : ℕ) : ℚ := (nat.choose n k : ℚ) / (2 ^ n)

-- The problem statement
theorem probability_of_one_head_in_two_tosses : fair_coin_toss 2 1 = 1 / 2 :=
sorry

end probability_of_one_head_in_two_tosses_l386_386138


namespace num_coprime_to_15_l386_386645

theorem num_coprime_to_15 : (filter (fun a => Nat.gcd a 15 = 1) (List.range 15)).length = 8 := by
  sorry

end num_coprime_to_15_l386_386645


namespace natural_values_sum_l386_386561

theorem natural_values_sum :
  let n_valid (n : ℕ) := (∀ k : ℕ, (n = 2 + 9 * k) → 
                                  6 < Math.log 2 ((n : ℝ)) < 7)
  ∧ (cos (2 * π / 9 : ℝ) + cos (4 * π / 9 : ℝ) + ... + cos ((2 * n) * π / 9 : ℝ) = 
      cos (π / 9 : ℝ))
  let solutions := Set.filter n_valid (Set.range (λ m, 2 + 9 * m)) in
  Set.sum solutions = 644 :=
sorry

end natural_values_sum_l386_386561


namespace required_jogging_speed_l386_386039

-- Definitions based on the conditions
def blocks_to_miles (blocks : ℕ) : ℚ := blocks * (1 / 8 : ℚ)
def time_in_hours (minutes : ℕ) : ℚ := minutes / 60

-- Constants provided by the problem
def beach_distance_in_blocks : ℕ := 16
def ice_cream_melt_time_in_minutes : ℕ := 10

-- The main statement to prove
theorem required_jogging_speed :
  let distance := blocks_to_miles beach_distance_in_blocks
  let time := time_in_hours ice_cream_melt_time_in_minutes
  (distance / time) = 12 := by
  sorry

end required_jogging_speed_l386_386039


namespace distance_from_apex_l386_386505

theorem distance_from_apex (A1 A2 : ℝ) (A1 = 324 * Real.sqrt 3) (A2 = 729 * Real.sqrt 3) (d : ℝ) (d = 12) :
  ∃ h : ℝ, h = 36 :=
by
  sorry

end distance_from_apex_l386_386505


namespace other_number_is_286_l386_386948

theorem other_number_is_286 (a b hcf lcm : ℕ) (h_hcf : hcf = 26) (h_lcm : lcm = 2310) (h_one_num : a = 210) 
  (rel : lcm * hcf = a * b) : b = 286 :=
by
  sorry

end other_number_is_286_l386_386948


namespace bus_time_reach_pune_l386_386966

noncomputable def initial_time_in_minutes (D : ℝ) (V₁ V₂ T₂ : ℝ) : ℝ :=
  let T₁ := D / V₁
  in T₁ * 60

theorem bus_time_reach_pune :
  ∀ (D : ℝ), (V₁ V₂ T₂ : ℝ),
  V₁ = 60 → V₂ = 65 → T₂ = 1 → D = V₂ * T₂ →
  initial_time_in_minutes D V₁ V₂ T₂ = 65 := 
by
  sorry

end bus_time_reach_pune_l386_386966


namespace find_digit_B_l386_386932

theorem find_digit_B (A B : ℕ) (h1 : 100 * A + 78 - (210 + B) = 364) : B = 4 :=
by sorry

end find_digit_B_l386_386932


namespace num_coprime_to_15_l386_386637

theorem num_coprime_to_15 :
  {a : ℕ | a < 15 ∧ Nat.gcd 15 a = 1}.toFinset.card = 8 :=
by
  sorry

end num_coprime_to_15_l386_386637


namespace min_gumballs_to_four_same_color_l386_386578

section gumball

-- Define the number of gumballs in each color
def red_gumballs : ℕ := 10
def white_gumballs : ℕ := 8
def blue_gumballs : ℕ := 9
def green_gumballs : ℕ := 6

-- Theorem statement: Minimum gumballs needed to ensure four of the same color
theorem min_gumballs_to_four_same_color : 13 = 
sorry

end gumball

end min_gumballs_to_four_same_color_l386_386578


namespace rate_of_dividend_l386_386177

-- Define the given conditions as constants
constant investment : ℝ := 4940
constant share_quote : ℝ := 9.50
constant face_value : ℝ := 10
constant annual_income : ℝ := 728

-- Define the theorem to prove the rate of dividend
theorem rate_of_dividend : 
  let n_shares := investment / share_quote,
      dividend_per_share := annual_income / n_shares,
      rate_of_dividend := (dividend_per_share / face_value) * 100
  in rate_of_dividend = 14 :=
by
  -- Skip the proof with sorry
  sorry

end rate_of_dividend_l386_386177


namespace average_visitors_per_day_l386_386973

theorem average_visitors_per_day
    (sundays_avg : ℕ)
    (other_avg : ℕ)
    (days_in_month : ℕ)
    (start_day_sunday : Bool) :
  sundays_avg = 510 →
  other_avg = 240 →
  days_in_month = 30 →
  start_day_sunday = true →
  (5 * sundays_avg + 25 * other_avg) / days_in_month = 285 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end average_visitors_per_day_l386_386973


namespace cost_of_7_cubic_yards_of_topsoil_is_1512_l386_386134

-- Definition of the given conditions
def cost_per_cubic_foot : ℕ := 8
def cubic_yards : ℕ := 7
def cubic_yards_to_cubic_feet : ℕ := 27

-- Problem definition
def cost_of_topsoil (cubic_yards : ℕ) (cost_per_cubic_foot : ℕ) (cubic_yards_to_cubic_feet : ℕ) : ℕ :=
  cubic_yards * cubic_yards_to_cubic_feet * cost_per_cubic_foot

-- The proof statement
theorem cost_of_7_cubic_yards_of_topsoil_is_1512 :
  cost_of_topsoil cubic_yards cost_per_cubic_foot cubic_yards_to_cubic_feet = 1512 := by
  sorry

end cost_of_7_cubic_yards_of_topsoil_is_1512_l386_386134


namespace time_walking_each_day_l386_386825

variable (days : Finset ℕ) (d1 : ℕ) (d2 : ℕ) (W : ℕ)

def time_spent_parking (days : Finset ℕ) : ℕ :=
  5 * days.card

def time_spent_metal_detector : ℕ :=
  2 * 30 + 3 * 10

def total_timespent (d1 d2 W : ℕ) : ℕ :=
  d1 + d2 + W

theorem time_walking_each_day (total_minutes : ℕ) (total_days : ℕ):
  total_timespent (time_spent_parking days) (time_spent_metal_detector) (total_minutes - time_spent_metal_detector - 5 * total_days)
  = total_minutes → W = 3 := by
  sorry

end time_walking_each_day_l386_386825


namespace binom_7_5_eq_21_l386_386610

theorem binom_7_5_eq_21 : binomial 7 5 = 21 := 
sorry

end binom_7_5_eq_21_l386_386610


namespace smaller_mold_radius_l386_386174

noncomputable def volume_of_hemisphere (r : ℝ) : ℝ := (2/3) * Real.pi * r^3

theorem smaller_mold_radius :
  let large_bowl_radius := 2
  let salvaged_fraction := 0.9
  let num_smaller_molds := 64
  let large_bowl_volume := volume_of_hemisphere large_bowl_radius
  let salvaged_chocolate_volume := salvaged_fraction * large_bowl_volume
  let smaller_mold_volume (r : ℝ) := volume_of_hemisphere r
  ∀ r : ℝ,
  num_smaller_molds * (smaller_mold_volume r) = salvaged_chocolate_volume →
  r = Real.cbrt (9/80) :=
by
sorry

end smaller_mold_radius_l386_386174


namespace simplify_f_l386_386059

-- Define the function f(x) using a sum over cyclic permutations
def f (x a b c : ℝ) : ℝ :=
  let term1 := (a^2 * (x - b) * (x - c)) / ((a - b) * (a - c))
  let term2 := (b^2 * (x - c) * (x - a)) / ((b - c) * (b - a))
  let term3 := (c^2 * (x - a) * (x - b)) / ((c - a) * (c - b))
  term1 + term2 + term3

-- The theorem stating the required simplification
theorem simplify_f (a b c : ℝ) (h_distinct: a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  (∀ x : ℝ, f x a b c = x^2) :=
by
  intros x 
  sorry

end simplify_f_l386_386059


namespace find_BD_l386_386907

-- Define the points and lengths given in the problem.
variable {A B C F A' B' C' A₁ C₁ D : Point}
variable (AA' BB' CC' AC BC DF AB CF A'B' B'C' C'A' : Line)
variable (A'F CF DF : ℝ)

-- Define that triangle ABC is acute and the points are collinear as described
axiom triangle_acute (h_acute: AcuteTriangle A B C) : True
axiom equilateral_ABC' (h_eq1: EquilateralTriangle A B C') : True
axiom equilateral_AB'C (h_eq2: EquilateralTriangle A B' C) : True
axiom equilateral_A'BC (h_eq3: EquilateralTriangle A' B C) : True
axiom collinear_BB'_CC' (h_col1: Line (BB') ∧ Line (CC') ∧ Intersection (BB') (CC') = F) : True
axiom intersection_CC'_AB (h_int1: Line (CC') ∧ Line (AB) ∧ Intersection (CC') (AB) = C₁) : True
axiom intersection_AA'_BC (h_int2: Line (AA') ∧ Line (BC) ∧ Intersection (AA') (BC) = A₁) : True
axiom intersection_A₁C₁_AC (h_int3: Line (Line_join A₁ C₁) ∧ Line (AC) ∧ Intersection (Line_join A₁ C₁) (AC) = D) : True
axiom segment_lengths (h_length1: Distance (A' F) = 23 ∧ Distance (C F) = 13 ∧ Distance (D F) = 24) : True

-- The theorem statement: given all the conditions, prove that BD = 26
theorem find_BD (h_acute: triangle_acute)
                (h_eq1: equilateral_ABC')
                (h_eq2: equilateral_AB'C)
                (h_eq3: equilateral_A'BC)
                (h_col1: collinear_BB'_CC')
                (h_int1: intersection_CC'_AB)
                (h_int2: intersection_AA'_BC)
                (h_int3: intersection_A₁C₁_AC)
                (h_length1: segment_lengths) : 
  Distance (B D) = 26 :=
  sorry

end find_BD_l386_386907


namespace inradius_sum_l386_386385

def triangle (A B C : Type) := true

def inradius (A B C : Type) := ℝ

variables {A B C K L M P : Type}

theorem inradius_sum :
  triangle A B C →
  (K ∈ A ∨ K ∈ B ∨ K ∈ C) →
  (L ∈ A ∨ L ∈ B ∨ L ∈ C) →
  (M ∈ A ∨ M ∈ B ∨ M ∈ C) →
  (P ∈ A ∨ P ∈ B ∨ P ∈ C) →
  (AK : A × K → P) →
  (BL : B × L → P) →
  (CM : C × M → P) →
  ∃ r1 r2 r3 rABC,
    inradius A L M = r1 ∧
    inradius B M K = r2 ∧
    inradius C K L = r3 ∧
    inradius A B C = rABC ∧
    ((r1 + r2 ≥ rABC) ∨ (r2 + r3 ≥ rABC) ∨ (r3 + r1 ≥ rABC)) :=
sorry

end inradius_sum_l386_386385


namespace PQRS_square_l386_386069

variables (A B C D E G H P Q R S : Type)
variables [EuclideanGeometry A B C D E G H P Q R S]
variables (triangleABC : Euclidean.Triangle A B C)

-- Conditions: Definitions of P, Q, R, and S based on the provided problem
def center_of_square_on_side (X Y : Type) (triangle : Euclidean.Triangle X Y) := sorry
def midpoint_of_segment (X Y : Type) := sorry

-- Definitions for P and Q
def P := center_of_square_on_side A B triangleABC
def Q := center_of_square_on_side B C triangleABC

-- Definitions for R and S
def R := midpoint_of_segment A C
def S := midpoint_of_segment D H

-- The main theorem
theorem PQRS_square : is_square P Q R S := sorry

end PQRS_square_l386_386069


namespace find_k_l386_386894

variable {a_n : ℕ → ℤ}    -- Define the arithmetic sequence as a function from natural numbers to integers
variable {a1 d : ℤ}        -- a1 is the first term, d is the common difference

-- Conditions
axiom seq_def : ∀ n, a_n n = a1 + (n - 1) * d
axiom sum_condition : 9 * a1 + 36 * d = 4 * a1 + 6 * d
axiom ak_a4_zero (k : ℕ): a_n 4 + a_n k = 0

-- Problem Statement to prove
theorem find_k : ∃ k : ℕ, a_n 4 + a_n k = 0 → k = 10 :=
by
  use 10
  intro h
  -- proof omitted
  sorry

end find_k_l386_386894


namespace john_mary_ages_l386_386051

theorem john_mary_ages (a b k : ℕ) (h_digits_a : a ≤ 9) (h_digits_b : b ≤ 9) (h_john_older : 10 * a + b > 10 * b + a) (h_square_diff : (10 * a + b)^2 - (10 * b + a)^2 = k^2) : 
  (a = 6 ∧ b = 5) := 
begin
  sorry
end

end john_mary_ages_l386_386051


namespace Megan_total_earnings_two_months_l386_386831

-- Define the conditions
def hours_per_day : ℕ := 8
def wage_per_hour : ℝ := 7.50
def days_per_month : ℕ := 20

-- Define the main question and correct answer
theorem Megan_total_earnings_two_months : 
  (2 * (days_per_month * (hours_per_day * wage_per_hour))) = 2400 := 
by
  -- In the problem statement, we are given conditions so we just state sorry because the focus is on the statement, not the solution steps.
  sorry

end Megan_total_earnings_two_months_l386_386831


namespace range_of_m_value_of_m_l386_386280

-- Define the conditions and the statements
theorem range_of_m (m : ℝ) :
  let C := λ x y : ℝ, x^2 + y^2 - 2*x - 4*y + m = 0
  in (∀ x y : ℝ, C x y = 0) → (m < 5) :=
by
  sorry

theorem value_of_m (m : ℝ) :
  let C := λ x y : ℝ, (x-1)^2 + (y-2)^2 = 5 - m
  in (∀ x y : ℝ, C x y = 0) ∧ (distance ⟨1, 2⟩ (λ x y, x + 2*y - 4 = 0) ∧ 
    ∃ M N : ℝ × ℝ, |M - N| = (4 * (real.sqrt 5) / 5)) → (m = 4) :=
by
  sorry

end range_of_m_value_of_m_l386_386280


namespace ounces_per_serving_l386_386378

def quart_in_ounces : ℕ := 32
def container_in_ounces := quart_in_ounces - 2
def servings_per_day : ℕ := 3
def days_last : ℕ := 20
def total_servings := days_last * servings_per_day
def total_ounces := container_in_ounces

theorem ounces_per_serving : total_ounces / total_servings = 0.5 :=
by
  sorry

end ounces_per_serving_l386_386378


namespace sum_first_100_terms_l386_386351

-- Define arithmetic sequences a_n and b_n with given conditions
variable (a : ℕ → ℕ) (b : ℕ → ℕ)
variable (h_a : a 1 = 25)
variable (h_b : b 1 = 15)
variable (h_ab100 : a 100 + b 100 = 139)

-- Define the sequence c_n as the sum of a_n and b_n
def c (n : ℕ) := a n + b n

-- Define the sum of the first 100 terms of c_n
def S (n : ℕ) := ∑ k in (Finset.range n).map Finset.natCast, c k

-- Theorem: the sum of the first 100 terms of the sequence c_n is 8950
theorem sum_first_100_terms (h : S 100 = 8950) : True := by
  sorry

end sum_first_100_terms_l386_386351


namespace part_a_part_b_l386_386142

-- Definition of "good" pair
def is_good_pair (a p : ℕ) : Prop := (a^3 + p^3) % (a^2 - p^2) = 0 ∧ a > p

-- Prime numbers less than 20
def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

-- Part (a): Possible values of a for (a, 13) is good
theorem part_a :
  (∀ a, is_good_pair a 13 → a = 14 ∨ a = 26 ∨ a = 182) :=
sorry

-- Part (b): Number of good pairs for p prime < 20
theorem part_b :
  (count (λ p, List.length (List.filter (λ a, is_good_pair a p)
    [p + 1, p + p, p * p])) primes_less_than_20 = 24) :=
sorry

end part_a_part_b_l386_386142


namespace volume_ratio_l386_386885

theorem volume_ratio (r : ℝ) (r_sphere : ℝ := r) (r_hemisphere : ℝ := 3 * r) :
  let V_sphere := (4 / 3) * Real.pi * r_sphere^3,
      V_hemisphere := (1 / 2) * (4 / 3) * Real.pi * r_hemisphere^3 in
  V_sphere / V_hemisphere = 1 / 13.5 :=
sorry

end volume_ratio_l386_386885


namespace lim_x_seq_infinity_lim_x_seq_cubed_over_n_squared_l386_386056

open Filter

noncomputable def x_seq : Nat → ℝ
| 0       := x_0
| (n + 1) := x_seq n + 1 / Real.sqrt (x_seq n)

theorem lim_x_seq_infinity (x_0 : ℝ) (h : x_0 > 0) : 
  Filter.Tendsto (fun n => x_seq x_0 n) Filter.atTop Filter.atTop :=
begin
  sorry
end

theorem lim_x_seq_cubed_over_n_squared (x_0 : ℝ) (h : x_0 > 0) : 
  Filter.Tendsto (fun n => (x_seq x_0 n) ^ 3 / (n ^ 2)) Filter.atTop (Filter.Pure (9 / 4)) :=
begin
  sorry
end

end lim_x_seq_infinity_lim_x_seq_cubed_over_n_squared_l386_386056


namespace incenter_rectangle_l386_386060

open Real

-- Setup definitions and conditions
variables {A B C D : Point}

-- Assume ABCD is a cyclic quadrilateral
axiom cyclic_ABCD : cyclic_quadrilateral A B C D

-- Define the centers of the incircles of triangles BCD, DCA, ADB, and BAC
noncomputable def I_A := incenter (triangle B C D) 
noncomputable def I_B := incenter (triangle D C A)
noncomputable def I_C := incenter (triangle A D B)
noncomputable def I_D := incenter (triangle B A C)

-- The goal to prove
theorem incenter_rectangle : is_rectangle (quadrilateral I_A I_B I_C I_D) :=
sorry

end incenter_rectangle_l386_386060


namespace B_catches_A_at_100_km_l386_386982

-- Define the speeds of A and B, and the time when B starts after A
def v_A : ℝ := 10 -- speed of A
def v_B : ℝ := 20 -- speed of B
def t_A : ℝ := 5  -- time when B starts after A

-- Define the distance where B catches up with A
def distance_catch_up : ℝ := 100

-- State the theorem to be proved
theorem B_catches_A_at_100_km : 
  let d_A := v_A * t_A in      -- Distance A has traveled by the time B starts
  let relative_speed := v_B - v_A in -- Relative speed of B with respect to A
  let t_catch_up := d_A / relative_speed in -- Time taken for B to catch up
  let d_B := v_B * t_catch_up in -- Distance traveled by B
  d_B = distance_catch_up := sorry

end B_catches_A_at_100_km_l386_386982


namespace move_digit_to_make_equation_correct_l386_386352

theorem move_digit_to_make_equation_correct :
  101 - 102 ≠ 1 → (101 - 10^2 = 1) :=
by
  sorry

end move_digit_to_make_equation_correct_l386_386352


namespace inf_common_seq_l386_386805

noncomputable def polynomial : ℝ → ℝ :=
  λ x, x^3 - 10 * x^2 + 29 * x - 25

theorem inf_common_seq :
  ∃ α β : ℝ, α ≠ β ∧ (1 < α ∧ α < 2) ∧ (2 < β ∧ β < 3) ∧
  (set.infinite {n : ℕ | ⟨n * α⟩ = ⟨n * β⟩ }) :=
by
  sorry

end inf_common_seq_l386_386805


namespace cylinder_unpainted_face_area_l386_386572

-- Definition of given conditions
def radius : ℝ := 8
def height : ℝ := 5
def angle_AB_arc : ℝ := 90
def plane_slice_points(A B O : ℝ × ℝ) : Prop := 
-- We encode the points A, B, O and the properties of the plane here
sorry

-- Condition of finding a, b, c such that a+b+c = 18
theorem cylinder_unpainted_face_area :
  ∃ (a b c : ℤ), 
  (∀ (A B O : ℝ × ℝ), 
    plane_slice_points A B O → 
      (c ≠ 0) ∧ (∀ (p : ℕ), p.prime → ¬p^2 ∣ c) ∧ 
        (a * real.pi + b * real.sqrt c = area_of_unpainted_face)) ∧
  a + b + c = 18 :=
sorry

end cylinder_unpainted_face_area_l386_386572


namespace cone_hemisphere_cosine_l386_386172

theorem cone_hemisphere_cosine (R θ H : ℝ) (π_ne_zero : π ≠ 0) :
  H = R * Real.cot θ →
  1 / 3 * π * R ^ 3 * Real.cot θ = 2 / 3 * π * R ^ 3 →
  Real.cos (2 * θ) = 3 / 5 :=
by
  intros h1 h2
  sorry

end cone_hemisphere_cosine_l386_386172


namespace solve_for_x_l386_386064

def floor (x : ℝ) : ℤ := ⌊x⌋

theorem solve_for_x:
  ∃ x : ℝ, 3 * x + 5 * (floor x) - 49 = 0 ∧ x = 19 / 3 :=
by
  sorry

end solve_for_x_l386_386064


namespace find_m_of_parallel_lines_l386_386333

theorem find_m_of_parallel_lines (m : ℝ) 
  (H1 : ∃ x y : ℝ, m * x + 2 * y + 6 = 0) 
  (H2 : ∃ x y : ℝ, x + (m - 1) * y + m^2 - 1 = 0) : 
  m = -1 := 
by
  sorry

end find_m_of_parallel_lines_l386_386333


namespace proof_problem_l386_386487

-- Defining the arithmetic sequence properties
variable (a₁ d : ℝ)
variable (S : ℕ → ℝ)

-- Conditions
def condition_1 := a₁ > 0
def condition_2 := d ≠ 0
def condition_3 := S 5 = S 9
def sum_of_arithmetic_sequence (n : ℕ) : ℝ := (n / 2) * (2 * a₁ + (n - 1) * d)

-- Statements to prove
def statement_A := ∀ n, S n = sum_of_arithmetic_sequence a₁ d n → n = 7 → S 7 = max_seq (λ m, S m)
def statement_C := S 14 = 0
def statement_D := ∀ (n : ℕ), S n > 0 → n ≤ 13

theorem proof_problem :
  condition_1 a₁ →
  condition_2 d →
  condition_3 S →
  (statement_A a₁ d S) ∧ (statement_C a₁ d S) ∧ (statement_D a₁ d S) :=
by
  sorry

end proof_problem_l386_386487


namespace arccos_cos_nine_l386_386621

theorem arccos_cos_nine : 
  ∀ (x : ℝ), x = 9 - 2 * Real.pi → Real.arccos (Real.cos 9) = x :=
by
  assume x h
  rw h
  sorry

end arccos_cos_nine_l386_386621


namespace log_expression_value_l386_386615

noncomputable def log_expression : ℝ := log 5^2 + (2 / 3) * log 8 + log 5 * log 20 + (log 2)^2

theorem log_expression_value : log_expression = 3 :=
by
  sorry

end log_expression_value_l386_386615


namespace solution_sqrt_eq_eight_l386_386247

theorem solution_sqrt_eq_eight (z : ℤ) (h : sqrt (10 + 3 * z) = 8) : z = 18 := by
  sorry

end solution_sqrt_eq_eight_l386_386247


namespace parabola_eq_of_focus_left_vertex_hyperbola_eq_sharing_foci_and_asymptotes_l386_386565

def ellipse_eq (x y : ℝ) : Prop := (x^2) / 64 + (y^2) / 16 = 1

theorem parabola_eq_of_focus_left_vertex :
  (∀ x y : ℝ, ellipse_eq x y -> (x, y) = (-8, 0)) ->
  ∃ p : ℝ, p > 0 ∧ ∀ x y : ℝ, y^2 = -2 * p * x -> p = 16 ∧ y^2 = -32 * x :=
by
  sorry

theorem hyperbola_eq_sharing_foci_and_asymptotes :
  (∀ x y : ℝ, ellipse_eq x y -> (x, y) = (4 * real.sqrt 3, 0) ∨ (x, y) = (-4 * real.sqrt 3, 0)) ->
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (a^2 + b^2 = 48) ∧ (b/a = real.sqrt 3) ∧ ∀ x y : ℝ, (x^2) / a^2 - (y^2) / b^2 = 1 -> a = 2 * real.sqrt 3 ∧ b = 6 ∧ (x^2) / 12 - (y^2) / 36 = 1 :=
by
  sorry

end parabola_eq_of_focus_left_vertex_hyperbola_eq_sharing_foci_and_asymptotes_l386_386565


namespace solution_set_inequality_l386_386126

theorem solution_set_inequality (x : ℝ) : 
  (x - 1) / (x + 2) > 0 ↔ x > 1 ∨ x < -2 :=
by {
  sorry -- proof omitted
}

end solution_set_inequality_l386_386126


namespace first_term_of_geometric_series_l386_386990

-- Define the conditions
def r : ℝ := -1 / 3
def S : ℝ := 18

-- Formulate the problem
theorem first_term_of_geometric_series (a : ℝ) (h : S = a / (1 - r)) : a = 24 :=
sorry

end first_term_of_geometric_series_l386_386990


namespace number_of_cuts_l386_386865

theorem number_of_cuts (num_sections : ℕ) (section_length total_length segment_length : ℕ) 
    (h1 : section_length = 4) (h2 : num_sections = 9) (h3 : total_length = num_sections * section_length) 
    (h4 : segment_length = 3) (total_segments : ℕ) (h5 : total_segments = total_length / segment_length)
    : (total_segments - 1) = 11 :=
by
  -- We can use the given conditions to re-establish needed parameters
  have H1 : total_length = 36 := by
    rw [h1, h2]
    exact mul_comm 9 4
  have H2 : total_segments = 12 := by
    rw [H1, h4]
    exact nat.div_self (by norm_num)
  assumption
sorry

end number_of_cuts_l386_386865


namespace part_I_probability_part_II_expectation_l386_386168

-- Definitions for conditions
def num_classes : ℕ := 8
def total_students : ℕ := 10
def select_students : Finset (Fin total_students) :=
  -- 3 students from Class 1 and 1 student from each of the 7 other classes.
  Finset.of_list [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

def select_community_group : Finset (Finset (Fin total_students)) :=
  Finset.powerset_len 3 select_students

def different_classes_event (g : Finset (Fin total_students)) : Prop :=
  g.card = 3 ∧ ∀ (i j : Fin total_students), i ∈ g → j ∈ g → i ≠ j → class_of_student i ≠ class_of_student j

def probability_of_event (E : Finset (Finset (Fin total_students))) : ℚ :=
  (E.card : ℚ) / (select_community_group.card : ℚ)

def class_of_student (s : Fin total_students) : Fin num_classes :=
  if s.val < 3 then ⟨1, sorry⟩ else ⟨(s.val - 2), sorry⟩ -- returns the class based on selection

theorem part_I_probability :
  probability_of_event (select_community_group.filter different_classes_event) = 49 / 60 :=
sorry

def prob_X_eq (k : ℕ) : ℚ :=
  if k = 0 then 7 / 24
  else if k = 1 then 21 / 40
  else if k = 2 then 7 / 40 
  else if k == 3 then 1 / 120 
  else 0

def expectation_X : ℚ := 
  0 * prob_X_eq (0) + 
  1 * prob_X_eq (1) + 
  2 * prob_X_eq (2) + 
  3 * prob_X_eq (3)

theorem part_II_expectation :
  expectation_X = 43 / 40 :=
sorry

end part_I_probability_part_II_expectation_l386_386168


namespace arithmetic_seq_middle_term_l386_386521

theorem arithmetic_seq_middle_term (a1 a3 y : ℤ) (h1 : a1 = 3^2) (h2 : a3 = 3^4)
    (h3 : y = (a1 + a3) / 2) : y = 45 :=
by
  rw [h1, h2] at h3
  simp at h3
  exact h3

end arithmetic_seq_middle_term_l386_386521


namespace sum_of_fractions_le_half_l386_386293

theorem sum_of_fractions_le_half {a b c : ℝ} (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) (h₄ : a * b * c = 1) :
  1 / (a^2 + 2 * b^2 + 3) + 1 / (b^2 + 2 * c^2 + 3) + 1 / (c^2 + 2 * a^2 + 3) ≤ 1 / 2 :=
by
  sorry

end sum_of_fractions_le_half_l386_386293


namespace polynomial_division_l386_386260

-- Define the initial polynomial
def p : Polynomial ℤ := Polynomial.C 1 * Polynomial.X ^ 5 
                      + Polynomial.C (-24) * Polynomial.X ^ 3 
                      + Polynomial.C 12 * Polynomial.X ^ 2 
                      + Polynomial.C (-1) * Polynomial.X 
                      + Polynomial.C 20

-- Define the divisor
def divisor : Polynomial ℤ := Polynomial.X - Polynomial.C 3

-- Define the expected quotient and remainder
def quotient : Polynomial ℤ := Polynomial.C 1 * Polynomial.X ^ 4 
                             + Polynomial.C 3 * Polynomial.X ^ 3 
                             + Polynomial.C (-15) * Polynomial.X ^ 2 
                             + Polynomial.C (-33) * Polynomial.X 
                             + Polynomial.C (-100)

def remainder : ℤ := -280

-- The theorem statement
theorem polynomial_division (p : Polynomial ℤ) (divisor : Polynomial ℤ) :
  Polynomial.divModByMonic p divisor = (quotient, Polynomial.C remainder) :=
by 
  sorry

end polynomial_division_l386_386260


namespace janeth_balloons_l386_386049

/-- Janeth's total remaining balloons after accounting for burst ones. -/
def total_remaining_balloons (round_bags : Nat) (round_per_bag : Nat) (burst_round : Nat)
    (long_bags : Nat) (long_per_bag : Nat) (burst_long : Nat)
    (heart_bags : Nat) (heart_per_bag : Nat) (burst_heart : Nat) : Nat :=
  let total_round := round_bags * round_per_bag - burst_round
  let total_long := long_bags * long_per_bag - burst_long
  let total_heart := heart_bags * heart_per_bag - burst_heart
  total_round + total_long + total_heart

theorem janeth_balloons :
  total_remaining_balloons 5 25 5 4 35 7 3 40 3 = 370 :=
by
  let round_bags := 5
  let round_per_bag := 25
  let burst_round := 5
  let long_bags := 4
  let long_per_bag := 35
  let burst_long := 7
  let heart_bags := 3
  let heart_per_bag := 40
  let burst_heart := 3
  show total_remaining_balloons round_bags round_per_bag burst_round long_bags long_per_bag burst_long heart_bags heart_per_bag burst_heart = 370
  sorry

end janeth_balloons_l386_386049


namespace rhombus_perimeter_l386_386451

-- Let's define the lengths of the diagonals
def d1 := 10
def d2 := 24

-- Half of the lengths of the diagonals
def half_d1 := d1 / 2
def half_d2 := d2 / 2

-- The length of one side of the rhombus, using the Pythagorean theorem
def side_length := Real.sqrt (half_d1^2 + half_d2^2)

-- The perimeter of the rhombus is 4 times the side length
def perimeter := 4 * side_length

-- Now we state the theorem to prove the perimeter is 52 inches
theorem rhombus_perimeter : perimeter = 52 := 
by
  -- Here you would normally provide the proof steps, but we insert 'sorry'
  sorry

end rhombus_perimeter_l386_386451


namespace find_a_l386_386701

-- Define ξ as a normal distribution with mean 1 and variance 4
def ξ : ℝ → ℝ := sorry  -- This will be the PDF or CDF of the normal distribution.

-- Given condition
def condition1 : Prop := sorry -- This should encapsulate the information that ξ follows N(1, 4).

noncomputable def a : ℝ := 0

theorem find_a : (ξ ∼ N(1, 4)) ∧ (P(ξ < 2) = 1 - P(ξ < a)) → a = 0 :=
by
  intros h
  sorry

end find_a_l386_386701


namespace cost_of_7_cubic_yards_of_topsoil_is_1512_l386_386135

-- Definition of the given conditions
def cost_per_cubic_foot : ℕ := 8
def cubic_yards : ℕ := 7
def cubic_yards_to_cubic_feet : ℕ := 27

-- Problem definition
def cost_of_topsoil (cubic_yards : ℕ) (cost_per_cubic_foot : ℕ) (cubic_yards_to_cubic_feet : ℕ) : ℕ :=
  cubic_yards * cubic_yards_to_cubic_feet * cost_per_cubic_foot

-- The proof statement
theorem cost_of_7_cubic_yards_of_topsoil_is_1512 :
  cost_of_topsoil cubic_yards cost_per_cubic_foot cubic_yards_to_cubic_feet = 1512 := by
  sorry

end cost_of_7_cubic_yards_of_topsoil_is_1512_l386_386135


namespace shadedQuadrilateralArea_is_13_l386_386281

noncomputable def calculateShadedQuadrilateralArea : ℝ :=
  let s1 := 2
  let s2 := 4
  let s3 := 6
  let s4 := 8
  let bases := s1 + s2
  let height_small := bases * (10 / 20)
  let height_large := 10
  let alt := s4 - s3
  let area := (1 / 2) * (height_small + height_large) * alt
  13

theorem shadedQuadrilateralArea_is_13 :
  calculateShadedQuadrilateralArea = 13 := by
sorry

end shadedQuadrilateralArea_is_13_l386_386281


namespace sqrt_inequality_l386_386850

theorem sqrt_inequality : (Real.sqrt 3 + Real.sqrt 7) < 2 * Real.sqrt 5 := 
  sorry

end sqrt_inequality_l386_386850


namespace initial_cat_dog_ratio_initial_ratio_is_15_7_l386_386341

theorem initial_cat_dog_ratio :
  ∀ (D : ℕ), D + 12 > 0 → 
  (45 : ℕ) * 11 = 15 * (D + 12) → 
  45.gcd D = 3 := 
begin
  intros D hD h,
  sorry,
end

theorem initial_ratio_is_15_7 :
  ∀ (D : ℕ), D + 12 > 0 → 
  (45 : ℕ) * 11 = 15 * (D + 12) → 
  (45:ℕ).nat_div_gcd D = 15 ∧ 
  D.nat_div_gcd 45 = 7 :=
begin
  intros D hD h,
  sorry,
end

end initial_cat_dog_ratio_initial_ratio_is_15_7_l386_386341


namespace area_of_triangle_OAP_l386_386562

-- Define the hexagon conditions and properties
structure RegularHexagon where
  center : Point
  vertices : Fin 6 → Point
  side_length : ℝ
  side_length_eq : ∀ i, dist (vertices i) (vertices ((i + 1) % 6)) = side_length
  center_dist_eq : ∀ i, dist center (vertices i) = side_length

-- Define the additional geometric conditions
structure GeometricConditions where
  hex : RegularHexagon
  A B : Point
  O_eq_center : A = hex.vertices 0
  B_eq_center : B = hex.vertices 1
  OA_perp_to_A_line : ∃ line, Set.Perpendicular line (segment hex.center A) ∧ A ∈ line
  P : Point
  OB_extended : ∃ q, OnRay hex.center B q ∧ P = Line.intersect q (Set.line_through hex.vertices 0 (Line.normal_set Point A))

-- Define the area of triangle OAP
def triangle_area : Point → Point → Point → ℝ
  | A, B, C => 0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

-- Define the theorem to be proved
theorem area_of_triangle_OAP (cond: GeometricConditions) : triangle_area cond.hex.center cond.hex.vertices 0 cond.P = 8 * sqrt 3 := 
sorry

end area_of_triangle_OAP_l386_386562


namespace positive_difference_is_correct_l386_386992

/-- Angela's compounded interest parameters -/
def angela_initial_deposit : ℝ := 9000
def angela_interest_rate : ℝ := 0.08
def years : ℕ := 25

/-- Bob's simple interest parameters -/
def bob_initial_deposit : ℝ := 11000
def bob_interest_rate : ℝ := 0.09

/-- Compound interest calculation for Angela -/
def angela_balance : ℝ := angela_initial_deposit * (1 + angela_interest_rate) ^ years

/-- Simple interest calculation for Bob -/
def bob_balance : ℝ := bob_initial_deposit * (1 + bob_interest_rate * years)

/-- Difference calculation -/
def balance_difference : ℝ := angela_balance - bob_balance

/-- The positive difference between their balances to the nearest dollar -/
theorem positive_difference_is_correct :
  abs (round balance_difference) = 25890 :=
by
  sorry

end positive_difference_is_correct_l386_386992


namespace find_y_equal_3_sqrt_71_l386_386243

theorem find_y_equal_3_sqrt_71 :
  ∀ (A B C D O : Point) (y : ℝ),
  AO = 5 ∧ OC = 12 ∧ OD = 5 ∧ OB = 6 ∧ BD = 11 ∧ ∠AOC = ∠BOD →
  y = dist A C →
  y = 3 * sqrt 71 := by
  sorry

end find_y_equal_3_sqrt_71_l386_386243


namespace rectangle_iff_cyclic_l386_386968

section inCircleOfQuadrilateral

open EuclideanGeometry

variables {A B C D A₁ B₁ C₁ D₁ E F G H : Point}
variables {incircle : Circle}

/- Conditions -/
def isInscribedInConvexQuad (incircle : Circle) (A B C D A₁ B₁ C₁ D₁ : Point) : Prop :=
  incircle.TangentialTo A B A₁ ∧
  incircle.TangentialTo B C B₁ ∧
  incircle.TangentialTo C D C₁ ∧
  incircle.TangentialTo D A D₁

def midpoint (P Q M : Point) : Prop :=
  dist P M = dist M Q ∧
  collinear P M Q

def EFGH_midpoints (A₁ B₁ C₁ D₁ E F G H : Point) : Prop :=
  midpoint A₁ B₁ E ∧
  midpoint B₁ C₁ F ∧
  midpoint C₁ D₁ G ∧
  midpoint D₁ A₁ H

def cyclic_quadrilateral (A B C D : Point) : Prop :=
  concyclic_points A B C D

def rectangle_ (E F G H : Point) : Prop :=
  right_angle (triangle_left_angle F E G) ∧
  right_angle (triangle_left_angle E F H) ∧
  right_angle (triangle_left_angle G H E) ∧
  right_angle (triangle_left_angle H G F)

theorem rectangle_iff_cyclic
  (hInscribed: isInscribedInConvexQuad incircle A B C D A₁ B₁ C₁ D₁)
  (hMidpoints: EFGH_midpoints A₁ B₁ C₁ D₁ E F G H):
  (rectangle_ E F G H) ↔ (cyclic_quadrilateral A B C D) := 
sorry

end inCircleOfQuadrilateral

end rectangle_iff_cyclic_l386_386968


namespace incorrect_statement_C_l386_386937

theorem incorrect_statement_C : 
  (∀ net_content : ℕ, net_content = 250 → true) 
  ∧ (∀ (x : ℤ), x = 41.03 → true)
  ∧ (∀ (a b : ℤ) (c : ℚ), a = 6 → b = 10 → c = 0.06 → a ≠ b * (c * 100))
  ∧ (∀ triangle_angles : ℕ, 45 + 90 = 135 → true) 
  → false := 
by 
  intro h,
  cases h with hA h1,
  cases h1 with hB h2,
  cases h2 with hC hD,
  exact hC 6 10 0.06 rfl rfl rfl

end incorrect_statement_C_l386_386937


namespace pamela_skittles_correct_l386_386087

def pamela_initial_skittles := 50
def pamela_gives_skittles_to_karen := 7
def pamela_receives_skittles_from_kevin := 3
def pamela_shares_percentage := 20

def pamela_final_skittles : Nat :=
  let after_giving := pamela_initial_skittles - pamela_gives_skittles_to_karen
  let after_receiving := after_giving + pamela_receives_skittles_from_kevin
  let share_amount := (after_receiving * pamela_shares_percentage) / 100
  let rounded_share := Nat.floor share_amount
  let final_count := after_receiving - rounded_share
  final_count

theorem pamela_skittles_correct :
  pamela_final_skittles = 37 := by
  sorry

end pamela_skittles_correct_l386_386087


namespace translate_down_three_units_l386_386782

def original_function (x : ℝ) : ℝ := 3 * x + 2

def translated_function (x : ℝ) : ℝ := 3 * x - 1

theorem translate_down_three_units :
  ∀ x : ℝ, translated_function x = original_function x - 3 :=
by
  intro x
  simp [original_function, translated_function]
  sorry

end translate_down_three_units_l386_386782


namespace average_first_21_multiples_l386_386917

theorem average_first_21_multiples (n : ℕ) (h : (n + 2 * n + 3 * n + ... + 21 * n) / 21 = 66) : n = 6 :=
by
  sorry

end average_first_21_multiples_l386_386917


namespace ratio_of_volumes_l386_386504

def radius_C : ℝ := 10
def height_C : ℝ := 20
def radius_D : ℝ := 20
def height_D : ℝ := 10

def volume_cone (r h : ℝ) : ℝ := (1 / 3) * Real.pi * r^2 * h

theorem ratio_of_volumes : volume_cone radius_C height_C / volume_cone radius_D height_D = 1 / 2 := by
  sorry

end ratio_of_volumes_l386_386504


namespace number_of_valid_n_l386_386277

def is_prime (p : ℤ) : Prop :=
  p > 1 ∧ ∀ m ∧ n, m * n = p → m = 1 ∨ n = 1

def num_prime_n_leq_2 : ℕ :=
  Nat.card {n : ℕ | n ≥ 2 ∧ is_prime (n^2 + 1)}

theorem number_of_valid_n : num_prime_n_leq_2 = 2 :=
by
  sorry

end number_of_valid_n_l386_386277


namespace sum_of_bn_l386_386712

theorem sum_of_bn (n : ℕ) (a : ℕ → ℕ) (b : ℕ → ℕ) 
  (h1 : a 1 = 1) 
  (h2 : b 1 = 2) 
  (h3 : b 2 = 5) 
  (h4 : ∀ n, a n * (b (n + 1) - b n) = a (n + 1)) 
  : (finset.range n).sum b = (3 * n^2 + n) / 2 :=
sorry

end sum_of_bn_l386_386712


namespace fewest_cookies_l386_386201

theorem fewest_cookies :
  (∀ D : ℝ, D > 0 → 
    Andy dq : ℝ, dq = 15 → n_andy = D / dq → n_andy = 20 →
    (∀ Bella dq : ℝ, dq = 10 → n_bella = D / dq) →
    (∀ Carlos dq : ℝ, dq = 10 → n_carlos = D / dq) →
    (∀ Diana dq : ℝ, dq = 7.5 → n_diana = D / dq) →
    n_andy < n_bella ∧ n_andy < n_carlos ∧ n_andy < n_diana) :=
by
  sorry

end fewest_cookies_l386_386201


namespace problem_solution_l386_386574

def mean (scores : List ℝ) : ℝ :=
  scores.sum / scores.length

def median (scores : List ℝ) : ℝ :=
  let sorted := scores.qsort (· ≤ ·)
  if sorted.length % 2 = 1 then
    sorted.nth_le (sorted.length / 2) sorry
  else
    (sorted.nth_le (sorted.length / 2) sorry + sorted.nth_le (sorted.length / 2 - 1) sorry) / 2

def mode (scores : List ℝ) : ℝ :=
  scores.groupBy id |> List.maxBy (·.length) |> λ l, l.head!

def variance (scores : List ℝ) : ℝ :=
  let μ := mean scores
  (scores.map (λ x, (x - μ) ^ 2)).sum / scores.length

theorem problem_solution :
  (let scores1 := [8, 8, 7, 8, 9] in
   let scores2 := [5, 9, 7, 10, 9] in
   mean scores2 = 8 ∧
   mode scores1 = 8 ∧
   median scores2 = 9 ∧
   variance scores1 < variance scores2 ∧
   (let new_scores2 := scores2.append [8] in mean new_scores2 = mean scores2)) :=
by
  sorry

end problem_solution_l386_386574


namespace num_coprime_to_15_l386_386635

theorem num_coprime_to_15 :
  {a : ℕ | a < 15 ∧ Nat.gcd 15 a = 1}.toFinset.card = 8 :=
by
  sorry

end num_coprime_to_15_l386_386635


namespace game_winner_Aerith_first_game_winner_Bob_first_l386_386985

-- Conditions: row of 20 squares, players take turns crossing out one square,
-- game ends when there are two squares left, Aerith wins if two remaining squares
-- are adjacent, Bob wins if they are not adjacent.

-- Definition of the game and winning conditions
inductive Player
| Aerith
| Bob

-- Function to determine the winner given the initial player
def winning_strategy (initial_player : Player) : Player :=
  match initial_player with
  | Player.Aerith => Player.Bob  -- Bob wins if Aerith goes first
  | Player.Bob    => Player.Aerith  -- Aerith wins if Bob goes first

-- Statement to prove
theorem game_winner_Aerith_first : 
  winning_strategy Player.Aerith = Player.Bob :=
by 
  sorry -- Proof is to be done

theorem game_winner_Bob_first :
  winning_strategy Player.Bob = Player.Aerith :=
by
  sorry -- Proof is to be done

end game_winner_Aerith_first_game_winner_Bob_first_l386_386985


namespace smallest_number_of_students_l386_386178

theorem smallest_number_of_students 
  (A6 A7 A8 : Nat)
  (h1 : A8 * 3 = A6 * 5)
  (h2 : A8 * 5 = A7 * 8) :
  A6 + A7 + A8 = 89 :=
sorry

end smallest_number_of_students_l386_386178


namespace gain_percent_is_40_l386_386947

-- Define the conditions
def purchase_price : ℕ := 800
def repair_costs : ℕ := 200
def selling_price : ℕ := 1400

-- Define the total cost
def total_cost : ℕ := purchase_price + repair_costs

-- Define the gain
def gain : ℕ := selling_price - total_cost

-- Define the gain percent
def gain_percent : ℕ := (gain * 100) / total_cost

theorem gain_percent_is_40 : gain_percent = 40 := by
  -- Placeholder for the proof
  sorry

end gain_percent_is_40_l386_386947


namespace speed_of_water_l386_386179

variable (v : ℝ)
variable (swimming_speed_still_water : ℝ := 10)
variable (time_against_current : ℝ := 8)
variable (distance_against_current : ℝ := 16)

theorem speed_of_water :
  distance_against_current = (swimming_speed_still_water - v) * time_against_current ↔ v = 8 := by
  sorry

end speed_of_water_l386_386179


namespace sphere_hemisphere_volume_ratio_l386_386883

theorem sphere_hemisphere_volume_ratio (r : ℝ) (π : ℝ) (hr : π ≠ 0): 
  let V_sphere := (4 / 3) * π * r^3,
      V_hemisphere := (1 / 2) * (4 / 3) * π * (3 * r)^3
  in V_sphere / V_hemisphere = 1 / 13.5 := 
by {
  let V_sphere := (4 / 3) * π * r^3,
      V_hemisphere := (1 / 2) * (4 / 3) * π * (3 * r)^3;
  have : V_hemisphere = (4 / 3) * π * (13.5 * r^3), {
    sorry
  },
  have ratio := V_sphere / V_hemisphere,
  rw this at ratio,
  simp [V_sphere, V_hemisphere],
  field_simp [hr],
  norm_num,
  rw ←mul_assoc,
  field_simp,
  norm_num,
}

end sphere_hemisphere_volume_ratio_l386_386883


namespace tank_maximum_volume_l386_386403

def volume_of_tank (x : ℝ) : ℝ := 
  let height := (120 - x) / 2
  x^2 * height

noncomputable def max_volume_of_tank : ℝ :=
  let derivative := - (3 / 2) * x^2 + 120 * x
  sorry -- the detailed proof will go here

theorem tank_maximum_volume :
  ∃ x : ℝ, 0 < x ∧ x < 120 ∧ 
  ∀ y, volume_of_tank y ≤ 128000 :=
by
  sorry -- the detailed proof will go here

end tank_maximum_volume_l386_386403


namespace rhombus_perimeter_l386_386458

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) (h_bisect : ∀ x, rhombus_diagonals_bisect x) :
  ∃ P, P = 52 := by
  sorry

end rhombus_perimeter_l386_386458


namespace sum_of_primes_eq_100_l386_386485

theorem sum_of_primes_eq_100 : 
  ∃ (S : Finset ℕ), (∀ (x : ℕ), x ∈ S → Nat.Prime x) ∧ S.sum id = 100 ∧ S.card = 9 :=
by
  sorry

end sum_of_primes_eq_100_l386_386485


namespace ms_henderson_fraction_l386_386994

variables (total_students_each_class total_students_goldfish a b c : ℕ) (ratio_johnson ratio_feldstein : ℝ)

-- Conditions
def miss_johnson_goldfish := total_students_each_class * ratio_johnson
def mr_feldstein_goldfish := total_students_each_class * ratio_feldstein
def ms_henderson_goldfish := total_students_goldfish - miss_johnson_goldfish - mr_feldstein_goldfish

-- Total students in each class and total students preferring goldfish
def total_students_each_class := 30
def total_students_goldfish := 31

-- Ratios for Miss Johnson's class and Mr. Feldstein's class
def ratio_johnson := 1 / 6
def ratio_feldstein := 2 / 3

-- Fraction of Ms. Henderson's class that preferred goldfish
def fraction_ms_henderson := ms_henderson_goldfish / total_students_each_class

-- Theorem to be proven
theorem ms_henderson_fraction : fraction_ms_henderson = 1 / 5 :=
by
  sorry

end ms_henderson_fraction_l386_386994


namespace coprime_count_15_l386_386664

theorem coprime_count_15 :
  {a : ℕ | a < 15 ∧ Nat.gcd 15 a = 1}.card = 8 :=
by
sorry

end coprime_count_15_l386_386664


namespace solve_abs_eq_l386_386433

theorem solve_abs_eq (x : ℝ) : |x - 4| = 3 - x ↔ x = 7 / 2 := by
  sorry

end solve_abs_eq_l386_386433


namespace double_sum_evaluation_l386_386677

theorem double_sum_evaluation :
  (∑ i in Finset.range 200, ∑ j in Finset.range 150, (i + 1) + 2 * (j + 1)) = 7545000 := 
sorry

end double_sum_evaluation_l386_386677


namespace prove_days_a_l386_386943

-- Given constants for the problem
noncomputable def combined_days := 4.117647058823529
noncomputable def days_b := 7
noncomputable def days_a : ℝ := 10

-- B's rate of working
noncomputable def rate_b := 1 / days_b

-- A's rate of working
noncomputable def rate_a := 1 / days_a

-- Combined rate of A and B working together
noncomputable def combined_rate := 1 / combined_days

-- The Lean theorem statement to prove the correctness of x (days_a) given the conditions
theorem prove_days_a : rate_a + rate_b = combined_rate := sorry

end prove_days_a_l386_386943


namespace sin_angle_HAC_l386_386954

-- Definition for the edge length of the cube
variable (s : ℝ)

-- Definition of cube vertices
def cube_vertices : Type := {P : ℝ × ℝ × ℝ // 
  P = (0, 0, 0) ∨ P = (s, 0, 0) ∨ P = (s, s, 0) ∨ P = (0, s, 0) ∨ 
  P = (0, 0, s) ∨ P = (s, 0, s) ∨ P = (s, s, s) ∨ P = (0, s, s)}

-- Definition of points A, H, and C
def A : cube_vertices s := ⟨(0, 0, 0), by simp⟩
def H : cube_vertices s := ⟨(0, s, s), by simp⟩
def C : cube_vertices s := ⟨(s, s, 0), by simp⟩

-- Definition of the angle HAC (we assume a function or way to compute angles between points in ℝ³ is available)
noncomputable def angle_HAC : ℝ := sorry -- Specify how to compute the angle HAC here

-- The final theorem stating the problem
theorem sin_angle_HAC : sin (angle_HAC s) = sqrt 3 / 2 := sorry

end sin_angle_HAC_l386_386954


namespace yellow_tint_percentage_new_mixture_l386_386974

def original_volume : ℝ := 40
def yellow_tint_percentage : ℝ := 0.35
def additional_yellow_tint : ℝ := 10
def new_volume : ℝ := original_volume + additional_yellow_tint
def original_yellow_tint : ℝ := yellow_tint_percentage * original_volume
def new_yellow_tint : ℝ := original_yellow_tint + additional_yellow_tint

theorem yellow_tint_percentage_new_mixture : 
  (new_yellow_tint / new_volume) * 100 = 48 := 
by
  sorry

end yellow_tint_percentage_new_mixture_l386_386974


namespace limit_sum_l386_386070

-- Definition of the function to get aₙ given n
def a_n (n : ℕ) : ℕ :=
  if 2 ≤ n then (n * (n - 1)) / 2 * 3^(n - 2) else 0

-- The limit we need to calculate
def limit_expr : ℕ → ℝ :=
  λ n, ∑ k in Icc 2 n, 3^k / (a_n k)

theorem limit_sum :
  (∃ l, tendsto limit_expr at_top (𝓝 l) ∧ l = 18) :=
by 
  sorry

end limit_sum_l386_386070


namespace mrs_hilt_current_rocks_l386_386082

-- Definitions based on conditions
def total_rocks_needed : ℕ := 125
def more_rocks_needed : ℕ := 61

-- Lean statement proving the required amount of currently held rocks
theorem mrs_hilt_current_rocks : (total_rocks_needed - more_rocks_needed) = 64 :=
by
  -- proof will be here
  sorry

end mrs_hilt_current_rocks_l386_386082


namespace car_A_overtakes_car_B_l386_386902

theorem car_A_overtakes_car_B (z : ℕ) :
  let y := (5 * z) / 4
  let x := (13 * z) / 10
  10 * y / (x - y) = 250 := 
by
  sorry

end car_A_overtakes_car_B_l386_386902


namespace count_coprimes_15_l386_386655

def count_coprimes_less_than (n : ℕ) : ℕ :=
  (Finset.range n).filter (λ a => Nat.gcd a n = 1).card

theorem count_coprimes_15 :
  count_coprimes_less_than 15 = 8 :=
by
  sorry

end count_coprimes_15_l386_386655


namespace sum_of_integers_l386_386545

theorem sum_of_integers : ∑ (i : ℤ) in Finset.range(21).map (Int.ofNat), (i - 15) = -105 :=
by
  sorry

end sum_of_integers_l386_386545


namespace triangle_inequality_on_triangulated_point_l386_386399

variable {α : Type*}
variables (A B C O : α)
variables [MetricSpace α] [NormedSpace ℝ α] [InnerProductSpace ℝ α]

theorem triangle_inequality_on_triangulated_point :
  (O ∈ ConvexHull ℝ ({A, B} : Set α)) →
  (O ≠ A ∧ O ≠ B) →
  norm (O - C) * norm (A - B) < norm (O - A) * norm (B - C) + norm (O - B) * norm (A - C) :=
by
  intro hO h_ne
  sorry

end triangle_inequality_on_triangulated_point_l386_386399


namespace pyramid_section_is_trapezoid_l386_386444

-- Let's define the conditions
variable {Point Line Plane : Type}
variable (A B C D S M K : Point)
variable (AB : Line) (CD : Line) (SC : Line) (SD : Line) (KM : Line) (ABM : Plane)

-- Given the conditions
variable (pyramid : (A ≠ B) ∧ (B ≠ C) ∧ (C ≠ D) ∧ (A ≠ D) ∧ (S ≠ A) ∧ (S ≠ B) 
∧ (S ≠ C) ∧ (S ≠ D))
variable (base_is_parallelogram : (A, B, C, D) → Parallelogram)
variable (M_is_on_SC : M ∈ SC)

-- Goal: The intersection of the pyramid with plane ABM is a trapezoid
theorem pyramid_section_is_trapezoid : Trapezoid (A, B, M, K) :=
by {
  -- The proof would go here 
  sorry
}

end pyramid_section_is_trapezoid_l386_386444


namespace crackers_count_l386_386405

theorem crackers_count (crackers_Marcus crackers_Mona crackers_Nicholas : ℕ) 
  (h1 : crackers_Marcus = 3 * crackers_Mona)
  (h2 : crackers_Nicholas = crackers_Mona + 6)
  (h3 : crackers_Marcus = 27) : crackers_Nicholas = 15 := 
by 
  sorry

end crackers_count_l386_386405


namespace smallest_possible_value_m_l386_386674

theorem smallest_possible_value_m (r y b : ℕ) (h : 16 * r = 18 * y ∧ 18 * y = 20 * b) : 
  ∃ m : ℕ, 30 * m = 16 * r ∧ 30 * m = 720 ∧ m = 24 :=
by {
  sorry
}

end smallest_possible_value_m_l386_386674


namespace range_of_f_l386_386888

noncomputable def f (x : ℝ) : ℝ := 
  (Real.sqrt (1 - Real.sin x ^ 2) / Real.cos x) + 
  (Real.sqrt (1 - Real.cos x ^ 2) / Real.sin x)

theorem range_of_f :
  Set.range (λ x, (Real.sqrt (1 - Real.sin x ^ 2) / Real.cos x + Real.sqrt (1 - Real.cos x ^ 2) / Real.sin x)) = {-2, 0, 2} :=
sorry

end range_of_f_l386_386888


namespace fleas_move_right_l386_386072

theorem fleas_move_right (n : Nat) (hn : 2 ≤ n) (λ : ℝ) (hλ : 0 < λ) : 
  (∀ M : ℝ, ∃ seq : List ℝ → List ℝ, 
    ∀ (l : List ℝ), 
    (0 < l.length) →  -- not all fleas are at the same point
    (∀ i, i < l.length → l.nthLe i sorry < l.nthLe (l.length - 1) sorry) →  -- fleas initially are all to the left of the rightmost flea which is furthest initially
    (∀ k, seq (l.take k).nth k sorry < M)
    ) ↔ (λ ≥ 1 / (n - 1)) sorry :=
begin
  sorry -- Proof omitted
end

end fleas_move_right_l386_386072


namespace journey_distance_l386_386942

theorem journey_distance :
  ∃ D : ℝ, ((D / 2) / 21 + (D / 2) / 24 = 10) ∧ D = 224 :=
by
  use 224
  sorry

end journey_distance_l386_386942


namespace volume_ratio_l386_386886

theorem volume_ratio (r : ℝ) (r_sphere : ℝ := r) (r_hemisphere : ℝ := 3 * r) :
  let V_sphere := (4 / 3) * Real.pi * r_sphere^3,
      V_hemisphere := (1 / 2) * (4 / 3) * Real.pi * r_hemisphere^3 in
  V_sphere / V_hemisphere = 1 / 13.5 :=
sorry

end volume_ratio_l386_386886


namespace square_area_l386_386839

theorem square_area (s : ℝ) (A B C D E F : Point) (h₁ : square A B C D) 
  (h₂ : E ∈ segment A D) (h₃ : F ∈ segment B C) 
  (h₄ : dist B E = 20) (h₅ : dist E F = 20) (h₆ : dist F D = 20) :
  (s = (side_length A B) ∧ s^2 = 720) := sorry

end square_area_l386_386839


namespace area_BGH_l386_386790

open Classical

variables {A B C G H : Type} [AffineSpace ℝ A] 

-- Define points A, B, and C
variables (a b c : A)

-- Define the conditions: total area, midpoints, and equal area division
variable (area_ABC : ℝ)
variable (midpoint_G : SVector A ℝ a b 2 G)
variable (midpoint_H : SVector A ℝ a c 2 H)
variable (equal_division : ∀ x y z, ∃ (bh_line : Line_segment A ℝ B H), 
  ∀ t : ℝ, 
  (area_of_triangle (Triangle.mk y x z) / 2))

-- The objective is to prove the area of triangle BGH
theorem area_BGH :
  (total_area : area_of_triangle (Triangle.mk b G H)) = 30 :=
sorry

end area_BGH_l386_386790


namespace correct_operation_l386_386547

theorem correct_operation : 
  (3 - Real.sqrt 2) ^ 2 = 11 - 6 * Real.sqrt 2 :=
sorry

end correct_operation_l386_386547


namespace middle_term_arithmetic_sequence_l386_386541

-- Definitions of the given conditions
def a := 3^2
def c := 3^4

-- Assertion that y is the middle term of the arithmetic sequence a, y, c
theorem middle_term_arithmetic_sequence : 
  let y := (a + c) / 2 in 
  y = 45 :=
by
  -- Since the final proof steps are not needed
  sorry

end middle_term_arithmetic_sequence_l386_386541


namespace arithmetic_geometric_sequence_ratio_l386_386710

theorem arithmetic_geometric_sequence_ratio (a : ℕ → ℕ) (d : ℕ)
  (h_arithmetic : ∀ n, a (n + 1) = a n + d)
  (h_positive_d : d > 0)
  (h_geometric : a 6 ^ 2 = a 2 * a 12) :
  (a 12) / (a 2) = 9 / 4 :=
sorry

end arithmetic_geometric_sequence_ratio_l386_386710


namespace problem_I_problem_II_l386_386315

theorem problem_I (n : ℕ) : 
  let B_n := (3 * n^2 - n) / 2
  in (B_n - (3 * (n - 1)^2 - (n - 1)) / 2 = 3 * n - 2) :=
by sorry

theorem problem_II (n : ℕ) :
  let b_n := 3 * n - 2
  let a_n := (b_n + (-1)^n) * 2^n
  let T_n := ∑ i in range n, a_n
  in T_n = (3 * n - 5) * 2^(n+1) + 10 - (2/3) * (1 - (-2) ^ n) :=
by sorry

end problem_I_problem_II_l386_386315


namespace value_of_a_l386_386764

theorem value_of_a
  (x y a : ℝ)
  (h1 : x + 2 * y = 2 * a - 1)
  (h2 : x - y = 6)
  (h3 : x = -y)
  : a = -1 :=
by
  sorry

end value_of_a_l386_386764


namespace smallest_n_not_divisible_by_10_l386_386262

theorem smallest_n_not_divisible_by_10 :
  ∃ n : ℕ, n > 2016 ∧ (1^n + 2^n + 3^n + 4^n) % 10 ≠ 0 ∧ n = 2020 := by
  sorry

end smallest_n_not_divisible_by_10_l386_386262


namespace verify_differential_eq_l386_386858

noncomputable def y (x : ℝ) : ℝ := (2 + 3 * x - 3 * x^2)^(1 / 3 : ℝ)
noncomputable def y_prime (x : ℝ) : ℝ := 
  1 / 3 * (2 + 3 * x - 3 * x^2)^(-2 / 3 : ℝ) * (3 - 6 * x)

theorem verify_differential_eq (x : ℝ) :
  y x * y_prime x = (1 - 2 * x) / y x :=
by
  sorry

end verify_differential_eq_l386_386858


namespace cost_price_correct_l386_386964

noncomputable
def cost_price_of_book (C : ℝ) : Prop :=
  let SP := 1.25 * C in
  let M := 67.47 / 0.88 in
  SP = M ∧ C = 61.34

theorem cost_price_correct : ∃ C : ℝ, cost_price_of_book C :=
by
  use 61.34
  unfold cost_price_of_book
  split
  sorry

end cost_price_correct_l386_386964


namespace eggs_left_in_box_l386_386897

theorem eggs_left_in_box (initial_eggs : ℕ) (taken_eggs : ℕ) (remaining_eggs : ℕ) : 
  initial_eggs = 47 → taken_eggs = 5 → remaining_eggs = initial_eggs - taken_eggs → remaining_eggs = 42 :=
by
  sorry

end eggs_left_in_box_l386_386897


namespace vectors_identity_l386_386935

variable (A B C : Type)
variable [AddGroup A] [AddGroup B] [AddGroup C]
variable (0 : A) (0 : B) (0 : C)

-- Given vectors AB, BC, and CA
variables (AB AC BC CA CB : A)

-- Define the basic properties of vectors
axiom AB_eq_neg_BA : AB = -CB

-- Hypothesize the setup
theorem vectors_identity : AB - CB + CA = 0 :=
sorry

end vectors_identity_l386_386935


namespace basketball_game_l386_386007

noncomputable def hawks_scores (a r : ℝ) : List ℝ := [a, a*r, a*r^2, a*r^3]

noncomputable def eagles_scores (b d : ℝ) : List ℝ := [b, b+d, b+2*d, b+3*d]

noncomputable def total_first_half (hawks eagles : List ℝ) : ℝ :=
  hawks.take 2.sum + eagles.take 2.sum

theorem basketball_game :
  ∃ (a r b d : ℝ),
  a < 0 ∧ r > 1 ∧ b > -a ∧ d > 0 ∧
  let S_H := hawks_scores a r in
  let S_E := eagles_scores b d in
  (S_H.sum + 2 = S_E.sum) ∧
  S_H.sum ≤ 100 ∧ S_E.sum ≤ 100 ∧
  total_first_half S_H S_E = 19 :=
begin
  sorry
end

end basketball_game_l386_386007


namespace china_GDP_in_2016_l386_386338

noncomputable def GDP_2016 (a r : ℝ) : ℝ := a * (1 + r / 100)^5

theorem china_GDP_in_2016 (a r : ℝ) :
  GDP_2016 a r = a * (1 + r / 100)^5 :=
by
  -- proof
  sorry

end china_GDP_in_2016_l386_386338


namespace exists_triangle_with_points_l386_386034

theorem exists_triangle_with_points (A B C X Y : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space X] [metric_space Y] : 
  (∃ (ABC : Triangle A B C), 
  (dist A X = dist B Y ∧ dist B Y = dist A B) ∧ 
  (dist B X = dist C Y ∧ dist C Y = dist B C) ∧ 
  (dist C X = dist A Y ∧ dist A Y = dist C A)) :=
sorry

end exists_triangle_with_points_l386_386034


namespace domain_of_function_l386_386684

theorem domain_of_function:
  { x : Real // 0 < x ∧ x ≤ Real.cbrt 3 } =
  { x : Real // ∃ y : Real, y = sqrt (1 / 3 - log 3 x) } :=
by
  sorry

end domain_of_function_l386_386684


namespace wine_cost_is_3_60_l386_386198

noncomputable def appetizer_cost : ℕ := 8
noncomputable def steak_cost : ℕ := 20
noncomputable def dessert_cost : ℕ := 6
noncomputable def total_spent : ℝ := 38
noncomputable def tip_percentage : ℝ := 0.20
noncomputable def number_of_wines : ℕ := 2

noncomputable def discounted_steak_cost : ℝ := steak_cost / 2
noncomputable def full_meal_cost : ℝ := appetizer_cost + steak_cost + dessert_cost
noncomputable def meal_cost_after_discount : ℝ := appetizer_cost + discounted_steak_cost + dessert_cost
noncomputable def full_meal_tip := tip_percentage * full_meal_cost
noncomputable def meal_cost_with_tip := meal_cost_after_discount + full_meal_tip
noncomputable def total_wine_cost := total_spent - meal_cost_with_tip
noncomputable def cost_per_wine := total_wine_cost / number_of_wines

theorem wine_cost_is_3_60 : cost_per_wine = 3.60 := by
  sorry

end wine_cost_is_3_60_l386_386198


namespace problem_length_YR_problem_length_m_plus_n_l386_386955

theorem problem_length_YR (PQ QR RP YR: ℕ) (X Y Z O1 O2 O3: Type) 
  (arc_PY_XQ arc_PZ_YR arc_PX_YZ: Type) :
  PQ = 29 ∧ QR = 31 ∧ RP = 30  ∧
  arc_PY_XQ = arc_PZ_YR ∧ arc_PX_YZ = arc_PY_XQ ∧ 
  2 * YR = RP
  → YR = 15 :=
begin
  sorry
end

theorem problem_length_m_plus_n (m n: ℕ) :
  (15 : ℚ).num = m → (15 : ℚ).denom = n → m + n = 16 :=
begin
  sorry
end

end problem_length_YR_problem_length_m_plus_n_l386_386955


namespace part1_part2_l386_386742

open Real

def f (a x : ℝ) : ℝ := ln (a * x + 1 / 2) + 2 / (2 * x + 1)

theorem part1 (a : ℝ) (h : a > 0) :
  (∀ x > 0, deriv (λ x : ℝ, ln (a * x + 1/2) + 2 / (2 * x + 1)) x ≥ 0) → a ≥ 2 := sorry

theorem part2 :
  ∃ a : ℝ, (∀ x > 0, f a x ≥ 1) ∧ (∀ x > 0, (deriv (λ x : ℝ, ln (a * x + 1 / 2) + 2 / (2 * x + 1)) x = 0 → x = sqrt ((2 - a) / (4 * a)))) ∧ a = 1 := sorry

end part1_part2_l386_386742


namespace value_of_y_in_arithmetic_sequence_l386_386534

theorem value_of_y_in_arithmetic_sequence :
    ∃ y : ℤ, (arithmetic_sequence (3^2) y (3^4)) ∧ y = 45 := by
  -- Here we define the arithmetic sequence condition.
  def arithmetic_sequence (a b c : ℤ) : Prop := b = (a + c) / 2
  sorry

end value_of_y_in_arithmetic_sequence_l386_386534


namespace range_of_a_decreasing_l386_386472

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 * a - 1) * x + 4 * a else a / x

def is_decreasing (f : ℝ → ℝ) := ∀ x y : ℝ, x < y → f x ≥ f y

theorem range_of_a_decreasing (a : ℝ) :
  (∃ a : ℝ, (1/6) ≤ a ∧ a < (1/3)) ↔ is_decreasing (f a) :=
sorry

end range_of_a_decreasing_l386_386472


namespace sphere_to_hemisphere_ratio_l386_386879

-- Definitions of the radii of the sphere and hemisphere
def r : ℝ := sorry -- We assume r is a positive real number, but not providing a specific value here

-- Volume of the sphere
def volume_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

-- Volume of the hemisphere with radius 3r
def volume_hemisphere (r : ℝ) : ℝ := (1 / 2) * volume_sphere (3 * r)

-- Ratio of volumes
noncomputable def ratio_volumes (r : ℝ) : ℝ := volume_sphere r / volume_hemisphere r

-- Statement to prove
theorem sphere_to_hemisphere_ratio : ratio_volumes r = 1 / 13.5 :=
by
  sorry

end sphere_to_hemisphere_ratio_l386_386879


namespace unique_pair_exists_l386_386421

theorem unique_pair_exists (c d : ℝ) (hcd : 0 < c ∧ c < d ∧ d < (π / 2)) :
  (∃! c, 0 < c ∧ c < (π / 2) ∧ sin (cos c) = c) ∧
  (∃! d, 0 < d ∧ d < (π / 2) ∧ cos (sin d) = d) :=
by
  sorry

end unique_pair_exists_l386_386421


namespace dilation_transformation_l386_386219

variables (m n x y : ℝ)

def transform (m n : ℝ) (x y: ℝ) : Prop :=
  2 * (m * x) + 3 * (n * y) = 6

theorem dilation_transformation :
  (∀ x y : ℝ, x + y = 1 → transform 2 3 x y) :=
by {
  intros x y h,
  rw [transform, h],
  ring,
  exact true.intro
}

end dilation_transformation_l386_386219


namespace cubic_equation_roots_l386_386506

theorem cubic_equation_roots (a b c d : ℝ) (h_a : a ≠ 0) 
(h_root1 : a * 4^3 + b * 4^2 + c * 4 + d = 0)
(h_root2 : a * (-3)^3 + b * (-3)^2 - 3 * c + d = 0) :
 (b + c) / a = -13 :=
by sorry

end cubic_equation_roots_l386_386506


namespace find_x_l386_386720

def a : (ℝ × ℝ × ℝ) := (2, -1, 2)
def b (x : ℝ) : (ℝ × ℝ × ℝ) := (-4, 2, x)

def are_parallel (u v : ℝ × ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ u = (k * v.1, k * v.2, k * v.3)

theorem find_x (x : ℝ) (h : are_parallel a (b x)) : x = -4 :=
  sorry

end find_x_l386_386720


namespace squares_area_difference_l386_386305

theorem squares_area_difference (BC EF : ℕ) (h1 : Square ABCD) (h2 : Square EGFO)
  (h3 : Center O ABCD) (h4 : EF ∥ BC) (h5 : (BC * BC).natAbs = 36) (h6 : (EF * EF).natAbs = 49) :
  BC * BC - EF * EF = 11.5 :=
by
  sorry

end squares_area_difference_l386_386305


namespace rhombus_perimeter_l386_386452

-- Let's define the lengths of the diagonals
def d1 := 10
def d2 := 24

-- Half of the lengths of the diagonals
def half_d1 := d1 / 2
def half_d2 := d2 / 2

-- The length of one side of the rhombus, using the Pythagorean theorem
def side_length := Real.sqrt (half_d1^2 + half_d2^2)

-- The perimeter of the rhombus is 4 times the side length
def perimeter := 4 * side_length

-- Now we state the theorem to prove the perimeter is 52 inches
theorem rhombus_perimeter : perimeter = 52 := 
by
  -- Here you would normally provide the proof steps, but we insert 'sorry'
  sorry

end rhombus_perimeter_l386_386452


namespace penguin_fish_distribution_l386_386492

theorem penguin_fish_distribution
  (E A : ℕ)
  (h_ratio : E / A = 3 / 5)
  (h_total : E + A = 48)
  (fish_per_E : ℝ := 1.5)
  (fish_per_A : ℝ := 2)
  (fish_limit : ℝ := 115)
  (total_fish : ℝ := 120) :
  (E * fish_per_E) + (A * fish_per_A) ≤ fish_limit :=
by
  -- Step 1: express E in terms of A using the ratio
  have hE := (* E A 3 5).2 h_ratio,
  -- Step 2: express E and A using the total number of penguins
  have hA := (* (E + A) 48).2 h_total,
  -- Now we need to prove these quantities satisfy the conditions
  have fish_needed_Emperor := E * fish_per_E,
  have fish_needed_Adelie := A * fish_per_A,
  have total_fish_needed := fish_needed_Emperor + fish_needed_Adelie,
  exact total_fish_needed ≤ fish_limit -- Result follows directly

end penguin_fish_distribution_l386_386492


namespace all_arrive_together_second_day_l386_386618

-- Definitions and assumptions based on conditions
variable anya_speed borya_speed vasya_speed : ℝ
variable anya_start borya_start vasya_start school_distance : ℝ

-- Conditions
-- Constant speeds with different values indicating they are different
axiom anya_slower_than_borya : anya_speed < borya_speed
axiom vasya_slower_than_anya_and_borya : vasya_speed < anya_speed

-- Leaving times and arrival at school some pair together on the first day
axiom anya_left_first_day : anya_start = 0
axiom borya_left_first_after_anya : 0 < borya_start < vasya_start
axiom vasya_left_first_day : borya_start < vasya_start

-- The next day conditions
axiom vasya_left_second_day_first : vasya_start = 0
axiom borya_left_second_after_vasya : 0 < borya_start < anya_start
axiom anya_left_second_day_last : borya_start < anya_start

-- Prove that they can all arrive at school together on the second day
theorem all_arrive_together_second_day :
  ∃ (speed : ℝ), (vasya_speed ≤ speed) ∧ (speed ≤ anya_speed) :=
by {
  sorry
}

end all_arrive_together_second_day_l386_386618


namespace johns_donation_l386_386327

theorem johns_donation
    (A T D : ℝ)
    (n : ℕ)
    (hA1 : A * 1.75 = 100)
    (hA2 : A = 100 / 1.75)
    (hT : T = 10 * A)
    (hD : D = 11 * 100 - T)
    (hn : n = 10) :
    D = 3700 / 7 := 
sorry

end johns_donation_l386_386327


namespace shirt_and_tie_outfits_l386_386862

theorem shirt_and_tie_outfits (shirts ties : ℕ) (h_shirts : shirts = 6) (h_ties : ties = 5) : shirts * ties = 30 :=
by
  rw [h_shirts, h_ties]
  norm_num

end shirt_and_tie_outfits_l386_386862


namespace general_formula_expression_for_T_n_l386_386362

noncomputable def sequence (n : ℕ) : ℤ :=
if n = 0 then 8 else if n = 3 then 2 else -2 * n + 10

def T_n (n : ℕ) : ℤ :=
if n ≤ 5 then -n^2 + 9 * n else n^2 - 9 * n + 40

theorem general_formula (n : ℕ) :
  (sequence n) = -2 * n + 10 := by
sorry

theorem expression_for_T_n (n : ℕ) :
  T_n n = (|sequence 1| + |sequence 2| + |sequence 3| + ... + |sequence n|) := by
sorry

end general_formula_expression_for_T_n_l386_386362


namespace prove_height_of_remaining_solid_l386_386569

-- Define the initial conditions
def cube_dim : ℝ := 2 -- The side length of the original cube is 2 units

-- Define a function to represent the height of the remaining solid after a specific cut
def height_of_remaining_solid (corner_cut : bool) : ℝ :=
  if corner_cut then 1 else cube_dim

-- Statement to prove the height of the remaining solid, given the conditions
theorem prove_height_of_remaining_solid :
  height_of_remaining_solid true = 1 :=
sorry

end prove_height_of_remaining_solid_l386_386569


namespace hotel_cost_l386_386169

theorem hotel_cost (x y : ℕ) (h1 : 3 * x + 6 * y = 1020) (h2 : x + 5 * y = 700) :
  5 * (x + y) = 1100 :=
sorry

end hotel_cost_l386_386169


namespace centroid_vector_relation_l386_386397

-- Definitions for centroids and vector operations
variables {V : Type*} [inner_product_space ℝ V]

def centroid (A B C : V) : V := (A + B + C) / 3

-- The theorem stating the required proof
theorem centroid_vector_relation (A B C P Q R : V) :
  let M := centroid A B C,
      N := centroid P Q R in
  N - M = (P - A + Q - B + R - C) / 3 :=
begin
  -- No need to provide the proof, as required
  sorry
end

end centroid_vector_relation_l386_386397


namespace sequence_term_l386_386488

noncomputable def a_n (n : ℕ) : ℕ :=
  if n = 1 then 3 else 4 * n - 2

def S_n (n : ℕ) : ℕ :=
  2 * n^2 + 1

theorem sequence_term (n : ℕ) : a_n n = if n = 1 then S_n 1 else S_n n - S_n (n - 1) :=
by 
  sorry

end sequence_term_l386_386488


namespace equilateral_triangle_problem_l386_386778

theorem equilateral_triangle_problem :
  let ABC := equilateral_triangle 6,
  let M := midpoint BC,
  ∃ I E : Point, lies_on I AC ∧ lies_on E AB ∧ AI < AE ∧ cyclic_quadrilateral AIME ∧
    let EMI := triangle E M I,
    area EMI = 9 * sqrt 3 →
    ∃ a b c : ℕ, CI = (a + sqrt b) / c ∧ b % (prime_factor b * prime_factor b) ≠ 0 ∧ a + b + c = 8 :=
sorry

end equilateral_triangle_problem_l386_386778


namespace probability_log_a_2b_1_l386_386586

def possible_pairs : List (ℕ × ℕ) := 
  List.product [1, 2, 3, 4, 5, 6] [1, 2, 3]

def favorable_pairs (p : ℕ × ℕ) : Bool :=
  match p with
  | (a, b) => a = 2 * b

theorem probability_log_a_2b_1 :
  (List.filter favorable_pairs possible_pairs).length / possible_pairs.length.toRat = 1 / 6 := 
by
  sorry

end probability_log_a_2b_1_l386_386586


namespace prime_dates_in_2008_l386_386837

def is_prime (n : ℕ) : Prop :=
  ∀ m ∈ {2, ..., n-1}, n % m ≠ 0

def is_leap_year (year : ℕ) : Prop :=
  (year % 4 = 0 ∧ year % 100 ≠ 0) ∨ (year % 400 = 0)

def prime_dates_in_year (year : ℕ) : ℕ :=
  if is_leap_year year then
    let prime_months := {2, 3, 5, 7, 11}
    let prime_days := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31}
    let feb_days := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}
    let nov_days := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}
    10 + 11 + 11 + 11 + 10
  else 0

theorem prime_dates_in_2008 : prime_dates_in_year 2008 = 53 :=
by sorry

end prime_dates_in_2008_l386_386837


namespace probability_triangle_or_square_l386_386342

theorem probability_triangle_or_square 
  (total_figures : ℕ)
  (triangles : ℕ)
  (squares : ℕ)
  (circles : ℕ)
  (h1 : total_figures = 10)
  (h2 : triangles = 4)
  (h3 : squares = 3)
  (h4 : circles = 3)
  : (triangles + squares) / total_figures = 7 / 10 := 
by
  rw [h1, h2, h3]
  norm_num

#check probability_triangle_or_square

end probability_triangle_or_square_l386_386342


namespace rhombus_perimeter_l386_386453

-- Let's define the lengths of the diagonals
def d1 := 10
def d2 := 24

-- Half of the lengths of the diagonals
def half_d1 := d1 / 2
def half_d2 := d2 / 2

-- The length of one side of the rhombus, using the Pythagorean theorem
def side_length := Real.sqrt (half_d1^2 + half_d2^2)

-- The perimeter of the rhombus is 4 times the side length
def perimeter := 4 * side_length

-- Now we state the theorem to prove the perimeter is 52 inches
theorem rhombus_perimeter : perimeter = 52 := 
by
  -- Here you would normally provide the proof steps, but we insert 'sorry'
  sorry

end rhombus_perimeter_l386_386453


namespace cement_mixture_weight_l386_386163

theorem cement_mixture_weight 
  (W : ℝ)
  (h1 : W = (2/5) * W + (1/6) * W + (1/10) * W + (1/8) * W + 12) :
  W = 57.6 := by
  sorry

end cement_mixture_weight_l386_386163


namespace solve_for_x_l386_386334

/-- Let f(x) = 2 - 1 / (2 - x)^3.
Proof that f(x) = 1 / (2 - x)^3 implies x = 1. -/
theorem solve_for_x (x : ℝ) (h : 2 - 1 / (2 - x)^3 = 1 / (2 - x)^3) : x = 1 :=
  sorry

end solve_for_x_l386_386334


namespace problem_inequality_l386_386295

noncomputable def a := Real.logBase 3 Real.exp 1
noncomputable def b := Real.exp 1.5
noncomputable def c := Real.logBase (1 / 3) (1 / 4)

theorem problem_inequality : a < c ∧ c < b := by
  sorry

end problem_inequality_l386_386295


namespace lines_not_neighbor_excluded_on_circle_l386_386901

theorem lines_not_neighbor_excluded_on_circle :
  ∀ (n : ℕ), n = 5 → (n * (n - 1) / 2) - n = 5 :=
by
  intros n hn
  rw hn 
  sorry

end lines_not_neighbor_excluded_on_circle_l386_386901


namespace interest_rate_difference_l386_386190

-- Definitions for given conditions
def principal : ℝ := 3000
def time : ℝ := 9
def additional_interest : ℝ := 1350

-- The Lean 4 statement for the equivalence
theorem interest_rate_difference 
  (R H : ℝ) 
  (h_interest_formula_original : principal * R * time / 100 = principal * R * time / 100) 
  (h_interest_formula_higher : principal * H * time / 100 = principal * R * time / 100 + additional_interest) 
  : (H - R) = 5 :=
sorry

end interest_rate_difference_l386_386190


namespace company_food_purchase_1_l386_386165

theorem company_food_purchase_1 (x y : ℕ) (h1: x + y = 170) (h2: 15 * x + 20 * y = 3000) : 
  x = 80 ∧ y = 90 := by
  sorry

end company_food_purchase_1_l386_386165


namespace solve_for_x_l386_386328

theorem solve_for_x (x y : ℝ) (h : (x + 1) / (x - 2) = (y^2 + 3 * y - 2) / (y^2 + 3 * y - 5)) : 
  x = (y^2 + 3 * y - 1) / 7 := 
by 
  sorry

end solve_for_x_l386_386328


namespace correct_option_is_D_l386_386986

-- Define the properties of geometric shapes and their rotations
def rotates_to_cone (shape : Type) : Prop := sorry
def rotates_to_frustum (shape : Type) : Prop := sorry
def rotates_to_cylinder (shape : Type) : Prop := sorry
def rotates_to_sphere (shape : Type) : Prop := sorry

-- Define the geometric shapes in question
def triangle : Type := sorry
def right_trapezoid : Type := sorry
def parallelogram : Type := sorry
def circular_plane : Type := sorry

-- The proof problem statement
theorem correct_option_is_D :
  ¬rotates_to_cone triangle ∧
  ¬rotates_to_frustum right_trapezoid ∧
  ¬rotates_to_cylinder parallelogram ∧
  rotates_to_sphere circular_plane :=
begin
  sorry
end

end correct_option_is_D_l386_386986


namespace hiker_distance_l386_386970

theorem hiker_distance :
  let northward : ℝ := 15
  let southward : ℝ := 9
  let eastward1 : ℝ := 8
  let eastward2 : ℝ := 2
  let net_northward := northward - southward
  let net_eastward := eastward1 + eastward2
  Math.sqrt (net_northward ^ 2 + net_eastward ^ 2) = 2 * Math.sqrt 34 := by
  sorry

end hiker_distance_l386_386970


namespace distance_between_foci_correct_l386_386100

noncomputable def hyperbola_foci_distance : ℚ :=
  let x_center := -1 / 2
  let y_center := 2
  let center := (x_center, y_center)
  let slope_pos := 2
  let slope_neg := -2
  let point_on_hyperbola := (4 : ℚ, 5 : ℚ)
  let k := (1 : ℚ) / (slope_pos * point_on_hyperbola.1 - center.1)^2
  let a := 1
  let b := k * (point_on_hyperbola.1 + center.1)^2
  let c := (a^2 + b^2).sqrt
  (2*c).num_div_den

theorem distance_between_foci_correct : 
  let x_center := -1 / 2
  let y_center := 2
  let center := (x_center, y_center)
  let slope_pos := 2
  let slope_neg := -2
  let point_on_hyperbola := (4 : ℚ, 5 : ℚ)
  let k := (1 : ℚ) / (slope_pos * point_on_hyperbola.1 - center.1)^2
  let a := 1
  let b := k * (point_on_hyperbola.1 + center.1)^2
  let c := (a^2 + b^2).sqrt
  2 * c = (2*(113:ℚ).sqrt / 9) := 
by
  -- Proof skipped
  sorry

end distance_between_foci_correct_l386_386100


namespace number_of_ordered_triples_l386_386259

open Nat

/-- Prove the number of ordered triples (a, b, c) of positive integers such that 
    lcm(a, b) = 4000, lcm(b, c) = 8000, and lcm(c, a) = 8000 is 143. -/
theorem number_of_ordered_triples : 
    ∃! (n : ℕ), (n = 143) ∧ (λ n, ∃ a b c : ℕ, (0 < a ∧ 0 < b ∧ 0 < c) ∧
                                    lcm a b = 4000 ∧ 
                                    lcm b c = 8000 ∧ 
                                    lcm c a = 8000) n :=
sorry

end number_of_ordered_triples_l386_386259


namespace middle_term_arithmetic_sequence_l386_386542

-- Definitions of the given conditions
def a := 3^2
def c := 3^4

-- Assertion that y is the middle term of the arithmetic sequence a, y, c
theorem middle_term_arithmetic_sequence : 
  let y := (a + c) / 2 in 
  y = 45 :=
by
  -- Since the final proof steps are not needed
  sorry

end middle_term_arithmetic_sequence_l386_386542


namespace measure_angle_C_value_of_sin_A_value_of_sin_2A_plus_pi_over_4_l386_386370

noncomputable def C (a b c : ℝ) : ℝ :=
  Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))

#eval C (2 * Real.sqrt 2) 5 (Real.sqrt 13) -- Should evaluate to π/4

theorem measure_angle_C : C (2 * Real.sqrt 2) 5 (Real.sqrt 13) = π / 4 :=
sorry

noncomputable def sin_A (a c C : ℝ) : ℝ :=
  a * Real.sin C / c

#eval sin_A (2 * Real.sqrt 2) (Real.sqrt 13) (π / 4) -- Should evaluate to 2 * Real.sqrt 13 / 13

theorem value_of_sin_A : sin_A (2 * Real.sqrt 2) (Real.sqrt 13) (π / 4) = 2 * Real.sqrt 13 / 13 :=
sorry

noncomputable def sin_2A_plus_pi_over_4 (A : ℝ) : ℝ :=
  Real.sin (2 * A + π / 4)

theorem value_of_sin_2A_plus_pi_over_4 (A : ℝ) : sin_2A_plus_pi_over_4 (Real.arcsin (2 * Real.sqrt 13 / 13)) = 17 * Real.sqrt 2 / 26 :=
sorry

end measure_angle_C_value_of_sin_A_value_of_sin_2A_plus_pi_over_4_l386_386370


namespace lion_turn_angles_l386_386582

-- Define the radius of the circle
def radius (r : ℝ) := r = 10

-- Define the path length the lion runs in meters
def path_length (d : ℝ) := d = 30000

-- Define the final goal: The sum of all the angles of its turns is at least 2998 radians
theorem lion_turn_angles (r d : ℝ) (α : ℝ) (hr : radius r) (hd : path_length d) (hα : d ≤ 10 * α) : α ≥ 2998 := 
sorry

end lion_turn_angles_l386_386582


namespace fraction_identity_l386_386392

def f (x : ℤ) : ℤ := 3 * x + 2
def g (x : ℤ) : ℤ := 2 * x - 3

theorem fraction_identity : 
  (f (g (f 3))) / (g (f (g 3))) = 59 / 19 := by
  sorry

end fraction_identity_l386_386392


namespace profit_shares_l386_386175

noncomputable def calculate_share (investment: ℝ) (months: ℝ) (total_weighted_investment: ℝ) (total_profit: ℝ) : ℝ :=
  (investment * months / total_weighted_investment) * total_profit

theorem profit_shares :
  let A_inv := 400
  let A_months := 12
  let B_inv := 200
  let B_months := 6
  let C_inv := 300
  let C_months := 2
  let total_profit := 450
  let A_weighted := A_inv * A_months
  let B_weighted := B_inv * B_months
  let C_weighted := C_inv * C_months
  let total_weighted_investment := A_weighted + B_weighted + C_weighted
  let A_share := calculate_share A_inv A_months total_weighted_investment total_profit
  let B_share := calculate_share B_inv B_months total_weighted_investment total_profit
  let C_share := calculate_share C_inv C_months total_weighted_investment total_profit
  A_share ≈ 327.27 ∧ B_share ≈ 81.82 ∧ C_share ≈ 40.91 :=
by
  sorry

end profit_shares_l386_386175


namespace range_of_a_l386_386738

noncomputable theory
open real

def f (a : ℝ) (x : ℝ) : ℝ :=
  (2 - x) * exp(x) - a * x - a

def g (x : ℝ) : ℝ :=
  (2 - x) * exp(x)

def h (a : ℝ) (x : ℝ) : ℝ :=
  a * x + a

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x ∧ (f a x) > 0) → (a ∈ set.Ico (- (exp 3) / 4) 0) :=
sorry

end range_of_a_l386_386738


namespace buckets_required_l386_386499

-- Statement of the problem in Lean
theorem buckets_required (B : ℝ) (h₁ : B > 0) (h₂ : 10 * B > 0) :
  let new_capacity := (2 / 5) * B in
  let new_buckets := (10 * B) / new_capacity in
  new_buckets = 25 :=
by
  sorry

end buckets_required_l386_386499


namespace solve_equation_l386_386251

theorem solve_equation (z : ℤ) (h : sqrt (10 + 3 * z) = 8) : z = 18 :=
by
  sorry

end solve_equation_l386_386251


namespace monotonically_increasing_implies_decreasing_l386_386819

noncomputable def f (a x : ℝ) : ℝ := Real.log a (abs x)

theorem monotonically_increasing_implies_decreasing (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) 
  (h3 : ∀ x y : ℝ, x < y → x < 0 → y < 0 → f(a, x) ≤ f(a, y)) : f(a, a + 1) < f(a, 1) :=
by
  sorry

end monotonically_increasing_implies_decreasing_l386_386819


namespace angle_A_eq_angle_FDE_l386_386844

theorem angle_A_eq_angle_FDE {A B C D E F : Type} [euclidean_space A B C] [euclidean_space D E F] 
  (h1 : is_isosceles_triangle A B C)
  (h2 : on_side D A C)
  (h3 : on_side E A B)
  (h4 : on_side F B C)
  (h5 : equal_length D E D F)
  (h6 : segment_sum A E F C A C)
  : angle_A_eq_angle_FDE A F D E := 
  sorry


end angle_A_eq_angle_FDE_l386_386844


namespace find_n_positive_integer_l386_386691

theorem find_n_positive_integer:
  ∀ n : ℕ, n > 0 → (∃ k : ℕ, 2^n + 12^n + 2011^n = k^2) ↔ n = 1 := 
by
  sorry

end find_n_positive_integer_l386_386691


namespace rhombus_perimeter_l386_386456

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) (h_bisect : ∀ x, rhombus_diagonals_bisect x) :
  ∃ P, P = 52 := by
  sorry

end rhombus_perimeter_l386_386456


namespace tangent_chord_angle_equality_l386_386511

theorem tangent_chord_angle_equality
    (k1 k2 : Circle)
    (D E : Point)
    (A : Point)
    (B C : Point)
    (h_intersect : k1 ∩ k2 = {D, E})
    (h_A_on_k1 : A ∈ k1)
    (h_B_on_k2 : B ∈ k2) 
    (h_C_on_k2 : C ∈ k2)
    (h_AD_B : line_through A D ∩ k2 = {B})
    (h_AE_C : line_through A E ∩ k2 = {C}) :
    ∀ γ δ β : Angle, 
    tangent_angle k1 D = γ ∧ tangent_angle k1 E = δ ∧ tangent_angle k2 B = β →
    γ + δ = γ + β :=
by sorry

end tangent_chord_angle_equality_l386_386511


namespace first_term_of_geometric_series_l386_386989

-- Define the conditions
def r : ℝ := -1 / 3
def S : ℝ := 18

-- Formulate the problem
theorem first_term_of_geometric_series (a : ℝ) (h : S = a / (1 - r)) : a = 24 :=
sorry

end first_term_of_geometric_series_l386_386989


namespace jonathan_additional_money_needed_l386_386381

theorem jonathan_additional_money_needed :
  let dictionary_cost := 11
  let dinosaur_book_cost := 19
  let cookbook_cost := 7
  let atlas_cost := 15
  let picture_books_cost := 32
  let total_cost := dictionary_cost + dinosaur_book_cost + cookbook_cost + atlas_cost + picture_books_cost
  let discount := 0.05 * total_cost
  let discounted_total := total_cost - discount
  let tax := 0.08 * discounted_total
  let final_amount := discounted_total + tax
  let saved_amount := 8
  final_amount - saved_amount = 78.18 :=
by
  let dictionary_cost := 11
  let dinosaur_book_cost := 19
  let cookbook_cost := 7
  let atlas_cost := 15
  let picture_books_cost := 32
  let total_cost := dictionary_cost + dinosaur_book_cost + cookbook_cost + atlas_cost + picture_books_cost
  let discount := 0.05 * total_cost
  let discounted_total := total_cost - discount
  let tax := 0.08 * discounted_total
  let final_amount := discounted_total + tax
  let saved_amount := 8
  have h : final_amount - saved_amount = 78.18 := by sorry
  exact h

end jonathan_additional_money_needed_l386_386381


namespace infinite_series_sum_l386_386614

theorem infinite_series_sum :
  let S := ∑' (n : ℕ), (n + 1) * ((1 : ℝ) / 2023) ^ n
  in S = 2024000495 / 2022000 :=
by
  let S := ∑' (n : ℕ), (n + 1) * ((1 : ℝ) / 2023) ^ n
  sorry

end infinite_series_sum_l386_386614


namespace sum_proper_divisors_243_l386_386925

theorem sum_proper_divisors_243 : (1 + 3 + 9 + 27 + 81) = 121 := by
  sorry

end sum_proper_divisors_243_l386_386925


namespace inequality_with_equality_condition_l386_386429

variable {a b c d : ℝ}

theorem inequality_with_equality_condition (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) : 
  (a^2 + b^2 + c^2 + d^2)^2 ≥ (a + b) * (b + c) * (c + d) * (d + a) ∧ 
  ((a^2 + b^2 + c^2 + d^2)^2 = (a + b) * (b + c) * (c + d) * (d + a) ↔ a = b ∧ b = c ∧ c = d) := sorry

end inequality_with_equality_condition_l386_386429


namespace quadratic_decreasing_range_l386_386107

theorem quadratic_decreasing_range (m c : ℝ) (h : ∀ x < 1, f' x < 0) : m ≥ 2 :=
  sorry

end quadratic_decreasing_range_l386_386107


namespace shortest_path_distance_l386_386679

noncomputable def origin : (ℝ × ℝ) := (0, 0)
noncomputable def river_y : ℝ := 50
noncomputable def barn : (ℝ × ℝ) := (80, -100)

def reflect_across_river_y (point : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := point
  (x, y + 2 * river_y)

noncomputable def reflected_barn : (ℝ × ℝ) := reflect_across_river_y barn

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem shortest_path_distance : distance origin reflected_barn = 40 * real.sqrt 29 :=
by
  sorry

end shortest_path_distance_l386_386679


namespace jim_profit_percentage_l386_386795

-- Define the constants for the problem
def selling_price : ℝ := 750
def cost : ℝ := 555.56

-- Define the profit calculation
def profit : ℝ := selling_price - cost

-- Define the percentage of profit calculation
def percentage_profit : ℝ := (profit / cost) * 100

-- State the theorem to prove
theorem jim_profit_percentage : percentage_profit ≈ 34.99 :=
by
  sorry

end jim_profit_percentage_l386_386795


namespace instantaneous_velocity_at_3_l386_386103

-- Definition of the given equation of motion
def equation_of_motion (t : ℝ) : ℝ := -t + t^2

-- Definition of the velocity function as the derivative of the equation of motion
def velocity_function (t : ℝ) : ℝ := deriv equation_of_motion t

-- Define the time at which to evaluate the instantaneous velocity
def t_val : ℝ := 3

-- The target proof statement
theorem instantaneous_velocity_at_3 : velocity_function t_val = 5 := by
  sorry

end instantaneous_velocity_at_3_l386_386103


namespace screen_to_body_ratio_increases_l386_386889

theorem screen_to_body_ratio_increases
  (a b m : ℝ)
  (h1 : a > b)
  (h2 : 0 < m)
  (h3 : m < 1) :
  (b + m) / (a + m) > b / a :=
by
  sorry

end screen_to_body_ratio_increases_l386_386889


namespace no_four_distinct_numbers_l386_386231

theorem no_four_distinct_numbers (x y : ℝ) (h : x ≠ y ∧ 
    (x^(10:ℕ) + (x^(9:ℕ)) * y + (x^(8:ℕ)) * (y^(2:ℕ)) + 
    (x^(7:ℕ)) * (y^(3:ℕ)) + (x^(6:ℕ)) * (y^(4:ℕ)) + 
    (x^(5:ℕ)) * (y^(5:ℕ)) + (x^(4:ℕ)) * (y^(6:ℕ)) + 
    (x^(3:ℕ)) * (y^(7:ℕ)) + (x^(2:ℕ)) * (y^(8:ℕ)) + 
    (x^(1:ℕ)) * (y^(9:ℕ)) + (y^(10:ℕ)) = 1)) : False :=
by
  sorry

end no_four_distinct_numbers_l386_386231


namespace number_of_vertically_placed_dominoes_is_even_l386_386849

-- Define the setup and conditions.
def grid : Type := sorry       -- Define the type for grid
def even_rows_white (g : grid) : Prop := sorry   -- Assume even rows are white
def odd_rows_black (g : grid) : Prop := sorry    -- Assume odd rows are black
def place_domino_vertically (g : grid) : Prop := sorry -- Each domino can be placed vertically

-- Define the statement to be proved.
theorem number_of_vertically_placed_dominoes_is_even
  (g : grid)
  (ev_rw : even_rows_white g)
  (od_rb : odd_rows_black g)
  (dom_v : place_domino_vertically g) :
  ∃ n, (number_of_vertical_dominoes g = 2 * n) :=
sorry

end number_of_vertically_placed_dominoes_is_even_l386_386849


namespace triangle_EF_value_l386_386366

variable (D E EF : ℝ)
variable (DE : ℝ)

theorem triangle_EF_value (h₁ : cos (2 * D - E) + sin (D + E) = 2) (h₂ : DE = 6) : EF = 3 :=
sorry

end triangle_EF_value_l386_386366


namespace number_of_two_marble_groups_l386_386501

def number_of_marbles := 3 -- Identical yellow marbles

theorem number_of_two_marble_groups :
  let red := 1,
      green := 1,
      blue := 1,
      yellow := number_of_marbles in
  (∑ r1 in {red, green, blue, yellow}, ∑ r2 in {red, green, blue, yellow}, if r1 = r2 then (choose (3, 2)) else (choose (4, 2))) = 7 := 
sorry

end number_of_two_marble_groups_l386_386501


namespace cube_volume_l386_386969

theorem cube_volume (s : ℝ) (V : ℝ) (h : s^2 = 25) : V = s^3 → V = 125 :=
by 
  have s_sq_rt : s = real.sqrt 25 := by
    rw h
    norm_num
  rw s_sq_rt
  norm_num
  sorry -- This states that we understand that s^3 = 125

end cube_volume_l386_386969


namespace distance_to_x_axis_l386_386780

theorem distance_to_x_axis (x y : ℝ) (h : (x, y) = (3, -4)) : abs y = 4 := sorry

end distance_to_x_axis_l386_386780


namespace simplify_expression_l386_386394

theorem simplify_expression :
  let A := (5 - 2 * Real.sqrt 13) ^ (1/3) + (5 + 2 * Real.sqrt 13) ^ (1/3)
  in A = 1 :=
by
  sorry

end simplify_expression_l386_386394


namespace a_is_4_when_b_is_3_l386_386436

theorem a_is_4_when_b_is_3 
  (a : ℝ) (b : ℝ) (k : ℝ)
  (h1 : ∀ b, a * b^2 = k)
  (h2 : a = 9 ∧ b = 2) :
  a = 4 :=
by
  sorry

end a_is_4_when_b_is_3_l386_386436


namespace possible_to_divide_into_two_groups_l386_386009

-- Define a type for People
universe u
variable {Person : Type u}

-- Define friend and enemy relations (assume they are given as functions)
variable (friend enemy : Person → Person)

-- Define the main statement
theorem possible_to_divide_into_two_groups (h_friend : ∀ p : Person, ∃ q : Person, friend p = q)
                                           (h_enemy : ∀ p : Person, ∃ q : Person, enemy p = q) :
  ∃ (company : Person → Bool),
    ∀ p : Person, company p ≠ company (friend p) ∧ company p ≠ company (enemy p) :=
by
  sorry

end possible_to_divide_into_two_groups_l386_386009


namespace pair_A_cannot_obtain_roots_l386_386890

-- Define the original polynomial and its roots
def polynomial := (x^2 + x - 6)

theorem pair_A_cannot_obtain_roots : 
  ¬ (∃ x, x^2 - 1 = x + 7 ∧ polynomial = 0) :=
sorry

end pair_A_cannot_obtain_roots_l386_386890


namespace differential_arctg_differential_exp_add_l386_386683

-- Problem 1
theorem differential_arctg (x y : ℝ) :
  let f := λ x y : ℝ, Real.atan (y / x)
  (df := λ x y : ℝ, (-y / (x^2 + y^2)) * dx + (x / (x^2 + y^2)) * dy)
  ∀ (x y : ℝ), (df = differential_of f) :=
sorry

-- Problem 2
theorem differential_exp_add (x y : ℝ) :
  let f := λ x y : ℝ, x * y = Real.exp (x + y) + 1
  (df := λ x y : ℝ, (y + Real.exp (x + y)) * dx + (x + Real.exp (x + y)) * dy)
  ∀ (x y : ℝ), (df = differential_of f) :=
sorry

end differential_arctg_differential_exp_add_l386_386683


namespace sum_of_digits_greatest_prime_divisor_l386_386213

theorem sum_of_digits_greatest_prime_divisor (h1 : 32767 = 2^15 - 1) :
  ((257.digits 10).sum = 14) :=
by
  sorry

end sum_of_digits_greatest_prime_divisor_l386_386213


namespace number_of_prime_pairs_satisfying_equation_l386_386278

open Nat

def is_prime_lt_100 (n : ℕ) : Prop := Prime n ∧ n ≤ 100

theorem number_of_prime_pairs_satisfying_equation :
  (∃ (a b : ℕ), is_prime_lt_100 a ∧ is_prime_lt_100 b ∧ a^2 - b^2 = 25) → False :=
by
  sorry

end number_of_prime_pairs_satisfying_equation_l386_386278


namespace solve_sqrt_eq_l386_386245

theorem solve_sqrt_eq (z : ℤ) (h : sqrt (10 + 3 * z) = 8) : z = 18 := 
by {
  sorry
}

end solve_sqrt_eq_l386_386245


namespace volume_of_prism_l386_386112

variable (a h Q : ℝ)

-- Given conditions
def lateral_edge_eq_height_of_base : Prop := h = (a * Real.sqrt 5) / 2
def cross_section_area : Prop := Q = (a * Real.sqrt 5) / 2 * h

-- Define the correct volume
def expected_volume (Q : ℝ) : ℝ := Q * Real.sqrt (9 / 3)

-- The theorem to prove
theorem volume_of_prism (a h Q : ℝ) 
  (cond1 : lateral_edge_eq_height_of_base a h)
  (cond2 : cross_section_area a h Q) :
  (sqrt 3 / 4) * (2 * sqrt (Q / 5))^2 * (a * sqrt 5 / 2) = expected_volume Q :=
by sorry

end volume_of_prism_l386_386112


namespace coprime_count_15_l386_386659

theorem coprime_count_15 :
  {a : ℕ | a < 15 ∧ Nat.gcd 15 a = 1}.card = 8 :=
by
sorry

end coprime_count_15_l386_386659


namespace evaluate_expression_l386_386239

theorem evaluate_expression : (20 * 3 + 10) / (5 + 3) = 9 := by
  sorry

end evaluate_expression_l386_386239


namespace train_passes_tree_in_16_seconds_l386_386554

noncomputable def time_to_pass_tree (length_train : ℕ) (speed_train_kmh : ℕ) : ℕ :=
  let speed_train_ms := (speed_train_kmh * 1000) / 3600
  length_train / speed_train_ms

theorem train_passes_tree_in_16_seconds :
  time_to_pass_tree 280 63 = 16 :=
  by
    sorry

end train_passes_tree_in_16_seconds_l386_386554


namespace first_term_of_geometric_series_l386_386988

theorem first_term_of_geometric_series :
  ∃ (a : ℝ), let r : ℝ := -1/3, S : ℝ := 18 in S = a / (1 - r) ∧ a = 24 :=
by
  use 24
  let r := -1 / 3
  let S := 18
  have h : S = 24 / (1 - r) := by
    rw [←S, rfl]; sorry
  exact ⟨h, rfl⟩

end first_term_of_geometric_series_l386_386988


namespace fractional_division_rounded_to_three_decimal_places_l386_386094

theorem fractional_division_rounded_to_three_decimal_places :
  (Real.fracRound (11 / 13) 3) = 0.846 := by
  sorry

end fractional_division_rounded_to_three_decimal_places_l386_386094


namespace find_PT_l386_386631

noncomputable def PQRS_is_convex_quadrilateral (P Q R S T : Type) [hPQRS : Quadrilateral P Q R S] [convex : ConvexQuadrilateral P Q R S] : Prop :=
  let PQ : ℝ := 15
  let RS : ℝ := 20
  let PR : ℝ := 22
  let T : IntersectionPoint (Diagonal P R) (Diagonal Q S)
  let equal_area : Area (Triangle P T R) = Area (Triangle Q T S)
  PT = 11

theorem find_PT 
   (P Q R S T : Point)
   (convex_quad : ConvexQuadrilateral P Q R S)
   (PQ : PQLength P Q = 15)
   (RS : RSLength R S = 20)
   (PR : DiagonalLength P R = 22)
   (intersect_T : IntersectDiagonal P R Q S = T)
   (equal_area : TriangleArea P T R = TriangleArea Q T S)
   : PTLength P T = 11 := 
by
  sorry

end find_PT_l386_386631


namespace significant_digits_of_square_side_l386_386477

theorem significant_digits_of_square_side (A : ℝ) (s : ℝ) (h : A = 0.6400) (hs : s^2 = A) : 
  s = 0.8000 :=
sorry

end significant_digits_of_square_side_l386_386477


namespace greatest_mean_YZ_l386_386497

noncomputable def X_mean := 60
noncomputable def Y_mean := 70
noncomputable def XY_mean := 64
noncomputable def XZ_mean := 66

theorem greatest_mean_YZ (Xn Yn Zn : ℕ) (m : ℕ) :
  (60 * Xn + 70 * Yn) / (Xn + Yn) = 64 →
  (60 * Xn + m) / (Xn + Zn) = 66 →
  ∃ (k : ℕ), k = 69 :=
by
  intro h1 h2
  -- Sorry is used to skip the proof
  sorry

end greatest_mean_YZ_l386_386497


namespace sin_phase_shift_right_l386_386500

theorem sin_phase_shift_right {x : ℝ} :
  (sin (2*x - π / 3)) = (sin (2 * (x - π / 6))) :=
by
  -- We would provide the proof here, but it's omitted as directed.
  sorry

end sin_phase_shift_right_l386_386500


namespace inequality_of_ab_l386_386284

theorem inequality_of_ab (a b : ℝ) (h₁ : a < 0) (h₂ : -1 < b ∧ b < 0) : ab > ab^2 ∧ ab^2 > a :=
by
  sorry

end inequality_of_ab_l386_386284


namespace flower_bed_dimensions_l386_386086

variable (l w : ℕ)

theorem flower_bed_dimensions :
  (l + 3) * (w + 2) = l * w + 64 →
  (l + 2) * (w + 3) = l * w + 68 →
  l = 14 ∧ w = 10 :=
by
  intro h1 h2
  sorry

end flower_bed_dimensions_l386_386086


namespace wedge_volume_calculation_l386_386380

theorem wedge_volume_calculation :
  let r := 5 
  let h := 8 
  let V := (1 / 3) * (Real.pi * r^2 * h) 
  V = (200 * Real.pi) / 3 :=
by
  let r := 5
  let h := 8
  let V := (1 / 3) * (Real.pi * r^2 * h)
  -- Prove the equality step is omitted as per the prompt
  sorry

end wedge_volume_calculation_l386_386380


namespace positive_integers_mod_l386_386323

theorem positive_integers_mod (n : ℕ) (h : n > 0) :
  ∃! (x : ℕ), x < 10^n ∧ x^2 % 10^n = x % 10^n :=
sorry

end positive_integers_mod_l386_386323


namespace graph_symmetry_l386_386872

def g (x : ℝ) : ℝ :=
if x ≥ -4 ∧ x ≤ -1 then x + 3
else if x ≥ -1 ∧ x ≤ 1 then real.sqrt (1 - (x + 1)^2) + 1
else if x ≥ 1 ∧ x ≤ 4 then -(x - 3)
else 0 -- This is just to make the function total, although it's irrelevant for the main proof.

def g_neg (x : ℝ) : ℝ :=
g (-x)

theorem graph_symmetry :
  ∀ (x : ℝ), ((-4 ≤ x ∧ x ≤ -1 ∧ g_neg x = x - 3) ∨
               (-1 ≤ x ∧ x ≤ 1 ∧ g_neg x = real.sqrt (1 - (x + 1)^2) + 1) ∨
               (1 ≤ x ∧ x ≤ 4 ∧ g_neg x = -x + 3)) :=
by
  intro x,
  sorry

end graph_symmetry_l386_386872


namespace region_area_l386_386916

-- Let x and y be real numbers
variables (x y : ℝ)

-- Define the inequality condition
def region_condition (x y : ℝ) : Prop := abs (4 * x - 20) + abs (3 * y + 9) ≤ 6

-- The statement that needs to be proved
theorem region_area : (∃ x y : ℝ, region_condition x y) → ∃ A : ℝ, A = 6 :=
by
  sorry

end region_area_l386_386916


namespace find_m_l386_386282

variables {V : Type*} [AddCommGroup V] [Module ℝ V] 
variables (a b : V)
variables (m : ℝ) (λ : ℝ)
variables (A B C : V)

-- Conditions
def non_collinear (a b : V) : Prop := ¬(∃ k : ℝ, a = k • b)
def collinear (a b : V) : Prop := ∃ k : ℝ, a = k • b

-- Given conditions
axiom h1 : non_collinear a b
axiom h2 : B - A = m • a + 2 • b
axiom h3 : C - B = 3 • a + m • b
axiom h4 : collinear (B - A) (C - B)

-- The statement we need to prove
theorem find_m : m = real.sqrt 6 ∨ m = -real.sqrt 6 := sorry

end find_m_l386_386282


namespace is_even_function_symmetric_about_line_symmetric_about_point_l386_386744

noncomputable def f (x : ℝ) : ℝ :=
  sqrt 3 * Real.sin (2 * x + Real.pi / 3) + 2 * Real.cos (x + Real.pi / 6) ^ 2

theorem is_even_function (x : ℝ) : f x = f (-x) :=
  sorry

theorem symmetric_about_line (x : ℝ) : f (-x - Real.pi / 2) = f (x - Real.pi / 2) :=
  sorry

theorem symmetric_about_point (x : ℝ) : f (x + Real.pi / 4) - 1 = 1 - f (x - Real.pi / 4) :=
  sorry

end is_even_function_symmetric_about_line_symmetric_about_point_l386_386744


namespace sequence_proof_l386_386709

noncomputable def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) :=
  ∀ n, a (n + 1) = a n + d

def S (a : ℕ → ℤ) (n : ℕ) :=
  (n + 1) * a 0 + n * (n + 1) / 2 * (a 1 - a 0)

def T (b : ℕ → ℤ) (n : ℕ) :=
  ∑ i in finset.range n, b i

theorem sequence_proof
  (a b : ℕ → ℤ)
  (d : ℤ)
  (h_arith : arithmetic_sequence a d)
  (h_d_neq_0 : d ≠ 0)
  (h_sum : S a 2 + S a 4 = 50)
  (h_geom : (a 0) * (a 12) = (a 3)^2)
  (h_bn_an : ∀ n, b n = a n * 3^(n - 1))
  (h_ineq : ∀ n : ℕ, λ * T b n - S a n + 2 * n^2 ≤ 0) :
  (∀ n, a n = 2 * n + 1) ∧
  (∀ n, T b n = n * 3^n) ∧
  λ = -1 / 27 :=
sorry

end sequence_proof_l386_386709


namespace equivalent_expression_l386_386145

theorem equivalent_expression (a b : ℕ) (h1: a = 2015) (h2: b = 2016) :
  (a^4 - 3 * a^3 * b + 3 * a^2 * b^2 - b^4 + a) / (a * b) = -2014^2 / 2015^3 :=
by
  sorry

end equivalent_expression_l386_386145


namespace total_cost_calculation_l386_386359

variable (a b : ℕ)

def price_tomatoes := 20 * a
def price_cabbage := 30 * b
def total_cost := price_tomatoes + price_cabbage

theorem total_cost_calculation : total_cost a b = 20 * a + 30 * b := by
  sorry

end total_cost_calculation_l386_386359


namespace locus_of_points_eq_boundary_and_interior_l386_386918

-- Define the sides of the rectangle
variables (a b c : ℝ)

-- Define the rectangle as a structure
structure Rectangle :=
  (A B C D : ℝ) -- We use ℝ for simplicity; in practice, these should be points

noncomputable def sum_of_distances_to_sides (A B C D : ℝ) : ℝ := sorry  
-- This is a placeholder; it should calculate the sum of distances to sides AB, BC, CD, DA

-- Define the extension points A1, B1, C1, D1 based on the problem conditions
noncomputable def extension_points (A B C D : ℝ) (c : ℝ) :
    ℝ := sorry -- This should define A1, B1, C1, D1 positions

-- Define the locus condition as a property
def locus_condition (P : ℝ) (A B C D : ℝ) (c : ℝ) : Prop :=
  sum_of_distances_to_sides A B C D = a + b + c

-- Theorem stating the proof problem
theorem locus_of_points_eq_boundary_and_interior (A B C D : ℝ) :
  ∀ (P : ℝ), locus_condition P A B C D c ↔
  (P lies on boundary or interior of polygon formed by A1 B1 B2 C2 C1 D1 D2 A2) :=
sorry

end locus_of_points_eq_boundary_and_interior_l386_386918


namespace picture_area_l386_386181

theorem picture_area (paper_width paper_length margin : ℝ) 
  (hw : paper_width = 8.5) 
  (hl : paper_length = 10) 
  (hm : margin = 1.5) : 
  (paper_length - 2 * margin) * (paper_width - 2 * margin) = 38.5 := by 
  have w := paper_width - 2 * margin,
  have l := paper_length - 2 * margin,
  sorry

end picture_area_l386_386181


namespace teams_in_each_group_l386_386349

theorem teams_in_each_group (n : ℕ) :
  (2 * (n * (n - 1) / 2) + 3 * n = 56) → n = 7 :=
by
  sorry

end teams_in_each_group_l386_386349


namespace melissa_annual_driving_hours_l386_386077

noncomputable theory -- Use noncomputable for handling non-computable definitions

-- Define the conditions
def hours_per_trip : ℕ := 3
def trips_per_month : ℕ := 2
def months_per_year : ℕ := 12

-- Define the monthly driving hours
def monthly_driving_hours := hours_per_trip * trips_per_month

-- Prove the number of hours Melissa spends driving in a year
theorem melissa_annual_driving_hours : monthly_driving_hours * months_per_year = 72 := sorry

end melissa_annual_driving_hours_l386_386077


namespace part1_part2_part3_l386_386704

-- Define the sequence recursively
def sequence (a : ℕ → ℝ) (a1 : ℝ) (h1 : a1 > 0) (h2 : a1 ≠ 1) :=
  ∀ n, a (n + 1) = (2 * a n) / (1 + a n)

-- Define the first proof goal
theorem part1 (a : ℕ → ℝ) (a1 : ℝ) (h1 : a1 > 0) (h2 : a1 ≠ 1) (seq : sequence a a1 h1 h2) :
  ∀ n, a (n + 1) ≠ a n := sorry

-- Define the second proof goal with specific initial condition
theorem part2 : ∀ n, ((2 : ℝ)^(n-1))/((2 : ℝ)^(n-1) + 1) = 
  let a : ℕ → ℝ := λ n, (2^(n-1))/(2^(n-1) + 1)
  a n := sorry

-- Define the third proof goal showing a specific constant p and common ratio q
theorem part3 (a : ℕ → ℝ) (a1 : ℝ) (h1 : a1 > 0) (h2 : a1 ≠ 1) (seq : sequence a a1 h1 h2) :
  ∃ (p q : ℝ), p ≠ 0 ∧ (q = 1/2) ∧ (∀ n, (a (n+1) + p) / a (n+1) = q * (a n + p) / a n) :=
  ⟨-1, 1/2, by simp [sequence, seq]; sorry⟩

end part1_part2_part3_l386_386704


namespace greatest_consecutive_integer_sum_l386_386143

theorem greatest_consecutive_integer_sum (a N : ℤ) :
  (∑ k in Finset.range N, a + k) = 136 → N * (2 * a + N - 1) = 272 → N = 272 :=
by
  intros h_sum h_eq
  sorry

end greatest_consecutive_integer_sum_l386_386143


namespace melissa_driving_hours_in_a_year_l386_386080

/-- Melissa drives to town twice each month, and each trip takes 3 hours.
    Prove that she spends 72 hours driving in a year. -/
theorem melissa_driving_hours_in_a_year 
  (trips_per_month : ℕ)
  (months_per_year : ℕ)
  (hours_per_trip : ℕ)
  (total_hours : ℕ)
  (h1 : trips_per_month = 2)
  (h2 : months_per_year = 12)
  (h3 : hours_per_trip = 3) :
  total_hours = trips_per_month * months_per_year * hours_per_trip :=
by {
  rw [h1, h2, h3],
  trivial,
}

end melissa_driving_hours_in_a_year_l386_386080


namespace solve_quadratic_equation_1_solve_quadratic_equation_2_l386_386860

theorem solve_quadratic_equation_1 (x : ℝ) :
  3 * x^2 + 2 * x - 1 = 0 ↔ x = 1/3 ∨ x = -1 :=
by sorry

theorem solve_quadratic_equation_2 (x : ℝ) :
  (x + 2) * (x - 3) = 5 * x - 15 ↔ x = 3 :=
by sorry

end solve_quadratic_equation_1_solve_quadratic_equation_2_l386_386860


namespace find_a_l386_386389

variable (a : ℝ)

theorem find_a (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x ∈ set.Icc (-1 : ℝ) 1, a^(2*x) + 2*a^x - 1 ≤ 14) →
  (∃ x₀ ∈ set.Icc (-1 : ℝ) 1, a^(2*x₀) + 2*a^x₀ - 1 = 14) →
  a = 3 ∨ a = 1/3 := 
by 
  intros hmax heq
  sorry

end find_a_l386_386389


namespace count_coprime_with_15_lt_15_l386_386649

theorem count_coprime_with_15_lt_15 :
  {a : ℕ // a < 15 ∧ Nat.coprime 15 a}.to_finset.card = 8 := 
sorry

end count_coprime_with_15_lt_15_l386_386649


namespace M_equals_nat_geq_1_l386_386061

variable (M : Set ℕ)

variables (h1 : 2018 ∈ M)
  (h2 : ∀ m ∈ M, ∀ d, d ∣ m → d > 0 → d ∈ M)
  (h3 : ∀ k m ∈ M, 1 < k → k < m → (k * m + 1) ∈ M)

theorem M_equals_nat_geq_1 : M = { n | n > 0 } :=
by
  -- The proof is omitted.
  sorry

end M_equals_nat_geq_1_l386_386061


namespace largest_possible_n_l386_386713

theorem largest_possible_n (k : ℕ) (hk : k > 0) : ∃ n, n = 3 * k - 1 := 
  sorry

end largest_possible_n_l386_386713


namespace tangent_line_at_Q_l386_386469

-- Let Q be the point (π/2, 1)
def Q : ℝ × ℝ := (Real.pi / 2, 1)

-- Let f be the function sin x - 2 cos x
def f (x : ℝ) : ℝ := Real.sin x - 2 * Real.cos x

-- The derivative of f is cos x + 2 sin x
def f' (x : ℝ) : ℝ := Real.cos x + 2 * Real.sin x

-- The slope of the tangent line at x = π/2
def slope_at_Q : ℝ := f' (Real.pi / 2)

-- The equation of the tangent line using point-slope form
def tangent_line (x : ℝ) : ℝ := slope_at_Q * (x - Q.1) + Q.2

-- Prove that the equation of the tangent line to the curve y = sin x - 2 cos x 
-- at the point Q(π/2, 1) is y = 2x - π + 1
theorem tangent_line_at_Q : ∀ x : ℝ, tangent_line x = 2 * x - Real.pi + 1 := 
by
  sorry

end tangent_line_at_Q_l386_386469


namespace number_of_diagonals_of_convex_nonagon_l386_386612

def is_convex_nonagon (sides : ℕ) : Prop := sides = 9

theorem number_of_diagonals_of_convex_nonagon (sides : ℕ) (h : is_convex_nonagon sides) : 
  let vertices := 9
  let diagonals_from_one_vertex := vertices - 3
  let total_possible_diagonals := vertices * diagonals_from_one_vertex
  let distinct_diagonals := total_possible_diagonals / 2
  in distinct_diagonals = 27 := 
by 
  sorry

end number_of_diagonals_of_convex_nonagon_l386_386612


namespace solution_sqrt_eq_eight_l386_386249

theorem solution_sqrt_eq_eight (z : ℤ) (h : sqrt (10 + 3 * z) = 8) : z = 18 := by
  sorry

end solution_sqrt_eq_eight_l386_386249


namespace smallest_quotient_l386_386274

theorem smallest_quotient :
  let s := {-5, -4, 4, 6} in
  ∃ x y ∈ s, x ≠ y ∧ (∀ a b ∈ s, a ≠ b → (a / b) ≥ (x / y)) ∧ x / y = -3 / 2 :=
sorry

end smallest_quotient_l386_386274


namespace hyperbola_eccentricity_proof_l386_386113

-- Definitions of the conditions
variables (a b x y : ℝ) (h₁ : a > 0) (h₂ : b > 0)
def hyperbola (x y : ℝ) := x^2 / a^2 - y^2 / b^2 = 1
def line (y : ℝ) := y = 2 * b
def is_isosceles_right_triangle (A B : ℝ × ℝ) (O : ℝ × ℝ) :=
  let BAO := A.1 * O.2 + A.2 * O.1 = B.1 * O.2 + B.2 * O.1 in
  let ABO := B.1 * O.2 + B.2 * O.1 = A.1 * O.2 + A.2 * O.1 in
  BAO ∧ ABO

-- Euler's eccentricity formula
def eccentricity (a b : ℝ) := sqrt (1 + b^2 / a^2)

-- Statements of the problem
theorem hyperbola_eccentricity_proof (h : hyperbola a b = 1 ∧ line y ∧ is_isosceles_right_triangle (a, b) (a, -b) (0, 0)) :
  eccentricity a b = 3 / 2 :=
sorry

end hyperbola_eccentricity_proof_l386_386113


namespace range_of_a_l386_386716

noncomputable def setM (a : ℝ) : Set ℝ := {x | x * (x - a - 1) < 0}
def setN : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

theorem range_of_a (a : ℝ) : setM a ∪ setN = setN ↔ (-2 ≤ a ∧ a ≤ 2) :=
sorry

end range_of_a_l386_386716


namespace factorize_poly_l386_386938

open Polynomial

theorem factorize_poly : 
  (X ^ 15 + X ^ 7 + 1 : Polynomial ℤ) =
    (X^2 + X + 1) * (X^13 - X^12 + X^10 - X^9 + X^7 - X^6 + X^4 - X^3 + X - 1) := 
  by
  sorry

end factorize_poly_l386_386938


namespace distinct_integers_prime_distinct_integers_composite_l386_386091

open Nat

theorem distinct_integers_prime (n : ℕ) (h : Prime (2 * n - 1)) :
  ∀ (a : Fin n → ℕ), ∃ i j : Fin n, i ≠ j ∧
  (a i + a j) / gcd (a i) (a j) ≥ 2 * n - 1 := sorry

theorem distinct_integers_composite (n : ℕ) (h : ¬ Prime (2 * n - 1)) :
  ∃ (a : Fin n → ℕ), ∀ i j : Fin n, i ≠ j →
  (a i + a j) / gcd (a i) (a j) < 2 * n - 1 := sorry

end distinct_integers_prime_distinct_integers_composite_l386_386091


namespace cost_of_7_cubic_yards_l386_386136

def cost_per_cubic_foot : ℕ := 8
def cubic_feet_per_cubic_yard : ℕ := 27
def cubic_yards : ℕ := 7

theorem cost_of_7_cubic_yards
  (c : ℕ) (c_cubic : c = cost_per_cubic_foot)
  (f : ℕ) (f_cubic : f = cubic_feet_per_cubic_yard)
  (y : ℕ) (y_cubic : y = cubic_yards) :
  c * f * y = 1512 :=
begin
  sorry
end

end cost_of_7_cubic_yards_l386_386136


namespace parabola_ellipse_focus_l386_386729

theorem parabola_ellipse_focus (p : ℝ) :
  (∃ (x y : ℝ), x^2 = 2 * p * y ∧ y = -1 ∧ x = 0) →
  p = -2 :=
by
  sorry

end parabola_ellipse_focus_l386_386729


namespace required_jogging_speed_l386_386040

-- Definitions based on the conditions
def blocks_to_miles (blocks : ℕ) : ℚ := blocks * (1 / 8 : ℚ)
def time_in_hours (minutes : ℕ) : ℚ := minutes / 60

-- Constants provided by the problem
def beach_distance_in_blocks : ℕ := 16
def ice_cream_melt_time_in_minutes : ℕ := 10

-- The main statement to prove
theorem required_jogging_speed :
  let distance := blocks_to_miles beach_distance_in_blocks
  let time := time_in_hours ice_cream_melt_time_in_minutes
  (distance / time) = 12 := by
  sorry

end required_jogging_speed_l386_386040


namespace determinant_zero_l386_386237

noncomputable def matrix_A (θ φ : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 2 * Real.sin θ, -Real.cos θ],
    ![-2 * Real.sin θ, 0, Real.sin φ],
    ![Real.cos θ, -Real.sin φ, 0]]

theorem determinant_zero (θ φ : ℝ) : Matrix.det (matrix_A θ φ) = 0 := by
  sorry

end determinant_zero_l386_386237


namespace compound_formed_with_NaOH_is_H2_l386_386253

/-- Given 2 moles of NaH react with H2O to form 2 moles of NaOH
    and that the required amount of H2O is 36 grams.
    Prove that the compound formed along with NaOH is hydrogen gas (H2). -/
theorem compound_formed_with_NaOH_is_H2 (NaH H2O NaOH H2 : Type)
  (reaction : ∀ (n m n' m' : nat), n = m → n' = m' → n + n' = m + m') :
  (reaction 2 36 2 2) → H2 = hydrogen_gas := 
by 
  sorry

/-- Definition of hydrogen_gas as H2 -/
def hydrogen_gas : Type := sorry

end compound_formed_with_NaOH_is_H2_l386_386253


namespace solve_pounds_l386_386220

def price_per_pound_corn : ℝ := 1.20
def price_per_pound_beans : ℝ := 0.60
def price_per_pound_rice : ℝ := 0.80
def total_weight : ℕ := 30
def total_cost : ℝ := 24.00
def equal_beans_rice (b r : ℕ) : Prop := b = r

theorem solve_pounds (c b r : ℕ) (h1 : price_per_pound_corn * ↑c + price_per_pound_beans * ↑b + price_per_pound_rice * ↑r = total_cost)
    (h2 : c + b + r = total_weight) (h3 : equal_beans_rice b r) : c = 6 ∧ b = 12 ∧ r = 12 := by
  sorry

end solve_pounds_l386_386220


namespace rhombus_perimeter_l386_386454

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) (h_bisect : ∀ x, rhombus_diagonals_bisect x) :
  ∃ P, P = 52 := by
  sorry

end rhombus_perimeter_l386_386454


namespace dave_trips_l386_386952

/-- Dave can only carry 9 trays at a time. -/
def trays_per_trip := 9

/-- Number of trays Dave has to pick up from one table. -/
def trays_from_table1 := 17

/-- Number of trays Dave has to pick up from another table. -/
def trays_from_table2 := 55

/-- Total number of trays Dave has to pick up. -/
def total_trays := trays_from_table1 + trays_from_table2

/-- The number of trips Dave will make. -/
def number_of_trips := total_trays / trays_per_trip

theorem dave_trips :
  number_of_trips = 8 :=
sorry

end dave_trips_l386_386952


namespace length_of_second_train_l386_386141

theorem length_of_second_train (speed1 speed2 : ℝ) (length1 time : ℝ) (h1 : speed1 = 60) (h2 : speed2 = 40) 
  (h3 : length1 = 450) (h4 : time = 26.99784017278618) :
  let speed1_mps := speed1 * 1000 / 3600
  let speed2_mps := speed2 * 1000 / 3600
  let relative_speed := speed1_mps + speed2_mps
  let total_distance := relative_speed * time
  let length2 := total_distance - length1
  length2 = 300 :=
by
  sorry

end length_of_second_train_l386_386141


namespace area_CM_KD_l386_386418

-- Definitions based on conditions
variables {A B C D M K : Type}
variables [Parallelogram ABCD A B C D] -- that ABCD is a parallelogram
variables (BM_ratio : ℚ) (MC_ratio : ℚ)
variables (area_parallelogram : ℚ)

-- Concrete values given in conditions
def BM_to_MC_ratio := 3 / 2
def area_ABCD := 1 -- area of the parallelogram ABCD

-- Condition: Point M divides BC in the given ratio (3:2)
def BM : ℚ := BM_ratio / (BM_ratio + MC_ratio) * (area_parallelogram)
def MC : ℚ := MC_ratio / (BM_ratio + MC_ratio) * (area_parallelogram)

-- Definition of point K
def K (A B D M : Type) := ∃ (K : Set Point), B ≠ D ∧ ∃ AM_intersect_BD (A M B D: Type)

-- The goal: Prove that the area of quadrilateral CMKD equals 31/80
theorem area_CM_KD (K : Set Point) : 
  BM_ratio = 3 → MC_ratio = 2 → area_parallelogram = 1 →
  area_CM_KD (A B C D M K : Type) = 31 / 80 :=
sorry

end area_CM_KD_l386_386418


namespace jason_initial_pears_l386_386794

-- Define the initial number of pears Jason picked.
variable (P : ℕ)

-- Conditions translated to Lean:
-- Jason gave Keith 47 pears and received 12 from Mike, leaving him with 11 pears.
variable (h1 : P - 47 + 12 = 11)

-- The theorem stating the problem:
theorem jason_initial_pears : P = 46 :=
by
  sorry

end jason_initial_pears_l386_386794


namespace log_base_4_of_64sqrt2_l386_386235

theorem log_base_4_of_64sqrt2 : log 4 (64 * sqrt 2) = 13 / 4 := by 
  sorry

end log_base_4_of_64sqrt2_l386_386235


namespace set_interval_representation_l386_386936

open Set

theorem set_interval_representation :
  {x : ℝ | (0 ≤ x ∧ x < 5) ∨ x > 10} = Ico 0 5 ∪ Ioi 10 :=
by
  sorry

end set_interval_representation_l386_386936


namespace max_chord_construction_is_perpendicular_l386_386427

-- Definitions of the elements in the problem
variables (A : Type) [EuclideanSpace A] [InversionTransform P]
variable (P : A)
variable (B C : A)
variable (max_chord_construction : (P - B).norm + (P - C).norm = max_over_chords)
variable (QR_perpendicular : ∀ (Q R : A), isChord(A, P, QR) → is_perpendicular(P, Q, R))

-- The theorem statement
theorem max_chord_construction_is_perpendicular :
  ( ∃ B C, isChord(A, P, B, C) ∧ isMaxSumInverses(P, B, C) ) →
  ( QR_perpendicular Q R ) :=
begin
  sorry
end

end max_chord_construction_is_perpendicular_l386_386427


namespace solution_sqrt_eq_eight_l386_386248

theorem solution_sqrt_eq_eight (z : ℤ) (h : sqrt (10 + 3 * z) = 8) : z = 18 := by
  sorry

end solution_sqrt_eq_eight_l386_386248


namespace max_area_of_rectangular_pen_l386_386703

theorem max_area_of_rectangular_pen :
  ∃ w l : ℝ, 6 * w = 60 ∧ l = 2 * w ∧ w * l = 200 :=
begin
  sorry
end

end max_area_of_rectangular_pen_l386_386703


namespace hurricane_damage_in_cad_l386_386971

def damage_usd : ℝ := 45000000
def conversion_rate : ℝ := 3 / 2

theorem hurricane_damage_in_cad : (conversion_rate * damage_usd) = 67500000 := 
by sorry

end hurricane_damage_in_cad_l386_386971


namespace infinite_composite_sequence_l386_386804

open Nat

def sequence (a b : ℕ) : ℕ → ℕ
| 0 => 1
| (n + 1) => a * sequence n + b

theorem infinite_composite_sequence (a b : ℕ) : ∃ᶠ n in at_top, ¬ is_prime (sequence a b n) := by
  sorry

end infinite_composite_sequence_l386_386804


namespace inequality_solution_set_l386_386124

theorem inequality_solution_set (x : ℝ) : 
  ( (x - 1) / (x + 2) > 0 ) ↔ ( x > 1 ∨ x < -2 ) :=
by sorry

end inequality_solution_set_l386_386124


namespace intersection_point_of_lines_l386_386256

theorem intersection_point_of_lines :
  ∃ x y : ℚ, 
    (y = -3 * x + 4) ∧ 
    (y = (1 / 3) * x + 1) ∧ 
    x = 9 / 10 ∧ 
    y = 13 / 10 :=
by sorry

end intersection_point_of_lines_l386_386256


namespace count_coprime_with_15_lt_15_l386_386651

theorem count_coprime_with_15_lt_15 :
  {a : ℕ // a < 15 ∧ Nat.coprime 15 a}.to_finset.card = 8 := 
sorry

end count_coprime_with_15_lt_15_l386_386651


namespace circumcenter_locus_of_projection_l386_386589

variables {A B C P A' B' C' : Type*}
variables [ordered_semiring A] [ordered_semiring B] [ordered_semiring C]
          [ordered_semiring P] [ordered_semiring A'] [ordered_semiring B'] [ordered_semiring C']

-- Definitions of the conditions
def is_acute (α β γ : ℝ) : Prop := α + β + γ = π ∧ α < π / 2 ∧ β < π / 2 ∧ γ < π / 2

def is_projection (P A' B' C' : ℝ) (A B C : ℝ) : Prop :=
  A' = orthogonal_projection P B C ∧
  B' = orthogonal_projection P C A ∧
  C' = orthogonal_projection P A B

def angles_equal (α β : ℝ) : Prop := α = β

-- Main theorem
theorem circumcenter_locus_of_projection
  {α β γ : ℝ} (acute_triangle : is_acute α β γ)
  (proj : is_projection P A' B' C' A B C)
  (angles_eq_1 : angles_equal α (angle B' A' C'))
  (angles_eq_2 : angles_equal β (angle C' B' A')) :
  is_circumcenter P A B C :=
sorry

end circumcenter_locus_of_projection_l386_386589


namespace solve_equation_l386_386252

theorem solve_equation (z : ℤ) (h : sqrt (10 + 3 * z) = 8) : z = 18 :=
by
  sorry

end solve_equation_l386_386252


namespace vertical_distance_to_Felix_l386_386184

/--
  Dora is at point (8, -15).
  Eli is at point (2, 18).
  Felix is at point (5, 7).
  Calculate the vertical distance they need to walk to reach Felix.
-/
theorem vertical_distance_to_Felix :
  let Dora := (8, -15)
  let Eli := (2, 18)
  let Felix := (5, 7)
  let midpoint := ((Dora.1 + Eli.1) / 2, (Dora.2 + Eli.2) / 2)
  let vertical_distance := Felix.2 - midpoint.2
  vertical_distance = 5.5 :=
by
  sorry

end vertical_distance_to_Felix_l386_386184


namespace birds_not_hawks_warbler_kingfisher_l386_386945

variables (B : ℝ)
variables (hawks paddyfield_warblers kingfishers : ℝ)

-- Conditions
def condition1 := hawks = 0.30 * B
def condition2 := paddyfield_warblers = 0.40 * (B - hawks)
def condition3 := kingfishers = 0.25 * paddyfield_warblers

-- Question: Prove the percentage of birds that are not hawks, paddyfield-warblers, or kingfishers is 35%
theorem birds_not_hawks_warbler_kingfisher (B hawks paddyfield_warblers kingfishers : ℝ) 
 (h1 : hawks = 0.30 * B) 
 (h2 : paddyfield_warblers = 0.40 * (B - hawks)) 
 (h3 : kingfishers = 0.25 * paddyfield_warblers) : 
 (1 - (hawks + paddyfield_warblers + kingfishers) / B) * 100 = 35 :=
by
  sorry

end birds_not_hawks_warbler_kingfisher_l386_386945


namespace num_coprime_to_15_l386_386642

theorem num_coprime_to_15 : (filter (fun a => Nat.gcd a 15 = 1) (List.range 15)).length = 8 := by
  sorry

end num_coprime_to_15_l386_386642


namespace minimum_value_of_f_l386_386741

def f (x φ : ℝ) := cos (2 * x - φ) - √3 * sin (2 * x - φ)

theorem minimum_value_of_f :
  ∀ (x : ℝ) (φ : ℝ), (|φ| < π / 2) → (x ∈ set.Icc (-π / 2) 0) → (f x φ ≥ -√3) :=
by
  intros
  -- Proof step to be filled
  sorry

end minimum_value_of_f_l386_386741


namespace sequence_hits_zero_before_1500th_term_l386_386068

theorem sequence_hits_zero_before_1500th_term :
  ∀ (x0 x1 : ℕ), x0 < 1000 → x1 < 1000 → x0 > 0 → x1 > 0 → 
  ∃ (i : ℕ), i < 1500 ∧ ∃ (a : ℕ → ℕ), (a 0 = x0 ∧ a 1 = x1 ∧ ∀ n, a (n + 2) = abs (a (n+1) - a n)) ∧ a i = 0 :=
by
  sorry

end sequence_hits_zero_before_1500th_term_l386_386068


namespace rhombus_perimeter_l386_386449

-- Let's define the lengths of the diagonals
def d1 := 10
def d2 := 24

-- Half of the lengths of the diagonals
def half_d1 := d1 / 2
def half_d2 := d2 / 2

-- The length of one side of the rhombus, using the Pythagorean theorem
def side_length := Real.sqrt (half_d1^2 + half_d2^2)

-- The perimeter of the rhombus is 4 times the side length
def perimeter := 4 * side_length

-- Now we state the theorem to prove the perimeter is 52 inches
theorem rhombus_perimeter : perimeter = 52 := 
by
  -- Here you would normally provide the proof steps, but we insert 'sorry'
  sorry

end rhombus_perimeter_l386_386449


namespace area_of_four_triangles_l386_386869

theorem area_of_four_triangles (a b : ℕ) (h1 : 2 * b = 28) (h2 : a + 2 * b = 30) :
    4 * (1 / 2 * a * b) = 56 := by
  sorry

end area_of_four_triangles_l386_386869


namespace trig_identity_proof_l386_386159

theorem trig_identity_proof : 
  (Real.cos (70 * Real.pi / 180) * Real.sin (80 * Real.pi / 180) + 
   Real.cos (20 * Real.pi / 180) * Real.sin (10 * Real.pi / 180) = 1 / 2) :=
by
  sorry

end trig_identity_proof_l386_386159


namespace proof_of_longest_side_l386_386768

noncomputable def longest_side_of_triangle
  (A B C : Real) -- angles in the triangle
  (a b c : Real) -- sides of the triangle
  (area : Real) : Prop :=
  (sin A + sin B) / (sin A + sin C) = 4 / 5 ∧
  (sin A + sin B) / (sin B + sin C) = 4 / 6 ∧
  (sin A + sin C) / (sin B + sin C) = 5 / 6 ∧
  0 < A ∧ A < π ∧
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧
  15 * sqrt 3 = area ∧
  1 / 2 * a * b * sin C = area ∧
  max a (max b c) = c ∧
  c = 14

theorem proof_of_longest_side
  (A B C : Real)
  (a b c : Real)
  (area : Real)
  (h1 : sin A + sin B / sin A + sin C = 4 / 5)
  (h2 : sin A + sin B / sin B + sin C = 4 / 6)
  (h3 : sin A + sin C / sin B + sin C = 5 / 6)
  (hA1 : 0 < A) (hA2 : A < π)
  (hB1 : 0 < B) (hB2 : B < π)
  (hC1 : 0 < C) (hC2 : C < π)
  (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0)
  (harea : 15 * sqrt 3 = area)
  (hareaFormula : 1 / 2 * a * b * sin C = area)
  (hmax : max a (max b c) = c)
  : c = 14 := sorry

end proof_of_longest_side_l386_386768


namespace trajectory_of_M_constant_NA_dot_NB_l386_386182

theorem trajectory_of_M (x y : ℝ) (hx : x^2 + 2*y^2 = 4) : 
  (x^2 / 4 + y^2 / 2 = 1) := 
sorry

theorem constant_NA_dot_NB (x1 x2 y1 y2 n : ℝ) 
  (hx1x2_sum : x1 + x2 = -4 * (y1^2 / (2 * y1^2 + 1)))
  (hx1x2_prod : x1 * x2 = (2 * y1^2 - 4) / (2 * y1^2 + 1))
  (hN : n = -7/4) :
  (1 + y1^2) * (x1 * x2) + (y1^2 - n) * (x1 + x2) + y1^2 + n^2 = -15/16 :=
sorry

end trajectory_of_M_constant_NA_dot_NB_l386_386182


namespace arithmetic_seq_middle_term_l386_386523

theorem arithmetic_seq_middle_term (a1 a3 y : ℤ) (h1 : a1 = 3^2) (h2 : a3 = 3^4)
    (h3 : y = (a1 + a3) / 2) : y = 45 :=
by
  rw [h1, h2] at h3
  simp at h3
  exact h3

end arithmetic_seq_middle_term_l386_386523


namespace ammonia_moles_formed_l386_386686

theorem ammonia_moles_formed :
  ∀ (Li₃N_eq H₂O_eq : ℕ), 
  Li₃N_eq = 3 → H₂O_eq = 9 → 
  (∃ NH₃_eq : ℕ, NH₃_eq = 3) :=
by
  intros Li₃N_eq H₂O_eq HLi₃N_eq HH₂O_eq
  use 3
  sorry

end ammonia_moles_formed_l386_386686


namespace mask_production_l386_386339

theorem mask_production (x : ℝ) :
  24 + 24 * (1 + x) + 24 * (1 + x)^2 = 88 :=
sorry

end mask_production_l386_386339


namespace triangle_property_proof_l386_386373

noncomputable def triangleABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  a = 2 * Real.sqrt 2 ∧
  b = 5 ∧
  c = Real.sqrt 13 ∧
  C = Real.pi / 4 ∧
  ∃ sinA : ℝ, sinA = 2 * Real.sqrt 13 / 13 ∧
  ∃ sin_2A_plus_pi_4 : ℝ, sin_2A_plus_pi_4 = 17 * Real.sqrt 2 / 26

theorem triangle_property_proof :
  ∃ (A B C : ℝ), 
  triangleABC (2 * Real.sqrt 2) 5 (Real.sqrt 13) A B C
:= sorry

end triangle_property_proof_l386_386373


namespace fraction_sum_simplified_l386_386115

theorem fraction_sum_simplified (a b : ℕ) (h1 : 0.6125 = (a : ℝ) / b) (h2 : Nat.gcd a b = 1) : a + b = 129 :=
sorry

end fraction_sum_simplified_l386_386115


namespace problem_statement_l386_386669

theorem problem_statement :
  (∃ n : ℕ, n = 8 ∧ ∀ a : ℕ, a < 15 → (∃ x : ℤ, a * x ≡ 1 [MOD 15]) ↔ gcd a 15 = 1) :=
by
  use 8
  intro a
  intro ha
  split
  sorry

end problem_statement_l386_386669


namespace power_increase_fourfold_l386_386597

theorem power_increase_fourfold 
    (F v : ℝ)
    (k : ℝ)
    (R : ℝ := k * v)
    (P_initial : ℝ := F * v)
    (v' : ℝ := 2 * v)
    (F' : ℝ := 2 * F)
    (R' : ℝ := k * v')
    (P_final : ℝ := F' * v') :
    P_final = 4 * P_initial := 
by
  sorry

end power_increase_fourfold_l386_386597


namespace length_AK_independent_of_D_l386_386552

theorem length_AK_independent_of_D (A B C D K : Point) (h1 : D ∈ line_segment B C) 
  (incircle_ABD : Incircle (Triangle A B D)) 
  (incircle_ACD : Incircle (Triangle A C D))
  (tangent_bc : IsTangent incircle_ABD incircle_ACD K AD)
  (common_tangent : CommonExternalTangent incircle_ABD incircle_ACD K AD) :
  AK = (AB + AC - BC) / 2 :=
sorry

end length_AK_independent_of_D_l386_386552


namespace minimum_surface_area_of_sphere_l386_386758

theorem minimum_surface_area_of_sphere (a b c : ℝ) (V : volume)
  (h1 : a * b * c = 4) (h2 : a * b = 1)
  (h3 : ∃ r, ∀ v ∈ vertices, dist v (0, 0, 0) = r) :
  ∃ r, 4 * π * r^2 = 18 * π :=
by
  sorry

end minimum_surface_area_of_sphere_l386_386758


namespace range_of_p_l386_386748

theorem range_of_p (p : ℝ) (a_n b_n : ℕ → ℝ)
  (ha : ∀ n, a_n n = -n + p)
  (hb : ∀ n, b_n n = 3^(n-4))
  (C_n : ℕ → ℝ)
  (hC : ∀ n, C_n n = if a_n n ≥ b_n n then a_n n else b_n n)
  (hc : ∀ n : ℕ, n ≥ 1 → C_n n > C_n 4) :
  4 < p ∧ p < 7 :=
sorry

end range_of_p_l386_386748


namespace simsons_theorem_l386_386161

variables (A B C D E F P : Type)
variables (O : circle)

-- Definitions for points and their properties
def triangle_inscribed (ABC : triangle A B C) (O : circle) : Prop :=
  inscribed_in ABC O

def on_circumcircle (P : Type) (O : circle) : Prop :=
  on_circle P O

def perpendicular (P1 P2 : Type) (line : line P1 P2) (P : Type) : Prop :=
  perpendicular_to P1 P2 P

def collinear (D E F : Type) : Prop :=
  collinear_points D E F

-- Statement for Simson's Theorem
theorem simsons_theorem (ABC : triangle A B C) (O : circle)
    (h1 : triangle_inscribed ABC O)
    (P : Type) (h2 : on_circumcircle P O)
    (D : Type) (h3 : perpendicular P B C D)
    (E : Type) (h4 : perpendicular P C A E)
    (F : Type) (h5 : perpendicular P A B F) :
  collinear D E F := sorry

end simsons_theorem_l386_386161


namespace complex_power_eq_neg_two_l386_386868

-- Define the complex number z as (1 + i)^(2 * i)
def z : ℂ := (1 + complex.I)^(2 * complex.I)

-- Lean statement that asserts z equals -2
theorem complex_power_eq_neg_two : z = -2 := by
  sorry

end complex_power_eq_neg_two_l386_386868


namespace power_increase_fourfold_l386_386596

theorem power_increase_fourfold 
    (F v : ℝ)
    (k : ℝ)
    (R : ℝ := k * v)
    (P_initial : ℝ := F * v)
    (v' : ℝ := 2 * v)
    (F' : ℝ := 2 * F)
    (R' : ℝ := k * v')
    (P_final : ℝ := F' * v') :
    P_final = 4 * P_initial := 
by
  sorry

end power_increase_fourfold_l386_386596


namespace maximum_area_triangle_l386_386816

noncomputable def parabola (x : ℝ) : ℝ := x^2 - 4 * x + 3

def point_A : (ℝ × ℝ) := (2, 0)
def point_B : (ℝ × ℝ) := (5, 2)

def point_C (p : ℝ) (h : 2 ≤ p ∧ p ≤ 5) : (ℝ × ℝ) := (p, parabola p)

def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  1 / 2 * abs (A.1 * B.2 + B.1 * C.2 + C.1 * A.2 - A.2 * B.1 - B.2 * C.1 - C.2 * A.1)

theorem maximum_area_triangle :
  ∃ (p : ℝ) (h : 2 ≤ p ∧ p ≤ 5),
    area_triangle point_A point_B (point_C p h) = 9 :=
sorry

end maximum_area_triangle_l386_386816


namespace smallest_n_not_divisible_by_10_l386_386263

theorem smallest_n_not_divisible_by_10 :
  ∃ n : ℕ, n > 2016 ∧ (1^n + 2^n + 3^n + 4^n) % 10 ≠ 0 ∧ n = 2020 := by
  sorry

end smallest_n_not_divisible_by_10_l386_386263


namespace annual_income_correct_l386_386682

-- Define the conditions as constants
def investment_amount : ℝ := 6800
def dividend_rate : ℝ := 0.50
def price_per_share : ℝ := 136

-- Define the annual income calculation
def annual_income : ℝ :=
  let number_of_shares := investment_amount / price_per_share
  let dividend_per_share := price_per_share * dividend_rate
  number_of_shares * dividend_per_share

-- Create the theorem that proves the annual income equals the expected value
theorem annual_income_correct : annual_income = 3400 := by
  sorry

end annual_income_correct_l386_386682


namespace problem_l386_386721

variable (a : ℕ → ℝ) -- {a_n} is a sequence
variable (S : ℕ → ℝ) -- S_n represents the sum of the first n terms
variable (d : ℝ) -- non-zero common difference
variable (a1 : ℝ) -- first term of the sequence

-- Define an arithmetic sequence with common difference d and first term a1
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) (a1 : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- S_n is the sum of the first n terms of an arithmetic sequence
def sum_of_arithmetic_sequence (S a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = n * (a 1 + a n) / 2

theorem problem 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (d : ℝ) 
  (a1 : ℝ) 
  (h_non_zero : d ≠ 0)
  (h_sequence : is_arithmetic_sequence a d a1)
  (h_sum : sum_of_arithmetic_sequence S a)
  (h_S5_eq_S6 : S 5 = S 6) :
  S 11 = 0 := 
sorry

end problem_l386_386721


namespace functional_equation_initial_condition_unique_f3_l386_386391

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation (x y : ℝ) : f (f x + y) = f (x ^ 2 - y) + 2 * f x * y := sorry

theorem initial_condition : f 1 = 1 := sorry

theorem unique_f3 : f 3 = 9 := sorry

end functional_equation_initial_condition_unique_f3_l386_386391


namespace jack_speed_to_beach_12_mph_l386_386043

theorem jack_speed_to_beach_12_mph :
  let distance := 16 * (1 / 8) -- distance in miles
  let time := 10 / 60        -- time in hours
  distance / time = 12 :=    -- speed in miles per hour
by
  let distance := 16 * (1 / 8) -- evaluation of distance
  let time := 10 / 60          -- evaluation of time
  show distance / time = 12    -- final speed calculation
  from sorry

end jack_speed_to_beach_12_mph_l386_386043


namespace solve_system_of_equations_l386_386435

theorem solve_system_of_equations :
  ∃ (x y z : ℝ),
    (x^2 + y^2 + 8 * x - 6 * y = -20) ∧
    (x^2 + z^2 + 8 * x + 4 * z = -10) ∧
    (y^2 + z^2 - 6 * y + 4 * z = 0) ∧
    ((x = -3 ∧ y = 1 ∧ z = 1) ∨
     (x = -3 ∧ y = 1 ∧ z = -5) ∨
     (x = -3 ∧ y = 5 ∧ z = 1) ∨
     (x = -3 ∧ y = 5 ∧ z = -5) ∨
     (x = -5 ∧ y = 1 ∧ z = 1) ∨
     (x = -5 ∧ y = 1 ∧ z = -5) ∨
     (x = -5 ∧ y = 5 ∧ z = 1) ∨
     (x = -5 ∧ y = 5 ∧ z = -5)) :=
sorry

end solve_system_of_equations_l386_386435


namespace chord_bisected_by_P_has_equation_3x_minus_y_minus_11_eq_0_l386_386314

variable (x y k : ℝ)
variable (x1 y1 x2 y2 : ℝ)
variable (P : ℝ × ℝ)
variable (l : ℝ → ℝ → Prop)

def parabola (x y : ℝ) : Prop := y^2 = 6 * x
def midpoint (P A B : ℝ × ℝ) : Prop := P.1 = (A.1 + B.1) / 2 ∧ P.2 = (A.2 + B.2) / 2
def line_equation (l : ℝ → ℝ → Prop) (m c : ℝ) : Prop := ∀ x y, l x y ↔ y = m * x + c

theorem chord_bisected_by_P_has_equation_3x_minus_y_minus_11_eq_0 :
  (∀ (A B : ℝ × ℝ), midpoint (4, 1) A B → parabola A.1 A.2 → parabola B.1 B.2 → ∃ m c, line_equation l m c ∧ (m = 3 ∧ c = -11)) :=
sorry

end chord_bisected_by_P_has_equation_3x_minus_y_minus_11_eq_0_l386_386314


namespace square_area_l386_386840

theorem square_area (s : ℝ) (A B C D E F : Point) (h₁ : square A B C D) 
  (h₂ : E ∈ segment A D) (h₃ : F ∈ segment B C) 
  (h₄ : dist B E = 20) (h₅ : dist E F = 20) (h₆ : dist F D = 20) :
  (s = (side_length A B) ∧ s^2 = 720) := sorry

end square_area_l386_386840


namespace no_line_intersects_five_regions_l386_386131

-- Definitions for mutually perpendicular planes and regions
def perpendicular_planes (p1 p2 p3 : Plane) : Prop := 
  p1 ⊥ p2 ∧ p2 ⊥ p3 ∧ p1 ⊥ p3

def divides_into_8_regions (p1 p2 p3 : Plane) : Prop := 
  perpendicular_planes p1 p2 p3

def line_intersects_regions (l : Line) (regions : Set Region) : Prop := 
  ∃ p, divides_into_8_regions p.1 p.2 p.3 ∧
       Set.cardinality (set_of (λ r, l ∩ r ≠ ∅)) regions

-- The statement to be proven
theorem no_line_intersects_five_regions (p1 p2 p3 : Plane) (regions : Set Region) :
  divides_into_8_regions p1 p2 p3 → ¬ ∃ l : Line, line_intersects_regions l regions 5 := 
sorry

end no_line_intersects_five_regions_l386_386131


namespace unique_solution_count_l386_386687

theorem unique_solution_count :
  ∃! (xy : ℝ × ℝ),
    let x := xy.1 in
    let y := xy.2 in
    9 ^ (x^2 + y) + 9 ^ (x + y^2) + 9 ^ (-x - y) = 3 := 
sorry

end unique_solution_count_l386_386687


namespace evaluate_expression_l386_386510

theorem evaluate_expression : 
  60 + 120 / 15 + 25 * 16 - 220 - 420 / 7 + 3 ^ 2 = 197 :=
by
  sorry

end evaluate_expression_l386_386510


namespace triangle_geometry_l386_386396

theorem triangle_geometry (A B C D : ℝ) 
  (H : D ∈ (B, C)) :
  (AD^2 * BC = AB^2 * CD + AC^2 * BD - BC * BD * CD) :=
sorry

end triangle_geometry_l386_386396


namespace arccos_cos_nine_l386_386622

theorem arccos_cos_nine : 
  ∀ (x : ℝ), x = 9 - 2 * Real.pi → Real.arccos (Real.cos 9) = x :=
by
  assume x h
  rw h
  sorry

end arccos_cos_nine_l386_386622


namespace find_f_l386_386285

-- Define the conditions
def g (x : ℝ) : ℝ := 2 * x + 3
def f (x : ℝ) : ℝ := g (x + 2)

-- State the theorem
theorem find_f :
  ∀ x : ℝ, f x = 2 * x + 7 :=
by
  sorry

end find_f_l386_386285


namespace triangle_circumcircle_ratio_l386_386336

variables {A B C P Q R X Y Z : Type}
variable [metric_space X] [metric_space Y] [metric_space Z]

-- Declare the points 
variables (A B C : Point X) (P : X) (Q : Y) (R : Z)

-- Assume P, Q, R are on the sides of triangle ABC
variable (hPBC : onSide P B C)
variable (hQCA : onSide Q C A)
variable (hRAB : onSide R A B)

-- Structures representing the circumcircles of triangles AQR, BRP, and CPQ
variable (ΓA : Circle (Triangle A Q R))
variable (ΓB : Circle (Triangle B R P))
variable (ΓC : Circle (Triangle C P Q))

-- Intersections of AP with the circumcircles at points X, Y, Z
variable (hX : intersects (lineThrough A P) ΓA X)
variable (hY : intersects (lineThrough A P) ΓB Y)
variable (hZ : intersects (lineThrough A P) ΓC Z)

theorem triangle_circumcircle_ratio (h1 : ∀ {P Q R X Y Z : X}, 
  onSide P B C ∧ onSide Q C A ∧ onSide R A B → 
  intersects (lineThrough A P) (circumcircle (triangle A Q R)) X → 
  intersects (lineThrough A P) (circumcircle (triangle B R P)) Y → 
  intersects (lineThrough A P) (circumcircle (triangle C P Q)) Z → 
  (radiusRatio Y X Z = distanceRatio B P C)):
(ProveProblem hPBC hQCA hRAB hX hY hZ triangle_circumcircle_ratio) :=
sorry

end triangle_circumcircle_ratio_l386_386336


namespace magic_square_min_changes_l386_386570

theorem magic_square_min_changes :
  let square := λ (r c : Fin 3), if r = 0
                                   then [2, 7, 6].get c
                                   else if r = 1
                                        then [9, 5, 1].get c
                                        else [4, 3, 8].get c,
      row_sum (r : Fin 3) := (Finset.univ.sum (λ c, square r c)),
      col_sum (c : Fin 3) := (Finset.univ.sum (λ r, square r c))
  in (∃ (change : Fin 3 × Fin 3 → ℕ),
        (∀ (r c : Fin 3), if r = 0 ∧ c = 0 then change (⟨r, c⟩) = 10
                         else if r = 0 ∧ c = 1 then change (⟨r, c⟩) = 0
                         else if r = 1 ∧ c = 0 then change (⟨r, c⟩) = 0
                         else if r = 1 ∧ c = 1 then change (⟨r, c⟩) = 5
                         else change (⟨r, c⟩) = square r c) ∧
        row_sum 0 ≠ row_sum 1 ∧ row_sum 1 ≠ row_sum 2 ∧ row_sum 2 ≠ row_sum 0 ∧
        col_sum 0 ≠ col_sum 1 ∧ col_sum 1 ≠ col_sum 2 ∧ col_sum 2 ≠ col_sum 0)
  ∧
  (∀ (change : Fin 3 × Fin 3 → ℕ),
     (∀ (r c : Fin 3), if r = 0 ∧ c = 0 then change (⟨r, c⟩) = 10
                      else if r = 0 ∧ c = 1 then change (⟨r, c⟩) = 0
                      else if r = 1 ∧ c = 0 then change (⟨r, c⟩) = 0
                      else if r = 1 ∧ c = 1 then change (⟨r, c⟩) = 5
                      else change (⟨r, c⟩) = square r c) →
     (row_sum 0 = row_sum 1 ∨ row_sum 1 = row_sum 2 ∨ row_sum 2 = row_sum 0 ∨
      col_sum 0 = col_sum 1 ∨ col_sum 1 = col_sum 2 ∨ col_sum 2 = col_sum 0) →
     (Σ (r c : Fin 3), square r c ≠ change (⟨r, c⟩)) ≥ 4) :=
by
  -- Proof to be provided
  sorry

end magic_square_min_changes_l386_386570


namespace sum_of_coefficients_l386_386689

theorem sum_of_coefficients : 
  (Polynomial.sum (5 * (2 * X^9 + 5 * X^7 - 4 * X^3 + 6) + 8 * (X^8 - 9 * X^3 + 3))) = 5 := 
by sorry

end sum_of_coefficients_l386_386689


namespace original_price_calculation_l386_386196

-- Definitions directly from problem conditions
def price_after_decrease (original_price : ℝ) : ℝ := 0.76 * original_price
def new_price : ℝ := 988

-- Statement embedding our problem
theorem original_price_calculation (x : ℝ) (hx : price_after_decrease x = new_price) : x = 1300 :=
by
  sorry

end original_price_calculation_l386_386196


namespace exists_multiple_of_three_in_circle_l386_386012

theorem exists_multiple_of_three_in_circle (n : ℕ) (h : n = 99) (a : ℕ → ℕ) :
  (∀ i : ℕ, i < n → (a ((i + 1) % n) - a i = 1 ∨ a ((i + 1) % n) - a i = 2 ∨ a ((i + 1) % n) = 2 * a i)) →
  ∃ i : ℕ, i < n ∧ a i % 3 = 0 :=
by
  have h := 99,
  sorry

end exists_multiple_of_three_in_circle_l386_386012


namespace minimum_vertical_distance_l386_386473

-- Definitions of the functions
def f (x : ℝ) : ℝ := abs x
def g (x : ℝ) : ℝ := -x^2 - 5*x - 4

-- Statement of the problem
theorem minimum_vertical_distance : 
  ∃ x : ℝ, (f x - g x) = 0 :=
begin
  sorry
end

end minimum_vertical_distance_l386_386473


namespace four_digit_numbers_count_l386_386856

theorem four_digit_numbers_count : 
  let total_count := 
    (Nat.choose 9 3) * (Nat.choose 3 1) * (Nat.choose 4 2) * 2! +
    (Nat.choose 9 2) * (Nat.choose 2 1) * 3 * 3 +
    (Nat.choose 9 2) * (Nat.choose 3 2) * 2! in
  total_count = 3888 :=
by
  sorry

end four_digit_numbers_count_l386_386856


namespace angle_APD_64_l386_386013

theorem angle_APD_64 
  {A B C D P : Point} 
  (h_AB_eq_CD : dist A B = dist C D)
  (h_angle_A : ∠A = 150)
  (h_angle_B : ∠B = 44)
  (h_angle_C : ∠C = 72)
  (h_perpendicular_bisector_AD_meets_P : is_perpendicular_bisector (segment A D) (segment B C) P)
  : ∠APD = 64 := 
sorry

end angle_APD_64_l386_386013


namespace volume_ratio_l386_386887

theorem volume_ratio (r : ℝ) (r_sphere : ℝ := r) (r_hemisphere : ℝ := 3 * r) :
  let V_sphere := (4 / 3) * Real.pi * r_sphere^3,
      V_hemisphere := (1 / 2) * (4 / 3) * Real.pi * r_hemisphere^3 in
  V_sphere / V_hemisphere = 1 / 13.5 :=
sorry

end volume_ratio_l386_386887


namespace card_game_digits_l386_386776

theorem card_game_digits :
  ∃ (A B : ℕ), A = 8 ∧ B = 0 ∧ (binom 60 8 = 7580800000) :=
by
  use 8
  use 0
  split; sorry


end card_game_digits_l386_386776


namespace num_coprime_to_15_l386_386636

theorem num_coprime_to_15 :
  {a : ℕ | a < 15 ∧ Nat.gcd 15 a = 1}.toFinset.card = 8 :=
by
  sorry

end num_coprime_to_15_l386_386636


namespace solution_for_system_l386_386749
open Real

noncomputable def solve_system (a b x y : ℝ) : Prop :=
  (a * x + b * y = 7 ∧ b * x + a * y = 8)

noncomputable def solve_linear (a b m n : ℝ) : Prop :=
  (a * (m + n) + b * (m - n) = 7 ∧ b * (m + n) + a * (m - n) = 8)

theorem solution_for_system (a b : ℝ) : solve_system a b 2 3 → solve_linear a b (5/2) (-1/2) :=
by {
  sorry
}

end solution_for_system_l386_386749


namespace roman_cannot_buy_all_l386_386093

theorem roman_cannot_buy_all (T x : ℝ) :
  (T - (x + 4) = 2) ∧ (T - (x + 3) = 3) ∧ (T - x = 6) →
  ¬ (T ≥ 3 * x + 7) :=
by
  intros h
  cases h with h1 h2
  cases h2 with h2 h3
  sorry

end roman_cannot_buy_all_l386_386093


namespace required_jogging_speed_l386_386041

-- Definitions based on the conditions
def blocks_to_miles (blocks : ℕ) : ℚ := blocks * (1 / 8 : ℚ)
def time_in_hours (minutes : ℕ) : ℚ := minutes / 60

-- Constants provided by the problem
def beach_distance_in_blocks : ℕ := 16
def ice_cream_melt_time_in_minutes : ℕ := 10

-- The main statement to prove
theorem required_jogging_speed :
  let distance := blocks_to_miles beach_distance_in_blocks
  let time := time_in_hours ice_cream_melt_time_in_minutes
  (distance / time) = 12 := by
  sorry

end required_jogging_speed_l386_386041


namespace min_n_geometric_series_l386_386752

theorem min_n_geometric_series (n : ℕ) (h : 1 + 2 + 2^2 + ... + 2^n > 128) : n = 7 := sorry

end min_n_geometric_series_l386_386752


namespace perpendicular_bisector_equidistant_l386_386854

variable {Point : Type} [MetricSpace Point] {a b p : Point}

def is_perpendicular_bisector (a b p : Point) : Prop :=
  dist a p = dist b p

theorem perpendicular_bisector_equidistant (p a b: Point) 
  (h : is_perpendicular_bisector a p b p) : 
  dist p a = dist p b :=
by
  sorry

end perpendicular_bisector_equidistant_l386_386854


namespace circles_intersect_l386_386227

noncomputable def Circle1 : Nat -> Nat -> Nat -> Prop := 
  fun x y r => x^2 + y^2 + 6 * x - 4 * y + 10 = 0

noncomputable def Circle2 : Nat -> Nat -> Nat -> Prop := 
  fun x y r => x^2 + y^2 = 4

theorem circles_intersect 
  (x1 y1 r1: Nat) (x2 y2 r2: Nat)
  (h1 : Circle1 x1 y1 r1) 
  (h2 : Circle2 x2 y2 r2) : 
  r1 - r2 < (sqrt ((x1 - x2)^2 + (y1 - y2)^2) : Real) ∧ 
  (sqrt ((x1 - x2)^2 + (y1 - y2)^2) : Real) < r1 + r2 :=
sorry

end circles_intersect_l386_386227


namespace value_of_y_in_arithmetic_sequence_l386_386535

theorem value_of_y_in_arithmetic_sequence :
    ∃ y : ℤ, (arithmetic_sequence (3^2) y (3^4)) ∧ y = 45 := by
  -- Here we define the arithmetic sequence condition.
  def arithmetic_sequence (a b c : ℤ) : Prop := b = (a + c) / 2
  sorry

end value_of_y_in_arithmetic_sequence_l386_386535


namespace rhombus_perimeter_l386_386467

-- Define the rhombus with diagonals of given lengths and prove the perimeter is 52 inches
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) (d1_pos : 0 < d1) (d2_pos : 0 < d2):
  let s := sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2) in 
  let perimeter := 4 * s in
  perimeter = 52 
  :=
by
  sorry  -- Proof is skipped

end rhombus_perimeter_l386_386467


namespace largest_prime_divisor_12plus13_l386_386257

noncomputable def factorial : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * factorial n

-- Assume factorial is already defined as expected by Mathlib 

theorem largest_prime_divisor_12plus13 :
  let a := factorial 12
  let b := factorial 13
  let N := a + 13 * a
  prime 13 ∧ ¬(∃ p, prime p ∧ p > 13 ∧ p ∣ N) :=
begin
  sorry, -- Proof goes here
end

end largest_prime_divisor_12plus13_l386_386257


namespace side_length_of_S2_l386_386423

variables (s r : ℕ)

-- Conditions
def combined_width_eq : Prop := 3 * s + 100 = 4000
def combined_height_eq : Prop := 2 * r + s = 2500

-- Conclusion we want to prove
theorem side_length_of_S2 : combined_width_eq s → combined_height_eq s r → s = 1300 :=
by
  intros h_width h_height
  sorry

end side_length_of_S2_l386_386423


namespace sum_real_product_real_sum_and_product_real_l386_386549

-- Part (a): Sum of z1 and z2 is a real number
theorem sum_real (a b c : ℝ) : ∃ (z1 z2 : ℂ), z1 = complex.ofReal a + complex.I * b ∧ z2 = complex.ofReal c - complex.I * b ∧ (z1 + z2).im = 0 :=
by
  let z1 : ℂ := complex.ofReal a + complex.I * b
  let z2 : ℂ := complex.ofReal c - complex.I * b
  use [z1, z2]
  simp [complex.add_im, complex.ofReal_im]

-- Part (b): Product of z1 and z2 is a real number
theorem product_real (a b k : ℝ) : ∃ (z1 z2 : ℂ), z1 = complex.ofReal a + complex.I * b ∧ z2 = k * (complex.ofReal a + complex.I * b) ∧ (z1 * z2).im = 0 :=
by
  let z1 : ℂ := complex.ofReal a + complex.I * b
  let z2 : ℂ := k * (complex.ofReal a + complex.I * b)
  use [z1, z2]
  simp [complex.mul_im, complex.ofReal_im]
  rw [mul_assoc, complex.ofReal_im, complex.I_im, mul_zero]

-- Part (c): Sum and product of z1 and z2 are real numbers
theorem sum_and_product_real (a b : ℝ) : ∃ (z1 z2 : ℂ), z1 = complex.ofReal a + complex.I * b ∧ z2 = complex.ofReal a - complex.I * b ∧ (z1 + z2).im = 0 ∧ (z1 * z2).im = 0 :=
by
  let z1 : ℂ := complex.ofReal a + complex.I * b
  let z2 : ℂ := complex.ofReal a - complex.I * b
  use [z1, z2]
  simp [complex.add_im, complex.mul_im, complex.ofReal_im, complex.I_im]
  rw [add_neg_eq_zero, sub_self]
  split
  . assumption
  . simp [complex.mul_im, mul_assoc]
  rw [complex.ofReal_im, complex.I_im, mul_zero]


end sum_real_product_real_sum_and_product_real_l386_386549


namespace prime_factors_30_factorial_l386_386224

theorem prime_factors_30_factorial : 
  ∀ (n : ℕ), (n = 30) → (∃ (primes : Finset ℕ), (∀ p ∈ primes, Nat.Prime p) ∧ primes.card = 10) :=
by
  intros n hn
  rw hn
  let primes := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}.toFinset
  have h_primes: ∀ p ∈ primes, Nat.Prime p := sorry
  have h_card: primes.card = 10 := by
    rw [Finset.card, List.length]
    rfl
  exact ⟨primes, h_primes, h_card⟩

end prime_factors_30_factorial_l386_386224


namespace ways_to_draw_balls_eq_total_ways_l386_386566

noncomputable def ways_to_draw_balls (n : Nat) :=
  if h : n = 15 then (15 * 14 * 13 * 12) else 0

noncomputable def valid_combinations : Nat := sorry

noncomputable def total_ways_to_draw : Nat :=
  valid_combinations * 24

theorem ways_to_draw_balls_eq_total_ways :
  ways_to_draw_balls 15 = total_ways_to_draw :=
sorry

end ways_to_draw_balls_eq_total_ways_l386_386566


namespace jack_jog_speed_l386_386036

theorem jack_jog_speed (melt_time_minutes : ℕ) (distance_blocks : ℕ) (block_length_miles : ℚ) 
    (h_melt_time : melt_time_minutes = 10)
    (h_distance : distance_blocks = 16)
    (h_block_length : block_length_miles = 1/8) :
    let time_hours := (melt_time_minutes : ℚ) / 60
    let distance_miles := (distance_blocks : ℚ) * block_length_miles
        12 = distance_miles / time_hours :=
by
  sorry

end jack_jog_speed_l386_386036


namespace linear_increasing_function_l386_386843

theorem linear_increasing_function (x : ℝ) : (∃ k b : ℝ, k > 0 ∧ (∀ x : ℝ, f x = k * x + b) ∧ f x = x - 2) :=
sorry

end linear_increasing_function_l386_386843


namespace inequality_proof_l386_386724

variables {x y a b ε m : ℝ}

theorem inequality_proof (h1 : |x - a| < ε / (2 * m))
                        (h2 : |y - b| < ε / (2 * |a|))
                        (h3 : 0 < y ∧ y < m) :
                        |x * y - a * b| < ε :=
sorry

end inequality_proof_l386_386724


namespace proof_A_and_area_l386_386006

noncomputable def angle_A (m n : ℝ × ℝ) := (1 : ℝ)
noncomputable def measure_angle_A (A : ℝ) :=
  let m := (Real.sin (A / 2), Real.cos (A / 2)) in
  let n := (Real.cos (A / 2), -Real.cos (A / 2)) in
  2 * (m.1 * n.1 + m.2 * n.2) + Real.sqrt (m.1 ^ 2 + m.2 ^ 2) = Real.sqrt 2 / 2

noncomputable def dot_product_ab_ac (A : ℝ) :=
  let AB := (Real.cos A, Real.sin A) in
  let AC := (Real.sin (A / 2), Real.cos (A / 2)) in
  AB.1 * AC.1 + AB.2 * AC.2 = 1

noncomputable def triangle_area (A : ℝ) :=
  (1/2) * Real.tan A = (2 + Real.sqrt 3) / 2

theorem proof_A_and_area:
  ∃ A : ℝ, measure_angle_A A ∧ dot_product_ab_ac A ∧ triangle_area A := 
sorry

end proof_A_and_area_l386_386006


namespace find_digits_l386_386671

theorem find_digits (A B C D : ℕ) (h1 : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
(h2 : 1 ≤ A ∧ A ≤ 9)
(h3 : 1 ≤ B ∧ B ≤ 9)
(h4 : 1 ≤ C ∧ C ≤ 9)
(h5 : 1 ≤ D ∧ D ≤ 9)
(h6 : (10 * A + B) * (10 * C + B) = 111 * D)
(h7 : (10 * A + B) < (10 * C + B)) :
A = 2 ∧ B = 7 ∧ C = 3 ∧ D = 9 :=
sorry

end find_digits_l386_386671


namespace greatest_possible_D_l386_386447

theorem greatest_possible_D :
  ∃ D : ℝ, (∀ (d : ℝ),
    (-D < d ∧ d < D) →
    let angles := [120 - 5 * d / 2, 120 - 3 * d / 2, 120 - d / 2, 120 + d / 2, 120 + 3 * d / 2, 120 + 5 * d / 2] in
    ∀ θ ∈ angles, 0 < θ ∧ θ < 180) ∧ D = 24 :=
sorry

end greatest_possible_D_l386_386447


namespace original_selling_price_of_book_l386_386965

theorem original_selling_price_of_book:
  ∃ (CP OSP: ℝ), (1.10 * CP = 880) ∧ (OSP = 0.90 * CP) ∧ (OSP = 720) :=
by
  use 800
  use 720
  split
  -- Condition matching
  · norm_num
  -- Second Condition matching
  · norm_num
  sorry -- The proof steps will go here but are omitted as per instructions.

end original_selling_price_of_book_l386_386965


namespace hyperbola_problem_l386_386110

theorem hyperbola_problem :
  let C : ℝ → ℝ → Prop := λ x y, 2*x^2 - y^2 = 1 in  -- Equation of the hyperbola
  let line: ℝ → ℝ → Prop := λ k, ∀ x y, y = k * x - 1 in -- Equation of the line
  (∃ k P Q, line k P.y ∧ line k Q.y ∧ C P.x P.y ∧ C Q.x Q.y ∧ 
  let OP : ℝ × ℝ := (P.x, P.y) in
  let OQ : ℝ × ℝ := (Q.x, Q.y) in
  (OP.1 * OQ.1 + OP.2 * OQ.2 = 0)) → k = 0 :=
begin
  sorry,
end

end hyperbola_problem_l386_386110


namespace asymptote_hole_count_l386_386357

noncomputable def f (x : ℝ) : ℝ := (x^2 - x - 2) / (x^3 - 2x^2 - x + 2)

theorem asymptote_hole_count :
  let a := 1 in
  let b := 2 in
  let c := 1 in
  let d := 0 in
  a + 2 * b + 3 * c + 4 * d = 8 := by
  sorry

end asymptote_hole_count_l386_386357


namespace days_to_fulfill_order_l386_386912

theorem days_to_fulfill_order (bags_per_batch : ℕ) (total_order : ℕ) (initial_bags : ℕ) (required_days : ℕ) :
  bags_per_batch = 10 →
  total_order = 60 →
  initial_bags = 20 →
  required_days = (total_order - initial_bags) / bags_per_batch →
  required_days = 4 :=
by
  intros
  sorry

end days_to_fulfill_order_l386_386912


namespace sum_proper_divisors_243_l386_386924

theorem sum_proper_divisors_243 : (1 + 3 + 9 + 27 + 81) = 121 := by
  sorry

end sum_proper_divisors_243_l386_386924


namespace brother_initially_had_17_l386_386833

/-
Michael has $42. Michael gives away half the money to his brother.
His brother buys $3 worth of candy. His brother has $35 left.
Prove that Michael's brother initially had $17.
-/

def michael_initial_money : ℕ := 42

def amount_given_to_brother : ℕ := michael_initial_money / 2

def candy_cost : ℕ := 3

def brother_money_left_after_candy : ℕ := 35

def brother_initial_money : ℕ :=
  brother_money_left_after_candy + candy_cost

theorem brother_initially_had_17 :
  brother_initial_money - amount_given_to_brother = 17 :=
by
  unfold michael_initial_money
  unfold amount_given_to_brother
  unfold candy_cost
  unfold brother_money_left_after_candy
  unfold brother_initial_money
  sorry

end brother_initially_had_17_l386_386833


namespace arithmetic_sequence_nth_term_l386_386121

noncomputable def geometric_mean (a b : ℝ) : ℝ := sqrt (a * b)

theorem arithmetic_sequence_nth_term (a : ℕ → ℝ) (d : ℝ) (h_nonzero_d : d ≠ 0) 
  (h_a1 : a 1 = 1) (h_a3 : a 3 = geometric_mean (a 1) (a 9))
  (h_arithmetic : ∀ n, a (n + 1) = a n + d) :
  ∀ n, a n = n := 
begin
  sorry
end

end arithmetic_sequence_nth_term_l386_386121


namespace find_x_l386_386000

theorem find_x (x : ℝ) (h : max (max 1 (max 2 (max 3 x))) (min (min 1 (min 2 (min 3 x)))) - min (min 1 (min 2 (min 3 x))) = 1 + 2 + 3 + x) : 
  x = -3 / 2 :=
sorry

end find_x_l386_386000


namespace hitting_target_at_least_once_complementary_event_l386_386180

theorem hitting_target_at_least_once_complementary_event :
  ∀ (n : ℕ), (n = 3) → (event_hitting_target_at_least_once n) = (event_not_hitting_target_a_single_time n) ∧ (is_complementary_event (event_hitting_target_at_least_once n) (event_not_hitting_target_a_single_time n)) := 
by
  intros n hn
  sorry

end hitting_target_at_least_once_complementary_event_l386_386180


namespace q_at_2_l386_386212

def p (x : ℝ) : ℝ := x^2 - 2 * x + 1
def q (x : ℝ) : ℝ := p (p x)

theorem q_at_2:
  (∀ x : ℝ, q(x) = 0 → (∃ a b c : ℝ, q(a) = 0 ∧ q(b) = 0 ∧ q(c) = 0 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c)) →
  q 2 = -1 :=
by
  sorry

end q_at_2_l386_386212


namespace floor_sum_inequality_l386_386276

theorem floor_sum_inequality (n : ℕ) (x : ℝ) : 
    (⟦n * x⟧ : ℝ) ≥ ∑ k in finset.range(n+1), (⟦k * x⟧ / k) :=
begin
  sorry
end

end floor_sum_inequality_l386_386276


namespace distance_between_A_and_C_l386_386760

-- Definitions
variable (A B C : Type) -- points
variable [metric_space A] [metric_space B] [metric_space C]
variable (AB AC BC : ℝ) -- lengths

-- Conditions
axiom AB_eq_5 : AB = 5
axiom BC_eq_4 : BC = 4
axiom AC_pos : 0 < AC

-- Theorem to prove
theorem distance_between_A_and_C : 
  ¬ (AC = 1 ∨ AC = 9) :=
  sorry

end distance_between_A_and_C_l386_386760


namespace bead_cost_l386_386616

/--
Given:
- 50 rows of purple beads, each with 20 beads.
- 40 rows of blue beads, each with 18 beads.
- 80 gold beads.
- Cost of beads is $1 per 10 beads.

Prove that the total cost of all beads is $180.
-/
theorem bead_cost : 
  let purple_beads := 50 * 20 in
  let blue_beads := 40 * 18 in
  let gold_beads := 80 in
  let total_beads := purple_beads + blue_beads + gold_beads in
  (total_beads / 10) = 180 := 
by
  let purple_beads := 50 * 20
  let blue_beads := 40 * 18
  let gold_beads := 80
  let total_beads := purple_beads + blue_beads + gold_beads
  trivial

end bead_cost_l386_386616


namespace circle_radius_value_l386_386697

theorem circle_radius_value (k : ℝ) :
  (∀ x y : ℝ, x^2 - 8*x + y^2 + 10*y + k = 0 ↔ (x - 4)^2 + (y + 5)^2 = 25) → k = 16 :=
by
  sorry

end circle_radius_value_l386_386697


namespace probability_at_least_one_black_ball_l386_386895

theorem probability_at_least_one_black_ball :
  let total_balls := 6
  let red_balls := 2
  let white_ball := 1
  let black_balls := 3
  let total_combinations := Nat.choose total_balls 2
  let non_black_combinations := Nat.choose (total_balls - black_balls) 2
  let probability := 1 - (non_black_combinations / total_combinations : ℚ)
  probability = 4 / 5 :=
by
  sorry

end probability_at_least_one_black_ball_l386_386895


namespace arithmetic_seq_middle_term_l386_386522

theorem arithmetic_seq_middle_term (a1 a3 y : ℤ) (h1 : a1 = 3^2) (h2 : a3 = 3^4)
    (h3 : y = (a1 + a3) / 2) : y = 45 :=
by
  rw [h1, h2] at h3
  simp at h3
  exact h3

end arithmetic_seq_middle_term_l386_386522


namespace reconstruct_quadrilateral_l386_386289

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (E E' F F' G G' H H' : V)

/-- Given conditions -/
def condition1 : E ≠ E' → F = (1/2 : ℝ) • E + (1/2 : ℝ) • E' := sorry

def condition2 : F ≠ F' → G = (1/2 : ℝ) • F + (1/2 : ℝ) • F' := sorry

def condition3 : G ≠ G' → H = (1/2 : ℝ) • G + (1/2 : ℝ) • G' := sorry

/-- Main theorem to prove -/
theorem reconstruct_quadrilateral (E' F' G' H' : V)
  (h1 : condition1 E E' F)
  (h2 : condition2 F F' G)
  (h3 : condition3 G G' H) :
  E = (1/15 : ℝ) • E' + (2/15 : ℝ) • F' +
      (4/15 : ℝ) • G' + (8/15 : ℝ) • H' := sorry

end reconstruct_quadrilateral_l386_386289


namespace eggs_left_after_taking_l386_386900

def eggs_in_box_initial : Nat := 47
def eggs_taken_by_Harry : Nat := 5
theorem eggs_left_after_taking : eggs_in_box_initial - eggs_taken_by_Harry = 42 := 
by
  -- Proof placeholder
  sorry

end eggs_left_after_taking_l386_386900


namespace sum_of_coefficients_256_l386_386303

theorem sum_of_coefficients_256 (n : ℕ) (h : (3 + 1)^n = 256) : n = 4 :=
sorry

end sum_of_coefficients_256_l386_386303


namespace count_coprime_with_15_lt_15_l386_386650

theorem count_coprime_with_15_lt_15 :
  {a : ℕ // a < 15 ∧ Nat.coprime 15 a}.to_finset.card = 8 := 
sorry

end count_coprime_with_15_lt_15_l386_386650


namespace triangle_sides_l386_386369

theorem triangle_sides
  (D E : ℝ) (DE EF : ℝ)
  (h1 : Real.cos (2 * D - E) + Real.sin (D + E) = 2)
  (hDE : DE = 6) :
  EF = 3 :=
by
  -- Proof is omitted
  sorry

end triangle_sides_l386_386369


namespace problem_statement_l386_386667

theorem problem_statement :
  (∃ n : ℕ, n = 8 ∧ ∀ a : ℕ, a < 15 → (∃ x : ℤ, a * x ≡ 1 [MOD 15]) ↔ gcd a 15 = 1) :=
by
  use 8
  intro a
  intro ha
  split
  sorry

end problem_statement_l386_386667


namespace largest_integer_smaller_than_expression_l386_386544

theorem largest_integer_smaller_than_expression :
  let expr := (Real.sqrt 5 + Real.sqrt (3 / 2)) ^ 8
  ⌊expr⌋ = 7168 :=
by
  let expr := (Real.sqrt 5 + Real.sqrt (3 / 2)) ^ 8
  have h : expr < 7169 := sorry
  have h' : 7168 ≤ expr := sorry
  exact Int.floor_eq_iff.mpr ⟨h', h⟩

end largest_integer_smaller_than_expression_l386_386544


namespace correct_number_of_true_propositions_l386_386737

section
-- Definitions of the propositions
def prop1 (A B : Set) : Prop := A ∩ B = A → A ⊆ B
def prop2 (p q : Prop) : Prop := (p ∨ q) → (p ∧ q)
def prop3 (a b m : ℝ) : Prop := a < b → a * m^2 < b * m^2
def prop4 (a : ℝ) : Prop := (∀ x y : ℝ, a*x + y + 1 = 0 → x - y + 1 = 0 → a = 1)

-- The main proposition evaluating the number of true propositions
def numberOfTruePropositions : Prop :=
  (prop1 Set.empty Set.univ) + 
  (¬ prop2 True False) + 
  (¬ prop3 1 2 0) + 
  (prop4 1) = 2
end

-- Statement of the theorem
theorem correct_number_of_true_propositions : numberOfTruePropositions := 
  sorry

end correct_number_of_true_propositions_l386_386737


namespace cost_of_7_cubic_yards_l386_386137

def cost_per_cubic_foot : ℕ := 8
def cubic_feet_per_cubic_yard : ℕ := 27
def cubic_yards : ℕ := 7

theorem cost_of_7_cubic_yards
  (c : ℕ) (c_cubic : c = cost_per_cubic_foot)
  (f : ℕ) (f_cubic : f = cubic_feet_per_cubic_yard)
  (y : ℕ) (y_cubic : y = cubic_yards) :
  c * f * y = 1512 :=
begin
  sorry
end

end cost_of_7_cubic_yards_l386_386137


namespace balance_difference_15_years_l386_386600

noncomputable def alice_balance (principal : ℕ) (rate : ℕ) (years : ℕ) : ℕ :=
  principal * (1.05 ^ years.to_real).to_nat

noncomputable def charlie_balance (principal : ℕ) (rate : ℕ) (years : ℕ) : ℕ :=
  principal * (1 + years * (rate / 100))

theorem balance_difference_15_years :
  let principal_alice := 9000
  let rate_alice := 5
  let principal_charlie := 11000
  let rate_charlie := 6
  let years := 15
  let diff := (charlie_balance principal_charlie rate_charlie years) - (alice_balance principal_alice rate_alice years)
  abs diff = 2189 :=
by
  sorry

end balance_difference_15_years_l386_386600


namespace rho_cos_phi_describes_sphere_l386_386694

noncomputable def sphere (r : ℝ) := ∀ (ϕ : ℝ), 0 ≤ ϕ ∧ ϕ ≤ π → ρ ϕ = real.cos ϕ

theorem rho_cos_phi_describes_sphere : 
  ∀ (ρ : ℝ → ℝ) (ϕ : ℝ),
    (0 ≤ ϕ ∧ ϕ ≤ π) →
    ρ ϕ = real.cos ϕ →
    ∀ (r : ℝ), r = 1 →
    sphere r :=
by
  intros ρ ϕ hϕ hρ r hr
  intro ϕ hϕ'
  sorry

end rho_cos_phi_describes_sphere_l386_386694


namespace coprime_count_15_l386_386660

theorem coprime_count_15 :
  {a : ℕ | a < 15 ∧ Nat.gcd 15 a = 1}.card = 8 :=
by
sorry

end coprime_count_15_l386_386660


namespace range_of_a_l386_386228

open Real

theorem range_of_a (a : ℝ) : (∀ x : ℝ, a * x ^ 2 + a * x - 1 < 0) ↔ -4 < a ∧ a ≤ 0 := 
by
  sorry

end range_of_a_l386_386228


namespace find_length_of_MN_l386_386375

open Classical

variables {a b c x : ℝ}
variable (x : ℝ)

-- conditions: triangle ABC with sides a, b, c
-- line MN is parallel to AC
-- AM = BN = x

theorem find_length_of_MN (h1 : x = (a * c) / (a + c)) :
  ∃ MN, MN = (b * c) / (a + c) :=
begin
  use (b * c) / (a + c),
  sorry
end

end find_length_of_MN_l386_386375


namespace triangle_property_proof_l386_386372

noncomputable def triangleABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  a = 2 * Real.sqrt 2 ∧
  b = 5 ∧
  c = Real.sqrt 13 ∧
  C = Real.pi / 4 ∧
  ∃ sinA : ℝ, sinA = 2 * Real.sqrt 13 / 13 ∧
  ∃ sin_2A_plus_pi_4 : ℝ, sin_2A_plus_pi_4 = 17 * Real.sqrt 2 / 26

theorem triangle_property_proof :
  ∃ (A B C : ℝ), 
  triangleABC (2 * Real.sqrt 2) 5 (Real.sqrt 13) A B C
:= sorry

end triangle_property_proof_l386_386372


namespace quadratic_no_real_roots_l386_386484

theorem quadratic_no_real_roots (a b c : ℝ) (h : a = 1) (h1 : b = 3) (h2 : c = 3) :
  let D := b^2 - 4 * a * c in D < 0 → ¬∃ x : ℝ, a * x^2 + b * x + c = 0 :=
begin
  -- By computation, we know D = -3 which is less than 0.
  -- Thus, we will skip the proof using sorry.
  sorry
end

end quadratic_no_real_roots_l386_386484


namespace apples_given_by_anita_l386_386441

variable (initial_apples current_apples needed_apples : ℕ)

theorem apples_given_by_anita (h1 : initial_apples = 4) 
                               (h2 : needed_apples = 10)
                               (h3 : needed_apples - current_apples = 1) : 
  current_apples - initial_apples = 5 := 
by
  sorry

end apples_given_by_anita_l386_386441


namespace sum_proper_divisors_243_l386_386921

theorem sum_proper_divisors_243 : (1 + 3 + 9 + 27 + 81) = 121 :=
by
  sorry

end sum_proper_divisors_243_l386_386921


namespace water_breaks_vs_sitting_breaks_l386_386045

theorem water_breaks_vs_sitting_breaks :
  (240 / 20) - (240 / 120) = 10 := by
  sorry

end water_breaks_vs_sitting_breaks_l386_386045


namespace smallest_n_not_divisible_by_10_l386_386264

theorem smallest_n_not_divisible_by_10 :
  ∃ n : ℕ, n > 2016 ∧ (1^n + 2^n + 3^n + 4^n) % 10 ≠ 0 ∧ n = 2020 := by
  sorry

end smallest_n_not_divisible_by_10_l386_386264


namespace arithmetic_sequence_y_value_l386_386515

theorem arithmetic_sequence_y_value :
  ∃ y : ℤ, (∃ a1 a3 : ℤ, a1 = 9 ∧ a3 = 81 ∧ y = (a1 + a3) / 2) → y = 45 :=
by
  sorry

end arithmetic_sequence_y_value_l386_386515


namespace incorrect_step_2_l386_386152

-- Definitions of square roots and the problem constants
def a1 := 2 * Real.sqrt 3
def a2 := Real.sqrt (2^2 * 3)
def a3 := Real.sqrt 12
def b1 := -2 * Real.sqrt 3
def b2 := Real.sqrt ((-2)^2 * 3)

-- Theorem statement: contradiction of step ②
theorem incorrect_step_2 : b1 ≠ b2 :=
by sorry

end incorrect_step_2_l386_386152


namespace geometric_sequence_sum_sequence_l386_386786

-- Definitions
def seq_a : ℕ → ℤ 
| 0 := 4
| (n+1) := 2 * (seq_b n)

def seq_b : ℕ → ℤ 
| 0 := 2
| (n+1) := (seq_a n) + 2

-- Property (1): Prove the sequence {a_{2n-1} + 4} is a geometric sequence with constant ratio 2.
theorem geometric_sequence (n : ℕ) : ∃ r, ∀ n, r * (seq_a (2*n - 1) + 4) = seq_a (2*n + 1) + 4 :=
by sorry

-- Property (2): Find the sum of the first 2n terms of sequence {b_n}
def sum_b (n : ℕ) : ℤ :=
  ∑ i in range n, seq_b i

theorem sum_sequence (n : ℕ) : sum_b (2*n) = 3 * 2^(n+2) - 4 * n - 12 :=
by sorry

end geometric_sequence_sum_sequence_l386_386786


namespace two_griffins_l386_386773

def mythicalBeast : Type :=
| Unicorn
| Griffin

def Zara_statement (Walo Zara : mythicalBeast) : Prop :=
  (Walo = mythicalBeast.Unicorn ∧ Zara = mythicalBeast.Griffin) ∨ (Walo = mythicalBeast.Griffin ∧ Zara = mythicalBeast.Unicorn)

def Yumi_statement (Xixi : mythicalBeast) : Prop :=
  Xixi = mythicalBeast.Griffin

def Xixi_statement (Yumi : mythicalBeast) : Prop :=
  Yumi = mythicalBeast.Griffin

def Walo_statement (Zara Yumi Xixi Walo : mythicalBeast) : Prop :=
  (if Zara = mythicalBeast.Unicorn then 1 else 0) +
  (if Yumi = mythicalBeast.Unicorn then 1 else 0) +
  (if Xixi = mythicalBeast.Unicorn then 1 else 0) +
  (if Walo = mythicalBeast.Unicorn then 1 else 0) ≤ 1

theorem two_griffins (Zara Yumi Xixi Walo : mythicalBeast) 
  (hz : Zara_statement Walo Zara)
  (hy : Yumi_statement Xixi)
  (hx : Xixi_statement Yumi)
  (hw : Walo_statement Zara Yumi Xixi Walo) :
  (if Zara = mythicalBeast.Griffin then 1 else 0) +
  (if Yumi = mythicalBeast.Griffin then 1 else 0) +
  (if Xixi = mythicalBeast.Griffin then 1 else 0) +
  (if Walo = mythicalBeast.Griffin then 1 else 0) = 2 :=
sorry

end two_griffins_l386_386773


namespace melted_ice_cream_depth_l386_386186

noncomputable def ice_cream_depth : ℝ :=
  let r1 := 3 -- radius of the sphere
  let r2 := 10 -- radius of the cylinder
  let V_sphere := (4/3) * Real.pi * r1^3 -- volume of the sphere
  let V_cylinder h := Real.pi * r2^2 * h -- volume of the cylinder
  V_sphere / (Real.pi * r2^2)

theorem melted_ice_cream_depth :
  ice_cream_depth = 9 / 25 :=
by
  sorry

end melted_ice_cream_depth_l386_386186


namespace geometric_series_sum_l386_386204

theorem geometric_series_sum (a r : ℚ) (n : ℕ) (h_a : a = 1) (h_r : r = 1 / 2) (h_n : n = 5) :
  ((a * (1 - r^n)) / (1 - r)) = 31 / 16 := 
by
  sorry

end geometric_series_sum_l386_386204


namespace find_eccentricity_of_ellipse_l386_386735

noncomputable def eccentricity_of_ellipse (a b : ℝ) : ℝ := 
  (real.sqrt (a ^ 2 - b ^ 2)) / a

theorem find_eccentricity_of_ellipse 
  (a b : ℝ)
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : (a > 0))  
  (h4 : (∃ (x y : ℝ), (x = real.sqrt 3) ∧ (y = 1/2) 
     ∧ (x^2 / a^2 + y^2 / b^2 = 1)))
  (h5 : 2 * a = 4) : 
  eccentricity_of_ellipse a b = real.sqrt 3 / 2 := 
by 
  sorry

end find_eccentricity_of_ellipse_l386_386735


namespace megan_total_earnings_l386_386828

-- Define the constants
def work_hours_per_day := 8
def earnings_per_hour := 7.50
def work_days_per_month := 20

-- Define Megan's total earnings for two months
def total_earnings (work_hours_per_day : ℕ) (earnings_per_hour : ℝ) (work_days_per_month : ℕ) : ℝ :=
  2 * (work_hours_per_day * earnings_per_hour * work_days_per_month)

-- Prove that the total earnings for two months is $2400
theorem megan_total_earnings : total_earnings work_hours_per_day earnings_per_hour work_days_per_month = 2400 :=
by
  sorry

end megan_total_earnings_l386_386828


namespace volume_after_increasing_edges_l386_386490

-- Defining the initial conditions and the theorem to prove regarding the volume.
theorem volume_after_increasing_edges {a b c : ℝ} 
  (h1 : a * b * c = 8) 
  (h2 : (a + 1) * (b + 1) * (c + 1) = 27) : 
  (a + 2) * (b + 2) * (c + 2) = 64 :=
sorry

end volume_after_increasing_edges_l386_386490


namespace volume_is_correct_l386_386991

def volume_of_box (x : ℝ) : ℝ :=
  (14 - 2 * x) * (10 - 2 * x) * x

theorem volume_is_correct (x : ℝ) :
  volume_of_box x = 140 * x - 48 * x^2 + 4 * x^3 :=
by
  sorry

end volume_is_correct_l386_386991


namespace complex_sum_modulus_l386_386676

-- Conditions are derived as definitions
def complex1 : ℂ := (1/5 : ℝ) - (2/5 : ℝ) * I
def complex2 : ℂ := (3/5 : ℝ) + (4/5 : ℝ) * I

-- The theorem we need to prove
theorem complex_sum_modulus :
  complex.abs complex1 + complex.abs complex2 = (1 + real.sqrt 5) / real.sqrt 5 :=
by
  sorry

end complex_sum_modulus_l386_386676


namespace total_people_seated_l386_386857

theorem total_people_seated (d b : ℕ) :
  (d - 7 = 12) → -- condition related to girls forming pairs
  (0.75 * b = 12) → -- condition related to boys with girls to their right
  d + b = 35 :=         -- conclusion
by
  intro h1 h2
  have h3 : 0.75 * (b : ℝ) = 12 := by exact_mod_cast h2
  have b_eq : b = 16 := by linarith
  have d_eq : d = 19 := by linarith
  linarith

end total_people_seated_l386_386857


namespace inequality_solution_set_l386_386311

variable {a b x : ℝ}

theorem inequality_solution_set (h : ∀ x : ℝ, ax - b > 0 ↔ x < -1) : 
  ∀ x : ℝ, (x-2) * (ax + b) < 0 ↔ x < 1 ∨ x > 2 :=
by sorry

end inequality_solution_set_l386_386311


namespace solution_set_f_eq_half_l386_386307

def f(x : ℝ) : ℝ :=
  if x ≤ 0 then 2^x else abs (Real.log2 x)

theorem solution_set_f_eq_half :
  {x : ℝ | f(x) = 1 / 2} = {-1, Real.sqrt 2, Real.sqrt 2 / 2} := 
by
  sorry

end solution_set_f_eq_half_l386_386307


namespace case_a_false_case_b_true_l386_386896

-- Definitions for Case (a) where n = k
def cannot_partition_when_n_equals_k (k : ℕ) (weights : List ℕ) : Prop :=
  weights.length = k ∧ weights.sum = 2 * k → ¬(∃ A B : List ℕ, A ++ B = weights ∧ A.sum = B.sum = k)

-- Definitions for Case (b) where n = k + 1
def can_partition_when_n_equals_k_plus_one (k : ℕ) (weights : List ℕ) : Prop :=
  weights.length = k + 1 ∧ weights.sum = 2 * k → ∃ A B : List ℕ, A ++ B = weights ∧ A.sum = B.sum = k

-- Lean 4 statements for the respective cases
theorem case_a_false (k : ℕ) (weights : List ℕ) :
  cannot_partition_when_n_equals_k k weights :=
by sorry

theorem case_b_true (k : ℕ) (weights : List ℕ) :
  can_partition_when_n_equals_k_plus_one k weights :=
by sorry

end case_a_false_case_b_true_l386_386896


namespace arithmetic_sum_example_l386_386731

def S (n : ℕ) (a1 d : ℤ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

def a (n : ℕ) (a1 d : ℤ) : ℤ := a1 + (n - 1) * d

theorem arithmetic_sum_example (a1 d : ℤ) 
  (S20_eq_340 : S 20 a1 d = 340) :
  a 6 a1 d + a 9 a1 d + a 11 a1 d + a 16 a1 d = 68 :=
by
  sorry

end arithmetic_sum_example_l386_386731


namespace min_distance_from_P_to_line_l_l386_386779

-- Given parametric equations of line l
def line_l (t : ℝ) : ℝ × ℝ :=
  (-2 + 1/2 * t, (sqrt 3) / 2 * t)

-- Polar equation of curve C1
def curve_C1 (θ : ℝ) : ℝ × ℝ :=
  (sqrt 6 * cos θ, sqrt 6 * sin θ)

-- Transformed parametric equations for curve C2
def curve_C2 (θ : ℝ) : ℝ × ℝ :=
  (cos θ, sqrt 3 * sin θ)

-- Distance function from a point P on C2 to the line l
def distance_from_P_to_line_l (θ : ℝ) : ℝ :=
  let P := curve_C2 θ in
  |sqrt 3 * P.1 - P.2 + 2 * sqrt 3| / sqrt (3 + 1)

-- Stating the minimum distance in Lean
theorem min_distance_from_P_to_line_l : 
  ∃ θ : ℝ, distance_from_P_to_line_l θ = sqrt 3 * (2 - sqrt 2) / 2 :=
sorry

end min_distance_from_P_to_line_l_l386_386779


namespace compute_x_squared_y_plus_xy_squared_l386_386718

theorem compute_x_squared_y_plus_xy_squared 
  (x y : ℝ)
  (h1 : (1 / x) + (1 / y) = 4)
  (h2 : x * y + x + y = 7) :
  x^2 * y + x * y^2 = 49 := 
  sorry

end compute_x_squared_y_plus_xy_squared_l386_386718


namespace max_marks_l386_386157

theorem max_marks (score shortfall passing_threshold : ℝ) (h1 : score = 212) (h2 : shortfall = 19) (h3 : passing_threshold = 0.30) :
  ∃ M, M = 770 :=
by
  sorry

end max_marks_l386_386157


namespace ratio_of_annes_potatoes_to_toms_potatoes_l386_386801

-- Definitions for conditions
def initial_potatoes : ℕ := 300
def ginas_potatoes : ℕ := 69
def toms_potatoes : ℕ := 2 * ginas_potatoes
def remaining_potatoes : ℕ := 47
def total_given_away : ℕ := initial_potatoes - remaining_potatoes
def annes_potatoes : ℕ := total_given_away - (ginas_potatoes + toms_potatoes)

-- Theorem statement
theorem ratio_of_annes_potatoes_to_toms_potatoes :
  (annes_potatoes.to_rat / toms_potatoes.to_rat) = (1 : ℚ) / (3 : ℚ) :=
sorry

end ratio_of_annes_potatoes_to_toms_potatoes_l386_386801


namespace money_depreciation_problem_l386_386864

theorem money_depreciation_problem (log3 : ℝ) (h_log3 : log3 = 0.477) : 
  ∃ (n : ℕ), (0.9)^n ≤ 0.1 ∧ ∀ m : ℕ, m < n → (0.9)^m > 0.1 :=
by
  use 22
  sorry

end money_depreciation_problem_l386_386864


namespace math_proof_problem_l386_386564

-- Define constants
def x := 2000000000000
def y := 1111111111111

-- Prove the main statement
theorem math_proof_problem :
  2 * (x - y) = 1777777777778 := 
  by
    sorry

end math_proof_problem_l386_386564


namespace natural_values_sum_l386_386560

theorem natural_values_sum :
  let n_valid (n : ℕ) := (∀ k : ℕ, (n = 2 + 9 * k) → 
                                  6 < Math.log 2 ((n : ℝ)) < 7)
  ∧ (cos (2 * π / 9 : ℝ) + cos (4 * π / 9 : ℝ) + ... + cos ((2 * n) * π / 9 : ℝ) = 
      cos (π / 9 : ℝ))
  let solutions := Set.filter n_valid (Set.range (λ m, 2 + 9 * m)) in
  Set.sum solutions = 644 :=
sorry

end natural_values_sum_l386_386560


namespace value_of_y_in_arithmetic_sequence_l386_386536

theorem value_of_y_in_arithmetic_sequence :
    ∃ y : ℤ, (arithmetic_sequence (3^2) y (3^4)) ∧ y = 45 := by
  -- Here we define the arithmetic sequence condition.
  def arithmetic_sequence (a b c : ℤ) : Prop := b = (a + c) / 2
  sorry

end value_of_y_in_arithmetic_sequence_l386_386536


namespace paige_scored_17_points_l386_386417

def paige_points (total_points : ℕ) (num_players : ℕ) (points_per_player_exclusive : ℕ) : ℕ :=
  total_points - ((num_players - 1) * points_per_player_exclusive)

theorem paige_scored_17_points :
  paige_points 41 5 6 = 17 :=
by
  sorry

end paige_scored_17_points_l386_386417


namespace constant_term_in_expansion_constant_term_is_60_l386_386513

theorem constant_term_in_expansion :
  (2 * x^3 + x^2 + 6) * (4 * x^4 + 3 * x^3 + 10) = 2 * 4 * x^7 + 2 * 3 * x^6 + 2 * 10 * x^3 + x^2 * 4 * x^4 + x^2 * 3 * x^3 + x^2 * 10 + 6 * 4 * x^4 + 6 * 3 * x^3 + 6 * 10 :=
sorry

theorem constant_term_is_60 :
  constant_term_in_expansion → constant_term_in_expansion = 60 :=
sorry

end constant_term_in_expansion_constant_term_is_60_l386_386513


namespace simplify_expression_l386_386097

theorem simplify_expression : 4 * (2 - Complex.i) + 2 * Complex.i * (3 - 2 * Complex.i) = 12 + 2 * Complex.i := 
by
  sorry

end simplify_expression_l386_386097


namespace buckets_taken_away_l386_386836

def bucket_size : ℕ := 120
def number_of_buckets_to_fill_tub : ℕ := 14
def weekly_water_usage : ℕ := 9240
def days_in_week : ℕ := 7

theorem buckets_taken_away :
  let total_water := bucket_size * number_of_buckets_to_fill_tub in
  let daily_water_usage := weekly_water_usage / days_in_week in
  let water_to_remove := total_water - daily_water_usage in
  water_to_remove / bucket_size = 3 :=
begin
  sorry
end

end buckets_taken_away_l386_386836


namespace sum_proper_divisors_243_l386_386927

theorem sum_proper_divisors_243 : 
  let proper_divisors_243 := [1, 3, 9, 27, 81] in
  proper_divisors_243.sum = 121 := 
by
  sorry

end sum_proper_divisors_243_l386_386927


namespace bugs_meet_again_l386_386909

-- Define the constants for radii and speeds
def r1 : ℝ := 7
def r2 : ℝ := 3
def s1 : ℝ := 4 * real.pi
def s2 : ℝ := 3 * real.pi

-- Define the circumferences of the circles
def C1 : ℝ := 2 * r1 * real.pi
def C2 : ℝ := 2 * r2 * real.pi

-- Define the times to complete one full rotation
def t1 : ℝ := C1 / s1
def t2 : ℝ := C2 / s2

-- Define the least common multiple of the rotation times
def lcm_t1_t2 : ℝ := real.lcm (int.of_real t1) (int.of_real t2)

theorem bugs_meet_again : lcm_t1_t2 = 7 := by
  -- We can provide the proof in this section
  sorry

end bugs_meet_again_l386_386909


namespace vertices_consecutive_l386_386073

noncomputable theory

-- Define a regular n-gon in Lean
structure RegularNGon (n : ℕ) (h : n ≥ 3) where
  -- Representation can be up to user decision, placeholders here
  vertices : list (ℝ × ℝ)
  regular : ∀ i : fin n, dist (vertices.nth_le i sorry) (vertices.nth_le ((i + 1) % n) sorry) = 1 -- Assuming unit distance uniformly, for simplicity
  
def PolygonInside {n : ℕ} (h : n ≥ 3) (A B : RegularNGon n h) : Prop :=
  ∀ v ∈ A.vertices, ∃ w ∈ B.vertices, dist v w ≤ 1 -- Placeholder distance definition and inclusion condition

theorem vertices_consecutive {n : ℕ} (h : n ≥ 3) (A B : RegularNGon n h) :
  PolygonInside h A B → (∃ l : (ℝ × ℝ) → bool, ∀ v ∈ A.vertices, l v = tt → ∃ w ∈ B.vertices, dist v w ≤ 1) :=
sorry

end vertices_consecutive_l386_386073


namespace primes_quadratic_roots_conditions_l386_386074

theorem primes_quadratic_roots_conditions (p q : ℕ)
  (hp : Prime p) (hq : Prime q)
  (h1 : ∃ (x y : ℕ), x ≠ y ∧ x * y = 2 * q ∧ x + y = p) :
  (¬ (∀ (x y : ℕ), x ≠ y ∧ x * y = 2 * q ∧ x + y = p → (x - y) % 2 = 0)) ∧
  (∃ (x : ℕ), x * 2 = 2 * q ∨ x * q = 2 * q ∧ Prime x) ∧
  (¬ Prime (p * p + 2 * q)) ∧
  (Prime (p - q)) :=
by sorry

end primes_quadratic_roots_conditions_l386_386074


namespace decimal_fraction_denominator_l386_386102

noncomputable def S := 0.666666... -- Representing 0.\overline{6}
def fraction := 2 / 3 -- The simplest form of the repeating decimal fraction

theorem decimal_fraction_denominator :
  (S = fraction) → (2/3).denom = 3 :=
by
  -- We can assume S is equivalent to 2/3 and then show the denominator directly
  intro h
  rw h
  exact rfl

end decimal_fraction_denominator_l386_386102


namespace chord_le_diameter_l386_386088

theorem chord_le_diameter (O : Point) (R : ℝ) (A B : Point) (hA : dist O A = R) (hB : dist O B = R) : 
  dist A B ≤ 2 * R ∧ (dist A B = 2 * R ↔ O ∈ segment A B) :=
by
  sorry

end chord_le_diameter_l386_386088


namespace product_of_numbers_is_178_5_l386_386893

variables (a b c d : ℚ)

def sum_eq_36 := a + b + c + d = 36
def first_num_cond := a = 3 * (b + c + d)
def second_num_cond := b = 5 * c
def fourth_num_cond := d = (1 / 2) * c

theorem product_of_numbers_is_178_5 (h1 : sum_eq_36 a b c d)
  (h2 : first_num_cond a b c d) (h3 : second_num_cond b c) (h4 : fourth_num_cond d c) :
  a * b * c * d = 178.5 :=
by
  sorry

end product_of_numbers_is_178_5_l386_386893


namespace fruit_picking_proof_l386_386202

noncomputable def total_fruits_and_ratio (pear_trees : ℕ) (apple_trees : ℕ)
    (k_pears_per_tree : ℕ) (k_apples_per_tree : ℕ)
    (j_pears_per_tree : ℕ) (j_apples_per_tree : ℕ)
    (jo_apples_per_tree : ℕ) 
    (half : ℝ → ℕ) :=
  let keith_pears := k_pears_per_tree * pear_trees
  let keith_apples := k_apples_per_tree * apple_trees
  let jason_pears := j_pears_per_tree * pear_trees
  let jason_apples := j_apples_per_tree * apple_trees
  let joan_apples := jo_apples_per_tree * apple_trees
  let joan_pears := half joan_apples.to_ℝ
  let total_pears := keith_pears + jason_pears + joan_pears
  let total_apples := keith_apples + jason_apples + joan_apples
  let total_fruits := total_pears + total_apples
  (total_fruits, total_apples, total_pears, total_apples.gcd total_pears)

theorem fruit_picking_proof :
  let (total, apples, pears, g) :=
    total_fruits_and_ratio 4 3 2 3 3 2 4 (λ x => (x / 2).to_nat) in
  total = 53 ∧ apples = 27 ∧ pears = 26 ∧ apples / g = 27 ∧ pears / g = 26 :=
by
  sorry

end fruit_picking_proof_l386_386202


namespace find_g2_of_polynomial_product_l386_386863

noncomputable def binom (n k : ℕ) : ℕ := Nat.choose n k

def g (x : ℝ) : ℝ := ∑ k in Finset.range 11, (binom (k + 11) k * x^(20 - k) - binom (21 - k) 11 * x^(k - 1) + binom 21 11 * x^(k - 1))
def h (x : ℝ) : ℝ := ∑ k in Finset.range 11, (binom (k + 11) k * x^(20 - k) - binom (21 - k) 11 * x^(k - 1) + binom 21 11 * x^(k - 1))

theorem find_g2_of_polynomial_product :
  ∃ (g h : ℝ → ℝ), (∀ x ≠ 0, g x * h x = ∑ k in Finset.range 11,
    (binom (k + 11) k * x^(20 - k) - binom (21 - k) 11 * x^(k - 1) + binom 21 11 * x^(k - 1)))
  ∧ g(2) < h(2)
  ∧ g(2) = 2047 :=
begin
  sorry
end

end find_g2_of_polynomial_product_l386_386863


namespace calculate_a2_b2_c2_l386_386766

theorem calculate_a2_b2_c2 (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a * b + a * c + b * c = -3) (h3 : a * b * c = 2) :
  a^2 + b^2 + c^2 = 6 :=
sorry

end calculate_a2_b2_c2_l386_386766


namespace total_fundamental_particles_l386_386603

def protons := 9
def neutrons := 19 - protons
def electrons := protons
def total_particles := protons + neutrons + electrons

theorem total_fundamental_particles : total_particles = 28 := by
  sorry

end total_fundamental_particles_l386_386603


namespace find_n_l386_386265

theorem find_n (n : ℕ) (h : n > 2016) (h_not_divisible : ¬ (1^n + 2^n + 3^n + 4^n) % 10 = 0) : n = 2020 :=
sorry

end find_n_l386_386265


namespace melissa_driving_hours_in_a_year_l386_386079

/-- Melissa drives to town twice each month, and each trip takes 3 hours.
    Prove that she spends 72 hours driving in a year. -/
theorem melissa_driving_hours_in_a_year 
  (trips_per_month : ℕ)
  (months_per_year : ℕ)
  (hours_per_trip : ℕ)
  (total_hours : ℕ)
  (h1 : trips_per_month = 2)
  (h2 : months_per_year = 12)
  (h3 : hours_per_trip = 3) :
  total_hours = trips_per_month * months_per_year * hours_per_trip :=
by {
  rw [h1, h2, h3],
  trivial,
}

end melissa_driving_hours_in_a_year_l386_386079


namespace find_y_find_x_l386_386784

-- Define vectors as per the conditions
def a : ℝ × ℝ := (3, -2)
def b (y : ℝ) : ℝ × ℝ := (-1, y)
def c (x : ℝ) : ℝ × ℝ := (x, 5)

-- Define the dot product of two vectors
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Define the condition for perpendicular vectors
def is_perpendicular (u v : ℝ × ℝ) : Prop :=
  dot_product u v = 0

-- Define the condition for parallel vectors
def is_parallel (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, u = (k * v.1, k * v.2)

-- Question 1 Proof Statement
theorem find_y : ∀ (y : ℝ), is_perpendicular a (b y) → y = 3 / 2 :=
by
  intros y h
  unfold is_perpendicular at h
  unfold dot_product at h
  sorry

-- Question 2 Proof Statement
theorem find_x : ∀ (x : ℝ), is_parallel a (c x) → x = 15 / 2 :=
by
  intros x h
  unfold is_parallel at h
  sorry

end find_y_find_x_l386_386784


namespace fraction_of_girls_on_trip_l386_386200

theorem fraction_of_girls_on_trip (b g : ℕ) (h : b = g) :
  ((2 / 3 * g) / (5 / 6 * b + 2 / 3 * g)) = 4 / 9 :=
by
  sorry

end fraction_of_girls_on_trip_l386_386200


namespace right_triangle_acute_angles_l386_386348

theorem right_triangle_acute_angles 
  (α β : ℝ) 
  (h₁ : α + β = 90) 
  (h₂ : ∃ (ε η : ℝ), ε = 90 - α / 2 ∧ η = 45 + α / 2 ∧ ε / η = 13 / 17) :
  α = 63 ∧ β = 27 :=
begin
  sorry
end

end right_triangle_acute_angles_l386_386348


namespace increasing_function_range_a_l386_386747

def f (a x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 + 4 * a * x else (2 * a + 3) * x - 4 * a + 5

theorem increasing_function_range_a :
  (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) ↔ (1/2 : ℝ) ≤ a ∧ a ≤ (3/2 : ℝ) :=
by
  sorry

end increasing_function_range_a_l386_386747


namespace breaks_difference_l386_386047

-- James works for 240 minutes
def total_work_time : ℕ := 240

-- He takes a water break every 20 minutes
def water_break_interval : ℕ := 20

-- He takes a sitting break every 120 minutes
def sitting_break_interval : ℕ := 120

-- Calculate the number of water breaks James takes
def number_of_water_breaks : ℕ := total_work_time / water_break_interval

-- Calculate the number of sitting breaks James takes
def number_of_sitting_breaks : ℕ := total_work_time / sitting_break_interval

-- Prove the difference between the number of water breaks and sitting breaks is 10
theorem breaks_difference :
  number_of_water_breaks - number_of_sitting_breaks = 10 :=
by
  -- calculate number_of_water_breaks = 12
  -- calculate number_of_sitting_breaks = 2
  -- check the difference 12 - 2 = 10
  sorry

end breaks_difference_l386_386047


namespace maximum_value_of_d_l386_386811

theorem maximum_value_of_d (a b c d : ℝ) 
  (h₁ : a + b + c + d = 10)
  (h₂ : ab + ac + ad + bc + bd + cd = 20) :
  d ≤ (5 + Real.sqrt 105) / 2 :=
sorry

end maximum_value_of_d_l386_386811


namespace sum_of_digits_product_is_13_l386_386613

def base_eight_to_base_ten (n : ℕ) : ℕ := sorry
def product_base_eight (n1 n2 : ℕ) : ℕ := sorry
def digits_sum_base_ten (n : ℕ) : ℕ := sorry

theorem sum_of_digits_product_is_13 :
  let N1 := base_eight_to_base_ten 35
  let N2 := base_eight_to_base_ten 42
  let product := product_base_eight N1 N2
  digits_sum_base_ten product = 13 :=
by
  sorry

end sum_of_digits_product_is_13_l386_386613


namespace value_of_a_plus_b_l386_386717

theorem value_of_a_plus_b (a b : ℕ) (h1 : 2^a = 2) (h2 : b = 2) : a + b = 3 :=
by
  have ha : a = 1 := by sorry
  rw [ha, h2]
  exact (1 + 2 : ℕ)

end value_of_a_plus_b_l386_386717


namespace Suzanne_bread_loaves_needed_l386_386085

theorem Suzanne_bread_loaves_needed :
  (let slices_per_day := 1 + 1 + 0.5 + 0.5 in
   let slices_per_weekend := slices_per_day * 2 in
   let total_slices := slices_per_weekend * 52 in
   let slices_per_loaf := 12 in
   total_slices / slices_per_loaf = 26) :=
by
  let slices_per_day := 1 + 1 + 0.5 + 0.5
  let slices_per_weekend := slices_per_day * 2
  let total_slices := slices_per_weekend * 52
  let slices_per_loaf := 12
  show total_slices / slices_per_loaf = 26
  from sorry

end Suzanne_bread_loaves_needed_l386_386085


namespace circle_annulus_area_l386_386489

open Real

noncomputable def circle_area (r : ℝ) : ℝ := π * r^2

theorem circle_annulus_area (C : Point) (A D B : Point) 
  (h1 : ∃ O : Point, ∀ P ∈ {A,D,B,C}, dist P O = dist C O)
  (h2 : is_tangent A D B)
  (h3 : dist C A = 12)
  (h4 : dist A D = 20) : 
  circle_area 12 - circle_area (2 * sqrt 11) = 100 * π := by
sorry

end circle_annulus_area_l386_386489


namespace fifth_team_points_l386_386275

theorem fifth_team_points (points_A points_B points_C points_D points_E : ℕ) 
(hA : points_A = 1) 
(hB : points_B = 2) 
(hC : points_C = 5) 
(hD : points_D = 7) 
(h_sum : points_A + points_B + points_C + points_D + points_E = 20) : 
points_E = 5 := 
sorry

end fifth_team_points_l386_386275


namespace sum_n_values_l386_386559

theorem sum_n_values
  (n : ℕ) :
  (∃ n ∈ ℕ,
    (∑ k in range(n + 1), cos (2 * π * k / 9)) = cos (π / 9)
    ∧ (log 2 n)^2 + 45 < log 2 (8 * n^13)) →
  n = 644 :=
by
  sorry

end sum_n_values_l386_386559


namespace power_of_seven_l386_386933

theorem power_of_seven : 
  (7 : ℝ) ^ (1 / 4) / (7 ^ (1 / 7)) = (7 ^ (3 / 28)) :=
by
  sorry

end power_of_seven_l386_386933


namespace area_of_triangle_DEF_l386_386254
open Real

noncomputable def triangle_area (DE DF : ℝ) (angle_EDF : ℝ) :=
  if h_EDF : angle_EDF = π / 2 then 
    1 / 2 * DE * DF 
  else 0

theorem area_of_triangle_DEF : 
  triangle_area 5 5 (π / 2) = 25 / 2 :=
by
  apply if_pos
  -- this is sufficient to show the statement is valid, the actual proof can replace the sorry
  simp
  linarith

end area_of_triangle_DEF_l386_386254


namespace count_coprimes_15_l386_386657

def count_coprimes_less_than (n : ℕ) : ℕ :=
  (Finset.range n).filter (λ a => Nat.gcd a n = 1).card

theorem count_coprimes_15 :
  count_coprimes_less_than 15 = 8 :=
by
  sorry

end count_coprimes_15_l386_386657


namespace rhombus_perimeter_l386_386460

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) : 
  let a := d1 / 2 in
  let b := d2 / 2 in
  let c := Real.sqrt (a^2 + b^2) in
  let side := c in
  let perimeter := 4 * side in
  perimeter = 52 := 
by 
  let a := 5 in 
  let b := 12 in 
  have h3 : a = d1 / 2, by rw [h1]; norm_num
  have h4 : b = d2 / 2, by rw [h2]; norm_num
  let c := Real.sqrt (5^2 + 12^2),
  let side := c,
  have h5 : c = 13, by norm_num,
  let perimeter := 4 * 13,
  show perimeter = 52, by norm_num; sorry

end rhombus_perimeter_l386_386460


namespace income_percentage_less_l386_386406

-- Definitions representing the conditions
variables (T M J : ℝ)
variables (h1 : M = 1.60 * T) (h2 : M = 1.12 * J)

-- The theorem stating the problem
theorem income_percentage_less : (100 - (T / J) * 100) = 30 :=
by
  sorry

end income_percentage_less_l386_386406


namespace num_150_ray_not_90_or_75_l386_386062

-- Define the unit square region
def unit_square : set (ℝ × ℝ) := 
  { p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1 }

-- Define n-ray partitional property
def n_ray_partitional (n : ℕ) (X : ℝ × ℝ) : Prop := 
  X ∈ unit_square ∧ n ≥ 4 ∧
  ∃ f : ℝ × ℝ → ℝ, (∀ i : ℕ, 1 ≤ i ∧ i ≤ n → 
  area (polygon i (X)) = (1 / n))

-- Predicate for specific n-ray partitionals
def is_150_ray (X : ℝ × ℝ) : Prop := n_ray_partitional 150 X
def is_90_ray (X : ℝ × ℝ) : Prop := n_ray_partitional 90 X
def is_75_ray (X : ℝ × ℝ) : Prop := n_ray_partitional 75 X

-- Define the goal statement
theorem num_150_ray_not_90_or_75 : 
  {X | is_150_ray X ∧ ¬is_90_ray X ∧ ¬is_75_ray X}.card = 0 :=
  by
    sorry

end num_150_ray_not_90_or_75_l386_386062


namespace arithmetic_sequence_y_value_l386_386517

theorem arithmetic_sequence_y_value :
  ∃ y : ℤ, (∃ a1 a3 : ℤ, a1 = 9 ∧ a3 = 81 ∧ y = (a1 + a3) / 2) → y = 45 :=
by
  sorry

end arithmetic_sequence_y_value_l386_386517


namespace problem_statement_l386_386666

theorem problem_statement :
  (∃ n : ℕ, n = 8 ∧ ∀ a : ℕ, a < 15 → (∃ x : ℤ, a * x ≡ 1 [MOD 15]) ↔ gcd a 15 = 1) :=
by
  use 8
  intro a
  intro ha
  split
  sorry

end problem_statement_l386_386666


namespace curve_C1_eq_max_AB_l386_386029

variable (t k x y : ℝ)

def l1 := (4 - t, k * t)
def l2 := (λ x : ℝ, (1/k) * x)
def l3 := (λ θ : ℝ, 2 * (sin (θ - Real.pi / 4)))

theorem curve_C1_eq : (x:ℝ ) (y:ℝ ), x^2 + y^2 - 4 * x = 0 := by
  sorry

theorem max_AB : ∃ (A B : ℝ×ℝ ), ∀ θ : ℝ, (A ∈ l3 θ) ∧ (B ∈ curve_C1_eq) ∧ ((angle_between A B = Real.pi / 4) → | dist A B | = 2 * sqrt 2 + 2) := by
  sorry

end curve_C1_eq_max_AB_l386_386029


namespace roots_difference_l386_386300

noncomputable def poly1 : polynomial ℝ := 2002^2 * polynomial.X^2 - 2003 * 2001 * polynomial.X - 1
noncomputable def poly2 : polynomial ℝ := 2001 * polynomial.X^2 - 2002 * polynomial.X + 1

axiom r : ℝ
axiom s : ℝ
axiom r_root : r = (root (polynomial.has_derivative.to_fun poly1) 1)
axiom s_root : s = (root (polynomial.has_derivative.to_fun poly2) 2)

theorem roots_difference : r - s = 2000 / 2001 :=
by
  sorry

end roots_difference_l386_386300


namespace sum_real_solutions_l386_386272

theorem sum_real_solutions (b : ℝ) (hb : b > 2) : 
  (∑ y in {y : ℝ | sqrt (b - sqrt (b + y)) = y + 1}, y) = sqrt (4 * b - 4) / 2 :=
by
  sorry

end sum_real_solutions_l386_386272


namespace triangle_ABC_area_l386_386032

open Real

noncomputable def triangle_area : ℝ :=
  let α := π / 4
  let BE := 8
  let b := BE / sqrt 2
  let a := b / 2
  (1 / 2) * a * b

theorem triangle_ABC_area :
  AB = BC →
  ∃ D, altitude BD →
  ∃ E, E ∈ line AC ∧ BE = 8 →
  (sin (angle CBE) * sin (angle ABE) * sin (angle DBE)) = (sin (α - β) * sin (α + β) * sin α) → -- implicitly geometric progression
  (tan (angle DBE), tan (angle CBE), tan (angle DBC)) = (tan α, tan (α - β), tan β) → -- implicitly arithmetic progression
  area_of_triangle ABC = 8 :=
by
  sorry

end triangle_ABC_area_l386_386032


namespace k_h_neg3_l386_386067

-- Definitions of h and k
def h (x : ℝ) : ℝ := 4 * x^2 - 12

variable (k : ℝ → ℝ) -- function k with range an ℝ

-- Given k(h(3)) = 16
axiom k_h_3 : k (h 3) = 16

-- Prove that k(h(-3)) = 16
theorem k_h_neg3 : k (h (-3)) = 16 :=
sorry

end k_h_neg3_l386_386067


namespace investment_time_l386_386590

variable (P : ℝ) (R : ℝ) (SI : ℝ)

theorem investment_time (hP : P = 800) (hR : R = 0.04) (hSI : SI = 160) :
  SI / (P * R) = 5 := by
  sorry

end investment_time_l386_386590


namespace selection_count_l386_386957

def choose (n k : ℕ) : ℕ := -- Binomial coefficient definition
  if h : 0 ≤ k ∧ k ≤ n then
    Nat.choose n k
  else
    0

theorem selection_count : choose 9 5 - choose 6 5 = 120 := by
  sorry

end selection_count_l386_386957


namespace range_of_g_l386_386688

noncomputable def g (x : ℝ) : ℝ :=
  (Real.arcsin (x / 3))^2 - Real.pi * Real.arccos (x / 3) + 
  (Real.arccos (x / 3))^2 + (Real.pi^2 / 18) * (x^2 - 3 * x + 9)

theorem range_of_g :
  ∀ x, -3 ≤ x ∧ x ≤ 3 →
    (g x) ∈ Icc ((2 * (Real.pi^2)) / 3) ((5 * (Real.pi^2)) / 6) :=
by
  sorry

end range_of_g_l386_386688


namespace count_shapes_on_grid_l386_386324

/-- The number of rectangles and right-angled triangles whose vertices are points on a 4x3 grid is 30. -/
theorem count_shapes_on_grid : 
  let rows := 4
      cols := 3 in
  (rows * (rows - 1) / 2) * (cols * (cols - 1) / 2) + 2 * (rows - 1) * (cols - 1) = 30 := sorry

end count_shapes_on_grid_l386_386324


namespace fractional_part_sum_eq_one_l386_386302

/-- Given that the real number x satisfies x^3 + 1/x^3 = 18, prove that the sum of the fractional parts of x and 1/x is 1. -/
theorem fractional_part_sum_eq_one (x : ℝ) (hx : x^3 + (1/x)^3 = 18) : 
  fractionalPart x + fractionalPart (1/x) = 1 :=
sorry

end fractional_part_sum_eq_one_l386_386302


namespace remaining_oak_trees_l386_386494

def initial_oak_trees : ℕ := 9
def cut_down_oak_trees : ℕ := 2

theorem remaining_oak_trees : initial_oak_trees - cut_down_oak_trees = 7 := 
by 
  sorry

end remaining_oak_trees_l386_386494


namespace parallel_vectors_have_given_scalar_relation_l386_386283

theorem parallel_vectors_have_given_scalar_relation
  (a : ℝ × ℝ) (b : ℝ × ℝ)
  (h₁ : a = (2, 3))
  (h₂ : b = (x, -6))
  (h₃ : (2 * a.1, 2 * a.2) ∥ b) :
  x = -4 :=
by
  sorry

end parallel_vectors_have_given_scalar_relation_l386_386283


namespace triangle_AB_C_min_perimeter_l386_386376

noncomputable def minimum_perimeter (a b c : ℕ) (A B C : ℝ) : ℝ := a + b + c

theorem triangle_AB_C_min_perimeter
  (a b c : ℕ)
  (A B C : ℝ)
  (h1 : A = 2 * B)
  (h2 : C > π / 2)
  (h3 : a^2 = b * (b + c))
  (h4 : ∀ x : ℕ, x > 0 → a ≠ 0)
  (h5 :  a + b > c ∧ a + c > b ∧ b + c > a) :
  minimum_perimeter a b c A B C = 77 := 
sorry

end triangle_AB_C_min_perimeter_l386_386376


namespace truck_travel_minimizes_fuel_use_l386_386193

variable (L a : ℕ)
noncomputable def distance := (23 * a) / 15

theorem truck_travel_minimizes_fuel_use 
  (L : ℕ) 
  (a : ℕ) 
  (can_carry : ∀ L a, L ≥ a)
  (fuel_strategy : ∀ a L, a + ((23 * a) / 15) = (23 * a) / 15)
  (no_stations : ∀ a d, d = (23 * a) / 15) 
  (one_truck : ∀ trucks, trucks = 1) : 
  ∃ (fuel_used : ℕ), fuel_used = 3 * L :=
sorry

end truck_travel_minimizes_fuel_use_l386_386193


namespace three_digit_number_divisible_by_7_l386_386765

theorem three_digit_number_divisible_by_7
  (a b : ℕ)
  (h1 : (a + b) % 7 = 0) :
  (100 * a + 10 * b + a) % 7 = 0 :=
sorry

end three_digit_number_divisible_by_7_l386_386765


namespace log_four_sixtyfour_sqrt_two_l386_386234

theorem log_four_sixtyfour_sqrt_two :
  log 4 (64 * Real.sqrt 2) = 13 / 4 := by
sorry

end log_four_sixtyfour_sqrt_two_l386_386234


namespace correct_equation_after_moving_digit_l386_386355

theorem correct_equation_after_moving_digit :
  (101 - 102 = 1) →
  101 - 10^2 = 1 :=
by
  intro h
  sorry

end correct_equation_after_moving_digit_l386_386355


namespace decimal_sum_of_fraction_1_over_909_l386_386546

theorem decimal_sum_of_fraction_1_over_909 :
  (let decimal_digits : List ℕ := [0, 0, 1, 1].cycle.take 30 in decimal_digits.sum = 14) := by
sorry

end decimal_sum_of_fraction_1_over_909_l386_386546


namespace problem_statement_l386_386670

theorem problem_statement :
  (∃ n : ℕ, n = 8 ∧ ∀ a : ℕ, a < 15 → (∃ x : ℤ, a * x ≡ 1 [MOD 15]) ↔ gcd a 15 = 1) :=
by
  use 8
  intro a
  intro ha
  split
  sorry

end problem_statement_l386_386670


namespace count_coprimes_15_l386_386656

def count_coprimes_less_than (n : ℕ) : ℕ :=
  (Finset.range n).filter (λ a => Nat.gcd a n = 1).card

theorem count_coprimes_15 :
  count_coprimes_less_than 15 = 8 :=
by
  sorry

end count_coprimes_15_l386_386656


namespace arithmetic_sequence_middle_term_l386_386526

theorem arithmetic_sequence_middle_term :
  let a1 := 3^2
  let a3 := 3^4
  let y := (a1 + a3) / 2
  y = 45 :=
by
  let a1 := (3:ℕ)^2
  let a3 := (3:ℕ)^4
  let y := (a1 + a3) / 2
  have : a1 = 9 := by norm_num
  have : a3 = 81 := by norm_num
  have : y = 45 := by norm_num
  exact this

end arithmetic_sequence_middle_term_l386_386526


namespace average_running_time_correct_l386_386998

variable (e : ℕ)  -- Number of eighth graders

-- Running times per day for each grade
def sixth_grader_running_time := 14 * 12 * e
def seventh_grader_running_time := 18 * 4 * e
def eighth_grader_running_time := 12 * e

-- Adjusted running times including the sports day
def sixth_grader_running_time_with_sports_day := (14 + 4 / 5) * 12 * e
def seventh_grader_running_time_with_sports_day := (18 + 4 / 5) * 4 * e
def eighth_grader_running_time_with_sports_day := (12 + 4 / 5) * e

-- Total running time with sports day included
def total_running_time_with_sports_day := 
  sixth_grader_running_time_with_sports_day e + 
  seventh_grader_running_time_with_sports_day e + 
  eighth_grader_running_time_with_sports_day e

-- Total number of students
def total_number_of_students := 12 * e + 4 * e + e

-- Average running time per student
def average_running_time_with_sports_day := total_running_time_with_sports_day e / total_number_of_students e

theorem average_running_time_correct :
  average_running_time_with_sports_day e = 15.6 := by
  sorry

end average_running_time_correct_l386_386998


namespace sum_proper_divisors_243_l386_386929

theorem sum_proper_divisors_243 : 
  let proper_divisors_243 := [1, 3, 9, 27, 81] in
  proper_divisors_243.sum = 121 := 
by
  sorry

end sum_proper_divisors_243_l386_386929


namespace number_of_pupils_not_in_programX_is_639_l386_386015

-- Definitions for the conditions
def total_girls_elementary : ℕ := 192
def total_boys_elementary : ℕ := 135
def total_girls_middle : ℕ := 233
def total_boys_middle : ℕ := 163
def total_girls_high : ℕ := 117
def total_boys_high : ℕ := 89

def programX_girls_elementary : ℕ := 48
def programX_boys_elementary : ℕ := 28
def programX_girls_middle : ℕ := 98
def programX_boys_middle : ℕ := 51
def programX_girls_high : ℕ := 40
def programX_boys_high : ℕ := 25

-- Question formulation
theorem number_of_pupils_not_in_programX_is_639 :
  (total_girls_elementary - programX_girls_elementary) +
  (total_boys_elementary - programX_boys_elementary) +
  (total_girls_middle - programX_girls_middle) +
  (total_boys_middle - programX_boys_middle) +
  (total_girls_high - programX_girls_high) +
  (total_boys_high - programX_boys_high) = 639 := 
  by
  sorry

end number_of_pupils_not_in_programX_is_639_l386_386015


namespace root_pow_simplify_l386_386609

theorem root_pow_simplify :
  (real.rpow (real.sqrt 5) 5)^(8/4) = 3125 :=
by 
  sorry

end root_pow_simplify_l386_386609


namespace tan_75_eq_2_plus_sqrt_3_l386_386210

theorem tan_75_eq_2_plus_sqrt_3 :
  Real.tan (75 * Real.pi / 180) = 2 + Real.sqrt 3 := by
  sorry

end tan_75_eq_2_plus_sqrt_3_l386_386210


namespace average_weight_of_section_A_l386_386493

theorem average_weight_of_section_A (nA nB : ℕ) (WB WC : ℝ) (WA : ℝ) :
  nA = 50 →
  nB = 40 →
  WB = 70 →
  WC = 58.89 →
  50 * WA + 40 * WB = 58.89 * 90 →
  WA = 50.002 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end average_weight_of_section_A_l386_386493


namespace dead_grass_area_l386_386584

-- Define the constants for the problem
def circular_path_radius : ℝ := 5
def sombrero_radius : ℝ := 3

-- Define the formula for the area of the annulus
def area_of_annulus (R_outer R_inner : ℝ) : ℝ := 
  π * (R_outer ^ 2) - π * (R_inner ^ 2)

-- Instantiate the problem-specific radii
def outer_radius := circular_path_radius + sombrero_radius
def inner_radius := circular_path_radius - sombrero_radius

-- The Lean theorem statement asserting the area of dead grass is 60π square feet
theorem dead_grass_area : area_of_annulus outer_radius inner_radius = 60 * π :=
by
  -- Proof is omitted
  sorry

end dead_grass_area_l386_386584


namespace subset_implies_element_l386_386316

theorem subset_implies_element {A B : set ℝ} (m : ℝ) 
  (hA : A = {1, 3})
  (hB : B = {1, 2, m}) 
  (h_subset : A ⊆ B) : 
  m = 3 :=
by
  sorry

end subset_implies_element_l386_386316


namespace find_quadruples_l386_386681

theorem find_quadruples (a b p n : ℕ) (h_prime : Prime p) (h_eq : a^3 + b^3 = p^n) :
  ∃ k : ℕ, (a, b, p, n) = (2^k, 2^k, 2, 3*k + 1) ∨ 
           (a, b, p, n) = (3^k, 2 * 3^k, 3, 3*k + 2) ∨ 
           (a, b, p, n) = (2 * 3^k, 3^k, 3, 3*k + 2) :=
sorry

end find_quadruples_l386_386681


namespace logistics_center_correct_l386_386781

noncomputable def rectilinear_distance (P Q : ℝ × ℝ) : ℝ :=
  |P.1 - Q.1| + |P.2 - Q.2|

def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (-6, 9)
def C : ℝ × ℝ := (-3, -8)

def logistics_center : ℝ × ℝ := (-5, 0)

theorem logistics_center_correct : 
  ∀ L : ℝ × ℝ, 
  (rectilinear_distance L A = rectilinear_distance L B) ∧ 
  (rectilinear_distance L B = rectilinear_distance L C) ∧
  (rectilinear_distance L A = rectilinear_distance L C) → 
  L = logistics_center := sorry

end logistics_center_correct_l386_386781


namespace exists_index_k_l386_386482

theorem exists_index_k (a₁ b₁ : ℕ) (h : a₁ > 0 ∧ b₁ > 0) :
  (∃ k, let (a_k, b_k) := 
    Nat.rec (λ _, ℕ × ℕ) (a₁, b₁) 
      (λ n (ab : ℕ × ℕ), match ab with | (a_n, b_n) :=
        if a_n ≥ b_n then (a_n - b_n, 2 * b_n)
        else (2 * a_n, b_n - a_n)
      end)
  in a_k = 0) ↔ ∃ m > 0, (a₁ + b₁) / Nat.gcd a₁ b₁ = 2^m :=
by
  sorry

end exists_index_k_l386_386482


namespace base3_sum_example_l386_386195

noncomputable def base3_add (a b : ℕ) : ℕ := sorry  -- Function to perform base-3 addition

theorem base3_sum_example : 
  base3_add (base3_add (base3_add (base3_add 2 120) 221) 1112) 1022 = 21201 := sorry

end base3_sum_example_l386_386195


namespace remainder_when_divided_by_100_l386_386398

-- Define the given m
def m : ℕ := 76^2006 - 76

-- State the theorem
theorem remainder_when_divided_by_100 : m % 100 = 0 :=
by
  sorry

end remainder_when_divided_by_100_l386_386398


namespace range_of_k_for_distinct_real_roots_l386_386761

theorem range_of_k_for_distinct_real_roots (k : ℝ) : 
  (∀ x : ℝ, (k - 1) * x^2 - 2 * x + 1 = 0) → (k < 2 ∧ k ≠ 1) :=
by
  sorry

end range_of_k_for_distinct_real_roots_l386_386761


namespace quadratic_function_properties_f_sin_x_extreme_values_l386_386705

noncomputable def f (x : ℝ) : ℝ := -(x - 1) * (x - 3)

theorem quadratic_function_properties :
  (f 0 = -3) ∧
  (∀ x, 1 < x ∧ x < 3 → f x > 0) ∧
  (∀ x, (x < 1 ∨ x > 3) → f x ≤ 0) ∧
  (∀ x, (f x = -(x - 2)^2 + 1)) :=
by
  split
  . exact sorry  -- Prove that f(0) = -3
  . split
    . intros x hx
      exact sorry  -- Prove that 1 < x < 3 implies f(x) > 0
    . intros x hx
      exact sorry  -- Prove that x < 1 or x > 3 implies f(x) ≤ 0
    . intros x
      exact sorry  -- Prove that f(x) = -(x - 2)^2 + 1

theorem f_sin_x_extreme_values :
  let g := λ x: ℝ, f (Real.sin x)
  ∃ (min_x max_x : ℝ), 
    (0 ≤ min_x ∧ min_x ≤ (Real.pi / 2)) ∧ 
    (0 ≤ max_x ∧ max_x ≤ (Real.pi / 2)) ∧ 
    (∀ x, 0 ≤ x ∧ x ≤ (Real.pi / 2) → g min_x ≤ g x ∧ g x ≤ g max_x) ∧ 
    (g min_x = -3) ∧ (g max_x = 0) :=
by
  let g := λ x: ℝ, f (Real.sin x)
  use [0, Real.pi / 2]
  split
  . exact sorry  -- Prove that min_x and max_x within interval
  . split
    . intros x hx
      exact sorry  -- Prove that g is bounded within min_x and max_x
    . exact sorry  -- Prove that minimum value of g is -3
    . exact sorry  -- Prove that maximum value of g is 0

end quadratic_function_properties_f_sin_x_extreme_values_l386_386705
