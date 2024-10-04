import Data.Real.Basic
import Mathlib
import Mathlib.Algebra.Combinatorics.Basic
import Mathlib.Algebra.Factorial
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Group.WithOne
import Mathlib.Algebra.Polynomial
import Mathlib.Algebra.Probability
import Mathlib.Algebra.SpecialFunctions.Pow
import Mathlib.Analysis.Geometry.Manifold
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Finset.Powerset
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.List.Perm
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Circle
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Geometry.Euclidean.Space
import Mathlib.LinearAlgebra.FinSupp
import Mathlib.NumberTheory.CubicResidue
import Mathlib.Tactic
import Mathlib.Tactic.Linarith
import Real

namespace number_of_pieces_from_rod_l33_33770

theorem number_of_pieces_from_rod (rod_length_m : ℕ) (piece_length_cm : ℕ) (meter_to_cm : ℕ) 
  (h1 : rod_length_m = 34) (h2 : piece_length_cm = 85) (h3 : meter_to_cm = 100) : 
  rod_length_m * meter_to_cm / piece_length_cm = 40 := by
  sorry

end number_of_pieces_from_rod_l33_33770


namespace bug_final_position_l33_33498

theorem bug_final_position : 
  ∀ n : ℕ, n = 2023 →
  ∀ start : ℕ, start = 7 →
  ∀ points : Finset ℕ, points = {1, 2, 3, 4, 5, 6, 7} →
  ∀ move : ℕ → ℕ,
    (∀ k, k ∈ points → (k % 2 = 0 → move k = (k + 2) % 7 ∧ k % 2 ≠ 0 → move k = (k + 3) % 7)) →
  ∃ final : ℕ, final = 1 ∧ (∀ steps, steps = n → ∃ path : List ℕ, path.head = start ∧ path.last steps = final) :=
sorry

end bug_final_position_l33_33498


namespace horses_b_put_in_l33_33988

def total_cost : ℕ := 841
def a_horse_months : ℕ := 12 * 8
def c_horse_months : ℕ := 18 * 6
def b_payment : ℕ := 348

theorem horses_b_put_in (H : ℕ) (total_cost > 0) (a_horse_months = 96) (c_horse_months = 108) (b_payment = 348) :
  (348 * (204 + 9 * H) = 841 * 9 * H) → H = 16 :=
by sorry

end horses_b_put_in_l33_33988


namespace min_period_max_value_f_l33_33139

def f (x : ℝ) : ℝ := sin (x / 3) + cos (x / 3)

theorem min_period_max_value_f :
  ∃ T M, (∀ x, f (x + T) = f x) ∧ 0 < T ∧
    0 < M ∧ (∀ x, f x ≤ M) ∧ (∃ x, f x = M) ∧ 
    T = 6 * π ∧ M = √2 := sorry

end min_period_max_value_f_l33_33139


namespace probability_at_least_one_8_l33_33962

theorem probability_at_least_one_8 (n : ℕ) (hn : n = 8) : 
  (1 - (7/8) * (7/8)) = 15 / 64 :=
by
  rw [← hn]
  sorry

end probability_at_least_one_8_l33_33962


namespace probability_of_exactly_one_head_l33_33189

theorem probability_of_exactly_one_head (h1 h2 : Bool) :
  let outcomes := [(true, true), (true, false), (false, true), (false, false)] in
  let favorable := [(true, false), (false, true)] in
  2 / 4 = 1 / 2 :=
by
  sorry

end probability_of_exactly_one_head_l33_33189


namespace carrots_weight_l33_33819

theorem carrots_weight (carrots_bed1: ℕ) (carrots_bed2: ℕ) (carrots_bed3: ℕ) (carrots_per_pound: ℕ)
  (h_bed1: carrots_bed1 = 55)
  (h_bed2: carrots_bed2 = 101)
  (h_bed3: carrots_bed3 = 78)
  (h_c_per_p: carrots_per_pound = 6) :
  (carrots_bed1 + carrots_bed2 + carrots_bed3) / carrots_per_pound = 39 := by
  sorry

end carrots_weight_l33_33819


namespace factorize_ax2_minus_a_l33_33697

theorem factorize_ax2_minus_a (a x : ℝ) : a * x^2 - a = a * (x + 1) * (x - 1) :=
by
  sorry

end factorize_ax2_minus_a_l33_33697


namespace votes_cast_l33_33868

theorem votes_cast (A F T : ℕ) (h1 : A = 40 * T / 100) (h2 : F = A + 58) (h3 : T = F + A) : 
  T = 290 := 
by
  sorry

end votes_cast_l33_33868


namespace solution_l33_33398

noncomputable def problem (m : ℝ) : Prop :=
  (∃ (x y : ℝ), 3 * x + 4 * y + 25 = 0 ∧ 
   (let P := (x, y),
        A := (-m, 0 : ℝ),
        B := (m, 0 : ℝ),
        AP := (x + m, y),
        BP := (x - m, y)
    in (AP.1 * BP.1 + AP.2 * BP.2 = 0)) ∧ m > 0) → m ≥ 5

theorem solution (m : ℝ) : problem m := 
begin
  sorry
end

end solution_l33_33398


namespace shaded_region_area_l33_33291

noncomputable def area_of_shaded_region : ℝ :=
  let A := (-6, 10)
  let E := (22, 10)
  let Base1 := (−6, 0)
  let Base2 := (22, 0)
  let height := 10
  (1/2) * ((abs (Base2.1 - Base1.1)) + (abs (Base2.1 - Base1.1))) * height

theorem shaded_region_area :
  (∃ (A B Base1 Base2 : ℝ × ℝ)
      (height : ℝ),
      A = (-6, 10) ∧ E = (22, 10) ∧ Base1 = (-6, 0) ∧ Base2 = (22, 0) ∧ height = 10) →
  area_of_shaded_region = 280 :=
begin
  let A := (-6, 10),
  let E := (22, 10),
  let Base1 := (-6, 0),
  let Base2 := (22, 0),
  let height := 10,
  let area := (1/2) * ((abs (Base2.1 - Base1.1)) + (abs (Base2.1 - Base1.1))) * height,
  exact sorry,
end

end shaded_region_area_l33_33291


namespace correct_equation_for_gift_exchanges_l33_33274

theorem correct_equation_for_gift_exchanges
  (x : ℕ)
  (H : (x * (x - 1)) = 56) :
  x * (x - 1) = 56 := 
by 
  exact H

end correct_equation_for_gift_exchanges_l33_33274


namespace pole_length_after_cut_l33_33648

theorem pole_length_after_cut (original_length : ℝ) (percentage_retained : ℝ) : 
  original_length = 20 → percentage_retained = 0.7 → 
  original_length * percentage_retained = 14 :=
by
  intros h0 h1
  rw [h0, h1]
  norm_num

end pole_length_after_cut_l33_33648


namespace min_positive_period_and_max_value_of_f_l33_33092

noncomputable def f (x : ℝ) : ℝ := sin (x / 3) + cos (x / 3)

theorem min_positive_period_and_max_value_of_f :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ T = 6 * π) ∧ 
  (∃ x, f x = sqrt 2) :=
by {
  sorry
}

end min_positive_period_and_max_value_of_f_l33_33092


namespace probability_two_spoons_one_knife_l33_33428

open Finset

theorem probability_two_spoons_one_knife (forks spoons knives total removed : ℕ) 
  (hf : forks = 4) (hs : spoons = 8) (hk : knives = 6) (ht : total = 18) 
  (hr : removed = 3) :
  ((choose spoons 2 * choose knives 1).to_rat / choose total removed).to_rat = (7 / 34) :=
by
  have h_total_eq : total = forks + spoons + knives := by
    rw [hf, hs, hk]
    norm_num
  have h_choose_total : choose total removed = choose 18 3 := by
    rw ht
  have h_choose_favored : choose spoons 2 * choose knives 1 = 28 * 6 := by
    rw [hs, hk]
    norm_num
  have h_total_ways : choose 18 3 = 816 := by
    norm_num
  have h_favored_outcomes : 28 * 6 = 168 := by
    norm_num
  have probability_eq : (168 : ℚ) / 816 = 7 / 34 := by
    norm_num
  sorry

end probability_two_spoons_one_knife_l33_33428


namespace greatest_x_l33_33040

def is_power_of_prime (x : ℕ) : Prop :=
  ∃ (p : ℕ) (k : ℕ), prime p ∧ x = p ^ k

theorem greatest_x (x : ℕ) : 
  (lcm x (lcm 15 21) = 105) ∧ is_power_of_prime x → x = 7 :=
by
  sorry

end greatest_x_l33_33040


namespace volume_of_sphere_l33_33243

open Real

noncomputable def sphereVolume : Real :=
  let r := sqrt 2
  (4 / 3) * π * (r ^ 3)

theorem volume_of_sphere :
  ∃ (s : Sphere),
    let (R : ℝ) := sqrt 2 in
    let (V : ℝ) := (4 / 3) * π * (R ^ 3) in
    (dist center plane = 1 ∧
    intersection_area plane sphere = π) →
    V = (8 * sqrt 2 / 3) * π :=
by
  sorry

end volume_of_sphere_l33_33243


namespace r_values_if_polynomial_divisible_l33_33716

noncomputable
def find_r_iff_divisible (r : ℝ) : Prop :=
  (10 * (r^2 * (1 - 2*r))) = -6 ∧ 
  (2 * r + (1 - 2*r)) = 1 ∧ 
  (r^2 + 2 * r * (1 - 2*r)) = -5.2

theorem r_values_if_polynomial_divisible (r : ℝ) :
  (find_r_iff_divisible r) ↔ 
  (r = (2 + Real.sqrt 30) / 5 ∨ r = (2 - Real.sqrt 30) / 5) := 
by
  sorry

end r_values_if_polynomial_divisible_l33_33716


namespace min_positive_period_and_max_value_l33_33084

def f (x : ℝ) := Real.sin (x / 3) + Real.cos (x / 3)

theorem min_positive_period_and_max_value : 
  (∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T' = T)) 
  ∧ 
  (∀ x, f x ≤ sqrt 2 ∧ (∃ x, f x = sqrt 2)) :=
sorry

end min_positive_period_and_max_value_l33_33084


namespace invitation_methods_l33_33254

-- Definitions
def num_ways_invite_6_out_of_10 : ℕ := Nat.choose 10 6
def num_ways_both_A_and_B : ℕ := Nat.choose 8 4

-- Theorem statement
theorem invitation_methods : num_ways_invite_6_out_of_10 - num_ways_both_A_and_B = 140 :=
by
  -- Proof should be provided here
  sorry

end invitation_methods_l33_33254


namespace attractions_visit_order_l33_33828

-- Define the conditions
def type_A_count : ℕ := 2
def type_B_count : ℕ := 4

-- Define the requirements that Javier visits type A attractions before type B
theorem attractions_visit_order : ∀ (A B : ℕ), 
  A = type_A_count → 
  B = type_B_count → 
  (fact A) * (fact B) = 48 :=
by
  intros A B hA hB
  rw [hA, hB]
  dsimp
  norm_num
  sorry

end attractions_visit_order_l33_33828


namespace train_length_proof_l33_33252

-- Define the conditions
def time_to_cross := 12 -- Time in seconds
def speed_km_per_h := 75 -- Speed in km/h

-- Convert the speed to m/s
def speed_m_per_s := speed_km_per_h * (5 / 18 : ℚ)

-- The length of the train using the formula: length = speed * time
def length_of_train := speed_m_per_s * (time_to_cross : ℚ)

-- The theorem to prove
theorem train_length_proof : length_of_train = 250 := by
  sorry

end train_length_proof_l33_33252


namespace card_placement_in_boxes_no_empty_boxes_l33_33560

theorem card_placement_in_boxes_no_empty_boxes : 
  (number_of_ways (cards : Finset ℕ) (boxes : Finset ℕ) (NoEmptyBoxes : ∀ b ∈ boxes, ∃ c ∈ cards, true) : ℕ) :=
  36 := by
  sorry

end card_placement_in_boxes_no_empty_boxes_l33_33560


namespace min_positive_period_f_max_value_f_l33_33065

noncomputable def f (x : ℝ) : ℝ := sin (x / 3) + cos (x / 3)

theorem min_positive_period_f : ∃ T > 0, (∀ x, f (x + T) = f x) ∧ ∀ T', (T' > 0 ∧ ∀ x, f (x + T') = f x) → T ≤ T' :=
  sorry

theorem max_value_f : ∃ M, (∀ x, f x ≤ M) ∧ (∀ ε > 0, ∃ x, M - ε < f x) ∧ M = sqrt 2 :=
  sorry

end min_positive_period_f_max_value_f_l33_33065


namespace domain_of_f_is_correct_l33_33027

def domain_of_f (x : ℝ) := (sqrt (x + 4) + sqrt (1 - x)) / x

theorem domain_of_f_is_correct : 
  {x : ℝ | (sqrt (x + 4) + sqrt (1 - x)) / x ≠ 0} = {x : ℝ | (-4 ≤ x ∧ x < 0) ∨ (0 < x ∧ x ≤ 1)} :=
by
  sorry

end domain_of_f_is_correct_l33_33027


namespace min_period_max_value_f_l33_33136

def f (x : ℝ) : ℝ := sin (x / 3) + cos (x / 3)

theorem min_period_max_value_f :
  ∃ T M, (∀ x, f (x + T) = f x) ∧ 0 < T ∧
    0 < M ∧ (∀ x, f x ≤ M) ∧ (∃ x, f x = M) ∧ 
    T = 6 * π ∧ M = √2 := sorry

end min_period_max_value_f_l33_33136


namespace min_pos_period_max_value_l33_33145

noncomputable def f (x : ℝ) := Real.sin (x / 3) + Real.cos (x / 3)

theorem min_pos_period_max_value :
  (∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ T) ∧ T = 6 * Real.pi) ∧ 
  (∀ x, f x ≤ sqrt 2 ∧ (∃ x, f x = sqrt 2)) :=
by
  sorry

end min_pos_period_max_value_l33_33145


namespace function_equals_parabola_l33_33815

theorem function_equals_parabola (f : ℝ → ℝ) 
  (h : ∀ l : ℝ → ℝ, l.is_line → 
         (∃ x₀ : ℝ, l x₀ = (x₀ ^ 2)) → 
         ∃ x₁ : ℝ, l x₁ = f x₁) : 
  f = (λ x : ℝ, x^2) := 
by 
  sorry

end function_equals_parabola_l33_33815


namespace incenter_inside_tangency_tetrahedron_l33_33880

structure Tetrahedron (V : Type) := 
  (A B C D : V)

noncomputable def center_of_inscribed_sphere {V : Type} [EuclideanSpace V] (T : Tetrahedron V) : V := sorry

noncomputable def points_of_tangency {V : Type} [EuclideanSpace V] (T : Tetrahedron V) : fin 4 → V := sorry

def lies_inside {V : Type} [EuclideanSpace V] (p : V) (T' : Tetrahedron V) : Prop := sorry

theorem incenter_inside_tangency_tetrahedron {V : Type} [EuclideanSpace V] 
    (T : Tetrahedron V) : 
    lies_inside (center_of_inscribed_sphere T) (Tetrahedron.mk 
        (points_of_tangency T 0) 
        (points_of_tangency T 1) 
        (points_of_tangency T 2) 
        (points_of_tangency T 3)) :=
sorry

end incenter_inside_tangency_tetrahedron_l33_33880


namespace find_m_such_that_no_linear_term_in_expansion_l33_33784

theorem find_m_such_that_no_linear_term_in_expansion :
  ∃ m : ℝ, ∀ x : ℝ, (x^2 - x + m) * (x - 8) = x^3 - 9 * x^2 - 8 * m ∧ ((8 + m) = 0) :=
by
  sorry

end find_m_such_that_no_linear_term_in_expansion_l33_33784


namespace Louisa_first_day_distance_l33_33872

theorem Louisa_first_day_distance (x : ℝ) (speed : ℝ) (distance_second_day : ℝ) (time_difference : ℝ) :
    speed = 33.333333333333336 ∧
    distance_second_day = 350 ∧
    time_difference = 3 → 
    x = speed * ((distance_second_day / speed) - time_difference) →
    x = 250 :=
by
  intro h
  rcases h with ⟨hs, hd, ht⟩
  rw [hs, hd, ht]
  norm_num
  exact eq.refl 250

end Louisa_first_day_distance_l33_33872


namespace possible_values_of_m_l33_33391

open Set

variable (A B : Set ℤ)
variable (m : ℤ)

theorem possible_values_of_m (h₁ : A = {1, 2, m * m}) (h₂ : B = {1, m}) (h₃ : B ⊆ A) :
  m = 0 ∨ m = 2 :=
  sorry

end possible_values_of_m_l33_33391


namespace rectangle_area_l33_33649

theorem rectangle_area (w l : ℝ) (hw : w = 2) (hl : l = 3) : w * l = 6 := by
  sorry

end rectangle_area_l33_33649


namespace xiaoli_time_l33_33173

variable {t : ℕ} -- Assuming t is a natural number (time in seconds)

theorem xiaoli_time (record_time : ℕ) (t_non_break : t ≥ record_time) (h : record_time = 14) : t ≥ 14 :=
by
  rw [h] at t_non_break
  exact t_non_break

end xiaoli_time_l33_33173


namespace age_ratio_l33_33628

theorem age_ratio (x : ℕ) (h : (5 * x - 4) = (3 * x + 4)) :
    (5 * x + 4) / (3 * x - 4) = 3 :=
by sorry

end age_ratio_l33_33628


namespace first_term_arith_seq_l33_33841

noncomputable def is_increasing (a b c : ℕ) (d : ℕ) : Prop := b = a + d ∧ c = a + 2 * d ∧ 0 < d

theorem first_term_arith_seq (a₁ a₂ a₃ : ℕ) (d: ℕ) :
  is_increasing a₁ a₂ a₃ d ∧ a₁ + a₂ + a₃ = 12 ∧ a₁ * a₂ * a₃ = 48 → a₁ = 2 := sorry

end first_term_arith_seq_l33_33841


namespace negation_of_universal_proposition_l33_33517

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 + 1 > 0)) ↔ ∃ x : ℝ, x^2 + 1 ≤ 0 :=
by
  sorry

end negation_of_universal_proposition_l33_33517


namespace fraction_to_decimal_l33_33643

theorem fraction_to_decimal : (9 : ℚ) / 25 = 0.36 :=
by
  sorry

end fraction_to_decimal_l33_33643


namespace average_transformed_data_l33_33651

theorem average_transformed_data (n : ℕ) (x : ℕ → ℝ) (h : (Finset.sum (Finset.range n) (λ i, x i)) / n = 30) : 
  (Finset.sum (Finset.range n) (λ i, 2 * x i + 1)) / n = 61 :=
by
  sorry

end average_transformed_data_l33_33651


namespace cubic_sum_l33_33779

theorem cubic_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 13) : x^3 + y^3 = 35 := 
sorry

end cubic_sum_l33_33779


namespace boxes_needed_to_complete_flooring_l33_33574

-- Definitions of given conditions
def length_of_living_room : ℕ := 16
def width_of_living_room : ℕ := 20
def sq_ft_per_box : ℕ := 10
def sq_ft_already_covered : ℕ := 250

-- Statement to prove
theorem boxes_needed_to_complete_flooring : 
  (length_of_living_room * width_of_living_room - sq_ft_already_covered) / sq_ft_per_box = 7 :=
by
  sorry

end boxes_needed_to_complete_flooring_l33_33574


namespace no_real_roots_example_problem_l33_33615

theorem no_real_roots_example_problem :
  (∀ (a b c : ℝ), (b^2 < 4 * a * c) → (a * (Option.isNone (nat.find_x (x^2 + 0 * x + (b / a))) = true)) → 
  (a = 1 ∧ b = -2 ∧ c = 0 ∨ 
   a = 1 ∧ b = 4 ∧ c = -1 ∨ 
   a = 2 ∧ b = -4 ∧ c = 3 ∨ 
   a = 3 ∧ b = -5 ∧ c = 2)) → false :=
begin
  sorry
end

end no_real_roots_example_problem_l33_33615


namespace count_valid_N_l33_33006

theorem count_valid_N : ∃ (count : ℕ), count = 10 ∧ 
    (∀ N : ℕ, (10 ≤ N ∧ N < 100) → 
        (∃ a b c d : ℕ, 
            a < 3 ∧ b < 3 ∧ c < 3 ∧ d < 4 ∧
            N = 3 * a + b ∧ N = 4 * c + d ∧
            2 * N % 50 = ((9 * a + b) + (8 * c + d)) % 50)) :=
sorry

end count_valid_N_l33_33006


namespace sculpture_cost_in_CNY_l33_33874

theorem sculpture_cost_in_CNY
  (usd_to_nad : 1 = 8)
  (usd_to_gbp : 1 = 5)
  (gbp_to_cny : 1 = 10)
  (sculpture_cost_nad : 160) :
  let sculpture_cost_usd := sculpture_cost_nad / 8
  let sculpture_cost_gbp := sculpture_cost_usd / 5
  let sculpture_cost_cny := sculpture_cost_gbp * 10
  sculpture_cost_cny = 40 :=
by
  sorry

end sculpture_cost_in_CNY_l33_33874


namespace net_amount_spent_l33_33863

theorem net_amount_spent : 
  let original_trumpet_price := 250
  let discount_rate := 0.30
  let music_stand_price := 25
  let sheet_music_price := 15
  let song_book_sale_price := 5.84
  let discounted_trumpet_price := original_trumpet_price - (original_trumpet_price * discount_rate)
  let total_additional_items_price := music_stand_price + sheet_music_price
  let total_spent_before_sale := discounted_trumpet_price + total_additional_items_price
  let net_spent := total_spent_before_sale - song_book_sale_price
  in net_spent = 209.16 :=
by
  sorry

end net_amount_spent_l33_33863


namespace minimum_period_and_max_value_of_f_l33_33053

noncomputable def f (x : ℝ) : ℝ := sin (x / 3) + cos (x / 3)

theorem minimum_period_and_max_value_of_f :
  (∃ T > 0, ∀ x: ℝ, f (x + T) = f x) ∧
  (∀ x: ℝ, f x ≤ sqrt 2 ∧ 
          (∃ y: ℝ, f y = sqrt 2)) :=
by 
  sorry

end minimum_period_and_max_value_of_f_l33_33053


namespace tom_needs_more_boxes_l33_33577

theorem tom_needs_more_boxes
    (living_room_length : ℕ)
    (living_room_width : ℕ)
    (box_coverage : ℕ)
    (already_installed : ℕ) :
    living_room_length = 16 →
    living_room_width = 20 →
    box_coverage = 10 →
    already_installed = 250 →
    (living_room_length * living_room_width - already_installed) / box_coverage = 7 :=
by
    intros h1 h2 h3 h4
    rw [h1, h2, h3, h4]
    sorry

end tom_needs_more_boxes_l33_33577


namespace find_third_number_in_second_set_l33_33917

theorem find_third_number_in_second_set (x y: ℕ) 
    (h1 : (28 + x + 42 + 78 + 104) / 5 = 90) 
    (h2 : (128 + 255 + y + 1023 + x) / 5 = 423) 
: y = 511 := 
sorry

end find_third_number_in_second_set_l33_33917


namespace find_initial_popular_trees_l33_33946

theorem find_initial_popular_trees (n_plant : ℕ) (n_total : ℕ) (h1 : n_plant = 73) (h2 : n_total = 98) :
  ∃ n_initial : ℕ, n_initial + n_plant = n_total ∧ n_initial = 25 :=
by
  use 25
  split
  · simp [h1, h2]
  · refl

end find_initial_popular_trees_l33_33946


namespace min_positive_period_and_max_value_l33_33105

noncomputable def f (x : ℝ) : ℝ := Real.sin (x / 3) + Real.cos (x / 3)

theorem min_positive_period_and_max_value :
  (∃ T > 0, ∀ x, f (x + T) = f x) ∧ ∃ x, f x = sqrt 2 :=
by
  sorry

end min_positive_period_and_max_value_l33_33105


namespace problem1_problem2_l33_33678

variable {x : ℝ} (hx : x > 0)

theorem problem1 : (2 / (3 * x)) * Real.sqrt (9 * x^3) + 6 * Real.sqrt (x / 4) - 2 * x * Real.sqrt (1 / x) = 3 * Real.sqrt x := 
by sorry

theorem problem2 : (Real.sqrt 24 + Real.sqrt 6) / Real.sqrt 3 + (Real.sqrt 3 + 1) * (Real.sqrt 3 - 1) = 3 * Real.sqrt 2 + 2 := 
by sorry

end problem1_problem2_l33_33678


namespace largest_prime_divisor_base6_100111011_is_181_l33_33698

def number_in_base_10 (n : Nat) (b : Nat) : Nat :=
  -- Converts a base-b number represented as a natural number n to its decimal form.
  let rec convert (n : Nat) (power : Nat) : Nat :=
    if n = 0 then 0
    else (n % 10) * (b ^ power) + convert (n / 10) (power + 1)
  convert n 0

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ (∀ k : Nat, k > 1 → k < n → n % k ≠ 0)

def largest_prime_divisor (n : Nat) : Nat :=
  -- Returns the largest prime divisor of n
  let rec find (d : Nat) (largest : Nat) :=
    if d = 0 then largest
    else if n % d = 0 ∧ is_prime d then find (d - 1) d
    else find (d - 1) largest
  find (n / 2) 1

theorem largest_prime_divisor_base6_100111011_is_181 :
  largest_prime_divisor (number_in_base_10 100111011 6) = 181 :=
by sorry

end largest_prime_divisor_base6_100111011_is_181_l33_33698


namespace B_is_perfect_square_D_is_perfect_square_E_is_perfect_square_l33_33614

-- Definitions for condition: Perfect square implies all exponents even.
def is_perfect_square (n : ℕ) : Prop :=
  ∀ p k : ℕ, prime p → n = p ^ (2 * k)

-- Define each option in problem (A), (B), (C), (D), (E)
def A : ℕ := 3^4 * 2^(2*5) * 7^7
def B : ℕ := 3^6 * 2^(2*4) * 7^6
def C : ℕ := 3^5 * 2^(2*6) * 7^5
def D : ℕ := 3^4 * 2^(2*7) * 7^4
def E : ℕ := 3^6 * 2^(2*6) * 7^6

-- Statements to prove each option is a perfect square
theorem B_is_perfect_square : is_perfect_square B := sorry
theorem D_is_perfect_square : is_perfect_square D := sorry
theorem E_is_perfect_square : is_perfect_square E := sorry

end B_is_perfect_square_D_is_perfect_square_E_is_perfect_square_l33_33614


namespace largest_valid_set_size_l33_33463

-- Define the function to compute the sum of digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define the property for set S such that for any n in S, sum of digits of n is not divisible by 11
def valid_set (S : Set ℕ) : Prop :=
  ∀ n ∈ S, sum_of_digits n % 11 ≠ 0

-- State the theorem
theorem largest_valid_set_size :
  ∃ (S : Set ℕ), valid_set S ∧ S.card = 38 :=
sorry  -- Proof omitted

end largest_valid_set_size_l33_33463


namespace min_positive_period_and_max_value_of_f_l33_33088

noncomputable def f (x : ℝ) : ℝ := sin (x / 3) + cos (x / 3)

theorem min_positive_period_and_max_value_of_f :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ T = 6 * π) ∧ 
  (∃ x, f x = sqrt 2) :=
by {
  sorry
}

end min_positive_period_and_max_value_of_f_l33_33088


namespace max_possible_words_l33_33806

-- Define the language and its constraints
def is_valid_word (w : List Char) : Prop :=
  w.length ≥ 1 ∧ w.length ≤ 13 ∧ ∀ v1 v2, v1 ++ v2 = w → ¬(is_valid_word v1 ∧ is_valid_word v2)

-- Note: The is_valid_word function is used to denote valid words based on the problem's constraints.
-- The proof or definition to check if the concatenated word is invalid is only represented as a proposition.

theorem max_possible_words : ∃ max_words, max_words = 16256 :=
begin
  sorry
end

end max_possible_words_l33_33806


namespace largest_prime_inequality_l33_33838

def largest_prime_divisor (n : Nat) : Nat :=
  sorry  -- Placeholder to avoid distractions in problem statement

theorem largest_prime_inequality (q : Nat) (h_q_prime : Prime q) (hq_odd : q % 2 = 1) :
    ∃ k : Nat, k > 0 ∧ largest_prime_divisor (q^(2^k) - 1) < q ∧ q < largest_prime_divisor (q^(2^k) + 1) :=
sorry

end largest_prime_inequality_l33_33838


namespace range_rational_function_l33_33608

noncomputable def rational_function (x : ℝ) : ℝ := (x^2 + 5*x + 6) / (x + 2)

theorem range_rational_function :
  (Set.range rational_function) = Set.Ioo (⊥ : ℝ) 1 ∪ Set.Ioo 1 ⊤ :=
by
  sorry

end range_rational_function_l33_33608


namespace opposite_face_of_7_l33_33511

/-- A type representing a die with faces labeled with 6, 7, 8, 9, 10, and 11. -/
def Die := {n : ℕ // n ∈ {6, 7, 8, 9, 10, 11}}

namespace Die

/-- Total sum of the face values of a die. -/
def total_sum : ℕ := 6 + 7 + 8 + 9 + 10 + 11

/-- Given the sum of the vertical faces in the first roll. -/
def sum_first_roll : ℕ := 33

/-- Given the sum of the vertical faces in the second roll. -/
def sum_second_roll : ℕ := 35

/-- Given the sum of the remaining two faces in the first roll. -/
def remaining_sum_first_roll := total_sum - sum_first_roll

/-- Given the sum of the remaining two faces in the second roll. -/
def remaining_sum_second_roll := total_sum - sum_second_roll

/-- Sum of the faces opposite each other must add up to either 16, 17, or 18. -/
lemma opposite_face_sum (a b : Die) : 
  a.1 + b.1 = remaining_sum_first_roll ∨ a.1 + b.1 = remaining_sum_second_roll ∨ a.1 + b.1 = 17 :=
sorry

/-- Prove the faces opposite the face with number 7. -/
theorem opposite_face_of_7 (x : Die) : 
  (x.1 = 9 ∨ x.1 = 11) → 
  (∀ p, ((p.1 = 6 ∨ p.1 = 8 ∨ p.1 = 10 ∨ p.1 = 11) → p.1 + 7 = remaining_sum_first_roll ∨ p.1 + 7 = remaining_sum_second_roll ∨ p.1 + 7 = 17)) :=
sorry

end Die

end opposite_face_of_7_l33_33511


namespace distribute_students_l33_33304

theorem distribute_students (total_students : ℕ) (class_student : ℕ → ℕ) (classes : ℕ) :
  total_students = 5 →
  classes = 3 →
  class_student 0 = 1 →
  ∀ c, class_student c ≥ 1 →
  (∃ dist : ℕ, (dist = 56)) :=
by
  intros h_total h_classes h_classA h_no_empty
  use 56
  sorry

end distribute_students_l33_33304


namespace remainder_identity_l33_33000

variable {n : ℕ}

theorem remainder_identity
  (a b a_1 b_1 a_2 b_2 : ℕ)
  (ha : a = a_1 + a_2 * n)
  (hb : b = b_1 + b_2 * n) :
  (((a + b) % n = (a_1 + b_1) % n) ∧ ((a - b) % n = (a_1 - b_1) % n)) ∧ ((a * b) % n = (a_1 * b_1) % n) := by
  sorry

end remainder_identity_l33_33000


namespace proposition_2_correct_l33_33461

variables (l m : Line) (α : Plane)

axiom diff_lines : l ≠ m

-- Proposition ②: If \( l \perp \alpha \) and \( l \parallel m \), then \( m \perp \alpha \)
theorem proposition_2_correct : (l ⟂ α) ∧ (l ⋈ m) → (m ⟂ α) :=
by
  sorry

end proposition_2_correct_l33_33461


namespace solution_set_ineq_l33_33746

-- Definitions based on the problem conditions
def f : ℝ → ℝ := sorry

axiom domain_f : ∀ x : ℝ, x ∈ (λ y, y ∈ ℝ)
axiom f_property : ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) > 3
axiom f_at_5 : f 5 = 18

-- Statement to prove the solution set of the given inequality
theorem solution_set_ineq (x : ℝ) : f (3 * x - 1) > 9 * x ↔ x > 2 :=
by
  sorry

end solution_set_ineq_l33_33746


namespace count_a_divisible_by_5_l33_33328

theorem count_a_divisible_by_5 : 
  (Finset.filter (λ a : ℕ, (a ^ 2014 + a ^ 2015) % 5 = 0) (Finset.range 11)).card = 4 := 
by 
  sorry

end count_a_divisible_by_5_l33_33328


namespace stored_bales_correct_l33_33565

theorem stored_bales_correct :
  let initial_bales := 28
  let new_bales := 54
  let stored_bales := new_bales - initial_bales
  stored_bales = 26 :=
by
  let initial_bales := 28
  let new_bales := 54
  let stored_bales := new_bales - initial_bales
  show stored_bales = 26
  sorry

end stored_bales_correct_l33_33565


namespace solve_ff_eq_x_l33_33501

noncomputable def f (x : ℝ) : ℝ := x^2 + 5 * x + 1

theorem solve_ff_eq_x :
  (∃ x : ℝ, f(f(x)) = x ∧ (x = -2 - Real.sqrt 3 ∨ x = -2 + Real.sqrt 3 ∨ x = -3 - Real.sqrt 2 ∨ x = -3 + Real.sqrt 2)) :=
by {
  sorry
}

end solve_ff_eq_x_l33_33501


namespace lcm_of_1716_924_1260_l33_33605

-- Define the prime factorizations
def prime_factorization_1716 : List (ℕ × ℕ) := [(2, 2), (3, 2), (19, 1)]
def prime_factorization_924 : List (ℕ × ℕ) := [(2, 2), (3, 1), (7, 1), (11, 1)]
def prime_factorization_1260 : List (ℕ × ℕ) := [(2, 2), (3, 2), (5, 1), (7, 1)]

-- Define the least common multiple
def LCM_1716_924_1260 := 13860

-- Proposition that the least common multiple of 1716, 924, and 1260 is 13860
theorem lcm_of_1716_924_1260 : (LCM 1716 924 1260) = 13860 := by
  sorry

end lcm_of_1716_924_1260_l33_33605


namespace min_period_max_value_f_l33_33135

def f (x : ℝ) : ℝ := sin (x / 3) + cos (x / 3)

theorem min_period_max_value_f :
  ∃ T M, (∀ x, f (x + T) = f x) ∧ 0 < T ∧
    0 < M ∧ (∀ x, f x ≤ M) ∧ (∃ x, f x = M) ∧ 
    T = 6 * π ∧ M = √2 := sorry

end min_period_max_value_f_l33_33135


namespace total_jumps_is_400_l33_33004

-- Define the variables according to the conditions 
def Ronald_jumps := 157
def Rupert_jumps := Ronald_jumps + 86

-- Prove the total jumps
theorem total_jumps_is_400 : Ronald_jumps + Rupert_jumps = 400 := by
  sorry

end total_jumps_is_400_l33_33004


namespace eval_expr_l33_33692

theorem eval_expr : (9⁻¹ - 6⁻¹)⁻¹ = -18 := 
by
sorry

end eval_expr_l33_33692


namespace min_period_and_max_value_l33_33062

def f (x : ℝ) : ℝ := Real.sin (x / 3) + Real.cos (x / 3)

noncomputable def minimum_positive_period (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x : ℝ, f (x + T) = f x ∧ T > 0 ∧ ∀ T' : ℝ, (T' > 0 ∧ ∀ x : ℝ, f (x + T') = f x) → T' ≥ T

noncomputable def is_max_value (f : ℝ → ℝ) (M : ℝ) : Prop :=
  ∀ x : ℝ, f x ≤ M ∧ ∃ x : ℝ, f x = M

theorem min_period_and_max_value :
  minimum_positive_period f (6 * Real.pi) ∧ is_max_value f (Real.sqrt 2) := by
  sorry

end min_period_and_max_value_l33_33062


namespace min_pos_period_max_value_l33_33144

noncomputable def f (x : ℝ) := Real.sin (x / 3) + Real.cos (x / 3)

theorem min_pos_period_max_value :
  (∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ T) ∧ T = 6 * Real.pi) ∧ 
  (∀ x, f x ≤ sqrt 2 ∧ (∃ x, f x = sqrt 2)) :=
by
  sorry

end min_pos_period_max_value_l33_33144


namespace range_of_x_for_f_geq_neg_one_l33_33342

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then (1/2) * x + 1 else -(x - 1)^2

theorem range_of_x_for_f_geq_neg_one :
  {x : ℝ | f x ≥ -1} = set.Icc (-4 : ℝ) 2 :=
sorry

end range_of_x_for_f_geq_neg_one_l33_33342


namespace complex_solution_l33_33730

noncomputable def complex_problem (z1 z2 : ℂ) : Prop :=
  abs z1 = 1 ∧ abs z2 = 2 ∧ 3 * z1 - z2 = 2 + (Real.sqrt 3) * Complex.I

theorem complex_solution (z1 z2 : ℂ) (h : complex_problem z1 z2) :
  2 * z1 + z2 = 3 - (Real.sqrt 3) * Complex.I ∨ 
  2 * z1 + z2 = -(9 : ℝ) / 7 + (13 * (Real.sqrt 3) / 7) * Complex.I :=
by sorry

end complex_solution_l33_33730


namespace sequence_5th_term_l33_33799

open scoped Nat

def sequence (s : ℕ → ℝ) : Prop :=
  s 0 = 3 ∧
  s 3 = 48 ∧
  ∀ n ≥ 1, s n = (1 / 4) * (s (n - 1) + s (n + 1))

theorem sequence_5th_term (s : ℕ → ℝ) (h : sequence s) : s 4 = 179 :=
by
  sorry

end sequence_5th_term_l33_33799


namespace min_positive_period_max_value_l33_33113
open Real

noncomputable def f (x : ℝ) : ℝ := sin (x / 3) + cos (x / 3)

theorem min_positive_period_max_value :
  (∃ T > 0, ∀ x, f(x + T) = f(x) ∧ T = 6 * π) ∧
  (∀ x, f(x) ≤ sqrt 2) ∧ 
  (∃ x, f(x) = sqrt 2) :=
by
  sorry

end min_positive_period_max_value_l33_33113


namespace min_pos_period_max_value_l33_33146

noncomputable def f (x : ℝ) := Real.sin (x / 3) + Real.cos (x / 3)

theorem min_pos_period_max_value :
  (∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ T) ∧ T = 6 * Real.pi) ∧ 
  (∀ x, f x ≤ sqrt 2 ∧ (∃ x, f x = sqrt 2)) :=
by
  sorry

end min_pos_period_max_value_l33_33146


namespace gecko_third_day_crickets_l33_33237

def total_crickets : ℕ := 70
def first_day_percentage : ℝ := 0.30
def first_day_crickets : ℝ := first_day_percentage * total_crickets
def second_day_crickets : ℝ := first_day_crickets - 6
def third_day_crickets : ℝ := total_crickets - (first_day_crickets + second_day_crickets)

theorem gecko_third_day_crickets :
  third_day_crickets = 34 :=
by
  sorry

end gecko_third_day_crickets_l33_33237


namespace ab_eq_one_l33_33379

theorem ab_eq_one (a b : ℝ) (h1 : a ≠ b) (h2 : abs (Real.log a) = abs (Real.log b)) : a * b = 1 := sorry

end ab_eq_one_l33_33379


namespace triangle_angle_120_l33_33424

theorem triangle_angle_120 (X Y Z W : Type) [metric_space X] [metric_space Y] [metric_space Z]
  [metric_space W]
  (h1 : W ∈ segment X Z)
  (h2 : dist X W = dist W Z)
  (h3 : angle X W Z = 60) : angle Z Y W = 120 :=
by 
  sorry

end triangle_angle_120_l33_33424


namespace ratio_of_gpa_l33_33038

variable (A B : ℝ)

-- Conditions
axiom gpa_A : A > 0
axiom gpa_B : B > 0
axiom avg15 : 15 * A
axiom avg18 : 18 * B
axiom total_avg : 17 * (A + B)

-- Proof statement
theorem ratio_of_gpa :
  (15 * A + 18 * B = 17 * (A + B)) →
  (A / (A + B)) = 1 / 3 :=
by
  sorry

end ratio_of_gpa_l33_33038


namespace green_eyes_students_l33_33866

def total_students := 45
def brown_hair_condition (green_eyes : ℕ) := 3 * green_eyes
def both_attributes := 9
def neither_attributes := 5

theorem green_eyes_students (green_eyes : ℕ) :
  (total_students = (green_eyes - both_attributes) + both_attributes
    + (brown_hair_condition green_eyes - both_attributes) + neither_attributes) →
  green_eyes = 10 :=
by
  sorry

end green_eyes_students_l33_33866


namespace ellipse_equation_parameters_absolute_sum_l33_33667

theorem ellipse_equation_parameters_absolute_sum : 
  ∃ A B C D E F : ℤ,
  (∀ t : ℝ, 
    let sin_t := (3 * (6 * t + 1) / (6 - t)) in
    let cos_t := 2 + (t * (3 - sin_t)) / 3 in
    (let x := (3 * (cos_t - 2)) / (3 - sin_t) in
    let y := 4 * (sin_t - 6) / (3 - sin_t) in
      A * x ^ 2 + B * x * y + C * y ^ 2 + D * x + E * y + F = 0)) ∧
  ∀g, g = Int.gcd A (Int.gcd B (Int.gcd C (Int.gcd D (Int.gcd E F))) ∧
  g = 1)
   ∧ |A| + |B| + |C| + |D| + |E| + |F| = 14288 := 
 sorry

end ellipse_equation_parameters_absolute_sum_l33_33667


namespace marbles_in_jar_l33_33945

theorem marbles_in_jar (T : ℕ) (T_half : T / 2 = 12) (red_marbles : ℕ) (orange_marbles : ℕ) (total_non_blue : red_marbles + orange_marbles = 12) (red_count : red_marbles = 6) (orange_count : orange_marbles = 6) : T = 24 :=
by
  sorry

end marbles_in_jar_l33_33945


namespace flowers_bloom_l33_33557

/-- There are 12 unicorns in the Enchanted Forest. Everywhere a unicorn steps, 
seven flowers spring into bloom. The 12 unicorns are going to walk all the way 
across the forest side-by-side, a journey of 15 kilometers. If each unicorn 
moves 3 meters forward with each step, prove the number of flowers bloom 
because of this trip is 420000. -/
theorem flowers_bloom (unicorns : ℕ) (steps_per_meter : ℕ) (journey_km : ℕ) (flowers_per_step : ℕ) :
  unicorns = 12 → steps_per_meter = 3 → journey_km = 15 → flowers_per_step = 7 → 
  unicorns * (journey_km * 1000 / steps_per_meter) * flowers_per_step = 420000 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end flowers_bloom_l33_33557


namespace polar_center_coordinates_l33_33544

theorem polar_center_coordinates (θ ρ : ℝ) (h : ρ = 2 * cos θ + 2 * sin θ) :
    (1, 1) = (sqrt 2, π / 4) := 
sorry

end polar_center_coordinates_l33_33544


namespace sum_of_base4_numbers_is_correct_l33_33324

-- Define the four base numbers
def n1 : ℕ := 2 * 4^2 + 1 * 4^1 + 2 * 4^0
def n2 : ℕ := 1 * 4^2 + 0 * 4^1 + 3 * 4^0
def n3 : ℕ := 3 * 4^2 + 2 * 4^1 + 1 * 4^0

-- Define the expected sum in base 4 interpreted as a natural number
def expected_sum : ℕ := 1 * 4^3 + 0 * 4^2 + 1 * 4^1 + 2 * 4^0

-- State the theorem
theorem sum_of_base4_numbers_is_correct : n1 + n2 + n3 = expected_sum := by
  sorry

end sum_of_base4_numbers_is_correct_l33_33324


namespace area_ratio_of_square_side_multiplied_by_10_l33_33992

theorem area_ratio_of_square_side_multiplied_by_10 (s : ℝ) (A_original A_resultant : ℝ) 
  (h1 : A_original = s^2)
  (h2 : A_resultant = (10 * s)^2) :
  (A_original / A_resultant) = (1 / 100) :=
by
  sorry

end area_ratio_of_square_side_multiplied_by_10_l33_33992


namespace park_area_approx_l33_33432

noncomputable def area_of_park (r L B : ℝ) (x : ℝ) : ℝ :=
  L * B

theorem park_area_approx :
  ∀ (L B : ℝ)
    (h1 : ∃ (x : ℝ), L = 3 * x ∧ B = 5 * x)
    (h2 : 15 * 1000 / 60 * 12 = 3.5 * 2 * (L + B)),
  area_of_park 43057.60 L B ≈ 43057.60 :=
by
  sorry

end park_area_approx_l33_33432


namespace people_to_left_of_kolya_l33_33527

theorem people_to_left_of_kolya (people_right_kolya people_left_sasha people_right_sasha : ℕ) (total_people : ℕ) :
  (people_right_kolya = 12) →
  (people_left_sasha = 20) →
  (people_right_sasha = 8) →
  (total_people = people_left_sasha + people_right_sasha + 1) →
  total_people - people_right_kolya - 1 = 16 :=
begin
  sorry
end

end people_to_left_of_kolya_l33_33527


namespace arithmetic_sequence_sum_13_l33_33726

theorem arithmetic_sequence_sum_13 (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h : ∀ n, S n = n * a 1 + (n * (n - 1) / 2) * ((a (n+1) - a n) / ((n+1).toReal - n.toReal)))
  (h1 : a 3 + a 7 - a 10 = 8)
  (h2 : a 11 - a 4 = 4) :
  S 13 = 156 := 
by 
  sorry

end arithmetic_sequence_sum_13_l33_33726


namespace least_prime_factor_of_5_pow_5_minus_5_pow_4_l33_33974

theorem least_prime_factor_of_5_pow_5_minus_5_pow_4 : 
  ∃ p : ℕ, nat.prime p ∧ p = 2 ∧ ∃ f : nat.factorization, 
  (5^5 - 5^4) = (f.prod) ∧ p ∈ f.support :=
begin
  sorry
end

end least_prime_factor_of_5_pow_5_minus_5_pow_4_l33_33974


namespace xiao_wang_programming_methods_l33_33987

theorem xiao_wang_programming_methods :
  ∃ (n : ℕ), n = 20 :=
by sorry

end xiao_wang_programming_methods_l33_33987


namespace x_intercept_of_perpendicular_line_l33_33592

noncomputable def x_intercept_perpendicular (m₁ m₂ : ℚ) : ℚ :=
  let m_perpendicular := -1 / m₁ in
  let b := -3 in
  -b / m_perpendicular

theorem x_intercept_of_perpendicular_line :
  (4 * x_intercept_perpendicular (-4/5) (5/4) + 5 * 0) = 10 :=
by
  sorry

end x_intercept_of_perpendicular_line_l33_33592


namespace min_positive_period_max_value_l33_33115
open Real

noncomputable def f (x : ℝ) : ℝ := sin (x / 3) + cos (x / 3)

theorem min_positive_period_max_value :
  (∃ T > 0, ∀ x, f(x + T) = f(x) ∧ T = 6 * π) ∧
  (∀ x, f(x) ≤ sqrt 2) ∧ 
  (∃ x, f(x) = sqrt 2) :=
by
  sorry

end min_positive_period_max_value_l33_33115


namespace line_passes_through_fixed_point_l33_33882

theorem line_passes_through_fixed_point :
  ∀ (m : ℝ), ∃ (x y : ℝ), (m - 1) * x + (2m - 1) * y = m - 5 ∧ x = 9 ∧ y = -4 :=
by
  intro m
  use 9, -4
  split
  { calc
      (m - 1) * 9 + (2m - 1) * (-4) 
        = 9m - 9 - 8m + 4 
        : by ring
     ... = m - 5 
        : by ring }
  split
  { refl }
  { refl }
  sorry

end line_passes_through_fixed_point_l33_33882


namespace min_period_and_max_value_l33_33056

def f (x : ℝ) : ℝ := Real.sin (x / 3) + Real.cos (x / 3)

noncomputable def minimum_positive_period (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x : ℝ, f (x + T) = f x ∧ T > 0 ∧ ∀ T' : ℝ, (T' > 0 ∧ ∀ x : ℝ, f (x + T') = f x) → T' ≥ T

noncomputable def is_max_value (f : ℝ → ℝ) (M : ℝ) : Prop :=
  ∀ x : ℝ, f x ≤ M ∧ ∃ x : ℝ, f x = M

theorem min_period_and_max_value :
  minimum_positive_period f (6 * Real.pi) ∧ is_max_value f (Real.sqrt 2) := by
  sorry

end min_period_and_max_value_l33_33056


namespace min_positive_period_and_max_value_l33_33080

def f (x : ℝ) := Real.sin (x / 3) + Real.cos (x / 3)

theorem min_positive_period_and_max_value : 
  (∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T' = T)) 
  ∧ 
  (∀ x, f x ≤ sqrt 2 ∧ (∃ x, f x = sqrt 2)) :=
sorry

end min_positive_period_and_max_value_l33_33080


namespace students_on_zoo_trip_l33_33870

theorem students_on_zoo_trip (buses : ℕ) (students_per_bus : ℕ) (students_in_cars : ℕ) 
  (h1 : buses = 7) (h2 : students_per_bus = 56) (h3 : students_in_cars = 4) : 
  buses * students_per_bus + students_in_cars = 396 :=
by
  sorry

end students_on_zoo_trip_l33_33870


namespace least_prime_factor_of_expression_l33_33976

theorem least_prime_factor_of_expression : ∃ (p : ℕ), prime p ∧ p ∣ (5^5 - 5^4) ∧ (∀ q : ℕ, prime q → q ∣ (5^5 - 5^4) → q ≥ p) :=
by
  sorry

end least_prime_factor_of_expression_l33_33976


namespace exist_intersection_points_l33_33581

noncomputable def intersection_points_of_circle_and_ellipse
  (F1 F2 : ℝ × ℝ) (a : ℝ) (O : ℝ × ℝ) (r : ℝ) : set (ℝ × ℝ) :=
{ (x, y) : ℝ × ℝ |
  let d1 := (x - F1.1)^2 + (y - F1.2)^2,
      d2 := (x - F2.1)^2 + (y - F2.2)^2 in
  d1 + d2 = (2 * a)^2 ∧ (x - O.1)^2 + (y - O.2)^2 = r^2 }

theorem exist_intersection_points
  (F1 F2 : ℝ × ℝ) (a : ℝ) (O : ℝ × ℝ) (r : ℝ) :
  ∃ (Q : set (ℝ × ℝ)), Q = intersection_points_of_circle_and_ellipse F1 F2 a O r :=
by
  sorry

end exist_intersection_points_l33_33581


namespace sister_pairs_of_f_l33_33803

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x^2 + 2 * x else (1 / Real.exp x)

def is_sister_pair (A B : ℝ × ℝ) : Prop :=
  A.1 = B.1 ∧ A.2 = f A.1 ∧ B.2 = f B.1 ∧ A.1 = -B.1 ∧ A.2 = -B.2

def sister_pairs_count : ℕ :=
  ((A : ℝ × ℝ) × (B : ℝ × ℝ) → is_sister_pair A B).card

theorem sister_pairs_of_f :
  sister_pairs_count = 2 := 
sorry

end sister_pairs_of_f_l33_33803


namespace order_of_numbers_l33_33300

-- Conditions provided in the problem
def a : ℝ := 5 ^ (Real.log 3.4 / Real.log 2)
def b : ℝ := 5 ^ (Real.log 3.6 / (2 * Real.log 2))
def c : ℝ := 5 ^ ((Real.log 10 - Real.log 3) / Real.log 2)

-- Main theorem statement
theorem order_of_numbers : a > c ∧ c > b :=
by
  -- Proof is omitted
  sorry

end order_of_numbers_l33_33300


namespace number_of_integer_coordinates_on_circle_l33_33419

theorem number_of_integer_coordinates_on_circle (r : ℕ) (h : r = 5) :
  ∃ n : ℕ, n = 12 ∧ (finset.univ.filter (λ p : ℤ × ℤ, p.1^2 + p.2^2 = (r : ℤ)^2)).card = n :=
begin
  use 12,
  split,
  { refl },
  { sorry }
end

end number_of_integer_coordinates_on_circle_l33_33419


namespace count_odd_digits_in_base4_of_181_l33_33292

-- Definition of converting 181 from base-10 to base-8
def base_10_to_base_8 (n : Nat) : List Nat := [2, 6, 5]

-- Definition of converting the resulting base-8 number to base-4
def base_8_to_base_4 (n : List Nat) : List Nat :=
  n.flat_map (fun d => match d with
    | 2 => [2]
    | 6 => [1, 2]
    | 5 => [1, 1]
    | _ => []
  )

-- Definition of counting odd digits in the base-4 representation
def count_odd_digits (n : List Nat) : Nat :=
  n.countp (λ d => d % 2 = 1)

-- Lean 4 theorem statement
theorem count_odd_digits_in_base4_of_181 : count_odd_digits (base_8_to_base_4 (base_10_to_base_8 181)) = 5 := by
  sorry

end count_odd_digits_in_base4_of_181_l33_33292


namespace Z_bijective_H_l33_33832

open Set

noncomputable def H : Set ℚ :=
  { x | x = 1/2 ∨ (∃ y, y ∈ H ∧ (x = 1 / (1 + y) ∨ x = y / (1 + y))) }

theorem Z_bijective_H : ∃ (f : ℤ → ℚ), Function.Bijective f ∧ ∀ i, f i ∈ H := by
  sorry

end Z_bijective_H_l33_33832


namespace no_solution_exists_l33_33701

theorem no_solution_exists : ¬ ∃ (n m : ℕ), (n + 1) * (2 * n + 1) = 2 * m^2 := by sorry

end no_solution_exists_l33_33701


namespace comb_club_ways_l33_33231

theorem comb_club_ways : (nat.choose 20 3) = 1140 :=
by
  sorry

end comb_club_ways_l33_33231


namespace sum_P_add_Q_l33_33765

-- Define the sets P and Q
def P : Set ℕ := {0, 2, 4}
def Q : Set ℕ := {1, 2, 3}

-- Define the set P + Q
def P_add_Q : Set ℕ := {x | ∃ a b, a ∈ P ∧ b ∈ Q ∧ x = a + b}

-- The theorem statement proving that the sum of all elements in P + Q is 28
theorem sum_P_add_Q : (Finset.sum (Finset.filter (λ x, x ∈ P_add_Q) (Finset.range 8))) = 28 :=
by
  sorry

end sum_P_add_Q_l33_33765


namespace ages_of_patients_l33_33996

theorem ages_of_patients (x y : ℕ) 
  (h1 : x - y = 44) 
  (h2 : x * y = 1280) : 
  (x = 64 ∧ y = 20) ∨ (x = 20 ∧ y = 64) := by
  sorry

end ages_of_patients_l33_33996


namespace sum_bn_over_3_pow_n_plus_1_eq_2_over_5_l33_33464

noncomputable def b : ℕ → ℚ
| 0     => 2
| 1     => 3
| (n+2) => 2 * b (n+1) + 3 * b n

theorem sum_bn_over_3_pow_n_plus_1_eq_2_over_5 :
  (∑' n : ℕ, (b n) / (3 ^ (n + 1))) = (2 / 5) :=
by
  sorry

end sum_bn_over_3_pow_n_plus_1_eq_2_over_5_l33_33464


namespace vertex_C_moves_along_ellipse_l33_33556

theorem vertex_C_moves_along_ellipse 
    (O A B C : Type) 
    [Geometry O A B C]
    (hA : A ∈ (x-axis))
    (hB : B ∈ (y-axis))
    (h_not_right_angle : ∠C ≠ 90°) : 
    ∃ (ellipse : Type), C ∈ ellipse :=
sorry

end vertex_C_moves_along_ellipse_l33_33556


namespace sine_neg_pi_over_3_l33_33204

def theta : Real := -π / 3

theorem sine_neg_pi_over_3 : Real.sin theta = - (Real.sin (π / 3)) := by
  sorry

end sine_neg_pi_over_3_l33_33204


namespace convex_polygon_max_acute_angles_l33_33401

theorem convex_polygon_max_acute_angles (n : ℕ) (h : n ≥ 3) : 
    (∑ i in finset.range n, external_angle i) = 360 → 
    ∀ P : convex_polygon n, ∃ (k : ℕ), k ≤ 3 ∧ acute_internal_angles P ≤ k :=
sorry

end convex_polygon_max_acute_angles_l33_33401


namespace distribution_problem_distribution_problem_variable_distribution_problem_equal_l33_33178

def books_distribution_fixed (n : ℕ) (a b c : ℕ) : ℕ :=
  if h : a + b + c = n then n.factorial / (a.factorial * b.factorial * c.factorial) else 0

theorem distribution_problem (n a b c : ℕ) (h : a + b + c = n) : 
  books_distribution_fixed n a b c = 1260 :=
sorry

def books_distribution_variable (n : ℕ) (a b c : ℕ) : ℕ :=
  if h : a + b + c = n then (n.factorial / (a.factorial * b.factorial * c.factorial)) * 6 else 0

theorem distribution_problem_variable (n a b c : ℕ) (h : a + b + c = n) : 
  books_distribution_variable n a b c = 7560 :=
sorry

def books_distribution_equal (n : ℕ) (k : ℕ) : ℕ :=
  if h : 3 * k = n then n.factorial / (k.factorial * k.factorial * k.factorial) else 0

theorem distribution_problem_equal (n k : ℕ) (h : 3 * k = n) : 
  books_distribution_equal n k = 1680 :=
sorry

end distribution_problem_distribution_problem_variable_distribution_problem_equal_l33_33178


namespace compute_expression_l33_33852

variable {R : Type*} [LinearOrderedField R]

theorem compute_expression (r s t : R)
  (h_eq_root: ∀ x, x^3 - 4 * x^2 + 4 * x - 6 = 0)
  (h1: r + s + t = 4)
  (h2: r * s + r * t + s * t = 4)
  (h3: r * s * t = 6) :
  r * s / t + s * t / r + t * r / s = -16 / 3 :=
sorry

end compute_expression_l33_33852


namespace probability_at_least_one_8_rolled_l33_33958

theorem probability_at_least_one_8_rolled :
  let total_outcomes := 64
  let no_8_outcomes := 49
  (total_outcomes - no_8_outcomes) / total_outcomes = 15 / 64 :=
by
  let total_outcomes := 8 * 8
  let no_8_outcomes := 7 * 7
  have h1 : total_outcomes = 64 := by norm_num
  have h2 : no_8_outcomes = 49 := by norm_num
  rw [← h1, ← h2]
  norm_num
  sorry

end probability_at_least_one_8_rolled_l33_33958


namespace min_positive_period_f_max_value_f_l33_33069

noncomputable def f (x : ℝ) : ℝ := sin (x / 3) + cos (x / 3)

theorem min_positive_period_f : ∃ T > 0, (∀ x, f (x + T) = f x) ∧ ∀ T', (T' > 0 ∧ ∀ x, f (x + T') = f x) → T ≤ T' :=
  sorry

theorem max_value_f : ∃ M, (∀ x, f x ≤ M) ∧ (∀ ε > 0, ∃ x, M - ε < f x) ∧ M = sqrt 2 :=
  sorry

end min_positive_period_f_max_value_f_l33_33069


namespace tetrahedron_volume_l33_33800

variable (a b c α β γ : ℝ)
variable (ω : ℝ := (α + β + γ) / 2)

theorem tetrahedron_volume (a b c α β γ : ℝ) (hω : ω = (α + β + γ) / 2) :
  let ω = (α + β + γ) / 2 in 
  V = (a * b * c / 3) * sqrt (sin ω * sin (ω - α) * sin (ω - β) * sin (ω - γ)) :=
sorry

end tetrahedron_volume_l33_33800


namespace attendees_gift_exchange_l33_33265

theorem attendees_gift_exchange (x : ℕ) (h1 : 56 = x * (x - 1) / 2) : 
  x * (x - 1) = 112 :=
by
  sorry

end attendees_gift_exchange_l33_33265


namespace largest_root_of_quadratic_l33_33971

theorem largest_root_of_quadratic :
  ∀ x : ℝ, (9 * x^2 - 45 * x + 50 = 0) → x = 10 / 3 ∨ x = 5 / 3 → x = 10 / 3 :=
by
  intros x h_eq h_roots
  cases h_roots with h_root1 h_root2
  · exact h_root1
  · sorry

end largest_root_of_quadratic_l33_33971


namespace christopher_stroll_time_l33_33680

def distance := 5
def speed := 4
def time := distance / speed

theorem christopher_stroll_time : time = 1.25 := 
by
  sorry

end christopher_stroll_time_l33_33680


namespace triangles_same_fermat_length_l33_33833

noncomputable def fermat_length {A B C D E F : ℝ} (R : ℝ) : Prop :=
(A + D = 120) ∧ (B + E = 120) ∧
∀ (circumradius : ℝ), circumradius = R →
3 * circumradius = 3 * circumradius

theorem triangles_same_fermat_length {A B C D E F : ℝ} (R : ℝ) :
  (A + D = 120) → (B + E = 120) → 
  (∀ circumradius, circumradius = R) → 
  fermat_length R :=
by
  intros h1 h2 h3
  unfold fermat_length
  split; assumption
  sorry

end triangles_same_fermat_length_l33_33833


namespace problem1_problem2_l33_33222

-- Problem 1
theorem problem1 (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) : a^2 / b ≥ 2 * a - b :=
by sorry

-- Problem 2
theorem problem2 (a b : ℝ) : a^2 + b^2 + 3 ≥ a * b + Real.sqrt 3 * (a + b) :=
by sorry

end problem1_problem2_l33_33222


namespace multiply_expression_l33_33969

variable (y : ℝ)

theorem multiply_expression : 
  (16 * y^3) * (12 * y^5) * (1 / (4 * y)^3) = 3 * y^5 := by
  sorry

end multiply_expression_l33_33969


namespace least_prime_factor_of_5_pow_5_minus_5_pow_4_l33_33980

theorem least_prime_factor_of_5_pow_5_minus_5_pow_4 : (∃ p : ℕ, p.prime ∧ p ≤ 5 - 1 ∧ p ∣ (5^5 - 5^4)) :=
  sorry

end least_prime_factor_of_5_pow_5_minus_5_pow_4_l33_33980


namespace find_complex_point_C_l33_33805

noncomputable def complex_point_C 
  (A B C : ℂ) 
  (A_val : A = 2 + complex.i)
  (BA_val : B - A = 1 + 2 * complex.i)
  (BC_val : C - B = 3 - complex.i) : Prop :=
  C = 4 - 2 * complex.i

theorem find_complex_point_C (A B C : ℂ) 
  (A_val : A = 2 + complex.i) 
  (BA_val : B - A = 1 + 2 * complex.i) 
  (BC_val : C - B = 3 - complex.i) :
  complex_point_C A B C A_val BA_val BC_val :=
sorry

end find_complex_point_C_l33_33805


namespace real_solutions_unique_l33_33702

theorem real_solutions_unique (a b c : ℝ) :
  (2 * a - b = a^2 * b ∧ 2 * b - c = b^2 * c ∧ 2 * c - a = c^2 * a) →
  (a, b, c) = (-1, -1, -1) ∨ (a, b, c) = (0, 0, 0) ∨ (a, b, c) = (1, 1, 1) :=
by
  sorry

end real_solutions_unique_l33_33702


namespace a_value_for_even_function_l33_33017

def f (x a : ℝ) := (x + 1) * (x + a)

theorem a_value_for_even_function (a : ℝ) (h : ∀ x, f x a = f (-x) a) : a = -1 :=
by
  sorry

end a_value_for_even_function_l33_33017


namespace height_percentage_of_new_cylinder_l33_33233

noncomputable def cylinder_height_percentage (h r : ℝ) (pi : ℝ) :=
  let V_orig := π * r^2 * h
  let V_water := (5 / 6) * V_orig
  let r_new := 1.25 * r
  let V_new := π * (r_new)^2 * h_new
  let h_new := (5 / 6) * h / (3 / 5 * (1.25^2))
  (h_new / h) * 100

theorem height_percentage_of_new_cylinder
  (h r : ℝ) (h_new : ℝ) (pi : ℝ) 
  (water_full  : (5 / 6) * π * r^2 * h = (3 / 5) * π * (1.25 * r)^2 * h_new) :
  cylinder_height_percentage h r pi = 71.11 := 
by
  sorry

end height_percentage_of_new_cylinder_l33_33233


namespace pond_area_l33_33656

theorem pond_area (P G : ℝ) (hP : P = 48) (hG : G = 124) : ∃ A, A = 20 :=
by
  let side_length := P / 4
  let total_area := side_length * side_length
  let pond_area := total_area - G
  use pond_area
  sorry

end pond_area_l33_33656


namespace apollonius_circle_area_l33_33668

-- Define points P and Q in the plane as given
def P : ℝ × ℝ := (1, 0)
def Q : ℝ × ℝ := (-1, 0)

-- Define the condition for moving point M
def condition (M : ℝ × ℝ) : Prop := 
  real.sqrt ((M.1 - P.1)^2 + M.2^2) = real.sqrt 2 * real.sqrt ((M.1 + Q.1)^2 + M.2^2)

-- Statement of the theorem
theorem apollonius_circle_area :
  (∀ M : ℝ × ℝ, condition M → 
  let r := real.sqrt 8 in 
  let area := real.pi * r^2 in 
  area = 8 * real.pi) := sorry

end apollonius_circle_area_l33_33668


namespace smallest_integer_from_operations_l33_33446

theorem smallest_integer_from_operations : ∃ (e : String), eval_expr e = 3 :=
by
  -- Nine 1's
  let nums := "111111111"

  -- Insert exactly two division signs
  have num_divisions : ∃ d, d = 2,
    sorry

  -- Insert exactly two addition signs
  have num_additions : ∃ a, a = 2,
    sorry

  -- Combined expression satisfying the conditions yields 3
  let e := "111 ÷ 111 + 1 ÷ 1 + 1"
  have eval_e : eval_expr e = 3,
    sorry

  exact ⟨e, eval_e⟩

-- Helper function to evaluate the string expressions (assumed defined elsewhere)
noncomputable def eval_expr (s : String) : Int :=
  sorry

end smallest_integer_from_operations_l33_33446


namespace min_positive_period_max_value_l33_33109
open Real

noncomputable def f (x : ℝ) : ℝ := sin (x / 3) + cos (x / 3)

theorem min_positive_period_max_value :
  (∃ T > 0, ∀ x, f(x + T) = f(x) ∧ T = 6 * π) ∧
  (∀ x, f(x) ≤ sqrt 2) ∧ 
  (∃ x, f(x) = sqrt 2) :=
by
  sorry

end min_positive_period_max_value_l33_33109


namespace max_value_of_f_l33_33299

noncomputable def f (t : ℝ) : ℝ :=
  ((3:ℝ)^t - 4 * t) * t / (9:ℝ)^t

theorem max_value_of_f : ∃ t : ℝ, f(t) = 1 / 16 :=
sorry

end max_value_of_f_l33_33299


namespace minimum_period_and_max_value_of_f_l33_33047

noncomputable def f (x : ℝ) : ℝ := sin (x / 3) + cos (x / 3)

theorem minimum_period_and_max_value_of_f :
  (∃ T > 0, ∀ x: ℝ, f (x + T) = f x) ∧
  (∀ x: ℝ, f x ≤ sqrt 2 ∧ 
          (∃ y: ℝ, f y = sqrt 2)) :=
by 
  sorry

end minimum_period_and_max_value_of_f_l33_33047


namespace count_valid_sequences_l33_33854

open Nat

def valid_sequence (l : List ℕ) : Prop :=
  l.length = 20 ∧ (∀ i, 2 ≤ i ∧ i ≤ 20 → (l.nth (i - 1) + 1 ∈ l.take (i - 1) ∨ l.nth (i - 1) - 1 ∈ l.take (i - 1)))

theorem count_valid_sequences : 
  (countp valid_sequence (List.permutations (List.range 1 21)) = 524288) :=
sorry

end count_valid_sequences_l33_33854


namespace symmetric_point_x_axis_l33_33780

theorem symmetric_point_x_axis (a b : ℝ) :
  (∃ a b, Q(a, b)) →
  P(a, b) :=
sorry

end symmetric_point_x_axis_l33_33780


namespace inequality_proof_l33_33336

noncomputable def a : ℝ := 2 ^ (Real.log 3 / Real.log 2)
noncomputable def b : ℝ := Real.log 3 - 2
noncomputable def c : ℝ := Real.pi ^ (-Real.exp 1)

theorem inequality_proof : a > c ∧ c > b :=
by {
  have h1 : a = 3,
  { have h : (2:ℝ) ^ (Real.log 3 / Real.log 2) = 3,
    { rw [←Real.log_inv_log₃_of_pos2, Real.exp_log],
      exact_mod_cast Real.log_pos 3 (show (3:ℝ) > 0, by norm_num) },
    exact h },

  have h2 : b < 0,
  { have h : Real.log 3 < 2 := by sorry,
    exact sub_lt_zero.mpr h },

  have h3 : 0 < c ∧ c < 1,
  { have h : Real.pi > 1 := by sorry,
    have h_neg : -Real.exp 1 < 0 := by sorry,
    have h_pos : 0 < Real.pi ^ (-Real.exp 1) := by sorry,
    have h_lt : Real.pi ^ (-Real.exp 1) < 1 := by sorry,
    exact ⟨h_pos, h_lt⟩ },

  have h4 : a > c := by linarith,
  have h5 : c > b := by linarith,
  exact ⟨h4, h5⟩
}


end inequality_proof_l33_33336


namespace min_positive_period_and_max_value_of_f_l33_33090

noncomputable def f (x : ℝ) : ℝ := sin (x / 3) + cos (x / 3)

theorem min_positive_period_and_max_value_of_f :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ T = 6 * π) ∧ 
  (∃ x, f x = sqrt 2) :=
by {
  sorry
}

end min_positive_period_and_max_value_of_f_l33_33090


namespace sin_x_odd_cos_x_even_tan_x_odd_sin_2x_odd_even_function_among_options_is_cos_x_l33_33510

-- Definitions of each option as functions
def sin_x (x : ℝ) : ℝ := Real.sin x
def cos_x (x : ℝ) : ℝ := Real.cos x
def tan_x (x : ℝ) : ℝ := Real.tan x
def sin_2x (x : ℝ) : ℝ := Real.sin (2 * x)

-- Conditions about the nature of each function
theorem sin_x_odd : ∀ x : ℝ, sin_x (-x) = -sin_x x :=
  by sorry

theorem cos_x_even : ∀ x : ℝ, cos_x (-x) = cos_x x :=
  by sorry

theorem tan_x_odd : ∀ x ∈ {x | x ≠ k * π + π / 2, k : ℤ } , tan_x (-x) = -tan_x x :=
  by sorry

theorem sin_2x_odd : ∀ x : ℝ, sin_2x (-x) = -sin_2x x :=
  by sorry

-- Main theorem: identifying the even function among the given options
theorem even_function_among_options_is_cos_x :
  (∀ f, f = sin_x → false) ∧
  (∀ f, f = cos_x → true) ∧
  (∀ f, f = tan_x → false) ∧
  (∀ f, f = sin_2x → false) :=
  by sorry

end sin_x_odd_cos_x_even_tan_x_odd_sin_2x_odd_even_function_among_options_is_cos_x_l33_33510


namespace min_positive_period_max_value_l33_33111
open Real

noncomputable def f (x : ℝ) : ℝ := sin (x / 3) + cos (x / 3)

theorem min_positive_period_max_value :
  (∃ T > 0, ∀ x, f(x + T) = f(x) ∧ T = 6 * π) ∧
  (∀ x, f(x) ≤ sqrt 2) ∧ 
  (∃ x, f(x) = sqrt 2) :=
by
  sorry

end min_positive_period_max_value_l33_33111


namespace range_of_a_l33_33547

theorem range_of_a (A : Set ℝ) (a : ℝ) (h1 : ∀ x ∈ A, log a (a - x^2 / 2) > log a (a - x)) 
  (h2 : A ∩ Set.univ.metric.z = {1}) : a ∈ Ioi 1 :=
sorry

end range_of_a_l33_33547


namespace midpoint_kn_distance_to_lm_l33_33169

theorem midpoint_kn_distance_to_lm (K L M N : Point) (h_cyclic: CyclicQuadrilateral K L M N)
  (h_side_MN : dist M N = 6)
  (h_side_KL : dist K L = 2)
  (h_side_LM : dist L M = 5)
  (h_perpendicular: Perpendicular (line_segment K M) (line_segment L N)):
  dist (midpoint K N) (line L M) = 3.44 := sorry

end midpoint_kn_distance_to_lm_l33_33169


namespace age_of_youngest_child_l33_33936

theorem age_of_youngest_child (x : ℕ) 
  (h : x + (x + 3) + (x + 6) + (x + 9) + (x + 12) = 65) : x = 7 :=
sorry

end age_of_youngest_child_l33_33936


namespace payment_for_30_kilograms_l33_33228

-- Define the price calculation based on quantity x
def payment_amount (x : ℕ) : ℕ :=
  if x ≤ 10 then 20 * x
  else 16 * x + 40

-- Prove that for x = 30, the payment amount y equals 520
theorem payment_for_30_kilograms : payment_amount 30 = 520 := by
  sorry

end payment_for_30_kilograms_l33_33228


namespace gathering_gift_exchange_l33_33261

def number_of_guests (x : ℕ) : Prop :=
  x * (x - 1) = 56

theorem gathering_gift_exchange :
  ∃ x : ℕ, number_of_guests x :=
sorry

end gathering_gift_exchange_l33_33261


namespace intersection_P_Q_l33_33392

-- Define the sets P and Q
def P : Set ℝ := {-1, 0, 1}
def Q : Set ℝ := {x | -1 ≤ x ∧ x < 1}

-- The proof statement
theorem intersection_P_Q : P ∩ Q = {-1, 0} :=
by
  sorry

end intersection_P_Q_l33_33392


namespace tenth_pair_in_twentieth_row_l33_33394

def nth_pair_in_row (n k : ℕ) : ℕ × ℕ :=
  if h : n > 0 ∧ k > 0 ∧ n >= k then (k, n + 1 - k)
  else (0, 0) -- define (0,0) as a default for invalid inputs

theorem tenth_pair_in_twentieth_row : nth_pair_in_row 20 10 = (10, 11) :=
by sorry

end tenth_pair_in_twentieth_row_l33_33394


namespace factorial_mod_5_l33_33312

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem factorial_mod_5 :
  (factorial 1 + factorial 2 + factorial 3 + factorial 4 + factorial 5 +
   factorial 6 + factorial 7 + factorial 8 + factorial 9 + factorial 10) % 5 = 3 :=
by
  sorry

end factorial_mod_5_l33_33312


namespace min_period_and_max_value_l33_33125

def f (x : ℝ) := Real.sin (x / 3) + Real.cos (x / 3)

theorem min_period_and_max_value :
  (∀ T > 0, T ≠ 6 * Real.pi → ¬∀ x : ℝ, f(x + T) = f(x)) ∧
  ∃ x : ℝ, f(x) = sqrt 2 :=
by
  sorry

end min_period_and_max_value_l33_33125


namespace area_of_triangle_formed_by_line_l33_33899

-- Define the line equation
def line_eq (x : ℝ) : ℝ := x + 3

-- Define the function to compute the area of the triangle
def triangle_area (base height : ℝ) : ℝ := (1 / 2) * base * height

-- The theorem statement
theorem area_of_triangle_formed_by_line :
  let base := 3
  let height := 3
  triangle_area base height = 9 / 2 :=
by
  -- This is where the proof would go
  sorry

end area_of_triangle_formed_by_line_l33_33899


namespace bishop_knight_count_l33_33014

-- Definitions
def bishop_knight_condition_1 (B K : Type) (on_same_diagonal : B → K → Prop) : Prop :=
  ∀ (b : B), ∃ (k : K), on_same_diagonal b k

def bishop_knight_condition_2 (B K : Type) (distance_sqrt5 : B → K → Prop) : Prop :=
  ∀ (k : K), ∃ (b : B), distance_sqrt5 b k

def not_satisfied_if_removed (B K : Type) (cond1 : B → Prop) (cond2 : K → Prop) : Prop :=
  ∀ (b : B) (k : K), (¬ cond1 b) ∨ (¬ cond2 k)

-- Theorem statement
theorem bishop_knight_count (B K : Type)
    (on_same_diagonal : B → K → Prop)
    (distance_sqrt5 : B → K → Prop)
    (h1 : bishop_knight_condition_1 B K on_same_diagonal)
    (h2 : bishop_knight_condition_2 B K distance_sqrt5)
    (h3 : not_satisfied_if_removed B K (λ b, ∃ k, on_same_diagonal b k) (λ k, ∃ b, distance_sqrt5 b k)) :
    ∃ k : ℕ, n = 4 * k :=
by
  sorry

end bishop_knight_count_l33_33014


namespace movie_theater_attendance_l33_33179

theorem movie_theater_attendance : 
  let total_seats := 750
  let empty_seats := 218
  let people := total_seats - empty_seats
  people = 532 :=
by
  sorry

end movie_theater_attendance_l33_33179


namespace probability_of_heads_or_five_tails_is_one_eighth_l33_33500

namespace coin_flip

def num_heads_or_at_least_five_tails : ℕ :=
1 + 6 + 1

def total_outcomes : ℕ :=
2^6

def probability_heads_or_five_tails : ℚ :=
num_heads_or_at_least_five_tails / total_outcomes

theorem probability_of_heads_or_five_tails_is_one_eighth :
  probability_heads_or_five_tails = 1 / 8 := by
  sorry

end coin_flip

end probability_of_heads_or_five_tails_is_one_eighth_l33_33500


namespace ratio_of_ages_l33_33829

variable (J L M : ℕ)

def louis_age := L = 14
def matilda_age := M = 35
def matilda_older := M = J + 7
def jerica_multiple := ∃ k : ℕ, J = k * L

theorem ratio_of_ages
  (hL : louis_age L)
  (hM : matilda_age M)
  (hMO : matilda_older J M)
  : J / L = 2 :=
by
  sorry

end ratio_of_ages_l33_33829


namespace age_ratio_in_77_years_l33_33183

-- Definitions based on the conditions from part a)
variables (m a x : ℕ)

-- Conditions converted into Lean expressions
def condition1 := m - 3 = 4 * (a - 3)
def condition2 := m - 7 = 5 * (a - 7)
def ratio_condition := (m + x) * 2 = (a + x) * 3

-- Proposition to prove
theorem age_ratio_in_77_years (h1 : condition1) (h2 : condition2) : x = 77 :=
by {
  -- Insert missing proof steps
  sorry
}

end age_ratio_in_77_years_l33_33183


namespace people_to_left_of_kolya_l33_33530

theorem people_to_left_of_kolya (people_right_kolya people_left_sasha people_right_sasha : ℕ) (total_people : ℕ) :
  (people_right_kolya = 12) →
  (people_left_sasha = 20) →
  (people_right_sasha = 8) →
  (total_people = people_left_sasha + people_right_sasha + 1) →
  total_people - people_right_kolya - 1 = 16 :=
begin
  sorry
end

end people_to_left_of_kolya_l33_33530


namespace range_of_f_l33_33171

noncomputable def f (x : ℝ) : ℤ :=
  Int.floor (sin (2 * x)) + Int.floor (real.sqrt 2 * sin (x + real.pi / 4))

theorem range_of_f : (set.range f) = { -2, -1, 0, 1, 2 } :=
by
  sorry

end range_of_f_l33_33171


namespace renovate_total_time_eq_79_5_l33_33953

-- Definitions based on the given conditions
def time_per_bedroom : ℝ := 4
def num_bedrooms : ℕ := 3
def time_per_kitchen : ℝ := time_per_bedroom * 1.5
def time_per_garden : ℝ := 3
def time_per_terrace : ℝ := time_per_garden - 2
def time_per_basement : ℝ := time_per_kitchen * 0.75

-- Total time excluding the living room
def total_time_excl_living_room : ℝ :=
  (num_bedrooms * time_per_bedroom) +
  time_per_kitchen +
  time_per_garden +
  time_per_terrace +
  time_per_basement

-- Time for the living room
def time_per_living_room : ℝ := 2 * total_time_excl_living_room

-- Total time for everything
def total_time : ℝ := total_time_excl_living_room + time_per_living_room

-- The theorem we need to prove
theorem renovate_total_time_eq_79_5 : total_time = 79.5 := by
  sorry

end renovate_total_time_eq_79_5_l33_33953


namespace problem1_problem2_period_problem3_min_value_l33_33377

-- Define the function f(x) and the given conditions
def f (x : ℝ) : ℝ := 2 * (Real.cos x) ^ 2 + 2 * (Real.sin x) * (Real.cos x)

-- Problem 1: Prove that f(π / 8) = √2 + 1
theorem problem1 : f (Real.pi / 8) = Real.sqrt 2 + 1 :=
by 
  sorry

-- Problem 2: Prove the smallest positive period of f(x) is π
theorem problem2_period : ∃ T > 0, (∀ x, f (x + T) = f x) ∧ T = Real.pi :=
by 
  sorry

-- Problem 3: Prove the minimum value of f(x) is 1 - √2
theorem problem3_min_value : ∃ x, f x = 1 - Real.sqrt 2 :=
by 
  sorry

end problem1_problem2_period_problem3_min_value_l33_33377


namespace min_positive_period_max_value_l33_33116
open Real

noncomputable def f (x : ℝ) : ℝ := sin (x / 3) + cos (x / 3)

theorem min_positive_period_max_value :
  (∃ T > 0, ∀ x, f(x + T) = f(x) ∧ T = 6 * π) ∧
  (∀ x, f(x) ≤ sqrt 2) ∧ 
  (∃ x, f(x) = sqrt 2) :=
by
  sorry

end min_positive_period_max_value_l33_33116


namespace find_positive_integers_l33_33211

theorem find_positive_integers (m : ℕ) :
  (∀ (α β : ℤ), α ≠ 0 → β ≠ 0 → 
    (2^m * α^m - (α + β)^m - (α - β)^m) % (3 * α^2 + β^2) = 0) ↔ (m % 6 = 1 ∨ m % 6 = 5) := 
begin
  sorry
end

end find_positive_integers_l33_33211


namespace min_period_and_max_value_l33_33059

def f (x : ℝ) : ℝ := Real.sin (x / 3) + Real.cos (x / 3)

noncomputable def minimum_positive_period (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x : ℝ, f (x + T) = f x ∧ T > 0 ∧ ∀ T' : ℝ, (T' > 0 ∧ ∀ x : ℝ, f (x + T') = f x) → T' ≥ T

noncomputable def is_max_value (f : ℝ → ℝ) (M : ℝ) : Prop :=
  ∀ x : ℝ, f x ≤ M ∧ ∃ x : ℝ, f x = M

theorem min_period_and_max_value :
  minimum_positive_period f (6 * Real.pi) ∧ is_max_value f (Real.sqrt 2) := by
  sorry

end min_period_and_max_value_l33_33059


namespace min_positive_period_f_max_value_f_l33_33072

noncomputable def f (x : ℝ) : ℝ := sin (x / 3) + cos (x / 3)

theorem min_positive_period_f : ∃ T > 0, (∀ x, f (x + T) = f x) ∧ ∀ T', (T' > 0 ∧ ∀ x, f (x + T') = f x) → T ≤ T' :=
  sorry

theorem max_value_f : ∃ M, (∀ x, f x ≤ M) ∧ (∀ ε > 0, ∃ x, M - ε < f x) ∧ M = sqrt 2 :=
  sorry

end min_positive_period_f_max_value_f_l33_33072


namespace eggs_per_dozen_l33_33475

theorem eggs_per_dozen (total_pounds required_pounds: ℝ) (egg_weight: ℝ) (dozen: ℕ) :
  (required_pounds = 6) ∧ (egg_weight = 1/16) ∧ (dozen = 12) →
  (required_pounds / egg_weight / dozen = 8) :=
by
  intros h
  cases h with h1 h
  cases h with h2 h3
  sorry

end eggs_per_dozen_l33_33475


namespace largest_angle_is_135_degrees_l33_33761

-- Conditions: Defining the side lengths of the triangle
def side_a : ℝ := 3 * Real.sqrt 2
def side_b : ℝ := 6
def side_c : ℝ := 3 * Real.sqrt 10

-- Main statement
theorem largest_angle_is_135_degrees
  (a b c : ℝ)
  (ha : a = side_a)
  (hb : b = side_b)
  (hc : c = side_c)
  (htri : a + b > c ∧ a + c > b ∧ b + c > a) :
  ∃ θ : ℝ, θ = 135 ∧ angle_is_max (a = a ∧ b = b ∧ c = c) θ :=
by
  sorry

end largest_angle_is_135_degrees_l33_33761


namespace part1_part2_l33_33425

variables {A B C a b c : ℝ}
variables (α β γ : ℝ) -- angles
variables (R : ℝ) -- circumradius

-- Given conditions
axiom cond1 : c = b * (1 + 2 * Real.cos A)
axiom sum_angles : A + B + C = Real.pi

-- Statements to prove
theorem part1 : cond1 → A = 2 * B :=
by
  sorry

theorem part2 (ha : a = 3) (hb : B = Real.pi / 6) : 
  cond1 → (∃ A C, area_triangle a b c = 3 * Real.sqrt 3 / 2) :=
by
  sorry

end part1_part2_l33_33425


namespace coefficient_x4_correct_l33_33902

-- Define the expression
def expression (x : ℝ) := (4 * x^2 + 6 * x + (9 / 4))^4

-- Define the target coefficient for x^4
def coefficient_of_x4 := 4374

-- Statement to prove
theorem coefficient_x4_correct : (expression x).coeff (monomial 4) = coefficient_of_x4 := by
  sorry

end coefficient_x4_correct_l33_33902


namespace min_period_and_max_value_l33_33128

def f (x : ℝ) := Real.sin (x / 3) + Real.cos (x / 3)

theorem min_period_and_max_value :
  (∀ T > 0, T ≠ 6 * Real.pi → ¬∀ x : ℝ, f(x + T) = f(x)) ∧
  ∃ x : ℝ, f(x) = sqrt 2 :=
by
  sorry

end min_period_and_max_value_l33_33128


namespace min_pos_period_max_value_l33_33147

noncomputable def f (x : ℝ) := Real.sin (x / 3) + Real.cos (x / 3)

theorem min_pos_period_max_value :
  (∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ T) ∧ T = 6 * Real.pi) ∧ 
  (∀ x, f x ≤ sqrt 2 ∧ (∃ x, f x = sqrt 2)) :=
by
  sorry

end min_pos_period_max_value_l33_33147


namespace velvet_needed_for_box_l33_33477

theorem velvet_needed_for_box : 
  let area_long_side := 8 * 6
  let area_short_side := 5 * 6
  let area_top_bottom := 40
  let total_area := (2 * area_long_side) + (2 * area_short_side) + (2 * area_top_bottom)
  total_area = 236 :=
by
  sorry

end velvet_needed_for_box_l33_33477


namespace journey_total_distance_l33_33208

theorem journey_total_distance (s1 s2 t1 t_total : ℕ) 
    (hs1 : s1 = 40) (hs2 : s2 = 60) (ht1 : t1 = 3) (ht_total : t_total = 5) :
    let d1 := s1 * t1,
        t2 := t_total - t1,
        d2 := s2 * t2,
        d_total := d1 + d2
    in d_total = 240 := 
by
  let d1 := s1 * t1
  let t2 := t_total - t1
  let d2 := s2 * t2
  let d_total := d1 + d2
  have hs1 : s1 = 40 := hs1
  have hs2 : s2 = 60 := hs2
  have ht1 : t1 = 3 := ht1
  have ht_total : t_total = 5 := ht_total
  rw [hs1, hs2, ht1, ht_total]
  simp only [mul_add, mul_sub, mul_one, mul_two]
  simp only [add_comm, add_left_comm, add_assoc]
  exact rfl

end journey_total_distance_l33_33208


namespace min_people_pictured_l33_33429

theorem min_people_pictured (n : ℕ) (photos : Fin n → α) (middle_man : α → α) (right_brother : α → α) (left_son : α → α)
  (h_unique_middle: ∀ i j, i ≠ j → middle_man (photos i) ≠ middle_man (photos j)) :
  n = 10 → (least_possible_total_people : ℕ) := 16 :=
by sorry

end min_people_pictured_l33_33429


namespace min_value_of_g_function_l33_33320

noncomputable def g (x : Real) := x + (x + 1) / (x^2 + 1) + (x * (x + 3)) / (x^2 + 3) + (3 * (x + 1)) / (x * (x^2 + 3))

theorem min_value_of_g_function : ∀ x : ℝ, x > 0 → g x ≥ 3 := sorry

end min_value_of_g_function_l33_33320


namespace stock_rise_in_morning_l33_33695

theorem stock_rise_in_morning (x : ℕ) (V : ℕ → ℕ) (h0 : V 0 = 100)
  (h100 : V 100 = 200) (h_recurrence : ∀ n, V n = 100 + n * x - n) :
  x = 2 :=
  by
  sorry

end stock_rise_in_morning_l33_33695


namespace sum_a4_a5_a6_l33_33900

section ArithmeticSequence

variable {a : ℕ → ℝ}

-- Condition 1: The sequence is arithmetic
def is_arithmetic_sequence (a : ℕ → ℝ) :=
  ∀ (n : ℕ), a (n + 1) - a n = a 1 - a 0

-- Condition 2: Given information
axiom a2_a8_eq_6 : a 2 + a 8 = 6

-- Question: Prove that a 4 + a 5 + a 6 = 9
theorem sum_a4_a5_a6 : is_arithmetic_sequence a → a 4 + a 5 + a 6 = 9 :=
by
  intro h_arith
  sorry

end ArithmeticSequence

end sum_a4_a5_a6_l33_33900


namespace inequality_partition_l33_33465

noncomputable def sum_x (x : Fin n → ℝ) : ℝ := 
  ∑ i in Finset.range n, x i

def isValidPartition (n : ℕ) (x : ℕ → ℝ) : Prop := 
  (n > 0) ∧ 
  (x 0 = 0) ∧ 
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ n → x i > 0) ∧ 
  (sum_x x = 1)

theorem inequality_partition (n : ℕ) (x : ℕ → ℝ) (h : isValidPartition n x) :
  1 ≤ ∑ i in Finset.range n, (x i) / (Real.sqrt (1 + ∑ j in Finset.range i, x j) * Real.sqrt (∑ j in Finset.Icc i n, x j)) 
  ∧ 
  ∑ i in Finset.range n, (x i) / (Real.sqrt (1 + ∑ j in Finset.range i, x j) * Real.sqrt (∑ j in Finset.Icc i n, x j)) < Real.pi / 2 := 
sorry

end inequality_partition_l33_33465


namespace K_equals_3H_l33_33455

-- Define the logarithmic function used in the condition
def H (x : ℝ) : ℝ := log ((2 + x) / (2 - x))

-- Define the transformation of x
def transform (x : ℝ) : ℝ := (4 * x - x ^ 3) / (1 + 4 * x ^ 2)

-- Define the function K by substituting x in H with the transformed x
def K (x : ℝ) : ℝ := log ((2 + (transform x)) / (2 - (transform x)))

-- The theorem statement to prove
theorem K_equals_3H (x : ℝ) : K x = 3 * H x :=
by
  sorry

end K_equals_3H_l33_33455


namespace least_expensive_cost_proof_l33_33877

noncomputable def DF := 4000
noncomputable def DE := 4500
noncomputable def bus_cost_per_km := 0.20
noncomputable def airplane_cost_per_km := 0.12
noncomputable def airplane_booking_fee := 120

-- Calculate EF using the Pythagorean theorem
noncomputable def EF := real.sqrt (DE^2 - DF^2)

-- Cost calculation functions:
noncomputable def bus_cost (distance : ℝ) : ℝ := distance * bus_cost_per_km
noncomputable def airplane_cost (distance : ℝ) : ℝ := distance * airplane_cost_per_km + airplane_booking_fee

-- Least expensive method calculation
noncomputable def cheapest_DE := min (bus_cost DE) (airplane_cost DE)
noncomputable def cheapest_EF := min (bus_cost EF) (airplane_cost EF)
noncomputable def cheapest_FD := min (bus_cost DF) (airplane_cost DF)

-- Total cost of the least expensive method
noncomputable def total_least_expensive_cost := cheapest_DE + cheapest_EF + cheapest_FD

theorem least_expensive_cost_proof : total_least_expensive_cost = 1627.44 := by
  sorry

end least_expensive_cost_proof_l33_33877


namespace evaluate_polynomial_at_neg2_l33_33691

theorem evaluate_polynomial_at_neg2 : 2 * (-2)^4 + 3 * (-2)^3 + 5 * (-2)^2 + (-2) + 4 = 30 :=
by 
  sorry

end evaluate_polynomial_at_neg2_l33_33691


namespace pete_flag_total_circles_squares_l33_33896

def US_flag_stars : ℕ := 50
def US_flag_stripes : ℕ := 13

def circles (stars : ℕ) : ℕ := (stars / 2) - 3
def squares (stripes : ℕ) : ℕ := (2 * stripes) + 6

theorem pete_flag_total_circles_squares : 
  circles US_flag_stars + squares US_flag_stripes = 54 := 
by
  unfold circles squares US_flag_stars US_flag_stripes
  sorry

end pete_flag_total_circles_squares_l33_33896


namespace area_WXYZ_l33_33834

structure Point :=
(x : ℝ)
(y : ℝ)

def midpoint (p1 p2 : Point) : Point :=
{ x := (p1.x + p2.x) / 2,
  y := (p1.y + p2.y) / 2 }

noncomputable def area_quadrilateral (p1 p2 p3 p4 : Point) : ℝ :=
(abs ((p1.x * p2.y + p2.x * p3.y + p3.x * p4.y + p4.x * p1.y) - 
      (p1.y * p2.x + p2.y * p3.x + p3.y * p4.x + p4.y * p1.x))) / 2

theorem area_WXYZ :
  let A := Point.mk 0 12 in  -- Given b = 12
  let B := Point.mk 0 0 in
  let C := Point.mk 8 0 in  -- Given c = 8
  let E := Point.mk 0 (3 * 12 / 4) in
  let F := Point.mk (3 * 8 / 4) 0 in
  let O := midpoint A C in
  let W := midpoint E B in
  let X := midpoint F B in
  let Y := midpoint F O in
  let Z := midpoint O E in
  area_quadrilateral W X Y Z = 18 :=
by
  sorry

end area_WXYZ_l33_33834


namespace never_repeat_except_one_l33_33638

theorem never_repeat_except_one (a b c d : ℝ) (h_pos: a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) :
  ¬ (∃ n, generate_set_n_times n (a, b, c, d) = (a, b, c, d)) ∨ a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 :=
sorry

end never_repeat_except_one_l33_33638


namespace find_x_cubed_l33_33503

-- Definitions based on the conditions provided
def varies_inversely (x : ℝ) (y : ℝ) (k : ℝ) : Prop :=
  (x^3) * (y^2) = k

def k_value : ℝ := (4^3) * (2^2)  -- Calculated as 256

-- Main theorem to state the problem
theorem find_x_cubed (x y : ℝ) : varies_inversely x y k_value → y = 8 → x^3 = 4 :=
  begin
    sorry,
  end

end find_x_cubed_l33_33503


namespace complex_value_l33_33853

-- Given a complex number z
variable (z : ℂ)
-- Given condition as an assumption
variable (h : 15 * complex.abs z ^ 2 = 5 * complex.abs (z + 1) ^ 2 + complex.abs (z ^ 2 - 1) ^ 2 + 44)

-- Proof problem to show that z² + 36/z² = 60
theorem complex_value (z : ℂ) (h : 15 * complex.abs z ^ 2 = 5 * complex.abs (z + 1) ^ 2 + complex.abs (z ^ 2 - 1) ^ 2 + 44) :
  z ^ 2 + 36 / z ^ 2 = 60 := 
by
  sorry

end complex_value_l33_33853


namespace remainder_1625_mul_1627_mul_1629_mod_12_l33_33610

theorem remainder_1625_mul_1627_mul_1629_mod_12 :
  let a := 1625
      b := 1627
      c := 1629
      n := 12
      ra := 5
      rb := 7
      rc := 9
  in a % n = ra ∧ b % n = rb ∧ c % n = rc → ((a * b * c) % n = 3) :=
by
  intros
  sorry

end remainder_1625_mul_1627_mul_1629_mod_12_l33_33610


namespace tank_fill_rate_l33_33967

theorem tank_fill_rate
  (length width depth : ℝ)
  (time_to_fill : ℝ)
  (h_length : length = 10)
  (h_width : width = 6)
  (h_depth : depth = 5)
  (h_time : time_to_fill = 60) : 
  (length * width * depth) / time_to_fill = 5 :=
by
  -- Proof would go here
  sorry

end tank_fill_rate_l33_33967


namespace perpendicular_bisector_intersects_external_angle_bisector_on_circumcircle_l33_33009

theorem perpendicular_bisector_intersects_external_angle_bisector_on_circumcircle
  {A B C D E : Type*} [Inhabited D] [Inhabited E]
  (circumcircle : Set (D))
  (is_midpoint : (D = ((B + C) / 2))) 
  (is_on_circumcircle : D ∈ circumcircle ∧ E ∈ circumcircle 
    ∧ ∃ line : Type*, line = segment (perpendicular_bisector B C) ∧ line ∈ circumcircle
    ∧ E = point_on_circle line circumcircle)
  (triangle : triangle A B C)
  (perpendicular_bisector : ∀ {x y : Type*}, x, y ≠ 0 → x • (1 / ∥x∥) ⊤ y • (1 / ∥x∥) = 0)
  : (exists O : Type*, is_external_bisector A circumcircle ∧ (external_angle_bisector(traingle A B C)) ∈ circumcircle) → 
    (perpendicular_bisector(BC)) ∈ circumcircle := sorry

end perpendicular_bisector_intersects_external_angle_bisector_on_circumcircle_l33_33009


namespace gift_exchange_equation_l33_33270

theorem gift_exchange_equation
  (x : ℕ)
  (total_gifts : ℕ)
  (H : total_gifts = 56)
  (H1 : 2 * total_gifts = x * (x - 1)) :
  x * (x - 1) = 56 :=
by
  sorry

end gift_exchange_equation_l33_33270


namespace part1_part2_l33_33859

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := log (4 ^ x + 1) / log 4 + a * x

theorem part1 (a : ℝ) : 
  (∀ x : ℝ, f x a = f (-x) a) → 
  a = -1 / 2 :=
by
  sorry

theorem part2 (m : ℝ) (a : ℝ) : 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → f x a + f (-x) a ≤ 2 * log m / log 4) → 
  m ≥ 17 / 4 :=
by
  sorry

end part1_part2_l33_33859


namespace min_period_and_max_value_l33_33054

def f (x : ℝ) : ℝ := Real.sin (x / 3) + Real.cos (x / 3)

noncomputable def minimum_positive_period (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x : ℝ, f (x + T) = f x ∧ T > 0 ∧ ∀ T' : ℝ, (T' > 0 ∧ ∀ x : ℝ, f (x + T') = f x) → T' ≥ T

noncomputable def is_max_value (f : ℝ → ℝ) (M : ℝ) : Prop :=
  ∀ x : ℝ, f x ≤ M ∧ ∃ x : ℝ, f x = M

theorem min_period_and_max_value :
  minimum_positive_period f (6 * Real.pi) ∧ is_max_value f (Real.sqrt 2) := by
  sorry

end min_period_and_max_value_l33_33054


namespace even_function_iff_b_zero_l33_33341

theorem even_function_iff_b_zero (b c : ℝ) :
  (∀ x : ℝ, (x^2 + b * x + c) = ((-x)^2 + b * (-x) + c)) ↔ b = 0 :=
by
  sorry

end even_function_iff_b_zero_l33_33341


namespace numPeopleToLeftOfKolya_l33_33522

-- Definitions based on the conditions.
def peopleToRightOfKolya := 12
def peopleToLeftOfSasha := 20
def peopleToRightOfSasha := 8

-- Theorem statement with the given conditions and conclusion.
theorem numPeopleToLeftOfKolya 
  (h1 : peopleToRightOfKolya = 12)
  (h2 : peopleToLeftOfSasha = 20)
  (h3 : peopleToRightOfSasha = 8) :
  ∃ n, n = 16 :=
by
  -- Proving the theorem will be done here.
  sorry

end numPeopleToLeftOfKolya_l33_33522


namespace external_tangency_sum_internal_tangency_diff_converse_sum_of_radii_converse_diff_of_radii_l33_33580

variables {O₁ O₂ : ℝ} {r R : ℝ}

-- External tangency implies sum of radii equals distance between centers
theorem external_tangency_sum {O₁ O₂ r R : ℝ} (h1 : O₁ ≠ O₂) (h2 : ∀ M, (dist O₁ M = r) ∧ (dist O₂ M = R) → dist O₁ O₂ = r + R) : 
  dist O₁ O₂ = r + R :=
sorry

-- Internal tangency implies difference of radii equals distance between centers
theorem internal_tangency_diff {O₁ O₂ r R : ℝ} 
  (h1 : O₁ ≠ O₂) 
  (h2 : ∀ M, (dist O₁ M = r) ∧ (dist O₂ M = R) → dist O₁ O₂ = abs (R - r)) : 
  dist O₁ O₂ = abs (R - r) :=
sorry

-- Converse for sum of radii equals distance between centers
theorem converse_sum_of_radii {O₁ O₂ r R : ℝ}
  (h1 : O₁ ≠ O₂) 
  (h2 : dist O₁ O₂ = r + R) : 
  ∃ M, (dist O₁ M = r) ∧ (dist O₂ M = R) ∧ (dist O₁ O₂ = r + R) :=
sorry

-- Converse for difference of radii equals distance between centers
theorem converse_diff_of_radii {O₁ O₂ r R : ℝ}
  (h1 : O₁ ≠ O₂) 
  (h2 : dist O₁ O₂ = abs (R - r)) : 
  ∃ M, (dist O₁ M = r) ∧ (dist O₂ M = R) ∧ (dist O₁ O₂ = abs (R - r)) :=
sorry

end external_tangency_sum_internal_tangency_diff_converse_sum_of_radii_converse_diff_of_radii_l33_33580


namespace hyperbola_equation_l33_33759

theorem hyperbola_equation (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) 
  (h_asymptote_point : (1 : ℝ), (2 : ℝ)) (h_semi_focal : a^2 + b^2 = 5)
  (h_asymptote : b / a = 2) : (∀ x y : ℝ, (x^2 - y^2 / 4 = 1) ↔ ⟨x, y⟩ ∈ set_of (λ p, (∃ x y : ℝ, (p.1 / a) ∧ (p.2 / b))) 
  sorry

end hyperbola_equation_l33_33759


namespace min_positive_period_f_max_value_f_l33_33067

noncomputable def f (x : ℝ) : ℝ := sin (x / 3) + cos (x / 3)

theorem min_positive_period_f : ∃ T > 0, (∀ x, f (x + T) = f x) ∧ ∀ T', (T' > 0 ∧ ∀ x, f (x + T') = f x) → T ≤ T' :=
  sorry

theorem max_value_f : ∃ M, (∀ x, f x ≤ M) ∧ (∀ ε > 0, ∃ x, M - ε < f x) ∧ M = sqrt 2 :=
  sorry

end min_positive_period_f_max_value_f_l33_33067


namespace smaller_tank_capacity_l33_33948

/-- Problem Statement:
Three-quarters of the oil from a certain tank (that was initially full) was poured into a
20000-liter capacity tanker that already had 3000 liters of oil.
To make the large tanker half-full, 4000 more liters of oil would be needed.
What is the capacity of the smaller tank?
-/

theorem smaller_tank_capacity (C : ℝ) 
  (h1 : 3 / 4 * C + 3000 + 4000 = 10000) : 
  C = 4000 :=
sorry

end smaller_tank_capacity_l33_33948


namespace y_squared_in_range_l33_33778

theorem y_squared_in_range (y : ℝ) 
  (h : (Real.sqrt (Real.sqrt (y + 16)) - Real.sqrt (Real.sqrt (y - 16)) = 2)) :
  270 ≤ y^2 ∧ y^2 ≤ 280 :=
sorry

end y_squared_in_range_l33_33778


namespace gecko_cricket_eating_l33_33235

theorem gecko_cricket_eating :
  ∀ (total_crickets : ℕ) (first_day_percent : ℚ) (second_day_less : ℕ),
    total_crickets = 70 →
    first_day_percent = 0.3 →
    second_day_less = 6 →
    let first_day_crickets := total_crickets * first_day_percent
    let second_day_crickets := first_day_crickets - second_day_less
    total_crickets - first_day_crickets - second_day_crickets = 34 :=
by
  intros total_crickets first_day_percent second_day_less h_total h_percent h_less
  let first_day_crickets := total_crickets * first_day_percent
  let second_day_crickets := first_day_crickets - second_day_less
  have : total_crickets - first_day_crickets - second_day_crickets = 34 := sorry
  exact this

end gecko_cricket_eating_l33_33235


namespace least_prime_factor_of_expression_l33_33975

theorem least_prime_factor_of_expression : ∃ (p : ℕ), prime p ∧ p ∣ (5^5 - 5^4) ∧ (∀ q : ℕ, prime q → q ∣ (5^5 - 5^4) → q ≥ p) :=
by
  sorry

end least_prime_factor_of_expression_l33_33975


namespace triangle_is_isosceles_right_triangle_l33_33443

noncomputable def triangle_side_lengths (a b c : ℝ) (B C : ℝ) : Prop :=
  ∃ A : ℝ, 
    A + B + C = π ∧
    a = 2 * c * Real.cos B ∧
    c * Real.cos B + b * Real.cos C = Real.sqrt 2 * c

theorem triangle_is_isosceles_right_triangle
  (a b c : ℝ) (B C : ℝ)
  (h : triangle_side_lengths a b c B C) : 
  A = π / 2 ∧ B = π / 4 ∧ C = π / 4 ∧ b = c :=
sorry

end triangle_is_isosceles_right_triangle_l33_33443


namespace range_of_a_l33_33738

theorem range_of_a (a : ℝ) (x₁ x₂ : ℝ) (e : ℝ) (f : ℝ → ℝ)
  (h1 : 0 < a ∧ a ≠ 1)
  (h2 : f x = 2 * a ^ x - e * x ^ 2)
  (h_min : ∃ x₁, is_local_min f x₁)
  (h_max : ∃ x₂, is_local_max f x₂)
  (h_inequality : x₁ < x₂) :
  a ∈ Ioo (1 / Real.exp 1) 1 :=
sorry

end range_of_a_l33_33738


namespace problem_part1_problem_part2_l33_33756

-- Definitions based on conditions
def f (x a : ℝ) : ℝ := x^2 + abs (2*x - a)

-- Proof statements
theorem problem_part1 (a : ℝ) (h_even : ∀ x : ℝ, f x a = f (-x) a) : a = 0 := sorry

theorem problem_part2 (a : ℝ) (h_a_gt_two : a > 2) : 
  ∃ x : ℝ, ∀ y : ℝ, f x a ≤ f y a ∧ f x a = a - 1 := sorry

end problem_part1_problem_part2_l33_33756


namespace proof_problem_l33_33985

noncomputable def math_proof_problem : Prop :=
  (π ∈ ℝ) ∧ (3 ∈ ℕ) ∧ (0.7 ∉ ℤ) ∧ (∅ ≠ 0)

theorem proof_problem : math_proof_problem := by
  sorry

end proof_problem_l33_33985


namespace hexagon_angle_in_arithmetic_progression_l33_33897

theorem hexagon_angle_in_arithmetic_progression (a d : ℝ) 
  (h_sum : 6 * a + 15 * d = 720) : 
  240 ∈ set.range (λ n : ℕ, a + n * d) :=
sorry

end hexagon_angle_in_arithmetic_progression_l33_33897


namespace find_coeffs_monotonic_intervals_l33_33845

-- Variables for the coefficients of the polynomial
variables {a b : ℝ}

-- Define the function f(x)
def f (x : ℝ) := a * x^3 + b * x + 1

-- Define the derivative of f(x)
def f_prime (x : ℝ) := 3 * a * x^2 + b

-- The first part of the problem: prove values of a and b
theorem find_coeffs (h1 : f 1 = -1) (h2 : f_prime 1 = 0) : a = 1 ∧ b = -3 :=
sorry

-- The second part of the problem: prove the monotonic intervals
theorem monotonic_intervals : 
  let f (x : ℝ) := x^3 - 3 * x + 1 in
  (∀ x : ℝ, x < -1 → f x < f (-1)) ∧
  (∀ x : ℝ, x > 1 → f x > f 1) ∧
  (∀ x : ℝ, x > -1 ∧ x < 1 → f x < f (-1)) :=
sorry

end find_coeffs_monotonic_intervals_l33_33845


namespace polynomial_calculation_l33_33674

theorem polynomial_calculation :
  (49^5 - 5 * 49^4 + 10 * 49^3 - 10 * 49^2 + 5 * 49 - 1) = 254804368 :=
by
  sorry

end polynomial_calculation_l33_33674


namespace kelly_harvested_pounds_l33_33825

def total_carrots (bed1 bed2 bed3 : ℕ) : ℕ :=
  bed1 + bed2 + bed3

def total_weight (total : ℕ) (carrots_per_pound : ℕ) : ℕ :=
  total / carrots_per_pound

theorem kelly_harvested_pounds :
  total_carrots 55 101 78 = 234 ∧ total_weight 234 6 = 39 :=
by {
  split,
  { exact rfl }, -- 234 = 234
  { exact rfl }  -- 234 / 6 = 39
}

end kelly_harvested_pounds_l33_33825


namespace almost_no_G_in_𝓟_H_as_n_tends_to_infty_l33_33409

variables (n : ℕ) (H : Type) (γ : ℝ) (ε : H → ℝ) (k ℓ : ℕ)
variables (G : Type) (𝓟_H : set G) (𝓖 : ℕ → ℝ → set G)

-- condition t = n^(-1 / ε(H))
def t := n^(-1 / ε H)

-- condition p = γ * t = γ * n^(-k/ℓ)
def p := γ * t n H ε

-- condition γ → 0 as n → ∞
def γ_tends_to_zero_as_n_tends_to_infty : Prop := filter.tendsto γ filter.at_top (𝓝 0)

-- statement to prove
theorem almost_no_G_in_𝓟_H_as_n_tends_to_infty :
  γ_tends_to_zero_as_n_tends_to_infty γ →
  filter.tendsto (λ n, P[λ (G : G), G ∈ 𝓟_H]) filter.at_top (𝓝 0) :=
sorry

end almost_no_G_in_𝓟_H_as_n_tends_to_infty_l33_33409


namespace expenditure_increase_l33_33430

theorem expenditure_increase (A : ℝ) 
  (h₁ : 100 * A + 25 * (A - 10) = 7500) :
  7500 - (100 * A) = 500 :=
by
  have hA : 125 * (A - 10) = 7500 := h₁
  sorry

end expenditure_increase_l33_33430


namespace find_value_of_f_l33_33735

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function : ∀ x : ℝ, f(-x) = -f(x)
axiom periodicity : ∀ x : ℝ, f(x + 2) = f(x)
axiom function_definition : ∀ x : ℝ, (0 < x ∧ x < 1) → f(x) = 2^x - 2

theorem find_value_of_f : f(real.logb (1/2) 6) = 1/2 :=
sorry

end find_value_of_f_l33_33735


namespace inequality_proof_l33_33340

-- Definitions based on the given conditions.
def a : ℝ := Real.log 3 / Real.log 2  -- a = log_2(3)
def b : ℝ := Real.log 4 / Real.log 3  -- b = log_3(4)
def c : ℝ := 3 / 2                    -- c = 3/2

-- The theorem statement proving the required inequality.
theorem inequality_proof : b < c ∧ c < a := by
  sorry

end inequality_proof_l33_33340


namespace sum_f_inv_l33_33684

def f (x : ℝ) : ℝ :=
  if x < 5 then x - 3 else x^2

def g_inv (y : ℝ) : ℝ := y + 3
def h_inv (y : ℝ) : ℝ := Real.sqrt y

lemma f_inv (y : ℝ) : ℝ :=
  if y < 2 then g_inv y else if y ≥ 25 then h_inv y else 0

theorem sum_f_inv :
  (Finset.range (1 - (-8) + 1)).sum (λ k, f_inv (k - 8)) + f_inv 25 = 5 :=
by sorry

end sum_f_inv_l33_33684


namespace find_k_l33_33354

theorem find_k (k : ℝ) :
  (∀ x y: ℝ, (y = k * x + 1) → (x^2 + y^2 - 2 * x - 2 * y + 1 = 0)) →
  (∃ A B : ℝ × ℝ, (A ≠ B) ∧ (|A - B| = Real.sqrt 2) ∧ (A.2 = k * A.1 + 1) ∧ (B.2 = k * B.1 + 1) ∧ ((A.1^2 + A.2^2 - 2 * A.1 - 2 * A.2 + 1 = 0) ∧ (B.1^2 + B.2^2 - 2 * B.1 - 2 * B.2 + 1 = 0))) →
  (k = 1 ∨ k = -1) :=
by
  sorry

end find_k_l33_33354


namespace rectangle_dimensions_l33_33166

variables (b l : ℝ)

def breadth_is_nonzero : b ≠ 0 := sorry
def length_is_three_times_breadth : l = 3 * b := sorry
def area_equals_perimeter (b l : ℝ) : 3 * b ^ 2 = 8 * b := sorry

theorem rectangle_dimensions :
  (∀ b l : ℝ, length_is_three_times_breadth b l → area_equals_perimeter b l → b = 8 / 3 ∧ l = 8) :=
by
  intros b l h1 h2
  sorry

end rectangle_dimensions_l33_33166


namespace sufficient_but_not_necessary_not_necessary_is_sufficient_not_necessary_l33_33721

theorem sufficient_but_not_necessary (a : ℝ) (h : a > 0) : |a| > 0 :=
by
  -- Proof would go here
  sorry

theorem not_necessary (a : ℝ) (h : |a| > 0) : (a > 0) ∨ (a < 0) :=
by
  -- Proof would go here
  sorry

theorem is_sufficient_not_necessary (a : ℝ) : (a > 0 → |a| > 0) ∧ (|a| > 0 → (a > 0 ∨ a < 0)) :=
by
  split
  · exact sufficient_but_not_necessary a
  · exact not_necessary a

end sufficient_but_not_necessary_not_necessary_is_sufficient_not_necessary_l33_33721


namespace BG_geq_BH_l33_33835

-- Definitions of the geometric objects and their properties
variables {A B C G H M O : Point}
variables (triangle_ABC : Triangle A B C)
variables (centroid_G : Centroid triangle_ABC G)
variables (orthocenter_H : Orthocenter triangle_ABC H)
variables (midpoint_arc_M : MidpointArc AC M)
variables (circumcenter_O : Circumcenter triangle_ABC O)
variables {R : ℝ}

-- Conditions
variable (MG_eq_R : distance M G = R)
variable (M_on_circumcircle : OnCircumcircle triangle_ABC M)

-- Statement to prove
theorem BG_geq_BH (triangle_ABC : Triangle A B C)
  (centroid_G : Centroid triangle_ABC G)
  (orthocenter_H : Orthocenter triangle_ABC H)
  (midpoint_arc_M : MidpointArc AC M)
  (circumcenter_O : Circumcenter triangle_ABC O)
  (MG_eq_R : distance M G = R)
  (M_on_circumcircle : OnCircumcircle triangle_ABC M)
  : distance B G ≥ distance B H :=
sorry

end BG_geq_BH_l33_33835


namespace number_of_female_students_l33_33901

theorem number_of_female_students
  (F : ℕ) -- number of female students
  (T : ℕ) -- total number of students
  (h1 : T = F + 8) -- total students = female students + 8 male students
  (h2 : 90 * T = 85 * 8 + 92 * F) -- equation from the sum of scores
  : F = 20 :=
sorry

end number_of_female_students_l33_33901


namespace colorings_count_l33_33552

theorem colorings_count :
  let grid_size := 7
  let valid_colorings := binomial (2 * grid_size) grid_size
  valid_colorings = 3432 :=
by
  let grid_size := 7
  let valid_colorings := binomial (2 * grid_size) grid_size
  have h : valid_colorings = 3432 := by {
    simp [valid_colorings],
    norm_num
  }
  exact h

end colorings_count_l33_33552


namespace lada_drew_isosceles_triangle_l33_33453

variables {α β γ : ℝ}
variables {δ ε : ℝ}

def isosceles_triangle (α β γ : ℝ) : Prop :=
  β = γ

theorem lada_drew_isosceles_triangle
  (Lada_triangle : α + β + γ = 180)
  (Lera_triangle_1 : δ = α + β)
  (Lera_triangle_2 : ε = α + γ)
  (sum_of_angles_lt : δ + ε ≤ 180) :
  isosceles_triangle α β γ :=
begin
  sorry
end

end lada_drew_isosceles_triangle_l33_33453


namespace base8_units_digit_l33_33579

theorem base8_units_digit (n m : ℕ) (h1 : n = 348) (h2 : m = 27) : 
  (n * m % 8) = 4 := sorry

end base8_units_digit_l33_33579


namespace sin_alpha_minus_95_l33_33369

variables {α : Real}

-- Assume α is an angle in the third quadrant.
-- condition: α ∈ [180°, 270°)
def in_third_quadrant (α : Real) : Prop :=
  180 * π / 180 ≤ α ∧ α < 270 * π / 180

-- Given the cosine condition.
-- condition: cos(85° + α) = 4/5
def given_cos_condition (α : Real) : Prop :=
  Real.cos (85 * π / 180 + α) = 4/5

-- Prove: sin(α - 95°) = 3/5
theorem sin_alpha_minus_95 (α : Real) (h1 : in_third_quadrant α) 
  (h2 : given_cos_condition α) : Real.sin (α - 95 * π / 180) = 3 / 5 :=
sorry

end sin_alpha_minus_95_l33_33369


namespace repetend_of_frac_4_div_17_is_235294_l33_33710

noncomputable def decimalRepetend_of_4_div_17 : String :=
  let frac := 4 / 17
  let repetend := "235294"
  repetend

theorem repetend_of_frac_4_div_17_is_235294 :
  (∃ n m : ℕ, (4 / 17 : ℚ) = n + (m / 10^6) ∧ m % 10^6 = 235294) :=
sorry

end repetend_of_frac_4_div_17_is_235294_l33_33710


namespace min_positive_period_f_max_value_f_l33_33073

noncomputable def f (x : ℝ) : ℝ := sin (x / 3) + cos (x / 3)

theorem min_positive_period_f : ∃ T > 0, (∀ x, f (x + T) = f x) ∧ ∀ T', (T' > 0 ∧ ∀ x, f (x + T') = f x) → T ≤ T' :=
  sorry

theorem max_value_f : ∃ M, (∀ x, f x ≤ M) ∧ (∀ ε > 0, ∃ x, M - ε < f x) ∧ M = sqrt 2 :=
  sorry

end min_positive_period_f_max_value_f_l33_33073


namespace trigonometric_identity_proof_l33_33802

noncomputable def m : ℝ := 2 * Real.sin (Real.pi / 10)
noncomputable def n : ℝ := 4 - m^2

theorem trigonometric_identity_proof :
  (m = 2 * Real.sin (Real.pi / 10)) →
  (m^2 + n = 4) →
  (m * Real.sqrt n) / (2 * Real.cos (3 * Real.pi / 20)^2 - 1) = 2 :=
by
  intros h1 h2
  sorry

end trigonometric_identity_proof_l33_33802


namespace tommy_finishes_first_l33_33018

theorem tommy_finishes_first
  (x z : ℝ)
  (Hs : 0 < x)
  (Hz : 0 < z):
  let A_s := 2 * x,
      A_t := 2 * x / 3,
      A_r := 4 * x,
      R_s := z / 2,
      R_t := z / 4,
      R_r := z,
      T_s := A_s / R_s,
      T_t := A_t / R_t,
      T_r := A_r / R_r in
  T_t < T_s ∧ T_t < T_r :=
by
  let A_s := 2 * x;
  let A_t := 2 * x / 3;
  let A_r := 4 * x;
  let R_s := z / 2;
  let R_t := z / 4;
  let R_r := z;
  let T_s := A_s / R_s;
  let T_t := A_t / R_t;
  let T_r := A_r / R_r;
  have Ts_eq := calc T_s = 2 * x / (z / 2) : by sorry
  have Tt_eq := calc T_t = (2 * x / 3) / (z / 4) : by sorry
  have Tr_eq := calc T_r = 4 * x / z : by sorry
  have T_s_reduced : T_s = 4 * x / z := by sorry
  have T_t_reduced : T_t = 8 * x / (3 * z) := by sorry
  have T_r_reduced : T_r = 4 * x / z := by sorry
  have comparison1 : 8 * x / (3 * z) < 4 * x / z := by sorry
  have comparison2 : 8 * x / (3 * z) < 4 * x / z := by sorry
  exact ⟨comparison1, comparison2⟩

#check tommy_finishes_first

end tommy_finishes_first_l33_33018


namespace Suzanna_ride_distance_l33_33514

/-- 
Given that Suzanna rides her bike at a constant rate such that every 10 minutes, 
her distance increases by 1.5 miles, and given that she rides for a total of 
40 minutes but takes a 10-minute break after the first 20 minutes, 
prove that she rides a total distance of 4.5 miles.
-/
theorem Suzanna_ride_distance (rate : ℝ) (time : ℕ) (break_time : ℕ) (distance_increase : ℝ) : 
    rate = 1.5 →
    time = 40 →
    break_time = 10 →
    distance_increase = 1.5 →
    ∃ distance : ℝ, distance = (time - break_time) / 10 * rate 
                    ∧ (time - break_time) = 30 ∧ distance = 4.5 :=
by 
  intros rate_eq time_eq break_time_eq distance_increase_eq
  use (time - break_time) / 10 * rate
  split
  { rw [rate_eq, time_eq, break_time_eq],
    norm_num },
  split
  { rw [time_eq, break_time_eq],
    norm_num },
  { rw [rate_eq, time_eq, break_time_eq],
    norm_num }

end Suzanna_ride_distance_l33_33514


namespace min_period_and_max_value_l33_33124

def f (x : ℝ) := Real.sin (x / 3) + Real.cos (x / 3)

theorem min_period_and_max_value :
  (∀ T > 0, T ≠ 6 * Real.pi → ¬∀ x : ℝ, f(x + T) = f(x)) ∧
  ∃ x : ℝ, f(x) = sqrt 2 :=
by
  sorry

end min_period_and_max_value_l33_33124


namespace find_k_l33_33762

def parabola (k : ℝ) : ℝ → ℝ := λ x, x^2 - k * x + k - 1

theorem find_k
  (k : ℝ)
  (condition1 : (4 * (k - 1) - k^2) / 4 = 0)
  (condition2 : (-k) / 2 = 0 ∨ (-k) / 2 = 1)
  (condition3 : ((-k) / 2 = -1 ∧ (4 * (k - 1) - k^2) / 4 = -4) ∨ parabola k 0 = 0)
  (condition4 : parabola k 1 = -1) :
  k = 2 ∨ k = 0 ∨ k = 1 ∨ k = 4 :=
by 
  sorry

end find_k_l33_33762


namespace phase_shift_correct_l33_33708

-- Given the function y = 3 * sin (x - π / 5)
-- We need to prove that the phase shift is π / 5.

theorem phase_shift_correct :
  ∀ x : ℝ, 3 * Real.sin (x - Real.pi / 5) = 3 * Real.sin (x - C) →
  C = Real.pi / 5 :=
by
  sorry

end phase_shift_correct_l33_33708


namespace max_students_satisfying_conditions_l33_33662

-- Definitions of the conditions
def condition1 (n : ℕ) (knows : fin n → fin n → Prop) : Prop :=
  ∀ (a b c : fin n), knows a b ∨ knows b c ∨ knows a c

def condition2 (n : ℕ) (knows : fin n → fin n → Prop) : Prop :=
  ∀ (a b c d : fin n), ¬(knows a b ∧ knows b c ∧ knows c d ∧ knows d a)

-- The main theorem stating the problem and its solution
theorem max_students_satisfying_conditions :
  ∃ (n : ℕ), 
    (∀ (knows : fin n → fin n → Prop), condition1 n knows ∧ condition2 n knows) 
    ∧ (n ≤ 8) := 
sorry

end max_students_satisfying_conditions_l33_33662


namespace valid_pairings_count_l33_33287

theorem valid_pairings_count :
  let colors := { "red", "blue", "yellow", "green", "purple" }
  in (∃ (f : colors → colors), (∀ b, f b ≠ b)) ->
  20 := 
sorry

end valid_pairings_count_l33_33287


namespace percentage_of_boys_l33_33791

theorem percentage_of_boys (total_students boys_per_group girls_per_group : ℕ)
  (ratio_condition : boys_per_group + girls_per_group = 7)
  (total_condition : total_students = 42)
  (ratio_b_condition : boys_per_group = 3)
  (ratio_g_condition : girls_per_group = 4) :
  (boys_per_group : ℚ) / (boys_per_group + girls_per_group : ℚ) * 100 = 42.86 :=
by sorry

end percentage_of_boys_l33_33791


namespace black_area_larger_by_one_l33_33331

theorem black_area_larger_by_one (a b c : ℝ) :
  let gray_area := 17 - a - b - c,
      black_area := 18 - a - b - c in
  black_area - gray_area = 1 :=
by
  sorry

end black_area_larger_by_one_l33_33331


namespace smallest_possible_input_l33_33644

def F (n : ℕ) := 9 * n + 120

theorem smallest_possible_input : ∃ n : ℕ, n > 0 ∧ F n = 129 :=
by {
  -- Here we would provide the proof steps, but we use sorry for now.
  sorry
}

end smallest_possible_input_l33_33644


namespace find_k_from_roots_ratio_l33_33920

theorem find_k_from_roots_ratio (k : ℝ) (x1 x2 : ℝ) 
  (h1 : x1 + x2 = -10)
  (h2 : x1 * x2 = k)
  (h3 : x1/x2 = 3) : 
  k = 18.75 := 
sorry

end find_k_from_roots_ratio_l33_33920


namespace minimum_value_lambda_mu_squared_is_half_l33_33458

open Real

noncomputable def minimum_lambda_mu_squared (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) : ℝ :=
  let constraint (λ μ : ℝ) := (λ + μ)^2 - (λ - μ)^2 = 1
  (λ^2 + μ^2)

theorem minimum_value_lambda_mu_squared_is_half (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) :
  Inf { z : ℝ | ∃ λ μ : ℝ, (λ + μ)^2 - (λ - μ)^2 = 1 ∧ z = λ^2 + μ^2 } = 1/2 :=
sorry

end minimum_value_lambda_mu_squared_is_half_l33_33458


namespace sum_of_all_possible_numbers_l33_33486

def digits := {2, 0, 1, 8} : Finset ℕ

-- A function to convert a list of digits to a number
def to_number (ds : List ℕ) : ℕ :=
  ds.foldl (λ acc d => acc * 10 + d) 0

-- A function to generate all possible numbers from the digits 
def generate_numbers : Finset ℕ :=
  (Finset.univ : Finset (Finset ℕ)).filter (λ s => s ⊆ digits ∧ s.card > 0).bUnion (λ s =>
    (s.powerset.filter (λ t => t.card = s.card)).map to_number)

-- The main sum of all generated numbers
def naturals_sum : ℕ :=
  generate_numbers.sum id

theorem sum_of_all_possible_numbers : naturals_sum = 78331 := by
  sorry

end sum_of_all_possible_numbers_l33_33486


namespace number_ordered_pairs_satisfying_condition_l33_33164

theorem number_ordered_pairs_satisfying_condition : (∑ (a b : ℝ), (a + b * Complex.i)^6 = a - b * Complex.i) = 8 := 
sorry

end number_ordered_pairs_satisfying_condition_l33_33164


namespace number_of_people_to_the_left_of_Kolya_l33_33535

-- Defining the conditions
variables (left_sasha right_sasha right_kolya total_students left_kolya : ℕ)

-- Condition definitions
def condition1 := right_kolya = 12
def condition2 := left_sasha = 20
def condition3 := right_sasha = 8

-- Calculate total number of students
def calc_total_students : ℕ := left_sasha + right_sasha + 1

-- Calculate number of students to the left of Kolya
def calc_left_kolya (total_students right_kolya : ℕ) : ℕ := total_students - right_kolya - 1

-- Problem statement to prove
theorem number_of_people_to_the_left_of_Kolya
    (H1 : condition1)
    (H2 : condition2)
    (H3 : condition3)
    (total_students : calc_total_students = 29) : 
    calc_left_kolya total_students right_kolya = 16 :=
by
  sorry

end number_of_people_to_the_left_of_Kolya_l33_33535


namespace people_left_of_Kolya_l33_33542

/-- Given:
    1. There are 12 people to the right of Kolya.
    2. There are 20 people to the left of Sasha.
    3. There are 8 people to the right of Sasha.
    4. The total number of people in the class (including Sasha) is 29.

    Prove:
    The number of people to the left of Kolya is 16.
-/
theorem people_left_of_Kolya : 
  ∀ (total_people right_of_Kolya left_of_Sasha right_of_Sasha : ℕ),
  right_of_Kolya = 12 →
  left_of_Sasha = 20 →
  right_of_Sasha = 8 →
  total_people = 29 →
  left_of_Kolya := total_people - right_of_Kolya - 1
  left_of_Kolya = 16 :=
by
  intros
  sorry

end people_left_of_Kolya_l33_33542


namespace regression_line_equation_l33_33750

theorem regression_line_equation 
  (a b : ℝ)
  (center : ℝ × ℝ)
  (h_center : center = (4, 5))
  (h_a : a = 0.2)
  (h_regression : (center.snd : ℝ) = 4 * b + a) :
  (b = 1.2) ∧ (a = 0.2) ∧ (∀ x, (4, 5).snd = b * 4 + a → 5 = b * 4 + a) :=
begin
  sorry
end

end regression_line_equation_l33_33750


namespace range_of_a_l33_33739

theorem range_of_a (a : ℝ) (x₁ x₂ : ℝ) (e : ℝ) (f : ℝ → ℝ)
  (h1 : 0 < a ∧ a ≠ 1)
  (h2 : f x = 2 * a ^ x - e * x ^ 2)
  (h_min : ∃ x₁, is_local_min f x₁)
  (h_max : ∃ x₂, is_local_max f x₂)
  (h_inequality : x₁ < x₂) :
  a ∈ Ioo (1 / Real.exp 1) 1 :=
sorry

end range_of_a_l33_33739


namespace problem_statement_l33_33384

def f (x : ℝ) : ℝ := log (sqrt (1 + x^2) - x) + 2

theorem problem_statement : f (log 5) + f (log (1/5)) = 4 := by
  sorry

end problem_statement_l33_33384


namespace solution_set_f_inequality_l33_33744

variable (f : ℝ → ℝ)

axiom domain_of_f : ∀ x : ℝ, true
axiom avg_rate_of_f : ∀ (x1 x2 : ℝ), x1 ≠ x2 → (f x1 - f x2) / (x1 - x2) > 3
axiom f_at_5 : f 5 = 18

theorem solution_set_f_inequality : {x : ℝ | f (3 * x - 1) > 9 * x} = {x : ℝ | x > 2} :=
by
  sorry

end solution_set_f_inequality_l33_33744


namespace exists_triangles_arrangement_l33_33669

noncomputable def triangles_arrangement (T : Fin 10 → Triangle ℝ) : Prop :=
  (∀ i j, i ≠ j → (∃ p, p ∈ (T i).points ∧ p ∈ (T j).points)) ∧
  (∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k → ¬(∃ p, p ∈ (T i).points ∧ p ∈ (T j).points ∧ p ∈ (T k).points))

theorem exists_triangles_arrangement :
  ∃ T : Fin 10 → Triangle ℝ, triangles_arrangement T :=
sorry

end exists_triangles_arrangement_l33_33669


namespace travel_time_l33_33473

noncomputable def distance_to_office (T : ℝ) : ℝ := 58 * T

noncomputable def distance_return_trip (T_prime : ℝ) : ℝ := 62 * T_prime

theorem travel_time 
  (T T_prime : ℝ)
  (H1 : distance_to_office T = distance_return_trip T_prime)
  (H2 : T + (1/6) + T_prime = 3) :
  T ≈ 1.4639 :=
begin
  sorry
end

end travel_time_l33_33473


namespace min_positive_period_and_max_value_of_f_l33_33095

noncomputable def f (x : ℝ) : ℝ := sin (x / 3) + cos (x / 3)

theorem min_positive_period_and_max_value_of_f :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ T = 6 * π) ∧ 
  (∃ x, f x = sqrt 2) :=
by {
  sorry
}

end min_positive_period_and_max_value_of_f_l33_33095


namespace cone_lateral_surface_area_l33_33414

theorem cone_lateral_surface_area (r : ℝ) (V : ℝ) (h : ℝ) (l : ℝ) 
  (h₁ : r = 3)
  (h₂ : V = 12 * Real.pi)
  (h₃ : V = (1 / 3) * Real.pi * r^2 * h)
  (h₄ : l = Real.sqrt (r^2 + h^2)) : 
  ∃ A : ℝ, A = Real.pi * r * l ∧ A = 15 * Real.pi := 
by
  use Real.pi * r * l
  have hr : r = 3 := by exact h₁
  have hV : V = 12 * Real.pi := by exact h₂
  have volume_formula : V = (1 / 3) * Real.pi * r^2 * h := by exact h₃
  have slant_height : l = Real.sqrt (r^2 + h^2) := by exact h₄
  sorry

end cone_lateral_surface_area_l33_33414


namespace average_monthly_income_l33_33020

theorem average_monthly_income (P Q R : ℝ) (h1 : (P + Q) / 2 = 5050)
  (h2 : (Q + R) / 2 = 6250) (h3 : P = 4000) : (P + R) / 2 = 5200 := by
  sorry

end average_monthly_income_l33_33020


namespace boxes_needed_l33_33571

noncomputable def living_room_length : ℝ := 16
noncomputable def living_room_width : ℝ := 20
noncomputable def sq_ft_per_box : ℝ := 10
noncomputable def already_floored : ℝ := 250

theorem boxes_needed : 
  (living_room_length * living_room_width - already_floored) / sq_ft_per_box = 7 :=
by 
  sorry

end boxes_needed_l33_33571


namespace division_multiplication_example_l33_33196

theorem division_multiplication_example : 120 / 4 / 2 * 3 = 45 := by
  sorry

end division_multiplication_example_l33_33196


namespace motorcyclists_speeds_l33_33964

theorem motorcyclists_speeds 
  (distance_AB : ℝ) (distance1 : ℝ) (distance2 : ℝ) (time_diff : ℝ) 
  (x y : ℝ) 
  (h1 : distance_AB = 600) 
  (h2 : distance1 = 250) 
  (h3 : distance2 = 200) 
  (h4 : time_diff = 3)
  (h5 : distance1 / x = distance2 / y)
  (h6 : distance_AB / x + time_diff = distance_AB / y) : 
  x = 50 ∧ y = 40 := 
sorry

end motorcyclists_speeds_l33_33964


namespace least_possible_value_of_x_minus_y_minus_z_l33_33423

theorem least_possible_value_of_x_minus_y_minus_z :
  ∀ (x y z : ℕ), x = 4 → y = 7 → z > 0 → (x - y - z) = -4 := by
  sorry

end least_possible_value_of_x_minus_y_minus_z_l33_33423


namespace people_left_of_Kolya_l33_33538

/-- Given:
    1. There are 12 people to the right of Kolya.
    2. There are 20 people to the left of Sasha.
    3. There are 8 people to the right of Sasha.
    4. The total number of people in the class (including Sasha) is 29.

    Prove:
    The number of people to the left of Kolya is 16.
-/
theorem people_left_of_Kolya : 
  ∀ (total_people right_of_Kolya left_of_Sasha right_of_Sasha : ℕ),
  right_of_Kolya = 12 →
  left_of_Sasha = 20 →
  right_of_Sasha = 8 →
  total_people = 29 →
  left_of_Kolya := total_people - right_of_Kolya - 1
  left_of_Kolya = 16 :=
by
  intros
  sorry

end people_left_of_Kolya_l33_33538


namespace kelly_harvested_pounds_l33_33823

def total_carrots (bed1 bed2 bed3 : ℕ) : ℕ :=
  bed1 + bed2 + bed3

def total_weight (total : ℕ) (carrots_per_pound : ℕ) : ℕ :=
  total / carrots_per_pound

theorem kelly_harvested_pounds :
  total_carrots 55 101 78 = 234 ∧ total_weight 234 6 = 39 :=
by {
  split,
  { exact rfl }, -- 234 = 234
  { exact rfl }  -- 234 / 6 = 39
}

end kelly_harvested_pounds_l33_33823


namespace max_median_soda_l33_33484

theorem max_median_soda (cans_sold : ℕ) (customers : ℕ) (min_cans_per_customer : ℕ)
  (h_total_cans : cans_sold = 300) (h_total_customers : customers = 120)
  (h_min_cans : ∀ c, c < customers → c >= min_cans_per_customer) :
  median (λ n < customers, n) = 3 :=
sorry

end max_median_soda_l33_33484


namespace area_PDE_l33_33807

variables {α β γ : Type*} [LinearOrderedField α]
variables {A B C P D Q E : Point α} -- Points in the triangle

open_locale big_operators classical

-- Assumptions
variables (a b : α) -- sides of triangle
variables (sin_alpha : α)

-- Conditions based on problem
axiom obtuse_angle_A : 90 < α
axiom midpoint_AN_NC : 2 * (AC n) = AC
axiom perpendicular_PD_AC : PD ⊥ AC
axiom perpendicular_QE_AB : QE ⊥ AB
axiom area_ABC : 1 / 2 * a * b * sin_alpha = 36

-- Proof statement indicating the problem and expected result
theorem area_PDE (a b : α) (sin_alpha : α) (α : angle) 
  (h1 : obtuse_angle_A) 
  (h2 : midpoint_AN_NC) 
  (h3 : perpendicular_PD_AC)
  (h4 : perpendicular_QE_AB) 
  (h5 : area_ABC) : 
  area P D E = 18 * sin_alpha :=
sorry

end area_PDE_l33_33807


namespace smallest_n_l33_33322

theorem smallest_n (n : ℕ) (h : 5 * n ≡ 850 [MOD 26]) : n = 14 :=
by
  sorry

end smallest_n_l33_33322


namespace angle_between_vectors_l33_33355

variables (a b : ℝ^3) [ne_zero a] [ne_zero b]

theorem angle_between_vectors (h : ‖a + b‖ = ‖a - b‖) : real.angle a b = real.pi / 2 :=
sorry

end angle_between_vectors_l33_33355


namespace smallest_repeating_block_of_5_over_13_l33_33687

theorem smallest_repeating_block_of_5_over_13 : 
  ∃ n, n = 6 ∧ (∃ m, (5 / 13 : ℚ) = (m/(10^6) : ℚ) ) := 
sorry

end smallest_repeating_block_of_5_over_13_l33_33687


namespace attendees_gift_exchange_l33_33267

theorem attendees_gift_exchange (x : ℕ) (h1 : 56 = x * (x - 1) / 2) : 
  x * (x - 1) = 112 :=
by
  sorry

end attendees_gift_exchange_l33_33267


namespace min_positive_period_and_max_value_of_f_l33_33091

noncomputable def f (x : ℝ) : ℝ := sin (x / 3) + cos (x / 3)

theorem min_positive_period_and_max_value_of_f :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ T = 6 * π) ∧ 
  (∃ x, f x = sqrt 2) :=
by {
  sorry
}

end min_positive_period_and_max_value_of_f_l33_33091


namespace min_positive_period_and_max_value_l33_33085

def f (x : ℝ) := Real.sin (x / 3) + Real.cos (x / 3)

theorem min_positive_period_and_max_value : 
  (∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T' = T)) 
  ∧ 
  (∀ x, f x ≤ sqrt 2 ∧ (∃ x, f x = sqrt 2)) :=
sorry

end min_positive_period_and_max_value_l33_33085


namespace number_of_people_to_the_left_of_Kolya_l33_33532

-- Defining the conditions
variables (left_sasha right_sasha right_kolya total_students left_kolya : ℕ)

-- Condition definitions
def condition1 := right_kolya = 12
def condition2 := left_sasha = 20
def condition3 := right_sasha = 8

-- Calculate total number of students
def calc_total_students : ℕ := left_sasha + right_sasha + 1

-- Calculate number of students to the left of Kolya
def calc_left_kolya (total_students right_kolya : ℕ) : ℕ := total_students - right_kolya - 1

-- Problem statement to prove
theorem number_of_people_to_the_left_of_Kolya
    (H1 : condition1)
    (H2 : condition2)
    (H3 : condition3)
    (total_students : calc_total_students = 29) : 
    calc_left_kolya total_students right_kolya = 16 :=
by
  sorry

end number_of_people_to_the_left_of_Kolya_l33_33532


namespace quadratic_inequality_solution_l33_33760

theorem quadratic_inequality_solution:
  ∃ P q : ℝ,
  (1 / P < 0) ∧
  (-P * q = 6) ∧
  (P^2 = 8) ∧
  (P = -2 * Real.sqrt 2) ∧
  (q = 3 / 2 * Real.sqrt 2) :=
by
  sorry

end quadratic_inequality_solution_l33_33760


namespace license_plate_count_l33_33402

theorem license_plate_count : 
  let vowels := 5
  let consonants := 21
  let digits := 10
  21 * 21 * 5 * 5 * 10 = 110250 := 
by 
  sorry

end license_plate_count_l33_33402


namespace complement_of_M_l33_33472

definition M : Set ℝ := {x | x^2 - 5 * x - 6 > 0}
definition U : Set ℝ := Set.univ

theorem complement_of_M (x : ℝ) : x ∈ U \ M ↔ x ∈ Set.Icc (-1) 6 := 
by sorry

end complement_of_M_l33_33472


namespace geometric_progression_sum_of_first_n_terms_l33_33723

def sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, a (n + 1) = 3 * a n + 4

theorem geometric_progression {a : ℕ → ℤ} (h : sequence a) :
  ∀ n : ℕ, (a (n + 1) + 2) = 3 * (a n + 2) :=
sorry

theorem sum_of_first_n_terms {a : ℕ → ℤ} (h : sequence a) (n : ℕ) (S_n : ℤ) :
  S_n = (∑ k in Finset.range n + 1, a (k + 1)) → 
  S_n = (3^(n + 1) - 3) / 2 - 2 * n :=
sorry

end geometric_progression_sum_of_first_n_terms_l33_33723


namespace no_values_make_expression_zero_l33_33908

-- Definition of the given function
def given_expression (t : ℂ) : ℂ := (sqrt(49 - t^2 : ℂ)) + 7

-- The main theorem to prove
theorem no_values_make_expression_zero : ∀ t : ℂ, given_expression t ≠ 0 := 
by
  intro t
  sorry

end no_values_make_expression_zero_l33_33908


namespace vertex_C_coordinates_l33_33505

/-
Theorem: For a triangle ABC with A(2,0), B(0,2) and the Euler line equation 2x - y - 2 = 0,
the coordinates of vertex C are (18/5, 16/5).
-/

noncomputable def A : (ℝ × ℝ) := (2, 0)
noncomputable def B : (ℝ × ℝ) := (0, 2)
noncomputable def euler_line_equation (x y : ℝ) : Prop := 2 * x - y - 2 = 0
def centroid (A B C : (ℝ × ℝ)) : (ℝ × ℝ) :=
  ( (A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3 )

theorem vertex_C_coordinates : ∃ C : (ℝ × ℝ), 
  let G := centroid A B C in
  euler_line_equation G.1 G.2 ∧ 
  ((G.1 = 18 / 5) ∧ (G.2 = 16 / 5)) :=
sorry

end vertex_C_coordinates_l33_33505


namespace value_of_w_over_y_l33_33748

theorem value_of_w_over_y (w x y : ℝ) (h1 : w / x = 1 / 3) (h2 : (x + y) / y = 3) : w / y = 2 / 3 :=
by
  sorry

end value_of_w_over_y_l33_33748


namespace number_of_integer_coordinates_on_circle_l33_33418

theorem number_of_integer_coordinates_on_circle (r : ℕ) (h : r = 5) :
  ∃ n : ℕ, n = 12 ∧ (finset.univ.filter (λ p : ℤ × ℤ, p.1^2 + p.2^2 = (r : ℤ)^2)).card = n :=
begin
  use 12,
  split,
  { refl },
  { sorry }
end

end number_of_integer_coordinates_on_circle_l33_33418


namespace degree_of_each_exterior_angle_of_regular_octagon_l33_33025
-- We import the necessary Lean libraries

-- Define the degrees of each exterior angle of a regular octagon
theorem degree_of_each_exterior_angle_of_regular_octagon : 
  (∑ (i : Fin 8), (360 / 8) = 360 → (360 / 8) = 45) :=
by
  sorry

end degree_of_each_exterior_angle_of_regular_octagon_l33_33025


namespace min_pos_period_max_value_l33_33151

noncomputable def f (x : ℝ) := Real.sin (x / 3) + Real.cos (x / 3)

theorem min_pos_period_max_value :
  (∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ T) ∧ T = 6 * Real.pi) ∧ 
  (∀ x, f x ≤ sqrt 2 ∧ (∃ x, f x = sqrt 2)) :=
by
  sorry

end min_pos_period_max_value_l33_33151


namespace min_positive_period_and_max_value_l33_33086

def f (x : ℝ) := Real.sin (x / 3) + Real.cos (x / 3)

theorem min_positive_period_and_max_value : 
  (∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T' = T)) 
  ∧ 
  (∀ x, f x ≤ sqrt 2 ∧ (∃ x, f x = sqrt 2)) :=
sorry

end min_positive_period_and_max_value_l33_33086


namespace min_positive_period_and_max_value_of_f_l33_33097

noncomputable def f (x : ℝ) : ℝ := sin (x / 3) + cos (x / 3)

theorem min_positive_period_and_max_value_of_f :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ T = 6 * π) ∧ 
  (∃ x, f x = sqrt 2) :=
by {
  sorry
}

end min_positive_period_and_max_value_of_f_l33_33097


namespace gift_exchange_equation_l33_33272

theorem gift_exchange_equation
  (x : ℕ)
  (total_gifts : ℕ)
  (H : total_gifts = 56)
  (H1 : 2 * total_gifts = x * (x - 1)) :
  x * (x - 1) = 56 :=
by
  sorry

end gift_exchange_equation_l33_33272


namespace find_length_XY_l33_33706

-- Define the scenario in Lean as per the given problem
def triangle_XYZ (X Y Z : Type) [ordered_field X] (XZ : X) (XY : X) (angleZ : X) (angleX : X) := 
  angleZ = 90 ∧ angleX = 30 ∧ XZ = 12

-- Express the theorem logically in Lean
theorem find_length_XY {X Y Z : Type} [ordered_field X] (XZ : X) (XY : X) (angleZ : X) (angleX : X) :
  triangle_XYZ X Y Z 12 24 90 30 → XY = 24 :=
by {
  sorry
}

end find_length_XY_l33_33706


namespace part1_part2_l33_33238

-- Let m be the cost price this year
-- Let x be the selling price per bottle
-- Assuming:
-- 1. The cost price per bottle increased by 4 yuan this year compared to last year.
-- 2. The quantity of detergent purchased for 1440 yuan this year equals to the quantity purchased for 1200 yuan last year.
-- 3. The selling price per bottle is 36 yuan with 600 bottles sold per week.
-- 4. Weekly sales increase by 100 bottles for every 1 yuan reduction in price.
-- 5. The selling price cannot be lower than the cost price.

-- Definition for improved readability:
def costPriceLastYear (m : ℕ) : ℕ := m - 4

-- Quantity equations
def quantityPurchasedThisYear (m : ℕ) : ℕ := 1440 / m
def quantityPurchasedLastYear (m : ℕ) : ℕ := 1200 / (costPriceLastYear m)

-- Profit Function
def profitFunction (m x : ℝ) : ℝ :=
  (x - m) * (600 + 100 * (36 - x))

-- Maximum Profit and Best Selling Price
def maxProfit : ℝ := 8100
def bestSellingPrice : ℝ := 33

theorem part1 (m : ℕ) (h₁ : 1440 / m = 1200 / costPriceLastYear m) : m = 24 := by
  sorry  -- Will be proved later

theorem part2 (m : ℝ) (x : ℝ)
    (h₀ : m = 24)
    (hx : 600 + 100 * (36 - x) > 0)
    (hx₁ : x ≥ m)
    : profitFunction m x ≤ maxProfit ∧ (∃! (y : ℝ), y = bestSellingPrice ∧ profitFunction m y = maxProfit) := by
  sorry  -- Will be proved later

end part1_part2_l33_33238


namespace people_left_of_Kolya_l33_33541

/-- Given:
    1. There are 12 people to the right of Kolya.
    2. There are 20 people to the left of Sasha.
    3. There are 8 people to the right of Sasha.
    4. The total number of people in the class (including Sasha) is 29.

    Prove:
    The number of people to the left of Kolya is 16.
-/
theorem people_left_of_Kolya : 
  ∀ (total_people right_of_Kolya left_of_Sasha right_of_Sasha : ℕ),
  right_of_Kolya = 12 →
  left_of_Sasha = 20 →
  right_of_Sasha = 8 →
  total_people = 29 →
  left_of_Kolya := total_people - right_of_Kolya - 1
  left_of_Kolya = 16 :=
by
  intros
  sorry

end people_left_of_Kolya_l33_33541


namespace problem_statement_l33_33343

noncomputable def f : ℕ+ → ℝ := sorry

theorem problem_statement (x : ℕ+) :
  (f 1 = 1) →
  (∀ x, f (x + 1) = (2 * f x) / (f x + 2)) →
  f x = 2 / (x + 1) := 
sorry

end problem_statement_l33_33343


namespace sum_in_base_b_l33_33860

noncomputable def s_in_base (b : ℕ) := 13 + 15 + 17

theorem sum_in_base_b (b : ℕ) (h : (13 * 15 * 17 : ℕ) = 4652) : s_in_base b = 51 := by
  sorry

end sum_in_base_b_l33_33860


namespace max_value_of_quadratic_l33_33416

theorem max_value_of_quadratic (a : ℝ) (h₀ : a ≠ 0) (h₁ : ∃ x ∈ set.Icc (0:ℝ) 3, ∀ x' ∈ set.Icc (0:ℝ) 3, a * x^2 - 2 * a * x ≤ a * x'^2 - 2 * a * (x')) : a = -3 ∨ a = 1 :=
sorry

end max_value_of_quadratic_l33_33416


namespace remainder_of_polynomial_l33_33611

-- Declare the polynomial
def f (x : ℝ) : ℝ := x^4 - 6 * x^3 + 12 * x^2 + 20 * x - 8

-- State the theorem using the Remainder Theorem
theorem remainder_of_polynomial :
  let f := λ x : ℝ, x^4 - 6 * x^3 + 12 * x^2 + 20 * x - 8 
  in f 4 = 136 :=
by
  sorry

end remainder_of_polynomial_l33_33611


namespace arithmetic_sequence_sum_not_always_decreasing_l33_33747

theorem arithmetic_sequence_sum_not_always_decreasing (n : ℕ) (h : n ≥ 1) :
  ¬ (∀ (m ≥ 1), (m ≤ n → (-2 * m + 7) + (-2 * (m - 1) + 7) + ... + (-2 * 1 + 7) ≥ (-2 * (m + 1) + 7) + (-2 * m + 7) + ... + (-2 * 2 + 7))) :=
by
  sorry

end arithmetic_sequence_sum_not_always_decreasing_l33_33747


namespace real_values_of_x_l33_33315

theorem real_values_of_x (x : ℝ) (h : x ≠ 4) :
  (x * (x + 1) / (x - 4)^2 ≥ 15) ↔ (x ≤ 3 ∨ (40/7 < x ∧ x < 4) ∨ x > 4) :=
by sorry

end real_values_of_x_l33_33315


namespace min_positive_period_and_max_value_l33_33098

noncomputable def f (x : ℝ) : ℝ := Real.sin (x / 3) + Real.cos (x / 3)

theorem min_positive_period_and_max_value :
  (∃ T > 0, ∀ x, f (x + T) = f x) ∧ ∃ x, f x = sqrt 2 :=
by
  sorry

end min_positive_period_and_max_value_l33_33098


namespace perpendicular_line_x_intercept_l33_33589

theorem perpendicular_line_x_intercept (x y : ℝ) :
  (4 * x + 5 * y = 10) →
  (1 * y + 0 = y → y = (5 / 4) * x - 3) →
  y = 0 →
  x = 12 / 5 :=
begin
  sorry
end

end perpendicular_line_x_intercept_l33_33589


namespace sum_lt_or_eq_03_l33_33944

-- Define the set of numbers
def number_set : Set ℚ := {0.1, 0.8, 1/2, 0.9}

-- Define the predicate for the numbers less than or equal to 0.3
def less_or_equal_03 (x : ℚ) : Prop := x ≤ 0.3

-- Define the set of numbers from the original set that are less than or equal to 0.3
def filtered_set : Set ℚ := {x | x ∈ number_set ∧ less_or_equal_03 x}

-- Define the sum of the filtered set
noncomputable def sum_filtered_set : ℚ := filtered_set.to_finset.sum id

-- Statement of the problem: Prove the sum of the numbers less than or equal to 0.3 is 0.1
theorem sum_lt_or_eq_03 : sum_filtered_set = 0.1 := by
  sorry

end sum_lt_or_eq_03_l33_33944


namespace probability_at_least_one_8_rolled_l33_33957

theorem probability_at_least_one_8_rolled :
  let total_outcomes := 64
  let no_8_outcomes := 49
  (total_outcomes - no_8_outcomes) / total_outcomes = 15 / 64 :=
by
  let total_outcomes := 8 * 8
  let no_8_outcomes := 7 * 7
  have h1 : total_outcomes = 64 := by norm_num
  have h2 : no_8_outcomes = 49 := by norm_num
  rw [← h1, ← h2]
  norm_num
  sorry

end probability_at_least_one_8_rolled_l33_33957


namespace min_positive_period_and_max_value_l33_33107

noncomputable def f (x : ℝ) : ℝ := Real.sin (x / 3) + Real.cos (x / 3)

theorem min_positive_period_and_max_value :
  (∃ T > 0, ∀ x, f (x + T) = f x) ∧ ∃ x, f x = sqrt 2 :=
by
  sorry

end min_positive_period_and_max_value_l33_33107


namespace object_speed_conversion_l33_33677

theorem object_speed_conversion 
  (distance : ℝ)
  (velocity : ℝ) 
  (conversion_factor : ℝ) 
  (distance_in_km : ℝ)
  (time_in_seconds : ℝ) 
  (time_in_minutes : ℝ) 
  (speed_in_kmh : ℝ) :
  distance = 200 ∧ 
  velocity = 1/3 ∧ 
  time_in_seconds = distance / velocity ∧ 
  time_in_minutes = time_in_seconds / 60 ∧ 
  conversion_factor = 3600 * 0.001 ∧ 
  speed_in_kmh = velocity * conversion_factor ↔ 
  speed_in_kmh = 0.4 :=
by sorry

end object_speed_conversion_l33_33677


namespace find_lambda_l33_33399

variable (λ : ℝ)
def a : ℝ → ℝ × ℝ × ℝ := λ _ => (1, -2, 3)
def b : ℝ → ℝ × ℝ × ℝ := λ λ => (λ - 1, 3 - λ, -6)
def parallel (v1 v2 : ℝ × ℝ × ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ v1 = (k * v2.1, k * v2.2, k * v2.3)

theorem find_lambda (λ : ℝ) (h : parallel (a 0) (b λ)) : λ = -1 := by
  sorry

end find_lambda_l33_33399


namespace interest_earned_l33_33827

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r) ^ n

theorem interest_earned (P : ℝ) (r : ℝ) (n : ℕ) (A : ℝ) (I : ℝ):
  P = 1500 → r = 0.12 → n = 4 →
  A = compound_interest P r n →
  I = A - P →
  I = 862.2 :=
by
  intros hP hr hn hA hI
  sorry

end interest_earned_l33_33827


namespace alex_initially_had_108_l33_33445

noncomputable def initial_peanuts (a b c : ℕ) (r : ℚ) (a' b' c' : ℕ) : Prop := 
  a + b + c = 444 ∧ 
  b = (a : ℚ) * r ∧ 
  c = (a : ℚ) * r^2 ∧ 
  a' = a - 5 ∧ 
  b' = b - 9 ∧ 
  c' = c - 25 ∧ 
  a' + b' + c' = 405 ∧ 
  (b' = a' + d ∧ c' = a' + 2 * d) → 
  a = 108

theorem alex_initially_had_108 (a b c : ℕ) (r : ℚ) (a' b' c' : ℕ) : 
  initial_peanuts a b c r a' b' c' :=
sorry

end alex_initially_had_108_l33_33445


namespace min_period_and_max_value_l33_33122

def f (x : ℝ) := Real.sin (x / 3) + Real.cos (x / 3)

theorem min_period_and_max_value :
  (∀ T > 0, T ≠ 6 * Real.pi → ¬∀ x : ℝ, f(x + T) = f(x)) ∧
  ∃ x : ℝ, f(x) = sqrt 2 :=
by
  sorry

end min_period_and_max_value_l33_33122


namespace triangle_area_l33_33990

-- Define the sides of the triangle
def a : ℝ := 30
def b : ℝ := 21
def c : ℝ := 10

-- Define the semi-perimeter
def s : ℝ := (a + b + c) / 2

-- Define the area using Heron's formula
def area : ℝ := Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Statement to prove: the area of the triangle with given sides equals 17.25 square centimeters
theorem triangle_area :
  area = 17.25 := 
by 
  sorry

end triangle_area_l33_33990


namespace triangle_inequality_l33_33999

theorem triangle_inequality
  (A B C D : Point)
  (h_triangle : is_triangle A B C)
  (h_in : is_interior D A B C)
  (angle_A : ℝ)
  (h_angle_decision: if angle_A < 90 then true else false) :
  let BC := dist B C in
  let AD := dist A D in
  let BD := dist B D in
  let CD := dist C D in
  let minor_dist := min (min AD BD) CD in
  BC / minor_dist ≥ 
  (if h_angle_decision 
    then 2 * sin angle_A 
    else 2) :=
sorry

end triangle_inequality_l33_33999


namespace range_of_function_l33_33606

noncomputable def f (x : ℝ) : ℝ := (x^2 + 5*x + 6) / (x + 2)

theorem range_of_function : 
  (∀ x : ℝ, x ≠ -2 → f x ≠ 1) ∧
  (∀ y : ℝ, y ≠ 1 → ∃ x : ℝ, f x = y) :=
sorry

end range_of_function_l33_33606


namespace domain_of_sqrt_function_l33_33319

def quadratic_expr (x : ℝ) : ℝ :=
  -15 * x^2 - 14 * x + 21

def sqrt_function (x : ℝ) : ℝ :=
  real.sqrt (quadratic_expr x)

theorem domain_of_sqrt_function :
  (∀ x, (√(quadratic_expr x)).is_real ↔ x ∈ (set.Iic (-7/5)) ∪ (set.Ici 1)) :=
by
  intros x
  have h : quadratic_expr x ≥ 0 ↔ x ∈ (set.Iic (-7/5)) ∪ (set.Ici 1),
  { 
    sorry 
  }
  simp_rw [sqrt_function, real.sqrt_nonneg_iff, h]
  sorry

end domain_of_sqrt_function_l33_33319


namespace simplified_value_l33_33613

-- Define the given expression
def expr := (10^0.6) * (10^0.4) * (10^0.4) * (10^0.1) * (10^0.5) / (10^0.3)

-- State the theorem
theorem simplified_value : expr = 10^1.7 :=
by
  sorry -- Proof omitted

end simplified_value_l33_33613


namespace construction_rates_construction_cost_l33_33184

-- Defining the conditions as Lean hypotheses

def length := 1650
def diff_rate := 30
def time_ratio := 3/2

-- Daily construction rates (questions answered as hypotheses as well)
def daily_rate_A := 60
def daily_rate_B := 90

-- Additional conditions for cost calculations
def cost_A_per_day := 90000
def cost_B_per_day := 120000
def total_days := 14
def alone_days_A := 5

-- Problem stated as proofs to be completed
theorem construction_rates :
  (∀ (x : ℕ), x = daily_rate_A ∧ (x + diff_rate) = daily_rate_B ∧ 
  (1650 / (x + diff_rate)) * (3/2) = (1650 / x) → 
  60 = daily_rate_A ∧ (60 + 30) = daily_rate_B ) :=
by sorry

theorem construction_cost :
  (∀ (m : ℕ), m = alone_days_A ∧ 
  (cost_A_per_day * total_days + cost_B_per_day * (total_days - alone_days_A)) / 1000 = 2340) :=
by sorry

end construction_rates_construction_cost_l33_33184


namespace range_of_function_l33_33607

noncomputable def f (x : ℝ) : ℝ := (x^2 + 5*x + 6) / (x + 2)

theorem range_of_function : 
  (∀ x : ℝ, x ≠ -2 → f x ≠ 1) ∧
  (∀ y : ℝ, y ≠ 1 → ∃ x : ℝ, f x = y) :=
sorry

end range_of_function_l33_33607


namespace sum_of_intersections_l33_33753

noncomputable def f (x : ℝ) := sorry  -- Placeholder for the actual function
def intersect_points (x : ℝ) := (x, f x)

theorem sum_of_intersections (h : ∀ x, f (4 - x) = -f x) :
  let points := { (2 - y, f (2 - y)) | y : ℝ } ∩ { (x, 1 / (2 - x)) | x : ℝ }
    ∑ p in points, p.1 + p.2 = points.card :=
by {
  sorry  
}

end sum_of_intersections_l33_33753


namespace minimally_intersecting_triples_modulo_1000_eq_344_l33_33454

def minimally_intersecting_triples_count_modulo : ℕ :=
  let total_count := 57344
  total_count % 1000

theorem minimally_intersecting_triples_modulo_1000_eq_344 :
  minimally_intersecting_triples_count_modulo = 344 := by
  sorry

end minimally_intersecting_triples_modulo_1000_eq_344_l33_33454


namespace geometry_problems_l33_33568

-- Given conditions
structure TriangleConfig where
  A B C D E F : Point
  AB BC : ℝ
  h1 : circle_through A B tangent BC
  h2 : circle_through B C tangent AB
  h3 : chord BD intersect AC at E
  h4 : chord AD intersects_other_circle_at F
  h5 : AB = 5
  h6 : BC = 9

-- Lean 4 statement for proving the ratio of AE to EC
def ratio_AE_EC (cfg : TriangleConfig) : Prop :=
  (segment_ratio cfg.AE cfg.EC) = (25 / 81)

-- Lean 4 statement for proving the areas of triangles are equal
def areas_equal (cfg : TriangleConfig) : Prop :=
  (area_triangle cfg.A cfg.B cfg.C) = (area_triangle cfg.A cfg.B cfg.F)

theorem geometry_problems (cfg : TriangleConfig) : ratio_AE_EC cfg ∧ areas_equal cfg :=
by
  sorry

end geometry_problems_l33_33568


namespace systematic_sampling_first_number_l33_33717

theorem systematic_sampling_first_number
    (n : ℕ)  -- total number of products
    (k : ℕ)  -- sample size
    (common_diff : ℕ)  -- common difference in the systematic sample
    (x : ℕ)  -- an element in the sample
    (first_num : ℕ)  -- first product number in the sample
    (h1 : n = 80)  -- total number of products is 80
    (h2 : k = 5)  -- sample size is 5
    (h3 : common_diff = 16)  -- common difference is 16
    (h4 : x = 42)  -- 42 is in the sample
    (h5 : x = common_diff * 2 + first_num)  -- position of 42 in the arithmetic sequence
: first_num = 10 := 
sorry

end systematic_sampling_first_number_l33_33717


namespace contestant_A_wins_by_100_meters_l33_33993

noncomputable theory

def race_distance : ℕ := 600
def speed_ratio_A : ℕ := 5
def speed_ratio_B : ℕ := 4
def head_start_A : ℕ := 100

theorem contestant_A_wins_by_100_meters
  (race_distance : ℕ) (speed_ratio_A speed_ratio_B head_start_A : ℕ)
  (h_ratio : speed_ratio_A = 5 ∧ speed_ratio_B = 4)
  (h_head_start : head_start_A = 100)
  (h_race_distance : race_distance = 600) :
  (race_distance - (race_distance - head_start_A) * speed_ratio_B / speed_ratio_A + head_start_A = 100) :=
by sorry

end contestant_A_wins_by_100_meters_l33_33993


namespace distance_from_dormitory_to_city_l33_33994

theorem distance_from_dormitory_to_city (D : ℝ) :
  (1 / 4) * D + (1 / 2) * D + 10 = D → D = 40 :=
by
  intro h
  sorry

end distance_from_dormitory_to_city_l33_33994


namespace unique_num_not_in_range_l33_33847

theorem unique_num_not_in_range (a b c d : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0)
  (h5 : ∀ x : ℝ, x ≠ -d / c → f (f x) = x) (h6 : f 5 = 5) (h7 : f 50 = 50)
  : ∃! y, ¬ (∃ x : ℝ, f x = y) ∧ y = 27.5 :=
by
  let f (x : ℝ) := (a * x + b) / (c * x + d)
  sorry

end unique_num_not_in_range_l33_33847


namespace A_eq_B_l33_33470

namespace SetsEquality

open Set

def A : Set ℝ := {x | ∃ a : ℝ, x = 5 - 4 * a + a^2}
def B : Set ℝ := {y | ∃ b : ℝ, y = 4 * b^2 + 4 * b + 2}

theorem A_eq_B : A = B := by
  sorry

end SetsEquality

end A_eq_B_l33_33470


namespace min_period_and_max_value_l33_33057

def f (x : ℝ) : ℝ := Real.sin (x / 3) + Real.cos (x / 3)

noncomputable def minimum_positive_period (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x : ℝ, f (x + T) = f x ∧ T > 0 ∧ ∀ T' : ℝ, (T' > 0 ∧ ∀ x : ℝ, f (x + T') = f x) → T' ≥ T

noncomputable def is_max_value (f : ℝ → ℝ) (M : ℝ) : Prop :=
  ∀ x : ℝ, f x ≤ M ∧ ∃ x : ℝ, f x = M

theorem min_period_and_max_value :
  minimum_positive_period f (6 * Real.pi) ∧ is_max_value f (Real.sqrt 2) := by
  sorry

end min_period_and_max_value_l33_33057


namespace probability_at_least_one_eight_l33_33954

theorem probability_at_least_one_eight :
  let total_outcomes := 64 in
  let outcomes_without_8 := 49 in
  let favorable_outcomes := total_outcomes - outcomes_without_8 in
  let probability := (favorable_outcomes : ℚ) / total_outcomes in
  probability = 15 / 64 :=
by
  let total_outcomes := 64
  let outcomes_without_8 := 49
  let favorable_outcomes := total_outcomes - outcomes_without_8
  let probability := (favorable_outcomes : ℚ) / total_outcomes
  show probability = 15 / 64
  sorry

end probability_at_least_one_eight_l33_33954


namespace min_pos_period_max_value_l33_33148

noncomputable def f (x : ℝ) := Real.sin (x / 3) + Real.cos (x / 3)

theorem min_pos_period_max_value :
  (∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ T) ∧ T = 6 * Real.pi) ∧ 
  (∀ x, f x ≤ sqrt 2 ∧ (∃ x, f x = sqrt 2)) :=
by
  sorry

end min_pos_period_max_value_l33_33148


namespace heartbeats_during_race_l33_33666

-- Define the conditions as constants
def heart_rate := 150 -- beats per minute
def race_distance := 26 -- miles
def pace := 5 -- minutes per mile

-- Formulate the statement
theorem heartbeats_during_race :
  heart_rate * (race_distance * pace) = 19500 :=
by
  sorry

end heartbeats_during_race_l33_33666


namespace expression_evaluation_l33_33407

theorem expression_evaluation (x y : ℝ) (h₁ : x > y) (h₂ : y > 0) : 
    (x^(2*y) * y^x) / (y^(2*x) * x^y) = (x / y)^(y - x) :=
by
  sorry

end expression_evaluation_l33_33407


namespace least_values_3198_l33_33205

theorem least_values_3198 (x y : ℕ) (hX : ∃ n : ℕ, 3198 + n * 9 = 27)
                         (hY : ∃ m : ℕ, 3198 + m * 11 = 11) :
  x = 6 ∧ y = 8 :=
by
  sorry

end least_values_3198_l33_33205


namespace fibonacci_tenth_term_is_89_l33_33895

noncomputable def fibonacci : ℕ → ℕ
| 1 := 1
| 2 := 2
| n := fibonacci (n - 1) + fibonacci (n - 2)

theorem fibonacci_tenth_term_is_89 : fibonacci 10 = 89 :=
by sorry

end fibonacci_tenth_term_is_89_l33_33895


namespace solution_set_ineq_l33_33745

-- Definitions based on the problem conditions
def f : ℝ → ℝ := sorry

axiom domain_f : ∀ x : ℝ, x ∈ (λ y, y ∈ ℝ)
axiom f_property : ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) > 3
axiom f_at_5 : f 5 = 18

-- Statement to prove the solution set of the given inequality
theorem solution_set_ineq (x : ℝ) : f (3 * x - 1) > 9 * x ↔ x > 2 :=
by
  sorry

end solution_set_ineq_l33_33745


namespace hexagon_angles_equal_l33_33308

theorem hexagon_angles_equal (A B C D E F: Point) 
  (convex: IsConvex (hexagon A B C D E F)) 
  (property: ∀ (P Q : Point), OppositeSides P Q (hexagon A B C D E F) → 
                    dist (midpoint P Q) = (sqrt 3 / 2) * (length P + length Q)) : 
  ∀ (α β γ δ ε ζ : ℝ), 
    (angle A B C = 120 ∧ angle B C D = 120 ∧ angle C D E = 120 ∧ 
     angle D E F = 120 ∧ angle E F A = 120 ∧ angle F A B = 120) :=
by
  sorry

end hexagon_angles_equal_l33_33308


namespace compute_value_of_expression_l33_33466

theorem compute_value_of_expression (p q : ℝ) (h1 : 3 * p^2 - 7 * p + 1 = 0) (h2 : 3 * q^2 - 7 * q + 1 = 0) :
  (9 * p^3 - 9 * q^3) / (p - q) = 46 :=
sorry

end compute_value_of_expression_l33_33466


namespace calculate_prime_factors_l33_33563
open scoped BigOperators

def prime_factors_count (n : ℕ) : ℕ :=
  ∑ p in (unique_factorization_monoid.factors n).to_finset, (unique_factorization_monoid.factors n).count p

theorem calculate_prime_factors (x y : ℕ)
    (h1 : log 10 x + 3 * log 10 (Nat.gcd x y) = 90)
    (h2 : log 10 y + 3 * log 10 (Nat.lcm x y) = 690) :
    2 * prime_factors_count x + 3 * prime_factors_count y = 555 := by
  sorry

end calculate_prime_factors_l33_33563


namespace new_median_after_adding_9_l33_33232

theorem new_median_after_adding_9 
  (s : Multiset ℕ) 
  (h_pos : ∀ x ∈ s, 0 < x) 
  (h_card : s.card = 6) 
  (h_mean : s.sum = 30) 
  (h_mode : ∃! k, s.count k > s.count 4 ∧ k = 4) 
  (h_median: (s.sort (· ≤ ·)).nth 3 = some 5) :
  let s' := s + {9}
  in (s'.sort (· ≤ ·)).nth 3 = some 5 := 
by
  sorry

end new_median_after_adding_9_l33_33232


namespace find_boxed_boxed_13_l33_33715

/-- Definition of boxed function which computes the sum of positive factors -/
def boxed (n : ℕ) : ℕ :=
  (Finset.filter (λ d, n % d = 0) (Finset.range (n + 1))).sum id

theorem find_boxed_boxed_13 : boxed (boxed 13) = 24 := by
  sorry

end find_boxed_boxed_13_l33_33715


namespace solve_system_l33_33013

def system_of_equations (x y : ℝ) :=
x^4 - y^4 = 3 * real.sqrt (|y|) - 3 * real.sqrt (|x|)
∧ x^2 - 2 * x * y = 27

theorem solve_system : 
  ∃ (x y : ℝ), system_of_equations x y ∧ ((x = 3 ∧ y = -3) ∨ (x = -3 ∧ y = 3)) :=
sorry

end solve_system_l33_33013


namespace min_period_and_max_value_l33_33063

def f (x : ℝ) : ℝ := Real.sin (x / 3) + Real.cos (x / 3)

noncomputable def minimum_positive_period (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x : ℝ, f (x + T) = f x ∧ T > 0 ∧ ∀ T' : ℝ, (T' > 0 ∧ ∀ x : ℝ, f (x + T') = f x) → T' ≥ T

noncomputable def is_max_value (f : ℝ → ℝ) (M : ℝ) : Prop :=
  ∀ x : ℝ, f x ≤ M ∧ ∃ x : ℝ, f x = M

theorem min_period_and_max_value :
  minimum_positive_period f (6 * Real.pi) ∧ is_max_value f (Real.sqrt 2) := by
  sorry

end min_period_and_max_value_l33_33063


namespace general_formula_sum_of_squares_bounded_l33_33350

-- Define the sequence with the given recurrence
def a : ℕ → ℝ
| 0       := 1
| (n + 1) := 2 * a n / (2 + a n)

theorem general_formula (n : ℕ) : a (n + 1) = 2 / (n + 2) :=
sorry

theorem sum_of_squares_bounded (n : ℕ) : (finset.range (n + 1)).sum (λ i, (a i) ^ 2) < 3 :=
sorry

end general_formula_sum_of_squares_bounded_l33_33350


namespace integral_of_3x_l33_33693

theorem integral_of_3x : ∫ x in -1..8, 3 * x = 45 / 4 :=
by 
  sorry

end integral_of_3x_l33_33693


namespace solution_set_of_inequality_l33_33550

theorem solution_set_of_inequality (x : ℝ) : (x - 1) * (2 - x) > 0 ↔ 1 < x ∧ x < 2 := by
  sorry

end solution_set_of_inequality_l33_33550


namespace min_period_of_f_max_value_of_f_l33_33158

def f : ℝ → ℝ := λ x, Real.sin (x / 3) + Real.cos (x / 3)

theorem min_period_of_f : ∃ T > 0, (∀ x, f (x + T) = f x) ∧ T = 6 * Real.pi :=
by
  sorry

theorem max_value_of_f : ∃ x, f x = Real.sqrt 2 :=
by
  sorry

end min_period_of_f_max_value_of_f_l33_33158


namespace slower_bike_longer_time_by_1_hour_l33_33583

/-- Speed of the slower bike in kmph -/
def speed_slow : ℕ := 60

/-- Speed of the faster bike in kmph -/
def speed_fast : ℕ := 64

/-- Distance both bikes travel in km -/
def distance : ℕ := 960

/-- Time taken to travel the distance by a bike going at a certain speed -/
def time (speed : ℕ) : ℕ :=
  distance / speed

/-- Proof that the slower bike takes 1 hour longer to cover the distance compared to the faster bike -/
theorem slower_bike_longer_time_by_1_hour : 
  (time speed_slow) = (time speed_fast) + 1 := by
sorry

end slower_bike_longer_time_by_1_hour_l33_33583


namespace triangle_count_l33_33771

-- Defining the coordinate grid with points (i, j) where i and j are integers from 1 to 4
def coordinate_points : List (ℤ × ℤ) := 
  List.product (List.range 1 5) (List.range 1 5)

-- Defining a function to count number of triangles with positive area
def count_positive_area_triangles (points : List (ℤ × ℤ)) : ℕ :=
  let all_combinations := points.combination 3
  let non_triangular_count :=
    let horizontal := 4 * (binom 4 3)
    let vertical := 4 * (binom 4 3)
    let diagonals := 2 * (binom 4 3)
    horizontal + vertical + diagonals
  all_combinations.length - non_triangular_count

theorem triangle_count : count_positive_area_triangles coordinate_points = 520 := by
  sorry

end triangle_count_l33_33771


namespace cylinder_volume_ratio_l33_33224

theorem cylinder_volume_ratio:
  (∀ l w : ℝ, l = 6 ∧ w = 10 →
      let r_A := 3 / (Real.pi) in
      let h_A := 10 in
      let V_A := Real.pi * r_A^2 * h_A in
      let r_B := 5 / (Real.pi) in
      let h_B := 6 in
      let V_B := Real.pi * r_B^2 * h_B in
    V_B / V_A = 5 / 3) :=
by
  intros l w h,
  sorry

end cylinder_volume_ratio_l33_33224


namespace unique_sequence_l33_33699

theorem unique_sequence (a : ℕ → ℝ) 
  (h1 : a 0 = 1) 
  (h2 : ∀ n : ℕ, a n > 0) 
  (h3 : ∀ n : ℕ, a n - a (n + 1) = a (n + 2)) : 
  ∀ n : ℕ, a n = ( (-1 + Real.sqrt 5) / 2)^n := 
sorry

end unique_sequence_l33_33699


namespace min_positive_period_f_max_value_f_l33_33075

noncomputable def f (x : ℝ) : ℝ := sin (x / 3) + cos (x / 3)

theorem min_positive_period_f : ∃ T > 0, (∀ x, f (x + T) = f x) ∧ ∀ T', (T' > 0 ∧ ∀ x, f (x + T') = f x) → T ≤ T' :=
  sorry

theorem max_value_f : ∃ M, (∀ x, f x ≤ M) ∧ (∀ ε > 0, ∃ x, M - ε < f x) ∧ M = sqrt 2 :=
  sorry

end min_positive_period_f_max_value_f_l33_33075


namespace inequality_proof_l33_33339

-- Definitions based on the given conditions.
def a : ℝ := Real.log 3 / Real.log 2  -- a = log_2(3)
def b : ℝ := Real.log 4 / Real.log 3  -- b = log_3(4)
def c : ℝ := 3 / 2                    -- c = 3/2

-- The theorem statement proving the required inequality.
theorem inequality_proof : b < c ∧ c < a := by
  sorry

end inequality_proof_l33_33339


namespace min_positive_period_and_max_value_l33_33082

def f (x : ℝ) := Real.sin (x / 3) + Real.cos (x / 3)

theorem min_positive_period_and_max_value : 
  (∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T' = T)) 
  ∧ 
  (∀ x, f x ≤ sqrt 2 ∧ (∃ x, f x = sqrt 2)) :=
sorry

end min_positive_period_and_max_value_l33_33082


namespace min_period_max_value_f_l33_33133

def f (x : ℝ) : ℝ := sin (x / 3) + cos (x / 3)

theorem min_period_max_value_f :
  ∃ T M, (∀ x, f (x + T) = f x) ∧ 0 < T ∧
    0 < M ∧ (∀ x, f x ≤ M) ∧ (∃ x, f x = M) ∧ 
    T = 6 * π ∧ M = √2 := sorry

end min_period_max_value_f_l33_33133


namespace min_period_of_f_max_value_of_f_l33_33159

def f : ℝ → ℝ := λ x, Real.sin (x / 3) + Real.cos (x / 3)

theorem min_period_of_f : ∃ T > 0, (∀ x, f (x + T) = f x) ∧ T = 6 * Real.pi :=
by
  sorry

theorem max_value_of_f : ∃ x, f x = Real.sqrt 2 :=
by
  sorry

end min_period_of_f_max_value_of_f_l33_33159


namespace workers_per_block_l33_33795

theorem workers_per_block (total_budget worth_of_gift num_blocks : ℕ) (h1 : total_budget = 4000) (h2 : worth_of_gift = 4) (h3 : num_blocks = 10) : 
  let total_gifts := total_budget / worth_of_gift in
  let workers_per_block := total_gifts / num_blocks in
  workers_per_block = 100 :=
by
  sorry

end workers_per_block_l33_33795


namespace exists_xi_l33_33909

variable (f : ℝ → ℝ)
variable (hf_diff : ∀ x, DifferentiableAt ℝ f x)
variable (hf_twice_diff : ∀ x, DifferentiableAt ℝ (deriv f) x)
variable (hf₀ : f 0 = 2)
variable (hf_prime₀ : deriv f 0 = -2)
variable (hf₁ : f 1 = 1)

theorem exists_xi (h0 : f 0 = 2) (h1 : deriv f 0 = -2) (h2 : f 1 = 1) :
  ∃ ξ ∈ Set.Ioo 0 1, f ξ * deriv f ξ + deriv (deriv f) ξ = 0 :=
sorry

end exists_xi_l33_33909


namespace suitcase_lock_settings_count_l33_33659

-- Define the conditions as functions and predicates
def valid_dial (n : ℕ) : Prop := n >= 0 ∧ n <= 7
def valid_setting (dials : List ℕ) : Prop :=
  dials.length = 4 ∧ ∀ (i : ℕ), i < dials.length - 1 → (abs (dials[i + 1] - dials[i]) ≠ 1)

-- Define the main theorem
theorem suitcase_lock_settings_count : ∃ (count : ℕ), count = 1728 ∧ ∀ (setting : List ℕ), valid_setting setting → setting.length = 4 → count = (List.range 8).product (λ d1, 
  (List.filter (λ d2, abs (d2 - d1) ≠ 1) (List.range 8)).product (λ d2, 
  (List.filter (λ d3, abs (d3 - d2) ≠ 1) (List.range 8)).product (λ d3, 
  (List.filter (λ d4, abs (d4 - d3) ≠ 1) (List.range 8)).length)))

-- Placeholder for the proof
sorry

end suitcase_lock_settings_count_l33_33659


namespace vector_v_exists_l33_33326

def proj (u v : ℝ × ℝ) : ℝ × ℝ := 
  let c := (u.1 * v.1 + u.2 * v.2) / (u.1 * u.1 + u.2 * u.2)
  (c * u.1, c * u.2)

theorem vector_v_exists (v : ℝ × ℝ) (h₁ : proj ⟨3, 2⟩ v = ⟨6, 4⟩) (h₂ : proj ⟨1, 4⟩ v = ⟨2, 8⟩) : 
  v = (3.6, 7.6) :=
by 
  sorry

end vector_v_exists_l33_33326


namespace line_through_point_and_equidistant_l33_33374

variables (x y : ℝ)

def point := (ℝ × ℝ)

noncomputable def line_equation_1 := (2 * x - y - 2 = 0)
noncomputable def line_equation_2 := (2 * x + 3 * y - 18 = 0)

theorem line_through_point_and_equidistant 
  (P : point) (A : point) (B : point) :
  (P = (3, 4)) →
  (A = (-2, 2)) →
  (B = (4, -2)) →
  (∃ k, line_equation_1) ∨ (∃ k, line_equation_2) :=
by
  sorry

end line_through_point_and_equidistant_l33_33374


namespace total_length_at_most_half_l33_33836

open MeasureTheory

noncomputable def unionDisjointSubintervals (I : Set (Set ℝ)) : Prop :=
  I.Finite ∧ ∀ i ∈ I, ∃ a b : ℝ, a < b ∧ i = Ioo a b ∧ ∀ j ∈ I, i ≠ j → disjoint i j

noncomputable def noDistance10 (S : Set ℝ) : Prop :=
  ∀ x y ∈ S, x ≠ y → |x - y| ≠ 1/10

theorem total_length_at_most_half
  (S : Set ℝ)
  (h1 : unionDisjointSubintervals S)
  (h2 : noDistance10 S) :
  MeasureTheory.Measure.measure S ≤ 1/2 := 
sorry

end total_length_at_most_half_l33_33836


namespace pell_equation_solutions_l33_33906

/-- 
Given \(x^2 - D y^2 = 1\), where \(D\) is a natural number that is not a perfect square, 
if \((x_0, y_0)\) is an integer solution to this equation, 
then the numbers defined by the formula \(x + y \sqrt{D} = (x_0 + y_0 \sqrt{D})^k\), 
where \(k = 1, 2, 3, \ldots\), are also solutions to this equation. 
Moreover, the obtained values \(x_0, x_1, x_2, \ldots, x_k, x_{k+1}, x_{k+2}, \ldots\) 
are related by the recurrence relation \(x_{k+2} = 2 x_0 \cdot x_{k+1} - x_k\).
-/
theorem pell_equation_solutions (D x_0 y_0 : ℤ) (k : ℕ) 
  (hD : nat_prime D) 
  (hx0 : x_0^2 - D * y_0^2 = 1) : 
  (∃ x y : ℤ, x^2 - D * y^2 = 1 ∧ 
    (x + y * (D.sqrt : ℝ)) = ((x_0 + y_0 * (D.sqrt : ℝ)) ^ k)) 
  ∧ ∀ k : ℕ, (x_{k+2} = 2 * x_0 * x_{k+1} - x_k) :=
sorry

end pell_equation_solutions_l33_33906


namespace sesame_seed_mass_l33_33516

theorem sesame_seed_mass :
  ∀ (n : ℕ) (m_seed_gram m_total_gram k : ℝ), n = 50000 → m_total_gram = 200 →
  k = 1000 →
  m_seed_gram = m_total_gram / n →
  (m_seed_gram / k) = 4 * 10^(-6) :=
by
  intros n m_seed_gram m_total_gram k hn hm_total hk hseed
  sorry

end sesame_seed_mass_l33_33516


namespace evaluate_cube_root_fraction_l33_33309

theorem evaluate_cube_root_fraction (h : 18.75 = 75 / 4) : (Real.cbrt (6 / 18.75)) = 2 / (Real.cbrt 25) :=
  sorry

end evaluate_cube_root_fraction_l33_33309


namespace trihedral_angle_bisectors_intersect_single_line_tetrahedron_bisector_lines_intersect_three_tetrahedron_bisectors_intersect_fourth_tetrahedron_bisectors_intersect_iff_sum_opposite_edges_equal_l33_33493

-- Part a)
theorem trihedral_angle_bisectors_intersect_single_line (α β γ : Type*) 
  [metric_space α] [metric_space β] [metric_space γ]
  (S A B C Sa Sb Sc : α) :
  -- Conditions
  (plane_angle BSC Sa ∧ plane_perpendicular_to Sa (plane SBC)) 
  ∧ (plane_angle ASC Sb ∧ plane_perpendicular_to Sb (plane SAC))
  ∧ (plane_angle ASB Sc ∧ plane_perpendicular_to Sc (plane SAB)) →
  -- Conclusion
  ∃ l : α, is_line (α ∩ β ∩ γ) (line_bisector S) := sorry

-- Part b)
theorem tetrahedron_bisector_lines_intersect (A1 A2 A3 A4 l1 l2 l3 l4 : Type*) 
  [metric_space A1] [metric_space A2] [metric_space A3] [metric_space A4]
  (B13 B24 : A1) :
  -- Conditions
  (intersect l1 l3 B13) →
  -- Conclusion
  ∃ B24 : A2, intersect l2 l4 B24 := sorry

-- Part c)
theorem three_tetrahedron_bisectors_intersect_fourth (A1 A2 A3 A4 l1 l2 l3 l4 : Type*) 
  [metric_space A1] [metric_space A2] [metric_space A3] [metric_space A4]
  (O : A1) :
  -- Conditions
  (intersect l1 l2 O ∧ intersect l2 l3 O) →
  -- Conclusion
  ∃ O : A2, intersect l4 (line_intersection l1 l2 l3) := sorry

-- Part d)
theorem tetrahedron_bisectors_intersect_iff_sum_opposite_edges_equal (AB CD AC BD AD BC : ℝ)
  (P : α) :
  -- Conditions and Conclusion
  (intersect_all l1 l2 l3 l4 P ↔ (AB + CD = AC + BD ∧ AC + BD = AD + BC))
:= sorry


end trihedral_angle_bisectors_intersect_single_line_tetrahedron_bisector_lines_intersect_three_tetrahedron_bisectors_intersect_fourth_tetrahedron_bisectors_intersect_iff_sum_opposite_edges_equal_l33_33493


namespace find_k_from_roots_ratio_l33_33921

theorem find_k_from_roots_ratio (k : ℝ) (x1 x2 : ℝ) 
  (h1 : x1 + x2 = -10)
  (h2 : x1 * x2 = k)
  (h3 : x1/x2 = 3) : 
  k = 18.75 := 
sorry

end find_k_from_roots_ratio_l33_33921


namespace sum_minimums_is_correct_l33_33864

def P (x : ℝ) (B C : ℝ) := x^2 + B * x + C
def Q (x : ℝ) (E F : ℝ) := x^2 + E * x + F

noncomputable def P_Q (x : ℝ) (B C E F : ℝ) := 
  P (Q x E F) B C = (x^2 + E * x + F)^2 + B * (x^2 + E * x + F) + C
noncomputable def Q_P (x : ℝ) (B C E F : ℝ) := 
  Q (P x B C) E F = (x^2 + B * x + C)^2 + E * (x^2 + B * x + C) + F

noncomputable def sum_minimums (B C : ℝ) (E F : ℝ) := 
  P (-B / 2) B C + Q (-E / 2) E F

theorem sum_minimums_is_correct (B C E F : ℝ) (hB : B = 88) (hE : E = 20) 
  (hP : P_Q (-19) hB C hE F ∧ P_Q (-13) hB C hE F ∧ P_Q (-7) hB C hE F ∧ P_Q (-1) hB C hE F)
  (hQ : Q_P (-53) hB C hE F ∧ Q_P (-47) hB C hE F ∧ Q_P (-41) hB C hE F ∧ Q_P (-35) hB C hE F) :
  sum_minimums hB C hE F = -74 := sorry

end sum_minimums_is_correct_l33_33864


namespace g_g_x_has_exactly_4_distinct_real_roots_l33_33460

noncomputable def g (d x : ℝ) : ℝ := x^2 + 8*x + d

theorem g_g_x_has_exactly_4_distinct_real_roots (d : ℝ) :
  (∃ x1 x2 x3 x4 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 ∧ g d (g d x1) = 0 ∧ g d (g d x2) = 0 ∧ g d (g d x3) = 0 ∧ g d (g d x4) = 0) ↔ d < 4 := by {
  sorry
}

end g_g_x_has_exactly_4_distinct_real_roots_l33_33460


namespace total_rowing_time_l33_33995

theorem total_rowing_time (s_b : ℕ) (s_s : ℕ) (d : ℕ) : 
  s_b = 9 → s_s = 6 → d = 170 → 
  (d / (s_b + s_s) + d / (s_b - s_s)) = 68 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end total_rowing_time_l33_33995


namespace find_k_l33_33922

-- Given conditions and hypothesis stated
axiom quadratic_eq (x k : ℝ) : x^2 + 10 * x + k = 0

def roots_in_ratio_3_1 (α β : ℝ) : Prop :=
  α / β = 3

-- Statement of the theorem to be proved
theorem find_k {α β k : ℝ} (h1 : quadratic_eq α k) (h2 : quadratic_eq β k)
               (h3 : α ≠ 0) (h4 : β ≠ 0) (h5 : roots_in_ratio_3_1 α β) :
  k = 18.75 :=
by
  sorry

end find_k_l33_33922


namespace unique_a_value_l33_33763

theorem unique_a_value (a : ℝ) :
  let M := { x : ℝ | x^2 = 2 }
  let N := { x : ℝ | a * x = 1 }
  N ⊆ M ↔ (a = 0 ∨ a = -Real.sqrt 2 / 2 ∨ a = Real.sqrt 2 / 2) :=
by
  sorry

end unique_a_value_l33_33763


namespace gift_exchange_equation_l33_33269

theorem gift_exchange_equation
  (x : ℕ)
  (total_gifts : ℕ)
  (H : total_gifts = 56)
  (H1 : 2 * total_gifts = x * (x - 1)) :
  x * (x - 1) = 56 :=
by
  sorry

end gift_exchange_equation_l33_33269


namespace pass_rate_correct_l33_33034

-- Conditions definitions
def is_passing_score (score : ℕ) : Prop := score >= 60

-- Assume we have a function that calculates the pass rate based on the histogram data
variable (histogram : List ℕ) -- List of scores from the histogram

noncomputable def pass_rate : ℕ := 
  let passing_scores := histogram.filter is_passing_score
  (passing_scores.length * 100) / histogram.length

-- Statement of the proof problem
theorem pass_rate_correct (h : histogram ≠ []): pass_rate histogram = sorry := 
by 
  sorry

end pass_rate_correct_l33_33034


namespace ellipse_distance_range_l33_33916

noncomputable def range_of_distances (major_axis : ℝ) (minor_axis : ℝ) : set ℝ :=
  { x : ℝ | 2 * (minor_axis / 2) ≤ x ∧ x ≤ 2 * (major_axis / 2) }

theorem ellipse_distance_range :
  range_of_distances 10 8 = { x : ℝ | 4 ≤ x ∧ x ≤ 5 } :=
by 
  sorry

end ellipse_distance_range_l33_33916


namespace kids_left_playing_l33_33181

-- Define the conditions
def initial_kids : ℝ := 22.0
def kids_went_home : ℝ := 14.0

-- Theorem statement: Prove that the number of kids left playing is 8.0
theorem kids_left_playing : initial_kids - kids_went_home = 8.0 :=
by
  sorry -- Proof is left as an exercise

end kids_left_playing_l33_33181


namespace min_period_max_value_f_l33_33137

def f (x : ℝ) : ℝ := sin (x / 3) + cos (x / 3)

theorem min_period_max_value_f :
  ∃ T M, (∀ x, f (x + T) = f x) ∧ 0 < T ∧
    0 < M ∧ (∀ x, f x ≤ M) ∧ (∃ x, f x = M) ∧ 
    T = 6 * π ∧ M = √2 := sorry

end min_period_max_value_f_l33_33137


namespace hypotenuse_length_l33_33620

-- Define the polynomial P(z) = z^3 + az + b
noncomputable def P (a b : ℂ) : Polynomial ℂ :=
Polynomial.C b + Polynomial.C a * Polynomial.X + Polynomial.X ^ 3

-- Conditions and question
theorem hypotenuse_length (a b z1 z2 z3 : ℂ) 
  (h_roots: Polynomial.roots (P a b) = {z1, z2, z3})
  (h_sum_zero: z1 + z2 + z3 = 0)
  (h_squares_sum: abs z1 ^ 2 + abs z2 ^ 2 + abs z3 ^ 2 = 250)
  (h_right_triangle: IsRightTriangle z1 z2 z3):
  abs (hypotenuse z1 z2 z3) = 5 * sqrt 15 := sorry

end hypotenuse_length_l33_33620


namespace probability_gx_ge_sqrt3_l33_33387

noncomputable def f (ω x : ℝ) : ℝ := sqrt 3 * sin (ω * x) + cos (ω * x)
noncomputable def g (ω x : ℝ) : ℝ := f ω (x + π / 6)

theorem probability_gx_ge_sqrt3 (ω : ℝ) (h₀ : ω > 0)
  (h₁ : ∃ d, d = π / 2 ∧ ∀ n : ℤ, f ω (n * d) = 0)
  (x : ℝ) (hx : x ∈ set.Icc 0 π) :
  ∃ P, P = 1 / 6 ∧ (set.probability {x | g ω x ≥ sqrt 3} (set.Icc 0 π) = P) :=
by sorry

end probability_gx_ge_sqrt3_l33_33387


namespace sum_of_possible_sums_l33_33476

-- Given conditions:
-- A quadratic equation with two distinct negative integer roots, and the product of the roots is 48
def isRoot (x : ℤ) (r s : ℤ) : Prop := x = -(r + s)
def isProduct (r s : ℤ) : Prop := r * s = 48
def isDistinctNegatives (r s : ℤ) : Prop := (r < 0) ∧ (s < 0) ∧ (r ≠ s)

-- Proof goal:
-- The sum of all possible distinct integers that could be the sum of the absolute values of the roots (which are negative) is 124
theorem sum_of_possible_sums : 
  ∑ n in {49, 26, 19, 16, 14}, n = 124 :=
by
  sorry

end sum_of_possible_sums_l33_33476


namespace cube_face_sharing_l33_33642

theorem cube_face_sharing (n : ℕ) :
  (∃ W B : ℕ, (W + B = n^3) ∧ (3 * W = 3 * B) ∧ W = B ∧ W = n^3 / 2) ↔ n % 2 = 0 :=
by
  sorry

end cube_face_sharing_l33_33642


namespace number_of_correct_statements_l33_33035

theorem number_of_correct_statements :
    ¬({0} = (∅ : Set ℕ)) ∧
    ¬(∅ ∈ ({0} : Set (Set ℕ))) ∧
    (∅ ⊆ ({0} : Set ℕ)) ∧
    ¬(∅ ⊈ ({0} : Set ℕ)) ∧
    ¬(0 ∈ (∅ : Set ℕ)) →
    1 = 1 :=
by
  intro h
  sorry

end number_of_correct_statements_l33_33035


namespace gcd_of_72_120_168_l33_33914

theorem gcd_of_72_120_168 : Nat.gcd (Nat.gcd 72 120) 168 = 24 := 
by
  sorry

end gcd_of_72_120_168_l33_33914


namespace smallest_possible_sector_angle_l33_33448

theorem smallest_possible_sector_angle :
  ∃ (a_1 d : ℤ), 
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ 10 → 
    ∃ a_i : ℤ, a_i = a_1 + (i - 1) * d ∧ a_i > 0) ∧
  ∑ i in (Finset.range 10).map (fun n => n + 1), (a_1 + n * d) = 360 ∧
  (a_1 > 0) ∧
  (∀ (b : ℤ), b > 0 → (∃ j : ℕ, 1 ≤ j ∧ j ≤ 10 → 
    a_1 + (j - 1) * d = b) → 27 ≤ b) := 
sorry

end smallest_possible_sector_angle_l33_33448


namespace katharina_order_is_correct_l33_33814

-- Define the mixed up order around a circle starting with L
def mixedUpOrder : List Char := ['L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S']

-- Define the positions and process of Jaxon's list generation
def jaxonList : List Nat := [1, 4, 7, 3, 8, 5, 2, 6]

-- Define the resulting order from Jaxon's process
def resultingOrder (initialList : List Char) (positions : List Nat) : List Char :=
  positions.map (λ i => initialList.get! (i - 1))

-- Define the function to prove Katharina's order
theorem katharina_order_is_correct :
  resultingOrder mixedUpOrder jaxonList = ['L', 'R', 'O', 'M', 'S', 'Q', 'N', 'P'] :=
by
  -- Proof omitted
  sorry

end katharina_order_is_correct_l33_33814


namespace double_integral_sin_cos_l33_33283

theorem double_integral_sin_cos :
  (∫ x in 0..π, ∫ y in 0..(1 + Real.cos x), y^2 * Real.sin x) = 4 / 3 := sorry

end double_integral_sin_cos_l33_33283


namespace plan_b_more_cost_effective_l33_33554

noncomputable def fare (x : ℝ) : ℝ :=
if x < 3 then 5
else if x <= 10 then 1.2 * x + 1.4
else 1.8 * x - 4.6

theorem plan_b_more_cost_effective :
  let plan_a := 2 * fare 15
  let plan_b := 3 * fare 10
  plan_a > plan_b :=
by
  let plan_a := 2 * fare 15
  let plan_b := 3 * fare 10
  sorry

end plan_b_more_cost_effective_l33_33554


namespace side_length_of_square_l33_33518

theorem side_length_of_square 
  (x : ℝ) 
  (h₁ : 4 * x = 2 * (x * x)) :
  x = 2 :=
by 
  sorry

end side_length_of_square_l33_33518


namespace sandy_hours_per_day_l33_33496

theorem sandy_hours_per_day (total_hours : ℕ) (days : ℕ) (H : total_hours = 45 ∧ days = 5) : total_hours / days = 9 :=
by
  sorry

end sandy_hours_per_day_l33_33496


namespace curve_C1_to_ordinary_curve_C2_to_rectangular_minimum_distance_C1_C2_l33_33634

noncomputable def curve_C1_parametric (θ : ℝ) : ℝ × ℝ :=
(x, y) where
  x = 2 * sqrt 2 * cos θ
  y = 2 * sin θ

noncomputable def curve_C2_polar (ρ θ : ℝ) : Prop :=
  ρ * cos θ - sqrt 2 * ρ * sin θ - 4 = 0

theorem curve_C1_to_ordinary :
  ∀ (x y : ℝ), (∃ θ : ℝ, curve_C1_parametric θ = (x, y)) ↔ (x^2 / 8 + y^2 / 4 = 1) :=
sorry

theorem curve_C2_to_rectangular :
  ∀ (x y : ℝ), (∃ ρ θ : ℝ, ρ = sqrt(x^2 + y^2) ∧ tan θ = y / x ∧ curve_C2_polar ρ θ) ↔ (x - sqrt 2 * y = 4) :=
sorry

theorem minimum_distance_C1_C2 :
  ∀ P : ℝ × ℝ, (∃ θ : ℝ, P = curve_C1_parametric θ) →
  ∀ Q : ℝ × ℝ, (Q.1 - sqrt 2 * Q.2 = 4) →
  ∃ θ : ℝ, dist P Q = 0 :=
sorry

end curve_C1_to_ordinary_curve_C2_to_rectangular_minimum_distance_C1_C2_l33_33634


namespace tank_fill_percentage_l33_33451

def tank_capacity : ℕ := 200
def empty_tank_weight : ℕ := 80
def water_weight_per_gallon : ℕ := 8
def full_tank_weight : ℤ := 1360

theorem tank_fill_percentage : 
  ∃ (percentage : ℚ), percentage = 80 ∧ 
    let water_weight := full_tank_weight - empty_tank_weight in
    let water_volume := water_weight / water_weight_per_gallon in
    percentage = (water_volume : ℚ) / tank_capacity * 100 := 
by
  sorry

end tank_fill_percentage_l33_33451


namespace min_positive_period_max_value_l33_33114
open Real

noncomputable def f (x : ℝ) : ℝ := sin (x / 3) + cos (x / 3)

theorem min_positive_period_max_value :
  (∃ T > 0, ∀ x, f(x + T) = f(x) ∧ T = 6 * π) ∧
  (∀ x, f(x) ≤ sqrt 2) ∧ 
  (∃ x, f(x) = sqrt 2) :=
by
  sorry

end min_positive_period_max_value_l33_33114


namespace sqrt_expression_range_l33_33413

theorem sqrt_expression_range (x : ℝ) : 
  (∃ a b : ℝ, a = real.sqrt (2 * x - 7) ∧ b = real.sqrt (5 - x)) ↔ (3.5 ≤ x ∧ x ≤ 5) := sorry

end sqrt_expression_range_l33_33413


namespace min_period_and_max_value_l33_33127

def f (x : ℝ) := Real.sin (x / 3) + Real.cos (x / 3)

theorem min_period_and_max_value :
  (∀ T > 0, T ≠ 6 * Real.pi → ¬∀ x : ℝ, f(x + T) = f(x)) ∧
  ∃ x : ℝ, f(x) = sqrt 2 :=
by
  sorry

end min_period_and_max_value_l33_33127


namespace circle_equation_tangent_line_l33_33816

theorem circle_equation
  (t : ℝ)
  (h1 : (3 * t - 0) ^ 2 < 0) -- center lies on the line x - 3y = 0
  (h2 : (2 * Real.sqrt 7)^2 = (3 * t)^2 - (Real.sqrt 2 * t)^2) -- chord intercepted by y = x has length 2sqrt(7)
  : ((3 * t) ^ 2 = 9 * t ^ 2 ∧ t = 1 ∨ t = -1) →
    ((x - 3) ^ 2 + (y - 1) ^ 2 = 9 ∨ (x + 3) ^ 2 + (y + 1) ^ 2 = 9) :=
  sorry

theorem tangent_line
  (center : ℝ × ℝ := (3, 1))
  (radius : ℝ := 3)
  (h3 : center.1 > 0 ∧ center.2 > 0) -- center in the first quadrant
  (h4 : ∀ k : ℝ, k * (center.1 - 6) + center.2 - 5 = 0 → by (7 * 6 %/% 24 * 6) + center.2 - 5 = 0 ∨ 6 = 6)
  : ((7 * 6 - 24 * 5 + 78) ^ 2 = 0 ∨ 6 = 6) :=
  sorry

end circle_equation_tangent_line_l33_33816


namespace sum_of_distances_l33_33793

theorem sum_of_distances (A B D : ℝ × ℝ)
  (hA : A = (15, 0))
  (hB : B = (0, 5))
  (hD : D = (6, 8)) :
  real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2) + real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) = 18 :=
by 
  -- Proof is omitted, providing only the statement as required
  sorry

end sum_of_distances_l33_33793


namespace pentagon_stack_l33_33724

/-- Given a stack of identical regular pentagons with vertices numbered from 1 to 5, rotated and flipped
such that the sums of numbers at each vertex are the same, the number of pentagons in the stacks can be
any natural number except 1 and 3. -/
theorem pentagon_stack (n : ℕ) (h0 : identical_pentagons_with_vertices_1_to_5)
  (h1 : pentagons_can_be_rotated_and_flipped)
  (h2 : stacked_vertex_to_vertex)
  (h3 : sums_at_each_vertex_are_equal) :
  ∃ k : ℕ, k = n ∧ n ≠ 1 ∧ n ≠ 3 :=
sorry

end pentagon_stack_l33_33724


namespace sum_of_ages_is_l33_33949

-- Define the ages of the triplets and twins
def age_triplet (x : ℕ) := x
def age_twin (x : ℕ) := x - 3

-- Define the total age sum
def total_age_sum (x : ℕ) := 3 * age_triplet x + 2 * age_twin x

-- State the theorem
theorem sum_of_ages_is (x : ℕ) (h : total_age_sum x = 89) : ∃ x : ℕ, total_age_sum x = 89 := 
sorry

end sum_of_ages_is_l33_33949


namespace students_taking_one_language_l33_33626

-- Definitions based on the conditions
def french_class_students : ℕ := 21
def spanish_class_students : ℕ := 21
def both_languages_students : ℕ := 6
def total_students : ℕ := french_class_students + spanish_class_students - both_languages_students

-- The theorem we want to prove
theorem students_taking_one_language :
    total_students = 36 :=
by
  -- Add the proof here
  sorry

end students_taking_one_language_l33_33626


namespace complex_value_l33_33840

variable (z : ℂ)
def conjugate (z : ℂ) : ℂ := Complex.conj z

theorem complex_value {z : ℂ} (h1 : z + conjugate z = 2) (h2 : (z - conjugate z) * Complex.I = 2) : 
  z = 1 - Complex.I := by
repeat
  sorry

end complex_value_l33_33840


namespace range_rational_function_l33_33609

noncomputable def rational_function (x : ℝ) : ℝ := (x^2 + 5*x + 6) / (x + 2)

theorem range_rational_function :
  (Set.range rational_function) = Set.Ioo (⊥ : ℝ) 1 ∪ Set.Ioo 1 ⊤ :=
by
  sorry

end range_rational_function_l33_33609


namespace gecko_cricket_eating_l33_33234

theorem gecko_cricket_eating :
  ∀ (total_crickets : ℕ) (first_day_percent : ℚ) (second_day_less : ℕ),
    total_crickets = 70 →
    first_day_percent = 0.3 →
    second_day_less = 6 →
    let first_day_crickets := total_crickets * first_day_percent
    let second_day_crickets := first_day_crickets - second_day_less
    total_crickets - first_day_crickets - second_day_crickets = 34 :=
by
  intros total_crickets first_day_percent second_day_less h_total h_percent h_less
  let first_day_crickets := total_crickets * first_day_percent
  let second_day_crickets := first_day_crickets - second_day_less
  have : total_crickets - first_day_crickets - second_day_crickets = 34 := sorry
  exact this

end gecko_cricket_eating_l33_33234


namespace evaluate_expression_l33_33619

theorem evaluate_expression :
  71 * Real.sqrt (3 + 2 * Real.sqrt 2) - Real.sqrt (3 - 2 * Real.sqrt 2) = 72 + 70 * Real.sqrt 2 :=
by
  sorry

end evaluate_expression_l33_33619


namespace perimeter_is_12_l33_33433

-- Define the isosceles triangle properties
def is_isosceles_triangle {a b c : ℝ} (h : a = b ∨ b = c ∨ c = a) : Prop :=
a + b > c ∧ b + c > a ∧ c + a > b

-- Hypotheses and conditions
variables (a b : ℝ)
hypothesis h1 : |a - 2| + (b - 5) ^ 2 = 0

-- The statement that needs to be proven
theorem perimeter_is_12 (h_iso : is_isosceles_triangle (a := 2) (b := 5) (c := 5)) : 
  2 * 5 + 2 = 12 :=
sorry

end perimeter_is_12_l33_33433


namespace unique_rectangle_l33_33904

theorem unique_rectangle (a b : ℝ) (h : a < b) :
  ∃! (x y : ℝ), (x < y) ∧ (2 * (x + y) = a + b) ∧ (x * y = (a * b) / 4) := 
sorry

end unique_rectangle_l33_33904


namespace quadratic_root_sum_eight_l33_33685

theorem quadratic_root_sum_eight (p r : ℝ) (hp : p > 0) (hr : r > 0) 
  (h : ∀ (x₁ x₂ : ℝ), (x₁ + x₂ = p) -> (x₁ * x₂ = r) -> (x₁ + x₂ = 8)) : r = 8 :=
sorry

end quadratic_root_sum_eight_l33_33685


namespace function_periodic_8_l33_33855

noncomputable def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f(x + p) = f(x)

theorem function_periodic_8 (f : ℝ → ℝ)
  (h : ∀ x, f(x + 1) + f(x - 1) = Real.sqrt 2 * f(x)) :
  is_periodic f 8 := 
by
  sorry

end function_periodic_8_l33_33855


namespace angle_equiv_470_110_l33_33257

theorem angle_equiv_470_110 : ∃ (k : ℤ), 470 = k * 360 + 110 :=
by
  use 1
  exact rfl

end angle_equiv_470_110_l33_33257


namespace gcd_8fact_11fact_9square_l33_33198

theorem gcd_8fact_11fact_9square : Nat.gcd (Nat.factorial 8) ((Nat.factorial 11) * 9^2) = 40320 := 
sorry

end gcd_8fact_11fact_9square_l33_33198


namespace sqrt_Seq_arithmetic_gen_formula_an_greatest_integer_sum_l33_33351

noncomputable def sequence_an (n : ℕ) : ℕ :=
if n = 1 then 1 else
let S : ℕ → ℕ := λ m, ∑ i in finset.range (m+1), sequence_an i
in (nat.sqrt (S n) + nat.sqrt (S (n-1)))

theorem sqrt_Seq_arithmetic (n : ℕ) (hn : 2 ≤ n) :
  nat.sqrt (∑ i in finset.range n, sequence_an i) - nat.sqrt (∑ i in finset.range (n-1), sequence_an (i)) = 1 := sorry

theorem gen_formula_an :
  (∀ n, 1 ≤ n → sequence_an n = 2 * n - 1) := sorry

theorem greatest_integer_sum {n : ℕ} (hn : 1 ≤ n) :
  ⌊∑ i in finset.range (n+1), 4 / (sequence_an (i+1) + 1)^2⌋ = 1 := sorry

end sqrt_Seq_arithmetic_gen_formula_an_greatest_integer_sum_l33_33351


namespace numPeopleToLeftOfKolya_l33_33525

-- Definitions based on the conditions.
def peopleToRightOfKolya := 12
def peopleToLeftOfSasha := 20
def peopleToRightOfSasha := 8

-- Theorem statement with the given conditions and conclusion.
theorem numPeopleToLeftOfKolya 
  (h1 : peopleToRightOfKolya = 12)
  (h2 : peopleToLeftOfSasha = 20)
  (h3 : peopleToRightOfSasha = 8) :
  ∃ n, n = 16 :=
by
  -- Proving the theorem will be done here.
  sorry

end numPeopleToLeftOfKolya_l33_33525


namespace tank_weight_after_rainstorm_l33_33450

def capacity : ℕ := 200
def empty_tank_weight : ℕ := 80
def fill_percentage : ℝ := 0.80
def water_weight_per_gallon : ℕ := 8

noncomputable def tank_weight_now : ℕ :=
  let water_volume := (capacity : ℕ) * (fill_percentage : ℝ).to_nat
  let water_weight := water_volume * water_weight_per_gallon
  empty_tank_weight + water_weight

theorem tank_weight_after_rainstorm :
  tank_weight_now = 1360 :=
sorry

end tank_weight_after_rainstorm_l33_33450


namespace range_of_a_l33_33360

-- The Lean theorem statement for the generated math problem
theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, 0 < x ∧ x < 1 → 2 ^ (1 / x) > x ^ a) : a > -Real.exp(1) * Real.log 2 :=
sorry

end range_of_a_l33_33360


namespace projection_onto_plane_l33_33321

def projection (v n : Matrix (Fin 3) (Fin 1) ℝ) : Matrix (Fin 3) (Fin 1) ℝ :=
  v - ((v.dot_product n) / (n.dot_product n)) • n

theorem projection_onto_plane :
  let v := ![4, -1, 2]
  let n := ![1, 2, -3]
  let p := ![30/7, -3/7, 8/7]
  projection v n = p :=
  by
    sorry

end projection_onto_plane_l33_33321


namespace prime_remainder_l33_33709

theorem prime_remainder (p : ℕ) (k : ℕ) (h1 : Prime p) (h2 : p > 3) :
  (∃ k, p = 6 * k + 1 ∧ (p^3 + 17) % 24 = 18) ∨
  (∃ k, p = 6 * k - 1 ∧ (p^3 + 17) % 24 = 16) :=
by
  sorry

end prime_remainder_l33_33709


namespace cousin_daily_payment_l33_33862

theorem cousin_daily_payment
(a b T : ℤ)
(friend_payment_per_day brother_payment_per_day total_collection : a = 5 ∧ b = 8 ∧ T = 119 ) :
  T = 7 * (a + b) + 7 * 4 := by
  -- Definitions from conditions
  let f := 5
  let b := 8
  let c := 4
  have equation : 119 = 7 * (f + b) + 7 * c
  sorry

end cousin_daily_payment_l33_33862


namespace min_positive_period_f_max_value_f_l33_33074

noncomputable def f (x : ℝ) : ℝ := sin (x / 3) + cos (x / 3)

theorem min_positive_period_f : ∃ T > 0, (∀ x, f (x + T) = f x) ∧ ∀ T', (T' > 0 ∧ ∀ x, f (x + T') = f x) → T ≤ T' :=
  sorry

theorem max_value_f : ∃ M, (∀ x, f x ≤ M) ∧ (∀ ε > 0, ∃ x, M - ε < f x) ∧ M = sqrt 2 :=
  sorry

end min_positive_period_f_max_value_f_l33_33074


namespace tom_needs_more_boxes_l33_33578

theorem tom_needs_more_boxes
    (living_room_length : ℕ)
    (living_room_width : ℕ)
    (box_coverage : ℕ)
    (already_installed : ℕ) :
    living_room_length = 16 →
    living_room_width = 20 →
    box_coverage = 10 →
    already_installed = 250 →
    (living_room_length * living_room_width - already_installed) / box_coverage = 7 :=
by
    intros h1 h2 h3 h4
    rw [h1, h2, h3, h4]
    sorry

end tom_needs_more_boxes_l33_33578


namespace shopkeeper_profit_percentage_l33_33654

theorem shopkeeper_profit_percentage
  (cost_price : ℝ)
  (goods_lost_pct : ℝ)
  (loss_pct : ℝ)
  (remaining_goods : ℝ)
  (selling_price : ℝ)
  (profit_pct : ℝ)
  (h1 : cost_price = 100)
  (h2 : goods_lost_pct = 0.20)
  (h3 : loss_pct = 0.12)
  (h4 : remaining_goods = cost_price * (1 - goods_lost_pct))
  (h5 : selling_price = cost_price * (1 - loss_pct))
  (h6 : profit_pct = ((selling_price - remaining_goods) / remaining_goods) * 100) : 
  profit_pct = 10 := 
sorry

end shopkeeper_profit_percentage_l33_33654


namespace f_constant_on_S_l33_33837

-- Let S be the set of all real numbers greater than or equal to 1
def S := { x : ℝ | 1 ≤ x }

-- Define the function f : S → S
variable (f : S → S)

-- Define the condition on f: for all x, y in S with x^2 - y^2 in S, f(x^2 - y^2) = f(xy)
def condition (f : S → S) : Prop :=
  ∀ (x y : S), (x.val^2 - y.val^2 ∈ S) → (f ⟨x.val^2 - y.val^2, sorry⟩ = f ⟨x.val * y.val, sorry⟩)

-- The theorem that f must be a constant function
theorem f_constant_on_S (f : S → S) (h : condition f) : ∃ c ∈ S, ∀ x ∈ S, f x = c :=
sorry

end f_constant_on_S_l33_33837


namespace find_ellipse_equation_product_of_slopes_l33_33353

-- Definition of the ellipse conditions
def ellipse (x y : ℝ) (a b : ℝ) := x^2 / a^2 + y^2 / b^2 = 1

-- Given the conditions
variables (a b c : ℝ)
variables (h_a_gt_b : a > b) (h_b_gt_0 : b > 0)
variables (h_e : c = a / 2)
variables (h_dist : a - c = 1)

-- Prove that the equation of the ellipse is correct
theorem find_ellipse_equation : ellipse 4 3 1 :=
by {
  -- Solution goes here
  sorry
}

-- Given points on the ellipse and slopes
variables (h_A : A = (-2, 0)) 
variables (h_B : B = (2, 0))
variables (P : ℝ × ℝ)
variables (h_P_on_ellipse : ellipse P.fst P.snd 4 3)

-- Slopes definition
def slope (P1 P2 : ℝ × ℝ) := (P2.snd - P1.snd) / (P2.fst - P1.fst)

-- Prove the product of slopes
theorem product_of_slopes : slope P A * slope P B = -3 / 4 :=
by {
  -- Solution goes here
  sorry
}

end find_ellipse_equation_product_of_slopes_l33_33353


namespace triangle_uniquely_determined_by_SAS_l33_33983

theorem triangle_uniquely_determined_by_SAS :
  ∀ (A B C : Type) (AB BC : ℝ) (angle_B : ℝ),
  AB = 5 → BC = 3 → angle_B = 30 → 
  (∃! (triangle : Type), is_triangle_with_sides_and_angle triangle AB BC angle_B) :=
begin
  -- Proof goes here
  sorry
end

end triangle_uniquely_determined_by_SAS_l33_33983


namespace PA_mul_PB_eq_3_l33_33731

noncomputable def dist (P Q : Point) : Real :=
  sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2)

theorem PA_mul_PB_eq_3 :
  let P := (-2, 0)
  let O := (0, 0)
  let r := 1
  let PA := dist P A
  let PB := dist P B
  (dist P O)^2 = 4 → circle_intersect l (x^2 + y^2 = 1) A B → |PA| * |PB| = 3 :=
by
  intro P O r PA PB h1 h2
  sorry

end PA_mul_PB_eq_3_l33_33731


namespace x_intercept_is_correct_l33_33600

-- Define the original line equation
def original_line (x y : ℝ) : Prop := 4 * x + 5 * y = 10

-- Define the perpendicular line's y-intercept
def y_intercept (y : ℝ) : Prop := y = -3

-- Define the equation of the perpendicular line in slope-intercept form
def perpendicular_line (x y : ℝ) : Prop := y = (5 / 4) * x + -3

-- Prove that the x-intercept of the perpendicular line is 12/5
theorem x_intercept_is_correct : ∃ x : ℝ, x ≠ 0 ∧ (∃ y : ℝ, y = 0) ∧ (perpendicular_line x y) :=
sorry

end x_intercept_is_correct_l33_33600


namespace min_period_and_max_value_l33_33060

def f (x : ℝ) : ℝ := Real.sin (x / 3) + Real.cos (x / 3)

noncomputable def minimum_positive_period (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x : ℝ, f (x + T) = f x ∧ T > 0 ∧ ∀ T' : ℝ, (T' > 0 ∧ ∀ x : ℝ, f (x + T') = f x) → T' ≥ T

noncomputable def is_max_value (f : ℝ → ℝ) (M : ℝ) : Prop :=
  ∀ x : ℝ, f x ≤ M ∧ ∃ x : ℝ, f x = M

theorem min_period_and_max_value :
  minimum_positive_period f (6 * Real.pi) ∧ is_max_value f (Real.sqrt 2) := by
  sorry

end min_period_and_max_value_l33_33060


namespace opposite_of_10_l33_33562

-- Definitions
def numbers_on_cube : Finset ℕ := {6, 7, 8, 9, 10, 11}

-- Conditions
axiom roll_1 : (∑ x in numbers_on_cube \ {6, 7}, id x) = 36
axiom roll_2 : (∑ x in numbers_on_cube \ {8, 9}, id x) = 33

-- Proposition
theorem opposite_of_10 : (∀ x ∈ numbers_on_cube, (x, 10) ∈ ({11, 7}| {8, 10})) := 
by {
  sorry
}

end opposite_of_10_l33_33562


namespace true_proposition_p_and_q_l33_33389

-- Define the proposition p
def p : Prop := ∀ x : ℝ, x^2 + x + 1 > 0

-- Define the proposition q
def q : Prop := ∃ x : ℝ, x^3 = 1 - x^2

-- Statement to prove the conjunction p ∧ q
theorem true_proposition_p_and_q : p ∧ q := 
by 
    sorry

end true_proposition_p_and_q_l33_33389


namespace g_definition_g_max_value_l33_33733

noncomputable def f (a x : ℝ) : ℝ :=
  a * x^2 - 2 * x + 1

noncomputable def M (a : ℝ) : ℝ :=
  if (1 : ℝ) ≤ 1 / a then
    f a 3
  else
    f a 1

noncomputable def N (a : ℝ) : ℝ :=
  f a (1 / a)

noncomputable def g (a : ℝ) : ℝ :=
  if a ≤ 1 / 2 then
    a + 1 / a - 2
  else
    9 * a + 1 / a - 6

theorem g_definition (a : ℝ) (h : 1/3 ≤ a ∧ a ≤ 1) :
  g(a) = if a ≤ 1/2 then
           a + 1/a - 2 
         else 
           9 * a + 1 / a - 6 :=
by sorry

theorem g_max_value :
  ∀ (a : ℝ), (1/3 ≤ a ∧ a ≤ 1) → g(a) ≤ 4 :=
by sorry

end g_definition_g_max_value_l33_33733


namespace triangle_angle_size_triangle_side_lengths_l33_33725

theorem triangle_angle_size (A B C a b c : ℝ)
  (h1: b = Real.sqrt 7)
  (h2: (√3) * b * Real.sin A - a * Real.cos B - 2 * a = 0)
  (h3: 1 / 2 * a * c * Real.sin B = √3 / 2) :
  B = 2 * Real.pi / 3 := 
sorry

theorem triangle_side_lengths (A B C a b c : ℝ)
  (h1: b = Real.sqrt 7)
  (h2: B = 2 * Real.pi / 3)
  (h3: 1 / 2 * a * c * Real.sin B = √3 / 2)
  (h4: a^2 + c^2 - 2 * a * c * Real.cos B = 7) :
  (a = 1 ∧ c = 2) ∨ (a = 2 ∧ c = 1) := 
sorry

end triangle_angle_size_triangle_side_lengths_l33_33725


namespace min_period_max_value_f_l33_33141

def f (x : ℝ) : ℝ := sin (x / 3) + cos (x / 3)

theorem min_period_max_value_f :
  ∃ T M, (∀ x, f (x + T) = f x) ∧ 0 < T ∧
    0 < M ∧ (∀ x, f x ≤ M) ∧ (∃ x, f x = M) ∧ 
    T = 6 * π ∧ M = √2 := sorry

end min_period_max_value_f_l33_33141


namespace quadratic_b_value_l33_33245
open Real

theorem quadratic_b_value (b n : ℝ) 
  (h1: b < 0) 
  (h2: ∀ x, x^2 + b * x + (1 / 4) = (x + n)^2 + (1 / 16)) :
  b = - (sqrt 3 / 2) :=
by
  -- sorry is used to skip the proof
  sorry

end quadratic_b_value_l33_33245


namespace boxes_needed_to_complete_flooring_l33_33575

-- Definitions of given conditions
def length_of_living_room : ℕ := 16
def width_of_living_room : ℕ := 20
def sq_ft_per_box : ℕ := 10
def sq_ft_already_covered : ℕ := 250

-- Statement to prove
theorem boxes_needed_to_complete_flooring : 
  (length_of_living_room * width_of_living_room - sq_ft_already_covered) / sq_ft_per_box = 7 :=
by
  sorry

end boxes_needed_to_complete_flooring_l33_33575


namespace race_distance_l33_33431

theorem race_distance (D : ℝ)
  (A_speed : ℝ := D / 20)
  (B_speed : ℝ := D / 25)
  (A_beats_B_by : ℝ := 18)
  (h1 : A_speed * 25 = D + A_beats_B_by)
  : D = 72 := 
by
  sorry

end race_distance_l33_33431


namespace least_prime_factor_of_5_pow_5_minus_5_pow_4_l33_33979

theorem least_prime_factor_of_5_pow_5_minus_5_pow_4 : (∃ p : ℕ, p.prime ∧ p ≤ 5 - 1 ∧ p ∣ (5^5 - 5^4)) :=
  sorry

end least_prime_factor_of_5_pow_5_minus_5_pow_4_l33_33979


namespace find_k_l33_33221

/-- Definitions of the vectors involved --/
def vec (x y : ℝ) : ℝ × ℝ := (x, y)
def e1 : ℝ × ℝ := vec 1 0
def e2 : ℝ × ℝ := vec 0 1
def AB : ℝ × ℝ := vec 1 (-1)
def BC : ℝ × ℝ := vec 3 2
def CD (k : ℝ) : ℝ × ℝ := vec k 2
def AC : ℝ × ℝ := vec 4 1

/-- Collinearity condition for points A, C, and D. We express that AC is a scalar multiple of CD. --/
def collinear (AC CD : ℝ × ℝ) (k : ℝ) : Prop := ∃ λ, AC = (λ * k, λ * 2)

/-- The proof statement --/
theorem find_k : ∃ k, collinear AC (CD k) k ∧ k = 8 :=
by
  sorry

end find_k_l33_33221


namespace number_of_people_to_the_left_of_Kolya_l33_33534

-- Defining the conditions
variables (left_sasha right_sasha right_kolya total_students left_kolya : ℕ)

-- Condition definitions
def condition1 := right_kolya = 12
def condition2 := left_sasha = 20
def condition3 := right_sasha = 8

-- Calculate total number of students
def calc_total_students : ℕ := left_sasha + right_sasha + 1

-- Calculate number of students to the left of Kolya
def calc_left_kolya (total_students right_kolya : ℕ) : ℕ := total_students - right_kolya - 1

-- Problem statement to prove
theorem number_of_people_to_the_left_of_Kolya
    (H1 : condition1)
    (H2 : condition2)
    (H3 : condition3)
    (total_students : calc_total_students = 29) : 
    calc_left_kolya total_students right_kolya = 16 :=
by
  sorry

end number_of_people_to_the_left_of_Kolya_l33_33534


namespace area_of_square_l33_33928

-- Definition of distance between two points in the plane
def distance (x1 y1 x2 y2 : ℝ) : ℝ := 
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Given points
def P1 : ℝ × ℝ := (1, 5)
def P2 : ℝ × ℝ := (4, -2)

-- Definition of side length of the square
def side_length : ℝ := distance P1.1 P1.2 P2.1 P2.2

-- Definition of the area of the square
def area : ℝ := side_length^2

-- Statement to prove
theorem area_of_square (x1 y1 x2 y2 : ℝ) 
  (hP1 : (x1, y1) = P1) (hP2 : (x2, y2) = P2) : area = 58 := 
by
  -- Skipping the proof
  sorry

end area_of_square_l33_33928


namespace correct_equation_for_gift_exchanges_l33_33273

theorem correct_equation_for_gift_exchanges
  (x : ℕ)
  (H : (x * (x - 1)) = 56) :
  x * (x - 1) = 56 := 
by 
  exact H

end correct_equation_for_gift_exchanges_l33_33273


namespace sum_possible_numbers_l33_33487

theorem sum_possible_numbers {a b c d : ℕ} (h1: a = 2) (h2: b = 0) (h3: c = 1) (h4: d = 8) :
  let digits := [a, b, c, d].erase b in
  let one_digit_numbers := digits in
  let two_digit_numbers := (list.permutations digits).filter (λ l, l.head ≠ 0) |>.map (λ l, 10 * l.head! + l.tail.head!) in
  let three_digit_numbers := (list.permutations digits).filter (λ l, l.head ≠ 0) |>.map (λ l, 100 * l.head! + 10 * l.tail.head! + l.tail.tail.head!) in
  let four_digit_numbers := (list.permutations digits).filter (λ l, l.head ≠ 0) |>.map (λ l, 1000 * l.head! + 100 * l.tail.head! + 10 * l.tail.tail.head! + l.tail.tail.tail.head!) in
  (one_digit_numbers ++ two_digit_numbers ++ three_digit_numbers ++ four_digit_numbers).sum = 78331
:= by sorry

end sum_possible_numbers_l33_33487


namespace division_of_fraction_simplified_l33_33676

theorem division_of_fraction_simplified :
  12 / (2 / (5 - 3)) = 12 := 
by
  sorry

end division_of_fraction_simplified_l33_33676


namespace second_test_point_using_618_method_l33_33307

-- Define the interval endpoints
def a := 2
def b := 4

-- Define the coefficient used in the 0.618 method
def coeff := 0.618

-- Define the first test point
def x1 := a + coeff * (b - a)

-- Define the second test point
def x2 := a + (b - x1)

-- Proposition to prove
theorem second_test_point_using_618_method :
  x2 = 2.764 := by
  sorry

end second_test_point_using_618_method_l33_33307


namespace min_period_and_max_value_l33_33120

def f (x : ℝ) := Real.sin (x / 3) + Real.cos (x / 3)

theorem min_period_and_max_value :
  (∀ T > 0, T ≠ 6 * Real.pi → ¬∀ x : ℝ, f(x + T) = f(x)) ∧
  ∃ x : ℝ, f(x) = sqrt 2 :=
by
  sorry

end min_period_and_max_value_l33_33120


namespace least_prime_factor_of_5_pow_5_minus_5_pow_4_l33_33973

theorem least_prime_factor_of_5_pow_5_minus_5_pow_4 : 
  ∃ p : ℕ, nat.prime p ∧ p = 2 ∧ ∃ f : nat.factorization, 
  (5^5 - 5^4) = (f.prod) ∧ p ∈ f.support :=
begin
  sorry
end

end least_prime_factor_of_5_pow_5_minus_5_pow_4_l33_33973


namespace smallest_value_of_m_l33_33658

noncomputable def smallest_m : ℕ :=
  if h : ∃ m : ℕ, m % 5 = 0 ∧ (∃ x : ℕ, x = 2000 * m / 21 ∧ 2000 * m % 21 = 0) then 
    Classical.choose h 
  else 0

theorem smallest_value_of_m : smallest_m = 21 :=
by
  sorry

end smallest_value_of_m_l33_33658


namespace Norris_saved_S_in_September_l33_33867

-- Definition of the conditions
variables
  (S : ℕ) -- amount saved in September
  (O : ℕ := 25) -- amount saved in October
  (N : ℕ := 31) -- amount saved in November
  (G : ℕ := 75) -- amount spent on the game
  (L : ℕ := 10) -- amount left after spending

-- Theorem: Proving the amount saved in September
theorem Norris_saved_S_in_September : S + O + N - G = L → S = 29 :=
by
  intros h
  have h1 : S + 25 + 31 - 75 = 10 := h
  sorry

end Norris_saved_S_in_September_l33_33867


namespace number_of_ways_to_draw_balls_eq_42_l33_33180

-- Initial setup and assumptions
variables {Color : Type} [Fintype Color] [DecidableEq Color] (draws : ℕ → Color)
noncomputable def draw_count : ℕ := 5
noncomputable def all_colors_drawn (n : ℕ) : Prop := 
  ∀ c : Color, ∃ i ≤ n, draws i = c

-- Main theorem statement
theorem number_of_ways_to_draw_balls_eq_42 :
  (∀ n < draw_count, all_colors_drawn n → n = draw_count) → 
  (∃! d : Finset (Fin draw_count → Color), d.card = 42) :=
sorry

end number_of_ways_to_draw_balls_eq_42_l33_33180


namespace gcd_three_numbers_l33_33911

theorem gcd_three_numbers (a b c : ℕ) (h1 : a = 72) (h2 : b = 120) (h3 : c = 168) :
  Nat.gcd (Nat.gcd a b) c = 24 :=
by
  rw [h1, h2, h3]
  exact sorry

end gcd_three_numbers_l33_33911


namespace find_n_l33_33313

theorem find_n : ∃ n : ℕ, 2^7 * 3^3 * 5 * n = Nat.factorial 12 ∧ n = 27720 :=
by
  use 27720
  have h1 : 2^7 * 3^3 * 5 * 27720 = Nat.factorial 12 :=
  sorry -- This will be the place to prove the given equation eventually.
  exact ⟨h1, rfl⟩

end find_n_l33_33313


namespace fill_time_approx_l33_33213

def rateA : ℝ := 1 / 36
def rateB : ℝ := 1 / 46
def combinedRate : ℝ := rateA + rateB
def timeToFill : ℝ := 1 / combinedRate

theorem fill_time_approx :
  pipeA  := 36,
  pipeB  := 46,
  (timeToFill ≈ 20.2) := sorry

end fill_time_approx_l33_33213


namespace probability_at_least_one_8_l33_33960

theorem probability_at_least_one_8 (n : ℕ) (hn : n = 8) : 
  (1 - (7/8) * (7/8)) = 15 / 64 :=
by
  rw [← hn]
  sorry

end probability_at_least_one_8_l33_33960


namespace floor_sum_example_l33_33965

def floor (x : ℝ) : ℤ := Int.floor x

theorem floor_sum_example : floor 5.5 + floor (-4.5) = 0 := by
  sorry

end floor_sum_example_l33_33965


namespace count_solutions_l33_33456

theorem count_solutions :
  (finset.card
    ((finset.Icc (0 : ℕ) 99).product (finset.Icc (0 : ℕ) 99).product (finset.Icc (0 : ℕ) 99).product (finset.Icc (0 : ℕ) 99)).filter
      (λ (xyzw : ℕ × ℕ × ℕ × ℕ), 2023 = xyzw.1 * 10^3 + xyzw.2.1 * 10^2 + xyzw.2.2.1 * 10 + xyzw.2.2.2)) = 203 := 
sorry

end count_solutions_l33_33456


namespace median_length_correct_l33_33968

structure Triangle where
  A B C : ℝ
  pos_A : 0 < A
  pos_B : 0 < B
  pos_C : 0 < C

noncomputable def median_length (t : Triangle) (M : ℝ) : ℝ := by
  sorry

theorem median_length_correct :
  let DEF := Triangle.mk 13 14 15 (by norm_num) (by norm_num) (by norm_num)
  let DM := median_length DEF (15 / 2)
  DM = sqrt (505 / 4) :=
by
  sorry

end median_length_correct_l33_33968


namespace second_differences_of_cubes_l33_33612

-- Define the first difference for cubes of consecutive natural numbers
def first_difference (n : ℕ) : ℕ :=
  ((n + 1) ^ 3) - (n ^ 3)

-- Define the second difference for the first differences
def second_difference (n : ℕ) : ℕ :=
  first_difference (n + 1) - first_difference n

-- Proof statement: Prove that second differences are equal to 6n + 6
theorem second_differences_of_cubes (n : ℕ) : second_difference n = 6 * n + 6 :=
  sorry

end second_differences_of_cubes_l33_33612


namespace angles_of_subtriangles_l33_33839

namespace TriangleAngles

variables {A B C H H_A H_B H_C : Point}
variables {angle_A angle_B angle_C : ℝ}

-- Define the triangle and properties related to its orthocenter and altitudes
def Triangle (A B C : Point) : Prop :=
  -- All angles of the triangle are acute
  angle_A > 0 ∧ angle_A < 90 ∧
  angle_B > 0 ∧ angle_B < 90 ∧
  angle_C > 0 ∧ angle_C < 90

def IsOrthocenter (H : Point) (A B C : Point) : Prop :=
  -- H is the orthocenter of triangle ABC
  true

def IsFootOfAltitude (H_A H_B H_C : Point) (A B C : Point) : Prop :=
  -- H_A, H_B, H_C are feet of the altitudes from vertices A, B, C, respectively
  true

-- Theorem stating the angles in terms of angles of triangle ABC
theorem angles_of_subtriangles 
  (h₁ : Triangle A B C) (h₂ : IsOrthocenter H A B C) (h₃ : IsFootOfAltitude H_A H_B H_C A B C) :
  -- Angles of triangle AH_BH_C
  (∠A H_B H_C = angle_B) ∧ (∠A H_C H_B = angle_C) ∧
  -- Angles of triangle H_ABH_C
  (∠B H_C H_A = angle_C) ∧ (∠B H_A H_C = angle_A) ∧
  -- Angles of triangle H_AH_BC
  (∠C H_A H_B = angle_A) ∧ (∠C H_B H_A = angle_B) ∧
  -- Angles of triangle H_AH_BH_C
  (∠H_A H_B H_C = 180 - 2 * angle_B) ∧ 
  (∠H_B H_C H_A = 180 - 2 * angle_C) ∧ 
  (∠H_C H_A H_B = 180 - 2 * angle_A) := 
  sorry -- Proof to be provided

end TriangleAngles

end angles_of_subtriangles_l33_33839


namespace feeder_contains_more_than_half_sunflower_seeds_on_thursday_l33_33481

theorem feeder_contains_more_than_half_sunflower_seeds_on_thursday :
    ∃ n : ℕ, n = 4 ∧ 0.7^n < (1 / 4) :=
by
  sorry

end feeder_contains_more_than_half_sunflower_seeds_on_thursday_l33_33481


namespace perpendicular_line_x_intercept_l33_33597

noncomputable def slope (a b : ℚ) : ℚ := - a / b

noncomputable def line_equation (m y_intercept : ℚ) (x : ℚ) : ℚ :=
  m * x + y_intercept

theorem perpendicular_line_x_intercept :
  let m1 := slope 4 5,
      m2 := (5 / 4),
      y_int := -3
  in 
  ∀ x, line_equation m2 y_int x = 0 → x = 12 / 5 :=
by
  intro x hx
  sorry

end perpendicular_line_x_intercept_l33_33597


namespace minimum_period_and_max_value_of_f_l33_33044

noncomputable def f (x : ℝ) : ℝ := sin (x / 3) + cos (x / 3)

theorem minimum_period_and_max_value_of_f :
  (∃ T > 0, ∀ x: ℝ, f (x + T) = f x) ∧
  (∀ x: ℝ, f x ≤ sqrt 2 ∧ 
          (∃ y: ℝ, f y = sqrt 2)) :=
by 
  sorry

end minimum_period_and_max_value_of_f_l33_33044


namespace minimum_norm_of_v_l33_33468

open Real 

-- Define the vector v and condition
noncomputable def v : ℝ × ℝ := sorry

-- Define the condition
axiom v_condition : ‖(v.1 + 4, v.2 + 2)‖ = 10

-- The statement that we need to prove
theorem minimum_norm_of_v : ‖v‖ = 10 - 2 * sqrt 5 :=
by
  sorry

end minimum_norm_of_v_l33_33468


namespace work_together_l33_33209

theorem work_together (A_rate B_rate : ℝ) (hA : A_rate = 1 / 9) (hB : B_rate = 1 / 18) : (1 / (A_rate + B_rate) = 6) :=
by
  -- we only need to write the statement, proof is not required.
  sorry

end work_together_l33_33209


namespace ceil_floor_diff_l33_33776

theorem ceil_floor_diff (x : ℝ) (h : ⌈x⌉ + ⌊x⌋ = 2 * x) : ⌈x⌉ - ⌊x⌋ = 1 := 
by 
  sorry

end ceil_floor_diff_l33_33776


namespace roots_product_eq_three_l33_33851

theorem roots_product_eq_three
  (p q r : ℝ)
  (h : (3:ℝ) * p ^ 3 - 8 * p ^ 2 + p - 9 = 0 ∧
       (3:ℝ) * q ^ 3 - 8 * q ^ 2 + q - 9 = 0 ∧
       (3:ℝ) * r ^ 3 - 8 * r ^ 2 + r - 9 = 0) :
  p * q * r = 3 :=
sorry

end roots_product_eq_three_l33_33851


namespace valid_participant_scores_l33_33427

theorem valid_participant_scores (n k : ℕ) (hn : n ≥ 2) (hk : k ≥ 1) 
    (hscore : ∀ p : ℕ, p < n → ∃! (scores : Fin k → Fin n), (Fin k → Fin n → ℕ)
        (λ day idx score, 1 ≤ score ∧ score ≤ n ∧ 
         ∀ d, ∑ i, scores d i = 26 ∧ 
         ∀ (d1 d2 : Fin k), d1 ≠ d2 → ∀ i j, scores d1 i ≠ scores d2 j)) :
    (n, k) = (25, 2) ∨ (n, k) = (12, 4) ∨ (n, k) = (3, 13) :=
sorry

end valid_participant_scores_l33_33427


namespace count_integers_with_digits_l33_33663

def has_required_digits (n : ℕ) : Prop :=
  let digits := [n / 10000 % 10, n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10]
  let required_digits := [2, 0, 1, 7]
  let matches := List.length (List.filter (λ (d : ℕ), d ∈ required_digits) digits)
  matches ≥ 2

theorem count_integers_with_digits : 
  (List.range 10000).filter (has_required_digits) |>.length = 3862 := by
  sorry

end count_integers_with_digits_l33_33663


namespace expression_X_l33_33411

variable {a b X : ℝ}

theorem expression_X (h1 : a / b = 4 / 3) (h2 : (3 * a + 2 * b) / X = 3) : X = 2 * b := 
sorry

end expression_X_l33_33411


namespace min_positive_period_and_max_value_l33_33106

noncomputable def f (x : ℝ) : ℝ := Real.sin (x / 3) + Real.cos (x / 3)

theorem min_positive_period_and_max_value :
  (∃ T > 0, ∀ x, f (x + T) = f x) ∧ ∃ x, f x = sqrt 2 :=
by
  sorry

end min_positive_period_and_max_value_l33_33106


namespace hyperbolic_canonical_form_l33_33216

theorem hyperbolic_canonical_form :
  ∀ (u : ℝ → ℝ → ℝ) (x y : ℝ),
    (∂^2 u / ∂ x^2) - 4 * (∂^2 u / (∂ x ∂ y)) - 21 * (∂^2 u / ∂ y^2) + 
    2 * (∂ u / ∂ x) - 3 * (∂ u / ∂ y) + 5 * u = x^2 →
    (∂^2 u / (∂ ((y + 7 * x) / 10) ∂ ((y - 3 * x) / 10))) = 
    (-10 / 112) * (∂ u / ∂ ((y - 3 * x) / 10)) + 
    (11 / 112) * (∂ u / ∂ ((y + 7 * x) / 10)) + 
    (5 / 112) * u - (1 / 112) * ((y + 7 * x) / 10 - (y - 3 * x) / 10)^2 :=
by
  intros u x y h
  sorry

end hyperbolic_canonical_form_l33_33216


namespace tom_needs_more_boxes_l33_33576

theorem tom_needs_more_boxes
    (living_room_length : ℕ)
    (living_room_width : ℕ)
    (box_coverage : ℕ)
    (already_installed : ℕ) :
    living_room_length = 16 →
    living_room_width = 20 →
    box_coverage = 10 →
    already_installed = 250 →
    (living_room_length * living_room_width - already_installed) / box_coverage = 7 :=
by
    intros h1 h2 h3 h4
    rw [h1, h2, h3, h4]
    sorry

end tom_needs_more_boxes_l33_33576


namespace people_left_of_Kolya_l33_33543

/-- Given:
    1. There are 12 people to the right of Kolya.
    2. There are 20 people to the left of Sasha.
    3. There are 8 people to the right of Sasha.
    4. The total number of people in the class (including Sasha) is 29.

    Prove:
    The number of people to the left of Kolya is 16.
-/
theorem people_left_of_Kolya : 
  ∀ (total_people right_of_Kolya left_of_Sasha right_of_Sasha : ℕ),
  right_of_Kolya = 12 →
  left_of_Sasha = 20 →
  right_of_Sasha = 8 →
  total_people = 29 →
  left_of_Kolya := total_people - right_of_Kolya - 1
  left_of_Kolya = 16 :=
by
  intros
  sorry

end people_left_of_Kolya_l33_33543


namespace find_perpendicular_line_through_circle_center_l33_33032

theorem find_perpendicular_line_through_circle_center :
  (∃ C : ℝ × ℝ, C = (-1, 1) ∧ 
                ( ∃ l : ℝ → ℝ, 
                    (∀ x, l(x) = x + 2) ∧ 
                    (∀ P : ℝ × ℝ, P = (-1, 1) → (l P.1 = P.2)) ∧ 
                    ( ∀ m : ℝ, m ≠ 1 → ¬ ∃ k : ℝ → ℝ, k = l) )
  ) :=
sorry

end find_perpendicular_line_through_circle_center_l33_33032


namespace pics_in_each_album_l33_33584

theorem pics_in_each_album :
  ∀ (phone_pics camera_pics albums : ℕ), 
    phone_pics = 23 → 
    camera_pics = 7 → 
    albums = 5 → 
    (phone_pics + camera_pics) / albums = 6 :=
by 
  intros phone_pics camera_pics albums h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end pics_in_each_album_l33_33584


namespace player_wins_l33_33645

theorem player_wins (N : ℕ) : (N = 1 → ∃ first_player_wins : true) ∧ (N > 1 → ∃ second_player_wins : true) :=
by {
    sorry
}

end player_wins_l33_33645


namespace PA_PB_product_l33_33434

-- Definitions of curves and line
def C1_parametric (φ : ℝ) : ℝ × ℝ := (2 * cos φ, 2 * sin φ)
def C1_cartesian (x y : ℝ) : Prop := x^2 + y^2 = 4

def C2_parametric (φ : ℝ) : ℝ × ℝ := (cos φ, 4 * sin φ)
def C2_cartesian (x y : ℝ) : Prop := x^2 + y^2 / 16 = 1

def l_parametric (t : ℝ) : ℝ × ℝ := (t, 1 + sqrt 3 * t)

-- Main theorem statement
theorem PA_PB_product : 
  (∀ φ : ℝ, C1_parametric φ = (2 * cos φ, 2 * sin φ)) →
  (∀ φ : ℝ, C1_cartesian (2 * (cos φ)) (2 * (sin φ))) →
  (∀ φ : ℝ, C2_parametric φ = (cos φ, 4 * sin φ)) →
  (∀ φ : ℝ, C2_cartesian (cos φ) (4 * (sin φ))) →
  (∀ t : ℝ, l_parametric t = (t, 1 + sqrt 3 * t)) →
  ∀ t1 t2 : ℝ, (19 * t1 ^ 2 + 4 * sqrt 3 * t1 - 60 = 0) ∧ (19 * t2 ^ 2 + 4 * sqrt 3 * t2 - 60 = 0) →
  abs (t1 * t2) = 60 / 19 := 
sorry

end PA_PB_product_l33_33434


namespace domain_of_function_l33_33508

theorem domain_of_function:
  {x : ℝ} → (x > 1) → (∃ y, y = (1 / (real.sqrt (x - 1))) + (x - 3)^0) := sorry

end domain_of_function_l33_33508


namespace find_f_l33_33380

-- Conditions stated as definitions
def f (x : ℝ) : ℝ := 2 * x * f' 1 + Real.log x
def f' (x : ℝ) : ℝ := 2 * f' 1 + 1 / x

-- The theorem statement
theorem find_f'_one (h : f 1 = 2 * 1 * f' 1 + Real.log 1) : f' 1 = -1 :=
by 
  sorry

end find_f_l33_33380


namespace crayons_in_drawer_before_l33_33564

theorem crayons_in_drawer_before (m c : ℕ) (h1 : m = 3) (h2 : c = 10) : c - m = 7 := 
  sorry

end crayons_in_drawer_before_l33_33564


namespace inequality_problem_l33_33740

noncomputable theory

variables {x y z : ℝ}

theorem inequality_problem (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x + y + z ≥ x * y * z) :
  x / (y * z) + y / (z * x) + z / (x * y) ≥ real.sqrt 3 :=
by {
  sorry
}

end inequality_problem_l33_33740


namespace min_positive_period_and_max_value_l33_33078

def f (x : ℝ) := Real.sin (x / 3) + Real.cos (x / 3)

theorem min_positive_period_and_max_value : 
  (∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T' = T)) 
  ∧ 
  (∀ x, f x ≤ sqrt 2 ∧ (∃ x, f x = sqrt 2)) :=
sorry

end min_positive_period_and_max_value_l33_33078


namespace football_team_starting_lineup_count_l33_33875

theorem football_team_starting_lineup_count :
  let total_members := 12
  let offensive_lineman_choices := 4
  let quarterback_choices := 2
  let remaining_after_ol := total_members - 1 -- after choosing one offensive lineman
  let remaining_after_qb := remaining_after_ol - 1 -- after choosing one quarterback
  let running_back_choices := remaining_after_ol
  let wide_receiver_choices := remaining_after_qb - 1
  let tight_end_choices := remaining_after_qb - 2
  offensive_lineman_choices * quarterback_choices * running_back_choices * wide_receiver_choices * tight_end_choices = 5760 := 
by
  sorry

end football_team_starting_lineup_count_l33_33875


namespace part1_part2_l33_33755

noncomputable def f (x a : ℝ) : ℝ := Real.log x + (a / 2) * x^2 - (a + 1) * x

theorem part1 (a : ℝ) (h : f 1 a = -2) :
  ∃ I1 I2, 
    I1 = set.Ioo 0 (1 / 2) ∧ I2 = set.Ioi 1 ∧ 
    (∀ x ∈ I1 ∪ I2, deriv (λ x, f x a) x > 0) ∧ 
    (∀ x ∈ set.Ioo (1 / 2) 1, deriv (λ x, f x a) x < 0) :=
sorry

theorem part2 (a : ℝ) (h : ∀ x > 0, (f x a) / x < deriv (λ x, f x a) x / 2) :
  a > 2 * Real.exp (-1 / 2) - 1 :=
sorry

#check part1
#check part2

end part1_part2_l33_33755


namespace highest_degree_has_asymptote_l33_33290

noncomputable def highest_degree_of_px (denom : ℕ → ℕ) (n : ℕ) : ℕ :=
  let deg := denom n
  deg

theorem highest_degree_has_asymptote (p : ℕ → ℕ) (denom : ℕ → ℕ) (n : ℕ)
  (h_denom : denom n = 6) :
  highest_degree_of_px denom n = 6 := by
  sorry

end highest_degree_has_asymptote_l33_33290


namespace missing_digit_is_four_l33_33019

open BigInt

noncomputable def set : List ℤ := [9, 9999, 99999999, 999999999999, 9999999999999999, 99999999999999999999]

noncomputable def N : BigInt :=
  (9 + 9999 + 99999999 + 999999999999 + 9999999999999999 + 99999999999999999999) / 6

theorem missing_digit_is_four :
  ∀ d : ℕ, d ∈ to_digits 10 N → d ≠ 4 := 
by
  sorry

end missing_digit_is_four_l33_33019


namespace range_of_x_inequality_l33_33719

theorem range_of_x_inequality (a : ℝ) (x : ℝ) (h : a ∈ set.Icc (-1:ℝ) 1) : 
  x^2 + (a-4)*x + 4 - 2*a > 0 ↔ (x < 1 ∨ x > 3) :=
by
  sorry

end range_of_x_inequality_l33_33719


namespace no_linear_term_l33_33786

theorem no_linear_term (m : ℤ) : (∀ (x : ℤ), (x^2 - x + m) * (x - 8) = x^3 - 9*x^2 - 8*m → (8 + m) = 0) → m = -8 :=
by
  sorry

end no_linear_term_l33_33786


namespace ice_cream_parlor_distance_l33_33002

theorem ice_cream_parlor_distance :
  let D be real,
  let upstream_speed := 3,
  let downstream_speed := 9,
  let wind_effect_upstream := 2,
  let wind_effect_downstream := 4,
  let effective_upstream_speed := upstream_speed - wind_effect_upstream,
  let effective_downstream_speed := downstream_speed + wind_effect_downstream,
  let break_time_hours := 0.25,
  let total_trip_time := 8,
  let paddling_time := total_trip_time - break_time_hours,
  let T_up := D / effective_upstream_speed,
  let T_down := D / effective_downstream_speed in
  T_up + T_down = paddling_time → D = 7.20 :=
by
  intro D upstream_speed downstream_speed wind_effect_upstream wind_effect_downstream effective_upstream_speed effective_downstream_speed break_time_hours total_trip_time paddling_time T_up T_down h
  sorry

end ice_cream_parlor_distance_l33_33002


namespace complex_triangle_solution_l33_33462

open Complex

def is_right_isosceles (z1 z2 : ℂ) : Prop :=
  ∃ ω : ℂ, ω = exp (π * I / 4) ∧ z2 = ω * z1

theorem complex_triangle_solution
  (a b z1 z2 : ℂ)
  (h1 : z2 = exp (π * I / 4) * z1)
  (h2 : z1 + z2 = -a)
  (h3 : z1 * z2 = b) :
  a^2 / b = (2 * sqrt 2 + 2 * I * sqrt 2) :=
by
  sorry

end complex_triangle_solution_l33_33462


namespace sphere_parallel_plane_distance_l33_33635

theorem sphere_parallel_plane_distance
    (R : ℝ) (r1_area r2_area : ℝ) (r1 r2 : ℝ)
    (hR : R = 10)
    (h1 : r1_area = 36 * π)
    (h2 : r2_area = 64 * π)
    (hr1 : r1 = √(r1_area / π))
    (hr2 : r2 = √(r2_area / π)) :
    (∃ d : ℝ, d = √(R^2 - r1^2) + √(R^2 - r2^2) ∨ 
              d = |√(R^2 - r1^2) - √(R^2 - r2^2)| ∧ 
              (d = 2 ∨ d = 14)) :=
by
  sorry

end sphere_parallel_plane_distance_l33_33635


namespace interest_rate_l33_33240

theorem interest_rate (SI P T : ℕ) (h1 : SI = 2000) (h2 : P = 5000) (h3 : T = 10) :
  (SI = (P * R * T) / 100) -> R = 4 :=
by
  sorry

end interest_rate_l33_33240


namespace distance_between_planes_l33_33703

-- Definitions of the planes
def plane1 (x y z : ℝ) : Prop := x + 2 * y - 2 * z + 3 = 0
def plane2 (x y z : ℝ) : Prop := 2 * x + 5 * y - 4 * z + 7 = 0

-- Definition of a point on the first plane
def point_on_plane1 : Prop := plane1 (-3) 0 0

-- The distance of a point to a plane formula
def distance_point_to_plane (A B C D x₀ y₀ z₀ : ℝ) : ℝ :=
  abs (A * x₀ + B * y₀ + C * z₀ + D) / real.sqrt (A ^ 2 + B ^ 2 + C ^ 2)

-- The distance from the point (-3, 0, 0) to the second plane
def distance_from_point_to_plane2 : ℝ :=
  distance_point_to_plane 2 5 (-4) 7 (-3) 0 0

-- The expected distance between the planes
def expected_distance : ℝ := real.sqrt 5 / 15

-- The proof problem statement
theorem distance_between_planes :
  point_on_plane1 →
  distance_from_point_to_plane2 = expected_distance :=
sorry

end distance_between_planes_l33_33703


namespace min_positive_period_max_value_l33_33118
open Real

noncomputable def f (x : ℝ) : ℝ := sin (x / 3) + cos (x / 3)

theorem min_positive_period_max_value :
  (∃ T > 0, ∀ x, f(x + T) = f(x) ∧ T = 6 * π) ∧
  (∀ x, f(x) ≤ sqrt 2) ∧ 
  (∃ x, f(x) = sqrt 2) :=
by
  sorry

end min_positive_period_max_value_l33_33118


namespace min_value_fraction_l33_33408

theorem min_value_fraction (x : ℝ) (hx : x < 2) : ∃ y : ℝ, y = (5 - 4 * x + x^2) / (2 - x) ∧ y = 2 :=
by sorry

end min_value_fraction_l33_33408


namespace limit_factorial_root_div_n_l33_33879

theorem limit_factorial_root_div_n : 
  tendsto (λ n : ℕ, (real.sqrt (n.fac : ℝ) ^ (1 / n : ℝ)) / n) at_top (𝓝 (1 / real.exp 1)) := by
  sorry

end limit_factorial_root_div_n_l33_33879


namespace teamA_win_first_two_sets_while_earning_3_points_l33_33441

-- Define the conditions
def best_of_five (scores : list nat) : Prop := scores.length = 5
def teamA_scores (s : nat) : Prop := s = 3
def teamB_scores (s : nat) : Prop := s ≤ 3
def prob_teamA_wins_set : ℚ := 3/5
def prob_teamA_earns_3_points : ℚ := (27/125) + (162/625)
def prob_teamA_wins_first_two_sets := prob_teamA_wins_set^2
def prob_AB : ℚ := (27/125) + ((3/5)^2 * (2/5) * (3/5))

-- Define the theorem
theorem teamA_win_first_two_sets_while_earning_3_points :
  (prob_AB / prob_teamA_earns_3_points) = (7 / 11) :=
sorry

end teamA_win_first_two_sets_while_earning_3_points_l33_33441


namespace min_period_max_value_f_l33_33131

def f (x : ℝ) : ℝ := sin (x / 3) + cos (x / 3)

theorem min_period_max_value_f :
  ∃ T M, (∀ x, f (x + T) = f x) ∧ 0 < T ∧
    0 < M ∧ (∀ x, f x ≤ M) ∧ (∃ x, f x = M) ∧ 
    T = 6 * π ∧ M = √2 := sorry

end min_period_max_value_f_l33_33131


namespace domain_of_f_l33_33905

-- Definition of the function f
def f (x : ℝ) : ℝ := real.log (x - 2)

-- Definition of the domain condition for the function f
def domain_condition (x : ℝ) : Prop := x > 2

-- Main theorem statement
theorem domain_of_f : ∀ x : ℝ, domain_condition x ↔ (∃ y : ℝ, f y = f x) := 
by sorry

end domain_of_f_l33_33905


namespace range_of_a_l33_33357

def proposition_p (a : ℝ) : Prop :=
f (-1) * f (1) = (1 - a - 2) * (1 + a - 2) ≤ 0 ∧ a ≠ 0

def proposition_q (a : ℝ) : Prop :=
∀ x ∈ set.Icc (1/2) (3/2), x^2 + 3*(a + 1)*x + 2 ≤ 0

theorem range_of_a (a : ℝ) : 
  ¬ (proposition_p a ∧ proposition_q a) ↔ a > -5/2 :=
by sorry

end range_of_a_l33_33357


namespace num_non_congruent_rectangles_6x6_grid_l33_33769

theorem num_non_congruent_rectangles_6x6_grid : 
  let grid_points := 7 in -- The number of grid points in one direction (0 to 6)
  (nat.choose grid_points 2) * (nat.choose grid_points 2) = 441 :=
by
  let grid_points := 7
  have h1 : nat.choose grid_points 2 = 21 := sorry
  have h2 : (21 : ℕ) * (21 : ℕ) = 441 := sorry
  exact h2

end num_non_congruent_rectangles_6x6_grid_l33_33769


namespace min_period_of_f_max_value_of_f_l33_33157

def f : ℝ → ℝ := λ x, Real.sin (x / 3) + Real.cos (x / 3)

theorem min_period_of_f : ∃ T > 0, (∀ x, f (x + T) = f x) ∧ T = 6 * Real.pi :=
by
  sorry

theorem max_value_of_f : ∃ x, f x = Real.sqrt 2 :=
by
  sorry

end min_period_of_f_max_value_of_f_l33_33157


namespace part_a_part_b_l33_33767

open Classical

namespace GraphTheory

-- Define the properties of G_n for part (a)
variables (n : ℕ) (G : SimpleGraph (Fin n)) (K23 : SimpleGraph (Fin 2) ⥤ SimpleGraph (Fin 3))

-- Assume G does not contain K_{2,3}
axiom K23_free (G : SimpleGraph (Fin n)) : ¬(∃ (f : SimpleGraph (Fin 2) ⥤ SimpleGraph (Fin 3)), F.obj G)

-- Part (a): Prove the edge bound
theorem part_a (hn : n > 0) (hK : K23_free G) : G.edge_finset.card ≤ (n * (n^0.5) / (=2^0.5)) + n := sorry

end GraphTheory

namespace Geometry

-- Define the properties of the points for part (b)
variables (P : Fin n → ℝ × ℝ) (n : ℕ)

-- Ensure n ≥ 16
axiom n_ge_16 (n : ℕ) : n ≥ 16

-- Function to check unit distance
def unit_distance (P : Fin n → ℝ × ℝ) (i j : Fin n) : Prop :=
  let (xi, yi) := P i
  let (xj, yj) := P j
  (xi - xj)^2 + (yi - yj)^2 = 1

-- Part (b): Prove the unit length bound
theorem part_b (hn : n ≥ 16) : (Finset.card { (i, j) | unit_distance P i j }) ≤ n * (n^0.5) := sorry

end Geometry

end part_a_part_b_l33_33767


namespace complex_conjugate_solution_l33_33903

theorem complex_conjugate_solution (z : ℂ) (h : conj z * (1 + complex.i) = 2 * complex.i) : z = 1 - complex.i :=
sorry

end complex_conjugate_solution_l33_33903


namespace probability_neither_red_nor_purple_l33_33622

theorem probability_neither_red_nor_purple :
  let total_balls := 100
  let red_balls := 37
  let purple_balls := 3
  let other_balls := total_balls - (red_balls + purple_balls)
  other_balls / total_balls.toReal = 0.6 :=
by
  sorry

end probability_neither_red_nor_purple_l33_33622


namespace maximize_expression_upper_bound_l33_33199

noncomputable def maximize_expression : ℝ → ℝ := λ t, (3^t - 4 * t) * t / 9^t 

theorem maximize_expression_upper_bound : 
  ∃ y ∈ (Set.Ioo 0 (real.exp real.pi)), 
  (maximize_expression y) = real.log 3 / 16 := sorry

end maximize_expression_upper_bound_l33_33199


namespace find_x_value_l33_33370

theorem find_x_value (x : ℝ) (h : 2^x + 2^x + 2^x + 2^x + 2^x + 2^x = 4096) : x = 9.415 :=
by
  -- Add the proof here
  sorry

end find_x_value_l33_33370


namespace distance_midpoint_kn_to_line_lm_l33_33167

theorem distance_midpoint_kn_to_line_lm
  (circumscribed : is_cyclic_quadrilateral K L M N)
  (hMN : dist M N = 6)
  (hKL : dist K L = 2)
  (hLM : dist L M = 5)
  (perpendicular : ∠ K M N = 90) :
  (distance_from_midpoint_of_segment_to_line K N L M LM) = (sqrt 15 + 3) / 2 :=
sorry

end distance_midpoint_kn_to_line_lm_l33_33167


namespace altitude_of_triangle_l33_33444

-- Definitions
variables {Point : Type*} [IsMetricSpace Point]

structure Triangle :=
  (A B C : Point)

def is_concurrent (AA1 BB1 CC1 : Point → Point) (S : Point) : Prop :=
  ∀ (X : Point → Point), collinear X S (AA1 S) ∧ collinear X S (BB1 S) ∧ collinear X S (CC1 S)

def is_angle_bisector (CC1 : Point → Point) (A1 B1 C1 : Point) : Prop :=
  -- The definition of the angle bisector property can be complex and usually involves ratios.
  sorry

def is_altitude (CC1 : Point → Point) (A B C : Point) :=
  ∃ (H : Point), right_angle C H B ∧ collinear H C A

-- Main theorem
theorem altitude_of_triangle
  {A B C A1 B1 C1 : Point}
  (InscribedTriangle : Triangle.A A1 B1 C1)
  (MainTriangle : Triangle.A A B C)
  (Concurrent : ∃ (S : Point), is_concurrent (λ _, A1) (λ _, B1) (λ _, C1) S)
  (AngleBisector : is_angle_bisector (λ _, C1) A1 B1 C1) :
  is_altitude (λ _, C1) A B C :=
begin
  sorry
end

end altitude_of_triangle_l33_33444


namespace sin_cos_sum_l33_33938

theorem sin_cos_sum (α : ℝ) (h : ∃ p : ℝ × ℝ, p = (2, -1) ∧ ∀ (x y : ℝ), p = (x, y) → ∃ r : ℝ, r = Real.sqrt (x^2 + y^2) ∧ sin α = y / r ∧ cos α = x / r) :
  sin α + cos α = Real.sqrt 5 / 5 :=
sorry

end sin_cos_sum_l33_33938


namespace const_term_is_neg_11_l33_33021

def const_term_expansion : ℤ :=
  let poly := (x^2 + 1) * (1/x - 1)^5 in
  sorry

theorem const_term_is_neg_11 :
  const_term_expansion = -11 :=
  by
  sorry

end const_term_is_neg_11_l33_33021


namespace normal_dist_equal_probability_eq_l33_33893

theorem normal_dist_equal_probability_eq (a : ℝ) (μ σ : ℝ) (h1 : μ = 3) (h2 : σ = 4) :
  let ξ := Normal μ σ in
  (ξ : Probability.MassFunction.toRealMeasure ξ).cumulative_distribution_function (2 * a - 3) = 
  1 - (ξ : Probability.MassFunction.toRealMeasure ξ).cumulative_distribution_function (a + 2) →
  a = 8 / 3 :=
by
  intro h
  sorry

end normal_dist_equal_probability_eq_l33_33893


namespace find_b9_l33_33039

theorem find_b9 {b : ℕ → ℕ} 
  (h1 : ∀ n, b (n + 2) = b (n + 1) + b n)
  (h2 : b 8 = 100) :
  b 9 = 194 :=
sorry

end find_b9_l33_33039


namespace min_positive_period_max_value_l33_33112
open Real

noncomputable def f (x : ℝ) : ℝ := sin (x / 3) + cos (x / 3)

theorem min_positive_period_max_value :
  (∃ T > 0, ∀ x, f(x + T) = f(x) ∧ T = 6 * π) ∧
  (∀ x, f(x) ≤ sqrt 2) ∧ 
  (∃ x, f(x) = sqrt 2) :=
by
  sorry

end min_positive_period_max_value_l33_33112


namespace min_positive_period_f_max_value_f_l33_33066

noncomputable def f (x : ℝ) : ℝ := sin (x / 3) + cos (x / 3)

theorem min_positive_period_f : ∃ T > 0, (∀ x, f (x + T) = f x) ∧ ∀ T', (T' > 0 ∧ ∀ x, f (x + T') = f x) → T ≤ T' :=
  sorry

theorem max_value_f : ∃ M, (∀ x, f x ≤ M) ∧ (∀ ε > 0, ∃ x, M - ε < f x) ∧ M = sqrt 2 :=
  sorry

end min_positive_period_f_max_value_f_l33_33066


namespace compare_logs_and_value_l33_33338

theorem compare_logs_and_value (a b c : ℝ) (ha : a = Real.logBase 2 3) (hb : b = Real.logBase 3 4) (hc : c = 3/2) :
  b < c ∧ c < a :=
by
  -- Initial set up for proof, assuming ha, hb, hc as hypotheses
  -- Providing only the statement as required
  sorry

end compare_logs_and_value_l33_33338


namespace four_digit_numbers_greater_than_3000_l33_33966

theorem four_digit_numbers_greater_than_3000 (digits : Finset ℕ) (H1 : digits = {0, 1, 2, 3, 4}) : 
  ∃ (count : ℕ), count = 48 ∧ 
  ∀ (n : ℕ), n > 3000 ∧ n < 10000 → 
    (∀ (d ∈ digits), ∃! (i : ℕ), n.digit i = d) → count = 48 :=
by
  simp [H1]
  sorry

end four_digit_numbers_greater_than_3000_l33_33966


namespace min_positive_period_and_max_value_of_f_l33_33096

noncomputable def f (x : ℝ) : ℝ := sin (x / 3) + cos (x / 3)

theorem min_positive_period_and_max_value_of_f :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ T = 6 * π) ∧ 
  (∃ x, f x = sqrt 2) :=
by {
  sorry
}

end min_positive_period_and_max_value_of_f_l33_33096


namespace values_for_f_le_0_l33_33932

def f (x : ℝ) : ℝ := 2^|x| - 1

theorem values_for_f_le_0 : {x : ℝ | f x ≤ 0} = {x | x = 0} :=
by
  sorry

end values_for_f_le_0_l33_33932


namespace problem_1_problem_2_mono_decreasing_1_problem_2_mono_increasing_1_problem_2_mono_decreasing_2_problem_3_l33_33378

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * log x + (a + 1) / 2 * x^2 + 1

theorem problem_1 (a : ℝ) (h : a = -1/2) :
  let I := Icc (1/exp 1) exp 1 in
  ∃ x_max x_min ∈ I, 
    (∀ x ∈ I, f a x ≤ f a x_max) ∧ 
    (∀ x ∈ I, f a x_min ≤ f a x) ∧
    f a x_max = (1/2) + (exp 1)^2 / 4 ∧
    f a x_min = 5 / 4 := by
  sorry

theorem problem_2_mono_decreasing_1 (a : ℝ) (h : a ≤ -1) :
  ∀ x > 0, ∀ y > 0, x < y → f a x > f a y := by
  sorry

theorem problem_2_mono_increasing_1 (a : ℝ) (h : a ≥ 0) :
  ∀ x > 0, ∀ y > 0, x < y → f a x < f a y := by
  sorry

theorem problem_2_mono_decreasing_2 (a : ℝ) (h : -1 < a ∧ a < 0) :
  ∀ x > 0, x < sqrt (-a / (a + 1)) → ∀ y > x, y > sqrt (-a / (a + 1)) → f a x > f a y ∧
  ∀ x > sqrt (-a / (a + 1)), ∀ y > sqrt (-a / (a + 1)), x < y → f a x < f a y := by
  sorry

theorem problem_3 (a : ℝ) (h : -1 < a ∧ a < 0) :
  ∀ x > 0, a > 1/exp 1 - 1 → a < 0 → f a x > 1 + a/2 * log (-a) := by
  sorry

end problem_1_problem_2_mono_decreasing_1_problem_2_mono_increasing_1_problem_2_mono_decreasing_2_problem_3_l33_33378


namespace minimum_period_and_max_value_of_f_l33_33043

noncomputable def f (x : ℝ) : ℝ := sin (x / 3) + cos (x / 3)

theorem minimum_period_and_max_value_of_f :
  (∃ T > 0, ∀ x: ℝ, f (x + T) = f x) ∧
  (∀ x: ℝ, f x ≤ sqrt 2 ∧ 
          (∃ y: ℝ, f y = sqrt 2)) :=
by 
  sorry

end minimum_period_and_max_value_of_f_l33_33043


namespace no_linear_term_l33_33787

theorem no_linear_term (m : ℤ) : (∀ (x : ℤ), (x^2 - x + m) * (x - 8) = x^3 - 9*x^2 - 8*m → (8 + m) = 0) → m = -8 :=
by
  sorry

end no_linear_term_l33_33787


namespace perpendicular_line_x_intercept_l33_33595

noncomputable def slope (a b : ℚ) : ℚ := - a / b

noncomputable def line_equation (m y_intercept : ℚ) (x : ℚ) : ℚ :=
  m * x + y_intercept

theorem perpendicular_line_x_intercept :
  let m1 := slope 4 5,
      m2 := (5 / 4),
      y_int := -3
  in 
  ∀ x, line_equation m2 y_int x = 0 → x = 12 / 5 :=
by
  intro x hx
  sorry

end perpendicular_line_x_intercept_l33_33595


namespace sum_of_squares_inequality_l33_33174

theorem sum_of_squares_inequality (n : ℕ) (a : Fin n → ℝ) 
  (h_pos : ∀ i, 0 < a i) 
  (h_sum : ∑ i, a i = 1) : 
  (∏ i, (1 / (a i)^2 - 1)) ≥ (n^2 - 1)^n :=
sorry

end sum_of_squares_inequality_l33_33174


namespace exterior_angle_of_regular_octagon_l33_33023

theorem exterior_angle_of_regular_octagon (sum_of_exterior_angles : ℝ) (n_sides : ℕ) (is_regular : n_sides = 8 ∧ sum_of_exterior_angles = 360) :
  sum_of_exterior_angles / n_sides = 45 := by
  sorry

end exterior_angle_of_regular_octagon_l33_33023


namespace quadrilateral_area_ADEC_l33_33439

/-- Given a quadrilateral \(ADEC\) with specific properties,
    prove its area is \(3.06\sqrt{119}\). 

    Conditions:
    1. \(C\) is a right angle.
    2. \(D\) is the midpoint of \(\overline{AB}\).
    3. \(DE\) is perpendicular to \(\overline{AB}\).
    4. Length of \(\overline{AB}\) is \(24\).
    5. Length of \(\overline{AC}\) is \(10\).
-/
theorem quadrilateral_area_ADEC 
  (A B C D E : Type) [angle : has_angle C 90]
  [midpoint : midpoint D A B]
  [perpendicular : perpendicular D E A B]
  [length_AB : has_length A B 24]
  [length_AC : has_length A C 10] :
  area_quadrilateral A D E C = 3.06 * sqrt 119 :=
sorry

end quadrilateral_area_ADEC_l33_33439


namespace number_of_eighth_graders_l33_33426

theorem number_of_eighth_graders (x y : ℕ) :
  (x > 0) ∧ (y > 0) ∧ (8 + x * y = (x * (x + 3) - 14) / 2) →
  x = 7 ∨ x = 14 :=
by
  sorry

end number_of_eighth_graders_l33_33426


namespace distance_midpoint_kn_to_line_lm_l33_33168

theorem distance_midpoint_kn_to_line_lm
  (circumscribed : is_cyclic_quadrilateral K L M N)
  (hMN : dist M N = 6)
  (hKL : dist K L = 2)
  (hLM : dist L M = 5)
  (perpendicular : ∠ K M N = 90) :
  (distance_from_midpoint_of_segment_to_line K N L M LM) = (sqrt 15 + 3) / 2 :=
sorry

end distance_midpoint_kn_to_line_lm_l33_33168


namespace tom_speed_B_to_C_proof_l33_33188

-- Define the constants and conditions
def speed_A_to_B : ℝ := 60
def average_speed_A_to_C : ℝ := 36

-- Define the function to calculate Tom's speed from B to C
def toms_speed_B_to_C (d : ℝ) (v : ℝ) : Prop := 
  v = 20 ↔ (average_speed_A_to_C = (3 * d) / ((2 * d / speed_A_to_B) + (d / v)))

theorem tom_speed_B_to_C_proof : ∀ d v : ℝ, 
  (speed_A_to_B = 60) → 
  (average_speed_A_to_C = 36) → 
  (tom_speed_B_to_C (2 * d) v) :=
by
  intros d v speedA C_ave
  sorry

end tom_speed_B_to_C_proof_l33_33188


namespace product_fraction_simplification_l33_33681

theorem product_fraction_simplification :
  (∏ n in finset.range 30, (n + 1 + 2) / (n + 1)) = 496 := 
by
  sorry

end product_fraction_simplification_l33_33681


namespace total_area_of_equilateral_triangles_l33_33797

noncomputable def side_length_extension : ℝ := 6
noncomputable def equilateral_triangle_area (s : ℝ) : ℝ := (sqrt 3 / 4) * s^2

theorem total_area_of_equilateral_triangles :
  let total_area := 3 * equilateral_triangle_area side_length_extension in
  total_area = 27 * sqrt 3 :=
by
  sorry

end total_area_of_equilateral_triangles_l33_33797


namespace BC_distance_l33_33809

noncomputable def point_A : ℝ × ℝ × ℝ := (1, 2, 1)
noncomputable def point_B : ℝ × ℝ × ℝ := (-3, -1, 4)
noncomputable def point_C : ℝ × ℝ × ℝ := (1, 2, -1)

noncomputable def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 + (p1.3 - p2.3)^2)

theorem BC_distance : distance point_B point_C = 5 * sqrt 2 := 
by 
  sorry

end BC_distance_l33_33809


namespace second_train_length_is_correct_l33_33621

noncomputable def length_of_second_train (length_first_train : ℝ) (speed_first_train_kmph : ℝ) (speed_second_train_kmph : ℝ) (time_crossing_seconds : ℝ) : ℝ :=
  let speed_first_train_mps := speed_first_train_kmph * (1000 / 3600)
  let speed_second_train_mps := speed_second_train_kmph * (1000 / 3600)
  let relative_speed := speed_first_train_mps + speed_second_train_mps
  let total_distance := relative_speed * time_crossing_seconds
  total_distance - length_first_train

theorem second_train_length_is_correct : length_of_second_train 360 120 80 9 = 139.95 :=
by
  sorry

end second_train_length_is_correct_l33_33621


namespace find_a_l33_33751

theorem find_a
  (a : ℝ)
  (h_max : a^2 / 6 ≤ 1 / 6)
  (h_interval : ∀ x : ℝ, x ∈ Icc (1/4) (1/2) → a * x - (3 / 2) * x^2 ≥ 1 / 8) :
  a = 1 :=
sorry

end find_a_l33_33751


namespace perpendicular_line_x_intercept_l33_33594

noncomputable def slope (a b : ℚ) : ℚ := - a / b

noncomputable def line_equation (m y_intercept : ℚ) (x : ℚ) : ℚ :=
  m * x + y_intercept

theorem perpendicular_line_x_intercept :
  let m1 := slope 4 5,
      m2 := (5 / 4),
      y_int := -3
  in 
  ∀ x, line_equation m2 y_int x = 0 → x = 12 / 5 :=
by
  intro x hx
  sorry

end perpendicular_line_x_intercept_l33_33594


namespace bonnie_egg_count_indeterminable_l33_33306

theorem bonnie_egg_count_indeterminable
    (eggs_Kevin : ℕ)
    (eggs_George : ℕ)
    (eggs_Cheryl : ℕ)
    (diff_Cheryl_combined : ℕ)
    (c1 : eggs_Kevin = 5)
    (c2 : eggs_George = 9)
    (c3 : eggs_Cheryl = 56)
    (c4 : diff_Cheryl_combined = 29)
    (h₁ : eggs_Cheryl = diff_Cheryl_combined + (eggs_Kevin + eggs_George + some_children)) :
    ∀ (eggs_Bonnie : ℕ), ∃ some_children : ℕ, eggs_Bonnie = eggs_Bonnie :=
by
  -- The proof is omitted here
  sorry

end bonnie_egg_count_indeterminable_l33_33306


namespace number_of_non_reduced_fractions_l33_33688

theorem number_of_non_reduced_fractions :
  {N : ℕ | 1 ≤ N ∧ N ≤ 1985 ∧ (Nat.gcd ((N^2 + 3 : ℕ), (N + 2 : ℕ)) > 1)}.card = 283 := sorry

end number_of_non_reduced_fractions_l33_33688


namespace min_period_and_max_value_l33_33123

def f (x : ℝ) := Real.sin (x / 3) + Real.cos (x / 3)

theorem min_period_and_max_value :
  (∀ T > 0, T ≠ 6 * Real.pi → ¬∀ x : ℝ, f(x + T) = f(x)) ∧
  ∃ x : ℝ, f(x) = sqrt 2 :=
by
  sorry

end min_period_and_max_value_l33_33123


namespace all_integers_equal_l33_33352

theorem all_integers_equal (k : ℕ) (a : Fin (2 * k + 1) → ℤ)
(h : ∀ b : Fin (2 * k + 1) → ℤ,
  (∀ i : Fin (2 * k + 1), b i = (a ((i : ℕ) % (2 * k + 1)) + a ((i + 1) % (2 * k + 1))) / 2) →
  ∀ i : Fin (2 * k + 1), ↑(b i) % 2 = 0) :
∀ i j : Fin (2 * k + 1), a i = a j :=
by
  sorry

end all_integers_equal_l33_33352


namespace min_period_and_max_value_l33_33064

def f (x : ℝ) : ℝ := Real.sin (x / 3) + Real.cos (x / 3)

noncomputable def minimum_positive_period (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x : ℝ, f (x + T) = f x ∧ T > 0 ∧ ∀ T' : ℝ, (T' > 0 ∧ ∀ x : ℝ, f (x + T') = f x) → T' ≥ T

noncomputable def is_max_value (f : ℝ → ℝ) (M : ℝ) : Prop :=
  ∀ x : ℝ, f x ≤ M ∧ ∃ x : ℝ, f x = M

theorem min_period_and_max_value :
  minimum_positive_period f (6 * Real.pi) ∧ is_max_value f (Real.sqrt 2) := by
  sorry

end min_period_and_max_value_l33_33064


namespace find_geometric_numbers_l33_33891

-- Definition of the problem setup
def is_geometric_progression (l : List ℕ) : Prop :=
  ∃ (q : ℕ), q > 1 ∧ ∀ i < l.length - 1, l[i + 1] = l[i] * q

def sum_of_elements (l : List ℕ) : ℕ :=
  l.foldr (· + ·) 0

theorem find_geometric_numbers (l : List ℕ) :
  is_geometric_progression l →
  sum_of_elements l = 211 →
  l = [211] ∨ l = [1, 210] ∨ l = [1, 14, 196] :=
by sorry

end find_geometric_numbers_l33_33891


namespace aquatic_reserve_total_fishes_l33_33259

-- Define the number of bodies of water
def bodies_of_water : ℕ := 6

-- Define the number of fishes per body of water
def fishes_per_body : ℕ := 175

-- Define the total number of fishes
def total_fishes : ℕ := bodies_of_water * fishes_per_body

theorem aquatic_reserve_total_fishes : bodies_of_water * fishes_per_body = 1050 := by
  -- The proof is omitted.
  sorry

end aquatic_reserve_total_fishes_l33_33259


namespace least_positive_difference_seq_C_seq_D_is_6_l33_33497

noncomputable def sequence_C := [3, 9, 27, 81, 243]
noncomputable def sequence_D := [15, 45, 75, 105, 135, 165, 195, 225, 255, 285, 315, 345, 375, 405, 435, 465, 495]

theorem least_positive_difference_seq_C_seq_D_is_6 : 
  let diffs := { abs (c - d) | c in sequence_C, d in sequence_D, c ≠ d, abs (c - d) > 0 } in
  Inf diffs = 6 :=
by sorry

end least_positive_difference_seq_C_seq_D_is_6_l33_33497


namespace length_of_AB_l33_33388

def parabola_eq (y : ℝ) : Prop := y^2 = 8 * y

def directrix_x : ℝ := 2

def dist_to_y_axis (E : ℝ × ℝ) : ℝ := E.1

theorem length_of_AB (A B F E : ℝ × ℝ)
  (p : parabola_eq A.2) (q : parabola_eq B.2) 
  (F_focus : F.1 = 2 ∧ F.2 = 0) 
  (midpoint_E : E = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (E_distance_from_y_axis : dist_to_y_axis E = 3) : 
  (abs (A.1 - B.1) + abs (A.2 - B.2)) = 10 := 
sorry

end length_of_AB_l33_33388


namespace find_x_values_l33_33316

theorem find_x_values (x : ℝ) :
  (x / 4 ≤ 3 + 2 * x ∧ 3 + 2 * x < -3 * (1 + 2 * x)) ↔ - (12 / 7 : ℝ) ≤ x ∧ x < - (3 / 4 : ℝ) :=
by
  sorry

end find_x_values_l33_33316


namespace find_angle_of_triangle_l33_33766

variable {a b l : ℝ}

theorem find_angle_of_triangle (h1 : a > 0) (h2 : b > 0) (h3 : l > 0) :
  let α := 2 * Real.arccos (l * (a + b) / (2 * a * b)) in
  α = 2 * Real.arccos (l * (a + b) / (2 * a * b)) :=
sorry

end find_angle_of_triangle_l33_33766


namespace imaginary_part_complex_l33_33347

open Complex

theorem imaginary_part_complex (z : ℂ) (h : I * conj z = 2 + I) : z.im = 2 := sorry

end imaginary_part_complex_l33_33347


namespace number_of_people_to_the_left_of_Kolya_l33_33537

-- Defining the conditions
variables (left_sasha right_sasha right_kolya total_students left_kolya : ℕ)

-- Condition definitions
def condition1 := right_kolya = 12
def condition2 := left_sasha = 20
def condition3 := right_sasha = 8

-- Calculate total number of students
def calc_total_students : ℕ := left_sasha + right_sasha + 1

-- Calculate number of students to the left of Kolya
def calc_left_kolya (total_students right_kolya : ℕ) : ℕ := total_students - right_kolya - 1

-- Problem statement to prove
theorem number_of_people_to_the_left_of_Kolya
    (H1 : condition1)
    (H2 : condition2)
    (H3 : condition3)
    (total_students : calc_total_students = 29) : 
    calc_left_kolya total_students right_kolya = 16 :=
by
  sorry

end number_of_people_to_the_left_of_Kolya_l33_33537


namespace minimum_period_and_max_value_of_f_l33_33045

noncomputable def f (x : ℝ) : ℝ := sin (x / 3) + cos (x / 3)

theorem minimum_period_and_max_value_of_f :
  (∃ T > 0, ∀ x: ℝ, f (x + T) = f x) ∧
  (∀ x: ℝ, f x ≤ sqrt 2 ∧ 
          (∃ y: ℝ, f y = sqrt 2)) :=
by 
  sorry

end minimum_period_and_max_value_of_f_l33_33045


namespace smallest_n_condition_find_smallest_n_l33_33323

-- A theorem statement that captures the described problem and conclusion
theorem smallest_n_condition (n : ℕ) :
  (∀ (A : Fin n → ℝ × ℝ), -- any n points on a circle represented as pairs of coordinates
  let angles := (Finset.univ.image (λ ⟨i, j⟩, angle (A i) (0,0) (A j))) in
  -- collection of angles between the points through the circle's center
  (angles.filter (λ θ, θ ≤ 120)).card ≥ 2007) ↔ n ≥ 91 :=
begin
  sorry
end

-- The Lean statement which concludes the smallest such n
theorem find_smallest_n : ∃ n : ℕ, 
  (∀ (A : Fin n → ℝ × ℝ),
  let angles := (Finset.univ.image (λ ⟨i, j⟩, angle (A i) (0,0) (A j))) in
  (angles.filter (λ θ, θ ≤ 120)).card ≥ 2007) 
  ∧ n = 91 :=
begin
  use 91,
  split,
  { sorry },
  { refl }
end

end smallest_n_condition_find_smallest_n_l33_33323


namespace gift_combinations_l33_33640

theorem gift_combinations (num_wrapping_paper : ℕ) (num_ribbon : ℕ) (num_gift_tag : ℕ) 
  (h1 : num_wrapping_paper = 10) 
  (h2 : num_ribbon = 5) 
  (h3 : num_gift_tag = 6) : 
  num_wrapping_paper * num_ribbon * num_gift_tag = 300 := 
by
  rw [h1, h2, h3]
  norm_num
  sorry

end gift_combinations_l33_33640


namespace log_inequality_reversed_l33_33359

variables {a x y : ℝ}

-- Conditions
variable (h1 : 0 < a ∧ a < 1)
variable (h2 : log a x < log a y ∧ log a y < 0)

-- The statement to be proved
theorem log_inequality_reversed (h1 : 0 < a ∧ a < 1) (h2 : log a x < log a y ∧ log a y < 0) : x > y ∧ y > 1 :=
by sorry

end log_inequality_reversed_l33_33359


namespace min_value_reciprocal_sum_l33_33362

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 3 ^ a * 3 ^ b = 3 ^ 2) :
  1 / a + 1 / b = 2 :=
by
  sorry

end min_value_reciprocal_sum_l33_33362


namespace subtraction_example_l33_33673

theorem subtraction_example : 3.57 - 1.45 = 2.12 :=
by 
  sorry

end subtraction_example_l33_33673


namespace sector_area_l33_33545

-- Definitions for conditions
def r : ℝ := 6  -- radius of the circle in cm
def alpha : ℝ := real.pi / 6  -- central angle in radians

-- Problem statement
theorem sector_area : (1/2 * alpha * r^2) = 3 * real.pi :=
by
  sorry

end sector_area_l33_33545


namespace slope_y_intercept_diff_l33_33490

-- Definitions for points C and D given their coordinates
variables (xC xD : ℝ) 
variable (xC_ne_xD : xC ≠ xD) -- Different x-coordinates
variables (y : ℝ) (h : y = 25) -- Identical y-coordinates

-- The theorem to be proved
theorem slope_y_intercept_diff :
  let m := 0 in          -- slope since y-coordinates are equal
  let y_intercept := 25 in  -- y-intercept since line is horizontal at y=25
  let difference := m - y_intercept in
  difference = -25 :=
by 
  sorry

end slope_y_intercept_diff_l33_33490


namespace find_k_l33_33925

theorem find_k (k : ℝ) (h : ∀ x : ℝ, x^2 + 10 * x + k = 0 → (∃ a : ℝ, a > 0 ∧ (x = -3 * a ∨ x = -a))) :
  k = 18.75 :=
sorry

end find_k_l33_33925


namespace number_of_people_to_the_left_of_Kolya_l33_33533

-- Defining the conditions
variables (left_sasha right_sasha right_kolya total_students left_kolya : ℕ)

-- Condition definitions
def condition1 := right_kolya = 12
def condition2 := left_sasha = 20
def condition3 := right_sasha = 8

-- Calculate total number of students
def calc_total_students : ℕ := left_sasha + right_sasha + 1

-- Calculate number of students to the left of Kolya
def calc_left_kolya (total_students right_kolya : ℕ) : ℕ := total_students - right_kolya - 1

-- Problem statement to prove
theorem number_of_people_to_the_left_of_Kolya
    (H1 : condition1)
    (H2 : condition2)
    (H3 : condition3)
    (total_students : calc_total_students = 29) : 
    calc_left_kolya total_students right_kolya = 16 :=
by
  sorry

end number_of_people_to_the_left_of_Kolya_l33_33533


namespace find_k_l33_33220

/-- Definitions of the vectors involved --/
def vec (x y : ℝ) : ℝ × ℝ := (x, y)
def e1 : ℝ × ℝ := vec 1 0
def e2 : ℝ × ℝ := vec 0 1
def AB : ℝ × ℝ := vec 1 (-1)
def BC : ℝ × ℝ := vec 3 2
def CD (k : ℝ) : ℝ × ℝ := vec k 2
def AC : ℝ × ℝ := vec 4 1

/-- Collinearity condition for points A, C, and D. We express that AC is a scalar multiple of CD. --/
def collinear (AC CD : ℝ × ℝ) (k : ℝ) : Prop := ∃ λ, AC = (λ * k, λ * 2)

/-- The proof statement --/
theorem find_k : ∃ k, collinear AC (CD k) k ∧ k = 8 :=
by
  sorry

end find_k_l33_33220


namespace triangle_area_l33_33792

noncomputable def area_of_triangle (A B C : ℝ) (AC BC : ℝ) [Fact (0 < A)] [Fact (A < π)] [Fact (0 < B)] [Fact (B < π)] : ℝ :=
  1/2 * AC * BC * Real.sin C

theorem triangle_area :
  let A := π / 3
  let AC := 4
  let BC := 2 * Real.sqrt 3 
  ∃ B C, (0 < B) ∧ (B < π) ∧ 
         (C = π - A - B) ∧
         (Real.sin B = AC * Real.sin A / BC) ∧
         area_of_triangle A B C AC BC = 2 * Real.sqrt 3 :=
by
  let A := π / 3
  let AC := 4
  let BC := 2 * Real.sqrt 3
  sorry

end triangle_area_l33_33792


namespace option_d_correct_l33_33986

theorem option_d_correct (a b c : ℝ) (h : a > b ∧ b > c ∧ c > 0) : a / b < a / c :=
by
  sorry

end option_d_correct_l33_33986


namespace angle_AOD_l33_33438

noncomputable theory
open_locale classical
open real

variables {O A B C D : Type}
variables (α β γ δ : ℝ)

-- Definition of the conditions
def is_perpendicular (α β : ℝ) := α = β + 90 ∨ α + 90 = β
def sum_of_angles (x : ℝ) := 180 - x
def angle_relationship (α ω : ℝ) := α = 2.5 * ω

-- Statement
theorem angle_AOD
  (h1 : is_perpendicular α β = is_perpendicular α γ)
  (h2 : is_perpendicular β δ = is_perpendicular β γ)
  (h3 : α = 2.5 * β):
  α = 128.57 :=
  sorry

end angle_AOD_l33_33438


namespace percentage_employees_four_years_or_more_l33_33555

theorem percentage_employees_four_years_or_more 
  (x : ℝ) 
  (less_than_one_year : ℝ := 6 * x)
  (one_to_two_years : ℝ := 4 * x)
  (two_to_three_years : ℝ := 7 * x)
  (three_to_four_years : ℝ := 3 * x)
  (four_to_five_years : ℝ := 3 * x)
  (five_to_six_years : ℝ := 1 * x)
  (six_to_seven_years : ℝ := 1 * x)
  (seven_to_eight_years : ℝ := 2 * x)
  (total_employees : ℝ := 27 * x)
  (employees_four_years_or_more : ℝ := 7 * x) : 
  (employees_four_years_or_more / total_employees) * 100 = 25.93 := 
by
  sorry

end percentage_employees_four_years_or_more_l33_33555


namespace x_intercept_of_perpendicular_line_l33_33591

noncomputable def x_intercept_perpendicular (m₁ m₂ : ℚ) : ℚ :=
  let m_perpendicular := -1 / m₁ in
  let b := -3 in
  -b / m_perpendicular

theorem x_intercept_of_perpendicular_line :
  (4 * x_intercept_perpendicular (-4/5) (5/4) + 5 * 0) = 10 :=
by
  sorry

end x_intercept_of_perpendicular_line_l33_33591


namespace distance_boguli_to_bolifoyn_l33_33970

-- Define all necessary variables and conditions
variables {x y : ℝ}

-- Given conditions
def condition1 : Prop := (x - y) / 2 = y
def condition2 : Prop := y = x / 3
def condition3 : Prop := x - 5 = (x / 3) + 5
def condition4 : Prop := x = 5 * 2 -- Total travel distance corresponds to travel time

-- Final statement to prove
theorem distance_boguli_to_bolifoyn (x y : ℝ) 
  (h1 : condition1)
  (h2 : condition2)
  (h3 : condition3)
  (h4 : condition4) : 
  x = 10 :=
sorry

end distance_boguli_to_bolifoyn_l33_33970


namespace collinear_E_I_F_l33_33878

theorem collinear_E_I_F
  {A B C D P I E F: Type*} [AddCommGroup P] [Module ℝ P]
  (hP: ∃ O: P, (P: ℝ) = O) -- P lies on segment AB
  (hω: ∃ r: ℝ, circle (triangle C P D) ∧ incenter (triangle C P D)) -- ω is the incircle of triangle CPD and I is its incenter
  (h_tangent1: ∃ K: P, tangent_incircle (triangle A P D) at_point K) -- ω tangent at K to the incircle of triangle APD
  (h_tangent2: ∃ L: P, tangent_incircle (triangle B P C) at_point L) -- ω tangent at L to the incircle of triangle BPC
  (hE: ∃ E: P, E = intersection_line (line A C) (line B D)) -- E is where AC and BD meet
  (hF: ∃ F: P, F = intersection_line (line A K) (line B L)) -- F is where AK and BL meet
  : collinear E I F :=
sorry  -- Proof omitted

end collinear_E_I_F_l33_33878


namespace min_pos_period_max_value_l33_33152

noncomputable def f (x : ℝ) := Real.sin (x / 3) + Real.cos (x / 3)

theorem min_pos_period_max_value :
  (∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ T) ∧ T = 6 * Real.pi) ∧ 
  (∀ x, f x ≤ sqrt 2 ∧ (∃ x, f x = sqrt 2)) :=
by
  sorry

end min_pos_period_max_value_l33_33152


namespace min_value_l33_33513

noncomputable def log_func (a : ℝ) (x : ℝ) : ℝ := log a (x + 3) - 1

theorem min_value (a m n : ℝ) (h_cond1: a > 0) (h_cond2: a ≠ 1) (h_cond3: log_func a (-2) = -1) 
(h_cond4: m > 0) (h_cond5: n > 0) (h_cond6: -2 * m + -1 * n + 1 = 0) : 
  1/m + 1/n = 3 + 2*sqrt(2) := 
sorry

end min_value_l33_33513


namespace min_pos_period_max_value_l33_33143

noncomputable def f (x : ℝ) := Real.sin (x / 3) + Real.cos (x / 3)

theorem min_pos_period_max_value :
  (∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ T) ∧ T = 6 * Real.pi) ∧ 
  (∀ x, f x ≤ sqrt 2 ∧ (∃ x, f x = sqrt 2)) :=
by
  sorry

end min_pos_period_max_value_l33_33143


namespace range_of_a_for_real_root_interval_l33_33415

theorem range_of_a_for_real_root_interval :
  (∃ x ∈ Ioo (-2 : ℝ) 0, a * x^2 - 2 * a * x + a - 9 = 0) ↔ 
  a ∈ Set.Iio (-9) ∪ Set.Ioo 1 9 ∪ Set.Ioi 9 :=
by
  sorry

end range_of_a_for_real_root_interval_l33_33415


namespace rocket_max_height_and_danger_l33_33028

theorem rocket_max_height_and_danger:
  (a : ℝ) (τ : ℝ) (g : ℝ) (y_obj : ℝ) 
  (a = 30) (τ = 30) (g = 10) (y_obj = 50000):
  let V₀ := a * τ,
      y₀ := (a * τ^2) / 2,
      t_high := V₀ / g,
      y_max := y₀ + V₀ * t_high - (g * t_high^2) / 2 in
  (y_max = 54000) ∧ (y_max > y_obj) :=
by
  dsimp [V₀, y₀, t_high, y_max]
  have V0_pos : V₀ = 900 := by sorry
  have y0_pos : y₀ = 13500 := by sorry
  have t_high_pos : t_high = 90 := by sorry
  have y_max_pos : y_max = 54000 := by sorry
  exact ⟨y_max_pos, by linarith [y_max_pos]⟩

end rocket_max_height_and_danger_l33_33028


namespace attendees_gift_exchange_l33_33268

theorem attendees_gift_exchange (x : ℕ) (h1 : 56 = x * (x - 1) / 2) : 
  x * (x - 1) = 112 :=
by
  sorry

end attendees_gift_exchange_l33_33268


namespace tangent_line_at_zero_l33_33033

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.sin x

theorem tangent_line_at_zero : ∀ x : ℝ, x = 0 → Real.exp x * Real.sin x = 0 ∧ (Real.exp x * (Real.sin x + Real.cos x)) = 1 → (∀ y, y = x) :=
  by
    sorry

end tangent_line_at_zero_l33_33033


namespace initial_bananas_tree_l33_33639

-- Definitions for the conditions
def bananas_left_on_tree : ℕ := 100
def bananas_eaten_by_raj : ℕ := 70
def bananas_in_basket_of_raj := 2 * bananas_eaten_by_raj
def bananas_cut_from_tree := bananas_eaten_by_raj + bananas_in_basket_of_raj
def initial_bananas_on_tree := bananas_cut_from_tree + bananas_left_on_tree

-- The theorem to be proven
theorem initial_bananas_tree : initial_bananas_on_tree = 310 :=
by sorry

end initial_bananas_tree_l33_33639


namespace velvet_needed_for_box_l33_33479

theorem velvet_needed_for_box :
  let long_side_area := 2 * (8 * 6)
  let short_side_area := 2 * (5 * 6)
  let top_and_bottom_area := 2 * 40
  long_side_area + short_side_area + top_and_bottom_area = 236 := by
{
  let long_side_area := 2 * (8 * 6)
  let short_side_area := 2 * (5 * 6)
  let top_and_bottom_area := 2 * 40
  sorry
}

end velvet_needed_for_box_l33_33479


namespace complex_sum_i_l33_33325

noncomputable def i : ℂ := complex.I

theorem complex_sum_i :
  let i_sum := ∑ k in finset.range 2014, i^k
  i^2 = -1 ∧ i^4 = 1 ∧ (∀ k, (i^(k + 4) = i^k)) → i_sum = i :=
by {
  sorry
}

end complex_sum_i_l33_33325


namespace question_I_question_II_question_III_l33_33385

noncomputable def f (x a : ℝ) : ℝ := ln (1 + x) - (a * x) / (x + 1)

theorem question_I (h : ∀ x, deriv (λ x, f x 2) 1 = 0) : a = 2 :=
sorry

theorem question_II (h : ∀ x, f x a ≥ 0) : 0 < a ∧ a ≤ 1 :=
sorry

theorem question_III : (2015 / 2016) ^ 2016 < 1 / Real.exp 1 :=
sorry

end question_I_question_II_question_III_l33_33385


namespace kelly_harvested_pounds_l33_33824

def total_carrots (bed1 bed2 bed3 : ℕ) : ℕ :=
  bed1 + bed2 + bed3

def total_weight (total : ℕ) (carrots_per_pound : ℕ) : ℕ :=
  total / carrots_per_pound

theorem kelly_harvested_pounds :
  total_carrots 55 101 78 = 234 ∧ total_weight 234 6 = 39 :=
by {
  split,
  { exact rfl }, -- 234 = 234
  { exact rfl }  -- 234 / 6 = 39
}

end kelly_harvested_pounds_l33_33824


namespace inner_cube_properties_l33_33250

theorem inner_cube_properties
    (outer_cube_surface_area : ℝ)
    (h₀ : outer_cube_surface_area = 54)
    (h₁ : ∃ s, 6 * s^2 = outer_cube_surface_area)
    (inner_cube_inside_sphere : ∀ d s, d = s * real.sqrt 3) :
    (∃ inner_cube_surface_area inner_cube_volume : ℝ,
        inner_cube_surface_area = 18 ∧
        inner_cube_volume = 3 * real.sqrt 3) :=
by
  sorry

end inner_cube_properties_l33_33250


namespace min_period_and_max_value_l33_33058

def f (x : ℝ) : ℝ := Real.sin (x / 3) + Real.cos (x / 3)

noncomputable def minimum_positive_period (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x : ℝ, f (x + T) = f x ∧ T > 0 ∧ ∀ T' : ℝ, (T' > 0 ∧ ∀ x : ℝ, f (x + T') = f x) → T' ≥ T

noncomputable def is_max_value (f : ℝ → ℝ) (M : ℝ) : Prop :=
  ∀ x : ℝ, f x ≤ M ∧ ∃ x : ℝ, f x = M

theorem min_period_and_max_value :
  minimum_positive_period f (6 * Real.pi) ∧ is_max_value f (Real.sqrt 2) := by
  sorry

end min_period_and_max_value_l33_33058


namespace complex_problem_solution_l33_33729

noncomputable def complex_problem : Prop :=
  ∃ (z1 z2 : ℂ), (complex.abs z1 = 2) ∧ (complex.abs z2 = 3) ∧ (3 * z1 - 2 * z2 = 2 - complex.I) ∧ (z1 * z2 = -18/5 + 24/5 * complex.I)

theorem complex_problem_solution : complex_problem :=
  sorry

end complex_problem_solution_l33_33729


namespace min_positive_period_and_max_value_l33_33100

noncomputable def f (x : ℝ) : ℝ := Real.sin (x / 3) + Real.cos (x / 3)

theorem min_positive_period_and_max_value :
  (∃ T > 0, ∀ x, f (x + T) = f x) ∧ ∃ x, f x = sqrt 2 :=
by
  sorry

end min_positive_period_and_max_value_l33_33100


namespace b_sequence_arithmetic_sum_c_sequence_formula_l33_33811

noncomputable def a_sequence (n : ℕ) : ℚ :=
  if n = 1 then 2 else 2 - 1 / a_sequence (n - 1)

noncomputable def b_sequence (n : ℕ) : ℚ :=
  1 / (a_sequence n - 1)

noncomputable def c_sequence (n : ℕ) : ℚ :=
  b_sequence n * 3^(b_sequence n - 1)

noncomputable def sum_c_sequence (n : ℕ) : ℚ :=
  ∑ k in finset.range n, c_sequence (k + 1)

theorem b_sequence_arithmetic (n : ℕ) : b_sequence n = n :=
  sorry

theorem sum_c_sequence_formula (n : ℕ) : sum_c_sequence n = (2 * n - 1) * 3^n / 4 + 1 / 4 :=
  sorry

end b_sequence_arithmetic_sum_c_sequence_formula_l33_33811


namespace problem_1_problem_2_problem_3_problem_4_l33_33675

-- Problem 1
theorem problem_1 (x y : ℝ) : 
  -4 * x^2 * y * (x * y - 5 * y^2 - 1) = -4 * x^3 * y^2 + 20 * x^2 * y^3 + 4 * x^2 * y :=
by
  sorry

-- Problem 2
theorem problem_2 (a : ℝ) :
  (-3 * a)^2 - (2 * a + 1) * (a - 2) = 7 * a^2 + 3 * a + 2 :=
by
  sorry

-- Problem 3
theorem problem_3 (x y : ℝ) :
  (-2 * x - 3 * y) * (3 * y - 2 * x) - (2 * x - 3 * y)^2 = 12 * x * y - 18 * y^2 :=
by
  sorry

-- Problem 4
theorem problem_4 : 2010^2 - 2011 * 2009 = 1 :=
by
  sorry

end problem_1_problem_2_problem_3_problem_4_l33_33675


namespace evaluate_expression_l33_33311

theorem evaluate_expression :
  (log 3 / log 2) * (log 4 / log 3) + (log 24 / log 2 - log 6 / log 2 + 6)^(2/3) = 6 :=
by
  sorry

end evaluate_expression_l33_33311


namespace min_period_max_value_f_l33_33134

def f (x : ℝ) : ℝ := sin (x / 3) + cos (x / 3)

theorem min_period_max_value_f :
  ∃ T M, (∀ x, f (x + T) = f x) ∧ 0 < T ∧
    0 < M ∧ (∀ x, f x ≤ M) ∧ (∃ x, f x = M) ∧ 
    T = 6 * π ∧ M = √2 := sorry

end min_period_max_value_f_l33_33134


namespace exterior_angle_of_regular_octagon_l33_33022

theorem exterior_angle_of_regular_octagon (sum_of_exterior_angles : ℝ) (n_sides : ℕ) (is_regular : n_sides = 8 ∧ sum_of_exterior_angles = 360) :
  sum_of_exterior_angles / n_sides = 45 := by
  sorry

end exterior_angle_of_regular_octagon_l33_33022


namespace minimum_period_and_max_value_of_f_l33_33046

noncomputable def f (x : ℝ) : ℝ := sin (x / 3) + cos (x / 3)

theorem minimum_period_and_max_value_of_f :
  (∃ T > 0, ∀ x: ℝ, f (x + T) = f x) ∧
  (∀ x: ℝ, f x ≤ sqrt 2 ∧ 
          (∃ y: ℝ, f y = sqrt 2)) :=
by 
  sorry

end minimum_period_and_max_value_of_f_l33_33046


namespace number_of_paths_from_A_to_B_is_11_l33_33768

structure PointedGraph (V : Type) :=
(adjacency : V → V → Prop)
(refl : ∀ v, adjacency v v → False)  -- no revisiting points

def vertices : Type := {A B C D E F G : ℕ}

def modified_adjacency : vertices → vertices → Prop
| A, C => True
| A, D => True
| C, A => True
| C, B => True
| D, A => True
| D, C => True
| D, E => True
| D, F => True
| D, G => True
| E, D => True
| E, F => True
| F, E => True
| F, D => True
| F, G => True
| F, C => True
| F, B => True
| G, D => True
| G, F => True
| _ , _ => False

def graph := PointedGraph vertices
{
  adjacency := modified_adjacency,
  refl := λ v h, by cases v; cases h,
}

theorem number_of_paths_from_A_to_B_is_11 : 
  (finset.filter (λ p : List vertices, (hd : p.head? = some A) ∧ (tl : p.last? = some B) ∧ 
                                             ∀ (i j), p[i] = p[j] → i = j ∧ (∀ k p[k+1], graph.adjacency p[k])) 
                 (finset.univ : finset (List vertices))).card = 11 := 
  sorry

end number_of_paths_from_A_to_B_is_11_l33_33768


namespace people_to_left_of_kolya_l33_33528

theorem people_to_left_of_kolya (people_right_kolya people_left_sasha people_right_sasha : ℕ) (total_people : ℕ) :
  (people_right_kolya = 12) →
  (people_left_sasha = 20) →
  (people_right_sasha = 8) →
  (total_people = people_left_sasha + people_right_sasha + 1) →
  total_people - people_right_kolya - 1 = 16 :=
begin
  sorry
end

end people_to_left_of_kolya_l33_33528


namespace petya_can_buy_ice_cream_l33_33452

-- Definitions for the problem conditions
variable (n : ℕ)
variable (total_money : ℕ)
variable (kolya_money : ℕ := n)
variable (vasya_money : ℕ := 18 * n)
variable (petya_money : ℕ := total_money - (kolya_money + vasya_money))

-- Main theorem to prove Petya can buy ice cream
theorem petya_can_buy_ice_cream (h_total: total_money = 2200) (h_kolya_vasya: kolya_money + vasya_money = 19 * n)
  (h_total_condition : total_money >= 19 * n) : petya_money >= 15 :=
by {
  -- Start with the given conditions
  rw [←h_total],
  rw [←h_kolya_vasya],
  -- Combine the conditions to show the required result
  simp [petya_money],
  sorry
}

end petya_can_buy_ice_cream_l33_33452


namespace find_set_A_and_range_g_l33_33382

noncomputable def f (x : ℝ) := 2^x
noncomputable def g (x : ℝ) := (Real.log x / Real.log 2)^2 - (2 * Real.log x / Real.log 2)

theorem find_set_A_and_range_g :
  (A = {x : ℝ | 1/2 ≤ x ∧ x ≤ 4}) ∧ (range (λ x : ℝ, g x) = Icc (-1 : ℝ) 3) :=
by
  let A := {x : ℝ | 1 / 2 ≤ x ∧ x ≤ 4}
  have h1 : ∀ x, (x ∈ A ↔ 1/2 ≤ x ∧ x ≤ 4) := sorry
  have h2 : ∀ y, ∃ x ∈ A, g x = y ↔ y ∈ Icc (-1 : ℝ) 3 := sorry
  exact ⟨h1, h2⟩

end find_set_A_and_range_g_l33_33382


namespace find_k_from_roots_ratio_l33_33919

theorem find_k_from_roots_ratio (k : ℝ) (x1 x2 : ℝ) 
  (h1 : x1 + x2 = -10)
  (h2 : x1 * x2 = k)
  (h3 : x1/x2 = 3) : 
  k = 18.75 := 
sorry

end find_k_from_roots_ratio_l33_33919


namespace nth_letter_120th_is_D_l33_33602

def pattern : List Char := ['A', 'B', 'C', 'D']

def nth_letter_in_pattern (n : ℕ) : Char :=
  pattern[(n - 1) % pattern.length]

theorem nth_letter_120th_is_D : nth_letter_in_pattern 120 = 'D' :=
sorry

end nth_letter_120th_is_D_l33_33602


namespace salmon_at_rest_oxygen_units_l33_33884

noncomputable def salmonSwimSpeed (x : ℝ) : ℝ := (1/2) * Real.log (x / 100 * Real.pi) / Real.log 3

theorem salmon_at_rest_oxygen_units :
  ∃ x : ℝ, salmonSwimSpeed x = 0 ∧ x = 100 / Real.pi :=
by
  sorry

end salmon_at_rest_oxygen_units_l33_33884


namespace find_n_22_or_23_l33_33850

theorem find_n_22_or_23 (n : ℕ) : 
  (∃ (sol_count : ℕ), sol_count = 30 ∧ (∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ 3 * x + 2 * y + 4 * z = n)) → 
  (n = 22 ∨ n = 23) := 
sorry

end find_n_22_or_23_l33_33850


namespace john_father_age_difference_l33_33935

theorem john_father_age_difference (J F X : ℕ) (h1 : J + F = 77) (h2 : J = 15) (h3 : F = 2 * J + X) : X = 32 :=
by
  -- Adding the "sory" to skip the proof
  sorry

end john_father_age_difference_l33_33935


namespace part1_part2_l33_33749

noncomputable def circle_equation (m : ℝ) : Prop := 
  ∀ (x y : ℝ), x^2 + y^2 - 2 * x - 4 * y + m = 0

def line_eq1 (x y : ℝ) : Prop :=
  x + y - 1 = 0

def line_eq2 (x y : ℝ) : Prop :=
  x + 2 * y - 4 = 0

def chord_length {m : ℝ} (h : m = 1) : Prop :=
  2 * Real.sqrt (4 - 2) = 2 * Real.sqrt 2

theorem part1 : chord_length (circle_equation 1) :=
sorry

-- Define perpendicular property
def orthogonal {x1 y1 x2 y2 : ℝ} (x1y1_eq x2y2_eq : ℝ) : Prop :=
  x1 * x2 + y1 * y2 = 0

theorem part2 : ∀ x1 y1 x2 y2 : ℝ, circle_equation (8 / 5) ∧ line_eq2 x1 y1 ∧ line_eq2 x2 y2 ∧ orthogonal x1 y1 x2 y2 :=
sorry

end part1_part2_l33_33749


namespace alpha_in_third_quadrant_l33_33375

theorem alpha_in_third_quadrant (α : ℝ) :
  let p := (Real.cos 2, Real.tan 2)
  in p.1 < 0 ∧ p.2 < 0 → (π < α ∧ α < 3 * π / 2) := 
by
  intros p hp
  sorry

end alpha_in_third_quadrant_l33_33375


namespace part_a_part_b_part_c_l33_33856

-- Definitions of the geometrical entities involved
structure Point where
  x : ℝ
  y : ℝ

def distance (A B : Point) : ℝ :=
  real.sqrt ((B.x - A.x)^2 + (B.y - A.y)^2)

-- Given conditions for Problem (a)
variables (A B C D P : Point)
variables (DA AP DC CP : ℝ)

-- Condition: ABCD is a parallelogram (e.g., opposite sides are equal)
axiom parallelogram (A B C D : Point) : distance A B = distance C D ∧ distance B C = distance D A

-- Circle is inscribed in ΔABC, touching AC at P
axiom incircle_touches_AC_at_P (A B C P : Point) : ∃ I : Point, distance I A = distance I B ∧ distance I B = distance I C ∧ P = (some x coordinates on AC)

-- Problem (a)
theorem part_a (Hparallelogram : parallelogram A B C D)
  (Hincircle_touches_AC_at_P : incircle_touches_AC_at_P A B C P)
  : DA + AP = DC + CP := 
sorry

-- Given conditions for Problem (b)
variables (r1 r2 : ℝ)

-- Circle inside ΔDAP tangent to all sides has radius r1
axiom incircle_DAP_radius (A D P : Point) : ∃ r1 : ℝ, r1 = some radius logic

-- Circle inside ΔDCP tangent to all sides has radius r2
axiom incircle_DCP_radius (C D P : Point) : ∃ r2 : ℝ, r2 = some radius logic

-- Problem (b)
theorem part_b (Hparallelogram : parallelogram A B C D)
  (Hincircle_touches_AC_at_P : incircle_touches_AC_at_P A B C P)
  (Hincircle_DAP_radius : incircle_DAP_radius A D P)
  (Hincircle_DCP_radius : incircle_DCP_radius C D P)
  : r1 / r2 = distance A P / distance C P := 
sorry

-- Additional conditions for Problem (c)
variables (AC : ℝ)

-- Given: DA + DC = 3 * AC and DA = DP
axiom given_conditions (A D C P : Point) (DA DC AC : ℝ)
  : DA + DC = 3 * AC ∧ DA = distance D P

-- Problem (c)
theorem part_c (Hparallelogram : parallelogram A B C D)
  (Hincircle_touches_AC_at_P : incircle_touches_AC_at_P A B C P)
  (Hincircle_DAP_radius : incircle_DAP_radius A D P)
  (Hincircle_DCP_radius : incircle_DCP_radius C D P)
  (Hgiven_conditions : given_conditions A D C P DA DC AC)
  : r1 / r2 = 4 / 3 :=
sorry

end part_a_part_b_part_c_l33_33856


namespace x_intercept_is_correct_l33_33599

-- Define the original line equation
def original_line (x y : ℝ) : Prop := 4 * x + 5 * y = 10

-- Define the perpendicular line's y-intercept
def y_intercept (y : ℝ) : Prop := y = -3

-- Define the equation of the perpendicular line in slope-intercept form
def perpendicular_line (x y : ℝ) : Prop := y = (5 / 4) * x + -3

-- Prove that the x-intercept of the perpendicular line is 12/5
theorem x_intercept_is_correct : ∃ x : ℝ, x ≠ 0 ∧ (∃ y : ℝ, y = 0) ∧ (perpendicular_line x y) :=
sorry

end x_intercept_is_correct_l33_33599


namespace number_of_valid_pairs_l33_33546

def interior_angle (n : ℕ) : ℚ :=
  180 - 360 / n

def valid_ratio (r k : ℕ) : Prop :=
  (interior_angle r) / (interior_angle k) = 4 / 3

def is_valid_pair (r k : ℕ) : Prop :=
  r > 2 ∧ k > 2 ∧ valid_ratio r k

def valid_pairs : list (ℕ × ℕ) :=
  [(42, 7), (18, 6), (10, 5)]

theorem number_of_valid_pairs : 
  (count : ℕ, h : list.countp is_valid_pair valid_pairs = count) : count = 3 :=
by
  sorry

end number_of_valid_pairs_l33_33546


namespace current_speed_l33_33256

theorem current_speed (r w : ℝ) 
  (h1 : 21 / (r + w) + 3 = 21 / (r - w))
  (h2 : 21 / (1.5 * r + w) + 0.75 = 21 / (1.5 * r - w)) 
  : w = 9.8 :=
by
  sorry

end current_speed_l33_33256


namespace range_of_a_l33_33754

noncomputable def f (x a : ℝ) : ℝ := (1 / 2) * x^2 + a * real.log x - (a + 1) * x 

noncomputable def F (x a : ℝ) : ℝ := f x a + (a - 1) * x 

def condition_F_extreme_points (x1 x2 a : ℝ) : Prop := 
  (F x1 a + F x2 a > -2 / real.exp 1 - 2) 

def has_two_extreme_points (a : ℝ) : Prop := 
  ∃ (x1 x2 : ℝ), condition_F_extreme_points x1 x2 a

theorem range_of_a (a : ℝ) (h : has_two_extreme_points a) : 
  0 < a ∧ a < 1 / real.exp 1 := 
sorry

end range_of_a_l33_33754


namespace sum_inequality_l33_33471

theorem sum_inequality (x : Fin 5 → ℝ) (hx : ∀ i, 0 ≤ x i) (hΣ : (∑ i, 1 / (1 + x i)) = 1) :
  (∑ i, x i / (4 + (x i)^2)) ≤ 1 :=
sorry

end sum_inequality_l33_33471


namespace sum_possible_numbers_l33_33488

theorem sum_possible_numbers {a b c d : ℕ} (h1: a = 2) (h2: b = 0) (h3: c = 1) (h4: d = 8) :
  let digits := [a, b, c, d].erase b in
  let one_digit_numbers := digits in
  let two_digit_numbers := (list.permutations digits).filter (λ l, l.head ≠ 0) |>.map (λ l, 10 * l.head! + l.tail.head!) in
  let three_digit_numbers := (list.permutations digits).filter (λ l, l.head ≠ 0) |>.map (λ l, 100 * l.head! + 10 * l.tail.head! + l.tail.tail.head!) in
  let four_digit_numbers := (list.permutations digits).filter (λ l, l.head ≠ 0) |>.map (λ l, 1000 * l.head! + 100 * l.tail.head! + 10 * l.tail.tail.head! + l.tail.tail.tail.head!) in
  (one_digit_numbers ++ two_digit_numbers ++ three_digit_numbers ++ four_digit_numbers).sum = 78331
:= by sorry

end sum_possible_numbers_l33_33488


namespace minimum_period_and_max_value_of_f_l33_33050

noncomputable def f (x : ℝ) : ℝ := sin (x / 3) + cos (x / 3)

theorem minimum_period_and_max_value_of_f :
  (∃ T > 0, ∀ x: ℝ, f (x + T) = f x) ∧
  (∀ x: ℝ, f x ≤ sqrt 2 ∧ 
          (∃ y: ℝ, f y = sqrt 2)) :=
by 
  sorry

end minimum_period_and_max_value_of_f_l33_33050


namespace ellipse_equation_range_of_m2_l33_33728

-- Given conditions
def eccentricity := (sqrt 3) / 2
def quad_perimeter := 4 * sqrt 5
def ellipse_eq (a b : ℝ) := a = 2 ∧ b = 1 ∧ (x^2 + (y^2 / 4) = 1)

def ap_eq_3pb (p a b : ℝ) := (a = -3 * b) ∧ (p = (0, b))

-- The Lean 4 statements

theorem ellipse_equation 
  (a b : ℝ) 
  (eccentricity_eq : eccentricity * a = sqrt(3) / 2 * a)
  (quad_perimeter_eq : sqrt(a^2 + b^2) * 4 = 4 * sqrt 5) : 
  ellipse_eq a b := by
{ 
  sorry 
}

theorem range_of_m2
  (m k : ℝ)
  (line_eq : y = k * x + m)
  (ellipse_eq : 4 * x^2 + y^2 - 4 = 0)
  (dist_eq : ap_eq_3pb m 3)
  (cond : k^2 - m^2 + 4 > 0) : 
  1 < m^2 ∧ m^2 < 4 := by
{ 
  sorry 
}

end ellipse_equation_range_of_m2_l33_33728


namespace max_gcd_b_n_b_n_plus_1_max_possible_value_of_e_n_l33_33844

def b_n (n : ℕ) : ℤ := (10 ^ n - 9) / 3
def e_n (n : ℕ) : ℤ := Int.gcd (b_n n) (b_n (n + 1))

theorem max_gcd_b_n_b_n_plus_1 : ∀ n : ℕ, e_n n ≤ 3 :=
by
  -- Provide the proof here
  sorry

theorem max_possible_value_of_e_n : ∃ n : ℕ, e_n n = 3 :=
by
  -- Provide the proof here
  sorry

end max_gcd_b_n_b_n_plus_1_max_possible_value_of_e_n_l33_33844


namespace probability_same_color_shoes_l33_33625

theorem probability_same_color_shoes (pairs : ℕ) (total_shoes : ℕ)
  (each_pair_diff_color : pairs * 2 = total_shoes)
  (select_2_without_replacement : total_shoes = 10 ∧ pairs = 5) :
  let successful_outcomes := pairs
  let total_outcomes := (total_shoes * (total_shoes - 1)) / 2
  successful_outcomes / total_outcomes = 1 / 9 :=
by
  sorry

end probability_same_color_shoes_l33_33625


namespace min_positive_period_max_value_l33_33119
open Real

noncomputable def f (x : ℝ) : ℝ := sin (x / 3) + cos (x / 3)

theorem min_positive_period_max_value :
  (∃ T > 0, ∀ x, f(x + T) = f(x) ∧ T = 6 * π) ∧
  (∀ x, f(x) ≤ sqrt 2) ∧ 
  (∃ x, f(x) = sqrt 2) :=
by
  sorry

end min_positive_period_max_value_l33_33119


namespace num_shelves_of_picture_books_l33_33288

-- Definitions of the conditions
def books_per_shelf : ℕ := 6
def mystery_shelves : ℕ := 5
def total_books : ℕ := 54

-- Statement of the math proof problem
theorem num_shelves_of_picture_books :
  let mystery_books := mystery_shelves * books_per_shelf in
  let remaining_books := total_books - mystery_books in
  let picture_shelves := remaining_books / books_per_shelf in
  picture_shelves = 4 :=
by
  sorry

end num_shelves_of_picture_books_l33_33288


namespace min_positive_period_and_max_value_of_f_l33_33089

noncomputable def f (x : ℝ) : ℝ := sin (x / 3) + cos (x / 3)

theorem min_positive_period_and_max_value_of_f :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ T = 6 * π) ∧ 
  (∃ x, f x = sqrt 2) :=
by {
  sorry
}

end min_positive_period_and_max_value_of_f_l33_33089


namespace value_to_add_to_make_divisible_l33_33549

noncomputable def smallestNumber : ℕ := 719
noncomputable def LCM_618_3648_60 : ℕ := Nat.lcm (Nat.lcm 618 3648) 60

theorem value_to_add_to_make_divisible :
  ∃ k : ℕ, smallestNumber + k = LCM_618_3648_60 ∧ k = 288721 :=
begin
  sorry
end

end value_to_add_to_make_divisible_l33_33549


namespace triangle_expression_eval_l33_33422

theorem triangle_expression_eval (PQ PR QR : ℝ) (hPQ : PQ = 7) (hPR : PR = 8) (hQR : QR = 6)
    (angle_P angle_Q angle_R : ℝ)
    (h1 : PQ = 7) (h2 : PR = 8) (h3 : QR = 6) :
    (cos ((angle_P - angle_Q) / 2) / sin (angle_R / 2)) - (sin ((angle_P - angle_Q) / 2) / cos (angle_R / 2)) = 16 / 7 := 
sorry

end triangle_expression_eval_l33_33422


namespace segment_division_ratio_l33_33582

-- Definitions and conditions used in the proof problem
variables {O M A B : Point}
variables (circle : Circle) (line1 line2 : Line)

-- Context: circle with center O
-- Point M is outside the circle
-- Lines passing through M touching the circle at A and B
-- Segment OM is divided in half by the circle

-- Definition of being outside the circle
def outside_circle := ¬ (M ∈ circle)

-- Definition of a line passing through a point and touching the circle at a point
def is_tangent_at (l : Line) (P : Point) (C : Circle) : Prop := 
  l ∈ PointsThrough P ∧ l ∩ C = {P}

-- The midpoint of segment OM is in the circle
def midpoint_in_circle := (OM / 2) ∈ circle

-- The ratio segment OM is divided by line AB
theorem segment_division_ratio
  (hM : outside_circle circle M)
  (h1 : is_tangent_at line1 A circle)
  (h2 : is_tangent_at line2 B circle)
  (h_mid : midpoint_in_circle circle OM) :
  Ratio := 
  let K := midpoint OM in 
  (OM ∩ line AB) = K.toRational = 1/4

end segment_division_ratio_l33_33582


namespace triangle_inequality_l33_33548

variable (a b c R r : ℝ)
variable (h1 : a ≤ 1)
variable (h2 : b ≤ 1)
variable (h3 : c ≤ 1)
variable (p := (a + b + c) / 2)
variable (h4 : a > 0)
variable (h5 : b > 0)
variable (h6 : c > 0)

theorem triangle_inequality (hR : R = (a * b * c) / (4 * sqrt((a + b + c)/2 * ((a + b + c)/2 - a) * ((a + b + c)/2 - b) * ((a + b + c)/2 - c))))
                             (hr : r = sqrt(((a + b + c)/2 - a) * ((a + b + c)/2 - b) * ((a + b + c)/2 - c) / (a + b + c)/2)) :
  p * (1 - 2 * R * r) ≤ 1 := by
  sorry

end triangle_inequality_l33_33548


namespace math_problem_l33_33734

theorem math_problem
  (a b c : ℝ)
  (h : a / (30 - a) + b / (70 - b) + c / (80 - c) = 8) :
  6 / (30 - a) + 14 / (70 - b) + 16 / (80 - c) = 5 :=
sorry

end math_problem_l33_33734


namespace min_positive_period_and_max_value_of_f_l33_33087

noncomputable def f (x : ℝ) : ℝ := sin (x / 3) + cos (x / 3)

theorem min_positive_period_and_max_value_of_f :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ T = 6 * π) ∧ 
  (∃ x, f x = sqrt 2) :=
by {
  sorry
}

end min_positive_period_and_max_value_of_f_l33_33087


namespace cookies_distribution_l33_33015

theorem cookies_distribution :
  let initial_boxes := 180
  let extra_boxes := 0.4 * initial_boxes
  let total_boxes := initial_boxes + extra_boxes
  let given_boxes := 35 + 28 + 22 + 17
  let remaining_boxes := total_boxes - given_boxes
  let num_friends := 3
  (remaining_boxes / num_friends) = 50 := 
by 
  simp only [mul_assoc, add_assoc, sub_eq_add_neg, neg_add_eq_sub]
  have initial_boxes_eq : initial_boxes = 180 := rfl
  have extra_boxes_eq : extra_boxes = 0.4 * 180 := rfl
  have total_boxes_eq : total_boxes = 180 + 0.4 * 180 := rfl
  have given_boxes_eq : given_boxes = 35 + 28 + 22 + 17 := rfl
  have remaining_boxes_eq : remaining_boxes = (180 + 0.4 * 180) - (35 + 28 + 22 + 17) := rfl
  have num_friends_eq : num_friends = 3 := rfl
  rw remaining_boxes_eq
  rw num_friends_eq
  norm_num
  sorry

end cookies_distribution_l33_33015


namespace third_racer_sent_time_l33_33246

theorem third_racer_sent_time (a : ℝ) (t t1 : ℝ) :
  t1 = 1.5 * t → 
  (1.25 * a) * (t1 - (1 / 2)) = 1.5 * a * t → 
  t = 5 / 3 → 
  (t1 - t) * 60 = 50 :=
by 
  intro h_t1_eq h_second_eq h_t_value
  rw [h_t1_eq] at h_second_eq
  have t_correct : t = 5 / 3 := h_t_value
  sorry

end third_racer_sent_time_l33_33246


namespace minimum_value_inequality_l33_33363

theorem minimum_value_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : a * b = 1) : 
  ∃ b_min : ℝ, (b = b_min) ∧ ( ∀ a b, a > b ∧ b > 0 ∧ a * b = 1 → 
  (a^2 + b^2)/(a - b) ≥ 2 * real.sqrt 2 ) :=
begin
  let b_min := (real.sqrt 6 - real.sqrt 2) / 2,
  use b_min,
  split,
  { sorry }, -- proof that b_min solves for b
  { intros a b h1 h2 h3,
    sorry } -- proof of the inequality
end

end minimum_value_inequality_l33_33363


namespace calculate_complex_product_l33_33284

theorem calculate_complex_product (a b : ℝ) : ((a + b * Complex.i) * (a - b * Complex.i) * (-a + b * Complex.i) * (-a - b * Complex.i) = (a^2 + b^2)^2) :=
by
  sorry

end calculate_complex_product_l33_33284


namespace distance_between_points_l33_33604

theorem distance_between_points :
  let p1 := (-2, 4, 1) in
  let p2 := (3, -8, 5) in
  dist3d p1 p2 = Real.sqrt 185 :=
by
  sorry

-- Helper definition for the 3-dimensional distance
def dist3d (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2 + (p2.3 - p1.3)^2)

end distance_between_points_l33_33604


namespace irrigation_canal_construction_l33_33187

theorem irrigation_canal_construction (x : ℕ) (m : ℕ) :
  (∀ y : ℕ, 1650 = 3 * 1650 / 2 / (y + 30) → y = 60) →
  (∀ n : ℕ, 14 * 90 + (90 + 120) * (14 - n) = 1650 → n = 5) →
  x = 60 ∧ (x + 30) = 90 ∧ 90 * 5 + (90 + 120) * 9 = 2340 :=
begin
  intros H1 H2,
  split,
  { exact H1 x },
  {
    split,
    { exact (H1 x) + 30 },
    { exact H2 5 },
  }
end

end irrigation_canal_construction_l33_33187


namespace least_prime_factor_of_expression_l33_33977

theorem least_prime_factor_of_expression : ∃ (p : ℕ), prime p ∧ p ∣ (5^5 - 5^4) ∧ (∀ q : ℕ, prime q → q ∣ (5^5 - 5^4) → q ≥ p) :=
by
  sorry

end least_prime_factor_of_expression_l33_33977


namespace equation_of_hyperbolaC1_l33_33371

noncomputable def hyperbolaC2_asymptote (x y : ℝ) : Prop := y = 2 * x ∨ y = -2 * x

noncomputable def hyperbolaC1 (x y a b c: ℝ) : Prop :=
  (a > 0) ∧ (b > 0) ∧ 
  (x^2 / a^2 - y^2 / b^2 = 1) ∧ 
  (c = sqrt 5) ∧ 
  (b = 2 * a) ∧ 
  (a^2 + b^2 = c^2)

theorem equation_of_hyperbolaC1 : 
  ∃ (a b: ℝ), hyperbolaC1 x y a b (sqrt 5) → 
  (x^2 - y^2/4 = 1) :=
by sorry

end equation_of_hyperbolaC1_l33_33371


namespace probability_at_least_one_8_rolled_l33_33959

theorem probability_at_least_one_8_rolled :
  let total_outcomes := 64
  let no_8_outcomes := 49
  (total_outcomes - no_8_outcomes) / total_outcomes = 15 / 64 :=
by
  let total_outcomes := 8 * 8
  let no_8_outcomes := 7 * 7
  have h1 : total_outcomes = 64 := by norm_num
  have h2 : no_8_outcomes = 49 := by norm_num
  rw [← h1, ← h2]
  norm_num
  sorry

end probability_at_least_one_8_rolled_l33_33959


namespace yellow_papers_count_l33_33657

theorem yellow_papers_count (n : ℕ) (total_papers : ℕ) (periphery_papers : ℕ) (inner_papers : ℕ) 
  (h1 : n = 10) 
  (h2 : total_papers = n * n) 
  (h3 : periphery_papers = 4 * n - 4)
  (h4 : inner_papers = total_papers - periphery_papers) :
  inner_papers = 64 :=
by
  sorry

end yellow_papers_count_l33_33657


namespace perpendicular_line_x_intercept_l33_33587

theorem perpendicular_line_x_intercept (x y : ℝ) :
  (4 * x + 5 * y = 10) →
  (1 * y + 0 = y → y = (5 / 4) * x - 3) →
  y = 0 →
  x = 12 / 5 :=
begin
  sorry
end

end perpendicular_line_x_intercept_l33_33587


namespace area_midpoints_of_segments_l33_33001

theorem area_midpoints_of_segments (AB BC : ℝ) (l : ℝ) (m : ℝ) : 
  AB = 3 → BC = 4 → l = 5 → 
  (∃ (P Q : ℝ × ℝ), P.1 ∈ (set.Icc 0 AB) ∧ P.2 = 0 ∧ Q.1 = AB ∧ Q.2 ∈ (set.Icc 0 BC) ∧ (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = l^2 ∧ 
  m = ((5 * Real.pi) / 2)) → 100 * m = 785 :=
begin
  sorry
end

end area_midpoints_of_segments_l33_33001


namespace both_buyers_correct_l33_33229

-- Define the total number of buyers
def total_buyers : ℕ := 100

-- Define the number of buyers who purchase cake mix
def cake_mix_buyers : ℕ := 50

-- Define the number of buyers who purchase muffin mix
def muffin_mix_buyers : ℕ := 40

-- Define the number of buyers who purchase neither cake mix nor muffin mix
def neither_buyers : ℕ := 29

-- Define the number of buyers who purchase both cake and muffin mix
def both_buyers : ℕ := 19

-- The assertion to be proved
theorem both_buyers_correct :
  neither_buyers = total_buyers - (cake_mix_buyers + muffin_mix_buyers - both_buyers) :=
sorry

end both_buyers_correct_l33_33229


namespace problem_min_value_l33_33707

open Real

noncomputable def f (x : ℝ) : ℝ :=
  (x^2 + 9) / sqrt (x^2 + 5)

theorem problem_min_value : (⨀ x : ℝ, f x) = (9 * sqrt 5 / 5) := sorry

end problem_min_value_l33_33707


namespace range_of_a_for_local_min_max_l33_33736

noncomputable def f (a e x : ℝ) : ℝ := 2 * a^x - e * x^2

theorem range_of_a_for_local_min_max (a e x1 x2 : ℝ) (h_a : 0 < a) (h_a_ne : a ≠ 1) (h_x1_x2 : x1 < x2) 
  (h_min : ∀ x, f a e x > f a e x1) (h_max : ∀ x, f a e x < f a e x2) : 
  (1 / Real.exp 1) < a ∧ a < 1 := 
sorry

end range_of_a_for_local_min_max_l33_33736


namespace initial_tiger_sharks_l33_33295

open Nat

theorem initial_tiger_sharks (initial_guppies : ℕ) (initial_angelfish : ℕ) (initial_oscar_fish : ℕ)
  (sold_guppies : ℕ) (sold_angelfish : ℕ) (sold_tiger_sharks : ℕ) (sold_oscar_fish : ℕ)
  (remaining_fish : ℕ) (initial_total_fish : ℕ) (total_guppies_angelfish_oscar : ℕ) (initial_tiger_sharks : ℕ) :
  initial_guppies = 94 → initial_angelfish = 76 → initial_oscar_fish = 58 →
  sold_guppies = 30 → sold_angelfish = 48 → sold_tiger_sharks = 17 → sold_oscar_fish = 24 →
  remaining_fish = 198 →
  initial_total_fish = (sold_guppies + sold_angelfish + sold_tiger_sharks + sold_oscar_fish + remaining_fish) →
  total_guppies_angelfish_oscar = (initial_guppies + initial_angelfish + initial_oscar_fish) →
  initial_tiger_sharks = (initial_total_fish - total_guppies_angelfish_oscar) →
  initial_tiger_sharks = 89 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11
  sorry

end initial_tiger_sharks_l33_33295


namespace incorrect_statement_l33_33984

variables (x : ℝ) (p q : Prop)

-- Conditions
axiom cond1 : x ≠ 2 → x^2 - 5 * x + 6 ≠ 0
axiom cond2 : ∃ x < 1, x^2 - 3 * x + 2 > 0
axiom cond3 : ∀ x ∈ ℝ, x^2 + x + 1 ≠ 0
axiom cond4 : p ∨ q

-- Statement to prove
theorem incorrect_statement : ¬ (p ∧ q) :=
begin
  sorry
end

end incorrect_statement_l33_33984


namespace chicken_burger_cost_l33_33652

namespace BurgerCost

variables (C B : ℕ)

theorem chicken_burger_cost (h1 : B = C + 300) 
                            (h2 : 3 * B + 3 * C = 21000) : 
                            C = 3350 := 
sorry

end BurgerCost

end chicken_burger_cost_l33_33652


namespace x_intercept_of_perpendicular_line_l33_33593

noncomputable def x_intercept_perpendicular (m₁ m₂ : ℚ) : ℚ :=
  let m_perpendicular := -1 / m₁ in
  let b := -3 in
  -b / m_perpendicular

theorem x_intercept_of_perpendicular_line :
  (4 * x_intercept_perpendicular (-4/5) (5/4) + 5 * 0) = 10 :=
by
  sorry

end x_intercept_of_perpendicular_line_l33_33593


namespace geometric_series_sum_eq_l33_33553

open Real

-- Statement of the problem
theorem geometric_series_sum_eq (
  (a : ℝ) (r : ℝ) (n : ℕ)
  (h1 : a = 1/3)
  (h2 : r = 1/3)
  (h3 : (1/2) * (1 - (r^n)) = 728/729)
) : n = 6 := by
sorr

end geometric_series_sum_eq_l33_33553


namespace three_digit_odd_numbers_count_l33_33194

theorem three_digit_odd_numbers_count : 
  (∀ (digits : Fin 4 → Fin 4), (∀ i j, i ≠ j → digits i ≠ digits j) →
  (digits 2 = 1 ∨ digits 2 = 3) →
  (digits 0 ≠ 0 ∨ (digits 0 = 0 → (digits 1 ≠ digits 2 ∧ digits 2 ≠ digits 1))) →
  ∃ l : List (Fin 4), l.length = 3 ∧ l.Nodup ∧ (l.nth 2 = some 1 ∨ l.nth 2 = some 3)) → 8 :=
by
  sorry

end three_digit_odd_numbers_count_l33_33194


namespace handshakes_at_convention_l33_33566

theorem handshakes_at_convention :
  let gremlins := 30
  let imps := 15
  let handshakes_among_gremlins := gremlins * (gremlins - 1) / 2
  let handshakes_between_imps_gremlins := imps * (gremlins / 2)
  handshakes_among_gremlins + handshakes_between_imps_gremlins = 660 :=
by
  let gremlins := 30
  let imps := 15
  let handshakes_among_gremlins := gremlins * (gremlins - 1) / 2
  let handshakes_between_imps_gremlins := imps * (gremlins / 2)
  show handshakes_among_gremlins + handshakes_between_imps_gremlins = 660
  sorry

end handshakes_at_convention_l33_33566


namespace probability_at_least_one_eight_l33_33956

theorem probability_at_least_one_eight :
  let total_outcomes := 64 in
  let outcomes_without_8 := 49 in
  let favorable_outcomes := total_outcomes - outcomes_without_8 in
  let probability := (favorable_outcomes : ℚ) / total_outcomes in
  probability = 15 / 64 :=
by
  let total_outcomes := 64
  let outcomes_without_8 := 49
  let favorable_outcomes := total_outcomes - outcomes_without_8
  let probability := (favorable_outcomes : ℚ) / total_outcomes
  show probability = 15 / 64
  sorry

end probability_at_least_one_eight_l33_33956


namespace applicant_permutation_inequalities_prob_123_over_70_prob_8910_at_most_10_l33_33940

-- Definitions based on the conditions provided
def is_accepted (a : List Nat) (i : Nat) : Bool :=
  i > 3 ∧ (∀ j < i, a[j] > a[i]) ∨ (i = 10 ∧ ∀ j < 10, ¬ is_accepted a j)

/-- There are 10 applicants labeled 1 to 10. The result n_1 > n_2 > ... > n_8 = n_9 = n_{10} holds. --/
theorem applicant_permutation_inequalities 
    (a : List Nat)
    (h1 : a.length = 10)
    (horder : a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) :
    (∑ j in [1], is_accepted a j > ∑ j in [2], is_accepted a j) ∧ 
    (∑ j in [2], is_accepted a j > ∑ j in [3], is_accepted a j) ∧ 
    (∑ j in [7], is_accepted a j > ∑ j in [8]) ∧
    (∑ j in [8], is_accepted a j = ∑ j in [9] = ∑ j in [10]) := sorry

/-- The probability that one of applicants 1, 2, 3 gets the job is greater than 70%. --/
theorem prob_123_over_70 (a : List Nat)
    (h1 : a.length = 10)
    (horder : a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) :
    (∑ i in [1, 2, 3], is_accepted a i) / 10! > 0.7 := sorry

/-- The probability that one of applicants 8, 9, 10 gets the job is not more than 10%. --/
theorem prob_8910_at_most_10 (a : List Nat)
    (h1 : a.length = 10)
    (horder : a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) :
    (∑ i in [8, 9, 10], is_accepted a i) / 10! ≤ 0.1 := sorry

end applicant_permutation_inequalities_prob_123_over_70_prob_8910_at_most_10_l33_33940


namespace tom_linda_distance_difference_l33_33952

theorem tom_linda_distance_difference :
  (let linda_speed := 2.0 -- rate in miles per hour
       tom_speed := 8.0 -- rate in miles per hour
       linda_distance := linda_speed --distance in the first hour
       half_distance_time := (linda_distance / 2) / tom_speed * 60 -- time in minutes for Tom to cover half of the distance Linda covered
       twice_distance_time := (2 * linda_distance) / tom_speed * 60 -- time in minutes for Tom to cover twice the distance Linda covered
    in (twice_distance_time - half_distance_time) = 22.5) :=
begin
  sorry
end

end tom_linda_distance_difference_l33_33952


namespace part1_part2_l33_33848

def f (x : ℝ) : ℝ := |2 * x - 2| + |x + 2|

theorem part1 (x : ℝ) : x ∈ set.Icc (-5 : ℝ) 1 ↔ f x ≤ 5 - 2 * x :=
by
  sorry

theorem part2 {a b : ℝ} (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : a^2 + b^2 + 2 * b = 3) : 
  a + b ≤ 2 * real.sqrt 2 - 1 :=
by
  sorry

end part1_part2_l33_33848


namespace area_of_square_EFGH_l33_33655

-- Define the conditions
def side_length_ABCD : ℝ := 10
def BE : ℝ := 3
def area_EFGH : ℝ := (Real.sqrt 91 - 3) ^ 2

-- Lean 4 statement to prove the area
theorem area_of_square_EFGH :
  area_EFGH = 100 - 6 * Real.sqrt 91 :=
sorry

end area_of_square_EFGH_l33_33655


namespace velocity_equal_distance_l33_33279

theorem velocity_equal_distance (v t : ℝ) (h : v * t = t) (ht : t ≠ 0) : v = 1 :=
by sorry

end velocity_equal_distance_l33_33279


namespace car_speed_l33_33227

theorem car_speed (v : ℝ) (h : true) :
  (v ≈ 65.45) ↔ (1 / v = 1 / 80 + 10 / 3600) := by
  sorry

end car_speed_l33_33227


namespace eval_expression_l33_33632

def expression : Real :=
  (2 + 1/4) ^ (1 / 2) - (-2023) ^ 0 - (27 / 8) ^ (-2 / 3) + (1.5) ^ (-2)

theorem eval_expression : expression = 1 / 2 := by
  sorry

end eval_expression_l33_33632


namespace construction_rates_construction_cost_l33_33185

-- Defining the conditions as Lean hypotheses

def length := 1650
def diff_rate := 30
def time_ratio := 3/2

-- Daily construction rates (questions answered as hypotheses as well)
def daily_rate_A := 60
def daily_rate_B := 90

-- Additional conditions for cost calculations
def cost_A_per_day := 90000
def cost_B_per_day := 120000
def total_days := 14
def alone_days_A := 5

-- Problem stated as proofs to be completed
theorem construction_rates :
  (∀ (x : ℕ), x = daily_rate_A ∧ (x + diff_rate) = daily_rate_B ∧ 
  (1650 / (x + diff_rate)) * (3/2) = (1650 / x) → 
  60 = daily_rate_A ∧ (60 + 30) = daily_rate_B ) :=
by sorry

theorem construction_cost :
  (∀ (m : ℕ), m = alone_days_A ∧ 
  (cost_A_per_day * total_days + cost_B_per_day * (total_days - alone_days_A)) / 1000 = 2340) :=
by sorry

end construction_rates_construction_cost_l33_33185


namespace limit_question_l33_33373

-- Define the function f and its derivative at x = 1.
variable {f : ℝ → ℝ}

-- Assume the condition that the derivative of f at x = 1 is 1.
axiom deriv_f_at_1 : deriv f 1 = 1

-- The theorem we need to prove
theorem limit_question : (lim (fun x => (f (1 - x) - f (1 + x)) / (3 * x)) (𝓝 0) = -2 / 3) :=
by {
  -- Skipping the proof for now.
  sorry
}

end limit_question_l33_33373


namespace integer_points_on_circle_l33_33420

theorem integer_points_on_circle (r : ℕ) (h : r = 5) : 
  ∃ n, n = 12 ∧ 
  {p : ℤ × ℤ | p.1 ^ 2 + p.2 ^ 2 = r ^ 2}.card = n := 
by
  sorry

end integer_points_on_circle_l33_33420


namespace range_of_omega_l33_33512

def f (x : ℝ) (ω : ℝ) : ℝ := 2 * Real.sin (ω * x + Real.pi / 6)

theorem range_of_omega (ω : ℝ) : (∀ x y : ℝ, (π / 2 < x) ∧ (x < y) ∧ (y < π) → f x ω < f y ω) ↔ (0 < ω ∧ ω ≤ 1 / 3) := 
by sorry

end range_of_omega_l33_33512


namespace triangle_problem_l33_33789

theorem triangle_problem
  (a b c : ℝ)
  (A B C : ℝ)
  (ha : a = √3)
  (hA : A = π / 3)
  (law_of_sines : 2 * (a / sin A) = 2) :
  (a + b + c) / (sin A + sin B + sin C) = 2 := by
  sorry

end triangle_problem_l33_33789


namespace coordinates_of_P_l33_33507

-- Define the point P with given coordinates
structure Point where
  x : ℝ
  y : ℝ

-- Define the point P(3, 5)
def P : Point := ⟨3, 5⟩

-- Define a theorem stating that the coordinates of P are (3, 5)
theorem coordinates_of_P : P = ⟨3, 5⟩ :=
  sorry

end coordinates_of_P_l33_33507


namespace find_m_such_that_no_linear_term_in_expansion_l33_33785

theorem find_m_such_that_no_linear_term_in_expansion :
  ∃ m : ℝ, ∀ x : ℝ, (x^2 - x + m) * (x - 8) = x^3 - 9 * x^2 - 8 * m ∧ ((8 + m) = 0) :=
by
  sorry

end find_m_such_that_no_linear_term_in_expansion_l33_33785


namespace hyperbola_eccentricity_l33_33509

theorem hyperbola_eccentricity (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b)
    (tangent : ∀ x y, (x − real.sqrt 3)^2 + (y - 1)^2 = 1 → 
                   |real.sqrt 3 * b - a| / real.sqrt (a^2 + b^2) = 1 ∨ 
                   |real.sqrt 3 * b + a| / real.sqrt (a^2 + b^2) = 1) :
  real.sqrt 3 * a = b → 
  (∀ c, c = real.sqrt (a^2 + b^2) → 2 = c / a) :=
begin
  sorry
end

end hyperbola_eccentricity_l33_33509


namespace required_earnings_correct_l33_33474

-- Definitions of the given conditions
def retail_price : ℝ := 600
def discount_rate : ℝ := 0.10
def sales_tax_rate : ℝ := 0.05
def amount_saved : ℝ := 120
def amount_given_by_mother : ℝ := 250
def additional_costs : ℝ := 50

-- Required amount Maria must earn
def required_earnings : ℝ := 247

-- Lean 4 theorem statement
theorem required_earnings_correct :
  let discount_amount := discount_rate * retail_price
  let discounted_price := retail_price - discount_amount
  let sales_tax_amount := sales_tax_rate * discounted_price
  let total_bike_cost := discounted_price + sales_tax_amount
  let total_cost := total_bike_cost + additional_costs
  let total_have := amount_saved + amount_given_by_mother
  required_earnings = total_cost - total_have :=
by
  sorry

end required_earnings_correct_l33_33474


namespace simplify_expression_l33_33499

theorem simplify_expression (x : ℝ) :
  (2 * x + 30) + (150 * x + 45) + 5 = 152 * x + 80 :=
by
  sorry

end simplify_expression_l33_33499


namespace people_left_of_Kolya_l33_33539

/-- Given:
    1. There are 12 people to the right of Kolya.
    2. There are 20 people to the left of Sasha.
    3. There are 8 people to the right of Sasha.
    4. The total number of people in the class (including Sasha) is 29.

    Prove:
    The number of people to the left of Kolya is 16.
-/
theorem people_left_of_Kolya : 
  ∀ (total_people right_of_Kolya left_of_Sasha right_of_Sasha : ℕ),
  right_of_Kolya = 12 →
  left_of_Sasha = 20 →
  right_of_Sasha = 8 →
  total_people = 29 →
  left_of_Kolya := total_people - right_of_Kolya - 1
  left_of_Kolya = 16 :=
by
  intros
  sorry

end people_left_of_Kolya_l33_33539


namespace rocket_maximum_height_rocket_danger_l33_33031

theorem rocket_maximum_height
  (a : ℝ) (τ : ℝ) (g : ℝ) (m_to_km : ℝ := 0.001)
  (h_a : a = 30) (h_tau : τ = 30) (h_g : g = 10) :
  let V_0 := a * τ,
      y_0 := (a * τ^2) / 2,
      t_max := V_0 / g,
      y_max := y_0 + V_0 * t_max - (g * t_max^2) / 2 
  in y_max * m_to_km = 54 :=
by
  sorry

theorem rocket_danger (y_max : ℝ) (h_max : y_max = 54000) :
  y_max > 50000 :=
by
  simp [h_max]

end rocket_maximum_height_rocket_danger_l33_33031


namespace probability_at_least_one_8_l33_33961

theorem probability_at_least_one_8 (n : ℕ) (hn : n = 8) : 
  (1 - (7/8) * (7/8)) = 15 / 64 :=
by
  rw [← hn]
  sorry

end probability_at_least_one_8_l33_33961


namespace people_to_left_of_kolya_l33_33531

theorem people_to_left_of_kolya (people_right_kolya people_left_sasha people_right_sasha : ℕ) (total_people : ℕ) :
  (people_right_kolya = 12) →
  (people_left_sasha = 20) →
  (people_right_sasha = 8) →
  (total_people = people_left_sasha + people_right_sasha + 1) →
  total_people - people_right_kolya - 1 = 16 :=
begin
  sorry
end

end people_to_left_of_kolya_l33_33531


namespace determine_omega_phi_l33_33381

noncomputable def f (x : ℝ) (ω φ : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

theorem determine_omega_phi :
  ∃ (ω φ : ℝ), ω > 0 ∧ |φ| < Real.pi ∧ 
  f (Real.pi * 5 / 8) ω φ = 2 ∧ 
  f (Real.pi * 11 / 8) ω φ = 0 ∧ 
  (∃ T > 2 * Real.pi, ∀ (x : ℝ), f (x + T) ω φ = f x ω φ) ∧ 
  ω = 2 / 3 ∧ φ = Real.pi / 12 :=
sorry

end determine_omega_phi_l33_33381


namespace solution_correct_l33_33665

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

def decreasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

def is_solution (f : ℝ → ℝ) : Prop :=
  odd_function f ∧ decreasing_function f

theorem solution_correct :
  is_solution (λ x : ℝ, -x^3) ∧
  ¬ is_solution (λ x : ℝ, sin x) ∧
  ¬ is_solution (λ x : ℝ, x) ∧
  ¬ is_solution (λ x : ℝ, (0.5)^x) :=
by sorry

end solution_correct_l33_33665


namespace perpendicular_line_x_intercept_l33_33586

theorem perpendicular_line_x_intercept (x y : ℝ) :
  (4 * x + 5 * y = 10) →
  (1 * y + 0 = y → y = (5 / 4) * x - 3) →
  y = 0 →
  x = 12 / 5 :=
begin
  sorry
end

end perpendicular_line_x_intercept_l33_33586


namespace find_k_l33_33923

-- Given conditions and hypothesis stated
axiom quadratic_eq (x k : ℝ) : x^2 + 10 * x + k = 0

def roots_in_ratio_3_1 (α β : ℝ) : Prop :=
  α / β = 3

-- Statement of the theorem to be proved
theorem find_k {α β k : ℝ} (h1 : quadratic_eq α k) (h2 : quadratic_eq β k)
               (h3 : α ≠ 0) (h4 : β ≠ 0) (h5 : roots_in_ratio_3_1 α β) :
  k = 18.75 :=
by
  sorry

end find_k_l33_33923


namespace least_prime_factor_of_5_pow_5_minus_5_pow_4_l33_33972

theorem least_prime_factor_of_5_pow_5_minus_5_pow_4 : 
  ∃ p : ℕ, nat.prime p ∧ p = 2 ∧ ∃ f : nat.factorization, 
  (5^5 - 5^4) = (f.prod) ∧ p ∈ f.support :=
begin
  sorry
end

end least_prime_factor_of_5_pow_5_minus_5_pow_4_l33_33972


namespace find_millet_weight_l33_33255

-- Definitions based on conditions
def cost_per_pound_millet := 0.60
def cost_per_pound_sunflower := 1.10
def pounds_sunflower := 25
def cost_per_pound_mixture := 0.70

-- Given conditions
def total_cost_millet (M : ℝ) := cost_per_pound_millet * M
def total_cost_sunflower := cost_per_pound_sunflower * pounds_sunflower
def total_cost_mixture (M : ℝ) := cost_per_pound_mixture * (M + pounds_sunflower)

-- Proof problem: Prove that M = 100
theorem find_millet_weight : ∃ M : ℝ, total_cost_millet M + total_cost_sunflower = total_cost_mixture M ∧ M = 100 :=
by
  sorry

end find_millet_weight_l33_33255


namespace sam_fixed_car_cost_l33_33495

noncomputable def sam_s_car_fix : ℝ := 340

def work_earnings (hours : ℕ) (rate_per_hour : ℝ) : ℝ :=
  hours * rate_per_hour

def savings_after_work (initial_savings : ℝ) (hours : ℕ) (rate_per_hour : ℝ) (future_hours : ℕ) : ℝ :=
  initial_savings + work_earnings hours rate_per_hour + work_earnings future_hours rate_per_hour

def total_expenses (total_savings : ℝ) (console_cost : ℝ) : ℝ :=
  total_savings - console_cost

theorem sam_fixed_car_cost {initial_savings hours future_hours console_cost total_hours : ℕ} {rate_per_hour : ℝ} :
  initial_savings = 460 →
  hours = 31 →
  future_hours = 16 →
  rate_per_hour = 20 →
  console_cost = 600 →
  work_earnings hours rate_per_hour + work_earnings 8 rate_per_hour + work_earnings future_hours rate_per_hour = 940 →
  total_expenses 940 600 = 340 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  rw [<-h1, <-h2, <-h3, h4, h5] at h6
  exact h7

example : sam_fixed_car_cost :=
  by
    sorry

end sam_fixed_car_cost_l33_33495


namespace part1_solution_part2_solution_minimum_value_l33_33344

def f (n : ℕ) (x : ℝ) : ℝ :=
  ∑ i in finset.range n, |x - (i + 1)|

theorem part1_solution (x : ℝ) : f 2 x < x + 1 ↔ x ∈ Ioo (2 / 3) 4 :=
by sorry 

theorem part2_solution_minimum_value : ∃ x, f 5 x = 6 :=
by sorry

end part1_solution_part2_solution_minimum_value_l33_33344


namespace gcd_three_numbers_l33_33912

theorem gcd_three_numbers (a b c : ℕ) (h1 : a = 72) (h2 : b = 120) (h3 : c = 168) :
  Nat.gcd (Nat.gcd a b) c = 24 :=
by
  rw [h1, h2, h3]
  exact sorry

end gcd_three_numbers_l33_33912


namespace inequality_proof_l33_33741

variable {n : ℕ} (a : ℝ)
variables (x : Fin n → ℝ) (h_nonneg : ∀ i, 0 ≤ x i) (h_sum : ∑ i, x i = a)

theorem inequality_proof :
  ∑ i in Finset.Ico 1 n, x (i - 1) * x i ≤ a^2 / 4 :=
sorry

end inequality_proof_l33_33741


namespace gathering_gift_exchange_l33_33262

def number_of_guests (x : ℕ) : Prop :=
  x * (x - 1) = 56

theorem gathering_gift_exchange :
  ∃ x : ℕ, number_of_guests x :=
sorry

end gathering_gift_exchange_l33_33262


namespace boxes_needed_to_complete_flooring_l33_33573

-- Definitions of given conditions
def length_of_living_room : ℕ := 16
def width_of_living_room : ℕ := 20
def sq_ft_per_box : ℕ := 10
def sq_ft_already_covered : ℕ := 250

-- Statement to prove
theorem boxes_needed_to_complete_flooring : 
  (length_of_living_room * width_of_living_room - sq_ft_already_covered) / sq_ft_per_box = 7 :=
by
  sorry

end boxes_needed_to_complete_flooring_l33_33573


namespace solution_set_f_inequality_l33_33743

variable (f : ℝ → ℝ)

axiom domain_of_f : ∀ x : ℝ, true
axiom avg_rate_of_f : ∀ (x1 x2 : ℝ), x1 ≠ x2 → (f x1 - f x2) / (x1 - x2) > 3
axiom f_at_5 : f 5 = 18

theorem solution_set_f_inequality : {x : ℝ | f (3 * x - 1) > 9 * x} = {x : ℝ | x > 2} :=
by
  sorry

end solution_set_f_inequality_l33_33743


namespace gathering_gift_exchange_l33_33264

def number_of_guests (x : ℕ) : Prop :=
  x * (x - 1) = 56

theorem gathering_gift_exchange :
  ∃ x : ℕ, number_of_guests x :=
sorry

end gathering_gift_exchange_l33_33264


namespace min_positive_period_and_max_value_l33_33083

def f (x : ℝ) := Real.sin (x / 3) + Real.cos (x / 3)

theorem min_positive_period_and_max_value : 
  (∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T' = T)) 
  ∧ 
  (∀ x, f x ≤ sqrt 2 ∧ (∃ x, f x = sqrt 2)) :=
sorry

end min_positive_period_and_max_value_l33_33083


namespace min_period_of_f_max_value_of_f_l33_33161

def f : ℝ → ℝ := λ x, Real.sin (x / 3) + Real.cos (x / 3)

theorem min_period_of_f : ∃ T > 0, (∀ x, f (x + T) = f x) ∧ T = 6 * Real.pi :=
by
  sorry

theorem max_value_of_f : ∃ x, f x = Real.sqrt 2 :=
by
  sorry

end min_period_of_f_max_value_of_f_l33_33161


namespace min_positive_period_and_max_value_l33_33102

noncomputable def f (x : ℝ) : ℝ := Real.sin (x / 3) + Real.cos (x / 3)

theorem min_positive_period_and_max_value :
  (∃ T > 0, ∀ x, f (x + T) = f x) ∧ ∃ x, f x = sqrt 2 :=
by
  sorry

end min_positive_period_and_max_value_l33_33102


namespace coefficient_x3_in_expansion_l33_33506

theorem coefficient_x3_in_expansion :
  let T (r : ℕ) := Nat.choose 8 r * 2^(8 - r) * (-1)^r * x^(r / 2)
  in (∀ x : ℕ, ∃ r : ℕ, r / 2 = 3 → T(r) = 112) :=
by
  intro x
  use 6
  sorry

end coefficient_x3_in_expansion_l33_33506


namespace probability_C_first_place_l33_33713

theorem probability_C_first_place :
  ∃ (A B C D E : Prop),
  (A > B) ∧ ¬(A = 1) ∧ ¬(B = 1) ∧ ¬(B = 5) →
  P(C = 1) = 1 / 3 :=
by
  assume A B C D E
  assume h1 : A > B
  assume h2 : ¬(A = 1)
  assume h3 : ¬(B = 1)
  assume h4 : ¬(B = 5)
  sorry

end probability_C_first_place_l33_33713


namespace numPeopleToLeftOfKolya_l33_33521

-- Definitions based on the conditions.
def peopleToRightOfKolya := 12
def peopleToLeftOfSasha := 20
def peopleToRightOfSasha := 8

-- Theorem statement with the given conditions and conclusion.
theorem numPeopleToLeftOfKolya 
  (h1 : peopleToRightOfKolya = 12)
  (h2 : peopleToLeftOfSasha = 20)
  (h3 : peopleToRightOfSasha = 8) :
  ∃ n, n = 16 :=
by
  -- Proving the theorem will be done here.
  sorry

end numPeopleToLeftOfKolya_l33_33521


namespace num_ways_to_choose_starters_l33_33931

theorem num_ways_to_choose_starters :
  let players := 16
  let triplets := 3
  let twins := 2
  let others := players - triplets
  let spots_after_triplets := 7 - triplets
  let scenario1_pairs_left := spots_after_triplets - twins
  let scenario2_spots := spots_after_triplets
  (Nat.choose (others - twins) scenario1_pairs_left + 
   Nat.choose (others - twins) scenario2_spots = 385) :=
by
  sorry

end num_ways_to_choose_starters_l33_33931


namespace sum_of_vars_l33_33764

theorem sum_of_vars 
  (x y z : ℝ) 
  (h1 : x + y = 4) 
  (h2 : y + z = 6) 
  (h3 : z + x = 8) : 
  x + y + z = 9 := 
by 
  sorry

end sum_of_vars_l33_33764


namespace largest_possible_n_l33_33502

open Nat

theorem largest_possible_n (n : ℕ) (x : Fin n.succ → ℕ) 
  (h1 : ∀ k : Fin (n - 2), x k + 2 nd α Term lt i ij.j y =Sin ⇑(.1'j}}+ ∀ j Ef:x_{a:∑ cx_{jk div x k + 2])+1 n≤ :max( 1000 = x{t :fin-sŅ i 'α x4}.)
  : n ≤ 13 :=
by 
  sorry

end largest_possible_n_l33_33502


namespace min_positive_period_and_max_value_l33_33099

noncomputable def f (x : ℝ) : ℝ := Real.sin (x / 3) + Real.cos (x / 3)

theorem min_positive_period_and_max_value :
  (∃ T > 0, ∀ x, f (x + T) = f x) ∧ ∃ x, f x = sqrt 2 :=
by
  sorry

end min_positive_period_and_max_value_l33_33099


namespace james_neither_crumpled_nor_blurred_l33_33826

theorem james_neither_crumpled_nor_blurred (total_pages : ℕ) (crumple_interval : ℕ) (blur_interval : ℕ):
  total_pages = 42 → crumple_interval = 7 → blur_interval = 3 → 
  let crumpled_pages := total_pages / crumple_interval in
  let blurred_pages := total_pages / blur_interval in
  let both_crumpled_and_blurred_pages := total_pages / Nat.lcm crumple_interval blur_interval in
  total_pages - crumpled_pages - blurred_pages + both_crumpled_and_blurred_pages = 24 :=
by
  intros h1 h2 h3
  -- Definitions based on problem conditions
  let crumpled_pages := total_pages / crumple_interval
  let blurred_pages := total_pages / blur_interval
  let both_crumpled_and_blurred_pages := total_pages / Nat.lcm crumple_interval blur_interval
  rw [h1, h2, h3]
  -- Sorry is used to skip the actual proof steps
  sorry

end james_neither_crumpled_nor_blurred_l33_33826


namespace find_k_l33_33218

def vec := ℝ × ℝ

def e1 : vec := (1, 0)
def e2 : vec := (0, 1)
def AB : vec := (1, -1)
def BC : vec := (3, 2)
def CD (k : ℝ) : vec := (k, 2)

def collinear (u v : vec) : Prop :=
  ∃ λ : ℝ, u = (λ • v)

noncomputable def k_value : ℝ :=
if h : collinear (AB + BC) (CD 8) then 8 else sorry

theorem find_k : k_value = 8 := by
  sorry

end find_k_l33_33218


namespace geese_count_l33_33182

theorem geese_count (initial : ℕ) (flown_away : ℕ) (left : ℕ) 
  (h₁ : initial = 51) (h₂ : flown_away = 28) : 
  left = initial - flown_away → left = 23 := 
by
  sorry

end geese_count_l33_33182


namespace optimal_post_office_minimizes_distance_l33_33244

noncomputable def optimal_post_office_location (n : ℕ) (x : ℕ → ℝ) (h : ∀ i j, i < j → x i ≤ x j) : ℝ :=
if n % 2 = 1 then 
  x ((n + 1) / 2 - 1)
else 
  (x (n / 2 - 1) + x (n / 2)) / 2

theorem optimal_post_office_minimizes_distance (n : ℕ) (x : ℕ → ℝ) (h : ∀ i j, i < j → x i ≤ x j) :
  let t := optimal_post_office_location n x h in
  ∀ t', (∑ i in Finset.range n, |x i - t|) ≤ (∑ i in Finset.range n, |x i - t') :=
sorry

end optimal_post_office_minimizes_distance_l33_33244


namespace velvet_needed_for_box_l33_33480

theorem velvet_needed_for_box :
  let long_side_area := 2 * (8 * 6)
  let short_side_area := 2 * (5 * 6)
  let top_and_bottom_area := 2 * 40
  long_side_area + short_side_area + top_and_bottom_area = 236 := by
{
  let long_side_area := 2 * (8 * 6)
  let short_side_area := 2 * (5 * 6)
  let top_and_bottom_area := 2 * 40
  sorry
}

end velvet_needed_for_box_l33_33480


namespace irrigation_canal_construction_l33_33186

theorem irrigation_canal_construction (x : ℕ) (m : ℕ) :
  (∀ y : ℕ, 1650 = 3 * 1650 / 2 / (y + 30) → y = 60) →
  (∀ n : ℕ, 14 * 90 + (90 + 120) * (14 - n) = 1650 → n = 5) →
  x = 60 ∧ (x + 30) = 90 ∧ 90 * 5 + (90 + 120) * 9 = 2340 :=
begin
  intros H1 H2,
  split,
  { exact H1 x },
  {
    split,
    { exact (H1 x) + 30 },
    { exact H2 5 },
  }
end

end irrigation_canal_construction_l33_33186


namespace min_period_and_max_value_l33_33121

def f (x : ℝ) := Real.sin (x / 3) + Real.cos (x / 3)

theorem min_period_and_max_value :
  (∀ T > 0, T ≠ 6 * Real.pi → ¬∀ x : ℝ, f(x + T) = f(x)) ∧
  ∃ x : ℝ, f(x) = sqrt 2 :=
by
  sorry

end min_period_and_max_value_l33_33121


namespace sin_cos_identity_l33_33333

variable {α : ℝ}

/-- Given 1 / sin(α) + 1 / cos(α) = √3, then sin(α) * cos(α) = -1 / 3 -/
theorem sin_cos_identity (h : 1 / Real.sin α + 1 / Real.cos α = Real.sqrt 3) : 
  Real.sin α * Real.cos α = -1 / 3 := 
sorry

end sin_cos_identity_l33_33333


namespace money_for_orders_correct_l33_33951

def operation_cost : ℝ := 4000
def employee_salary_fraction : ℝ := 2 / 5
def delivery_cost_fraction : ℝ := 1 / 4

-- Define the remaining amount after paying employees' salary
def remaining_amount := operation_cost * (1 - employee_salary_fraction)

-- Define the total delivery cost
def delivery_cost := remaining_amount * delivery_cost_fraction

-- Calculate how much money is paid for the orders done
def money_paid_for_orders := operation_cost - (operation_cost * employee_salary_fraction + delivery_cost)

theorem money_for_orders_correct :
  money_paid_for_orders = 1800 :=
by
  sorry

end money_for_orders_correct_l33_33951


namespace axis_of_symmetry_l33_33383

noncomputable def f (x : ℝ) : ℝ :=
  (Real.cos (x + (Real.pi / 2))) * (Real.cos (x + (Real.pi / 4)))

theorem axis_of_symmetry : 
  ∃ (a : ℝ), a = 5 * Real.pi / 8 ∧ ∀ x : ℝ, f (2 * a - x) = f x := 
by
  sorry

end axis_of_symmetry_l33_33383


namespace pool_depth_l33_33718

theorem pool_depth 
  (length width : ℝ) 
  (chlorine_per_120_cubic_feet chlorine_cost : ℝ) 
  (total_spent volume_per_quart_of_chlorine : ℝ) 
  (H_length : length = 10) 
  (H_width : width = 8)
  (H_chlorine_per_120_cubic_feet : chlorine_per_120_cubic_feet = 1 / 120)
  (H_chlorine_cost : chlorine_cost = 3)
  (H_total_spent : total_spent = 12)
  (H_volume_per_quart_of_chlorine : volume_per_quart_of_chlorine = 120) :
  ∃ depth : ℝ, total_spent / chlorine_cost * volume_per_quart_of_chlorine = length * width * depth ∧ depth = 6 :=
by 
  sorry

end pool_depth_l33_33718


namespace Hamilton_marching_band_members_l33_33939

theorem Hamilton_marching_band_members (m : ℤ) (k : ℤ) :
  30 * m ≡ 5 [ZMOD 31] ∧ m = 26 + 31 * k ∧ 30 * m < 1500 → 30 * m = 780 :=
by
  intro h
  have hmod : 30 * m ≡ 5 [ZMOD 31] := h.left
  have m_eq : m = 26 + 31 * k := h.right.left
  have hlt : 30 * m < 1500 := h.right.right
  sorry

end Hamilton_marching_band_members_l33_33939


namespace station_distances_station_distances_a_42_l33_33567

-- Define the problem conditions and parameter
variables (x y z a : ℝ)
axioms
  (h1 : x + y = 3 * z)
  (h2 : z + y = x + a)
  (h3 : x + z = 60)

-- The main theorem we want to prove
theorem station_distances (a : ℝ) (h_a : 0 < a ∧ a < 60) :
  ∃ (x y z : ℝ), x + y = 3 * z ∧ z + y = x + a ∧ x + z = 60 :=
sorry

-- Additional theorem for a = 42
theorem station_distances_a_42 :
  ∃ (x y z : ℝ), x + y = 3 * z ∧ z + y = x + 42 ∧ x + z = 60 :=
sorry

end station_distances_station_distances_a_42_l33_33567


namespace complement_intersection_l33_33397

open Set

variable {U : Set ℝ} (A B : Set ℝ)

def A_def : Set ℝ := { x | 3 ≤ x ∧ x < 7 }
def B_def : Set ℝ := { x | 2 < x ∧ x < 10 }

theorem complement_intersection :
  (U = univ ∧ A = A_def ∧ B = B_def) →
  (compl (A ∩ B) = {x | x < 3 ∨ x ≥ 7}) :=
by
  sorry

end complement_intersection_l33_33397


namespace angle_equality_of_folded_equilateral_triangle_l33_33647

namespace EquilateralTriangleFolding

-- Define the equilateral triangle
variables {A B C M : Type} [EuclideanGeometry]

-- Define the properties of the equilateral triangle and the folding
axiom equilateral_triangle (ABC : triangle A B C) : 
  angle A B C = 60 ∧ angle B C A = 60 ∧ angle C A B = 60

axiom midpoint (BC : segment B C) (M : point) : Midpoint M B C

axiom fold_vertex (A : point) (M : point) : A.folds_to M

-- Statement of the theorem
theorem angle_equality_of_folded_equilateral_triangle 
  (ABC : triangle A B C) (BC : segment B C) (M : point)
  (H1 : equilateral_triangle ABC) (H2 : midpoint BC M) (H3 : fold_vertex A M) : 
  angles_equal (AMC : triangle A M C) (ABM : triangle A B M) := sorry

end EquilateralTriangleFolding

end angle_equality_of_folded_equilateral_triangle_l33_33647


namespace meeting_time_l33_33963

def initial_speed1 : ℝ := 8
def initial_speed2 : ℝ := 6
def decay_rate1 : ℝ := 1
def decay_rate2 : ℝ := 0.5
def distance_between_cities : ℝ := 60

theorem meeting_time :
  ∃ t : ℝ, (∫ t in (0:ℝ) .. t, (initial_speed1 - t)) + (∫ t in (0:ℝ) .. t, (initial_speed2 - 0.5 * t)) = distance_between_cities ∧ t ≈ 6.67 := sorry

end meeting_time_l33_33963


namespace rachelle_meat_requirement_l33_33881

noncomputable def total_initial_meat_required (meat_per_10_hamburgers : ℝ) (hamburger_count : ℕ) (wastage_rate : ℝ) : ℝ :=
let meat_per_hamburger := meat_per_10_hamburgers / 10 in
let effective_meat_per_hamburger := meat_per_hamburger * (1 + wastage_rate) in
effective_meat_per_hamburger * hamburger_count

theorem rachelle_meat_requirement
  (meat_per_10_hamburgers : ℝ)
  (hamburger_count : ℕ)
  (wastage_rate : ℝ)
  (h1 : meat_per_10_hamburgers = 5)
  (h2 : hamburger_count = 30)
  (h3 : wastage_rate = 0.1) :
  total_initial_meat_required meat_per_10_hamburgers hamburger_count wastage_rate = 16.5 := by
  sorry

end rachelle_meat_requirement_l33_33881


namespace tetrahedron_ratio_range_proof_l33_33195

noncomputable def tetrahedron_ratio_range (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : set ℝ :=
{v | ∃ (T : tetrahedron), (T.edges.length = a ∨ T.edges.length = b) ∧ v = a / b}

theorem tetrahedron_ratio_range_proof (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  tetrahedron_ratio_range a b h1 h2 = set.Ioo 0 ((sqrt 6 + sqrt 2) / 2) := sorry

end tetrahedron_ratio_range_proof_l33_33195


namespace seating_arrangement_l33_33869

/--
Prove that the number of ways for Xiao Zhang's son, daughter, and any one other person
to sit together in a row of 6 people (Xiao Zhang, his son, daughter, father, mother,
and younger brother) is 216.
-/
theorem seating_arrangement :
  let arrangements := 6 * 3 * 12 in
  arrangements = 216 :=
by
  simp [arrangements]
  sorry

end seating_arrangement_l33_33869


namespace correct_equation_for_gift_exchanges_l33_33276

theorem correct_equation_for_gift_exchanges
  (x : ℕ)
  (H : (x * (x - 1)) = 56) :
  x * (x - 1) = 56 := 
by 
  exact H

end correct_equation_for_gift_exchanges_l33_33276


namespace negation_proposition_l33_33781

theorem negation_proposition (p : Prop) : 
  (∀ x : ℝ, 2 * x^2 - 1 > 0) ↔ ¬ (∃ x : ℝ, 2 * x^2 - 1 ≤ 0) :=
by
  sorry

end negation_proposition_l33_33781


namespace min_period_and_max_value_l33_33061

def f (x : ℝ) : ℝ := Real.sin (x / 3) + Real.cos (x / 3)

noncomputable def minimum_positive_period (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x : ℝ, f (x + T) = f x ∧ T > 0 ∧ ∀ T' : ℝ, (T' > 0 ∧ ∀ x : ℝ, f (x + T') = f x) → T' ≥ T

noncomputable def is_max_value (f : ℝ → ℝ) (M : ℝ) : Prop :=
  ∀ x : ℝ, f x ≤ M ∧ ∃ x : ℝ, f x = M

theorem min_period_and_max_value :
  minimum_positive_period f (6 * Real.pi) ∧ is_max_value f (Real.sqrt 2) := by
  sorry

end min_period_and_max_value_l33_33061


namespace domain_of_sqrt_log2_l33_33704

theorem domain_of_sqrt_log2 (x : ℝ) : 
  (log 2 (3 - 4 * x) ≥ 0 ∧ 3 - 4 * x > 0) ↔ x ≤ 1 / 2 :=
sorry

end domain_of_sqrt_log2_l33_33704


namespace proof_problem_l33_33727

noncomputable def ellipse_c_equation : Prop :=
  let a := 2 in
  let c := sqrt 3 in
  let b := 1 in
  (a > b > 0) ∧
  (c / a = sqrt 3 / 2) ∧
  (b^2 = a^2 - c^2) ∧
  ( ∀ x y : ℝ, (x^2 / (a^2) + y^2 / (b^2) = 1) ↔ (x^2 / 4 + y^2 = 1))

noncomputable def max_chord_length : Prop :=
  ∀ m : ℝ, (|m| ≥ 1) →
  (∃ AB : ℝ, (|AB| ≤ 2) ∧ 
  ( ∀ l : ℝ, is_tangent l (x^2 + y^2 = 1) ↔
  |AB| ≤ 4 * sqrt 3 * |m| / (m^2 + 3)))

theorem proof_problem : ellipse_c_equation ∧ max_chord_length :=
by {
  sorry
}

end proof_problem_l33_33727


namespace hamburger_varieties_l33_33400

theorem hamburger_varieties : 
  let condiments := 10
  let condiment_combinations := 2^condiments
  let meat_patties := 3
  let bun_types := 2
  in condiment_combinations * meat_patties * bun_types = 6144 :=
by
  sorry

end hamburger_varieties_l33_33400


namespace min_positive_period_and_max_value_l33_33076

def f (x : ℝ) := Real.sin (x / 3) + Real.cos (x / 3)

theorem min_positive_period_and_max_value : 
  (∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T' = T)) 
  ∧ 
  (∀ x, f x ≤ sqrt 2 ∧ (∃ x, f x = sqrt 2)) :=
sorry

end min_positive_period_and_max_value_l33_33076


namespace gift_exchange_equation_l33_33271

theorem gift_exchange_equation
  (x : ℕ)
  (total_gifts : ℕ)
  (H : total_gifts = 56)
  (H1 : 2 * total_gifts = x * (x - 1)) :
  x * (x - 1) = 56 :=
by
  sorry

end gift_exchange_equation_l33_33271


namespace sum_of_log_table_min_value_l33_33858

noncomputable def log_x_div_9 {x i : ℕ} (hx : x > 2) : ℝ := Real.log x / Real.log i - Real.log 9 / Real.log x

theorem sum_of_log_table_min_value :
  ∀ (x : ℕ → ℕ) 
    (hx : ∀ i, 1 ≤ i → i ≤ 200 → x i > 2),
    let val := λ i j, log_x_div_9 (hx j (Nat.succ_pos' j) (by linarith [Nat.succ_le_succ (Nat.succ_le_self j)])) in
    ∑ i in finset.range 200, ∑ j in finset.range 200, val i j = -40000 :=
sorry

end sum_of_log_table_min_value_l33_33858


namespace sector_radius_l33_33650

theorem sector_radius (θ : ℝ) (s : ℝ) (R : ℝ) 
  (hθ : θ = 150)
  (hs : s = (5 / 2) * Real.pi)
  : (θ / 360) * (2 * Real.pi * R) = (5 / 2) * Real.pi → 
  R = 3 := 
sorry

end sector_radius_l33_33650


namespace simplified_fraction_l33_33010

noncomputable def simplify_and_rationalize (a b c d e f : ℝ) : ℝ :=
  (Real.sqrt a / Real.sqrt b) * (Real.sqrt c / Real.sqrt d) * (Real.sqrt e / Real.sqrt f)

theorem simplified_fraction :
  simplify_and_rationalize 3 7 5 9 6 8 = Real.sqrt 35 / 14 :=
by
  sorry

end simplified_fraction_l33_33010


namespace cord_length_before_cut_l33_33641

-- Definitions based on the conditions
def parts_after_cut := 20
def longest_piece := 8
def shortest_piece := 2
def initial_parts := 19

-- Lean statement to prove the length of the cord before it was cut
theorem cord_length_before_cut : 
  (initial_parts * ((longest_piece / 2) + shortest_piece) = 114) :=
by 
  sorry

end cord_length_before_cut_l33_33641


namespace line_equation_general_form_l33_33037

theorem line_equation_general_form (slope : ℝ) (x_intercept : ℝ) 
  (hx : x_intercept = 2) (hs : slope = -3) : 
  ∃ (a b c : ℝ), a = 3 ∧ b = 1 ∧ c = -6 := 
by
  use 3, 1, -6
  split; 
  { simp },
  { simp },
  { simp }
  sorry  -- proof steps are omitted as per instructions

end line_equation_general_form_l33_33037


namespace problem_inequality_solution_set_inequality_proof_l33_33757

def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

theorem problem_inequality_solution_set :
  {x : ℝ | f x < 4} = {x : ℝ | -2 < x ∧ x < 2} :=
sorry

theorem inequality_proof (x y : ℝ) (hx : -2 < x ∧ x < 2) (hy : -2 < y ∧ y < 2) :
  |x + y| < |(x * y) / 2 + 2| :=
sorry

end problem_inequality_solution_set_inequality_proof_l33_33757


namespace bridge_length_is_235_l33_33253

noncomputable def length_of_bridge
  (train_length : ℕ)
  (speed_km_hr : ℕ)
  (crossing_time_sec : ℕ) : ℕ :=
let speed_m_s := (speed_km_hr * 1000) / 3600 in
let total_distance := speed_m_s * crossing_time_sec in
total_distance - train_length

theorem bridge_length_is_235
  (train_length : ℕ := 140)
  (speed_km_hr : ℕ := 45)
  (crossing_time_sec : ℕ := 30) :
  length_of_bridge train_length speed_km_hr crossing_time_sec = 235 :=
by
  unfold length_of_bridge
  sorry

end bridge_length_is_235_l33_33253


namespace find_f_105_5_l33_33367

noncomputable def f : ℝ → ℝ :=
sorry -- Definition of f

-- Hypotheses
axiom even_function (x : ℝ) : f x = f (-x)
axiom functional_equation (x : ℝ) : f (x + 2) = -f x
axiom function_values (x : ℝ) (hx : 2 ≤ x ∧ x ≤ 3) : f x = x

-- Goal
theorem find_f_105_5 : f 105.5 = 2.5 :=
sorry

end find_f_105_5_l33_33367


namespace min_value_expression_l33_33459

variable {a b c k : ℝ}

theorem min_value_expression (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hk : 0 < k) :
  ∃ x, x = (∑ cyc in [(a, b), (b, c), (c, a)], (cyc.1^2) / (k * cyc.2)) ∧ x = 3 / k :=
by
  sorry

end min_value_expression_l33_33459


namespace problem_l33_33395

def I := {1, 2, 3, 4, 5, 6}
def M := {3, 4, 5}
def N := {1, 3, 6}
def C (S : Set ℕ) := I ∖ S

theorem problem :
  {2, 7} = (C M) ∩ (C N) :=
sorry

end problem_l33_33395


namespace pyramid_height_l33_33248

-- The definitions and conditions
def pyramid := Type
def is_right_pyramid (p : pyramid) : Prop := sorry
def is_square_base (p : pyramid) (perimeter : ℝ) : Prop := sorry
def apex_distance (p : pyramid) (distance : ℝ) : Prop := sorry

-- The pyramid in question
def myPyramid : pyramid := sorry
def base_perimeter : ℝ := 40
def apex_distance_to_vertex : ℝ := 15

-- The hypothesis: It's a right pyramid with the described properties
axiom H1 : is_right_pyramid myPyramid
axiom H2 : is_square_base myPyramid base_perimeter
axiom H3 : apex_distance myPyramid apex_distance_to_vertex

-- The target: height of the pyramid from its peak to the center of the square base
def height (p : pyramid) : ℝ := sorry

-- The theorem we need to prove
theorem pyramid_height : height myPyramid = 5 * Real.sqrt 7 :=
by {
  sorry
}

end pyramid_height_l33_33248


namespace caleb_ice_cream_l33_33679

theorem caleb_ice_cream (x : ℕ) (hx1 : ∃ x, x ≥ 0) (hx2 : 4 * x - 36 = 4) : x = 10 :=
by {
  sorry
}

end caleb_ice_cream_l33_33679


namespace composite_2000_digit_number_l33_33348

open Nat

theorem composite_2000_digit_number 
  (digits_seq : List ℕ) 
  (h_digits : ∀ d ∈ [1, 9, 8, 7], ∃ i < 2000, digits_seq.nth i = some d) 
  (h_length : digits_seq.length = 2000) 
  : ∃ m n : ℕ, 1 < m ∧ 1 < n ∧ m * n = digits_seq.foldr (λ d acc => acc + d) 0 := 
sorry

end composite_2000_digit_number_l33_33348


namespace numPeopleToLeftOfKolya_l33_33520

-- Definitions based on the conditions.
def peopleToRightOfKolya := 12
def peopleToLeftOfSasha := 20
def peopleToRightOfSasha := 8

-- Theorem statement with the given conditions and conclusion.
theorem numPeopleToLeftOfKolya 
  (h1 : peopleToRightOfKolya = 12)
  (h2 : peopleToLeftOfSasha = 20)
  (h3 : peopleToRightOfSasha = 8) :
  ∃ n, n = 16 :=
by
  -- Proving the theorem will be done here.
  sorry

end numPeopleToLeftOfKolya_l33_33520


namespace find_range_of_a_l33_33296

-- Define the operation ⊗ on ℝ: x ⊗ y = x(1 - y)
def tensor (x y : ℝ) : ℝ := x * (1 - y)

-- State the inequality condition for all real numbers x
def inequality_condition (a : ℝ) : Prop :=
  ∀ (x : ℝ), tensor (x - a) (x + 1) < 1

-- State the theorem to prove the range of a
theorem find_range_of_a (a : ℝ) (h : inequality_condition a) : -2 < a ∧ a < 2 :=
  sorry

end find_range_of_a_l33_33296


namespace sum_of_all_possible_numbers_l33_33485

def digits := {2, 0, 1, 8} : Finset ℕ

-- A function to convert a list of digits to a number
def to_number (ds : List ℕ) : ℕ :=
  ds.foldl (λ acc d => acc * 10 + d) 0

-- A function to generate all possible numbers from the digits 
def generate_numbers : Finset ℕ :=
  (Finset.univ : Finset (Finset ℕ)).filter (λ s => s ⊆ digits ∧ s.card > 0).bUnion (λ s =>
    (s.powerset.filter (λ t => t.card = s.card)).map to_number)

-- The main sum of all generated numbers
def naturals_sum : ℕ :=
  generate_numbers.sum id

theorem sum_of_all_possible_numbers : naturals_sum = 78331 := by
  sorry

end sum_of_all_possible_numbers_l33_33485


namespace solve_system_of_equations_l33_33012

theorem solve_system_of_equations :
  ∀ (x : Fin 50 → ℝ),
    (1 - x 0 * x 1 * x 2 = 0) →
    (1 + x 1 * x 2 * x 3 = 0) →
    (1 - x 2 * x 3 * x 4 = 0) →
    (1 + x 3 * x 4 * x 5 = 0) →
    -- Similar equations go here up until
    (1 - x 46 * x 47 * x 48 = 0) →
    (1 + x 47 * x 48  * x 49 = 0) →
    (1 - x 48 * x 49 * x 0 = 0) →
    (1 + x 49 * x 0 * x 1 = 0) →
    (∀ i : Fin 50, x i = -1 ↔ i % 2 = 0) ∧ (∀ i : Fin 50, x i = 1 ↔ i % 2 = 1) :=
by
  intros
  split; intros; sorry

end solve_system_of_equations_l33_33012


namespace problem_l33_33366

noncomputable def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
noncomputable def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

variable (f g : ℝ → ℝ)
variable (h₁ : is_odd f)
variable (h₂ : is_even g)
variable (h₃ : ∀ x, f x - g x = 2 * x^3 + x^2 + 3)

theorem problem : f 2 + g 2 = 9 :=
by sorry

end problem_l33_33366


namespace determine_n_l33_33297

noncomputable def is_rational (x : ℝ) : Prop := ∃ (a b : ℤ), x = a / b ∧ b ≠ 0

theorem determine_n (n : ℕ) (h₁ : n ≥ 2) (a : ℝ)
  (h₂ : is_rational (a + real.sqrt 2))
  (h₃ : is_rational (a^n + real.sqrt 2)) : n = 2 :=
sorry

end determine_n_l33_33297


namespace rest_days_in_1000_days_l33_33294

def schedule := ℕ → bool

def craig_schedule (d : ℕ) : bool :=
  if d % 6 = 4 ∨ d % 6 = 5 then true else false

def dana_schedule (d : ℕ) : bool :=
  if d % 6 = 5 then true else false

def coinciding_rest_days (n : ℕ) : ℕ :=
  (list.range n).filter (λ d, craig_schedule d = true ∧ dana_schedule d = true).length

theorem rest_days_in_1000_days :
  coinciding_rest_days 1000 = 166 := 
sorry

end rest_days_in_1000_days_l33_33294


namespace woman_wait_time_for_man_to_catch_up_l33_33646

theorem woman_wait_time_for_man_to_catch_up :
  ∀ (mans_speed womans_speed : ℕ) (time_after_passing : ℕ) (distance_up_slope : ℕ) (incline_percentage : ℕ),
  mans_speed = 5 →
  womans_speed = 25 →
  time_after_passing = 5 →
  distance_up_slope = 1 →
  incline_percentage = 5 →
  max 0 (mans_speed - incline_percentage * 1) = 0 →
  time_after_passing = 0 :=
by
  intros
  -- Insert proof here when needed
  sorry

end woman_wait_time_for_man_to_catch_up_l33_33646


namespace min_positive_period_and_max_value_l33_33101

noncomputable def f (x : ℝ) : ℝ := Real.sin (x / 3) + Real.cos (x / 3)

theorem min_positive_period_and_max_value :
  (∃ T > 0, ∀ x, f (x + T) = f x) ∧ ∃ x, f x = sqrt 2 :=
by
  sorry

end min_positive_period_and_max_value_l33_33101


namespace Danielle_age_l33_33258

-- Define the ages of Anna, Ben, Carlos, and Danielle
variables (A B C D : ℕ)

-- State the conditions
def condition1 : Prop := A = B - 4
def condition2 : Prop := B = C + 3
def condition3 : Prop := D = C + 6
def condition4 : Prop := A = 15

-- The proof goal
theorem Danielle_age (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) : D = 22 :=
by
  sorry

-- Ensuring conditions are used only from the problem statement
#check Danielle_age

end Danielle_age_l33_33258


namespace nonempty_subsets_card_l33_33165

open Set

theorem nonempty_subsets_card :
  { A : Set ℕ | ∅ ⊂ A ∧ A ⊆ {1, 2, 3} }.to_finset.card = 7 := 
sorry

end nonempty_subsets_card_l33_33165


namespace option_C_forms_a_set_l33_33982

def elements_are_definite_unordered_distinct := 
  ∀ (S : Set α) (x y : α), (x ∈ S ∧ y ∈ S) → (x ≠ y → unordered S)

def famous_TV_hosts_of_CCTV : Prop := ¬elements_are_definite_unordered_distinct {x ∈ α | Famous x}
def fastest_cars_in_city : Prop := ¬elements_are_definite_unordered_distinct {x ∈ α | FastestCar x}
def middle_school_students_in_Zhengyang_County : Prop := elements_are_definite_unordered_distinct {x ∈ α | MiddleSchoolStudent x}
def tall_buildings_in_Zhengyang : Prop := ¬elements_are_definite_unordered_distinct {x ∈ α | TallBuilding x}

theorem option_C_forms_a_set (h : elements_are_definite_unordered_distinct) : 
  middle_school_students_in_Zhengyang_County :=
by 
  sorry

end option_C_forms_a_set_l33_33982


namespace min_period_max_value_f_l33_33140

def f (x : ℝ) : ℝ := sin (x / 3) + cos (x / 3)

theorem min_period_max_value_f :
  ∃ T M, (∀ x, f (x + T) = f x) ∧ 0 < T ∧
    0 < M ∧ (∀ x, f x ≤ M) ∧ (∃ x, f x = M) ∧ 
    T = 6 * π ∧ M = √2 := sorry

end min_period_max_value_f_l33_33140


namespace circles_intersect_l33_33929

open set real

def circle1_eq (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 16*y - 48 = 0
def circle2_eq (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 8*y - 44 = 0

theorem circles_intersect : ∃ (x y : ℝ), circle1_eq x y ∧ circle2_eq x y :=
sorry

end circles_intersect_l33_33929


namespace heights_on_equal_sides_are_equal_l33_33616

-- Given conditions as definitions
def is_isosceles_triangle (a b c : ℝ) := (a = b ∨ b = c ∨ c = a)
def height_on_equal_sides_equal (a b c : ℝ) := is_isosceles_triangle a b c → a = b

-- Lean theorem statement to prove
theorem heights_on_equal_sides_are_equal {a b c : ℝ} : is_isosceles_triangle a b c → height_on_equal_sides_equal a b c := 
sorry

end heights_on_equal_sides_are_equal_l33_33616


namespace min_positive_period_and_max_value_l33_33108

noncomputable def f (x : ℝ) : ℝ := Real.sin (x / 3) + Real.cos (x / 3)

theorem min_positive_period_and_max_value :
  (∃ T > 0, ∀ x, f (x + T) = f x) ∧ ∃ x, f x = sqrt 2 :=
by
  sorry

end min_positive_period_and_max_value_l33_33108


namespace some_students_not_club_members_l33_33260

open Classical

variable (Student : Type) (ClubMember : Type)

-- Condition 1
variable (studies_science : Student → Prop)
variable (not_studies_science : ∃ (s : Student), ¬ studies_science s)

-- Condition 2
variable (is_honest : ClubMember → Prop)
variable (clubmember_studies_science : ∀ (cm : ClubMember), studies_science (cm : Student) ∧ is_honest cm)

-- The conclusion to be proved
theorem some_students_not_club_members : ∃ (s : Student), ¬ (∃ (cm : ClubMember), cm = (s : ClubMember) ∧ studies_science s) :=
by
  sorry

end some_students_not_club_members_l33_33260


namespace closest_integer_to_sqrt6_l33_33873

theorem closest_integer_to_sqrt6 : 
  forall (x : ℝ), 2 < x ∧ x < 2.5 -> (∃ n : ℤ, n = 2 ∧ abs x - 2 <= abs (x - m) for all m ≠ 2) := 
by {
  sorry
}

end closest_integer_to_sqrt6_l33_33873


namespace domain_of_function_l33_33705

theorem domain_of_function :
  { x : ℝ // (6 - x - x^2) > 0 } = { x : ℝ // -3 < x ∧ x < 2 } :=
by
  sorry

end domain_of_function_l33_33705


namespace number_of_common_tangents_l33_33519

def parametric_curve_C1 (θ : ℝ) : ℝ × ℝ :=
  (3 * Real.cos θ, 3 * Real.sin θ)

def polar_curve_C2 (ρ θ : ℝ) : Prop :=
  ρ^2 + 6 * Real.cos θ - 2 * ρ * Real.sin θ + 6 = 0

theorem number_of_common_tangents :
  (∀ θ : ℝ, ∃ ρ : ℝ, polar_curve_C2 ρ θ) →
  ∃ n : ℕ, n = 2 :=
sorry

end number_of_common_tangents_l33_33519


namespace length_of_PC_l33_33790

theorem length_of_PC (AB BC CA : ℝ) (hAB : AB = 10) (hBC : BC = 9) (hCA : CA = 7)
  (hSim : similar (triangle PAB) (triangle PCA)) : PC = 1.5 :=
by
  sorry

end length_of_PC_l33_33790


namespace triangle_area_leq_one_sixth_l33_33332

theorem triangle_area_leq_one_sixth (h : real) (points : set (real × real)) (h_points : ∥points∥ = 7) 
  (hexagon : set (real × real)) (h_hex : is_regular_hexagon hexagon ∧ area hexagon = h)
  (points_in_hex : ∀ p ∈ points, p ∈ hexagon) :
  ∃ (p1 p2 p3 : (real × real)), p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧ 
    triangle_area p1 p2 p3 ≤ h / 6 :=
begin
  sorry
end

end triangle_area_leq_one_sixth_l33_33332


namespace complex_division_l33_33712

theorem complex_division : (2 - complex.i) / (1 + 2 * complex.i) = -complex.i := by
  sorry

end complex_division_l33_33712


namespace x_intercept_is_correct_l33_33601

-- Define the original line equation
def original_line (x y : ℝ) : Prop := 4 * x + 5 * y = 10

-- Define the perpendicular line's y-intercept
def y_intercept (y : ℝ) : Prop := y = -3

-- Define the equation of the perpendicular line in slope-intercept form
def perpendicular_line (x y : ℝ) : Prop := y = (5 / 4) * x + -3

-- Prove that the x-intercept of the perpendicular line is 12/5
theorem x_intercept_is_correct : ∃ x : ℝ, x ≠ 0 ∧ (∃ y : ℝ, y = 0) ∧ (perpendicular_line x y) :=
sorry

end x_intercept_is_correct_l33_33601


namespace probability_at_least_one_eight_l33_33955

theorem probability_at_least_one_eight :
  let total_outcomes := 64 in
  let outcomes_without_8 := 49 in
  let favorable_outcomes := total_outcomes - outcomes_without_8 in
  let probability := (favorable_outcomes : ℚ) / total_outcomes in
  probability = 15 / 64 :=
by
  let total_outcomes := 64
  let outcomes_without_8 := 49
  let favorable_outcomes := total_outcomes - outcomes_without_8
  let probability := (favorable_outcomes : ℚ) / total_outcomes
  show probability = 15 / 64
  sorry

end probability_at_least_one_eight_l33_33955


namespace min_period_and_max_value_l33_33055

def f (x : ℝ) : ℝ := Real.sin (x / 3) + Real.cos (x / 3)

noncomputable def minimum_positive_period (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x : ℝ, f (x + T) = f x ∧ T > 0 ∧ ∀ T' : ℝ, (T' > 0 ∧ ∀ x : ℝ, f (x + T') = f x) → T' ≥ T

noncomputable def is_max_value (f : ℝ → ℝ) (M : ℝ) : Prop :=
  ∀ x : ℝ, f x ≤ M ∧ ∃ x : ℝ, f x = M

theorem min_period_and_max_value :
  minimum_positive_period f (6 * Real.pi) ∧ is_max_value f (Real.sqrt 2) := by
  sorry

end min_period_and_max_value_l33_33055


namespace difference_max_min_OP_l33_33190

theorem difference_max_min_OP (A B C O P : ℂ) (h₁ : is_equilateral_triangle A B C)
  (h₂ : circumcenter A B C O) (h₃ : complex.abs ((P - A) * (P - B) * (P - C)) = 7) : 
  complex.abs (P - O) = 2 - complex.abs (complex.root3 6) :=
sorry

end difference_max_min_OP_l33_33190


namespace min_positive_period_and_max_value_l33_33077

def f (x : ℝ) := Real.sin (x / 3) + Real.cos (x / 3)

theorem min_positive_period_and_max_value : 
  (∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T' = T)) 
  ∧ 
  (∀ x, f x ≤ sqrt 2 ∧ (∃ x, f x = sqrt 2)) :=
sorry

end min_positive_period_and_max_value_l33_33077


namespace custom_op_difference_l33_33410

def custom_op (x y : ℕ) : ℕ := x * y - (x + y)

theorem custom_op_difference : custom_op 7 4 - custom_op 4 7 = 0 :=
by
  sorry

end custom_op_difference_l33_33410


namespace pizza_cost_is_correct_l33_33489

noncomputable def total_pizza_cost : ℝ :=
  let triple_cheese_pizza_cost := (3 * 10) + (6 * 2 * 2.5)
  let meat_lovers_pizza_cost := (3 * 8) + (4 * 3 * 2.5)
  let veggie_delight_pizza_cost := (6 * 5) + (10 * 1 * 2.5)
  triple_cheese_pizza_cost + meat_lovers_pizza_cost + veggie_delight_pizza_cost

theorem pizza_cost_is_correct : total_pizza_cost = 169 := by
  sorry

end pizza_cost_is_correct_l33_33489


namespace product_of_cd_l33_33907

theorem product_of_cd : 
  ∃ (c d : ℂ), 
    ∀ (z : ℂ), 
    (c * z + d * conj(z) = k) → c * d = 29 :=
by 
  let u := -3 + 4 * complex.I
  let v := 2 + 2 * complex.I
  let k := -2
  let c := 2 - 5 * complex.I
  let d := 2 + 5 * complex.I
  use [c, d]
  intro z
  sorry

end product_of_cd_l33_33907


namespace cows_eat_husk_l33_33796

theorem cows_eat_husk :
  ∀ (cows : ℕ) (days : ℕ) (husk_per_cow : ℕ),
    cows = 45 →
    days = 45 →
    husk_per_cow = 1 →
    (cows * husk_per_cow = 45) :=
by
  intros cows days husk_per_cow h_cows h_days h_husk_per_cow
  sorry

end cows_eat_husk_l33_33796


namespace dot_product_vec1_vec2_l33_33682

-- Define the vectors
def vec1 := (⟨-4, -1⟩ : ℤ × ℤ)
def vec2 := (⟨6, 8⟩ : ℤ × ℤ)

-- Define the dot product function
def dot_product (v1 v2 : ℤ × ℤ) : ℤ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Prove that the dot product of vec1 and vec2 is -32
theorem dot_product_vec1_vec2 : dot_product vec1 vec2 = -32 :=
by
  sorry

end dot_product_vec1_vec2_l33_33682


namespace mid_base_is_half_sum_l33_33482

variables {A B C D E : Type*} [point_space A]
variables [segment_space B] [segment_space C] 
variables (AD BC : B) (ABCDE : trapezoid AD BC E)

-- Given: Trapezoid ABCD with bases AD and BC, and point E on AD
-- Where the perimeters of triangles ABE, BCE, and CDE are equal.
axiom perimeters_equal (ABE BCE CDE : triangle E) :
  (ABE.perimeter = BCE.perimeter) ∧ (BCE.perimeter = CDE.perimeter)

-- Prove that BC = AD / 2
theorem mid_base_is_half_sum (h : perimeters_equal) : 
  BC = AD / 2 := 
sorry

end mid_base_is_half_sum_l33_33482


namespace min_positive_period_max_value_l33_33117
open Real

noncomputable def f (x : ℝ) : ℝ := sin (x / 3) + cos (x / 3)

theorem min_positive_period_max_value :
  (∃ T > 0, ∀ x, f(x + T) = f(x) ∧ T = 6 * π) ∧
  (∀ x, f(x) ≤ sqrt 2) ∧ 
  (∃ x, f(x) = sqrt 2) :=
by
  sorry

end min_positive_period_max_value_l33_33117


namespace right_trapezoid_base_ratio_l33_33798

variables (h a b : ℝ) -- h: altitude, a: smaller base, b: larger base

noncomputable def ratio_of_bases (a b : ℝ) : ℝ := a / b

theorem right_trapezoid_base_ratio (h a b : ℝ) (H1 : b = 4 * h)
  (H2 : 1 / 2 * (a + b) * h = 2 * h^2) : ratio_of_bases a b = 1 :=
begin
  -- Conditions given in the problem
  -- 1. b = 4 * h
  -- 2. 1/2 * (a + b) * h = 2 * h^2
  sorry
end

end right_trapezoid_base_ratio_l33_33798


namespace tangent_condition_for_k_l33_33629

-- Definitions for the line and circle equations
def line_eq (k : ℝ) (x y : ℝ) := k * x - y - 3 * real.sqrt 2 = 0
def circle_eq (x y : ℝ) := x * x + y * y = 9

-- Tangency condition function
def is_tangent (k : ℝ) : Prop :=
  ∃ x y, line_eq k x y ∧ circle_eq x y

-- Main statement: k = 1 is a sufficient but not necessary condition 
-- for the line to be tangent to the circle
theorem tangent_condition_for_k : is_tangent 1 ∧ (∀ k, is_tangent k → k = 1 ∨ k = -1) :=
begin
  sorry,
end

end tangent_condition_for_k_l33_33629


namespace sum_of_prime_factors_1320_l33_33981

theorem sum_of_prime_factors_1320 : 
  let smallest_prime := 2
  let largest_prime := 11
  smallest_prime + largest_prime = 13 :=
by
  sorry

end sum_of_prime_factors_1320_l33_33981


namespace ratio_CP_PA_l33_33812

variables {A B C D M P : Type}
variables [IsTriangle ABC : triangle (point A) (point B) (point C)]
variables (AB AC : ℝ) (D : point BC) (M : midpoint (segment A D)) (P : intersection (segment AC) (line BM))

theorem ratio_CP_PA (hAB : AB = 15) (hAC : AC = 18) (hD_on_BC : segment (D on BC))
  (hMidpoint_M : segment (M midpoint AD)) (hIntersection_P : intersection (P on AC) (line BM)) :
  m + n = 16 :=
  sorry

end ratio_CP_PA_l33_33812


namespace log_inequality_solution_set_l33_33551

theorem log_inequality_solution_set (x : ℝ) (hx : 2 ≤ x ∧ x < 4) : log 2 (x^3) + 2 > 0 :=
sorry

end log_inequality_solution_set_l33_33551


namespace lines_intersect_l33_33192

theorem lines_intersect (m b : ℝ) (h1 : 17 = 2 * m * 4 + 5) (h2 : 17 = 4 * 4 + b) : b + m = 2.5 :=
by {
    sorry
}

end lines_intersect_l33_33192


namespace least_possible_area_of_square_l33_33933

theorem least_possible_area_of_square :
  (∃ (side_length : ℝ), 3.5 ≤ side_length ∧ side_length < 4.5 ∧ 
    (∃ (area : ℝ), area = side_length * side_length ∧ 
    (∀ (side : ℝ), 3.5 ≤ side ∧ side < 4.5 → side * side ≥ 12.25))) :=
sorry

end least_possible_area_of_square_l33_33933


namespace joe_monthly_taxes_l33_33636

theorem joe_monthly_taxes (income joe_tax_rate : ℝ) (h_income : income = 2120) (h_tax_rate : joe_tax_rate = 0.4) : income * joe_tax_rate = 848 := by
  rw [h_income, h_tax_rate]
  norm_num
  sorry

end joe_monthly_taxes_l33_33636


namespace carrots_weight_l33_33818

theorem carrots_weight (carrots_bed1: ℕ) (carrots_bed2: ℕ) (carrots_bed3: ℕ) (carrots_per_pound: ℕ)
  (h_bed1: carrots_bed1 = 55)
  (h_bed2: carrots_bed2 = 101)
  (h_bed3: carrots_bed3 = 78)
  (h_c_per_p: carrots_per_pound = 6) :
  (carrots_bed1 + carrots_bed2 + carrots_bed3) / carrots_per_pound = 39 := by
  sorry

end carrots_weight_l33_33818


namespace smallest_number_gt_1992_with_rem_7_div_9_l33_33201

theorem smallest_number_gt_1992_with_rem_7_div_9 :
  ∃ n > 1992, n % 9 = 7 ∧ ∀ m, m > 1992 ∧ m % 9 = 7 → n ≤ m :=
begin
  use 1996,
  split,
  { linarith },
  split,
  { norm_num },
  { intros m hm1 hm2,
    have : m % 9 = 7 := hm2,
    obtain ⟨k, hk⟩ := nat.mod_add_div m 9,
    rw [hk] at this,
    nth_rewrite 1 [←this],
    replace hm2 := nat.le_of_sub_eq_zero (nat.sub_eq_zero_of_le (nat.mod_add_div m 9).symm.ge),
    linarith,
    sorry
  }
end

end smallest_number_gt_1992_with_rem_7_div_9_l33_33201


namespace students_arrangement_l33_33889

-- Given conditions
variables (students : Fin 6)
variable (A B C : students)
variable (AB_group : students × students)
variable (others : Fin 3)

-- Theorem to be proved
theorem students_arrangement :
  let AB_group_standing := 2 -- Two ways to arrange A and B internally.
  let others_standing := factorial 3 -- Three other students can be arranged in 3! = 6 ways.
  let valid_positions_C := 12 -- Considering insertion of C into valid positions.
  AB_group_standing * others_standing * valid_positions_C = 144 :=
sorry

end students_arrangement_l33_33889


namespace min_pos_period_max_value_l33_33142

noncomputable def f (x : ℝ) := Real.sin (x / 3) + Real.cos (x / 3)

theorem min_pos_period_max_value :
  (∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ T) ∧ T = 6 * Real.pi) ∧ 
  (∀ x, f x ≤ sqrt 2 ∧ (∃ x, f x = sqrt 2)) :=
by
  sorry

end min_pos_period_max_value_l33_33142


namespace min_positive_period_max_value_l33_33110
open Real

noncomputable def f (x : ℝ) : ℝ := sin (x / 3) + cos (x / 3)

theorem min_positive_period_max_value :
  (∃ T > 0, ∀ x, f(x + T) = f(x) ∧ T = 6 * π) ∧
  (∀ x, f(x) ≤ sqrt 2) ∧ 
  (∃ x, f(x) = sqrt 2) :=
by
  sorry

end min_positive_period_max_value_l33_33110


namespace min_period_of_f_max_value_of_f_l33_33155

def f : ℝ → ℝ := λ x, Real.sin (x / 3) + Real.cos (x / 3)

theorem min_period_of_f : ∃ T > 0, (∀ x, f (x + T) = f x) ∧ T = 6 * Real.pi :=
by
  sorry

theorem max_value_of_f : ∃ x, f x = Real.sqrt 2 :=
by
  sorry

end min_period_of_f_max_value_of_f_l33_33155


namespace estimated_probability_l33_33742

def num_groups : ℕ := 20
def random_groups : list (list ℕ) :=
  [ [9, 0, 7, 5], [9, 6, 6, 0], [1, 9, 1, 8], [9, 2, 5, 7],
    [2, 7, 1, 6], [9, 3, 2, 5], [8, 1, 2, 1], [4, 5, 8, 9],
    [5, 6, 9, 0], [6, 8, 3, 2], [4, 3, 1, 5], [2, 5, 7, 3],
    [3, 9, 3, 7], [9, 2, 7, 9], [5, 5, 6, 3], [4, 8, 8, 2],
    [7, 3, 5, 8], [1, 1, 3, 5], [1, 5, 8, 7], [4, 9, 8, 9] ]

def success_count (group: list ℕ) : ℕ :=
  group.countp (λ x => x ≤ 4)

def is_target_group (group: list ℕ) : Prop :=
  success_count group = 2

theorem estimated_probability : 
  (real.of_rat (list.countp is_target_group random_groups : ℚ) / real.of_rat num_groups) = 0.35 := by sorry

end estimated_probability_l33_33742


namespace correct_addition_result_l33_33617

theorem correct_addition_result (x : ℚ) (h : x - 13/5 = 9/7) : x + 13/5 = 227/35 := 
by sorry

end correct_addition_result_l33_33617


namespace dot_product_a_b_l33_33334

noncomputable def vector_a : ℝ × ℝ :=
  (2 * real.sin (real.pi / 180 * 13), 2 * real.sin (real.pi / 180 * 77))

variable (b : ℝ × ℝ)

theorem dot_product_a_b :
  (euclidean_distance (vector_a) ((vector_a).1 - b.1, (vector_a).2 - b.2) = 1) →
  (real.angle (vector_a) ((vector_a).1 - b.1, (vector_a).2 - b.2) = real.pi / 3) →
  ((vector_a.1, vector_a.2) ⬝ (b.1, b.2) = 3) :=
begin
  sorry
end

end dot_product_a_b_l33_33334


namespace number_of_possible_denominators_l33_33016

def digits (n : Nat) : Prop := n < 10
def repeating_decimal_to_fraction (a b c : Nat) := a * 100 + b * 10 + c

theorem number_of_possible_denominators :
  ∀ a b c : Nat,
    digits a →
    digits b →
    digits c →
    ¬ (a = 9 ∧ b = 9 ∧ c = 9) →
    ∃ denom_set : Set Nat,
      denom_set = {3, 9, 27, 37, 111, 333, 999} ∧
      denom_set.card = 7 :=
by
  sorry

end number_of_possible_denominators_l33_33016


namespace max_positive_integer_difference_l33_33991

theorem max_positive_integer_difference (x y : ℝ) (hx : 4 < x ∧ x < 8) (hy : 8 < y ∧ y < 12) : ∃ d : ℕ, d = 6 :=
by
  sorry

end max_positive_integer_difference_l33_33991


namespace min_pos_period_max_value_l33_33150

noncomputable def f (x : ℝ) := Real.sin (x / 3) + Real.cos (x / 3)

theorem min_pos_period_max_value :
  (∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ T) ∧ T = 6 * Real.pi) ∧ 
  (∀ x, f x ≤ sqrt 2 ∧ (∃ x, f x = sqrt 2)) :=
by
  sorry

end min_pos_period_max_value_l33_33150


namespace max_intersections_l33_33941

theorem max_intersections (P_x : Finset ℝ) (P_y : Finset ℝ)
  (h1 : P_x.card = 10) (h2 : ∀ x ∈ P_x, 0 < x) 
  (h3 : P_y.card = 5) (h4 : ∀ y ∈ P_y, 0 < y) : 
  ∃ (I : ℕ), I = 450 :=
by
  exists 450
  sorry

end max_intersections_l33_33941


namespace sequence_contains_30_l33_33390

theorem sequence_contains_30 :
  ∃ n : ℕ, n * (n + 1) = 30 :=
sorry

end sequence_contains_30_l33_33390


namespace boxes_needed_l33_33570

noncomputable def living_room_length : ℝ := 16
noncomputable def living_room_width : ℝ := 20
noncomputable def sq_ft_per_box : ℝ := 10
noncomputable def already_floored : ℝ := 250

theorem boxes_needed : 
  (living_room_length * living_room_width - already_floored) / sq_ft_per_box = 7 :=
by 
  sorry

end boxes_needed_l33_33570


namespace min_period_of_f_max_value_of_f_l33_33156

def f : ℝ → ℝ := λ x, Real.sin (x / 3) + Real.cos (x / 3)

theorem min_period_of_f : ∃ T > 0, (∀ x, f (x + T) = f x) ∧ T = 6 * Real.pi :=
by
  sorry

theorem max_value_of_f : ∃ x, f x = Real.sqrt 2 :=
by
  sorry

end min_period_of_f_max_value_of_f_l33_33156


namespace area_of_triangle_l33_33442

theorem area_of_triangle {A B C : ℝ} (b c : ℝ) (A_angle : ℝ) 
  (hb : b = 1) 
  (hc : c = sqrt 3) 
  (hA_angle : A_angle = π / 4) : 
  (1 / 2) * b * c * sin A_angle = sqrt 6 / 4 := 
by
  -- proof is omitted
  sorry

end area_of_triangle_l33_33442


namespace beta_largest_possible_success_ratio_l33_33661

noncomputable def alpha_day1_success_ratio : ℚ := 160 / 300
noncomputable def alpha_day2_success_ratio : ℚ := 140 / 200
constant b d : ℕ
constant a c : ℕ
axiom b_d_sum_eq_500 : b + d = 500
axiom daily_success_ratios_less_than_alpha :
  a > 0 ∧ c > 0 ∧ b ≤ 299 ∧ (a : ℚ) / b < alpha_day1_success_ratio ∧ d ≤ 200 ∧ (c : ℚ) / d < alpha_day2_success_ratio 

theorem beta_largest_possible_success_ratio :
  (a + c : ℚ) / 500 = 349 / 500 :=
sorry

end beta_largest_possible_success_ratio_l33_33661


namespace distance_from_M0_to_plane_l33_33318

structure Point3D :=
  (x y z : ℝ)

def plane_equation (A B C D : ℝ) (p : Point3D) : ℝ :=
  A * p.x + B * p.y + C * p.z + D

def distance_from_point_to_plane (A B C D : ℝ) (p0 : Point3D) : ℝ :=
  abs (plane_equation A B C D p0) / real.sqrt (A^2 + B^2 + C^2)

def M1 : Point3D := ⟨-1, 2, 4⟩
def M2 : Point3D := ⟨-1, -2, -4⟩
def M3 : Point3D := ⟨3, 0, -1⟩
def M0 : Point3D := ⟨-2, 3, 5⟩

-- We know the desired coefficients of the plane equation are:
def plane_A : ℝ := 1
def plane_B : ℝ := -8
def plane_C : ℝ := 4
def plane_D : ℝ := 1

theorem distance_from_M0_to_plane :
  distance_from_point_to_plane plane_A plane_B plane_C plane_D M0 = 5 / 9 :=
by
  -- Proof placeholder
  sorry

end distance_from_M0_to_plane_l33_33318


namespace degree_of_each_exterior_angle_of_regular_octagon_l33_33024
-- We import the necessary Lean libraries

-- Define the degrees of each exterior angle of a regular octagon
theorem degree_of_each_exterior_angle_of_regular_octagon : 
  (∑ (i : Fin 8), (360 / 8) = 360 → (360 / 8) = 45) :=
by
  sorry

end degree_of_each_exterior_angle_of_regular_octagon_l33_33024


namespace sin_x_plus_sin_a_ge_b_cos_x_forall_x_l33_33700

theorem sin_x_plus_sin_a_ge_b_cos_x_forall_x (a b : ℝ) (n : ℤ) :
  (∀ x : ℝ, sin x + sin a ≥ b * cos x) ↔ 
  (a = (4 * n + 1) * π / 2 ∧ b = 0) :=
by
  sorry

end sin_x_plus_sin_a_ge_b_cos_x_forall_x_l33_33700


namespace line_through_points_slope_intercept_l33_33301

theorem line_through_points_slope_intercept :
  ∀ (m b : ℝ), (m = (7 - 3) / (3 - 1)) → (b = 1) → m + b = 3 := 
by
  intros m b
  intro h1
  intro h2
  rw [h1, h2]
  norm_num
  sorry

end line_through_points_slope_intercept_l33_33301


namespace geometric_sequence_product_l33_33801

theorem geometric_sequence_product
  (a : ℕ → ℝ)
  (b : ℕ → ℝ)
  (common_diff : ℝ)
  (h1 : ∀ n m, a (n + 1) = a n + common_diff)
  (h2 : 3 * a 2005 - (a 2007)^2 + 3 * a 2009 = 0)
  (h3 : ∀ n m, b (n + 1) / b n = b (m + 1) / b m) 
  (h4 : b 2007 = a 2007)
  : b 2006 * b 2008 = 36 := 
begin
  sorry
end

end geometric_sequence_product_l33_33801


namespace product_evaluation_l33_33694

-- Define the variables and conditions
variable (b : ℝ)
variable (h : b = 3)

-- Prove the main statement
theorem product_evaluation : (b - 5) * (b - 4) * (b - 3) * (b - 2) * (b - 1) * b = 0 :=
by {
  -- substitution of h
  rw h,
  -- actual calculation is omitted with sorry
  sorry,
}

end product_evaluation_l33_33694


namespace Z_3_5_value_l33_33417

def Z (a b : ℕ) : ℕ :=
  b + 12 * a - a ^ 2

theorem Z_3_5_value : Z 3 5 = 32 := by
  sorry

end Z_3_5_value_l33_33417


namespace train_pass_time_is_24_seconds_l33_33210

noncomputable def time_to_pass_man (train_length : ℝ) (train_speed_kmph : ℝ) (man_speed_kmph : ℝ) : ℝ :=
  let relative_speed_kmph := train_speed_kmph + man_speed_kmph in
  let relative_speed_mps := relative_speed_kmph * (1000 / 3600) in
  train_length / relative_speed_mps

theorem train_pass_time_is_24_seconds :
  time_to_pass_man 440 60 6 = 24 := by
  sorry

end train_pass_time_is_24_seconds_l33_33210


namespace concyclic_proof_l33_33670

/-!
  In quadrilateral \(ABCD\), \(\angle BAD \neq 90^\circ\).
  Circle centered at \(B\) with a radius of \(BA\) intersects the extensions of \(AB\) and \(CB\) at points \(E\) and \(F\).
  Circle centered at \(D\) with a radius of \(DA\) intersects the extensions of \(AD\) and \(CD\) at points \(M\) and \(N\).
  Line \(EN\) intersects line \(FM\) at point \(G\).
  Line \(AG\) intersects line \(ME\) at point \(T\).
  The second intersection point of line \(EN\) with circle \(\odot D\) is point \(P\).
  The second intersection point of line \(MF\) with circle \(\odot B\) is point \(Q\).
  Prove that \(G\), \(P\), \(T\), and \(Q\) are concyclic.
-/

variables {A B C D E F G M N P Q T : Point} [Nonempty Set]

def quadrilateral (A B C D : Point) : Prop := true
def circle (center radius : Point) (p1 p2 : Point) : Prop := true
def intersects (line1 line2 : Line) (point : Point) : Prop := true
def second_intersection (line : Line) (circle : Circle) (point : Point) : Prop := true
def concyclic (G P T Q : Point) : Prop := true
def angle_not_90 (A B D : Point) : Prop := true

theorem concyclic_proof :
  quadrilateral A B C D →
  angle_not_90 A B D →
  circle B A E F →
  circle D A M N →
  intersects (line EN) (line FM) G →
  intersects (line AG) (line ME) T →
  second_intersection (line EN) (circle D A M N) P →
  second_intersection (line MF) (circle B A E F) Q →
  concyclic G P T Q :=
by sorry

end concyclic_proof_l33_33670


namespace minimum_period_and_max_value_of_f_l33_33052

noncomputable def f (x : ℝ) : ℝ := sin (x / 3) + cos (x / 3)

theorem minimum_period_and_max_value_of_f :
  (∃ T > 0, ∀ x: ℝ, f (x + T) = f x) ∧
  (∀ x: ℝ, f x ≤ sqrt 2 ∧ 
          (∃ y: ℝ, f y = sqrt 2)) :=
by 
  sorry

end minimum_period_and_max_value_of_f_l33_33052


namespace parallelogram_incenters_trapezoid_l33_33810

/-- Given right trapezoid ABCD, E is the midpoint of AB, and ∠CED = 90°.
O₁, O₂ and O₃ are the incenters of ΔDAE, ΔCBE, and ΔCED respectively.
Prove that EO₁O₃O₂ is a parallelogram. -/
theorem parallelogram_incenters_trapezoid (A B C D E O₁ O₂ O₃ : Point)
  (h_trapezoid : IsRightTrapezoid A B C D) 
  (h_midpoint : E = midpoint A B )
  (h_right_angle : ∠ C E D = 90)
  (h_incenter_DAE : Incenter O₁ (Triangle D A E))
  (h_incenter_CBE : Incenter O₂ (Triangle C B E))
  (h_incenter_CED : Incenter O₃ (Triangle C E D)) :
  Parallelogram (Quad E O₁ O₃ O₂) := 
sorry

end parallelogram_incenters_trapezoid_l33_33810


namespace radius_of_cookie_eq_sqrt17_l33_33293

theorem radius_of_cookie_eq_sqrt17 (x y : ℝ) :
  x^2 + y^2 + 17 = 6 * x + 10 * y →
  ∃ (c : ℝ), (x - 3)^2 + (y - 5)^2 = c^2 ∧ c = sqrt 17 :=
by
  intro h
  sorry

end radius_of_cookie_eq_sqrt17_l33_33293


namespace min_period_max_value_f_l33_33132

def f (x : ℝ) : ℝ := sin (x / 3) + cos (x / 3)

theorem min_period_max_value_f :
  ∃ T M, (∀ x, f (x + T) = f x) ∧ 0 < T ∧
    0 < M ∧ (∀ x, f x ≤ M) ∧ (∃ x, f x = M) ∧ 
    T = 6 * π ∧ M = √2 := sorry

end min_period_max_value_f_l33_33132


namespace incorrectConclusion_l33_33808

variable {A B C D : Type}
variable [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D]

structure Parallelogram (A B C D : Type) :=
(isParallelogram : ∀ {a b c d : A}, a + b = c + d)

structure Rectangle (A B C D : Type) extends Parallelogram A B C D :=
(isRectangle : ∀ {ac bd : A}, ac * bd = 0)

structure Rhombus (A B C D : Type) extends Parallelogram A B C D :=
(isRhombus : ∀ {ac bd : A}, ac = bd)

theorem incorrectConclusion (ABCD : Parallelogram A B C D)
  (h : ∀ {ac bd : A}, ac * bd = 0)
  : ¬(∀ {ac bd : A}, ac * bd = h → Rectangle A B C D) :=
sorry

end incorrectConclusion_l33_33808


namespace smallest_positive_x_l33_33202

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

theorem smallest_positive_x : ∃ x : ℕ, x > 0 ∧ is_palindrome (x + 6789) ∧ x = 218 := by
  sorry

end smallest_positive_x_l33_33202


namespace line_equation_l33_33239

theorem line_equation (b r S : ℝ) (h : ℝ) (m : ℝ) (eq_one : S = 1/2 * b * h) (eq_two : h = 2*S / b) (eq_three : |m| = r / b) 
  (eq_four : m = r / b) : 
  (∀ x y : ℝ, y = m * (x - b) → b > 0 → r > 0 → S > 0 → rx - bry - rb = 0) := 
sorry

end line_equation_l33_33239


namespace compare_abc_l33_33720

noncomputable def a : ℝ := 1 / 2
noncomputable def b : ℝ := real.sqrt 2
noncomputable def c : ℝ := real.log 2 / real.log 3

theorem compare_abc : b > c ∧ c > a := by
  -- Proof steps would go here
  sorry

end compare_abc_l33_33720


namespace negation_equiv_l33_33918

theorem negation_equiv {x : ℝ} : 
  (¬ (x^2 < 1 → -1 < x ∧ x < 1)) ↔ (x^2 ≥ 1 → x ≥ 1 ∨ x ≤ -1) :=
by
  sorry

end negation_equiv_l33_33918


namespace probability_two_white_balls_sequential_l33_33225

theorem probability_two_white_balls_sequential :
  let total_balls := 15 in
  let white_balls := 7 in
  let black_balls := 8 in
  let first_white_prob := (white_balls:ℝ) / (total_balls:ℝ) in
  let second_white_prob_given_first_white := (white_balls - 1)/(total_balls - 1:ℝ) in
  first_white_prob * second_white_prob_given_first_white = 1 / 5 :=
by
  sorry

end probability_two_white_balls_sequential_l33_33225


namespace sequence_period_16_l33_33686

theorem sequence_period_16 (a : ℝ) (h : a > 0) 
  (u : ℕ → ℝ) (h1 : u 1 = a) (h2 : ∀ n, u (n + 1) = -1 / (u n + 1)) : 
  u 16 = a :=
sorry

end sequence_period_16_l33_33686


namespace find_real_numbers_prove_irrational_numbers_l33_33998

-- Define the condition for Question 1
def fractional_part (x : ℝ) : ℝ := x - ⌊x⌋

-- Theorem for the first part
theorem find_real_numbers (x : ℝ) :
  fractional_part x + fractional_part (1 / x) = 1 →
  x = (3 + Real.sqrt 5) / 2 ∨ x = (3 - Real.sqrt 5) / 2 :=
sorry

-- Theorem for the second part
theorem prove_irrational_numbers (x : ℝ) :
  fractional_part x + fractional_part (1 / x) = 1 →
  ¬ ∃ (r : ℚ), x = r :=
sorry

end find_real_numbers_prove_irrational_numbers_l33_33998


namespace numPeopleToLeftOfKolya_l33_33524

-- Definitions based on the conditions.
def peopleToRightOfKolya := 12
def peopleToLeftOfSasha := 20
def peopleToRightOfSasha := 8

-- Theorem statement with the given conditions and conclusion.
theorem numPeopleToLeftOfKolya 
  (h1 : peopleToRightOfKolya = 12)
  (h2 : peopleToLeftOfSasha = 20)
  (h3 : peopleToRightOfSasha = 8) :
  ∃ n, n = 16 :=
by
  -- Proving the theorem will be done here.
  sorry

end numPeopleToLeftOfKolya_l33_33524


namespace river_depth_l33_33249

theorem river_depth (w : ℝ) (v_kmph : ℝ) (V_m3_min : ℝ) (h : ℝ) 
    (hw : w = 32) (hv_kmph : v_kmph = 2) (hV_m3_min : V_m3_min = 3200) 
    (hv_m_min : v_kmph * 1000 / 60 = v_m_min) : 
    V_m3_min = w * v_m_min * h → h = 3 :=
begin
  sorry
end

end river_depth_l33_33249


namespace complex_power_example_l33_33217

theorem complex_power_example : ( (1 / 2 : ℂ) - (complex.I * (real.sqrt 3 / 2)) )^3 = -1 :=
by
  -- placeholder for the actual proof
  sorry

end complex_power_example_l33_33217


namespace range_x_is_union_l33_33175

noncomputable def range_of_x (x : ℝ) : Prop :=
  log (-x^2 + 5*x + 24) < 1

theorem range_x_is_union (x : ℝ) :
  range_of_x x ↔ (x > -3 ∧ x < -2) ∨ (x > 7 ∧ x < 8) :=
sorry

end range_x_is_union_l33_33175


namespace logarithm_division_simplified_l33_33206

theorem logarithm_division_simplified :
  ∀ (a : ℝ), a > 0 → log 16 / log (1 / 16) = -1 :=
by
  sorry

end logarithm_division_simplified_l33_33206


namespace car_cost_l33_33892

theorem car_cost (days_in_week : ℕ) (sue_days : ℕ) (sister_days : ℕ) 
  (sue_payment : ℕ) (car_cost : ℕ) 
  (h1 : days_in_week = 7)
  (h2 : sue_days = days_in_week - sister_days)
  (h3 : sister_days = 4)
  (h4 : sue_payment = 900)
  (h5 : sue_payment * days_in_week = sue_days * car_cost) :
  car_cost = 2100 := 
by {
  sorry
}

end car_cost_l33_33892


namespace find_other_coin_denomination_l33_33177

theorem find_other_coin_denomination :
  ∀ (total_coins total_value num_20_paise_coins denomination_20_paise other_coin_denomination num_other_coins : ℕ),
    total_coins = 344 →
    total_value = 7100 →
    num_20_paise_coins = 300 →
    denomination_20_paise = 20 →
    num_other_coins = total_coins - num_20_paise_coins →
    other_coin_denomination = (total_value - num_20_paise_coins * denomination_20_paise) / num_other_coins →
    other_coin_denomination = 25 :=
by
  intros total_coins total_value num_20_paise_coins denomination_20_paise other_coin_denomination num_other_coins
  assume h_total_coins h_total_value h_num_20_paise_coins h_denomination_20_paise h_num_other_coins h_other_coin_denomination
  sorry

end find_other_coin_denomination_l33_33177


namespace problem1_divisibility_by_11_problem2_divisibility_by_11_l33_33630

-- Problem 1: Proving divisibility by 11 for the sum of a two-digit number and its digit-swapped counterpart
theorem problem1_divisibility_by_11 (a b : ℤ) : 11 ∣ (10 * a + b + 10 * b + a) :=
by
  have : 10 * a + b + 10 * b + a = 11 * (a + b), sorry
  exact dvd.intro (a + b) this

-- Problem 2: Proving divisibility by 11 for a four-digit number with specific digit positions
theorem problem2_divisibility_by_11 (m n : ℤ) : 11 ∣ (1000 * m + 100 * n + 10 * n + m) :=
by
  have : 1000 * m + 100 * n + 10 * n + m = 11 * (91 * m + 10 * n), sorry
  exact dvd.intro (91 * m + 10 * n) this

end problem1_divisibility_by_11_problem2_divisibility_by_11_l33_33630


namespace inequality_l33_33857

variable (a b c : ℝ)

noncomputable def condition (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c = 1 / 8

theorem inequality (h : condition a b c) : 
  a^2 + b^2 + c^2 + a^2 * b^2 + a^2 * c^2 + b^2 * c^2 ≥ 15 / 16 :=
sorry

end inequality_l33_33857


namespace factorial_expression_value_l33_33203

theorem factorial_expression_value : (12! - 11! + 10!) / 10! = 122 := by
  sorry

end factorial_expression_value_l33_33203


namespace number_of_proper_subsets_l33_33412

theorem number_of_proper_subsets (A : Set α) (h : A.card = 5) : (A.proper_subset_count = 31) :=
by
  -- Proof goes here
  sorry

end number_of_proper_subsets_l33_33412


namespace find_k_l33_33927

theorem find_k (k : ℝ) (h : ∀ x : ℝ, x^2 + 10 * x + k = 0 → (∃ a : ℝ, a > 0 ∧ (x = -3 * a ∨ x = -a))) :
  k = 18.75 :=
sorry

end find_k_l33_33927


namespace velvet_needed_for_box_l33_33478

theorem velvet_needed_for_box : 
  let area_long_side := 8 * 6
  let area_short_side := 5 * 6
  let area_top_bottom := 40
  let total_area := (2 * area_long_side) + (2 * area_short_side) + (2 * area_top_bottom)
  total_area = 236 :=
by
  sorry

end velvet_needed_for_box_l33_33478


namespace correct_propositions_l33_33849

-- Definitions for lines and planes
variables {m n : Line} {α β γ : Plane}

-- Conditions:
axiom m_n_diff : m ≠ n
axiom α_β_γ_diff : α ≠ β ∧ β ≠ γ ∧ α ≠ γ

-- Propositions:
axiom prop1 : (α ∥ β) → (α ∥ γ) → (β ∥ γ)
axiom prop2 : (α ⟂ β) → (m ∥ α) → (m ⟂ β)
axiom prop3 : (m ⊆ α) → (n ⟂ β) → (α ∥ β) → (m ⟂ n)
axiom prop4 : (m ∥ n) → (n ⊆ α) → (m ∥ α)

-- Statement to prove:
theorem correct_propositions : ∃ p q : Prop, p ∧ q ∧ (Prop) ↔ prop1 ∧ prop3 := sorry 

end correct_propositions_l33_33849


namespace find_m_of_symmetric_y_axis_l33_33435

/-- Define the properties of a point in the Cartesian coordinate system -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Define the points A and B and the condition of symmetry with respect to the y-axis -/
def A : Point := {x := -3, y := 4}
def B (m : ℝ) : Point := {x := 3, y := m}

/-- The theorem stating that if points A and B are symmetric with respect to the y-axis, then m = 4 -/
theorem find_m_of_symmetric_y_axis (m : ℝ) : 
  B(m).x = -A.x ∧ B(m).y = A.y → m = 4 :=
by 
  intro h
  cases h 
  sorry

end find_m_of_symmetric_y_axis_l33_33435


namespace relationship_between_A_B_C_l33_33345

-- Definitions based on the problem conditions
def A : Set ℝ := {θ | ∃ k : ℤ, 2 * k * Real.pi < θ ∧ θ < 2 * k * Real.pi + Real.pi / 2}
def B : Set ℝ := {θ | 0 < θ ∧ θ < Real.pi / 2}
def C : Set ℝ := {θ | θ < Real.pi / 2}

-- Proof statement: Prove the specified relationship
theorem relationship_between_A_B_C : B ∪ C = C := by
  sorry

end relationship_between_A_B_C_l33_33345


namespace perpendicular_line_x_intercept_l33_33588

theorem perpendicular_line_x_intercept (x y : ℝ) :
  (4 * x + 5 * y = 10) →
  (1 * y + 0 = y → y = (5 / 4) * x - 3) →
  y = 0 →
  x = 12 / 5 :=
begin
  sorry
end

end perpendicular_line_x_intercept_l33_33588


namespace min_period_of_f_max_value_of_f_l33_33153

def f : ℝ → ℝ := λ x, Real.sin (x / 3) + Real.cos (x / 3)

theorem min_period_of_f : ∃ T > 0, (∀ x, f (x + T) = f x) ∧ T = 6 * Real.pi :=
by
  sorry

theorem max_value_of_f : ∃ x, f x = Real.sqrt 2 :=
by
  sorry

end min_period_of_f_max_value_of_f_l33_33153


namespace attendees_gift_exchange_l33_33266

theorem attendees_gift_exchange (x : ℕ) (h1 : 56 = x * (x - 1) / 2) : 
  x * (x - 1) = 112 :=
by
  sorry

end attendees_gift_exchange_l33_33266


namespace round_periodic_decimal_l33_33494

-- Round to the nearest thousandth
def rounding_to_thousandth (x : ℝ) : ℝ := 
  (Real.floor (x * 1000) / 1000 : ℝ)

noncomputable def periodic_decimal : ℝ := 
  (37.259 + 0.259 / 1000 / (1 - 1 / 1000^3) : ℝ)

theorem round_periodic_decimal : rounding_to_thousandth periodic_decimal = 37.259 :=
by
  sorry

end round_periodic_decimal_l33_33494


namespace cone_apex_angle_l33_33242

-- Define the problem conditions in mathematical form
def max_cross_section_area (A B C : Type) : ℝ := sorry  -- Represents maximum cross-sectional area
def axial_cross_section_area (A B C : Type) : ℝ := sorry  -- Represents axial cross-sectional area

-- Statement of the theorem to prove
theorem cone_apex_angle {A B C : Type} (max_area_condition : max_cross_section_area A B C = 2 * axial_cross_section_area A B C) :
  -- Prove that the angle at the vertex of the cone's axial cross-section is 120 degrees
  ∃ φ : ℝ, φ = 120 ∧ (sin φ = 0.5) :=
  sorry  -- Proof obligation left for further development

end cone_apex_angle_l33_33242


namespace people_to_left_of_kolya_l33_33526

theorem people_to_left_of_kolya (people_right_kolya people_left_sasha people_right_sasha : ℕ) (total_people : ℕ) :
  (people_right_kolya = 12) →
  (people_left_sasha = 20) →
  (people_right_sasha = 8) →
  (total_people = people_left_sasha + people_right_sasha + 1) →
  total_people - people_right_kolya - 1 = 16 :=
begin
  sorry
end

end people_to_left_of_kolya_l33_33526


namespace smallest_positive_period_pi_increasing_interval_range_on_interval_l33_33386

noncomputable def f (x : ℝ) : ℝ := sin (2 * x + π / 3) + sin (2 * x - π / 3) + 2 * cos x ^ 2 - 1

-- Problem 1: Smallest positive period
theorem smallest_positive_period_pi : 
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = π := sorry

-- Problem 2: Interval where the function is increasing
theorem increasing_interval (k : ℤ) : 
  ∀ x, x ∈ [-3 * π / 8 + k * π, π / 8 + k * π] → 
  (2 * x + π / 4) ∈ [-π / 2 + 2 * k * π, π / 2 + 2 * k * π] := sorry

-- Problem 3: Range of the function on [-π/4, π/4]
theorem range_on_interval : 
  ∀ x, x ∈ [-π / 4, π / 4] → f x ∈ [-1, sqrt 2] := sorry

end smallest_positive_period_pi_increasing_interval_range_on_interval_l33_33386


namespace second_shortest_tape_is_Juman_l33_33041

structure TapeLengths where
  Juman : ℝ
  Jeongho : ℝ
  Jiyong : ℝ
  Cheoljung : ℝ

def initial_lengths : TapeLengths := {
  Juman := 0.6,
  Jeongho := 0.6 + 3 / 10,
  Jiyong := 19 / 25,
  Cheoljung := (19 / 25) - 0.2
}

theorem second_shortest_tape_is_Juman :
  (List.sort (λ x y => x < y) [initial_lengths.Cheoljung, initial_lengths.Juman, initial_lengths.Jiyong, initial_lengths.Jeongho]).nth 1 = some initial_lengths.Juman :=
by
  sorry

end second_shortest_tape_is_Juman_l33_33041


namespace betty_honey_oats_problem_l33_33282

theorem betty_honey_oats_problem
  (o h : ℝ)
  (h_condition1 : o ≥ 8 + h / 3)
  (h_condition2 : o ≤ 3 * h) :
  h ≥ 3 :=
sorry

end betty_honey_oats_problem_l33_33282


namespace socks_count_l33_33773

/-- Prove that the number of socks needed to ensure at least two of the same color is 4,
    given there are 12 white socks, 14 green socks, and 15 red socks. -/
theorem socks_count (whitesocks greensocks redsocks : ℕ) : whitesocks = 12 ∧ greensocks = 14 ∧ redsocks = 15 → ensures_two_same_color = 4 :=
begin
  intro h,
  sorry
end

end socks_count_l33_33773


namespace min_positive_period_f_max_value_f_l33_33068

noncomputable def f (x : ℝ) : ℝ := sin (x / 3) + cos (x / 3)

theorem min_positive_period_f : ∃ T > 0, (∀ x, f (x + T) = f x) ∧ ∀ T', (T' > 0 ∧ ∀ x, f (x + T') = f x) → T ≤ T' :=
  sorry

theorem max_value_f : ∃ M, (∀ x, f x ≤ M) ∧ (∀ ε > 0, ∃ x, M - ε < f x) ∧ M = sqrt 2 :=
  sorry

end min_positive_period_f_max_value_f_l33_33068


namespace find_pairs_l33_33314

def is_prime (p : ℕ) : Prop := (p ≥ 2) ∧ ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

theorem find_pairs (a p : ℕ) (h_pos_a : a > 0) (h_prime_p : is_prime p) :
  (∀ m n : ℕ, 0 < m → 0 < n → (a ^ (2 ^ n) % p ^ n = a ^ (2 ^ m) % p ^ m ∧ a ^ (2 ^ n) % p ^ n ≠ 0))
  ↔ (∃ k : ℕ, a = 2 * k + 1 ∧ p = 2) :=
sorry

end find_pairs_l33_33314


namespace sum_divisors_inequality_l33_33491

def d (n : ℕ) : ℕ := (n.divisors).card

theorem sum_divisors_inequality (n : ℕ) (hn: n ≥ 1) : 
  (∑ k in Finset.range (n + 1), 1 / (d (k + 1) : ℝ)) > (Real.sqrt (n + 1) - 1) := 
by
  sorry

end sum_divisors_inequality_l33_33491


namespace smallest_number_among_neg4_neg2_0_1_l33_33330

theorem smallest_number_among_neg4_neg2_0_1 : 
  ∀ (a b c d : ℤ), a = -4 → b = -2 → c = 0 → d = 1 → 
  min (min a b) (min c d) = -4 :=
by
  intros a b c d ha hb hc hd
  rw [ha, hb, hc, hd]
  exact min_eq_left (min_eq_left (show -4 < -2 from neg_lt_neg_iff.mpr (by norm_num)))

end smallest_number_among_neg4_neg2_0_1_l33_33330


namespace num_possible_sequences_l33_33561

theorem num_possible_sequences (n : Nat) (A_not_first_nor_last : ∀ (seq : Fin n → Nat), seq 0 ≠ 1 ∧ seq (n-1) ≠ 1) : 
  ∃ (total_sequences : Nat), total_sequences = 480 :=
sorry

end num_possible_sequences_l33_33561


namespace range_of_a_l33_33842

noncomputable def f (a x : ℝ) := x + a^2 / x
noncomputable def g (x : ℝ) := x - Real.log x

theorem range_of_a (a : ℝ) (h : a > 0) :
  (∀ x1 : ℝ, ∀ x2 : ℝ, 1 ≤ x2 ∧ x2 ≤ Real.exp 1 → f a x1 ≥ g x2) →
  a ≥ (Real.exp 1 - 1) / 2 :=
by
  intro hfg
  let g_max := g (Real.exp 1)
  have : g_max = Real.exp 1 - 1, by rw [g]; simp
  have h_min := f a a
  have : h_min = 2 * a, by rw [f]; simp [Real.exp, mul_div_cancel', ne_of_gt h]
  sorry

end range_of_a_l33_33842


namespace weight_of_3_moles_HBrO3_is_386_73_l33_33200

noncomputable def H_weight : ℝ := 1.01
noncomputable def Br_weight : ℝ := 79.90
noncomputable def O_weight : ℝ := 16.00
noncomputable def HBrO3_weight : ℝ := H_weight + Br_weight + 3 * O_weight
noncomputable def weight_of_3_moles_of_HBrO3 : ℝ := 3 * HBrO3_weight

theorem weight_of_3_moles_HBrO3_is_386_73 : weight_of_3_moles_of_HBrO3 = 386.73 := by
  sorry

end weight_of_3_moles_HBrO3_is_386_73_l33_33200


namespace triangle_max_perimeter_l33_33813

noncomputable def max_perimeter_triangle_ABC (a b c : ℝ) (A B C : ℝ) (h1 : B = 60) (h2 : b = 2 * Real.sqrt 3) : ℝ := 
  a + b + c

theorem triangle_max_perimeter (a b c A B C : ℝ) (h1 : B = 60) (h2 : b = 2 * Real.sqrt 3) :
  max_perimeter_triangle_ABC a b c A B C h1 h2 ≤ 6 * Real.sqrt 3 :=
sorry

end triangle_max_perimeter_l33_33813


namespace triangle_area_l33_33603

theorem triangle_area (c a : ℝ) (h₁ : c = 15) (h₂ : a = 9) :
  let b := Math.sqrt (c^2 - a^2) in
  let area := (a * b) / 2 in
  area = 54 := by
  sorry

end triangle_area_l33_33603


namespace bishop_stays_on_white_l33_33008

theorem bishop_stays_on_white (i j: ℕ) 
  (h: (i + j) % 2 = 1) 
  (h₀: 0 ≤ i ∧ i < 8) 
  (h₁: 0 ≤ j ∧ j < 8) 
  (k: ℤ) 
  (valid_move: (0 ≤ i + k) ∧ (i + k < 8) ∨ (0 ≤ i - k) ∧ (i - k < 8) ∨ (0 ≤ j + k) ∧ (j + k < 8) ∨ (0 ≤ j - k) ∧ (j - k < 8)) :
  let move1 := (i + k : ℤ)
  let move2 := (j + k : ℤ) in
  ((move1 + move2) % 2) = 1 :=
sorry

end bishop_stays_on_white_l33_33008


namespace min_period_of_f_max_value_of_f_l33_33162

def f : ℝ → ℝ := λ x, Real.sin (x / 3) + Real.cos (x / 3)

theorem min_period_of_f : ∃ T > 0, (∀ x, f (x + T) = f x) ∧ T = 6 * Real.pi :=
by
  sorry

theorem max_value_of_f : ∃ x, f x = Real.sqrt 2 :=
by
  sorry

end min_period_of_f_max_value_of_f_l33_33162


namespace girls_at_start_l33_33947

theorem girls_at_start (B G : ℕ) (h1 : B + G = 600) (h2 : 6 * B + 7 * G = 3840) : G = 240 :=
by
  -- actual proof is omitted
  sorry

end girls_at_start_l33_33947


namespace correct_articles_for_sentence_l33_33633

theorem correct_articles_for_sentence : 
  (∀ (sky world : Type), 
   (∀ (adj_sky adj_world : Type -> Prop),
    (adj_sky sky → adj_world world → 
     (adj_sky sky → ∃ (a : Prop), a = true) →
     (adj_world world → ∃ (a : Prop), a = true) →
     true)) = true :=
begin
  sorry
end

end correct_articles_for_sentence_l33_33633


namespace gecko_third_day_crickets_l33_33236

def total_crickets : ℕ := 70
def first_day_percentage : ℝ := 0.30
def first_day_crickets : ℝ := first_day_percentage * total_crickets
def second_day_crickets : ℝ := first_day_crickets - 6
def third_day_crickets : ℝ := total_crickets - (first_day_crickets + second_day_crickets)

theorem gecko_third_day_crickets :
  third_day_crickets = 34 :=
by
  sorry

end gecko_third_day_crickets_l33_33236


namespace condition_necessary_but_not_sufficient_l33_33365

-- Definitions based on given conditions
variables {a b c : ℝ}

-- The condition that needs to be qualified
def condition (a b c : ℝ) := a > 0 ∧ b^2 - 4 * a * c < 0

-- The statement to be verified
def statement (a b c : ℝ) := ∀ x : ℝ, a * x^2 + b * x + c > 0

-- Prove that the condition is a necessary but not sufficient condition for the statement
theorem condition_necessary_but_not_sufficient :
  condition a b c → (¬ (condition a b c ↔ statement a b c)) :=
by
  sorry

end condition_necessary_but_not_sufficient_l33_33365


namespace quarters_initial_l33_33005

-- Define the given conditions
def candies_cost_dimes : Nat := 4 * 3
def candies_cost_cents : Nat := candies_cost_dimes * 10
def lollipop_cost_quarters : Nat := 1
def lollipop_cost_cents : Nat := lollipop_cost_quarters * 25
def total_spent_cents : Nat := candies_cost_cents + lollipop_cost_cents
def money_left_cents : Nat := 195
def total_initial_money_cents : Nat := money_left_cents + total_spent_cents
def dimes_count : Nat := 19
def dimes_value_cents : Nat := dimes_count * 10

-- Prove that the number of quarters initially is 6
theorem quarters_initial (quarters_count : Nat) (h : quarters_count * 25 = total_initial_money_cents - dimes_value_cents) : quarters_count = 6 :=
by
  sorry

end quarters_initial_l33_33005


namespace complex_modulus_pure_imaginary_l33_33782

noncomputable def is_pure_imaginary (z : ℂ) : Prop := 
  z.re = 0 ∧ z.im ≠ 0

theorem complex_modulus_pure_imaginary 
  (m : ℝ) 
  (z : ℂ) 
  (h1 : z = (m^2 - 9) + (m^2 + 2m - 3) * Complex.i) 
  (h2 : is_pure_imaginary z) : 
  abs z = 12 := 
sorry

end complex_modulus_pure_imaginary_l33_33782


namespace rocket_max_height_and_danger_l33_33029

theorem rocket_max_height_and_danger:
  (a : ℝ) (τ : ℝ) (g : ℝ) (y_obj : ℝ) 
  (a = 30) (τ = 30) (g = 10) (y_obj = 50000):
  let V₀ := a * τ,
      y₀ := (a * τ^2) / 2,
      t_high := V₀ / g,
      y_max := y₀ + V₀ * t_high - (g * t_high^2) / 2 in
  (y_max = 54000) ∧ (y_max > y_obj) :=
by
  dsimp [V₀, y₀, t_high, y_max]
  have V0_pos : V₀ = 900 := by sorry
  have y0_pos : y₀ = 13500 := by sorry
  have t_high_pos : t_high = 90 := by sorry
  have y_max_pos : y_max = 54000 := by sorry
  exact ⟨y_max_pos, by linarith [y_max_pos]⟩

end rocket_max_height_and_danger_l33_33029


namespace min_period_of_f_max_value_of_f_l33_33154

def f : ℝ → ℝ := λ x, Real.sin (x / 3) + Real.cos (x / 3)

theorem min_period_of_f : ∃ T > 0, (∀ x, f (x + T) = f x) ∧ T = 6 * Real.pi :=
by
  sorry

theorem max_value_of_f : ∃ x, f x = Real.sqrt 2 :=
by
  sorry

end min_period_of_f_max_value_of_f_l33_33154


namespace t_50_mod_7_l33_33915

-- Conditions
def t : ℕ → ℕ
| 1       := 3
| (n + 1) := 3^(t n)

-- Proof Statement
theorem t_50_mod_7 : t 50 % 7 = 6 := by sorry

end t_50_mod_7_l33_33915


namespace gcd_of_72_120_168_l33_33913

theorem gcd_of_72_120_168 : Nat.gcd (Nat.gcd 72 120) 168 = 24 := 
by
  sorry

end gcd_of_72_120_168_l33_33913


namespace members_not_in_A_nor_B_l33_33558

variable (U A B : Finset ℕ) -- We define the sets as finite sets of natural numbers.
variable (hU_size : U.card = 190) -- Size of set U is 190.
variable (hB_size : (U ∩ B).card = 49) -- 49 items are in set B.
variable (hAB_size : (A ∩ U ∩ B).card = 23) -- 23 items are in both A and B.
variable (hA_size : (U ∩ A).card = 105) -- 105 items are in set A.

theorem members_not_in_A_nor_B :
  (U \ (A ∪ B)).card = 59 := sorry

end members_not_in_A_nor_B_l33_33558


namespace min_positive_period_and_max_value_of_f_l33_33093

noncomputable def f (x : ℝ) : ℝ := sin (x / 3) + cos (x / 3)

theorem min_positive_period_and_max_value_of_f :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ T = 6 * π) ∧ 
  (∃ x, f x = sqrt 2) :=
by {
  sorry
}

end min_positive_period_and_max_value_of_f_l33_33093


namespace friend_reading_time_l33_33404

theorem friend_reading_time (S : ℝ) (H1 : S > 0) (H2 : 3 = 2 * (3 / 2)) : 
  (1.5 / (5 * S)) = 0.3 :=
by 
  sorry

end friend_reading_time_l33_33404


namespace min_positive_period_f_max_min_values_f_l33_33752

def f (x : ℝ) : ℝ := sin x * cos x - sqrt 3 * (cos x)^2

theorem min_positive_period_f : (∃ P > 0, ∀ x, f (x + P) = f x) ∧ (∀ P' > 0, (∃ x, x < P' ∧ f (x + P') = f x) → P' ≥ π) :=
by sorry

theorem max_min_values_f (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ π / 2) : 
  -sqrt 3 ≤ f x ∧ f x ≤ 1 - sqrt 3 / 2 :=
by sorry

end min_positive_period_f_max_min_values_f_l33_33752


namespace veronica_yellow_balls_count_l33_33585

theorem veronica_yellow_balls_count (T Y B : ℕ) (h1 : B = 33) (h2 : Y = 0.45 * T) (h3 : T = Y + B) : Y = 27 :=
by
  sorry

end veronica_yellow_balls_count_l33_33585


namespace min_positive_period_and_max_value_of_f_l33_33094

noncomputable def f (x : ℝ) : ℝ := sin (x / 3) + cos (x / 3)

theorem min_positive_period_and_max_value_of_f :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ T = 6 * π) ∧ 
  (∃ x, f x = sqrt 2) :=
by {
  sorry
}

end min_positive_period_and_max_value_of_f_l33_33094


namespace arcs_on_circle_21_points_l33_33871

theorem arcs_on_circle_21_points :
  ∃ (arcs : set (set ℕ)), (finite arcs) ∧ (∀ arc ∈ arcs, arc.card = 2) ∧ 
  (21 ∈ arcs) ∧ (∃ arc, ¬ arc ⊆ (⋃ arc ∈ arcs, arc)) ∧ 
  (card (filter (λ arc, arc.2 ≤ 120) arcs) ≥ 100) := 
sorry

end arcs_on_circle_21_points_l33_33871


namespace equal_triples_l33_33406

theorem equal_triples (a b c x : ℝ) (h_abc : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (h_eq : (xb + (1 - x) * c) / a = (x * c + (1 - x) * a) / b ∧ 
          (x * c + (1 - x) * a) / b = (x * a + (1 - x) * b) / c) : a = b ∧ b = c := by
  sorry

end equal_triples_l33_33406


namespace x_intercept_is_correct_l33_33598

-- Define the original line equation
def original_line (x y : ℝ) : Prop := 4 * x + 5 * y = 10

-- Define the perpendicular line's y-intercept
def y_intercept (y : ℝ) : Prop := y = -3

-- Define the equation of the perpendicular line in slope-intercept form
def perpendicular_line (x y : ℝ) : Prop := y = (5 / 4) * x + -3

-- Prove that the x-intercept of the perpendicular line is 12/5
theorem x_intercept_is_correct : ∃ x : ℝ, x ≠ 0 ∧ (∃ y : ℝ, y = 0) ∧ (perpendicular_line x y) :=
sorry

end x_intercept_is_correct_l33_33598


namespace proposition_range_l33_33356

theorem proposition_range (m : ℝ) : 
  (m < 1/2 ∧ m ≠ 1/3) ∨ (m = 3) ↔ m ∈ Set.Iio (1/3:ℝ) ∪ Set.Ioo (1/3:ℝ) (1/2:ℝ) ∪ {3} :=
sorry

end proposition_range_l33_33356


namespace sum_123_consecutive_even_numbers_l33_33711

theorem sum_123_consecutive_even_numbers :
  let n := 123
  let a := 2
  let d := 2
  let sum_arithmetic_series (n a l : ℕ) := n * (a + l) / 2
  let last_term := a + (n - 1) * d
  sum_arithmetic_series n a last_term = 15252 :=
by
  sorry

end sum_123_consecutive_even_numbers_l33_33711


namespace max_quadratic_value_l33_33298

def quadratic (x : ℝ) : ℝ :=
  -2 * x^2 + 4 * x + 3

theorem max_quadratic_value : ∃ x : ℝ, ∀ y : ℝ, quadratic x = y → y ≤ 5 ∧ (∀ z : ℝ, quadratic z ≤ y) := 
by
  sorry

end max_quadratic_value_l33_33298


namespace min_period_max_value_f_l33_33138

def f (x : ℝ) : ℝ := sin (x / 3) + cos (x / 3)

theorem min_period_max_value_f :
  ∃ T M, (∀ x, f (x + T) = f x) ∧ 0 < T ∧
    0 < M ∧ (∀ x, f x ≤ M) ∧ (∃ x, f x = M) ∧ 
    T = 6 * π ∧ M = √2 := sorry

end min_period_max_value_f_l33_33138


namespace Rahul_Deepak_age_ratio_l33_33172

-- Definitions of the conditions
def Deepak_age := 21
def Rahul_future_age := 34
def years_added := 6
def Rahul_age := Rahul_future_age - years_added

-- The main theorem to prove the ratio
theorem Rahul_Deepak_age_ratio : (Rahul_age / Deepak_age) = (4 / 3) :=
by
  -- Provided conditions
  let D := Deepak_age
  let R := Rahul_age
  have h1 : D = 21 := rfl
  have h2 : R = 34 - 6 := rfl
  -- Calculation
  have R_value : R = 28 := by simp [h2]
  have ratio : R / D = 28 / 21 := by simp [R_value, h1]
  -- Simplifying the ratio
  have simplified_ratio : 28 / 21 = 4 / 3 := by norm_num
  exact eq.trans ratio simplified_ratio

end Rahul_Deepak_age_ratio_l33_33172


namespace false_statements_about_skew_lines_l33_33775

/-- Define the geometric setup for skew lines and a point P -/
structure SkewLinesAndPoint :=
  (P : Point)
  (l m : Line)
  (P_not_on_l : ¬ P ∈ l)
  (P_not_on_m : ¬ P ∈ m)
  (l_skew_m : are_skew l m)

theorem false_statements_about_skew_lines (setup : SkewLinesAndPoint) :
  ¬ ∃ line_through_P, line_through_P ∥ setup.l ∧ line_through_P ∥ setup.m ∧
  (∃ line_through_P, line_through_P ⟂ setup.l ∧ line_through_P ⟂ setup.m) ∧
  ¬ ∃ line_through_P, line_through_P ∩ setup.l ≠ ∅ ∧ line_through_P ∩ setup.m ≠ ∅ ∧
  (∃ line_through_P, is_skew line_through_P setup.l ∧ is_skew line_through_P setup.m) :=
sorry

end false_statements_about_skew_lines_l33_33775


namespace min_positive_period_and_max_value_l33_33081

def f (x : ℝ) := Real.sin (x / 3) + Real.cos (x / 3)

theorem min_positive_period_and_max_value : 
  (∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T' = T)) 
  ∧ 
  (∀ x, f x ≤ sqrt 2 ∧ (∃ x, f x = sqrt 2)) :=
sorry

end min_positive_period_and_max_value_l33_33081


namespace fleas_move_to_right_l33_33349

theorem fleas_move_to_right {n : ℕ} (hn : n ≥ 2) {λ : ℝ} (hλ : λ ≥ 0) :
  (∀ M : ℝ, ∀ (pos : Fin n → ℝ), (¬ (∀ (i j : Fin n), pos i = pos j)) → 
    ∃ (N : ℕ), ∀ (i : Fin n), pos i > M) ↔ (λ ≥ 1 / (n - 1)) := 
sorry

end fleas_move_to_right_l33_33349


namespace robert_ate_more_l33_33003

variable (robert_chocolates : ℕ) (nickel_chocolates : ℕ)
variable (robert_ate_9 : robert_chocolates = 9) (nickel_ate_2 : nickel_chocolates = 2)

theorem robert_ate_more : robert_chocolates - nickel_chocolates = 7 :=
  by
    sorry

end robert_ate_more_l33_33003


namespace min_positive_period_f_max_value_f_l33_33070

noncomputable def f (x : ℝ) : ℝ := sin (x / 3) + cos (x / 3)

theorem min_positive_period_f : ∃ T > 0, (∀ x, f (x + T) = f x) ∧ ∀ T', (T' > 0 ∧ ∀ x, f (x + T') = f x) → T ≤ T' :=
  sorry

theorem max_value_f : ∃ M, (∀ x, f x ≤ M) ∧ (∀ ε > 0, ∃ x, M - ε < f x) ∧ M = sqrt 2 :=
  sorry

end min_positive_period_f_max_value_f_l33_33070


namespace sum_of_distinct_x_l33_33758

noncomputable def g : ℝ → ℝ := λ x, x^2 / 4 + x + 1

theorem sum_of_distinct_x (h : ∀ x, g(g(g(x))) = 1) : ∑ (x : ℝ) in {x | g(g(g(x))) = 1}.to_finset, x = -8 :=
sorry

end sum_of_distinct_x_l33_33758


namespace min_value_of_xy_l33_33368

-- Definitions based on the conditions of the problem
def x : ℝ := sorry
def y : ℝ := sorry

axiom x_gt_1 : x > 1
axiom y_gt_1 : y > 1
axiom geom_seq_condition : Real.logBase 2 x * Real.logBase 2 y = 1 / 16

-- Statement of the problem
theorem min_value_of_xy (x y : ℝ) (x_gt_1 : x > 1) (y_gt_1 : y > 1) (geom_seq_condition : Real.logBase 2 x * Real.logBase 2 y = 1 / 16) : x * y ≥ Real.sqrt 2 :=
by
  sorry

end min_value_of_xy_l33_33368


namespace Emma_hits_11_l33_33011

-- Define the friends and their corresponding scores
def scores : List (String × Nat) :=
  [("Alice", 21), ("Ben", 10), ("Cindy", 18), ("Dave", 15), ("Emma", 30), ("Felix", 22)]

-- Define the regions score values
def regions : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

-- State the problem to prove that Emma hits the region worth 11 points
theorem Emma_hits_11 (scores : List (String × Nat)) (regions : List Nat) :
  (∃ a b c : Nat, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (a + b + c = 30) ∧ (a = 11 ∨ b = 11 ∨ c = 11) ∧ 
    List.mem a regions ∧ List.mem b regions ∧ List.mem c regions) := 
sorry

end Emma_hits_11_l33_33011


namespace minimum_period_and_max_value_of_f_l33_33051

noncomputable def f (x : ℝ) : ℝ := sin (x / 3) + cos (x / 3)

theorem minimum_period_and_max_value_of_f :
  (∃ T > 0, ∀ x: ℝ, f (x + T) = f x) ∧
  (∀ x: ℝ, f x ≤ sqrt 2 ∧ 
          (∃ y: ℝ, f y = sqrt 2)) :=
by 
  sorry

end minimum_period_and_max_value_of_f_l33_33051


namespace no_finite_union_arithmetic_progressions_l33_33492

def no_solution_set (n : ℕ) : Prop :=
  ∀ x y : ℕ, (1 : ℚ) / x + (1 : ℚ) / y ≠ 3 / n

theorem no_finite_union_arithmetic_progressions :
  ¬∃ (M : ℕ → Prop), (∀ n : ℕ, no_solution_set n ↔ M n) ∧
  (∃ (A : list (ℕ × ℕ)), ∀ n, M n ↔ (∃ a d k, (a, d) ∈ A ∧ n = a + k * d)) :=
sorry

end no_finite_union_arithmetic_progressions_l33_33492


namespace find_sin_cos_alpha_find_tan_alpha_plus_pi_over_4_find_expression_l33_33804

noncomputable def x := -3
noncomputable def y := 4
noncomputable def r := Real.sqrt (x^2 + y^2)

noncomputable def alpha : Angle := sorry -- angle alpha represented in a meaningful way

noncomputable def sin_alpha := y / r
noncomputable def cos_alpha := x / r

theorem find_sin_cos_alpha :
  sin_alpha = 4 / 5 ∧ cos_alpha = -3 / 5 := sorry

noncomputable def tan_alpha := y / x

noncomputable def alpha_plus_pi_over_4 := Angle.add alpha (Angle.pi / 4)
noncomputable def tan_alpha_plus_pi_over_4 := (tan_alpha + 1) / (1 - tan_alpha)

theorem find_tan_alpha_plus_pi_over_4 :
  tan_alpha_plus_pi_over_4 = -1 / 7 := sorry

noncomputable def sin_alpha_plus_pi_over_4 := 
  tan_alpha_plus_pi_over_4 / Real.sqrt (1 + tan_alpha_plus_pi_over_4^2)
noncomputable def cos_alpha_plus_pi_over_4 := 
  1 / Real.sqrt (1 + tan_alpha_plus_pi_over_4^2)

theorem find_expression :
  sin_alpha_plus_pi_over_4^2 + sin_alpha_plus_pi_over_4 * cos_alpha_plus_pi_over_4 = 
  -3 / 25 := sorry

end find_sin_cos_alpha_find_tan_alpha_plus_pi_over_4_find_expression_l33_33804


namespace kelly_carrot_weight_l33_33821

-- Define the number of carrots harvested from each bed
def carrots_bed1 : ℕ := 55
def carrots_bed2 : ℕ := 101
def carrots_bed3 : ℕ := 78
def carrots_per_pound : ℕ := 6

-- Define the total number of carrots
def total_carrots := carrots_bed1 + carrots_bed2 + carrots_bed3

-- Define the total weight in pounds
def total_weight := total_carrots / carrots_per_pound

-- The theorem to prove the total weight is 39 pounds
theorem kelly_carrot_weight : total_weight = 39 := by
  sorry

end kelly_carrot_weight_l33_33821


namespace rocket_maximum_height_rocket_danger_l33_33030

theorem rocket_maximum_height
  (a : ℝ) (τ : ℝ) (g : ℝ) (m_to_km : ℝ := 0.001)
  (h_a : a = 30) (h_tau : τ = 30) (h_g : g = 10) :
  let V_0 := a * τ,
      y_0 := (a * τ^2) / 2,
      t_max := V_0 / g,
      y_max := y_0 + V_0 * t_max - (g * t_max^2) / 2 
  in y_max * m_to_km = 54 :=
by
  sorry

theorem rocket_danger (y_max : ℝ) (h_max : y_max = 54000) :
  y_max > 50000 :=
by
  simp [h_max]

end rocket_maximum_height_rocket_danger_l33_33030


namespace comb_club_ways_l33_33230

theorem comb_club_ways : (nat.choose 20 3) = 1140 :=
by
  sorry

end comb_club_ways_l33_33230


namespace selling_price_before_brokerage_l33_33627

variables (CR BR SP : ℝ)
variables (hCR : CR = 120.50) (hBR : BR = 1 / 400)

theorem selling_price_before_brokerage :
  SP = (CR * 400) / (399) := 
by
  sorry

end selling_price_before_brokerage_l33_33627


namespace shift_sin_graph_l33_33950

theorem shift_sin_graph :
  ∀ x : ℝ, (sin (2 * x - π / 3)) = (sin 2 (x - π / 6)) := by
sorry

end shift_sin_graph_l33_33950


namespace boxes_needed_l33_33572

noncomputable def living_room_length : ℝ := 16
noncomputable def living_room_width : ℝ := 20
noncomputable def sq_ft_per_box : ℝ := 10
noncomputable def already_floored : ℝ := 250

theorem boxes_needed : 
  (living_room_length * living_room_width - already_floored) / sq_ft_per_box = 7 :=
by 
  sorry

end boxes_needed_l33_33572


namespace track_length_l33_33672

theorem track_length (x : ℝ) 
  (h1 : ∀ {d1 d2 : ℝ}, (d1 + d2 = x / 2) → (d1 = 120) → d2 = x / 2 - 120)
  (h2 : ∀ {d1 d2 : ℝ}, (d1 = x / 2 - 120 + 170) → (d1 = x / 2 + 50))
  (h3 : ∀ {d3 : ℝ}, (d3 = 3 * x / 2 - 170)) :
  x = 418 :=
by
  sorry

end track_length_l33_33672


namespace find_a_for_inverse_proportion_l33_33361

theorem find_a_for_inverse_proportion (a : ℝ)
  (h_A : ∃ k : ℝ, 4 = k / (-1))
  (h_B : ∃ k : ℝ, 2 = k / a) :
  a = -2 :=
sorry

end find_a_for_inverse_proportion_l33_33361


namespace modulus_complex_fraction_l33_33372

theorem modulus_complex_fraction (z : ℂ) (hz : z = -1 + Complex.i) : 
  Complex.abs ((z + 3) / (z + 2)) = (Real.sqrt 10) / 2 :=
by
  rw [hz]
  sorry

end modulus_complex_fraction_l33_33372


namespace relationship_l33_33997

def opposite (x : ℝ) : ℝ := -x

def absolute_value (x : ℝ) : ℝ := if x < 0 then -x else x

def reciprocal (x : ℝ) : ℝ := 1 / x

theorem relationship (x : ℝ) : x = (-2) → 
  (opposite x ≠ 1 / 2 ∧ absolute_value x ≠ 1 / 2 ∧ reciprocal x ≠ 1 / 2) → 
  "none of the above" :=
by
  sorry

end relationship_l33_33997


namespace average_speed_of_train_l33_33660

-- Condition: Distance traveled is 42 meters
def distance : ℕ := 42

-- Condition: Time taken is 6 seconds
def time : ℕ := 6

-- Average speed computation
theorem average_speed_of_train : distance / time = 7 := by
  -- Left to the prover
  sorry

end average_speed_of_train_l33_33660


namespace min_positive_period_and_max_value_l33_33104

noncomputable def f (x : ℝ) : ℝ := Real.sin (x / 3) + Real.cos (x / 3)

theorem min_positive_period_and_max_value :
  (∃ T > 0, ∀ x, f (x + T) = f x) ∧ ∃ x, f x = sqrt 2 :=
by
  sorry

end min_positive_period_and_max_value_l33_33104


namespace base_three_102012_eq_302_l33_33197

/-- Convert a base 3 number to its base 10 equivalent. -/
def base_three_to_base_ten (n : ℕ) : ℕ :=
  let digits : List ℕ := [1, 0, 2, 0, 1, 2] -- Representation of 102012_3 with the least significant digit first
  let powers_of_three : List ℕ := List.map (λ k, 3 ^ k) (List.range digits.length)
  List.sum (List.zipWith (fun d p => d * p) digits powers_of_three)

theorem base_three_102012_eq_302 : base_three_to_base_ten 102012 = 302 :=
  sorry

end base_three_102012_eq_302_l33_33197


namespace distribution_count_l33_33302

-- Define the problem constraints and conditions
def ticketNumbers : List ℕ := [1, 2, 3, 4, 5]
def numPeople : ℕ := 4
def consecutivePairs : List (ℕ × ℕ) := [(1, 2), (2, 3), (3, 4), (4, 5)]

-- Define the required theorem
theorem distribution_count :
  ∃ (f : List ℕ → List ℕ), (∀ (p1 p2 : ℕ), p1 ≠ p2 → f ticketNumbers = [p1, p2, (ticketNumbers.erase [p1, p2])]) ∧
                           (∃ (pairs_mem : (ℕ × ℕ)), pairs_mem ∈ consecutivePairs) ∧
                           (list.length ticketNumbers = 5) ∧
                           (numPeople = 4) →
                           (∃ (h : ℕ), h = 96) :=
by
  sorry

end distribution_count_l33_33302


namespace doubled_radius_surface_area_l33_33212

theorem doubled_radius_surface_area (A : ℝ) (r : ℝ) (π : ℝ) (hπ : π = real.pi) (hA : A = 4 * π * r^2) : 
  16 * A = 39376 :=
by
  rw [hA, ←mul_assoc, mul_comm 4 4, mul_assoc]
  exact congr_arg (λ x, 4 * x) hA
  sorry

end doubled_radius_surface_area_l33_33212


namespace binary_to_decimal_l33_33223

theorem binary_to_decimal :
  (1 * 2^0 + 1 * 2^1 + 0 * 2^2 + 1 * 2^3 + 1 * 2^4 + 1 * 2^5 + 1 * 2^6 + 0 * 2^7 + 1 * 2^8) = 379 := 
by
  sorry

end binary_to_decimal_l33_33223


namespace num_students_B_eq_44_l33_33942

/-- Problem Conditions -/
def num_students_A : ℕ := 36
def avg_weight_A : ℝ := 40
def avg_weight_B : ℝ := 35
def avg_weight_whole : ℝ := 37.25

/-- Problem Statement to Prove -/
theorem num_students_B_eq_44 (x : ℕ) :
  (36 * 40 + x * 35) / (36 + x) = 37.25 →
  x = 44 := by
  sorry

end num_students_B_eq_44_l33_33942


namespace range_of_a_for_local_min_max_l33_33737

noncomputable def f (a e x : ℝ) : ℝ := 2 * a^x - e * x^2

theorem range_of_a_for_local_min_max (a e x1 x2 : ℝ) (h_a : 0 < a) (h_a_ne : a ≠ 1) (h_x1_x2 : x1 < x2) 
  (h_min : ∀ x, f a e x > f a e x1) (h_max : ∀ x, f a e x < f a e x2) : 
  (1 / Real.exp 1) < a ∧ a < 1 := 
sorry

end range_of_a_for_local_min_max_l33_33737


namespace time_to_pass_telegraph_post_l33_33624

-- Definitions based on the conditions
def train_length : ℕ := 120
def train_speed_kmph : ℕ := 36
def kmph_to_mps (v : ℕ) : ℕ := v * 1000 / 3600

-- Theorem statement to prove that the time to pass the telegraph post is 12 seconds
theorem time_to_pass_telegraph_post :
  let train_speed_mps := kmph_to_mps train_speed_kmph in
  let time := train_length / train_speed_mps in
  time = 12 :=
by
  let train_speed_mps := kmph_to_mps train_speed_kmph
  let time := train_length / train_speed_mps
  -- the proof would go here
  sorry

end time_to_pass_telegraph_post_l33_33624


namespace circle_with_diameter_AB_tangent_parabola_l33_33305

noncomputable def parabola (x y : ℝ) : Prop := x^2 = 4 * y

noncomputable def tangent_line (x1 y1 x y : ℝ) : Prop := x1 * x - 2 * y - 2 * y1 = 0

noncomputable def tangent_through_H (x1 y1 : ℝ) : Prop := let H := (1 : ℝ, -1 : ℝ)
  in x1 * H.1 - 2 * H.2 - 2 * y1 = 2

noncomputable def line_AB (x y : ℝ) : Prop := x - 2 * y + 2 = 0

noncomputable def circle (x y : ℝ) : Prop := (x - 1)^2 + (y - 3/2)^2 = 25/4

theorem circle_with_diameter_AB_tangent_parabola :
  ∃ x1 y1 x2 y2 : ℝ, parabola x1 y1 ∧ parabola x2 y2 ∧ tangent_through_H x1 y1 ∧
                  tangent_through_H x2 y2 ∧ line_AB x1 y1 ∧ line_AB x2 y2 ∧ circle x1 y1 ∧ circle x2 y2 :=
by {
  sorry
}

end circle_with_diameter_AB_tangent_parabola_l33_33305


namespace purely_imaginary_satisfies_condition_l33_33722

theorem purely_imaginary_satisfies_condition (m : ℝ) (h1 : m^2 + 3 * m - 4 = 0) (h2 : m + 4 ≠ 0) : m = 1 :=
by
  sorry

end purely_imaginary_satisfies_condition_l33_33722


namespace min_positive_period_f_max_value_f_l33_33071

noncomputable def f (x : ℝ) : ℝ := sin (x / 3) + cos (x / 3)

theorem min_positive_period_f : ∃ T > 0, (∀ x, f (x + T) = f x) ∧ ∀ T', (T' > 0 ∧ ∀ x, f (x + T') = f x) → T ≤ T' :=
  sorry

theorem max_value_f : ∃ M, (∀ x, f x ≤ M) ∧ (∀ ε > 0, ∃ x, M - ε < f x) ∧ M = sqrt 2 :=
  sorry

end min_positive_period_f_max_value_f_l33_33071


namespace equal_candies_l33_33286

theorem equal_candies
  (sweet_math_per_box : ℕ := 12)
  (geometry_nuts_per_box : ℕ := 15)
  (sweet_math_boxes : ℕ := 5)
  (geometry_nuts_boxes : ℕ := 4) :
  sweet_math_boxes * sweet_math_per_box = geometry_nuts_boxes * geometry_nuts_per_box := 
  by
  sorry

end equal_candies_l33_33286


namespace sphere_surface_area_l33_33732

theorem sphere_surface_area (r : ℝ) (A B C O : ℝ) (h1 : dist A B = 3) (h2 : dist B C = 3) 
  (h3 : dist C A = 3) (h4 : dist O (plane_through_triangle A B C) = r / 3)
  : 4 * π * r^2 = (27 / 2) * π :=
sorry

end sphere_surface_area_l33_33732


namespace trigonometric_identity_l33_33631

noncomputable def cos := real.cos
noncomputable def tan := real.tan
noncomputable def sin := real.sin

theorem trigonometric_identity :
  4 * cos (50 * real.pi / 180) - tan (40 * real.pi / 180) = real.sqrt 3 :=
by
  have h1 : cos (50 * real.pi / 180) = sin (40 * real.pi / 180), from sorry,
  sorry

end trigonometric_identity_l33_33631


namespace john_bought_9_25_meters_l33_33447

-- Define the given condition: total cost of cloth and cost per meter.
def total_cost : ℝ := 397.75
def cost_per_meter : ℝ := 43

-- Define the question: how many meters of cloth did John buy?
def number_of_meters : ℝ := total_cost / cost_per_meter

-- State the theorem : John bought 9.25 meters of cloth.
theorem john_bought_9_25_meters (h1 : total_cost = 397.75) (h2 : cost_per_meter = 43) : number_of_meters = 9.25 :=
by
  sorry

end john_bought_9_25_meters_l33_33447


namespace average_books_per_student_l33_33794

theorem average_books_per_student:
  ∀ (class_size : ℕ) (zero_books : ℕ) (one_book : ℕ) (two_books : ℕ) (at_least_three_books : ℕ) (total_books : ℕ),
  class_size = 30 →
  zero_books = 5 →
  one_book = 12 →
  two_books = 8 →
  at_least_three_books = class_size - (zero_books + one_book + two_books) →
  total_books = (0 * zero_books) + (1 * one_book) + (2 * two_books) + (3 * at_least_three_books) →
  (total_books : ℚ) / class_size = 1.43 := 
by
  intros class_size zero_books one_book two_books at_least_three_books total_books,
  assume h_class_size h_zero_books h_one_book h_two_books h_at_least_three_books h_total_books,
  have h1 : class_size = 30 := h_class_size,
  have h2 : zero_books = 5 := h_zero_books,
  have h3 : one_book = 12 := h_one_book,
  have h4 : two_books = 8 := h_two_books,
  have h5 : at_least_three_books = class_size - (zero_books + one_book + two_books) := h_at_least_three_books,
  have h6 : total_books = (0 * zero_books) + (1 * one_book) + (2 * two_books) + (3 * at_least_three_books) := h_total_books,
sorry

end average_books_per_student_l33_33794


namespace tan_theta_value_l33_33358

theorem tan_theta_value (θ : ℝ) (a : ℝ) (h1 : -π / 2 < θ) (h2 : θ < π / 2) (h3 : sin θ + cos θ = a) (h4 : 0 < a) (h5 : a < 1) :
  (tan θ = -1 / 3) :=
sorry

end tan_theta_value_l33_33358


namespace min_pq_value_l33_33289

theorem min_pq_value : 
  ∃ (p q : ℕ), p > 0 ∧ q > 0 ∧ 98 * p = q ^ 3 ∧ (∀ p' q' : ℕ, p' > 0 ∧ q' > 0 ∧ 98 * p' = q' ^ 3 → p' + q' ≥ p + q) ∧ p + q = 42 :=
sorry

end min_pq_value_l33_33289


namespace rate_of_interest_l33_33934

theorem rate_of_interest (P T SI: ℝ) (h1 : P = 2500) (h2 : T = 5) (h3 : SI = P - 2000) (h4 : SI = (P * R * T) / 100):
  R = 4 :=
by
  sorry

end rate_of_interest_l33_33934


namespace regular_dinosaur_weight_l33_33671

namespace DinosaurWeight

-- Given Conditions
def Barney_weight (x : ℝ) : ℝ := 5 * x + 1500
def combined_weight (x : ℝ) : ℝ := Barney_weight x + 5 * x

-- Target Proof
theorem regular_dinosaur_weight :
  (∃ x : ℝ, combined_weight x = 9500) -> 
  ∃ x : ℝ, x = 800 :=
by {
  sorry
}

end DinosaurWeight

end regular_dinosaur_weight_l33_33671


namespace cone_lateral_area_l33_33042

-- Definitions from the conditions
def radius_base : ℝ := 1 -- in cm
def slant_height : ℝ := 2 -- in cm

-- Statement to be proved: The lateral area of the cone is 2π cm²
theorem cone_lateral_area : 
  1/2 * (2 * π * radius_base) * slant_height = 2 * π :=
by
  sorry

end cone_lateral_area_l33_33042


namespace gathering_gift_exchange_l33_33263

def number_of_guests (x : ℕ) : Prop :=
  x * (x - 1) = 56

theorem gathering_gift_exchange :
  ∃ x : ℕ, number_of_guests x :=
sorry

end gathering_gift_exchange_l33_33263


namespace find_k_l33_33926

theorem find_k (k : ℝ) (h : ∀ x : ℝ, x^2 + 10 * x + k = 0 → (∃ a : ℝ, a > 0 ∧ (x = -3 * a ∨ x = -a))) :
  k = 18.75 :=
sorry

end find_k_l33_33926


namespace probability_point_closer_to_center_l33_33683

theorem probability_point_closer_to_center (R r : ℝ) (hR : R = 6) (hr : r = 2) :
  let A_outer := Real.pi * R^2,
      A_inner := Real.pi * r^2 in
  (A_inner / A_outer) = 1 / 9 :=
by
  have hR_squared : R^2 = 36 := by rw [hR]; norm_num,
  have hr_squared : r^2 = 4 := by rw [hr]; norm_num,
  have A_outer_def : A_outer = 36 * Real.pi := by rw [←hR_squared]; norm_num,
  have A_inner_def : A_inner = 4 * Real.pi := by rw [←hr_squared]; norm_num,
  rw [A_outer_def, A_inner_def],
  norm_num

end probability_point_closer_to_center_l33_33683


namespace min_period_of_f_max_value_of_f_l33_33163

def f : ℝ → ℝ := λ x, Real.sin (x / 3) + Real.cos (x / 3)

theorem min_period_of_f : ∃ T > 0, (∀ x, f (x + T) = f x) ∧ T = 6 * Real.pi :=
by
  sorry

theorem max_value_of_f : ∃ x, f x = Real.sqrt 2 :=
by
  sorry

end min_period_of_f_max_value_of_f_l33_33163


namespace points_in_plane_are_odd_l33_33346

theorem points_in_plane_are_odd (n : ℕ) (Q : Point) (P : Fin n.succ → Point)
  (h_nonlinear : ∀ i j k : Fin n.succ, i ≠ j → j ≠ k → i ≠ k → ¬Collinear (P i) (P j) (P k))
  (h_triangle : ∀ i j : Fin n.succ, i ≠ j → ∃ k : Fin n.succ, i ≠ k ∧ j ≠ k ∧ InsideTriangle Q (P i) (P j) (P k)) :
  Odd n :=
begin
  sorry
end

end points_in_plane_are_odd_l33_33346


namespace min_period_of_f_max_value_of_f_l33_33160

def f : ℝ → ℝ := λ x, Real.sin (x / 3) + Real.cos (x / 3)

theorem min_period_of_f : ∃ T > 0, (∀ x, f (x + T) = f x) ∧ T = 6 * Real.pi :=
by
  sorry

theorem max_value_of_f : ∃ x, f x = Real.sqrt 2 :=
by
  sorry

end min_period_of_f_max_value_of_f_l33_33160


namespace periodic_f_l33_33846

noncomputable def f_periodic : ℝ → ℝ :=
λ x, if x ∈ (Set.Ico (-2 * Real.pi / 3) 0) then Real.sin x else Real.cos x

def smallest_positive_period : ℝ := (5 * Real.pi) / 3

theorem periodic_f (x : ℝ) : f_periodic (x + smallest_positive_period) = f_periodic x :=
begin
  -- This statement follows from the problem
  sorry
end

example : f_periodic (-16 * Real.pi / 3) = -Real.sqrt 3 / 2 := by
  have h_period : ∀ n : ℤ, f_periodic (-16 * Real.pi / 3 + n * smallest_positive_period) = f_periodic (-16 * Real.pi / 3) := sorry,
  rw [h_period 3],
  exact congr_arg f_periodic (-16 * Real.pi / 3 + 15 * Real.pi / 3 = - Real.pi / 3),
  suffices : -Real.pi / 3 ∈ Set.Ico (-2 * Real.pi / 3) 0,
  exact f_periodic ↑(-Real.pi / 3 : ℝ) = Real.sin (-Real.pi / 3),
  exact Real.sin_neg_pi_div_three,
  exact Real.sin_pi_div_three_ne_minus_sqrt_three_div_two
end

end periodic_f_l33_33846


namespace box_length_l33_33226

theorem box_length (width depth: ℕ) (num_cubes: ℕ) (s: ℕ) 
  (h_width: width = 48) 
  (h_depth: depth = 12) 
  (h_num_cubes: num_cubes = 80)
  (h_volume_eq: (240 * width * depth) = num_cubes * (s^3)) 
  (h_cube_factors: ∀ k : ℕ, k ∈ [12])
  : 240 := by
  have h_len : 240 := 240 -- This would be the step to assert 240, but details are omitted
  sorry

end box_length_l33_33226


namespace minimum_norm_of_v_l33_33469

open Real 

-- Define the vector v and condition
noncomputable def v : ℝ × ℝ := sorry

-- Define the condition
axiom v_condition : ‖(v.1 + 4, v.2 + 2)‖ = 10

-- The statement that we need to prove
theorem minimum_norm_of_v : ‖v‖ = 10 - 2 * sqrt 5 :=
by
  sorry

end minimum_norm_of_v_l33_33469


namespace recommended_cups_l33_33569

theorem recommended_cups (current_cups : ℕ) (R : ℕ) : 
  current_cups = 20 →
  R = current_cups + (6 / 10) * current_cups →
  R = 32 :=
by
  intros h1 h2
  sorry

end recommended_cups_l33_33569


namespace simplify_sqrt_expr_l33_33888

-- We need to prove that simplifying √(5 - 2√6) is equal to √3 - √2.
theorem simplify_sqrt_expr : 
  Real.sqrt (5 - 2 * Real.sqrt 6) = Real.sqrt 3 - Real.sqrt 2 :=
by 
  sorry

end simplify_sqrt_expr_l33_33888


namespace proof_of_min_distance_l33_33376

-- Given Curves C₁ and C₂ and Line C₃
def C₁ (t : ℝ) : ℝ × ℝ :=
  (-4 + Real.cos t, 3 + Real.sin t)

def C₂ (θ : ℝ) : ℝ × ℝ :=
  (8 * Real.cos θ, 3 * Real.sin θ)

def C₃ (t : ℝ) : ℝ × ℝ :=
  (3 + 2 * t, -2 + t)

-- Curve Equation Transformation
def curve_C1_circle (x y : ℝ) : Prop :=
  (x + 4) ^ 2 + (y - 3) ^ 2 = 1

def curve_C2_ellipse (x y : ℝ) : Prop :=
  (x ^ 2) / 64 + (y ^ 2) / 9 = 1

-- Point P on C₁ for t = π/2
def P : ℝ × ℝ :=
  C₁ (Real.pi / 2)

-- Point Q on C₂
def Q (θ : ℝ) : ℝ × ℝ :=
  C₂ θ

-- Midpoint M of PQ
def M (θ : ℝ) : ℝ × ℝ :=
  let P := C₁ (Real.pi / 2)
  let Q := C₂ θ
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Distance from M to line C₃
def distance_M_C3 (θ : ℝ) : ℝ :=
  let Mx := (P (Real.pi / 2)).1 + (Q θ).1 / 2
  let My := (P (Real.pi / 2)).2 + (Q θ).2 / 2
  Real.sqrt 5 / 5 * abs (4 * Real.cos θ - 3 * Real.sin θ - 13)

-- Minimum distance
def min_distance : ℝ :=
  8 * Real.sqrt 5 / 5

-- Lean Statement
theorem proof_of_min_distance :
  ∃ θ : ℝ, distance_M_C3 θ = min_distance := sorry

end proof_of_min_distance_l33_33376


namespace math_problem_l33_33777

theorem math_problem (a b c : ℝ) (h1 : a + b + c = 12) (h2 : ab + ac + bc = 30) :
  a^3 + b^3 + c^3 - 3 * a * b * c + 2 * (a + b + c) = 672 :=
by
  sorry

end math_problem_l33_33777


namespace least_prime_factor_of_5_pow_5_minus_5_pow_4_l33_33978

theorem least_prime_factor_of_5_pow_5_minus_5_pow_4 : (∃ p : ℕ, p.prime ∧ p ≤ 5 - 1 ∧ p ∣ (5^5 - 5^4)) :=
  sorry

end least_prime_factor_of_5_pow_5_minus_5_pow_4_l33_33978


namespace find_k_l33_33219

def vec := ℝ × ℝ

def e1 : vec := (1, 0)
def e2 : vec := (0, 1)
def AB : vec := (1, -1)
def BC : vec := (3, 2)
def CD (k : ℝ) : vec := (k, 2)

def collinear (u v : vec) : Prop :=
  ∃ λ : ℝ, u = (λ • v)

noncomputable def k_value : ℝ :=
if h : collinear (AB + BC) (CD 8) then 8 else sorry

theorem find_k : k_value = 8 := by
  sorry

end find_k_l33_33219


namespace area_of_figure_parameter_values_for_one_solution_l33_33393

theorem area_of_figure : 
  ∀ x y : ℝ, 
    abs (9 - x^2 - y^2 - 2 * y) + abs (-2 * y) = 9 - x^2 - y^2 - 4 * y →
    set_of_points_form_shape_with_area (10 * π + 3 - 10 * arctan 3) :=
by 
  sorry

theorem parameter_values_for_one_solution : 
  ∀ a x y : ℝ, 
    abs (9 - x^2 - y^2 - 2 * y) + abs (-2 * y) = 9 - x^2 - y^2 - 4 * y ∧
    15 * y + 3 * a = (4 * a + 15) * x → 
    (a = -5 ∨ a = -3) :=
by 
  sorry

end area_of_figure_parameter_values_for_one_solution_l33_33393


namespace extra_men_required_l33_33989

-- Define the conditions
def total_length : ℝ := 15        -- Length of the road in km
def total_time : ℕ := 300         -- Total time for the project in days
def initial_men : ℕ := 50         -- Initial number of men
def length_done_100_days : ℝ := 2.5  -- Length of road done in 100 days
def days_passed : ℕ := 100        -- Days passed

-- Define the remaining length and time
def remaining_length := total_length - length_done_100_days
def remaining_time := total_time - days_passed

-- Calculate the rates
def current_work_rate := length_done_100_days / (days_passed : ℝ)
def required_work_rate := remaining_length / (remaining_time : ℝ)

-- Define the proof problem
theorem extra_men_required :
  let original_men := initial_men in
  let needed_men := (required_work_rate * original_men) / current_work_rate in
  let extra_men := needed_men - original_men in
  extra_men = 75 := by
  sorry

end extra_men_required_l33_33989


namespace factorize_ax2_minus_a_l33_33696

theorem factorize_ax2_minus_a (a x : ℝ) : a * x^2 - a = a * (x + 1) * (x - 1) :=
by
  sorry

end factorize_ax2_minus_a_l33_33696


namespace participation_plans_correctness_l33_33885

noncomputable def totalParticipationPlans (students : Finset ℕ) (subjects : ℕ) : ℕ :=
  let A := 0
  let others := students.erase A
  let choose2 := others.card.choose 2
  let arrange := subjects.factorial
  choose2 * arrange

theorem participation_plans_correctness :
  let students := {0, 1, 2, 3}
  let subjects := 3
  totalParticipationPlans students subjects = 18 :=
by
  sorry

end participation_plans_correctness_l33_33885


namespace find_k_l33_33924

-- Given conditions and hypothesis stated
axiom quadratic_eq (x k : ℝ) : x^2 + 10 * x + k = 0

def roots_in_ratio_3_1 (α β : ℝ) : Prop :=
  α / β = 3

-- Statement of the theorem to be proved
theorem find_k {α β k : ℝ} (h1 : quadratic_eq α k) (h2 : quadratic_eq β k)
               (h3 : α ≠ 0) (h4 : β ≠ 0) (h5 : roots_in_ratio_3_1 α β) :
  k = 18.75 :=
by
  sorry

end find_k_l33_33924


namespace compute_difference_of_reciprocals_l33_33714

theorem compute_difference_of_reciprocals
  (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x - y = x / y) :
  (1 / x) - (1 / y) = - (1 / y^2) :=
by
  sorry

end compute_difference_of_reciprocals_l33_33714


namespace dima_can_find_heavy_ball_l33_33689

noncomputable def find_heavy_ball
  (balls : Fin 9) -- 9 balls, indexed from 0 to 8 representing the balls 1 to 9
  (heavy : Fin 9) -- One of the balls is heavier
  (weigh : Fin 9 → Fin 9 → Ordering) -- A function that compares two groups of balls and gives an Ordering: .lt, .eq, or .gt
  (predetermined_sets : List (Fin 9 × Fin 9)) -- A list of tuples representing balls on each side for the two weighings
  (valid_sets : predetermined_sets.length ≤ 2) : Prop := -- Not more than two weighings
  ∃ idx : Fin 9, idx = heavy -- Need to prove that we can always find the heavier ball

theorem dima_can_find_heavy_ball :
  ∀ (balls : Fin 9) (heavy : Fin 9)
    (weigh : Fin 9 → Fin 9 → Ordering)
    (predetermined_sets : List (Fin 9 × Fin 9))
    (valid_sets : predetermined_sets.length ≤ 2),
  find_heavy_ball balls heavy weigh predetermined_sets valid_sets :=
sorry -- Proof is omitted

end dima_can_find_heavy_ball_l33_33689


namespace integer_points_on_circle_l33_33421

theorem integer_points_on_circle (r : ℕ) (h : r = 5) : 
  ∃ n, n = 12 ∧ 
  {p : ℤ × ℤ | p.1 ^ 2 + p.2 ^ 2 = r ^ 2}.card = n := 
by
  sorry

end integer_points_on_circle_l33_33421


namespace people_left_of_Kolya_l33_33540

/-- Given:
    1. There are 12 people to the right of Kolya.
    2. There are 20 people to the left of Sasha.
    3. There are 8 people to the right of Sasha.
    4. The total number of people in the class (including Sasha) is 29.

    Prove:
    The number of people to the left of Kolya is 16.
-/
theorem people_left_of_Kolya : 
  ∀ (total_people right_of_Kolya left_of_Sasha right_of_Sasha : ℕ),
  right_of_Kolya = 12 →
  left_of_Sasha = 20 →
  right_of_Sasha = 8 →
  total_people = 29 →
  left_of_Kolya := total_people - right_of_Kolya - 1
  left_of_Kolya = 16 :=
by
  intros
  sorry

end people_left_of_Kolya_l33_33540


namespace cyclic_quadrilateral_BCED_l33_33690

-- Definitions and conditions
variables {A B C B1 C1 D E : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace B1] [MetricSpace C1] [MetricSpace D] [MetricSpace E]
variables α β γ : ℝ
variables (BB1_parallel : Parallel (Line B B1) (Line C C1))
variables (CC1_parallel : Parallel (Line C C1) (Line B B1))
variables (external_angle_bisector_A : Line A (IntersectionPoint (ParallelLineFrom C1 (Line B B1)) (ParallelLineFrom B1 (Line C C1))) = (Line E D))

-- Theorem Statement
theorem cyclic_quadrilateral_BCED :
  CyclicQuadrilateral B C E D :=
sorry

end cyclic_quadrilateral_BCED_l33_33690


namespace similar_triangles_side_ratio_l33_33335

-- Define the conditions of the problem
def are_similar (ABC A1B1C1 : Triangle) : Prop :=
  is_similar ABC A1B1C1

def area_ratio (ABC A1B1C1 : Triangle) (r : ℝ) : Prop :=
  (area ABC) / (area A1B1C1) = r

-- Main theorem statement deriving from the problem conditions
theorem similar_triangles_side_ratio
  (ABC A1B1C1 : Triangle)
  (h_sim : are_similar ABC A1B1C1)
  (h_area_ratio : area_ratio ABC A1B1C1 0.25) :
  (side_length ABC AB) / (side_length A1B1C1 A1B1) = 0.5 :=
by
  -- Taller's proof for the incomplete portion
  sorry

end similar_triangles_side_ratio_l33_33335


namespace find_a5_l33_33937

variable {a : ℕ → ℝ}

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

def sum_arithmetic_sequence (a : ℕ → ℝ) (n : ℕ) :=
  ∑ i in Finset.range n, a (i + 1)

noncomputable def S (n : ℕ) : ℝ := 30
noncomputable def a2_a3_sum : ℝ := 9

theorem find_a5 (a : ℕ → ℝ) (d : ℝ)
  (h_arith : arithmetic_sequence a d)
  (h_sum : sum_arithmetic_sequence a 5 = 30)
  (h_a2_a3 : a 2 + a 3 = 9) :
  a 5 = 12 := by
  sorry

end find_a5_l33_33937


namespace compare_logs_and_value_l33_33337

theorem compare_logs_and_value (a b c : ℝ) (ha : a = Real.logBase 2 3) (hb : b = Real.logBase 3 4) (hc : c = 3/2) :
  b < c ∧ c < a :=
by
  -- Initial set up for proof, assuming ha, hb, hc as hypotheses
  -- Providing only the statement as required
  sorry

end compare_logs_and_value_l33_33337


namespace distance_from_circle_center_to_line_l33_33026

def circle := { x y : ℝ | x^2 + y^2 - 2 * x + 2 * y = 0 }
def line (x y : ℝ) : Prop := y = x + 1

def center_of_circle : ℝ × ℝ := (1, -1)
def distance_from_point_to_line (p : ℝ × ℝ) := abs ((-1) * p.1 + 1 * p.2 - 1) / sqrt ((-1)^2 + (1)^2)

theorem distance_from_circle_center_to_line : distance_from_point_to_line center_of_circle = 3 * sqrt 2 / 2 :=
by
  sorry

end distance_from_circle_center_to_line_l33_33026


namespace number_of_fowls_l33_33831

theorem number_of_fowls (chickens : ℕ) (ducks : ℕ) (h1 : chickens = 28) (h2 : ducks = 18) : chickens + ducks = 46 :=
by
  sorry

end number_of_fowls_l33_33831


namespace number_of_people_to_the_left_of_Kolya_l33_33536

-- Defining the conditions
variables (left_sasha right_sasha right_kolya total_students left_kolya : ℕ)

-- Condition definitions
def condition1 := right_kolya = 12
def condition2 := left_sasha = 20
def condition3 := right_sasha = 8

-- Calculate total number of students
def calc_total_students : ℕ := left_sasha + right_sasha + 1

-- Calculate number of students to the left of Kolya
def calc_left_kolya (total_students right_kolya : ℕ) : ℕ := total_students - right_kolya - 1

-- Problem statement to prove
theorem number_of_people_to_the_left_of_Kolya
    (H1 : condition1)
    (H2 : condition2)
    (H3 : condition3)
    (total_students : calc_total_students = 29) : 
    calc_left_kolya total_students right_kolya = 16 :=
by
  sorry

end number_of_people_to_the_left_of_Kolya_l33_33536


namespace find_a_purely_imaginary_l33_33364

noncomputable def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem find_a_purely_imaginary (a : ℝ) (z1 z2 : ℂ) : 
  z1 = (1 : ℂ) - 2 * complex.I →
  z2 = (a : ℂ) + 2 * complex.I →
  is_purely_imaginary (z2 / z1) →
  a = 4 :=
by
  sorry

end find_a_purely_imaginary_l33_33364


namespace simons_change_l33_33886

noncomputable def calc_change (Pansies_cost Hydrangea_cost Petunias_cost Lilies_cost Orchids_cost Paid_amount : ℚ) 
                              (Pansies_discount Hydrangea_discount Petunias_discount Lilies_discount Orchids_discount Sales_tax_rate : ℚ) : ℚ :=
let Pansies_total := 5 * Pansies_cost * (1 - Pansies_discount / 100) in
let Hydrangea_total := Hydrangea_cost * (1 - Hydrangea_discount / 100) in
let Petunias_total := 5 * Petunias_cost * (1 - Petunias_discount / 100) in
let Lilies_total := 3 * Lilies_cost * (1 - Lilies_discount / 100) in
let Orchids_total := (Orchids_cost + 0.5 * Orchids_cost * (1 - Orchids_discount / 100)) in
let total_cost_after_discounts := Pansies_total + Hydrangea_total + Petunias_total + Lilies_total + Orchids_total in
let total_cost_with_tax := total_cost_after_discounts * (1 + Sales_tax_rate / 100) in
Paid_amount - total_cost_with_tax

theorem simons_change : calc_change 2.50 12.50 1.00 5.00 7.50 100 10 15 20 12 8 5.25 ≈ 47.35 :=
by
  sorry

end simons_change_l33_33886


namespace num_solutions_cos2_sin2_l33_33403

theorem num_solutions_cos2_sin2 :
  (∃ (x : ℝ) (n : Finset ℤ), -12 < x ∧ x < 128 ∧ (cos x)^2 + 2 * (sin x)^2 = 1) ↔ 
  n.card = 44 :=
by
  sorry

end num_solutions_cos2_sin2_l33_33403


namespace average_weight_section_A_l33_33176

theorem average_weight_section_A :
  ∀ (A : ℝ) (students_A students_B : ℕ)
  (weight_B total_students : ℝ),
  students_A = 50 →
  students_B = 50 →
  weight_B = 80 →
  total_students = (students_A + students_B) →
  (50 * A + 50 * weight_B) / total_students = 70 →
  A = 60 := by
  intros A students_A students_B weight_B total_students
  assume hA hB hw total_eq
  assume class_average
  sorry

end average_weight_section_A_l33_33176


namespace kelly_carrot_weight_l33_33820

-- Define the number of carrots harvested from each bed
def carrots_bed1 : ℕ := 55
def carrots_bed2 : ℕ := 101
def carrots_bed3 : ℕ := 78
def carrots_per_pound : ℕ := 6

-- Define the total number of carrots
def total_carrots := carrots_bed1 + carrots_bed2 + carrots_bed3

-- Define the total weight in pounds
def total_weight := total_carrots / carrots_per_pound

-- The theorem to prove the total weight is 39 pounds
theorem kelly_carrot_weight : total_weight = 39 := by
  sorry

end kelly_carrot_weight_l33_33820


namespace log_a_x_plus_2_fixed_point_l33_33327

theorem log_a_x_plus_2_fixed_point (a : ℝ) (h : a ∈ set.Icc 0 1 ∪ set.Ioo 1 +∞): 
  (∀ x : ℝ, f(x) = log a x + 2) → f(1) = 2 := 
  sorry

end log_a_x_plus_2_fixed_point_l33_33327


namespace evaluate_expression_l33_33310

theorem evaluate_expression : 12 * ((1/3 : ℚ) + (1/4) + (1/6))⁻¹ = 16 := 
by 
  sorry

end evaluate_expression_l33_33310


namespace second_gym_signup_fee_covers_4_months_l33_33830

-- Define constants
def cheap_gym_monthly_fee : ℕ := 10
def cheap_gym_signup_fee : ℕ := 50
def total_spent_first_year : ℕ := 650

-- Define the monthly fee of the second gym
def second_gym_monthly_fee : ℕ := 3 * cheap_gym_monthly_fee

-- Calculate the amount spent on the second gym
def spent_on_second_gym : ℕ := total_spent_first_year - (12 * cheap_gym_monthly_fee + cheap_gym_signup_fee)

-- Define the number of months the sign-up fee covers
def months_covered_by_signup_fee : ℕ := spent_on_second_gym / second_gym_monthly_fee

theorem second_gym_signup_fee_covers_4_months :
  months_covered_by_signup_fee = 4 :=
by
  sorry

end second_gym_signup_fee_covers_4_months_l33_33830


namespace min_positive_period_and_max_value_l33_33103

noncomputable def f (x : ℝ) : ℝ := Real.sin (x / 3) + Real.cos (x / 3)

theorem min_positive_period_and_max_value :
  (∃ T > 0, ∀ x, f (x + T) = f x) ∧ ∃ x, f x = sqrt 2 :=
by
  sorry

end min_positive_period_and_max_value_l33_33103


namespace fixed_points_bound_l33_33457

-- Define the polynomial P and the degree condition
variable {R : Type*} [CommRing R]
variable (P : R[X])

-- Assume that P has integer coefficients and a degree of at least 2
noncomputable def polynomial_has_integer_coefficients_and_degree_n (P : ℤ[X]) (n : ℕ) : Prop :=
  P.degree = n ∧ n ≥ 2

-- Q is defined as P applied k times
def Q (k : ℕ) (P : R[X]) : R[X] :=
  Nat.iterate P.eval k

-- Main statement
theorem fixed_points_bound (P : ℤ[X]) (n k : ℕ) (h : polynomial_has_integer_coefficients_and_degree_n P n) :
  ∃ m ≤ n, ∀ a, Q k P a = a → ∑ i in Finset.range m, 1 = n :=
by
  sorry

end fixed_points_bound_l33_33457


namespace linear_eq_in_one_variable_proof_l33_33664

-- Define a structure for a linear equation in one variable
def is_linear_eq_in_one_variable (eq : Prop) : Prop :=
  ∃ (a b : ℝ), eq = (a ≠ 0 ∧ b ≠ 0 ∧ a * x + b = 0)

-- Define each equation
def eq1 : Prop := 4 * x - 3 = x
def eq2 : Prop := 3 * x * (x - 2) = 1
def eq3 : Prop := 1 - 2 * a = 2 * a + 1
def eq4 : Prop := 3 * a^2 = 5
def eq5 : Prop := (2 * x + 4) / 3 = 3 * x - 2
def eq6 : Prop := x + 1 = 1 / x
def eq7 : Prop := 2 * x - 6 * y = 3 * x - 1
def eq8 : Prop := x = 1

-- The theorem we need to prove
theorem linear_eq_in_one_variable_proof : 
  (is_linear_eq_in_one_variable eq1) ∧ 
  (¬is_linear_eq_in_one_variable eq2) ∧ 
  (is_linear_eq_in_one_variable eq3) ∧ 
  (¬is_linear_eq_in_one_variable eq4) ∧ 
  (is_linear_eq_in_one_variable eq5) ∧ 
  (¬is_linear_eq_in_one_variable eq6) ∧ 
  (¬is_linear_eq_in_one_variable eq7) ∧ 
  (is_linear_eq_in_one_variable eq8) :=
by
  sorry

end linear_eq_in_one_variable_proof_l33_33664


namespace perpendicular_line_x_intercept_l33_33596

noncomputable def slope (a b : ℚ) : ℚ := - a / b

noncomputable def line_equation (m y_intercept : ℚ) (x : ℚ) : ℚ :=
  m * x + y_intercept

theorem perpendicular_line_x_intercept :
  let m1 := slope 4 5,
      m2 := (5 / 4),
      y_int := -3
  in 
  ∀ x, line_equation m2 y_int x = 0 → x = 12 / 5 :=
by
  intro x hx
  sorry

end perpendicular_line_x_intercept_l33_33596


namespace simplify_expression_l33_33887

theorem simplify_expression :
  ∀ (a b c : ℝ), a = √6 ∧ b = √3 ∧ c = √2 → 
  ( (a + 4 * b + 3 * c) / ((a + b) * (b + c)) = a - c ) :=
by 
  intros a b c h,
  rcases h with ⟨ha, hb, hc⟩,
  rw [ha, hb, hc],
  sorry

end simplify_expression_l33_33887


namespace baker_initial_cakes_l33_33280

theorem baker_initial_cakes (sold_cakes bought_cakes : ℕ) (h1 : sold_cakes = 78) (h2 : bought_cakes = 31)
(h3 : sold_cakes = bought_cakes + 47) : 
  let x := 78 + 31
  in x = 109 :=
by
  sorry

end baker_initial_cakes_l33_33280


namespace distribution_count_l33_33303

-- Define the problem constraints and conditions
def ticketNumbers : List ℕ := [1, 2, 3, 4, 5]
def numPeople : ℕ := 4
def consecutivePairs : List (ℕ × ℕ) := [(1, 2), (2, 3), (3, 4), (4, 5)]

-- Define the required theorem
theorem distribution_count :
  ∃ (f : List ℕ → List ℕ), (∀ (p1 p2 : ℕ), p1 ≠ p2 → f ticketNumbers = [p1, p2, (ticketNumbers.erase [p1, p2])]) ∧
                           (∃ (pairs_mem : (ℕ × ℕ)), pairs_mem ∈ consecutivePairs) ∧
                           (list.length ticketNumbers = 5) ∧
                           (numPeople = 4) →
                           (∃ (h : ℕ), h = 96) :=
by
  sorry

end distribution_count_l33_33303


namespace shark_teeth_total_l33_33894

theorem shark_teeth_total :
  let tiger_shark := 180 in
  let hammerhead_shark := (1 / 6) * tiger_shark in
  let great_white_shark := 2 * (tiger_shark + hammerhead_shark) in
  let mako_shark := (5 / 3) * hammerhead_shark in
  tiger_shark + hammerhead_shark + great_white_shark + mako_shark = 680 :=
by {
  let tiger_shark := 180,
  let hammerhead_shark := (1 / 6) * tiger_shark,
  let great_white_shark := 2 * (tiger_shark + hammerhead_shark),
  let mako_shark := (5 / 3) * hammerhead_shark,
  calc
    tiger_shark + hammerhead_shark + great_white_shark + mako_shark
        = 180 + ((1 / 6) * 180) + (2 * (180 + ((1 / 6) * 180))) + ((5 / 3) * ((1 / 6) * 180)) : by sorry
    ... = 180 + 30 + 420 + 50 : by sorry
    ... = 680 : by sorry
}

end shark_teeth_total_l33_33894


namespace x_and_y_complete_work_in_12_days_l33_33215

noncomputable def work_rate_x : ℚ := 1 / 24
noncomputable def work_rate_y : ℚ := 1 / 24
noncomputable def combined_work_rate : ℚ := work_rate_x + work_rate_y

theorem x_and_y_complete_work_in_12_days : (1 / combined_work_rate) = 12 :=
by
  sorry

end x_and_y_complete_work_in_12_days_l33_33215


namespace cardinality_of_symmetric_difference_l33_33623

variable (x y : Set ℕ)
variable (h1 : x.card = 12) (h2 : y.card = 18) (h3 : (x ∩ y).card = 6)

theorem cardinality_of_symmetric_difference : (x ∆ y).card = 18 :=
by
  sorry

end cardinality_of_symmetric_difference_l33_33623


namespace circle_area_in_square_l33_33772

theorem circle_area_in_square
  (playground_area : ℝ)
  (pi_val : ℝ)
  (h1 : playground_area = 400)
  (h2 : pi_val = 3.1) :
  let side_length := real.sqrt playground_area in
  let diameter := side_length in
  let radius := diameter / 2 in
  let circle_area := pi_val * radius^2 in
  circle_area = 310 :=
by
  sorry

end circle_area_in_square_l33_33772


namespace arithmetic_sequence_value_l33_33436

theorem arithmetic_sequence_value 
  (a : ℕ → ℤ) 
  (d : ℤ) 
  (h1 : ∀ n, a n = a 1 + (n - 1) * d)
  (h2 : 4 * a 3 + a 11 - 3 * a 5 = 10) : 
  (1 / 5 * a 4 = 1) := 
by
  sorry

end arithmetic_sequence_value_l33_33436


namespace john_oil_change_cost_l33_33449

theorem john_oil_change_cost
  (monthly_miles : ℕ) (oil_change_miles : ℕ) (free_oil_change_per_year : ℕ) (oil_change_cost : ℕ)
  (H1 : monthly_miles = 1000)
  (H2 : oil_change_miles = 3000)
  (H3 : free_oil_change_per_year = 1)
  (H4 : oil_change_cost = 50) :
  let yearly_miles := 12 * monthly_miles,
      total_oil_changes := yearly_miles / oil_change_miles,
      paid_oil_changes := total_oil_changes - free_oil_change_per_year,
      total_cost := paid_oil_changes * oil_change_cost in
  total_cost = 150 := by
  sorry

end john_oil_change_cost_l33_33449


namespace correct_equation_for_gift_exchanges_l33_33275

theorem correct_equation_for_gift_exchanges
  (x : ℕ)
  (H : (x * (x - 1)) = 56) :
  x * (x - 1) = 56 := 
by 
  exact H

end correct_equation_for_gift_exchanges_l33_33275


namespace equation_represents_single_point_l33_33504

theorem equation_represents_single_point (d : ℝ) :
  (∀ x y : ℝ, 3*x^2 + 4*y^2 + 6*x - 8*y + d = 0 ↔ (x = -1 ∧ y = 1)) → d = 7 :=
sorry

end equation_represents_single_point_l33_33504


namespace bicycle_wheels_l33_33559

theorem bicycle_wheels :
  ∀ (b : ℕ),
  let bicycles := 24
  let tricycles := 14
  let wheels_per_tricycle := 3
  let total_wheels := 90
  ((bicycles * b) + (tricycles * wheels_per_tricycle) = total_wheels) → b = 2 :=
by {
  sorry
}

end bicycle_wheels_l33_33559


namespace lisa_flew_distance_l33_33861

-- Define the given conditions
def speed := 32  -- speed in miles per hour
def time := 8    -- time in hours

-- Define the derived distance
def distance := speed * time  -- using the formula Distance = Speed × Time

-- Prove that the calculated distance is 256 miles
theorem lisa_flew_distance : distance = 256 :=
by
  sorry

end lisa_flew_distance_l33_33861


namespace maximum_inequality_l33_33843

theorem maximum_inequality (a b c : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) 
    (h₃ : a + b + c = 2) : 
    (ab_val : (a * b) / (a + b) + (a * c) / (a + c) + (b * c) / (b + c) ≤ 1) := by 
    sorry

end maximum_inequality_l33_33843


namespace gasoline_tank_capacity_l33_33277

def car_speed : ℝ := 40
def fuel_efficiency : ℝ := 1 / 40
def travel_time : ℝ := 5
def fraction_used : ℝ := 0.4166666666666667
def gasoline_used : ℝ := 5

theorem gasoline_tank_capacity : (5 / 0.4166666666666667) = 12 :=
by
  sorry

end gasoline_tank_capacity_l33_33277


namespace minimum_period_and_max_value_of_f_l33_33049

noncomputable def f (x : ℝ) : ℝ := sin (x / 3) + cos (x / 3)

theorem minimum_period_and_max_value_of_f :
  (∃ T > 0, ∀ x: ℝ, f (x + T) = f x) ∧
  (∀ x: ℝ, f x ≤ sqrt 2 ∧ 
          (∃ y: ℝ, f y = sqrt 2)) :=
by 
  sorry

end minimum_period_and_max_value_of_f_l33_33049


namespace find_P_l33_33278

variable (P : ℕ) 

-- Conditions
def cost_samosas : ℕ := 3 * 2
def cost_mango_lassi : ℕ := 2
def cost_per_pakora : ℕ := 3
def total_cost : ℕ := 25
def tip_rate : ℚ := 0.25

-- Total cost before tip
def total_cost_before_tip (P : ℕ) : ℕ := cost_samosas + cost_mango_lassi + cost_per_pakora * P

-- Total cost with tip
def total_cost_with_tip (P : ℕ) : ℚ := 
  (total_cost_before_tip P : ℚ) + (tip_rate * total_cost_before_tip P : ℚ)

-- Proof Goal
theorem find_P (h : total_cost_with_tip P = total_cost) : P = 4 :=
by
  sorry

end find_P_l33_33278


namespace x_intercept_of_perpendicular_line_l33_33590

noncomputable def x_intercept_perpendicular (m₁ m₂ : ℚ) : ℚ :=
  let m_perpendicular := -1 / m₁ in
  let b := -3 in
  -b / m_perpendicular

theorem x_intercept_of_perpendicular_line :
  (4 * x_intercept_perpendicular (-4/5) (5/4) + 5 * 0) = 10 :=
by
  sorry

end x_intercept_of_perpendicular_line_l33_33590


namespace calc_value_l33_33440

noncomputable def parametric_x (θ : ℝ) : ℝ := 3 * Real.cos θ
noncomputable def parametric_y (θ : ℝ) : ℝ := 3 * Real.sin θ

def polar_line_ρ (ρ θ : ℝ) : Prop := ρ * (Real.cos θ - Real.sin θ) = 1

def rectangular_curve (x y : ℝ) : Prop := x^2 + y^2 = 9
def rectangular_line (x y : ℝ) : Prop := x - y - 1 = 0

def point_M : ℝ × ℝ := (0, -1)

def parametric_line_l (t : ℝ) : ℝ × ℝ := ((Real.sqrt 2 / 2) * t, -1 + (Real.sqrt 2 / 2) * t)

def intersections (t1 t2 : ℝ) : Prop := t1 + t2 = Real.sqrt 2 ∧ t1 * t2 = -8

theorem calc_value (M : ℝ × ℝ) (A B : ℝ × ℝ) :
  let t1 := (A.2 + 1) / (Real.sqrt 2 / 2)
  let t2 := (B.2 + 1) / (Real.sqrt 2 / 2)
  t1 + t2 = Real.sqrt 2 ∧ t1 * t2 = -8 →
  |1 / Real.sqrt ((A.1 - M.1)^2 + (A.2 - M.2)^2) -
   1 / Real.sqrt ((B.1 - M.1)^2 + (B.2 - M.2)^2)| = Real.sqrt 2 / 8 :=
sorry

end calc_value_l33_33440


namespace ron_l33_33637

-- Definitions for the given problem conditions
def cost_of_chocolate_bar : ℝ := 1.5
def s'mores_per_chocolate_bar : ℕ := 3
def number_of_scouts : ℕ := 15
def s'mores_per_scout : ℕ := 2

-- Proof that Ron will spend $15.00 on chocolate bars
theorem ron's_chocolate_bar_cost :
  (number_of_scouts * s'mores_per_scout / s'mores_per_chocolate_bar) * cost_of_chocolate_bar = 15 :=
by
  sorry

end ron_l33_33637


namespace people_to_left_of_kolya_l33_33529

theorem people_to_left_of_kolya (people_right_kolya people_left_sasha people_right_sasha : ℕ) (total_people : ℕ) :
  (people_right_kolya = 12) →
  (people_left_sasha = 20) →
  (people_right_sasha = 8) →
  (total_people = people_left_sasha + people_right_sasha + 1) →
  total_people - people_right_kolya - 1 = 16 :=
begin
  sorry
end

end people_to_left_of_kolya_l33_33529


namespace wheel_sum_even_and_greater_than_10_l33_33193

-- Definitions based on conditions
def prob_even_A : ℚ := 3 / 8
def prob_odd_A : ℚ := 5 / 8
def prob_even_B : ℚ := 1 / 4
def prob_odd_B : ℚ := 3 / 4

-- Event probabilities from solution steps
def prob_both_even : ℚ := prob_even_A * prob_even_B
def prob_both_odd : ℚ := prob_odd_A * prob_odd_B
def prob_even_sum : ℚ := prob_both_even + prob_both_odd
def prob_even_sum_greater_10 : ℚ := 1 / 3

-- Compute final probability
def final_probability : ℚ := prob_even_sum * prob_even_sum_greater_10

-- The statement that needs proving
theorem wheel_sum_even_and_greater_than_10 : final_probability = 3 / 16 := by
  sorry

end wheel_sum_even_and_greater_than_10_l33_33193


namespace problem_statement_l33_33036

noncomputable def given_function (x : ℝ) : ℝ := Real.sin (Real.pi / 2 - 2 * x)

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def smallest_positive_period (f : ℝ → ℝ) (T : ℝ) : Prop :=
  T > 0 ∧ ∀ x : ℝ, f (x + T) = f x

theorem problem_statement :
  is_even_function given_function ∧ smallest_positive_period given_function Real.pi :=
by
  sorry

end problem_statement_l33_33036


namespace exp_log_identity_l33_33774

theorem exp_log_identity
    (x y : ℝ)
    (hx : 30^x = 2)
    (hy : 30^y = 3) :
    6^((1 - x - y) / (2 * (1 - y))) = 5 :=
by
  sorry

end exp_log_identity_l33_33774


namespace smallest_k_l33_33467

-- Define the set S
def S (m : ℕ) : Finset ℕ :=
  (Finset.range (30 * m)).filter (λ n => n % 2 = 1 ∧ n % 5 ≠ 0)

-- Theorem statement
theorem smallest_k (m : ℕ) (k : ℕ) : 
  (∀ (A : Finset ℕ), A ⊆ S m → A.card = k → ∃ (x y : ℕ), x ∈ A ∧ y ∈ A ∧ x ≠ y ∧ (x ∣ y ∨ y ∣ x)) ↔ k ≥ 8 * m + 1 :=
sorry

end smallest_k_l33_33467


namespace polygon_sum_of_sides_l33_33898

-- Define the problem conditions and statement
theorem polygon_sum_of_sides :
  ∀ (A B C D E F : ℝ)
    (area_polygon : ℝ)
    (AB BC FA DE horizontal_distance_DF : ℝ),
    area_polygon = 75 →
    AB = 7 →
    BC = 10 →
    FA = 6 →
    DE = AB →
    horizontal_distance_DF = 8 →
    (DE + (2 * area_polygon - AB * BC) / (2 * horizontal_distance_DF) = 8.25) := 
by
  intro A B C D E F area_polygon AB BC FA DE horizontal_distance_DF
  intro h_area_polygon h_AB h_BC h_FA h_DE h_horizontal_distance_DF
  sorry

end polygon_sum_of_sides_l33_33898


namespace original_price_of_petrol_l33_33247

theorem original_price_of_petrol (P : ℝ) :
  (∃ P, 
    ∀ (GA GB GC : ℝ),
    0.8 * P = 0.8 * P ∧
    GA = 200 / P ∧
    GB = 300 / P ∧
    GC = 400 / P ∧
    200 = (GA + 8) * 0.8 * P ∧
    300 = (GB + 15) * 0.8 * P ∧
    400 = (GC + 22) * 0.8 * P) → 
  P = 6.25 :=
by
  sorry

end original_price_of_petrol_l33_33247


namespace min_period_and_max_value_l33_33130

def f (x : ℝ) := Real.sin (x / 3) + Real.cos (x / 3)

theorem min_period_and_max_value :
  (∀ T > 0, T ≠ 6 * Real.pi → ¬∀ x : ℝ, f(x + T) = f(x)) ∧
  ∃ x : ℝ, f(x) = sqrt 2 :=
by
  sorry

end min_period_and_max_value_l33_33130


namespace convex_quad_parallelogram_or_trapezoid_l33_33405

open Real

-- Define the problem conditions
variables {α β γ δ : ℝ}
axiom quad_angles_sum (h : α + β + γ + δ = 360)
axiom sines_equal (h2 : sin α + sin γ = sin β + sin δ)

-- Problem statement in Lean 4
theorem convex_quad_parallelogram_or_trapezoid
  (h : α + β + γ + δ = 360)
  (h2 : sin α + sin γ = sin β + sin δ) :
  (α + β = 180) ∨ (γ + β = 180) :=
sorry

end convex_quad_parallelogram_or_trapezoid_l33_33405


namespace kelly_carrot_weight_l33_33822

-- Define the number of carrots harvested from each bed
def carrots_bed1 : ℕ := 55
def carrots_bed2 : ℕ := 101
def carrots_bed3 : ℕ := 78
def carrots_per_pound : ℕ := 6

-- Define the total number of carrots
def total_carrots := carrots_bed1 + carrots_bed2 + carrots_bed3

-- Define the total weight in pounds
def total_weight := total_carrots / carrots_per_pound

-- The theorem to prove the total weight is 39 pounds
theorem kelly_carrot_weight : total_weight = 39 := by
  sorry

end kelly_carrot_weight_l33_33822


namespace shopkeeper_loss_l33_33653

theorem shopkeeper_loss (x : ℝ) (A B C : ℝ) (hA : A = 2 * x) (hB : B = 3 * x) (hC : C = 4 * x) 
  (hA_profit : 0.25 * A * 0.15 + 0.75 * A * (-0.10) = -0.0375 * A)
  (hB_profit : 0.30 * B * 0.20 + 0.70 * B * (-0.05) = 0.025 * B)
  (hC_profit : 0.40 * C * 0.10 + 0.60 * C * (-0.08) = -0.008 * C)
  (total_loss : -0.0375 * A + 0.025 * B - 0.008 * C = -750) : 
  A = 46875 ∧ B = 70312.5 ∧ C = 93750 :=
begin
  -- Definitions: A = 2 * 23437.5, B = 3 * 23437.5, C = 4 * 23437.5
  sorry
end

end shopkeeper_loss_l33_33653


namespace Mr_Wilson_gained_money_l33_33865

noncomputable def SP1 := 150
noncomputable def SP2 := 150
noncomputable def profit_percentage1 := 25
noncomputable def loss_percentage2 := 15

def cost1 : ℝ := SP1 / (1 + profit_percentage1 / 100)
def cost2 : ℝ := SP2 / (1 - loss_percentage2 / 100)
def total_cost : ℝ := cost1 + cost2
def total_revenue : ℝ := SP1 + SP2
def net_result : ℝ := total_revenue - total_cost

theorem Mr_Wilson_gained_money : net_result > 0 :=
by
  -- The proof goes here
  sorry

end Mr_Wilson_gained_money_l33_33865


namespace solve_for_m_l33_33783

theorem solve_for_m {m : ℝ} (h : ∀ x : ℝ, (m - 5) * x = 0) : m = 5 :=
sorry

end solve_for_m_l33_33783


namespace midpoint_kn_distance_to_lm_l33_33170

theorem midpoint_kn_distance_to_lm (K L M N : Point) (h_cyclic: CyclicQuadrilateral K L M N)
  (h_side_MN : dist M N = 6)
  (h_side_KL : dist K L = 2)
  (h_side_LM : dist L M = 5)
  (h_perpendicular: Perpendicular (line_segment K M) (line_segment L N)):
  dist (midpoint K N) (line L M) = 3.44 := sorry

end midpoint_kn_distance_to_lm_l33_33170


namespace vector_operation_result_l33_33930

variables {V : Type*} [AddCommGroup V] [Module ℝ V] (A B C O E : V)

theorem vector_operation_result :
  (A - B) - (C - B) + (O - E) - (O - C) = (A - E) :=
by
  sorry

end vector_operation_result_l33_33930


namespace elite_salespersons_l33_33618

open Real

theorem elite_salespersons (a : ℝ) (x : ℤ) (hx1 : (100 - x.toReal) * (1 + 0.2) * a ≥ 100 * a) 
  (hx2 : 3.5 * a * x.toReal ≥ (1 / 2) * 100 * a):
  x = 15 ∨ x = 16 := 
sorry

end elite_salespersons_l33_33618


namespace min_positive_period_and_max_value_l33_33079

def f (x : ℝ) := Real.sin (x / 3) + Real.cos (x / 3)

theorem min_positive_period_and_max_value : 
  (∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T' = T)) 
  ∧ 
  (∀ x, f x ≤ sqrt 2 ∧ (∃ x, f x = sqrt 2)) :=
sorry

end min_positive_period_and_max_value_l33_33079


namespace train_speed_is_correct_l33_33251

noncomputable def speed_of_train (distance : ℝ) (time : ℝ) : ℝ :=
  (distance / time) * 3.6

theorem train_speed_is_correct :
  speed_of_train 200 19.99840012798976 = 36.00287976960864 :=
by
  sorry

end train_speed_is_correct_l33_33251


namespace bisection_next_interval_l33_33207

-- Define the function f(x) = x^3 - 2x - 1
def f (x : ℝ) : ℝ := x^3 - 2*x - 1

-- Define the intervals (1, 2) and (1.5, 2)
def interval_initial : Set ℝ := {x | 1 < x ∧ x < 2}
def interval_next : Set ℝ := {x | 1.5 < x ∧ x < 2}

-- State the theorem, with conditions
theorem bisection_next_interval 
  (root_in_interval_initial : ∃ x, f x = 0 ∧ x ∈ interval_initial)
  (f_1_negative : f 1 < 0)
  (f_2_positive : f 2 > 0)
  : ∃ x, f x = 0 ∧ x ∈ interval_next :=
sorry

end bisection_next_interval_l33_33207


namespace page_shoes_count_l33_33876

theorem page_shoes_count (p_i : ℕ) (d : ℝ) (b : ℕ) (h1 : p_i = 120) (h2 : d = 0.45) (h3 : b = 15) : 
  (p_i - (d * p_i)) + b = 81 :=
by
  sorry

end page_shoes_count_l33_33876


namespace length_of_BC_l33_33788

-- Definitions of given conditions
def AB : ℝ := 4
def AC : ℝ := 3
def dot_product_AC_BC : ℝ := 1

-- Hypothesis used in the problem
axiom nonneg_AC (AC : ℝ) : AC ≥ 0
axiom nonneg_AB (AB : ℝ) : AB ≥ 0

-- Statement to be proved
theorem length_of_BC (AB AC dot_product_AC_BC : ℝ)
  (h1 : AB = 4) (h2 : AC = 3) (h3 : dot_product_AC_BC = 1) : exists (BC : ℝ), BC = 3 := by
  sorry

end length_of_BC_l33_33788


namespace added_number_after_doubling_l33_33241

theorem added_number_after_doubling (original_number : ℕ) (result : ℕ) (added_number : ℕ) 
  (h1 : original_number = 7)
  (h2 : 3 * (2 * original_number + added_number) = result)
  (h3 : result = 69) :
  added_number = 9 :=
by
  sorry

end added_number_after_doubling_l33_33241


namespace problem1_problem2_problem3_problem4_l33_33285

theorem problem1 : -20 + (-14) - (-18) - 13 = -29 := by
  sorry

theorem problem2 : (-2) * 3 + (-5) - 4 / (-1/2) = -3 := by
  sorry

theorem problem3 : (-3/8 - 1/6 + 3/4) * (-24) = -5 := by
  sorry

theorem problem4 : -81 / (9/4) * abs (-4/9) - (-3)^3 / 27 = -15 := by
  sorry

end problem1_problem2_problem3_problem4_l33_33285


namespace focus_of_parabola_l33_33317

-- Define the given parabola equation
def given_parabola (x : ℝ) : ℝ := 4 * x^2

-- Define what it means to be the focus of this parabola
def is_focus (focus : ℝ × ℝ) : Prop :=
  focus = (0, 1 / 16)

-- The theorem to prove
theorem focus_of_parabola : ∃ focus : ℝ × ℝ, is_focus focus :=
  by 
    use (0, 1 / 16)
    exact sorry

end focus_of_parabola_l33_33317


namespace travel_ways_l33_33943

theorem travel_ways (buses : Nat) (trains : Nat) (boats : Nat) 
  (hb : buses = 5) (ht : trains = 6) (hb2 : boats = 2) : 
  buses + trains + boats = 13 := by
  sorry

end travel_ways_l33_33943


namespace gravitational_force_at_distance_l33_33910

theorem gravitational_force_at_distance 
  (d_surface : ℕ)
  (f_surface : ℕ)
  (d_space : ℕ)
  (f_expected : ℚ) :
  (f_surface * d_surface ^ 2 = 28800000000) →
  (f_expected = (28800000000 : ℚ) / (d_space ^ 2)) →
  (6000 = d_surface ∧ 800 = f_surface ∧ 360000 = d_space) →
  f_expected = 2 / 9 :=
by
  intros h_constant h_calc h_values
  cases h_values with h_ds hs
  cases hs with h_fs h_dsp
  subst h_ds
  subst h_fs
  subst h_dsp
  rw [h_calc, h_constant]
  norm_num

#print gravitational_force_at_distance

end gravitational_force_at_distance_l33_33910


namespace min_period_and_max_value_l33_33129

def f (x : ℝ) := Real.sin (x / 3) + Real.cos (x / 3)

theorem min_period_and_max_value :
  (∀ T > 0, T ≠ 6 * Real.pi → ¬∀ x : ℝ, f(x + T) = f(x)) ∧
  ∃ x : ℝ, f(x) = sqrt 2 :=
by
  sorry

end min_period_and_max_value_l33_33129


namespace train_speed_ratio_l33_33214

variable (V1 V2 : ℝ)

theorem train_speed_ratio (H1 : V1 * 4 = D1) (H2 : V2 * 36 = D2) (H3 : D1 / D2 = 1 / 9) :
  V1 / V2 = 1 := 
by
  sorry

end train_speed_ratio_l33_33214


namespace problem_to_prove_l33_33396

noncomputable def set_A : Set ℝ := { x | 2^x > (1 / 2) }
noncomputable def set_B : Set ℝ := { x | Real.log x / Real.log 3 < 1 }

theorem problem_to_prove :
  (set_A ∩ (Set.univ \ set_B)) = { x | (-1:ℝ) ≤ x ∧ x ≤ 0 } ∪ { x | (3:ℝ) ≤ x } :=
by 
  sorry

end problem_to_prove_l33_33396


namespace sandy_siding_cost_l33_33007

theorem sandy_siding_cost
  (wall_length wall_height roof_base roof_height : ℝ)
  (siding_length siding_height siding_cost : ℝ)
  (num_walls num_roof_faces num_siding_sections : ℝ)
  (total_cost : ℝ)
  (h_wall_length : wall_length = 10)
  (h_wall_height : wall_height = 7)
  (h_roof_base : roof_base = 10)
  (h_roof_height : roof_height = 6)
  (h_siding_length : siding_length = 10)
  (h_siding_height : siding_height = 15)
  (h_siding_cost : siding_cost = 35)
  (h_num_walls : num_walls = 2)
  (h_num_roof_faces : num_roof_faces = 1)
  (h_num_siding_sections : num_siding_sections = 2)
  (h_total_cost : total_cost = 70) :
  (siding_cost * num_siding_sections) = total_cost := 
by
  sorry

end sandy_siding_cost_l33_33007


namespace find_cement_used_lexi_l33_33883

def cement_used_total : ℝ := 15.1
def cement_used_tess : ℝ := 5.1
def cement_used_lexi : ℝ := cement_used_total - cement_used_tess

theorem find_cement_used_lexi : cement_used_lexi = 10 := by
  sorry

end find_cement_used_lexi_l33_33883


namespace carrots_weight_l33_33817

theorem carrots_weight (carrots_bed1: ℕ) (carrots_bed2: ℕ) (carrots_bed3: ℕ) (carrots_per_pound: ℕ)
  (h_bed1: carrots_bed1 = 55)
  (h_bed2: carrots_bed2 = 101)
  (h_bed3: carrots_bed3 = 78)
  (h_c_per_p: carrots_per_pound = 6) :
  (carrots_bed1 + carrots_bed2 + carrots_bed3) / carrots_per_pound = 39 := by
  sorry

end carrots_weight_l33_33817


namespace distinct_collections_count_l33_33483

def vowels : multiset Char := {'A', 'A', 'E', 'I'}
def consonants : multiset Char := {'M', 'M', 'T', 'T', 'H', 'C', 'S'}

theorem distinct_collections_count :
  (vowels.choose 2).card * (consonants.choose 4).card = 72 :=
  sorry

end distinct_collections_count_l33_33483


namespace numPeopleToLeftOfKolya_l33_33523

-- Definitions based on the conditions.
def peopleToRightOfKolya := 12
def peopleToLeftOfSasha := 20
def peopleToRightOfSasha := 8

-- Theorem statement with the given conditions and conclusion.
theorem numPeopleToLeftOfKolya 
  (h1 : peopleToRightOfKolya = 12)
  (h2 : peopleToLeftOfSasha = 20)
  (h3 : peopleToRightOfSasha = 8) :
  ∃ n, n = 16 :=
by
  -- Proving the theorem will be done here.
  sorry

end numPeopleToLeftOfKolya_l33_33523


namespace find_amplitude_l33_33281

theorem find_amplitude (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
  (h4 : ∀ x, a * Real.cos (b * x - c) ≤ 3) 
  (h5 : ∀ x, abs (a * Real.cos (b * x - c) - a * Real.cos (b * (x + 2 * π / b) - c)) = 0) :
  a = 3 := 
sorry

end find_amplitude_l33_33281


namespace min_pos_period_max_value_l33_33149

noncomputable def f (x : ℝ) := Real.sin (x / 3) + Real.cos (x / 3)

theorem min_pos_period_max_value :
  (∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ T) ∧ T = 6 * Real.pi) ∧ 
  (∀ x, f x ≤ sqrt 2 ∧ (∃ x, f x = sqrt 2)) :=
by
  sorry

end min_pos_period_max_value_l33_33149


namespace height_through_incenter_or_excenter_l33_33515

/-- A triangular pyramid with lateral faces forming equal angles with the base's plane. -/
variables {A B C D : Point} -- Define points for the vertices of the pyramid
variable {H : Point} -- Define the point for the foot of the height from D to the base
variable {triangle_ABC : Triangle ABC} -- Base triangle

/-- Lateral faces form equal angles with the base plane. -/
variable lateral_faces_equal_angle : ∀ face ∈ {Triangle ABD, Triangle BCD, Triangle ACD}, 
  angle (D, get_plane base) = angle (face, get_plane base)

/-- Height of the pyramid is perpendicular to the base. -/
variable height_perpendicular : perpendicular (D, get_plane base)

/-- Prove that the height passes through the incenter or one of the excenters of the base triangle. -/
theorem height_through_incenter_or_excenter 
  (h_base : is_triangle base)
  (h1 : angled_pyramid_equal_angle_faces_lateral D ABC)
  (h2 : perpendicular D base) :
  is_incenter_of_base_triangle base H ∨ is_excenter_of_base_triangle base H :=
sorry

end height_through_incenter_or_excenter_l33_33515


namespace quadratic_has_real_roots_l33_33329

theorem quadratic_has_real_roots (m : ℝ) : (∃ x : ℝ, x^2 + x - 4 * m = 0) ↔ m ≥ -1 / 16 :=
by
  sorry

end quadratic_has_real_roots_l33_33329


namespace salary_percentage_l33_33191

theorem salary_percentage (m n : ℝ) (P : ℝ) (h1 : m + n = 572) (h2 : n = 260) (h3 : m = (P / 100) * n) : P = 120 := 
by
  sorry

end salary_percentage_l33_33191


namespace common_difference_arithmetic_sequence_l33_33437

-- Define the arithmetic sequence properties and common difference
variables {a : ℕ → ℝ} -- The arithmetic sequence
variables {d : ℝ} -- The common difference

-- Define the conditions
def condition1 := a 2 + a 6 = 8
def condition2 := a 3 + a 4 = 3

-- Define the statement to prove the common difference
theorem common_difference_arithmetic_sequence (h1 : condition1) (h2 : condition2) :
  d = 5 :=
sorry

end common_difference_arithmetic_sequence_l33_33437


namespace min_period_and_max_value_l33_33126

def f (x : ℝ) := Real.sin (x / 3) + Real.cos (x / 3)

theorem min_period_and_max_value :
  (∀ T > 0, T ≠ 6 * Real.pi → ¬∀ x : ℝ, f(x + T) = f(x)) ∧
  ∃ x : ℝ, f(x) = sqrt 2 :=
by
  sorry

end min_period_and_max_value_l33_33126


namespace solve_system_l33_33890

open Real

theorem solve_system (x y : ℝ) :
  (9 ^ (xy ^ (1 / 4) * y ^ (1 / 2 / 2)) - 27 * 3 ^ (sqrt y) = 0) ∧
  ((1 / 4) * log x + (1 / 2) * log y = log (4 - x ^ (1 / 4))) ∧
  (0 < x ∧ x < 256) ∧ (0 < y) →
  (x = 1 ∧ y = 9) ∨ (x = 16 ∧ y = 1) :=
by
  sorry

end solve_system_l33_33890


namespace minimum_period_and_max_value_of_f_l33_33048

noncomputable def f (x : ℝ) : ℝ := sin (x / 3) + cos (x / 3)

theorem minimum_period_and_max_value_of_f :
  (∃ T > 0, ∀ x: ℝ, f (x + T) = f x) ∧
  (∀ x: ℝ, f x ≤ sqrt 2 ∧ 
          (∃ y: ℝ, f y = sqrt 2)) :=
by 
  sorry

end minimum_period_and_max_value_of_f_l33_33048
