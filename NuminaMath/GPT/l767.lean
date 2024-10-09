import Mathlib

namespace b_minus_a_l767_76752

theorem b_minus_a (a b : ℕ) : (a * b = 2 * (a + b) + 12) → (b = 10) → (b - a = 6) :=
by
  sorry

end b_minus_a_l767_76752


namespace delta_eq_bullet_l767_76781

-- Definitions of all variables involved
variables (Δ Θ σ : ℕ)

-- Condition 1: Δ + Δ = σ
def cond1 : Prop := Δ + Δ = σ

-- Condition 2: σ + Δ = Θ
def cond2 : Prop := σ + Δ = Θ

-- Condition 3: Θ = 3Δ
def cond3 : Prop := Θ = 3 * Δ

-- The proof problem
theorem delta_eq_bullet (Δ Θ σ : ℕ) (h1 : Δ + Δ = σ) (h2 : σ + Δ = Θ) (h3 : Θ = 3 * Δ) : 3 * Δ = Θ :=
by
  -- Simply restate the conditions and ensure the proof
  sorry

end delta_eq_bullet_l767_76781


namespace store_A_cheaper_than_store_B_l767_76762

noncomputable def store_A_full_price : ℝ := 125
noncomputable def store_A_discount_pct : ℝ := 0.08
noncomputable def store_B_full_price : ℝ := 130
noncomputable def store_B_discount_pct : ℝ := 0.10

noncomputable def final_price_A : ℝ :=
  store_A_full_price * (1 - store_A_discount_pct)

noncomputable def final_price_B : ℝ :=
  store_B_full_price * (1 - store_B_discount_pct)

theorem store_A_cheaper_than_store_B :
  final_price_B - final_price_A = 2 :=
by
  sorry

end store_A_cheaper_than_store_B_l767_76762


namespace gnollish_valid_sentences_l767_76704

def valid_sentences_count : ℕ :=
  let words := ["splargh", "glumph", "amr", "krack"]
  let total_words := 4
  let total_sentences := total_words ^ 3
  let invalid_splargh_glumph := 2 * total_words
  let invalid_amr_krack := 2 * total_words
  let total_invalid := invalid_splargh_glumph + invalid_amr_krack
  total_sentences - total_invalid

theorem gnollish_valid_sentences : valid_sentences_count = 48 :=
by
  sorry

end gnollish_valid_sentences_l767_76704


namespace pq_even_impossible_l767_76749

theorem pq_even_impossible {p q : ℤ} (h : (p^2 + q^2 + p*q) % 2 = 1) : ¬(p % 2 = 0 ∧ q % 2 = 0) :=
by
  sorry

end pq_even_impossible_l767_76749


namespace small_gifts_combinations_large_gifts_combinations_l767_76775

/-
  Definitions based on the given conditions:
  - 12 varieties of wrapping paper.
  - 3 colors of ribbon.
  - 6 types of gift cards.
  - Small gifts can use only 2 out of the 3 ribbon colors.
-/

def wrapping_paper_varieties : ℕ := 12
def ribbon_colors : ℕ := 3
def gift_card_types : ℕ := 6
def small_gift_ribbon_colors : ℕ := 2

/-
  Proof problems:

  - For small gifts, there are 12 * 2 * 6 combinations.
  - For large gifts, there are 12 * 3 * 6 combinations.
-/

theorem small_gifts_combinations :
  wrapping_paper_varieties * small_gift_ribbon_colors * gift_card_types = 144 :=
by
  sorry

theorem large_gifts_combinations :
  wrapping_paper_varieties * ribbon_colors * gift_card_types = 216 :=
by
  sorry

end small_gifts_combinations_large_gifts_combinations_l767_76775


namespace congruence_solution_count_l767_76759

theorem congruence_solution_count :
  ∃! x : ℕ, x < 50 ∧ x + 20 ≡ 75 [MOD 43] := 
by
  sorry

end congruence_solution_count_l767_76759


namespace find_x_values_l767_76741

theorem find_x_values (x : ℝ) : 
  ((x + 1)^2 = 36 ∨ (x + 10)^3 = -27) ↔ (x = 5 ∨ x = -7 ∨ x = -13) :=
by
  sorry

end find_x_values_l767_76741


namespace color_schemes_equivalence_l767_76751

noncomputable def number_of_non_equivalent_color_schemes (n : Nat) : Nat :=
  let total_ways := Nat.choose (n * n) 2
  -- Calculate the count for non-diametrically opposite positions (4 rotations)
  let non_diametric := (total_ways - 24) / 4
  -- Calculate the count for diametrically opposite positions (2 rotations)
  let diametric := 24 / 2
  -- Sum both counts
  non_diametric + diametric

theorem color_schemes_equivalence (n : Nat) (h : n = 7) : number_of_non_equivalent_color_schemes n = 300 :=
  by
    rw [h]
    sorry

end color_schemes_equivalence_l767_76751


namespace double_rooms_booked_l767_76782

theorem double_rooms_booked (S D : ℕ) 
(rooms_booked : S + D = 260) 
(single_room_cost : 35 * S + 60 * D = 14000) : 
D = 196 := 
sorry

end double_rooms_booked_l767_76782


namespace blue_paint_cans_needed_l767_76773

theorem blue_paint_cans_needed (ratio_bg : ℤ × ℤ) (total_cans : ℤ) (r : ratio_bg = (4, 3)) (t : total_cans = 42) :
  let ratio_bw : ℚ := 4 / (4 + 3) 
  let blue_cans : ℚ := ratio_bw * total_cans 
  blue_cans = 24 :=
by
  sorry

end blue_paint_cans_needed_l767_76773


namespace derivative_y_l767_76705

noncomputable def y (x : ℝ) : ℝ := 
  Real.arcsin (1 / (2 * x + 3)) + 2 * Real.sqrt (x^2 + 3 * x + 2)

variable {x : ℝ}

theorem derivative_y :
  2 * x + 3 > 0 → 
  HasDerivAt y (4 * Real.sqrt (x^2 + 3 * x + 2) / (2 * x + 3)) x :=
by 
  sorry

end derivative_y_l767_76705


namespace valid_combinations_l767_76765

-- Definitions based on conditions
def h : Nat := 4  -- number of herbs
def c : Nat := 6  -- number of crystals
def r : Nat := 3  -- number of negative reactions

-- Theorem statement based on the problem and solution
theorem valid_combinations : (h * c) - r = 21 := by
  sorry

end valid_combinations_l767_76765


namespace arithmetic_sequence_theorem_l767_76737

noncomputable def S (a : ℕ → ℝ) (n : ℕ) : ℝ := (n * (a 1 + a n)) / 2

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_theorem (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_arith_seq : arithmetic_sequence a)
  (h_sum : ∀ n, S n = (n * (a 1 + a n)) / 2)
  (h_a1_pos : a 1 > 0)
  (h_condition : -1 < a 7 / a 6 ∧ a 7 / a 6 < 0) :
  (∃ d, d < 0) ∧ (∀ n, S n > 0 → n ≤ 12) :=
sorry

end arithmetic_sequence_theorem_l767_76737


namespace max_triangle_area_l767_76710

theorem max_triangle_area :
  ∃ a b c : ℝ, 0 ≤ a ∧ a ≤ 1 ∧ 1 ≤ b ∧ b ≤ 2 ∧ 2 ≤ c ∧ c ≤ 3 ∧ 
  (a + b > c ∧ a + c > b ∧ b + c > a) ∧ (1 ≤ 0.5 * a * b) := sorry

end max_triangle_area_l767_76710


namespace union_complement_set_l767_76797

open Set

variable (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)

def complement_in_U (U N : Set ℕ) : Set ℕ :=
  U \ N

theorem union_complement_set :
  U = {0, 1, 2, 4, 6, 8} →
  M = {0, 4, 6} →
  N = {0, 1, 6} →
  M ∪ (complement_in_U U N) = {0, 2, 4, 6, 8} :=
by
  intros
  rw [complement_in_U, union_comm]
  sorry

end union_complement_set_l767_76797


namespace remainder_2001_to_2005_mod_19_l767_76794

theorem remainder_2001_to_2005_mod_19 :
  (2001 * 2002 * 2003 * 2004 * 2005) % 19 = 11 :=
by
  -- Use modular arithmetic properties to convert each factor
  have h2001 : 2001 % 19 = 6 := by sorry
  have h2002 : 2002 % 19 = 7 := by sorry
  have h2003 : 2003 % 19 = 8 := by sorry
  have h2004 : 2004 % 19 = 9 := by sorry
  have h2005 : 2005 % 19 = 10 := by sorry

  -- Compute the product modulo 19
  have h_prod : (6 * 7 * 8 * 9 * 10) % 19 = 11 := by sorry

  -- Combining these results
  have h_final : ((2001 * 2002 * 2003 * 2004 * 2005) % 19) = (6 * 7 * 8 * 9 * 10) % 19 := by sorry
  exact Eq.trans h_final h_prod

end remainder_2001_to_2005_mod_19_l767_76794


namespace num_subsets_containing_6_l767_76730

open Finset

-- Define the set S
def S : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- Define the subset containing number 6
def subsets_with_6 (s : Finset ℕ) : Finset (Finset ℕ) :=
  s.powerset.filter (λ x => 6 ∈ x)

-- Theorem: The number of subsets of {1, 2, 3, 4, 5, 6} containing the number 6 is 32
theorem num_subsets_containing_6 : (subsets_with_6 S).card = 32 := by
  sorry

end num_subsets_containing_6_l767_76730


namespace monotonicity_intervals_m0_monotonicity_intervals_m_positive_intersection_points_m1_inequality_a_b_l767_76760

noncomputable def f (x m : ℝ) : ℝ := x - m * (x + 1) * Real.log (x + 1)

theorem monotonicity_intervals_m0 :
  ∀ x : ℝ, x > -1 → f x 0 = x - 0 * (x + 1) * Real.log (x + 1) ∧ f x 0 > 0 := 
sorry

theorem monotonicity_intervals_m_positive (m : ℝ) (hm : m > 0) :
  ∀ x : ℝ, x > -1 → 
  (f x m > f (x + e ^ ((1 - m) / m) - 1) m ∧ 
  f (x + e ^ ((1 - m) / m) - 1) m < f (x + e ^ ((1 - m) / m) - 1 + 1) m) :=
sorry

theorem intersection_points_m1 (t : ℝ) (hx_rng : -1 / 2 ≤ t ∧ t < 1) :
  (∃ x1 x2 : ℝ, x1 > -1/2 ∧ x1 ≤ 1 ∧ x2 > -1/2 ∧ x2 ≤ 1 ∧ f x1 1 = t ∧ f x2 1 = t) ↔ 
  (-1 / 2 + 1 / 2 * Real.log 2 ≤ t ∧ t < 0) :=
sorry

theorem inequality_a_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  (1 + a) ^ b < (1 + b) ^ a :=
sorry

end monotonicity_intervals_m0_monotonicity_intervals_m_positive_intersection_points_m1_inequality_a_b_l767_76760


namespace arithmetic_progression_squares_l767_76774

theorem arithmetic_progression_squares :
  ∃ (n : ℤ), ((3 * n^2 + 8 = 1111 * 5) ∧ (n-2, n, n+2) = (41, 43, 45)) :=
by
  sorry

end arithmetic_progression_squares_l767_76774


namespace magnitude_of_difference_is_3sqrt5_l767_76770

noncomputable def vector_a : ℝ × ℝ := (1, -2)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (x, 4)

def parallel (a b : ℝ × ℝ) : Prop := ∃ k : ℝ, a = (k * b.1, k * b.2)

theorem magnitude_of_difference_is_3sqrt5 (x : ℝ) (h_parallel : parallel vector_a (vector_b x)) :
  (Real.sqrt ((vector_a.1 - (vector_b x).1) ^ 2 + (vector_a.2 - (vector_b x).2) ^ 2)) = 3 * Real.sqrt 5 :=
sorry

end magnitude_of_difference_is_3sqrt5_l767_76770


namespace quotient_is_seven_l767_76757

def dividend : ℕ := 22
def divisor : ℕ := 3
def remainder : ℕ := 1

theorem quotient_is_seven : ∃ quotient : ℕ, dividend = (divisor * quotient) + remainder ∧ quotient = 7 := by
  sorry

end quotient_is_seven_l767_76757


namespace probability_of_winning_pair_is_correct_l767_76707

noncomputable def probability_of_winning_pair : ℚ :=
  let total_cards := 10
  let red_cards := 5
  let blue_cards := 5
  let total_ways := Nat.choose total_cards 2 -- Combination C(10,2)
  let same_color_ways := Nat.choose red_cards 2 + Nat.choose blue_cards 2 -- Combination C(5,2) for each color
  let consecutive_pairs_per_color := 4
  let consecutive_ways := 2 * consecutive_pairs_per_color -- Two colors
  let favorable_ways := same_color_ways + consecutive_ways
  favorable_ways / total_ways

theorem probability_of_winning_pair_is_correct : 
  probability_of_winning_pair = 28 / 45 := sorry

end probability_of_winning_pair_is_correct_l767_76707


namespace catering_budget_total_l767_76788

theorem catering_budget_total 
  (total_guests : ℕ)
  (guests_want_chicken guests_want_steak : ℕ)
  (cost_steak cost_chicken : ℕ) 
  (H1 : total_guests = 80)
  (H2 : guests_want_steak = 3 * guests_want_chicken)
  (H3 : cost_steak = 25)
  (H4 : cost_chicken = 18)
  (H5 : guests_want_chicken + guests_want_steak = 80) :
  (guests_want_chicken * cost_chicken + guests_want_steak * cost_steak = 1860) := 
by
  sorry

end catering_budget_total_l767_76788


namespace correct_judgments_about_f_l767_76720

-- Define the function f with its properties
variable {f : ℝ → ℝ} 

-- f is an even function
axiom even_function : ∀ x, f (-x) = f x

-- f satisfies f(x + 1) = -f(x)
axiom function_property : ∀ x, f (x + 1) = -f x

-- f is increasing on [-1, 0]
axiom increasing_on_interval : ∀ x y, -1 ≤ x → x ≤ y → y ≤ 0 → f x ≤ f y

theorem correct_judgments_about_f :
  (∀ x, f x = f (x + 2)) ∧
  (∀ x, f x = f (-x + 2)) ∧
  (f 2 = f 0) :=
by 
  sorry

end correct_judgments_about_f_l767_76720


namespace angle_difference_l767_76740

theorem angle_difference (A B : ℝ) 
  (h1 : A = 85) 
  (h2 : A + B = 180) : B - A = 10 := 
by sorry

end angle_difference_l767_76740


namespace number_of_floors_l767_76798

def hours_per_room : ℕ := 6
def hourly_rate : ℕ := 15
def total_earnings : ℕ := 3600
def rooms_per_floor : ℕ := 10

theorem number_of_floors : 
  (total_earnings / hourly_rate / hours_per_room) / rooms_per_floor = 4 := by
  sorry

end number_of_floors_l767_76798


namespace smallest_integer_solution_l767_76783

theorem smallest_integer_solution (y : ℤ) : (10 - 5 * y < 5) → y = 2 := by
  sorry

end smallest_integer_solution_l767_76783


namespace class_average_score_l767_76763

theorem class_average_score (n_boys n_girls : ℕ) (avg_score_boys avg_score_girls : ℕ) 
  (h_nb : n_boys = 12)
  (h_ng : n_girls = 4)
  (h_ab : avg_score_boys = 84)
  (h_ag : avg_score_girls = 92) : 
  (n_boys * avg_score_boys + n_girls * avg_score_girls) / (n_boys + n_girls) = 86 := 
by 
  sorry

end class_average_score_l767_76763


namespace birth_rate_calculation_l767_76748

theorem birth_rate_calculation (D : ℕ) (G : ℕ) (P : ℕ) (NetGrowth : ℕ) (B : ℕ) (h1 : D = 16) (h2 : G = 12) (h3 : P = 3000) (h4 : NetGrowth = G * P / 100) (h5 : NetGrowth = B - D) : B = 52 := by
  sorry

end birth_rate_calculation_l767_76748


namespace perpendicular_line_through_point_l767_76792

open Real

def line (a b c x y : ℝ) : Prop := a * x + b * y + c = 0

theorem perpendicular_line_through_point (x y : ℝ) (c : ℝ) :
  (line 2 1 (-5) x y) → (x = 3) ∧ (y = 0) → (line 1 (-2) 3 x y) := by
sorry

end perpendicular_line_through_point_l767_76792


namespace phone_numbers_count_l767_76755

theorem phone_numbers_count : (2^5 = 32) :=
by sorry

end phone_numbers_count_l767_76755


namespace n_gon_angles_l767_76703

theorem n_gon_angles (n : ℕ) (h1 : n > 7) (h2 : n < 12) : 
  (∃ x : ℝ, (150 * (n - 1) + x = 180 * (n - 2)) ∧ (x < 150)) :=
by {
  sorry
}

end n_gon_angles_l767_76703


namespace morgan_change_l767_76726

-- Define the costs of the items and the amount paid
def hamburger_cost : ℕ := 4
def onion_rings_cost : ℕ := 2
def smoothie_cost : ℕ := 3
def amount_paid : ℕ := 20

-- Define total cost
def total_cost := hamburger_cost + onion_rings_cost + smoothie_cost

-- Define the change received
def change_received := amount_paid - total_cost

-- Statement of the problem in Lean 4
theorem morgan_change : change_received = 11 := by
  -- include proof steps here
  sorry

end morgan_change_l767_76726


namespace value_of_expression_l767_76767

theorem value_of_expression (x y : ℝ) (hy : y > 0) (h : x = 3 * y) :
  (x^y * y^x) / (y^y * x^x) = 3^(-2 * y) := by
  sorry

end value_of_expression_l767_76767


namespace total_sales_l767_76716

theorem total_sales (S : ℕ) (h1 : (1 / 3 : ℚ) * S + (1 / 4 : ℚ) * S = (1 - (1 / 3 + 1 / 4)) * S + 15) : S = 36 :=
by
  sorry

end total_sales_l767_76716


namespace distinct_roots_of_transformed_polynomial_l767_76734

theorem distinct_roots_of_transformed_polynomial
  (a b c : ℝ)
  (h : ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
                    (a * x^5 + b * x^4 + c = 0) ∧ 
                    (a * y^5 + b * y^4 + c = 0) ∧ 
                    (a * z^5 + b * z^4 + c = 0)) :
  ∃ u v w : ℝ, u ≠ v ∧ v ≠ w ∧ u ≠ w ∧ 
               (c * u^5 + b * u + a = 0) ∧ 
               (c * v^5 + b * v + a = 0) ∧ 
               (c * w^5 + b * w + a = 0) :=
  sorry

end distinct_roots_of_transformed_polynomial_l767_76734


namespace max_hours_worked_l767_76717

theorem max_hours_worked
  (r : ℝ := 8)  -- Regular hourly rate
  (h_r : ℝ := 20)  -- Hours at regular rate
  (r_o : ℝ := r + 0.25 * r)  -- Overtime hourly rate
  (E : ℝ := 410)  -- Total weekly earnings
  : (h_r + (E - r * h_r) / r_o) = 45 :=
by
  sorry

end max_hours_worked_l767_76717


namespace age_of_B_l767_76761

theorem age_of_B (A B C : ℕ) (h1 : A = 2 * C + 2) (h2 : B = 2 * C) (h3 : A + B + C = 27) : B = 10 :=
by
  sorry

end age_of_B_l767_76761


namespace tangent_line_at_origin_l767_76778

noncomputable def f (x : ℝ) : ℝ := Real.exp x + Real.sin x

theorem tangent_line_at_origin :
  ∃ (m b : ℝ), (m = 2) ∧ (b = 1) ∧ (∀ x, f x - (m * x + b) = 0 → 2 * x - f x + 1 = 0) :=
sorry

end tangent_line_at_origin_l767_76778


namespace sqrt_mixed_number_simplify_l767_76796

open Real

theorem sqrt_mixed_number_simplify :
  sqrt (8 + 9 / 16) = sqrt 137 / 4 :=
by 
  sorry

end sqrt_mixed_number_simplify_l767_76796


namespace parallel_line_through_point_l767_76729

-- Problem: Prove the equation of the line that passes through the point (1, 1)
-- and is parallel to the line 2x - y + 1 = 0 is 2x - y - 1 = 0.

theorem parallel_line_through_point (x y : ℝ) (c : ℝ) :
  (2*x - y + 1 = 0) → (x = 1) → (y = 1) → (2*1 - 1 + c = 0) → c = -1 → (2*x - y - 1 = 0) :=
by
  sorry

end parallel_line_through_point_l767_76729


namespace minimum_value_l767_76791

noncomputable def min_expression (a b : ℝ) : ℝ :=
  a^2 + b^2 + 1 / (a + b)^2 + 1 / (a^2 * b^2)

theorem minimum_value (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) : 
  ∃ (c : ℝ), c = 2 * Real.sqrt 2 + 3 ∧ min_expression a b ≥ c :=
by
  use 2 * Real.sqrt 2 + 3
  sorry

end minimum_value_l767_76791


namespace correct_calculation_l767_76709

theorem correct_calculation : ∀ (a : ℝ), a^3 * a^2 = a^5 := 
by
  intro a
  sorry

end correct_calculation_l767_76709


namespace yoojeong_rabbits_l767_76780

theorem yoojeong_rabbits :
  ∀ (R C : ℕ), 
  let minyoung_dogs := 9
  let minyoung_cats := 3
  let minyoung_rabbits := 5
  let minyoung_total := minyoung_dogs + minyoung_cats + minyoung_rabbits
  let yoojeong_total := minyoung_total + 2
  let yoojeong_dogs := 7
  let yoojeong_cats := R - 2
  yoojeong_total = yoojeong_dogs + (R - 2) + R → 
  R = 7 :=
by
  intros R C minyoung_dogs minyoung_cats minyoung_rabbits minyoung_total yoojeong_total yoojeong_dogs yoojeong_cats
  have h1 : minyoung_total = 9 + 3 + 5 := rfl
  have h2 : yoojeong_total = minyoung_total + 2 := by sorry
  have h3 : yoojeong_dogs = 7 := rfl
  have h4 : yoojeong_cats = R - 2 := by sorry
  sorry

end yoojeong_rabbits_l767_76780


namespace even_and_period_pi_l767_76744

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 2)

theorem even_and_period_pi :
  (∀ x : ℝ, f (-x) = f x) ∧ (∃ T > 0, ∀ x : ℝ, f (x + T) = f x) ∧ T = Real.pi :=
by
  -- First, prove that f(x) is an even function: ∀ x, f(-x) = f(x)
  -- Next, find the smallest positive period T: ∃ T > 0, ∀ x, f(x + T) = f(x)
  -- Finally, show that this period is pi: T = π
  sorry

end even_and_period_pi_l767_76744


namespace evaluate_fraction_l767_76772

theorem evaluate_fraction (x y : ℝ) (h₁ : x ≠ 0) (h₂ : y ≠ 0) (h₃ : x - 1 / y ≠ 0) :
  (y - 1 / x) / (x - 1 / y) + y / x = 2 * y / x :=
by sorry

end evaluate_fraction_l767_76772


namespace find_n_l767_76795

theorem find_n (n : ℕ) (h1 : 0 ≤ n) (h2 : n ≤ 14) : n ≡ 14567 [MOD 15] → n = 2 := 
by
  sorry

end find_n_l767_76795


namespace solve_problem_l767_76753

open Real

noncomputable def problem_statement : Prop :=
  ∃ (p q : ℝ), 1 < p ∧ p < q ∧ (1 / p + 1 / q = 1) ∧ (p * q = 8) ∧ (q = 4 + 2 * sqrt 2)
  
theorem solve_problem : problem_statement :=
sorry

end solve_problem_l767_76753


namespace boat_goes_6_km_upstream_l767_76745

variable (speed_in_still_water : ℕ) (distance_downstream : ℕ) (time_downstream : ℕ) (effective_speed_downstream : ℕ) (speed_of_stream : ℕ)

-- Given conditions
def condition1 : Prop := speed_in_still_water = 11
def condition2 : Prop := distance_downstream = 16
def condition3 : Prop := time_downstream = 1
def condition4 : Prop := effective_speed_downstream = speed_in_still_water + speed_of_stream
def condition5 : Prop := effective_speed_downstream = 16

-- Prove that the boat goes 6 km against the stream in one hour.
theorem boat_goes_6_km_upstream : speed_of_stream = 5 →
  11 - 5 = 6 :=
by
  intros
  sorry

end boat_goes_6_km_upstream_l767_76745


namespace avg_and_variance_decrease_l767_76777

noncomputable def original_heights : List ℝ := [180, 184, 188, 190, 192, 194]
noncomputable def new_heights : List ℝ := [180, 184, 188, 190, 192, 188]

noncomputable def avg (heights : List ℝ) : ℝ :=
  heights.sum / heights.length

noncomputable def variance (heights : List ℝ) (mean : ℝ) : ℝ :=
  (heights.map (λ h => (h - mean) ^ 2)).sum / heights.length

theorem avg_and_variance_decrease :
  let original_mean := avg original_heights
  let new_mean := avg new_heights
  let original_variance := variance original_heights original_mean
  let new_variance := variance new_heights new_mean
  new_mean < original_mean ∧ new_variance < original_variance :=
by
  sorry

end avg_and_variance_decrease_l767_76777


namespace parabola_tangents_coprime_l767_76771

theorem parabola_tangents_coprime {d e f : ℤ} (hd : d ≠ 0) (he : e ≠ 0)
  (h_coprime: Int.gcd (Int.gcd d e) f = 1)
  (h_tangent1 : d^2 - 4 * e * (2 * e - f) = 0)
  (h_tangent2 : (e + d)^2 - 4 * d * (8 * d - f) = 0) :
  d + e + f = 8 := by
  sorry

end parabola_tangents_coprime_l767_76771


namespace maximum_value_of_expression_l767_76787

noncomputable def max_value (x y z w : ℝ) : ℝ := 2 * x + 3 * y + 5 * z - 4 * w

theorem maximum_value_of_expression 
  (x y z w : ℝ)
  (h : 9 * x^2 + 4 * y^2 + 25 * z^2 + 16 * w^2 = 4) : 
  max_value x y z w ≤ 6 * Real.sqrt 6 :=
sorry

end maximum_value_of_expression_l767_76787


namespace weeks_to_buy_iphone_l767_76743

-- Definitions based on conditions
def iphone_cost : ℝ := 800
def trade_in_value : ℝ := 240
def earnings_per_week : ℝ := 80

-- Mathematically equivalent proof problem
theorem weeks_to_buy_iphone : 
  ∀ (iphone_cost trade_in_value earnings_per_week : ℝ), 
  (iphone_cost - trade_in_value) / earnings_per_week = 7 :=
by
  -- Using the given conditions directly.
  intros iphone_cost trade_in_value earnings_per_week
  sorry

end weeks_to_buy_iphone_l767_76743


namespace infinite_sequence_domain_l767_76701

def seq_domain (f : ℕ → ℕ) : Set ℕ := {n | 0 < n}

theorem infinite_sequence_domain (f : ℕ → ℕ) (a_n : ℕ → ℕ)
   (h : ∀ (n : ℕ), a_n n = f n) : 
   seq_domain f = {n | 0 < n} :=
sorry

end infinite_sequence_domain_l767_76701


namespace max_sum_abc_l767_76742

theorem max_sum_abc
  (a b c : ℤ)
  (A : Matrix (Fin 2) (Fin 2) ℚ)
  (hA1 : A = (1/7 : ℚ) • ![![(-5 : ℚ), a], ![b, c]])
  (hA2 : A * A = 2 • (1 : Matrix (Fin 2) (Fin 2) ℚ)) :
  a + b + c ≤ 79 :=
by
  sorry

end max_sum_abc_l767_76742


namespace geometric_progression_common_ratio_l767_76715

theorem geometric_progression_common_ratio (y r : ℝ) (h : (40 + y)^2 = (10 + y) * (90 + y)) :
  r = (40 + y) / (10 + y) → r = (90 + y) / (40 + y) → r = 5 / 3 :=
by
  sorry

end geometric_progression_common_ratio_l767_76715


namespace determine_x_l767_76733

variable (A B C x : ℝ)
variable (hA : A = x)
variable (hB : B = 2 * x)
variable (hC : C = 45)
variable (hSum : A + B + C = 180)

theorem determine_x : x = 45 := 
by
  -- proof steps would go here
  sorry

end determine_x_l767_76733


namespace problem_1_problem_2_problem_3_l767_76725

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x
noncomputable def g (x : ℝ) : ℝ := f (x + Real.pi / 4) + f (x + 3 * Real.pi / 4)

theorem problem_1 : f (Real.pi / 2) = 1 := 
sorry

theorem problem_2 : (∃ p > 0, ∀ x, f (x + p) = f x) ∧ (∀ p, p > 0 ∧ (∀ x, f (x + p) = f x) → p ≥ 2 * Real.pi) := 
sorry

theorem problem_3 : ∃ x : ℝ, g x = -2 := 
sorry

end problem_1_problem_2_problem_3_l767_76725


namespace find_n_l767_76718

open Nat

def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p

def twin_primes (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ q = p + 2

def is_twins_prime_sum (n p q : ℕ) : Prop :=
  twin_primes p q ∧ is_prime (2^n + p) ∧ is_prime (2^n + q)

theorem find_n :
  ∀ (n : ℕ), (∃ (p q : ℕ), is_twins_prime_sum n p q) → (n = 1 ∨ n = 3) :=
sorry

end find_n_l767_76718


namespace sequence_formula_l767_76721

theorem sequence_formula (a : ℕ → ℝ) (h1 : a 1 = -2) (h2 : a 2 = -1.2) :
  ∀ n, a n = 0.8 * n - 2.8 :=
by
  sorry

end sequence_formula_l767_76721


namespace probability_red_side_given_observed_l767_76768

def total_cards : ℕ := 9
def black_black_cards : ℕ := 4
def black_red_cards : ℕ := 2
def red_red_cards : ℕ := 3

def red_sides : ℕ := red_red_cards * 2 + black_red_cards
def red_red_sides : ℕ := red_red_cards * 2
def probability_other_side_is_red (total_red_sides red_red_sides : ℕ) : ℚ :=
  red_red_sides / total_red_sides

theorem probability_red_side_given_observed :
  probability_other_side_is_red red_sides red_red_sides = 3 / 4 :=
by
  unfold red_sides
  unfold red_red_sides
  unfold probability_other_side_is_red
  sorry

end probability_red_side_given_observed_l767_76768


namespace ratio_soda_water_l767_76700

variables (W S : ℕ) (k : ℕ)

-- Conditions of the problem
def condition1 : Prop := S = k * W - 6
def condition2 : Prop := W + S = 54
def positive_integer_k : Prop := k > 0

-- The theorem we want to prove
theorem ratio_soda_water (h1 : condition1 W S k) (h2 : condition2 W S) (h3 : positive_integer_k k) : S / gcd S W = 4 ∧ W / gcd S W = 5 :=
sorry

end ratio_soda_water_l767_76700


namespace jake_total_work_hours_l767_76714

def initial_debt_A := 150
def payment_A := 60
def hourly_rate_A := 15
def remaining_debt_A := initial_debt_A - payment_A
def hours_to_work_A := remaining_debt_A / hourly_rate_A

def initial_debt_B := 200
def payment_B := 80
def hourly_rate_B := 20
def remaining_debt_B := initial_debt_B - payment_B
def hours_to_work_B := remaining_debt_B / hourly_rate_B

def initial_debt_C := 250
def payment_C := 100
def hourly_rate_C := 25
def remaining_debt_C := initial_debt_C - payment_C
def hours_to_work_C := remaining_debt_C / hourly_rate_C

def total_hours_to_work := hours_to_work_A + hours_to_work_B + hours_to_work_C

theorem jake_total_work_hours :
  total_hours_to_work = 18 :=
sorry

end jake_total_work_hours_l767_76714


namespace value_of_frac_l767_76793

theorem value_of_frac (x y z w : ℕ) 
  (hz : z = 5 * w) 
  (hy : y = 3 * z) 
  (hx : x = 4 * y) : 
  x * z / (y * w) = 20 := 
  sorry

end value_of_frac_l767_76793


namespace simplify_expression_l767_76736

theorem simplify_expression (y : ℝ) : 7 * y + 8 - 3 * y + 16 = 4 * y + 24 :=
by
  sorry

end simplify_expression_l767_76736


namespace profit_made_after_two_years_l767_76758

variable (present_value : ℝ) (depreciation_rate : ℝ) (selling_price : ℝ) 

def value_after_one_year (present_value depreciation_rate : ℝ) : ℝ :=
  present_value - (depreciation_rate * present_value)

def value_after_two_years (value_after_one_year : ℝ) (depreciation_rate : ℝ) : ℝ :=
  value_after_one_year - (depreciation_rate * value_after_one_year)

def profit (selling_price value_after_two_years : ℝ) : ℝ :=
  selling_price - value_after_two_years

theorem profit_made_after_two_years
  (h_present_value : present_value = 150000)
  (h_depreciation_rate : depreciation_rate = 0.22)
  (h_selling_price : selling_price = 115260) :
  profit selling_price (value_after_two_years (value_after_one_year present_value depreciation_rate) depreciation_rate) = 24000 := 
by
  sorry

end profit_made_after_two_years_l767_76758


namespace no_solution_of_abs_sum_l767_76738

theorem no_solution_of_abs_sum (a : ℝ) : (∀ x : ℝ, |x - 2| + |x + 3| < a → false) ↔ a ≤ 5 := sorry

end no_solution_of_abs_sum_l767_76738


namespace number_difference_l767_76746

theorem number_difference (a b : ℕ) (h1 : a + b = 25650) (h2 : a % 100 = 0) (h3 : b = a / 100) :
  a - b = 25146 :=
sorry

end number_difference_l767_76746


namespace smallest_hamburger_packages_l767_76779

theorem smallest_hamburger_packages (h_num : ℕ) (b_num : ℕ) (h_bag_num : h_num = 10) (b_bag_num : b_num = 15) :
  ∃ (n : ℕ), n = 3 ∧ (n * h_num) = (2 * b_num) := by
  sorry

end smallest_hamburger_packages_l767_76779


namespace triangle_angles_inequality_l767_76786

theorem triangle_angles_inequality (A B C : ℝ) (h1 : A + B + C = Real.pi) (h2 : 0 < A) (h3 : 0 < B) (h4 : 0 < C) 
(h5 : A < Real.pi) (h6 : B < Real.pi) (h7 : C < Real.pi) : 
  A * Real.cos B + Real.sin A * Real.sin C > 0 := 
by 
  sorry

end triangle_angles_inequality_l767_76786


namespace emily_total_cost_l767_76712

-- Definition of the monthly cell phone plan costs and usage details
def base_cost : ℝ := 30
def cost_per_text : ℝ := 0.10
def cost_per_extra_minute : ℝ := 0.15
def cost_per_extra_gb : ℝ := 5
def free_hours : ℝ := 25
def free_gb : ℝ := 15
def texts : ℝ := 150
def hours : ℝ := 26
def gb : ℝ := 16

-- Calculate the total cost
def total_cost : ℝ :=
  base_cost +
  (texts * cost_per_text) +
  ((hours - free_hours) * 60 * cost_per_extra_minute) +
  ((gb - free_gb) * cost_per_extra_gb)

-- The proof statement that Emily had to pay $59
theorem emily_total_cost :
  total_cost = 59 := by
  sorry

end emily_total_cost_l767_76712


namespace meaningful_if_and_only_if_l767_76731

theorem meaningful_if_and_only_if (x : ℝ) : (∃ y : ℝ, y = (1 / (x - 1))) ↔ x ≠ 1 :=
by 
  sorry

end meaningful_if_and_only_if_l767_76731


namespace cost_of_double_room_l767_76764

theorem cost_of_double_room (total_rooms : ℕ) (cost_single_room : ℕ) (total_revenue : ℕ) 
  (double_rooms_booked : ℕ) (single_rooms_booked := total_rooms - double_rooms_booked) 
  (total_single_revenue := single_rooms_booked * cost_single_room) : 
  total_rooms = 260 → cost_single_room = 35 → total_revenue = 14000 → double_rooms_booked = 196 → 
  196 * 60 + 64 * 35 = total_revenue :=
by
  intros h1 h2 h3 h4
  sorry

end cost_of_double_room_l767_76764


namespace poly_divisible_by_seven_l767_76719

-- Define the given polynomial expression
def poly_expr (x n : ℕ) : ℕ := (1 + x)^n - 1

-- Define the proof statement
theorem poly_divisible_by_seven :
  ∀ x n : ℕ, x = 5 ∧ n = 4 → poly_expr x n % 7 = 0 :=
by
  intro x n h
  cases h
  sorry

end poly_divisible_by_seven_l767_76719


namespace min_x2_y2_l767_76747

theorem min_x2_y2 (x y : ℝ) (h : 2 * (x^2 + y^2) = x^2 + y + x * y) : 
  (∃ x y, x = 0 ∧ y = 0) ∨ x^2 + y^2 >= 1 := 
sorry

end min_x2_y2_l767_76747


namespace dog_weight_ratio_l767_76702

theorem dog_weight_ratio :
  ∀ (brown black white grey : ℕ),
    brown = 4 →
    black = brown + 1 →
    grey = black - 2 →
    (brown + black + white + grey) / 4 = 5 →
    white / brown = 2 :=
by
  intros brown black white grey h_brown h_black h_grey h_avg
  sorry

end dog_weight_ratio_l767_76702


namespace distinct_rational_numbers_count_l767_76784

theorem distinct_rational_numbers_count :
  ∃ N : ℕ, 
    (N = 49) ∧
    ∀ (k : ℚ), |k| < 50 →
      (∃ x : ℤ, x^2 - k * x + 18 = 0) →
        ∃ m: ℤ, k = 2 * m ∧ |m| < 25 :=
sorry

end distinct_rational_numbers_count_l767_76784


namespace no_strictly_greater_polynomials_l767_76776

noncomputable def transformation (P : Polynomial ℝ) (k : ℕ) (a : ℝ) : Polynomial ℝ := 
  P + Polynomial.monomial k (2 * a) - Polynomial.monomial (k + 1) a

theorem no_strictly_greater_polynomials (P Q : Polynomial ℝ) 
  (H1 : ∃ (n : ℕ) (a : ℝ), Q = transformation P n a)
  (H2 : ∃ (n : ℕ) (a : ℝ), P = transformation Q n a) : 
  ∃ x : ℝ, P.eval x = Q.eval x :=
sorry

end no_strictly_greater_polynomials_l767_76776


namespace zero_lies_in_interval_l767_76750

def f (x : ℝ) : ℝ := -|x - 5| + 2 * x - 1

theorem zero_lies_in_interval (k : ℤ) (h : ∃ x : ℝ, k < x ∧ x < k + 1 ∧ f x = 0) : k = 2 := 
sorry

end zero_lies_in_interval_l767_76750


namespace find_X_value_l767_76769

-- Given definitions and conditions
def X (n : ℕ) : ℕ := 3 + 2 * (n - 1)
def S (n : ℕ) : ℕ := n * (n + 2)

-- Proposition we need to prove
theorem find_X_value : ∃ n : ℕ, S n ≥ 10000 ∧ X n = 201 :=
by
  -- Placeholder for proof
  sorry

end find_X_value_l767_76769


namespace consecutive_rolls_probability_l767_76766

theorem consecutive_rolls_probability : 
  let total_outcomes := 36
  let consecutive_events := 10
  (consecutive_events / total_outcomes : ℚ) = 5 / 18 :=
by
  sorry

end consecutive_rolls_probability_l767_76766


namespace play_area_l767_76735

theorem play_area (posts : ℕ) (space : ℝ) (extra_posts : ℕ) (short_posts long_posts : ℕ) (short_spaces long_spaces : ℕ) 
  (short_length long_length area : ℝ)
  (h1 : posts = 24) 
  (h2 : space = 5)
  (h3 : extra_posts = 6)
  (h4 : long_posts = short_posts + extra_posts)
  (h5 : 2 * short_posts + 2 * long_posts - 4 = posts)
  (h6 : short_spaces = short_posts - 1)
  (h7 : long_spaces = long_posts - 1)
  (h8 : short_length = short_spaces * space)
  (h9 : long_length = long_spaces * space)
  (h10 : area = short_length * long_length) :
  area = 675 := 
sorry

end play_area_l767_76735


namespace train_distance_30_minutes_l767_76789

theorem train_distance_30_minutes (h : ∀ (t : ℝ), 0 < t → (1 / 2) * t = 1 / 2 * t) : 
  (1 / 2) * 30 = 15 :=
by
  sorry

end train_distance_30_minutes_l767_76789


namespace actual_average_speed_l767_76723

theorem actual_average_speed 
  (v t : ℝ)
  (h : v * t = (v + 21) * (2/3) * t) : 
  v = 42 :=
by
  sorry

end actual_average_speed_l767_76723


namespace rectangle_perimeter_l767_76790

theorem rectangle_perimeter (a b : ℕ) (h1 : a ≠ b) (h2 : a * b = 2 * (2 * a + 2 * b)) : 2 * (a + b) = 36 :=
by
  sorry

end rectangle_perimeter_l767_76790


namespace gain_percent_l767_76722

theorem gain_percent (C S S_d : ℝ) 
  (h1 : 50 * C = 20 * S) 
  (h2 : S_d = S * (1 - 0.15)) : 
  ((S_d - C) / C) * 100 = 112.5 := 
by 
  sorry

end gain_percent_l767_76722


namespace find_m_plus_n_l767_76713

theorem find_m_plus_n (PQ QR RP : ℕ) (x y : ℕ) 
  (h1 : PQ = 26) 
  (h2 : QR = 29) 
  (h3 : RP = 25) 
  (h4 : PQ = x + y) 
  (h5 : QR = x + (QR - x))
  (h6 : RP = x + (RP - x)) : 
  30 = 29 + 1 :=
by
  -- assumptions already provided in problem statement
  sorry

end find_m_plus_n_l767_76713


namespace bellas_score_l767_76799

theorem bellas_score (sum_19 : ℕ) (sum_20 : ℕ) (avg_19 : ℕ) (avg_20 : ℕ) (n_19 : ℕ) (n_20 : ℕ) :
  avg_19 = 82 → avg_20 = 85 → n_19 = 19 → n_20 = 20 → sum_19 = n_19 * avg_19 → sum_20 = n_20 * avg_20 →
  sum_20 - sum_19 = 142 := 
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end bellas_score_l767_76799


namespace total_flowers_l767_76756

theorem total_flowers (initial_rosas_flowers andre_gifted_flowers : ℝ) 
  (h1 : initial_rosas_flowers = 67.0) 
  (h2 : andre_gifted_flowers = 90.0) : 
  initial_rosas_flowers + andre_gifted_flowers = 157.0 :=
  by
  sorry

end total_flowers_l767_76756


namespace trig_expression_simplification_l767_76724

theorem trig_expression_simplification :
  ∃ a b : ℕ, 
  0 < b ∧ b < 90 ∧ 
  (1000 * Real.sin (10 * Real.pi / 180) * Real.cos (20 * Real.pi / 180) * Real.cos (30 * Real.pi / 180) * Real.cos (40 * Real.pi / 180) = ↑a * Real.sin (b * Real.pi / 180)) ∧ 
  (100 * a + b = 12560) :=
sorry

end trig_expression_simplification_l767_76724


namespace weight_of_new_person_l767_76785

variable (avg_increase : ℝ) (n_persons : ℕ) (weight_replaced : ℝ)

theorem weight_of_new_person (h1 : avg_increase = 3.5) (h2 : n_persons = 8) (h3 : weight_replaced = 65) :
  let total_weight_increase := n_persons * avg_increase
  let weight_new := weight_replaced + total_weight_increase
  weight_new = 93 := by
  sorry

end weight_of_new_person_l767_76785


namespace parabola_focus_l767_76727

-- Definitions and conditions from the original problem
def parabola_eq (x y : ℝ) : Prop := x^2 = (1/2) * y 

-- Define the problem to prove the coordinates of the focus
theorem parabola_focus (x y : ℝ) (h : parabola_eq x y) : (x = 0 ∧ y = 1/8) :=
sorry

end parabola_focus_l767_76727


namespace arithmetic_sum_sequences_l767_76708

theorem arithmetic_sum_sequences (a b : ℕ → ℕ) (h1 : ∀ n, a n = a 0 + n * (a 1 - a 0)) (h2 : ∀ n, b n = b 0 + n * (b 1 - b 0)) (h3 : a 2 + b 2 = 3) (h4 : a 4 + b 4 = 5): a 7 + b 7 = 8 := by
  sorry

end arithmetic_sum_sequences_l767_76708


namespace initially_calculated_average_l767_76732

theorem initially_calculated_average 
  (correct_sum : ℤ)
  (incorrect_diff : ℤ)
  (num_numbers : ℤ)
  (correct_average : ℤ)
  (h1 : correct_sum = correct_average * num_numbers)
  (h2 : incorrect_diff = 20)
  (h3 : num_numbers = 10)
  (h4 : correct_average = 18) :
  (correct_sum - incorrect_diff) / num_numbers = 16 := by
  sorry

end initially_calculated_average_l767_76732


namespace volume_of_rectangular_solid_l767_76706

theorem volume_of_rectangular_solid (a b c : ℝ) (h1 : a * b = Real.sqrt 2) (h2 : b * c = Real.sqrt 3) (h3 : c * a = Real.sqrt 6) : a * b * c = Real.sqrt 6 :=
sorry

end volume_of_rectangular_solid_l767_76706


namespace initial_speed_of_car_l767_76754

-- Definition of conditions
def distance_from_A_to_B := 100  -- km
def time_remaining_first_reduction := 30 / 60  -- hours
def speed_reduction_first := 10  -- km/h
def time_remaining_second_reduction := 20 / 60  -- hours
def speed_reduction_second := 10  -- km/h
def additional_time_reduced_speeds := 5 / 60  -- hours

-- Variables for initial speed and intermediate distances
variables (v x : ℝ)

-- Proposition to prove the initial speed
theorem initial_speed_of_car :
  (100 - (v / 2 + x + 20)) / v + 
  (v / 2) / (v - 10) + 
  20 / (v - 20) - 
  20 / (v - 10) 
  = 5 / 60 →
  v = 100 :=
by
  sorry

end initial_speed_of_car_l767_76754


namespace solve_inequalities_l767_76739

theorem solve_inequalities :
  {x : ℝ | 4 ≤ (2*x) / (3*x - 7) ∧ (2*x) / (3*x - 7) < 9} = {x : ℝ | (63 / 25) < x ∧ x ≤ 2.8} :=
by
  sorry

end solve_inequalities_l767_76739


namespace add_in_base_7_l767_76711

theorem add_in_base_7 (X Y : ℕ) (h1 : (X + 5) % 7 = 0) (h2 : (Y + 2) % 7 = X) : X + Y = 2 :=
by
  sorry

end add_in_base_7_l767_76711


namespace intersection_M_N_l767_76728

-- Definitions based on the conditions
def M : Set ℝ := {x | x ≥ 2}
def N : Set ℝ := {x | x^2 - 25 < 0}

-- Theorem asserting the intersection of sets M and N
theorem intersection_M_N : M ∩ N = {x | 2 ≤ x ∧ x < 5} := 
by
  sorry

end intersection_M_N_l767_76728
