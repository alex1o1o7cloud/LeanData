import Mathlib

namespace TruckY_average_speed_is_63_l887_88783

noncomputable def average_speed_TruckY (initial_gap : ℕ) (extra_distance : ℕ) (hours : ℕ) (distance_X_per_hour : ℕ) : ℕ :=
  let distance_X := distance_X_per_hour * hours
  let total_distance_Y := distance_X + initial_gap + extra_distance
  total_distance_Y / hours

theorem TruckY_average_speed_is_63 
  (initial_gap : ℕ := 14) 
  (extra_distance : ℕ := 4) 
  (hours : ℕ := 3)
  (distance_X_per_hour : ℕ := 57) : 
  average_speed_TruckY initial_gap extra_distance hours distance_X_per_hour = 63 :=
by
  -- Proof goes here
  sorry

end TruckY_average_speed_is_63_l887_88783


namespace eq_of_symmetric_translation_l887_88723

noncomputable def parabola (x : ℝ) : ℝ := 2 * x^2 - 4 * x - 5

noncomputable def translate_left (f : ℝ → ℝ) (k : ℝ) (x : ℝ) : ℝ := f (x + k)

noncomputable def translate_up (g : ℝ → ℝ) (k : ℝ) (x : ℝ) : ℝ := g x + k

noncomputable def translate_parabola (x : ℝ) : ℝ := translate_up (translate_left parabola 3) 2 x

noncomputable def symmetric_parabola (h : ℝ → ℝ) (x : ℝ) : ℝ := h (-x)

theorem eq_of_symmetric_translation :
  symmetric_parabola translate_parabola x = 2 * x^2 - 8 * x + 3 :=
by
  sorry

end eq_of_symmetric_translation_l887_88723


namespace cost_price_of_article_l887_88745

theorem cost_price_of_article (C : ℝ) (h1 : 86 - C = C - 42) : C = 64 :=
by
  sorry

end cost_price_of_article_l887_88745


namespace find_AC_l887_88757

def vector_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 + v2.1, v1.2 + v2.2)

theorem find_AC :
  let AB := (2, 3)
  let BC := (1, -4)
  vector_add AB BC = (3, -1) :=
by 
  sorry

end find_AC_l887_88757


namespace parabola_constant_term_l887_88743

theorem parabola_constant_term
  (a b c : ℝ)
  (h1 : ∀ x, (-2 * (x - 1)^2 + 3) = a * x^2 + b * x + c ) :
  c = 2 :=
sorry

end parabola_constant_term_l887_88743


namespace denominator_of_first_fraction_l887_88724

theorem denominator_of_first_fraction (y x : ℝ) (h : y > 0) (h_eq : (9 * y) / x + (3 * y) / 10 = 0.75 * y) : x = 20 :=
by
  sorry

end denominator_of_first_fraction_l887_88724


namespace money_lent_to_C_l887_88759

theorem money_lent_to_C (X : ℝ) (interest_rate : ℝ) (P_b : ℝ) (T_b : ℝ) (T_c : ℝ) (total_interest : ℝ) :
  interest_rate = 0.09 →
  P_b = 5000 →
  T_b = 2 →
  T_c = 4 →
  total_interest = 1980 →
  (P_b * interest_rate * T_b + X * interest_rate * T_c = total_interest) →
  X = 500 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end money_lent_to_C_l887_88759


namespace point_on_y_axis_l887_88793

theorem point_on_y_axis (a : ℝ) :
  (a + 2 = 0) -> a = -2 :=
by
  intro h
  sorry

end point_on_y_axis_l887_88793


namespace domain_of_sqrt_expression_l887_88787

theorem domain_of_sqrt_expression :
  {x : ℝ | x^2 - 5 * x - 6 ≥ 0} = {x : ℝ | x ≤ -1} ∪ {x : ℝ | x ≥ 6} := by
sorry

end domain_of_sqrt_expression_l887_88787


namespace floor_sqrt_80_eq_8_l887_88750

theorem floor_sqrt_80_eq_8 (h1: 8 * 8 = 64) (h2: 9 * 9 = 81) (h3: 8 < Real.sqrt 80) (h4: Real.sqrt 80 < 9) :
  Int.floor (Real.sqrt 80) = 8 :=
sorry

end floor_sqrt_80_eq_8_l887_88750


namespace find_x_l887_88722

theorem find_x (x n q r : ℕ) (h_n : n = 220080) (h_sum : n = (x + 445) * (2 * (x - 445)) + r) (h_r : r = 80) : 
  x = 555 :=
by
  have eq1 : n = 220080 := h_n
  have eq2 : n =  (x + 445) * (2 * (x - 445)) + r := h_sum
  have eq3 : r = 80 := h_r
  sorry

end find_x_l887_88722


namespace total_spent_on_toys_and_clothes_l887_88751

def cost_toy_cars : ℝ := 14.88
def cost_skateboard : ℝ := 4.88
def cost_toy_trucks : ℝ := 5.86
def cost_pants : ℝ := 14.55
def cost_shirt : ℝ := 7.43
def cost_hat : ℝ := 12.50

theorem total_spent_on_toys_and_clothes :
  (cost_toy_cars + cost_skateboard + cost_toy_trucks) + (cost_pants + cost_shirt + cost_hat) = 60.10 :=
by
  sorry

end total_spent_on_toys_and_clothes_l887_88751


namespace proof_C_D_values_l887_88718

-- Given the conditions
def denominator_factorization (x : ℝ) : Prop :=
  3 * x ^ 2 - x - 14 = (3 * x + 7) * (x - 2)

def fraction_equality (x : ℝ) (C D : ℝ) : Prop :=
  (3 * x ^ 2 + 7 * x - 20) / (3 * x ^ 2 - x - 14) =
  C / (x - 2) + D / (3 * x + 7)

-- The values to be proven
def values_C_D : Prop :=
  ∃ C D : ℝ, C = -14 / 13 ∧ D = 81 / 13 ∧ ∀ x : ℝ, (denominator_factorization x → fraction_equality x C D)

theorem proof_C_D_values : values_C_D :=
sorry

end proof_C_D_values_l887_88718


namespace selection_schemes_l887_88748

theorem selection_schemes (boys girls : ℕ) (hb : boys = 4) (hg : girls = 2) :
  (boys * girls = 8) :=
by
  -- Proof goes here
  intros
  sorry

end selection_schemes_l887_88748


namespace room_height_l887_88702

-- Define the conditions
def total_curtain_length : ℕ := 101
def extra_material : ℕ := 5

-- Define the statement to be proven
theorem room_height : total_curtain_length - extra_material = 96 :=
by
  sorry

end room_height_l887_88702


namespace mean_of_two_remaining_numbers_l887_88754

theorem mean_of_two_remaining_numbers (a b c: ℝ) (h1: (a + b + c + 100) / 4 = 90) (h2: a = 70) : (b + c) / 2 = 95 := by
  sorry

end mean_of_two_remaining_numbers_l887_88754


namespace parallelLines_perpendicularLines_l887_88765

-- Problem A: Parallel lines
theorem parallelLines (a : ℝ) : 
  (∀x y : ℝ, y = -x + 2 * a → y = (a^2 - 2) * x + 2 → -1 = a^2 - 2) → 
  a = -1 := 
sorry

-- Problem B: Perpendicular lines
theorem perpendicularLines (a : ℝ) : 
  (∀x y : ℝ, y = (2 * a - 1) * x + 3 → y = 4 * x - 3 → (2 * a - 1) * 4 = -1) →
  a = 3 / 8 := 
sorry

end parallelLines_perpendicularLines_l887_88765


namespace brendan_match_ratio_l887_88760

noncomputable def brendanMatches (totalMatches firstRound secondRound matchesWonFirstTwoRounds matchesWonTotal matchesInLastRound : ℕ) :=
  matchesWonFirstTwoRounds = firstRound + secondRound ∧
  matchesWonFirstTwoRounds = 12 ∧
  totalMatches = matchesWonTotal ∧
  matchesWonTotal = 14 ∧
  firstRound = 6 ∧
  secondRound = 6 ∧
  matchesInLastRound = 4

theorem brendan_match_ratio :
  ∃ ratio: ℕ × ℕ,
    let firstRound := 6
    let secondRound := 6
    let matchesInLastRound := 4
    let matchesWonFirstTwoRounds := firstRound + secondRound
    let matchesWonTotal := 14
    let matchesWonLastRound := matchesWonTotal - matchesWonFirstTwoRounds
    let ratio := (matchesWonLastRound, matchesInLastRound)
    brendanMatches matchesWonTotal firstRound secondRound matchesWonFirstTwoRounds matchesWonTotal matchesInLastRound ∧
    ratio = (1, 2) :=
by
  sorry

end brendan_match_ratio_l887_88760


namespace max_non_overlapping_areas_l887_88721

theorem max_non_overlapping_areas (n : ℕ) (h : 0 < n) :
  ∃ k : ℕ, k = 4 * n + 1 :=
sorry

end max_non_overlapping_areas_l887_88721


namespace game_of_24_l887_88713

theorem game_of_24 : 
  let a := 3
  let b := -5
  let c := 6
  let d := -8
  ((b + c / a) * d = 24) :=
by
  let a := 3
  let b := -5
  let c := 6
  let d := -8
  show (b + c / a) * d = 24
  sorry

end game_of_24_l887_88713


namespace inequality_proof_l887_88797

theorem inequality_proof (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum : a + b + c = 3) :
    (a^2 + 6) / (2*a^2 + 2*b^2 + 2*c^2 + 2*a - 1) +
    (b^2 + 6) / (2*a^2 + 2*b^2 + 2*c^2 + 2*b - 1) +
    (c^2 + 6) / (2*a^2 + 2*b^2 + 2*c^2 + 2*c - 1) ≤
    3 := by
  sorry

end inequality_proof_l887_88797


namespace box_dimensions_l887_88703

theorem box_dimensions (a b c : ℕ) (h1 : a + c = 17) (h2 : a + b = 13) (h3 : b + c = 20) :
  a = 5 ∧ b = 8 ∧ c = 12 :=
by
  sorry

end box_dimensions_l887_88703


namespace system_sol_l887_88777

theorem system_sol {x y : ℝ} (h1 : x + 2 * y = -1) (h2 : 2 * x + y = 3) : x - y = 4 := by
  sorry

end system_sol_l887_88777


namespace rainfall_ratio_l887_88752

noncomputable def total_rainfall := 35
noncomputable def rainfall_second_week := 21

theorem rainfall_ratio 
  (R1 R2 : ℝ)
  (hR2 : R2 = rainfall_second_week)
  (hTotal : R1 + R2 = total_rainfall) :
  R2 / R1 = 3 / 2 := 
by 
  sorry

end rainfall_ratio_l887_88752


namespace molecular_weight_correct_l887_88763

noncomputable def molecular_weight_compound : ℝ :=
  (3 * 12.01) + (6 * 1.008) + (1 * 16.00)

theorem molecular_weight_correct :
  molecular_weight_compound = 58.078 := by
  sorry

end molecular_weight_correct_l887_88763


namespace new_persons_joined_l887_88747

theorem new_persons_joined :
  ∀ (A : ℝ) (N : ℕ) (avg_new : ℝ) (avg_combined : ℝ), 
  N = 15 → avg_new = 15 → avg_combined = 15.5 → 1 = (N * avg_combined + N * avg_new - 232.5) / (avg_combined - avg_new) := by
  intros A N avg_new avg_combined
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end new_persons_joined_l887_88747


namespace fraction_of_arith_geo_seq_l887_88780

theorem fraction_of_arith_geo_seq (a : ℕ → ℝ) (d : ℝ) (h_d : d ≠ 0)
  (h_seq_arith : ∀ n, a (n+1) = a n + d)
  (h_seq_geo : (a 1 + 2 * d)^2 = a 1 * (a 1 + 8 * d)) :
  (a 1 + a 3 + a 9) / (a 2 + a 4 + a 10) = 13 / 16 :=
by
  sorry

end fraction_of_arith_geo_seq_l887_88780


namespace only_powers_of_2_satisfy_condition_l887_88701

theorem only_powers_of_2_satisfy_condition:
  ∀ (n : ℕ), n ≥ 2 →
  (∃ (x : ℕ → ℕ), 
    ∀ (i j : ℕ), 
      0 < i ∧ i < n → 0 < j ∧ j < n → i ≠ j ∧ (n ∣ (2 * i + j)) → x i < x j) ↔
      ∃ (s : ℕ), n = 2^s ∧ s ≥ 1 :=
by
  sorry

end only_powers_of_2_satisfy_condition_l887_88701


namespace fraction_of_sum_l887_88734

theorem fraction_of_sum (S n : ℝ) (h1 : n = S / 6) : n / (S + n) = 1 / 7 :=
by sorry

end fraction_of_sum_l887_88734


namespace mutually_exclusive_A_C_l887_88794

-- Definitions based on the given conditions
def all_not_defective (A : Prop) : Prop := A
def all_defective (B : Prop) : Prop := B
def at_least_one_defective (C : Prop) : Prop := C

-- Theorem to prove A and C are mutually exclusive
theorem mutually_exclusive_A_C (A B C : Prop) 
  (H1 : all_not_defective A) 
  (H2 : all_defective B) 
  (H3 : at_least_one_defective C) : 
  (A ∧ C) → False :=
sorry

end mutually_exclusive_A_C_l887_88794


namespace eating_possible_values_l887_88790

def A : Set ℝ := {-1, 1 / 2, 1}
def B (a : ℝ) : Set ℝ := {x : ℝ | a * x ^ 2 = 1 ∧ a ≥ 0}

-- Full eating relation: B ⊆ A
-- Partial eating relation: (B ∩ A).Nonempty ∧ ¬(B ⊆ A ∨ A ⊆ B)

def is_full_eating (a : ℝ) : Prop := B a ⊆ A
def is_partial_eating (a : ℝ) : Prop :=
  (B a ∩ A).Nonempty ∧ ¬(B a ⊆ A ∨ A ⊆ B a)

theorem eating_possible_values :
  {a : ℝ | is_full_eating a ∨ is_partial_eating a} = {0, 1, 4} :=
by
  sorry

end eating_possible_values_l887_88790


namespace sum_possible_values_A_B_l887_88762

theorem sum_possible_values_A_B : 
  ∀ (A B : ℕ), 
  (0 ≤ A ∧ A ≤ 9) ∧ 
  (0 ≤ B ∧ B ≤ 9) ∧ 
  ∃ k : ℕ, 28 + A + B = 9 * k 
  → (A + B = 8 ∨ A + B = 17) 
  → A + B = 25 :=
by
  sorry

end sum_possible_values_A_B_l887_88762


namespace discount_per_coupon_l887_88766

-- Definitions and conditions from the problem
def num_cans : ℕ := 9
def cost_per_can : ℕ := 175 -- in cents
def num_coupons : ℕ := 5
def total_payment : ℕ := 2000 -- $20 in cents
def change_received : ℕ := 550 -- $5.50 in cents
def amount_paid := total_payment - change_received

-- Mathematical proof problem
theorem discount_per_coupon :
  let total_cost_without_coupons := num_cans * cost_per_can 
  let total_discount := total_cost_without_coupons - amount_paid
  let discount_per_coupon := total_discount / num_coupons
  discount_per_coupon = 25 :=
by
  sorry

end discount_per_coupon_l887_88766


namespace A_days_to_complete_work_alone_l887_88778

theorem A_days_to_complete_work_alone (x : ℝ) (h1 : 0 < x) (h2 : 0 < 18) (h3 : 1/x + 1/18 = 1/6) : x = 9 :=
by
  sorry

end A_days_to_complete_work_alone_l887_88778


namespace number_of_squirrels_l887_88709

/-
Problem: Some squirrels collected 575 acorns. If each squirrel needs 130 acorns to get through the winter, each squirrel needs to collect 15 more acorns. 
Question: How many squirrels are there?
Conditions:
 1. Some squirrels collected 575 acorns.
 2. Each squirrel needs 130 acorns to get through the winter.
 3. Each squirrel needs to collect 15 more acorns.
Answer: 5 squirrels
-/

theorem number_of_squirrels (acorns_total : ℕ) (acorns_needed : ℕ) (acorns_short : ℕ) (S : ℕ)
  (h1 : acorns_total = 575)
  (h2 : acorns_needed = 130)
  (h3 : acorns_short = 15)
  (h4 : S * (acorns_needed - acorns_short) = acorns_total) :
  S = 5 :=
by
  sorry

end number_of_squirrels_l887_88709


namespace irreducible_fraction_l887_88769

theorem irreducible_fraction (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := 
  sorry

end irreducible_fraction_l887_88769


namespace contribution_per_person_correct_l887_88785

-- Definitions from conditions
def total_fundraising_goal : ℕ := 2400
def number_of_participants : ℕ := 8
def administrative_fee_per_person : ℕ := 20

-- Desired answer
def total_contribution_per_person : ℕ := total_fundraising_goal / number_of_participants + administrative_fee_per_person

-- Proof statement
theorem contribution_per_person_correct :
  total_contribution_per_person = 320 :=
by
  sorry  -- Proof to be provided

end contribution_per_person_correct_l887_88785


namespace max_profit_l887_88733

noncomputable def profit_A (x : ℕ) : ℝ := -↑x^2 + 21 * ↑x
noncomputable def profit_B (x : ℕ) : ℝ := 2 * ↑x
noncomputable def total_profit (x : ℕ) : ℝ := profit_A x + profit_B (15 - x)

theorem max_profit : 
  ∃ x : ℕ, 0 ≤ x ∧ x ≤ 15 ∧ total_profit x = 120 := sorry

end max_profit_l887_88733


namespace compute_expression_l887_88726

theorem compute_expression : 19 * 42 + 81 * 19 = 2337 := by
  sorry

end compute_expression_l887_88726


namespace sheila_earning_per_hour_l887_88789

def sheila_hours_per_day_mwf : ℕ := 8
def sheila_days_mwf : ℕ := 3
def sheila_hours_per_day_tt : ℕ := 6
def sheila_days_tt : ℕ := 2
def sheila_total_earnings : ℕ := 432

theorem sheila_earning_per_hour : (sheila_total_earnings / (sheila_hours_per_day_mwf * sheila_days_mwf + sheila_hours_per_day_tt * sheila_days_tt)) = 12 := by
  sorry

end sheila_earning_per_hour_l887_88789


namespace fraction_identity_l887_88774

theorem fraction_identity (m n : ℕ) (h : (m : ℚ) / n = 3 / 7) : ((m + n) : ℚ) / n = 10 / 7 := 
sorry

end fraction_identity_l887_88774


namespace animal_lifespan_probability_l887_88768

theorem animal_lifespan_probability
    (P_B : ℝ) (hP_B : P_B = 0.8)
    (P_A : ℝ) (hP_A : P_A = 0.4)
    : (P_A / P_B = 0.5) :=
by
    sorry

end animal_lifespan_probability_l887_88768


namespace find_min_n_l887_88776

theorem find_min_n (k : ℕ) : ∃ n, 
  (∀ (m : ℕ), (k = 2 * m → n = 100 * (m + 1)) ∨ (k = 2 * m + 1 → n = 100 * (m + 1) + 1)) ∧
  (∀ n', (∀ (m : ℕ), (k = 2 * m → n' ≥ 100 * (m + 1)) ∨ (k = 2 * m + 1 → n' ≥ 100 * (m + 1) + 1)) → n' ≥ n) :=
by {
  sorry
}

end find_min_n_l887_88776


namespace total_pastries_l887_88753

-- Defining the initial conditions
def Grace_pastries : ℕ := 30
def Calvin_pastries : ℕ := Grace_pastries - 5
def Phoebe_pastries : ℕ := Grace_pastries - 5
def Frank_pastries : ℕ := Calvin_pastries - 8

-- The theorem we want to prove
theorem total_pastries : 
  Calvin_pastries + Phoebe_pastries + Frank_pastries + Grace_pastries = 97 := by
  sorry

end total_pastries_l887_88753


namespace cost_of_watch_l887_88731

variable (saved amount_needed total_cost : ℕ)

-- Conditions
def connie_saved : Prop := saved = 39
def connie_needs : Prop := amount_needed = 16

-- Theorem to prove
theorem cost_of_watch : connie_saved saved → connie_needs amount_needed → total_cost = 55 :=
by
  sorry

end cost_of_watch_l887_88731


namespace find_a3_l887_88798

open Nat

def seq (a : ℕ → ℕ) : Prop := 
  (a 1 = 1) ∧ (∀ n : ℕ, n > 0 → a (n + 1) - a n = n)

theorem find_a3 (a : ℕ → ℕ) (h : seq a) : a 3 = 4 := by
  sorry

end find_a3_l887_88798


namespace poly_expansion_sum_l887_88729

theorem poly_expansion_sum (A B C D E : ℤ) (x : ℤ):
  (x + 3) * (4 * x^3 - 2 * x^2 + 3 * x - 1) = A * x^4 + B * x^3 + C * x^2 + D * x + E → 
  A + B + C + D + E = 16 :=
by
  sorry

end poly_expansion_sum_l887_88729


namespace polynomial_unique_f_g_l887_88775

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := sorry

theorem polynomial_unique_f_g :
  (∀ x : ℝ, (x^2 + x + 1) * f (x^2 - x + 1) = (x^2 - x + 1) * g (x^2 + x + 1)) →
  (∃ k : ℝ, ∀ x : ℝ, f x = k * x ∧ g x = k * x) :=
sorry

end polynomial_unique_f_g_l887_88775


namespace determine_n_l887_88741

theorem determine_n (n : ℕ) : (2 : ℕ)^n = 2 * 4^2 * 16^3 ↔ n = 17 := 
by
  sorry

end determine_n_l887_88741


namespace children_selection_l887_88767

-- Conditions and definitions
def comb (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Proof problem statement
theorem children_selection : ∃ r : ℕ, comb 10 r = 210 ∧ r = 4 :=
by
  sorry

end children_selection_l887_88767


namespace negation_of_proposition_true_l887_88732

theorem negation_of_proposition_true :
  (¬ (∀ x: ℝ, x^2 < 1 → -1 < x ∧ x < 1)) ↔ (∃ x: ℝ, x^2 ≥ 1 ∧ (x ≤ -1 ∨ x ≥ 1)) :=
by
  sorry

end negation_of_proposition_true_l887_88732


namespace y_intercept_of_line_b_is_minus_8_l887_88782

/-- Define a line in slope-intercept form y = mx + c --/
structure Line :=
  (m : ℝ)   -- slope
  (c : ℝ)   -- y-intercept

/-- Define a point in 2D Cartesian coordinate system --/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Define conditions for the problem --/
def line_b_parallel_to (l: Line) (p: Point) : Prop :=
  l.m = 2 ∧ p.x = 3 ∧ p.y = -2

/-- Define the target statement to prove --/
theorem y_intercept_of_line_b_is_minus_8 :
  ∀ (b: Line) (p: Point), line_b_parallel_to b p → b.c = -8 := by
  -- proof goes here
  sorry

end y_intercept_of_line_b_is_minus_8_l887_88782


namespace total_eyes_insects_l887_88740

-- Defining the conditions given in the problem
def numSpiders : Nat := 3
def numAnts : Nat := 50
def eyesPerSpider : Nat := 8
def eyesPerAnt : Nat := 2

-- Statement to prove: the total number of eyes among Nina's pet insects is 124
theorem total_eyes_insects : (numSpiders * eyesPerSpider + numAnts * eyesPerAnt) = 124 := by
  sorry

end total_eyes_insects_l887_88740


namespace count_multiples_5_or_10_l887_88755

theorem count_multiples_5_or_10 (n : ℕ) (hn : n = 999) : 
  ∃ k : ℕ, k = 199 ∧ (∀ i : ℕ, i < 1000 → (i % 5 = 0 ∨ i % 10 = 0) → i = k) := 
by {
  sorry
}

end count_multiples_5_or_10_l887_88755


namespace unique_scalar_matrix_l887_88736

theorem unique_scalar_matrix (N : Matrix (Fin 3) (Fin 3) ℝ) :
  (∀ v : Fin 3 → ℝ, Matrix.mulVec N v = 5 • v) → 
  N = !![5, 0, 0; 0, 5, 0; 0, 0, 5] :=
by
  intro hv
  sorry -- Proof omitted as per instructions

end unique_scalar_matrix_l887_88736


namespace max_notebooks_l887_88772

-- Definitions based on the conditions
def joshMoney : ℕ := 1050
def notebookCost : ℕ := 75

-- Statement to prove
theorem max_notebooks (x : ℕ) : notebookCost * x ≤ joshMoney → x ≤ 14 := by
  -- Placeholder for the proof
  sorry

end max_notebooks_l887_88772


namespace cassie_and_brian_meet_at_1111am_l887_88795

theorem cassie_and_brian_meet_at_1111am :
  ∃ t : ℕ, t = 11*60 + 11 ∧
    (∃ x : ℚ, x = 51/16 ∧ 
      14 * x + 18 * (x - 1) = 84) :=
sorry

end cassie_and_brian_meet_at_1111am_l887_88795


namespace problem_1_problem_2_l887_88786

variable {m n x : ℝ}

theorem problem_1 (m n : ℝ) : (m + n) * (2 * m + n) + n * (m - n) = 2 * m ^ 2 + 4 * m * n := 
by
  sorry

theorem problem_2 (x : ℝ) (h : x ≠ 0) : ((x + 3) / x - 2) / ((x ^ 2 - 9) / (4 * x)) = -(4  / (x + 3)) :=
by
  sorry

end problem_1_problem_2_l887_88786


namespace handshakes_total_count_l887_88706

/-
Statement:
There are 30 gremlins and 20 imps at a Regional Mischief Meet. Only half of the imps are willing to shake hands with each other.
All cooperative imps shake hands with each other. All imps shake hands with each gremlin. Gremlins shake hands with every
other gremlin as well as all the imps. Each pair of creatures shakes hands at most once. Prove that the total number of handshakes is 1080.
-/

theorem handshakes_total_count (gremlins imps cooperative_imps : ℕ)
  (H1 : gremlins = 30)
  (H2 : imps = 20)
  (H3 : cooperative_imps = imps / 2) :
  let handshakes_gremlins := gremlins * (gremlins - 1) / 2
  let handshakes_cooperative_imps := cooperative_imps * (cooperative_imps - 1) / 2
  let handshakes_imps_gremlins := imps * gremlins
  handshakes_gremlins + handshakes_cooperative_imps + handshakes_imps_gremlins = 1080 := 
by {
  sorry
}

end handshakes_total_count_l887_88706


namespace seeds_per_watermelon_l887_88756

theorem seeds_per_watermelon (total_seeds : ℕ) (num_watermelons : ℕ) (h : total_seeds = 400 ∧ num_watermelons = 4) : total_seeds / num_watermelons = 100 :=
by
  sorry

end seeds_per_watermelon_l887_88756


namespace max_pencils_to_buy_l887_88714

-- Definition of costs and budget
def pin_cost : ℕ := 3
def pen_cost : ℕ := 4
def pencil_cost : ℕ := 9
def total_budget : ℕ := 72

-- Minimum purchase required: one pin and one pen
def min_purchase : ℕ := pin_cost + pen_cost

-- Remaining budget after minimum purchase
def remaining_budget : ℕ := total_budget - min_purchase

-- Maximum number of pencils can be bought with the remaining budget
def max_pencils := remaining_budget / pencil_cost

-- Theorem stating the maximum number of pencils Alice can purchase
theorem max_pencils_to_buy : max_pencils = 7 :=
by
  -- Proof would go here
  sorry

end max_pencils_to_buy_l887_88714


namespace sum_of_three_integers_with_product_5_pow_4_l887_88784

noncomputable def a : ℕ := 1
noncomputable def b : ℕ := 5
noncomputable def c : ℕ := 125

theorem sum_of_three_integers_with_product_5_pow_4 (h : a * b * c = 5^4) : 
  a + b + c = 131 := by
  have ha : a = 1 := rfl
  have hb : b = 5 := rfl
  have hc : c = 125 := rfl
  rw [ha, hb, hc, mul_assoc] at h
  exact sorry

end sum_of_three_integers_with_product_5_pow_4_l887_88784


namespace part1_part2_l887_88770

-- Definitions for Part (1)
def A : Set ℝ := { x | -2 ≤ x ∧ x ≤ 4 }
def B (m : ℝ) : Set ℝ := { x | -1 ≤ x ∧ x ≤ 3 }

-- Part (1) Statement
theorem part1 (m : ℝ) (hm : m = 2) : A ∩ ((compl B m)) = {x | (-2 ≤ x ∧ x < -1) ∨ (3 < x ∧ x ≤ 4)} := 
by
  sorry

-- Definitions for Part (2)
def B_interval (m : ℝ) : Set ℝ := { x | (1 - m) ≤ x ∧ x ≤ (1 + m) }

-- Part (2) Statement
theorem part2 (m : ℝ) (h : ∀ x, (x ∈ A → x ∈ B_interval m)) : 0 < m ∧ m < 3 := 
by
  sorry

end part1_part2_l887_88770


namespace complex_eq_l887_88737

theorem complex_eq : ∀ (z : ℂ), (i * z = i + z) → (z = (1 - i) / 2) :=
by
  intros z h
  sorry

end complex_eq_l887_88737


namespace probability_single_trial_l887_88727

open Real

theorem probability_single_trial :
  ∃ p : ℝ, 0 ≤ p ∧ p ≤ 1 ∧ (1 - p)^4 = 16 / 81 ∧ p = 1 / 3 :=
by
  -- The proof steps have been skipped.
  sorry

end probability_single_trial_l887_88727


namespace find_z_l887_88791

open Complex

theorem find_z (z : ℂ) (h : z * (2 - I) = 5 * I) : z = -1 + 2 * I :=
sorry

end find_z_l887_88791


namespace teachers_photos_l887_88744

theorem teachers_photos (n : ℕ) (ht : n = 5) : 6 * 7 = 42 :=
by
  sorry

end teachers_photos_l887_88744


namespace sum_100_consecutive_from_neg49_l887_88708

noncomputable def sum_of_consecutive_integers (n : ℕ) (first_term : ℤ) : ℤ :=
  n * ( first_term + (first_term + n - 1) ) / 2

theorem sum_100_consecutive_from_neg49 : sum_of_consecutive_integers 100 (-49) = 50 :=
by sorry

end sum_100_consecutive_from_neg49_l887_88708


namespace volume_of_pyramid_l887_88742

theorem volume_of_pyramid 
  (QR RS : ℝ) (PT : ℝ) 
  (hQR_pos : 0 < QR) (hRS_pos : 0 < RS) (hPT_pos : 0 < PT)
  (perp1 : ∃ x : ℝ, ∀ y : ℝ, y ≠ x -> (PT * QR) * (x * y) = 0)
  (perp2 : ∃ x : ℝ, ∀ y : ℝ, y ≠ x -> (PT * RS) * (x * y) = 0) :
  QR = 10 -> RS = 5 -> PT = 9 -> 
  (1/3) * QR * RS * PT = 150 :=
by
  sorry

end volume_of_pyramid_l887_88742


namespace true_propositions_l887_88738

open Set

theorem true_propositions (M N : Set ℕ) (a b m : ℕ) (h1 : M ⊆ N) 
  (h2 : a > b) (h3 : b > 0) (h4 : m > 0) (p : ∀ x : ℝ, x > 0) :
  (M ⊆ M ∪ N) ∧ ((b + m) / (a + m) > b / a) ∧ 
  ¬(∀ (a b c : ℝ), a = b ↔ a * c ^ 2 = b * c ^ 2) ∧ 
  ¬(∃ x₀ : ℝ, x₀ ≤ 0) := sorry

end true_propositions_l887_88738


namespace crayons_lost_or_given_away_total_l887_88728

def initial_crayons_box1 := 479
def initial_crayons_box2 := 352
def initial_crayons_box3 := 621

def remaining_crayons_box1 := 134
def remaining_crayons_box2 := 221
def remaining_crayons_box3 := 487

def crayons_lost_or_given_away_box1 := initial_crayons_box1 - remaining_crayons_box1
def crayons_lost_or_given_away_box2 := initial_crayons_box2 - remaining_crayons_box2
def crayons_lost_or_given_away_box3 := initial_crayons_box3 - remaining_crayons_box3

def total_crayons_lost_or_given_away := crayons_lost_or_given_away_box1 + crayons_lost_or_given_away_box2 + crayons_lost_or_given_away_box3

theorem crayons_lost_or_given_away_total : total_crayons_lost_or_given_away = 610 :=
by
  -- Proof should go here
  sorry

end crayons_lost_or_given_away_total_l887_88728


namespace Craig_initial_apples_l887_88788

variable (j : ℕ) (shared : ℕ) (left : ℕ)

theorem Craig_initial_apples (HJ : j = 11) (HS : shared = 7) (HL : left = 13) :
  shared + left = 20 := by
  sorry

end Craig_initial_apples_l887_88788


namespace strongest_correlation_l887_88781

variables (r1 : ℝ) (r2 : ℝ) (r3 : ℝ) (r4 : ℝ)
variables (abs_r3 : ℝ)

-- Define conditions as hypotheses
def conditions :=
  r1 = 0 ∧ r2 = -0.95 ∧ abs_r3 = 0.89 ∧ r4 = 0.75 ∧ abs r3 = abs_r3

-- Theorem stating the correct answer
theorem strongest_correlation (hyp : conditions r1 r2 r3 r4 abs_r3) : 
  abs r2 > abs r1 ∧ abs r2 > abs r3 ∧ abs r2 > abs r4 :=
by sorry

end strongest_correlation_l887_88781


namespace ratio_of_largest_to_smallest_root_in_geometric_progression_l887_88764

theorem ratio_of_largest_to_smallest_root_in_geometric_progression 
    (a b c d : ℝ) (r s t : ℝ) 
    (h_poly : 81 * r^3 - 243 * r^2 + 216 * r - 64 = 0)
    (h_geo_prog : r > 0 ∧ s > 0 ∧ t > 0 ∧ ∃ (k : ℝ),  k > 0 ∧ s = r * k ∧ t = s * k) :
    ∃ (k : ℝ), k = r^2 ∧ s = r * k ∧ t = s * k := 
sorry

end ratio_of_largest_to_smallest_root_in_geometric_progression_l887_88764


namespace sqrt_product_simplified_l887_88704

theorem sqrt_product_simplified (q : ℝ) (hq : 0 < q) :
  Real.sqrt (42 * q) * Real.sqrt (7 * q) * Real.sqrt (3 * q) = 21 * q * Real.sqrt (2 * q) :=
by
  sorry

end sqrt_product_simplified_l887_88704


namespace negation_of_proposition_l887_88719

theorem negation_of_proposition :
  (∀ x : ℝ, x > 0 → x + (1 / x) ≥ 2) →
  (∃ x₀ : ℝ, x₀ > 0 ∧ x₀ + (1 / x₀) < 2) :=
sorry

end negation_of_proposition_l887_88719


namespace price_reductions_l887_88773

theorem price_reductions (a : ℝ) : 18400 * (1 - a / 100)^2 = 16000 :=
sorry

end price_reductions_l887_88773


namespace inequality_solution_l887_88758

theorem inequality_solution :
  { x : ℝ | (x^3 - 4 * x) / (x^2 - 1) > 0 } = { x : ℝ | x < -2 ∨ (0 < x ∧ x < 1) ∨ 2 < x } :=
by
  sorry

end inequality_solution_l887_88758


namespace total_dollars_l887_88720

def mark_dollars : ℚ := 4 / 5
def carolyn_dollars : ℚ := 2 / 5
def jack_dollars : ℚ := 1 / 2

theorem total_dollars :
  mark_dollars + carolyn_dollars + jack_dollars = 1.7 := 
sorry

end total_dollars_l887_88720


namespace value_of_expression_l887_88799

def f (x : ℝ) : ℝ := x^2 - 3*x + 7
def g (x : ℝ) : ℝ := x + 2

theorem value_of_expression : f (g 3) - g (f 3) = 8 :=
by
  sorry

end value_of_expression_l887_88799


namespace h_value_at_3_l887_88779

noncomputable def f (x : ℝ) : ℝ := 3 * x + 4
noncomputable def g (x : ℝ) : ℝ := (Real.sqrt (f x) - 3) ^ 2
noncomputable def h (x : ℝ) : ℝ := f (g x)

theorem h_value_at_3 : h 3 = 70 - 18 * Real.sqrt 13 := 
by
  -- Proof goes here
  sorry

end h_value_at_3_l887_88779


namespace find_solutions_l887_88796

def equation (x : ℝ) : Prop :=
  (1 / (x^2 + 12*x - 8)) + (1 / (x^2 + 3*x - 8)) + (1 / (x^2 - 14*x - 8)) = 0

theorem find_solutions : {x : ℝ | equation x} = {2, -4, 1, -8} :=
  by
  sorry

end find_solutions_l887_88796


namespace remaining_coins_denomination_l887_88707

def denomination_of_remaining_coins (total_coins : ℕ) (total_value : ℕ) (paise_20_count : ℕ) (paise_20_value : ℕ) : ℕ :=
  let remaining_coins := total_coins - paise_20_count
  let remaining_value := total_value - paise_20_count * paise_20_value
  remaining_value / remaining_coins

theorem remaining_coins_denomination :
  denomination_of_remaining_coins 334 7100 250 20 = 25 :=
by
  sorry

end remaining_coins_denomination_l887_88707


namespace find_number_l887_88735

theorem find_number 
    (x : ℝ)
    (h1 : 3 < x) 
    (h2 : x < 8) 
    (h3 : 6 < x) 
    (h4 : x < 10) : 
    x = 7 :=
sorry

end find_number_l887_88735


namespace factorize_expression_l887_88792

theorem factorize_expression (x : ℝ) : 
  x^3 - 5 * x^2 + 4 * x = x * (x - 1) * (x - 4) :=
by
  sorry

end factorize_expression_l887_88792


namespace sum_of_integers_l887_88711

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 8) (h3 : x * y = 240) : x + y = 32 :=
by
  sorry

end sum_of_integers_l887_88711


namespace diophantine_no_nonneg_solutions_l887_88725

theorem diophantine_no_nonneg_solutions {a b : ℕ} (ha : 0 < a) (hb : 0 < b) (h_gcd : Nat.gcd a b = 1) :
  ∃ (c : ℕ), (a * b - a - b + 1) / 2 = (a - 1) * (b - 1) / 2 := 
sorry

end diophantine_no_nonneg_solutions_l887_88725


namespace f_identity_l887_88746

def f (x : ℝ) : ℝ := (2 * x + 1)^5 - 5 * (2 * x + 1)^4 + 10 * (2 * x + 1)^3 - 10 * (2 * x + 1)^2 + 5 * (2 * x + 1) - 1

theorem f_identity (x : ℝ) : f x = 32 * x^5 :=
by
  -- the proof is omitted
  sorry

end f_identity_l887_88746


namespace cost_price_of_article_l887_88705

theorem cost_price_of_article (C MP : ℝ) (h1 : 0.90 * MP = 1.25 * C) (h2 : 1.25 * C = 65.97) : C = 52.776 :=
by
  sorry

end cost_price_of_article_l887_88705


namespace distance_between_x_intercepts_l887_88739

theorem distance_between_x_intercepts :
  let slope1 := 4
  let slope2 := -2
  let point := (8, 20)
  let line1 (x : ℝ) := slope1 * (x - point.1) + point.2
  let line2 (x : ℝ) := slope2 * (x - point.1) + point.2
  let x_intercept1 := (0 - point.2) / slope1 + point.1
  let x_intercept2 := (0 - point.2) / slope2 + point.1
  abs (x_intercept1 - x_intercept2) = 15 := sorry

end distance_between_x_intercepts_l887_88739


namespace a_eq_one_sufficient_not_necessary_P_subset_M_iff_l887_88730

open Set

-- Define sets P and M based on conditions
def P : Set ℝ := {x | x > 2}
def M (a : ℝ) : Set ℝ := {x | x > a}

-- State the theorem
theorem a_eq_one_sufficient_not_necessary (a : ℝ) : (a = 1) → (P ⊆ M a) := 
by
  sorry

theorem P_subset_M_iff (a : ℝ) : (P ⊆ M a) ↔ (a < 2) :=
by
  sorry

end a_eq_one_sufficient_not_necessary_P_subset_M_iff_l887_88730


namespace minimum_value_of_expression_l887_88771

theorem minimum_value_of_expression (x y : ℝ) : 
    ∃ (x y : ℝ), (2 * x * y - 1) ^ 2 + (x - y) ^ 2 = 0 :=
by
  sorry

end minimum_value_of_expression_l887_88771


namespace largest_divisor_of_5_consecutive_integers_l887_88761

theorem largest_divisor_of_5_consecutive_integers :
  ∃ d : ℤ, (∀ n : ℤ, d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4))) ∧ d = 120 :=
sorry

end largest_divisor_of_5_consecutive_integers_l887_88761


namespace unique_solution_f_l887_88717

def f : ℕ → ℕ
  := sorry

namespace ProofProblem

theorem unique_solution_f (f : ℕ → ℕ)
  (h1 : ∀ (m n : ℕ), f m + f n - m * n ≠ 0)
  (h2 : ∀ (m n : ℕ), f m + f n - m * n ∣ m * f m + n * f n)
  : (∀ n : ℕ, f n = n^2) :=
sorry

end ProofProblem

end unique_solution_f_l887_88717


namespace FGH_supermarkets_total_l887_88700

theorem FGH_supermarkets_total (US Canada : ℕ) 
  (h1 : US = 49) 
  (h2 : US = Canada + 14) : 
  US + Canada = 84 := 
by 
  sorry

end FGH_supermarkets_total_l887_88700


namespace distance_from_O_is_450_l887_88712

noncomputable def find_distance_d (A B C P Q O : Type) 
  (side_length : ℝ) (PA PB PC QA QB QC dist_OA dist_OB dist_OC dist_OP dist_OQ : ℝ) : ℝ :=
    if h : PA = PB ∧ PB = PC ∧ QA = QB ∧ QB = QC ∧ side_length = 600 ∧
           dist_OA = dist_OB ∧ dist_OB = dist_OC ∧ dist_OC = dist_OP ∧ dist_OP = dist_OQ ∧
           -- condition of 120 degree dihedral angle translates to specific geometric constraints
           true -- placeholder for the actual geometrical configuration that proves the problem
    then 450
    else 0 -- default or indication of inconsistency in conditions

-- Assuming all conditions hold true
theorem distance_from_O_is_450 (A B C P Q O : Type) 
  (side_length : ℝ) (PA PB PC QA QB QC dist_OA dist_OB dist_OC dist_OP dist_OQ : ℝ)
  (h : PA = PB ∧ PB = PC ∧ QA = QB ∧ QB = QC ∧ side_length = 600 ∧
       dist_OA = dist_OB ∧ dist_OB = dist_OC ∧ dist_OC = dist_OP ∧ dist_OP = dist_OQ ∧
       -- adding condition of 120 degree dihedral angle
       true) -- true is a placeholder, the required proof to be filled in
  : find_distance_d A B C P Q O side_length PA PB PC QA QB QC dist_OA dist_OB dist_OC dist_OP dist_OQ = 450 :=
by
  -- proof goes here
  sorry

end distance_from_O_is_450_l887_88712


namespace lark_lock_combination_count_l887_88715

-- Definitions for the conditions
def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_multiple_of_5 (n : ℕ) : Prop := n % 5 = 0

def lark_lock_combination (a b c : ℕ) : Prop := 
  is_odd a ∧ is_even b ∧ is_multiple_of_5 c ∧ 1 ≤ a ∧ a ≤ 30 ∧ 1 ≤ b ∧ b ≤ 30 ∧ 1 ≤ c ∧ c ≤ 30

-- The core theorem
theorem lark_lock_combination_count : 
  (∃ a b c : ℕ, lark_lock_combination a b c) ↔ (15 * 15 * 6 = 1350) :=
by
  sorry

end lark_lock_combination_count_l887_88715


namespace eq_expression_l887_88710

theorem eq_expression :
  (2 + 3) * (2^2 + 3^2) * (2^4 + 3^4) * (2^8 + 3^8) * (2^16 + 3^16) * (2^32 + 3^32) * (2^64 + 3^64) = 3^128 - 2^128 :=
by
  sorry

end eq_expression_l887_88710


namespace opposite_of_x_abs_of_x_recip_of_x_l887_88749

noncomputable def x : ℝ := 1 - Real.sqrt 2

theorem opposite_of_x : -x = Real.sqrt 2 - 1 := 
by sorry

theorem abs_of_x : |x| = Real.sqrt 2 - 1 :=
by sorry

theorem recip_of_x : 1/x = -1 - Real.sqrt 2 :=
by sorry

end opposite_of_x_abs_of_x_recip_of_x_l887_88749


namespace seismic_activity_mismatch_percentage_l887_88716

theorem seismic_activity_mismatch_percentage
  (total_days : ℕ)
  (quiet_days_percentage : ℝ)
  (prediction_accuracy : ℝ)
  (predicted_quiet_days_percentage : ℝ)
  (quiet_prediction_correctness : ℝ)
  (active_days_percentage : ℝ)
  (incorrect_quiet_predictions : ℝ) :
  quiet_days_percentage = 0.8 →
  predicted_quiet_days_percentage = 0.64 →
  quiet_prediction_correctness = 0.7 →
  active_days_percentage = 0.2 →
  incorrect_quiet_predictions = predicted_quiet_days_percentage - (quiet_prediction_correctness * quiet_days_percentage) →
  (incorrect_quiet_predictions / active_days_percentage) * 100 = 40 := by
  sorry

end seismic_activity_mismatch_percentage_l887_88716
