import Mathlib

namespace NUMINAMATH_GPT_tom_teaching_years_l2303_230316

theorem tom_teaching_years :
  ∃ T D : ℕ, T + D = 70 ∧ D = (1 / 2) * T - 5 ∧ T = 50 :=
by
  sorry

end NUMINAMATH_GPT_tom_teaching_years_l2303_230316


namespace NUMINAMATH_GPT_cross_section_area_l2303_230356

-- Definitions representing the conditions
variables (AK KD BP PC DM DC : ℝ)
variable (h : ℝ)
variable (Volume : ℝ)

-- Conditions
axiom hyp1 : AK = KD
axiom hyp2 : BP = PC
axiom hyp3 : DM = 0.4 * DC
axiom hyp4 : h = 1
axiom hyp5 : Volume = 5

-- Proof problem: Prove that the area S of the cross-section of the pyramid is 3
theorem cross_section_area (S : ℝ) : S = 3 :=
by sorry

end NUMINAMATH_GPT_cross_section_area_l2303_230356


namespace NUMINAMATH_GPT_cost_of_7_enchiladas_and_6_tacos_l2303_230378

theorem cost_of_7_enchiladas_and_6_tacos (e t : ℝ) 
  (h₁ : 4 * e + 5 * t = 5.00) 
  (h₂ : 6 * e + 3 * t = 5.40) : 
  7 * e + 6 * t = 7.47 := 
sorry

end NUMINAMATH_GPT_cost_of_7_enchiladas_and_6_tacos_l2303_230378


namespace NUMINAMATH_GPT_arithmetic_seq_75th_term_difference_l2303_230373

theorem arithmetic_seq_75th_term_difference :
  ∃ (d : ℝ), 300 * (50 + d) = 15000 ∧ -30 / 299 ≤ d ∧ d ≤ 30 / 299 ∧
  let L := 50 - 225 * (30 / 299)
  let G := 50 + 225 * (30 / 299)
  G - L = 13500 / 299 := by
sorry

end NUMINAMATH_GPT_arithmetic_seq_75th_term_difference_l2303_230373


namespace NUMINAMATH_GPT_incorrect_statement_A_l2303_230348

-- Define the statements based on conditions
def statementA : String := "INPUT \"MATH=\"; a+b+c"
def statementB : String := "PRINT \"MATH=\"; a+b+c"
def statementC : String := "a=b+c"
def statementD : String := "a=b-c"

-- Define a function to check if a statement is valid syntax
noncomputable def isValidSyntax : String → Prop :=
  λ stmt => 
    stmt = statementB ∨ stmt = statementC ∨ stmt = statementD

-- The proof problem
theorem incorrect_statement_A : ¬ isValidSyntax statementA :=
  sorry

end NUMINAMATH_GPT_incorrect_statement_A_l2303_230348


namespace NUMINAMATH_GPT_find_r_in_geometric_series_l2303_230343

theorem find_r_in_geometric_series
  (a r : ℝ)
  (h1 : a / (1 - r) = 15)
  (h2 : a / (1 - r^2) = 6) :
  r = 2 / 3 :=
sorry

end NUMINAMATH_GPT_find_r_in_geometric_series_l2303_230343


namespace NUMINAMATH_GPT_worst_player_is_son_l2303_230357

-- Define the types of players and relationships
inductive Sex
| male
| female

structure Player where
  name : String
  sex : Sex
  age : Nat

-- Define the four players
def woman := Player.mk "woman" Sex.female 30  -- Age is arbitrary
def brother := Player.mk "brother" Sex.male 30
def son := Player.mk "son" Sex.male 10
def daughter := Player.mk "daughter" Sex.female 10

-- Define the conditions
def opposite_sex (p1 p2 : Player) : Prop := p1.sex ≠ p2.sex
def same_age (p1 p2 : Player) : Prop := p1.age = p2.age

-- Define the worst player and the best player
variable (worst_player : Player) (best_player : Player)

-- Conditions as hypotheses
axiom twin_condition : ∃ twin : Player, (twin ≠ worst_player) ∧ (opposite_sex twin best_player)
axiom age_condition : same_age worst_player best_player
axiom not_same_player : worst_player ≠ best_player

-- Prove that the worst player is the son
theorem worst_player_is_son : worst_player = son :=
by
  sorry

end NUMINAMATH_GPT_worst_player_is_son_l2303_230357


namespace NUMINAMATH_GPT_compute_expression_l2303_230335

theorem compute_expression : 1004^2 - 996^2 - 1002^2 + 998^2 = 8000 := by
  sorry

end NUMINAMATH_GPT_compute_expression_l2303_230335


namespace NUMINAMATH_GPT_positive_integer_not_in_S_l2303_230351

noncomputable def S : Set ℤ :=
  {n | ∃ (i : ℕ), n = 4^i * 3 ∨ n = -4^i * 2}

theorem positive_integer_not_in_S (n : ℤ) (hn : 0 < n) (hnS : n ∉ S) :
  ∃ (x y : ℤ), x ≠ y ∧ x ∈ S ∧ y ∈ S ∧ x + y = n :=
sorry

end NUMINAMATH_GPT_positive_integer_not_in_S_l2303_230351


namespace NUMINAMATH_GPT_ratio_songs_kept_to_deleted_l2303_230380

theorem ratio_songs_kept_to_deleted (initial_songs deleted_songs kept_songs : ℕ) 
  (h_initial : initial_songs = 54) (h_deleted : deleted_songs = 9) (h_kept : kept_songs = initial_songs - deleted_songs) :
  (kept_songs : ℚ) / (deleted_songs : ℚ) = 5 / 1 :=
by
  sorry

end NUMINAMATH_GPT_ratio_songs_kept_to_deleted_l2303_230380


namespace NUMINAMATH_GPT_train_speed_l2303_230358

/-- 
A man sitting in a train which is traveling at a certain speed observes 
that a goods train, traveling in the opposite direction, takes 9 seconds 
to pass him. The goods train is 280 m long and its speed is 52 kmph. 
Prove that the speed of the train the man is sitting in is 60 kmph.
-/
theorem train_speed (t : ℝ) (h1 : 0 < t)
  (goods_speed_kmph : ℝ := 52)
  (goods_length_m : ℝ := 280)
  (time_seconds : ℝ := 9)
  (h2 : goods_length_m / time_seconds = (t + goods_speed_kmph) * (5 / 18)) :
  t = 60 :=
sorry

end NUMINAMATH_GPT_train_speed_l2303_230358


namespace NUMINAMATH_GPT_total_distance_covered_l2303_230304

noncomputable def radius : ℝ := 0.242
noncomputable def circumference : ℝ := 2 * Real.pi * radius
noncomputable def number_of_revolutions : ℕ := 500
noncomputable def total_distance : ℝ := circumference * number_of_revolutions

theorem total_distance_covered :
  total_distance = 760 :=
by
  -- sorry Re-enable this line for the solver to automatically skip the proof 
  sorry

end NUMINAMATH_GPT_total_distance_covered_l2303_230304


namespace NUMINAMATH_GPT_ratio_of_sums_l2303_230397

theorem ratio_of_sums (p q r u v w : ℝ) 
  (h1 : p > 0) (h2 : q > 0) (h3 : r > 0) (h4 : u > 0) (h5 : v > 0) (h6 : w > 0)
  (h7 : p^2 + q^2 + r^2 = 49) (h8 : u^2 + v^2 + w^2 = 64)
  (h9 : p * u + q * v + r * w = 56) : 
  (p + q + r) / (u + v + w) = 7 / 8 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_sums_l2303_230397


namespace NUMINAMATH_GPT_family_reunion_kids_l2303_230381

theorem family_reunion_kids (adults : ℕ) (tables : ℕ) (people_per_table : ℕ) 
  (h_adults : adults = 123) (h_tables : tables = 14) 
  (h_people_per_table : people_per_table = 12) :
  (tables * people_per_table - adults) = 45 :=
by
  sorry

end NUMINAMATH_GPT_family_reunion_kids_l2303_230381


namespace NUMINAMATH_GPT_total_ranking_sequences_at_end_l2303_230345

-- Define the teams
inductive Team
| E
| F
| G
| H

open Team

-- Conditions of the problem
def split_groups : (Team × Team) × (Team × Team) :=
  ((E, F), (G, H))

def saturday_matches : (Team × Team) × (Team × Team) :=
  ((E, F), (G, H))

-- Function to count total ranking sequences
noncomputable def total_ranking_sequences : ℕ := 4

-- Define the main theorem
theorem total_ranking_sequences_at_end : total_ranking_sequences = 4 :=
by
  sorry

end NUMINAMATH_GPT_total_ranking_sequences_at_end_l2303_230345


namespace NUMINAMATH_GPT_prove_n_prime_l2303_230346

theorem prove_n_prime (n : ℕ) (p : ℕ) (k : ℕ) (hp : Prime p) (h1 : n > 0) (h2 : 3^n - 2^n = p^k) : Prime n :=
by {
  sorry
}

end NUMINAMATH_GPT_prove_n_prime_l2303_230346


namespace NUMINAMATH_GPT_rectangle_area_k_l2303_230374

theorem rectangle_area_k (d : ℝ) (length width : ℝ) (h_ratio : length / width = 5 / 2)
  (h_diag : (length ^ 2 + width ^ 2) = d ^ 2) :
  ∃ (k : ℝ), k = 10 / 29 ∧ length * width = k * d ^ 2 := by
  sorry

end NUMINAMATH_GPT_rectangle_area_k_l2303_230374


namespace NUMINAMATH_GPT_Gianna_daily_savings_l2303_230306

theorem Gianna_daily_savings 
  (total_saved : ℕ) (days_in_year : ℕ) 
  (H1 : total_saved = 14235) 
  (H2 : days_in_year = 365) : 
  total_saved / days_in_year = 39 := 
by 
  sorry

end NUMINAMATH_GPT_Gianna_daily_savings_l2303_230306


namespace NUMINAMATH_GPT_estimate_number_of_blue_cards_l2303_230330

-- Define the given conditions:
def red_cards : ℕ := 8
def frequency_blue_card : ℚ := 0.6

-- Define the statement that needs to be proved:
theorem estimate_number_of_blue_cards (x : ℕ) 
  (h : (x : ℚ) / (x + red_cards) = frequency_blue_card) : 
  x = 12 :=
  sorry

end NUMINAMATH_GPT_estimate_number_of_blue_cards_l2303_230330


namespace NUMINAMATH_GPT_track_length_l2303_230333

theorem track_length (V_A V_B V_C : ℝ) (x : ℝ) 
  (h1 : x / V_A = (x - 1) / V_B) 
  (h2 : x / V_A = (x - 2) / V_C) 
  (h3 : x / V_B = (x - 1.01) / V_C) : 
  110 - x = 9 :=
by 
  sorry

end NUMINAMATH_GPT_track_length_l2303_230333


namespace NUMINAMATH_GPT_father_age_l2303_230302

theorem father_age (M F : ℕ) 
  (h1 : M = 2 * F / 5) 
  (h2 : M + 10 = (F + 10) / 2) : F = 50 :=
sorry

end NUMINAMATH_GPT_father_age_l2303_230302


namespace NUMINAMATH_GPT_smallest_bob_number_l2303_230388

-- Definitions and conditions
def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def prime_factors (n : ℕ) : Set ℕ := { p | is_prime p ∧ p ∣ n }

def alice_number := 36
def bob_number (m : ℕ) : Prop := prime_factors alice_number ⊆ prime_factors m

-- Proof problem statement
theorem smallest_bob_number :
  ∃ m, bob_number m ∧ m = 6 :=
sorry

end NUMINAMATH_GPT_smallest_bob_number_l2303_230388


namespace NUMINAMATH_GPT_algebraic_expression_eq_five_l2303_230367

theorem algebraic_expression_eq_five (a b : ℝ)
  (h₁ : a^2 - a = 1)
  (h₂ : b^2 - b = 1) :
  3 * a^2 + 2 * b^2 - 3 * a - 2 * b = 5 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_eq_five_l2303_230367


namespace NUMINAMATH_GPT_smallest_number_when_diminished_by_7_is_divisible_l2303_230385

-- Variables for divisors
def divisor1 : Nat := 12
def divisor2 : Nat := 16
def divisor3 : Nat := 18
def divisor4 : Nat := 21
def divisor5 : Nat := 28

-- The smallest number x which, when diminished by 7, is divisible by the divisors.
theorem smallest_number_when_diminished_by_7_is_divisible (x : Nat) : 
  (x - 7) % divisor1 = 0 ∧ 
  (x - 7) % divisor2 = 0 ∧ 
  (x - 7) % divisor3 = 0 ∧ 
  (x - 7) % divisor4 = 0 ∧ 
  (x - 7) % divisor5 = 0 → 
  x = 1015 := 
sorry

end NUMINAMATH_GPT_smallest_number_when_diminished_by_7_is_divisible_l2303_230385


namespace NUMINAMATH_GPT_no_preimage_for_p_gt_1_l2303_230383

def f (x : ℝ) : ℝ := -x^2 + 2 * x

theorem no_preimage_for_p_gt_1 (P : ℝ) (hP : P > 1) : ¬ ∃ x : ℝ, f x = P :=
sorry

end NUMINAMATH_GPT_no_preimage_for_p_gt_1_l2303_230383


namespace NUMINAMATH_GPT_pow_mod_equality_l2303_230314

theorem pow_mod_equality (h : 2^3 ≡ 1 [MOD 7]) : 2^30 ≡ 1 [MOD 7] :=
sorry

end NUMINAMATH_GPT_pow_mod_equality_l2303_230314


namespace NUMINAMATH_GPT_smallest_natural_number_l2303_230362

theorem smallest_natural_number (x : ℕ) : 
  (x % 5 = 2) ∧ (x % 6 = 2) ∧ (x % 7 = 3) → x = 122 := 
by
  sorry

end NUMINAMATH_GPT_smallest_natural_number_l2303_230362


namespace NUMINAMATH_GPT_fraction_addition_l2303_230368

theorem fraction_addition :
  (5 / (8 / 13) + 4 / 7) = (487 / 56) := by
  sorry

end NUMINAMATH_GPT_fraction_addition_l2303_230368


namespace NUMINAMATH_GPT_compute_H_five_times_l2303_230370

def H (x : ℝ) : ℝ := x^2 - 2 * x - 1

theorem compute_H_five_times : H (H (H (H (H 2)))) = -1 := by
  sorry

end NUMINAMATH_GPT_compute_H_five_times_l2303_230370


namespace NUMINAMATH_GPT_differential_equation_solution_l2303_230338

def C1 : ℝ := sorry
def C2 : ℝ := sorry

noncomputable def y (x : ℝ) : ℝ := C1 * Real.cos x + C2 * Real.sin x
noncomputable def z (x : ℝ) : ℝ := -C1 * Real.sin x + C2 * Real.cos x

theorem differential_equation_solution : 
  (∀ x : ℝ, deriv y x = z x) ∧ 
  (∀ x : ℝ, deriv z x = -y x) :=
by
  sorry

end NUMINAMATH_GPT_differential_equation_solution_l2303_230338


namespace NUMINAMATH_GPT_real_roots_quadratic_l2303_230309

theorem real_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, (k - 2) * x^2 - 2 * k * x + k - 6 = 0) ↔ (k ≥ 1.5 ∧ k ≠ 2) :=
by {
  sorry
}

end NUMINAMATH_GPT_real_roots_quadratic_l2303_230309


namespace NUMINAMATH_GPT_most_efficient_packing_l2303_230377

theorem most_efficient_packing :
  ∃ box_size, 
  (box_size = 3 ∨ box_size = 6 ∨ box_size = 9) ∧ 
  (∀ q ∈ [21, 18, 15, 12, 9], q % box_size = 0) ∧
  box_size = 3 :=
by
  sorry

end NUMINAMATH_GPT_most_efficient_packing_l2303_230377


namespace NUMINAMATH_GPT_correct_option_D_l2303_230376

theorem correct_option_D (y : ℝ): 
  3 * y^2 - 2 * y^2 = y^2 :=
by
  sorry

end NUMINAMATH_GPT_correct_option_D_l2303_230376


namespace NUMINAMATH_GPT_eval_nabla_l2303_230328

def nabla (a b : ℕ) : ℕ := 3 + b^(a-1)

theorem eval_nabla : nabla (nabla 2 3) 4 = 1027 := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_eval_nabla_l2303_230328


namespace NUMINAMATH_GPT_smallest_integer_expression_l2303_230337

theorem smallest_integer_expression :
  ∃ m n : ℤ, 1237 * m + 78653 * n = 1 :=
sorry

end NUMINAMATH_GPT_smallest_integer_expression_l2303_230337


namespace NUMINAMATH_GPT_distance_point_to_line_zero_or_four_l2303_230322

theorem distance_point_to_line_zero_or_four {b : ℝ} 
(h : abs (b - 2) / Real.sqrt 2 = Real.sqrt 2) : 
b = 0 ∨ b = 4 := 
sorry

end NUMINAMATH_GPT_distance_point_to_line_zero_or_four_l2303_230322


namespace NUMINAMATH_GPT_totalPlayers_l2303_230360

def kabadiParticipants : ℕ := 50
def khoKhoParticipants : ℕ := 80
def soccerParticipants : ℕ := 30
def kabadiAndKhoKhoParticipants : ℕ := 15
def kabadiAndSoccerParticipants : ℕ := 10
def khoKhoAndSoccerParticipants : ℕ := 25
def allThreeParticipants : ℕ := 8

theorem totalPlayers : kabadiParticipants + khoKhoParticipants + soccerParticipants 
                       - kabadiAndKhoKhoParticipants - kabadiAndSoccerParticipants 
                       - khoKhoAndSoccerParticipants + allThreeParticipants = 118 :=
by 
  sorry

end NUMINAMATH_GPT_totalPlayers_l2303_230360


namespace NUMINAMATH_GPT_cost_to_cover_wall_with_tiles_l2303_230318

/--
There is a wall in the shape of a rectangle with a width of 36 centimeters (cm) and a height of 72 centimeters (cm).
On this wall, you want to attach tiles that are 3 centimeters (cm) and 4 centimeters (cm) in length and width, respectively,
without any empty space. If it costs 2500 won per tile, prove that the total cost to cover the wall is 540,000 won.

Conditions:
- width_wall = 36
- height_wall = 72
- width_tile = 3
- height_tile = 4
- cost_per_tile = 2500

Target:
- Total_cost = 540,000 won
-/
theorem cost_to_cover_wall_with_tiles :
  let width_wall := 36
  let height_wall := 72
  let width_tile := 3
  let height_tile := 4
  let cost_per_tile := 2500
  let area_wall := width_wall * height_wall
  let area_tile := width_tile * height_tile
  let number_of_tiles := area_wall / area_tile
  let total_cost := number_of_tiles * cost_per_tile
  total_cost = 540000 := by
  sorry

end NUMINAMATH_GPT_cost_to_cover_wall_with_tiles_l2303_230318


namespace NUMINAMATH_GPT_f_neg_l2303_230369

-- Define the function f and its properties
noncomputable def f (x : ℝ) : ℝ := if x ≥ 0 then x^2 - 2*x else sorry

-- Define the property of f being an odd function
axiom f_odd : ∀ x : ℝ, f (-x) = -f x

-- Define the property of f for non-negative x
axiom f_nonneg : ∀ x : ℝ, x ≥ 0 → f x = x^2 - 2*x

-- The theorem to be proven
theorem f_neg : ∀ x : ℝ, x < 0 → f x = -x^2 - 2*x := by
  sorry

end NUMINAMATH_GPT_f_neg_l2303_230369


namespace NUMINAMATH_GPT_six_digit_number_consecutive_evens_l2303_230392

theorem six_digit_number_consecutive_evens :
  ∃ n : ℕ,
    287232 = (2 * n - 2) * (2 * n) * (2 * n + 2) ∧
    287232 / 100000 = 2 ∧
    287232 % 10 = 2 :=
by
  sorry

end NUMINAMATH_GPT_six_digit_number_consecutive_evens_l2303_230392


namespace NUMINAMATH_GPT_a100_gt_two_pow_99_l2303_230307

theorem a100_gt_two_pow_99 
  (a : ℕ → ℤ) 
  (h1 : a 1 > a 0)
  (h2 : a 1 > 0)
  (h3 : ∀ r : ℕ, r ≤ 98 → a (r + 2) = 3 * a (r + 1) - 2 * a r) : 
  a 100 > 2 ^ 99 :=
sorry

end NUMINAMATH_GPT_a100_gt_two_pow_99_l2303_230307


namespace NUMINAMATH_GPT_friend_redistribution_l2303_230359

-- Definitions of friends' earnings
def earnings := [18, 22, 26, 32, 47]

-- Definition of total earnings
def totalEarnings := earnings.sum

-- Definition of equal share
def equalShare := totalEarnings / earnings.length

-- The amount that the friend who earned 47 needs to redistribute
def redistributionAmount := 47 - equalShare

-- The goal to prove
theorem friend_redistribution:
  redistributionAmount = 18 := by
  sorry

end NUMINAMATH_GPT_friend_redistribution_l2303_230359


namespace NUMINAMATH_GPT_pencils_profit_goal_l2303_230363

theorem pencils_profit_goal (n : ℕ) (price_purchase price_sale cost_goal : ℚ) (purchase_quantity : ℕ) 
  (h1 : price_purchase = 0.10) 
  (h2 : price_sale = 0.25) 
  (h3 : cost_goal = 100) 
  (h4 : purchase_quantity = 1500) 
  (h5 : n * price_sale ≥ purchase_quantity * price_purchase + cost_goal) :
  n ≥ 1000 :=
sorry

end NUMINAMATH_GPT_pencils_profit_goal_l2303_230363


namespace NUMINAMATH_GPT_rectangular_garden_side_length_l2303_230317

theorem rectangular_garden_side_length (a b : ℝ) (h1 : 2 * a + 2 * b = 60) (h2 : a * b = 200) (h3 : b = 10) : a = 20 :=
by
  sorry

end NUMINAMATH_GPT_rectangular_garden_side_length_l2303_230317


namespace NUMINAMATH_GPT_average_price_of_racket_l2303_230339

theorem average_price_of_racket
  (total_amount_made : ℝ)
  (number_of_pairs_sold : ℕ)
  (h1 : total_amount_made = 490) 
  (h2 : number_of_pairs_sold = 50) : 
  (total_amount_made / number_of_pairs_sold : ℝ) = 9.80 := 
  by
  sorry

end NUMINAMATH_GPT_average_price_of_racket_l2303_230339


namespace NUMINAMATH_GPT_greg_initial_money_eq_36_l2303_230329

theorem greg_initial_money_eq_36 
  (Earl_initial Fred_initial : ℕ)
  (Greg_initial : ℕ)
  (Earl_owes_Fred Fred_owes_Greg Greg_owes_Earl : ℕ)
  (Total_after_debt : ℕ)
  (hEarl_initial : Earl_initial = 90)
  (hFred_initial : Fred_initial = 48)
  (hEarl_owes_Fred : Earl_owes_Fred = 28)
  (hFred_owes_Greg : Fred_owes_Greg = 32)
  (hGreg_owes_Earl : Greg_owes_Earl = 40)
  (hTotal_after_debt : Total_after_debt = 130) :
  Greg_initial = 36 :=
sorry

end NUMINAMATH_GPT_greg_initial_money_eq_36_l2303_230329


namespace NUMINAMATH_GPT_diamond_comm_not_assoc_l2303_230301

def diamond (a b : ℤ) : ℤ := (a * b + 5) / (a + b)

-- Lemma: Verify commutativity of the diamond operation
lemma diamond_comm (a b : ℤ) (ha : a > 1) (hb : b > 1) : 
  diamond a b = diamond b a := by
  sorry

-- Lemma: Verify non-associativity of the diamond operation
lemma diamond_not_assoc (a b c : ℤ) (ha : a > 1) (hb : b > 1) (hc : c > 1) :
  diamond (diamond a b) c ≠ diamond a (diamond b c) := by
  sorry

-- Theorem: The diamond operation is commutative but not associative
theorem diamond_comm_not_assoc (a b c : ℤ) (ha : a > 1) (hb : b > 1) (hc : c > 1) :
  diamond a b = diamond b a ∧ diamond (diamond a b) c ≠ diamond a (diamond b c) := by
  apply And.intro
  · apply diamond_comm
    apply ha
    apply hb
  · apply diamond_not_assoc
    apply ha
    apply hb
    apply hc

end NUMINAMATH_GPT_diamond_comm_not_assoc_l2303_230301


namespace NUMINAMATH_GPT_brittany_average_correct_l2303_230327

def brittany_first_score : ℤ :=
78

def brittany_second_score : ℤ :=
84

def brittany_average_after_second_test (score1 score2 : ℤ) : ℤ :=
(score1 + score2) / 2

theorem brittany_average_correct : 
  brittany_average_after_second_test brittany_first_score brittany_second_score = 81 := 
by
  sorry

end NUMINAMATH_GPT_brittany_average_correct_l2303_230327


namespace NUMINAMATH_GPT_system_no_solution_iff_n_eq_neg_one_l2303_230300

def no_solution_system (n : ℝ) : Prop :=
  ¬∃ x y z : ℝ, (n * x + y = 1) ∧ (n * y + z = 1) ∧ (x + n * z = 1)

theorem system_no_solution_iff_n_eq_neg_one (n : ℝ) : no_solution_system n ↔ n = -1 :=
sorry

end NUMINAMATH_GPT_system_no_solution_iff_n_eq_neg_one_l2303_230300


namespace NUMINAMATH_GPT_sum_first_15_terms_l2303_230389

-- Define the arithmetic sequence
def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

-- Define the sum of the first n terms of the sequence
def sum_first_n_terms (a d : ℤ) (n : ℕ) : ℤ := n * (2 * a + (n - 1) * d) / 2

-- Conditions
def a_7 := 1
def a_9 := 5

-- Prove that S_15 = 45
theorem sum_first_15_terms : 
  ∃ (a d : ℤ), 
    (arithmetic_sequence a d 7 = a_7) ∧ 
    (arithmetic_sequence a d 9 = a_9) ∧ 
    (sum_first_n_terms a d 15 = 45) :=
sorry

end NUMINAMATH_GPT_sum_first_15_terms_l2303_230389


namespace NUMINAMATH_GPT_cylinder_volume_increase_l2303_230355

theorem cylinder_volume_increase (r h : ℝ) (V : ℝ) (hV : V = Real.pi * r^2 * h) :
    let new_height := 3 * h
    let new_radius := 2.5 * r
    let new_volume := Real.pi * (new_radius ^ 2) * new_height
    new_volume = 18.75 * V :=
by
  sorry

end NUMINAMATH_GPT_cylinder_volume_increase_l2303_230355


namespace NUMINAMATH_GPT_rational_function_domain_l2303_230334

noncomputable def h (x : ℝ) : ℝ := (x^3 - 3*x^2 - 4*x + 5) / (x^2 - 5*x + 4)

theorem rational_function_domain :
  {x : ℝ | ∃ y, h y = h x } = {x : ℝ | x ≠ 1 ∧ x ≠ 4} := 
sorry

end NUMINAMATH_GPT_rational_function_domain_l2303_230334


namespace NUMINAMATH_GPT_tangent_circle_line_l2303_230308

theorem tangent_circle_line (r : ℝ) (h_pos : 0 < r) 
  (h_circle : ∀ x y : ℝ, x^2 + y^2 = r^2) 
  (h_line : ∀ x y : ℝ, x + y = r + 1) : 
  r = 1 + Real.sqrt 2 := 
by 
  sorry

end NUMINAMATH_GPT_tangent_circle_line_l2303_230308


namespace NUMINAMATH_GPT_probability_median_five_l2303_230313

theorem probability_median_five {S : Finset ℕ} (hS : S = {1, 2, 3, 4, 5, 6, 7, 8}) :
  let n := 8
  let k := 5
  let total_ways := Nat.choose n k
  let ways_median_5 := Nat.choose 4 2 * Nat.choose 3 2
  (ways_median_5 : ℚ) / (total_ways : ℚ) = (9 : ℚ) / (28 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_probability_median_five_l2303_230313


namespace NUMINAMATH_GPT_original_number_of_cats_l2303_230349

theorem original_number_of_cats (C : ℕ) : 
  (C - 600) / 2 = 600 → C = 1800 :=
by
  sorry

end NUMINAMATH_GPT_original_number_of_cats_l2303_230349


namespace NUMINAMATH_GPT_find_k_l2303_230393

theorem find_k (k : ℝ) : 
  (∃ (x y : ℝ), 2 * x + 3 * y + 8 = 0 ∧ x - y - 1 = 0 ∧ x + k * y = 0) → k = -1 / 2 :=
by 
  sorry

end NUMINAMATH_GPT_find_k_l2303_230393


namespace NUMINAMATH_GPT_gcf_and_multiples_l2303_230399

theorem gcf_and_multiples (a b gcf : ℕ) : 
  (a = 90) → (b = 135) → gcd a b = gcf → 
  (gcf = 45) ∧ (45 % gcf = 0) ∧ (90 % gcf = 0) ∧ (135 % gcf = 0) := 
by
  intros ha hb hgcf
  rw [ha, hb] at hgcf
  sorry

end NUMINAMATH_GPT_gcf_and_multiples_l2303_230399


namespace NUMINAMATH_GPT_cross_section_is_rectangle_l2303_230344

def RegularTetrahedron : Type := sorry

def Plane : Type := sorry

variable (T : RegularTetrahedron) (P : Plane)

-- Conditions
axiom regular_tetrahedron (T : RegularTetrahedron) : Prop
axiom plane_intersects_tetrahedron (P : Plane) (T : RegularTetrahedron) : Prop
axiom plane_parallel_opposite_edges (P : Plane) (T : RegularTetrahedron) : Prop

-- The cross-section formed by intersecting a regular tetrahedron with a plane
-- that is parallel to two opposite edges is a rectangle.
theorem cross_section_is_rectangle (T : RegularTetrahedron) (P : Plane) 
  (hT : regular_tetrahedron T) 
  (hI : plane_intersects_tetrahedron P T) 
  (hP : plane_parallel_opposite_edges P T) :
  ∃ (shape : Type), shape = Rectangle := 
  sorry

end NUMINAMATH_GPT_cross_section_is_rectangle_l2303_230344


namespace NUMINAMATH_GPT_spherical_to_rectangular_coordinates_l2303_230395

-- Define the given conditions
variable (ρ : ℝ) (θ : ℝ) (φ : ℝ)
variable (hρ : ρ = 6) (hθ : θ = 7 * Real.pi / 4) (hφ : φ = Real.pi / 2)

-- Convert spherical coordinates (ρ, θ, φ) to rectangular coordinates (x, y, z) and prove the values
theorem spherical_to_rectangular_coordinates :
  let x := ρ * Real.sin φ * Real.cos θ
  let y := ρ * Real.sin φ * Real.sin θ
  let z := ρ * Real.cos φ
  x = 3 * Real.sqrt 2 ∧ y = -3 * Real.sqrt 2 ∧ z = 0 :=
by
  sorry

end NUMINAMATH_GPT_spherical_to_rectangular_coordinates_l2303_230395


namespace NUMINAMATH_GPT_money_left_l2303_230390

noncomputable def initial_amount : ℕ := 100
noncomputable def cost_roast : ℕ := 17
noncomputable def cost_vegetables : ℕ := 11

theorem money_left (init_amt cost_r cost_v : ℕ) 
  (h1 : init_amt = 100)
  (h2 : cost_r = 17)
  (h3 : cost_v = 11) : init_amt - (cost_r + cost_v) = 72 := by
  sorry

end NUMINAMATH_GPT_money_left_l2303_230390


namespace NUMINAMATH_GPT_problem_statement_l2303_230325

theorem problem_statement {a b c d : ℝ} (h1 : a > b) (h2 : b > c) (h3 : c > d) :
  (1 / (a - b)) + (4 / (b - c)) + (9 / (c - d)) ≥ (36 / (a - d)) :=
by
  sorry -- proof is omitted according to the instructions

end NUMINAMATH_GPT_problem_statement_l2303_230325


namespace NUMINAMATH_GPT_voter_ratio_l2303_230364

theorem voter_ratio (Vx Vy : ℝ) (hx : 0.72 * Vx + 0.36 * Vy = 0.60 * (Vx + Vy)) : Vx = 2 * Vy :=
by
sorry

end NUMINAMATH_GPT_voter_ratio_l2303_230364


namespace NUMINAMATH_GPT_suzie_store_revenue_l2303_230347

theorem suzie_store_revenue 
  (S B : ℝ) 
  (h1 : B = S + 15) 
  (h2 : 22 * S + 16 * B = 460) : 
  8 * S + 32 * B = 711.60 :=
by
  sorry

end NUMINAMATH_GPT_suzie_store_revenue_l2303_230347


namespace NUMINAMATH_GPT_sequence_is_decreasing_l2303_230303

noncomputable def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n, a (n + 1) = r * a n

theorem sequence_is_decreasing (a : ℕ → ℝ) (h1 : a 1 < 0) (h2 : is_geometric_sequence a (1/3)) :
  ∀ n, a (n + 1) < a n :=
by
  -- Here should be the proof
  sorry

end NUMINAMATH_GPT_sequence_is_decreasing_l2303_230303


namespace NUMINAMATH_GPT_total_books_l2303_230365

def sam_books := 110
def joan_books := 102
def tom_books := 125
def alice_books := 97

theorem total_books : sam_books + joan_books + tom_books + alice_books = 434 :=
by
  sorry

end NUMINAMATH_GPT_total_books_l2303_230365


namespace NUMINAMATH_GPT_not_a_fraction_l2303_230394

axiom x : ℝ
axiom a : ℝ
axiom b : ℝ

noncomputable def A := 1 / (x^2)
noncomputable def B := (b + 3) / a
noncomputable def C := (x^2 - 1) / (x + 1)
noncomputable def D := (2 / 7) * a

theorem not_a_fraction : ¬ (D = A) ∧ ¬ (D = B) ∧ ¬ (D = C) :=
by 
  sorry

end NUMINAMATH_GPT_not_a_fraction_l2303_230394


namespace NUMINAMATH_GPT_price_of_each_bottle_is_3_l2303_230387

/-- Each bottle of iced coffee has 6 servings. -/
def servings_per_bottle : ℕ := 6

/-- Tricia drinks half a container (bottle) a day. -/
def daily_consumption_rate : ℕ := servings_per_bottle / 2

/-- Number of days in 2 weeks. -/
def duration_days : ℕ := 14

/-- Number of servings Tricia consumes in 2 weeks. -/
def total_servings : ℕ := daily_consumption_rate * duration_days

/-- Number of bottles needed to get the total servings. -/
def bottles_needed : ℕ := total_servings / servings_per_bottle

/-- The total cost of the bottles is $21. -/
def total_cost : ℕ := 21

/-- The price per bottle is the total cost divided by the number of bottles. -/
def price_per_bottle : ℕ := total_cost / bottles_needed

/-- The price of each bottle is $3. -/
theorem price_of_each_bottle_is_3 : price_per_bottle = 3 :=
by
  -- We assume the necessary steps and mathematical verifications have been done.
  sorry

end NUMINAMATH_GPT_price_of_each_bottle_is_3_l2303_230387


namespace NUMINAMATH_GPT_original_bet_is_40_l2303_230353

-- Definition relating payout ratio and payout to original bet
def calculate_original_bet (payout_ratio payout : ℚ) : ℚ :=
  payout / payout_ratio

-- Given conditions
def payout_ratio : ℚ := 3 / 2
def received_payout : ℚ := 60

-- The proof goal
theorem original_bet_is_40 : calculate_original_bet payout_ratio received_payout = 40 :=
by
  sorry

end NUMINAMATH_GPT_original_bet_is_40_l2303_230353


namespace NUMINAMATH_GPT_original_population_l2303_230320

theorem original_population (n : ℕ) (h1 : n + 1500 * 85 / 100 = n - 45) : n = 8800 := 
by
  sorry

end NUMINAMATH_GPT_original_population_l2303_230320


namespace NUMINAMATH_GPT_infinite_squares_of_form_l2303_230391

theorem infinite_squares_of_form (k : ℕ) (hk : k > 0) : ∃ᶠ n in at_top, ∃ m : ℕ, n * 2^k - 7 = m^2 := sorry

end NUMINAMATH_GPT_infinite_squares_of_form_l2303_230391


namespace NUMINAMATH_GPT_jericho_owes_annika_l2303_230310

variable (J A M : ℝ)
variable (h1 : 2 * J = 60)
variable (h2 : M = A / 2)
variable (h3 : 30 - A - M = 9)

theorem jericho_owes_annika :
  A = 14 :=
by
  sorry

end NUMINAMATH_GPT_jericho_owes_annika_l2303_230310


namespace NUMINAMATH_GPT_total_socks_l2303_230321

-- Definitions based on conditions
def red_pairs : ℕ := 20
def red_socks : ℕ := red_pairs * 2
def black_socks : ℕ := red_socks / 2
def white_socks : ℕ := 2 * (red_socks + black_socks)

-- The main theorem we want to prove
theorem total_socks :
  (red_socks + black_socks + white_socks) = 180 := by
  sorry

end NUMINAMATH_GPT_total_socks_l2303_230321


namespace NUMINAMATH_GPT_minuend_calculation_l2303_230379

theorem minuend_calculation (subtrahend difference : ℕ) (h : subtrahend + difference + 300 = 600) :
  300 = 300 :=
sorry

end NUMINAMATH_GPT_minuend_calculation_l2303_230379


namespace NUMINAMATH_GPT_find_train_speed_l2303_230312

def train_speed (v t_pole t_stationary d_stationary : ℕ) : ℕ := v

theorem find_train_speed (v : ℕ) (t_pole : ℕ) (t_stationary : ℕ) (d_stationary : ℕ) :
  t_pole = 5 →
  t_stationary = 25 →
  d_stationary = 360 →
  25 * v = 5 * v + d_stationary →
  v = 18 :=
by intros h1 h2 h3 h4; sorry

end NUMINAMATH_GPT_find_train_speed_l2303_230312


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l2303_230366

def p (x : ℝ) := x^2 + x - 2 > 0
def q (x a : ℝ) := x > a

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (∀ x, q x a → p x) ∧ (∃ x, ¬q x a ∧ p x) → a ∈ Set.Ici 1 :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l2303_230366


namespace NUMINAMATH_GPT_combined_weight_loss_l2303_230396

theorem combined_weight_loss :
  let aleesia_loss_per_week := 1.5
  let aleesia_weeks := 10
  let alexei_loss_per_week := 2.5
  let alexei_weeks := 8
  (aleesia_loss_per_week * aleesia_weeks) + (alexei_loss_per_week * alexei_weeks) = 35 := by
sorry

end NUMINAMATH_GPT_combined_weight_loss_l2303_230396


namespace NUMINAMATH_GPT_positive_divisors_60_l2303_230311

theorem positive_divisors_60 : ∃ n : ℕ, n = 12 ∧ (∀ d : ℕ, d ∣ 60 → d > 0 → ∃ (divisors_set : Finset ℕ), divisors_set.card = n ∧ ∀ x, x ∈ divisors_set ↔ x ∣ 60 ) :=
by
  sorry

end NUMINAMATH_GPT_positive_divisors_60_l2303_230311


namespace NUMINAMATH_GPT_exams_in_fourth_year_l2303_230336

noncomputable def student_exam_counts 
  (a_1 a_2 a_3 a_4 a_5 : ℕ) : Prop :=
  a_1 + a_2 + a_3 + a_4 + a_5 = 31 ∧ 
  a_5 = 3 * a_1 ∧ 
  a_1 < a_2 ∧ 
  a_2 < a_3 ∧ 
  a_3 < a_4 ∧ 
  a_4 < a_5

theorem exams_in_fourth_year 
  (a_1 a_2 a_3 a_4 a_5 : ℕ) (h : student_exam_counts a_1 a_2 a_3 a_4 a_5) : 
  a_4 = 8 :=
sorry

end NUMINAMATH_GPT_exams_in_fourth_year_l2303_230336


namespace NUMINAMATH_GPT_phone_numbers_even_phone_numbers_odd_phone_numbers_ratio_l2303_230361

def even_digits : Set ℕ := { 0, 2, 4, 6, 8 }
def odd_digits : Set ℕ := { 1, 3, 5, 7, 9 }

theorem phone_numbers_even : (4 * 5^6) = 62500 := by
  sorry

theorem phone_numbers_odd : 5^7 = 78125 := by
  sorry

theorem phone_numbers_ratio
  (evens : (4 * 5^6) = 62500)
  (odds : 5^7 = 78125) :
  (78125 / 62500 : ℝ) = 1.25 := by
    sorry

end NUMINAMATH_GPT_phone_numbers_even_phone_numbers_odd_phone_numbers_ratio_l2303_230361


namespace NUMINAMATH_GPT_sum_of_faces_edges_vertices_l2303_230384

def rectangular_prism_faces : ℕ := 6
def rectangular_prism_edges : ℕ := 12
def rectangular_prism_vertices : ℕ := 8

theorem sum_of_faces_edges_vertices : 
  rectangular_prism_faces + rectangular_prism_edges + rectangular_prism_vertices = 26 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_faces_edges_vertices_l2303_230384


namespace NUMINAMATH_GPT_line_ellipse_intersection_l2303_230315

theorem line_ellipse_intersection (m : ℝ) : 
  (∀ x y : ℝ, y = 2 * x + m ∧ (x^2 / 4 + y^2 / 2 = 1)) →
  (-3 * Real.sqrt 2 < m ∧ m < 3 * Real.sqrt 2) ∨
  (m = 3 * Real.sqrt 2 ∨ m = -3 * Real.sqrt 2) ∨ 
  (m < -3 * Real.sqrt 2 ∨ m > 3 * Real.sqrt 2) :=
sorry

end NUMINAMATH_GPT_line_ellipse_intersection_l2303_230315


namespace NUMINAMATH_GPT_upsilon_value_l2303_230372

theorem upsilon_value (Upsilon : ℤ) (h : 5 * (-3) = Upsilon - 3) : Upsilon = -12 :=
by
  sorry

end NUMINAMATH_GPT_upsilon_value_l2303_230372


namespace NUMINAMATH_GPT_problem_solution_l2303_230331

-- Definitions
def has_property_P (A : List ℕ) : Prop :=
  ∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ A.length →
    (A.get! (j - 1) + A.get! (i - 1) ∈ A ∨ A.get! (j - 1) - A.get! (i - 1) ∈ A)

def sequence_01234 := [0, 2, 4, 6]

-- Propositions
def proposition_1 : Prop := has_property_P sequence_01234

def proposition_2 (A : List ℕ) : Prop := 
  has_property_P A → (A.headI = 0)

def proposition_3 (A : List ℕ) : Prop :=
  has_property_P A → A.headI ≠ 0 →
  ∀ k, 1 ≤ k ∧ k < A.length → A.get! (A.length - 1) - A.get! (A.length - 1 - k) = A.get! k

def proposition_4 (A : List ℕ) : Prop :=
  has_property_P A → A.length = 3 →
  A.get! 2 = A.get! 0 + A.get! 1

-- Main statement
theorem problem_solution : 
  (proposition_1) ∧
  (∃ A, ¬ (proposition_2 A)) ∧
  (∃ A, proposition_3 A) ∧
  (∃ A, proposition_4 A) →
  3 = 3 := 
by sorry

end NUMINAMATH_GPT_problem_solution_l2303_230331


namespace NUMINAMATH_GPT_river_length_l2303_230305

theorem river_length (S C : ℝ) (h1 : S = C / 3) (h2 : S + C = 80) : S = 20 :=
by 
  sorry

end NUMINAMATH_GPT_river_length_l2303_230305


namespace NUMINAMATH_GPT_triangle_inequality_l2303_230375

theorem triangle_inequality
  (A B C : ℝ)
  (hA : 0 < A)
  (hB : 0 < B)
  (hC : 0 < C)
  (hABC : A + B + C = Real.pi) :
  Real.sin (3 * A / 2) + Real.sin (3 * B / 2) + Real.sin (3 * C / 2) ≤
  Real.cos ((A - B) / 2) + Real.cos ((B - C) / 2) + Real.cos ((C - A) / 2) :=
by
  sorry

end NUMINAMATH_GPT_triangle_inequality_l2303_230375


namespace NUMINAMATH_GPT_weight_of_person_replaced_l2303_230350

def initial_total_weight (W : ℝ) : ℝ := W
def new_person_weight : ℝ := 137
def average_increase : ℝ := 7.2
def group_size : ℕ := 10

theorem weight_of_person_replaced 
(W : ℝ) 
(weight_replaced : ℝ) 
(h1 : (W / group_size) + average_increase = (W - weight_replaced + new_person_weight) / group_size) : 
weight_replaced = 65 := 
sorry

end NUMINAMATH_GPT_weight_of_person_replaced_l2303_230350


namespace NUMINAMATH_GPT_lowest_possible_students_l2303_230382

-- Definitions based on conditions
def isDivisibleBy (n m : ℕ) : Prop := n % m = 0

def canBeDividedIntoTeams (num_students num_teams : ℕ) : Prop := isDivisibleBy num_students num_teams

-- Theorem statement for the lowest possible number of students
theorem lowest_possible_students (n : ℕ) : 
  (canBeDividedIntoTeams n 8) ∧ (canBeDividedIntoTeams n 12) → n = 24 := by
  sorry

end NUMINAMATH_GPT_lowest_possible_students_l2303_230382


namespace NUMINAMATH_GPT_min_value_N_l2303_230340

theorem min_value_N (a b c d e f : ℤ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : 0 < d) (h₄ : 0 < e) (h₅ : 0 < f)
  (h_sum : a + b + c + d + e + f = 4020) :
  ∃ N : ℤ, N = max (a + b) (max (b + c) (max (c + d) (max (d + e) (e + f)))) ∧ N = 805 :=
by
  sorry

end NUMINAMATH_GPT_min_value_N_l2303_230340


namespace NUMINAMATH_GPT_correct_phone_call_sequence_l2303_230342

-- Define the six steps as an enumerated type.
inductive Step
| Dial
| WaitDialTone
| PickUpHandset
| StartConversationOrHangUp
| WaitSignal
| EndCall

open Step

-- Define the problem as a theorem.
theorem correct_phone_call_sequence : 
  ∃ sequence : List Step, sequence = [PickUpHandset, WaitDialTone, Dial, WaitSignal, StartConversationOrHangUp, EndCall] :=
sorry

end NUMINAMATH_GPT_correct_phone_call_sequence_l2303_230342


namespace NUMINAMATH_GPT_school_fee_correct_l2303_230319

-- Definitions
def mother_fifty_bills : ℕ := 1
def mother_twenty_bills : ℕ := 2
def mother_ten_bills : ℕ := 3

def father_fifty_bills : ℕ := 4
def father_twenty_bills : ℕ := 1
def father_ten_bills : ℕ := 1

def total_fifty_bills : ℕ := mother_fifty_bills + father_fifty_bills
def total_twenty_bills : ℕ := mother_twenty_bills + father_twenty_bills
def total_ten_bills : ℕ := mother_ten_bills + father_ten_bills

def value_fifty_bills : ℕ := 50 * total_fifty_bills
def value_twenty_bills : ℕ := 20 * total_twenty_bills
def value_ten_bills : ℕ := 10 * total_ten_bills

-- Theorem
theorem school_fee_correct :
  value_fifty_bills + value_twenty_bills + value_ten_bills = 350 :=
by
  sorry

end NUMINAMATH_GPT_school_fee_correct_l2303_230319


namespace NUMINAMATH_GPT_add_candies_to_equalize_l2303_230323

-- Define the initial number of candies in basket A and basket B
def candiesInA : ℕ := 8
def candiesInB : ℕ := 17

-- Problem statement: Prove that adding 9 more candies to basket A
-- makes the number of candies in basket A equal to that in basket B.
theorem add_candies_to_equalize : ∃ n : ℕ, candiesInA + n = candiesInB :=
by
  use 9  -- The value we are adding to the candies in basket A
  sorry  -- Proof goes here

end NUMINAMATH_GPT_add_candies_to_equalize_l2303_230323


namespace NUMINAMATH_GPT_number_satisfies_equation_l2303_230352

theorem number_satisfies_equation :
  ∃ x : ℝ, (x^2 + 100 = (x - 20)^2) ∧ x = 7.5 :=
by
  use 7.5
  sorry

end NUMINAMATH_GPT_number_satisfies_equation_l2303_230352


namespace NUMINAMATH_GPT_point_A_in_Quadrant_IV_l2303_230324

-- Define the coordinates of point A
def A : ℝ × ℝ := (5, -4)

-- Define the quadrants based on x and y signs
def in_Quadrant_I (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 > 0
def in_Quadrant_II (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 > 0
def in_Quadrant_III (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 < 0
def in_Quadrant_IV (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 < 0

-- Statement to prove that point A lies in Quadrant IV
theorem point_A_in_Quadrant_IV : in_Quadrant_IV A :=
by
  sorry

end NUMINAMATH_GPT_point_A_in_Quadrant_IV_l2303_230324


namespace NUMINAMATH_GPT_intersection_result_l2303_230386

open Set

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := { x : ℝ | abs (x - 1) > 2 }

-- Define set B
def B : Set ℝ := { x : ℝ | -x^2 + 6 * x - 8 > 0 }

-- Define the complement of A in U
def compl_A : Set ℝ := U \ A

-- Define the intersection of compl_A and B
def inter_complA_B : Set ℝ := compl_A ∩ B

-- Prove that the intersection is equal to the given set
theorem intersection_result : inter_complA_B = { x : ℝ | 2 < x ∧ x ≤ 3 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_result_l2303_230386


namespace NUMINAMATH_GPT_students_count_inconsistent_l2303_230371

-- Define the conditions
variables (total_students boys_more_than_girls : ℤ)

-- Define the main theorem: The computed number of girls is not an integer
theorem students_count_inconsistent 
  (h1 : total_students = 3688) 
  (h2 : boys_more_than_girls = 373) 
  : ¬ ∃ x : ℤ, 2 * x + boys_more_than_girls = total_students := 
by
  sorry

end NUMINAMATH_GPT_students_count_inconsistent_l2303_230371


namespace NUMINAMATH_GPT_arithmetic_sum_problem_l2303_230326

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (n * (a 1 + a n)) / 2

theorem arithmetic_sum_problem (a : ℕ → ℤ) (S : ℕ → ℤ) 
  (h_arith_seq : arithmetic_sequence a)
  (h_S_def : ∀ n : ℕ, S n = sum_of_first_n_terms a n)
  (h_S13 : S 13 = 52) : a 4 + a 8 + a 9 = 12 :=
sorry

end NUMINAMATH_GPT_arithmetic_sum_problem_l2303_230326


namespace NUMINAMATH_GPT_work_together_days_l2303_230354

-- Define the days it takes for A and B to complete the work individually.
def days_A : ℕ := 3
def days_B : ℕ := 6

-- Define the combined work rate.
def combined_work_rate : ℚ := (1 / days_A) + (1 / days_B)

-- State the theorem for the number of days A and B together can complete the work.
theorem work_together_days :
  1 / combined_work_rate = 2 := by
  sorry

end NUMINAMATH_GPT_work_together_days_l2303_230354


namespace NUMINAMATH_GPT_johns_profit_l2303_230341

-- Definitions based on Conditions
def original_price_per_bag : ℝ := 4
def discount_percentage : ℝ := 0.10
def discounted_price_per_bag := original_price_per_bag * (1 - discount_percentage)
def bags_bought : ℕ := 30
def cost_per_bag : ℝ := if bags_bought >= 20 then discounted_price_per_bag else original_price_per_bag
def total_cost := bags_bought * cost_per_bag
def bags_sold_to_adults : ℕ := 20
def bags_sold_to_children : ℕ := 10
def price_per_bag_for_adults : ℝ := 8
def price_per_bag_for_children : ℝ := 6
def revenue_from_adults := bags_sold_to_adults * price_per_bag_for_adults
def revenue_from_children := bags_sold_to_children * price_per_bag_for_children
def total_revenue := revenue_from_adults + revenue_from_children
def profit := total_revenue - total_cost

-- Lean Statement to be Proven
theorem johns_profit : profit = 112 :=
by
  sorry

end NUMINAMATH_GPT_johns_profit_l2303_230341


namespace NUMINAMATH_GPT_combined_area_difference_l2303_230332

theorem combined_area_difference :
  let area_11x11 := 2 * (11 * 11)
  let area_5_5x11 := 2 * (5.5 * 11)
  area_11x11 - area_5_5x11 = 121 :=
by
  sorry

end NUMINAMATH_GPT_combined_area_difference_l2303_230332


namespace NUMINAMATH_GPT_smallest_coprime_to_210_l2303_230398

theorem smallest_coprime_to_210 : ∃ x : ℕ, x > 1 ∧ Nat.gcd x 210 = 1 ∧ ∀ y : ℕ, y > 1 ∧ Nat.gcd y 210 = 1 → y ≥ x :=
by
  sorry

end NUMINAMATH_GPT_smallest_coprime_to_210_l2303_230398
