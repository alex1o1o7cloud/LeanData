import Mathlib

namespace NUMINAMATH_GPT_least_positive_integer_multiple_of_53_l1922_192204

-- Define the problem in a Lean statement.
theorem least_positive_integer_multiple_of_53 :
  ∃ x : ℕ, (3 * x) ^ 2 + 2 * 58 * 3 * x + 58 ^ 2 % 53 = 0 ∧ x = 16 :=
by
  sorry

end NUMINAMATH_GPT_least_positive_integer_multiple_of_53_l1922_192204


namespace NUMINAMATH_GPT_expand_binomials_l1922_192226

theorem expand_binomials (x : ℝ) : (3 * x + 4) * (2 * x + 7) = 6 * x^2 + 29 * x + 28 := 
by 
  sorry

end NUMINAMATH_GPT_expand_binomials_l1922_192226


namespace NUMINAMATH_GPT_negation_equivalence_l1922_192289

-- Define the propositions
def proposition (a b : ℝ) : Prop := a > b → a + 1 > b

def negation_proposition (a b : ℝ) : Prop := a ≤ b → a + 1 ≤ b

-- Statement to prove
theorem negation_equivalence (a b : ℝ) : ¬(proposition a b) ↔ negation_proposition a b := 
sorry

end NUMINAMATH_GPT_negation_equivalence_l1922_192289


namespace NUMINAMATH_GPT_mary_balloon_count_l1922_192279

theorem mary_balloon_count (n m : ℕ) (hn : n = 7) (hm : m = 4 * n) : m = 28 :=
by
  sorry

end NUMINAMATH_GPT_mary_balloon_count_l1922_192279


namespace NUMINAMATH_GPT_find_multiple_l1922_192207

theorem find_multiple :
  ∀ (total_questions correct_answers score : ℕ) (m : ℕ),
  total_questions = 100 →
  correct_answers = 90 →
  score = 70 →
  score = correct_answers - m * (total_questions - correct_answers) →
  m = 2 :=
by
  intros total_questions correct_answers score m h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_find_multiple_l1922_192207


namespace NUMINAMATH_GPT_population_doubles_in_35_years_l1922_192229

noncomputable def birth_rate : ℝ := 39.4 / 1000
noncomputable def death_rate : ℝ := 19.4 / 1000
noncomputable def natural_increase_rate : ℝ := birth_rate - death_rate
noncomputable def doubling_time (r: ℝ) : ℝ := 70 / (r * 100)

theorem population_doubles_in_35_years :
  doubling_time natural_increase_rate = 35 := by sorry

end NUMINAMATH_GPT_population_doubles_in_35_years_l1922_192229


namespace NUMINAMATH_GPT_square_side_length_l1922_192251

/-- If the area of a square is 9m^2 + 24mn + 16n^2, then the length of the side of the square is |3m + 4n|. -/
theorem square_side_length (m n : ℝ) (a : ℝ) (h : a^2 = 9 * m^2 + 24 * m * n + 16 * n^2) : a = |3 * m + 4 * n| :=
sorry

end NUMINAMATH_GPT_square_side_length_l1922_192251


namespace NUMINAMATH_GPT_greatest_sum_first_quadrant_l1922_192272

theorem greatest_sum_first_quadrant (x y : ℤ) (hx_pos : 0 < x) (hy_pos : 0 < y) (h_circle : x^2 + y^2 = 49) : x + y ≤ 7 :=
sorry

end NUMINAMATH_GPT_greatest_sum_first_quadrant_l1922_192272


namespace NUMINAMATH_GPT_solve_for_y_l1922_192244

theorem solve_for_y (y : ℕ) : 8^4 = 2^y → y = 12 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_y_l1922_192244


namespace NUMINAMATH_GPT_sum_arithmetic_sequence_l1922_192246

def first_term (k : ℕ) : ℕ := k^2 - k + 1

def sum_of_first_k_plus_3_terms (k : ℕ) : ℕ := (k + 3) * (k^2 + (k / 2) + 2)

theorem sum_arithmetic_sequence (k : ℕ) (k_pos : 0 < k) : 
    sum_of_first_k_plus_3_terms k = k^3 + (7 * k^2) / 2 + (15 * k) / 2 + 6 := 
by
  sorry

end NUMINAMATH_GPT_sum_arithmetic_sequence_l1922_192246


namespace NUMINAMATH_GPT_fair_coin_flip_probability_difference_l1922_192285

noncomputable def choose : ℕ → ℕ → ℕ
| _, 0 => 1
| 0, _ => 0
| n + 1, r + 1 => choose n r + choose n (r + 1)

theorem fair_coin_flip_probability_difference :
  let p := 1 / 2
  let p_heads_4_out_of_5 := choose 5 4 * p^4 * p
  let p_heads_5_out_of_5 := p^5
  p_heads_4_out_of_5 - p_heads_5_out_of_5 = 1 / 8 :=
by
  sorry

end NUMINAMATH_GPT_fair_coin_flip_probability_difference_l1922_192285


namespace NUMINAMATH_GPT_Owen_spent_720_dollars_on_burgers_l1922_192254

def days_in_June : ℕ := 30
def burgers_per_day : ℕ := 2
def cost_per_burger : ℕ := 12

def total_burgers (days : ℕ) (burgers_per_day : ℕ) : ℕ :=
  days * burgers_per_day

def total_cost (burgers : ℕ) (cost_per_burger : ℕ) : ℕ :=
  burgers * cost_per_burger

theorem Owen_spent_720_dollars_on_burgers :
  total_cost (total_burgers days_in_June burgers_per_day) cost_per_burger = 720 := by
  sorry

end NUMINAMATH_GPT_Owen_spent_720_dollars_on_burgers_l1922_192254


namespace NUMINAMATH_GPT_simplify_expression_l1922_192252

theorem simplify_expression (x y z : ℝ) :
  3 * (x - (2 * y - 3 * z)) - 2 * ((3 * x - 2 * y) - 4 * z) = -3 * x - 2 * y + 17 * z :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1922_192252


namespace NUMINAMATH_GPT_find_largest_number_l1922_192218

noncomputable def largest_of_three_numbers (x y z : ℝ) : ℝ :=
  if x ≥ y ∧ x ≥ z then x
  else if y ≥ x ∧ y ≥ z then y
  else z

theorem find_largest_number (x y z : ℝ) (h1 : x + y + z = 3) (h2 : xy + xz + yz = -11) (h3 : xyz = 15) :
  largest_of_three_numbers x y z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_GPT_find_largest_number_l1922_192218


namespace NUMINAMATH_GPT_alex_not_read_probability_l1922_192270

def probability_reads : ℚ := 5 / 8
def probability_not_reads : ℚ := 3 / 8

theorem alex_not_read_probability : (1 - probability_reads) = probability_not_reads := 
by
  sorry

end NUMINAMATH_GPT_alex_not_read_probability_l1922_192270


namespace NUMINAMATH_GPT_assignment_plans_l1922_192233

theorem assignment_plans (students locations : ℕ) (library science_museum nursing_home : ℕ) 
  (students_eq : students = 5) (locations_eq : locations = 3) 
  (lib_gt0 : library > 0) (sci_gt0 : science_museum > 0) (nur_gt0 : nursing_home > 0) 
  (lib_science_nursing : library + science_museum + nursing_home = students) : 
  ∃ (assignments : ℕ), assignments = 150 :=
by
  sorry

end NUMINAMATH_GPT_assignment_plans_l1922_192233


namespace NUMINAMATH_GPT_find_n_from_degree_l1922_192286

theorem find_n_from_degree (n : ℕ) (h : 2 + n = 5) : n = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_n_from_degree_l1922_192286


namespace NUMINAMATH_GPT_total_first_half_points_l1922_192245

-- Define the sequences for Tigers and Lions
variables (a ar b d : ℕ)
-- Defining conditions
def tied_first_quarter : Prop := a = b
def geometric_tigers : Prop := ∃ r : ℕ, ar = a * r ∧ ar^2 = a * r^2 ∧ ar^3 = a * r^3
def arithmetic_lions : Prop := b+d = b + d ∧ b+2*d = b + 2*d ∧ b+3*d = b + 3*d
def tigers_win_by_four : Prop := (a + ar + ar^2 + ar^3) = (b + (b + d) + (b + 2*d) + (b + 3*d)) + 4
def score_limit : Prop := (a + ar + ar^2 + ar^3) ≤ 120 ∧ (b + (b + d) + (b + 2*d) + (b + 3*d)) ≤ 120

-- Goal: The total number of points scored by the two teams in the first half is 23
theorem total_first_half_points : tied_first_quarter a b ∧ geometric_tigers a ar ∧ arithmetic_lions b d ∧ tigers_win_by_four a ar b d ∧ score_limit a ar b d → 
(a + ar) + (b + d) = 23 := 
by {
  sorry
}

end NUMINAMATH_GPT_total_first_half_points_l1922_192245


namespace NUMINAMATH_GPT_number_of_girls_l1922_192214

variable (boys : ℕ) (total_children : ℕ)

theorem number_of_girls (h1 : boys = 40) (h2 : total_children = 117) : total_children - boys = 77 :=
by
  sorry

end NUMINAMATH_GPT_number_of_girls_l1922_192214


namespace NUMINAMATH_GPT_find_larger_number_l1922_192203

theorem find_larger_number (L S : ℕ) (h1 : L - S = 2500) (h2 : L = 6 * S + 15) : L = 2997 :=
sorry

end NUMINAMATH_GPT_find_larger_number_l1922_192203


namespace NUMINAMATH_GPT_q_can_be_true_or_false_l1922_192236

theorem q_can_be_true_or_false (p q : Prop) (h1 : ¬(p ∧ q)) (h2 : ¬p) : q ∨ ¬q :=
by
  sorry

end NUMINAMATH_GPT_q_can_be_true_or_false_l1922_192236


namespace NUMINAMATH_GPT_arithmetic_seq_a11_l1922_192267

variable (a : ℕ → ℤ)
variable (d : ℕ → ℤ)

-- Conditions
def arithmetic_sequence : Prop := ∀ n, a (n + 2) - a n = 6
def a1 : Prop := a 1 = 1

-- Statement of the problem
theorem arithmetic_seq_a11 : arithmetic_sequence a ∧ a1 a → a 11 = 31 :=
by sorry

end NUMINAMATH_GPT_arithmetic_seq_a11_l1922_192267


namespace NUMINAMATH_GPT_efficiency_ratio_l1922_192263

theorem efficiency_ratio (r : ℚ) (work_B : ℚ) (work_AB : ℚ) (B_alone : ℚ) (AB_together : ℚ) (efficiency_A : ℚ) (B_efficiency : ℚ) :
  B_alone = 30 ∧ AB_together = 20 ∧ B_efficiency = (1/B_alone) ∧ efficiency_A = (r * B_efficiency) ∧ (efficiency_A + B_efficiency) = (1 / AB_together) → r = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_efficiency_ratio_l1922_192263


namespace NUMINAMATH_GPT_y_intercept_probability_l1922_192228

theorem y_intercept_probability (b : ℝ) (hb : b ∈ Set.Icc (-2 : ℝ) 3 ) :
  (∃ P : ℚ, P = (2 / 5)) := 
by 
  sorry

end NUMINAMATH_GPT_y_intercept_probability_l1922_192228


namespace NUMINAMATH_GPT_tan_ratio_is_7_over_3_l1922_192294

open Real

theorem tan_ratio_is_7_over_3 (a b : ℝ) (h1 : sin (a + b) = 5 / 8) (h2 : sin (a - b) = 1 / 4) : (tan a / tan b) = 7 / 3 :=
by
  sorry

end NUMINAMATH_GPT_tan_ratio_is_7_over_3_l1922_192294


namespace NUMINAMATH_GPT_trays_needed_l1922_192258

theorem trays_needed (cookies_classmates cookies_teachers cookies_per_tray : ℕ) 
  (hc1 : cookies_classmates = 276) 
  (hc2 : cookies_teachers = 92) 
  (hc3 : cookies_per_tray = 12) : 
  (cookies_classmates + cookies_teachers + cookies_per_tray - 1) / cookies_per_tray = 31 :=
by
  sorry

end NUMINAMATH_GPT_trays_needed_l1922_192258


namespace NUMINAMATH_GPT_hexagon_chord_problem_l1922_192265

-- Define the conditions of the problem
structure Hexagon :=
  (circumcircle : Type*)
  (inscribed : Prop)
  (AB BC CD : ℕ)
  (DE EF FA : ℕ)
  (chord_length_fraction_form : ℚ) 

-- Define the unique problem from given conditions and correct answer
theorem hexagon_chord_problem (hex : Hexagon) 
  (h1 : hex.inscribed)
  (h2 : hex.AB = 3) (h3 : hex.BC = 3) (h4 : hex.CD = 3)
  (h5 : hex.DE = 5) (h6 : hex.EF = 5) (h7 : hex.FA = 5)
  (h8 : hex.chord_length_fraction_form = 360 / 49) :
  let m := 360
  let n := 49
  m + n = 409 :=
by
  sorry

end NUMINAMATH_GPT_hexagon_chord_problem_l1922_192265


namespace NUMINAMATH_GPT_range_of_a_l1922_192259

def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2 * a * x + 4 > 0
def q (a : ℝ) : Prop := ∀ x : ℝ, (3 - 2 * a) ^ x > 0 -- using our characterization for 'increasing'

theorem range_of_a (a : ℝ) : (p a ∨ q a) ∧ ¬ (p a ∧ q a) ↔ (a ≤ -2 ∨ (1 ≤ a ∧ a < 2)) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1922_192259


namespace NUMINAMATH_GPT_cans_of_soda_l1922_192250

variable (T R E : ℝ)

theorem cans_of_soda (hT: T > 0) (hR: R > 0) (hE: E > 0) : 5 * E * T / R = (5 * E) / R * T :=
by
  sorry

end NUMINAMATH_GPT_cans_of_soda_l1922_192250


namespace NUMINAMATH_GPT_new_fish_received_l1922_192274

def initial_fish := 14
def added_fish := 2
def eaten_fish := 6
def final_fish := 11

def current_fish := initial_fish + added_fish - eaten_fish
def returned_fish := 2
def exchanged_fish := final_fish - current_fish

theorem new_fish_received : exchanged_fish = 1 := by
  sorry

end NUMINAMATH_GPT_new_fish_received_l1922_192274


namespace NUMINAMATH_GPT_simplify_fraction_l1922_192292

theorem simplify_fraction (a b : ℝ) (h : a ≠ b): 
  (a - b) / a / (a - (2 * a * b - b^2) / a) = 1 / (a - b) :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1922_192292


namespace NUMINAMATH_GPT_largest_x_l1922_192215

def largest_x_with_condition_eq_7_over_8 (x : ℝ) : Prop :=
  ⌊x⌋ / x = 7 / 8

theorem largest_x (x : ℝ) (h : largest_x_with_condition_eq_7_over_8 x) :
  x = 48 / 7 :=
sorry

end NUMINAMATH_GPT_largest_x_l1922_192215


namespace NUMINAMATH_GPT_bronchitis_option_D_correct_l1922_192273

noncomputable def smoking_related_to_bronchitis : Prop :=
  -- Conclusion that "smoking is related to chronic bronchitis"
sorry

noncomputable def confidence_level : ℝ :=
  -- Confidence level in the conclusion
  0.99

theorem bronchitis_option_D_correct :
  smoking_related_to_bronchitis →
  (confidence_level > 0.99) →
  -- Option D is correct: "Among 100 smokers, it is possible that not a single person has chronic bronchitis"
  ∃ (P : ℕ → Prop), (∀ n : ℕ, n ≤ 100 → P n = False) :=
by sorry

end NUMINAMATH_GPT_bronchitis_option_D_correct_l1922_192273


namespace NUMINAMATH_GPT_find_g2_l1922_192249

theorem find_g2 (g : ℝ → ℝ) (h : ∀ x : ℝ, x ≠ 0 → g x - 3 * g (1 / x) = 3 ^ x) : 
  g 2 = (9 - 3 * Real.sqrt 3) / 8 := 
sorry

end NUMINAMATH_GPT_find_g2_l1922_192249


namespace NUMINAMATH_GPT_cat_head_start_15_minutes_l1922_192200

theorem cat_head_start_15_minutes :
  ∀ (t : ℕ), (25 : ℝ) = (20 : ℝ) * (1 + (t : ℝ) / 60) → t = 15 := by
  sorry

end NUMINAMATH_GPT_cat_head_start_15_minutes_l1922_192200


namespace NUMINAMATH_GPT_scrabble_letter_values_l1922_192210

-- Definitions based on conditions
def middle_letter_value : ℕ := 8
def final_score : ℕ := 30

-- The theorem we need to prove
theorem scrabble_letter_values (F T : ℕ)
  (h1 : 3 * (F + middle_letter_value + T) = final_score) :
  F = 1 ∧ T = 1 :=
sorry

end NUMINAMATH_GPT_scrabble_letter_values_l1922_192210


namespace NUMINAMATH_GPT_solve_for_n_l1922_192290

theorem solve_for_n (n : ℕ) (h : 2 * 2 * 3 * 3 * 5 * 6 = 5 * 6 * n * n) : n = 6 := by
  sorry

end NUMINAMATH_GPT_solve_for_n_l1922_192290


namespace NUMINAMATH_GPT_grid_square_division_l1922_192206

theorem grid_square_division (m n k : ℕ) (h : m * m = n * k) : ℕ := sorry

end NUMINAMATH_GPT_grid_square_division_l1922_192206


namespace NUMINAMATH_GPT_total_pokemon_cards_l1922_192219

def pokemon_cards (sam dan tom keith : Nat) : Nat :=
  sam + dan + tom + keith

theorem total_pokemon_cards :
  pokemon_cards 14 14 14 14 = 56 := by
  sorry

end NUMINAMATH_GPT_total_pokemon_cards_l1922_192219


namespace NUMINAMATH_GPT_dimension_proof_l1922_192242

noncomputable def sports_field_dimensions (x y: ℝ) : Prop :=
  -- Given conditions
  x^2 + y^2 = 185^2 ∧
  (x - 4) * (y - 4) = x * y - 1012 ∧
  -- Seeking to prove dimensions
  ((x = 153 ∧ y = 104) ∨ (x = 104 ∧ y = 153))

theorem dimension_proof : ∃ x y: ℝ, sports_field_dimensions x y := by
  sorry

end NUMINAMATH_GPT_dimension_proof_l1922_192242


namespace NUMINAMATH_GPT_edward_original_amount_l1922_192261

theorem edward_original_amount (spent left total : ℕ) (h1 : spent = 13) (h2 : left = 6) (h3 : total = spent + left) : total = 19 := by 
  sorry

end NUMINAMATH_GPT_edward_original_amount_l1922_192261


namespace NUMINAMATH_GPT_probability_diff_suits_l1922_192216

theorem probability_diff_suits (n : ℕ) (h₁ : n = 65) (suits : ℕ) (h₂ : suits = 5) (cards_per_suit : ℕ) (h₃ : cards_per_suit = n / suits) : 
  (52 : ℚ) / (64 : ℚ) = (13 : ℚ) / (16 : ℚ) := 
by 
  sorry

end NUMINAMATH_GPT_probability_diff_suits_l1922_192216


namespace NUMINAMATH_GPT_ratio_time_B_to_A_l1922_192283

-- Definitions for the given conditions
def T_A : ℕ := 10
def work_rate_A : ℚ := 1 / T_A
def combined_work_rate : ℚ := 0.3

-- Lean 4 statement for the problem
theorem ratio_time_B_to_A (T_B : ℚ) (h : (work_rate_A + 1 / T_B) = combined_work_rate) :
  (T_B / T_A) = (1 / 2) := by
  sorry

end NUMINAMATH_GPT_ratio_time_B_to_A_l1922_192283


namespace NUMINAMATH_GPT_no_arrangement_of_1_to_1978_coprime_l1922_192222

theorem no_arrangement_of_1_to_1978_coprime :
  ¬ ∃ (a : Fin 1978 → ℕ), 
    (∀ i : Fin 1977, Nat.gcd (a i) (a (i + 1)) = 1) ∧ 
    (∀ i : Fin 1976, Nat.gcd (a i) (a (i + 2)) = 1) ∧ 
    (∀ i : Fin 1978, 1 ≤ a i ∧ a i ≤ 1978 ∧ ∀ j : Fin 1978, (i ≠ j → a i ≠ a j)) :=
sorry

end NUMINAMATH_GPT_no_arrangement_of_1_to_1978_coprime_l1922_192222


namespace NUMINAMATH_GPT_packs_of_cake_l1922_192235

-- Given conditions
def total_grocery_packs : ℕ := 27
def cookie_packs : ℕ := 23

-- Question: How many packs of cake did Lucy buy?
-- Mathematically equivalent problem: Proving that cake_packs is 4
theorem packs_of_cake : (total_grocery_packs - cookie_packs) = 4 :=
by
  -- Proof goes here. Using sorry to skip the proof.
  sorry

end NUMINAMATH_GPT_packs_of_cake_l1922_192235


namespace NUMINAMATH_GPT_eval_expression_l1922_192231

theorem eval_expression : 0.5 * 0.8 - 0.2 = 0.2 := by
  sorry

end NUMINAMATH_GPT_eval_expression_l1922_192231


namespace NUMINAMATH_GPT_proof_part1_proof_part2_l1922_192243

variable {a b c : ℝ}

def pos_and_sum_of_powers (a b c : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

theorem proof_part1 (h : pos_and_sum_of_powers a b c) : abc ≤ 1 / 9 :=
sorry

theorem proof_part2 (h : pos_and_sum_of_powers a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end NUMINAMATH_GPT_proof_part1_proof_part2_l1922_192243


namespace NUMINAMATH_GPT_chocolate_bar_cost_l1922_192221

-- Define the quantities Jessica bought
def chocolate_bars := 10
def gummy_bears_packs := 10
def chocolate_chips_bags := 20

-- Define the costs
def total_cost := 150
def gummy_bears_pack_cost := 2
def chocolate_chips_bag_cost := 5

-- Define what we want to prove (the cost of one chocolate bar)
theorem chocolate_bar_cost : 
  ∃ chocolate_bar_cost, 
    chocolate_bars * chocolate_bar_cost + 
    gummy_bears_packs * gummy_bears_pack_cost + 
    chocolate_chips_bags * chocolate_chips_bag_cost = total_cost ∧
    chocolate_bar_cost = 3 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_chocolate_bar_cost_l1922_192221


namespace NUMINAMATH_GPT_jose_marks_difference_l1922_192240

theorem jose_marks_difference (M J A : ℕ) 
  (h1 : M = J - 20)
  (h2 : J + M + A = 210)
  (h3 : J = 90) : (J - A) = 40 :=
by
  sorry

end NUMINAMATH_GPT_jose_marks_difference_l1922_192240


namespace NUMINAMATH_GPT_factorial_sum_perfect_square_iff_l1922_192268

def is_perfect_square (n : Nat) : Prop := ∃ m : Nat, m * m = n

def sum_of_factorials (n : Nat) : Nat :=
  (List.range (n + 1)).map Nat.factorial |>.sum

theorem factorial_sum_perfect_square_iff (n : Nat) :
  n = 1 ∨ n = 3 ↔ is_perfect_square (sum_of_factorials n) := by {
  sorry
}

end NUMINAMATH_GPT_factorial_sum_perfect_square_iff_l1922_192268


namespace NUMINAMATH_GPT_todd_savings_l1922_192288

-- Define the initial conditions
def original_price : ℝ := 125
def sale_discount : ℝ := 0.20
def coupon : ℝ := 10
def card_discount : ℝ := 0.10

-- Define the resulting values after applying discounts
def sale_price := original_price * (1 - sale_discount)
def after_coupon := sale_price - coupon
def final_price := after_coupon * (1 - card_discount)

-- Define the total savings
def savings := original_price - final_price

-- The proof statement
theorem todd_savings : savings = 44 := by
  sorry

end NUMINAMATH_GPT_todd_savings_l1922_192288


namespace NUMINAMATH_GPT_tank_capacity_l1922_192282

theorem tank_capacity (C : ℝ) :
  (C / 10 - 960 = C / 18) → C = 21600 := by
  intro h
  sorry

end NUMINAMATH_GPT_tank_capacity_l1922_192282


namespace NUMINAMATH_GPT_price_of_each_shirt_l1922_192208

theorem price_of_each_shirt 
  (toys_cost : ℕ := 3 * 10)
  (cards_cost : ℕ := 2 * 5)
  (total_spent : ℕ := 70)
  (remaining_cost: ℕ := total_spent - (toys_cost + cards_cost))
  (num_shirts : ℕ := 3 + 2) :
  (remaining_cost / num_shirts) = 6 :=
by
  sorry

end NUMINAMATH_GPT_price_of_each_shirt_l1922_192208


namespace NUMINAMATH_GPT_sqrt_k_kn_eq_k_sqrt_kn_l1922_192269

theorem sqrt_k_kn_eq_k_sqrt_kn (k n : ℕ) (h : k = Nat.sqrt (n + 1)) : 
  Real.sqrt (k * (k / n)) = k * Real.sqrt (k / n) := 
sorry

end NUMINAMATH_GPT_sqrt_k_kn_eq_k_sqrt_kn_l1922_192269


namespace NUMINAMATH_GPT_ratio_a7_b7_l1922_192276

-- Definitions of the conditions provided in the problem
variables {a b : ℕ → ℝ}   -- Arithmetic sequences {a_n} and {b_n}
variables {S T : ℕ → ℝ}   -- Sums of the first n terms of {a_n} and {b_n}

-- Condition: For any positive integer n, S_n / T_n = (3n + 5) / (2n + 3)
axiom condition_S_T (n : ℕ) (hn : 0 < n) : S n / T n = (3 * n + 5) / (2 * n + 3)

-- Goal: Prove that a_7 / b_7 = 44 / 29
theorem ratio_a7_b7 : a 7 / b 7 = 44 / 29 := 
sorry

end NUMINAMATH_GPT_ratio_a7_b7_l1922_192276


namespace NUMINAMATH_GPT_combined_area_of_tracts_l1922_192239

theorem combined_area_of_tracts :
  let length1 := 300
  let width1 := 500
  let length2 := 250
  let width2 := 630
  let area1 := length1 * width1
  let area2 := length2 * width2
  let combined_area := area1 + area2
  combined_area = 307500 :=
by
  sorry

end NUMINAMATH_GPT_combined_area_of_tracts_l1922_192239


namespace NUMINAMATH_GPT_division_example_l1922_192241

theorem division_example : ∃ A B : ℕ, 23 = 6 * A + B ∧ A = 3 ∧ B < 6 := 
by sorry

end NUMINAMATH_GPT_division_example_l1922_192241


namespace NUMINAMATH_GPT_second_divisor_is_340_l1922_192225

theorem second_divisor_is_340 
  (n : ℕ)
  (h1 : n = 349)
  (h2 : n % 13 = 11)
  (h3 : n % D = 9) : D = 340 :=
by
  sorry

end NUMINAMATH_GPT_second_divisor_is_340_l1922_192225


namespace NUMINAMATH_GPT_jon_weekly_speed_gain_l1922_192232

-- Definitions based on the conditions
def initial_speed : ℝ := 80
def speed_increase_percentage : ℝ := 0.20
def training_sessions : ℕ := 4
def weeks_per_session : ℕ := 4
def total_training_duration : ℕ := training_sessions * weeks_per_session

-- The calculated final speed
def final_speed : ℝ := initial_speed + initial_speed * speed_increase_percentage

theorem jon_weekly_speed_gain : 
  (final_speed - initial_speed) / total_training_duration = 1 :=
by
  -- This is the statement we want to prove
  sorry

end NUMINAMATH_GPT_jon_weekly_speed_gain_l1922_192232


namespace NUMINAMATH_GPT_calculation_l1922_192201

theorem calculation : (-6)^6 / 6^4 + 4^3 - 7^2 * 2 = 2 :=
by
  -- We add "sorry" here to indicate where the proof would go.
  sorry

end NUMINAMATH_GPT_calculation_l1922_192201


namespace NUMINAMATH_GPT_period_start_time_l1922_192234

/-- A period of time had 4 hours of rain and 5 hours without rain, ending at 5 pm. 
Prove that the period started at 8 am. -/
theorem period_start_time :
  let end_time := 17 -- 5 pm in 24-hour format
  let rainy_hours := 4
  let non_rainy_hours := 5
  let total_hours := rainy_hours + non_rainy_hours
  let start_time := end_time - total_hours
  start_time = 8 :=
by
  sorry

end NUMINAMATH_GPT_period_start_time_l1922_192234


namespace NUMINAMATH_GPT_number_of_math_books_l1922_192213

theorem number_of_math_books (M H : ℕ) (h1 : M + H = 90) (h2 : 4 * M + 5 * H = 397) : M = 53 :=
by
  sorry

end NUMINAMATH_GPT_number_of_math_books_l1922_192213


namespace NUMINAMATH_GPT_average_payment_l1922_192280

-- Each condition from part a) is used as a definition here
variable (n : Nat) (p1 p2 first_payment remaining_payment : Nat)

-- Conditions given in natural language
def payments_every_year : Prop :=
  n = 52 ∧
  first_payment = 410 ∧
  remaining_payment = first_payment + 65 ∧
  p1 = 8 * first_payment ∧
  p2 = 44 * remaining_payment ∧
  p2 = 44 * (first_payment + 65) ∧
  p1 + p2 = 24180

-- The theorem to prove based on the conditions
theorem average_payment 
  (h : payments_every_year n p1 p2 first_payment remaining_payment) 
  : (p1 + p2) / n = 465 := 
sorry  -- Proof is omitted intentionally

end NUMINAMATH_GPT_average_payment_l1922_192280


namespace NUMINAMATH_GPT_interest_rate_correct_l1922_192266

theorem interest_rate_correct :
  let SI := 155
  let P := 810
  let T := 4
  let R := SI * 100 / (P * T)
  R = 155 * 100 / (810 * 4) := 
sorry

end NUMINAMATH_GPT_interest_rate_correct_l1922_192266


namespace NUMINAMATH_GPT_Kat_training_hours_l1922_192291

theorem Kat_training_hours
  (h_strength_times : ℕ)
  (h_strength_hours : ℝ)
  (h_boxing_times : ℕ)
  (h_boxing_hours : ℝ)
  (h_times : h_strength_times = 3)
  (h_strength : h_strength_hours = 1)
  (b_times : h_boxing_times = 4)
  (b_hours : h_boxing_hours = 1.5) :
  h_strength_times * h_strength_hours + h_boxing_times * h_boxing_hours = 9 :=
by
  sorry

end NUMINAMATH_GPT_Kat_training_hours_l1922_192291


namespace NUMINAMATH_GPT_time_to_eliminate_mice_l1922_192256

def total_work : ℝ := 1
def work_done_by_2_cats_in_5_days : ℝ := 0.5
def initial_2_cats : ℕ := 2
def additional_cats : ℕ := 3
def total_initial_days : ℝ := 5
def total_cats : ℕ := initial_2_cats + additional_cats

theorem time_to_eliminate_mice (h : total_initial_days * (work_done_by_2_cats_in_5_days / total_initial_days) = work_done_by_2_cats_in_5_days) : 
  total_initial_days + (total_work - work_done_by_2_cats_in_5_days) / (total_cats * (work_done_by_2_cats_in_5_days / total_initial_days / initial_2_cats)) = 7 := 
by
  sorry

end NUMINAMATH_GPT_time_to_eliminate_mice_l1922_192256


namespace NUMINAMATH_GPT_ten_percent_of_n_l1922_192262

variable (n f : ℝ)

theorem ten_percent_of_n (h : n - (1 / 4 * 2) - (1 / 3 * 3) - f * n = 27) : 
  0.10 * n = 0.10 * (28.5 / (1 - f)) :=
by
  simp only [*, mul_one_div_cancel, mul_sub, sub_eq_add_neg, add_div, div_self, one_div, mul_add]
  sorry

end NUMINAMATH_GPT_ten_percent_of_n_l1922_192262


namespace NUMINAMATH_GPT_rationalization_sum_l1922_192248

noncomputable def rationalize_denominator : ℚ :=
  let A := 5
  let B := 49
  let C := 21
  A + B + C

theorem rationalization_sum : rationalize_denominator = 75 := by
  sorry

end NUMINAMATH_GPT_rationalization_sum_l1922_192248


namespace NUMINAMATH_GPT_fraction_addition_l1922_192247

variable {w x y : ℝ}

theorem fraction_addition (h1 : w / x = 1 / 3) (h2 : w / y = 3 / 4) : (x + y) / y = 13 / 4 := by
  sorry

end NUMINAMATH_GPT_fraction_addition_l1922_192247


namespace NUMINAMATH_GPT_sequence_product_l1922_192264

-- Definitions for the arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a1 d : ℝ), ∀ n, a n = a1 + (n - 1) * d

-- Definitions for the geometric sequence
def is_geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ (b1 r : ℝ), ∀ n, b n = b1 * r ^ (n - 1)

-- Defining the main proposition
theorem sequence_product (a b : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a)
  (h_geom  : is_geometric_sequence b)
  (h_eq    : b 7 = a 7)
  (h_cond  : 2 * a 2 - (a 7) ^ 2 + 2 * a 12 = 0) :
  b 3 * b 11 = 16 :=
sorry

end NUMINAMATH_GPT_sequence_product_l1922_192264


namespace NUMINAMATH_GPT_total_surface_area_of_cylinder_l1922_192255

theorem total_surface_area_of_cylinder 
  (r h : ℝ) 
  (hr : r = 3) 
  (hh : h = 8) : 
  2 * Real.pi * r * h + 2 * Real.pi * r^2 = 66 * Real.pi := by
  sorry

end NUMINAMATH_GPT_total_surface_area_of_cylinder_l1922_192255


namespace NUMINAMATH_GPT_download_time_l1922_192295

theorem download_time (avg_speed : ℤ) (size_A size_B size_C : ℤ) (gb_to_mb : ℤ) (secs_in_min : ℤ) :
  avg_speed = 30 →
  size_A = 450 →
  size_B = 240 →
  size_C = 120 →
  gb_to_mb = 1000 →
  secs_in_min = 60 →
  ( (size_A * gb_to_mb + size_B * gb_to_mb + size_C * gb_to_mb ) / avg_speed ) / secs_in_min = 450 := by
  intros h_avg h_A h_B h_C h_gb h_secs
  sorry

end NUMINAMATH_GPT_download_time_l1922_192295


namespace NUMINAMATH_GPT_time_for_B_alone_to_paint_l1922_192227

noncomputable def rate_A := 1 / 4
noncomputable def rate_BC := 1 / 3
noncomputable def rate_AC := 1 / 2
noncomputable def rate_DB := 1 / 6

theorem time_for_B_alone_to_paint :
  (1 / (rate_BC - (rate_AC - rate_A))) = 12 := by
  sorry

end NUMINAMATH_GPT_time_for_B_alone_to_paint_l1922_192227


namespace NUMINAMATH_GPT_largest_digit_divisible_by_6_l1922_192237

theorem largest_digit_divisible_by_6 : ∃ N : ℕ, N ≤ 9 ∧ (∃ d : ℕ, 3456 * 10 + N = 6 * d) ∧ (∀ M : ℕ, M ≤ 9 ∧ (∃ d : ℕ, 3456 * 10 + M = 6 * d) → M ≤ N) :=
sorry

end NUMINAMATH_GPT_largest_digit_divisible_by_6_l1922_192237


namespace NUMINAMATH_GPT_extra_pieces_of_gum_l1922_192284

theorem extra_pieces_of_gum (total_packages : ℕ) (pieces_per_package : ℕ) (total_pieces : ℕ) : ℕ :=
  if total_packages = 43 ∧ pieces_per_package = 23 ∧ total_pieces = 997 then
    997 - (43 * 23)
  else
    0  -- This is a dummy value for other cases, as they do not satisfy our conditions.

#print extra_pieces_of_gum

end NUMINAMATH_GPT_extra_pieces_of_gum_l1922_192284


namespace NUMINAMATH_GPT_sales_tax_difference_l1922_192230

theorem sales_tax_difference :
  let price : ℝ := 50
  let tax_rate1 : ℝ := 0.075
  let tax_rate2 : ℝ := 0.0625
  let tax1 := price * tax_rate1
  let tax2 := price * tax_rate2
  let difference := tax1 - tax2
  difference = 0.625 :=
by
  sorry

end NUMINAMATH_GPT_sales_tax_difference_l1922_192230


namespace NUMINAMATH_GPT_speed_downstream_is_correct_l1922_192209

-- Definitions corresponding to the conditions
def speed_boat_still_water : ℕ := 60
def speed_current : ℕ := 17

-- Definition of speed downstream from the conditions and proving the result
theorem speed_downstream_is_correct :
  speed_boat_still_water + speed_current = 77 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_speed_downstream_is_correct_l1922_192209


namespace NUMINAMATH_GPT_composite_exists_for_x_64_l1922_192278

-- Define the conditions
def is_composite (n : ℕ) : Prop := ∃ m k : ℕ, m > 1 ∧ k > 1 ∧ n = m * k

-- Main statement
theorem composite_exists_for_x_64 :
  ∃ n : ℕ, is_composite (n^4 + 64) :=
sorry

end NUMINAMATH_GPT_composite_exists_for_x_64_l1922_192278


namespace NUMINAMATH_GPT_product_of_two_numbers_l1922_192275

theorem product_of_two_numbers (x y : ℝ) (h1 : x - y = 12) (h2 : x^2 + y^2 = 340) : x * y = 97.9450625 :=
by
  sorry

end NUMINAMATH_GPT_product_of_two_numbers_l1922_192275


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l1922_192212

theorem hyperbola_eccentricity (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) 
  (h₂ : 4 * c^2 = 25) (h₃ : a = 1/2) : c/a = 5 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l1922_192212


namespace NUMINAMATH_GPT_train_length_proof_l1922_192224

def speed_kmph : ℝ := 54
def time_seconds : ℝ := 54.995600351971845
def bridge_length_m : ℝ := 660
def train_length_approx : ℝ := 164.93

noncomputable def speed_m_s : ℝ := speed_kmph * (1000 / 3600)
noncomputable def total_distance : ℝ := speed_m_s * time_seconds
noncomputable def train_length : ℝ := total_distance - bridge_length_m

theorem train_length_proof :
  abs (train_length - train_length_approx) < 0.01 :=
by
  sorry

end NUMINAMATH_GPT_train_length_proof_l1922_192224


namespace NUMINAMATH_GPT_rectangular_field_length_l1922_192238

theorem rectangular_field_length {w l : ℝ} (h1 : l = 2 * w) (h2 : (8 : ℝ) * 8 = 1 / 18 * (l * w)) : l = 48 :=
by sorry

end NUMINAMATH_GPT_rectangular_field_length_l1922_192238


namespace NUMINAMATH_GPT_range_of_a_l1922_192296

noncomputable def f (a x : ℝ) : ℝ := Real.exp x - a * x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f a x ≥ 0) ↔ (0 ≤ a ∧ a ≤ Real.exp 1) := by
  sorry

end NUMINAMATH_GPT_range_of_a_l1922_192296


namespace NUMINAMATH_GPT_gary_money_after_sale_l1922_192287

theorem gary_money_after_sale :
  let initial_money := 73.0
  let sale_amount := 55.0
  initial_money + sale_amount = 128.0 :=
by
  let initial_money := 73.0
  let sale_amount := 55.0
  show initial_money + sale_amount = 128.0
  sorry

end NUMINAMATH_GPT_gary_money_after_sale_l1922_192287


namespace NUMINAMATH_GPT_value_of_expression_l1922_192253

theorem value_of_expression (x y : ℚ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (x / |x| + |y| / y = 2) ∨ (x / |x| + |y| / y = 0) ∨ (x / |x| + |y| / y = -2) :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l1922_192253


namespace NUMINAMATH_GPT_gate_perimeter_l1922_192205

theorem gate_perimeter (r : ℝ) (theta : ℝ) (h1 : r = 2) (h2 : theta = π / 2) :
  let arc_length := (3 / 4) * (2 * π * r)
  let radii_length := 2 * r
  arc_length + radii_length = 3 * π + 4 :=
by
  simp [h1, h2]
  sorry

end NUMINAMATH_GPT_gate_perimeter_l1922_192205


namespace NUMINAMATH_GPT_determine_b_l1922_192281

noncomputable def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem determine_b (a b c m1 m2 : ℝ) (h1 : a > b) (h2 : b > c) (h3 : f a b c 1 = 0)
  (h4 : a^2 + (f a b c m1 + f a b c m2) * a + (f a b c m1) * (f a b c m2) = 0) : 
  b ≥ 0 := 
by
  -- Proof logic goes here
  sorry

end NUMINAMATH_GPT_determine_b_l1922_192281


namespace NUMINAMATH_GPT_fraction_equality_l1922_192297

theorem fraction_equality (a b : ℚ) (h₁ : a = 1/2) (h₂ : b = 2/3) : 
    (6 * a + 18 * b) / (12 * a + 6 * b) = 3 / 2 := by
  sorry

end NUMINAMATH_GPT_fraction_equality_l1922_192297


namespace NUMINAMATH_GPT_algebra_problem_l1922_192202

noncomputable def expression (a b : ℝ) : ℝ :=
  (3 * a + b / 3)⁻¹ * ((3 * a)⁻¹ + (b / 3)⁻¹)

theorem algebra_problem (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  expression a b = (a * b)⁻¹ :=
by
  sorry

end NUMINAMATH_GPT_algebra_problem_l1922_192202


namespace NUMINAMATH_GPT_area_of_triangle_l1922_192277

theorem area_of_triangle (a b c : ℝ) (h₁ : a + b = 14) (h₂ : c = 10) (h₃ : a^2 + b^2 = c^2) :
  (1 / 2) * a * b = 24 :=
  sorry

end NUMINAMATH_GPT_area_of_triangle_l1922_192277


namespace NUMINAMATH_GPT_intersection_of_lines_l1922_192298

theorem intersection_of_lines : ∃ (x y : ℝ), 9 * x - 4 * y = 6 ∧ 7 * x + y = 17 ∧ (x, y) = (2, 3) := 
by
  sorry

end NUMINAMATH_GPT_intersection_of_lines_l1922_192298


namespace NUMINAMATH_GPT_percent_calculation_l1922_192271

theorem percent_calculation (x : ℝ) (h : 0.30 * 0.40 * x = 24) : 0.20 * 0.60 * x = 24 := 
by
  sorry

end NUMINAMATH_GPT_percent_calculation_l1922_192271


namespace NUMINAMATH_GPT_prove_union_l1922_192220

variable (M N : Set ℕ)
variable (x : ℕ)

def M_definition := (0 ∈ M) ∧ (x ∈ M) ∧ (M = {0, x})
def N_definition := (N = {1, 2})
def intersection_condition := (M ∩ N = {2})
def union_result := (M ∪ N = {0, 1, 2})

theorem prove_union (M : Set ℕ) (N : Set ℕ) (x : ℕ) :
  M_definition M x → N_definition N → intersection_condition M N → union_result M N :=
by
  sorry

end NUMINAMATH_GPT_prove_union_l1922_192220


namespace NUMINAMATH_GPT_country_albums_count_l1922_192299

-- Definitions based on conditions
def pop_albums : Nat := 8
def songs_per_album : Nat := 7
def total_songs : Nat := 70

-- Theorem to prove the number of country albums
theorem country_albums_count : (total_songs - pop_albums * songs_per_album) / songs_per_album = 2 := by
  sorry

end NUMINAMATH_GPT_country_albums_count_l1922_192299


namespace NUMINAMATH_GPT_spend_money_l1922_192217

theorem spend_money (n : ℕ) (h : n > 7) : ∃ a b : ℕ, 3 * a + 5 * b = n :=
by
  sorry

end NUMINAMATH_GPT_spend_money_l1922_192217


namespace NUMINAMATH_GPT_union_of_M_N_l1922_192257

-- Definitions of sets M and N
def M : Set ℕ := {0, 1}
def N : Set ℕ := {1, 2}

-- The theorem to prove
theorem union_of_M_N : M ∪ N = {0, 1, 2} :=
  by sorry

end NUMINAMATH_GPT_union_of_M_N_l1922_192257


namespace NUMINAMATH_GPT_area_between_hexagon_and_square_l1922_192260

noncomputable def circleRadius : ℝ := 6

noncomputable def centralAngleSquare : ℝ := Real.pi / 2

noncomputable def centralAngleHexagon : ℝ := Real.pi / 3

noncomputable def areaSegment (r α : ℝ) : ℝ :=
  0.5 * r^2 * (α - Real.sin α)

noncomputable def areaBetweenArcs : ℝ :=
  let r := circleRadius
  let T_AB := areaSegment r centralAngleSquare
  let T_CD := areaSegment r centralAngleHexagon
  2 * (T_AB - T_CD)

theorem area_between_hexagon_and_square :
  abs (areaBetweenArcs - 14.03) < 0.01 :=
by
  sorry

end NUMINAMATH_GPT_area_between_hexagon_and_square_l1922_192260


namespace NUMINAMATH_GPT_arithmetic_sqrt_of_49_l1922_192223

theorem arithmetic_sqrt_of_49 : ∃ x : ℕ, x^2 = 49 ∧ x = 7 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sqrt_of_49_l1922_192223


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l1922_192293

theorem sufficient_but_not_necessary (x y : ℝ) :
  (x ≥ 2 ∧ y ≥ 2 → x^2 + y^2 ≥ 4) ∧ ∃ x y : ℝ, x^2 + y^2 ≥ 4 ∧ ¬(x ≥ 2 ∧ y ≥ 2) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l1922_192293


namespace NUMINAMATH_GPT_solution_to_equation_l1922_192211

theorem solution_to_equation : 
    (∃ x : ℤ, (x = 2 ∨ x = -2 ∨ x = 1 ∨ x = -1) ∧ (2 * x - 3 = -1)) → x = 1 :=
by
  sorry

end NUMINAMATH_GPT_solution_to_equation_l1922_192211
