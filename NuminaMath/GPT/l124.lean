import Mathlib

namespace NUMINAMATH_GPT_min_value_of_f_l124_12419

open Real

noncomputable def f (x : ℝ) : ℝ := (x^2 + 9) / sqrt (x^2 + 5)

theorem min_value_of_f : ∀ x : ℝ, f x ≥ 4 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_f_l124_12419


namespace NUMINAMATH_GPT_percentage_male_red_ants_proof_l124_12452

noncomputable def percentage_red_ants : ℝ := 0.85
noncomputable def percentage_female_red_ants : ℝ := 0.45
noncomputable def percentage_male_red_ants : ℝ := percentage_red_ants * (1 - percentage_female_red_ants)

theorem percentage_male_red_ants_proof : percentage_male_red_ants = 0.4675 :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_percentage_male_red_ants_proof_l124_12452


namespace NUMINAMATH_GPT_problem_statement_l124_12476

noncomputable def a : ℝ := 31 / 32
noncomputable def b : ℝ := Real.cos (1 / 4)
noncomputable def c : ℝ := 4 * Real.sin (1 / 4)

theorem problem_statement : c > b ∧ b > a := 
by {
  sorry
}

end NUMINAMATH_GPT_problem_statement_l124_12476


namespace NUMINAMATH_GPT_quadratic_inequality_solution_set_quadratic_inequality_solution_set2_l124_12456

-- Proof Problem 1 Statement
theorem quadratic_inequality_solution_set (a b : ℝ) (h : ∀ x : ℝ, b < x ∧ x < 1 → ax^2 + 3 * x + 2 > 0) : 
  a = -5 ∧ b = -2/5 := sorry

-- Proof Problem 2 Statement
theorem quadratic_inequality_solution_set2 (a : ℝ) (h_pos : a > 0) : 
  ((0 < a ∧ a < 3) → (∀ x : ℝ, x < -3 / a ∨ x > -1 → ax^2 + 3 * x + 2 > -ax - 1)) ∧
  (a = 3 → (∀ x : ℝ, x ≠ -1 → ax^2 + 3 * x + 2 > -ax - 1)) ∧
  (a > 3 → (∀ x : ℝ, x < -1 ∨ x > -3 / a → ax^2 + 3 * x + 2 > -ax - 1)) := sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_set_quadratic_inequality_solution_set2_l124_12456


namespace NUMINAMATH_GPT_regular_price_correct_l124_12417

noncomputable def regular_price_of_one_tire (x : ℝ) : Prop :=
  3 * x + 5 - 10 = 302

theorem regular_price_correct (x : ℝ) : regular_price_of_one_tire x → x = 307 / 3 := by
  intro h
  sorry

end NUMINAMATH_GPT_regular_price_correct_l124_12417


namespace NUMINAMATH_GPT_MarcoScoresAreCorrect_l124_12466

noncomputable def MarcoTestScores : List ℕ := [94, 82, 76, 75, 64]

theorem MarcoScoresAreCorrect : 
  ∀ (scores : List ℕ),
    scores = [82, 76, 75] ∧ 
    (∃ t4 t5, t4 < 95 ∧ t5 < 95 ∧ 82 ≠ t4 ∧ 82 ≠ t5 ∧ 76 ≠ t4 ∧ 76 ≠ t5 ∧ 75 ≠ t4 ∧ 75 ≠ t5 ∧ 
       t4 ≠ t5 ∧
       (82 + 76 + 75 + t4 + t5 = 5 * 85) ∧ 
       (82 + 76 = t4 + t5)) → 
    (scores = [94, 82, 76, 75, 64]) := 
by 
  sorry

end NUMINAMATH_GPT_MarcoScoresAreCorrect_l124_12466


namespace NUMINAMATH_GPT_prob_same_color_l124_12455

-- Define the given conditions
def total_pieces : ℕ := 15
def black_pieces : ℕ := 6
def white_pieces : ℕ := 9
def prob_two_black : ℚ := 1/7
def prob_two_white : ℚ := 12/35

-- Define the statement to be proved
theorem prob_same_color : prob_two_black + prob_two_white = 17 / 35 := by
  sorry

end NUMINAMATH_GPT_prob_same_color_l124_12455


namespace NUMINAMATH_GPT_worst_player_is_son_or_sister_l124_12467

axiom Family : Type
axiom Woman : Family
axiom Brother : Family
axiom Son : Family
axiom Daughter : Family
axiom Sister : Family

axiom are_chess_players : ∀ f : Family, Prop
axiom is_twin : Family → Family → Prop
axiom is_best_player : Family → Prop
axiom is_worst_player : Family → Prop
axiom same_age : Family → Family → Prop
axiom opposite_sex : Family → Family → Prop
axiom is_sibling : Family → Family → Prop

-- Conditions
axiom all_are_chess_players : ∀ f, are_chess_players f
axiom worst_best_opposite_sex : ∀ w b, is_worst_player w → is_best_player b → opposite_sex w b
axiom worst_best_same_age : ∀ w b, is_worst_player w → is_best_player b → same_age w b
axiom twins_relationship : ∀ t1 t2, is_twin t1 t2 → (is_sibling t1 t2 ∨ (t1 = Woman ∧ t2 = Sister))

-- Goal
theorem worst_player_is_son_or_sister :
  ∃ w, (is_worst_player w ∧ (w = Son ∨ w = Sister)) :=
sorry

end NUMINAMATH_GPT_worst_player_is_son_or_sister_l124_12467


namespace NUMINAMATH_GPT_minimum_value_f_l124_12477

noncomputable def f (x : ℝ) : ℝ := max (3 - x) (x^2 - 4 * x + 3)

theorem minimum_value_f : ∃ m : ℝ, ∀ x : ℝ, f x ≥ m ∧ (∀ ε > 0, ∃ x : ℝ, x ≥ 0 ∧ f x < m + ε) ∧ m = 0 := 
sorry

end NUMINAMATH_GPT_minimum_value_f_l124_12477


namespace NUMINAMATH_GPT_number_of_sunflowers_l124_12459

noncomputable def cost_per_red_rose : ℝ := 1.5
noncomputable def cost_per_sunflower : ℝ := 3
noncomputable def total_cost : ℝ := 45
noncomputable def cost_of_red_roses : ℝ := 24 * cost_per_red_rose
noncomputable def money_left_for_sunflowers : ℝ := total_cost - cost_of_red_roses

theorem number_of_sunflowers :
  (money_left_for_sunflowers / cost_per_sunflower) = 3 :=
by
  sorry

end NUMINAMATH_GPT_number_of_sunflowers_l124_12459


namespace NUMINAMATH_GPT_men_entered_l124_12481

theorem men_entered (M W x : ℕ) (h1 : 4 * W = 5 * M)
                    (h2 : M + x = 14)
                    (h3 : 2 * (W - 3) = 24) :
                    x = 2 :=
by
  sorry

end NUMINAMATH_GPT_men_entered_l124_12481


namespace NUMINAMATH_GPT_part1_l124_12431

noncomputable def f (x : ℝ) : ℝ := x^2 - 1
noncomputable def g (x a : ℝ) : ℝ := a * |x - 1|

theorem part1 (a : ℝ) : (∀ x : ℝ, f x ≥ g x a) ↔ a ≤ -2 := by
  sorry

end NUMINAMATH_GPT_part1_l124_12431


namespace NUMINAMATH_GPT_cakes_served_yesterday_l124_12416

theorem cakes_served_yesterday:
  ∃ y : ℕ, (5 + 6 + y = 14) ∧ y = 3 := 
by
  sorry

end NUMINAMATH_GPT_cakes_served_yesterday_l124_12416


namespace NUMINAMATH_GPT_range_of_m_l124_12411

theorem range_of_m (m : ℝ) : (∀ x : ℝ, m^2 * x^2 + 2 * m * x - 4 < 2 * x^2 + 4 * x) ↔ -2 < m ∧ m ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l124_12411


namespace NUMINAMATH_GPT_integer_solution_count_l124_12490

theorem integer_solution_count :
  (∃ x : ℤ, -4 * x ≥ x + 9 ∧ -3 * x ≤ 15 ∧ -5 * x ≥ 3 * x + 24) ↔
  (∃ n : ℕ, n = 3) :=
by
  sorry

end NUMINAMATH_GPT_integer_solution_count_l124_12490


namespace NUMINAMATH_GPT_max_four_by_one_in_six_by_six_grid_l124_12448

-- Define the grid and rectangle dimensions
def grid_width : ℕ := 6
def grid_height : ℕ := 6
def rect_width : ℕ := 4
def rect_height : ℕ := 1

-- Define the maximum number of rectangles that can be placed
def max_rectangles (grid_w grid_h rect_w rect_h : ℕ) (non_overlapping : Bool) (within_boundaries : Bool) : ℕ :=
  if grid_w = 6 ∧ grid_h = 6 ∧ rect_w = 4 ∧ rect_h = 1 ∧ non_overlapping ∧ within_boundaries then
    8
  else
    0

-- The theorem stating the maximum number of 4x1 rectangles in a 6x6 grid
theorem max_four_by_one_in_six_by_six_grid
  : max_rectangles grid_width grid_height rect_width rect_height true true = 8 := 
sorry

end NUMINAMATH_GPT_max_four_by_one_in_six_by_six_grid_l124_12448


namespace NUMINAMATH_GPT_can_spend_all_money_l124_12449

theorem can_spend_all_money (n : Nat) (h : n > 7) : 
  ∃ (x y : Nat), 3 * x + 5 * y = n :=
by
  sorry

end NUMINAMATH_GPT_can_spend_all_money_l124_12449


namespace NUMINAMATH_GPT_find_m_l124_12436

def g (n : ℤ) : ℤ :=
if n % 2 = 1 then n + 5 else n / 2

theorem find_m (m : ℤ) (h1 : m % 2 = 1) (h2 : g (g (g m)) = 35) : m = 135 :=
sorry

end NUMINAMATH_GPT_find_m_l124_12436


namespace NUMINAMATH_GPT_joshua_total_bottle_caps_l124_12495

def initial_bottle_caps : ℕ := 40
def bought_bottle_caps : ℕ := 7

theorem joshua_total_bottle_caps : initial_bottle_caps + bought_bottle_caps = 47 := 
by
  sorry

end NUMINAMATH_GPT_joshua_total_bottle_caps_l124_12495


namespace NUMINAMATH_GPT_expectation_of_two_fair_dice_l124_12443

noncomputable def E_X : ℝ :=
  (2 * (1/36) + 3 * (2/36) + 4 * (3/36) + 5 * (4/36) + 6 * (5/36) + 7 * (6/36) + 
   8 * (5/36) + 9 * (4/36) + 10 * (3/36) + 11 * (2/36) + 12 * (1/36))

theorem expectation_of_two_fair_dice : E_X = 7 := by
  sorry

end NUMINAMATH_GPT_expectation_of_two_fair_dice_l124_12443


namespace NUMINAMATH_GPT_a_squared_gt_b_squared_l124_12421

theorem a_squared_gt_b_squared {a b : ℝ} (h : a ≠ 0) (hb : b ≠ 0) (hb_domain : b > -1 ∧ b < 1) (h_eq : a = Real.log (1 + b) - Real.log (1 - b)) :
  a^2 > b^2 := 
sorry

end NUMINAMATH_GPT_a_squared_gt_b_squared_l124_12421


namespace NUMINAMATH_GPT_tan_105_eq_neg2_sub_sqrt3_l124_12482

theorem tan_105_eq_neg2_sub_sqrt3 :
  Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_tan_105_eq_neg2_sub_sqrt3_l124_12482


namespace NUMINAMATH_GPT_max_abs_c_l124_12437

theorem max_abs_c (a b c d e : ℝ) (h : ∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 → -1 ≤ a * x^4 + b * x^3 + c * x^2 + d * x + e ∧ a * x^4 + b * x^3 + c * x^2 + d * x + e ≤ 1) : abs c ≤ 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_max_abs_c_l124_12437


namespace NUMINAMATH_GPT_inequalities_hold_l124_12406

theorem inequalities_hold (b : ℝ) :
  (b ∈ Set.Ioo (-(1 : ℝ) - Real.sqrt 2 / 4) (0 : ℝ) ∨ b < -(1 : ℝ) - Real.sqrt 2 / 4) →
  (∀ x y : ℝ, 2 * b * Real.cos (2 * (x - y)) + 8 * b^2 * Real.cos (x - y) + 8 * b^2 * (b + 1) + 5 * b < 0) ∧ 
  (∀ x y : ℝ, x^2 + y^2 + 1 > 2 * b * x + 2 * y + b - b^2) :=
by 
  intro h
  sorry

end NUMINAMATH_GPT_inequalities_hold_l124_12406


namespace NUMINAMATH_GPT_sum_of_variables_l124_12415

variables (a b c d : ℝ)

theorem sum_of_variables :
  (a - 2)^2 + (b - 5)^2 + (c - 6)^2 + (d - 3)^2 = 0 → a + b + c + d = 16 :=
by
  intro h
  -- your proof goes here
  sorry

end NUMINAMATH_GPT_sum_of_variables_l124_12415


namespace NUMINAMATH_GPT_correct_word_to_complete_sentence_l124_12471

theorem correct_word_to_complete_sentence
  (parents_spoke_language : Bool)
  (learning_difficulty : String) :
  learning_difficulty = "It was hard for him to learn English in a family, in which neither of the parents spoke the language." :=
by
  sorry

end NUMINAMATH_GPT_correct_word_to_complete_sentence_l124_12471


namespace NUMINAMATH_GPT_simplify_expression_l124_12474

variable (a : ℝ)

theorem simplify_expression : 5 * a + 2 * a + 3 * a - 2 * a = 8 * a :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l124_12474


namespace NUMINAMATH_GPT_base_log_eq_l124_12429

theorem base_log_eq (x : ℝ) : (5 : ℝ)^(x + 7) = (6 : ℝ)^x → x = Real.logb (6 / 5 : ℝ) (5^7 : ℝ) := by
  sorry

end NUMINAMATH_GPT_base_log_eq_l124_12429


namespace NUMINAMATH_GPT_monomial_properties_l124_12460

def coefficient (m : String) : ℤ := 
  if m = "-2xy^3" then -2 
  else sorry

def degree (m : String) : ℕ := 
  if m = "-2xy^3" then 4 
  else sorry

theorem monomial_properties : coefficient "-2xy^3" = -2 ∧ degree "-2xy^3" = 4 := 
by 
  exact ⟨rfl, rfl⟩

end NUMINAMATH_GPT_monomial_properties_l124_12460


namespace NUMINAMATH_GPT_find_x_for_f_eq_f_inv_l124_12494

def f (x : ℝ) : ℝ := 3 * x - 8

noncomputable def f_inv (x : ℝ) : ℝ := (x + 8) / 3

theorem find_x_for_f_eq_f_inv : ∃ x : ℝ, f x = f_inv x ∧ x = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_x_for_f_eq_f_inv_l124_12494


namespace NUMINAMATH_GPT_fraction_of_area_below_line_l124_12446

noncomputable def rectangle_area_fraction (x1 y1 x2 y2 : ℝ) (x3 y3 x4 y4 : ℝ) : ℝ :=
  let m := (y2 - y1) / (x2 - x1)
  let b := y1 - m * x1
  let y_intercept := b
  let base := x4 - x1
  let height := y4 - y3
  let triangle_area := 0.5 * base * height
  triangle_area / (base * height)

theorem fraction_of_area_below_line : 
  rectangle_area_fraction 1 3 5 1 1 0 5 4 = 1 / 8 := 
by
  sorry

end NUMINAMATH_GPT_fraction_of_area_below_line_l124_12446


namespace NUMINAMATH_GPT_coinsSold_l124_12426

-- Given conditions
def initialCoins : Nat := 250
def additionalCoins : Nat := 75
def coinsToKeep : Nat := 135

-- Theorem to prove
theorem coinsSold : (initialCoins + additionalCoins - coinsToKeep) = 190 := 
by
  -- Proof omitted 
  sorry

end NUMINAMATH_GPT_coinsSold_l124_12426


namespace NUMINAMATH_GPT_problem_solution_l124_12445

-- Definitions for given conditions
variables {a_n b_n : ℕ → ℝ} -- Sequences {a_n} and {b_n}
variables {S T : ℕ → ℝ} -- Sums of the first n terms of {a_n} and {b_n}
variables (h1 : ∀ n, S n = (n * (a_n 1 + a_n n)) / 2)
variables (h2 : ∀ n, T n = (n * (b_n 1 + b_n n)) / 2)
variables (h3 : ∀ n, n > 0 → S n / T n = (2 * n + 1) / (n + 2))

-- The goal
theorem problem_solution :
  (a_n 7) / (b_n 7) = 9 / 5 :=
sorry

end NUMINAMATH_GPT_problem_solution_l124_12445


namespace NUMINAMATH_GPT_point_on_circle_l124_12424

noncomputable def distance_from_origin (x : ℝ) (y : ℝ) : ℝ :=
  Real.sqrt (x^2 + y^2)

theorem point_on_circle : distance_from_origin (-3) 4 = 5 := by
  sorry

end NUMINAMATH_GPT_point_on_circle_l124_12424


namespace NUMINAMATH_GPT_total_selection_ways_l124_12401

-- Defining the conditions
def groupA_male_students : ℕ := 5
def groupA_female_students : ℕ := 3
def groupB_male_students : ℕ := 6
def groupB_female_students : ℕ := 2

-- Define combinations (choose function)
noncomputable def C (n k : ℕ) : ℕ := Nat.choose n k

-- The required theorem statement
theorem total_selection_ways :
  C groupA_female_students 1 * C groupA_male_students 1 * C groupB_male_students 2 +
  C groupB_female_students 1 * C groupB_male_students 1 * C groupA_male_students 2 = 345 :=
by
  sorry

end NUMINAMATH_GPT_total_selection_ways_l124_12401


namespace NUMINAMATH_GPT_number_of_students_l124_12423

theorem number_of_students (avg_age_students : ℕ) (teacher_age : ℕ) (new_avg_age : ℕ) (n : ℕ) (T : ℕ) 
    (h1 : avg_age_students = 10) (h2 : teacher_age = 26) (h3 : new_avg_age = 11)
    (h4 : T = n * avg_age_students) 
    (h5 : (T + teacher_age) / (n + 1) = new_avg_age) : n = 15 :=
by
  -- Proof should go here
  sorry

end NUMINAMATH_GPT_number_of_students_l124_12423


namespace NUMINAMATH_GPT_max_buses_in_city_l124_12407

theorem max_buses_in_city (n : ℕ) (stops : ℕ) (shared : ℕ) (condition1 : n = 9) (condition2 : stops = 3) (condition3 : shared ≤ 1) : n = 12 :=
sorry

end NUMINAMATH_GPT_max_buses_in_city_l124_12407


namespace NUMINAMATH_GPT_coin_loading_impossible_l124_12418

theorem coin_loading_impossible (p q : ℝ) (h1 : p ≠ 1 - p) (h2 : q ≠ 1 - q) 
    (h3 : p * q = 1/4) (h4 : p * (1 - q) = 1/4) (h5 : (1 - p) * q = 1/4) (h6 : (1 - p) * (1 - q) = 1/4) : 
    false := 
by 
  sorry

end NUMINAMATH_GPT_coin_loading_impossible_l124_12418


namespace NUMINAMATH_GPT_fraction_left_handed_l124_12438

-- Definitions based on given conditions
def red_ratio : ℝ := 10
def blue_ratio : ℝ := 5
def green_ratio : ℝ := 3
def yellow_ratio : ℝ := 2

def red_left_handed_percent : ℝ := 0.37
def blue_left_handed_percent : ℝ := 0.61
def green_left_handed_percent : ℝ := 0.26
def yellow_left_handed_percent : ℝ := 0.48

-- Statement we want to prove
theorem fraction_left_handed : 
  (red_left_handed_percent * red_ratio + blue_left_handed_percent * blue_ratio +
  green_left_handed_percent * green_ratio + yellow_left_handed_percent * yellow_ratio) /
  (red_ratio + blue_ratio + green_ratio + yellow_ratio) = 8.49 / 20 :=
  sorry

end NUMINAMATH_GPT_fraction_left_handed_l124_12438


namespace NUMINAMATH_GPT_cube_painted_four_faces_l124_12440

theorem cube_painted_four_faces (n : ℕ) (hn : n ≠ 0) (h : (4 * n^2) / (6 * n^3) = 1 / 3) : n = 2 :=
by
  have : 4 * n^2 = 4 * n^2 := by rfl
  sorry

end NUMINAMATH_GPT_cube_painted_four_faces_l124_12440


namespace NUMINAMATH_GPT_distance_between_first_and_last_bushes_l124_12479

theorem distance_between_first_and_last_bushes 
  (bushes : Nat)
  (spaces_per_bush : ℕ) 
  (distance_first_to_fifth : ℕ) 
  (total_bushes : bushes = 10)
  (fifth_bush_distance : distance_first_to_fifth = 100)
  : ∃ (d : ℕ), d = 225 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_first_and_last_bushes_l124_12479


namespace NUMINAMATH_GPT_salt_solution_proof_l124_12422

theorem salt_solution_proof (x : ℝ) (P : ℝ) (hx : x = 28.571428571428573) :
  ((P / 100) * 100 + x) = 0.30 * (100 + x) → P = 10 :=
by
  sorry

end NUMINAMATH_GPT_salt_solution_proof_l124_12422


namespace NUMINAMATH_GPT_find_g_value_l124_12414

noncomputable def g (x : ℝ) (a b c : ℝ) : ℝ := a * x^6 + b * x^4 + c * x^2 + 7

theorem find_g_value (a b c : ℝ) (h1 : g (-4) a b c = 13) : g 4 a b c = 13 := by
  sorry

end NUMINAMATH_GPT_find_g_value_l124_12414


namespace NUMINAMATH_GPT_psychologist_diagnosis_l124_12480

theorem psychologist_diagnosis :
  let initial_patients := 26
  let doubling_factor := 2
  let probability := 1 / 4
  let total_patients := initial_patients * doubling_factor
  let expected_patients_with_ZYX := total_patients * probability
  expected_patients_with_ZYX = 13 := by
  sorry

end NUMINAMATH_GPT_psychologist_diagnosis_l124_12480


namespace NUMINAMATH_GPT_rocco_total_usd_l124_12434

noncomputable def total_usd_quarters : ℝ := 40 * 0.25
noncomputable def total_usd_nickels : ℝ := 90 * 0.05

noncomputable def cad_to_usd : ℝ := 0.8
noncomputable def eur_to_usd : ℝ := 1.18
noncomputable def gbp_to_usd : ℝ := 1.4

noncomputable def total_cad_dimes : ℝ := 60 * 0.10 * 0.8
noncomputable def total_eur_cents : ℝ := 50 * 0.01 * 1.18
noncomputable def total_gbp_pence : ℝ := 30 * 0.01 * 1.4

noncomputable def total_usd : ℝ :=
  total_usd_quarters + total_usd_nickels + total_cad_dimes +
  total_eur_cents + total_gbp_pence

theorem rocco_total_usd : total_usd = 20.31 := sorry

end NUMINAMATH_GPT_rocco_total_usd_l124_12434


namespace NUMINAMATH_GPT_jack_evening_emails_l124_12485

theorem jack_evening_emails (ema_morning ema_afternoon ema_afternoon_evening ema_evening : ℕ)
  (h1 : ema_morning = 4)
  (h2 : ema_afternoon = 5)
  (h3 : ema_afternoon_evening = 13)
  (h4 : ema_afternoon_evening = ema_afternoon + ema_evening) :
  ema_evening = 8 :=
by
  sorry

end NUMINAMATH_GPT_jack_evening_emails_l124_12485


namespace NUMINAMATH_GPT_part1_solution_part2_solution_l124_12468

variable {x a : ℝ}

def f (x a : ℝ) : ℝ := abs (x - a)

theorem part1_solution (h1 : 0 ≤ x) (h2 : x ≤ 4) (h3 : f x a ≤ 2) : a = 2 :=
  sorry

theorem part2_solution (ha : 0 ≤ a) (hb : a ≤ 3) : (f (x + a) a + f (x - a) a ≥ f (a * x) a - a * f x a) :=
  sorry

end NUMINAMATH_GPT_part1_solution_part2_solution_l124_12468


namespace NUMINAMATH_GPT_equal_probabilities_partitioned_nonpartitioned_conditions_for_equal_probabilities_l124_12470

variable (v1 v2 f1 f2 : ℝ)

theorem equal_probabilities_partitioned_nonpartitioned :
  (v1 * (v2 + f2) + v2 * (v1 + f1)) / (2 * (v1 + f1) * (v2 + f2)) =
  (v1 + v2) / ((v1 + f1) + (v2 + f2)) :=
by sorry

theorem conditions_for_equal_probabilities :
  (v1 * f2 = v2 * f1) ∨ (v1 + f1 = v2 + f2) :=
by sorry

end NUMINAMATH_GPT_equal_probabilities_partitioned_nonpartitioned_conditions_for_equal_probabilities_l124_12470


namespace NUMINAMATH_GPT_evaluate_combinations_l124_12442

theorem evaluate_combinations (n : ℕ) (h1 : 0 ≤ 5 - n) (h2 : 5 - n ≤ n) (h3 : 0 ≤ 10 - n) (h4 : 10 - n ≤ n + 1) (h5 : n > 0) :
  Nat.choose n (5 - n) + Nat.choose (n + 1) (10 - n) = 7 :=
sorry

end NUMINAMATH_GPT_evaluate_combinations_l124_12442


namespace NUMINAMATH_GPT_smallest_positive_int_l124_12428

open Nat

theorem smallest_positive_int (x : ℕ) :
  (x % 6 = 3) ∧ (x % 8 = 5) ∧ (x % 9 = 2) → x = 237 := by
  sorry

end NUMINAMATH_GPT_smallest_positive_int_l124_12428


namespace NUMINAMATH_GPT_ratio_of_ages_l124_12496

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

end NUMINAMATH_GPT_ratio_of_ages_l124_12496


namespace NUMINAMATH_GPT_dartboard_points_proof_l124_12454

variable (points_one points_two points_three points_four : ℕ)

theorem dartboard_points_proof
  (h1 : points_one = 30)
  (h2 : points_two = 38)
  (h3 : points_three = 41)
  (h4 : 2 * points_four = points_one + points_two) :
  points_four = 34 :=
by {
  sorry
}

end NUMINAMATH_GPT_dartboard_points_proof_l124_12454


namespace NUMINAMATH_GPT_contrapositive_of_zero_squared_l124_12447

theorem contrapositive_of_zero_squared {x y : ℝ} :
  (x^2 + y^2 = 0 → x = 0 ∧ y = 0) →
  (x ≠ 0 ∨ y ≠ 0 → x^2 + y^2 ≠ 0) :=
by
  intro h1
  intro h2
  sorry

end NUMINAMATH_GPT_contrapositive_of_zero_squared_l124_12447


namespace NUMINAMATH_GPT_solve_quad_1_solve_quad_2_l124_12420

theorem solve_quad_1 :
  ∀ (x : ℝ), x^2 - 5 * x - 6 = 0 ↔ x = 6 ∨ x = -1 := by
  sorry

theorem solve_quad_2 :
  ∀ (x : ℝ), (x + 1) * (x - 1) + x * (x + 2) = 7 + 6 * x ↔ x = 1 + Real.sqrt 5 ∨ x = 1 - Real.sqrt 5 := by
  sorry

end NUMINAMATH_GPT_solve_quad_1_solve_quad_2_l124_12420


namespace NUMINAMATH_GPT_min_value_l124_12412

theorem min_value (x y : ℝ) (h1 : xy > 0) (h2 : x + 4 * y = 3) : 
  ∃ (m : ℝ), m = 3 ∧ ∀ x y, xy > 0 → x + 4 * y = 3 → (1 / x + 1 / y) ≥ 3 := sorry

end NUMINAMATH_GPT_min_value_l124_12412


namespace NUMINAMATH_GPT_probability_both_selected_l124_12435

def P_X : ℚ := 1 / 3
def P_Y : ℚ := 2 / 7

theorem probability_both_selected : P_X * P_Y = 2 / 21 :=
by
  sorry

end NUMINAMATH_GPT_probability_both_selected_l124_12435


namespace NUMINAMATH_GPT_toms_weekly_income_l124_12409

variable (num_buckets : ℕ) (crabs_per_bucket : ℕ) (price_per_crab : ℕ) (days_per_week : ℕ)

def daily_crabs := num_buckets * crabs_per_bucket
def daily_income := daily_crabs * price_per_crab
def weekly_income := daily_income * days_per_week

theorem toms_weekly_income 
  (h1 : num_buckets = 8)
  (h2 : crabs_per_bucket = 12)
  (h3 : price_per_crab = 5)
  (h4 : days_per_week = 7) :
  weekly_income num_buckets crabs_per_bucket price_per_crab days_per_week = 3360 :=
by
  sorry

end NUMINAMATH_GPT_toms_weekly_income_l124_12409


namespace NUMINAMATH_GPT_sample_size_divided_into_six_groups_l124_12475

theorem sample_size_divided_into_six_groups
  (n : ℕ)
  (c1 c2 c3 : ℕ)
  (k : ℚ)
  (h1 : c1 + c2 + c3 = 36)
  (h2 : 20 * k = 1)
  (h3 : 2 * k * n = c1)
  (h4 : 3 * k * n = c2)
  (h5 : 4 * k * n = c3) :
  n = 80 :=
by
  sorry

end NUMINAMATH_GPT_sample_size_divided_into_six_groups_l124_12475


namespace NUMINAMATH_GPT_gcd_gx_x_is_450_l124_12464

def g (x : ℕ) : ℕ := (3 * x + 2) * (8 * x + 3) * (14 * x + 5) * (x + 15)

noncomputable def gcd_gx_x (x : ℕ) (h : 49356 ∣ x) : ℕ :=
  Nat.gcd (g x) x

theorem gcd_gx_x_is_450 (x : ℕ) (h : 49356 ∣ x) : gcd_gx_x x h = 450 := by
  sorry

end NUMINAMATH_GPT_gcd_gx_x_is_450_l124_12464


namespace NUMINAMATH_GPT_thirty_three_and_one_third_percent_of_330_l124_12486

theorem thirty_three_and_one_third_percent_of_330 :
  (33 + 1 / 3) / 100 * 330 = 110 :=
sorry

end NUMINAMATH_GPT_thirty_three_and_one_third_percent_of_330_l124_12486


namespace NUMINAMATH_GPT_rational_solution_l124_12498

theorem rational_solution (m n : ℤ) (h : a = (m^4 + n^4 + m^2 * n^2) / (4 * m^2 * n^2)) : 
  ∃ a : ℚ, a = (m^4 + n^4 + m^2 * n^2) / (4 * m^2 * n^2) :=
by {
  sorry
}

end NUMINAMATH_GPT_rational_solution_l124_12498


namespace NUMINAMATH_GPT_mooncake_packaging_problem_l124_12487

theorem mooncake_packaging_problem :
  ∃ x y : ℕ, 9 * x + 4 * y = 35 ∧ x + y = 5 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_mooncake_packaging_problem_l124_12487


namespace NUMINAMATH_GPT_inequality_problem_l124_12405

open Real

theorem inequality_problem 
  (x y : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (h1 : x + y^2016 ≥ 1) : 
  x^2016 + y > 1 - 1/100 :=
by
  sorry

end NUMINAMATH_GPT_inequality_problem_l124_12405


namespace NUMINAMATH_GPT_average_increase_l124_12484

-- Definitions
def runs_11 := 90
def avg_11 := 40

-- Conditions
def total_runs_before (A : ℕ) := A * 10
def total_runs_after (runs_11 : ℕ) (total_runs_before : ℕ) := total_runs_before + runs_11
def increased_average (avg_11 : ℕ) (avg_before : ℕ) := avg_11 = avg_before + 5

-- Theorem stating the equivalent proof problem
theorem average_increase
  (A : ℕ)
  (H1 : total_runs_after runs_11 (total_runs_before A) = 40 * 11)
  (H2 : avg_11 = 40) :
  increased_average 40 A := 
sorry

end NUMINAMATH_GPT_average_increase_l124_12484


namespace NUMINAMATH_GPT_reverse_geometric_diff_l124_12469

-- A digit must be between 0 and 9
def digit (d : ℕ) : Prop := d ≥ 0 ∧ d ≤ 9

-- Distinct digits
def distinct_digits (a b c : ℕ) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c

-- Reverse geometric sequence 
def reverse_geometric (a b c : ℕ) : Prop := ∃ r : ℚ, b = c * r ∧ a = b * r

-- Check if abc forms a valid 3-digit reverse geometric sequence
def valid_reverse_geometric_number (a b c : ℕ) : Prop :=
  digit a ∧ digit b ∧ digit c ∧ distinct_digits a b c ∧ reverse_geometric a b c

theorem reverse_geometric_diff (a b c d e f : ℕ) 
  (h1: valid_reverse_geometric_number a b c) 
  (h2: valid_reverse_geometric_number d e f) :
  (a * 100 + b * 10 + c) - (d * 100 + e * 10 + f) = 789 :=
sorry

end NUMINAMATH_GPT_reverse_geometric_diff_l124_12469


namespace NUMINAMATH_GPT_domain_of_log_function_l124_12425

theorem domain_of_log_function (x : ℝ) : 1 - x > 0 ↔ x < 1 := by
  sorry

end NUMINAMATH_GPT_domain_of_log_function_l124_12425


namespace NUMINAMATH_GPT_calculate_product_l124_12403

noncomputable def complex_number_r (r : ℂ) : Prop :=
r^6 = 1 ∧ r ≠ 1

theorem calculate_product (r : ℂ) (h₁ : r^6 = 1) (h₂ : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) = 2 := 
sorry

end NUMINAMATH_GPT_calculate_product_l124_12403


namespace NUMINAMATH_GPT_remainder_div_product_l124_12488

theorem remainder_div_product (P D D' D'' Q R Q' R' Q'' R'' : ℕ) 
  (h1 : P = Q * D + R) 
  (h2 : Q = Q' * D' + R') 
  (h3 : Q' = Q'' * D'' + R'') :
  P % (D * D' * D'') = D * D' * R'' + D * R' + R := 
sorry

end NUMINAMATH_GPT_remainder_div_product_l124_12488


namespace NUMINAMATH_GPT_probability_distribution_correct_l124_12489

noncomputable def numCombinations (n k : ℕ) : ℕ :=
  (Nat.choose n k)

theorem probability_distribution_correct :
  let totalCombinations := numCombinations 5 2
  let prob_two_red := (numCombinations 3 2 : ℚ) / totalCombinations
  let prob_two_white := (numCombinations 2 2 : ℚ) / totalCombinations
  let prob_one_red_one_white := ((numCombinations 3 1) * (numCombinations 2 1) : ℚ) / totalCombinations
  (prob_two_red, prob_one_red_one_white, prob_two_white) = (0.3, 0.6, 0.1) :=
by
  sorry

end NUMINAMATH_GPT_probability_distribution_correct_l124_12489


namespace NUMINAMATH_GPT_carp_and_population_l124_12408

-- Define the characteristics of an individual and a population
structure Individual where
  birth : Prop
  death : Prop
  gender : Prop
  age : Prop

structure Population where
  birth_rate : Prop
  death_rate : Prop
  gender_ratio : Prop
  age_composition : Prop

-- Define the conditions as hypotheses
axiom a : Individual
axiom b : Population

-- State the theorem: If "a" has characteristics of an individual and "b" has characteristics
-- of a population, then "a" is a carp and "b" is a carp population
theorem carp_and_population : 
  (a.birth ∧ a.death ∧ a.gender ∧ a.age) ∧
  (b.birth_rate ∧ b.death_rate ∧ b.gender_ratio ∧ b.age_composition) →
  (a = ⟨True, True, True, True⟩ ∧ b = ⟨True, True, True, True⟩) := 
by 
  sorry

end NUMINAMATH_GPT_carp_and_population_l124_12408


namespace NUMINAMATH_GPT_axes_are_not_vectors_l124_12444

def is_vector (v : Type) : Prop :=
  ∃ (magnitude : ℝ) (direction : ℝ), magnitude > 0

def x_axis : Type := ℝ
def y_axis : Type := ℝ

-- The Cartesian x-axis and y-axis are not vectors
theorem axes_are_not_vectors : ¬ (is_vector x_axis) ∧ ¬ (is_vector y_axis) :=
by
  sorry

end NUMINAMATH_GPT_axes_are_not_vectors_l124_12444


namespace NUMINAMATH_GPT_tub_drain_time_l124_12493

theorem tub_drain_time (t : ℝ) (p q : ℝ) (h1 : t = 4) (h2 : p = 5 / 7) (h3 : q = 2 / 7) :
  q * t / p = 1.6 := by
  sorry

end NUMINAMATH_GPT_tub_drain_time_l124_12493


namespace NUMINAMATH_GPT_number_of_biscuits_per_day_l124_12491

theorem number_of_biscuits_per_day 
  (price_cupcake : ℝ) (price_cookie : ℝ) (price_biscuit : ℝ)
  (cupcakes_per_day : ℕ) (cookies_per_day : ℕ) (total_earnings_five_days : ℝ) :
  price_cupcake = 1.5 → 
  price_cookie = 2 → 
  price_biscuit = 1 → 
  cupcakes_per_day = 20 → 
  cookies_per_day = 10 → 
  total_earnings_five_days = 350 →
  (total_earnings_five_days - 
   (5 * (cupcakes_per_day * price_cupcake + cookies_per_day * price_cookie))) / (5 * price_biscuit) = 20 :=
by
  intros price_cupcake_eq price_cookie_eq price_biscuit_eq cupcakes_per_day_eq cookies_per_day_eq total_earnings_five_days_eq
  sorry

end NUMINAMATH_GPT_number_of_biscuits_per_day_l124_12491


namespace NUMINAMATH_GPT_problem_l124_12433

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x - 2
noncomputable def g (x : ℝ) : ℝ := Real.log x + x^2 - 3

theorem problem (a b : ℝ) (H1 : f a = 0) (H2 : g b = 0) : g a < 0 ∧ 0 < f b :=
by
  sorry

end NUMINAMATH_GPT_problem_l124_12433


namespace NUMINAMATH_GPT_find_y_l124_12462

theorem find_y (y : ℕ) : (8000 * 6000 = 480 * 10 ^ y) → y = 5 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_y_l124_12462


namespace NUMINAMATH_GPT_find_c_l124_12400

theorem find_c (a : ℕ) (c : ℕ) 
  (h1 : a = 105) 
  (h2 : a ^ 5 = 3 ^ 3 * 5 ^ 2 * 7 ^ 2 * 11 ^ 2 * 13 * c) : 
  c = 385875 := by 
  sorry

end NUMINAMATH_GPT_find_c_l124_12400


namespace NUMINAMATH_GPT_total_oranges_correct_l124_12463

-- Define the conditions
def oranges_per_child : Nat := 3
def number_of_children : Nat := 4

-- Define the total number of oranges and the statement to be proven
def total_oranges : Nat := oranges_per_child * number_of_children

theorem total_oranges_correct : total_oranges = 12 := by
  sorry

end NUMINAMATH_GPT_total_oranges_correct_l124_12463


namespace NUMINAMATH_GPT_jerry_age_l124_12499

theorem jerry_age (M J : ℕ) (h1 : M = 2 * J - 6) (h2 : M = 16) : J = 11 :=
sorry

end NUMINAMATH_GPT_jerry_age_l124_12499


namespace NUMINAMATH_GPT_paul_spent_252_dollars_l124_12497

noncomputable def total_cost_before_discounts : ℝ :=
  let dress_shirts := 4 * 15
  let pants := 2 * 40
  let suit := 150
  let sweaters := 2 * 30
  dress_shirts + pants + suit + sweaters

noncomputable def store_discount : ℝ := 0.20

noncomputable def coupon_discount : ℝ := 0.10

noncomputable def total_cost_after_store_discount : ℝ :=
  let initial_total := total_cost_before_discounts
  initial_total - store_discount * initial_total

noncomputable def final_total : ℝ :=
  let intermediate_total := total_cost_after_store_discount
  intermediate_total - coupon_discount * intermediate_total

theorem paul_spent_252_dollars :
  final_total = 252 := by
  sorry

end NUMINAMATH_GPT_paul_spent_252_dollars_l124_12497


namespace NUMINAMATH_GPT_problem_l124_12450

theorem problem (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 3) :
  x + (x^2 / y) + (y^2 / x) + y = 95 / 3 := by
  sorry

end NUMINAMATH_GPT_problem_l124_12450


namespace NUMINAMATH_GPT_find_c_l124_12453

-- Definitions from the problem conditions
variables (a c : ℕ)
axiom cond1 : 2 ^ a = 8
axiom cond2 : a = 3 * c

-- The goal is to prove c = 1
theorem find_c : c = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_c_l124_12453


namespace NUMINAMATH_GPT_product_base_8_units_digit_l124_12402

theorem product_base_8_units_digit :
  let sum := 324 + 73
  let product := sum * 27
  product % 8 = 7 :=
by
  let sum := 324 + 73
  let product := sum * 27
  have h : product % 8 = 7 := by
    sorry
  exact h

end NUMINAMATH_GPT_product_base_8_units_digit_l124_12402


namespace NUMINAMATH_GPT_find_y_in_terms_of_x_l124_12458

theorem find_y_in_terms_of_x (x y : ℝ) (h : x - 2 = 4 * (y - 1) + 3) : 
  y = (1 / 4) * x - (1 / 4) := 
by
  sorry

end NUMINAMATH_GPT_find_y_in_terms_of_x_l124_12458


namespace NUMINAMATH_GPT_product_of_areas_eq_square_of_volume_l124_12404

theorem product_of_areas_eq_square_of_volume 
(x y z d : ℝ) 
(h1 : d^2 = x^2 + y^2 + z^2) :
  (x * y) * (y * z) * (z * x) = (x * y * z) ^ 2 :=
by sorry

end NUMINAMATH_GPT_product_of_areas_eq_square_of_volume_l124_12404


namespace NUMINAMATH_GPT_contrapositive_proof_l124_12432

-- Defining the necessary variables and the hypothesis
variables (a b : ℝ)

theorem contrapositive_proof (h : a^2 - b^2 + 2 * a - 4 * b - 3 ≠ 0) : a - b ≠ 1 :=
sorry

end NUMINAMATH_GPT_contrapositive_proof_l124_12432


namespace NUMINAMATH_GPT_twelfth_term_arithmetic_sequence_l124_12478

-- Given conditions
def first_term : ℚ := 1 / 4
def common_difference : ℚ := 1 / 2

-- Statement to prove
theorem twelfth_term_arithmetic_sequence :
  (first_term + 11 * common_difference) = 23 / 4 :=
by
  sorry

end NUMINAMATH_GPT_twelfth_term_arithmetic_sequence_l124_12478


namespace NUMINAMATH_GPT_percentage_of_page_used_l124_12451

theorem percentage_of_page_used (length width side_margin top_margin : ℝ) (h_length : length = 30) (h_width : width = 20) (h_side_margin : side_margin = 2) (h_top_margin : top_margin = 3) :
  ( ((length - 2 * top_margin) * (width - 2 * side_margin)) / (length * width) ) * 100 = 64 := 
by
  sorry

end NUMINAMATH_GPT_percentage_of_page_used_l124_12451


namespace NUMINAMATH_GPT_line_from_complex_condition_l124_12441

theorem line_from_complex_condition (z : ℂ) (h : ∃ x y : ℝ, z = x + y * I ∧ (3 * y + 4 * x = 0)) : 
  ∃ (a b : ℝ), (∀ (x y : ℝ), z = x + y * I → 3 * y + 4 * x = 0 → z = a + b * I ∧ 4 * x + 3 * y = 0) := 
sorry

end NUMINAMATH_GPT_line_from_complex_condition_l124_12441


namespace NUMINAMATH_GPT_solution_set_of_inequality_group_l124_12430

theorem solution_set_of_inequality_group (x : ℝ) : (x > -3 ∧ x < 5) ↔ (-3 < x ∧ x < 5) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_group_l124_12430


namespace NUMINAMATH_GPT_arithmetic_mean_bc_diff_l124_12473

variables (a b c μ : ℝ)

theorem arithmetic_mean_bc_diff 
  (h1 : (a + b) / 2 = μ + 5)
  (h2 : (a + c) / 2 = μ - 8)
  (h3 : μ = (a + b + c) / 3) :
  (b + c) / 2 = μ + 3 :=
sorry

end NUMINAMATH_GPT_arithmetic_mean_bc_diff_l124_12473


namespace NUMINAMATH_GPT_total_payment_is_correct_l124_12457

-- Define the number of friends
def number_of_friends : ℕ := 7

-- Define the amount each friend paid
def amount_per_friend : ℝ := 70.0

-- Define the total amount paid
def total_amount_paid : ℝ := number_of_friends * amount_per_friend

-- Prove that the total amount paid is 490.0
theorem total_payment_is_correct : total_amount_paid = 490.0 := by 
  -- Here, the proof would be filled in
  sorry

end NUMINAMATH_GPT_total_payment_is_correct_l124_12457


namespace NUMINAMATH_GPT_number_of_cans_per_set_l124_12465

noncomputable def ice_cream_original_price : ℝ := 12
noncomputable def ice_cream_discount : ℝ := 2
noncomputable def ice_cream_sale_price : ℝ := ice_cream_original_price - ice_cream_discount
noncomputable def number_of_tubs : ℝ := 2
noncomputable def total_money_spent : ℝ := 24
noncomputable def cost_of_juice_set : ℝ := 2
noncomputable def number_of_cans_in_juice_set : ℕ := 10

theorem number_of_cans_per_set (n : ℕ) (h : cost_of_juice_set * n = number_of_cans_in_juice_set) : (n / 2) = 5 :=
by sorry

end NUMINAMATH_GPT_number_of_cans_per_set_l124_12465


namespace NUMINAMATH_GPT_area_inner_square_l124_12492

theorem area_inner_square (ABCD_side : ℝ) (BE : ℝ) (EFGH_area : ℝ) 
  (h1 : ABCD_side = Real.sqrt 50) 
  (h2 : BE = 1) :
  EFGH_area = 36 :=
by
  sorry

end NUMINAMATH_GPT_area_inner_square_l124_12492


namespace NUMINAMATH_GPT_train_length_l124_12410

/-- Given problem conditions -/
def speed_kmh := 72
def length_platform_m := 270
def time_sec := 26

/-- Convert speed to meters per second -/
def speed_mps := speed_kmh * 1000 / 3600

/-- Calculate the total distance covered -/
def distance_covered := speed_mps * time_sec

theorem train_length :
  (distance_covered - length_platform_m) = 250 :=
by
  sorry

end NUMINAMATH_GPT_train_length_l124_12410


namespace NUMINAMATH_GPT_no_two_points_same_color_distance_one_l124_12472

/-- Prove that if a plane is colored using seven colors, it is not necessary that there will be two points of the same color exactly 1 unit apart. -/
theorem no_two_points_same_color_distance_one (coloring : ℝ × ℝ → Fin 7) :
  ¬ ∀ (x y : ℝ × ℝ), (dist x y = 1) → (coloring x = coloring y) :=
by
  sorry

end NUMINAMATH_GPT_no_two_points_same_color_distance_one_l124_12472


namespace NUMINAMATH_GPT_log_difference_l124_12439

theorem log_difference {x y a : ℝ} (h : Real.log x - Real.log y = a) :
  Real.log ((x / 2)^3) - Real.log ((y / 2)^3) = 3 * a :=
by 
  sorry

end NUMINAMATH_GPT_log_difference_l124_12439


namespace NUMINAMATH_GPT_incorrect_statement_C_l124_12413

noncomputable def f (x a b c : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

theorem incorrect_statement_C (a b c : ℝ) (x0 : ℝ) (h_local_min : ∀ y, f x0 a b c ≤ f y a b c) :
  ∃ z, z < x0 ∧ ¬ (f z a b c ≤ f (z + ε) a b c) := sorry

end NUMINAMATH_GPT_incorrect_statement_C_l124_12413


namespace NUMINAMATH_GPT_ones_digit_of_73_pow_351_l124_12483

-- Definition of the problem in Lean 4
theorem ones_digit_of_73_pow_351 : (73 ^ 351) % 10 = 7 := by
  sorry

end NUMINAMATH_GPT_ones_digit_of_73_pow_351_l124_12483


namespace NUMINAMATH_GPT_gcd_of_198_and_286_l124_12427

theorem gcd_of_198_and_286:
  let a := 198 
  let b := 286 
  let pf1 : a = 2 * 3^2 * 11 := by rfl
  let pf2 : b = 2 * 11 * 13 := by rfl
  gcd a b = 22 := by sorry

end NUMINAMATH_GPT_gcd_of_198_and_286_l124_12427


namespace NUMINAMATH_GPT_find_x0_l124_12461

def f (x : ℝ) := x * abs x

theorem find_x0 (x0 : ℝ) (h : f x0 = 4) : x0 = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_x0_l124_12461
