import Mathlib

namespace NUMINAMATH_GPT_paving_rate_correct_l1325_132539

-- Define the constants
def length (L : ℝ) := L = 5.5
def width (W : ℝ) := W = 4
def cost (C : ℝ) := C = 15400
def area (A : ℝ) := A = 22

-- Given the definitions above, prove the rate per sq. meter
theorem paving_rate_correct (L W C A : ℝ) (hL : length L) (hW : width W) (hC : cost C) (hA : area A) :
  C / A = 700 := 
sorry

end NUMINAMATH_GPT_paving_rate_correct_l1325_132539


namespace NUMINAMATH_GPT_ratio_of_ages_in_two_years_l1325_132519

theorem ratio_of_ages_in_two_years
  (S M : ℕ) 
  (h1 : M = S + 22)
  (h2 : S = 20)
  (h3 : ∃ k : ℕ, M + 2 = k * (S + 2)) :
  (M + 2) / (S + 2) = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_ages_in_two_years_l1325_132519


namespace NUMINAMATH_GPT_geometric_sequence_a_10_l1325_132555

noncomputable def geometric_sequence := ℕ → ℝ

def a_3 (a r : ℝ) := a * r^2 = 3
def a_5_equals_8a_7 (a r : ℝ) := a * r^4 = 8 * a * r^6

theorem geometric_sequence_a_10 (a r : ℝ) (seq : geometric_sequence) (h₁ : a_3 a r) (h₂ : a_5_equals_8a_7 a r) :
  seq 10 = a * r^9 := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_a_10_l1325_132555


namespace NUMINAMATH_GPT_range_a_l1325_132540

open Set Real

-- Define the predicate p: real number x satisfies x^2 - 4ax + 3a^2 < 0, where a < 0
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0 ∧ a < 0

-- Define the predicate q: real number x satisfies x^2 - x - 6 ≤ 0, or x^2 + 2x - 8 > 0
def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∨ x^2 + 2*x - 8 > 0

-- Define the complement sets
def not_p_set (a : ℝ) : Set ℝ := {x | ¬p x a}
def not_q_set : Set ℝ := {x | ¬q x}

-- Define p as necessary but not sufficient condition for q
def necessary_but_not_sufficient (a : ℝ) : Prop := 
  (not_q_set ⊆ not_p_set a) ∧ ¬(not_p_set a ⊆ not_q_set)

-- The main theorem to prove
theorem range_a : {a : ℝ | necessary_but_not_sufficient a} = {a : ℝ | -4 ≤ a ∧ a < 0 ∨ a ≤ -4} :=
by
  sorry

end NUMINAMATH_GPT_range_a_l1325_132540


namespace NUMINAMATH_GPT_geometric_series_common_ratio_l1325_132597

theorem geometric_series_common_ratio 
  (a : ℝ) (S : ℝ) (h_a : a = 500) (h_S : S = 3000) :
  ∃ r : ℝ, r = 5 / 6 :=
by
  sorry

end NUMINAMATH_GPT_geometric_series_common_ratio_l1325_132597


namespace NUMINAMATH_GPT_graph_is_empty_l1325_132554

theorem graph_is_empty :
  ¬∃ x y : ℝ, 4 * x^2 + 9 * y^2 - 16 * x - 36 * y + 64 = 0 :=
by
  -- the proof logic will go here
  sorry

end NUMINAMATH_GPT_graph_is_empty_l1325_132554


namespace NUMINAMATH_GPT_solve_for_n_l1325_132512

theorem solve_for_n (n : ℕ) (h : 2^n * 8^n = 64^(n - 30)) : n = 90 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_for_n_l1325_132512


namespace NUMINAMATH_GPT_average_rate_of_change_l1325_132561

noncomputable def f (x : ℝ) : ℝ :=
  -2 * x^2 + 1

theorem average_rate_of_change : 
  ((f 1 - f 0) / (1 - 0)) = -2 :=
by
  sorry

end NUMINAMATH_GPT_average_rate_of_change_l1325_132561


namespace NUMINAMATH_GPT_negation_of_universal_proposition_l1325_132577

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x ≥ 2 → x^2 ≥ 4)) ↔ (∃ x : ℝ, x ≥ 2 ∧ x^2 < 4) :=
by sorry

end NUMINAMATH_GPT_negation_of_universal_proposition_l1325_132577


namespace NUMINAMATH_GPT_quadratic_roots_eccentricities_l1325_132523

theorem quadratic_roots_eccentricities :
  (∃ x y : ℝ, 3 * x^2 - 4 * x + 1 = 0 ∧ 3 * y^2 - 4 * y + 1 = 0 ∧ 
              (0 ≤ x ∧ x < 1) ∧ y = 1) :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_quadratic_roots_eccentricities_l1325_132523


namespace NUMINAMATH_GPT_right_triangle_sides_l1325_132534

theorem right_triangle_sides (r R : ℝ) (a b c : ℝ) 
    (r_eq : r = 8)
    (R_eq : R = 41)
    (right_angle : a^2 + b^2 = c^2)
    (inradius : 2*r = a + b - c)
    (circumradius : 2*R = c) :
    (a = 18 ∧ b = 80 ∧ c = 82) ∨ (a = 80 ∧ b = 18 ∧ c = 82) :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_sides_l1325_132534


namespace NUMINAMATH_GPT_janet_total_miles_run_l1325_132508

/-- Janet was practicing for a marathon. She practiced for 9 days, running 8 miles each day.
Prove that Janet ran 72 miles in total. -/
theorem janet_total_miles_run (days_practiced : ℕ) (miles_per_day : ℕ) (total_miles : ℕ) 
  (h1 : days_practiced = 9) (h2 : miles_per_day = 8) : total_miles = 72 := by
  sorry

end NUMINAMATH_GPT_janet_total_miles_run_l1325_132508


namespace NUMINAMATH_GPT_find_sum_principal_l1325_132562

theorem find_sum_principal (P R : ℝ) :
  (P * (R + 5) * 10) / 100 = (P * R * 10) / 100 + 150 → P = 300 :=
by
  sorry

end NUMINAMATH_GPT_find_sum_principal_l1325_132562


namespace NUMINAMATH_GPT_percentage_peanut_clusters_is_64_l1325_132528

def total_chocolates := 50
def caramels := 3
def nougats := 2 * caramels
def truffles := caramels + 6
def other_chocolates := caramels + nougats + truffles
def peanut_clusters := total_chocolates - other_chocolates
def percentage_peanut_clusters := (peanut_clusters * 100) / total_chocolates

theorem percentage_peanut_clusters_is_64 :
  percentage_peanut_clusters = 64 := by
  sorry

end NUMINAMATH_GPT_percentage_peanut_clusters_is_64_l1325_132528


namespace NUMINAMATH_GPT_distance_between_foci_of_ellipse_l1325_132541

theorem distance_between_foci_of_ellipse : 
  let a := 5
  let b := 3
  2 * Real.sqrt (a^2 - b^2) = 8 := by
  let a := 5
  let b := 3
  sorry

end NUMINAMATH_GPT_distance_between_foci_of_ellipse_l1325_132541


namespace NUMINAMATH_GPT_angle_not_45_or_135_l1325_132513

variable {a b S : ℝ}
variable {C : ℝ} (h : S = (1/2) * a * b * Real.cos C)

theorem angle_not_45_or_135 (h : S = (1/2) * a * b * Real.cos C) : ¬ (C = 45 ∨ C = 135) :=
sorry

end NUMINAMATH_GPT_angle_not_45_or_135_l1325_132513


namespace NUMINAMATH_GPT_eggs_leftover_l1325_132549

theorem eggs_leftover :
  let abigail_eggs := 58
  let beatrice_eggs := 35
  let carson_eggs := 27
  let total_eggs := abigail_eggs + beatrice_eggs + carson_eggs
  total_eggs % 10 = 0 := by
  let abigail_eggs := 58
  let beatrice_eggs := 35
  let carson_eggs := 27
  let total_eggs := abigail_eggs + beatrice_eggs + carson_eggs
  exact Nat.mod_eq_zero_of_dvd (show 10 ∣ total_eggs from by norm_num)

end NUMINAMATH_GPT_eggs_leftover_l1325_132549


namespace NUMINAMATH_GPT_residue_of_5_pow_2023_mod_11_l1325_132536

theorem residue_of_5_pow_2023_mod_11 : (5 ^ 2023) % 11 = 4 := by
  sorry

end NUMINAMATH_GPT_residue_of_5_pow_2023_mod_11_l1325_132536


namespace NUMINAMATH_GPT_find_x_l1325_132560

theorem find_x
  (x : ℤ)
  (h : 3 * x + 3 * 15 + 3 * 18 + 11 = 152) :
  x = 14 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1325_132560


namespace NUMINAMATH_GPT_sacred_k_words_n10_k4_l1325_132548

/- Definitions for the problem -/
def sacred_k_words_count (n k : ℕ) (hk : k < n / 2) : ℕ :=
  n * Nat.choose (n - k - 1) (k - 1) * (Nat.factorial k / k)

theorem sacred_k_words_n10_k4 : sacred_k_words_count 10 4 (by norm_num : 4 < 10 / 2) = 600 := by
  sorry

end NUMINAMATH_GPT_sacred_k_words_n10_k4_l1325_132548


namespace NUMINAMATH_GPT_sum_of_longest_altitudes_l1325_132592

theorem sum_of_longest_altitudes (a b c : ℝ) (h₁ : a = 6) (h₂ : b = 8) (h₃ : c = 10) (h₄ : a^2 + b^2 = c^2) :
  a + b = 14 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_longest_altitudes_l1325_132592


namespace NUMINAMATH_GPT_range_of_k_l1325_132599

theorem range_of_k 
  (k : ℝ) 
  (line_intersects_hyperbola : ∃ x y : ℝ, y = k * x + 2 ∧ x^2 - y^2 = 6) : 
  -Real.sqrt (15) / 3 < k ∧ k < Real.sqrt (15) / 3 := 
by
  sorry

end NUMINAMATH_GPT_range_of_k_l1325_132599


namespace NUMINAMATH_GPT_range_of_m_l1325_132565

theorem range_of_m (m : ℝ) (h : ∀ x : ℝ, x > 4 ↔ x > m) : m ≤ 4 :=
by {
  -- here we state the necessary assumptions and conclude the theorem
  -- detailed proof steps are not needed, hence sorry is used to skip the proof
  sorry
}

end NUMINAMATH_GPT_range_of_m_l1325_132565


namespace NUMINAMATH_GPT_possible_values_for_abc_l1325_132515

theorem possible_values_for_abc (a b c : ℝ)
  (h : ∀ x y z : ℤ, (a * x + b * y + c * z) ∣ (b * x + c * y + a * z)) :
  (a, b, c) = (1, 0, 0) ∨ (a, b, c) = (0, 1, 0) ∨ (a, b, c) = (0, 0, 1) ∨
  (a, b, c) = (-1, 0, 0) ∨ (a, b, c) = (0, -1, 0) ∨ (a, b, c) = (0, 0, -1) :=
sorry

end NUMINAMATH_GPT_possible_values_for_abc_l1325_132515


namespace NUMINAMATH_GPT_total_problems_l1325_132596

-- We define the conditions as provided.
variables (p t : ℕ) -- p and t are positive whole numbers
variables (p_gt_10 : 10 < p) -- p is more than 10

theorem total_problems (p t : ℕ) (p_gt_10 : 10 < p) (h : p * t = (2 * p - 4) * (t - 2)):
  p * t = 60 :=
by
  sorry

end NUMINAMATH_GPT_total_problems_l1325_132596


namespace NUMINAMATH_GPT_adah_practiced_total_hours_l1325_132501

theorem adah_practiced_total_hours :
  let minutes_per_day := 86
  let days_practiced := 2
  let minutes_other_days := 278
  let total_minutes := (minutes_per_day * days_practiced) + minutes_other_days
  let total_hours := total_minutes / 60
  total_hours = 7.5 :=
by
  sorry

end NUMINAMATH_GPT_adah_practiced_total_hours_l1325_132501


namespace NUMINAMATH_GPT_vegan_menu_fraction_suitable_l1325_132535

theorem vegan_menu_fraction_suitable (vegan_dishes total_dishes vegan_dishes_with_gluten_or_dairy : ℕ)
  (h1 : vegan_dishes = 9)
  (h2 : vegan_dishes = 3 * total_dishes / 10)
  (h3 : vegan_dishes_with_gluten_or_dairy = 7) :
  (vegan_dishes - vegan_dishes_with_gluten_or_dairy) / total_dishes = 1 / 15 := by
  sorry

end NUMINAMATH_GPT_vegan_menu_fraction_suitable_l1325_132535


namespace NUMINAMATH_GPT_olivia_packs_of_basketball_cards_l1325_132506

-- Definitions for the given conditions
def pack_cost : ℕ := 3
def deck_cost : ℕ := 4
def number_of_decks : ℕ := 5
def total_money : ℕ := 50
def change_received : ℕ := 24

-- Statement to be proved
theorem olivia_packs_of_basketball_cards (x : ℕ) (hx : pack_cost * x + deck_cost * number_of_decks = total_money - change_received) : x = 2 :=
by 
  sorry

end NUMINAMATH_GPT_olivia_packs_of_basketball_cards_l1325_132506


namespace NUMINAMATH_GPT_find_constant_l1325_132531

-- Define the conditions
def is_axles (x : ℕ) : Prop := x = 5
def toll_for_truck (t : ℝ) : Prop := t = 4

-- Define the formula for the toll
def toll_formula (t : ℝ) (constant : ℝ) (x : ℕ) : Prop :=
  t = 2.50 + constant * (x - 2)

-- Proof problem statement
theorem find_constant : ∃ (constant : ℝ), 
  ∀ x : ℕ, is_axles x → toll_for_truck 4 →
  toll_formula 4 constant x → constant = 0.50 :=
sorry

end NUMINAMATH_GPT_find_constant_l1325_132531


namespace NUMINAMATH_GPT_oranges_bought_l1325_132564

theorem oranges_bought (total_cost : ℝ) 
  (selling_price_per_orange : ℝ) 
  (profit_per_orange : ℝ) 
  (cost_price_per_orange : ℝ) 
  (h1 : total_cost = 12.50)
  (h2 : selling_price_per_orange = 0.60)
  (h3 : profit_per_orange = 0.10)
  (h4 : cost_price_per_orange = selling_price_per_orange - profit_per_orange) :
  (total_cost / cost_price_per_orange) = 25 := 
by
  sorry

end NUMINAMATH_GPT_oranges_bought_l1325_132564


namespace NUMINAMATH_GPT_baseball_games_in_season_l1325_132525

def games_per_month : ℕ := 7
def months_in_season : ℕ := 2
def total_games_in_season : ℕ := games_per_month * months_in_season

theorem baseball_games_in_season : total_games_in_season = 14 := by
  sorry

end NUMINAMATH_GPT_baseball_games_in_season_l1325_132525


namespace NUMINAMATH_GPT_average_age_of_all_l1325_132581

theorem average_age_of_all (students parents : ℕ) (student_avg parent_avg : ℚ) 
  (h_students: students = 40) 
  (h_student_avg: student_avg = 12) 
  (h_parents: parents = 60) 
  (h_parent_avg: parent_avg = 36)
  : (students * student_avg + parents * parent_avg) / (students + parents) = 26.4 :=
by
  sorry

end NUMINAMATH_GPT_average_age_of_all_l1325_132581


namespace NUMINAMATH_GPT_exists_n_l1325_132576

theorem exists_n (a b c : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_gcd : Nat.gcd (Nat.gcd a b) c = 1) :
  ∃ n : ℕ, n > 0 ∧ ∀ k : ℕ, k > 0 → ¬(2^n ∣ a^k + b^k + c^k) :=
by
  sorry

end NUMINAMATH_GPT_exists_n_l1325_132576


namespace NUMINAMATH_GPT_circle_line_bisect_l1325_132533

theorem circle_line_bisect (a : ℝ) :
    (∀ x y : ℝ, (x - (-1))^2 + (y - 2)^2 = 5 → 3 * x + y + a = 0) → a = 1 :=
sorry

end NUMINAMATH_GPT_circle_line_bisect_l1325_132533


namespace NUMINAMATH_GPT_carB_distance_traveled_l1325_132521

-- Define the initial conditions
def initial_separation : ℝ := 150
def distance_carA_main_road : ℝ := 25
def distance_between_cars : ℝ := 38

-- Define the question as a theorem where we need to show the distance Car B traveled
theorem carB_distance_traveled (initial_separation distance_carA_main_road distance_between_cars : ℝ) :
  initial_separation - (distance_carA_main_road + distance_between_cars) = 87 :=
  sorry

end NUMINAMATH_GPT_carB_distance_traveled_l1325_132521


namespace NUMINAMATH_GPT_altitude_inequality_not_universally_true_l1325_132553

noncomputable def altitudes (a b c : ℝ) (h : a ≥ b ∧ b ≥ c) : Prop :=
  ∃ m_a m_b m_c : ℝ, m_a ≤ m_b ∧ m_b ≤ m_c 

noncomputable def seg_to_orthocenter (a b c : ℝ) (h : a ≥ b ∧ b ≥ c) : Prop :=
  ∃ m_a_star m_b_star m_c_star : ℝ, True

theorem altitude_inequality (a b c m_a m_b m_c : ℝ) 
  (h₀ : a ≥ b) (h₁ : b ≥ c) (h₂ : m_a ≤ m_b) (h₃ : m_b ≤ m_c) :
  (a + m_a ≥ b + m_b) ∧ (b + m_b ≥ c + m_c) :=
by
  sorry

theorem not_universally_true (a b c m_a_star m_b_star m_c_star : ℝ)
  (h₀ : a ≥ b) (h₁ : b ≥ c) :
  ¬(a + m_a_star ≥ b + m_b_star ∧ b + m_b_star ≥ c + m_c_star) :=
by
  sorry

end NUMINAMATH_GPT_altitude_inequality_not_universally_true_l1325_132553


namespace NUMINAMATH_GPT_sum_of_first_15_terms_l1325_132502

theorem sum_of_first_15_terms (a d : ℝ) 
  (h : (a + 3 * d) + (a + 11 * d) = 24) : 
  (15 / 2) * (2 * a + 14 * d) = 180 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_first_15_terms_l1325_132502


namespace NUMINAMATH_GPT_schedule_arrangement_count_l1325_132545

-- Given subjects
inductive Subject
| Chinese
| Mathematics
| Politics
| English
| PhysicalEducation
| Art

open Subject

-- Define a function to get the total number of different arrangements
def arrangement_count : Nat := 192

-- The proof statement (problem restated in Lean 4)
theorem schedule_arrangement_count :
  arrangement_count = 192 :=
by
  sorry

end NUMINAMATH_GPT_schedule_arrangement_count_l1325_132545


namespace NUMINAMATH_GPT_find_sinα_and_tanα_l1325_132527

open Real 

noncomputable def vectors (α : ℝ) := (Real.cos α, 1)

noncomputable def vectors_perpendicular (α : ℝ) := (Real.sin α, -2)

theorem find_sinα_and_tanα (α: ℝ) (hα: π < α ∧ α < 3 * π / 2)
  (h_perp: vectors_perpendicular α = (Real.sin α, -2) ∧ vectors α = (Real.cos α, 1) ∧ (vectors α).1 * (vectors_perpendicular α).1 + (vectors α).2 * (vectors_perpendicular α).2 = 0):
  (Real.sin α = - (2 * Real.sqrt 5) / 5) ∧ 
  (Real.tan (α + π / 4) = -3) := 
sorry 

end NUMINAMATH_GPT_find_sinα_and_tanα_l1325_132527


namespace NUMINAMATH_GPT_ball_redistribution_impossible_l1325_132591

noncomputable def white_boxes_initial_ball_count := 31
noncomputable def black_boxes_initial_ball_count := 26
noncomputable def white_boxes_new_ball_count := 21
noncomputable def black_boxes_new_ball_count := 16
noncomputable def white_boxes_target_ball_count := 15
noncomputable def black_boxes_target_ball_count := 10

theorem ball_redistribution_impossible
  (initial_white_boxes : ℕ)
  (initial_black_boxes : ℕ)
  (new_white_boxes : ℕ)
  (new_black_boxes : ℕ)
  (total_white_boxes : ℕ)
  (total_black_boxes : ℕ) :
  initial_white_boxes * white_boxes_initial_ball_count +
    initial_black_boxes * black_boxes_initial_ball_count =
  total_white_boxes * white_boxes_target_ball_count +
    total_black_boxes * black_boxes_target_ball_count →
  (new_white_boxes, new_black_boxes) = (total_white_boxes - initial_white_boxes, total_black_boxes - initial_black_boxes) →
  ¬(∃ total_white_boxes total_black_boxes, 
    total_white_boxes * white_boxes_target_ball_count +
    total_black_boxes * black_boxes_target_ball_count =
    initial_white_boxes * white_boxes_initial_ball_count +
    initial_black_boxes * black_boxes_initial_ball_count) :=
by sorry

end NUMINAMATH_GPT_ball_redistribution_impossible_l1325_132591


namespace NUMINAMATH_GPT_max_product_of_two_integers_l1325_132589

theorem max_product_of_two_integers (x y : ℤ) (h : x + y = 2024) : 
  x * y ≤ 1024144 := sorry

end NUMINAMATH_GPT_max_product_of_two_integers_l1325_132589


namespace NUMINAMATH_GPT_max_quarters_l1325_132586

-- Definitions stating the conditions
def total_money_in_dollars : ℝ := 4.80
def value_of_quarter : ℝ := 0.25
def value_of_dime : ℝ := 0.10

-- Theorem statement
theorem max_quarters (q : ℕ) (h1 : total_money_in_dollars = (q * value_of_quarter) + (2 * q * value_of_dime)) : q ≤ 10 :=
by {
  -- Injecting a placeholder to facilitate proof development
  sorry
}

end NUMINAMATH_GPT_max_quarters_l1325_132586


namespace NUMINAMATH_GPT_problem_statement_l1325_132552

theorem problem_statement (a : ℝ) :
  (∀ x : ℝ, (1/2 < x ∧ x < 2 → ax^2 + 5 * x - 2 > 0)) →
  a = -2 ∧ (∀ x : ℝ, -3 < x ∧ x < (1/2) → ax^2 - 5 * x + a^2 - 1 > 0) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1325_132552


namespace NUMINAMATH_GPT_consecutive_page_numbers_sum_l1325_132543

theorem consecutive_page_numbers_sum (n : ℕ) (h : n * (n + 1) * (n + 2) = 35280) :
  n + (n + 1) + (n + 2) = 96 := sorry

end NUMINAMATH_GPT_consecutive_page_numbers_sum_l1325_132543


namespace NUMINAMATH_GPT_triangle_medians_inequality_l1325_132511

-- Define the parameters
variables {a b c t_a t_b t_c D : ℝ}

-- Assume the sides and medians of the triangle and the diameter of the circumcircle
axiom sides_of_triangle (a b c : ℝ) : Prop
axiom medians_of_triangle (t_a t_b t_c : ℝ) : Prop
axiom diameter_of_circumcircle (D : ℝ) : Prop

-- The theorem to prove
theorem triangle_medians_inequality
  (h_sides : sides_of_triangle a b c)
  (h_medians : medians_of_triangle t_a t_b t_c)
  (h_diameter : diameter_of_circumcircle D)
  : (a^2 + b^2) / t_c + (b^2 + c^2) / t_a + (c^2 + a^2) / t_b ≤ 6 * D :=
sorry -- proof omitted

end NUMINAMATH_GPT_triangle_medians_inequality_l1325_132511


namespace NUMINAMATH_GPT_solve_custom_eq_l1325_132544

namespace CustomProof

def custom_mul (a b : ℕ) : ℕ := a * b + a + b

theorem solve_custom_eq (x : ℕ) (h : custom_mul 3 x = 31) : x = 7 := 
by
  sorry

end CustomProof

end NUMINAMATH_GPT_solve_custom_eq_l1325_132544


namespace NUMINAMATH_GPT_largest_angle_of_convex_hexagon_l1325_132529

theorem largest_angle_of_convex_hexagon (a d : ℕ) (h_seq : ∀ i, a + i * d < 180 ∧ a + i * d > 0)
  (h_sum : 6 * a + 15 * d = 720)
  (h_seq_arithmetic : ∀ (i j : ℕ), (a + i * d) < (a + j * d) ↔ i < j) :
  ∃ m : ℕ, (m = a + 5 * d ∧ m = 175) :=
by
  sorry

end NUMINAMATH_GPT_largest_angle_of_convex_hexagon_l1325_132529


namespace NUMINAMATH_GPT_find_coordinates_of_P_l1325_132588

-- Define the points
def P1 : ℝ × ℝ := (2, -1)
def P2 : ℝ × ℝ := (0, 5)

-- Define the point P
def P : ℝ × ℝ := (-2, 11)

-- Conditions encoded as vector relationships
def vector_P1_P (p : ℝ × ℝ) := (p.1 - P1.1, p.2 - P1.2)
def vector_PP2 (p : ℝ × ℝ) := (P2.1 - p.1, P2.2 - p.2)

-- The hypothesis that | P1P | = 2 * | PP2 |
axiom vector_relation : ∀ (p : ℝ × ℝ), 
  vector_P1_P p = (-2 * (vector_PP2 p).1, -2 * (vector_PP2 p).2) → p = P

theorem find_coordinates_of_P : P = (-2, 11) :=
by
  sorry

end NUMINAMATH_GPT_find_coordinates_of_P_l1325_132588


namespace NUMINAMATH_GPT_simplify_and_evaluate_l1325_132509

theorem simplify_and_evaluate (x : ℝ) (h : x = -2) :
  (1 / (x - 1) - 2 / (x ^ 2 - 1)) = -1 := by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l1325_132509


namespace NUMINAMATH_GPT_find_x_prime_l1325_132578

theorem find_x_prime (x : ℕ) (h1 : x > 0) (h2 : Prime (x^5 + x + 1)) : x = 1 := sorry

end NUMINAMATH_GPT_find_x_prime_l1325_132578


namespace NUMINAMATH_GPT_factor_sum_l1325_132550

theorem factor_sum (R S : ℝ) (h : ∃ (b c : ℝ), (x^2 + 3*x + 7) * (x^2 + b*x + c) = x^4 + R*x^2 + S) : R + S = 54 :=
sorry

end NUMINAMATH_GPT_factor_sum_l1325_132550


namespace NUMINAMATH_GPT_tom_gaming_system_value_l1325_132520

theorem tom_gaming_system_value
    (V : ℝ) 
    (h1 : 0.80 * V + 80 - 10 = 160 + 30) 
    : V = 150 :=
by
  -- Logical steps for the proof will be added here.
  sorry

end NUMINAMATH_GPT_tom_gaming_system_value_l1325_132520


namespace NUMINAMATH_GPT_optimal_addition_amount_l1325_132569

theorem optimal_addition_amount (a b g : ℝ) (h₁ : a = 628) (h₂ : b = 774) (h₃ : g = 718) : 
    b + a - g = 684 :=
by
  sorry

end NUMINAMATH_GPT_optimal_addition_amount_l1325_132569


namespace NUMINAMATH_GPT_canoe_downstream_speed_l1325_132593

-- Definitions based on conditions
def upstream_speed : ℝ := 9  -- upspeed
def stream_speed : ℝ := 1.5  -- vspeed

-- Theorem to prove the downstream speed
theorem canoe_downstream_speed (V_c : ℝ) (V_d : ℝ) :
  (V_c - stream_speed = upstream_speed) →
  (V_d = V_c + stream_speed) →
  V_d = 12 := by 
  intro h1 h2
  sorry

end NUMINAMATH_GPT_canoe_downstream_speed_l1325_132593


namespace NUMINAMATH_GPT_range_of_a_l1325_132556

theorem range_of_a (a : ℝ) (an bn : ℕ → ℝ)
  (h_an : ∀ n, an n = (-1) ^ (n + 2013) * a)
  (h_bn : ∀ n, bn n = 2 + (-1) ^ (n + 2014) / n)
  (h_condition : ∀ n : ℕ, 1 ≤ n → an n < bn n) :
  -2 ≤ a ∧ a < 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1325_132556


namespace NUMINAMATH_GPT_hyperbola_center_l1325_132538

theorem hyperbola_center :
  ∀ (x y : ℝ), 
  (4 * x + 8)^2 / 36 - (3 * y - 6)^2 / 25 = 1 → (x, y) = (-2, 2) :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_hyperbola_center_l1325_132538


namespace NUMINAMATH_GPT_product_grades_probabilities_l1325_132590

theorem product_grades_probabilities (P_Q P_S : ℝ) (h1 : P_Q = 0.98) (h2 : P_S = 0.21) :
  P_Q - P_S = 0.77 ∧ 1 - P_Q = 0.02 :=
by
  sorry

end NUMINAMATH_GPT_product_grades_probabilities_l1325_132590


namespace NUMINAMATH_GPT_tan_22_5_eq_half_l1325_132517

noncomputable def tan_h_LHS (θ : Real) := Real.tan θ / (1 - Real.tan θ ^ 2)

theorem tan_22_5_eq_half :
    tan_h_LHS (Real.pi / 8) = 1 / 2 :=
  sorry

end NUMINAMATH_GPT_tan_22_5_eq_half_l1325_132517


namespace NUMINAMATH_GPT_actual_total_discount_discount_difference_l1325_132507

variable {original_price : ℝ}
variable (first_discount second_discount claimed_discount actual_discount : ℝ)

-- Definitions based on the problem conditions
def discount_1 (p : ℝ) : ℝ := (1 - first_discount) * p
def discount_2 (p : ℝ) : ℝ := (1 - second_discount) * discount_1 first_discount p

-- Statements we need to prove
theorem actual_total_discount (original_price : ℝ)
  (first_discount : ℝ := 0.40) (second_discount : ℝ := 0.30) (claimed_discount : ℝ := 0.70) :
  actual_discount = 1 - discount_2 first_discount second_discount original_price := 
by 
  sorry

theorem discount_difference (original_price : ℝ)
  (first_discount : ℝ := 0.40) (second_discount : ℝ := 0.30) (claimed_discount : ℝ := 0.70)
  (actual_discount : ℝ := 0.58) :
  claimed_discount - actual_discount = 0.12 := 
by 
  sorry

end NUMINAMATH_GPT_actual_total_discount_discount_difference_l1325_132507


namespace NUMINAMATH_GPT_abs_diff_less_abs_one_minus_prod_l1325_132587

theorem abs_diff_less_abs_one_minus_prod (x y : ℝ) (hx : |x| < 1) (hy : |y| < 1) :
  |x - y| < |1 - x * y| := by
  sorry

end NUMINAMATH_GPT_abs_diff_less_abs_one_minus_prod_l1325_132587


namespace NUMINAMATH_GPT_scientific_notation_population_l1325_132579

theorem scientific_notation_population :
    ∃ (a b : ℝ), (b = 5 ∧ 1412.60 * 10 ^ 6 = a * 10 ^ b ∧ a = 1.4126) :=
sorry

end NUMINAMATH_GPT_scientific_notation_population_l1325_132579


namespace NUMINAMATH_GPT_sequence_eq_l1325_132510

-- Define the sequence and the conditions
def is_sequence (a : ℕ → ℕ) :=
  (∀ i, a i > 0) ∧ (∀ i j, i ≠ j → Nat.gcd (a i) (a j) = Nat.gcd i j)

-- The theorem we want to prove: for all i, a_i = i
theorem sequence_eq (a : ℕ → ℕ) (h : is_sequence a) : ∀ i, a i = i :=
by
  sorry

end NUMINAMATH_GPT_sequence_eq_l1325_132510


namespace NUMINAMATH_GPT_no_nonconstant_arithmetic_progression_l1325_132567

theorem no_nonconstant_arithmetic_progression (x : ℝ) :
  2 * (2 : ℝ)^(x^2) ≠ (2 : ℝ)^x + (2 : ℝ)^(x^3) :=
sorry

end NUMINAMATH_GPT_no_nonconstant_arithmetic_progression_l1325_132567


namespace NUMINAMATH_GPT_wind_power_in_scientific_notation_l1325_132522

theorem wind_power_in_scientific_notation :
  (56 * 10^6) = (5.6 * 10^7) :=
by
  sorry

end NUMINAMATH_GPT_wind_power_in_scientific_notation_l1325_132522


namespace NUMINAMATH_GPT_steve_speed_back_home_l1325_132542

-- Define a structure to hold the given conditions:
structure Conditions where
  home_to_work_distance : Float := 35 -- km
  v  : Float -- speed on the way to work in km/h
  additional_stop_time : Float := 0.25 -- hours
  total_weekly_time : Float := 30 -- hours

-- Define the main proposition:
theorem steve_speed_back_home (c: Conditions)
  (h1 : 5 * ((c.home_to_work_distance / c.v) + (c.home_to_work_distance / (2 * c.v))) + 3 * c.additional_stop_time = c.total_weekly_time) :
  2 * c.v = 18 := by
  sorry

end NUMINAMATH_GPT_steve_speed_back_home_l1325_132542


namespace NUMINAMATH_GPT_length_of_faster_train_l1325_132572

theorem length_of_faster_train
    (speed_faster : ℕ)
    (speed_slower : ℕ)
    (time_cross : ℕ)
    (h_fast : speed_faster = 72)
    (h_slow : speed_slower = 36)
    (h_time : time_cross = 15) :
    (speed_faster - speed_slower) * (1000 / 3600) * time_cross = 150 := 
by
  sorry

end NUMINAMATH_GPT_length_of_faster_train_l1325_132572


namespace NUMINAMATH_GPT_perimeter_of_shaded_region_correct_l1325_132524

noncomputable def perimeter_of_shaded_region : ℝ :=
  let r := 7
  let perimeter := 2 * r + (3 / 4) * (2 * Real.pi * r)
  perimeter

theorem perimeter_of_shaded_region_correct :
  perimeter_of_shaded_region = 14 + 10.5 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_perimeter_of_shaded_region_correct_l1325_132524


namespace NUMINAMATH_GPT_library_hospital_community_center_bells_ring_together_l1325_132503

theorem library_hospital_community_center_bells_ring_together :
  ∀ (library hospital community : ℕ), 
    (library = 18) → (hospital = 24) → (community = 30) → 
    (∀ t, (t = 0) ∨ (∃ n₁ n₂ n₃ : ℕ, 
      t = n₁ * library ∧ t = n₂ * hospital ∧ t = n₃ * community)) → 
    true :=
by
  intros
  sorry

end NUMINAMATH_GPT_library_hospital_community_center_bells_ring_together_l1325_132503


namespace NUMINAMATH_GPT_ethan_coconut_oil_per_candle_l1325_132546

noncomputable def ounces_of_coconut_oil_per_candle (candles: ℕ) (total_weight: ℝ) (beeswax_per_candle: ℝ) : ℝ :=
(total_weight - candles * beeswax_per_candle) / candles

theorem ethan_coconut_oil_per_candle :
  ounces_of_coconut_oil_per_candle 7 63 8 = 1 :=
by
  sorry

end NUMINAMATH_GPT_ethan_coconut_oil_per_candle_l1325_132546


namespace NUMINAMATH_GPT_ordered_notebooks_amount_l1325_132559

def initial_notebooks : ℕ := 10
def ordered_notebooks (x : ℕ) : ℕ := x
def lost_notebooks : ℕ := 2
def current_notebooks : ℕ := 14

theorem ordered_notebooks_amount (x : ℕ) (h : initial_notebooks + ordered_notebooks x - lost_notebooks = current_notebooks) : x = 6 :=
by
  sorry

end NUMINAMATH_GPT_ordered_notebooks_amount_l1325_132559


namespace NUMINAMATH_GPT_sum_remainder_l1325_132537

theorem sum_remainder (a b c : ℕ) (h1 : a % 53 = 33) (h2 : b % 53 = 14) (h3 : c % 53 = 9) : 
  (a + b + c) % 53 = 3 := 
by 
  sorry

end NUMINAMATH_GPT_sum_remainder_l1325_132537


namespace NUMINAMATH_GPT_f_greater_than_fp_3_2_l1325_132504

noncomputable def f (x : ℝ) (a : ℝ) := a * (x - Real.log x) + (2 * x - 1) / (x ^ 2)
noncomputable def f' (x : ℝ) (a : ℝ) := (a * x^3 - a * x^2 + 2 - 2*x) / x^3

theorem f_greater_than_fp_3_2 (x : ℝ) (h : 1 ≤ x ∧ x ≤ 2) :
  f x 1 > f' x 1 + 3 / 2 := sorry

end NUMINAMATH_GPT_f_greater_than_fp_3_2_l1325_132504


namespace NUMINAMATH_GPT_jerry_earnings_per_task_l1325_132571

theorem jerry_earnings_per_task :
  ∀ (task_hours : ℕ) (daily_hours : ℕ) (days_per_week : ℕ) (total_earnings : ℕ),
    task_hours = 2 →
    daily_hours = 10 →
    days_per_week = 5 →
    total_earnings = 1400 →
    total_earnings / ((daily_hours / task_hours) * days_per_week) = 56 :=
by
  intros task_hours daily_hours days_per_week total_earnings
  intros h_task_hours h_daily_hours h_days_per_week h_total_earnings
  sorry

end NUMINAMATH_GPT_jerry_earnings_per_task_l1325_132571


namespace NUMINAMATH_GPT_find_a5_l1325_132595

noncomputable def geometric_sequence (a : ℕ → ℝ) :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem find_a5 (a : ℕ → ℝ) (h_seq : geometric_sequence a) (h_a2 : a 2 = 2) (h_a8 : a 8 = 32) :
  a 5 = 8 :=
by
  sorry

end NUMINAMATH_GPT_find_a5_l1325_132595


namespace NUMINAMATH_GPT_range_of_a_l1325_132574

noncomputable def satisfies_condition (a : ℝ) : Prop :=
∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → abs ((1 / 2) * x^3 - a * x) ≤ 1

theorem range_of_a :
  {a : ℝ | satisfies_condition a} = {a : ℝ | - (1 / 2) ≤ a ∧ a ≤ (3 / 2)} :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1325_132574


namespace NUMINAMATH_GPT_compare_negatives_l1325_132566

noncomputable def isNegative (x : ℝ) : Prop := x < 0
noncomputable def absValue (x : ℝ) : ℝ := if x < 0 then -x else x
noncomputable def sqrt14 : ℝ := Real.sqrt 14

theorem compare_negatives : -4 < -Real.sqrt 14 := by
  have h1: Real.sqrt 16 = 4 := by
    sorry
  
  have h2: absValue (-4) = 4 := by
    sorry

  have h3: absValue (-(sqrt14)) = sqrt14 := by
    sorry

  have h4: Real.sqrt 16 > Real.sqrt 14 := by
    sorry

  show -4 < -Real.sqrt 14
  sorry

end NUMINAMATH_GPT_compare_negatives_l1325_132566


namespace NUMINAMATH_GPT_ryan_hours_english_is_6_l1325_132514

def hours_chinese : Nat := 2

def hours_english (C : Nat) : Nat := C + 4

theorem ryan_hours_english_is_6 (C : Nat) (hC : C = hours_chinese) : hours_english C = 6 :=
by
  sorry

end NUMINAMATH_GPT_ryan_hours_english_is_6_l1325_132514


namespace NUMINAMATH_GPT_platform_length_l1325_132500

theorem platform_length
  (train_length : ℤ)
  (speed_kmph : ℤ)
  (time_sec : ℤ)
  (speed_mps : speed_kmph * 1000 / 3600 = 20)
  (distance_eq : (train_length + 220) = (20 * time_sec))
  (train_length_val : train_length = 180)
  (time_sec_val : time_sec = 20) :
  220 = 220 := by
  sorry

end NUMINAMATH_GPT_platform_length_l1325_132500


namespace NUMINAMATH_GPT_simplify_expression_l1325_132551

theorem simplify_expression (x : ℝ) (h : x = 9) : 
  ((x^9 - 27 * x^6 + 729) / (x^6 - 27) = 730 + 1 / 26) :=
by {
 sorry
}

end NUMINAMATH_GPT_simplify_expression_l1325_132551


namespace NUMINAMATH_GPT_sequence_term_25_l1325_132568

theorem sequence_term_25 
  (a : ℕ → ℤ) 
  (h1 : ∀ n, n ≥ 2 → a n = (a (n - 1) + a (n + 1)) / 4)
  (h2 : a 1 = 1)
  (h3 : a 9 = 40545) : 
  a 25 = 57424611447841 := 
sorry

end NUMINAMATH_GPT_sequence_term_25_l1325_132568


namespace NUMINAMATH_GPT_num_ordered_pairs_of_squares_diff_by_144_l1325_132573

theorem num_ordered_pairs_of_squares_diff_by_144 :
  ∃ (p : Finset (ℕ × ℕ)), p.card = 4 ∧ ∀ (a b : ℕ), (a, b) ∈ p → a ≥ b ∧ a^2 - b^2 = 144 := by
  sorry

end NUMINAMATH_GPT_num_ordered_pairs_of_squares_diff_by_144_l1325_132573


namespace NUMINAMATH_GPT_prime_condition_l1325_132516

theorem prime_condition (p : ℕ) [Fact (Nat.Prime p)] :
  (∀ (a : ℕ), (1 < a ∧ a < p / 2) → (∃ (b : ℕ), (p / 2 < b ∧ b < p) ∧ p ∣ (a * b - 1))) ↔ (p = 5 ∨ p = 7 ∨ p = 13) := by
  sorry

end NUMINAMATH_GPT_prime_condition_l1325_132516


namespace NUMINAMATH_GPT_number_of_roots_l1325_132547

-- Definitions for the conditions
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def is_monotonic_in_interval (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x y, 0 ≤ x → x ≤ y → y ≤ a → f x ≤ f y

-- Main theorem to prove
theorem number_of_roots (f : ℝ → ℝ) (a : ℝ) (h1 : 0 < a) 
  (h2 : is_even_function f) (h3 : is_monotonic_in_interval f a) 
  (h4 : f 0 * f a < 0) : ∃ x0 > 0, f x0 = 0 ∧ ∃ x1 < 0, f x1 = 0 :=
sorry

end NUMINAMATH_GPT_number_of_roots_l1325_132547


namespace NUMINAMATH_GPT_problem_solution_l1325_132570

theorem problem_solution :
  (12345 * 5 + 23451 * 4 + 34512 * 3 + 45123 * 2 + 51234 * 1 = 400545) :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l1325_132570


namespace NUMINAMATH_GPT_shortest_chord_through_point_l1325_132563

theorem shortest_chord_through_point 
  (P : ℝ × ℝ) (hx : P = (2, 1))
  (h_circle : ∀ x y : ℝ, (x - 1)^2 + y^2 = 4 → (x, y) ∈ {p : ℝ × ℝ | (p.fst - 1)^2 + p.snd^2 = 4}) :
  ∃ a b c : ℝ, a = 1 ∧ b = 1 ∧ c = -3 ∧ a * (P.1) + b * (P.2) + c = 0 := 
by
  -- proof skipped
  sorry

end NUMINAMATH_GPT_shortest_chord_through_point_l1325_132563


namespace NUMINAMATH_GPT_sum_of_solutions_l1325_132557

theorem sum_of_solutions (x y : ℝ) (h : 2 * x^2 + 2 * y^2 = 20 * x - 12 * y + 68) : x + y = 2 := 
sorry

end NUMINAMATH_GPT_sum_of_solutions_l1325_132557


namespace NUMINAMATH_GPT_sum_of_three_numbers_l1325_132558

theorem sum_of_three_numbers {a b c : ℝ} (h₁ : a ≤ b ∧ b ≤ c) (h₂ : b = 10)
  (h₃ : (a + b + c) / 3 = a + 20) (h₄ : (a + b + c) / 3 = c - 25) :
  a + b + c = 45 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_three_numbers_l1325_132558


namespace NUMINAMATH_GPT_find_d_l1325_132526

def f (x : ℝ) (c : ℝ) : ℝ := 5 * x + c
def g (x : ℝ) (c : ℝ) : ℝ := c * x + 3
def h (x : ℝ) (c : ℝ) (d : ℝ) : Prop := f (g x c) c = 15 * x + d

theorem find_d (c d : ℝ) (h : ∀ x : ℝ, f (g x c) c = 15 * x + d) : d = 18 :=
by
  sorry

end NUMINAMATH_GPT_find_d_l1325_132526


namespace NUMINAMATH_GPT_triangle_orthocenter_example_l1325_132518

open Real EuclideanGeometry

def point_3d := (ℝ × ℝ × ℝ)

def orthocenter (A B C : point_3d) : point_3d := sorry

theorem triangle_orthocenter_example :
  orthocenter (2, 4, 6) (6, 5, 3) (4, 6, 7) = (4/5, 38/5, 59/5) := sorry

end NUMINAMATH_GPT_triangle_orthocenter_example_l1325_132518


namespace NUMINAMATH_GPT_range_of_k_l1325_132575

theorem range_of_k (k : ℝ) :
  (∃ x : ℝ, x^2 - 2 * x + k^2 - 1 ≤ 0) ↔ (-Real.sqrt 2 ≤ k ∧ k ≤ Real.sqrt 2) :=
by 
  sorry

end NUMINAMATH_GPT_range_of_k_l1325_132575


namespace NUMINAMATH_GPT_non_divisible_l1325_132585

theorem non_divisible (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ¬ ∃ k : ℤ, x^2 + y^2 + z^2 = k * 3 * (x * y + y * z + z * x) :=
by sorry

end NUMINAMATH_GPT_non_divisible_l1325_132585


namespace NUMINAMATH_GPT_equation_of_line_l_l1325_132530

theorem equation_of_line_l (P : ℝ × ℝ) (hP : P = (1, -1)) (θ₁ θ₂ : ℕ) (hθ₁ : θ₁ = 45) (hθ₂ : θ₂ = θ₁ * 2) (hθ₂_90 : θ₂ = 90) : 
  ∃ l : ℝ → ℝ, (∀ x, l x = l (P.fst)) := 
sorry

end NUMINAMATH_GPT_equation_of_line_l_l1325_132530


namespace NUMINAMATH_GPT_proof_a_eq_b_pow_n_l1325_132584

theorem proof_a_eq_b_pow_n
  (a b n : ℕ)
  (h : ∀ k : ℕ, k ≠ b → (b - k) ∣ (a - k^n)) :
  a = b^n := 
by sorry

end NUMINAMATH_GPT_proof_a_eq_b_pow_n_l1325_132584


namespace NUMINAMATH_GPT_vertex_h_is_3_l1325_132583

open Real

theorem vertex_h_is_3 (a b c : ℝ) (h : ℝ)
    (h_cond : 3 * (a * 3^2 + b * 3 + c) + 6 = 3) : 
    4 * (a * x^2 + b * x + c) = 12 * (x - 3)^2 + 24 → 
    h = 3 := 
by 
sorry

end NUMINAMATH_GPT_vertex_h_is_3_l1325_132583


namespace NUMINAMATH_GPT_tangent_line_at_point_l1325_132505

theorem tangent_line_at_point (x y : ℝ) (h : y = x / (x - 2)) (hx : x = 1) (hy : y = -1) : y = -2 * x + 1 :=
sorry

end NUMINAMATH_GPT_tangent_line_at_point_l1325_132505


namespace NUMINAMATH_GPT_real_root_of_P_l1325_132580

noncomputable def P : ℕ → ℝ → ℝ
| 0, x => 0
| 1, x => x
| n+2, x => x * P (n + 1) x + (1 - x) * P n x

theorem real_root_of_P (n : ℕ) (hn : 1 ≤ n) : ∀ x : ℝ, P n x = 0 → x = 0 := 
by 
  sorry

end NUMINAMATH_GPT_real_root_of_P_l1325_132580


namespace NUMINAMATH_GPT_polynomial_divisibility_l1325_132532

theorem polynomial_divisibility (
  p q r s : ℝ
) :
  (x^5 + 5 * x^4 + 10 * p * x^3 + 10 * q * x^2 + 5 * r * x + s) % (x^4 + 4 * x^3 + 6 * x^2 + 4 * x + 1) = 0 ->
  (p + q + r) * s = -2 :=
by {
  sorry
}

end NUMINAMATH_GPT_polynomial_divisibility_l1325_132532


namespace NUMINAMATH_GPT_molecular_weight_of_oxygen_part_l1325_132598

-- Define the known variables as constants
def atomic_weight_oxygen : ℝ := 16.00
def num_oxygen_atoms : ℕ := 2
def molecular_weight_compound : ℝ := 88.00

-- Define the problem as a theorem
theorem molecular_weight_of_oxygen_part :
  16.00 * 2 = 32.00 :=
by
  -- The proof will be filled in here
  sorry

end NUMINAMATH_GPT_molecular_weight_of_oxygen_part_l1325_132598


namespace NUMINAMATH_GPT_number_of_hens_l1325_132582

theorem number_of_hens (H C : Nat) (h1 : H + C = 48) (h2 : 2 * H + 4 * C = 140) : H = 26 := 
by
  sorry

end NUMINAMATH_GPT_number_of_hens_l1325_132582


namespace NUMINAMATH_GPT_eight_disks_area_sum_final_result_l1325_132594

theorem eight_disks_area_sum (r : ℝ) (C : ℝ) :
  C = 1 ∧ r = (Real.sqrt 2 + 1) / 2 → 
  8 * (π * (r ^ 2)) = 2 * π * (3 + 2 * Real.sqrt 2) :=
by
  intros h
  sorry

theorem final_result :
  let a := 6
  let b := 4
  let c := 2
  a + b + c = 12 :=
by
  intros
  norm_num

end NUMINAMATH_GPT_eight_disks_area_sum_final_result_l1325_132594
