import Mathlib

namespace NUMINAMATH_GPT_ratio_Q_P_l2360_236085

theorem ratio_Q_P : 
  ∀ (P Q : ℚ), (∀ x : ℚ, x ≠ -3 → x ≠ 0 → x ≠ 5 → 
    (P / (x + 3) + Q / (x * (x - 5)) = (x^2 - 3*x + 12) / (x^3 + x^2 - 15*x))) →
    (Q / P) = 20 / 9 :=
by
  intros P Q h
  sorry

end NUMINAMATH_GPT_ratio_Q_P_l2360_236085


namespace NUMINAMATH_GPT_burger_cost_l2360_236029

theorem burger_cost (days_in_june : ℕ) (burgers_per_day : ℕ) (total_spent : ℕ) (h1 : days_in_june = 30) (h2 : burgers_per_day = 2) (h3 : total_spent = 720) : 
  total_spent / (burgers_per_day * days_in_june) = 12 :=
by
  -- We will prove this in Lean, but skipping the proof here
  sorry

end NUMINAMATH_GPT_burger_cost_l2360_236029


namespace NUMINAMATH_GPT_sequence_bound_l2360_236075

theorem sequence_bound
  (a : ℕ → ℕ)
  (h_base0 : a 0 < a 1)
  (h_base1 : 0 < a 0 ∧ 0 < a 1)
  (h_recur : ∀ n, 2 ≤ n → a n = 3 * a (n - 1) - 2 * a (n - 2)) :
  a 100 > 2^99 :=
by
  sorry

end NUMINAMATH_GPT_sequence_bound_l2360_236075


namespace NUMINAMATH_GPT_chemical_x_added_l2360_236003

theorem chemical_x_added (initial_volume : ℝ) (initial_percentage : ℝ) (final_percentage : ℝ) : 
  initial_volume = 80 → initial_percentage = 0.2 → final_percentage = 0.36 → 
  ∃ (a : ℝ), 0.20 * initial_volume + a = 0.36 * (initial_volume + a) ∧ a = 20 :=
by
  intros h1 h2 h3
  use 20
  sorry

end NUMINAMATH_GPT_chemical_x_added_l2360_236003


namespace NUMINAMATH_GPT_owner_overtakes_thief_l2360_236017

theorem owner_overtakes_thief :
  ∀ (speed_thief speed_owner : ℕ) (time_theft_discovered : ℝ), 
    speed_thief = 45 →
    speed_owner = 50 →
    time_theft_discovered = 0.5 →
    (time_theft_discovered + (45 * 0.5) / (speed_owner - speed_thief)) = 5 := 
by
  intros speed_thief speed_owner time_theft_discovered h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  done

end NUMINAMATH_GPT_owner_overtakes_thief_l2360_236017


namespace NUMINAMATH_GPT_evaluate_double_sum_l2360_236045

theorem evaluate_double_sum :
  ∑' m : ℕ, ∑' n : ℕ, (1 : ℝ) / (m + 1) ^ 2 / (n + 1) / (m + n + 3) = 1 := by
  sorry

end NUMINAMATH_GPT_evaluate_double_sum_l2360_236045


namespace NUMINAMATH_GPT_num_zeros_in_expansion_l2360_236074

noncomputable def bigNum := (10^11 - 2) ^ 2

theorem num_zeros_in_expansion : ∀ n : ℕ, bigNum = n ↔ (n = 9999999999900000000004) := sorry

end NUMINAMATH_GPT_num_zeros_in_expansion_l2360_236074


namespace NUMINAMATH_GPT_unique_solution_l2360_236035

theorem unique_solution (x y z n : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hn : 2 ≤ n) (h_y_bound : y ≤ 5 * 2^(2*n)) :
  x^(2*n+1) - y^(2*n+1) = x * y * z + 2^(2*n+1) → (x, y, z, n) = (3, 1, 70, 2) :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_l2360_236035


namespace NUMINAMATH_GPT_find_a_b_sum_l2360_236021

-- Definitions for the conditions
def equation1 (a : ℝ) : Prop := 3 = (1 / 3) * 6 + a
def equation2 (b : ℝ) : Prop := 6 = (1 / 3) * 3 + b

theorem find_a_b_sum : 
  ∃ (a b : ℝ), equation1 a ∧ equation2 b ∧ (a + b = 6) :=
sorry

end NUMINAMATH_GPT_find_a_b_sum_l2360_236021


namespace NUMINAMATH_GPT_determine_x_l2360_236071

noncomputable def x_candidates := { x : ℝ | x = (3 + Real.sqrt 105) / 24 ∨ x = (3 - Real.sqrt 105) / 24 }

theorem determine_x (x y : ℝ) (h_y : y = 3 * x) 
  (h_eq : 4 * y ^ 2 + 2 * y + 7 = 3 * (8 * x ^ 2 + y + 3)) :
  x ∈ x_candidates :=
by
  sorry

end NUMINAMATH_GPT_determine_x_l2360_236071


namespace NUMINAMATH_GPT_common_solution_ys_l2360_236013

theorem common_solution_ys : 
  {y : ℝ | ∃ x : ℝ, x^2 + y^2 = 9 ∧ x^2 + 2*y = 7} = {1 + Real.sqrt 3, 1 - Real.sqrt 3} :=
sorry

end NUMINAMATH_GPT_common_solution_ys_l2360_236013


namespace NUMINAMATH_GPT_Rajesh_days_to_complete_l2360_236034

theorem Rajesh_days_to_complete (Mahesh_days : ℕ) (Rajesh_days : ℕ) (Total_days : ℕ)
  (h1 : Mahesh_days = 45) (h2 : Total_days - 20 = Rajesh_days) (h3 : Total_days = 54) :
  Rajesh_days = 34 :=
by
  sorry

end NUMINAMATH_GPT_Rajesh_days_to_complete_l2360_236034


namespace NUMINAMATH_GPT_truth_values_set1_truth_values_set2_l2360_236002

-- Definitions for set (1)
def p1 : Prop := Prime 3
def q1 : Prop := Even 3

-- Definitions for set (2)
def p2 (x : Int) : Prop := x = -2 ∧ (x^2 + x - 2 = 0)
def q2 (x : Int) : Prop := x = 1 ∧ (x^2 + x - 2 = 0)

-- Theorem for set (1)
theorem truth_values_set1 : 
  (p1 ∨ q1) = true ∧ (p1 ∧ q1) = false ∧ (¬p1) = false := by sorry

-- Theorem for set (2)
theorem truth_values_set2 (x : Int) :
  (p2 x ∨ q2 x) = true ∧ (p2 x ∧ q2 x) = true ∧ (¬p2 x) = false := by sorry

end NUMINAMATH_GPT_truth_values_set1_truth_values_set2_l2360_236002


namespace NUMINAMATH_GPT_negation_exists_to_forall_l2360_236066

theorem negation_exists_to_forall (P : ℝ → Prop) (h : ∃ x : ℝ, x^2 + 3 * x + 2 < 0) :
  (¬ (∃ x : ℝ, x^2 + 3 * x + 2 < 0)) ↔ (∀ x : ℝ, x^2 + 3 * x + 2 ≥ 0) := by
sorry

end NUMINAMATH_GPT_negation_exists_to_forall_l2360_236066


namespace NUMINAMATH_GPT_water_left_l2360_236038

-- Conditions
def initial_water : ℚ := 3
def water_used : ℚ := 11 / 8

-- Proposition to be proven
theorem water_left :
  initial_water - water_used = 13 / 8 := by
  sorry

end NUMINAMATH_GPT_water_left_l2360_236038


namespace NUMINAMATH_GPT_parallel_lines_slope_l2360_236068

theorem parallel_lines_slope (a : ℝ) : 
  let m1 := - (a / 2)
  let m2 := 3
  ax + 2 * y + 2 = 0 ∧ 3 * x - y - 2 = 0 → m1 = m2 → a = -6 := 
by
  intros
  sorry

end NUMINAMATH_GPT_parallel_lines_slope_l2360_236068


namespace NUMINAMATH_GPT_fifth_dog_weight_l2360_236009

theorem fifth_dog_weight (y : ℝ) (h : (25 + 31 + 35 + 33) / 4 = (25 + 31 + 35 + 33 + y) / 5) : y = 31 :=
by
  sorry

end NUMINAMATH_GPT_fifth_dog_weight_l2360_236009


namespace NUMINAMATH_GPT_inequality_abc_distinct_positive_l2360_236056

theorem inequality_abc_distinct_positive
  (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hab : a ≠ b) (hac : a ≠ c) (had : a ≠ d) (hbc : b ≠ c) (hbd : b ≠ d) (hcd : c ≠ d) :
  (a^2 / b + b^2 / c + c^2 / d + d^2 / a > a + b + c + d) := 
by
  sorry

end NUMINAMATH_GPT_inequality_abc_distinct_positive_l2360_236056


namespace NUMINAMATH_GPT_trajectory_curve_point_F_exists_l2360_236052

noncomputable def curve_C := { p : ℝ × ℝ | (p.1 - 1/2)^2 + (p.2 - 1/2)^2 = 4 }

theorem trajectory_curve (M : ℝ × ℝ) (p : ℝ × ℝ) (q : ℝ × ℝ) :
    M = ((p.1 + q.1) / 2, (p.2 + q.2) / 2) → 
    p.1^2 + p.2^2 = 9 → 
    q.1^2 + q.2^2 = 9 →
    (p.1 - 1)^2 + (p.2 - 1)^2 > 0 → 
    (q.1 - 1)^2 + (q.2 - 1)^2 > 0 → 
    ((p.1 - 1) * (q.1 - 1) + (p.2 - 1) * (q.2 - 1) = 0) →
    (M.1 - 1/2)^2 + (M.2 - 1/2)^2 = 4 :=
sorry

theorem point_F_exists (E D : ℝ × ℝ) (F : ℝ × ℝ) (H : ℝ × ℝ) :
    E = (9/2, 1/2) → D = (1/2, 1/2) → F.2 = 1/2 → 
    (∃ t : ℝ, t ≠ 9/2 ∧ F.1 = t) →
    (H ∈ curve_C) →
    ((H.1 - 9/2)^2 + (H.2 - 1/2)^2) / ((H.1 - F.1)^2 + (H.2 - 1/2)^2) = 24 * (15 - 8 * H.1) / ((t^2 + 15/4) * (24)) :=
sorry

end NUMINAMATH_GPT_trajectory_curve_point_F_exists_l2360_236052


namespace NUMINAMATH_GPT_profit_percentage_calculation_l2360_236069

def selling_price : ℝ := 120
def cost_price : ℝ := 96

theorem profit_percentage_calculation (sp cp : ℝ) (hsp : sp = selling_price) (hcp : cp = cost_price) : 
  ((sp - cp) / cp) * 100 = 25 := 
 by
  sorry

end NUMINAMATH_GPT_profit_percentage_calculation_l2360_236069


namespace NUMINAMATH_GPT_projections_on_hypotenuse_l2360_236033

variables {a b c p q : ℝ}
variables {ρa ρb : ℝ}

-- Given conditions
variable (h1 : 0 < a ∧ 0 < b ∧ 0 < c)
variable (h2 : a < b)
variable (h3 : p = a * a / c)
variable (h4 : q = b * b / c)
variable (h5 : ρa = (a * (b + c - a)) / (a + b + c))
variable (h6 : ρb = (b * (a + c - b)) / (a + b + c))

-- Proof goal
theorem projections_on_hypotenuse 
  (h_right_triangle: a^2 + b^2 = c^2) : p < ρa ∧ q > ρb :=
by
  sorry

end NUMINAMATH_GPT_projections_on_hypotenuse_l2360_236033


namespace NUMINAMATH_GPT_soccer_league_games_l2360_236039

theorem soccer_league_games (n_teams games_played : ℕ) (h1 : n_teams = 10) (h2 : games_played = 45) :
  ∃ k : ℕ, (n_teams * (n_teams - 1)) / 2 = games_played ∧ k = 1 :=
by
  sorry

end NUMINAMATH_GPT_soccer_league_games_l2360_236039


namespace NUMINAMATH_GPT_number_of_students_l2360_236096

theorem number_of_students (x : ℕ) (total_cards : ℕ) (h : x * (x - 1) = total_cards) (h_total : total_cards = 182) : x = 14 :=
by
  sorry

end NUMINAMATH_GPT_number_of_students_l2360_236096


namespace NUMINAMATH_GPT_find_value_of_p_l2360_236089

theorem find_value_of_p (p q : ℚ) (h1 : p + q = 3 / 4)
    (h2 : 45 * p^8 * q^2 = 120 * p^7 * q^3) : p = 6 / 11 :=
by
    sorry

end NUMINAMATH_GPT_find_value_of_p_l2360_236089


namespace NUMINAMATH_GPT_circle_graph_to_bar_graph_correct_l2360_236079

theorem circle_graph_to_bar_graph_correct :
  ∀ (white black gray blue : ℚ) (w_proportion b_proportion g_proportion blu_proportion : ℚ),
    white = 1/2 →
    black = 1/4 →
    gray = 1/8 →
    blue = 1/8 →
    w_proportion = 1/2 →
    b_proportion = 1/4 →
    g_proportion = 1/8 →
    blu_proportion = 1/8 →
    white = w_proportion ∧ black = b_proportion ∧ gray = g_proportion ∧ blue = blu_proportion :=
by
sorry

end NUMINAMATH_GPT_circle_graph_to_bar_graph_correct_l2360_236079


namespace NUMINAMATH_GPT_complement_of_A_in_U_l2360_236094

namespace SetTheory

def U : Set ℤ := {-1, 0, 1, 2}
def A : Set ℤ := {-1, 1, 2}

theorem complement_of_A_in_U :
  (U \ A) = {0} := by
  sorry

end SetTheory

end NUMINAMATH_GPT_complement_of_A_in_U_l2360_236094


namespace NUMINAMATH_GPT_balls_distribution_ways_l2360_236091

theorem balls_distribution_ways : 
  ∃ (ways : ℕ), ways = 15 := by
  sorry

end NUMINAMATH_GPT_balls_distribution_ways_l2360_236091


namespace NUMINAMATH_GPT_poodle_barks_proof_l2360_236073

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

end NUMINAMATH_GPT_poodle_barks_proof_l2360_236073


namespace NUMINAMATH_GPT_avg_percentage_students_l2360_236077

-- Define the function that calculates the average percentage of all students
def average_percent (n1 n2 : ℕ) (p1 p2 : ℕ) : ℕ :=
  (n1 * p1 + n2 * p2) / (n1 + n2)

-- Define the properties of the numbers of students and their respective percentages
def students_avg : Prop :=
  average_percent 15 10 70 90 = 78

-- The main theorem: Prove that given the conditions, the average percentage is 78%
theorem avg_percentage_students : students_avg :=
  by
    -- The proof will be provided here.
    sorry

end NUMINAMATH_GPT_avg_percentage_students_l2360_236077


namespace NUMINAMATH_GPT_gasoline_added_correct_l2360_236000

def tank_capacity := 48
def initial_fraction := 3 / 4
def final_fraction := 9 / 10

def gasoline_at_initial_fraction (capacity: ℝ) (fraction: ℝ) : ℝ := capacity * fraction
def gasoline_at_final_fraction (capacity: ℝ) (fraction: ℝ) : ℝ := capacity * fraction
def gasoline_added (initial: ℝ) (final: ℝ) : ℝ := final - initial

theorem gasoline_added_correct (capacity: ℝ) (initial_fraction: ℝ) (final_fraction: ℝ)
  (h_capacity : capacity = 48) (h_initial : initial_fraction = 3 / 4) (h_final : final_fraction = 9 / 10) :
  gasoline_added (gasoline_at_initial_fraction capacity initial_fraction) (gasoline_at_final_fraction capacity final_fraction) = 7.2 :=
by
  sorry

end NUMINAMATH_GPT_gasoline_added_correct_l2360_236000


namespace NUMINAMATH_GPT_area_under_curve_l2360_236040

theorem area_under_curve : 
  ∫ x in (1/2 : ℝ)..(2 : ℝ), (1 / x) = 2 * Real.log 2 := by
  sorry

end NUMINAMATH_GPT_area_under_curve_l2360_236040


namespace NUMINAMATH_GPT_maximal_cardinality_set_l2360_236078

theorem maximal_cardinality_set (n : ℕ) (h_n : n ≥ 2) :
  ∃ M : Finset (ℕ × ℕ), ∀ (j k : ℕ), (1 ≤ j ∧ j < k ∧ k ≤ n) → 
  ((j, k) ∈ M → ∀ m, (k, m) ∉ M) ∧ 
  M.card = ⌊(n * n / 4 : ℝ)⌋ :=
by
  sorry

end NUMINAMATH_GPT_maximal_cardinality_set_l2360_236078


namespace NUMINAMATH_GPT_derivative_of_f_l2360_236011

noncomputable def f (x : ℝ) : ℝ := Real.exp (-x) * (Real.cos x + Real.sin x)

theorem derivative_of_f (x : ℝ) : deriv f x = -2 * Real.exp (-x) * Real.sin x :=
by sorry

end NUMINAMATH_GPT_derivative_of_f_l2360_236011


namespace NUMINAMATH_GPT_smallest_positive_angle_l2360_236048

theorem smallest_positive_angle :
  ∀ (x : ℝ), 12 * (Real.sin x)^3 * (Real.cos x)^3 - 2 * (Real.sin x)^3 * (Real.cos x)^3 = 1 → 
  x = 15 * (Real.pi / 180) :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_smallest_positive_angle_l2360_236048


namespace NUMINAMATH_GPT_chandler_bike_purchase_l2360_236084

theorem chandler_bike_purchase : 
  ∀ (x : ℕ), (120 + 20 * x = 640) → x = 26 := 
by
  sorry

end NUMINAMATH_GPT_chandler_bike_purchase_l2360_236084


namespace NUMINAMATH_GPT_all_defective_is_impossible_l2360_236093

def total_products : ℕ := 10
def defective_products : ℕ := 2
def selected_products : ℕ := 3

theorem all_defective_is_impossible :
  ∀ (products : Finset ℕ),
  products.card = selected_products →
  ∀ (product_ids : Finset ℕ),
  product_ids.card = defective_products →
  products ⊆ product_ids → False :=
by
  sorry

end NUMINAMATH_GPT_all_defective_is_impossible_l2360_236093


namespace NUMINAMATH_GPT_find_value_l2360_236057

theorem find_value (
  a b c d e f : ℝ) 
  (h1 : a * b * c = 65) 
  (h2 : b * c * d = 65) 
  (h3 : c * d * e = 1000) 
  (h4 : (a * f) / (c * d) = 0.25) :
  d * e * f = 250 := 
sorry

end NUMINAMATH_GPT_find_value_l2360_236057


namespace NUMINAMATH_GPT_emily_subtracts_99_from_50sq_to_get_49sq_l2360_236031

-- Define the identity for squares
theorem emily_subtracts_99_from_50sq_to_get_49sq :
  ∀ (x : ℕ), (49 : ℕ) = (50 - 1) → (x = 50 → 49^2 = 50^2 - 99) := by
  intro x h1 h2
  sorry

end NUMINAMATH_GPT_emily_subtracts_99_from_50sq_to_get_49sq_l2360_236031


namespace NUMINAMATH_GPT_words_per_page_l2360_236024

theorem words_per_page 
    (p : ℕ) 
    (h1 : 150 > 0) 
    (h2 : 150 * p ≡ 200 [MOD 221]) :
    p = 118 := 
by sorry

end NUMINAMATH_GPT_words_per_page_l2360_236024


namespace NUMINAMATH_GPT_prob_three_students_exactly_two_absent_l2360_236018

def prob_absent : ℚ := 1 / 30
def prob_present : ℚ := 29 / 30

theorem prob_three_students_exactly_two_absent :
  (prob_absent * prob_absent * prob_present) * 3 = 29 / 9000 := by
  sorry

end NUMINAMATH_GPT_prob_three_students_exactly_two_absent_l2360_236018


namespace NUMINAMATH_GPT_induction_base_case_not_necessarily_one_l2360_236010

theorem induction_base_case_not_necessarily_one :
  (∀ (P : ℕ → Prop) (n₀ : ℕ), (P n₀) → (∀ n, n ≥ n₀ → P n → P (n + 1)) → ∀ n, n ≥ n₀ → P n) ↔
  (∃ n₀ : ℕ, n₀ ≠ 1) :=
sorry

end NUMINAMATH_GPT_induction_base_case_not_necessarily_one_l2360_236010


namespace NUMINAMATH_GPT_marikas_father_twice_her_age_l2360_236030

theorem marikas_father_twice_her_age (birth_year : ℤ) (marika_age : ℤ) (father_multiple : ℕ) :
  birth_year = 2006 ∧ marika_age = 10 ∧ father_multiple = 5 →
  ∃ x : ℤ, birth_year + x = 2036 ∧ (father_multiple * marika_age + x) = 2 * (marika_age + x) :=
by {
  sorry
}

end NUMINAMATH_GPT_marikas_father_twice_her_age_l2360_236030


namespace NUMINAMATH_GPT_probability_of_D_given_T_l2360_236004

-- Definitions based on the conditions given in the problem.
def pr_D : ℚ := 1 / 400
def pr_Dc : ℚ := 399 / 400
def pr_T_given_D : ℚ := 1
def pr_T_given_Dc : ℚ := 0.05
def pr_T : ℚ := pr_T_given_D * pr_D + pr_T_given_Dc * pr_Dc

-- Statement to prove 
theorem probability_of_D_given_T : pr_T ≠ 0 → (pr_T_given_D * pr_D) / pr_T = 20 / 419 :=
by
  intros h1
  unfold pr_T pr_D pr_Dc pr_T_given_D pr_T_given_Dc
  -- Mathematical steps are skipped in Lean by inserting sorry
  sorry

-- Check that the statement can be built successfully
example : pr_D = 1 / 400 := by rfl
example : pr_Dc = 399 / 400 := by rfl
example : pr_T_given_D = 1 := by rfl
example : pr_T_given_Dc = 0.05 := by rfl
example : pr_T = (1 * (1 / 400) + 0.05 * (399 / 400)) := by rfl

end NUMINAMATH_GPT_probability_of_D_given_T_l2360_236004


namespace NUMINAMATH_GPT_four_numbers_sum_divisible_by_2016_l2360_236067

theorem four_numbers_sum_divisible_by_2016 {x : Fin 65 → ℕ} (h_distinct: Function.Injective x) (h_range: ∀ i, x i ≤ 2016) :
  ∃ a b c d : Fin 65, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ (x a + x b - x c - x d) % 2016 = 0 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_four_numbers_sum_divisible_by_2016_l2360_236067


namespace NUMINAMATH_GPT_fraction_equality_l2360_236051

theorem fraction_equality (a b : ℝ) (h : (1 / a) - (1 / b) = 4) :
  (a - 2 * a * b - b) / (2 * a - 2 * b + 7 * a * b) = 6 :=
by
  sorry

end NUMINAMATH_GPT_fraction_equality_l2360_236051


namespace NUMINAMATH_GPT_geographic_info_tech_helps_western_development_l2360_236044

namespace GeographicInfoTech

def monitors_three_gorges_project : Prop :=
  -- Point ①
  true

def monitors_ecological_environment_meteorological_changes_and_provides_accurate_info : Prop :=
  -- Point ②
  true

def tracks_migration_tibetan_antelopes : Prop :=
  -- Point ③
  true

def addresses_ecological_environment_issues_in_southwest : Prop :=
  -- Point ④
  true

noncomputable def provides_services_for_development_western_regions : Prop :=
  monitors_three_gorges_project ∧ 
  monitors_ecological_environment_meteorological_changes_and_provides_accurate_info ∧ 
  tracks_migration_tibetan_antelopes -- A (①②③)

-- Theorem stating that geographic information technology helps in ①, ②, ③ given its role
theorem geographic_info_tech_helps_western_development (h : provides_services_for_development_western_regions) :
  monitors_three_gorges_project ∧ 
  monitors_ecological_environment_meteorological_changes_and_provides_accurate_info ∧ 
  tracks_migration_tibetan_antelopes := 
by
  exact h

end GeographicInfoTech

end NUMINAMATH_GPT_geographic_info_tech_helps_western_development_l2360_236044


namespace NUMINAMATH_GPT_ab_conditions_l2360_236032

theorem ab_conditions (a b : ℝ) : ¬((a > b → a^2 > b^2) ∧ (a^2 > b^2 → a > b)) :=
by 
  sorry

end NUMINAMATH_GPT_ab_conditions_l2360_236032


namespace NUMINAMATH_GPT_Molly_swam_on_Saturday_l2360_236058

variable (total_meters : ℕ) (sunday_meters : ℕ)

def saturday_meters := total_meters - sunday_meters

theorem Molly_swam_on_Saturday : 
  total_meters = 73 ∧ sunday_meters = 28 → saturday_meters total_meters sunday_meters = 45 := by
sorry

end NUMINAMATH_GPT_Molly_swam_on_Saturday_l2360_236058


namespace NUMINAMATH_GPT_radius_of_circumscribed_circle_l2360_236006

-- Definitions based on conditions
def sector (radius : ℝ) (central_angle : ℝ) : Prop :=
  central_angle = 120 ∧ radius = 10

-- Statement of the theorem we want to prove
theorem radius_of_circumscribed_circle (r R : ℝ) (h : sector r 120) : R = 20 := 
by
  sorry

end NUMINAMATH_GPT_radius_of_circumscribed_circle_l2360_236006


namespace NUMINAMATH_GPT_side_of_larger_square_l2360_236083

theorem side_of_larger_square (s S : ℕ) (h₁ : s = 5) (h₂ : S^2 = 4 * s^2) : S = 10 := 
by sorry

end NUMINAMATH_GPT_side_of_larger_square_l2360_236083


namespace NUMINAMATH_GPT_initial_matches_l2360_236076

theorem initial_matches (x : ℕ) (h1 : (34 * x + 89) / (x + 1) = 39) : x = 10 := by
  sorry

end NUMINAMATH_GPT_initial_matches_l2360_236076


namespace NUMINAMATH_GPT_cylinder_projections_tangency_l2360_236060

def plane1 : Type := sorry
def plane2 : Type := sorry
def projection_axis : Type := sorry
def is_tangent_to (cylinder : Type) (plane : Type) : Prop := sorry
def is_base_tangent_to (cylinder : Type) (axis : Type) : Prop := sorry
def cylinder : Type := sorry

theorem cylinder_projections_tangency (P1 P2 : Type) (axis : Type)
  (h1 : is_tangent_to cylinder P1) 
  (h2 : is_tangent_to cylinder P2) 
  (h3 : is_base_tangent_to cylinder axis) : 
  ∃ (solutions : ℕ), solutions = 4 :=
sorry

end NUMINAMATH_GPT_cylinder_projections_tangency_l2360_236060


namespace NUMINAMATH_GPT_triangle_is_equilateral_l2360_236082

-- Define a triangle with angles A, B, and C
variables (A B C : ℝ)

-- The conditions of the problem
def log_sin_arithmetic_sequence : Prop :=
  Real.log (Real.sin A) + Real.log (Real.sin C) = 2 * Real.log (Real.sin B)

def angles_arithmetic_sequence : Prop :=
  2 * B = A + C

-- The theorem that the triangle is equilateral given these conditions
theorem triangle_is_equilateral :
  log_sin_arithmetic_sequence A B C → angles_arithmetic_sequence A B C → 
  A = 60 ∧ B = 60 ∧ C = 60 :=
by
  sorry

end NUMINAMATH_GPT_triangle_is_equilateral_l2360_236082


namespace NUMINAMATH_GPT_smallest_model_length_l2360_236062

theorem smallest_model_length (full_size : ℕ) (mid_size_factor smallest_size_factor : ℚ) :
  full_size = 240 →
  mid_size_factor = 1 / 10 →
  smallest_size_factor = 1 / 2 →
  (full_size * mid_size_factor) * smallest_size_factor = 12 :=
by
  intros h_full_size h_mid_size_factor h_smallest_size_factor
  sorry

end NUMINAMATH_GPT_smallest_model_length_l2360_236062


namespace NUMINAMATH_GPT_rain_difference_l2360_236008

theorem rain_difference (r_m r_t : ℝ) (h_monday : r_m = 0.9) (h_tuesday : r_t = 0.2) : r_m - r_t = 0.7 :=
by sorry

end NUMINAMATH_GPT_rain_difference_l2360_236008


namespace NUMINAMATH_GPT_money_saved_l2360_236081

noncomputable def total_savings :=
  let fox_price := 15
  let pony_price := 18
  let num_fox_pairs := 3
  let num_pony_pairs := 2
  let total_discount_rate := 0.22
  let pony_discount_rate := 0.10999999999999996
  let fox_discount_rate := total_discount_rate - pony_discount_rate
  let fox_savings := fox_price * fox_discount_rate * num_fox_pairs
  let pony_savings := pony_price * pony_discount_rate * num_pony_pairs
  fox_savings + pony_savings

theorem money_saved :
  total_savings = 8.91 :=
by
  -- We assume the savings calculations are correct as per the problem statement
  sorry

end NUMINAMATH_GPT_money_saved_l2360_236081


namespace NUMINAMATH_GPT_valid_numbers_count_l2360_236027

def is_valid_digit (d : ℕ) : Prop :=
  d ≠ 5 ∧ d < 10

def count_valid_numbers : ℕ :=
  let first_digit_choices := 8 -- from 1 to 9 excluding 5
  let second_digit_choices := 8 -- from the digits (0-9 excluding 5 and first digit)
  let third_digit_choices := 7 -- from the digits (0-9 excluding 5 and first two digits)
  let fourth_digit_choices := 6 -- from the digits (0-9 excluding 5 and first three digits)
  first_digit_choices * second_digit_choices * third_digit_choices * fourth_digit_choices

theorem valid_numbers_count : count_valid_numbers = 2688 :=
  by
  sorry

end NUMINAMATH_GPT_valid_numbers_count_l2360_236027


namespace NUMINAMATH_GPT_carA_speed_calc_l2360_236020

-- Defining the conditions of the problem
def carA_time : ℕ := 8
def carB_speed : ℕ := 25
def carB_time : ℕ := 4
def distance_ratio : ℕ := 4
def carB_distance : ℕ := carB_speed * carB_time
def carA_distance : ℕ := distance_ratio * carB_distance

-- Mathematical statement to be proven
theorem carA_speed_calc : carA_distance / carA_time = 50 := by
  sorry

end NUMINAMATH_GPT_carA_speed_calc_l2360_236020


namespace NUMINAMATH_GPT_solve_for_k_l2360_236037

theorem solve_for_k :
  ∀ (k : ℝ), (∃ x : ℝ, (3*x + 8)*(x - 6) = -50 + k*x) ↔
    k = -10 + 2*Real.sqrt 6 ∨ k = -10 - 2*Real.sqrt 6 := by
  sorry

end NUMINAMATH_GPT_solve_for_k_l2360_236037


namespace NUMINAMATH_GPT_trigonometric_identity_l2360_236090

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = -1 / 2) : 
  (1 + 2 * Real.sin α * Real.cos α) / (Real.sin α ^ 2 - Real.cos α ^ 2) = -1 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l2360_236090


namespace NUMINAMATH_GPT_volume_ratio_l2360_236065

theorem volume_ratio (a : ℕ) (b : ℕ) (ft_to_inch : ℕ) (h1 : a = 4) (h2 : b = 2 * ft_to_inch) (ft_to_inch_value : ft_to_inch = 12) :
  (a^3) / (b^3) = 1 / 216 :=
by
  sorry

end NUMINAMATH_GPT_volume_ratio_l2360_236065


namespace NUMINAMATH_GPT_num_possible_values_of_M_l2360_236049

theorem num_possible_values_of_M :
  ∃ n : ℕ, n = 8 ∧
  ∃ (a b : ℕ), (10 <= 10*a + b) ∧ (10*a + b < 100) ∧ (9*(a - b) ∈ {k : ℕ | ∃ m : ℕ, k = m^2}) := sorry

end NUMINAMATH_GPT_num_possible_values_of_M_l2360_236049


namespace NUMINAMATH_GPT_master_craftsman_total_parts_l2360_236014

theorem master_craftsman_total_parts
  (N : ℕ) -- Additional parts to be produced after the first hour
  (initial_rate : ℕ := 35) -- Initial production rate (35 parts/hour)
  (increased_rate : ℕ := initial_rate + 15) -- Increased production rate (50 parts/hour)
  (time_difference : ℝ := 1.5) -- Time difference in hours between the rates
  (eq_time_diff : (N / initial_rate) - (N / increased_rate) = time_difference) -- The given time difference condition
  : 35 + N = 210 := -- Conclusion we need to prove
sorry

end NUMINAMATH_GPT_master_craftsman_total_parts_l2360_236014


namespace NUMINAMATH_GPT_germination_rate_proof_l2360_236036

def random_number_table := [[78226, 85384, 40527, 48987, 60602, 16085, 29971, 61279],
                            [43021, 92980, 27768, 26916, 27783, 84572, 78483, 39820],
                            [61459, 39073, 79242, 20372, 21048, 87088, 34600, 74636],
                            [63171, 58247, 12907, 50303, 28814, 40422, 97895, 61421],
                            [42372, 53183, 51546, 90385, 12120, 64042, 51320, 22983]]

noncomputable def first_4_tested_seeds : List Nat :=
  let numbers_in_random_table := [390, 737, 924, 220, 372]
  numbers_in_random_table.filter (λ x => x < 850) |>.take 4

theorem germination_rate_proof :
  first_4_tested_seeds = [390, 737, 220, 372] := 
by 
  sorry

end NUMINAMATH_GPT_germination_rate_proof_l2360_236036


namespace NUMINAMATH_GPT_jeremy_gifted_37_goats_l2360_236053

def initial_horses := 100
def initial_sheep := 29
def initial_chickens := 9

def total_initial_animals := initial_horses + initial_sheep + initial_chickens
def animals_bought_by_brian := total_initial_animals / 2
def animals_left_after_brian := total_initial_animals - animals_bought_by_brian

def total_male_animals := 53
def total_female_animals := 53
def total_remaining_animals := total_male_animals + total_female_animals

def goats_gifted_by_jeremy := total_remaining_animals - animals_left_after_brian

theorem jeremy_gifted_37_goats :
  goats_gifted_by_jeremy = 37 := 
by 
  sorry

end NUMINAMATH_GPT_jeremy_gifted_37_goats_l2360_236053


namespace NUMINAMATH_GPT_stream_current_rate_l2360_236072

theorem stream_current_rate (r w : ℝ) : (
  (18 / (r + w) + 6 = 18 / (r - w)) ∧ 
  (18 / (3 * r + w) + 2 = 18 / (3 * r - w))
) → w = 6 := 
by {
  sorry
}

end NUMINAMATH_GPT_stream_current_rate_l2360_236072


namespace NUMINAMATH_GPT_find_a_given_difference_l2360_236019

theorem find_a_given_difference (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) (h₂ : |a - a^2| = 6) : a = 3 :=
sorry

end NUMINAMATH_GPT_find_a_given_difference_l2360_236019


namespace NUMINAMATH_GPT_football_team_progress_l2360_236005

theorem football_team_progress (lost_yards gained_yards : Int) : lost_yards = -5 → gained_yards = 13 → lost_yards + gained_yards = 8 := 
by
  intros h_lost h_gained
  rw [h_lost, h_gained]
  sorry

end NUMINAMATH_GPT_football_team_progress_l2360_236005


namespace NUMINAMATH_GPT_find_solutions_l2360_236087

noncomputable
def is_solution (a b c d : ℝ) : Prop :=
  a + b + c = d ∧ (1 / a + 1 / b + 1 / c = 1 / d)

theorem find_solutions (a b c d : ℝ) :
  is_solution a b c d ↔ (c = -a ∧ d = b) ∨ (c = -b ∧ d = a) :=
by
  sorry

end NUMINAMATH_GPT_find_solutions_l2360_236087


namespace NUMINAMATH_GPT_range_of_a_exists_x_ax2_ax_1_lt_0_l2360_236064

theorem range_of_a_exists_x_ax2_ax_1_lt_0 :
  {a : ℝ | ∃ x : ℝ, a * x^2 + a * x + 1 < 0} = {a : ℝ | a < 0 ∨ a > 4} :=
sorry

end NUMINAMATH_GPT_range_of_a_exists_x_ax2_ax_1_lt_0_l2360_236064


namespace NUMINAMATH_GPT_ratio_of_speeds_l2360_236012

theorem ratio_of_speeds (P R : ℝ) (total_time : ℝ) (time_rickey : ℝ)
  (h1 : total_time = 70)
  (h2 : time_rickey = 40)
  (h3 : total_time - time_rickey = 30) :
  P / R = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_speeds_l2360_236012


namespace NUMINAMATH_GPT_blue_paint_needed_l2360_236041

/-- 
If the ratio of blue paint to green paint is \(4:1\), and Sarah wants to make 40 cans of the mixture,
prove that the number of cans of blue paint needed is 32.
-/
theorem blue_paint_needed (r: ℕ) (total_cans: ℕ) (h_ratio: r = 4) (h_total: total_cans = 40) : 
  ∃ b: ℕ, b = 4 / 5 * total_cans ∧ b = 32 :=
by
  sorry

end NUMINAMATH_GPT_blue_paint_needed_l2360_236041


namespace NUMINAMATH_GPT_area_is_prime_number_l2360_236001

open Real Int

noncomputable def area_of_triangle (a : Int) : Real :=
  (a * a : Real) / 20

theorem area_is_prime_number 
  (a : Int) 
  (h1 : ∃ p : ℕ, Nat.Prime p ∧ p = ((a * a) / 20 : Real)) :
  ((a * a) / 20 : Real) = 5 :=
by 
  sorry

end NUMINAMATH_GPT_area_is_prime_number_l2360_236001


namespace NUMINAMATH_GPT_day_after_7k_days_is_friday_day_before_7k_days_is_thursday_day_after_100_days_is_sunday_l2360_236061

-- Definitions based on problem conditions and questions
def day_of_week_after (n : ℤ) (current_day : String) : String :=
  if n % 7 = 0 then current_day else
    if n % 7 = 1 then "Saturday" else
    if n % 7 = 2 then "Sunday" else
    if n % 7 = 3 then "Monday" else
    if n % 7 = 4 then "Tuesday" else
    if n % 7 = 5 then "Wednesday" else
    "Thursday"

def day_of_week_before (n : ℤ) (current_day : String) : String :=
  day_of_week_after (-n) current_day

-- Conditions
def today : String := "Friday"

-- Prove the following
theorem day_after_7k_days_is_friday (k : ℤ) : day_of_week_after (7 * k) today = "Friday" :=
by sorry

theorem day_before_7k_days_is_thursday (k : ℤ) : day_of_week_before (7 * k) today = "Thursday" :=
by sorry

theorem day_after_100_days_is_sunday : day_of_week_after 100 today = "Sunday" :=
by sorry

end NUMINAMATH_GPT_day_after_7k_days_is_friday_day_before_7k_days_is_thursday_day_after_100_days_is_sunday_l2360_236061


namespace NUMINAMATH_GPT_expression_value_l2360_236046

theorem expression_value {a b c d m : ℝ} (h1 : a + b = 0) (h2 : c * d = 1) (h3 : |m| = 1) : 
  (a + b) * c * d - 2014 * m = -2014 ∨ (a + b) * c * d - 2014 * m = 2014 := 
by
  sorry

end NUMINAMATH_GPT_expression_value_l2360_236046


namespace NUMINAMATH_GPT_jo_reading_time_l2360_236050

structure Book :=
  (totalPages : Nat)
  (currentPage : Nat)
  (pageOneHourAgo : Nat)

def readingTime (b : Book) : Nat :=
  let pagesRead := b.currentPage - b.pageOneHourAgo
  let pagesLeft := b.totalPages - b.currentPage
  pagesLeft / pagesRead

theorem jo_reading_time :
  ∀ (b : Book), b.totalPages = 210 → b.currentPage = 90 → b.pageOneHourAgo = 60 → readingTime b = 4 :=
by
  intro b h1 h2 h3
  sorry

end NUMINAMATH_GPT_jo_reading_time_l2360_236050


namespace NUMINAMATH_GPT_ladder_leaning_distance_l2360_236086

variable (m f h : ℝ)
variable (f_pos : f > 0) (h_pos : h > 0)

def distance_to_wall_upper_bound : ℝ := 12.46
def distance_to_wall_lower_bound : ℝ := 8.35

theorem ladder_leaning_distance (m f h : ℝ) (f_pos : f > 0) (h_pos : h > 0) :
  ∃ x : ℝ, x = 12.46 ∨ x = 8.35 := 
sorry

end NUMINAMATH_GPT_ladder_leaning_distance_l2360_236086


namespace NUMINAMATH_GPT_pool_ratio_three_to_one_l2360_236095

theorem pool_ratio_three_to_one (P : ℕ) (B B' : ℕ) (k : ℕ) :
  (P = 5 * B + 2) → (k * P = 5 * B' + 1) → k = 3 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_pool_ratio_three_to_one_l2360_236095


namespace NUMINAMATH_GPT_find_a_l2360_236042

noncomputable def A := {x : ℝ | x^2 - 8 * x + 15 = 0}
noncomputable def B (a : ℝ) := {x : ℝ | a * x - 1 = 0}

theorem find_a (a : ℝ) : (A ∩ B a = B a) ↔ (a = 0 ∨ a = 1/3 ∨ a = 1/5) :=
by
  sorry

end NUMINAMATH_GPT_find_a_l2360_236042


namespace NUMINAMATH_GPT_inclination_angle_of_line_3x_sqrt3y_minus1_l2360_236099

noncomputable def inclination_angle_of_line (A B C : ℝ) (h : A ≠ 0 ∧ B ≠ 0) : ℝ :=
  let m := -A / B 
  if m = Real.tan (120 * Real.pi / 180) then 120
  else 0 -- This will return 0 if the slope m does not match, for simplifying purposes

theorem inclination_angle_of_line_3x_sqrt3y_minus1 :
  inclination_angle_of_line 3 (Real.sqrt 3) (-1) (by sorry) = 120 := 
sorry

end NUMINAMATH_GPT_inclination_angle_of_line_3x_sqrt3y_minus1_l2360_236099


namespace NUMINAMATH_GPT_rectangle_side_divisible_by_4_l2360_236015

theorem rectangle_side_divisible_by_4 (a b : ℕ)
  (h1 : ∀ i, 1 ≤ i ∧ i ≤ a → i % 4 = 0)
  (h2 : ∀ j, 1 ≤ j ∧ j ≤ b → j % 4 = 0): 
  (a % 4 = 0) ∨ (b % 4 = 0) :=
sorry

end NUMINAMATH_GPT_rectangle_side_divisible_by_4_l2360_236015


namespace NUMINAMATH_GPT_only_n_is_zero_l2360_236055

theorem only_n_is_zero (n : ℕ) (h : (n^2 + 1) ∣ n) : n = 0 := 
by sorry

end NUMINAMATH_GPT_only_n_is_zero_l2360_236055


namespace NUMINAMATH_GPT_find_x_value_l2360_236070

theorem find_x_value (x : ℝ) (a b c : ℝ × ℝ × ℝ) 
  (h_a : a = (1, 1, x)) 
  (h_b : b = (1, 2, 1)) 
  (h_c : c = (1, 1, 1)) 
  (h_cond : (c - a) • (2 • b) = -2) : 
  x = 2 := 
by 
  -- the proof goes here
  sorry

end NUMINAMATH_GPT_find_x_value_l2360_236070


namespace NUMINAMATH_GPT_rod_length_l2360_236022

theorem rod_length (num_pieces : ℝ) (length_per_piece : ℝ) (h1 : num_pieces = 118.75) (h2 : length_per_piece = 0.40) : 
  num_pieces * length_per_piece = 47.5 := by
  sorry

end NUMINAMATH_GPT_rod_length_l2360_236022


namespace NUMINAMATH_GPT_compare_y1_y2_l2360_236043

noncomputable def quadratic (x : ℝ) : ℝ := -x^2 + 2

theorem compare_y1_y2 :
  let y1 := quadratic 1
  let y2 := quadratic 3
  y1 > y2 :=
by
  let y1 := quadratic 1
  let y2 := quadratic 3
  sorry

end NUMINAMATH_GPT_compare_y1_y2_l2360_236043


namespace NUMINAMATH_GPT_quarterback_passes_left_l2360_236059

noncomputable def number_of_passes (L : ℕ) : Prop :=
  let R := 2 * L
  let C := L + 2
  L + R + C = 50

theorem quarterback_passes_left : ∃ L, number_of_passes L ∧ L = 12 := by
  sorry

end NUMINAMATH_GPT_quarterback_passes_left_l2360_236059


namespace NUMINAMATH_GPT_smallest_integer_n_satisfying_inequality_l2360_236023

theorem smallest_integer_n_satisfying_inequality 
  (x y z : ℝ) : 
  (x^2 + y^2 + z^2)^2 ≤ 3 * (x^4 + y^4 + z^4) :=
sorry

end NUMINAMATH_GPT_smallest_integer_n_satisfying_inequality_l2360_236023


namespace NUMINAMATH_GPT_equal_clubs_and_students_l2360_236026

theorem equal_clubs_and_students (S C : ℕ) 
  (h1 : ∀ c : ℕ, c < C → ∃ (m : ℕ → Prop), (∃ p, m p ∧ p = 3))
  (h2 : ∀ s : ℕ, s < S → ∃ (n : ℕ → Prop), (∃ p, n p ∧ p = 3)) :
  S = C := 
by
  sorry

end NUMINAMATH_GPT_equal_clubs_and_students_l2360_236026


namespace NUMINAMATH_GPT_common_sum_of_matrix_l2360_236028

theorem common_sum_of_matrix :
  let S := (1 / 2 : ℝ) * 25 * (10 + 34)
  let adjusted_total := S + 10
  let common_sum := adjusted_total / 6
  common_sum = 93.33 :=
by
  sorry

end NUMINAMATH_GPT_common_sum_of_matrix_l2360_236028


namespace NUMINAMATH_GPT_equivalent_product_lists_l2360_236098

-- Definitions of the value assigned to each letter.
def letter_value (c : Char) : ℕ :=
  match c with
  | 'A' => 1
  | 'B' => 2
  | 'C' => 3
  | 'D' => 4
  | 'E' => 5
  | 'F' => 6
  | 'G' => 7
  | 'H' => 8
  | 'I' => 9
  | 'J' => 10
  | 'K' => 11
  | 'L' => 12
  | 'M' => 13
  | 'N' => 14
  | 'O' => 15
  | 'P' => 16
  | 'Q' => 17
  | 'R' => 18
  | 'S' => 19
  | 'T' => 20
  | 'U' => 21
  | 'V' => 22
  | 'W' => 23
  | 'X' => 24
  | 'Y' => 25
  | 'Z' => 26
  | _ => 0  -- We only care about uppercase letters A-Z

def list_product (l : List Char) : ℕ :=
  l.foldl (λ acc c => acc * (letter_value c)) 1

-- Given the list MNOP with their products equals letter values.
def MNOP := ['M', 'N', 'O', 'P']
def BJUZ := ['B', 'J', 'U', 'Z']

-- Lean statement to assert the equivalence of the products.
theorem equivalent_product_lists :
  list_product MNOP = list_product BJUZ :=
by
  sorry

end NUMINAMATH_GPT_equivalent_product_lists_l2360_236098


namespace NUMINAMATH_GPT_similar_right_triangles_l2360_236088

open Real

theorem similar_right_triangles (x : ℝ) (h : ℝ)
  (h₁: 12^2 + 9^2 = (12^2 + 9^2))
  (similarity : (12 / x) = (9 / 6))
  (p : hypotenuse = 12*12) :
  x = 8 ∧ h = 10 := by
  sorry

end NUMINAMATH_GPT_similar_right_triangles_l2360_236088


namespace NUMINAMATH_GPT_cost_of_three_tshirts_l2360_236054

-- Defining the conditions
def saving_per_tshirt : ℝ := 5.50
def full_price_per_tshirt : ℝ := 16.50
def number_of_tshirts : ℕ := 3
def number_of_paid_tshirts : ℕ := 2

-- Statement of the problem
theorem cost_of_three_tshirts :
  (number_of_paid_tshirts * full_price_per_tshirt) = 33 := 
by
  -- Proof steps go here (using sorry as a placeholder)
  sorry

end NUMINAMATH_GPT_cost_of_three_tshirts_l2360_236054


namespace NUMINAMATH_GPT_volume_PQRS_is_48_39_cm3_l2360_236016

noncomputable def area_of_triangle (a h : ℝ) : ℝ := 0.5 * a * h

noncomputable def volume_of_tetrahedron (base_area height : ℝ) : ℝ := (1/3) * base_area * height

noncomputable def height_from_area (area base : ℝ) : ℝ := (2 * area) / base

noncomputable def volume_of_tetrahedron_PQRS : ℝ :=
  let PQ := 5
  let area_PQR := 18
  let area_PQS := 16
  let angle_PQ := 45
  let h_PQR := height_from_area area_PQR PQ
  let h_PQS := height_from_area area_PQS PQ
  let h := h_PQS * (Real.sin (angle_PQ * Real.pi / 180))
  volume_of_tetrahedron area_PQR h

theorem volume_PQRS_is_48_39_cm3 : volume_of_tetrahedron_PQRS = 48.39 := by
  sorry

end NUMINAMATH_GPT_volume_PQRS_is_48_39_cm3_l2360_236016


namespace NUMINAMATH_GPT_g_h_of_2_eq_869_l2360_236047

-- Define the functions g and h
def g (x : ℝ) : ℝ := 3 * x^2 + 2
def h (x : ℝ) : ℝ := -2 * x^3 - 1

-- State the theorem we need to prove
theorem g_h_of_2_eq_869 : g (h 2) = 869 := by
  sorry

end NUMINAMATH_GPT_g_h_of_2_eq_869_l2360_236047


namespace NUMINAMATH_GPT_length_of_equal_pieces_l2360_236007

theorem length_of_equal_pieces (total_length : ℕ) (num_pieces : ℕ) (num_unequal_pieces : ℕ) (unequal_piece_length : ℕ)
    (equal_pieces : ℕ) (equal_piece_length : ℕ) :
    total_length = 11650 ∧ num_pieces = 154 ∧ num_unequal_pieces = 4 ∧ unequal_piece_length = 100 ∧ equal_pieces = 150 →
    equal_piece_length = 75 :=
by
  sorry

end NUMINAMATH_GPT_length_of_equal_pieces_l2360_236007


namespace NUMINAMATH_GPT_number_of_possible_values_for_b_l2360_236092

theorem number_of_possible_values_for_b : 
  ∃ (n : ℕ), n = 2 ∧ ∀ b : ℕ, (b ≥ 2 ∧ b^3 ≤ 197 ∧ 197 < b^4) → b = 4 ∨ b = 5 :=
sorry

end NUMINAMATH_GPT_number_of_possible_values_for_b_l2360_236092


namespace NUMINAMATH_GPT_find_square_l2360_236063

-- Define the conditions as hypotheses
theorem find_square (p : ℕ) (sq : ℕ)
  (h1 : sq + p = 75)
  (h2 : (sq + p) + p = 142) :
  sq = 8 := by
  sorry

end NUMINAMATH_GPT_find_square_l2360_236063


namespace NUMINAMATH_GPT_largest_expression_is_D_l2360_236025

-- Define each expression
def exprA : ℤ := 3 - 1 + 4 + 6
def exprB : ℤ := 3 - 1 * 4 + 6
def exprC : ℤ := 3 - (1 + 4) * 6
def exprD : ℤ := 3 - 1 + 4 * 6
def exprE : ℤ := 3 * (1 - 4) + 6

-- The theorem stating that exprD is the largest value among the given expressions.
theorem largest_expression_is_D : 
  exprD = 26 ∧ 
  exprD > exprA ∧ 
  exprD > exprB ∧ 
  exprD > exprC ∧ 
  exprD > exprE := 
by {
  sorry
}

end NUMINAMATH_GPT_largest_expression_is_D_l2360_236025


namespace NUMINAMATH_GPT_math_problem_l2360_236080

open Real -- Open the real number namespace

theorem math_problem (a b : ℝ) (n : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 1 / a + 1 / b = 1) : 
  (a + b)^n - a^n - b^n ≥ 2^(2 * n) - 2^(n + 1) :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l2360_236080


namespace NUMINAMATH_GPT_asymptotes_of_hyperbola_eq_m_l2360_236097

theorem asymptotes_of_hyperbola_eq_m :
  ∀ (m : ℝ), (∀ (x y : ℝ), (x^2 / 16 - y^2 / 25 = 1) → (y = m * x ∨ y = -m * x)) → m = 5 / 4 :=
by 
  sorry

end NUMINAMATH_GPT_asymptotes_of_hyperbola_eq_m_l2360_236097
