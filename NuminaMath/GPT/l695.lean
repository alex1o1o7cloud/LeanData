import Mathlib

namespace perimeter_after_growth_operations_perimeter_after_four_growth_operations_l695_69543

theorem perimeter_after_growth_operations (initial_perimeter : ℝ) (growth_factor : ℝ) (growth_steps : ℕ):
  initial_perimeter = 27 ∧ growth_factor = 4/3 ∧ growth_steps = 2 → 
    initial_perimeter * growth_factor^growth_steps = 48 :=
by
  sorry

theorem perimeter_after_four_growth_operations (initial_perimeter : ℝ) (growth_factor : ℝ) (growth_steps : ℕ):
  initial_perimeter = 27 ∧ growth_factor = 4/3 ∧ growth_steps = 4 → 
    initial_perimeter * growth_factor^growth_steps = 256/3 :=
by
  sorry

end perimeter_after_growth_operations_perimeter_after_four_growth_operations_l695_69543


namespace valid_ATM_passwords_l695_69599

theorem valid_ATM_passwords : 
  let total_passwords := 10^4
  let restricted_passwords := 10
  total_passwords - restricted_passwords = 9990 :=
by
  sorry

end valid_ATM_passwords_l695_69599


namespace number_of_terms_arithmetic_sequence_l695_69520

-- Definitions for the arithmetic sequence conditions
open Nat

noncomputable def S4 := 26
noncomputable def Sn := 187
noncomputable def last4_sum (n : ℕ) (a d : ℕ) := 
  (n - 3) * a + 3 * (n - 2) * d + 3 * (n - 1) * d + n * d

-- Statement for the problem
theorem number_of_terms_arithmetic_sequence 
  (a d n : ℕ) (h1 : 4 * a + 6 * d = S4) (h2 : n * (2 * a + (n - 1) * d) / 2 = Sn) 
  (h3 : last4_sum n a d = 110) : 
  n = 11 :=
sorry

end number_of_terms_arithmetic_sequence_l695_69520


namespace product_is_2008th_power_l695_69500

theorem product_is_2008th_power (a b c : ℕ) (h1 : a = (b + c) / 2) (h2 : b ≠ c) (h3 : c ≠ a) (h4 : a ≠ b) :
  ∃ k : ℕ, (a * b * c) = k^2008 :=
by
  sorry

end product_is_2008th_power_l695_69500


namespace value_of_b_l695_69501

theorem value_of_b (f : ℝ → ℝ) (a b : ℝ) (h1 : ∀ x ≠ 0, f x = -1 / x) (h2 : f a = -1 / 3) (h3 : f (a * b) = 1 / 6) : b = -2 :=
sorry

end value_of_b_l695_69501


namespace olivia_probability_l695_69575

noncomputable def total_outcomes (n m : ℕ) : ℕ := Nat.choose n m

noncomputable def favorable_outcomes : ℕ :=
  let choose_three_colors := total_outcomes 4 3
  let choose_one_for_pair := total_outcomes 3 1
  let choose_socks :=
    (total_outcomes 3 2) * (total_outcomes 3 1) * (total_outcomes 3 1)
  choose_three_colors * choose_one_for_pair * choose_socks

def probability (n m : ℕ) : ℚ := n / m

theorem olivia_probability :
  probability favorable_outcomes (total_outcomes 12 5) = 9 / 22 :=
by
  sorry

end olivia_probability_l695_69575


namespace maximum_sin_C_in_triangle_l695_69570

theorem maximum_sin_C_in_triangle 
  (A B C : ℝ)
  (h1 : A + B + C = π) 
  (h2 : 1 / Real.tan A + 1 / Real.tan B = 6 / Real.tan C) : 
  Real.sin C = Real.sqrt 15 / 4 :=
sorry

end maximum_sin_C_in_triangle_l695_69570


namespace possible_N_l695_69509

/-- 
  Let N be an integer with N ≥ 3, and let a₀, a₁, ..., a_(N-1) be pairwise distinct reals such that 
  aᵢ ≥ a_(2i mod N) for all i. Prove that N must be a power of 2.
-/
theorem possible_N (N : ℕ) (hN : N ≥ 3) (a : Fin N → ℝ) (h_distinct: Function.Injective a) 
  (h_condition : ∀ i : Fin N, a i ≥ a (⟨(2 * i) % N, sorry⟩)) 
  : ∃ k : ℕ, N = 2^k := 
sorry

end possible_N_l695_69509


namespace money_constraints_l695_69505

variable (a b : ℝ)

theorem money_constraints (h1 : 8 * a - b = 98) (h2 : 2 * a + b > 36) : a > 13.4 ∧ b > 9.2 :=
sorry

end money_constraints_l695_69505


namespace solve_fraction_eqn_l695_69503

def fraction_eqn_solution : Prop :=
  ∃ (x : ℝ), (x + 2) / (x - 1) = 0 ∧ x ≠ 1 ∧ x = -2

theorem solve_fraction_eqn : fraction_eqn_solution :=
sorry

end solve_fraction_eqn_l695_69503


namespace find_x_pow_y_l695_69551

theorem find_x_pow_y (x y : ℝ) : |x + 2| + (y - 3)^2 = 0 → x ^ y = -8 :=
by
  sorry

end find_x_pow_y_l695_69551


namespace new_energy_vehicles_l695_69590

-- Given conditions
def conditions (a b : ℕ) : Prop :=
  3 * a + 2 * b = 95 ∧ 4 * a + 1 * b = 110

-- Given prices
def purchase_prices : Prop :=
  ∃ a b, conditions a b ∧ a = 25 ∧ b = 10

-- Total value condition for different purchasing plans
def purchase_plans (m n : ℕ) : Prop :=
  25 * m + 10 * n = 250 ∧ m > 0 ∧ n > 0

-- Number of different purchasing plans
def num_purchase_plans : Prop :=
  ∃ num_plans, num_plans = 4

-- Profit calculation for a given plan
def profit (m n : ℕ) : ℕ :=
  12 * m + 8 * n

-- Maximum profit condition
def max_profit : Prop :=
  ∃ max_profit, max_profit = 184 ∧ ∀ (m n : ℕ), purchase_plans m n → profit m n ≤ 184

-- Main theorem
theorem new_energy_vehicles : purchase_prices ∧ num_purchase_plans ∧ max_profit :=
  sorry

end new_energy_vehicles_l695_69590


namespace find_a_l695_69523

theorem find_a (S_n : ℕ → ℝ) (a_n : ℕ → ℝ) (a : ℝ) (h1 : ∀ n, S_n n = 3^(n+1) + a)
  (h2 : ∀ n, a_n (n+1) = S_n (n+1) - S_n n)
  (h3 : ∀ n m k, a_n m * a_n k = (a_n n)^2 → n = m + k) : 
  a = -3 := 
sorry

end find_a_l695_69523


namespace runner_injury_point_l695_69561

-- Define the initial setup conditions
def total_distance := 40
def second_half_time := 10
def first_half_additional_time := 5

-- Prove that given the conditions, the runner injured her foot at 20 miles.
theorem runner_injury_point : 
  ∃ (d v : ℝ), (d = 5 * v) ∧ (total_distance - d = 5 * v) ∧ (10 = second_half_time) ∧ (first_half_additional_time = 5) ∧ (d = 20) :=
by
  sorry

end runner_injury_point_l695_69561


namespace maddie_weekend_watch_time_l695_69528

-- Defining the conditions provided in the problem
def num_episodes : ℕ := 8
def duration_per_episode : ℕ := 44
def minutes_on_monday : ℕ := 138
def minutes_on_tuesday : ℕ := 0
def minutes_on_wednesday : ℕ := 0
def minutes_on_thursday : ℕ := 21
def episodes_on_friday : ℕ := 2

-- Define the total time watched from Monday to Friday
def total_minutes_week : ℕ := num_episodes * duration_per_episode
def total_minutes_mon_to_fri : ℕ := 
  minutes_on_monday + 
  minutes_on_tuesday + 
  minutes_on_wednesday + 
  minutes_on_thursday + 
  (episodes_on_friday * duration_per_episode)

-- Define the weekend watch time
def weekend_watch_time : ℕ := total_minutes_week - total_minutes_mon_to_fri

-- The theorem to prove the correct answer
theorem maddie_weekend_watch_time : weekend_watch_time = 105 := by
  sorry

end maddie_weekend_watch_time_l695_69528


namespace gnomes_telling_the_truth_l695_69586

-- Conditions
def gnome_height_claims : List ℕ := [60, 61, 62, 63, 64, 65, 66]

-- Question and Proof Problem in Lean 4
theorem gnomes_telling_the_truth :
  (∀ (actual_heights : List ℕ), 
    actual_heights.length = 7 →
    (∀ i, i < 7 → i > 0 → 
    actual_heights.get! i > actual_heights.get! (i - 1) → gnome_height_claims.get! i ≠ actual_heights.get! i)) →
  -- conclusion
  (∃ count, count = 1) :=
by
  sorry

end gnomes_telling_the_truth_l695_69586


namespace find_x_for_sin_cos_l695_69568

theorem find_x_for_sin_cos (x : ℝ) (h1 : 0 ≤ x) (h2 : x < 2 * Real.pi) (h3 : Real.sin x + Real.cos x = Real.sqrt 2) : x = Real.pi / 4 :=
sorry

end find_x_for_sin_cos_l695_69568


namespace polyhedron_value_calculation_l695_69525

noncomputable def calculate_value (P T V : ℕ) : ℕ :=
  100 * P + 10 * T + V

theorem polyhedron_value_calculation :
  ∀ (P T V E F : ℕ),
    F = 36 ∧
    T + P = 36 ∧
    E = (3 * T + 5 * P) / 2 ∧
    V = E - F + 2 →
    calculate_value P T V = 2018 :=
by
  intros P T V E F h
  sorry

end polyhedron_value_calculation_l695_69525


namespace not_enough_evidence_to_show_relationship_l695_69541

noncomputable def isEvidenceToShowRelationship (table : Array (Array Nat)) : Prop :=
  ∃ evidence : Bool, ¬evidence

theorem not_enough_evidence_to_show_relationship :
  isEvidenceToShowRelationship #[#[5, 15, 20], #[40, 10, 50], #[45, 25, 70]] :=
sorry 

end not_enough_evidence_to_show_relationship_l695_69541


namespace maximum_xy_l695_69589

theorem maximum_xy (x y : ℝ) (h : x^2 + 2 * y^2 - 2 * x * y = 4) : 
  xy ≤ 2 * (Float.sqrt 2) + 2 :=
sorry

end maximum_xy_l695_69589


namespace foil_covered_prism_width_l695_69550

def inner_prism_dimensions (l w h : ℕ) : Prop :=
  w = 2 * l ∧ w = 2 * h ∧ l * w * h = 128

def outer_prism_width (l w h outer_width : ℕ) : Prop :=
  inner_prism_dimensions l w h ∧ outer_width = w + 2

theorem foil_covered_prism_width (l w h outer_width : ℕ) (h_inner_prism : inner_prism_dimensions l w h) :
  outer_prism_width l w h outer_width → outer_width = 10 :=
by
  intro h_outer_prism
  obtain ⟨h_w_eq, h_w_eq_2, h_volume_eq⟩ := h_inner_prism
  obtain ⟨_, h_outer_width_eq⟩ := h_outer_prism
  sorry

end foil_covered_prism_width_l695_69550


namespace James_balloons_correct_l695_69585

def Amy_balloons : ℕ := 101
def diff_balloons : ℕ := 131
def James_balloons (a : ℕ) (d : ℕ) : ℕ := a + d

theorem James_balloons_correct : James_balloons Amy_balloons diff_balloons = 232 :=
by
  sorry

end James_balloons_correct_l695_69585


namespace students_not_enrolled_in_either_course_l695_69558

theorem students_not_enrolled_in_either_course 
  (total : ℕ) (french : ℕ) (german : ℕ) (both : ℕ) 
  (h_total : total = 87) (h_french : french = 41) (h_german : german = 22) (h_both : both = 9) : 
  ∃ (not_enrolled : ℕ), not_enrolled = (total - (french + german - both)) ∧ not_enrolled = 33 := by
  have h_french_or_german : ℕ := french + german - both
  have h_not_enrolled : ℕ := total - h_french_or_german
  use h_not_enrolled
  sorry

end students_not_enrolled_in_either_course_l695_69558


namespace tetrahedron_circumsphere_radius_l695_69539

theorem tetrahedron_circumsphere_radius :
  ∃ (r : ℝ), 
    (∀ (A B C P : ℝ × ℝ × ℝ),
      (dist A B = 5) ∧
      (dist A C = 5) ∧
      (dist A P = 5) ∧
      (dist B C = 5) ∧
      (dist B P = 5) ∧
      (dist C P = 6) →
      r = (20 * Real.sqrt 39) / 39) :=
sorry

end tetrahedron_circumsphere_radius_l695_69539


namespace man_age_difference_l695_69573

theorem man_age_difference (S M : ℕ) (h1 : S = 22) (h2 : M + 2 = 2 * (S + 2)) :
  M - S = 24 :=
by sorry

end man_age_difference_l695_69573


namespace machines_job_completion_time_l695_69578

theorem machines_job_completion_time (t : ℕ) 
  (hR_rate : ∀ t, 1 / t = 1 / 216) 
  (hS_rate : ∀ t, 1 / t = 1 / 216) 
  (same_num_machines : ∀ R S, R = 9 ∧ S = 9) 
  (total_time : 12 = 12) 
  (jobs_completed : 1 = (18 / t) * 12) : 
  t = 216 := 
sorry

end machines_job_completion_time_l695_69578


namespace remainder_98_pow_50_mod_100_l695_69513

/-- 
Theorem: The remainder when \(98^{50}\) is divided by 100 is 24.
-/
theorem remainder_98_pow_50_mod_100 : (98^50 % 100) = 24 := by
  sorry

end remainder_98_pow_50_mod_100_l695_69513


namespace vanya_faster_speed_l695_69592

def vanya_speed (v : ℝ) : Prop :=
  (v + 2) / v = 2.5

theorem vanya_faster_speed (v : ℝ) (h : vanya_speed v) : (v + 4) / v = 4 :=
by
  sorry

end vanya_faster_speed_l695_69592


namespace figure_perimeter_equals_26_l695_69552

noncomputable def rectangle_perimeter : ℕ := 26

def figure_arrangement (width height : ℕ) : Prop :=
width = 2 ∧ height = 1

theorem figure_perimeter_equals_26 {width height : ℕ} (h : figure_arrangement width height) :
  rectangle_perimeter = 26 :=
by
  sorry

end figure_perimeter_equals_26_l695_69552


namespace marble_problem_l695_69565

theorem marble_problem (a : ℚ) (total : ℚ) 
  (h1 : total = a + 2 * a + 6 * a + 42 * a) :
  a = 42 / 17 :=
by 
  sorry

end marble_problem_l695_69565


namespace total_income_by_nth_year_max_m_and_k_range_l695_69546

noncomputable def total_income (a : ℝ) (k : ℝ) (n : ℕ) : ℝ :=
  (6 - (n + 6) * 0.1 ^ n) * a

theorem total_income_by_nth_year (a : ℝ) (n : ℕ) :
  total_income a 0.1 n = (6 - (n + 6) * 0.1 ^ n) * a :=
sorry

theorem max_m_and_k_range (a : ℝ) (m : ℕ) :
  (m = 4 ∧ 1 ≤ 1) ∧ (∀ k, k ≥ 1 → m = 4) :=
sorry

end total_income_by_nth_year_max_m_and_k_range_l695_69546


namespace total_questions_correct_total_answers_correct_l695_69524

namespace ForumCalculation

def members : ℕ := 200
def questions_per_hour_per_user : ℕ := 3
def hours_in_day : ℕ := 24
def answers_multiplier : ℕ := 3

def total_questions_per_user_per_day : ℕ :=
  questions_per_hour_per_user * hours_in_day

def total_questions_in_a_day : ℕ :=
  members * total_questions_per_user_per_day

def total_answers_per_user_per_day : ℕ :=
  answers_multiplier * total_questions_per_user_per_day

def total_answers_in_a_day : ℕ :=
  members * total_answers_per_user_per_day

theorem total_questions_correct :
  total_questions_in_a_day = 14400 :=
by
  sorry

theorem total_answers_correct :
  total_answers_in_a_day = 43200 :=
by
  sorry

end ForumCalculation

end total_questions_correct_total_answers_correct_l695_69524


namespace solution_l695_69549

noncomputable def determine_numbers (x y : ℚ) : Prop :=
  x^2 + y^2 = 45 / 4 ∧ x - y = x * y

theorem solution (x y : ℚ) :
  determine_numbers x y → (x = -3 ∧ y = 3/2) ∨ (x = -3/2 ∧ y = 3) :=
-- We state the main theorem that relates the determine_numbers predicate to the specific pairs of numbers
sorry

end solution_l695_69549


namespace coin_difference_l695_69582

noncomputable def max_value (p n d : ℕ) : ℕ := p + 5 * n + 10 * d
noncomputable def min_value (p n d : ℕ) : ℕ := p + 5 * n + 10 * d

theorem coin_difference (p n d : ℕ) (h₁ : p + n + d = 3030) (h₂ : 10 ≤ p) (h₃ : 10 ≤ n) (h₄ : 10 ≤ d) :
  max_value 10 10 3010 - min_value 3010 10 10 = 27000 := by
  sorry

end coin_difference_l695_69582


namespace roots_quadratic_expression_l695_69579

theorem roots_quadratic_expression :
  ∀ (a b : ℝ), (a^2 - 5 * a + 6 = 0) ∧ (b^2 - 5 * b + 6 = 0) → 
  a^3 + a^4 * b^2 + a^2 * b^4 + b^3 + a * b * (a + b) = 533 :=
by
  intros a b h
  sorry

end roots_quadratic_expression_l695_69579


namespace perfect_squares_diff_consecutive_l695_69557

theorem perfect_squares_diff_consecutive (h1 : ∀ a : ℕ, a^2 < 1000000 → ∃ b : ℕ, a^2 = (b + 1)^2 - b^2) : 
  (∃ n : ℕ, n = 500) := 
by 
  sorry

end perfect_squares_diff_consecutive_l695_69557


namespace integer_exponentiation_l695_69564

theorem integer_exponentiation
  (a b x y : ℕ)
  (h_gcd : a.gcd b = 1)
  (h_pos_a : 1 < a)
  (h_pos_b : 1 < b)
  (h_pos_x : 1 < x)
  (h_pos_y : 1 < y)
  (h_eq : x^a = y^b) :
  ∃ n : ℕ, 1 < n ∧ x = n^b ∧ y = n^a :=
by sorry

end integer_exponentiation_l695_69564


namespace problem_solution_l695_69506

theorem problem_solution (m n : ℕ) (h1 : m + 7 < n + 3) 
  (h2 : (m + (m+3) + (m+7) + (n+3) + (n+6) + 2 * n) / 6 = n + 3) 
  (h3 : (m + 7 + n + 3) / 2 = n + 3) : m + n = 12 := 
  sorry

end problem_solution_l695_69506


namespace area_trapezoid_def_l695_69530

noncomputable def area_trapezoid (a : ℝ) (h : a ≠ 0) : ℝ :=
  let b := 108 / a
  let DE := a / 2
  let FG := b / 3
  let height := b / 2
  (DE + FG) * height / 2

theorem area_trapezoid_def (a : ℝ) (h : a ≠ 0) :
  area_trapezoid a h = 18 + 18 / a :=
by
  sorry

end area_trapezoid_def_l695_69530


namespace remainder_9053_div_98_l695_69534

theorem remainder_9053_div_98 : 9053 % 98 = 37 :=
by sorry

end remainder_9053_div_98_l695_69534


namespace count_divisible_neither_5_nor_7_below_500_l695_69574

def count_divisible_by (n k : ℕ) : ℕ := (n - 1) / k

def count_divisible_by_5_or_7_below (n : ℕ) : ℕ :=
  let count_5 := count_divisible_by n 5
  let count_7 := count_divisible_by n 7
  let count_35 := count_divisible_by n 35
  count_5 + count_7 - count_35

def count_divisible_neither_5_nor_7_below (n : ℕ) : ℕ :=
  n - 1 - count_divisible_by_5_or_7_below n

theorem count_divisible_neither_5_nor_7_below_500 : count_divisible_neither_5_nor_7_below 500 = 343 :=
by
  sorry

end count_divisible_neither_5_nor_7_below_500_l695_69574


namespace decomposition_of_5_to_4_eq_125_l695_69559

theorem decomposition_of_5_to_4_eq_125 :
  (∃ a b c : ℕ, (5^4 = a + b + c) ∧ 
                (a = 121) ∧ 
                (b = 123) ∧ 
                (c = 125)) := by 
sorry

end decomposition_of_5_to_4_eq_125_l695_69559


namespace solution_volume_l695_69545

theorem solution_volume (x : ℝ) (h1 : (0.16 * x) / (x + 13) = 0.0733333333333333) : x = 11 :=
by sorry

end solution_volume_l695_69545


namespace parallelogram_area_l695_69548

open Matrix

noncomputable def u : Fin 2 → ℝ := ![7, -4]
noncomputable def z : Fin 2 → ℝ := ![8, -1]

theorem parallelogram_area :
  let matrix := ![u, z]
  |det (of fun (i j : Fin 2) => (matrix i) j)| = 25 :=
by
  sorry

end parallelogram_area_l695_69548


namespace a_gt_b_l695_69556

noncomputable def a (R : Type*) [OrderedRing R] := {x : R // 0 < x ∧ x ^ 3 = x + 1}
noncomputable def b (R : Type*) [OrderedRing R] (a : R) := {y : R // 0 < y ∧ y ^ 6 = y + 3 * a}

theorem a_gt_b (R : Type*) [OrderedRing R] (a_pos_real : a R) (b_pos_real : b R (a_pos_real.val)) : a_pos_real.val > b_pos_real.val :=
sorry

end a_gt_b_l695_69556


namespace leaves_fall_total_l695_69504

theorem leaves_fall_total : 
  let planned_cherry_trees := 7 
  let planned_maple_trees := 5 
  let actual_cherry_trees := 2 * planned_cherry_trees
  let actual_maple_trees := 3 * planned_maple_trees
  let leaves_per_cherry_tree := 100
  let leaves_per_maple_tree := 150
  actual_cherry_trees * leaves_per_cherry_tree + actual_maple_trees * leaves_per_maple_tree = 3650 :=
by
  let planned_cherry_trees := 7 
  let planned_maple_trees := 5 
  let actual_cherry_trees := 2 * planned_cherry_trees
  let actual_maple_trees := 3 * planned_maple_trees
  let leaves_per_cherry_tree := 100
  let leaves_per_maple_tree := 150
  sorry

end leaves_fall_total_l695_69504


namespace questionnaires_drawn_from_unit_D_l695_69553

theorem questionnaires_drawn_from_unit_D 
  (arith_seq_collected : ∃ a1 d : ℕ, [a1, a1 + d, a1 + 2 * d, a1 + 3 * d] = [aA, aB, aC, aD] ∧ aA + aB + aC + aD = 1000)
  (stratified_sample : [30 - d, 30, 30 + d, 30 + 2 * d] = [sA, sB, sC, sD] ∧ sA + sB + sC + sD = 150)
  (B_drawn : 30 = sB) :
  sD = 60 := 
by {
  sorry
}

end questionnaires_drawn_from_unit_D_l695_69553


namespace probability_of_defective_on_second_draw_l695_69567

-- Define the conditions
variable (batch_size : ℕ) (defective_items : ℕ) (good_items : ℕ)
variable (first_draw_good : Prop)
variable (without_replacement : Prop)

-- Given conditions
def batch_conditions : Prop :=
  batch_size = 10 ∧ defective_items = 3 ∧ good_items = 7 ∧ first_draw_good ∧ without_replacement

-- The desired probability as a proof
theorem probability_of_defective_on_second_draw
  (h : batch_conditions batch_size defective_items good_items first_draw_good without_replacement) : 
  (3 / 9 : ℝ) = 1 / 3 :=
sorry

end probability_of_defective_on_second_draw_l695_69567


namespace diagonal_length_not_possible_l695_69544

-- Define the side lengths of the parallelogram
def sides_of_parallelogram : ℕ × ℕ := (6, 8)

-- Define the length of a diagonal that cannot exist
def invalid_diagonal_length : ℕ := 15

-- Statement: Prove that a diagonal of length 15 cannot exist for such a parallelogram.
theorem diagonal_length_not_possible (a b d : ℕ) 
  (h₁ : sides_of_parallelogram = (a, b)) 
  (h₂ : d = invalid_diagonal_length) 
  : d ≥ a + b := 
sorry

end diagonal_length_not_possible_l695_69544


namespace exists_polynomial_p_l695_69517

theorem exists_polynomial_p (x : ℝ) (h : x ∈ Set.Icc (1 / 10 : ℝ) (9 / 10 : ℝ)) :
  ∃ (P : ℝ → ℝ), (∀ (k : ℤ), P k = P k) ∧ (∀ (x : ℝ), x ∈ Set.Icc (1 / 10 : ℝ) (9 / 10 : ℝ) → 
  abs (P x - 1 / 2) < 1 / 1000) :=
by
  sorry

end exists_polynomial_p_l695_69517


namespace number_of_children_l695_69518

-- Definitions for the conditions
def adult_ticket_cost : ℕ := 8
def child_ticket_cost : ℕ := 3
def total_amount : ℕ := 35

-- Theorem stating the proof problem
theorem number_of_children (A C T : ℕ) (hc: A = adult_ticket_cost) (ha: C = child_ticket_cost) (ht: T = total_amount) :
  (T - A) / C = 9 :=
by
  sorry

end number_of_children_l695_69518


namespace unoccupied_seats_l695_69519

theorem unoccupied_seats (rows chairs_per_row seats_taken : Nat) (h1 : rows = 40)
  (h2 : chairs_per_row = 20) (h3 : seats_taken = 790) :
  rows * chairs_per_row - seats_taken = 10 :=
by
  sorry

end unoccupied_seats_l695_69519


namespace max_black_cells_in_101x101_grid_l695_69521

theorem max_black_cells_in_101x101_grid :
  ∀ (k : ℕ), k ≤ 101 → 2 * k * (101 - k) ≤ 5100 :=
by
  sorry

end max_black_cells_in_101x101_grid_l695_69521


namespace sock_pairs_l695_69576

open Nat

theorem sock_pairs (r g y : ℕ) (hr : r = 5) (hg : g = 6) (hy : y = 4) :
  (choose r 2) + (choose g 2) + (choose y 2) = 31 :=
by
  rw [hr, hg, hy]
  norm_num
  sorry

end sock_pairs_l695_69576


namespace smallest_d_for_inverse_l695_69580

def g (x : ℝ) : ℝ := (x - 3) ^ 2 - 1

theorem smallest_d_for_inverse :
  ∃ d : ℝ, (∀ x1 x2 : ℝ, x1 ≠ x2 → (d ≤ x1) → (d ≤ x2) → g x1 ≠ g x2) ∧ d = 3 :=
by
  sorry

end smallest_d_for_inverse_l695_69580


namespace tie_rate_correct_l695_69588

-- Define the fractions indicating win rates for Amy, Lily, and John
def AmyWinRate : ℚ := 4 / 9
def LilyWinRate : ℚ := 1 / 3
def JohnWinRate : ℚ := 1 / 6

-- Define the fraction they tie
def TieRate : ℚ := 1 / 18

-- The theorem for proving the tie rate
theorem tie_rate_correct : AmyWinRate + LilyWinRate + JohnWinRate = 17 / 18 → (1 : ℚ) - (17 / 18) = TieRate :=
by
  sorry -- Proof is omitted

-- Define the win rate sums and tie rate equivalence
example : (AmyWinRate + LilyWinRate + JohnWinRate = 17 / 18) ∧ (TieRate = 1 - 17 / 18) :=
by
  sorry -- Proof is omitted

end tie_rate_correct_l695_69588


namespace cats_on_edges_l695_69563

variables {W1 W2 B1 B2 : ℕ}  -- representing positions of cats on a line

def distance_from_white_to_black_sum_1 (a1 a2 : ℕ) : Prop := a1 + a2 = 4
def distance_from_white_to_black_sum_2 (b1 b2 : ℕ) : Prop := b1 + b2 = 8
def distance_from_black_to_white_sum_1 (b1 a1 : ℕ) : Prop := b1 + a1 = 9
def distance_from_black_to_white_sum_2 (b2 a2 : ℕ) : Prop := b2 + a2 = 3

theorem cats_on_edges
  (a1 a2 b1 b2 : ℕ)
  (h1 : distance_from_white_to_black_sum_1 a1 a2)
  (h2 : distance_from_white_to_black_sum_2 b1 b2)
  (h3 : distance_from_black_to_white_sum_1 b1 a1)
  (h4 : distance_from_black_to_white_sum_2 b2 a2) :
  (a1 = 2) ∧ (a2 = 2) ∧ (b1 = 7) ∧ (b2 = 1) ∧ (W1 = min W1 W2) ∧ (B2 = max B1 B2) :=
sorry

end cats_on_edges_l695_69563


namespace cubic_polynomial_solution_l695_69511

theorem cubic_polynomial_solution (x : ℝ) :
  x^3 + 6*x^2 + 11*x + 6 = 12 ↔ x = -1 ∨ x = -2 ∨ x = -3 := by
  sorry

end cubic_polynomial_solution_l695_69511


namespace probability_of_purple_l695_69595

def total_faces := 10
def purple_faces := 3

theorem probability_of_purple : (purple_faces : ℚ) / (total_faces : ℚ) = 3 / 10 := 
by 
  sorry

end probability_of_purple_l695_69595


namespace modular_inverse_example_l695_69593

open Int

theorem modular_inverse_example :
  ∃ b : ℤ, 0 ≤ b ∧ b < 120 ∧ (7 * b) % 120 = 1 ∧ b = 103 :=
by
  sorry

end modular_inverse_example_l695_69593


namespace solve_for_x_l695_69510

theorem solve_for_x : ∃ x : ℝ, (x + 36) / 3 = (7 - 2 * x) / 6 ∧ x = -65 / 4 := by
  sorry

end solve_for_x_l695_69510


namespace math_problem_l695_69583

variable (x y : ℚ)

theorem math_problem (h : 1.5 * x = 0.04 * y) : (y - x) / (y + x) = 73 / 77 := by
  sorry

end math_problem_l695_69583


namespace problem1_solution_problem2_solution_l695_69577

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 4|
def g (x : ℝ) : ℝ := |2 * x + 1|

-- Problem 1
theorem problem1_solution :
  {x : ℝ | f x < g x} = {x : ℝ | x < -5 ∨ x > 1} :=
sorry

-- Problem 2
theorem problem2_solution :
  ∀ (a : ℝ), (∀ x : ℝ, 2 * f x + g x > a * x) ↔ -4 ≤ a ∧ a < 9 / 4 :=
sorry

end problem1_solution_problem2_solution_l695_69577


namespace earbuds_cost_before_tax_l695_69531

-- Define the conditions
variable (C : ℝ) -- The cost before tax
variable (taxRate : ℝ := 0.15)
variable (totalPaid : ℝ := 230)

-- Define the main question in Lean
theorem earbuds_cost_before_tax : C + taxRate * C = totalPaid → C = 200 :=
by
  sorry

end earbuds_cost_before_tax_l695_69531


namespace cos2_plus_sin2_given_tan_l695_69542

noncomputable def problem_cos2_plus_sin2_given_tan : Prop :=
  ∀ (α : ℝ), Real.tan α = 2 → Real.cos α ^ 2 + Real.sin (2 * α) = 1

-- Proof is omitted
theorem cos2_plus_sin2_given_tan : problem_cos2_plus_sin2_given_tan := sorry

end cos2_plus_sin2_given_tan_l695_69542


namespace max_value_expression_l695_69554

theorem max_value_expression (p : ℝ) (q : ℝ) (h : q = p - 2) :
  ∃ M : ℝ, M = -70 + 96.66666666666667 ∧ (∀ p : ℝ, -3 * p^2 + 24 * p - 50 + 10 * q ≤ M) :=
sorry

end max_value_expression_l695_69554


namespace solve_quadratic_l695_69591

theorem solve_quadratic {x : ℝ} (h : 2 * (x - 1)^2 = x - 1) : x = 1 ∨ x = 3 / 2 :=
sorry

end solve_quadratic_l695_69591


namespace soda_relationship_l695_69581

theorem soda_relationship (J : ℝ) (L : ℝ) (A : ℝ) (hL : L = 1.75 * J) (hA : A = 1.20 * J) : 
  (L - A) / A = 0.46 := 
by
  sorry

end soda_relationship_l695_69581


namespace gcd_three_numbers_l695_69527

theorem gcd_three_numbers :
  gcd (gcd 324 243) 135 = 27 :=
by
  sorry

end gcd_three_numbers_l695_69527


namespace find_lesser_fraction_l695_69532

theorem find_lesser_fraction (x y : ℚ) (h₁ : x + y = 3 / 4) (h₂ : x * y = 1 / 8) : min x y = 1 / 4 := 
by 
  sorry

end find_lesser_fraction_l695_69532


namespace exists_X_Y_sum_not_in_third_subset_l695_69535

open Nat Set

theorem exists_X_Y_sum_not_in_third_subset :
  ∀ (M_1 M_2 M_3 : Set ℕ), 
  Disjoint M_1 M_2 ∧ Disjoint M_2 M_3 ∧ Disjoint M_1 M_3 → 
  ∃ (X Y : ℕ), (X ∈ M_1 ∪ M_2 ∪ M_3) ∧ (Y ∈ M_1 ∪ M_2 ∪ M_3) ∧  
  (X ∈ M_1 → Y ∈ M_2 ∨ Y ∈ M_3) ∧
  (X ∈ M_2 → Y ∈ M_1 ∨ Y ∈ M_3) ∧
  (X ∈ M_3 → Y ∈ M_1 ∨ Y ∈ M_2) ∧
  (X + Y ∉ M_3) :=
by
  intros M_1 M_2 M_3 disj
  sorry

end exists_X_Y_sum_not_in_third_subset_l695_69535


namespace range_of_a_l695_69536

theorem range_of_a (a : ℝ) : (∀ x : ℕ, 4 * x + a ≤ 5 → x ≥ 1 → x ≤ 3) ↔ (-11 < a ∧ a ≤ -7) :=
by sorry

end range_of_a_l695_69536


namespace johns_total_cost_after_discount_l695_69502

/-- Price of different utensils for John's purchase --/
def forks_cost : ℕ := 25
def knives_cost : ℕ := 30
def spoons_cost : ℕ := 20
def dinner_plate_cost (silverware_cost : ℕ) : ℚ := 0.5 * silverware_cost

/-- Calculating the total cost of silverware --/
def total_silverware_cost : ℕ := forks_cost + knives_cost + spoons_cost

/-- Calculating the total cost before discount --/
def total_cost_before_discount : ℚ := total_silverware_cost + dinner_plate_cost total_silverware_cost

/-- Discount rate --/
def discount_rate : ℚ := 0.10

/-- Discount amount --/
def discount_amount (total_cost : ℚ) : ℚ := discount_rate * total_cost

/-- Total cost after applying discount --/
def total_cost_after_discount : ℚ := total_cost_before_discount - discount_amount total_cost_before_discount

/-- John's total cost after the discount should be $101.25 --/
theorem johns_total_cost_after_discount : total_cost_after_discount = 101.25 := by
  sorry

end johns_total_cost_after_discount_l695_69502


namespace hexagon_points_fourth_layer_l695_69560

theorem hexagon_points_fourth_layer :
  ∃ (h : ℕ → ℕ), h 1 = 1 ∧ (∀ n ≥ 2, h n = h (n - 1) + 6 * (n - 1)) ∧ h 4 = 37 :=
by
  sorry

end hexagon_points_fourth_layer_l695_69560


namespace no_solution_m_l695_69597

theorem no_solution_m {
  m : ℚ
  } (h : ∀ x : ℚ, x ≠ 3 → (3 - 2 * x) / (x - 3) - (m * x - 2) / (3 - x) ≠ -1) : 
  m = 1 ∨ m = 5 / 3 :=
sorry

end no_solution_m_l695_69597


namespace slices_with_both_onions_and_olives_l695_69537

noncomputable def slicesWithBothToppings (total_slices slices_with_onions slices_with_olives : Nat) : Nat :=
  slices_with_onions + slices_with_olives - total_slices

theorem slices_with_both_onions_and_olives 
  (total_slices : Nat) (slices_with_onions : Nat) (slices_with_olives : Nat) :
  total_slices = 18 ∧ slices_with_onions = 10 ∧ slices_with_olives = 10 →
  slicesWithBothToppings total_slices slices_with_onions slices_with_olives = 2 :=
by
  sorry

end slices_with_both_onions_and_olives_l695_69537


namespace intersection_M_N_is_valid_l695_69515

-- Define the conditions given in the problem
def M := {x : ℝ |  3 / 4 < x ∧ x ≤ 1}
def N := {y : ℝ | 0 ≤ y}

-- State the theorem that needs to be proved
theorem intersection_M_N_is_valid : M ∩ N = {x : ℝ | 3 / 4 < x ∧ x ≤ 1} :=
by 
  sorry

end intersection_M_N_is_valid_l695_69515


namespace min_value_of_polynomial_l695_69533

theorem min_value_of_polynomial :
  ∃ x : ℝ, ∀ y, y = (x - 16) * (x - 14) * (x + 14) * (x + 16) → y ≥ -900 :=
by
  sorry

end min_value_of_polynomial_l695_69533


namespace nonnegative_diff_roots_eq_8sqrt2_l695_69508

noncomputable def roots_diff (a b c : ℝ) : ℝ :=
  if h : b^2 - 4*a*c ≥ 0 then 
    let root1 := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
    let root2 := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
    abs (root1 - root2)
  else 
    0

theorem nonnegative_diff_roots_eq_8sqrt2 : 
  roots_diff 1 42 409 = 8 * Real.sqrt 2 :=
sorry

end nonnegative_diff_roots_eq_8sqrt2_l695_69508


namespace pentagon_rectangle_ratio_l695_69596

theorem pentagon_rectangle_ratio (p w : ℝ) 
    (pentagon_perimeter : 5 * p = 30) 
    (rectangle_perimeter : ∃ l, 2 * w + 2 * l = 30 ∧ l = 2 * w) : 
    p / w = 6 / 5 := 
by
  sorry

end pentagon_rectangle_ratio_l695_69596


namespace S₉_eq_81_l695_69529

variable (aₙ : ℕ → ℕ) (S : ℕ → ℕ)
variable (n : ℕ)
variable (a₁ d : ℕ)

-- Conditions
axiom S₃_eq_9 : S 3 = 9
axiom S₆_eq_36 : S 6 = 36
axiom S_n_def : ∀ n, S n = n * a₁ + n * (n - 1) / 2 * d

-- Proof obligation
theorem S₉_eq_81 : S 9 = 81 :=
by
  sorry

end S₉_eq_81_l695_69529


namespace digit_relationship_l695_69507

theorem digit_relationship (d1 d2 : ℕ) (h1 : d1 * 10 + d2 = 16) (h2 : d1 + d2 = 7) : d2 = 6 * d1 :=
by
  sorry

end digit_relationship_l695_69507


namespace production_steps_use_process_flowchart_l695_69526

def describe_production_steps (task : String) : Prop :=
  task = "describe production steps of a certain product in a factory"

def correct_diagram (diagram : String) : Prop :=
  diagram = "Process Flowchart"

theorem production_steps_use_process_flowchart (task : String) (diagram : String) :
  describe_production_steps task → correct_diagram diagram :=
sorry

end production_steps_use_process_flowchart_l695_69526


namespace moses_more_than_esther_l695_69538

theorem moses_more_than_esther (total_amount: ℝ) (moses_share: ℝ) (tony_esther_share: ℝ) :
  total_amount = 50 → moses_share = 0.40 * total_amount → 
  tony_esther_share = (total_amount - moses_share) / 2 → 
  moses_share - tony_esther_share = 5 :=
by
  intros h1 h2 h3
  sorry

end moses_more_than_esther_l695_69538


namespace laser_beam_total_distance_l695_69571

theorem laser_beam_total_distance :
  let A := (4, 7)
  let B := (-4, 7)
  let C := (-4, -7)
  let D := (4, -7)
  let E := (9, 7)
  let dist (p1 p2 : (ℤ × ℤ)) : ℝ := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  dist A B + dist B C + dist C D + dist D E = 30 + Real.sqrt 221 :=
by
  sorry

end laser_beam_total_distance_l695_69571


namespace integer_values_sides_triangle_l695_69540

theorem integer_values_sides_triangle (x : ℝ) (hx_pos : x > 0) (hx1 : x + 15 > 40) (hx2 : x + 40 > 15) (hx3 : 15 + 40 > x) : 
    (∃ (n : ℤ), ∃ (hn : 0 < n) (hn1 : (n : ℝ) = x) (hn2 : 26 ≤ n) (hn3 : n ≤ 54), 
    ∀ (y : ℤ), (26 ≤ y ∧ y ≤ 54) → (∃ (m : ℤ), y = 26 + m ∧ m < 29 ∧ m ≥ 0)) := 
sorry

end integer_values_sides_triangle_l695_69540


namespace sasha_made_50_muffins_l695_69555

/-- 
Sasha made some chocolate muffins for her school bake sale fundraiser. Melissa made 4 times as many 
muffins as Sasha, and Tiffany made half of Sasha and Melissa's total number of muffins. They 
contributed $900 to the fundraiser by selling muffins at $4 each. Prove that Sasha made 50 muffins.
-/
theorem sasha_made_50_muffins 
  (S : ℕ)
  (Melissa_made : ℕ := 4 * S)
  (Tiffany_made : ℕ := (1 / 2) * (S + Melissa_made))
  (Total_muffins : ℕ := S + Melissa_made + Tiffany_made)
  (total_income : ℕ := 900)
  (price_per_muffin : ℕ := 4)
  (muffins_sold : ℕ := total_income / price_per_muffin)
  (eq_muffins_sold : Total_muffins = muffins_sold) : 
  S = 50 := 
by sorry

end sasha_made_50_muffins_l695_69555


namespace solve_equation_l695_69516

theorem solve_equation (x : ℝ) (h : x ≠ 1) :
  (3 * x + 6) / (x^2 + 5 * x - 6) = (3 - x) / (x - 1) → x = -4 :=
by
  intro hyp
  sorry

end solve_equation_l695_69516


namespace robins_hair_length_l695_69566

-- Conditions:
-- Robin cut off 4 inches of his hair.
-- After cutting, his hair is now 13 inches long.
-- Question: How long was Robin's hair before he cut it? Answer: 17 inches

theorem robins_hair_length (current_length : ℕ) (cut_length : ℕ) (initial_length : ℕ) 
  (h_cut_length : cut_length = 4) 
  (h_current_length : current_length = 13) 
  (h_initial : initial_length = current_length + cut_length) :
  initial_length = 17 :=
sorry

end robins_hair_length_l695_69566


namespace common_ratio_of_geometric_series_l695_69547

theorem common_ratio_of_geometric_series (a S r : ℝ) (h₁ : a = 400) (h₂ : S = 2500) :
  S = a / (1 - r) → r = 21 / 25 :=
by
  intros h₃
  rw [h₁, h₂] at h₃
  sorry

end common_ratio_of_geometric_series_l695_69547


namespace factor_poly_l695_69598

theorem factor_poly (a b : ℤ) (h : 3*(y^2) - y - 24 = (3*y + a)*(y + b)) : a - b = 11 :=
sorry

end factor_poly_l695_69598


namespace min_value_reciprocal_sum_l695_69594

theorem min_value_reciprocal_sum 
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h_sum : a + b = 2) : 
  ∃ x, x = 2 ∧ (∀ y, y = (1 / a) + (1 / b) → x ≤ y) := 
sorry

end min_value_reciprocal_sum_l695_69594


namespace avg_abc_l695_69572

variable (A B C : ℕ)

-- Conditions
def avg_ac : Prop := (A + C) / 2 = 29
def age_b : Prop := B = 26

-- Theorem stating the average age of a, b, and c
theorem avg_abc (h1 : avg_ac A C) (h2 : age_b B) : (A + B + C) / 3 = 28 := by
  sorry

end avg_abc_l695_69572


namespace tax_rate_as_percent_l695_69587

def TaxAmount (amount : ℝ) : Prop := amount = 82
def BaseAmount (amount : ℝ) : Prop := amount = 100

theorem tax_rate_as_percent {tax_amt base_amt : ℝ} 
  (h_tax : TaxAmount tax_amt) (h_base : BaseAmount base_amt) : 
  (tax_amt / base_amt) * 100 = 82 := 
by 
  sorry

end tax_rate_as_percent_l695_69587


namespace tidy_up_time_l695_69562

theorem tidy_up_time (A B C : ℕ) (tidyA : A = 5 * 3600) (tidyB : B = 5 * 60) (tidyC : C = 5) :
  B < A ∧ B > C :=
by
  sorry

end tidy_up_time_l695_69562


namespace ratio_of_rectangle_sides_l695_69514

theorem ratio_of_rectangle_sides (x y : ℝ) (h : x < y) 
  (hs : x + y - Real.sqrt (x^2 + y^2) = (1 / 3) * y) : 
  x / y = 5 / 12 :=
by
  sorry

end ratio_of_rectangle_sides_l695_69514


namespace k_value_l695_69569

noncomputable def find_k : ℚ := 49 / 15

theorem k_value :
  ∀ (a b : ℚ), (3 * a^2 + 7 * a + find_k = 0) ∧ (3 * b^2 + 7 * b + find_k = 0) →
                (a^2 + b^2 = 3 * a * b) →
                find_k = 49 / 15 :=
by
  intros a b h_eq_root h_rel
  sorry

end k_value_l695_69569


namespace regular_polygon_sides_l695_69512

theorem regular_polygon_sides (n : ℕ) (h : ∀ (polygon : ℕ), (polygon = 160) → 2 < polygon ∧ (180 * (polygon - 2) / polygon) = 160) : n = 18 := 
sorry

end regular_polygon_sides_l695_69512


namespace farmer_eggs_per_week_l695_69522

theorem farmer_eggs_per_week (E : ℝ) (chickens : ℝ) (price_per_dozen : ℝ) (total_revenue : ℝ) (num_weeks : ℝ) (total_chickens : ℝ) (dozen : ℝ) 
    (H1 : total_chickens = 46)
    (H2 : price_per_dozen = 3)
    (H3 : total_revenue = 552)
    (H4 : num_weeks = 8)
    (H5 : dozen = 12)
    (H6 : chickens = 46)
    : E = 6 :=
by
  sorry

end farmer_eggs_per_week_l695_69522


namespace perfect_play_winner_l695_69584

theorem perfect_play_winner (A B : ℕ) :
    (A = B → (∃ f : ℕ → ℕ, ∀ n, 0 < f n ∧ f n ≤ B ∧ f n = B - A → false)) ∧
    (A ≠ B → (∃ g : ℕ → ℕ, ∀ n, 0 < g n ∧ g n ≤ B ∧ g n = A - B → false)) :=
sorry

end perfect_play_winner_l695_69584
