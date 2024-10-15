import Mathlib

namespace NUMINAMATH_GPT_overall_labor_costs_l1569_156942

noncomputable def construction_worker_daily_wage : ℝ := 100
noncomputable def electrician_daily_wage : ℝ := 2 * construction_worker_daily_wage
noncomputable def plumber_daily_wage : ℝ := 2.5 * construction_worker_daily_wage

noncomputable def total_construction_work : ℝ := 2 * construction_worker_daily_wage
noncomputable def total_electrician_work : ℝ := electrician_daily_wage
noncomputable def total_plumber_work : ℝ := plumber_daily_wage

theorem overall_labor_costs :
  total_construction_work + total_electrician_work + total_plumber_work = 650 :=
by
  sorry

end NUMINAMATH_GPT_overall_labor_costs_l1569_156942


namespace NUMINAMATH_GPT_ratio_equivalence_l1569_156951

theorem ratio_equivalence (x : ℝ) (h : 3 / x = 3 / 16) : x = 16 := 
by
  sorry

end NUMINAMATH_GPT_ratio_equivalence_l1569_156951


namespace NUMINAMATH_GPT_total_number_of_notes_l1569_156992

theorem total_number_of_notes (x : ℕ) (h₁ : 37 * 50 + x * 500 = 10350) : 37 + x = 54 :=
by
  -- We state that the total value of 37 Rs. 50 notes plus x Rs. 500 notes equals Rs. 10350.
  -- According to this information, we prove that the total number of notes is 54.
  sorry

end NUMINAMATH_GPT_total_number_of_notes_l1569_156992


namespace NUMINAMATH_GPT_fraction_zero_l1569_156924

theorem fraction_zero (x : ℝ) (h : (x^2 - 1) / (x + 1) = 0) : x = 1 := 
sorry

end NUMINAMATH_GPT_fraction_zero_l1569_156924


namespace NUMINAMATH_GPT_cubic_foot_to_cubic_inches_l1569_156965

theorem cubic_foot_to_cubic_inches (foot_to_inch : 1 = 12) : 12 ^ 3 = 1728 :=
by
  have h1 : 1^3 = 1 := by norm_num
  have h2 : (12^3) = 1728 := by norm_num
  rw [foot_to_inch] at h1
  exact h2

end NUMINAMATH_GPT_cubic_foot_to_cubic_inches_l1569_156965


namespace NUMINAMATH_GPT_roma_can_ensure_no_more_than_50_chips_end_up_in_last_cells_l1569_156911

theorem roma_can_ensure_no_more_than_50_chips_end_up_in_last_cells 
  (k n : ℕ) (h_k : k = 4) (h_n : n = 100)
  (shift_rule : ∀ (m : ℕ), m ≤ n → 
    ∃ (chips_moved : ℕ), chips_moved = 1 ∧ chips_moved ≤ m) 
  : ∃ m, m ≤ n ∧ m = 50 := 
by
  sorry

end NUMINAMATH_GPT_roma_can_ensure_no_more_than_50_chips_end_up_in_last_cells_l1569_156911


namespace NUMINAMATH_GPT_neg_ten_plus_three_l1569_156948

theorem neg_ten_plus_three :
  -10 + 3 = -7 := by
  sorry

end NUMINAMATH_GPT_neg_ten_plus_three_l1569_156948


namespace NUMINAMATH_GPT_a_2016_value_l1569_156974

theorem a_2016_value (a : ℕ → ℤ) (h1 : a 1 = 3) (h2 : a 2 = 6) 
  (rec : ∀ n, a (n + 2) = a (n + 1) - a n) : a 2016 = -3 :=
sorry

end NUMINAMATH_GPT_a_2016_value_l1569_156974


namespace NUMINAMATH_GPT_simplify_sqrt_expression_l1569_156996

theorem simplify_sqrt_expression :
  2 * Real.sqrt 12 - Real.sqrt 27 - (Real.sqrt 3 * Real.sqrt (1 / 9)) = (2 * Real.sqrt 3) / 3 := 
by
  sorry

end NUMINAMATH_GPT_simplify_sqrt_expression_l1569_156996


namespace NUMINAMATH_GPT_number_of_elements_in_M_l1569_156975

theorem number_of_elements_in_M :
  (∃! (M : Finset ℕ), M = {m | ∃ (n : ℕ), n > 0 ∧ m = 2*n - 1 ∧ m < 60 } ∧ M.card = 30) :=
sorry

end NUMINAMATH_GPT_number_of_elements_in_M_l1569_156975


namespace NUMINAMATH_GPT_students_calculation_l1569_156971

def number_of_stars : ℝ := 3.0
def students_per_star : ℝ := 41.33333333
def total_students : ℝ := 124

theorem students_calculation : number_of_stars * students_per_star = total_students := 
by
  sorry

end NUMINAMATH_GPT_students_calculation_l1569_156971


namespace NUMINAMATH_GPT_fractional_equation_no_solution_l1569_156941

theorem fractional_equation_no_solution (x : ℝ) (h1 : x ≠ 3) : (2 - x) / (x - 3) ≠ 1 + 1 / (3 - x) :=
by
  sorry

end NUMINAMATH_GPT_fractional_equation_no_solution_l1569_156941


namespace NUMINAMATH_GPT_first_valve_fill_time_l1569_156983

theorem first_valve_fill_time (V1 V2: ℕ) (capacity: ℕ) (t_combined t1: ℕ) 
  (h1: t_combined = 48)
  (h2: V2 = V1 + 50)
  (h3: capacity = 12000)
  (h4: V1 + V2 = capacity / t_combined)
  : t1 = 2 * 60 :=
by
  -- The proof would come here
  sorry

end NUMINAMATH_GPT_first_valve_fill_time_l1569_156983


namespace NUMINAMATH_GPT_find_digits_sum_l1569_156925

theorem find_digits_sum (a b c : Nat) (ha : 0 <= a ∧ a <= 9) (hb : 0 <= b ∧ b <= 9) 
  (hc : 0 <= c ∧ c <= 9) 
  (h1 : 2 * a = c) 
  (h2 : b = b) : 
  a + b + c = 11 :=
  sorry

end NUMINAMATH_GPT_find_digits_sum_l1569_156925


namespace NUMINAMATH_GPT_probability_of_event_a_l1569_156901

-- Given conditions and question
variables (a b : Prop)
variables (p : Prop → ℝ)

-- Given conditions
axiom p_a : p a = 4 / 5
axiom p_b : p b = 2 / 5
axiom p_a_and_b_given : p (a ∧ b) = 0.32
axiom independent_a_b : p (a ∧ b) = p a * p b

-- The proof statement we need to prove: p a = 0.8
theorem probability_of_event_a :
  p a = 0.8 :=
sorry

end NUMINAMATH_GPT_probability_of_event_a_l1569_156901


namespace NUMINAMATH_GPT_distance_between_closest_points_l1569_156953

noncomputable def distance_closest_points :=
  let center1 : ℝ × ℝ := (5, 3)
  let center2 : ℝ × ℝ := (20, 7)
  let radius1 := center1.2  -- radius of first circle is y-coordinate of its center
  let radius2 := center2.2  -- radius of second circle is y-coordinate of its center
  let distance_centers := Real.sqrt ((center2.1 - center1.1)^2 + (center2.2 - center1.2)^2)
  distance_centers - radius1 - radius2

theorem distance_between_closest_points :
  distance_closest_points = Real.sqrt 241 - 10 :=
sorry

end NUMINAMATH_GPT_distance_between_closest_points_l1569_156953


namespace NUMINAMATH_GPT_least_possible_value_of_b_plus_c_l1569_156936

theorem least_possible_value_of_b_plus_c :
  ∃ (b c : ℕ), (b > 0) ∧ (c > 0) ∧ (∃ (r1 r2 : ℝ), r1 - r2 = 30 ∧ 2 * r1 ^ 2 + b * r1 + c = 0 ∧ 2 * r2 ^ 2 + b * r2 + c = 0) ∧ b + c = 126 := 
by
  sorry 

end NUMINAMATH_GPT_least_possible_value_of_b_plus_c_l1569_156936


namespace NUMINAMATH_GPT_minimize_distance_l1569_156966

noncomputable def f (x : ℝ) := x^2 - 2 * x
noncomputable def P (x : ℝ) : ℝ × ℝ := (x, f x)
def Q : ℝ × ℝ := (4, -1)

theorem minimize_distance : ∃ (x : ℝ), dist (P x) Q = Real.sqrt 5 := by
  sorry

end NUMINAMATH_GPT_minimize_distance_l1569_156966


namespace NUMINAMATH_GPT_max_volume_tank_l1569_156995

theorem max_volume_tank (a b h : ℝ) (ha : a ≤ 1.5) (hb : b ≤ 1.5) (hh : h = 1.5) :
  a * b * h ≤ 3.375 :=
by {
  sorry
}

end NUMINAMATH_GPT_max_volume_tank_l1569_156995


namespace NUMINAMATH_GPT_find_smallest_y_l1569_156981

noncomputable def x : ℕ := 5 * 15 * 35

def is_perfect_fourth_power (n : ℕ) : Prop :=
  ∃ m : ℕ, m ^ 4 = n

theorem find_smallest_y : ∃ y : ℕ, y > 0 ∧ is_perfect_fourth_power (x * y) ∧ y = 46485 := by
  sorry

end NUMINAMATH_GPT_find_smallest_y_l1569_156981


namespace NUMINAMATH_GPT_cos_555_value_l1569_156935

noncomputable def cos_555_equals_neg_sqrt6_add_sqrt2_div4 : Prop :=
  (Real.cos 555 = -((Real.sqrt 6 + Real.sqrt 2) / 4))

theorem cos_555_value : cos_555_equals_neg_sqrt6_add_sqrt2_div4 :=
  by sorry

end NUMINAMATH_GPT_cos_555_value_l1569_156935


namespace NUMINAMATH_GPT_sum_of_six_terms_l1569_156926

theorem sum_of_six_terms (a1 : ℝ) (S4 : ℝ) (d : ℝ) (a1_eq : a1 = 1 / 2) (S4_eq : S4 = 20) :
  S4 = (4 * a1 + (4 * (4 - 1) / 2) * d) → (S4 = 20) →
  (6 * a1 + (6 * (6 - 1) / 2) * d = 48) :=
by
  intros
  sorry

end NUMINAMATH_GPT_sum_of_six_terms_l1569_156926


namespace NUMINAMATH_GPT_order_of_scores_l1569_156980

variables (E L T N : ℝ)

-- Conditions
axiom Lana_condition_1 : L ≠ T
axiom Lana_condition_2 : L ≠ N
axiom Lana_condition_3 : T ≠ N

axiom Tom_condition : ∃ L' E', L' ≠ T ∧ E' > L' ∧ E' ≠ T ∧ E' ≠ L' 

axiom Nina_condition : N < E

-- Proof statement
theorem order_of_scores :
  N < L ∧ L < T :=
sorry

end NUMINAMATH_GPT_order_of_scores_l1569_156980


namespace NUMINAMATH_GPT_minimum_time_to_finish_food_l1569_156961

-- Define the constants involved in the problem
def carrots_total : ℕ := 1000
def muffins_total : ℕ := 1000
def amy_carrots_rate : ℝ := 40 -- carrots per minute
def amy_muffins_rate : ℝ := 70 -- muffins per minute
def ben_carrots_rate : ℝ := 60 -- carrots per minute
def ben_muffins_rate : ℝ := 30 -- muffins per minute

-- Proof statement
theorem minimum_time_to_finish_food : 
  ∃ T : ℝ, 
  (∀ c : ℝ, c = 5 → 
  (∀ T_1 : ℝ, T_1 = (carrots_total / (amy_carrots_rate + ben_carrots_rate)) → 
  (∀ T_2 : ℝ, T_2 = ((muffins_total + (amy_muffins_rate * c)) / (amy_muffins_rate + ben_muffins_rate)) +
  (muffins_total / ben_muffins_rate) - T_1 - c →
  T = T_1 + T_2) ∧
  T = 23.5 )) :=
sorry

end NUMINAMATH_GPT_minimum_time_to_finish_food_l1569_156961


namespace NUMINAMATH_GPT_stock_increase_l1569_156931

theorem stock_increase (x : ℝ) (h₁ : x > 0) :
  (1.25 * (0.85 * x) - x) / x * 100 = 6.25 :=
by 
  -- {proof steps would go here}
  sorry

end NUMINAMATH_GPT_stock_increase_l1569_156931


namespace NUMINAMATH_GPT_gcd_max_value_l1569_156914

theorem gcd_max_value (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1008) : 
  ∃ d, d = Nat.gcd a b ∧ d = 504 :=
by
  sorry

end NUMINAMATH_GPT_gcd_max_value_l1569_156914


namespace NUMINAMATH_GPT_cone_cannot_have_rectangular_projection_l1569_156947

def orthographic_projection (solid : Type) : Type := sorry

theorem cone_cannot_have_rectangular_projection :
  (∀ (solid : Type), orthographic_projection solid = Rectangle → solid ≠ Cone) :=
sorry

end NUMINAMATH_GPT_cone_cannot_have_rectangular_projection_l1569_156947


namespace NUMINAMATH_GPT_max_sum_is_2017_l1569_156976

theorem max_sum_is_2017 (a b c : ℕ) 
  (h1 : a + b = 1014) 
  (h2 : c - b = 497) 
  (h3 : a > b) : 
  (a + b + c) ≤ 2017 := sorry

end NUMINAMATH_GPT_max_sum_is_2017_l1569_156976


namespace NUMINAMATH_GPT_right_handed_players_total_l1569_156985

theorem right_handed_players_total
    (total_players : ℕ)
    (throwers : ℕ)
    (left_handed : ℕ)
    (right_handed : ℕ) :
    total_players = 150 →
    throwers = 60 →
    left_handed = (total_players - throwers) / 2 →
    right_handed = (total_players - throwers) / 2 →
    total_players - throwers = 2 * left_handed →
    left_handed + right_handed + throwers = total_players →
    ∀ throwers : ℕ, throwers = 60 →
    right_handed + throwers = 105 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end NUMINAMATH_GPT_right_handed_players_total_l1569_156985


namespace NUMINAMATH_GPT_nat_digit_problem_l1569_156906

theorem nat_digit_problem :
  ∀ n : Nat, (n % 10 = (2016 * (n / 2016)) % 10) → (n = 4032 ∨ n = 8064 ∨ n = 12096 ∨ n = 16128) :=
by
  sorry

end NUMINAMATH_GPT_nat_digit_problem_l1569_156906


namespace NUMINAMATH_GPT_neither_sufficient_nor_necessary_l1569_156956

variable {a b : ℝ}

theorem neither_sufficient_nor_necessary (hab_ne_zero : a * b ≠ 0) :
  ¬ (a * b > 1 → a > (1 / b)) ∧ ¬ (a > (1 / b) → a * b > 1) :=
sorry

end NUMINAMATH_GPT_neither_sufficient_nor_necessary_l1569_156956


namespace NUMINAMATH_GPT_Sam_wins_probability_l1569_156939

-- Define the basic probabilities
def prob_hit : ℚ := 2 / 5
def prob_miss : ℚ := 3 / 5

-- Define the desired probability that Sam wins
noncomputable def p : ℚ := 5 / 8

-- The mathematical problem statement in Lean
theorem Sam_wins_probability :
  p = prob_hit + (prob_miss * prob_miss * p) := 
sorry

end NUMINAMATH_GPT_Sam_wins_probability_l1569_156939


namespace NUMINAMATH_GPT_mrs_berkeley_A_students_first_class_mrs_berkeley_A_students_extended_class_l1569_156933

noncomputable def ratio_of_A_students (total_students_A : ℕ) (A_students_A : ℕ) : ℚ :=
  A_students_A / total_students_A

theorem mrs_berkeley_A_students_first_class :
  ∀ (total_students_A : ℕ) (A_students_A : ℕ) (total_students_B : ℕ),
    total_students_A = 30 →
    A_students_A = 20 →
    total_students_B = 18 →
    (A_students_A / total_students_A) * total_students_B = 12 :=
by
  intros total_students_A A_students_A total_students_B hA1 hA2 hB
  sorry

theorem mrs_berkeley_A_students_extended_class :
  ∀ (total_students_A : ℕ) (A_students_A : ℕ) (total_students_B : ℕ),
    total_students_A = 30 →
    A_students_A = 20 →
    total_students_B = 27 →
    (A_students_A / total_students_A) * total_students_B = 18 :=
by
  intros total_students_A A_students_A total_students_B hA1 hA2 hB
  sorry

end NUMINAMATH_GPT_mrs_berkeley_A_students_first_class_mrs_berkeley_A_students_extended_class_l1569_156933


namespace NUMINAMATH_GPT_negation_of_universal_statement_l1569_156977

open Real

theorem negation_of_universal_statement :
  ¬(∀ x : ℝ, x^3 > x^2) ↔ ∃ x : ℝ, x^3 ≤ x^2 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_universal_statement_l1569_156977


namespace NUMINAMATH_GPT_games_in_tournament_l1569_156945

def single_elimination_games (n : Nat) : Nat :=
  n - 1

theorem games_in_tournament : single_elimination_games 24 = 23 := by
  sorry

end NUMINAMATH_GPT_games_in_tournament_l1569_156945


namespace NUMINAMATH_GPT_monotone_f_find_m_l1569_156986

noncomputable def f (x : ℝ) : ℝ := (2 * x - 2) / (x + 2)

theorem monotone_f : ∀ x1 x2 : ℝ, 0 ≤ x1 → 0 ≤ x2 → x1 < x2 → f x1 < f x2 :=
by
  sorry

theorem find_m (m : ℝ) : 
  (∃ m, (f m - f 1 = 1/2)) ↔ m = 2 :=
by
  sorry

end NUMINAMATH_GPT_monotone_f_find_m_l1569_156986


namespace NUMINAMATH_GPT_every_real_has_cube_root_l1569_156900

theorem every_real_has_cube_root : ∀ y : ℝ, ∃ x : ℝ, x^3 = y := 
by
  sorry

end NUMINAMATH_GPT_every_real_has_cube_root_l1569_156900


namespace NUMINAMATH_GPT_monotonicity_intervals_number_of_zeros_l1569_156978

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (x - 1) * Real.exp x - k / 2 * x^2

theorem monotonicity_intervals (k : ℝ) :
  (k ≤ 0 → (∀ x, x < 0 → f k x < 0) ∧ (∀ x, x ≥ 0 → f k x > 0)) ∧
  (0 < k ∧ k < 1 → 
    (∀ x, x < Real.log k → f k x < 0) ∧ (∀ x, x ≥ Real.log k ∧ x < 0 → f k x > 0) ∧ 
    (∀ x, x > 0 → f k x > 0)) ∧
  (k = 1 → ∀ x, f k x > 0) ∧
  (k > 1 → 
    (∀ x, x < 0 → f k x < 0) ∧ 
    (∀ x, x ≥ 0 ∧ x < Real.log k → f k x > 0) ∧ 
    (∀ x, x > Real.log k → f k x > 0)) :=
sorry

theorem number_of_zeros (k : ℝ) (h_nonpos : k ≤ 0) :
  (k < 0 → (∃ a b : ℝ, a < 0 ∧ b > 0 ∧ f k a = 0 ∧ f k b = 0)) ∧
  (k = 0 → f k 1 = 0 ∧ (∀ x, x ≠ 1 → f k x ≠ 0)) :=
sorry

end NUMINAMATH_GPT_monotonicity_intervals_number_of_zeros_l1569_156978


namespace NUMINAMATH_GPT_factor_expression_l1569_156964

theorem factor_expression (x : ℝ) : 75 * x^12 + 225 * x^24 = 75 * x^12 * (1 + 3 * x^12) :=
by sorry

end NUMINAMATH_GPT_factor_expression_l1569_156964


namespace NUMINAMATH_GPT_average_marks_of_first_class_l1569_156913

theorem average_marks_of_first_class (n1 n2 : ℕ) (avg2 avg_all : ℝ)
  (h_n1 : n1 = 25) (h_n2 : n2 = 40) (h_avg2 : avg2 = 65) (h_avg_all : avg_all = 59.23076923076923) :
  ∃ (A : ℝ), A = 50 :=
by 
  sorry

end NUMINAMATH_GPT_average_marks_of_first_class_l1569_156913


namespace NUMINAMATH_GPT_least_common_multiple_prime_numbers_l1569_156987

theorem least_common_multiple_prime_numbers (x y : ℕ) (hx_prime : Prime x) (hy_prime : Prime y)
  (hxy : y < x) (h_eq : 2 * x + y = 12) : Nat.lcm x y = 10 :=
by
  sorry

end NUMINAMATH_GPT_least_common_multiple_prime_numbers_l1569_156987


namespace NUMINAMATH_GPT_find_value_of_expression_l1569_156954

variable {x : ℝ}

theorem find_value_of_expression (h : x^2 - 2 * x = 3) : 3 * x^2 - 6 * x - 4 = 5 :=
sorry

end NUMINAMATH_GPT_find_value_of_expression_l1569_156954


namespace NUMINAMATH_GPT_mean_of_six_numbers_l1569_156927

theorem mean_of_six_numbers (sum_of_six: ℚ) (H: sum_of_six = 3 / 4) : sum_of_six / 6 = 1 / 8 := by
  sorry

end NUMINAMATH_GPT_mean_of_six_numbers_l1569_156927


namespace NUMINAMATH_GPT_tom_and_eva_children_count_l1569_156902

theorem tom_and_eva_children_count (karen_donald_children : ℕ)
  (total_legs_in_pool : ℕ) (people_not_in_pool : ℕ) 
  (total_legs_each_person : ℕ) (karen_donald : ℕ) (tom_eva : ℕ) 
  (total_people_in_pool : ℕ) (total_people : ℕ) :
  karen_donald_children = 6 ∧ total_legs_in_pool = 16 ∧ people_not_in_pool = 6 ∧ total_legs_each_person = 2 ∧
  karen_donald = 2 ∧ tom_eva = 2 ∧ total_people_in_pool = total_legs_in_pool / total_legs_each_person ∧ 
  total_people = total_people_in_pool + people_not_in_pool ∧ 
  total_people - (karen_donald + karen_donald_children + tom_eva) = 4 :=
by
  intros
  sorry

end NUMINAMATH_GPT_tom_and_eva_children_count_l1569_156902


namespace NUMINAMATH_GPT_length_MN_proof_l1569_156903

-- Declare a noncomputable section to avoid computational requirements
noncomputable section

-- Define the quadrilateral ABCD with given sides
structure Quadrilateral :=
  (BC AD AB CD : ℕ)
  (BC_AD_parallel : Prop)

-- Define a theorem to calculate the length MN
theorem length_MN_proof (ABCD : Quadrilateral) 
  (M N : ℝ) (BisectorsIntersect_M : Prop) (BisectorsIntersect_N : Prop) : 
  ABCD.BC = 26 → ABCD.AD = 5 → ABCD.AB = 10 → ABCD.CD = 17 → 
  (MN = 2 ↔ (BC + AD - AB - CD) / 2 = 2) :=
by
  sorry

end NUMINAMATH_GPT_length_MN_proof_l1569_156903


namespace NUMINAMATH_GPT_problem_not_true_equation_l1569_156972

theorem problem_not_true_equation
  (a b : ℝ)
  (h : a / b = 2 / 3) : a / b ≠ (a + 2) / (b + 2) := 
sorry

end NUMINAMATH_GPT_problem_not_true_equation_l1569_156972


namespace NUMINAMATH_GPT_work_problem_l1569_156938

theorem work_problem (W : ℝ) (A_rate : ℝ) (AB_rate : ℝ) : A_rate = W / 14 ∧ AB_rate = W / 10 → 1 / (AB_rate - A_rate) = 35 :=
by
  sorry

end NUMINAMATH_GPT_work_problem_l1569_156938


namespace NUMINAMATH_GPT_hike_took_one_hour_l1569_156957

-- Define the constants and conditions
def initial_cups : ℕ := 6
def remaining_cups : ℕ := 1
def leak_rate : ℕ := 1 -- cups per hour
def drank_last_mile : ℚ := 1
def drank_first_3_miles_per_mile : ℚ := 2/3
def first_3_miles : ℕ := 3

-- Define the hike duration we want to prove
def hike_duration := 1

-- The total water drank
def total_drank := drank_last_mile + drank_first_3_miles_per_mile * first_3_miles

-- Prove the hike took 1 hour
theorem hike_took_one_hour :
  ∃ hours : ℕ, (initial_cups - remaining_cups = hours * leak_rate + total_drank) ∧ (hours = hike_duration) :=
by
  sorry

end NUMINAMATH_GPT_hike_took_one_hour_l1569_156957


namespace NUMINAMATH_GPT_graph_not_in_first_quadrant_l1569_156910

theorem graph_not_in_first_quadrant (a b : ℝ) (h_a_pos : a > 0) (h_a_ne_one : a ≠ 1) 
  (h_not_in_first_quadrant : ∀ x : ℝ, a^x + b - 1 ≤ 0) : 
  0 < a ∧ a < 1 ∧ b ≤ 0 :=
sorry

end NUMINAMATH_GPT_graph_not_in_first_quadrant_l1569_156910


namespace NUMINAMATH_GPT_range_of_a_min_value_reciprocals_l1569_156969

noncomputable def f (x a : ℝ) : ℝ := |x - 2| + |x - a^2|

theorem range_of_a (a : ℝ) : (∃ x : ℝ, f x a ≤ a) ↔ 1 ≤ a ∧ a ≤ 2 := by
  sorry

theorem min_value_reciprocals (m n a : ℝ) (h : m + 2 * n = a) (ha : a = 2) : (1/m + 1/n) ≥ (3/2 + Real.sqrt 2) := by
  sorry

end NUMINAMATH_GPT_range_of_a_min_value_reciprocals_l1569_156969


namespace NUMINAMATH_GPT_colleague_typing_time_l1569_156999

theorem colleague_typing_time (T : ℝ) : 
  (∀ me_time : ℝ, (me_time = 180) →
  (∀ my_speed my_colleague_speed : ℝ, (my_speed = T / me_time) →
  (my_colleague_speed = 4 * my_speed) →
  (T / my_colleague_speed = 45))) :=
  sorry

end NUMINAMATH_GPT_colleague_typing_time_l1569_156999


namespace NUMINAMATH_GPT_number_of_foals_l1569_156916

theorem number_of_foals (t f : ℕ) (h1 : t + f = 11) (h2 : 2 * t + 4 * f = 30) : f = 4 :=
by
  sorry

end NUMINAMATH_GPT_number_of_foals_l1569_156916


namespace NUMINAMATH_GPT_permutation_equals_power_l1569_156940

-- Definition of permutation with repetition
def permutation_with_repetition (n k : ℕ) : ℕ := n ^ k

-- Theorem to prove
theorem permutation_equals_power (n k : ℕ) : permutation_with_repetition n k = n ^ k :=
by
  sorry

end NUMINAMATH_GPT_permutation_equals_power_l1569_156940


namespace NUMINAMATH_GPT_remaining_speed_l1569_156943
open Real

theorem remaining_speed
  (D T : ℝ) (h1 : 40 * (T / 3) = (2 / 3) * D)
  (h2 : (T / 3) * 3 = T) :
  (D / 3) / ((2 * ((2 / 3) * D) / (40) / (3)) * 2 / 3) = 10 :=
by
  sorry

end NUMINAMATH_GPT_remaining_speed_l1569_156943


namespace NUMINAMATH_GPT_baskets_delivered_l1569_156907

theorem baskets_delivered 
  (peaches_per_basket : ℕ := 25)
  (boxes : ℕ := 8)
  (peaches_per_box : ℕ := 15)
  (peaches_eaten : ℕ := 5)
  (peaches_in_boxes := boxes * peaches_per_box) 
  (total_peaches := peaches_in_boxes + peaches_eaten) : 
  total_peaches / peaches_per_basket = 5 :=
by
  sorry

end NUMINAMATH_GPT_baskets_delivered_l1569_156907


namespace NUMINAMATH_GPT_count_4_digit_numbers_with_property_l1569_156932

noncomputable def count_valid_4_digit_numbers : ℕ :=
  let valid_units (t : ℕ) : List ℕ := List.filter (λ u => u ≥ 3 * t) [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  let choices_for_tu : ℕ := (List.length (valid_units 0)) + (List.length (valid_units 1)) + (List.length (valid_units 2))
  choices_for_tu * 9 * 9

theorem count_4_digit_numbers_with_property : count_valid_4_digit_numbers = 1701 := by
  sorry

end NUMINAMATH_GPT_count_4_digit_numbers_with_property_l1569_156932


namespace NUMINAMATH_GPT_volume_ratio_of_cones_l1569_156912

theorem volume_ratio_of_cones (R : ℝ) (hR : 0 < R) :
  let circumference := 2 * Real.pi * R
  let sector1_circumference := (2 / 3) * circumference
  let sector2_circumference := (1 / 3) * circumference
  let r1 := sector1_circumference / (2 * Real.pi)
  let r2 := sector2_circumference / (2 * Real.pi)
  let s := R
  let h1 := Real.sqrt (R^2 - r1^2)
  let h2 := Real.sqrt (R^2 - r2^2)
  let V1 := (Real.pi * r1^2 * h1) / 3
  let V2 := (Real.pi * r2^2 * h2) / 3
  V1 / V2 = Real.sqrt 10 := 
by
  sorry

end NUMINAMATH_GPT_volume_ratio_of_cones_l1569_156912


namespace NUMINAMATH_GPT_sufficient_not_necessary_l1569_156944

theorem sufficient_not_necessary (a : ℝ) : (a > 1 → 1 / a < 1) ∧ (∃ x, 1 / x < 1 ∧ ¬(x > 1)) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_l1569_156944


namespace NUMINAMATH_GPT_tangent_line_eqn_l1569_156921

noncomputable def f (x : ℝ) : ℝ := 5 * x + Real.log x

theorem tangent_line_eqn : ∀ x y : ℝ, (x, y) = (1, f 1) → 6 * x - y - 1 = 0 := 
by
  intro x y h
  sorry

end NUMINAMATH_GPT_tangent_line_eqn_l1569_156921


namespace NUMINAMATH_GPT_smallest_n_exists_l1569_156919

theorem smallest_n_exists (n : ℤ) (r : ℝ) : 
  (∃ m : ℤ, m = (↑n + r) ^ 3 ∧ r > 0 ∧ r < 1 / 1000) ∧ n > 0 → n = 19 := 
by sorry

end NUMINAMATH_GPT_smallest_n_exists_l1569_156919


namespace NUMINAMATH_GPT_river_depth_mid_June_l1569_156988

theorem river_depth_mid_June (D : ℝ) : 
    (∀ (mid_May mid_June mid_July : ℝ),
    mid_May = 5 →
    mid_June = mid_May + D →
    mid_July = 3 * mid_June →
    mid_July = 45) →
    D = 10 :=
by
    sorry

end NUMINAMATH_GPT_river_depth_mid_June_l1569_156988


namespace NUMINAMATH_GPT_intersection_of_sets_l1569_156929

-- Definitions from the conditions.
def A := { x : ℝ | x^2 - 2 * x ≤ 0 }
def B := { x : ℝ | x > 1 }

-- The proof problem statement.
theorem intersection_of_sets :
  A ∩ B = { x : ℝ | 1 < x ∧ x ≤ 2 } :=
sorry

end NUMINAMATH_GPT_intersection_of_sets_l1569_156929


namespace NUMINAMATH_GPT_aquarium_final_volume_l1569_156918

theorem aquarium_final_volume :
  let length := 4
  let width := 6
  let height := 3
  let total_volume := length * width * height
  let initial_volume := total_volume / 2
  let spilled_volume := initial_volume / 2
  let remaining_volume := initial_volume - spilled_volume
  let final_volume := remaining_volume * 3
  final_volume = 54 :=
by sorry

end NUMINAMATH_GPT_aquarium_final_volume_l1569_156918


namespace NUMINAMATH_GPT_range_of_x0_l1569_156950

noncomputable def point_on_circle_and_line (x0 : ℝ) (y0 : ℝ) : Prop :=
(x0^2 + y0^2 = 1) ∧ (3 * x0 + 2 * y0 = 4)

theorem range_of_x0 
  (x0 : ℝ) (y0 : ℝ) 
  (h1 : 3 * x0 + 2 * y0 = 4)
  (h2 : ∃ A B : ℝ × ℝ, (A.1^2 + A.2^2 = 1) ∧ (B.1^2 + B.2^2 = 1) ∧ (A ≠ B) ∧ (A + B = (x0, y0))) :
  0 < x0 ∧ x0 < 24 / 13 :=
sorry

end NUMINAMATH_GPT_range_of_x0_l1569_156950


namespace NUMINAMATH_GPT_percentage_increase_l1569_156989

variables (J T P : ℝ)

def income_conditions (J T P : ℝ) : Prop :=
  (T = 0.5 * J) ∧ (P = 0.8 * J)

theorem percentage_increase (J T P : ℝ) (h : income_conditions J T P) :
  ((P / T) - 1) * 100 = 60 :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_l1569_156989


namespace NUMINAMATH_GPT_sum_ratio_l1569_156930

noncomputable def geometric_sequence_sum (a1 q : ℝ) (n : ℕ) : ℝ := 
  a1 * (1 - q^n) / (1 - q)

theorem sum_ratio (a1 q : ℝ) 
  (h : 8 * (a1 * q) + (a1 * q^4) = 0) :
  geometric_sequence_sum a1 q 6 / geometric_sequence_sum a1 q 3 = -7 := 
by
  sorry

end NUMINAMATH_GPT_sum_ratio_l1569_156930


namespace NUMINAMATH_GPT_abc_eq_l1569_156960

theorem abc_eq (a b : ℕ) (ha : 0 < a) (hb : 0 < b)
  (h : 4 * a * b - 1 ∣ (4 * a * a - 1) ^ 2) : a = b :=
sorry

end NUMINAMATH_GPT_abc_eq_l1569_156960


namespace NUMINAMATH_GPT_change_given_l1569_156962

theorem change_given (pants_cost : ℕ) (shirt_cost : ℕ) (tie_cost : ℕ) (total_paid : ℕ) (total_cost : ℕ) (change : ℕ) :
  pants_cost = 140 ∧ shirt_cost = 43 ∧ tie_cost = 15 ∧ total_paid = 200 ∧ total_cost = (pants_cost + shirt_cost + tie_cost) ∧ change = (total_paid - total_cost) → change = 2 :=
by
  sorry

end NUMINAMATH_GPT_change_given_l1569_156962


namespace NUMINAMATH_GPT_casey_pumping_time_l1569_156979

structure PlantRow :=
  (rows : ℕ) (plants_per_row : ℕ) (water_per_plant : ℚ)

structure Animal :=
  (count : ℕ) (water_per_animal : ℚ)

def morning_pump_rate := 3 -- gallons per minute
def afternoon_pump_rate := 5 -- gallons per minute

def corn := PlantRow.mk 4 15 0.5
def pumpkin := PlantRow.mk 3 10 0.8
def pigs := Animal.mk 10 4
def ducks := Animal.mk 20 0.25
def cows := Animal.mk 5 8

def total_water_needed_for_plants (corn pumpkin : PlantRow) : ℚ :=
  (corn.rows * corn.plants_per_row * corn.water_per_plant) +
  (pumpkin.rows * pumpkin.plants_per_row * pumpkin.water_per_plant)

def total_water_needed_for_animals (pigs ducks cows : Animal) : ℚ :=
  (pigs.count * pigs.water_per_animal) +
  (ducks.count * ducks.water_per_animal) +
  (cows.count * cows.water_per_animal)

def time_to_pump (total_water pump_rate : ℚ) : ℚ :=
  total_water / pump_rate

theorem casey_pumping_time :
  let total_water_plants := total_water_needed_for_plants corn pumpkin
  let total_water_animals := total_water_needed_for_animals pigs ducks cows
  let time_morning := time_to_pump total_water_plants morning_pump_rate
  let time_afternoon := time_to_pump total_water_animals afternoon_pump_rate
  time_morning + time_afternoon = 35 := by
sorry

end NUMINAMATH_GPT_casey_pumping_time_l1569_156979


namespace NUMINAMATH_GPT_find_omega_l1569_156915

theorem find_omega (ω : Real) (h : ∀ x : Real, (1 / 2) * Real.cos (ω * x - (Real.pi / 6)) = (1 / 2) * Real.cos (ω * (x + Real.pi) - (Real.pi / 6))) : ω = 2 ∨ ω = -2 :=
by
  sorry

end NUMINAMATH_GPT_find_omega_l1569_156915


namespace NUMINAMATH_GPT_store_A_more_advantageous_l1569_156934

theorem store_A_more_advantageous (x : ℕ) (h : x > 5) : 
  6000 + 4500 * (x - 1) < 4800 * x := 
by 
  sorry

end NUMINAMATH_GPT_store_A_more_advantageous_l1569_156934


namespace NUMINAMATH_GPT_a100_gt_two_pow_99_l1569_156922

theorem a100_gt_two_pow_99 (a : ℕ → ℤ) (h_pos : ∀ n, 0 < a n) 
  (h1 : a 1 > a 0) (h_rec : ∀ n ≥ 2, a n = 3 * a (n - 1) - 2 * a (n - 2)) :
  a 100 > 2 ^ 99 :=
sorry

end NUMINAMATH_GPT_a100_gt_two_pow_99_l1569_156922


namespace NUMINAMATH_GPT_simplest_square_root_l1569_156994

theorem simplest_square_root (a b c d : ℝ) (h1 : a = 3) (h2 : b = 2 * Real.sqrt 3) (h3 : c = (Real.sqrt 2) / 2) (h4 : d = Real.sqrt 10) :
  d = Real.sqrt 10 ∧ (a ≠ Real.sqrt 10) ∧ (b ≠ Real.sqrt 10) ∧ (c ≠ Real.sqrt 10) := 
by 
  sorry

end NUMINAMATH_GPT_simplest_square_root_l1569_156994


namespace NUMINAMATH_GPT_sequence_geometric_and_formula_l1569_156997

theorem sequence_geometric_and_formula (a : ℕ → ℕ) (h₁ : a 1 = 1) (h₂ : ∀ n, a (n + 1) = 2 * a n + 1) :
  (∀ n, a n + 1 = 2 ^ n) ∧ (a n = 2 ^ n - 1) :=
sorry

end NUMINAMATH_GPT_sequence_geometric_and_formula_l1569_156997


namespace NUMINAMATH_GPT_max_odd_integers_chosen_l1569_156958

theorem max_odd_integers_chosen (a b c d e f : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0) (h_prod_even : a * b * c * d * e * f % 2 = 0) : 
  (∀ n : ℕ, n = 5 → ∃ a b c d e, (a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1 ∧ d % 2 = 1 ∧ e % 2 = 1) ∧ f % 2 = 0) :=
sorry

end NUMINAMATH_GPT_max_odd_integers_chosen_l1569_156958


namespace NUMINAMATH_GPT_min_value_of_alpha_beta_l1569_156998

theorem min_value_of_alpha_beta 
  (k : ℝ)
  (h_k : k ≤ -4 ∨ k ≥ 5)
  (α β : ℝ)
  (h_αβ : α^2 - 2 * k * α + (k + 20) = 0 ∧ β^2 - 2 * k * β + (k + 20) = 0) :
  (α + 1) ^ 2 + (β + 1) ^ 2 = 18 → k = -4 :=
sorry

end NUMINAMATH_GPT_min_value_of_alpha_beta_l1569_156998


namespace NUMINAMATH_GPT_polynomial_condition_degree_n_l1569_156909

open Polynomial

theorem polynomial_condition_degree_n 
  (P_n : ℤ[X]) (n : ℕ) (hn_pos : 0 < n) (hn_deg : P_n.degree = n) 
  (hx0 : P_n.eval 0 = 0)
  (hx_conditions : ∃ (a : ℤ) (b : Fin n → ℤ), ∀ i, P_n.eval (b i) = n) : 
  n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 := 
sorry

end NUMINAMATH_GPT_polynomial_condition_degree_n_l1569_156909


namespace NUMINAMATH_GPT_problem_statement_l1569_156968

theorem problem_statement (x y : ℤ) (h1 : x = 8) (h2 : y = 3) :
  (x - 2 * y) * (x + 2 * y) = 28 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1569_156968


namespace NUMINAMATH_GPT_find_constants_l1569_156973

theorem find_constants (a b c d : ℚ) :
  (6 * x^3 - 4 * x + 2) * (a * x^3 + b * x^2 + c * x + d) =
  18 * x^6 - 2 * x^5 + 16 * x^4 - (28 / 3) * x^3 + (8 / 3) * x^2 - 4 * x + 2 →
  a = 3 ∧ b = -1 / 3 ∧ c = 14 / 9 :=
by
  sorry

end NUMINAMATH_GPT_find_constants_l1569_156973


namespace NUMINAMATH_GPT_points_meet_every_720_seconds_l1569_156952

theorem points_meet_every_720_seconds
    (v1 v2 : ℝ) 
    (h1 : v1 - v2 = 1/720) 
    (h2 : (1/v2) - (1/v1) = 10) :
    v1 = 1/80 ∧ v2 = 1/90 :=
by
  sorry

end NUMINAMATH_GPT_points_meet_every_720_seconds_l1569_156952


namespace NUMINAMATH_GPT_cos_angle_of_vectors_l1569_156904

variables (a b : EuclideanSpace ℝ (Fin 2))

theorem cos_angle_of_vectors (h1 : ‖a‖ = 2) (h2 : ‖b‖ = 1) (h3 : ‖a - b‖ = 2) :
  (inner a b) / (‖a‖ * ‖b‖) = 1/4 :=
by
  sorry

end NUMINAMATH_GPT_cos_angle_of_vectors_l1569_156904


namespace NUMINAMATH_GPT_number_divisors_product_l1569_156959

theorem number_divisors_product :
  ∃ N : ℕ, (∃ a b : ℕ, N = 3^a * 5^b ∧ (N^((a+1)*(b+1) / 2)) = 3^30 * 5^40) ∧ N = 3^3 * 5^4 :=
sorry

end NUMINAMATH_GPT_number_divisors_product_l1569_156959


namespace NUMINAMATH_GPT_option_A_option_B_option_C_option_D_l1569_156991

theorem option_A : (-4:ℤ)^2 ≠ -(4:ℤ)^2 := sorry
theorem option_B : (-2:ℤ)^3 = -2^3 := sorry
theorem option_C : (-1:ℤ)^2020 ≠ (-1:ℤ)^2021 := sorry
theorem option_D : ((2:ℚ)/(3:ℚ))^3 = ((2:ℚ)/(3:ℚ))^3 := sorry

end NUMINAMATH_GPT_option_A_option_B_option_C_option_D_l1569_156991


namespace NUMINAMATH_GPT_fraction_of_yard_occupied_by_flower_beds_l1569_156917

theorem fraction_of_yard_occupied_by_flower_beds :
  let leg_length := (36 - 26) / 3
  let triangle_area := (1 / 2) * leg_length^2
  let total_flower_bed_area := 3 * triangle_area
  let yard_area := 36 * 6
  (total_flower_bed_area / yard_area) = 25 / 324
  := by
  let leg_length := (36 - 26) / 3
  let triangle_area := (1 / 2) * leg_length^2
  let total_flower_bed_area := 3 * triangle_area
  let yard_area := 36 * 6
  have h1 : leg_length = 10 / 3 := by sorry
  have h2 : triangle_area = (1 / 2) * (10 / 3)^2 := by sorry
  have h3 : total_flower_bed_area = 3 * ((1 / 2) * (10 / 3)^2) := by sorry
  have h4 : yard_area = 216 := by sorry
  have h5 : total_flower_bed_area / yard_area = 25 / 324 := by sorry
  exact h5

end NUMINAMATH_GPT_fraction_of_yard_occupied_by_flower_beds_l1569_156917


namespace NUMINAMATH_GPT_petya_pencils_l1569_156993

theorem petya_pencils (x : ℕ) (promotion : x + 12 = 61) :
  x = 49 :=
by
  sorry

end NUMINAMATH_GPT_petya_pencils_l1569_156993


namespace NUMINAMATH_GPT_blue_balls_unchanged_l1569_156955

def initial_red_balls : ℕ := 3
def initial_blue_balls : ℕ := 2
def initial_yellow_balls : ℕ := 5
def added_yellow_balls : ℕ := 4

theorem blue_balls_unchanged :
  initial_blue_balls = 2 := by
  sorry

end NUMINAMATH_GPT_blue_balls_unchanged_l1569_156955


namespace NUMINAMATH_GPT_geom_seq_general_term_sum_geometric_arithmetic_l1569_156937

noncomputable def a_n (n : ℕ) : ℕ := 2^n
def b_n (n : ℕ) : ℕ := 2*n - 1

theorem geom_seq_general_term (a : ℕ → ℕ) (a1 : a 1 = 2)
  (a2 : a 3 = (a 2) + 4) : ∀ n, a n = a_n n :=
by
  sorry

theorem sum_geometric_arithmetic (a b : ℕ → ℕ) 
  (a_def : ∀ n, a n = 2 ^ n) (b_def : ∀ n, b n = 2 * n - 1) : 
  ∀ n, (Finset.range n).sum (λ i => (a (i + 1) + b (i + 1))) = 2^(n+1) + n^2 - 2 :=
by
  sorry

end NUMINAMATH_GPT_geom_seq_general_term_sum_geometric_arithmetic_l1569_156937


namespace NUMINAMATH_GPT_quadrilateral_area_correct_l1569_156990

-- Definitions of given conditions
structure Quadrilateral :=
(W X Y Z : Type)
(WX XY YZ YW : ℝ)
(angle_WXY : ℝ)
(area : ℝ)

-- Quadrilateral satisfies given conditions
def quadrilateral_WXYZ : Quadrilateral :=
{ W := ℝ,
  X := ℝ,
  Y := ℝ,
  Z := ℝ,
  WX := 9,
  XY := 5,
  YZ := 12,
  YW := 15,
  angle_WXY := 90,
  area := 76.5 }

-- The theorem stating the area of quadrilateral WXYZ is 76.5
theorem quadrilateral_area_correct : quadrilateral_WXYZ.area = 76.5 :=
sorry

end NUMINAMATH_GPT_quadrilateral_area_correct_l1569_156990


namespace NUMINAMATH_GPT_number_of_houses_with_neither_feature_l1569_156984

variable (T G P B : ℕ)

theorem number_of_houses_with_neither_feature 
  (hT : T = 90)
  (hG : G = 50)
  (hP : P = 40)
  (hB : B = 35) : 
  T - (G + P - B) = 35 := 
    by
      sorry

end NUMINAMATH_GPT_number_of_houses_with_neither_feature_l1569_156984


namespace NUMINAMATH_GPT_find_x_minus_y_l1569_156908

theorem find_x_minus_y (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 16) : x - y = 2 :=
by
  have h3 : x^2 - y^2 = (x + y) * (x - y) := by sorry
  have h4 : (x + y) * (x - y) = 8 * (x - y) := by sorry
  have h5 : 16 = 8 * (x - y) := by sorry
  have h6 : 16 = 8 * (x - y) := by sorry
  have h7 : x - y = 2 := by sorry
  exact h7

end NUMINAMATH_GPT_find_x_minus_y_l1569_156908


namespace NUMINAMATH_GPT_min_w_value_l1569_156928

def w (x y : ℝ) : ℝ := 3 * x^2 + 5 * y^2 + 12 * x - 10 * y + 45

theorem min_w_value : ∀ x y : ℝ, (w x y) ≥ 28 ∧ (∃ x y : ℝ, (w x y) = 28) :=
by
  sorry

end NUMINAMATH_GPT_min_w_value_l1569_156928


namespace NUMINAMATH_GPT_sequence_a_n_eq_T_n_formula_C_n_formula_l1569_156982

noncomputable def sequence_S (n : ℕ) : ℕ := n * (2 * n - 1)

def arithmetic_seq (n : ℕ) : ℚ := 2 * n - 1

def a_n (n : ℕ) : ℤ := 4 * n - 3

def b_n (n : ℕ) : ℚ := 1 / (a_n n * a_n (n + 1))

def T_n (n : ℕ) : ℚ := (n : ℚ) / (4 * n + 1)

def c_n (n : ℕ) : ℚ := 3^(n - 1)

def C_n (n : ℕ) : ℚ := (3^n - 1) / 2

theorem sequence_a_n_eq (n : ℕ) : a_n n = 4 * n - 3 := by sorry

theorem T_n_formula (n : ℕ) : T_n n = (n : ℚ) / (4 * n + 1) := by sorry

theorem C_n_formula (n : ℕ) : C_n n = (3^n - 1) / 2 := by sorry

end NUMINAMATH_GPT_sequence_a_n_eq_T_n_formula_C_n_formula_l1569_156982


namespace NUMINAMATH_GPT_neg_div_neg_eq_pos_division_of_negatives_example_l1569_156970

theorem neg_div_neg_eq_pos (a b : Int) (hb : b ≠ 0) : (-a) / (-b) = a / b := by
  -- You can complete the proof here
  sorry

theorem division_of_negatives_example : (-81 : Int) / (-9) = 9 :=
  neg_div_neg_eq_pos 81 9 (by decide)

end NUMINAMATH_GPT_neg_div_neg_eq_pos_division_of_negatives_example_l1569_156970


namespace NUMINAMATH_GPT_three_digit_numbers_distinct_base_l1569_156923

theorem three_digit_numbers_distinct_base (b : ℕ) (h : (b - 1) ^ 2 * (b - 2) = 250) : b = 8 :=
sorry

end NUMINAMATH_GPT_three_digit_numbers_distinct_base_l1569_156923


namespace NUMINAMATH_GPT_circle_center_coordinates_l1569_156905

theorem circle_center_coordinates :
  ∀ x y, (x^2 + y^2 - 4 * x - 2 * y - 5 = 0) → (x, y) = (2, 1) :=
by
  sorry

end NUMINAMATH_GPT_circle_center_coordinates_l1569_156905


namespace NUMINAMATH_GPT_least_positive_integer_congruences_l1569_156949

theorem least_positive_integer_congruences :
  ∃ n : ℕ, 
    n > 0 ∧ 
    (n % 4 = 1) ∧ 
    (n % 5 = 2) ∧ 
    (n % 6 = 3) ∧ 
    (n = 57) :=
by
  sorry

end NUMINAMATH_GPT_least_positive_integer_congruences_l1569_156949


namespace NUMINAMATH_GPT_number_of_ordered_triples_l1569_156967

theorem number_of_ordered_triples (a b c : ℕ) (h1 : a ≤ b) (h2 : b ≤ c) (h3 : b = 3969) (h4 : a * c = 3969^2) :
    ∃ n : ℕ, n = 12 := sorry

end NUMINAMATH_GPT_number_of_ordered_triples_l1569_156967


namespace NUMINAMATH_GPT_age_difference_l1569_156920

/-- The age difference between each child d -/
theorem age_difference (d : ℝ) 
  (h1 : ∃ a b c e : ℝ, d = a ∧ 2*d = b ∧ 3*d = c ∧ 4*d = e)
  (h2 : 12 + (12 - d) + (12 - 2*d) + (12 - 3*d) + (12 - 4*d) = 40) : 
  d = 2 := 
sorry

end NUMINAMATH_GPT_age_difference_l1569_156920


namespace NUMINAMATH_GPT_soccer_ball_cost_l1569_156946

theorem soccer_ball_cost :
  ∃ x y : ℝ, x + y = 100 ∧ 2 * x + 3 * y = 262 ∧ x = 38 :=
by
  sorry

end NUMINAMATH_GPT_soccer_ball_cost_l1569_156946


namespace NUMINAMATH_GPT_sum_D_E_correct_sum_of_all_possible_values_of_D_E_l1569_156963

theorem sum_D_E_correct :
  ∀ (D E : ℕ), (D < 10) → (E < 10) →
  (∃ k : ℕ, (10^8 * D + 4650000 + 1000 * E + 32) = 7 * k) →
  D + E = 1 ∨ D + E = 8 ∨ D + E = 15 :=
by sorry

theorem sum_of_all_possible_values_of_D_E :
  (1 + 8 + 15) = 24 :=
by norm_num

end NUMINAMATH_GPT_sum_D_E_correct_sum_of_all_possible_values_of_D_E_l1569_156963
