import Mathlib

namespace NUMINAMATH_GPT_number_of_ordered_triples_l2340_234040

/-- 
Prove the number of ordered triples (x, y, z) of positive integers that satisfy 
  lcm(x, y) = 180, lcm(x, z) = 210, and lcm(y, z) = 420 is 2.
-/
theorem number_of_ordered_triples (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h₁ : Nat.lcm x y = 180) (h₂ : Nat.lcm x z = 210) (h₃ : Nat.lcm y z = 420) : 
  ∃ (n : ℕ), n = 2 := 
sorry

end NUMINAMATH_GPT_number_of_ordered_triples_l2340_234040


namespace NUMINAMATH_GPT_similar_triangles_side_length_l2340_234081

theorem similar_triangles_side_length
  (A1 A2 : ℕ) (k : ℕ) (h1 : A1 - A2 = 18)
  (h2 : A1 = k^2 * A2) (h3 : ∃ n : ℕ, A2 = n)
  (s : ℕ) (h4 : s = 3) :
  s * k = 6 :=
by
  sorry

end NUMINAMATH_GPT_similar_triangles_side_length_l2340_234081


namespace NUMINAMATH_GPT_brad_zip_code_l2340_234035

theorem brad_zip_code (x y : ℕ) (h1 : x + x + 0 + 2 * x + y = 10) : 2 * x + y = 8 :=
by 
  sorry

end NUMINAMATH_GPT_brad_zip_code_l2340_234035


namespace NUMINAMATH_GPT_actual_time_when_watch_shows_8_PM_l2340_234091

-- Definitions based on the problem's conditions
def initial_time := 8  -- 8:00 AM
def incorrect_watch_time := 14 * 60 + 42  -- 2:42 PM converted to minutes
def actual_time := 15 * 60  -- 3:00 PM converted to minutes
def target_watch_time := 20 * 60  -- 8:00 PM converted to minutes

-- Define to calculate the rate of time loss
def time_loss_rate := (actual_time - incorrect_watch_time) / (actual_time - initial_time * 60)

-- Hypothesis that the watch loses time at a constant rate
axiom constant_rate : ∀ t, t >= initial_time * 60 ∧ t <= actual_time → (t * time_loss_rate) = (actual_time - incorrect_watch_time)

-- Define the target time based on watch reading 8:00 PM
noncomputable def target_actual_time := target_watch_time / time_loss_rate

-- Main theorem: Prove that given the conditions, the target actual time is 8:32 PM
theorem actual_time_when_watch_shows_8_PM : target_actual_time = (20 * 60 + 32) :=
sorry

end NUMINAMATH_GPT_actual_time_when_watch_shows_8_PM_l2340_234091


namespace NUMINAMATH_GPT_sufficient_condition_for_increasing_l2340_234029

theorem sufficient_condition_for_increasing (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) :
  (∀ x y : ℝ, x < y → a^y < a^x) →
  (∀ x y : ℝ, x < y → (2 - a) * y ^ 3 > (2 - a) * x ^ 3) :=
sorry

end NUMINAMATH_GPT_sufficient_condition_for_increasing_l2340_234029


namespace NUMINAMATH_GPT_squares_perimeter_and_rectangle_area_l2340_234001

theorem squares_perimeter_and_rectangle_area (x y : ℝ) (hx : x^2 + y^2 = 145) (hy : x^2 - y^2 = 105) : 
  (4 * x + 4 * y = 28 * Real.sqrt 5) ∧ ((x + y) * x = 175) := 
by 
  sorry

end NUMINAMATH_GPT_squares_perimeter_and_rectangle_area_l2340_234001


namespace NUMINAMATH_GPT_pat_oj_consumption_l2340_234052

def initial_oj : ℚ := 3 / 4
def alex_fraction : ℚ := 1 / 2
def pat_fraction : ℚ := 1 / 3

theorem pat_oj_consumption : pat_fraction * (initial_oj * (1 - alex_fraction)) = 1 / 8 := by
  -- This will be the proof part which can be filled later
  sorry

end NUMINAMATH_GPT_pat_oj_consumption_l2340_234052


namespace NUMINAMATH_GPT_average_GPA_of_whole_class_l2340_234047

variable (n : ℕ)

def GPA_first_group : ℕ := 54 * (n / 3)
def GPA_second_group : ℕ := 45 * (2 * n / 3)
def total_GPA : ℕ := GPA_first_group n + GPA_second_group n

theorem average_GPA_of_whole_class : total_GPA n / n = 48 := by
  sorry

end NUMINAMATH_GPT_average_GPA_of_whole_class_l2340_234047


namespace NUMINAMATH_GPT_determine_expr_l2340_234058

noncomputable def expr (a b c d : ℝ) : ℝ :=
  (1 + a + a * b) / (1 + a + a * b + a * b * c) +
  (1 + b + b * c) / (1 + b + b * c + b * c * d) +
  (1 + c + c * d) / (1 + c + c * d + c * d * a) +
  (1 + d + d * a) / (1 + d + d * a + d * a * b)

theorem determine_expr (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : a * b * c * d = 1) :
  expr a b c d = 2 :=
sorry

end NUMINAMATH_GPT_determine_expr_l2340_234058


namespace NUMINAMATH_GPT_expected_score_of_basketball_player_l2340_234048

theorem expected_score_of_basketball_player :
  let p_inside : ℝ := 0.7
  let p_outside : ℝ := 0.4
  let attempts_inside : ℕ := 10
  let attempts_outside : ℕ := 5
  let points_inside : ℕ := 2
  let points_outside : ℕ := 3
  let E_inside : ℝ := attempts_inside * p_inside * points_inside
  let E_outside : ℝ := attempts_outside * p_outside * points_outside
  E_inside + E_outside = 20 :=
by
  sorry

end NUMINAMATH_GPT_expected_score_of_basketball_player_l2340_234048


namespace NUMINAMATH_GPT_game_is_not_fair_l2340_234026

noncomputable def expected_winnings : ℚ := 
  let p_1 := 1 / 8
  let p_2 := 7 / 8
  let gain_case_1 := 2
  let loss_case_2 := -1 / 7
  (p_1 * gain_case_1) + (p_2 * loss_case_2)

theorem game_is_not_fair : expected_winnings = 1 / 8 :=
sorry

end NUMINAMATH_GPT_game_is_not_fair_l2340_234026


namespace NUMINAMATH_GPT_eli_age_difference_l2340_234095

theorem eli_age_difference (kaylin_age : ℕ) (freyja_age : ℕ) (sarah_age : ℕ) (eli_age : ℕ) 
  (H1 : kaylin_age = 33)
  (H2 : freyja_age = 10)
  (H3 : kaylin_age + 5 = sarah_age)
  (H4 : sarah_age = 2 * eli_age) :
  eli_age - freyja_age = 9 := 
sorry

end NUMINAMATH_GPT_eli_age_difference_l2340_234095


namespace NUMINAMATH_GPT_simplify_fraction_l2340_234059

theorem simplify_fraction : (3 : ℚ) / 462 + 17 / 42 = 95 / 231 :=
by sorry

end NUMINAMATH_GPT_simplify_fraction_l2340_234059


namespace NUMINAMATH_GPT_div_sqrt_81_by_3_is_3_l2340_234039

-- Definitions based on conditions
def sqrt_81 := Nat.sqrt 81
def number_3 := 3

-- Problem statement
theorem div_sqrt_81_by_3_is_3 : sqrt_81 / number_3 = 3 := by
  sorry

end NUMINAMATH_GPT_div_sqrt_81_by_3_is_3_l2340_234039


namespace NUMINAMATH_GPT_num_ordered_triples_l2340_234069

/-
Let Q be a right rectangular prism with integral side lengths a, b, and c such that a ≤ b ≤ c, and b = 2023.
A plane parallel to one of the faces of Q cuts Q into two prisms, one of which is similar to Q, and both have nonzero volume.
Prove that the number of ordered triples (a, b, c) such that b = 2023 is 7.
-/

theorem num_ordered_triples (a c : ℕ) (h : a ≤ 2023 ∧ 2023 ≤ c) (ac_eq_2023_squared : a * c = 2023^2) :
  ∃ count, count = 7 :=
by {
  sorry
}

end NUMINAMATH_GPT_num_ordered_triples_l2340_234069


namespace NUMINAMATH_GPT_total_students_sampled_l2340_234075

theorem total_students_sampled :
  ∀ (seniors juniors freshmen sampled_seniors sampled_juniors sampled_freshmen total_students : ℕ),
    seniors = 1000 →
    juniors = 1200 →
    freshmen = 1500 →
    sampled_freshmen = 75 →
    sampled_seniors = seniors * (sampled_freshmen / freshmen) →
    sampled_juniors = juniors * (sampled_freshmen / freshmen) →
    total_students = sampled_seniors + sampled_juniors + sampled_freshmen →
    total_students = 185 :=
by
sorry

end NUMINAMATH_GPT_total_students_sampled_l2340_234075


namespace NUMINAMATH_GPT_recipe_serves_correctly_l2340_234092

theorem recipe_serves_correctly:
  ∀ (cream_fat_per_cup : ℝ) (cream_amount_cup : ℝ) (fat_per_serving : ℝ) (total_servings: ℝ),
    cream_fat_per_cup = 88 →
    cream_amount_cup = 0.5 →
    fat_per_serving = 11 →
    total_servings = (cream_amount_cup * cream_fat_per_cup) / fat_per_serving →
    total_servings = 4 :=
by
  intros cream_fat_per_cup cream_amount_cup fat_per_serving total_servings
  intros hcup hccup hfserv htserv
  sorry

end NUMINAMATH_GPT_recipe_serves_correctly_l2340_234092


namespace NUMINAMATH_GPT_tan_theta_l2340_234006

theorem tan_theta (θ : ℝ) (x y : ℝ) (hx : x = - (Real.sqrt 3) / 2) (hy : y = 1 / 2) (h_terminal : True) : 
  Real.tan θ = - (Real.sqrt 3) / 3 :=
sorry

end NUMINAMATH_GPT_tan_theta_l2340_234006


namespace NUMINAMATH_GPT_find_multiplier_l2340_234065

theorem find_multiplier :
  ∀ (x n : ℝ), (x = 5) → (x * n = (16 - x) + 4) → n = 3 :=
by
  intros x n hx heq
  sorry

end NUMINAMATH_GPT_find_multiplier_l2340_234065


namespace NUMINAMATH_GPT_max_a_value_l2340_234071

theorem max_a_value :
  ∀ (a x : ℝ), 
  (x - 1) * x - (a - 2) * (a + 1) ≥ 1 → a ≤ 3 / 2 := sorry

end NUMINAMATH_GPT_max_a_value_l2340_234071


namespace NUMINAMATH_GPT_committee_selection_correct_l2340_234093

def num_ways_to_choose_committee : ℕ :=
  let total_people := 10
  let president_ways := total_people
  let vp_ways := total_people - 1
  let remaining_people := total_people - 2
  let committee_ways := Nat.choose remaining_people 2
  president_ways * vp_ways * committee_ways

theorem committee_selection_correct :
  num_ways_to_choose_committee = 2520 :=
by
  sorry

end NUMINAMATH_GPT_committee_selection_correct_l2340_234093


namespace NUMINAMATH_GPT_total_blankets_collected_l2340_234045

theorem total_blankets_collected : 
  let original_members := 15
  let new_members := 5
  let blankets_per_original_member_first_day := 2
  let blankets_per_original_member_second_day := 2
  let blankets_per_new_member_second_day := 4
  let tripled_first_day_total := 3
  let blankets_school_third_day := 22
  let blankets_online_third_day := 30
  let first_day_blankets := original_members * blankets_per_original_member_first_day
  let second_day_original_members_blankets := original_members * blankets_per_original_member_second_day
  let second_day_new_members_blankets := new_members * blankets_per_new_member_second_day
  let second_day_additional_blankets := tripled_first_day_total * first_day_blankets
  let second_day_blankets := second_day_original_members_blankets + second_day_new_members_blankets + second_day_additional_blankets
  let third_day_blankets := blankets_school_third_day + blankets_online_third_day
  let total_blankets := first_day_blankets + second_day_blankets + third_day_blankets
  -- Prove that
  total_blankets = 222 :=
by 
  sorry

end NUMINAMATH_GPT_total_blankets_collected_l2340_234045


namespace NUMINAMATH_GPT_pizza_topping_cost_l2340_234032

/- 
   Given:
   1. Ruby ordered 3 pizzas.
   2. Each pizza costs $10.00.
   3. The total number of toppings were 4.
   4. Ruby added a $5.00 tip to the order.
   5. The total cost of the order, including tip, was $39.00.

   Prove: The cost per topping is $1.00.
-/
theorem pizza_topping_cost (cost_per_pizza : ℝ) (total_pizzas : ℕ) (tip : ℝ) (total_cost : ℝ) 
    (total_toppings : ℕ) (x : ℝ) : 
    cost_per_pizza = 10 → total_pizzas = 3 → tip = 5 → total_cost = 39 → total_toppings = 4 → 
    total_cost = cost_per_pizza * total_pizzas + x * total_toppings + tip →
    x = 1 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_pizza_topping_cost_l2340_234032


namespace NUMINAMATH_GPT_tree_height_l2340_234055

theorem tree_height (B h : ℕ) (H : ℕ) (h_eq : h = 16) (B_eq : B = 12) (L : ℕ) (L_def : L ^ 2 = B ^ 2 + h ^ 2) (H_def : H = h + L) :
    H = 36 := by
  -- We do not need to provide the proof steps as per the instructions
  sorry

end NUMINAMATH_GPT_tree_height_l2340_234055


namespace NUMINAMATH_GPT_bob_total_distance_traveled_over_six_days_l2340_234022

theorem bob_total_distance_traveled_over_six_days (x : ℤ) (hx1 : 3 ≤ x) (hx2 : x % 3 = 0):
  (90 / x + 90 / (x + 3) + 90 / (x + 6) + 90 / (x + 9) + 90 / (x + 12) + 90 / (x + 15) : ℝ) = 73.5 :=
by
  sorry

end NUMINAMATH_GPT_bob_total_distance_traveled_over_six_days_l2340_234022


namespace NUMINAMATH_GPT_complement_A_in_U_l2340_234005

def U : Set ℤ := {-2, -1, 0, 1, 2}
def A : Set ℤ := {x | x^2 + x - 2 < 0}

theorem complement_A_in_U :
  (U \ A) = {-2, 1, 2} :=
by 
  -- proof will be done here
  sorry

end NUMINAMATH_GPT_complement_A_in_U_l2340_234005


namespace NUMINAMATH_GPT_largest_among_four_theorem_l2340_234031

noncomputable def largest_among_four (a b : ℝ) (h1 : 0 < a ∧ a < b) (h2 : a + b = 1) : Prop :=
  (a^2 + b^2 > 1) ∧ (a^2 + b^2 > 2 * a * b) ∧ (a^2 + b^2 > a)

theorem largest_among_four_theorem (a b : ℝ) (h1 : 0 < a ∧ a < b) (h2 : a + b = 1) :
  largest_among_four a b h1 h2 :=
sorry

end NUMINAMATH_GPT_largest_among_four_theorem_l2340_234031


namespace NUMINAMATH_GPT_fraction_meaningful_l2340_234034

theorem fraction_meaningful (x : ℝ) : (x-5) ≠ 0 ↔ (1 / (x - 5)) = (1 / (x - 5)) := 
by 
  sorry

end NUMINAMATH_GPT_fraction_meaningful_l2340_234034


namespace NUMINAMATH_GPT_least_remaining_marbles_l2340_234046

/-- 
There are 60 identical marbles forming a tetrahedral pile.
The formula for the number of marbles in a tetrahedral pile up to the k-th level is given by:
∑_(i=1)^k (i * (i + 1)) / 6 = k * (k + 1) * (k + 2) / 6.

We must show that the least number of remaining marbles when 60 marbles are used to form the pile is 4.
-/
theorem least_remaining_marbles : ∃ k : ℕ, (60 - k * (k + 1) * (k + 2) / 6) = 4 :=
by
  sorry

end NUMINAMATH_GPT_least_remaining_marbles_l2340_234046


namespace NUMINAMATH_GPT_proposition2_and_4_correct_l2340_234050

theorem proposition2_and_4_correct (a b : ℝ) : 
  (a > b ∧ b > 0 → a^2 - a > b^2 - b) ∧ 
  (a > 0 ∧ b > 0 ∧ 2 * a + b = 1 → a^2 + b^2 = 9) :=
by
  sorry

end NUMINAMATH_GPT_proposition2_and_4_correct_l2340_234050


namespace NUMINAMATH_GPT_mary_brought_stickers_l2340_234023

theorem mary_brought_stickers (friends_stickers : Nat) (other_stickers : Nat) (left_stickers : Nat) 
                              (total_students : Nat) (num_friends : Nat) (stickers_per_friend : Nat) 
                              (stickers_per_other_student : Nat) :
  friends_stickers = num_friends * stickers_per_friend →
  left_stickers = 8 →
  total_students = 17 →
  num_friends = 5 →
  stickers_per_friend = 4 →
  stickers_per_other_student = 2 →
  other_stickers = (total_students - 1 - num_friends) * stickers_per_other_student →
  (friends_stickers + other_stickers + left_stickers) = 50 :=
by
  intros
  sorry

end NUMINAMATH_GPT_mary_brought_stickers_l2340_234023


namespace NUMINAMATH_GPT_total_salmon_now_l2340_234000

def initial_salmon : ℕ := 500

def increase_factor : ℕ := 10

theorem total_salmon_now : initial_salmon * increase_factor = 5000 := by
  sorry

end NUMINAMATH_GPT_total_salmon_now_l2340_234000


namespace NUMINAMATH_GPT_factor_quadratic_l2340_234084

theorem factor_quadratic (x : ℝ) : 
  x^2 + 6 * x = 1 → (x + 3)^2 = 10 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_factor_quadratic_l2340_234084


namespace NUMINAMATH_GPT_instantaneous_velocity_at_3_l2340_234088

noncomputable def motion_equation (t : ℝ) : ℝ := 1 - t + t^2

theorem instantaneous_velocity_at_3 :
  (deriv (motion_equation) 3 = 5) :=
by
  sorry

end NUMINAMATH_GPT_instantaneous_velocity_at_3_l2340_234088


namespace NUMINAMATH_GPT_triangle_existence_l2340_234061

theorem triangle_existence (n : ℕ) (h : 2 * n > 0) (segments : Finset (ℕ × ℕ))
  (h_segments : segments.card = n^2 + 1)
  (points_in_segment : ∀ {a b : ℕ}, (a, b) ∈ segments → a < 2 * n ∧ b < 2 * n) :
  ∃ x y z, x < 2 * n ∧ y < 2 * n ∧ z < 2 * n ∧ (x ≠ y ∧ y ≠ z ∧ z ≠ x) ∧
  ((x, y) ∈ segments ∨ (y, x) ∈ segments) ∧
  ((y, z) ∈ segments ∨ (z, y) ∈ segments) ∧
  ((z, x) ∈ segments ∨ (x, z) ∈ segments) :=
by
  sorry

end NUMINAMATH_GPT_triangle_existence_l2340_234061


namespace NUMINAMATH_GPT_annual_production_2010_l2340_234062

-- Defining the parameters
variables (a x : ℝ)

-- Define the growth formula
def annual_growth (initial : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  initial * (1 + rate)^years

-- The statement we need to prove
theorem annual_production_2010 :
  annual_growth a x 5 = a * (1 + x) ^ 5 :=
by
  sorry

end NUMINAMATH_GPT_annual_production_2010_l2340_234062


namespace NUMINAMATH_GPT_color_dot_figure_l2340_234051

-- Definitions reflecting the problem conditions
def num_colors : ℕ := 3
def first_triangle_coloring_ways : ℕ := 6
def subsequent_triangle_coloring_ways : ℕ := 3
def additional_dot_coloring_ways : ℕ := 2

-- The theorem stating the required proof
theorem color_dot_figure : first_triangle_coloring_ways * 
                           subsequent_triangle_coloring_ways * 
                           subsequent_triangle_coloring_ways * 
                           additional_dot_coloring_ways = 108 := by
sorry

end NUMINAMATH_GPT_color_dot_figure_l2340_234051


namespace NUMINAMATH_GPT_gravitational_force_on_asteroid_l2340_234070

theorem gravitational_force_on_asteroid :
  ∃ (k : ℝ), ∃ (f : ℝ), 
  (∀ (d : ℝ), f = k / d^2) ∧
  (d = 5000 → f = 700) →
  (∃ (f_asteroid : ℝ), f_asteroid = k / 300000^2 ∧ f_asteroid = 7 / 36) :=
sorry

end NUMINAMATH_GPT_gravitational_force_on_asteroid_l2340_234070


namespace NUMINAMATH_GPT_problem_statement_l2340_234038

theorem problem_statement (x θ : ℝ) (h : Real.logb 2 x + Real.cos θ = 2) : |x - 8| + |x + 2| = 10 :=
sorry

end NUMINAMATH_GPT_problem_statement_l2340_234038


namespace NUMINAMATH_GPT_reciprocal_of_sum_of_fractions_l2340_234003

theorem reciprocal_of_sum_of_fractions :
  (1 / (1 / 4 + 1 / 6)) = 12 / 5 :=
by
  sorry

end NUMINAMATH_GPT_reciprocal_of_sum_of_fractions_l2340_234003


namespace NUMINAMATH_GPT_sum_of_solutions_l2340_234015

theorem sum_of_solutions (y : ℤ) (x1 x2 : ℤ) (h1 : y = 8) (h2 : x1^2 + y^2 = 145) (h3 : x2^2 + y^2 = 145) : x1 + x2 = 0 := by
  sorry

end NUMINAMATH_GPT_sum_of_solutions_l2340_234015


namespace NUMINAMATH_GPT_find_a3_l2340_234097

-- Define the geometric sequence and its properties.
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- Given conditions
variables (a : ℕ → ℝ) (q : ℝ)
variable (h_GeoSeq : is_geometric_sequence a q)
variable (h_a1 : a 1 = 1)
variable (h_a5 : a 5 = 9)

-- Define what we need to prove
theorem find_a3 : a 3 = 3 :=
sorry

end NUMINAMATH_GPT_find_a3_l2340_234097


namespace NUMINAMATH_GPT_find_S20_l2340_234085

variable {α : Type*} [AddCommGroup α] [Module ℝ α]
variable (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Conditions
axiom sum_first_n_terms (n : ℕ) : S n = n * (a 1 + a n) / 2
axiom points_collinear (A B C O : α) : Collinear ℝ ({A, B, C} : Set α) ∧ O = 0
axiom vector_relationship (A B C O : α) : O = 0 → C = (a 12) • A + (a 9) • B
axiom line_not_through_origin (A B O : α) : ¬Collinear ℝ ({O, A, B} : Set α)

-- Question: To find S 20
theorem find_S20 (A B C O : α) (h_collinear : Collinear ℝ ({A, B, C} : Set α)) 
  (h_vector : O = 0 → C = (a 12) • A + (a 9) • B) 
  (h_origin : O = 0)
  (h_not_through_origin : ¬Collinear ℝ ({O, A, B} : Set α)) : 
  S 20 = 10 := by
  sorry

end NUMINAMATH_GPT_find_S20_l2340_234085


namespace NUMINAMATH_GPT_range_of_a_l2340_234076

theorem range_of_a {a : ℝ} : 
  (∃ x : ℝ, (1 / 2 < x ∧ x < 3) ∧ (x ^ 2 - a * x + 1 = 0)) ↔ (2 ≤ a ∧ a < 10 / 3) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2340_234076


namespace NUMINAMATH_GPT_binomial_divisible_by_prime_l2340_234014

-- Define the conditions: p is prime and 0 < k < p
variables (p k : ℕ)
variable (hp : Nat.Prime p)
variable (hk : 0 < k ∧ k < p)

-- State that the binomial coefficient \(\binom{p}{k}\) is divisible by \( p \)
theorem binomial_divisible_by_prime
  (p k : ℕ) (hp : Nat.Prime p) (hk : 0 < k ∧ k < p) :
  p ∣ Nat.choose p k :=
by
  sorry

end NUMINAMATH_GPT_binomial_divisible_by_prime_l2340_234014


namespace NUMINAMATH_GPT_trig_identity_l2340_234096

open Real

theorem trig_identity (θ : ℝ) (h : tan θ = 2) :
  ((sin θ + cos θ) * cos (2 * θ)) / sin θ = -9 / 10 :=
sorry

end NUMINAMATH_GPT_trig_identity_l2340_234096


namespace NUMINAMATH_GPT_minimum_value_l2340_234053

theorem minimum_value (x : ℝ) (h : x > 0) :
  x^3 + 12*x + 81 / x^4 = 24 := 
sorry

end NUMINAMATH_GPT_minimum_value_l2340_234053


namespace NUMINAMATH_GPT_solve_for_y_in_terms_of_x_l2340_234083

theorem solve_for_y_in_terms_of_x (x y : ℝ) (h : 2 * x - 7 * y = 5) : y = (2 * x - 5) / 7 :=
sorry

end NUMINAMATH_GPT_solve_for_y_in_terms_of_x_l2340_234083


namespace NUMINAMATH_GPT_correct_operation_l2340_234043

theorem correct_operation (a b : ℝ) : 
  ¬(a^2 + a^3 = a^5) ∧ ¬((a^2)^3 = a^8) ∧ (a^3 / a^2 = a) ∧ ¬((a - b)^2 = a^2 - b^2) := 
by {
  sorry
}

end NUMINAMATH_GPT_correct_operation_l2340_234043


namespace NUMINAMATH_GPT_ticket_price_divisor_l2340_234041

def is_divisor (a b : ℕ) : Prop := ∃ k, b = k * a

def GCD (a b : ℕ) := Nat.gcd a b

theorem ticket_price_divisor :
  let total7 := 70
  let total8 := 98
  let y := 4
  is_divisor (GCD total7 total8) y :=
by
  sorry

end NUMINAMATH_GPT_ticket_price_divisor_l2340_234041


namespace NUMINAMATH_GPT_Kyler_wins_1_game_l2340_234079

theorem Kyler_wins_1_game
  (peter_wins : ℕ)
  (peter_losses : ℕ)
  (emma_wins : ℕ)
  (emma_losses : ℕ)
  (kyler_losses : ℕ)
  (total_games : ℕ)
  (kyler_wins : ℕ)
  (htotal : total_games = (peter_wins + peter_losses + emma_wins + emma_losses + kyler_wins + kyler_losses) / 2)
  (hpeter : peter_wins = 4 ∧ peter_losses = 2)
  (hemma : emma_wins = 3 ∧ emma_losses = 3)
  (hkyler_losses : kyler_losses = 3)
  (htotal_wins_losses : total_games = peter_wins + emma_wins + kyler_wins) : kyler_wins = 1 :=
by
  sorry

end NUMINAMATH_GPT_Kyler_wins_1_game_l2340_234079


namespace NUMINAMATH_GPT_costPerUse_l2340_234027

-- Definitions based on conditions
def heatingPadCost : ℝ := 30
def usesPerWeek : ℕ := 3
def totalWeeks : ℕ := 2

-- Calculate the total number of uses
def totalUses : ℕ := usesPerWeek * totalWeeks

-- The amount spent per use
theorem costPerUse : heatingPadCost / totalUses = 5 := by
  sorry

end NUMINAMATH_GPT_costPerUse_l2340_234027


namespace NUMINAMATH_GPT_herman_days_per_week_l2340_234063

-- Defining the given conditions as Lean definitions
def total_meals : ℕ := 4
def cost_per_meal : ℕ := 4
def total_weeks : ℕ := 16
def total_cost : ℕ := 1280

-- Calculating derived facts based on given conditions
def cost_per_day : ℕ := total_meals * cost_per_meal
def cost_per_week : ℕ := total_cost / total_weeks

-- Our main theorem that states Herman buys breakfast combos 5 days per week
theorem herman_days_per_week : cost_per_week / cost_per_day = 5 :=
by
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_herman_days_per_week_l2340_234063


namespace NUMINAMATH_GPT_marbles_in_larger_bottle_l2340_234011

theorem marbles_in_larger_bottle 
  (small_bottle_volume : ℕ := 20)
  (small_bottle_marbles : ℕ := 40)
  (larger_bottle_volume : ℕ := 60) :
  (small_bottle_marbles / small_bottle_volume) * larger_bottle_volume = 120 := 
by
  sorry

end NUMINAMATH_GPT_marbles_in_larger_bottle_l2340_234011


namespace NUMINAMATH_GPT_sum_2_75_0_003_0_158_l2340_234094

theorem sum_2_75_0_003_0_158 : 2.75 + 0.003 + 0.158 = 2.911 :=
by
  -- Lean proof goes here  
  sorry

end NUMINAMATH_GPT_sum_2_75_0_003_0_158_l2340_234094


namespace NUMINAMATH_GPT_problem_solution_l2340_234099

theorem problem_solution (a b c d e : ℤ) (h : (x - 3)^4 = ax^4 + bx^3 + cx^2 + dx + e) :
  b + c + d + e = 15 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l2340_234099


namespace NUMINAMATH_GPT_find_number_l2340_234067

theorem find_number (n : ℝ) : (2629.76 / n = 528.0642570281125) → n = 4.979 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_number_l2340_234067


namespace NUMINAMATH_GPT_range_of_expressions_l2340_234074

theorem range_of_expressions (x y : ℝ) (h1 : 30 < x ∧ x < 42) (h2 : 16 < y ∧ y < 24) :
  46 < x + y ∧ x + y < 66 ∧ -18 < x - 2 * y ∧ x - 2 * y < 10 ∧ (5 / 4) < (x / y) ∧ (x / y) < (21 / 8) :=
sorry

end NUMINAMATH_GPT_range_of_expressions_l2340_234074


namespace NUMINAMATH_GPT_unique_midpoints_are_25_l2340_234021

/-- Define the properties of a parallelogram with marked points such as vertices, midpoints of sides, and intersection point of diagonals --/
structure Parallelogram :=
(vertices : Set ℝ)
(midpoints : Set ℝ)
(diagonal_intersection : ℝ)

def congruent_parallelograms (P P' : Parallelogram) : Prop :=
  P.vertices = P'.vertices ∧ P.midpoints = P'.midpoints ∧ P.diagonal_intersection = P'.diagonal_intersection

def unique_midpoints_count (P P' : Parallelogram) : ℕ := sorry

theorem unique_midpoints_are_25
  (P P' : Parallelogram)
  (h_congruent : congruent_parallelograms P P') :
  unique_midpoints_count P P' = 25 := sorry

end NUMINAMATH_GPT_unique_midpoints_are_25_l2340_234021


namespace NUMINAMATH_GPT_cube_minus_self_divisible_by_10_l2340_234060

theorem cube_minus_self_divisible_by_10 (k : ℤ) : 10 ∣ ((5 * k) ^ 3 - 5 * k) :=
by sorry

end NUMINAMATH_GPT_cube_minus_self_divisible_by_10_l2340_234060


namespace NUMINAMATH_GPT_magician_card_pairs_l2340_234019

theorem magician_card_pairs:
  ∃ (f : Fin 65 → Fin 65 × Fin 65), 
  (∀ m n : Fin 65, ∃ k l : Fin 65, (f m = (k, l) ∧ f n = (l, k))) := 
sorry

end NUMINAMATH_GPT_magician_card_pairs_l2340_234019


namespace NUMINAMATH_GPT_solve_for_x_l2340_234044

theorem solve_for_x (x : ℝ) (h : 3 * x = 16 - x + 4) : x = 5 := 
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l2340_234044


namespace NUMINAMATH_GPT_beth_gave_away_54_crayons_l2340_234010

-- Define the initial number of crayons
def initialCrayons : ℕ := 106

-- Define the number of crayons left
def remainingCrayons : ℕ := 52

-- Define the number of crayons given away
def crayonsGiven (initial remaining: ℕ) : ℕ := initial - remaining

-- The goal is to prove that Beth gave away 54 crayons
theorem beth_gave_away_54_crayons : crayonsGiven initialCrayons remainingCrayons = 54 :=
by
  sorry

end NUMINAMATH_GPT_beth_gave_away_54_crayons_l2340_234010


namespace NUMINAMATH_GPT_song_book_cost_correct_l2340_234057

/-- Define the constants for the problem. -/
def clarinet_cost : ℝ := 130.30
def pocket_money : ℝ := 12.32
def total_spent : ℝ := 141.54

/-- Prove the cost of the song book. -/
theorem song_book_cost_correct :
  (total_spent - clarinet_cost) = 11.24 :=
by
  sorry

end NUMINAMATH_GPT_song_book_cost_correct_l2340_234057


namespace NUMINAMATH_GPT_alice_paper_cranes_l2340_234087

theorem alice_paper_cranes (T : ℕ)
  (h1 : T / 2 - T / 10 = 400) : T = 1000 :=
sorry

end NUMINAMATH_GPT_alice_paper_cranes_l2340_234087


namespace NUMINAMATH_GPT_power_seven_evaluation_l2340_234012

theorem power_seven_evaluation (a b : ℝ) (h : a = (7 : ℝ)^(1/4) ∧ b = (7 : ℝ)^(1/7)) : 
  a / b = (7 : ℝ)^(3/28) :=
  sorry

end NUMINAMATH_GPT_power_seven_evaluation_l2340_234012


namespace NUMINAMATH_GPT_exists_t_perpendicular_min_dot_product_coordinates_l2340_234008

-- Definitions of points
def OA : ℝ × ℝ := (5, 1)
def OB : ℝ × ℝ := (1, 7)
def OC : ℝ × ℝ := (4, 2)

-- Definition of vector OM depending on t
def OM (t : ℝ) : ℝ × ℝ := (4 * t, 2 * t)

-- Definition of vector MA and MB
def MA (t : ℝ) : ℝ × ℝ := (5 - 4 * t, 1 - 2 * t)
def MB (t : ℝ) : ℝ × ℝ := (1 - 4 * t, 7 - 2 * t)

-- Dot product of vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Proof that there exists a t such that MA ⊥ MB
theorem exists_t_perpendicular : ∃ t : ℝ, dot_product (MA t) (MB t) = 0 :=
by 
  sorry

-- Proof that coordinates of M minimizing MA ⋅ MB is (4, 2)
theorem min_dot_product_coordinates : ∃ t : ℝ, t = 1 ∧ (OM t) = (4, 2) :=
by
  sorry

end NUMINAMATH_GPT_exists_t_perpendicular_min_dot_product_coordinates_l2340_234008


namespace NUMINAMATH_GPT_range_of_a_l2340_234082

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 1 ≤ x → x ≤ 4 → (a * x^2 - 2 * x + 2) > 0) ↔ (a > 1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2340_234082


namespace NUMINAMATH_GPT_hall_length_l2340_234068

theorem hall_length (L h : ℝ) (width volume : ℝ) 
  (h_width : width = 6) 
  (h_volume : L * width * h = 108) 
  (h_area : 12 * L = 2 * L * h + 12 * h) : 
  L = 6 := 
  sorry

end NUMINAMATH_GPT_hall_length_l2340_234068


namespace NUMINAMATH_GPT_average_visitors_on_Sundays_l2340_234078

theorem average_visitors_on_Sundays (S : ℕ) 
  (h1 : 30 % 7 = 2)  -- The month begins with a Sunday
  (h2 : 25 = 30 - 5)  -- The month has 25 non-Sundays
  (h3 : (120 * 25) = 3000) -- Total visitors on non-Sundays
  (h4 : (125 * 30) = 3750) -- Total visitors for the month
  (h5 : 5 * 30 > 0) -- There are a positive number of Sundays
  : S = 150 :=
by
  sorry

end NUMINAMATH_GPT_average_visitors_on_Sundays_l2340_234078


namespace NUMINAMATH_GPT_oil_amount_to_add_l2340_234016

variable (a b : ℝ)
variable (h1 : a = 0.16666666666666666)
variable (h2 : b = 0.8333333333333334)

theorem oil_amount_to_add (a b : ℝ) (h1 : a = 0.16666666666666666) (h2 : b = 0.8333333333333334) : 
  b - a = 0.6666666666666667 := by
  rw [h1, h2]
  norm_num
  sorry

end NUMINAMATH_GPT_oil_amount_to_add_l2340_234016


namespace NUMINAMATH_GPT_Joe_total_income_l2340_234080

theorem Joe_total_income : 
  (∃ I : ℝ, 0.1 * 1000 + 0.2 * 3000 + 0.3 * (I - 500 - 4000) = 848 ∧ I - 500 > 4000) → I = 4993.33 :=
by
  sorry

end NUMINAMATH_GPT_Joe_total_income_l2340_234080


namespace NUMINAMATH_GPT_probability_at_least_seven_heads_or_tails_l2340_234089

open Nat

-- Define the probability of getting at least seven heads or tails in eight coin flips
theorem probability_at_least_seven_heads_or_tails :
  let total_outcomes := 2^8
  let favorable_outcomes := (choose 8 7) + (choose 8 7) + 1 + 1
  let probability := (favorable_outcomes : ℝ) / total_outcomes
  probability = 9 / 128 := by
  sorry

end NUMINAMATH_GPT_probability_at_least_seven_heads_or_tails_l2340_234089


namespace NUMINAMATH_GPT_hot_dogs_per_pack_l2340_234086

-- Define the givens / conditions
def total_hot_dogs : ℕ := 36
def buns_pack_size : ℕ := 9
def same_quantity (h : ℕ) (b : ℕ) := h = b

-- State the theorem to be proven
theorem hot_dogs_per_pack : ∃ h : ℕ, (total_hot_dogs / h = buns_pack_size) ∧ same_quantity (total_hot_dogs / h) (total_hot_dogs / buns_pack_size) := 
sorry

end NUMINAMATH_GPT_hot_dogs_per_pack_l2340_234086


namespace NUMINAMATH_GPT_determine_x_l2340_234020

theorem determine_x (x : ℝ) (A B : Set ℝ) (H1 : A = {-1, 0}) (H2 : B = {0, 1, x + 2}) (H3 : A ⊆ B) : x = -3 :=
sorry

end NUMINAMATH_GPT_determine_x_l2340_234020


namespace NUMINAMATH_GPT_seq_problem_l2340_234098

theorem seq_problem (a : ℕ → ℚ) (d : ℚ) (h_arith : ∀ n : ℕ, a (n + 1) = a n + d )
 (h1 : a 1 = 2)
 (h_geom : (a 1 - 1) * (a 5 + 5) = (a 3)^2) :
  a 2017 = 1010 := 
sorry

end NUMINAMATH_GPT_seq_problem_l2340_234098


namespace NUMINAMATH_GPT_find_angle_F_l2340_234024

-- Define the angles of the triangle
variables (D E F : ℝ)

-- Define the conditions given in the problem
def angle_conditions (D E F : ℝ) : Prop :=
  (D = 3 * E) ∧ (E = 18) ∧ (D + E + F = 180)

-- The theorem to prove that angle F is 108 degrees
theorem find_angle_F (D E F : ℝ) (h : angle_conditions D E F) : 
  F = 108 :=
by
  -- The proof body is omitted
  sorry

end NUMINAMATH_GPT_find_angle_F_l2340_234024


namespace NUMINAMATH_GPT_min_length_PQ_l2340_234072

noncomputable def minimum_length (a : ℝ) : ℝ :=
  let x := 2 * a
  let y := a + 2
  let d := |2 * 2 - 2 * 0 + 4| / Real.sqrt (1^2 + (-2)^2)
  let r := Real.sqrt 5
  d - r

theorem min_length_PQ : ∀ (a : ℝ), P ∈ {P : ℝ × ℝ | (P.1 - 2)^2 + P.2^2 = 5} ∧ Q = (2 * a, a + 2) →
  minimum_length a = 3 * Real.sqrt 5 / 5 :=
by
  intro a
  intro h
  rcases h with ⟨hP, hQ⟩
  sorry

end NUMINAMATH_GPT_min_length_PQ_l2340_234072


namespace NUMINAMATH_GPT_product_of_distinct_integers_l2340_234064

def is2008thPower (n : ℕ) : Prop := ∃ k : ℕ, n = k ^ 2008

theorem product_of_distinct_integers {x y z : ℕ} (h1 : x ≠ y) (h2 : y ≠ z) (h3 : z ≠ x)
  (h4 : y = (x + z) / 2) (h5 : x > 0) (h6 : y > 0) (h7 : z > 0) 
  : is2008thPower (x * y * z) :=
  sorry

end NUMINAMATH_GPT_product_of_distinct_integers_l2340_234064


namespace NUMINAMATH_GPT_problem_l2340_234036

theorem problem : 
  let b := 2 ^ 51
  let c := 4 ^ 25
  b > c :=
by 
  let b := 2 ^ 51
  let c := 4 ^ 25
  sorry

end NUMINAMATH_GPT_problem_l2340_234036


namespace NUMINAMATH_GPT_Euler_theorem_l2340_234066

theorem Euler_theorem {m a : ℕ} (hm : m ≥ 1) (h_gcd : Nat.gcd a m = 1) : a ^ Nat.totient m ≡ 1 [MOD m] :=
by
  sorry

end NUMINAMATH_GPT_Euler_theorem_l2340_234066


namespace NUMINAMATH_GPT_sum_series_l2340_234033

noncomputable def b : ℕ → ℝ
| 0     => 2
| 1     => 2
| (n+2) => b (n+1) + b n

theorem sum_series : (∑' n, b n / 3^(n+1)) = 1 / 3 := by
  sorry

end NUMINAMATH_GPT_sum_series_l2340_234033


namespace NUMINAMATH_GPT_existence_of_five_regular_polyhedra_l2340_234056

def regular_polyhedron (n m : ℕ) : Prop :=
  n ≥ 3 ∧ m ≥ 3 ∧ (2 / m + 2 / n > 1)

theorem existence_of_five_regular_polyhedra :
  ∃ (n m : ℕ), regular_polyhedron n m → 
    (n = 3 ∧ m = 3 ∨ 
     n = 4 ∧ m = 3 ∨ 
     n = 3 ∧ m = 4 ∨ 
     n = 5 ∧ m = 3 ∨ 
     n = 3 ∧ m = 5) :=
by
  sorry

end NUMINAMATH_GPT_existence_of_five_regular_polyhedra_l2340_234056


namespace NUMINAMATH_GPT_billy_piles_l2340_234017

theorem billy_piles (Q D : ℕ) (h : 2 * Q + 3 * D = 20) :
  Q = 4 ∧ D = 4 :=
sorry

end NUMINAMATH_GPT_billy_piles_l2340_234017


namespace NUMINAMATH_GPT_pages_read_per_day_l2340_234013

-- Define the total number of pages in the book
def total_pages := 96

-- Define the number of days it took to finish the book
def number_of_days := 12

-- Define pages read per day for Charles
def pages_per_day := total_pages / number_of_days

-- Prove that the number of pages read per day is equal to 8
theorem pages_read_per_day : pages_per_day = 8 :=
by
  sorry

end NUMINAMATH_GPT_pages_read_per_day_l2340_234013


namespace NUMINAMATH_GPT_area_is_12_l2340_234042

-- Definitions based on conditions
def isosceles_triangle (a b m : ℝ) : Prop :=
  a = b ∧ m > 0 ∧ a > 0

def median (height base_length : ℝ) : Prop :=
  height > 0 ∧ base_length > 0

noncomputable def area_of_isosceles_triangle_with_given_median (a m : ℝ) : ℝ :=
  let base_half := Real.sqrt (a^2 - m^2)
  let base := 2 * base_half
  (1 / 2) * base * m

-- Prove that the area of the isosceles triangle is correct given conditions
theorem area_is_12 :
  ∀ (a m : ℝ), isosceles_triangle a a m → median m (2 * Real.sqrt (a^2 - m^2)) → area_of_isosceles_triangle_with_given_median a m = 12 := 
by
  intros a m hiso hmed
  sorry  -- Proof steps are omitted

end NUMINAMATH_GPT_area_is_12_l2340_234042


namespace NUMINAMATH_GPT_value_of_expression_l2340_234018

theorem value_of_expression : (5^2 - 4^2 + 3^2) = 18 := 
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l2340_234018


namespace NUMINAMATH_GPT_ratio_adidas_skechers_l2340_234025

-- Conditions
def total_expenditure : ℤ := 8000
def expenditure_adidas : ℤ := 600
def expenditure_clothes : ℤ := 2600
def expenditure_nike := 3 * expenditure_adidas

-- Calculation for sneakers
def total_sneakers := total_expenditure - expenditure_clothes
def expenditure_nike_adidas := expenditure_nike + expenditure_adidas
def expenditure_skechers := total_sneakers - expenditure_nike_adidas

-- Prove the ratio
theorem ratio_adidas_skechers (H1 : total_expenditure = 8000)
                              (H2 : expenditure_adidas = 600)
                              (H3 : expenditure_nike = 3 * expenditure_adidas)
                              (H4 : expenditure_clothes = 2600) :
  expenditure_adidas / expenditure_skechers = 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_ratio_adidas_skechers_l2340_234025


namespace NUMINAMATH_GPT_maximum_cards_l2340_234002

def total_budget : ℝ := 15
def card_cost : ℝ := 1.25
def transaction_fee : ℝ := 2
def desired_savings : ℝ := 3

theorem maximum_cards : ∃ n : ℕ, n ≤ 8 ∧ (card_cost * (n : ℝ) + transaction_fee ≤ total_budget - desired_savings) :=
by sorry

end NUMINAMATH_GPT_maximum_cards_l2340_234002


namespace NUMINAMATH_GPT_jason_fires_weapon_every_15_seconds_l2340_234073

theorem jason_fires_weapon_every_15_seconds
    (flame_duration_per_fire : ℕ)
    (total_flame_duration_per_minute : ℕ)
    (seconds_per_minute : ℕ)
    (h1 : flame_duration_per_fire = 5)
    (h2 : total_flame_duration_per_minute = 20)
    (h3 : seconds_per_minute = 60) :
    seconds_per_minute / (total_flame_duration_per_minute / flame_duration_per_fire) = 15 := 
by
  sorry

end NUMINAMATH_GPT_jason_fires_weapon_every_15_seconds_l2340_234073


namespace NUMINAMATH_GPT_best_fitting_model_l2340_234009

/-- A type representing the coefficient of determination of different models -/
def r_squared (m : ℕ) : ℝ :=
  match m with
  | 1 => 0.98
  | 2 => 0.80
  | 3 => 0.50
  | 4 => 0.25
  | _ => 0 -- An auxiliary value for invalid model numbers

/-- The best fitting model is the one with the highest r_squared value --/
theorem best_fitting_model : r_squared 1 = max (r_squared 1) (max (r_squared 2) (max (r_squared 3) (r_squared 4))) :=
by
  sorry

end NUMINAMATH_GPT_best_fitting_model_l2340_234009


namespace NUMINAMATH_GPT_contractor_fine_per_absent_day_l2340_234049

theorem contractor_fine_per_absent_day :
  ∃ x : ℝ, (∀ (total_days absent_days worked_days earnings_per_day total_earnings : ℝ),
   total_days = 30 →
   earnings_per_day = 25 →
   total_earnings = 490 →
   absent_days = 8 →
   worked_days = total_days - absent_days →
   25 * worked_days - absent_days * x = total_earnings
  ) → x = 7.5 :=
by
  existsi 7.5
  intros
  sorry

end NUMINAMATH_GPT_contractor_fine_per_absent_day_l2340_234049


namespace NUMINAMATH_GPT_small_mold_radius_l2340_234037

theorem small_mold_radius (r : ℝ) (n : ℝ) (s : ℝ) :
    r = 2 ∧ n = 8 ∧ (1 / 2) * (2 / 3) * Real.pi * r^3 = (8 * (2 / 3) * Real.pi * s^3) → s = 1 :=
by
  sorry

end NUMINAMATH_GPT_small_mold_radius_l2340_234037


namespace NUMINAMATH_GPT_line_intersects_circle_l2340_234077

theorem line_intersects_circle
  (a b r : ℝ)
  (r_nonzero : r ≠ 0)
  (h_outside : a^2 + b^2 > r^2) :
  ∃ x y : ℝ, (x^2 + y^2 = r^2) ∧ (a * x + b * y = r^2) :=
sorry

end NUMINAMATH_GPT_line_intersects_circle_l2340_234077


namespace NUMINAMATH_GPT_frequency_of_largest_rectangle_area_l2340_234030

theorem frequency_of_largest_rectangle_area (a : ℕ → ℝ) (sample_size : ℕ)
    (h_geom : ∀ n, a (n + 1) = 2 * a n) (h_sum : a 0 + a 1 + a 2 + a 3 = 1)
    (h_sample : sample_size = 300) : 
    sample_size * a 3 = 160 := by
  sorry

end NUMINAMATH_GPT_frequency_of_largest_rectangle_area_l2340_234030


namespace NUMINAMATH_GPT_gifts_left_l2340_234004

variable (initial_gifts : ℕ)
variable (gifts_sent : ℕ)

theorem gifts_left (h_initial : initial_gifts = 77) (h_sent : gifts_sent = 66) : initial_gifts - gifts_sent = 11 := by
  sorry

end NUMINAMATH_GPT_gifts_left_l2340_234004


namespace NUMINAMATH_GPT_intersection_A_B_l2340_234007

-- Definition of sets A and B
def A := {x : ℝ | x > 2}
def B := { x : ℝ | (x - 1) * (x - 3) < 0 }

-- Claim that A ∩ B = {x : ℝ | 2 < x < 3}
theorem intersection_A_B :
  {x : ℝ | x ∈ A ∧ x ∈ B} = {x : ℝ | 2 < x ∧ x < 3} :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l2340_234007


namespace NUMINAMATH_GPT_charlie_fewer_games_than_dana_l2340_234090

theorem charlie_fewer_games_than_dana
  (P D C Ph : ℕ)
  (h1 : P = D + 5)
  (h2 : C < D)
  (h3 : Ph = C + 3)
  (h4 : Ph = 12)
  (h5 : P = Ph + 4) :
  D - C = 2 :=
by
  sorry

end NUMINAMATH_GPT_charlie_fewer_games_than_dana_l2340_234090


namespace NUMINAMATH_GPT_largest_common_remainder_l2340_234054

theorem largest_common_remainder : 
  ∃ n r, 2013 ≤ n ∧ n ≤ 2156 ∧ (n % 5 = r) ∧ (n % 11 = r) ∧ (n % 13 = r) ∧ (r = 4) := 
by
  sorry

end NUMINAMATH_GPT_largest_common_remainder_l2340_234054


namespace NUMINAMATH_GPT_equivalence_a_gt_b_and_inv_a_lt_inv_b_l2340_234028

variable {a b : ℝ}

theorem equivalence_a_gt_b_and_inv_a_lt_inv_b (h : a * b > 0) : 
  (a > b) ↔ (1 / a < 1 / b) := 
sorry

end NUMINAMATH_GPT_equivalence_a_gt_b_and_inv_a_lt_inv_b_l2340_234028
