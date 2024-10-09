import Mathlib

namespace amount_daria_needs_l1212_121204

theorem amount_daria_needs (ticket_cost : ℕ) (total_tickets : ℕ) (current_money : ℕ) (needed_money : ℕ) 
  (h1 : ticket_cost = 90) 
  (h2 : total_tickets = 4) 
  (h3 : current_money = 189) 
  (h4 : needed_money = 360 - 189) 
  : needed_money = 171 := by
  -- proof omitted
  sorry

end amount_daria_needs_l1212_121204


namespace xy_zero_l1212_121299

theorem xy_zero (x y : ℝ) (h1 : x + y = 4) (h2 : x^3 - y^3 = 64) : x * y = 0 := by
  sorry

end xy_zero_l1212_121299


namespace contrapositive_p_l1212_121228

-- Definitions
def A_score := 70
def B_score := 70
def C_score := 65
def p := ∀ (passing_score : ℕ), passing_score < 70 → (A_score < passing_score ∧ B_score < passing_score ∧ C_score < passing_score)

-- Statement to be proved
theorem contrapositive_p : 
  ∀ (passing_score : ℕ), (A_score ≥ passing_score ∨ B_score ≥ passing_score ∨ C_score ≥ passing_score) → (¬ passing_score < 70) := 
by
  sorry

end contrapositive_p_l1212_121228


namespace probability_x_lt_y_in_rectangle_l1212_121244

noncomputable def probability_point_in_triangle : ℚ :=
  let rectangle_area : ℚ := 4 * 3
  let triangle_area : ℚ := (1/2) * 3 * 3
  let probability : ℚ := triangle_area / rectangle_area
  probability

theorem probability_x_lt_y_in_rectangle :
  probability_point_in_triangle = 3 / 8 :=
by
  sorry

end probability_x_lt_y_in_rectangle_l1212_121244


namespace concert_ticket_revenue_l1212_121274

theorem concert_ticket_revenue :
  let original_price := 20
  let first_group_discount := 0.40
  let second_group_discount := 0.15
  let third_group_premium := 0.10
  let first_group_size := 10
  let second_group_size := 20
  let third_group_size := 15
  (first_group_size * (original_price - first_group_discount * original_price)) +
  (second_group_size * (original_price - second_group_discount * original_price)) +
  (third_group_size * (original_price + third_group_premium * original_price)) = 790 :=
by
  simp
  sorry

end concert_ticket_revenue_l1212_121274


namespace range_of_a_l1212_121207

def sets_nonempty_intersect (a : ℝ) : Prop :=
  ∃ x, -1 ≤ x ∧ x < 2 ∧ x < a

theorem range_of_a (a : ℝ) (h : sets_nonempty_intersect a) : a > -1 :=
by
  sorry

end range_of_a_l1212_121207


namespace evaluate_g_at_8_l1212_121225

def g (x : ℝ) : ℝ := 3 * x ^ 4 - 22 * x ^ 3 + 37 * x ^ 2 - 28 * x - 84

theorem evaluate_g_at_8 : g 8 = 1036 :=
by
  sorry

end evaluate_g_at_8_l1212_121225


namespace solve_equation1_solve_equation2_solve_system1_solve_system2_l1212_121200

-- Problem 1
theorem solve_equation1 (x : ℚ) : 3 * (x + 8) - 5 = 6 * (2 * x - 1) → x = 25 / 9 :=
by sorry

-- Problem 2
theorem solve_equation2 (x : ℚ) : (3 * x - 2) / 2 = (4 * x + 2) / 3 - 1 → x = 4 :=
by sorry

-- Problem 3
theorem solve_system1 (x y : ℚ) : (3 * x - 7 * y = 8) ∧ (2 * x + y = 11) → x = 5 ∧ y = 1 :=
by sorry

-- Problem 4
theorem solve_system2 (a b c : ℚ) : (a - b + c = 0) ∧ (4 * a + 2 * b + c = 3) ∧ (25 * a + 5 * b + c = 60) → (a = 3) ∧ (b = -2) ∧ (c = -5) :=
by sorry

end solve_equation1_solve_equation2_solve_system1_solve_system2_l1212_121200


namespace increase_in_lighting_power_l1212_121201

-- Conditions
def N_before : ℕ := 240
def N_after : ℕ := 300

-- Theorem
theorem increase_in_lighting_power : N_after - N_before = 60 := by
  sorry

end increase_in_lighting_power_l1212_121201


namespace cookies_difference_l1212_121231

theorem cookies_difference 
    (initial_sweet : ℕ) (initial_salty : ℕ) (initial_chocolate : ℕ)
    (ate_sweet : ℕ) (ate_salty : ℕ) (ate_chocolate : ℕ)
    (ratio_sweet : ℕ) (ratio_salty : ℕ) (ratio_chocolate : ℕ) :
    initial_sweet = 39 →
    initial_salty = 18 →
    initial_chocolate = 12 →
    ate_sweet = 27 →
    ate_salty = 6 →
    ate_chocolate = 8 →
    ratio_sweet = 3 →
    ratio_salty = 1 →
    ratio_chocolate = 2 →
    ate_sweet - ate_salty = 21 :=
by
  intros _ _ _ _ _ _ _ _ _
  sorry

end cookies_difference_l1212_121231


namespace area_larger_sphere_red_is_83_point_25_l1212_121262

-- Define the radii and known areas

def radius_smaller_sphere := 4 -- cm
def radius_larger_sphere := 6 -- cm
def area_smaller_sphere_red := 37 -- square cm

-- Prove the area of the region outlined in red on the larger sphere
theorem area_larger_sphere_red_is_83_point_25 :
  ∃ (area_larger_sphere_red : ℝ),
    area_larger_sphere_red = 83.25 ∧
    area_larger_sphere_red = area_smaller_sphere_red * (radius_larger_sphere ^ 2 / radius_smaller_sphere ^ 2) :=
by {
  sorry
}

end area_larger_sphere_red_is_83_point_25_l1212_121262


namespace value_of_percent_l1212_121261

theorem value_of_percent (x : ℝ) (h : 0.50 * x = 200) : 0.40 * x = 160 :=
sorry

end value_of_percent_l1212_121261


namespace reflections_in_mirrors_l1212_121259

theorem reflections_in_mirrors (x : ℕ)
  (h1 : 30 = 10 * 3)
  (h2 : 18 = 6 * 3)
  (h3 : 88 = 30 + 5 * x + 18 + 3 * x) :
  x = 5 := by
  sorry

end reflections_in_mirrors_l1212_121259


namespace gcd_lcm_lemma_l1212_121222

theorem gcd_lcm_lemma (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_sum : a + b = 33) (h_lcm : Nat.lcm a b = 90) : Nat.gcd a b = 3 :=
by
  sorry

end gcd_lcm_lemma_l1212_121222


namespace cyclist_speed_25_l1212_121293

def speeds_system_eqns (x : ℝ) (y : ℝ) : Prop :=
  (20 / x - 20 / 50 = y) ∧ (70 - (8 / 3) * x = 50 * (7 / 15 - y))

theorem cyclist_speed_25 :
  ∃ y : ℝ, speeds_system_eqns 25 y :=
by
  sorry

end cyclist_speed_25_l1212_121293


namespace kat_average_training_hours_l1212_121268

def strength_training_sessions_per_week : ℕ := 3
def strength_training_hour_per_session : ℕ := 1
def strength_training_missed_sessions_per_2_weeks : ℕ := 1

def boxing_training_sessions_per_week : ℕ := 4
def boxing_training_hour_per_session : ℝ := 1.5
def boxing_training_skipped_sessions_per_2_weeks : ℕ := 1

def cardio_workout_sessions_per_week : ℕ := 2
def cardio_workout_minutes_per_session : ℕ := 30

def flexibility_training_sessions_per_week : ℕ := 1
def flexibility_training_minutes_per_session : ℕ := 45

def interval_training_sessions_per_week : ℕ := 1
def interval_training_hour_per_session : ℝ := 1.25 -- 1 hour and 15 minutes 

noncomputable def average_hours_per_week : ℝ :=
  let strength_training_per_week : ℝ := ((5 / 2) * strength_training_hour_per_session)
  let boxing_training_per_week : ℝ := ((7 / 2) * boxing_training_hour_per_session)
  let cardio_workout_per_week : ℝ := (cardio_workout_sessions_per_week * cardio_workout_minutes_per_session / 60)
  let flexibility_training_per_week : ℝ := (flexibility_training_sessions_per_week * flexibility_training_minutes_per_session / 60)
  let interval_training_per_week : ℝ := interval_training_hour_per_session
  strength_training_per_week + boxing_training_per_week + cardio_workout_per_week + flexibility_training_per_week + interval_training_per_week

theorem kat_average_training_hours : average_hours_per_week = 10.75 := by
  unfold average_hours_per_week
  norm_num
  sorry

end kat_average_training_hours_l1212_121268


namespace speed_first_32_miles_l1212_121254

theorem speed_first_32_miles (x : ℝ) (y : ℝ) : 
  (100 / x + 0.52 * 100 / x = 32 / y + 68 / (x / 2)) → 
  y = 2 * x :=
by
  sorry

end speed_first_32_miles_l1212_121254


namespace max_min_sums_l1212_121236

def P (x y : ℤ) := x^2 + y^2 = 50

theorem max_min_sums : 
  ∃ (x₁ y₁ x₂ y₂ : ℤ), P x₁ y₁ ∧ P x₂ y₂ ∧ 
    (x₁ + y₁ = 8) ∧ (x₂ + y₂ = -8) :=
by
  sorry

end max_min_sums_l1212_121236


namespace final_answer_is_15_l1212_121251

-- We will translate the conditions from the problem into definitions and then formulate the theorem

-- Define the product of 10 and 12
def product : ℕ := 10 * 12

-- Define the result of dividing this product by 2
def divided_result : ℕ := product / 2

-- Define one-fourth of the divided result
def one_fourth : ℚ := (1/4 : ℚ) * divided_result

-- The theorem statement that verifies the final answer
theorem final_answer_is_15 : one_fourth = 15 := by
  sorry

end final_answer_is_15_l1212_121251


namespace closest_to_zero_is_neg_1001_l1212_121298

-- Definitions used in the conditions
def list_of_integers : List Int := [-1101, 1011, -1010, -1001, 1110]

-- Problem statement
theorem closest_to_zero_is_neg_1001 (x : Int) (H : x ∈ list_of_integers) :
  x = -1001 ↔ ∀ y ∈ list_of_integers, abs x ≤ abs y :=
sorry

end closest_to_zero_is_neg_1001_l1212_121298


namespace find_possible_K_l1212_121255

theorem find_possible_K (K : ℕ) (N : ℕ) (h1 : K * (K + 1) / 2 = N^2) (h2 : N < 150)
  (h3 : ∃ m : ℕ, N^2 = m * (m + 1) / 2) : K = 1 ∨ K = 8 ∨ K = 39 ∨ K = 92 ∨ K = 168 := by
  sorry

end find_possible_K_l1212_121255


namespace solve_problem_l1212_121281

def problem_statement : Prop :=
  ∀ (n1 n2 c1 : ℕ) (C : ℕ),
  n1 = 18 → 
  c1 = 60 → 
  n2 = 216 →
  n1 * c1 = n2 * C →
  C = 5

theorem solve_problem : problem_statement := by
  intros n1 n2 c1 C h1 h2 h3 h4
  -- Proof steps go here
  sorry

end solve_problem_l1212_121281


namespace circle_equation_l1212_121227

-- Defining the points A and B
def A : ℝ × ℝ := (-1, 1)
def B : ℝ × ℝ := (1, 3)

-- Defining the center M of the circle on the x-axis
def M (a : ℝ) : ℝ × ℝ := (a, 0)

-- Defining the squared distance function between two points
def dist_sq (P Q : ℝ × ℝ) : ℝ :=
  (P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2

-- Statement: Prove that the standard equation of the circle is (x - 2)² + y² = 10
theorem circle_equation : ∃ a : ℝ, (dist_sq (M a) A = dist_sq (M a) B) ∧ ((M a).1 = 2) ∧ (dist_sq (M a) A = 10) :=
sorry

end circle_equation_l1212_121227


namespace greatest_common_divisor_XYXY_pattern_l1212_121266

theorem greatest_common_divisor_XYXY_pattern (X Y : ℕ) (hX : X ≥ 0 ∧ X ≤ 9) (hY : Y ≥ 0 ∧ Y ≤ 9) :
  ∃ k, 11 * k = 1001 * X + 10 * Y :=
by
  sorry

end greatest_common_divisor_XYXY_pattern_l1212_121266


namespace proof_l1212_121245

noncomputable def M : Set ℝ := {x | 1 - (2 / x) > 0}
noncomputable def N : Set ℝ := {x | x ≥ 1}

theorem proof : (Mᶜ ∪ N) = {x | x ≥ 0} := sorry

end proof_l1212_121245


namespace John_profit_is_1500_l1212_121219

-- Defining the conditions
def P_initial : ℕ := 8
def Puppies_given_away : ℕ := P_initial / 2
def Puppies_kept : ℕ := 1
def Price_per_puppy : ℕ := 600
def Payment_stud_owner : ℕ := 300

-- Define the number of puppies John's selling
def Puppies_selling := Puppies_given_away - Puppies_kept

-- Define the total revenue from selling the puppies
def Total_revenue := Puppies_selling * Price_per_puppy

-- Define John’s profit 
def John_profit := Total_revenue - Payment_stud_owner

-- The statement to prove
theorem John_profit_is_1500 : John_profit = 1500 := by
  sorry

end John_profit_is_1500_l1212_121219


namespace corrected_mean_l1212_121238

theorem corrected_mean (n : ℕ) (incorrect_mean old_obs new_obs : ℚ) 
  (hn : n = 50) (h_mean : incorrect_mean = 40) (hold : old_obs = 15) (hnew : new_obs = 45) :
  ((n * incorrect_mean + (new_obs - old_obs)) / n) = 40.6 :=
by
  sorry

end corrected_mean_l1212_121238


namespace sufficient_but_not_necessary_condition_l1212_121257

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (∀ x : ℝ, |x - 3/4| ≤ 1/4 → (x - a) * (x - (a + 1)) ≤ 0) ∧
  ¬(∀ x : ℝ, (x - a) * (x - (a + 1)) ≤ 0 → |x - 3/4| ≤ 1/4) ↔ (0 ≤ a ∧ a ≤ 1/2) :=
by
  sorry

end sufficient_but_not_necessary_condition_l1212_121257


namespace sqrt8_sub_sqrt2_sub_sqrt1div3_mul_sqrt6_sqrt15_div_sqrt3_add_sqrt5_sub1_sq_l1212_121253

theorem sqrt8_sub_sqrt2_sub_sqrt1div3_mul_sqrt6 : 
  (Nat.sqrt 8 - Nat.sqrt 2 - (Nat.sqrt (1 / 3) * Nat.sqrt 6) = 0) :=
by
  sorry

theorem sqrt15_div_sqrt3_add_sqrt5_sub1_sq : 
  (Nat.sqrt 15 / Nat.sqrt 3 + (Nat.sqrt 5 - 1) ^ 2 = 6 - Nat.sqrt 5) :=
by
  sorry

end sqrt8_sub_sqrt2_sub_sqrt1div3_mul_sqrt6_sqrt15_div_sqrt3_add_sqrt5_sub1_sq_l1212_121253


namespace eq_implies_neq_neq_not_implies_eq_l1212_121229

variable (a b : ℝ)

-- Define the conditions
def condition1 : Prop := a^2 = b^2
def condition2 : Prop := a^2 + b^2 = 2 * a * b

-- Theorem statement representing the problem and conclusion
theorem eq_implies_neq (h : condition2 a b) : condition1 a b :=
by
  sorry

theorem neq_not_implies_eq (h : condition1 a b) : ¬ condition2 a b :=
by
  sorry

end eq_implies_neq_neq_not_implies_eq_l1212_121229


namespace min_value_objective_function_l1212_121202

theorem min_value_objective_function :
  (∃ x y : ℝ, x ≥ 1 ∧ x + y ≤ 3 ∧ x - 2 * y - 3 ≤ 0 ∧ (∀ x' y', (x' ≥ 1 ∧ x' + y' ≤ 3 ∧ x' - 2 * y' - 3 ≤ 0) → 2 * x' + y' ≥ 2 * x + y)) →
  2 * x + y = 1 :=
by
  sorry

end min_value_objective_function_l1212_121202


namespace rectangular_field_length_l1212_121241

theorem rectangular_field_length (w l : ℝ) (h1 : l = w + 10) (h2 : l^2 + w^2 = 22^2) : l = 22 := 
sorry

end rectangular_field_length_l1212_121241


namespace functional_equation_solution_l1212_121239

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x * f y - y * f x) = f (x * y) - x * y) →
  (f = id ∨ f = abs) :=
by sorry

end functional_equation_solution_l1212_121239


namespace kombucha_cost_l1212_121264

variable (C : ℝ)

-- Henry drinks 15 bottles of kombucha every month
def bottles_per_month : ℝ := 15

-- A year has 12 months
def months_per_year : ℝ := 12

-- Total bottles consumed in a year
def total_bottles := bottles_per_month * months_per_year

-- Cash refund per bottle
def refund_per_bottle : ℝ := 0.10

-- Total cash refund for all bottles in a year
def total_refund := total_bottles * refund_per_bottle

-- Number of bottles he can buy with the total refund
def bottles_purchasable_with_refund : ℝ := 6

-- Given that the total refund allows purchasing 6 bottles
def cost_per_bottle_eq : Prop := bottles_purchasable_with_refund * C = total_refund

-- Statement to prove
theorem kombucha_cost : cost_per_bottle_eq C → C = 3 := by
  intros
  sorry

end kombucha_cost_l1212_121264


namespace min_value_at_constraints_l1212_121276

open Classical

noncomputable def min_value (x y : ℝ) : ℝ := (x^2 + y^2 + x) / (x * y)

def constraints (x y : ℝ) : Prop := x > 0 ∧ y > 0 ∧ x + 2 * y = 1

theorem min_value_at_constraints : 
∃ (x y : ℝ), constraints x y ∧ min_value x y = 2 * Real.sqrt 2 + 2 :=
by
  sorry

end min_value_at_constraints_l1212_121276


namespace minimum_value_of_z_l1212_121297

theorem minimum_value_of_z : ∃ (x : ℝ), ∀ (z : ℝ), (z = 4 * x^2 + 8 * x + 16) → z ≥ 12 :=
by
  sorry

end minimum_value_of_z_l1212_121297


namespace find_values_of_a_and_b_l1212_121291

-- Definition of the problem and required conditions:
def symmetric_point (a b : ℝ) : Prop :=
  (a = -2) ∧ (b = -3)

theorem find_values_of_a_and_b (a b : ℝ) 
  (h : (a, -3) = (-2, -3) ∨ (2, b) = (2, -3) ∧ (a = -2)) :
  symmetric_point a b :=
by
  sorry

end find_values_of_a_and_b_l1212_121291


namespace annie_miles_l1212_121230

theorem annie_miles (x : ℝ) :
  2.50 + (0.25 * 42) = 2.50 + 5.00 + (0.25 * x) → x = 22 :=
by
  sorry

end annie_miles_l1212_121230


namespace problem_1_problem_2_l1212_121234

-- Definitions for the sets A and B
def A (x : ℝ) : Prop := -1 < x ∧ x < 2
def B (a : ℝ) (x : ℝ) : Prop := 2 * a - 1 < x ∧ x < 2 * a + 3

-- Problem 1: Range of values for a such that A ⊂ B
theorem problem_1 (a : ℝ) : (∀ x, A x → B a x) ↔ (-1/2 ≤ a ∧ a ≤ 0) := sorry

-- Problem 2: Range of values for a such that A ∩ B = ∅
theorem problem_2 (a : ℝ) : (∀ x, A x → ¬ B a x) ↔ (a ≤ -2 ∨ 3/2 ≤ a) := sorry

end problem_1_problem_2_l1212_121234


namespace perfect_square_trinomial_m6_l1212_121290

theorem perfect_square_trinomial_m6 (m : ℚ) (h₁ : 0 < m) (h₂ : ∃ a : ℚ, x^2 - 2 * m * x + 36 = (x - a)^2) : m = 6 :=
sorry

end perfect_square_trinomial_m6_l1212_121290


namespace inheritance_problem_l1212_121235

variables (x1 x2 x3 x4 : ℕ)

theorem inheritance_problem
  (h1 : x1 + x2 + x3 + x4 = 1320)
  (h2 : x1 + x4 = x2 + x3)
  (h3 : x2 + x4 = 2 * (x1 + x3))
  (h4 : x3 + x4 = 3 * (x1 + x2)) :
  x1 = 55 ∧ x2 = 275 ∧ x3 = 385 ∧ x4 = 605 :=
by sorry

end inheritance_problem_l1212_121235


namespace right_triangle_third_side_l1212_121256

theorem right_triangle_third_side (a b : ℝ) (h : a^2 + b^2 = c^2 ∨ a^2 = c^2 + b^2 ∨ b^2 = c^2 + a^2)
  (h1 : a = 3 ∧ b = 5 ∨ a = 5 ∧ b = 3) : c = 4 ∨ c = Real.sqrt 34 :=
sorry

end right_triangle_third_side_l1212_121256


namespace probability_of_reaching_last_floor_l1212_121277

noncomputable def probability_of_open_paths (n : ℕ) : ℝ :=
  2^(n-1) / (Nat.choose (2*(n-1)) (n-1))

theorem probability_of_reaching_last_floor (n : ℕ) :
  probability_of_open_paths n = 2^(n-1) / (Nat.choose (2*(n-1)) (n-1)) :=
by
  sorry

end probability_of_reaching_last_floor_l1212_121277


namespace smallest_value_expression_l1212_121249

theorem smallest_value_expression (n : ℕ) (hn : n > 0) : (n = 8) ↔ ((n / 2) + (32 / n) = 8) := by
  sorry

end smallest_value_expression_l1212_121249


namespace sin_330_eq_neg_one_half_l1212_121216

theorem sin_330_eq_neg_one_half : 
  Real.sin (330 * Real.pi / 180) = -1 / 2 := 
sorry

end sin_330_eq_neg_one_half_l1212_121216


namespace platform_length_l1212_121275

theorem platform_length (train_length : ℕ) (time_pole : ℕ) (time_platform : ℕ) 
  (h_train_length : train_length = 300) (h_time_pole : time_pole = 12) (h_time_platform : time_platform = 39) : 
  ∃ L : ℕ, L = 675 :=
by
  sorry

end platform_length_l1212_121275


namespace tangent_315_deg_l1212_121287

theorem tangent_315_deg : Real.tan (315 * (Real.pi / 180)) = -1 :=
by
  sorry

end tangent_315_deg_l1212_121287


namespace sophia_book_problem_l1212_121248

/-
Prove that the total length of the book P is 270 pages, and verify the number of pages read by Sophia
on the 4th and 5th days (50 and 40 pages respectively), given the following conditions:
1. Sophia finished 2/3 of the book in the first three days.
2. She calculated that she finished 90 more pages than she has yet to read.
3. She plans to finish the entire book within 5 days.
4. She will read 10 fewer pages each day from the 4th day until she finishes.
-/

theorem sophia_book_problem
  (P : ℕ)
  (h1 : (2/3 : ℝ) * P = P - (90 + (1/3 : ℝ) * P))
  (h2 : P = 3 * 90)
  (remaining_pages : ℕ := P / 3)
  (h3 : remaining_pages = 90)
  (pages_day4 : ℕ)
  (pages_day5 : ℕ := pages_day4 - 10)
  (h4 : pages_day4 + pages_day4 - 10 = 90)
  (h5 : 2 * pages_day4 - 10 = 90)
  (h6 : 2 * pages_day4 = 100)
  (h7 : pages_day4 = 50) :
  P = 270 ∧ pages_day4 = 50 ∧ pages_day5 = 40 := 
by {
  sorry -- Proof is skipped
}

end sophia_book_problem_l1212_121248


namespace stratified_sampling_expected_elderly_chosen_l1212_121265

theorem stratified_sampling_expected_elderly_chosen :
  let total := 165
  let to_choose := 15
  let elderly := 22
  (22 : ℚ) / 165 * 15 = 2 := sorry

end stratified_sampling_expected_elderly_chosen_l1212_121265


namespace fenced_area_correct_l1212_121267

-- Define the dimensions of the rectangle
def length := 20
def width := 18

-- Define the dimensions of the cutouts
def square_cutout1 := 4
def square_cutout2 := 2

-- Define the areas of the rectangle and the cutouts
def area_rectangle := length * width
def area_cutout1 := square_cutout1 * square_cutout1
def area_cutout2 := square_cutout2 * square_cutout2

-- Define the total area within the fence
def total_area_within_fence := area_rectangle - area_cutout1 - area_cutout2

-- The theorem that needs to be proven
theorem fenced_area_correct : total_area_within_fence = 340 := by
  sorry

end fenced_area_correct_l1212_121267


namespace determine_k_l1212_121289

variable (x y z k : ℝ)

theorem determine_k (h : 9 / (x + y) = k / (y + z) ∧ k / (y + z) = 15 / (x - z)) : k = 0 := by
  sorry

end determine_k_l1212_121289


namespace average_of_first_two_numbers_l1212_121232

theorem average_of_first_two_numbers (s1 s2 s3 s4 s5 s6 a b c : ℝ) 
  (h_average_six : (s1 + s2 + s3 + s4 + s5 + s6) / 6 = 4.6)
  (h_average_set2 : (s3 + s4) / 2 = 3.8)
  (h_average_set3 : (s5 + s6) / 2 = 6.6)
  (h_total_sum : s1 + s2 + s3 + s4 + s5 + s6 = 27.6) : 
  (s1 + s2) / 2 = 3.4 :=
sorry

end average_of_first_two_numbers_l1212_121232


namespace find_dividend_l1212_121246

-- Definitions from conditions
def divisor : ℕ := 14
def quotient : ℕ := 12
def remainder : ℕ := 8

-- The problem statement to prove
theorem find_dividend : (divisor * quotient + remainder) = 176 := by
  sorry

end find_dividend_l1212_121246


namespace total_seeds_eaten_proof_l1212_121224

-- Define the information about the number of seeds eaten by each player
def first_player_seeds : ℕ := 78
def second_player_seeds : ℕ := 53
def third_player_seeds : ℕ := second_player_seeds + 30
def fourth_player_seeds : ℕ := 2 * third_player_seeds

-- Sum the seeds eaten by all the players
def total_seeds_eaten : ℕ := first_player_seeds + second_player_seeds + third_player_seeds + fourth_player_seeds

-- Prove that the total number of seeds eaten is 380
theorem total_seeds_eaten_proof : total_seeds_eaten = 380 :=
by
  -- To be filled in by actual proof steps
  sorry

end total_seeds_eaten_proof_l1212_121224


namespace sufficient_not_necessary_for_one_zero_l1212_121215

variable {a x : ℝ}

def f (a x : ℝ) : ℝ := a * x ^ 2 - 2 * x + 1

theorem sufficient_not_necessary_for_one_zero :
  (∃ x : ℝ, f 1 x = 0) ∧ (∀ x : ℝ, f 0 x = -2 * x + 1 → x ≠ 0) → 
  (∃ x : ℝ, f a x = 0) → (a = 1 ∨ f 0 x = 0)  :=
sorry

end sufficient_not_necessary_for_one_zero_l1212_121215


namespace cone_rotation_ratio_l1212_121206

theorem cone_rotation_ratio (r h : ℝ) (h_pos : 0 < h) (r_pos : 0 < r) 
  (rotation_eq : (20 : ℝ) * (2 * Real.pi * r) = 2 * Real.pi * Real.sqrt (r^2 + h^2)) :
  let p := 1
  let q := 399
  1 + 399 = 400 := by
{
  sorry
}

end cone_rotation_ratio_l1212_121206


namespace matrix_determinant_l1212_121247

theorem matrix_determinant (x : ℝ) :
  Matrix.det ![![x, x + 2], ![3, 2 * x]] = 2 * x^2 - 3 * x - 6 :=
by
  sorry

end matrix_determinant_l1212_121247


namespace arccos_neg_one_eq_pi_l1212_121208

theorem arccos_neg_one_eq_pi : Real.arccos (-1) = Real.pi :=
by
  sorry

end arccos_neg_one_eq_pi_l1212_121208


namespace how_many_both_books_l1212_121223

-- Definitions based on the conditions
def total_workers : ℕ := 40
def saramago_workers : ℕ := total_workers / 4
def kureishi_workers : ℕ := (total_workers * 5) / 8
def both_books (B : ℕ) : Prop :=
  B + (saramago_workers - B) + (kureishi_workers - B) + (9 - B) = total_workers

theorem how_many_both_books : ∃ B : ℕ, both_books B ∧ B = 4 := by
  use 4
  -- Proof goes here, skipped by using sorry
  sorry

end how_many_both_books_l1212_121223


namespace proof_expression_l1212_121220

open Real

theorem proof_expression (x y : ℝ) (h1 : P = 2 * (x + y)) (h2 : Q = 3 * (x - y)) :
  (P + Q) / (P - Q) - (P - Q) / (P + Q) + (x + y) / (x - y) = (28 * x^2 - 20 * y^2) / ((x - y) * (5 * x - y) * (-x + 5 * y)) :=
by
  sorry

end proof_expression_l1212_121220


namespace subset_P_Q_l1212_121292

-- Definitions of the sets P and Q
def P : Set ℝ := {x | x^2 - 3 * x + 2 < 0}
def Q : Set ℝ := {x | 1 < x ∧ x < 3}

-- Statement to prove P ⊆ Q
theorem subset_P_Q : P ⊆ Q :=
sorry

end subset_P_Q_l1212_121292


namespace number_divisible_by_37_l1212_121296

def consecutive_ones_1998 : ℕ := (10 ^ 1998 - 1) / 9

theorem number_divisible_by_37 : 37 ∣ consecutive_ones_1998 :=
sorry

end number_divisible_by_37_l1212_121296


namespace teacher_student_arrangements_boy_girl_selection_program_arrangements_l1212_121282

-- Question 1
theorem teacher_student_arrangements : 
  let positions := 5
  let student_arrangements := 720
  positions * student_arrangements = 3600 :=
by
  sorry

-- Question 2
theorem boy_girl_selection :
  let total_selections := 330
  let opposite_selections := 20
  total_selections - opposite_selections = 310 :=
by
  sorry

-- Question 3
theorem program_arrangements :
  let total_permutations := 120
  let relative_order_permutations := 6
  total_permutations / relative_order_permutations = 20 :=
by
  sorry

end teacher_student_arrangements_boy_girl_selection_program_arrangements_l1212_121282


namespace friend_spent_more_l1212_121243

/-- Given that the total amount spent for lunch is $15 and your friend spent $8 on their lunch,
we need to prove that your friend spent $1 more than you did. -/
theorem friend_spent_more (total_spent friend_spent : ℤ) (h1 : total_spent = 15) (h2 : friend_spent = 8) :
  friend_spent - (total_spent - friend_spent) = 1 :=
by
  sorry

end friend_spent_more_l1212_121243


namespace batsman_average_20th_l1212_121263

noncomputable def average_after_20th (A : ℕ) : ℕ :=
  let total_runs_19 := 19 * A
  let total_runs_20 := total_runs_19 + 85
  let new_average := (total_runs_20) / 20
  new_average
  
theorem batsman_average_20th (A : ℕ) (h1 : 19 * A + 85 = 20 * (A + 4)) : average_after_20th A = 9 := by
  sorry

end batsman_average_20th_l1212_121263


namespace goldfish_equal_after_8_months_l1212_121211

noncomputable def B (n : ℕ) : ℝ := 3^(n + 1)
noncomputable def G (n : ℕ) : ℝ := 243 * 1.5^n

theorem goldfish_equal_after_8_months :
  ∃ n : ℕ, B n = G n ∧ n = 8 :=
by
  sorry

end goldfish_equal_after_8_months_l1212_121211


namespace find_a10_l1212_121214

noncomputable def arithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d a₁, a 1 = a₁ ∧ ∀ n, a (n + 1) = a n + d

theorem find_a10 (a : ℕ → ℤ) (h_seq : arithmeticSequence a) 
  (h1 : a 1 + a 3 + a 5 = 9) 
  (h2 : a 3 * (a 4) ^ 2 = 27) :
  a 10 = -39 ∨ a 10 = 30 :=
sorry

end find_a10_l1212_121214


namespace johns_calorie_intake_l1212_121252

theorem johns_calorie_intake
  (servings : ℕ)
  (calories_per_serving : ℕ)
  (total_calories : ℕ)
  (half_package_calories : ℕ)
  (h1 : servings = 3)
  (h2 : calories_per_serving = 120)
  (h3 : total_calories = servings * calories_per_serving)
  (h4 : half_package_calories = total_calories / 2)
  : half_package_calories = 180 :=
by sorry

end johns_calorie_intake_l1212_121252


namespace quadratic_has_only_positive_roots_l1212_121212

theorem quadratic_has_only_positive_roots (m : ℝ) :
  (∀ (x : ℝ), x^2 + (m + 2) * x + (m + 5) = 0 → x > 0) →
  -5 < m ∧ m ≤ -4 :=
by 
  -- added sorry to skip the proof.
  sorry

end quadratic_has_only_positive_roots_l1212_121212


namespace fraction_saved_l1212_121240

-- Definitions and given conditions
variables {P : ℝ} {f : ℝ}

-- Worker saves the same fraction each month, the same take-home pay each month
-- Total annual savings = 12fP and total annual savings = 2 * (amount not saved monthly)
theorem fraction_saved (h : 12 * f * P = 2 * (1 - f) * P) (P_ne_zero : P ≠ 0) : f = 1 / 7 :=
by
  -- The proof of the theorem goes here
  sorry

end fraction_saved_l1212_121240


namespace tan_sin_cos_eq_l1212_121209

theorem tan_sin_cos_eq (x : ℝ) (h : 0 < x ∧ x < 180) :
  (Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) → x = 120 :=
by
  sorry

end tan_sin_cos_eq_l1212_121209


namespace find_divisor_l1212_121233

theorem find_divisor (d : ℕ) (q r : ℕ) (h₁ : 190 = q * d + r) (h₂ : q = 9) (h₃ : r = 1) : d = 21 :=
by
  sorry

end find_divisor_l1212_121233


namespace incorrect_statement_d_l1212_121210

variable (x : ℝ)
variables (p q : Prop)

-- Proving D is incorrect given defined conditions
theorem incorrect_statement_d :
  ∀ (x : ℝ), (¬ (x = 1) → ¬ (x^2 - 3 * x + 2 = 0)) ∧
  ((x > 2) → (x^2 - 3 * x + 2 > 0) ∧
  (¬ (x^2 + x + 1 = 0))) ∧
  ((p ∨ q) → ¬ (p ∧ q)) :=
by
  -- A detailed proof would be required here
  sorry

end incorrect_statement_d_l1212_121210


namespace total_red_and_green_peaches_l1212_121221

def red_peaches : ℕ := 6
def green_peaches : ℕ := 16

theorem total_red_and_green_peaches :
  red_peaches + green_peaches = 22 :=
  by 
    sorry

end total_red_and_green_peaches_l1212_121221


namespace number_of_movies_l1212_121269

theorem number_of_movies (B M : ℕ)
  (h1 : B = 15)
  (h2 : B = M + 1) : M = 14 :=
by sorry

end number_of_movies_l1212_121269


namespace min_perimeter_l1212_121280

theorem min_perimeter :
  ∃ (a b c : ℕ), 
  (2 * a + 18 * c = 2 * b + 20 * c) ∧ 
  (9 * Real.sqrt (a^2 - (9 * c)^2) = 10 * Real.sqrt (b^2 - (10 * c)^2)) ∧ 
  (10 * (a - b) = 9 * c) ∧ 
  2 * a + 18 * c = 362 := 
sorry

end min_perimeter_l1212_121280


namespace train_speed_l1212_121279

theorem train_speed (length : ℝ) (time : ℝ) (h_length : length = 350) (h_time : time = 7) : 
  length / time = 50 :=
by
  rw [h_length, h_time]
  norm_num

end train_speed_l1212_121279


namespace geometric_sequence_seventh_term_l1212_121271

theorem geometric_sequence_seventh_term (a1 : ℕ) (a6 : ℕ) (r : ℚ)
  (ha1 : a1 = 3) (ha6 : a1 * r^5 = 972) : 
  a1 * r^6 = 2187 := 
by
  sorry

end geometric_sequence_seventh_term_l1212_121271


namespace determine_a_l1212_121286

theorem determine_a (a : ℕ) (h : a / (a + 36) = 9 / 10) : a = 324 :=
sorry

end determine_a_l1212_121286


namespace bin_to_oct_l1212_121205

theorem bin_to_oct (n : ℕ) (hn : n = 0b11010) : n = 0o32 := by
  sorry

end bin_to_oct_l1212_121205


namespace find_f_l1212_121283

-- Definitions of odd and even functions
def odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x
def even_function (g : ℝ → ℝ) := ∀ x, g (-x) = g x

-- Main theorem
theorem find_f (f g : ℝ → ℝ) (h_odd_f : odd_function f) (h_even_g : even_function g) 
    (h_eq : ∀ x, f x + g x = 1 / (x - 1)) :
  ∀ x, f x = x / (x ^ 2 - 1) :=
by
  sorry

end find_f_l1212_121283


namespace solve_system_of_equations_l1212_121285

theorem solve_system_of_equations :
    ∀ (x y : ℝ), 
    (x^3 * y + x * y^3 = 10) ∧ (x^4 + y^4 = 17) ↔
    (x = 2 ∧ y = 1) ∨ (x = 1 ∧ y = 2) ∨ (x = -1 ∧ y = -2) ∨ (x = -2 ∧ y = -1) :=
by
    sorry

end solve_system_of_equations_l1212_121285


namespace geometric_sequence_S4_l1212_121270

noncomputable section

def geometric_series_sum (a1 q : ℚ) (n : ℕ) : ℚ := 
  a1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_S4 (a1 : ℚ) (q : ℚ)
  (h1 : a1 * q^3 = 2 * a1)
  (h2 : 5 / 2 = a1 * (q^3 + 2 * q^6)) :
  geometric_series_sum a1 q 4 = 30 := by
  sorry

end geometric_sequence_S4_l1212_121270


namespace unit_digit_of_expression_l1212_121294

theorem unit_digit_of_expression :
  let expr := (2 + 1) * (2^2 + 1) * (2^4 + 1) * (2^8 + 1) * (2^16 + 1) * (2^32 + 1)
  (expr - 1) % 10 = 4 :=
by
  let expr := (2 + 1) * (2^2 + 1) * (2^4 + 1) * (2^8 + 1) * (2^16 + 1) * (2^32 + 1)
  sorry

end unit_digit_of_expression_l1212_121294


namespace most_reasonable_sampling_method_l1212_121295

-- Define the conditions
def significant_difference_by_educational_stage := true
def no_significant_difference_by_gender := true

-- Define the statement
theorem most_reasonable_sampling_method :
  (significant_difference_by_educational_stage ∧ no_significant_difference_by_gender) →
  "Stratified sampling by educational stage" = "most reasonable sampling method" :=
by
  sorry

end most_reasonable_sampling_method_l1212_121295


namespace range_of_x_for_sqrt_l1212_121242

theorem range_of_x_for_sqrt (x : ℝ) (h : x - 5 ≥ 0) : x ≥ 5 :=
sorry

end range_of_x_for_sqrt_l1212_121242


namespace throws_to_return_to_elsa_l1212_121218

theorem throws_to_return_to_elsa :
  ∃ n, n = 5 ∧ (∀ (k : ℕ), k < n → ((1 + 5 * k) % 13 ≠ 1)) ∧ (1 + 5 * n) % 13 = 1 :=
by
  sorry

end throws_to_return_to_elsa_l1212_121218


namespace find_percentage_l1212_121278

theorem find_percentage (P : ℝ) : 100 * (P / 100) + 20 = 100 → P = 80 :=
by
  sorry

end find_percentage_l1212_121278


namespace find_sum_of_terms_l1212_121226

noncomputable def arithmetic_sequence (a₁ d n : ℕ) : ℕ := a₁ + (n - 1) * d
noncomputable def sum_of_first_n_terms (a₁ d n : ℕ) : ℕ := n * (2 * a₁ + (n - 1) * d) / 2

theorem find_sum_of_terms (a₁ d : ℕ) (S : ℕ → ℕ) (h1 : S 4 = 8) (h2 : S 8 = 20) :
    S 4 = 4 * (2 * a₁ + 3 * d) / 2 → S 8 = 8 * (2 * a₁ + 7 * d) / 2 →
    a₁ = 13 / 8 ∧ d = 1 / 4 →
    a₁ + 10 * d + a₁ + 11 * d + a₁ + 12 * d + a₁ + 13 * d = 18 :=
by 
  sorry

end find_sum_of_terms_l1212_121226


namespace part1_part2_l1212_121258

open Real

noncomputable def f (x a : ℝ) : ℝ := exp x - x^a

theorem part1 (a : ℝ) : (∀ x > 0, f x a ≥ 0) → a ≤ exp 1 :=
sorry

theorem part2 (a x1 x2 : ℝ) (hx1 : 0 < x1) (hx2 : 0 < x2) (hx : x1 > x2) :
  f x1 a = 0 → f x2 a = 0 → x1 + x2 > 2 * a :=
sorry

end part1_part2_l1212_121258


namespace suitcase_problem_l1212_121272

noncomputable def weight_of_electronics (k : ℝ) : ℝ :=
  2 * k

theorem suitcase_problem (k : ℝ) (B C E T : ℝ) (hc1 : B = 5 * k) (hc2 : C = 4 * k) (hc3 : E = 2 * k) (hc4 : T = 3 * k) (new_ratio : 5 * k / (4 * k - 7) = 3) :
  E = 6 :=
by
  sorry

end suitcase_problem_l1212_121272


namespace ab_is_zero_l1212_121250

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (x^2 + 1) + x)

theorem ab_is_zero (a b : ℝ) (h : a - 1 = 0) : a * b = 0 := by
  sorry

end ab_is_zero_l1212_121250


namespace trader_profit_percent_l1212_121284

-- Definitions based on the conditions
variables (P : ℝ) -- Original price of the car
def discount_price := 0.95 * P
def taxes := 0.03 * P
def maintenance := 0.02 * P
def total_cost := discount_price + taxes + maintenance 
def selling_price := 0.95 * P * 1.60
def profit := selling_price - total_cost

-- Theorem
theorem trader_profit_percent : (profit P / P) * 100 = 52 :=
by
  sorry

end trader_profit_percent_l1212_121284


namespace cost_of_25kg_l1212_121260

-- Definitions and conditions
def price_33kg (l q : ℕ) : Prop := 30 * l + 3 * q = 360
def price_36kg (l q : ℕ) : Prop := 30 * l + 6 * q = 420

-- Theorem statement
theorem cost_of_25kg (l q : ℕ) (h1 : 30 * l + 3 * q = 360) (h2 : 30 * l + 6 * q = 420) : 25 * l = 250 :=
by
  sorry

end cost_of_25kg_l1212_121260


namespace total_distance_race_l1212_121213

theorem total_distance_race
  (t_Sadie : ℝ) (s_Sadie : ℝ) (t_Ariana : ℝ) (s_Ariana : ℝ) 
  (s_Sarah : ℝ) (tt : ℝ)
  (h_Sadie : t_Sadie = 2) (hs_Sadie : s_Sadie = 3) 
  (h_Ariana : t_Ariana = 0.5) (hs_Ariana : s_Ariana = 6) 
  (hs_Sarah : s_Sarah = 4)
  (h_tt : tt = 4.5) : 
  (s_Sadie * t_Sadie + s_Ariana * t_Ariana + s_Sarah * (tt - (t_Sadie + t_Ariana))) = 17 := 
  by {
    sorry -- proof goes here
  }

end total_distance_race_l1212_121213


namespace intersection_of_A_and_B_l1212_121203

def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x | x ≥ 2}

theorem intersection_of_A_and_B :
  (A ∩ B) = {2} := 
by {
  sorry
}

end intersection_of_A_and_B_l1212_121203


namespace animals_per_aquarium_l1212_121217

theorem animals_per_aquarium (total_animals : ℕ) (number_of_aquariums : ℕ) (h1 : total_animals = 40) (h2 : number_of_aquariums = 20) : 
  total_animals / number_of_aquariums = 2 :=
by
  sorry

end animals_per_aquarium_l1212_121217


namespace draw_points_worth_two_l1212_121273

/-
In a certain football competition, a victory is worth 3 points, a draw is worth some points, and a defeat is worth 0 points. Each team plays 20 matches. A team scored 14 points after 5 games. The team needs to win at least 6 of the remaining matches to reach the 40-point mark by the end of the tournament. Prove that the number of points a draw is worth is 2.
-/

theorem draw_points_worth_two :
  ∃ D, (∀ (victory_points draw_points defeat_points total_matches matches_played points_scored remaining_matches wins_needed target_points),
    victory_points = 3 ∧
    defeat_points = 0 ∧
    total_matches = 20 ∧
    matches_played = 5 ∧
    points_scored = 14 ∧
    remaining_matches = total_matches - matches_played ∧
    wins_needed = 6 ∧
    target_points = 40 ∧
    points_scored + 6 * victory_points + (remaining_matches - wins_needed) * D = target_points ∧
    draw_points = D) →
    D = 2 :=
by
  sorry

end draw_points_worth_two_l1212_121273


namespace jogging_distance_apart_l1212_121288

theorem jogging_distance_apart 
  (anna_rate : ℕ) (mark_rate : ℕ) (time_hours : ℕ) :
  anna_rate = (1 / 20) ∧ mark_rate = (3 / 40) ∧ time_hours = 2 → 
  6 + 3 = 9 :=
by
  -- setting up constants and translating conditions into variables
  have anna_distance : ℕ := 6
  have mark_distance : ℕ := 3
  sorry

end jogging_distance_apart_l1212_121288


namespace system_of_equations_solutions_l1212_121237

theorem system_of_equations_solutions (x y : ℝ) (h1 : x ^ 5 + y ^ 5 = 1) (h2 : x ^ 6 + y ^ 6 = 1) :
    (x = 1 ∧ y = 0) ∨ (x = 0 ∧ y = 1) :=
sorry

end system_of_equations_solutions_l1212_121237
