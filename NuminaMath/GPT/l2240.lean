import Mathlib

namespace NUMINAMATH_GPT_speed_in_still_water_l2240_224088

-- Defining the terms as given conditions in the problem
def speed_downstream (v_m v_s : ℝ) : ℝ := v_m + v_s
def speed_upstream (v_m v_s : ℝ) : ℝ := v_m - v_s

-- Given conditions translated into Lean definitions
def downstream_condition : Prop :=
  ∃ (v_m v_s : ℝ), speed_downstream v_m v_s = 7

def upstream_condition : Prop :=
  ∃ (v_m v_s : ℝ), speed_upstream v_m v_s = 4

-- The problem statement to prove
theorem speed_in_still_water : 
  downstream_condition ∧ upstream_condition → ∃ v_m : ℝ, v_m = 5.5 :=
by 
  intros
  sorry

end NUMINAMATH_GPT_speed_in_still_water_l2240_224088


namespace NUMINAMATH_GPT_problem_statement_l2240_224013

variables {Line Plane : Type}
variables {m n : Line} {alpha beta : Plane}

-- Define parallel and perpendicular relations
def parallel (l1 l2 : Line) : Prop := sorry
def perp (l : Line) (p : Plane) : Prop := sorry

-- Define that m and n are different lines
axiom diff_lines (m n : Line) : m ≠ n 

-- Define that alpha and beta are different planes
axiom diff_planes (alpha beta : Plane) : alpha ≠ beta

-- Statement to prove: If m ∥ n and m ⟂ α, then n ⟂ α
theorem problem_statement (h1 : parallel m n) (h2 : perp m alpha) : perp n alpha := 
sorry

end NUMINAMATH_GPT_problem_statement_l2240_224013


namespace NUMINAMATH_GPT_fraction_product_108_l2240_224014

theorem fraction_product_108 : (1/2 : ℚ) * (1/3) * (1/6) * 108 = 3 := by
  sorry

end NUMINAMATH_GPT_fraction_product_108_l2240_224014


namespace NUMINAMATH_GPT_find_radius_of_sphere_l2240_224019

noncomputable def radius_of_sphere (R : ℝ) : Prop :=
  ∃ a b c : ℝ, 
  (R = |a| ∧ R = |b| ∧ R = |c|) ∧ 
  ((3 - R)^2 + (2 - R)^2 + (1 - R)^2 = R^2)

theorem find_radius_of_sphere : radius_of_sphere (3 + Real.sqrt 2) ∨ radius_of_sphere (3 - Real.sqrt 2) :=
sorry

end NUMINAMATH_GPT_find_radius_of_sphere_l2240_224019


namespace NUMINAMATH_GPT_replaced_person_weight_l2240_224076

theorem replaced_person_weight (W : ℝ) (increase : ℝ) (new_weight : ℝ) (average_increase : ℝ) (number_of_persons : ℕ) :
  average_increase = 2.5 →
  new_weight = 70 →
  number_of_persons = 8 →
  increase = number_of_persons * average_increase →
  W + increase = W - replaced_weight + new_weight →
  replaced_weight = 50 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_replaced_person_weight_l2240_224076


namespace NUMINAMATH_GPT_machine_working_time_l2240_224008

theorem machine_working_time (shirts_per_minute : ℕ) (total_shirts : ℕ) (h1 : shirts_per_minute = 3) (h2 : total_shirts = 6) :
  (total_shirts / shirts_per_minute) = 2 :=
by
  -- Begin the proof
  sorry

end NUMINAMATH_GPT_machine_working_time_l2240_224008


namespace NUMINAMATH_GPT_student_correct_answers_l2240_224062

theorem student_correct_answers (C I : ℕ) (h₁ : C + I = 100) (h₂ : C - 2 * I = 61) : C = 87 :=
sorry

end NUMINAMATH_GPT_student_correct_answers_l2240_224062


namespace NUMINAMATH_GPT_cost_of_fencing_each_side_l2240_224055

theorem cost_of_fencing_each_side (x : ℝ) (h : 4 * x = 316) : x = 79 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_fencing_each_side_l2240_224055


namespace NUMINAMATH_GPT_train_speed_l2240_224026

theorem train_speed (D T : ℝ) (h1 : D = 160) (h2 : T = 16) : D / T = 10 :=
by 
  -- given D = 160 and T = 16, we need to prove D / T = 10
  sorry

end NUMINAMATH_GPT_train_speed_l2240_224026


namespace NUMINAMATH_GPT_cash_price_of_tablet_l2240_224036

-- Define the conditions
def down_payment : ℕ := 100
def first_4_months_payment : ℕ := 4 * 40
def next_4_months_payment : ℕ := 4 * 35
def last_4_months_payment : ℕ := 4 * 30
def savings : ℕ := 70

-- Define the total installment payments
def total_installment_payments : ℕ := down_payment + first_4_months_payment + next_4_months_payment + last_4_months_payment

-- The statement to prove
theorem cash_price_of_tablet : total_installment_payments - savings = 450 := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_cash_price_of_tablet_l2240_224036


namespace NUMINAMATH_GPT_find_number_l2240_224086

theorem find_number (x : ℝ) (h : 0.5 * x = 0.25 * x + 2) : x = 8 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l2240_224086


namespace NUMINAMATH_GPT_find_n_l2240_224044

variable {a b c : ℝ} (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0)
variable (h : 1/a + 1/b + 1/c = 1/(a + b + c))

theorem find_n (n : ℤ) : (∃ k : ℕ, n = 2 * k - 1) → 
  (1 / a^n + 1 / b^n + 1 / c^n = 1 / (a^n + b^n + c^n)) :=
by
  sorry

end NUMINAMATH_GPT_find_n_l2240_224044


namespace NUMINAMATH_GPT_regression_is_appropriate_l2240_224023

-- Definitions for the different analysis methods
inductive AnalysisMethod
| ResidualAnalysis : AnalysisMethod
| RegressionAnalysis : AnalysisMethod
| IsoplethBarChart : AnalysisMethod
| IndependenceTest : AnalysisMethod

-- Relating height and weight with an appropriate analysis method
def appropriateMethod (method : AnalysisMethod) : Prop :=
  method = AnalysisMethod.RegressionAnalysis

-- Stating the theorem that regression analysis is the appropriate method
theorem regression_is_appropriate : appropriateMethod AnalysisMethod.RegressionAnalysis :=
by sorry

end NUMINAMATH_GPT_regression_is_appropriate_l2240_224023


namespace NUMINAMATH_GPT_hyperbola_equation_l2240_224001

-- Definitions based on the conditions:
def hyperbola (x y a b : ℝ) : Prop := (y^2 / a^2) - (x^2 / b^2) = 1

def point_on_hyperbola (a b : ℝ) : Prop := hyperbola 2 (-2) a b

def asymptotes (a b : ℝ) : Prop := a / b = (Real.sqrt 2) / 2

-- Prove the equation of the hyperbola
theorem hyperbola_equation :
  ∃ a b, a = Real.sqrt 2 ∧ b = 2 ∧ hyperbola y x (Real.sqrt 2) 2 :=
by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_hyperbola_equation_l2240_224001


namespace NUMINAMATH_GPT_lines_intersect_first_quadrant_l2240_224078

theorem lines_intersect_first_quadrant (k : ℝ) :
  (∃ (x y : ℝ), 2 * x + 7 * y = 14 ∧ k * x - y = k + 1 ∧ x > 0 ∧ y > 0) ↔ k > 0 :=
by
  sorry

end NUMINAMATH_GPT_lines_intersect_first_quadrant_l2240_224078


namespace NUMINAMATH_GPT_xy_product_l2240_224012

noncomputable def f (t : ℝ) : ℝ := Real.sqrt (t^2 + 1) - t + 1

theorem xy_product (x y : ℝ)
  (h : (Real.sqrt (x^2 + 1) - x + 1) * (Real.sqrt (y^2 + 1) - y + 1) = 2) :
  x * y = 1 := by
  sorry

end NUMINAMATH_GPT_xy_product_l2240_224012


namespace NUMINAMATH_GPT_typing_time_together_l2240_224043

theorem typing_time_together 
  (jonathan_time : ℝ)
  (susan_time : ℝ)
  (jack_time : ℝ)
  (document_pages : ℝ)
  (combined_time : ℝ) :
  jonathan_time = 40 →
  susan_time = 30 →
  jack_time = 24 →
  document_pages = 10 →
  combined_time = document_pages / ((document_pages / jonathan_time) + (document_pages / susan_time) + (document_pages / jack_time)) →
  combined_time = 10 :=
by sorry

end NUMINAMATH_GPT_typing_time_together_l2240_224043


namespace NUMINAMATH_GPT_find_x_coordinate_l2240_224098

theorem find_x_coordinate (m b x y : ℝ) (h1: m = 4) (h2: b = 100) (h3: y = 300) (line_eq: y = m * x + b) : x = 50 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_x_coordinate_l2240_224098


namespace NUMINAMATH_GPT_stu_books_count_l2240_224092

theorem stu_books_count (S : ℕ) (h1 : S + 4 * S = 45) : S = 9 := 
by
  sorry

end NUMINAMATH_GPT_stu_books_count_l2240_224092


namespace NUMINAMATH_GPT_sum_squares_reciprocal_l2240_224059

variable (x y : ℝ)

theorem sum_squares_reciprocal (h₁ : x + y = 12) (h₂ : x * y = 32) :
  (1/x)^2 + (1/y)^2 = 5/64 := by
  sorry

end NUMINAMATH_GPT_sum_squares_reciprocal_l2240_224059


namespace NUMINAMATH_GPT_mowing_field_l2240_224091

theorem mowing_field (x : ℝ) 
  (h1 : 1 / 84 + 1 / x = 1 / 21) : 
  x = 28 := 
sorry

end NUMINAMATH_GPT_mowing_field_l2240_224091


namespace NUMINAMATH_GPT_multiple_of_bees_l2240_224002

theorem multiple_of_bees (b₁ b₂ : ℕ) (h₁ : b₁ = 144) (h₂ : b₂ = 432) : b₂ / b₁ = 3 := 
by
  sorry

end NUMINAMATH_GPT_multiple_of_bees_l2240_224002


namespace NUMINAMATH_GPT_chocolates_not_in_box_initially_l2240_224041

theorem chocolates_not_in_box_initially 
  (total_chocolates : ℕ) 
  (chocolates_friend_brought : ℕ) 
  (initial_boxes : ℕ) 
  (additional_boxes : ℕ)
  (total_after_friend : ℕ)
  (chocolates_each_box : ℕ)
  (total_chocolates_initial : ℕ) :
  total_chocolates = 50 ∧ initial_boxes = 3 ∧ chocolates_friend_brought = 25 ∧ total_after_friend = 75 
  ∧ additional_boxes = 2 ∧ chocolates_each_box = 15 ∧ total_chocolates_initial = 50
  → (total_chocolates_initial - (initial_boxes * chocolates_each_box)) = 5 :=
by
  sorry

end NUMINAMATH_GPT_chocolates_not_in_box_initially_l2240_224041


namespace NUMINAMATH_GPT_Debby_bought_bottles_l2240_224016

theorem Debby_bought_bottles :
  (5 : ℕ) * (71 : ℕ) = 355 :=
by
  -- Math proof goes here
  sorry

end NUMINAMATH_GPT_Debby_bought_bottles_l2240_224016


namespace NUMINAMATH_GPT_min_value_expr_l2240_224027

open Real

theorem min_value_expr(p q r : ℝ)(hp : 0 < p)(hq : 0 < q)(hr : 0 < r) :
  (5 * r / (3 * p + q) + 5 * p / (q + 3 * r) + 4 * q / (2 * p + 2 * r)) ≥ 5 / 2 :=
sorry

end NUMINAMATH_GPT_min_value_expr_l2240_224027


namespace NUMINAMATH_GPT_james_profit_correct_l2240_224004

noncomputable def jamesProfit : ℝ :=
  let tickets_bought := 200
  let cost_per_ticket := 2
  let winning_ticket_percentage := 0.20
  let percentage_one_dollar := 0.50
  let percentage_three_dollars := 0.30
  let percentage_four_dollars := 0.20
  let percentage_five_dollars := 0.80
  let grand_prize_ticket_count := 1
  let average_remaining_winner := 15
  let tax_percentage := 0.10
  let total_cost := tickets_bought * cost_per_ticket
  let winning_tickets := tickets_bought * winning_ticket_percentage
  let tickets_five_dollars := winning_tickets * percentage_five_dollars
  let other_winning_tickets := winning_tickets - tickets_five_dollars - grand_prize_ticket_count
  let total_winnings_before_tax := (tickets_five_dollars * 5) + (grand_prize_ticket_count * 5000) + (other_winning_tickets * average_remaining_winner)
  let total_tax := total_winnings_before_tax * tax_percentage
  let total_winnings_after_tax := total_winnings_before_tax - total_tax
  total_winnings_after_tax - total_cost

theorem james_profit_correct : jamesProfit = 4338.50 := by
  sorry

end NUMINAMATH_GPT_james_profit_correct_l2240_224004


namespace NUMINAMATH_GPT_perfect_squares_count_in_range_l2240_224040

theorem perfect_squares_count_in_range :
  ∃ (n : ℕ), (
    (∀ (k : ℕ), (50 < k^2 ∧ k^2 < 500) → (8 ≤ k ∧ k ≤ 22)) ∧
    (15 = 22 - 8 + 1)
  ) := sorry

end NUMINAMATH_GPT_perfect_squares_count_in_range_l2240_224040


namespace NUMINAMATH_GPT_XiaoMing_selection_l2240_224024

def final_positions (n : Nat) : List Nat :=
  if n <= 2 then
    List.range n
  else
    final_positions (n / 2) |>.filter (λ k => k % 2 = 0) |>.map (λ k => k / 2)

theorem XiaoMing_selection (n : Nat) (h : n = 32) : final_positions n = [16, 32] :=
  by
  sorry

end NUMINAMATH_GPT_XiaoMing_selection_l2240_224024


namespace NUMINAMATH_GPT_fraction_relevant_quarters_l2240_224066

-- Define the total number of quarters and the number of relevant quarters
def total_quarters : ℕ := 50
def relevant_quarters : ℕ := 10

-- Define the theorem that states the fraction of relevant quarters is 1/5
theorem fraction_relevant_quarters : (relevant_quarters : ℚ) / total_quarters = 1 / 5 := by
  sorry

end NUMINAMATH_GPT_fraction_relevant_quarters_l2240_224066


namespace NUMINAMATH_GPT_shuxue_count_l2240_224093

theorem shuxue_count : 
  (∃ (count : ℕ), count = (List.length (List.filter (λ n => (30 * n.1 + 3 * n.2 < 100) 
    ∧ (30 * n.1 + 3 * n.2 > 9)) 
      (List.product 
        (List.range' 1 3) -- Possible values for "a" are 1 to 3
        (List.range' 1 9)) -- Possible values for "b" are 1 to 9
    ))) ∧ count = 9 :=
  sorry

end NUMINAMATH_GPT_shuxue_count_l2240_224093


namespace NUMINAMATH_GPT_alec_votes_l2240_224042

variable (students totalVotes goalVotes neededVotes : ℕ)

theorem alec_votes (h1 : students = 60)
                   (h2 : goalVotes = 3 * students / 4)
                   (h3 : totalVotes = students / 2 + 5 + (students - (students / 2 + 5)) / 5)
                   (h4 : neededVotes = goalVotes - totalVotes) :
                   neededVotes = 5 :=
by sorry

end NUMINAMATH_GPT_alec_votes_l2240_224042


namespace NUMINAMATH_GPT_solve_oranges_problem_find_plans_and_max_profit_l2240_224094

theorem solve_oranges_problem :
  ∃ (a b : ℕ), 15 * a + 20 * b = 430 ∧ 10 * a + 8 * b = 212 ∧ a = 10 ∧ b = 14 := by
    sorry

theorem find_plans_and_max_profit (a b : ℕ) (h₁ : 15 * a + 20 * b = 430) (h₂ : 10 * a + 8 * b = 212) (ha : a = 10) (hb : b = 14) :
  ∃ (x : ℕ), 58 ≤ x ∧ x ≤ 60 ∧ (10 * x + 14 * (100 - x) ≥ 1160) ∧ (10 * x + 14 * (100 - x) ≤ 1168) ∧ (1000 - 4 * x = 768) :=
    sorry

end NUMINAMATH_GPT_solve_oranges_problem_find_plans_and_max_profit_l2240_224094


namespace NUMINAMATH_GPT_inequality_add_l2240_224005

theorem inequality_add {a b c : ℝ} (h : a > b) : a + c > b + c :=
sorry

end NUMINAMATH_GPT_inequality_add_l2240_224005


namespace NUMINAMATH_GPT_pages_per_sheet_is_one_l2240_224009

-- Definition of conditions
def stories_per_week : Nat := 3
def pages_per_story : Nat := 50
def num_weeks : Nat := 12
def reams_bought : Nat := 3
def sheets_per_ream : Nat := 500

-- Calculate total pages written over num_weeks (short stories only)
def total_pages : Nat := stories_per_week * pages_per_story * num_weeks

-- Calculate total sheets available
def total_sheets : Nat := reams_bought * sheets_per_ream

-- Calculate pages per sheet, rounding to nearest whole number
def pages_per_sheet : Nat := (total_pages / total_sheets)

-- The main statement to prove
theorem pages_per_sheet_is_one : pages_per_sheet = 1 :=
by
  sorry

end NUMINAMATH_GPT_pages_per_sheet_is_one_l2240_224009


namespace NUMINAMATH_GPT_solve_for_x_l2240_224006

theorem solve_for_x (x : ℝ) (h : x / 6 = 15 / 10) : x = 9 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l2240_224006


namespace NUMINAMATH_GPT_final_people_amount_l2240_224084

def initial_people : ℕ := 250
def people_left1 : ℕ := 35
def people_joined1 : ℕ := 20
def percentage_left : ℕ := 10
def groups_joined : ℕ := 4
def group_size : ℕ := 15

theorem final_people_amount :
  let intermediate_people1 := initial_people - people_left1;
  let intermediate_people2 := intermediate_people1 + people_joined1;
  let people_left2 := (intermediate_people2 * percentage_left) / 100;
  let rounded_people_left2 := people_left2;
  let intermediate_people3 := intermediate_people2 - rounded_people_left2;
  let total_new_join := groups_joined * group_size;
  let final_people := intermediate_people3 + total_new_join;
  final_people = 272 :=
by sorry

end NUMINAMATH_GPT_final_people_amount_l2240_224084


namespace NUMINAMATH_GPT_slower_speed_percentage_l2240_224029

noncomputable def usual_speed_time : ℕ := 16  -- usual time in minutes
noncomputable def additional_time : ℕ := 24  -- additional time in minutes

theorem slower_speed_percentage (S S_slow : ℝ) (D : ℝ) 
  (h1 : D = S * usual_speed_time) 
  (h2 : D = S_slow * (usual_speed_time + additional_time)) : 
  (S_slow / S) * 100 = 40 :=
by 
  sorry

end NUMINAMATH_GPT_slower_speed_percentage_l2240_224029


namespace NUMINAMATH_GPT_time_before_Car_Y_started_in_minutes_l2240_224046

noncomputable def timeBeforeCarYStarted (speedX speedY distanceX : ℝ) : ℝ :=
  let t := distanceX / speedX
  (speedY * t - distanceX) / speedX

theorem time_before_Car_Y_started_in_minutes 
  (speedX speedY distanceX : ℝ)
  (h_speedX : speedX = 35)
  (h_speedY : speedY = 70)
  (h_distanceX : distanceX = 42) : 
  (timeBeforeCarYStarted speedX speedY distanceX) * 60 = 72 :=
by
  sorry

end NUMINAMATH_GPT_time_before_Car_Y_started_in_minutes_l2240_224046


namespace NUMINAMATH_GPT_car_mpg_in_city_l2240_224052

theorem car_mpg_in_city 
    (miles_per_tank_highway : Real)
    (miles_per_tank_city : Real)
    (mpg_difference : Real)
    : True := by
  let H := 21.05
  let T := 720 / H
  let C := H - 10
  have h1 : 720 = H * T := by
    sorry
  have h2 : 378 = C * T := by
    sorry
  exact True.intro

end NUMINAMATH_GPT_car_mpg_in_city_l2240_224052


namespace NUMINAMATH_GPT_g_sum_1_2_3_2_l2240_224032

def g (a b : ℚ) : ℚ :=
  if a + b ≤ 4 then
    (a * b + a - 3) / (3 * a)
  else
    (a * b + b + 3) / (-3 * b)

theorem g_sum_1_2_3_2 : g 1 2 + g 3 2 = -11 / 6 :=
by sorry

end NUMINAMATH_GPT_g_sum_1_2_3_2_l2240_224032


namespace NUMINAMATH_GPT_min_dot_product_on_hyperbola_l2240_224082

theorem min_dot_product_on_hyperbola (x1 y1 x2 y2 : ℝ) 
  (hA : x1^2 - y1^2 = 2) 
  (hB : x2^2 - y2^2 = 2)
  (h_x1 : x1 > 0) 
  (h_x2 : x2 > 0) : 
  x1 * x2 + y1 * y2 ≥ 2 :=
sorry

end NUMINAMATH_GPT_min_dot_product_on_hyperbola_l2240_224082


namespace NUMINAMATH_GPT_recommended_apps_l2240_224097

namespace RogerPhone

-- Let's define the conditions.
def optimalApps : ℕ := 50
def currentApps (R : ℕ) : ℕ := 2 * R
def appsToDelete : ℕ := 20

-- Defining the problem as a theorem.
theorem recommended_apps (R : ℕ) (h1 : 2 * R = optimalApps + appsToDelete) : R = 35 := by
  sorry

end RogerPhone

end NUMINAMATH_GPT_recommended_apps_l2240_224097


namespace NUMINAMATH_GPT_pete_total_miles_l2240_224039

-- Definitions based on conditions
def flip_step_count : ℕ := 89999
def steps_full_cycle : ℕ := 90000
def total_flips : ℕ := 52
def end_year_reading : ℕ := 55555
def steps_per_mile : ℕ := 1900

-- Total steps Pete walked
def total_steps_pete_walked (flips : ℕ) (end_reading : ℕ) : ℕ :=
  flips * steps_full_cycle + end_reading

-- Total miles Pete walked
def total_miles_pete_walked (steps : ℕ) (steps_per_mile : ℕ) : ℕ :=
  steps / steps_per_mile

-- Given the parameters, closest number of miles Pete walked should be 2500
theorem pete_total_miles : total_miles_pete_walked (total_steps_pete_walked total_flips end_year_reading) steps_per_mile = 2500 :=
by
  sorry

end NUMINAMATH_GPT_pete_total_miles_l2240_224039


namespace NUMINAMATH_GPT_num_12_digit_with_consecutive_ones_l2240_224051

theorem num_12_digit_with_consecutive_ones :
  let total := 3^12
  let F12 := 985
  total - F12 = 530456 :=
by
  let total := 3^12
  let F12 := 985
  have h : total - F12 = 530456
  sorry
  exact h

end NUMINAMATH_GPT_num_12_digit_with_consecutive_ones_l2240_224051


namespace NUMINAMATH_GPT_maximal_length_sequence_l2240_224079

theorem maximal_length_sequence :
  ∃ (a : ℕ → ℤ) (n : ℕ), (∀ i, 1 ≤ i → i + 6 ≤ n → (a i + a (i + 1) + a (i + 2) + a (i + 3) + a (i + 4) + a (i + 5) + a (i + 6) > 0)) ∧ 
                          (∀ j, 1 ≤ j → j + 10 ≤ n → (a j + a (j + 1) + a (j + 2) + a (j + 3) + a (j + 4) + a (j + 5) + a (j + 6) + a (j + 7) + a (j + 8) + a (j + 9) + a (j + 10) < 0)) ∧ 
                          n = 16 :=
sorry

end NUMINAMATH_GPT_maximal_length_sequence_l2240_224079


namespace NUMINAMATH_GPT_angle_bisector_theorem_l2240_224072

noncomputable def ratio_of_segments (x y z p q : ℝ) :=
  q / x = y / (y + x)

theorem angle_bisector_theorem (x y z p q : ℝ) (h1 : p / x = q / y)
  (h2 : p + q = z) : ratio_of_segments x y z p q :=
by
  sorry

end NUMINAMATH_GPT_angle_bisector_theorem_l2240_224072


namespace NUMINAMATH_GPT_original_class_size_l2240_224054

theorem original_class_size
  (N : ℕ)
  (h1 : 40 * N = T)
  (h2 : T + 15 * 32 = 36 * (N + 15)) :
  N = 15 := by
  sorry

end NUMINAMATH_GPT_original_class_size_l2240_224054


namespace NUMINAMATH_GPT_find_multiplier_l2240_224034

theorem find_multiplier (n x : ℝ) (h1 : n = 1.0) (h2 : 3 * n - 1 = x * n) : x = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_multiplier_l2240_224034


namespace NUMINAMATH_GPT_directors_dividends_correct_l2240_224047

theorem directors_dividends_correct :
  let net_profit : ℝ := (1500000 - 674992) - 0.2 * (1500000 - 674992)
  let total_loan_payments : ℝ := 23914 * 12 - 74992
  let profit_for_dividends : ℝ := net_profit - total_loan_payments
  let dividend_per_share : ℝ := profit_for_dividends / 1000
  let total_dividends_director : ℝ := dividend_per_share * 550
  total_dividends_director = 246400.0 :=
by
  sorry

end NUMINAMATH_GPT_directors_dividends_correct_l2240_224047


namespace NUMINAMATH_GPT_probability_red_or_white_correct_l2240_224017

-- Define the conditions
def totalMarbles : ℕ := 30
def blueMarbles : ℕ := 5
def redMarbles : ℕ := 9
def whiteMarbles : ℕ := totalMarbles - (blueMarbles + redMarbles)

-- Define the calculated probability
def probabilityRedOrWhite : ℚ := (redMarbles + whiteMarbles) / totalMarbles

-- Verify the probability is equal to 5 / 6
theorem probability_red_or_white_correct :
  probabilityRedOrWhite = 5 / 6 := by
  sorry

end NUMINAMATH_GPT_probability_red_or_white_correct_l2240_224017


namespace NUMINAMATH_GPT_ratio_major_minor_is_15_4_l2240_224089

-- Define the given conditions
def main_characters : ℕ := 5
def minor_characters : ℕ := 4
def minor_character_pay : ℕ := 15000
def total_payment : ℕ := 285000

-- Define the total pay to minor characters
def minor_total_pay : ℕ := minor_characters * minor_character_pay

-- Define the total pay to major characters
def major_total_pay : ℕ := total_payment - minor_total_pay

-- Define the ratio computation
def ratio_major_minor : ℕ × ℕ := (major_total_pay / 15000, minor_total_pay / 15000)

-- State the theorem
theorem ratio_major_minor_is_15_4 : ratio_major_minor = (15, 4) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_ratio_major_minor_is_15_4_l2240_224089


namespace NUMINAMATH_GPT_compute_binom_product_l2240_224087

/-- Definition of binomial coefficient -/
def binomial (n k : ℕ) : ℕ := n.choose k

/-- The main theorem to prove -/
theorem compute_binom_product : binomial 10 3 * binomial 8 3 = 6720 :=
by
  sorry

end NUMINAMATH_GPT_compute_binom_product_l2240_224087


namespace NUMINAMATH_GPT_sequence_all_ones_l2240_224061

theorem sequence_all_ones (k : ℕ) (n : ℕ → ℕ) (h_k : 2 ≤ k)
  (h1 : ∀ i, 1 ≤ i → i ≤ k → 1 ≤ n i) 
  (h2 : n 2 ∣ 2^(n 1) - 1) 
  (h3 : n 3 ∣ 2^(n 2) - 1) 
  (h4 : n 4 ∣ 2^(n 3) - 1)
  (h5 : ∀ i, 2 ≤ i → i < k → n (i + 1) ∣ 2^(n i) - 1)
  (h6 : n 1 ∣ 2^(n k) - 1) : 
  ∀ i, 1 ≤ i → i ≤ k → n i = 1 := 
by 
  sorry

end NUMINAMATH_GPT_sequence_all_ones_l2240_224061


namespace NUMINAMATH_GPT_solve_for_x_l2240_224048

-- Define the new operation m ※ n
def operation (m n : ℤ) : ℤ :=
  if m ≥ 0 then m + n else m / n

-- Define the condition given in the problem
def condition (x : ℤ) : Prop :=
  operation (-9) (-x) = x

-- The main theorem to prove
theorem solve_for_x (x : ℤ) : condition x ↔ (x = 3 ∨ x = -3) :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l2240_224048


namespace NUMINAMATH_GPT_car_fuel_efficiency_in_city_l2240_224045

theorem car_fuel_efficiency_in_city 
    (H C T : ℝ) 
    (h1 : H * T = 462) 
    (h2 : (H - 15) * T = 336) : 
    C = 40 :=
by 
    sorry

end NUMINAMATH_GPT_car_fuel_efficiency_in_city_l2240_224045


namespace NUMINAMATH_GPT_net_profit_calc_l2240_224060

theorem net_profit_calc (purchase_price : ℕ) (overhead_percentage : ℝ) (markup : ℝ) 
  (h_pp : purchase_price = 48) (h_op : overhead_percentage = 0.10) (h_markup : markup = 35) :
  let overhead := overhead_percentage * purchase_price
  let net_profit := markup - overhead
  net_profit = 30.20 := by
    sorry

end NUMINAMATH_GPT_net_profit_calc_l2240_224060


namespace NUMINAMATH_GPT_sum_of_coefficients_l2240_224038

def polynomial (x y : ℕ) : ℕ := (x^2 - 3*x*y + y^2)^8

theorem sum_of_coefficients : polynomial 1 1 = 1 :=
sorry

end NUMINAMATH_GPT_sum_of_coefficients_l2240_224038


namespace NUMINAMATH_GPT_smallest_integer_y_l2240_224083

theorem smallest_integer_y (y : ℤ) : (5 : ℝ) / 8 < (y : ℝ) / 17 → y = 11 := by
  sorry

end NUMINAMATH_GPT_smallest_integer_y_l2240_224083


namespace NUMINAMATH_GPT_probability_all_selected_l2240_224068

theorem probability_all_selected (P_Ram P_Ravi P_Ritu : ℚ) 
  (h1 : P_Ram = 3 / 7) 
  (h2 : P_Ravi = 1 / 5) 
  (h3 : P_Ritu = 2 / 9) : 
  P_Ram * P_Ravi * P_Ritu = 2 / 105 := 
by
  sorry

end NUMINAMATH_GPT_probability_all_selected_l2240_224068


namespace NUMINAMATH_GPT_solve_system_of_inequalities_l2240_224025

variable {R : Type*} [LinearOrderedField R]

theorem solve_system_of_inequalities (x1 x2 x3 x4 x5 : R)
  (h1 : x1 > 0) (h2 : x2 > 0) (h3 : x3 > 0) (h4 : x4 > 0) (h5 : x5 > 0) :
  (x1^2 - x3^2) * (x2^2 - x3^2) ≤ 0 ∧ 
  (x3^2 - x1^2) * (x3^2 - x1^2) ≤ 0 ∧ 
  (x3^2 - x3 * x2) * (x1^2 - x3 * x2) ≤ 0 ∧ 
  (x1^2 - x1 * x3) * (x3^2 - x1 * x3) ≤ 0 ∧ 
  (x3^2 - x2 * x1) * (x1^2 - x2 * x1) ≤ 0 →
  x1 = x2 ∧ x2 = x3 ∧ x3 = x4 ∧ x4 = x5 :=
sorry

end NUMINAMATH_GPT_solve_system_of_inequalities_l2240_224025


namespace NUMINAMATH_GPT_find_total_roses_l2240_224035

open Nat

theorem find_total_roses 
  (a : ℕ)
  (h1 : 300 ≤ a)
  (h2 : a ≤ 400)
  (h3 : a % 21 = 13)
  (h4 : a % 15 = 7) : 
  a = 307 := 
sorry

end NUMINAMATH_GPT_find_total_roses_l2240_224035


namespace NUMINAMATH_GPT_spiral_wire_length_l2240_224096

noncomputable def wire_length (turns : ℕ) (height : ℝ) (circumference : ℝ) : ℝ :=
  Real.sqrt (height^2 + (turns * circumference)^2)

theorem spiral_wire_length
  (turns : ℕ) (height : ℝ) (circumference : ℝ)
  (turns_eq : turns = 10)
  (height_eq : height = 9)
  (circumference_eq : circumference = 4) :
  wire_length turns height circumference = 41 := 
by
  rw [turns_eq, height_eq, circumference_eq]
  simp [wire_length]
  norm_num
  rw [Real.sqrt_eq_rpow]
  norm_num
  sorry

end NUMINAMATH_GPT_spiral_wire_length_l2240_224096


namespace NUMINAMATH_GPT_eighth_term_of_geometric_sequence_l2240_224000

def geometric_sequence_term (a₁ r : ℚ) (n : ℕ) : ℚ :=
  a₁ * r^(n - 1)

theorem eighth_term_of_geometric_sequence : 
  geometric_sequence_term 12 (1 / 3) 8 = 4 / 729 :=
by 
  sorry

end NUMINAMATH_GPT_eighth_term_of_geometric_sequence_l2240_224000


namespace NUMINAMATH_GPT_point_not_in_second_quadrant_l2240_224030

theorem point_not_in_second_quadrant (a : ℝ) :
  (∃ b : ℝ, b = 2 * a - 1) ∧ ¬(a < 0 ∧ (2 * a - 1 > 0)) := 
by sorry

end NUMINAMATH_GPT_point_not_in_second_quadrant_l2240_224030


namespace NUMINAMATH_GPT_minimum_y_squared_l2240_224070

theorem minimum_y_squared :
  let consecutive_sum (x : ℤ) := (x-5)^2 + (x-4)^2 + (x-3)^2 + (x-2)^2 + (x-1)^2 + x^2 + (x+1)^2 + (x+2)^2 + (x+3)^2 + (x+4)^2 + (x+5)^2
  ∃ y : ℤ, y^2 = 11 * (1^2 + 10) ∧ ∀ z : ℤ, z^2 = 11 * consecutive_sum z → y^2 ≤ z^2 := by
sorry

end NUMINAMATH_GPT_minimum_y_squared_l2240_224070


namespace NUMINAMATH_GPT_change_received_l2240_224049

def totalCostBeforeDiscount : ℝ :=
  5.75 + 2.50 + 3.25 + 3.75 + 4.20

def discount : ℝ :=
  (3.75 + 4.20) * 0.10

def totalCostAfterDiscount : ℝ :=
  totalCostBeforeDiscount - discount

def salesTax : ℝ :=
  totalCostAfterDiscount * 0.06

def finalTotalCost : ℝ :=
  totalCostAfterDiscount + salesTax

def amountPaid : ℝ :=
  50.00

def change : ℝ :=
  amountPaid - finalTotalCost

theorem change_received (h : change = 30.34) : change = 30.34 := by
  sorry

end NUMINAMATH_GPT_change_received_l2240_224049


namespace NUMINAMATH_GPT_interest_calculation_years_l2240_224071

theorem interest_calculation_years (P n : ℝ) (r : ℝ) (SI CI : ℝ)
  (h₁ : SI = P * r * n / 100)
  (h₂ : r = 5)
  (h₃ : SI = 50)
  (h₄ : CI = P * ((1 + r / 100)^n - 1))
  (h₅ : CI = 51.25) :
  n = 2 := by
  sorry

end NUMINAMATH_GPT_interest_calculation_years_l2240_224071


namespace NUMINAMATH_GPT_units_digit_sum_cubes_l2240_224031

theorem units_digit_sum_cubes (n1 n2 : ℕ) 
  (h1 : n1 = 24) (h2 : n2 = 42) : 
  (n1 ^ 3 + n2 ^ 3) % 10 = 6 :=
by 
  -- substitution based on h1 and h2 can be done here.
  sorry

end NUMINAMATH_GPT_units_digit_sum_cubes_l2240_224031


namespace NUMINAMATH_GPT_sum_seven_terms_l2240_224028

-- Definitions and conditions
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (d : ℝ), ∀ n : ℕ, a (n + 1) - a n = d

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n : ℝ) * (a 1 + a n) / 2

-- Given condition
def given_condition (a : ℕ → ℝ) : Prop :=
  a 2 + a 4 + a 6 = 42

-- Proof statement
theorem sum_seven_terms (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : sum_of_first_n_terms a S) 
  (h_cond : given_condition a) : 
  S 7 = 98 := 
sorry

end NUMINAMATH_GPT_sum_seven_terms_l2240_224028


namespace NUMINAMATH_GPT_problem_statement_l2240_224081

def oper (x : ℕ) (w : ℕ) := (2^x) / (2^w)

theorem problem_statement : ∃ n : ℕ, oper (oper 4 2) n = 2 ↔ n = 3 :=
by sorry

end NUMINAMATH_GPT_problem_statement_l2240_224081


namespace NUMINAMATH_GPT_simplify_expression_l2240_224065

theorem simplify_expression (x : ℝ) : (3 * x) ^ 5 - (4 * x) * (x ^ 4) = 239 * x ^ 5 := 
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2240_224065


namespace NUMINAMATH_GPT_delivery_newspapers_15_houses_l2240_224018

-- State the problem using Lean 4 syntax

noncomputable def delivery_sequences (n : ℕ) : ℕ :=
  if h : n < 3 then 2^n
  else if n = 3 then 6
  else delivery_sequences (n-1) + delivery_sequences (n-2) + delivery_sequences (n-3)

theorem delivery_newspapers_15_houses :
  delivery_sequences 15 = 849 :=
sorry

end NUMINAMATH_GPT_delivery_newspapers_15_houses_l2240_224018


namespace NUMINAMATH_GPT_largest_fraction_l2240_224075

theorem largest_fraction :
  let frac1 := (5 : ℚ) / 12
  let frac2 := (7 : ℚ) / 15
  let frac3 := (23 : ℚ) / 45
  let frac4 := (89 : ℚ) / 178
  let frac5 := (199 : ℚ) / 400
  frac3 > frac1 ∧ frac3 > frac2 ∧ frac3 > frac4 ∧ frac3 > frac5 :=
by
  let frac1 := (5 : ℚ) / 12
  let frac2 := (7 : ℚ) / 15
  let frac3 := (23 : ℚ) / 45
  let frac4 := (89 : ℚ) / 178
  let frac5 := (199 : ℚ) / 400
  sorry

end NUMINAMATH_GPT_largest_fraction_l2240_224075


namespace NUMINAMATH_GPT_original_number_exists_l2240_224022

theorem original_number_exists 
  (N: ℤ)
  (h1: ∃ (k: ℤ), N - 6 = 16 * k)
  (h2: ∀ (m: ℤ), (N - m) % 16 = 0 → m ≥ 6) : 
  N = 22 :=
sorry

end NUMINAMATH_GPT_original_number_exists_l2240_224022


namespace NUMINAMATH_GPT_seven_pow_l2240_224099

theorem seven_pow (k : ℕ) (h : 7 ^ k = 2) : 7 ^ (4 * k + 2) = 784 :=
by 
  sorry

end NUMINAMATH_GPT_seven_pow_l2240_224099


namespace NUMINAMATH_GPT_solve_x_values_l2240_224064

theorem solve_x_values : ∀ (x : ℝ), (x + 45 / (x - 4) = -10) ↔ (x = -1 ∨ x = -5) :=
by
  intro x
  sorry

end NUMINAMATH_GPT_solve_x_values_l2240_224064


namespace NUMINAMATH_GPT_find_a_l2240_224011

-- Define the hyperbola equation and the asymptote conditions
def hyperbola (a : ℝ) (x y : ℝ) : Prop := (x^2 / a^2) - (y^2 / 9) = 1

def asymptote1 (x y : ℝ) : Prop := 3 * x + 2 * y = 0
def asymptote2 (x y : ℝ) : Prop := 3 * x - 2 * y = 0

-- Prove that if asymptote conditions hold, a = 2
theorem find_a (a : ℝ) (ha : a > 0) :
  (∀ x y, asymptote1 x y) ∧ (∀ x y, asymptote2 x y) → a = 2 :=
sorry

end NUMINAMATH_GPT_find_a_l2240_224011


namespace NUMINAMATH_GPT_negation_of_p_l2240_224010

def p := ∀ x : ℝ, x^2 ≥ 0

theorem negation_of_p : ¬p = (∃ x : ℝ, x^2 < 0) :=
  sorry

end NUMINAMATH_GPT_negation_of_p_l2240_224010


namespace NUMINAMATH_GPT_johnny_needs_45_planks_l2240_224053

theorem johnny_needs_45_planks
  (legs_per_table : ℕ)
  (planks_per_leg : ℕ)
  (surface_planks_per_table : ℕ)
  (number_of_tables : ℕ)
  (h1 : legs_per_table = 4)
  (h2 : planks_per_leg = 1)
  (h3 : surface_planks_per_table = 5)
  (h4 : number_of_tables = 5) :
  number_of_tables * (legs_per_table * planks_per_leg + surface_planks_per_table) = 45 :=
by
  sorry

end NUMINAMATH_GPT_johnny_needs_45_planks_l2240_224053


namespace NUMINAMATH_GPT_average_last_4_matches_l2240_224015

theorem average_last_4_matches (avg_10: ℝ) (avg_6: ℝ) (total_matches: ℕ) (first_matches: ℕ) :
  avg_10 = 38.9 → avg_6 = 42 → total_matches = 10 → first_matches = 6 → 
  (avg_10 * total_matches - avg_6 * first_matches) / (total_matches - first_matches) = 34.25 :=
by 
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_average_last_4_matches_l2240_224015


namespace NUMINAMATH_GPT_mia_has_110_l2240_224085

def darwin_money : ℕ := 45
def mia_money (darwin_money : ℕ) : ℕ := 2 * darwin_money + 20

theorem mia_has_110 :
  mia_money darwin_money = 110 := sorry

end NUMINAMATH_GPT_mia_has_110_l2240_224085


namespace NUMINAMATH_GPT_x3_plus_y3_values_l2240_224063

noncomputable def x_y_satisfy_eqns (x y : ℝ) : Prop :=
  y^2 - 3 = (x - 3)^3 ∧ x^2 - 3 = (y - 3)^2 ∧ x ≠ y

theorem x3_plus_y3_values (x y : ℝ) (h : x_y_satisfy_eqns x y) :
  x^3 + y^3 = 27 + 3 * Real.sqrt 3 ∨ x^3 + y^3 = 27 - 3 * Real.sqrt 3 :=
  sorry

end NUMINAMATH_GPT_x3_plus_y3_values_l2240_224063


namespace NUMINAMATH_GPT_triangle_angle_sum_l2240_224021

/-- In any triangle ABC, the sum of angle A and angle B is given to be 80 degrees.
    We need to prove that the measure of angle C is 100 degrees. -/
theorem triangle_angle_sum (A B C : ℝ) 
  (h1 : A + B = 80)
  (h2 : A + B + C = 180) : C = 100 :=
sorry

end NUMINAMATH_GPT_triangle_angle_sum_l2240_224021


namespace NUMINAMATH_GPT_log_change_of_base_log_change_of_base_with_b_l2240_224077

variable {a b x : ℝ}
variable (h₁ : 0 < a ∧ a ≠ 1)
variable (h₂ : 0 < b ∧ b ≠ 1)
variable (h₃ : 0 < x)

theorem log_change_of_base (h₁ : 0 < a ∧ a ≠ 1) (h₂ : 0 < b ∧ b ≠ 1) (h₃ : 0 < x) : 
  Real.log x / Real.log a = Real.log x / Real.log b := by
  sorry

theorem log_change_of_base_with_b (h₁ : 0 < a ∧ a ≠ 1) (h₂ : 0 < b ∧ b ≠ 1) : 
  Real.log b / Real.log a = 1 / Real.log a := by
  sorry

end NUMINAMATH_GPT_log_change_of_base_log_change_of_base_with_b_l2240_224077


namespace NUMINAMATH_GPT_correct_time_after_2011_minutes_l2240_224074

def time_2011_minutes_after_midnight : String :=
  "2011 minutes after midnight on January 1, 2011 is January 2 at 9:31AM"

theorem correct_time_after_2011_minutes :
  time_2011_minutes_after_midnight = "2011 minutes after midnight on January 1, 2011 is January 2 at 9:31AM" :=
sorry

end NUMINAMATH_GPT_correct_time_after_2011_minutes_l2240_224074


namespace NUMINAMATH_GPT_find_initial_divisor_l2240_224058

theorem find_initial_divisor (N D : ℤ) (h1 : N = 2 * D) (h2 : N % 4 = 2) : D = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_initial_divisor_l2240_224058


namespace NUMINAMATH_GPT_tangent_identity_l2240_224033

theorem tangent_identity (x y z : ℝ) (h : x + y + z = x * y * z) :
  (3 * x - x^3) / (1 - 3 * x^2) + (3 * y - y^3) / (1 - 3 * y^2) + (3 * z - z^3) / (1 - 3 * z^2)
  = ((3 * x - x^3) / (1 - 3 * x^2)) * ((3 * y - y^3) / (1 - 3 * y^2)) * ((3 * z - z^3) / (1 - 3 * z^2)) :=
sorry

end NUMINAMATH_GPT_tangent_identity_l2240_224033


namespace NUMINAMATH_GPT_ice_cream_eaten_l2240_224080

variables (f : ℝ)

theorem ice_cream_eaten (h : f + 0.25 = 3.5) : f = 3.25 :=
sorry

end NUMINAMATH_GPT_ice_cream_eaten_l2240_224080


namespace NUMINAMATH_GPT_find_y_l2240_224003

theorem find_y (y : ℝ) : 3 + 1 / (2 - y) = 2 * (1 / (2 - y)) → y = 5 / 3 := 
by
  sorry

end NUMINAMATH_GPT_find_y_l2240_224003


namespace NUMINAMATH_GPT_vacation_days_proof_l2240_224037

-- Define the conditions
def family_vacation (total_days rain_days clear_afternoons : ℕ) : Prop :=
  total_days = 18 ∧ rain_days = 13 ∧ clear_afternoons = 12

-- State the theorem to be proved
theorem vacation_days_proof : family_vacation 18 13 12 → 18 = 18 :=
by
  -- Skip the proof
  intro h
  sorry

end NUMINAMATH_GPT_vacation_days_proof_l2240_224037


namespace NUMINAMATH_GPT_division_addition_correct_l2240_224069

-- Define a function that performs the arithmetic operations described
def calculateResult : ℕ :=
  let division := 12 * 4 -- dividing 12 by 1/4 is the same as multiplying by 4
  division + 5 -- then add 5 to the result

-- The theorem statement to prove
theorem division_addition_correct : calculateResult = 53 := by
  sorry

end NUMINAMATH_GPT_division_addition_correct_l2240_224069


namespace NUMINAMATH_GPT_fraction_red_marbles_after_doubling_l2240_224020

theorem fraction_red_marbles_after_doubling (x : ℕ) (h : x > 0) :
  let blue_fraction : ℚ := 3 / 5
  let red_fraction := 1 - blue_fraction
  let initial_blue_marbles := blue_fraction * x
  let initial_red_marbles := red_fraction * x
  let new_red_marbles := 2 * initial_red_marbles
  let new_total_marbles := initial_blue_marbles + new_red_marbles
  let new_red_fraction := new_red_marbles / new_total_marbles
  new_red_fraction = 4 / 7 :=
sorry

end NUMINAMATH_GPT_fraction_red_marbles_after_doubling_l2240_224020


namespace NUMINAMATH_GPT_union_A_B_union_complement_A_B_l2240_224067

open Set

-- Definitions for sets U, A, and B
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {1, 3, 5, 7}
def B : Set ℕ := {3, 5}

-- Statement 1: Prove that A ∪ B = {1, 3, 5, 7}
theorem union_A_B : A ∪ B = {1, 3, 5, 7} := by
  sorry

-- Definition for complement of A in U
def complement_A_U : Set ℕ := {x ∈ U | x ∉ A}

-- Statement 2: Prove that (complement of A in U) ∪ B = {2, 3, 4, 5, 6}
theorem union_complement_A_B : complement_A_U ∪ B = {2, 3, 4, 5, 6} := by
  sorry

end NUMINAMATH_GPT_union_A_B_union_complement_A_B_l2240_224067


namespace NUMINAMATH_GPT_solutions_of_quadratic_eq_l2240_224090

theorem solutions_of_quadratic_eq : 
    {x : ℝ | x^2 - 3 * x = 0} = {0, 3} :=
sorry

end NUMINAMATH_GPT_solutions_of_quadratic_eq_l2240_224090


namespace NUMINAMATH_GPT_average_score_correct_l2240_224007

-- Define the conditions
def simplified_scores : List Int := [10, -5, 0, 8, -3]
def base_score : Int := 90

-- Translate simplified score to actual score
def actual_score (s : Int) : Int :=
  base_score + s

-- Calculate the average of the actual scores
def average_score : Int :=
  (simplified_scores.map actual_score).sum / simplified_scores.length

-- The proof statement
theorem average_score_correct : average_score = 92 := 
by 
  -- Steps to compute the average score
  -- sorry is used since the proof steps are not required
  sorry

end NUMINAMATH_GPT_average_score_correct_l2240_224007


namespace NUMINAMATH_GPT_bowling_ball_weight_l2240_224073

def weight_of_canoe : ℕ := 32
def weight_of_canoes (n : ℕ) := n * weight_of_canoe
def weight_of_bowling_balls (n : ℕ) := 128

theorem bowling_ball_weight :
  (128 / 5 : ℚ) = (weight_of_bowling_balls 5 / 5 : ℚ) :=
by
  -- Theorems and calculations would typically be carried out here
  sorry

end NUMINAMATH_GPT_bowling_ball_weight_l2240_224073


namespace NUMINAMATH_GPT_common_measure_angle_l2240_224056

theorem common_measure_angle (α β : ℝ) (m n : ℕ) (h : α = β * (m / n)) : α / m = β / n :=
by 
  sorry

end NUMINAMATH_GPT_common_measure_angle_l2240_224056


namespace NUMINAMATH_GPT_number_of_groups_of_oranges_l2240_224057

-- Defining the conditions
def total_oranges : ℕ := 356
def oranges_per_group : ℕ := 2

-- The proof statement
theorem number_of_groups_of_oranges : total_oranges / oranges_per_group = 178 := 
by 
  sorry

end NUMINAMATH_GPT_number_of_groups_of_oranges_l2240_224057


namespace NUMINAMATH_GPT_sum_of_areas_of_triangles_in_cube_l2240_224095

theorem sum_of_areas_of_triangles_in_cube : 
  let m := 48
  let n := 4608
  let p := 576
  m + n + p = 5232 := by 
    sorry

end NUMINAMATH_GPT_sum_of_areas_of_triangles_in_cube_l2240_224095


namespace NUMINAMATH_GPT_tank_ratio_two_l2240_224050

variable (T1 : ℕ) (F1 : ℕ) (F2 : ℕ) (T2 : ℕ)

-- Assume the given conditions
axiom h1 : T1 = 48
axiom h2 : F1 = T1 / 3
axiom h3 : F1 - 1 = F2 + 3
axiom h4 : T2 = F2 * 2

-- The theorem to prove
theorem tank_ratio_two (h1 : T1 = 48) (h2 : F1 = T1 / 3) (h3 : F1 - 1 = F2 + 3) (h4 : T2 = F2 * 2) : T1 / T2 = 2 := by
  sorry

end NUMINAMATH_GPT_tank_ratio_two_l2240_224050
