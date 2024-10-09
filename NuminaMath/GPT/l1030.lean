import Mathlib

namespace domain_of_f_eq_l1030_103030

noncomputable def domain_of_f (x : ℝ) : Prop :=
  (x + 1 ≥ 0) ∧ (x ≠ 0)

theorem domain_of_f_eq :
  { x : ℝ | domain_of_f x} = { x : ℝ | -1 ≤ x ∧ x < 0 } ∪ { x : ℝ | 0 < x } :=
by
  sorry

end domain_of_f_eq_l1030_103030


namespace tangents_secant_intersect_l1030_103065

variable {A B C O1 P Q R : Type} 
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace P] [MetricSpace Q] [MetricSpace R]
variables (AB AC : Set (MetricSpace A)) (t : Tangent AB) (s : Tangent AC)

variable (BC : line ( Set A))
variable (APQ : secant A P Q) 

theorem tangents_secant_intersect { AR AP AQ : ℝ } :
  2 / AR = 1 / AP + 1 / AQ :=
by
  sorry

end tangents_secant_intersect_l1030_103065


namespace lily_pads_half_lake_l1030_103085

noncomputable def size (n : ℕ) : ℝ := sorry

theorem lily_pads_half_lake {n : ℕ} (h : size 48 = size 0 * 2^48) : size 47 = (size 48) / 2 :=
by 
  sorry

end lily_pads_half_lake_l1030_103085


namespace debt_amount_is_40_l1030_103092

theorem debt_amount_is_40 (l n t debt remaining : ℕ) (h_l : l = 6)
  (h_n1 : n = 5 * l) (h_n2 : n = 3 * t) (h_remaining : remaining = 6) 
  (h_share : ∀ x y z : ℕ, x = y ∧ y = z ∧ z = 2) :
  debt = 40 := 
by
  sorry

end debt_amount_is_40_l1030_103092


namespace find_reading_l1030_103009

variable (a_1 a_2 a_3 a_4 : ℝ) (x : ℝ)
variable (h1 : a_1 = 2) (h2 : a_2 = 2.1) (h3 : a_3 = 2) (h4 : a_4 = 2.2)
variable (mean : (a_1 + a_2 + a_3 + a_4 + x) / 5 = 2)

theorem find_reading : x = 1.7 :=
by
  sorry

end find_reading_l1030_103009


namespace distance_A_B_l1030_103031

noncomputable def distance_3d (x1 y1 z1 x2 y2 z2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2 + (z2 - z1) ^ 2)

theorem distance_A_B :
  distance_3d 4 1 9 10 (-1) 6 = 7 :=
by
  sorry

end distance_A_B_l1030_103031


namespace find_AC_find_area_l1030_103010

theorem find_AC (BC : ℝ) (angleA : ℝ) (cosB : ℝ) 
(hBC : BC = Real.sqrt 7) (hAngleA : angleA = 60) (hCosB : cosB = Real.sqrt 6 / 3) :
  (AC : ℝ) → (hAC : AC = 2 * Real.sqrt 7 / 3) → Prop :=
by
  sorry

theorem find_area (BC AB : ℝ) (angleA : ℝ) 
(hBC : BC = Real.sqrt 7) (hAB : AB = 2) (hAngleA : angleA = 60) :
  (area : ℝ) → (hArea : area = 3 * Real.sqrt 3 / 2) → Prop :=
by
  sorry

end find_AC_find_area_l1030_103010


namespace baba_yagas_savings_plan_l1030_103015

-- Definitions for income and expenses
def salary (gross: ℝ) (taxRate: ℝ) : ℝ := gross * (1 - taxRate)

def familyIncome (ivansSalary vasilisasSalary mothersSalary fathersSalary sonsStateScholarship sonsNonStateScholarship: ℝ)
  (mothersPension: ℝ) (taxRate: ℝ) (date: ℕ) : ℝ :=
  if date < 20180501 then
    (salary ivansSalary taxRate) + (salary vasilisasSalary taxRate) + (salary mothersSalary taxRate) + 
    (salary fathersSalary taxRate) + sonsStateScholarship
  else if date < 20180901 then
    (salary ivansSalary taxRate) + (salary vasilisasSalary taxRate) + mothersPension + 
    (salary fathersSalary taxRate) + sonsStateScholarship
  else
    (salary ivansSalary taxRate) + (salary vasilisasSalary taxRate) + mothersPension + 
    (salary fathersSalary taxRate) + sonsStateScholarship + (salary sonsNonStateScholarship taxRate)

def monthlyExpenses : ℝ := 74000

def monthlySavings (income: ℝ) (expenses: ℝ) : ℝ := income - expenses

-- Theorem to prove
theorem baba_yagas_savings_plan :
  ∀ (ivansSalary vasilisasSalary mothersSalary fathersSalary sonsStateScholarship sonsNonStateScholarship mothersPension: ℝ)
  (taxRate: ℝ),
  ivansSalary = 55000 → vasilisasSalary = 45000 → mothersSalary = 18000 →
  fathersSalary = 20000 → sonsStateScholarship = 3000 → sonsNonStateScholarship = 15000 →
  mothersPension = 10000 → taxRate = 0.13 →
  monthlySavings (familyIncome ivansSalary vasilisasSalary mothersSalary fathersSalary sonsStateScholarship
    sonsNonStateScholarship mothersPension taxRate 20180430) monthlyExpenses = 49060 ∧
  monthlySavings (familyIncome ivansSalary vasilisasSalary mothersSalary fathersSalary sonsStateScholarship
    sonsNonStateScholarship mothersPension taxRate 20180501) monthlyExpenses = 43400 ∧
  monthlySavings (familyIncome ivansSalary vasilisasSalary mothersSalary fathersSalary sonsStateScholarship
    sonsNonStateScholarship mothersPension taxRate 20180901) monthlyExpenses = 56450 :=
by
  sorry

end baba_yagas_savings_plan_l1030_103015


namespace original_money_l1030_103063
noncomputable def original_amount (x : ℝ) :=
  let after_first_loss := (2/3) * x
  let after_first_win := after_first_loss + 10
  let after_second_loss := after_first_win - (1/3) * after_first_win
  let after_second_win := after_second_loss + 20
  after_second_win

theorem original_money (x : ℝ) (h : original_amount x = x) : x = 48 :=
by {
  sorry
}

end original_money_l1030_103063


namespace length_of_bridge_l1030_103045

theorem length_of_bridge 
    (length_of_train : ℕ)
    (speed_of_train_km_per_hr : ℕ)
    (time_to_cross_seconds : ℕ)
    (bridge_length : ℕ) 
    (h_train_length : length_of_train = 130)
    (h_speed_train : speed_of_train_km_per_hr = 54)
    (h_time_cross : time_to_cross_seconds = 30)
    (h_bridge_length : bridge_length = 320) : 
    bridge_length = 320 :=
by sorry

end length_of_bridge_l1030_103045


namespace find_smaller_number_l1030_103068

-- Define the conditions as hypotheses and the goal as a proposition
theorem find_smaller_number (x y : ℕ) (h1 : x + y = 77) (h2 : x = 42 ∨ y = 42) (h3 : 5 * x = 6 * y) : x = 35 :=
sorry

end find_smaller_number_l1030_103068


namespace odd_function_value_at_neg2_l1030_103094

variable (f : ℝ → ℝ)
variable (h_odd : ∀ x, f (-x) = -f x)
variable (h_ge_one : ∀ x, 1 ≤ x → f x = 3 * x - 7)

theorem odd_function_value_at_neg2 : f (-2) = 1 :=
by
  -- Proof goes here
  sorry

end odd_function_value_at_neg2_l1030_103094


namespace problem_l1030_103047

noncomputable def g (x : ℝ) : ℝ := 3^x + 2

theorem problem (x : ℝ) : g (x + 1) - g x = 2 * g x - 2 := sorry

end problem_l1030_103047


namespace min_value_of_sum_l1030_103074

theorem min_value_of_sum (x y : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : x * y + 2 * x + y = 4) : x + y ≥ 2 * Real.sqrt 6 - 3 :=
sorry

end min_value_of_sum_l1030_103074


namespace coord_of_point_B_l1030_103055
-- Necessary import for mathematical definitions and structures

-- Define the initial point A and the translation conditions
def point_A : ℝ × ℝ := (1, -2)
def translation_up (p : ℝ × ℝ) (units : ℝ) : ℝ × ℝ := (p.1, p.2 + units)

-- The target point B after translation
def point_B := translation_up point_A 1

-- The theorem to prove that the coordinates of B are (1, -1)
theorem coord_of_point_B : point_B = (1, -1) :=
by
  -- Placeholder for proof
  sorry

end coord_of_point_B_l1030_103055


namespace brick_length_is_20_cm_l1030_103090

theorem brick_length_is_20_cm
    (courtyard_length_m : ℕ) (courtyard_width_m : ℕ)
    (brick_length_cm : ℕ) (brick_width_cm : ℕ)
    (total_bricks_required : ℕ)
    (h1 : courtyard_length_m = 25)
    (h2 : courtyard_width_m = 16)
    (h3 : brick_length_cm = 20)
    (h4 : brick_width_cm = 10)
    (h5 : total_bricks_required = 20000) :
    brick_length_cm = 20 := 
by
    sorry

end brick_length_is_20_cm_l1030_103090


namespace probability_interval_l1030_103025

/-- 
The probability of event A occurring is 4/5, the probability of event B occurring is 3/4,
and the probability of event C occurring is 2/3. The smallest interval necessarily containing
the probability q that all three events occur is [0, 2/3].
-/
theorem probability_interval (P_A P_B P_C q : ℝ)
  (hA : P_A = 4 / 5) (hB : P_B = 3 / 4) (hC : P_C = 2 / 3)
  (h_q_le_A : q ≤ P_A) (h_q_le_B : q ≤ P_B) (h_q_le_C : q ≤ P_C) :
  0 ≤ q ∧ q ≤ 2 / 3 := by
  sorry

end probability_interval_l1030_103025


namespace proof_solution_arithmetic_progression_l1030_103086

noncomputable def system_has_solution (a b c m : ℝ) : Prop :=
  (m = 1 → a = b ∧ b = c) ∧
  (m = -2 → a + b + c = 0) ∧ 
  (m ≠ -2 ∧ m ≠ 1 → ∃ x y z : ℝ, x + y + m * z = a ∧ x + m * y + z = b ∧ m * x + y + z = c)

def abc_arithmetic_progression (a b c : ℝ) : Prop :=
  2 * b = a + c

theorem proof_solution_arithmetic_progression (a b c m : ℝ) : 
  system_has_solution a b c m → 
  (∃ x y z : ℝ, x + y + m * z = a ∧ x + m * y + z = b ∧ m * x + y + z = c ∧ 2 * y = x + z) ↔
  abc_arithmetic_progression a b c := 
by 
  sorry

end proof_solution_arithmetic_progression_l1030_103086


namespace count_even_numbers_between_250_and_600_l1030_103081

theorem count_even_numbers_between_250_and_600 : 
  ∃ n : ℕ, (n = 175 ∧ 
    ∀ k : ℕ, (250 < 2 * k ∧ 2 * k ≤ 600) ↔ (126 ≤ k ∧ k ≤ 300)) :=
by
  sorry

end count_even_numbers_between_250_and_600_l1030_103081


namespace loisa_saves_70_l1030_103032

def tablet_cash_price : ℕ := 450
def down_payment : ℕ := 100
def first_4_months_payment : ℕ := 40
def next_4_months_payment : ℕ := 35
def last_4_months_payment : ℕ := 30
def total_installment_payment : ℕ := down_payment + (4 * first_4_months_payment) + (4 * next_4_months_payment) + (4 * last_4_months_payment)
def savings : ℕ := total_installment_payment - tablet_cash_price

theorem loisa_saves_70 : savings = 70 := by
  sorry

end loisa_saves_70_l1030_103032


namespace percent_profit_l1030_103048

theorem percent_profit (cost : ℝ) (markup_percent : ℝ) (discount_percent : ℝ) (final_profit_percent : ℝ)
  (h1 : cost = 50)
  (h2 : markup_percent = 30)
  (h3 : discount_percent = 10)
  (h4 : final_profit_percent = 17)
  : (markup_percent / 100 * cost - discount_percent / 100 * (cost + markup_percent / 100 * cost)) / cost * 100 = final_profit_percent := 
by
  sorry

end percent_profit_l1030_103048


namespace runner_speed_ratio_l1030_103083

noncomputable def speed_ratio (u1 u2 : ℝ) : ℝ := u1 / u2

theorem runner_speed_ratio (u1 u2 : ℝ) (h1 : u1 > u2) (h2 : u1 + u2 = 5) (h3 : u1 - u2 = 5/3) :
  speed_ratio u1 u2 = 2 :=
by
  sorry

end runner_speed_ratio_l1030_103083


namespace volume_of_rectangular_prism_l1030_103049

theorem volume_of_rectangular_prism (x y z : ℝ) 
  (h1 : x * y = 30) 
  (h2 : x * z = 45) 
  (h3 : y * z = 75) : 
  x * y * z = 150 :=
sorry

end volume_of_rectangular_prism_l1030_103049


namespace fraction_meaningful_l1030_103053

theorem fraction_meaningful (x : ℝ) : (x ≠ 5) ↔ (x-5 ≠ 0) :=
by simp [sub_eq_zero]

end fraction_meaningful_l1030_103053


namespace yearly_water_consumption_correct_l1030_103091

def monthly_water_consumption : ℝ := 182.88
def months_in_a_year : ℕ := 12
def yearly_water_consumption : ℝ := monthly_water_consumption * (months_in_a_year : ℝ)

theorem yearly_water_consumption_correct :
  yearly_water_consumption = 2194.56 :=
by
  sorry

end yearly_water_consumption_correct_l1030_103091


namespace relationship_f_minus_a2_f_minus_1_l1030_103043

noncomputable def f (x : ℝ) : ℝ := x^3 - x^2 - x

-- Theorem statement translation
theorem relationship_f_minus_a2_f_minus_1 (a : ℝ) : f (-a^2) ≤ f (-1) := 
sorry

end relationship_f_minus_a2_f_minus_1_l1030_103043


namespace intersection_of_lines_l1030_103026

theorem intersection_of_lines : ∃ x y : ℚ, y = 3 * x ∧ y - 5 = -7 * x ∧ x = 1 / 2 ∧ y = 3 / 2 :=
by
  sorry

end intersection_of_lines_l1030_103026


namespace determine_a_b_l1030_103018

-- Definitions
def num (a b : ℕ) := 10000*a + 1000*6 + 100*7 + 10*9 + b

def divisible_by_72 (n : ℕ) : Prop := n % 72 = 0

noncomputable def a : ℕ := 3
noncomputable def b : ℕ := 2

-- Theorem statement
theorem determine_a_b : divisible_by_72 (num a b) :=
by
  -- The proof will be inserted here
  sorry

end determine_a_b_l1030_103018


namespace complex_expr_evaluation_l1030_103057

def complex_expr : ℤ :=
  2 * (3 * (2 * (3 * (2 * (3 * (2 + 1) * 2) + 2) * 2) + 2) * 2) + 2

theorem complex_expr_evaluation : complex_expr = 5498 := by
  sorry

end complex_expr_evaluation_l1030_103057


namespace charlie_third_week_data_l1030_103037

theorem charlie_third_week_data (d3 : ℕ) : 
  let data_plan := 8
  let cost_per_GB := 10
  let extra_charge := 120
  let week1 := 2
  let week2 := 3
  let week4 := 10
  let total_extra_GB := extra_charge / cost_per_GB
  let total_data := week1 + week2 + week4 + d3
  let overage_GB := total_data - data_plan
  overage_GB = total_extra_GB -> d3 = 5 := 
by
  let data_plan := 8
  let cost_per_GB := 10
  let extra_charge := 120
  let week1 := 2
  let week2 := 3
  let week4 := 10
  let total_extra_GB := extra_charge / cost_per_GB
  let total_data := week1 + week2 + week4 + d3
  let overage_GB := total_data - data_plan
  have : overage_GB = total_extra_GB := sorry
  have : d3 = 5 := sorry
  sorry

end charlie_third_week_data_l1030_103037


namespace percentage_customers_return_books_l1030_103011

theorem percentage_customers_return_books 
  (total_customers : ℕ) (price_per_book : ℕ) (sales_after_returns : ℕ) 
  (h1 : total_customers = 1000) 
  (h2 : price_per_book = 15) 
  (h3 : sales_after_returns = 9450) : 
  ((total_customers - (sales_after_returns / price_per_book)) / total_customers) * 100 = 37 := 
by
  sorry

end percentage_customers_return_books_l1030_103011


namespace each_child_plays_40_minutes_l1030_103098

variable (TotalMinutes : ℕ)
variable (NumChildren : ℕ)
variable (ChildPairs : ℕ)

theorem each_child_plays_40_minutes (h1 : TotalMinutes = 120) 
                                    (h2 : NumChildren = 6) 
                                    (h3 : ChildPairs = 2) :
  (ChildPairs * TotalMinutes) / NumChildren = 40 :=
by
  sorry

end each_child_plays_40_minutes_l1030_103098


namespace coral_must_read_pages_to_finish_book_l1030_103021

theorem coral_must_read_pages_to_finish_book
  (total_pages first_week_read second_week_percentage pages_remaining first_week_left second_week_read : ℕ)
  (initial_pages_read : ℕ := total_pages / 2)
  (remaining_after_first_week : ℕ := total_pages - initial_pages_read)
  (read_second_week : ℕ := remaining_after_first_week * second_week_percentage / 100)
  (remaining_after_second_week : ℕ := remaining_after_first_week - read_second_week)
  (final_pages_to_read : ℕ := remaining_after_second_week):
  total_pages = 600 → first_week_read = 300 → second_week_percentage = 30 →
  pages_remaining = 300 → first_week_left = 300 → second_week_read = 90 →
  remaining_after_first_week = 300 - 300 →
  remaining_after_second_week = remaining_after_first_week - second_week_read →
  third_week_read = remaining_after_second_week →
  third_week_read = 210 := by
  sorry

end coral_must_read_pages_to_finish_book_l1030_103021


namespace whitewashing_cost_l1030_103014

noncomputable def cost_of_whitewashing (l w h : ℝ) (c : ℝ) (door_area window_area : ℝ) (num_windows : ℝ) : ℝ :=
  let perimeter := 2 * (l + w)
  let total_wall_area := perimeter * h
  let total_window_area := num_windows * window_area
  let total_paintable_area := total_wall_area - (door_area + total_window_area)
  total_paintable_area * c

theorem whitewashing_cost:
  cost_of_whitewashing 25 15 12 6 (6 * 3) (4 * 3) 3 = 5436 := by
  sorry

end whitewashing_cost_l1030_103014


namespace graph_of_f_4_minus_x_l1030_103070

theorem graph_of_f_4_minus_x (f : ℝ → ℝ) (h : f 0 = 1) : f (4 - 4) = 1 :=
by
  rw [sub_self]
  exact h

end graph_of_f_4_minus_x_l1030_103070


namespace find_sum_of_coordinates_of_other_endpoint_l1030_103075

theorem find_sum_of_coordinates_of_other_endpoint :
  ∃ (x y : ℤ), (7, -5) = (10 + x / 2, 4 + y / 2) ∧ x + y = -10 :=
by
  sorry

end find_sum_of_coordinates_of_other_endpoint_l1030_103075


namespace alphametic_puzzle_l1030_103077

theorem alphametic_puzzle (I D A M E R O : ℕ) 
  (h1 : R = 0) 
  (h2 : D + E = 10)
  (h3 : I + M + 1 = O)
  (h4 : A = D + 1) :
  I + 1 + M + 10 + 1 = O + 0 + A := sorry

end alphametic_puzzle_l1030_103077


namespace find_lighter_ball_min_weighings_l1030_103034

noncomputable def min_weighings_to_find_lighter_ball (balls : Fin 9 → ℕ) : ℕ :=
  2

-- Given: 9 balls, where 8 weigh 10 grams and 1 weighs 9 grams, and a balance scale.
theorem find_lighter_ball_min_weighings :
  (∃ i : Fin 9, balls i = 9 ∧ (∀ j : Fin 9, j ≠ i → balls j = 10)) 
  → min_weighings_to_find_lighter_ball balls = 2 :=
by
  intros
  sorry

end find_lighter_ball_min_weighings_l1030_103034


namespace probability_of_condition_l1030_103013

def Q_within_square (x y : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1

def condition (x y : ℝ) : Prop :=
  y > (1/2) * x

theorem probability_of_condition : 
  ∀ x y, Q_within_square x y → (0.75 = 3 / 4) :=
by
  sorry

end probability_of_condition_l1030_103013


namespace rowing_students_l1030_103089

theorem rowing_students (X Y : ℕ) (N : ℕ) :
  (17 * X + 6 = N) →
  (10 * Y + 2 = N) →
  100 < N →
  N < 200 →
  5 ≤ X ∧ X ≤ 11 →
  10 ≤ Y ∧ Y ≤ 19 →
  N = 142 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end rowing_students_l1030_103089


namespace rectangle_area_diagonal_l1030_103093

theorem rectangle_area_diagonal (r l w d : ℝ) (h_ratio : r = 5 / 2) (h_diag : d^2 = l^2 + w^2) : ∃ k : ℝ, (k = 10 / 29) ∧ (l / w = r) ∧ (l^2 + w^2 = d^2) :=
by
  sorry

end rectangle_area_diagonal_l1030_103093


namespace eight_points_in_circle_l1030_103004

theorem eight_points_in_circle :
  ∀ (P : Fin 8 → ℝ × ℝ), 
  (∀ i, (P i).1^2 + (P i).2^2 ≤ 1) → 
  ∃ (i j : Fin 8), i ≠ j ∧ ((P i).1 - (P j).1)^2 + ((P i).2 - (P j).2)^2 < 1 :=
by
  sorry

end eight_points_in_circle_l1030_103004


namespace largest_possible_b_l1030_103072

theorem largest_possible_b (a b c : ℕ) (h1 : 1 < c) (h2 : c < b) (h3 : b < a) (h4 : a * b * c = 360) : b ≤ 12 :=
by
  sorry

end largest_possible_b_l1030_103072


namespace polynomial_factorization_example_l1030_103062

open Polynomial

theorem polynomial_factorization_example
  (a_5 a_4 a_3 a_2 a_1 a_0 : ℤ) (hf : ∀ i ∈ [a_5, a_4, a_3, a_2, a_1, a_0], |i| ≤ 4)
  (b_3 b_2 b_1 b_0 : ℤ) (hg : ∀ i ∈ [b_3, b_2, b_1, b_0], |i| ≤ 1)
  (c_2 c_1 c_0 : ℤ) (hh : ∀ i ∈ [c_2, c_1, c_0], |i| ≤ 1)
  (h : (C a_5 * X^5 + C a_4 * X^4 + C a_3 * X^3 + C a_2 * X^2 + C a_1 * X + C a_0).eval 10 =
       ((C b_3 * X^3 + C b_2 * X^2 + C b_1 * X + C b_0) * (C c_2 * X^2 + C c_1 * X + C c_0)).eval 10) :
  (C a_5 * X^5 + C a_4 * X^4 + C a_3 * X^3 + C a_2 * X^2 + C a_1 * X + C a_0) =
  (C b_3 * X^3 + C b_2 * X^2 + C b_1 * X + C b_0) * (C c_2 * X^2 + C c_1 * X + C c_0) :=
sorry

end polynomial_factorization_example_l1030_103062


namespace connie_start_marbles_l1030_103022

variable (marbles_total marbles_given marbles_left : ℕ)

theorem connie_start_marbles :
  marbles_given = 73 → marbles_left = 70 → marbles_total = marbles_given + marbles_left → marbles_total = 143 :=
by intros; sorry

end connie_start_marbles_l1030_103022


namespace truck_tank_percentage_increase_l1030_103017

-- Declaration of the initial conditions (as given in the problem)
def service_cost : ℝ := 2.20
def fuel_cost_per_liter : ℝ := 0.70
def num_minivans : ℕ := 4
def num_trucks : ℕ := 2
def total_cost : ℝ := 395.40
def minivan_tank_size : ℝ := 65.0

-- Proof statement with the conditions declared above
theorem truck_tank_percentage_increase :
  ∃ p : ℝ, p = 120 ∧ (minivan_tank_size * (p + 100) / 100 = 143) :=
sorry

end truck_tank_percentage_increase_l1030_103017


namespace x_y_difference_l1030_103006

theorem x_y_difference
    (x y : ℚ)
    (h1 : x + y = 780)
    (h2 : x / y = 1.25) :
    x - y = 86.66666666666667 :=
by
  sorry

end x_y_difference_l1030_103006


namespace total_seats_l1030_103050

theorem total_seats (s : ℕ) 
  (h1 : 30 + (0.20 * s : ℝ) + (0.60 * s : ℝ) = s) : s = 150 :=
  sorry

end total_seats_l1030_103050


namespace find_a_l1030_103016

theorem find_a 
  (a b c : ℚ) 
  (h1 : b = 4 * a) 
  (h2 : b = 15 - 4 * a - c) 
  (h3 : c = a + 2) : 
  a = 13 / 9 := 
by 
  sorry

end find_a_l1030_103016


namespace problem_statement_l1030_103064

open Complex

noncomputable def a : ℂ := 5 - 3 * I
noncomputable def b : ℂ := 2 + 4 * I

theorem problem_statement : 3 * a - 4 * b = 7 - 25 * I :=
by { sorry }

end problem_statement_l1030_103064


namespace union_sets_l1030_103078

def A := { x : ℝ | x^2 ≤ 1 }
def B := { x : ℝ | 0 < x }

theorem union_sets : A ∪ B = { x | -1 ≤ x } :=
by {
  sorry -- Proof is omitted as per the instructions
}

end union_sets_l1030_103078


namespace diameter_of_outer_edge_l1030_103012

-- Defining the conditions as variables
variable (pathWidth gardenWidth statueDiameter fountainDiameter : ℝ)
variable (hPathWidth : pathWidth = 10)
variable (hGardenWidth : gardenWidth = 12)
variable (hStatueDiameter : statueDiameter = 6)
variable (hFountainDiameter : fountainDiameter = 14)

-- Lean statement to prove the diameter
theorem diameter_of_outer_edge :
  2 * ((fountainDiameter / 2) + gardenWidth + pathWidth) = 58 :=
by
  rw [hPathWidth, hGardenWidth, hFountainDiameter]
  sorry

end diameter_of_outer_edge_l1030_103012


namespace olivia_total_payment_l1030_103001

theorem olivia_total_payment : 
  (4 / 4 + 12 / 4 = 4) :=
by
  sorry

end olivia_total_payment_l1030_103001


namespace both_selected_prob_l1030_103024

noncomputable def prob_X : ℚ := 1 / 3
noncomputable def prob_Y : ℚ := 2 / 7
noncomputable def combined_prob : ℚ := prob_X * prob_Y

theorem both_selected_prob :
  combined_prob = 2 / 21 :=
by
  unfold combined_prob prob_X prob_Y
  sorry

end both_selected_prob_l1030_103024


namespace total_price_of_houses_l1030_103039

theorem total_price_of_houses (price_first price_second total_price : ℝ)
    (h1 : price_first = 200000)
    (h2 : price_second = 2 * price_first)
    (h3 : total_price = price_first + price_second) :
  total_price = 600000 := by
  sorry

end total_price_of_houses_l1030_103039


namespace division_of_8_identical_books_into_3_piles_l1030_103040

-- Definitions for the conditions
def identical_books_division_ways (n : ℕ) (p : ℕ) : ℕ :=
  if n = 8 ∧ p = 3 then 5 else sorry

-- Theorem statement
theorem division_of_8_identical_books_into_3_piles :
  identical_books_division_ways 8 3 = 5 := by
  sorry

end division_of_8_identical_books_into_3_piles_l1030_103040


namespace max_quotient_l1030_103096

-- Define the given conditions
def conditions (a b : ℝ) :=
  100 ≤ a ∧ a ≤ 250 ∧ 700 ≤ b ∧ b ≤ 1400

-- State the theorem for the largest value of the quotient b / a
theorem max_quotient (a b : ℝ) (h : conditions a b) : b / a ≤ 14 :=
by
  sorry

end max_quotient_l1030_103096


namespace solve_system_of_inequalities_l1030_103002

open Set

theorem solve_system_of_inequalities : ∀ x : ℕ, (2 * (x - 1) < x + 3) ∧ ((2 * x + 1) / 3 > x - 1) → x ∈ ({0, 1, 2, 3} : Set ℕ) :=
by
  intro x
  intro h
  sorry

end solve_system_of_inequalities_l1030_103002


namespace fraction_area_outside_circle_l1030_103080

theorem fraction_area_outside_circle (r : ℝ) (h1 : r > 0) :
  let side_length := 2 * r
  let area_square := side_length ^ 2
  let area_circle := π * r ^ 2
  let area_outside := area_square - area_circle
  (area_outside / area_square) = 1 - ↑π / 4 :=
by
  sorry

end fraction_area_outside_circle_l1030_103080


namespace problem1_problem2_problem3_l1030_103073

theorem problem1 : -3^2 + (-1/2)^2 + (2023 - Real.pi)^0 - |-2| = -47/4 :=
by
  sorry

theorem problem2 (a : ℝ) : (-2 * a^2)^3 * a^2 + a^8 = -7 * a^8 :=
by
  sorry

theorem problem3 : 2023^2 - 2024 * 2022 = 1 :=
by
  sorry

end problem1_problem2_problem3_l1030_103073


namespace min_omega_l1030_103051

theorem min_omega (f : Real → Real) (ω φ : Real) (φ_bound : |φ| < π / 2) 
  (h1 : ω > 0) (h2 : f = fun x => Real.sin (ω * x + φ)) 
  (h3 : f 0 = 1/2) 
  (h4 : ∀ x, f x ≤ f (π / 12)) : ω = 4 := 
by
  sorry

end min_omega_l1030_103051


namespace sum_of_a5_a6_l1030_103099

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n * q

noncomputable def geometric_conditions (a : ℕ → ℝ) (q : ℝ) : Prop :=
a 1 + a 2 = 1 ∧ a 3 + a 4 = 4 ∧ q^2 = 4

theorem sum_of_a5_a6 (a : ℕ → ℝ) (q : ℝ) (h_seq : geometric_sequence a q) (h_cond : geometric_conditions a q) :
  a 5 + a 6 = 16 :=
sorry

end sum_of_a5_a6_l1030_103099


namespace ratio_of_carpets_l1030_103058

theorem ratio_of_carpets (h1 h2 h3 h4 : ℕ) (total : ℕ) 
  (H1 : h1 = 12) (H2 : h2 = 20) (H3 : h3 = 10) (H_total : total = 62) 
  (H_all_houses : h1 + h2 + h3 + h4 = total) : h4 / h3 = 2 :=
by
  sorry

end ratio_of_carpets_l1030_103058


namespace min_value_of_a_plus_2b_l1030_103061

theorem min_value_of_a_plus_2b (a b : ℝ) (h1: a > 0) (h2: b > 0) (h3: 1 / (a + 1) + 1 / (b + 1) = 1) : 
  a + 2 * b ≥ 2 * Real.sqrt 2 :=
by
  sorry

end min_value_of_a_plus_2b_l1030_103061


namespace find_second_number_l1030_103020

def sum_of_three (a b c : ℚ) : Prop :=
  a + b + c = 120

def ratio_first_to_second (a b : ℚ) : Prop :=
  a / b = 3 / 4

def ratio_second_to_third (b c : ℚ) : Prop :=
  b / c = 3 / 5

theorem find_second_number (a b c : ℚ) 
  (h_sum : sum_of_three a b c)
  (h_ratio_ab : ratio_first_to_second a b)
  (h_ratio_bc : ratio_second_to_third b c) : 
  b = 1440 / 41 := 
sorry

end find_second_number_l1030_103020


namespace sum_of_digits_1_to_1000_l1030_103071

/--  sum_of_digits calculates the sum of digits of a given number n -/
def sum_of_digits(n : ℕ) : ℕ :=
  n.digits 10 |>.sum

/-- sum_of_digits_in_range calculates the sum of the digits 
of all numbers in the inclusive range from 1 to m -/
def sum_of_digits_in_range (m : ℕ) : ℕ :=
  (Finset.range (m + 1)).sum sum_of_digits

theorem sum_of_digits_1_to_1000 : sum_of_digits_in_range 1000 = 13501 :=
by
  sorry

end sum_of_digits_1_to_1000_l1030_103071


namespace amount_after_two_years_l1030_103059

def amount_after_years (P : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  P * ((r + 1) ^ n) / (r ^ n)

theorem amount_after_two_years :
  let P : ℕ := 70400
  let r : ℕ := 8
  amount_after_years P r 2 = 89070 :=
  by
    sorry

end amount_after_two_years_l1030_103059


namespace boys_speed_l1030_103097

-- Define the conditions
def sideLength : ℕ := 50
def timeTaken : ℕ := 72

-- Define the goal
theorem boys_speed (sideLength timeTaken : ℕ) (D T : ℝ) :
  D = (4 * sideLength : ℕ) / 1000 ∧
  T = timeTaken / 3600 →
  (D / T = 10) := by
  sorry

end boys_speed_l1030_103097


namespace total_property_price_l1030_103044

theorem total_property_price :
  let price_per_sqft : ℝ := 98
  let house_sqft : ℝ := 2400
  let barn_sqft : ℝ := 1000
  let house_price : ℝ := house_sqft * price_per_sqft
  let barn_price : ℝ := barn_sqft * price_per_sqft
  let total_price : ℝ := house_price + barn_price
  total_price = 333200 := by
  sorry

end total_property_price_l1030_103044


namespace symmetry_implies_value_l1030_103003

noncomputable def f (x : ℝ) (ω : ℝ) : ℝ := 2 * Real.sin (ω * x - Real.pi / 3)
noncomputable def g (x : ℝ) (φ : ℝ) : ℝ := Real.cos (2 * x + φ)

theorem symmetry_implies_value :
  (∀ (x : ℝ), ∃ (k : ℤ), ω * x - Real.pi / 3 = k * Real.pi + Real.pi / 2) →
  (∀ (x : ℝ), ∃ (k : ℤ), 2 * x + φ = k * Real.pi) →
  0 < φ → φ < Real.pi →
  ω = 2 →
  φ = Real.pi / 6 →
  g (Real.pi / 3) φ = -Real.sqrt 3 / 2 :=
by
  intros h1 h2 h3 h4 h5 h6
  exact sorry

end symmetry_implies_value_l1030_103003


namespace vacant_student_seats_given_to_parents_l1030_103052

-- Definitions of the conditions
def total_seats : Nat := 150

def awardees_seats : Nat := 15
def admins_teachers_seats : Nat := 45
def students_seats : Nat := 60
def parents_seats : Nat := 30

def awardees_occupied_seats : Nat := 15
def admins_teachers_occupied_seats : Nat := 9 * admins_teachers_seats / 10
def students_occupied_seats : Nat := 4 * students_seats / 5
def parents_occupied_seats : Nat := 7 * parents_seats / 10

-- Vacant seats calculation
def awardees_vacant_seats : Nat := awardees_seats - awardees_occupied_seats
def admins_teachers_vacant_seats : Nat := admins_teachers_seats - admins_teachers_occupied_seats
def students_vacant_seats : Nat := students_seats - students_occupied_seats
def parents_vacant_seats : Nat := parents_seats - parents_occupied_seats

-- Theorem statement
theorem vacant_student_seats_given_to_parents :
  students_vacant_seats = 12 →
  parents_vacant_seats = 9 →
  9 ≤ students_vacant_seats ∧ 9 ≤ parents_vacant_seats :=
by
  sorry

end vacant_student_seats_given_to_parents_l1030_103052


namespace incorrect_statement_l1030_103027

-- Definition of the universal set U
def U : Set ℕ := {1, 2, 3, 4}

-- Definition of set M
def M : Set ℕ := {1, 2}

-- Definition of set N
def N : Set ℕ := {2, 4}

-- Complement of set in a universal set
def complement (S : Set ℕ) : Set ℕ := U \ S

-- Statement that D is incorrect
theorem incorrect_statement :
  M ∩ complement N ≠ {1, 2, 3} :=
by
  sorry

end incorrect_statement_l1030_103027


namespace find_missing_number_l1030_103082

theorem find_missing_number :
  ∀ (x y : ℝ),
    (12 + x + 42 + 78 + 104) / 5 = 62 →
    (128 + y + 511 + 1023 + x) / 5 = 398.2 →
    y = 255 :=
by
  intros x y h1 h2
  sorry

end find_missing_number_l1030_103082


namespace base_of_first_term_is_two_l1030_103007

-- Define h as a positive integer
variable (h : ℕ) (a b c : ℕ)

-- Conditions
variables 
  (h_positive : h > 0)
  (divisor_225 : 225 ∣ h)
  (divisor_216 : 216 ∣ h)

-- Given h can be expressed as specified and a + b + c = 8
variable (h_expression : ∃ k : ℕ, h = k^a * 3^b * 5^c)
variable (sum_eight : a + b + c = 8)

-- Prove the base of the first term in the expression for h is 2.
theorem base_of_first_term_is_two : (∃ k : ℕ, k^a * 3^b * 5^c = h) → k = 2 :=
by 
  sorry

end base_of_first_term_is_two_l1030_103007


namespace domain_all_real_l1030_103095

theorem domain_all_real (p : ℝ) : 
  (∀ x : ℝ, -3 * x ^ 2 + 3 * x + p ≠ 0) ↔ p < -3 / 4 := 
by
  sorry

end domain_all_real_l1030_103095


namespace greatest_consecutive_sum_l1030_103038

theorem greatest_consecutive_sum (S : ℤ) (hS : S = 105) : 
  ∃ N : ℤ, (∃ a : ℤ, (N * (2 * a + N - 1) = 2 * S)) ∧ 
  (∀ M : ℤ, (∃ b : ℤ, (M * (2 * b + M - 1) = 2 * S)) → M ≤ N) ∧ N = 210 := 
sorry

end greatest_consecutive_sum_l1030_103038


namespace hyperbola_foci_on_x_axis_l1030_103060

theorem hyperbola_foci_on_x_axis (a : ℝ) 
  (h1 : 1 - a < 0)
  (h2 : a - 3 > 0)
  (h3 : ∀ c, c = 2 → 2 * c = 4) : 
  a = 4 := 
sorry

end hyperbola_foci_on_x_axis_l1030_103060


namespace integral_transform_eq_l1030_103041

open MeasureTheory

variable (f : ℝ → ℝ)

theorem integral_transform_eq (hf_cont : Continuous f) (hL_exists : ∃ L, ∫ x in (Set.univ : Set ℝ), f x = L) :
  ∃ L, ∫ x in (Set.univ : Set ℝ), f (x - 1/x) = L :=
by
  cases' hL_exists with L hL
  use L
  have h_transform : ∫ x in (Set.univ : Set ℝ), f (x - 1/x) = ∫ x in (Set.univ : Set ℝ), f x := sorry
  rw [h_transform]
  exact hL

end integral_transform_eq_l1030_103041


namespace salary_increase_l1030_103023

theorem salary_increase (prev_income : ℝ) (prev_percentage : ℝ) (new_percentage : ℝ) (rent_utilities : ℝ) (new_income : ℝ) :
  prev_income = 1000 ∧ prev_percentage = 0.40 ∧ new_percentage = 0.25 ∧ rent_utilities = prev_percentage * prev_income ∧
  rent_utilities = new_percentage * new_income → new_income - prev_income = 600 :=
by 
  sorry

end salary_increase_l1030_103023


namespace hall_reunion_attendees_l1030_103069

theorem hall_reunion_attendees
  (total_guests : ℕ)
  (oates_attendees : ℕ)
  (both_attendees : ℕ)
  (h : total_guests = 100 ∧ oates_attendees = 50 ∧ both_attendees = 12) :
  ∃ (hall_attendees : ℕ), hall_attendees = 62 :=
by
  sorry

end hall_reunion_attendees_l1030_103069


namespace sum_of_midpoint_coordinates_l1030_103067

theorem sum_of_midpoint_coordinates :
  let (x1, y1) := (8, 16)
  let (x2, y2) := (2, -8)
  let midpoint_x := (x1 + x2) / 2
  let midpoint_y := (y1 + y2) / 2
  midpoint_x + midpoint_y = 9 :=
by
  sorry

end sum_of_midpoint_coordinates_l1030_103067


namespace total_students_in_class_l1030_103088

theorem total_students_in_class (R S : ℕ) (h1 : 2 + 12 + 14 + R = S) (h2 : 2 * S = 40 + 3 * R) : S = 44 :=
by
  sorry

end total_students_in_class_l1030_103088


namespace sum_of_squares_fraction_l1030_103042

variable {x1 x2 x3 y1 y2 y3 : ℝ}

theorem sum_of_squares_fraction :
  x1 + x2 + x3 = 0 → y1 + y2 + y3 = 0 → x1 * y1 + x2 * y2 + x3 * y3 = 0 →
  (x1^2 / (x1^2 + x2^2 + x3^2)) + (y1^2 / (y1^2 + y2^2 + y3^2)) = 2 / 3 :=
by
  intros h1 h2 h3
  sorry

end sum_of_squares_fraction_l1030_103042


namespace complement_U_A_intersection_A_B_complement_U_intersection_A_B_complement_A_intersection_B_l1030_103054

open Set -- Open the Set namespace for convenience

-- Define the universal set U, and sets A and B
def U : Set ℝ := univ
def A : Set ℝ := {x | -2 < x ∧ x < 3}
def B : Set ℝ := {x | -3 < x ∧ x ≤ 3}

-- Proof statements
theorem complement_U_A : U \ A = {x | x ≥ 3 ∨ x ≤ -2} :=
by sorry

theorem intersection_A_B : A ∩ B = {x | -2 < x ∧ x < 3} :=
by sorry

theorem complement_U_intersection_A_B : U \ (A ∩ B) = {x | x ≥ 3 ∨ x ≤ -2} :=
by sorry

theorem complement_A_intersection_B : (U \ A) ∩ B = {x | (-3 < x ∧ x ≤ -2) ∨ x = 3} :=
by sorry

end complement_U_A_intersection_A_B_complement_U_intersection_A_B_complement_A_intersection_B_l1030_103054


namespace fraction_equation_solution_l1030_103087

theorem fraction_equation_solution (x : ℝ) (h₁ : x ≠ 3) (h₂ : x ≠ 4) :
  (3 / (x - 3) = 4 / (x - 4)) → x = 0 :=
by
  sorry

end fraction_equation_solution_l1030_103087


namespace animals_not_like_either_l1030_103029

def total_animals : ℕ := 75
def animals_eat_carrots : ℕ := 26
def animals_like_hay : ℕ := 56
def animals_like_both : ℕ := 14

theorem animals_not_like_either : (total_animals - (animals_eat_carrots - animals_like_both + animals_like_hay - animals_like_both + animals_like_both)) = 7 := by
  sorry

end animals_not_like_either_l1030_103029


namespace niki_money_l1030_103008

variables (N A : ℕ)

def condition1 (N A : ℕ) : Prop := N = 2 * A + 15
def condition2 (N A : ℕ) : Prop := N - 30 = (A + 30) / 2

theorem niki_money : condition1 N A ∧ condition2 N A → N = 55 :=
by
  sorry

end niki_money_l1030_103008


namespace adam_money_given_l1030_103000

theorem adam_money_given (original_money : ℕ) (final_money : ℕ) (money_given : ℕ) :
  original_money = 79 →
  final_money = 92 →
  money_given = final_money - original_money →
  money_given = 13 := by
sorry

end adam_money_given_l1030_103000


namespace zoe_pop_albums_l1030_103084

theorem zoe_pop_albums (total_songs country_albums songs_per_album : ℕ) (h1 : total_songs = 24) (h2 : country_albums = 3) (h3 : songs_per_album = 3) :
  total_songs - (country_albums * songs_per_album) = 15 ↔ (total_songs - (country_albums * songs_per_album)) / songs_per_album = 5 :=
by
  sorry

end zoe_pop_albums_l1030_103084


namespace average_income_P_Q_l1030_103005

   variable (P Q R : ℝ)

   theorem average_income_P_Q
     (h1 : (Q + R) / 2 = 6250)
     (h2 : (P + R) / 2 = 5200)
     (h3 : P = 4000) :
     (P + Q) / 2 = 5050 := by
   sorry
   
end average_income_P_Q_l1030_103005


namespace trapezoid_inequality_l1030_103019

theorem trapezoid_inequality (a b R : ℝ) (h : a > 0) (h1 : b > 0) (h2 : R > 0) 
  (circumscribed : ∃ (x y : ℝ), x + y = a ∧ R^2 * (1/x + 1/y) = b) : 
  a * b ≥ 4 * R^2 :=
by
  sorry

end trapezoid_inequality_l1030_103019


namespace smallest_root_of_quadratic_l1030_103079

theorem smallest_root_of_quadratic :
  ∃ x : ℝ, (12 * x^2 - 50 * x + 48 = 0) ∧ x = 1.333 := 
sorry

end smallest_root_of_quadratic_l1030_103079


namespace tensor_op_correct_l1030_103066

-- Define the operation ⊗
def tensor_op (x y : ℝ) : ℝ := x^2 + y

-- Goal: Prove h ⊗ (h ⊗ h) = 2h^2 + h for some h in ℝ
theorem tensor_op_correct (h : ℝ) : tensor_op h (tensor_op h h) = 2 * h^2 + h :=
by
  sorry

end tensor_op_correct_l1030_103066


namespace current_balance_after_deduction_l1030_103035

theorem current_balance_after_deduction :
  ∀ (original_balance deduction_percent : ℕ), 
  original_balance = 100000 →
  deduction_percent = 10 →
  original_balance - (deduction_percent * original_balance / 100) = 90000 :=
by
  intros original_balance deduction_percent h1 h2
  sorry

end current_balance_after_deduction_l1030_103035


namespace dad_steps_l1030_103033

theorem dad_steps (masha_steps : ℕ) (yasha_steps : ℕ) (dad_steps : ℕ) :
  (∀ d m, m = 5 * d / 3) → (∀ m y, y = 5 * m / 3) → (masha_steps + yasha_steps = 400) → 
  dad_steps = 90 :=
by
  sorry

end dad_steps_l1030_103033


namespace cube_pyramid_volume_l1030_103076

theorem cube_pyramid_volume (s b h : ℝ) 
  (hcube : s = 6) 
  (hbase : b = 10)
  (eq_volumes : (s ^ 3) = (1 / 3) * (b ^ 2) * h) : 
  h = 162 / 25 := 
by 
  sorry

end cube_pyramid_volume_l1030_103076


namespace rate_percent_simple_interest_l1030_103028

theorem rate_percent_simple_interest (P SI T : ℝ) (hP : P = 720) (hSI : SI = 180) (hT : T = 4) :
  (SI = P * (R / 100) * T) → R = 6.25 :=
by
  sorry

end rate_percent_simple_interest_l1030_103028


namespace apples_problem_l1030_103056

variable (K A : ℕ)

theorem apples_problem (K A : ℕ) (h1 : K + (3 / 4) * K + 600 = 2600) (h2 : A + (3 / 4) * A + 600 = 2600) :
  K = 1142 ∧ A = 1142 :=
by
  sorry

end apples_problem_l1030_103056


namespace find_difference_l1030_103046

theorem find_difference (x y : ℝ) (h1 : 3 = 0.15 * x) (h2 : 3 = 0.30 * y) : x - y = 10 := 
by
  sorry

end find_difference_l1030_103046


namespace museum_revenue_l1030_103036

theorem museum_revenue (V : ℕ) (H : V = 500)
  (R : ℕ) (H_R : R = 60 * V / 100)
  (C_p : ℕ) (H_C_p : C_p = 40 * R / 100)
  (S_p : ℕ) (H_S_p : S_p = 30 * R / 100)
  (A_p : ℕ) (H_A_p : A_p = 30 * R / 100)
  (C_t S_t A_t : ℕ) (H_C_t : C_t = 4) (H_S_t : S_t = 6) (H_A_t : A_t = 12) :
  C_p * C_t + S_p * S_t + A_p * A_t = 2100 :=
by 
  sorry

end museum_revenue_l1030_103036
