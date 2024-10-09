import Mathlib

namespace scooter_value_depreciation_l288_28838

theorem scooter_value_depreciation (V0 Vn : ℝ) (rate : ℝ) (n : ℕ) 
  (hV0 : V0 = 40000) 
  (hVn : Vn = 9492.1875) 
  (hRate : rate = 3 / 4) 
  (hValue : Vn = V0 * rate ^ n) : 
  n = 5 := 
by 
  -- Conditions are set up, proof needs to be constructed.
  sorry

end scooter_value_depreciation_l288_28838


namespace locus_of_P_l288_28866

theorem locus_of_P
  (a b x y : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : (x ≠ 0 ∧ y ≠ 0))
  (h4 : x^2 / a^2 - y^2 / b^2 = 1) :
  (x / a)^2 - (y / b)^2 = ((a^2 + b^2) / (a^2 - b^2))^2 := by
  sorry

end locus_of_P_l288_28866


namespace geometric_sequence_third_term_and_sum_l288_28843

noncomputable def geometric_sequence (b1 r : ℝ) (n : ℕ) : ℝ :=
  b1 * r^(n - 1)

theorem geometric_sequence_third_term_and_sum (b2 b5 : ℝ) (h1 : b2 = 24.5) (h2 : b5 = 196) :
  (∃ b1 r : ℝ, r ≠ 0 ∧ geometric_sequence b1 r 2 = b2 ∧ geometric_sequence b1 r 5 = b5 ∧
  geometric_sequence b1 r 3 = 49 ∧
  b1 * (r^4 - 1) / (r - 1) = 183.75) :=
by sorry

end geometric_sequence_third_term_and_sum_l288_28843


namespace no_integer_roots_of_quadratic_l288_28883

theorem no_integer_roots_of_quadratic (n : ℤ) : 
  ¬ ∃ (x : ℤ), x^2 - 16 * n * x + 7^5 = 0 := by
  sorry

end no_integer_roots_of_quadratic_l288_28883


namespace student_losses_one_mark_l288_28814

def number_of_marks_lost_per_wrong_answer (correct_ans marks_attempted total_questions total_marks correct_questions : ℤ) : ℤ :=
  (correct_ans * correct_questions - total_marks) / (total_questions - correct_questions)

theorem student_losses_one_mark
  (correct_ans : ℤ)
  (marks_attempted : ℤ)
  (total_questions : ℤ)
  (total_marks : ℤ)
  (correct_questions : ℤ)
  (total_wrong : ℤ):
  correct_ans = 4 →
  total_questions = 80 →
  total_marks = 120 →
  correct_questions = 40 →
  total_wrong = total_questions - correct_questions →
  number_of_marks_lost_per_wrong_answer correct_ans marks_attempted total_questions total_marks correct_questions = 1 :=
by
  sorry

end student_losses_one_mark_l288_28814


namespace find_a_value_l288_28824

theorem find_a_value (a : ℤ) (h : a^3 = 21 * 25 * 45 * 49) : a = 105 :=
by
  sorry

end find_a_value_l288_28824


namespace area_of_square_field_l288_28849

-- Define side length
def sideLength : ℕ := 14

-- Define the area function for a square
def area_of_square (side : ℕ) : ℕ := side * side

-- Prove that the area of the square with side length 14 meters is 196 square meters
theorem area_of_square_field : area_of_square sideLength = 196 := by
  sorry

end area_of_square_field_l288_28849


namespace mean_tasks_b_l288_28825

variable (a b : ℕ)
variable (m_a m_b : ℕ)
variable (h1 : a + b = 260)
variable (h2 : a = 3 * b / 10 + b)
variable (h3 : m_a = 80)
variable (h4 : m_b = 12 * m_a / 10)

theorem mean_tasks_b :
  m_b = 96 := by
  -- This is where the proof would go
  sorry

end mean_tasks_b_l288_28825


namespace james_profit_l288_28833

theorem james_profit
  (tickets_bought : ℕ)
  (cost_per_ticket : ℕ)
  (percentage_winning : ℕ)
  (winning_tickets_percentage_5dollars : ℕ)
  (grand_prize : ℕ)
  (average_other_prizes : ℕ)
  (total_tickets : ℕ)
  (total_cost : ℕ)
  (winning_tickets : ℕ)
  (tickets_prize_5dollars : ℕ)
  (amount_won_5dollars : ℕ)
  (other_winning_tickets : ℕ)
  (other_tickets_prize : ℕ)
  (total_winning_amount : ℕ)
  (profit : ℕ) :

  tickets_bought = 200 →
  cost_per_ticket = 2 →
  percentage_winning = 20 →
  winning_tickets_percentage_5dollars = 80 →
  grand_prize = 5000 →
  average_other_prizes = 10 →
  total_tickets = tickets_bought →
  total_cost = total_tickets * cost_per_ticket →
  winning_tickets = (percentage_winning * total_tickets) / 100 →
  tickets_prize_5dollars = (winning_tickets_percentage_5dollars * winning_tickets) / 100 →
  amount_won_5dollars = tickets_prize_5dollars * 5 →
  other_winning_tickets = winning_tickets - 1 →
  other_tickets_prize = (other_winning_tickets - tickets_prize_5dollars) * average_other_prizes →
  total_winning_amount = amount_won_5dollars + grand_prize + other_tickets_prize →
  profit = total_winning_amount - total_cost →
  profit = 4830 := 
sorry

end james_profit_l288_28833


namespace braden_total_amount_after_winning_l288_28840

noncomputable def initial_amount := 400
noncomputable def multiplier := 2

def total_amount_after_winning (initial: ℕ) (mult: ℕ) : ℕ := initial + (mult * initial)

theorem braden_total_amount_after_winning : total_amount_after_winning initial_amount multiplier = 1200 := by
  sorry

end braden_total_amount_after_winning_l288_28840


namespace balance_five_diamonds_bullets_l288_28807

variables (a b c : ℝ)

-- Conditions
def condition1 : Prop := 4 * a + 2 * b = 12 * c
def condition2 : Prop := 2 * a = b + 4 * c

-- Theorem statement
theorem balance_five_diamonds_bullets (h1 : condition1 a b c) (h2 : condition2 a b c) : 5 * b = 5 * c :=
by
  sorry

end balance_five_diamonds_bullets_l288_28807


namespace cubed_gt_if_gt_l288_28855

theorem cubed_gt_if_gt {a b : ℝ} (h : a > b) : a^3 > b^3 :=
sorry

end cubed_gt_if_gt_l288_28855


namespace find_width_of_plot_l288_28815

def length : ℕ := 90
def poles : ℕ := 52
def distance_between_poles : ℕ := 5
def perimeter : ℕ := poles * distance_between_poles

theorem find_width_of_plot (perimeter_eq : perimeter = 2 * (length + width)) : width = 40 := by
  sorry

end find_width_of_plot_l288_28815


namespace positive_difference_of_sums_l288_28816

def sum_first_n (n : Nat) : Nat := n * (n + 1) / 2

def sum_first_n_even (n : Nat) : Nat := 2 * sum_first_n n

def sum_first_n_odd (n : Nat) : Nat := n * n

theorem positive_difference_of_sums :
  let S1 := sum_first_n_even 25
  let S2 := sum_first_n_odd 20
  S1 - S2 = 250 := by
  sorry

end positive_difference_of_sums_l288_28816


namespace necessary_but_not_sufficient_condition_l288_28897

theorem necessary_but_not_sufficient_condition (a : ℝ) :
  (a < 2 → a^2 < 2 * a) ∧ (a^2 < 2 * a → 0 < a ∧ a < 2) :=
sorry

end necessary_but_not_sufficient_condition_l288_28897


namespace mary_money_after_purchase_l288_28802

def mary_initial_money : ℕ := 58
def pie_cost : ℕ := 6
def mary_friend_money : ℕ := 43  -- This is an extraneous condition, included for completeness.

theorem mary_money_after_purchase : mary_initial_money - pie_cost = 52 := by
  sorry

end mary_money_after_purchase_l288_28802


namespace triangle_inequality_a2_lt_ab_ac_l288_28850

theorem triangle_inequality_a2_lt_ab_ac {a b c : ℝ} (h1 : a < b + c) (h2 : 0 < a) : a^2 < a * b + a * c := 
sorry

end triangle_inequality_a2_lt_ab_ac_l288_28850


namespace nat_add_ge_3_implies_at_least_one_ge_2_l288_28856

theorem nat_add_ge_3_implies_at_least_one_ge_2 (a b : ℕ) (h : a + b ≥ 3) : a ≥ 2 ∨ b ≥ 2 :=
by {
  sorry
}

end nat_add_ge_3_implies_at_least_one_ge_2_l288_28856


namespace annika_total_east_hike_distance_l288_28876

def annika_flat_rate : ℝ := 10 -- minutes per kilometer on flat terrain
def annika_initial_distance : ℝ := 2.75 -- kilometers already hiked east
def total_time : ℝ := 45 -- minutes
def uphill_rate : ℝ := 15 -- minutes per kilometer on uphill
def downhill_rate : ℝ := 5 -- minutes per kilometer on downhill
def uphill_distance : ℝ := 0.5 -- kilometer of uphill section
def downhill_distance : ℝ := 0.5 -- kilometer of downhill section

theorem annika_total_east_hike_distance :
  let total_uphill_time := uphill_distance * uphill_rate
  let total_downhill_time := downhill_distance * downhill_rate
  let time_for_uphill_and_downhill := total_uphill_time + total_downhill_time
  let time_available_for_outward_hike := total_time / 2
  let remaining_time_after_up_down := time_available_for_outward_hike - time_for_uphill_and_downhill
  let additional_flat_distance := remaining_time_after_up_down / annika_flat_rate
  (annika_initial_distance + additional_flat_distance) = 4 :=
by
  sorry

end annika_total_east_hike_distance_l288_28876


namespace meaningful_expression_iff_l288_28832

theorem meaningful_expression_iff (x : ℝ) : (∃ y, y = (2 : ℝ) / (2*x - 1)) ↔ x ≠ (1 / 2 : ℝ) :=
by
  sorry

end meaningful_expression_iff_l288_28832


namespace x_increase_80_percent_l288_28896

noncomputable def percentage_increase (x1 x2 : ℝ) : ℝ :=
  ((x2 / x1) - 1) * 100

theorem x_increase_80_percent
  (x1 y1 x2 y2 : ℝ)
  (h1 : x1 * y1 = x2 * y2)
  (h2 : y2 = (5 / 9) * y1) :
  percentage_increase x1 x2 = 80 :=
by
  sorry

end x_increase_80_percent_l288_28896


namespace inverse_proportion_graph_l288_28890

theorem inverse_proportion_graph (m n : ℝ) (h : n = -2 / m) : m = -2 / n :=
by
  sorry

end inverse_proportion_graph_l288_28890


namespace domain_of_f_range_of_f_monotonic_increasing_interval_of_f_l288_28889

open Real

noncomputable def f (x : ℝ) : ℝ := log (9 - x^2)

theorem domain_of_f : Set.Ioo (-3 : ℝ) 3 = {x : ℝ | -3 < x ∧ x < 3} :=
by
  sorry

theorem range_of_f : ∃ y : ℝ, y ∈ Set.Iic (2 * log 3) :=
by
  sorry

theorem monotonic_increasing_interval_of_f : 
  {x : ℝ | -3 < x} ∩ {x : ℝ | 0 ≥ x} = Set.Ioc (-3 : ℝ) 0 :=
by
  sorry

end domain_of_f_range_of_f_monotonic_increasing_interval_of_f_l288_28889


namespace negation_of_proposition_l288_28895

open Classical

theorem negation_of_proposition :
  (∃ x : ℝ, x^2 + 2 * x + 5 ≤ 0) ↔ ¬(∀ x : ℝ, x^2 + 2 * x + 5 > 0) := by
  sorry

end negation_of_proposition_l288_28895


namespace sufficient_y_wages_l288_28810

noncomputable def days_sufficient_for_y_wages (Wx Wy : ℝ) (total_money : ℝ) : ℝ :=
  total_money / Wy

theorem sufficient_y_wages
  (Wx Wy : ℝ)
  (H1 : ∀(D : ℝ), total_money = D * Wx → D = 36 )
  (H2 : total_money = 20 * (Wx + Wy)) :
  days_sufficient_for_y_wages Wx Wy total_money = 45 := by
  sorry

end sufficient_y_wages_l288_28810


namespace largest_power_of_two_dividing_7_pow_2048_minus_1_l288_28830

theorem largest_power_of_two_dividing_7_pow_2048_minus_1 :
  ∃ n : ℕ, 2^n ∣ (7^2048 - 1) ∧ n = 14 :=
by
  use 14
  sorry

end largest_power_of_two_dividing_7_pow_2048_minus_1_l288_28830


namespace ellipse_eccentricity_l288_28809

theorem ellipse_eccentricity (a b : ℝ) (c : ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : c^2 = a^2 - b^2) : (c / a = Real.sqrt 5 / 5) :=
by
  sorry

end ellipse_eccentricity_l288_28809


namespace empty_pencil_cases_l288_28804

theorem empty_pencil_cases (total_cases pencil_cases pen_cases both_cases : ℕ) 
  (h1 : total_cases = 10)
  (h2 : pencil_cases = 5)
  (h3 : pen_cases = 4)
  (h4 : both_cases = 2) : total_cases - (pencil_cases + pen_cases - both_cases) = 3 := by
  sorry

end empty_pencil_cases_l288_28804


namespace number_of_dimes_paid_l288_28869

theorem number_of_dimes_paid (cost_in_dollars : ℕ) (value_of_dime_in_cents : ℕ) (value_of_dollar_in_cents : ℕ) 
  (h_cost : cost_in_dollars = 9) (h_dime : value_of_dime_in_cents = 10) (h_dollar : value_of_dollar_in_cents = 100) : 
  (cost_in_dollars * value_of_dollar_in_cents) / value_of_dime_in_cents = 90 := by
  -- Proof to be provided here
  sorry

end number_of_dimes_paid_l288_28869


namespace correct_statements_about_C_l288_28888

-- Conditions: Curve C is defined by the equation x^4 + y^2 = 1
def C (x y : ℝ) : Prop := x^4 + y^2 = 1

-- Prove the properties of curve C
theorem correct_statements_about_C :
  (-- 1. Symmetric about the x-axis
    (∀ x y : ℝ, C x y → C x (-y)) ∧
    -- 2. Symmetric about the y-axis
    (∀ x y : ℝ, C x y → C (-x) y) ∧
    -- 3. Symmetric about the origin
    (∀ x y : ℝ, C x y → C (-x) (-y)) ∧
    -- 6. A closed figure with an area greater than π
    (∃ (area : ℝ), area > π)) := sorry

end correct_statements_about_C_l288_28888


namespace john_pin_discount_l288_28857

theorem john_pin_discount :
  ∀ (n_pins price_per_pin amount_spent discount_rate : ℝ),
    n_pins = 10 →
    price_per_pin = 20 →
    amount_spent = 170 →
    discount_rate = ((n_pins * price_per_pin - amount_spent) / (n_pins * price_per_pin)) * 100 →
    discount_rate = 15 :=
by
  intros n_pins price_per_pin amount_spent discount_rate h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  sorry

end john_pin_discount_l288_28857


namespace right_triangle_inscribed_circle_inequality_l288_28871

theorem right_triangle_inscribed_circle_inequality 
  {a b c r : ℝ} (h : a^2 + b^2 = c^2) (hr : r = (a + b - c) / 2) : 
  r ≤ (c / 2) * (Real.sqrt 2 - 1) :=
sorry

end right_triangle_inscribed_circle_inequality_l288_28871


namespace scientific_notation_650000_l288_28823

theorem scientific_notation_650000 : 
  ∃ (a : ℝ) (n : ℤ), (1 ≤ |a| ∧ |a| < 10) ∧ 650000 = a * 10 ^ n ∧ a = 6.5 ∧ n = 5 :=
  sorry

end scientific_notation_650000_l288_28823


namespace part1_part2_part3_l288_28872

variable {x : ℝ}

def A := {x : ℝ | x^2 + 3 * x - 4 > 0}
def B := {x : ℝ | x^2 - x - 6 < 0}
def C_R (S : Set ℝ) := {x : ℝ | x ∉ S}

theorem part1 : (A ∩ B) = {x : ℝ | 1 < x ∧ x < 3} := sorry

theorem part2 : (C_R (A ∩ B)) = {x : ℝ | x ≤ 1 ∨ x ≥ 3} := sorry

theorem part3 : (A ∪ (C_R B)) = {x : ℝ | x ≤ -2 ∨ x > 1} := sorry

end part1_part2_part3_l288_28872


namespace bones_in_beef_l288_28806

def price_of_beef_with_bones : ℝ := 78
def price_of_boneless_beef : ℝ := 90
def price_of_bones : ℝ := 15
def fraction_of_bones_in_kg : ℝ := 0.16
def grams_per_kg : ℝ := 1000

theorem bones_in_beef :
  (fraction_of_bones_in_kg * grams_per_kg = 160) :=
by
  sorry

end bones_in_beef_l288_28806


namespace sqrt_pow_simplification_l288_28870

theorem sqrt_pow_simplification :
  (Real.sqrt ((Real.sqrt 5) ^ 5)) ^ 6 = 125 * (5 ^ (3 / 4)) :=
by
  sorry

end sqrt_pow_simplification_l288_28870


namespace rent_cost_l288_28892

-- Definitions based on conditions
def daily_supplies_cost : ℕ := 12
def price_per_pancake : ℕ := 2
def pancakes_sold_per_day : ℕ := 21

-- Proving the daily rent cost
theorem rent_cost (total_sales : ℕ) (rent : ℕ) :
  total_sales = pancakes_sold_per_day * price_per_pancake →
  rent = total_sales - daily_supplies_cost →
  rent = 30 :=
by
  intro h_total_sales h_rent
  sorry

end rent_cost_l288_28892


namespace range_of_a_l288_28894

theorem range_of_a (a : ℝ) : ((1 - a)^2 + (1 + a)^2 < 4) ↔ (-1 < a ∧ a < 1) := 
sorry

end range_of_a_l288_28894


namespace probability_of_different_colors_l288_28873

theorem probability_of_different_colors :
  let total_chips := 12
  let prob_blue_then_yellow_red := ((6 / total_chips) * ((4 + 2) / total_chips))
  let prob_yellow_then_blue_red := ((4 / total_chips) * ((6 + 2) / total_chips))
  let prob_red_then_blue_yellow := ((2 / total_chips) * ((6 + 4) / total_chips))
  prob_blue_then_yellow_red + prob_yellow_then_blue_red + prob_red_then_blue_yellow = 11 / 18 := by
    sorry

end probability_of_different_colors_l288_28873


namespace arithmetic_sequence_nth_term_l288_28893

theorem arithmetic_sequence_nth_term (x n : ℝ) 
  (h1 : 3*x - 4 = a1)
  (h2 : 7*x - 14 = a2)
  (h3 : 4*x + 6 = a3)
  (h4 : a_n = 3012) :
n = 392 :=
  sorry

end arithmetic_sequence_nth_term_l288_28893


namespace find_x_l288_28834

theorem find_x (x : ℝ) (h : 1 - 1 / (1 - x) = 1 / (1 - x)) : x = -1 :=
by
  sorry

end find_x_l288_28834


namespace total_area_of_field_l288_28848

theorem total_area_of_field (A1 A2 : ℝ) (h1 : A1 = 225)
    (h2 : A2 - A1 = (1 / 5) * ((A1 + A2) / 2)) :
  A1 + A2 = 500 := by
  sorry

end total_area_of_field_l288_28848


namespace value_of_h_otimes_h_otimes_h_l288_28877

variable (h x y : ℝ)

-- Define the new operation
def otimes (x y : ℝ) := x^3 - x * y + y^2

-- Prove that h ⊗ (h ⊗ h) = h^6 - h^4 + h^3
theorem value_of_h_otimes_h_otimes_h :
  otimes h (otimes h h) = h^6 - h^4 + h^3 := by
  sorry

end value_of_h_otimes_h_otimes_h_l288_28877


namespace elijah_total_cards_l288_28827

-- Define the conditions
def num_decks : ℕ := 6
def cards_per_deck : ℕ := 52

-- The main statement that we need to prove
theorem elijah_total_cards : num_decks * cards_per_deck = 312 := by
  -- We skip the proof
  sorry

end elijah_total_cards_l288_28827


namespace average_weight_of_class_l288_28879

theorem average_weight_of_class (students_a students_b : ℕ) (avg_weight_a avg_weight_b : ℝ)
  (h_students_a : students_a = 24)
  (h_students_b : students_b = 16)
  (h_avg_weight_a : avg_weight_a = 40)
  (h_avg_weight_b : avg_weight_b = 35) :
  ((students_a * avg_weight_a + students_b * avg_weight_b) / (students_a + students_b)) = 38 := 
by
  sorry

end average_weight_of_class_l288_28879


namespace fg_of_3_eq_neg5_l288_28818

-- Definitions from the conditions
def f (x : ℝ) : ℝ := 2 * x - 5
def g (x : ℝ) : ℝ := x^2 - 4 * x + 3

-- Lean statement to prove question == answer
theorem fg_of_3_eq_neg5 : f (g 3) = -5 := by
  sorry

end fg_of_3_eq_neg5_l288_28818


namespace line_through_point_and_isosceles_triangle_l288_28836

def is_line_eq (a b c : ℝ) (x y : ℝ) : Prop :=
  a * x + b * y + c = 0

def is_isosceles_right_triangle_with_axes (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∨ a < 0 ∧ b < 0

theorem line_through_point_and_isosceles_triangle (a b c : ℝ) (hx : ℝ) (hy : ℝ) :
  is_line_eq a b c hx hy ∧ is_isosceles_right_triangle_with_axes a b → 
  ((a = 1 ∧ b = 1 ∧ c = -3) ∨ (a = 1 ∧ b = -1 ∧ c = -1)) :=
by
  sorry

end line_through_point_and_isosceles_triangle_l288_28836


namespace price_of_one_rose_l288_28828

theorem price_of_one_rose
  (tulips1 tulips2 tulips3 roses1 roses2 roses3 : ℕ)
  (price_tulip : ℕ)
  (total_earnings : ℕ)
  (R : ℕ) :
  tulips1 = 30 →
  roses1 = 20 →
  tulips2 = 2 * tulips1 →
  roses2 = 2 * roses1 →
  tulips3 = 10 * tulips2 / 100 →  -- simplification of 0.1 * tulips2
  roses3 = 16 →
  price_tulip = 2 →
  total_earnings = 420 →
  (96 * price_tulip + 76 * R) = total_earnings →
  R = 3 :=
by
  intros
  -- Proof will go here
  sorry

end price_of_one_rose_l288_28828


namespace value_of_x_l288_28821

theorem value_of_x (x : ℝ) (h : x = 80 + 0.2 * 80) : x = 96 :=
sorry

end value_of_x_l288_28821


namespace sum_of_integers_between_neg20_5_and_10_5_l288_28861

noncomputable def sum_arithmetic_series (a l n : ℤ) : ℤ :=
  n * (a + l) / 2

theorem sum_of_integers_between_neg20_5_and_10_5 :
  (sum_arithmetic_series (-20) 10 31) = -155 := by
  sorry

end sum_of_integers_between_neg20_5_and_10_5_l288_28861


namespace milkman_total_profit_l288_28813

-- Declare the conditions
def initialMilk : ℕ := 50
def initialWater : ℕ := 15
def firstMixtureMilk : ℕ := 30
def firstMixtureWater : ℕ := 8
def remainingMilk : ℕ := initialMilk - firstMixtureMilk
def secondMixtureMilk : ℕ := remainingMilk
def secondMixtureWater : ℕ := 7
def costOfMilkPerLiter : ℕ := 20
def sellingPriceFirstMixturePerLiter : ℕ := 17
def sellingPriceSecondMixturePerLiter : ℕ := 15
def totalCostOfMilk := (firstMixtureMilk + secondMixtureMilk) * costOfMilkPerLiter
def totalRevenueFirstMixture := (firstMixtureMilk + firstMixtureWater) * sellingPriceFirstMixturePerLiter
def totalRevenueSecondMixture := (secondMixtureMilk + secondMixtureWater) * sellingPriceSecondMixturePerLiter
def totalRevenue := totalRevenueFirstMixture + totalRevenueSecondMixture
def totalProfit := totalRevenue - totalCostOfMilk

-- Proof statement
theorem milkman_total_profit : totalProfit = 51 := by
  sorry

end milkman_total_profit_l288_28813


namespace crease_length_l288_28842

theorem crease_length (A B C : ℝ) (h1 : A = 5) (h2 : B = 12) (h3 : C = 13) : ∃ D, D = 6.5 :=
by
  sorry

end crease_length_l288_28842


namespace find_point_on_parabola_l288_28812

noncomputable def parabola (x y : ℝ) : Prop := y^2 = 6 * x
def positive_y (y : ℝ) : Prop := y > 0
def distance_to_focus (x y : ℝ) : Prop := (x - 3/2)^2 + y^2 = (5/2)^2 

theorem find_point_on_parabola (x y : ℝ) :
  parabola x y ∧ positive_y y ∧ distance_to_focus x y → (x = 1 ∧ y = Real.sqrt 6) :=
by
  sorry

end find_point_on_parabola_l288_28812


namespace boaster_guarantee_distinct_balls_l288_28852

noncomputable def canGuaranteeDistinctBallCounts (boxes : Fin 2018 → ℕ) (pairs : Fin 4032 → (Fin 2018 × Fin 2018)) : Prop :=
  ∀ i j : Fin 2018, i ≠ j → boxes i ≠ boxes j

theorem boaster_guarantee_distinct_balls :
  ∃ (boxes : Fin 2018 → ℕ) (pairs : Fin 4032 → (Fin 2018 × Fin 2018)),
  canGuaranteeDistinctBallCounts boxes pairs :=
sorry

end boaster_guarantee_distinct_balls_l288_28852


namespace statue_of_liberty_model_height_l288_28867

theorem statue_of_liberty_model_height :
  let scale_ratio : Int := 30
  let actual_height : Int := 305
  round (actual_height / scale_ratio) = 10 := by
  sorry

end statue_of_liberty_model_height_l288_28867


namespace length_of_FD_l288_28882

-- Define the conditions
def is_square (ABCD : ℝ) (side_length : ℝ) : Prop :=
  side_length = 8 ∧ ABCD = 4 * side_length

def point_E (x : ℝ) : Prop :=
  x = 8 / 3

def point_F (CD : ℝ) (x : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 8

-- State the theorem
theorem length_of_FD (side_length : ℝ) (x : ℝ) (CD ED FD : ℝ) :
  is_square 4 side_length → 
  point_E ED → 
  point_F CD x → 
  FD = 20 / 9 :=
by
  sorry

end length_of_FD_l288_28882


namespace sum_three_circles_l288_28839

theorem sum_three_circles (a b : ℚ) 
  (h1 : 5 * a + 2 * b = 27)
  (h2 : 2 * a + 5 * b = 29) :
  3 * b = 13 :=
by
  sorry

end sum_three_circles_l288_28839


namespace population_present_l288_28862

variable (P : ℝ)

theorem population_present (h1 : P * 0.90 = 450) : P = 500 :=
by
  sorry

end population_present_l288_28862


namespace exists_multiple_of_power_of_two_non_zero_digits_l288_28853

open Nat

theorem exists_multiple_of_power_of_two_non_zero_digits (k : ℕ) (h : 0 < k) : 
  ∃ m : ℕ, (2^k ∣ m) ∧ (∀ d ∈ digits 10 m, d ≠ 0) :=
sorry

end exists_multiple_of_power_of_two_non_zero_digits_l288_28853


namespace triangles_from_decagon_l288_28874

theorem triangles_from_decagon (vertices : Fin 10 → Prop) 
  (h : ∀ (a b c : Fin 10), a ≠ b ∧ b ≠ c ∧ a ≠ c → Prop) :
  ∃ triangles : ℕ, triangles = 120 :=
by
  sorry

end triangles_from_decagon_l288_28874


namespace average_salary_of_all_workers_l288_28875

theorem average_salary_of_all_workers :
  let technicians := 7
  let technicians_avg_salary := 20000
  let rest := 49 - technicians
  let rest_avg_salary := 6000
  let total_workers := 49
  let total_tech_salary := technicians * technicians_avg_salary
  let total_rest_salary := rest * rest_avg_salary
  let total_salary := total_tech_salary + total_rest_salary
  (total_salary / total_workers) = 8000 := by
  sorry

end average_salary_of_all_workers_l288_28875


namespace fish_remain_approximately_correct_l288_28801

noncomputable def remaining_fish : ℝ :=
  let west_initial := 1800
  let east_initial := 3200
  let north_initial := 500
  let south_initial := 2300
  let a := 3
  let b := 4
  let c := 2
  let d := 5
  let e := 1
  let f := 3
  let west_caught := (a / b) * west_initial
  let east_caught := (c / d) * east_initial
  let south_caught := (e / f) * south_initial
  let west_left := west_initial - west_caught
  let east_left := east_initial - east_caught
  let south_left := south_initial - south_caught
  let north_left := north_initial
  west_left + east_left + south_left + north_left

theorem fish_remain_approximately_correct :
  abs (remaining_fish - 4403) < 1 := 
  sorry

end fish_remain_approximately_correct_l288_28801


namespace range_a_sub_b_mul_c_l288_28887

theorem range_a_sub_b_mul_c (a b c : ℝ) (h1 : -1 < a) (h2 : a < b) (h3 : b < 1) (h4 : 2 < c) (h5 : c < 3) :
  -6 < (a - b) * c ∧ (a - b) * c < 0 :=
by
  -- We need to prove the range of (a - b) * c is within (-6, 0)
  sorry

end range_a_sub_b_mul_c_l288_28887


namespace tangent_line_at_point_l288_28847

noncomputable def tangent_line_equation (f : ℝ → ℝ) (x : ℝ) : Prop :=
  x - f 0 + 2 = 0

theorem tangent_line_at_point (f : ℝ → ℝ)
  (h_mono : ∀ x y : ℝ, x ≤ y → f x ≤ f y)
  (h_eq : ∀ x : ℝ, f (f x - Real.exp x) = Real.exp 1 + 1) :
  tangent_line_equation f 0 :=
by
  sorry

end tangent_line_at_point_l288_28847


namespace dice_probability_sum_18_l288_28808

theorem dice_probability_sum_18 : 
  (∃ d1 d2 d3 : ℕ, 1 ≤ d1 ∧ d1 ≤ 8 ∧ 1 ≤ d2 ∧ d2 ≤ 8 ∧ 1 ≤ d3 ∧ d3 ≤ 8 ∧ d1 + d2 + d3 = 18) →
  (1/8 : ℚ) * (1/8) * (1/8) * 9 = 9 / 512 :=
by 
  sorry

end dice_probability_sum_18_l288_28808


namespace fraction_problem_l288_28805

theorem fraction_problem (x : ℝ) (h₁ : x * 180 = 18) (h₂ : x < 0.15) : x = 1/10 :=
by sorry

end fraction_problem_l288_28805


namespace diving_classes_on_weekdays_l288_28858

theorem diving_classes_on_weekdays 
  (x : ℕ) 
  (weekend_classes_per_day : ℕ := 4)
  (people_per_class : ℕ := 5)
  (total_people_3_weeks : ℕ := 270)
  (weekend_days : ℕ := 2)
  (total_weeks : ℕ := 3)
  (weekend_total_classes : ℕ := weekend_classes_per_day * weekend_days * total_weeks) 
  (total_people_weekends : ℕ := weekend_total_classes * people_per_class) 
  (total_people_weekdays : ℕ := total_people_3_weeks - total_people_weekends)
  (weekday_classes_needed : ℕ := total_people_weekdays / people_per_class)
  (weekly_weekday_classes : ℕ := weekday_classes_needed / total_weeks)
  (h : weekly_weekday_classes = x)
  : x = 10 := sorry

end diving_classes_on_weekdays_l288_28858


namespace min_value_x_plus_y_l288_28885

theorem min_value_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1/x + 9/y = 1) : x + y ≥ 16 :=
sorry

end min_value_x_plus_y_l288_28885


namespace count_valid_integers_1_to_999_l288_28817

-- Define a function to count the valid integers
def count_valid_integers : Nat :=
  let digits := [1, 2, 6, 7, 9]
  let one_digit_count := 5
  let two_digit_count := 5 * 5
  let three_digit_count := 5 * 5 * 5
  one_digit_count + two_digit_count + three_digit_count

-- The theorem we want to prove
theorem count_valid_integers_1_to_999 : count_valid_integers = 155 := by
  sorry

end count_valid_integers_1_to_999_l288_28817


namespace price_per_glass_second_day_l288_28803

theorem price_per_glass_second_day 
  (O W : ℕ)  -- O is the amount of orange juice used on each day, W is the amount of water used on the first day
  (V : ℕ)   -- V is the volume of one glass
  (P₁ : ℚ)  -- P₁ is the price per glass on the first day
  (P₂ : ℚ)  -- P₂ is the price per glass on the second day
  (h1 : W = O)  -- First day, water is equal to orange juice
  (h2 : V > 0)  -- Volume of one glass > 0
  (h3 : P₁ = 0.48)  -- Price per glass on the first day
  (h4 : (2 * O / V) * P₁ = (3 * O / V) * P₂)  -- Revenue's are the same
  : P₂ = 0.32 :=  -- Prove that price per glass on the second day is 0.32
by
  sorry

end price_per_glass_second_day_l288_28803


namespace anne_initial_sweettarts_l288_28845

variable (x : ℕ)
variable (num_friends : ℕ := 3)
variable (sweettarts_per_friend : ℕ := 5)
variable (total_sweettarts_given : ℕ := num_friends * sweettarts_per_friend)

theorem anne_initial_sweettarts 
  (h1 : ∀ person, person < num_friends → sweettarts_per_friend = 5)
  (h2 : total_sweettarts_given = 15) : 
  total_sweettarts_given = 15 := 
by 
  sorry

end anne_initial_sweettarts_l288_28845


namespace race_course_length_proof_l288_28835

def race_course_length (L : ℝ) (v_A v_B : ℝ) : Prop :=
  v_A = 4 * v_B ∧ (L / v_A = (L - 66) / v_B) → L = 88

theorem race_course_length_proof (v_A v_B : ℝ) : race_course_length 88 v_A v_B :=
by 
  intros
  sorry

end race_course_length_proof_l288_28835


namespace fraction_of_number_l288_28831

theorem fraction_of_number (x f : ℚ) (h1 : x = 2/3) (h2 : f * x = (64/216) * (1/x)) : f = 2/3 :=
by
  sorry

end fraction_of_number_l288_28831


namespace geometric_then_sum_geometric_l288_28886

variable {a b c d : ℝ}

def geometric_sequence (a b c d : ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ c = b * r ∧ d = c * r

def forms_geometric_sequence (x y z : ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ y = x * r ∧ z = y * r

theorem geometric_then_sum_geometric (h : geometric_sequence a b c d) :
  forms_geometric_sequence (a + b) (b + c) (c + d) :=
sorry

end geometric_then_sum_geometric_l288_28886


namespace xy2_plus_2y_divides_2x2y_plus_xy2_plus_8x_l288_28800

theorem xy2_plus_2y_divides_2x2y_plus_xy2_plus_8x (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) :
  (x * y^2 + 2 * y) ∣ (2 * x^2 * y + x * y^2 + 8 * x) ↔ 
  (∃ a : ℕ, 0 < a ∧ x = a ∧ y = 2 * a) ∨ (x = 3 ∧ y = 1) ∨ (x = 8 ∧ y = 1) :=
by
  sorry

end xy2_plus_2y_divides_2x2y_plus_xy2_plus_8x_l288_28800


namespace no_valid_road_network_l288_28863

theorem no_valid_road_network
  (k_A k_B k_C : ℕ)
  (h_kA : k_A ≥ 2)
  (h_kB : k_B ≥ 2)
  (h_kC : k_C ≥ 2) :
  ¬ ∃ (t : ℕ) (d : ℕ → ℕ), t ≥ 7 ∧ 
    (∀ i j, i ≠ j → d i ≠ d j) ∧
    (∀ i, i < 4 * (k_A + k_B + k_C) + 4 → d i = i + 1) :=
sorry

end no_valid_road_network_l288_28863


namespace walter_coins_value_l288_28899

theorem walter_coins_value :
  let pennies : ℕ := 2
  let nickels : ℕ := 2
  let dimes : ℕ := 1
  let quarters : ℕ := 1
  let half_dollars : ℕ := 1
  let penny_value : ℕ := 1
  let nickel_value : ℕ := 5
  let dime_value : ℕ := 10
  let quarter_value : ℕ := 25
  let half_dollar_value : ℕ := 50
  (pennies * penny_value + nickels * nickel_value + dimes * dime_value + quarters * quarter_value + half_dollars * half_dollar_value) = 97 := 
sorry

end walter_coins_value_l288_28899


namespace range_of_f_l288_28878

noncomputable def f (x : ℝ) : ℝ :=
  (Real.exp (3 * x) - 2) / (Real.exp (3 * x) + 2)

theorem range_of_f (x : ℝ) : -1 < f x ∧ f x < 1 :=
by
  sorry

end range_of_f_l288_28878


namespace find_a_l288_28880

def star (a b : ℕ) : ℕ := 3 * a - b ^ 2

theorem find_a (a : ℕ) (b : ℕ) (h : star a b = 14) : a = 10 :=
by sorry

end find_a_l288_28880


namespace find_solution_l288_28884

theorem find_solution (x : ℝ) (h : (5 + x / 3)^(1/3) = -4) : x = -207 :=
sorry

end find_solution_l288_28884


namespace sales_tax_difference_l288_28811

theorem sales_tax_difference
  (item_price : ℝ)
  (rate1 rate2 : ℝ)
  (h_rate1 : rate1 = 0.0725)
  (h_rate2 : rate2 = 0.0675)
  (h_item_price : item_price = 40) :
  item_price * rate1 - item_price * rate2 = 0.20 :=
by
  -- Since we are required to skip the proof, we put sorry here.
  sorry

end sales_tax_difference_l288_28811


namespace male_listeners_l288_28837

structure Survey :=
  (males_dont_listen : Nat)
  (females_listen : Nat)
  (total_listeners : Nat)
  (total_dont_listen : Nat)

def number_of_females_dont_listen (s : Survey) : Nat :=
  s.total_dont_listen - s.males_dont_listen

def number_of_males_listen (s : Survey) : Nat :=
  s.total_listeners - s.females_listen

theorem male_listeners (s : Survey) (h : s = { males_dont_listen := 85, females_listen := 75, total_listeners := 180, total_dont_listen := 160 }) :
  number_of_males_listen s = 105 :=
by
  sorry

end male_listeners_l288_28837


namespace find_ending_number_l288_28851

theorem find_ending_number (N : ℕ) :
  (∃ k : ℕ, N = 3 * k) ∧ (∀ x,  40 < x ∧ x ≤ N → x % 3 = 0) ∧ (∃ avg, avg = (N + 42) / 2 ∧ avg = 60) → N = 78 :=
by
  sorry

end find_ending_number_l288_28851


namespace quadratic_solutions_l288_28865

theorem quadratic_solutions:
  (2 * (x : ℝ)^2 - 5 * x + 2 = 0) ↔ (x = 2 ∨ x = 1 / 2) :=
sorry

end quadratic_solutions_l288_28865


namespace find_2n_plus_m_l288_28822

theorem find_2n_plus_m (n m : ℤ) (h1 : 3 * n - m < 5) (h2 : n + m > 26) (h3 : 3 * m - 2 * n < 46) : 2 * n + m = 36 :=
sorry

end find_2n_plus_m_l288_28822


namespace absolute_value_inequality_l288_28854

theorem absolute_value_inequality (x : ℝ) : 
  (|3 * x + 1| > 2) ↔ (x > 1/3 ∨ x < -1) := by
  sorry

end absolute_value_inequality_l288_28854


namespace set_intersection_l288_28868

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3, 4}

noncomputable def complement_U_A := U \ A
noncomputable def intersection := B ∩ complement_U_A

theorem set_intersection :
  intersection = ({3, 4} : Set ℕ) := by
  sorry

end set_intersection_l288_28868


namespace focus_of_parabola_l288_28829

theorem focus_of_parabola : 
  (∃ p : ℝ, y^2 = 4 * p * x ∧ p = 1 ∧ ∃ c : ℝ × ℝ, c = (1, 0)) :=
sorry

end focus_of_parabola_l288_28829


namespace point_A_in_third_quadrant_l288_28846

-- Defining the point A with its coordinates
structure Point :=
  (x : Int)
  (y : Int)

def A : Point := ⟨-1, -3⟩

-- The definition of quadrants in Cartesian coordinate system
def quadrant (p : Point) : String :=
  if p.x > 0 ∧ p.y > 0 then "first"
  else if p.x < 0 ∧ p.y > 0 then "second"
  else if p.x < 0 ∧ p.y < 0 then "third"
  else if p.x > 0 ∧ p.y < 0 then "fourth"
  else "boundary"

-- The theorem we want to prove
theorem point_A_in_third_quadrant : quadrant A = "third" :=
by 
  sorry

end point_A_in_third_quadrant_l288_28846


namespace solve_equation_l288_28841

theorem solve_equation : ∀ x : ℝ, (2 * x - 8 = 0) ↔ (x = 4) :=
by sorry

end solve_equation_l288_28841


namespace division_pow_zero_l288_28844

theorem division_pow_zero (a b : ℝ) (hb : b ≠ 0) : ((a / b) ^ 0 = (1 : ℝ)) :=
by
  sorry

end division_pow_zero_l288_28844


namespace correct_cost_per_piece_l288_28826

-- Definitions for the given conditions
def totalPaid : ℝ := 20700
def reimbursement : ℝ := 600
def numberOfPieces : ℝ := 150
def correctTotal := totalPaid - reimbursement

-- Theorem stating the correct cost per piece of furniture
theorem correct_cost_per_piece : correctTotal / numberOfPieces = 134 := 
by
  sorry

end correct_cost_per_piece_l288_28826


namespace part1_f0_f1_part1_f_neg1_f2_part1_f_neg2_f3_part2_conjecture_l288_28860

noncomputable def f (x : ℝ) : ℝ := 1 / (3^x + Real.sqrt 3)

theorem part1_f0_f1 : f 0 + f 1 = Real.sqrt 3 / 3 := sorry

theorem part1_f_neg1_f2 : f (-1) + f 2 = Real.sqrt 3 / 3 := sorry

theorem part1_f_neg2_f3 : f (-2) + f 3 = Real.sqrt 3 / 3 := sorry

theorem part2_conjecture (x1 x2 : ℝ) (h : x1 + x2 = 1) : f x1 + f x2 = Real.sqrt 3 / 3 := sorry

end part1_f0_f1_part1_f_neg1_f2_part1_f_neg2_f3_part2_conjecture_l288_28860


namespace sum_of_squares_l288_28864

theorem sum_of_squares (R r r1 r2 r3 d d1 d2 d3 : ℝ) 
  (h1 : d^2 = R^2 - 2 * R * r)
  (h2 : d1^2 = R^2 + 2 * R * r1)
  (h3 : d^2 + d1^2 + d2^2 + d3^2 = 12 * R^2) :
  d^2 + d1^2 + d2^2 + d3^2 = 12 * R^2 :=
by
  sorry

end sum_of_squares_l288_28864


namespace area_of_fourth_rectangle_l288_28819

-- The conditions provided in the problem
variables (x y z w : ℝ)
variables (h1 : x * y = 24) (h2 : x * w = 12) (h3 : z * w = 8)

-- The problem statement with the conclusion
theorem area_of_fourth_rectangle :
  (∃ (x y z w : ℝ), ((x * y = 24 ∧ x * w = 12 ∧ z * w = 8) ∧ y * z = 16)) :=
sorry

end area_of_fourth_rectangle_l288_28819


namespace wire_cut_problem_l288_28898

-- Conditions
variable (x y : ℝ)
variable (h1 : x = y)
variable (hx : x > 0) -- Assuming positive lengths for the wire pieces

-- Statement to prove
theorem wire_cut_problem : x / y = 1 :=
by sorry

end wire_cut_problem_l288_28898


namespace product_gcd_lcm_150_90_l288_28820

theorem product_gcd_lcm_150_90 (a b : ℕ) (h1 : a = 150) (h2 : b = 90): Nat.gcd a b * Nat.lcm a b = a * b := by
  rw [h1, h2]
  sorry

end product_gcd_lcm_150_90_l288_28820


namespace parts_repetition_cycle_l288_28881

noncomputable def parts_repetition_condition (t : ℕ) : Prop := sorry
def parts_initial_condition : Prop := sorry

theorem parts_repetition_cycle :
  parts_initial_condition →
  parts_repetition_condition 2 ∧
  parts_repetition_condition 4 ∧
  parts_repetition_condition 38 ∧
  parts_repetition_condition 76 :=
sorry


end parts_repetition_cycle_l288_28881


namespace roots_of_quadratic_l288_28859

open Real

theorem roots_of_quadratic (r s : ℝ) (h1 : r + s = 2 * sqrt 3) (h2 : r * s = 2) :
  r^6 + s^6 = 3104 :=
sorry

end roots_of_quadratic_l288_28859


namespace box_office_collection_l288_28891

open Nat

/-- Define the total tickets sold -/
def total_tickets : ℕ := 1500

/-- Define the price of an adult ticket -/
def price_adult_ticket : ℕ := 12

/-- Define the price of a student ticket -/
def price_student_ticket : ℕ := 6

/-- Define the number of student tickets sold -/
def student_tickets : ℕ := 300

/-- Define the number of adult tickets sold -/
def adult_tickets : ℕ := total_tickets - student_tickets

/-- Define the revenue from adult tickets -/
def revenue_adult_tickets : ℕ := adult_tickets * price_adult_ticket

/-- Define the revenue from student tickets -/
def revenue_student_tickets : ℕ := student_tickets * price_student_ticket

/-- Define the total amount collected -/
def total_amount_collected : ℕ := revenue_adult_tickets + revenue_student_tickets

/-- Theorem to prove the total amount collected at the box office -/
theorem box_office_collection : total_amount_collected = 16200 := by
  sorry

end box_office_collection_l288_28891
