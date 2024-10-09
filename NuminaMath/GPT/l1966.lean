import Mathlib

namespace whitney_total_cost_l1966_196612

-- Definitions of the number of items and their costs
def w := 15
def c_w := 14
def f := 12
def c_f := 13
def s := 5
def c_s := 10
def m := 8
def c_m := 3

-- The total cost Whitney spent
theorem whitney_total_cost :
  w * c_w + f * c_f + s * c_s + m * c_m = 440 := by
  sorry

end whitney_total_cost_l1966_196612


namespace find_number_of_children_l1966_196669

theorem find_number_of_children (adults children : ℕ) (adult_ticket_price child_ticket_price total_money change : ℕ) 
    (h1 : adult_ticket_price = 9) 
    (h2 : child_ticket_price = adult_ticket_price - 2) 
    (h3 : total_money = 40) 
    (h4 : change = 1) 
    (h5 : adults = 2) 
    (total_cost : total_money - change = adults * adult_ticket_price + children * child_ticket_price) : 
    children = 3 :=
sorry

end find_number_of_children_l1966_196669


namespace find_b_if_lines_parallel_l1966_196648

theorem find_b_if_lines_parallel (b : ℝ) :
  (∀ x y : ℝ, 3 * y - 3 * b = 9 * x → y = 3 * x + b) ∧
  (∀ x y : ℝ, y + 2 = (b + 9) * x → y = (b + 9) * x - 2) →
  3 = b + 9 →
  b = -6 :=
by {
  sorry
}

end find_b_if_lines_parallel_l1966_196648


namespace max_value_90_l1966_196666

noncomputable def max_value_expression (a b c d : ℝ) : ℝ :=
  a + 2 * b + c + 2 * d - a * b - b * c - c * d - d * a

theorem max_value_90 (a b c d : ℝ) (h₁ : -4.5 ≤ a) (h₂ : a ≤ 4.5)
                                   (h₃ : -4.5 ≤ b) (h₄ : b ≤ 4.5)
                                   (h₅ : -4.5 ≤ c) (h₆ : c ≤ 4.5)
                                   (h₇ : -4.5 ≤ d) (h₈ : d ≤ 4.5) :
  max_value_expression a b c d ≤ 90 :=
sorry

end max_value_90_l1966_196666


namespace quadratic_roots_real_and_equal_l1966_196649

theorem quadratic_roots_real_and_equal (m : ℤ) :
  (∀ x : ℝ, 3 * x^2 + (2 - m) * x + 12 = 0 →
   (∃ r, x = r ∧ 3 * r^2 + (2 - m) * r + 12 = 0)) →
   (m = -10 ∨ m = 14) :=
sorry

end quadratic_roots_real_and_equal_l1966_196649


namespace max_a_if_monotonically_increasing_l1966_196680

noncomputable def f (x a : ℝ) : ℝ := x^3 + Real.exp x - a * x

theorem max_a_if_monotonically_increasing (a : ℝ) : 
  (∀ x, 0 ≤ x → 3 * x^2 + Real.exp x - a ≥ 0) ↔ a ≤ 1 :=
by
  sorry

end max_a_if_monotonically_increasing_l1966_196680


namespace tank_capacity_l1966_196610

theorem tank_capacity (liters_cost : ℕ) (liters_amount : ℕ) (full_tank_cost : ℕ) (h₁ : liters_cost = 18) (h₂ : liters_amount = 36) (h₃ : full_tank_cost = 32) : 
  (full_tank_cost * liters_amount / liters_cost) = 64 :=
by 
  sorry

end tank_capacity_l1966_196610


namespace value_of_a_l1966_196655

-- Definitions of sets A and B
def A : Set ℝ := {x | x^2 = 1}
def B (a : ℝ) : Set ℝ := {x | a * x = 1}

-- The main theorem statement
theorem value_of_a (a : ℝ) (H : B a ⊆ A) : a = 0 ∨ a = 1 ∨ a = -1 :=
by 
  sorry

end value_of_a_l1966_196655


namespace range_of_a_l1966_196642

theorem range_of_a (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x + y = 2 ∧ 
    (if x > 1 then (x^2 + 1) / x else Real.log (x + a)) = 
    (if y > 1 then (y^2 + 1) / y else Real.log (y + a))) ↔ 
    a > Real.exp 2 - 1 :=
by sorry

end range_of_a_l1966_196642


namespace avg_hamburgers_per_day_l1966_196611

theorem avg_hamburgers_per_day (total_hamburgers : ℕ) (days_in_week : ℕ) (h1 : total_hamburgers = 63) (h2 : days_in_week = 7) :
  total_hamburgers / days_in_week = 9 := by
  sorry

end avg_hamburgers_per_day_l1966_196611


namespace real_root_exists_l1966_196694

theorem real_root_exists (a : ℝ) : 
    (∃ x : ℝ, x^4 - a * x^3 - x^2 - a * x + 1 = 0) ↔ (-1 / 2 ≤ a) := by
  sorry

end real_root_exists_l1966_196694


namespace find_g_l1966_196651

noncomputable def g : ℝ → ℝ := sorry

theorem find_g :
  (g 1 = 2) ∧ (∀ x y : ℝ, g (x + y) = 4^y * g x + 3^x * g y) ↔ (∀ x : ℝ, g x = 2 * (4^x - 3^x)) := 
by
  sorry

end find_g_l1966_196651


namespace average_possible_k_l1966_196627

theorem average_possible_k (k : ℕ) (r1 r2 : ℕ) (h : r1 * r2 = 24) (h_pos : r1 > 0 ∧ r2 > 0) (h_eq_k : r1 + r2 = k) : 
  (25 + 14 + 11 + 10) / 4 = 15 :=
by 
  sorry

end average_possible_k_l1966_196627


namespace net_hourly_rate_correct_l1966_196678

noncomputable def net_hourly_rate
    (hours : ℕ) 
    (speed : ℕ) 
    (fuel_efficiency : ℕ) 
    (earnings_per_mile : ℝ) 
    (cost_per_gallon : ℝ) 
    (distance := speed * hours) 
    (gasoline_used := distance / fuel_efficiency) 
    (earnings := earnings_per_mile * distance) 
    (cost_of_gasoline := cost_per_gallon * gasoline_used) 
    (net_earnings := earnings - cost_of_gasoline) : ℝ :=
  net_earnings / hours

theorem net_hourly_rate_correct : 
  net_hourly_rate 3 45 25 0.6 1.8 = 23.76 := 
by 
  unfold net_hourly_rate
  norm_num
  sorry

end net_hourly_rate_correct_l1966_196678


namespace find_3a_plus_3b_l1966_196670

theorem find_3a_plus_3b (a b : ℚ) (h1 : 2 * a + 5 * b = 47) (h2 : 8 * a + 2 * b = 50) :
  3 * a + 3 * b = 73 / 2 := 
sorry

end find_3a_plus_3b_l1966_196670


namespace num_girls_on_trip_l1966_196641

/-- Given the conditions: 
  * Three adults each eating 3 eggs.
  * Ten boys each eating one more egg than each girl.
  * A total of 36 eggs.
  Prove that there are 7 girls on the trip. -/
theorem num_girls_on_trip (adults boys girls eggs : ℕ) 
  (H1 : adults = 3)
  (H2 : boys = 10)
  (H3 : eggs = 36)
  (H4 : ∀ g, (girls * g) + (boys * (g + 1)) + (adults * 3) = eggs)
  (H5 : ∀ g, g = 1) :
  girls = 7 :=
by
  sorry

end num_girls_on_trip_l1966_196641


namespace greatest_possible_triangle_perimeter_l1966_196644

noncomputable def triangle_perimeter (x : ℕ) : ℕ :=
  x + 2 * x + 18

theorem greatest_possible_triangle_perimeter :
  (∃ (x : ℕ), 7 ≤ x ∧ x < 18 ∧ ∀ y : ℕ, (7 ≤ y ∧ y < 18) → triangle_perimeter y ≤ triangle_perimeter x) ∧
  triangle_perimeter 17 = 69 :=
by
  sorry

end greatest_possible_triangle_perimeter_l1966_196644


namespace pieces_after_cuts_l1966_196639

theorem pieces_after_cuts (n : ℕ) : 
  (∃ n, (8 * n + 1 = 2009)) ↔ (n = 251) :=
by 
  sorry

end pieces_after_cuts_l1966_196639


namespace lawrence_walking_speed_l1966_196654

theorem lawrence_walking_speed :
  let distance := 4
  let time := (4 : ℝ) / 3
  let speed := distance / time
  speed = 3 := 
by
  sorry

end lawrence_walking_speed_l1966_196654


namespace find_k_range_for_two_roots_l1966_196620

noncomputable def f (k x : ℝ) : ℝ := (Real.log x / x) - k * x

theorem find_k_range_for_two_roots :
  ∃ k_min k_max : ℝ, k_min = (2 / (Real.exp 4)) ∧ k_max = (1 / (2 * Real.exp 1)) ∧
  ∀ k : ℝ, (k_min ≤ k ∧ k < k_max) ↔
    ∃ x1 x2 : ℝ, 
    (1 / Real.exp 1) ≤ x1 ∧ x1 ≤ Real.exp 2 ∧ 
    (1 / Real.exp 1) ≤ x2 ∧ x2 ≤ Real.exp 2 ∧ 
    f k x1 = 0 ∧ f k x2 = 0 ∧ 
    x1 ≠ x2 :=
sorry

end find_k_range_for_two_roots_l1966_196620


namespace range_of_a_l1966_196671

def A (a : ℝ) : Set ℝ := {x | |x - a| ≤ 1}
def B : Set ℝ := {x | x ≤ 1 ∨ x ≥ 4}

theorem range_of_a (a : ℝ) (h : A a ∩ B = ∅) : 2 < a ∧ a < 3 := sorry

end range_of_a_l1966_196671


namespace find_root_of_equation_l1966_196614

theorem find_root_of_equation (a b c d x : ℕ) (h_ad : a + d = 2016) (h_bc : b + c = 2016) (h_ac : a ≠ c) :
  (x - a) * (x - b) = (x - c) * (x - d) → x = 1008 :=
by
  sorry

end find_root_of_equation_l1966_196614


namespace discounted_price_per_bag_l1966_196650

theorem discounted_price_per_bag
  (cost_per_bag : ℝ)
  (num_bags : ℕ)
  (initial_price : ℝ)
  (num_sold_initial : ℕ)
  (net_profit : ℝ)
  (discounted_revenue : ℝ)
  (discounted_price : ℝ) :
  cost_per_bag = 3.0 →
  num_bags = 20 →
  initial_price = 6.0 →
  num_sold_initial = 15 →
  net_profit = 50 →
  discounted_revenue = (net_profit + (num_bags * cost_per_bag) - (num_sold_initial * initial_price) ) →
  discounted_price = (discounted_revenue / (num_bags - num_sold_initial)) →
  discounted_price = 4.0 :=
by
  sorry

end discounted_price_per_bag_l1966_196650


namespace third_racer_sent_time_l1966_196663

theorem third_racer_sent_time (a : ℝ) (t t1 : ℝ) :
  t1 = 1.5 * t → 
  (1.25 * a) * (t1 - (1 / 2)) = 1.5 * a * t → 
  t = 5 / 3 → 
  (t1 - t) * 60 = 50 :=
by 
  intro h_t1_eq h_second_eq h_t_value
  rw [h_t1_eq] at h_second_eq
  have t_correct : t = 5 / 3 := h_t_value
  sorry

end third_racer_sent_time_l1966_196663


namespace value_of_w_l1966_196637

theorem value_of_w (x : ℝ) (hx : x + 1/x = 5) : x^2 + (1/x)^2 = 23 :=
by
  sorry

end value_of_w_l1966_196637


namespace max_sequence_is_ten_l1966_196607

noncomputable def max_int_sequence_length : Prop :=
  ∀ (a : ℕ → ℤ), 
    (∀ i : ℕ, a i + a (i+1) + a (i+2) + a (i+3) + a (i+4) > 0) ∧
    (∀ i : ℕ, a i + a (i+1) + a (i+2) + a (i+3) + a (i+4) + a (i+5) + a (i+6) < 0) →
    (∃ n ≤ 10, ∀ i ≥ n, a i = 0)

theorem max_sequence_is_ten : max_int_sequence_length :=
sorry

end max_sequence_is_ten_l1966_196607


namespace contradictory_goldbach_l1966_196692

theorem contradictory_goldbach : ¬ (∀ n : ℕ, 2 < n ∧ Even n → ∃ p q : ℕ, Prime p ∧ Prime q ∧ n = p + q) :=
sorry

end contradictory_goldbach_l1966_196692


namespace trigonometric_identity_l1966_196679

theorem trigonometric_identity (θ : ℝ) (h : 2 * (Real.cos θ) + (Real.sin θ) = 0) :
  Real.cos (2 * θ) + 1/2 * Real.sin (2 * θ) = -1 := 
sorry

end trigonometric_identity_l1966_196679


namespace distance_hyperbola_focus_to_line_l1966_196601

def hyperbola_right_focus : Type := { x : ℝ // x = 3 } -- Right focus is at (3, 0)
def line : Type := { x // x + 2 * (0 : ℝ) - 8 = 0 } -- Represents the line x + 2y - 8 = 0

theorem distance_hyperbola_focus_to_line : Real.sqrt 5 = 
  abs (1 * 3 + 2 * 0 - 8) / Real.sqrt (1^2 + 2^2) := 
by
  sorry

end distance_hyperbola_focus_to_line_l1966_196601


namespace work_efficiency_l1966_196615

theorem work_efficiency (orig_time : ℝ) (new_time : ℝ) (work : ℝ) 
  (h1 : orig_time = 1)
  (h2 : new_time = orig_time * (1 - 0.20))
  (h3 : work = 1) :
  (orig_time / new_time) * 100 = 125 :=
by
  sorry

end work_efficiency_l1966_196615


namespace grape_rate_per_kg_l1966_196684

theorem grape_rate_per_kg (G : ℝ) : 
    (8 * G) + (9 * 55) = 1055 → G = 70 := by
  sorry

end grape_rate_per_kg_l1966_196684


namespace expected_value_is_6_5_l1966_196661

noncomputable def expected_value_12_sided_die : ℚ :=
  (1 / 12) * (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12)

theorem expected_value_is_6_5 : expected_value_12_sided_die = 6.5 := 
by
  sorry

end expected_value_is_6_5_l1966_196661


namespace john_ingrid_combined_weighted_average_tax_rate_l1966_196604

noncomputable def john_employment_income : ℕ := 57000
noncomputable def john_employment_tax_rate : ℚ := 0.30
noncomputable def john_rental_income : ℕ := 11000
noncomputable def john_rental_tax_rate : ℚ := 0.25

noncomputable def ingrid_employment_income : ℕ := 72000
noncomputable def ingrid_employment_tax_rate : ℚ := 0.40
noncomputable def ingrid_investment_income : ℕ := 4500
noncomputable def ingrid_investment_tax_rate : ℚ := 0.15

noncomputable def combined_weighted_average_tax_rate : ℚ :=
  let john_total_tax := john_employment_income * john_employment_tax_rate + john_rental_income * john_rental_tax_rate
  let john_total_income := john_employment_income + john_rental_income
  let ingrid_total_tax := ingrid_employment_income * ingrid_employment_tax_rate + ingrid_investment_income * ingrid_investment_tax_rate
  let ingrid_total_income := ingrid_employment_income + ingrid_investment_income
  let combined_total_tax := john_total_tax + ingrid_total_tax
  let combined_total_income := john_total_income + ingrid_total_income
  (combined_total_tax / combined_total_income) * 100

theorem john_ingrid_combined_weighted_average_tax_rate :
  combined_weighted_average_tax_rate = 34.14 := by
  sorry

end john_ingrid_combined_weighted_average_tax_rate_l1966_196604


namespace second_year_associates_l1966_196685

theorem second_year_associates (total_associates : ℕ) (not_first_year : ℕ) (more_than_two_years : ℕ) 
  (h1 : not_first_year = 60 * total_associates / 100) 
  (h2 : more_than_two_years = 30 * total_associates / 100) :
  not_first_year - more_than_two_years = 30 * total_associates / 100 :=
by
  sorry

end second_year_associates_l1966_196685


namespace henrys_friend_money_l1966_196683

theorem henrys_friend_money (h1 h2 : ℕ) (T : ℕ) (f : ℕ) : h1 = 5 → h2 = 2 → T = 20 → h1 + h2 + f = T → f = 13 :=
by
  intros h1_eq h2_eq T_eq total_eq
  rw [h1_eq, h2_eq, T_eq] at total_eq
  sorry

end henrys_friend_money_l1966_196683


namespace negation_of_P_is_there_exists_x_ge_0_l1966_196658

-- Define the proposition P
def P : Prop := ∀ x : ℝ, x^2 + x - 1 < 0

-- State the theorem of the negation of P
theorem negation_of_P_is_there_exists_x_ge_0 : ¬P ↔ ∃ x : ℝ, x^2 + x - 1 ≥ 0 :=
by sorry

end negation_of_P_is_there_exists_x_ge_0_l1966_196658


namespace time_to_pass_pole_l1966_196600

def train_length : ℕ := 250
def platform_length : ℕ := 1250
def time_to_pass_platform : ℕ := 60

theorem time_to_pass_pole : 
  (train_length + platform_length) / time_to_pass_platform * train_length = 10 :=
by
  sorry

end time_to_pass_pole_l1966_196600


namespace set_intersection_l1966_196626

open Set

universe u

variables {U : Type u} (A B : Set ℝ) (x : ℝ)

def universal_set : Set ℝ := univ
def set_A : Set ℝ := {x | abs x < 1}
def set_B : Set ℝ := {x | x > -1/2}
def complement_B : Set ℝ := {x | x ≤ -1/2}
def intersection : Set ℝ := {x | -1 < x ∧ x ≤ -1/2}

theorem set_intersection :
  (universal_set \ set_B) ∩ set_A = {x | -1 < x ∧ x ≤ -1/2} :=
by 
  -- The actual proof steps would go here
  sorry

end set_intersection_l1966_196626


namespace avg_speed_round_trip_l1966_196659

-- Definitions for the conditions
def speed_P_to_Q : ℝ := 80
def distance (D : ℝ) : ℝ := D
def speed_increase_percentage : ℝ := 0.1
def speed_Q_to_P : ℝ := speed_P_to_Q * (1 + speed_increase_percentage)

-- Average speed calculation function
noncomputable def average_speed (D : ℝ) : ℝ := 
  let total_distance := 2 * D
  let time_P_to_Q := D / speed_P_to_Q
  let time_Q_to_P := D / speed_Q_to_P
  let total_time := time_P_to_Q + time_Q_to_P
  total_distance / total_time

-- Theorem: Average speed for the round trip is 83.81 km/hr
theorem avg_speed_round_trip (D : ℝ) : average_speed D = 83.81 := 
by 
  -- Dummy proof placeholder
  sorry

end avg_speed_round_trip_l1966_196659


namespace value_of_expression_l1966_196608

theorem value_of_expression
  (m n : ℝ)
  (h1 : n = -2 * m + 3) :
  4 * m + 2 * n + 1 = 7 :=
sorry

end value_of_expression_l1966_196608


namespace altitude_of_triangle_l1966_196673

theorem altitude_of_triangle
  (a b c : ℝ)
  (h₁ : a = 13)
  (h₂ : b = 15)
  (h₃ : c = 22)
  (h₄ : a + b > c)
  (h₅ : a + c > b)
  (h₆ : b + c > a) :
  let s := (a + b + c) / 2
  let A := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let h := (2 * A) / c
  h = (30 * Real.sqrt 10) / 11 :=
by
  sorry

end altitude_of_triangle_l1966_196673


namespace intersection_of_P_and_Q_l1966_196653

def P : Set ℝ := {x | 1 ≤ x}
def Q : Set ℝ := {x | x < 2}

theorem intersection_of_P_and_Q : P ∩ Q = {x | 1 ≤ x ∧ x < 2} :=
by
  sorry

end intersection_of_P_and_Q_l1966_196653


namespace max_x_of_conditions_l1966_196606

theorem max_x_of_conditions (x y z : ℝ) (h1 : x + y + z = 6) (h2 : xy + xz + yz = 11) : x ≤ 2 :=
by
  -- Placeholder for the actual proof
  sorry

end max_x_of_conditions_l1966_196606


namespace no_solution_intervals_l1966_196698

theorem no_solution_intervals :
    ¬ ∃ x : ℝ, (2 / 3 < x ∧ x < 4 / 3) ∧ (1 / 5 < x ∧ x < 3 / 5) :=
by
  sorry

end no_solution_intervals_l1966_196698


namespace sandwiches_difference_l1966_196631

theorem sandwiches_difference :
  let monday_lunch := 3
  let monday_dinner := 2 * monday_lunch
  let monday_total := monday_lunch + monday_dinner

  let tuesday_lunch := 4
  let tuesday_dinner := tuesday_lunch / 2
  let tuesday_total := tuesday_lunch + tuesday_dinner

  let wednesday_lunch := 2 * tuesday_lunch
  let wednesday_dinner := 3 * tuesday_lunch
  let wednesday_total := wednesday_lunch + wednesday_dinner

  let total_mw := monday_total + tuesday_total + wednesday_total

  let thursday_lunch := 3 * 2
  let thursday_dinner := 5
  let thursday_total := thursday_lunch + thursday_dinner

  total_mw - thursday_total = 24 :=
by
  sorry

end sandwiches_difference_l1966_196631


namespace percent_non_union_women_l1966_196691

-- Definitions used in the conditions:
def total_employees := 100
def percent_men := 50 / 100
def percent_union := 60 / 100
def percent_union_men := 70 / 100

-- Calculate intermediate values
def num_men := total_employees * percent_men
def num_union := total_employees * percent_union
def num_union_men := num_union * percent_union_men
def num_non_union := total_employees - num_union
def num_non_union_men := num_men - num_union_men
def num_non_union_women := num_non_union - num_non_union_men

-- Statement of the problem in Lean
theorem percent_non_union_women : (num_non_union_women / num_non_union) * 100 = 80 := 
by {
  sorry
}

end percent_non_union_women_l1966_196691


namespace range_of_m_l1966_196682

noncomputable def A := {x : ℝ | x^2 - 3 * x + 2 = 0}
noncomputable def B (m : ℝ) := {x : ℝ | x^2 - m * x + 2 = 0}

theorem range_of_m (m : ℝ) (h : ∀ x, x ∈ B m → x ∈ A) : m = 3 ∨ -2 * Real.sqrt 2 < m ∧ m < 2 * Real.sqrt 2 :=
by
  sorry

end range_of_m_l1966_196682


namespace razorback_tshirts_sold_l1966_196634

variable (T : ℕ) -- Number of t-shirts sold
variable (price_per_tshirt : ℕ := 62) -- Price of each t-shirt
variable (total_revenue : ℕ := 11346) -- Total revenue from t-shirts

theorem razorback_tshirts_sold :
  (price_per_tshirt * T = total_revenue) → T = 183 :=
by
  sorry

end razorback_tshirts_sold_l1966_196634


namespace length_of_chord_l1966_196623

theorem length_of_chord 
  (a : ℝ)
  (h_sym : ∀ (x y : ℝ), (x^2 + y^2 - 2*x + 4*y = 0) → (3*x - a*y - 11 = 0))
  (h_line : 3 * 1 - a * (-2) - 11 = 0)
  (h_midpoint : (1 : ℝ) = (a / 4) ∧ (-1 : ℝ) = (-a / 4)) :
  let r := Real.sqrt 5
  let d := Real.sqrt ((1 - 1)^2 + (-1 + 2)^2)
  (2 * Real.sqrt (r^2 - d^2)) = 4 :=
by {
  -- Variables and assumptions would go here
  sorry
}

end length_of_chord_l1966_196623


namespace triangle_third_side_l1966_196676

theorem triangle_third_side (a b x : ℝ) (h : (a - 3) ^ 2 + |b - 4| = 0) :
  x = 5 ∨ x = Real.sqrt 7 :=
sorry

end triangle_third_side_l1966_196676


namespace regression_line_is_y_eq_x_plus_1_l1966_196628

theorem regression_line_is_y_eq_x_plus_1 :
  let points : List (ℝ × ℝ) := [(1, 2), (2, 3), (3, 4), (4, 5)]
  ∃ (m b : ℝ), (∀ (x y : ℝ), (x, y) ∈ points → y = m * x + b) ∧ m = 1 ∧ b = 1 :=
by
  sorry 

end regression_line_is_y_eq_x_plus_1_l1966_196628


namespace darks_washing_time_l1966_196699

theorem darks_washing_time (x : ℕ) :
  (72 + x + 45) + (50 + 65 + 54) = 344 → x = 58 :=
by
  sorry

end darks_washing_time_l1966_196699


namespace wesley_breenah_ages_l1966_196668

theorem wesley_breenah_ages (w b : ℕ) (h₁ : w = 15) (h₂ : b = 7) (h₃ : w + b = 22) :
  ∃ n : ℕ, 2 * (w + b) = (w + n) + (b + n) := by
  exists 11
  sorry

end wesley_breenah_ages_l1966_196668


namespace new_person_weight_l1966_196632

theorem new_person_weight (W : ℝ) (N : ℝ) (avg_increase : ℝ := 2.5) (replaced_weight : ℝ := 35) :
  (W - replaced_weight + N) = (W + (8 * avg_increase)) → N = 55 := sorry

end new_person_weight_l1966_196632


namespace total_vehicles_l1966_196645

-- Define the conditions
def num_trucks_per_lane := 60
def num_lanes := 4
def total_trucks := num_trucks_per_lane * num_lanes
def num_cars_per_lane := 2 * total_trucks
def total_cars := num_cars_per_lane * num_lanes

-- Prove the total number of vehicles in all lanes
theorem total_vehicles : total_trucks + total_cars = 2160 := by
  sorry

end total_vehicles_l1966_196645


namespace find_phi_monotone_interval_1_monotone_interval_2_l1966_196687

-- Definitions related to the function f
noncomputable def f (x φ a : ℝ) : ℝ :=
  Real.sin (x + φ) + a * Real.cos x

-- Problem Part 1: Given f(π/2) = √2 / 2, find φ
theorem find_phi (a : ℝ) (φ : ℝ) (h : |φ| < Real.pi / 2) (hf : f (π / 2) φ a = Real.sqrt 2 / 2) :
  φ = π / 4 ∨ φ = -π / 4 :=
  sorry

-- Problem Part 2 Condition 1: Given a = √3, φ = -π/3, find the monotonically increasing interval
theorem monotone_interval_1 :
  ∀ k : ℤ, ∀ x : ℝ, 
  ((-5 * π / 6) + 2 * k * π) ≤ x ∧ x ≤ (π / 6 + 2 * k * π) → 
  f x (-π / 3) (Real.sqrt 3) = Real.sin (x + π / 3) :=
  sorry

-- Problem Part 2 Condition 2: Given a = -1, φ = π/6, find the monotonically increasing interval
theorem monotone_interval_2 :
  ∀ k : ℤ, ∀ x : ℝ, 
  ((-π / 3) + 2 * k * π) ≤ x ∧ x ≤ ((2 * π / 3) + 2 * k * π) → 
  f x (π / 6) (-1) = Real.sin (x - π / 6) :=
  sorry

end find_phi_monotone_interval_1_monotone_interval_2_l1966_196687


namespace max_notebooks_no_more_than_11_l1966_196664

noncomputable def maxNotebooks (money : ℕ) (cost_single : ℕ) (cost_pack4 : ℕ) (cost_pack7 : ℕ) (max_pack7 : ℕ) : ℕ :=
if money >= cost_pack7 then
  if (money - cost_pack7) >= cost_pack4 then 7 + 4
  else if (money - cost_pack7) >= cost_single then 7 + 1
  else 7
else if money >= cost_pack4 then
  if (money - cost_pack4) >= cost_pack4 then 4 + 4
  else if (money - cost_pack4) >= cost_single then 4 + 1
  else 4
else
  money / cost_single

theorem max_notebooks_no_more_than_11 :
  maxNotebooks 15 2 6 9 1 = 11 :=
by
  sorry

end max_notebooks_no_more_than_11_l1966_196664


namespace line_slope_l1966_196619

theorem line_slope (x y : ℝ) : 3 * y - (1 / 2) * x = 9 → ∃ m, m = 1 / 6 :=
by
  sorry

end line_slope_l1966_196619


namespace distance_between_trees_l1966_196689

theorem distance_between_trees (yard_length : ℕ) (number_of_trees : ℕ) (number_of_gaps : ℕ)
  (h1 : yard_length = 400) (h2 : number_of_trees = 26) (h3 : number_of_gaps = number_of_trees - 1) :
  yard_length / number_of_gaps = 16 := by
  sorry

end distance_between_trees_l1966_196689


namespace ratio_of_money_spent_on_clothes_is_1_to_2_l1966_196609

-- Definitions based on conditions
def allowance1 : ℕ := 5
def weeks1 : ℕ := 8
def allowance2 : ℕ := 6
def weeks2 : ℕ := 6
def cost_video : ℕ := 35
def remaining_money : ℕ := 3

-- Calculations
def total_saved : ℕ := (allowance1 * weeks1) + (allowance2 * weeks2)
def total_expended : ℕ := cost_video + remaining_money
def spent_on_clothes : ℕ := total_saved - total_expended

-- Prove the ratio of money spent on clothes to the total money saved is 1:2
theorem ratio_of_money_spent_on_clothes_is_1_to_2 :
  (spent_on_clothes : ℚ) / (total_saved : ℚ) = 1 / 2 :=
by
  sorry

end ratio_of_money_spent_on_clothes_is_1_to_2_l1966_196609


namespace tax_percentage_first_40000_l1966_196686

theorem tax_percentage_first_40000 (P : ℝ) :
  (0 < P) → 
  (P / 100) * 40000 + 0.20 * 10000 = 8000 →
  P = 15 :=
by
  intros hP h
  sorry

end tax_percentage_first_40000_l1966_196686


namespace initial_milk_amount_l1966_196638

theorem initial_milk_amount (M : ℝ) (H1 : 0.05 * M = 0.02 * (M + 15)) : M = 10 :=
by
  sorry

end initial_milk_amount_l1966_196638


namespace movie_theater_loss_l1966_196635

theorem movie_theater_loss :
  let capacity := 50
  let ticket_price := 8.0
  let tickets_sold := 24
  (capacity * ticket_price - tickets_sold * ticket_price) = 208 := by
  sorry

end movie_theater_loss_l1966_196635


namespace NaNO3_moles_l1966_196602

theorem NaNO3_moles (moles_NaCl moles_HNO3 moles_NaNO3 : ℝ) (h_HNO3 : moles_HNO3 = 2) (h_ratio : moles_NaNO3 = moles_NaCl) (h_NaNO3 : moles_NaNO3 = 2) :
  moles_NaNO3 = 2 :=
sorry

end NaNO3_moles_l1966_196602


namespace gpa_at_least_3_5_l1966_196629

noncomputable def prob_gpa_at_least_3_5 : ℚ :=
  let p_A_eng := 1 / 3
  let p_B_eng := 1 / 5
  let p_C_eng := 7 / 15 -- 1 - p_A_eng - p_B_eng
  
  let p_A_hist := 1 / 5
  let p_B_hist := 1 / 4
  let p_C_hist := 11 / 20 -- 1 - p_A_hist - p_B_hist

  let prob_two_As := p_A_eng * p_A_hist
  let prob_A_eng_B_hist := p_A_eng * p_B_hist
  let prob_A_hist_B_eng := p_A_hist * p_B_eng
  let prob_two_Bs := p_B_eng * p_B_hist

  let total_prob := prob_two_As + prob_A_eng_B_hist + prob_A_hist_B_eng + prob_two_Bs
  total_prob

theorem gpa_at_least_3_5 : prob_gpa_at_least_3_5 = 6 / 25 := by {
  sorry
}

end gpa_at_least_3_5_l1966_196629


namespace circle_equation_l1966_196646

theorem circle_equation (x y : ℝ) :
  let center := (0, 4)
  let point_on_circle := (3, 0)
  (x - center.1)^2 + (y - center.2)^2 = 25 :=
by
  sorry

end circle_equation_l1966_196646


namespace saline_solution_mixture_l1966_196616

theorem saline_solution_mixture 
  (x : ℝ) 
  (h₁ : 20 + 0.1 * x = 0.25 * (50 + x)) 
  : x = 50 := 
by 
  sorry

end saline_solution_mixture_l1966_196616


namespace expression_eval_l1966_196622

theorem expression_eval : (-4)^7 / 4^5 + 5^3 * 2 - 7^2 = 185 := by
  sorry

end expression_eval_l1966_196622


namespace james_nickels_l1966_196647

theorem james_nickels (p n : ℕ) (h₁ : p + n = 50) (h₂ : p + 5 * n = 150) : n = 25 :=
by
  -- Skipping the proof since only the statement is required
  sorry

end james_nickels_l1966_196647


namespace pentagon_inequality_l1966_196667

-- Definitions
variables {S R1 R2 R3 R4 R5 : ℝ}
noncomputable def sine108 := Real.sin (108 * Real.pi / 180)

-- Theorem statement
theorem pentagon_inequality (h_area : S > 0) (h_radii : R1 > 0 ∧ R2 > 0 ∧ R3 > 0 ∧ R4 > 0 ∧ R5 > 0) :
  R1^4 + R2^4 + R3^4 + R4^4 + R5^4 ≥ (4 / (5 * sine108^2)) * S^2 :=
by
  sorry

end pentagon_inequality_l1966_196667


namespace solve_for_x_l1966_196656

theorem solve_for_x (x : ℝ) (h : 3 - 1 / (1 - x) = 2 * (1 / (1 - x))) : x = 0 :=
by
  sorry

end solve_for_x_l1966_196656


namespace division_addition_l1966_196652

theorem division_addition :
  (-150 + 50) / (-50) = 2 := by
  sorry

end division_addition_l1966_196652


namespace solve_cubic_root_eq_l1966_196630

theorem solve_cubic_root_eq (x : ℝ) (h : (5 - x)^(1/3) = 4) : x = -59 := 
by
  sorry

end solve_cubic_root_eq_l1966_196630


namespace fraction_of_price_l1966_196690

theorem fraction_of_price (d : ℝ) : d * 0.65 * 0.70 = d * 0.455 :=
by
  sorry

end fraction_of_price_l1966_196690


namespace simplify_expression_l1966_196688

variable (x y : ℝ)

theorem simplify_expression :
  (2 * x + 3 * y) ^ 2 - 2 * x * (2 * x - 3 * y) = 18 * x * y + 9 * y ^ 2 :=
by
  sorry

end simplify_expression_l1966_196688


namespace quadratic_has_sum_r_s_l1966_196674

/-
  Define the quadratic equation 6x^2 - 24x - 54 = 0
-/
def quadratic_eq (x : ℝ) : Prop :=
  6 * x^2 - 24 * x - 54 = 0

/-
  Define the value 11 which is the sum r + s when completing the square
  for the above quadratic equation  
-/
def result_value := -2 + 13

/-
  State the proof that r + s = 11 given the quadratic equation.
-/
theorem quadratic_has_sum_r_s : ∀ x : ℝ, quadratic_eq x → -2 + 13 = 11 :=
by
  intros
  exact rfl

end quadratic_has_sum_r_s_l1966_196674


namespace remainder_of_7_pow_7_pow_7_pow_7_mod_500_l1966_196633

theorem remainder_of_7_pow_7_pow_7_pow_7_mod_500 :
    (7 ^ (7 ^ (7 ^ 7))) % 500 = 343 := 
sorry

end remainder_of_7_pow_7_pow_7_pow_7_mod_500_l1966_196633


namespace cost_of_12_cheaper_fruits_l1966_196675

-- Defining the price per 10 apples in cents.
def price_per_10_apples : ℕ := 200

-- Defining the price per 5 oranges in cents.
def price_per_5_oranges : ℕ := 150

-- No bulk discount means per item price is just total cost divided by the number of items
def price_per_apple := price_per_10_apples / 10
def price_per_orange := price_per_5_oranges / 5

-- Given the calculation steps, we have to prove that the cost for 12 cheaper fruits (apples) is 240
theorem cost_of_12_cheaper_fruits : 12 * price_per_apple = 240 := by
  -- This step performs the proof, which we skip with sorry
  sorry

end cost_of_12_cheaper_fruits_l1966_196675


namespace div_by_3_implies_one_div_by_3_l1966_196662

theorem div_by_3_implies_one_div_by_3 (a b : ℕ) (h_ab : 3 ∣ (a * b)) (h_na : ¬ 3 ∣ a) (h_nb : ¬ 3 ∣ b) : false :=
sorry

end div_by_3_implies_one_div_by_3_l1966_196662


namespace money_left_after_shopping_l1966_196660

-- Definitions based on conditions
def initial_amount : ℝ := 5000
def percentage_spent : ℝ := 0.30
def amount_spent : ℝ := percentage_spent * initial_amount
def remaining_amount : ℝ := initial_amount - amount_spent

-- Theorem statement based on the question and correct answer
theorem money_left_after_shopping : remaining_amount = 3500 :=
by
  sorry

end money_left_after_shopping_l1966_196660


namespace find_angle_ACB_l1966_196665

theorem find_angle_ACB
    (convex_quadrilateral : Prop)
    (angle_BAC : ℝ)
    (angle_CAD : ℝ)
    (angle_ADB : ℝ)
    (angle_BDC : ℝ)
    (h1 : convex_quadrilateral)
    (h2 : angle_BAC = 20)
    (h3 : angle_CAD = 60)
    (h4 : angle_ADB = 50)
    (h5 : angle_BDC = 10)
    : ∃ angle_ACB : ℝ, angle_ACB = 80 :=
by
  -- Here use sorry to skip the proof.
  sorry

end find_angle_ACB_l1966_196665


namespace arithmetic_sequence_solution_l1966_196618

noncomputable def arithmetic_sequence (a : ℕ → ℤ) (a1 d : ℤ) : Prop :=
∀ n : ℕ, a n = a1 + n * d

noncomputable def S (a : ℕ → ℤ) (n : ℕ) : ℤ :=
n * a 0 + (n * (n - 1) / 2) * (a 1 - a 0)

theorem arithmetic_sequence_solution :
  ∃ d : ℤ,
  (∀ n : ℕ, n > 0 ∧ n < 10 → a n = 23 + n * d) ∧
  (23 + 5 * d > 0) ∧
  (23 + 6 * d < 0) ∧
  d = -4 ∧
  S a 6 = 78 ∧
  ∀ n : ℕ, S a n > 0 → n ≤ 12 :=
by
  sorry

end arithmetic_sequence_solution_l1966_196618


namespace sqrt_floor_squared_l1966_196657

theorem sqrt_floor_squared (h1 : 7^2 = 49) (h2 : 8^2 = 64) (h3 : 7 < Real.sqrt 50) (h4 : Real.sqrt 50 < 8) : (Int.floor (Real.sqrt 50))^2 = 49 :=
by
  sorry

end sqrt_floor_squared_l1966_196657


namespace find_a_l1966_196605

theorem find_a (a : ℝ) (α : ℝ) (P : ℝ × ℝ) 
  (h_P : P = (3 * a, 4)) 
  (h_cos : Real.cos α = -3/5) : 
  a = -1 := 
by
  sorry

end find_a_l1966_196605


namespace brother_15th_birthday_day_of_week_carlos_age_on_brothers_15th_birthday_l1966_196636

def march_13_2007_day_of_week : String := "Tuesday"

def days_until_brothers_birthday : Nat := 2000

def start_date := (2007, 3, 13)  -- (year, month, day)

def days_per_week := 7

def carlos_initial_age := 7

def day_of_week_after_n_days (start_day : String) (n : Nat) : String :=
  match n % 7 with
  | 0 => "Tuesday"
  | 1 => "Wednesday"
  | 2 => "Thursday"
  | 3 => "Friday"
  | 4 => "Saturday"
  | 5 => "Sunday"
  | 6 => "Monday"
  | _ => "Unknown" -- This case should never happen

def carlos_age_after_n_days (initial_age : Nat) (n : Nat) : Nat :=
  initial_age + n / 365

theorem brother_15th_birthday_day_of_week : 
  day_of_week_after_n_days march_13_2007_day_of_week days_until_brothers_birthday = "Sunday" := 
by sorry

theorem carlos_age_on_brothers_15th_birthday :
  carlos_age_after_n_days carlos_initial_age days_until_brothers_birthday = 12 :=
by sorry

end brother_15th_birthday_day_of_week_carlos_age_on_brothers_15th_birthday_l1966_196636


namespace ratio_of_area_of_inscribed_circle_to_triangle_l1966_196617

theorem ratio_of_area_of_inscribed_circle_to_triangle (h r : ℝ) (h_pos : 0 < h) (r_pos : 0 < r) :
  let a := (3 / 5) * h
  let b := (4 / 5) * h
  let A := (1 / 2) * a * b
  let s := (a + b + h) / 2
  (π * r) / s = (5 * π * r) / (12 * h) :=
by
  let a := (3 / 5) * h
  let b := (4 / 5) * h
  let A := (1 / 2) * a * b
  let s := (a + b + h) / 2
  sorry

end ratio_of_area_of_inscribed_circle_to_triangle_l1966_196617


namespace surface_area_of_cube_l1966_196697

theorem surface_area_of_cube (a : ℝ) : 
  let edge_length := 4 * a
  let face_area := edge_length ^ 2
  let total_surface_area := 6 * face_area
  total_surface_area = 96 * a^2 := by
  sorry

end surface_area_of_cube_l1966_196697


namespace carrots_total_l1966_196643

theorem carrots_total 
  (picked_1 : Nat) 
  (thrown_out : Nat) 
  (picked_2 : Nat) 
  (total_carrots : Nat) 
  (h_picked1 : picked_1 = 23) 
  (h_thrown_out : thrown_out = 10) 
  (h_picked2 : picked_2 = 47) : 
  total_carrots = 60 := 
by
  sorry

end carrots_total_l1966_196643


namespace distance_of_point_P_to_base_AB_l1966_196672

theorem distance_of_point_P_to_base_AB :
  ∀ (P : ℝ) (A B C : ℝ → ℝ)
    (h : ∀ (x : ℝ), A x = B x)
    (altitude : ℝ)
    (area_ratio : ℝ),
  altitude = 6 →
  area_ratio = 1 / 3 →
  (∃ d : ℝ, d = 6 - (2 / 3) * 6 ∧ d = 2) := 
  sorry

end distance_of_point_P_to_base_AB_l1966_196672


namespace solve_star_op_eq_l1966_196695

def star_op (a b : ℕ) : ℕ :=
  if a < b then b * b else b * b * b

theorem solve_star_op_eq :
  ∃ x : ℕ, 5 * star_op 5 x = 64 ∧ (x = 4 ∨ x = 8) :=
sorry

end solve_star_op_eq_l1966_196695


namespace points_on_parabola_l1966_196681

theorem points_on_parabola (s : ℝ) : ∃ (u v : ℝ), u = 3^s - 4 ∧ v = 9^s - 7 * 3^s - 2 ∧ v = u^2 + u - 14 :=
by
  sorry

end points_on_parabola_l1966_196681


namespace fraction_of_area_above_line_l1966_196693

theorem fraction_of_area_above_line :
  let A := (3, 2)
  let B := (6, 0)
  let side_length := B.fst - A.fst
  let square_area := side_length ^ 2
  let triangle_base := B.fst - A.fst
  let triangle_height := A.snd
  let triangle_area := (1 / 2 : ℚ) * triangle_base * triangle_height
  let area_above_line := square_area - triangle_area
  let fraction_above_line := area_above_line / square_area
  fraction_above_line = (2 / 3 : ℚ) :=
by
  sorry

end fraction_of_area_above_line_l1966_196693


namespace circular_garden_radius_l1966_196640

theorem circular_garden_radius (r : ℝ) (h : 2 * Real.pi * r = (1 / 8) * Real.pi * r^2) : r = 16 :=
sorry

end circular_garden_radius_l1966_196640


namespace unused_sector_angle_l1966_196677

noncomputable def cone_volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h
noncomputable def slant_height (r h : ℝ) : ℝ := Real.sqrt (r^2 + h^2)
noncomputable def central_angle (r base_circumference : ℝ) : ℝ := (base_circumference / (2 * Real.pi * r)) * 360
noncomputable def unused_angle (total_degrees used_angle : ℝ) : ℝ := total_degrees - used_angle

theorem unused_sector_angle (R : ℝ)
  (cone_radius := 15)
  (cone_volume := 675 * Real.pi)
  (total_circumference := 2 * Real.pi * R)
  (cone_height := 9)
  (slant_height := Real.sqrt (cone_radius^2 + cone_height^2))
  (base_circumference := 2 * Real.pi * cone_radius)
  (used_angle := central_angle slant_height base_circumference) :

  unused_angle 360 used_angle = 164.66 := by
  sorry

end unused_sector_angle_l1966_196677


namespace basketball_competition_l1966_196624

theorem basketball_competition:
  (∃ x : ℕ, (0 ≤ x) ∧ (x ≤ 12) ∧ (3 * x - (12 - x) ≥ 28)) := by
  sorry

end basketball_competition_l1966_196624


namespace cost_percentage_l1966_196603

-- Define the original and new costs
def original_cost (t b : ℝ) : ℝ := t * b^4
def new_cost (t b : ℝ) : ℝ := t * (2 * b)^4

-- Define the theorem to prove the percentage relationship
theorem cost_percentage (t b : ℝ) (C R : ℝ) (h1 : C = original_cost t b) (h2 : R = new_cost t b) :
  (R / C) * 100 = 1600 :=
by sorry

end cost_percentage_l1966_196603


namespace smallest_positive_period_l1966_196613

theorem smallest_positive_period :
  ∀ (x : ℝ), 5 * Real.sin ((π / 6) - (π / 3) * x) = 5 * Real.sin ((π / 6) - (π / 3) * (x + 6)) :=
by
  sorry

end smallest_positive_period_l1966_196613


namespace smallest_integer_condition_l1966_196696

theorem smallest_integer_condition (p d n x : ℕ) (h1 : 1 ≤ d ∧ d ≤ 9) 
  (h2 : x = 10^p * d + n) (h3 : x = 19 * n) : 
  x = 95 := by
  sorry

end smallest_integer_condition_l1966_196696


namespace usual_time_56_l1966_196621

theorem usual_time_56 (S : ℝ) (T : ℝ) (h : (T + 24) * S = T * (0.7 * S)) : T = 56 :=
by sorry

end usual_time_56_l1966_196621


namespace age_ratio_in_1_year_l1966_196625

variable (j m x : ℕ)

-- Conditions
def condition1 (j m : ℕ) : Prop :=
  j - 3 = 2 * (m - 3)

def condition2 (j m : ℕ) : Prop :=
  j - 5 = 3 * (m - 5)

-- Question
def age_ratio (j m x : ℕ) : Prop :=
  (j + x) * 2 = 3 * (m + x)

theorem age_ratio_in_1_year (j m x : ℕ) :
  condition1 j m → condition2 j m → age_ratio j m 1 :=
by
  sorry

end age_ratio_in_1_year_l1966_196625
