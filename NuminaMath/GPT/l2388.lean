import Mathlib

namespace range_of_m_l2388_238896

noncomputable def f (x m : ℝ) : ℝ := x^2 - x + m * (2 * x + 1)

theorem range_of_m (m : ℝ) : (∀ x > 1, 0 < 2 * x + (2 * m - 1)) ↔ (m ≥ -1/2) := by
  sorry

end range_of_m_l2388_238896


namespace math_problem_proof_l2388_238801

-- Define the problem statement
def problem_expr : ℕ :=
  28 * 7 * 25 + 12 * 7 * 25 + 7 * 11 * 3 + 44

-- Prove the problem statement equals to the correct answer
theorem math_problem_proof : problem_expr = 7275 := by
  sorry

end math_problem_proof_l2388_238801


namespace train_speed_is_252_144_l2388_238861

/-- Train and pedestrian problem setup -/
noncomputable def train_speed (train_length : ℕ) (cross_time : ℕ) (man_speed_kmph : ℕ) : ℝ :=
  let man_speed_mps := (man_speed_kmph : ℝ) * 1000 / 3600
  let relative_speed_mps := (train_length : ℝ) / (cross_time : ℝ)
  let train_speed_mps := relative_speed_mps - man_speed_mps
  train_speed_mps * 3600 / 1000

theorem train_speed_is_252_144 :
  train_speed 500 7 5 = 252.144 := by
  sorry

end train_speed_is_252_144_l2388_238861


namespace yella_computer_usage_difference_l2388_238802

-- Define the given conditions
def last_week_usage : ℕ := 91
def this_week_daily_usage : ℕ := 8
def days_in_week : ℕ := 7

-- Compute this week's total usage
def this_week_total_usage := this_week_daily_usage * days_in_week

-- Statement to prove
theorem yella_computer_usage_difference :
  last_week_usage - this_week_total_usage = 35 := 
by
  -- The proof will be filled in here
  sorry

end yella_computer_usage_difference_l2388_238802


namespace romance_movie_tickets_l2388_238894

-- Define the given conditions.
def horror_movie_tickets := 93
def relationship (R : ℕ) := 3 * R + 18 = horror_movie_tickets

-- The theorem we need to prove
theorem romance_movie_tickets (R : ℕ) (h : relationship R) : R = 25 :=
by sorry

end romance_movie_tickets_l2388_238894


namespace minimum_value_expression_l2388_238836

theorem minimum_value_expression (a b c : ℤ) (h : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
    3 * a^2 + 2 * b^2 + 4 * c^2 - a * b - 3 * b * c - 5 * c * a ≥ 6 :=
sorry

end minimum_value_expression_l2388_238836


namespace coin_flip_probability_l2388_238870

def total_outcomes := 2^6
def favorable_outcomes := 2^3
def probability := favorable_outcomes / total_outcomes

theorem coin_flip_probability :
  probability = 1 / 8 :=
by
  unfold probability total_outcomes favorable_outcomes
  sorry

end coin_flip_probability_l2388_238870


namespace total_winter_clothing_l2388_238814

def first_box_items : Nat := 3 + 5 + 2
def second_box_items : Nat := 4 + 3 + 1
def third_box_items : Nat := 2 + 6 + 3
def fourth_box_items : Nat := 1 + 7 + 2

theorem total_winter_clothing : first_box_items + second_box_items + third_box_items + fourth_box_items = 39 := by
  sorry

end total_winter_clothing_l2388_238814


namespace robin_candy_consumption_l2388_238841

theorem robin_candy_consumption (x : ℕ) : 23 - x + 21 = 37 → x = 7 :=
by
  intros h
  sorry

end robin_candy_consumption_l2388_238841


namespace uncle_wang_withdraw_amount_l2388_238834

noncomputable def total_amount (principal : ℕ) (rate : ℚ) (time : ℕ) : ℚ :=
  principal + principal * rate * time

theorem uncle_wang_withdraw_amount :
  total_amount 100000 (315/10000) 2 = 106300 := by
  sorry

end uncle_wang_withdraw_amount_l2388_238834


namespace smallest_EF_minus_DE_l2388_238803

theorem smallest_EF_minus_DE (x y z : ℕ) (h1 : x < y) (h2 : y ≤ z) (h3 : x + y + z = 2050)
  (h4 : x + y > z) (h5 : y + z > x) (h6 : z + x > y) : y - x = 1 :=
by
  sorry

end smallest_EF_minus_DE_l2388_238803


namespace sales_tax_difference_l2388_238817

theorem sales_tax_difference :
  let price : ℝ := 50
  let tax_rate1 : ℝ := 0.075
  let tax_rate2 : ℝ := 0.07
  (price * tax_rate1) - (price * tax_rate2) = 0.25 := by
  sorry

end sales_tax_difference_l2388_238817


namespace alice_score_l2388_238866

variables (correct_answers wrong_answers unanswered_questions : ℕ)
variables (points_correct points_incorrect : ℚ)

def compute_score (correct_answers wrong_answers : ℕ) (points_correct points_incorrect : ℚ) : ℚ :=
    (correct_answers : ℚ) * points_correct + (wrong_answers : ℚ) * points_incorrect

theorem alice_score : 
    correct_answers = 15 → 
    wrong_answers = 5 → 
    unanswered_questions = 10 → 
    points_correct = 1 → 
    points_incorrect = -0.25 → 
    compute_score 15 5 1 (-0.25) = 13.75 := 
by intros; sorry

end alice_score_l2388_238866


namespace matching_charge_and_minutes_l2388_238829

def charge_at_time (x : ℕ) : ℕ :=
  100 - x / 6

def minutes_past_midnight (x : ℕ) : ℕ :=
  x % 60

theorem matching_charge_and_minutes :
  ∃ x, (x = 292 ∨ x = 343 ∨ x = 395 ∨ x = 446 ∨ x = 549) ∧ 
       charge_at_time x = minutes_past_midnight x :=
by {
  sorry
}

end matching_charge_and_minutes_l2388_238829


namespace number_of_A_items_number_of_A_proof_l2388_238851

def total_items : ℕ := 600
def ratio_A_B_C := (1, 2, 3)
def selected_items : ℕ := 120

theorem number_of_A_items (total_items : ℕ) (selected_items : ℕ) (rA rB rC : ℕ) (ratio_proof : rA + rB + rC = 6) : ℕ :=
  let total_ratio := rA + rB + rC
  let A_ratio := rA
  (selected_items * A_ratio) / total_ratio

theorem number_of_A_proof : number_of_A_items total_items selected_items 1 2 3 (rfl) = 20 := by
  sorry

end number_of_A_items_number_of_A_proof_l2388_238851


namespace circumradius_relation_l2388_238804

-- Definitions of the geometric constructs from the problem
open EuclideanGeometry

noncomputable def circumradius (A B C : Point) : Real := sorry

-- Given conditions
def angle_bisectors_intersect_at_point (A B C B1 C1 I : Point) : Prop := sorry
def line_intersects_circumcircle_at_points (B1 C1 : Point) (circumcircle : Circle) (M N : Point) : Prop := sorry

-- Main statement to prove
theorem circumradius_relation
  (A B C B1 C1 I M N : Point)
  (circumcircle : Circle)
  (h1 : angle_bisectors_intersect_at_point A B C B1 C1 I)
  (h2 : line_intersects_circumcircle_at_points B1 C1 circumcircle M N) :
  circumradius M I N = 2 * circumradius A B C :=
sorry

end circumradius_relation_l2388_238804


namespace reflect_over_x_axis_reflect_over_y_axis_l2388_238874

-- Mathematical Definitions
def Point := (ℝ × ℝ)

-- Reflect a point over the x-axis
def reflectOverX (M : Point) : Point :=
  (M.1, -M.2)

-- Reflect a point over the y-axis
def reflectOverY (M : Point) : Point :=
  (-M.1, M.2)

-- Theorem statements
theorem reflect_over_x_axis (M : Point) : reflectOverX M = (M.1, -M.2) :=
by
  sorry

theorem reflect_over_y_axis (M : Point) : reflectOverY M = (-M.1, M.2) :=
by
  sorry

end reflect_over_x_axis_reflect_over_y_axis_l2388_238874


namespace car_travel_distance_l2388_238849

-- Definitions of conditions
def speed_kmph : ℝ := 27 -- 27 kilometers per hour
def time_sec : ℝ := 50 -- 50 seconds

-- Equivalent in Lean 4 for car moving distance in meters
theorem car_travel_distance : (speed_kmph * 1000 / 3600) * time_sec = 375 := by
  sorry

end car_travel_distance_l2388_238849


namespace percentage_of_8thgraders_correct_l2388_238873

def total_students_oakwood : ℕ := 150
def total_students_pinecrest : ℕ := 250

def percent_8thgraders_oakwood : ℕ := 60
def percent_8thgraders_pinecrest : ℕ := 55

def number_of_8thgraders_oakwood : ℚ := (percent_8thgraders_oakwood * total_students_oakwood) / 100
def number_of_8thgraders_pinecrest : ℚ := (percent_8thgraders_pinecrest * total_students_pinecrest) / 100

def total_number_of_8thgraders : ℚ := number_of_8thgraders_oakwood + number_of_8thgraders_pinecrest
def total_number_of_students : ℕ := total_students_oakwood + total_students_pinecrest

def percent_8thgraders_combined : ℚ := (total_number_of_8thgraders / total_number_of_students) * 100

theorem percentage_of_8thgraders_correct : percent_8thgraders_combined = 57 := 
by
  sorry

end percentage_of_8thgraders_correct_l2388_238873


namespace Jayden_less_Coraline_l2388_238800

variables (M J : ℕ)
def Coraline_number := 80
def total_sum := 180

theorem Jayden_less_Coraline
  (h1 : M = J + 20)
  (h2 : J < Coraline_number)
  (h3 : M + J + Coraline_number = total_sum) :
  Coraline_number - J = 40 := by
  sorry

end Jayden_less_Coraline_l2388_238800


namespace value_of_y_l2388_238882

theorem value_of_y : exists y : ℝ, (∀ k : ℝ, (∀ x y : ℝ, x = k / y^2 → (x = 1 → y = 2 → k = 4)) ∧ (x = 0.1111111111111111 → k = 4 → y = 6)) := by
  sorry

end value_of_y_l2388_238882


namespace total_earnings_l2388_238878

def oil_change_cost : ℕ := 20
def repair_cost : ℕ := 30
def car_wash_cost : ℕ := 5

def num_oil_changes : ℕ := 5
def num_repairs : ℕ := 10
def num_car_washes : ℕ := 15

theorem total_earnings :
  (num_oil_changes * oil_change_cost) +
  (num_repairs * repair_cost) +
  (num_car_washes * car_wash_cost) = 475 :=
by
  sorry

end total_earnings_l2388_238878


namespace equation_three_no_real_roots_l2388_238880

theorem equation_three_no_real_roots
  (a₁ a₂ a₃ : ℝ)
  (h₁ : a₁^2 - 4 ≥ 0)
  (h₂ : a₂^2 - 8 < 0)
  (h₃ : a₂^2 = a₁ * a₃) :
  a₃^2 - 16 < 0 :=
sorry

end equation_three_no_real_roots_l2388_238880


namespace max_y_value_l2388_238808

theorem max_y_value (x : ℝ) : ∃ y : ℝ, y = -x^2 + 4 * x + 3 ∧ y ≤ 7 :=
by
  sorry

end max_y_value_l2388_238808


namespace evaluate_fraction_l2388_238837

theorem evaluate_fraction (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a - b * (1 / a) ≠ 0) :
  (a^2 - 1 / b^2) / (b^2 - 1 / a^2) = a^2 / b^2 :=
by
  sorry

end evaluate_fraction_l2388_238837


namespace total_distance_traveled_l2388_238898

/--
A spider is on the edge of a ceiling of a circular room with a radius of 65 feet. 
The spider walks straight across the ceiling to the opposite edge, passing through 
the center. It then walks straight to another point on the edge of the circle but 
not back through the center. The third part of the journey is straight back to the 
original starting point. If the third part of the journey was 90 feet long, then 
the total distance traveled by the spider is 313.81 feet.
-/
theorem total_distance_traveled (r : ℝ) (d1 d2 d3 : ℝ) (h1 : r = 65) (h2 : d1 = 2 * r) (h3 : d3 = 90) :
  d1 + d2 + d3 = 313.81 :=
by
  sorry

end total_distance_traveled_l2388_238898


namespace probability_of_B_given_A_l2388_238824

noncomputable def balls_in_box : Prop :=
  let total_balls := 12
  let yellow_balls := 5
  let blue_balls := 4
  let green_balls := 3
  let event_A := (yellow_balls * green_balls + yellow_balls * blue_balls + green_balls * blue_balls) / (total_balls * (total_balls - 1) / 2)
  let event_B := (yellow_balls * blue_balls) / (total_balls * (total_balls - 1) / 2)
  (event_B / event_A) = 20 / 47

theorem probability_of_B_given_A : balls_in_box := sorry

end probability_of_B_given_A_l2388_238824


namespace average_percent_increase_in_profit_per_car_l2388_238854

theorem average_percent_increase_in_profit_per_car
  (N P : ℝ) -- N: Number of cars sold last year, P: Profit per car last year
  (HP1 : N > 0) -- Non-zero number of cars
  (HP2 : P > 0) -- Non-zero profit
  (HProfitIncrease : 1.3 * (N * P) = 1.3 * N * P) -- Total profit increased by 30%
  (HCarDecrease : 0.7 * N = 0.7 * N) -- Number of cars decreased by 30%
  : ((1.3 / 0.7) - 1) * 100 = 85.7 := sorry

end average_percent_increase_in_profit_per_car_l2388_238854


namespace expression_equals_thirteen_l2388_238859

-- Define the expression
def expression : ℤ :=
    8 + 15 / 3 - 4 * 2 + Nat.pow 2 3

-- State the theorem that proves the value of the expression
theorem expression_equals_thirteen : expression = 13 :=
by
  sorry

end expression_equals_thirteen_l2388_238859


namespace cards_value_1_count_l2388_238811

/-- There are 4 different suits in a deck of cards containing a total of 52 cards.
  Each suit has 13 cards numbered from 1 to 13.
  Feifei draws 2 hearts, 3 spades, 4 diamonds, and 5 clubs.
  The sum of the face values of these 14 cards is exactly 35.
  Prove that 4 of these cards have a face value of 1. -/
theorem cards_value_1_count :
  ∃ (hearts spades diamonds clubs : List ℕ),
  hearts.length = 2 ∧ spades.length = 3 ∧ diamonds.length = 4 ∧ clubs.length = 5 ∧
  (∀ v, v ∈ hearts → v ∈ List.range 13) ∧ 
  (∀ v, v ∈ spades → v ∈ List.range 13) ∧
  (∀ v, v ∈ diamonds → v ∈ List.range 13) ∧
  (∀ v, v ∈ clubs → v ∈ List.range 13) ∧
  (hearts.sum + spades.sum + diamonds.sum + clubs.sum = 35) ∧
  ((hearts ++ spades ++ diamonds ++ clubs).count 1 = 4) := sorry

end cards_value_1_count_l2388_238811


namespace value_of_sum_l2388_238890

theorem value_of_sum (x y z : ℝ) 
    (h1 : x + 2*y + 3*z = 10) 
    (h2 : 4*x + 3*y + 2*z = 15) : 
    x + y + z = 5 :=
by
    sorry

end value_of_sum_l2388_238890


namespace intersection_of_lines_l2388_238864

theorem intersection_of_lines : 
  (∃ x y : ℚ, y = -3 * x + 1 ∧ y = 5 * x + 4) ↔ 
  (∃ x y : ℚ, x = -3 / 8 ∧ y = 17 / 8) :=
by
  sorry

end intersection_of_lines_l2388_238864


namespace value_of_polynomial_l2388_238879

theorem value_of_polynomial :
  98^3 + 3 * (98^2) + 3 * 98 + 1 = 970299 :=
by sorry

end value_of_polynomial_l2388_238879


namespace val_of_7c_plus_7d_l2388_238897

noncomputable def h (x : ℝ) : ℝ := 7 * x - 6

noncomputable def f_inv (x : ℝ) : ℝ := 7 * x - 4

noncomputable def f (c d x : ℝ) : ℝ := c * x + d

theorem val_of_7c_plus_7d (c d : ℝ) (h_eq : ∀ x, h x = f_inv x - 2) 
  (inv_prop : ∀ x, f c d (f_inv x) = x) : 7 * c + 7 * d = 5 :=
by
  sorry

end val_of_7c_plus_7d_l2388_238897


namespace min_value_of_fractions_l2388_238827

theorem min_value_of_fractions (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
    (a+b)/(c+d) + (a+c)/(b+d) + (a+d)/(b+c) + (b+c)/(a+d) + (b+d)/(a+c) + (c+d)/(a+b) ≥ 6 :=
by
  sorry

end min_value_of_fractions_l2388_238827


namespace problem1_problem2_problem3_problem4_l2388_238888

-- Problem 1
theorem problem1 (a : ℝ) : -2 * a^3 * 3 * a^2 = -6 * a^5 := 
by
  sorry

-- Problem 2
theorem problem2 (m : ℝ) : m^4 * (m^2)^3 / m^8 = m^2 := 
by
  sorry

-- Problem 3
theorem problem3 (x : ℝ) : (-2 * x - 1) * (2 * x - 1) = 1 - 4 * x^2 := 
by
  sorry

-- Problem 4
theorem problem4 (x : ℝ) : (-3 * x + 2)^2 = 9 * x^2 - 12 * x + 4 := 
by
  sorry

end problem1_problem2_problem3_problem4_l2388_238888


namespace percentage_of_invalid_votes_calculation_l2388_238863

theorem percentage_of_invalid_votes_calculation
  (total_votes_poled : ℕ)
  (valid_votes_B : ℕ)
  (additional_percent_votes_A : ℝ)
  (Vb : ℝ)
  (total_valid_votes : ℝ)
  (P : ℝ) :
  total_votes_poled = 8720 →
  valid_votes_B = 2834 →
  additional_percent_votes_A = 0.15 →
  Vb = valid_votes_B →
  total_valid_votes = (2 * Vb) + (additional_percent_votes_A * total_votes_poled) →
  total_valid_votes / total_votes_poled = 1 - P/100 →
  P = 20 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end percentage_of_invalid_votes_calculation_l2388_238863


namespace exercise_l2388_238820

open Set

theorem exercise (U A B : Set ℕ) (hU : U = {0, 1, 2, 3, 4, 5, 6}) (hA : A = {1, 3, 5}) (hB : B = {2, 4, 5}) :
  A ∩ (U \ B) = {1, 3} := by
  sorry

end exercise_l2388_238820


namespace multiplication_of_exponents_l2388_238885

theorem multiplication_of_exponents (x : ℝ) : (x ^ 4) * (x ^ 2) = x ^ 6 := 
by
  sorry

end multiplication_of_exponents_l2388_238885


namespace pooh_piglet_cake_sharing_l2388_238876

theorem pooh_piglet_cake_sharing (a b : ℚ) (h1 : a + b = 1) (h2 : b + a/3 = 3*b) : 
  a = 6/7 ∧ b = 1/7 :=
by
  sorry

end pooh_piglet_cake_sharing_l2388_238876


namespace last_three_digits_l2388_238857

theorem last_three_digits (n : ℕ) : 7^106 % 1000 = 321 :=
by
  sorry

end last_three_digits_l2388_238857


namespace carpet_interior_length_l2388_238807

/--
A carpet is designed using three different colors, forming three nested rectangles with different areas in an arithmetic progression. 
The innermost rectangle has a width of two feet. Each of the two colored borders is 2 feet wide on all sides.
Determine the length in feet of the innermost rectangle. 
-/
theorem carpet_interior_length 
  (x : ℕ) -- length of the innermost rectangle
  (hp : ∀ (a b c : ℕ), a = 2 * x ∧ b = (4 * x + 24) ∧ c = (4 * x + 56) → (b - a) = (c - b)) 
  : x = 4 :=
by
  sorry

end carpet_interior_length_l2388_238807


namespace smallest_value_of_sum_l2388_238832

theorem smallest_value_of_sum (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : 3 * a = 4 * b ∧ 4 * b = 7 * c) : a + b + c = 61 :=
sorry

end smallest_value_of_sum_l2388_238832


namespace train_length_l2388_238887

theorem train_length (V L : ℝ) 
  (h1 : L = V * 18) 
  (h2 : L + 600.0000000000001 = V * 54) : 
  L = 300.00000000000005 :=
by 
  sorry

end train_length_l2388_238887


namespace correct_distribution_l2388_238886

-- Define the conditions
def num_students : ℕ := 40
def ratio_A_to_B : ℚ := 0.8
def ratio_C_to_B : ℚ := 1.2

-- Definitions for the number of students earning each grade
def num_B (x : ℕ) : ℕ := x
def num_A (x : ℕ) : ℕ := Nat.floor (ratio_A_to_B * x)
def num_C (x : ℕ) : ℕ := Nat.ceil (ratio_C_to_B * x)

-- Prove the distribution is correct
theorem correct_distribution :
  ∃ x : ℕ, num_A x + num_B x + num_C x = num_students ∧ 
           num_A x = 10 ∧ num_B x = 14 ∧ num_C x = 16 :=
by
  sorry

end correct_distribution_l2388_238886


namespace percentage_palm_oil_in_cheese_l2388_238840

theorem percentage_palm_oil_in_cheese
  (initial_cheese_price: ℝ := 100)
  (cheese_price_increase: ℝ := 3)
  (palm_oil_price_increase_percentage: ℝ := 0.10)
  (expected_palm_oil_percentage : ℝ := 30):
  ∃ (palm_oil_initial_price: ℝ),
  cheese_price_increase = palm_oil_initial_price * palm_oil_price_increase_percentage ∧
  expected_palm_oil_percentage = 100 * (palm_oil_initial_price / initial_cheese_price) := by
  sorry

end percentage_palm_oil_in_cheese_l2388_238840


namespace south_walk_correct_representation_l2388_238869

theorem south_walk_correct_representation {north south : ℤ} (h_north : north = 3) (h_representation : south = -north) : south = -5 :=
by
  have h1 : -north = -3 := by rw [h_north]
  have h2 : -3 = -5 := by sorry
  rw [h_representation, h1]
  exact h2

end south_walk_correct_representation_l2388_238869


namespace solve_m_n_l2388_238868

theorem solve_m_n (m n : ℝ) (h : m^2 + 2 * m + n^2 - 6 * n + 10 = 0) :
  m = -1 ∧ n = 3 :=
sorry

end solve_m_n_l2388_238868


namespace cylinder_volume_eq_l2388_238860

variable (α β l : ℝ)

theorem cylinder_volume_eq (hα_pos : 0 < α ∧ α < π/2) (hβ_pos : 0 < β ∧ β < π/2) (hl_pos : 0 < l) :
  let V := (π * l^3 * Real.sin (2 * β) * Real.cos β) / (8 * (Real.cos α)^2)
  V = (π * l^3 * Real.sin (2 * β) * Real.cos β) / (8 * (Real.cos α)^2) :=
by 
  sorry

end cylinder_volume_eq_l2388_238860


namespace sum_of_highest_powers_of_10_and_6_dividing_20_factorial_l2388_238884

def legendre (n p : Nat) : Nat :=
  if p > 1 then (Nat.div n p + Nat.div n (p * p) + Nat.div n (p * p * p) + Nat.div n (p * p * p * p)) else 0

theorem sum_of_highest_powers_of_10_and_6_dividing_20_factorial :
  let highest_power_5 := legendre 20 5
  let highest_power_2 := legendre 20 2
  let highest_power_3 := legendre 20 3
  let highest_power_10 := min highest_power_2 highest_power_5
  let highest_power_6 := min highest_power_2 highest_power_3
  highest_power_10 + highest_power_6 = 12 :=
by
  sorry

end sum_of_highest_powers_of_10_and_6_dividing_20_factorial_l2388_238884


namespace evaluate_s_squared_plus_c_squared_l2388_238828

variable {x y : ℝ}

theorem evaluate_s_squared_plus_c_squared (r : ℝ) (h_r_def : r = Real.sqrt (x^2 + y^2))
                                          (s : ℝ) (h_s_def : s = y / r)
                                          (c : ℝ) (h_c_def : c = x / r) :
  s^2 + c^2 = 1 :=
sorry

end evaluate_s_squared_plus_c_squared_l2388_238828


namespace find_f_prime_zero_l2388_238899

noncomputable def f (a : ℝ) (fd0 : ℝ) (x : ℝ) : ℝ :=
  (a * x^2 + x - 1) * Real.exp x + fd0

theorem find_f_prime_zero (a fd0 : ℝ) : (deriv (f a fd0) 0 = 0) :=
by
  -- the proof would go here
  sorry

end find_f_prime_zero_l2388_238899


namespace find_y_l2388_238847

theorem find_y (x y : ℝ) (h1 : 9823 + x = 13200) (h2 : x = y / 3 + 37.5) : y = 10018.5 :=
by
  sorry

end find_y_l2388_238847


namespace incorrect_ac_bc_impl_a_b_l2388_238826

theorem incorrect_ac_bc_impl_a_b : ∀ (a b c : ℝ), (ac = bc → a = b) ↔ c ≠ 0 :=
by sorry

end incorrect_ac_bc_impl_a_b_l2388_238826


namespace number_of_two_point_safeties_l2388_238805

variables (f g s : ℕ)

theorem number_of_two_point_safeties (h1 : 4 * f = 6 * g) 
                                    (h2 : s = g + 2) 
                                    (h3 : 4 * f + 3 * g + 2 * s = 50) : 
                                    s = 6 := 
by sorry

end number_of_two_point_safeties_l2388_238805


namespace max_mn_on_parabola_l2388_238895

theorem max_mn_on_parabola :
  ∀ m n : ℝ, (n = -m^2 + 3) → (m + n ≤ 13 / 4) :=
by
  sorry

end max_mn_on_parabola_l2388_238895


namespace thirtieth_entry_satisfies_l2388_238844

def r_9 (n : ℕ) : ℕ := n % 9

theorem thirtieth_entry_satisfies (n : ℕ) (h : ∃ k : ℕ, k < 30 ∧ ∀ m < 30, k ≠ m → 
    (r_9 (7 * n + 3) ≤ 4) ∧ 
    ((r_9 (7 * n + 3) ≤ 4) ↔ 
    (r_9 (7 * m + 3) > 4))) :
  n = 37 :=
sorry

end thirtieth_entry_satisfies_l2388_238844


namespace even_function_value_sum_l2388_238865

noncomputable def g (x : ℝ) (d e f : ℝ) : ℝ :=
  d * x^8 - e * x^6 + f * x^2 + 5

theorem even_function_value_sum (d e f : ℝ) (h : g 15 d e f = 7) :
  g 15 d e f + g (-15) d e f = 14 := by
  sorry

end even_function_value_sum_l2388_238865


namespace y_work_duration_l2388_238846

theorem y_work_duration (x_rate y_rate : ℝ) (d : ℝ) :
  -- 1. x and y together can do the work in 20 days.
  (x_rate + y_rate = 1/20) →
  -- 2. x started the work alone and after 4 days y joined him till the work completed.
  -- 3. The total work lasted 10 days.
  (4 * x_rate + 6 * (x_rate + y_rate) = 1) →
  -- Prove: y can do the work alone in 12 days.
  y_rate = 1/12 :=
by {
  sorry
}

end y_work_duration_l2388_238846


namespace total_dresses_l2388_238838

theorem total_dresses (D M E : ℕ) (h1 : E = 16) (h2 : M = E / 2) (h3 : D = M + 12) : D + M + E = 44 :=
by
  sorry

end total_dresses_l2388_238838


namespace radius_of_smaller_circle_l2388_238872

theorem radius_of_smaller_circle (R r : ℝ) (h1 : R = 6)
  (h2 : 2 * R = 3 * 2 * r) : r = 2 :=
by
  sorry

end radius_of_smaller_circle_l2388_238872


namespace common_ratio_of_geometric_series_l2388_238818

theorem common_ratio_of_geometric_series :
  let a := (8:ℚ) / 10
  let second_term := (-6:ℚ) / 15 
  let r := second_term / a
  r = -1 / 2 :=
by
  let a := (8:ℚ) / 10
  let second_term := (-6:ℚ) / 15 
  let r := second_term / a
  have : r = -1 / 2 := sorry
  exact this

end common_ratio_of_geometric_series_l2388_238818


namespace regression_analysis_incorrect_statement_l2388_238852

theorem regression_analysis_incorrect_statement
  (y : ℕ → ℝ) (x : ℕ → ℝ) (b a : ℝ)
  (r : ℝ) (l : ℝ → ℝ) (P : ℝ × ℝ)
  (H1 : ∀ i, y i = b * x i + a)
  (H2 : abs r = 1 → ∀ x1 x2, l x1 = l x2 → x1 = x2)
  (H3 : ∃ m k, ∀ x, l x = m * x + k)
  (H4 : P.1 = b → l P.1 = P.2)
  (cond_A : ∀ i, y i ≠ b * x i + a) : false := 
sorry

end regression_analysis_incorrect_statement_l2388_238852


namespace sum_of_altitudes_of_triangle_l2388_238809

theorem sum_of_altitudes_of_triangle (a b c : ℝ) (h_line : ∀ x y, 8 * x + 10 * y = 80 → x = 10 ∨ y = 8) :
  (8 + 10 + 40/Real.sqrt 41) = 18 + 40/Real.sqrt 41 :=
by
  sorry

end sum_of_altitudes_of_triangle_l2388_238809


namespace initial_tiger_sharks_l2388_238867

open Nat

theorem initial_tiger_sharks (initial_guppies : ℕ) (initial_angelfish : ℕ) (initial_oscar_fish : ℕ)
  (sold_guppies : ℕ) (sold_angelfish : ℕ) (sold_tiger_sharks : ℕ) (sold_oscar_fish : ℕ)
  (remaining_fish : ℕ) (initial_total_fish : ℕ) (total_guppies_angelfish_oscar : ℕ) (initial_tiger_sharks : ℕ) :
  initial_guppies = 94 → initial_angelfish = 76 → initial_oscar_fish = 58 →
  sold_guppies = 30 → sold_angelfish = 48 → sold_tiger_sharks = 17 → sold_oscar_fish = 24 →
  remaining_fish = 198 →
  initial_total_fish = (sold_guppies + sold_angelfish + sold_tiger_sharks + sold_oscar_fish + remaining_fish) →
  total_guppies_angelfish_oscar = (initial_guppies + initial_angelfish + initial_oscar_fish) →
  initial_tiger_sharks = (initial_total_fish - total_guppies_angelfish_oscar) →
  initial_tiger_sharks = 89 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11
  sorry

end initial_tiger_sharks_l2388_238867


namespace ratio_of_ages_three_years_from_now_l2388_238862

theorem ratio_of_ages_three_years_from_now :
  ∃ L B : ℕ,
  (L + B = 6) ∧ 
  (L = (1/2 : ℝ) * B) ∧ 
  (L + 3 = 5) ∧ 
  (B + 3 = 7) → 
  (L + 3) / (B + 3) = (5/7 : ℝ) :=
by
  sorry

end ratio_of_ages_three_years_from_now_l2388_238862


namespace gopi_servant_salary_l2388_238815

theorem gopi_servant_salary (S : ℕ) (turban_price : ℕ) (cash_received : ℕ) (months_worked : ℕ) (total_months : ℕ) :
  turban_price = 70 →
  cash_received = 50 →
  months_worked = 9 →
  total_months = 12 →
  S = 160 :=
by
  sorry

end gopi_servant_salary_l2388_238815


namespace least_value_of_x_l2388_238893

theorem least_value_of_x 
  (x : ℕ) 
  (p : ℕ) 
  (hx : 0 < x) 
  (hp : Prime p) 
  (h : x = 2 * 11 * p) : x = 44 := 
by
  sorry

end least_value_of_x_l2388_238893


namespace linear_equation_a_ne_1_l2388_238806

theorem linear_equation_a_ne_1 (a : ℝ) : (∀ x : ℝ, (a - 1) * x - 6 = 0 → a ≠ 1) :=
sorry

end linear_equation_a_ne_1_l2388_238806


namespace intersection_point_of_lines_l2388_238810

theorem intersection_point_of_lines (n : ℕ) (x y : ℤ) :
  15 * x + 18 * y = 1005 ∧ y = n * x + 2 → n = 2 :=
by
  sorry

end intersection_point_of_lines_l2388_238810


namespace min_value_expr_l2388_238858

theorem min_value_expr (x y : ℝ) (h1 : x^2 + y^2 = 2) (h2 : |x| ≠ |y|) : 
  (∃ m : ℝ, (∀ x y : ℝ, x^2 + y^2 = 2 ∧ |x| ≠ |y| → m ≤ (1 / (x + y)^2 + 1 / (x - y)^2)) ∧ m = 1) :=
by
  sorry

end min_value_expr_l2388_238858


namespace Ian_hours_worked_l2388_238821

theorem Ian_hours_worked (money_left: ℝ) (hourly_rate: ℝ) (spent: ℝ) (earned: ℝ) (hours: ℝ) :
  money_left = 72 → hourly_rate = 18 → spent = earned / 2 → earned = money_left * 2 → 
  earned = hourly_rate * hours → hours = 8 :=
by
  intros h1 h2 h3 h4 h5
  -- Begin mathematical validation process here
  sorry

end Ian_hours_worked_l2388_238821


namespace range_of_expression_positive_range_of_expression_negative_l2388_238848

theorem range_of_expression_positive (x : ℝ) : 
  (2 * x ^ 2 - 5 * x - 12 > 0) ↔ (x < -3/2 ∨ x > 4) :=
sorry

theorem range_of_expression_negative (x : ℝ) : 
  (2 * x ^ 2 - 5 * x - 12 < 0) ↔ ( -3/2 < x ∧ x < 4) :=
sorry

end range_of_expression_positive_range_of_expression_negative_l2388_238848


namespace probability_more_than_60000_l2388_238823

def boxes : List ℕ := [8, 800, 8000, 40000, 80000]

def probability_keys (keys : ℕ) : ℚ :=
  1 / keys

def probability_winning (n : ℕ) : ℚ :=
  if n = 4 then probability_keys 5 + probability_keys 5 * probability_keys 4 else 0

theorem probability_more_than_60000 : 
  probability_winning 4 = 1/4 := sorry

end probability_more_than_60000_l2388_238823


namespace find_divisor_l2388_238875

theorem find_divisor (n k : ℤ) (h1 : n % 30 = 16) : (2 * n) % 30 = 2 :=
by
  sorry

end find_divisor_l2388_238875


namespace x_squared_plus_y_squared_l2388_238877

theorem x_squared_plus_y_squared (x y : ℝ) (h1 : (x + y)^2 = 1) (h2 : x * y = -4) : x^2 + y^2 = 9 :=
sorry

end x_squared_plus_y_squared_l2388_238877


namespace geometric_sequence_x_value_l2388_238833

theorem geometric_sequence_x_value (x : ℝ) (r : ℝ) 
  (h1 : 12 * r = x) 
  (h2 : x * r = 2 / 3) 
  (h3 : 0 < x) :
  x = 2 * Real.sqrt 2 :=
by
  sorry

end geometric_sequence_x_value_l2388_238833


namespace avg_of_all_5_is_8_l2388_238853

-- Let a1, a2, a3 be three quantities such that their average is 4.
def is_avg_4 (a1 a2 a3 : ℝ) : Prop :=
  (a1 + a2 + a3) / 3 = 4

-- Let a4, a5 be the remaining two quantities such that their average is 14.
def is_avg_14 (a4 a5 : ℝ) : Prop :=
  (a4 + a5) / 2 = 14

-- Prove that the average of all 5 quantities is 8.
theorem avg_of_all_5_is_8 (a1 a2 a3 a4 a5 : ℝ) :
  is_avg_4 a1 a2 a3 ∧ is_avg_14 a4 a5 → 
  ((a1 + a2 + a3 + a4 + a5) / 5 = 8) :=
by
  intro h
  sorry

end avg_of_all_5_is_8_l2388_238853


namespace distance_between_sasha_and_kolya_is_19_meters_l2388_238835

theorem distance_between_sasha_and_kolya_is_19_meters
  (v_S v_L v_K : ℝ)
  (h1 : v_L = 0.9 * v_S)
  (h2 : v_K = 0.81 * v_S)
  (h3 : ∀ t_S : ℝ, t_S = 100 / v_S) :
  (∀ t_S : ℝ, 100 - v_K * t_S = 19) :=
by
  intros t_S
  have vL_defined : v_L = 0.9 * v_S := h1
  have vK_defined : v_K = 0.81 * v_S := h2
  have time_S : t_S = 100 / v_S := h3 t_S
  sorry

end distance_between_sasha_and_kolya_is_19_meters_l2388_238835


namespace thirteen_pow_2011_mod_100_l2388_238891

theorem thirteen_pow_2011_mod_100 : (13^2011) % 100 = 37 := by
  sorry

end thirteen_pow_2011_mod_100_l2388_238891


namespace inscribed_circle_radius_l2388_238855

theorem inscribed_circle_radius (R r x : ℝ) (hR : R = 18) (hr : r = 9) :
  x = 8 :=
sorry

end inscribed_circle_radius_l2388_238855


namespace production_equation_l2388_238843

-- Definitions based on the problem conditions
def original_production_rate (x : ℕ) := x
def additional_parts_per_day := 4
def original_days := 20
def actual_days := 15
def extra_parts := 10

-- Prove the equation
theorem production_equation (x : ℕ) :
  original_days * original_production_rate x = actual_days * (original_production_rate x + additional_parts_per_day) - extra_parts :=
by
  simp [original_production_rate, additional_parts_per_day, original_days, actual_days, extra_parts]
  sorry

end production_equation_l2388_238843


namespace no_positive_integer_solutions_l2388_238813

theorem no_positive_integer_solutions : ¬∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x ^ 4004 + y ^ 4004 = z ^ 2002 :=
by
  sorry

end no_positive_integer_solutions_l2388_238813


namespace tim_more_points_than_joe_l2388_238819

variable (J K T : ℕ)

theorem tim_more_points_than_joe (h1 : T = 30) (h2 : T = K / 2) (h3 : J + T + K = 100) : T - J = 20 :=
by
  sorry

end tim_more_points_than_joe_l2388_238819


namespace right_triangle_shorter_leg_l2388_238812

theorem right_triangle_shorter_leg (a b c : ℕ) (h1 : a^2 + b^2 = c^2) (h2 : c = 65) : a = 25 ∨ b = 25 := 
by
  sorry

end right_triangle_shorter_leg_l2388_238812


namespace sum_of_roots_eq_3n_l2388_238845

variable {n : ℝ} 

-- Define the conditions
def quadratic_eq (x : ℝ) (m : ℝ) (n : ℝ) : Prop :=
  x^2 - (m + n) * x + m * n = 0

theorem sum_of_roots_eq_3n (m : ℝ) (n : ℝ) 
  (hm : m = 2 * n)
  (hroot_m : quadratic_eq m m n)
  (hroot_n : quadratic_eq n m n) :
  m + n = 3 * n :=
by sorry

end sum_of_roots_eq_3n_l2388_238845


namespace maximum_positive_numbers_l2388_238889

theorem maximum_positive_numbers (a : ℕ → ℝ) (n : ℕ) (h₀ : n = 100)
  (h₁ : ∀ i : ℕ, 0 < a i) 
  (h₂ : ∀ i : ℕ, a i > a ((i + 1) % n) * a ((i + 2) % n)) : 
  ∃ m : ℕ, m ≤ 50 ∧ (∀ k : ℕ, k < m → (a k) > 0) :=
by sorry

end maximum_positive_numbers_l2388_238889


namespace inequality_proof_l2388_238881

theorem inequality_proof (a b : ℝ) (h : a + b ≠ 0) :
  (a + b) / (a^2 - a * b + b^2) ≤ 4 / |a + b| ∧
  ((a + b) / (a^2 - a * b + b^2) = 4 / |a + b| ↔ a = b) :=
by
  sorry

end inequality_proof_l2388_238881


namespace part_one_part_two_l2388_238842

def f (x a : ℝ) : ℝ :=
  x^2 + a * (abs x) + x 

theorem part_one (x1 x2 a : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) :
  (1 / 2) * (f x1 a + f x2 a) ≥ f ((x1 + x2) / 2) a :=
sorry

theorem part_two (a : ℝ) (ha : 0 ≤ a) (x1 x2 : ℝ) :
  (1 / 2) * (f x1 a + f x2 a) ≥ f ((x1 + x2) / 2) a :=
sorry

end part_one_part_two_l2388_238842


namespace find_a_solution_l2388_238825

open Complex

noncomputable def find_a : Prop := 
  ∃ a : ℂ, ((1 + a * I) / (2 + I) = 1 + 2 * I) ∧ (a = 5 + I)

theorem find_a_solution : find_a := 
  by
    sorry

end find_a_solution_l2388_238825


namespace sphere_surface_area_of_circumscribing_cuboid_l2388_238839

theorem sphere_surface_area_of_circumscribing_cuboid :
  ∀ (a b c : ℝ), a = 5 ∧ b = 4 ∧ c = 3 → 4 * Real.pi * ((Real.sqrt ((a^2 + b^2 + c^2)) / 2) ^ 2) = 50 * Real.pi :=
by
  -- introduction of variables and conditions
  intros a b c h
  obtain ⟨_, _, _⟩ := h -- decomposing the conditions
  -- the proof is skipped
  sorry

end sphere_surface_area_of_circumscribing_cuboid_l2388_238839


namespace polynomial_remainder_is_zero_l2388_238830

theorem polynomial_remainder_is_zero :
  ∀ (x : ℤ), ((x^5 - 1) * (x^3 - 1)) % (x^2 + x + 1) = 0 := 
by
  sorry

end polynomial_remainder_is_zero_l2388_238830


namespace largest_divisor_l2388_238816

theorem largest_divisor (n : ℕ) (h1 : 0 < n) (h2 : 450 ∣ n ^ 2) : 30 ∣ n :=
sorry

end largest_divisor_l2388_238816


namespace value_expression_eq_zero_l2388_238883

theorem value_expression_eq_zero (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
    (h_condition : a / (b - c) + b / (c - a) + c / (a - b) = 1) :
    a / (b - c)^2 + b / (c - a)^2 + c / (a - b)^2 = 0 :=
by
  sorry

end value_expression_eq_zero_l2388_238883


namespace least_n_l2388_238822

theorem least_n (n : ℕ) (h_pos : n > 0) (h_ineq : 1 / n - 1 / (n + 1) < 1 / 15) : n = 4 :=
sorry

end least_n_l2388_238822


namespace quadratic_inequality_solution_l2388_238831

theorem quadratic_inequality_solution (x : ℝ) : (-x^2 + 5 * x - 4 < 0) ↔ (1 < x ∧ x < 4) :=
sorry

end quadratic_inequality_solution_l2388_238831


namespace find_range_m_l2388_238856

noncomputable def p (m : ℝ) : Prop :=
  ∃ x₀ : ℝ, m * x₀^2 + 1 < 1

def q (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + m * x + 1 ≥ 0

theorem find_range_m (m : ℝ) : ¬ (p m ∨ ¬ q m) ↔ -2 ≤ m ∧ m ≤ 2 :=
  sorry

end find_range_m_l2388_238856


namespace integral_sign_l2388_238871

noncomputable def I : ℝ := ∫ x in -Real.pi..0, Real.sin x

theorem integral_sign : I < 0 := sorry

end integral_sign_l2388_238871


namespace line_perp_to_plane_imp_perp_to_line_l2388_238850

def Line := Type
def Plane := Type

variables (m n : Line) (α : Plane)

def is_parallel (l : Line) (p : Plane) : Prop := sorry
def is_perpendicular (l1 l2 : Line) : Prop := sorry
def is_contained (l : Line) (p : Plane) : Prop := sorry

theorem line_perp_to_plane_imp_perp_to_line :
  (is_perpendicular m α) ∧ (is_contained n α) → (is_perpendicular m n) :=
sorry

end line_perp_to_plane_imp_perp_to_line_l2388_238850


namespace aladdin_can_find_heavy_coins_l2388_238892

theorem aladdin_can_find_heavy_coins :
  ∃ (x y : ℕ), 1 ≤ x ∧ x ≤ 20 ∧ 1 ≤ y ∧ y ≤ 20 ∧ x ≠ y ∧ (x + y ≥ 28) :=
by
  sorry

end aladdin_can_find_heavy_coins_l2388_238892
