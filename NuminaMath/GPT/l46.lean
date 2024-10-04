import Mathlib

namespace equation_solution_l46_46949

noncomputable def solve_equation : Prop :=
∃ (x : ℝ), x^6 + (3 - x)^6 = 730 ∧ (x = 1.5 + Real.sqrt 5 ∨ x = 1.5 - Real.sqrt 5)

theorem equation_solution : solve_equation :=
sorry

end equation_solution_l46_46949


namespace midpoint_fraction_l46_46212

theorem midpoint_fraction (a b : ℚ) (h1 : a = 3 / 4) (h2 : b = 5 / 6) :
  (a + b) / 2 = 19 / 24 :=
by {
  sorry
}

end midpoint_fraction_l46_46212


namespace best_purchase_option_l46_46532

-- Define the prices and discount conditions for each store
def technik_city_price_before_discount : ℝ := 2000 + 4000
def technomarket_price_before_discount : ℝ := 1500 + 4800

def technik_city_discount : ℝ := technik_city_price_before_discount * 0.10
def technomarket_bonus : ℝ := technomarket_price_before_discount * 0.20

def technik_city_final_price : ℝ := technik_city_price_before_discount - technik_city_discount
def technomarket_final_price : ℝ := technomarket_price_before_discount

-- The theorem stating the ultimate proof problem
theorem best_purchase_option : technik_city_final_price < technomarket_final_price :=
by
  -- Replace 'sorry' with the actual proof if required
  sorry

end best_purchase_option_l46_46532


namespace T_number_square_l46_46506

theorem T_number_square (a b : ℤ) : ∃ c d : ℤ, (a^2 + a * b + b^2)^2 = c^2 + c * d + d^2 := by
  sorry

end T_number_square_l46_46506


namespace baseball_cap_problem_l46_46733

theorem baseball_cap_problem 
  (n_first_week n_second_week n_third_week n_fourth_week total_caps : ℕ) 
  (h2 : n_second_week = 400) 
  (h3 : n_third_week = 300) 
  (h4 : n_fourth_week = (n_first_week + n_second_week + n_third_week) / 3) 
  (h_total : n_first_week + n_second_week + n_third_week + n_fourth_week = 1360) : 
  n_first_week = 320 := 
by 
  sorry

end baseball_cap_problem_l46_46733


namespace find_number_of_children_l46_46080

theorem find_number_of_children (adults children : ℕ) (adult_ticket_price child_ticket_price total_money change : ℕ) 
    (h1 : adult_ticket_price = 9) 
    (h2 : child_ticket_price = adult_ticket_price - 2) 
    (h3 : total_money = 40) 
    (h4 : change = 1) 
    (h5 : adults = 2) 
    (total_cost : total_money - change = adults * adult_ticket_price + children * child_ticket_price) : 
    children = 3 :=
sorry

end find_number_of_children_l46_46080


namespace polynomial_divisibility_l46_46536

noncomputable def A := sorry
noncomputable def B := sorry
noncomputable def C := sorry

theorem polynomial_divisibility (A B C : ℝ) 
    (h : ∀ x : ℂ, x^2 + x + 1 = 0 → x^103 + C * x^2 + A * x + B = 0) : 
    A + B + C = 3 * C - 1 :=
by
  sorry

end polynomial_divisibility_l46_46536


namespace bookshop_inventory_l46_46773

theorem bookshop_inventory
  (initial_inventory : ℕ := 743)
  (saturday_sales_instore : ℕ := 37)
  (saturday_sales_online : ℕ := 128)
  (sunday_sales_instore : ℕ := 2 * saturday_sales_instore)
  (sunday_sales_online : ℕ := saturday_sales_online + 34)
  (new_shipment : ℕ := 160) :
  (initial_inventory - (saturday_sales_instore + saturday_sales_online + sunday_sales_instore + sunday_sales_online) + new_shipment = 502) :=
by
  sorry

end bookshop_inventory_l46_46773


namespace toothpicks_per_card_l46_46799

-- Define the conditions of the problem
def numCardsInDeck : ℕ := 52
def numCardsNotUsed : ℕ := 16
def numCardsUsed : ℕ := numCardsInDeck - numCardsNotUsed

def numBoxesToothpicks : ℕ := 6
def toothpicksPerBox : ℕ := 450
def totalToothpicksUsed : ℕ := numBoxesToothpicks * toothpicksPerBox

-- Prove the number of toothpicks used per card
theorem toothpicks_per_card : totalToothpicksUsed / numCardsUsed = 75 := 
  by sorry

end toothpicks_per_card_l46_46799


namespace red_balls_l46_46698

theorem red_balls (w r : ℕ) (h1 : w = 12) (h2 : w * 3 = r * 4) : r = 9 :=
sorry

end red_balls_l46_46698


namespace largest_side_of_triangle_l46_46541

theorem largest_side_of_triangle (x y Δ c : ℕ)
  (h1 : (x + 2 * Δ / x = y + 2 * Δ / y))
  (h2 : x = 60)
  (h3 : y = 63) :
  c = 87 :=
sorry

end largest_side_of_triangle_l46_46541


namespace longest_side_of_triangle_l46_46535

-- Definitions of the conditions in a)
def side1 : ℝ := 9
def side2 (x : ℝ) : ℝ := x + 5
def side3 (x : ℝ) : ℝ := 2 * x + 3
def perimeter : ℝ := 40

-- Statement of the mathematically equivalent proof problem.
theorem longest_side_of_triangle (x : ℝ) (h : side1 + side2 x + side3 x = perimeter) : 
  max side1 (max (side2 x) (side3 x)) = side3 x := 
sorry

end longest_side_of_triangle_l46_46535


namespace find_point_B_coordinates_l46_46225

theorem find_point_B_coordinates (a : ℝ) : 
  (∀ (x y : ℝ), x^2 - 4*x + y^2 = 0 → (x - a)^2 + y^2 = 4 * ((x - 1)^2 + y^2)) →
  a = -2 :=
by
  sorry

end find_point_B_coordinates_l46_46225


namespace probability_of_sum_8_9_10_l46_46056

/-- The list of face values for the first die. -/
def first_die : List ℕ := [1, 1, 3, 3, 5, 6]

/-- The list of face values for the second die. -/
def second_die : List ℕ := [1, 2, 4, 5, 7, 9]

/-- The condition to verify if the sum is 8, 9, or 10. -/
def valid_sum (s : ℕ) : Bool := s = 8 ∨ s = 9 ∨ s = 10

/-- Calculate probability of the sum being 8, 9, or 10 for the two dice. -/
def calculate_probability : ℚ :=
  let total_rolls := first_die.length * second_die.length
  let valid_rolls := 
    first_die.foldl (fun acc d1 =>
      acc + second_die.foldl (fun acc' d2 => 
        if valid_sum (d1 + d2) then acc' + 1 else acc') 0) 0
  valid_rolls / total_rolls

/-- The required probability is 7/18. -/
theorem probability_of_sum_8_9_10 : calculate_probability = 7 / 18 := 
  sorry

end probability_of_sum_8_9_10_l46_46056


namespace complex_magnitude_sixth_power_l46_46801

noncomputable def z := (2 : ℂ) + (2 * Real.sqrt 3) * Complex.I

theorem complex_magnitude_sixth_power :
  Complex.abs (z^6) = 4096 := 
by
  sorry

end complex_magnitude_sixth_power_l46_46801


namespace perpendicular_lines_sum_is_minus_four_l46_46819

theorem perpendicular_lines_sum_is_minus_four 
  (a b c : ℝ) 
  (h1 : (a * 2) / (4 * 5) = 1)
  (h2 : 10 * 1 + 4 * c - 2 = 0)
  (h3 : 2 * 1 - 5 * (-2) + b = 0) : 
  a + b + c = -4 := 
sorry

end perpendicular_lines_sum_is_minus_four_l46_46819


namespace relationship_among_a_b_c_l46_46004

noncomputable def a : ℝ := (0.6:ℝ) ^ (0.2:ℝ)
noncomputable def b : ℝ := (0.2:ℝ) ^ (0.2:ℝ)
noncomputable def c : ℝ := (0.2:ℝ) ^ (0.6:ℝ)

theorem relationship_among_a_b_c : a > b ∧ b > c :=
by
  -- The proof can be added here if needed
  sorry

end relationship_among_a_b_c_l46_46004


namespace rice_in_first_5_days_l46_46411

-- Define the arithmetic sequence for number of workers dispatched each day
def num_workers (n : ℕ) : ℕ := 64 + (n - 1) * 7

-- Function to compute the sum of the first n terms of the arithmetic sequence
def sum_workers (n : ℕ) : ℕ := n * 64 + (n * (n - 1)) / 2 * 7

-- Given the rice distribution conditions
def rice_per_worker : ℕ := 3

-- Given the problem specific conditions
def total_rice_distributed_first_5_days : ℕ := 
  rice_per_worker * (sum_workers 1 + sum_workers 2 + sum_workers 3 + sum_workers 4 + sum_workers 5)
  
-- Proof goal
theorem rice_in_first_5_days : total_rice_distributed_first_5_days = 3300 :=
  by
  sorry

end rice_in_first_5_days_l46_46411


namespace reciprocal_neg_one_over_2023_eq_neg_2023_l46_46040

theorem reciprocal_neg_one_over_2023_eq_neg_2023 : (1 / (-1 / (2023 : ℝ))) = -2023 :=
by
  sorry

end reciprocal_neg_one_over_2023_eq_neg_2023_l46_46040


namespace bella_more_than_max_l46_46026

noncomputable def num_students : ℕ := 10
noncomputable def bananas_eaten_by_bella : ℕ := 7
noncomputable def bananas_eaten_by_max : ℕ := 1

theorem bella_more_than_max : 
  bananas_eaten_by_bella - bananas_eaten_by_max = 6 :=
by
  sorry

end bella_more_than_max_l46_46026


namespace cuckoo_chime_78_l46_46727

-- Define the arithmetic sum for the cuckoo clock problem
def cuckoo_chime_sum (n a l : Nat) : Nat :=
  n * (a + l) / 2

-- Main theorem
theorem cuckoo_chime_78 : 
  cuckoo_chime_sum 12 1 12 = 78 := 
by
  -- Proof part can be written here
  sorry

end cuckoo_chime_78_l46_46727


namespace find_c_l46_46358

theorem find_c
  (m b d c : ℝ)
  (h : m = b * d * c / (d + c)) :
  c = m * d / (b * d - m) :=
sorry

end find_c_l46_46358


namespace erin_days_to_receive_30_l46_46117

theorem erin_days_to_receive_30 (x : ℕ) (h : 3 * x = 30) : x = 10 :=
by
  sorry

end erin_days_to_receive_30_l46_46117


namespace line_intersects_curve_l46_46033

theorem line_intersects_curve (k : ℝ) :
  (∃ x y : ℝ, y + k * x + 2 = 0 ∧ x^2 + y^2 = 2 * x) ↔ k ≤ -3/4 := by
  sorry

end line_intersects_curve_l46_46033


namespace OQ_value_l46_46998

variables {X Y Z N O Q R : Type}
variables [MetricSpace X] [MetricSpace Y] [MetricSpace Z]
variables [MetricSpace N] [MetricSpace O] [MetricSpace Q] [MetricSpace R]
variables (XY YZ XN NY ZO XO OZ YN XR OQ RQ : ℝ)
variables (triangle_XYZ : Triangle X Y Z)
variables (X_equal_midpoint_XY : XY = 540)
variables (Y_equal_midpoint_YZ : YZ = 360)
variables (XN_equal_NY : XN = NY)
variables (ZO_is_angle_bisector : is_angle_bisector Z O X Y)
variables (intersection_YN_ZO : Q = intersection YN ZO)
variables (N_midpoint_RQ : is_midpoint N R Q)
variables (XR_value : XR = 216)

theorem OQ_value : OQ = 216 := sorry

end OQ_value_l46_46998


namespace inequality_selection_l46_46664

theorem inequality_selection (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  1/a + 4/b ≥ 9/(a + b) :=
sorry

end inequality_selection_l46_46664


namespace smallest_possible_a_l46_46003

theorem smallest_possible_a (a b c : ℤ) (h1 : a < b) (h2 : b < c)
  (h3 : 2 * b = a + c) (h4 : a^2 = c * b) : a = 1 :=
by
  sorry

end smallest_possible_a_l46_46003


namespace coordinates_of_C_l46_46437

theorem coordinates_of_C (A B : ℝ × ℝ) (C : ℝ × ℝ) 
    (hA : A = (1, 3)) (hB : B = (9, -3)) (hBC_AB : dist B C = 1/2 * dist A B) : 
    C = (13, -6) :=
sorry

end coordinates_of_C_l46_46437


namespace sum_of_consecutive_integers_of_sqrt3_l46_46636

theorem sum_of_consecutive_integers_of_sqrt3 {a b : ℤ} (h1 : a + 1 = b) (h2 : (a : ℝ) < Real.sqrt 3) (h3 : Real.sqrt 3 < (b : ℝ)) :
  a + b = 3 := by
  sorry

end sum_of_consecutive_integers_of_sqrt3_l46_46636


namespace max_real_roots_among_polynomials_l46_46490

noncomputable def largest_total_real_roots (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) : ℕ :=
  4  -- representing the largest total number of real roots

theorem max_real_roots_among_polynomials
  (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) :
  largest_total_real_roots a b c h_a h_b h_c = 4 :=
sorry

end max_real_roots_among_polynomials_l46_46490


namespace solve_division_problem_l46_46023

-- Problem Conditions
def division_problem : ℚ := 0.25 / 0.005

-- Proof Problem Statement
theorem solve_division_problem : division_problem = 50 := by
  sorry

end solve_division_problem_l46_46023


namespace total_watermelons_l46_46441

theorem total_watermelons 
  (A B C : ℕ) 
  (h1 : A + B = C - 6) 
  (h2 : B + C = A + 16) 
  (h3 : C + A = B + 8) :
  A + B + C = 18 :=
by
  sorry

end total_watermelons_l46_46441


namespace solve_marble_problem_l46_46988

noncomputable def marble_problem : Prop :=
  ∃ k : ℕ, k ≥ 0 ∧ k ≤ 50 ∧ 
  (∀ initial_white initial_black : ℕ, initial_white = 50 ∧ initial_black = 50 → 
  ∃ w b : ℕ, w = 50 + k - initial_black ∧ b = 50 - k ∧ (w, b) = (2, 0))

theorem solve_marble_problem: marble_problem :=
sorry

end solve_marble_problem_l46_46988


namespace length_of_rectangle_l46_46384

-- Given conditions as per the problem statement
variables {s l : ℝ} -- side length of the square, length of the rectangle
def width_rectangle : ℝ := 10 -- width of the rectangle

-- Conditions
axiom sq_perimeter : 4 * s = 200
axiom area_relation : s^2 = 5 * (l * width_rectangle)

-- Goal to prove
theorem length_of_rectangle : l = 50 :=
by
  sorry

end length_of_rectangle_l46_46384


namespace mr_johnson_needs_additional_volunteers_l46_46175

-- Definitions for the given conditions
def math_classes := 5
def students_per_class := 4
def total_students := math_classes * students_per_class

def total_teachers := 10
def carpentry_skilled_teachers := 3

def total_parents := 15
def lighting_sound_experienced_parents := 6

def total_volunteers_needed := 100
def carpentry_volunteers_needed := 8
def lighting_sound_volunteers_needed := 10

-- Total current volunteers
def current_volunteers := total_students + total_teachers + total_parents

-- Volunteers with specific skills
def current_carpentry_skilled := carpentry_skilled_teachers
def current_lighting_sound_experienced := lighting_sound_experienced_parents

-- Additional volunteers needed
def additional_carpentry_needed :=
  carpentry_volunteers_needed - current_carpentry_skilled
def additional_lighting_sound_needed :=
  lighting_sound_volunteers_needed - current_lighting_sound_experienced

-- Total additional volunteer needed
def additional_volunteers_needed :=
  additional_carpentry_needed + additional_lighting_sound_needed

-- The theorem we need to prove:
theorem mr_johnson_needs_additional_volunteers :
  additional_volunteers_needed = 9 := by
  sorry

end mr_johnson_needs_additional_volunteers_l46_46175


namespace find_x_l46_46224

theorem find_x (x y : ℤ) (h1 : x + 2 * y = 100) (h2 : y = 25) : x = 50 :=
by
  sorry

end find_x_l46_46224


namespace find_x_in_interval_l46_46802

theorem find_x_in_interval (x : ℝ) : x^2 + 5 * x < 10 ↔ -5 < x ∧ x < 2 :=
sorry

end find_x_in_interval_l46_46802


namespace total_ticket_income_l46_46077

-- All given conditions as definitions/assumptions
def total_seats : ℕ := 200
def children_tickets : ℕ := 60
def adult_ticket_price : ℝ := 3.00
def children_ticket_price : ℝ := 1.50
def adult_tickets : ℕ := total_seats - children_tickets

-- The claim we need to prove
theorem total_ticket_income :
  (adult_tickets * adult_ticket_price + children_tickets * children_ticket_price) = 510.00 :=
by
  -- Placeholder to complete proof later
  sorry

end total_ticket_income_l46_46077


namespace similar_triangles_side_length_l46_46194

theorem similar_triangles_side_length
  (A1 A2 : ℕ) (k : ℕ) (h1 : A1 - A2 = 18)
  (h2 : A1 = k^2 * A2) (h3 : ∃ n : ℕ, A2 = n)
  (s : ℕ) (h4 : s = 3) :
  s * k = 6 :=
by
  sorry

end similar_triangles_side_length_l46_46194


namespace find_width_of_room_eq_l46_46388

noncomputable def total_cost : ℝ := 20625
noncomputable def rate_per_sqm : ℝ := 1000
noncomputable def length_of_room : ℝ := 5.5
noncomputable def area_paved : ℝ := total_cost / rate_per_sqm
noncomputable def width_of_room : ℝ := area_paved / length_of_room

theorem find_width_of_room_eq :
  width_of_room = 3.75 :=
sorry

end find_width_of_room_eq_l46_46388


namespace probability_of_drawing_jingyuetan_ticket_l46_46706

-- Definitions from the problem
def num_jingyuetan_tickets : ℕ := 3
def num_changying_tickets : ℕ := 2
def total_tickets : ℕ := num_jingyuetan_tickets + num_changying_tickets
def num_envelopes : ℕ := total_tickets

-- Probability calculation
def probability_jingyuetan : ℚ := (num_jingyuetan_tickets : ℚ) / (num_envelopes : ℚ)

-- Theorem statement
theorem probability_of_drawing_jingyuetan_ticket : probability_jingyuetan = 3 / 5 :=
by
  sorry

end probability_of_drawing_jingyuetan_ticket_l46_46706


namespace frequency_of_largest_rectangle_area_l46_46486

theorem frequency_of_largest_rectangle_area (a : ℕ → ℝ) (sample_size : ℕ)
    (h_geom : ∀ n, a (n + 1) = 2 * a n) (h_sum : a 0 + a 1 + a 2 + a 3 = 1)
    (h_sample : sample_size = 300) : 
    sample_size * a 3 = 160 := by
  sorry

end frequency_of_largest_rectangle_area_l46_46486


namespace bread_rolls_count_l46_46054

theorem bread_rolls_count (total_items croissants bagels : Nat) 
  (h1 : total_items = 90) 
  (h2 : croissants = 19) 
  (h3 : bagels = 22) : 
  total_items - croissants - bagels = 49 := 
by
  sorry

end bread_rolls_count_l46_46054


namespace largest_and_smallest_values_quartic_real_roots_l46_46473

noncomputable def function_y (a b x : ℝ) : ℝ :=
  (4 * a^2 * x^2 + b^2 * (x^2 - 1)^2) / (x^2 + 1)^2

theorem largest_and_smallest_values (a b : ℝ) (h : a > b) :
  ∃ x y, function_y a b x = y^2 ∧ y = a ∧ y = b :=
by
  sorry

theorem quartic_real_roots (a b y : ℝ) (h₁ : a > b) (h₂ : y > b) (h₃ : y < a) :
  ∃ x₀ x₁ x₂ x₃, function_y a b x₀ = y^2 ∧ function_y a b x₁ = y^2 ∧ function_y a b x₂ = y^2 ∧ function_y a b x₃ = y^2 :=
by
  sorry

end largest_and_smallest_values_quartic_real_roots_l46_46473


namespace area_enclosed_by_S_l46_46381

open Complex

def five_presentable (v : ℂ) : Prop := abs v = 5

def S : Set ℂ := {u | ∃ v : ℂ, five_presentable v ∧ u = v - (1 / v)}

theorem area_enclosed_by_S : 
  ∃ (area : ℝ), area = 624 / 25 * Real.pi :=
by
  sorry

end area_enclosed_by_S_l46_46381


namespace max_side_range_of_triangle_l46_46279

-- Define the requirement on the sides a and b
def side_condition (a b : ℝ) : Prop :=
  |a - 3| + (b - 7)^2 = 0

-- Prove the range of side c
theorem max_side_range_of_triangle (a b c : ℝ) (h : side_condition a b) (hc : c = max a (max b c)) :
  7 ≤ c ∧ c < 10 :=
sorry

end max_side_range_of_triangle_l46_46279


namespace candy_difference_l46_46686

theorem candy_difference 
  (total_candies : ℕ)
  (strawberry_candies : ℕ)
  (total_eq : total_candies = 821)
  (strawberry_eq : strawberry_candies = 267) : 
  (total_candies - strawberry_candies - strawberry_candies = 287) :=
by
  sorry

end candy_difference_l46_46686


namespace add_sub_decimals_l46_46582

theorem add_sub_decimals :
  (0.513 + 0.0067 - 0.048 = 0.4717) :=
by
  sorry

end add_sub_decimals_l46_46582


namespace test_question_total_l46_46716

theorem test_question_total
  (total_points : ℕ)
  (points_2q : ℕ)
  (points_4q : ℕ)
  (num_2q : ℕ)
  (num_4q : ℕ)
  (H1 : total_points = 100)
  (H2 : points_2q = 2)
  (H3 : points_4q = 4)
  (H4 : num_2q = 30)
  (H5 : total_points = num_2q * points_2q + num_4q * points_4q) :
  num_2q + num_4q = 40 := 
sorry

end test_question_total_l46_46716


namespace distance_walked_on_third_day_l46_46661

theorem distance_walked_on_third_day:
  ∃ x : ℝ, 
    4 * x + 2 * x + x + (1 / 2) * x + (1 / 4) * x + (1 / 8) * x = 378 ∧
    x = 48 := 
by
  sorry

end distance_walked_on_third_day_l46_46661


namespace employee_age_when_hired_l46_46910

theorem employee_age_when_hired
    (hire_year retire_year : ℕ)
    (rule_of_70 : ∀ A Y, A + Y = 70)
    (years_worked : ∀ hire_year retire_year, retire_year - hire_year = 19)
    (hire_year_eqn : hire_year = 1987)
    (retire_year_eqn : retire_year = 2006) :
  ∃ A : ℕ, A = 51 :=
by
  have Y := 19
  have A := 70 - Y
  use A
  sorry

end employee_age_when_hired_l46_46910


namespace solve_the_problem_l46_46469
-- Import required Lean libraries

-- Define the given functions and prove the properties
open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a ^ x - a + 1
noncomputable def g (x : ℝ) : ℝ := f (1 / 2) (x + 1 / 2) - 1
noncomputable def F (m : ℝ) (x : ℝ) : ℝ := g (2 * x) - m * g (x - 1)
noncomputable def h (m : ℝ) : ℝ :=
if h: m ≤ 1 then
  1 - 2 * m
else if h: 1 < m ∧ m < 2 then
  -m ^ 2
else
  4 - 4 * m

-- Define the theorem with equivalent conditions and answer
theorem solve_the_problem :
  (∃ a : ℝ, 0 < a ∧ a ≠ 1 ∧ f a (1/2) = 2 ∧ a = 1/2) 
  ∧ (∀ x : ℝ, g x = (1/2) ^ x) 
  ∧ (∀ m : ℝ, ∀ x ∈ Icc (-1 : ℝ) (0 : ℝ), h m = (if h: m ≤ 1 then 1 - 2 * m else if h: 1 < m ∧ m < 2 then -m ^ 2 else 4 - 4 * m))
  := by
  sorry

end solve_the_problem_l46_46469


namespace bookshop_inventory_l46_46774

theorem bookshop_inventory
  (initial_inventory : ℕ := 743)
  (saturday_sales_instore : ℕ := 37)
  (saturday_sales_online : ℕ := 128)
  (sunday_sales_instore : ℕ := 2 * saturday_sales_instore)
  (sunday_sales_online : ℕ := saturday_sales_online + 34)
  (new_shipment : ℕ := 160) :
  (initial_inventory - (saturday_sales_instore + saturday_sales_online + sunday_sales_instore + sunday_sales_online) + new_shipment = 502) :=
by
  sorry

end bookshop_inventory_l46_46774


namespace lisa_minimum_fifth_term_score_l46_46403

theorem lisa_minimum_fifth_term_score :
  ∀ (score1 score2 score3 score4 average_needed total_terms : ℕ),
  score1 = 84 →
  score2 = 80 →
  score3 = 82 →
  score4 = 87 →
  average_needed = 85 →
  total_terms = 5 →
  (∃ (score5 : ℕ), 
     (score1 + score2 + score3 + score4 + score5) / total_terms ≥ average_needed ∧ 
     score5 = 92) :=
by
  sorry

end lisa_minimum_fifth_term_score_l46_46403


namespace abs_inequality_solution_l46_46229

theorem abs_inequality_solution (x : ℝ) : |x + 2| + |x - 1| ≥ 5 ↔ x ≤ -3 ∨ x ≥ 2 :=
sorry

end abs_inequality_solution_l46_46229


namespace correct_proposition_is_B_l46_46019

variables {m n : Type} {α β : Type}

-- Define parallel and perpendicular relationships
def parallel (l₁ l₂ : Type) : Prop := sorry
def perpendicular (l₁ l₂ : Type) : Prop := sorry

def lies_in (l : Type) (p : Type) : Prop := sorry

-- The problem statement
theorem correct_proposition_is_B
  (H1 : perpendicular m α)
  (H2 : perpendicular n β)
  (H3 : perpendicular α β) :
  perpendicular m n :=
sorry

end correct_proposition_is_B_l46_46019


namespace max_vertex_sum_l46_46808

theorem max_vertex_sum (a T : ℤ) (hT : T ≠ 0)
  (h₁ : ∀ x, a * x * (x - 2 * T) = 0 → (0 = x ∨ 2 * T = x))
  (h₂ : ∀ x, (a * x^2 + (a * -2 * T) * x + 0 = a * x * (x - 2 * T)) ∧ ((T + 2) * (a * (T + 2 - 2 * T)) = 32))
  (N : ℤ) :
  (N = T - a * T^2) → max_vertex_sum = 68 :=
by
  sorry

end max_vertex_sum_l46_46808


namespace percentage_of_knives_is_40_l46_46449

theorem percentage_of_knives_is_40 
  (initial_knives : ℕ) (initial_forks : ℕ) (initial_spoons : ℕ) 
  (traded_knives : ℕ) (traded_spoons : ℕ) : 
  initial_knives = 6 → 
  initial_forks = 12 → 
  initial_spoons = 3 * initial_knives → 
  traded_knives = 10 → 
  traded_spoons = 6 → 
  let final_knives := initial_knives + traded_knives in
  let final_spoons := initial_spoons - traded_spoons in
  let total_silverware := final_knives + final_spoons + initial_forks in
  (final_knives : ℝ) / total_silverware * 100 = 40 :=
by sorry

end percentage_of_knives_is_40_l46_46449


namespace planet_combinations_count_l46_46977
  
theorem planet_combinations_count :
  ∃ a b : ℕ, a ≤ 5 ∧ b ≤ 9 ∧ 2 * a + b = 14 ∧ 
  (finset.card (finset.univ.choose a) * finset.card (finset.univ.choose b) =
  636) := sorry

end planet_combinations_count_l46_46977


namespace min_value_expression_l46_46353

variable (a b c : ℝ)
variable (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
variable (h_eq : a * b * c = 64)

theorem min_value_expression :
  a^2 + 6 * a * b + 9 * b^2 + 3 * c^2 ≥ 192 :=
by {
  sorry
}

end min_value_expression_l46_46353


namespace nick_charges_l46_46013

theorem nick_charges (y : ℕ) :
  let travel_cost := 7
  let hourly_rate := 10
  10 * y + 7 = travel_cost + hourly_rate * y :=
by sorry

end nick_charges_l46_46013


namespace value_of_n_l46_46657

-- Define required conditions
variables (n : ℕ) (f : ℕ → ℕ → ℕ)

-- Conditions
axiom cond1 : n > 7
axiom cond2 : ∀ m k : ℕ, f m k = 2^(n - m) * Nat.choose m k

-- Given condition
axiom after_seventh_round : f 7 5 = 42

-- Theorem to prove
theorem value_of_n : n = 8 :=
by
  -- Proof goes here
  sorry

end value_of_n_l46_46657


namespace consecutive_sums_permutations_iff_odd_l46_46795

theorem consecutive_sums_permutations_iff_odd (n : ℕ) (h : n ≥ 2) :
  (∃ (a b : Fin n → ℕ), (∀ i, 1 ≤ a i ∧ a i ≤ n) ∧ (∀ i, 1 ≤ b i ∧ b i ≤ n) ∧
    ∃ N, ∀ i, a i + b i = N + i) ↔ (Odd n) :=
by
  sorry

end consecutive_sums_permutations_iff_odd_l46_46795


namespace max_x_plus_y_range_y_plus_1_over_x_extrema_x2_minus_2x_plus_y2_plus_1_l46_46975

namespace Geometry

variables {x y : ℝ}

-- Given condition
def satisfies_circle (x y : ℝ) : Prop := x^2 + y^2 - 4 * y + 1 = 0

-- Proof problems
theorem max_x_plus_y (h : satisfies_circle x y) : 
  x + y ≤ 2 + Real.sqrt 6 :=
sorry

theorem range_y_plus_1_over_x (h : satisfies_circle x y) : 
  -Real.sqrt 2 ≤ (y + 1) / x ∧ (y + 1) / x ≤ Real.sqrt 2 :=
sorry

theorem extrema_x2_minus_2x_plus_y2_plus_1 (h : satisfies_circle x y) : 
  8 - 2 * Real.sqrt 15 ≤ x^2 - 2 * x + y^2 + 1 ∧ x^2 - 2 * x + y^2 + 1 ≤ 8 + 2 * Real.sqrt 15 :=
sorry

end Geometry

end max_x_plus_y_range_y_plus_1_over_x_extrema_x2_minus_2x_plus_y2_plus_1_l46_46975


namespace no_four_distinct_real_roots_l46_46254

theorem no_four_distinct_real_roots (a b : ℝ) :
  ¬ ∃ x1 x2 x3 x4 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 ∧ 
  (x1^4 - 4*x1^3 + 6*x1^2 + a*x1 + b = 0) ∧
  (x2^4 - 4*x2^3 + 6*x2^2 + a*x2 + b = 0) ∧
  (x3^4 - 4*x3^3 + 6*x3^2 + a*x3 + b = 0) ∧
  (x4^4 - 4*x4^3 + 6*x4^2 + a*x4 + b = 0) := 
by {
  sorry
}

end no_four_distinct_real_roots_l46_46254


namespace exradii_product_eq_area_squared_l46_46870

variable (a b c : ℝ) (t : ℝ)
variable (s := (a + b + c) / 2)
variable (exradius_a exradius_b exradius_c : ℝ)

-- Define the conditions
axiom Heron : t^2 = s * (s - a) * (s - b) * (s - c)
axiom exradius_definitions : exradius_a = t / (s - a) ∧ exradius_b = t / (s - b) ∧ exradius_c = t / (s - c)

-- The theorem we want to prove
theorem exradii_product_eq_area_squared : exradius_a * exradius_b * exradius_c = t^2 := sorry

end exradii_product_eq_area_squared_l46_46870


namespace geometric_sequence_eighth_term_l46_46086

variable (a r : ℕ)
variable (h1 : a = 3)
variable (h2 : a * r^6 = 2187)
variable (h3 : a = 3)

theorem geometric_sequence_eighth_term (a r : ℕ) (h1 : a = 3) (h2 : a * r^6 = 2187) (h3 : a = 3) :
  a * r^7 = 6561 := by
  sorry

end geometric_sequence_eighth_term_l46_46086


namespace largest_num_consecutive_integers_sum_45_l46_46263

theorem largest_num_consecutive_integers_sum_45 : 
  ∃ n : ℕ, (0 < n) ∧ (n * (n + 1) / 2 = 45) ∧ (∀ m : ℕ, (0 < m) → m * (m + 1) / 2 = 45 → m ≤ n) :=
by {
  sorry
}

end largest_num_consecutive_integers_sum_45_l46_46263


namespace sum_of_consecutive_integers_l46_46632

theorem sum_of_consecutive_integers {a b : ℤ} (h1 : a < b)
  (h2 : b = a + 1)
  (h3 : a < Real.sqrt 3)
  (h4 : Real.sqrt 3 < b) :
  a + b = 3 := 
sorry

end sum_of_consecutive_integers_l46_46632


namespace patch_area_difference_l46_46583

theorem patch_area_difference :
  let alan_length := 30
  let alan_width := 50
  let betty_length := 35
  let betty_width := 40
  let alan_area := alan_length * alan_width
  let betty_area := betty_length * betty_width
  let area_difference := alan_area - betty_area
  area_difference = 100 := 
by
  -- Definitions
  let alan_length := 30
  let alan_width := 50
  let betty_length := 35
  let betty_width := 40
  let alan_area := alan_length * alan_width
  let betty_area := betty_length * betty_width
  let area_difference := alan_area - betty_area
  -- Proof (intentionally left as sorry)
  -- Lean should be able to use the initial definitions to verify the theorem statement.
  sorry

end patch_area_difference_l46_46583


namespace find_a_and_union_l46_46624

open Set

theorem find_a_and_union (a : ℤ) (A B : Set ℤ) (hA : A = {2, 3, a ^ 2 + 4 * a + 2})
    (hB : B = {0, 7, 2 - a, a ^ 2 + 4 * a - 2}) (h_inter : A ∩ B = {3, 7}) :
    a = 1 ∧ A ∪ B = {0, 1, 2, 3, 7} := by
  -- Proof to be provided
  sorry

end find_a_and_union_l46_46624


namespace value_of_f_eval_at_pi_over_12_l46_46622

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

theorem value_of_f_eval_at_pi_over_12 : f (Real.pi / 12) = (Real.sqrt 6) / 2 :=
by
  sorry

end value_of_f_eval_at_pi_over_12_l46_46622


namespace route_inequality_l46_46435

noncomputable def f (m n : ℕ) : ℕ :=
-- Definition of f based on the problem context which could be obtained
-- Possibly using binomial coefficients or simple combinatorial properties

theorem route_inequality (m n : ℕ) : f(m, n) ≤ 2^(m * n) := sorry

end route_inequality_l46_46435


namespace red_to_green_speed_ratio_l46_46108

-- Conditions
def blue_car_speed : Nat := 80 -- The blue car's speed is 80 miles per hour
def green_car_speed : Nat := 8 * blue_car_speed -- The green car's speed is 8 times the blue car's speed
def red_car_speed : Nat := 1280 -- The red car's speed is 1280 miles per hour

-- Theorem stating the ratio of red car's speed to green car's speed
theorem red_to_green_speed_ratio : red_car_speed / green_car_speed = 2 := by
  sorry -- proof goes here

end red_to_green_speed_ratio_l46_46108


namespace factorize_expression_l46_46257

variable (a b : ℝ)

theorem factorize_expression : a^2 - 4 * b^2 - 2 * a + 4 * b = (a + 2 * b - 2) * (a - 2 * b) := 
  sorry

end factorize_expression_l46_46257


namespace no_integral_points_on_AB_l46_46818

theorem no_integral_points_on_AB (k m n : ℤ) (h1: ((m^3 - m)^2 + (n^3 - n)^2 > (3*k + 1)^2)) :
  ¬ ∃ (x y : ℤ), (m^3 - m) * x + (n^3 - n) * y = (3*k + 1)^2 :=
by {
  sorry
}

end no_integral_points_on_AB_l46_46818


namespace greater_number_is_33_l46_46196

theorem greater_number_is_33 (A B : ℕ) (hcf_11 : Nat.gcd A B = 11) (product_363 : A * B = 363) :
  max A B = 33 :=
by
  sorry

end greater_number_is_33_l46_46196


namespace sum_of_consecutive_integers_l46_46633

theorem sum_of_consecutive_integers {a b : ℤ} (h1 : a < b)
  (h2 : b = a + 1)
  (h3 : a < Real.sqrt 3)
  (h4 : Real.sqrt 3 < b) :
  a + b = 3 := 
sorry

end sum_of_consecutive_integers_l46_46633


namespace reciprocal_of_minus_one_over_2023_l46_46044

theorem reciprocal_of_minus_one_over_2023 : (1 / (- (1 / 2023))) = -2023 := 
by
  sorry

end reciprocal_of_minus_one_over_2023_l46_46044


namespace consecutive_sum_l46_46630

theorem consecutive_sum (a b : ℤ) (h1 : a + 1 = b) (h2 : (a : ℝ) < real.sqrt 3) (h3 : real.sqrt 3 < (b : ℝ)) : a + b = 3 := 
sorry

end consecutive_sum_l46_46630


namespace average_interest_rate_l46_46096

theorem average_interest_rate (I : ℝ) (r1 r2 : ℝ) (y : ℝ)
  (h0 : I = 6000)
  (h1 : r1 = 0.05)
  (h2 : r2 = 0.07)
  (h3 : 0.05 * (6000 - y) = 0.07 * y) :
  ((r1 * (I - y) + r2 * y) / I) = 0.05833 :=
by
  sorry

end average_interest_rate_l46_46096


namespace proof_solution_l46_46831

noncomputable def proof_problem : Prop :=
  ∀ (x y z : ℝ), 3 * x - 4 * y - 2 * z = 0 ∧ x - 2 * y - 8 * z = 0 ∧ z ≠ 0 → 
  (x^2 + 3 * x * y) / (y^2 + z^2) = 329 / 61

theorem proof_solution : proof_problem :=
by
  intros x y z h
  sorry

end proof_solution_l46_46831


namespace length_of_longer_leg_of_smallest_triangle_l46_46114

theorem length_of_longer_leg_of_smallest_triangle :
  ∀ (a b c : ℝ), 
  is_30_60_90_triangle (a, b, c) ∧ c = 16 
  ∧ (∀ (a₁ b₁ c₁ : ℝ), is_30_60_90_triangle (a₁, b₁, c₁) → b = c₁ → true) 
  ∧ (∀ (a₂ b₂ c₂ : ℝ), is_30_60_90_triangle (a₂, b₂, c₂) → true) 
  ∧ (∀ (a₃ b₃ c₃ : ℝ), is_30_60_90_triangle (a₃, b₃, c₃) → true) 
  → ∃ (a₄ b₄ c₄ : ℝ), is_30_60_90_triangle (a₄, b₄, c₄) ∧ b₄ = 9 :=
sorry

end length_of_longer_leg_of_smallest_triangle_l46_46114


namespace sum_of_consecutive_integers_l46_46644

theorem sum_of_consecutive_integers (a b : ℤ) (h1 : a + 1 = b) (h2 : a < real.sqrt 3) (h3 : real.sqrt 3 < b) : a + b = 3 :=
sorry

end sum_of_consecutive_integers_l46_46644


namespace find_two_digit_integers_l46_46027

theorem find_two_digit_integers :
  ∃ (m n : ℕ), 10 ≤ m ∧ m < 100 ∧ 10 ≤ n ∧ n < 100 ∧
    (∃ (a b : ℚ), a = m ∧ b = n ∧ (a + b) / 2 = b + a / 100) ∧ (m + n < 150) ∧ m = 50 ∧ n = 49 := 
by
  sorry

end find_two_digit_integers_l46_46027


namespace reciprocal_is_correct_l46_46049

-- Define the initial number
def num : ℚ := -1 / 2023

-- Define the expected reciprocal
def reciprocal : ℚ := -2023

-- Theorem stating the reciprocal of the given number is the expected reciprocal
theorem reciprocal_is_correct : 1 / num = reciprocal :=
  by
    -- The actual proof can be filled in here
    sorry

end reciprocal_is_correct_l46_46049


namespace initial_walking_rate_proof_l46_46916

noncomputable def initial_walking_rate (d : ℝ) (v_miss : ℝ) (t_miss : ℝ) (v_early : ℝ) (t_early : ℝ) : ℝ :=
  d / ((d / v_early) + t_early - t_miss)

theorem initial_walking_rate_proof :
  initial_walking_rate 6 5 (7/60) 6 (5/60) = 5 := by
  sorry

end initial_walking_rate_proof_l46_46916


namespace perfect_square_for_n_l46_46363

theorem perfect_square_for_n 
  (a b : ℕ)
  (h1 : ∃ x : ℕ, ab = x^2)
  (h2 : ∃ y : ℕ, (a + 1) * (b + 1) = y^2) 
  : ∃ n : ℕ, n > 1 ∧ ∃ z : ℕ, (a + n) * (b + n) = z^2 :=
by
  let n := ab
  have h3 : n > 1 := sorry
  have h4 : ∃ z : ℕ, (a + n) * (b + n) = z^2 := sorry
  exact ⟨n, h3, h4⟩

end perfect_square_for_n_l46_46363


namespace find_expression_for_f_x_neg_l46_46967

theorem find_expression_for_f_x_neg (f : ℝ → ℝ) 
  (h_odd : ∀ x, f (-x) = -f x) 
  (h_pos : ∀ x, 0 < x → f x = x - Real.log (abs x)) :
  ∀ x, x < 0 → f x = x + Real.log (abs x) :=
by
  sorry

end find_expression_for_f_x_neg_l46_46967


namespace find_d_l46_46390

-- Given conditions
def line_eq (x y : ℚ) : Prop := y = (3 * x - 4) / 4

def parametrized_eq (v d : ℚ × ℚ) (t x y : ℚ) : Prop :=
  (x, y) = (v.1 + t * d.1, v.2 + t * d.2)

def distance_eq (x y : ℚ) (t : ℚ) : Prop :=
  (x - 3) * (x - 3) + (y - 1) * (y - 1) = t * t

-- The proof problem statement
theorem find_d (d : ℚ × ℚ) 
  (h_d : d = (7/2, 5/2)) :
  ∀ (x y t : ℚ) (v : ℚ × ℚ) (h_v : v = (3, 1)),
    (x ≥ 3) → 
    line_eq x y → 
    parametrized_eq v d t x y → 
    distance_eq x y t → 
    d = (7/2, 5/2) := 
by 
  intros;
  sorry


end find_d_l46_46390


namespace evening_sales_l46_46184

theorem evening_sales
  (remy_bottles_morning : ℕ := 55)
  (nick_bottles_fewer : ℕ := 6)
  (price_per_bottle : ℚ := 0.50)
  (evening_sales_more : ℚ := 3) :
  let nick_bottles_morning := remy_bottles_morning - nick_bottles_fewer
  let remy_sales_morning := remy_bottles_morning * price_per_bottle
  let nick_sales_morning := nick_bottles_morning * price_per_bottle
  let total_morning_sales := remy_sales_morning + nick_sales_morning
  let total_evening_sales := total_morning_sales + evening_sales_more
  total_evening_sales = 55 :=
by
  sorry

end evening_sales_l46_46184


namespace south_side_students_count_l46_46204

variables (N : ℕ)
def students_total := 41
def difference := 3

theorem south_side_students_count (N : ℕ) (h₁ : 2 * N + difference = students_total) : N + difference = 22 :=
sorry

end south_side_students_count_l46_46204


namespace expected_value_and_variance_of_X1_l46_46352

noncomputable def X : Type → ℝ := sorry -- a discrete random variable
axiom E_of_X (Ω : Type) : ℝ -- expected value of X
axiom D_of_X (Ω : Type) : ℝ -- variance of X

-- Given conditions
axiom ex_X (Ω : Type) : E_of_X Ω = 6 -- E(X) = 6
axiom var_X (Ω : Type) : D_of_X Ω = 0.5 -- D(X) = 0.5

-- Definitions
noncomputable def X1 (Ω : Type) : ℝ := 2 * X Ω - 5

-- Proof goal
theorem expected_value_and_variance_of_X1 (Ω : Type) : 
  E_of_X1 Ω = 7 ∧ D_of_X1 Ω = 2 :=
by
  -- use the linearity properties of expectation and variance
  sorry

end expected_value_and_variance_of_X1_l46_46352


namespace exists_x_quadratic_eq_zero_iff_le_one_l46_46180

variable (a : ℝ)

theorem exists_x_quadratic_eq_zero_iff_le_one : (∃ x : ℝ, x^2 - 2 * x + a = 0) ↔ a ≤ 1 :=
sorry

end exists_x_quadratic_eq_zero_iff_le_one_l46_46180


namespace distance_between_foci_of_ellipse_l46_46589

theorem distance_between_foci_of_ellipse :
  let c := (5, 2)
  let a := 5
  let b := 2
  2 * Real.sqrt (a^2 - b^2) = 2 * Real.sqrt 21 :=
by
  let c := (5, 2)
  let a := 5
  let b := 2
  show 2 * Real.sqrt (a^2 - b^2) = 2 * Real.sqrt 21
  sorry

end distance_between_foci_of_ellipse_l46_46589


namespace find_x_l46_46674

-- Introducing the main theorem
theorem find_x (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) (x : ℝ) (h_x : 0 < x) : 
  let r := (4 * a) ^ (4 * b)
  let y := x ^ 2
  r = a ^ b * y → 
  x = 16 ^ b * a ^ (1.5 * b) :=
by
  sorry

end find_x_l46_46674


namespace average_speed_trip_l46_46413

theorem average_speed_trip 
  (total_distance : ℕ)
  (first_distance : ℕ)
  (first_speed : ℕ)
  (second_distance : ℕ)
  (second_speed : ℕ)
  (h1 : total_distance = 60)
  (h2 : first_distance = 30)
  (h3 : first_speed = 60)
  (h4 : second_distance = 30)
  (h5 : second_speed = 30) :
  40 = total_distance / ((first_distance / first_speed) + (second_distance / second_speed)) :=
by sorry

end average_speed_trip_l46_46413


namespace find_x_for_divisibility_18_l46_46268

theorem find_x_for_divisibility_18 (x : ℕ) (h_digits : x < 10) :
  (1001 * x + 150) % 18 = 0 ↔ x = 6 :=
by
  sorry

end find_x_for_divisibility_18_l46_46268


namespace four_digit_multiples_of_five_count_l46_46323

theorem four_digit_multiples_of_five_count : 
  {n : ℕ | 1000 ≤ n ∧ n ≤ 9999 ∧ n % 5 = 0}.toFinset.card = 1800 :=
by
  sorry

end four_digit_multiples_of_five_count_l46_46323


namespace sum_of_consecutive_integers_l46_46634

theorem sum_of_consecutive_integers {a b : ℤ} (h1 : a < b)
  (h2 : b = a + 1)
  (h3 : a < Real.sqrt 3)
  (h4 : Real.sqrt 3 < b) :
  a + b = 3 := 
sorry

end sum_of_consecutive_integers_l46_46634


namespace Mo_tea_cups_l46_46179

theorem Mo_tea_cups (n t : ℕ) 
  (h1 : 2 * n + 5 * t = 36)
  (h2 : 5 * t = 2 * n + 14) : 
  t = 5 :=
by
  sorry

end Mo_tea_cups_l46_46179


namespace combined_tax_rate_is_correct_l46_46171

noncomputable def combined_tax_rate (john_income : ℝ) (ingrid_income : ℝ) (john_tax_rate : ℝ) (ingrid_tax_rate : ℝ) : ℝ :=
  let john_tax := john_tax_rate * john_income
  let ingrid_tax := ingrid_tax_rate * ingrid_income
  let total_tax := john_tax + ingrid_tax
  let total_income := john_income + ingrid_income
  total_tax / total_income

theorem combined_tax_rate_is_correct :
  combined_tax_rate 56000 72000 0.30 0.40 = 0.35625 := 
by
  sorry

end combined_tax_rate_is_correct_l46_46171


namespace condition1_condition2_condition3_l46_46231

-- Condition 1 statement
theorem condition1: (number_of_ways_condition1 : ℕ) = 5520 := by
  -- Expected proof that number_of_ways_condition1 = 5520
  sorry

-- Condition 2 statement
theorem condition2: (number_of_ways_condition2 : ℕ) = 3360 := by
  -- Expected proof that number_of_ways_condition2 = 3360
  sorry

-- Condition 3 statement
theorem condition3: (number_of_ways_condition3 : ℕ) = 360 := by
  -- Expected proof that number_of_ways_condition3 = 360
  sorry

end condition1_condition2_condition3_l46_46231


namespace total_letters_sent_l46_46521

def letters_January : ℕ := 6
def letters_February : ℕ := 9
def letters_March : ℕ := 3 * letters_January

theorem total_letters_sent : letters_January + letters_February + letters_March = 33 :=
by
  -- This is where the proof would go
  sorry

end total_letters_sent_l46_46521


namespace quadratic_rewrite_l46_46883

theorem quadratic_rewrite :
  ∃ a d : ℤ, (∀ x : ℝ, x^2 + 500 * x + 2500 = (x + a)^2 + d) ∧ (d / a) = -240 := by
  sorry

end quadratic_rewrite_l46_46883


namespace john_needs_29_planks_for_house_wall_l46_46809

def total_number_of_planks (large_planks small_planks : ℕ) : ℕ :=
  large_planks + small_planks

theorem john_needs_29_planks_for_house_wall :
  total_number_of_planks 12 17 = 29 :=
by
  sorry

end john_needs_29_planks_for_house_wall_l46_46809


namespace four_digit_multiples_of_five_count_l46_46321

theorem four_digit_multiples_of_five_count : 
  {n : ℕ | 1000 ≤ n ∧ n ≤ 9999 ∧ n % 5 = 0}.toFinset.card = 1800 :=
by
  sorry

end four_digit_multiples_of_five_count_l46_46321


namespace four_digit_multiples_of_5_l46_46319

theorem four_digit_multiples_of_5 : 
  let f := fun n => 1000 <= n ∧ n <= 9999 ∧ n % 5 = 0
  in finset.card (finset.filter f (finset.range 10000)) = 1800 :=
by
  sorry

end four_digit_multiples_of_5_l46_46319


namespace cubic_identity_l46_46653

theorem cubic_identity (x : ℝ) (hx : x + 1/x = -5) : x^3 + 1/x^3 = -110 := by
  sorry

end cubic_identity_l46_46653


namespace evaluate_expression_l46_46283

variable (a b c : ℝ)

theorem evaluate_expression 
  (h : a / (20 - a) + b / (75 - b) + c / (55 - c) = 8) :
  4 / (20 - a) + 15 / (75 - b) + 11 / (55 - c) = 8.8 :=
sorry

end evaluate_expression_l46_46283


namespace movies_left_to_watch_l46_46396

theorem movies_left_to_watch (total_movies watched_movies : Nat) (h_total : total_movies = 12) (h_watched : watched_movies = 6) : total_movies - watched_movies = 6 :=
by
  sorry

end movies_left_to_watch_l46_46396


namespace count_four_digit_multiples_of_5_l46_46308

def is_multiple_of_5 (n : ℕ) : Prop :=
  n % 5 = 0

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def four_digit_multiples_of_5_count (lower upper : ℕ) :=
  (upper - lower + 1)

theorem count_four_digit_multiples_of_5 : 
  four_digit_multiples_of_5_count 200 1999 = 1800 :=
by
  sorry

end count_four_digit_multiples_of_5_l46_46308


namespace four_digit_multiples_of_5_count_l46_46313

theorem four_digit_multiples_of_5_count :
  let lower_bound := 200
  let upper_bound := 1999
  (upper_bound - lower_bound + 1) = 1800 :=
by
  let lower_bound := 200
  let upper_bound := 1999
  show (upper_bound - lower_bound + 1) = 1800,
  from sorry

end four_digit_multiples_of_5_count_l46_46313


namespace total_necklaces_made_l46_46946

-- Definitions based on conditions
def first_machine_necklaces : ℝ := 45
def second_machine_necklaces : ℝ := 2.4 * first_machine_necklaces

-- Proof statement
theorem total_necklaces_made : (first_machine_necklaces + second_machine_necklaces) = 153 := by
  sorry

end total_necklaces_made_l46_46946


namespace root_relation_l46_46524

theorem root_relation (a b x y : ℝ)
  (h1 : x + y = a)
  (h2 : (1 / x) + (1 / y) = 1 / b)
  (h3 : x = 3 * y)
  (h4 : y = a / 4) :
  b = 3 * a / 16 :=
by
  sorry

end root_relation_l46_46524


namespace find_X_sum_coordinates_l46_46493

/- Define points and their coordinates -/
variables (X Y Z : ℝ × ℝ)
variable  (XY XZ ZY : ℝ)
variable  (k : ℝ)
variable  (hxz : XZ = (3/4) * XY)
variable  (hzy : ZY = (1/4) * XY)
variable  (hy : Y = (2, 9))
variable  (hz : Z = (1, 5))

/-- Lean 4 statement for the proof problem -/
theorem find_X_sum_coordinates :
  (Y.1 = 2) ∧ (Y.2 = 9) ∧ (Z.1 = 1) ∧ (Z.2 = 5) ∧
  XZ = (3/4) * XY ∧ ZY = (1/4) * XY →
  (X.1 + X.2) = -9 := 
by
  sorry

end find_X_sum_coordinates_l46_46493


namespace curve_transformation_l46_46442

-- Define the scaling transformation
def scaling_transform (x y : ℝ) : ℝ × ℝ :=
  (5 * x, 3 * y)

-- Define the transformed curve
def transformed_curve (x' y' : ℝ) : Prop :=
  2 * x' ^ 2 + 8 * y' ^ 2 = 1

-- Define the curve C's equation after scaling
def curve_C (x y : ℝ) : Prop :=
  50 * x ^ 2 + 72 * y ^ 2 = 1

-- Statement of the proof problem
theorem curve_transformation (x y : ℝ) (h : transformed_curve (5 * x) (3 * y)) : curve_C x y :=
by {
  -- The actual proof would be filled in here
  sorry
}

end curve_transformation_l46_46442


namespace tom_average_speed_l46_46000

theorem tom_average_speed 
  (karen_speed : ℕ) (tom_distance : ℕ) (karen_advantage : ℕ) (delay : ℚ)
  (h1 : karen_speed = 60)
  (h2 : tom_distance = 24)
  (h3 : karen_advantage = 4)
  (h4 : delay = 4/60) :
  ∃ (v : ℚ), v = 45 := by
  sorry

end tom_average_speed_l46_46000


namespace percentage_caught_sampling_candy_l46_46844

theorem percentage_caught_sampling_candy
  (S : ℝ) (C : ℝ)
  (h1 : 0.1 * S = 0.1 * 24.444444444444443) -- 10% of the customers who sample the candy are not caught
  (h2 : S = 24.444444444444443)  -- The total percent of all customers who sample candy is 24.444444444444443%
  :
  C = 0.9 * 24.444444444444443 := -- Equivalent \( C \approx 22 \% \)
by
  sorry

end percentage_caught_sampling_candy_l46_46844


namespace fault_line_movement_l46_46934

theorem fault_line_movement (total_movement: ℝ) (past_year: ℝ) (prev_year: ℝ) (total_eq: total_movement = 6.5) (past_eq: past_year = 1.25) :
  prev_year = 5.25 := by
  sorry

end fault_line_movement_l46_46934


namespace charlie_share_l46_46071

theorem charlie_share (A B C D E : ℝ) (h1 : A = (1/3) * B)
  (h2 : B = (1/2) * C) (h3 : C = 0.75 * D) (h4 : D = 2 * E) 
  (h5 : A + B + C + D + E = 15000) : C = 15000 * (3 / 11) :=
by
  sorry

end charlie_share_l46_46071


namespace dartboard_distribution_count_l46_46594

-- Definition of the problem in Lean 4
def count_dartboard_distributions : ℕ :=
  -- We directly use the identified correct answer
  5

theorem dartboard_distribution_count :
  count_dartboard_distributions = 5 :=
sorry

end dartboard_distribution_count_l46_46594


namespace sam_and_erica_money_total_l46_46185

def sam_money : ℕ := 38
def erica_money : ℕ := 53

theorem sam_and_erica_money_total : sam_money + erica_money = 91 :=
by
  -- the proof is not required; hence we skip it
  sorry

end sam_and_erica_money_total_l46_46185


namespace circle_not_pass_second_quadrant_l46_46074

theorem circle_not_pass_second_quadrant (a : ℝ) : ¬(∃ x y : ℝ, x < 0 ∧ y > 0 ∧ (x - a)^2 + y^2 = 4) → a ≥ 2 :=
by
  intro h
  by_contra
  sorry

end circle_not_pass_second_quadrant_l46_46074


namespace consecutive_sum_l46_46628

theorem consecutive_sum (a b : ℤ) (h1 : a + 1 = b) (h2 : (a : ℝ) < real.sqrt 3) (h3 : real.sqrt 3 < (b : ℝ)) : a + b = 3 := 
sorry

end consecutive_sum_l46_46628


namespace probability_no_gp_l46_46928

/-- 
Alice has six magical pies: Two are growth pies (GP), and four are shrink pies (SP).
Alice randomly picks three pies out of the six and gives them to Mary. We want to find the 
probability that one of the girls does not have a single growth pie (GP).
-/
theorem probability_no_gp : 
  let total_pies := 6
  let gp := 2 -- number of growth pies
  let sp := 4 -- number of shrink pies
  let chosen_pies := 3 -- pies given to Mary
  (let total_ways := Nat.choose total_pies chosen_pies in -- total ways to choose 3 out of 6
  let favorable_ways := Nat.choose sp 2 in -- ways to choose 2 SPs out of 4 (ensuring both have at least one GP)
  (total_ways - favorable_ways) / total_ways = (7 / 10 : ℚ)) :=
  sorry

end probability_no_gp_l46_46928


namespace maximum_triangles_in_right_angle_triangle_l46_46680

-- Definition of grid size and right-angled triangle on graph paper
def grid_size : Nat := 7

-- Definition of the vertices of the right-angled triangle
def vertices : List (Nat × Nat) := [(0,0), (grid_size,0), (0,grid_size)]

-- Total number of unique triangles that can be identified
theorem maximum_triangles_in_right_angle_triangle (grid_size : Nat) (vertices : List (Nat × Nat)) : 
  Nat :=
  if vertices = [(0,0), (grid_size,0), (0,grid_size)] then 28 else 0

end maximum_triangles_in_right_angle_triangle_l46_46680


namespace carls_membership_number_l46_46129

-- Definitions for the conditions
def is_two_digit_prime (n : ℕ) : Prop := 10 ≤ n ∧ n < 100 ∧ Nat.Prime n

variable (a b c d : ℕ)

-- Stating the conditions
hypothesis h1 : is_two_digit_prime a
hypothesis h2 : is_two_digit_prime b
hypothesis h3 : is_two_digit_prime c
hypothesis h4 : is_two_digit_prime d
hypothesis h_sum_all : a + b + c + d = 100
hypothesis h_sum_birthday_ben : a + c + d = 30
hypothesis h_sum_birthday_carl : a + b + d = 29
hypothesis h_sum_birthday_david : a + b + c = 23

-- Proposition to prove
theorem carls_membership_number : c = 23 :=
by {
    sorry  -- Proof is not required as per the instructions
}

end carls_membership_number_l46_46129


namespace can_lid_boxes_count_l46_46097

theorem can_lid_boxes_count 
  (x y : ℕ) 
  (h1 : 3 * x + y + 14 = 75) : 
  x = 20 ∧ y = 1 :=
by 
  sorry

end can_lid_boxes_count_l46_46097


namespace average_weight_of_remaining_carrots_l46_46075

noncomputable def total_weight_30_carrots : ℕ := 5940
noncomputable def total_weight_3_carrots : ℕ := 540
noncomputable def carrots_count_30 : ℕ := 30
noncomputable def carrots_count_3_removed : ℕ := 3
noncomputable def carrots_count_remaining : ℕ := 27
noncomputable def average_weight_of_removed_carrots : ℕ := 180

theorem average_weight_of_remaining_carrots :
  (total_weight_30_carrots - total_weight_3_carrots) / carrots_count_remaining = 200 :=
  by
  sorry

end average_weight_of_remaining_carrots_l46_46075


namespace positive_value_of_A_l46_46499

theorem positive_value_of_A (A : ℝ) :
  (A ^ 2 + 7 ^ 2 = 200) → A = Real.sqrt 151 :=
by
  intros h
  sorry

end positive_value_of_A_l46_46499


namespace sphere_to_cube_volume_ratio_l46_46580

noncomputable def volume_ratio (s : ℝ) : ℝ :=
  let r := s / 4
  let V_s := (4/3:ℝ) * Real.pi * r^3 
  let V_c := s^3
  V_s / V_c

theorem sphere_to_cube_volume_ratio (s : ℝ) (h : s > 0) : volume_ratio s = Real.pi / 48 := by
  sorry

end sphere_to_cube_volume_ratio_l46_46580


namespace find_initial_cookies_l46_46867

-- Definitions based on problem conditions
def initial_cookies (x : ℕ) : Prop :=
  let after_eating := x - 2
  let after_buying := after_eating + 37
  after_buying = 75

-- Main statement to be proved
theorem find_initial_cookies : ∃ x, initial_cookies x ∧ x = 40 :=
by
  sorry

end find_initial_cookies_l46_46867


namespace consecutive_sum_l46_46629

theorem consecutive_sum (a b : ℤ) (h1 : a + 1 = b) (h2 : (a : ℝ) < real.sqrt 3) (h3 : real.sqrt 3 < (b : ℝ)) : a + b = 3 := 
sorry

end consecutive_sum_l46_46629


namespace part1_part2_part3_l46_46504

noncomputable def f (a x : ℝ) : ℝ := Real.log (x + 1) + a * (x^2 - x)

theorem part1 (x : ℝ) (hx : 0 < x) : f 0 x < x := by sorry

theorem part2 (a x : ℝ) :
  (0 ≤ a ∧ a ≤ 8/9 → 0 = 0) ∧
  (a > 8/9 → 2 = 2) ∧
  (a < 0 → 1 = 1) := by sorry

theorem part3 (a : ℝ) (h : ∀ x > 0, f a x ≥ 0) : 0 ≤ a ∧ a ≤ 1 := by sorry

end part1_part2_part3_l46_46504


namespace systematic_sampling_l46_46094

theorem systematic_sampling (total_employees groups group_size draw_5th draw_10th : ℕ)
  (h1 : total_employees = 200)
  (h2 : groups = 40)
  (h3 : group_size = total_employees / groups)
  (h4 : draw_5th = 22)
  (h5 : ∃ x : ℕ, draw_5th = (5-1) * group_size + x)
  (h6 : ∀ k : ℕ, 1 ≤ k ∧ k ≤ groups → draw_10th = (k-1) * group_size + x) :
  draw_10th = 47 := 
by
  sorry

end systematic_sampling_l46_46094


namespace impossible_circular_arrangement_1_to_60_l46_46072

theorem impossible_circular_arrangement_1_to_60 :
  (∀ (f : ℕ → ℕ), 
      (∀ n, 1 ≤ f n ∧ f n ≤ 60) ∧ 
      (∀ n, f (n + 2) + f n ≡ 0 [MOD 2]) ∧ 
      (∀ n, f (n + 3) + f n ≡ 0 [MOD 3]) ∧ 
      (∀ n, f (n + 7) + f n ≡ 0 [MOD 7]) 
      → false) := 
  sorry

end impossible_circular_arrangement_1_to_60_l46_46072


namespace frog_reaches_top_l46_46574

theorem frog_reaches_top (x : ℕ) (h1 : ∀ d ≤ x - 1, 3 * d + 5 ≥ 50) : x = 16 := by
  sorry

end frog_reaches_top_l46_46574


namespace sum_of_consecutive_integers_l46_46647

theorem sum_of_consecutive_integers (a b : ℤ) (h1 : a + 1 = b) (h2 : a < real.sqrt 3) (h3 : real.sqrt 3 < b) : a + b = 3 :=
sorry

end sum_of_consecutive_integers_l46_46647


namespace area_at_stage_8_l46_46332

-- Defining the constants and initial settings
def first_term : ℕ := 1
def common_difference : ℕ := 1
def stage : ℕ := 8
def square_side_length : ℕ := 4

-- Calculating the number of squares at the given stage
def num_squares : ℕ := first_term + (stage - 1) * common_difference

--Calculating the area of one square
def area_one_square : ℕ := square_side_length * square_side_length

-- Calculating the total area at the given stage
def total_area : ℕ := num_squares * area_one_square

-- Proving the total area equals 128 at Stage 8
theorem area_at_stage_8 : total_area = 128 := 
by
  sorry

end area_at_stage_8_l46_46332


namespace prob_at_least_6_heads_eq_l46_46752

-- define the number of coin flips
def n := 8

-- define the number of possible outcomes (2^n)
def total_outcomes := 2 ^ n

-- define the binomial coefficients for cases: 6 heads, 7 heads, 8 heads
def binom_8_6 := Nat.choose 8 6
def binom_8_7 := Nat.choose 8 7
def binom_8_8 := Nat.choose 8 8

-- calculate the favorable outcomes for at least 6 heads
def favorable_outcomes := binom_8_6 + binom_8_7 + binom_8_8

-- define the probability of getting at least 6 heads
def probability := (favorable_outcomes : ℚ) / total_outcomes

theorem prob_at_least_6_heads_eq : probability = 37 / 256 := by
  sorry

end prob_at_least_6_heads_eq_l46_46752


namespace find_tan_beta_l46_46957

variable (α β : ℝ)

def condition1 : Prop := Real.tan α = 3
def condition2 : Prop := Real.tan (α + β) = 2

theorem find_tan_beta (h1 : condition1 α) (h2 : condition2 α β) : Real.tan β = -1 / 7 := 
by {
  sorry
}

end find_tan_beta_l46_46957


namespace lcm_of_ratio_and_hcf_l46_46884

theorem lcm_of_ratio_and_hcf (a b : ℕ) (h1 : a = 3 * 8) (h2 : b = 4 * 8) (h3 : Nat.gcd a b = 8) : Nat.lcm a b = 96 :=
  sorry

end lcm_of_ratio_and_hcf_l46_46884


namespace tickets_left_l46_46565

-- Definitions for the conditions given in the problem
def tickets_whack_a_mole : ℕ := 32
def tickets_skee_ball : ℕ := 25
def tickets_spent_on_hat : ℕ := 7

-- The main proof statement to verify
theorem tickets_left : tickets_whack_a_mole + tickets_skee_ball - tickets_spent_on_hat = 50 :=
by
  sorry

end tickets_left_l46_46565


namespace fulfill_customer_order_in_nights_l46_46407

structure JerkyCompany where
  batch_size : ℕ
  nightly_batches : ℕ

def customerOrder (ordered : ℕ) (current_stock : ℕ) : ℕ :=
  ordered - current_stock

def batchesNeeded (required : ℕ) (batch_size : ℕ) : ℕ :=
  required / batch_size

def daysNeeded (batches_needed : ℕ) (nightly_batches : ℕ) : ℕ :=
  batches_needed / nightly_batches

theorem fulfill_customer_order_in_nights :
  ∀ (ordered current_stock : ℕ) (jc : JerkyCompany),
    jc.batch_size = 10 →
    jc.nightly_batches = 1 →
    ordered = 60 →
    current_stock = 20 →
    daysNeeded (batchesNeeded (customerOrder ordered current_stock) jc.batch_size) jc.nightly_batches = 4 :=
by
  intros ordered current_stock jc h1 h2 h3 h4
  sorry

end fulfill_customer_order_in_nights_l46_46407


namespace max_value_of_function_in_interval_l46_46264

open Real

noncomputable def my_function : ℝ → ℝ := λ x, 2 * x ^ 3 - 3 * x ^ 2

theorem max_value_of_function_in_interval :
  ∃ x ∈ Icc (-1 : ℝ) (2 : ℝ), ∀ y ∈ Icc (-1 : ℝ) (2 : ℝ), my_function x ≥ my_function y ∧ my_function x = 4 :=
by
  sorry

end max_value_of_function_in_interval_l46_46264


namespace board_partition_possible_l46_46426

variable (m n : ℕ)

theorem board_partition_possible (hm : m > 15) (hn : n > 15) :
  ((∃ k1, m = 5 * k1 ∧ ∃ k2, n = 4 * k2) ∨ (∃ k3, m = 4 * k3 ∧ ∃ k4, n = 5 * k4)) :=
sorry

end board_partition_possible_l46_46426


namespace four_digit_multiples_of_five_count_l46_46322

theorem four_digit_multiples_of_five_count : 
  {n : ℕ | 1000 ≤ n ∧ n ≤ 9999 ∧ n % 5 = 0}.toFinset.card = 1800 :=
by
  sorry

end four_digit_multiples_of_five_count_l46_46322


namespace determine_x_l46_46687

theorem determine_x (A B C : ℝ) (x : ℝ) (h1 : C > B) (h2 : B > A) (h3 : A > 0)
  (h4 : A = B - (x / 100) * B) (h5 : C = A + 2 * B) :
  x = 100 * ((B - A) / B) :=
sorry

end determine_x_l46_46687


namespace elena_butter_l46_46457

theorem elena_butter (cups_flour butter : ℕ) (h1 : cups_flour * 4 = 28) (h2 : butter * 4 = 12) : butter = 3 := 
by
  sorry

end elena_butter_l46_46457


namespace efficiency_ratio_l46_46905

theorem efficiency_ratio (E_A E_B : ℝ) 
  (h1 : E_B = 1 / 18) 
  (h2 : E_A + E_B = 1 / 6) : 
  E_A / E_B = 2 :=
by {
  sorry
}

end efficiency_ratio_l46_46905


namespace jenny_cat_expense_l46_46170

def adoption_fee : ℕ := 50
def vet_visits_cost : ℕ := 500
def monthly_food_cost : ℕ := 25
def jenny_toy_expenses : ℕ := 200
def split_factor : ℕ := 2

-- Given conditions, prove that Jenny spent $625 on the cat in the first year.
theorem jenny_cat_expense : 
  let yearly_food_cost := 12 * monthly_food_cost 
  let total_shared_expenses := adoption_fee + vet_visits_cost + yearly_food_cost 
  let jenny_shared_expenses := total_shared_expenses / split_factor 
  let total_jenny_cost := jenny_shared_expenses + jenny_toy_expenses
  in total_jenny_cost = 625 := 
by 
  sorry

end jenny_cat_expense_l46_46170


namespace find_n_that_satisfies_l46_46953

theorem find_n_that_satisfies :
  ∃ (n : ℕ), (1 / (n + 2 : ℕ) + 2 / (n + 2) + (n + 1) / (n + 2) = 2) ∧ (n = 0) :=
by 
  existsi (0 : ℕ)
  sorry

end find_n_that_satisfies_l46_46953


namespace remainder_div_38_l46_46152

theorem remainder_div_38 (n : ℕ) (h : n = 432 * 44) : n % 38 = 32 :=
sorry

end remainder_div_38_l46_46152


namespace reciprocal_is_correct_l46_46048

-- Define the initial number
def num : ℚ := -1 / 2023

-- Define the expected reciprocal
def reciprocal : ℚ := -2023

-- Theorem stating the reciprocal of the given number is the expected reciprocal
theorem reciprocal_is_correct : 1 / num = reciprocal :=
  by
    -- The actual proof can be filled in here
    sorry

end reciprocal_is_correct_l46_46048


namespace solve_for_y_l46_46830

theorem solve_for_y (x y : ℝ) (h : 2 * x - 7 * y = 8) : y = (2 * x - 8) / 7 := by
  sorry

end solve_for_y_l46_46830


namespace umbrella_numbers_are_40_l46_46155

open Finset

def is_umbrella_number (x y z : ℕ) : Prop :=
  x < y ∧ z < y ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x ∈ {1, 2, 3, 4, 5, 6} ∧ y ∈ {1, 2, 3, 4, 5, 6} ∧ z ∈ {1, 2, 3, 4, 5, 6}

def umbrella_numbers_count : ℕ :=
  (univ : Finset (ℕ × ℕ × ℕ)).filter (λ n, is_umbrella_number n.1 n.2.1 n.2.2).card

theorem umbrella_numbers_are_40 : umbrella_numbers_count = 40 := by
  sorry

end umbrella_numbers_are_40_l46_46155


namespace probability_at_least_6_heads_l46_46757

-- Definitions of the binomial coefficient and probability function
def binom (n k : ℕ) : ℕ := Nat.choose n k

def probability (favorable total : ℕ) : ℚ := favorable / total

-- Proof problem statement
theorem probability_at_least_6_heads (flips : ℕ) (p : ℚ) 
  (h_flips : flips = 8) 
  (h_probability : p = probability (binom 8 6 + binom 8 7 + binom 8 8) (2 ^ flips)) : 
  p = 37 / 256 :=
by
  sorry

end probability_at_least_6_heads_l46_46757


namespace length_AC_correct_l46_46481

noncomputable def length_AC (A B C D : Type) : ℝ := 105 / 17

variable {A B C D : Type}
variables (angle_BAC angle_ADB length_AD length_BC : ℝ)

theorem length_AC_correct
  (h1 : angle_BAC = 60)
  (h2 : angle_ADB = 30)
  (h3 : length_AD = 3)
  (h4 : length_BC = 9) :
  length_AC A B C D = 105 / 17 :=
sorry

end length_AC_correct_l46_46481


namespace solve_for_a_l46_46656

theorem solve_for_a (a : ℚ) (h : 2 * a - 3 = 5 - a) : a = 8 / 3 :=
by
  sorry

end solve_for_a_l46_46656


namespace range_of_theta_div_4_l46_46618

noncomputable def theta_third_quadrant (k : ℤ) (θ : ℝ) : Prop :=
  (2 * k * Real.pi + Real.pi < θ) ∧ (θ < 2 * k * Real.pi + 3 * Real.pi / 2)

noncomputable def sin_lt_cos (θ : ℝ) : Prop :=
  Real.sin (θ / 4) < Real.cos (θ / 4)

theorem range_of_theta_div_4 (k : ℤ) (θ : ℝ) :
  theta_third_quadrant k θ →
  sin_lt_cos θ →
  (2 * k * Real.pi + 5 * Real.pi / 4 < θ / 4 ∧ θ / 4 < 2 * k * Real.pi + 11 * Real.pi / 8) ∨
  (2 * k * Real.pi + 7 * Real.pi / 4 < θ / 4 ∧ θ / 4 < 2 * k * Real.pi + 15 * Real.pi / 8) := 
  by
    sorry

end range_of_theta_div_4_l46_46618


namespace fraction_eaten_on_third_day_l46_46007

theorem fraction_eaten_on_third_day
  (total_pieces : ℕ)
  (first_day_fraction : ℚ)
  (second_day_fraction : ℚ)
  (remaining_after_third_day : ℕ)
  (initial_pieces : total_pieces = 200)
  (first_day_eaten : first_day_fraction = 1/4)
  (second_day_eaten : second_day_fraction = 2/5)
  (remaining_bread_after_third_day : remaining_after_third_day = 45) :
  (1 : ℚ) / 2 = 1/2 := sorry

end fraction_eaten_on_third_day_l46_46007


namespace max_of_function_l46_46605

-- Using broader imports can potentially bring in necessary libraries, so let's use Mathlib

noncomputable def max_value_in_domain : ℝ :=
  sup ((λ (p : ℝ × ℝ), p.1 * p.2 / (p.1^2 + p.2^2)) '' { x | 1/4 ≤ x.1 ∧ x.1 ≤ 3/5 ∧ 2/7 ≤ x.2 ∧ x.2 ≤ 1/2 } : set ℝ)

theorem max_of_function :
  max_value_in_domain = 2 / 5 :=
sorry

end max_of_function_l46_46605


namespace soccer_most_students_l46_46247

def sports := ["hockey", "basketball", "soccer", "volleyball", "badminton"]
def num_students (sport : String) : Nat :=
  match sport with
  | "hockey" => 30
  | "basketball" => 35
  | "soccer" => 50
  | "volleyball" => 20
  | "badminton" => 25
  | _ => 0

theorem soccer_most_students : ∀ sport ∈ sports, num_students "soccer" ≥ num_students sport := by
  sorry

end soccer_most_students_l46_46247


namespace expression_positive_intervals_l46_46600
open Real

theorem expression_positive_intervals (x : ℝ) :
  ((x + 2) * (x - 3) > 0) ↔ (x < -2 ∨ x > 3) := by
  sorry

end expression_positive_intervals_l46_46600


namespace probability_at_most_one_female_is_correct_l46_46914

-- Defining the number of male and female students
def num_males : ℕ := 3
def num_females : ℕ := 2
def total_students : ℕ := num_males + num_females

-- The event of selecting 2 students from the total students
def total_selections : ℕ := Nat.choose total_students 2

-- The event of selecting at most one female student (0 or 1 female)
def favorable_selections : ℕ := Nat.choose num_males 2 + num_males * num_females

-- The required probability
noncomputable def prob_at_most_one_female : ℚ := favorable_selections / total_selections

theorem probability_at_most_one_female_is_correct :
  prob_at_most_one_female = 9 / 10 :=
by
  sorry

end probability_at_most_one_female_is_correct_l46_46914


namespace problem_statement_l46_46836

variable (p q : ℝ)

def condition := p ^ 2 / q ^ 3 = 4 / 5

theorem problem_statement (hpq : condition p q) : 11 / 7 + (2 * q ^ 3 - p ^ 2) / (2 * q ^ 3 + p ^ 2) = 2 :=
sorry

end problem_statement_l46_46836


namespace mark_deposit_is_88_l46_46860

-- Definitions according to the conditions
def markDeposit := 88
def bryanDeposit (m : ℕ) := 5 * m - 40

-- The theorem we need to prove
theorem mark_deposit_is_88 : markDeposit = 88 := 
by 
  -- Since the condition states Mark deposited $88,
  -- this is trivially true.
  sorry

end mark_deposit_is_88_l46_46860


namespace dolly_dresses_shipment_l46_46091

variable (T : ℕ)

/-- Given that 70% of the total number of Dolly Dresses in the shipment is equal to 140,
    prove that the total number of Dolly Dresses in the shipment is 200. -/
theorem dolly_dresses_shipment (h : (7 * T) / 10 = 140) : T = 200 :=
sorry

end dolly_dresses_shipment_l46_46091


namespace children_less_than_adults_l46_46429

theorem children_less_than_adults (total_members : ℕ)
  (percent_adults : ℝ) (percent_teenagers : ℝ) (percent_children : ℝ) :
  total_members = 500 →
  percent_adults = 0.45 →
  percent_teenagers = 0.25 →
  percent_children = 1 - percent_adults - percent_teenagers →
  (percent_children * total_members) - (percent_adults * total_members) = -75 := 
by
  intros h_total h_adults h_teenagers h_children
  sorry

end children_less_than_adults_l46_46429


namespace sum_of_consecutive_integers_of_sqrt3_l46_46638

theorem sum_of_consecutive_integers_of_sqrt3 {a b : ℤ} (h1 : a + 1 = b) (h2 : (a : ℝ) < Real.sqrt 3) (h3 : Real.sqrt 3 < (b : ℝ)) :
  a + b = 3 := by
  sorry

end sum_of_consecutive_integers_of_sqrt3_l46_46638


namespace parallelepiped_volume_l46_46270

noncomputable def volume_of_parallelepiped (a : ℝ) : ℝ :=
  (a^3 * Real.sqrt 2) / 2

theorem parallelepiped_volume (a : ℝ) (h_pos : 0 < a) :
  volume_of_parallelepiped a = (a^3 * Real.sqrt 2) / 2 :=
by
  sorry

end parallelepiped_volume_l46_46270


namespace projectile_height_l46_46387

theorem projectile_height (t : ℝ) : 
  (∃ t : ℝ, (-4.9 * t^2 + 30.4 * t = 35)) → 
  (0 < t ∧ t ≤ 5) → 
  t = 10 / 7 :=
by
  sorry

end projectile_height_l46_46387


namespace incorrect_statement_B_l46_46621

-- Define the function f(x)
def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := (x - 1)^3 - a * x - b + 2

-- Condition for statement B
axiom eqn_B (a b : ℝ) : (∀ x : ℝ, f (2 - x) a b = 1 - f x a b) → a + b ≠ -1

-- The theorem to prove:
theorem incorrect_statement_B (a b : ℝ) : (∀ x : ℝ, f (2 - x) a b = 1 - f x a b) → a + b ≠ -1 := by
  exact eqn_B a b

end incorrect_statement_B_l46_46621


namespace four_digit_multiples_of_5_count_l46_46309

-- Define the range of four-digit numbers
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

-- Define a multiple of 5
def is_multiple_of_five (n : ℕ) : Prop := n % 5 = 0

-- Define the required proof problem
theorem four_digit_multiples_of_5_count : 
  ∃ n : ℕ, (n = 1800) ∧ (∀ k : ℕ, is_four_digit k ∧ is_multiple_of_five k → n = 1800) :=
sorry

end four_digit_multiples_of_5_count_l46_46309


namespace infinite_triples_of_coprime_integers_l46_46018

open Nat

theorem infinite_triples_of_coprime_integers :
  ∃ᶠ (a b c : ℕ),
  (coprime a b) ∧ (coprime b c) ∧ (coprime c a) ∧
  (coprime (a * b + c) (b * c + a)) ∧
  (coprime (b * c + a) (c * a + b)) ∧
  (coprime (c * a + b) (a * b + c)) :=
  sorry

end infinite_triples_of_coprime_integers_l46_46018


namespace second_fragment_speed_is_l46_46085

variables (u t g vₓ₁ : ℝ)
variables (vₓ₂ vᵧ₂ : ℝ)

-- Given conditions
def initial_vertical_velocity : ℝ := u
def time_of_explosion : ℝ := t
def gravity_acceleration : ℝ := g
def first_fragment_horizontal_velocity : ℝ := vₓ₁

noncomputable def second_fragment_speed : ℝ :=
  let vᵧ := initial_vertical_velocity - gravity_acceleration * time_of_explosion in
  let vₓ₂ := -first_fragment_horizontal_velocity in
  let vᵧ₂ := vᵧ in
  real.sqrt (vₓ₂^2 + vᵧ₂^2)

theorem second_fragment_speed_is : second_fragment_speed u t g vₓ₁ = real.sqrt 2404 := 
sorry

end second_fragment_speed_is_l46_46085


namespace max_min_cos_sin_product_l46_46132

theorem max_min_cos_sin_product (x y z : ℝ) 
  (h1 : x ≥ y) (h2 : y ≥ z) (h3 : z ≥ π / 12) (h4 : x + y + z = π / 2) :
  ∃ (maximum minimum : ℝ), maximum = (2 + Real.sqrt 3) / 8 ∧ minimum = 1 / 8 := by
  sorry

end max_min_cos_sin_product_l46_46132


namespace train_speed_calculation_l46_46434

variable (p : ℝ) (h_p : p > 0)

/-- The speed calculation of a train that covers 200 meters in p seconds is correctly given by 720 / p km/hr. -/
theorem train_speed_calculation (h_p : p > 0) : (200 / p * 3.6 = 720 / p) :=
by
  sorry

end train_speed_calculation_l46_46434


namespace number_of_players_l46_46024

theorem number_of_players (n : ℕ) (G : ℕ) (h : G = 2 * n * (n - 1)) : n = 19 :=
by {
  sorry
}

end number_of_players_l46_46024


namespace A_independent_P_A_n_eq_1_div_n_l46_46853

open MeasureTheory Probability

-- Definitions for X_i sequence and exchangeability conditions.
variable {Ω : Type*} [SampleSpace Ω] {X : ℕ → Ω → ℝ}

axiom X_exchangeable : ∀ (π : ℕ → ℕ) (hπ : Function.Bijective π), 
  (∀ (i : ℕ), map (X i) = map (X (π i)))

axiom X_i_eq_X_j_zero : ∀ {i j : ℕ}, i ≠ j → 
  Measure.measure (set_of (λ ω : Ω, X i ω = X j ω)) = 0

-- Definition of A_n events
def A : ℕ → Set Ω
| 1       := set.univ
| (n + 1) := {ω | ∀ m < n + 1, X (n + 1) ω > X m ω}

-- The proof objectives
theorem A_independent : ∀ n ≥ 1, Indep (λ i, A i) :=
sorry

theorem P_A_n_eq_1_div_n : ∀ n ≥ 1, prob (A n) = 1 / n :=
sorry

end A_independent_P_A_n_eq_1_div_n_l46_46853


namespace best_purchase_option_l46_46531

theorem best_purchase_option 
  (blend_techn_city : ℕ := 2000)
  (meat_techn_city : ℕ := 4000)
  (discount_techn_city : ℤ := 10)
  (blend_techn_market : ℕ := 1500)
  (meat_techn_market : ℕ := 4800)
  (bonus_techn_market : ℤ := 20) : 
  0.9 * (blend_techn_city + meat_techn_city) < (blend_techn_market + meat_techn_market) :=
by
  sorry

end best_purchase_option_l46_46531


namespace total_interest_calculation_l46_46923

-- Define the total investment
def total_investment : ℝ := 20000

-- Define the fractional part of investment at 9 percent rate
def fraction_higher_rate : ℝ := 0.55

-- Define the investment amounts based on the fractional part
def investment_higher_rate : ℝ := fraction_higher_rate * total_investment
def investment_lower_rate : ℝ := total_investment - investment_higher_rate

-- Define interest rates
def rate_lower : ℝ := 0.06
def rate_higher : ℝ := 0.09

-- Define time period (in years)
def time_period : ℝ := 1

-- Define interest calculations
def interest_lower : ℝ := investment_lower_rate * rate_lower * time_period
def interest_higher : ℝ := investment_higher_rate * rate_higher * time_period

-- Define the total interest
def total_interest : ℝ := interest_lower + interest_higher

-- Theorem stating the total interest earned
theorem total_interest_calculation : total_interest = 1530 := by
  -- skip proof using sorry
  sorry

end total_interest_calculation_l46_46923


namespace height_of_boxes_l46_46909

-- Conditions
def total_volume : ℝ := 1.08 * 10^6
def cost_per_box : ℝ := 0.2
def total_monthly_cost : ℝ := 120

-- Target height of the boxes
def target_height : ℝ := 12.2

-- Problem: Prove that the height of each box is 12.2 inches
theorem height_of_boxes : 
  (total_monthly_cost / cost_per_box) * ((total_volume / (total_monthly_cost / cost_per_box))^(1/3)) = target_height := 
sorry

end height_of_boxes_l46_46909


namespace polynomial_condition_l46_46262

noncomputable def polynomial_of_degree_le (n : ℕ) (P : Polynomial ℝ) :=
  P.degree ≤ n

noncomputable def has_nonneg_coeff (P : Polynomial ℝ) :=
  ∀ i, 0 ≤ P.coeff i

theorem polynomial_condition
  (n : ℕ) (P : Polynomial ℝ)
  (h1 : polynomial_of_degree_le n P)
  (h2 : has_nonneg_coeff P)
  (h3 : ∀ x : ℝ, x > 0 → P.eval x * P.eval (1 / x) ≤ (P.eval 1) ^ 2) : 
  ∃ a_n : ℝ, 0 ≤ a_n ∧ P = Polynomial.C a_n * Polynomial.X^n :=
sorry

end polynomial_condition_l46_46262


namespace trapezoid_area_correct_l46_46926

noncomputable def trapezoid_area (x : ℝ) : ℝ :=
  let base1 := 3 * x
  let base2 := 5 * x + 2
  (base1 + base2) / 2 * x

theorem trapezoid_area_correct (x : ℝ) : trapezoid_area x = 4 * x^2 + x :=
  by
  sorry

end trapezoid_area_correct_l46_46926


namespace largest_n_unique_k_l46_46221

theorem largest_n_unique_k :
  ∃ (n : ℕ), ( ∃! (k : ℕ), (5 : ℚ) / 11 < (n : ℚ) / (n + k) ∧ (n : ℚ) / (n + k) < 6 / 11 )
    ∧ n = 359 :=
sorry

end largest_n_unique_k_l46_46221


namespace probability_of_at_least_six_heads_is_correct_l46_46763

-- Definitions for the given problem
def binomial_coefficient (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

def total_possible_outcomes : ℕ :=
  2^8

def favorable_outcomes : ℕ :=
  binomial_coefficient 8 6 + binomial_coefficient 8 7 + binomial_coefficient 8 8

def probability_of_at_least_6_heads : ℚ :=
  favorable_outcomes / total_possible_outcomes

-- The proof statement
theorem probability_of_at_least_six_heads_is_correct :
  probability_of_at_least_6_heads = 37 / 256 :=
by sorry

end probability_of_at_least_six_heads_is_correct_l46_46763


namespace intersection_complement_eq_l46_46472

open Set

def U : Set Int := univ
def M : Set Int := {-1, 0, 1}
def N : Set Int := {0, 1, 3}

theorem intersection_complement_eq :
  (U \ M) ∩ N = {3} :=
  by sorry

end intersection_complement_eq_l46_46472


namespace probability_at_least_6_heads_in_8_flips_l46_46762

theorem probability_at_least_6_heads_in_8_flips : 
  (∑ k in finset.range 3, nat.choose 8 (6 + k)) / (2 ^ 8) = 37 / 256 :=
by sorry

end probability_at_least_6_heads_in_8_flips_l46_46762


namespace quadratic_term_free_solution_l46_46839

theorem quadratic_term_free_solution (m : ℝ) : 
  (∀ x : ℝ, ∃ (p : ℝ → ℝ), (x + m) * (x^2 + 2*x - 1) = p x + (2 + m) * x^2) → m = -2 :=
by
  intro H
  sorry

end quadratic_term_free_solution_l46_46839


namespace books_per_bookshelf_l46_46595

theorem books_per_bookshelf (total_books bookshelves : ℕ) (h_total_books : total_books = 34) (h_bookshelves : bookshelves = 2) : total_books / bookshelves = 17 :=
by
  sorry

end books_per_bookshelf_l46_46595


namespace negation_universal_prop_l46_46695

theorem negation_universal_prop :
  (¬ (∀ x : ℝ, x^2 ≥ 0)) ↔ ∃ x : ℝ, x^2 < 0 :=
sorry

end negation_universal_prop_l46_46695


namespace quadratic_has_two_real_roots_find_m_l46_46289

def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem quadratic_has_two_real_roots (m : ℝ) :
  discriminant 1 (-4 * m) (3 * m^2) ≥ 0 :=
by
  unfold discriminant
  have h : (-4 * m)^2 - 4 * 1 * (3 * m^2) = 4 * m^2
  ring
  exact ge_of_eq h

theorem find_m (h : 0 < m) (root_diff : ℝ) 
  (diff_eq_two : root_diff = 2) : m = 1 :=
by
  -- Let the roots be x1 and x2
  let x1 := (4 * m + root_diff) / 2
  let x2 := (4 * m - root_diff) / 2
  have : x1 - x2 = root_diff :=
    by
      field_simp
      exact diff_eq_two
  have sum_eq := (x1 - x2) * (x1 + x2) - (x1 + x2) * (x1 - x2) = 4
  ring
  have h_m_eq_1 : 4 * m = 4,
  by field_simp
  exact h_m_eq_1

  have h_m_1 : m = 1,
  sorry
  exact ge_of_eq h_m_1

end quadratic_has_two_real_roots_find_m_l46_46289


namespace count_correct_statements_l46_46197

theorem count_correct_statements : 
  let statement_1 := (a : ℕ) → a^2 * a^2 = 2 * a^2
  let statement_2 := (a b : ℕ) → (a - b)^2 = a^2 - b^2
  let statement_3 := (a : ℕ) → a^2 + a^3 = a^5
  let statement_4 := (a b : ℕ) → (-2 * a^2 * b^3)^3 = -6 * a^6 * b^3
  let statement_5 := (a : ℕ) → (-a^3)^2 / a = a^5
  1 = (if statement_1 then 1 else 0) +
      (if statement_2 then 1 else 0) +
      (if statement_3 then 1 else 0) +
      (if statement_4 then 1 else 0) +
      (if statement_5 then 1 else 0) :=
by
  sorry

end count_correct_statements_l46_46197


namespace matrix_expression_l46_46855

noncomputable theory

def B : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![3, 4], ![0, 2]]

theorem matrix_expression :
  B^10 - 3 • B^9 = ![![0, 4], ![0, -1]] :=
  sorry

end matrix_expression_l46_46855


namespace fraction_age_28_to_32_l46_46106

theorem fraction_age_28_to_32 (F : ℝ) (total_participants : ℝ) 
  (next_year_fraction_increase : ℝ) (next_year_fraction : ℝ) 
  (h1 : total_participants = 500)
  (h2 : next_year_fraction_increase = (1 / 8 : ℝ))
  (h3 : next_year_fraction = 0.5625) 
  (h4 : F + next_year_fraction_increase * F = next_year_fraction) :
  F = 0.5 :=
by
  sorry

end fraction_age_28_to_32_l46_46106


namespace sum_of_two_numbers_l46_46699

theorem sum_of_two_numbers (x y : ℕ) (h1 : 3 * x = 180) (h2 : 4 * x = y) : x + y = 420 := by
  sorry

end sum_of_two_numbers_l46_46699


namespace count_four_digit_multiples_of_5_l46_46307

def is_multiple_of_5 (n : ℕ) : Prop :=
  n % 5 = 0

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def four_digit_multiples_of_5_count (lower upper : ℕ) :=
  (upper - lower + 1)

theorem count_four_digit_multiples_of_5 : 
  four_digit_multiples_of_5_count 200 1999 = 1800 :=
by
  sorry

end count_four_digit_multiples_of_5_l46_46307


namespace final_price_is_correct_l46_46921

def original_price : ℝ := 450
def discounts : List ℝ := [0.10, 0.20, 0.05]

def apply_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

noncomputable def final_sale_price (price : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl apply_discount price

theorem final_price_is_correct:
  final_sale_price original_price discounts = 307.8 :=
by
  sorry

end final_price_is_correct_l46_46921


namespace initial_caterpillars_l46_46999

theorem initial_caterpillars (C : ℕ) 
    (hatch_eggs : C + 4 - 8 = 10) : C = 14 :=
by
  sorry

end initial_caterpillars_l46_46999


namespace tangent_line_through_origin_to_circle_in_third_quadrant_l46_46431

theorem tangent_line_through_origin_to_circle_in_third_quadrant :
  ∃ m : ℝ, (∀ x y : ℝ, y = m * x) ∧ (∀ x y : ℝ, x^2 + y^2 + 4 * x + 3 = 0) ∧ (x < 0 ∧ y < 0) ∧ y = -3 * x :=
sorry

end tangent_line_through_origin_to_circle_in_third_quadrant_l46_46431


namespace no_rational_x_y_m_n_with_conditions_l46_46491

noncomputable def f (t : ℚ) : ℚ := t^3 + t

theorem no_rational_x_y_m_n_with_conditions :
  ¬ ∃ (x y : ℚ) (m n : ℕ), xy = 3 ∧ m > 0 ∧ n > 0 ∧
    (f^[m] x = f^[n] y) := 
sorry

end no_rational_x_y_m_n_with_conditions_l46_46491


namespace problem_I_problem_II_l46_46824

-- Define the function f(x) = |x+1| + |x+m+1|
def f (x : ℝ) (m : ℝ) : ℝ := |x+1| + |x+(m+1)|

-- Define the problem (Ⅰ): f(x) ≥ |m-2| for all x implies m ≥ 1
theorem problem_I (m : ℝ) (h : ∀ x : ℝ, f x m ≥ |m-2|) : m ≥ 1 := sorry

-- Define the problem (Ⅱ): Find the solution set for f(-x) < 2m
theorem problem_II (m : ℝ) :
  (m ≤ 0 → ∀ x : ℝ, ¬ (f (-x) m < 2 * m)) ∧
  (m > 0 → ∀ x : ℝ, (1 - m / 2 < x ∧ x < 3 * m / 2 + 1) ↔ f (-x) m < 2 * m) := sorry

end problem_I_problem_II_l46_46824


namespace find_a_l46_46985

def are_parallel (a : ℝ) : Prop :=
  (a + 1) = (2 - a)

theorem find_a (a : ℝ) (h : are_parallel a) : a = 0 :=
sorry

end find_a_l46_46985


namespace distance_between_foci_of_ellipse_l46_46587

theorem distance_between_foci_of_ellipse :
  ∀ (ellipse: ℝ × ℝ → Prop),
    (∀ x y, ellipse (x, y) ↔ (x - 5)^2 / 25 + (y - 2)^2 / 4 = 1) →
    ∃ c : ℝ, 2 * c = 2 * Real.sqrt (25 - 4) :=
by
  intro ellipse h
  use Real.sqrt (25 - 4)
  sorry

end distance_between_foci_of_ellipse_l46_46587


namespace nth_derivative_correct_l46_46417

noncomputable def y (x : ℝ) : ℝ :=
  Real.sin (3 * x + 1) + Real.cos (5 * x)

noncomputable def n_th_derivative (n : ℕ) (x : ℝ) : ℝ :=
  3^n * Real.sin ((3 * Real.pi / 2) * n + 3 * x + 1) + 5^n * Real.cos ((3 * Real.pi / 2) * n + 5 * x)

theorem nth_derivative_correct (x : ℝ) (n : ℕ) :
  derivative^[n] y x = n_th_derivative n x :=
by
  sorry

end nth_derivative_correct_l46_46417


namespace total_letters_correct_l46_46515

-- Define the conditions
def letters_January := 6
def letters_February := 9
def letters_March := 3 * letters_January

-- Definition of the total number of letters sent
def total_letters := letters_January + letters_February + letters_March

-- The statement we need to prove in Lean
theorem total_letters_correct : total_letters = 33 := 
by
  sorry

end total_letters_correct_l46_46515


namespace vertex_of_parabola_l46_46876

theorem vertex_of_parabola :
  ∀ x : ℝ, (x - 2) ^ 2 + 4 = (x - 2) ^ 2 + 4 → (2, 4) = (2, 4) :=
by
  intro x
  intro h
  -- We know that the vertex of y = (x - 2)^2 + 4 is at (2, 4)
  admit

end vertex_of_parabola_l46_46876


namespace part_a_part_b_l46_46561

-- Part A: Proving the specific values of p and q
theorem part_a (p q : ℝ) : 
  (∀ x : ℝ, (x + 3) ^ 2 + (7 * x + p) ^ 2 = (kx + m) ^ 2) ∧
  (∀ x : ℝ, (3 * x + 5) ^ 2 + (p * x + q) ^ 2 = (cx + d) ^ 2) → 
  p = 21 ∧ q = 35 :=
sorry

-- Part B: Proving the new polynomial is a square of a linear polynomial
theorem part_b (a b c A B C : ℝ) (hab : a ≠ 0) (hA : A ≠ 0) (hb : b ≠ 0) (hB : B ≠ 0)
  (habc : (∀ x : ℝ, (a * x + b) ^ 2 + (A * x + B) ^ 2 = (kx + m) ^ 2) ∧
         (∀ x : ℝ, (b * x + c) ^ 2 + (B * x + C) ^ 2 = (cx + d) ^ 2)) :
  ∀ x : ℝ, (c * x + a) ^ 2 + (C * x + A) ^ 2 = (lx + n) ^ 2 :=
sorry

end part_a_part_b_l46_46561


namespace total_laps_jogged_l46_46669

-- Defining the conditions
def jogged_PE_class : ℝ := 1.12
def jogged_track_practice : ℝ := 2.12

-- Statement to prove
theorem total_laps_jogged : jogged_PE_class + jogged_track_practice = 3.24 := by
  -- Proof would go here
  sorry

end total_laps_jogged_l46_46669


namespace sufficient_but_not_necessary_condition_l46_46962

noncomputable def P := {x : ℝ | 0 < x ∧ x < 3}
noncomputable def Q := {x : ℝ | -3 < x ∧ x < 3}

theorem sufficient_but_not_necessary_condition :
  (∀ x, x ∈ P → x ∈ Q) ∧ ¬(∀ x, x ∈ Q → x ∈ P) := by
  sorry

end sufficient_but_not_necessary_condition_l46_46962


namespace inequality_holds_l46_46815

variable (a b c : ℝ)

theorem inequality_holds (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 + b*c) / (a * (b + c)) + 
  (b^2 + c*a) / (b * (c + a)) + 
  (c^2 + a*b) / (c * (a + b)) ≥ 3 :=
sorry

end inequality_holds_l46_46815


namespace michael_needs_flour_l46_46011

-- Define the given conditions
def total_flour : ℕ := 8
def measuring_cup : ℚ := 1/4
def scoops_to_remove : ℕ := 8

-- Prove the amount of flour Michael needs is 6 cups
theorem michael_needs_flour : 
  (total_flour - (scoops_to_remove * measuring_cup)) = 6 := 
by
  sorry

end michael_needs_flour_l46_46011


namespace ratio_of_powers_l46_46652

theorem ratio_of_powers (a x : ℝ) (h : a^(2 * x) = Real.sqrt 2 - 1) : (a^(3 * x) + a^(-3 * x)) / (a^x + a^(-x)) = 2 * Real.sqrt 2 - 1 := by
  sorry

end ratio_of_powers_l46_46652


namespace reciprocal_of_minus_one_over_2023_l46_46046

theorem reciprocal_of_minus_one_over_2023 : (1 / (- (1 / 2023))) = -2023 := 
by
  sorry

end reciprocal_of_minus_one_over_2023_l46_46046


namespace find_value_of_A_l46_46885

theorem find_value_of_A (A B : ℕ) (h_ratio : A * 5 = 3 * B) (h_diff : B - A = 12) : A = 18 :=
by
  sorry

end find_value_of_A_l46_46885


namespace parallel_lines_sufficient_condition_l46_46728

theorem parallel_lines_sufficient_condition :
  ∀ a : ℝ, (a^2 - a) = 2 → (a = 2 ∨ a = -1) :=
by
  intro a h
  sorry

end parallel_lines_sufficient_condition_l46_46728


namespace colin_speed_l46_46598

noncomputable def B : Real := 1
noncomputable def T : Real := 2 * B
noncomputable def Br : Real := (1/3) * T
noncomputable def C : Real := 6 * Br

theorem colin_speed : C = 4 := by
  sorry

end colin_speed_l46_46598


namespace value_of_b_l46_46269

theorem value_of_b (b : ℝ) (h : 4 * ((3.6 * b * 2.50) / (0.12 * 0.09 * 0.5)) = 3200.0000000000005) : b = 0.48 :=
by {
  sorry
}

end value_of_b_l46_46269


namespace last_digit_of_3_pow_2012_l46_46256

-- Theorem: The last digit of 3^2012 is 1 given the cyclic pattern of last digits for powers of 3.
theorem last_digit_of_3_pow_2012 : (3 ^ 2012) % 10 = 1 :=
by
  sorry

end last_digit_of_3_pow_2012_l46_46256


namespace find_principal_l46_46030

noncomputable def principal_amount (P : ℝ) : Prop :=
  let r := 0.05
  let t := 2
  let SI := P * r * t
  let CI := P * (1 + r) ^ t - P
  CI - SI = 15

theorem find_principal : principal_amount 6000 :=
by
  simp [principal_amount]
  sorry

end find_principal_l46_46030


namespace expression_value_l46_46557

theorem expression_value :
  let x := (3 + 1 : ℚ)⁻¹ * 2
  let y := x⁻¹ * 2
  let z := y⁻¹ * 2
  z = (1 / 2 : ℚ) :=
by
  sorry

end expression_value_l46_46557


namespace votes_cast_l46_46907

theorem votes_cast (V : ℝ) (hv1 : 0.35 * V + (0.35 * V + 1800) = V) : V = 6000 :=
sorry

end votes_cast_l46_46907


namespace total_students_l46_46207

-- Definition of variables and conditions
def M := 50
def E := 4 * M - 3

-- Statement of the theorem to prove
theorem total_students : E + M = 247 := by
  sorry

end total_students_l46_46207


namespace weeks_to_buy_iphone_l46_46940

-- Definitions based on conditions
def iphone_cost : ℝ := 800
def trade_in_value : ℝ := 240
def earnings_per_week : ℝ := 80

-- Mathematically equivalent proof problem
theorem weeks_to_buy_iphone : 
  ∀ (iphone_cost trade_in_value earnings_per_week : ℝ), 
  (iphone_cost - trade_in_value) / earnings_per_week = 7 :=
by
  -- Using the given conditions directly.
  intros iphone_cost trade_in_value earnings_per_week
  sorry

end weeks_to_buy_iphone_l46_46940


namespace diophantine_eq_solutions_l46_46261

theorem diophantine_eq_solutions (p q r k : ℕ) (hp : p > 1) (hq : q > 1) (hr : r > 1) 
  (hp_prime : Prime p) (hq_prime : Prime q) (hr_prime : Prime r) (hk : k > 0) :
  p^2 + q^2 + 49*r^2 = 9*k^2 - 101 ↔ 
  (p = 3 ∧ q = 5 ∧ r = 3 ∧ k = 8) ∨ (p = 5 ∧ q = 3 ∧ r = 3 ∧ k = 8) :=
by sorry

end diophantine_eq_solutions_l46_46261


namespace harkamal_total_amount_l46_46146

def cost_grapes (quantity rate : ℕ) : ℕ := quantity * rate
def cost_mangoes (quantity rate : ℕ) : ℕ := quantity * rate
def total_amount_paid (cost1 cost2 : ℕ) : ℕ := cost1 + cost2

theorem harkamal_total_amount :
  let grapes_quantity := 8
  let grapes_rate := 70
  let mangoes_quantity := 9
  let mangoes_rate := 65
  total_amount_paid (cost_grapes grapes_quantity grapes_rate) (cost_mangoes mangoes_quantity mangoes_rate) = 1145 := 
by
  sorry

end harkamal_total_amount_l46_46146


namespace acme_vs_beta_l46_46780

theorem acme_vs_beta (x : ℕ) :
  (80 + 10 * x < 20 + 15 * x) → (13 ≤ x) :=
by
  intro h
  sorry

end acme_vs_beta_l46_46780


namespace no_nat_pairs_satisfy_eq_l46_46726

theorem no_nat_pairs_satisfy_eq (a b : ℕ) : ¬ (2019 * a ^ 2018 = 2017 + b ^ 2016) :=
sorry

end no_nat_pairs_satisfy_eq_l46_46726


namespace p_and_q_necessary_but_not_sufficient_l46_46419

theorem p_and_q_necessary_but_not_sufficient (p q : Prop) :
  ((p ∧ q) → (p ∨ q)) ∧ ¬((p ∨ q) → (p ∧ q)) := 
by 
  sorry

end p_and_q_necessary_but_not_sufficient_l46_46419


namespace iron_conducts_electricity_l46_46566

-- Define the predicates
def Metal (x : Type) : Prop := sorry
def ConductsElectricity (x : Type) : Prop := sorry
noncomputable def Iron : Type := sorry
  
theorem iron_conducts_electricity (h1 : ∀ x, Metal x → ConductsElectricity x)
  (h2 : Metal Iron) : ConductsElectricity Iron :=
by
  sorry

end iron_conducts_electricity_l46_46566


namespace find_percentage_l46_46477

theorem find_percentage (P N : ℕ) (h1 : N = 100) (h2 : (P : ℝ) / 100 * N = 50 / 100 * 40 + 10) :
  P = 30 :=
by
  sorry

end find_percentage_l46_46477


namespace abs_eq_implies_y_eq_half_l46_46713

theorem abs_eq_implies_y_eq_half (y : ℝ) (h : |y - 3| = |y + 2|) : y = 1 / 2 :=
by 
  sorry

end abs_eq_implies_y_eq_half_l46_46713


namespace area_of_rectangle_at_stage_8_l46_46327

-- Define the conditions given in the problem
def square_side_length : ℝ := 4
def square_area : ℝ := square_side_length * square_side_length
def stages : ℕ := 8

-- Define the statement to be proved
theorem area_of_rectangle_at_stage_8 : (stages * square_area) = 128 := 
by 
  have h1 : square_area = 16 := by
    unfold square_area
    norm_num
  have h2 : (stages * square_area) = 8 * 16 := by
    unfold stages
    rw h1
  rw h2
  norm_num
  sorry

end area_of_rectangle_at_stage_8_l46_46327


namespace ice_floe_mass_l46_46016

/-- Given conditions: 
 - The bear's mass is 600 kg
 - The diameter of the bear's trail on the ice floe is 9.5 meters
 - The observed diameter of the trajectory from the helicopter is 10 meters

 We need to prove that the mass of the ice floe is 11400 kg.
 -/
 theorem ice_floe_mass (m d D : ℝ) (hm : m = 600) (hd : d = 9.5) (hD : D = 10) :
   let M := m * d / (D - d)
   in M = 11400 := by 
 sorry

end ice_floe_mass_l46_46016


namespace petya_wins_with_optimal_play_l46_46164

theorem petya_wins_with_optimal_play :
  ∃ (n m : ℕ), n = 2000 ∧ m = (n * (n - 1)) / 2 ∧
  (∀ (v_cut : ℕ), ∀ (p_cut : ℕ), v_cut = 1 ∧ (p_cut = 2 ∨ p_cut = 3) ∧
  ((∃ k, m - v_cut = 4 * k) → ∃ k, m - v_cut - p_cut = 4 * k + 1) → 
  ∃ k, m - p_cut = 4 * k + 3) :=
sorry

end petya_wins_with_optimal_play_l46_46164


namespace ap_number_of_terms_l46_46484

theorem ap_number_of_terms (a d : ℕ) (n : ℕ) (ha1 : (n - 1) * d = 12) (ha2 : a + 2 * d = 6)
  (h_odd_sum : (n / 2) * (2 * a + (n - 2) * d) = 36) (h_even_sum : (n / 2) * (2 * a + n * d) = 42) :
    n = 12 :=
by
  sorry

end ap_number_of_terms_l46_46484


namespace perfect_square_for_n_l46_46364

theorem perfect_square_for_n 
  (a b : ℕ)
  (h1 : ∃ x : ℕ, ab = x^2)
  (h2 : ∃ y : ℕ, (a + 1) * (b + 1) = y^2) 
  : ∃ n : ℕ, n > 1 ∧ ∃ z : ℕ, (a + n) * (b + n) = z^2 :=
by
  let n := ab
  have h3 : n > 1 := sorry
  have h4 : ∃ z : ℕ, (a + n) * (b + n) = z^2 := sorry
  exact ⟨n, h3, h4⟩

end perfect_square_for_n_l46_46364


namespace bunnies_out_of_burrow_l46_46981

theorem bunnies_out_of_burrow:
  (3 * 60 * 10 * 20) = 36000 :=
by 
  sorry

end bunnies_out_of_burrow_l46_46981


namespace current_books_l46_46776

def initial_books : ℕ := 743
def sold_instore_saturday : ℕ := 37
def sold_online_saturday : ℕ := 128
def sold_instore_sunday : ℕ := 2 * sold_instore_saturday
def sold_online_sunday : ℕ := sold_online_saturday + 34
def total_books_sold_saturday : ℕ := sold_instore_saturday + sold_online_saturday
def total_books_sold_sunday : ℕ := sold_instore_sunday + sold_online_sunday
def total_books_sold_weekend : ℕ := total_books_sold_saturday + total_books_sold_sunday
def books_received_shipment : ℕ := 160
def net_change_books : ℤ := books_received_shipment - total_books_sold_weekend

theorem current_books
  (initial_books : ℕ) 
  (sold_instore_saturday : ℕ) 
  (sold_online_saturday : ℕ) 
  (sold_instore_sunday : ℕ)
  (sold_online_sunday : ℕ)
  (total_books_sold_saturday : ℕ)
  (total_books_sold_sunday : ℕ)
  (total_books_sold_weekend : ℕ)
  (books_received_shipment : ℕ)
  (net_change_books : ℤ) : (initial_books - net_change_books) = 502 := 
by {
  sorry
}

end current_books_l46_46776


namespace probability_at_least_6_heads_8_flips_l46_46742

-- Define the probability calculation of getting at least 6 heads in 8 coin flips.
def probability_at_least_6_heads (n : ℕ) (k : ℕ) : ℚ :=
  (Nat.choose n k + Nat.choose n (k + 1) + Nat.choose n (k + 2)) / 2^n

theorem probability_at_least_6_heads_8_flips : 
  probability_at_least_6_heads 8 6 = 37 / 256 := 
by
  sorry

end probability_at_least_6_heads_8_flips_l46_46742


namespace train_passes_tree_in_20_seconds_l46_46560

def train_passing_time 
  (length_of_train : ℕ)
  (speed_kmh : ℕ)
  (conversion_factor : ℚ) : ℚ :=
  length_of_train / (speed_kmh * conversion_factor)

theorem train_passes_tree_in_20_seconds 
  (length_of_train : ℕ := 350)
  (speed_kmh : ℕ := 63)
  (conversion_factor : ℚ := 1000 / 3600) : 
  train_passing_time length_of_train speed_kmh conversion_factor = 20 :=
  sorry

end train_passes_tree_in_20_seconds_l46_46560


namespace sum_of_coefficients_l46_46672

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) 
  (h : (1 + 2*x)^7 = a + a₁*(1 - x) + a₂*(1 - x)^2 + a₃*(1 - x)^3 + a₄*(1 - x)^4 + a₅*(1 - x)^5 + a₆*(1 - x)^6 + a₇*(1 - x)^7) :
  a + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 1 :=
by 
  sorry

end sum_of_coefficients_l46_46672


namespace notepad_last_duration_l46_46243

def note_duration (folds_per_paper : ℕ) (pieces_of_paper : ℕ) (notes_per_day : ℕ) : ℕ :=
  let note_size_papers_per_letter_paper := 2 ^ folds_per_paper
  let total_note_size_papers := pieces_of_paper * note_size_papers_per_letter_paper
  total_note_size_papers / notes_per_day

theorem notepad_last_duration :
  note_duration 3 5 10 = 4 := by
  sorry

end notepad_last_duration_l46_46243


namespace tan_seven_pi_over_four_l46_46119

theorem tan_seven_pi_over_four : Real.tan (7 * Real.pi / 4) = -1 := 
by
  -- In this case, we are proving a specific trigonometric identity
  sorry

end tan_seven_pi_over_four_l46_46119


namespace total_letters_sent_l46_46520

def letters_January : ℕ := 6
def letters_February : ℕ := 9
def letters_March : ℕ := 3 * letters_January

theorem total_letters_sent : letters_January + letters_February + letters_March = 33 :=
by
  -- This is where the proof would go
  sorry

end total_letters_sent_l46_46520


namespace probability_at_least_6_heads_in_8_flips_l46_46768

open scoped BigOperators

def binom (n k : ℕ) : ℕ := nat.choose n k

def total_outcomes (n : ℕ) := 2^n

def successful_outcomes (n k : ℕ) :=
  (finset.range (n + 1)).filter (λ x, x ≥ k).sum (λ x, binom n x)

def probability (n k : ℕ) :=
  (successful_outcomes n k) / (total_outcomes n : ℚ)

theorem probability_at_least_6_heads_in_8_flips :
  probability 8 6 = 37 / 256 := sorry

end probability_at_least_6_heads_in_8_flips_l46_46768


namespace always_two_real_roots_find_m_l46_46299

-- Given quadratic equation: x^2 - 4mx + 3m^2 = 0
-- Definitions for the problem
def quadratic_eq (m x : ℝ) : Prop := x^2 - 4 * m * x + 3 * m^2 = 0

-- Q1: Prove that this equation always has two real roots.
theorem always_two_real_roots (m : ℝ) : ∃ x₁ x₂ : ℝ, quadratic_eq m x₁ ∧ quadratic_eq m x₂ :=
by
  sorry

-- Q2: If m > 0 and the difference between the two real roots is 2, find the value of m.
theorem find_m (m : ℝ) (h₁ : m > 0) (h₂ : ∃ x₁ x₂ : ℝ, quadratic_eq m x₁ ∧ quadratic_eq m x₂ ∧ |x₁ - x₂| = 2) : m = 1 :=
by
  sorry

end always_two_real_roots_find_m_l46_46299


namespace sum_of_rationals_l46_46174

-- Definition of the conditions
def pairwise_products (a1 a2 a3 a4 : ℚ) : set ℚ :=
  {a1 * a2, a1 * a3, a1 * a4, a2 * a3, a2 * a4, a3 * a4}

-- Statement of the proof problem
theorem sum_of_rationals 
  (a1 a2 a3 a4 : ℚ)
  (h : pairwise_products a1 a2 a3 a4 = {-24, -2, -3/2, -1/8, 1, 3}) :
    a1 + a2 + a3 + a4 = 9/4 ∨ a1 + a2 + a3 + a4 = -9/4 :=
sorry

end sum_of_rationals_l46_46174


namespace time_to_cross_pole_correct_l46_46238

-- Definitions of the conditions
def trainSpeed_kmh : ℝ := 120 -- km/hr
def trainLength_m : ℝ := 300 -- meters

-- Assumed conversions
def kmToMeters : ℝ := 1000 -- meters in a km
def hoursToSeconds : ℝ := 3600 -- seconds in an hour

-- Conversion of speed from km/hr to m/s
noncomputable def trainSpeed_ms := (trainSpeed_kmh * kmToMeters) / hoursToSeconds

-- Time to cross the pole
noncomputable def timeToCrossPole := trainLength_m / trainSpeed_ms

-- The theorem stating the proof problem
theorem time_to_cross_pole_correct : timeToCrossPole = 9 := by
  sorry

end time_to_cross_pole_correct_l46_46238


namespace quadratic_distinct_real_roots_l46_46654

open Real

theorem quadratic_distinct_real_roots (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ k * x ^ 2 - 2 * x - 1 = 0 ∧ k * y ^ 2 - 2 * y - 1 = 0) ↔ k > -1 ∧ k ≠ 0 :=
by
  sorry

end quadratic_distinct_real_roots_l46_46654


namespace find_a_set_l46_46471

-- Given the set A and the condition
def setA (a : ℝ) : Set ℝ := {a + 2, (a + 1)^2, a^2 + 3a + 3}

-- The main statement
theorem find_a_set (a : ℝ) : (1 ∈ setA a) → a = 0 :=
sorry

end find_a_set_l46_46471


namespace ratio_of_carpets_l46_46203

theorem ratio_of_carpets (h1 h2 h3 h4 : ℕ) (total : ℕ) 
  (H1 : h1 = 12) (H2 : h2 = 20) (H3 : h3 = 10) (H_total : total = 62) 
  (H_all_houses : h1 + h2 + h3 + h4 = total) : h4 / h3 = 2 :=
by
  sorry

end ratio_of_carpets_l46_46203


namespace largest_constant_C_l46_46604

theorem largest_constant_C :
  ∃ C : ℝ, 
    (∀ x y z : ℝ, x^2 + y^2 + z^2 + 2 ≥ C * (x + y + z - 1)) 
      ∧ (∀ D : ℝ, (∀ x y z : ℝ, x^2 + y^2 + z^2 + 2 ≥ D * (x + y + z - 1)) → C ≥ D)
    ∧ C = (2 + 2 * Real.sqrt 7) / 3 :=
sorry

end largest_constant_C_l46_46604


namespace train_passing_time_l46_46847

noncomputable def train_length : ℝ := 180
noncomputable def train_speed_km_hr : ℝ := 36
noncomputable def train_speed_m_s : ℝ := train_speed_km_hr * (1000 / 3600)

theorem train_passing_time : train_length / train_speed_m_s = 18 := by
  sorry

end train_passing_time_l46_46847


namespace quadratic_two_real_roots_find_m_l46_46296

theorem quadratic_two_real_roots (m : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁ * x₂ = 3 * m^2 ∧ x₁ + x₂ = 4 * m :=
by
  sorry

theorem find_m (m : ℝ) (h : m > 0) (h_diff : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ - x₂ = 2) : m = 1 :=
by
  sorry

end quadratic_two_real_roots_find_m_l46_46296


namespace transport_cost_B_condition_l46_46704

-- Define the parameters for coal from Mine A
def calories_per_gram_A := 4
def price_per_ton_A := 20
def transport_cost_A := 8

-- Define the parameters for coal from Mine B
def calories_per_gram_B := 6
def price_per_ton_B := 24

-- Define the total cost for transporting one ton from Mine A to city N
def total_cost_A := price_per_ton_A + transport_cost_A

-- Define the question as a Lean theorem
theorem transport_cost_B_condition : 
  ∀ (transport_cost_B : ℝ), 
  (total_cost_A : ℝ) / (calories_per_gram_A : ℝ) = (price_per_ton_B + transport_cost_B) / (calories_per_gram_B : ℝ) → 
  transport_cost_B = 18 :=
by
  intros transport_cost_B h
  have h_eq : (total_cost_A : ℝ) / (calories_per_gram_A : ℝ) = (price_per_ton_B + transport_cost_B) / (calories_per_gram_B : ℝ) := h
  sorry

end transport_cost_B_condition_l46_46704


namespace rectangle_area_l46_46564

theorem rectangle_area (L B : ℕ) 
  (h1 : L - B = 23)
  (h2 : 2 * L + 2 * B = 186) :
  L * B = 2030 := by
  sorry

end rectangle_area_l46_46564


namespace probability_at_least_6_heads_in_8_flips_l46_46772

open scoped BigOperators

def binom (n k : ℕ) : ℕ := nat.choose n k

def total_outcomes (n : ℕ) := 2^n

def successful_outcomes (n k : ℕ) :=
  (finset.range (n + 1)).filter (λ x, x ≥ k).sum (λ x, binom n x)

def probability (n k : ℕ) :=
  (successful_outcomes n k) / (total_outcomes n : ℚ)

theorem probability_at_least_6_heads_in_8_flips :
  probability 8 6 = 37 / 256 := sorry

end probability_at_least_6_heads_in_8_flips_l46_46772


namespace algebraic_expression_positive_l46_46866

theorem algebraic_expression_positive (a b : ℝ) : 
  a^2 + b^2 + 4*b - 2*a + 6 > 0 :=
by sorry

end algebraic_expression_positive_l46_46866


namespace four_digit_multiples_of_5_l46_46320

theorem four_digit_multiples_of_5 : 
  let f := fun n => 1000 <= n ∧ n <= 9999 ∧ n % 5 = 0
  in finset.card (finset.filter f (finset.range 10000)) = 1800 :=
by
  sorry

end four_digit_multiples_of_5_l46_46320


namespace ten_times_hundred_eq_thousand_ten_times_thousand_eq_ten_thousand_hundreds_in_ten_thousand_tens_in_one_thousand_l46_46230

theorem ten_times_hundred_eq_thousand : 10 * 100 = 1000 := 
by sorry

theorem ten_times_thousand_eq_ten_thousand : 10 * 1000 = 10000 := 
by sorry

theorem hundreds_in_ten_thousand : 10000 / 100 = 100 := 
by sorry

theorem tens_in_one_thousand : 1000 / 10 = 100 := 
by sorry

end ten_times_hundred_eq_thousand_ten_times_thousand_eq_ten_thousand_hundreds_in_ten_thousand_tens_in_one_thousand_l46_46230


namespace consecutive_integers_sum_l46_46651

theorem consecutive_integers_sum (a b : ℤ) (h1 : b = a + 1) (h2 : a < Real.sqrt 3) (h3 : Real.sqrt 3 < b) : a + b = 3 :=
sorry

end consecutive_integers_sum_l46_46651


namespace bottles_of_regular_soda_l46_46576

theorem bottles_of_regular_soda (R : ℕ) : 
  let apples := 36 
  let diet_soda := 54
  let total_bottles := apples + 98 
  R + diet_soda = total_bottles → R = 80 :=
by
  sorry

end bottles_of_regular_soda_l46_46576


namespace total_wheels_eq_90_l46_46890

def total_wheels (num_bicycles : Nat) (wheels_per_bicycle : Nat) (num_tricycles : Nat) (wheels_per_tricycle : Nat) :=
  num_bicycles * wheels_per_bicycle + num_tricycles * wheels_per_tricycle

theorem total_wheels_eq_90 : total_wheels 24 2 14 3 = 90 :=
by
  sorry

end total_wheels_eq_90_l46_46890


namespace four_digit_multiples_of_5_count_l46_46311

-- Define the range of four-digit numbers
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

-- Define a multiple of 5
def is_multiple_of_five (n : ℕ) : Prop := n % 5 = 0

-- Define the required proof problem
theorem four_digit_multiples_of_5_count : 
  ∃ n : ℕ, (n = 1800) ∧ (∀ k : ℕ, is_four_digit k ∧ is_multiple_of_five k → n = 1800) :=
sorry

end four_digit_multiples_of_5_count_l46_46311


namespace meaningful_fraction_range_l46_46158

theorem meaningful_fraction_range (x : ℝ) : (x - 1 ≠ 0) ↔ (fraction_meaningful := x ≠ 1) :=
by
  sorry

end meaningful_fraction_range_l46_46158


namespace maria_remaining_money_l46_46008

theorem maria_remaining_money (initial_amount ticket_cost : ℕ) (h_initial : initial_amount = 760) (h_ticket : ticket_cost = 300) :
  let hotel_cost := ticket_cost / 2
  let total_spent := ticket_cost + hotel_cost
  let remaining := initial_amount - total_spent
  remaining = 310 :=
by
  intros
  sorry

end maria_remaining_money_l46_46008


namespace probability_one_girl_no_growth_pie_l46_46931

-- Definitions based on the conditions
def total_pies := 6
def growth_pies := 2
def shrink_pies := total_pies - growth_pies
def total_selections := ((total_pies).choose(3) : ℚ)
def favorable_selections := ((shrink_pies).choose(2) : ℚ)

-- Calculation of the probability
noncomputable def probability_no_growth_pie := 1 - favorable_selections / total_selections

-- Proving the required probability
theorem probability_one_girl_no_growth_pie : probability_no_growth_pie = 0.4 :=
by
  sorry

end probability_one_girl_no_growth_pie_l46_46931


namespace solve_eq_integers_l46_46375

theorem solve_eq_integers (x y : ℤ) : 
    x^2 - x * y - 6 * y^2 + 2 * x + 19 * y = 18 ↔ (x = 2 ∧ y = 2) ∨ (x = -2 ∧ y = 2) := by
    sorry

end solve_eq_integers_l46_46375


namespace find_certain_number_l46_46405

theorem find_certain_number :
  ∃ C, ∃ A B, (A + B = 15) ∧ (A = 7) ∧ (C * B = 5 * A - 11) ∧ (C = 3) :=
by
  sorry

end find_certain_number_l46_46405


namespace remainder_six_pow_4032_mod_13_l46_46063

theorem remainder_six_pow_4032_mod_13 : (6 ^ 4032) % 13 = 1 := 
by
  sorry

end remainder_six_pow_4032_mod_13_l46_46063


namespace illiterate_employee_count_l46_46483

variable (I : ℕ) -- Number of illiterate employees
variable (literate_count : ℕ) -- Number of literate employees
variable (initial_wage_illiterate : ℕ) -- Initial average wage of illiterate employees
variable (new_wage_illiterate : ℕ) -- New average wage of illiterate employees
variable (average_salary_decrease : ℕ) -- Decrease in the average salary of all employees

-- Given conditions:
def condition1 : initial_wage_illiterate = 25 := by sorry
def condition2 : new_wage_illiterate = 10 := by sorry
def condition3 : average_salary_decrease = 10 := by sorry
def condition4 : literate_count = 10 := by sorry

-- Main proof statement:
theorem illiterate_employee_count :
  initial_wage_illiterate - new_wage_illiterate = 15 →
  average_salary_decrease * (literate_count + I) = (initial_wage_illiterate - new_wage_illiterate) * I →
  I = 20 := by
  intros h1 h2
  -- provided conditions
  exact sorry

end illiterate_employee_count_l46_46483


namespace concert_attendance_l46_46025

-- Define the given conditions
def buses : ℕ := 8
def students_per_bus : ℕ := 45

-- Statement of the problem
theorem concert_attendance :
  buses * students_per_bus = 360 :=
sorry

end concert_attendance_l46_46025


namespace distinct_digit_sums_l46_46659

theorem distinct_digit_sums (A B C E D : ℕ) (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ E ∧ A ≠ D ∧ B ≠ C ∧ B ≠ E ∧ B ≠ D ∧ C ≠ E ∧ C ≠ D ∧ E ≠ D)
 (h_ab : A + B = D) (h_ab_lt_10 : A + B < 10) (h_ce : C + E = D) :
  ∃ (x : ℕ), x = 8 := 
sorry

end distinct_digit_sums_l46_46659


namespace minimum_value_of_f_l46_46952

def f (x : ℝ) : ℝ := |3 - x| + |x - 7|

theorem minimum_value_of_f : ∃ x : ℝ, f x = 4 ∧ ∀ y : ℝ, f y ≥ 4 :=
by {
  sorry
}

end minimum_value_of_f_l46_46952


namespace find_c_l46_46200

theorem find_c (a b c : ℝ) (h1 : ∃ x y : ℝ, x = a * (y - 2)^2 + 3 ∧ (x,y) = (3,2))
  (h2 : (1 : ℝ) = a * ((4 : ℝ) - 2)^2 + 3) : c = 1 :=
sorry

end find_c_l46_46200


namespace solve_for_A_l46_46500

def hash (A B : ℝ) : ℝ := A^2 + B^2

theorem solve_for_A (A : ℝ) (h : hash A 7 = 200) : A = Real.sqrt 151 :=
by
  sorry

end solve_for_A_l46_46500


namespace julia_money_left_l46_46348

def initial_money : ℕ := 40
def spent_on_game : ℕ := initial_money / 2
def money_left_after_game : ℕ := initial_money - spent_on_game
def spent_on_in_game_purchases : ℕ := money_left_after_game / 4
def final_money_left : ℕ := money_left_after_game - spent_on_in_game_purchases

theorem julia_money_left : final_money_left = 15 := by
  sorry

end julia_money_left_l46_46348


namespace ascetic_height_l46_46551

theorem ascetic_height (h m : ℝ) (x : ℝ) (hx : h * (m + 1) = (x + h)^2 + (m * h)^2) : x = h * m / (m + 2) :=
sorry

end ascetic_height_l46_46551


namespace find_c_l46_46275

theorem find_c (c : ℝ) (f : ℝ → ℝ) (f_inv : ℝ → ℝ)
  (hf : ∀ x, f x = 2 / (3 * x + c))
  (hfinv : ∀ x, f_inv x = (2 - 3 * x) / (3 * x)) :
  c = 3 :=
by
  sorry

end find_c_l46_46275


namespace representative_arrangements_l46_46399

theorem representative_arrangements
  (boys girls : ℕ) 
  (subjects : ℕ) 
  (pe_rep : ∀ n, n = 1) 
  (eng_rep : ∀ n, n = 1) 
  : boys = 4 → girls = 3 → subjects = 7 → (4 * 3 * (Nat.factorial 5) = 1440) :=
by
  intros hb hg hs
  rw [hb, hg, hs]
  exact Nat.mul_assoc 4 3 (Nat.factorial 5) = 1440
  sorry

end representative_arrangements_l46_46399


namespace odd_function_condition_l46_46694

noncomputable def f (x a b : ℝ) : ℝ :=
  x * abs (x + a) + b

theorem odd_function_condition (a b : ℝ) :
  (∀ x : ℝ, f x a b = f (-x) a b) ↔ (a = 0 ∧ b = 0) := 
by
  sorry

end odd_function_condition_l46_46694


namespace ratio_A_B_l46_46383

-- Define constants for non-zero numbers A and B
variables {A B : ℕ} (h1 : A ≠ 0) (h2 : B ≠ 0)

-- Define the given condition
theorem ratio_A_B (h : (2 * A) * 7 = (3 * B) * 3) : A / B = 9 / 14 := by
  sorry

end ratio_A_B_l46_46383


namespace solution_set_inequality_l46_46822

variable {x : ℝ}
variable {a b : ℝ}

theorem solution_set_inequality (h₁ : ∀ x : ℝ, (ax^2 + bx - 1 > 0) ↔ (-1/2 < x ∧ x < -1/3)) :
  ∀ x : ℝ, (x^2 - bx - a ≥ 0) ↔ (x ≤ -3 ∨ x ≥ -2) := 
sorry

end solution_set_inequality_l46_46822


namespace find_m_l46_46137

-- Definitions for the sets A and B
def A (m : ℝ) : Set ℝ := {3, 4, 4 * m - 4}
def B (m : ℝ) : Set ℝ := {3, m^2}

-- Problem statement
theorem find_m {m : ℝ} (h : B m ⊆ A m) : m = -2 :=
sorry

end find_m_l46_46137


namespace rate_of_current_l46_46051

/-- The speed of a boat in still water is 20 km/hr, and the rate of current is c km/hr.
    The distance travelled downstream in 24 minutes is 9.2 km. What is the rate of the current? -/
theorem rate_of_current (c : ℝ) (h : 24/60 = 0.4 ∧ 9.2 = (20 + c) * 0.4) : c = 3 :=
by
  sorry  -- Proof is not required, only the statement is necessary.

end rate_of_current_l46_46051


namespace quadratic_two_real_roots_find_m_l46_46295

theorem quadratic_two_real_roots (m : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁ * x₂ = 3 * m^2 ∧ x₁ + x₂ = 4 * m :=
by
  sorry

theorem find_m (m : ℝ) (h : m > 0) (h_diff : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ - x₂ = 2) : m = 1 :=
by
  sorry

end quadratic_two_real_roots_find_m_l46_46295


namespace find_line_equation_l46_46619

noncomputable def line_equation : real × real × real := sorry

theorem find_line_equation (m : real) :
  (∃ (m : real), (3, 4, m) = line_equation ∧
    let a := -m / 4 in
    let b := -m / 3 in
    (1 / 2) * |a| * |b| = 24) :=
begin
  sorry
end

end find_line_equation_l46_46619


namespace net_change_in_price_l46_46479

theorem net_change_in_price (P : ℝ) : 
  ((P * 0.75) * 1.2 = P * 0.9) → 
  ((P * 0.9 - P) / P = -0.1) :=
by
  intro h
  sorry

end net_change_in_price_l46_46479


namespace bunnies_out_of_burrows_l46_46982

theorem bunnies_out_of_burrows 
  (bunnies_per_min : Nat) 
  (num_bunnies : Nat) 
  (hours : Nat) 
  (mins_per_hour : Nat):
  bunnies_per_min = 3 -> num_bunnies = 20 -> mins_per_hour = 60 -> hours = 10 -> 
  (bunnies_per_min * mins_per_hour * hours * num_bunnies = 36000) :=
by
  intros,
  sorry

end bunnies_out_of_burrows_l46_46982


namespace find_original_polynomial_calculate_correct_result_l46_46559

variable {P : Polynomial ℝ}
variable (Q : Polynomial ℝ := 2 * X ^ 2 + X - 5)
variable (R : Polynomial ℝ := X ^ 2 + 3 * X - 1)

theorem find_original_polynomial (h : P - Q = R) : P = 3 * X ^ 2 + 4 * X - 6 :=
by
  sorry

theorem calculate_correct_result (h : P = 3 * X ^ 2 + 4 * X - 6) : P - Q = X ^ 2 + X + 9 :=
by
  sorry

end find_original_polynomial_calculate_correct_result_l46_46559


namespace quadratic_no_real_roots_range_l46_46336

theorem quadratic_no_real_roots_range (k : ℝ) : 
  (∀ x : ℝ, ¬ (x^2 + 2 * x - k = 0)) ↔ k < -1 :=
by
  sorry

end quadratic_no_real_roots_range_l46_46336


namespace always_two_real_roots_find_m_l46_46300

-- Given quadratic equation: x^2 - 4mx + 3m^2 = 0
-- Definitions for the problem
def quadratic_eq (m x : ℝ) : Prop := x^2 - 4 * m * x + 3 * m^2 = 0

-- Q1: Prove that this equation always has two real roots.
theorem always_two_real_roots (m : ℝ) : ∃ x₁ x₂ : ℝ, quadratic_eq m x₁ ∧ quadratic_eq m x₂ :=
by
  sorry

-- Q2: If m > 0 and the difference between the two real roots is 2, find the value of m.
theorem find_m (m : ℝ) (h₁ : m > 0) (h₂ : ∃ x₁ x₂ : ℝ, quadratic_eq m x₁ ∧ quadratic_eq m x₂ ∧ |x₁ - x₂| = 2) : m = 1 :=
by
  sorry

end always_two_real_roots_find_m_l46_46300


namespace ages_of_siblings_l46_46078

-- Define the variables representing the ages of the siblings
variables (R D S E : ℕ)

-- Define the conditions
def conditions := 
  R = D + 6 ∧ 
  D = S + 8 ∧ 
  E = R - 5 ∧ 
  R + 8 = 2 * (S + 8)

-- Define the statement to be proved
theorem ages_of_siblings (h : conditions R D S E) : 
  R = 20 ∧ D = 14 ∧ S = 6 ∧ E = 15 :=
sorry

end ages_of_siblings_l46_46078


namespace reciprocal_is_correct_l46_46047

-- Define the initial number
def num : ℚ := -1 / 2023

-- Define the expected reciprocal
def reciprocal : ℚ := -2023

-- Theorem stating the reciprocal of the given number is the expected reciprocal
theorem reciprocal_is_correct : 1 / num = reciprocal :=
  by
    -- The actual proof can be filled in here
    sorry

end reciprocal_is_correct_l46_46047


namespace halfway_fraction_l46_46216

theorem halfway_fraction (a b : ℚ) (ha : a = 3/4) (hb : b = 5/6) : (a + b) / 2 = 19/24 :=
by
  rw [ha, hb] -- replace a and b with 3/4 and 5/6 respectively
  have h1 : 3/4 + 5/6 = 19/12,
  { norm_num, -- ensures 3/4 + 5/6 = 19/12
    linarith },
  rw h1, -- replace a + b with 19/12
  norm_num -- ensures (19/12) / 2 = 19/24

end halfway_fraction_l46_46216


namespace solution_for_b_l46_46338

theorem solution_for_b (x y b : ℚ) (h1 : 4 * x + 3 * y = b) (h2 : 3 * x + 4 * y = 3 * b) (hx : x = 3) : b = -21 / 5 := by
  sorry

end solution_for_b_l46_46338


namespace area_at_stage_8_l46_46331

-- Defining the constants and initial settings
def first_term : ℕ := 1
def common_difference : ℕ := 1
def stage : ℕ := 8
def square_side_length : ℕ := 4

-- Calculating the number of squares at the given stage
def num_squares : ℕ := first_term + (stage - 1) * common_difference

--Calculating the area of one square
def area_one_square : ℕ := square_side_length * square_side_length

-- Calculating the total area at the given stage
def total_area : ℕ := num_squares * area_one_square

-- Proving the total area equals 128 at Stage 8
theorem area_at_stage_8 : total_area = 128 := 
by
  sorry

end area_at_stage_8_l46_46331


namespace tetrahedron_edges_vertices_product_l46_46922

theorem tetrahedron_edges_vertices_product :
  let vertices := 4
  let edges := 6
  edges * vertices = 24 :=
by
  let vertices := 4
  let edges := 6
  sorry

end tetrahedron_edges_vertices_product_l46_46922


namespace randy_trip_distance_l46_46182

theorem randy_trip_distance (x : ℝ) (h1 : x = x / 4 + 30 + x / 10 + (x - (x / 4 + 30 + x / 10))) :
  x = 60 :=
by {
  sorry -- Placeholder for the actual proof
}

end randy_trip_distance_l46_46182


namespace current_books_l46_46775

def initial_books : ℕ := 743
def sold_instore_saturday : ℕ := 37
def sold_online_saturday : ℕ := 128
def sold_instore_sunday : ℕ := 2 * sold_instore_saturday
def sold_online_sunday : ℕ := sold_online_saturday + 34
def total_books_sold_saturday : ℕ := sold_instore_saturday + sold_online_saturday
def total_books_sold_sunday : ℕ := sold_instore_sunday + sold_online_sunday
def total_books_sold_weekend : ℕ := total_books_sold_saturday + total_books_sold_sunday
def books_received_shipment : ℕ := 160
def net_change_books : ℤ := books_received_shipment - total_books_sold_weekend

theorem current_books
  (initial_books : ℕ) 
  (sold_instore_saturday : ℕ) 
  (sold_online_saturday : ℕ) 
  (sold_instore_sunday : ℕ)
  (sold_online_sunday : ℕ)
  (total_books_sold_saturday : ℕ)
  (total_books_sold_sunday : ℕ)
  (total_books_sold_weekend : ℕ)
  (books_received_shipment : ℕ)
  (net_change_books : ℤ) : (initial_books - net_change_books) = 502 := 
by {
  sorry
}

end current_books_l46_46775


namespace probability_at_least_6_heads_l46_46754

-- Definitions of the binomial coefficient and probability function
def binom (n k : ℕ) : ℕ := Nat.choose n k

def probability (favorable total : ℕ) : ℚ := favorable / total

-- Proof problem statement
theorem probability_at_least_6_heads (flips : ℕ) (p : ℚ) 
  (h_flips : flips = 8) 
  (h_probability : p = probability (binom 8 6 + binom 8 7 + binom 8 8) (2 ^ flips)) : 
  p = 37 / 256 :=
by
  sorry

end probability_at_least_6_heads_l46_46754


namespace inequality_solution_set_minimum_value_mn_squared_l46_46972

noncomputable def f (x : ℝ) := |x - 2| + |x + 1|

theorem inequality_solution_set : 
  (∀ x, f x > 7 ↔ x > 4 ∨ x < -3) :=
by sorry

theorem minimum_value_mn_squared (m n : ℝ) (hm : n > 0) (hmin : ∀ x, f x ≥ m + n) :
  m^2 + n^2 = 9 / 2 ∧ m = 3 / 2 ∧ n = 3 / 2 :=
by sorry

end inequality_solution_set_minimum_value_mn_squared_l46_46972


namespace number_of_girls_in_class_l46_46846

theorem number_of_girls_in_class (B S G : ℕ)
  (h1 : 3 * B = 4 * 18)  -- 3/4 * B = 18
  (h2 : 2 * S = 3 * B)  -- 2/3 * S = B
  (h3 : G = S - B) : G = 12 :=
by
  sorry

end number_of_girls_in_class_l46_46846


namespace sequence_value_at_5_l46_46166

def seq (a : ℕ → ℚ) : Prop :=
  a 1 = 1 / 3 ∧ ∀ n, 1 < n → a n = (-1) ^ n * 2 * a (n - 1)

theorem sequence_value_at_5 (a : ℕ → ℚ) (h : seq a) : a 5 = 16 / 3 :=
by 
  sorry

end sequence_value_at_5_l46_46166


namespace sum_remainders_l46_46783

theorem sum_remainders (n : ℤ) (h : n % 20 = 13) : (n % 4 + n % 5 = 4) :=
by
  sorry

end sum_remainders_l46_46783


namespace min_attempts_sufficient_a_l46_46719

theorem min_attempts_sufficient_a (n : ℕ) (h : n > 2)
  (good_batteries bad_batteries : ℕ)
  (h1 : good_batteries = n + 1)
  (h2 : bad_batteries = n)
  (total_batteries := 2 * n + 1) :
  (∃ attempts, attempts = n + 1) := sorry

end min_attempts_sufficient_a_l46_46719


namespace tan_seven_pi_over_four_l46_46118

theorem tan_seven_pi_over_four : Real.tan (7 * Real.pi / 4) = -1 := 
by
  -- In this case, we are proving a specific trigonometric identity
  sorry

end tan_seven_pi_over_four_l46_46118


namespace Joey_age_digit_sum_l46_46668

structure Ages :=
  (joey_age : ℕ)
  (chloe_age : ℕ)
  (zoe_age : ℕ)

def is_multiple (a b : ℕ) : Prop :=
  ∃ k, a = k * b

def sum_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

theorem Joey_age_digit_sum
  (C J Z : ℕ)
  (h1 : J = C + 1)
  (h2 : Z = 1)
  (h3 : ∃ n, C + n = (n + 1) * m)
  (m : ℕ) (hm : m = 9)
  (h4 : C - 1 = 36) :
  sum_digits (J + 37) = 12 :=
by
  sorry

end Joey_age_digit_sum_l46_46668


namespace snow_probability_january_first_week_l46_46127

noncomputable def P_snow_at_least_once_first_week : ℚ :=
  1 - ((2 / 3) ^ 4 * (3 / 4) ^ 3)

theorem snow_probability_january_first_week :
  P_snow_at_least_once_first_week = 11 / 12 :=
by
  sorry

end snow_probability_january_first_week_l46_46127


namespace televisions_selection_ways_l46_46612

noncomputable def combination (n k : ℕ) : ℕ :=
if h : k ≤ n then Nat.choose n k else 0

theorem televisions_selection_ways :
  let TypeA := 4
  let TypeB := 5
  let choosen := 3
  (∃ (n m : ℕ), n + m = choosen ∧ 1 ≤ n ∧ n ≤ TypeA ∧ 1 ≤ m ∧ m ≤ TypeB ∧
    combination TypeA n * combination TypeB m = 70) :=
by
  sorry

end televisions_selection_ways_l46_46612


namespace independent_trials_probability_l46_46810

theorem independent_trials_probability (p : ℝ) (q : ℝ) (ε : ℝ) (desired_prob : ℝ) 
    (h_p : p = 0.7) (h_q : q = 0.3) (h_ε : ε = 0.2) (h_desired_prob : desired_prob = 0.96) :
    ∃ n : ℕ, n > (p * q) / (desired_prob * ε^2) ∧ n = 132 :=
by
  sorry

end independent_trials_probability_l46_46810


namespace A_intersection_B_eq_C_l46_46281

def A := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def B := {x : ℝ | 0 < x ∧ x < 3}
def C := {x : ℝ | 0 < x ∧ x ≤ 2}

theorem A_intersection_B_eq_C : A ∩ B = C := 
by sorry

end A_intersection_B_eq_C_l46_46281


namespace least_addition_l46_46061

theorem least_addition (a b n : ℕ) (h_a : Nat.Prime a) (h_b : Nat.Prime b) (h_a_val : a = 23) (h_b_val : b = 29) (h_n : n = 1056) :
  ∃ m : ℕ, (m + n) % (a * b) = 0 ∧ m = 278 :=
by
  sorry

end least_addition_l46_46061


namespace hostel_initial_plan_l46_46577

variable (x : ℕ) -- representing the initial number of days

-- Define the conditions
def provisions_for_250_men (x : ℕ) : ℕ := 250 * x
def provisions_for_200_men_45_days : ℕ := 200 * 45

-- Prove the statement
theorem hostel_initial_plan (x : ℕ) (h : provisions_for_250_men x = provisions_for_200_men_45_days) :
  x = 36 :=
by
  sorry

end hostel_initial_plan_l46_46577


namespace infinite_bad_integers_l46_46615

theorem infinite_bad_integers (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  ∃ᶠ n in at_top, (¬(n^b + 1) ∣ (a^n + 1)) :=
by
  sorry

end infinite_bad_integers_l46_46615


namespace power_cycle_i_l46_46947

theorem power_cycle_i (i : ℂ) (h1 : i^1 = i) (h2 : i^2 = -1) (h3 : i^3 = -i) (h4 : i^4 = 1) :
  i^23 + i^75 = -2 * i :=
by
  sorry

end power_cycle_i_l46_46947


namespace fourth_number_unit_digit_l46_46701

def unit_digit (n : ℕ) : ℕ := n % 10

theorem fourth_number_unit_digit (a b c d : ℕ) (h₁ : a = 7858) (h₂: b = 1086) (h₃ : c = 4582) (h₄ : unit_digit (a * b * c * d) = 8) :
  unit_digit d = 4 :=
sorry

end fourth_number_unit_digit_l46_46701


namespace contrapositive_equivalence_l46_46539

variable (p q : Prop)

theorem contrapositive_equivalence : (p → ¬q) ↔ (q → ¬p) := by
  sorry

end contrapositive_equivalence_l46_46539


namespace calculate_expression_l46_46936

theorem calculate_expression :
  (10^4 - 9^4 + 8^4 - 7^4 + 6^4 - 5^4 + 4^4 - 3^4 + 2^4 - 1^4) +
  (10^2 + 9^2 + 5 * 8^2 + 5 * 7^2 + 9 * 6^2 + 9 * 5^2 + 13 * 4^2 + 13 * 3^2) = 7615 := by
  sorry

end calculate_expression_l46_46936


namespace mike_arcade_ratio_l46_46863

theorem mike_arcade_ratio :
  ∀ (weekly_pay food_cost hourly_rate play_minutes : ℕ),
    weekly_pay = 100 →
    food_cost = 10 →
    hourly_rate = 8 →
    play_minutes = 300 →
    (food_cost + (play_minutes / 60) * hourly_rate) / weekly_pay = 1 / 2 := 
by
  intros weekly_pay food_cost hourly_rate play_minutes h1 h2 h3 h4
  sorry

end mike_arcade_ratio_l46_46863


namespace compound_interest_doubling_time_l46_46804

theorem compound_interest_doubling_time :
  ∃ (t : ℕ), (0.15 : ℝ) = 0.15 ∧ ∀ (n : ℕ), (n = 1) →
               (2 : ℝ) < (1 + 0.15) ^ t ∧ t = 5 :=
by
  sorry

end compound_interest_doubling_time_l46_46804


namespace four_digit_div_by_14_l46_46272

theorem four_digit_div_by_14 (n : ℕ) (h₁ : 9450 + n < 10000) :
  (∃ k : ℕ, 9450 + n = 14 * k) ↔ (n = 8) := by
  sorry

end four_digit_div_by_14_l46_46272


namespace magnitude_of_power_l46_46800

open Complex

theorem magnitude_of_power (a b : ℝ) : abs ((Complex.mk 2 (2 * Real.sqrt 3)) ^ 6) = 4096 := by
  sorry

end magnitude_of_power_l46_46800


namespace total_weight_of_dumbbell_system_l46_46249

-- Definitions from the given conditions
def weight_pair1 : ℕ := 3
def weight_pair2 : ℕ := 5
def weight_pair3 : ℕ := 8

-- Goal: Prove that the total weight of the dumbbell system is 32 lbs
theorem total_weight_of_dumbbell_system :
  2 * weight_pair1 + 2 * weight_pair2 + 2 * weight_pair3 = 32 :=
by sorry

end total_weight_of_dumbbell_system_l46_46249


namespace multiples_of_5_in_4_digit_range_l46_46305

theorem multiples_of_5_in_4_digit_range : 
  let count_multiples := (9995 - 1000) / 5 + 1
  in count_multiples = 1800 :=
by
  sorry

end multiples_of_5_in_4_digit_range_l46_46305


namespace sum_of_consecutive_integers_l46_46635

theorem sum_of_consecutive_integers {a b : ℤ} (h1 : a < b)
  (h2 : b = a + 1)
  (h3 : a < Real.sqrt 3)
  (h4 : Real.sqrt 3 < b) :
  a + b = 3 := 
sorry

end sum_of_consecutive_integers_l46_46635


namespace combine_syllables_to_computer_l46_46546

/-- Conditions provided in the problem -/
def first_syllable : String := "ком" -- A big piece of a snowman
def second_syllable : String := "пьют" -- Something done by elephants at a watering hole
def third_syllable : String := "ер" -- The old name of the hard sign

/-- The result obtained by combining the three syllables should be "компьютер" -/
theorem combine_syllables_to_computer :
  (first_syllable ++ second_syllable ++ third_syllable) = "компьютер" :=
by
  -- Proof to be provided
  sorry

end combine_syllables_to_computer_l46_46546


namespace volume_of_box_is_correct_l46_46069

def metallic_sheet_initial_length : ℕ := 48
def metallic_sheet_initial_width : ℕ := 36
def square_cut_side_length : ℕ := 8

def box_length : ℕ := metallic_sheet_initial_length - 2 * square_cut_side_length
def box_width : ℕ := metallic_sheet_initial_width - 2 * square_cut_side_length
def box_height : ℕ := square_cut_side_length

def box_volume : ℕ := box_length * box_width * box_height

theorem volume_of_box_is_correct : box_volume = 5120 := by
  sorry

end volume_of_box_is_correct_l46_46069


namespace find_x_eq_l46_46458

-- Given conditions
variables (c b θ : ℝ)

-- The proof problem
theorem find_x_eq :
  ∃ x : ℝ, x^2 + c^2 * (Real.sin θ)^2 = (b - x)^2 ∧
          x = (b^2 - c^2 * (Real.sin θ)^2) / (2 * b) :=
by
    sorry

end find_x_eq_l46_46458


namespace maximum_minimum_cos_sin_cos_l46_46133

noncomputable def max_min_cos_sin_cos_product (x y z : ℝ) : ℝ × ℝ :=
  if x + y + z = π / 2 ∧ x ≥ y ∧ y ≥ z ∧ z ≥ π / 12 then
    let product := cos x * sin y * cos z
    (max product, min product)
  else (0, 0)

theorem maximum_minimum_cos_sin_cos :
  ∃ x y z : ℝ, 
    x + y + z = π / 2 ∧ x ≥ y ∧ y ≥ z ∧ z ≥ π / 12 ∧
    max_min_cos_sin_cos_product x y z = ( (2 + real.sqrt 3) / 8, 1 / 8) :=
by
  sorry

end maximum_minimum_cos_sin_cos_l46_46133


namespace jenny_spent_625_dollars_l46_46169

def adoption_fee := 50
def vet_visits_cost := 500
def monthly_food_cost := 25
def toys_cost := 200
def year_months := 12

def jenny_adoption_vet_share := (adoption_fee + vet_visits_cost) / 2
def jenny_food_share := (monthly_food_cost * year_months) / 2
def jenny_total_cost := jenny_adoption_vet_share + jenny_food_share + toys_cost

theorem jenny_spent_625_dollars :
  jenny_total_cost = 625 := by
  sorry

end jenny_spent_625_dollars_l46_46169


namespace common_difference_arithmetic_sequence_l46_46965

theorem common_difference_arithmetic_sequence
  (a : ℕ → ℝ)
  (h1 : ∃ a1 d, (∀ n, a n = a1 + (n - 1) * d))
  (h2 : a 7 - 2 * a 4 = -1)
  (h3 : a 3 = 0) :
  ∃ d, (∀ a1, (a1 + 2 * d = 0 ∧ -d = -1) → d = -1/2) :=
by
  sorry

end common_difference_arithmetic_sequence_l46_46965


namespace solve_equation1_solve_equation2_l46_46685

def equation1 (x : ℝ) := (x - 1) ^ 2 = 4
def equation2 (x : ℝ) := 2 * x ^ 3 = -16

theorem solve_equation1 (x : ℝ) (h : equation1 x) : x = 3 ∨ x = -1 := 
sorry

theorem solve_equation2 (x : ℝ) (h : equation2 x) : x = -2 := 
sorry

end solve_equation1_solve_equation2_l46_46685


namespace garage_sale_total_l46_46550

theorem garage_sale_total (treadmill chest_of_drawers television total_sales : ℝ)
  (h1 : treadmill = 100) 
  (h2 : chest_of_drawers = treadmill / 2) 
  (h3 : television = treadmill * 3) 
  (partial_sales : ℝ) 
  (h4 : partial_sales = treadmill + chest_of_drawers + television) 
  (h5 : partial_sales = total_sales * 0.75) : 
  total_sales = 600 := 
by
  sorry

end garage_sale_total_l46_46550


namespace probability_at_least_6_heads_8_flips_l46_46739

-- Define the probability calculation of getting at least 6 heads in 8 coin flips.
def probability_at_least_6_heads (n : ℕ) (k : ℕ) : ℚ :=
  (Nat.choose n k + Nat.choose n (k + 1) + Nat.choose n (k + 2)) / 2^n

theorem probability_at_least_6_heads_8_flips : 
  probability_at_least_6_heads 8 6 = 37 / 256 := 
by
  sorry

end probability_at_least_6_heads_8_flips_l46_46739


namespace ratio_of_r_l46_46721

theorem ratio_of_r
  (total : ℕ) (r_amount : ℕ) (pq_amount : ℕ)
  (h_total : total = 7000 )
  (h_r_amount : r_amount = 2800 )
  (h_pq_amount : pq_amount = total - r_amount) :
  (r_amount / Nat.gcd r_amount pq_amount, pq_amount / Nat.gcd r_amount pq_amount) = (2, 3) :=
by
  sorry

end ratio_of_r_l46_46721


namespace median_CD_eq_altitude_from_C_eq_centroid_G_eq_l46_46136

namespace Geometry

/-- Vertices of the triangle -/
def A : ℝ × ℝ := (4, 4)
def B : ℝ × ℝ := (-4, 2)
def C : ℝ × ℝ := (2, 0)

/-- Proof of the equation of the median CD on the side AB -/
theorem median_CD_eq : ∀ (x y : ℝ), 3 * x + 2 * y - 6 = 0 :=
sorry

/-- Proof of the equation of the altitude from C to AB -/
theorem altitude_from_C_eq : ∀ (x y : ℝ), 4 * x + y - 8 = 0 :=
sorry

/-- Proof of the coordinates of the centroid G of triangle ABC -/
theorem centroid_G_eq : ∃ (x y : ℝ), x = 2 / 3 ∧ y = 2 :=
sorry

end Geometry

end median_CD_eq_altitude_from_C_eq_centroid_G_eq_l46_46136


namespace rotary_club_eggs_needed_l46_46191

theorem rotary_club_eggs_needed 
  (small_children_tickets : ℕ := 53)
  (older_children_tickets : ℕ := 35)
  (adult_tickets : ℕ := 75)
  (senior_tickets : ℕ := 37)
  (waste_percentage : ℝ := 0.03)
  (extra_omelets : ℕ := 25)
  (eggs_per_extra_omelet : ℝ := 2.5) :
  53 * 1 + 35 * 2 + 75 * 3 + 37 * 4 + 
  Nat.ceil (waste_percentage * (53 * 1 + 35 * 2 + 75 * 3 + 37 * 4)) + 
  Nat.ceil (extra_omelets * eggs_per_extra_omelet) = 574 := 
by 
  sorry

end rotary_club_eggs_needed_l46_46191


namespace percentage_of_people_win_a_prize_l46_46168

-- Define the constants used in the problem
def totalMinnows : Nat := 600
def minnowsPerPrize : Nat := 3
def totalPlayers : Nat := 800
def minnowsLeft : Nat := 240

-- Calculate the number of minnows given away as prizes
def minnowsGivenAway : Nat := totalMinnows - minnowsLeft

-- Calculate the number of prizes given away
def prizesGivenAway : Nat := minnowsGivenAway / minnowsPerPrize

-- Calculate the percentage of people winning a prize
def percentageWinners : Nat := (prizesGivenAway * 100) / totalPlayers

-- Theorem to prove the percentage of winners
theorem percentage_of_people_win_a_prize : 
    percentageWinners = 15 := 
sorry

end percentage_of_people_win_a_prize_l46_46168


namespace calc_305_squared_minus_295_squared_l46_46597

theorem calc_305_squared_minus_295_squared :
  305^2 - 295^2 = 6000 := 
  by
    sorry

end calc_305_squared_minus_295_squared_l46_46597


namespace find_b_plus_c_l46_46986

-- Definitions based on the given conditions.
variables {A : ℝ} {a b c : ℝ}

-- The conditions in the problem
theorem find_b_plus_c
  (h_cosA : Real.cos A = 1 / 3)
  (h_a : a = Real.sqrt 3)
  (h_bc : b * c = 3 / 2) :
  b + c = Real.sqrt 7 :=
sorry

end find_b_plus_c_l46_46986


namespace tan_alpha_plus_pi_over_4_l46_46464

noncomputable def tan_sum_formula (α : ℝ) : ℝ :=
  (Real.tan α + Real.tan (Real.pi / 4)) / (1 - Real.tan α * Real.tan (Real.pi / 4))

theorem tan_alpha_plus_pi_over_4 
  (α : ℝ) 
  (h1 : Real.cos (2 * α) + Real.sin α * (2 * Real.sin α - 1) = 2 / 5) 
  (h2 : α ∈ Set.Ioo (Real.pi / 2) Real.pi) : 
  tan_sum_formula α = 1 / 7 := 
sorry

end tan_alpha_plus_pi_over_4_l46_46464


namespace calculate_order_cost_l46_46512

-- Defining the variables and given conditions
variables (C E S D W : ℝ)

-- Given conditions as assumptions
axiom h1 : (2 / 5) * C = E * S
axiom h2 : (1 / 4) * (3 / 5) * C = D * W

-- Theorem statement for the amount paid for the orders
theorem calculate_order_cost (C E S D W : ℝ) (h1 : (2 / 5) * C = E * S) (h2 : (1 / 4) * (3 / 5) * C = D * W) : 
  (9 / 20) * C = C - ((2 / 5) * C + (3 / 20) * C) :=
sorry

end calculate_order_cost_l46_46512


namespace number_of_pairs_satisfying_equation_l46_46608

theorem number_of_pairs_satisfying_equation :
  ∃ n : ℕ, n = 4998 ∧ (∀ x y : ℤ, x^2 + 7 * x * y + 6 * y^2 = 15^50 → (x, y) ≠ (0, 0)) ∧
  (∀ x y : ℤ, x^2 + 7 * x * y + 6 * y^2 = 15^50 → ((x + 6 * y) = (3 * 5) ^ a ∧ (x + y) = (3 ^ (50 - a) * 5 ^ (50 - b)) ∨
        (x + 6 * y) = -(3 * 5) ^ a ∧ (x + y) = -(3 ^ (50 - a) * 5 ^ (50 - b)) → (a + b = 50))) :=
sorry

end number_of_pairs_satisfying_equation_l46_46608


namespace max_candies_l46_46544

/-- There are 28 ones written on the board. Every minute, Karlsson erases two arbitrary numbers
and writes their sum on the board, and then eats an amount of candy equal to the product of 
the two erased numbers. Prove that the maximum number of candies he could eat in 28 minutes is 378. -/
theorem max_candies (karlsson_eats_max_candies : ℕ → ℕ → ℕ) (n : ℕ) (initial_count : n = 28) :
  (∀ a b, karlsson_eats_max_candies a b = a * b) →
  (∃ max_candies, max_candies = 378) :=
sorry

end max_candies_l46_46544


namespace tan_alpha_l46_46274

theorem tan_alpha {α : ℝ} (h : 3 * Real.sin α + 4 * Real.cos α = 5) : Real.tan α = 3 / 4 :=
by
  -- Proof goes here
  sorry

end tan_alpha_l46_46274


namespace probability_no_growth_pie_l46_46930

def total_pies : ℕ := 6
def growth_pies : ℕ := 2
def shrink_pies : ℕ := 4
def pies_given_mary : ℕ := 3

theorem probability_no_growth_pie : 
  (probability (λ distribution : finset (fin total_pies), 
                distribution.card = pies_given_mary ∧ 
                (distribution.count (λ x, x < growth_pies) = 0 ∨ 
                 (finset.range total_pies \ distribution).count (λ x, x < growth_pies) = 0)) = 0.4) :=
sorry

end probability_no_growth_pie_l46_46930


namespace find_positive_value_of_A_l46_46496

variable (A : ℝ)

-- Given conditions
def relation (A B : ℝ) : ℝ := A^2 + B^2

-- The proof statement
theorem find_positive_value_of_A (h : relation A 7 = 200) : A = Real.sqrt 151 := sorry

end find_positive_value_of_A_l46_46496


namespace initial_men_l46_46380

variable (P M : ℕ) -- P represents the provisions and M represents the initial number of men.

-- Conditons
def provision_lasts_20_days : Prop := P / (M * 20) = P / ((M + 200) * 15)

-- The proof problem
theorem initial_men (h : provision_lasts_20_days P M) : M = 600 :=
sorry

end initial_men_l46_46380


namespace count_four_digit_multiples_of_5_l46_46306

def is_multiple_of_5 (n : ℕ) : Prop :=
  n % 5 = 0

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def four_digit_multiples_of_5_count (lower upper : ℕ) :=
  (upper - lower + 1)

theorem count_four_digit_multiples_of_5 : 
  four_digit_multiples_of_5_count 200 1999 = 1800 :=
by
  sorry

end count_four_digit_multiples_of_5_l46_46306


namespace find_number_of_children_l46_46081

theorem find_number_of_children (adults children : ℕ) (adult_ticket_price child_ticket_price total_money change : ℕ) 
    (h1 : adult_ticket_price = 9) 
    (h2 : child_ticket_price = adult_ticket_price - 2) 
    (h3 : total_money = 40) 
    (h4 : change = 1) 
    (h5 : adults = 2) 
    (total_cost : total_money - change = adults * adult_ticket_price + children * child_ticket_price) : 
    children = 3 :=
sorry

end find_number_of_children_l46_46081


namespace cyclic_quadrilateral_JMIT_l46_46670

theorem cyclic_quadrilateral_JMIT
  (a b c : ℂ)
  (I J M N T : ℂ)
  (hI : I = -(a*b + b*c + c*a))
  (hJ : J = a*b - b*c + c*a)
  (hM : M = (b^2 + c^2) / 2)
  (hN : N = b*c)
  (hT : T = 2*a^2 - b*c) :
  ∃ (k : ℝ), k = ((M - I) * (T - J)) / ((J - I) * (T - M)) :=
by
  sorry

end cyclic_quadrilateral_JMIT_l46_46670


namespace no_growth_pie_probability_l46_46929

noncomputable def probability_no_growth_pies : ℝ :=
  let total_pies := 6
  let growth_pies := 2
  let shrink_pies := 4
  let pies_given := 3
  let total_combinations := Nat.choose total_pies pies_given
  let favorable_outcomes := Nat.choose shrink_pies 3 + Nat.choose shrink_pies 2 * Nat.choose growth_pies 1 + Nat.choose shrink_pies 1 * Nat.choose growth_pies 2
  in favorable_outcomes / total_combinations

theorem no_growth_pie_probability :
  probability_no_growth_pies = 0.4 :=
sorry

end no_growth_pie_probability_l46_46929


namespace trigonometric_inequality_solution_l46_46525

theorem trigonometric_inequality_solution (k : ℤ) :
  ∃ x : ℝ, x = - (3 * Real.pi) / 2 + 4 * Real.pi * k ∧
           (Real.cos (x / 2) + Real.sin (x / 2) ≤ (Real.sin x - 3) / Real.sqrt 2) :=
by
  sorry

end trigonometric_inequality_solution_l46_46525


namespace find_a_for_arithmetic_progression_roots_l46_46803

theorem find_a_for_arithmetic_progression_roots (x a : ℝ) : 
  (∀ (x : ℝ), x^4 - a*x^2 + 1 = 0) → 
  (∃ (t1 t2 : ℝ), t1 > 0 ∧ t2 > 0 ∧ (t2 = 9*t1) ∧ (t1 + t2 = a) ∧ (t1 * t2 = 1)) → 
  (a = 10/3) := 
  by 
    intros h1 h2
    sorry

end find_a_for_arithmetic_progression_roots_l46_46803


namespace train_speed_is_correct_l46_46093

-- Conditions
def train_length := 190.0152  -- in meters
def crossing_time := 17.1     -- in seconds

-- Convert units
def train_length_km := train_length / 1000  -- in kilometers
def crossing_time_hr := crossing_time / 3600  -- in hours

-- Statement of the proof problem
theorem train_speed_is_correct :
  (train_length_km / crossing_time_hr) = 40 :=
sorry

end train_speed_is_correct_l46_46093


namespace pole_intersection_height_l46_46895

theorem pole_intersection_height 
  (h1 h2 d : ℝ) 
  (h1pos : h1 = 30) 
  (h2pos : h2 = 90) 
  (dpos : d = 150) : 
  ∃ y, y = 22.5 :=
by
  sorry

end pole_intersection_height_l46_46895


namespace julia_money_left_l46_46349

def initial_amount : ℕ := 40

def amount_spent_on_game (initial : ℕ) : ℕ := initial / 2

def amount_left_after_game (initial : ℕ) (spent_game : ℕ) : ℕ := initial - spent_game

def amount_spent_on_in_game (left_after_game : ℕ) : ℕ := left_after_game / 4

def final_amount (left_after_game : ℕ) (spent_in_game : ℕ) : ℕ := left_after_game - spent_in_game

theorem julia_money_left (initial : ℕ) 
  (h_init : initial = initial_amount)
  (spent_game : ℕ)
  (h_spent_game : spent_game = amount_spent_on_game initial)
  (left_after_game : ℕ)
  (h_left_after_game : left_after_game = amount_left_after_game initial spent_game)
  (spent_in_game : ℕ)
  (h_spent_in_game : spent_in_game = amount_spent_on_in_game left_after_game)
  : final_amount left_after_game spent_in_game = 15 := by 
  sorry

end julia_money_left_l46_46349


namespace train_speed_on_time_l46_46714

theorem train_speed_on_time :
  ∃ (v : ℝ), 
  (∀ (d : ℝ) (t : ℝ),
    d = 133.33 ∧ 
    80 * (t + 1/3) = d ∧ 
    v * t = d) → 
  v = 100 :=
by
  sorry

end train_speed_on_time_l46_46714


namespace worm_in_apple_l46_46927

theorem worm_in_apple (radius : ℝ) (travel_distance : ℝ) (h_radius : radius = 31) (h_travel_distance : travel_distance = 61) :
  ∃ S : Set ℝ, ∀ point_on_path : ℝ, (point_on_path ∈ S) → false :=
by
  sorry

end worm_in_apple_l46_46927


namespace price_of_first_tea_x_l46_46189

theorem price_of_first_tea_x (x : ℝ) :
  let price_second := 135
  let price_third := 173.5
  let avg_price := 152
  let ratio := [1, 1, 2]
  1 * x + 1 * price_second + 2 * price_third = 4 * avg_price -> x = 126 :=
by
  intros price_second price_third avg_price ratio h
  sorry

end price_of_first_tea_x_l46_46189


namespace average_age_condition_l46_46070

theorem average_age_condition (n : ℕ) 
  (h1 : (↑n * 14) / n = 14) 
  (h2 : ((↑n * 14) + 34) / (n + 1) = 16) : 
  n = 9 := 
by 
-- Proof goes here
sorry

end average_age_condition_l46_46070


namespace max_distinct_rectangles_l46_46076

theorem max_distinct_rectangles : 
  ∃ (rectangles : Finset ℕ), (∀ n ∈ rectangles, n > 0) ∧ rectangles.sum id = 100 ∧ rectangles.card = 14 :=
by 
  sorry

end max_distinct_rectangles_l46_46076


namespace board_partition_possible_l46_46423

-- Definition of natural numbers m and n greater than 15
variables (m n : ℕ)
-- m > 15
def m_greater_than_15 := m > 15
-- n > 15
def n_greater_than_15 := n > 15

-- Definition of m and n divisibility conditions
def divisible_by_4_or_5 (x : ℕ) : Prop :=
  x % 4 = 0 ∨ x % 5 = 0

def partition_possible (m n : ℕ) : Prop :=
  (m % 4 = 0 ∧ n % 5 = 0) ∨ (m % 5 = 0 ∧ n % 4 = 0)

-- The final statement of Lean
theorem board_partition_possible :
  m_greater_than_15 m → n_greater_than_15 n → partition_possible m n :=
by
  intro h_m h_n
  sorry

end board_partition_possible_l46_46423


namespace probability_at_least_6_heads_l46_46756

-- Definitions of the binomial coefficient and probability function
def binom (n k : ℕ) : ℕ := Nat.choose n k

def probability (favorable total : ℕ) : ℚ := favorable / total

-- Proof problem statement
theorem probability_at_least_6_heads (flips : ℕ) (p : ℚ) 
  (h_flips : flips = 8) 
  (h_probability : p = probability (binom 8 6 + binom 8 7 + binom 8 8) (2 ^ flips)) : 
  p = 37 / 256 :=
by
  sorry

end probability_at_least_6_heads_l46_46756


namespace sequence_pattern_l46_46265

theorem sequence_pattern (a b c : ℝ) (h1 : a = 19.8) (h2 : b = 18.6) (h3 : c = 17.4) 
  (h4 : ∀ n, n = a ∨ n = b ∨ n = c ∨ n = 16.2 ∨ n = 15) 
  (H : ∀ x y, (y = x - 1.2) → 
    (x = a ∨ x = b ∨ x = c ∨ y = 16.2 ∨ y = 15)) :
  (16.2 = c - 1.2) ∧ (15 = (c - 1.2) - 1.2) :=
by
  sorry

end sequence_pattern_l46_46265


namespace total_letters_sent_l46_46518

-- Define the number of letters sent in each month
def letters_in_January : ℕ := 6
def letters_in_February : ℕ := 9
def letters_in_March : ℕ := 3 * letters_in_January

-- Theorem statement: the total number of letters sent across the three months
theorem total_letters_sent :
  letters_in_January + letters_in_February + letters_in_March = 33 := by
  sorry

end total_letters_sent_l46_46518


namespace bank_balance_after_two_years_l46_46918

-- Define the original amount deposited
def original_amount : ℝ := 5600

-- Define the interest rate
def interest_rate : ℝ := 0.07

-- Define the interest for each year based on the original amount
def interest_per_year : ℝ := original_amount * interest_rate

-- Define the total amount after two years
def total_amount_after_two_years : ℝ := original_amount + interest_per_year + interest_per_year

-- Define the target value
def target_value : ℝ := 6384

-- The theorem we aim to prove
theorem bank_balance_after_two_years : 
  total_amount_after_two_years = target_value := 
by
  -- Proof goes here
  sorry

end bank_balance_after_two_years_l46_46918


namespace lcm_fractions_l46_46941

theorem lcm_fractions (x : ℕ) (hx : x ≠ 0) : 
  (∀ (a b c : ℕ), (a = 4*x ∧ b = 5*x ∧ c = 6*x) → (Nat.lcm (Nat.lcm a b) c = 60 * x)) :=
by
  sorry

end lcm_fractions_l46_46941


namespace find_S6_l46_46887

variable (a_n : ℕ → ℝ) -- Assume a_n gives the nth term of an arithmetic sequence.
variable (S_n : ℕ → ℝ) -- Assume S_n gives the sum of the first n terms of the sequence.

-- Conditions:
axiom S_2_eq : S_n 2 = 2
axiom S_4_eq : S_n 4 = 10

-- Define what it means to find S_6
theorem find_S6 : S_n 6 = 18 :=
by
  sorry

end find_S6_l46_46887


namespace polynomial_condition_degree_n_l46_46253

open Polynomial

theorem polynomial_condition_degree_n 
  (P_n : ℤ[X]) (n : ℕ) (hn_pos : 0 < n) (hn_deg : P_n.degree = n) 
  (hx0 : P_n.eval 0 = 0)
  (hx_conditions : ∃ (a : ℤ) (b : Fin n → ℤ), ∀ i, P_n.eval (b i) = n) : 
  n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 := 
sorry

end polynomial_condition_degree_n_l46_46253


namespace total_letters_correct_l46_46516

-- Define the conditions
def letters_January := 6
def letters_February := 9
def letters_March := 3 * letters_January

-- Definition of the total number of letters sent
def total_letters := letters_January + letters_February + letters_March

-- The statement we need to prove in Lean
theorem total_letters_correct : total_letters = 33 := 
by
  sorry

end total_letters_correct_l46_46516


namespace age_problem_l46_46987

theorem age_problem (A B : ℕ) 
  (h1 : A + 10 = 2 * (B - 10))
  (h2 : A = B + 12) :
  B = 42 :=
sorry

end age_problem_l46_46987


namespace prob_at_least_6_heads_eq_l46_46750

-- define the number of coin flips
def n := 8

-- define the number of possible outcomes (2^n)
def total_outcomes := 2 ^ n

-- define the binomial coefficients for cases: 6 heads, 7 heads, 8 heads
def binom_8_6 := Nat.choose 8 6
def binom_8_7 := Nat.choose 8 7
def binom_8_8 := Nat.choose 8 8

-- calculate the favorable outcomes for at least 6 heads
def favorable_outcomes := binom_8_6 + binom_8_7 + binom_8_8

-- define the probability of getting at least 6 heads
def probability := (favorable_outcomes : ℚ) / total_outcomes

theorem prob_at_least_6_heads_eq : probability = 37 / 256 := by
  sorry

end prob_at_least_6_heads_eq_l46_46750


namespace projective_transformation_is_cross_ratio_preserving_l46_46134

theorem projective_transformation_is_cross_ratio_preserving (P : ℝ → ℝ) :
  (∃ a b c d : ℝ, (ad - bc ≠ 0) ∧ (∀ x : ℝ, P x = (a * x + b) / (c * x + d))) ↔
  (∀ x1 x2 x3 x4 : ℝ, (x1 - x3) * (x2 - x4) / ((x1 - x4) * (x2 - x3)) =
       (P x1 - P x3) * (P x2 - P x4) / ((P x1 - P x4) * (P x2 - P x3))) :=
sorry

end projective_transformation_is_cross_ratio_preserving_l46_46134


namespace third_pipe_empty_time_l46_46709

theorem third_pipe_empty_time (x : ℝ) :
  (1 / 60 : ℝ) + (1 / 120) - (1 / x) = (1 / 60) →
  x = 120 :=
by
  intros h
  sorry

end third_pipe_empty_time_l46_46709


namespace articles_selling_price_to_cost_price_eq_l46_46193

theorem articles_selling_price_to_cost_price_eq (C N : ℝ) (h_gain : 2 * C * N = 20 * C) : N = 10 :=
by
  sorry

end articles_selling_price_to_cost_price_eq_l46_46193


namespace same_terminal_side_l46_46244

theorem same_terminal_side (θ : ℝ) : (∃ k : ℤ, θ = 2 * k * π - π / 6) → θ = 11 * π / 6 :=
sorry

end same_terminal_side_l46_46244


namespace polly_to_sandy_ratio_l46_46693

variable {W P S : ℝ}
variable (h1 : S = (5/2) * W) (h2 : P = 2 * W)

theorem polly_to_sandy_ratio : P = (4/5) * S := by
  sorry

end polly_to_sandy_ratio_l46_46693


namespace meeting_anniversary_day_l46_46105

-- Define the input parameters for the problem
def initial_years : Set ℕ := {1668, 1669, 1670, 1671}
def meeting_day := "Friday"
def is_leap_year (year : ℕ) : Bool := (year % 4 = 0)

-- Define the theorem for the problem statement
theorem meeting_anniversary_day :
  ∀ (year : ℕ), year ∈ initial_years →
  let leap_years := (∑ n in range 1668, if is_leap_year n then 1 else 0)
  let total_days := 11 * 365 + leap_years
  let day_of_week := total_days % 7
  in (day_of_week = 0 ∧ probability Friday = 3 / 4) ∨ (day_of_week = 6 ∧ probability Thursday 1 / 4) :=
by
  sorry

end meeting_anniversary_day_l46_46105


namespace directrix_of_parabola_l46_46031

theorem directrix_of_parabola :
  ∀ (x : ℝ), y = x^2 / 4 → y = -1 :=
sorry

end directrix_of_parabola_l46_46031


namespace fraction_cubed_sum_l46_46151

theorem fraction_cubed_sum (x y : ℤ) (h1 : x = 3) (h2 : y = 4) :
  (x^3 + 3 * y^3) / 7 = 31 + 3 / 7 := by
  sorry

end fraction_cubed_sum_l46_46151


namespace distance_between_foci_of_ellipse_l46_46586

theorem distance_between_foci_of_ellipse :
  ∀ (ellipse: ℝ × ℝ → Prop),
    (∀ x y, ellipse (x, y) ↔ (x - 5)^2 / 25 + (y - 2)^2 / 4 = 1) →
    ∃ c : ℝ, 2 * c = 2 * Real.sqrt (25 - 4) :=
by
  intro ellipse h
  use Real.sqrt (25 - 4)
  sorry

end distance_between_foci_of_ellipse_l46_46586


namespace bottles_from_shop_c_correct_l46_46456

-- Definitions for the given conditions
def total_bottles := 550
def bottles_from_shop_a := 150
def bottles_from_shop_b := 180

-- Definition for the bottles from Shop C
def bottles_from_shop_c := total_bottles - (bottles_from_shop_a + bottles_from_shop_b)

-- The statement to prove
theorem bottles_from_shop_c_correct : bottles_from_shop_c = 220 :=
by
  -- proof will be filled later
  sorry

end bottles_from_shop_c_correct_l46_46456


namespace sum_of_consecutive_integers_of_sqrt3_l46_46639

theorem sum_of_consecutive_integers_of_sqrt3 {a b : ℤ} (h1 : a + 1 = b) (h2 : (a : ℝ) < Real.sqrt 3) (h3 : Real.sqrt 3 < (b : ℝ)) :
  a + b = 3 := by
  sorry

end sum_of_consecutive_integers_of_sqrt3_l46_46639


namespace decreased_amount_l46_46904

theorem decreased_amount {N A : ℝ} (h₁ : 0.20 * N - A = 6) (h₂ : N = 50) : A = 4 := by
  sorry

end decreased_amount_l46_46904


namespace courses_choice_l46_46400

theorem courses_choice (total_courses : ℕ) (chosen_courses : ℕ)
  (h_total_courses : total_courses = 5)
  (h_chosen_courses : chosen_courses = 2) :
  ∃ (ways : ℕ), ways = 60 ∧
    (ways = ((Nat.choose total_courses chosen_courses)^2) - 
            (Nat.choose total_courses chosen_courses) - 
            ((Nat.choose total_courses chosen_courses) * 
             (Nat.choose (total_courses - chosen_courses) chosen_courses))) :=
by
  sorry

end courses_choice_l46_46400


namespace interval_of_decrease_for_f_x_plus_1_l46_46154

def f_prime (x : ℝ) : ℝ := x^2 - 4*x + 3

theorem interval_of_decrease_for_f_x_plus_1 : 
  ∀ x, (f_prime (x + 1) < 0 ↔ 0 < x ∧ x < 2) :=
by 
  intro x
  sorry

end interval_of_decrease_for_f_x_plus_1_l46_46154


namespace probability_at_least_6_heads_l46_46753

-- Definitions of the binomial coefficient and probability function
def binom (n k : ℕ) : ℕ := Nat.choose n k

def probability (favorable total : ℕ) : ℚ := favorable / total

-- Proof problem statement
theorem probability_at_least_6_heads (flips : ℕ) (p : ℚ) 
  (h_flips : flips = 8) 
  (h_probability : p = probability (binom 8 6 + binom 8 7 + binom 8 8) (2 ^ flips)) : 
  p = 37 / 256 :=
by
  sorry

end probability_at_least_6_heads_l46_46753


namespace books_loaned_l46_46089

theorem books_loaned (L : ℕ)
  (initial_books : ℕ := 150)
  (end_year_books : ℕ := 100)
  (return_rate : ℝ := 0.60)
  (loan_rate : ℝ := 0.40)
  (returned_books : ℕ := (initial_books - end_year_books)) :
  loan_rate * (L : ℝ) = (returned_books : ℝ) → L = 125 := by
  intro h
  sorry

end books_loaned_l46_46089


namespace find_real_solution_to_given_equation_l46_46459

noncomputable def sqrt_96_minus_sqrt_84 : ℝ := Real.sqrt 96 - Real.sqrt 84

theorem find_real_solution_to_given_equation (x : ℝ) (hx : x + 4 ≥ 0) :
  x^2 + 6 * x + 6 * x * Real.sqrt (x + 4) = 60 ↔ x = sqrt_96_minus_sqrt_84 := 
by
  sorry

end find_real_solution_to_given_equation_l46_46459


namespace at_least_one_is_half_l46_46881

theorem at_least_one_is_half (x y z : ℝ) (h : x + y + z - 2 * (x * y + y * z + z * x) + 4 * x * y * z = 1 / 2) :
  x = 1 / 2 ∨ y = 1 / 2 ∨ z = 1 / 2 :=
sorry

end at_least_one_is_half_l46_46881


namespace comb_12_9_eq_220_l46_46791

theorem comb_12_9_eq_220 : (Nat.choose 12 9) = 220 := by
  sorry

end comb_12_9_eq_220_l46_46791


namespace michelle_total_payment_l46_46161
noncomputable def michelle_base_cost := 25
noncomputable def included_talk_time := 40 -- in hours
noncomputable def text_cost := 10 -- in cents per message
noncomputable def extra_talk_cost := 15 -- in cents per minute
noncomputable def february_texts_sent := 200
noncomputable def february_talk_time := 41 -- in hours

theorem michelle_total_payment : 
  25 + ((200 * 10) / 100) + (((41 - 40) * 60 * 15) / 100) = 54 := by
  sorry

end michelle_total_payment_l46_46161


namespace problem_solution_l46_46945

theorem problem_solution 
  (C : ℝ × ℝ := (0, -2)) 
  (r : ℝ := 2) 
  (ellipse : ℝ → ℝ × ℝ := λ phi, (3 * Real.cos phi, Real.sqrt 3 * Real.sin phi)) 
  (line_l : ℝ → ℝ × ℝ := λ theta, ((Real.sqrt 2) / 2, (Real.sqrt 2) / 2)) :
  (∀ x y : ℝ, x - y + 1 ≠ 0 ∨ x^2 + (y + 2)^2 ≠ 4) ∧
  (∀ t1 t2 : ℝ, let AB := Real.sqrt ((t1 + t2)^2 - 4 * t1 * t2) 
   in AB = (12 * Real.sqrt 2) / 7) :=
begin
  sorry
end

end problem_solution_l46_46945


namespace chemistry_more_than_physics_l46_46052

noncomputable def M : ℕ := sorry
noncomputable def P : ℕ := sorry
noncomputable def C : ℕ := sorry
noncomputable def x : ℕ := sorry

theorem chemistry_more_than_physics :
  M + P = 20 ∧ C = P + x ∧ (M + C) / 2 = 20 → x = 20 :=
by
  sorry

end chemistry_more_than_physics_l46_46052


namespace hydrogen_moles_formed_l46_46607

open Function

-- Define types for the substances involved in the reaction
structure Substance :=
  (name : String)
  (moles : ℕ)

-- Define the reaction
def reaction (NaH H2O NaOH H2 : Substance) : Prop :=
  NaH.moles = H2O.moles ∧ NaOH.moles = H2.moles

-- Given conditions
def NaH_initial : Substance := ⟨"NaH", 2⟩
def H2O_initial : Substance := ⟨"H2O", 2⟩
def NaOH_final : Substance := ⟨"NaOH", 2⟩
def H2_final : Substance := ⟨"H2", 2⟩

-- Problem statement in Lean
theorem hydrogen_moles_formed :
  reaction NaH_initial H2O_initial NaOH_final H2_final → H2_final.moles = 2 :=
by
  -- Skip proof
  sorry

end hydrogen_moles_formed_l46_46607


namespace sequence_b_l46_46454

theorem sequence_b (b : ℕ → ℝ) (h₁ : b 1 = 1)
  (h₂ : ∀ n : ℕ, n ≥ 1 → (b (n + 1)) ^ 4 = 64 * (b n) ^ 4) :
  b 50 = 2 ^ 49 := by
  sorry

end sequence_b_l46_46454


namespace more_time_running_than_skipping_l46_46410

def time_running : ℚ := 17 / 20
def time_skipping_rope : ℚ := 83 / 100

theorem more_time_running_than_skipping :
  time_running > time_skipping_rope :=
by
  -- sorry skips the proof
  sorry

end more_time_running_than_skipping_l46_46410


namespace batting_average_drop_l46_46688

theorem batting_average_drop 
    (avg : ℕ)
    (innings : ℕ)
    (high : ℕ)
    (high_low_diff : ℕ)
    (low : ℕ)
    (total_runs : ℕ)
    (new_avg : ℕ)

    (h1 : avg = 50)
    (h2 : innings = 40)
    (h3 : high = 174)
    (h4 : high = low + 172)
    (h5 : total_runs = avg * innings)
    (h6 : new_avg = (total_runs - high - low) / (innings - 2)) :

  avg - new_avg = 2 :=
by
  sorry

end batting_average_drop_l46_46688


namespace number_of_solutions_l46_46942

open Real

-- Define main condition
def condition (θ : ℝ) : Prop := sin θ * tan θ = 2 * (cos θ)^2

-- Define the interval and exclusions
def valid_theta (θ : ℝ) : Prop := 
  0 ≤ θ ∧ θ ≤ 2 * π ∧ ¬ ( ∃ k : ℤ, (θ = k * (π/2)) )

-- Define the set of thetas that satisfy both the condition and the valid interval
def valid_solutions (θ : ℝ) : Prop := valid_theta θ ∧ condition θ

-- Formal statement of the problem
theorem number_of_solutions : 
  ∃ (s : Finset ℝ), (∀ θ ∈ s, valid_solutions θ) ∧ (s.card = 4) := by
  sorry

end number_of_solutions_l46_46942


namespace apples_left_total_l46_46811

-- Define the initial conditions
def FrankApples : ℕ := 36
def SusanApples : ℕ := 3 * FrankApples
def SusanLeft : ℕ := SusanApples / 2
def FrankLeft : ℕ := (2 / 3) * FrankApples

-- Define the total apples left
def total_apples_left (SusanLeft FrankLeft : ℕ) : ℕ := SusanLeft + FrankLeft

-- Given conditions transformed to Lean
theorem apples_left_total : 
  total_apples_left (SusanApples / 2) ((2 / 3) * FrankApples) = 78 := by
  sorry

end apples_left_total_l46_46811


namespace carolyn_silverware_knives_percentage_l46_46450

theorem carolyn_silverware_knives_percentage :
  (let knives_initial := 6 in
   let forks_initial := 12 in
   let spoons_initial := 3 * knives_initial in
   let total_silverware_initial := knives_initial + forks_initial + spoons_initial in
   let knives_after_trade := 0 in
   let spoons_after_trade := spoons_initial + 6 in
   let total_silverware_after_trade := knives_after_trade + forks_initial + spoons_after_trade in
   percentage_knives := (knives_after_trade * 100) / total_silverware_after_trade in
   percentage_knives = 0) :=
by
  sorry

end carolyn_silverware_knives_percentage_l46_46450


namespace geometric_sequence_third_term_l46_46267

theorem geometric_sequence_third_term (q : ℝ) (b1 : ℝ) (h1 : abs q < 1)
    (h2 : b1 / (1 - q) = 8 / 5) (h3 : b1 * q = -1 / 2) :
    b1 * q^2 / 2 = 1 / 8 := by
  sorry

end geometric_sequence_third_term_l46_46267


namespace unique_solution_of_system_l46_46260

theorem unique_solution_of_system (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z)
  (h1 : x * (x + y + z) = 26) (h2 : y * (x + y + z) = 27) (h3 : z * (x + y + z) = 28) :
  x = 26 / 9 ∧ y = 3 ∧ z = 28 / 9 :=
by
  sorry

end unique_solution_of_system_l46_46260


namespace math_problem_l46_46130

theorem math_problem (a b : ℝ) (h : a * b < 0) : a^2 * |b| - b^2 * |a| + a * b * (|a| - |b|) = 0 :=
sorry

end math_problem_l46_46130


namespace average_speed_round_trip_l46_46428

/--
Let \( d = 150 \) miles be the distance from City \( X \) to City \( Y \).
Let \( v1 = 50 \) mph be the speed from \( X \) to \( Y \).
Let \( v2 = 30 \) mph be the speed from \( Y \) to \( X \).
Then the average speed for the round trip is 37.5 mph.
-/
theorem average_speed_round_trip :
  let d := 150
  let v1 := 50
  let v2 := 30
  (2 * d) / ((d / v1) + (d / v2)) = 37.5 :=
by
  sorry

end average_speed_round_trip_l46_46428


namespace reciprocal_of_minus_one_over_2023_l46_46045

theorem reciprocal_of_minus_one_over_2023 : (1 / (- (1 / 2023))) = -2023 := 
by
  sorry

end reciprocal_of_minus_one_over_2023_l46_46045


namespace problem_equivalence_l46_46958

noncomputable def a : ℝ := -1
noncomputable def b : ℝ := 3

theorem problem_equivalence (i : ℂ) (hi : i^2 = -1) : 
  (a + 3 * i = (b + i) * i) :=
by
  -- The complex number definitions
  let lhs := a + 3 * i
  let rhs := (b + i) * i

  -- Confirming the parts
  calc
  lhs = -1 + 3 * i : by rfl
  ... = rhs       : by
    simp [a, b]
    rw [hi]
    ring

-- To skip the proof add sorry at the end.
sorry

end problem_equivalence_l46_46958


namespace pyramid_volume_l46_46530

noncomputable def volume_of_pyramid (AB AD BD AE : ℝ) (p : AB = 9 ∧ AD = 10 ∧ BD = 11 ∧ AE = 10.5) : ℝ :=
  1 / 3 * (60 * (2 ^ (1 / 2))) * (5 * (2 ^ (1 / 2)))

theorem pyramid_volume (AB AD BD AE : ℝ) (h1 : AB = 9) (h2 : AD = 10) (h3 : BD = 11) (h4 : AE = 10.5)
  (V : ℝ) (hV : V = 200) : 
  volume_of_pyramid AB AD BD AE (⟨h1, ⟨h2, ⟨h3, h4⟩⟩⟩) = V :=
sorry

end pyramid_volume_l46_46530


namespace f_2_plus_f_5_eq_2_l46_46462

noncomputable def f : ℝ → ℝ := sorry

open Real

-- Conditions: f(3^x) = x * log 9
axiom f_cond (x : ℝ) : f (3^x) = x * log 9

-- Question: f(2) + f(5) = 2
theorem f_2_plus_f_5_eq_2 : f 2 + f 5 = 2 := sorry

end f_2_plus_f_5_eq_2_l46_46462


namespace number_of_correct_calculations_is_one_l46_46198

/- Given conditions -/
def cond1 (a : ℝ) : Prop := a^2 * a^2 = 2 * a^2
def cond2 (a b : ℝ) : Prop := (a - b)^2 = a^2 - b^2
def cond3 (a : ℝ) : Prop := a^2 + a^3 = a^5
def cond4 (a b : ℝ) : Prop := (-2 * a^2 * b^3)^3 = -6 * a^6 * b^3
def cond5 (a : ℝ) : Prop := (-a^3)^2 / a = a^5

/- Statement to prove the number of correct calculations is 1 -/
theorem number_of_correct_calculations_is_one :
  (¬ (cond1 a)) ∧ (¬ (cond2 a b)) ∧ (¬ (cond3 a)) ∧ (¬ (cond4 a b)) ∧ (cond5 a) → 1 = 1 :=
by
  sorry

end number_of_correct_calculations_is_one_l46_46198


namespace range_of_x_l46_46528

theorem range_of_x (a : ℝ) (x : ℝ) (h₁ : a = 1) (h₂ : (x - a) * (x - 3 * a) < 0) (h₃ : 2 < x ∧ x ≤ 3) : 2 < x ∧ x < 3 :=
by sorry

end range_of_x_l46_46528


namespace race_track_cost_l46_46602

def toy_car_cost : ℝ := 0.95
def num_toy_cars : ℕ := 4
def total_money : ℝ := 17.80
def money_left : ℝ := 8.00

theorem race_track_cost :
  total_money - num_toy_cars * toy_car_cost - money_left = 6.00 :=
by
  sorry

end race_track_cost_l46_46602


namespace alfred_bill_days_l46_46067

-- Definitions based on conditions
def combined_work_rate := 1 / 24
def alfred_to_bill_ratio := 2 / 3

-- Theorem statement
theorem alfred_bill_days (A B : ℝ) (ha : A = alfred_to_bill_ratio * B) (hcombined : A + B = combined_work_rate) : 
  A = 1 / 60 ∧ B = 1 / 40 :=
by
  sorry

end alfred_bill_days_l46_46067


namespace area_of_L_equals_22_l46_46734

-- Define the dimensions of the rectangles
def big_rectangle_length := 8
def big_rectangle_width := 5
def small_rectangle_length := big_rectangle_length - 2
def small_rectangle_width := big_rectangle_width - 2

-- Define the areas
def area_big_rectangle := big_rectangle_length * big_rectangle_width
def area_small_rectangle := small_rectangle_length * small_rectangle_width

-- Define the area of the "L" shape
def area_L := area_big_rectangle - area_small_rectangle

-- State the theorem
theorem area_of_L_equals_22 : area_L = 22 := by
  -- The proof would go here
  sorry

end area_of_L_equals_22_l46_46734


namespace paper_thickness_after_folds_l46_46223

def folded_thickness (initial_thickness : ℝ) (folds : ℕ) : ℝ :=
  initial_thickness * 2^folds

theorem paper_thickness_after_folds :
  folded_thickness 0.1 4 = 1.6 :=
by
  sorry

end paper_thickness_after_folds_l46_46223


namespace tan_alpha_of_sin_alpha_eq_neg_sqrt3_div_2_l46_46956

theorem tan_alpha_of_sin_alpha_eq_neg_sqrt3_div_2 (alpha : ℝ) 
  (h1 : Real.sin alpha = - (Real.sqrt 3) / 2) 
  (h2 : 3 * π / 2 < alpha ∧ alpha < 2 * π) : 
  Real.tan alpha = - Real.sqrt 3 := 
by 
  sorry

end tan_alpha_of_sin_alpha_eq_neg_sqrt3_div_2_l46_46956


namespace trip_time_difference_l46_46235

-- Define the speed of the motorcycle
def speed : ℤ := 60

-- Define the distances for the two trips
def distance1 : ℤ := 360
def distance2 : ℤ := 420

-- Define the time calculation function
def time (distance speed : ℤ) : ℤ := distance / speed

-- Prove the problem statement
theorem trip_time_difference : (time distance2 speed - time distance1 speed) * 60 = 60 := by
  -- Provide the proof here
  sorry

end trip_time_difference_l46_46235


namespace gcd_of_repeated_three_digit_integers_is_1001001_l46_46438

theorem gcd_of_repeated_three_digit_integers_is_1001001 :
  ∀ (n : ℕ), (100 ≤ n ∧ n <= 999) →
  ∃ d : ℕ, d = 1001001 ∧
    (∀ m : ℕ, m = n * 1001001 →
      ∃ k : ℕ, m = k * d) :=
by
  sorry

end gcd_of_repeated_three_digit_integers_is_1001001_l46_46438


namespace find_first_number_l46_46540

theorem find_first_number 
  (second_number : ℕ)
  (increment : ℕ)
  (final_number : ℕ)
  (h1 : second_number = 45)
  (h2 : increment = 11)
  (h3 : final_number = 89)
  : ∃ first_number : ℕ, first_number + increment = second_number := 
by
  sorry

end find_first_number_l46_46540


namespace integer_solutions_set_l46_46603

theorem integer_solutions_set :
  {x : ℤ | 2 * x + 4 > 0 ∧ 1 + x ≥ 2 * x - 1} = {-1, 0, 1, 2} :=
by {
  sorry
}

end integer_solutions_set_l46_46603


namespace election_winner_votes_l46_46415

theorem election_winner_votes (V : ℝ) : (0.62 * V = 806) → (0.62 * V) - (0.38 * V) = 312 → 0.62 * V = 806 :=
by
  intro hWin hDiff
  exact hWin

end election_winner_votes_l46_46415


namespace total_fruits_is_174_l46_46087

def basket1_apples : ℕ := 9
def basket1_oranges : ℕ := 15
def basket1_bananas : ℕ := 14
def basket1_grapes : ℕ := 12

def basket4_apples : ℕ := basket1_apples - 2
def basket4_oranges : ℕ := basket1_oranges - 2
def basket4_bananas : ℕ := basket1_bananas - 2
def basket4_grapes : ℕ := basket1_grapes - 2

def basket5_apples : ℕ := basket1_apples + 3
def basket5_oranges : ℕ := basket1_oranges - 5
def basket5_bananas : ℕ := basket1_bananas
def basket5_grapes : ℕ := basket1_grapes

def basket6_bananas : ℕ := basket1_bananas * 2
def basket6_grapes : ℕ := basket1_grapes / 2

def total_fruits_b1_3 : ℕ := basket1_apples + basket1_oranges + basket1_bananas + basket1_grapes
def total_fruits_b4 : ℕ := basket4_apples + basket4_oranges + basket4_bananas + basket4_grapes
def total_fruits_b5 : ℕ := basket5_apples + basket5_oranges + basket5_bananas + basket5_grapes
def total_fruits_b6 : ℕ := basket6_bananas + basket6_grapes

def total_fruits_all : ℕ := total_fruits_b1_3 + total_fruits_b4 + total_fruits_b5 + total_fruits_b6

theorem total_fruits_is_174 : total_fruits_all = 174 := by
  -- proof will go here
  sorry

end total_fruits_is_174_l46_46087


namespace x_coordinate_of_tangent_point_l46_46286

noncomputable def f (x a : ℝ) : ℝ := Real.exp x + a * Real.exp (-x)

theorem x_coordinate_of_tangent_point 
  (a : ℝ) 
  (h_even : ∀ x : ℝ, f x a = f (-x) a)
  (h_slope : ∃ m : ℝ, Real.exp m - Real.exp (-m) = 3 / 2) : 
  ∃ m : ℝ, m = Real.log 2 := 
by
  sorry

end x_coordinate_of_tangent_point_l46_46286


namespace head_start_l46_46436

theorem head_start (V_b : ℝ) (S : ℝ) : 
  ((7 / 4) * V_b) = V_b → 
  196 = (196 - S) → 
  S = 84 := 
sorry

end head_start_l46_46436


namespace max_profit_is_45_6_l46_46232

noncomputable def profit_A (x : ℝ) : ℝ := 5.06 * x - 0.15 * x^2
noncomputable def profit_B (x : ℝ) : ℝ := 2 * x

noncomputable def total_profit (x : ℝ) : ℝ :=
  profit_A x + profit_B (15 - x)

theorem max_profit_is_45_6 : 
  ∃ x, 0 ≤ x ∧ x ≤ 15 ∧ total_profit x = 45.6 :=
by
  sorry

end max_profit_is_45_6_l46_46232


namespace any_nat_in_frac_l46_46715

theorem any_nat_in_frac (n : ℕ) : ∃ x y : ℕ, y ≠ 0 ∧ x^2 = y^3 * n := by
  sorry

end any_nat_in_frac_l46_46715


namespace cos_of_vector_dot_product_l46_46145

open Real

noncomputable def cos_value (x : ℝ) : ℝ := cos (x + π / 4)

theorem cos_of_vector_dot_product (x : ℝ)
  (h1 : π / 4 < x)
  (h2 : x < π / 2)
  (h3 : (sqrt 2) * cos x + (sqrt 2) * sin x = 8 / 5) :
  cos_value x = - 3 / 5 :=
by
  sorry

end cos_of_vector_dot_product_l46_46145


namespace negation_of_universal_prop_l46_46126

theorem negation_of_universal_prop :
  (¬ ∀ x : ℝ, x^2 - 2*x + 4 ≤ 0) ↔ (∃ x : ℝ, x^2 - 2*x + 4 > 0) :=
by
  sorry

end negation_of_universal_prop_l46_46126


namespace log_exp_identity_l46_46475

noncomputable def a : ℝ := Real.log 3 / Real.log 4

theorem log_exp_identity : 2^a + 2^(-a) = (4 * Real.sqrt 3) / 3 := 
by
  sorry

end log_exp_identity_l46_46475


namespace sum_mod_five_l46_46868

theorem sum_mod_five {n : ℕ} (h_pos : 0 < n) :
  (1^n + 2^n + 3^n + 4^n) % 5 = 0 ↔ ¬ (∃ k : ℕ, n = 4 * k) :=
sorry

end sum_mod_five_l46_46868


namespace probability_one_no_GP_l46_46932

def num_pies : ℕ := 6
def growth_pies : ℕ := 2
def shrink_pies : ℕ := 4
def picked_pies : ℕ := 3
def total_outcomes : ℕ := Nat.choose num_pies picked_pies

def fav_outcomes : ℕ := Nat.choose shrink_pies 2 -- Choosing 2 out of the 4 SP

def probability_complementary : ℚ := fav_outcomes / total_outcomes
def probability : ℚ := 1 - probability_complementary

theorem probability_one_no_GP :
  probability = 0.4 := by
  sorry

end probability_one_no_GP_l46_46932


namespace eleventh_anniversary_days_l46_46101

-- Define the conditions
def is_leap_year (year : ℕ) : Prop := year % 4 = 0

def initial_years : Set ℕ := {1668, 1669, 1670, 1671}

def initial_day := "Friday"

noncomputable def day_after_11_years (start_year : ℕ) : String :=
  let total_days := 4015 + (if is_leap_year start_year then 3 else 2)
  if total_days % 7 = 0 then "Friday"
  else "Thursday"

-- Define the proposition to prove
theorem eleventh_anniversary_days : 
  (∀ year ∈ initial_years, 
    (if day_after_11_years year = "Friday" then (3 : ℝ) / 4 else (1 : ℝ) / 4) = 
    (if year = 1668 ∨ year = 1670 ∨ year = 1671 then (3 : ℝ) / 4 else (1 : ℝ) / 4)) := 
sorry

end eleventh_anniversary_days_l46_46101


namespace subtraction_result_l46_46805

theorem subtraction_result : 3.05 - 5.678 = -2.628 := 
by
  sorry

end subtraction_result_l46_46805


namespace find_a_l46_46276

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 + a / ((Real.exp (2 * x)) - 1)

theorem find_a : ∃ a : ℝ, (∀ x : ℝ, f a x = -f a (-x)) → a = 2 :=
by
  sorry

end find_a_l46_46276


namespace distant_a3b3_2ab_sqrt_ab_l46_46188

def is_distant (x y m : ℝ) : Prop :=
  |x - |y - m|| = m

theorem distant_a3b3_2ab_sqrt_ab (a b : ℝ) (h : a ≠ b) (ha : 0 < a) (hb : 0 < b) :
  is_distant (a^3 + b^3) (2 * a * b * Real.sqrt (a * b)) (a^3 + b^3) := by
  -- Proof to be provided
  sorry

end distant_a3b3_2ab_sqrt_ab_l46_46188


namespace lowest_price_correct_l46_46717

noncomputable def lowest_price (cost_per_component shipping_cost_per_unit fixed_costs number_of_components : ℕ) : ℕ :=
(cost_per_component + shipping_cost_per_unit) * number_of_components + fixed_costs

theorem lowest_price_correct :
  lowest_price 80 5 16500 150 / 150 = 195 :=
by
  sorry

end lowest_price_correct_l46_46717


namespace arun_weight_upper_limit_l46_46843

theorem arun_weight_upper_limit (weight : ℝ) (avg_weight : ℝ) 
  (arun_opinion : 66 < weight ∧ weight < 72) 
  (brother_opinion : 60 < weight ∧ weight < 70) 
  (average_condition : avg_weight = 68) : weight ≤ 70 :=
by
  sorry

end arun_weight_upper_limit_l46_46843


namespace original_number_is_19_l46_46899

theorem original_number_is_19 (x : ℤ) (h : (x + 4) % 23 = 0) : x = 19 := 
by 
  sorry

end original_number_is_19_l46_46899


namespace max_min_cos_sin_cos_l46_46131

theorem max_min_cos_sin_cos (x y z : ℝ) (h1 : x ≥ y) (h2 : y ≥ z) (h3 : z ≥ π / 12) (h4 : x + y + z = π / 2) :
  ∃ (max_val min_val : ℝ), 
    (max_val = (2 + Real.sqrt 3) / 8) ∧ 
    (min_val = 1 / 8) ∧ 
    max_val = max (cos x * sin y * cos z) ∧ 
    min_val = min (cos x * sin y * cos z) :=
  sorry

end max_min_cos_sin_cos_l46_46131


namespace max_odd_numbers_in_pyramid_l46_46785

-- Define the properties of the pyramid
def is_sum_of_immediate_below (p : Nat → Nat → Nat) : Prop :=
  ∀ r c : Nat, r > 0 → p r c = p (r - 1) c + p (r - 1) (c + 1)

-- Define what it means for a number to be odd
def is_odd (n : Nat) : Prop := n % 2 = 1

-- Define the pyramid structure and number of rows
def pyramid (n : Nat) := { p : Nat → Nat → Nat // is_sum_of_immediate_below p ∧ n = 6 }

-- Theorem statement
theorem max_odd_numbers_in_pyramid (p : Nat → Nat → Nat) (h : is_sum_of_immediate_below p ∧ 6 = 6) : ∃ k : Nat, (∀ i j, is_odd (p i j) → k ≤ 14) := 
sorry

end max_odd_numbers_in_pyramid_l46_46785


namespace range_of_a_l46_46160

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, |x - a| + |x - 1| ≤ 3) → -2 ≤ a ∧ a ≤ 4 :=
by
  intro h
  sorry

end range_of_a_l46_46160


namespace total_books_l46_46401

-- Conditions
def TimsBooks : Nat := 44
def SamsBooks : Nat := 52
def AlexsBooks : Nat := 65
def KatiesBooks : Nat := 37

-- Theorem Statement
theorem total_books :
  TimsBooks + SamsBooks + AlexsBooks + KatiesBooks = 198 :=
by
  sorry

end total_books_l46_46401


namespace converse_proposition_l46_46192

-- Define the predicate variables p and q
variables (p q : Prop)

-- State the theorem about the converse of the proposition
theorem converse_proposition (hpq : p → q) : q → p :=
sorry

end converse_proposition_l46_46192


namespace tree_drops_leaves_on_fifth_day_l46_46996

def initial_leaves := 340
def daily_drop_fraction := 1 / 10

noncomputable def leaves_after_day (n: ℕ) : ℕ :=
  match n with
  | 0 => initial_leaves
  | 1 => initial_leaves - Nat.floor (initial_leaves * daily_drop_fraction)
  | 2 => leaves_after_day 1 - Nat.floor (leaves_after_day 1 * daily_drop_fraction)
  | 3 => leaves_after_day 2 - Nat.floor (leaves_after_day 2 * daily_drop_fraction)
  | 4 => leaves_after_day 3 - Nat.floor (leaves_after_day 3 * daily_drop_fraction)
  | _ => 0  -- beyond the 4th day

theorem tree_drops_leaves_on_fifth_day : leaves_after_day 4 = 225 := by
  -- We'll skip the detailed proof here, focusing on the statement
  sorry

end tree_drops_leaves_on_fifth_day_l46_46996


namespace sum_of_consecutive_integers_l46_46645

theorem sum_of_consecutive_integers (a b : ℤ) (h1 : a + 1 = b) (h2 : a < real.sqrt 3) (h3 : real.sqrt 3 < b) : a + b = 3 :=
sorry

end sum_of_consecutive_integers_l46_46645


namespace smallest_n_with_290_trailing_zeros_in_factorial_l46_46806

def trailing_zeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 5^2) + (n / 5^3) + (n / 5^4) + (n / 5^5) + (n / 5^6) -- sum until the division becomes zero

theorem smallest_n_with_290_trailing_zeros_in_factorial : 
  ∀ (n : ℕ), n >= 1170 ↔ trailing_zeros n >= 290 ∧ trailing_zeros (n-1) < 290 := 
by { sorry }

end smallest_n_with_290_trailing_zeros_in_factorial_l46_46806


namespace percentage_of_first_relative_to_second_l46_46335

theorem percentage_of_first_relative_to_second (X : ℝ) 
  (first_number : ℝ := 8/100 * X) 
  (second_number : ℝ := 16/100 * X) :
  (first_number / second_number) * 100 = 50 := 
sorry

end percentage_of_first_relative_to_second_l46_46335


namespace solve_equation_l46_46379

theorem solve_equation (x y z : ℕ) (h1 : 2^x + 5^y + 63 = z!) (h2 : z ≥ 5) : 
  (x = 5 ∧ y = 2 ∧ z = 5) ∨ (x = 4 ∧ y = 4 ∧ z = 6) :=
sorry

end solve_equation_l46_46379


namespace point_B_in_first_quadrant_l46_46065

def is_first_quadrant (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 > 0

theorem point_B_in_first_quadrant : is_first_quadrant (1, 2) :=
by
  sorry

end point_B_in_first_quadrant_l46_46065


namespace find_x_satisfies_equation_l46_46858

theorem find_x_satisfies_equation :
  let x : ℤ := -14
  ∃ x : ℤ, (36 - x) - (14 - x) = 2 * ((36 - x) - (18 - x)) :=
by
  let x := -14
  use x
  sorry

end find_x_satisfies_equation_l46_46858


namespace baker_sold_more_cakes_than_pastries_l46_46246

theorem baker_sold_more_cakes_than_pastries (cakes_sold pastries_sold : ℕ) 
  (h_cakes_sold : cakes_sold = 158) (h_pastries_sold : pastries_sold = 147) : 
  (cakes_sold - pastries_sold) = 11 := by
  sorry

end baker_sold_more_cakes_than_pastries_l46_46246


namespace percent_of_flowers_are_daisies_l46_46233

-- Definitions for the problem
def total_flowers (F : ℕ) := F
def blue_flowers (F : ℕ) := (7/10) * F
def red_flowers (F : ℕ) := (3/10) * F
def blue_tulips (F : ℕ) := (1/2) * (7/10) * F
def blue_daisies (F : ℕ) := (7/10) * F - (1/2) * (7/10) * F
def red_daisies (F : ℕ) := (2/3) * (3/10) * F
def total_daisies (F : ℕ) := blue_daisies F + red_daisies F
def percentage_of_daisies (F : ℕ) := (total_daisies F / F) * 100

-- The statement to prove
theorem percent_of_flowers_are_daisies (F : ℕ) (hF : F > 0) :
  percentage_of_daisies F = 55 := by
  sorry

end percent_of_flowers_are_daisies_l46_46233


namespace cinematic_academy_members_l46_46402

theorem cinematic_academy_members (h1 : ∀ x, x / 4 ≥ 196.25 → x ≥ 785) : 
  ∃ n : ℝ, 1 / 4 * n = 196.25 ∧ n = 785 :=
by
  sorry

end cinematic_academy_members_l46_46402


namespace even_parts_impossible_odd_parts_possible_l46_46778

theorem even_parts_impossible (n m : ℕ) (h₁ : n = 1) (h₂ : ∀ k, m = n + 2 * k) : n + 2 * m ≠ 100 := by
  -- Proof omitted
  sorry

theorem odd_parts_possible (n m : ℕ) (h₁ : n = 1) (h₂ : ∀ k, m = n + 2 * k) : ∃ k, n + 2 * k = 2017 := by
  -- Proof omitted
  sorry

end even_parts_impossible_odd_parts_possible_l46_46778


namespace perp_line_slope_zero_l46_46838

theorem perp_line_slope_zero {k : ℝ} (h : ∀ x : ℝ, ∃ y : ℝ, y = k * x + 1 ∧ x = 1 → false) : k = 0 :=
sorry

end perp_line_slope_zero_l46_46838


namespace area_of_rectangle_at_stage_8_l46_46328

-- Define the conditions given in the problem
def square_side_length : ℝ := 4
def square_area : ℝ := square_side_length * square_side_length
def stages : ℕ := 8

-- Define the statement to be proved
theorem area_of_rectangle_at_stage_8 : (stages * square_area) = 128 := 
by 
  have h1 : square_area = 16 := by
    unfold square_area
    norm_num
  have h2 : (stages * square_area) = 8 * 16 := by
    unfold stages
    rw h1
  rw h2
  norm_num
  sorry

end area_of_rectangle_at_stage_8_l46_46328


namespace neg_forall_sin_gt_zero_l46_46142

theorem neg_forall_sin_gt_zero :
  ¬ (∀ x : ℝ, Real.sin x > 0) ↔ ∃ x : ℝ, Real.sin x ≤ 0 := 
sorry

end neg_forall_sin_gt_zero_l46_46142


namespace reciprocal_neg_one_over_2023_eq_neg_2023_l46_46042

theorem reciprocal_neg_one_over_2023_eq_neg_2023 : (1 / (-1 / (2023 : ℝ))) = -2023 :=
by
  sorry

end reciprocal_neg_one_over_2023_eq_neg_2023_l46_46042


namespace halfway_fraction_l46_46215

theorem halfway_fraction (a b : ℚ) (ha : a = 3/4) (hb : b = 5/6) : (a + b) / 2 = 19/24 :=
by
  rw [ha, hb] -- replace a and b with 3/4 and 5/6 respectively
  have h1 : 3/4 + 5/6 = 19/12,
  { norm_num, -- ensures 3/4 + 5/6 = 19/12
    linarith },
  rw h1, -- replace a + b with 19/12
  norm_num -- ensures (19/12) / 2 = 19/24

end halfway_fraction_l46_46215


namespace sum_of_consecutive_integers_a_lt_sqrt3_lt_b_l46_46642

theorem sum_of_consecutive_integers_a_lt_sqrt3_lt_b 
  (a b : ℤ) (h1 : a < b) (h2 : ∀ x : ℤ, x ≤ a → x < b) (h3 : a < Real.sqrt 3) (h4 : Real.sqrt 3 < b) : 
  a + b = 3 :=
by
  sorry

end sum_of_consecutive_integers_a_lt_sqrt3_lt_b_l46_46642


namespace not_prime_5n_plus_3_l46_46871

theorem not_prime_5n_plus_3 (n a b : ℕ) (hn_pos : n > 0) (ha_pos : a > 0) (hb_pos : b > 0)
  (ha : 2 * n + 1 = a^2) (hb : 3 * n + 1 = b^2) : ¬Prime (5 * n + 3) :=
by
  sorry

end not_prime_5n_plus_3_l46_46871


namespace find_x_l46_46807

theorem find_x (x : ℚ) (h : x ≠ 2 ∧ x ≠ 4/5) :
  (x^2 - 11*x + 24)/(x - 2) + (5*x^2 + 22*x - 48)/(5*x - 4) = -7 → x = -4/3 :=
by
  intro h1
  sorry

end find_x_l46_46807


namespace leaves_dropped_on_fifth_day_l46_46994

theorem leaves_dropped_on_fifth_day 
  (initial_leaves : ℕ)
  (days : ℕ)
  (drops_per_day : ℕ)
  (total_dropped_four_days : ℕ)
  (leaves_dropped_fifth_day : ℕ)
  (h1 : initial_leaves = 340)
  (h2 : days = 4)
  (h3 : drops_per_day = initial_leaves / 10)
  (h4 : total_dropped_four_days = drops_per_day * days)
  (h5 : leaves_dropped_fifth_day = initial_leaves - total_dropped_four_days) :
  leaves_dropped_fifth_day = 204 :=
by
  sorry

end leaves_dropped_on_fifth_day_l46_46994


namespace ellipse_foci_distance_l46_46591

noncomputable def distance_between_foci_of_ellipse (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 - b^2)

theorem ellipse_foci_distance :
  ∀ (a b : ℝ), (a = 5) → (b = 2) →
  distance_between_foci_of_ellipse a b = Real.sqrt 21 :=
by
  intros a b ha hb
  rw [ha, hb]
  -- The rest of the proof is omitted
  sorry

end ellipse_foci_distance_l46_46591


namespace correct_negation_of_exactly_one_even_l46_46226

-- Define a predicate to check if a natural number is even
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Define a predicate to check if a natural number is odd
def is_odd (n : ℕ) : Prop := n % 2 = 1

-- Problem statement in Lean
theorem correct_negation_of_exactly_one_even (a b c : ℕ) :
  ¬ ( (is_even a ∧ is_odd b ∧ is_odd c) ∨ 
      (is_odd a ∧ is_even b ∧ is_odd c) ∨ 
      (is_odd a ∧ is_odd b ∧ is_even c) ) ↔ 
  ( (is_odd a ∧ is_odd b ∧ is_odd c) ∨ 
    (is_even a ∧ is_even b ∧ is_even c) ) :=
by 
  sorry

end correct_negation_of_exactly_one_even_l46_46226


namespace weeks_to_work_l46_46938

def iPhone_cost : ℕ := 800
def trade_in_value : ℕ := 240
def weekly_earnings : ℕ := 80

theorem weeks_to_work (iPhone_cost trade_in_value weekly_earnings : ℕ) :
  (iPhone_cost - trade_in_value) / weekly_earnings = 7 :=
by
  sorry

end weeks_to_work_l46_46938


namespace geometric_series_sum_l46_46797

theorem geometric_series_sum :
  (1 / 5 - 1 / 25 + 1 / 125 - 1 / 625 + 1 / 3125) = 521 / 3125 :=
by
  sorry

end geometric_series_sum_l46_46797


namespace reciprocal_of_neg_one_div_2023_l46_46036

theorem reciprocal_of_neg_one_div_2023 : 1 / (-1 / (2023 : ℤ)) = -2023 := sorry

end reciprocal_of_neg_one_div_2023_l46_46036


namespace problem1_inner_problem2_inner_l46_46954

-- Problem 1
theorem problem1_inner {m n : ℤ} (hm : |m| = 5) (hn : |n| = 4) (opposite_signs : m * n < 0) :
  m^2 - m * n + n = 41 ∨ m^2 - m * n + n = 49 :=
sorry

-- Problem 2
theorem problem2_inner {a b c d x : ℝ} (opposite_ab : a + b = 0) (reciprocals_cd : c * d = 1) (hx : |x| = 5) (hx_pos : x > 0) :
  3 * (a + b) - 2 * (c * d) + x = 3 :=
sorry

end problem1_inner_problem2_inner_l46_46954


namespace circles_through_two_points_in_4x4_grid_l46_46534

noncomputable def number_of_circles (n : ℕ) : ℕ :=
  if n = 4 then
    52
  else
    sorry

theorem circles_through_two_points_in_4x4_grid :
  number_of_circles 4 = 52 :=
by
  exact rfl  -- Reflexivity of equality shows the predefined value of 52

end circles_through_two_points_in_4x4_grid_l46_46534


namespace leap_day_2040_is_tuesday_l46_46489

-- Define the given condition that 29th February 2012 is Wednesday
def feb_29_2012_is_wednesday : Prop := sorry

-- Define the calculation of the day of the week for February 29, 2040
def day_of_feb_29_2040 (initial_day : Nat) : Nat := (10228 % 7 + initial_day) % 7

-- Define the proof statement
theorem leap_day_2040_is_tuesday : feb_29_2012_is_wednesday →
  (day_of_feb_29_2040 3 = 2) := -- Here, 3 represents Wednesday and 2 represents Tuesday
sorry

end leap_day_2040_is_tuesday_l46_46489


namespace ellipse_condition_necessary_but_not_sufficient_l46_46282

-- Define the conditions and proof statement in Lean 4
theorem ellipse_condition (m : ℝ) (h₁ : 2 < m) (h₂ : m < 6) : 
  (6 - m ≠ m - 2) -> 
  (∃ x y : ℝ, (x^2) / (m - 2) + (y^2) / (6 - m)= 1) :=
by
  sorry

theorem necessary_but_not_sufficient : (2 < m ∧ m < 6) ↔ (2 < m ∧ m < 6 ∧ m ≠ 4) :=
by
  sorry

end ellipse_condition_necessary_but_not_sufficient_l46_46282


namespace calculate_expression_l46_46784

theorem calculate_expression :
  ((650^2 - 350^2) * 3 = 900000) := by
  sorry

end calculate_expression_l46_46784


namespace number_multiplied_by_3_l46_46875

variable (A B C D E : ℝ) -- Declare the five numbers

theorem number_multiplied_by_3 (h1 : (A + B + C + D + E) / 5 = 6.8) 
    (h2 : ∃ X : ℝ, (A + B + C + D + E + 2 * X) / 5 = 9.2) : 
    ∃ X : ℝ, X = 6 := 
  sorry

end number_multiplied_by_3_l46_46875


namespace sum_of_squares_of_coefficients_l46_46474

theorem sum_of_squares_of_coefficients :
  ∃ (a b c d e f : ℤ), (∀ x : ℤ, 8 * x ^ 3 + 64 = (a * x ^ 2 + b * x + c) * (d * x ^ 2 + e * x + f)) ∧ 
  (a ^ 2 + b ^ 2 + c ^ 2 + d ^ 2 + e ^ 2 + f ^ 2 = 356) := 
by
  sorry

end sum_of_squares_of_coefficients_l46_46474


namespace perfect_square_trinomial_l46_46840

theorem perfect_square_trinomial (a b m : ℝ) :
  (∃ x : ℝ, a^2 + mab + b^2 = (x + b)^2 ∨ a^2 + mab + b^2 = (x - b)^2) ↔ (m = 2 ∨ m = -2) :=
by
  sorry

end perfect_square_trinomial_l46_46840


namespace max_consecutive_sum_l46_46552

theorem max_consecutive_sum (N a : ℕ) (h : N * (2 * a + N - 1) = 240) : N ≤ 15 :=
by
  -- proof goes here
  sorry

end max_consecutive_sum_l46_46552


namespace min_value_f_l46_46951

noncomputable def f (x : ℝ) := 2 * (Real.sin x)^3 + (Real.cos x)^2

theorem min_value_f : ∃ x, 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ f x = 26 / 27 :=
by
  sorry

end min_value_f_l46_46951


namespace cost_to_buy_450_candies_l46_46779

-- Define a structure representing the problem conditions
structure CandyStore where
  candies_per_box : Nat
  regular_price : Nat
  discounted_price : Nat
  discount_threshold : Nat

-- Define parameters for this specific problem
def store : CandyStore :=
  { candies_per_box := 15,
    regular_price := 5,
    discounted_price := 4,
    discount_threshold := 10 }

-- Define the cost function with the given conditions
def cost (store : CandyStore) (candies : Nat) : Nat :=
  let boxes := candies / store.candies_per_box
  if boxes >= store.discount_threshold then
    boxes * store.discounted_price
  else
    boxes * store.regular_price

-- State the theorem we want to prove
theorem cost_to_buy_450_candies (store : CandyStore) (candies := 450) :
  store.candies_per_box = 15 →
  store.discounted_price = 4 →
  store.discount_threshold = 10 →
  cost store candies = 120 := by
  sorry

end cost_to_buy_450_candies_l46_46779


namespace pam_age_l46_46149

-- Given conditions:
-- 1. Pam is currently twice as young as Rena.
-- 2. In 10 years, Rena will be 5 years older than Pam.

variable (Pam Rena : ℕ)

theorem pam_age
  (h1 : 2 * Pam = Rena)
  (h2 : Rena + 10 = Pam + 15)
  : Pam = 5 := 
sorry

end pam_age_l46_46149


namespace smallest_N_for_triangle_sides_l46_46124

theorem smallest_N_for_triangle_sides (a b c : ℝ) (h_triangle : a + b > c) (h_a_ne_b : a ≠ b) : (a^2 + b^2) / c^2 < 1 := 
sorry

end smallest_N_for_triangle_sides_l46_46124


namespace sum_of_consecutive_page_numbers_l46_46392

def consecutive_page_numbers_product_and_sum (n m : ℤ) :=
  n * m = 20412

theorem sum_of_consecutive_page_numbers (n : ℤ) (h1 : consecutive_page_numbers_product_and_sum n (n + 1)) : n + (n + 1) = 285 :=
by
  sorry

end sum_of_consecutive_page_numbers_l46_46392


namespace expand_product_l46_46948

-- Definitions of the polynomial functions
def f (x : ℝ) : ℝ := x + 3
def g (x : ℝ) : ℝ := x^2 + x + 1

-- Statement of the theorem
theorem expand_product : ∀ x : ℝ, (f x) * (g x) = x^3 + 4*x^2 + 4*x + 3 :=
by
  -- Proof goes here, but is omitted for the statement only
  sorry

end expand_product_l46_46948


namespace _l46_46782

def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

example : triangle_inequality 1 1 1 := 
by {
  -- Prove using the triangle inequality theorem that the sides form a triangle.
  -- This part is left as an exercise to the reader.
  sorry
}

end _l46_46782


namespace MK_eq_ML_l46_46173

open EuclideanGeometry

variables {A B C C0 X K L M : Point}

noncomputable def is_right_angle (a b c : Point) : Prop :=
  ∠b c a = π / 2

noncomputable def is_tangent_circle (o : Point) (r : ℝ) (p : Point) : Prop :=
  dist o p = r

noncomputable def reflect (a b : Point) : Point :=
  (2 • b - a)

-- Given conditions
axiom angle_BCA_90 : is_right_angle B C A
axiom C0_foot : foot_of_perpendicular C (line_through A B) = C0
axiom X_within_C0C : between X C C0
axiom BK_eq_BC : dist B K = dist B C
axiom AL_eq_AC : dist A L = dist A C
axiom M_intersection : ∃ t : ℝ, M = t • (AL) + (1 - t) • (BK)

-- The theorem to prove
theorem MK_eq_ML : dist M K = dist M L :=
by
  sorry

end MK_eq_ML_l46_46173


namespace total_pies_l46_46850

theorem total_pies {team1 team2 team3 total_pies : ℕ} 
  (h1 : team1 = 235) 
  (h2 : team2 = 275) 
  (h3 : team3 = 240) 
  (h4 : total_pies = team1 + team2 + team3) : 
  total_pies = 750 := by 
  sorry

end total_pies_l46_46850


namespace necklaces_sold_correct_l46_46014

-- Define the given constants and conditions
def necklace_price : ℕ := 25
def bracelet_price : ℕ := 15
def earring_price : ℕ := 10
def ensemble_price : ℕ := 45
def bracelets_sold : ℕ := 10
def earrings_sold : ℕ := 20
def ensembles_sold : ℕ := 2
def total_revenue : ℕ := 565

-- Define the equation to calculate the total revenue
def total_revenue_calculation (N : ℕ) : ℕ :=
  (necklace_price * N) + (bracelet_price * bracelets_sold) + (earring_price * earrings_sold) + (ensemble_price * ensembles_sold)

-- Define the proof problem
theorem necklaces_sold_correct : 
  ∃ N : ℕ, total_revenue_calculation N = total_revenue ∧ N = 5 := by
  sorry

end necklaces_sold_correct_l46_46014


namespace inradius_of_triangle_area_twice_perimeter_l46_46992

theorem inradius_of_triangle_area_twice_perimeter (A p r s : ℝ) (hA : A = 2 * p) (hs : p = 2 * s)
  (hA_formula : A = r * s) : r = 4 :=
by
  sorry

end inradius_of_triangle_area_twice_perimeter_l46_46992


namespace number_of_bananas_l46_46913

-- Define costs as constants
def cost_per_banana := 1
def cost_per_apple := 2
def cost_per_twelve_strawberries := 4
def cost_per_avocado := 3
def cost_per_half_bunch_grapes := 2
def total_cost := 28

-- Define quantities as constants
def number_of_apples := 3
def number_of_strawberries := 24
def number_of_avocados := 2
def number_of_half_bunches_grapes := 2

-- Define calculated costs
def cost_of_apples := number_of_apples * cost_per_apple
def cost_of_strawberries := (number_of_strawberries / 12) * cost_per_twelve_strawberries
def cost_of_avocados := number_of_avocados * cost_per_avocado
def cost_of_grapes := number_of_half_bunches_grapes * cost_per_half_bunch_grapes

-- Define total cost of other fruits
def total_cost_of_other_fruits := cost_of_apples + cost_of_strawberries + cost_of_avocados + cost_of_grapes

-- Define the remaining cost for bananas
def remaining_cost := total_cost - total_cost_of_other_fruits

-- Prove the number of bananas
theorem number_of_bananas : remaining_cost / cost_per_banana = 4 :=
by
  -- This is a placeholder to indicate a non-implemented proof
  sorry

end number_of_bananas_l46_46913


namespace first_discount_percentage_l46_46393

theorem first_discount_percentage (original_price final_price : ℝ) (additional_discount : ℝ) (x : ℝ) 
  (h1 : original_price = 400) 
  (h2 : additional_discount = 0.05) 
  (h3 : final_price = 342) 
  (hx : (original_price * (100 - x) / 100) * (1 - additional_discount) = final_price) :
  x = 10 := 
sorry

end first_discount_percentage_l46_46393


namespace license_plate_count_l46_46627

-- Define the conditions as constants
def even_digit_count : Nat := 5
def consonant_count : Nat := 20
def vowel_count : Nat := 6

-- Define the problem as a theorem to prove
theorem license_plate_count : even_digit_count * consonant_count * vowel_count * consonant_count = 12000 := 
by
  -- The proof is not required, so we leave it as sorry
  sorry

end license_plate_count_l46_46627


namespace ambulance_ride_cost_is_correct_l46_46250

-- Define all the constants and conditions
def daily_bed_cost : ℝ := 900
def bed_days : ℕ := 3
def specialist_rate_per_hour : ℝ := 250
def specialist_minutes_per_day : ℕ := 15
def specialists_count : ℕ := 2
def total_bill : ℝ := 4625

noncomputable def ambulance_cost : ℝ :=
  total_bill - ((daily_bed_cost * bed_days) + (specialist_rate_per_hour * (specialist_minutes_per_day / 60) * specialists_count))

-- The proof statement
theorem ambulance_ride_cost_is_correct : ambulance_cost = 1675 := by
  sorry

end ambulance_ride_cost_is_correct_l46_46250


namespace solve_eq_integers_l46_46374

theorem solve_eq_integers (x y : ℤ) : 
    x^2 - x * y - 6 * y^2 + 2 * x + 19 * y = 18 ↔ (x = 2 ∧ y = 2) ∨ (x = -2 ∧ y = 2) := by
    sorry

end solve_eq_integers_l46_46374


namespace sum_of_digits_of_N_eq_14_l46_46440

theorem sum_of_digits_of_N_eq_14 :
  ∃ N : ℕ, (N * (N + 1)) / 2 = 3003 ∧ (N % 10 + N / 10 % 10 = 14) :=
by
  sorry

end sum_of_digits_of_N_eq_14_l46_46440


namespace train_pass_station_time_l46_46718

-- Define the lengths of the train and station
def length_train : ℕ := 250
def length_station : ℕ := 200

-- Define the speed of the train in km/hour
def speed_kmh : ℕ := 36

-- Convert the speed to meters per second
def speed_mps : ℕ := speed_kmh * 1000 / 3600

-- Calculate the total distance the train needs to cover
def total_distance : ℕ := length_train + length_station

-- Define the expected time to pass the station
def expected_time : ℕ := 45

-- State the theorem that needs to be proven
theorem train_pass_station_time :
  total_distance / speed_mps = expected_time := by
  sorry

end train_pass_station_time_l46_46718


namespace volume_of_cuboid_l46_46414

theorem volume_of_cuboid (l w h : ℝ) (hl_pos : 0 < l) (hw_pos : 0 < w) (hh_pos : 0 < h) 
  (h1 : l * w = 120) (h2 : w * h = 72) (h3 : h * l = 60) : l * w * h = 4320 :=
by
  sorry

end volume_of_cuboid_l46_46414


namespace sin_alpha_minus_beta_l46_46961

variables (α β : ℝ)

theorem sin_alpha_minus_beta (h1 : (Real.tan α / Real.tan β) = 7 / 13) 
    (h2 : Real.sin (α + β) = 2 / 3) :
    Real.sin (α - β) = -1 / 5 := 
sorry

end sin_alpha_minus_beta_l46_46961


namespace probability_at_least_6_heads_in_8_flips_l46_46770

open scoped BigOperators

def binom (n k : ℕ) : ℕ := nat.choose n k

def total_outcomes (n : ℕ) := 2^n

def successful_outcomes (n k : ℕ) :=
  (finset.range (n + 1)).filter (λ x, x ≥ k).sum (λ x, binom n x)

def probability (n k : ℕ) :=
  (successful_outcomes n k) / (total_outcomes n : ℚ)

theorem probability_at_least_6_heads_in_8_flips :
  probability 8 6 = 37 / 256 := sorry

end probability_at_least_6_heads_in_8_flips_l46_46770


namespace flight_time_NY_to_CT_l46_46786

def travelTime (start_time_NY : ℕ) (end_time_CT : ℕ) (layover_Johannesburg : ℕ) : ℕ :=
  end_time_CT - start_time_NY + layover_Johannesburg

theorem flight_time_NY_to_CT :
  let start_time_NY := 0 -- 12:00 a.m. Tuesday as 0 hours from midnight in ET
  let end_time_CT := 10  -- 10:00 a.m. Tuesday as 10 hours from midnight in ET
  let layover_Johannesburg := 4
  travelTime start_time_NY end_time_CT layover_Johannesburg = 10 :=
by
  sorry

end flight_time_NY_to_CT_l46_46786


namespace longest_side_of_enclosure_l46_46408

theorem longest_side_of_enclosure
  (l w : ℝ)
  (h1 : 2 * l + 2 * w = 180)
  (h2 : l * w = 1440) :
  l = 72 ∨ w = 72 :=
by {
  sorry
}

end longest_side_of_enclosure_l46_46408


namespace probability_of_at_least_six_heads_is_correct_l46_46767

-- Definitions for the given problem
def binomial_coefficient (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

def total_possible_outcomes : ℕ :=
  2^8

def favorable_outcomes : ℕ :=
  binomial_coefficient 8 6 + binomial_coefficient 8 7 + binomial_coefficient 8 8

def probability_of_at_least_6_heads : ℚ :=
  favorable_outcomes / total_possible_outcomes

-- The proof statement
theorem probability_of_at_least_six_heads_is_correct :
  probability_of_at_least_6_heads = 37 / 256 :=
by sorry

end probability_of_at_least_six_heads_is_correct_l46_46767


namespace quadratic_inequality_solution_l46_46206

theorem quadratic_inequality_solution (m : ℝ) :
    (∃ x : ℝ, x^2 - m * x + 1 ≤ 0) ↔ m ≥ 2 ∨ m ≤ -2 := by
  sorry

end quadratic_inequality_solution_l46_46206


namespace pow_div_l46_46448

theorem pow_div (a : ℝ) : (-a) ^ 6 / a ^ 3 = a ^ 3 := by
  sorry

end pow_div_l46_46448


namespace total_letters_correct_l46_46514

-- Define the conditions
def letters_January := 6
def letters_February := 9
def letters_March := 3 * letters_January

-- Definition of the total number of letters sent
def total_letters := letters_January + letters_February + letters_March

-- The statement we need to prove in Lean
theorem total_letters_correct : total_letters = 33 := 
by
  sorry

end total_letters_correct_l46_46514


namespace part1_part2_part3_l46_46463

variable {x y : ℚ}

def star (x y : ℚ) : ℚ := x * y + 1

theorem part1 : star 2 4 = 9 := by
  sorry

theorem part2 : star (star 1 4) (-2) = -9 := by
  sorry

theorem part3 (a b c : ℚ) : star a (b + c) + 1 = star a b + star a c := by
  sorry

end part1_part2_part3_l46_46463


namespace first_player_guaranteed_win_l46_46889

def is_power_of_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2 ^ k

theorem first_player_guaranteed_win (n : ℕ) (h : n > 1) : 
  ¬ is_power_of_two n ↔ ∃ m : ℕ, 1 ≤ m ∧ m < n ∧ (∀ k : ℕ, m ≤ k + 1 → ∀ t, t ≤ m → ∃ r, r = k + 1 ∧ r <= m) → 
                                (∃ l : ℕ, (l = 1) → true) :=
sorry

end first_player_guaranteed_win_l46_46889


namespace arrange_f_values_l46_46455

noncomputable def f : ℝ → ℝ := sorry -- Assuming the actual definition is not necessary

-- The function f is even
def even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

-- The function f is strictly decreasing on (-∞, 0)
def strictly_decreasing_on_negative (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1 < 0 ∧ x2 < 0 ∧ x1 ≠ x2 → (x1 < x2 ↔ f x1 > f x2)

theorem arrange_f_values (f : ℝ → ℝ)
  (h_even : even_function f)
  (h_decreasing : strictly_decreasing_on_negative f) :
  f 3 < f (-2) ∧ f (-2) < f 1 :=
by
  -- The actual proof would go here.
  sorry

end arrange_f_values_l46_46455


namespace trigonometric_identity_l46_46814

theorem trigonometric_identity (α : ℝ) (h : Real.sin α = 2 * Real.cos α) :
  Real.sin (π / 2 + 2 * α) = -3 / 5 :=
by
  sorry

end trigonometric_identity_l46_46814


namespace inequality_least_n_l46_46138

theorem inequality_least_n (n : ℕ) (h : (1 : ℝ) / n - (1 : ℝ) / (n + 2) < 1 / 15) : n = 5 :=
sorry

end inequality_least_n_l46_46138


namespace convert_base8_to_base7_l46_46452

def base8_to_base10 (n : ℕ) : ℕ :=
  5 * 8^2 + 3 * 8^1 + 1 * 8^0

def base10_to_base7 (n : ℕ) : ℕ :=
  1002  -- Directly providing the result from conditions given.

theorem convert_base8_to_base7 :
  base10_to_base7 (base8_to_base10 531) = 1002 := by
  sorry

end convert_base8_to_base7_l46_46452


namespace lowest_fraction_of_job_in_one_hour_l46_46563

-- Define the rates at which each person can work
def rate_A : ℚ := 1/3
def rate_B : ℚ := 1/4
def rate_C : ℚ := 1/6

-- Define the combined rates for each pair of people
def combined_rate_AB : ℚ := rate_A + rate_B
def combined_rate_AC : ℚ := rate_A + rate_C
def combined_rate_BC : ℚ := rate_B + rate_C

-- The Lean 4 statement to prove
theorem lowest_fraction_of_job_in_one_hour : min combined_rate_AB (min combined_rate_AC combined_rate_BC) = 5/12 :=
by 
  -- Here we state that the minimum combined rate is 5/12
  sorry

end lowest_fraction_of_job_in_one_hour_l46_46563


namespace find_n_l46_46852

theorem find_n (x : ℝ) (n : ℝ) (G : ℝ) (hG : G = (7*x^2 + 21*x + 5*n) / 7) :
  (∃ c d : ℝ, c^2 * x^2 + 2*c*d*x + d^2 = G) ↔ n = 63 / 20 :=
by
  sorry

end find_n_l46_46852


namespace find_x_l46_46886

theorem find_x (x : ℝ) : (x * 16) / 100 = 0.051871999999999995 → x = 0.3242 := by
  intro h
  sorry

end find_x_l46_46886


namespace probability_at_least_6_heads_in_8_flips_l46_46744

def binomial (n k : ℕ) : ℕ :=
  if h : k ≤ n then (Nat.choose n k) else 0

theorem probability_at_least_6_heads_in_8_flips :
  let total_outcomes := 2^8
  let successful_outcomes := binomial 8 6 + binomial 8 7 + binomial 8 8
  let probability := (successful_outcomes : ℚ) / total_outcomes
  probability = 37 / 256 :=
by
  sorry

end probability_at_least_6_heads_in_8_flips_l46_46744


namespace factorization_solution_1_factorization_solution_2_factorization_solution_3_l46_46258

noncomputable def factorization_problem_1 (m : ℝ) : Prop :=
  -3 * m^3 + 12 * m = -3 * m * (m + 2) * (m - 2)

noncomputable def factorization_problem_2 (x y : ℝ) : Prop :=
  2 * x^2 * y - 8 * x * y + 8 * y = 2 * y * (x - 2)^2

noncomputable def factorization_problem_3 (a : ℝ) : Prop :=
  a^4 + 3 * a^2 - 4 = (a^2 + 4) * (a + 1) * (a - 1)

-- Lean statements for the proofs
theorem factorization_solution_1 (m : ℝ) : factorization_problem_1 m :=
  by sorry

theorem factorization_solution_2 (x y : ℝ) : factorization_problem_2 x y :=
  by sorry

theorem factorization_solution_3 (a : ℝ) : factorization_problem_3 a :=
  by sorry

end factorization_solution_1_factorization_solution_2_factorization_solution_3_l46_46258


namespace solve_equation_l46_46394

theorem solve_equation (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 1) :
  (3 / (x - 2) = 2 / (x - 1)) ↔ (x = -1) :=
sorry

end solve_equation_l46_46394


namespace factorize_m4_minus_5m_plus_4_factorize_x3_plus_2x2_plus_4x_plus_3_factorize_x5_minus_1_l46_46729

-- Statement for question 1
theorem factorize_m4_minus_5m_plus_4 (m : ℤ) : 
  (m ^ 4 - 5 * m + 4) = (m ^ 4 - 5 * m + 4) := sorry

-- Statement for question 2
theorem factorize_x3_plus_2x2_plus_4x_plus_3 (x : ℝ) :
  (x ^ 3 + 2 * x ^ 2 + 4 * x + 3) = (x + 1) * (x ^ 2 + x + 3) := sorry

-- Statement for question 3
theorem factorize_x5_minus_1 (x : ℝ) :
  (x ^ 5 - 1) = (x - 1) * (x ^ 4 + x ^ 3 + x ^ 2 + x + 1) := sorry

end factorize_m4_minus_5m_plus_4_factorize_x3_plus_2x2_plus_4x_plus_3_factorize_x5_minus_1_l46_46729


namespace count_four_digit_multiples_of_5_l46_46316

theorem count_four_digit_multiples_of_5 : 
  let first_4_digit := 1000
  let last_4_digit := 9999
  let first_multiple_of_5 := 1000
  let last_multiple_of_5 := 9995
  let total_multiples_of_5 := (1999 - 200 + 1)
  first_multiple_of_5 % 5 = 0 ∧ last_multiple_of_5 % 5 = 0 ∧ first_4_digit ≤ first_multiple_of_5 ∧ last_multiple_of_5 ≤ last_4_digit
  → total_multiples_of_5 = 1800 :=
by
  sorry

end count_four_digit_multiples_of_5_l46_46316


namespace change_in_total_berries_l46_46898

-- Define the initial conditions
def blue_box_berries : ℕ := 35
def increase_diff : ℕ := 100

-- Define the number of strawberries in red boxes
def red_box_berries : ℕ := 100

-- Formulate the change in total number of berries
theorem change_in_total_berries :
  (red_box_berries - blue_box_berries) = 65 :=
by
  have h1 : red_box_berries = increase_diff := rfl
  have h2 : blue_box_berries = 35 := rfl
  rw [h1, h2]
  exact rfl

end change_in_total_berries_l46_46898


namespace book_original_price_l46_46537

noncomputable def original_price : ℝ := 420 / 1.40

theorem book_original_price (new_price : ℝ) (percentage_increase : ℝ) : 
  new_price = 420 → percentage_increase = 0.40 → original_price = 300 :=
by
  intros h1 h2
  exact sorry

end book_original_price_l46_46537


namespace totalMilkConsumption_l46_46339

-- Conditions
def regularMilk (week: ℕ) : ℝ := 0.5
def soyMilk (week: ℕ) : ℝ := 0.1

-- Theorem statement
theorem totalMilkConsumption : regularMilk 1 + soyMilk 1 = 0.6 := 
by 
  sorry

end totalMilkConsumption_l46_46339


namespace proposition_C_correct_l46_46271

theorem proposition_C_correct (a b c : ℝ) (h : a * c ^ 2 > b * c ^ 2) : a > b :=
sorry

end proposition_C_correct_l46_46271


namespace consecutive_integers_sum_l46_46650

theorem consecutive_integers_sum (a b : ℤ) (h1 : b = a + 1) (h2 : a < Real.sqrt 3) (h3 : Real.sqrt 3 < b) : a + b = 3 :=
sorry

end consecutive_integers_sum_l46_46650


namespace binom_12_9_is_220_l46_46787

def choose (n k : ℕ) : ℕ := n.choose k

theorem binom_12_9_is_220 :
  choose 12 9 = 220 :=
by {
  -- Proof is omitted
  sorry
}

end binom_12_9_is_220_l46_46787


namespace find_x_squared_plus_y_squared_find_xy_l46_46813

variable {x y : ℝ}

theorem find_x_squared_plus_y_squared (h1 : (x - y)^2 = 4) (h2 : (x + y)^2 = 64) : x^2 + y^2 = 34 :=
sorry

theorem find_xy (h1 : (x - y)^2 = 4) (h2 : (x + y)^2 = 64) : x * y = 15 :=
sorry

end find_x_squared_plus_y_squared_find_xy_l46_46813


namespace abs_a_k_le_fractional_l46_46507

variable (a : ℕ → ℝ) (n : ℕ)

-- Condition 1: a_0 = a_(n+1) = 0
axiom a_0 : a 0 = 0
axiom a_n1 : a (n + 1) = 0

-- Condition 2: |a_{k-1} - 2a_k + a_{k+1}| ≤ 1 for k = 1, 2, ..., n
axiom abs_diff_ineq (k : ℕ) (h : 1 ≤ k ∧ k ≤ n) : 
  |a (k - 1) - 2 * a k + a (k + 1)| ≤ 1

-- Theorem statement
theorem abs_a_k_le_fractional (k : ℕ) (h : 0 ≤ k ∧ k ≤ n + 1) : 
  |a k| ≤ k * (n + 1 - k) / 2 := sorry

end abs_a_k_le_fractional_l46_46507


namespace probability_target_A_destroyed_probability_exactly_one_target_destroyed_l46_46894

-- Definition of probabilities
def prob_A_hits_target_A := 1 / 2
def prob_A_hits_target_B := 1 / 2
def prob_B_hits_target_A := 1 / 3
def prob_B_hits_target_B := 2 / 5

-- The event of target A being destroyed
def prob_target_A_destroyed := prob_A_hits_target_A * prob_B_hits_target_A

-- The event of target B being destroyed
def prob_target_B_destroyed := prob_A_hits_target_B * prob_B_hits_target_B

-- Complementary events
def prob_target_A_not_destroyed := 1 - prob_target_A_destroyed
def prob_target_B_not_destroyed := 1 - prob_target_B_destroyed

-- Exactly one target being destroyed
def prob_exactly_one_target_destroyed := 
  (prob_target_A_destroyed * prob_target_B_not_destroyed) +
  (prob_target_B_destroyed * prob_target_A_not_destroyed)

theorem probability_target_A_destroyed : prob_target_A_destroyed = 1 / 6 := by
  -- Proof needed here
  sorry

theorem probability_exactly_one_target_destroyed : prob_exactly_one_target_destroyed = 3 / 10 := by
  -- Proof needed here
  sorry

end probability_target_A_destroyed_probability_exactly_one_target_destroyed_l46_46894


namespace boat_travel_distance_per_day_l46_46178

-- Definitions from conditions
def men : ℕ := 25
def water_daily_per_man : ℚ := 1/2
def travel_distance : ℕ := 4000
def total_water : ℕ := 250

-- Main theorem
theorem boat_travel_distance_per_day : 
  ∀ (men : ℕ) (water_daily_per_man : ℚ) (travel_distance : ℕ) (total_water : ℕ), 
  men = 25 ∧ water_daily_per_man = 1/2 ∧ travel_distance = 4000 ∧ total_water = 250 ->
  travel_distance / (total_water / (men * water_daily_per_man)) = 200 :=
by
  sorry

end boat_travel_distance_per_day_l46_46178


namespace Nick_total_money_l46_46176

variable (nickels : Nat) (dimes : Nat) (quarters : Nat)
variable (value_nickel : Nat := 5) (value_dime : Nat := 10) (value_quarter : Nat := 25)

def total_value (nickels dimes quarters : Nat) : Nat :=
  nickels * value_nickel + dimes * value_dime + quarters * value_quarter

theorem Nick_total_money :
  total_value 6 2 1 = 75 := by
  sorry

end Nick_total_money_l46_46176


namespace handshakes_max_number_of_men_l46_46545

theorem handshakes_max_number_of_men (n : ℕ) (h: n * (n - 1) / 2 = 190) : n = 20 :=
sorry

end handshakes_max_number_of_men_l46_46545


namespace pier_to_village_trip_l46_46935

theorem pier_to_village_trip :
  ∃ (x t : ℝ), 
  (x / 10 + x / 8 = t + 1 / 60) ∧
  (5 * t / 2 + 4 * t / 2 = x) ∧
  (x = 6) ∧
  (t = 4 / 3) :=
by
  sorry

end pier_to_village_trip_l46_46935


namespace lunch_break_duration_l46_46017

theorem lunch_break_duration
  (p h L : ℝ)
  (monday_eq : (9 - L) * (p + h) = 0.4)
  (tuesday_eq : (8 - L) * h = 0.33)
  (wednesday_eq : (12 - L) * p = 0.27) :
  L = 7.0 ∨ L * 60 = 420 :=
by
  sorry

end lunch_break_duration_l46_46017


namespace rectangle_sides_l46_46610

def side_length_square : ℝ := 18
def num_rectangles : ℕ := 5

variable (a b : ℝ)
variables (h1 : 2 * (a + b) = side_length_square) (h2 : 3 * a = side_length_square)

theorem rectangle_sides : a = 6 ∧ b = 3 :=
by {
  sorry
}

end rectangle_sides_l46_46610


namespace smallest_N_exists_l46_46482

def find_smallest_N (N : ℕ) : Prop :=
  ∃ (c1 c2 c3 c4 c5 c6 : ℕ),
  (N ≠ 0) ∧ 
  (c1 = 6 * c2 - 1) ∧ 
  (N + c2 = 6 * c3 - 2) ∧ 
  (2 * N + c3 = 6 * c4 - 3) ∧ 
  (3 * N + c4 = 6 * c5 - 4) ∧ 
  (4 * N + c5 = 6 * c6 - 5) ∧ 
  (5 * N + c6 = 6 * c1)

theorem smallest_N_exists : ∃ (N : ℕ), find_smallest_N N :=
sorry

end smallest_N_exists_l46_46482


namespace initial_investment_B_l46_46422
-- Import necessary Lean library

-- Define the necessary conditions and theorems
theorem initial_investment_B (x : ℝ) (profit_A : ℝ) (profit_total : ℝ)
  (initial_A : ℝ) (initial_A_after_8_months : ℝ) (profit_B : ℝ) 
  (initial_A_months : ℕ) (initial_A_after_8_months_months : ℕ) 
  (initial_B_months : ℕ) (initial_B_after_8_months_months : ℕ) : 
  initial_A = 3000 ∧ initial_A_after_8_months = 2000 ∧
  profit_A = 240 ∧ profit_total = 630 ∧ 
  profit_B = profit_total - profit_A ∧
  (initial_A * initial_A_months + initial_A_after_8_months * initial_A_after_8_months_months) /
  ((initial_B_months * x + initial_B_after_8_months_months * (x + 1000))) = 
  profit_A / profit_B →
  x = 4000 :=
by
  sorry

end initial_investment_B_l46_46422


namespace solve_equation_in_natural_numbers_l46_46377

-- Define the main theorem
theorem solve_equation_in_natural_numbers :
  (∃ (x y z : ℕ), 2^x + 5^y + 63 = z! ∧ ((x = 5 ∧ y = 2 ∧ z = 5) ∨ (x = 4 ∧ y = 4 ∧ z = 6))) :=
sorry

end solve_equation_in_natural_numbers_l46_46377


namespace not_or_false_implies_or_true_l46_46655

variable (p q : Prop)

theorem not_or_false_implies_or_true (h : ¬(p ∨ q) = False) : p ∨ q :=
by
  sorry

end not_or_false_implies_or_true_l46_46655


namespace train_speed_is_correct_l46_46092

-- Definitions based on the conditions
def length_of_train : ℝ := 120       -- Train is 120 meters long
def time_to_cross : ℝ := 16          -- The train takes 16 seconds to cross the post

-- Conversion constants
def seconds_to_hours : ℝ := 3600
def meters_to_kilometers : ℝ := 1000

-- The speed of the train in km/h
noncomputable def speed_of_train (length : ℝ) (time : ℝ) : ℝ :=
  (length / time) * (seconds_to_hours / meters_to_kilometers)

-- Theorem: The speed of the train is 27 km/h
theorem train_speed_is_correct : speed_of_train length_of_train time_to_cross = 27 :=
by
  -- This is where the proof should be, but we leave it as sorry as instructed
  sorry

end train_speed_is_correct_l46_46092


namespace symmetry_condition_l46_46692

theorem symmetry_condition (p q r s t u : ℤ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) (ht : t ≠ 0) (hu : u ≠ 0)
  (yx_eq : ∀ x y, y = (p * x ^ 2 + q * x + r) / (s * x ^ 2 + t * x + u) ↔ x = (p * y ^ 2 + q * y + r) / (s * y ^ 2 + t * y + u)) :
  p = s ∧ q = t ∧ r = u :=
sorry

end symmetry_condition_l46_46692


namespace trig_expression_value_l46_46878

theorem trig_expression_value :
  (3 / (Real.sin (140 * Real.pi / 180))^2 - 1 / (Real.cos (140 * Real.pi / 180))^2) * (1 / (2 * Real.sin (10 * Real.pi / 180))) = 16 := 
by
  -- placeholder for proof
  sorry

end trig_expression_value_l46_46878


namespace range_of_m_l46_46156

def isDistinctRealRootsInInterval (a b x : ℝ) : Prop :=
  a * x^2 + b * x + 4 = 0 ∧ 0 < x ∧ x ≤ 3

theorem range_of_m (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ isDistinctRealRootsInInterval 1 (- (m + 1)) x ∧ isDistinctRealRootsInInterval 1 (- (m + 1)) y) ↔
  (3 < m ∧ m ≤ 10 / 3) :=
sorry

end range_of_m_l46_46156


namespace parabola_focus_at_centroid_l46_46614

theorem parabola_focus_at_centroid (A B C : ℝ × ℝ) (a : ℝ) 
  (hA : A = (-1, 2))
  (hB : B = (3, 4))
  (hC : C = (4, -6))
  (h_focus : (a/4, 0) = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)) :
  a = 8 :=
by
  sorry

end parabola_focus_at_centroid_l46_46614


namespace probability_at_least_6_heads_in_8_flips_l46_46745

def binomial (n k : ℕ) : ℕ :=
  if h : k ≤ n then (Nat.choose n k) else 0

theorem probability_at_least_6_heads_in_8_flips :
  let total_outcomes := 2^8
  let successful_outcomes := binomial 8 6 + binomial 8 7 + binomial 8 8
  let probability := (successful_outcomes : ℚ) / total_outcomes
  probability = 37 / 256 :=
by
  sorry

end probability_at_least_6_heads_in_8_flips_l46_46745


namespace inradius_of_triangle_area_twice_perimeter_l46_46991

theorem inradius_of_triangle_area_twice_perimeter (A p r s : ℝ) (hA : A = 2 * p) (hs : p = 2 * s)
  (hA_formula : A = r * s) : r = 4 :=
by
  sorry

end inradius_of_triangle_area_twice_perimeter_l46_46991


namespace simplify_expression_l46_46021

theorem simplify_expression : 4 * (12 / 9) * (36 / -45) = -12 / 5 :=
by
  sorry

end simplify_expression_l46_46021


namespace percentage_of_goals_by_two_players_l46_46710

-- Definitions from conditions
def total_goals_league := 300
def goals_per_player := 30
def number_of_players := 2

-- Mathematically equivalent proof problem
theorem percentage_of_goals_by_two_players :
  let combined_goals := number_of_players * goals_per_player
  let percentage := (combined_goals / total_goals_league : ℝ) * 100 
  percentage = 20 :=
by
  sorry

end percentage_of_goals_by_two_players_l46_46710


namespace prime_divides_expression_l46_46851

theorem prime_divides_expression (p : ℕ) (hp : p > 5 ∧ Prime p) : 
  ∃ n : ℕ, p ∣ (20^n + 15^n - 12^n) := 
  by
  use (p - 3)
  sorry

end prime_divides_expression_l46_46851


namespace speed_of_second_fragment_l46_46084

noncomputable def magnitude_speed_of_second_fragment 
  (u : ℝ) (t : ℝ) (g : ℝ) (v_x1 : ℝ) (v_y1 : ℝ := - (u - g * t)) 
  (v_x2 : ℝ := -v_x1) (v_y2 : ℝ := v_y1) : ℝ :=
Real.sqrt ((v_x2 ^ 2) + (v_y2 ^ 2))

theorem speed_of_second_fragment 
  (u : ℝ) (t : ℝ) (g : ℝ) (v_x1 : ℝ) 
  (h_u : u = 20) (h_t : t = 3) (h_g : g = 10) (h_vx1 : v_x1 = 48) :
  magnitude_speed_of_second_fragment u t g v_x1 = Real.sqrt 2404 :=
by
  -- Proof
  sorry

end speed_of_second_fragment_l46_46084


namespace consecutive_integers_sum_l46_46648

theorem consecutive_integers_sum (a b : ℤ) (h1 : b = a + 1) (h2 : a < Real.sqrt 3) (h3 : Real.sqrt 3 < b) : a + b = 3 :=
sorry

end consecutive_integers_sum_l46_46648


namespace topological_sort_possible_l46_46960
-- Import the necessary library

-- Definition of simple, directed, and acyclic graph (DAG)
structure SimpleDirectedAcyclicGraph (V : Type*) :=
  (E : V → V → Prop)
  (acyclic : ∀ v : V, ¬(E v v)) -- no loops
  (simple : ∀ (u v : V), (E u v) → ¬(E v u)) -- no bidirectional edges
  (directional : ∀ (u v w : V), E u v → E v w → E u w) -- directional transitivity

-- Existence of topological sort definition
def topological_sort_exists {V : Type*} (G : SimpleDirectedAcyclicGraph V) : Prop :=
  ∃ (numbering : V → ℕ), ∀ (u v : V), (G.E u v) → (numbering u > numbering v)

-- Theorem statement
theorem topological_sort_possible (V : Type*) (G : SimpleDirectedAcyclicGraph V) : topological_sort_exists G :=
  sorry

end topological_sort_possible_l46_46960


namespace solve_equation_l46_46378

theorem solve_equation (x y z : ℕ) (h1 : 2^x + 5^y + 63 = z!) (h2 : z ≥ 5) : 
  (x = 5 ∧ y = 2 ∧ z = 5) ∨ (x = 4 ∧ y = 4 ∧ z = 6) :=
sorry

end solve_equation_l46_46378


namespace apples_left_total_l46_46812

def Frank_apples : ℕ := 36
def Susan_apples : ℕ := 3 * Frank_apples
def Frank_sold : ℕ := Frank_apples / 3
def Susan_given : ℕ := Susan_apples / 2

theorem apples_left_total : Susan_apples - Susan_given + Frank_apples - Frank_sold = 78 := 
by
  have h1 : Susan_apples = 3 * Frank_apples := rfl
  have h2 : Frank_apples = 36 := rfl
  have h3 : Susan_given = Susan_apples / 2 := rfl
  have h4 : Frank_sold = Frank_apples / 3 := rfl
  -- since we know the values, we could calculate directly
  have h5 : Susan_apples = 108 := by rw [h1, h2]; norm_num
  have h6 : Susan_given = 54 := by rw [h5]; norm_num
  have h7 : Frank_sold = 12 := by rw [h2]; norm_num
  calc
    Susan_apples - Susan_given + Frank_apples - Frank_sold
        = 108 - 54 + 36 - 12 := by rw [h5, h6, h2, h7]
    ... = 78 := by norm_num

end apples_left_total_l46_46812


namespace sum_of_n_plus_k_l46_46533

theorem sum_of_n_plus_k (n k : ℕ) (h1 : 2 * (n - k) = 3 * (k + 1)) (h2 : 3 * (n - k - 1) = 4 * (k + 2)) : n + k = 47 := by
  sorry

end sum_of_n_plus_k_l46_46533


namespace bank_balance_after_two_years_l46_46919

-- Define the original amount deposited
def original_amount : ℝ := 5600

-- Define the interest rate
def interest_rate : ℝ := 0.07

-- Define the interest for each year based on the original amount
def interest_per_year : ℝ := original_amount * interest_rate

-- Define the total amount after two years
def total_amount_after_two_years : ℝ := original_amount + interest_per_year + interest_per_year

-- Define the target value
def target_value : ℝ := 6384

-- The theorem we aim to prove
theorem bank_balance_after_two_years : 
  total_amount_after_two_years = target_value := 
by
  -- Proof goes here
  sorry

end bank_balance_after_two_years_l46_46919


namespace fourth_person_height_l46_46724

theorem fourth_person_height (H : ℝ) 
  (h_avg : (H + (H + 2) + (H + 4) + (H + 10)) / 4 = 77) : 
  H + 10 = 83 :=
sorry

end fourth_person_height_l46_46724


namespace train_speed_l46_46925

noncomputable def train_length : ℝ := 150
noncomputable def bridge_length : ℝ := 250
noncomputable def crossing_time : ℝ := 28.79769618430526

noncomputable def speed_m_per_s : ℝ := (train_length + bridge_length) / crossing_time
noncomputable def speed_kmph : ℝ := speed_m_per_s * 3.6

theorem train_speed : speed_kmph = 50 := by
  sorry

end train_speed_l46_46925


namespace four_digit_number_sum_eq_4983_l46_46735

def reverse_number (n : ℕ) : ℕ :=
  let d1 := n / 1000
  let d2 := (n % 1000) / 100
  let d3 := (n % 100) / 10
  let d4 := n % 10
  1000 * d4 + 100 * d3 + 10 * d2 + d1

theorem four_digit_number_sum_eq_4983 (n : ℕ) :
  n + reverse_number n = 4983 ↔ n = 1992 ∨ n = 2991 :=
by sorry

end four_digit_number_sum_eq_4983_l46_46735


namespace longer_leg_of_minimum_30_60_90_triangle_is_9_l46_46116

-- Define the properties of a 30-60-90 triangle
def sideRatios := (1 : ℝ, Real.sqrt 3, 2 : ℝ)

noncomputable def longer_leg_of_smallest_triangle (hypotenuse_largest : ℝ) : ℝ :=
  let hypotenuse1  := hypotenuse_largest
  let shorter_leg1 := hypotenuse1 / 2
  let longer_leg1  := shorter_leg1 * Real.sqrt 3
  let hypotenuse2  := longer_leg1
  let shorter_leg2 := hypotenuse2 / 2
  let longer_leg2  := shorter_leg2 * Real.sqrt 3
  let hypotenuse3  := longer_leg2
  let shorter_leg3 := hypotenuse3 / 2
  let longer_leg3  := shorter_leg3 * Real.sqrt 3
  let hypotenuse4  := longer_leg3
  let shorter_leg4 := hypotenuse4 / 2
  let longer_leg4  := shorter_leg4 * Real.sqrt 3
  longer_leg4

theorem longer_leg_of_minimum_30_60_90_triangle_is_9 (hypotenuse_largest : ℝ) 
  (H : hypotenuse_largest = 16) : longer_leg_of_smallest_triangle hypotenuse_largest = 9 := by
  sorry

end longer_leg_of_minimum_30_60_90_triangle_is_9_l46_46116


namespace compute_f_l46_46736

theorem compute_f (f : ℕ → ℚ) (h1 : f 1 = 1 / 3)
  (h2 : ∀ n : ℕ, n ≥ 2 → f n = (2 * (n - 1) - 1) / (2 * (n - 1) + 3) * f (n - 1)) :
  ∀ n : ℕ, n ≥ 1 → f n = 1 / ((2 * n - 1) * (2 * n + 1)) :=
by
  sorry

end compute_f_l46_46736


namespace function_y_increases_when_x_gt_1_l46_46970

theorem function_y_increases_when_x_gt_1 :
  ∀ (x : ℝ), (x > 1 → 2*x^2 > 2*(x-1)^2) :=
by
  sorry

end function_y_increases_when_x_gt_1_l46_46970


namespace sums_same_remainder_exists_l46_46725

theorem sums_same_remainder_exists (n : ℕ) (h : n > 0) (a : Fin (2 * n) → Fin (2 * n)) (ha_permutation : Function.Bijective a) :
  ∃ (i j : Fin (2 * n)), i ≠ j ∧ ((a i + i) % (2 * n) = (a j + j) % (2 * n)) :=
by sorry

end sums_same_remainder_exists_l46_46725


namespace sale_in_third_month_l46_46575

theorem sale_in_third_month 
  (sale1 sale2 sale4 sale5 sale6 : ℕ) 
  (average_sales : ℕ)
  (h1 : sale1 = 5420)
  (h2 : sale2 = 5660)
  (h4 : sale4 = 6350)
  (h5 : sale5 = 6500)
  (h6 : sale6 = 6470)
  (h_avg : average_sales = 6100) : 
  ∃ sale3, sale1 + sale2 + sale3 + sale4 + sale5 + sale6 = average_sales * 6 ∧ sale3 = 6200 :=
by
  sorry

end sale_in_third_month_l46_46575


namespace solve_equation_in_natural_numbers_l46_46376

-- Define the main theorem
theorem solve_equation_in_natural_numbers :
  (∃ (x y z : ℕ), 2^x + 5^y + 63 = z! ∧ ((x = 5 ∧ y = 2 ∧ z = 5) ∨ (x = 4 ∧ y = 4 ∧ z = 6))) :=
sorry

end solve_equation_in_natural_numbers_l46_46376


namespace matrix_pow_sub_l46_46854

open Matrix

noncomputable def B : Matrix (Fin 2) (Fin 2) ℚ := !![3, 4; 0, 2]

theorem matrix_pow_sub : 
  B^10 - 3 • B^9 = !![0, 4; 0, -1] := 
by
  sorry

end matrix_pow_sub_l46_46854


namespace union_A_B_eq_univ_inter_compl_A_B_eq_interval_subset_B_range_of_a_l46_46006

variable (A B C : Set ℝ)
variable (a : ℝ)

-- Condition definitions
def set_A : Set ℝ := {x | x ≤ 3 ∨ x ≥ 6}
def set_B : Set ℝ := {x | -2 < x ∧ x < 9}
def set_C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

-- Proof statement (1)
theorem union_A_B_eq_univ (A B : Set ℝ) (h₁ : A = set_A) (h₂ : B = set_B) :
  A ∪ B = Set.univ := by sorry

theorem inter_compl_A_B_eq_interval (A B : Set ℝ) (h₁ : A = set_A) (h₂ : B = set_B) :
  (Set.univ \ A) ∩ B = {x | 3 < x ∧ x < 6} := by sorry

-- Proof statement (2)
theorem subset_B_range_of_a (a : ℝ) (h : set_C a ⊆ set_B) :
  -2 ≤ a ∧ a ≤ 8 := by sorry

end union_A_B_eq_univ_inter_compl_A_B_eq_interval_subset_B_range_of_a_l46_46006


namespace infinite_k_lcm_gt_ck_l46_46673

theorem infinite_k_lcm_gt_ck 
  (a : ℕ → ℕ) 
  (distinct_pos : ∀ n m : ℕ, n ≠ m → a n ≠ a m) 
  (pos : ∀ n, 0 < a n) 
  (c : ℝ) 
  (c_pos : 0 < c) 
  (c_lt : c < 1.5) : 
  ∃ᶠ k in at_top, (Nat.lcm (a k) (a (k + 1)) : ℝ) > c * k :=
sorry

end infinite_k_lcm_gt_ck_l46_46673


namespace product_gcd_lcm_8_12_l46_46555

theorem product_gcd_lcm_8_12 : Nat.gcd 8 12 * Nat.lcm 8 12 = 96 := by
  sorry

end product_gcd_lcm_8_12_l46_46555


namespace lenny_remaining_amount_l46_46002

theorem lenny_remaining_amount :
  let initial_amount := 270
  let console_price := 149
  let console_discount := 0.15 * console_price
  let final_console_price := console_price - console_discount
  let groceries_price := 60
  let groceries_discount := 0.10 * groceries_price
  let final_groceries_price := groceries_price - groceries_discount
  let lunch_cost := 30
  let magazine_cost := 3.99
  let total_expenses := final_console_price + final_groceries_price + lunch_cost + magazine_cost
  initial_amount - total_expenses = 55.36 :=
by
  sorry

end lenny_remaining_amount_l46_46002


namespace wallace_fulfills_orders_in_13_days_l46_46897

def batch_small_bags_production := 12
def batch_large_bags_production := 8
def time_per_small_batch := 8
def time_per_large_batch := 12
def daily_production_limit := 18

def initial_stock_small := 18
def initial_stock_large := 10

def order1_small := 45
def order1_large := 30
def order2_small := 60
def order2_large := 25
def order3_small := 52
def order3_large := 42

def total_small_bags_needed := order1_small + order2_small + order3_small
def total_large_bags_needed := order1_large + order2_large + order3_large
def small_bags_to_produce := total_small_bags_needed - initial_stock_small
def large_bags_to_produce := total_large_bags_needed - initial_stock_large

def small_batches_needed := (small_bags_to_produce + batch_small_bags_production - 1) / batch_small_bags_production
def large_batches_needed := (large_bags_to_produce + batch_large_bags_production - 1) / batch_large_bags_production

def total_time_small_batches := small_batches_needed * time_per_small_batch
def total_time_large_batches := large_batches_needed * time_per_large_batch
def total_production_time := total_time_small_batches + total_time_large_batches

def days_needed := (total_production_time + daily_production_limit - 1) / daily_production_limit

theorem wallace_fulfills_orders_in_13_days :
  days_needed = 13 := by
  sorry

end wallace_fulfills_orders_in_13_days_l46_46897


namespace anniversary_day_of_week_probability_l46_46104

/-- The 11th anniversary of Robinson Crusoe and Friday's meeting can fall on a Friday with a
probability of 3/4 and on a Thursday with a probability of 1/4, given that the meeting occurred
in any year from 1668 to 1671 with equal probability. -/
theorem anniversary_day_of_week_probability :
  let years := {1668, 1669, 1670, 1671},
      leap (y : ℕ) := y % 4 = 0,
      days_in_year := λ y, if leap y then 366 else 365,
      total_days (yr : ℕ) := list.sum (list.map days_in_year (list.range' yr 11)),
      day_of_week_after_11_years (initial_year : ℕ) := total_days initial_year % 7 = 0,
      events := {week_day | ∀ y ∈ years, (day_of_week_after_11_years y)},
      friday_probability := rat.mk 3 4,
      thursday_probability := rat.mk 1 4
  in
  (events = {0} ∨ events = {6}) ∧
  (events = {0} → friday_probability = rat.mk 3 4 ∧ thursday_probability = rat.mk 1 4) ∧
  (events = {6} → friday_probability = rat.mk 1 4 ∧ thursday_probability = rat.mk 3 4):=
begin
  sorry
end

end anniversary_day_of_week_probability_l46_46104


namespace triangle_cosine_theorem_l46_46478

def triangle_sums (a b c : ℝ) : ℝ := 
  b^2 + c^2 - a^2 + a^2 + c^2 - b^2 + a^2 + b^2 - c^2

theorem triangle_cosine_theorem (a b c : ℝ) (cos_A cos_B cos_C : ℝ) :
  a = 2 → b = 3 → c = 4 → 2 * b * c * cos_A + 2 * c * a * cos_B + 2 * a * b * cos_C = 29 :=
by
  intros h₁ h₂ h₃
  sorry

end triangle_cosine_theorem_l46_46478


namespace smallest_angle_between_lines_l46_46055

theorem smallest_angle_between_lines (r1 r2 r3 : ℝ) (S U : ℝ) (h1 : r1 = 4) (h2 : r2 = 3) 
  (h3 : r3 = 2) (total_area : ℝ := π * (r1^2 + r2^2 + r3^2)) 
  (h4 : S = (5 / 8) * U) (h5 : S + U = total_area) : 
  ∃ θ : ℝ, θ = (5 * π) / 13 :=
by
  sorry

end smallest_angle_between_lines_l46_46055


namespace minibus_seat_count_l46_46730

theorem minibus_seat_count 
  (total_children : ℕ) 
  (seats_with_3_children : ℕ) 
  (children_per_3_child_seat : ℕ) 
  (remaining_children : ℕ) 
  (children_per_2_child_seat : ℕ) 
  (total_seats : ℕ) :
  total_children = 19 →
  seats_with_3_children = 5 →
  children_per_3_child_seat = 3 →
  remaining_children = total_children - (seats_with_3_children * children_per_3_child_seat) →
  children_per_2_child_seat = 2 →
  total_seats = seats_with_3_children + (remaining_children / children_per_2_child_seat) →
  total_seats = 7 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end minibus_seat_count_l46_46730


namespace estimate_mass_of_ice_floe_l46_46015

noncomputable def mass_of_ice_floe (d : ℝ) (D : ℝ) (m : ℝ) : ℝ :=
  (m * d) / (D - d)

theorem estimate_mass_of_ice_floe :
  mass_of_ice_floe 9.5 10 600 = 11400 := 
by
  sorry

end estimate_mass_of_ice_floe_l46_46015


namespace tangent_line_to_circle_l46_46691

theorem tangent_line_to_circle (a : ℝ) :
  (∃ k : ℝ, k = a ∧ (∀ x y : ℝ, y = x + 4 → (x - k)^2 + (y - 3)^2 = 8)) ↔ (a = 3 ∨ a = -5) := by
  sorry

end tangent_line_to_circle_l46_46691


namespace sum_abs_coeffs_l46_46148

theorem sum_abs_coeffs (a : ℝ → ℝ) :
  (∀ x, (1 - 3 * x)^9 = a 0 + a 1 * x + a 2 * x^2 + a 3 * x^3 + a 4 * x^4 + a 5 * x^5 + a 6 * x^6 + a 7 * x^7 + a 8 * x^8 + a 9 * x^9) →
  |a 0| + |a 1| + |a 2| + |a 3| + |a 4| + |a 5| + |a 6| + |a 7| + |a 8| + |a 9| = 4^9 := by
  sorry

end sum_abs_coeffs_l46_46148


namespace mushroom_collectors_l46_46020

theorem mushroom_collectors :
  ∃ (n m : ℕ), 13 * n - 10 * m = 2 ∧ 9 ≤ n ∧ n ≤ 15 ∧ 11 ≤ m ∧ m ≤ 20 ∧ n = 14 ∧ m = 18 := by sorry

end mushroom_collectors_l46_46020


namespace exists_distinct_positive_integers_l46_46420

theorem exists_distinct_positive_integers (n : ℕ) (h : 0 < n) :
  ∃ (x y z : ℕ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x^(n-1) + y^n = z^(n+1) :=
sorry

end exists_distinct_positive_integers_l46_46420


namespace find_numbers_l46_46057

theorem find_numbers (a b : ℝ) (h1 : a * b = 5) (h2 : a + b = 8) (ha_pos : 0 < a) (hb_pos : 0 < b) :
  (a = 4 + Real.sqrt 11 ∧ b = 4 - Real.sqrt 11) ∨ (a = 4 - Real.sqrt 11 ∧ b = 4 + Real.sqrt 11) :=
sorry

end find_numbers_l46_46057


namespace inequality_transformation_l46_46568

variable {a b : ℝ}

theorem inequality_transformation (h : a < b) : -a / 3 > -b / 3 :=
  sorry

end inequality_transformation_l46_46568


namespace trig_identity_l46_46731

theorem trig_identity : 2 * Real.sin (75 * Real.pi / 180) * Real.cos (15 * Real.pi / 180) - 1 = Real.sqrt 3 / 2 :=
by
  sorry

end trig_identity_l46_46731


namespace sqrt_sum_of_fractions_l46_46901

theorem sqrt_sum_of_fractions :
  (Real.sqrt ((25 / 36) + (16 / 9)) = (Real.sqrt 89) / 6) :=
by
  sorry

end sqrt_sum_of_fractions_l46_46901


namespace total_yearly_interest_l46_46177

/-- Mathematical statement:
Given Nina's total inheritance of $12,000, with $5,000 invested at 6% interest and the remainder invested at 8% interest, the total yearly interest from both investments is $860.
-/
theorem total_yearly_interest (principal : ℕ) (principal_part : ℕ) (rate1 rate2 : ℚ) (interest_part1 interest_part2 : ℚ) (total_interest : ℚ) :
  principal = 12000 ∧ principal_part = 5000 ∧ rate1 = 0.06 ∧ rate2 = 0.08 ∧
  interest_part1 = (principal_part : ℚ) * rate1 ∧ interest_part2 = ((principal - principal_part) : ℚ) * rate2 →
  total_interest = interest_part1 + interest_part2 → 
  total_interest = 860 := by
  sorry

end total_yearly_interest_l46_46177


namespace fraction_halfway_l46_46219

theorem fraction_halfway (a b c d : ℚ) (h₁ : a = 3) (h₂ : b = 4) (h₃ : c = 5) (h₄ : d = 6) :
  (1 / 2) * ((a / b) + (c / d)) = 19 / 24 := 
by
  sorry

end fraction_halfway_l46_46219


namespace mark_deposit_is_88_l46_46859

-- Definitions according to the conditions
def markDeposit := 88
def bryanDeposit (m : ℕ) := 5 * m - 40

-- The theorem we need to prove
theorem mark_deposit_is_88 : markDeposit = 88 := 
by 
  -- Since the condition states Mark deposited $88,
  -- this is trivially true.
  sorry

end mark_deposit_is_88_l46_46859


namespace expected_difference_is_91_5_l46_46099

noncomputable def expected_difference_between_toast_days : ℚ :=
  let die_faces := {1, 2, 3, 4, 5, 6, 7, 8}
  let perfect_squares := {1, 4}
  let primes := {2, 3, 5, 7}
  let prob_perfect_square := (perfect_squares.card : ℚ) / (die_faces.card : ℚ)
  let prob_prime := (primes.card : ℚ) / (die_faces.card : ℚ)
  let days_in_leap_year : ℚ := 366
  let expected_days_toast_jam := prob_perfect_square * days_in_leap_year
  let expected_days_toast_butter := prob_prime * days_in_leap_year
  expected_days_toast_butter - expected_days_toast_jam

theorem expected_difference_is_91_5 :
  expected_difference_between_toast_days = 91.5 :=
by
  sorry

end expected_difference_is_91_5_l46_46099


namespace max_gcd_of_linear_combinations_l46_46418

theorem max_gcd_of_linear_combinations (a b c : ℕ) (h1 : a + b + c ≤ 3000000) (h2 : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  gcd (a * b + 1) (gcd (a * c + 1) (b * c + 1)) ≤ 998285 :=
sorry

end max_gcd_of_linear_combinations_l46_46418


namespace total_bouncy_balls_l46_46508

def red_packs := 4
def yellow_packs := 8
def green_packs := 4
def blue_packs := 6

def red_balls_per_pack := 12
def yellow_balls_per_pack := 10
def green_balls_per_pack := 14
def blue_balls_per_pack := 8

def total_red_balls := red_packs * red_balls_per_pack
def total_yellow_balls := yellow_packs * yellow_balls_per_pack
def total_green_balls := green_packs * green_balls_per_pack
def total_blue_balls := blue_packs * blue_balls_per_pack

def total_balls := total_red_balls + total_yellow_balls + total_green_balls + total_blue_balls

theorem total_bouncy_balls : total_balls = 232 :=
by
  -- calculation proof goes here
  sorry

end total_bouncy_balls_l46_46508


namespace arithmetic_sequence_common_difference_l46_46280

theorem arithmetic_sequence_common_difference :
  ∃ d : ℤ, 
    (∀ n, n ≤ 6 → 23 + (n - 1) * d > 0) ∧ 
    (∀ n, n ≥ 7 → 23 + (n - 1) * d < 0) ∧
    d = -4 :=
by
  sorry

end arithmetic_sequence_common_difference_l46_46280


namespace modular_inverse_of_17_mod_800_l46_46554

    theorem modular_inverse_of_17_mod_800 :
      ∃ x : ℤ, 0 ≤ x ∧ x < 800 ∧ (17 * x) % 800 = 1 :=
    by
      use 47
      sorry
    
end modular_inverse_of_17_mod_800_l46_46554


namespace Bryan_deposit_amount_l46_46862

theorem Bryan_deposit_amount (deposit_mark : ℕ) (deposit_bryan : ℕ)
  (h1 : deposit_mark = 88)
  (h2 : deposit_bryan = 5 * deposit_mark - 40) : 
  deposit_bryan = 400 := 
by
  sorry

end Bryan_deposit_amount_l46_46862


namespace associate_professors_bring_2_pencils_l46_46444

theorem associate_professors_bring_2_pencils (A B P : ℕ) 
  (h1 : A + B = 5)
  (h2 : P * A + B = 10)
  (h3 : A + 2 * B = 5)
  : P = 2 :=
by {
  -- Proof goes here
  sorry
}

end associate_professors_bring_2_pencils_l46_46444


namespace distance_walked_on_third_day_l46_46660

theorem distance_walked_on_third_day:
  ∃ x : ℝ, 
    4 * x + 2 * x + x + (1 / 2) * x + (1 / 4) * x + (1 / 8) * x = 378 ∧
    x = 48 := 
by
  sorry

end distance_walked_on_third_day_l46_46660


namespace probability_at_least_6_heads_in_8_flips_l46_46747

def binomial (n k : ℕ) : ℕ :=
  if h : k ≤ n then (Nat.choose n k) else 0

theorem probability_at_least_6_heads_in_8_flips :
  let total_outcomes := 2^8
  let successful_outcomes := binomial 8 6 + binomial 8 7 + binomial 8 8
  let probability := (successful_outcomes : ℚ) / total_outcomes
  probability = 37 / 256 :=
by
  sorry

end probability_at_least_6_heads_in_8_flips_l46_46747


namespace reciprocal_of_minus_one_over_2023_l46_46043

theorem reciprocal_of_minus_one_over_2023 : (1 / (- (1 / 2023))) = -2023 := 
by
  sorry

end reciprocal_of_minus_one_over_2023_l46_46043


namespace intersection_of_A_and_B_l46_46827

def A : Set ℤ := {-2, -1}
def B : Set ℤ := {-1, 2, 3}

theorem intersection_of_A_and_B :
  A ∩ B = {-1} :=
sorry

end intersection_of_A_and_B_l46_46827


namespace initial_population_first_village_equals_l46_46581

-- Definitions of the conditions
def initial_population_second_village : ℕ := 42000
def decrease_first_village_per_year : ℕ := 1200
def increase_second_village_per_year : ℕ := 800
def years : ℕ := 13

-- Proposition we want to prove
/-- The initial population of the first village such that both villages have the same population after 13 years. -/
theorem initial_population_first_village_equals :
  ∃ (P : ℕ), (P - decrease_first_village_per_year * years) = (initial_population_second_village + increase_second_village_per_year * years) 
  := sorry

end initial_population_first_village_equals_l46_46581


namespace area_of_circle_l46_46796

noncomputable def circle_eq : ℝ × ℝ → Prop :=
  λ (x, y), x^2 + y^2 - 6 * x + 8 * y - 11 = 0

theorem area_of_circle : ∃ (area : ℝ), circle_eq (x, y) → area = 14 * Real.pi :=
begin
  sorry
end

end area_of_circle_l46_46796


namespace probability_at_least_6_heads_in_8_flips_l46_46746

def binomial (n k : ℕ) : ℕ :=
  if h : k ≤ n then (Nat.choose n k) else 0

theorem probability_at_least_6_heads_in_8_flips :
  let total_outcomes := 2^8
  let successful_outcomes := binomial 8 6 + binomial 8 7 + binomial 8 8
  let probability := (successful_outcomes : ℚ) / total_outcomes
  probability = 37 / 256 :=
by
  sorry

end probability_at_least_6_heads_in_8_flips_l46_46746


namespace length_of_bridge_is_255_l46_46879

noncomputable def bridge_length (train_length : ℕ) (train_speed_kph : ℕ) (cross_time_sec : ℕ) : ℕ :=
  let train_speed_mps := train_speed_kph * 1000 / (60 * 60)
  let total_distance := train_speed_mps * cross_time_sec
  total_distance - train_length

theorem length_of_bridge_is_255 :
  ∀ (train_length : ℕ) (train_speed_kph : ℕ) (cross_time_sec : ℕ), 
    train_length = 120 →
    train_speed_kph = 45 →
    cross_time_sec = 30 →
    bridge_length train_length train_speed_kph cross_time_sec = 255 :=
by
  intros train_length train_speed_kph cross_time_sec htl htsk hcts
  simp [bridge_length]
  rw [htl, htsk, hcts]
  norm_num
  sorry

end length_of_bridge_is_255_l46_46879


namespace ellipse_foci_distance_l46_46590

noncomputable def distance_between_foci_of_ellipse (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 - b^2)

theorem ellipse_foci_distance :
  ∀ (a b : ℝ), (a = 5) → (b = 2) →
  distance_between_foci_of_ellipse a b = Real.sqrt 21 :=
by
  intros a b ha hb
  rw [ha, hb]
  -- The rest of the proof is omitted
  sorry

end ellipse_foci_distance_l46_46590


namespace chromatic_number_bound_l46_46816

-- Definitions of the conditions
variable (G : SimpleGraph V) [DecidableRel G.Adj] (Δ : ℕ)

-- Definition of the maximum degree of a vertex
def max_degree (G : SimpleGraph V) : ℕ := 
  Finset.sup (G.support) (λ v, G.degree v)

-- Assertion about the chromatic number
theorem chromatic_number_bound (hΔ : Δ = max_degree G) : 
    G.chromatic_number ≤ Δ + 1 := sorry

end chromatic_number_bound_l46_46816


namespace number_of_legs_twice_heads_diff_eq_22_l46_46341

theorem number_of_legs_twice_heads_diff_eq_22 (P H : ℕ) (L : ℤ) (Heads : ℕ) (X : ℤ) (h1 : P = 11)
  (h2 : L = 4 * P + 2 * H) (h3 : Heads = P + H) (h4 : L = 2 * Heads + X) : X = 22 :=
by
  sorry

end number_of_legs_twice_heads_diff_eq_22_l46_46341


namespace symmetric_line_l46_46877

theorem symmetric_line (x y : ℝ) : 
  (∀ (x y  : ℝ), 2 * x + y - 1 = 0) ∧ (∀ (x  : ℝ), x = 1) → (2 * x - y - 3 = 0) :=
by
  sorry

end symmetric_line_l46_46877


namespace arithmetic_sequence_problem_l46_46821

variable (a : ℕ → ℕ)
variable (d : ℕ) -- common difference for the arithmetic sequence
variable (h1 : ∀ n : ℕ, a (n + 1) = a n + d)
variable (h2 : a 1 - a 9 + a 17 = 7)

theorem arithmetic_sequence_problem : a 3 + a 15 = 14 := by
  sorry

end arithmetic_sequence_problem_l46_46821


namespace perfect_square_difference_l46_46385

def lastDigit (n : ℕ) : ℕ :=
  n % 10

theorem perfect_square_difference :
  ∃ a b : ℕ, ∃ x y : ℕ,
    a = x^2 ∧ b = y^2 ∧
    lastDigit a = 6 ∧
    lastDigit b = 4 ∧
    lastDigit (a - b) = 2 ∧
    lastDigit a > lastDigit b :=
by
  sorry

end perfect_square_difference_l46_46385


namespace transform_equation_l46_46548

theorem transform_equation (x : ℝ) : x^2 - 2 * x - 2 = 0 ↔ (x - 1)^2 = 3 :=
sorry

end transform_equation_l46_46548


namespace exists_sol_in_naturals_l46_46368

theorem exists_sol_in_naturals : ∃ (x y : ℕ), x^2 + y^2 = 61^3 := 
sorry

end exists_sol_in_naturals_l46_46368


namespace anniversary_day_probability_l46_46102

-- Define the years in which the meeting could take place
def years := {1668, 1669, 1670, 1671}

-- Define a function to check if a year is a leap year
def is_leap_year (y : ℕ) : Prop := (y % 4 = 0)

-- Define the meeting date
def meeting_day := 5 -- Friday as the 5th day of the week (assuming 0 = Sunday)

-- Define the anniversary function that computes the day of the 11th anniversary
def anniversary_day (start_year : ℕ) : ℕ :=
  let days_in_year (y : ℕ) := if is_leap_year y then 366 else 365
  let total_days := (List.range 11).map (λ i => days_in_year (start_year + i))
  (total_days.sum + meeting_day) % 7

-- Define the probability computation
def probability (day : ℕ) : ℚ :=
  let occurences := (years.toList.map anniversary_day).count (λ d => d = day)
  occurences / years.toList.length

-- Statement of the theorem
theorem anniversary_day_probability :
  probability 5 = 3 / 4 ∧ probability 4 = 1 / 4 := -- 5 is Friday, 4 is Thursday
  by
    sorry -- Proof goes here

end anniversary_day_probability_l46_46102


namespace taxi_speed_is_60_l46_46237

theorem taxi_speed_is_60 (v_b v_t : ℝ) (h1 : v_b = v_t - 30) (h2 : 3 * v_t = 6 * v_b) : v_t = 60 := 
by 
  sorry

end taxi_speed_is_60_l46_46237


namespace sum_of_consecutive_integers_a_lt_sqrt3_lt_b_l46_46643

theorem sum_of_consecutive_integers_a_lt_sqrt3_lt_b 
  (a b : ℤ) (h1 : a < b) (h2 : ∀ x : ℤ, x ≤ a → x < b) (h3 : a < Real.sqrt 3) (h4 : Real.sqrt 3 < b) : 
  a + b = 3 :=
by
  sorry

end sum_of_consecutive_integers_a_lt_sqrt3_lt_b_l46_46643


namespace sufficient_but_not_necessary_condition_l46_46278

theorem sufficient_but_not_necessary_condition (m : ℝ) :
  (∀ x > 0, (m^2 - m - 1) * x^(m - 1) > 0 → m = 2) →
  (|m - 2| < 1) :=
by
  sorry

end sufficient_but_not_necessary_condition_l46_46278


namespace probability_at_least_6_heads_in_8_flips_l46_46761

theorem probability_at_least_6_heads_in_8_flips : 
  (∑ k in finset.range 3, nat.choose 8 (6 + k)) / (2 ^ 8) = 37 / 256 :=
by sorry

end probability_at_least_6_heads_in_8_flips_l46_46761


namespace A_salary_l46_46723

theorem A_salary (x y : ℝ) (h1 : x + y = 7000) (h2 : 0.05 * x = 0.15 * y) : x = 5250 :=
by
  sorry

end A_salary_l46_46723


namespace range_of_k_l46_46109

def tensor (a b : ℝ) : ℝ := a * b + a + b^2

theorem range_of_k (k : ℝ) : (∀ x : ℝ, tensor k x > 0) ↔ (0 < k ∧ k < 4) :=
by
  sorry

end range_of_k_l46_46109


namespace x_squared_minus_y_squared_l46_46835

theorem x_squared_minus_y_squared
    (x y : ℚ) 
    (h1 : x + y = 3 / 8) 
    (h2 : x - y = 1 / 4) : x^2 - y^2 = 3 / 32 := 
by 
    sorry

end x_squared_minus_y_squared_l46_46835


namespace simplify_expression_l46_46370

variable (x y : ℝ)

theorem simplify_expression (h : x ≠ y ∧ x ≠ -y) : 
  ((1 / (x - y) - 1 / (x + y)) / (x * y / (x^2 - y^2)) = 2 / x) :=
by sorry

end simplify_expression_l46_46370


namespace groupB_is_basis_l46_46781

section
variables (eA1 eA2 : ℝ × ℝ) (eB1 eB2 : ℝ × ℝ) (eC1 eC2 : ℝ × ℝ) (eD1 eD2 : ℝ × ℝ)

def is_collinear (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = (k • w) ∨ w = (k • v)

-- Define each vector group
def groupA := eA1 = (0, 0) ∧ eA2 = (1, -2)
def groupB := eB1 = (-1, 2) ∧ eB2 = (5, 7)
def groupC := eC1 = (3, 5) ∧ eC2 = (6, 10)
def groupD := eD1 = (2, -3) ∧ eD2 = (1/2, -3/4)

-- The goal is to prove that group B vectors can serve as a basis
theorem groupB_is_basis : ¬ is_collinear eB1 eB2 :=
sorry
end

end groupB_is_basis_l46_46781


namespace probability_at_least_6_heads_in_8_flips_l46_46743

def binomial (n k : ℕ) : ℕ :=
  if h : k ≤ n then (Nat.choose n k) else 0

theorem probability_at_least_6_heads_in_8_flips :
  let total_outcomes := 2^8
  let successful_outcomes := binomial 8 6 + binomial 8 7 + binomial 8 8
  let probability := (successful_outcomes : ℚ) / total_outcomes
  probability = 37 / 256 :=
by
  sorry

end probability_at_least_6_heads_in_8_flips_l46_46743


namespace chocolate_chips_needed_l46_46903

-- Define the variables used in the conditions
def cups_per_recipe := 2
def number_of_recipes := 23

-- State the theorem
theorem chocolate_chips_needed : (cups_per_recipe * number_of_recipes) = 46 := 
by sorry

end chocolate_chips_needed_l46_46903


namespace greatest_product_three_integers_sum_2000_l46_46060

noncomputable def maxProduct (s : ℝ) : ℝ := 
  s * s * (2000 - 2 * s)

theorem greatest_product_three_integers_sum_2000 : ∃ x : ℝ, 0 ≤ x ∧ x ≤ 2000 / 2 ∧ maxProduct x = 8000000000 / 27 := sorry

end greatest_product_three_integers_sum_2000_l46_46060


namespace convert_decimal_to_vulgar_fraction_l46_46562

theorem convert_decimal_to_vulgar_fraction : (32 : ℝ) / 100 = (8 : ℝ) / 25 :=
by
  sorry

end convert_decimal_to_vulgar_fraction_l46_46562


namespace evaluate_expression_l46_46872

open BigOperators

theorem evaluate_expression : 
  ∀ (x y : ℤ), x = -1 → y = 1 → 2 * (x^2 * y + x * y) - 3 * (x^2 * y - x * y) - 5 * x * y = -1 :=
by
  intros x y hx hy
  rw [hx, hy]
  sorry

end evaluate_expression_l46_46872


namespace find_F2_l46_46896

-- Set up the conditions as definitions
def m : ℝ := 1 -- in kg
def R1 : ℝ := 0.5 -- in meters
def R2 : ℝ := 1 -- in meters
def F1 : ℝ := 1 -- in Newtons

-- Rotational inertia I formula
def I (R : ℝ) : ℝ := m * R^2

-- Equality of angular accelerations
def alpha_eq (F1 F2 R1 R2 : ℝ) : Prop :=
  (F1 * R1) / (I R1) = (F2 * R2) / (I R2)

-- The proof goal
theorem find_F2 (F2 : ℝ) : 
  alpha_eq F1 F2 R1 R2 → F2 = 2 :=
by
  sorry

end find_F2_l46_46896


namespace product_of_roots_l46_46476

theorem product_of_roots (x1 x2 : ℝ) (h : x1 * x2 = -2) :
  (∀ x : ℝ, x^2 + x - 2 = 0 → (x = x1 ∨ x = x2)) → x1 * x2 = -2 :=
by
  intros h_root
  exact h

end product_of_roots_l46_46476


namespace Tom_time_to_complete_wall_after_one_hour_l46_46665

noncomputable def avery_rate : ℝ := 1 / 2
noncomputable def tom_rate : ℝ := 1 / 4
noncomputable def combined_rate : ℝ := avery_rate + tom_rate
noncomputable def wall_built_in_first_hour : ℝ := combined_rate * 1
noncomputable def remaining_wall : ℝ := 1 - wall_built_in_first_hour 
noncomputable def tom_time_to_complete_remaining_wall : ℝ := remaining_wall / tom_rate

theorem Tom_time_to_complete_wall_after_one_hour : 
  tom_time_to_complete_remaining_wall = 1 :=
by
  sorry

end Tom_time_to_complete_wall_after_one_hour_l46_46665


namespace tangent_of_7pi_over_4_l46_46120

   theorem tangent_of_7pi_over_4 : Real.tan (7 * Real.pi / 4) = -1 := 
   sorry
   
end tangent_of_7pi_over_4_l46_46120


namespace prob_two_out_of_three_l46_46053

open ProbabilityTheory MeasureTheory

noncomputable def dist_weight_cow : ℕ → ℝ → ℝ → prob := sorry
noncomputable def normal_prob_2_of_3 (μ σ : ℝ) (p k n : ℝ) : ℝ := 
  ∑ y in (finset.Icc 0 2), (Nat.choose n k : ℝ) * (p ^ k) * ((1 - p) ^ (n - k))

theorem prob_two_out_of_three (μ σ : ℝ) (prob_interval : ℝ) : 
  μ = 470 ∧ σ = 30 ∧ prob_interval = (cdf (NormalDistribution.mk μ σ^2) 530 - cdf (NormalDistribution.mk μ σ^2) 470) / ∑ y in (finset.Icc 0 2), (Nat.choose 3 2 * ((cdf (NormalDistribution.mk μ σ^2) 530 - cdf (NormalDistribution.mk μ σ^2) 470) ^ 2) * ((1 - (cdf (NormalDistribution.mk μ σ^2) 530 - cdf (NormalDistribution.mk μ σ^2) 470)) ^ 1)) :
  normal_prob_2_of_3 μ σ 2 3 = 0.357 := sorry

end prob_two_out_of_three_l46_46053


namespace range_of_a_l46_46675

theorem range_of_a (a : ℝ) : 
  (∀ (x1 : ℝ), ∃ (x2 : ℝ), |x1| = Real.log (a * x2^2 - 4 * x2 + 1)) → (0 ≤ a) :=
by
  sorry

end range_of_a_l46_46675


namespace other_number_is_twelve_l46_46079

variable (x certain_number : ℕ)
variable (h1: certain_number = 60)
variable (h2: certain_number = 5 * x)

theorem other_number_is_twelve :
  x = 12 :=
by
  sorry

end other_number_is_twelve_l46_46079


namespace proof_equivalent_problem_l46_46825

noncomputable def polar_equation_curve : Prop :=
  ∀ (α : ℝ), 
    let x := 3 + 2 * Real.cos α;
    let y := 1 - 2 * Real.sin α;
    (x - 3) ^ 2 + (y - 1) ^ 2 - 4 = 0

noncomputable def polar_equation_line : Prop :=
  ∀ (θ ρ : ℝ), 
  (Real.sin θ - 2 * Real.cos θ = 1 / ρ) → (2 * (ρ * Real.cos θ) - (ρ * Real.sin θ) + 1 = 0)

noncomputable def distance_from_curve_to_line : Prop :=
  ∀ (α : ℝ), 
    let x := 3 + 2 * Real.cos α;
    let y := 1 - 2 * Real.sin α;
    ∃ d : ℝ, d = (|2 * x - y + 1| / Real.sqrt (2 ^ 2 + 1)) ∧
    d + 2 = (6 * Real.sqrt 5 / 5) + 2

theorem proof_equivalent_problem :
  polar_equation_curve ∧ polar_equation_line ∧ distance_from_curve_to_line :=
by
  constructor
  · exact sorry  -- polar_equation_curve proof
  · constructor
    · exact sorry  -- polar_equation_line proof
    · exact sorry  -- distance_from_curve_to_line proof

end proof_equivalent_problem_l46_46825


namespace evaluate_expression_l46_46794

theorem evaluate_expression :
  (4 * 10^2011 - 1) / (4 * (3 * (10^2011 - 1) / 9) + 1) = 3 :=
by
  sorry

end evaluate_expression_l46_46794


namespace sum_six_seven_l46_46465

variable (a : ℕ → ℝ)
variable (d : ℝ)

axiom arithmetic_sequence : ∀ (n : ℕ), a (n + 1) = a n + d
axiom sum_condition : a 2 + a 5 + a 8 + a 11 = 48

theorem sum_six_seven : a 6 + a 7 = 24 :=
by
  -- Using given axioms and properties of arithmetic sequence
  sorry

end sum_six_seven_l46_46465


namespace reciprocal_of_neg_one_div_2023_l46_46038

theorem reciprocal_of_neg_one_div_2023 : 1 / (-1 / (2023 : ℤ)) = -2023 := sorry

end reciprocal_of_neg_one_div_2023_l46_46038


namespace lana_extra_flowers_l46_46351

theorem lana_extra_flowers :
  ∀ (tulips roses used total_extra : ℕ),
    tulips = 36 →
    roses = 37 →
    used = 70 →
    total_extra = (tulips + roses - used) →
    total_extra = 3 :=
by
  intros tulips roses used total_extra ht hr hu hte
  rw [ht, hr, hu] at hte
  sorry

end lana_extra_flowers_l46_46351


namespace compare_groups_l46_46845

noncomputable def mean (scores : List ℝ) : ℝ :=
  (scores.sum / scores.length)

noncomputable def variance (scores : List ℝ) : ℝ :=
  let m := mean scores
  (scores.map (λ x => (x - m) ^ 2)).sum / scores.length

noncomputable def stddev (scores : List ℝ) : ℝ :=
  (variance scores).sqrt

def groupA_scores : List ℝ := [88, 100, 95, 86, 95, 91, 84, 74, 92, 83]
def groupB_scores : List ℝ := [93, 89, 81, 77, 96, 78, 77, 85, 89, 86]

theorem compare_groups :
  mean groupA_scores > mean groupB_scores ∧ stddev groupA_scores > stddev groupB_scores :=
by
  sorry

end compare_groups_l46_46845


namespace quadratic_has_real_roots_find_value_of_m_l46_46293

theorem quadratic_has_real_roots (m : ℝ) : 
  ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (x1 - x2) ≠ 0 ∧ (x^2 - 4 * m * x + 3 * m^2 = 0) := 
by 
  sorry

theorem find_value_of_m (m : ℝ) (h1 : m > 0) (h2 : ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (x1 - x2 = 2)) :
  m = 1 := 
by 
  sorry

end quadratic_has_real_roots_find_value_of_m_l46_46293


namespace weeks_to_work_l46_46937

def iPhone_cost : ℕ := 800
def trade_in_value : ℕ := 240
def weekly_earnings : ℕ := 80

theorem weeks_to_work (iPhone_cost trade_in_value weekly_earnings : ℕ) :
  (iPhone_cost - trade_in_value) / weekly_earnings = 7 :=
by
  sorry

end weeks_to_work_l46_46937


namespace clusters_per_spoonful_l46_46510

theorem clusters_per_spoonful (spoonfuls_per_bowl : ℕ) (clusters_per_box : ℕ) (bowls_per_box : ℕ) 
  (h_spoonfuls : spoonfuls_per_bowl = 25) 
  (h_clusters : clusters_per_box = 500)
  (h_bowls : bowls_per_box = 5) : 
  clusters_per_box / bowls_per_box / spoonfuls_per_bowl = 4 := 
by 
  have clusters_per_bowl := clusters_per_box / bowls_per_box
  have clusters_per_spoonful := clusters_per_bowl / spoonfuls_per_bowl
  sorry

end clusters_per_spoonful_l46_46510


namespace hyperbola_sum_l46_46340

noncomputable def h : ℝ := 3
noncomputable def k : ℝ := -4
noncomputable def a : ℝ := 4
noncomputable def c : ℝ := Real.sqrt 53
noncomputable def b : ℝ := Real.sqrt (c^2 - a^2)

theorem hyperbola_sum : h + k + a + b = 3 + Real.sqrt 37 :=
by
  -- sorry is used to skip the proof as per the instruction
  sorry
  -- exact calc
  --   h + k + a + b = 3 + (-4) + 4 + Real.sqrt 37 : by simp
  --             ... = 3 + Real.sqrt 37 : by simp

end hyperbola_sum_l46_46340


namespace perpendicular_lines_a_eq_0_or_neg1_l46_46389

theorem perpendicular_lines_a_eq_0_or_neg1 (a : ℝ) :
  (∃ (k₁ k₂: ℝ), (k₁ = a ∧ k₂ = (2 * a - 1)) ∧ ∃ (k₃ k₄: ℝ), (k₃ = 3 ∧ k₄ = a) ∧ k₁ * k₃ + k₂ * k₄ = 0) →
  (a = 0 ∨ a = -1) := 
sorry

end perpendicular_lines_a_eq_0_or_neg1_l46_46389


namespace IsoperimetricQuotient_Inequality_l46_46153

theorem IsoperimetricQuotient_Inequality 
  (ABC : Triangle) (A1 A2 : Point) 
  (h1 : A1 ∈ interior ABC)
  (h2 : A2 ∈ interior (Triangle.mk A1 ABC.B ABC.C)) :
  IQ (Triangle.mk A1 ABC.B ABC.C) > IQ (Triangle.mk A2 ABC.B ABC.C) := 
sorry

end IsoperimetricQuotient_Inequality_l46_46153


namespace avg_of_last_11_eq_41_l46_46529

def sum_of_first_11 : ℕ := 11 * 48
def sum_of_all_21 : ℕ := 21 * 44
def eleventh_number : ℕ := 55

theorem avg_of_last_11_eq_41 (S1 S : ℕ) :
  S1 = sum_of_first_11 →
  S = sum_of_all_21 →
  (S - S1 + eleventh_number) / 11 = 41 :=
by
  sorry

end avg_of_last_11_eq_41_l46_46529


namespace smallest_Norwegian_l46_46579

def is_Norwegian (n : ℕ) : Prop :=
  ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ n = a * b * c ∧ a + b + c = 2022

theorem smallest_Norwegian :
  ∀ n : ℕ, is_Norwegian n → 1344 ≤ n := by
  sorry

end smallest_Norwegian_l46_46579


namespace tank_capacity_l46_46239

theorem tank_capacity (C : ℝ) : 
  (0.5 * C = 0.9 * C - 45) → C = 112.5 :=
by
  intro h
  sorry

end tank_capacity_l46_46239


namespace number_of_episodes_l46_46404

def episode_length : ℕ := 20
def hours_per_day : ℕ := 2
def days : ℕ := 15

theorem number_of_episodes : (days * hours_per_day * 60) / episode_length = 90 :=
by
  sorry

end number_of_episodes_l46_46404


namespace four_digit_multiples_of_5_count_l46_46312

theorem four_digit_multiples_of_5_count :
  let lower_bound := 200
  let upper_bound := 1999
  (upper_bound - lower_bound + 1) = 1800 :=
by
  let lower_bound := 200
  let upper_bound := 1999
  show (upper_bound - lower_bound + 1) = 1800,
  from sorry

end four_digit_multiples_of_5_count_l46_46312


namespace find_positive_value_of_A_l46_46494

variable (A : ℝ)

-- Given conditions
def relation (A B : ℝ) : ℝ := A^2 + B^2

-- The proof statement
theorem find_positive_value_of_A (h : relation A 7 = 200) : A = Real.sqrt 151 := sorry

end find_positive_value_of_A_l46_46494


namespace find_a_range_l46_46976

-- Definitions as per conditions
def prop_P (a : ℝ) : Prop := ∀ x : ℝ, x^2 + a*x + 4 > 0
def prop_Q (a : ℝ) : Prop := ∃ x : ℝ, x^2 - 2*x + a = 0

-- Given conditions
def P_true (a : ℝ) (h : prop_P a) : Prop :=
  ∀ (a : ℝ), a^2 - 16 < 0

def Q_false (a : ℝ) (h : ¬prop_Q a) : Prop :=
  ∀ (a : ℝ), a > 1

-- Main theorem
theorem find_a_range (a : ℝ) (hP : prop_P a) (hQ : ¬prop_Q a) : 1 < a ∧ a < 4 :=
sorry

end find_a_range_l46_46976


namespace gloria_pencils_total_l46_46302

-- Define the number of pencils Gloria initially has.
def pencils_gloria_initial : ℕ := 2

-- Define the number of pencils Lisa initially has.
def pencils_lisa_initial : ℕ := 99

-- Define the final number of pencils Gloria will have after receiving all of Lisa's pencils.
def pencils_gloria_final : ℕ := pencils_gloria_initial + pencils_lisa_initial

-- Prove that the final number of pencils Gloria will have is 101.
theorem gloria_pencils_total : pencils_gloria_final = 101 :=
by sorry

end gloria_pencils_total_l46_46302


namespace probability_at_least_6_heads_in_8_flips_l46_46759

theorem probability_at_least_6_heads_in_8_flips : 
  (∑ k in finset.range 3, nat.choose 8 (6 + k)) / (2 ^ 8) = 37 / 256 :=
by sorry

end probability_at_least_6_heads_in_8_flips_l46_46759


namespace tyler_age_l46_46406

theorem tyler_age (T B : ℕ) (h1 : T = B - 3) (h2 : T + B = 11) : T = 4 :=
  sorry

end tyler_age_l46_46406


namespace find_pairs_of_numbers_l46_46888

theorem find_pairs_of_numbers (a b : ℝ) :
  (a^2 + b^2 = 15 * (a + b)) ∧ (a^2 - b^2 = 3 * (a - b) ∨ a^2 - b^2 = -3 * (a - b))
  ↔ (a = 6 ∧ b = -3) ∨ (a = -3 ∧ b = 6) ∨ (a = 0 ∧ b = 0) ∨ (a = 15 ∧ b = 15) :=
sorry

end find_pairs_of_numbers_l46_46888


namespace probability_divisible_by_5_l46_46113

def spinner_nums : List ℕ := [1, 2, 3, 5]

def total_outcomes (spins : ℕ) : ℕ :=
  List.length spinner_nums ^ spins

def count_divisible_by_5 (spins : ℕ) : ℕ :=
  let units_digit := 1
  let rest_combinations := (List.length spinner_nums) ^ (spins - units_digit)
  rest_combinations

theorem probability_divisible_by_5 : 
  let spins := 3 
  let successful_cases := count_divisible_by_5 spins
  let all_cases := total_outcomes spins
  successful_cases / all_cases = 1 / 4 :=
by
  sorry

end probability_divisible_by_5_l46_46113


namespace area_of_triangle_ABC_l46_46010

theorem area_of_triangle_ABC (BD CE : ℝ) (angle_BD_CE : ℝ) (BD_len : BD = 9) (CE_len : CE = 15) (angle_BD_CE_deg : angle_BD_CE = 60) : 
  ∃ area : ℝ, 
    area = 90 * Real.sqrt 3 := 
by
  sorry

end area_of_triangle_ABC_l46_46010


namespace superhero_speed_conversion_l46_46902

theorem superhero_speed_conversion
    (speed_km_per_min : ℕ)
    (conversion_factor : ℝ)
    (minutes_in_hour : ℕ)
    (H1 : speed_km_per_min = 1000)
    (H2 : conversion_factor = 0.6)
    (H3 : minutes_in_hour = 60) :
    (speed_km_per_min * conversion_factor * minutes_in_hour = 36000) :=
by
    sorry

end superhero_speed_conversion_l46_46902


namespace cost_of_four_dozen_l46_46100

-- Defining the conditions
def cost_of_three_dozen (cost : ℚ) : Prop :=
  cost = 25.20

-- The theorem to prove the cost of four dozen apples at the same rate
theorem cost_of_four_dozen (cost : ℚ) :
  cost_of_three_dozen cost →
  (4 * (cost / 3) = 33.60) :=
by
  sorry

end cost_of_four_dozen_l46_46100


namespace midpoint_fraction_l46_46214

theorem midpoint_fraction (a b : ℚ) (h1 : a = 3 / 4) (h2 : b = 5 / 6) :
  (a + b) / 2 = 19 / 24 :=
by {
  sorry
}

end midpoint_fraction_l46_46214


namespace modular_inverse_17_mod_800_l46_46553

theorem modular_inverse_17_mod_800 :
  ∃ x : ℤ, 17 * x ≡ 1 [MOD 800] ∧ 0 ≤ x ∧ x < 800 ∧ x = 753 := by
  sorry

end modular_inverse_17_mod_800_l46_46553


namespace thirty_times_multiple_of_every_integer_is_zero_l46_46409

theorem thirty_times_multiple_of_every_integer_is_zero (n : ℤ) (h : ∀ x : ℤ, n = 30 * x ∧ x = 0 → n = 0) : n = 0 :=
by
  sorry

end thirty_times_multiple_of_every_integer_is_zero_l46_46409


namespace arithmetic_contains_geometric_l46_46681

theorem arithmetic_contains_geometric (a d : ℕ) (h_pos_a : 0 < a) (h_pos_d : 0 < d) : 
  ∃ b q : ℕ, (b = a) ∧ (q = 1 + d) ∧ (∀ n : ℕ, ∃ k : ℕ, a * (1 + d)^n = a + k * d) :=
by
  sorry

end arithmetic_contains_geometric_l46_46681


namespace point_K_outside_hexagon_and_length_KC_l46_46857

theorem point_K_outside_hexagon_and_length_KC :
    ∀ (A B C K : ℝ × ℝ),
    A = (0, 0) →
    B = (3, 0) →
    C = (3 / 2, (3 * Real.sqrt 3) / 2) →
    K = (15 / 2, - (3 * Real.sqrt 3) / 2) →
    (¬ (0 ≤ K.1 ∧ K.1 ≤ 3 ∧ 0 ≤ K.2 ∧ K.2 ≤ 3 * Real.sqrt 3)) ∧
    Real.sqrt ((K.1 - C.1) ^ 2 + (K.2 - C.2) ^ 2) = 3 * Real.sqrt 7 :=
by
  intros A B C K hA hB hC hK
  sorry

end point_K_outside_hexagon_and_length_KC_l46_46857


namespace inscribed_circle_radius_l46_46989

theorem inscribed_circle_radius 
  (A : ℝ) -- Area of the triangle
  (p : ℝ) -- Perimeter of the triangle
  (r : ℝ) -- Radius of the inscribed circle
  (s : ℝ) -- Semiperimeter of the triangle
  (h1 : A = 2 * p) -- Condition: Area is numerically equal to twice the perimeter
  (h2 : p = 2 * s) -- Perimeter is twice the semiperimeter
  (h3 : A = r * s) -- Formula: Area in terms of inradius and semiperimeter
  (h4 : s ≠ 0) -- Semiperimeter is non-zero
  : r = 4 := 
sorry

end inscribed_circle_radius_l46_46989


namespace range_of_a_sqrt10_e_bounds_l46_46470

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log (x + 1)
noncomputable def g (x : ℝ) : ℝ := Real.exp x - 1

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ≥ 0 → f a x ≤ g x) ↔ a ≤ 1 :=
by
  sorry

theorem sqrt10_e_bounds : 
  (1095 / 1000 : ℝ) < Real.exp (1/10 : ℝ) ∧ Real.exp (1/10 : ℝ) < (2000 / 1791 : ℝ) :=
by
  sorry

end range_of_a_sqrt10_e_bounds_l46_46470


namespace mod_computation_l46_46059

theorem mod_computation (n : ℤ) : 
  0 ≤ n ∧ n < 23 ∧ 47582 % 23 = n ↔ n = 3 := 
by 
  -- Proof omitted
  sorry

end mod_computation_l46_46059


namespace nina_widgets_after_reduction_is_approx_8_l46_46865

noncomputable def nina_total_money : ℝ := 16.67
noncomputable def widgets_before_reduction : ℝ := 5
noncomputable def cost_reduction_per_widget : ℝ := 1.25

noncomputable def cost_per_widget_before_reduction : ℝ := nina_total_money / widgets_before_reduction
noncomputable def cost_per_widget_after_reduction : ℝ := cost_per_widget_before_reduction - cost_reduction_per_widget
noncomputable def widgets_after_reduction : ℝ := nina_total_money / cost_per_widget_after_reduction

-- Prove that Nina can purchase approximately 8 widgets after the cost reduction
theorem nina_widgets_after_reduction_is_approx_8 : abs (widgets_after_reduction - 8) < 1 :=
by
  sorry

end nina_widgets_after_reduction_is_approx_8_l46_46865


namespace four_digit_multiples_of_5_l46_46318

theorem four_digit_multiples_of_5 : 
  let f := fun n => 1000 <= n ∧ n <= 9999 ∧ n % 5 = 0
  in finset.card (finset.filter f (finset.range 10000)) = 1800 :=
by
  sorry

end four_digit_multiples_of_5_l46_46318


namespace minimum_area_of_rectangle_l46_46445

theorem minimum_area_of_rectangle (x y : ℝ) (h1 : x = 3) (h2 : y = 4) : 
  (min_area : ℝ) = (2.3 * 3.3) :=
by
  have length_min := x - 0.7
  have width_min := y - 0.7
  have min_area := length_min * width_min
  sorry

end minimum_area_of_rectangle_l46_46445


namespace distance_third_day_l46_46663

theorem distance_third_day (total_distance : ℝ) (days : ℕ) (first_day_factor : ℝ) (halve_factor : ℝ) (third_day_distance : ℝ) :
  total_distance = 378 ∧ days = 6 ∧ first_day_factor = 4 ∧ halve_factor = 0.5 →
  third_day_distance = 48 := sorry

end distance_third_day_l46_46663


namespace four_digit_multiples_of_5_count_l46_46314

theorem four_digit_multiples_of_5_count :
  let lower_bound := 200
  let upper_bound := 1999
  (upper_bound - lower_bound + 1) = 1800 :=
by
  let lower_bound := 200
  let upper_bound := 1999
  show (upper_bound - lower_bound + 1) = 1800,
  from sorry

end four_digit_multiples_of_5_count_l46_46314


namespace find_four_letter_list_with_equal_product_l46_46112

open Nat

theorem find_four_letter_list_with_equal_product :
  ∃ (L T M W : ℕ), 
  (L * T * M * W = 23 * 24 * 25 * 26) 
  ∧ (1 ≤ L ∧ L ≤ 26) ∧ (1 ≤ T ∧ T ≤ 26) ∧ (1 ≤ M ∧ M ≤ 26) ∧ (1 ≤ W ∧ W ≤ 26) 
  ∧ (L ≠ T) ∧ (T ≠ M) ∧ (M ≠ W) ∧ (W ≠ L) ∧ (L ≠ M) ∧ (T ≠ W)
  ∧ (L * T * M * W) = (12 * 20 * 13 * 23) :=
by
  sorry

end find_four_letter_list_with_equal_product_l46_46112


namespace percentage_spent_l46_46712

theorem percentage_spent (initial_amount remaining_amount : ℝ) 
  (h_initial : initial_amount = 1200) 
  (h_remaining : remaining_amount = 840) : 
  (initial_amount - remaining_amount) / initial_amount * 100 = 30 :=
by
  sorry

end percentage_spent_l46_46712


namespace length_of_boat_l46_46346

-- Define Josie's jogging variables and problem conditions
variables (L J B : ℝ)
axiom eqn1 : 130 * J = L + 130 * B
axiom eqn2 : 70 * J = L - 70 * B

-- The theorem to prove that the length of the boat L equals 91 steps (i.e., 91 * J)
theorem length_of_boat : L = 91 * J :=
by
  sorry

end length_of_boat_l46_46346


namespace find_a_l46_46823

noncomputable def f (a x : ℝ) : ℝ := (1 / 2) * x^2 + 2 * a * x - Real.log x

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
∀ x y, a ≤ x → x ≤ b → a ≤ y → y ≤ b → x ≤ y → f x ≤ f y

theorem find_a (a : ℝ) :
  is_increasing_on (f a) (1 / 3) 2 → a ≥ 4 / 3 :=
sorry

end find_a_l46_46823


namespace tiffany_found_bags_l46_46707

theorem tiffany_found_bags (initial_bags : ℕ) (total_bags : ℕ) (found_bags : ℕ) :
  initial_bags = 4 ∧ total_bags = 12 ∧ total_bags = initial_bags + found_bags → found_bags = 8 :=
by
  sorry

end tiffany_found_bags_l46_46707


namespace ratio_time_B_to_A_l46_46427

-- Definitions for the given conditions
def T_A : ℕ := 10
def work_rate_A : ℚ := 1 / T_A
def combined_work_rate : ℚ := 0.3

-- Lean 4 statement for the problem
theorem ratio_time_B_to_A (T_B : ℚ) (h : (work_rate_A + 1 / T_B) = combined_work_rate) :
  (T_B / T_A) = (1 / 2) := by
  sorry

end ratio_time_B_to_A_l46_46427


namespace factor_1_factor_2_triangle_is_isosceles_l46_46183

-- Factorization problems
theorem factor_1 (x y : ℝ) : 
  (x^2 - x * y + 4 * x - 4 * y) = ((x - y) * (x + 4)) :=
sorry

theorem factor_2 (x y : ℝ) : 
  (x^2 - y^2 + 4 * y - 4) = ((x + y - 2) * (x - y + 2)) :=
sorry

-- Triangle shape problem
theorem triangle_is_isosceles (a b c : ℝ) (h : a^2 - a * c - b^2 + b * c = 0) : 
  a = b ∨ a = c ∨ b = c :=
sorry

end factor_1_factor_2_triangle_is_isosceles_l46_46183


namespace part_a_part_b_l46_46678

-- Define the natural numbers m and n
variable (m n : Nat)

-- Condition: m * n is divisible by m + n
def divisible_condition : Prop :=
  ∃ (k : Nat), m * n = k * (m + n)

-- Define prime number
def is_prime (p : Nat) : Prop :=
  p > 1 ∧ ∀ d : Nat, d ∣ p → d = 1 ∨ d = p

-- Define n as the product of two distinct primes
def is_product_of_two_distinct_primes (n : Nat) : Prop :=
  ∃ (p₁ p₂ : Nat), is_prime p₁ ∧ is_prime p₂ ∧ p₁ ≠ p₂ ∧ n = p₁ * p₂

-- Problem (a): Prove that m is divisible by n when n is a prime number and m * n is divisible by m + n
theorem part_a (prime_n : is_prime n) (h : divisible_condition m n) : n ∣ m := sorry

-- Problem (b): Prove that m is not necessarily divisible by n when n is a product of two distinct prime numbers
theorem part_b (prod_of_primes_n : is_product_of_two_distinct_primes n) (h : divisible_condition m n) :
  ¬ (n ∣ m) := sorry

end part_a_part_b_l46_46678


namespace set_difference_equality_l46_46360

theorem set_difference_equality :
  let A := {x : ℝ | x < 4}
  let B := {x : ℝ | x^2 - 4 * x + 3 > 0}
  let S := {x : ℝ | x ∈ A ∧ x ∉ (set.Inter {x ∈ A | x ∈ B})}
  S = {x : ℝ | 1 ≤ x ∧ x ≤ 3} :=
by {
  let A := {x : ℝ | x < 4},
  let B := {x : ℝ | x^2 - 4 * x + 3 > 0},
  let S := {x : ℝ | x ∈ A ∧ x ∉ (A ∩ B)},
  sorry
}

end set_difference_equality_l46_46360


namespace total_letters_sent_l46_46522

def letters_January : ℕ := 6
def letters_February : ℕ := 9
def letters_March : ℕ := 3 * letters_January

theorem total_letters_sent : letters_January + letters_February + letters_March = 33 :=
by
  -- This is where the proof would go
  sorry

end total_letters_sent_l46_46522


namespace B_is_not_15_percent_less_than_A_l46_46980

noncomputable def A (B : ℝ) : ℝ := 1.15 * B

theorem B_is_not_15_percent_less_than_A (B : ℝ) (h : B > 0) : A B ≠ 0.85 * (A B) :=
by
  unfold A
  suffices 1.15 * B ≠ 0.85 * (1.15 * B) by
    intro h1
    exact this h1
  sorry

end B_is_not_15_percent_less_than_A_l46_46980


namespace reciprocal_of_neg_one_div_2023_l46_46037

theorem reciprocal_of_neg_one_div_2023 : 1 / (-1 / (2023 : ℤ)) = -2023 := sorry

end reciprocal_of_neg_one_div_2023_l46_46037


namespace board_partition_possible_l46_46424

-- Definition of natural numbers m and n greater than 15
variables (m n : ℕ)
-- m > 15
def m_greater_than_15 := m > 15
-- n > 15
def n_greater_than_15 := n > 15

-- Definition of m and n divisibility conditions
def divisible_by_4_or_5 (x : ℕ) : Prop :=
  x % 4 = 0 ∨ x % 5 = 0

def partition_possible (m n : ℕ) : Prop :=
  (m % 4 = 0 ∧ n % 5 = 0) ∨ (m % 5 = 0 ∧ n % 4 = 0)

-- The final statement of Lean
theorem board_partition_possible :
  m_greater_than_15 m → n_greater_than_15 n → partition_possible m n :=
by
  intro h_m h_n
  sorry

end board_partition_possible_l46_46424


namespace always_two_real_roots_find_m_l46_46298

-- Given quadratic equation: x^2 - 4mx + 3m^2 = 0
-- Definitions for the problem
def quadratic_eq (m x : ℝ) : Prop := x^2 - 4 * m * x + 3 * m^2 = 0

-- Q1: Prove that this equation always has two real roots.
theorem always_two_real_roots (m : ℝ) : ∃ x₁ x₂ : ℝ, quadratic_eq m x₁ ∧ quadratic_eq m x₂ :=
by
  sorry

-- Q2: If m > 0 and the difference between the two real roots is 2, find the value of m.
theorem find_m (m : ℝ) (h₁ : m > 0) (h₂ : ∃ x₁ x₂ : ℝ, quadratic_eq m x₁ ∧ quadratic_eq m x₂ ∧ |x₁ - x₂| = 2) : m = 1 :=
by
  sorry

end always_two_real_roots_find_m_l46_46298


namespace probability_at_least_6_heads_8_flips_l46_46738

-- Define the probability calculation of getting at least 6 heads in 8 coin flips.
def probability_at_least_6_heads (n : ℕ) (k : ℕ) : ℚ :=
  (Nat.choose n k + Nat.choose n (k + 1) + Nat.choose n (k + 2)) / 2^n

theorem probability_at_least_6_heads_8_flips : 
  probability_at_least_6_heads 8 6 = 37 / 256 := 
by
  sorry

end probability_at_least_6_heads_8_flips_l46_46738


namespace board_partition_possible_l46_46425

variable (m n : ℕ)

theorem board_partition_possible (hm : m > 15) (hn : n > 15) :
  ((∃ k1, m = 5 * k1 ∧ ∃ k2, n = 4 * k2) ∨ (∃ k3, m = 4 * k3 ∧ ∃ k4, n = 5 * k4)) :=
sorry

end board_partition_possible_l46_46425


namespace dave_initial_video_games_l46_46599

theorem dave_initial_video_games (non_working_games working_game_price total_earnings : ℕ) 
  (h1 : non_working_games = 2) 
  (h2 : working_game_price = 4) 
  (h3 : total_earnings = 32) : 
  non_working_games + total_earnings / working_game_price = 10 := 
by 
  sorry

end dave_initial_video_games_l46_46599


namespace find_other_number_l46_46874

theorem find_other_number (HCF LCM num1 num2 : ℕ) (h1 : HCF = 16) (h2 : LCM = 396) (h3 : num1 = 36) (h4 : HCF * LCM = num1 * num2) : num2 = 176 :=
sorry

end find_other_number_l46_46874


namespace part_a_l46_46367

theorem part_a : (2^41 + 1) % 83 = 0 :=
  sorry

end part_a_l46_46367


namespace machine_C_works_in_6_hours_l46_46362

theorem machine_C_works_in_6_hours :
  ∃ C : ℝ, (0 < C ∧ (1/4 + 1/12 + 1/C = 1/2)) → C = 6 :=
by
  sorry

end machine_C_works_in_6_hours_l46_46362


namespace distance_from_origin_to_point_on_parabola_l46_46620

theorem distance_from_origin_to_point_on_parabola
  (y x : ℝ)
  (focus : ℝ × ℝ := (4, 0))
  (on_parabola : y^2 = 8 * x)
  (distance_to_focus : Real.sqrt ((x - 4)^2 + y^2) = 4) :
  Real.sqrt (x^2 + y^2) = 2 * Real.sqrt 5 :=
by
  sorry

end distance_from_origin_to_point_on_parabola_l46_46620


namespace polygon_diagonalization_l46_46181

theorem polygon_diagonalization (n : ℕ) (h : n ≥ 3) : 
  ∃ (triangles : ℕ), triangles = n - 2 ∧ 
  (∀ (polygons : ℕ), 3 ≤ polygons → polygons < n → ∃ k, k = polygons - 2) := 
by {
  -- base case
  sorry
}

end polygon_diagonalization_l46_46181


namespace circle_area_eq_pi_div_4_l46_46344

theorem circle_area_eq_pi_div_4 :
  ∀ (x y : ℝ), 3*x^2 + 3*y^2 - 9*x + 12*y + 27 = 0 -> (π * (1 / 2)^2 = π / 4) :=
by
  sorry

end circle_area_eq_pi_div_4_l46_46344


namespace intersection_A_B_l46_46826

/-- Define the set A -/
def A : Set ℝ := { x | ∃ y, y = Real.log (2 - x) }

/-- Define the set B -/
def B : Set ℝ := { y | ∃ x, y = Real.sqrt x }

/-- Define the intersection of A and B and prove that it equals [0, 2) -/
theorem intersection_A_B : (A ∩ B) = { x | 0 ≤ x ∧ x < 2 } :=
by
  sorry

end intersection_A_B_l46_46826


namespace prob_not_has_bio_test_combined_prob_neither_bio_nor_chem_l46_46538

def prob_has_bio_test : ℚ := 5 / 8
def prob_not_has_chem_test : ℚ := 1 / 2

theorem prob_not_has_bio_test : 1 - 5 / 8 = 3 / 8 := by
  sorry

theorem combined_prob_neither_bio_nor_chem :
  (1 - 5 / 8) * (1 / 2) = 3 / 16 := by
  sorry

end prob_not_has_bio_test_combined_prob_neither_bio_nor_chem_l46_46538


namespace anthony_transactions_more_percentage_l46_46679

def transactions (Mabel Anthony Cal Jade : ℕ) : Prop := 
  Mabel = 90 ∧ 
  Jade = 84 ∧ 
  Jade = Cal + 18 ∧ 
  Cal = (2 * Anthony) / 3 ∧ 
  Anthony = Mabel + (Mabel * 10 / 100)

theorem anthony_transactions_more_percentage (Mabel Anthony Cal Jade : ℕ) 
    (h : transactions Mabel Anthony Cal Jade) : 
  (Anthony = Mabel + (Mabel * 10 / 100)) :=
by 
  sorry

end anthony_transactions_more_percentage_l46_46679


namespace chocolate_more_expensive_l46_46569

variables (C P : ℝ)
theorem chocolate_more_expensive (h : 7 * C > 8 * P) : 8 * C > 9 * P :=
sorry

end chocolate_more_expensive_l46_46569


namespace no_intersection_points_l46_46324

theorem no_intersection_points :
  ∀ x y : ℝ, y = abs (3 * x + 6) ∧ y = -2 * abs (2 * x - 1) → false :=
by
  intros x y h
  cases h
  sorry

end no_intersection_points_l46_46324


namespace probability_of_at_least_six_heads_is_correct_l46_46764

-- Definitions for the given problem
def binomial_coefficient (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

def total_possible_outcomes : ℕ :=
  2^8

def favorable_outcomes : ℕ :=
  binomial_coefficient 8 6 + binomial_coefficient 8 7 + binomial_coefficient 8 8

def probability_of_at_least_6_heads : ℚ :=
  favorable_outcomes / total_possible_outcomes

-- The proof statement
theorem probability_of_at_least_six_heads_is_correct :
  probability_of_at_least_6_heads = 37 / 256 :=
by sorry

end probability_of_at_least_six_heads_is_correct_l46_46764


namespace anniversary_day_probability_l46_46103

/- Definitions based on the conditions -/
def is_leap_year (year : ℕ) : Prop :=
  year % 4 = 0

def days_in_year (year : ℕ) : ℕ :=
  if is_leap_year year then 366 else 365

def total_days (start_year : ℕ) : ℕ :=
  (list.sum (list.map days_in_year (list.range' start_year 11)))

/- Prove the day of the 11th anniversary and its probabilities -/
theorem anniversary_day_probability (start_year : ℕ) (h : start_year ∈ {1668, 1669, 1670, 1671}) :
  let days := total_days start_year % 7
  in (days = 0 ∧ 0.75 ≤ 1) ∨ (days = 6 ∧ 0.25 ≤ 1) :=
by
  sorry

end anniversary_day_probability_l46_46103


namespace ratio_of_ages_in_two_years_l46_46433

-- Define the constants
def son_age : ℕ := 24
def age_difference : ℕ := 26

-- Define the equations based on conditions
def man_age := son_age + age_difference
def son_future_age := son_age + 2
def man_future_age := man_age + 2

-- State the theorem for the required ratio
theorem ratio_of_ages_in_two_years : man_future_age / son_future_age = 2 := by
  sorry

end ratio_of_ages_in_two_years_l46_46433


namespace four_digit_multiples_of_5_count_l46_46310

-- Define the range of four-digit numbers
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

-- Define a multiple of 5
def is_multiple_of_five (n : ℕ) : Prop := n % 5 = 0

-- Define the required proof problem
theorem four_digit_multiples_of_5_count : 
  ∃ n : ℕ, (n = 1800) ∧ (∀ k : ℕ, is_four_digit k ∧ is_multiple_of_five k → n = 1800) :=
sorry

end four_digit_multiples_of_5_count_l46_46310


namespace tina_first_hour_coins_l46_46708

variable (X : ℕ)

theorem tina_first_hour_coins :
  let first_hour_coins := X
  let second_third_hour_coins := 30 + 30
  let fourth_hour_coins := 40
  let fifth_hour_removed_coins := 20
  let total_coins := first_hour_coins + second_third_hour_coins + fourth_hour_coins - fifth_hour_removed_coins
  total_coins = 100 → X = 20 :=
by
  intro h
  sorry

end tina_first_hour_coins_l46_46708


namespace range_of_a_l46_46841

theorem range_of_a (a : ℝ) : (∀ (x : ℝ), (x^2 - 2*a*x + a) > 0) → (a ≤ 0 ∨ a ≥ 1) :=
by
  -- Proof goes here
  sorry

end range_of_a_l46_46841


namespace find_f_neg_a_l46_46277

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin x - 4 * Real.tan x + 1

theorem find_f_neg_a (a : ℝ) (h : f a = 2) : f (-a) = 0 :=
by
  sorry

end find_f_neg_a_l46_46277


namespace num_ways_to_assign_grades_l46_46088

-- Define the number of students
def num_students : ℕ := 12

-- Define the number of grades available to each student
def num_grades : ℕ := 4

-- The theorem stating that the total number of ways to assign grades is 4^12
theorem num_ways_to_assign_grades : num_grades ^ num_students = 16777216 := by
  sorry

end num_ways_to_assign_grades_l46_46088


namespace water_to_concentrate_ratio_l46_46098

theorem water_to_concentrate_ratio (servings : ℕ) (serving_size_oz concentrate_size_oz : ℕ)
                                (cans_of_concentrate required_juice_oz : ℕ)
                                (h_servings : servings = 280)
                                (h_serving_size : serving_size_oz = 6)
                                (h_concentrate_size : concentrate_size_oz = 12)
                                (h_cans_of_concentrate : cans_of_concentrate = 35)
                                (h_required_juice : required_juice_oz = servings * serving_size_oz)
                                (h_made_juice : required_juice_oz = 1680)
                                (h_concentrate_volume : cans_of_concentrate * concentrate_size_oz = 420)
                                (h_water_volume : required_juice_oz - (cans_of_concentrate * concentrate_size_oz) = 1260)
                                (h_water_cans : 1260 / concentrate_size_oz = 105) :
                                105 / 35 = 3 :=
by
  sorry

end water_to_concentrate_ratio_l46_46098


namespace find_P_l46_46689

theorem find_P 
  (digits : Finset ℕ) 
  (h_digits : digits = {1, 2, 3, 4, 5, 6}) 
  (P Q R S T U : ℕ)
  (h_unique : P ∈ digits ∧ Q ∈ digits ∧ R ∈ digits ∧ S ∈ digits ∧ T ∈ digits ∧ U ∈ digits ∧ 
              P ≠ Q ∧ P ≠ R ∧ P ≠ S ∧ P ≠ T ∧ P ≠ U ∧
              Q ≠ R ∧ Q ≠ S ∧ Q ≠ T ∧ Q ≠ U ∧
              R ≠ S ∧ R ≠ T ∧ R ≠ U ∧ 
              S ≠ T ∧ S ≠ U ∧ 
              T ≠ U) 
  (h_div5 : (100 * P + 10 * Q + R) % 5 = 0)
  (h_div3 : (100 * Q + 10 * R + S) % 3 = 0)
  (h_div2 : (100 * R + 10 * S + T) % 2 = 0) :
  P = 2 :=
sorry

end find_P_l46_46689


namespace long_show_episodes_correct_l46_46208

variable {short_show_episodes : ℕ} {short_show_duration : ℕ} {total_watched_time : ℕ} {long_show_episode_duration : ℕ}

def episodes_long_show (short_episodes_duration total_duration long_episode_duration : ℕ) : ℕ :=
  (total_duration - short_episodes_duration) / long_episode_duration

theorem long_show_episodes_correct :
  ∀ (short_show_episodes short_show_duration total_watched_time long_show_episode_duration : ℕ),
  short_show_episodes = 24 →
  short_show_duration = 1 / 2 →
  total_watched_time = 24 →
  long_show_episode_duration = 1 →
  episodes_long_show (short_show_episodes * short_show_duration) total_watched_time long_show_episode_duration = 12 := by
  intros
  sorry

end long_show_episodes_correct_l46_46208


namespace ratio_bones_child_to_adult_woman_l46_46163

noncomputable def num_skeletons : ℕ := 20
noncomputable def num_adult_women : ℕ := num_skeletons / 2
noncomputable def num_adult_men_and_children : ℕ := num_skeletons - num_adult_women
noncomputable def num_adult_men : ℕ := num_adult_men_and_children / 2
noncomputable def num_children : ℕ := num_adult_men_and_children / 2
noncomputable def bones_per_adult_woman : ℕ := 20
noncomputable def bones_per_adult_man : ℕ := bones_per_adult_woman + 5
noncomputable def total_bones : ℕ := 375
noncomputable def bones_per_child : ℕ := (total_bones - (num_adult_women * bones_per_adult_woman + num_adult_men * bones_per_adult_man)) / num_children

theorem ratio_bones_child_to_adult_woman : 
  (bones_per_child : ℚ) / (bones_per_adult_woman : ℚ) = 1 / 2 := by
sorry

end ratio_bones_child_to_adult_woman_l46_46163


namespace none_satisfied_l46_46285

-- Define the conditions
variables {a b c x y z : ℝ}
  
-- Theorem that states that none of the given inequalities are satisfied strictly
theorem none_satisfied (h1 : x ≤ a) (h2 : y ≤ b) (h3 : z ≤ c) :
  ¬(x^2 * y + y^2 * z + z^2 * x < a^2 * b + b^2 * c + c^2 * a) ∧
  ¬(x^3 + y^3 + z^3 < a^3 + b^3 + c^3) :=
  by
    sorry

end none_satisfied_l46_46285


namespace value_of_m_l46_46567

theorem value_of_m (m : ℕ) : (5^m = 5 * 25^2 * 125^3) → m = 14 :=
by
  sorry

end value_of_m_l46_46567


namespace sum_of_consecutive_integers_l46_46646

theorem sum_of_consecutive_integers (a b : ℤ) (h1 : a + 1 = b) (h2 : a < real.sqrt 3) (h3 : real.sqrt 3 < b) : a + b = 3 :=
sorry

end sum_of_consecutive_integers_l46_46646


namespace stock_price_is_500_l46_46485

-- Conditions
def income : ℝ := 1000
def dividend_rate : ℝ := 0.50
def investment : ℝ := 10000
def face_value : ℝ := 100

-- Theorem Statement
theorem stock_price_is_500 : 
  (dividend_rate * face_value / (investment / 1000)) = 500 := by
  sorry

end stock_price_is_500_l46_46485


namespace cantaloupe_total_l46_46273

theorem cantaloupe_total (Fred Tim Alicia : ℝ) 
  (hFred : Fred = 38.5) 
  (hTim : Tim = 44.2)
  (hAlicia : Alicia = 29.7) : 
  Fred + Tim + Alicia = 112.4 :=
by
  sorry

end cantaloupe_total_l46_46273


namespace trigonometric_relation_for_given_sum_l46_46228

theorem trigonometric_relation_for_given_sum (a : ℕ → ℕ) (h : ∀ i j, i ≠ j → a i ≠ a j)
  (sum_eq : (∑ i in finset.range 7, 2 ^ (a i)) = 2008):
  let m := ∑ i in finset.range 7, a i in
  sin (m : ℝ) > tan (m : ℝ) ∧ tan (m : ℝ) > cos (m : ℝ) :=
by
  -- We can assume the values based on the binary representation of 2008
  have a_values : (a 0 = 10 ∧ a 1 = 9 ∧ a 2 = 8 ∧ a 3 = 7 ∧ a 4 = 6 ∧ a 5 = 4 ∧ a 6 = 3) := sorry,
  let m := 10 + 9 + 8 + 7 + 6 + 4 + 3,
  have m_val : m = 47 := by norm_num,
  have sin_val := real.sin_pos_of_pos_of_lt_pi (by norm_num : 0 < 47) (by norm_num : 47 < 3.14 * 15),
  have cos_val := real.cos_neg_of_pi_div_two_lt_of_lt (by norm_num : 3.14 * 14.5 < 47) (by norm_num : 47 < 3.14 * 14.7),
  have tan_val := real.tan_pos_of_div_two_pi_lt_of_lt (by norm_num : 3.14 * 14 + 5 * 3.14/6 < 47) (by norm_num : 47 < 3.14 *15),
  exact ⟨sin_val, tan_val, cos_val⟩

end trigonometric_relation_for_given_sum_l46_46228


namespace triangle_QR_length_l46_46073

/-- Conditions for the triangles PQR and SQR sharing a side QR with given side lengths. -/
structure TriangleSetup where
  (PQ PR SR SQ QR : ℝ)
  (PQ_pos : PQ > 0)
  (PR_pos : PR > 0)
  (SR_pos : SR > 0)
  (SQ_pos : SQ > 0)
  (shared_side_QR : QR = QR)

/-- The problem statement asserting the least possible length of QR. -/
theorem triangle_QR_length (t : TriangleSetup) 
  (h1 : t.PQ = 8)
  (h2 : t.PR = 15)
  (h3 : t.SR = 10)
  (h4 : t.SQ = 25) :
  t.QR = 15 :=
by
  sorry

end triangle_QR_length_l46_46073


namespace probability_at_least_6_heads_in_8_flips_l46_46769

open scoped BigOperators

def binom (n k : ℕ) : ℕ := nat.choose n k

def total_outcomes (n : ℕ) := 2^n

def successful_outcomes (n k : ℕ) :=
  (finset.range (n + 1)).filter (λ x, x ≥ k).sum (λ x, binom n x)

def probability (n k : ℕ) :=
  (successful_outcomes n k) / (total_outcomes n : ℚ)

theorem probability_at_least_6_heads_in_8_flips :
  probability 8 6 = 37 / 256 := sorry

end probability_at_least_6_heads_in_8_flips_l46_46769


namespace reciprocal_is_correct_l46_46050

-- Define the initial number
def num : ℚ := -1 / 2023

-- Define the expected reciprocal
def reciprocal : ℚ := -2023

-- Theorem stating the reciprocal of the given number is the expected reciprocal
theorem reciprocal_is_correct : 1 / num = reciprocal :=
  by
    -- The actual proof can be filled in here
    sorry

end reciprocal_is_correct_l46_46050


namespace johnny_age_multiple_l46_46558

theorem johnny_age_multiple
  (current_age : ℕ)
  (age_in_2_years : ℕ)
  (age_3_years_ago : ℕ)
  (k : ℕ)
  (h1 : current_age = 8)
  (h2 : age_in_2_years = current_age + 2)
  (h3 : age_3_years_ago = current_age - 3)
  (h4 : age_in_2_years = k * age_3_years_ago) :
  k = 2 :=
by
  sorry

end johnny_age_multiple_l46_46558


namespace stacy_history_paper_pages_l46_46526

def stacy_paper := 1 -- Number of pages Stacy writes per day
def days_to_finish := 12 -- Number of days Stacy has to finish the paper

theorem stacy_history_paper_pages : stacy_paper * days_to_finish = 12 := by
  sorry

end stacy_history_paper_pages_l46_46526


namespace inscribed_circle_radius_l46_46990

theorem inscribed_circle_radius 
  (A : ℝ) -- Area of the triangle
  (p : ℝ) -- Perimeter of the triangle
  (r : ℝ) -- Radius of the inscribed circle
  (s : ℝ) -- Semiperimeter of the triangle
  (h1 : A = 2 * p) -- Condition: Area is numerically equal to twice the perimeter
  (h2 : p = 2 * s) -- Perimeter is twice the semiperimeter
  (h3 : A = r * s) -- Formula: Area in terms of inradius and semiperimeter
  (h4 : s ≠ 0) -- Semiperimeter is non-zero
  : r = 4 := 
sorry

end inscribed_circle_radius_l46_46990


namespace ned_trips_l46_46012

theorem ned_trips : 
  ∀ (carry_capacity : ℕ) (table1 : ℕ) (table2 : ℕ) (table3 : ℕ) (table4 : ℕ),
  carry_capacity = 5 →
  table1 = 7 →
  table2 = 10 →
  table3 = 12 →
  table4 = 3 →
  (table1 + table2 + table3 + table4 + carry_capacity - 1) / carry_capacity = 8 :=
by
  intro carry_capacity table1 table2 table3 table4
  intro h1 h2 h3 h4 h5
  sorry

end ned_trips_l46_46012


namespace sum_u_t_values_l46_46287

open BigOperators

def t : Fin 4 → Fin 4
| ⟨0, _⟩ => ⟨1, by decide⟩
| ⟨1, _⟩ => ⟨3, by decide⟩
| ⟨2, _⟩ => ⟨5, by decide⟩
| ⟨3, _⟩ => ⟨7, by decide⟩

def u : Fin 5 → Fin 5
| ⟨x, h⟩ => ⟨x - 1, by linarith [h]⟩

theorem sum_u_t_values : (∑ x in (Finset.filter (λ x => x.val ∈ {3, 5}) (Finset.image t Finset.univ)), u x).val = 6 := 
sorry

end sum_u_t_values_l46_46287


namespace inequality_holds_for_all_real_l46_46974

theorem inequality_holds_for_all_real (k : ℝ) :
  (∀ x : ℝ, k * x ^ 2 - 6 * k * x + k + 8 ≥ 0) ↔ (0 ≤ k ∧ k ≤ 1) :=
sorry

end inequality_holds_for_all_real_l46_46974


namespace total_area_of_rug_l46_46917

theorem total_area_of_rug :
  let length_rect := 6
  let width_rect := 4
  let base_parallelogram := 3
  let height_parallelogram := 4
  let area_rect := length_rect * width_rect
  let area_parallelogram := base_parallelogram * height_parallelogram
  let total_area := area_rect + 2 * area_parallelogram
  total_area = 48 := by sorry

end total_area_of_rug_l46_46917


namespace range_of_k_l46_46968

theorem range_of_k (k : ℝ) (h : k ≠ 0) : (k^2 - 6 * k + 8 ≥ 0) ↔ (k ≥ 4 ∨ k ≤ 2) := 
by sorry

end range_of_k_l46_46968


namespace jessica_cut_r_l46_46705

variable (r_i r_t r_c : ℕ)

theorem jessica_cut_r : r_i = 7 → r_g = 59 → r_t = 20 → r_c = r_t - r_i → r_c = 13 :=
by
  intros h_i h_g h_t h_c
  have h1 : r_i = 7 := h_i
  have h2 : r_t = 20 := h_t
  have h3 : r_c = r_t - r_i := h_c
  have h_correct : r_c = 13
  · sorry
  exact h_correct

end jessica_cut_r_l46_46705


namespace unoccupied_volume_of_tank_l46_46382

theorem unoccupied_volume_of_tank (length width height : ℝ) (num_marbles : ℕ) (marble_radius : ℝ) (fill_fraction : ℝ) :
    length = 12 → width = 12 → height = 15 → num_marbles = 5 → marble_radius = 1.5 → fill_fraction = 1/3 →
    (length * width * height * (1 - fill_fraction) - num_marbles * (4 / 3 * Real.pi * marble_radius^3) = 1440 - 22.5 * Real.pi) :=
by
  intros
  sorry

end unoccupied_volume_of_tank_l46_46382


namespace range_of_a_l46_46252

def tensor (x y : ℝ) : ℝ := x * (1 - y)

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → tensor (x - a) (x + a) < 2) → -1 < a ∧ a < 2 := by
  sorry

end range_of_a_l46_46252


namespace solve_m_l46_46828

def f (x : ℝ) := 4 * x ^ 2 - 3 * x + 5
def g (x : ℝ) (m : ℝ) := x ^ 2 - m * x - 8

theorem solve_m : ∃ (m : ℝ), f 8 - g 8 m = 20 ∧ m = -25.5 := by
  sorry

end solve_m_l46_46828


namespace halfway_fraction_l46_46217

theorem halfway_fraction (a b : ℚ) (ha : a = 3/4) (hb : b = 5/6) : (a + b) / 2 = 19/24 :=
by
  rw [ha, hb] -- replace a and b with 3/4 and 5/6 respectively
  have h1 : 3/4 + 5/6 = 19/12,
  { norm_num, -- ensures 3/4 + 5/6 = 19/12
    linarith },
  rw h1, -- replace a + b with 19/12
  norm_num -- ensures (19/12) / 2 = 19/24

end halfway_fraction_l46_46217


namespace range_of_a_l46_46334

theorem range_of_a (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) : 1 < a ∧ a < 2 :=
by
  -- Insert the proof here
  sorry

end range_of_a_l46_46334


namespace fixed_costs_16699_50_l46_46572

noncomputable def fixed_monthly_costs (production_cost shipping_cost units_sold price_per_unit : ℝ) : ℝ :=
  let total_variable_cost := (production_cost + shipping_cost) * units_sold
  let total_revenue := price_per_unit * units_sold
  total_revenue - total_variable_cost

theorem fixed_costs_16699_50 :
  fixed_monthly_costs 80 7 150 198.33 = 16699.5 :=
by
  sorry

end fixed_costs_16699_50_l46_46572


namespace binomial_12_9_l46_46789

def binomial (n k : ℕ) := nat.choose n k

theorem binomial_12_9 : binomial 12 9 = 220 :=
by
  have step1 : binomial 12 9 = binomial 12 3 := nat.choose_symm 12 9
  have step2 : binomial 12 3 = 220 := by sorry
  rw [step1, step2]

end binomial_12_9_l46_46789


namespace prob_at_least_6_heads_eq_l46_46751

-- define the number of coin flips
def n := 8

-- define the number of possible outcomes (2^n)
def total_outcomes := 2 ^ n

-- define the binomial coefficients for cases: 6 heads, 7 heads, 8 heads
def binom_8_6 := Nat.choose 8 6
def binom_8_7 := Nat.choose 8 7
def binom_8_8 := Nat.choose 8 8

-- calculate the favorable outcomes for at least 6 heads
def favorable_outcomes := binom_8_6 + binom_8_7 + binom_8_8

-- define the probability of getting at least 6 heads
def probability := (favorable_outcomes : ℚ) / total_outcomes

theorem prob_at_least_6_heads_eq : probability = 37 / 256 := by
  sorry

end prob_at_least_6_heads_eq_l46_46751


namespace odd_number_as_difference_of_squares_l46_46167

theorem odd_number_as_difference_of_squares (n : ℤ) (h : ∃ k : ℤ, n = 2 * k + 1) :
  ∃ a b : ℤ, n = a^2 - b^2 :=
by
  sorry

end odd_number_as_difference_of_squares_l46_46167


namespace average_weight_increase_l46_46028

variable {W : ℝ} -- Total weight before replacement
variable {n : ℝ} -- Number of men in the group

theorem average_weight_increase
  (h1 : (W - 58 + 83) / n - W / n = 2.5) : n = 10 :=
by
  sorry

end average_weight_increase_l46_46028


namespace cube_painting_probability_l46_46210

theorem cube_painting_probability :
  let total_configurations := 2^6 * 2^6
  let identical_configurations := 90
  (identical_configurations / total_configurations : ℚ) = 45 / 2048 :=
by
  sorry

end cube_painting_probability_l46_46210


namespace rectangle_area_at_stage_8_l46_46329

-- Definitions based on conditions
def area_of_square (side_length : ℕ) : ℕ := side_length * side_length
def number_of_squares_in_stage (stage : ℕ) : ℕ := stage

-- The main theorem to prove
theorem rectangle_area_at_stage_8 : 
  area_of_square 4 * number_of_squares_in_stage 8 = 128 := by
  sorry

end rectangle_area_at_stage_8_l46_46329


namespace interval_of_monotonic_increase_l46_46139

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem interval_of_monotonic_increase : {x : ℝ | -1 ≤ x} ⊆ {x : ℝ | 0 < deriv f x} :=
by
  sorry

end interval_of_monotonic_increase_l46_46139


namespace man_born_in_1892_l46_46234

-- Define the conditions and question
def man_birth_year (x : ℕ) : ℕ :=
x^2 - x

-- Conditions:
variable (x : ℕ)
-- 1. The man was born in the first half of the 20th century
variable (h1 : man_birth_year x < 1950)
-- 2. The man's age x and the conditions in the problem
variable (h2 : x^2 - x < 1950)

-- The statement we aim to prove
theorem man_born_in_1892 (x : ℕ) (h1 : man_birth_year x < 1950) (h2 : x = 44) : man_birth_year x = 1892 := by
  sorry

end man_born_in_1892_l46_46234


namespace probability_at_least_6_heads_8_flips_l46_46741

-- Define the probability calculation of getting at least 6 heads in 8 coin flips.
def probability_at_least_6_heads (n : ℕ) (k : ℕ) : ℚ :=
  (Nat.choose n k + Nat.choose n (k + 1) + Nat.choose n (k + 2)) / 2^n

theorem probability_at_least_6_heads_8_flips : 
  probability_at_least_6_heads 8 6 = 37 / 256 := 
by
  sorry

end probability_at_least_6_heads_8_flips_l46_46741


namespace product_of_solutions_of_quadratic_l46_46556

theorem product_of_solutions_of_quadratic :
  ∀ (x p q : ℝ), 36 - 9 * x - x^2 = 0 ∧ (x = p ∨ x = q) → p * q = -36 :=
by sorry

end product_of_solutions_of_quadratic_l46_46556


namespace sin_cos_value_l46_46832

variable (x : ℝ)

theorem sin_cos_value (h : Real.sin x = 4 * Real.cos x) : Real.sin x * Real.cos x = 4 / 17 := 
sorry

end sin_cos_value_l46_46832


namespace multiples_of_5_in_4_digit_range_l46_46303

theorem multiples_of_5_in_4_digit_range : 
  let count_multiples := (9995 - 1000) / 5 + 1
  in count_multiples = 1800 :=
by
  sorry

end multiples_of_5_in_4_digit_range_l46_46303


namespace midpoint_fraction_l46_46213

theorem midpoint_fraction (a b : ℚ) (h1 : a = 3 / 4) (h2 : b = 5 / 6) :
  (a + b) / 2 = 19 / 24 :=
by {
  sorry
}

end midpoint_fraction_l46_46213


namespace data_set_variance_l46_46107

def data_set : List ℕ := [2, 4, 5, 3, 6]

noncomputable def mean (l : List ℕ) : ℝ :=
  l.sum / l.length

noncomputable def variance (l : List ℕ) : ℝ :=
  let m : ℝ := mean l
  (l.map (fun x => (x - m) ^ 2)).sum / l.length

theorem data_set_variance : variance data_set = 2 := by
  sorry

end data_set_variance_l46_46107


namespace imo_2007_p6_l46_46412

theorem imo_2007_p6 (x y : ℕ) (hx : 0 < x) (hy : 0 < y) :
  ∃ k : ℕ, (x = 11 * k^2) ∧ (y = 11 * k) ↔
  ∃ k : ℕ, (∃ k₁ : ℤ, k₁ = (x^2 * y + x + y) / (x * y^2 + y + 11)) :=
sorry

end imo_2007_p6_l46_46412


namespace fraction_halfway_l46_46220

theorem fraction_halfway (a b c d : ℚ) (h₁ : a = 3) (h₂ : b = 4) (h₃ : c = 5) (h₄ : d = 6) :
  (1 / 2) * ((a / b) + (c / d)) = 19 / 24 := 
by
  sorry

end fraction_halfway_l46_46220


namespace hypotenuse_intersection_incircle_diameter_l46_46255

/-- Let \( a \) and \( b \) be the legs of a right triangle with hypotenuse \( c \). 
    Let two circles be centered at the endpoints of the hypotenuse, with radii \( a \) and \( b \). 
    Prove that the segment of the hypotenuse that lies in the intersection of the two circles is equal in length to the diameter of the incircle of the triangle. -/
theorem hypotenuse_intersection_incircle_diameter (a b : ℝ) :
    let c := Real.sqrt (a^2 + b^2)
    let x := a + b - c
    let r := (a + b - c) / 2
    x = 2 * r :=
by
  let c := Real.sqrt (a^2 + b^2)
  let x := a + b - c
  let r := (a + b - c) / 2
  show x = 2 * r
  sorry

end hypotenuse_intersection_incircle_diameter_l46_46255


namespace find_a_find_m_l46_46971

noncomputable def f (x a : ℝ) : ℝ := Real.exp 1 * x - a * Real.log x

theorem find_a {a : ℝ} (h : ∀ x, f x a = Real.exp 1 - a / x)
  (hx : f (1 / Real.exp 1) a = 0) :
  a = 1 :=
by
  sorry

theorem find_m (a : ℝ) (h_a : a = 1)
  (h_exists : ∃ (x₀ : ℝ), x₀ ∈ Set.Icc (1 / Real.exp 1) (Real.exp 1) 
    ∧ f x₀ a < x₀ + m) :
  1 + Real.log (Real.exp 1 - 1) < m :=
by
  sorry

end find_a_find_m_l46_46971


namespace triangle_inequality_l46_46359

variables {a b c x y z : ℝ}

theorem triangle_inequality 
  (h1 : ∀ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a)
  (h2 : x + y + z = 0) :
  a^2 * y * z + b^2 * z * x + c^2 * x * y ≤ 0 :=
sorry

end triangle_inequality_l46_46359


namespace cos_A_half_triangle_area_sqrt7_bc4_l46_46848

open Real

variable {A B C : ℝ}
variable {a b c : ℝ}
 
-- Condition: In triangle ABC, we have a, b, c as sides opposite angles A, B, and C, with: 
-- equation: c * cos A + a * cos C = 2 * b * cos A
theorem cos_A_half (h : c * cos A + a * cos C = 2 * b * cos A) :
  cos A = 1 / 2 :=
sorry

-- Additional conditions: a = sqrt 7 and b + c = 4
-- Prove the area of triangle ABC
theorem triangle_area_sqrt7_bc4 (h₁ : a = sqrt 7) (h₂ : b + c = 4) (h₃ : c * cos A + a * cos C = 2 * b * cos A) :
  let area := (1 / 2) * b * c * sqrt (1 - cos A^2)
  cos A = 1 / 2 → 
  area = 3 * sqrt 3 / 4 :=
sorry

end cos_A_half_triangle_area_sqrt7_bc4_l46_46848


namespace tangent_of_7pi_over_4_l46_46121

   theorem tangent_of_7pi_over_4 : Real.tan (7 * Real.pi / 4) = -1 := 
   sorry
   
end tangent_of_7pi_over_4_l46_46121


namespace total_value_of_bills_l46_46933

theorem total_value_of_bills 
  (total_bills : Nat := 12) 
  (num_5_dollar_bills : Nat := 4) 
  (num_10_dollar_bills : Nat := 8)
  (value_5_dollar_bill : Nat := 5)
  (value_10_dollar_bill : Nat := 10) :
  (num_5_dollar_bills * value_5_dollar_bill + num_10_dollar_bills * value_10_dollar_bill = 100) :=
by
  sorry

end total_value_of_bills_l46_46933


namespace monotonic_increasing_on_interval_l46_46141

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x^2 - a * Real.log x

theorem monotonic_increasing_on_interval (a : ℝ) :
  (∀ x > 1, 2 * x - a / x ≥ 0) → a ≤ 2 :=
sorry

end monotonic_increasing_on_interval_l46_46141


namespace max_sum_nonneg_l46_46421

theorem max_sum_nonneg (a b c d : ℝ) (h : a + b + c + d = 0) : 
  max a b + max a c + max a d + max b c + max b d + max c d ≥ 0 := 
sorry

end max_sum_nonneg_l46_46421


namespace reciprocal_neg_one_over_2023_eq_neg_2023_l46_46041

theorem reciprocal_neg_one_over_2023_eq_neg_2023 : (1 / (-1 / (2023 : ℝ))) = -2023 :=
by
  sorry

end reciprocal_neg_one_over_2023_eq_neg_2023_l46_46041


namespace quadratic_inequality_l46_46337

theorem quadratic_inequality (a : ℝ) (h : ∀ x : ℝ, a * x^2 + 2 * x + 1 > 0) : a ≥ 1 :=
sorry

end quadratic_inequality_l46_46337


namespace pam_age_l46_46150

-- Given conditions:
-- 1. Pam is currently twice as young as Rena.
-- 2. In 10 years, Rena will be 5 years older than Pam.

variable (Pam Rena : ℕ)

theorem pam_age
  (h1 : 2 * Pam = Rena)
  (h2 : Rena + 10 = Pam + 15)
  : Pam = 5 := 
sorry

end pam_age_l46_46150


namespace v3_value_at_2_l46_46446

def f (x : ℝ) : ℝ :=
  x^6 - 12 * x^5 + 60 * x^4 - 160 * x^3 + 240 * x^2 - 192 * x + 64

def v3 (x : ℝ) : ℝ :=
  ((x - 12) * x + 60) * x - 160

theorem v3_value_at_2 :
  v3 2 = -80 :=
by
  sorry

end v3_value_at_2_l46_46446


namespace count_four_digit_multiples_of_5_l46_46315

theorem count_four_digit_multiples_of_5 : 
  let first_4_digit := 1000
  let last_4_digit := 9999
  let first_multiple_of_5 := 1000
  let last_multiple_of_5 := 9995
  let total_multiples_of_5 := (1999 - 200 + 1)
  first_multiple_of_5 % 5 = 0 ∧ last_multiple_of_5 % 5 = 0 ∧ first_4_digit ≤ first_multiple_of_5 ∧ last_multiple_of_5 ≤ last_4_digit
  → total_multiples_of_5 = 1800 :=
by
  sorry

end count_four_digit_multiples_of_5_l46_46315


namespace relationship_between_a_and_b_l46_46503

noncomputable section
open Classical

theorem relationship_between_a_and_b (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : Real.exp a + 2 * a = Real.exp b + 3 * b) : a > b :=
sorry

end relationship_between_a_and_b_l46_46503


namespace clothes_add_percentage_l46_46068

theorem clothes_add_percentage (W : ℝ) (C : ℝ) (h1 : W > 0) 
  (h2 : C = 0.0174 * W) : 
  ((C / (0.87 * W)) * 100) = 2 :=
by
  sorry

end clothes_add_percentage_l46_46068


namespace solve_diophantine_equation_l46_46373

theorem solve_diophantine_equation :
  ∃ (x y : ℤ), x^2 - x * y - 6 * y^2 + 2 * x + 19 * y = 18 ∧ (x = 2 ∧ y = 2 ∨ x = -2 ∧ y = 2) :=
  sorry

end solve_diophantine_equation_l46_46373


namespace problem_solution_l46_46371

theorem problem_solution (n : Real) (h : 0.04 * n + 0.1 * (30 + n) = 15.2) : n = 89.09 := 
sorry

end problem_solution_l46_46371


namespace sum_of_sequence_l46_46543

variable (S a b : ℝ)

theorem sum_of_sequence :
  (S - a) / 100 = 2022 →
  (S - b) / 100 = 2023 →
  (a + b) / 2 = 51 →
  S = 202301 :=
by
  intros h1 h2 h3
  sorry

end sum_of_sequence_l46_46543


namespace triangle_perimeter_problem_l46_46032

theorem triangle_perimeter_problem : 
  ∀ (c : ℝ), 20 + 15 > c ∧ 20 + c > 15 ∧ 15 + c > 20 → ¬ (35 + c = 72) :=
by
  intros c h
  sorry

end triangle_perimeter_problem_l46_46032


namespace percentage_gain_is_20_percent_l46_46034

theorem percentage_gain_is_20_percent (manufacturing_cost transportation_cost total_shoes selling_price : ℝ)
(h1 : manufacturing_cost = 220)
(h2 : transportation_cost = 500)
(h3 : total_shoes = 100)
(h4 : selling_price = 270) :
  let cost_per_shoe := manufacturing_cost + transportation_cost / total_shoes
  let profit_per_shoe := selling_price - cost_per_shoe
  let percentage_gain := (profit_per_shoe / cost_per_shoe) * 100
  percentage_gain = 20 :=
by
  sorry

end percentage_gain_is_20_percent_l46_46034


namespace practice_minutes_other_days_l46_46241

-- Definitions based on given conditions
def total_hours_practiced : ℕ := 7.5 * 60 -- converting hours to minutes
def minutes_per_day := 86
def days_practiced := 2

-- Lean 4 statement for the proof problem
theorem practice_minutes_other_days :
  let total_minutes := total_hours_practiced
  let minutes_2_days := minutes_per_day * days_practiced
  total_minutes - minutes_2_days = 278 := by
  sorry

end practice_minutes_other_days_l46_46241


namespace julia_money_left_l46_46350

def initial_amount : ℕ := 40

def amount_spent_on_game (initial : ℕ) : ℕ := initial / 2

def amount_left_after_game (initial : ℕ) (spent_game : ℕ) : ℕ := initial - spent_game

def amount_spent_on_in_game (left_after_game : ℕ) : ℕ := left_after_game / 4

def final_amount (left_after_game : ℕ) (spent_in_game : ℕ) : ℕ := left_after_game - spent_in_game

theorem julia_money_left (initial : ℕ) 
  (h_init : initial = initial_amount)
  (spent_game : ℕ)
  (h_spent_game : spent_game = amount_spent_on_game initial)
  (left_after_game : ℕ)
  (h_left_after_game : left_after_game = amount_left_after_game initial spent_game)
  (spent_in_game : ℕ)
  (h_spent_in_game : spent_in_game = amount_spent_on_in_game left_after_game)
  : final_amount left_after_game spent_in_game = 15 := by 
  sorry

end julia_money_left_l46_46350


namespace longest_leg_of_smallest_triangle_l46_46115

-- Definitions based on conditions
def is306090Triangle (h : ℝ) (s : ℝ) (l : ℝ) : Prop :=
  s = h / 2 ∧ l = s * (Real.sqrt 3)

def chain_of_306090Triangles (H : ℝ) : Prop :=
  ∃ h1 s1 l1 h2 s2 l2 h3 s3 l3 h4 s4 l4,
    is306090Triangle h1 s1 l1 ∧
    is306090Triangle h2 s2 l2 ∧
    is306090Triangle h3 s3 l3 ∧
    is306090Triangle h4 s4 l4 ∧
    h1 = H ∧ l1 = h2 ∧ l2 = h3 ∧ l3 = h4

-- Main theorem
theorem longest_leg_of_smallest_triangle (H : ℝ) (h : ℝ) (l : ℝ) (H_cond : H = 16) 
  (h_cond : h = 9) :
  chain_of_306090Triangles H →
  ∃ h4 s4 l4, is306090Triangle h4 s4 l4 ∧ l = h4 →
  l = 9 := 
by
  sorry

end longest_leg_of_smallest_triangle_l46_46115


namespace playground_ball_cost_l46_46251

-- Define the given conditions
def cost_jump_rope : ℕ := 7
def cost_board_game : ℕ := 12
def saved_by_dalton : ℕ := 6
def given_by_uncle : ℕ := 13
def additional_needed : ℕ := 4

-- Total money Dalton has
def total_money : ℕ := saved_by_dalton + given_by_uncle

-- Total cost needed to buy all three items
def total_cost_needed : ℕ := total_money + additional_needed

-- Combined cost of the jump rope and the board game
def combined_cost : ℕ := cost_jump_rope + cost_board_game

-- Prove the cost of the playground ball
theorem playground_ball_cost : ℕ := total_cost_needed - combined_cost

-- Expected result
example : playground_ball_cost = 4 := by
  sorry

end playground_ball_cost_l46_46251


namespace quadratic_two_real_roots_find_m_l46_46297

theorem quadratic_two_real_roots (m : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁ * x₂ = 3 * m^2 ∧ x₁ + x₂ = 4 * m :=
by
  sorry

theorem find_m (m : ℝ) (h : m > 0) (h_diff : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ - x₂ = 2) : m = 1 :=
by
  sorry

end quadratic_two_real_roots_find_m_l46_46297


namespace cubes_product_fraction_l46_46513

theorem cubes_product_fraction :
  (4^3 * 6^3 * 8^3 * 9^3 : ℚ) / (10^3 * 12^3 * 14^3 * 15^3) = 576 / 546875 := 
sorry

end cubes_product_fraction_l46_46513


namespace common_difference_is_3_l46_46467

variable {a : ℕ → ℤ} {d : ℤ}

-- Definitions of conditions
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def condition_1 (a : ℕ → ℤ) (d : ℤ) : Prop :=
  a 3 + a 11 = 24

def condition_2 (a : ℕ → ℤ) : Prop :=
  a 4 = 3

-- Theorem statement to prove
theorem common_difference_is_3 (a : ℕ → ℤ) (d : ℤ)
  (ha : is_arithmetic_sequence a d)
  (hc1 : condition_1 a d)
  (hc2 : condition_2 a) :
  d = 3 := by
  sorry

end common_difference_is_3_l46_46467


namespace multiples_of_5_in_4_digit_range_l46_46304

theorem multiples_of_5_in_4_digit_range : 
  let count_multiples := (9995 - 1000) / 5 + 1
  in count_multiples = 1800 :=
by
  sorry

end multiples_of_5_in_4_digit_range_l46_46304


namespace distance_third_day_l46_46662

theorem distance_third_day (total_distance : ℝ) (days : ℕ) (first_day_factor : ℝ) (halve_factor : ℝ) (third_day_distance : ℝ) :
  total_distance = 378 ∧ days = 6 ∧ first_day_factor = 4 ∧ halve_factor = 0.5 →
  third_day_distance = 48 := sorry

end distance_third_day_l46_46662


namespace radius_of_the_wheel_l46_46199

open Real

noncomputable def radius_of_wheel (speed_kmh : ℝ) (rpm : ℝ) : ℝ :=
  let speed_cms_min := (speed_kmh * 100000) / 60
  let circumference := speed_cms_min / rpm
  circumference / (2 * π)

theorem radius_of_the_wheel (speed_kmh : ℝ) (rpm : ℝ) (r : ℝ) :
  speed_kmh = 66 →
  rpm = 125.11373976342128 →
  abs (r - 140.007) < 0.001 :=
by
  intros h_speed h_rpm
  have r_def := radius_of_wheel speed_kmh rpm
  have hr : r = r_def := by sorry
  rw [r_def, h_speed, h_rpm]
  sorry

end radius_of_the_wheel_l46_46199


namespace glass_bottles_count_l46_46190

-- Declare the variables for the conditions
variable (G : ℕ)

-- Define the conditions
def aluminum_cans : ℕ := 8
def total_litter : ℕ := 18

-- State the theorem
theorem glass_bottles_count : G + aluminum_cans = total_litter → G = 10 :=
by
  intro h
  -- place proof here
  sorry

end glass_bottles_count_l46_46190


namespace remi_water_intake_l46_46682

def bottle_capacity := 20
def daily_refills := 3
def num_days := 7
def spill1 := 5
def spill2 := 8

def daily_intake := daily_refills * bottle_capacity
def total_intake_without_spill := daily_intake * num_days
def total_spill := spill1 + spill2
def total_intake_with_spill := total_intake_without_spill - total_spill

theorem remi_water_intake : total_intake_with_spill = 407 := 
by
  -- Provide proof here
  sorry

end remi_water_intake_l46_46682


namespace find_m_value_l46_46955

theorem find_m_value : ∃ m : ℤ, 81 - 6 = 25 + m ∧ m = 50 :=
by
  sorry

end find_m_value_l46_46955


namespace tree_drops_leaves_on_fifth_day_l46_46997

def initial_leaves := 340
def daily_drop_fraction := 1 / 10

noncomputable def leaves_after_day (n: ℕ) : ℕ :=
  match n with
  | 0 => initial_leaves
  | 1 => initial_leaves - Nat.floor (initial_leaves * daily_drop_fraction)
  | 2 => leaves_after_day 1 - Nat.floor (leaves_after_day 1 * daily_drop_fraction)
  | 3 => leaves_after_day 2 - Nat.floor (leaves_after_day 2 * daily_drop_fraction)
  | 4 => leaves_after_day 3 - Nat.floor (leaves_after_day 3 * daily_drop_fraction)
  | _ => 0  -- beyond the 4th day

theorem tree_drops_leaves_on_fifth_day : leaves_after_day 4 = 225 := by
  -- We'll skip the detailed proof here, focusing on the statement
  sorry

end tree_drops_leaves_on_fifth_day_l46_46997


namespace number_of_attendants_writing_with_both_l46_46593

-- Definitions for each of the conditions
def attendants_using_pencil : ℕ := 25
def attendants_using_pen : ℕ := 15
def attendants_using_only_one : ℕ := 20

-- Theorem that states the mathematically equivalent proof problem
theorem number_of_attendants_writing_with_both 
  (p : ℕ := attendants_using_pencil)
  (e : ℕ := attendants_using_pen)
  (o : ℕ := attendants_using_only_one) : 
  ∃ x, (p - x) + (e - x) = o ∧ x = 10 :=
by
  sorry

end number_of_attendants_writing_with_both_l46_46593


namespace find_f_value_l46_46973

noncomputable def f (a b x : ℝ) : ℝ := a * (Real.cos x)^2 - b * (Real.sin x) * (Real.cos x) - a / 2

theorem find_f_value (a b : ℝ)
  (h_max : ∀ x, f a b x ≤ 1/2)
  (h_at_pi_over_3 : f a b (Real.pi / 3) = (Real.sqrt 3) / 4) :
  f a b (-Real.pi / 3) = 0 ∨ f a b (-Real.pi / 3) = -(Real.sqrt 3) / 4 :=
sorry

end find_f_value_l46_46973


namespace binom_12_9_is_220_l46_46788

def choose (n k : ℕ) : ℕ := n.choose k

theorem binom_12_9_is_220 :
  choose 12 9 = 220 :=
by {
  -- Proof is omitted
  sorry
}

end binom_12_9_is_220_l46_46788


namespace total_parts_in_order_l46_46908

theorem total_parts_in_order (total_cost : ℕ) (cost_20 : ℕ) (cost_50 : ℕ) (num_50_dollar_parts : ℕ) (num_20_dollar_parts : ℕ) :
  total_cost = 2380 → cost_20 = 20 → cost_50 = 50 → num_50_dollar_parts = 40 → (total_cost = num_50_dollar_parts * cost_50 + num_20_dollar_parts * cost_20) → (num_50_dollar_parts + num_20_dollar_parts = 59) :=
by
  intro h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  sorry

end total_parts_in_order_l46_46908


namespace peter_age_l46_46703

theorem peter_age (P Q : ℕ) (h1 : Q - P = P / 2) (h2 : P + Q = 35) : Q = 21 :=
  sorry

end peter_age_l46_46703


namespace probability_53_sundays_in_leap_year_l46_46609

-- Define the conditions
def num_days_in_leap_year : ℕ := 366
def num_weeks_in_leap_year : ℕ := 52
def extra_days_in_leap_year : ℕ := 2
def num_combinations : ℕ := 7
def num_sunday_combinations : ℕ := 2

-- Define the problem statement
theorem probability_53_sundays_in_leap_year (hdays : num_days_in_leap_year = 52 * 7 + extra_days_in_leap_year) :
  (num_sunday_combinations / num_combinations : ℚ) = 2 / 7 :=
by
  sorry

end probability_53_sundays_in_leap_year_l46_46609


namespace solve_for_x_l46_46683

theorem solve_for_x : ∃ x : ℤ, 24 - 5 = 3 + x ∧ x = 16 :=
by
  sorry

end solve_for_x_l46_46683


namespace probability_not_sit_next_to_each_other_l46_46677

noncomputable def total_ways_to_choose_two_chairs_excluding_broken : ℕ := 28

noncomputable def unfavorable_outcomes : ℕ := 6

theorem probability_not_sit_next_to_each_other :
  (1 - (unfavorable_outcomes / total_ways_to_choose_two_chairs_excluding_broken) = (11 / 14)) :=
by sorry

end probability_not_sit_next_to_each_other_l46_46677


namespace continuous_linear_function_l46_46466

theorem continuous_linear_function {f : ℝ → ℝ} (h_cont : Continuous f) 
  (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_a_half : a < 1/2) (h_b_half : b < 1/2) 
  (h_eq : ∀ x : ℝ, f (f x) = a * f x + b * x) : 
  ∃ k : ℝ, (∀ x : ℝ, f x = k * x) ∧ (k * k - a * k - b = 0) := 
sorry

end continuous_linear_function_l46_46466


namespace sum_of_roots_eq_five_thirds_l46_46172

-- Define the quadratic equation
def quadratic_eq (n : ℝ) : Prop := 3 * n^2 - 5 * n - 4 = 0

-- Prove that the sum of the solutions to the quadratic equation is 5/3
theorem sum_of_roots_eq_five_thirds :
  (∃ a b : ℝ, quadratic_eq a ∧ quadratic_eq b ∧ a + b = 5 / 3) :=
sorry

end sum_of_roots_eq_five_thirds_l46_46172


namespace exists_city_reaching_all_l46_46162

variables {City : Type} (canReach : City → City → Prop)

-- Conditions from the problem
axiom reach_itself (A : City) : canReach A A
axiom reach_transitive {A B C : City} : canReach A B → canReach B C → canReach A C
axiom reach_any_two {P Q : City} : ∃ R : City, canReach R P ∧ canReach R Q

-- The proof problem
theorem exists_city_reaching_all (cities : City → Prop) :
  (∀ P Q, P ≠ Q → cities P → cities Q → ∃ R, cities R ∧ canReach R P ∧ canReach R Q) →
  ∃ C, ∀ A, cities A → canReach C A :=
by
  intros H
  sorry

end exists_city_reaching_all_l46_46162


namespace solve_inequality_l46_46873

def inequality_solution :=
  {x : ℝ // x < -3 ∨ x > -6/5}

theorem solve_inequality (x : ℝ) : 
  |2*x - 4| - |3*x + 9| < 1 → x < -3 ∨ x > -6/5 :=
by
  sorry

end solve_inequality_l46_46873


namespace find_dividend_l46_46416

-- Define the conditions
def divisor : ℕ := 20
def quotient : ℕ := 8
def remainder : ℕ := 6

-- Lean 4 statement to prove the dividend
theorem find_dividend : (divisor * quotient + remainder) = 166 := by
  sorry

end find_dividend_l46_46416


namespace p_sufficient_not_necessary_for_q_l46_46978

-- Define the conditions p and q
def p (x : ℝ) := x^2 < 5 * x - 6
def q (x : ℝ) := |x + 1| ≤ 4

-- The goal to prove
theorem p_sufficient_not_necessary_for_q :
  (∀ x, p x → q x) ∧ ¬ (∀ x, q x → p x) :=
by 
  sorry

end p_sufficient_not_necessary_for_q_l46_46978


namespace find_expression_for_3f_l46_46326

theorem find_expression_for_3f (f : ℝ → ℝ) (h : ∀ x > 0, f (3 * x) = 3 / (3 + 2 * x)) : 
  ∀ x > 0, 3 * f x = 27 / (9 + 2 * x) :=
by
  intro x hx
  have hx' : 3 * (x / 3) > 0 := by linarith
  rw [← (h (x / 3) hx')]
  sorry

end find_expression_for_3f_l46_46326


namespace no_two_digit_factorization_1729_l46_46829

noncomputable def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem no_two_digit_factorization_1729 :
  ¬ ∃ (a b : ℕ), a * b = 1729 ∧ is_two_digit a ∧ is_two_digit b :=
by
  sorry

end no_two_digit_factorization_1729_l46_46829


namespace leaves_dropped_on_fifth_day_l46_46995

theorem leaves_dropped_on_fifth_day 
  (initial_leaves : ℕ)
  (days : ℕ)
  (drops_per_day : ℕ)
  (total_dropped_four_days : ℕ)
  (leaves_dropped_fifth_day : ℕ)
  (h1 : initial_leaves = 340)
  (h2 : days = 4)
  (h3 : drops_per_day = initial_leaves / 10)
  (h4 : total_dropped_four_days = drops_per_day * days)
  (h5 : leaves_dropped_fifth_day = initial_leaves - total_dropped_four_days) :
  leaves_dropped_fifth_day = 204 :=
by
  sorry

end leaves_dropped_on_fifth_day_l46_46995


namespace positive_value_of_A_l46_46498

theorem positive_value_of_A (A : ℝ) :
  (A ^ 2 + 7 ^ 2 = 200) → A = Real.sqrt 151 :=
by
  intros h
  sorry

end positive_value_of_A_l46_46498


namespace probability_not_red_is_two_thirds_l46_46578

-- Given conditions as definitions
def number_of_orange_marbles : ℕ := 4
def number_of_purple_marbles : ℕ := 7
def number_of_red_marbles : ℕ := 8
def number_of_yellow_marbles : ℕ := 5

-- Define the total number of marbles
def total_marbles : ℕ :=
  number_of_orange_marbles + 
  number_of_purple_marbles + 
  number_of_red_marbles + 
  number_of_yellow_marbles

def number_of_non_red_marbles : ℕ :=
  number_of_orange_marbles + 
  number_of_purple_marbles + 
  number_of_yellow_marbles

-- Define the probability
def probability_not_red : ℚ :=
  number_of_non_red_marbles / total_marbles

-- The theorem that states the probability of not picking a red marble is 2/3
theorem probability_not_red_is_two_thirds :
  probability_not_red = 2 / 3 :=
by
  sorry

end probability_not_red_is_two_thirds_l46_46578


namespace probability_of_at_least_six_heads_is_correct_l46_46766

-- Definitions for the given problem
def binomial_coefficient (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

def total_possible_outcomes : ℕ :=
  2^8

def favorable_outcomes : ℕ :=
  binomial_coefficient 8 6 + binomial_coefficient 8 7 + binomial_coefficient 8 8

def probability_of_at_least_6_heads : ℚ :=
  favorable_outcomes / total_possible_outcomes

-- The proof statement
theorem probability_of_at_least_six_heads_is_correct :
  probability_of_at_least_6_heads = 37 / 256 :=
by sorry

end probability_of_at_least_six_heads_is_correct_l46_46766


namespace consecutive_integers_sum_l46_46649

theorem consecutive_integers_sum (a b : ℤ) (h1 : b = a + 1) (h2 : a < Real.sqrt 3) (h3 : Real.sqrt 3 < b) : a + b = 3 :=
sorry

end consecutive_integers_sum_l46_46649


namespace total_letters_sent_l46_46517

-- Define the number of letters sent in each month
def letters_in_January : ℕ := 6
def letters_in_February : ℕ := 9
def letters_in_March : ℕ := 3 * letters_in_January

-- Theorem statement: the total number of letters sent across the three months
theorem total_letters_sent :
  letters_in_January + letters_in_February + letters_in_March = 33 := by
  sorry

end total_letters_sent_l46_46517


namespace david_more_pushups_than_zachary_l46_46453

-- Definitions based on conditions
def david_pushups : ℕ := 37
def zachary_pushups : ℕ := 7

-- Theorem statement proving the answer
theorem david_more_pushups_than_zachary : david_pushups - zachary_pushups = 30 := by
  sorry

end david_more_pushups_than_zachary_l46_46453


namespace find_m_l46_46005

theorem find_m (m : ℕ) (hm : 0 < m)
  (a : ℕ := Nat.choose (2 * m) m)
  (b : ℕ := Nat.choose (2 * m + 1) m)
  (h : 13 * a = 7 * b) : m = 6 := by
  sorry

end find_m_l46_46005


namespace multiply_powers_same_base_l46_46596

theorem multiply_powers_same_base (a : ℝ) : a^3 * a = a^4 :=
by
  sorry

end multiply_powers_same_base_l46_46596


namespace quadratic_has_two_real_roots_find_m_l46_46291

def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem quadratic_has_two_real_roots (m : ℝ) :
  discriminant 1 (-4 * m) (3 * m^2) ≥ 0 :=
by
  unfold discriminant
  have h : (-4 * m)^2 - 4 * 1 * (3 * m^2) = 4 * m^2
  ring
  exact ge_of_eq h

theorem find_m (h : 0 < m) (root_diff : ℝ) 
  (diff_eq_two : root_diff = 2) : m = 1 :=
by
  -- Let the roots be x1 and x2
  let x1 := (4 * m + root_diff) / 2
  let x2 := (4 * m - root_diff) / 2
  have : x1 - x2 = root_diff :=
    by
      field_simp
      exact diff_eq_two
  have sum_eq := (x1 - x2) * (x1 + x2) - (x1 + x2) * (x1 - x2) = 4
  ring
  have h_m_eq_1 : 4 * m = 4,
  by field_simp
  exact h_m_eq_1

  have h_m_1 : m = 1,
  sorry
  exact ge_of_eq h_m_1

end quadratic_has_two_real_roots_find_m_l46_46291


namespace solve_for_x_l46_46144

noncomputable def vec (x y : ℝ) : ℝ × ℝ := (x, y)

theorem solve_for_x (x : ℝ) :
  let a := vec 1 2
  let b := vec x 1
  let u := (a.1 + 2 * b.1, a.2 + 2 * b.2)
  let v := (2 * a.1 - 2 * b.1, 2 * a.2 - 2 * b.2)
  (u.1 * v.2 = u.2 * v.1) → x = 1 / 2 := by
  sorry

end solve_for_x_l46_46144


namespace minutes_practiced_other_days_l46_46242

theorem minutes_practiced_other_days (total_hours : ℕ) (minutes_per_day : ℕ) (num_days : ℕ) :
  total_hours = 450 ∧ minutes_per_day = 86 ∧ num_days = 2 → (total_hours - num_days * minutes_per_day) = 278 := by
  sorry

end minutes_practiced_other_days_l46_46242


namespace sum_of_consecutive_integers_of_sqrt3_l46_46637

theorem sum_of_consecutive_integers_of_sqrt3 {a b : ℤ} (h1 : a + 1 = b) (h2 : (a : ℝ) < Real.sqrt 3) (h3 : Real.sqrt 3 < (b : ℝ)) :
  a + b = 3 := by
  sorry

end sum_of_consecutive_integers_of_sqrt3_l46_46637


namespace initial_weight_cucumbers_l46_46798

theorem initial_weight_cucumbers (W : ℝ) (h1 : 0.99 * W + 0.01 * W = W) 
                                  (h2 : W = (50 - 0.98 * 50 + 0.01 * W))
                                  (h3 : 50 > 0) : W = 100 := 
sorry

end initial_weight_cucumbers_l46_46798


namespace fraction_halfway_l46_46218

theorem fraction_halfway (a b c d : ℚ) (h₁ : a = 3) (h₂ : b = 4) (h₃ : c = 5) (h₄ : d = 6) :
  (1 / 2) * ((a / b) + (c / d)) = 19 / 24 := 
by
  sorry

end fraction_halfway_l46_46218


namespace sum_of_squares_eq_ten_l46_46613

noncomputable def x1 : ℝ := Real.sqrt 3 - Real.sqrt 2
noncomputable def x2 : ℝ := Real.sqrt 3 + Real.sqrt 2

theorem sum_of_squares_eq_ten : x1^2 + x2^2 = 10 := 
by
  sorry

end sum_of_squares_eq_ten_l46_46613


namespace variation_of_x_l46_46834

theorem variation_of_x (k j z : ℝ) : ∃ m : ℝ, ∀ x y : ℝ, (x = k * y^2) ∧ (y = j * z^(1 / 3)) → (x = m * z^(2 / 3)) :=
sorry

end variation_of_x_l46_46834


namespace symmetric_circle_eq_l46_46690

theorem symmetric_circle_eq {x y : ℝ} :
  (∃ x y : ℝ, (x+2)^2 + (y-1)^2 = 5) →
  (x - 1)^2 + (y + 2)^2 = 5 :=
sorry

end symmetric_circle_eq_l46_46690


namespace altitude_line_eq_circumcircle_eq_l46_46143

noncomputable def point := ℝ × ℝ

noncomputable def A : point := (5, 1)
noncomputable def B : point := (1, 3)
noncomputable def C : point := (4, 4)

theorem altitude_line_eq : ∃ (k b : ℝ), (k = 2 ∧ b = -4) ∧ (∀ x y : ℝ, y = k * x + b ↔ 2 * x - y - 4 = 0) :=
sorry

theorem circumcircle_eq : ∃ (h k r : ℝ), (h = 3 ∧ k = 2 ∧ r = 5) ∧ (∀ x y : ℝ, (x - h)^2 + (y - k)^2 = r ↔ (x - 3)^2 + (y - 2)^2 = 5) :=
sorry

end altitude_line_eq_circumcircle_eq_l46_46143


namespace standard_eq_minimal_circle_l46_46356

-- Definitions
variables {x y : ℝ}
variables (h₀ : 0 < x) (h₁ : 0 < y)
variables (h₂ : 3 / (2 + x) + 3 / (2 + y) = 1)

-- Theorem statement
theorem standard_eq_minimal_circle : (x - 4)^2 + (y - 4)^2 = 16^2 :=
sorry

end standard_eq_minimal_circle_l46_46356


namespace probability_at_least_6_heads_8_flips_l46_46740

-- Define the probability calculation of getting at least 6 heads in 8 coin flips.
def probability_at_least_6_heads (n : ℕ) (k : ℕ) : ℚ :=
  (Nat.choose n k + Nat.choose n (k + 1) + Nat.choose n (k + 2)) / 2^n

theorem probability_at_least_6_heads_8_flips : 
  probability_at_least_6_heads 8 6 = 37 / 256 := 
by
  sorry

end probability_at_least_6_heads_8_flips_l46_46740


namespace compound_interest_rate_l46_46122

theorem compound_interest_rate (P A : ℝ) (t n : ℕ) (CI r : ℝ)
  (hP : P = 1200)
  (hCI : CI = 1785.98)
  (ht : t = 5)
  (hn : n = 1)
  (hA : A = P * (1 + r/n)^(n * t)) :
  A = P + CI → 
  r = 0.204 :=
by
  sorry

end compound_interest_rate_l46_46122


namespace average_number_of_carnations_l46_46209

-- Define the number of carnations in each bouquet
def n1 : ℤ := 9
def n2 : ℤ := 23
def n3 : ℤ := 13
def n4 : ℤ := 36
def n5 : ℤ := 28
def n6 : ℤ := 45

-- Define the number of bouquets
def number_of_bouquets : ℤ := 6

-- Prove that the average number of carnations in the bouquets is 25.67
theorem average_number_of_carnations :
  ((n1 + n2 + n3 + n4 + n5 + n6) : ℚ) / (number_of_bouquets : ℚ) = 25.67 := 
by
  sorry

end average_number_of_carnations_l46_46209


namespace ben_has_56_marbles_l46_46892

-- We define the conditions first
variables (B : ℕ) (L : ℕ)

-- Leo has 20 more marbles than Ben
def condition1 : Prop := L = B + 20

-- Total number of marbles is 132
def condition2 : Prop := B + L = 132

-- The goal: proving the number of marbles Ben has is 56
theorem ben_has_56_marbles (h1 : condition1 B L) (h2 : condition2 B L) : B = 56 :=
by sorry

end ben_has_56_marbles_l46_46892


namespace ratio_red_to_green_apple_l46_46245

def total_apples : ℕ := 496
def green_apples : ℕ := 124
def red_apples : ℕ := total_apples - green_apples

theorem ratio_red_to_green_apple :
  red_apples / green_apples = 93 / 31 :=
by
  sorry

end ratio_red_to_green_apple_l46_46245


namespace molecular_weight_C7H6O2_l46_46062

noncomputable def molecular_weight_one_mole (w_9moles : ℕ) (m_9moles : ℕ) : ℕ :=
  m_9moles / w_9moles

theorem molecular_weight_C7H6O2 :
  molecular_weight_one_mole 9 1098 = 122 := by
  sorry

end molecular_weight_C7H6O2_l46_46062


namespace probability_non_defective_second_draw_l46_46584

theorem probability_non_defective_second_draw 
  (total_products : ℕ)
  (defective_products : ℕ)
  (first_draw_defective : Bool)
  (second_draw_non_defective_probability : ℚ) : 
  total_products = 100 → 
  defective_products = 3 → 
  first_draw_defective = true → 
  second_draw_non_defective_probability = 97 / 99 :=
by
  intros h_total h_defective h_first_draw
  subst h_total
  subst h_defective
  subst h_first_draw
  sorry

end probability_non_defective_second_draw_l46_46584


namespace non_positive_sequence_l46_46671

theorem non_positive_sequence
  (N : ℕ)
  (a : ℕ → ℝ)
  (h₀ : a 0 = 0)
  (hN : a N = 0)
  (h_rec : ∀ i, 1 ≤ i ∧ i ≤ N - 1 → a (i + 1) - 2 * a i + a (i - 1) = a i ^ 2) :
  ∀ i, 1 ≤ i ∧ i ≤ N - 1 → a i ≤ 0 := sorry

end non_positive_sequence_l46_46671


namespace expression_evaluation_l46_46944

theorem expression_evaluation : 
  76 + (144 / 12) + (15 * 19)^2 - 350 - (270 / 6) = 80918 :=
by
  sorry

end expression_evaluation_l46_46944


namespace point_in_fourth_quadrant_l46_46882

-- Define complex number and evaluate it
noncomputable def z : ℂ := (2 - (1 : ℂ) * Complex.I) / (1 + (1 : ℂ) * Complex.I)

-- Prove that the complex number z lies in the fourth quadrant
theorem point_in_fourth_quadrant (hz: z = (1/2 : ℂ) - (3/2 : ℂ) * Complex.I) : z.im < 0 ∧ z.re > 0 :=
by
  -- Skipping the proof here
  sorry

end point_in_fourth_quadrant_l46_46882


namespace volume_of_water_cylinder_l46_46573

theorem volume_of_water_cylinder :
  let r := 5
  let h := 10
  let depth := 3
  let θ := Real.arccos (3 / 5)
  let sector_area := (2 * θ) / (2 * Real.pi) * Real.pi * r^2
  let triangle_area := r * (2 * r * Real.sin θ)
  let water_surface_area := sector_area - triangle_area
  let volume := h * water_surface_area
  volume = 232.6 * Real.pi - 160 :=
by
  sorry

end volume_of_water_cylinder_l46_46573


namespace mowing_work_rate_l46_46227

variables (A B C : ℚ)

theorem mowing_work_rate :
  A + B = 1/28 → A + B + C = 1/21 → C = 1/84 :=
by
  intros h1 h2
  sorry

end mowing_work_rate_l46_46227


namespace chocolate_type_probability_l46_46592

noncomputable def probability_same_type_chocolate (steps : ℕ) : ℝ :=
  -- Symmetric Binomial Distribution Property
  1/2 * (1 + (1/3 : ℝ) ^ steps)

theorem chocolate_type_probability : 
  probability_same_type_chocolate 100 = 1/2 * (1 + (1/3 : ℝ) ^ 100) := by
  sorry

end chocolate_type_probability_l46_46592


namespace repeating_pattern_sum_23_l46_46432

def repeating_pattern_sum (n : ℕ) : ℤ :=
  let pattern := [4, -3, 2, -1, 0]
  let block_sum := List.sum pattern
  let complete_blocks := n / pattern.length
  let remainder := n % pattern.length
  complete_blocks * block_sum + List.sum (pattern.take remainder)

theorem repeating_pattern_sum_23 : repeating_pattern_sum 23 = 11 := 
  sorry

end repeating_pattern_sum_23_l46_46432


namespace rectangle_area_l46_46700

theorem rectangle_area (L W P : ℝ) (hL : L = 13) (hP : P = 50) (hP_eq : P = 2 * L + 2 * W) :
  L * W = 156 :=
by
  have hL_val : L = 13 := hL
  have hP_val : P = 50 := hP
  have h_perimeter : P = 2 * L + 2 * W := hP_eq
  sorry

end rectangle_area_l46_46700


namespace discount_allowed_l46_46924

-- Define the conditions
def CP : ℝ := 100 -- Cost Price (CP) is $100 for simplicity
def MP : ℝ := CP + 0.12 * CP -- Selling price marked 12% above cost price
def Loss : ℝ := 0.01 * CP -- Trader suffers a loss of 1% on CP
def SP : ℝ := CP - Loss -- Selling price after suffering the loss

-- State the equivalent proof problem in Lean
theorem discount_allowed : MP - SP = 13 := by
  sorry

end discount_allowed_l46_46924


namespace necessary_but_not_sufficient_l46_46817

theorem necessary_but_not_sufficient (x : ℝ) : (x < 0) -> (x^2 + x < 0 ↔ -1 < x ∧ x < 0) :=
by
  sorry

end necessary_but_not_sufficient_l46_46817


namespace x_power6_y_power6_l46_46492

theorem x_power6_y_power6 (x y a b : ℝ) (h1 : x + y = a) (h2 : x * y = b) :
  x^6 + y^6 = a^6 - 6 * a^4 * b + 9 * a^2 * b^2 - 2 * b^3 :=
sorry

end x_power6_y_power6_l46_46492


namespace expression_evaluation_l46_46058

theorem expression_evaluation :
  100 + (120 / 15) + (18 * 20) - 250 - (360 / 12) = 188 := by
  sorry

end expression_evaluation_l46_46058


namespace neg_q_sufficient_not_necc_neg_p_l46_46966

variable (p q : Prop)

theorem neg_q_sufficient_not_necc_neg_p (hp: p → q) (hnpq: ¬(q → p)) : (¬q → ¬p) ∧ (¬(¬p → ¬q)) :=
by
  sorry

end neg_q_sufficient_not_necc_neg_p_l46_46966


namespace probability_at_least_6_heads_l46_46755

-- Definitions of the binomial coefficient and probability function
def binom (n k : ℕ) : ℕ := Nat.choose n k

def probability (favorable total : ℕ) : ℚ := favorable / total

-- Proof problem statement
theorem probability_at_least_6_heads (flips : ℕ) (p : ℚ) 
  (h_flips : flips = 8) 
  (h_probability : p = probability (binom 8 6 + binom 8 7 + binom 8 8) (2 ^ flips)) : 
  p = 37 / 256 :=
by
  sorry

end probability_at_least_6_heads_l46_46755


namespace exists_pretty_hexagon_max_area_pretty_hexagon_l46_46430

-- Define the condition of a "pretty" hexagon
structure PrettyHexagon (L ℓ h : ℝ) : Prop :=
  (diag1 : (L + ℓ)^2 + h^2 = 1)
  (diag2 : (L + ℓ)^2 + h^2 = 1)
  (diag3 : (L + ℓ)^2 + h^2 = 1)
  (diag4 : (L + ℓ)^2 + h^2 = 1)
  (L_pos : L > 0) (L_lt_1 : L < 1)
  (ℓ_pos : ℓ > 0) (ℓ_lt_1 : ℓ < 1)
  (h_pos : h > 0) (h_lt_1 : h < 1)

-- Area of the hexagon given L, ℓ, and h
def hexagon_area (L ℓ h : ℝ) := 2 * (L + ℓ) * h

-- Question (a): Existence of a pretty hexagon with a given area
theorem exists_pretty_hexagon (k : ℝ) (hk : 0 < k ∧ k < 1) : 
  ∃ L ℓ h : ℝ, PrettyHexagon L ℓ h ∧ hexagon_area L ℓ h = k :=
sorry

-- Question (b): Maximum area of any pretty hexagon is at most 1
theorem max_area_pretty_hexagon : 
  ∀ L ℓ h : ℝ, PrettyHexagon L ℓ h → hexagon_area L ℓ h ≤ 1 :=
sorry

end exists_pretty_hexagon_max_area_pretty_hexagon_l46_46430


namespace marys_final_amount_l46_46009

def simple_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P * r * t

def final_amount (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P + simple_interest P r t

theorem marys_final_amount 
  (P : ℝ := 200)
  (A_after_2_years : ℝ := 260)
  (t1 : ℝ := 2)
  (t2 : ℝ := 6)
  (r : ℝ := (A_after_2_years - P) / (P * t1)) :
  final_amount P r t2 = 380 := 
by
  sorry

end marys_final_amount_l46_46009


namespace shadow_boundary_l46_46236

theorem shadow_boundary (r : ℝ) (O P : ℝ × ℝ × ℝ) :
  r = 2 → O = (0, 0, 2) → P = (0, -2, 4) → ∀ x : ℝ, ∃ y : ℝ, y = -10 :=
by sorry

end shadow_boundary_l46_46236


namespace find_number_l46_46732

theorem find_number (x : ℝ) (h : 0.8 * x = (4/5 : ℝ) * 25 + 16) : x = 45 :=
by
  sorry

end find_number_l46_46732


namespace solve_diophantine_equation_l46_46372

theorem solve_diophantine_equation :
  ∃ (x y : ℤ), x^2 - x * y - 6 * y^2 + 2 * x + 19 * y = 18 ∧ (x = 2 ∧ y = 2 ∨ x = -2 ∧ y = 2) :=
  sorry

end solve_diophantine_equation_l46_46372


namespace toby_total_sales_at_garage_sale_l46_46549

noncomputable def treadmill_price : ℕ := 100
noncomputable def chest_of_drawers_price : ℕ := treadmill_price / 2
noncomputable def television_price : ℕ := treadmill_price * 3
noncomputable def three_items_total : ℕ := treadmill_price + chest_of_drawers_price + television_price
noncomputable def total_sales : ℕ := three_items_total / (3 / 4) -- 75% is 0.75 or 3/4

theorem toby_total_sales_at_garage_sale : total_sales = 600 :=
by
  unfold treadmill_price chest_of_drawers_price television_price three_items_total total_sales
  simp
  exact sorry

end toby_total_sales_at_garage_sale_l46_46549


namespace julia_money_left_l46_46347

def initial_money : ℕ := 40
def spent_on_game : ℕ := initial_money / 2
def money_left_after_game : ℕ := initial_money - spent_on_game
def spent_on_in_game_purchases : ℕ := money_left_after_game / 4
def final_money_left : ℕ := money_left_after_game - spent_on_in_game_purchases

theorem julia_money_left : final_money_left = 15 := by
  sorry

end julia_money_left_l46_46347


namespace probability_odd_divisor_25_factorial_l46_46880

theorem probability_odd_divisor_25_factorial : 
  let divisors := (22 + 1) * (10 + 1) * (6 + 1) * (3 + 1) * (2 + 1) * (1 + 1) * (1 + 1) * (1 + 1) * (1 + 1)
  let odd_divisors := (10 + 1) * (6 + 1) * (3 + 1) * (2 + 1) * (1 + 1) * (1 + 1) * (1 + 1) * (1 + 1)
  (odd_divisors / divisors = 1 / 23) :=
sorry

end probability_odd_divisor_25_factorial_l46_46880


namespace probability_of_at_least_six_heads_is_correct_l46_46765

-- Definitions for the given problem
def binomial_coefficient (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

def total_possible_outcomes : ℕ :=
  2^8

def favorable_outcomes : ℕ :=
  binomial_coefficient 8 6 + binomial_coefficient 8 7 + binomial_coefficient 8 8

def probability_of_at_least_6_heads : ℚ :=
  favorable_outcomes / total_possible_outcomes

-- The proof statement
theorem probability_of_at_least_six_heads_is_correct :
  probability_of_at_least_6_heads = 37 / 256 :=
by sorry

end probability_of_at_least_six_heads_is_correct_l46_46765


namespace find_positive_value_of_A_l46_46495

variable (A : ℝ)

-- Given conditions
def relation (A B : ℝ) : ℝ := A^2 + B^2

-- The proof statement
theorem find_positive_value_of_A (h : relation A 7 = 200) : A = Real.sqrt 151 := sorry

end find_positive_value_of_A_l46_46495


namespace consecutive_sum_l46_46631

theorem consecutive_sum (a b : ℤ) (h1 : a + 1 = b) (h2 : (a : ℝ) < real.sqrt 3) (h3 : real.sqrt 3 < (b : ℝ)) : a + b = 3 := 
sorry

end consecutive_sum_l46_46631


namespace find_m_l46_46676

theorem find_m (θ₁ θ₂ : ℝ) (l : ℝ → ℝ) (m : ℕ) 
  (hθ₁ : θ₁ = Real.pi / 100) 
  (hθ₂ : θ₂ = Real.pi / 75)
  (hl : ∀ x, l x = x / 4) 
  (R : ((ℝ → ℝ) → (ℝ → ℝ)))
  (H_R : ∀ l, R l = (sorry : ℝ → ℝ)) 
  (R_n : ℕ → (ℝ → ℝ) → (ℝ → ℝ)) 
  (H_R1 : R_n 1 l = R l) 
  (H_Rn : ∀ n, R_n (n + 1) l = R (R_n n l)) :
  m = 1500 :=
sorry

end find_m_l46_46676


namespace binomial_12_9_l46_46790

def binomial (n k : ℕ) := nat.choose n k

theorem binomial_12_9 : binomial 12 9 = 220 :=
by
  have step1 : binomial 12 9 = binomial 12 3 := nat.choose_symm 12 9
  have step2 : binomial 12 3 = 220 := by sorry
  rw [step1, step2]

end binomial_12_9_l46_46790


namespace arc_length_sector_l46_46159

theorem arc_length_sector (r : ℝ) (α : ℝ) (h1 : r = 2) (h2 : α = π / 3) : 
  α * r = 2 * π / 3 := 
by 
  sorry

end arc_length_sector_l46_46159


namespace expression_equals_two_l46_46357

noncomputable def expression (a b c : ℝ) : ℝ :=
  (1 + a) / (1 + a + a * b) + (1 + b) / (1 + b + b * c) + (1 + c) / (1 + c + c * a)

theorem expression_equals_two (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) :
  expression a b c = 2 := by
  sorry

end expression_equals_two_l46_46357


namespace combined_average_speed_l46_46547

theorem combined_average_speed 
    (dA tA dB tB dC tC : ℝ)
    (mile_feet : ℝ)
    (hA : dA = 300) (hTA : tA = 6)
    (hB : dB = 400) (hTB : tB = 8)
    (hC : dC = 500) (hTC : tC = 10)
    (hMileFeet : mile_feet = 5280) :
    (1200 / 5280) / (24 / 3600) = 34.09 := 
by
  sorry

end combined_average_speed_l46_46547


namespace gcd_polynomial_997_l46_46616

theorem gcd_polynomial_997 (b : ℤ) (h : ∃ k : ℤ, b = 997 * k ∧ k % 2 = 1) :
  Int.gcd (3 * b ^ 2 + 17 * b + 31) (b + 7) = 1 := by
  sorry

end gcd_polynomial_997_l46_46616


namespace total_flowers_l46_46095

def number_of_flowers (F : ℝ) : Prop :=
  let vases := (F - 7.0) / 6.0
  vases = 6.666666667

theorem total_flowers : number_of_flowers 47.0 :=
by
  sorry

end total_flowers_l46_46095


namespace problem_statement_eq_l46_46611

variable (x y : ℝ)

def dollar (a b : ℝ) : ℝ := (a - b) ^ 2

theorem problem_statement_eq :
  dollar ((x + y) ^ 2) ((y + x) ^ 2) = 0 := by
  sorry

end problem_statement_eq_l46_46611


namespace distance_between_foci_of_ellipse_l46_46588

theorem distance_between_foci_of_ellipse :
  let c := (5, 2)
  let a := 5
  let b := 2
  2 * Real.sqrt (a^2 - b^2) = 2 * Real.sqrt 21 :=
by
  let c := (5, 2)
  let a := 5
  let b := 2
  show 2 * Real.sqrt (a^2 - b^2) = 2 * Real.sqrt 21
  sorry

end distance_between_foci_of_ellipse_l46_46588


namespace cone_and_sphere_volume_l46_46202

theorem cone_and_sphere_volume (π : ℝ) (r h : ℝ) (V_cylinder : ℝ) (V_cone V_sphere V_total : ℝ) 
  (h_cylinder : V_cylinder = 54 * π) 
  (h_radius : h = 3 * r)
  (h_cone : V_cone = (1 / 3) * π * r^2 * h) 
  (h_sphere : V_sphere = (4 / 3) * π * r^3) :
  V_total = 42 * π := 
by
  sorry

end cone_and_sphere_volume_l46_46202


namespace cost_of_each_shirt_l46_46366

theorem cost_of_each_shirt (initial_money : ℕ) (cost_pants : ℕ) (money_left : ℕ) (shirt_cost : ℕ)
  (h1 : initial_money = 109)
  (h2 : cost_pants = 13)
  (h3 : money_left = 74)
  (h4 : initial_money - (2 * shirt_cost + cost_pants) = money_left) :
  shirt_cost = 11 :=
by
  sorry

end cost_of_each_shirt_l46_46366


namespace red_balls_l46_46697

theorem red_balls (w r : ℕ) (h1 : w = 12) (h2 : w * 3 = r * 4) : r = 9 :=
sorry

end red_balls_l46_46697


namespace smallest_integer_solution_of_inequality_l46_46542

theorem smallest_integer_solution_of_inequality : ∃ x : ℤ, (3 * x ≥ x - 5) ∧ (∀ y : ℤ, 3 * y ≥ y - 5 → y ≥ -2) := 
sorry

end smallest_integer_solution_of_inequality_l46_46542


namespace rectangle_area_at_stage_8_l46_46330

-- Definitions based on conditions
def area_of_square (side_length : ℕ) : ℕ := side_length * side_length
def number_of_squares_in_stage (stage : ℕ) : ℕ := stage

-- The main theorem to prove
theorem rectangle_area_at_stage_8 : 
  area_of_square 4 * number_of_squares_in_stage 8 = 128 := by
  sorry

end rectangle_area_at_stage_8_l46_46330


namespace bird_difference_l46_46342

-- Variables representing given conditions
def num_migrating_families : Nat := 86
def num_remaining_families : Nat := 45
def avg_birds_per_migrating_family : Nat := 12
def avg_birds_per_remaining_family : Nat := 8

-- Definition to calculate total number of birds for migrating families
def total_birds_migrating : Nat := num_migrating_families * avg_birds_per_migrating_family

-- Definition to calculate total number of birds for remaining families
def total_birds_remaining : Nat := num_remaining_families * avg_birds_per_remaining_family

-- The statement that we need to prove
theorem bird_difference (h : total_birds_migrating - total_birds_remaining = 672) : 
  total_birds_migrating - total_birds_remaining = 672 := 
sorry

end bird_difference_l46_46342


namespace ticket_distribution_count_l46_46111

-- Defining the parameters
def tickets : Finset ℕ := {1, 2, 3, 4, 5, 6}
def people : ℕ := 4

-- Condition: Each person gets at least 1 ticket and at most 2 tickets, consecutive if 2.
def valid_distribution (dist: Finset (Finset ℕ)) :=
  dist.card = 4 ∧ ∀ s ∈ dist, s.card >= 1 ∧ s.card <= 2 ∧ (s.card = 1 ∨ (∃ x, s = {x, x+1}))

-- Question: Prove that there are 144 valid ways to distribute the tickets.
theorem ticket_distribution_count :
  ∃ dist: Finset (Finset ℕ), valid_distribution dist ∧ dist.card = 144 :=
by {
  sorry -- Proof is omitted as per instructions.
}

-- This statement checks distribution of 6 tickets to 4 people with given constraints is precisely 144

end ticket_distribution_count_l46_46111


namespace find_multiplier_l46_46617

theorem find_multiplier (x : ℕ) (h1 : 268 * x = 19832) (h2 : 2.68 * 0.74 = 1.9832) : x = 74 :=
sorry

end find_multiplier_l46_46617


namespace metal_waste_l46_46920

theorem metal_waste (l w : ℝ) (h : l > w) :
  let area_rectangle := l * w
  let area_circle := Real.pi * (w / 2) ^ 2
  let area_square := (w / Real.sqrt 2) ^ 2
  let wasted_metal := area_rectangle - area_circle + area_circle - area_square
  wasted_metal = l * w - w ^ 2 / 2 :=
by
  let area_rectangle := l * w
  let area_circle := Real.pi * (w / 2) ^ 2
  let area_square := (w / Real.sqrt 2) ^ 2
  let wasted_metal := area_rectangle - area_circle + area_circle - area_square
  sorry

end metal_waste_l46_46920


namespace probability_at_least_6_heads_in_8_flips_l46_46771

open scoped BigOperators

def binom (n k : ℕ) : ℕ := nat.choose n k

def total_outcomes (n : ℕ) := 2^n

def successful_outcomes (n k : ℕ) :=
  (finset.range (n + 1)).filter (λ x, x ≥ k).sum (λ x, binom n x)

def probability (n k : ℕ) :=
  (successful_outcomes n k) / (total_outcomes n : ℚ)

theorem probability_at_least_6_heads_in_8_flips :
  probability 8 6 = 37 / 256 := sorry

end probability_at_least_6_heads_in_8_flips_l46_46771


namespace total_weight_of_dumbbell_system_l46_46248

-- Definitions from the given conditions
def weight_pair1 : ℕ := 3
def weight_pair2 : ℕ := 5
def weight_pair3 : ℕ := 8

-- Goal: Prove that the total weight of the dumbbell system is 32 lbs
theorem total_weight_of_dumbbell_system :
  2 * weight_pair1 + 2 * weight_pair2 + 2 * weight_pair3 = 32 :=
by sorry

end total_weight_of_dumbbell_system_l46_46248


namespace area_of_region_B_l46_46793

noncomputable def region_B_area : ℝ :=
  let square_area := 900
  let excluded_area := 28.125 * Real.pi
  square_area - excluded_area

theorem area_of_region_B : region_B_area = 900 - 28.125 * Real.pi :=
by {
  sorry
}

end area_of_region_B_l46_46793


namespace brick_width_l46_46125

variable (w : ℝ)

theorem brick_width :
  ∃ (w : ℝ), 2 * (10 * w + 10 * 3 + 3 * w) = 164 → w = 4 :=
by
  sorry

end brick_width_l46_46125


namespace possible_marks_l46_46128

theorem possible_marks (n : ℕ) : n = 3 ∨ n = 6 ↔
  ∃ (m : ℕ), n = (m * (m - 1)) / 2 ∧ (∀ (k : ℕ), 1 ≤ k ∧ k ≤ n → ∃ (i j : ℕ), i < j ∧ j - i = k ∧ (∀ (x y : ℕ), x < y → x ≠ i ∨ y ≠ j)) :=
by sorry

end possible_marks_l46_46128


namespace ship_navigation_avoid_reefs_l46_46205

theorem ship_navigation_avoid_reefs (a : ℝ) (h : a > 0) :
  (10 * a) * 40 / Real.sqrt ((10 * a) ^ 2 + 40 ^ 2) > 20 ↔
  a > (4 * Real.sqrt 3 / 3) :=
by
  sorry

end ship_navigation_avoid_reefs_l46_46205


namespace quadratic_has_real_roots_find_value_of_m_l46_46292

theorem quadratic_has_real_roots (m : ℝ) : 
  ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (x1 - x2) ≠ 0 ∧ (x^2 - 4 * m * x + 3 * m^2 = 0) := 
by 
  sorry

theorem find_value_of_m (m : ℝ) (h1 : m > 0) (h2 : ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (x1 - x2 = 2)) :
  m = 1 := 
by 
  sorry

end quadratic_has_real_roots_find_value_of_m_l46_46292


namespace kelly_initially_had_l46_46001

def kelly_needs_to_pick : ℕ := 49
def kelly_will_have : ℕ := 105

theorem kelly_initially_had :
  kelly_will_have - kelly_needs_to_pick = 56 :=
by
  sorry

end kelly_initially_had_l46_46001


namespace tap_emptying_time_l46_46571

theorem tap_emptying_time
  (F : ℝ := 1 / 3)
  (T_combined : ℝ := 7.5):
  ∃ x : ℝ, x = 5 ∧ (F - (1 / x) = 1 / T_combined) := 
sorry

end tap_emptying_time_l46_46571


namespace three_digit_number_addition_l46_46090

theorem three_digit_number_addition (a b : ℕ) (ha : a < 10) (hb : b < 10) (h1 : 307 + 294 = 6 * 100 + b * 10 + 1)
  (h2 : (6 * 100 + b * 10 + 1) % 7 = 0) : a + b = 8 :=
by {
  sorry  -- Proof steps not needed
}

end three_digit_number_addition_l46_46090


namespace correct_statement_l46_46066

theorem correct_statement : -3 > -5 := 
by {
  sorry
}

end correct_statement_l46_46066


namespace max_M_l46_46625

noncomputable def conditions (x y z u : ℝ) : Prop :=
  (x - 2 * y = z - 2 * u) ∧ (2 * y * z = u * x) ∧ (0 < x) ∧ (0 < y) ∧ (0 < z) ∧ (0 < u) ∧ (z ≥ y)

theorem max_M (x y z u : ℝ) : conditions x y z u → ∃ M : ℝ, M = 6 + 4 * Real.sqrt 2 ∧ M ≤ z / y :=
by {
  sorry
}

end max_M_l46_46625


namespace possible_values_of_m_l46_46964

theorem possible_values_of_m (m : ℝ) (h1 : |m| = 2) (h2 : m - 2 ≠ 0) : m = -2 :=
by
  sorry

end possible_values_of_m_l46_46964


namespace find_k_for_perpendicular_lines_l46_46301

theorem find_k_for_perpendicular_lines (k : ℝ) :
  (∀ x y : ℝ, (k-3) * x + (5 - k) * y + 1 = 0) →
  (∀ x y : ℝ, 2 * (k-3) * x - 2 * y + 3 = 0) →
  (k = 1 ∨ k = 4) :=
by
  sorry

end find_k_for_perpendicular_lines_l46_46301


namespace sum_of_first_six_terms_l46_46029

def geometric_seq_sum (a r : ℕ) (n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

theorem sum_of_first_six_terms (a : ℕ) (r : ℕ) (h1 : r = 2) (h2 : a * (1 + r + r^2) = 3) :
  geometric_seq_sum a r 6 = 27 :=
by
  sorry

end sum_of_first_six_terms_l46_46029


namespace algebraic_expression_value_l46_46135

theorem algebraic_expression_value (x : ℝ) (h : (x^2 - x)^2 - 4 * (x^2 - x) - 12 = 0) : x^2 - x + 1 = 7 :=
sorry

end algebraic_expression_value_l46_46135


namespace insert_zeros_between_digits_is_cube_l46_46869

def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k^3 = n

theorem insert_zeros_between_digits_is_cube (k b : ℕ) (h_b : b ≥ 4) 
  : is_perfect_cube (1 * b^(3*(1+k)) + 3 * b^(2*(1+k)) + 3 * b^(1+k) + 1) :=
sorry

end insert_zeros_between_digits_is_cube_l46_46869


namespace calculate_expression_l46_46447

theorem calculate_expression : -1^2021 + 1^2022 = 0 := by
  sorry

end calculate_expression_l46_46447


namespace sequence_general_term_l46_46959

theorem sequence_general_term {a : ℕ → ℝ} (S : ℕ → ℝ) (n : ℕ) 
  (hS : ∀ n, S n = 4 * a n - 3) :
  a n = (4/3)^(n-1) :=
sorry

end sequence_general_term_l46_46959


namespace Bryan_deposit_amount_l46_46861

theorem Bryan_deposit_amount (deposit_mark : ℕ) (deposit_bryan : ℕ)
  (h1 : deposit_mark = 88)
  (h2 : deposit_bryan = 5 * deposit_mark - 40) : 
  deposit_bryan = 400 := 
by
  sorry

end Bryan_deposit_amount_l46_46861


namespace power_function_below_identity_l46_46900

theorem power_function_below_identity {α : ℝ} :
  (∀ x : ℝ, 1 < x → x^α < x) → α < 1 :=
by
  intro h
  sorry

end power_function_below_identity_l46_46900


namespace practice_other_days_l46_46240

-- Defining the total practice time for the week and the practice time for two days 
variable (total_minutes_week : ℤ) (total_minutes_two_days : ℤ)

-- Given conditions
axiom total_minutes_week_eq : total_minutes_week = 450
axiom total_minutes_two_days_eq : total_minutes_two_days = 172

-- The proof goal
theorem practice_other_days : (total_minutes_week - total_minutes_two_days) = 278 :=
by
  rw [total_minutes_week_eq, total_minutes_two_days_eq]
  show 450 - 172 = 278
  -- The proof goes here
  sorry

end practice_other_days_l46_46240


namespace correct_statements_l46_46969

theorem correct_statements (a b c : ℝ) (h : ∀ x, ax^2 + bx + c > 0 ↔ -2 < x ∧ x < 3) :
  ( ∃ (x : ℝ), c*x^2 + b*x + a < 0 ↔ -1/2 < x ∧ x < 1/3 ) ∧
  ( ∃ (b : ℝ), ∀ b, 12/(3*b + 4) + b = 8/3 ) ∧
  ( ∀ m, ¬ (m < -1 ∨ m > 2) ) ∧
  ( c = 2 → ∀ n1 n2, (3*a*n1^2 + 6*b*n1 = -3 ∧ 3*a*n2^2 + 6*b*n2 = 1) → n2 - n1 ∈ [2, 4] ) :=
sorry

end correct_statements_l46_46969


namespace kids_wearing_shoes_l46_46398

-- Definitions based on the problem's conditions
def total_kids := 22
def kids_with_socks := 12
def kids_with_both := 6
def barefoot_kids := 8

-- Theorem statement
theorem kids_wearing_shoes :
  (∃ (kids_with_shoes : ℕ), 
     (kids_with_shoes = (total_kids - barefoot_kids) - (kids_with_socks - kids_with_both) + kids_with_both) ∧ 
     kids_with_shoes = 8) :=
by
  sorry

end kids_wearing_shoes_l46_46398


namespace lydia_eats_apple_age_l46_46345

-- Define the conditions
def years_to_bear_fruit : ℕ := 7
def age_when_planted : ℕ := 4
def current_age : ℕ := 9

-- Define the theorem statement
theorem lydia_eats_apple_age : 
  (age_when_planted + years_to_bear_fruit = 11) :=
by
  sorry

end lydia_eats_apple_age_l46_46345


namespace probability_at_least_6_heads_in_8_flips_l46_46760

theorem probability_at_least_6_heads_in_8_flips : 
  (∑ k in finset.range 3, nat.choose 8 (6 + k)) / (2 ^ 8) = 37 / 256 :=
by sorry

end probability_at_least_6_heads_in_8_flips_l46_46760


namespace min_value_of_polynomial_l46_46391

theorem min_value_of_polynomial : ∃ x : ℝ, (x^2 + x + 1) = 3 / 4 :=
by {
  -- Solution steps are omitted
  sorry
}

end min_value_of_polynomial_l46_46391


namespace line_equation_of_projection_l46_46201

noncomputable def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_uv := u.1 * v.1 + u.2 * v.2
  let norm_v2 := v.1 * v.1 + v.2 * v.2
  (dot_uv / norm_v2 * v.1, dot_uv / norm_v2 * v.2)

theorem line_equation_of_projection (x y : ℝ) :
  proj (x, y) (3, -4) = (9 / 5, -12 / 5) ↔ y = (3 / 4) * x - 15 / 4 :=
sorry

end line_equation_of_projection_l46_46201


namespace find_ellipse_focus_l46_46950

theorem find_ellipse_focus :
  ∀ (a b : ℝ), a^2 = 5 → b^2 = 4 → 
  (∀ x y, (x^2)/(a^2) + (y^2)/(b^2) = 1) →
  ((∃ c : ℝ, c^2 = a^2 - b^2) ∧ (∃ x y, x = 0 ∧ (y = 1 ∨ y = -1))) :=
by
  sorry

end find_ellipse_focus_l46_46950


namespace product_of_real_values_eq_4_l46_46266

theorem product_of_real_values_eq_4 : ∀ s : ℝ, 
  (∃ x : ℝ, x ≠ 0 ∧ (1/(3*x) = (s - x)/9) → 
  (∀ x : ℝ, x ≠ 0 → (1/(3*x) = (s - x)/9 → x = s - 3))) → s = 4 :=
by
  sorry

end product_of_real_values_eq_4_l46_46266


namespace part1_solution_set_part2_no_real_x_l46_46623

-- Condition and problem definitions
def f (x a : ℝ) : ℝ := a^2 * x^2 + 2 * a * x - a^2 + 1

theorem part1_solution_set :
  (∀ x : ℝ, f x 2 ≤ 0 ↔ -3 / 2 ≤ x ∧ x ≤ 1 / 2) := sorry

theorem part2_no_real_x :
  ¬ ∃ x : ℝ, ∀ a : ℝ, -2 ≤ a ∧ a ≤ 2 → f x a ≥ 0 := sorry

end part1_solution_set_part2_no_real_x_l46_46623


namespace solve_equation_l46_46684

theorem solve_equation (x a b : ℝ) (h : x^2 - 6*x + 11 = 27) (sol_a : a = 8) (sol_b : b = -2) :
  3 * a - 2 * b = 28 :=
by
  sorry

end solve_equation_l46_46684


namespace spring_length_5kg_weight_l46_46849

variable {x y : ℝ}

-- Given conditions
def spring_length_no_weight : y = 6 := sorry
def spring_length_4kg_weight : y = 7.2 := sorry

-- The problem: to find the length of the spring for 5 kilograms
theorem spring_length_5kg_weight :
  (∃ (k b : ℝ), (∀ x, y = k * x + b) ∧ (b = 6) ∧ (4 * k + b = 7.2)) →
  y = 0.3 * 5 + 6 :=
  sorry

end spring_length_5kg_weight_l46_46849


namespace find_a_if_odd_l46_46984

theorem find_a_if_odd :
  ∀ (a : ℝ), (∀ x : ℝ, (a * (-x)^3 + (a - 1) * (-x)^2 + (-x) = -(a * x^3 + (a - 1) * x^2 + x))) → 
  a = 1 :=
by
  sorry

end find_a_if_odd_l46_46984


namespace range_of_a_l46_46837

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x| = ax + 1 → x < 0) → a > 1 :=
by
  sorry

end range_of_a_l46_46837


namespace Joey_age_digit_sum_l46_46667

structure Ages :=
  (joey_age : ℕ)
  (chloe_age : ℕ)
  (zoe_age : ℕ)

def is_multiple (a b : ℕ) : Prop :=
  ∃ k, a = k * b

def sum_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

theorem Joey_age_digit_sum
  (C J Z : ℕ)
  (h1 : J = C + 1)
  (h2 : Z = 1)
  (h3 : ∃ n, C + n = (n + 1) * m)
  (m : ℕ) (hm : m = 9)
  (h4 : C - 1 = 36) :
  sum_digits (J + 37) = 12 :=
by
  sorry

end Joey_age_digit_sum_l46_46667


namespace num_children_eq_3_l46_46083

-- Definitions from the conditions
def regular_ticket_cost : ℕ := 9
def child_ticket_discount : ℕ := 2
def given_amount : ℕ := 20 * 2
def received_change : ℕ := 1
def num_adults : ℕ := 2

-- Derived data
def total_ticket_cost : ℕ := given_amount - received_change
def adult_ticket_cost : ℕ := num_adults * regular_ticket_cost
def children_ticket_cost : ℕ := total_ticket_cost - adult_ticket_cost
def child_ticket_cost : ℕ := regular_ticket_cost - child_ticket_discount

-- Statement to prove
theorem num_children_eq_3 : (children_ticket_cost / child_ticket_cost) = 3 := by
  sorry

end num_children_eq_3_l46_46083


namespace vacant_seats_l46_46343

theorem vacant_seats (total_seats filled_percentage : ℕ) (h_filled_percentage : filled_percentage = 62) (h_total_seats : total_seats = 600) : 
  (total_seats - total_seats * filled_percentage / 100) = 228 :=
by
  sorry

end vacant_seats_l46_46343


namespace total_letters_sent_l46_46519

-- Define the number of letters sent in each month
def letters_in_January : ℕ := 6
def letters_in_February : ℕ := 9
def letters_in_March : ℕ := 3 * letters_in_January

-- Theorem statement: the total number of letters sent across the three months
theorem total_letters_sent :
  letters_in_January + letters_in_February + letters_in_March = 33 := by
  sorry

end total_letters_sent_l46_46519


namespace positive_integer_solutions_l46_46186

theorem positive_integer_solutions : 
  (∀ x : ℤ, ((1 + 2 * (x:ℝ)) / 4 - (1 - 3 * (x:ℝ)) / 10 > -1 / 5) ∧ (3 * (x:ℝ) - 1 < 2 * ((x:ℝ) + 1)) → (x = 1 ∨ x = 2)) :=
by 
  sorry

end positive_integer_solutions_l46_46186


namespace function_decreasing_on_interval_l46_46369

noncomputable def g (x : ℝ) := -(1 / 3) * Real.sin (4 * x - Real.pi / 3)
noncomputable def f (x : ℝ) := -(1 / 3) * Real.sin (4 * x)

theorem function_decreasing_on_interval :
  ∀ x y : ℝ, (-Real.pi / 8 ≤ x ∧ x ≤ Real.pi / 8) → (-Real.pi / 8 ≤ y ∧ y ≤ Real.pi / 8) → x < y → f x > f y :=
sorry

end function_decreasing_on_interval_l46_46369


namespace price_increase_solution_l46_46915

variable (x : ℕ)

def initial_profit := 10
def initial_sales := 500
def price_increase_effect := 20
def desired_profit := 6000

theorem price_increase_solution :
  ((initial_sales - price_increase_effect * x) * (initial_profit + x) = desired_profit) → (x = 5) :=
by
  sorry

end price_increase_solution_l46_46915


namespace min_value_of_x4_y3_z2_l46_46505

noncomputable def min_value_x4_y3_z2 (x y z : ℝ) : ℝ :=
  x^4 * y^3 * z^2

theorem min_value_of_x4_y3_z2 (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : 1/x + 1/y + 1/z = 9) : 
  min_value_x4_y3_z2 x y z = 1 / 3456 :=
by
  sorry

end min_value_of_x4_y3_z2_l46_46505


namespace simplify_and_evaluate_l46_46523

theorem simplify_and_evaluate (x : ℝ) (h : x = Real.sin (Real.pi / 6)) :
  (1 - 2 / (x - 1)) / ((x - 3) / (x^2 - 1)) = 3 / 2 :=
by
  -- simplify and evaluate the expression given the condition on x
  sorry

end simplify_and_evaluate_l46_46523


namespace quadratic_solution_identity_l46_46979

theorem quadratic_solution_identity (a b : ℤ) (h : (1 : ℤ)^2 + a * 1 + 2 * b = 0) : 2 * a + 4 * b = -2 := by
  sorry

end quadratic_solution_identity_l46_46979


namespace smallest_value_of_expression_l46_46325

theorem smallest_value_of_expression (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a^2 - b^2 = 16) : 
  (∃ k : ℚ, k = (a + b) / (a - b) + (a - b) / (a + b) ∧ (∀ x : ℚ, x = (a + b) / (a - b) + (a - b) / (a + b) → x ≥ 9/4)) :=
sorry

end smallest_value_of_expression_l46_46325


namespace identity_of_brothers_l46_46365

theorem identity_of_brothers
  (first_brother_speaks : Prop)
  (second_brother_speaks : Prop)
  (one_tells_truth : first_brother_speaks → ¬ second_brother_speaks)
  (other_tells_truth : ¬first_brother_speaks → second_brother_speaks) :
  first_brother_speaks = false ∧ second_brother_speaks = true :=
by
  sorry

end identity_of_brothers_l46_46365


namespace fraction_meaningful_l46_46157

theorem fraction_meaningful (x : ℝ) : (x - 1 ≠ 0) ↔ (∃ (y : ℝ), y = 3 / (x - 1)) :=
by sorry

end fraction_meaningful_l46_46157


namespace steve_first_stack_plastic_cups_l46_46527

theorem steve_first_stack_plastic_cups (cups_n : ℕ -> ℕ)
  (h_prop : ∀ n, cups_n (n + 1) = cups_n n + 4)
  (h_second : cups_n 2 = 21)
  (h_third : cups_n 3 = 25)
  (h_fourth : cups_n 4 = 29) :
  cups_n 1 = 17 :=
sorry

end steve_first_stack_plastic_cups_l46_46527


namespace problem_l46_46222

theorem problem : 3 + 15 / 3 - 2^3 = 0 := by
  sorry

end problem_l46_46222


namespace equation_represents_point_l46_46195

theorem equation_represents_point (a b x y : ℝ) :
  x^2 + y^2 + 2 * a * x + 2 * b * y + a^2 + b^2 = 0 ↔ x = -a ∧ y = -b := 
by sorry

end equation_represents_point_l46_46195


namespace average_score_l46_46722

theorem average_score (avg1 avg2 : ℕ) (matches1 matches2 : ℕ) (h_avg1 : avg1 = 60) (h_matches1 : matches1 = 10) (h_avg2 : avg2 = 70) (h_matches2 : matches2 = 15) : 
  (matches1 * avg1 + matches2 * avg2) / (matches1 + matches2) = 66 :=
by
  sorry

end average_score_l46_46722


namespace number_of_Ca_atoms_in_compound_l46_46911

theorem number_of_Ca_atoms_in_compound
  (n : ℤ)
  (total_weight : ℝ)
  (ca_weight : ℝ)
  (i_weight : ℝ)
  (n_i_atoms : ℤ)
  (molecular_weight : ℝ) :
  n_i_atoms = 2 →
  molecular_weight = 294 →
  ca_weight = 40.08 →
  i_weight = 126.90 →
  n * ca_weight + n_i_atoms * i_weight = molecular_weight →
  n = 1 :=
by
  sorry

end number_of_Ca_atoms_in_compound_l46_46911


namespace cylinder_area_l46_46983

noncomputable def cylinder_surface_area : ℝ :=
sorry

theorem cylinder_area
  (area_axial_section : ℝ)
  (h_area_axial_section : area_axial_section = 4) :
  cylinder_surface_area = 6 * real.pi :=
by sorry

end cylinder_area_l46_46983


namespace proof_problem_l46_46361

def A : Set ℝ := {x | x < 4}
def B : Set ℝ := {x | x^2 - 4 * x + 3 > 0}

theorem proof_problem : {x | x ∈ A ∧ x ∉ (A ∩ B)} = {x | 1 ≤ x ∧ x ≤ 3} :=
by {
  sorry
}

end proof_problem_l46_46361


namespace reciprocal_of_neg_one_div_2023_l46_46035

theorem reciprocal_of_neg_one_div_2023 : 1 / (-1 / (2023 : ℤ)) = -2023 := sorry

end reciprocal_of_neg_one_div_2023_l46_46035


namespace application_methods_count_l46_46165

theorem application_methods_count :
  let S := 5; -- number of students
  let U := 3; -- number of universities
  let unrestricted := U^S; -- unrestricted distribution
  let restricted_one_university_empty := (U - 1)^S * U; -- one university empty
  let restricted_two_universities_empty := 0; -- invalid scenario
  let valid_methods := unrestricted - restricted_one_university_empty - restricted_two_universities_empty;
  valid_methods - U = 144 :=
by
  let S := 5
  let U := 3
  let unrestricted := U^S
  let restricted_one_university_empty := (U - 1)^S * U
  let restricted_two_universities_empty := 0
  let valid_methods := unrestricted - restricted_one_university_empty - restricted_two_universities_empty
  have : valid_methods - U = 144 := by sorry
  exact this

end application_methods_count_l46_46165


namespace proof_of_arithmetic_sequence_l46_46395

theorem proof_of_arithmetic_sequence 
  (x y z : ℕ) 
  (h1 : x + y + z = 15) 
  (h2 : x < y) 
  (h3 : y < z)
  (h4 : (x + 1) * (z + 9) = (y + 3) ^ 2) : 
  (x, y, z) = (3, 5, 7) :=
sorry

end proof_of_arithmetic_sequence_l46_46395


namespace exists_monotonically_decreasing_interval_unique_tangent_line_intersection_l46_46140

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 
  (1 / 2) * m * (x - 1) ^ 2 - 2 * x + 3 + Real.log x

theorem exists_monotonically_decreasing_interval {m : ℝ} (hm : m ≥ 1) : 
  ∃ (a b : ℝ), 0 < a ∧ a < b ∧ ∀ x ∈ Icc a b, f m x ≤ f m a :=
sorry

theorem unique_tangent_line_intersection {m : ℝ} (hm : m ≥ 1) : 
  (∃ x : ℝ, f m x = -x + 2 ∧ f m 1 = 1 ∧ (∀ y, f m y = -y + 2 → y = 1)) ↔ (m = 1) :=
sorry

end exists_monotonically_decreasing_interval_unique_tangent_line_intersection_l46_46140


namespace red_balls_count_l46_46570

theorem red_balls_count (y : ℕ) (p_yellow : ℚ) (h1 : y = 10)
  (h2 : p_yellow = 5/8) (total_balls_le : ∀ r : ℕ, y + r ≤ 32) :
  ∃ r : ℕ, 10 + r > 0 ∧ p_yellow = 10 / (10 + r) ∧ r = 6 :=
by
  sorry

end red_balls_count_l46_46570


namespace num_children_eq_3_l46_46082

-- Definitions from the conditions
def regular_ticket_cost : ℕ := 9
def child_ticket_discount : ℕ := 2
def given_amount : ℕ := 20 * 2
def received_change : ℕ := 1
def num_adults : ℕ := 2

-- Derived data
def total_ticket_cost : ℕ := given_amount - received_change
def adult_ticket_cost : ℕ := num_adults * regular_ticket_cost
def children_ticket_cost : ℕ := total_ticket_cost - adult_ticket_cost
def child_ticket_cost : ℕ := regular_ticket_cost - child_ticket_discount

-- Statement to prove
theorem num_children_eq_3 : (children_ticket_cost / child_ticket_cost) = 3 := by
  sorry

end num_children_eq_3_l46_46082


namespace chips_probability_l46_46906

def total_chips : ℕ := 12
def blue_chips : ℕ := 4
def green_chips : ℕ := 3
def red_chips : ℕ := 5

def total_ways : ℕ := Nat.factorial total_chips

def blue_group_ways : ℕ := Nat.factorial blue_chips
def green_group_ways : ℕ := Nat.factorial green_chips
def red_group_ways : ℕ := Nat.factorial red_chips
def group_permutations : ℕ := Nat.factorial 3

def satisfying_arrangements : ℕ :=
  group_permutations * blue_group_ways * green_group_ways * red_group_ways

noncomputable def probability_of_event_B : ℚ :=
  (satisfying_arrangements : ℚ) / (total_ways : ℚ)

theorem chips_probability :
  probability_of_event_B = 1 / 4620 :=
by
  sorry

end chips_probability_l46_46906


namespace Mona_bikes_30_miles_each_week_l46_46864

theorem Mona_bikes_30_miles_each_week :
  let monday_distance := 6
  let wednesday_distance := 12
  let saturday_distance := 2 * monday_distance
  let total_distance := monday_distance + wednesday_distance + saturday_distance
  total_distance = 30 := by
  let monday_distance := 6
  let wednesday_distance := 12
  let saturday_distance := 2 * monday_distance
  let total_distance := monday_distance + wednesday_distance + saturday_distance
  show total_distance = 30
  sorry

end Mona_bikes_30_miles_each_week_l46_46864


namespace count_four_digit_multiples_of_5_l46_46317

theorem count_four_digit_multiples_of_5 : 
  let first_4_digit := 1000
  let last_4_digit := 9999
  let first_multiple_of_5 := 1000
  let last_multiple_of_5 := 9995
  let total_multiples_of_5 := (1999 - 200 + 1)
  first_multiple_of_5 % 5 = 0 ∧ last_multiple_of_5 % 5 = 0 ∧ first_4_digit ≤ first_multiple_of_5 ∧ last_multiple_of_5 ≤ last_4_digit
  → total_multiples_of_5 = 1800 :=
by
  sorry

end count_four_digit_multiples_of_5_l46_46317


namespace greatest_n_l46_46963

def S := { xy : ℕ × ℕ | ∃ x y : ℕ, xy = (x * y, x + y) }

def in_S (a : ℕ) : Prop := ∃ x y : ℕ, a = x * y * (x + y)

def pow_mod (a b m : ℕ) : ℕ := (a ^ b) % m

def satisfies_condition (a : ℕ) (n : ℕ) : Prop :=
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → in_S (a + pow_mod 2 k 9)

theorem greatest_n (a : ℕ) (n : ℕ) : 
  satisfies_condition a n → n ≤ 3 :=
sorry

end greatest_n_l46_46963


namespace probability_at_least_6_heads_in_8_flips_l46_46758

theorem probability_at_least_6_heads_in_8_flips : 
  (∑ k in finset.range 3, nat.choose 8 (6 + k)) / (2 ^ 8) = 37 / 256 :=
by sorry

end probability_at_least_6_heads_in_8_flips_l46_46758


namespace find_f_2008_l46_46284

variable (f : ℝ → ℝ) 
variable (g : ℝ → ℝ) -- g is the inverse of f

def satisfies_conditions (f : ℝ → ℝ) (g : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (g x) = x) ∧ (∀ y : ℝ, g (f y) = y) ∧ 
  (f 9 = 18) ∧ (∀ x : ℝ, g (x + 1) = (f (x + 1)))

theorem find_f_2008 (f : ℝ → ℝ) (g : ℝ → ℝ) 
  (h : satisfies_conditions f g) : f 2008 = -1981 :=
sorry

end find_f_2008_l46_46284


namespace find_n_l46_46386

-- Define the hyperbola and its properties
def hyperbola (m n : ℝ) : Prop :=
  m > 0 ∧ n > 0 ∧ 2 = (m / (m / 2)) ∧ ∃ f : ℝ × ℝ, f = (m, 0)

-- Define the parabola and its properties
def parabola_focus (m : ℝ) : Prop :=
  (m, 0) = (m, 0)

-- The statement we want to prove
theorem find_n (m : ℝ) (n : ℝ) (H_hyperbola : hyperbola m n) (H_parabola : parabola_focus m) : n = 12 :=
sorry

end find_n_l46_46386


namespace jake_more_peaches_than_jill_l46_46187

theorem jake_more_peaches_than_jill :
  let steven_peaches := 14
  let jake_peaches := steven_peaches - 6
  let jill_peaches := 5
  jake_peaches - jill_peaches = 3 :=
by
  let steven_peaches := 14
  let jake_peaches := steven_peaches - 6
  let jill_peaches := 5
  sorry

end jake_more_peaches_than_jill_l46_46187


namespace number_of_solutions_l46_46354

noncomputable def g (x : ℝ) : ℝ := -3 * Real.sin (2 * Real.pi * x)

theorem number_of_solutions (h : -1 ≤ x ∧ x ≤ 1) : 
  (∃ s : ℕ, s = 21 ∧ ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → g (g (g x)) = g x) :=
sorry

end number_of_solutions_l46_46354


namespace percentage_of_managers_l46_46397

theorem percentage_of_managers (P : ℝ) :
  (200 : ℝ) * (P / 100) - 99.99999999999991 = 0.98 * (200 - 99.99999999999991) →
  P = 99 := 
sorry

end percentage_of_managers_l46_46397


namespace line_ellipse_intersection_l46_46288

theorem line_ellipse_intersection (m : ℝ) : 
  (∀ x y : ℝ, y = 2 * x + m ∧ (x^2 / 4 + y^2 / 2 = 1)) →
  (-3 * Real.sqrt 2 < m ∧ m < 3 * Real.sqrt 2) ∨
  (m = 3 * Real.sqrt 2 ∨ m = -3 * Real.sqrt 2) ∨ 
  (m < -3 * Real.sqrt 2 ∨ m > 3 * Real.sqrt 2) :=
sorry

end line_ellipse_intersection_l46_46288


namespace arithmetic_progression_20th_term_and_sum_l46_46123

theorem arithmetic_progression_20th_term_and_sum :
  let a := 3
  let d := 4
  let n := 20
  let a_20 := a + (n - 1) * d
  let S_20 := n / 2 * (a + a_20)
  a_20 = 79 ∧ S_20 = 820 := by
    let a := 3
    let d := 4
    let n := 20
    let a_20 := a + (n - 1) * d
    let S_20 := n / 2 * (a + a_20)
    sorry

end arithmetic_progression_20th_term_and_sum_l46_46123


namespace identity_equality_l46_46487

theorem identity_equality (a b m n x y : ℝ) :
  ((a^2 + b^2) * (m^2 + n^2) * (x^2 + y^2)) =
  ((a * n * y - a * m * x - b * m * y + b * n * x)^2 + (a * m * y + a * n * x + b * m * x - b * n * y)^2) :=
by
  sorry

end identity_equality_l46_46487


namespace solve_for_A_l46_46501

def hash (A B : ℝ) : ℝ := A^2 + B^2

theorem solve_for_A (A : ℝ) (h : hash A 7 = 200) : A = Real.sqrt 151 :=
by
  sorry

end solve_for_A_l46_46501


namespace natural_number_factors_of_M_l46_46110

def M : ℕ := (2^3) * (3^2) * (5^5) * (7^1) * (11^2)

theorem natural_number_factors_of_M : ∃ n : ℕ, n = 432 ∧ (∀ d, d ∣ M → d > 0 → d ≤ M) :=
by
  let number_of_factors := (3 + 1) * (2 + 1) * (5 + 1) * (1 + 1) * (2 + 1)
  use number_of_factors
  sorry

end natural_number_factors_of_M_l46_46110


namespace prob_at_least_6_heads_eq_l46_46748

-- define the number of coin flips
def n := 8

-- define the number of possible outcomes (2^n)
def total_outcomes := 2 ^ n

-- define the binomial coefficients for cases: 6 heads, 7 heads, 8 heads
def binom_8_6 := Nat.choose 8 6
def binom_8_7 := Nat.choose 8 7
def binom_8_8 := Nat.choose 8 8

-- calculate the favorable outcomes for at least 6 heads
def favorable_outcomes := binom_8_6 + binom_8_7 + binom_8_8

-- define the probability of getting at least 6 heads
def probability := (favorable_outcomes : ℚ) / total_outcomes

theorem prob_at_least_6_heads_eq : probability = 37 / 256 := by
  sorry

end prob_at_least_6_heads_eq_l46_46748


namespace balance_after_transactions_l46_46509

variable (x : ℝ)

def monday_spent : ℝ := 0.525 * x
def tuesday_spent (remaining : ℝ) : ℝ := 0.106875 * remaining
def wednesday_spent (remaining : ℝ) : ℝ := 0.131297917 * remaining
def thursday_spent (remaining : ℝ) : ℝ := 0.040260605 * remaining

def final_balance (x : ℝ) : ℝ :=
  let after_monday := x - monday_spent x
  let after_tuesday := after_monday - tuesday_spent after_monday
  let after_wednesday := after_tuesday - wednesday_spent after_tuesday
  after_wednesday - thursday_spent after_wednesday

theorem balance_after_transactions (x : ℝ) :
  final_balance x = 0.196566478 * x :=
by
  sorry

end balance_after_transactions_l46_46509


namespace divisor_probability_of_25_factorial_is_odd_and_multiple_of_5_l46_46696

theorem divisor_probability_of_25_factorial_is_odd_and_multiple_of_5 :
  let prime_factors_25 := 2^22 * 3^10 * 5^6 * 7^3 * 11^2 * 13^1 * 17^1 * 19^1 * 23^1
  let total_divisors := (22+1) * (10+1) * (6+1) * (3+1) * (2+1) * (1+1) * (1+1) * (1+1)
  let odd_and_multiple_of_5_divisors := (6+1) * (3+1) * (2+1) * (1+1) * (1+1)
  (odd_and_multiple_of_5_divisors / total_divisors : ℚ) = 7 / 23 := 
sorry

end divisor_probability_of_25_factorial_is_odd_and_multiple_of_5_l46_46696


namespace milan_total_minutes_l46_46511

-- Conditions
variables (x : ℝ) -- minutes on the second phone line
variables (minutes_first : ℝ := x + 20) -- minutes on the first phone line
def total_cost (x : ℝ) := 3 + 0.15 * (x + 20) + 4 + 0.10 * x

-- Statement to prove
theorem milan_total_minutes (x : ℝ) (h : total_cost x = 56) :
  x + (x + 20) = 252 :=
sorry

end milan_total_minutes_l46_46511


namespace inequalities_hold_l46_46601

theorem inequalities_hold (a b c x y z : ℝ) (hxa : x ≤ a) (hyb : y ≤ b) (hzc : z ≤ c) :
  (x * y + y * z + z * x ≤ a * b + b * c + c * a) ∧
  (x^2 + y^2 + z^2 ≤ a^2 + b^2 + c^2) ∧
  (x * y * z ≤ a * b * c) :=
by
  sorry

end inequalities_hold_l46_46601


namespace terminating_decimal_expansion_l46_46460

theorem terminating_decimal_expansion : (11 / 125 : ℝ) = 0.088 := 
by
  sorry

end terminating_decimal_expansion_l46_46460


namespace weeks_to_buy_iphone_l46_46939

-- Definitions based on conditions
def iphone_cost : ℝ := 800
def trade_in_value : ℝ := 240
def earnings_per_week : ℝ := 80

-- Mathematically equivalent proof problem
theorem weeks_to_buy_iphone : 
  ∀ (iphone_cost trade_in_value earnings_per_week : ℝ), 
  (iphone_cost - trade_in_value) / earnings_per_week = 7 :=
by
  -- Using the given conditions directly.
  intros iphone_cost trade_in_value earnings_per_week
  sorry

end weeks_to_buy_iphone_l46_46939


namespace symmetry_center_cos_sin_l46_46943

theorem symmetry_center_cos_sin (f : ℝ → ℝ)
  (h : ∀ x, f x = cos (2 * x - π / 6) * sin (2 * x) - 1 / 4) :
  (7 * π / 24, (0 : ℝ)) = (k / 4 + π / 24, 0)
  ∧ k = 1 :=
by
  sorry

end symmetry_center_cos_sin_l46_46943


namespace solve_for_k_l46_46833

theorem solve_for_k (x k : ℝ) (h : x = -3) (h_eq : k * (x + 4) - 2 * k - x = 5) : k = -2 :=
by sorry

end solve_for_k_l46_46833


namespace combines_like_terms_l46_46064

theorem combines_like_terms (a : ℝ) : 2 * a - 5 * a = -3 * a := 
by sorry

end combines_like_terms_l46_46064


namespace smallest_resolvable_debt_l46_46893

def pig_value : ℤ := 450
def goat_value : ℤ := 330
def gcd_pig_goat : ℤ := Int.gcd pig_value goat_value

theorem smallest_resolvable_debt :
  ∃ p g : ℤ, gcd_pig_goat * 4 = pig_value * p + goat_value * g := 
by
  sorry

end smallest_resolvable_debt_l46_46893


namespace remove_7_increases_probability_l46_46211

open Finset
open BigOperators

variable {α : Type*} [Fintype α] [DecidableEq α] (S : Finset α) (n : α)

noncomputable def isValidPairSum14 (pair : α × α) : Prop :=
  pair.1 ≠ pair.2 ∧ pair.1 + pair.2 = 14

noncomputable def validPairs (S : Finset α) : Finset (α × α) :=
  (S ×ˢ S).filter isValidPairSum14

noncomputable def probabilitySum14 (S : Finset α) : ℚ :=
  (validPairs S).card / S.card.choose 2

theorem remove_7_increases_probability :
  probabilitySum14 (erase S 7) > probabilitySum14 S :=
  sorry

end remove_7_increases_probability_l46_46211


namespace max_xyz_l46_46355

theorem max_xyz (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) 
(h4 : (x * y) + 3 * z = (x + 3 * z) * (y + 3 * z)) 
: ∀ x y z, ∃ (a : ℝ), a = (x * y * z) ∧ a ≤ (1/81) :=
sorry

end max_xyz_l46_46355


namespace neutralization_reaction_l46_46606

/-- When combining 2 moles of CH3COOH and 2 moles of NaOH, 2 moles of H2O are formed
    given the balanced chemical reaction CH3COOH + NaOH → CH3COONa + H2O 
    with a molar ratio of 1:1:1 (CH3COOH:NaOH:H2O). -/
theorem neutralization_reaction
  (mCH3COOH : ℕ) (mNaOH : ℕ) :
  (mCH3COOH = 2) → (mNaOH = 2) → (mCH3COOH = mNaOH) →
  ∃ mH2O : ℕ, mH2O = 2 :=
by intros; existsi 2; sorry

end neutralization_reaction_l46_46606


namespace find_n_from_equation_l46_46468

theorem find_n_from_equation (n m : ℕ) (h1 : (1^m / 5^m) * (1^n / 4^n) = 1 / (2 * 10^31)) (h2 : m = 31) : n = 16 := 
by
  sorry

end find_n_from_equation_l46_46468


namespace dubblefud_red_balls_l46_46720

theorem dubblefud_red_balls (R B : ℕ) 
  (h1 : 2 ^ R * 4 ^ B * 5 ^ B = 16000)
  (h2 : B = G) : R = 6 :=
by
  -- Skipping the actual proof
  sorry

end dubblefud_red_balls_l46_46720


namespace solve_for_A_l46_46502

def hash (A B : ℝ) : ℝ := A^2 + B^2

theorem solve_for_A (A : ℝ) (h : hash A 7 = 200) : A = Real.sqrt 151 :=
by
  sorry

end solve_for_A_l46_46502


namespace quadratic_has_real_roots_find_value_of_m_l46_46294

theorem quadratic_has_real_roots (m : ℝ) : 
  ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (x1 - x2) ≠ 0 ∧ (x^2 - 4 * m * x + 3 * m^2 = 0) := 
by 
  sorry

theorem find_value_of_m (m : ℝ) (h1 : m > 0) (h2 : ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (x1 - x2 = 2)) :
  m = 1 := 
by 
  sorry

end quadratic_has_real_roots_find_value_of_m_l46_46294


namespace total_weight_full_l46_46912

theorem total_weight_full {x y p q : ℝ}
    (h1 : x + (3/4) * y = p)
    (h2 : x + (1/3) * y = q) :
    x + y = (8/5) * p - (3/5) * q :=
by
  sorry

end total_weight_full_l46_46912


namespace knives_percentage_l46_46451

-- Definitions based on conditions
def initial_knives : ℕ := 6
def initial_forks : ℕ := 12
def initial_spoons : ℕ := 3 * initial_knives
def traded_knives : ℕ := 10
def traded_spoons : ℕ := 6

-- Definitions for calculations
def final_knives : ℕ := initial_knives + traded_knives
def final_spoons : ℕ := initial_spoons - traded_spoons
def total_silverware : ℕ := final_knives + final_spoons + initial_forks

-- Theorem to prove the percentage of knives
theorem knives_percentage : (final_knives * 100) / total_silverware = 40 := by
  sorry

end knives_percentage_l46_46451


namespace arithmetic_sequence_sum_l46_46993

-- Let {a_n} be an arithmetic sequence.
-- Define Sn as the sum of the first n terms.
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n+1) = a n + d

def S (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (n * (a 0 + a (n-1))) / 2

theorem arithmetic_sequence_sum
  (a : ℕ → ℤ)
  (h_arith : is_arithmetic_sequence a)
  (h_condition : 2 * a 6 = a 7 + 5) :
  S a 11 = 55 :=
sorry

end arithmetic_sequence_sum_l46_46993


namespace initial_price_after_markup_l46_46777

theorem initial_price_after_markup 
  (wholesale_price : ℝ) 
  (h_markup_80 : ∀ P, P = wholesale_price → 1.80 * P = 1.80 * wholesale_price)
  (h_markup_diff : ∀ P, P = wholesale_price → 2.00 * P - 1.80 * P = 3) 
  : 1.80 * wholesale_price = 27 := 
by
  sorry

end initial_price_after_markup_l46_46777


namespace complement_A_in_U_l46_46626

/-- Problem conditions -/
def is_universal_set (x : ℕ) : Prop := (x - 6) * (x + 1) ≤ 0
def A : Set ℕ := {1, 2, 4}
def U : Set ℕ := { x | is_universal_set x }

/-- Proof statement -/
theorem complement_A_in_U : (U \ A) = {3, 5, 6} :=
by
  sorry  -- replacement for the proof

end complement_A_in_U_l46_46626


namespace sum_of_digits_of_N_eq_14_l46_46439

theorem sum_of_digits_of_N_eq_14 :
  ∃ N : ℕ, (N * (N + 1)) / 2 = 3003 ∧ (N % 10 + N / 10 % 10 = 14) :=
by
  sorry

end sum_of_digits_of_N_eq_14_l46_46439


namespace positive_value_of_A_l46_46497

theorem positive_value_of_A (A : ℝ) :
  (A ^ 2 + 7 ^ 2 = 200) → A = Real.sqrt 151 :=
by
  intros h
  sorry

end positive_value_of_A_l46_46497


namespace point_on_y_axis_l46_46658

theorem point_on_y_axis (x y : ℝ) (h : x = 0 ∧ y = -1) : y = -1 := by
  -- Using the conditions directly
  cases h with
  | intro hx hy =>
    -- The proof would typically follow, but we include sorry to complete the statement
    sorry

end point_on_y_axis_l46_46658


namespace total_selection_methods_l46_46443

-- Define the students and days
inductive Student
| S1 | S2 | S3 | S4 | S5

inductive Day
| Wednesday | Thursday | Friday | Saturday | Sunday

-- The condition where S1 cannot be on Saturday and S2 cannot be on Sunday
def valid_arrangement (arrangement : Day → Student) : Prop :=
  arrangement Day.Saturday ≠ Student.S1 ∧
  arrangement Day.Sunday ≠ Student.S2

-- The main statement
theorem total_selection_methods : ∃ (arrangement_count : ℕ), 
  arrangement_count = 78 ∧
  ∀ (arrangement : Day → Student), valid_arrangement arrangement → 
  arrangement_count = 78 :=
sorry

end total_selection_methods_l46_46443


namespace quadratic_has_two_real_roots_find_m_l46_46290

def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem quadratic_has_two_real_roots (m : ℝ) :
  discriminant 1 (-4 * m) (3 * m^2) ≥ 0 :=
by
  unfold discriminant
  have h : (-4 * m)^2 - 4 * 1 * (3 * m^2) = 4 * m^2
  ring
  exact ge_of_eq h

theorem find_m (h : 0 < m) (root_diff : ℝ) 
  (diff_eq_two : root_diff = 2) : m = 1 :=
by
  -- Let the roots be x1 and x2
  let x1 := (4 * m + root_diff) / 2
  let x2 := (4 * m - root_diff) / 2
  have : x1 - x2 = root_diff :=
    by
      field_simp
      exact diff_eq_two
  have sum_eq := (x1 - x2) * (x1 + x2) - (x1 + x2) * (x1 - x2) = 4
  ring
  have h_m_eq_1 : 4 * m = 4,
  by field_simp
  exact h_m_eq_1

  have h_m_1 : m = 1,
  sorry
  exact ge_of_eq h_m_1

end quadratic_has_two_real_roots_find_m_l46_46290


namespace textbook_weight_ratio_l46_46488

def jon_textbooks_weights : List ℕ := [2, 8, 5, 9]
def brandon_textbooks_weight : ℕ := 8

theorem textbook_weight_ratio : 
  (jon_textbooks_weights.sum : ℚ) / (brandon_textbooks_weight : ℚ) = 3 :=
by 
  sorry

end textbook_weight_ratio_l46_46488


namespace comb_12_9_eq_220_l46_46792

theorem comb_12_9_eq_220 : (Nat.choose 12 9) = 220 := by
  sorry

end comb_12_9_eq_220_l46_46792


namespace ratio_of_a_b_to_b_c_l46_46147

theorem ratio_of_a_b_to_b_c (a b c : ℝ) (h₁ : b / a = 3) (h₂ : c / b = 2) : 
  (a + b) / (b + c) = 4 / 9 := by
  sorry

end ratio_of_a_b_to_b_c_l46_46147


namespace reciprocal_neg_one_over_2023_eq_neg_2023_l46_46039

theorem reciprocal_neg_one_over_2023_eq_neg_2023 : (1 / (-1 / (2023 : ℝ))) = -2023 :=
by
  sorry

end reciprocal_neg_one_over_2023_eq_neg_2023_l46_46039


namespace range_of_a_l46_46842

theorem range_of_a (a : ℝ) (h : ∃ x : ℝ, |x - a| + |x - 1| ≤ 3) : -2 ≤ a ∧ a ≤ 4 :=
sorry

end range_of_a_l46_46842


namespace arithmetic_mean_rational_numbers_between_zero_and_one_l46_46891

/-- Prove that starting with 0 and 1, using the arithmetic mean,
    one can obtain the number 1/5
    and any rational number between 0 and 1. -/
theorem arithmetic_mean_rational_numbers_between_zero_and_one :
  (∃ s : Finset ℚ, {0, 1} ⊆ s ∧ ∀ (a b ∈ s), (a + b) / 2 ∈ s) ∧
  (∃ s : Finset ℚ, {0, 1} ⊆ s ∧ (1 / 5 : ℚ) ∈ s ∧ ∀ (a b ∈ s), (a + b) / 2 ∈ s) ∧
  (∀ (r : ℚ), 0 < r ∧ r < 1 → ∃ s : Finset ℚ, {0, 1} ⊆ s ∧ r ∈ s ∧ ∀ (a b ∈ s), (a + b) / 2 ∈ s) :=
sorry

end arithmetic_mean_rational_numbers_between_zero_and_one_l46_46891


namespace sum_of_consecutive_integers_a_lt_sqrt3_lt_b_l46_46640

theorem sum_of_consecutive_integers_a_lt_sqrt3_lt_b 
  (a b : ℤ) (h1 : a < b) (h2 : ∀ x : ℤ, x ≤ a → x < b) (h3 : a < Real.sqrt 3) (h4 : Real.sqrt 3 < b) : 
  a + b = 3 :=
by
  sorry

end sum_of_consecutive_integers_a_lt_sqrt3_lt_b_l46_46640


namespace laps_remaining_l46_46666

theorem laps_remaining 
  (total_laps : ℕ)
  (saturday_laps : ℕ)
  (sunday_morning_laps : ℕ)
  (total_laps_eq : total_laps = 98)
  (saturday_laps_eq : saturday_laps = 27)
  (sunday_morning_laps_eq : sunday_morning_laps = 15) :
  total_laps - saturday_laps - sunday_morning_laps = 56 :=
by
  rw [total_laps_eq, saturday_laps_eq, sunday_morning_laps_eq]
  norm_num

end laps_remaining_l46_46666


namespace sum_of_consecutive_integers_a_lt_sqrt3_lt_b_l46_46641

theorem sum_of_consecutive_integers_a_lt_sqrt3_lt_b 
  (a b : ℤ) (h1 : a < b) (h2 : ∀ x : ℤ, x ≤ a → x < b) (h3 : a < Real.sqrt 3) (h4 : Real.sqrt 3 < b) : 
  a + b = 3 :=
by
  sorry

end sum_of_consecutive_integers_a_lt_sqrt3_lt_b_l46_46641


namespace assembly_line_average_output_l46_46585

theorem assembly_line_average_output :
  (60 / 90) + (60 / 60) = (5 / 3) →
  60 + 60 = 120 →
  120 / (5 / 3) = 72 :=
by
  intros h1 h2
  -- Proof follows, but we will end with 'sorry' to indicate further proof steps need to be done.
  sorry

end assembly_line_average_output_l46_46585


namespace interchanged_digits_subtraction_l46_46333

theorem interchanged_digits_subtraction (a b k : ℤ) (h1 : 10 * a + b = 2 * k * (a + b)) :
  10 * b + a - 3 * (a + b) = (9 - 4 * k) * (a + b) :=
by sorry

end interchanged_digits_subtraction_l46_46333


namespace simplify_expression_l46_46022

theorem simplify_expression : (Real.cos (18 * Real.pi / 180) * Real.cos (42 * Real.pi / 180) - 
                              Real.cos (72 * Real.pi / 180) * Real.sin (42 * Real.pi / 180) = 1 / 2) :=
by
  sorry

end simplify_expression_l46_46022


namespace max_value_expression_l46_46461

noncomputable def max_expression (a b c : ℝ) : ℝ :=
  (a * b * c * (a + b + c)) / ((a + b)^2 * (b + c)^3)

theorem max_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  max_expression a b c ≤ 1 / 12 := 
sorry

end max_value_expression_l46_46461


namespace find_side_b_of_triangle_l46_46480

theorem find_side_b_of_triangle
  (A B : Real) (a b : Real)
  (hA : A = Real.pi / 6)
  (hB : B = Real.pi / 4)
  (ha : a = 2) :
  b = 2 * Real.sqrt 2 :=
sorry

end find_side_b_of_triangle_l46_46480


namespace max_digit_d_for_number_divisible_by_33_l46_46259

theorem max_digit_d_for_number_divisible_by_33 : ∃ d e : ℕ, d ≤ 9 ∧ e ≤ 9 ∧ 8 * 100000 + d * 10000 + 8 * 1000 + 3 * 100 + 3 * 10 + e % 33 = 0 ∧  d = 8 :=
by {
  sorry
}

end max_digit_d_for_number_divisible_by_33_l46_46259


namespace rise_in_water_level_l46_46737

-- Define the conditions related to the cube and the vessel
def edge_length := 15 -- in cm
def base_length := 20 -- in cm
def base_width := 15 -- in cm

-- Calculate volumes and areas
def V_cube := edge_length ^ 3
def A_base := base_length * base_width

-- Declare the mathematical proof problem statement
theorem rise_in_water_level : 
  (V_cube / A_base : ℝ) = 11.25 :=
by
  -- edge_length, V_cube, A_base are all already defined
  -- This particularly proves (15^3) / (20 * 15) = 11.25
  sorry

end rise_in_water_level_l46_46737


namespace solve_linear_system_l46_46820

-- Given conditions
def matrix : Matrix (Fin 2) (Fin 3) ℚ :=
  ![![1, -1, 1], ![1, 1, 3]]

def system_of_equations (x y : ℚ) : Prop :=
  (x - y = 1) ∧ (x + y = 3)

-- Desired solution
def solution (x y : ℚ) : Prop :=
  x = 2 ∧ y = 1

-- Proof problem statement
theorem solve_linear_system : ∃ x y : ℚ, system_of_equations x y ∧ solution x y := by
  sorry

end solve_linear_system_l46_46820


namespace prob_at_least_6_heads_eq_l46_46749

-- define the number of coin flips
def n := 8

-- define the number of possible outcomes (2^n)
def total_outcomes := 2 ^ n

-- define the binomial coefficients for cases: 6 heads, 7 heads, 8 heads
def binom_8_6 := Nat.choose 8 6
def binom_8_7 := Nat.choose 8 7
def binom_8_8 := Nat.choose 8 8

-- calculate the favorable outcomes for at least 6 heads
def favorable_outcomes := binom_8_6 + binom_8_7 + binom_8_8

-- define the probability of getting at least 6 heads
def probability := (favorable_outcomes : ℚ) / total_outcomes

theorem prob_at_least_6_heads_eq : probability = 37 / 256 := by
  sorry

end prob_at_least_6_heads_eq_l46_46749


namespace complex_sum_l46_46856

noncomputable def omega : ℂ := sorry
axiom omega_power_five : omega^5 = 1
axiom omega_not_one : omega ≠ 1

theorem complex_sum :
  (omega^20 + omega^25 + omega^30 + omega^35 + omega^40 + omega^45 + omega^50 + omega^55 + omega^60 + omega^65 + omega^70) = 11 :=
by
  sorry

end complex_sum_l46_46856


namespace maintain_constant_chromosomes_l46_46702

-- Definitions
def meiosis_reduces_chromosomes (original_chromosomes : ℕ) : ℕ := original_chromosomes / 2

def fertilization_restores_chromosomes (half_chromosomes : ℕ) : ℕ := half_chromosomes * 2

-- The proof problem
theorem maintain_constant_chromosomes (original_chromosomes : ℕ) (somatic_chromosomes : ℕ) :
  meiosis_reduces_chromosomes original_chromosomes = somatic_chromosomes / 2 ∧
  fertilization_restores_chromosomes (meiosis_reduces_chromosomes original_chromosomes) = somatic_chromosomes :=
sorry

end maintain_constant_chromosomes_l46_46702


namespace fermat_numbers_pairwise_coprime_l46_46711

theorem fermat_numbers_pairwise_coprime :
  ∀ i j : ℕ, i ≠ j → Nat.gcd (2 ^ (2 ^ i) + 1) (2 ^ (2 ^ j) + 1) = 1 :=
sorry

end fermat_numbers_pairwise_coprime_l46_46711
