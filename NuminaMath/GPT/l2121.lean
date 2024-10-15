import Mathlib

namespace NUMINAMATH_GPT_latte_cost_l2121_212197

theorem latte_cost (L : ℝ) 
  (latte_days : ℝ := 5)
  (iced_coffee_cost : ℝ := 2)
  (iced_coffee_days : ℝ := 3)
  (weeks_in_year : ℝ := 52)
  (spending_reduction : ℝ := 0.25)
  (savings : ℝ := 338) 
  (current_annual_spending : ℝ := 4 * savings)
  (weekly_spending : ℝ := latte_days * L + iced_coffee_days * iced_coffee_cost)
  (annual_spending_eq : weeks_in_year * weekly_spending = current_annual_spending) :
  L = 4 := 
sorry

end NUMINAMATH_GPT_latte_cost_l2121_212197


namespace NUMINAMATH_GPT_min_area_triangle_l2121_212124

-- Define the points and line equation
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (30, 10)
def line (x : ℤ) : ℤ := 2 * x - 5

-- Define a function to calculate the area using Shoelace formula
noncomputable def area (C : ℤ × ℤ) : ℝ :=
  (1 / 2) * |(A.1 * B.2 + B.1 * C.2 + C.1 * A.2) - (A.2 * B.1 + B.2 * C.1 + C.2 * A.1)|

-- Prove that the minimum area of the triangle with the given conditions is 15
theorem min_area_triangle : ∃ (C : ℤ × ℤ), C.2 = line C.1 ∧ area C = 15 := sorry

end NUMINAMATH_GPT_min_area_triangle_l2121_212124


namespace NUMINAMATH_GPT_total_tickets_sold_l2121_212144

theorem total_tickets_sold 
  (ticket_price : ℕ) 
  (discount_40_percent : ℕ → ℕ) 
  (discount_15_percent : ℕ → ℕ) 
  (revenue : ℕ) 
  (people_10_discount_40 : ℕ) 
  (people_20_discount_15 : ℕ) 
  (people_full_price : ℕ)
  (h_ticket_price : ticket_price = 20)
  (h_discount_40 : ∀ n, discount_40_percent n = n * 12)
  (h_discount_15 : ∀ n, discount_15_percent n = n * 17)
  (h_revenue : revenue = 760)
  (h_people_10_discount_40 : people_10_discount_40 = 10)
  (h_people_20_discount_15 : people_20_discount_15 = 20)
  (h_people_full_price : people_full_price * ticket_price = 300) :
  (people_10_discount_40 + people_20_discount_15 + people_full_price = 45) :=
by
  sorry

end NUMINAMATH_GPT_total_tickets_sold_l2121_212144


namespace NUMINAMATH_GPT_intersection_correct_l2121_212192

-- Conditions
def M : Set ℤ := { -1, 0, 1, 3, 5 }
def N : Set ℤ := { -2, 1, 2, 3, 5 }

-- Statement to prove
theorem intersection_correct : M ∩ N = { 1, 3, 5 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_correct_l2121_212192


namespace NUMINAMATH_GPT_tangent_line_curve_l2121_212106

theorem tangent_line_curve (x₀ : ℝ) (a : ℝ) :
  (ax₀ + 2 = e^x₀ + 1) ∧ (a = e^x₀) → a = 1 := by
  sorry

end NUMINAMATH_GPT_tangent_line_curve_l2121_212106


namespace NUMINAMATH_GPT_factor_x12_minus_4096_l2121_212154

theorem factor_x12_minus_4096 (x : ℝ) : 
  x^12 - 4096 = (x^6 + 64) * (x + 2) * (x^2 - 2*x + 4) * (x - 2) * (x^2 + 2*x + 4) :=
by
  sorry

end NUMINAMATH_GPT_factor_x12_minus_4096_l2121_212154


namespace NUMINAMATH_GPT_upper_limit_arun_weight_l2121_212184

theorem upper_limit_arun_weight (x w : ℝ) :
  (65 < w ∧ w < x) ∧
  (60 < w ∧ w < 70) ∧
  (w ≤ 68) ∧
  (w = 67) →
  x = 68 :=
by
  sorry

end NUMINAMATH_GPT_upper_limit_arun_weight_l2121_212184


namespace NUMINAMATH_GPT_find_value_of_x_l2121_212136

theorem find_value_of_x (x y : ℤ) (h1 : x + y = 20) (h2 : x - y = 36) : x = 28 := 
sorry

end NUMINAMATH_GPT_find_value_of_x_l2121_212136


namespace NUMINAMATH_GPT_volleyball_not_basketball_l2121_212168

def class_size : ℕ := 40
def basketball_enjoyers : ℕ := 15
def volleyball_enjoyers : ℕ := 20
def neither_sport : ℕ := 10

theorem volleyball_not_basketball :
  (volleyball_enjoyers - (basketball_enjoyers + volleyball_enjoyers - (class_size - neither_sport))) = 15 :=
by
  sorry

end NUMINAMATH_GPT_volleyball_not_basketball_l2121_212168


namespace NUMINAMATH_GPT_julie_bought_boxes_l2121_212181

-- Definitions for the conditions
def packages_per_box := 5
def sheets_per_package := 250
def sheets_per_newspaper := 25
def newspapers := 100

-- Calculations based on conditions
def total_sheets_needed := newspapers * sheets_per_newspaper
def sheets_per_box := packages_per_box * sheets_per_package

-- The goal: to prove that the number of boxes of paper Julie bought is 2
theorem julie_bought_boxes : total_sheets_needed / sheets_per_box = 2 :=
  by
    sorry

end NUMINAMATH_GPT_julie_bought_boxes_l2121_212181


namespace NUMINAMATH_GPT_avg_age_assist_coaches_l2121_212128

-- Define the conditions given in the problem

def total_members := 50
def avg_age_total := 22
def girls := 30
def boys := 15
def coaches := 5
def avg_age_girls := 18
def avg_age_boys := 20
def head_coaches := 3
def assist_coaches := 2
def avg_age_head_coaches := 30

-- Define the target theorem to prove
theorem avg_age_assist_coaches : 
  (avg_age_total * total_members - avg_age_girls * girls - avg_age_boys * boys - avg_age_head_coaches * head_coaches) / assist_coaches = 85 := 
  by
    sorry

end NUMINAMATH_GPT_avg_age_assist_coaches_l2121_212128


namespace NUMINAMATH_GPT_train_cross_pole_time_l2121_212133

noncomputable def time_to_cross_pole : ℝ :=
  let speed_km_hr := 60
  let speed_m_s := speed_km_hr * 1000 / 3600
  let length_of_train := 50
  length_of_train / speed_m_s

theorem train_cross_pole_time :
  time_to_cross_pole = 3 := 
by
  sorry

end NUMINAMATH_GPT_train_cross_pole_time_l2121_212133


namespace NUMINAMATH_GPT_bryan_books_l2121_212131

theorem bryan_books (books_per_continent : ℕ) (total_books : ℕ) 
  (h1 : books_per_continent = 122) 
  (h2 : total_books = 488) : 
  total_books / books_per_continent = 4 := 
by 
  sorry

end NUMINAMATH_GPT_bryan_books_l2121_212131


namespace NUMINAMATH_GPT_convex_quadrilateral_probability_l2121_212143

noncomputable def probability_convex_quadrilateral (n : ℕ) : ℚ :=
  if n = 6 then (Nat.choose 6 4 : ℚ) / (Nat.choose 15 4 : ℚ) else 0

theorem convex_quadrilateral_probability :
  probability_convex_quadrilateral 6 = 1 / 91 :=
by
  sorry

end NUMINAMATH_GPT_convex_quadrilateral_probability_l2121_212143


namespace NUMINAMATH_GPT_inequality_transformation_l2121_212123

variable {a b c d : ℝ}

theorem inequality_transformation
  (h1 : a < b)
  (h2 : b < 0)
  (h3 : c < d)
  (h4 : d < 0) :
  (d / a) < (c / a) :=
by
  sorry

end NUMINAMATH_GPT_inequality_transformation_l2121_212123


namespace NUMINAMATH_GPT_tan_double_angle_l2121_212120

open Real

theorem tan_double_angle {θ : ℝ} (h1 : tan (π / 2 - θ) = 4 * cos (2 * π - θ)) (h2 : abs θ < π / 2) : 
  tan (2 * θ) = sqrt 15 / 7 :=
sorry

end NUMINAMATH_GPT_tan_double_angle_l2121_212120


namespace NUMINAMATH_GPT_cubic_difference_l2121_212101

theorem cubic_difference (x y : ℤ) (h1 : x + y = 14) (h2 : 3 * x + y = 20) : x^3 - y^3 = -1304 :=
sorry

end NUMINAMATH_GPT_cubic_difference_l2121_212101


namespace NUMINAMATH_GPT_subcommittee_combinations_l2121_212150

open Nat

theorem subcommittee_combinations :
  (choose 8 3) * (choose 6 2) = 840 := by
  sorry

end NUMINAMATH_GPT_subcommittee_combinations_l2121_212150


namespace NUMINAMATH_GPT_triangle_minimum_area_l2121_212125

theorem triangle_minimum_area :
  ∃ p q : ℤ, p ≠ 0 ∧ q ≠ 0 ∧ (1 / 2) * |30 * q - 18 * p| = 3 :=
sorry

end NUMINAMATH_GPT_triangle_minimum_area_l2121_212125


namespace NUMINAMATH_GPT_find_c_l2121_212130

theorem find_c (c : ℝ)
  (h1 : ∃ y : ℝ, y = (-2)^2 - (-2) + c)
  (h2 : ∃ m : ℝ, m = 2 * (-2) - 1)
  (h3 : ∃ x y, y - (4 + c) = -5 * (x + 2) ∧ x = 0 ∧ y = 0) :
  c = 4 :=
sorry

end NUMINAMATH_GPT_find_c_l2121_212130


namespace NUMINAMATH_GPT_remainder_24_l2121_212160

-- Statement of the problem in Lean 4
theorem remainder_24 (y : ℤ) (h : y % 288 = 45) : y % 24 = 21 :=
by
  sorry

end NUMINAMATH_GPT_remainder_24_l2121_212160


namespace NUMINAMATH_GPT_steve_average_speed_l2121_212180

theorem steve_average_speed 
  (Speed1 Time1 Speed2 Time2 : ℝ) 
  (cond1 : Speed1 = 40) 
  (cond2 : Time1 = 5)
  (cond3 : Speed2 = 80) 
  (cond4 : Time2 = 3) 
: 
(Speed1 * Time1 + Speed2 * Time2) / (Time1 + Time2) = 55 := 
sorry

end NUMINAMATH_GPT_steve_average_speed_l2121_212180


namespace NUMINAMATH_GPT_integer_solutions_x2_minus_y2_equals_12_l2121_212149

theorem integer_solutions_x2_minus_y2_equals_12 : 
  ∃! (s : Finset (ℤ × ℤ)), (∀ (xy : ℤ × ℤ), xy ∈ s → xy.1^2 - xy.2^2 = 12) ∧ s.card = 4 :=
sorry

end NUMINAMATH_GPT_integer_solutions_x2_minus_y2_equals_12_l2121_212149


namespace NUMINAMATH_GPT_ratio_of_p_to_q_l2121_212193

theorem ratio_of_p_to_q (p q : ℝ) (h₁ : (p + q) / (p - q) = 4 / 3) (h₂ : p / q = r) : r = 7 :=
sorry

end NUMINAMATH_GPT_ratio_of_p_to_q_l2121_212193


namespace NUMINAMATH_GPT_first_player_wins_game_l2121_212174

theorem first_player_wins_game :
  ∀ (coins : ℕ), coins = 2019 →
  (∀ (n : ℕ), n % 2 = 1 ∧ 1 ≤ n ∧ n ≤ 99) →
  (∀ (m : ℕ), m % 2 = 0 ∧ 2 ≤ m ∧ m ≤ 100) →
  ∃ (f : ℕ → ℕ → ℕ), (∀ (c : ℕ), c <= coins → c = 0) :=
by
  sorry

end NUMINAMATH_GPT_first_player_wins_game_l2121_212174


namespace NUMINAMATH_GPT_real_solutions_quadratic_l2121_212198

theorem real_solutions_quadratic (d : ℝ) (h : 0 < d) :
  ∃ x : ℝ, x^2 - 8 * x + d < 0 ↔ 0 < d ∧ d < 16 :=
by
  sorry

end NUMINAMATH_GPT_real_solutions_quadratic_l2121_212198


namespace NUMINAMATH_GPT_no_integer_solution_l2121_212186

theorem no_integer_solution (a b : ℤ) : ¬(a^2 + b^2 = 10^100 + 3) :=
sorry

end NUMINAMATH_GPT_no_integer_solution_l2121_212186


namespace NUMINAMATH_GPT_calculate_max_income_l2121_212175

variables 
  (total_lunch_pasta : ℕ) (total_lunch_chicken : ℕ) (total_lunch_fish : ℕ)
  (sold_lunch_pasta : ℕ) (sold_lunch_chicken : ℕ) (sold_lunch_fish : ℕ)
  (dinner_pasta : ℕ) (dinner_chicken : ℕ) (dinner_fish : ℕ)
  (price_pasta : ℝ) (price_chicken : ℝ) (price_fish : ℝ)
  (discount : ℝ)
  (max_income : ℝ)

def unsold_lunch_pasta := total_lunch_pasta - sold_lunch_pasta
def unsold_lunch_chicken := total_lunch_chicken - sold_lunch_chicken
def unsold_lunch_fish := total_lunch_fish - sold_lunch_fish

def discounted_price (price : ℝ) := price * (1 - discount)

def income_lunch (sold : ℕ) (price : ℝ) := sold * price
def income_dinner (fresh : ℕ) (price : ℝ) := fresh * price
def income_unsold (unsold : ℕ) (price : ℝ) := unsold * discounted_price price

theorem calculate_max_income 
  (h_pasta_total : total_lunch_pasta = 8) (h_chicken_total : total_lunch_chicken = 5) (h_fish_total : total_lunch_fish = 4)
  (h_pasta_sold : sold_lunch_pasta = 6) (h_chicken_sold : sold_lunch_chicken = 3) (h_fish_sold : sold_lunch_fish = 3)
  (h_dinner_pasta : dinner_pasta = 2) (h_dinner_chicken : dinner_chicken = 2) (h_dinner_fish : dinner_fish = 1)
  (h_price_pasta: price_pasta = 12) (h_price_chicken: price_chicken = 15) (h_price_fish: price_fish = 18)
  (h_discount: discount = 0.10) 
  : max_income = 136.80 :=
  sorry

end NUMINAMATH_GPT_calculate_max_income_l2121_212175


namespace NUMINAMATH_GPT_union_of_A_and_B_l2121_212112

-- Definition of the sets A and B
def A : Set ℕ := {1, 2}
def B : Set ℕ := ∅

-- The theorem to prove
theorem union_of_A_and_B : A ∪ B = {1, 2} := 
by sorry

end NUMINAMATH_GPT_union_of_A_and_B_l2121_212112


namespace NUMINAMATH_GPT_at_least_100_arcs_of_21_points_l2121_212162

noncomputable def count_arcs (n : ℕ) (θ : ℝ) : ℕ :=
-- Please note this function needs to be defined appropriately, here we assume it computes the number of arcs of θ degrees or fewer between n points on a circle
sorry

theorem at_least_100_arcs_of_21_points :
  ∃ (n : ℕ), n = 21 ∧ count_arcs n (120 : ℝ) ≥ 100 :=
sorry

end NUMINAMATH_GPT_at_least_100_arcs_of_21_points_l2121_212162


namespace NUMINAMATH_GPT_square_segment_ratio_l2121_212179

theorem square_segment_ratio
  (A B C D E M P Q : ℝ × ℝ)
  (h_square: A = (0, 16) ∧ B = (16, 16) ∧ C = (16, 0) ∧ D = (0, 0))
  (h_E: E = (7, 0))
  (h_midpoint: M = ((0 + 7) / 2, (16 + 0) / 2))
  (h_bisector_P: P = (P.1, 16) ∧ (16 - 8 = (7 / 16) * (P.1 - 3.5)))
  (h_bisector_Q: Q = (Q.1, 0) ∧ (0 - 8 = (7 / 16) * (Q.1 - 3.5)))
  (h_PM: abs (16 - 8) = abs (P.2 - M.2))
  (h_MQ: abs (8 - 0) = abs (M.2 - Q.2)) :
  abs (P.2 - M.2) = abs (M.2 - Q.2) :=
sorry

end NUMINAMATH_GPT_square_segment_ratio_l2121_212179


namespace NUMINAMATH_GPT_area_of_fourth_rectangle_l2121_212141

theorem area_of_fourth_rectangle
  (A1 A2 A3 A_total : ℕ)
  (h1 : A1 = 24)
  (h2 : A2 = 30)
  (h3 : A3 = 18)
  (h_total : A_total = 100) :
  ∃ A4 : ℕ, A1 + A2 + A3 + A4 = A_total ∧ A4 = 28 :=
by
  sorry

end NUMINAMATH_GPT_area_of_fourth_rectangle_l2121_212141


namespace NUMINAMATH_GPT_number_of_students_l2121_212147

theorem number_of_students (N : ℕ) (h1 : (1/5 : ℚ) * N + (1/4 : ℚ) * N + (1/2 : ℚ) * N + 5 = N) : N = 100 :=
by
  sorry

end NUMINAMATH_GPT_number_of_students_l2121_212147


namespace NUMINAMATH_GPT_oplus_self_twice_l2121_212148

def my_oplus (x y : ℕ) := 3^x - y

theorem oplus_self_twice (a : ℕ) : my_oplus a (my_oplus a a) = a := by
  sorry

end NUMINAMATH_GPT_oplus_self_twice_l2121_212148


namespace NUMINAMATH_GPT_factorize_polynomial_l2121_212159

theorem factorize_polynomial {x : ℝ} : x^3 + 2 * x^2 - 3 * x = x * (x + 3) * (x - 1) :=
by sorry

end NUMINAMATH_GPT_factorize_polynomial_l2121_212159


namespace NUMINAMATH_GPT_green_notebook_cost_l2121_212173

-- Define the conditions
def num_notebooks : Nat := 4
def num_green_notebooks : Nat := 2
def num_black_notebooks : Nat := 1
def num_pink_notebooks : Nat := 1
def total_cost : ℕ := 45
def black_notebook_cost : ℕ := 15
def pink_notebook_cost : ℕ := 10

-- Define what we need to prove: The cost of each green notebook
def green_notebook_cost_each : ℕ := 10

-- The statement that combines the conditions with the goal to prove
theorem green_notebook_cost : 
  num_notebooks = 4 ∧ 
  num_green_notebooks = 2 ∧ 
  num_black_notebooks = 1 ∧ 
  num_pink_notebooks = 1 ∧ 
  total_cost = 45 ∧ 
  black_notebook_cost = 15 ∧ 
  pink_notebook_cost = 10 →
  2 * green_notebook_cost_each = total_cost - (black_notebook_cost + pink_notebook_cost) :=
by
  sorry

end NUMINAMATH_GPT_green_notebook_cost_l2121_212173


namespace NUMINAMATH_GPT_function_identity_l2121_212172

theorem function_identity {f : ℕ → ℕ} (h₀ : f 1 > 0) 
  (h₁ : ∀ m n : ℕ, f (m^2 + n^2) = f m^2 + f n^2) : 
  ∀ n : ℕ, f n = n :=
by
  sorry

end NUMINAMATH_GPT_function_identity_l2121_212172


namespace NUMINAMATH_GPT_division_remainder_l2121_212105

def p (x : ℝ) := x^5 + 2 * x^3 - x + 4
def a : ℝ := 2
def remainder : ℝ := 50

theorem division_remainder :
  p a = remainder :=
sorry

end NUMINAMATH_GPT_division_remainder_l2121_212105


namespace NUMINAMATH_GPT_range_of_c_l2121_212185

variable {a c : ℝ}

theorem range_of_c (h : a ≥ 1 / 8) (sufficient_but_not_necessary : ∀ x > 0, 2 * x + a / x ≥ c) : c ≤ 1 := 
sorry

end NUMINAMATH_GPT_range_of_c_l2121_212185


namespace NUMINAMATH_GPT_max_value_fraction_squares_l2121_212108

-- Let x and y be positive real numbers
variable (x y : ℝ)
variable (hx : 0 < x)
variable (hy : 0 < y)

theorem max_value_fraction_squares (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (∃ k, (x + 2 * y)^2 / (x^2 + y^2) ≤ k) ∧ (∀ z, (x + 2 * y)^2 / (x^2 + y^2) ≤ z) → k = 9 / 2 :=
by
  sorry

end NUMINAMATH_GPT_max_value_fraction_squares_l2121_212108


namespace NUMINAMATH_GPT_average_weight_children_l2121_212107

theorem average_weight_children 
  (n_boys : ℕ)
  (w_boys : ℕ)
  (avg_w_boys : ℕ)
  (n_girls : ℕ)
  (w_girls : ℕ)
  (avg_w_girls : ℕ)
  (h1 : n_boys = 8)
  (h2 : avg_w_boys = 140)
  (h3 : n_girls = 6)
  (h4 : avg_w_girls = 130)
  (h5 : w_boys = n_boys * avg_w_boys)
  (h6 : w_girls = n_girls * avg_w_girls)
  (total_w : ℕ)
  (h7 : total_w = w_boys + w_girls)
  (avg_w : ℚ)
  (h8 : avg_w = total_w / (n_boys + n_girls)) :
  avg_w = 135 :=
by
  sorry

end NUMINAMATH_GPT_average_weight_children_l2121_212107


namespace NUMINAMATH_GPT_books_fraction_sold_l2121_212194

theorem books_fraction_sold (B : ℕ) (h1 : B - 36 * 2 = 144) :
  (B - 36) / B = 2 / 3 := by
  sorry

end NUMINAMATH_GPT_books_fraction_sold_l2121_212194


namespace NUMINAMATH_GPT_xiaoming_grade_is_89_l2121_212114

noncomputable def xiaoming_physical_education_grade
  (extra_activity_score : ℕ) (midterm_score : ℕ) (final_exam_score : ℕ)
  (ratio_extra : ℕ) (ratio_mid : ℕ) (ratio_final : ℕ) : ℝ :=
  (extra_activity_score * ratio_extra + midterm_score * ratio_mid + final_exam_score * ratio_final) / (ratio_extra + ratio_mid + ratio_final)

theorem xiaoming_grade_is_89 :
  xiaoming_physical_education_grade 95 90 85 2 4 4 = 89 := by
    sorry

end NUMINAMATH_GPT_xiaoming_grade_is_89_l2121_212114


namespace NUMINAMATH_GPT_value_of_a2018_l2121_212191

noncomputable def a : ℕ → ℝ
| 0       => 2
| (n + 1) => (1 + a n) / (1 - a n)

theorem value_of_a2018 : a 2017 = -3 := sorry

end NUMINAMATH_GPT_value_of_a2018_l2121_212191


namespace NUMINAMATH_GPT_solve_fractional_equation_l2121_212113

theorem solve_fractional_equation : ∀ x : ℝ, (2 * x + 1) / 5 - x / 10 = 2 → x = 6 :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_solve_fractional_equation_l2121_212113


namespace NUMINAMATH_GPT_common_real_solution_unique_y_l2121_212139

theorem common_real_solution_unique_y (x y : ℝ) 
  (h1 : x^2 + y^2 = 16) 
  (h2 : x^2 - 3 * y + 12 = 0) : 
  y = 4 :=
by
  sorry

end NUMINAMATH_GPT_common_real_solution_unique_y_l2121_212139


namespace NUMINAMATH_GPT_find_a_plus_b_l2121_212118

theorem find_a_plus_b (a b : ℚ)
  (h1 : 3 = a + b / (2^2 + 1))
  (h2 : 2 = a + b / (1^2 + 1)) :
  a + b = 1 / 3 := 
sorry

end NUMINAMATH_GPT_find_a_plus_b_l2121_212118


namespace NUMINAMATH_GPT_prove_dollar_op_l2121_212138

variable {a b x y : ℝ}

def dollar_op (a b : ℝ) : ℝ := (a - b) ^ 2

theorem prove_dollar_op :
  dollar_op (x^2 - y^2) (y^2 - x^2) = 4 * (x^4 - 2 * x^2 * y^2 + y^4) := by
  sorry

end NUMINAMATH_GPT_prove_dollar_op_l2121_212138


namespace NUMINAMATH_GPT_nature_of_roots_indeterminate_l2121_212189

variable (a b c : ℝ)
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

theorem nature_of_roots_indeterminate (h : b^2 - 4 * a * c = 0) : 
  ((b + 2) ^ 2 - 4 * (a + 1) * (c + 1) = 0) ∨ ((b + 2) ^ 2 - 4 * (a + 1) * (c + 1) < 0) ∨ ((b + 2) ^ 2 - 4 * (a + 1) * (c + 1) > 0) :=
sorry

end NUMINAMATH_GPT_nature_of_roots_indeterminate_l2121_212189


namespace NUMINAMATH_GPT_grocer_pounds_of_bananas_purchased_l2121_212152

/-- 
Given:
1. The grocer purchased bananas at a rate of 3 pounds for $0.50.
2. The grocer sold the entire quantity at a rate of 4 pounds for $1.00.
3. The profit from selling the bananas was $11.00.

Prove that the number of pounds of bananas the grocer purchased is 132. 
-/
theorem grocer_pounds_of_bananas_purchased (P : ℕ) 
    (h1 : ∃ P, (3 * P / 0.5) - (4 * P / 1.0) = 11) : 
    P = 132 := 
sorry

end NUMINAMATH_GPT_grocer_pounds_of_bananas_purchased_l2121_212152


namespace NUMINAMATH_GPT_small_boxes_in_large_box_l2121_212121

def number_of_chocolate_bars_in_small_box := 25
def total_number_of_chocolate_bars := 375

theorem small_boxes_in_large_box : total_number_of_chocolate_bars / number_of_chocolate_bars_in_small_box = 15 := by
  sorry

end NUMINAMATH_GPT_small_boxes_in_large_box_l2121_212121


namespace NUMINAMATH_GPT_simplify_expression_l2121_212164

theorem simplify_expression (a : ℚ) (h : a^2 - a - 7/2 = 0) : 
  a^2 - (a - (2 * a) / (a + 1)) / ((a^2 - 2 * a + 1) / (a^2 - 1)) = 7 / 2 := 
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2121_212164


namespace NUMINAMATH_GPT_kocourkov_coins_l2121_212169

theorem kocourkov_coins :
  ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ 
  (∀ n > 53, ∃ x y : ℕ, n = x * a + y * b) ∧ 
  ¬ (∃ x y : ℕ, 53 = x * a + y * b) ∧
  ((a = 2 ∧ b = 55) ∨ (a = 3 ∧ b = 28)) :=
by {
  sorry
}

end NUMINAMATH_GPT_kocourkov_coins_l2121_212169


namespace NUMINAMATH_GPT_volume_of_prism_l2121_212163

variables (a b : ℝ) (α β : ℝ)
  (h1 : a > b)
  (h2 : 0 < α ∧ α < π / 2)
  (h3 : 0 < β ∧ β < π / 2)

noncomputable def volume_prism : ℝ :=
  (a^2 - b^2) * (a - b) / 8 * (Real.tan α)^2 * Real.tan β

theorem volume_of_prism (a b α β : ℝ) (h1 : a > b) (h2 : 0 < α ∧ α < π / 2) (h3 : 0 < β ∧ β < π / 2) :
  volume_prism a b α β = (a^2 - b^2) * (a - b) / 8 * (Real.tan α)^2 * Real.tan β := by
  sorry

end NUMINAMATH_GPT_volume_of_prism_l2121_212163


namespace NUMINAMATH_GPT_final_number_is_correct_l2121_212151

-- Define the problem conditions as Lean definitions/statements
def original_number : ℤ := 4
def doubled_number (x : ℤ) : ℤ := 2 * x
def resultant_number (x : ℤ) : ℤ := doubled_number x + 9
def final_number (x : ℤ) : ℤ := 3 * resultant_number x

-- Formulate the theorem using the conditions
theorem final_number_is_correct :
  final_number original_number = 51 :=
by
  sorry

end NUMINAMATH_GPT_final_number_is_correct_l2121_212151


namespace NUMINAMATH_GPT_friends_total_candies_l2121_212165

noncomputable def total_candies (T S J C V B : ℕ) : ℕ :=
  T + S + J + C + V + B

theorem friends_total_candies :
  let T := 22
  let S := 16
  let J := T / 2
  let C := 2 * S
  let V := J + S
  let B := (T + C) / 2 + 9
  total_candies T S J C V B = 144 := by
  sorry

end NUMINAMATH_GPT_friends_total_candies_l2121_212165


namespace NUMINAMATH_GPT_max_points_on_circle_l2121_212155

noncomputable def circleMaxPoints (P C : ℝ × ℝ) (r1 r2 d : ℝ) : ℕ :=
  if d = r1 + r2 ∨ d = abs (r1 - r2) then 1 else 
  if d < r1 + r2 ∧ d > abs (r1 - r2) then 2 else 0

theorem max_points_on_circle (P : ℝ × ℝ) (C : ℝ × ℝ) :
  let rC := 5
  let distPC := 9
  let rP := 4
  circleMaxPoints P C rC rP distPC = 1 :=
by sorry

end NUMINAMATH_GPT_max_points_on_circle_l2121_212155


namespace NUMINAMATH_GPT_complex_number_solution_l2121_212145

theorem complex_number_solution (z : ℂ) (i : ℂ) (hi : i * i = -1) (hz : z * (i - 1) = 2 * i) : 
z = 1 - i :=
by 
  sorry

end NUMINAMATH_GPT_complex_number_solution_l2121_212145


namespace NUMINAMATH_GPT_specific_value_eq_l2121_212199

def specific_value (x : ℕ) : ℕ := 25 * x

theorem specific_value_eq : specific_value 27 = 675 := by
  sorry

end NUMINAMATH_GPT_specific_value_eq_l2121_212199


namespace NUMINAMATH_GPT_sum_of_cubes_of_consecutive_integers_l2121_212119

theorem sum_of_cubes_of_consecutive_integers (n : ℕ) (h : (n-1)^2 + n^2 + (n+1)^2 = 8450) : 
  (n-1)^3 + n^3 + (n+1)^3 = 446949 := 
sorry

end NUMINAMATH_GPT_sum_of_cubes_of_consecutive_integers_l2121_212119


namespace NUMINAMATH_GPT_range_of_a_l2121_212190

theorem range_of_a {a : ℝ} (h : ∀ x : ℝ, a * x^2 + a * x + 1 > 0) : a ∈ Set.Icc 0 4 :=
sorry

end NUMINAMATH_GPT_range_of_a_l2121_212190


namespace NUMINAMATH_GPT_polar_not_one_to_one_correspondence_l2121_212117

theorem polar_not_one_to_one_correspondence :
  ¬ ∃ f : ℝ × ℝ → ℝ × ℝ, (∀ p1 p2 : ℝ × ℝ, f p1 = f p2 → p1 = p2) ∧
  (∀ q : ℝ × ℝ, ∃ p : ℝ × ℝ, q = f p) :=
by
  sorry

end NUMINAMATH_GPT_polar_not_one_to_one_correspondence_l2121_212117


namespace NUMINAMATH_GPT_certain_number_is_correct_l2121_212158

theorem certain_number_is_correct (x : ℝ) (h : x / 1.45 = 17.5) : x = 25.375 :=
sorry

end NUMINAMATH_GPT_certain_number_is_correct_l2121_212158


namespace NUMINAMATH_GPT_value_of_k_l2121_212122

-- Let k be a real number
variable (k : ℝ)

-- The given condition as a hypothesis
def condition := ∀ x : ℝ, (x + 3) * (x + 2) = k + 3 * x

-- The statement to prove
theorem value_of_k (h : ∀ x : ℝ, (x + 3) * (x + 2) = k + 3 * x) : k = 5 :=
sorry

end NUMINAMATH_GPT_value_of_k_l2121_212122


namespace NUMINAMATH_GPT_greatest_product_of_two_integers_sum_2006_l2121_212129

theorem greatest_product_of_two_integers_sum_2006 :
  ∃ (x y : ℤ), x + y = 2006 ∧ x * y = 1006009 :=
by
  sorry

end NUMINAMATH_GPT_greatest_product_of_two_integers_sum_2006_l2121_212129


namespace NUMINAMATH_GPT_find_k_and_b_l2121_212103

noncomputable def setA := {p : ℝ × ℝ | p.2^2 - p.1 - 1 = 0}
noncomputable def setB := {p : ℝ × ℝ | 4 * p.1^2 + 2 * p.1 - 2 * p.2 + 5 = 0}
noncomputable def setC (k b : ℝ) := {p : ℝ × ℝ | p.2 = k * p.1 + b}

theorem find_k_and_b (k b : ℕ) : 
  (setA ∪ setB) ∩ setC k b = ∅ ↔ (k = 1 ∧ b = 2) := 
sorry

end NUMINAMATH_GPT_find_k_and_b_l2121_212103


namespace NUMINAMATH_GPT_rowing_distance_l2121_212110

theorem rowing_distance
  (rowing_speed : ℝ)
  (current_speed : ℝ)
  (total_time : ℝ)
  (D : ℝ)
  (h1 : rowing_speed = 10)
  (h2 : current_speed = 2)
  (h3 : total_time = 15)
  (h4 : D / (rowing_speed + current_speed) + D / (rowing_speed - current_speed) = total_time) :
  D = 72 := 
sorry

end NUMINAMATH_GPT_rowing_distance_l2121_212110


namespace NUMINAMATH_GPT_binom_25_5_l2121_212156

theorem binom_25_5 :
  (Nat.choose 23 3 = 1771) ∧
  (Nat.choose 23 4 = 8855) ∧
  (Nat.choose 23 5 = 33649) → 
  Nat.choose 25 5 = 53130 := by
sorry

end NUMINAMATH_GPT_binom_25_5_l2121_212156


namespace NUMINAMATH_GPT_anicka_savings_l2121_212176

theorem anicka_savings (x y : ℕ) (h1 : x + y = 290) (h2 : (1/4 : ℚ) * (2 * y) = (1/3 : ℚ) * x) : 2 * y + x = 406 :=
by
  sorry

end NUMINAMATH_GPT_anicka_savings_l2121_212176


namespace NUMINAMATH_GPT_sum_of_power_of_2_plus_1_divisible_by_3_iff_odd_l2121_212116

theorem sum_of_power_of_2_plus_1_divisible_by_3_iff_odd (n : ℕ) : 
  (3 ∣ (2^n + 1)) ↔ (n % 2 = 1) :=
sorry

end NUMINAMATH_GPT_sum_of_power_of_2_plus_1_divisible_by_3_iff_odd_l2121_212116


namespace NUMINAMATH_GPT_average_speed_correct_l2121_212178

noncomputable def average_speed (initial_odometer : ℝ) (lunch_odometer : ℝ) (final_odometer : ℝ) (total_time : ℝ) : ℝ :=
  (final_odometer - initial_odometer) / total_time

theorem average_speed_correct :
  average_speed 212.3 372 467.2 6.25 = 40.784 :=
by
  unfold average_speed
  sorry

end NUMINAMATH_GPT_average_speed_correct_l2121_212178


namespace NUMINAMATH_GPT_proof_for_y_l2121_212177

theorem proof_for_y (x y : ℝ) (h1 : 3 * x^2 + 5 * x + 4 * y + 2 = 0) (h2 : 3 * x + y + 4 = 0) : 
  y^2 + 15 * y + 2 = 0 :=
sorry

end NUMINAMATH_GPT_proof_for_y_l2121_212177


namespace NUMINAMATH_GPT_tan_lt_neg_one_implies_range_l2121_212132

theorem tan_lt_neg_one_implies_range {x : ℝ} (h1 : 0 < x) (h2 : x < π) (h3 : Real.tan x < -1) :
  (π / 2 < x) ∧ (x < 3 * π / 4) :=
sorry

end NUMINAMATH_GPT_tan_lt_neg_one_implies_range_l2121_212132


namespace NUMINAMATH_GPT_emily_cell_phone_cost_l2121_212104

noncomputable def base_cost : ℝ := 25
noncomputable def included_hours : ℝ := 25
noncomputable def cost_per_text : ℝ := 0.1
noncomputable def cost_per_extra_minute : ℝ := 0.15
noncomputable def cost_per_gigabyte : ℝ := 2

noncomputable def emily_texts : ℝ := 150
noncomputable def emily_hours : ℝ := 26
noncomputable def emily_data : ℝ := 3

theorem emily_cell_phone_cost : 
  let texts_cost := emily_texts * cost_per_text
  let extra_minutes_cost := (emily_hours - included_hours) * 60 * cost_per_extra_minute
  let data_cost := emily_data * cost_per_gigabyte
  base_cost + texts_cost + extra_minutes_cost + data_cost = 55 := by
  sorry

end NUMINAMATH_GPT_emily_cell_phone_cost_l2121_212104


namespace NUMINAMATH_GPT_gain_amount_is_ten_l2121_212187

theorem gain_amount_is_ten (S : ℝ) (C : ℝ) (g : ℝ) (G : ℝ) 
  (h1 : S = 110) (h2 : g = 0.10) (h3 : S = C + g * C) : G = 10 :=
by 
  sorry

end NUMINAMATH_GPT_gain_amount_is_ten_l2121_212187


namespace NUMINAMATH_GPT_convert_to_cylindrical_l2121_212127

theorem convert_to_cylindrical (x y z : ℝ) (r θ : ℝ) 
  (h₀ : x = 3) 
  (h₁ : y = -3 * Real.sqrt 3) 
  (h₂ : z = 2) 
  (h₃ : r > 0) 
  (h₄ : 0 ≤ θ) 
  (h₅ : θ < 2 * Real.pi) 
  (h₆ : r = Real.sqrt (x^2 + y^2)) 
  (h₇ : x = r * Real.cos θ) 
  (h₈ : y = r * Real.sin θ) : 
  (r, θ, z) = (6, 5 * Real.pi / 3, 2) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_convert_to_cylindrical_l2121_212127


namespace NUMINAMATH_GPT_triangle_angles_in_given_ratio_l2121_212161

theorem triangle_angles_in_given_ratio (x : ℝ) (y : ℝ) (z : ℝ) (h : x + y + z = 180) (r : x / 1 = y / 4 ∧ y / 4 = z / 7) : 
  x = 15 ∧ y = 60 ∧ z = 105 :=
by
  sorry

end NUMINAMATH_GPT_triangle_angles_in_given_ratio_l2121_212161


namespace NUMINAMATH_GPT_diamond_not_commutative_diamond_not_associative_l2121_212115

noncomputable def diamond (x y : ℝ) : ℝ :=
  x^2 * y / (x + y + 1)

theorem diamond_not_commutative (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  x ≠ y → diamond x y ≠ diamond y x :=
by
  intro hxy
  unfold diamond
  intro h
  -- Assume the contradiction leads to this equality not holding
  have eq : x^2 * y * (y + x + 1) = y^2 * x * (x + y + 1) := by
    sorry
  -- Simplify the equation to show the contradiction
  sorry

theorem diamond_not_associative (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (diamond x y) ≠ (diamond y x) → (diamond (diamond x y) z) ≠ (diamond x (diamond y z)) :=
by
  unfold diamond
  intro h
  -- Assume the contradiction leads to this equality not holding
  have eq : (diamond x y)^2 * z / (diamond x y + z + 1) ≠ (x^2 * (diamond y z) / (x + diamond y z + 1)) :=
    by sorry
  -- Simplify the equation to show the contradiction
  sorry

end NUMINAMATH_GPT_diamond_not_commutative_diamond_not_associative_l2121_212115


namespace NUMINAMATH_GPT_find_m_n_l2121_212182

-- Define the set A
def set_A : Set ℝ := {x | |x + 2| < 3}

-- Define the set B in terms of a variable m
def set_B (m : ℝ) : Set ℝ := {x | (x - m) * (x - 2) < 0}

-- State the theorem
theorem find_m_n (m n : ℝ) (hA : set_A = {x | -5 < x ∧ x < 1}) (h_inter : set_A ∩ set_B m = {x | -1 < x ∧ x < n}) : 
  m = -1 ∧ n = 1 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_find_m_n_l2121_212182


namespace NUMINAMATH_GPT_surface_area_of_solid_l2121_212100

-- Definitions about the problem
def is_prime (n : ℕ) : Prop := Nat.Prime n
def is_rectangular_solid (a b c : ℕ) : Prop := is_prime a ∧ is_prime b ∧ is_prime c ∧ (a * b * c = 399)

-- Main statement of the problem
theorem surface_area_of_solid (a b c : ℕ) (h : is_rectangular_solid a b c) : 
  2 * (a * b + b * c + c * a) = 422 := sorry

end NUMINAMATH_GPT_surface_area_of_solid_l2121_212100


namespace NUMINAMATH_GPT_line_segment_length_l2121_212126

theorem line_segment_length (x : ℝ) (h : x > 0) :
  (Real.sqrt ((x - 2)^2 + (6 - 2)^2) = 5) → (x = 5) :=
by
  intro h1
  sorry

end NUMINAMATH_GPT_line_segment_length_l2121_212126


namespace NUMINAMATH_GPT_gcd_1728_1764_l2121_212142

theorem gcd_1728_1764 : Int.gcd 1728 1764 = 36 := by
  sorry

end NUMINAMATH_GPT_gcd_1728_1764_l2121_212142


namespace NUMINAMATH_GPT_hyperbola_eq_l2121_212166

theorem hyperbola_eq (a b c : ℝ) (h1 : a > 0) (h2 : b > 0)
  (hyp_eq : ∀ x y, (x ^ 2 / a ^ 2) - (y ^ 2 / b ^ 2) = 1)
  (asymptote : b / a = Real.sqrt 3)
  (focus_parabola : c = 4) : 
  a^2 = 4 ∧ b^2 = 12 := by
sorry

end NUMINAMATH_GPT_hyperbola_eq_l2121_212166


namespace NUMINAMATH_GPT_find_a_l2121_212102

def system_of_equations (a x y : ℝ) : Prop :=
  y - 2 = a * (x - 4) ∧ (2 * x) / (|y| + y) = Real.sqrt x

def domain_constraints (x y : ℝ) : Prop :=
  y > 0 ∧ x ≥ 0

def valid_a (a : ℝ) : Prop :=
  (∃ x y, domain_constraints x y ∧ system_of_equations a x y)

theorem find_a :
  ∀ a : ℝ, valid_a a ↔
  ((a < 0.5 ∧ ∃ y, y = 2 - 4 * a ∧ y > 0) ∨ 
   (∃ x y, x = 4 ∧ y = 2 ∧ x ≥ 0 ∧ y > 0) ∨
   (0 < a ∧ a ≠ 0.25 ∧ a < 0.5 ∧ ∃ x y, x = (1 - 2 * a) / a ∧ y = (1 - 2 * a) / a)) :=
by sorry

end NUMINAMATH_GPT_find_a_l2121_212102


namespace NUMINAMATH_GPT_sequence_next_term_l2121_212195

theorem sequence_next_term (a b c d e : ℕ) (h1 : a = 34) (h2 : b = 45) (h3 : c = 56) (h4 : d = 67) (h5 : e = 78) (h6 : b = a + 11) (h7 : c = b + 11) (h8 : d = c + 11) (h9 : e = d + 11) : e + 11 = 89 :=
by
  sorry

end NUMINAMATH_GPT_sequence_next_term_l2121_212195


namespace NUMINAMATH_GPT_number_to_remove_l2121_212153

def mean (s : List ℕ) : ℚ := (s.sum : ℚ) / s.length

theorem number_to_remove (s : List ℕ) (x : ℕ) 
  (h₀ : s = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
  (h₁ : x ∈ s)
  (h₂ : mean (List.erase s x) = 6.1) : x = 5 := sorry

end NUMINAMATH_GPT_number_to_remove_l2121_212153


namespace NUMINAMATH_GPT_divide_cakes_l2121_212188

/-- Statement: Eleven cakes can be divided equally among six girls without cutting any cake into 
exactly six equal parts such that each girl receives 1 + 1/2 + 1/4 + 1/12 cakes -/
theorem divide_cakes (cakes girls : ℕ) (h_cakes : cakes = 11) (h_girls : girls = 6) :
  ∃ (parts : ℕ → ℝ), (∀ i, parts i = 1 + 1 / 2 + 1 / 4 + 1 / 12) ∧ (cakes = girls * (1 + 1 / 2 + 1 / 4 + 1 / 12)) :=
by
  sorry

end NUMINAMATH_GPT_divide_cakes_l2121_212188


namespace NUMINAMATH_GPT_integer_solutions_to_inequality_l2121_212196

noncomputable def count_integer_solutions (n : ℕ) : ℕ :=
1 + 2 * n^2 + 2 * n

theorem integer_solutions_to_inequality (n : ℕ) :
  ∃ (count : ℕ), count = count_integer_solutions n ∧ 
  ∀ (x y : ℤ), |x| + |y| ≤ n → (∃ (k : ℕ), k = count) :=
by
  sorry

end NUMINAMATH_GPT_integer_solutions_to_inequality_l2121_212196


namespace NUMINAMATH_GPT_people_distribution_l2121_212183

theorem people_distribution (x : ℕ) (h1 : x > 5):
  100 / (x - 5) = 150 / x :=
sorry

end NUMINAMATH_GPT_people_distribution_l2121_212183


namespace NUMINAMATH_GPT_compute_f_at_919_l2121_212157

-- Given conditions
def is_even_function (f : ℝ → ℝ) : Prop :=
∀ x, f x = f (-x)

def periodic_function (f : ℝ → ℝ) : Prop :=
∀ x, f (x + 4) = f (x - 2)

noncomputable def f (x : ℝ) : ℝ :=
if h : x ∈ [-3, 0] then 6^(-x) else sorry

-- Lean statement for the proof problem
theorem compute_f_at_919 (f : ℝ → ℝ)
    (h_even : is_even_function f)
    (h_periodic : periodic_function f)
    (h_defined : ∀ x ∈ [-3, 0], f x = 6^(-x)) :
    f 919 = 6 := sorry

end NUMINAMATH_GPT_compute_f_at_919_l2121_212157


namespace NUMINAMATH_GPT_range_of_B_l2121_212146

theorem range_of_B (a b c : ℝ) (h : a + c = 2 * b) :
  ∃ B : ℝ, 0 < B ∧ B ≤ π / 3 ∧
  ∃ A C : ℝ, ∃ ha : a = c, 
    a > 0 ∧ b > 0 ∧ c > 0 ∧ A + B + C = π :=
sorry

end NUMINAMATH_GPT_range_of_B_l2121_212146


namespace NUMINAMATH_GPT_quadratic_has_two_zeros_l2121_212171

theorem quadratic_has_two_zeros {a b c : ℝ} (h : a * c < 0) : 
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_has_two_zeros_l2121_212171


namespace NUMINAMATH_GPT_determine_value_of_c_l2121_212135

theorem determine_value_of_c (b : ℝ) (h₁ : ∀ x : ℝ, 0 ≤ x^2 + x + b) (h₂ : ∃ m : ℝ, ∀ x : ℝ, x^2 + x + b < c ↔ x = m + 8) : 
    c = 16 :=
sorry

end NUMINAMATH_GPT_determine_value_of_c_l2121_212135


namespace NUMINAMATH_GPT_solve_A_solve_area_l2121_212137

noncomputable def angle_A (A : ℝ) : Prop :=
  2 * (Real.cos (A / 2))^2 + Real.cos A = 0

noncomputable def area_triangle (a b c : ℝ) (A : ℝ) : Prop :=
  a = 2 * Real.sqrt 3 → b + c = 4 → A = 2 * Real.pi / 3 → 
  (1/2) * b * c * Real.sin A = Real.sqrt 3

theorem solve_A (A : ℝ) : angle_A A → A = 2 * Real.pi / 3 :=
sorry

theorem solve_area (a b c A S : ℝ) : 
  a = 2 * Real.sqrt 3 →
  b + c = 4 →
  A = 2 * Real.pi / 3 →
  area_triangle a b c A →
  S = Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_solve_A_solve_area_l2121_212137


namespace NUMINAMATH_GPT_value_division_l2121_212134

theorem value_division (x : ℝ) (h1 : 54 / x = 54 - 36) : x = 3 := by
  sorry

end NUMINAMATH_GPT_value_division_l2121_212134


namespace NUMINAMATH_GPT_smallest_n_in_T_and_largest_N_not_in_T_l2121_212140

def T : Set ℝ := {y | ∃ x : ℝ, x ≥ 0 ∧ y = (3 * x + 4) / (x + 3)}

theorem smallest_n_in_T_and_largest_N_not_in_T :
  (∀ n, n = 4 / 3 → n ∈ T) ∧ (∀ N, N = 3 → N ∉ T) :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_in_T_and_largest_N_not_in_T_l2121_212140


namespace NUMINAMATH_GPT_painting_rate_l2121_212170

/-- Define various dimensions and costs for the room -/
def room_length : ℝ := 10
def room_width  : ℝ := 7
def room_height : ℝ := 5

def door_width  : ℝ := 1
def door_height : ℝ := 3
def num_doors   : ℕ := 2

def large_window_width  : ℝ := 2
def large_window_height : ℝ := 1.5
def num_large_windows   : ℕ := 1

def small_window_width  : ℝ := 1
def small_window_height : ℝ := 1.5
def num_small_windows   : ℕ := 2

def painting_cost : ℝ := 474

/-- The rate for painting the walls is Rs. 3 per sq m -/
theorem painting_rate : (painting_cost / 
  ((2 * (room_length * room_height) + 2 * (room_width * room_height)) -
   (num_doors * (door_width * door_height) +
    num_large_windows * (large_window_width * large_window_height) +
    num_small_windows * (small_window_width * small_window_height)))) = 3 := 
by 
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_painting_rate_l2121_212170


namespace NUMINAMATH_GPT_trajectory_of_N_l2121_212109

variables {x y x₀ y₀ : ℝ}

def F : ℝ × ℝ := (1, 0)

def M (x₀ : ℝ) : ℝ × ℝ := (x₀, 0)
def P (y₀ : ℝ) : ℝ × ℝ := (0, y₀)
def N (x y : ℝ) : ℝ × ℝ := (x, y)

def PM (x₀ y₀ : ℝ) : ℝ × ℝ := (x₀, -y₀)
def PF (y₀ : ℝ) : ℝ × ℝ := (1, -y₀)

def perpendicular (v1 v2 : ℝ × ℝ) := v1.fst * v2.fst + v1.snd * v2.snd = 0

def MN_eq_2MP (x y x₀ y₀ : ℝ) := ((x - x₀), y) = (2 * (-x₀), 2 * y₀)

theorem trajectory_of_N (h1 : perpendicular (PM x₀ y₀) (PF y₀))
  (h2 : MN_eq_2MP x y x₀ y₀) :
  y^2 = 4*x :=
by
  sorry

end NUMINAMATH_GPT_trajectory_of_N_l2121_212109


namespace NUMINAMATH_GPT_triangle_area_relation_l2121_212111

theorem triangle_area_relation :
  let A := (1 / 2) * 5 * 5
  let B := (1 / 2) * 12 * 12
  let C := (1 / 2) * 13 * 13
  A + B = C :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_relation_l2121_212111


namespace NUMINAMATH_GPT_total_value_is_correct_l2121_212167

-- Define the conditions from the problem
def totalCoins : Nat := 324
def twentyPaiseCoins : Nat := 220
def twentyPaiseValue : Nat := 20
def twentyFivePaiseValue : Nat := 25
def paiseToRupees : Nat := 100

-- Calculate the number of 25 paise coins
def twentyFivePaiseCoins : Nat := totalCoins - twentyPaiseCoins

-- Calculate the total value of 20 paise and 25 paise coins in paise
def totalValueInPaise : Nat :=
  (twentyPaiseCoins * twentyPaiseValue) + 
  (twentyFivePaiseCoins * twentyFivePaiseValue)

-- Convert the total value from paise to rupees
def totalValueInRupees : Nat := totalValueInPaise / paiseToRupees

-- The theorem to be proved
theorem total_value_is_correct : totalValueInRupees = 70 := by
  sorry

end NUMINAMATH_GPT_total_value_is_correct_l2121_212167
