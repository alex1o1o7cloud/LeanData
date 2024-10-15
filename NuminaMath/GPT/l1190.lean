import Mathlib

namespace NUMINAMATH_GPT_building_height_l1190_119006

theorem building_height (H : ℝ) 
                        (bounced_height : ℕ → ℝ) 
                        (h_bounce : ∀ n, bounced_height n = H / 2 ^ (n + 1)) 
                        (h_fifth : bounced_height 5 = 3) : 
    H = 96 := 
by {
  sorry
}

end NUMINAMATH_GPT_building_height_l1190_119006


namespace NUMINAMATH_GPT_almost_square_as_quotient_l1190_119067

-- Defining what almost squares are
def isAlmostSquare (k : ℕ) : Prop := ∃ n : ℕ, k = n * (n + 1)

-- Statement of the theorem
theorem almost_square_as_quotient (n : ℕ) (hn : n > 0) :
  ∃ a b : ℕ, isAlmostSquare a ∧ isAlmostSquare b ∧ n * (n + 1) = a / b := by
  sorry

end NUMINAMATH_GPT_almost_square_as_quotient_l1190_119067


namespace NUMINAMATH_GPT_cost_price_watch_l1190_119063

variable (cost_price : ℚ)

-- Conditions
def sold_at_loss (cost_price : ℚ) := 0.90 * cost_price
def sold_at_gain (cost_price : ℚ) := 1.03 * cost_price
def price_difference (cost_price : ℚ) := sold_at_gain cost_price - sold_at_loss cost_price = 140

-- Theorem
theorem cost_price_watch (h : price_difference cost_price) : cost_price = 1076.92 := by
  sorry

end NUMINAMATH_GPT_cost_price_watch_l1190_119063


namespace NUMINAMATH_GPT_shape_area_l1190_119092

-- Define the conditions as Lean definitions
def side_length : ℝ := 3
def num_squares : ℕ := 4

-- Prove that the area of the shape is 36 cm² given the conditions
theorem shape_area : num_squares * (side_length * side_length) = 36 := by
    -- The proof is skipped with sorry
    sorry

end NUMINAMATH_GPT_shape_area_l1190_119092


namespace NUMINAMATH_GPT_max_xy_l1190_119088

theorem max_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x / 3 + y / 4 = 1) : xy ≤ 3 :=
by {
  -- proof omitted
  sorry
}

end NUMINAMATH_GPT_max_xy_l1190_119088


namespace NUMINAMATH_GPT_find_n_l1190_119008

noncomputable
def equilateral_triangle_area_ratio (n : ℕ) (h : n > 4) : Prop :=
  let ratio := (2 : ℚ) / (n - 2 : ℚ)
  let area_PQR := (1 / 7 : ℚ)
  let menelaus_ap_pd := (n * (n - 2) : ℚ) / 4
  let area_triangle_ABP := (2 * (n - 2) : ℚ) / (n * (n - 2) + 4)
  let area_sum := 3 * area_triangle_ABP
  (area_sum * 7 = 6 * (n * (n - 2) + 4))

theorem find_n (n : ℕ) (h : n > 4) : 
  (equilateral_triangle_area_ratio n h) → n = 6 := sorry

end NUMINAMATH_GPT_find_n_l1190_119008


namespace NUMINAMATH_GPT_brother_paint_time_is_4_l1190_119048

noncomputable def brother_paint_time (B : ℝ) : Prop :=
  (1 / 3) + (1 / B) = 1 / 1.714

theorem brother_paint_time_is_4 : ∃ B, brother_paint_time B ∧ abs (B - 4) < 0.001 :=
by {
  sorry -- Proof to be filled in later.
}

end NUMINAMATH_GPT_brother_paint_time_is_4_l1190_119048


namespace NUMINAMATH_GPT_find_integer_l1190_119005

theorem find_integer (n : ℕ) (hn1 : n % 20 = 0) (hn2 : 8.2 < (n : ℝ)^(1/3)) (hn3 : (n : ℝ)^(1/3) < 8.3) : n = 560 := sorry

end NUMINAMATH_GPT_find_integer_l1190_119005


namespace NUMINAMATH_GPT_pencils_given_out_l1190_119055

-- Defining the conditions
def num_children : ℕ := 4
def pencils_per_child : ℕ := 2

-- Formulating the problem statement, with the goal to prove the total number of pencils
theorem pencils_given_out : num_children * pencils_per_child = 8 := 
by 
  sorry

end NUMINAMATH_GPT_pencils_given_out_l1190_119055


namespace NUMINAMATH_GPT_characteristic_triangle_smallest_angle_l1190_119089

theorem characteristic_triangle_smallest_angle 
  (α β : ℝ)
  (h1 : α = 2 * β)
  (h2 : α = 100)
  (h3 : β + α + γ = 180) : 
  min α (min β γ) = 30 := 
by 
  sorry

end NUMINAMATH_GPT_characteristic_triangle_smallest_angle_l1190_119089


namespace NUMINAMATH_GPT_max_value_of_linear_combination_l1190_119073

theorem max_value_of_linear_combination
  (x y : ℝ)
  (h : x^2 + y^2 = 16 * x + 8 * y + 10) :
  ∃ z, z = 4.58 ∧ (∀ x y, (4 * x + 3 * y) ≤ z ∧ (x^2 + y^2 = 16 * x + 8 * y + 10) → (4 * x + 3 * y) ≤ 4.58) :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_linear_combination_l1190_119073


namespace NUMINAMATH_GPT_determine_even_condition_l1190_119013

theorem determine_even_condition (x : ℤ) (m : ℤ) (h : m = x % 2) : m = 0 ↔ x % 2 = 0 :=
by sorry

end NUMINAMATH_GPT_determine_even_condition_l1190_119013


namespace NUMINAMATH_GPT_product_of_roots_l1190_119070

theorem product_of_roots (x1 x2 : ℝ) (h : x1 * x2 = -2) :
  (∀ x : ℝ, x^2 + x - 2 = 0 → (x = x1 ∨ x = x2)) → x1 * x2 = -2 :=
by
  intros h_root
  exact h

end NUMINAMATH_GPT_product_of_roots_l1190_119070


namespace NUMINAMATH_GPT_a_is_perfect_square_l1190_119042

variable (a b : ℕ)
variable (h1 : 0 < a) 
variable (h2 : 0 < b)
variable (h3 : b % 2 = 1)
variable (h4 : ∃ k : ℕ, (a + b) ^ 2 + 4 * a = k * a * b)

theorem a_is_perfect_square (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : b % 2 = 1) 
  (h4 : ∃ k : ℕ, (a + b) ^ 2 + 4 * a = k * a * b) : ∃ n : ℕ, a = n ^ 2 :=
sorry

end NUMINAMATH_GPT_a_is_perfect_square_l1190_119042


namespace NUMINAMATH_GPT_fourth_hexagon_dots_l1190_119045

   -- Define the number of dots in the first, second, and third hexagons
   def hexagon_dots (n : ℕ) : ℕ :=
     match n with
     | 1 => 1
     | 2 => 8
     | 3 => 22
     | 4 => 46
     | _ => 0

   -- State the theorem to be proved
   theorem fourth_hexagon_dots : hexagon_dots 4 = 46 :=
   by
     sorry
   
end NUMINAMATH_GPT_fourth_hexagon_dots_l1190_119045


namespace NUMINAMATH_GPT_total_cost_of_shirts_l1190_119095

theorem total_cost_of_shirts 
    (first_shirt_cost : ℤ)
    (second_shirt_cost : ℤ)
    (h1 : first_shirt_cost = 15)
    (h2 : first_shirt_cost = second_shirt_cost + 6) : 
    first_shirt_cost + second_shirt_cost = 24 := 
by
  sorry

end NUMINAMATH_GPT_total_cost_of_shirts_l1190_119095


namespace NUMINAMATH_GPT_math_problem_l1190_119094

theorem math_problem:
  (5 + 7) * (5^2 + 7^2) * (5^4 + 7^4) * (5^8 + 7^8) * (5^16 + 7^16) * (5^32 + 7^32) * (5^64 + 7^64) = 7^128 - 5^128 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l1190_119094


namespace NUMINAMATH_GPT_john_allowance_calculation_l1190_119093

theorem john_allowance_calculation (initial_money final_money game_cost allowance: ℕ) 
(h_initial: initial_money = 5) 
(h_game_cost: game_cost = 2) 
(h_final: final_money = 29) 
(h_allowance: final_money = initial_money - game_cost + allowance) : 
  allowance = 26 :=
by
  sorry

end NUMINAMATH_GPT_john_allowance_calculation_l1190_119093


namespace NUMINAMATH_GPT_ratio_turkeys_to_ducks_l1190_119084

theorem ratio_turkeys_to_ducks (chickens ducks turkeys total_birds : ℕ)
  (h1 : chickens = 200)
  (h2 : ducks = 2 * chickens)
  (h3 : total_birds = 1800)
  (h4 : total_birds = chickens + ducks + turkeys) :
  (turkeys : ℚ) / ducks = 3 := by
sorry

end NUMINAMATH_GPT_ratio_turkeys_to_ducks_l1190_119084


namespace NUMINAMATH_GPT_cupcakes_frosted_in_10_minutes_l1190_119004

def frosting_rate (time: ℕ) (cupcakes: ℕ) : ℚ := cupcakes / time

noncomputable def combined_frosting_rate : ℚ :=
  (frosting_rate 25 1) + (frosting_rate 35 1)

def effective_working_time (total_time: ℕ) (work_period: ℕ) (break_time: ℕ) : ℕ :=
  let break_intervals := total_time / work_period
  total_time - break_intervals * break_time

def total_cupcakes (working_time: ℕ) (rate: ℚ) : ℚ :=
  working_time * rate

theorem cupcakes_frosted_in_10_minutes :
  total_cupcakes (effective_working_time 600 240 30) combined_frosting_rate = 36 := by
  sorry

end NUMINAMATH_GPT_cupcakes_frosted_in_10_minutes_l1190_119004


namespace NUMINAMATH_GPT_find_b_minus_c_l1190_119017

noncomputable def a (n : ℕ) : ℝ :=
  if h : n > 1 then 1 / Real.log 1009 * Real.log n else 0

noncomputable def b : ℝ :=
  a 2 + a 3 + a 4 + a 5 + a 6

noncomputable def c : ℝ :=
  a 15 + a 16 + a 17 + a 18 + a 19

theorem find_b_minus_c : b - c = -Real.logb 1009 1938 := by
  sorry

end NUMINAMATH_GPT_find_b_minus_c_l1190_119017


namespace NUMINAMATH_GPT_fifty_three_days_from_friday_is_tuesday_l1190_119029

inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

open DayOfWeek

def dayAfter (d : DayOfWeek) (n : ℕ) : DayOfWeek :=
match n % 7 with
| 0 => d
| 1 => match d with
       | Sunday    => Monday
       | Monday    => Tuesday
       | Tuesday   => Wednesday
       | Wednesday => Thursday
       | Thursday  => Friday
       | Friday    => Saturday
       | Saturday  => Sunday
| 2 => match d with
       | Sunday    => Tuesday
       | Monday    => Wednesday
       | Tuesday   => Thursday
       | Wednesday => Friday
       | Thursday  => Saturday
       | Friday    => Sunday
       | Saturday  => Monday
| 3 => match d with
       | Sunday    => Wednesday
       | Monday    => Thursday
       | Tuesday   => Friday
       | Wednesday => Saturday
       | Thursday  => Sunday
       | Friday    => Monday
       | Saturday  => Tuesday
| 4 => match d with
       | Sunday    => Thursday
       | Monday    => Friday
       | Tuesday   => Saturday
       | Wednesday => Sunday
       | Thursday  => Monday
       | Friday    => Tuesday
       | Saturday  => Wednesday
| 5 => match d with
       | Sunday    => Friday
       | Monday    => Saturday
       | Tuesday   => Sunday
       | Wednesday => Monday
       | Thursday  => Tuesday
       | Friday    => Wednesday
       | Saturday  => Thursday
| 6 => match d with
       | Sunday    => Saturday
       | Monday    => Sunday
       | Tuesday   => Monday
       | Wednesday => Tuesday
       | Thursday  => Wednesday
       | Friday    => Thursday
       | Saturday  => Friday
| _ => d  -- although all cases are covered

theorem fifty_three_days_from_friday_is_tuesday :
  dayAfter Friday 53 = Tuesday :=
by
  sorry

end NUMINAMATH_GPT_fifty_three_days_from_friday_is_tuesday_l1190_119029


namespace NUMINAMATH_GPT_shirts_washed_total_l1190_119046

theorem shirts_washed_total (short_sleeve_shirts long_sleeve_shirts : Nat) (h1 : short_sleeve_shirts = 4) (h2 : long_sleeve_shirts = 5) : short_sleeve_shirts + long_sleeve_shirts = 9 := by
  sorry

end NUMINAMATH_GPT_shirts_washed_total_l1190_119046


namespace NUMINAMATH_GPT_probability_player_A_wins_first_B_wins_second_l1190_119011

theorem probability_player_A_wins_first_B_wins_second :
  (1 / 2) * (4 / 5) * (2 / 3) + (1 / 2) * (1 / 3) * (2 / 3) = 17 / 45 :=
by
  sorry

end NUMINAMATH_GPT_probability_player_A_wins_first_B_wins_second_l1190_119011


namespace NUMINAMATH_GPT_find_rate_of_interest_l1190_119044

theorem find_rate_of_interest (P SI : ℝ) (r : ℝ) (hP : P = 1200) (hSI : SI = 108) (ht : r = r) :
  SI = P * r * r / 100 → r = 3 := by
  intros
  sorry

end NUMINAMATH_GPT_find_rate_of_interest_l1190_119044


namespace NUMINAMATH_GPT_constant_term_2x3_minus_1_over_sqrtx_pow_7_l1190_119061

noncomputable def constant_term_in_expansion (n : ℕ) (x : ℝ) : ℝ :=
  (2 : ℝ) * (Nat.choose 7 6 : ℝ)

theorem constant_term_2x3_minus_1_over_sqrtx_pow_7 :
  constant_term_in_expansion 7 (2 : ℝ) = 14 :=
by
  -- proof is omitted
  sorry

end NUMINAMATH_GPT_constant_term_2x3_minus_1_over_sqrtx_pow_7_l1190_119061


namespace NUMINAMATH_GPT_isosceles_triangle_l1190_119086

theorem isosceles_triangle
  (a b c : ℝ)
  (A B C : ℝ)
  (h : a + b = (Real.tan (C / 2)) * (a * Real.tan A + b * Real.tan B)) :
  A = B := 
sorry

end NUMINAMATH_GPT_isosceles_triangle_l1190_119086


namespace NUMINAMATH_GPT_determine_c_l1190_119096

-- Assume we have three integers a, b, and unique x, y, z such that
variables (a b c x y z : ℕ)

-- Define the conditions
def condition1 : Prop := a = Nat.lcm y z
def condition2 : Prop := b = Nat.lcm x z
def condition3 : Prop := c = Nat.lcm x y

-- Prove that Bob can determine c based on a and b
theorem determine_c (h1 : condition1 a y z) (h2 : condition2 b x z) (h3 : ∀ u v w : ℕ, (Nat.lcm u w = a ∧ Nat.lcm v w = b ∧ Nat.lcm u v = c) → (u = x ∧ v = y ∧ w = z) ) : ∃ c, condition3 c x y :=
by sorry

end NUMINAMATH_GPT_determine_c_l1190_119096


namespace NUMINAMATH_GPT_julie_reads_tomorrow_l1190_119028

theorem julie_reads_tomorrow :
  let total_pages := 120
  let pages_read_yesterday := 12
  let pages_read_today := 2 * pages_read_yesterday
  let pages_read_so_far := pages_read_yesterday + pages_read_today
  let remaining_pages := total_pages - pages_read_so_far
  remaining_pages / 2 = 42 :=
by
  sorry

end NUMINAMATH_GPT_julie_reads_tomorrow_l1190_119028


namespace NUMINAMATH_GPT_min_major_axis_ellipse_l1190_119072

theorem min_major_axis_ellipse (a b c : ℝ) (h1 : b * c = 1) (h2 : a^2 = b^2 + c^2) :
  2 * a ≥ 2 * Real.sqrt 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_min_major_axis_ellipse_l1190_119072


namespace NUMINAMATH_GPT_equal_distribution_l1190_119001

theorem equal_distribution 
  (total_profit : ℕ) 
  (num_employees : ℕ) 
  (profit_kept_percent : ℕ) 
  (remaining_to_distribute : ℕ)
  (each_employee_gets : ℕ) :
  total_profit = 50 →
  num_employees = 9 →
  profit_kept_percent = 10 →
  remaining_to_distribute = total_profit - (total_profit * profit_kept_percent / 100) →
  each_employee_gets = remaining_to_distribute / num_employees →
  each_employee_gets = 5 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_equal_distribution_l1190_119001


namespace NUMINAMATH_GPT_triangle_internal_angle_60_l1190_119012

theorem triangle_internal_angle_60 (A B C : ℝ) (h_sum : A + B + C = 180) : A >= 60 ∨ B >= 60 ∨ C >= 60 :=
sorry

end NUMINAMATH_GPT_triangle_internal_angle_60_l1190_119012


namespace NUMINAMATH_GPT_intersection_of_lines_l1190_119081

theorem intersection_of_lines : 
  let x := (5 : ℚ) / 9
  let y := (5 : ℚ) / 3
  (y = 3 * x ∧ y - 5 = -6 * x) ↔ (x, y) = ((5 : ℚ) / 9, (5 : ℚ) / 3) := 
by 
  sorry

end NUMINAMATH_GPT_intersection_of_lines_l1190_119081


namespace NUMINAMATH_GPT_find_principal_amount_l1190_119076

theorem find_principal_amount (P r : ℝ) 
    (h1 : 815 - P = P * r * 3) 
    (h2 : 850 - P = P * r * 4) : 
    P = 710 :=
by
  -- proof steps will go here
  sorry

end NUMINAMATH_GPT_find_principal_amount_l1190_119076


namespace NUMINAMATH_GPT_recreation_spending_l1190_119009

theorem recreation_spending : 
  ∀ (W : ℝ), 
  (last_week_spent : ℝ) -> last_week_spent = 0.20 * W →
  (this_week_wages : ℝ) -> this_week_wages = 0.80 * W →
  (this_week_spent : ℝ) -> this_week_spent = 0.40 * this_week_wages →
  this_week_spent / last_week_spent * 100 = 160 :=
by
  sorry

end NUMINAMATH_GPT_recreation_spending_l1190_119009


namespace NUMINAMATH_GPT_joan_number_of_games_l1190_119031

open Nat

theorem joan_number_of_games (a b c d e : ℕ) (h_a : a = 10) (h_b : b = 12) (h_c : c = 6) (h_d : d = 9) (h_e : e = 4) :
  a + b + c + d + e = 41 :=
by
  sorry

end NUMINAMATH_GPT_joan_number_of_games_l1190_119031


namespace NUMINAMATH_GPT_problem_statement_l1190_119022

theorem problem_statement 
  (x y z : ℝ)
  (h1 : 5 = 0.25 * x)
  (h2 : 5 = 0.10 * y)
  (h3 : z = 2 * y) :
  x - z = -80 :=
sorry

end NUMINAMATH_GPT_problem_statement_l1190_119022


namespace NUMINAMATH_GPT_domain_of_sqrt_expression_l1190_119036

def isDomain (x : ℝ) : Prop := x ≥ -3 ∧ x < 7

theorem domain_of_sqrt_expression : 
  { x : ℝ | isDomain x } = { x | x ≥ -3 ∧ x < 7 } :=
by
  sorry

end NUMINAMATH_GPT_domain_of_sqrt_expression_l1190_119036


namespace NUMINAMATH_GPT_tetrahedron_three_edges_form_triangle_l1190_119018

-- Defining a tetrahedron
structure Tetrahedron := (A B C D : ℝ)
-- length of edges - since it's a geometry problem using the absolute value
def edge_length (x y : ℝ) := abs (x - y)

theorem tetrahedron_three_edges_form_triangle (T : Tetrahedron) :
  ∃ v : ℕ, ∃ e1 e2 e3 : ℝ, 
    (edge_length T.A T.B = e1 ∨ edge_length T.A T.C = e1 ∨ edge_length T.A T.D = e1) ∧ 
    (edge_length T.B T.C = e2 ∨ edge_length T.B T.D = e2 ∨ edge_length T.C T.D = e2) ∧
    (edge_length T.A T.B < e2 + e3 ∧ edge_length T.B T.C < e1 + e3 ∧ edge_length T.C T.D < e1 + e2) := 
sorry

end NUMINAMATH_GPT_tetrahedron_three_edges_form_triangle_l1190_119018


namespace NUMINAMATH_GPT_find_height_of_cylinder_l1190_119068

theorem find_height_of_cylinder (r SA : ℝ) (h : ℝ) (h_r : r = 3) (h_SA : SA = 30 * Real.pi) :
  SA = 2 * Real.pi * r^2 + 2 * Real.pi * r * h → h = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_height_of_cylinder_l1190_119068


namespace NUMINAMATH_GPT_stormi_lawns_mowed_l1190_119026

def num_lawns_mowed (cars_washed : ℕ) (money_per_car : ℕ) 
                    (lawns_mowed : ℕ) (money_per_lawn : ℕ) 
                    (bike_cost : ℕ) (money_needed : ℕ) : Prop :=
  (cars_washed * money_per_car + lawns_mowed * money_per_lawn) = (bike_cost - money_needed)

theorem stormi_lawns_mowed : num_lawns_mowed 3 10 2 13 80 24 :=
by
  sorry

end NUMINAMATH_GPT_stormi_lawns_mowed_l1190_119026


namespace NUMINAMATH_GPT_unknown_number_is_six_l1190_119087

theorem unknown_number_is_six (n : ℝ) (h : 12 * n^4 / 432 = 36) : n = 6 :=
by 
  -- This will be the placeholder for the proof
  sorry

end NUMINAMATH_GPT_unknown_number_is_six_l1190_119087


namespace NUMINAMATH_GPT_largest_integer_less_85_with_remainder_3_l1190_119069

theorem largest_integer_less_85_with_remainder_3 (n : ℕ) : 
  n < 85 ∧ n % 9 = 3 → n ≤ 84 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_largest_integer_less_85_with_remainder_3_l1190_119069


namespace NUMINAMATH_GPT_perimeter_of_triangle_l1190_119030

theorem perimeter_of_triangle (x y : ℝ) (h : 0 < x) (h1 : 0 < y) (h2 : x < y) :
  let leg_length := (y - x) / 2
  let hypotenuse := (y - x) / (Real.sqrt 2)
  (2 * leg_length + hypotenuse = (y - x) * (1 + 1 / Real.sqrt 2)) :=
by
  let leg_length := (y - x) / 2
  let hypotenuse := (y - x) / (Real.sqrt 2)
  sorry

end NUMINAMATH_GPT_perimeter_of_triangle_l1190_119030


namespace NUMINAMATH_GPT_johnny_words_l1190_119019

def words_johnny (J : ℕ) :=
  let words_madeline := 2 * J
  let words_timothy := 2 * J + 30
  let total_words := J + words_madeline + words_timothy
  total_words = 3 * 260 → J = 150

-- Statement of the main theorem (no proof provided, hence sorry is used)
theorem johnny_words (J : ℕ) : words_johnny J :=
by sorry

end NUMINAMATH_GPT_johnny_words_l1190_119019


namespace NUMINAMATH_GPT_meadow_income_is_960000_l1190_119091

theorem meadow_income_is_960000 :
  let boxes := 30
  let packs_per_box := 40
  let diapers_per_pack := 160
  let price_per_diaper := 5
  (boxes * packs_per_box * diapers_per_pack * price_per_diaper) = 960000 := 
by
  sorry

end NUMINAMATH_GPT_meadow_income_is_960000_l1190_119091


namespace NUMINAMATH_GPT_problem_statement_l1190_119025

theorem problem_statement (a b : ℝ) (h1 : a + b = 3) (h2 : a - b = 5) : b^2 - a^2 = -15 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l1190_119025


namespace NUMINAMATH_GPT_min_odd_integers_l1190_119090

-- Definitions of the conditions
variable (a b c d e f : ℤ)

-- The mathematical theorem statement
theorem min_odd_integers 
  (h1 : a + b = 30)
  (h2 : a + b + c + d = 50) 
  (h3 : a + b + c + d + e + f = 70)
  (h4 : e + f % 2 = 1) : 
  ∃ n, n ≥ 1 ∧ n = (if a % 2 = 1 then 1 else 0) + (if b % 2 = 1 then 1 else 0) + 
                    (if c % 2 = 1 then 1 else 0) + (if d % 2 = 1 then 1 else 0) + 
                    (if e % 2 = 1 then 1 else 0) + (if f % 2 = 1 then 1 else 0) :=
sorry

end NUMINAMATH_GPT_min_odd_integers_l1190_119090


namespace NUMINAMATH_GPT_total_money_l1190_119032

def JamesPocketBills : Nat := 3
def BillValue : Nat := 20
def WalletMoney : Nat := 75

theorem total_money (JamesPocketBills BillValue WalletMoney : Nat) : 
  (JamesPocketBills * BillValue + WalletMoney) = 135 :=
by
  sorry

end NUMINAMATH_GPT_total_money_l1190_119032


namespace NUMINAMATH_GPT_monthly_growth_rate_selling_price_april_l1190_119007

-- First problem: Proving the monthly average growth rate
theorem monthly_growth_rate (sales_jan sales_mar : ℝ) (x : ℝ) 
    (h1 : sales_jan = 256)
    (h2 : sales_mar = 400)
    (h3 : sales_mar = sales_jan * (1 + x)^2) :
  x = 0.25 := 
sorry

-- Second problem: Proving the selling price in April
theorem selling_price_april (unit_profit desired_profit current_sales sales_increase_per_yuan_change current_price new_price : ℝ)
    (h1 : unit_profit = new_price - 25)
    (h2 : desired_profit = 4200)
    (h3 : current_sales = 400)
    (h4 : sales_increase_per_yuan_change = 4)
    (h5 : current_price = 40)
    (h6 : desired_profit = unit_profit * (current_sales + sales_increase_per_yuan_change * (current_price - new_price))) :
  new_price = 35 := 
sorry

end NUMINAMATH_GPT_monthly_growth_rate_selling_price_april_l1190_119007


namespace NUMINAMATH_GPT_Claudia_solution_l1190_119097

noncomputable def Claudia_coins : Prop :=
  ∃ (x y : ℕ), x + y = 12 ∧ 23 - x = 17 ∧ y = 6

theorem Claudia_solution : Claudia_coins :=
by
  existsi 6
  existsi 6
  sorry

end NUMINAMATH_GPT_Claudia_solution_l1190_119097


namespace NUMINAMATH_GPT_triangle_is_right_angle_l1190_119079

theorem triangle_is_right_angle (a b c : ℝ) 
  (h : a^2 + b^2 + c^2 - 12*a - 16*b - 20*c + 200 = 0) : 
  a^2 + b^2 = c^2 :=
by 
  sorry

end NUMINAMATH_GPT_triangle_is_right_angle_l1190_119079


namespace NUMINAMATH_GPT_allison_greater_prob_l1190_119027

noncomputable def prob_allison_greater (p_brian : ℝ) (p_noah : ℝ) : ℝ :=
  p_brian * p_noah

theorem allison_greater_prob : prob_allison_greater (2/3) (1/2) = 1/3 :=
by {
  -- Calculate the combined probability
  sorry
}

end NUMINAMATH_GPT_allison_greater_prob_l1190_119027


namespace NUMINAMATH_GPT_terry_daily_driving_time_l1190_119014

theorem terry_daily_driving_time 
  (d1: ℝ) (s1: ℝ)
  (d2: ℝ) (s2: ℝ)
  (d3: ℝ) (s3: ℝ)
  (h1 : d1 = 15) (h2 : s1 = 30)
  (h3 : d2 = 35) (h4 : s2 = 50)
  (h5 : d3 = 10) (h6 : s3 = 40) : 
  2 * ((d1 / s1) + (d2 / s2) + (d3 / s3)) = 2.9 := 
by
  sorry

end NUMINAMATH_GPT_terry_daily_driving_time_l1190_119014


namespace NUMINAMATH_GPT_prime_divides_a_minus_3_l1190_119065

theorem prime_divides_a_minus_3 (a p : ℤ) (hp : Prime p) (h1 : p ∣ 5 * a - 1) (h2 : p ∣ a - 10) : p ∣ a - 3 := by
  sorry

end NUMINAMATH_GPT_prime_divides_a_minus_3_l1190_119065


namespace NUMINAMATH_GPT_leak_empty_time_l1190_119021

theorem leak_empty_time (A L : ℝ) (h1 : A = 1 / 8) (h2 : A - L = 1 / 12) : 1 / L = 24 :=
by
  -- The proof will be provided here
  sorry

end NUMINAMATH_GPT_leak_empty_time_l1190_119021


namespace NUMINAMATH_GPT_percentage_saved_on_hats_l1190_119075

/-- Suppose the regular price of a hat is $60 and Maria buys four hats with progressive discounts: 
20% off the second hat, 40% off the third hat, and 50% off the fourth hat.
Prove that the percentage saved on the regular price for four hats is 27.5%. -/
theorem percentage_saved_on_hats :
  let regular_price := 60
  let discount_2 := 0.2 * regular_price
  let discount_3 := 0.4 * regular_price
  let discount_4 := 0.5 * regular_price
  let price_1 := regular_price
  let price_2 := regular_price - discount_2
  let price_3 := regular_price - discount_3
  let price_4 := regular_price - discount_4
  let total_regular := 4 * regular_price
  let total_discounted := price_1 + price_2 + price_3 + price_4
  let savings := total_regular - total_discounted
  let percentage_saved := (savings / total_regular) * 100
  percentage_saved = 27.5 :=
by
  sorry

end NUMINAMATH_GPT_percentage_saved_on_hats_l1190_119075


namespace NUMINAMATH_GPT_problem1_problem2_l1190_119054

-- Problem 1: Prove the expression equals the calculated value
theorem problem1 : (-2:ℝ)^0 + (1 / Real.sqrt 2) - Real.sqrt 9 = (Real.sqrt 2) / 2 - 2 :=
by sorry

-- Problem 2: Prove the solution to the system of linear equations
theorem problem2 (x y : ℝ) (h1 : 2 * x - y = 3) (h2 : x + y = -2) :
  x = 1/3 ∧ y = -(7/3) :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_l1190_119054


namespace NUMINAMATH_GPT_initial_pieces_of_gum_l1190_119039

theorem initial_pieces_of_gum (additional_pieces given_pieces leftover_pieces initial_pieces : ℕ)
  (h_additional : additional_pieces = 3)
  (h_given : given_pieces = 11)
  (h_leftover : leftover_pieces = 2)
  (h_initial : initial_pieces + additional_pieces = given_pieces + leftover_pieces) :
  initial_pieces = 10 :=
by
  sorry

end NUMINAMATH_GPT_initial_pieces_of_gum_l1190_119039


namespace NUMINAMATH_GPT_Megan_bought_24_eggs_l1190_119099

def eggs_problem : Prop :=
  ∃ (p c b : ℕ),
    b = 3 ∧
    c = 2 * b ∧
    p - c = 9 ∧
    p + c + b = 24

theorem Megan_bought_24_eggs : eggs_problem :=
  sorry

end NUMINAMATH_GPT_Megan_bought_24_eggs_l1190_119099


namespace NUMINAMATH_GPT_volume_ratio_of_rotated_solids_l1190_119016

theorem volume_ratio_of_rotated_solids (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let V1 := π * b^2 * a
  let V2 := π * a^2 * b
  V1 / V2 = b / a :=
by
  intros
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_volume_ratio_of_rotated_solids_l1190_119016


namespace NUMINAMATH_GPT_value_of_a_l1190_119059

theorem value_of_a (a : ℝ) :
  (∀ x : ℝ, (x^2 - 4*a*x + 5*a^2 - 6*a = 0 → 
    ∃ x₁ x₂, x₁ + x₂ = 4*a ∧ x₁ * x₂ = 5*a^2 - 6*a ∧ |x₁ - x₂| = 6)) → a = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_value_of_a_l1190_119059


namespace NUMINAMATH_GPT_John_more_marbles_than_Ben_l1190_119038

theorem John_more_marbles_than_Ben :
  let ben_initial := 18
  let john_initial := 17
  let ben_gave := ben_initial / 2
  let ben_final := ben_initial - ben_gave
  let john_final := john_initial + ben_gave
  john_final - ben_final = 17 :=
by
  sorry

end NUMINAMATH_GPT_John_more_marbles_than_Ben_l1190_119038


namespace NUMINAMATH_GPT_johns_improvement_l1190_119020

-- Declare the variables for the initial and later lap times.
def initial_minutes : ℕ := 50
def initial_laps : ℕ := 25
def later_minutes : ℕ := 54
def later_laps : ℕ := 30

-- Calculate the initial and later lap times in seconds, and the improvement.
def initial_lap_time_seconds := (initial_minutes * 60) / initial_laps 
def later_lap_time_seconds := (later_minutes * 60) / later_laps
def improvement := initial_lap_time_seconds - later_lap_time_seconds

-- State the theorem to prove the improvement is 12 seconds per lap.
theorem johns_improvement : improvement = 12 := by
  sorry

end NUMINAMATH_GPT_johns_improvement_l1190_119020


namespace NUMINAMATH_GPT_malvina_correct_l1190_119082
noncomputable def angle (x : ℝ) : Prop := 0 < x ∧ x < 180
noncomputable def malvina_identifies (x : ℝ) : Prop := x > 90

noncomputable def sum_of_values := (Real.sqrt 5 + Real.sqrt 2) / 2

theorem malvina_correct (x : ℝ) (h1 : angle x) (h2 : malvina_identifies x) :
  sum_of_values = (Real.sqrt 5 + Real.sqrt 2) / 2 :=
by sorry

end NUMINAMATH_GPT_malvina_correct_l1190_119082


namespace NUMINAMATH_GPT_solve_quadratic1_solve_quadratic2_l1190_119035

open Real

-- Equation 1
theorem solve_quadratic1 (x : ℝ) : x^2 - 6 * x + 8 = 0 → x = 2 ∨ x = 4 := 
by sorry

-- Equation 2
theorem solve_quadratic2 (x : ℝ) : x^2 - 8 * x + 1 = 0 → x = 4 + sqrt 15 ∨ x = 4 - sqrt 15 := 
by sorry

end NUMINAMATH_GPT_solve_quadratic1_solve_quadratic2_l1190_119035


namespace NUMINAMATH_GPT_evaluate_expression_l1190_119037

theorem evaluate_expression:
  let a := 11
  let b := 13
  let c := 17
  (121 * (1/b - 1/c) + 169 * (1/c - 1/a) + 289 * (1/a - 1/b)) / 
  (11 * (1/b - 1/c) + 13 * (1/c - 1/a) + 17 * (1/a - 1/b)) = 41 :=
by
  let a := 11
  let b := 13
  let c := 17
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1190_119037


namespace NUMINAMATH_GPT_shaniqua_style_income_correct_l1190_119000

def shaniqua_income_per_style (haircut_income : ℕ) (total_income : ℕ) (number_of_haircuts : ℕ) (number_of_styles : ℕ) : ℕ :=
  (total_income - (number_of_haircuts * haircut_income)) / number_of_styles

theorem shaniqua_style_income_correct :
  shaniqua_income_per_style 12 221 8 5 = 25 :=
by
  sorry

end NUMINAMATH_GPT_shaniqua_style_income_correct_l1190_119000


namespace NUMINAMATH_GPT_cook_remaining_potatoes_l1190_119043

def total_time_to_cook_remaining_potatoes (total_potatoes cooked_potatoes time_per_potato : ℕ) : ℕ :=
  (total_potatoes - cooked_potatoes) * time_per_potato

theorem cook_remaining_potatoes 
  (total_potatoes cooked_potatoes time_per_potato : ℕ) 
  (h_total_potatoes : total_potatoes = 13)
  (h_cooked_potatoes : cooked_potatoes = 5)
  (h_time_per_potato : time_per_potato = 6) : 
  total_time_to_cook_remaining_potatoes total_potatoes cooked_potatoes time_per_potato = 48 :=
by
  -- Proof not required
  sorry

end NUMINAMATH_GPT_cook_remaining_potatoes_l1190_119043


namespace NUMINAMATH_GPT_smallest_sum_of_three_diff_numbers_l1190_119052

theorem smallest_sum_of_three_diff_numbers : 
  ∀ (s : Set ℤ), s = {8, -7, 2, -4, 20} → ∃ a b c, a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (a + b + c = -9) :=
by
  sorry

end NUMINAMATH_GPT_smallest_sum_of_three_diff_numbers_l1190_119052


namespace NUMINAMATH_GPT_hattie_jumps_l1190_119041

theorem hattie_jumps (H : ℝ) (h1 : Lorelei_jumps1 = (3/4) * H)
  (h2 : Hattie_jumps2 = (2/3) * H)
  (h3 : Lorelei_jumps2 = (2/3) * H + 50)
  (h4 : H + Lorelei_jumps1 + Hattie_jumps2 + Lorelei_jumps2 = 605) : H = 180 :=
by
  sorry

noncomputable def Lorelei_jumps1 (H : ℝ) := (3/4) * H
noncomputable def Hattie_jumps2 (H : ℝ) := (2/3) * H
noncomputable def Lorelei_jumps2 (H : ℝ) := (2/3) * H + 50

end NUMINAMATH_GPT_hattie_jumps_l1190_119041


namespace NUMINAMATH_GPT_train_speed_l1190_119062

theorem train_speed (v : ℝ) :
  let speed_train1 := 80  -- speed of the first train in km/h
  let length_train1 := 150 / 1000 -- length of the first train in km
  let length_train2 := 100 / 1000 -- length of the second train in km
  let total_time := 5.999520038396928 / 3600 -- time in hours
  let total_length := length_train1 + length_train2 -- total length in km
  let relative_speed := total_length / total_time -- relative speed in km/h
  relative_speed = speed_train1 + v → v = 70 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_l1190_119062


namespace NUMINAMATH_GPT_f_log_sum_l1190_119098

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + x ^ 2) - x) + 2

theorem f_log_sum (x : ℝ) : f (Real.log 5) + f (Real.log (1 / 5)) = 4 :=
by
  sorry

end NUMINAMATH_GPT_f_log_sum_l1190_119098


namespace NUMINAMATH_GPT_calculate_gallons_of_milk_l1190_119071

-- Definitions of the given constants and conditions
def price_of_soup : Nat := 2
def price_of_bread : Nat := 5
def price_of_cereal : Nat := 3
def price_of_milk : Nat := 4
def total_amount_paid : Nat := 4 * 10

-- Calculation of total cost of non-milk items
def total_cost_non_milk : Nat :=
  (6 * price_of_soup) + (2 * price_of_bread) + (2 * price_of_cereal)

-- The function to calculate the remaining amount to be spent on milk
def remaining_amount : Nat := total_amount_paid - total_cost_non_milk

-- Statement to compute the number of gallons of milk
def gallons_of_milk (remaining : Nat) (price_per_gallon : Nat) : Nat :=
  remaining / price_per_gallon

-- Proof theorem statement (no implementation required, proof skipped)
theorem calculate_gallons_of_milk : 
  gallons_of_milk remaining_amount price_of_milk = 3 := 
by
  sorry

end NUMINAMATH_GPT_calculate_gallons_of_milk_l1190_119071


namespace NUMINAMATH_GPT_jellybeans_needed_l1190_119053

theorem jellybeans_needed (n : ℕ) : (n ≥ 120 ∧ n % 15 = 14) → n = 134 :=
by sorry

end NUMINAMATH_GPT_jellybeans_needed_l1190_119053


namespace NUMINAMATH_GPT_integer_value_of_fraction_l1190_119034

theorem integer_value_of_fraction (m n p : ℕ) (hm_diff: m ≠ n) (hn_diff: n ≠ p) (hp_diff: m ≠ p) 
  (hm_range: 2 ≤ m ∧ m ≤ 9) (hn_range: 2 ≤ n ∧ n ≤ 9) (hp_range: 2 ≤ p ∧ p ≤ 9) :
  (m + n + p) / (m + n) = 2 :=
by
  sorry

end NUMINAMATH_GPT_integer_value_of_fraction_l1190_119034


namespace NUMINAMATH_GPT_otimes_identity_l1190_119003

def otimes (x y : ℝ) : ℝ := x^2 - y^2

theorem otimes_identity (h : ℝ) : otimes h (otimes h h) = h^2 :=
by
  sorry

end NUMINAMATH_GPT_otimes_identity_l1190_119003


namespace NUMINAMATH_GPT_one_point_shots_count_l1190_119060

-- Define the given conditions
def three_point_shots : Nat := 15
def two_point_shots : Nat := 12
def total_points : Nat := 75
def points_per_three_shot : Nat := 3
def points_per_two_shot : Nat := 2

-- Define the total points contributed by three-point and two-point shots
def three_point_total : Nat := three_point_shots * points_per_three_shot
def two_point_total : Nat := two_point_shots * points_per_two_shot
def combined_point_total : Nat := three_point_total + two_point_total

-- Formulate the theorem to prove the number of one-point shots Tyson made
theorem one_point_shots_count : combined_point_total <= total_points →
  (total_points - combined_point_total = 6) :=
by 
  -- Skip the proof
  sorry

end NUMINAMATH_GPT_one_point_shots_count_l1190_119060


namespace NUMINAMATH_GPT_coefficients_equal_l1190_119015

theorem coefficients_equal (n : ℕ) (h : n ≥ 6) : 
  (n = 7) ↔ 
  (Nat.choose n 5 * 3 ^ 5 = Nat.choose n 6 * 3 ^ 6) := by
  sorry

end NUMINAMATH_GPT_coefficients_equal_l1190_119015


namespace NUMINAMATH_GPT_number_of_teachers_at_Queen_Middle_School_l1190_119056

-- Conditions
def num_students : ℕ := 1500
def classes_per_student : ℕ := 6
def classes_per_teacher : ℕ := 5
def students_per_class : ℕ := 25

-- Proof that the number of teachers is 72
theorem number_of_teachers_at_Queen_Middle_School :
  (num_students * classes_per_student) / students_per_class / classes_per_teacher = 72 :=
by sorry

end NUMINAMATH_GPT_number_of_teachers_at_Queen_Middle_School_l1190_119056


namespace NUMINAMATH_GPT_total_birds_count_l1190_119078

def cage1_parrots := 9
def cage1_finches := 4
def cage1_canaries := 7

def cage2_parrots := 5
def cage2_parakeets := 8
def cage2_finches := 10

def cage3_parakeets := 15
def cage3_finches := 7
def cage3_canaries := 3

def cage4_parrots := 10
def cage4_parakeets := 5
def cage4_finches := 12

def total_birds := cage1_parrots + cage1_finches + cage1_canaries +
                   cage2_parrots + cage2_parakeets + cage2_finches +
                   cage3_parakeets + cage3_finches + cage3_canaries +
                   cage4_parrots + cage4_parakeets + cage4_finches

theorem total_birds_count : total_birds = 95 :=
by
  -- Proof is omitted here.
  sorry

end NUMINAMATH_GPT_total_birds_count_l1190_119078


namespace NUMINAMATH_GPT_surface_area_of_sphere_with_diameter_two_l1190_119002

theorem surface_area_of_sphere_with_diameter_two :
  let diameter := 2
  let radius := diameter / 2
  4 * Real.pi * radius ^ 2 = 4 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_surface_area_of_sphere_with_diameter_two_l1190_119002


namespace NUMINAMATH_GPT_problem_simplify_l1190_119033

variable (a : ℝ)

theorem problem_simplify (h1 : a ≠ 3) (h2 : a ≠ -3) :
  (1 / (a - 3) - 6 / (a^2 - 9) = 1 / (a + 3)) :=
sorry

end NUMINAMATH_GPT_problem_simplify_l1190_119033


namespace NUMINAMATH_GPT_number_of_games_in_complete_season_l1190_119064

-- Define the number of teams in each division
def teams_in_division_A : Nat := 6
def teams_in_division_B : Nat := 7
def teams_in_division_C : Nat := 5

-- Define the number of games each team must play within their division
def games_per_team_within_division (teams : Nat) : Nat :=
  (teams - 1) * 2

-- Calculate the total number of games within a division
def total_games_within_division (teams : Nat) : Nat :=
  (games_per_team_within_division teams * teams) / 2

-- Calculate cross-division games for a team in one division
def cross_division_games_per_team (teams_other_div1 : Nat) (teams_other_div2 : Nat) : Nat :=
  (teams_other_div1 + teams_other_div2) * 2

-- Calculate total cross-division games from all teams in one division
def total_cross_division_games (teams_div : Nat) (teams_other_div1 : Nat) (teams_other_div2 : Nat) : Nat :=
  cross_division_games_per_team teams_other_div1 teams_other_div2 * teams_div

-- Given conditions translated to definitions
def games_in_division_A : Nat := total_games_within_division teams_in_division_A
def games_in_division_B : Nat := total_games_within_division teams_in_division_B
def games_in_division_C : Nat := total_games_within_division teams_in_division_C

def cross_division_games_A : Nat := total_cross_division_games teams_in_division_A teams_in_division_B teams_in_division_C
def cross_division_games_B : Nat := total_cross_division_games teams_in_division_B teams_in_division_A teams_in_division_C
def cross_division_games_C : Nat := total_cross_division_games teams_in_division_C teams_in_division_A teams_in_division_B

-- Total cross-division games with each game counted twice
def total_cross_division_games_in_season : Nat :=
  (cross_division_games_A + cross_division_games_B + cross_division_games_C) / 2

-- Total number of games in the season
def total_games_in_season : Nat :=
  games_in_division_A + games_in_division_B + games_in_division_C + total_cross_division_games_in_season

-- The final proof statement
theorem number_of_games_in_complete_season : total_games_in_season = 306 :=
by
  -- This is the place where the proof would go if it were required.
  sorry

end NUMINAMATH_GPT_number_of_games_in_complete_season_l1190_119064


namespace NUMINAMATH_GPT_inequality_holds_l1190_119010

theorem inequality_holds (x : ℝ) : (∀ y : ℝ, y > 0 → (4 * (x^2 * y^2 + 4 * x * y^2 + 4 * x^2 * y + 16 * y^2 + 12 * x^2 * y) / (x + y) > 3 * x^2 * y)) ↔ x > 0 := 
sorry

end NUMINAMATH_GPT_inequality_holds_l1190_119010


namespace NUMINAMATH_GPT_prob_lfloor_XZ_YZ_product_eq_33_l1190_119074

noncomputable def XZ_YZ_product : ℝ :=
  let AB := 15
  let BC := 14
  let CA := 13
  -- Definition of points and conditions
  -- Note: Specific geometric definitions and conditions need to be properly defined as per Lean's geometry library. This is a simplified placeholder.
  sorry

theorem prob_lfloor_XZ_YZ_product_eq_33 :
  (⌊XZ_YZ_product⌋ = 33) := sorry

end NUMINAMATH_GPT_prob_lfloor_XZ_YZ_product_eq_33_l1190_119074


namespace NUMINAMATH_GPT_initial_garrison_men_l1190_119057

theorem initial_garrison_men (M : ℕ) (h1 : 62 * M = 62 * M) 
  (h2 : M * 47 = (M + 2700) * 20) : M = 2000 := by
  sorry

end NUMINAMATH_GPT_initial_garrison_men_l1190_119057


namespace NUMINAMATH_GPT_remainder_of_sum_mod_9_l1190_119049

theorem remainder_of_sum_mod_9 :
  (9023 + 9024 + 9025 + 9026 + 9027) % 9 = 2 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_sum_mod_9_l1190_119049


namespace NUMINAMATH_GPT_y_coordinate_in_fourth_quadrant_l1190_119058
-- Importing the necessary libraries

-- Definition of the problem statement
theorem y_coordinate_in_fourth_quadrant (x y : ℝ) (h : x = 5 ∧ y < 0) : y < 0 :=
by 
  sorry

end NUMINAMATH_GPT_y_coordinate_in_fourth_quadrant_l1190_119058


namespace NUMINAMATH_GPT_correct_option_is_optionB_l1190_119051

-- Definitions based on conditions
def optionA : ℝ := 0.37 * 1.5
def optionB : ℝ := 3.7 * 1.5
def optionC : ℝ := 0.37 * 1500
def original : ℝ := 0.37 * 15

-- Statement to prove that the correct answer (optionB) yields the same result as the original expression
theorem correct_option_is_optionB : optionB = original :=
sorry

end NUMINAMATH_GPT_correct_option_is_optionB_l1190_119051


namespace NUMINAMATH_GPT_no_function_exists_l1190_119080

-- Main theorem statement
theorem no_function_exists : ¬ ∃ f : ℝ → ℝ, 
  (∀ x y : ℝ, 0 < x → 0 < y → (x + y) * f (2 * y * f x + f y) = x^3 * f (y * f x)) ∧ 
  (∀ z : ℝ, 0 < z → f z > 0) :=
sorry

end NUMINAMATH_GPT_no_function_exists_l1190_119080


namespace NUMINAMATH_GPT_area_of_region_l1190_119023

theorem area_of_region :
  let x := fun t : ℝ => 6 * Real.cos t
  let y := fun t : ℝ => 2 * Real.sin t
  (∫ t in (Real.pi / 3)..(Real.pi / 2), (x t) * (deriv y t)) * 2 = 2 * Real.pi - 3 * Real.sqrt 3 := by
  let x := fun t : ℝ => 6 * Real.cos t
  let y := fun t : ℝ => 2 * Real.sin t
  have h1 : ∫ t in (Real.pi / 3)..(Real.pi / 2), x t * deriv y t = 12 * ∫ t in (Real.pi / 3)..(Real.pi / 2), (1 + Real.cos (2*t)) / 2 := sorry
  have h2 : 12 * ∫ t in (Real.pi / 3)..(Real.pi / 2), (1 + Real.cos (2 * t)) / 2 = 2 * Real.pi - 3 * Real.sqrt 3 := sorry
  sorry

end NUMINAMATH_GPT_area_of_region_l1190_119023


namespace NUMINAMATH_GPT_rice_pounds_l1190_119083

noncomputable def pounds_of_rice (r p : ℝ) : Prop :=
  r + p = 30 ∧ 1.10 * r + 0.55 * p = 23.50

theorem rice_pounds (r p : ℝ) (h : pounds_of_rice r p) : r = 12.7 :=
sorry

end NUMINAMATH_GPT_rice_pounds_l1190_119083


namespace NUMINAMATH_GPT_integral_sin_from_0_to_pi_div_2_l1190_119050

theorem integral_sin_from_0_to_pi_div_2 :
  ∫ x in (0 : ℝ)..(Real.pi / 2), Real.sin x = 1 := by
  sorry

end NUMINAMATH_GPT_integral_sin_from_0_to_pi_div_2_l1190_119050


namespace NUMINAMATH_GPT_line_eq1_line_eq2_l1190_119085

-- Define the line equations
def l1 (x y : ℝ) : Prop := 4 * x + y + 6 = 0
def l2 (x y : ℝ) : Prop := 3 * x - 5 * y - 6 = 0

-- Theorem for when midpoint is at (0, 0)
theorem line_eq1 : ∀ x y : ℝ, (x + 6 * y = 0) ↔
  ∃ (a : ℝ), 
    l1 a (-(a / 6)) ∧
    l2 (-a) ((a / 6)) ∧
    (a + -a = 0) ∧ (-(a / 6) + a / 6 = 0) := 
by 
  sorry

-- Theorem for when midpoint is at (0, 1)
theorem line_eq2 : ∀ x y : ℝ, (x + 2 * y - 2 = 0) ↔
  ∃ (b : ℝ),
    l1 b (-b / 2 + 1) ∧
    l2 (-b) (1 - (-b / 2)) ∧
    (b + -b = 0) ∧ (-b / 2 + 1 + (1 - (-b / 2)) = 2) := 
by 
  sorry

end NUMINAMATH_GPT_line_eq1_line_eq2_l1190_119085


namespace NUMINAMATH_GPT_expected_greetings_l1190_119024

theorem expected_greetings :
  let p1 := 1       -- Probability 1
  let p2 := 0.8     -- Probability 0.8
  let p3 := 0.5     -- Probability 0.5
  let p4 := 0       -- Probability 0
  let n1 := 8       -- Number of colleagues with probability 1
  let n2 := 15      -- Number of colleagues with probability 0.8
  let n3 := 14      -- Number of colleagues with probability 0.5
  let n4 := 3       -- Number of colleagues with probability 0
  p1 * n1 + p2 * n2 + p3 * n3 + p4 * n4 = 27 :=
by
  sorry

end NUMINAMATH_GPT_expected_greetings_l1190_119024


namespace NUMINAMATH_GPT_line_up_including_A_line_up_excluding_all_ABC_line_up_adjacent_AB_not_adjacent_C_l1190_119047

-- Define the set of people
def people : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : ℕ := 1 
def B : ℕ := 2
def C : ℕ := 3

-- Proof Problem 1: Prove that there are 1800 ways to line up 5 people out of 7 given A must be included.
theorem line_up_including_A : Finset ℕ → ℕ :=
by
  sorry

-- Proof Problem 2: Prove that there are 1800 ways to line up 5 people out of 7 given A, B, and C are not all included.
theorem line_up_excluding_all_ABC : Finset ℕ → ℕ :=
by
  sorry

-- Proof Problem 3: Prove that there are 144 ways to line up 5 people out of 7 given A, B, and C are all included, A and B are adjacent, and C is not adjacent to A or B.
theorem line_up_adjacent_AB_not_adjacent_C : Finset ℕ → ℕ :=
by
  sorry

end NUMINAMATH_GPT_line_up_including_A_line_up_excluding_all_ABC_line_up_adjacent_AB_not_adjacent_C_l1190_119047


namespace NUMINAMATH_GPT_picking_time_l1190_119077

theorem picking_time (x : ℝ) 
  (h_wang : x * 8 - 0.25 = x * 7) : 
  x = 0.25 := 
by
  sorry

end NUMINAMATH_GPT_picking_time_l1190_119077


namespace NUMINAMATH_GPT_students_participated_in_function_l1190_119066

theorem students_participated_in_function :
  ∀ (B G : ℕ),
  B + G = 800 →
  (3 / 4 : ℚ) * G = 150 →
  (2 / 3 : ℚ) * B + 150 = 550 :=
by
  intros B G h1 h2
  sorry

end NUMINAMATH_GPT_students_participated_in_function_l1190_119066


namespace NUMINAMATH_GPT_sum_of_three_numbers_l1190_119040

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 52) 
  (h2 : ab + bc + ca = 72) : 
  a + b + c = 14 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_three_numbers_l1190_119040
