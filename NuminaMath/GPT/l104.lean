import Mathlib

namespace father_l104_104739

variable (S F : ℕ)

theorem father's_age (h1 : F = 3 * S + 3) (h2 : F + 3 = 2 * (S + 3) + 10) : F = 33 := by
  sorry

end father_l104_104739


namespace b_is_square_of_positive_integer_l104_104987

theorem b_is_square_of_positive_integer 
  (a b : ℕ) (ha : 0 < a) (hb : 0 < b) 
  (h : b^2 = a^2 + ab + b) : 
  ∃ k : ℕ, b = k^2 := 
by 
  sorry

end b_is_square_of_positive_integer_l104_104987


namespace football_daily_practice_hours_l104_104745

-- Define the total practice hours and the days missed.
def total_hours := 30
def days_missed := 1
def days_in_week := 7

-- Calculate the number of days practiced.
def days_practiced := days_in_week - days_missed

-- Define the daily practice hours.
def daily_practice_hours := total_hours / days_practiced

-- State the proposition.
theorem football_daily_practice_hours :
  daily_practice_hours = 5 := sorry

end football_daily_practice_hours_l104_104745


namespace total_number_of_students_is_40_l104_104272

variables (S R : ℕ)

-- Conditions
def students_not_borrowed_any_books := 2
def students_borrowed_1_book := 12
def students_borrowed_2_books := 10
def average_books_per_student := 2

-- Definition of total books borrowed
def total_books_borrowed := (0 * students_not_borrowed_any_books) + (1 * students_borrowed_1_book) + (2 * students_borrowed_2_books) + (3 * R)

-- Expression for total number of students
def total_students := students_not_borrowed_any_books + students_borrowed_1_book + students_borrowed_2_books + R

-- Mathematical statement to prove
theorem total_number_of_students_is_40 (h : total_books_borrowed R / total_students R = average_books_per_student) : total_students R = 40 :=
sorry

end total_number_of_students_is_40_l104_104272


namespace price_increase_count_l104_104930

-- Conditions
def original_price (P : ℝ) : ℝ := P
def increase_factor : ℝ := 1.15
def final_factor : ℝ := 1.3225

-- The theorem that states the number of times the price increased
theorem price_increase_count (n : ℕ) :
  increase_factor ^ n = final_factor → n = 2 :=
by
  sorry

end price_increase_count_l104_104930


namespace max_value_of_function_cos_sin_l104_104492

noncomputable def max_value_function (x : ℝ) : ℝ := 
  (Real.cos x)^3 + (Real.sin x)^2 - Real.cos x

theorem max_value_of_function_cos_sin : 
  ∃ x ∈ (Set.univ : Set ℝ), max_value_function x = (32 / 27) := 
sorry

end max_value_of_function_cos_sin_l104_104492


namespace color_of_182nd_marble_l104_104173

-- conditions
def pattern_length : ℕ := 15
def blue_length : ℕ := 6
def red_length : ℕ := 5
def green_length : ℕ := 4

def marble_color (n : ℕ) : String :=
  let cycle_pos := n % pattern_length
  if cycle_pos < blue_length then
    "blue"
  else if cycle_pos < blue_length + red_length then
    "red"
  else
    "green"

theorem color_of_182nd_marble : marble_color 182 = "blue" :=
by
  sorry

end color_of_182nd_marble_l104_104173


namespace length_of_hypotenuse_l104_104776

theorem length_of_hypotenuse (a b c : ℝ) (h1 : a^2 + b^2 + c^2 = 2450) (h2 : c = b + 10) (h3 : a^2 + b^2 = c^2) : c = 35 :=
by
  sorry

end length_of_hypotenuse_l104_104776


namespace marcy_sip_amount_l104_104485

theorem marcy_sip_amount (liters : ℕ) (ml_per_liter : ℕ) (total_minutes : ℕ) (interval_minutes : ℕ) (total_ml : ℕ) (total_sips : ℕ) (ml_per_sip : ℕ) 
  (h1 : liters = 2) 
  (h2 : ml_per_liter = 1000)
  (h3 : total_minutes = 250) 
  (h4 : interval_minutes = 5)
  (h5 : total_ml = liters * ml_per_liter)
  (h6 : total_sips = total_minutes / interval_minutes)
  (h7 : ml_per_sip = total_ml / total_sips) : 
  ml_per_sip = 40 := 
by
  sorry

end marcy_sip_amount_l104_104485


namespace Sally_out_of_pocket_payment_l104_104049

theorem Sally_out_of_pocket_payment :
  let amount_given : ℕ := 320
  let cost_per_book : ℕ := 12
  let number_of_students : ℕ := 30
  let total_cost : ℕ := cost_per_book * number_of_students
  let out_of_pocket_cost : ℕ := total_cost - amount_given
  out_of_pocket_cost = 40 := by
  sorry

end Sally_out_of_pocket_payment_l104_104049


namespace find_some_expression_l104_104394

noncomputable def problem_statement : Prop :=
  ∃ (some_expression : ℝ), 
    (5 + 7 / 12 = 6 - some_expression) ∧ 
    (some_expression = 0.4167)

theorem find_some_expression : problem_statement := 
  sorry

end find_some_expression_l104_104394


namespace math_dance_residents_l104_104089

theorem math_dance_residents (p a b : ℕ) (hp : Nat.Prime p) 
    (h1 : b ≥ 1) 
    (h2 : (a + b)^2 = (p + 1) * a + b) :
    b = 1 := by
  sorry

end math_dance_residents_l104_104089


namespace polynomial_factors_l104_104433

theorem polynomial_factors (h k : ℤ)
  (h1 : 3 * (-2)^4 - 2 * h * (-2)^2 + h * (-2) + k = 0)
  (h2 : 3 * 1^4 - 2 * h * 1^2 + h * 1 + k = 0)
  (h3 : 3 * (-3)^4 - 2 * h * (-3)^2 + h * (-3) + k = 0) :
  |3 * h - 2 * k| = 11 :=
by
  sorry

end polynomial_factors_l104_104433


namespace avg_difference_is_5_l104_104126

def avg (s : List ℕ) : ℕ :=
  s.sum / s.length

def set1 := [20, 40, 60]
def set2 := [20, 60, 25]

theorem avg_difference_is_5 :
  avg set1 - avg set2 = 5 :=
by
  sorry

end avg_difference_is_5_l104_104126


namespace crickets_needed_to_reach_11_l104_104204

theorem crickets_needed_to_reach_11 (collected_crickets : ℕ) (wanted_crickets : ℕ) 
                                     (h : collected_crickets = 7) (h2 : wanted_crickets = 11) :
  wanted_crickets - collected_crickets = 4 :=
sorry

end crickets_needed_to_reach_11_l104_104204


namespace area_of_union_of_triangles_l104_104306

-- Define the vertices of the original triangle
def A : ℝ × ℝ := (3, 4)
def B : ℝ × ℝ := (5, -2)
def C : ℝ × ℝ := (7, 3)

-- Define the reflection function across the line x=5
def reflect_x5 (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  (10 - x, y)

-- Define the vertices of the reflected triangle
def A' : ℝ × ℝ := reflect_x5 A
def B' : ℝ × ℝ := reflect_x5 B
def C' : ℝ × ℝ := reflect_x5 C

-- Function to calculate the area of a triangle given its vertices
def triangle_area (P Q R : ℝ × ℝ) : ℝ :=
  let (x1, y1) := P
  let (x2, y2) := Q
  let (x3, y3) := R
  0.5 * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

-- Prove that the area of the union of both triangles is 22
theorem area_of_union_of_triangles : triangle_area A B C + triangle_area A' B' C' = 22 := by
  sorry

end area_of_union_of_triangles_l104_104306


namespace archibald_percentage_games_won_l104_104566

theorem archibald_percentage_games_won
  (A B F1 F2 : ℝ) -- number of games won by Archibald, his brother, and his two friends
  (total_games : ℝ)
  (A_eq_1_1B : A = 1.1 * B)
  (F_eq_2_1B : F1 + F2 = 2.1 * B)
  (total_games_eq : A + B + F1 + F2 = total_games)
  (total_games_val : total_games = 280) :
  (A / total_games * 100) = 26.19 :=
by
  sorry

end archibald_percentage_games_won_l104_104566


namespace expression_equivalence_l104_104210

def algebraicExpression : String := "5 - 4a"
def wordExpression : String := "the difference of 5 and 4 times a"

theorem expression_equivalence : algebraicExpression = wordExpression := 
sorry

end expression_equivalence_l104_104210


namespace range_of_a_for_empty_solution_set_l104_104415

theorem range_of_a_for_empty_solution_set :
  ∀ a : ℝ, (∀ x : ℝ, ¬ (|x - 3| + |x - 4| < a)) ↔ a ≤ 1 := 
sorry

end range_of_a_for_empty_solution_set_l104_104415


namespace rate_in_still_water_l104_104679

-- Definitions of given conditions
def downstream_speed : ℝ := 26
def upstream_speed : ℝ := 12

-- The statement we need to prove
theorem rate_in_still_water : (downstream_speed + upstream_speed) / 2 = 19 := by
  sorry

end rate_in_still_water_l104_104679


namespace part1_part2_l104_104977

noncomputable def probability_A_receives_one_red_envelope : ℚ :=
  sorry

theorem part1 (P_A1 : ℚ) (P_not_A1 : ℚ) (P_A2 : ℚ) (P_not_A2 : ℚ) :
  P_A1 = 1/3 ∧ P_not_A1 = 2/3 ∧ P_A2 = 1/3 ∧ P_not_A2 = 2/3 →
  probability_A_receives_one_red_envelope = 4/9 :=
sorry

noncomputable def probability_B_receives_at_least_10_yuan : ℚ :=
  sorry

theorem part2 (P_B1 : ℚ) (P_not_B1 : ℚ) (P_B2 : ℚ) (P_not_B2 : ℚ) (P_B3 : ℚ) (P_not_B3 : ℚ) :
  P_B1 = 1/3 ∧ P_not_B1 = 2/3 ∧ P_B2 = 1/3 ∧ P_not_B2 = 2/3 ∧ P_B3 = 1/3 ∧ P_not_B3 = 2/3 →
  probability_B_receives_at_least_10_yuan = 11/27 :=
sorry

end part1_part2_l104_104977


namespace complement_A_union_B_in_U_l104_104735

-- Define the universe set U
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | (x - 2) * (x + 1) ≤ 0}

-- Define set B
def B : Set ℝ := {x | 0 ≤ x ∧ x < 3}

-- Define the union of A and B
def A_union_B : Set ℝ := {x | (-1 ≤ x ∧ x < 3)}

-- Define the complement of A ∪ B in U
def C_U_A_union_B : Set ℝ := {x | x < -1 ∨ x ≥ 3}

-- Proof Statement
theorem complement_A_union_B_in_U :
  {x | x < -1 ∨ x ≥ 3} = {x | x ∈ U ∧ (x ∉ A_union_B)} :=
sorry

end complement_A_union_B_in_U_l104_104735


namespace expression_eval_l104_104438

theorem expression_eval :
  5 * 399 + 4 * 399 + 3 * 399 + 397 = 5185 :=
by
  sorry

end expression_eval_l104_104438


namespace rice_grain_difference_l104_104584

theorem rice_grain_difference :
  (3^8) - (3^1 + 3^2 + 3^3 + 3^4 + 3^5) = 6198 :=
by
  sorry

end rice_grain_difference_l104_104584


namespace mr_thompson_third_score_is_78_l104_104387

theorem mr_thompson_third_score_is_78 :
  ∃ (a b c d : ℕ), a < b ∧ b < c ∧ c < d ∧ 
                   (a = 58 ∧ b = 65 ∧ c = 70 ∧ d = 78) ∧ 
                   (a + b + c + d) % 4 = 3 ∧ 
                   (∀ i j k, (a + i + j + k) % 4 = 0) ∧ -- This checks that average is integer
                   c = 78 := sorry

end mr_thompson_third_score_is_78_l104_104387


namespace ship_speed_in_still_water_eq_25_l104_104627

-- Definitions and conditions
variable (x : ℝ) (h1 : 81 / (x + 2) = 69 / (x - 2)) (h2 : x ≠ -2) (h3 : x ≠ 2)

-- Theorem statement
theorem ship_speed_in_still_water_eq_25 : x = 25 :=
by
  sorry

end ship_speed_in_still_water_eq_25_l104_104627


namespace emily_toys_l104_104708

theorem emily_toys (initial_toys sold_toys: Nat) (h₀ : initial_toys = 7) (h₁ : sold_toys = 3) : initial_toys - sold_toys = 4 := by
  sorry

end emily_toys_l104_104708


namespace least_addition_l104_104270

theorem least_addition (a b n : ℕ) (h_a : Nat.Prime a) (h_b : Nat.Prime b) (h_a_val : a = 23) (h_b_val : b = 29) (h_n : n = 1056) :
  ∃ m : ℕ, (m + n) % (a * b) = 0 ∧ m = 278 :=
by
  sorry

end least_addition_l104_104270


namespace green_blue_tile_difference_is_15_l104_104498

def initial_blue_tiles : Nat := 13
def initial_green_tiles : Nat := 6
def second_blue_tiles : Nat := 2 * initial_blue_tiles
def second_green_tiles : Nat := 2 * initial_green_tiles
def border_green_tiles : Nat := 36
def total_blue_tiles : Nat := initial_blue_tiles + second_blue_tiles
def total_green_tiles : Nat := initial_green_tiles + second_green_tiles + border_green_tiles
def tile_difference : Nat := total_green_tiles - total_blue_tiles

theorem green_blue_tile_difference_is_15 : tile_difference = 15 := by
  sorry

end green_blue_tile_difference_is_15_l104_104498


namespace find_divisor_l104_104665

theorem find_divisor (dividend remainder quotient : ℕ) (h1 : dividend = 76) (h2 : remainder = 8) (h3 : quotient = 4) : ∃ d : ℕ, dividend = (d * quotient) + remainder ∧ d = 17 :=
by
  sorry

end find_divisor_l104_104665


namespace abs_gt_not_implies_gt_l104_104355

noncomputable def abs_gt_implies_gt (a b : ℝ) : Prop :=
  |a| > |b| → a > b

theorem abs_gt_not_implies_gt (a b : ℝ) :
  ¬ abs_gt_implies_gt a b :=
sorry

end abs_gt_not_implies_gt_l104_104355


namespace total_tickets_sold_l104_104320

theorem total_tickets_sold (A C : ℕ) (total_revenue : ℝ) (cost_adult cost_child : ℝ) :
  (cost_adult = 6.00) →
  (cost_child = 4.50) →
  (total_revenue = 2100.00) →
  (C = 200) →
  (cost_adult * ↑A + cost_child * ↑C = total_revenue) →
  A + C = 400 :=
by
  intros h1 h2 h3 h4 h5
  -- Proof omitted
  sorry

end total_tickets_sold_l104_104320


namespace tangent_slope_correct_l104_104543

noncomputable def slope_of_directrix (focus: ℝ × ℝ) (p1: ℝ × ℝ) (p2: ℝ × ℝ) : ℝ :=
  let c1 := p1
  let c2 := p2
  let radius1 := Real.sqrt ((c1.1 + 1)^2 + (c1.2 + 1)^2)
  let radius2 := Real.sqrt ((c2.1 - 2)^2 + (c2.2 - 2)^2)
  let dist := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
  let slope := (focus.2 - p1.2) / (focus.1 - p1.1)
  let tangent_slope := (9 : ℝ) / (7 : ℝ) + (4 * Real.sqrt 2) / 7
  tangent_slope

theorem tangent_slope_correct :
  (slope_of_directrix (0, 0) (-1, -1) (2, 2) = (9 + 4 * Real.sqrt 2) / 7) ∨
  (slope_of_directrix (0, 0) (-1, -1) (2, 2) = (9 - 4 * Real.sqrt 2) / 7) :=
by
  -- Proof omitted here
  sorry

end tangent_slope_correct_l104_104543


namespace average_speed_lila_l104_104629

-- Definitions
def distance1 : ℝ := 50 -- miles
def speed1 : ℝ := 20 -- miles per hour
def distance2 : ℝ := 20 -- miles
def speed2 : ℝ := 40 -- miles per hour
def break_time : ℝ := 0.5 -- hours

-- Question to prove: Lila's average speed for the entire ride is 20 miles per hour
theorem average_speed_lila (d1 d2 s1 s2 bt : ℝ) 
  (h1 : d1 = distance1) (h2 : s1 = speed1) (h3 : d2 = distance2) (h4 : s2 = speed2) (h5 : bt = break_time) :
  (d1 + d2) / (d1 / s1 + d2 / s2 + bt) = 20 :=
by
  sorry

end average_speed_lila_l104_104629


namespace percent_swans_non_ducks_l104_104790

def percent_ducks : ℝ := 35
def percent_swans : ℝ := 30
def percent_herons : ℝ := 20
def percent_geese : ℝ := 15
def percent_non_ducks := 100 - percent_ducks

theorem percent_swans_non_ducks : (percent_swans / percent_non_ducks) * 100 = 46.15 := 
by
  sorry

end percent_swans_non_ducks_l104_104790


namespace bob_final_total_score_l104_104633

theorem bob_final_total_score 
  (points_per_correct : ℕ := 5)
  (points_per_incorrect : ℕ := 2)
  (correct_answers : ℕ := 18)
  (incorrect_answers : ℕ := 2) :
  (points_per_correct * correct_answers - points_per_incorrect * incorrect_answers) = 86 :=
by 
  sorry

end bob_final_total_score_l104_104633


namespace cone_volume_l104_104850

theorem cone_volume (V_cylinder V_frustum V_cone : ℝ)
  (h₁ : V_cylinder = 9)
  (h₂ : V_frustum = 63) :
  V_cone = 64 :=
sorry

end cone_volume_l104_104850


namespace part1_solution_set_part2_range_of_a_l104_104041

-- Part 1: Prove the solution set of the inequality f(x) < 6 is (-8/3, 4/3)
theorem part1_solution_set (x : ℝ) :
  (|2 * x + 3| + |x - 1| < 6) ↔ (-8 / 3 : ℝ) < x ∧ x < 4 / 3 :=
by sorry

-- Part 2: Prove the range of values for a that makes f(x) + f(-x) ≥ 5 is (-∞, -3/2] ∪ [3/2, +∞)
theorem part2_range_of_a (a : ℝ) (x : ℝ) :
  (|2 * x + a| + |x - 1| + |-2 * x + a| + |-x - 1| ≥ 5) ↔ 
  (a ≤ -3 / 2 ∨ a ≥ 3 / 2) :=
by sorry

end part1_solution_set_part2_range_of_a_l104_104041


namespace area_of_circular_flower_bed_l104_104287

theorem area_of_circular_flower_bed (C : ℝ) (hC : C = 62.8) : ∃ (A : ℝ), A = 314 :=
by
  sorry

end area_of_circular_flower_bed_l104_104287


namespace A_pow_101_l104_104573

def A : Matrix (Fin 3) (Fin 3) ℝ := ![
  ![0, 0, 1],
  ![1, 0, 0],
  ![0, 1, 0]
]

theorem A_pow_101 :
  A ^ 101 = ![
    ![0, 1, 0],
    ![0, 0, 1],
    ![1, 0, 0]
  ] := by
  sorry

end A_pow_101_l104_104573


namespace jose_profit_share_l104_104189

def investment_share (toms_investment : ℕ) (jose_investment : ℕ) 
  (toms_duration : ℕ) (jose_duration : ℕ) (total_profit : ℕ) : ℕ :=
  let toms_capital_months := toms_investment * toms_duration
  let jose_capital_months := jose_investment * jose_duration
  let total_capital_months := toms_capital_months + jose_capital_months
  let jose_share_ratio := jose_capital_months / total_capital_months
  jose_share_ratio * total_profit

theorem jose_profit_share 
  (toms_investment : ℕ := 3000)
  (jose_investment : ℕ := 4500)
  (toms_duration : ℕ := 12)
  (jose_duration : ℕ := 10)
  (total_profit : ℕ := 6300) :
  investment_share toms_investment jose_investment toms_duration jose_duration total_profit = 3500 := 
sorry

end jose_profit_share_l104_104189


namespace part_I_part_II_l104_104622

open Real

variable (a b : ℝ)

theorem part_I (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + b = 1) : (1 / a^2) + (1 / b^2) ≥ 8 := 
sorry

theorem part_II (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + b = 1) : (1 / a) + (1 / b) + (1 / (a * b)) ≥ 8 := 
sorry

end part_I_part_II_l104_104622


namespace find_ellipse_and_hyperbola_equations_l104_104696

-- Define the conditions
def eccentricity (e : ℝ) (a b : ℝ) : Prop :=
  e = (Real.sqrt (a ^ 2 - b ^ 2)) / a

def focal_distance (f : ℝ) (a b : ℝ) : Prop :=
  f = 2 * Real.sqrt (a ^ 2 + b ^ 2)

-- Define the problem to prove the equations of the ellipse and hyperbola
theorem find_ellipse_and_hyperbola_equations (a b : ℝ) (e : ℝ) (f : ℝ)
  (h1 : eccentricity e a b) (h2 : focal_distance f a b) 
  (h3 : e = 4 / 5) (h4 : f = 2 * Real.sqrt 34) 
  (h5 : a > b) (h6 : 0 < b) :
  (a^2 = 25 ∧ b^2 = 9) → 
  (∀ x y, (x^2 / 25 + y^2 / 9 = 1) ∧ (x^2 / 25 - y^2 / 9 = 1)) :=
sorry

end find_ellipse_and_hyperbola_equations_l104_104696


namespace gross_profit_value_l104_104484

theorem gross_profit_value (sales_price : ℝ) (cost : ℝ) (gross_profit : ℝ) 
    (h1 : sales_price = 54) 
    (h2 : gross_profit = 1.25 * cost) 
    (h3 : sales_price = cost + gross_profit): gross_profit = 30 := 
  sorry

end gross_profit_value_l104_104484


namespace tank_height_l104_104885

theorem tank_height
  (r_A r_B h_A h_B : ℝ)
  (h₁ : 8 = 2 * Real.pi * r_A)
  (h₂ : h_B = 8)
  (h₃ : 10 = 2 * Real.pi * r_B)
  (h₄ : π * r_A ^ 2 * h_A = 0.56 * (π * r_B ^ 2 * h_B)) :
  h_A = 7 :=
sorry

end tank_height_l104_104885


namespace total_canoes_built_l104_104198

def geometric_sum (a r n : ℕ) : ℕ :=
  a * ((r^n - 1) / (r - 1))

theorem total_canoes_built : geometric_sum 10 3 7 = 10930 := 
  by
    -- The proof will go here.
    sorry

end total_canoes_built_l104_104198


namespace damaged_books_count_l104_104354

variables (o d : ℕ)

theorem damaged_books_count (h1 : o + d = 69) (h2 : o = 6 * d - 8) : d = 11 := 
by 
  sorry

end damaged_books_count_l104_104354


namespace moles_of_HCl_needed_l104_104836

theorem moles_of_HCl_needed : ∀ (moles_KOH : ℕ), moles_KOH = 2 →
  (moles_HCl : ℕ) → moles_HCl = 2 :=
by
  sorry

end moles_of_HCl_needed_l104_104836


namespace index_cards_per_pack_l104_104743

-- Definitions of the conditions
def students_per_period := 30
def periods_per_day := 6
def index_cards_per_student := 10
def total_spent := 108
def pack_cost := 3

-- Helper Definitions
def total_students := periods_per_day * students_per_period
def total_index_cards_needed := total_students * index_cards_per_student
def packs_bought := total_spent / pack_cost

-- Theorem to prove
theorem index_cards_per_pack :
  total_index_cards_needed / packs_bought = 50 := by
  sorry

end index_cards_per_pack_l104_104743


namespace problem_inequality_l104_104868

theorem problem_inequality (k m n : ℕ) (hk1 : 1 < k) (hkm : k ≤ m) (hmn : m < n) :
  (1 + m) ^ 2 > (1 + n) ^ m :=
  sorry

end problem_inequality_l104_104868


namespace range_of_m_l104_104736

def p (m : ℝ) : Prop := ∀ x : ℝ, ¬ (x ^ 2 - 2 * m * x + 1 < 0)
def q (m : ℝ) : Prop := ∃ x y : ℝ, (x ^ 2) / (m - 2) + (y ^ 2) / m = 1

theorem range_of_m (m : ℝ) :
  (∃ x y : ℝ, (x ^ 2) / (m - 2) + (y ^ 2) / m = 1 ∨ ∀ x : ℝ, ¬ (x ^ 2 - 2 * m * x + 1 < 0))
  ∧ (¬ (∀ x : ℝ, ¬ (x ^ 2 - 2 * m * x + 1 < 0) → ∃ x y : ℝ, (x ^ 2) / (m - 2) + (y ^ 2) / m = 1)) ↔
  (-1 ≤ m ∧ m ≤ 0) ∨ (1 < m ∧ m < 2) :=
  sorry

end range_of_m_l104_104736


namespace domain_of_f_univ_l104_104958

noncomputable def f (x : ℝ) : ℝ := (x^2 - 2 * x + 1)^(1 / 3) + (9 - x^2)^(1 / 3)

theorem domain_of_f_univ : ∀ x : ℝ, true :=
by
  intro x
  sorry

end domain_of_f_univ_l104_104958


namespace max_lessons_l104_104070

theorem max_lessons (x y z : ℕ) 
  (h1 : 3 * y * z = 18) 
  (h2 : 3 * x * z = 63) 
  (h3 : 3 * x * y = 42) :
  3 * x * y * z = 126 :=
by
  sorry

end max_lessons_l104_104070


namespace find_theta_l104_104554

-- Definitions based on conditions
def angle_A : ℝ := 10
def angle_B : ℝ := 14
def angle_C : ℝ := 26
def angle_D : ℝ := 33
def sum_rect_angles : ℝ := 360
def sum_triangle_angles : ℝ := 180
def sum_right_triangle_acute_angles : ℝ := 90

-- Main theorem statement
theorem find_theta (A B C D : ℝ)
  (hA : A = angle_A)
  (hB : B = angle_B)
  (hC : C = angle_C)
  (hD : D = angle_D)
  (sum_rect : sum_rect_angles = 360)
  (sum_triangle : sum_triangle_angles = 180) :
  ∃ θ : ℝ, θ = 11 := 
sorry

end find_theta_l104_104554


namespace find_d_l104_104581

noncomputable def quadratic_roots (d : ℝ) : Prop :=
∀ x : ℝ, x^2 + 7*x + d = 0 ↔ x = (-7 + Real.sqrt d) / 2 ∨ x = (-7 - Real.sqrt d) / 2

theorem find_d : ∃ d : ℝ, d = 9.8 ∧ quadratic_roots d :=
sorry

end find_d_l104_104581


namespace bernoulli_inequality_l104_104330

theorem bernoulli_inequality (x : ℝ) (n : ℕ) (hx : x ≥ -1) (hn : n ≥ 1) : (1 + x)^n ≥ 1 + n * x :=
by sorry

end bernoulli_inequality_l104_104330


namespace sum_of_fourth_powers_l104_104504

theorem sum_of_fourth_powers (a b c : ℝ) 
  (h1 : a + b + c = 2)
  (h2 : a^2 + b^2 + c^2 = 5)
  (h3 : a^3 + b^3 + c^3 = 8) :
  a^4 + b^4 + c^4 = 18.5 :=
sorry

end sum_of_fourth_powers_l104_104504


namespace function_in_second_quadrant_l104_104435

theorem function_in_second_quadrant (k : ℝ) : (∀ x₁ x₂ : ℝ, x₁ < 0 → x₂ < 0 → x₁ < x₂ → (k / x₁ < k / x₂)) → (∀ x : ℝ, x < 0 → (k > 0)) :=
sorry

end function_in_second_quadrant_l104_104435


namespace solve_quadratic_expr_l104_104463

theorem solve_quadratic_expr (x : ℝ) (h : 2 * x^2 - 5 = 11) : 
  4 * x^2 + 4 * x + 1 = 33 + 8 * Real.sqrt 2 ∨ 4 * x^2 + 4 * x + 1 = 33 - 8 * Real.sqrt 2 := 
by 
  sorry

end solve_quadratic_expr_l104_104463


namespace revenue_effect_l104_104710

noncomputable def price_increase_factor : ℝ := 1.425
noncomputable def sales_decrease_factor : ℝ := 0.627

theorem revenue_effect (P Q R_new : ℝ) (h_price_increase : P ≠ 0) (h_sales_decrease : Q ≠ 0) :
  R_new = (P * price_increase_factor) * (Q * sales_decrease_factor) →
  ((R_new - P * Q) / (P * Q)) * 100 = -10.6825 :=
by
  sorry

end revenue_effect_l104_104710


namespace Gina_makes_30_per_hour_l104_104660

variable (rose_cups_per_hour lily_cups_per_hour : ℕ)
variable (rose_cup_order lily_cup_order total_payment : ℕ)
variable (total_hours : ℕ)

def Gina_hourly_rate (rose_cups_per_hour: ℕ) (lily_cups_per_hour: ℕ) (rose_cup_order: ℕ) (lily_cup_order: ℕ) (total_payment: ℕ) : Prop :=
    let rose_time := rose_cup_order / rose_cups_per_hour
    let lily_time := lily_cup_order / lily_cups_per_hour
    let total_time := rose_time + lily_time
    total_payment / total_time = total_hours

theorem Gina_makes_30_per_hour :
    let rose_cups_per_hour := 6
    let lily_cups_per_hour := 7
    let rose_cup_order := 6
    let lily_cup_order := 14
    let total_payment := 90
    Gina_hourly_rate rose_cups_per_hour lily_cups_per_hour rose_cup_order lily_cup_order total_payment 30 :=
by
    sorry

end Gina_makes_30_per_hour_l104_104660


namespace min_sugar_l104_104569

variable (f s : ℝ)

theorem min_sugar (h1 : f ≥ 10 + 3 * s) (h2 : f ≤ 4 * s) : s ≥ 10 := by
  sorry

end min_sugar_l104_104569


namespace rational_solutions_k_l104_104677

theorem rational_solutions_k (k : ℕ) (hpos : k > 0) : (∃ x : ℚ, k * x^2 + 22 * x + k = 0) ↔ k = 11 :=
by
  sorry

end rational_solutions_k_l104_104677


namespace smallest_integer_in_correct_range_l104_104223

theorem smallest_integer_in_correct_range :
  ∃ (n : ℤ), n > 1 ∧ n % 3 = 1 ∧ n % 5 = 1 ∧ n % 8 = 1 ∧ n % 7 = 2 ∧ 161 ≤ n ∧ n ≤ 200 :=
by
  sorry

end smallest_integer_in_correct_range_l104_104223


namespace intersection_claim_union_claim_l104_104957

def A : Set ℝ := {x | (x - 2) * (x + 5) < 0}
def B : Set ℝ := {x | x^2 - 2 * x - 3 ≥ 0}
def U : Set ℝ := Set.univ

-- Claim 1: Prove that A ∩ B = {x | -5 < x ∧ x ≤ -1}
theorem intersection_claim : A ∩ B = {x | -5 < x ∧ x ≤ -1} :=
by
  sorry

-- Claim 2: Prove that A ∪ (U \ B) = {x | -5 < x ∧ x < 3}
theorem union_claim : A ∪ (U \ B) = {x | -5 < x ∧ x < 3} :=
by
  sorry

end intersection_claim_union_claim_l104_104957


namespace total_water_in_containers_l104_104449

/-
We have four containers. The first three contain water, while the fourth is empty. 
The second container holds twice as much water as the first, and the third holds twice as much water as the second. 
We transfer half of the water from the first container, one-third of the water from the second container, 
and one-quarter of the water from the third container into the fourth container. 
Now, there are 26 liters of water in the fourth container. Prove that initially, 
there were 84 liters of water in total in the first three containers.
-/

theorem total_water_in_containers (x : ℕ) (h1 : x / 2 + 2 * x / 3 + x = 26) : x + 2 * x + 4 * x = 84 := 
sorry

end total_water_in_containers_l104_104449


namespace last_three_digits_of_power_l104_104158

theorem last_three_digits_of_power (h : 3^400 ≡ 1 [MOD 800]) : 3^8000 ≡ 1 [MOD 800] :=
by {
  sorry
}

end last_three_digits_of_power_l104_104158


namespace no_2000_digit_perfect_square_with_1999_digits_of_5_l104_104832

theorem no_2000_digit_perfect_square_with_1999_digits_of_5 :
  ¬ (∃ n : ℕ,
      (Nat.digits 10 n).length = 2000 ∧
      ∃ k : ℕ, n = k * k ∧
      (Nat.digits 10 n).count 5 ≥ 1999) :=
sorry

end no_2000_digit_perfect_square_with_1999_digits_of_5_l104_104832


namespace chess_group_unique_pairings_l104_104921

theorem chess_group_unique_pairings:
  ∀ (players games : ℕ), players = 50 → games = 1225 →
  (∃ (games_per_pair : ℕ), games_per_pair = 1 ∧ (∀ p: ℕ, p < players → (players - 1) * games_per_pair = games)) :=
by
  sorry

end chess_group_unique_pairings_l104_104921


namespace inequality_solution_set_l104_104309

theorem inequality_solution_set : 
  (∃ (x : ℝ), (4 / (x - 1) ≤ x - 1) ↔ (x ≥ 3 ∨ (-1 ≤ x ∧ x < 1))) :=
by
  sorry

end inequality_solution_set_l104_104309


namespace ratio_of_men_to_women_l104_104412

theorem ratio_of_men_to_women 
  (M W : ℕ) 
  (h1 : W = M + 5) 
  (h2 : M + W = 15): M = 5 ∧ W = 10 ∧ (M + W) / Nat.gcd M W = 1 ∧ (W + M) / Nat.gcd M W = 2 :=
by 
  sorry

end ratio_of_men_to_women_l104_104412


namespace stratified_sampling_number_of_products_drawn_l104_104873

theorem stratified_sampling_number_of_products_drawn (T S W X : ℕ) 
  (h1 : T = 1024) (h2 : S = 64) (h3 : W = 128) :
  X = S * (W / T) → X = 8 :=
by
  sorry

end stratified_sampling_number_of_products_drawn_l104_104873


namespace expected_total_rain_l104_104662

noncomputable def expected_daily_rain : ℝ :=
  (0.50 * 0) + (0.30 * 3) + (0.20 * 8)

theorem expected_total_rain :
  (5 * expected_daily_rain) = 12.5 :=
by
  sorry

end expected_total_rain_l104_104662


namespace jesse_gave_pencils_l104_104726

theorem jesse_gave_pencils (initial_pencils : ℕ) (final_pencils : ℕ) (pencils_given : ℕ) :
  initial_pencils = 78 → final_pencils = 34 → pencils_given = initial_pencils - final_pencils → pencils_given = 44 :=
by
  intro h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end jesse_gave_pencils_l104_104726


namespace length_of_AC_l104_104576

variable (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
variables (AB BC AC : ℝ)
variables (right_triangle : AB ^ 2 + BC ^ 2 = AC ^ 2)
variables (tan_A : BC / AB = 4 / 3)
variable (AB_val : AB = 4)

theorem length_of_AC :
  AC = 20 / 3 :=
sorry

end length_of_AC_l104_104576


namespace problem_l104_104257

variables {a b c d : ℝ}

theorem problem (h1 : c + d = 14 * a) (h2 : c * d = 15 * b) (h3 : a + b = 14 * c) (h4 : a * b = 15 * d) (distinct : a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a) :
  a + b + c + d = 3150 := sorry

end problem_l104_104257


namespace calculate_expression_l104_104796

theorem calculate_expression :
  4 * Real.sqrt 24 * (Real.sqrt 6 / 8) / Real.sqrt 3 - 3 * Real.sqrt 3 = -Real.sqrt 3 :=
by
  sorry

end calculate_expression_l104_104796


namespace executed_is_9_l104_104365

-- Define the conditions based on given problem
variables (x K I : ℕ)

-- Condition 1: Number of killed
def number_killed (x : ℕ) : ℕ := 2 * x + 4

-- Condition 2: Number of injured
def number_injured (x : ℕ) : ℕ := (16 * x) / 3 + 8

-- Condition 3: Total of killed, injured, and executed is less than 98
def total_less_than_98 (x : ℕ) (k : ℕ) (i : ℕ) : Prop := k + i + x < 98

-- Condition 4: Relation between killed and executed
def killed_relation (x : ℕ) (k : ℕ) : Prop := k - 4 = 2 * x

-- The final theorem statement to prove
theorem executed_is_9 : ∃ x, number_killed x = 2 * x + 4 ∧
                       number_injured x = (16 * x) / 3 + 8 ∧
                       total_less_than_98 x (number_killed x) (number_injured x) ∧
                       killed_relation x (number_killed x) ∧
                       x = 9 :=
by
  sorry

end executed_is_9_l104_104365


namespace isosceles_right_triangle_area_l104_104371

theorem isosceles_right_triangle_area (a b : ℝ) (h₁ : a = b) (h₂ : a + b = 20) : 
  (1 / 2) * a * b = 50 := 
by 
  sorry

end isosceles_right_triangle_area_l104_104371


namespace solve_equation_l104_104655

theorem solve_equation (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ -2) :
    (3 / (x + 2) - 1 / x = 0) → x = 1 :=
  by sorry

end solve_equation_l104_104655


namespace value_of_b_l104_104018

theorem value_of_b (a b : ℕ) (h1 : a * b = 2 * (a + b) + 10) (h2 : b - a = 5) : b = 9 := 
by {
  -- Proof is not required, so we use sorry to complete the statement
  sorry
}

end value_of_b_l104_104018


namespace cards_problem_l104_104586

-- Definitions of the cards and their arrangement
def cards : List ℕ := [1, 3, 4, 6, 7, 8]
def missing_numbers : List ℕ := [2, 5, 9]

-- Function to check no three consecutive numbers are in ascending or descending order
def no_three_consec (ls : List ℕ) : Prop :=
  ∀ (a b c : ℕ), a < b → b < c → b - a = 1 → c - b = 1 → False ∧
                a > b → b > c → a - b = 1 → b - c = 1 → False

-- Assume that cards A, B, and C are not visible
variables (A B C : ℕ)

-- Ensure that A, B, and C are among the missing numbers
axiom A_in_missing : A ∈ missing_numbers
axiom B_in_missing : B ∈ missing_numbers
axiom C_in_missing : C ∈ missing_numbers

-- Ensuring no three consecutive cards are in ascending or descending order
axiom no_three_consec_cards : no_three_consec (cards ++ [A, B, C])

-- The final proof problem
theorem cards_problem : A = 5 ∧ B = 2 ∧ C = 9 :=
by
  sorry

end cards_problem_l104_104586


namespace solve_inequality_l104_104142

theorem solve_inequality (x : ℝ) (h : 1 / (x - 1) < -1) : 0 < x ∧ x < 1 :=
sorry

end solve_inequality_l104_104142


namespace xy_problem_l104_104659

theorem xy_problem (x y : ℝ) (h1 : (x + y)^2 = 36) (h2 : x * y = 8) : x^2 + y^2 = 20 :=
by
  sorry

end xy_problem_l104_104659


namespace point_C_values_l104_104169

variable (B C : ℝ)
variable (distance_BC : ℝ)
variable (hB : B = 3)
variable (hDistance : distance_BC = 2)

theorem point_C_values (hBC : abs (C - B) = distance_BC) : (C = 1 ∨ C = 5) := 
by
  sorry

end point_C_values_l104_104169


namespace solve_fractional_eq1_l104_104882

theorem solve_fractional_eq1 : ¬ ∃ (x : ℝ), 1 / (x - 2) = (1 - x) / (2 - x) - 3 :=
by sorry

end solve_fractional_eq1_l104_104882


namespace crosswalk_red_light_wait_l104_104372

theorem crosswalk_red_light_wait :
  let red_light_duration := 40
  let wait_time_requirement := 15
  let favorable_duration := red_light_duration - wait_time_requirement
  (favorable_duration : ℝ) / red_light_duration = (5 : ℝ) / 8 :=
by
  sorry

end crosswalk_red_light_wait_l104_104372


namespace sally_quarters_l104_104950

theorem sally_quarters (initial_quarters spent_quarters final_quarters : ℕ) 
  (h1 : initial_quarters = 760) 
  (h2 : spent_quarters = 418) 
  (calc_final : final_quarters = initial_quarters - spent_quarters) : 
  final_quarters = 342 := 
by 
  rw [h1, h2] at calc_final 
  exact calc_final

end sally_quarters_l104_104950


namespace strategy_for_antonio_l104_104216

-- We define the concept of 'winning' and 'losing' positions
def winning_position (m n : ℕ) : Prop :=
  ¬ (m % 2 = 0 ∧ n % 2 = 0)

-- Now create the main theorem
theorem strategy_for_antonio (m n : ℕ) : winning_position m n ↔ 
  (¬(m % 2 = 0 ∧ n % 2 = 0)) :=
by
  unfold winning_position
  sorry

end strategy_for_antonio_l104_104216


namespace probability_greater_difficulty_probability_same_difficulty_l104_104553

/-- A datatype representing the difficulty levels of questions. -/
inductive Difficulty
| easy : Difficulty
| medium : Difficulty
| difficult : Difficulty

/-- A datatype representing the four questions with their difficulties. -/
inductive Question
| A1 : Question
| A2 : Question
| B : Question
| C : Question

/-- The function to get the difficulty of a question. -/
def difficulty (q : Question) : Difficulty :=
  match q with
  | Question.A1 => Difficulty.easy
  | Question.A2 => Difficulty.easy
  | Question.B  => Difficulty.medium
  | Question.C  => Difficulty.difficult

/-- The set of all possible pairings of questions selected by two students A and B. -/
def all_pairs : List (Question × Question) :=
  [ (Question.A1, Question.A1), (Question.A1, Question.A2), (Question.A1, Question.B), (Question.A1, Question.C),
    (Question.A2, Question.A1), (Question.A2, Question.A2), (Question.A2, Question.B), (Question.A2, Question.C),
    (Question.B, Question.A1), (Question.B, Question.A2), (Question.B, Question.B), (Question.B, Question.C),
    (Question.C, Question.A1), (Question.C, Question.A2), (Question.C, Question.B), (Question.C, Question.C) ]

/-- The event that the difficulty of the question selected by student A is greater than that selected by student B. -/
def event_N : List (Question × Question) :=
  [ (Question.B, Question.A1), (Question.B, Question.A2), (Question.C, Question.A1), (Question.C, Question.A2), (Question.C, Question.B) ]

/-- The event that the difficulties of the questions selected by both students are the same. -/
def event_M : List (Question × Question) :=
  [ (Question.A1, Question.A1), (Question.A1, Question.A2), (Question.A2, Question.A1), (Question.A2, Question.A2), 
    (Question.B, Question.B), (Question.C, Question.C) ]

/-- The probabilities of the events. -/
noncomputable def probability_event_N : ℚ := (event_N.length : ℚ) / (all_pairs.length : ℚ)
noncomputable def probability_event_M : ℚ := (event_M.length : ℚ) / (all_pairs.length : ℚ)

/-- The theorem statements -/
theorem probability_greater_difficulty : probability_event_N = 5 / 16 := sorry
theorem probability_same_difficulty : probability_event_M = 3 / 8 := sorry

end probability_greater_difficulty_probability_same_difficulty_l104_104553


namespace xyz_inequality_l104_104490

theorem xyz_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z + x * y + y * z + z * x = 4) : x + y + z ≥ 3 := 
by
  sorry

end xyz_inequality_l104_104490


namespace problem1_problem2_l104_104291

-- Definitions of sets A and B
def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 3}
def B : Set ℝ := {x | x > 5 ∨ x < -1}

-- First problem: A ∩ B
theorem problem1 (a : ℝ) (ha : a = 4) : A a ∩ B = {x | 6 < x ∧ x ≤ 7} :=
by sorry

-- Second problem: A ∪ B = B
theorem problem2 (a : ℝ) : (A a ∪ B = B) ↔ (a < -4 ∨ a > 5) :=
by sorry

end problem1_problem2_l104_104291


namespace beth_wins_if_arjun_plays_first_l104_104618

/-- 
In the game where players take turns removing one, two adjacent, or two non-adjacent bricks from 
walls, given certain configurations, the configuration where Beth has a guaranteed winning 
strategy if Arjun plays first is (7, 3, 1).
-/
theorem beth_wins_if_arjun_plays_first :
  let nim_value_1 := 1
  let nim_value_2 := 2
  let nim_value_3 := 3
  let nim_value_7 := 2 -- computed as explained in the solution
  ∀ config : List ℕ,
    config = [7, 1, 1] ∨ config = [7, 2, 1] ∨ config = [7, 2, 2] ∨ config = [7, 3, 1] ∨ config = [7, 3, 2] →
    match config with
    | [7, 3, 1] => true
    | _ => false :=
by
  sorry

end beth_wins_if_arjun_plays_first_l104_104618


namespace tan_150_degree_is_correct_l104_104319

noncomputable def tan_150_degree_is_negative_sqrt_3_div_3 : Prop :=
  let theta := Real.pi * 150 / 180
  let ref_angle := Real.pi * 30 / 180
  let cos_150 := -Real.cos ref_angle
  let sin_150 := Real.sin ref_angle
  Real.tan theta = -Real.sqrt 3 / 3

theorem tan_150_degree_is_correct :
  tan_150_degree_is_negative_sqrt_3_div_3 :=
by
  sorry

end tan_150_degree_is_correct_l104_104319


namespace analysis_method_inequality_l104_104037

def analysis_method_seeks (inequality : Prop) : Prop :=
  ∃ (sufficient_condition : Prop), (inequality → sufficient_condition)

theorem analysis_method_inequality (inequality : Prop) :
  (∃ sufficient_condition, (inequality → sufficient_condition)) :=
sorry

end analysis_method_inequality_l104_104037


namespace quadratic_distinct_roots_l104_104787

theorem quadratic_distinct_roots (c : ℝ) (h : c < 1 / 4) : 
  ∃ r1 r2 : ℝ, r1 ≠ r2 ∧ (r1 * r1 + 2 * r1 + 4 * c = 0) ∧ (r2 * r2 + 2 * r2 + 4 * c = 0) :=
by 
sorry

end quadratic_distinct_roots_l104_104787


namespace scientific_notation_of_510000000_l104_104952

theorem scientific_notation_of_510000000 :
  (510000000 : ℝ) = 5.1 * 10^8 := 
sorry

end scientific_notation_of_510000000_l104_104952


namespace find_d_minus_a_l104_104582

theorem find_d_minus_a (a b c d : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h1 : a * b = 240)
  (h2 : (b + c) / 2 = 60)
  (h3 : (c + d) / 2 = 90) : d - a = 116 :=
sorry

end find_d_minus_a_l104_104582


namespace triangle_equilateral_l104_104676

noncomputable def is_equilateral {R p : ℝ} (A B C : ℝ) : Prop :=
  R * (Real.tan A + Real.tan B + Real.tan C) = 2 * p  →
  ∀ {a b c : ℝ}, a = b ∧ b = c ∧ c = a

theorem triangle_equilateral
  {A B C : ℝ}
  {R p : ℝ}
  (h : R * (Real.tan A + Real.tan B + Real.tan C) = 2 * p) :
  ∀ {a b c : ℝ}, a = b ∧ b = c ∧ c = a :=
sorry

end triangle_equilateral_l104_104676


namespace average_billboards_per_hour_l104_104226

-- Define the number of billboards seen in each hour
def billboards_first_hour := 17
def billboards_second_hour := 20
def billboards_third_hour := 23

-- Define the number of hours
def total_hours := 3

-- Prove that the average number of billboards per hour is 20
theorem average_billboards_per_hour : 
  (billboards_first_hour + billboards_second_hour + billboards_third_hour) / total_hours = 20 :=
by
  sorry

end average_billboards_per_hour_l104_104226


namespace handshakes_min_l104_104682

-- Define the number of people and the number of handshakes each person performs
def numPeople : ℕ := 35
def handshakesPerPerson : ℕ := 3

-- Define the minimum possible number of unique handshakes
theorem handshakes_min : (numPeople * handshakesPerPerson) / 2 = 105 := by
  sorry

end handshakes_min_l104_104682


namespace inversely_proportional_ratio_l104_104186

theorem inversely_proportional_ratio (x y x1 x2 y1 y2 : ℝ) 
  (h_inv_prop : x * y = x1 * y2) 
  (h_ratio : x1 / x2 = 3 / 5) 
  (x1_nonzero : x1 ≠ 0) 
  (x2_nonzero : x2 ≠ 0) 
  (y1_nonzero : y1 ≠ 0) 
  (y2_nonzero : y2 ≠ 0) : 
  y1 / y2 = 5 / 3 := 
sorry

end inversely_proportional_ratio_l104_104186


namespace fraction_equivalence_l104_104376

theorem fraction_equivalence (x y : ℝ) (h : x ≠ y) :
  (x - y)^2 / (x^2 - y^2) = (x - y) / (x + y) :=
by
  sorry

end fraction_equivalence_l104_104376


namespace minimum_value_of_expression_l104_104246

theorem minimum_value_of_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (sum_eq : x + y + z = 5) :
  (9 / x + 4 / y + 25 / z) ≥ 20 :=
sorry

end minimum_value_of_expression_l104_104246


namespace probability_x_y_less_than_3_l104_104734

theorem probability_x_y_less_than_3 :
  let A := 6 * 2
  let triangle_area := (1 / 2) * 3 * 2
  let P := triangle_area / A
  P = 1 / 4 := by sorry

end probability_x_y_less_than_3_l104_104734


namespace solve_quadratic_eq_l104_104535

theorem solve_quadratic_eq (b c : ℝ) :
  (∀ x : ℝ, |x - 3| = 4 ↔ x = 7 ∨ x = -1) →
  (∀ x : ℝ, x^2 + b * x + c = 0 ↔ x = 7 ∨ x = -1) →
  b = -6 ∧ c = -7 :=
by
  intros h_abs_val_eq h_quad_eq
  sorry

end solve_quadratic_eq_l104_104535


namespace initially_calculated_average_weight_l104_104177

theorem initially_calculated_average_weight (n : ℕ) (misread_diff correct_avg_weight : ℝ)
  (hn : n = 20) (hmisread_diff : misread_diff = 10) (hcorrect_avg_weight : correct_avg_weight = 58.9) :
  ((correct_avg_weight * n - misread_diff) / n) = 58.4 :=
by
  rw [hn, hmisread_diff, hcorrect_avg_weight]
  sorry

end initially_calculated_average_weight_l104_104177


namespace initial_pens_eq_42_l104_104344

-- Definitions based on the conditions
def initial_books : ℕ := 143
def remaining_books : ℕ := 113
def remaining_pens : ℕ := 19
def sold_pens : ℕ := 23

-- Theorem to prove that the initial number of pens was 42
theorem initial_pens_eq_42 (b_init b_remain p_remain p_sold : ℕ) 
    (H_b_init : b_init = initial_books)
    (H_b_remain : b_remain = remaining_books)
    (H_p_remain : p_remain = remaining_pens)
    (H_p_sold : p_sold = sold_pens) : 
    (p_sold + p_remain = 42) := 
by {
    -- Provide proof later
    sorry
}

end initial_pens_eq_42_l104_104344


namespace midpoint_sum_l104_104579

theorem midpoint_sum (x1 y1 x2 y2 : ℝ) (hx1 : x1 = 10) (hy1 : y1 = 3) (hx2 : x2 = -4) (hy2 : y2 = -7) :
  (x1 + x2) / 2 + (y1 + y2) / 2 = 1 :=
by
  rw [hx1, hy1, hx2, hy2]
  norm_num

end midpoint_sum_l104_104579


namespace divisibility_by_3_l104_104499

theorem divisibility_by_3 (a b c : ℤ) (h1 : c ≠ b)
    (h2 : ∃ x : ℂ, (a * x^2 + b * x + c = 0 ∧ (c - b) * x^2 + (c - a) * x + (a + b) = 0)) :
    3 ∣ (a + b + 2 * c) :=
by
  sorry

end divisibility_by_3_l104_104499


namespace minimum_value_of_x_plus_y_existence_of_minimum_value_l104_104143

theorem minimum_value_of_x_plus_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y + 2 * x + y = 8) :
  x + y ≥ 2 * Real.sqrt 10 - 3 :=
sorry

theorem existence_of_minimum_value (x y : ℝ) :
  ∃ x y, x > 0 ∧ y > 0 ∧ x * y + 2 * x + y = 8 ∧ x + y = 2 * Real.sqrt 10 - 3 :=
sorry

end minimum_value_of_x_plus_y_existence_of_minimum_value_l104_104143


namespace fraction_identity_l104_104274

theorem fraction_identity :
  ( (2^4 - 1) / (2^4 + 1) * (3^4 - 1) / (3^4 + 1) * (4^4 - 1) / (4^4 + 1) * (5^4 - 1) / (5^4 + 1) = (432 / 1105) ) :=
by
  sorry

end fraction_identity_l104_104274


namespace lions_deers_15_minutes_l104_104328

theorem lions_deers_15_minutes :
  ∀ (n : ℕ), (15 * n = 15 * 15 → n = 15 → ∀ t, t = 15) := by
  sorry

end lions_deers_15_minutes_l104_104328


namespace polynomial_root_reciprocal_square_sum_l104_104265

theorem polynomial_root_reciprocal_square_sum :
  ∀ (a b c : ℝ), (a + b + c = 6) → (a * b + b * c + c * a = 11) → (a * b * c = 6) →
  (1 / a ^ 2 + 1 / b ^ 2 + 1 / c ^ 2 = 49 / 36) :=
by
  intros a b c h_sum h_prod_sum h_prod
  sorry

end polynomial_root_reciprocal_square_sum_l104_104265


namespace triangle_area_l104_104199

theorem triangle_area (a b : ℝ) (h1 : b = (24 / a)) (h2 : 3 * 4 + a * (12 / a) = 12) : b = 3 / 2 :=
by
  sorry

end triangle_area_l104_104199


namespace gracie_height_is_56_l104_104016

noncomputable def Gracie_height : Nat := 56

theorem gracie_height_is_56 : Gracie_height = 56 := by
  sorry

end gracie_height_is_56_l104_104016


namespace subtract_500_from_sum_of_calculations_l104_104991

theorem subtract_500_from_sum_of_calculations (x : ℕ) (h : 423 - x = 421) : 
  (421 + 423 * x) - 500 = 767 := 
by
  sorry

end subtract_500_from_sum_of_calculations_l104_104991


namespace inequality_solution_l104_104134

theorem inequality_solution (x : ℝ) :
  (2 / (x - 2) - 5 / (x - 3) + 5 / (x - 4) - 2 / (x - 5) < 1 / 15) ↔ (x < 1 ∨ x > 3) ∧ (x ≠ 2 ∧ x ≠ 3 ∧ x ≠ 4 ∧ x ≠ 5) :=
by
  sorry

end inequality_solution_l104_104134


namespace solution_set_of_inequality_l104_104340

theorem solution_set_of_inequality (x : ℝ) : x * (2 - x) ≤ 0 ↔ x ≤ 0 ∨ x ≥ 2 := by
  sorry

end solution_set_of_inequality_l104_104340


namespace plane_equation_proof_l104_104440

-- Define the parametric representation of the plane
def plane_parametric (s t : ℝ) : ℝ × ℝ × ℝ :=
  (2 + 2 * s - t, 1 + 2 * s, 4 - s + 3 * t)

-- Define the plane equation form
def plane_equation (x y z : ℝ) (A B C D : ℤ) : Prop :=
  (A : ℝ) * x + (B : ℝ) * y + (C : ℝ) * z + (D : ℝ) = 0

-- Define the normal vector derived from the cross product
def normal_vector : ℝ × ℝ × ℝ := (6, -5, 2)

-- Define the initial point used to calculate D
def initial_point : ℝ × ℝ × ℝ := (2, 1, 4)

-- Proposition to prove the equation of the plane
theorem plane_equation_proof :
  ∃ (A B C D : ℤ), A = 6 ∧ B = -5 ∧ C = 2 ∧ D = -15 ∧
    ∀ x y z : ℝ, plane_equation x y z A B C D ↔
      ∃ s t : ℝ, plane_parametric s t = (x, y, z) :=
by
  sorry

end plane_equation_proof_l104_104440


namespace scientific_notation_of_8_36_billion_l104_104804

theorem scientific_notation_of_8_36_billion : 
  ∃ (a : ℝ) (n : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ 8.36 * 10^9 = a * 10^n := 
by
  use 8.36
  use 9
  simp
  sorry

end scientific_notation_of_8_36_billion_l104_104804


namespace bridge_length_correct_l104_104600

noncomputable def length_of_bridge : ℝ :=
  let train_length := 110 -- in meters
  let train_speed_kmh := 72 -- in km/hr
  let crossing_time := 14.248860091192705 -- in seconds
  let speed_in_mps := train_speed_kmh * (1000 / 3600)
  let distance := speed_in_mps * crossing_time
  distance - train_length

theorem bridge_length_correct :
  length_of_bridge = 174.9772018238541 := by
  sorry

end bridge_length_correct_l104_104600


namespace final_result_is_102_l104_104308

-- Definitions and conditions from the problem
def chosen_number : ℕ := 120
def multiplied_result : ℕ := 2 * chosen_number
def final_result : ℕ := multiplied_result - 138

-- The proof statement
theorem final_result_is_102 : final_result = 102 := 
by 
sorry

end final_result_is_102_l104_104308


namespace system_solution_l104_104641

theorem system_solution (x y z a : ℝ) (h1 : x + y + z = 1) (h2 : 1/x + 1/y + 1/z = 1) (h3 : x * y * z = a) :
    (x = 1 ∧ y = Real.sqrt (-a) ∧ z = -Real.sqrt (-a)) ∨
    (x = 1 ∧ y = -Real.sqrt (-a) ∧ z = Real.sqrt (-a)) ∨
    (x = Real.sqrt (-a) ∧ y = -Real.sqrt (-a) ∧ z = 1) ∨
    (x = -Real.sqrt (-a) ∧ y = Real.sqrt (-a) ∧ z = 1) ∨
    (x = Real.sqrt (-a) ∧ y = 1 ∧ z = -Real.sqrt (-a)) ∨
    (x = -Real.sqrt (-a) ∧ y = 1 ∧ z = Real.sqrt (-a)) :=
sorry

end system_solution_l104_104641


namespace total_reading_materials_l104_104721

theorem total_reading_materials 
  (magazines : ℕ) 
  (newspapers : ℕ) 
  (h_magazines : magazines = 425) 
  (h_newspapers : newspapers = 275) : 
  magazines + newspapers = 700 := 
by 
  sorry

end total_reading_materials_l104_104721


namespace sum_of_reciprocals_of_roots_l104_104146

theorem sum_of_reciprocals_of_roots (s₁ s₂ : ℝ) (h₀ : s₁ + s₂ = 15) (h₁ : s₁ * s₂ = 36) :
  (1 / s₁) + (1 / s₂) = 5 / 12 :=
by
  sorry

end sum_of_reciprocals_of_roots_l104_104146


namespace jose_peanuts_l104_104059

/-- If Kenya has 133 peanuts and this is 48 more than what Jose has,
    then Jose has 85 peanuts. -/
theorem jose_peanuts (j k : ℕ) (h1 : k = j + 48) (h2 : k = 133) : j = 85 :=
by
  -- Proof goes here
  sorry

end jose_peanuts_l104_104059


namespace abs_eq_two_l104_104648

theorem abs_eq_two (m : ℤ) (h : |m| = 2) : m = 2 ∨ m = -2 :=
sorry

end abs_eq_two_l104_104648


namespace line_equation_through_M_P_Q_l104_104839

-- Given that M is the midpoint between P and Q, we should have:
-- M = (1, -2)
-- P = (2, 0)
-- Q = (0, -4)
-- We need to prove that the line passing through these points has the equation 2x - y - 4 = 0

theorem line_equation_through_M_P_Q :
  ∀ (x y : ℝ), (1 - 2 = (2 * (x - 1)) ∧ 0 - 2 = (2 * (0 - (-2)))) ->
  (x - y - 4 = 0) := 
by
  sorry

end line_equation_through_M_P_Q_l104_104839


namespace work_completion_alternate_days_l104_104029

theorem work_completion_alternate_days (h₁ : ∀ (work : ℝ), ∃ a_days : ℝ, a_days = 12 → (∀ t : ℕ, t / a_days <= work / 12))
                                      (h₂ : ∀ (work : ℝ), ∃ b_days : ℝ, b_days = 36 → (∀ t : ℕ, t / b_days <= work / 36)) :
  ∃ days : ℝ, days = 18 := by
  sorry

end work_completion_alternate_days_l104_104029


namespace num_non_fiction_books_l104_104963

-- Definitions based on the problem conditions
def num_fiction_configurations : ℕ := 24
def total_configurations : ℕ := 36

-- Non-computable definition for factorial
noncomputable def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

-- Theorem to prove the number of new non-fiction books
theorem num_non_fiction_books (n : ℕ) :
  num_fiction_configurations * factorial n = total_configurations → n = 2 :=
by
  sorry

end num_non_fiction_books_l104_104963


namespace pentagonal_pyramid_faces_l104_104156

-- Definition of a pentagonal pyramid
structure PentagonalPyramid where
  base_sides : Nat := 5
  triangular_faces : Nat := 5

-- The goal is to prove that the total number of faces is 6
theorem pentagonal_pyramid_faces (P : PentagonalPyramid) : P.base_sides + 1 = 6 :=
  sorry

end pentagonal_pyramid_faces_l104_104156


namespace intersection_of_sets_l104_104260

def set_M : Set ℝ := { x | x >= 2 }
def set_N : Set ℝ := { x | -1 <= x ∧ x <= 3 }
def set_intersection : Set ℝ := { x | 2 <= x ∧ x <= 3 }

theorem intersection_of_sets : (set_M ∩ set_N) = set_intersection := by
  sorry

end intersection_of_sets_l104_104260


namespace coin_flip_probability_l104_104221

theorem coin_flip_probability (p : ℝ) 
  (h : p^2 + (1 - p)^2 = 4 * p * (1 - p)) : 
  p = (3 + Real.sqrt 3) / 6 :=
sorry

end coin_flip_probability_l104_104221


namespace positive_real_solution_eq_l104_104489

theorem positive_real_solution_eq :
  ∃ x : ℝ, 0 < x ∧ ( (1/4) * (5 * x^2 - 4) = (x^2 - 40 * x - 5) * (x^2 + 20 * x + 2) ) ∧ x = 20 + 10 * Real.sqrt 41 :=
by
  sorry

end positive_real_solution_eq_l104_104489


namespace lydia_eats_apple_age_l104_104619

-- Define the conditions
def years_to_bear_fruit : ℕ := 7
def age_when_planted : ℕ := 4
def current_age : ℕ := 9

-- Define the theorem statement
theorem lydia_eats_apple_age : 
  (age_when_planted + years_to_bear_fruit = 11) :=
by
  sorry

end lydia_eats_apple_age_l104_104619


namespace fraction_zero_implies_x_eq_one_l104_104615

theorem fraction_zero_implies_x_eq_one (x : ℝ) (h : (x - 1) / (x + 1) = 0) : x = 1 :=
sorry

end fraction_zero_implies_x_eq_one_l104_104615


namespace find_g5_l104_104812

def g : ℤ → ℤ := sorry

axiom g1 : g 1 > 1
axiom g2 : ∀ x y : ℤ, g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y
axiom g3 : ∀ x : ℤ, 3 * g x = g (x + 1) + 2 * x - 1

theorem find_g5 : g 5 = 248 :=
sorry

end find_g5_l104_104812


namespace problem1_problem2_l104_104527

-- Define the required conditions
variables {a b : ℤ}
-- Conditions
axiom h1 : a ≥ 1
axiom h2 : b ≥ 1

-- Proof statement for question 1
theorem problem1 : ¬ (a ∣ b^2 ↔ a ∣ b) := by
  sorry

-- Proof statement for question 2
theorem problem2 : (a^2 ∣ b^2 ↔ a ∣ b) := by
  sorry

end problem1_problem2_l104_104527


namespace personal_income_tax_correct_l104_104032

-- Defining the conditions
def monthly_income : ℕ := 30000
def vacation_bonus : ℕ := 20000
def car_sale_income : ℕ := 250000
def land_purchase_cost : ℕ := 300000

def standard_deduction_car_sale : ℕ := 250000
def property_deduction_land_purchase : ℕ := 300000

-- Define total income
def total_income : ℕ := (monthly_income * 12) + vacation_bonus + car_sale_income

-- Define total deductions
def total_deductions : ℕ := standard_deduction_car_sale + property_deduction_land_purchase

-- Define taxable income (total income - total deductions)
def taxable_income : ℕ := total_income - total_deductions

-- Define tax rate
def tax_rate : ℚ := 0.13

-- Define the correct answer for the tax payable
def tax_payable : ℚ := taxable_income * tax_rate

-- Prove the tax payable is 10400 rubles
theorem personal_income_tax_correct : tax_payable = 10400 := by
  sorry

end personal_income_tax_correct_l104_104032


namespace circumcircle_radius_of_right_triangle_l104_104903

theorem circumcircle_radius_of_right_triangle (a b c : ℝ) (h1: a = 8) (h2: b = 6) (h3: c = 10) (h4: a^2 + b^2 = c^2) : (c / 2) = 5 := 
by
  sorry

end circumcircle_radius_of_right_triangle_l104_104903


namespace total_pull_ups_per_week_l104_104054

-- Definitions from the conditions
def pull_ups_per_time := 2
def visits_per_day := 5
def days_per_week := 7

-- The Math proof problem statement
theorem total_pull_ups_per_week :
  pull_ups_per_time * visits_per_day * days_per_week = 70 := by
  sorry

end total_pull_ups_per_week_l104_104054


namespace solve_modified_system_l104_104256

theorem solve_modified_system (a1 b1 c1 a2 b2 c2 : ℝ) (h1 : 4 * a1 + 6 * b1 = c1) 
  (h2 : 4 * a2 + 6 * b2 = c2) :
  (4 * a1 * 5 + 3 * b1 * 10 = 5 * c1) ∧ (4 * a2 * 5 + 3 * b2 * 10 = 5 * c2) :=
by
  sorry

end solve_modified_system_l104_104256


namespace abs_ab_eq_2_sqrt_65_l104_104077

theorem abs_ab_eq_2_sqrt_65
  (a b : ℝ)
  (h1 : b^2 - a^2 = 16)
  (h2 : a^2 + b^2 = 36) :
  |a * b| = 2 * Real.sqrt 65 := 
sorry

end abs_ab_eq_2_sqrt_65_l104_104077


namespace cos_double_angle_l104_104071

theorem cos_double_angle (α : ℝ) (h : Real.cos α = -3/5) : Real.cos (2 * α) = -7/25 :=
by
  sorry

end cos_double_angle_l104_104071


namespace equivalent_statements_l104_104814
  
variables {A B : Prop}

theorem equivalent_statements :
  ((A ∧ B) → ¬ (A ∨ B)) ↔ ((A ∨ B) → ¬ (A ∧ B)) :=
sorry

end equivalent_statements_l104_104814


namespace tan_alpha_eq_neg_four_thirds_l104_104707

theorem tan_alpha_eq_neg_four_thirds
  (α : ℝ) (hα1 : 0 < α ∧ α < π) 
  (hα2 : Real.sin α + Real.cos α = 1 / 5) : 
  Real.tan α = - 4 / 3 := 
  sorry

end tan_alpha_eq_neg_four_thirds_l104_104707


namespace wand_cost_l104_104004

theorem wand_cost (c : ℕ) (h1 : 3 * c = 3 * c) (h2 : 2 * (c + 5) = 130) : c = 60 :=
by
  sorry

end wand_cost_l104_104004


namespace age_proof_l104_104595

theorem age_proof (A B C D k m : ℕ)
  (h1 : A + B + C + D = 76)
  (h2 : A - 3 = k)
  (h3 : B - 3 = 2*k)
  (h4 : C - 3 = 3*k)
  (h5 : A - 5 = 3*m)
  (h6 : D - 5 = 4*m)
  (h7 : B - 5 = 5*m) :
  A = 11 := 
sorry

end age_proof_l104_104595


namespace radius_of_spheres_in_cone_l104_104536

def base_radius := 8
def cone_height := 15
def num_spheres := 3
def spheres_are_tangent := true

theorem radius_of_spheres_in_cone :
  ∃ (r : ℝ), r = (280 - 100 * Real.sqrt 3) / 121 :=
sorry

end radius_of_spheres_in_cone_l104_104536


namespace solve_inequality_l104_104292

noncomputable def solution_set (a : ℝ) : Set ℝ :=
  if h : a > -1 then { x : ℝ | -1 < x ∧ x < a }
  else if h : a < -1 then { x : ℝ | a < x ∧ x < -1 }
  else ∅

theorem solve_inequality (x a : ℝ) :
  (x^2 + (1 - a)*x - a < 0) ↔ (
    (a > -1 → x ∈ { x : ℝ | -1 < x ∧ x < a }) ∧
    (a < -1 → x ∈ { x : ℝ | a < x ∧ x < -1 }) ∧
    (a = -1 → False)
  ) :=
sorry

end solve_inequality_l104_104292


namespace convert_to_base7_l104_104430

theorem convert_to_base7 : 3589 = 1 * 7^4 + 3 * 7^3 + 3 * 7^2 + 1 * 7^1 + 5 * 7^0 :=
by
  sorry

end convert_to_base7_l104_104430


namespace curve_intersection_one_point_l104_104159

theorem curve_intersection_one_point (a : ℝ) :
  (∀ x y : ℝ, (x^2 + y^2 = a^2 ↔ y = x^2 + a) → (x, y) = (0, a)) ↔ (a ≥ -1/2) := 
sorry

end curve_intersection_one_point_l104_104159


namespace surface_area_increase_l104_104714

theorem surface_area_increase (r h : ℝ) (cs : Bool) : -- cs is a condition switch, True for circular cut, False for rectangular cut
  0 < r ∧ 0 < h →
  let inc_area := if cs then 2 * π * r^2 else 2 * h * r 
  inc_area > 0 :=
by 
  sorry

end surface_area_increase_l104_104714


namespace isosceles_trapezoid_ratio_l104_104951

theorem isosceles_trapezoid_ratio (a b d_E d_G : ℝ) (h1 : a > b)
  (h2 : (1/2) * b * d_G = 3) (h3 : (1/2) * a * d_E = 7)
  (h4 : (1/2) * (a + b) * (d_E + d_G) = 24) :
  (a / b) = 7 / 3 :=
sorry

end isosceles_trapezoid_ratio_l104_104951


namespace neg_neg_eq_l104_104606

theorem neg_neg_eq (n : ℤ) : -(-n) = n :=
  sorry

example : -(-2023) = 2023 :=
by apply neg_neg_eq

end neg_neg_eq_l104_104606


namespace printing_shop_paper_boxes_l104_104160

variable (x y : ℕ) -- Assuming x and y are natural numbers since the number of boxes can't be negative.

theorem printing_shop_paper_boxes (h1 : 80 * x + 180 * y = 2660)
                                  (h2 : x = 5 * y - 3) :
    x = 22 ∧ y = 5 := sorry

end printing_shop_paper_boxes_l104_104160


namespace net_cut_square_l104_104240

-- Define the dimensions of the parallelepiped
structure Parallelepiped :=
  (length width height : ℕ)
  (length_eq : length = 2)
  (width_eq : width = 1)
  (height_eq : height = 1)

-- Define the net of the parallelepiped
structure NetConfig :=
  (total_squares : ℕ)
  (cut_squares : ℕ)
  (remaining_squares : ℕ)
  (cut_positions : Fin 5) -- Five possible cut positions

-- The remaining net has 9 squares after cutting one square
theorem net_cut_square (p : Parallelepiped) : 
  ∃ net : NetConfig, net.total_squares = 10 ∧ net.cut_squares = 1 ∧ net.remaining_squares = 9 ∧ net.cut_positions = 5 := 
sorry

end net_cut_square_l104_104240


namespace stickers_left_after_giving_away_l104_104989

/-- Willie starts with 36 stickers and gives 7 to Emily. 
    We want to prove that Willie ends up with 29 stickers. -/
theorem stickers_left_after_giving_away (init_stickers : ℕ) (given_away : ℕ) (end_stickers : ℕ) : 
  init_stickers = 36 ∧ given_away = 7 → end_stickers = init_stickers - given_away → end_stickers = 29 :=
by
  intro h
  sorry

end stickers_left_after_giving_away_l104_104989


namespace find_a_l104_104222

-- Definitions of the conditions
variables {a b c : ℤ} 

-- Theorem statement
theorem find_a (h1: a + b = c) (h2: b + c = 7) (h3: c = 4) : a = 1 :=
by
  -- Using sorry to skip the proof
  sorry

end find_a_l104_104222


namespace frank_spends_more_l104_104369

def cost_computer_table : ℕ := 140
def cost_computer_chair : ℕ := 100
def cost_joystick : ℕ := 20
def frank_share_joystick : ℕ := cost_joystick / 4
def eman_share_joystick : ℕ := cost_joystick * 3 / 4

def total_spent_frank : ℕ := cost_computer_table + frank_share_joystick
def total_spent_eman : ℕ := cost_computer_chair + eman_share_joystick

theorem frank_spends_more : total_spent_frank - total_spent_eman = 30 :=
by
  sorry

end frank_spends_more_l104_104369


namespace fraction_of_fraction_of_fraction_l104_104227

theorem fraction_of_fraction_of_fraction (a b c d : ℝ) (h₁ : a = 1/5) (h₂ : b = 1/3) (h₃ : c = 1/6) (h₄ : d = 90) :
  (a * b * c * d) = 1 :=
by
  rw [h₁, h₂, h₃, h₄]
  simp
  sorry -- To indicate that the proof is missing

end fraction_of_fraction_of_fraction_l104_104227


namespace traveler_drank_32_ounces_l104_104941

-- Definition of the given condition
def total_gallons : ℕ := 2
def ounces_per_gallon : ℕ := 128
def total_ounces := total_gallons * ounces_per_gallon
def camel_multiple : ℕ := 7
def traveler_ounces (T : ℕ) := T
def camel_ounces (T : ℕ) := camel_multiple * T
def total_drunk (T : ℕ) := traveler_ounces T + camel_ounces T

-- Theorem to prove
theorem traveler_drank_32_ounces :
  ∃ T : ℕ, total_drunk T = total_ounces ∧ T = 32 :=
by 
  sorry

end traveler_drank_32_ounces_l104_104941


namespace octahedron_sum_l104_104317

-- Define the properties of an octahedron
def octahedron_edges := 12
def octahedron_vertices := 6
def octahedron_faces := 8

theorem octahedron_sum : octahedron_edges + octahedron_vertices + octahedron_faces = 26 := by
  -- Here we state that the sum of edges, vertices, and faces equals 26
  sorry

end octahedron_sum_l104_104317


namespace quotient_of_larger_divided_by_smaller_l104_104506

theorem quotient_of_larger_divided_by_smaller
  (x y : ℕ)
  (h1 : x * y = 9375)
  (h2 : x + y = 400)
  (h3 : x > y) :
  x / y = 15 :=
sorry

end quotient_of_larger_divided_by_smaller_l104_104506


namespace simplify_fraction_l104_104149

theorem simplify_fraction :
  (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 :=
by
  sorry

end simplify_fraction_l104_104149


namespace Lakeview_High_School_Basketball_Team_l104_104532

theorem Lakeview_High_School_Basketball_Team :
  ∀ (total_players taking_physics taking_both statistics: ℕ),
  total_players = 25 →
  taking_physics = 10 →
  taking_both = 5 →
  statistics = 20 :=
sorry

end Lakeview_High_School_Basketball_Team_l104_104532


namespace assign_teachers_to_classes_l104_104775

-- Define the given conditions as variables and constants
theorem assign_teachers_to_classes :
  (∃ ways : ℕ, ways = 36) :=
by
  sorry

end assign_teachers_to_classes_l104_104775


namespace range_of_MF_plus_MN_l104_104779

open Real

noncomputable def point_on_parabola (x y : ℝ) : Prop := y^2 = 4 * x

theorem range_of_MF_plus_MN (M : ℝ × ℝ) (N : ℝ × ℝ) (F : ℝ × ℝ) (hM : point_on_parabola M.1 M.2) (hN : N = (2, 2)) (hF : F = (1, 0)) :
  ∃ y : ℝ, y ≥ 3 ∧ ∀ MF MN : ℝ, MF = abs (M.1 - F.1) + abs (M.2 - F.2) ∧ MN = abs (M.1 - N.1) + abs (M.2 - N.2) → MF + MN = y :=
sorry

end range_of_MF_plus_MN_l104_104779


namespace conditions_for_right_triangle_l104_104533

universe u

variables {A B C : Type u}
variables [OrderedAddCommGroup A] [OrderedAddCommGroup B] [OrderedAddCommGroup C]

noncomputable def is_right_triangle (A B C : ℝ) : Prop :=
  A + B + C = 180 ∧ (A = 90 ∨ B = 90 ∨ C = 90)

theorem conditions_for_right_triangle :
  (∀ (A B C : ℝ), A + B = C → is_right_triangle A B C) ∧
  (∀ (A B C : ℝ), ( A / C = 1 / 6 ) → is_right_triangle A B C) ∧
  (∀ (A B C : ℝ), A = 90 - B → is_right_triangle A B C) ∧
  (∀ (A B C : ℝ), (A = B → B = C / 2) → is_right_triangle A B C) ∧
  ∀ (A B C : ℝ), ¬ ((A = 2 * B) ∧ B = 3 * C) 
:=
sorry

end conditions_for_right_triangle_l104_104533


namespace complex_expression_simplified_l104_104179

theorem complex_expression_simplified :
  let z1 := (1 + 3 * Complex.I) / (1 - 3 * Complex.I)
  let z2 := (1 - 3 * Complex.I) / (1 + 3 * Complex.I)
  let z3 := 1 / (8 * Complex.I^3)
  z1 + z2 + z3 = -1.6 + 0.125 * Complex.I := 
by
  sorry

end complex_expression_simplified_l104_104179


namespace shirts_count_l104_104466

theorem shirts_count (S : ℕ) (hours_per_shirt hours_per_pant cost_per_hour total_pants total_cost : ℝ) :
  hours_per_shirt = 1.5 →
  hours_per_pant = 3 →
  cost_per_hour = 30 →
  total_pants = 12 →
  total_cost = 1530 →
  45 * S + 1080 = total_cost →
  S = 10 :=
by
  intros hps hpp cph tp tc cost_eq
  sorry

end shirts_count_l104_104466


namespace part1_part2_l104_104224
open Real

-- Part 1
theorem part1 (x : ℝ) (h : 0 ≤ x ∧ x ≤ 1) :
  0 < (sqrt (1 + x) + sqrt (1 - x) + 2) * (sqrt (1 - x^2) + 1) ∧
  (sqrt (1 + x) + sqrt (1 - x) + 2) * (sqrt (1 - x^2) + 1) ≤ 8 := 
sorry

-- Part 2
theorem part2 (x : ℝ) (h : 0 ≤ x ∧ x ≤ 1) :
  ∃ β > 0, β = 4 ∧ sqrt (1 + x) + sqrt (1 - x) ≤ 2 - x^2 / β :=
sorry

end part1_part2_l104_104224


namespace triangle_altitude_l104_104979

theorem triangle_altitude (b : ℕ) (h : ℕ) (area : ℕ) (h_area : area = 800) (h_base : b = 40) (h_formula : area = (1 / 2) * b * h) : h = 40 :=
by
  sorry

end triangle_altitude_l104_104979


namespace problem1_problem2_l104_104831

-- Problem I
def A (a : ℝ) : Set ℝ := {x | x^2 - a*x + a^2 - 12 = 0}
def B : Set ℝ := {-2, 4}

theorem problem1 (a : ℝ) (h : A a = B) : a = 2 :=
sorry

-- Problem II
def C (m : ℝ) : Set ℝ := {x | m * x + 1 = 0}
def B' : Set ℝ := {-2, 4}

theorem problem2 (m : ℝ) (h : B' ∪ C m = B') : 
  m = -1/2 ∨ m = -1/4 ∨ m = 0 :=
sorry

end problem1_problem2_l104_104831


namespace students_play_neither_l104_104325

-- Defining the problem parameters
def total_students : ℕ := 36
def football_players : ℕ := 26
def tennis_players : ℕ := 20
def both_players : ℕ := 17

-- Statement to be proved
theorem students_play_neither : (total_students - (football_players + tennis_players - both_players)) = 7 :=
by show total_students - (football_players + tennis_players - both_players) = 7; sorry

end students_play_neither_l104_104325


namespace relationship_abc_l104_104730

noncomputable def a : ℝ := 5 ^ (Real.log 3.4 / Real.log 3)
noncomputable def b : ℝ := 5 ^ (Real.log 3.6 / Real.log 3)
noncomputable def c : ℝ := (1 / 5) ^ (Real.log 0.5 / Real.log 3)

theorem relationship_abc : b > a ∧ a > c :=
by
  -- Assumptions derived from logarithmic properties.
  have h1 : Real.log 2 < Real.log 3.4 := sorry
  have h2 : Real.log 3.4 < Real.log 3.6 := sorry
  have h3 : Real.log 0.5 < 0 := sorry
  have h4 : Real.log 2 / Real.log 3 = Real.log 2 := sorry
  have h5 : Real.log 0.5 / Real.log 3 = -Real.log 2 := sorry

  -- Monotonicity of exponential function.
  apply And.intro
  { exact sorry }
  { exact sorry }

end relationship_abc_l104_104730


namespace henrietta_has_three_bedrooms_l104_104375

theorem henrietta_has_three_bedrooms
  (living_room_walls_sqft : ℕ)
  (bedroom_walls_sqft : ℕ)
  (num_bedrooms : ℕ)
  (gallon_coverage_sqft : ℕ)
  (h1 : living_room_walls_sqft = 600)
  (h2 : bedroom_walls_sqft = 400)
  (h3 : gallon_coverage_sqft = 600)
  (h4 : num_bedrooms = 3) : 
  num_bedrooms = 3 :=
by
  exact h4

end henrietta_has_three_bedrooms_l104_104375


namespace min_days_equal_shifts_l104_104088

theorem min_days_equal_shifts (k n : ℕ) (h : 9 * k + 10 * n = 66) : k + n = 7 :=
sorry

end min_days_equal_shifts_l104_104088


namespace smallest_dividend_l104_104588

   theorem smallest_dividend (b a : ℤ) (q : ℤ := 12) (r : ℤ := 3) (h : a = b * q + r) (h' : r < b) : a = 51 :=
   by
     sorry
   
end smallest_dividend_l104_104588


namespace sequence_and_sum_l104_104599

variable (a : ℕ → ℝ) (S : ℕ → ℝ)

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n = a 0 + n * (a 1 - a 0)

def sum_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = (n / 2) * (2 * a 0 + (n - 1) * (a 1 - a 0))

theorem sequence_and_sum
  (h_arith : arithmetic_sequence a)
  (h_sum : sum_first_n_terms a S)
  (cond : a 2 + a 8 = 15 - a 5) :
  S 9 = 45 :=
sorry

end sequence_and_sum_l104_104599


namespace triangle_largest_angle_l104_104131

theorem triangle_largest_angle {k : ℝ} (h1 : k > 0)
  (h2 : k + 2 * k + 3 * k = 180) : 3 * k = 90 := 
sorry

end triangle_largest_angle_l104_104131


namespace mean_score_of_seniors_l104_104763

theorem mean_score_of_seniors 
  (s n : ℕ)
  (ms mn : ℝ)
  (h1 : s + n = 120)
  (h2 : n = 2 * s)
  (h3 : ms = 1.5 * mn)
  (h4 : (s : ℝ) * ms + (n : ℝ) * mn = 13200)
  : ms = 141.43 :=
by
  sorry

end mean_score_of_seniors_l104_104763


namespace answered_both_correctly_l104_104510

variable (A B : Prop)
variable (P_A P_B P_not_A_and_not_B P_A_and_B : ℝ)

axiom P_A_eq : P_A = 0.75
axiom P_B_eq : P_B = 0.35
axiom P_not_A_and_not_B_eq : P_not_A_and_not_B = 0.20

theorem answered_both_correctly (h1 : P_A = 0.75) (h2 : P_B = 0.35) (h3 : P_not_A_and_not_B = 0.20) : 
  P_A_and_B = 0.30 :=
by
  sorry

end answered_both_correctly_l104_104510


namespace problem_eq_995_l104_104352

theorem problem_eq_995 :
  (12^4 + 400) * (24^4 + 400) * (36^4 + 400) * (48^4 + 400) * (60^4 + 400) /
  ((6^4 + 400) * (18^4 + 400) * (30^4 + 400) * (42^4 + 400) * (54^4 + 400))
  = 995 := sorry

end problem_eq_995_l104_104352


namespace ratio_t_q_l104_104470

theorem ratio_t_q (q r s t : ℚ) (h1 : q / r = 8) (h2 : s / r = 4) (h3 : s / t = 1 / 3) : 
  t / q = 3 / 2 :=
by
  sorry

end ratio_t_q_l104_104470


namespace least_number_divisible_by_11_and_remainder_2_l104_104598

theorem least_number_divisible_by_11_and_remainder_2 :
  ∃ n, (∀ k : ℕ, 3 ≤ k ∧ k ≤ 7 → n % k = 2) ∧ n % 11 = 0 ∧ n = 1262 :=
by
  sorry

end least_number_divisible_by_11_and_remainder_2_l104_104598


namespace calvin_score_l104_104611

theorem calvin_score (C : ℚ) (h_paislee_score : (3/4) * C = 125) : C = 167 := 
  sorry

end calvin_score_l104_104611


namespace cupcakes_per_child_l104_104094

theorem cupcakes_per_child (total_cupcakes children : ℕ) (h1 : total_cupcakes = 96) (h2 : children = 8) : total_cupcakes / children = 12 :=
by
  sorry

end cupcakes_per_child_l104_104094


namespace locus_of_midpoint_of_chord_l104_104816

theorem locus_of_midpoint_of_chord
  (x y : ℝ)
  (hx : (x - 1)^2 + y^2 ≠ 0)
  : (x - 1) * (x - 1) + y * y = 1 :=
by
  sorry

end locus_of_midpoint_of_chord_l104_104816


namespace shaded_area_fraction_l104_104230

theorem shaded_area_fraction (ABCD_area : ℝ) (shaded_square1_area : ℝ) (shaded_rectangle_area : ℝ) (shaded_square2_area : ℝ) (total_shaded_area : ℝ)
  (h_ABCD : ABCD_area = 36) 
  (h_shaded_square1 : shaded_square1_area = 4)
  (h_shaded_rectangle : shaded_rectangle_area = 12)
  (h_shaded_square2 : shaded_square2_area = 36)
  (h_total_shaded : total_shaded_area = 16) :
  (total_shaded_area / ABCD_area) = 4 / 9 :=
by 
  simp [h_ABCD, h_total_shaded]
  sorry

end shaded_area_fraction_l104_104230


namespace expression_undefined_at_12_l104_104749

theorem expression_undefined_at_12 :
  ¬ ∃ x : ℝ, x = 12 ∧ (x^2 - 24 * x + 144 = 0) →
  (∃ y : ℝ, y = (3 * x^3 + 5) / (x^2 - 24 * x + 144)) :=
by
  sorry

end expression_undefined_at_12_l104_104749


namespace fraction_to_decimal_l104_104962

theorem fraction_to_decimal : (3 : ℚ) / 80 = 0.0375 :=
by
  sorry

end fraction_to_decimal_l104_104962


namespace Telegraph_Road_length_is_162_l104_104128

-- Definitions based on the conditions
def meters_to_kilometers (meters : ℕ) : ℕ := meters / 1000
def Pardee_Road_length_meters : ℕ := 12000
def Telegraph_Road_extra_length_kilometers : ℕ := 150

-- The length of Pardee Road in kilometers
def Pardee_Road_length_kilometers : ℕ := meters_to_kilometers Pardee_Road_length_meters

-- Lean statement to prove the length of Telegraph Road in kilometers
theorem Telegraph_Road_length_is_162 :
  Pardee_Road_length_kilometers + Telegraph_Road_extra_length_kilometers = 162 :=
sorry

end Telegraph_Road_length_is_162_l104_104128


namespace range_of_a_l104_104408

-- Definitions for propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, ¬(x^2 + (a-1)*x + 1 ≤ 0)

def q (a : ℝ) : Prop := ∀ x₁ x₂ : ℝ, x₁ < x₂ → (a - 1)^x₁ < (a - 1)^x₂

-- The final theorem to prove
theorem range_of_a (a : ℝ) : ¬(p a ∧ q a) ∧ (p a ∨ q a) → (-1 < a ∧ a ≤ 2) ∨ (a ≥ 3) :=
by
  sorry

end range_of_a_l104_104408


namespace min_value_a_plus_2b_l104_104193

theorem min_value_a_plus_2b (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a * b = 20) : a + 2 * b = 4 * Real.sqrt 10 :=
by
  sorry

end min_value_a_plus_2b_l104_104193


namespace problem_statement_l104_104949

theorem problem_statement (x y : ℕ) (hx : x = 3) (hy : y = 4) :
  (x^3 + 3 * y^2) / 7 = 75 / 7 :=
by 
  -- proof goes here
  sorry

end problem_statement_l104_104949


namespace coin_and_die_probability_l104_104914

-- Probability of a coin showing heads
def P_heads : ℚ := 2 / 3

-- Probability of a die showing 5
def P_die_5 : ℚ := 1 / 6

-- Probability of both events happening together
def P_heads_and_die_5 : ℚ := P_heads * P_die_5

-- Theorem statement: Proving the calculated probability equals the expected value
theorem coin_and_die_probability : P_heads_and_die_5 = 1 / 9 := by
  -- The detailed proof is omitted here.
  sorry

end coin_and_die_probability_l104_104914


namespace common_ratio_geometric_sequence_l104_104693

theorem common_ratio_geometric_sequence (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) ∧ a 1 = 32 ∧ a 6 = -1 → q = -1/2 :=
by
  sorry

end common_ratio_geometric_sequence_l104_104693


namespace inequality_of_trig_function_l104_104139

theorem inequality_of_trig_function 
  (a b A B : ℝ) 
  (h : ∀ x : ℝ, 1 - a * Real.cos x - b * Real.sin x - A * Real.cos (2 * x) - B * Real.sin (2 * x) ≥ 0) :
  a^2 + b^2 ≤ 2 ∧ A^2 + B^2 ≤ 1 :=
sorry

end inequality_of_trig_function_l104_104139


namespace one_inch_represents_feet_l104_104374

def height_statue : ℕ := 80 -- Height of the statue in feet

def height_model : ℕ := 5 -- Height of the model in inches

theorem one_inch_represents_feet : (height_statue / height_model) = 16 := 
by
  sorry

end one_inch_represents_feet_l104_104374


namespace fraction_sum_l104_104252

theorem fraction_sum :
  (1 / 4 : ℚ) + (2 / 9) + (3 / 6) = 35 / 36 := 
sorry

end fraction_sum_l104_104252


namespace product_of_roots_l104_104632

noncomputable def is_root (a b c x : ℝ) : Prop :=
  a * x^2 + b * x + c = 0

theorem product_of_roots :
  ∀ (x1 x2 : ℝ), is_root 1 (-4) 3 x1 ∧ is_root 1 (-4) 3 x2 ∧ x1 ≠ x2 → x1 * x2 = 3 :=
by
  intros x1 x2 h
  sorry

end product_of_roots_l104_104632


namespace limo_cost_is_correct_l104_104920

def prom_tickets_cost : ℕ := 2 * 100
def dinner_cost : ℕ := 120
def dinner_tip : ℕ := (30 * dinner_cost) / 100
def total_cost_before_limo : ℕ := prom_tickets_cost + dinner_cost + dinner_tip
def total_cost : ℕ := 836
def limo_hours : ℕ := 6
def limo_total_cost : ℕ := total_cost - total_cost_before_limo
def limo_cost_per_hour : ℕ := limo_total_cost / limo_hours

theorem limo_cost_is_correct : limo_cost_per_hour = 80 := 
by
  sorry

end limo_cost_is_correct_l104_104920


namespace correct_product_l104_104079

theorem correct_product (a b : ℕ)
  (h1 : 10 ≤ a ∧ a < 100)  -- a is a two-digit number
  (h2 : 0 < b)  -- b is a positive integer
  (h3 : (a % 10) * 10 + (a / 10) * b = 161)  -- Reversing the digits of a and multiplying by b yields 161
  : a * b = 224 := 
sorry

end correct_product_l104_104079


namespace geometric_sequence_product_l104_104840

theorem geometric_sequence_product (a1 a5 : ℚ) (a b c : ℚ) (q : ℚ) 
  (h1 : a1 = 8 / 3) 
  (h5 : a5 = 27 / 2)
  (h_common_ratio_pos : q = 3 / 2)
  (h_a : a = a1 * q)
  (h_b : b = a * q)
  (h_c : c = b * q)
  (h5_eq : a5 = a1 * q^4)
  (h_common_ratio_neg : q = -3 / 2 ∨ q = 3 / 2) :
  a * b * c = 216 := by
    sorry

end geometric_sequence_product_l104_104840


namespace values_only_solution_l104_104043

variables (m n : ℝ) (x a b c : ℝ)

noncomputable def equation := (x + m)^3 - (x + n)^3 = (m + n)^3

theorem values_only_solution (hm : m ≠ 0) (hn : n ≠ 0) (hne : m ≠ n)
  (hx : x = a * m + b * n + c) : a = 0 ∧ b = 0 ∧ c = 0 :=
by
  sorry

end values_only_solution_l104_104043


namespace ab_diff_2023_l104_104248

theorem ab_diff_2023 (a b : ℝ) 
  (h : a^2 + b^2 - 4 * a - 6 * b + 13 = 0) : (a - b) ^ 2023 = -1 :=
sorry

end ab_diff_2023_l104_104248


namespace toys_per_rabbit_l104_104144

-- Define the conditions
def rabbits : ℕ := 34
def toys_mon : ℕ := 8
def toys_tue : ℕ := 3 * toys_mon
def toys_wed : ℕ := 2 * toys_tue
def toys_thu : ℕ := toys_mon
def toys_fri : ℕ := 5 * toys_mon
def toys_sat : ℕ := toys_wed / 2

-- Define the total number of toys
def total_toys : ℕ := toys_mon + toys_tue + toys_wed + toys_thu + toys_fri + toys_sat

-- Define the proof statement
theorem toys_per_rabbit : total_toys / rabbits = 4 :=
by
  -- Proof will go here
  sorry

end toys_per_rabbit_l104_104144


namespace geom_seq_fraction_l104_104468

theorem geom_seq_fraction (a_1 a_2 a_3 a_4 a_5 q : ℝ)
  (h1 : q > 0)
  (h2 : a_2 = q * a_1)
  (h3 : a_3 = q^2 * a_1)
  (h4 : a_4 = q^3 * a_1)
  (h5 : a_5 = q^4 * a_1)
  (h_arith : a_2 - (1/2) * a_3 = (1/2) * a_3 - a_1) :
  (a_3 + a_4) / (a_4 + a_5) = (Real.sqrt 5 - 1) / 2 :=
by
  sorry

end geom_seq_fraction_l104_104468


namespace base8_246_is_166_in_base10_l104_104219

def convert_base8_to_base10 (n : ℕ) : ℕ :=
  let d0 := n % 10;
  let d1 := (n / 10) % 10;
  let d2 := (n / 100) % 10;
  d2 * 8^2 + d1 * 8^1 + d0 * 8^0

theorem base8_246_is_166_in_base10 : convert_base8_to_base10 246 = 166 :=
  sorry

end base8_246_is_166_in_base10_l104_104219


namespace find_cost_price_l104_104860

-- Given conditions
variables (CP SP1 SP2 : ℝ)
def condition1 : Prop := SP1 = 0.90 * CP
def condition2 : Prop := SP2 = 1.10 * CP
def condition3 : Prop := SP2 - SP1 = 500

-- Prove that CP is 2500 
theorem find_cost_price 
  (CP SP1 SP2 : ℝ)
  (h1 : condition1 CP SP1)
  (h2 : condition2 CP SP2)
  (h3 : condition3 SP1 SP2) : 
  CP = 2500 :=
sorry -- proof not required

end find_cost_price_l104_104860


namespace system_infinite_solutions_l104_104773

theorem system_infinite_solutions :
  ∃ (x y : ℚ), (3 * x - 4 * y = 5) ∧ (9 * x - 12 * y = 15) ↔ (3 * x - 4 * y = 5) :=
by
  sorry

end system_infinite_solutions_l104_104773


namespace probability_more_wins_than_losses_l104_104692

theorem probability_more_wins_than_losses
  (n_matches : ℕ)
  (win_prob lose_prob tie_prob : ℚ)
  (h_sum_probs : win_prob + lose_prob + tie_prob = 1)
  (h_win_prob : win_prob = 1/3)
  (h_lose_prob : lose_prob = 1/3)
  (h_tie_prob : tie_prob = 1/3)
  (h_n_matches : n_matches = 8) :
  ∃ (m n : ℕ), Nat.gcd m n = 1 ∧ m / n = 5483 / 13122 ∧ (m + n) = 18605 :=
by
  sorry

end probability_more_wins_than_losses_l104_104692


namespace cost_of_renting_per_month_l104_104591

namespace RentCarProblem

def cost_new_car_per_month : ℕ := 30
def months_per_year : ℕ := 12
def yearly_difference : ℕ := 120

theorem cost_of_renting_per_month (R : ℕ) :
  (cost_new_car_per_month * months_per_year + yearly_difference) / months_per_year = R → 
  R = 40 :=
by
  sorry

end RentCarProblem

end cost_of_renting_per_month_l104_104591


namespace alien_saturday_sequence_l104_104220

def a_1 : String := "A"
def a_2 : String := "AY"
def a_3 : String := "AYYA"
def a_4 : String := "AYYAYAAY"

noncomputable def a_5 : String := a_4 ++ "YAAYAYYA"
noncomputable def a_6 : String := a_5 ++ "YAAYAYYAAAYAYAAY"

theorem alien_saturday_sequence : 
  a_6 = "AYYAYAAYYAAYAYYAYAAYAYYAAAYAYAAY" :=
sorry

end alien_saturday_sequence_l104_104220


namespace smallest_m_for_integral_roots_l104_104791

theorem smallest_m_for_integral_roots :
  ∃ (m : ℕ), (∃ (p q : ℤ), p * q = 30 ∧ m = 12 * (p + q)) ∧ m = 132 := by
  sorry

end smallest_m_for_integral_roots_l104_104791


namespace number_of_rectangles_l104_104264

open Real Set

-- Given points A, B, C, D on a line L and a length k
variables {A B C D : ℝ} (L : Set ℝ) (k : ℝ)

-- The points are distinct and ordered on the line
axiom h1 : A ≠ B ∧ B ≠ C ∧ C ≠ D
axiom h2 : A < B ∧ B < C ∧ C < D

-- We need to show there are two rectangles with certain properties
theorem number_of_rectangles : 
  (∃ (rect1 rect2 : Set ℝ), 
    rect1 ≠ rect2 ∧ 
    (∃ (a1 b1 c1 d1 : ℝ), rect1 = {a1, b1, c1, d1} ∧ 
      a1 < b1 ∧ b1 < c1 ∧ c1 < d1 ∧ 
      (d1 - c1 = k ∨ c1 - b1 = k)) ∧ 
    (∃ (a2 b2 c2 d2 : ℝ), rect2 = {a2, b2, c2, d2} ∧ 
      a2 < b2 ∧ b2 < c2 ∧ c2 < d2 ∧ 
      (d2 - c2 = k ∨ c2 - b2 = k))
  ) :=
sorry

end number_of_rectangles_l104_104264


namespace compute_expression_l104_104437

theorem compute_expression :
  6 * (2 / 3)^4 - 1 / 6 = 55 / 54 :=
by
  sorry

end compute_expression_l104_104437


namespace hyperbola_eccentricity_a_l104_104891

theorem hyperbola_eccentricity_a (a : ℝ) (ha : a > 0) :
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / 3 = 1) ∧ (∃ (e : ℝ), e = 2 ∧ e = Real.sqrt (a^2 + 3) / a) → a = 1 :=
by
  sorry

end hyperbola_eccentricity_a_l104_104891


namespace at_least_one_not_less_than_l104_104542

variables {A B C D a b c : ℝ}

theorem at_least_one_not_less_than :
  (a = A * C) →
  (b = A * D + B * C) →
  (c = B * D) →
  (a + b + c = (A + B) * (C + D)) →
  a ≥ (4 * (A + B) * (C + D) / 9) ∨ b ≥ (4 * (A + B) * (C + D) / 9) ∨ c ≥ (4 * (A + B) * (C + D) / 9) :=
by
  intro h1 h2 h3 h4
  sorry

end at_least_one_not_less_than_l104_104542


namespace latest_first_pump_time_l104_104971

theorem latest_first_pump_time 
  (V : ℝ) -- Volume of the pool
  (x y : ℝ) -- Productivity of first and second pumps respectively
  (t : ℝ) -- Time of operation of the first pump until the second pump is turned on
  (h1 : 2*x + 2*y = V/2) -- Condition from 10 AM to 12 PM
  (h2 : 5*x + 5*y = V/2) -- Condition from 12 PM to 5 PM
  (h3 : t*x + 2*x + 2*y = V/2) -- Condition for early morning until 12 PM
  (hx_pos : 0 < x) -- Assume productivity of first pump is positive
  (hy_pos : 0 < y) -- Assume productivity of second pump is positive
  : t ≥ 3 :=
by
  -- The proof goes here...
  sorry

end latest_first_pump_time_l104_104971


namespace max_possible_value_l104_104731

-- Define the number of cities and the structure of roads.
def numCities : ℕ := 110

-- Condition: Each city has either a road or no road to another city
def Road (city1 city2 : ℕ) : Prop := sorry  -- A placeholder definition for the road relationship

-- Condition: Number of roads leading out of each city.
def numRoads (city : ℕ) : ℕ := sorry  -- A placeholder for the actual function counting the number of roads from a city

-- Condition: The driver starts at a city with exactly one road leading out.
def startCity : ℕ := sorry  -- A placeholder for the starting city

-- Main theorem statement to prove the maximum possible value of N is 107
theorem max_possible_value : ∃ N : ℕ, N ≤ 107 ∧ (∀ k : ℕ, 2 ≤ k ∧ k ≤ N → numRoads k = k) :=
by
  sorry  -- Actual proof is not required, hence we use sorry to indicate the proof step is skipped.

end max_possible_value_l104_104731


namespace jellybeans_left_l104_104936

theorem jellybeans_left :
  let initial_jellybeans := 500
  let total_kindergarten := 10
  let total_firstgrade := 10
  let total_secondgrade := 10
  let sick_kindergarten := 2
  let sick_secondgrade := 3
  let jellybeans_sick_kindergarten := 5
  let jellybeans_sick_secondgrade := 10
  let jellybeans_remaining_kindergarten := 3
  let jellybeans_firstgrade := 5
  let jellybeans_secondgrade_per_firstgrade := 5 / 2 * total_firstgrade
  let consumed_by_sick := sick_kindergarten * jellybeans_sick_kindergarten + sick_secondgrade * jellybeans_sick_secondgrade
  let remaining_kindergarten := total_kindergarten - sick_kindergarten
  let consumed_by_remaining := remaining_kindergarten * jellybeans_remaining_kindergarten + total_firstgrade * jellybeans_firstgrade + total_secondgrade * jellybeans_secondgrade_per_firstgrade
  let total_consumed := consumed_by_sick + consumed_by_remaining
  initial_jellybeans - total_consumed = 176 := by 
  sorry

end jellybeans_left_l104_104936


namespace chord_intersection_l104_104000

theorem chord_intersection {AP BP CP DP : ℝ} (hAP : AP = 2) (hBP : BP = 6) (hCP_DP : ∃ k : ℝ, CP = k ∧ DP = 3 * k) :
  DP = 6 :=
by sorry

end chord_intersection_l104_104000


namespace books_received_l104_104495

theorem books_received (initial_books : ℕ) (total_books : ℕ) (h1 : initial_books = 54) (h2 : total_books = 77) : (total_books - initial_books) = 23 :=
by
  sorry

end books_received_l104_104495


namespace james_birthday_stickers_l104_104686

def initial_stickers : ℕ := 39
def final_stickers : ℕ := 61

def birthday_stickers (s_initial s_final : ℕ) : ℕ := s_final - s_initial

theorem james_birthday_stickers :
  birthday_stickers initial_stickers final_stickers = 22 := by
  sorry

end james_birthday_stickers_l104_104686


namespace arithmetic_mean_is_one_l104_104200

theorem arithmetic_mean_is_one (x a : ℝ) (hx : x ≠ 0) (hx2a : x^2 ≠ a) :
  (1 / 2 * ((x^2 + a) / x^2 + (x^2 - a) / x^2) = 1) :=
by
  sorry

end arithmetic_mean_is_one_l104_104200


namespace tan_identity_l104_104455

variable (α β : ℝ)

theorem tan_identity (h₁ : 0 < α ∧ α < π / 2) (h₂ : 0 < β ∧ β < π / 2) (h₃ : Real.sin (2 * α) = 2 * Real.sin (2 * β)) : 
  Real.tan (α + β) = 3 * Real.tan (α - β) := 
by 
  sorry

end tan_identity_l104_104455


namespace range_of_a_l104_104674

theorem range_of_a (a x : ℝ) (h1 : 1 ≤ x ∧ x ≤ 3) (h2 : ∀ x, 1 ≤ x ∧ x ≤ 3 → |x - a| < 2) : 1 < a ∧ a < 3 := by
  sorry

end range_of_a_l104_104674


namespace divisibility_of_n_l104_104695

def n : ℕ := (2^4 - 1) * (3^6 - 1) * (5^10 - 1) * (7^12 - 1)

theorem divisibility_of_n : 
    (5 ∣ n) ∧ (7 ∣ n) ∧ (11 ∣ n) ∧ (13 ∣ n) := 
by 
  sorry

end divisibility_of_n_l104_104695


namespace roger_trays_l104_104782

theorem roger_trays (trays_per_trip trips trays_first_table : ℕ) 
  (h1 : trays_per_trip = 4) 
  (h2 : trips = 3) 
  (h3 : trays_first_table = 10) : 
  trays_per_trip * trips - trays_first_table = 2 :=
by
  -- Step proofs are omitted
  sorry

end roger_trays_l104_104782


namespace bottles_not_in_crates_l104_104335

def total_bottles : ℕ := 250
def num_small_crates : ℕ := 5
def num_medium_crates : ℕ := 5
def num_large_crates : ℕ := 5
def bottles_per_small_crate : ℕ := 8
def bottles_per_medium_crate : ℕ := 12
def bottles_per_large_crate : ℕ := 20

theorem bottles_not_in_crates : 
  num_small_crates * bottles_per_small_crate + 
  num_medium_crates * bottles_per_medium_crate + 
  num_large_crates * bottles_per_large_crate = 200 → 
  total_bottles - 200 = 50 := 
by
  sorry

end bottles_not_in_crates_l104_104335


namespace sum_series_eq_4_l104_104107

theorem sum_series_eq_4 : 
  (∑' n : ℕ, (4 * (n + 1) - 2) / (3 ^ (n + 1))) = 4 := 
by
  sorry

end sum_series_eq_4_l104_104107


namespace inequality_for_positive_integers_l104_104066

theorem inequality_for_positive_integers 
  (a b : ℝ)
  (h₀ : a > 0)
  (h₁ : b > 0)
  (h₂ : 1/a + 1/b = 1)
  (n : ℕ)
  (hn : n > 0) : 
  (a + b) ^ n - a ^ n - b ^ n ≥ 2^(2*n) - 2^(n + 1) :=
sorry

end inequality_for_positive_integers_l104_104066


namespace polygon_sides_l104_104835

-- Define the conditions
def side_length : ℝ := 7
def perimeter : ℝ := 42

-- The statement to prove: number of sides is 6
theorem polygon_sides : (perimeter / side_length) = 6 := by
  sorry

end polygon_sides_l104_104835


namespace binomial_coefficient_x5_l104_104817

theorem binomial_coefficient_x5 :
  let binomial_term (r : ℕ) : ℕ := Nat.choose 7 r * (21 - 4 * r)
  35 = binomial_term 4 :=
by
  sorry

end binomial_coefficient_x5_l104_104817


namespace exists_strictly_increasing_sequences_l104_104057

theorem exists_strictly_increasing_sequences :
  ∃ u v : ℕ → ℕ, (∀ n, u n < u (n + 1)) ∧ (∀ n, v n < v (n + 1)) ∧ (∀ n, 5 * u n * (u n + 1) = v n ^ 2 + 1) :=
sorry

end exists_strictly_increasing_sequences_l104_104057


namespace polynomial_divisible_by_24_l104_104452

theorem polynomial_divisible_by_24 (n : ℤ) : 24 ∣ (n^4 + 6 * n^3 + 11 * n^2 + 6 * n) :=
sorry

end polynomial_divisible_by_24_l104_104452


namespace exam_scores_l104_104114

theorem exam_scores (A B C D : ℤ) 
  (h1 : A + B = C + D + 17) 
  (h2 : A = B - 4) 
  (h3 : C = D + 5) :
  ∃ highest lowest, (highest - lowest = 13) ∧ 
                   (highest = A ∨ highest = B ∨ highest = C ∨ highest = D) ∧ 
                   (lowest = A ∨ lowest = B ∨ lowest = C ∨ lowest = D) :=
by
  sorry

end exam_scores_l104_104114


namespace domain_eq_l104_104397

theorem domain_eq (f : ℝ → ℝ) : 
  (∀ x : ℝ, -1 ≤ 3 - 2 * x ∧ 3 - 2 * x ≤ 2) ↔ ∀ x : ℝ, -1 ≤ x ∧ x ≤ 5 :=
by sorry

end domain_eq_l104_104397


namespace sin_zero_degrees_l104_104902

theorem sin_zero_degrees : Real.sin 0 = 0 := 
by {
  -- The proof is added here (as requested no proof is required, hence using sorry)
  sorry
}

end sin_zero_degrees_l104_104902


namespace no_real_solution_l104_104643

theorem no_real_solution :
  ¬ ∃ x : ℝ, (1 / (x + 2) + 8 / (x + 6) ≥ 2) ∧ (5 / (x + 1) - 2 ≤ 1) :=
by
  sorry

end no_real_solution_l104_104643


namespace fg_of_3_eq_83_l104_104333

def g (x : ℝ) : ℝ := x^3
def f (x : ℝ) : ℝ := 3 * x + 2

theorem fg_of_3_eq_83 : f (g 3) = 83 := by
  sorry

end fg_of_3_eq_83_l104_104333


namespace fraction_spent_on_DVDs_l104_104610

theorem fraction_spent_on_DVDs (initial_money spent_on_books additional_books_cost remaining_money_spent fraction remaining_money_after_DVDs : ℚ) : 
  initial_money = 320 ∧
  spent_on_books = initial_money / 4 ∧
  additional_books_cost = 10 ∧
  remaining_money_spent = 230 ∧
  remaining_money_after_DVDs = 130 ∧
  remaining_money_spent = initial_money - (spent_on_books + additional_books_cost) ∧
  remaining_money_after_DVDs = remaining_money_spent - (fraction * remaining_money_spent + 8) 
  → fraction = 46 / 115 :=
by
  intros
  sorry

end fraction_spent_on_DVDs_l104_104610


namespace intersection_A_complement_B_l104_104687

-- Definition of real numbers
def R := ℝ

-- Definitions of sets A and B
def A := {x : ℝ | x > 0}
def B := {x : ℝ | x^2 - x - 2 > 0}

-- Definition of the complement of B in R
def B_complement := {x : ℝ | -1 ≤ x ∧ x ≤ 2}

-- The final statement we need to prove
theorem intersection_A_complement_B :
  A ∩ B_complement = {x : ℝ | 0 < x ∧ x ≤ 2} :=
sorry

end intersection_A_complement_B_l104_104687


namespace maria_total_flowers_l104_104713

-- Define the initial conditions
def dozens := 3
def flowers_per_dozen := 12
def free_flowers_per_dozen := 2

-- Define the total number of flowers
def total_flowers := dozens * flowers_per_dozen + dozens * free_flowers_per_dozen

-- Assert the proof statement
theorem maria_total_flowers : total_flowers = 42 := sorry

end maria_total_flowers_l104_104713


namespace min_value_reciprocal_l104_104649

theorem min_value_reciprocal (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + b = 1) :
  (1 / a + 1 / b) ≥ 4 :=
by
  sorry

end min_value_reciprocal_l104_104649


namespace nearest_integer_power_l104_104153

noncomputable def power_expression := (3 + Real.sqrt 2)^6

theorem nearest_integer_power :
  Int.floor power_expression = 7414 :=
sorry

end nearest_integer_power_l104_104153


namespace basketball_weight_l104_104286

-- Definitions based on the given conditions
variables (b c : ℕ) -- weights of basketball and bicycle in pounds

-- Condition 1: Nine basketballs weigh the same as six bicycles
axiom condition1 : 9 * b = 6 * c

-- Condition 2: Four bicycles weigh a total of 120 pounds
axiom condition2 : 4 * c = 120

-- The proof statement we need to prove
theorem basketball_weight : b = 20 :=
by
  sorry

end basketball_weight_l104_104286


namespace multiply_by_5_l104_104823

theorem multiply_by_5 (x : ℤ) (h : x - 7 = 9) : x * 5 = 80 := by
  sorry

end multiply_by_5_l104_104823


namespace part_a_part_b_l104_104541

-- Part (a)
theorem part_a (a b c : ℚ) (z : ℚ) (h : a * z^2 + b * z + c = 0) (n : ℕ) (hn : n > 0) :
  ∃ f : ℚ → ℚ, z = f (z^n) :=
sorry

-- Part (b)
theorem part_b (x : ℚ) (h : x ≠ 0) :
  x = (x^3 + (x + 1/x)) / ((x + 1/x)^2 - 1) :=
sorry

end part_a_part_b_l104_104541


namespace total_seats_round_table_l104_104121

theorem total_seats_round_table 
  (a : ℕ) (b : ℕ) 
  (h₀ : a ≠ b)
  (h₁ : a + b = 39) 
  : ∃ n, n = 38 := 
by {
  sorry
}

end total_seats_round_table_l104_104121


namespace eval_expression_l104_104512

theorem eval_expression :
  (2^2003 * 3^2005) / (6^2004) = 3 / 2 := by
  sorry

end eval_expression_l104_104512


namespace potatoes_leftover_l104_104413

-- Define the necessary conditions
def fries_per_potato : ℕ := 25
def total_potatoes : ℕ := 15
def fries_needed : ℕ := 200

-- Prove the goal
theorem potatoes_leftover : total_potatoes - (fries_needed / fries_per_potato) = 7 :=
sorry

end potatoes_leftover_l104_104413


namespace dr_jones_remaining_salary_l104_104900

noncomputable def remaining_salary (salary rent food utilities insurances taxes transport emergency loan retirement : ℝ) : ℝ :=
  salary - (rent + food + utilities + insurances + taxes + transport + emergency + loan + retirement)

theorem dr_jones_remaining_salary :
  remaining_salary 6000 640 385 (1/4 * 6000) (1/5 * 6000) (0.10 * 6000) (0.03 * 6000) (0.02 * 6000) 300 (0.05 * 6000) = 1275 :=
by
  sorry

end dr_jones_remaining_salary_l104_104900


namespace sum_nine_terms_l104_104132

-- Definitions required based on conditions provided in Step a)
variables {a : ℕ → ℝ} {S : ℕ → ℝ} {d : ℝ}

-- The arithmetic sequence condition is encapsulated here
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- The definition of S_n being the sum of the first n terms
def sum_first_n (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n * (a 1 + a n)) / 2

-- The given condition from the problem
def given_condition (a : ℕ → ℝ) : Prop :=
  2 * a 8 = 6 + a 1

-- The proof statement to show S_9 = 54 given the above conditions
theorem sum_nine_terms (h_arith : is_arithmetic_sequence a d)
                        (h_sum : sum_first_n a S) 
                        (h_given : given_condition a): 
                        S 9 = 54 :=
  by sorry

end sum_nine_terms_l104_104132


namespace each_dolphin_training_hours_l104_104837

theorem each_dolphin_training_hours
  (num_dolphins : ℕ)
  (num_trainers : ℕ)
  (hours_per_trainer : ℕ)
  (total_hours : ℕ := num_trainers * hours_per_trainer)
  (hours_per_dolphin_daily : ℕ := total_hours / num_dolphins)
  (h1 : num_dolphins = 4)
  (h2 : num_trainers = 2)
  (h3 : hours_per_trainer = 6) :
  hours_per_dolphin_daily = 3 :=
  by sorry

end each_dolphin_training_hours_l104_104837


namespace three_circles_area_less_than_total_radius_squared_l104_104234

theorem three_circles_area_less_than_total_radius_squared
    (x y z R : ℝ)
    (h1 : x > 0)
    (h2 : y > 0)
    (h3 : z > 0)
    (h4 : R > 0)
    (descartes_theorem : ( (1/x + 1/y + 1/z - 1/R)^2 = 2 * ( (1/x)^2 + (1/y)^2 + (1/z)^2 + (1/R)^2 ) )) :
    x^2 + y^2 + z^2 < 4 * R^2 := 
sorry

end three_circles_area_less_than_total_radius_squared_l104_104234


namespace rectangle_perimeter_equal_area_l104_104349

theorem rectangle_perimeter_equal_area (a b : ℕ) (h1 : a ≠ b) (h2 : a * b = 2 * a + 2 * b) : 2 * (a + b) = 18 := 
by 
  sorry

end rectangle_perimeter_equal_area_l104_104349


namespace two_bishops_placement_l104_104209

theorem two_bishops_placement :
  let squares := 64
  let white_squares := 32
  let black_squares := 32
  let first_bishop_white_positions := 32
  let second_bishop_black_positions := 32 - 8
  first_bishop_white_positions * second_bishop_black_positions = 768 := by
  sorry

end two_bishops_placement_l104_104209


namespace trigonometric_product_l104_104358

theorem trigonometric_product :
  (1 - Real.sin (Real.pi / 12)) * 
  (1 - Real.sin (5 * Real.pi / 12)) * 
  (1 - Real.sin (7 * Real.pi / 12)) * 
  (1 - Real.sin (11 * Real.pi / 12)) = 1 / 4 :=
by sorry

end trigonometric_product_l104_104358


namespace gcd_45_75_105_l104_104362

theorem gcd_45_75_105 : Nat.gcd (45 : ℕ) (Nat.gcd 75 105) = 15 := 
by
  sorry

end gcd_45_75_105_l104_104362


namespace problem_statement_l104_104828

-- Define the function f(x)
variable (f : ℝ → ℝ)

-- Conditions
axiom odd_function : ∀ x, f (-x) = -f x
axiom periodic_function : ∀ x, f (x + 4) = -f x
axiom increasing_on_0_2 : ∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 2 → f x < f y

-- Theorem to prove
theorem problem_statement : f (-10) < f 40 ∧ f 40 < f 3 :=
by
  sorry

end problem_statement_l104_104828


namespace emily_extra_distance_five_days_l104_104045

-- Define the distances
def distance_troy : ℕ := 75
def distance_emily : ℕ := 98

-- Emily's extra walking distance in one-way
def extra_one_way : ℕ := distance_emily - distance_troy

-- Emily's extra walking distance in a round trip
def extra_round_trip : ℕ := extra_one_way * 2

-- The extra distance Emily walks in five days
def extra_five_days : ℕ := extra_round_trip * 5

-- Theorem to be proven
theorem emily_extra_distance_five_days : extra_five_days = 230 := by
  -- Proof will go here
  sorry

end emily_extra_distance_five_days_l104_104045


namespace red_light_after_two_red_light_expectation_and_variance_l104_104596

noncomputable def prob_red_light_after_two : ℚ := (2/3) * (2/3) * (1/3)
theorem red_light_after_two :
  prob_red_light_after_two = 4/27 :=
by
  -- We have defined the probability calculation directly
  sorry

noncomputable def expected_red_lights (n : ℕ) (p : ℚ) : ℚ := n * p
noncomputable def variance_red_lights (n : ℕ) (p : ℚ) : ℚ := n * p * (1 - p)

theorem red_light_expectation_and_variance :
  expected_red_lights 6 (1/3) = 2 ∧ variance_red_lights 6 (1/3) = 4/3 :=
by
  -- We have defined expectation and variance calculations directly
  sorry

end red_light_after_two_red_light_expectation_and_variance_l104_104596


namespace shrimp_cost_per_pound_l104_104858

theorem shrimp_cost_per_pound 
    (shrimp_per_guest : ℕ) 
    (num_guests : ℕ) 
    (shrimp_per_pound : ℕ) 
    (total_cost : ℝ)
    (H1 : shrimp_per_guest = 5)
    (H2 : num_guests = 40)
    (H3 : shrimp_per_pound = 20)
    (H4 : total_cost = 170) : 
    (total_cost / ((num_guests * shrimp_per_guest) / shrimp_per_pound) = 17) :=
by
    sorry

end shrimp_cost_per_pound_l104_104858


namespace concrete_pillars_l104_104090

-- Definitions based on the conditions of the problem
def C_deck : ℕ := 1600
def C_anchor : ℕ := 700
def C_total : ℕ := 4800

-- Theorem to prove the concrete required for supporting pillars
theorem concrete_pillars : C_total - (C_deck + 2 * C_anchor) = 1800 :=
by sorry

end concrete_pillars_l104_104090


namespace correct_calculation_l104_104751

theorem correct_calculation :
  (- (4 + 2 / 3) - (1 + 5 / 6) - (- (18 + 1 / 2)) + (- (13 + 3 / 4))) = - (7 / 4) :=
by 
  sorry

end correct_calculation_l104_104751


namespace double_root_condition_l104_104802

theorem double_root_condition (a : ℝ) : 
  (∃! x : ℝ, (x+2)^2 * (x+7)^2 + a = 0) ↔ a = -625 / 16 :=
sorry

end double_root_condition_l104_104802


namespace expand_expression_l104_104651

theorem expand_expression (a b : ℤ) : (-1 + a * b^2)^2 = 1 - 2 * a * b^2 + a^2 * b^4 :=
by sorry

end expand_expression_l104_104651


namespace unique_solution_for_system_l104_104702

theorem unique_solution_for_system (a : ℝ) :
  (∃! (x y : ℝ), x^2 + 4 * y^2 = 1 ∧ x + 2 * y = a) ↔ a = -1.41 :=
by
  sorry

end unique_solution_for_system_l104_104702


namespace binomial_constant_term_l104_104540

theorem binomial_constant_term : 
  ∃ (c : ℚ), (x : ℝ) → (x^2 + (1 / (2 * x)))^6 = c ∧ c = 15 / 16 := by
  sorry

end binomial_constant_term_l104_104540


namespace cost_to_fill_pool_l104_104874

noncomputable def pool_cost : ℝ :=
  let base_width := 6
  let top_width := 4
  let length := 20
  let depth := 10
  let conversion_factor := 25
  let price_per_liter := 3
  let tax_rate := 0.08
  let discount_rate := 0.05
  let volume := 0.5 * depth * (base_width + top_width) * length
  let liters := volume * conversion_factor
  let initial_cost := liters * price_per_liter
  let cost_with_tax := initial_cost * (1 + tax_rate)
  let final_cost := cost_with_tax * (1 - discount_rate)
  final_cost

theorem cost_to_fill_pool : pool_cost = 76950 := by
  sorry

end cost_to_fill_pool_l104_104874


namespace total_cost_correct_l104_104065

-- Definitions for the conditions
def num_ladders_1 : ℕ := 10
def rungs_1 : ℕ := 50
def cost_per_rung_1 : ℕ := 2

def num_ladders_2 : ℕ := 20
def rungs_2 : ℕ := 60
def cost_per_rung_2 : ℕ := 3

def num_ladders_3 : ℕ := 30
def rungs_3 : ℕ := 80
def cost_per_rung_3 : ℕ := 4

-- Total cost calculation for the client
def total_cost : ℕ :=
  (num_ladders_1 * rungs_1 * cost_per_rung_1) +
  (num_ladders_2 * rungs_2 * cost_per_rung_2) +
  (num_ladders_3 * rungs_3 * cost_per_rung_3)

-- Statement to be proved
theorem total_cost_correct : total_cost = 14200 :=
by {
  sorry
}

end total_cost_correct_l104_104065


namespace total_points_l104_104316

theorem total_points (paul_points cousin_points : ℕ) 
  (h_paul : paul_points = 3103) 
  (h_cousin : cousin_points = 2713) : 
  paul_points + cousin_points = 5816 := by
sorry

end total_points_l104_104316


namespace max_value_of_expression_l104_104917

theorem max_value_of_expression (x y : ℝ) (h : 3 * x^2 + y^2 ≤ 3) : 2 * x + 3 * y ≤ Real.sqrt 31 :=
sorry

end max_value_of_expression_l104_104917


namespace jenny_sold_boxes_l104_104425

-- Given conditions as definitions
def cases : ℕ := 3
def boxes_per_case : ℕ := 8

-- Mathematically equivalent proof problem
theorem jenny_sold_boxes : cases * boxes_per_case = 24 := by
  sorry

end jenny_sold_boxes_l104_104425


namespace wheel_speed_l104_104694

theorem wheel_speed (s : ℝ) (t : ℝ) :
  (12 / 5280) * 3600 = s * t →
  (12 / 5280) * 3600 = (s + 4) * (t - (1 / 18000)) →
  s = 8 :=
by
  intro h1 h2
  sorry

end wheel_speed_l104_104694


namespace problem_statement_l104_104322

noncomputable def f (x : ℝ) : ℝ := sorry

axiom even_function : ∀ x : ℝ, f (-x) = f (x)
axiom functional_eq : ∀ x : ℝ, f (x + 6) = f (x) + f 3
axiom initial_condition : f 2 = 3

theorem problem_statement : f 2006 + f 2007 = 3 :=
by
  sorry

end problem_statement_l104_104322


namespace simplify_tan_alpha_l104_104937

noncomputable def f (α : ℝ) : ℝ :=
(Real.sin (Real.pi / 2 + α) + Real.sin (-Real.pi - α)) /
  (3 * Real.cos (2 * Real.pi - α) + Real.cos (3 * Real.pi / 2 - α))

theorem simplify_tan_alpha (α : ℝ) (h : Real.tan α = 1) : f α = 1 := by
  sorry

end simplify_tan_alpha_l104_104937


namespace f_increasing_on_pos_real_l104_104297

noncomputable def f (x : ℝ) : ℝ := x^2 / (x^2 + 1)

theorem f_increasing_on_pos_real : ∀ x1 x2 : ℝ, 0 < x1 → 0 < x2 → x1 < x2 → f x1 < f x2 :=
by sorry

end f_increasing_on_pos_real_l104_104297


namespace units_digit_47_pow_47_l104_104055

theorem units_digit_47_pow_47 : (47^47) % 10 = 3 :=
  sorry

end units_digit_47_pow_47_l104_104055


namespace expand_product_l104_104170

theorem expand_product (x : ℝ) :
  (x + 4) * (x - 5) = x^2 - x - 20 :=
by
  -- The proof will use algebraic identities and simplifications.
  sorry

end expand_product_l104_104170


namespace ca_co3_to_ca_cl2_l104_104507

theorem ca_co3_to_ca_cl2 (caCO3 HCl : ℕ) (main_reaction : caCO3 = 1 ∧ HCl = 2) : ∃ CaCl2, CaCl2 = 1 :=
by
  -- The proof of the theorem will go here.
  sorry

end ca_co3_to_ca_cl2_l104_104507


namespace work_finish_in_3_days_l104_104960

-- Define the respective rates of work
def A_rate := 1/4
def B_rate := 1/14
def C_rate := 1/7

-- Define the duration they start working together
def initial_duration := 2
def after_C_joining := 1 -- time after C joins before A leaves

-- From the third day, consider A leaving the job
theorem work_finish_in_3_days :
  (initial_duration * (A_rate + B_rate)) + 
  (after_C_joining * (A_rate + B_rate + C_rate)) + 
  ((1 : ℝ) - after_C_joining) * (B_rate + C_rate) >= 1 :=
by
  sorry

end work_finish_in_3_days_l104_104960


namespace water_polo_team_selection_l104_104236

theorem water_polo_team_selection :
  let total_players := 20
  let team_size := 9
  let goalies := 2
  let remaining_players := total_players - goalies
  let combination (n k : ℕ) := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
  combination total_players goalies * combination remaining_players (team_size - goalies) = 6046560 :=
by
  -- Definitions and calculations to be filled here.
  sorry

end water_polo_team_selection_l104_104236


namespace sufficient_but_not_necessary_condition_for_intersections_l104_104794

theorem sufficient_but_not_necessary_condition_for_intersections
  (k : ℝ) (h : 0 < k ∧ k < 3) :
  ∃ x y : ℝ, (x - y - k = 0) ∧ ((x - 1)^2 + y^2 = 2) :=
sorry

end sufficient_but_not_necessary_condition_for_intersections_l104_104794


namespace find_t_l104_104496

theorem find_t (t : ℚ) : 
  ((t + 2) * (3 * t - 2) = (3 * t - 4) * (t + 1) + 5) → t = 5 / 3 :=
by
  intro h
  sorry

end find_t_l104_104496


namespace solution_set_of_inequality_l104_104886

theorem solution_set_of_inequality (x : ℝ) : (x * (2 - x) ≤ 0) ↔ (x ≤ 0 ∨ x ≥ 2) :=
by
  sorry

end solution_set_of_inequality_l104_104886


namespace fg_sqrt3_eq_neg3_minus_2sqrt3_l104_104583
noncomputable def f (x : ℝ) : ℝ := 5 - 2 * x
noncomputable def g (x : ℝ) : ℝ := x^2 + x + 1

theorem fg_sqrt3_eq_neg3_minus_2sqrt3 : f (g (Real.sqrt 3)) = -3 - 2 * Real.sqrt 3 := 
by sorry

end fg_sqrt3_eq_neg3_minus_2sqrt3_l104_104583


namespace regular_polygon_sides_l104_104564

-- Conditions
def central_angle (θ : ℝ) := θ = 30
def sum_of_central_angles (sumθ : ℝ) := sumθ = 360

-- The proof problem
theorem regular_polygon_sides (θ sumθ : ℝ) (h₁ : central_angle θ) (h₂ : sum_of_central_angles sumθ) :
  sumθ / θ = 12 := by
  sorry

end regular_polygon_sides_l104_104564


namespace total_games_is_seven_l104_104238

def total_football_games (games_missed : ℕ) (games_attended : ℕ) : ℕ :=
  games_missed + games_attended

theorem total_games_is_seven : total_football_games 4 3 = 7 := 
by
  sorry

end total_games_is_seven_l104_104238


namespace max_distance_between_vertices_l104_104800

theorem max_distance_between_vertices (inner_perimeter outer_perimeter : ℕ) 
  (inner_perimeter_eq : inner_perimeter = 20) 
  (outer_perimeter_eq : outer_perimeter = 28) : 
  ∃ x y, x + y = 7 ∧ x^2 + y^2 = 25 ∧ (x^2 + (x + y)^2 = 65) :=
by
  sorry

end max_distance_between_vertices_l104_104800


namespace car_speed_conversion_l104_104456

theorem car_speed_conversion :
  let speed_mps := 10 -- speed of the car in meters per second
  let conversion_factor := 3.6 -- conversion factor from m/s to km/h
  let speed_kmph := speed_mps * conversion_factor -- speed of the car in kilometers per hour
  speed_kmph = 36 := 
by
  sorry

end car_speed_conversion_l104_104456


namespace share_of_c_l104_104454

variable (a b c : ℝ)

theorem share_of_c (h1 : a + b + c = 427) (h2 : 3 * a = 7 * c) (h3 : 4 * b = 7 * c) : c = 84 :=
  by
  sorry

end share_of_c_l104_104454


namespace obtuse_triangle_iff_l104_104639

theorem obtuse_triangle_iff (x : ℝ) :
    (x > 1 ∧ x < 3) ↔ (x + (x + 1) > (x + 2) ∧
                        (x + 1) + (x + 2) > x ∧
                        (x + 2) + x > (x + 1) ∧
                        (x + 2)^2 > x^2 + (x + 1)^2) :=
by
  sorry

end obtuse_triangle_iff_l104_104639


namespace symmetric_coordinates_l104_104001

structure Point :=
  (x : Int)
  (y : Int)

def symmetric_about_origin (p : Point) : Point :=
  ⟨-p.x, -p.y⟩

theorem symmetric_coordinates (P : Point) (h : P = Point.mk (-1) 2) :
  symmetric_about_origin P = Point.mk 1 (-2) :=
by
  sorry

end symmetric_coordinates_l104_104001


namespace exists_large_p_l104_104984

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.sin (Real.pi * x)

theorem exists_large_p (d : ℝ) (h : d > 0) : ∃ p : ℝ, ∀ x : ℝ, |f (x + p) - f x| < d ∧ ∃ M : ℝ, M > 0 ∧ p > M :=
by {
  sorry
}

end exists_large_p_l104_104984


namespace distance_to_canada_l104_104266

theorem distance_to_canada (speed : ℝ) (total_time : ℝ) (stop_time : ℝ) (driving_time : ℝ) (distance : ℝ) :
  speed = 60 ∧ total_time = 7 ∧ stop_time = 1 ∧ driving_time = total_time - stop_time ∧
  distance = speed * driving_time → distance = 360 :=
by
  sorry

end distance_to_canada_l104_104266


namespace intersection_point_correct_l104_104545

-- Points in 3D coordinate space
def P : ℝ × ℝ × ℝ := (3, -9, 6)
def Q : ℝ × ℝ × ℝ := (13, -19, 11)
def R : ℝ × ℝ × ℝ := (1, 4, -7)
def S : ℝ × ℝ × ℝ := (3, -6, 9)

-- Vectors for parameterization
def pq_vector (t : ℝ) : ℝ × ℝ × ℝ := (3 + 10 * t, -9 - 10 * t, 6 + 5 * t)
def rs_vector (s : ℝ) : ℝ × ℝ × ℝ := (1 + 2 * s, 4 - 10 * s, -7 + 16 * s)

-- The proof of the intersection point equals the correct answer
theorem intersection_point_correct : 
  ∃ t s : ℝ, pq_vector t = rs_vector s ∧ 
  pq_vector t = (-19 / 3, 10 / 3, 4 / 3) := 
by
  sorry

end intersection_point_correct_l104_104545


namespace james_total_money_l104_104755

section
-- Conditions
def number_of_bills : ℕ := 3
def value_of_each_bill : ℕ := 20
def initial_wallet_amount : ℕ := 75

-- Question:
-- What is the total amount of money James has now?
def total_value_of_bills : ℕ := number_of_bills * value_of_each_bill
def total_money_now : ℕ := initial_wallet_amount + total_value_of_bills

-- Theorem stating that he has $135 now.
theorem james_total_money : total_money_now = 135 := 
  by
    sorry
end

end james_total_money_l104_104755


namespace evaluate_expr_l104_104505

theorem evaluate_expr : 3 * (3 * (3 * (3 * (3 * (3 * 2 * 2) * 2) * 2) * 2) * 2) * 2 = 1458 := by
  sorry

end evaluate_expr_l104_104505


namespace degree_product_l104_104366

-- Define the degrees of the polynomials p and q
def degree_p : ℕ := 3
def degree_q : ℕ := 4

-- Define the functions p(x) and q(x) as polynomials and their respective degrees
axiom degree_p_definition (p : Polynomial ℝ) : p.degree = degree_p
axiom degree_q_definition (q : Polynomial ℝ) : q.degree = degree_q

-- Define the degree of the product p(x^2) * q(x^4)
noncomputable def degree_p_x2_q_x4 (p q : Polynomial ℝ) : ℕ :=
  2 * degree_p + 4 * degree_q

-- Prove that the degree of p(x^2) * q(x^4) is 22
theorem degree_product (p q : Polynomial ℝ) (hp : p.degree = degree_p) (hq : q.degree = degree_q) :
  degree_p_x2_q_x4 p q = 22 :=
by
  sorry

end degree_product_l104_104366


namespace pipe_filling_time_l104_104589

theorem pipe_filling_time (T : ℝ) (h1 : T > 0) (h2 : 1/(3:ℝ) = 1/T - 1/(6:ℝ)) : T = 2 := 
by sorry

end pipe_filling_time_l104_104589


namespace K_travel_time_40_miles_l104_104767

noncomputable def K_time (x : ℝ) : ℝ := 40 / x

theorem K_travel_time_40_miles (x : ℝ) (d : ℝ) (Δt : ℝ)
  (h1 : d = 40)
  (h2 : Δt = 1 / 3)
  (h3 : ∃ (Kmiles_r : ℝ) (Mmiles_r : ℝ), Kmiles_r = x ∧ Mmiles_r = x - 0.5)
  (h4 : ∃ (Ktime : ℝ) (Mtime : ℝ), Ktime = d / x ∧ Mtime = d / (x - 0.5) ∧ Mtime - Ktime = Δt) :
  K_time x = 5 := sorry

end K_travel_time_40_miles_l104_104767


namespace initial_girls_are_11_l104_104658

variable {n : ℕ}  -- Assume n (the total number of students initially) is a natural number

def initial_num_girls (n : ℕ) : ℕ := (n / 2)

def total_students_after_changes (n : ℕ) : ℕ := n - 2

def num_girls_after_changes (n : ℕ) : ℕ := (n / 2) - 3

def is_40_percent_girls (n : ℕ) : Prop := (num_girls_after_changes n) * 10 = 4 * (total_students_after_changes n)

theorem initial_girls_are_11 :
  is_40_percent_girls 22 → initial_num_girls 22 = 11 :=
by
  sorry

end initial_girls_are_11_l104_104658


namespace ellipse_conjugate_diameters_l104_104838

variable (A B C D E : ℝ)

theorem ellipse_conjugate_diameters :
  (A * E - B * D = 0) ∧ (2 * B ^ 2 + (A - C) * A = 0) :=
sorry

end ellipse_conjugate_diameters_l104_104838


namespace leap_years_count_l104_104597

theorem leap_years_count :
  let is_leap_year (y : ℕ) := (y % 900 = 150 ∨ y % 900 = 450) ∧ y % 100 = 0
  let range_start := 2100
  let range_end := 4200
  ∃ L, L = [2250, 2850, 3150, 3750, 4050] ∧ (∀ y ∈ L, is_leap_year y ∧ range_start ≤ y ∧ y ≤ range_end)
  ∧ L.length = 5 :=
by
  sorry

end leap_years_count_l104_104597


namespace geometric_sequence_178th_term_l104_104343

-- Conditions of the problem as definitions
def first_term : ℤ := 5
def second_term : ℤ := -20
def common_ratio : ℤ := second_term / first_term
def nth_term (a : ℤ) (r : ℤ) (n : ℕ) : ℤ := a * r^(n-1)

-- The translated problem statement in Lean 4
theorem geometric_sequence_178th_term :
  nth_term first_term common_ratio 178 = -5 * 4^177 :=
by
  repeat { sorry }

end geometric_sequence_178th_term_l104_104343


namespace parallel_vectors_implies_value_of_x_l104_104928

-- Define the vectors a and b
def a : ℝ × ℝ := (4, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 3)

-- Define the condition for parallel vectors
def parallel (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, (u.1 = k * v.1) ∧ (u.2 = k * v.2)

-- The proof statement
theorem parallel_vectors_implies_value_of_x : ∀ (x : ℝ), parallel a (b x) → x = 6 :=
by
  intro x
  intro h
  sorry

end parallel_vectors_implies_value_of_x_l104_104928


namespace smallest_positive_multiple_of_37_l104_104923

theorem smallest_positive_multiple_of_37 (a : ℕ) (h1 : 37 * a ≡ 3 [MOD 101]) (h2 : ∀ b : ℕ, 0 < b ∧ (37 * b ≡ 3 [MOD 101]) → a ≤ b) : 37 * a = 1628 :=
sorry

end smallest_positive_multiple_of_37_l104_104923


namespace find_y_when_x_is_4_l104_104285

def inverse_proportional (x y : ℝ) : Prop :=
  ∃ C : ℝ, x * y = C

theorem find_y_when_x_is_4 :
  ∀ x y : ℝ,
  inverse_proportional x y →
  (x + y = 20) →
  (x - y = 4) →
  (∃ y, y = 24 ∧ x = 4) :=
by
  sorry

end find_y_when_x_is_4_l104_104285


namespace driving_time_l104_104638

-- Conditions from problem
variable (distance1 : ℕ) (time1 : ℕ) (distance2 : ℕ)
variable (same_speed : distance1 / time1 = distance2 / (5 : ℕ))

-- Statement to prove
theorem driving_time (h1 : distance1 = 120) (h2 : time1 = 3) (h3 : distance2 = 200)
  : distance2 / (40 : ℕ) = (5 : ℕ) := by
  sorry

end driving_time_l104_104638


namespace chocolate_bars_l104_104022

theorem chocolate_bars (num_small_boxes : ℕ) (num_bars_per_box : ℕ) (total_bars : ℕ) (h1 : num_small_boxes = 20) (h2 : num_bars_per_box = 32) (h3 : total_bars = num_small_boxes * num_bars_per_box) :
  total_bars = 640 :=
by
  sorry

end chocolate_bars_l104_104022


namespace find_a_l104_104447

def f (a x : ℝ) : ℝ := a * x^3 + 3 * x^2 + 2

theorem find_a (a : ℝ) 
  (h : deriv (f a) (-1) = 4) : 
  a = 10 / 3 :=
sorry

end find_a_l104_104447


namespace average_annual_growth_rate_in_2014_and_2015_l104_104175

noncomputable def average_annual_growth_rate (p2013 p2015 : ℝ) (x : ℝ) : Prop :=
  p2013 * (1 + x)^2 = p2015

theorem average_annual_growth_rate_in_2014_and_2015 :
  average_annual_growth_rate 6.4 10 0.25 :=
by
  unfold average_annual_growth_rate
  sorry

end average_annual_growth_rate_in_2014_and_2015_l104_104175


namespace fred_fewer_games_l104_104894

/-- Fred went to 36 basketball games last year -/
def games_last_year : ℕ := 36

/-- Fred went to 25 basketball games this year -/
def games_this_year : ℕ := 25

/-- Prove that Fred went to 11 fewer games this year compared to last year -/
theorem fred_fewer_games : games_last_year - games_this_year = 11 := by
  sorry

end fred_fewer_games_l104_104894


namespace necessary_condition_inequality_l104_104271

theorem necessary_condition_inequality (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  ∀ x : ℝ, x^2 - 2 * a * x + a > 0 := 
sorry

end necessary_condition_inequality_l104_104271


namespace mary_bought_48_cards_l104_104165

variable (M T F C B : ℕ)

theorem mary_bought_48_cards
  (h1 : M = 18)
  (h2 : T = 8)
  (h3 : F = 26)
  (h4 : C = 84) :
  B = C - (M - T + F) :=
by
  -- Proof would go here
  sorry

end mary_bought_48_cards_l104_104165


namespace star_property_l104_104756

-- Define the custom operation
def star (a b : ℝ) : ℝ := a^2 - b

-- Define the property to prove
theorem star_property (x y : ℝ) : star (x - y) (x + y) = x^2 - x - 2 * x * y + y^2 - y :=
by sorry

end star_property_l104_104756


namespace part_one_part_two_part_three_l104_104086

theorem part_one : 12 - (-11) - 1 = 22 := 
by
  sorry

theorem part_two : -(1 ^ 4) / ((-3) ^ 2) / (9 / 5) = -5 / 81 := 
by
  sorry

theorem part_three : -8 * (1/2 - 3/4 + 5/8) = -3 := 
by
  sorry

end part_one_part_two_part_three_l104_104086


namespace probability_point_between_lines_l104_104171

theorem probability_point_between_lines {x y : ℝ} :
  (∀ x, y = -2 * x + 8) →
  (∀ x, y = -3 * x + 8) →
  0.33 = 0.33 :=
by
  intro hl hm
  sorry

end probability_point_between_lines_l104_104171


namespace quadratic_trinomial_m_eq_2_l104_104669

theorem quadratic_trinomial_m_eq_2 (m : ℤ) (P : |m| = 2 ∧ m + 2 ≠ 0) : m = 2 :=
  sorry

end quadratic_trinomial_m_eq_2_l104_104669


namespace ratio_a3_a6_l104_104035

variable (a : ℕ → ℝ) (d : ℝ)
-- aₙ is an arithmetic sequence
variable (h_arith_seq : ∀ n : ℕ, a (n + 1) = a n + d)
-- d ≠ 0
variable (h_d_nonzero : d ≠ 0)
-- a₃² = a₁a₉
variable (h_condition : (a 2)^2 = (a 0) * (a 8))

theorem ratio_a3_a6 : (a 2) / (a 5) = 1 / 2 :=
by
  -- Proof omitted
  sorry

end ratio_a3_a6_l104_104035


namespace sum_of_reciprocals_l104_104042

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 3 * x * y) : 
  1 / x + 1 / y = 3 :=
by
  sorry

end sum_of_reciprocals_l104_104042


namespace set_M_listed_correctly_l104_104241

theorem set_M_listed_correctly :
  {a : ℕ+ | ∃ (n : ℤ), 4 = n * (1 - a)} = {2, 3, 4} := by
sorry

end set_M_listed_correctly_l104_104241


namespace triangle_BC_60_l104_104138

theorem triangle_BC_60 {A B C X : Type}
    (AB AC BX CX : ℕ) (h1 : AB = 70) (h2 : AC = 80) 
    (h3 : AB^2 - BX^2 = CX*(CX + BX)) 
    (h4 : BX % 7 = 0)
    (h5 : BX + CX = (BC : ℕ)) 
    (h6 : BC = 60) :
  BC = 60 := 
sorry

end triangle_BC_60_l104_104138


namespace exists_four_distinct_natural_numbers_sum_any_three_prime_l104_104737

theorem exists_four_distinct_natural_numbers_sum_any_three_prime :
  ∃ a b c d : ℕ, (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧ 
  (Prime (a + b + c) ∧ Prime (a + b + d) ∧ Prime (a + c + d) ∧ Prime (b + c + d)) :=
sorry

end exists_four_distinct_natural_numbers_sum_any_three_prime_l104_104737


namespace num_points_common_to_graphs_l104_104201

theorem num_points_common_to_graphs :
  (∃ (x y : ℝ), (2 * x - y + 3 = 0 ∧ x + y - 3 = 0)) ∧
  (∃ (x y : ℝ), (2 * x - y + 3 = 0 ∧ 3 * x - 4 * y + 8 = 0)) ∧
  (∃ (x y : ℝ), (4 * x + y - 5 = 0 ∧ x + y - 3 = 0)) ∧
  (∃ (x y : ℝ), (4 * x + y - 5 = 0 ∧ 3 * x - 4 * y + 8 = 0)) ∧
  ∀ (x y : ℝ), ((2 * x - y + 3 = 0 ∨ 4 * x + y - 5 = 0) ∧ (x + y - 3 = 0 ∨ 3 * x - 4 * y + 8 = 0)) →
  ∃ (p1 p2 p3 p4 : ℝ × ℝ), 
  p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 :=
sorry

end num_points_common_to_graphs_l104_104201


namespace board_partition_possible_l104_104933

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

end board_partition_possible_l104_104933


namespace radius_of_cylinder_l104_104559

-- Define the main parameters and conditions
def diameter_cone := 8
def radius_cone := diameter_cone / 2
def altitude_cone := 10
def height_cylinder (r : ℝ) := 2 * r

-- Assume similarity of triangles
theorem radius_of_cylinder (r : ℝ) (h_c := height_cylinder r) :
  altitude_cone - h_c / r = altitude_cone / radius_cone → r = 20 / 9 := 
by
  intro h
  sorry

end radius_of_cylinder_l104_104559


namespace roots_of_quadratic_solve_inequality_l104_104940

theorem roots_of_quadratic (a b : ℝ) (h1 : ∀ x : ℝ, a * x^2 - 3 * x + 2 > 0 ↔ x < 1 ∨ x > b) :
  a = 1 ∧ b = 2 :=
by
  sorry

theorem solve_inequality (c : ℝ) :
  let a := 1
  let b := 2
  ∀ x : ℝ, a * x^2 - (a * c + b) * x + b * x < 0 ↔
    (c > 0 → (0 < x ∧ x < c)) ∧
    (c = 0 → false) ∧
    (c < 0 → (c < x ∧ x < 0)) :=
by
  sorry

end roots_of_quadratic_solve_inequality_l104_104940


namespace stock_initial_value_l104_104020

theorem stock_initial_value (V : ℕ) (h : ∀ n ≤ 99, V + n = 200 - (99 - n)) : V = 101 :=
sorry

end stock_initial_value_l104_104020


namespace sin_value_proof_l104_104385

theorem sin_value_proof (θ : ℝ) (h : Real.cos (5 * Real.pi / 12 - θ) = 1 / 3) :
  Real.sin (Real.pi / 12 + θ) = 1 / 3 := by
  sorry

end sin_value_proof_l104_104385


namespace possible_c_value_l104_104969

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) (c : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

theorem possible_c_value (a b c : ℝ) 
  (h1 : f (-1) a b c = f (-2) a b c) 
  (h2 : f (-2) a b c = f (-3) a b c) 
  (h3 : 0 ≤ f (-1) a b c) 
  (h4 : f (-1) a b c ≤ 3) : 
  6 ≤ c ∧ c ≤ 9 := sorry

end possible_c_value_l104_104969


namespace average_age_of_large_family_is_correct_l104_104417

def average_age_of_family 
  (num_grandparents : ℕ) (avg_age_grandparents : ℕ) 
  (num_parents : ℕ) (avg_age_parents : ℕ) 
  (num_children : ℕ) (avg_age_children : ℕ) 
  (num_siblings : ℕ) (avg_age_siblings : ℕ)
  (num_cousins : ℕ) (avg_age_cousins : ℕ)
  (num_aunts : ℕ) (avg_age_aunts : ℕ) : ℕ := 
  let total_age := num_grandparents * avg_age_grandparents + 
                   num_parents * avg_age_parents + 
                   num_children * avg_age_children + 
                   num_siblings * avg_age_siblings + 
                   num_cousins * avg_age_cousins + 
                   num_aunts * avg_age_aunts
  let total_family_members := num_grandparents + num_parents + num_children + num_siblings + num_cousins + num_aunts
  (total_age : ℕ) / total_family_members

theorem average_age_of_large_family_is_correct :
  average_age_of_family 4 67 3 41 5 8 2 35 3 22 2 45 = 35 := 
by 
  sorry

end average_age_of_large_family_is_correct_l104_104417


namespace jamie_school_distance_l104_104409

theorem jamie_school_distance
  (v : ℝ) -- usual speed in miles per hour
  (d : ℝ) -- distance to school in miles
  (h1 : (20 : ℝ) / 60 = 1 / 3) -- usual time to school in hours
  (h2 : (10 : ℝ) / 60 = 1 / 6) -- lighter traffic time in hours
  (h3 : d = v * (1 / 3)) -- distance equation for usual traffic
  (h4 : d = (v + 15) * (1 / 6)) -- distance equation for lighter traffic
  : d = 5 := by
  sorry

end jamie_school_distance_l104_104409


namespace distinct_ways_to_distribute_l104_104919

theorem distinct_ways_to_distribute :
  ∃ (n : ℕ), n = 7 ∧ ∀ (balls : ℕ) (boxes : ℕ)
    (indistinguishable_balls : Prop := true) 
    (indistinguishable_boxes : Prop := true), 
    balls = 6 → boxes = 3 → 
    indistinguishable_balls → 
    indistinguishable_boxes → 
    n = 7 :=
by
  sorry

end distinct_ways_to_distribute_l104_104919


namespace least_number_with_remainder_l104_104508

theorem least_number_with_remainder (n : ℕ) (d₁ d₂ d₃ d₄ r : ℕ) 
  (h₁ : d₁ = 5) (h₂ : d₂ = 6) (h₃ : d₃ = 9) (h₄ : d₄ = 12) (hr : r = 184) :
  (∀ d, d ∈ [d₁, d₂, d₃, d₄] → n % d = r % d) → n = 364 := 
sorry

end least_number_with_remainder_l104_104508


namespace haley_money_difference_l104_104420

def initial_amount : ℕ := 2
def chores : ℕ := 5
def birthday : ℕ := 10
def neighbor : ℕ := 7
def candy : ℕ := 3
def lost : ℕ := 2

theorem haley_money_difference : (initial_amount + chores + birthday + neighbor - candy - lost) - initial_amount = 17 := by
  sorry

end haley_money_difference_l104_104420


namespace add_fractions_l104_104866

theorem add_fractions :
  (11 / 12) + (7 / 8) + (3 / 4) = 61 / 24 :=
by
  sorry

end add_fractions_l104_104866


namespace solve_for_q_l104_104202

theorem solve_for_q (p q : ℚ) (h1 : 5 * p + 6 * q = 20) (h2 : 6 * p + 5 * q = 29) : q = -25 / 11 :=
by {
  sorry
}

end solve_for_q_l104_104202


namespace sin_cos_sum_l104_104975

open Real

theorem sin_cos_sum : sin (47 : ℝ) * cos (43 : ℝ) + cos (47 : ℝ) * sin (43 : ℝ) = 1 :=
by
  sorry

end sin_cos_sum_l104_104975


namespace real_y_values_l104_104283

theorem real_y_values (x : ℝ) :
  (∃ y : ℝ, 2 * y^2 + 3 * x * y - x + 8 = 0) ↔ (x ≤ -23 / 9 ∨ x ≥ 5 / 3) :=
by
  sorry

end real_y_values_l104_104283


namespace third_sec_second_chap_more_than_first_sec_third_chap_l104_104453

-- Define the page lengths for each section in each chapter
def first_chapter : List ℕ := [20, 10, 30]
def second_chapter : List ℕ := [5, 12, 8, 22]
def third_chapter : List ℕ := [7, 11]

-- Define the specific sections of interest
def third_section_second_chapter := second_chapter[2]  -- 8
def first_section_third_chapter := third_chapter[0]   -- 7

-- The theorem we want to prove
theorem third_sec_second_chap_more_than_first_sec_third_chap :
  third_section_second_chapter - first_section_third_chapter = 1 :=
by
  -- Sorry is used here to skip the proof.
  sorry

end third_sec_second_chap_more_than_first_sec_third_chap_l104_104453


namespace taxi_ride_cost_l104_104565

-- Definitions based on the conditions
def fixed_cost : ℝ := 2.00
def variable_cost_per_mile : ℝ := 0.30
def distance_traveled : ℝ := 7

-- Theorem statement
theorem taxi_ride_cost : fixed_cost + (variable_cost_per_mile * distance_traveled) = 4.10 :=
by
  sorry

end taxi_ride_cost_l104_104565


namespace graphs_symmetric_y_axis_l104_104959

theorem graphs_symmetric_y_axis : ∀ (x : ℝ), (-x) ∈ { y | y = 3^(-x) } ↔ x ∈ { y | y = 3^x } :=
by
  intro x
  sorry

end graphs_symmetric_y_axis_l104_104959


namespace ted_age_l104_104461

theorem ted_age (t s : ℝ) 
  (h1 : t = 3 * s - 20) 
  (h2: t + s = 70) : 
  t = 47.5 := 
by
  sorry

end ted_age_l104_104461


namespace sequence_property_l104_104497

theorem sequence_property (a : ℕ+ → ℚ)
  (h1 : ∀ p q : ℕ+, a p + a q = a (p + q))
  (h2 : a 1 = 1 / 9) :
  a 36 = 4 :=
sorry

end sequence_property_l104_104497


namespace combined_salaries_of_B_C_D_E_l104_104039

theorem combined_salaries_of_B_C_D_E
    (A_salary : ℕ)
    (average_salary_all : ℕ)
    (total_individuals : ℕ)
    (combined_salaries_B_C_D_E : ℕ) :
    A_salary = 8000 →
    average_salary_all = 8800 →
    total_individuals = 5 →
    combined_salaries_B_C_D_E = (average_salary_all * total_individuals) - A_salary →
    combined_salaries_B_C_D_E = 36000 :=
by
  sorry

end combined_salaries_of_B_C_D_E_l104_104039


namespace gcd_of_7854_and_15246_is_6_six_is_not_prime_l104_104005

theorem gcd_of_7854_and_15246_is_6 : gcd 7854 15246 = 6 := sorry

theorem six_is_not_prime : ¬ Prime 6 := sorry

end gcd_of_7854_and_15246_is_6_six_is_not_prime_l104_104005


namespace ratio_third_to_others_l104_104762

-- Definitions of the heights
def H1 := 600
def H2 := 2 * H1
def H3 := 7200 - (H1 + H2)

-- Definition of the ratio to be proved
def ratio := H3 / (H1 + H2)

-- The theorem statement in Lean 4
theorem ratio_third_to_others : ratio = 3 := by
  have hH1 : H1 = 600 := rfl
  have hH2 : H2 = 2 * 600 := rfl
  have hH3 : H3 = 7200 - (600 + 1200) := rfl
  have h_total : 600 + 1200 + H3 = 7200 := sorry
  have h_ratio : (7200 - (600 + 1200)) / (600 + 1200) = 3 := by sorry
  sorry

end ratio_third_to_others_l104_104762


namespace red_balls_in_bag_l104_104517

/-- Given the conditions of the ball distribution in the bag,
we need to prove the number of red balls is 9. -/
theorem red_balls_in_bag (total_balls white_balls green_balls yellow_balls purple_balls : ℕ)
  (prob_neither_red_nor_purple : ℝ) (h_total : total_balls = 100)
  (h_white : white_balls = 50) (h_green : green_balls = 30)
  (h_yellow : yellow_balls = 8) (h_purple : purple_balls = 3)
  (h_prob : prob_neither_red_nor_purple = 0.88) :
  ∃ R : ℕ, (total_balls = white_balls + green_balls + yellow_balls + purple_balls + R) ∧ R = 9 :=
by {
  sorry
}

end red_balls_in_bag_l104_104517


namespace ab_value_is_3360_l104_104432

noncomputable def find_ab (a b : ℤ) : Prop :=
  (a ≠ 0 ∧ b ≠ 0) ∧
  (∃ r s : ℤ, 
    (x : ℤ) → 
      (x^3 + a * x^2 + b * x + 16 * a = (x - r)^2 * (x - s)) ∧ 
      (2 * r + s = -a) ∧ 
      (r^2 + 2 * r * s = b) ∧ 
      (r^2 * s = -16 * a))

theorem ab_value_is_3360 (a b : ℤ) (h : find_ab a b) : |a * b| = 3360 :=
sorry

end ab_value_is_3360_l104_104432


namespace distance_travelled_l104_104237

def speed : ℕ := 3 -- speed in feet per second
def time : ℕ := 3600 -- time in seconds (1 hour)

theorem distance_travelled : speed * time = 10800 := by
  sorry

end distance_travelled_l104_104237


namespace number_of_girls_l104_104082

theorem number_of_girls {total_children boys girls : ℕ} 
  (h_total : total_children = 60) 
  (h_boys : boys = 18) 
  (h_girls : girls = total_children - boys) : 
  girls = 42 := by 
  sorry

end number_of_girls_l104_104082


namespace minimum_selling_price_l104_104389

theorem minimum_selling_price (total_cost : ℝ) (total_fruit : ℝ) (spoilage : ℝ) (min_price : ℝ) :
  total_cost = 760 ∧ total_fruit = 80 ∧ spoilage = 0.05 ∧ min_price = 10 → 
  ∀ price : ℝ, (price * total_fruit * (1 - spoilage) >= total_cost) → price >= min_price :=
by
  intros h price hp
  rcases h with ⟨hc, hf, hs, hm⟩
  sorry

end minimum_selling_price_l104_104389


namespace calc_problem1_calc_problem2_l104_104719

-- Proof Problem 1
theorem calc_problem1 : 
  (Real.sqrt 3 + 2 * Real.sqrt 2) - (3 * Real.sqrt 3 + Real.sqrt 2) = -2 * Real.sqrt 3 + Real.sqrt 2 := 
by 
  sorry

-- Proof Problem 2
theorem calc_problem2 : 
  Real.sqrt 2 * (Real.sqrt 2 + 1 / Real.sqrt 2) - abs (2 - Real.sqrt 6) = 5 - Real.sqrt 6 := 
by 
  sorry

end calc_problem1_calc_problem2_l104_104719


namespace superior_sequences_count_l104_104954

noncomputable def number_of_superior_sequences (n : ℕ) : ℕ :=
  Nat.choose (2 * n + 1) (n + 1) * 2^n

theorem superior_sequences_count (n : ℕ) (h : 2 ≤ n) 
  (x : Fin (n + 1) → ℤ)
  (h1 : ∀ i, 0 ≤ i ∧ i ≤ n → |x i| ≤ n)
  (h2 : ∀ i j, 0 ≤ i ∧ i < j ∧ j ≤ n → x i ≠ x j)
  (h3 : ∀ (i j k : Nat), 0 ≤ i ∧ i < j ∧ j < k ∧ k ≤ n → 
    max (|x k - x i|) (|x k - x j|) = 
    (|x i - x j| + |x j - x k| + |x k - x i|) / 2) :
  number_of_superior_sequences n = Nat.choose (2 * n + 1) (n + 1) * 2^n :=
sorry

end superior_sequences_count_l104_104954


namespace no_solution_inequality_l104_104302

theorem no_solution_inequality (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - |x + 1| + 2 * a ≥ 0) → a ≥ (Real.sqrt 3 + 1) / 4 :=
by
  intro h
  sorry

end no_solution_inequality_l104_104302


namespace court_cost_proof_l104_104105

-- Define all the given conditions
def base_fine : ℕ := 50
def penalty_rate : ℕ := 2
def mark_speed : ℕ := 75
def speed_limit : ℕ := 30
def school_zone_multiplier : ℕ := 2
def lawyer_fee_rate : ℕ := 80
def lawyer_hours : ℕ := 3
def total_owed : ℕ := 820

-- Define the calculation for the additional penalty
def additional_penalty : ℕ := (mark_speed - speed_limit) * penalty_rate

-- Define the calculation for the total fine
def total_fine : ℕ := (base_fine + additional_penalty) * school_zone_multiplier

-- Define the calculation for the lawyer's fee
def lawyer_fee : ℕ := lawyer_fee_rate * lawyer_hours

-- Define the calculation for the total of fine and lawyer's fee
def fine_and_lawyer_fee := total_fine + lawyer_fee

-- Prove the court costs
theorem court_cost_proof : total_owed - fine_and_lawyer_fee = 300 := by
  sorry

end court_cost_proof_l104_104105


namespace max_m_value_l104_104727

theorem max_m_value (a b m : ℝ) (ha : a > 0) (hb : b > 0) (H : (3/a + 1/b) ≥ m / (a + 3 * b)) : m ≤ 12 :=
sorry

end max_m_value_l104_104727


namespace min_a_minus_b_when_ab_eq_156_l104_104953

theorem min_a_minus_b_when_ab_eq_156 : ∃ a b : ℤ, (a * b = 156 ∧ a - b = -155) :=
by
  sorry

end min_a_minus_b_when_ab_eq_156_l104_104953


namespace scientific_notation_103M_l104_104916

theorem scientific_notation_103M : 103000000 = 1.03 * 10^8 := sorry

end scientific_notation_103M_l104_104916


namespace harvest_duration_l104_104218

theorem harvest_duration (total_earnings earnings_per_week : ℕ) (h1 : total_earnings = 1216) (h2 : earnings_per_week = 16) :
  total_earnings / earnings_per_week = 76 :=
by
  sorry

end harvest_duration_l104_104218


namespace symmetry_axis_is_2_range_of_a_l104_104712

-- Definitions given in the conditions
def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Condition 1: Constants a, b, c and a ≠ 0
variables (a b c : ℝ) (a_ne_zero : a ≠ 0)

-- Condition 2: Inequality constraint
axiom inequality_constraint : a^2 + 2 * a * c + c^2 < b^2

-- Condition 3: y-values are the same when x=t+2 and x=-t+2
axiom y_symmetry (t : ℝ) : quadratic_function a b c (t + 2) = quadratic_function a b c (-t + 2)

-- Question 1: Proving the symmetry axis is x=2
theorem symmetry_axis_is_2 : ∀ t : ℝ, (t + 2 + (-t + 2)) / 2 = 2 :=
by sorry

-- Question 2: Proving the range of a if y=2 when x=-2
theorem range_of_a (h : quadratic_function a b c (-2) = 2) (b_eq_neg4a : b = -4 * a) : 2 / 15 < a ∧ a < 2 / 7 :=
by sorry

end symmetry_axis_is_2_range_of_a_l104_104712


namespace exotic_meat_original_price_l104_104118

theorem exotic_meat_original_price (y : ℝ) :
  (0.75 * (y / 4) = 4.5) → y = 96 :=
by
  intro h
  sorry

end exotic_meat_original_price_l104_104118


namespace one_third_of_four_l104_104401

theorem one_third_of_four : (1/3) * 4 = 2 :=
by
  sorry

end one_third_of_four_l104_104401


namespace sandy_found_additional_money_l104_104834

-- Define the initial amount of money Sandy had
def initial_amount : ℝ := 13.99

-- Define the cost of the shirt
def shirt_cost : ℝ := 12.14

-- Define the cost of the jacket
def jacket_cost : ℝ := 9.28

-- Define the remaining amount after buying the shirt
def remaining_after_shirt : ℝ := initial_amount - shirt_cost

-- Define the additional money found in Sandy's pocket
def additional_found_money : ℝ := jacket_cost - remaining_after_shirt

-- State the theorem to prove the amount of additional money found
theorem sandy_found_additional_money :
  additional_found_money = 11.13 :=
by sorry

end sandy_found_additional_money_l104_104834


namespace marble_probability_is_correct_l104_104391

def marbles_probability
  (total_marbles: ℕ) 
  (red_marbles: ℕ) 
  (blue_marbles: ℕ) 
  (green_marbles: ℕ)
  (choose_marbles: ℕ) 
  (required_red: ℕ) 
  (required_blue: ℕ) 
  (required_green: ℕ): ℚ := sorry

-- Define conditions
def total_marbles := 7
def red_marbles := 3
def blue_marbles := 2
def green_marbles := 2
def choose_marbles := 4
def required_red := 2
def required_blue := 1
def required_green := 1

-- Proof statement
theorem marble_probability_is_correct : 
  marbles_probability total_marbles red_marbles blue_marbles green_marbles choose_marbles required_red required_blue required_green = (12 / 35 : ℚ) :=
sorry

end marble_probability_is_correct_l104_104391


namespace calculate_cost_price_l104_104145

/-
Given:
  SP (Selling Price) is 18000
  If a 10% discount is applied on the SP, the effective selling price becomes 16200
  This effective selling price corresponds to an 8% profit over the cost price
  
Prove:
  The cost price (CP) is 15000
-/

theorem calculate_cost_price (SP : ℝ) (d : ℝ) (p : ℝ) (effective_SP : ℝ) (CP : ℝ) :
  SP = 18000 →
  d = 0.1 →
  p = 0.08 →
  effective_SP = SP - (d * SP) →
  effective_SP = CP * (1 + p) →
  CP = 15000 :=
by
  intros _
  sorry

end calculate_cost_price_l104_104145


namespace ratio_B_A_l104_104699

theorem ratio_B_A (A B : ℤ) (h : ∀ (x : ℝ), x ≠ -6 → x ≠ 0 → x ≠ 5 → 
  (A / (x + 6) + B / (x^2 - 5*x) = (x^3 - 3*x^2 + 12) / (x^3 + x^2 - 30*x))) :
  (B : ℚ) / A = 2.2 := by
  sorry

end ratio_B_A_l104_104699


namespace conversion1_conversion2_conversion3_minutes_conversion3_seconds_conversion4_l104_104052

theorem conversion1 : 4 * 60 + 35 = 275 := by
  sorry

theorem conversion2 : 4 * 1000 + 35 = 4035 := by
  sorry

theorem conversion3_minutes : 678 / 60 = 11 := by
  sorry

theorem conversion3_seconds : 678 % 60 = 18 := by
  sorry

theorem conversion4 : 120000 / 10000 = 12 := by
  sorry

end conversion1_conversion2_conversion3_minutes_conversion3_seconds_conversion4_l104_104052


namespace total_goals_l104_104910

theorem total_goals (B M : ℕ) (hB : B = 4) (hM : M = 3 * B) : B + M = 16 := by
  sorry

end total_goals_l104_104910


namespace remainder_when_divided_by_x_plus_2_l104_104120

def q (x D E F : ℝ) : ℝ := D*x^4 + E*x^2 + F*x - 2

theorem remainder_when_divided_by_x_plus_2 (D E F : ℝ) (h : q 2 D E F = 14) : q (-2) D E F = -18 := 
by 
     sorry

end remainder_when_divided_by_x_plus_2_l104_104120


namespace cube_volume_l104_104181

-- Define the surface area constant
def surface_area : ℝ := 725.9999999999998

-- Define the formula for surface area of a cube and solve for volume given the conditions
theorem cube_volume (SA : ℝ) (h : SA = surface_area) : 11^3 = 1331 :=
by sorry

end cube_volume_l104_104181


namespace simplify_complex_subtraction_l104_104895

-- Definition of the nested expression
def complex_subtraction (x : ℝ) : ℝ :=
  1 - (2 - (3 - (4 - (5 - (6 - x)))))

-- Statement of the theorem to be proven
theorem simplify_complex_subtraction (x : ℝ) : complex_subtraction x = x - 3 :=
by {
  -- This proof will need to be filled in to verify the statement
  sorry
}

end simplify_complex_subtraction_l104_104895


namespace osmotic_pressure_independence_l104_104833

-- definitions for conditions
def osmotic_pressure_depends_on (osmotic_pressure protein_content Na_content Cl_content : Prop) : Prop :=
  (osmotic_pressure = protein_content ∧ osmotic_pressure = Na_content ∧ osmotic_pressure = Cl_content)

-- statement of the problem to be proved
theorem osmotic_pressure_independence 
  (osmotic_pressure : Prop) 
  (protein_content : Prop) 
  (Na_content : Prop) 
  (Cl_content : Prop) 
  (mw_plasma_protein : Prop)
  (dependence : osmotic_pressure_depends_on osmotic_pressure protein_content Na_content Cl_content) :
  ¬(osmotic_pressure = mw_plasma_protein) :=
sorry

end osmotic_pressure_independence_l104_104833


namespace algebraic_expression_value_l104_104516

theorem algebraic_expression_value (x y : ℝ) (h : x^4 + 6*x^2*y + 9*y^2 + 2*x^2 + 6*y + 4 = 7) :
(x^4 + 6*x^2*y + 9*y^2 - 2*x^2 - 6*y - 1 = -2) ∨ (x^4 + 6*x^2*y + 9*y^2 - 2*x^2 - 6*y - 1 = 14) :=
sorry

end algebraic_expression_value_l104_104516


namespace find_a_l104_104778

theorem find_a {S : ℕ → ℤ} (a : ℤ)
  (hS : ∀ n : ℕ, S n = 5 ^ (n + 1) + a) : a = -5 :=
sorry

end find_a_l104_104778


namespace integer_solution_interval_l104_104140

theorem integer_solution_interval {f : ℝ → ℝ} (m : ℝ) :
  (∀ x : ℤ, (-x^2 + x + m + 2 ≥ |x| ↔ (x : ℝ) = n)) ↔ (-2 ≤ m ∧ m < -1) := 
sorry

end integer_solution_interval_l104_104140


namespace factorize_difference_of_squares_l104_104881

theorem factorize_difference_of_squares (a : ℝ) : a^2 - 4 = (a + 2) * (a - 2) :=
by
  sorry

end factorize_difference_of_squares_l104_104881


namespace time_difference_l104_104429

/-
Malcolm's speed: 5 minutes per mile
Joshua's speed: 7 minutes per mile
Race length: 12 miles
Question: Prove that the time difference between Joshua crossing the finish line after Malcolm is 24 minutes
-/
noncomputable def time_taken (speed: ℕ) (distance: ℕ) : ℕ :=
  speed * distance

theorem time_difference :
  let malcolm_speed := 5
  let joshua_speed := 7
  let race_length := 12
  let malcolm_time := time_taken malcolm_speed race_length
  let joshua_time := time_taken joshua_speed race_length
  malcolm_time < joshua_time →
  joshua_time - malcolm_time = 24 :=
by
  intros malcolm_speed joshua_speed race_length malcolm_time joshua_time malcolm_time_lt_joshua_time
  sorry

end time_difference_l104_104429


namespace min_unit_cubes_intersect_all_l104_104932

theorem min_unit_cubes_intersect_all (n : ℕ) : 
  let A_n := if n % 2 = 0 then n^2 / 2 else (n^2 + 1) / 2
  A_n = if n % 2 = 0 then n^2 / 2 else (n^2 + 1) / 2 :=
sorry

end min_unit_cubes_intersect_all_l104_104932


namespace proof_expr_is_neg_four_ninths_l104_104370

noncomputable def example_expr : ℚ := (-3 / 2) ^ 2021 * (2 / 3) ^ 2023

theorem proof_expr_is_neg_four_ninths : example_expr = (-4 / 9) := 
by 
  -- Here the proof would be placed
  sorry

end proof_expr_is_neg_four_ninths_l104_104370


namespace votes_difference_l104_104012

theorem votes_difference (T : ℕ) (V_a : ℕ) (V_f : ℕ) 
  (h1 : T = 330) (h2 : V_a = 40 * T / 100) (h3 : V_f = T - V_a) : V_f - V_a = 66 :=
by
  sorry

end votes_difference_l104_104012


namespace investment_period_l104_104656

theorem investment_period (P : ℝ) (r1 r2 : ℝ) (diff : ℝ) (t : ℝ) :
  P = 900 ∧ r1 = 0.04 ∧ r2 = 0.045 ∧ (P * r2 * t) - (P * r1 * t) = 31.50 → t = 7 :=
by
  sorry

end investment_period_l104_104656


namespace linda_babysitting_hours_l104_104085

-- Define constants
def hourly_wage : ℝ := 10.0
def application_fee : ℝ := 25.0
def number_of_colleges : ℝ := 6.0

-- Theorem statement
theorem linda_babysitting_hours : 
    (application_fee * number_of_colleges) / hourly_wage = 15 := 
by
  -- Here the proof would go, but we'll use sorry as per instructions
  sorry

end linda_babysitting_hours_l104_104085


namespace coin_combinations_l104_104744

-- Define the coins and their counts
def one_cent_count := 1
def two_cent_count := 1
def five_cent_count := 1
def ten_cent_count := 4
def fifty_cent_count := 2

-- Define the expected number of different possible amounts
def expected_amounts := 119

-- Prove that the expected number of possible amounts can be achieved given the coins
theorem coin_combinations : 
  (∃ sums : Finset ℕ, 
    sums.card = expected_amounts ∧ 
    (∀ n ∈ sums, n = one_cent_count * 1 + 
                          two_cent_count * 2 + 
                          five_cent_count * 5 + 
                          ten_cent_count * 10 + 
                          fifty_cent_count * 50)) :=
sorry

end coin_combinations_l104_104744


namespace factors_2310_l104_104978

theorem factors_2310 : ∃ (S : Finset ℕ), (∀ p ∈ S, Nat.Prime p) ∧ S.card = 5 ∧ (2310 = S.prod id) :=
by
  sorry

end factors_2310_l104_104978


namespace total_marbles_l104_104379

theorem total_marbles (y b g : ℝ) (h1 : y = 1.4 * b) (h2 : g = 1.75 * y) :
  b + y + g = 3.4643 * y :=
sorry

end total_marbles_l104_104379


namespace correct_answer_is_C_l104_104124

def exactly_hits_n_times (n k : ℕ) : Prop :=
  n = k

def hits_no_more_than (n k : ℕ) : Prop :=
  n ≤ k

def hits_at_least (n k : ℕ) : Prop :=
  n ≥ k

def is_mutually_exclusive (P Q : Prop) : Prop :=
  ¬ (P ∧ Q)

def is_non_opposing (P Q : Prop) : Prop :=
  ¬ P ∧ ¬ Q

def events_are_mutually_exclusive_and_non_opposing (n : ℕ) : Prop :=
  let event1 := exactly_hits_n_times 5 3
  let event2 := exactly_hits_n_times 5 4
  is_mutually_exclusive event1 event2 ∧ is_non_opposing event1 event2

theorem correct_answer_is_C : events_are_mutually_exclusive_and_non_opposing 5 :=
by
  sorry

end correct_answer_is_C_l104_104124


namespace library_books_difference_l104_104604

theorem library_books_difference (total_books : ℕ) (borrowed_percentage : ℕ) 
  (initial_books : total_books = 400) 
  (percentage_borrowed : borrowed_percentage = 30) :
  (total_books - (borrowed_percentage * total_books / 100)) = 280 :=
by
  sorry

end library_books_difference_l104_104604


namespace speed_in_still_water_l104_104758

variable (v_m v_s : ℝ)

def swims_downstream (v_m v_s : ℝ) : Prop :=
  54 = (v_m + v_s) * 3

def swims_upstream (v_m v_s : ℝ) : Prop :=
  18 = (v_m - v_s) * 3

theorem speed_in_still_water : swims_downstream v_m v_s ∧ swims_upstream v_m v_s → v_m = 12 :=
by
  sorry

end speed_in_still_water_l104_104758


namespace correct_statement_a_incorrect_statement_b_incorrect_statement_c_incorrect_statement_d_incorrect_statement_e_l104_104110

theorem correct_statement_a (x : ℝ) : x > 1 → x^2 > x :=
by sorry

theorem incorrect_statement_b (x : ℝ) : ¬ (x^2 < 0 → x < 0) :=
by sorry

theorem incorrect_statement_c (x : ℝ) : ¬ (x^2 < x → x < 0) :=
by sorry

theorem incorrect_statement_d (x : ℝ) : ¬ (x^2 < 1 → x < 1) :=
by sorry

theorem incorrect_statement_e (x : ℝ) : ¬ (x > 0 → x^2 > x) :=
by sorry

end correct_statement_a_incorrect_statement_b_incorrect_statement_c_incorrect_statement_d_incorrect_statement_e_l104_104110


namespace pizzas_served_during_lunch_l104_104780

def total_pizzas : ℕ := 15
def dinner_pizzas : ℕ := 6

theorem pizzas_served_during_lunch :
  ∃ lunch_pizzas : ℕ, lunch_pizzas = total_pizzas - dinner_pizzas :=
by
  use 9
  exact rfl

end pizzas_served_during_lunch_l104_104780


namespace petals_per_ounce_l104_104845

-- Definitions of the given conditions
def petals_per_rose : ℕ := 8
def roses_per_bush : ℕ := 12
def bushes_harvested : ℕ := 800
def bottles_produced : ℕ := 20
def ounces_per_bottle : ℕ := 12

-- Calculation of petals per bush
def petals_per_bush : ℕ := roses_per_bush * petals_per_rose

-- Calculation of total petals harvested
def total_petals_harvested : ℕ := bushes_harvested * petals_per_bush

-- Calculation of total ounces of perfume
def total_ounces_produced : ℕ := bottles_produced * ounces_per_bottle

-- Main theorem statement
theorem petals_per_ounce : total_petals_harvested / total_ounces_produced = 320 :=
by
  sorry

end petals_per_ounce_l104_104845


namespace gcd_735_1287_l104_104336

theorem gcd_735_1287 : Int.gcd 735 1287 = 3 := by
  sorry

end gcd_735_1287_l104_104336


namespace louisa_average_speed_l104_104009

theorem louisa_average_speed :
  ∃ v : ℝ, 
  (100 / v = 175 / v - 3) ∧ 
  v = 25 :=
by
  sorry

end louisa_average_speed_l104_104009


namespace inequality_solution_b_range_l104_104996

-- Given conditions
variables (a b : ℝ)

def condition1 : Prop := (1 - a < 0) ∧ (a = 3)
def condition2 : Prop := ∀ (x : ℝ), (3 * x^2 + b * x + 3) ≥ 0

-- Assertions to be proved
theorem inequality_solution (a : ℝ) (ha : condition1 a) : 
  ∀ (x : ℝ), (2 * x^2 + (2 - a) * x - a > 0) ↔ (x < -1 ∨ x > 3/2) :=
sorry

theorem b_range (a : ℝ) (hb : condition1 a) : 
  condition2 b ↔ (-6 ≤ b ∧ b ≤ 6) :=
sorry

end inequality_solution_b_range_l104_104996


namespace ratio_boys_girls_l104_104830

theorem ratio_boys_girls
  (B G : ℕ)  -- Number of boys and girls
  (h_ratio : 75 * G = 80 * B)
  (h_total_no_scholarship : 100 * (3 * B + 4 * G) = 7772727272727272 * (B + G)) :
  B = 5 * G := sorry

end ratio_boys_girls_l104_104830


namespace bird_count_l104_104232

theorem bird_count (num_cages : ℕ) (parrots_per_cage : ℕ) (parakeets_per_cage : ℕ) 
  (total_birds : ℕ) (h1 : num_cages = 8) (h2 : parrots_per_cage = 2) (h3 : parakeets_per_cage = 7) 
  (h4 : total_birds = num_cages * (parrots_per_cage + parakeets_per_cage)) : 
  total_birds = 72 := 
  by
  sorry

end bird_count_l104_104232


namespace find_f_2010_l104_104972

noncomputable def f (a b α β : ℝ) (x : ℝ) :=
  a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β) + 4

theorem find_f_2010 {a b α β : ℝ} (h : f a b α β 2009 = 5) : f a b α β 2010 = 3 :=
sorry

end find_f_2010_l104_104972


namespace bicycle_final_price_l104_104777

-- Define initial conditions
def original_price : ℝ := 200
def wednesday_discount : ℝ := 0.40
def friday_increase : ℝ := 0.20
def saturday_discount : ℝ := 0.25

-- Statement to prove that the final price, after all discounts and increases, is $108
theorem bicycle_final_price :
  (original_price * (1 - wednesday_discount) * (1 + friday_increase) * (1 - saturday_discount)) = 108 := by
  sorry

end bicycle_final_price_l104_104777


namespace difference_is_24_l104_104383

namespace BuffaloesAndDucks

def numLegs (B D : ℕ) : ℕ := 4 * B + 2 * D

def numHeads (B D : ℕ) : ℕ := B + D

def diffLegsAndHeads (B D : ℕ) : ℕ := numLegs B D - 2 * numHeads B D

theorem difference_is_24 (D : ℕ) : diffLegsAndHeads 12 D = 24 := by
  sorry

end BuffaloesAndDucks

end difference_is_24_l104_104383


namespace arithmetic_seq_geom_eq_div_l104_104056

noncomputable def a (n : ℕ) (a1 d : ℝ) : ℝ := a1 + n * d

theorem arithmetic_seq_geom_eq_div (a1 d : ℝ) (h1 : d ≠ 0) (h2 : a1 ≠ 0) 
    (h_geom : (a 3 a1 d) ^ 2 = (a 1 a1 d) * (a 7 a1 d)) :
    (a 2 a1 d + a 5 a1 d + a 8 a1 d) / (a 3 a1 d + a 4 a1 d) = 2 := 
by
  sorry

end arithmetic_seq_geom_eq_div_l104_104056


namespace simplify_expression_l104_104242

theorem simplify_expression : 
  (Real.sqrt 2 * 2^(1/2) * 2) + (18 / 3 * 2) - (8^(1/2) * 4) = 16 - 8 * Real.sqrt 2 :=
by 
  sorry  -- proof omitted

end simplify_expression_l104_104242


namespace ellipse_eccentricity_l104_104846

theorem ellipse_eccentricity (a : ℝ) (h : a > 0) 
  (ell_eq : ∀ x y : ℝ, x^2 / a^2 + y^2 / 5 = 1 ↔ x^2 / a^2 + y^2 / 5 = 1)
  (ecc_eq : (eccentricity : ℝ) = 2 / 3) : 
  a = 3 := 
sorry

end ellipse_eccentricity_l104_104846


namespace find_function_value_at_2_l104_104943

variables {f : ℕ → ℕ}

theorem find_function_value_at_2 (H : ∀ x : ℕ, Nat.succ (Nat.succ x * Nat.succ x + f x) = 12) : f 2 = 4 :=
by
  sorry

end find_function_value_at_2_l104_104943


namespace comprehensive_score_l104_104410

theorem comprehensive_score :
  let w_c := 0.4
  let w_u := 0.6
  let s_c := 80
  let s_u := 90
  s_c * w_c + s_u * w_u = 86 :=
by
  sorry

end comprehensive_score_l104_104410


namespace income_to_expenditure_ratio_l104_104102

theorem income_to_expenditure_ratio (I E S : ℝ) (hI : I = 10000) (hS : S = 4000) (hSavings : S = I - E) : I / E = 5 / 3 := by
  -- To prove: I / E = 5 / 3 given hI, hS, and hSavings
  sorry

end income_to_expenditure_ratio_l104_104102


namespace arrasta_um_proof_l104_104929

variable (n : ℕ)

def arrasta_um_possible_moves (n : ℕ) : ℕ :=
  6 * n - 8

theorem arrasta_um_proof (n : ℕ) (h : n ≥ 2) : arrasta_um_possible_moves n =
6 * n - 8 := by
  sorry

end arrasta_um_proof_l104_104929


namespace axis_of_symmetry_parabola_l104_104760

theorem axis_of_symmetry_parabola : 
  ∀ (x : ℝ), 2 * (x - 3)^2 - 5 = 2 * (x - 3)^2 - 5 → (∃ h : ℝ, h = 3 ∧ ∀ x : ℝ, h = 3) :=
by
  sorry

end axis_of_symmetry_parabola_l104_104760


namespace zoo_animals_total_l104_104821

-- Conditions as definitions
def initial_animals : ℕ := 68
def gorillas_sent_away : ℕ := 6
def hippopotamus_adopted : ℕ := 1
def rhinos_taken_in : ℕ := 3
def lion_cubs_born : ℕ := 8
def meerkats_per_cub : ℕ := 2

-- Theorem to prove the resulting number of animals
theorem zoo_animals_total :
  (initial_animals - gorillas_sent_away + hippopotamus_adopted + rhinos_taken_in + lion_cubs_born + meerkats_per_cub * lion_cubs_born) = 90 :=
by 
  sorry

end zoo_animals_total_l104_104821


namespace range_of_f_l104_104807

noncomputable def f (x : ℝ) : ℝ := (1/2) ^ |x + 1|

theorem range_of_f : Set.Ioo 0 1 ∪ {1} = {y : ℝ | ∃ x : ℝ, f x = y} :=
by
  sorry

end range_of_f_l104_104807


namespace max_value_exp_l104_104388

theorem max_value_exp (x y z : ℝ) (h_nonneg : 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z) (h_constraint : x^2 + y^2 + z^2 = 1) :
  2 * x * y * Real.sqrt 8 + 7 * y * z + 5 * x * z ≤ 23.0219 :=
sorry

end max_value_exp_l104_104388


namespace golden_section_point_l104_104195

noncomputable def golden_ratio : ℝ := (1 + Real.sqrt 5) / 2

theorem golden_section_point (AB AP PB : ℝ)
  (h1 : AP + PB = AB)
  (h2 : AB = 5)
  (h3 : (AB / AP) = (AP / PB))
  (h4 : AP > PB) :
  AP = (5 * Real.sqrt 5 - 5) / 2 :=
by sorry

end golden_section_point_l104_104195


namespace min_contribution_proof_l104_104098

noncomputable def min_contribution (total_contribution : ℕ) (num_people : ℕ) (max_contribution: ℕ) :=
  ∃ (min_each_person: ℕ), num_people * min_each_person ≤ total_contribution ∧ max_contribution * (num_people - 1) + min_each_person ≥ total_contribution ∧ min_each_person = 2

theorem min_contribution_proof :
  min_contribution 30 15 16 :=
sorry

end min_contribution_proof_l104_104098


namespace range_of_a_l104_104326

theorem range_of_a (a : ℝ) : (∀ x : ℝ, a * x^2 + 2 * a * x - (a + 2) < 0) ↔ (-1 < a ∧ a ≤ 0) :=
by
  sorry

end range_of_a_l104_104326


namespace min_value_of_A_div_B_l104_104359

noncomputable def A (g1 : Finset ℕ) : ℕ :=
  g1.prod id

noncomputable def B (g2 : Finset ℕ) : ℕ :=
  g2.prod id

theorem min_value_of_A_div_B : ∃ (g1 g2 : Finset ℕ), 
  g1 ∪ g2 = (Finset.range 31).erase 0 ∧ g1 ∩ g2 = ∅ ∧ A g1 % B g2 = 0 ∧ A g1 / B g2 = 1077205 :=
by
  sorry

end min_value_of_A_div_B_l104_104359


namespace quadratic_solution_l104_104774

theorem quadratic_solution (m n x : ℝ)
  (h1 : (x - m)^2 + n = 0) 
  (h2 : ∃ (a b : ℝ), a ≠ b ∧ (x = a ∨ x = b) ∧ (a - m)^2 + n = 0 ∧ (b - m)^2 + n = 0
    ∧ (a = -1 ∨ a = 3) ∧ (b = -1 ∨ b = 3)) :
  x = -3 ∨ x = 1 :=
by {
  sorry
}

end quadratic_solution_l104_104774


namespace moe_pie_share_l104_104890

theorem moe_pie_share
  (leftover_pie : ℚ)
  (num_people : ℕ)
  (H_leftover : leftover_pie = 5 / 8)
  (H_people : num_people = 4) :
  (leftover_pie / num_people = 5 / 32) :=
by
  sorry

end moe_pie_share_l104_104890


namespace compound_interest_rate_l104_104704

theorem compound_interest_rate (P A : ℝ) (t n : ℝ)
  (hP : P = 5000) 
  (hA : A = 7850)
  (ht : t = 8)
  (hn : n = 1) : 
  ∃ r : ℝ, 0.057373 ≤ (r * 100) ∧ (r * 100) ≤ 5.7373 :=
by
  sorry

end compound_interest_rate_l104_104704


namespace miss_hilt_apples_l104_104980

theorem miss_hilt_apples (h : ℕ) (a_per_hour : ℕ) (total_apples : ℕ) 
    (H1 : a_per_hour = 5) (H2 : total_apples = 15) (H3 : total_apples = h * a_per_hour) : 
  h = 3 :=
by
  sorry

end miss_hilt_apples_l104_104980


namespace roots_n_not_divisible_by_5_for_any_n_l104_104864

theorem roots_n_not_divisible_by_5_for_any_n (x1 x2 : ℝ) (n : ℕ)
  (hx : x1^2 - 6 * x1 + 1 = 0)
  (hy : x2^2 - 6 * x2 + 1 = 0)
  : ¬(∃ (k : ℕ), (x1^k + x2^k) % 5 = 0) :=
sorry

end roots_n_not_divisible_by_5_for_any_n_l104_104864


namespace solve_system1_solve_system2_l104_104361

-- Definitions for the first system of equations
def system1_equation1 (x y : ℚ) := 3 * x - 6 * y = 4
def system1_equation2 (x y : ℚ) := x + 5 * y = 6

-- Definitions for the second system of equations
def system2_equation1 (x y : ℚ) := x / 4 + y / 3 = 3
def system2_equation2 (x y : ℚ) := 3 * (x - 4) - 2 * (y - 1) = -1

-- Lean statement for proving the solution to the first system
theorem solve_system1 :
  ∃ (x y : ℚ), system1_equation1 x y ∧ system1_equation2 x y ∧ x = 8 / 3 ∧ y = 2 / 3 :=
by
  sorry

-- Lean statement for proving the solution to the second system
theorem solve_system2 :
  ∃ (x y : ℚ), system2_equation1 x y ∧ system2_equation2 x y ∧ x = 6 ∧ y = 9 / 2 :=
by
  sorry

end solve_system1_solve_system2_l104_104361


namespace shortest_travel_time_to_sunny_town_l104_104501

-- Definitions based on the given conditions
def highway_length : ℕ := 12

def railway_crossing_closed (t : ℕ) : Prop :=
  ∃ k : ℕ, t = 6 * k + 0 ∨ t = 6 * k + 1 ∨ t = 6 * k + 2

def traffic_light_red (t : ℕ) : Prop :=
  ∃ k1 : ℕ, t = 5 * k1 + 0 ∨ t = 5 * k1 + 1

def initial_conditions (t : ℕ) : Prop :=
  railway_crossing_closed 0 ∧ traffic_light_red 0

def shortest_time_to_sunny_town (time : ℕ) : Prop := 
  time = 24

-- The proof statement
theorem shortest_travel_time_to_sunny_town :
  ∃ time : ℕ, shortest_time_to_sunny_town time ∧
  (∀ t : ℕ, 0 ≤ t → t ≤ time → ¬railway_crossing_closed t ∧ ¬traffic_light_red t) :=
sorry

end shortest_travel_time_to_sunny_town_l104_104501


namespace min_colors_shapes_l104_104275

def representable_centers (C S : Nat) : Nat :=
  C + (C * (C - 1)) / 2 + S + S * (S - 1)

theorem min_colors_shapes (C S : Nat) :
  ∀ (C S : Nat), (C + (C * (C - 1)) / 2 + S + S * (S - 1)) ≥ 12 → (C, S) = (3, 3) :=
sorry

end min_colors_shapes_l104_104275


namespace impossible_to_use_up_all_parts_l104_104368

theorem impossible_to_use_up_all_parts (p q r : ℕ) :
  (∃ p q r : ℕ,
    2 * p + 2 * r + 2 = A ∧
    2 * p + q + 1 = B ∧
    q + r = C) → false :=
by {
  sorry
}

end impossible_to_use_up_all_parts_l104_104368


namespace spider_total_distance_l104_104644

theorem spider_total_distance :
  let start := 3
  let mid := -4
  let final := 8
  let dist1 := abs (mid - start)
  let dist2 := abs (final - mid)
  let total_distance := dist1 + dist2
  total_distance = 19 :=
by
  sorry

end spider_total_distance_l104_104644


namespace remaining_batches_l104_104104

def flour_per_batch : ℕ := 2
def batches_baked : ℕ := 3
def initial_flour : ℕ := 20

theorem remaining_batches : (initial_flour - flour_per_batch * batches_baked) / flour_per_batch = 7 := by
  sorry

end remaining_batches_l104_104104


namespace multiple_of_960_l104_104488

theorem multiple_of_960 (a : ℤ) (h1 : a % 10 = 4) (h2 : ¬ (a % 4 = 0)) :
  ∃ k : ℤ, a * (a^2 - 1) * (a^2 - 4) = 960 * k :=
  sorry

end multiple_of_960_l104_104488


namespace problem_part1_problem_part2_l104_104416

variable (A B : Set ℝ)
def C_R (s : Set ℝ) : Set ℝ := { x | x ∉ s }

theorem problem_part1 :
  A = { x : ℝ | 3 ≤ x ∧ x < 6 } →
  B = { x : ℝ | 2 < x ∧ x < 9 } →
  C_R (A ∩ B) = { x : ℝ | x < 3 ∨ x ≥ 6 } :=
by
  intros hA hB
  sorry

theorem problem_part2 :
  A = { x : ℝ | 3 ≤ x ∧ x < 6 } →
  B = { x : ℝ | 2 < x ∧ x < 9 } →
  (C_R B) ∪ A = { x : ℝ | x ≤ 2 ∨ (3 ≤ x ∧ x < 6) ∨ x ≥ 9 } :=
by
  intros hA hB
  sorry

end problem_part1_problem_part2_l104_104416


namespace functional_eq_linear_l104_104723

theorem functional_eq_linear {f : ℝ → ℝ} (h : ∀ x y : ℝ, f (x ^ 2 - y ^ 2) = (x + y) * (f x - f y)) :
  ∃ c : ℝ, ∀ x : ℝ, f x = c * x :=
sorry

end functional_eq_linear_l104_104723


namespace probability_of_F_l104_104072

theorem probability_of_F (P : String → ℚ) (hD : P "D" = 1/4) (hE : P "E" = 1/3) (hG : P "G" = 1/6) (total : P "D" + P "E" + P "F" + P "G" = 1) :
  P "F" = 1/4 :=
by
  sorry

end probability_of_F_l104_104072


namespace find_x2_plus_y2_l104_104820

theorem find_x2_plus_y2 
  (x y : ℕ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (h1 : x * y + x + y = 117) 
  (h2 : x^2 * y + x * y^2 = 1512) : 
  x^2 + y^2 = 549 := 
sorry

end find_x2_plus_y2_l104_104820


namespace original_recipe_serves_7_l104_104500

theorem original_recipe_serves_7 (x : ℕ)
  (h1 : 2 / x = 10 / 35) :
  x = 7 := by
  sorry

end original_recipe_serves_7_l104_104500


namespace max_intersections_l104_104329

theorem max_intersections (n1 n2 k : ℕ) 
  (h1 : n1 ≤ n2)
  (h2 : k ≤ n1) : 
  ∃ max_intersections : ℕ, 
  max_intersections = k * n2 :=
by
  sorry

end max_intersections_l104_104329


namespace sum_of_roots_l104_104531

def quadratic_polynomial (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + b * x + c

def condition (a b c : ℝ) (x : ℝ) :=
  quadratic_polynomial a b c (x^3 + x) ≥ quadratic_polynomial a b c (x^2 + 1)

theorem sum_of_roots (a b c : ℝ) (h : ∀ x : ℝ, condition a b c x) :
  b = -4 * a → -(b / a) = 4 :=
by
  sorry

end sum_of_roots_l104_104531


namespace gift_arrangement_l104_104519

theorem gift_arrangement (n k : ℕ) (h_n : n = 5) (h_k : k = 4) : 
  (n * Nat.factorial k) = 120 :=
by
  sorry

end gift_arrangement_l104_104519


namespace total_friends_met_l104_104151

def num_friends_with_pears : Nat := 9
def num_friends_with_oranges : Nat := 6

theorem total_friends_met : num_friends_with_pears + num_friends_with_oranges = 15 :=
by
  sorry

end total_friends_met_l104_104151


namespace probability_not_miss_is_correct_l104_104028

-- Define the probability that Peter will miss his morning train
def p_miss : ℚ := 5 / 12

-- Define the probability that Peter does not miss his morning train
def p_not_miss : ℚ := 1 - p_miss

-- The theorem to prove
theorem probability_not_miss_is_correct : p_not_miss = 7 / 12 :=
by
  -- Proof omitted
  sorry

end probability_not_miss_is_correct_l104_104028


namespace Brandy_energy_drinks_l104_104123

theorem Brandy_energy_drinks 
  (maximum_safe_amount : ℕ)
  (caffeine_per_drink : ℕ)
  (extra_safe_caffeine : ℕ)
  (x : ℕ)
  (h1 : maximum_safe_amount = 500)
  (h2 : caffeine_per_drink = 120)
  (h3 : extra_safe_caffeine = 20)
  (h4 : caffeine_per_drink * x + extra_safe_caffeine = maximum_safe_amount) :
  x = 4 :=
by
  sorry

end Brandy_energy_drinks_l104_104123


namespace largest_number_value_l104_104233

theorem largest_number_value 
  (a b c : ℚ)
  (h_sum : a + b + c = 100)
  (h_diff1 : c - b = 10)
  (h_diff2 : b - a = 5) : 
  c = 125 / 3 := 
sorry

end largest_number_value_l104_104233


namespace mechanic_earns_on_fourth_day_l104_104307

theorem mechanic_earns_on_fourth_day 
  (E1 E2 E3 E4 E5 E6 E7 : ℝ)
  (h1 : (E1 + E2 + E3 + E4) / 4 = 18)
  (h2 : (E4 + E5 + E6 + E7) / 4 = 22)
  (h3 : (E1 + E2 + E3 + E4 + E5 + E6 + E7) / 7 = 21) 
  : E4 = 13 := 
by 
  sorry

end mechanic_earns_on_fourth_day_l104_104307


namespace ratio_of_constants_l104_104191

theorem ratio_of_constants (a b c: ℝ) (h1 : 8 = 0.02 * a) (h2 : 2 = 0.08 * b) (h3 : c = b / a) : c = 1 / 16 :=
by sorry

end ratio_of_constants_l104_104191


namespace largest_odd_digit_multiple_of_11_l104_104491

theorem largest_odd_digit_multiple_of_11 (n : ℕ) (h1 : n < 10000) (h2 : ∀ d ∈ (n.digits 10), d % 2 = 1) (h3 : 11 ∣ n) : n ≤ 9559 :=
sorry

end largest_odd_digit_multiple_of_11_l104_104491


namespace solve_for_y_l104_104111

/-- Given the equation 7(2y + 3) - 5 = -3(2 - 5y), solve for y. -/
theorem solve_for_y (y : ℤ) : 7 * (2 * y + 3) - 5 = -3 * (2 - 5 * y) → y = 22 :=
by
  intros h
  sorry

end solve_for_y_l104_104111


namespace cone_from_sector_l104_104327

theorem cone_from_sector 
  (sector_angle : ℝ) (sector_radius : ℝ)
  (circumference : ℝ := (sector_angle / 360) * (2 * Real.pi * sector_radius))
  (base_radius : ℝ := circumference / (2 * Real.pi))
  (slant_height : ℝ := sector_radius) :
  sector_angle = 270 ∧ sector_radius = 12 → base_radius = 9 ∧ slant_height = 12 :=
by
  sorry

end cone_from_sector_l104_104327


namespace cheese_pizzas_l104_104407

theorem cheese_pizzas (p b c total : ℕ) (h1 : p = 2) (h2 : b = 6) (h3 : total = 14) (ht : p + b + c = total) : c = 6 := 
by
  sorry

end cheese_pizzas_l104_104407


namespace smallest_n_terminating_decimal_l104_104229

theorem smallest_n_terminating_decimal :
  ∃ n : ℕ, (∀ m : ℕ, (∃ k : ℕ, m + 150 = 2^k * (5^0))
  → (m ≥ n)) ∧ (n > 0) ∧ (∃ k : ℕ, n + 150 = 2^k * (5^0)) := 
sorry

end smallest_n_terminating_decimal_l104_104229


namespace ava_first_coupon_day_l104_104338

theorem ava_first_coupon_day (first_coupon_day : ℕ) (coupon_interval : ℕ) 
    (closed_day : ℕ) (days_in_week : ℕ):
  first_coupon_day = 2 →  -- starting on Tuesday (considering Monday as 1)
  coupon_interval = 13 →
  closed_day = 7 →        -- Saturday is represented by 7
  days_in_week = 7 →
  ∀ n : ℕ, ((first_coupon_day + n * coupon_interval) % days_in_week) ≠ closed_day :=
by 
  -- Proof can be filled here.
  sorry

end ava_first_coupon_day_l104_104338


namespace xiao_ming_english_score_l104_104068

theorem xiao_ming_english_score :
  let a := 92
  let b := 90
  let c := 95
  let w_a := 3
  let w_b := 3
  let w_c := 4
  let total_weight := (w_a + w_b + w_c)
  let score := (a * w_a + b * w_b + c * w_c) / total_weight
  score = 92.6 :=
by
  sorry

end xiao_ming_english_score_l104_104068


namespace power_function_increasing_l104_104635

theorem power_function_increasing (m : ℝ) : 
  (∀ x : ℝ, 0 < x → (m^2 - 2*m - 2) * x^(-4*m - 2) > 0) ↔ m = -1 :=
by sorry

end power_function_increasing_l104_104635


namespace arithmetic_sequence_sum_l104_104161

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (d : ℕ → ℕ)
  (is_arithmetic_seq : ∀ n, a (n + 1) = a n + d n)
  (h : (a 2) + (a 5) + (a 8) = 39) :
  (a 1) + (a 2) + (a 3) + (a 4) + (a 5) + (a 6) + (a 7) + (a 8) + (a 9) = 117 := 
sorry

end arithmetic_sequence_sum_l104_104161


namespace find_hypotenuse_of_right_angle_triangle_l104_104154

theorem find_hypotenuse_of_right_angle_triangle
  (PR : ℝ) (angle_QPR : ℝ)
  (h1 : PR = 16)
  (h2 : angle_QPR = Real.pi / 4) :
  ∃ PQ : ℝ, PQ = 16 * Real.sqrt 2 :=
by
  sorry

end find_hypotenuse_of_right_angle_triangle_l104_104154


namespace find_triples_l104_104562

theorem find_triples (x y z : ℕ) :
  (1 / x + 2 / y - 3 / z = 1) ↔ 
  ((x = 2 ∧ y = 1 ∧ z = 2) ∨
   (x = 2 ∧ y = 3 ∧ z = 18) ∨
   ∃ (n : ℕ), n ≥ 1 ∧ x = 1 ∧ y = 2 * n ∧ z = 3 * n ∨
   ∃ (k : ℕ), k ≥ 1 ∧ x = k ∧ y = 2 ∧ z = 3 * k) := sorry

end find_triples_l104_104562


namespace forum_posting_total_l104_104347

theorem forum_posting_total (num_members : ℕ) (num_answers_per_question : ℕ) (num_questions_per_hour : ℕ) (hours_per_day : ℕ) :
  num_members = 1000 ->
  num_answers_per_question = 5 ->
  num_questions_per_hour = 7 ->
  hours_per_day = 24 ->
  ((num_questions_per_hour * hours_per_day * num_members) + (num_answers_per_question * num_questions_per_hour * hours_per_day * num_members)) = 1008000 :=
by
  intros
  sorry

end forum_posting_total_l104_104347


namespace smallest_n_for_property_l104_104867

theorem smallest_n_for_property (n x : ℕ) (d : ℕ) (c : ℕ) 
  (hx : x = 10 * c + d) 
  (hx_prop : 10^(n-1) * d + c = 2 * x) :
  n = 18 := 
sorry

end smallest_n_for_property_l104_104867


namespace imaginary_part_of_z_l104_104403

open Complex

theorem imaginary_part_of_z (z : ℂ) (h : (3 - 4 * I) * z = abs (4 + 3 * I)) : im z = 4 / 5 :=
sorry

end imaginary_part_of_z_l104_104403


namespace relationship_x_y_l104_104747

theorem relationship_x_y (a b c : ℝ) (h₀ : a > b) (h₁ : b > c) (h₂ : x = Real.sqrt ((a - b) * (b - c))) (h₃ : y = (a - c) / 2) : 
  x ≤ y :=
by
  sorry

end relationship_x_y_l104_104747


namespace lawn_length_is_70_l104_104023

-- Definitions for conditions
def width_of_lawn : ℕ := 50
def road_width : ℕ := 10
def cost_of_roads : ℕ := 3600
def cost_per_sqm : ℕ := 3

-- Proof problem
theorem lawn_length_is_70 :
  ∃ L : ℕ, 10 * L + 10 * width_of_lawn = cost_of_roads / cost_per_sqm ∧ L = 70 := by
  sorry

end lawn_length_is_70_l104_104023


namespace sum_even_integers_correct_l104_104578

variable (S1 S2 : ℕ)

-- Definition: The sum of the first 50 positive even integers
def sum_first_50_even_integers : ℕ := 2550

-- Definition: The sum of even integers from 102 to 200 inclusive
def sum_even_integers_from_102_to_200 : ℕ := 7550

-- Condition: The sum of the first 50 positive even integers is 2550
axiom sum_first_50_even_integers_given : S1 = sum_first_50_even_integers

-- Problem statement: Prove that the sum of even integers from 102 to 200 inclusive is 7550
theorem sum_even_integers_correct :
  S1 = sum_first_50_even_integers →
  S2 = sum_even_integers_from_102_to_200 →
  S2 = 7550 :=
by
  intros h1 h2
  rw [h2]
  sorry

end sum_even_integers_correct_l104_104578


namespace find_a3_l104_104765

variable {α : Type} [LinearOrderedField α]

def geometric_sequence (a : ℕ → α) :=
  ∃ r : α, ∀ n, a (n + 1) = a n * r

theorem find_a3 (a : ℕ → α) (h : geometric_sequence a) (h1 : a 0 * a 4 = 16) :
  a 2 = 4 ∨ a 2 = -4 :=
by
  sorry

end find_a3_l104_104765


namespace find_m_n_l104_104664

theorem find_m_n (m n : ℤ) (h : |m - 2| + (n^2 - 8 * n + 16) = 0) : m = 2 ∧ n = 4 :=
by
  sorry

end find_m_n_l104_104664


namespace minimize_expression_l104_104534

variables {x y : ℝ}

theorem minimize_expression : ∃ (x y : ℝ), 2 * x^2 + 2 * x * y + y^2 - 2 * x - 1 = -2 :=
by sorry

end minimize_expression_l104_104534


namespace configuration_count_l104_104715

theorem configuration_count :
  (∃ (w h s : ℕ), 2 * (w + h + 2 * s) = 120 ∧ w < h ∧ s % 2 = 0) →
  ∃ n, n = 196 := 
sorry

end configuration_count_l104_104715


namespace largest_integer_divides_expression_l104_104672

theorem largest_integer_divides_expression (x : ℤ) (h : Even x) :
  3 ∣ (10 * x + 1) * (10 * x + 5) * (5 * x + 3) :=
sorry

end largest_integer_divides_expression_l104_104672


namespace solve_quadratic_l104_104024

theorem solve_quadratic (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) 
  (h : ∀ x : ℝ, x^2 + c*x + d = 0 ↔ x = c ∨ x = d) : (c, d) = (1, -2) :=
sorry

end solve_quadratic_l104_104024


namespace range_a_and_inequality_l104_104849

noncomputable def f (x a : ℝ) : ℝ := x^2 - a * Real.log (x + 2)
noncomputable def f' (x a : ℝ) : ℝ := 2 * x - a / (x + 2)

theorem range_a_and_inequality (a x1 x2 : ℝ) (h_deriv: ∀ (x : ℝ), f' x a = 0 → x = x1 ∨ x = x2) (h_lt: x1 < x2) (h_extreme: f (x1) a = f (x2) a):
  (-2 < a ∧ a < 0) → 
  (f (x1) a / x2 + 1 < 0) :=
by
  sorry

end range_a_and_inequality_l104_104849


namespace quadratic_roots_interval_l104_104729

theorem quadratic_roots_interval (b : ℝ) :
  ∃ (x : ℝ), x^2 + b * x + 25 = 0 → b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end quadratic_roots_interval_l104_104729


namespace find_m_value_l104_104301

theorem find_m_value (m : ℝ) (A : Set ℝ) (h₁ : A = {0, m, m^2 - 3 * m + 2}) (h₂ : 2 ∈ A) : m = 3 :=
by
  sorry

end find_m_value_l104_104301


namespace moles_of_HCl_combined_l104_104741

/-- Prove the number of moles of Hydrochloric acid combined is 1, given that 
1 mole of Sodium hydroxide and some moles of Hydrochloric acid react to produce 
1 mole of Water, based on the balanced chemical equation: NaOH + HCl → NaCl + H2O -/
theorem moles_of_HCl_combined (moles_NaOH : ℕ) (moles_HCl : ℕ) (moles_H2O : ℕ)
  (h1 : moles_NaOH = 1) (h2 : moles_H2O = 1) 
  (balanced_eq : moles_NaOH = moles_HCl ∧ moles_HCl = moles_H2O) : 
  moles_HCl = 1 :=
by
  sorry

end moles_of_HCl_combined_l104_104741


namespace find_arith_seq_sum_l104_104944

noncomputable def arith_seq_sum : ℕ → ℕ → ℕ
| 0, d => 2
| (n+1), d => arith_seq_sum n d + d

theorem find_arith_seq_sum :
  ∃ d : ℕ, 
    arith_seq_sum 1 d + arith_seq_sum 2 d = 13 ∧
    arith_seq_sum 3 d + arith_seq_sum 4 d + arith_seq_sum 5 d = 42 :=
by
  sorry

end find_arith_seq_sum_l104_104944


namespace point_on_parabola_distance_to_directrix_is_4_l104_104299

noncomputable def distance_from_point_to_directrix (x y : ℝ) (directrix : ℝ) : ℝ :=
  abs (x - directrix)

def parabola (t : ℝ) : ℝ × ℝ :=
  (4 * t^2, 4 * t)

theorem point_on_parabola_distance_to_directrix_is_4 (m : ℝ) (t : ℝ) :
  parabola t = (3, m) → distance_from_point_to_directrix 3 m (-1) = 4 :=
by
  sorry

end point_on_parabola_distance_to_directrix_is_4_l104_104299


namespace sequence_sum_l104_104213

theorem sequence_sum (x y : ℕ) 
  (r : ℚ) 
  (h1 : 4 * r = 1) 
  (h2 : x = 256 * r)
  (h3 : y = x * r): 
  x + y = 80 := 
by 
  sorry

end sequence_sum_l104_104213


namespace sum_multiple_of_3_probability_l104_104051

noncomputable def probability_sum_multiple_of_3 (faces : List ℕ) (rolls : ℕ) (multiple : ℕ) : ℚ :=
  if rolls = 3 ∧ multiple = 3 ∧ faces = [1, 2, 3, 4, 5, 6] then 1 / 3 else 0

theorem sum_multiple_of_3_probability :
  probability_sum_multiple_of_3 [1, 2, 3, 4, 5, 6] 3 3 = 1 / 3 :=
by
  sorry

end sum_multiple_of_3_probability_l104_104051


namespace abs_equation_solution_l104_104994

theorem abs_equation_solution (x : ℝ) (h : |x - 3| = 2 * x + 4) : x = -1 / 3 :=
by
  sorry

end abs_equation_solution_l104_104994


namespace min_chips_to_A10_l104_104097

theorem min_chips_to_A10 (n : ℕ) (A : ℕ → ℕ) (hA1 : A 1 = n) :
  (∃ (σ : ℕ → ℕ), 
    (∀ i, 1 ≤ i ∧ i < 10 → (σ i = A i - 2) ∧ (σ (i + 1) = A (i + 1) + 1)) ∨ 
    (∀ i, 1 ≤ i ∧ i < 9 → (σ (i + 1) = A (i + 1) - 2) ∧ (σ (i + 2) = A (i + 2) + 1) ∧ (σ i = A i + 1)) ∧ 
    (∃ (k : ℕ), k = 10 ∧ σ k = 1)) →
  n ≥ 46 := sorry

end min_chips_to_A10_l104_104097


namespace tip_per_person_l104_104701

-- Define the necessary conditions
def hourly_wage : ℝ := 12
def people_served : ℕ := 20
def total_amount_made : ℝ := 37

-- Define the problem statement
theorem tip_per_person : (total_amount_made - hourly_wage) / people_served = 1.25 :=
by
  sorry

end tip_per_person_l104_104701


namespace original_slices_proof_l104_104384

def original_slices (andy_consumption toast_slices leftover_slice: ℕ) : ℕ :=
  andy_consumption + toast_slices + leftover_slice

theorem original_slices_proof :
  original_slices (3 * 2) (10 * 2) 1 = 27 :=
by
  sorry

end original_slices_proof_l104_104384


namespace number_of_boxes_l104_104609

variable (boxes : ℕ) -- number of boxes
variable (mangoes_per_box : ℕ) -- mangoes per box
variable (total_mangoes : ℕ) -- total mangoes

def dozen : ℕ := 12

-- Condition: each box contains 10 dozen mangoes
def condition1 : mangoes_per_box = 10 * dozen := by 
  sorry

-- Condition: total mangoes in all boxes together is 4320
def condition2 : total_mangoes = 4320 := by
  sorry

-- Proof problem: prove that the number of boxes is 36
theorem number_of_boxes (h1 : mangoes_per_box = 10 * dozen) 
                        (h2 : total_mangoes = 4320) :
  boxes = 4320 / (10 * dozen) :=
  by
  sorry

end number_of_boxes_l104_104609


namespace proposition_form_l104_104982

-- Definitions based on the conditions
def p : Prop := (12 % 4 = 0)
def q : Prop := (12 % 3 = 0)

-- Problem statement to prove
theorem proposition_form : p ∧ q :=
by
  sorry

end proposition_form_l104_104982


namespace larger_number_is_450_l104_104550

-- Given conditions
def HCF := 30
def Factor1 := 10
def Factor2 := 15

-- Derived definitions needed for the proof
def LCM := HCF * Factor1 * Factor2

def Number1 := LCM / Factor1
def Number2 := LCM / Factor2

-- The goal is to prove the larger of the two numbers is 450
theorem larger_number_is_450 : max Number1 Number2 = 450 :=
by
  sorry

end larger_number_is_450_l104_104550


namespace simplify_expression_l104_104590

theorem simplify_expression (x y z : ℝ) : (x - (2 * y + z)) - ((x + 2 * y) - 3 * z) = -4 * y + 2 * z := 
by 
sorry

end simplify_expression_l104_104590


namespace tickets_used_l104_104166

variable (C T : Nat)

theorem tickets_used (h1 : C = 7) (h2 : T = C + 5) : T = 12 := by
  sorry

end tickets_used_l104_104166


namespace numCounterexamplesCorrect_l104_104473

-- Define a function to calculate the sum of digits of a number
def digitSum (n : Nat) : Nat := 
  n.digits 10 |>.sum

-- Predicate to check if a number is prime
def isPrime (n : Nat) : Prop := 
  Nat.Prime n

-- Set definition where the sum of digits must be 5 and all digits are non-zero
def validSet (n : Nat) : Prop :=
  digitSum n = 5 ∧ ∀ d ∈ n.digits 10, d ≠ 0

-- Define the number of counterexamples
def numCounterexamples : Nat := 6

-- The final theorem stating the number of counterexamples
theorem numCounterexamplesCorrect :
  (∃ ns : Finset Nat, 
    (∀ n ∈ ns, validSet n) ∧ 
    (∀ n ∈ ns, ¬ isPrime n) ∧ 
    ns.card = numCounterexamples) :=
sorry

end numCounterexamplesCorrect_l104_104473


namespace identify_counterfeit_13_coins_identify_and_determine_weight_14_coins_impossible_with_14_coins_l104_104901

-- Proving the identification of the counterfeit coin among 13 coins in 3 weighings
theorem identify_counterfeit_13_coins (coins : Fin 13 → Real) (is_counterfeit : ∃! i, coins i ≠ coins 0) :
  ∃ measure_count : ℕ, measure_count <= 3 ∧ 
    ∃ i, (coins i ≠ coins 0) :=
sorry

-- Proving counterfeit coin weight determination with an additional genuine coin using 3 weighings
theorem identify_and_determine_weight_14_coins (coins : Fin 14 → Real) (genuine : Real) (is_counterfeit : ∃! i, coins i ≠ genuine) :
  ∃ method_exists : Prop, 
    (method_exists ∧ ∃ measure_count : ℕ, measure_count <= 3 ∧ 
    ∃ (i : Fin 14), coins i ≠ genuine) :=
sorry

-- Proving the impossibility of identifying counterfeit coin among 14 coins using 3 weighings
theorem impossible_with_14_coins (coins : Fin 14 → Real) (is_counterfeit : ∃! i, coins i ≠ coins 0) :
  ¬ (∃ measure_count : ℕ, measure_count <= 3 ∧ 
    ∃ i, (coins i ≠ coins 0)) :=
sorry

end identify_counterfeit_13_coins_identify_and_determine_weight_14_coins_impossible_with_14_coins_l104_104901


namespace total_seeds_in_garden_l104_104038

-- Definitions based on conditions
def large_bed_rows : Nat := 4
def large_bed_seeds_per_row : Nat := 25
def medium_bed_rows : Nat := 3
def medium_bed_seeds_per_row : Nat := 20
def num_large_beds : Nat := 2
def num_medium_beds : Nat := 2

-- Theorem statement to show total seeds
theorem total_seeds_in_garden : 
  num_large_beds * (large_bed_rows * large_bed_seeds_per_row) + 
  num_medium_beds * (medium_bed_rows * medium_bed_seeds_per_row) = 320 := 
by
  sorry

end total_seeds_in_garden_l104_104038


namespace stars_total_is_correct_l104_104799

-- Define the given conditions
def number_of_stars_per_student : ℕ := 6
def number_of_students : ℕ := 210

-- Define total number of stars calculation
def total_number_of_stars : ℕ := number_of_stars_per_student * number_of_students

-- Proof statement that the total number of stars is correct
theorem stars_total_is_correct : total_number_of_stars = 1260 := by
  sorry

end stars_total_is_correct_l104_104799


namespace original_number_of_students_l104_104426

theorem original_number_of_students (x : ℕ)
  (h1: 40 * x / x = 40)
  (h2: 12 * 34 = 408)
  (h3: (40 * x + 408) / (x + 12) = 36) : x = 6 :=
by
  sorry

end original_number_of_students_l104_104426


namespace intersection_points_sum_l104_104061

theorem intersection_points_sum (x1 x2 x3 y1 y2 y3 A B : ℝ)
(h1 : y1 = x1^3 - 3 * x1 + 2)
(h2 : x1 + 6 * y1 = 6)
(h3 : y2 = x2^3 - 3 * x2 + 2)
(h4 : x2 + 6 * y2 = 6)
(h5 : y3 = x3^3 - 3 * x3 + 2)
(h6 : x3 + 6 * y3 = 6)
(hA : A = x1 + x2 + x3)
(hB : B = y1 + y2 + y3) :
A = 0 ∧ B = 3 := 
by
  sorry

end intersection_points_sum_l104_104061


namespace glass_bottles_count_l104_104841

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

end glass_bottles_count_l104_104841


namespace min_number_of_each_coin_l104_104927

def total_cost : ℝ := 1.30 + 0.75 + 0.50 + 0.45

def nickel_value : ℝ := 0.05
def dime_value : ℝ := 0.10
def quarter_value : ℝ := 0.25
def half_dollar_value : ℝ := 0.50

def min_coins :=
  ∃ (n q d h : ℕ), 
  (n ≥ 1) ∧ (q ≥ 1) ∧ (d ≥ 1) ∧ (h ≥ 1) ∧ 
  ((n * nickel_value) + (q * quarter_value) + (d * dime_value) + (h * half_dollar_value) = total_cost)

theorem min_number_of_each_coin :
  min_coins ↔ (5 * half_dollar_value + 1 * quarter_value + 2 * dime_value + 1 * nickel_value = total_cost) :=
by sorry

end min_number_of_each_coin_l104_104927


namespace money_distribution_l104_104030

variable (A B C : ℝ)

theorem money_distribution
  (h₁ : A + B + C = 500)
  (h₂ : A + C = 200)
  (h₃ : C = 60) :
  B + C = 360 :=
by
  sorry

end money_distribution_l104_104030


namespace inequality_abc_l104_104593

theorem inequality_abc (a b c : ℝ) (h₁ : 0 ≤ a) (h₂ : a ≤ 2) (h₃ : 0 ≤ b) (h₄ : b ≤ 2) (h₅ : 0 ≤ c) (h₆ : c ≤ 2) :
  (a - b) * (b - c) * (a - c) ≤ 2 :=
sorry

end inequality_abc_l104_104593


namespace jenicek_decorated_cookies_total_time_for_work_jenicek_decorating_time_l104_104947

/-- Conditions:
1. The grandmother decorates five gingerbread cookies for every cycle.
2. Little Mary decorates three gingerbread cookies for every cycle.
3. Little John decorates two gingerbread cookies for every cycle.
4. All three together decorated five trays, with each tray holding twelve gingerbread cookies.
5. Little John also sorted the gingerbread cookies onto trays twelve at a time and carried them to the pantry.
6. The grandmother decorates one gingerbread cookie in four minutes.
-/

def decorated_cookies_per_cycle := 10
def total_trays := 5
def cookies_per_tray := 12
def total_cookies := total_trays * cookies_per_tray
def babicka_cookies_per_cycle := 5
def marenka_cookies_per_cycle := 3
def jenicek_cookies_per_cycle := 2
def babicka_time_per_cookie := 4

theorem jenicek_decorated_cookies :
  (total_cookies - (total_cookies / decorated_cookies_per_cycle * marenka_cookies_per_cycle + total_cookies / decorated_cookies_per_cycle * babicka_cookies_per_cycle)) = 4 :=
sorry

theorem total_time_for_work :
  (total_cookies / decorated_cookies_per_cycle * babicka_time_per_cookie * babicka_cookies_per_cycle) = 140 :=
sorry

theorem jenicek_decorating_time :
  (4 / jenicek_cookies_per_cycle * babicka_time_per_cookie * babicka_cookies_per_cycle) = 40 :=
sorry

end jenicek_decorated_cookies_total_time_for_work_jenicek_decorating_time_l104_104947


namespace triangle_problem_l104_104654

def is_isosceles_triangle (a b c : ℕ) : Prop :=
  (a = b ∨ b = c ∨ c = a)

def has_same_area (a b : ℕ) (area : ℝ) : Prop :=
  let s := (2 * a + b) / 2
  let areaT := Real.sqrt (s * (s - a) * (s - a) * (s - b))
  areaT = area

def has_same_perimeter (a b : ℕ) (perimeter : ℕ) : Prop :=
  2 * a + b = perimeter

def correct_b (b : ℕ) : Prop :=
  b = 5

theorem triangle_problem
  (a1 a2 b1 b2 : ℕ)
  (h1 : is_isosceles_triangle a1 a1 b1)
  (h2 : is_isosceles_triangle a2 a2 b2)
  (h3 : has_same_area a1 b1 (Real.sqrt 275))
  (h4 : has_same_perimeter a1 b1 22)
  (h5 : has_same_area a2 b2 (Real.sqrt 275))
  (h6 : has_same_perimeter a2 b2 22)
  (h7 : ¬(a1 = a2 ∧ b1 = b2)) : correct_b b2 :=
by
  sorry

end triangle_problem_l104_104654


namespace runway_show_total_time_l104_104188

-- Define the conditions
def time_per_trip : Nat := 2
def num_models : Nat := 6
def trips_bathing_suits_per_model : Nat := 2
def trips_evening_wear_per_model : Nat := 3
def trips_per_model : Nat := trips_bathing_suits_per_model + trips_evening_wear_per_model
def total_trips : Nat := trips_per_model * num_models

-- State the theorem
theorem runway_show_total_time : total_trips * time_per_trip = 60 := by
  -- fill in the proof here
  sorry

end runway_show_total_time_l104_104188


namespace largest_x_solution_l104_104568

noncomputable def solve_eq (x : ℝ) : Prop :=
  (15 * x^2 - 40 * x + 16) / (4 * x - 3) + 3 * x = 7 * x + 2

theorem largest_x_solution : 
  ∃ x : ℝ, solve_eq x ∧ x = -14 + Real.sqrt 218 := 
sorry

end largest_x_solution_l104_104568


namespace corn_cobs_each_row_l104_104288

theorem corn_cobs_each_row (x : ℕ) 
  (h1 : 13 * x + 16 * x = 116) : 
  x = 4 :=
by sorry

end corn_cobs_each_row_l104_104288


namespace contractor_engaged_days_l104_104524

theorem contractor_engaged_days (x y : ℕ) (earnings_per_day : ℕ) (fine_per_day : ℝ) 
    (total_earnings : ℝ) (absent_days : ℕ) 
    (h1 : earnings_per_day = 25) 
    (h2 : fine_per_day = 7.50) 
    (h3 : total_earnings = 555) 
    (h4 : absent_days = 6) 
    (h5 : total_earnings = (earnings_per_day * x : ℝ) - fine_per_day * y) 
    (h6 : y = absent_days) : 
    x = 24 := 
by
  sorry

end contractor_engaged_days_l104_104524


namespace ralphStartsWith_l104_104053

def ralphEndsWith : ℕ := 15
def ralphLoses : ℕ := 59

theorem ralphStartsWith : (ralphEndsWith + ralphLoses = 74) :=
by
  sorry

end ralphStartsWith_l104_104053


namespace probability_of_two_black_balls_relationship_x_y_l104_104080

-- Conditions
def initial_black_balls : ℕ := 3
def initial_white_balls : ℕ := 2

variable (x y : ℕ)

-- Given relationship
def total_white_balls := x + 2
def total_black_balls := y + 3
def white_ball_probability := (total_white_balls x) / (total_white_balls x + total_black_balls y + 5)

-- Proof goals
theorem probability_of_two_black_balls :
  (3 / 5) * (2 / 4) = 3 / 10 := by sorry

theorem relationship_x_y :
  white_ball_probability x y = 1 / 3 → y = 2 * x + 1 := by sorry

end probability_of_two_black_balls_relationship_x_y_l104_104080


namespace abscissa_midpoint_range_l104_104174

-- Definitions based on the given conditions.
def circle_eq (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 6
def on_circle (x y : ℝ) : Prop := circle_eq x y
def chord_length (A B : ℝ × ℝ) : Prop := (A.1 - B.1)^2 + (A.2 - B.2)^2 = (2 * Real.sqrt 2)^2
def line_eq (x y : ℝ) : Prop := x - y - 2 = 0
def on_line (x y : ℝ) : Prop := line_eq x y
def segment_length (P Q : ℝ × ℝ) : Prop := (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 4
def acute_angle (P Q G : ℝ × ℝ) : Prop := -- definition of acute angle condition
  sorry -- placeholder for the actual definition

-- The proof statement.
theorem abscissa_midpoint_range {A B P Q G M : ℝ × ℝ}
  (h_A_on_circle : on_circle A.1 A.2)
  (h_B_on_circle : on_circle B.1 B.2)
  (h_AB_length : chord_length A B)
  (h_P_on_line : on_line P.1 P.2)
  (h_Q_on_line : on_line Q.1 Q.2)
  (h_PQ_length : segment_length P Q)
  (h_angle_acute : acute_angle P Q G)
  (h_G_mid : G = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (h_M_mid : M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)) :
  (M.1 < 0) ∨ (M.1 > 3) :=
sorry

end abscissa_midpoint_range_l104_104174


namespace max_length_PC_l104_104458

-- Define the circle C and its properties
def Circle (x y : ℝ) : Prop := x^2 + (y-1)^2 = 4

-- The equilateral triangle condition and what we need to prove
theorem max_length_PC :
  (∃ (P A B : ℝ × ℝ), 
    (Circle A.1 A.2) ∧
    (Circle B.1 B.2) ∧
    (Circle ((A.1 + B.1) / 2) ((A.2 + B.2) / 2)) ∧
    (A ≠ B) ∧
    (∃ r : ℝ, (A.1 - B.1)^2 + (A.2 - B.2)^2 = r^2 ∧ 
               (P.1 - A.1)^2 + (P.2 - A.2)^2 = r^2 ∧ 
               (P.1 - B.1)^2 + (P.2 - B.2)^2 = r^2)) → 
  (∀ (P : ℝ × ℝ), 
     ∃ (max_val : ℝ), max_val = 4 ∧
     (¬(∃ (Q : ℝ × ℝ), (Circle P.1 P.2) ∧ ((Q.1 - 0)^2 + (Q.2 - 1)^2 > max_val^2))))
:= 
sorry

end max_length_PC_l104_104458


namespace determine_a_l104_104630

-- Define the function f(x)
def f (x a : ℝ) : ℝ := x^2 - a * x + 3

-- Define the condition that f(x) >= a for all x in the interval [-1, +∞)
def condition (a : ℝ) : Prop := ∀ x : ℝ, x ≥ -1 → f x a ≥ a

-- The theorem to prove:
theorem determine_a : ∀ a : ℝ, condition a ↔ a ≤ 2 :=
by
  sorry

end determine_a_l104_104630


namespace find_remainder_l104_104296

def dividend : ℕ := 997
def divisor : ℕ := 23
def quotient : ℕ := 43

theorem find_remainder : ∃ r : ℕ, dividend = (divisor * quotient) + r ∧ r = 8 :=
by
  sorry

end find_remainder_l104_104296


namespace lana_total_pages_l104_104526

theorem lana_total_pages (lana_initial_pages : ℕ) (duane_total_pages : ℕ) :
  lana_initial_pages = 8 ∧ duane_total_pages = 42 →
  (lana_initial_pages + duane_total_pages / 2) = 29 :=
by
  sorry

end lana_total_pages_l104_104526


namespace false_props_count_is_3_l104_104095

-- Define the propositions and their inferences

noncomputable def original_prop (m n : ℝ) : Prop := m > -n → m^2 > n^2
noncomputable def contrapositive (m n : ℝ) : Prop := ¬(m^2 > n^2) → ¬(m > -n)
noncomputable def inverse (m n : ℝ) : Prop := m^2 > n^2 → m > -n
noncomputable def negation (m n : ℝ) : Prop := ¬(m > -n → m^2 > n^2)

-- The main statement to be proved
theorem false_props_count_is_3 (m n : ℝ) : 
  ¬ (original_prop m n) ∧ ¬ (contrapositive m n) ∧ ¬ (inverse m n) ∧ ¬ (negation m n) →
  (3 = 3) :=
by
  sorry

end false_props_count_is_3_l104_104095


namespace weight_of_new_person_is_correct_l104_104563

noncomputable def weight_new_person (increase_per_person : ℝ) (old_weight : ℝ) (group_size : ℝ) : ℝ :=
  old_weight + group_size * increase_per_person

theorem weight_of_new_person_is_correct :
  weight_new_person 7.2 65 10 = 137 :=
by
  sorry

end weight_of_new_person_is_correct_l104_104563


namespace compare_P_Q_l104_104990

-- Define the structure of the number a with 2010 digits of 1
def a := 10^2010 - 1

-- Define P and Q based on a
def P := 24 * a^2
def Q := 24 * a^2 + 4 * a

-- Define the theorem to compare P and Q
theorem compare_P_Q : Q > P := by
  sorry

end compare_P_Q_l104_104990


namespace sachin_rahul_age_ratio_l104_104483

theorem sachin_rahul_age_ratio :
  ∀ (Sachin_age Rahul_age: ℕ),
    Sachin_age = 49 →
    Rahul_age = Sachin_age + 14 →
    Nat.gcd Sachin_age Rahul_age = 7 →
    (Sachin_age / Nat.gcd Sachin_age Rahul_age) = 7 ∧ (Rahul_age / Nat.gcd Sachin_age Rahul_age) = 9 :=
by
  intros Sachin_age Rahul_age h1 h2 h3
  rw [h1, h2]
  sorry

end sachin_rahul_age_ratio_l104_104483


namespace consecutive_even_integers_sum_l104_104577

theorem consecutive_even_integers_sum (n : ℕ) (h : n % 2 = 0) (h_pro : n * (n + 2) * (n + 4) = 3360) :
  n + (n + 2) + (n + 4) = 48 :=
by sorry

end consecutive_even_integers_sum_l104_104577


namespace sum_of_ages_eq_19_l104_104548

theorem sum_of_ages_eq_19 :
  ∃ (a b s : ℕ), (3 * a + 5 + b = s) ∧ (6 * s^2 = 2 * a^2 + 10 * b^2) ∧ (Nat.gcd a (Nat.gcd b s) = 1 ∧ a + b + s = 19) :=
sorry

end sum_of_ages_eq_19_l104_104548


namespace rectangular_prism_width_l104_104897

variables (w : ℝ)

theorem rectangular_prism_width (h : ℝ) (l : ℝ) (d : ℝ) (hyp_l : l = 5) (hyp_h : h = 7) (hyp_d : d = 15) :
  w = Real.sqrt 151 :=
by 
  -- Proof goes here
  sorry

end rectangular_prism_width_l104_104897


namespace sum_of_powers_of_4_l104_104197

theorem sum_of_powers_of_4 : 4^0 + 4^1 + 4^2 + 4^3 = 85 :=
by
  sorry

end sum_of_powers_of_4_l104_104197


namespace total_combined_area_l104_104380

-- Definition of the problem conditions
def base_parallelogram : ℝ := 20
def height_parallelogram : ℝ := 4
def base_triangle : ℝ := 20
def height_triangle : ℝ := 2

-- Given the conditions, we want to prove:
theorem total_combined_area :
  (base_parallelogram * height_parallelogram) + (0.5 * base_triangle * height_triangle) = 100 :=
by
  sorry  -- proof goes here

end total_combined_area_l104_104380


namespace fish_remaining_when_discovered_l104_104459

def start_fish := 60
def fish_eaten_per_day := 2
def days_two_weeks := 2 * 7
def fish_added_after_two_weeks := 8
def days_one_week := 7

def fish_after_two_weeks (start: ℕ) (eaten_per_day: ℕ) (days: ℕ) (added: ℕ): ℕ :=
  start - eaten_per_day * days + added

def fish_after_three_weeks (fish_after_two_weeks: ℕ) (eaten_per_day: ℕ) (days: ℕ): ℕ :=
  fish_after_two_weeks - eaten_per_day * days

theorem fish_remaining_when_discovered :
  (fish_after_three_weeks (fish_after_two_weeks start_fish fish_eaten_per_day days_two_weeks fish_added_after_two_weeks) fish_eaten_per_day days_one_week) = 26 := 
by {
  sorry
}

end fish_remaining_when_discovered_l104_104459


namespace distance_between_point_and_center_l104_104580

noncomputable def polar_to_rectangular_point (rho theta : ℝ) : ℝ × ℝ :=
  (rho * Real.cos theta, rho * Real.sin theta)

noncomputable def center_of_circle : ℝ × ℝ := (1, 0)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem distance_between_point_and_center :
  distance (polar_to_rectangular_point 2 (Real.pi / 3)) center_of_circle = Real.sqrt 3 := 
sorry

end distance_between_point_and_center_l104_104580


namespace egg_cost_l104_104155

theorem egg_cost (toast_cost : ℝ) (E : ℝ) (total_cost : ℝ)
  (dales_toast : ℝ) (dales_eggs : ℝ) (andrews_toast : ℝ) (andrews_eggs : ℝ) :
  toast_cost = 1 → 
  dales_toast = 2 → 
  dales_eggs = 2 → 
  andrews_toast = 1 → 
  andrews_eggs = 2 → 
  total_cost = 15 →
  total_cost = (dales_toast * toast_cost + dales_eggs * E) + 
               (andrews_toast * toast_cost + andrews_eggs * E) →
  E = 3 :=
by
  sorry

end egg_cost_l104_104155


namespace peter_age_fraction_l104_104934

theorem peter_age_fraction 
  (harriet_age : ℕ) 
  (mother_age : ℕ) 
  (peter_age_plus_four : ℕ) 
  (harriet_age_plus_four : ℕ) 
  (harriet_age_current : harriet_age = 13)
  (mother_age_current : mother_age = 60)
  (peter_age_condition : peter_age_plus_four = 2 * harriet_age_plus_four)
  (harriet_four_years : harriet_age_plus_four = harriet_age + 4)
  (peter_four_years : ∀ P : ℕ, peter_age_plus_four = P + 4)
: ∃ P : ℕ, P = 30 ∧ P = mother_age / 2 := 
sorry

end peter_age_fraction_l104_104934


namespace max_gas_tank_capacity_l104_104759

-- Definitions based on conditions
def start_gas : ℕ := 10
def gas_used_store : ℕ := 6
def gas_used_doctor : ℕ := 2
def refill_needed : ℕ := 10

-- Theorem statement based on the equivalence proof problem
theorem max_gas_tank_capacity : 
  start_gas - (gas_used_store + gas_used_doctor) + refill_needed = 12 :=
by
  -- Proof steps go here
  sorry

end max_gas_tank_capacity_l104_104759


namespace product_of_three_numbers_summing_to_eleven_l104_104705

def numbers : List ℕ := [2, 3, 4, 6]

theorem product_of_three_numbers_summing_to_eleven : 
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a ∈ numbers ∧ b ∈ numbers ∧ c ∈ numbers ∧ a + b + c = 11 ∧ a * b * c = 36 := 
by
  sorry

end product_of_three_numbers_summing_to_eleven_l104_104705


namespace initial_incorrect_average_l104_104477

theorem initial_incorrect_average :
  let avg_correct := 24
  let incorrect_insertion := 26
  let correct_insertion := 76
  let n := 10  
  let correct_sum := avg_correct * n
  let incorrect_sum := correct_sum - correct_insertion + incorrect_insertion   
  avg_correct * n - correct_insertion + incorrect_insertion = incorrect_sum →
  incorrect_sum / n = 19 :=
by 
  sorry

end initial_incorrect_average_l104_104477


namespace assignment_increment_l104_104315

theorem assignment_increment (M : ℤ) : (M = M + 3) → false :=
by
  sorry

end assignment_increment_l104_104315


namespace arithmetic_sequence_count_l104_104889

noncomputable def count_arithmetic_triplets : ℕ := 17

theorem arithmetic_sequence_count :
  ∃ S : Finset (Finset ℕ), 
    (∀ s ∈ S, s.card = 3 ∧ (∃ d, ∀ x ∈ s, ∀ y ∈ s, ∀ z ∈ s, (x ≠ y ∧ y ≠ z ∧ x ≠ z) → ((x = y + d ∨ x = z + d ∨ y = z + d) ∧ x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ x ≤ 9 ∧ y ≤ 9 ∧ z ≤ 9))) ∧ 
    S.card = count_arithmetic_triplets :=
by
  -- placeholder for proof
  sorry

end arithmetic_sequence_count_l104_104889


namespace spider_total_distance_l104_104703

theorem spider_total_distance : 
  ∀ (pos1 pos2 pos3 : ℝ), pos1 = 3 → pos2 = -1 → pos3 = 8.5 → 
  |pos2 - pos1| + |pos3 - pos2| = 13.5 := 
by 
  intros pos1 pos2 pos3 hpos1 hpos2 hpos3 
  sorry

end spider_total_distance_l104_104703


namespace tom_age_ratio_l104_104688

theorem tom_age_ratio (T N : ℕ) 
  (h1 : T = T)
  (h2 : T - N = 3 * (T - 5 * N)) : T / N = 7 :=
by sorry

end tom_age_ratio_l104_104688


namespace minimum_value_of_fm_plus_fp_l104_104427

def f (x a : ℝ) : ℝ := -x^3 + a * x^2 - 4

def f_prime (x a : ℝ) : ℝ := -3 * x^2 + 2 * a * x

theorem minimum_value_of_fm_plus_fp (a : ℝ) (h_extremum : f_prime 2 a = 0) (m n : ℝ) 
  (hm : -1 ≤ m ∧ m ≤ 1) (hn : -1 ≤ n ∧ n ≤ 1) : 
  f m a + f_prime n a = -13 := 
by
  -- steps of the proof would go here
  sorry

end minimum_value_of_fm_plus_fp_l104_104427


namespace calculate_x_l104_104135

theorem calculate_x :
  (422 + 404) ^ 2 - (4 * 422 * 404) = 324 :=
by
  -- proof goes here
  sorry

end calculate_x_l104_104135


namespace mutually_exclusive_not_contradictory_l104_104523

namespace BallProbability
  -- Definitions of events based on the conditions
  def at_least_two_white (outcome : Multiset (String)) : Prop := 
    Multiset.count "white" outcome ≥ 2

  def all_red (outcome : Multiset (String)) : Prop := 
    Multiset.count "red" outcome = 3

  -- Problem statement
  theorem mutually_exclusive_not_contradictory :
    ∀ outcome : Multiset (String),
    Multiset.card outcome = 3 →
    (at_least_two_white outcome → ¬all_red outcome) ∧
    ¬(∀ outcome, at_least_two_white outcome ↔ ¬all_red outcome) := 
  by
    intros
    sorry
end BallProbability

end mutually_exclusive_not_contradictory_l104_104523


namespace find_y_l104_104462

theorem find_y (x : ℝ) (y : ℝ) (h : (3 + y)^5 = (1 + 3 * y)^4) (hx : x = 1.5) : y = 1.5 :=
by
  -- Proof steps go here
  sorry

end find_y_l104_104462


namespace greatest_number_l104_104298

-- Define the base conversions
def octal_to_decimal (n : Nat) : Nat := 3 * 8^1 + 2
def quintal_to_decimal (n : Nat) : Nat := 1 * 5^2 + 1 * 5^1 + 1
def binary_to_decimal (n : Nat) : Nat := 1 * 2^5 + 0 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 0
def senary_to_decimal (n : Nat) : Nat := 5 * 6^1 + 4

theorem greatest_number :
  max (max (octal_to_decimal 32) (quintal_to_decimal 111)) (max (binary_to_decimal 101010) (senary_to_decimal 54))
  = binary_to_decimal 101010 := by sorry

end greatest_number_l104_104298


namespace hexagon_rotation_angle_l104_104295

theorem hexagon_rotation_angle (θ : ℕ) : θ = 90 → ¬ ∃ k, k * 60 = θ ∨ θ = 360 :=
by
  sorry

end hexagon_rotation_angle_l104_104295


namespace unique_solution_exists_l104_104276

def f (x y z : ℕ) : ℕ := (x + y - 2) * (x + y - 1) / 2 - z

theorem unique_solution_exists :
  ∀ (a b c d : ℕ), f a b c = 1993 ∧ f c d a = 1993 → (a = 23 ∧ b = 42 ∧ c = 23 ∧ d = 42) :=
by
  intros a b c d h
  sorry

end unique_solution_exists_l104_104276


namespace four_digit_number_exists_l104_104125

theorem four_digit_number_exists :
  ∃ (x1 x2 y1 y2 : ℕ), (x1 > 0) ∧ (x2 > 0) ∧ (y1 > 0) ∧ (y2 > 0) ∧
                       (x2 * y2 - x1 * y1 = 67) ∧ (x2 > y2) ∧ (x1 < y1) ∧
                       (x1 * 10^3 + x2 * 10^2 + y2 * 10 + y1 = 1985) := sorry

end four_digit_number_exists_l104_104125


namespace alpha_add_beta_eq_pi_div_two_l104_104788

open Real

theorem alpha_add_beta_eq_pi_div_two (α β : ℝ) (h₁ : 0 < α ∧ α < π / 2) (h₂ : 0 < β ∧ β < π / 2) (h₃ : (sin α) ^ 4 / (cos β) ^ 2 + (cos α) ^ 4 / (sin β) ^ 2 = 1) :
  α + β = π / 2 :=
sorry

end alpha_add_beta_eq_pi_div_two_l104_104788


namespace radius_inscribed_circle_ABC_l104_104829

noncomputable def radius_of_inscribed_circle (AB AC BC : ℝ) : ℝ :=
  let s := (AB + AC + BC) / 2
  let K := Real.sqrt (s * (s - AB) * (s - AC) * (s - BC))
  K / s

theorem radius_inscribed_circle_ABC (hAB : AB = 18) (hAC : AC = 18) (hBC : BC = 24) :
  radius_of_inscribed_circle 18 18 24 = 2 * Real.sqrt 6 := by
  sorry

end radius_inscribed_circle_ABC_l104_104829


namespace earning_hours_per_week_l104_104008

theorem earning_hours_per_week (totalEarnings : ℝ) (originalWeeks : ℝ) (missedWeeks : ℝ) 
  (originalHoursPerWeek : ℝ) : 
  missedWeeks = 3 → originalWeeks = 15 → originalHoursPerWeek = 25 → totalEarnings = 3750 → 
  (totalEarnings / ((totalEarnings / (originalWeeks * originalHoursPerWeek)) * (originalWeeks - missedWeeks))) = 31.25 :=
by
  intros
  sorry

end earning_hours_per_week_l104_104008


namespace man_age_year_l104_104942

theorem man_age_year (x : ℕ) (h1 : x^2 = 1892) (h2 : 1850 ≤ x ∧ x ≤ 1900) :
  (x = 44) → (1892 = 1936) := by
sorry

end man_age_year_l104_104942


namespace least_pos_integer_to_yield_multiple_of_5_l104_104152

theorem least_pos_integer_to_yield_multiple_of_5 (n : ℕ) (h : n > 0) :
  ((567 + n) % 5 = 0) ↔ (n = 3) :=
by {
  sorry
}

end least_pos_integer_to_yield_multiple_of_5_l104_104152


namespace exist_n_consecutive_not_perfect_power_l104_104571

theorem exist_n_consecutive_not_perfect_power (n : ℕ) (h : n > 0) : 
  ∃ m : ℕ, ∀ k : ℕ, k < n → ¬ (∃ a b : ℕ, a > 1 ∧ b > 1 ∧ (m + k) = a ^ b) :=
sorry

end exist_n_consecutive_not_perfect_power_l104_104571


namespace find_range_of_a_l104_104486

noncomputable def range_of_a (a : ℝ) (n : ℕ) : Prop :=
  1 + 1 / (n : ℝ) ≤ a ∧ a < 1 + 1 / ((n - 1) : ℝ)

theorem find_range_of_a (a : ℝ) (n : ℕ) (h1 : 1 < a) (h2 : 2 ≤ n) :
  (∃ x : ℕ, ∀ x₀ < x, (⌊a * (x₀ : ℝ)⌋ : ℝ) = x₀) ↔ range_of_a a n := by
  sorry

end find_range_of_a_l104_104486


namespace find_AE_l104_104168

-- Define the given conditions as hypotheses
variables (AB CD AC AE EC : ℝ)
variables (E : Type _)
variables (triangle_AED triangle_BEC : E)

-- Assume the given conditions
axiom AB_eq_9 : AB = 9
axiom CD_eq_12 : CD = 12
axiom AC_eq_14 : AC = 14
axiom areas_equal : ∀ h : ℝ, 1/2 * AE * h = 1/2 * EC * h

-- Declare the theorem statement to prove AE
theorem find_AE (h : ℝ) (h' : EC = AC - AE) (h'' : 4 * AE = 3 * EC) : AE = 6 :=
by {
  -- proof steps as intermediate steps
  sorry
}

end find_AE_l104_104168


namespace intersection_M_N_l104_104478

def M : Set ℕ := {3, 5, 6, 8}
def N : Set ℕ := {4, 5, 7, 8}

theorem intersection_M_N : M ∩ N = {5, 8} :=
  sorry

end intersection_M_N_l104_104478


namespace base_amount_calculation_l104_104360

theorem base_amount_calculation (tax_amount : ℝ) (tax_rate : ℝ) (base_amount : ℝ) 
  (h1 : tax_amount = 82) (h2 : tax_rate = 82) : base_amount = 100 :=
by
  -- Proof will be provided here.
  sorry

end base_amount_calculation_l104_104360


namespace perp_line_eq_l104_104935

theorem perp_line_eq (x y : ℝ) (h1 : (x, y) = (1, 1)) (h2 : y = 2 * x) :
  ∃ a b c : ℝ, a * x + b * y + c = 0 ∧ a = 1 ∧ b = 2 ∧ c = -3 :=
by 
  sorry

end perp_line_eq_l104_104935


namespace no_domovoi_exists_l104_104017

variables {Domovoi Creature : Type}

def likes_pranks (c : Creature) : Prop := sorry
def likes_cleanliness_order (c : Creature) : Prop := sorry
def is_domovoi (c : Creature) : Prop := sorry

axiom all_domovoi_like_pranks : ∀ (c : Creature), is_domovoi c → likes_pranks c
axiom all_domovoi_like_cleanliness : ∀ (c : Creature), is_domovoi c → likes_cleanliness_order c
axiom cleanliness_implies_no_pranks : ∀ (c : Creature), likes_cleanliness_order c → ¬ likes_pranks c

theorem no_domovoi_exists : ¬ ∃ (c : Creature), is_domovoi c := 
sorry

end no_domovoi_exists_l104_104017


namespace six_digit_palindromes_count_l104_104103

def is_digit (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9
def is_non_zero_digit (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9

theorem six_digit_palindromes_count : 
  (∃a b c : ℕ, is_non_zero_digit a ∧ is_digit b ∧ is_digit c) → 
  (∃ n : ℕ, n = 900) :=
by
  sorry

end six_digit_palindromes_count_l104_104103


namespace dot_product_is_six_l104_104530

def a : ℝ × ℝ := (-2, 4)
def b : ℝ × ℝ := (1, 2)

theorem dot_product_is_six : (a.1 * b.1 + a.2 * b.2) = 6 := 
by 
  -- definition and proof logic follows
  sorry

end dot_product_is_six_l104_104530


namespace average_of_sequence_l104_104880

theorem average_of_sequence (z : ℝ) : 
  (0 + 3 * z + 9 * z + 27 * z + 81 * z) / 5 = 24 * z :=
by
  sorry

end average_of_sequence_l104_104880


namespace product_of_areas_eq_square_of_volume_l104_104805

theorem product_of_areas_eq_square_of_volume (w : ℝ) :
  let l := 2 * w
  let h := 3 * w
  let A_bottom := l * w
  let A_side := w * h
  let A_front := l * h
  let volume := l * w * h
  A_bottom * A_side * A_front = volume^2 :=
by
  sorry

end product_of_areas_eq_square_of_volume_l104_104805


namespace fibonacci_sequence_x_l104_104822

theorem fibonacci_sequence_x {a : ℕ → ℕ} 
  (h1 : a 1 = 1) 
  (h2 : a 2 = 2) 
  (h3 : a 3 = 3) 
  (h_fib : ∀ n, n ≥ 3 → a (n + 1) = a n + a (n - 1)) : 
  a 5 = 8 := 
sorry

end fibonacci_sequence_x_l104_104822


namespace coefficient_of_x4_in_expansion_of_2x_plus_sqrtx_l104_104129

noncomputable def coefficient_of_x4_expansion : ℕ :=
  let r := 2;
  let n := 5;
  let general_term_coefficient := Nat.choose n r * 2^(n-r);
  general_term_coefficient

theorem coefficient_of_x4_in_expansion_of_2x_plus_sqrtx :
  coefficient_of_x4_expansion = 80 :=
by
  -- We can bypass the actual proving steps by
  -- acknowledging that the necessary proof mechanism
  -- will properly verify the calculation:
  sorry

end coefficient_of_x4_in_expansion_of_2x_plus_sqrtx_l104_104129


namespace field_division_l104_104673

theorem field_division (A B : ℝ) (h1 : A + B = 700) (h2 : B - A = (1 / 5) * ((A + B) / 2)) : A = 315 :=
by
  sorry

end field_division_l104_104673


namespace maximize_profit_l104_104332
-- Importing the entire necessary library

-- Definitions and conditions
def cost_price : ℕ := 40
def minimum_selling_price : ℕ := 44
def maximum_profit_margin : ℕ := 30
def sales_at_minimum_price : ℕ := 300
def price_increase_effect : ℕ := 10
def max_profit_price := 52
def max_profit := 2640

-- Function relationship between y and x
def sales_volume (x : ℕ) : ℕ := 300 - 10 * (x - 44)

-- Range of x
def valid_price (x : ℕ) : Prop := 44 ≤ x ∧ x ≤ 52

-- Statement of the problem
theorem maximize_profit (x : ℕ) (hx : valid_price x) : 
  sales_volume x = 300 - 10 * (x - 44) ∧
  44 ≤ x ∧ x ≤ 52 ∧
  x = 52 → 
  (x - cost_price) * (sales_volume x) = max_profit :=
sorry

end maximize_profit_l104_104332


namespace angle_between_AD_and_BC_l104_104521

variables {a b c : ℝ} 
variables {θ : ℝ}
variables {α β γ δ ε ζ : ℝ} -- representing the angles

-- Conditions of the problem
def conditions (a b c : ℝ) (α β γ δ ε ζ : ℝ) : Prop :=
  (α + β + γ = 180) ∧ (δ + ε + ζ = 180) ∧ 
  (a > 0) ∧ (b > 0) ∧ (c > 0)

-- Definition of the theorem to prove the angle between AD and BC
theorem angle_between_AD_and_BC
  (a b c : ℝ) (α β γ δ ε ζ : ℝ)
  (h : conditions a b c α β γ δ ε ζ) :
  θ = Real.arccos ((|b^2 - c^2|) / a^2) :=
sorry

end angle_between_AD_and_BC_l104_104521


namespace min_max_of_f_l104_104002

def f (x : ℝ) : ℝ := -2 * x + 1

-- defining the minimum and maximum values
def min_val : ℝ := -3
def max_val : ℝ := 5

theorem min_max_of_f :
  (∀ x : ℝ, x ∈ Set.Icc (-2) 2 → f x ≥ min_val) ∧ 
  (∀ x : ℝ, x ∈ Set.Icc (-2) 2 → f x ≤ max_val) :=
by 
  sorry

end min_max_of_f_l104_104002


namespace union_of_A_and_B_intersection_of_A_and_complementB_range_of_m_l104_104418

open Set

def setA : Set ℝ := {x | -4 < x ∧ x < 2}
def setB : Set ℝ := {x | x < -5 ∨ x > 1}
def setComplementB : Set ℝ := {x | -5 ≤ x ∧ x ≤ 1}

theorem union_of_A_and_B : setA ∪ setB = {x | x < -5 ∨ x > -4} := by
  sorry

theorem intersection_of_A_and_complementB : setA ∩ setComplementB = {x | -4 < x ∧ x ≤ 1} := by
  sorry

noncomputable def setC (m : ℝ) : Set ℝ := {x | m - 1 < x ∧ x < m + 1}

theorem range_of_m (m : ℝ) (h : setB ∩ (setC m) = ∅) : -4 ≤ m ∧ m ≤ 0 := by
  sorry

end union_of_A_and_B_intersection_of_A_and_complementB_range_of_m_l104_104418


namespace zoo_animal_difference_l104_104225

theorem zoo_animal_difference :
  let parrots := 8
  let snakes := 3 * parrots
  let monkeys := 2 * snakes
  let elephants := (parrots + snakes) / 2
  let zebras := elephants - 3
  monkeys - zebras = 35 := by
  let parrots := 8
  let snakes := 3 * parrots
  let monkeys := 2 * snakes
  let elephants := (parrots + snakes) / 2
  let zebras := elephants - 3
  show monkeys - zebras = 35
  sorry

end zoo_animal_difference_l104_104225


namespace pathway_bricks_total_is_280_l104_104268

def total_bricks (n : ℕ) : ℕ :=
  let odd_bricks := 2 * (1 + 1 + ((n / 2) - 1) * 2)
  let even_bricks := 4 * (1 + 2 + (n / 2 - 1) * 2)
  odd_bricks + even_bricks
   
theorem pathway_bricks_total_is_280 (n : ℕ) (h : total_bricks n = 280) : n = 10 :=
sorry

end pathway_bricks_total_is_280_l104_104268


namespace linear_regression_forecast_l104_104617

variable (x : ℝ) (y : ℝ)
variable (b : ℝ) (a : ℝ) (center_x : ℝ) (center_y : ℝ)

theorem linear_regression_forecast :
  b=-2 → center_x=4 → center_y=50 → (center_y = b * center_x + a) →
  (a = 58) → (x = 6) → y = b * x + a → y = 46 :=
by
  intros hb hcx hcy heq ha hx hy
  sorry

end linear_regression_forecast_l104_104617


namespace tim_income_less_juan_l104_104373

variable {T M J : ℝ}

theorem tim_income_less_juan :
  (M = 1.60 * T) → (M = 0.6400000000000001 * J) → T = 0.4 * J :=
by
  sorry

end tim_income_less_juan_l104_104373


namespace correct_option_l104_104106

def condition_A (a : ℝ) : Prop := a^3 * a^4 = a^12
def condition_B (a b : ℝ) : Prop := (-3 * a * b^3)^2 = -6 * a * b^6
def condition_C (a : ℝ) : Prop := (a - 3)^2 = a^2 - 9
def condition_D (x y : ℝ) : Prop := (-x + y) * (x + y) = y^2 - x^2

theorem correct_option (x y : ℝ) : condition_D x y := by
  sorry

end correct_option_l104_104106


namespace original_height_in_feet_l104_104419

-- Define the current height in inches
def current_height_in_inches : ℚ := 180

-- Define the percentage increase in height
def percentage_increase : ℚ := 0.5

-- Define the conversion factor from inches to feet
def inches_to_feet : ℚ := 12

-- Define the initial height in inches
def initial_height_in_inches : ℚ := current_height_in_inches / (1 + percentage_increase)

-- Prove that the original height in feet was 10 feet
theorem original_height_in_feet : initial_height_in_inches / inches_to_feet = 10 :=
by
  -- Placeholder for the full proof
  sorry

end original_height_in_feet_l104_104419


namespace solve_variables_l104_104863

theorem solve_variables (x y z : ℝ)
  (h1 : (x / 6) * 12 = 10)
  (h2 : (y / 4) * 8 = x)
  (h3 : (z / 3) * 5 + y = 20) :
  x = 5 ∧ y = 2.5 ∧ z = 10.5 :=
by { sorry }

end solve_variables_l104_104863


namespace sum_of_other_endpoint_coords_l104_104750

theorem sum_of_other_endpoint_coords (x y : ℝ) (hx : (6 + x) / 2 = 5) (hy : (2 + y) / 2 = 7) : x + y = 16 := 
  sorry

end sum_of_other_endpoint_coords_l104_104750


namespace jen_age_difference_l104_104537

-- Definitions as conditions given in the problem
def son_present_age := 16
def jen_present_age := 41

-- The statement to be proved
theorem jen_age_difference :
  3 * son_present_age - jen_present_age = 7 :=
by
  sorry

end jen_age_difference_l104_104537


namespace radius_increase_l104_104096

theorem radius_increase (C1 C2 : ℝ) (h1 : C1 = 30) (h2 : C2 = 40) : 
  let r1 := C1 / (2 * Real.pi)
  let r2 := C2 / (2 * Real.pi)
  let Δr := r2 - r1
  Δr = 5 / Real.pi := by
sorry

end radius_increase_l104_104096


namespace sheena_sewing_hours_weekly_l104_104601

theorem sheena_sewing_hours_weekly
  (hours_per_dress : ℕ)
  (number_of_dresses : ℕ)
  (weeks_to_complete : ℕ)
  (total_sewing_hours : ℕ)
  (hours_per_week : ℕ) :
  hours_per_dress = 12 →
  number_of_dresses = 5 →
  weeks_to_complete = 15 →
  total_sewing_hours = number_of_dresses * hours_per_dress →
  hours_per_week = total_sewing_hours / weeks_to_complete →
  hours_per_week = 4 := by
  intros h1 h2 h3 h4 h5
  sorry

end sheena_sewing_hours_weekly_l104_104601


namespace find_incorrect_value_of_observation_l104_104187

noncomputable def incorrect_observation_value (mean1 : ℝ) (mean2 : ℝ) (n : ℕ) : ℝ :=
  let old_sum := mean1 * n
  let new_sum := mean2 * n
  let correct_value := 45
  let incorrect_value := (old_sum - new_sum + correct_value)
  (incorrect_value / -1)

theorem find_incorrect_value_of_observation :
  incorrect_observation_value 36 36.5 50 = 20 :=
by
  -- By the problem setup, incorrect_observation_value 36 36.5 50 is as defined in the proof steps.
  -- As per the proof steps and calculation, incorrect_observation_value 36 36.5 50 should evaluate to 20.
  sorry

end find_incorrect_value_of_observation_l104_104187


namespace distance_to_grocery_store_l104_104007

-- Definitions of given conditions
def miles_to_mall := 6
def miles_to_pet_store := 5
def miles_back_home := 9
def miles_per_gallon := 15
def cost_per_gallon := 3.5
def total_cost := 7

-- The Lean statement to prove the distance driven to the grocery store.
theorem distance_to_grocery_store (miles_to_mall miles_to_pet_store miles_back_home miles_per_gallon cost_per_gallon total_cost : ℝ) :
(total_cost / cost_per_gallon) * miles_per_gallon - (miles_to_mall + miles_to_pet_store + miles_back_home) = 10 := by
  sorry

end distance_to_grocery_store_l104_104007


namespace find_cashew_kilos_l104_104993

variables (x : ℕ)

def cashew_cost_per_kilo := 210
def peanut_cost_per_kilo := 130
def total_weight := 5
def peanuts_weight := 2
def avg_price_per_kilo := 178

-- Given conditions
def cashew_total_cost := cashew_cost_per_kilo * x
def peanut_total_cost := peanut_cost_per_kilo * peanuts_weight
def total_price := total_weight * avg_price_per_kilo

theorem find_cashew_kilos (h1 : cashew_total_cost + peanut_total_cost = total_price) : x = 3 :=
by
  sorry

end find_cashew_kilos_l104_104993


namespace lockers_count_l104_104277

theorem lockers_count 
(TotalCost : ℝ) 
(first_cents : ℝ) 
(additional_cents : ℝ) 
(locker_start : ℕ) 
(locker_end : ℕ) : 
  TotalCost = 155.94 
  → first_cents = 0 
  → additional_cents = 0.03 
  → locker_start = 2 
  → locker_end = 1825 := 
by
  -- Declare the number of lockers as a variable and use it to construct the proof
  let num_lockers := locker_end - locker_start + 1
  -- The cost for labeling can be calculated and matched with TotalCost
  sorry

end lockers_count_l104_104277


namespace eighth_graders_ninth_grader_points_l104_104621

noncomputable def eighth_grader_points (y : ℚ) (x : ℕ) : Prop :=
  x * y + 8 = ((x + 2) * (x + 1)) / 2

theorem eighth_graders (x : ℕ) (y : ℚ) (hx : eighth_grader_points y x) :
  x = 7 ∨ x = 14 :=
sorry

noncomputable def tenth_grader_points (z y : ℚ) (x : ℕ) : Prop :=
  10 * z = 4.5 * y ∧ x * z = y

theorem ninth_grader_points (y : ℚ) (x : ℕ) (z : ℚ)
  (hx : tenth_grader_points z y x) :
  y = 10 :=
sorry

end eighth_graders_ninth_grader_points_l104_104621


namespace find_minimum_l104_104879

theorem find_minimum (a b c : ℝ) : ∃ (m : ℝ), m = min a (min b c) := 
  sorry

end find_minimum_l104_104879


namespace arrange_abc_l104_104878

open Real

noncomputable def a := log 4 / log 5
noncomputable def b := (log 3 / log 5)^2
noncomputable def c := 1 / (log 4 / log 5)

theorem arrange_abc : b < a ∧ a < c :=
by
  -- Mathematical translations as Lean proof obligations
  have a_lt_one : a < 1 := by sorry
  have c_gt_one : c > 1 := by sorry
  have b_lt_a : b < a := by sorry
  have a_lt_c : a < c := by sorry
  exact ⟨b_lt_a, a_lt_c⟩

end arrange_abc_l104_104878


namespace right_triangle_perimeter_l104_104628

theorem right_triangle_perimeter (a b : ℝ) (c : ℝ) (h1 : a * b = 72) 
  (h2 : c ^ 2 = a ^ 2 + b ^ 2) (h3 : a = 12) :
  a + b + c = 18 + 6 * Real.sqrt 5 := 
by
  sorry

end right_triangle_perimeter_l104_104628


namespace cost_of_pencil_and_pen_l104_104190

variable (p q : ℝ)

axiom condition1 : 4 * p + 3 * q = 4.20
axiom condition2 : 3 * p + 4 * q = 4.55

theorem cost_of_pencil_and_pen : p + q = 1.25 :=
by
  sorry

end cost_of_pencil_and_pen_l104_104190


namespace parallelogram_angle_l104_104280

theorem parallelogram_angle (a b : ℕ) (h : a + b = 180) (exceed_by_10 : b = a + 10) : a = 85 := by
  -- proof skipped
  sorry

end parallelogram_angle_l104_104280


namespace Donovan_percentage_correct_l104_104067

-- Definitions based on conditions from part a)
def fullyCorrectAnswers : ℕ := 35
def incorrectAnswers : ℕ := 13
def partiallyCorrectAnswers : ℕ := 7
def pointPerFullAnswer : ℝ := 1
def pointPerPartialAnswer : ℝ := 0.5

-- Lean 4 statement to prove the problem mathematically
theorem Donovan_percentage_correct : 
  (fullyCorrectAnswers * pointPerFullAnswer + partiallyCorrectAnswers * pointPerPartialAnswer) / 
  (fullyCorrectAnswers + incorrectAnswers + partiallyCorrectAnswers) * 100 = 70.00 :=
by
  sorry

end Donovan_percentage_correct_l104_104067


namespace part1_part2_part3_l104_104717

noncomputable def quadratic_has_real_roots (k : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, x1^2 - 2*k*x1 + k^2 + k + 1 = 0 ∧ x2^2 - 2*k*x2 + k^2 + k + 1 = 0

theorem part1 (k : ℝ) :
  quadratic_has_real_roots k → k ≤ -1 :=
sorry

theorem part2 (k : ℝ) (x1 x2 : ℝ) :
  quadratic_has_real_roots k ∧ x1^2 + x2^2 = 10 → k = -2 :=
sorry

theorem part3 (k : ℝ) (x1 x2 : ℝ) :
  quadratic_has_real_roots k ∧ (|x1| + |x2| = 2) → k = -1 :=
sorry

end part1_part2_part3_l104_104717


namespace largest_negative_root_l104_104549

theorem largest_negative_root : 
  ∃ x : ℝ, (∃ k : ℤ, x = -1/2 + 2 * ↑k) ∧ 
  ∀ y : ℝ, (∃ k : ℤ, (y = -1/2 + 2 * ↑k ∨ y = 1/6 + 2 * ↑k ∨ y = 5/6 + 2 * ↑k)) → y < 0 → y ≤ x :=
sorry

end largest_negative_root_l104_104549


namespace perpendicular_lines_l104_104856

theorem perpendicular_lines (a : ℝ) :
  (∃ x y : ℝ, ax + 2 * y + 6 = 0) ∧ (∃ x y : ℝ, x + (a - 1) * y + a^2 - 1 = 0) ∧ (∀ m1 m2 : ℝ, m1 * m2 = -1) →
  a = 2/3 :=
by
  sorry

end perpendicular_lines_l104_104856


namespace calculate_change_l104_104646

theorem calculate_change : 
  let bracelet_cost := 15
  let necklace_cost := 10
  let mug_cost := 20
  let num_bracelets := 3
  let num_necklaces := 2
  let num_mugs := 1
  let discount := 0.10
  let total_cost := (num_bracelets * bracelet_cost) + (num_necklaces * necklace_cost) + (num_mugs * mug_cost)
  let discount_amount := total_cost * discount
  let final_amount := total_cost - discount_amount
  let payment := 100
  let change := payment - final_amount
  change = 23.50 :=
by
  -- Intentionally skipping the proof
  sorry

end calculate_change_l104_104646


namespace sum_of_fifths_divisible_by_30_l104_104809

open BigOperators

theorem sum_of_fifths_divisible_by_30 {a : ℕ → ℕ} {n : ℕ} 
  (h : 30 ∣ ∑ i in Finset.range n, a i) : 
  30 ∣ ∑ i in Finset.range n, (a i) ^ 5 := 
by sorry

end sum_of_fifths_divisible_by_30_l104_104809


namespace class1_qualified_l104_104539

variables (Tardiness : ℕ → ℕ) -- Tardiness function mapping days to number of tardy students

def classQualified (mean variance median mode : ℕ) : Prop :=
  (mean = 2 ∧ variance = 2) ∨
  (mean = 3 ∧ median = 3) ∨
  (mean = 2 ∧ variance > 0) ∨
  (median = 2 ∧ mode = 2)

def eligible (Tardiness : ℕ → ℕ) : Prop :=
  ∀ i, i < 5 → Tardiness i ≤ 5

theorem class1_qualified : 
  (∀ Tardiness, (∃ mean variance median mode,
    classQualified mean variance median mode 
    ∧ mean = 2 ∧ variance = 2 
    ∧ eligible Tardiness)) → 
  (∀ Tardiness, eligible Tardiness) :=
by
  sorry

end class1_qualified_l104_104539


namespace proof_of_greatest_sum_quotient_remainder_l104_104511

def greatest_sum_quotient_remainder : Prop :=
  ∃ q r : ℕ, 1051 = 23 * q + r ∧ q > 0 ∧ r > 0 ∧ q + r = 61

theorem proof_of_greatest_sum_quotient_remainder : greatest_sum_quotient_remainder := 
sorry

end proof_of_greatest_sum_quotient_remainder_l104_104511


namespace race_head_start_l104_104825

variable (vA vB L h : ℝ)
variable (hva_vb : vA = (16 / 15) * vB)

theorem race_head_start (hL_pos : L > 0) (hvB_pos : vB > 0) 
    (h_times_eq : (L / vA) = ((L - h) / vB)) : h = L / 16 :=
by
  sorry

end race_head_start_l104_104825


namespace units_digit_in_base_7_l104_104386

theorem units_digit_in_base_7 (n m : ℕ) (h1 : n = 312) (h2 : m = 57) : (n * m) % 7 = 4 :=
by
  sorry

end units_digit_in_base_7_l104_104386


namespace total_area_correct_l104_104671

noncomputable def radius : ℝ := 7
noncomputable def width : ℝ := 2 * radius
noncomputable def length : ℝ := 3 * width
noncomputable def rect_area : ℝ := length * width
noncomputable def square_side : ℝ := radius * Real.sqrt 2
noncomputable def square_area : ℝ := square_side ^ 2
noncomputable def total_area : ℝ := rect_area + square_area

theorem total_area_correct : total_area = 686 := 
by
  -- Definitions provided above represent the problem's conditions
  -- The value calculated manually is 686
  -- Proof steps skipped for initial statement creation
  sorry

end total_area_correct_l104_104671


namespace coordinate_minimizes_z_l104_104985

-- Definitions for conditions
def in_first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

def equation_holds (x y : ℝ) : Prop := (1 / x) + (1 / (2 * y)) + (3 / (2 * x * y)) = 1

def z_def (x y : ℝ) : ℝ := x * y

-- Statement
theorem coordinate_minimizes_z (x y : ℝ) (h1 : in_first_quadrant x y) (h2 : equation_holds x y) :
    z_def x y = 9 / 2 ∧ (x = 3 ∧ y = 3 / 2) :=
    sorry

end coordinate_minimizes_z_l104_104985


namespace abc_value_l104_104869

theorem abc_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
    (h1 : a * (b + c) = 171) 
    (h2 : b * (c + a) = 180) 
    (h3 : c * (a + b) = 189) :
    a * b * c = 270 :=
by
  -- Place proofs here
  sorry

end abc_value_l104_104869


namespace masha_final_number_stabilizes_masha_smallest_initial_number_ends_with_09_l104_104460

/-- 
Part (a): Define the problem statement where, given the iterative process on a number,
it stabilizes at 17.
-/
theorem masha_final_number_stabilizes (x y : ℕ) (n : ℕ) (h_stable : ∀ x y, 10 * x + y = 3 * x + 2 * y) :
  n = 17 :=
by
  sorry

/--
Part (b): Define the problem statement to find the smallest 2015-digit number ending with the
digits 09 that eventually stabilizes to 17.
-/
theorem masha_smallest_initial_number_ends_with_09 :
  ∃ (n : ℕ), n ≥ 10^2014 ∧ n % 100 = 9 ∧ (∃ k : ℕ, 10^2014 + k = n ∧ (10 * ((n - k) / 10) + (n % 10)) = 17) :=
by
  sorry

end masha_final_number_stabilizes_masha_smallest_initial_number_ends_with_09_l104_104460


namespace andy_late_l104_104789

theorem andy_late
  (school_start : ℕ := 480) -- 8:00 AM in minutes since midnight
  (normal_travel_time : ℕ := 30)
  (red_lights : ℕ := 4)
  (red_light_wait_time : ℕ := 3)
  (construction_wait_time : ℕ := 10)
  (departure_time : ℕ := 435) -- 7:15 AM in minutes since midnight
  : ((school_start - departure_time) < (normal_travel_time + (red_lights * red_light_wait_time) + construction_wait_time)) →
    school_start + (normal_travel_time + (red_lights * red_light_wait_time) + construction_wait_time - (school_start - departure_time)) = school_start + 7 :=
by
  -- This skips the proof part
  sorry

end andy_late_l104_104789


namespace range_of_m_l104_104792
-- Import the essential libraries

-- Define the problem conditions and state the theorem
theorem range_of_m (f : ℝ → ℝ) (h_even : ∀ x : ℝ, f x = f (-x)) (h_mono_dec : ∀ x y : ℝ, 0 ≤ x → x ≤ y → f y ≤ f x)
  (m : ℝ) (h_ineq : f m > f (1 - m)) : m < 1 / 2 :=
sorry

end range_of_m_l104_104792


namespace range_of_m_l104_104926

noncomputable def satisfies_inequality (m : ℝ) : Prop :=
  ∀ x : ℝ, |x + 3| - |x - 1| ≤ m^2 - 3 * m

theorem range_of_m (m : ℝ) : 
  satisfies_inequality m ↔ (m ≥ 4 ∨ m ≤ -1) :=
by
  sorry

end range_of_m_l104_104926


namespace iter_f_eq_l104_104439

namespace IteratedFunction

def f (n : ℕ) (x : ℕ) : ℕ :=
  if 2 * x <= n then
    2 * x
  else
    2 * n - 2 * x + 1

def iter_f (n m : ℕ) (x : ℕ) : ℕ :=
  (Nat.iterate (f n) m) x

variables (n m : ℕ) (S : Fin n.succ → Fin n.succ)

theorem iter_f_eq (h : iter_f n m 1 = 1) (k : Fin n.succ) :
  iter_f n m k = k := by
  sorry

end IteratedFunction

end iter_f_eq_l104_104439


namespace find_matrix_N_l104_104217

-- Define the given matrix equation
def condition (N : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  N ^ 3 - 3 * N ^ 2 + 4 * N = ![![8, 16], ![4, 8]]

-- State the theorem
theorem find_matrix_N (N : Matrix (Fin 2) (Fin 2) ℝ) (h : condition N) :
  N = ![![2, 4], ![1, 2]] :=
sorry

end find_matrix_N_l104_104217


namespace upper_seat_ticket_price_l104_104906

variable (U : ℝ) 

-- Conditions
def lower_seat_price : ℝ := 30
def total_tickets_sold : ℝ := 80
def total_revenue : ℝ := 2100
def lower_tickets_sold : ℝ := 50

theorem upper_seat_ticket_price :
  (lower_seat_price * lower_tickets_sold + (total_tickets_sold - lower_tickets_sold) * U = total_revenue) →
  U = 20 := by
  sorry

end upper_seat_ticket_price_l104_104906


namespace values_of_a_and_b_solution_set_inequality_l104_104250

-- Part (I)
theorem values_of_a_and_b (a b : ℝ) (h : ∀ x, -1 < x ∧ x < 1 → x^2 - a * x - x + b < 0) :
  a = -1 ∧ b = -1 := sorry

-- Part (II)
theorem solution_set_inequality (a : ℝ) (h : a = b) :
  (∀ x, x^2 - a * x - x + a < 0 → (x = 1 → false) 
      ∧ (0 < 1 - a → (x = 1 → false))
      ∧ (1 < - a → (x = 1 → false))) := sorry

end values_of_a_and_b_solution_set_inequality_l104_104250


namespace max_area_of_rectangle_with_perimeter_40_l104_104502

theorem max_area_of_rectangle_with_perimeter_40 :
  ∃ (A : ℝ), (A = 100) ∧ (∀ (length width : ℝ), 2 * (length + width) = 40 → length * width ≤ A) :=
by
  sorry

end max_area_of_rectangle_with_perimeter_40_l104_104502


namespace smaller_number_in_ratio_l104_104099

noncomputable def LCM (a b : ℕ) : ℕ := (a * b) / Nat.gcd a b

theorem smaller_number_in_ratio (x : ℕ) (a b : ℕ) (h1 : a = 4 * x) (h2 : b = 5 * x) (h3 : LCM a b = 180) : a = 36 := 
by
  sorry

end smaller_number_in_ratio_l104_104099


namespace triangle_perimeter_l104_104346

def right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

variable (a b c : ℝ)

theorem triangle_perimeter
  (h1 : 90 = (1/2) * 18 * b)
  (h2 : right_triangle 18 b c) :
  18 + b + c = 28 + 2 * Real.sqrt 106 :=
by
  sorry

end triangle_perimeter_l104_104346


namespace harold_wrapping_paper_cost_l104_104683

theorem harold_wrapping_paper_cost :
  let rolls_for_shirt_boxes := 20 / 5
  let rolls_for_xl_boxes := 12 / 3
  let total_rolls := rolls_for_shirt_boxes + rolls_for_xl_boxes
  let cost_per_roll := 4  -- dollars
  (total_rolls * cost_per_roll) = 32 := by
  sorry

end harold_wrapping_paper_cost_l104_104683


namespace leftover_cents_l104_104399

noncomputable def total_cents (pennies nickels dimes quarters : Nat) : Nat :=
  (pennies * 1) + (nickels * 5) + (dimes * 10) + (quarters * 25)

noncomputable def total_cost (num_people : Nat) (cost_per_person : Nat) : Nat :=
  num_people * cost_per_person

theorem leftover_cents (h₁ : total_cents 123 85 35 26 = 1548)
                       (h₂ : total_cost 5 300 = 1500) :
  1548 - 1500 = 48 :=
sorry

end leftover_cents_l104_104399


namespace find_omega_l104_104259

theorem find_omega 
  (w : ℝ) 
  (h₁ : 0 < w)
  (h₂ : (π / w) = (π / 2)) : w = 2 :=
by
  sorry

end find_omega_l104_104259


namespace parallelogram_midpoints_XY_square_l104_104423

theorem parallelogram_midpoints_XY_square (A B C D X Y : ℝ)
  (AB CD : ℝ) (BC DA : ℝ) (angle_D : ℝ)
  (mid_X : X = (B + C) / 2) (mid_Y : Y = (D + A) / 2)
  (h1: AB = 10) (h2: BC = 17) (h3: CD = 10) (h4 : angle_D = 60) :
  (XY ^ 2 = 219 / 4) :=
by
  sorry

end parallelogram_midpoints_XY_square_l104_104423


namespace integers_a_b_c_d_arbitrarily_large_l104_104955

theorem integers_a_b_c_d_arbitrarily_large (n : ℤ) : 
  ∃ (a b c d : ℤ), (a^2 + b^2 + c^2 + d^2 = a * b * c + a * b * d + a * c * d + b * c * d) ∧ 
    min (min a b) (min c d) ≥ n := 
by sorry

end integers_a_b_c_d_arbitrarily_large_l104_104955


namespace graph_of_abs_g_l104_104764

noncomputable def g (x : ℝ) : ℝ :=
  if -4 ≤ x ∧ x ≤ -1 then x + 3
  else if -1 < x ∧ x ≤ 1 then -x^2 + 2
  else if 1 < x ∧ x ≤ 4 then x - 2
  else 0

noncomputable def abs_g (x : ℝ) : ℝ :=
  if -4 ≤ x ∧ x ≤ -3 then -(x + 3)
  else if -3 < x ∧ x ≤ -1 then x + 3
  else if -1 < x ∧ x ≤ 1 then -x^2 + 2
  else if 1 < x ∧ x ≤ 2 then -(x - 2)
  else if 2 < x ∧ x ≤ 4 then x - 2
  else 0

theorem graph_of_abs_g :
  ∀ x : ℝ, abs_g x = |g x| :=
by
  sorry

end graph_of_abs_g_l104_104764


namespace ones_digit_8_power_32_l104_104185

theorem ones_digit_8_power_32 : (8^32) % 10 = 6 :=
by sorry

end ones_digit_8_power_32_l104_104185


namespace value_of_f_at_2_l104_104803

def f (x : ℝ) : ℝ := x^2 + 2*x - 3

theorem value_of_f_at_2 : f 2 = 5 :=
by
  -- proof steps would go here
  sorry

end value_of_f_at_2_l104_104803


namespace tom_total_money_l104_104058

theorem tom_total_money :
  let initial_amount := 74
  let additional_amount := 86
  initial_amount + additional_amount = 160 :=
by
  let initial_amount := 74
  let additional_amount := 86
  show initial_amount + additional_amount = 160
  sorry

end tom_total_money_l104_104058


namespace mandy_cinnamon_nutmeg_difference_l104_104093

theorem mandy_cinnamon_nutmeg_difference :
  0.67 - 0.5 = 0.17 :=
by
  sorry

end mandy_cinnamon_nutmeg_difference_l104_104093


namespace sum_sequence_l104_104681

theorem sum_sequence (a : ℕ → ℚ) (S : ℕ → ℚ)
  (h1 : a 1 = -2/3)
  (h2 : ∀ n, n ≥ 2 → S n = -1 / (S (n - 1) + 2)) :
  ∀ n, S n = -(n + 1) / (n + 2) := 
by 
  sorry

end sum_sequence_l104_104681


namespace emily_necklaces_l104_104657

theorem emily_necklaces (total_beads : ℤ) (beads_per_necklace : ℤ) 
(h_total_beads : total_beads = 16) (h_beads_per_necklace : beads_per_necklace = 8) : 
  total_beads / beads_per_necklace = 2 := 
by
  sorry

end emily_necklaces_l104_104657


namespace minimum_value_ineq_l104_104528

variable {a b c : ℝ}

theorem minimum_value_ineq (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 + 2 * a + 1) * (b^2 + 2 * b + 1) * (c^2 + 2 * c + 1) / (a * b * c) ≥ 64 :=
sorry

end minimum_value_ineq_l104_104528


namespace captain_age_l104_104848

theorem captain_age
  (C W : ℕ)
  (avg_team_age : ℤ)
  (avg_remaining_players_age : ℤ)
  (total_team_age : ℤ)
  (total_remaining_players_age : ℤ)
  (remaining_players_count : ℕ)
  (total_team_count : ℕ)
  (total_team_age_eq : total_team_age = total_team_count * avg_team_age)
  (remaining_players_age_eq : total_remaining_players_age = remaining_players_count * avg_remaining_players_age)
  (total_team_eq : total_team_count = 11)
  (remaining_players_eq : remaining_players_count = 9)
  (avg_team_age_eq : avg_team_age = 23)
  (avg_remaining_players_age_eq : avg_remaining_players_age = avg_team_age - 1)
  (age_diff : W = C + 5)
  (players_age_sum : total_team_age = total_remaining_players_age + C + W) :
  C = 25 :=
by
  sorry

end captain_age_l104_104848


namespace evaluate_polynomial_at_minus_two_l104_104907

def P (x : ℝ) : ℝ := x^3 - 2*x^2 + 3*x + 4

theorem evaluate_polynomial_at_minus_two :
  P (-2) = -18 :=
by
  sorry

end evaluate_polynomial_at_minus_two_l104_104907


namespace leonid_painted_cells_l104_104087

theorem leonid_painted_cells (k l : ℕ) (hkl : k * l = 74) :
  ∃ (painted_cells : ℕ), painted_cells = ((2 * k + 1) * (2 * l + 1) - 74) ∧ (painted_cells = 373 ∨ painted_cells = 301) :=
by
  sorry

end leonid_painted_cells_l104_104087


namespace relationship_P_Q_l104_104026

variable (a : ℝ)
variable (P : ℝ := Real.sqrt a + Real.sqrt (a + 5))
variable (Q : ℝ := Real.sqrt (a + 2) + Real.sqrt (a + 3))

theorem relationship_P_Q (h : 0 ≤ a) : P < Q :=
by
  sorry

end relationship_P_Q_l104_104026


namespace arithmetic_sequence_general_term_sum_of_first_n_terms_l104_104284

theorem arithmetic_sequence_general_term 
  (a_n : ℕ → ℕ) (S : ℕ → ℕ)
  (d : ℕ) (h_d_nonzero : d ≠ 0)
  (h_arith : ∀ n, a_n = a_n 0 + n * d)
  (h_S9 : S 9 = 90)
  (h_geom : ∃ (a1 a2 a4 : ℕ), a2^2 = a1 * a4)
  (h_common_diff : d = a_n 1 - a_n 0)
  : ∀ n, a_n = 2 * n  := 
sorry

theorem sum_of_first_n_terms
  (b_n : ℕ → ℕ)
  (T : ℕ → ℕ)
  (a_n : ℕ → ℕ) 
  (h_b_def : ∀ n, b_n = 1 / (a_n n * a_n (n+1)))
  (h_a_form : ∀ n, a_n = 2 * n)
  : ∀ n, T n = n / (4 * n + 4) :=
sorry

end arithmetic_sequence_general_term_sum_of_first_n_terms_l104_104284


namespace f_eq_four_or_seven_l104_104205

noncomputable def f (a b : ℕ) : ℚ := (a^2 + a * b + b^2) / (a * b - 1)

theorem f_eq_four_or_seven (a b : ℕ) (h : a > 0) (h1 : b > 0) (h2 : a * b ≠ 1) : 
  f a b = 4 ∨ f a b = 7 := 
sorry

end f_eq_four_or_seven_l104_104205


namespace math_problem_l104_104728

theorem math_problem
  (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h : x = z * (1 / y)) : 
  (x - z / x) * (y + 1 / (z * y)) = (x^4 - z^3 + x^2 * (z^2 - z)) / (z * x^2) :=
by
  sorry

end math_problem_l104_104728


namespace find_friends_l104_104475

-- Definitions
def shells_Jillian : Nat := 29
def shells_Savannah : Nat := 17
def shells_Clayton : Nat := 8
def shells_per_friend : Nat := 27

-- Main statement
theorem find_friends :
  (shells_Jillian + shells_Savannah + shells_Clayton) / shells_per_friend = 2 :=
by
  sorry

end find_friends_l104_104475


namespace units_digit_expression_l104_104620

theorem units_digit_expression: 
  (8 * 19 * 1981 + 6^3 - 2^5) % 10 = 6 := 
by
  sorry

end units_digit_expression_l104_104620


namespace one_over_a_lt_one_over_b_iff_ab_over_a3_minus_b3_gt_zero_l104_104350

theorem one_over_a_lt_one_over_b_iff_ab_over_a3_minus_b3_gt_zero
  (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (1 / a < 1 / b) ↔ ((a * b) / (a^3 - b^3) > 0) := 
by
  sorry

end one_over_a_lt_one_over_b_iff_ab_over_a3_minus_b3_gt_zero_l104_104350


namespace hiker_walked_distance_first_day_l104_104742

theorem hiker_walked_distance_first_day (h d_1 d_2 d_3 : ℕ) (H₁ : d_1 = 3 * h)
    (H₂ : d_2 = 4 * (h - 1)) (H₃ : d_3 = 30) (H₄ : d_1 + d_2 + d_3 = 68) :
    d_1 = 18 := 
by 
  sorry

end hiker_walked_distance_first_day_l104_104742


namespace problem_statement_l104_104436

def are_collinear (A B C : Point) : Prop := sorry -- Definition for collinearity should be expanded.
def area (A B C : Point) : ℝ := sorry -- Definition for area must be provided.

theorem problem_statement :
  ∀ n : ℕ, (n > 3) →
  (∃ (A : Fin n → Point) (r : Fin n → ℝ),
    (∀ i j k : Fin n, i ≠ j → j ≠ k → k ≠ i → ¬ are_collinear (A i) (A j) (A k)) ∧
    (∀ i j k : Fin n, area (A i) (A j) (A k) = r i + r j + r k)) →
  n = 4 :=
by sorry

end problem_statement_l104_104436


namespace find_A_satisfy_3A_multiple_of_8_l104_104100

theorem find_A_satisfy_3A_multiple_of_8 (A : ℕ) (h : 0 ≤ A ∧ A < 10) : 8 ∣ (30 + A) ↔ A = 2 := 
by
  sorry

end find_A_satisfy_3A_multiple_of_8_l104_104100


namespace minimum_dimes_needed_l104_104842

theorem minimum_dimes_needed (n : ℕ) 
  (sneaker_cost : ℝ := 58) 
  (ten_bills : ℝ := 50)
  (five_quarters : ℝ := 1.25) :
  ten_bills + five_quarters + (0.10 * n) ≥ sneaker_cost ↔ n ≥ 68 := 
by 
  sorry

end minimum_dimes_needed_l104_104842


namespace cube_sum_equal_one_l104_104047

theorem cube_sum_equal_one (x y z : ℝ) (h1 : x + y + z = 3) (h2 : xy + xz + yz = 1) (h3 : xyz = 1) :
  x^3 + y^3 + z^3 = 1 := 
sorry

end cube_sum_equal_one_l104_104047


namespace sum_infinite_partial_fraction_l104_104392

theorem sum_infinite_partial_fraction :
  ∑' n : ℕ, n > 0 → (3 * n - 2) / (n * (n + 1) * (n + 3)) = 11 / 12 :=
by
  sorry

end sum_infinite_partial_fraction_l104_104392


namespace deficit_calculation_l104_104594

theorem deficit_calculation
    (L W : ℝ)  -- Length and Width
    (dW : ℝ)  -- Deficit in width
    (h1 : (1.08 * L) * (W - dW) = 1.026 * (L * W))  -- Condition on the calculated area
    : dW / W = 0.05 := 
by
    sorry

end deficit_calculation_l104_104594


namespace total_garbage_collected_correct_l104_104069

def Lizzie_group_collected : ℕ := 387
def other_group_collected : ℕ := Lizzie_group_collected - 39
def total_garbage_collected : ℕ := Lizzie_group_collected + other_group_collected

theorem total_garbage_collected_correct :
  total_garbage_collected = 735 :=
sorry

end total_garbage_collected_correct_l104_104069


namespace find_product_l104_104364

theorem find_product (a b c d : ℝ) 
  (h_avg : (a + b + c + d) / 4 = 7.1)
  (h_rel : 2.5 * a = b - 1.2 ∧ b - 1.2 = c + 4.8 ∧ c + 4.8 = 0.25 * d) :
  a * b * c * d = 49.6 := 
sorry

end find_product_l104_104364


namespace group_purchase_cheaper_l104_104961

-- Define the initial conditions
def initial_price : ℕ := 10
def bulk_price : ℕ := 7
def delivery_cost : ℕ := 100
def group_size : ℕ := 50

-- Define the costs for individual and group purchases
def individual_cost : ℕ := initial_price
def group_cost : ℕ := bulk_price + (delivery_cost / group_size)

-- Statement to prove: cost per participant in a group purchase is less than cost per participant in individual purchases
theorem group_purchase_cheaper : group_cost < individual_cost := by
  sorry

end group_purchase_cheaper_l104_104961


namespace find_number_l104_104746

theorem find_number (x a_3 a_4 : ℕ) (h1 : x + a_4 = 5574) (h2 : x + a_3 = 557) : x = 5567 :=
  sorry

end find_number_l104_104746


namespace correct_quadratic_eq_l104_104109

-- Define the given conditions
def first_student_sum (b : ℝ) : Prop := 5 + 3 = -b
def second_student_product (c : ℝ) : Prop := (-12) * (-4) = c

-- Define the proof statement
theorem correct_quadratic_eq (b c : ℝ) (h1 : first_student_sum b) (h2 : second_student_product c) :
    b = -8 ∧ c = 48 ∧ (∀ x : ℝ, x^2 + b * x + c = 0 → (x=5 ∨ x=3 ∨ x=-12 ∨ x=-4)) :=
by
  sorry

end correct_quadratic_eq_l104_104109


namespace not_all_zero_iff_at_least_one_non_zero_l104_104212

theorem not_all_zero_iff_at_least_one_non_zero (a b c : ℝ) : ¬ (a = 0 ∧ b = 0 ∧ c = 0) ↔ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) :=
by
  sorry

end not_all_zero_iff_at_least_one_non_zero_l104_104212


namespace integer_value_of_a_l104_104966

theorem integer_value_of_a (a x y z k : ℤ) :
  (x = k) ∧ (y = 4 * k) ∧ (z = 5 * k) ∧ (y = 9 * a^2 - 2 * a - 8) ∧ (z = 10 * a + 2) → a = 5 :=
by 
  sorry

end integer_value_of_a_l104_104966


namespace min_value_exp_l104_104431

theorem min_value_exp (x y : ℝ) (h : x + 2 * y = 4) : ∃ z : ℝ, (2^x + 4^y = z) ∧ (∀ (a b : ℝ), a + 2 * b = 4 → 2^a + 4^b ≥ z) :=
sorry

end min_value_exp_l104_104431


namespace seat_39_l104_104678

-- Defining the main structure of the problem
def circle_seating_arrangement (n k : ℕ) : ℕ :=
  if k = 1 then 1
  else sorry -- The pattern-based implementation goes here

-- The theorem to state the problem
theorem seat_39 (n k : ℕ) (h_n : n = 128) (h_k : k = 39) :
  circle_seating_arrangement n k = 51 :=
sorry

end seat_39_l104_104678


namespace complement_intersection_l104_104444

universe u

-- Define the universal set U, and sets A and B
def U : Set ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
def A : Set ℕ := {0, 1, 3, 5, 8}
def B : Set ℕ := {2, 4, 5, 6, 8}

-- Define the complements of A and B with respect to U
def complement_U (s : Set ℕ) := { x ∈ U | x ∉ s }

-- The theorem to prove the intersection of the complements
theorem complement_intersection :
  (complement_U A) ∩ (complement_U B) = {7, 9} :=
sorry

end complement_intersection_l104_104444


namespace number_of_three_digit_integers_congruent_to_2_mod_4_l104_104612

theorem number_of_three_digit_integers_congruent_to_2_mod_4 : 
  ∃ (count : ℕ), count = 225 ∧ ∀ (n : ℕ), 100 ≤ n ∧ n ≤ 999 ∧ n % 4 = 2 ↔ (∃ k : ℕ, 25 ≤ k ∧ k ≤ 249 ∧ n = 4 * k + 2) := 
by {
  sorry
}

end number_of_three_digit_integers_congruent_to_2_mod_4_l104_104612


namespace part_i_part_ii_l104_104574

noncomputable def f (x a : ℝ) : ℝ := Real.exp x + x - a
noncomputable def g (x a : ℝ) : ℝ := Real.sqrt (f x a)

theorem part_i (a : ℝ) :
  (∀ x ∈ Set.Icc (0:ℝ) (1:ℝ), f x a ≥ 0) ↔ (a ≤ 1) :=
by {
  -- Suppose it is already known that theorem is true.
  sorry
}

theorem part_ii (a : ℝ) :
  (∃ x0 y0 : ℝ, (x0, y0) ∈ (Set.Icc (-1) 1) ∧ y0 = Real.cos (2 * x0) ∧ g (g y0 a) a = y0) ↔ (1 ≤ a ∧ a ≤ Real.exp 1) :=
by {
  -- Suppose it is already known that theorem is true.
  sorry
}

end part_i_part_ii_l104_104574


namespace max_value_of_f_l104_104116

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2

theorem max_value_of_f : ∀ x ∈ Set.Icc (-1 : ℝ) 1, f x ≤ 2 := 
by
  sorry

end max_value_of_f_l104_104116


namespace product_two_digit_numbers_l104_104211

theorem product_two_digit_numbers (a b : ℕ) (ha : 10 ≤ a ∧ a < 100) (hb : 10 ≤ b ∧ b < 100) (h : a * b = 777) : (a = 21 ∧ b = 37) ∨ (a = 37 ∧ b = 21) := 
  sorry

end product_two_digit_numbers_l104_104211


namespace compare_fractions_l104_104544

theorem compare_fractions : (- (4 / 5) < - (2 / 3)) :=
by
  sorry

end compare_fractions_l104_104544


namespace mixed_fractions_calculation_l104_104443

theorem mixed_fractions_calculation :
  2017 + (2016 / 2017) / (2019 + (1 / 2016)) + (1 / 2017) = 1 :=
by
  sorry

end mixed_fractions_calculation_l104_104443


namespace yura_finishes_on_correct_date_l104_104689

-- Let there be 91 problems in the textbook.
-- Let Yura start solving problems on September 6th.
def total_problems : Nat := 91
def start_date : Nat := 6

-- Each morning, starting from September 7, he solves one problem less than the previous morning.
def problems_solved (n : Nat) : Nat := if n = 0 then 0 else problems_solved (n - 1) - 1

-- On the evening of September 8, there are 46 problems left to solve.
def remaining_problems_sept_8 : Nat := 46

-- The question is to find on which date Yura will finish solving the textbook
def finish_date : Nat := 12

theorem yura_finishes_on_correct_date : ∃ z : Nat, (problems_solved z * 3 = total_problems - remaining_problems_sept_8) ∧ (z = 15) ∧ finish_date = 12 := by sorry

end yura_finishes_on_correct_date_l104_104689


namespace nap_hours_in_70_days_l104_104064

-- Define the variables and conditions
variable (n d a b c e : ℕ)  -- assuming they are natural numbers

-- Define the total nap hours function
noncomputable def total_nap_hours (n d a b c e : ℕ) : ℕ :=
  (a + b) * 10

-- The statement to prove
theorem nap_hours_in_70_days (n d a b c e : ℕ) :
  total_nap_hours n d a b c e = (a + b) * 10 :=
by sorry

end nap_hours_in_70_days_l104_104064


namespace find_number_of_pairs_l104_104661

variable (n : ℕ)
variable (prob_same_color : ℚ := 0.09090909090909091)
variable (total_shoes : ℕ := 12)
variable (pairs_of_shoes : ℕ)

-- The condition on the probability of selecting two shoes of the same color
def condition_probability : Prop :=
  (1 : ℚ) / ((2 * n - 1) : ℚ) = prob_same_color

-- The condition on the total number of shoes
def condition_total_shoes : Prop :=
  2 * n = total_shoes

-- The goal to prove that the number of pairs of shoes is 6 given the conditions
theorem find_number_of_pairs (h1 : condition_probability n) (h2 : condition_total_shoes n) : n = 6 :=
by
  sorry

end find_number_of_pairs_l104_104661


namespace alice_leaves_30_minutes_after_bob_l104_104557

theorem alice_leaves_30_minutes_after_bob :
  ∀ (distance : ℝ) (speed_bob : ℝ) (speed_alice : ℝ) (time_diff : ℝ),
  distance = 220 ∧ speed_bob = 40 ∧ speed_alice = 44 ∧ 
  time_diff = (distance / speed_bob) - (distance / speed_alice) →
  (time_diff * 60 = 30) := by
  intro distance speed_bob speed_alice time_diff
  intro h
  have h1 : distance = 220 := h.1
  have h2 : speed_bob = 40 := h.2.1
  have h3 : speed_alice = 44 := h.2.2.1
  have h4 : time_diff = (distance / speed_bob) - (distance / speed_alice) := h.2.2.2
  sorry

end alice_leaves_30_minutes_after_bob_l104_104557


namespace GCF_LCM_computation_l104_104691

-- Definitions and axioms we need
def GCF (a b : ℕ) : ℕ := Nat.gcd a b
def LCM (a b : ℕ) : ℕ := Nat.lcm a b

-- The theorem to prove
theorem GCF_LCM_computation : GCF (LCM 8 14) (LCM 7 12) = 28 :=
by sorry

end GCF_LCM_computation_l104_104691


namespace car_travel_distance_l104_104922

noncomputable def distance_in_miles (b t : ℝ) : ℝ :=
  (25 * b) / (1320 * t)

theorem car_travel_distance (b t : ℝ) : 
  let distance_in_feet := (b / 3) * (300 / t)
  let distance_in_miles' := distance_in_feet / 5280
  distance_in_miles' = distance_in_miles b t := 
by
  sorry

end car_travel_distance_l104_104922


namespace remainder_of_3042_div_98_l104_104075

theorem remainder_of_3042_div_98 : 3042 % 98 = 4 := 
by
  sorry

end remainder_of_3042_div_98_l104_104075


namespace snowboard_final_price_l104_104194

noncomputable def original_price : ℝ := 200
noncomputable def discount_friday : ℝ := 0.40
noncomputable def discount_monday : ℝ := 0.25

noncomputable def price_after_friday_discount (orig : ℝ) (discount : ℝ) : ℝ :=
  (1 - discount) * orig

noncomputable def final_price (price_friday : ℝ) (discount : ℝ) : ℝ :=
  (1 - discount) * price_friday

theorem snowboard_final_price :
  final_price (price_after_friday_discount original_price discount_friday) discount_monday = 90 := 
sorry

end snowboard_final_price_l104_104194


namespace factorize_expr_l104_104150

theorem factorize_expr (a b : ℝ) : a * b^2 - 8 * a * b + 16 * a = a * (b - 4)^2 := 
by
  sorry

end factorize_expr_l104_104150


namespace men_left_hostel_l104_104314

-- Definitions based on the conditions given
def initialMen : ℕ := 250
def initialDays : ℕ := 28
def remainingDays : ℕ := 35

-- The theorem we need to prove
theorem men_left_hostel (x : ℕ) (h : initialMen * initialDays = (initialMen - x) * remainingDays) : x = 50 :=
by
  sorry

end men_left_hostel_l104_104314


namespace sum_of_u_and_v_l104_104973

theorem sum_of_u_and_v (u v : ℤ) (h1 : 1 ≤ v) (h2 : v < u) (h3 : u^2 + v^2 = 500) : u + v = 20 := by
  sorry

end sum_of_u_and_v_l104_104973


namespace time_upstream_is_correct_l104_104034

-- Define the conditions
def speed_of_stream : ℝ := 3
def speed_in_still_water : ℝ := 15
def downstream_time : ℝ := 1
def downstream_speed : ℝ := speed_in_still_water + speed_of_stream
def distance_downstream : ℝ := downstream_speed * downstream_time
def upstream_speed : ℝ := speed_in_still_water - speed_of_stream

-- Theorem statement
theorem time_upstream_is_correct :
  (distance_downstream / upstream_speed) = 1.5 := by
  sorry

end time_upstream_is_correct_l104_104034


namespace average_annual_growth_rate_l104_104074

-- Define the conditions
def revenue_current_year : ℝ := 280
def revenue_planned_two_years : ℝ := 403.2

-- Define the growth equation
def growth_equation (x : ℝ) : Prop :=
  revenue_current_year * (1 + x)^2 = revenue_planned_two_years

-- State the theorem
theorem average_annual_growth_rate : ∃ x : ℝ, growth_equation x ∧ x = 0.2 := by
  sorry

end average_annual_growth_rate_l104_104074


namespace number_is_2_point_5_l104_104811

theorem number_is_2_point_5 (x : ℝ) (h: x^2 + 50 = (x - 10)^2) : x = 2.5 := 
by
  sorry

end number_is_2_point_5_l104_104811


namespace union_sets_l104_104115

open Set

variable {α : Type*}

def A : Set ℝ := {x | -2 < x ∧ x < 2}

def B : Set ℝ := {y | ∃ x, x ∈ A ∧ y = 2^x}

theorem union_sets : A ∪ B = {z | -2 < z ∧ z < 4} :=
by sorry

end union_sets_l104_104115


namespace smallest_m_satisfying_conditions_l104_104642

theorem smallest_m_satisfying_conditions :
  ∃ m : ℕ, m = 4 ∧ (∃ k : ℕ, 0 ≤ k ∧ k ≤ m ∧ (m^2 + m) % k ≠ 0) ∧ (∀ k : ℕ, (0 ≤ k ∧ k ≤ m) → (k ≠ 0 → (m^2 + m) % k = 0)) :=
sorry

end smallest_m_satisfying_conditions_l104_104642


namespace cos_sin_fraction_l104_104761

theorem cos_sin_fraction (α β : ℝ) (h1 : Real.tan (α + β) = 2 / 5) 
                         (h2 : Real.tan (β - Real.pi / 4) = 1 / 4) :
  (Real.cos α + Real.sin α) / (Real.cos α - Real.sin α) = 3 / 22 := 
  sorry

end cos_sin_fraction_l104_104761


namespace smallest_solution_eq_l104_104872

theorem smallest_solution_eq (x : ℝ) (h : 1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) :
  x = 4 - Real.sqrt 2 := 
  sorry

end smallest_solution_eq_l104_104872


namespace intersection_of_A_and_B_l104_104421

open Set

theorem intersection_of_A_and_B (A B : Set ℕ) (hA : A = {1, 2, 4}) (hB : B = {2, 4, 6}) : A ∩ B = {2, 4} :=
by
  rw [hA, hB]
  apply Set.ext
  intro x
  simp
  sorry

end intersection_of_A_and_B_l104_104421


namespace pentagon_area_l104_104988

/-- This Lean statement represents the problem of finding the y-coordinate of vertex C
    in a pentagon with given vertex positions and specific area constraint. -/
theorem pentagon_area (y : ℝ) 
  (h_sym : true) -- The pentagon ABCDE has a vertical line of symmetry
  (h_A : (0, 0) = (0, 0)) -- A(0,0)
  (h_B : (0, 5) = (0, 5)) -- B(0, 5)
  (h_C : (3, y) = (3, y)) -- C(3, y)
  (h_D : (6, 5) = (6, 5)) -- D(6, 5)
  (h_E : (6, 0) = (6, 0)) -- E(6, 0)
  (h_area : 50 = 50) -- The total area of the pentagon is 50 square units
  : y = 35 / 3 :=
sorry

end pentagon_area_l104_104988


namespace roots_of_equation_l104_104356

theorem roots_of_equation (a x : ℝ) : x * (x + 5)^2 * (a - x) = 0 ↔ (x = 0 ∨ x = -5 ∨ x = a) :=
by
  sorry

end roots_of_equation_l104_104356


namespace roberts_monthly_expenses_l104_104305

-- Conditions
def basic_salary : ℝ := 1250
def commission_rate : ℝ := 0.1
def total_sales : ℝ := 23600
def savings_rate : ℝ := 0.2

-- Definitions derived from the conditions
noncomputable def commission : ℝ := commission_rate * total_sales
noncomputable def total_earnings : ℝ := basic_salary + commission
noncomputable def savings : ℝ := savings_rate * total_earnings
noncomputable def monthly_expenses : ℝ := total_earnings - savings

-- The statement to be proved
theorem roberts_monthly_expenses : monthly_expenses = 2888 := by
  sorry

end roberts_monthly_expenses_l104_104305


namespace evaluate_expression_l104_104513

theorem evaluate_expression : 
  (5^1001 + 6^1002)^2 - (5^1001 - 6^1002)^2 = 24 * 30^1001 :=
by
  sorry

end evaluate_expression_l104_104513


namespace sum_ages_l104_104956

theorem sum_ages (x : ℕ) (h_triple : True) (h_sons_age : ∀ a, a ∈ [16, 16, 16]) (h_beau_age : 42 = 42) :
  3 * (16 - x) = 42 - x → x = 3 := by
  sorry

end sum_ages_l104_104956


namespace geometric_sequence_problem_l104_104666

noncomputable def geometric_sequence_sum_condition 
  (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (a 1 + a 2 + a 3 + a 4 + a 5 = 6) ∧ 
  (a 1 ^ 2 + a 2 ^ 2 + a 3 ^ 2 + a 4 ^ 2 + a 5 ^ 2 = 18) ∧ 
  (∀ n, a n = a 1 * q ^ (n - 1)) ∧ 
  (q ≠ 1)

theorem geometric_sequence_problem 
  (a : ℕ → ℝ) (q : ℝ) 
  (h : geometric_sequence_sum_condition a q) : 
  a 1 - a 2 + a 3 - a 4 + a 5 = 3 := 
by 
  sorry

end geometric_sequence_problem_l104_104666


namespace ribbon_each_box_fraction_l104_104986

-- Define the conditions
def total_ribbon_used : ℚ := 5 / 12
def number_of_boxes : ℕ := 5
def ribbon_per_box : ℚ := total_ribbon_used / number_of_boxes

-- Statement to be proved
theorem ribbon_each_box_fraction :
  ribbon_per_box = 1 / 12 :=
  sorry

end ribbon_each_box_fraction_l104_104986


namespace rational_expr_evaluation_l104_104457

theorem rational_expr_evaluation (a b c : ℚ) (h1 : 0 < a ∧ 0 < b ∧ 0 < c) (h2 : a + b + c = a * b * c) :
  (a / b + a / c + b / a + b / c + c / a + c / b - a * b - b * c - c * a) = -3 :=
by
  sorry

end rational_expr_evaluation_l104_104457


namespace larger_number_of_product_and_sum_l104_104031

theorem larger_number_of_product_and_sum (x y : ℕ) (h_prod : x * y = 35) (h_sum : x + y = 12) : max x y = 7 :=
by {
  sorry
}

end larger_number_of_product_and_sum_l104_104031


namespace combination_10_3_l104_104718

theorem combination_10_3 : Nat.choose 10 3 = 120 := by
  -- use the combination formula: \binom{n}{r} = n! / (r! * (n-r)!)
  sorry

end combination_10_3_l104_104718


namespace minimum_value_expression_l104_104469

theorem minimum_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a = 1 ∧ b = 1 ∧ c = 1) :
  (a^2 + 4 * a + 2) * (b^2 + 4 * b + 2) * (c^2 + 4 * c + 2) / (a * b * c) = 48 * Real.sqrt 6 := 
by
  sorry

end minimum_value_expression_l104_104469


namespace length_of_bridge_is_correct_l104_104824

def length_of_bridge (train_length : ℕ) (train_speed_kmh : ℕ) (crossing_time_s : ℕ) : ℕ :=
  let train_speed_ms := (train_speed_kmh * 1000) / 3600
  let total_distance := train_speed_ms * crossing_time_s
  total_distance - train_length

theorem length_of_bridge_is_correct : 
  length_of_bridge 170 45 30 = 205 :=
by
  -- we state the translation and prove here (proof omitted, just the structure is present)
  sorry

end length_of_bridge_is_correct_l104_104824


namespace range_of_x_l104_104965

noncomputable def f (x : ℝ) : ℝ := sorry -- Define f to satisfy given conditions later

theorem range_of_x (hf_odd : ∀ x : ℝ, f (-x) = - f x)
                   (hf_inc_mono_neg : ∀ x y : ℝ, x ≤ y → y ≤ 0 → f x ≤ f y)
                   (h_ineq : f 1 + f (Real.log x - 2) < 0) : (0 < x) ∧ (x < 10) :=
by
  sorry

end range_of_x_l104_104965


namespace terminal_side_in_fourth_quadrant_l104_104647

theorem terminal_side_in_fourth_quadrant (θ : ℝ) (h1 : Real.cos θ > 0) (h2 : Real.sin (2 * θ) < 0) : 
  (θ ≥ 0 ∧ θ < Real.pi/2) ∨ (θ > 3 * Real.pi / 2 ∧ θ < 2 * Real.pi) :=
sorry

end terminal_side_in_fourth_quadrant_l104_104647


namespace clara_hardcover_books_l104_104312

-- Define the variables and conditions
variables (h p : ℕ)

-- Conditions based on the problem statement
def volumes_total : Prop := h + p = 12
def total_cost (total : ℕ) : Prop := 28 * h + 18 * p = total

-- The theorem to prove
theorem clara_hardcover_books (h p : ℕ) (H1 : volumes_total h p) (H2 : total_cost h p 270) : h = 6 :=
by
  sorry

end clara_hardcover_books_l104_104312


namespace haley_deleted_pictures_l104_104905

variable (zoo_pictures : ℕ) (museum_pictures : ℕ) (remaining_pictures : ℕ) (deleted_pictures : ℕ)

theorem haley_deleted_pictures :
  zoo_pictures = 50 → museum_pictures = 8 → remaining_pictures = 20 →
  deleted_pictures = zoo_pictures + museum_pictures - remaining_pictures →
  deleted_pictures = 38 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  simp at h4
  exact h4

end haley_deleted_pictures_l104_104905


namespace solve_for_F_l104_104345

theorem solve_for_F (C F : ℝ) (h1 : C = 5 / 9 * (F - 32)) (h2 : C = 40) : F = 104 :=
by
  sorry

end solve_for_F_l104_104345


namespace find_k_l104_104570

-- Given definition for a quadratic expression that we want to be a square of a binomial
def quadratic_expression (x k : ℝ) := x^2 - 20 * x + k

-- The binomial square matching.
def binomial_square (x b : ℝ) := (x + b)^2

-- Statement to prove that k = 100 makes the quadratic_expression to be a square of binomial
theorem find_k :
  (∃ k : ℝ, ∀ x : ℝ, quadratic_expression x k = binomial_square x (-10)) ↔ k = 100 :=
by
  sorry

end find_k_l104_104570


namespace eval_expression_l104_104464

theorem eval_expression : (20 - 16) * (12 + 8) / 4 = 20 := 
by 
  sorry

end eval_expression_l104_104464


namespace math_problem_l104_104575

theorem math_problem (x y z : ℤ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : (x + y * Complex.I)^2 - 46 * Complex.I = z) :
  x + y + z = 552 :=
by
  sorry

end math_problem_l104_104575


namespace sin_theta_plus_45_l104_104857

-- Statement of the problem in Lean 4

theorem sin_theta_plus_45 (θ : ℝ) (h : 0 < θ ∧ θ < π / 2) (sin_θ_eq : Real.sin θ = 3 / 5) :
  Real.sin (θ + π / 4) = 7 * Real.sqrt 2 / 10 :=
sorry

end sin_theta_plus_45_l104_104857


namespace smallest_sum_of_four_distinct_numbers_l104_104607

theorem smallest_sum_of_four_distinct_numbers 
  (S : Finset ℤ) 
  (h : S = {8, 26, -2, 13, -4, 0}) :
  ∃ (a b c d : ℤ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ a + b + c + d = 2 :=
sorry

end smallest_sum_of_four_distinct_numbers_l104_104607


namespace factorize_expr1_factorize_expr2_l104_104558

theorem factorize_expr1 (x y : ℝ) : 
  3 * (x + y) * (x - y) - (x - y)^2 = 2 * (x - y) * (x + 2 * y) :=
by
  sorry

theorem factorize_expr2 (x y : ℝ) : 
  x^2 * (y^2 - 1) + 2 * x * (y^2 - 1) = x * (y + 1) * (y - 1) * (x + 2) :=
by
  sorry

end factorize_expr1_factorize_expr2_l104_104558


namespace largest_integer_less_than_120_with_remainder_5_div_8_l104_104487

theorem largest_integer_less_than_120_with_remainder_5_div_8 :
  ∃ n : ℤ, n < 120 ∧ n % 8 = 5 ∧ ∀ m : ℤ, m < 120 → m % 8 = 5 → m ≤ n :=
sorry

end largest_integer_less_than_120_with_remainder_5_div_8_l104_104487


namespace average_bracelets_per_day_l104_104081

theorem average_bracelets_per_day
  (cost_of_bike : ℕ)
  (price_per_bracelet : ℕ)
  (weeks : ℕ)
  (days_per_week : ℕ)
  (h1 : cost_of_bike = 112)
  (h2 : price_per_bracelet = 1)
  (h3 : weeks = 2)
  (h4 : days_per_week = 7) :
  (cost_of_bike / price_per_bracelet) / (weeks * days_per_week) = 8 :=
by
  sorry

end average_bracelets_per_day_l104_104081


namespace sum_of_x_coordinates_where_g_eq_2_5_l104_104414

def g1 (x : ℝ) : ℝ := 3 * x + 6
def g2 (x : ℝ) : ℝ := -x + 2
def g3 (x : ℝ) : ℝ := 2 * x - 2
def g4 (x : ℝ) : ℝ := -2 * x + 8

def is_within (x : ℝ) (a b : ℝ) : Prop := a ≤ x ∧ x ≤ b

theorem sum_of_x_coordinates_where_g_eq_2_5 :
     (∀ x, g1 x = 2.5 → (is_within x (-4) (-2) → false)) ∧
     (∀ x, g2 x = 2.5 → (is_within x (-2) (0) → x = -0.5)) ∧
     (∀ x, g3 x = 2.5 → (is_within x 0 3 → x = 2.25)) ∧
     (∀ x, g4 x = 2.5 → (is_within x 3 5 → x = 2.75)) →
     (-0.5 + 2.25 + 2.75 = 4.5) :=
by { sorry }

end sum_of_x_coordinates_where_g_eq_2_5_l104_104414


namespace maximum_range_of_temperatures_l104_104769

variable (T1 T2 T3 T4 T5 : ℝ)

-- Given conditions
def average_condition : Prop := (T1 + T2 + T3 + T4 + T5) / 5 = 50
def lowest_temperature_condition : Prop := T1 = 45

-- Question to prove
def possible_maximum_range : Prop := T5 - T1 = 25

-- The final theorem statement
theorem maximum_range_of_temperatures 
  (h_avg : average_condition T1 T2 T3 T4 T5) 
  (h_lowest : lowest_temperature_condition T1) 
  : possible_maximum_range T1 T5 := by
  sorry

end maximum_range_of_temperatures_l104_104769


namespace parallel_lines_condition_iff_l104_104613

def line_parallel (a : ℝ) : Prop :=
  let l1_slope := -1 / -a
  let l2_slope := -(a - 1) / -12
  l1_slope = l2_slope

theorem parallel_lines_condition_iff (a : ℝ) :
  (a = 4) ↔ line_parallel a := by
  sorry

end parallel_lines_condition_iff_l104_104613


namespace financing_term_years_l104_104653

def monthly_payment : Int := 150
def total_financed_amount : Int := 9000

theorem financing_term_years : 
  (total_financed_amount / monthly_payment) / 12 = 5 := 
by
  sorry

end financing_term_years_l104_104653


namespace pizza_cost_per_slice_correct_l104_104251

noncomputable def pizza_cost_per_slice : ℝ :=
  let base_pizza_cost := 10.00
  let first_topping_cost := 2.00
  let next_two_toppings_cost := 2.00
  let remaining_toppings_cost := 2.00
  let total_cost := base_pizza_cost + first_topping_cost + next_two_toppings_cost + remaining_toppings_cost
  total_cost / 8

theorem pizza_cost_per_slice_correct :
  pizza_cost_per_slice = 2.00 :=
by
  unfold pizza_cost_per_slice
  sorry

end pizza_cost_per_slice_correct_l104_104251


namespace coloring_ways_l104_104859

-- Define a factorial function
def factorial : Nat → Nat
| 0       => 1
| (n + 1) => (n + 1) * factorial n

-- Define a derangement function
def derangement : Nat → Nat
| 0       => 1
| 1       => 0
| (n + 1) => n * (derangement n + derangement (n - 1))

-- Prove the main theorem
theorem coloring_ways : 
  let six_factorial := factorial 6
  let derangement_6 := derangement 6
  let derangement_5 := derangement 5
  720 * (derangement_6 + derangement_5) = 222480 := by
    let six_factorial := 720
    let derangement_6 := derangement 6
    let derangement_5 := derangement 5
    show six_factorial * (derangement_6 + derangement_5) = 222480
    sorry

end coloring_ways_l104_104859


namespace min_value_of_m_l104_104357

open Real

-- Definitions from the conditions
def condition1 (m : ℝ) : Prop :=
  m > 0

def condition2 (m : ℝ) : Prop :=
  ∀ (x : ℝ), 1 < x → 2 * exp (2 * m * x) - (log x) / m ≥ 0

-- The theorem statement for the minimum value of m
theorem min_value_of_m (m : ℝ) : condition1 m → condition2 m → m ≥ 1 / (2 * exp 1) := 
sorry

end min_value_of_m_l104_104357


namespace incorrect_statement_D_l104_104918

noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2

theorem incorrect_statement_D :
  (∃ T : ℝ, ∀ x : ℝ, f (x + T) = f x) ∧
  (∀ x : ℝ, f (π / 2 + x) = f (π / 2 - x)) ∧
  (f (π / 2 + π / 4) = 0) ∧ ¬(∀ x : ℝ, (π / 2 < x ∧ x < π) → f x < f (x - 0.1)) := by
  sorry

end incorrect_statement_D_l104_104918


namespace line_l_statements_correct_l104_104446

theorem line_l_statements_correct
  (A B C : ℝ)
  (hAB : ¬(A = 0 ∧ B = 0)) :
  ( (2 * A + B + C = 0 → ∀ x y, A * (x - 2) + B * (y - 1) = 0 ↔ A * x + B * y + C = 0 ) ∧
    ((A ≠ 0 ∧ B ≠ 0) → ∃ x, A * x + C = 0 ∧ ∃ y, B * y + C = 0) ∧
    (A = 0 ∧ B ≠ 0 ∧ C ≠ 0 → ∀ y, B * y + C = 0 ↔ y = -C / B) ∧
    (A ≠ 0 ∧ B^2 + C^2 = 0 → ∀ x, A * x = 0 ↔ x = 0) ) :=
by
  sorry

end line_l_statements_correct_l104_104446


namespace savanna_more_giraffes_l104_104631

-- Definitions based on conditions
def lions_safari := 100
def snakes_safari := lions_safari / 2
def giraffes_safari := snakes_safari - 10

def lions_savanna := 2 * lions_safari
def snakes_savanna := 3 * snakes_safari

-- Totals given and to calculate giraffes in Savanna
def total_animals_savanna := 410

-- Prove that Savanna has 20 more giraffes than Safari
theorem savanna_more_giraffes :
  ∃ (giraffes_savanna : ℕ), giraffes_savanna = total_animals_savanna - lions_savanna - snakes_savanna ∧
  giraffes_savanna - giraffes_safari = 20 :=
  by
  sorry

end savanna_more_giraffes_l104_104631


namespace range_of_z_l104_104560

theorem range_of_z (x y : ℝ) 
  (h1 : x + 2 ≥ y) 
  (h2 : x + 2 * y ≥ 4) 
  (h3 : y ≤ 5 - 2 * x) : 
  ∃ (z_min z_max : ℝ), 
    (z_min = 1) ∧ 
    (z_max = 2) ∧ 
    (∀ z, z = (2 * x + y - 1) / (x + 1) → z_min ≤ z ∧ z ≤ z_max) :=
by
  sorry

end range_of_z_l104_104560


namespace number_of_real_zeros_l104_104732

def f (x : ℝ) : ℝ := x^3 - x^2 - x + 2

theorem number_of_real_zeros : ∃! x : ℝ, f x = 0 := sorry

end number_of_real_zeros_l104_104732


namespace find_m_value_l104_104267

theorem find_m_value 
  (h : ∀ x y m : ℝ, 2*x + y + m = 0 → (1 : ℝ)*x + (-2 : ℝ)*y + 0 = 0)
  (h_circle : ∀ x y : ℝ, x^2 + y^2 - 2*x + 4*y = 0) :
  ∃ m : ℝ, m = 0 :=
sorry

end find_m_value_l104_104267


namespace find_g_neg_2_l104_104402

-- Definitions
variable {R : Type*} [CommRing R] [Inhabited R]
variable (f g : R → R)

-- Conditions
axiom odd_y (x : R) : f (-x) + 2 * x^2 = -(f x + 2 * x^2)
axiom definition_g (x : R) : g x = f x + 1
axiom value_f_2 : f 2 = 2

-- Goal
theorem find_g_neg_2 : g (-2) = -17 :=
by
  sorry

end find_g_neg_2_l104_104402


namespace acute_triangle_probability_correct_l104_104801

noncomputable def acute_triangle_probability : ℝ :=
  let l_cube_vol := 1
  let quarter_cone_vol := (1/4) * (1/3) * Real.pi * (1^2) * 1
  let total_unfavorable_vol := 3 * quarter_cone_vol
  let favorable_vol := l_cube_vol - total_unfavorable_vol
  favorable_vol / l_cube_vol

theorem acute_triangle_probability_correct : abs (acute_triangle_probability - 0.2146) < 0.0001 :=
  sorry

end acute_triangle_probability_correct_l104_104801


namespace original_number_input_0_2_l104_104448

theorem original_number_input_0_2 (x : ℝ) (hx : x ≠ 0) (h : (1 / (1 / x - 1) - 1 = -0.75)) : x = 0.2 := 
sorry

end original_number_input_0_2_l104_104448


namespace necessary_but_not_sufficient_converse_implies_l104_104450

theorem necessary_but_not_sufficient (x : ℝ) (hx1 : 1 < x) (hx2 : x < Real.exp 1) : 
  (x * (Real.log x) ^ 2 < 1) → (x * Real.log x < 1) :=
sorry

theorem converse_implies (x : ℝ) (hx1 : 1 < x) (hx2 : x < Real.exp 1) : 
  (x * Real.log x < 1) → (x * (Real.log x) ^ 2 < 1) :=
sorry

end necessary_but_not_sufficient_converse_implies_l104_104450


namespace sin_zero_range_valid_m_l104_104178

noncomputable def sin_zero_range (m : ℝ) : Prop :=
  ∀ f : ℝ → ℝ, 
    (∀ x : ℝ, f x = Real.sin (2 * x - Real.pi / 6) - m) →
    (∃ x1 x2 : ℝ, (0 ≤ x1 ∧ x1 ≤ Real.pi / 2) ∧ (0 ≤ x2 ∧ x2 ≤ Real.pi / 2) ∧ x1 ≠ x2 ∧ f x1 = 0 ∧ f x2 = 0)

theorem sin_zero_range_valid_m : 
  ∀ m : ℝ, sin_zero_range m ↔ (1 / 2 ≤ m ∧ m < 1) :=
sorry

end sin_zero_range_valid_m_l104_104178


namespace problem_statement_l104_104798

def setA : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 3}
def setB : Set ℝ := {x : ℝ | x ≥ 2}
def setC (a : ℝ) : Set ℝ := {x : ℝ | 2 * x + a ≥ 0}

theorem problem_statement (a : ℝ):
  (setA ∩ setB = {x : ℝ | 2 ≤ x ∧ x < 3}) ∧ 
  (setA ∪ setB = {x : ℝ | x ≥ -1}) ∧ 
  (setB ⊆ setC a → a > -4) :=
by
  sorry

end problem_statement_l104_104798


namespace staircase_toothpicks_l104_104706

theorem staircase_toothpicks (a : ℕ) (r : ℕ) (n : ℕ) :
  a = 9 ∧ r = 3 ∧ n = 3 + 4 
  → (a * r ^ 3 + a * r ^ 2 + a * r + a) + (a * r ^ 2 + a * r + a) + (a * r + a) + a = 351 :=
by
  sorry

end staircase_toothpicks_l104_104706


namespace average_of_xyz_l104_104441

theorem average_of_xyz (x y z : ℝ) (h : (5 / 4) * (x + y + z) = 15) : 
  (x + y + z) / 3 = 4 :=
sorry

end average_of_xyz_l104_104441


namespace total_books_left_l104_104400

def sandy_books : ℕ := 10
def tim_books : ℕ := 33
def benny_lost_books : ℕ := 24

theorem total_books_left : sandy_books + tim_books - benny_lost_books = 19 :=
by
  sorry

end total_books_left_l104_104400


namespace special_op_eight_four_l104_104381

def special_op (a b : ℕ) : ℕ := 2 * a + a / b

theorem special_op_eight_four : special_op 8 4 = 18 := by
  sorry

end special_op_eight_four_l104_104381


namespace trapezoid_circle_tangent_ratio_l104_104976

/-- Given trapezoid EFGH with specified side lengths,
    where EF is parallel to GH, and a circle with
    center Q on EF tangent to FG and HE,
    the ratio EQ : QF is 12 : 37. -/
theorem trapezoid_circle_tangent_ratio :
  ∀ (EF FG GH HE : ℝ) (EQ QF : ℝ),
  EF = 40 → FG = 25 → GH = 12 → HE = 35 →
  ∃ (Q : ℝ) (EQ QF : ℝ),
  EQ + QF = EF ∧ EQ / QF = 12 / 37 ∧ gcd 12 37 = 1 :=
by
  sorry

end trapezoid_circle_tangent_ratio_l104_104976


namespace more_bottle_caps_than_wrappers_l104_104112

namespace DannyCollection

def bottle_caps_found := 50
def wrappers_found := 46

theorem more_bottle_caps_than_wrappers :
  bottle_caps_found - wrappers_found = 4 :=
by
  -- We skip the proof here with "sorry"
  sorry

end DannyCollection

end more_bottle_caps_than_wrappers_l104_104112


namespace quadratic_has_non_real_roots_l104_104983

theorem quadratic_has_non_real_roots (c : ℝ) (h : c > 16) :
    ∃ (a b : ℂ), (x^2 - 8 * x + c = 0) = (a * a = -1) ∧ (b * b = -1) :=
sorry

end quadratic_has_non_real_roots_l104_104983


namespace number_of_figures_l104_104898

theorem number_of_figures (num_squares num_rectangles : ℕ) 
  (h1 : 8 * 8 / 4 = num_squares + num_rectangles) 
  (h2 : 2 * 54 + 4 * 8 = 8 * num_squares + 10 * num_rectangles) :
  num_squares = 10 ∧ num_rectangles = 6 :=
sorry

end number_of_figures_l104_104898


namespace tom_shirts_total_cost_l104_104909

theorem tom_shirts_total_cost 
  (num_tshirts_per_fandom : ℕ)
  (num_fandoms : ℕ)
  (cost_per_shirt : ℕ)
  (discount_rate : ℚ)
  (tax_rate : ℚ)
  (total_shirts : ℕ := num_tshirts_per_fandom * num_fandoms)
  (discount_per_shirt : ℚ := (cost_per_shirt : ℚ) * discount_rate)
  (cost_per_shirt_after_discount : ℚ := (cost_per_shirt : ℚ) - discount_per_shirt)
  (total_cost_before_tax : ℚ := (total_shirts * cost_per_shirt_after_discount))
  (tax_added : ℚ := total_cost_before_tax * tax_rate)
  (total_amount_paid : ℚ := total_cost_before_tax + tax_added)
  (h1 : num_tshirts_per_fandom = 5)
  (h2 : num_fandoms = 4)
  (h3 : cost_per_shirt = 15) 
  (h4 : discount_rate = 0.2)
  (h5 : tax_rate = 0.1)
  : total_amount_paid = 264 := 
by 
  sorry

end tom_shirts_total_cost_l104_104909


namespace a3_5a6_value_l104_104520

variable {a : ℕ → ℤ}

-- Conditions: The sequence {a_n} is an arithmetic sequence, and a_4 + a_7 = 19
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

axiom a_seq_arithmetic : is_arithmetic_sequence a
axiom a4_a7_sum : a 4 + a 7 = 19

-- Problem statement: Prove that a_3 + 5a_6 = 57
theorem a3_5a6_value : a 3 + 5 * a 6 = 57 :=
by
  -- Proof goes here
  sorry

end a3_5a6_value_l104_104520


namespace no_solution_a_solution_b_l104_104130

def f (n : ℕ) : ℕ :=
  if n = 0 then
    0
  else
    n / 7 + f (n / 7)

theorem no_solution_a :
  ¬ ∃ n : ℕ, 7 ^ 399 ∣ n! ∧ ¬ 7 ^ 400 ∣ n! := sorry

theorem solution_b :
  {n : ℕ | 7 ^ 400 ∣ n! ∧ ¬ 7 ^ 401 ∣ n!} = {2401, 2402, 2403, 2404, 2405, 2406, 2407} := sorry

end no_solution_a_solution_b_l104_104130


namespace central_angle_of_cone_development_diagram_l104_104711

-- Given conditions: radius of the base of the cone and slant height
def radius_base := 1
def slant_height := 3

-- Target theorem: prove the central angle of the lateral surface development diagram is 120 degrees
theorem central_angle_of_cone_development_diagram : 
  ∃ n : ℝ, (2 * π) = (n * π * slant_height) / 180 ∧ n = 120 :=
by
  use 120
  sorry

end central_angle_of_cone_development_diagram_l104_104711


namespace problem_180_180_minus_12_l104_104411

namespace MathProof

theorem problem_180_180_minus_12 :
  180 * (180 - 12) - (180 * 180 - 12) = -2148 := 
by
  -- Placeholders for computation steps
  sorry

end MathProof

end problem_180_180_minus_12_l104_104411


namespace volume_ratio_of_spheres_l104_104640

theorem volume_ratio_of_spheres 
  (r1 r2 : ℝ) 
  (h_surface_area : (4 * Real.pi * r1^2) / (4 * Real.pi * r2^2) = 1 / 16) : 
  (4 / 3 * Real.pi * r1^3) / (4 / 3 * Real.pi * r2^3) = 1 / 64 :=
by 
  sorry

end volume_ratio_of_spheres_l104_104640


namespace base_of_isosceles_triangle_l104_104995

theorem base_of_isosceles_triangle (a b side equil_perim iso_perim : ℕ) 
  (h1 : equil_perim = 60)
  (h2 : 3 * side = equil_perim)
  (h3 : iso_perim = 50)
  (h4 : 2 * side + b = iso_perim)
  : b = 10 :=
by
  sorry

end base_of_isosceles_triangle_l104_104995


namespace find_line_eq_l104_104815

theorem find_line_eq (l : ℝ → ℝ → Prop) :
  (∃ A B : ℝ × ℝ, l A.fst A.snd ∧ l B.fst B.snd ∧ ((A.fst + 1)^2 + (A.snd - 2)^2 = 100 ∧ (B.fst + 1)^2 + (B.snd - 2)^2 = 100)) ∧
  (∃ M : ℝ × ℝ, M = (-2, 3) ∧ (l M.fst M.snd)) →
  (∀ x y : ℝ, l x y ↔ x - y + 5 = 0) :=
by
  sorry

end find_line_eq_l104_104815


namespace hernandez_state_tax_l104_104572

theorem hernandez_state_tax 
    (res_months : ℕ) (total_months : ℕ) 
    (taxable_income : ℝ) (tax_rate : ℝ) 
    (prorated_income : ℝ) (state_tax : ℝ) 
    (h1 : res_months = 9) 
    (h2 : total_months = 12) 
    (h3 : taxable_income = 42500) 
    (h4 : tax_rate = 0.04) 
    (h5 : prorated_income = taxable_income * (res_months / total_months)) 
    (h6 : state_tax = prorated_income * tax_rate) : 
    state_tax = 1275 := 
by 
  -- this is where the proof would go
  sorry

end hernandez_state_tax_l104_104572


namespace black_beans_count_l104_104624

theorem black_beans_count (B G O : ℕ) (h₁ : G = B + 2) (h₂ : O = G - 1) (h₃ : B + G + O = 27) : B = 8 := by
  sorry

end black_beans_count_l104_104624


namespace simplify_expression_l104_104164

theorem simplify_expression (x y : ℝ) (h1 : x = 1) (h2 : y = 2) : 
  ((x + y) * (x - y) - (x - y)^2 + 2 * y * (x - y)) / (4 * y) = -1 :=
by
  sorry

end simplify_expression_l104_104164


namespace S_gt_inverse_1988_cubed_l104_104324

theorem S_gt_inverse_1988_cubed (a b c d : ℕ) (hb: 0 < b) (hd: 0 < d) 
  (h1: a + c < 1988) (h2: 1 - (a / b) - (c / d) > 0) : 
  1 - (a / b) - (c / d) > 1 / (1988^3) := 
sorry

end S_gt_inverse_1988_cubed_l104_104324


namespace trigonometric_identity_example_l104_104847

theorem trigonometric_identity_example :
  2 * Real.sin (75 * Real.pi / 180) * Real.cos (75 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end trigonometric_identity_example_l104_104847


namespace arithmetic_expression_value_l104_104482

theorem arithmetic_expression_value :
  2 - (-3) * 2 - 4 - (-5) * 2 - 6 = 8 :=
by {
  sorry
}

end arithmetic_expression_value_l104_104482


namespace parallel_lines_determine_plane_l104_104263

def determine_plane_by_parallel_lines := 
  let condition_4 := true -- Two parallel lines
  condition_4 = true

theorem parallel_lines_determine_plane : determine_plane_by_parallel_lines = true :=
by 
  sorry

end parallel_lines_determine_plane_l104_104263


namespace find_M_l104_104025

theorem find_M (p q r s M : ℚ)
  (h1 : p + q + r + s = 100)
  (h2 : p + 10 = M)
  (h3 : q - 5 = M)
  (h4 : 10 * r = M)
  (h5 : s / 2 = M) :
  M = 1050 / 41 :=
by
  sorry

end find_M_l104_104025


namespace number_of_bikes_l104_104626

theorem number_of_bikes (total_wheels : ℕ) (car_wheels : ℕ) (tricycle_wheels : ℕ) (roller_skate_wheels : ℕ) (trash_can_wheels : ℕ) (bike_wheels : ℕ) (num_bikes : ℕ) :
  total_wheels = 25 →
  car_wheels = 2 * 4 →
  tricycle_wheels = 3 →
  roller_skate_wheels = 4 →
  trash_can_wheels = 2 →
  bike_wheels = 2 →
  (total_wheels - (car_wheels + tricycle_wheels + roller_skate_wheels + trash_can_wheels)) = bike_wheels * num_bikes →
  num_bikes = 4 := 
by
  intros total_wheels_eq total_car_wheels_eq tricycle_wheels_eq roller_skate_wheels_eq trash_can_wheels_eq bike_wheels_eq remaining_wheels_eq
  sorry

end number_of_bikes_l104_104626


namespace arithmetic_sequence_S12_l104_104083

def arithmetic_sequence_sum (a d n : ℕ) : ℕ :=
  n * (2*a + (n-1)*d) / 2

def a_n (a d n : ℕ) : ℕ :=
  a + (n-1)*d

variable (a d : ℕ)

theorem arithmetic_sequence_S12 (h : a_n a d 4 + a_n a d 9 = 10) :
  arithmetic_sequence_sum a d 12 = 60 :=
by sorry

end arithmetic_sequence_S12_l104_104083


namespace find_parts_per_hour_find_min_A_machines_l104_104999

-- Conditions
variable (x y : ℕ) -- x is parts per hour by B, y is parts per hour by A

-- Definitions based on conditions
def machineA_speed_relation (x y : ℕ) : Prop :=
  y = x + 2

def time_relation (x y : ℕ) : Prop :=
  80 / y = 60 / x

def min_A_machines (x y : ℕ) (m : ℕ) : Prop :=
  8 * m + 6 * (10 - m) ≥ 70

-- Problem statements
theorem find_parts_per_hour (x y : ℕ) (h1 : machineA_speed_relation x y) (h2 : time_relation x y) :
  x = 6 ∧ y = 8 :=
sorry

theorem find_min_A_machines (m : ℕ) (h1 : machineA_speed_relation 6 8) (h2 : time_relation 6 8) (h3 : min_A_machines 6 8 m) :
  m ≥ 5 :=
sorry

end find_parts_per_hour_find_min_A_machines_l104_104999


namespace braking_distance_non_alcohol_reaction_time_after_alcohol_braking_distance_after_alcohol_increase_in_braking_distance_max_reaction_time_l104_104806

section braking_distance

variables {t k v s : ℝ}

-- Problem 1
theorem braking_distance_non_alcohol: 
  (t = 0.5) ∧ (v = 10) ∧ (k = 0.1) → (s = t * v + k * v^2) → (s = 15) :=
by intros; sorry

-- Problem 2a
theorem reaction_time_after_alcohol:
  (v = 15) ∧ (s = 52.5) ∧ (k = 0.1) → (s = t * v + k * v^2) → (t = 2) :=
by intros; sorry

-- Problem 2b
theorem braking_distance_after_alcohol:
  (t = 2) ∧ (v = 10) ∧ (k = 0.1) → (s = t * v + k * v^2) → (s = 30) :=
by intros; sorry

-- Problem 2c
theorem increase_in_braking_distance:
  (s_after = 30) ∧ (s_before = 15) → (diff = s_after - s_before) → (diff = 15) :=
by intros; sorry

-- Problem 3
theorem max_reaction_time:
  (v = 12) ∧ (k = 0.1) ∧ (s ≤ 42) → (s = t * v + k * v^2) → (t ≤ 2.3) :=
by intros; sorry

end braking_distance

end braking_distance_non_alcohol_reaction_time_after_alcohol_braking_distance_after_alcohol_increase_in_braking_distance_max_reaction_time_l104_104806


namespace lowest_point_on_graph_l104_104893

noncomputable def f (x : ℝ) : ℝ := (x^2 + 2 * x + 2) / (x + 1)

theorem lowest_point_on_graph : ∃ (x y : ℝ), x = 0 ∧ y = 2 ∧ ∀ z > -1, f z ≥ y ∧ f x = y := by
  sorry

end lowest_point_on_graph_l104_104893


namespace hide_and_seek_l104_104518

theorem hide_and_seek
  (A B V G D : Prop)
  (h1 : A → (B ∧ ¬V))
  (h2 : B → (G ∨ D))
  (h3 : ¬V → (¬B ∧ ¬D))
  (h4 : ¬A → (B ∧ ¬G)) :
  (B ∧ V ∧ D) :=
by
  sorry

end hide_and_seek_l104_104518


namespace range_of_a_l104_104122

noncomputable def f (a x : ℝ) : ℝ := (2 - a^2) * x + a

theorem range_of_a (a : ℝ) :
  (∀ x, 0 ≤ x ∧ x ≤ 1 → f a x > 0) ↔ (0 < a ∧ a < 2) :=
by
  sorry

end range_of_a_l104_104122


namespace value_of_k_l104_104783

theorem value_of_k (k : ℝ) : 
  (∃ P Q R : ℝ × ℝ, P = (5, 12) ∧ Q = (0, k) ∧ dist (0, 0) P = dist (0, 0) Q + 5) → 
  k = 8 := 
by
  sorry

end value_of_k_l104_104783


namespace dewei_less_than_daliah_l104_104668

theorem dewei_less_than_daliah
  (daliah_amount : ℝ := 17.5)
  (zane_amount : ℝ := 62)
  (zane_multiple_dewei : zane_amount = 4 * (zane_amount / 4)) :
  (daliah_amount - (zane_amount / 4)) = 2 :=
by
  sorry

end dewei_less_than_daliah_l104_104668


namespace find_student_ticket_price_l104_104208

variable (S : ℝ)
variable (student_tickets non_student_tickets total_tickets : ℕ)
variable (non_student_ticket_price total_revenue : ℝ)

theorem find_student_ticket_price 
  (h1 : student_tickets = 90)
  (h2 : non_student_tickets = 60)
  (h3 : total_tickets = student_tickets + non_student_tickets)
  (h4 : non_student_ticket_price = 8)
  (h5 : total_revenue = 930)
  (h6 : 90 * S + 60 * non_student_ticket_price = total_revenue) : 
  S = 5 := 
sorry

end find_student_ticket_price_l104_104208


namespace record_expenditure_20_l104_104884

-- Define the concept of recording financial transactions
def record_income (amount : ℤ) : ℤ := amount

def record_expenditure (amount : ℤ) : ℤ := -amount

-- Given conditions
variable (income : ℤ) (expenditure : ℤ)

-- Condition: the income of 30 yuan is recorded as +30 yuan
axiom income_record : record_income 30 = 30

-- Prove an expenditure of 20 yuan is recorded as -20 yuan
theorem record_expenditure_20 : record_expenditure 20 = -20 := 
  by sorry

end record_expenditure_20_l104_104884


namespace simon_change_l104_104293

noncomputable def calculate_change 
  (pansies_count : ℕ) (pansies_price : ℚ) (pansies_discount : ℚ) 
  (hydrangea_count : ℕ) (hydrangea_price : ℚ) (hydrangea_discount : ℚ) 
  (petunias_count : ℕ) (petunias_price : ℚ) (petunias_discount : ℚ) 
  (lilies_count : ℕ) (lilies_price : ℚ) (lilies_discount : ℚ) 
  (orchids_count : ℕ) (orchids_price : ℚ) (orchids_discount : ℚ) 
  (sales_tax : ℚ) (payment : ℚ) : ℚ :=
  let pansies_total := (pansies_count * pansies_price) * (1 - pansies_discount)
  let hydrangea_total := (hydrangea_count * hydrangea_price) * (1 - hydrangea_discount)
  let petunias_total := (petunias_count * petunias_price) * (1 - petunias_discount)
  let lilies_total := (lilies_count * lilies_price) * (1 - lilies_discount)
  let orchids_total := (orchids_count * orchids_price) * (1 - orchids_discount)
  let total_price := pansies_total + hydrangea_total + petunias_total + lilies_total + orchids_total
  let final_price := total_price * (1 + sales_tax)
  payment - final_price

theorem simon_change : calculate_change
  5 2.50 0.10
  1 12.50 0.15
  5 1.00 0.20
  3 5.00 0.12
  2 7.50 0.08
  0.06 100 = 43.95 := by sorry

end simon_change_l104_104293


namespace parker_shorter_than_daisy_l104_104167

noncomputable def solve_height_difference : Nat :=
  let R := 60
  let D := R + 8
  let avg := 64
  ((3 * avg) - (D + R))

theorem parker_shorter_than_daisy :
  let P := solve_height_difference
  D - P = 4 := by
  sorry

end parker_shorter_than_daisy_l104_104167


namespace value_of_x_l104_104281

theorem value_of_x (x : ℝ) (h : 4 * x + 5 * x + x + 2 * x = 360) : x = 30 := 
by
  sorry

end value_of_x_l104_104281


namespace sum_youngest_oldest_l104_104239

variables {a1 a2 a3 a4 a5 : ℕ}

def mean_age (x y z u v : ℕ) : ℕ := (x + y + z + u + v) / 5
def median_age (x y z u v : ℕ) : ℕ := z

theorem sum_youngest_oldest
  (h_mean: mean_age a1 a2 a3 a4 a5 = 10) 
  (h_median: median_age a1 a2 a3 a4 a5 = 7)
  (h_sorted: a1 ≤ a2 ∧ a2 ≤ a3 ∧ a3 ≤ a4 ∧ a4 ≤ a5) :
  a1 + a5 = 23 :=
sorry

end sum_youngest_oldest_l104_104239


namespace margie_change_l104_104027

def cost_of_banana_cents : ℕ := 30
def cost_of_orange_cents : ℕ := 60
def num_bananas : ℕ := 4
def num_oranges : ℕ := 2
def amount_paid_dollars : ℝ := 10.0

noncomputable def cost_of_banana_dollars := (cost_of_banana_cents : ℝ) / 100
noncomputable def cost_of_orange_dollars := (cost_of_orange_cents : ℝ) / 100

noncomputable def total_cost := 
  (num_bananas * cost_of_banana_dollars) + (num_oranges * cost_of_orange_dollars)

noncomputable def change_received := amount_paid_dollars - total_cost

theorem margie_change : change_received = 7.60 := 
by sorry

end margie_change_l104_104027


namespace probability_of_drawing_diamond_or_ace_l104_104206

-- Define the number of diamonds
def numDiamonds : ℕ := 13

-- Define the number of other Aces
def numOtherAces : ℕ := 3

-- Define the total number of cards in the deck
def totalCards : ℕ := 52

-- Define the number of desirable outcomes (either diamonds or Aces)
def numDesirableOutcomes : ℕ := numDiamonds + numOtherAces

-- Define the probability of drawing a diamond or an Ace
def desiredProbability : ℚ := numDesirableOutcomes / totalCards

theorem probability_of_drawing_diamond_or_ace :
  desiredProbability = 4 / 13 :=
by
  sorry

end probability_of_drawing_diamond_or_ace_l104_104206


namespace am_gm_example_l104_104667

open Real

theorem am_gm_example (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a + 1)^3 / b + (b + 1)^3 / c + (c + 1)^3 / a ≥ 81 / 4 := 
by 
  sorry

end am_gm_example_l104_104667


namespace no_solution_for_k_eq_six_l104_104852

theorem no_solution_for_k_eq_six :
  ∀ x k : ℝ, k = 6 → (x ≠ 2 ∧ x ≠ 7) → (x - 1) / (x - 2) = (x - k) / (x - 7) → false :=
by 
  intros x k hk hnx_eq h_eq
  sorry

end no_solution_for_k_eq_six_l104_104852


namespace hall_length_width_difference_l104_104525

theorem hall_length_width_difference :
  ∃ (L W : ℝ), 
  (W = (1 / 2) * L) ∧
  (L * W = 288) ∧
  (L - W = 12) :=
by
  -- The mathematical proof follows from the conditions given
  sorry

end hall_length_width_difference_l104_104525


namespace residue_neg_998_mod_28_l104_104865

theorem residue_neg_998_mod_28 : ∃ r : ℤ, r = -998 % 28 ∧ 0 ≤ r ∧ r < 28 ∧ r = 10 := 
by sorry

end residue_neg_998_mod_28_l104_104865


namespace Mike_gave_marbles_l104_104341

variables (original_marbles given_marbles remaining_marbles : ℕ)

def Mike_original_marbles : ℕ := 8
def Mike_remaining_marbles : ℕ := 4
def Mike_given_marbles (original remaining : ℕ) : ℕ := original - remaining

theorem Mike_gave_marbles :
  Mike_given_marbles Mike_original_marbles Mike_remaining_marbles = 4 :=
sorry

end Mike_gave_marbles_l104_104341


namespace mushrooms_left_l104_104997

-- Define the initial amount of mushrooms.
def init_mushrooms : ℕ := 15

-- Define the amount of mushrooms eaten.
def eaten_mushrooms : ℕ := 8

-- Define the resulting amount of mushrooms.
def remaining_mushrooms (init : ℕ) (eaten : ℕ) : ℕ := init - eaten

-- The proof statement
theorem mushrooms_left : remaining_mushrooms init_mushrooms eaten_mushrooms = 7 :=
by
    sorry

end mushrooms_left_l104_104997


namespace isosceles_triangle_relation_range_l104_104311

-- Definitions of the problem conditions and goal
variables (x y : ℝ)

-- Given conditions
def isosceles_triangle (x y : ℝ) :=
  x + x + y = 10

-- Prove the relationship and range 
theorem isosceles_triangle_relation_range (h : isosceles_triangle x y) :
  y = 10 - 2 * x ∧ (5 / 2 < x ∧ x < 5) :=
  sorry

end isosceles_triangle_relation_range_l104_104311


namespace matrix_pow_2018_l104_104279

open Matrix

-- Define the specific matrix
def A : Matrix (Fin 2) (Fin 2) ℤ := ![![1, 0], ![1, 1]]

-- Formalize the statement
theorem matrix_pow_2018 : A ^ 2018 = ![![1, 0], ![2018, 1]] :=
  sorry

end matrix_pow_2018_l104_104279


namespace unique_solution_l104_104137

def is_prime (p : ℕ) : Prop := ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

theorem unique_solution (n : ℕ) :
  (0 < n ∧ is_prime (n + 1) ∧ is_prime (n + 3) ∧
   is_prime (n + 7) ∧ is_prime (n + 9) ∧
   is_prime (n + 13) ∧ is_prime (n + 15)) ↔ n = 4 :=
by
  sorry

end unique_solution_l104_104137


namespace team_won_five_games_l104_104476
-- Import the entire Mathlib library

-- Number of games played (given as a constant)
def numberOfGamesPlayed : ℕ := 10

-- Number of losses definition based on the ratio condition
def numberOfLosses : ℕ := numberOfGamesPlayed / 2

-- The number of wins is defined as the total games played minus the number of losses
def numberOfWins : ℕ := numberOfGamesPlayed - numberOfLosses

-- Proof statement: The number of wins is 5
theorem team_won_five_games :
  numberOfWins = 5 := by
  sorry

end team_won_five_games_l104_104476


namespace Mona_bikes_30_miles_each_week_l104_104262

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

end Mona_bikes_30_miles_each_week_l104_104262


namespace sum_quotient_dividend_divisor_l104_104514

theorem sum_quotient_dividend_divisor (N : ℕ) (divisor : ℕ) (quotient : ℕ) (sum : ℕ) 
    (h₁ : N = 40) (h₂ : divisor = 2) (h₃ : quotient = N / divisor)
    (h₄ : sum = quotient + N + divisor) : sum = 62 := by
  -- proof goes here
  sorry

end sum_quotient_dividend_divisor_l104_104514


namespace solve_system_of_inequalities_l104_104162

theorem solve_system_of_inequalities (x : ℝ) :
  (2 * x - 2 > 0) ∧ (3 * (x - 1) - 7 < -2 * x) → 1 < x ∧ x < 2 :=
by
  sorry

end solve_system_of_inequalities_l104_104162


namespace matrix_cubed_l104_104684

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℤ := ![![2, -2], ![2, -1]]

theorem matrix_cubed :
  (A * A * A) = ![![ -4, 2], ![-2, 1]] :=
by
  sorry

end matrix_cubed_l104_104684


namespace parabola_vertex_l104_104182

-- Define the condition: the equation of the parabola
def parabola_equation (x y : ℝ) : Prop :=
  y^2 + 4 * y + 3 * x + 1 = 0

-- Define the statement: prove that the vertex of the parabola is (1, -2)
theorem parabola_vertex :
  parabola_equation 1 (-2) :=
by
  sorry

end parabola_vertex_l104_104182


namespace sum_of_digits_of_largest_n_l104_104948

def is_prime (p : ℕ) : Prop := Nat.Prime p

def is_single_digit_prime (p : ℕ) : Prop := is_prime p ∧ p < 10

noncomputable def required_n (d e : ℕ) : ℕ := d * e * (d^2 + 10 * e)

def sum_of_digits (n : ℕ) : ℕ := 
  if n < 10 then n 
  else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_of_largest_n : 
  ∃ (d e : ℕ), 
    is_single_digit_prime d ∧ is_single_digit_prime e ∧ 
    is_prime (d^2 + 10 * e) ∧ 
    (∀ d' e' : ℕ, is_single_digit_prime d' ∧ is_single_digit_prime e' ∧ is_prime (d'^2 + 10 * e') → required_n d e ≥ required_n d' e') ∧ 
    sum_of_digits (required_n d e) = 9 :=
sorry

end sum_of_digits_of_largest_n_l104_104948


namespace halfway_between_fractions_l104_104826

theorem halfway_between_fractions : ( (1/8 : ℚ) + (1/3 : ℚ) ) / 2 = 11 / 48 :=
by
  sorry

end halfway_between_fractions_l104_104826


namespace cube_edge_length_l104_104725

theorem cube_edge_length {e : ℝ} (h : 12 * e = 108) : e = 9 :=
by sorry

end cube_edge_length_l104_104725


namespace probability_two_green_marbles_l104_104784

open Classical

section
variable (num_red num_green num_white num_blue : ℕ)
variable (total_marbles : ℕ := num_red + num_green + num_white + num_blue)

def probability_green_two_draws (num_green : ℕ) (total_marbles : ℕ) : ℚ :=
  (num_green / total_marbles : ℚ) * ((num_green - 1) / (total_marbles - 1))

theorem probability_two_green_marbles :
  probability_green_two_draws 4 (3 + 4 + 8 + 5) = 3 / 95 := by
  sorry
end

end probability_two_green_marbles_l104_104784


namespace david_initial_money_l104_104968

theorem david_initial_money (S X : ℕ) (h1 : S - 800 = 500) (h2 : X = S + 500) : X = 1800 :=
by
  sorry

end david_initial_money_l104_104968


namespace perimeter_C_l104_104700

theorem perimeter_C : 
  ∀ {x y : ℕ}, 
    (6 * x + 2 * y = 56) → (4 * x + 6 * y = 56) → 
    (2 * x + 6 * y = 40) :=
by
  intros x y h1 h2
  sorry

end perimeter_C_l104_104700


namespace find_f_neg_one_l104_104393

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
if x >= 0 then 2^x + 2 * x + m else -(2^(-x) + 2 * (-x) + m)

theorem find_f_neg_one (m : ℝ) (h_m : f 0 m = 0) : f (-1) (-1) = -3 :=
by
  sorry

end find_f_neg_one_l104_104393


namespace correct_choice_d_l104_104912

def is_quadrant_angle (alpha : ℝ) (k : ℤ) : Prop :=
  2 * k * Real.pi - Real.pi / 2 < alpha ∧ alpha < 2 * k * Real.pi

theorem correct_choice_d (alpha : ℝ) (k : ℤ) :
  is_quadrant_angle alpha k ↔ (2 * k * Real.pi - Real.pi / 2 < alpha ∧ alpha < 2 * k * Real.pi) := by
sorry

end correct_choice_d_l104_104912


namespace cuboid_diagonal_length_l104_104091

theorem cuboid_diagonal_length (x y z : ℝ) 
  (h1 : y * z = Real.sqrt 2) 
  (h2 : z * x = Real.sqrt 3)
  (h3 : x * y = Real.sqrt 6) : 
  Real.sqrt (x^2 + y^2 + z^2) = Real.sqrt 6 :=
sorry

end cuboid_diagonal_length_l104_104091


namespace square_vertex_distance_l104_104924

noncomputable def inner_square_perimeter : ℝ := 24
noncomputable def outer_square_perimeter : ℝ := 32
noncomputable def greatest_distance : ℝ := 7 * Real.sqrt 2

theorem square_vertex_distance :
  let inner_side := inner_square_perimeter / 4
  let outer_side := outer_square_perimeter / 4
  let inner_diagonal := Real.sqrt (inner_side ^ 2 + inner_side ^ 2)
  let outer_diagonal := Real.sqrt (outer_side ^ 2 + outer_side ^ 2)
  let distance := (inner_diagonal / 2) + (outer_diagonal / 2)
  distance = greatest_distance :=
by
  sorry

end square_vertex_distance_l104_104924


namespace arithmetic_sequence_a8_l104_104339

theorem arithmetic_sequence_a8 (a : ℕ → ℤ) (h_arith : ∀ n m, a (n + 1) - a n = a m + 1 - a m) 
  (h1 : a 2 = 3) (h2 : a 5 = 12) : a 8 = 21 := 
by 
  sorry

end arithmetic_sequence_a8_l104_104339


namespace smallest_total_students_l104_104172

theorem smallest_total_students (n : ℕ) (h1 : 25 * n % 100 = 0)
  (h2 : 10 * n % 4 = 0)
  (h3 : ∃ (y z : ℕ), y = 3 * z / 2 ∧ (y + z - n / 40 = n / 4)) :
  ∃ k : ℕ, n = 200 * k :=
by
  sorry

end smallest_total_students_l104_104172


namespace similar_triangles_iff_sides_proportional_l104_104716

theorem similar_triangles_iff_sides_proportional
  (a b c a1 b1 c1 : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < a1 ∧ 0 < b1 ∧ 0 < c1) :
  (Real.sqrt (a * a1) + Real.sqrt (b * b1) + Real.sqrt (c * c1) =
   Real.sqrt ((a + b + c) * (a1 + b1 + c1))) ↔
  (a / a1 = b / b1 ∧ b / b1 = c / c1) :=
by
  sorry

end similar_triangles_iff_sides_proportional_l104_104716


namespace number_of_ways_to_choose_teams_l104_104724

theorem number_of_ways_to_choose_teams : 
  ∃ (n : ℕ), n = Nat.choose 5 2 ∧ n = 10 :=
by
  have h : Nat.choose 5 2 = 10 := by sorry
  use 10
  exact ⟨h, rfl⟩

end number_of_ways_to_choose_teams_l104_104724


namespace smallest_n_l104_104040

theorem smallest_n (n : ℕ) (h1 : n > 1) (h2 : 2016 ∣ (3 * n^3 + 2013)) : n = 193 := 
sorry

end smallest_n_l104_104040


namespace rain_stop_time_on_first_day_l104_104046

-- Define the problem conditions
def raining_time_day1 (x : ℕ) : Prop :=
  let start_time := 7 * 60 -- start time in minutes
  let stop_time := start_time + x * 60 -- stop time in minutes
  stop_time = 17 * 60 -- stop at 17:00 (5:00 PM)

def total_raining_time_46_hours (x : ℕ) : Prop :=
  x + (x + 2) + 2 * (x + 2) = 46

-- Main statement
theorem rain_stop_time_on_first_day (x : ℕ) (h1 : total_raining_time_46_hours x) : raining_time_day1 x :=
  sorry

end rain_stop_time_on_first_day_l104_104046


namespace ear_muffs_total_l104_104757

theorem ear_muffs_total (a b : ℕ) (h1 : a = 1346) (h2 : b = 6444) : a + b = 7790 :=
by
  sorry

end ear_muffs_total_l104_104757


namespace house_total_volume_l104_104509

def room_volume (length : ℝ) (width : ℝ) (height : ℝ) : ℝ :=
  length * width * height

def bathroom_volume := room_volume 4 2 7
def bedroom_volume := room_volume 12 10 8
def livingroom_volume := room_volume 15 12 9

def total_volume := bathroom_volume + bedroom_volume + livingroom_volume

theorem house_total_volume : total_volume = 2636 := by
  sorry

end house_total_volume_l104_104509


namespace prob_bigger_number_correct_l104_104720

def bernardo_picks := {n | 1 ≤ n ∧ n ≤ 10}
def silvia_picks := {n | 1 ≤ n ∧ n ≤ 8}

noncomputable def prob_bigger_number : ℚ :=
  let prob_bern_picks_10 : ℚ := 3 / 10
  let prob_bern_not_10_larger_silvia : ℚ := 55 / 112
  let prob_bern_not_picks_10 : ℚ := 7 / 10
  prob_bern_picks_10 + prob_bern_not_10_larger_silvia * prob_bern_not_picks_10

theorem prob_bigger_number_correct :
  prob_bigger_number = 9 / 14 := by
  sorry

end prob_bigger_number_correct_l104_104720


namespace area_of_triangle_l104_104249

-- Definitions
variables {A B C : Type}
variables {i j k : ℕ}
variables (AB AC : ℝ)
variables (s t : ℝ)
variables (sinA : ℝ) (cosA : ℝ)

-- Conditions 
axiom sin_A : sinA = 4 / 5
axiom dot_product : s * t * cosA = 6

-- The problem theorem
theorem area_of_triangle : (1 / 2) * s * t * sinA = 4 :=
by
  sorry

end area_of_triangle_l104_104249


namespace total_weight_of_rings_l104_104255

-- Define the weights of the rings
def weight_orange : Real := 0.08
def weight_purple : Real := 0.33
def weight_white : Real := 0.42
def weight_blue : Real := 0.59
def weight_red : Real := 0.24
def weight_green : Real := 0.16

-- Define the total weight of the rings
def total_weight : Real :=
  weight_orange + weight_purple + weight_white + weight_blue + weight_red + weight_green

-- The task is to prove that the total weight equals 1.82
theorem total_weight_of_rings : total_weight = 1.82 := 
  by
    sorry

end total_weight_of_rings_l104_104255


namespace fibonacci_sum_of_squares_l104_104974

def fibonacci : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_sum_of_squares (n : ℕ) (hn : n ≥ 1) :
  (Finset.range n).sum (λ i => (fibonacci (i + 1))^2) = fibonacci n * fibonacci (n + 1) :=
sorry

end fibonacci_sum_of_squares_l104_104974


namespace bond_value_after_8_years_l104_104254

theorem bond_value_after_8_years :
  ∀ (P A r t : ℝ), P = 240 → r = 0.0833333333333332 → t = 8 →
  (A = P * (1 + r * t)) → A = 400 :=
by
  sorry

end bond_value_after_8_years_l104_104254


namespace problem_l104_104911

noncomputable def f (x : ℝ) : ℝ := (Real.sin (x / 4)) ^ 6 + (Real.cos (x / 4)) ^ 6

theorem problem : (derivative^[2008] f 0) = 3 / 8 := by sorry

end problem_l104_104911


namespace sum_of_areas_is_correct_l104_104282

/-- Define the lengths of the rectangles -/
def lengths : List ℕ := [4, 16, 36, 64, 100]

/-- Define the common base width of the rectangles -/
def base_width : ℕ := 3

/-- Define the area of a rectangle given its length and a common base width -/
def area (length : ℕ) : ℕ := base_width * length

/-- Compute the total area of the given rectangles -/
def total_area : ℕ := (lengths.map area).sum

/-- Theorem stating that the total area of the five rectangles is 660 -/
theorem sum_of_areas_is_correct : total_area = 660 := by
  sorry

end sum_of_areas_is_correct_l104_104282


namespace min_value_of_m_l104_104768

noncomputable def a_n (n : ℕ) : ℝ := 2 * 3^(n - 1)
noncomputable def b_n (n : ℕ) : ℝ := 2 * n - 9
noncomputable def c_n (n : ℕ) : ℝ := b_n n / a_n n

theorem min_value_of_m (m : ℝ) : (∀ n : ℕ, c_n n ≤ m) → m ≥ 1/162 :=
by
  sorry

end min_value_of_m_l104_104768


namespace sum_of_two_numbers_l104_104793

theorem sum_of_two_numbers (x y : ℕ) (h : x = 11) (h1 : y = 3 * x + 11) : x + y = 55 := by
  sorry

end sum_of_two_numbers_l104_104793


namespace find_s_for_g_neg1_zero_l104_104180

def g (x s : ℝ) : ℝ := 3 * x^4 + 2 * x^3 - x^2 - 4 * x + s

theorem find_s_for_g_neg1_zero (s : ℝ) : g (-1) s = 0 ↔ s = -4 :=
by
  sorry

end find_s_for_g_neg1_zero_l104_104180


namespace compute_105_squared_l104_104623

theorem compute_105_squared :
  let a := 100
  let b := 5
  (a + b)^2 = 11025 :=
by
  sorry

end compute_105_squared_l104_104623


namespace committee_formation_l104_104602

theorem committee_formation :
  let club_size := 15
  let num_roles := 2
  let num_members := 3
  let total_ways := (15 * 14) * Nat.choose (15 - num_roles) num_members
  total_ways = 60060 := by
    let club_size := 15
    let num_roles := 2
    let num_members := 3
    let total_ways := (15 * 14) * Nat.choose (15 - num_roles) num_members
    show total_ways = 60060
    sorry

end committee_formation_l104_104602


namespace complement_of_A_l104_104998

open Set

theorem complement_of_A (U : Set ℕ) (A : Set ℕ) (hU : U = {1, 2, 3, 4, 5, 6}) (hA : A = {3, 4, 5}) :
  (U \ A) = {1, 2, 6} :=
by
  sorry

end complement_of_A_l104_104998


namespace avg_difference_l104_104652

def avg (a b c : ℕ) := (a + b + c) / 3

theorem avg_difference : avg 14 32 53 - avg 21 47 22 = 3 :=
by
  sorry

end avg_difference_l104_104652


namespace range_of_a_l104_104471

theorem range_of_a (a : ℝ) (h_decreasing : ∀ x y : ℝ, x < y → (a-1)^x > (a-1)^y) : 1 < a ∧ a < 2 :=
sorry

end range_of_a_l104_104471


namespace moles_of_MgSO4_formed_l104_104157

def moles_of_Mg := 3
def moles_of_H2SO4 := 3

theorem moles_of_MgSO4_formed
  (Mg : ℕ)
  (H2SO4 : ℕ)
  (react : ℕ → ℕ → ℕ × ℕ)
  (initial_Mg : Mg = moles_of_Mg)
  (initial_H2SO4 : H2SO4 = moles_of_H2SO4)
  (balanced_eq : react Mg H2SO4 = (Mg, H2SO4)) :
  (react Mg H2SO4).1 = 3 :=
by
  sorry

end moles_of_MgSO4_formed_l104_104157


namespace spider_distance_l104_104680

/--
A spider crawls along a number line, starting at -3.
It crawls to -7, then turns around and crawls to 8.
--/
def spiderCrawl (start : ℤ) (point1 : ℤ) (point2 : ℤ): ℤ :=
  let dist1 := abs (point1 - start)
  let dist2 := abs (point2 - point1)
  dist1 + dist2

theorem spider_distance :
  spiderCrawl (-3) (-7) 8 = 19 :=
by
  sorry

end spider_distance_l104_104680


namespace park_attraction_children_count_l104_104522

theorem park_attraction_children_count
  (C : ℕ) -- Number of children
  (entrance_fee : ℕ := 5) -- Entrance fee per person
  (kids_attr_fee : ℕ := 2) -- Attraction fee for kids
  (adults_attr_fee : ℕ := 4) -- Attraction fee for adults
  (parents : ℕ := 2) -- Number of parents
  (grandmother : ℕ := 1) -- Number of grandmothers
  (total_cost : ℕ := 55) -- Total cost paid
  (entry_eq : entrance_fee * (C + parents + grandmother) + kids_attr_fee * C + adults_attr_fee * (parents + grandmother) = total_cost) :
  C = 4 :=
by
  sorry

end park_attraction_children_count_l104_104522


namespace problem_inequality_l104_104044

-- Definitions and conditions
noncomputable def f (x : ℝ) : ℝ := x * Real.log x
noncomputable def g (x : ℝ) (k : ℝ) : ℝ := f x + f (k - x)

-- The Lean proof problem
theorem problem_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  f a + (a + b) * Real.log 2 ≥ f (a + b) - f b := sorry

end problem_inequality_l104_104044


namespace train_speed_l104_104616

def train_length : ℝ := 110
def bridge_length : ℝ := 265
def crossing_time : ℝ := 30
def conversion_factor : ℝ := 3.6

theorem train_speed (train_length bridge_length crossing_time conversion_factor : ℝ) :
  (train_length + bridge_length) / crossing_time * conversion_factor = 45 :=
by
  sorry

end train_speed_l104_104616


namespace distance_to_directrix_l104_104334

theorem distance_to_directrix (p : ℝ) (h1 : ∃ (x y : ℝ), y^2 = 2 * p * x ∧ (x = 2 ∧ y = 2 * Real.sqrt 2)) :
  abs (2 - (-1)) = 3 :=
by
  sorry

end distance_to_directrix_l104_104334


namespace roots_of_polynomial_l104_104887

theorem roots_of_polynomial : 
  ∀ (x : ℝ), (x^2 + 4) * (x^2 - 4) = 0 ↔ (x = -2 ∨ x = 2) :=
by 
  sorry

end roots_of_polynomial_l104_104887


namespace cube_sum_eq_2702_l104_104253

noncomputable def x : ℝ := (2 + Real.sqrt 3) / (2 - Real.sqrt 3)
noncomputable def y : ℝ := (2 - Real.sqrt 3) / (2 + Real.sqrt 3)

theorem cube_sum_eq_2702 : x^3 + y^3 = 2702 :=
by
  sorry

end cube_sum_eq_2702_l104_104253


namespace arithmetic_sequence_terms_l104_104556

variable (n : ℕ)
variable (sumOdd sumEven : ℕ)
variable (terms : ℕ)

theorem arithmetic_sequence_terms
  (h1 : sumOdd = 120)
  (h2 : sumEven = 110)
  (h3 : terms = 2 * n + 1)
  (h4 : sumOdd + sumEven = 230) :
  terms = 23 := 
sorry

end arithmetic_sequence_terms_l104_104556


namespace jeffs_mean_l104_104913

-- Define Jeff's scores as a list or array
def jeffsScores : List ℚ := [86, 94, 87, 96, 92, 89]

-- Prove that the arithmetic mean of Jeff's scores is 544 / 6
theorem jeffs_mean : (jeffsScores.sum / jeffsScores.length) = (544 / 6) := by
  sorry

end jeffs_mean_l104_104913


namespace value_of_M_l104_104645

theorem value_of_M (x y z M : ℚ) 
  (h1 : x + y + z = 120)
  (h2 : x - 10 = M)
  (h3 : y + 10 = M)
  (h4 : 10 * z = M) : 
  M = 400 / 7 :=
sorry

end value_of_M_l104_104645


namespace polygon_is_decagon_l104_104992

-- Definitions based on conditions
def exterior_angles_sum (x : ℕ) : ℝ := 360

def interior_angles_sum (x : ℕ) : ℝ := 4 * exterior_angles_sum x

def interior_sum_formula (n : ℕ) : ℝ := (n - 2) * 180

-- Mathematically equivalent proof problem
theorem polygon_is_decagon (n : ℕ) (h1 : exterior_angles_sum n = 360)
  (h2 : interior_angles_sum n = 4 * exterior_angles_sum n)
  (h3 : interior_sum_formula n = interior_angles_sum n) : n = 10 :=
sorry

end polygon_is_decagon_l104_104992


namespace lily_petals_l104_104772

theorem lily_petals (L : ℕ) (h1 : 8 * L + 15 = 63) : L = 6 :=
by sorry

end lily_petals_l104_104772


namespace largest_divisor_of_n4_minus_n2_l104_104472

theorem largest_divisor_of_n4_minus_n2 :
  ∀ n : ℤ, 12 ∣ (n^4 - n^2) :=
by
  sorry

end largest_divisor_of_n4_minus_n2_l104_104472


namespace parallel_lines_if_and_only_if_perpendicular_lines_if_and_only_if_l104_104748

variable (m x y : ℝ)

def l1_eq : Prop := (3 - m) * x + 2 * m * y + 1 = 0
def l2_eq : Prop := 2 * m * x + 2 * y + m = 0

theorem parallel_lines_if_and_only_if : l1_eq m x y → l2_eq m x y → (m = -3/2) :=
by sorry

theorem perpendicular_lines_if_and_only_if : l1_eq m x y → l2_eq m x y → (m = 0 ∨ m = 5) :=
by sorry

end parallel_lines_if_and_only_if_perpendicular_lines_if_and_only_if_l104_104748


namespace estate_value_l104_104176

theorem estate_value (E : ℝ) (x : ℝ) (y: ℝ) (z: ℝ) 
  (h1 : 9 * x = 3 / 4 * E) 
  (h2 : z = 8 * x) 
  (h3 : y = 600) 
  (h4 : E = z + 9 * x + y):
  E = 1440 := 
sorry

end estate_value_l104_104176


namespace reflection_equation_l104_104036

theorem reflection_equation
  (incident_line : ∀ x y : ℝ, 2 * x - y + 2 = 0)
  (reflection_axis : ∀ x y : ℝ, x + y - 5 = 0) :
  ∃ x y : ℝ, x - 2 * y + 7 = 0 :=
by
  sorry

end reflection_equation_l104_104036


namespace mixed_oil_rate_l104_104351

theorem mixed_oil_rate :
  let oil1 := (10, 50)
  let oil2 := (5, 68)
  let oil3 := (8, 42)
  let oil4 := (7, 62)
  let oil5 := (12, 55)
  let oil6 := (6, 75)
  let total_cost := oil1.1 * oil1.2 + oil2.1 * oil2.2 + oil3.1 * oil3.2 + oil4.1 * oil4.2 + oil5.1 * oil5.2 + oil6.1 * oil6.2
  let total_volume := oil1.1 + oil2.1 + oil3.1 + oil4.1 + oil5.1 + oil6.1
  (total_cost / total_volume : ℝ) = 56.67 :=
by
  sorry

end mixed_oil_rate_l104_104351


namespace Q_cannot_be_log_x_l104_104964

def P : Set ℝ := {y | y ≥ 0}

theorem Q_cannot_be_log_x (Q : Set ℝ) :
  (P ∩ Q = Q) → Q ≠ {y | ∃ x, y = Real.log x} :=
by
  sorry

end Q_cannot_be_log_x_l104_104964


namespace sequence_property_l104_104515

theorem sequence_property : 
  (∀ (a : ℕ → ℝ), a 1 = 2 ∧ (∀ n ≥ 1, a (n + 1) = a n + (2 * a n) / n) → a 200 = 40200) :=
by
  sorry

end sequence_property_l104_104515


namespace sum_of_roots_of_quadratic_eq_l104_104481

theorem sum_of_roots_of_quadratic_eq (x : ℝ) :
  (x + 3) * (x - 4) = 18 → (∃ a b : ℝ, x ^ 2 + a * x + b = 0) ∧ (a = -1) ∧ (b = -30) :=
by
  sorry

end sum_of_roots_of_quadratic_eq_l104_104481


namespace train_crossing_time_l104_104605

-- Definitions for the conditions
def speed_kmph : Float := 72
def speed_mps : Float := speed_kmph * (1000 / 3600)
def length_train_m : Float := 240.0416
def length_platform_m : Float := 280
def total_distance_m : Float := length_train_m + length_platform_m

-- The problem statement
theorem train_crossing_time :
  (total_distance_m / speed_mps) = 26.00208 :=
by
  sorry

end train_crossing_time_l104_104605


namespace compute_five_fold_application_l104_104396

def f (x : ℤ) : ℤ :=
  if x ≥ 0 then -2 * x^2 else x^2 + 4 * x + 12

theorem compute_five_fold_application :
  f (f (f (f (f 2)))) = -449183247763232 :=
  by
    sorry

end compute_five_fold_application_l104_104396


namespace number_of_integer_chords_through_point_l104_104625

theorem number_of_integer_chords_through_point {r : ℝ} {c : ℝ} 
    (hr: r = 13) (hc : c = 12) : 
    ∃ n : ℕ, n = 17 :=
by
  -- Suppose O is the center and P is a point inside the circle such that OP = 12
  -- Given radius r = 13, we need to show there are 17 different integer chord lengths
  sorry  -- Proof is omitted

end number_of_integer_chords_through_point_l104_104625


namespace min_le_max_condition_l104_104851

variable (a b c : ℝ)

theorem min_le_max_condition
  (h1 : a ≠ 0)
  (h2 : ∃ t : ℝ, 2*a*t^2 + b*t + c = 0 ∧ |t| ≤ 1) :
  min c (a + c + 1) ≤ max (|b - a + 1|) (|b + a - 1|) :=
sorry

end min_le_max_condition_l104_104851


namespace isabella_stops_l104_104313

def P (n : ℕ) : ℚ := 1 / (n * (n + 1))

theorem isabella_stops (P : ℕ → ℚ) (h : ∀ n, P n = 1 / (n * (n + 1))) : 
  ∃ n : ℕ, n = 55 ∧ P n < 1 / 3000 :=
by {
  sorry
}

end isabella_stops_l104_104313


namespace overall_profit_percentage_is_30_l104_104406

noncomputable def overall_profit_percentage (n_A n_B : ℕ) (price_A price_B profit_A profit_B : ℝ) : ℝ :=
  (n_A * profit_A + n_B * profit_B) / (n_A * price_A + n_B * price_B) * 100

theorem overall_profit_percentage_is_30 :
  overall_profit_percentage 5 10 850 950 225 300 = 30 :=
by
  sorry

end overall_profit_percentage_is_30_l104_104406


namespace exponential_fixed_point_l104_104261

theorem exponential_fixed_point (a : ℝ) (hx₁ : a > 0) (hx₂ : a ≠ 1) : (0, 1) ∈ { p : ℝ × ℝ | ∃ x : ℝ, p = (x, a ^ x) } := by
  sorry 

end exponential_fixed_point_l104_104261


namespace green_notebook_cost_each_l104_104390

-- Definitions for conditions:
def num_notebooks := 4
def num_green_notebooks := 2
def num_black_notebooks := 1
def num_pink_notebooks := 1
def total_cost := 45
def black_notebook_cost := 15
def pink_notebook_cost := 10

-- Define the problem statement:
theorem green_notebook_cost_each : 
  (2 * g + black_notebook_cost + pink_notebook_cost = total_cost) → 
  g = 10 := 
by 
  intros h
  sorry

end green_notebook_cost_each_l104_104390


namespace shaded_region_correct_l104_104938

def side_length_ABCD : ℝ := 8
def side_length_BEFG : ℝ := 6

def area_square (side_length : ℝ) : ℝ := side_length ^ 2

def area_ABCD : ℝ := area_square side_length_ABCD
def area_BEFG : ℝ := area_square side_length_BEFG

def shaded_region_area : ℝ :=
  area_ABCD + area_BEFG - 32

theorem shaded_region_correct :
  shaded_region_area = 32 :=
by
  -- Proof omitted, but placeholders match problem conditions and answer
  sorry

end shaded_region_correct_l104_104938


namespace percentage_error_in_area_l104_104147

-- Definitions based on conditions
def actual_side (s : ℝ) := s
def measured_side (s : ℝ) := s * 1.01
def actual_area (s : ℝ) := s^2
def calculated_area (s : ℝ) := (measured_side s)^2

-- Theorem statement of the proof problem
theorem percentage_error_in_area (s : ℝ) : 
  (calculated_area s - actual_area s) / actual_area s * 100 = 2.01 := 
by 
  -- Proof is omitted
  sorry

end percentage_error_in_area_l104_104147


namespace mark_buttons_l104_104925

theorem mark_buttons (initial_buttons : ℕ) (shane_buttons : ℕ) (sam_buttons : ℕ) :
  initial_buttons = 14 →
  shane_buttons = 3 * initial_buttons →
  sam_buttons = (initial_buttons + shane_buttons) / 2 →
  final_buttons = (initial_buttons + shane_buttons) - sam_buttons →
  final_buttons = 28 :=
by
  sorry

end mark_buttons_l104_104925


namespace stratified_sampling_employees_over_50_l104_104428

theorem stratified_sampling_employees_over_50 :
  let total_employees := 500
  let employees_under_35 := 125
  let employees_35_to_50 := 280
  let employees_over_50 := 95
  let total_samples := 100
  (employees_over_50 / total_employees * total_samples) = 19 := by
  sorry

end stratified_sampling_employees_over_50_l104_104428


namespace smallest_sum_l104_104101

theorem smallest_sum (x y : ℕ) (hx : x ≠ y) (hxy : (1/x:ℚ) + (1/y:ℚ) = 1/15) : x + y = 64 :=
sorry

end smallest_sum_l104_104101


namespace find_x2_y2_l104_104377

theorem find_x2_y2 (x y : ℝ) (h1 : x * y = 12) (h2 : x^2 * y + x * y^2 + x + y = 120) :
  x^2 + y^2 = (10344 / 169) := by
  sorry

end find_x2_y2_l104_104377


namespace pastries_total_l104_104050

-- We start by defining the conditions
def Calvin_pastries (Frank_pastries Grace_pastries : ℕ) : ℕ := Frank_pastries + 8
def Phoebe_pastries (Frank_pastries Grace_pastries : ℕ) : ℕ := Frank_pastries + 8
def Grace_pastries : ℕ := 30
def have_same_pastries (Calvin_pastries Phoebe_pastries Grace_pastries : ℕ) : Prop :=
  Calvin_pastries + 5 = Grace_pastries ∧ Phoebe_pastries + 5 = Grace_pastries

-- Total number of pastries held by Calvin, Phoebe, Frank, and Grace
def total_pastries (Frank_pastries Calvin_pastries Phoebe_pastries Grace_pastries : ℕ) : ℕ :=
  Frank_pastries + Calvin_pastries + Phoebe_pastries + Grace_pastries

-- The statement to prove
theorem pastries_total (Frank_pastries : ℕ) : 
  have_same_pastries (Calvin_pastries Frank_pastries Grace_pastries) (Phoebe_pastries Frank_pastries Grace_pastries) Grace_pastries → 
  Frank_pastries + Calvin_pastries Frank_pastries Grace_pastries + Phoebe_pastries Frank_pastries Grace_pastries + Grace_pastries = 97 :=
by
  sorry

end pastries_total_l104_104050


namespace find_x_of_orthogonal_vectors_l104_104382

theorem find_x_of_orthogonal_vectors (x : ℝ) : 
  (⟨3, -4, 1⟩ : ℝ × ℝ × ℝ) • (⟨x, 2, -7⟩ : ℝ × ℝ × ℝ) = 0 → x = 5 := 
by
  sorry

end find_x_of_orthogonal_vectors_l104_104382


namespace water_usage_difference_l104_104981

variable (a b : ℝ)
variable (ha : a ≠ 0)
variable (hb : b ≠ 0)
variable (ha_plus_4 : a + 4 ≠ 0)

theorem water_usage_difference :
  b / a - b / (a + 4) = 4 * b / (a * (a + 4)) :=
by
  sorry

end water_usage_difference_l104_104981


namespace sequence_2018_value_l104_104119

theorem sequence_2018_value :
  ∃ a : ℕ → ℤ, a 1 = 3 ∧ a 2 = 6 ∧ (∀ n, a (n + 2) = a (n + 1) - a n) ∧ a 2018 = -3 :=
sorry

end sequence_2018_value_l104_104119


namespace regular_polygon_sides_l104_104243

theorem regular_polygon_sides (n : ℕ) (h1 : 0 < n) (h2 : 18 = 360 / n) : n = 20 :=
sorry

end regular_polygon_sides_l104_104243


namespace probability_300_feet_or_less_l104_104442

noncomputable def calculate_probability : ℚ :=
  let gates := 16
  let distance := 75
  let max_distance := 300
  let initial_choices := gates
  let final_choices := gates - 1 -- because the final choice cannot be the same as the initial one
  let total_choices := initial_choices * final_choices
  let valid_choices :=
    (2 * 4 + 2 * 5 + 2 * 6 + 2 * 7 + 8 * 8) -- the total valid assignments as calculated in the solution
  (valid_choices : ℚ) / total_choices

theorem probability_300_feet_or_less : calculate_probability = 9 / 20 := 
by 
  sorry

end probability_300_feet_or_less_l104_104442


namespace units_digit_of_p_is_6_l104_104378

theorem units_digit_of_p_is_6 (p : ℤ) (h1 : p % 10 > 0) 
                             (h2 : ((p^3) % 10 - (p^2) % 10) = 0) 
                             (h3 : (p + 1) % 10 = 7) : 
                             p % 10 = 6 :=
by sorry

end units_digit_of_p_is_6_l104_104378


namespace delta_evaluation_l104_104585

def delta (a b : ℕ) : ℕ := a^3 - b

theorem delta_evaluation :
  delta (2^(delta 3 8)) (5^(delta 4 9)) = 2^19 - 5^55 := 
sorry

end delta_evaluation_l104_104585


namespace mult_base7_correct_l104_104810

def base7_to_base10 (n : ℕ) : ℕ :=
  -- assume conversion from base-7 to base-10 is already defined
  sorry 

def base10_to_base7 (n : ℕ) : ℕ :=
  -- assume conversion from base-10 to base-7 is already defined
  sorry

theorem mult_base7_correct : (base7_to_base10 325) * (base7_to_base10 4) = base7_to_base10 1656 :=
by
  sorry

end mult_base7_correct_l104_104810


namespace minimum_time_for_tomato_egg_soup_l104_104033

noncomputable def cracking_egg_time : ℕ := 1
noncomputable def washing_chopping_tomatoes_time : ℕ := 2
noncomputable def boiling_tomatoes_time : ℕ := 3
noncomputable def adding_eggs_heating_time : ℕ := 1
noncomputable def stirring_egg_time : ℕ := 1

theorem minimum_time_for_tomato_egg_soup :
  washing_chopping_tomatoes_time + boiling_tomatoes_time + adding_eggs_heating_time = 6 :=
by
  -- proof to be filled
  sorry

end minimum_time_for_tomato_egg_soup_l104_104033


namespace find_fifth_number_l104_104904

-- Define the sets and their conditions
def first_set : List ℕ := [28, 70, 88, 104]
def second_set : List ℕ := [50, 62, 97, 124]

-- Define the means
def mean_first_set (x : ℕ) (y : ℕ) : ℚ := (28 + x + 70 + 88 + y) / 5
def mean_second_set (x : ℕ) : ℚ := (50 + 62 + 97 + 124 + x) / 5

-- Conditions given in the problem
axiom mean_first_set_condition (x y : ℕ) : mean_first_set x y = 67
axiom mean_second_set_condition (x : ℕ) : mean_second_set x = 75.6

-- Lean 4 theorem statement to prove the fifth number in the first set is 104 given above conditions
theorem find_fifth_number : ∃ x y, mean_first_set x y = 67 ∧ mean_second_set x = 75.6 ∧ y = 104 := by
  sorry

end find_fifth_number_l104_104904


namespace cyclist_speed_l104_104228

noncomputable def required_speed (d t : ℝ) : ℝ := d / t

theorem cyclist_speed :
  ∀ (d t : ℝ), 
  (d / 10 = t + 1) → 
  (d / 15 = t - 1) →
  required_speed d t = 12 := 
by
  intros d t h1 h2
  sorry

end cyclist_speed_l104_104228


namespace positive_integers_m_divisors_l104_104006

theorem positive_integers_m_divisors :
  ∃ n, n = 3 ∧ ∀ m : ℕ, (0 < m ∧ ∃ k, 2310 = k * (m^2 + 2)) ↔ m = 1 ∨ m = 2 ∨ m = 3 :=
by
  sorry

end positive_integers_m_divisors_l104_104006


namespace Vanya_Journey_Five_times_Anya_Journey_l104_104915

theorem Vanya_Journey_Five_times_Anya_Journey (a_start a_end v_start v_end : ℕ)
  (h1 : a_start = 1) (h2 : a_end = 2) (h3 : v_start = 1) (h4 : v_end = 6) :
  (v_end - v_start) = 5 * (a_end - a_start) :=
  sorry

end Vanya_Journey_Five_times_Anya_Journey_l104_104915


namespace circle_parametric_solution_l104_104076

theorem circle_parametric_solution (θ : ℝ) (hθ : 0 ≤ θ ∧ θ < 2 * Real.pi)
    (hx : 4 * Real.cos θ = -2) (hy : 4 * Real.sin θ = 2 * Real.sqrt 3) :
    θ = 2 * Real.pi / 3 :=
sorry

end circle_parametric_solution_l104_104076


namespace totalTrianglesInFigure_l104_104207

-- Definition of the problem involving a rectangle with subdivisions creating triangles
def numberOfTrianglesInRectangle : Nat :=
  let smallestTriangles := 24   -- Number of smallest triangles
  let nextSizeTriangles1 := 8   -- Triangles formed by combining smallest triangles
  let nextSizeTriangles2 := 12
  let nextSizeTriangles3 := 16
  let largestTriangles := 4
  smallestTriangles + nextSizeTriangles1 + nextSizeTriangles2 + nextSizeTriangles3 + largestTriangles

-- The Lean 4 theorem statement, stating that the total number of triangles equals 64
theorem totalTrianglesInFigure : numberOfTrianglesInRectangle = 64 := 
by
  sorry

end totalTrianglesInFigure_l104_104207


namespace brochures_multiple_of_6_l104_104337

theorem brochures_multiple_of_6 (n : ℕ) (P : ℕ) (B : ℕ) 
  (hP : P = 12) (hn : n = 6) : ∃ k : ℕ, B = 6 * k := 
sorry

end brochures_multiple_of_6_l104_104337


namespace complement_supplement_measure_l104_104278

theorem complement_supplement_measure (x : ℝ) (h : 180 - x = 3 * (90 - x)) : 
  (180 - x = 135) ∧ (90 - x = 45) :=
by {
  sorry
}

end complement_supplement_measure_l104_104278


namespace SUCCESSOR_arrangement_count_l104_104733

theorem SUCCESSOR_arrangement_count :
  (Nat.factorial 9) / (Nat.factorial 3 * Nat.factorial 2) = 30240 :=
by
  sorry

end SUCCESSOR_arrangement_count_l104_104733


namespace find_other_vertices_l104_104697

theorem find_other_vertices
  (A : ℝ × ℝ) (B C : ℝ × ℝ)
  (S : ℝ × ℝ) (M : ℝ × ℝ)
  (hA : A = (7, 3))
  (hS : S = (5, -5 / 3))
  (hM : M = (3, -1))
  (h_centroid : S = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)) 
  (h_orthocenter : ∀ u v : ℝ × ℝ, u ≠ v → u - v = (4, 4) → (u - v) • (C - B) = 0) :
  B = (1, -1) ∧ C = (7, -7) :=
sorry

end find_other_vertices_l104_104697


namespace fraction_product_l104_104538

theorem fraction_product :
  (8 / 4) * (10 / 5) * (21 / 14) * (16 / 8) * (45 / 15) * (30 / 10) * (49 / 35) * (32 / 16) = 302.4 := by
  sorry

end fraction_product_l104_104538


namespace red_ball_second_given_red_ball_first_l104_104015

noncomputable def probability_of_red_second_given_first : ℚ :=
  let totalBalls := 6
  let redBallsOnFirst := 4
  let whiteBalls := 2
  let redBallsOnSecond := 3
  let remainingBalls := 5

  let P_A := redBallsOnFirst / totalBalls
  let P_AB := (redBallsOnFirst / totalBalls) * (redBallsOnSecond / remainingBalls)
  P_AB / P_A

theorem red_ball_second_given_red_ball_first :
  probability_of_red_second_given_first = 3 / 5 :=
sorry

end red_ball_second_given_red_ball_first_l104_104015


namespace find_A_coordinates_l104_104196

-- Given conditions
variable (B : (ℝ × ℝ)) (hB1 : B = (1, 2))

-- Definitions to translate problem conditions into Lean
def symmetric_y (P B : ℝ × ℝ) : Prop :=
  P.1 = -B.1 ∧ P.2 = B.2

def symmetric_x (A P : ℝ × ℝ) : Prop :=
  A.1 = P.1 ∧ A.2 = -P.2

-- Theorem statement
theorem find_A_coordinates (A P B : ℝ × ℝ) (hB1 : B = (1, 2))
    (h_symm_y: symmetric_y P B) (h_symm_x: symmetric_x A P) : 
    A = (-1, -2) :=
by
  sorry

end find_A_coordinates_l104_104196


namespace unique_solution_quadratic_eq_l104_104603

theorem unique_solution_quadratic_eq (p : ℝ) (h_nonzero : p ≠ 0) : (∀ x : ℝ, p * x^2 - 20 * x + 4 = 0) → p = 25 :=
by
  sorry

end unique_solution_quadratic_eq_l104_104603


namespace alex_mother_age_proof_l104_104363

-- Define the initial conditions
def alex_age_2004 : ℕ := 7
def mother_age_2004 : ℕ := 35
def initial_year : ℕ := 2004

-- Define the time variable and the relationship conditions
def years_after_2004 (x : ℕ) : Prop :=
  let alex_age := alex_age_2004 + x
  let mother_age := mother_age_2004 + x
  mother_age = 2 * alex_age

-- State the theorem to be proved
theorem alex_mother_age_proof : ∃ x : ℕ, years_after_2004 x ∧ initial_year + x = 2025 :=
by
  sorry

end alex_mother_age_proof_l104_104363


namespace function_decreasing_interval_l104_104675

noncomputable def function_y (x : ℝ) : ℝ := (1/2) * x^2 - Real.log x

noncomputable def derivative_y' (x : ℝ) : ℝ := (x + 1) * (x - 1) / x

theorem function_decreasing_interval : ∀ x: ℝ, 0 < x ∧ x < 1 → (derivative_y' x < 0) := by
  sorry

end function_decreasing_interval_l104_104675


namespace min_bottles_needed_l104_104946

theorem min_bottles_needed (num_people : ℕ) (exchange_rate : ℕ) (bottles_needed_per_person : ℕ) (total_bottles_purchased : ℕ):
  num_people = 27 → exchange_rate = 3 → bottles_needed_per_person = 1 → total_bottles_purchased = 18 → 
  ∀ n, n = num_people → (n / bottles_needed_per_person) = 27 ∧ (num_people * 2 / 3) = 18 :=
by
  intros
  sorry

end min_bottles_needed_l104_104946


namespace number_of_boxes_l104_104019

theorem number_of_boxes (total_eggs : ℕ) (eggs_per_box : ℕ) (boxes : ℕ) : 
  total_eggs = 21 → eggs_per_box = 7 → boxes = total_eggs / eggs_per_box → boxes = 3 :=
by
  intros h_total_eggs h_eggs_per_box h_boxes
  rw [h_total_eggs, h_eggs_per_box] at h_boxes
  exact h_boxes

end number_of_boxes_l104_104019


namespace Hulk_jump_l104_104404

theorem Hulk_jump :
  ∃ n : ℕ, 2^n > 500 ∧ ∀ m : ℕ, m < n → 2^m ≤ 500 :=
by
  sorry

end Hulk_jump_l104_104404


namespace prime_sum_divisible_l104_104424

theorem prime_sum_divisible (p : Fin 2021 → ℕ) (prime : ∀ i, Nat.Prime (p i))
  (h : 6060 ∣ Finset.univ.sum (fun i => (p i)^4)) : 4 ≤ Finset.card (Finset.univ.filter (fun i => p i < 2021)) :=
sorry

end prime_sum_divisible_l104_104424


namespace sum_even_numbered_terms_l104_104781

variable (n : ℕ)

def a_n (n : ℕ) : ℕ := 2 * 3^(n-1)

def new_sequence (n : ℕ) : ℕ := a_n (2 * n)

def Sn (n : ℕ) : ℕ := (6 * (1 - 9^n)) / (1 - 9)

theorem sum_even_numbered_terms (n : ℕ) : Sn n = 3 * (9^n - 1) / 4 :=
by sorry

end sum_even_numbered_terms_l104_104781


namespace final_problem_l104_104970

def problem1 : Prop :=
  ∃ (x y : ℝ), 10 * x + 20 * y = 3000 ∧ 8 * x + 24 * y = 2800 ∧ x = 200 ∧ y = 50

def problem2 : Prop :=
  ∀ (m : ℕ), 10 ≤ m ∧ m ≤ 12 ∧ 
  200 * m + 50 * (40 - m) ≤ 3800 ∧ 
  (40 - m) ≤ 3 * m →
  (m = 10 ∧ (40 - m) = 30) ∨ 
  (m = 11 ∧ (40 - m) = 29) ∨ 
  (m = 12 ∧ (40 - m) = 28)

theorem final_problem : problem1 ∧ problem2 :=
by
  sorry

end final_problem_l104_104970


namespace slower_train_speed_l104_104269

theorem slower_train_speed
  (v : ℝ)  -- The speed of the slower train
  (faster_train_speed : ℝ := 46)  -- The speed of the faster train
  (train_length : ℝ := 37.5)  -- The length of each train in meters
  (time_to_pass : ℝ := 27)  -- Time taken to pass in seconds
  (kms_to_ms : ℝ := 1000 / 3600)  -- Conversion factor from km/hr to m/s
  (relative_distance : ℝ := 2 * train_length)  -- Distance covered when passing

  (h : relative_distance = (faster_train_speed - v) * kms_to_ms * time_to_pass) :
  v = 36 :=
by
  -- The proof should be placed here
  sorry

end slower_train_speed_l104_104269


namespace ratio_a_div_8_to_b_div_7_l104_104235

theorem ratio_a_div_8_to_b_div_7 (a b : ℝ) (h1 : 7 * a = 8 * b) (h2 : a ≠ 0 ∧ b ≠ 0) :
  (a / 8) / (b / 7) = 1 :=
sorry

end ratio_a_div_8_to_b_div_7_l104_104235


namespace sum_of_a_b_l104_104939

theorem sum_of_a_b (a b : ℝ) (h1 : a > b) (h2 : |a| = 9) (h3 : b^2 = 4) : a + b = 11 ∨ a + b = 7 := 
sorry

end sum_of_a_b_l104_104939


namespace seq_50th_term_eq_327_l104_104552

theorem seq_50th_term_eq_327 : 
  let n := 50
  let binary_representation : List Nat := [1, 1, 0, 0, 1, 0] -- 50 in binary
  let powers_of_3 := [5, 4, 1] -- Positions of 1s in the binary representation 
  let term := List.sum (powers_of_3.map (λ k => 3^k))
  term = 327 := by
  sorry

end seq_50th_term_eq_327_l104_104552


namespace root_diff_condition_l104_104493

noncomputable def g (x : ℝ) : ℝ := 4^x + 2*x - 2
noncomputable def f (x : ℝ) : ℝ := 4*x - 1

theorem root_diff_condition :
  ∃ x₀, g x₀ = 0 ∧ |x₀ - 1/4| ≤ 1/4 ∧ ∃ y₀, f y₀ = 0 ∧ |y₀ - x₀| ≤ 0.25 :=
sorry

end root_diff_condition_l104_104493


namespace gangster_avoid_police_l104_104289

variable (a v : ℝ)
variable (house_side_length streets_distance neighbouring_distance police_interval : ℝ)
variable (police_speed gangster_speed_to_avoid_police : ℝ)

-- Given conditions
axiom house_properties : house_side_length = a ∧ neighbouring_distance = 2 * a
axiom streets_properties : streets_distance = 3 * a
axiom police_properties : police_interval = 9 * a ∧ police_speed = v

-- Correct answer in terms of Lean
theorem gangster_avoid_police :
  gangster_speed_to_avoid_police = 2 * v ∨ gangster_speed_to_avoid_police = v / 2 :=
by
  sorry

end gangster_avoid_police_l104_104289


namespace third_red_yellow_flash_is_60_l104_104244

-- Define the flashing intervals for red, yellow, and green lights
def red_interval : Nat := 3
def yellow_interval : Nat := 4
def green_interval : Nat := 8

-- Define the function for finding the time of the third occurrence of only red and yellow lights flashing together
def third_red_yellow_flash : Nat :=
  let lcm_red_yellow := Nat.lcm red_interval yellow_interval
  let times := (List.range (100)).filter (fun t => t % lcm_red_yellow = 0 ∧ t % green_interval ≠ 0)
  times[2] -- Getting the third occurrence

-- Prove that the third occurrence time is 60 seconds
theorem third_red_yellow_flash_is_60 :
  third_red_yellow_flash = 60 :=
  by
    -- Proof goes here
    sorry

end third_red_yellow_flash_is_60_l104_104244


namespace cone_height_from_sphere_l104_104892

theorem cone_height_from_sphere (d_sphere d_base : ℝ) (h : ℝ) (V_sphere : ℝ) (V_cone : ℝ) 
  (h₁ : d_sphere = 6) 
  (h₂ : d_base = 12)
  (h₃ : V_sphere = 36 * Real.pi)
  (h₄ : V_cone = (1/3) * Real.pi * (d_base / 2)^2 * h) 
  (h₅ : V_sphere = V_cone) :
  h = 3 := by
  sorry

end cone_height_from_sphere_l104_104892


namespace Lorin_black_marbles_l104_104048

variable (B : ℕ)

def Jimmy_yellow_marbles := 22
def Alex_yellow_marbles := Jimmy_yellow_marbles / 2
def Alex_black_marbles := 2 * B
def Alex_total_marbles := Alex_yellow_marbles + Alex_black_marbles

theorem Lorin_black_marbles : Alex_total_marbles = 19 → B = 4 :=
by
  intros h
  unfold Alex_total_marbles at h
  unfold Alex_yellow_marbles at h
  unfold Alex_black_marbles at h
  norm_num at h
  exact sorry

end Lorin_black_marbles_l104_104048


namespace simplify_expression_of_triangle_side_lengths_l104_104740

theorem simplify_expression_of_triangle_side_lengths
  (a b c : ℝ) 
  (h1 : a + b > c)
  (h2 : a + c > b)
  (h3 : b + c > a) :
  |a - b - c| - |c - a + b| = 0 :=
by
  sorry

end simplify_expression_of_triangle_side_lengths_l104_104740


namespace perpendicular_to_plane_l104_104395

theorem perpendicular_to_plane (Line : Type) (Plane : Type) (triangle : Plane) (circle : Plane)
  (perpendicular1 : Line → Plane → Prop)
  (perpendicular2 : Line → Plane → Prop) :
  (∀ l, ∃ t, perpendicular1 l t ∧ t = triangle) ∧ (∀ l, ∃ c, perpendicular2 l c ∧ c = circle) →
  (∀ l, ∃ p, (perpendicular1 l p ∨ perpendicular2 l p) ∧ (p = triangle ∨ p = circle)) :=
by
  sorry

end perpendicular_to_plane_l104_104395


namespace range_of_a_l104_104785

theorem range_of_a (x y a : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 / x + 1 / y = 1) : 
  (x + y + a > 0) ↔ (a > -3 - 2 * Real.sqrt 2) :=
sorry

end range_of_a_l104_104785


namespace reciprocal_of_5_over_7_l104_104931

theorem reciprocal_of_5_over_7 : (5 / 7 : ℚ) * (7 / 5) = 1 := by
  sorry

end reciprocal_of_5_over_7_l104_104931


namespace triangle_area_l104_104063

theorem triangle_area (a b c : ℝ) (h1 : a = 5) (h2 : a + b = 13) (h3 : c = Real.sqrt (a^2 + b^2)) : 
  (1 / 2) * a * b = 20 :=
by
  sorry

end triangle_area_l104_104063


namespace reliability_is_correct_l104_104479

-- Define the probabilities of each switch functioning properly.
def P_A : ℝ := 0.9
def P_B : ℝ := 0.8
def P_C : ℝ := 0.7

-- Define the system reliability.
def reliability : ℝ := P_A * P_B * P_C

-- The theorem stating the reliability of the system.
theorem reliability_is_correct : reliability = 0.504 := by
  sorry

end reliability_is_correct_l104_104479


namespace find_a_plus_c_l104_104608

theorem find_a_plus_c {a b c d : ℝ} 
  (h1 : ∀ x, -|x - a| + b = |x - c| + d → x = 4 ∧ -|4 - a| + b = 7 ∨ x = 10 ∧ -|10 - a| + b = 3)
  (h2 : b + d = 12): a + c = 14 := by
  sorry

end find_a_plus_c_l104_104608


namespace arithmetic_seq_sum_l104_104555

theorem arithmetic_seq_sum {a : ℕ → ℝ} (h_arith : ∃ d, ∀ n, a (n + 1) = a n + d)
  (ha4 : a 4 = 5) : a 3 + a 5 = 10 :=
sorry

end arithmetic_seq_sum_l104_104555


namespace john_total_time_spent_l104_104148

-- Define conditions
def num_pictures : ℕ := 10
def draw_time_per_picture : ℝ := 2
def color_time_reduction : ℝ := 0.3

-- Define the actual color time per picture
def color_time_per_picture : ℝ := draw_time_per_picture * (1 - color_time_reduction)

-- Define the total time per picture
def total_time_per_picture : ℝ := draw_time_per_picture + color_time_per_picture

-- Define the total time for all pictures
def total_time_for_all_pictures : ℝ := total_time_per_picture * num_pictures

-- The theorem we need to prove
theorem john_total_time_spent : total_time_for_all_pictures = 34 :=
by
sorry

end john_total_time_spent_l104_104148


namespace half_abs_diff_of_squares_l104_104136

theorem half_abs_diff_of_squares (x y : ℤ) (h1 : x = 21) (h2 : y = 19) :
  (|x^2 - y^2| / 2) = 40 := 
by
  subst h1
  subst h2
  sorry

end half_abs_diff_of_squares_l104_104136


namespace employees_without_any_benefit_l104_104862

def employees_total : ℕ := 480
def employees_salary_increase : ℕ := 48
def employees_travel_increase : ℕ := 96
def employees_both_increases : ℕ := 24
def employees_vacation_days : ℕ := 72

theorem employees_without_any_benefit : (employees_total - ((employees_salary_increase + employees_travel_increase + employees_vacation_days) - employees_both_increases)) = 288 :=
by
  sorry

end employees_without_any_benefit_l104_104862


namespace intersection_A_B_l104_104896

def A : Set ℝ := { x | Real.log x > 0 }

def B : Set ℝ := { x | Real.exp x < 3 }

theorem intersection_A_B :
  A ∩ B = { x | 1 < x ∧ x < Real.log 3 / Real.log 2 } :=
sorry

end intersection_A_B_l104_104896


namespace probability_of_three_faces_painted_l104_104273

def total_cubes : Nat := 27
def corner_cubes_painted (total : Nat) : Nat := 8
def probability_of_corner_cube (corner : Nat) (total : Nat) : Rat := corner / total

theorem probability_of_three_faces_painted :
    probability_of_corner_cube (corner_cubes_painted total_cubes) total_cubes = 8 / 27 := 
by 
  sorry

end probability_of_three_faces_painted_l104_104273


namespace max_gcd_value_l104_104797

theorem max_gcd_value (n : ℕ) (hn : 0 < n) : ∃ k, k = gcd (13 * n + 4) (8 * n + 3) ∧ k <= 7 := sorry

end max_gcd_value_l104_104797


namespace unit_digit_7_power_2023_l104_104685

theorem unit_digit_7_power_2023 : (7 ^ 2023) % 10 = 3 := by
  sorry

end unit_digit_7_power_2023_l104_104685


namespace period_and_symmetry_of_function_l104_104529

-- Given conditions
variables (f : ℝ → ℝ)
variable (hf_odd : ∀ x, f (-x) = -f x)
variable (hf_cond : ∀ x, f (-2 * x + 4) = -f (2 * x))
variable (hf_def : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 2 ^ x - 1)

-- Prove that 4 is a period and x=1 is a line of symmetry for the graph of f(x)
theorem period_and_symmetry_of_function :
  (∀ x, f (x + 4) = f x) ∧ (∀ x, f (x) + f (4 - x) = 0) :=
by sorry

end period_and_symmetry_of_function_l104_104529


namespace arithmetic_sequence_properties_l104_104231

noncomputable def arithmeticSeq (a₁ d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

theorem arithmetic_sequence_properties (a₁ d : ℕ) (n : ℕ) (h1 : d = 2)
  (h2 : (a₁ + d)^2 = a₁ * (a₁ + 3 * d)) :
  (a₁ = 2) ∧ (∃ S, S = (n * (2 * a₁ + (n - 1) * d)) / 2 ∧ S = n^2 + n) :=
by 
  sorry

end arithmetic_sequence_properties_l104_104231


namespace volume_of_rice_pile_l104_104010

theorem volume_of_rice_pile
  (arc_length_bottom : ℝ)
  (height : ℝ)
  (one_fourth_cone : ℝ)
  (approx_pi : ℝ)
  (h_arc : arc_length_bottom = 8)
  (h_height : height = 5)
  (h_one_fourth_cone : one_fourth_cone = 1/4)
  (h_approx_pi : approx_pi = 3) :
  ∃ V : ℝ, V = one_fourth_cone * (1 / 3) * π * (16^2 / π^2) * height :=
by
  sorry

end volume_of_rice_pile_l104_104010


namespace tangent_line_of_ellipse_l104_104062

variable {a b x y x₀ y₀ : ℝ}

theorem tangent_line_of_ellipse
    (h1 : 0 < a)
    (h2 : a > b)
    (h3 : b > 0)
    (h4 : (x₀, y₀) ∈ { p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / b^2 = 1 }) :
    (x₀ * x) / (a^2) + (y₀ * y) / (b^2) = 1 :=
sorry

end tangent_line_of_ellipse_l104_104062


namespace isosceles_triangle_perimeter_l104_104245

theorem isosceles_triangle_perimeter (a b : ℝ) (h₁ : a = 6 ∨ a = 7) (h₂ : b = 6 ∨ b = 7) (h₃ : a ≠ b) :
  (2 * a + b = 19) ∨ (2 * b + a = 20) :=
by
  -- Proof omitted
  sorry

end isosceles_triangle_perimeter_l104_104245


namespace square_area_from_diagonal_l104_104113

theorem square_area_from_diagonal (d : ℝ) (h : d = 12) : (d^2 / 2) = 72 :=
by sorry

end square_area_from_diagonal_l104_104113


namespace complement_of_A_inter_B_eq_l104_104883

noncomputable def A : Set ℝ := {x | abs (x - 1) ≤ 1}
noncomputable def B : Set ℝ := {y | ∃ x, y = -x^2 ∧ -Real.sqrt 2 ≤ x ∧ x < 1}
noncomputable def A_inter_B : Set ℝ := {x | x ∈ A ∧ x ∈ B}
noncomputable def complement_A_inter_B : Set ℝ := {x | x ∉ A_inter_B}

theorem complement_of_A_inter_B_eq :
  complement_A_inter_B = {x : ℝ | x ≠ 0} :=
  sorry

end complement_of_A_inter_B_eq_l104_104883


namespace geometric_extraction_from_arithmetic_l104_104786

theorem geometric_extraction_from_arithmetic (a b : ℤ) :
  ∃ k : ℕ → ℤ, (∀ n : ℕ, k n = a * (b + 1) ^ n) ∧ (∀ n : ℕ, ∃ m : ℕ, k n = a + b * m) :=
by sorry

end geometric_extraction_from_arithmetic_l104_104786


namespace candy_bar_calories_l104_104003

theorem candy_bar_calories (calories : ℕ) (bars : ℕ) (dozen : ℕ) (total_calories : ℕ) 
  (H1 : total_calories = 2016) (H2 : bars = 42) (H3 : dozen = 12) 
  (H4 : total_calories = bars * calories) : 
  calories / dozen = 4 := 
by 
  sorry

end candy_bar_calories_l104_104003


namespace value_of_a_l104_104722

-- Definition of the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 + 2

-- Definition of the derivative f'(-1)
def f_prime_at_neg1 (a : ℝ) : ℝ := 3 * a - 6

-- The theorem to prove the value of a
theorem value_of_a (a : ℝ) (h : f_prime_at_neg1 a = 3) : a = 3 :=
by
  sorry

end value_of_a_l104_104722


namespace min_value_l104_104184

theorem min_value (x : ℝ) (h : 0 < x) : x + 2 / (2 * x + 1) - 3 / 2 ≥ 0 :=
sorry

end min_value_l104_104184


namespace fraction_sum_proof_l104_104853

theorem fraction_sum_proof :
    (19 / ((2^3 - 1) * (3^3 - 1)) + 
     37 / ((3^3 - 1) * (4^3 - 1)) + 
     61 / ((4^3 - 1) * (5^3 - 1)) + 
     91 / ((5^3 - 1) * (6^3 - 1))) = (208 / 1505) :=
by
  -- Proof goes here
  sorry

end fraction_sum_proof_l104_104853


namespace fifteenth_number_in_base_8_l104_104465

theorem fifteenth_number_in_base_8 : (15 : ℕ) = 1 * 8 + 7 := 
sorry

end fifteenth_number_in_base_8_l104_104465


namespace number_of_possible_values_of_a_l104_104310

theorem number_of_possible_values_of_a :
  ∃ a_count : ℕ, (∃ (a b c d : ℕ), a > b ∧ b > c ∧ c > d ∧ a + b + c + d = 2020 ∧ a^2 - b^2 + c^2 - d^2 = 2020 ∧ a_count = 501) :=
sorry

end number_of_possible_values_of_a_l104_104310


namespace max_equilateral_triangles_l104_104871

theorem max_equilateral_triangles (length : ℕ) (n : ℕ) (segments : ℕ) : 
  (length = 2) → (segments = 6) → (∀ t, 1 ≤ t ∧ t ≤ 4 → t = 4) :=
by 
  intros length_eq segments_eq h
  sorry

end max_equilateral_triangles_l104_104871


namespace inverse_of_A_cubed_l104_104843

noncomputable def A_inv : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![3, 7],
    ![-2, -5]]

theorem inverse_of_A_cubed :
  (A_inv ^ 3)⁻¹ = ![![13, -15],
                     ![-14, -29]] :=
by
  sorry

end inverse_of_A_cubed_l104_104843


namespace quadratic_no_discriminant_23_l104_104752

theorem quadratic_no_discriminant_23 (a b c : ℤ) (h_eq : b^2 - 4 * a * c = 23) : False := sorry

end quadratic_no_discriminant_23_l104_104752


namespace number_of_action_figures_removed_l104_104766

-- Definitions for conditions
def initial : ℕ := 15
def added : ℕ := 2
def current : ℕ := 10

-- The proof statement
theorem number_of_action_figures_removed (initial added current : ℕ) : 
  (initial + added - current) = 7 := by
  sorry

end number_of_action_figures_removed_l104_104766


namespace principal_amount_is_26_l104_104670

-- Define the conditions
def rate : Real := 0.07
def time : Real := 6
def simple_interest : Real := 10.92

-- Define the simple interest formula
def simple_interest_formula (P R T : Real) : Real := P * R * T

-- State the theorem to prove
theorem principal_amount_is_26 : 
  ∃ (P : Real), simple_interest_formula P rate time = simple_interest ∧ P = 26 :=
by
  sorry

end principal_amount_is_26_l104_104670


namespace value_of_expression_l104_104861

theorem value_of_expression (x y : ℝ) (h1 : x + y = 3) (h2 : x^2 + y^2 - x * y = 4) : 
  x^4 + y^4 + x^3 * y + x * y^3 = 36 :=
by
  sorry

end value_of_expression_l104_104861


namespace calculate_molecular_weight_l104_104467

def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00

def num_atoms_C := 3
def num_atoms_H := 6
def num_atoms_O := 1

def molecular_weight (nC nH nO : ℕ) (wC wH wO : ℝ) : ℝ :=
  (nC * wC) + (nH * wH) + (nO * wO)

theorem calculate_molecular_weight :
  molecular_weight num_atoms_C num_atoms_H num_atoms_O atomic_weight_C atomic_weight_H atomic_weight_O = 58.078 :=
by
  sorry

end calculate_molecular_weight_l104_104467


namespace arithmetic_sequence_k_value_l104_104323

theorem arithmetic_sequence_k_value (a1 d : ℤ) (S : ℕ → ℤ)
  (h1 : a1 = 1)
  (h2 : d = 2)
  (h3 : ∀ k : ℕ, S (k+2) - S k = 24) : k = 5 := 
sorry

end arithmetic_sequence_k_value_l104_104323


namespace area_under_f_l104_104808

noncomputable def f (x : ℝ) : ℝ := Real.log x + x^2 - 3 * x

noncomputable def f' (x : ℝ) : ℝ := 1 / x + 2 * x - 3

theorem area_under_f' : 
  - ∫ x in (1/2 : ℝ)..1, f' x = (3 / 4) - Real.log 2 := 
by
  sorry

end area_under_f_l104_104808


namespace alissa_presents_l104_104567

theorem alissa_presents :
  let Ethan_presents := 31
  let Alissa_presents := Ethan_presents + 22
  Alissa_presents = 53 :=
by
  sorry

end alissa_presents_l104_104567


namespace geometric_sequence_first_term_l104_104300

theorem geometric_sequence_first_term (a1 q : ℝ) 
  (h1 : (a1 * (1 - q^4)) / (1 - q) = 240)
  (h2 : a1 * q + a1 * q^3 = 180) : 
  a1 = 6 :=
by
  sorry

end geometric_sequence_first_term_l104_104300


namespace max_value_a_l104_104474

theorem max_value_a (a b c d : ℝ) 
  (h1 : a ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2))
  (h2 : b ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2))
  (h3 : c ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2))
  (h4 : d ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2))
  (h5 : Real.sin a + Real.sin b + Real.sin c + Real.sin d = 1)
  (h6 : Real.cos (2 * a) + Real.cos (2 * b) + Real.cos (2 * c) + Real.cos (2 * d) ≥ 10 / 3) : 
  a ≤ Real.arcsin (1 / 2) := 
sorry

end max_value_a_l104_104474


namespace cubes_with_one_colored_face_l104_104636

theorem cubes_with_one_colored_face (n : ℕ) (c1 : ℕ) (c2 : ℕ) :
  (n = 64) ∧ (c1 = 4) ∧ (c2 = 4) → ((4 * n) * 2) / n = 32 :=
by 
  sorry

end cubes_with_one_colored_face_l104_104636


namespace range_k_fx_greater_than_ln_l104_104818

noncomputable def f (x : ℝ) : ℝ := Real.exp x

theorem range_k (k : ℝ) : 0 ≤ k ∧ k ≤ Real.exp 1 ↔ ∀ x : ℝ, f x ≥ k * x := 
by 
  sorry

theorem fx_greater_than_ln (t : ℝ) (x : ℝ) : t ≤ 2 ∧ 0 < x → f x > t + Real.log x :=
by
  sorry

end range_k_fx_greater_than_ln_l104_104818


namespace cole_runs_7_miles_l104_104637

theorem cole_runs_7_miles
  (xavier_miles : ℕ)
  (katie_miles : ℕ)
  (cole_miles : ℕ)
  (h1 : xavier_miles = 3 * katie_miles)
  (h2 : katie_miles = 4 * cole_miles)
  (h3 : xavier_miles = 84)
  (h4 : katie_miles = 28) :
  cole_miles = 7 := 
sorry

end cole_runs_7_miles_l104_104637


namespace village_population_percentage_l104_104141

theorem village_population_percentage 
  (part : ℝ)
  (whole : ℝ)
  (h_part : part = 8100)
  (h_whole : whole = 9000) : 
  (part / whole) * 100 = 90 :=
by
  sorry

end village_population_percentage_l104_104141


namespace percentage_increase_in_expenditure_l104_104908

/-- Given conditions:
- The price of sugar increased by 32%
- The family's original monthly sugar consumption was 30 kg
- The family's new monthly sugar consumption is 25 kg
- The family's expenditure on sugar increased by 10%

Prove that the percentage increase in the family's expenditure on sugar is 10%. -/
theorem percentage_increase_in_expenditure (P : ℝ) :
  let initial_consumption := 30
  let new_consumption := 25
  let price_increase := 0.32
  let original_price := P
  let new_price := (1 + price_increase) * original_price
  let original_expenditure := initial_consumption * original_price
  let new_expenditure := new_consumption * new_price
  let expenditure_increase := new_expenditure - original_expenditure
  let percentage_increase := (expenditure_increase / original_expenditure) * 100
  percentage_increase = 10 := sorry

end percentage_increase_in_expenditure_l104_104908


namespace batsman_average_l104_104854

theorem batsman_average (A : ℕ) (H : (16 * A + 82) / 17 = A + 3) : (A + 3 = 34) :=
sorry

end batsman_average_l104_104854


namespace find_six_digit_number_l104_104303

theorem find_six_digit_number (a b c d e f : ℕ) (N : ℕ) :
  a = 1 ∧ f = 7 ∧
  N = 100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f ∧
  (f - 1) * 10^5 + 10^4 * a + 10^3 * b + 10^2 * c + 10 * d + e = 5 * N →
  N = 142857 :=
by
  sorry

end find_six_digit_number_l104_104303


namespace rectangle_R2_area_l104_104183

theorem rectangle_R2_area
  (side1_R1 : ℝ) (area_R1 : ℝ) (diag_R2 : ℝ)
  (h_side1_R1 : side1_R1 = 4)
  (h_area_R1 : area_R1 = 32)
  (h_diag_R2 : diag_R2 = 20) :
  ∃ (area_R2 : ℝ), area_R2 = 160 :=
by
  sorry

end rectangle_R2_area_l104_104183


namespace correct_system_of_equations_l104_104060

theorem correct_system_of_equations (x y : ℝ) :
  (y = x + 4.5 ∧ 0.5 * y = x - 1) ↔
  (y = x + 4.5 ∧ 0.5 * y = x - 1) :=
by sorry

end correct_system_of_equations_l104_104060


namespace sum_of_variables_l104_104967

theorem sum_of_variables (x y z : ℝ) (h : x^2 + y^2 + z^2 - 2 * x + 4 * y - 6 * z + 14 = 0) : x + y + z = 2 :=
sorry

end sum_of_variables_l104_104967


namespace sample_size_calculation_l104_104367

-- Definitions based on the conditions
def num_classes : ℕ := 40
def num_representatives_per_class : ℕ := 3

-- Theorem statement we aim to prove
theorem sample_size_calculation : num_classes * num_representatives_per_class = 120 :=
by
  sorry

end sample_size_calculation_l104_104367


namespace area_of_four_triangles_l104_104117

theorem area_of_four_triangles (a b : ℕ) (h1 : 2 * b = 28) (h2 : a + 2 * b = 30) :
    4 * (1 / 2 * a * b) = 56 := by
  sorry

end area_of_four_triangles_l104_104117


namespace largest_divisor_of_square_l104_104073

theorem largest_divisor_of_square (n : ℕ) (h_pos : 0 < n) (h_div : 72 ∣ n ^ 2) : 12 ∣ n := 
sorry

end largest_divisor_of_square_l104_104073


namespace a_1_value_l104_104353

variable (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ)

axiom a_n_def : ∀ n ≥ 2, a n + 2 * (S n) * (S (n - 1)) = 0
axiom S_5_value : S 5 = 1/11
axiom summation_def : ∀ k ≥ 1, S k = S (k - 1) + a k

theorem a_1_value : a 1 = 1/3 := by
  sorry

end a_1_value_l104_104353


namespace solve_inequality_l104_104318

open Set

-- Define the inequality
def inequality (a x : ℝ) : Prop := a * x^2 - (a + 1) * x + 1 < 0

-- Define the solution sets for different cases of a
def solution_set (a : ℝ) : Set ℝ :=
  if a = 0 then {x | x > 1}
  else if a < 0 then {x | x < 1 / a ∨ x > 1}
  else if 0 < a ∧ a < 1 then {x | 1 < x ∧ x < 1 / a}
  else if a > 1 then {x | 1 / a < x ∧ x < 1}
  else ∅

-- State the theorem
theorem solve_inequality (a : ℝ) : 
  {x : ℝ | inequality a x} = solution_set a :=
by
  sorry

end solve_inequality_l104_104318


namespace Rachel_brought_25_cookies_l104_104709

def Mona_cookies : ℕ := 20
def Jasmine_cookies : ℕ := Mona_cookies - 5
def Total_cookies : ℕ := 60

theorem Rachel_brought_25_cookies : (Total_cookies - (Mona_cookies + Jasmine_cookies) = 25) :=
by
  sorry

end Rachel_brought_25_cookies_l104_104709


namespace fraction_august_tips_l104_104078

variable (A : ℝ) -- Define the average monthly tips A for March, April, May, June, July, and September
variable (august_tips : ℝ) -- Define the tips for August
variable (total_tips : ℝ) -- Define the total tips for all months

-- Define the conditions
def condition_average_tips : Prop := total_tips = 12 * A
def condition_august_tips : Prop := august_tips = 6 * A

-- The theorem we need to prove
theorem fraction_august_tips :
  condition_average_tips A total_tips →
  condition_august_tips A august_tips →
  (august_tips / total_tips) = (1 / 2) :=
by
  intros h_avg h_aug
  rw [condition_average_tips] at h_avg
  rw [condition_august_tips] at h_aug
  rw [h_avg, h_aug]
  simp
  sorry

end fraction_august_tips_l104_104078


namespace domain_of_sqrt_function_l104_104561

theorem domain_of_sqrt_function (m : ℝ) : (∀ x : ℝ, mx^2 + mx + 1 ≥ 0) ↔ 0 ≤ m ∧ m ≤ 4 := sorry

end domain_of_sqrt_function_l104_104561


namespace alice_current_age_l104_104551

theorem alice_current_age (a b : ℕ) 
  (h1 : a + 8 = 2 * (b + 8)) 
  (h2 : (a - 10) + (b - 10) = 21) : 
  a = 30 := 
by 
  sorry

end alice_current_age_l104_104551


namespace find_number_l104_104614

theorem find_number (x : ℕ) (h : 23 + x = 34) : x = 11 :=
by
  sorry

end find_number_l104_104614


namespace shifted_polynomial_sum_l104_104663

theorem shifted_polynomial_sum (a b c : ℝ) :
  (∀ x : ℝ, (3 * x^2 + 2 * x + 5) = (a * (x + 5)^2 + b * (x + 5) + c)) →
  a + b + c = 125 :=
by
  sorry

end shifted_polynomial_sum_l104_104663


namespace charge_per_block_l104_104546

noncomputable def family_vacation_cost : ℝ := 1000
noncomputable def family_members : ℝ := 5
noncomputable def walk_start_fee : ℝ := 2
noncomputable def dogs_walked : ℝ := 20
noncomputable def total_blocks : ℝ := 128

theorem charge_per_block : 
  (family_vacation_cost / family_members) = 200 →
  (dogs_walked * walk_start_fee) = 40 →
  ((family_vacation_cost / family_members) - (dogs_walked * walk_start_fee)) = 160 →
  (((family_vacation_cost / family_members) - (dogs_walked * walk_start_fee)) / total_blocks) = 1.25 :=
by intros h1 h2 h3; sorry

end charge_per_block_l104_104546


namespace math_problem_l104_104214

theorem math_problem (a : ℝ) (h : a^2 - 4 * a + 3 = 0) (h_ne : a ≠ 2 ∧ a ≠ 3 ∧ a ≠ -3) :
  (9 - 3 * a) / (2 * a - 4) / (a + 2 - 5 / (a - 2)) = -3 / 8 :=
sorry

end math_problem_l104_104214


namespace cannot_make_120_cents_with_6_coins_l104_104348

def Coin := ℕ → ℕ -- represents a number of each type of coin

noncomputable def coin_value (c : Coin) : ℕ :=
  c 0 * 1 + c 1 * 5 + c 2 * 10 + c 3 * 25

def total_coins (c : Coin) : ℕ :=
  c 0 + c 1 + c 2 + c 3

theorem cannot_make_120_cents_with_6_coins (c : Coin) (h1 : total_coins c = 6) :
  coin_value c ≠ 120 :=
sorry

end cannot_make_120_cents_with_6_coins_l104_104348


namespace quadratic_eq_solutions_l104_104877

theorem quadratic_eq_solutions (x : ℝ) : x * (x + 1) = 3 * (x + 1) ↔ x = -1 ∨ x = 3 := by
  sorry

end quadratic_eq_solutions_l104_104877


namespace surface_area_of_sphere_with_diameter_4_l104_104203

theorem surface_area_of_sphere_with_diameter_4 :
    let diameter := 4
    let radius := diameter / 2
    let surface_area := 4 * Real.pi * radius^2
    surface_area = 16 * Real.pi :=
by
  -- Sorry is used in place of the actual proof.
  sorry

end surface_area_of_sphere_with_diameter_4_l104_104203


namespace sum_of_primes_1_to_20_l104_104294

theorem sum_of_primes_1_to_20 : 
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 := by
  sorry

end sum_of_primes_1_to_20_l104_104294


namespace part1_part2_l104_104587

variable {α : Type*}
def A : Set ℝ := {x | 0 < x ∧ x < 9}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

-- Part (1)
theorem part1 : B 5 ∩ A = {x | 6 ≤ x ∧ x < 9} := 
sorry

-- Part (2)
theorem part2 (m : ℝ): A ∩ B m = B m ↔ m < 5 :=
sorry

end part1_part2_l104_104587


namespace algebraic_expression_equals_one_l104_104875

variable (m n : ℝ)

theorem algebraic_expression_equals_one
  (hm : m ≠ 0)
  (hn : n ≠ 0)
  (h_eq : m - n = 1 / 2) :
  (m^2 - n^2) / (2 * m^2 + 2 * m * n) / (m - (2 * m * n - n^2) / m) = 1 :=
by
  sorry

end algebraic_expression_equals_one_l104_104875


namespace number_of_bottom_row_bricks_l104_104445

theorem number_of_bottom_row_bricks :
  ∃ (x : ℕ), (x + (x - 1) + (x - 2) + (x - 3) + (x - 4) = 100) ∧ x = 22 :=
by 
  sorry

end number_of_bottom_row_bricks_l104_104445


namespace value_of_a2_l104_104547

theorem value_of_a2 (a : ℕ → ℤ) (h1 : ∀ n : ℕ, a (n + 1) = a n + 2)
  (h2 : ∃ r : ℤ, a 3 = r * a 1 ∧ a 4 = r * a 3) :
  a 2 = -6 :=
by
  sorry

end value_of_a2_l104_104547


namespace find_unknown_number_l104_104133

theorem find_unknown_number (x : ℝ) (h : (45 + 23 / x) * x = 4028) : x = 89 :=
sorry

end find_unknown_number_l104_104133


namespace circle_value_of_m_l104_104304

theorem circle_value_of_m (m : ℝ) : (∃ a b r : ℝ, r > 0 ∧ (x - a) ^ 2 + (y - b) ^ 2 = r ^ 2) ↔ m < 1/2 := by
  sorry

end circle_value_of_m_l104_104304


namespace next_podcast_duration_l104_104795

def minutes_in_an_hour : ℕ := 60

def first_podcast_minutes : ℕ := 45
def second_podcast_minutes : ℕ := 2 * first_podcast_minutes
def third_podcast_minutes : ℕ := 105
def fourth_podcast_minutes : ℕ := 60

def total_podcast_minutes : ℕ := first_podcast_minutes + second_podcast_minutes + third_podcast_minutes + fourth_podcast_minutes

def drive_minutes : ℕ := 6 * minutes_in_an_hour

theorem next_podcast_duration :
  (drive_minutes - total_podcast_minutes) / minutes_in_an_hour = 1 :=
by
  sorry

end next_podcast_duration_l104_104795


namespace special_op_2_4_5_l104_104342

def special_op (a b c : ℝ) : ℝ := b ^ 2 - 4 * a * c

theorem special_op_2_4_5 : special_op 2 4 5 = -24 := by
  sorry

end special_op_2_4_5_l104_104342


namespace problem1_problem2_problem3_problem4_l104_104503

-- Problem (1)
theorem problem1 : 6 - -2 + -4 - 3 = 1 :=
by sorry

-- Problem (2)
theorem problem2 : 8 / -2 * (1 / 3 : ℝ) * (-(1 + 1/2: ℝ)) = 2 :=
by sorry

-- Problem (3)
theorem problem3 : (13 + (2 / 7 - 1 / 14) * 56) / (-1 / 4) = -100 :=
by sorry

-- Problem (4)
theorem problem4 : 
  |-(5 / 6 : ℝ)| / ((-(3 + 1 / 5: ℝ)) / (-4)^2 + (-7 / 4) * (4 / 7)) = -(25 / 36) :=
by sorry

end problem1_problem2_problem3_problem4_l104_104503


namespace arrangement_count_l104_104827

-- Definitions corresponding to the given problem conditions
def numMathBooks : Nat := 3
def numPhysicsBooks : Nat := 2
def numChemistryBooks : Nat := 1
def totalArrangements : Nat := 2592

-- Statement of the theorem
theorem arrangement_count :
  ∃ (numM numP numC : Nat), 
    numM = 3 ∧ 
    numP = 2 ∧ 
    numC = 1 ∧ 
    (numM + numP + numC = 6) ∧ 
    allMathBooksAdjacent ∧ 
    physicsBooksNonAdjacent → 
    totalArrangements = 2592 :=
by
  sorry

end arrangement_count_l104_104827


namespace solution_set_of_inequality_l104_104405

theorem solution_set_of_inequality : 
  {x : ℝ | x * (x + 3) ≥ 0} = {x : ℝ | x ≥ 0 ∨ x ≤ -3} := 
by sorry

end solution_set_of_inequality_l104_104405


namespace log_equality_implies_exp_equality_l104_104127

theorem log_equality_implies_exp_equality (x y z a : ℝ) (h : (x * (y + z - x)) / (Real.log x) = (y * (x + z - y)) / (Real.log y) ∧ (y * (x + z - y)) / (Real.log y) = (z * (x + y - z)) / (Real.log z)) :
  x^y * y^x = z^x * x^z ∧ z^x * x^z = y^z * z^y :=
by
  sorry

end log_equality_implies_exp_equality_l104_104127


namespace set_operations_l104_104434

open Set

variable (U : Set ℕ) (A B : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5, 6})
variable (hA : A = {2, 4, 5})
variable (hB : B = {1, 2, 5})

theorem set_operations :
  (A ∩ B = {2, 5}) ∧ (A ∪ (U \ B) = {2, 3, 4, 5, 6}) :=
by
  sorry

end set_operations_l104_104434


namespace length_of_AB_l104_104258

theorem length_of_AB
  (AP PB AQ QB : ℝ) 
  (h_ratioP : 5 * AP = 3 * PB)
  (h_ratioQ : 3 * AQ = 2 * QB)
  (h_PQ : AQ = AP + 3 ∧ QB = PB - 3)
  (h_PQ_length : AQ - AP = 3)
  : AP + PB = 120 :=
by {
  sorry
}

end length_of_AB_l104_104258


namespace total_amount_owed_l104_104592

-- Conditions
def borrowed_amount : ℝ := 500
def monthly_interest_rate : ℝ := 0.02
def months_not_paid : ℕ := 3

-- Compounded monthly formula
def amount_after_n_months (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r) ^ n

-- Theorem statement
theorem total_amount_owed :
  amount_after_n_months borrowed_amount monthly_interest_rate months_not_paid = 530.604 :=
by
  -- Proof to be filled in here
  sorry

end total_amount_owed_l104_104592


namespace smallest_area_right_triangle_l104_104753

theorem smallest_area_right_triangle (a b : ℕ) (ha : a = 7) (hb : b = 10): 
  ∃ (A : ℕ), A = 35 :=
  by
    have hab := 1/2 * a * b
    sorry

-- Note: "sorry" is used as a placeholder for the proof.

end smallest_area_right_triangle_l104_104753


namespace probability_of_number_less_than_three_l104_104844

theorem probability_of_number_less_than_three :
  let faces : Finset ℕ := {1, 2, 3, 4, 5, 6}
  let favorable_outcomes : Finset ℕ := {1, 2}
  (favorable_outcomes.card : ℚ) / (faces.card : ℚ) = 1 / 3 :=
by
  -- This is the placeholder for the actual proof.
  sorry

end probability_of_number_less_than_three_l104_104844


namespace sale_in_third_month_l104_104855

def average_sale (s1 s2 s3 s4 s5 s6 : ℕ) : ℕ :=
  (s1 + s2 + s3 + s4 + s5 + s6) / 6

theorem sale_in_third_month
  (S1 S2 S3 S4 S5 S6 : ℕ)
  (h1 : S1 = 6535)
  (h2 : S2 = 6927)
  (h4 : S4 = 7230)
  (h5 : S5 = 6562)
  (h6 : S6 = 4891)
  (havg : average_sale S1 S2 S3 S4 S5 S6 = 6500) :
  S3 = 6855 := 
sorry

end sale_in_third_month_l104_104855


namespace tire_price_l104_104163

theorem tire_price (payment : ℕ) (price_ratio : ℕ → ℕ → Prop)
  (h1 : payment = 345)
  (h2 : price_ratio 3 1)
  : ∃ x : ℕ, x = 99 := 
sorry

end tire_price_l104_104163


namespace cos_difference_l104_104084

theorem cos_difference (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 1 / 2) 
  (h2 : Real.cos A + Real.cos B = 3 / 2) : 
  Real.cos (A - B) = 1 / 4 :=
by
  sorry

end cos_difference_l104_104084


namespace find_T_value_l104_104247

theorem find_T_value (x y : ℤ) (R : ℤ) (h : R = 30) (h2 : (R / 2) * x * y = 21 * x + 20 * y - 13) :
    x = 3 ∧ y = 2 → x * y = 6 := by
  sorry

end find_T_value_l104_104247


namespace polynomial_characterization_l104_104011

noncomputable def homogeneous_polynomial (P : ℝ → ℝ → ℝ) (n : ℕ) :=
  ∀ t x y : ℝ, P (t * x) (t * y) = t^n * P x y

def polynomial_condition (P : ℝ → ℝ → ℝ) :=
  ∀ a b c : ℝ, P (a + b) c + P (b + c) a + P (c + a) b = 0

def P_value (P : ℝ → ℝ → ℝ) :=
  P 1 0 = 1

theorem polynomial_characterization (P : ℝ → ℝ → ℝ) (n : ℕ) :
  homogeneous_polynomial P n →
  polynomial_condition P →
  P_value P →
  ∃ A : ℝ → ℝ → ℝ, ∀ x y : ℝ, P x y = (x + y)^(n - 1) * (x - 2 * y) :=
by
  sorry

end polynomial_characterization_l104_104011


namespace minimum_additional_marbles_l104_104876

theorem minimum_additional_marbles (friends marbles : ℕ) (h_friends : friends = 12) (h_marbles : marbles = 34) : 
  ∃ additional_marbles : ℕ, additional_marbles = 44 :=
by
  -- The formal proof would go here.
  sorry

end minimum_additional_marbles_l104_104876


namespace andrey_wins_iff_irreducible_fraction_l104_104819

def is_irreducible_fraction (p : ℝ) : Prop :=
  ∃ m n : ℕ, p = m / 2^n ∧ gcd m (2^n) = 1

def can_reach_0_or_1 (p : ℝ) : Prop :=
  ∀ move : ℝ, ∃ dir : ℝ, (p + dir * move = 0 ∨ p + dir * move = 1)

theorem andrey_wins_iff_irreducible_fraction (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) :
  (∃ move_sequence : ℕ → ℝ, ∀ n, can_reach_0_or_1 (move_sequence n)) ↔ is_irreducible_fraction p :=
sorry

end andrey_wins_iff_irreducible_fraction_l104_104819


namespace solution_set_absolute_value_l104_104451

theorem solution_set_absolute_value (x : ℝ) : 
  (|2 - x| ≥ 1) ↔ (x ≤ 1 ∨ x ≥ 3) :=
by
  -- Proof goes here
  sorry

end solution_set_absolute_value_l104_104451


namespace dodecagon_area_constraint_l104_104754

theorem dodecagon_area_constraint 
    (a : ℕ) -- side length of the square
    (N : ℕ) -- a large number with 2017 digits, breaking it down as 2 * (10^2017 - 1) / 9
    (hN : N = (2 * (10^2017 - 1)) / 9) 
    (H : ∃ n : ℕ, (n * n) = 3 * a^2 / 2) :
    False :=
by
    sorry

end dodecagon_area_constraint_l104_104754


namespace smallest_missing_unit_digit_l104_104738

theorem smallest_missing_unit_digit :
  (∀ n, n ∈ [0, 1, 4, 5, 6, 9]) → ∃ smallest_digit, smallest_digit = 2 :=
by
  sorry

end smallest_missing_unit_digit_l104_104738


namespace no_such_function_l104_104215

theorem no_such_function (f : ℕ → ℕ) : ¬ (∀ n : ℕ, f (f n) = n + 2019) :=
sorry

end no_such_function_l104_104215


namespace simplify_expression_l104_104634

variables (a b : ℝ)
noncomputable def x := (1 / 2) * (Real.sqrt (a / b) - Real.sqrt (b / a))

theorem simplify_expression (ha : a > 0) (hb : b > 0) :
  (2 * a * Real.sqrt (1 + x a b ^ 2)) / (x a b + Real.sqrt (1 + x a b ^ 2)) = a + b :=
sorry

end simplify_expression_l104_104634


namespace min_slope_of_tangent_l104_104888

def f (x : ℝ) : ℝ := x^3 + 3 * x^2 + 6 * x - 10

theorem min_slope_of_tangent : (∀ x : ℝ, 3 * (x + 1)^2 + 3 ≥ 3) :=
by 
  sorry

end min_slope_of_tangent_l104_104888


namespace vertex_of_parabola_l104_104690

theorem vertex_of_parabola :
  ∃ h k : ℝ, (∀ x : ℝ, 3 * (x + 4)^2 - 9 = 3 * (x - h)^2 + k) ∧ (h, k) = (-4, -9) :=
by
  sorry

end vertex_of_parabola_l104_104690


namespace range_of_a_l104_104398

noncomputable def f (a x : ℝ) : ℝ := (1/3) * x^3 + x^2 + a * x + 1

def is_monotonic_increasing (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x ∈ Set.Icc (-2) a, 0 ≤ (deriv f) x

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-2) a, 0 ≤ (deriv (f a)) x) → 1 ≤ a := 
sorry

end range_of_a_l104_104398


namespace right_triangle_ineq_l104_104494

-- Definitions based on conditions in (a)
variables {a b c m f : ℝ}
variable (h_a : a ≥ 0)
variable (h_b : b ≥ 0)
variable (h_c : c > 0)
variable (h_a_b : a ≤ b)
variable (h_triangle : c = Real.sqrt (a^2 + b^2))
variable (h_m : m = a * b / c)
variable (h_f : f = (Real.sqrt 2 * a * b) / (a + b))

-- Proof goal based on the problem in (c)
theorem right_triangle_ineq : m + f ≤ c :=
sorry

end right_triangle_ineq_l104_104494


namespace tractor_planting_rate_l104_104290

theorem tractor_planting_rate
  (acres : ℕ) (days : ℕ) (first_crew_tractors : ℕ) (first_crew_days : ℕ) 
  (second_crew_tractors : ℕ) (second_crew_days : ℕ) 
  (total_acres : ℕ) (total_days : ℕ) 
  (first_crew_days_calculated : ℕ) 
  (second_crew_days_calculated : ℕ) 
  (total_tractor_days : ℕ) 
  (acres_per_tractor_day : ℕ) :
  total_acres = acres → 
  total_days = days → 
  first_crew_tractors * first_crew_days = first_crew_days_calculated → 
  second_crew_tractors * second_crew_days = second_crew_days_calculated → 
  first_crew_days_calculated + second_crew_days_calculated = total_tractor_days → 
  total_acres / total_tractor_days = acres_per_tractor_day → 
  acres_per_tractor_day = 68 :=
by
  intros
  sorry

end tractor_planting_rate_l104_104290


namespace pascal_triangle_row51_sum_l104_104945

theorem pascal_triangle_row51_sum : (Nat.choose 51 4) + (Nat.choose 51 6) = 18249360 :=
by
  sorry

end pascal_triangle_row51_sum_l104_104945


namespace inequality_proof_l104_104013

-- Define the context of non-negative real numbers and sum to 1
variable {x y z : ℝ}
variable (h_nonneg : 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z)
variable (h_sum : x + y + z = 1)

-- State the theorem to be proved
theorem inequality_proof (h_nonneg : 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z) (h_sum : x + y + z = 1) :
    0 ≤ x * y + y * z + z * x - 2 * x * y * z ∧ x * y + y * z + z * x - 2 * x * y * z ≤ 7 / 27 :=
    sorry

end inequality_proof_l104_104013


namespace evaluate_expression_l104_104771

theorem evaluate_expression : 
  (16 = 2^4) → 
  (32 = 2^5) → 
  (16^24 / 32^12 = 8^12) :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end evaluate_expression_l104_104771


namespace find_a_plus_b_l104_104480

theorem find_a_plus_b (a b : ℝ) 
  (h_a : a^3 - 3 * a^2 + 5 * a = 1) 
  (h_b : b^3 - 3 * b^2 + 5 * b = 5) : 
  a + b = 2 := 
sorry

end find_a_plus_b_l104_104480


namespace probabilities_equal_l104_104770

def roll := {n : ℕ // n ≥ 1 ∧ n ≤ 6}

def is_successful (r : roll) : Prop := r.val ≥ 3

def prob_successful : ℚ := 4 / 6

def prob_unsuccessful : ℚ := 1 - prob_successful

def prob_at_least_one_success_two_rolls : ℚ := 1 - (prob_unsuccessful ^ 2)

def prob_at_least_two_success_four_rolls : ℚ :=
  let zero_success := prob_unsuccessful ^ 4
  let one_success := 4 * (prob_unsuccessful ^ 3) * prob_successful
  1 - (zero_success + one_success)

theorem probabilities_equal :
  prob_at_least_one_success_two_rolls = prob_at_least_two_success_four_rolls := by
  sorry

end probabilities_equal_l104_104770


namespace games_played_by_player_3_l104_104331

theorem games_played_by_player_3 (games_1 games_2 : ℕ) (rotation_system : ℕ) :
  games_1 = 10 → games_2 = 21 →
  rotation_system = (games_2 - games_1) →
  rotation_system = 11 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end games_played_by_player_3_l104_104331


namespace find_multiple_l104_104698

-- Given conditions
variables (P W m : ℕ)
variables (h1 : P * 24 = W) (h2 : m * P * 6 = W / 2)

-- The statement to prove
theorem find_multiple (P W m : ℕ) (h1 : P * 24 = W) (h2 : m * P * 6 = W / 2) : m = 4 :=
by
  sorry

end find_multiple_l104_104698


namespace minimum_value_of_f_l104_104422

def f (x : ℝ) : ℝ := x^2 - 2 * x

theorem minimum_value_of_f : ∃ x : ℝ, (∀ y : ℝ, f x ≤ f y) ∧ f x = -1 :=
by
  sorry

end minimum_value_of_f_l104_104422


namespace lowest_score_l104_104650

theorem lowest_score (max_mark : ℕ) (n_tests : ℕ) (avg_mark : ℕ) (h_avg : n_tests * avg_mark = 352) (h_max : ∀ k, k < n_tests → k ≤ max_mark) :
  ∃ x, (x ≤ max_mark ∧ (3 * max_mark + x) = 352) ∧ x = 52 :=
by
  sorry

end lowest_score_l104_104650


namespace minimum_elapsed_time_l104_104899

theorem minimum_elapsed_time : 
  let initial_time := 45  -- in minutes
  let final_time := 3 * 60 + 30  -- 3 hours 30 minutes in minutes
  let elapsed_time := final_time - initial_time
  elapsed_time = 2 * 60 + 45 :=
by
  sorry

end minimum_elapsed_time_l104_104899


namespace Sam_age_l104_104192

theorem Sam_age (S D : ℕ) (h1 : S + D = 54) (h2 : S = D / 2) : S = 18 :=
by
  -- Proof omitted
  sorry

end Sam_age_l104_104192


namespace set_intersection_l104_104870

theorem set_intersection :
  let A := {x : ℝ | 0 < x}
  let B := {x : ℝ | -1 ≤ x ∧ x < 3}
  A ∩ B = {x : ℝ | 0 < x ∧ x < 3} := 
by
  sorry

end set_intersection_l104_104870


namespace hexagon_area_l104_104108

theorem hexagon_area (ABCDEF : Type) (l : ℕ) (h : l = 3) (p q : ℕ)
  (area_hexagon : ℝ) (area_formula : area_hexagon = Real.sqrt p + Real.sqrt q) :
  p + q = 54 := by
  sorry

end hexagon_area_l104_104108


namespace mans_speed_against_current_l104_104021

theorem mans_speed_against_current (V_with_current V_current V_against : ℝ) (h1 : V_with_current = 21) (h2 : V_current = 4.3) : 
  V_against = V_with_current - 2 * V_current := 
sorry

end mans_speed_against_current_l104_104021


namespace shorts_cost_l104_104813

theorem shorts_cost :
  let football_cost := 3.75
  let shoes_cost := 11.85
  let zachary_money := 10
  let additional_needed := 8
  ∃ S, football_cost + shoes_cost + S = zachary_money + additional_needed ∧ S = 2.40 :=
by
  sorry

end shorts_cost_l104_104813


namespace number_of_bowls_l104_104321

-- Let n be the number of bowls on the table.
variable (n : ℕ)

-- Condition 1: There are n bowls, and each contain some grapes.
-- Condition 2: Adding 8 grapes to each of 12 specific bowls increases the average number of grapes in all bowls by 6.
-- Let's formalize the condition given in the problem
theorem number_of_bowls (h1 : 12 * 8 = 96) (h2 : 6 * n = 96) : n = 16 :=
by
  -- omitting the proof here
  sorry

end number_of_bowls_l104_104321


namespace total_profit_correct_l104_104014

-- We define the conditions
variables (a m : ℝ)

-- The item's cost per piece
def cost_per_piece : ℝ := a
-- The markup percentage
def markup_percentage : ℝ := 0.20
-- The discount percentage
def discount_percentage : ℝ := 0.10
-- The number of pieces sold
def pieces_sold : ℝ := m

-- Definitions derived from conditions
def selling_price_markup : ℝ := cost_per_piece a * (1 + markup_percentage)
def selling_price_discount : ℝ := selling_price_markup a * (1 - discount_percentage)
def profit_per_piece : ℝ := selling_price_discount a - cost_per_piece a
def total_profit : ℝ := profit_per_piece a * pieces_sold m

theorem total_profit_correct (a m : ℝ) : total_profit a m = 0.08 * a * m :=
by sorry

end total_profit_correct_l104_104014


namespace stream_speed_l104_104092

variable (D v : ℝ)

/--
The time taken by a man to row his boat upstream is twice the time taken by him to row the same distance downstream.
If the speed of the boat in still water is 63 kmph, prove that the speed of the stream is 21 kmph.
-/
theorem stream_speed (h : D / (63 - v) = 2 * (D / (63 + v))) : v = 21 := 
sorry

end stream_speed_l104_104092
