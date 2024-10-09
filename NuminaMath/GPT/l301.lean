import Mathlib

namespace pear_sales_l301_30140

theorem pear_sales (sale_afternoon : ℕ) (h1 : sale_afternoon = 260)
  (h2 : ∃ sale_morning : ℕ, sale_afternoon = 2 * sale_morning) :
  sale_afternoon / 2 + sale_afternoon = 390 :=
by
  sorry

end pear_sales_l301_30140


namespace eval_five_over_two_l301_30147

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 2^x - 2 else Real.log (x - 1) / Real.log 2

theorem eval_five_over_two : f (5 / 2) = -1 := by
  sorry

end eval_five_over_two_l301_30147


namespace Kelly_egg_price_l301_30184

/-- Kelly has 8 chickens, and each chicken lays 3 eggs per day.
Kelly makes $280 in 4 weeks by selling all the eggs.
We want to prove that Kelly sells a dozen eggs for $5. -/
theorem Kelly_egg_price (chickens : ℕ) (eggs_per_day_per_chicken : ℕ) (earnings_in_4_weeks : ℕ)
  (days_in_4_weeks : ℕ) (eggs_per_dozen : ℕ) (price_per_dozen : ℕ) :
  chickens = 8 →
  eggs_per_day_per_chicken = 3 →
  earnings_in_4_weeks = 280 →
  days_in_4_weeks = 28 →
  eggs_per_dozen = 12 →
  price_per_dozen = earnings_in_4_weeks / ((chickens * eggs_per_day_per_chicken * days_in_4_weeks) / eggs_per_dozen) →
  price_per_dozen = 5 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end Kelly_egg_price_l301_30184


namespace range_of_x_for_fx1_positive_l301_30190

-- Define the conditions
def is_even (f : ℝ → ℝ) := ∀ x, f x = f (-x)
def is_monotonic_decreasing_on_nonneg (f : ℝ → ℝ) := ∀ ⦃x y⦄, 0 ≤ x → x ≤ y → f y ≤ f x
def f_at_2_eq_zero (f : ℝ → ℝ) := f 2 = 0

-- Define the problem statement that needs to be proven
theorem range_of_x_for_fx1_positive (f : ℝ → ℝ) :
  is_even f →
  is_monotonic_decreasing_on_nonneg f →
  f_at_2_eq_zero f →
  ∀ x, f (x - 1) > 0 ↔ -1 < x ∧ x < 3 :=
by sorry

end range_of_x_for_fx1_positive_l301_30190


namespace new_number_of_groups_l301_30168

-- Define the number of students
def total_students : ℕ := 2808

-- Define the initial and new number of groups
def initial_groups (n : ℕ) : ℕ := n + 4
def new_groups (n : ℕ) : ℕ := n

-- Condition: Fewer than 30 students per new group
def fewer_than_30_students_per_group (n : ℕ) : Prop :=
  total_students / n < 30

-- Condition: n and n + 4 must be divisors of total_students
def is_divisor (d : ℕ) (a : ℕ) : Prop :=
  a % d = 0

def valid_group_numbers (n : ℕ) : Prop :=
  is_divisor n total_students ∧ is_divisor (n + 4) total_students ∧ n > 93

-- The main theorem
theorem new_number_of_groups : ∃ n : ℕ, valid_group_numbers n ∧ fewer_than_30_students_per_group n ∧ n = 104 :=
by
  sorry

end new_number_of_groups_l301_30168


namespace fraction_of_shaded_area_l301_30103

theorem fraction_of_shaded_area
  (total_smaller_rectangles : ℕ)
  (shaded_smaller_rectangles : ℕ)
  (h1 : total_smaller_rectangles = 18)
  (h2 : shaded_smaller_rectangles = 4) :
  (shaded_smaller_rectangles : ℚ) / total_smaller_rectangles = 1 / 4 := 
sorry

end fraction_of_shaded_area_l301_30103


namespace resulting_solution_percentage_l301_30128

theorem resulting_solution_percentage (w_original: ℝ) (w_replaced: ℝ) (c_original: ℝ) (c_new: ℝ) :
  c_original = 0.9 → w_replaced = 0.7142857142857143 → c_new = 0.2 →
  (0.2571428571428571 + 0.14285714285714285) / (0.2857142857142857 + 0.7142857142857143) * 100 = 40 := 
by
  intros h1 h2 h3
  sorry

end resulting_solution_percentage_l301_30128


namespace circulation_ratio_l301_30106

theorem circulation_ratio (A C_1971 C_total : ℕ) 
(hC1971 : C_1971 = 4 * A) 
(hCtotal : C_total = C_1971 + 9 * A) : 
(C_1971 : ℚ) / (C_total : ℚ) = 4 / 13 := 
sorry

end circulation_ratio_l301_30106


namespace swan_percentage_not_ducks_l301_30166

theorem swan_percentage_not_ducks (total_birds geese swans herons ducks : ℝ)
  (h_total : total_birds = 100)
  (h_geese : geese = 0.30 * total_birds)
  (h_swans : swans = 0.20 * total_birds)
  (h_herons : herons = 0.20 * total_birds)
  (h_ducks : ducks = 0.30 * total_birds) :
  (swans / (total_birds - ducks) * 100) = 28.57 :=
by
  sorry

end swan_percentage_not_ducks_l301_30166


namespace first_car_distance_l301_30101

-- Definitions for conditions
variable (x : ℝ) -- distance the first car ran before taking the right turn
def distance_apart_initial := 150 -- initial distance between the cars
def distance_first_car_main_road := 2 * x -- total distance first car ran on the main road
def distance_second_car := 62 -- distance the second car ran due to breakdown
def distance_between_cars := 38 -- distance between the cars after running 

-- Proof (statement only, no solution steps)
theorem first_car_distance (hx : distance_apart_initial = distance_first_car_main_road + distance_second_car + distance_between_cars) : 
  x = 25 :=
by
  unfold distance_apart_initial distance_first_car_main_road distance_second_car distance_between_cars at hx
  -- Implementation placeholder
  sorry

end first_car_distance_l301_30101


namespace min_vertical_segment_length_l301_30117

noncomputable def minVerticalSegLength : ℤ → ℝ 
| x => abs (2 * abs x + x^2 + 4 * x + 1)

theorem min_vertical_segment_length :
  ∀ x : ℤ, minVerticalSegLength x = 1 ↔  x = 0 := 
by
  intros x
  sorry

end min_vertical_segment_length_l301_30117


namespace jessica_watermelons_l301_30124

theorem jessica_watermelons (original : ℕ) (eaten : ℕ) (remaining : ℕ) 
    (h1 : original = 35) 
    (h2 : eaten = 27) 
    (h3 : remaining = original - eaten) : 
  remaining = 8 := 
by {
    -- This is where the proof would go
    sorry
}

end jessica_watermelons_l301_30124


namespace heartsuit_ratio_l301_30129

def heartsuit (n m : ℕ) : ℕ := n^4 * m^3

theorem heartsuit_ratio :
  (heartsuit 2 4) / (heartsuit 4 2) = 1 / 2 := by
  sorry

end heartsuit_ratio_l301_30129


namespace find_number_l301_30187

theorem find_number 
  (x : ℝ)
  (h : (258 / 100 * x) / 6 = 543.95) :
  x = 1265 :=
sorry

end find_number_l301_30187


namespace total_infections_second_wave_l301_30120

theorem total_infections_second_wave (cases_per_day_first_wave : ℕ)
                                     (factor_increase : ℕ)
                                     (duration_weeks : ℕ)
                                     (days_per_week : ℕ) :
                                     cases_per_day_first_wave = 300 →
                                     factor_increase = 4 →
                                     duration_weeks = 2 →
                                     days_per_week = 7 →
                                     (duration_weeks * days_per_week) * (cases_per_day_first_wave + factor_increase * cases_per_day_first_wave) = 21000 :=
by 
  intros h1 h2 h3 h4
  sorry

end total_infections_second_wave_l301_30120


namespace find_number_l301_30152

theorem find_number (x : ℝ) (h : x + (2/3) * x + 1 = 10) : x = 27/5 := 
by
  sorry

end find_number_l301_30152


namespace intersection_line_circle_diameter_l301_30164

noncomputable def length_of_AB : ℝ := 2

theorem intersection_line_circle_diameter 
  (x y : ℝ)
  (h_line : x - 2*y - 1 = 0)
  (h_circle : (x - 1)^2 + y^2 = 1) :
  |(length_of_AB)| = 2 := 
sorry

end intersection_line_circle_diameter_l301_30164


namespace fraction_upgraded_l301_30194

theorem fraction_upgraded :
  ∀ (N U : ℕ), 24 * N = 6 * U → (U : ℚ) / (24 * N + U) = 1 / 7 :=
by
  intros N U h_eq
  sorry

end fraction_upgraded_l301_30194


namespace strawberry_blueberry_price_difference_l301_30197

theorem strawberry_blueberry_price_difference
  (s p t : ℕ → ℕ)
  (strawberries_sold blueberries_sold strawberries_sale_revenue blueberries_sale_revenue strawberries_loss blueberries_loss : ℕ)
  (h1 : strawberries_sold = 54)
  (h2 : strawberries_sale_revenue = 216)
  (h3 : strawberries_loss = 108)
  (h4 : blueberries_sold = 36)
  (h5 : blueberries_sale_revenue = 144)
  (h6 : blueberries_loss = 72)
  (h7 : p strawberries_sold = strawberries_sale_revenue + strawberries_loss)
  (h8 : p blueberries_sold = blueberries_sale_revenue + blueberries_loss)
  : p strawberries_sold / strawberries_sold - p blueberries_sold / blueberries_sold = 0 :=
by
  sorry

end strawberry_blueberry_price_difference_l301_30197


namespace shorter_piece_length_l301_30177

def wireLength := 150
def ratioLongerToShorter := 5 / 8

theorem shorter_piece_length : ∃ x : ℤ, x + (5 / 8) * x = wireLength ∧ x = 92 := by
  sorry

end shorter_piece_length_l301_30177


namespace fraction_inequality_solution_l301_30113

theorem fraction_inequality_solution (x : ℝ) :
  (x < -5 ∨ x ≥ 2) ↔ (x-2) / (x+5) ≥ 0 :=
sorry

end fraction_inequality_solution_l301_30113


namespace simplify_polynomial_l301_30145

theorem simplify_polynomial : 
  ∀ (x : ℝ), 
    (2 * x + 1) ^ 5 - 5 * (2 * x + 1) ^ 4 + 10 * (2 * x + 1) ^ 3 - 10 * (2 * x + 1) ^ 2 + 5 * (2 * x + 1) - 1 
    = 32 * x ^ 5 := 
by sorry

end simplify_polynomial_l301_30145


namespace max_product_is_negative_one_l301_30136

def f (x : ℝ) : ℝ := sorry    -- Assume some function f
def g (x : ℝ) : ℝ := sorry    -- Assume some function g

theorem max_product_is_negative_one (h_f_range : ∀ y, 1 ≤ y ∧ y ≤ 6 → ∃ x, f x = y) 
    (h_g_range : ∀ y, -4 ≤ y ∧ y ≤ -1 → ∃ x, g x = y) : 
    ∃ b, b = -1 ∧ ∀ x, f x * g x ≤ b :=
sorry

end max_product_is_negative_one_l301_30136


namespace coeff_x3_in_expansion_l301_30180

noncomputable def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

theorem coeff_x3_in_expansion :
  (2 : ℚ)^(4 - 2) * binomial_coeff 4 2 = 24 := by 
  sorry

end coeff_x3_in_expansion_l301_30180


namespace number_of_students_in_third_grade_l301_30169

theorem number_of_students_in_third_grade
    (total_students : ℕ)
    (sample_size : ℕ)
    (students_first_grade : ℕ)
    (students_second_grade : ℕ)
    (sample_first_and_second : ℕ)
    (students_in_third_grade : ℕ)
    (h1 : total_students = 2000)
    (h2 : sample_size = 100)
    (h3 : sample_first_and_second = students_first_grade + students_second_grade)
    (h4 : students_first_grade = 30)
    (h5 : students_second_grade = 30)
    (h6 : sample_first_and_second = 60)
    (h7 : sample_size - sample_first_and_second = students_in_third_grade)
    (h8 : students_in_third_grade * total_students = 40 * total_students / 100) :
  students_in_third_grade = 800 :=
sorry

end number_of_students_in_third_grade_l301_30169


namespace xyz_distinct_real_squares_l301_30198

theorem xyz_distinct_real_squares (x y z : ℝ) 
  (h1 : x^2 = 2 + y)
  (h2 : y^2 = 2 + z)
  (h3 : z^2 = 2 + x) 
  (h4 : x ≠ y ∧ y ≠ z ∧ z ≠ x) :
  x^2 + y^2 + z^2 = 5 ∨ x^2 + y^2 + z^2 = 6 ∨ x^2 + y^2 + z^2 = 9 :=
by 
  sorry

end xyz_distinct_real_squares_l301_30198


namespace none_of_these_l301_30121

variables (a b c d e f : Prop)

-- Given conditions
axiom condition1 : a > b → c > d
axiom condition2 : c < d → e > f

-- Invalid conclusions
theorem none_of_these :
  ¬(a < b → e > f) ∧
  ¬(e > f → a < b) ∧
  ¬(e < f → a > b) ∧
  ¬(a > b → e < f) := sorry

end none_of_these_l301_30121


namespace no_such_number_exists_l301_30119

theorem no_such_number_exists : ¬ ∃ n : ℕ, 10^(n+1) + 35 ≡ 0 [MOD 63] :=
by {
  sorry 
}

end no_such_number_exists_l301_30119


namespace number_of_diagonals_in_octagon_l301_30142

theorem number_of_diagonals_in_octagon :
  let n : ℕ := 8
  let num_diagonals := n * (n - 3) / 2
  num_diagonals = 20 := by
  sorry

end number_of_diagonals_in_octagon_l301_30142


namespace polygon_not_hexagon_if_quadrilateral_after_cut_off_l301_30182

-- Definition of polygonal shape and quadrilateral condition
def is_quadrilateral (sides : Nat) : Prop := sides = 4

-- Definition of polygonal shape with general condition of cutting off one angle
def after_cut_off (original_sides : Nat) (remaining_sides : Nat) : Prop :=
  original_sides > remaining_sides ∧ remaining_sides + 1 = original_sides

-- Problem statement: If a polygon's one angle cut-off results in a quadrilateral, then it is not a hexagon
theorem polygon_not_hexagon_if_quadrilateral_after_cut_off
  (original_sides : Nat) (remaining_sides : Nat) :
  after_cut_off original_sides remaining_sides → is_quadrilateral remaining_sides → original_sides ≠ 6 :=
by
  sorry

end polygon_not_hexagon_if_quadrilateral_after_cut_off_l301_30182


namespace shareCoins_l301_30133

theorem shareCoins (a b c d e d : ℝ)
  (h1 : b = a - d)
  (h2 : ((a-2*d) + b = a + (a+d) + (a+2*d)))
  (h3 : (a-2*d) + b + a + (a+d) + (a+2*d) = 5) :
  b = 7 / 6 :=
by
  sorry

end shareCoins_l301_30133


namespace truck_and_trailer_total_weight_l301_30183

def truck_weight : ℝ := 4800
def trailer_weight (truck_weight : ℝ) : ℝ := 0.5 * truck_weight - 200
def total_weight (truck_weight trailer_weight : ℝ) : ℝ := truck_weight + trailer_weight 

theorem truck_and_trailer_total_weight : 
  total_weight truck_weight (trailer_weight truck_weight) = 7000 :=
by 
  sorry

end truck_and_trailer_total_weight_l301_30183


namespace system1_solution_system2_solution_l301_30141

theorem system1_solution (x y : ℤ) 
  (h1 : x = 2 * y - 1) 
  (h2 : 3 * x + 4 * y = 17) : 
  x = 3 ∧ y = 2 :=
by 
  sorry

theorem system2_solution (x y : ℤ) 
  (h1 : 2 * x - y = 0) 
  (h2 : 3 * x - 2 * y = 5) : 
  x = -5 ∧ y = -10 := 
by 
  sorry

end system1_solution_system2_solution_l301_30141


namespace liz_three_pointers_l301_30153

-- Define the points scored by Liz's team in the final quarter.
def points_scored_by_liz (free_throws jump_shots three_pointers : ℕ) : ℕ :=
  free_throws * 1 + jump_shots * 2 + three_pointers * 3

-- Define the points needed to tie the game.
def points_needed_to_tie (initial_deficit points_lost other_team_points : ℕ) : ℕ :=
  points_lost + (initial_deficit - points_lost) + other_team_points

-- The total points scored by Liz from free throws and jump shots.
def liz_regular_points (free_throws jump_shots : ℕ) : ℕ :=
  free_throws * 1 + jump_shots * 2

theorem liz_three_pointers :
  ∀ (free_throws jump_shots liz_team_deficit_final quarter_deficit other_team_points liz_team_deficit_end final_deficit : ℕ),
    liz_team_deficit_final = 20 →
    free_throws = 5 →
    jump_shots = 4 →
    other_team_points = 10 →
    liz_team_deficit_end = 8 →
    final_deficit = liz_team_deficit_final - liz_team_deficit_end →
    (free_throws * 1 + jump_shots * 2 + 3 * final_deficit) = 
      points_needed_to_tie 20 other_team_points 8 →
    (3 * final_deficit) = 9 →
    final_deficit = 3 →
    final_deficit = 3 :=
by
  intros 
  try sorry

end liz_three_pointers_l301_30153


namespace six_digit_numbers_with_zero_l301_30181

theorem six_digit_numbers_with_zero :
  let total_digits := (9 * 10 * 10 * 10 * 10 * 10 : ℕ)
  let non_zero_digits := (9 * 9 * 9 * 9 * 9 * 9 : ℕ)
  total_digits - non_zero_digits = 368559 :=
by
  sorry

end six_digit_numbers_with_zero_l301_30181


namespace suzanna_textbooks_page_total_l301_30151

theorem suzanna_textbooks_page_total :
  let H := 160
  let G := H + 70
  let M := (H + G) / 2
  let S := 2 * H
  let L := (H + G) - 30
  let E := M + L + 25
  H + G + M + S + L + E = 1845 := by
  sorry

end suzanna_textbooks_page_total_l301_30151


namespace remainder_of_86_l301_30165

theorem remainder_of_86 {m : ℕ} (h1 : m ≠ 1) 
  (h2 : 69 % m = 90 % m) (h3 : 90 % m = 125 % m) : 86 % m = 2 := 
by
  sorry

end remainder_of_86_l301_30165


namespace larger_number_l301_30125

theorem larger_number (L S : ℕ) (h1 : L - S = 1345) (h2 : L = 6 * S + 15) : L = 1611 :=
by
  sorry

end larger_number_l301_30125


namespace find_g_3_8_l301_30138

variable (g : ℝ → ℝ)
variable (x : ℝ)

-- Conditions
axiom g0 : g 0 = 0
axiom monotonicity (x y : ℝ) : 0 ≤ x → x < y → y ≤ 1 → g x ≤ g y
axiom symmetry (x : ℝ) : 0 ≤ x → x ≤ 1 → g (1 - x) = 1 - g x
axiom scaling (x : ℝ) : 0 ≤ x → x ≤ 1 → g (x / 4) = g x / 3

-- Statement to prove
theorem find_g_3_8 : g (3 / 8) = 2 / 9 := 
sorry

end find_g_3_8_l301_30138


namespace max_chord_length_l301_30100

theorem max_chord_length (x1 y1 x2 y2 : ℝ) (h_parabola1 : x1^2 = 8 * y1) (h_parabola2 : x2^2 = 8 * y2)
  (h_midpoint_ordinate : (y1 + y2) / 2 = 4) :
  abs ((y1 + y2) + 4) = 12 :=
by
  sorry

end max_chord_length_l301_30100


namespace expression_divisible_by_11_l301_30148

theorem expression_divisible_by_11 (n : ℕ) : (6^(2*n) + 3^(n+2) + 3^n) % 11 = 0 := 
sorry

end expression_divisible_by_11_l301_30148


namespace sequence_formula_l301_30122

theorem sequence_formula (a : ℕ → ℕ) (h₁ : a 1 = 1) (h₂ : ∀ n ≥ 2, a n = 2 * a (n - 1) + 1) :
  ∀ n : ℕ, a n = 2 ^ n - 1 :=
sorry

end sequence_formula_l301_30122


namespace find_m_l301_30104

theorem find_m (x y m : ℤ) (h1 : x = 1) (h2 : y = -1) (h3 : 2 * x + m + y = 0) : m = -1 := by
  -- Proof can be completed here
  sorry

end find_m_l301_30104


namespace milk_production_days_l301_30163

theorem milk_production_days (x : ℕ) (h : x > 0) :
  let daily_production_per_cow := (x + 1) / (x * (x + 2))
  let total_daily_production := (x + 4) * daily_production_per_cow
  ((x + 7) / total_daily_production) = (x * (x + 2) * (x + 7)) / ((x + 1) * (x + 4)) := 
by
  sorry

end milk_production_days_l301_30163


namespace even_number_less_than_its_square_l301_30155

theorem even_number_less_than_its_square (m : ℕ) (h1 : 2 ∣ m) (h2 : m > 1) : m < m^2 :=
by
sorry

end even_number_less_than_its_square_l301_30155


namespace square_side_length_theorem_l301_30160

-- Define the properties of the geometric configurations
def is_tangent_to_extension_segments (circle_radius : ℝ) (segment_length : ℝ) : Prop :=
  segment_length = circle_radius

def angle_between_tangents_from_point (angle : ℝ) : Prop :=
  angle = 60 

def square_side_length (side : ℝ) : Prop :=
  side = 4 * (Real.sqrt 2 - 1)

-- Main theorem
theorem square_side_length_theorem (circle_radius : ℝ) (segment_length : ℝ) (angle : ℝ) (side : ℝ)
  (h1 : is_tangent_to_extension_segments circle_radius segment_length)
  (h2 : angle_between_tangents_from_point angle) :
  square_side_length side :=
by
  sorry

end square_side_length_theorem_l301_30160


namespace selling_price_of_article_l301_30157

theorem selling_price_of_article (cost_price gain_percent : ℝ) (h1 : cost_price = 100) (h2 : gain_percent = 30) : 
  cost_price + (gain_percent / 100) * cost_price = 130 := 
by 
  sorry

end selling_price_of_article_l301_30157


namespace red_users_count_l301_30126

noncomputable def total_students : ℕ := 70
noncomputable def green_users : ℕ := 52
noncomputable def both_colors_users : ℕ := 38

theorem red_users_count : 
  ∀ (R : ℕ), total_students = green_users + R - both_colors_users → R = 56 :=
by
  sorry

end red_users_count_l301_30126


namespace voting_problem_l301_30102

theorem voting_problem (x y x' y' : ℕ) (m : ℕ) (h1 : x + y = 500) (h2 : y > x)
    (h3 : y - x = m) (h4 : x' = (10 * y) / 9) (h5 : x' + y' = 500)
    (h6 : x' - y' = 3 * m) :
    x' - x = 59 := 
sorry

end voting_problem_l301_30102


namespace sum_of_fractions_l301_30154

theorem sum_of_fractions : (3 / 20 : ℝ) + (5 / 50 : ℝ) + (7 / 2000 : ℝ) = 0.2535 :=
by sorry

end sum_of_fractions_l301_30154


namespace caleb_ice_cream_l301_30156

theorem caleb_ice_cream (x : ℕ) (hx1 : ∃ x, x ≥ 0) (hx2 : 4 * x - 36 = 4) : x = 10 :=
by {
  sorry
}

end caleb_ice_cream_l301_30156


namespace range_of_2alpha_minus_beta_l301_30107

def condition_range_alpha_beta (α β : ℝ) : Prop := 
  - (Real.pi / 2) < α ∧ α < β ∧ β < (Real.pi / 2)

theorem range_of_2alpha_minus_beta (α β : ℝ) (h : condition_range_alpha_beta α β) : 
  - Real.pi < 2 * α - β ∧ 2 * α - β < Real.pi / 2 :=
sorry

end range_of_2alpha_minus_beta_l301_30107


namespace cubes_with_two_or_three_blue_faces_l301_30192

theorem cubes_with_two_or_three_blue_faces 
  (four_inch_cube : ℝ)
  (painted_blue_faces : ℝ)
  (one_inch_cubes : ℝ) :
  (four_inch_cube = 4) →
  (painted_blue_faces = 6) →
  (one_inch_cubes = 64) →
  (num_cubes_with_two_or_three_blue_faces = 32) :=
sorry

end cubes_with_two_or_three_blue_faces_l301_30192


namespace simplify_expression_l301_30146

variable (b : ℝ)

theorem simplify_expression (b : ℝ) : 
  (3 * b + 7 - 5 * b) / 3 = (-2 / 3) * b + (7 / 3) :=
by
  sorry

end simplify_expression_l301_30146


namespace calculate_principal_amount_l301_30139

theorem calculate_principal_amount (P : ℝ) (h1 : P * 0.1025 - P * 0.1 = 25) : 
  P = 10000 :=
by
  sorry

end calculate_principal_amount_l301_30139


namespace Mary_bought_stickers_initially_l301_30115

variable (S A M : ℕ) -- Define S, A, and M as natural numbers

-- Given conditions in the problem
def condition1 : Prop := S = A
def condition2 : Prop := M = 3 * A
def condition3 : Prop := A + (2 / 3) * M = 900

-- The theorem we need to prove
theorem Mary_bought_stickers_initially
  (h1 : condition1 S A)
  (h2 : condition2 A M)
  (h3 : condition3 A M)
  : S + A + M = 1500 :=
sorry -- Proof

end Mary_bought_stickers_initially_l301_30115


namespace game_rounds_l301_30123

noncomputable def play_game (A B C D : ℕ) : ℕ := sorry

theorem game_rounds : play_game 16 15 14 13 = 49 :=
by
  sorry

end game_rounds_l301_30123


namespace total_buttons_l301_30143

-- Define the conditions
def shirts_per_kid : Nat := 3
def number_of_kids : Nat := 3
def buttons_per_shirt : Nat := 7

-- Define the statement to prove
theorem total_buttons : shirts_per_kid * number_of_kids * buttons_per_shirt = 63 := by
  sorry

end total_buttons_l301_30143


namespace megan_broke_3_eggs_l301_30162

variables (total_eggs B C P : ℕ)

theorem megan_broke_3_eggs (h1 : total_eggs = 24) (h2 : C = 2 * B) (h3 : P = 24 - (B + C)) (h4 : P - C = 9) : B = 3 := by
  sorry

end megan_broke_3_eggs_l301_30162


namespace theta_solutions_count_l301_30173

theorem theta_solutions_count :
  (∃ (count : ℕ), count = 4 ∧ ∀ θ, 0 < θ ∧ θ ≤ 2 * Real.pi ∧ 1 - 4 * Real.sin θ + 5 * Real.cos (2 * θ) = 0 ↔ count = 4) :=
sorry

end theta_solutions_count_l301_30173


namespace boat_travel_distance_l301_30110

theorem boat_travel_distance
  (D : ℝ) -- Distance traveled in both directions
  (t : ℝ) -- Time in hours it takes to travel upstream
  (speed_boat : ℝ) -- Speed of the boat in still water
  (speed_stream : ℝ) -- Speed of the stream
  (time_diff : ℝ) -- Difference in time between downstream and upstream travel
  (h1 : speed_boat = 10)
  (h2 : speed_stream = 2)
  (h3 : time_diff = 1.5)
  (h4 : D = 8 * t)
  (h5 : D = 12 * (t - time_diff)) :
  D = 36 := by
  sorry

end boat_travel_distance_l301_30110


namespace distinct_diagonals_nonagon_l301_30158

def n : ℕ := 9

def diagonals_nonagon (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

theorem distinct_diagonals_nonagon : diagonals_nonagon n = 27 :=
by
  unfold diagonals_nonagon
  norm_num
  sorry

end distinct_diagonals_nonagon_l301_30158


namespace smallest_n_for_divisibility_l301_30171

theorem smallest_n_for_divisibility (n : ℕ) (h : 2 ∣ 3^(2*n) - 1) (k : ℕ) : n = 2^(2007) := by
  sorry

end smallest_n_for_divisibility_l301_30171


namespace workshopA_more_stable_than_B_l301_30199

-- Given data sets for workshops A and B
def workshopA_data := [102, 101, 99, 98, 103, 98, 99]
def workshopB_data := [110, 115, 90, 85, 75, 115, 110]

-- Define stability of a product in terms of the standard deviation or similar metric
def is_more_stable (dataA dataB : List ℕ) : Prop :=
  sorry -- Replace with a definition comparing stability based on a chosen metric, e.g., standard deviation

-- Prove that Workshop A's product is more stable than Workshop B's product
theorem workshopA_more_stable_than_B : is_more_stable workshopA_data workshopB_data :=
  sorry

end workshopA_more_stable_than_B_l301_30199


namespace find_n_that_satisfies_l301_30191

theorem find_n_that_satisfies :
  ∃ (n : ℕ), (1 / (n + 2 : ℕ) + 2 / (n + 2) + (n + 1) / (n + 2) = 2) ∧ (n = 0) :=
by 
  existsi (0 : ℕ)
  sorry

end find_n_that_satisfies_l301_30191


namespace sum_of_series_l301_30132

theorem sum_of_series (h1 : 2 + 4 + 6 + 8 + 10 = 30) (h2 : 1 + 3 + 5 + 7 + 9 = 25) : 
  ((2 + 4 + 6 + 8 + 10) / (1 + 3 + 5 + 7 + 9)) + ((1 + 3 + 5 + 7 + 9) / (2 + 4 + 6 + 8 + 10)) = 61 / 30 := by
  sorry

end sum_of_series_l301_30132


namespace total_spent_on_birthday_presents_l301_30150

noncomputable def leonards_total_before_discount :=
  (3 * 35.50) + (2 * 120.75) + 44.25

noncomputable def leonards_total_after_discount :=
  leonards_total_before_discount - (0.10 * leonards_total_before_discount)

noncomputable def michaels_total_before_discount :=
  89.50 + (3 * 54.50) + 24.75

noncomputable def michaels_total_after_discount :=
  michaels_total_before_discount - (0.15 * michaels_total_before_discount)

noncomputable def emilys_total_before_tax :=
  (2 * 69.25) + (4 * 14.80)

noncomputable def emilys_total_after_tax :=
  emilys_total_before_tax + (0.08 * emilys_total_before_tax)

noncomputable def total_amount_spent :=
  leonards_total_after_discount + michaels_total_after_discount + emilys_total_after_tax

theorem total_spent_on_birthday_presents :
  total_amount_spent = 802.64 :=
by
  sorry

end total_spent_on_birthday_presents_l301_30150


namespace brandon_businesses_l301_30188

theorem brandon_businesses (total_businesses: ℕ) (fire_fraction: ℚ) (quit_fraction: ℚ) 
  (h_total: total_businesses = 72) 
  (h_fire_fraction: fire_fraction = 1/2) 
  (h_quit_fraction: quit_fraction = 1/3) : 
  total_businesses - (total_businesses * fire_fraction + total_businesses * quit_fraction) = 12 :=
by 
  sorry

end brandon_businesses_l301_30188


namespace student_opinion_change_l301_30196

theorem student_opinion_change (init_enjoy : ℕ) (init_not_enjoy : ℕ)
                               (final_enjoy : ℕ) (final_not_enjoy : ℕ) :
  init_enjoy = 40 ∧ init_not_enjoy = 60 ∧ final_enjoy = 75 ∧ final_not_enjoy = 25 →
  ∃ y_min y_max : ℕ, 
    y_min = 35 ∧ y_max = 75 ∧ (y_max - y_min = 40) :=
by
  sorry

end student_opinion_change_l301_30196


namespace not_true_diamond_self_zero_l301_30105

-- Define the operator ⋄
def diamond (x y : ℝ) := |x - 2*y|

-- The problem statement in Lean4
theorem not_true_diamond_self_zero : ¬ (∀ x : ℝ, diamond x x = 0) := by
  sorry

end not_true_diamond_self_zero_l301_30105


namespace max_value_of_m_l301_30109

variable (m : ℝ)

noncomputable def satisfies_inequality (m : ℝ) : Prop :=
∀ x > 0, m * x * Real.log x - (x + m) * Real.exp ((x - m) / m) ≤ 0

theorem max_value_of_m (h1 : 0 < m) (h2 : satisfies_inequality m) : m ≤ Real.exp 2 := sorry

end max_value_of_m_l301_30109


namespace sum_product_distinct_zero_l301_30134

open BigOperators

theorem sum_product_distinct_zero {n : ℕ} (h : n ≥ 3) (a : Fin n → ℝ) (ha : Function.Injective a) :
  (∑ i, (a i) * ∏ j in Finset.univ \ {i}, (1 / (a i - a j))) = 0 := 
by
  sorry

end sum_product_distinct_zero_l301_30134


namespace determine_d_l301_30185

theorem determine_d (f g : ℝ → ℝ) (c d : ℝ) (h1 : ∀ x, f x = 5 * x + c) (h2 : ∀ x, g x = c * x + 3) (h3 : ∀ x, f (g x) = 15 * x + d) : d = 18 := 
  sorry

end determine_d_l301_30185


namespace scientific_notation_256000_l301_30172

theorem scientific_notation_256000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ 256000 = a * 10^n ∧ a = 2.56 ∧ n = 5 :=
by
  sorry

end scientific_notation_256000_l301_30172


namespace square_side_length_l301_30178

theorem square_side_length (s : ℝ) (h : s^2 = 12 * s) : s = 12 :=
by
  sorry

end square_side_length_l301_30178


namespace sum_of_values_of_N_l301_30127

-- Given conditions
variables (N R : ℝ)
-- Condition that needs to be checked
def condition (N R : ℝ) : Prop := N + 3 / N = R ∧ N ≠ 0

-- The statement to prove
theorem sum_of_values_of_N (N R : ℝ) (h: condition N R) : N + (3 / N) = R :=
sorry

end sum_of_values_of_N_l301_30127


namespace number_condition_l301_30167

theorem number_condition (x : ℤ) (h : x - 7 = 9) : 5 * x = 80 := by
  sorry

end number_condition_l301_30167


namespace max_value_is_one_l301_30170

noncomputable def max_value_fraction (x : ℝ) : ℝ :=
  (1 + Real.cos x) / (Real.sin x + Real.cos x + 2)

theorem max_value_is_one : ∃ x : ℝ, max_value_fraction x = 1 := by
  sorry

end max_value_is_one_l301_30170


namespace last_ball_probability_l301_30186

variables (p q : ℕ)

def probability_white_last_ball (p : ℕ) : ℝ :=
  if p % 2 = 0 then 0 else 1

theorem last_ball_probability :
  ∀ {p q : ℕ},
    probability_white_last_ball p = if p % 2 = 0 then 0 else 1 :=
by
  intros
  sorry

end last_ball_probability_l301_30186


namespace speed_ratio_l301_30176

def distance_to_work := 28
def speed_back := 14
def total_time := 6

theorem speed_ratio 
  (d : ℕ := distance_to_work) 
  (v_2 : ℕ := speed_back) 
  (t : ℕ := total_time) : 
  ∃ v_1 : ℕ, (d / v_1 + d / v_2 = t) ∧ (v_2 / v_1 = 2) :=
by 
  sorry

end speed_ratio_l301_30176


namespace num_valid_values_n_l301_30137

theorem num_valid_values_n :
  ∃ n : ℕ, (∃ a b c : ℕ,
    8 * a + 88 * b + 888 * c = 8880 ∧
    n = a + 2 * b + 3  * c) ∧
  (∃! k : ℕ, k = 119) :=
by sorry

end num_valid_values_n_l301_30137


namespace square_area_parabola_inscribed_l301_30193

theorem square_area_parabola_inscribed (s : ℝ) (x y : ℝ) :
  (y = x^2 - 6 * x + 8) ∧
  (s = -2 + 2 * Real.sqrt 5) ∧
  (x = 3 - s / 2 ∨ x = 3 + s / 2) →
  s ^ 2 = 24 - 8 * Real.sqrt 5 :=
by
  sorry

end square_area_parabola_inscribed_l301_30193


namespace part1_part2_l301_30112

variable {A B C a b c : ℝ}

-- Part (1): Prove that 2a^2 = b^2 + c^2 given the condition
theorem part1 (h : Real.sin C * Real.sin (A - B) = Real.sin B * Real.sin (C - A)) : 2 * a^2 = b^2 + c^2 := 
sorry

-- Part (2): Prove the perimeter of triangle ABC
theorem part2 (a : ℝ) (h_a : a = 5) (cosA : ℝ) (h_cosA : cosA = 25 / 31) : 5 + b + c = 14 := 
sorry

end part1_part2_l301_30112


namespace enchanted_creatures_gala_handshakes_l301_30149

theorem enchanted_creatures_gala_handshakes :
  let goblins := 30
  let trolls := 20
  let goblin_handshakes := goblins * (goblins - 1) / 2
  let troll_to_goblin_handshakes := trolls * goblins
  goblin_handshakes + troll_to_goblin_handshakes = 1035 := 
by
  sorry

end enchanted_creatures_gala_handshakes_l301_30149


namespace inequality_solution_l301_30135

theorem inequality_solution (x y : ℝ) : y - x < abs x ↔ y < 0 ∨ y < 2 * x :=
by sorry

end inequality_solution_l301_30135


namespace goats_count_l301_30144

variable (h d c t g : Nat)
variable (l : Nat)

theorem goats_count 
  (h_eq : h = 2)
  (d_eq : d = 5)
  (c_eq : c = 7)
  (t_eq : t = 3)
  (l_eq : l = 72)
  (legs_eq : 4 * h + 4 * d + 4 * c + 4 * t + 4 * g = l) : 
  g = 1 := by
  sorry

end goats_count_l301_30144


namespace probability_of_karnataka_student_l301_30189

-- Defining the conditions

-- Number of students from each region
def total_students : ℕ := 10
def maharashtra_students : ℕ := 4
def karnataka_students : ℕ := 3
def goa_students : ℕ := 3

-- Number of students to be selected
def students_to_select : ℕ := 4

-- Total ways to choose 4 students out of 10
def C_total : ℕ := Nat.choose total_students students_to_select

-- Ways to select 4 students from the 7 students not from Karnataka
def non_karnataka_students : ℕ := maharashtra_students + goa_students
def C_non_karnataka : ℕ := Nat.choose non_karnataka_students students_to_select

-- Probability calculations
def P_no_karnataka : ℚ := C_non_karnataka / C_total
def P_at_least_one_karnataka : ℚ := 1 - P_no_karnataka

-- The statement to be proved
theorem probability_of_karnataka_student :
  P_at_least_one_karnataka = 5 / 6 :=
sorry

end probability_of_karnataka_student_l301_30189


namespace radius_of_circle_proof_l301_30174

noncomputable def radius_of_circle (x y : ℝ) (h1 : x = Real.pi * r ^ 2) (h2 : y = 2 * Real.pi * r) (h3 : x + y = 100 * Real.pi) : ℝ :=
  r

theorem radius_of_circle_proof (r x y : ℝ) (h1 : x = Real.pi * r ^ 2) (h2 : y = 2 * Real.pi * r) (h3 : x + y = 100 * Real.pi) : r = 10 :=
by
  sorry

end radius_of_circle_proof_l301_30174


namespace find_a_value_l301_30130

def line1 (a : ℝ) (x y : ℝ) : ℝ := a * x + (a + 2) * y + 1
def line2 (a : ℝ) (x y : ℝ) : ℝ := a * x - y + 2

-- Define what it means for two lines to be not parallel
def not_parallel (a : ℝ) : Prop :=
  ∀ x y : ℝ, (line1 a x y ≠ 0 ∧ line2 a x y ≠ 0)

theorem find_a_value (a : ℝ) (h : not_parallel a) : a = 0 ∨ a = -3 :=
  sorry

end find_a_value_l301_30130


namespace find_f_log2_3_l301_30179

noncomputable def f : ℝ → ℝ := sorry

axiom f_mono : ∀ x y : ℝ, x ≤ y → f x ≤ f y
axiom f_condition : ∀ x : ℝ, f (f x + 2 / (2^x + 1)) = (1 / 3)

theorem find_f_log2_3 : f (Real.log 3 / Real.log 2) = (1 / 2) :=
by
  sorry

end find_f_log2_3_l301_30179


namespace find_p_l301_30161

theorem find_p (p q : ℚ) (h1 : 5 * p + 6 * q = 10) (h2 : 6 * p + 5 * q = 17) : p = 52 / 11 :=
by {
  -- Proof steps would go here
  sorry
}

end find_p_l301_30161


namespace gcf_90_108_l301_30111

-- Given two integers 90 and 108
def a : ℕ := 90
def b : ℕ := 108

-- Question: What is the greatest common factor (GCF) of 90 and 108?
theorem gcf_90_108 : Nat.gcd a b = 18 :=
by {
  sorry
}

end gcf_90_108_l301_30111


namespace time_to_fill_pool_l301_30114

-- Define constants based on the conditions
def pool_capacity : ℕ := 30000
def hose_count : ℕ := 5
def flow_rate_per_hose : ℕ := 25 / 10  -- 2.5 gallons per minute
def conversion_minutes_to_hours : ℕ := 60

-- Define the total flow rate per minute
def total_flow_rate_per_minute : ℕ := hose_count * flow_rate_per_hose

-- Define the total flow rate per hour
def total_flow_rate_per_hour : ℕ := total_flow_rate_per_minute * conversion_minutes_to_hours

-- Theorem stating the number of hours required to fill the pool
theorem time_to_fill_pool : pool_capacity / total_flow_rate_per_hour = 40 := by
  sorry -- Proof will be provided here

end time_to_fill_pool_l301_30114


namespace distance_from_axis_gt_l301_30116

theorem distance_from_axis_gt 
  (a b x1 x2 y1 y2 : ℝ) (h₁ : a > 0) 
  (h₂ : y1 = a * x1^2 - 2 * a * x1 + b) 
  (h₃ : y2 = a * x2^2 - 2 * a * x2 + b) 
  (h₄ : y1 > y2) : 
  |x1 - 1| > |x2 - 1| := 
sorry

end distance_from_axis_gt_l301_30116


namespace tom_tickets_left_l301_30108

-- Define the conditions
def tickets_whack_a_mole : ℕ := 32
def tickets_skee_ball : ℕ := 25
def tickets_spent_on_hat : ℕ := 7

-- Define what we need to prove
theorem tom_tickets_left : tickets_whack_a_mole + tickets_skee_ball - tickets_spent_on_hat = 50 :=
by sorry

end tom_tickets_left_l301_30108


namespace beast_of_war_running_time_correct_l301_30159

def running_time_millennium : ℕ := 120

def running_time_alpha_epsilon (rt_millennium : ℕ) : ℕ := rt_millennium - 30

def running_time_beast_of_war (rt_alpha_epsilon : ℕ) : ℕ := rt_alpha_epsilon + 10

theorem beast_of_war_running_time_correct :
  running_time_beast_of_war (running_time_alpha_epsilon running_time_millennium) = 100 := by sorry

end beast_of_war_running_time_correct_l301_30159


namespace number_of_perfect_cubes_l301_30118

theorem number_of_perfect_cubes (n : ℤ) : 
  (∃ (count : ℤ), (∀ (x : ℤ), (100 < x^3 ∧ x^3 < 400) ↔ x = 5 ∨ x = 6 ∨ x = 7) ∧ (count = 3)) := 
sorry

end number_of_perfect_cubes_l301_30118


namespace exists_rational_non_integer_xy_no_rational_non_integer_xy_l301_30131

-- Part (a)
theorem exists_rational_non_integer_xy 
  (x y : ℚ) (h1 : ¬ ∃ z : ℤ, x = z ∧ y = z) : 
  (∃ x y : ℚ, ¬(∃ z : ℤ, x = z ∨ y = z) ∧ 
   ∃ z1 z2 : ℤ, 19 * x + 8 * y = ↑z1 ∧ 8 * x + 3 * y = ↑z2) :=
sorry

-- Part (b)
theorem no_rational_non_integer_xy 
  (x y : ℚ) (h1 : ¬ ∃ z : ℤ, x = z ∧ y = z) : 
  ¬ ∃ x y : ℚ, ¬(∃ z : ℤ, x = z ∨ y = z) ∧ 
  ∃ z1 z2 : ℤ, 19 * x^2 + 8 * y^2 = ↑z1 ∧ 8 * x^2 + 3 * y^2 = ↑z2 :=
sorry

end exists_rational_non_integer_xy_no_rational_non_integer_xy_l301_30131


namespace solve_system_of_equations_l301_30175

theorem solve_system_of_equations (x1 x2 x3 x4 x5 y : ℝ) :
  x5 + x2 = y * x1 ∧
  x1 + x3 = y * x2 ∧
  x2 + x4 = y * x3 ∧
  x3 + x5 = y * x4 ∧
  x4 + x1 = y * x5 →
  (y = 2 ∧ x1 = x2 ∧ x2 = x3 ∧ x3 = x4 ∧ x4 = x5) ∨
  (y ≠ 2 ∧ (y^2 + y - 1 ≠ 0 ∧ x1 = 0 ∧ x2 = 0 ∧ x3 = 0 ∧ x4 = 0 ∧ x5 = 0) ∨
  (y^2 + y - 1 = 0 ∧ y = (1 / 2) * (-1 + Real.sqrt 5) ∨ y = (1 / 2) * (-1 - Real.sqrt 5) ∧
    ∃ a b : ℝ, x1 = a ∧ x2 = b ∧ x3 = y * b - a ∧ x4 = - y * (a + b) ∧ x5 = y * a - b))
:=
sorry

end solve_system_of_equations_l301_30175


namespace polynomial_evaluation_l301_30195

theorem polynomial_evaluation :
  ∀ x : ℤ, x = -2 → (x^3 + x^2 + x + 1 = -5) :=
by
  intros x hx
  rw [hx]
  norm_num

end polynomial_evaluation_l301_30195
