import Mathlib

namespace cos_double_angle_l642_64262
open Real

theorem cos_double_angle (α : ℝ) (h : tan (α - π / 4) = 2) : cos (2 * α) = -4 / 5 := 
sorry

end cos_double_angle_l642_64262


namespace jar_total_value_l642_64288

def total_value_in_jar (p n q : ℕ) (total_coins : ℕ) (value : ℝ) : Prop :=
  p + n + q = total_coins ∧
  n = 3 * p ∧
  q = 4 * n ∧
  value = p * 0.01 + n * 0.05 + q * 0.25

theorem jar_total_value (p : ℕ) (h₁ : 16 * p = 240) : 
  ∃ value, total_value_in_jar p (3 * p) (12 * p) 240 value ∧ value = 47.4 :=
by
  sorry

end jar_total_value_l642_64288


namespace inequality_must_hold_l642_64256

theorem inequality_must_hold (a b c : ℝ) (h : a > b) : (a - b) * c^2 ≥ 0 := 
sorry

end inequality_must_hold_l642_64256


namespace lateral_surface_area_of_pyramid_l642_64282

theorem lateral_surface_area_of_pyramid
  (sin_alpha : ℝ)
  (A_section : ℝ)
  (h1 : sin_alpha = 15 / 17)
  (h2 : A_section = 3 * Real.sqrt 34) :
  ∃ A_lateral : ℝ, A_lateral = 68 :=
sorry

end lateral_surface_area_of_pyramid_l642_64282


namespace simplify_fraction_l642_64228

theorem simplify_fraction (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) : 
  1 / x - 1 / (x - 1) = -1 / (x * (x - 1)) :=
by
  sorry

end simplify_fraction_l642_64228


namespace eggs_per_box_l642_64210

-- Conditions
def num_eggs : ℝ := 3.0
def num_boxes : ℝ := 2.0

-- Theorem statement
theorem eggs_per_box (h1 : num_eggs = 3.0) (h2 : num_boxes = 2.0) : (num_eggs / num_boxes = 1.5) :=
sorry

end eggs_per_box_l642_64210


namespace find_remainder_l642_64229

theorem find_remainder (x : ℤ) (h : 0 < x ∧ 7 * x % 26 = 1) : (13 + 3 * x) % 26 = 6 :=
sorry

end find_remainder_l642_64229


namespace real_roots_for_all_a_b_l642_64209

theorem real_roots_for_all_a_b (a b : ℝ) : ∃ x : ℝ, (x^2 / (x^2 - a^2) + x^2 / (x^2 - b^2) = 4) :=
sorry

end real_roots_for_all_a_b_l642_64209


namespace pizza_slices_left_l642_64216

theorem pizza_slices_left (total_slices : ℕ) (angeli_slices : ℚ) (marlon_slices : ℚ) 
  (H1 : total_slices = 8) (H2 : angeli_slices = 3/2) (H3 : marlon_slices = 3/2) :
  total_slices - (angeli_slices + marlon_slices) = 5 :=
by
  sorry

end pizza_slices_left_l642_64216


namespace rational_expression_iff_rational_square_l642_64237

theorem rational_expression_iff_rational_square (x : ℝ) :
  (∃ r : ℚ, x^2 + (Real.sqrt (x^4 + 1)) - 1 / (x^2 + (Real.sqrt (x^4 + 1))) = r) ↔
  (∃ q : ℚ, x^2 = q) := by
  sorry

end rational_expression_iff_rational_square_l642_64237


namespace range_of_a_l642_64241

noncomputable def f (x a : ℝ) := Real.log x + a / x

theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : ∀ x > 0, x * (2 * Real.log a - Real.log x) ≤ a) : 
  0 < a ∧ a ≤ 1 / Real.exp 1 :=
by
  sorry

end range_of_a_l642_64241


namespace breadth_of_rectangular_plot_l642_64280

theorem breadth_of_rectangular_plot :
  ∃ b : ℝ, (∃ l : ℝ, l = 3 * b ∧ l * b = 867) ∧ b = 17 :=
by
  sorry

end breadth_of_rectangular_plot_l642_64280


namespace shaded_percentage_l642_64204

-- Definition for the six-by-six grid and total squares
def total_squares : ℕ := 36
def shaded_squares : ℕ := 16

-- Definition of the problem: to prove the percentage of shaded squares
theorem shaded_percentage : (shaded_squares : ℚ) / total_squares * 100 = 44.4 :=
by
  sorry

end shaded_percentage_l642_64204


namespace athlete_running_minutes_l642_64219

theorem athlete_running_minutes (r w : ℕ) 
  (h1 : r + w = 60)
  (h2 : 10 * r + 4 * w = 450) : 
  r = 35 := 
sorry

end athlete_running_minutes_l642_64219


namespace volume_ratio_l642_64274

noncomputable def V_D (s : ℝ) := (15 + 7 * Real.sqrt 5) * s^3 / 4
noncomputable def a (s : ℝ) := s / 2 * (1 + Real.sqrt 5)
noncomputable def V_I (a : ℝ) := 5 * (3 + Real.sqrt 5) * a^3 / 12

theorem volume_ratio (s : ℝ) (h₁ : 0 < s) :
  V_I (a s) / V_D s = (5 * (3 + Real.sqrt 5) * (1 + Real.sqrt 5)^3) / (12 * 2 * (15 + 7 * Real.sqrt 5)) :=
by
  sorry

end volume_ratio_l642_64274


namespace ellens_initial_legos_l642_64201

-- Define the initial number of Legos as a proof goal
theorem ellens_initial_legos : ∀ (x y : ℕ), (y = x - 17) → (x = 2080) :=
by
  intros x y h
  sorry

end ellens_initial_legos_l642_64201


namespace solution_set_x_squared_minus_3x_lt_0_l642_64235

theorem solution_set_x_squared_minus_3x_lt_0 : { x : ℝ | x^2 - 3 * x < 0 } = { x : ℝ | 0 < x ∧ x < 3 } :=
by {
  sorry
}

end solution_set_x_squared_minus_3x_lt_0_l642_64235


namespace shaded_area_is_correct_l642_64239

-- Defining the conditions
def grid_width : ℝ := 15 -- in units
def grid_height : ℝ := 5 -- in units
def total_grid_area : ℝ := grid_width * grid_height -- in square units

def larger_triangle_base : ℝ := grid_width -- in units
def larger_triangle_height : ℝ := grid_height -- in units
def larger_triangle_area : ℝ := 0.5 * larger_triangle_base * larger_triangle_height -- in square units

def smaller_triangle_base : ℝ := 3 -- in units
def smaller_triangle_height : ℝ := 2 -- in units
def smaller_triangle_area : ℝ := 0.5 * smaller_triangle_base * smaller_triangle_height -- in square units

-- The total area of the triangles that are not shaded
def unshaded_areas : ℝ := larger_triangle_area + smaller_triangle_area

-- The area of the shaded region
def shaded_area : ℝ := total_grid_area - unshaded_areas

-- The statement to be proven
theorem shaded_area_is_correct : shaded_area = 34.5 := 
by 
  -- This is a placeholder for the actual proof, which would normally go here
  sorry

end shaded_area_is_correct_l642_64239


namespace inequality_bound_l642_64259

theorem inequality_bound (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  ( (2 * a + b + c)^2 / (2 * a ^ 2 + (b + c) ^2) + 
    (2 * b + c + a)^2 / (2 * b ^ 2 + (c + a) ^2) + 
    (2 * c + a + b)^2 / (2 * c ^ 2 + (a + b) ^2) ) ≤ 8 := 
sorry

end inequality_bound_l642_64259


namespace bicycle_route_total_length_l642_64267

theorem bicycle_route_total_length :
  let horizontal_length := 13 
  let vertical_length := 13 
  2 * horizontal_length + 2 * vertical_length = 52 :=
by
  let horizontal_length := 13
  let vertical_length := 13
  sorry

end bicycle_route_total_length_l642_64267


namespace tree_height_end_of_2_years_l642_64278

theorem tree_height_end_of_2_years (h4 : ℕ → ℕ)
  (h_tripling : ∀ n, h4 (n + 1) = 3 * h4 n)
  (h4_at_4 : h4 4 = 81) :
  h4 2 = 9 :=
by
  sorry

end tree_height_end_of_2_years_l642_64278


namespace ratio_of_blue_fish_to_total_fish_l642_64218

-- Define the given conditions
def total_fish : ℕ := 30
def blue_spotted_fish : ℕ := 5
def half (n : ℕ) : ℕ := n / 2

-- Calculate the number of blue fish using the conditions
def blue_fish : ℕ := blue_spotted_fish * 2

-- Define the ratio of blue fish to total fish
def ratio (num denom : ℕ) : ℚ := num / denom

-- The theorem to prove
theorem ratio_of_blue_fish_to_total_fish :
  ratio blue_fish total_fish = 1 / 3 := by
  sorry

end ratio_of_blue_fish_to_total_fish_l642_64218


namespace not_linear_eq_l642_64223

-- Representing the given equations
def eq1 (x : ℝ) : Prop := 5 * x + 3 = 3 * x - 7
def eq2 (x : ℝ) : Prop := 1 + 2 * x = 3
def eq4 (x : ℝ) : Prop := x - 7 = 0

-- The equation to verify if it's not linear
def eq3 (x : ℝ) : Prop := abs (2 * x) / 3 + 5 / x = 3

-- Stating the Lean statement to be proved
theorem not_linear_eq : ¬ (eq3 x) := by
  sorry

end not_linear_eq_l642_64223


namespace regular_polygon_of_45_deg_l642_64236

def is_regular_polygon (n : ℕ) : Prop :=
  ∃ k : ℕ, k > 2 ∧ 360 % k = 0 ∧ n = 360 / k

def regular_polygon_is_octagon (angle : ℕ) : Prop :=
  is_regular_polygon 8 ∧ angle = 45

theorem regular_polygon_of_45_deg : regular_polygon_is_octagon 45 :=
  sorry

end regular_polygon_of_45_deg_l642_64236


namespace effect_on_revenue_l642_64207

-- Define the conditions using parameters and variables

variables {P Q : ℝ} -- Original price and quantity of TV sets

def new_price (P : ℝ) : ℝ := P * 1.60 -- New price after 60% increase
def new_quantity (Q : ℝ) : ℝ := Q * 0.80 -- New quantity after 20% decrease

def original_revenue (P Q : ℝ) : ℝ := P * Q -- Original revenue
def new_revenue (P Q : ℝ) : ℝ := (new_price P) * (new_quantity Q) -- New revenue

theorem effect_on_revenue
  (P Q : ℝ) :
  new_revenue P Q = original_revenue P Q * 1.28 :=
by
  sorry

end effect_on_revenue_l642_64207


namespace probability_john_david_chosen_l642_64272

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def choose (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem probability_john_david_chosen :
  let total_workers := 6
  let choose_two := choose total_workers 2
  let favorable_outcomes := 1
  choose_two = 15 → (favorable_outcomes / choose_two : ℝ) = 1 / 15 :=
by
  intros
  sorry

end probability_john_david_chosen_l642_64272


namespace total_seats_in_stadium_l642_64227

theorem total_seats_in_stadium (people_at_game : ℕ) (empty_seats : ℕ) (total_seats : ℕ)
  (h1 : people_at_game = 47) (h2 : empty_seats = 45) :
  total_seats = people_at_game + empty_seats :=
by
  rw [h1, h2]
  show total_seats = 47 + 45
  sorry

end total_seats_in_stadium_l642_64227


namespace problem1_correct_problem2_correct_l642_64214

noncomputable def problem1_solution_set : Set ℝ := {x | x ≤ -3 ∨ x ≥ 1}

noncomputable def problem2_solution_set : Set ℝ := {x | (-3 ≤ x ∧ x < 1) ∨ (3 < x ∧ x ≤ 7)}

theorem problem1_correct (x : ℝ) :
  (4 - x) / (x^2 + x + 1) ≤ 1 ↔ x ∈ problem1_solution_set :=
sorry

theorem problem2_correct (x : ℝ) :
  (1 < |x - 2| ∧ |x - 2| ≤ 5) ↔ x ∈ problem2_solution_set :=
sorry

end problem1_correct_problem2_correct_l642_64214


namespace arithmetic_sequence_max_min_b_l642_64212

-- Define the sequence a_n
def S (n : ℕ) : ℚ := (1/2) * n^2 - 2 * n
def a (n : ℕ) : ℚ := S n - S (n - 1)

-- Question 1: Prove that {a_n} is an arithmetic sequence with a common difference of 1
theorem arithmetic_sequence (n : ℕ) (hn : n ≥ 2) : 
  a n - a (n - 1) = 1 :=
sorry

-- Define the sequence b_n
def b (n : ℕ) : ℚ := (a n + 1) / a n

-- Question 2: Prove that b_3 is the maximum value and b_2 is the minimum value in {b_n}
theorem max_min_b (hn2 : 2 ≥ 1) (hn3 : 3 ≥ 1) : 
  b 3 = 3 ∧ b 2 = -1 :=
sorry

end arithmetic_sequence_max_min_b_l642_64212


namespace product_of_fractions_l642_64296

theorem product_of_fractions : 
  (1 + 1/2) * (1 + 1/3) * (1 + 1/4) * (1 + 1/5) * (1 + 1/6) * (1 + 1/7) = 8 :=
by
  sorry

end product_of_fractions_l642_64296


namespace find_possible_m_values_l642_64268

theorem find_possible_m_values (m : ℕ) (a : ℕ) (h₀ : m > 1) (h₁ : m * a + (m * (m - 1) / 2) = 33) :
  m = 2 ∨ m = 3 ∨ m = 6 :=
by
  sorry

end find_possible_m_values_l642_64268


namespace arithmetic_sequence_a3_l642_64277

variable (a : ℕ → ℕ)
variable (S5 : ℕ)
variable (arithmetic_seq : Prop)

def is_arithmetic_seq (a : ℕ → ℕ) : Prop := ∀ n, a (n + 1) - a n = a 2 - a 1

theorem arithmetic_sequence_a3 (h1 : is_arithmetic_seq a) (h2 : (a 1 + a 2 + a 3 + a 4 + a 5) = 25) : a 3 = 5 :=
by
  sorry

end arithmetic_sequence_a3_l642_64277


namespace initial_invited_people_l642_64291

theorem initial_invited_people (not_showed_up : ℕ) (table_capacity : ℕ) (tables_needed : ℕ) 
  (H1 : not_showed_up = 12) (H2 : table_capacity = 3) (H3 : tables_needed = 2) :
  not_showed_up + (table_capacity * tables_needed) = 18 :=
by
  sorry

end initial_invited_people_l642_64291


namespace replace_digits_divisible_by_13_l642_64231

def is_digit (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9

theorem replace_digits_divisible_by_13 :
  ∃ (x y : ℕ), is_digit x ∧ is_digit y ∧ 
  (3 * 10^6 + x * 10^4 + y * 10^2 + 3) % 13 = 0 ∧
  (x = 2 ∧ y = 3 ∨ 
   x = 5 ∧ y = 2 ∨ 
   x = 8 ∧ y = 1 ∨ 
   x = 9 ∧ y = 5 ∨ 
   x = 6 ∧ y = 6 ∨ 
   x = 3 ∧ y = 7 ∨ 
   x = 0 ∧ y = 8) :=
by
  sorry

end replace_digits_divisible_by_13_l642_64231


namespace bajazet_winning_strategy_l642_64225

-- Define the polynomial P with place holder coefficients a, b, c (assuming they are real numbers)
def P (a b c : ℝ) (x : ℝ) := x^4 + a * x^3 + b * x^2 + c * x + 1

-- The statement that regardless of how Alcina plays, Bajazet can ensure that P has a real root.
theorem bajazet_winning_strategy :
  ∃ (a b c : ℝ), ∃ (x : ℝ), P a b c x = 0 :=
sorry

end bajazet_winning_strategy_l642_64225


namespace solve_inequality_l642_64221

noncomputable def P (x : ℝ) : ℝ := x^2 - 3 * x + 2

theorem solve_inequality (x : ℝ) : (P x > 0) ↔ (x < 1 ∨ x > 2) := 
  sorry

end solve_inequality_l642_64221


namespace usual_time_is_75_l642_64226

variable (T : ℕ) -- let T be the usual time in minutes

theorem usual_time_is_75 (h1 : (6 * T) / 5 = T + 15) : T = 75 :=
by
  sorry

end usual_time_is_75_l642_64226


namespace yellow_white_flowers_count_l642_64244

theorem yellow_white_flowers_count
    (RY RW : Nat)
    (hRY : RY = 17)
    (hRW : RW = 14)
    (hRedMoreThanWhite : (RY + RW) - (RW + YW) = 4) :
    ∃ YW, YW = 13 := 
by
  sorry

end yellow_white_flowers_count_l642_64244


namespace sum_of_three_distinct_integers_l642_64299

theorem sum_of_three_distinct_integers (a b c : ℕ) (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : a ≠ c) 
  (h₄ : a * b * c = 5^3) (h₅ : a > 0) (h₆ : b > 0) (h₇ : c > 0) : a + b + c = 31 :=
by
  sorry

end sum_of_three_distinct_integers_l642_64299


namespace am_gm_four_vars_l642_64294

theorem am_gm_four_vars {a b c d : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b + c + d) * (1 / a + 1 / b + 1 / c + 1 / d) ≥ 16 :=
by
  sorry

end am_gm_four_vars_l642_64294


namespace snow_on_Monday_l642_64283

def snow_on_Tuesday : ℝ := 0.21
def snow_on_Monday_and_Tuesday : ℝ := 0.53

theorem snow_on_Monday : snow_on_Monday_and_Tuesday - snow_on_Tuesday = 0.32 :=
by
  sorry

end snow_on_Monday_l642_64283


namespace new_ratio_is_one_half_l642_64264

theorem new_ratio_is_one_half (x : ℕ) (y : ℕ) (h1 : y = 4 * x) (h2 : y = 48) :
  (x + 12) / y = 1 / 2 :=
by
  sorry

end new_ratio_is_one_half_l642_64264


namespace area_inside_quadrilateral_BCDE_outside_circle_l642_64257

noncomputable def hexagon_area (side_length : ℝ) : ℝ :=
  (3 * Real.sqrt 3) / 2 * side_length ^ 2

noncomputable def circle_area (radius : ℝ) : ℝ :=
  Real.pi * radius ^ 2

theorem area_inside_quadrilateral_BCDE_outside_circle :
  let side_length := 2
  let hex_area := hexagon_area side_length
  let hex_area_large := hexagon_area (2 * side_length)
  let circle_radius := 3
  let circle_area_A := circle_area circle_radius
  let total_area_of_interest := hex_area_large - circle_area_A
  let area_of_one_region := total_area_of_interest / 6
  area_of_one_region = 4 * Real.sqrt 3 - (3 / 2) * Real.pi :=
by
  sorry

end area_inside_quadrilateral_BCDE_outside_circle_l642_64257


namespace birds_in_tree_l642_64215

theorem birds_in_tree (initial_birds : ℝ) (birds_flew_away : ℝ) (h : initial_birds = 21.0) (h_flew : birds_flew_away = 14.0) : 
initial_birds - birds_flew_away = 7.0 :=
by
  -- proof goes here
  sorry

end birds_in_tree_l642_64215


namespace ratio_of_allergic_to_peanut_to_total_l642_64265

def total_children : ℕ := 34
def children_not_allergic_to_cashew : ℕ := 10
def children_allergic_to_both : ℕ := 10
def children_allergic_to_cashew : ℕ := 18
def children_not_allergic_to_any : ℕ := 6
def children_allergic_to_peanut : ℕ := 20

theorem ratio_of_allergic_to_peanut_to_total :
  (children_allergic_to_peanut : ℚ) / (total_children : ℚ) = 10 / 17 :=
by
  sorry

end ratio_of_allergic_to_peanut_to_total_l642_64265


namespace lines_are_perpendicular_l642_64247

-- Define the first line equation
def line1 (x y : ℝ) : Prop := x + y - 2 = 0

-- Define the second line equation
def line2 (x y : ℝ) : Prop := x - y + 3 = 0

-- Definition to determine the perpendicularity of two lines
def are_perpendicular (k1 k2 : ℝ) : Prop := k1 * k2 = -1

theorem lines_are_perpendicular :
  are_perpendicular (-1) (1) := 
by
  sorry

end lines_are_perpendicular_l642_64247


namespace xiaoming_minimum_time_l642_64281

theorem xiaoming_minimum_time :
  let review_time := 30
  let rest_time := 30
  let boil_time := 15
  let homework_time := 25
  (boil_time ≤ rest_time) → 
  (review_time + rest_time + homework_time = 85) :=
by
  intros review_time rest_time boil_time homework_time h_boil_le_rest
  sorry

end xiaoming_minimum_time_l642_64281


namespace ratio_D_to_C_l642_64233

-- Defining the terms and conditions
def speed_ratio (C Ch D : ℝ) : Prop :=
  (C = 2 * Ch) ∧
  (D / Ch = 6)

-- The theorem statement
theorem ratio_D_to_C (C Ch D : ℝ) (h : speed_ratio C Ch D) : (D / C = 3) :=
by
  sorry

end ratio_D_to_C_l642_64233


namespace graphs_intersect_once_l642_64253

theorem graphs_intersect_once : 
  ∃! (x : ℝ), |3 * x + 6| = -|4 * x - 3| :=
sorry

end graphs_intersect_once_l642_64253


namespace percentage_increase_l642_64287

theorem percentage_increase (L : ℕ) (h1 : L + 450 = 1350) :
  (450 / L : ℚ) * 100 = 50 := by
  sorry

end percentage_increase_l642_64287


namespace equal_numbers_possible_l642_64206

noncomputable def circle_operations (n : ℕ) (α : ℝ) : Prop :=
  (n ≥ 3) ∧ (∃ k : ℤ, α = 2 * Real.cos (k * Real.pi / n))

-- Statement of the theorem
theorem equal_numbers_possible (n : ℕ) (α : ℝ) (h1 : n ≥ 3) (h2 : α > 0) :
  circle_operations n α ↔ ∃ k : ℤ, α = 2 * Real.cos (k * Real.pi / n) :=
sorry

end equal_numbers_possible_l642_64206


namespace part_one_part_two_l642_64292

variable {x m : ℝ}

theorem part_one (h1 : ∀ x : ℝ, ¬(m * x^2 - (m + 1) * x + (m + 1) ≥ 0)) : m < -1 := sorry

theorem part_two (h2 : ∀ x : ℝ, 1 < x → m * x^2 - (m + 1) * x + (m + 1) ≥ 0) : m ≥ 1 / 3 := sorry

end part_one_part_two_l642_64292


namespace ages_total_l642_64252

-- Define the variables and conditions
variables (A B C : ℕ)

-- State the conditions
def condition1 (B : ℕ) : Prop := B = 14
def condition2 (A B : ℕ) : Prop := A = B + 2
def condition3 (B C : ℕ) : Prop := B = 2 * C

-- The main theorem to prove
theorem ages_total (h1 : condition1 B) (h2 : condition2 A B) (h3 : condition3 B C) : A + B + C = 37 :=
by
  sorry

end ages_total_l642_64252


namespace recreation_spent_percent_l642_64269

variable (W : ℝ) -- Assume W is the wages last week

-- Conditions
def last_week_spent_on_recreation (W : ℝ) : ℝ := 0.25 * W
def this_week_wages (W : ℝ) : ℝ := 0.70 * W
def this_week_spent_on_recreation (W : ℝ) : ℝ := 0.50 * (this_week_wages W)

-- Proof statement
theorem recreation_spent_percent (W : ℝ) :
  (this_week_spent_on_recreation W / last_week_spent_on_recreation W) * 100 = 140 := by
  sorry

end recreation_spent_percent_l642_64269


namespace solve_abs_inequality_l642_64234

theorem solve_abs_inequality (x : ℝ) :
  3 ≤ abs ((x - 3)^2 - 4) ∧ abs ((x - 3)^2 - 4) ≤ 7 ↔ 3 - Real.sqrt 11 ≤ x ∧ x ≤ 3 + Real.sqrt 11 :=
sorry

end solve_abs_inequality_l642_64234


namespace polynomial_remainder_l642_64240

theorem polynomial_remainder (x : ℂ) (hx : x^5 = 1) :
  (x^25 + x^20 + x^15 + x^10 + x^5 + 1) % (x^5 - 1) = 6 :=
by
  -- Proof will go here
  sorry

end polynomial_remainder_l642_64240


namespace police_speed_l642_64270

/-- 
A thief runs away from a location with a speed of 20 km/hr.
A police officer starts chasing him from a location 60 km away after 1 hour.
The police officer catches the thief after 4 hours.
Prove that the speed of the police officer is 40 km/hr.
-/
theorem police_speed
  (thief_speed : ℝ)
  (police_start_distance : ℝ)
  (police_chase_time : ℝ)
  (time_head_start : ℝ)
  (police_distance_to_thief : ℝ)
  (thief_distance_after_time : ℝ)
  (total_distance_police_officer : ℝ) :
  thief_speed = 20 ∧
  police_start_distance = 60 ∧
  police_chase_time = 4 ∧
  time_head_start = 1 ∧
  police_distance_to_thief = police_start_distance + 100 ∧
  thief_distance_after_time = thief_speed * police_chase_time + thief_speed * time_head_start ∧
  total_distance_police_officer = police_start_distance + (thief_speed * (police_chase_time + time_head_start)) →
  (total_distance_police_officer / police_chase_time) = 40 := by
  sorry

end police_speed_l642_64270


namespace length_DE_l642_64258

theorem length_DE (AB : ℝ) (h_base : AB = 15) (DE_parallel : ∀ x y z : Triangle, True) (area_ratio : ℝ) (h_area_ratio : area_ratio = 0.25) : 
  ∃ DE : ℝ, DE = 7.5 :=
by
  sorry

end length_DE_l642_64258


namespace share_of_B_is_2400_l642_64245

noncomputable def share_of_B (total_profit : ℝ) (B_investment : ℝ) (A_months B_months C_months D_months : ℝ) : ℝ :=
  let A_investment := 3 * B_investment
  let C_investment := (3/2) * B_investment
  let D_investment := (1/2) * A_investment
  let A_inv_months := A_investment * A_months
  let B_inv_months := B_investment * B_months
  let C_inv_months := C_investment * C_months
  let D_inv_months := D_investment * D_months
  let total_inv_months := A_inv_months + B_inv_months + C_inv_months + D_inv_months
  (B_inv_months / total_inv_months) * total_profit

theorem share_of_B_is_2400 :
  share_of_B 27000 (1000 : ℝ) 12 6 9 8 = 2400 := 
sorry

end share_of_B_is_2400_l642_64245


namespace roman_created_171_roman_created_1513_m1_roman_created_1513_m2_roman_created_largest_l642_64298

-- Lean 4 statements to capture the proofs without computation.
theorem roman_created_171 (a b : ℕ) (h_sum : a + b = 17) (h_diff : a - b = 1) : 
  a = 9 ∧ b = 8 ∨ a = 8 ∧ b = 9 := 
  sorry

theorem roman_created_1513_m1 (a b : ℕ) (h_sum : a + b = 15) (h_diff : a - b = 13) : 
  a = 14 ∧ b = 1 ∨ a = 1 ∧ b = 14 := 
  sorry

theorem roman_created_1513_m2 (a b : ℕ) (h_sum : a + b = 151) (h_diff : a - b = 3) : 
  a = 77 ∧ b = 74 ∨ a = 74 ∧ b = 77 := 
  sorry

theorem roman_created_largest (a b : ℕ) (h_sum : a + b = 188) (h_diff : a - b = 10) : 
  a = 99 ∧ b = 89 ∨ a = 89 ∧ b = 99 := 
  sorry

end roman_created_171_roman_created_1513_m1_roman_created_1513_m2_roman_created_largest_l642_64298


namespace am_gm_inequality_l642_64251

theorem am_gm_inequality (x y z : ℝ) (n : ℕ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) (h_sum : x + y + z = 1) (h_pos_n : n > 0):
  x^n + y^n + z^n ≥ 1 / 3^(n-1) :=
by
  sorry

end am_gm_inequality_l642_64251


namespace find_y_z_l642_64254

theorem find_y_z (y z : ℝ) : 
  (∃ k : ℝ, (1:ℝ) = -k ∧ (2:ℝ) = k * y ∧ (3:ℝ) = k * z) → y = -2 ∧ z = -3 :=
by
  sorry

end find_y_z_l642_64254


namespace problem1_solution_set_problem2_range_of_a_l642_64208

-- Definitions and statements for Problem 1
def f1 (x : ℝ) : ℝ := -12 * x ^ 2 - 2 * x + 2

theorem problem1_solution_set :
  (∃ a b : ℝ, a = -12 ∧ b = -2 ∧
    ∀ x : ℝ, f1 x > 0 → -1 / 2 < x ∧ x < 1 / 3) :=
by sorry

-- Definitions and statements for Problem 2
def f2 (x a : ℝ) : ℝ := a * x ^ 2 - x + 2

theorem problem2_range_of_a :
  (∃ b : ℝ, b = -1 ∧
    ∀ a : ℝ, (∀ x : ℝ, f2 x a < 0 → false) → a ≥ 1 / 8) :=
by sorry

end problem1_solution_set_problem2_range_of_a_l642_64208


namespace find_a_value_l642_64279

-- Definitions of conditions
def eq_has_positive_root (a : ℝ) : Prop :=
  ∃ (x : ℝ), x > 0 ∧ (x / (x - 5) = 3 - (a / (x - 5)))

-- Statement of the theorem
theorem find_a_value (a : ℝ) (h : eq_has_positive_root a) : a = -5 := 
  sorry

end find_a_value_l642_64279


namespace sum_of_arithmetic_sequence_l642_64290

variables (a_n : Nat → Int) (S_n : Nat → Int)
variable (n : Nat)

-- Definitions based on given conditions:
def is_arithmetic_sequence (a_n : Nat → Int) : Prop :=
∀ n, a_n (n + 1) = a_n n + a_n 1 - a_n 0

def a_1 : Int := -2018

def arithmetic_sequence_sum (S_n : Nat → Int) (a_n : Nat → Int) (n : Nat) : Prop :=
S_n n = n * a_n 0 + (n * (n - 1) / 2 * (a_n 1 - a_n 0))

-- Given condition S_12 / 12 - S_10 / 10 = 2
def condition (S_n : Nat → Int) : Prop :=
S_n 12 / 12 - S_n 10 / 10 = 2

-- Goal: Prove S_2018 = -2018
theorem sum_of_arithmetic_sequence (a_n S_n : Nat → Int)
  (h1 : a_n 1 = -2018)
  (h2 : is_arithmetic_sequence a_n)
  (h3 : ∀ n, arithmetic_sequence_sum S_n a_n n)
  (h4 : condition S_n) :
  S_n 2018 = -2018 :=
sorry

end sum_of_arithmetic_sequence_l642_64290


namespace tan_simplification_l642_64293

theorem tan_simplification 
  (θ : ℝ) 
  (h : Real.tan θ = 3) : 
  (1 - Real.sin θ) / (Real.cos θ) - (Real.cos θ) / (1 + Real.sin θ) = 0 := 
by 
  sorry

end tan_simplification_l642_64293


namespace exp_gt_pow_l642_64220

theorem exp_gt_pow (x : ℝ) (h_pos : 0 < x) (h_ne : x ≠ Real.exp 1) : Real.exp x > x ^ Real.exp 1 := by
  sorry

end exp_gt_pow_l642_64220


namespace sum_of_polynomials_l642_64246

open Polynomial

noncomputable def f : ℚ[X] := -4 * X^2 + 2 * X - 5
noncomputable def g : ℚ[X] := -6 * X^2 + 4 * X - 9
noncomputable def h : ℚ[X] := 6 * X^2 + 6 * X + 2

theorem sum_of_polynomials :
  f + g + h = -4 * X^2 + 12 * X - 12 :=
by sorry

end sum_of_polynomials_l642_64246


namespace single_discount_percentage_l642_64243

noncomputable def original_price : ℝ := 9795.3216374269
noncomputable def sale_price : ℝ := 6700
noncomputable def discount_percentage (p₀ p₁ : ℝ) : ℝ := ((p₀ - p₁) / p₀) * 100

theorem single_discount_percentage :
  discount_percentage original_price sale_price = 31.59 := 
by
  sorry

end single_discount_percentage_l642_64243


namespace find_original_number_l642_64242

theorem find_original_number
  (n : ℤ)
  (h : (2 * (n + 2) - 2) / 2 = 7) :
  n = 6 := 
sorry

end find_original_number_l642_64242


namespace hotel_flat_fee_l642_64230

theorem hotel_flat_fee
  (f n : ℝ)
  (h1 : f + 3 * n = 195)
  (h2 : f + 7 * n = 380) :
  f = 56.25 :=
by sorry

end hotel_flat_fee_l642_64230


namespace compute_result_l642_64203

theorem compute_result : (300000 * 200000) / 100000 = 600000 := by
  sorry

end compute_result_l642_64203


namespace remainder_ab_div_48_is_15_l642_64248

noncomputable def remainder_ab_div_48 (a b : ℕ) (ha : a % 8 = 3) (hb : b % 6 = 5) : ℕ :=
  (a * b) % 48

theorem remainder_ab_div_48_is_15 {a b : ℕ} (ha : a % 8 = 3) (hb : b % 6 = 5) : remainder_ab_div_48 a b ha hb = 15 :=
  sorry

end remainder_ab_div_48_is_15_l642_64248


namespace zoo_visitors_l642_64261

theorem zoo_visitors (P : ℕ) (h : 3 * P = 3750) : P = 1250 :=
by 
  sorry

end zoo_visitors_l642_64261


namespace sum_h_k_a_b_l642_64275

-- Defining h, k, a, and b with their respective given values
def h : Int := -4
def k : Int := 2
def a : Int := 5
def b : Int := 3

-- Stating the theorem to prove \( h + k + a + b = 6 \)
theorem sum_h_k_a_b : h + k + a + b = 6 := by
  /- Proof omitted as per instructions -/
  sorry

end sum_h_k_a_b_l642_64275


namespace pages_per_day_difference_l642_64255

theorem pages_per_day_difference :
  let songhee_pages := 288
  let songhee_days := 12
  let eunju_pages := 243
  let eunju_days := 9
  let songhee_per_day := songhee_pages / songhee_days
  let eunju_per_day := eunju_pages / eunju_days
  eunju_per_day - songhee_per_day = 3 := by
  sorry

end pages_per_day_difference_l642_64255


namespace eighth_term_of_arithmetic_sequence_l642_64213

noncomputable def arithmetic_sequence (n : ℕ) (a1 an : ℚ) (k : ℕ) : ℚ :=
  a1 + (k - 1) * ((an - a1) / (n - 1))

theorem eighth_term_of_arithmetic_sequence :
  ∀ (a1 a30 : ℚ), a1 = 5 → a30 = 86 → 
  arithmetic_sequence 30 a1 a30 8 = 592 / 29 :=
by
  intros a1 a30 h_a1 h_a30
  rw [h_a1, h_a30]
  dsimp [arithmetic_sequence]
  sorry

end eighth_term_of_arithmetic_sequence_l642_64213


namespace calculation_l642_64285

theorem calculation (a b : ℕ) (h1 : a = 7) (h2 : b = 5) : (a^2 - b^2) ^ 2 = 576 :=
by
  sorry

end calculation_l642_64285


namespace fixed_point_l642_64271

noncomputable def func (a : ℝ) (x : ℝ) : ℝ := a^(x-1)

theorem fixed_point (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) : func a 1 = 1 :=
by {
  -- We need to prove that func a 1 = 1 for any a > 0 and a ≠ 1
  sorry
}

end fixed_point_l642_64271


namespace problem_statement_l642_64273

noncomputable def U : Set Int := {-2, -1, 0, 1, 2}
noncomputable def A : Set Int := {x : Int | -2 ≤ x ∧ x < 0}
noncomputable def B : Set Int := {x : Int | (x = 0 ∨ x = 1)} -- since natural numbers typically include positive integers, adapting B contextually

theorem problem_statement : ((U \ A) ∩ B) = {0, 1} := by
  sorry

end problem_statement_l642_64273


namespace prove_b_zero_l642_64211

variables {a b c : ℕ}

theorem prove_b_zero (h1 : ∃ (a b c : ℕ), a^5 + 4 * b^5 = c^5 ∧ c % 2 = 0) : b = 0 :=
sorry

end prove_b_zero_l642_64211


namespace sqrt_x_div_sqrt_y_as_fraction_l642_64260

theorem sqrt_x_div_sqrt_y_as_fraction 
  (x y : ℝ)
  (h : (1/3)^2 + (1/4)^2 + (1/6)^2 = 54 * x / 115 * y * ((1/5)^2 + (1/7)^2 + (1/8)^2)) : 
  (Real.sqrt x) / (Real.sqrt y) = 49 / 29 :=
by
  sorry

end sqrt_x_div_sqrt_y_as_fraction_l642_64260


namespace find_x_l642_64297

variable (x : ℝ)
variable (y : ℝ := x * 3.5)
variable (z : ℝ := y / 0.00002)

theorem find_x (h : z = 840) : x = 0.0048 :=
sorry

end find_x_l642_64297


namespace find_speed_l642_64276

theorem find_speed (v : ℝ) (t : ℝ) (h : t = 5 * v^2) (ht : t = 20) : v = 2 :=
by
  sorry

end find_speed_l642_64276


namespace complement_of_M_in_U_l642_64217

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {x | x^2 - 5*x + 6 = 0}
def C_U (M : Set ℕ) (U : Set ℕ) : Set ℕ := U \ M

theorem complement_of_M_in_U : C_U M U = {1, 4} :=
by
  sorry

end complement_of_M_in_U_l642_64217


namespace boys_girls_rel_l642_64200

theorem boys_girls_rel (b g : ℕ) (h : g = 7 + 2 * (b - 1)) : b = (g - 5) / 2 := 
by sorry

end boys_girls_rel_l642_64200


namespace interval_satisfies_ineq_l642_64238

theorem interval_satisfies_ineq (p : ℝ) (h1 : 18 * p < 10) (h2 : 0.5 < p) : 0.5 < p ∧ p < 5 / 9 :=
by {
  sorry -- Proof not required, only the statement.
}

end interval_satisfies_ineq_l642_64238


namespace floor_sqrt_27_square_l642_64295

theorem floor_sqrt_27_square : (Int.floor (Real.sqrt 27))^2 = 25 :=
by
  sorry

end floor_sqrt_27_square_l642_64295


namespace four_distinct_real_roots_l642_64202

noncomputable def f (x d : ℝ) : ℝ := x^2 + 10*x + d

theorem four_distinct_real_roots (d : ℝ) :
  (∀ r, f r d = 0 → (∃! x, f x d = r)) → d < 25 :=
by
  sorry

end four_distinct_real_roots_l642_64202


namespace increasing_function_unique_root_proof_l642_64286

noncomputable def increasing_function_unique_root (f : ℝ → ℝ) :=
  (∀ x y : ℝ, x < y → f x ≤ f y) -- condition for increasing function
  ∧ ∃! x : ℝ, f x = 0 -- exists exactly one root

theorem increasing_function_unique_root_proof
  (f : ℝ → ℝ)
  (h_inc : ∀ x y : ℝ, x < y → f x ≤ f y)
  (h_ex : ∃ x : ℝ, f x = 0) :
  ∃! x : ℝ, f x = 0 := sorry

end increasing_function_unique_root_proof_l642_64286


namespace range_of_a_proof_l642_64289

noncomputable def range_of_a (a : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 + a * x + 1 > 0

theorem range_of_a_proof (a : ℝ) : range_of_a a ↔ 0 ≤ a ∧ a < 4 :=
by
  sorry

end range_of_a_proof_l642_64289


namespace certain_number_d_sq_l642_64224

theorem certain_number_d_sq (d n m : ℕ) (hd : d = 14) (h : n * d = m^2) : n = 14 :=
by
  sorry

end certain_number_d_sq_l642_64224


namespace sum_of_x_for_ggg_eq_neg2_l642_64205

noncomputable def g (x : ℝ) := (x^2) / 3 + x - 2

theorem sum_of_x_for_ggg_eq_neg2 : (∃ x1 x2 : ℝ, (g (g (g x1)) = -2 ∧ g (g (g x2)) = -2 ∧ x1 ≠ x2)) ∧ (x1 + x2 = 0) :=
by
  sorry

end sum_of_x_for_ggg_eq_neg2_l642_64205


namespace minimal_n_is_40_l642_64284

def sequence_minimal_n (p : ℝ) (a : ℕ → ℝ) : Prop :=
  a 1 = p ∧
  a 2 = p + 1 ∧
  (∀ n, n ≥ 1 → a (n + 2) - 2 * a (n + 1) + a n = n - 20) ∧
  (∀ n, a n ≥ p) -- Since minimal \(a_n\) implies non-negative with given \(a_1, a_2\)

theorem minimal_n_is_40 (p : ℝ) (a : ℕ → ℝ) (h : sequence_minimal_n p a) : ∃ n, n = 40 ∧ (∀ m, n ≠ m → a n ≤ a m) :=
by
  obtain ⟨h1, h2, h3⟩ := h
  sorry

end minimal_n_is_40_l642_64284


namespace last_two_digits_sum_is_32_l642_64266

-- Definitions for digit representation
variables (z a r l m : ℕ)

-- Numbers definitions
def ZARAZA := z * 10^5 + a * 10^4 + r * 10^3 + a * 10^2 + z * 10 + a
def ALMAZ := a * 10^4 + l * 10^3 + m * 10^2 + a * 10 + z

-- Condition that ZARAZA is divisible by 4
def divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

-- Condition that ALMAZ is divisible by 28
def divisible_by_28 (n : ℕ) : Prop := n % 28 = 0

-- The theorem to prove
theorem last_two_digits_sum_is_32
  (hz4 : divisible_by_4 (ZARAZA z a r))
  (ha28 : divisible_by_28 (ALMAZ a l m z))
  : (ZARAZA z a r + ALMAZ a l m z) % 100 = 32 :=
by sorry

end last_two_digits_sum_is_32_l642_64266


namespace find_k_value_l642_64232

theorem find_k_value : 
  let a := 3 ^ 1001
  let b := 4 ^ 1002
  (a + b) ^ 2 - (a - b) ^ 2 = 16 * 12 ^ 1001 :=
by
  let a := 3 ^ 1001
  let b := 4 ^ 1002
  sorry

end find_k_value_l642_64232


namespace square_side_length_l642_64250

theorem square_side_length (a b s : ℝ) 
  (h_area : a * b = 54) 
  (h_square_condition : 3 * a = b / 2) : 
  s = 9 :=
by 
  sorry

end square_side_length_l642_64250


namespace proportional_segments_l642_64249

theorem proportional_segments (a1 a2 a3 a4 b1 b2 b3 b4 c1 c2 c3 c4 d1 d2 d3 d4 : ℕ)
  (hA : a1 = 1 ∧ a2 = 2 ∧ a3 = 3 ∧ a4 = 4)
  (hB : b1 = 1 ∧ b2 = 2 ∧ b3 = 2 ∧ b4 = 4)
  (hC : c1 = 3 ∧ c2 = 5 ∧ c3 = 9 ∧ c4 = 13)
  (hD : d1 = 1 ∧ d2 = 2 ∧ d3 = 2 ∧ d4 = 3) :
  (b1 * b4 = b2 * b3) :=
by
  sorry

end proportional_segments_l642_64249


namespace geometric_sequence_sum_ratio_l642_64222

theorem geometric_sequence_sum_ratio (a_n : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (h_nonzero_q : q ≠ 0) 
  (a2 : a_n 2 = a_n 1 * q) (a5 : a_n 5 = a_n 1 * q^4) 
  (h_condition : 8 * a_n 2 + a_n 5 = 0)
  (h_sum : ∀ n, S n = a_n 1 * (1 - q^n) / (1 - q)) : 
  S 5 / S 2 = -11 :=
by 
  sorry

end geometric_sequence_sum_ratio_l642_64222


namespace total_fish_at_wedding_l642_64263

def num_tables : ℕ := 32
def fish_per_table_except_one : ℕ := 2
def fish_on_special_table : ℕ := 3
def number_of_special_tables : ℕ := 1
def number_of_regular_tables : ℕ := num_tables - number_of_special_tables

theorem total_fish_at_wedding : 
  (number_of_regular_tables * fish_per_table_except_one) + (number_of_special_tables * fish_on_special_table) = 65 :=
by
  sorry

end total_fish_at_wedding_l642_64263
