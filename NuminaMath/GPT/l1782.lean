import Mathlib

namespace cans_in_each_package_of_cat_food_l1782_178296

-- Definitions and conditions
def cans_per_package_cat (c : ℕ) := 9 * c
def cans_per_package_dog := 7 * 5
def extra_cans_cat := 55

-- Theorem stating the problem and the answer
theorem cans_in_each_package_of_cat_food (c : ℕ) (h: cans_per_package_cat c = cans_per_package_dog + extra_cans_cat) :
  c = 10 :=
sorry

end cans_in_each_package_of_cat_food_l1782_178296


namespace existence_of_positive_numbers_l1782_178230

open Real

theorem existence_of_positive_numbers {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^2 + b^2 + c^2 > 2 ∧ a^3 + b^3 + c^3 < 2 ∧ a^4 + b^4 + c^4 > 2 :=
sorry

end existence_of_positive_numbers_l1782_178230


namespace white_washing_cost_correct_l1782_178269

def room_length : ℝ := 25
def room_width : ℝ := 15
def room_height : ℝ := 12

def door_length : ℝ := 6
def door_width : ℝ := 3

def window_length : ℝ := 4
def window_width : ℝ := 3

def cost_per_sq_ft : ℝ := 8

def calculate_white_washing_cost : ℝ :=
  let total_wall_area := 2 * (room_length * room_height) + 2 * (room_width * room_height)
  let door_area := door_length * door_width
  let window_area := 3 * (window_length * window_width)
  let effective_area := total_wall_area - door_area - window_area
  effective_area * cost_per_sq_ft

theorem white_washing_cost_correct : calculate_white_washing_cost = 7248 := by
  sorry

end white_washing_cost_correct_l1782_178269


namespace negation_exists_x_squared_lt_zero_l1782_178235

open Classical

theorem negation_exists_x_squared_lt_zero :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x : ℝ, x^2 < 0) :=
by 
  sorry

end negation_exists_x_squared_lt_zero_l1782_178235


namespace flower_cost_l1782_178250

theorem flower_cost (F : ℕ) (h1 : F + (F + 20) + (F - 2) = 45) : F = 9 :=
by
  sorry

end flower_cost_l1782_178250


namespace voronovich_inequality_l1782_178224

theorem voronovich_inequality (a b c : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : a + b + c = 1) :
  (a^2 + b^2 + c^2)^2 + 6 * a * b * c ≥ a * b + b * c + c * a :=
by
  sorry

end voronovich_inequality_l1782_178224


namespace arc_length_120_degrees_l1782_178242

theorem arc_length_120_degrees (π : ℝ) : 
  let R := π
  let n := 120
  (n * π * R) / 180 = (2 * π^2) / 3 := 
by
  let R := π
  let n := 120
  sorry

end arc_length_120_degrees_l1782_178242


namespace b_l1782_178234

def initial_marbles : Nat := 24
def lost_through_hole : Nat := 4
def given_away : Nat := 2 * lost_through_hole
def eaten_by_dog : Nat := lost_through_hole / 2

theorem b {m : Nat} (h₁ : m = initial_marbles - lost_through_hole)
  (h₂ : m - given_away = m₁)
  (h₃ : m₁ - eaten_by_dog = 10) :
  m₁ - eaten_by_dog = 10 := sorry

end b_l1782_178234


namespace compute_r_l1782_178299

variables {j p t m n x y r : ℝ}

theorem compute_r
    (h1 : j = 0.75 * p)
    (h2 : j = 0.80 * t)
    (h3 : t = p - r * p / 100)
    (h4 : m = 1.10 * p)
    (h5 : n = 0.70 * m)
    (h6 : j + p + t = m * n)
    (h7 : x = 1.15 * j)
    (h8 : y = 0.80 * n)
    (h9 : x * y = (j + p + t) ^ 2) : r = 6.25 := by
  sorry

end compute_r_l1782_178299


namespace paul_money_duration_l1782_178216

theorem paul_money_duration
  (mow_earnings : ℕ)
  (weed_earnings : ℕ)
  (weekly_expenses : ℕ)
  (earnings_mow : mow_earnings = 3)
  (earnings_weed : weed_earnings = 3)
  (expenses : weekly_expenses = 3) :
  (mow_earnings + weed_earnings) / weekly_expenses = 2 := 
by
  sorry

end paul_money_duration_l1782_178216


namespace fair_dice_can_be_six_l1782_178238

def fair_dice_outcomes : Set ℕ := {1, 2, 3, 4, 5, 6}

theorem fair_dice_can_be_six : 6 ∈ fair_dice_outcomes :=
by {
  -- This formally states that 6 is a possible outcome when throwing a fair dice
  sorry
}

end fair_dice_can_be_six_l1782_178238


namespace convert_neg_900_deg_to_rad_l1782_178210

theorem convert_neg_900_deg_to_rad : (-900 : ℝ) * (Real.pi / 180) = -5 * Real.pi :=
by
  sorry

end convert_neg_900_deg_to_rad_l1782_178210


namespace ratio_black_haired_children_l1782_178267

theorem ratio_black_haired_children 
  (n_red : ℕ) (n_total : ℕ) (ratio_red : ℕ) (ratio_blonde : ℕ) (ratio_black : ℕ)
  (h_ratio : ratio_red / ratio_red = 1 ∧ ratio_blonde / ratio_red = 2 ∧ ratio_black / ratio_red = 7 / 3)
  (h_n_red : n_red = 9)
  (h_n_total : n_total = 48) :
  (7 : ℚ) / (16 : ℚ) = (n_total * 7 / 16 : ℚ) :=
sorry

end ratio_black_haired_children_l1782_178267


namespace final_score_l1782_178207

theorem final_score (questions_first_half questions_second_half points_per_question : ℕ) (h1 : questions_first_half = 5) (h2 : questions_second_half = 5) (h3 : points_per_question = 5) : 
  (questions_first_half + questions_second_half) * points_per_question = 50 :=
by
  sorry

end final_score_l1782_178207


namespace first_player_wins_l1782_178278

noncomputable def game_win_guarantee : Prop :=
  ∃ (first_can_guarantee_win : Bool),
    first_can_guarantee_win = true

theorem first_player_wins :
  ∀ (nuts : ℕ) (players : (ℕ × ℕ)) (move : ℕ → ℕ) (end_condition : ℕ → Prop),
    nuts = 10 →
    players = (1, 2) →
    (∀ n, 0 < n ∧ n ≤ nuts → move n = n - 1) →
    (end_condition 3 = true) →
    (∀ x y z, x + y + z = 3 ↔ end_condition (x + y + z)) → 
    game_win_guarantee :=
by
  intros nuts players move end_condition H1 H2 H3 H4 H5
  sorry

end first_player_wins_l1782_178278


namespace solution_count_l1782_178228

noncomputable def equation_has_one_solution : Prop :=
∀ x : ℝ, (x - (8 / (x - 2))) = (4 - (8 / (x - 2))) → x = 4

theorem solution_count : equation_has_one_solution :=
by
  sorry

end solution_count_l1782_178228


namespace triangle_sides_arithmetic_progression_l1782_178276

theorem triangle_sides_arithmetic_progression (a d : ℤ) (h : 3 * a = 15) (h1 : a > 0) (h2 : d ≥ 0) :
  (a - d = 5 ∨ a - d = 4 ∨ a - d = 3) ∧ 
  (a = 5) ∧ 
  (a + d = 5 ∨ a + d = 6 ∨ a + d = 7) := 
  sorry

end triangle_sides_arithmetic_progression_l1782_178276


namespace units_digit_of_product_l1782_178254

theorem units_digit_of_product : 
  (27 % 10 = 7) ∧ (68 % 10 = 8) → ((27 * 68) % 10 = 6) :=
by sorry

end units_digit_of_product_l1782_178254


namespace taxi_fare_total_distance_l1782_178208

theorem taxi_fare_total_distance (initial_fare additional_fare : ℝ) (total_fare : ℝ) (initial_distance additional_distance : ℝ) :
  initial_fare = 10 ∧ additional_fare = 1 ∧ initial_distance = 1/5 ∧ (total_fare = 59) →
  (total_distance = initial_distance + additional_distance * ((total_fare - initial_fare) / additional_fare)) →
  total_distance = 10 := 
by 
  sorry

end taxi_fare_total_distance_l1782_178208


namespace lines_intersect_at_same_points_l1782_178266

-- Definitions of linear equations in system 1 and system 2
def line1 (a1 b1 c1 x y : ℝ) := a1 * x + b1 * y = c1
def line2 (a2 b2 c2 x y : ℝ) := a2 * x + b2 * y = c2
def line3 (a3 b3 c3 x y : ℝ) := a3 * x + b3 * y = c3
def line4 (a4 b4 c4 x y : ℝ) := a4 * x + b4 * y = c4

-- Equivalence condition of the systems
def systems_equivalent (a1 b1 c1 a2 b2 c2 a3 b3 c3 a4 b4 c4 : ℝ) :=
  ∀ (x y : ℝ), (line1 a1 b1 c1 x y ∧ line2 a2 b2 c2 x y) ↔ (line3 a3 b3 c3 x y ∧ line4 a4 b4 c4 x y)

-- Proof statement that the four lines intersect at the same set of points
theorem lines_intersect_at_same_points (a1 b1 c1 a2 b2 c2 a3 b3 c3 a4 b4 c4 : ℝ) :
  systems_equivalent a1 b1 c1 a2 b2 c2 a3 b3 c3 a4 b4 c4 →
  ∀ (x y : ℝ), (line1 a1 b1 c1 x y ∧ line2 a2 b2 c2 x y) ↔ (line3 a3 b3 c3 x y ∧ line4 a4 b4 c4 x y) :=
by
  intros h_equiv x y
  exact h_equiv x y

end lines_intersect_at_same_points_l1782_178266


namespace factor_expression_l1782_178237

variable (a : ℤ)

theorem factor_expression : 58 * a^2 + 174 * a = 58 * a * (a + 3) := by
  sorry

end factor_expression_l1782_178237


namespace factor_polynomial_l1782_178270

noncomputable def gcd_coeffs : ℕ := Nat.gcd 72 180

theorem factor_polynomial (x : ℝ) (GCD_72_180 : gcd_coeffs = 36)
    (GCD_x5_x9 : ∃ (y: ℝ), x^5 = y ∧ x^9 = y * x^4) :
    72 * x^5 - 180 * x^9 = -36 * x^5 * (5 * x^4 - 2) :=
by
  sorry

end factor_polynomial_l1782_178270


namespace remainder_of_5_pow_2023_mod_17_l1782_178213

theorem remainder_of_5_pow_2023_mod_17 :
  5^2023 % 17 = 11 :=
by
  have h1 : 5^2 % 17 = 8 := by sorry
  have h2 : 5^4 % 17 = 13 := by sorry
  have h3 : 5^8 % 17 = -1 := by sorry
  have h4 : 5^16 % 17 = 1 := by sorry
  have h5 : 2023 = 16 * 126 + 7 := by sorry
  sorry

end remainder_of_5_pow_2023_mod_17_l1782_178213


namespace factorize_polynomial_l1782_178298

theorem factorize_polynomial :
  ∀ (x : ℝ), x^4 + 2021 * x^2 + 2020 * x + 2021 = (x^2 + x + 1) * (x^2 - x + 2021) :=
by
  intros x
  sorry

end factorize_polynomial_l1782_178298


namespace probability_of_drawing_1_red_1_white_l1782_178215

-- Definitions
def total_balls : ℕ := 5
def red_balls : ℕ := 2
def white_balls : ℕ := 3

-- Probabilities
def p_red_first_white_second : ℚ := (red_balls / total_balls : ℚ) * (white_balls / total_balls : ℚ)
def p_white_first_red_second : ℚ := (white_balls / total_balls : ℚ) * (red_balls / total_balls : ℚ)

-- Total probability
def total_probability : ℚ := p_red_first_white_second + p_white_first_red_second

theorem probability_of_drawing_1_red_1_white :
  total_probability = 12 / 25 := by
  sorry

end probability_of_drawing_1_red_1_white_l1782_178215


namespace rational_solutions_of_quadratic_l1782_178256

theorem rational_solutions_of_quadratic (k : ℕ) (hk : 0 < k ∧ k ≤ 10) :
  ∃ (x : ℚ), k * x^2 + 20 * x + k = 0 ↔ (k = 6 ∨ k = 8 ∨ k = 10) :=
by sorry

end rational_solutions_of_quadratic_l1782_178256


namespace solve_linear_function_l1782_178285

theorem solve_linear_function :
  (∀ (x y : ℤ), (x = -3 ∧ y = -4) ∨ (x = -2 ∧ y = -2) ∨ (x = -1 ∧ y = 0) ∨ 
                      (x = 0 ∧ y = 2) ∨ (x = 1 ∧ y = 4) ∨ (x = 2 ∧ y = 6) →
   ∃ (a b : ℤ), y = a * x + b ∧ a * 1 + b = 4) :=
sorry

end solve_linear_function_l1782_178285


namespace health_risk_probability_l1782_178245

theorem health_risk_probability :
  let a := 0.08 * 500
  let b := 0.08 * 500
  let c := 0.08 * 500
  let d := 0.18 * 500
  let e := 0.18 * 500
  let f := 0.18 * 500
  let g := 0.05 * 500
  let h := 500 - (3 * 40 + 3 * 90 + 25)
  let q := 500 - (a + d + e + g)
  let p := 1
  let q := 3
  p + q = 4 := sorry

end health_risk_probability_l1782_178245


namespace minimum_value_property_l1782_178259

noncomputable def min_value_expression (x : ℝ) (h : x > 10) : ℝ :=
  (x^2 + 36) / (x - 10)

noncomputable def min_value : ℝ := 4 * Real.sqrt 34 + 20

theorem minimum_value_property (x : ℝ) (h : x > 10) :
  min_value_expression x h >= min_value := by
  sorry

end minimum_value_property_l1782_178259


namespace carol_initial_cupcakes_l1782_178220

variable (x : ℕ)

theorem carol_initial_cupcakes (h : (x - 9) + 28 = 49) : x = 30 := 
  sorry

end carol_initial_cupcakes_l1782_178220


namespace hyperbola_ellipse_equations_l1782_178212

theorem hyperbola_ellipse_equations 
  (F1 F2 P : ℝ × ℝ) 
  (hF1 : F1 = (0, -5))
  (hF2 : F2 = (0, 5))
  (hP : P = (3, 4)) :
  (∃ a b : ℝ, a^2 = 40 ∧ b^2 = 16 ∧ 
    ∀ x y : ℝ, (y^2 / 40 + x^2 / 15 = 1 ↔ y^2 / a^2 + x^2 / (a^2 - 25) = 1) ∧
    (y^2 / 16 - x^2 / 9 = 1 ↔ y^2 / b^2 - x^2 / (25 - b^2) = 1)) :=
sorry

end hyperbola_ellipse_equations_l1782_178212


namespace pages_with_same_units_digit_count_l1782_178233

theorem pages_with_same_units_digit_count {n : ℕ} (h1 : n = 67) :
  ∃ k : ℕ, k = 13 ∧
  (∀ x : ℕ, 1 ≤ x ∧ x ≤ n → 
    (x ≡ (n + 1 - x) [MOD 10] ↔ 
     (x % 10 = 4 ∨ x % 10 = 9))) :=
by
  sorry

end pages_with_same_units_digit_count_l1782_178233


namespace jellybean_probability_l1782_178206

/-- A bowl contains 15 jellybeans: five red, three blue, five white, and two green. If you pick four 
    jellybeans from the bowl at random and without replacement, the probability that exactly three will 
    be red is 20/273. -/
theorem jellybean_probability :
  let total_jellybeans := 15
  let red_jellybeans := 5
  let blue_jellybeans := 3
  let white_jellybeans := 5
  let green_jellybeans := 2
  let total_combinations := Nat.choose total_jellybeans 4
  let favorable_combinations := (Nat.choose red_jellybeans 3) * (Nat.choose (total_jellybeans - red_jellybeans) 1)
  let probability := favorable_combinations / total_combinations
  probability = 20 / 273 :=
by
  sorry

end jellybean_probability_l1782_178206


namespace value_of_2_star_3_l1782_178271

def star (a b : ℕ) : ℕ := a * b ^ 3 - b + 2

theorem value_of_2_star_3 : star 2 3 = 53 :=
by
  -- This is where the proof would go
  sorry

end value_of_2_star_3_l1782_178271


namespace cracked_to_broken_eggs_ratio_l1782_178243

theorem cracked_to_broken_eggs_ratio (total_eggs : ℕ) (broken_eggs : ℕ) (P C : ℕ)
  (h1 : total_eggs = 24)
  (h2 : broken_eggs = 3)
  (h3 : P - C = 9)
  (h4 : P + C = 21) :
  (C : ℚ) / (broken_eggs : ℚ) = 2 :=
by
  sorry

end cracked_to_broken_eggs_ratio_l1782_178243


namespace rehabilitation_centers_l1782_178260

def Lisa : ℕ := 6 
def Jude : ℕ := Lisa / 2
def Han : ℕ := 2 * Jude - 2
def Jane : ℕ := 27 - Lisa - Jude - Han
def x : ℕ := 2

theorem rehabilitation_centers:
  Jane = x * Han + 6 := 
by
  -- Proof goes here (not required)
  sorry

end rehabilitation_centers_l1782_178260


namespace necessary_and_sufficient_condition_l1782_178264

variables (x y : ℝ)

theorem necessary_and_sufficient_condition (h1 : x > y) (h2 : 1/x > 1/y) : x * y < 0 :=
sorry

end necessary_and_sufficient_condition_l1782_178264


namespace new_average_weight_l1782_178225

-- Statement only
theorem new_average_weight (avg_weight_29: ℝ) (weight_new_student: ℝ) (total_students: ℕ) 
  (h1: avg_weight_29 = 28) (h2: weight_new_student = 22) (h3: total_students = 29) : 
  (avg_weight_29 * total_students + weight_new_student) / (total_students + 1) = 27.8 :=
by
  -- declare local variables for simpler proof
  let total_weight := avg_weight_29 * total_students
  let new_total_weight := total_weight + weight_new_student
  let new_total_students := total_students + 1
  have t_weight : total_weight = 812 := by sorry
  have new_t_weight : new_total_weight = 834 := by sorry
  have n_total_students : new_total_students = 30 := by sorry
  exact sorry

end new_average_weight_l1782_178225


namespace divisibility_by_seven_l1782_178292

theorem divisibility_by_seven : (∃ k : ℤ, (-8)^2019 + (-8)^2018 = 7 * k) :=
sorry

end divisibility_by_seven_l1782_178292


namespace value_of_x_squared_plus_inverse_squared_l1782_178231

theorem value_of_x_squared_plus_inverse_squared (x : ℝ) (hx : x ≠ 0) (h : x^4 + (1 / x^4) = 2) : x^2 + (1 / x^2) = 2 :=
sorry

end value_of_x_squared_plus_inverse_squared_l1782_178231


namespace solve_inequality_l1782_178286

theorem solve_inequality (x : ℝ) : x^3 - 9*x^2 - 16*x > 0 ↔ (x < -1 ∨ x > 16) := by
  sorry

end solve_inequality_l1782_178286


namespace Portia_school_students_l1782_178261

theorem Portia_school_students:
  ∃ (P L : ℕ), P = 2 * L ∧ P + L = 3000 ∧ P = 2000 :=
by
  sorry

end Portia_school_students_l1782_178261


namespace complementary_angle_l1782_178277

-- Define the complementary angle condition
def complement (angle : ℚ) := 90 - angle

theorem complementary_angle : complement 30.467 = 59.533 :=
by
  -- Adding sorry to signify the missing proof to ensure Lean builds successfully
  sorry

end complementary_angle_l1782_178277


namespace max_min_sum_l1782_178282

noncomputable def f : ℝ → ℝ := sorry

-- Define the interval and properties of the function f
def within_interval (x : ℝ) : Prop := -2016 ≤ x ∧ x ≤ 2016
def functional_eq (x1 x2 : ℝ) : Prop := f (x1 + x2) = f x1 + f x2 - 2016
def less_than_2016_proof (x : ℝ) : Prop := x > 0 → f x < 2016

-- Define the minimum and maximum values of the function f
def M : ℝ := sorry
def N : ℝ := sorry

-- Prove that M + N = 4032 given the properties and conditions
theorem max_min_sum : 
  (∀ x1 x2, within_interval x1 → within_interval x2 → functional_eq x1 x2) →
  (∀ x, x > 0 → less_than_2016_proof x) →
  M + N = 4032 :=
by {
  -- Define the formal proof here, placeholder for actual proof
  sorry
}

end max_min_sum_l1782_178282


namespace min_shaded_triangles_l1782_178294

-- Definitions (conditions) directly from the problem
def Triangle (n : ℕ) := { x : ℕ // x ≤ n }
def side_length := 8
def smaller_side_length := 1

-- Goal (question == correct answer)
theorem min_shaded_triangles : ∃ (shaded : ℕ), shaded = 15 :=
by {
  sorry
}

end min_shaded_triangles_l1782_178294


namespace coffee_containers_used_l1782_178252

theorem coffee_containers_used :
  let Suki_coffee := 6.5 * 22
  let Jimmy_coffee := 4.5 * 18
  let combined_coffee := Suki_coffee + Jimmy_coffee
  let containers := combined_coffee / 8
  containers = 28 := 
by
  sorry

end coffee_containers_used_l1782_178252


namespace tan_150_deg_l1782_178280

theorem tan_150_deg : Real.tan (150 * Real.pi / 180) = - Real.sqrt 3 / 3 := by
  sorry

end tan_150_deg_l1782_178280


namespace simplify_expression_l1782_178281

theorem simplify_expression :
  (0.7264 * 0.4329 * 0.5478) + (0.1235 * 0.3412 * 0.6214) - ((0.1289 * 0.5634 * 0.3921) / (0.3785 * 0.4979 * 0.2884)) - (0.2956 * 0.3412 * 0.6573) = -0.3902 :=
by
  sorry

end simplify_expression_l1782_178281


namespace carter_has_255_cards_l1782_178219

-- Definition of the number of baseball cards Marcus has.
def marcus_cards : ℕ := 350

-- Definition of the number of more cards Marcus has than Carter.
def difference : ℕ := 95

-- Definition of the number of baseball cards Carter has.
def carter_cards : ℕ := marcus_cards - difference

-- Theorem stating that Carter has 255 baseball cards.
theorem carter_has_255_cards : carter_cards = 255 :=
sorry

end carter_has_255_cards_l1782_178219


namespace exists_plane_through_point_parallel_to_line_at_distance_l1782_178223

-- Definitions of the given entities
structure Point :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

structure Line :=
(point : Point)
(direction : Point) -- Considering direction as a point vector for simplicity

def distance (P : Point) (L : Line) : ℝ := 
  -- Define the distance from point P to line L
  sorry

noncomputable def construct_plane (P : Point) (L : Line) (d : ℝ) : Prop :=
  -- Define when a plane can be constructed as stated in the problem.
  sorry

-- The main proof problem statement without the solution steps
theorem exists_plane_through_point_parallel_to_line_at_distance (P : Point) (L : Line) (d : ℝ) (h : distance P L > d) :
  construct_plane P L d :=
sorry

end exists_plane_through_point_parallel_to_line_at_distance_l1782_178223


namespace smallest_integer_20p_larger_and_19p_smaller_l1782_178241

theorem smallest_integer_20p_larger_and_19p_smaller :
  ∃ (N x y : ℕ), N = 162 ∧ N = 12 / 10 * x ∧ N = 81 / 100 * y :=
by
  sorry

end smallest_integer_20p_larger_and_19p_smaller_l1782_178241


namespace distance_to_school_l1782_178218

variables (d : ℝ)
def jog_rate := 5
def bus_rate := 30
def total_time := 1 

theorem distance_to_school :
  (d / jog_rate) + (d / bus_rate) = total_time ↔ d = 30 / 7 :=
by
  sorry

end distance_to_school_l1782_178218


namespace problem1_problem2_l1782_178291

theorem problem1 : (Real.tan (10 * Real.pi / 180) - Real.sqrt 3) * Real.sin (40 * Real.pi / 180) = -1 := 
by
  sorry

theorem problem2 (x : ℝ) : 
  (2 * Real.cos x ^ 4 - 2 * Real.cos x ^ 2 + 1 / 2) /
  (2 * Real.tan (Real.pi / 4 - x) * Real.sin (Real.pi / 4 + x) ^ 2) = 
  Real.sin (2 * x) / 4 := 
by
  sorry

end problem1_problem2_l1782_178291


namespace time_after_6666_seconds_l1782_178248

noncomputable def initial_time : Nat := 3 * 3600
noncomputable def additional_seconds : Nat := 6666

-- Function to convert total seconds to "HH:MM:SS" format
def time_in_seconds (h m s : Nat) : Nat :=
  h*3600 + m*60 + s

noncomputable def new_time : Nat :=
  initial_time + additional_seconds

-- Convert the new total time back to "HH:MM:SS" format (expected: 4:51:06)
def hours (secs : Nat) : Nat := secs / 3600
def minutes (secs : Nat) : Nat := (secs % 3600) / 60
def seconds (secs : Nat) : Nat := (secs % 3600) % 60

theorem time_after_6666_seconds :
  hours new_time = 4 ∧ minutes new_time = 51 ∧ seconds new_time = 6 :=
by
  sorry

end time_after_6666_seconds_l1782_178248


namespace average_marks_correct_l1782_178201

def marks := [76, 65, 82, 62, 85]
def num_subjects := 5
def total_marks := marks.sum
def avg_marks := total_marks / num_subjects

theorem average_marks_correct : avg_marks = 74 :=
by sorry

end average_marks_correct_l1782_178201


namespace calculation_result_l1782_178275

theorem calculation_result : (1955 - 1875)^2 / 64 = 100 := by
  sorry

end calculation_result_l1782_178275


namespace rachel_homework_total_l1782_178272

-- Definitions based on conditions
def math_homework : Nat := 8
def biology_homework : Nat := 3

-- Theorem based on the problem statement
theorem rachel_homework_total : math_homework + biology_homework = 11 := by
  -- typically, here you would provide a proof, but we use sorry to skip it
  sorry

end rachel_homework_total_l1782_178272


namespace third_side_not_one_l1782_178203

theorem third_side_not_one (a b c : ℝ) (ha : a = 5) (hb : b = 7) (hc : c ≠ 1) :
  a + b > c ∧ a + c > b ∧ b + c > a :=
by
  sorry

end third_side_not_one_l1782_178203


namespace inequality_solution_l1782_178209

noncomputable def f (x : ℝ) : ℝ :=
  (2 / (x + 2)) + (4 / (x + 8))

theorem inequality_solution {x : ℝ} :
  f x ≥ 1/2 ↔ ((-8 < x ∧ x ≤ -4) ∨ (-2 ≤ x ∧ x ≤ 2)) :=
sorry

end inequality_solution_l1782_178209


namespace sufficient_condition_for_m_l1782_178283

variable (x m : ℝ)

def p (x : ℝ) : Prop := abs (x - 4) ≤ 6
def q (x m : ℝ) : Prop := x ≤ 1 + m

theorem sufficient_condition_for_m (h : ∀ x, p x → q x m ∧ ∃ x, ¬p x ∧ q x m) : m ≥ 9 :=
sorry

end sufficient_condition_for_m_l1782_178283


namespace dot_product_value_l1782_178227

def vector_a : ℝ × ℝ := (1, -2)
def vector_b : ℝ × ℝ := (3, 1)

theorem dot_product_value :
  vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2 = 1 :=
by
  -- Proof goes here
  sorry

end dot_product_value_l1782_178227


namespace next_chime_time_l1782_178229

theorem next_chime_time (chime1_interval : ℕ) (chime2_interval : ℕ) (chime3_interval : ℕ) (start_time : ℕ) 
  (h1 : chime1_interval = 18) (h2 : chime2_interval = 24) (h3 : chime3_interval = 30) (h4 : start_time = 9) : 
  ((start_time * 60 + 6 * 60) % (24 * 60)) / 60 = 15 :=
by
  sorry

end next_chime_time_l1782_178229


namespace remainder_when_divided_by_2_l1782_178211

theorem remainder_when_divided_by_2 (n : ℕ) (h₁ : n > 0) (h₂ : (n + 1) % 6 = 4) : n % 2 = 1 :=
by sorry

end remainder_when_divided_by_2_l1782_178211


namespace exists_increasing_infinite_sequence_of_perfect_squares_divisible_by_13_power_l1782_178200

open Nat

theorem exists_increasing_infinite_sequence_of_perfect_squares_divisible_by_13_power :
  ∃ (a : ℕ → ℕ), (∀ k : ℕ, (∃ b : ℕ, a k = b ^ 2)) ∧ (StrictMono a) ∧ (∀ k : ℕ, 13^k ∣ (a k + 1)) :=
sorry

end exists_increasing_infinite_sequence_of_perfect_squares_divisible_by_13_power_l1782_178200


namespace percent_increase_share_price_l1782_178202

theorem percent_increase_share_price (P : ℝ) 
  (h1 : ∃ P₁ : ℝ, P₁ = P + 0.25 * P)
  (h2 : ∃ P₂ : ℝ, P₂ = P + 0.80 * P)
  : ∃ percent_increase : ℝ, percent_increase = 44 := by
  sorry

end percent_increase_share_price_l1782_178202


namespace probability_phone_not_answered_l1782_178221

noncomputable def P_first_ring : ℝ := 0.1
noncomputable def P_second_ring : ℝ := 0.3
noncomputable def P_third_ring : ℝ := 0.4
noncomputable def P_fourth_ring : ℝ := 0.1

theorem probability_phone_not_answered : 
  1 - P_first_ring - P_second_ring - P_third_ring - P_fourth_ring = 0.1 := 
by
  sorry

end probability_phone_not_answered_l1782_178221


namespace new_apples_grew_l1782_178222

-- The number of apples originally on the tree.
def original_apples : ℕ := 11

-- The number of apples picked by Rachel.
def picked_apples : ℕ := 7

-- The number of apples currently on the tree.
def current_apples : ℕ := 6

-- The number of apples left on the tree after picking.
def remaining_apples : ℕ := original_apples - picked_apples

-- The number of new apples that grew on the tree.
def new_apples : ℕ := current_apples - remaining_apples

-- The theorem we need to prove.
theorem new_apples_grew :
  new_apples = 2 := by
    sorry

end new_apples_grew_l1782_178222


namespace multiply_fractions_l1782_178244

theorem multiply_fractions :
  (1 / 3) * (3 / 5) * (5 / 7) = (1 / 7) := by
  sorry

end multiply_fractions_l1782_178244


namespace simplify_and_evaluate_l1782_178274

theorem simplify_and_evaluate (x : ℝ) (h : x = 3 / 2) : 
  (2 + x) * (2 - x) + (x - 1) * (x + 5) = 5 := 
by
  sorry

end simplify_and_evaluate_l1782_178274


namespace fraction_reducible_to_17_l1782_178255

theorem fraction_reducible_to_17 (m n : ℕ) (h_coprime : Nat.gcd m n = 1)
  (h_reducible : ∃ d : ℕ, d ∣ (3 * m - n) ∧ d ∣ (5 * n + 2 * m)) :
  ∃ k : ℕ, (3 * m - n) / k = 17 ∧ (5 * n + 2 * m) / k = 17 :=
by
  have key : Nat.gcd (3 * m - n) (5 * n + 2 * m) = 17 := sorry
  -- using the result we need to construct our desired k
  use 17 / (Nat.gcd (3 * m - n) (5 * n + 2 * m))
  -- rest of intimate proof here
  sorry

end fraction_reducible_to_17_l1782_178255


namespace votes_cast_l1782_178295

theorem votes_cast (V : ℝ) (h1 : V = 0.33 * V + (0.33 * V + 833)) : V = 2447 := 
by
  sorry

end votes_cast_l1782_178295


namespace log_equality_ineq_l1782_178257

--let a = \log_{\sqrt{5x-1}}(4x+1)
--let b = \log_{4x+1}\left(\frac{x}{2} + 2\right)^2
--let c = \log_{\frac{x}{2} + 2}(5x-1)

noncomputable def a (x : ℝ) : ℝ := 
  Real.log (4 * x + 1) / Real.log (Real.sqrt (5 * x - 1))

noncomputable def b (x : ℝ) : ℝ := 
  2 * (Real.log ((x / 2) + 2) / Real.log (4 * x + 1))

noncomputable def c (x : ℝ) : ℝ := 
  Real.log (5 * x - 1) / Real.log ((x / 2) + 2)

theorem log_equality_ineq (x : ℝ) : 
  a x = b x ∧ c x = a x - 1 ↔ x = 2 := 
by
  sorry

end log_equality_ineq_l1782_178257


namespace find_x1_l1782_178273

theorem find_x1 (x1 x2 x3 x4 : ℝ) (h1 : 0 ≤ x4) (h2 : x4 ≤ x3) (h3 : x3 ≤ x2) (h4 : x2 ≤ x1) (h5 : x1 ≤ 1)
  (h6 : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + (x3 - x4)^2 + x4^2 = 1 / 5) : x1 = 4 / 5 := 
  sorry

end find_x1_l1782_178273


namespace solve_equation_1_solve_equation_2_solve_equation_3_l1782_178287

theorem solve_equation_1 : ∀ x : ℝ, (4 * (x + 3) = 25) ↔ (x = 13 / 4) :=
by
  sorry

theorem solve_equation_2 : ∀ x : ℝ, (5 * x^2 - 3 * x = x + 1) ↔ (x = -1 / 5 ∨ x = 1) :=
by
  sorry

theorem solve_equation_3 : ∀ x : ℝ, (2 * (x - 2)^2 - (x - 2) = 0) ↔ (x = 2 ∨ x = 5 / 2) :=
by
  sorry

end solve_equation_1_solve_equation_2_solve_equation_3_l1782_178287


namespace part_one_max_value_range_of_a_l1782_178205

def f (x a : ℝ) : ℝ := |x + 2| - |x - 3| - a

theorem part_one_max_value (a : ℝ) (h : a = 1) : ∃ x : ℝ, f x a = 4 := 
by sorry

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, f x a ≤ 4 / a) :  (0 < a ∧ a ≤ 1) ∨ 4 ≤ a :=
by sorry

end part_one_max_value_range_of_a_l1782_178205


namespace sweater_markup_percentage_l1782_178288

-- The wholesale cost W and retail price R
variables (W R : ℝ)

-- The given condition
variable (h : 0.30 * R = 1.40 * W)

-- The theorem to prove
theorem sweater_markup_percentage (h : 0.30 * R = 1.40 * W) : (R - W) / W * 100 = 366.67 :=
by
  -- The solution steps would be placed here, if we were proving.
  sorry

end sweater_markup_percentage_l1782_178288


namespace minimum_participants_l1782_178236

theorem minimum_participants
  (correct_first : ℕ)
  (correct_second : ℕ)
  (correct_third : ℕ)
  (correct_fourth : ℕ)
  (H_first : correct_first = 90)
  (H_second : correct_second = 50)
  (H_third : correct_third = 40)
  (H_fourth : correct_fourth = 20)
  (H_max_two : ∀ p : ℕ, 1 ≤ p ∧ p ≤ correct_first + correct_second + correct_third + correct_fourth → p ≤ 2 * (correct_first + correct_second + correct_third + correct_fourth))
  : ∃ n : ℕ, (correct_first + correct_second + correct_third + correct_fourth) / 2 = 100 :=
by
  sorry

end minimum_participants_l1782_178236


namespace inequality_proof_l1782_178289

variable {a b c d : ℝ}

theorem inequality_proof
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (hd : 0 < d)
  (h_sum : a + b + c + d = 3) :
  1 / a^3 + 1 / b^3 + 1 / c^3 + 1 / d^3 ≤ 1 / (a^3 * b^3 * c^3 * d^3) :=
sorry

end inequality_proof_l1782_178289


namespace range_of_a_when_min_f_ge_neg_a_l1782_178246

noncomputable def f (a x : ℝ) := a * Real.log x + 2 * x

theorem range_of_a_when_min_f_ge_neg_a (a : ℝ) (h₀ : a ≠ 0)
  (h₁ : ∀ x > 0, f a x ≥ -a) :
  -2 ≤ a ∧ a < 0 :=
sorry

end range_of_a_when_min_f_ge_neg_a_l1782_178246


namespace number_of_ways_difference_of_squares_l1782_178253

-- Lean statement
theorem number_of_ways_difference_of_squares (n k : ℕ) (h1 : n > 10^k) (h2 : n % 10^k = 0) (h3 : k ≥ 2) :
  ∃ D, D = k^2 - 1 ∧ ∀ (a b : ℕ), n = a^2 - b^2 → D = k^2 - 1 :=
by
  sorry

end number_of_ways_difference_of_squares_l1782_178253


namespace polynomial_expansion_sum_l1782_178204

theorem polynomial_expansion_sum :
  let A := 4
  let B := 10
  let C := 1
  let D := 21
  (x + 3) * (4 * x^2 - 2 * x + 7) = A * x^3 + B * x^2 + C * x + D →
  A + B + C + D = 36 := 
by
  -- Proof needs to be filled
  sorry

end polynomial_expansion_sum_l1782_178204


namespace matching_function_l1782_178217

open Real

def table_data : List (ℝ × ℝ) := [(1, 4), (2, 2), (4, 1)]

theorem matching_function :
  ∃ a b c : ℝ, a > 0 ∧ 
               (∀ x y, (x, y) ∈ table_data → y = a * x^2 + b * x + c) := 
sorry

end matching_function_l1782_178217


namespace pipe_filling_time_l1782_178290

/-- 
A problem involving two pipes filling and emptying a tank. 
Time taken for the first pipe to fill the tank is proven to be 16.8 minutes.
-/
theorem pipe_filling_time :
  ∃ T : ℝ, (∀ T, let r1 := 1 / T
                let r2 := 1 / 24
                let time_both_pipes_open := 36
                let time_first_pipe_only := 6
                (r1 - r2) * time_both_pipes_open + r1 * time_first_pipe_only = 1) ∧
           T = 16.8 :=
by
  sorry

end pipe_filling_time_l1782_178290


namespace unique_solution_l1782_178268

theorem unique_solution (n m : ℕ) (hn : 0 < n) (hm : 0 < m) : 
  n^2 = m^4 + m^3 + m^2 + m + 1 ↔ (n, m) = (11, 3) :=
by sorry

end unique_solution_l1782_178268


namespace find_P_l1782_178263

-- We start by defining the cubic polynomial
def cubic_eq (P : ℝ) (x : ℝ) := 5 * x^3 - 5 * (P + 1) * x^2 + (71 * P - 1) * x + 1

-- Define the condition that all roots are natural numbers
def has_three_natural_roots (P : ℝ) : Prop :=
  ∃ a b c : ℕ, 
    cubic_eq P a = 66 * P ∧ cubic_eq P b = 66 * P ∧ cubic_eq P c = 66 * P

-- Prove the value of P that satisfies the condition
theorem find_P : ∀ P : ℝ, has_three_natural_roots P → P = 76 := 
by
  -- We start the proof here
  sorry

end find_P_l1782_178263


namespace calculate_expression_l1782_178249

theorem calculate_expression : 2 * (-2) + (-3) = -7 := 
  sorry

end calculate_expression_l1782_178249


namespace numberOfBaseballBoxes_l1782_178284

-- Given conditions as Lean definitions and assumptions
def numberOfBasketballBoxes : ℕ := 4
def basketballCardsPerBox : ℕ := 10
def baseballCardsPerBox : ℕ := 8
def cardsGivenToClassmates : ℕ := 58
def cardsLeftAfterGiving : ℕ := 22

def totalBasketballCards : ℕ := numberOfBasketballBoxes * basketballCardsPerBox
def totalCardsBeforeGiving : ℕ := cardsLeftAfterGiving + cardsGivenToClassmates

-- Target number of baseball cards
def totalBaseballCards : ℕ := totalCardsBeforeGiving - totalBasketballCards

-- Prove that the number of baseball boxes is 5
theorem numberOfBaseballBoxes :
  totalBaseballCards / baseballCardsPerBox = 5 :=
sorry

end numberOfBaseballBoxes_l1782_178284


namespace John_scored_24point5_goals_l1782_178262

theorem John_scored_24point5_goals (T G : ℝ) (n : ℕ) (A : ℝ)
  (h1 : T = 65)
  (h2 : n = 9)
  (h3 : A = 4.5) :
  G = T - (n * A) :=
by
  sorry

end John_scored_24point5_goals_l1782_178262


namespace condition_sufficient_but_not_necessary_l1782_178251
noncomputable def sufficient_but_not_necessary (a b : ℝ) : Prop :=
∀ (a b : ℝ), a < 0 → -1 < b ∧ b < 0 → a + a * b < 0

-- Define the theorem stating the proof problem
theorem condition_sufficient_but_not_necessary (a b : ℝ) :
  (a < 0 ∧ -1 < b ∧ b < 0 → a + a * b < 0) ∧ 
  (a + a * b < 0 → a < 0 ∧ 1 + b > 0 ∨ a > 0 ∧ 1 + b < 0) :=
sorry

end condition_sufficient_but_not_necessary_l1782_178251


namespace intersect_three_points_l1782_178293

noncomputable def f (x : ℝ) : ℝ := (Real.cos x) / x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := Real.sin x + a * x

theorem intersect_three_points (a : ℝ) :
  (∃ (t1 t2 t3 : ℝ), t1 > 0 ∧ t2 > 0 ∧ t3 > 0 ∧ t1 ≠ t2 ∧ t2 ≠ t3 ∧ t1 ≠ t3 ∧ 
    f t1 = g t1 a ∧ f t2 = g t2 a ∧ f t3 = g t3 a) ↔ 
  a ∈ Set.Ioo (2 / (7 * Real.pi)) (2 / (3 * Real.pi)) ∨ a = -2 / (5 * Real.pi) :=
sorry

end intersect_three_points_l1782_178293


namespace A_alone_days_l1782_178240

variable (x : ℝ) -- Number of days A takes to do the work alone
variable (B_rate : ℝ := 1 / 12) -- Work rate of B
variable (Together_rate : ℝ := 1 / 4) -- Combined work rate of A and B

theorem A_alone_days :
  (1 / x + B_rate = Together_rate) → (x = 6) := by
  intro h
  sorry

end A_alone_days_l1782_178240


namespace min_people_like_mozart_bach_not_beethoven_l1782_178239

-- Define the initial conditions
variables {n a b c : ℕ}
variables (total_people := 150)
variables (likes_mozart := 120)
variables (likes_bach := 105)
variables (likes_beethoven := 45)

theorem min_people_like_mozart_bach_not_beethoven : 
  ∃ (x : ℕ), 
    total_people = 150 ∧ 
    likes_mozart = 120 ∧ 
    likes_bach = 105 ∧ 
    likes_beethoven = 45 ∧ 
    x = (likes_mozart + likes_bach - total_people) := 
    sorry

end min_people_like_mozart_bach_not_beethoven_l1782_178239


namespace hash_op_example_l1782_178258

def hash_op (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem hash_op_example : hash_op 2 5 3 = 1 := 
by 
  sorry

end hash_op_example_l1782_178258


namespace find_f_function_l1782_178226

def oddFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem find_f_function (f : ℝ → ℝ) (h_odd : oddFunction f) (h_pos : ∀ x, 0 < x → f x = x * (1 + x)) :
  ∀ x, x < 0 → f x = -x - x^2 :=
by
  sorry

end find_f_function_l1782_178226


namespace Tom_time_to_complete_wall_after_one_hour_l1782_178265

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

end Tom_time_to_complete_wall_after_one_hour_l1782_178265


namespace rectangle_shaded_area_fraction_l1782_178279

-- Defining necessary parameters and conditions
variables {R : Type} [LinearOrderedField R]

noncomputable def shaded_fraction (length width : R) : R :=
  let P : R × R := (0, width / 2)
  let Q : R × R := (length / 2, width)
  let rect_area := length * width
  let tri_area := (1 / 2) * (length / 2) * (width / 2)
  let shaded_area := rect_area - tri_area
  shaded_area / rect_area

-- The theorem stating our desired proof goal
theorem rectangle_shaded_area_fraction (length width : R) (h_length : 0 < length) (h_width : 0 < width) :
  shaded_fraction length width = 7 / 8 := by
  sorry

end rectangle_shaded_area_fraction_l1782_178279


namespace solve_for_S_l1782_178214

variable (D S : ℝ)
variable (h1 : D > 0)
variable (h2 : S > 0)
variable (h3 : ((0.75 * D) / 50 + (0.25 * D) / S) / D = 1 / 50)

theorem solve_for_S :
  S = 50 :=
by
  sorry

end solve_for_S_l1782_178214


namespace height_of_parallelogram_l1782_178247

theorem height_of_parallelogram
  (A B H : ℝ)
  (h1 : A = 480)
  (h2 : B = 32)
  (h3 : A = B * H) : 
  H = 15 := sorry

end height_of_parallelogram_l1782_178247


namespace three_digit_difference_l1782_178232

theorem three_digit_difference (x : ℕ) (a b c : ℕ)
  (h1 : a = x + 2)
  (h2 : b = x + 1)
  (h3 : c = x)
  (h4 : a > b)
  (h5 : b > c) :
  (100 * a + 10 * b + c) - (100 * c + 10 * b + a) = 198 :=
by
  sorry

end three_digit_difference_l1782_178232


namespace only_polyC_is_square_of_binomial_l1782_178297

-- Defining the polynomials
def polyA (m n : ℤ) : ℤ := (-m + n) * (m - n)
def polyB (a b : ℤ) : ℤ := (1/2 * a + b) * (b - 1/2 * a)
def polyC (x : ℤ) : ℤ := (x + 5) * (x + 5)
def polyD (a b : ℤ) : ℤ := (3 * a - 4 * b) * (3 * b + 4 * a)

-- Proving that only polyC fits the square of a binomial formula
theorem only_polyC_is_square_of_binomial (x : ℤ) :
  (polyC x) = (x + 5) * (x + 5) ∧
  (∀ m n : ℤ, polyA m n ≠ (m - n)^2) ∧
  (∀ a b : ℤ, polyB a b ≠ (1/2 * a + b)^2) ∧
  (∀ a b : ℤ, polyD a b ≠ (3 * a - 4 * b)^2) :=
by
  sorry

end only_polyC_is_square_of_binomial_l1782_178297
