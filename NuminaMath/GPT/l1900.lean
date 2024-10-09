import Mathlib

namespace coin_flip_probability_l1900_190057

theorem coin_flip_probability (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1)
  (h_win : ∑' n, (1 - p) ^ n * p ^ (n + 1) = 1 / 2) :
  p = (3 - Real.sqrt 5) / 2 :=
by
  sorry

end coin_flip_probability_l1900_190057


namespace ratio_of_square_sides_sum_l1900_190031

theorem ratio_of_square_sides_sum 
  (h : (75:ℚ) / 128 = (75:ℚ) / 128) :
  let a := 5
  let b := 6
  let c := 16
  a + b + c = 27 :=
by
  -- Our goal is to show that the sum of a + b + c equals 27
  let a := 5
  let b := 6
  let c := 16
  have h1 : a + b + c = 27 := sorry
  exact h1

end ratio_of_square_sides_sum_l1900_190031


namespace satellite_orbit_time_approx_l1900_190076

noncomputable def earth_radius_km : ℝ := 6371
noncomputable def satellite_speed_kmph : ℝ := 7000

theorem satellite_orbit_time_approx :
  let circumference := 2 * Real.pi * earth_radius_km 
  let time := circumference / satellite_speed_kmph 
  5.6 < time ∧ time < 5.8 :=
by
  sorry

end satellite_orbit_time_approx_l1900_190076


namespace root_properties_of_cubic_l1900_190034

theorem root_properties_of_cubic (z1 z2 : ℂ) (h1 : z1^2 + z1 + 1 = 0) (h2 : z2^2 + z2 + 1 = 0) :
  z1 * z2 = 1 ∧ z1^3 = 1 ∧ z2^3 = 1 :=
by
  -- Proof omitted
  sorry

end root_properties_of_cubic_l1900_190034


namespace solve_X_l1900_190080

def diamond (X Y : ℝ) : ℝ := 4 * X - 3 * Y + 2

theorem solve_X :
  (∃ X : ℝ, diamond X 6 = 35) ↔ (X = 51 / 4) := by
  sorry

end solve_X_l1900_190080


namespace acute_angle_alpha_range_l1900_190074

theorem acute_angle_alpha_range (x : ℝ) (α : ℝ) (h1 : 0 < x) (h2 : x < 90) (h3 : α = 180 - 2 * x) : 0 < α ∧ α < 180 :=
by
  sorry

end acute_angle_alpha_range_l1900_190074


namespace exponential_order_l1900_190071

theorem exponential_order (x y : ℝ) (a : ℝ) (hx : x > y) (hy : y > 1) (ha1 : 0 < a) (ha2 : a < 1) : a^x < a^y :=
sorry

end exponential_order_l1900_190071


namespace bagel_spending_l1900_190006

variable (B D : ℝ)

theorem bagel_spending (h1 : B - D = 12.50) (h2 : D = B * 0.75) : B + D = 87.50 := 
sorry

end bagel_spending_l1900_190006


namespace money_made_is_40_l1900_190065

-- Definitions based on conditions
def BettysStrawberries : ℕ := 16
def MatthewsStrawberries : ℕ := BettysStrawberries + 20
def NataliesStrawberries : ℕ := MatthewsStrawberries / 2
def TotalStrawberries : ℕ := BettysStrawberries + MatthewsStrawberries + NataliesStrawberries
def JarsOfJam : ℕ := TotalStrawberries / 7
def MoneyMade : ℕ := JarsOfJam * 4

-- The theorem to prove
theorem money_made_is_40 : MoneyMade = 40 :=
by
  sorry

end money_made_is_40_l1900_190065


namespace compute_expression_l1900_190018

theorem compute_expression :
    ( (2 / 3) * Real.sqrt 15 - Real.sqrt 20 ) / ( (1 / 3) * Real.sqrt 5 ) = 2 * Real.sqrt 3 - 6 :=
by
  sorry

end compute_expression_l1900_190018


namespace find_c_of_perpendicular_lines_l1900_190052

theorem find_c_of_perpendicular_lines (c : ℤ) :
  (∀ x y : ℤ, y = -3 * x + 4 → ∃ y' : ℤ, y' = (c * x + 18) / 9) →
  c = 3 :=
by
  sorry

end find_c_of_perpendicular_lines_l1900_190052


namespace final_cost_is_30_l1900_190069

-- Define conditions as constants
def cost_of_repair : ℝ := 7
def sales_tax : ℝ := 0.50
def number_of_tires : ℕ := 4

-- Define the cost for one tire repair
def cost_one_tire : ℝ := cost_of_repair + sales_tax

-- Define the cost for all tires
def total_cost : ℝ := cost_one_tire * number_of_tires

-- Theorem stating that the total cost is $30
theorem final_cost_is_30 : total_cost = 30 :=
by
  sorry

end final_cost_is_30_l1900_190069


namespace factor_t_sq_minus_64_l1900_190058

def isDifferenceOfSquares (a b : Int) : Prop := a = b^2

theorem factor_t_sq_minus_64 (t : Int) (h : isDifferenceOfSquares 64 8) : (t^2 - 64) = (t - 8) * (t + 8) := by
  sorry

end factor_t_sq_minus_64_l1900_190058


namespace negative_solution_condition_l1900_190042

variable {a b c x y : ℝ}

theorem negative_solution_condition (h1 : a * x + b * y = c)
    (h2 : b * x + c * y = a)
    (h3 : c * x + a * y = b)
    (hx : x < 0)
    (hy : y < 0) :
    a + b + c = 0 :=
sorry

end negative_solution_condition_l1900_190042


namespace mark_reading_time_l1900_190066

-- Definitions based on conditions
def daily_reading_hours : ℕ := 3
def days_in_week : ℕ := 7
def weekly_increase : ℕ := 6

-- Proof statement
theorem mark_reading_time : daily_reading_hours * days_in_week + weekly_increase = 27 := by
  -- placeholder for the proof
  sorry

end mark_reading_time_l1900_190066


namespace calc_x_l1900_190073

theorem calc_x : 484 + 2 * 22 * 7 + 49 = 841 := by
  sorry

end calc_x_l1900_190073


namespace evaluate_f_5_minus_f_neg_5_l1900_190051

def f (x : ℝ) : ℝ := x^4 + x^2 + 5 * x^3

theorem evaluate_f_5_minus_f_neg_5 : f 5 - f (-5) = 1250 :=
by 
  sorry

end evaluate_f_5_minus_f_neg_5_l1900_190051


namespace solve_x_l1900_190043

theorem solve_x (x : ℚ) : (∀ z : ℚ, 10 * x * z - 15 * z + 3 * x - 9 / 2 = 0) ↔ x = 3 / 2 := 
by
  sorry

end solve_x_l1900_190043


namespace reciprocal_neg3_l1900_190020

-- Define the problem
def reciprocal (x : ℚ) : ℚ := 1 / x

-- The required proof statement
theorem reciprocal_neg3 : reciprocal (-3) = -1 / 3 :=
by
  sorry

end reciprocal_neg3_l1900_190020


namespace probability_of_red_black_or_white_l1900_190090

def numberOfBalls := 12
def redBalls := 5
def blackBalls := 4
def whiteBalls := 2
def greenBalls := 1

def favorableOutcomes : Nat := redBalls + blackBalls + whiteBalls
def totalOutcomes : Nat := numberOfBalls

theorem probability_of_red_black_or_white :
  (favorableOutcomes : ℚ) / (totalOutcomes : ℚ) = 11 / 12 :=
by
  sorry

end probability_of_red_black_or_white_l1900_190090


namespace playground_area_l1900_190007

open Real

theorem playground_area (l w : ℝ) (h1 : 2*l + 2*w = 100) (h2 : l = 2*w) : l * w = 5000 / 9 :=
by
  sorry

end playground_area_l1900_190007


namespace problem_CorrectOption_l1900_190037

def setA : Set ℝ := {y | ∃ x : ℝ, y = |x| - 1}
def setB : Set ℝ := {x | x ≥ 2}

theorem problem_CorrectOption : setA ∩ setB = setB := 
  sorry

end problem_CorrectOption_l1900_190037


namespace sum_gcd_lcm_eq_180195_l1900_190040

def gcd_60_45045 := Nat.gcd 60 45045
def lcm_60_45045 := Nat.lcm 60 45045

theorem sum_gcd_lcm_eq_180195 : gcd_60_45045 + lcm_60_45045 = 180195 := by
  sorry

end sum_gcd_lcm_eq_180195_l1900_190040


namespace milk_water_mixture_initial_volume_l1900_190054

theorem milk_water_mixture_initial_volume
  (M W : ℝ)
  (h1 : 2 * M = 3 * W)
  (h2 : 4 * M = 3 * (W + 58)) :
  M + W = 145 := by
  sorry

end milk_water_mixture_initial_volume_l1900_190054


namespace least_b_not_in_range_l1900_190045

theorem least_b_not_in_range : ∃ b : ℤ, -10 = b ∧ ∀ x : ℝ, x^2 + b * x + 20 ≠ -10 :=
sorry

end least_b_not_in_range_l1900_190045


namespace not_possible_cut_l1900_190088

theorem not_possible_cut (n : ℕ) : 
  let chessboard_area := 8 * 8
  let rectangle_area := 3
  let rectangles_needed := chessboard_area / rectangle_area
  rectangles_needed ≠ n :=
by
  sorry

end not_possible_cut_l1900_190088


namespace prime_ge_7_divides_30_l1900_190081

theorem prime_ge_7_divides_30 (p : ℕ) (hp : p ≥ 7) (hp_prime : Nat.Prime p) : 30 ∣ (p^2 - 1) := by
  sorry

end prime_ge_7_divides_30_l1900_190081


namespace paint_cost_is_200_l1900_190000

-- Define the basic conditions and parameters
def side_length : ℕ := 5
def faces_of_cube : ℕ := 6
def area_per_face (side : ℕ) : ℕ := side * side
def total_surface_area (side : ℕ) (faces : ℕ) : ℕ := faces * area_per_face side
def coverage_per_kg : ℕ := 15
def cost_per_kg : ℕ := 20

-- Calculate total cost
def total_cost (side : ℕ) (faces : ℕ) (coverage : ℕ) (cost : ℕ) : ℕ :=
  let total_area := total_surface_area side faces
  let kgs_required := total_area / coverage
  kgs_required * cost

theorem paint_cost_is_200 :
  total_cost side_length faces_of_cube coverage_per_kg cost_per_kg = 200 :=
by
  sorry

end paint_cost_is_200_l1900_190000


namespace tan_ratio_l1900_190012

theorem tan_ratio (a b : ℝ)
  (h1 : Real.cos (a + b) = 1 / 3)
  (h2 : Real.cos (a - b) = 1 / 2) :
  (Real.tan a) / (Real.tan b) = 5 :=
sorry

end tan_ratio_l1900_190012


namespace vector_division_by_three_l1900_190075

def OA : ℝ × ℝ := (2, 8)
def OB : ℝ × ℝ := (-7, 2)
def vector_sub (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - v.1, u.2 - v.2)
noncomputable def scalar_mult (k : ℝ) (u : ℝ × ℝ) : ℝ × ℝ := (k * u.1, k * u.2)

theorem vector_division_by_three :
  scalar_mult (1 / 3) (vector_sub OB OA) = (-3, -2) :=
sorry

end vector_division_by_three_l1900_190075


namespace five_times_remaining_is_400_l1900_190089

-- Define the conditions
def original_marbles := 800
def marbles_per_friend := 120
def num_friends := 6

-- Calculate total marbles given away
def marbles_given_away := num_friends * marbles_per_friend

-- Calculate marbles remaining after giving away
def marbles_remaining := original_marbles - marbles_given_away

-- Question: what is five times the marbles remaining?
def five_times_remaining_marbles := 5 * marbles_remaining

-- The proof problem: prove that this equals 400
theorem five_times_remaining_is_400 : five_times_remaining_marbles = 400 :=
by
  -- The proof would go here
  sorry

end five_times_remaining_is_400_l1900_190089


namespace rhombus_area_2400_l1900_190004

noncomputable def area_of_rhombus (x y : ℝ) : ℝ :=
  2 * x * y

theorem rhombus_area_2400 (x y : ℝ) 
  (hx : x = 15) 
  (hy : y = (16 / 3) * x) 
  (rx : 18.75 * 4 * x * y = x * y * (78.75)) 
  (ry : 50 * 4 * x * y = x * y * (200)) : 
  area_of_rhombus 15 80 = 2400 :=
by
  sorry

end rhombus_area_2400_l1900_190004


namespace box_weights_l1900_190061

theorem box_weights (a b c : ℕ) (h1 : a + b = 132) (h2 : b + c = 135) (h3 : c + a = 137) (h4 : a > 40) (h5 : b > 40) (h6 : c > 40) : a + b + c = 202 :=
by 
  sorry

end box_weights_l1900_190061


namespace ramesh_installation_cost_l1900_190056

noncomputable def labelled_price (discounted_price : ℝ) (discount_rate : ℝ) : ℝ :=
  discounted_price / (1 - discount_rate)

noncomputable def selling_price (labelled_price : ℝ) (profit_rate : ℝ) : ℝ :=
  labelled_price * (1 + profit_rate)

def ramesh_total_cost (purchase_price transport_cost : ℝ) (installation_cost : ℝ) : ℝ :=
  purchase_price + transport_cost + installation_cost

theorem ramesh_installation_cost :
  ∀ (purchase_price discounted_price transport_cost labelled_price profit_rate selling_price installation_cost : ℝ),
  discounted_price = 12500 → transport_cost = 125 → profit_rate = 0.18 → selling_price = 18880 →
  labelled_price = discounted_price / (1 - 0.20) →
  selling_price = labelled_price * (1 + profit_rate) →
  ramesh_total_cost purchase_price transport_cost installation_cost = selling_price →
  installation_cost = 6255 :=
by
  intros
  sorry

end ramesh_installation_cost_l1900_190056


namespace Jenny_original_number_l1900_190021

theorem Jenny_original_number (y : ℝ) (h : 10 * (y / 2 - 6) = 70) : y = 26 :=
by
  sorry

end Jenny_original_number_l1900_190021


namespace water_depth_upright_l1900_190026

def tank_is_right_cylindrical := true
def tank_height := 18.0
def tank_diameter := 6.0
def tank_initial_position_is_flat := true
def water_depth_flat := 4.0

theorem water_depth_upright : water_depth_flat = 4.0 :=
by
  sorry

end water_depth_upright_l1900_190026


namespace cos_triple_angle_l1900_190028

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (3 * θ) = -117 / 125 := by
  sorry

end cos_triple_angle_l1900_190028


namespace factorize_expression_l1900_190087

theorem factorize_expression (a x : ℝ) : -a*x^2 + 2*a*x - a = -a*(x - 1)^2 := 
sorry

end factorize_expression_l1900_190087


namespace contrapositive_true_l1900_190055

theorem contrapositive_true (h : ∀ x : ℝ, x < 0 → x^2 > 0) : 
  (∀ x : ℝ, ¬ (x^2 > 0) → ¬ (x < 0)) :=
by 
  sorry

end contrapositive_true_l1900_190055


namespace shirt_cost_correct_l1900_190050

-- Define the conditions
def pants_cost : ℝ := 9.24
def bill_amount : ℝ := 20
def change_received : ℝ := 2.51

-- Calculate total spent and shirt cost
def total_spent : ℝ := bill_amount - change_received
def shirt_cost : ℝ := total_spent - pants_cost

-- The theorem statement
theorem shirt_cost_correct : shirt_cost = 8.25 := by
  sorry

end shirt_cost_correct_l1900_190050


namespace total_height_of_tower_l1900_190011

theorem total_height_of_tower :
  let S₃₅ : ℕ := (35 * (35 + 1)) / 2
  let S₆₅ : ℕ := (65 * (65 + 1)) / 2
  S₃₅ + S₆₅ = 2775 :=
by
  let S₃₅ := (35 * (35 + 1)) / 2
  let S₆₅ := (65 * (65 + 1)) / 2
  sorry

end total_height_of_tower_l1900_190011


namespace problem_statement_l1900_190077

noncomputable def perimeter_rectangle 
  (a b c w : ℝ) 
  (h1 : a^2 + b^2 = c^2) 
  (area_triangle : ℝ := (1/2) * a * b) 
  (area_rectangle : ℝ := area_triangle) 
  (l : ℝ := area_rectangle / w) : ℝ :=
2 * (w + l)

theorem problem_statement 
  (a b c w : ℝ) 
  (h1 : a^2 + b^2 = c^2) 
  (h_a : a = 9) 
  (h_b : b = 12) 
  (h_c : c = 15) 
  (h_w : w = 6) : 
  perimeter_rectangle a b c w h1 = 30 :=
by 
  sorry

end problem_statement_l1900_190077


namespace at_least_one_solves_l1900_190091

/--
Given probabilities p1, p2, p3 that individuals A, B, and C solve a problem respectively,
prove that the probability that at least one of them solves the problem is 
1 - (1 - p1) * (1 - p2) * (1 - p3).
-/
theorem at_least_one_solves (p1 p2 p3 : ℝ) (h1 : 0 ≤ p1 ∧ p1 ≤ 1) (h2 : 0 ≤ p2 ∧ p2 ≤ 1) (h3 : 0 ≤ p3 ∧ p3 ≤ 1) :
  1 - (1 - p1) * (1 - p2) * (1 - p3) = 1 - (1 - p1) * (1 - p2) * (1 - p3) :=
by
  sorry

end at_least_one_solves_l1900_190091


namespace inscribed_sphere_radius_in_regular_octahedron_l1900_190001

theorem inscribed_sphere_radius_in_regular_octahedron (a : ℝ) (r : ℝ) 
  (h1 : a = 6)
  (h2 : let V := 72 * Real.sqrt 2; V = (1 / 3) * ((8 * (3 * Real.sqrt 3)) * r)) : 
  r = Real.sqrt 6 :=
by
  sorry

end inscribed_sphere_radius_in_regular_octahedron_l1900_190001


namespace total_masks_correct_l1900_190099

-- Define the conditions
def boxes := 18
def capacity_per_box := 15
def deficiency_per_box := 3
def masks_per_box := capacity_per_box - deficiency_per_box
def total_masks := boxes * masks_per_box

-- The theorem statement we need to prove
theorem total_masks_correct : total_masks = 216 := by
  unfold total_masks boxes masks_per_box capacity_per_box deficiency_per_box
  sorry

end total_masks_correct_l1900_190099


namespace andy_solves_16_problems_l1900_190029

theorem andy_solves_16_problems :
  ∃ N : ℕ, 
    N = (125 - 78)/3 + 1 ∧
    (78 + (N - 1) * 3 <= 125) ∧
    N = 16 := 
by 
  sorry

end andy_solves_16_problems_l1900_190029


namespace find_number_l1900_190072

theorem find_number (x : ℝ) (h_Pos : x > 0) (h_Eq : x + 17 = 60 * (1/x)) : x = 3 :=
by
  sorry

end find_number_l1900_190072


namespace solve_for_x_over_z_l1900_190068

variables (x y z : ℝ)

theorem solve_for_x_over_z
  (h1 : x + y = 2 * x + z)
  (h2 : x - 2 * y = 4 * z)
  (h3 : x + y + z = 21)
  (h4 : y = 6 * z) :
  x / z = 5 :=
sorry

end solve_for_x_over_z_l1900_190068


namespace solve_simultaneous_equations_l1900_190094

theorem solve_simultaneous_equations :
  (∃ x y : ℝ, x^2 + 3 * y = 10 ∧ 3 + y = 10 / x) ↔ 
  (x = 3 ∧ y = 1 / 3) ∨ 
  (x = 2 ∧ y = 2) ∨ 
  (x = -5 ∧ y = -5) := by sorry

end solve_simultaneous_equations_l1900_190094


namespace sin_C_value_proof_a2_b2_fraction_proof_sides_sum_comparison_l1900_190070

variables (A B C a b c S : ℝ)
variables (h_area : S = (a + b) ^ 2 - c ^ 2) (h_sum : a + b = 4)
variables (h_triangle : ∀ (x : ℝ), x = sin C)

open Real

theorem sin_C_value_proof :
  sin C = 8 / 17 :=
sorry

theorem a2_b2_fraction_proof :
  (a ^ 2 - b ^ 2) / c ^ 2 = sin (A - B) / sin C :=
sorry

theorem sides_sum_comparison :
  a ^ 2 + b ^ 2 + c ^ 2 ≥ 4 * sqrt 3 * S :=
sorry

end sin_C_value_proof_a2_b2_fraction_proof_sides_sum_comparison_l1900_190070


namespace rate_of_grapes_l1900_190025

theorem rate_of_grapes (G : ℝ) 
  (h_grapes : 8 * G + 9 * 60 = 1100) : 
  G = 70 := 
by
  sorry

end rate_of_grapes_l1900_190025


namespace am_gm_inequality_l1900_190019

theorem am_gm_inequality
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a * b + b * c + c * a) :=
by
  sorry

end am_gm_inequality_l1900_190019


namespace min_value_frac_l1900_190053

theorem min_value_frac (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : 2 * m + n = 1) : 
  (1 / m + 2 / n) = 8 :=
sorry

end min_value_frac_l1900_190053


namespace nathalie_cake_fraction_l1900_190015

theorem nathalie_cake_fraction
    (cake_weight : ℕ)
    (pierre_ate : ℕ)
    (double_what_nathalie_ate : pierre_ate = 2 * (pierre_ate / 2))
    (pierre_ate_correct : pierre_ate = 100) :
    (pierre_ate / 2) / cake_weight = 1 / 8 :=
by
  sorry

end nathalie_cake_fraction_l1900_190015


namespace total_inflation_time_l1900_190027

/-- 
  Assume a soccer ball takes 20 minutes to inflate.
  Alexia inflates 20 soccer balls.
  Ermias inflates 5 more balls than Alexia.
  Prove that the total time in minutes taken to inflate all the balls is 900 minutes.
-/
theorem total_inflation_time 
  (alexia_balls : ℕ) (ermias_balls : ℕ) (each_ball_time : ℕ)
  (h1 : alexia_balls = 20)
  (h2 : ermias_balls = alexia_balls + 5)
  (h3 : each_ball_time = 20) :
  (alexia_balls + ermias_balls) * each_ball_time = 900 :=
by
  sorry

end total_inflation_time_l1900_190027


namespace num_integer_values_satisfying_condition_l1900_190038

theorem num_integer_values_satisfying_condition : 
  ∃ s : Finset ℤ, (∀ x ∈ s, 4 < Real.sqrt (3 * x) ∧ Real.sqrt (3 * x) < 5) ∧ s.card = 3 :=
by
  sorry

end num_integer_values_satisfying_condition_l1900_190038


namespace line_eq_slope_form_l1900_190083

theorem line_eq_slope_form (a b c : ℝ) (h : b ≠ 0) :
    ∃ k l : ℝ, ∀ x y : ℝ, (a * x + b * y + c = 0) ↔ (y = k * x + l) := 
sorry

end line_eq_slope_form_l1900_190083


namespace increase_by_50_percent_l1900_190079

def original : ℕ := 350
def increase_percent : ℕ := 50
def increased_number : ℕ := original * increase_percent / 100
def final_number : ℕ := original + increased_number

theorem increase_by_50_percent : final_number = 525 := 
by
  sorry

end increase_by_50_percent_l1900_190079


namespace marbles_left_l1900_190041

theorem marbles_left (red_marble_count blue_marble_count broken_marble_count : ℕ)
  (h1 : red_marble_count = 156)
  (h2 : blue_marble_count = 267)
  (h3 : broken_marble_count = 115) :
  red_marble_count + blue_marble_count - broken_marble_count = 308 :=
by
  sorry

end marbles_left_l1900_190041


namespace problem1_part1_problem1_part2_problem2_l1900_190084

-- Definitions
def quadratic (a b c x : ℝ) := a * x ^ 2 + b * x + c
def has_two_real_roots (a b c : ℝ) := b ^ 2 - 4 * a * c ≥ 0 
def neighboring_root_equation (a b c : ℝ) :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic a b c x₁ = 0 ∧ quadratic a b c x₂ = 0 ∧ |x₁ - x₂| = 1

-- Proof problem 1: Prove whether x^2 + x - 6 = 0 is a neighboring root equation
theorem problem1_part1 : ¬ neighboring_root_equation 1 1 (-6) := 
sorry

-- Proof problem 2: Prove whether 2x^2 - 2√5x + 2 = 0 is a neighboring root equation
theorem problem1_part2 : neighboring_root_equation 2 (-2 * Real.sqrt 5) 2 := 
sorry

-- Proof problem 3: Prove that m = -1 or m = -3 for x^2 - (m-2)x - 2m = 0 to be a neighboring root equation
theorem problem2 (m : ℝ) (h : neighboring_root_equation 1 (-(m-2)) (-2*m)) : 
  m = -1 ∨ m = -3 := 
sorry

end problem1_part1_problem1_part2_problem2_l1900_190084


namespace inequality_solution_set_ab2_bc_ca_a3b_ge_1_4_l1900_190024

theorem inequality_solution_set (x : ℝ) : (|x - 1| + |2 * x + 5| < 8) ↔ (-4 < x ∧ x < 4 / 3) :=
by
  sorry

theorem ab2_bc_ca_a3b_ge_1_4 (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) (h₃ : a + b + c = 1) :
  (a^2 / (b + 3 * c) + b^2 / (c + 3 * a) + c^2 / (a + 3 * b) ≥ 1 / 4) :=
by
  sorry

end inequality_solution_set_ab2_bc_ca_a3b_ge_1_4_l1900_190024


namespace words_per_page_l1900_190078

theorem words_per_page (p : ℕ) (h1 : 150 * p ≡ 270 [MOD 221]) (h2 : p ≤ 120) : p = 107 :=
sorry

end words_per_page_l1900_190078


namespace line_passes_through_fixed_point_l1900_190035

theorem line_passes_through_fixed_point 
  (a b : ℝ) 
  (h : 2 * a + b = 1) : 
  a * 4 + b * 2 = 2 :=
sorry

end line_passes_through_fixed_point_l1900_190035


namespace fred_baseball_cards_l1900_190010

theorem fred_baseball_cards :
  ∀ (fred_cards_initial melanie_bought : ℕ), fred_cards_initial = 5 → melanie_bought = 3 → fred_cards_initial - melanie_bought = 2 :=
by
  intros fred_cards_initial melanie_bought h1 h2
  sorry

end fred_baseball_cards_l1900_190010


namespace sum_of_digits_is_base_6_l1900_190022

def is_valid_digit (x : ℕ) : Prop := x > 0 ∧ x < 6 
def distinct_3 (a b c : ℕ) : Prop := a ≠ b ∧ b ≠ c ∧ c ≠ a  

theorem sum_of_digits_is_base_6 :
  ∃ (S H E : ℕ), is_valid_digit S ∧ is_valid_digit H ∧ is_valid_digit E
  ∧ distinct_3 S H E 
  ∧ (E + E) % 6 = S 
  ∧ (S + H) % 6 = E 
  ∧ (S + H + E) % 6 = 11 % 6 :=
by 
  sorry

end sum_of_digits_is_base_6_l1900_190022


namespace vacation_books_l1900_190093

-- Define the number of mystery, fantasy, and biography novels.
def num_mystery : ℕ := 3
def num_fantasy : ℕ := 4
def num_biography : ℕ := 3

-- Define the condition that we want to choose three books with no more than one from each genre.
def num_books_to_choose : ℕ := 3
def max_books_per_genre : ℕ := 1

-- The number of ways to choose one book from each genre
def num_combinations (m f b : ℕ) : ℕ :=
  m * f * b

-- Prove that the number of possible sets of books is 36
theorem vacation_books : num_combinations num_mystery num_fantasy num_biography = 36 := by
  sorry

end vacation_books_l1900_190093


namespace max_profit_l1900_190002

variables (x y : ℝ)

def profit (x y : ℝ) : ℝ := 50000 * x + 30000 * y

theorem max_profit :
  (3 * x + y ≤ 13) ∧ (2 * x + 3 * y ≤ 18) ∧ (x ≥ 0) ∧ (y ≥ 0) →
  (∃ x y, profit x y = 390000) :=
by
  sorry

end max_profit_l1900_190002


namespace intersection_complement_eq_l1900_190097

open Set

namespace MathProof

variable (U A B : Set ℕ)

theorem intersection_complement_eq :
  U = {1, 2, 3, 4, 5, 6, 7} →
  A = {3, 4, 5} →
  B = {1, 3, 6} →
  A ∩ (U \ B) = {4, 5} :=
by
  intros hU hA hB
  sorry

end MathProof

end intersection_complement_eq_l1900_190097


namespace divisible_by_factorial_l1900_190008

def f : ℕ → ℕ → ℕ
| 0, 0 => 1
| 0, _ => 0
| _, 0 => 0
| n + 1, k + 1 => (n + 1) * (f (n + 1) k + f n k)

theorem divisible_by_factorial (n k : ℕ) : n! ∣ f n k := by sorry

end divisible_by_factorial_l1900_190008


namespace pairs_with_green_shirts_l1900_190013

theorem pairs_with_green_shirts (red_shirts green_shirts total_pairs red_pairs : ℕ) 
    (h1 : red_shirts = 70) 
    (h2 : green_shirts = 58) 
    (h3 : total_pairs = 64) 
    (h4 : red_pairs = 34) 
    : (∃ green_pairs : ℕ, green_pairs = 28) := 
by 
    sorry

end pairs_with_green_shirts_l1900_190013


namespace longest_side_similar_triangle_l1900_190046

noncomputable def internal_angle (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem longest_side_similar_triangle (a b c A : ℝ) (h₁ : a = 4) (h₂ : b = 6) (h₃ : c = 7) (h₄ : A = 132) :
  let k := Real.sqrt (132 / internal_angle 4 6 7)
  7 * k = 73.5 :=
by
  sorry

end longest_side_similar_triangle_l1900_190046


namespace inequality_ac2_bc2_implies_a_b_l1900_190086

theorem inequality_ac2_bc2_implies_a_b (a b c : ℝ) (h : a * c^2 > b * c^2) : a > b :=
sorry

end inequality_ac2_bc2_implies_a_b_l1900_190086


namespace find_two_digit_number_l1900_190014

theorem find_two_digit_number : ∃ (x y : ℕ), 1 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 ∧ 10 * x + y = x^3 + y^2 ∧ 10 * x + y = 24 := by
  sorry

end find_two_digit_number_l1900_190014


namespace gift_spending_l1900_190017

def total_amount : ℝ := 700.00
def wrapping_expenses : ℝ := 139.00
def amount_spent_on_gifts : ℝ := 700.00 - 139.00

theorem gift_spending :
  (total_amount - wrapping_expenses) = 561.00 :=
by
  sorry

end gift_spending_l1900_190017


namespace factorial_division_l1900_190098

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem factorial_division (n : ℕ) (h : n > 0) : factorial (n) / factorial (n - 1) = n :=
by
  sorry

example : factorial 12 / factorial 11 = 12 :=
by
  exact factorial_division 12 (by norm_num)

end factorial_division_l1900_190098


namespace arithmetic_sequence_sum_l1900_190005

def f (x : ℝ) : ℝ := (x - 3) ^ 3 + x - 1

-- Define the arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d ≠ 0, ∀ n : ℕ, a (n + 1) = a n + d

-- Problem Statement
theorem arithmetic_sequence_sum 
  (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_f_sum : f (a 1) + f (a 2) + f (a 3) + f (a 4) + f (a 5) + f (a 6) + f (a 7) = 14) : 
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 21 :=
sorry

end arithmetic_sequence_sum_l1900_190005


namespace solve_for_x_l1900_190036

theorem solve_for_x : 
  ∀ (x : ℝ), (∀ (a b : ℝ), a * b = 4 * a - 2 * b) → (3 * (6 * x) = -2) → (x = 17 / 2) :=
by
  sorry

end solve_for_x_l1900_190036


namespace ratios_of_square_areas_l1900_190016

variable (x : ℝ)

def square_area (side_length : ℝ) : ℝ := side_length^2

theorem ratios_of_square_areas (hA : square_area x = x^2)
                               (hB : square_area (5 * x) = 25 * x^2)
                               (hC : square_area (2 * x) = 4 * x^2) :
  (square_area x / square_area (5 * x) = 1 / 25 ∧
   square_area (2 * x) / square_area (5 * x) = 4 / 25) := 
by {
  sorry
}

end ratios_of_square_areas_l1900_190016


namespace find_m_l1900_190003

def A : Set ℕ := {1, 3}
def B (m : ℕ) : Set ℕ := {1, 2, m}

theorem find_m (m : ℕ) (h : A ⊆ B m) : m = 3 :=
sorry

end find_m_l1900_190003


namespace gcd_a_b_l1900_190082

noncomputable def a : ℕ := 3333333
noncomputable def b : ℕ := 666666666

theorem gcd_a_b : Nat.gcd a b = 3 := by
  sorry

end gcd_a_b_l1900_190082


namespace solution_set_x_l1900_190048

theorem solution_set_x (x : ℝ) : 
  (|x^2 - x - 2| + |1 / x| = |x^2 - x - 2 + 1 / x|) ↔ 
  (x ∈ {y : ℝ | -1 ≤ y ∧ y < 0} ∨ x ≥ 2) :=
sorry

end solution_set_x_l1900_190048


namespace total_marbles_l1900_190064

theorem total_marbles (r b g y : ℝ)
  (h1 : r = 1.35 * b)
  (h2 : g = 1.5 * r)
  (h3 : y = 2 * b) :
  r + b + g + y = 4.72 * r :=
by
  sorry

end total_marbles_l1900_190064


namespace find_number_l1900_190044

theorem find_number (n : ℕ) (h : 582964 * n = 58293485180) : n = 100000 :=
by
  sorry

end find_number_l1900_190044


namespace largest_integer_dividing_sum_of_5_consecutive_integers_l1900_190059

theorem largest_integer_dividing_sum_of_5_consecutive_integers :
  ∀ (a : ℤ), ∃ (n : ℤ), n = 5 ∧ 5 ∣ ((a - 2) + (a - 1) + a + (a + 1) + (a + 2)) := by
  sorry

end largest_integer_dividing_sum_of_5_consecutive_integers_l1900_190059


namespace max_value_of_g_l1900_190023

noncomputable def g (x : ℝ) : ℝ := min (min (3 * x + 3) ((1 / 3) * x + 1)) (-2 / 3 * x + 8)

theorem max_value_of_g : ∃ x : ℝ, g x = 10 / 3 :=
by
  sorry

end max_value_of_g_l1900_190023


namespace exactly_one_divisible_by_5_l1900_190009

def a (n : ℕ) : ℕ := 2^(2*n + 1) - 2^(n + 1) + 1
def b (n : ℕ) : ℕ := 2^(2*n + 1) + 2^(n + 1) + 1

theorem exactly_one_divisible_by_5 (n : ℕ) (hn : 0 < n) : (a n % 5 = 0 ∧ b n % 5 ≠ 0) ∨ (a n % 5 ≠ 0 ∧ b n % 5 = 0) :=
  sorry

end exactly_one_divisible_by_5_l1900_190009


namespace periodicity_f_l1900_190085

noncomputable def vectorA (x : ℝ) : ℝ × ℝ := (2 * Real.sin x, Real.cos x)
noncomputable def vectorB (x : ℝ) : ℝ × ℝ := (Real.cos x, 2 * Real.sqrt 3 * Real.cos x)
noncomputable def f (x : ℝ) : ℝ :=
  let a := vectorA x
  let b := vectorB x
  a.1 * b.1 + a.2 * b.2

theorem periodicity_f :
  (∀ x, f (x + Real.pi) = f x) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), (f x = 2 + Real.sqrt 3 ∨ f x = 0)) :=
by
  sorry

end periodicity_f_l1900_190085


namespace minimum_reciprocal_sum_l1900_190033

noncomputable def minimum_value_of_reciprocal_sum (x y z : ℝ) : ℝ :=
  if x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 2 then 
    max (1/x + 1/y + 1/z) (9/2)
  else
    0
  
theorem minimum_reciprocal_sum (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 2): 
  1/x + 1/y + 1/z ≥ 9/2 :=
sorry

end minimum_reciprocal_sum_l1900_190033


namespace average_side_length_of_squares_l1900_190096

theorem average_side_length_of_squares :
  let a1 := 25
  let a2 := 64
  let a3 := 144
  let s1 := Real.sqrt a1
  let s2 := Real.sqrt a2
  let s3 := Real.sqrt a3
  (s1 + s2 + s3) / 3 = 25 / 3 := 
by
  sorry

end average_side_length_of_squares_l1900_190096


namespace find_original_one_digit_number_l1900_190067

theorem find_original_one_digit_number (x : ℕ) (h1 : x < 10) (h2 : (x + 10) * (x + 10) / x = 72) : x = 2 :=
sorry

end find_original_one_digit_number_l1900_190067


namespace square_area_from_diagonal_l1900_190049

theorem square_area_from_diagonal
  (d : ℝ) (h : d = 10) : ∃ (A : ℝ), A = 50 :=
by {
  -- here goes the proof
  sorry
}

end square_area_from_diagonal_l1900_190049


namespace proportion_calculation_l1900_190095

theorem proportion_calculation (x y : ℝ) (h1 : 0.75 / x = 5 / y) (h2 : x = 1.2) : y = 8 :=
by
  sorry

end proportion_calculation_l1900_190095


namespace man_swim_distance_downstream_l1900_190063

noncomputable def DistanceDownstream (Vm : ℝ) (Vupstream : ℝ) (time : ℝ) : ℝ :=
  let Vs := Vm - Vupstream
  let Vdownstream := Vm + Vs
  Vdownstream * time

theorem man_swim_distance_downstream :
  let Vm : ℝ := 3  -- speed of man in still water in km/h
  let time : ℝ := 6 -- time taken in hours
  let d_upstream : ℝ := 12 -- distance swum upstream in km
  let Vupstream : ℝ := d_upstream / time
  DistanceDownstream Vm Vupstream time = 24 := sorry

end man_swim_distance_downstream_l1900_190063


namespace area_of_region_B_l1900_190060

-- Given conditions
def region_B (z : ℂ) : Prop :=
  (0 ≤ (z.re / 50) ∧ (z.re / 50) ≤ 1 ∧ 0 ≤ (z.im / 50) ∧ (z.im / 50) ≤ 1)
  ∧
  (0 ≤ (50 * z.re / (z.re^2 + z.im^2)) ∧ (50 * z.re / (z.re^2 + z.im^2)) ≤ 1 ∧ 
  0 ≤ (50 * z.im / (z.re^2 + z.im^2)) ∧ (50 * z.im / (z.re^2 + z.im^2)) ≤ 1)

-- Theorem to be proved
theorem area_of_region_B : 
  (∫ z in {z : ℂ | region_B z}, 1) = 1875 - 312.5 * Real.pi :=
by
  sorry

end area_of_region_B_l1900_190060


namespace graph_avoid_third_quadrant_l1900_190030

theorem graph_avoid_third_quadrant (k : ℝ) : 
  (∀ x y : ℝ, y = (2 * k - 1) * x + k → ¬ (x < 0 ∧ y < 0)) ↔ 0 ≤ k ∧ k < (1 / 2) :=
by sorry

end graph_avoid_third_quadrant_l1900_190030


namespace infinitely_many_arithmetic_sequences_l1900_190032

theorem infinitely_many_arithmetic_sequences (x : ℕ) (hx : 0 < x) :
  ∃ y z : ℕ, y = 5 * x + 2 ∧ z = 7 * x + 3 ∧ x * (x + 1) < y * (y + 1) ∧ y * (y + 1) < z * (z + 1) ∧
  y * (y + 1) - x * (x + 1) = z * (z + 1) - y * (y + 1) :=
by
  sorry

end infinitely_many_arithmetic_sequences_l1900_190032


namespace num_pure_Gala_trees_l1900_190092

-- Define the problem statement conditions
variables (T F G H : ℝ)
variables (c1 : 0.125 * F + 0.075 * F + F = 315)
variables (c2 : F = (2 / 3) * T)
variables (c3 : H = (1 / 6) * T)
variables (c4 : T = F + G + H)

-- Prove the number of pure Gala trees G is 66
theorem num_pure_Gala_trees : G = 66 :=
by
  -- Proof will be filled out here
  sorry

end num_pure_Gala_trees_l1900_190092


namespace sum_of_possible_values_l1900_190039

theorem sum_of_possible_values (x : ℝ) (h : |x - 5| + 2 = 4) :
  x = 7 ∨ x = 3 → x = 10 := 
by sorry

end sum_of_possible_values_l1900_190039


namespace g_of_2_eq_14_l1900_190062

theorem g_of_2_eq_14 (g : ℝ → ℝ) (h : ∀ x : ℝ, g (3 * x - 4) = 4 * x + 6) : g 2 = 14 := 
sorry

end g_of_2_eq_14_l1900_190062


namespace find_ratio_l1900_190047

variables {EF GH EH EG EQ ER ES Q R S : ℝ}
variables (x : ℝ)
variables (E F G H : ℝ)

-- Conditions
def is_parallelogram : Prop := 
  -- Placeholder for parallelogram properties, not relevant for this example
  true

def point_on_segment (Q R : ℝ) (segment_length: ℝ) (ratio: ℝ): Prop := Q = segment_length * ratio ∧ R = segment_length * ratio

def intersect (EG QR : ℝ) (S : ℝ): Prop := 
  -- Placeholder for segment intersection properties, not relevant for this example
  true

-- Question
theorem find_ratio 
  (H_parallelogram: is_parallelogram)
  (H_pointQ: point_on_segment EQ ER EF (1/8))
  (H_pointR: point_on_segment ER ES EH (1/9))
  (H_intersection: intersect EG QR ES):
  (ES / EG) = (1/9) := 
by
  sorry

end find_ratio_l1900_190047
