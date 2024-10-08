import Mathlib

namespace chef_michel_total_pies_l90_90880

theorem chef_michel_total_pies
  (shepherd_pie_pieces : ℕ)
  (chicken_pot_pie_pieces : ℕ)
  (shepherd_pie_customers : ℕ)
  (chicken_pot_pie_customers : ℕ)
  (H1 : shepherd_pie_pieces = 4)
  (H2 : chicken_pot_pie_pieces = 5)
  (H3 : shepherd_pie_customers = 52)
  (H4 : chicken_pot_pie_customers = 80) :
  (shepherd_pie_customers / shepherd_pie_pieces) + (chicken_pot_pie_customers / chicken_pot_pie_pieces) = 29 :=
by
  sorry

end chef_michel_total_pies_l90_90880


namespace inequality_holds_for_all_real_numbers_l90_90535

theorem inequality_holds_for_all_real_numbers (x : ℝ) : 3 * x - 5 ≤ 12 - 2 * x + x^2 :=
by sorry

end inequality_holds_for_all_real_numbers_l90_90535


namespace man_is_older_by_20_l90_90515

variables (M S : ℕ)
axiom h1 : S = 18
axiom h2 : M + 2 = 2 * (S + 2)

theorem man_is_older_by_20 :
  M - S = 20 :=
by {
  sorry
}

end man_is_older_by_20_l90_90515


namespace square_non_negative_is_universal_l90_90819

/-- The square of any real number is non-negative, which is a universal proposition. -/
theorem square_non_negative_is_universal : 
  ∀ x : ℝ, x^2 ≥ 0 :=
by
  sorry

end square_non_negative_is_universal_l90_90819


namespace jenna_more_than_four_times_martha_l90_90676

noncomputable def problems : ℝ := 20
noncomputable def martha_problems : ℝ := 2
noncomputable def angela_problems : ℝ := 9
noncomputable def jenna_problems : ℝ := 6  -- We calculated J = 6 from the conditions
noncomputable def mark_problems : ℝ := jenna_problems / 2

theorem jenna_more_than_four_times_martha :
  (jenna_problems - 4 * martha_problems = 2) :=
by
  sorry

end jenna_more_than_four_times_martha_l90_90676


namespace inequality_proof_l90_90824

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (b+c-a)^2 / ((b+c)^2+a^2) + (c+a-b)^2 / ((c+a)^2+b^2) + (a+b-c)^2 / ((a+b)^2+c^2) ≥ 3 / 5 :=
by sorry

end inequality_proof_l90_90824


namespace correct_calculation_l90_90675

theorem correct_calculation (a b : ℝ) : 
  ¬(a * a^3 = a^3) ∧ ¬((a^2)^3 = a^5) ∧ (-a^2 * b)^2 = a^4 * b^2 ∧ ¬(a^3 / a = a^3) :=
by {
  sorry
}

end correct_calculation_l90_90675


namespace equiv_proof_problem_l90_90357

theorem equiv_proof_problem (b c : ℝ) (h1 : b ≠ 1 ∨ c ≠ 1) (h2 : ∃ n : ℝ, b = 1 + n ∧ c = 1 + 2 * n) (h3 : b * 1 = c * c) : 
  100 * (b - c) = 75 := 
by sorry

end equiv_proof_problem_l90_90357


namespace candy_problem_l90_90992

-- Definitions of conditions
def total_candies : ℕ := 91
def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

-- The minimum number of kinds of candies function
def min_kinds : ℕ := 46

-- Lean statement for the proof problem
theorem candy_problem : 
  ∀ (kinds : ℕ), 
    (∀ c : ℕ, c < total_candies → c % kinds < 2) → (∃ n : ℕ, kinds = min_kinds) := 
sorry

end candy_problem_l90_90992


namespace value_of_x_l90_90381

theorem value_of_x (x : ℝ) (h : x = 88 * 1.2) : x = 105.6 :=
by
  sorry

end value_of_x_l90_90381


namespace unique_solution_iff_a_eq_2019_l90_90335

theorem unique_solution_iff_a_eq_2019 (x a : ℝ) :
  (∃! x : ℝ, x - 1000 ≥ 1018 ∧ x + 1 ≤ a) ↔ a = 2019 :=
by
  sorry

end unique_solution_iff_a_eq_2019_l90_90335


namespace q1_correct_q2_correct_l90_90236

-- Defining the necessary operations
def q1_lhs := 8 / (-2) - (-4) * (-3)
def q2_lhs := (-2) ^ 3 / 4 * (5 - (-3) ^ 2)

-- Theorem statements to prove that they are equal to 8
theorem q1_correct : q1_lhs = 8 := sorry
theorem q2_correct : q2_lhs = 8 := sorry

end q1_correct_q2_correct_l90_90236


namespace rainfall_difference_l90_90613

-- Defining the conditions
def march_rainfall : ℝ := 0.81
def april_rainfall : ℝ := 0.46

-- Stating the theorem
theorem rainfall_difference : march_rainfall - april_rainfall = 0.35 := by
  -- insert proof steps here
  sorry

end rainfall_difference_l90_90613


namespace rectangular_prism_surface_area_l90_90399

/-- The surface area of a rectangular prism with edge lengths 2, 3, and 4 is 52. -/
theorem rectangular_prism_surface_area :
  let a := 2
  let b := 3
  let c := 4
  2 * (a * b + a * c + b * c) = 52 :=
by
  let a := 2
  let b := 3
  let c := 4
  show 2 * (a * b + a * c + b * c) = 52
  sorry

end rectangular_prism_surface_area_l90_90399


namespace find_large_number_l90_90271

theorem find_large_number (L S : ℕ) (h1 : L - S = 1311) (h2 : L = 11 * S + 11) : L = 1441 :=
sorry

end find_large_number_l90_90271


namespace rajan_income_l90_90926

theorem rajan_income (x y : ℝ) 
  (h₁ : 7 * x - 6 * y = 1000) 
  (h₂ : 6 * x - 5 * y = 1000) : 
  7 * x = 7000 :=
by 
  sorry

end rajan_income_l90_90926


namespace barbara_shopping_l90_90576

theorem barbara_shopping :
  let total_paid := 56
  let tuna_cost := 5 * 2
  let water_cost := 4 * 1.5
  let other_goods_cost := total_paid - tuna_cost - water_cost
  other_goods_cost = 40 :=
by
  sorry

end barbara_shopping_l90_90576


namespace profit_calculation_l90_90631

theorem profit_calculation (investment_john investment_mike profit_john profit_mike: ℕ) 
  (total_profit profit_shared_ratio profit_remaining_profit: ℚ)
  (h_investment_john : investment_john = 700)
  (h_investment_mike : investment_mike = 300)
  (h_total_profit : total_profit = 3000)
  (h_shared_ratio : profit_shared_ratio = total_profit / 3 / 2)
  (h_remaining_profit : profit_remaining_profit = 2 * total_profit / 3)
  (h_profit_john : profit_john = profit_shared_ratio + (7 / 10) * profit_remaining_profit)
  (h_profit_mike : profit_mike = profit_shared_ratio + (3 / 10) * profit_remaining_profit)
  (h_profit_difference : profit_john = profit_mike + 800) :
  total_profit = 3000 := 
by
  sorry

end profit_calculation_l90_90631


namespace parabola_constants_sum_l90_90233

-- Definition based on the given conditions
structure Parabola where
  a: ℝ
  b: ℝ
  c: ℝ
  vertex_x: ℝ
  vertex_y: ℝ
  point_x: ℝ
  point_y: ℝ

-- Definitions of the specific parabola based on the problem's conditions
noncomputable def givenParabola : Parabola := {
  a := -1/4,
  b := -5/2,
  c := -1/4,
  vertex_x := 6,
  vertex_y := -5,
  point_x := 2,
  point_y := -1
}

-- Theorem proving the required value of a + b + c
theorem parabola_constants_sum : givenParabola.a + givenParabola.b + givenParabola.c = -3.25 :=
  by
  sorry

end parabola_constants_sum_l90_90233


namespace babylon_game_proof_l90_90597

section BabylonGame

-- Defining the number of holes on the sphere
def number_of_holes : Nat := 26

-- The number of 45° angles formed by the pairs of rays
def num_45_degree_angles : Nat := 40

-- The number of 60° angles formed by the pairs of rays
def num_60_degree_angles : Nat := 48

-- The other angles that can occur between pairs of rays
def other_angles : List Real := [31.4, 81.6, 90]

-- Constructs possible given the conditions
def constructible (shape : String) : Bool :=
  shape = "regular tetrahedron" ∨ shape = "regular octahedron"

-- Constructs not possible given the conditions
def non_constructible (shape : String) : Bool :=
  shape = "joined regular tetrahedrons"

-- Proof problem statement
theorem babylon_game_proof :
  (number_of_holes = 26) →
  (num_45_degree_angles = 40) →
  (num_60_degree_angles = 48) →
  (other_angles = [31.4, 81.6, 90]) →
  (constructible "regular tetrahedron" = True) →
  (constructible "regular octahedron" = True) →
  (non_constructible "joined regular tetrahedrons" = True) :=
  by
    sorry

end BabylonGame

end babylon_game_proof_l90_90597


namespace tom_total_spent_on_video_games_l90_90828

-- Conditions
def batman_game_cost : ℝ := 13.6
def superman_game_cost : ℝ := 5.06

-- Statement to be proved
theorem tom_total_spent_on_video_games : batman_game_cost + superman_game_cost = 18.66 := by
  sorry

end tom_total_spent_on_video_games_l90_90828


namespace barbara_weekly_allowance_l90_90712

theorem barbara_weekly_allowance (W C S : ℕ) (H : W = 100) (A : S = 20) (N : C = 16) :
  (W - S) / C = 5 :=
by
  -- definitions to match conditions
  have W_def : W = 100 := H
  have S_def : S = 20 := A
  have C_def : C = 16 := N
  sorry

end barbara_weekly_allowance_l90_90712


namespace evaluate_expression_l90_90408

-- Defining the conditions and constants as per the problem statement
def factor_power_of_2 (n : ℕ) : ℕ :=
  if n % 8 = 0 then 3 else 0 -- Greatest power of 2 in 360
  
def factor_power_of_5 (n : ℕ) : ℕ :=
  if n % 5 = 0 then 1 else 0 -- Greatest power of 5 in 360

def expression (b a : ℕ) : ℚ := (2 / 3)^(b - a)

noncomputable def target_value : ℚ := 9 / 4

theorem evaluate_expression : expression (factor_power_of_5 360) (factor_power_of_2 360) = target_value := 
  by
    sorry

end evaluate_expression_l90_90408


namespace describe_graph_l90_90176

theorem describe_graph : 
  ∀ (x y : ℝ), x^2 * (x + y + 1) = y^3 * (x + y + 1) ↔ (x^2 = y^3 ∨ y = -x - 1)
:= sorry

end describe_graph_l90_90176


namespace even_function_zero_coefficient_l90_90447

theorem even_function_zero_coefficient: ∀ a : ℝ, (∀ x : ℝ, (x^2 + a * x + 1) = ((-x)^2 + a * (-x) + 1)) → a = 0 :=
by
  intros a h
  sorry

end even_function_zero_coefficient_l90_90447


namespace leap_year_53_sundays_and_february_5_sundays_l90_90818

theorem leap_year_53_sundays_and_february_5_sundays :
  let Y := 366
  let W := 52
  ∃ (p : ℚ), p = (2/7) * (1/7) → p = 2/49
:=
by
  sorry

end leap_year_53_sundays_and_february_5_sundays_l90_90818


namespace ratio_y_x_l90_90413

variable {c x y : ℝ}

-- Conditions stated as assumptions
theorem ratio_y_x (h1 : x = 0.80 * c) (h2 : y = 1.25 * c) : y / x = 25 / 16 :=
by
  sorry

end ratio_y_x_l90_90413


namespace total_weight_of_remaining_eggs_is_correct_l90_90420

-- Define the initial conditions and the question as Lean definitions
def total_eggs : Nat := 12
def weight_per_egg : Nat := 10
def num_boxes : Nat := 4
def melted_boxes : Nat := 1

-- Calculate the total weight of the eggs
def total_weight : Nat := total_eggs * weight_per_egg

-- Calculate the number of eggs per box
def eggs_per_box : Nat := total_eggs / num_boxes

-- Calculate the weight per box
def weight_per_box : Nat := eggs_per_box * weight_per_egg

-- Calculate the number of remaining boxes after one is tossed out
def remaining_boxes : Nat := num_boxes - melted_boxes

-- Calculate the total weight of the remaining chocolate eggs
def remaining_weight : Nat := remaining_boxes * weight_per_box

-- The proof task
theorem total_weight_of_remaining_eggs_is_correct : remaining_weight = 90 := by
  sorry

end total_weight_of_remaining_eggs_is_correct_l90_90420


namespace find_a_l90_90931

theorem find_a (a : ℝ) (hne : a ≠ 1) (eq_sets : ∀ x : ℝ, (a-1) * x < a + 5 ↔ 2 * x < 4) : a = 7 :=
sorry

end find_a_l90_90931


namespace speed_of_second_car_l90_90888

theorem speed_of_second_car
  (t : ℝ) (d : ℝ) (d1 : ℝ) (d2 : ℝ) (v : ℝ)
  (h1 : t = 2.5)
  (h2 : d = 175)
  (h3 : d1 = 25 * t)
  (h4 : d2 = v * t)
  (h5 : d1 + d2 = d) :
  v = 45 := by sorry

end speed_of_second_car_l90_90888


namespace Pete_latest_time_to_LA_l90_90359

def minutesInHour := 60
def minutesOfWalk := 10
def minutesOfTrain := 80
def departureTime := 7 * minutesInHour + 30

def latestArrivalTime : Prop :=
  9 * minutesInHour = departureTime + minutesOfWalk + minutesOfTrain 

theorem Pete_latest_time_to_LA : latestArrivalTime :=
by
  sorry

end Pete_latest_time_to_LA_l90_90359


namespace coloring_triangles_l90_90591

theorem coloring_triangles (n : ℕ) (k : ℕ) (h_n : n = 18) (h_k : k = 6) :
  (Nat.choose n k) = 18564 :=
by
  rw [h_n, h_k]
  sorry

end coloring_triangles_l90_90591


namespace right_triangle_345_l90_90873

def is_right_triangle (a b c : ℕ) : Prop :=
  a ^ 2 + b ^ 2 = c ^ 2

theorem right_triangle_345 : is_right_triangle 3 4 5 :=
by
  sorry

end right_triangle_345_l90_90873


namespace chosen_number_l90_90148

theorem chosen_number (x : ℕ) (h : 5 * x - 138 = 102) : x = 48 :=
sorry

end chosen_number_l90_90148


namespace rectangular_prism_diagonals_l90_90159

theorem rectangular_prism_diagonals :
  let l := 3
  let w := 4
  let h := 5
  let face_diagonals := 6 * 2
  let space_diagonals := 4
  face_diagonals + space_diagonals = 16 := 
by
  sorry

end rectangular_prism_diagonals_l90_90159


namespace find_a_l90_90221

noncomputable def point1 : ℝ × ℝ := (-3, 6)
noncomputable def point2 : ℝ × ℝ := (2, -1)

theorem find_a (a : ℝ) :
  let direction : ℝ × ℝ := (point2.1 - point1.1, point2.2 - point1.2)
  direction = (5, -7) →
  let normalized_direction : ℝ × ℝ := (direction.1 / -7, direction.2 / -7)
  normalized_direction = (a, -1) →
  a = -5 / 7 :=
by 
  intros 
  sorry

end find_a_l90_90221


namespace max_value_of_expression_l90_90940

open Real

theorem max_value_of_expression (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 1) : 
  2 * x * y + y * z + 2 * z * x ≤ 4 / 7 := 
sorry

end max_value_of_expression_l90_90940


namespace paint_weight_correct_l90_90336

def weight_of_paint (total_weight : ℕ) (half_paint_weight : ℕ) : ℕ :=
  2 * (total_weight - half_paint_weight)

theorem paint_weight_correct :
  weight_of_paint 24 14 = 20 := by 
  sorry

end paint_weight_correct_l90_90336


namespace hyperbola_asymptotes_l90_90707

theorem hyperbola_asymptotes :
  ∀ (x y : ℝ), (x^2 - (y^2 / 9) = 1) → (y = 3 * x ∨ y = -3 * x) :=
by
  -- conditions and theorem to prove
  sorry

end hyperbola_asymptotes_l90_90707


namespace father_age_l90_90164

theorem father_age (F D : ℕ) (h1 : F = 4 * D) (h2 : (F + 5) + (D + 5) = 50) : F = 32 :=
by
  sorry

end father_age_l90_90164


namespace infinite_double_perfect_squares_l90_90162

-- Definition of a double number
def is_double_number (n : ℕ) : Prop :=
  ∃ (k : ℕ), ∃ (d : ℕ), d ≠ 0 ∧ 10^k * d + d = n ∧ 10^k ≤ d ∧ d < 10^(k+1)

-- The theorem statement
theorem infinite_double_perfect_squares :
  ∃ (S : Set ℕ), (∀ n ∈ S, is_double_number n ∧ ∃ m, m * m = n) ∧
  Set.Infinite S :=
sorry

end infinite_double_perfect_squares_l90_90162


namespace family_gathering_total_people_l90_90483

theorem family_gathering_total_people (P : ℕ) 
  (h1 : P / 2 = 10) : 
  P = 20 := by
  sorry

end family_gathering_total_people_l90_90483


namespace solve_phi_l90_90936

noncomputable def find_phi (phi : ℝ) : Prop :=
  2 * Real.cos phi - Real.sin phi = Real.sqrt 3 * Real.sin (20 / 180 * Real.pi)

theorem solve_phi (phi : ℝ) :
  find_phi phi ↔ (phi = 140 / 180 * Real.pi ∨ phi = 40 / 180 * Real.pi) :=
sorry

end solve_phi_l90_90936


namespace original_time_taken_by_bullet_train_is_50_minutes_l90_90266

-- Define conditions as assumptions
variables (T D : ℝ) (h0 : D = 48 * T) (h1 : D = 60 * (40 / 60))

-- Define the theorem we want to prove
theorem original_time_taken_by_bullet_train_is_50_minutes :
  T = 50 / 60 :=
by
  sorry

end original_time_taken_by_bullet_train_is_50_minutes_l90_90266


namespace prime_divides_expression_l90_90796

theorem prime_divides_expression (p : ℕ) (hp : p > 5 ∧ Prime p) : 
  ∃ n : ℕ, p ∣ (20^n + 15^n - 12^n) := 
  by
  use (p - 3)
  sorry

end prime_divides_expression_l90_90796


namespace central_angle_l90_90175

-- Definition: percentage corresponds to central angle
def percentage_equal_ratio (P : ℝ) (θ : ℝ) : Prop :=
  P = θ / 360

-- Theorem statement: Given that P = θ / 360, we want to prove θ = 360 * P
theorem central_angle (P θ : ℝ) (h : percentage_equal_ratio P θ) : θ = 360 * P :=
sorry

end central_angle_l90_90175


namespace luna_budget_l90_90459

variable {H F P : ℝ}

theorem luna_budget (h1: F = 0.60 * H) (h2: P = 0.10 * F) (h3: H + F + P = 249) :
  H + F = 240 :=
by
  -- The proof will be filled in here. For now, we use sorry.
  sorry

end luna_budget_l90_90459


namespace simplify_expr1_simplify_expr2_l90_90723

variable {a b : ℝ} -- Assume a and b are arbitrary real numbers

-- Part 1: Prove that 2a - [-3b - 3(3a - b)] = 11a
theorem simplify_expr1 : (2 * a - (-3 * b - 3 * (3 * a - b))) = 11 * a :=
by
  sorry

-- Part 2: Prove that 12ab^2 - [7a^2b - (ab^2 - 3a^2b)] = 13ab^2 - 10a^2b
theorem simplify_expr2 : (12 * a * b^2 - (7 * a^2 * b - (a * b^2 - 3 * a^2 * b))) = (13 * a * b^2 - 10 * a^2 * b) :=
by
  sorry

end simplify_expr1_simplify_expr2_l90_90723


namespace fraction_of_upgraded_sensors_l90_90955

theorem fraction_of_upgraded_sensors (N U : ℕ) (h1 : N = U / 6) :
  (U / (24 * N + U)) = 1 / 5 :=
by
  sorry

end fraction_of_upgraded_sensors_l90_90955


namespace no_yarn_earnings_l90_90741

noncomputable def yarn_cost : Prop :=
  let monday_yards := 20
  let tuesday_yards := 2 * monday_yards
  let wednesday_yards := (1 / 4) * tuesday_yards
  let total_yards := monday_yards + tuesday_yards + wednesday_yards
  let fabric_cost_per_yard := 2
  let total_fabric_earnings := total_yards * fabric_cost_per_yard
  let total_earnings := 140
  total_fabric_earnings = total_earnings

theorem no_yarn_earnings:
  yarn_cost :=
sorry

end no_yarn_earnings_l90_90741


namespace express_train_speed_ratio_l90_90441

noncomputable def speed_ratio (c h : ℝ) (x : ℝ) : Prop :=
  let t1 := h / ((1 + x) * c)
  let t2 := h / ((x - 1) * c)
  x = t2 / t1

theorem express_train_speed_ratio 
  (c h : ℝ) (x : ℝ) 
  (hc : c > 0) (hh : h > 0) (hx : x > 1) : 
  speed_ratio c h (1 + Real.sqrt 2) := 
by
  sorry

end express_train_speed_ratio_l90_90441


namespace profit_percentage_previous_year_l90_90080

-- Declaring variables
variables (R P : ℝ) -- revenues and profits in the previous year
variable (revenues_1999 := 0.8 * R) -- revenues in 1999
variable (profits_1999 := 0.14 * revenues_1999) -- profits in 1999

-- Given condition: profits in 1999 were 112.00000000000001 percent of the profits in the previous year
axiom profits_ratio : 0.112 * R = 1.1200000000000001 * P

-- Prove the profit as a percentage of revenues in the previous year was 10%
theorem profit_percentage_previous_year : (P / R) * 100 = 10 := by
  sorry

end profit_percentage_previous_year_l90_90080


namespace GP_GQ_GR_proof_l90_90394

open Real

noncomputable def GP_GQ_GR_sum (XY XZ YZ : ℝ) (G : (ℝ × ℝ × ℝ)) (P Q R : (ℝ × ℝ × ℝ)) : ℝ :=
  let GP := dist G P
  let GQ := dist G Q
  let GR := dist G R
  GP + GQ + GR

theorem GP_GQ_GR_proof (XY XZ YZ : ℝ) (hXY : XY = 4) (hXZ : XZ = 3) (hYZ : YZ = 5)
  (G P Q R : (ℝ × ℝ × ℝ))
  (GP := dist G P) (GQ := dist G Q) (GR := dist G R)
  (hG : GP_GQ_GR_sum XY XZ YZ G P Q R = GP + GQ + GR) :
  GP + GQ + GR = 47 / 15 :=
sorry

end GP_GQ_GR_proof_l90_90394


namespace monomial_same_type_l90_90347

theorem monomial_same_type (a b : ℕ) (h1 : a + 1 = 3) (h2 : b = 3) : a + b = 5 :=
by 
  -- proof goes here
  sorry

end monomial_same_type_l90_90347


namespace difference_in_tiles_l90_90779

-- Definition of side length of the nth square
def side_length (n : ℕ) : ℕ := n

-- Theorem stating the difference in tiles between the 10th and 9th squares
theorem difference_in_tiles : (side_length 10) ^ 2 - (side_length 9) ^ 2 = 19 := 
by {
  sorry
}

end difference_in_tiles_l90_90779


namespace cranberry_juice_cost_l90_90350

theorem cranberry_juice_cost 
  (cost_per_ounce : ℕ) (number_of_ounces : ℕ) 
  (h1 : cost_per_ounce = 7) 
  (h2 : number_of_ounces = 12) : 
  cost_per_ounce * number_of_ounces = 84 := 
by 
  sorry

end cranberry_juice_cost_l90_90350


namespace right_triangle_sides_l90_90324

theorem right_triangle_sides (a b c : ℝ) (h_ratio : ∃ x : ℝ, a = 3 * x ∧ b = 4 * x ∧ c = 5 * x) 
(h_area : 1 / 2 * a * b = 24) : a = 6 ∧ b = 8 ∧ c = 10 :=
by
  sorry

end right_triangle_sides_l90_90324


namespace inequality_proof_l90_90457

open Real

theorem inequality_proof (a b c x y z : ℝ) 
  (ha : 0 < a ∧ a < 1) 
  (hb : 0 < b ∧ b < 1) 
  (hc : 0 < c ∧ c < 1) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (hz : 0 < z) 
  (h1 : a ^ x = b * c) 
  (h2 : b ^ y = c * a) 
  (h3 : c ^ z = a * b) :
  (1 / (2 + x) + 1 / (2 + y) + 1 / (2 + z)) ≤ 3 / 4 := 
sorry

end inequality_proof_l90_90457


namespace squirrel_walnuts_l90_90465

theorem squirrel_walnuts :
  let boy_gathered := 6
  let boy_dropped := 1
  let initial_in_burrow := 12
  let girl_brought := 5
  let girl_ate := 2
  initial_in_burrow + (boy_gathered - boy_dropped) + girl_brought - girl_ate = 20 :=
by
  sorry

end squirrel_walnuts_l90_90465


namespace rectangular_field_length_l90_90847

   theorem rectangular_field_length (w l : ℝ) 
     (h1 : l = 2 * w)
     (h2 : 64 = 8 * 8)
     (h3 : 64 = (1/72) * (l * w)) :
     l = 96 :=
   sorry
   
end rectangular_field_length_l90_90847


namespace find_number_of_tails_l90_90718

-- Definitions based on conditions
variables (T H : ℕ)
axiom total_coins : T + H = 1250
axiom heads_more_than_tails : H = T + 124

-- The goal is to prove T = 563
theorem find_number_of_tails : T = 563 :=
sorry

end find_number_of_tails_l90_90718


namespace max_subjects_per_teacher_l90_90166

theorem max_subjects_per_teacher (maths physics chemistry : ℕ) (min_teachers : ℕ)
  (h_math : maths = 6) (h_physics : physics = 5) (h_chemistry : chemistry = 5) (h_min_teachers : min_teachers = 4) :
  (maths + physics + chemistry) / min_teachers = 4 :=
by
  -- the proof will be here
  sorry

end max_subjects_per_teacher_l90_90166


namespace smallest_possible_difference_after_101_years_l90_90861

theorem smallest_possible_difference_after_101_years {D E : ℤ} 
  (init_dollar : D = 6) 
  (init_euro : E = 7)
  (transformations : ∀ D E : ℤ, 
    (D', E') = (D + E, 2 * D + 1) ∨ (D', E') = (D + E, 2 * D - 1) ∨ 
    (D', E') = (D + E, 2 * E + 1) ∨ (D', E') = (D + E, 2 * E - 1)) :
  ∃ n_diff : ℤ, 101 = 2 * n_diff ∧ n_diff = 2 :=
sorry

end smallest_possible_difference_after_101_years_l90_90861


namespace final_rider_is_C_l90_90512

def initial_order : List Char := ['A', 'B', 'C']

def leader_changes : Nat := 19
def third_place_changes : Nat := 17

def B_finishes_third (final_order: List Char) : Prop :=
  final_order.get! 2 = 'B'

def total_transpositions (a b : Nat) : Nat :=
  a + b

theorem final_rider_is_C (final_order: List Char) :
  B_finishes_third final_order →
  total_transpositions leader_changes third_place_changes % 2 = 0 →
  final_order = ['C', 'A', 'B'] → 
  final_order.get! 0 = 'C' :=
by
  sorry

end final_rider_is_C_l90_90512


namespace perimeter_shaded_area_is_942_l90_90258

-- Definition involving the perimeter of the shaded area of the circles
noncomputable def perimeter_shaded_area (s : ℝ) : ℝ := 
  4 * 75 * 3.14

-- Main theorem stating that if the side length of the octagon is 100 cm,
-- then the perimeter of the shaded area is 942 cm.
theorem perimeter_shaded_area_is_942 :
  perimeter_shaded_area 100 = 942 := 
  sorry

end perimeter_shaded_area_is_942_l90_90258


namespace distance_comparison_l90_90623

def distance_mart_to_home : ℕ := 800
def distance_home_to_academy : ℕ := 1300
def distance_academy_to_restaurant : ℕ := 1700

theorem distance_comparison :
  (distance_mart_to_home + distance_home_to_academy) - distance_academy_to_restaurant = 400 :=
by
  sorry

end distance_comparison_l90_90623


namespace cost_of_7_cubic_yards_of_topsoil_is_1512_l90_90864

-- Definition of the given conditions
def cost_per_cubic_foot : ℕ := 8
def cubic_yards : ℕ := 7
def cubic_yards_to_cubic_feet : ℕ := 27

-- Problem definition
def cost_of_topsoil (cubic_yards : ℕ) (cost_per_cubic_foot : ℕ) (cubic_yards_to_cubic_feet : ℕ) : ℕ :=
  cubic_yards * cubic_yards_to_cubic_feet * cost_per_cubic_foot

-- The proof statement
theorem cost_of_7_cubic_yards_of_topsoil_is_1512 :
  cost_of_topsoil cubic_yards cost_per_cubic_foot cubic_yards_to_cubic_feet = 1512 := by
  sorry

end cost_of_7_cubic_yards_of_topsoil_is_1512_l90_90864


namespace backyard_area_proof_l90_90564

-- Condition: Walking the length of 40 times covers 1000 meters
def length_times_40_eq_1000 (L: ℝ) : Prop := 40 * L = 1000

-- Condition: Walking the perimeter 8 times covers 1000 meters
def perimeter_times_8_eq_1000 (P: ℝ) : Prop := 8 * P = 1000

-- Given the conditions, we need to find the Length and Width of the backyard
def is_backyard_dimensions (L W: ℝ) : Prop := 
  length_times_40_eq_1000 L ∧ 
  perimeter_times_8_eq_1000 (2 * (L + W))

-- We need to calculate the area
def backyard_area (L W: ℝ) : ℝ := L * W

-- The theorem to prove
theorem backyard_area_proof (L W: ℝ) 
  (h1: length_times_40_eq_1000 L) 
  (h2: perimeter_times_8_eq_1000 (2 * (L + W))) :
  backyard_area L W = 937.5 := 
  by 
    sorry

end backyard_area_proof_l90_90564


namespace arithmetic_expression_result_l90_90419

theorem arithmetic_expression_result :
  (24 / (8 + 2 - 5)) * 7 = 33.6 :=
by
  sorry

end arithmetic_expression_result_l90_90419


namespace triangle_condition_isosceles_or_right_l90_90451

theorem triangle_condition_isosceles_or_right {A B C : ℝ} {a b c : ℝ} 
  (h_triangle : A + B + C = π) (h_cos_eq : a * Real.cos A = b * Real.cos B) : 
  (A = B) ∨ (A + B = π / 2) :=
sorry

end triangle_condition_isosceles_or_right_l90_90451


namespace total_journey_distance_l90_90713

theorem total_journey_distance (D : ℝ)
  (h1 : (D / 2) / 21 + (D / 2) / 24 = 25) : D = 560 := by
  sorry

end total_journey_distance_l90_90713


namespace total_amount_l90_90670

theorem total_amount (x y z : ℝ) (hx : y = 45 / 0.45)
  (hy : z = (45 / 0.45) * 0.30)
  (hx_total : y = 45) :
  x + y + z = 175 :=
by
  -- Proof is omitted as per instructions
  sorry

end total_amount_l90_90670


namespace find_number_l90_90568

theorem find_number (x : ℕ) (hx : (x / 100) * 100 = 20) : x = 20 :=
sorry

end find_number_l90_90568


namespace roots_of_quadratic_implies_values_l90_90709

theorem roots_of_quadratic_implies_values (a b : ℝ) :
  (∃ x : ℝ, x^2 + 2 * (1 + a) * x + (3 * a^2 + 4 * a * b + 4 * b^2 + 2) = 0) →
  a = 1 ∧ b = -1/2 :=
by
  sorry

end roots_of_quadratic_implies_values_l90_90709


namespace angle_same_terminal_side_l90_90185

theorem angle_same_terminal_side (α : ℝ) : 
  (∃ k : ℤ, α = k * 360 - 100) ↔ (∃ k : ℤ, α = k * 360 + (-100)) :=
sorry

end angle_same_terminal_side_l90_90185


namespace concert_ticket_cost_l90_90728

-- Definitions based on the conditions
def hourlyWage : ℝ := 18
def hoursPerWeek : ℝ := 30
def drinkTicketCost : ℝ := 7
def numberOfDrinkTickets : ℝ := 5
def outingPercentage : ℝ := 0.10
def weeksPerMonth : ℝ := 4

-- Proof statement
theorem concert_ticket_cost (hourlyWage hoursPerWeek drinkTicketCost numberOfDrinkTickets outingPercentage weeksPerMonth : ℝ)
  (monthlySalary := weeksPerMonth * (hoursPerWeek * hourlyWage))
  (outingAmount := outingPercentage * monthlySalary)
  (costOfDrinkTickets := numberOfDrinkTickets * drinkTicketCost)
  (costOfConcertTicket := outingAmount - costOfDrinkTickets)
  : costOfConcertTicket = 181 := 
sorry

end concert_ticket_cost_l90_90728


namespace part_a_l90_90365

theorem part_a (p : ℕ) (hp : Nat.Prime p) (odd_p : p % 2 = 1) :
  ∃ (m : ℕ) (x1 x2 x3 x4 : ℤ), m < p ∧ (x1^2 + x2^2 + x3^2 + x4^2 = m * p) :=
sorry

end part_a_l90_90365


namespace geometric_sequence_sum_l90_90332

open Real

variable {a a5 a3 a4 S4 q : ℝ}

theorem geometric_sequence_sum (h1 : q < 1)
                             (h2 : a + a5 = 20)
                             (h3 : a3 * a5 = 64) :
                             S4 = 120 := by
  sorry

end geometric_sequence_sum_l90_90332


namespace total_tbs_of_coffee_l90_90521

theorem total_tbs_of_coffee (guests : ℕ) (weak_drinkers : ℕ) (medium_drinkers : ℕ) (strong_drinkers : ℕ) 
                           (cups_per_weak_drinker : ℕ) (cups_per_medium_drinker : ℕ) (cups_per_strong_drinker : ℕ) 
                           (tbsp_per_cup_weak : ℕ) (tbsp_per_cup_medium : ℝ) (tbsp_per_cup_strong : ℕ) :
  guests = 18 ∧ 
  weak_drinkers = 6 ∧ 
  medium_drinkers = 6 ∧ 
  strong_drinkers = 6 ∧ 
  cups_per_weak_drinker = 2 ∧ 
  cups_per_medium_drinker = 3 ∧ 
  cups_per_strong_drinker = 1 ∧ 
  tbsp_per_cup_weak = 1 ∧ 
  tbsp_per_cup_medium = 1.5 ∧ 
  tbsp_per_cup_strong = 2 →
  (weak_drinkers * cups_per_weak_drinker * tbsp_per_cup_weak + 
   medium_drinkers * cups_per_medium_drinker * tbsp_per_cup_medium + 
   strong_drinkers * cups_per_strong_drinker * tbsp_per_cup_strong) = 51 :=
by
  sorry

end total_tbs_of_coffee_l90_90521


namespace arithmetic_sequence_common_difference_l90_90703

theorem arithmetic_sequence_common_difference (a d : ℕ) (n : ℕ) :
  a = 5 →
  (a + (n - 1) * d = 50) →
  (n * (a + (a + (n - 1) * d)) / 2 = 275) →
  d = 5 := 
by
  intros ha ha_n hs_n
  sorry

end arithmetic_sequence_common_difference_l90_90703


namespace product_of_constants_l90_90551

theorem product_of_constants :
  ∃ M₁ M₂ : ℝ, 
    (∀ x : ℝ, x ≠ 2 ∧ x ≠ 3 → (45 * x - 82) / (x^2 - 5 * x + 6) = M₁ / (x - 2) + M₂ / (x - 3)) ∧ 
    M₁ * M₂ = -424 :=
by
  sorry

end product_of_constants_l90_90551


namespace combination_seven_choose_three_l90_90113

-- Define the combination formula
def combination (n k : ℕ) : ℕ :=
  n.choose k

-- Define the problem-specific values
def n : ℕ := 7
def k : ℕ := 3

-- Problem statement: Prove that the number of combinations of 3 toppings from 7 is 35
theorem combination_seven_choose_three : combination 7 3 = 35 :=
  by
    sorry

end combination_seven_choose_three_l90_90113


namespace neg_power_of_square_l90_90870

theorem neg_power_of_square (a : ℝ) : (-a^2)^3 = -a^6 :=
by sorry

end neg_power_of_square_l90_90870


namespace points_on_line_with_slope_l90_90830

theorem points_on_line_with_slope :
  ∃ a b : ℝ, 
  (a - 3) ≠ 0 ∧ (b - 5) ≠ 0 ∧
  (7 - 5) / (a - 3) = 4 ∧ (b - 5) / (-1 - 3) = 4 ∧
  a = 7 / 2 ∧ b = -11 := 
by
  existsi 7 / 2
  existsi -11
  repeat {split}
  all_goals { sorry }

end points_on_line_with_slope_l90_90830


namespace number_of_cubes_l90_90225

theorem number_of_cubes (L W H V_cube : ℝ) (L_eq : L = 9) (W_eq : W = 12) (H_eq : H = 3) (V_cube_eq : V_cube = 3) :
  L * W * H / V_cube = 108 :=
by
  sorry

end number_of_cubes_l90_90225


namespace heather_final_blocks_l90_90997

def heather_initial_blocks : ℝ := 86.0
def jose_shared_blocks : ℝ := 41.0

theorem heather_final_blocks : heather_initial_blocks + jose_shared_blocks = 127.0 :=
by
  sorry

end heather_final_blocks_l90_90997


namespace cubic_roots_identity_l90_90567

noncomputable def roots_of_cubic (a b c : ℝ) : Prop :=
  (5 * a^3 - 2019 * a + 4029 = 0) ∧ 
  (5 * b^3 - 2019 * b + 4029 = 0) ∧ 
  (5 * c^3 - 2019 * c + 4029 = 0)

theorem cubic_roots_identity (a b c : ℝ) (h_roots : roots_of_cubic a b c) : 
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 12087 / 5 :=
by 
  -- proof steps
  sorry

end cubic_roots_identity_l90_90567


namespace shapes_fit_exactly_l90_90702

-- Conditions: Shapes are drawn on a piece of paper and folded along a central bold line
def shapes_drawn_on_paper := true
def paper_folded_along_central_line := true

-- Define the main proof problem
theorem shapes_fit_exactly : shapes_drawn_on_paper ∧ paper_folded_along_central_line → 
  number_of_shapes_fitting_exactly_on_top = 3 :=
by
  intros h
  sorry

end shapes_fit_exactly_l90_90702


namespace quadratic_result_l90_90228

noncomputable def quadratic_has_two_positive_integer_roots (k p : ℕ) : Prop :=
  ∃ x1 x2 : ℕ, x1 > 0 ∧ x2 > 0 ∧ (k - 1) * x1 * x1 - p * x1 + k = 0 ∧ (k - 1) * x2 * x2 - p * x2 + k = 0

theorem quadratic_result (k p : ℕ) (h1 : k = 2) (h2 : quadratic_has_two_positive_integer_roots k p) :
  k^(k*p) * (p^p + k^k) = 1984 :=
by
  sorry

end quadratic_result_l90_90228


namespace exists_root_in_interval_l90_90295

noncomputable def f (x : ℝ) : ℝ := x^3 + Real.log x - 2

theorem exists_root_in_interval :
  (f 1 < 0) ∧ (f 2 > 0) ∧ (∀ x > 0, ContinuousAt f x) → (∃ c : ℝ, 1 < c ∧ c < 2 ∧ f c = 0) :=
by sorry

end exists_root_in_interval_l90_90295


namespace second_field_full_rows_l90_90241

theorem second_field_full_rows 
    (rows_field1 : ℕ) (cobs_per_row : ℕ) (total_cobs : ℕ)
    (H1 : rows_field1 = 13)
    (H2 : cobs_per_row = 4)
    (H3 : total_cobs = 116) : 
    (total_cobs - rows_field1 * cobs_per_row) / cobs_per_row = 16 :=
by sorry

end second_field_full_rows_l90_90241


namespace original_number_l90_90069

variable (n : ℝ)

theorem original_number :
  (2 * (n + 3)^2 - 3) / 2 = 49 → n = Real.sqrt (101 / 2) - 3 :=
by
  sorry

end original_number_l90_90069


namespace Earl_owes_Fred_l90_90519

-- Define initial amounts of money each person has
def Earl_initial : ℤ := 90
def Fred_initial : ℤ := 48
def Greg_initial : ℤ := 36

-- Define debts
def Fred_owes_Greg : ℤ := 32
def Greg_owes_Earl : ℤ := 40

-- Define the total money Greg and Earl have together after debts are settled
def Greg_Earl_total_after_debts : ℤ := 130

-- Define the final amounts after debts are settled
def Earl_final (E : ℤ) : ℤ := Earl_initial - E + Greg_owes_Earl
def Fred_final (E : ℤ) : ℤ := Fred_initial + E - Fred_owes_Greg
def Greg_final : ℤ := Greg_initial + Fred_owes_Greg - Greg_owes_Earl

-- Prove that the total money Greg and Earl have together after debts are settled is 130
theorem Earl_owes_Fred (E : ℤ) (H : Greg_final + Earl_final E = Greg_Earl_total_after_debts) : E = 28 := 
by sorry

end Earl_owes_Fred_l90_90519


namespace solve_quadratic_eq_l90_90973

theorem solve_quadratic_eq : ∀ x : ℝ, (12 - 3 * x)^2 = x^2 ↔ x = 6 ∨ x = 3 :=
by
  intro x
  sorry

end solve_quadratic_eq_l90_90973


namespace simplify_expression_l90_90149

theorem simplify_expression (x : ℝ) (h : x ≠ 1) : 
  ((x^2 + 1) / (x - 1) - 2 * x / (x - 1)) = x - 1 :=
by
  -- Proof goes here.
  sorry

end simplify_expression_l90_90149


namespace minimum_value_of_polynomial_l90_90708

def polynomial (x : ℝ) : ℝ := (12 - x) * (10 - x) * (12 + x) * (10 + x)

theorem minimum_value_of_polynomial : ∃ x : ℝ, polynomial x = -484 :=
by
  sorry

end minimum_value_of_polynomial_l90_90708


namespace find_function_expression_l90_90100

theorem find_function_expression (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x^2 - 1) = x^4 + 1) :
  ∀ x : ℝ, x ≥ -1 → f x = x^2 + 2*x + 2 :=
sorry

end find_function_expression_l90_90100


namespace tim_balloons_proof_l90_90301

-- Define the number of balloons Dan has
def dan_balloons : ℕ := 29

-- Define the relationship between Tim's and Dan's balloons
def balloons_ratio : ℕ := 7

-- Define the number of balloons Tim has
def tim_balloons : ℕ := balloons_ratio * dan_balloons

-- Prove that the number of balloons Tim has is 203
theorem tim_balloons_proof : tim_balloons = 203 :=
sorry

end tim_balloons_proof_l90_90301


namespace least_possible_b_l90_90665

open Nat

/-- 
  Given conditions:
  a and b are consecutive Fibonacci numbers with a > b,
  and their sum is 100 degrees.
  We need to prove that the least possible value of b is 21 degrees.
-/
theorem least_possible_b (a b : ℕ) (h1 : fib a = fib (b + 1))
  (h2 : a > b) (h3 : a + b = 100) : b = 21 :=
sorry

end least_possible_b_l90_90665


namespace inequality_solution_value_l90_90165

theorem inequality_solution_value 
  (a : ℝ)
  (h : ∀ x, (1 < x ∧ x < 2) ↔ (ax / (x - 1) > 1)) :
  a = 1 / 2 :=
sorry

end inequality_solution_value_l90_90165


namespace trigonometric_identity_l90_90275

theorem trigonometric_identity 
  (x : ℝ) 
  (h : Real.sin (x + Real.pi / 3) = 1 / 3) :
  Real.sin (5 * Real.pi / 3 - x) - Real.cos (2 * x - Real.pi / 3) = 4 / 9 :=
by
  sorry

end trigonometric_identity_l90_90275


namespace remaining_dimes_l90_90533

-- Define the initial quantity of dimes Joan had
def initial_dimes : Nat := 5

-- Define the quantity of dimes Joan spent
def dimes_spent : Nat := 2

-- State the theorem we need to prove
theorem remaining_dimes : initial_dimes - dimes_spent = 3 := by
  sorry

end remaining_dimes_l90_90533


namespace parabola_equation_line_intersection_proof_l90_90771

-- Define the parabola and its properties
def parabola (p x y : ℝ) := y^2 = 2 * p * x

-- Define point A
def A_point (x y₀ : ℝ) := (x, y₀)

-- Define the conditions
axiom p_pos (p : ℝ) : p > 0
axiom passes_A (y₀ : ℝ) (p : ℝ) : parabola p 2 y₀
axiom distance_A_axis (p : ℝ) : 2 + p / 2 = 4

-- Prove the equation of the parabola given the conditions
theorem parabola_equation : ∃ p, parabola p x y ∧ p = 4 := sorry

-- Define line l and its intersection properties
def line_l (m x y : ℝ) := y = x + m
def intersection_PQ (m x₁ x₂ y₁ y₂ : ℝ) := 
  line_l m x₁ y₁ ∧ line_l m x₂ y₂ ∧ y₁^2 = 8 * x₁ ∧ y₂^2 = 8 * x₂ ∧ 
  x₁ + x₂ = 8 - 2 * m ∧ x₁ * x₂ = m^2 ∧ y₁ + y₂ = 8 ∧ y₁ * y₂ = 8 * m ∧ 
  x₁ * x₂ + y₁ * y₂ = 0

-- Prove the value of m
theorem line_intersection_proof : ∃ m, ∀ (x₁ x₂ y₁ y₂ : ℝ), 
  intersection_PQ m x₁ x₂ y₁ y₂ -> m = -8 := sorry

end parabola_equation_line_intersection_proof_l90_90771


namespace intersection_is_correct_l90_90627

-- Define the sets A and B based on given conditions
def setA : Set ℝ := {x | ∃ y, y = Real.log (x - 2)}
def setB : Set ℝ := {y | ∃ x, y = Real.sqrt x + 4}

-- Define the intersection of sets A and B
def intersection : Set ℝ := {z | z ≥ 4}

-- The theorem stating that the intersection of A and B is exactly the set [4, +∞)
theorem intersection_is_correct : {x | ∃ y, y = Real.log (x - 2)} ∩ {y | ∃ x, y = Real.sqrt x + 4} = {z | z ≥ 4} :=
by
  sorry

end intersection_is_correct_l90_90627


namespace binom_sum_l90_90327

theorem binom_sum :
  (Nat.choose 15 12) + 10 = 465 := by
  sorry

end binom_sum_l90_90327


namespace circumference_is_720_l90_90339

-- Given conditions
def uniform_speed (A_speed B_speed : ℕ) : Prop := A_speed > 0 ∧ B_speed > 0
def diametrically_opposite_start (A_pos B_pos : ℕ) (circumference : ℕ) : Prop := A_pos = 0 ∧ B_pos = circumference / 2
def meets_first_after_B_travel (A_distance B_distance : ℕ) : Prop := B_distance = 150
def meets_second_90_yards_before_A_lap (A_distance_lap B_distance_lap A_distance B_distance : ℕ) : Prop := 
  A_distance_lap = A_distance + 2 * (A_distance - B_distance) - 90 ∧ B_distance_lap = A_distance - B_distance_lap + (B_distance + 90)

theorem circumference_is_720 (circumference A_speed B_speed A_pos B_pos
                     A_distance B_distance
                     A_distance_lap B_distance_lap : ℕ) :
  uniform_speed A_speed B_speed →
  diametrically_opposite_start A_pos B_pos circumference →
  meets_first_after_B_travel A_distance B_distance →
  meets_second_90_yards_before_A_lap A_distance_lap B_distance_lap A_distance B_distance →
  circumference = 720 :=
sorry

end circumference_is_720_l90_90339


namespace selling_price_before_brokerage_l90_90392

variables (CR BR SP : ℝ)
variables (hCR : CR = 120.50) (hBR : BR = 1 / 400)

theorem selling_price_before_brokerage :
  SP = (CR * 400) / (399) := 
by
  sorry

end selling_price_before_brokerage_l90_90392


namespace marys_garbage_bill_l90_90340

def weekly_cost_trash (trash_count : ℕ) := 10 * trash_count
def weekly_cost_recycling (recycling_count : ℕ) := 5 * recycling_count

def weekly_cost (trash_count : ℕ) (recycling_count : ℕ) : ℕ :=
  weekly_cost_trash trash_count + weekly_cost_recycling recycling_count

def monthly_cost (weekly_cost : ℕ) := 4 * weekly_cost

def elderly_discount (total_cost : ℕ) : ℕ :=
  total_cost * 18 / 100

def final_bill (monthly_cost : ℕ) (discount : ℕ) (fine : ℕ) : ℕ :=
  monthly_cost - discount + fine

theorem marys_garbage_bill : final_bill
  (monthly_cost (weekly_cost 2 1))
  (elderly_discount (monthly_cost (weekly_cost 2 1)))
  20 = 102 := by
{
  sorry -- The proof steps are omitted as per the instructions.
}

end marys_garbage_bill_l90_90340


namespace hexagon_label_count_l90_90170

def hexagon_label (s : Finset ℕ) (a b c d e f g : ℕ) : Prop :=
  s = Finset.range 8 ∧ 
  (a ∈ s) ∧ (b ∈ s) ∧ (c ∈ s) ∧ (d ∈ s) ∧ (e ∈ s) ∧ (f ∈ s) ∧ (g ∈ s) ∧
  a + b + c + d + e + f + g = 28 ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ 
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧
  e ≠ f ∧ e ≠ g ∧
  f ≠ g ∧
  a + g + d = b + g + e ∧ b + g + e = c + g + f

theorem hexagon_label_count : ∃ s a b c d e f g, hexagon_label s a b c d e f g ∧ 
  (s.card = 8) ∧ (a + g + d = 10) ∧ (b + g + e = 10) ∧ (c + g + f = 10) ∧ 
  144 = 3 * 48 :=
sorry

end hexagon_label_count_l90_90170


namespace problem_1_problem_2_l90_90240

def f (a x : ℝ) : ℝ := |a - 3 * x| - |2 + x|

theorem problem_1 (x : ℝ) : f 2 x ≤ 3 ↔ -3 / 4 ≤ x ∧ x ≤ 7 / 2 := by
  sorry

theorem problem_2 (a x : ℝ) : f a x ≥ 1 - a + 2 * |2 + x| → a ≥ -5 / 2 := by
  sorry

end problem_1_problem_2_l90_90240


namespace find_f_2023_4_l90_90015

noncomputable def f : ℕ → ℝ → ℝ
| 0, x => 2 * Real.sqrt x
| (n + 1), x => 4 / (2 - f n x)

theorem find_f_2023_4 : f 2023 4 = -2 := sorry

end find_f_2023_4_l90_90015


namespace base3_to_base10_conversion_l90_90118

theorem base3_to_base10_conversion : ∀ n : ℕ, n = 120102 → (1 * 3^5 + 2 * 3^4 + 0 * 3^3 + 1 * 3^2 + 0 * 3^1 + 2 * 3^0) = 416 :=
by
  intro n hn
  sorry

end base3_to_base10_conversion_l90_90118


namespace no_three_distinct_rational_roots_l90_90434

theorem no_three_distinct_rational_roots (a b : ℝ) : 
  ¬ ∃ (u v w : ℚ), 
    u + v + w = -(2 * a + 1) ∧ 
    u * v + v * w + w * u = (2 * a^2 + 2 * a - 3) ∧ 
    u * v * w = b := sorry

end no_three_distinct_rational_roots_l90_90434


namespace max_cubes_fit_l90_90383

theorem max_cubes_fit (L S : ℕ) (hL : L = 10) (hS : S = 2) : (L * L * L) / (S * S * S) = 125 := by
  sorry

end max_cubes_fit_l90_90383


namespace circle_equation_standard_l90_90755

def center : ℝ × ℝ := (-1, 1)
def radius : ℝ := 2

theorem circle_equation_standard:
  (∀ x y : ℝ, ((x + 1)^2 + (y - 1)^2 = 4) ↔ ((x - center.1)^2 + (y - center.2)^2 = radius^2)) :=
by 
  intros x y
  rw [center, radius]
  simp
  sorry

end circle_equation_standard_l90_90755


namespace mary_initial_pokemon_cards_l90_90954

theorem mary_initial_pokemon_cards (x : ℕ) (torn_cards : ℕ) (new_cards : ℕ) (current_cards : ℕ) 
  (h1 : torn_cards = 6) 
  (h2 : new_cards = 23) 
  (h3 : current_cards = 56) 
  (h4 : current_cards = x - torn_cards + new_cards) : 
  x = 39 := 
by
  sorry

end mary_initial_pokemon_cards_l90_90954


namespace probability_A_not_losing_l90_90031

theorem probability_A_not_losing (P_draw P_win : ℚ) (h1 : P_draw = 1/2) (h2 : P_win = 1/3) : 
  P_draw + P_win = 5/6 :=
by
  sorry

end probability_A_not_losing_l90_90031


namespace only_B_is_like_terms_l90_90096

def is_like_terms (terms : List (String × String)) : List Bool :=
  let like_term_checker := fun (term1 term2 : String) =>
    -- The function to check if two terms are like terms
    sorry
  terms.map (fun (term1, term2) => like_term_checker term1 term2)

theorem only_B_is_like_terms :
  is_like_terms [("−2x^3", "−3x^2"), ("−(1/4)ab", "18ba"), ("a^2b", "−ab^2"), ("4m", "6mn")] =
  [false, true, false, false] :=
by
  sorry

end only_B_is_like_terms_l90_90096


namespace find_cost_of_baseball_l90_90008

noncomputable def total_amount : ℝ := 20.52
noncomputable def cost_of_marbles : ℝ := 9.05
noncomputable def cost_of_football : ℝ := 4.95
noncomputable def cost_of_baseball : ℝ := total_amount - (cost_of_marbles + cost_of_football)

theorem find_cost_of_baseball : cost_of_baseball = 6.52 := sorry

end find_cost_of_baseball_l90_90008


namespace additional_telephone_lines_l90_90528

theorem additional_telephone_lines :
  let lines_six_digits := 9 * 10^5
  let lines_seven_digits := 9 * 10^6
  let additional_lines := lines_seven_digits - lines_six_digits
  additional_lines = 81 * 10^5 :=
by
  sorry

end additional_telephone_lines_l90_90528


namespace circumcircle_diameter_l90_90699

-- Given that the perimeter of triangle ABC is equal to 3 times the sum of the sines of its angles
-- and the Law of Sines holds for this triangle, we need to prove the diameter of the circumcircle is 3.
theorem circumcircle_diameter (a b c : ℝ) (A B C : ℝ) (R : ℝ)
  (h_perimeter : a + b + c = 3 * (Real.sin A + Real.sin B + Real.sin C))
  (h_law_of_sines : a / Real.sin A = 2 * R ∧ b / Real.sin B = 2 * R ∧ c / Real.sin C = 2 * R) :
  2 * R = 3 := 
by
  sorry

end circumcircle_diameter_l90_90699


namespace hazel_walked_distance_l90_90690

theorem hazel_walked_distance
  (first_hour_distance : ℕ)
  (second_hour_distance : ℕ)
  (h1 : first_hour_distance = 2)
  (h2 : second_hour_distance = 2 * first_hour_distance) :
  (first_hour_distance + second_hour_distance = 6) :=
by {
  sorry
}

end hazel_walked_distance_l90_90690


namespace first_digit_of_sum_l90_90025

theorem first_digit_of_sum (n : ℕ) (a : ℕ) (hs : 9 * a = n)
  (h_sum : n = 43040102 - (10^7 * d - 10^7 * 4)) : 
  (10^7 * d - 10^7 * 4) / 10^7 = 8 :=
by
  sorry

end first_digit_of_sum_l90_90025


namespace line_through_point_equal_intercepts_locus_equidistant_lines_l90_90546

theorem line_through_point_equal_intercepts (x y : ℝ) (hx : x = 1) (hy : y = 3) :
  (∃ k : ℝ, y = k * x ∧ k = 3) ∨ (∃ a : ℝ, x + y = a ∧ a = 4) :=
sorry

theorem locus_equidistant_lines (x y : ℝ) :
  ∀ (a b : ℝ), (2 * x + 3 * y - a = 0) ∧ (4 * x + 6 * y + b = 0) →
  ∀ b : ℝ, |b + 10| = |b - 8| → b = -9 → 
  4 * x + 6 * y - 9 = 0 :=
sorry

end line_through_point_equal_intercepts_locus_equidistant_lines_l90_90546


namespace brick_length_l90_90200

theorem brick_length (w h SA : ℝ) (h_w : w = 6) (h_h : h = 2) (h_SA : SA = 152) :
  ∃ l : ℝ, 2 * l * w + 2 * l * h + 2 * w * h = SA ∧ l = 8 := 
sorry

end brick_length_l90_90200


namespace negation_exists_negation_proposition_l90_90982

theorem negation_exists (P : ℝ → Prop) :
  (∃ x : ℝ, P x) ↔ ¬ (∀ x : ℝ, ¬ P x) :=
by sorry

theorem negation_proposition :
  (¬ ∃ x : ℝ, x^2 + 1 < 0) ↔ (∀ x : ℝ, x^2 + 1 ≥ 0) :=
by sorry

end negation_exists_negation_proposition_l90_90982


namespace compare_fractions_l90_90642

theorem compare_fractions (a : ℝ) : 
  (a = 0 → (1 / (1 - a)) = (1 + a)) ∧ 
  (0 < a ∧ a < 1 → (1 / (1 - a)) > (1 + a)) ∧ 
  (a > 1 → (1 / (1 - a)) < (1 + a)) := by
  sorry

end compare_fractions_l90_90642


namespace betty_height_correct_l90_90525

-- Definitions for the conditions
def dog_height : ℕ := 24
def carter_height : ℕ := 2 * dog_height
def betty_height_inches : ℕ := carter_height - 12
def betty_height_feet : ℕ := betty_height_inches / 12

-- Theorem that we need to prove
theorem betty_height_correct : betty_height_feet = 3 :=
by
  sorry

end betty_height_correct_l90_90525


namespace ben_mms_count_l90_90193

theorem ben_mms_count (S M : ℕ) (hS : S = 50) (h_diff : S = M + 30) : M = 20 := by
  sorry

end ben_mms_count_l90_90193


namespace domino_chain_can_be_built_l90_90333

def domino_chain_possible : Prop :=
  let total_pieces := 28
  let pieces_with_sixes_removed := 7
  let remaining_pieces := total_pieces - pieces_with_sixes_removed
  (∀ n : ℕ, n < 6 → (∃ k : ℕ, k = 6) → (remaining_pieces % 2 = 0))

theorem domino_chain_can_be_built (h : domino_chain_possible) : Prop :=
  sorry

end domino_chain_can_be_built_l90_90333


namespace transistors_in_2005_l90_90937

theorem transistors_in_2005
  (initial_count : ℕ)
  (doubles_every : ℕ)
  (triples_every : ℕ)
  (years : ℕ) :
  initial_count = 500000 ∧ doubles_every = 2 ∧ triples_every = 6 ∧ years = 15 →
  (initial_count * 2^(years/doubles_every) + initial_count * 3^(years/triples_every)) = 68500000 :=
by
  sorry

end transistors_in_2005_l90_90937


namespace minimum_dwarfs_l90_90202

theorem minimum_dwarfs (n : ℕ) (C : ℕ → Prop) (h_nonempty : ∀ i, ∃ j, j = (i + 1) % 30 ∨ j = (i + 2) % 30 ∨ j = (i + 3) % 30 → C j) :
  ∃ m, 10 ≤ m ∧ (∀ i, ∃ j, j = (i + 1) % 30 ∨ j = (i + 2) % 30 ∨ j = (i + 3) % 30 → C j) :=
sorry

end minimum_dwarfs_l90_90202


namespace arithmetic_seq_a3_value_l90_90085

-- Given the arithmetic sequence {a_n}, where
-- a_1 + a_2 + a_3 + a_4 + a_5 = 20
def arithmetic_seq (a : ℕ → ℝ) := ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n

theorem arithmetic_seq_a3_value {a : ℕ → ℝ}
    (h_seq : arithmetic_seq a)
    (h_sum : a 1 + a 2 + a 3 + a 4 + a 5 = 20) :
  a 3 = 4 :=
by
  sorry

end arithmetic_seq_a3_value_l90_90085


namespace sequence_x_value_l90_90287

theorem sequence_x_value (x : ℕ) (h1 : 3 - 1 = 2) (h2 : 6 - 3 = 3) (h3 : 10 - 6 = 4) (h4 : x - 10 = 5) : x = 15 :=
by
  sorry

end sequence_x_value_l90_90287


namespace proposition_holds_for_all_positive_odd_numbers_l90_90695

theorem proposition_holds_for_all_positive_odd_numbers
  (P : ℕ → Prop)
  (h1 : P 1)
  (h2 : ∀ k, k ≥ 1 → P k → P (k + 2)) :
  ∀ n, n % 2 = 1 → n ≥ 1 → P n :=
by
  sorry

end proposition_holds_for_all_positive_odd_numbers_l90_90695


namespace carousel_ratio_l90_90874

theorem carousel_ratio (P : ℕ) (h : 3 + P + 2*P + P/3 = 33) : P / 3 = 3 := 
by 
  sorry

end carousel_ratio_l90_90874


namespace employee_pay_l90_90493

theorem employee_pay (y : ℝ) (x : ℝ) (h1 : x = 1.2 * y) (h2 : x + y = 700) : y = 318.18 :=
by
  sorry

end employee_pay_l90_90493


namespace graph_through_origin_y_increases_with_x_not_pass_third_quadrant_l90_90043

-- Part 1: Prove that if the graph passes through the origin, then m ≠ 2/3 and n = 1
theorem graph_through_origin {m n : ℝ} : 
  (3 * m - 2 ≠ 0) → (1 - n = 0) ↔ (m ≠ 2/3 ∧ n = 1) :=
by sorry

-- Part 2: Prove that if y increases as x increases, then m > 2/3 and n is any real number
theorem y_increases_with_x {m n : ℝ} : 
  (3 * m - 2 > 0) ↔ (m > 2/3 ∧ ∀ n : ℝ, True) :=
by sorry

-- Part 3: Prove that if the graph does not pass through the third quadrant, then m < 2/3 and n ≤ 1
theorem not_pass_third_quadrant {m n : ℝ} : 
  (3 * m - 2 < 0) ∧ (1 - n ≥ 0) ↔ (m < 2/3 ∧ n ≤ 1) :=
by sorry

end graph_through_origin_y_increases_with_x_not_pass_third_quadrant_l90_90043


namespace max_candies_ben_eats_l90_90889

theorem max_candies_ben_eats (total_candies : ℕ) (k : ℕ) (h_pos_k : k > 0) (b : ℕ) 
  (h_total : b + 2 * b + k * b = total_candies) (h_total_candies : total_candies = 30) : b = 6 :=
by
  -- placeholder for proof steps
  sorry

end max_candies_ben_eats_l90_90889


namespace problem_l90_90622

theorem problem (f : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, f (x * y) + x = x * f y + f x)
  (h2 : f (1 / 2) = 0) : 
  f (-201) = 403 :=
sorry

end problem_l90_90622


namespace simplify_and_evaluate_l90_90074

noncomputable def a := 2 * Real.sin (Real.pi / 4) + (1 / 2) ^ (-1 : ℤ)

theorem simplify_and_evaluate :
  (a^2 - 4) / a / ((4 * a - 4) / a - a) + 2 / (a - 2) = -1 - Real.sqrt 2 := by
  sorry

end simplify_and_evaluate_l90_90074


namespace smallest_x_for_perfect_cube_l90_90505

theorem smallest_x_for_perfect_cube (x N : ℕ) (hN : 1260 * x = N^3) (h_fact : 1260 = 2^2 * 3^2 * 5 * 7): x = 7350 := sorry

end smallest_x_for_perfect_cube_l90_90505


namespace double_root_values_l90_90514

theorem double_root_values (b₃ b₂ b₁ s : ℤ) (h : ∀ x : ℤ, (x * (x - s)) ∣ (x^4 + b₃ * x^3 + b₂ * x^2 + b₁ * x + 36)) 
  : s = -6 ∨ s = -3 ∨ s = -2 ∨ s = -1 ∨ s = 1 ∨ s = 2 ∨ s = 3 ∨ s = 6 :=
sorry

end double_root_values_l90_90514


namespace minutes_in_3_5_hours_l90_90791

theorem minutes_in_3_5_hours : 3.5 * 60 = 210 := 
by
  sorry

end minutes_in_3_5_hours_l90_90791


namespace circle_ratio_l90_90022

theorem circle_ratio (R r a c : ℝ) (hR : 0 < R) (hr : 0 < r) (h_c_lt_a : 0 < c ∧ c < a) 
  (condition : π * R^2 = (a - c) * (π * R^2 - π * r^2)) :
  R / r = Real.sqrt ((a - c) / (c + 1 - a)) :=
by
  sorry

end circle_ratio_l90_90022


namespace find_positive_integer_pairs_l90_90265

theorem find_positive_integer_pairs :
  ∀ (m n : ℕ), m > 0 ∧ n > 0 → ∃ k : ℕ, (2^n - 13^m = k^3) ↔ (m = 2 ∧ n = 9) :=
by
  sorry

end find_positive_integer_pairs_l90_90265


namespace min_value_expression_l90_90867

/-- 
Given real numbers a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p such that 
abcd = 16, efgh = 16, ijkl = 16, and mnop = 16, prove that the minimum value of 
(aeim)^2 + (bfjn)^2 + (cgko)^2 + (dhlp)^2 is 1024. 
-/
theorem min_value_expression (a b c d e f g h i j k l m n o p : ℝ) 
  (h1 : a * b * c * d = 16) 
  (h2 : e * f * g * h = 16) 
  (h3 : i * j * k * l = 16) 
  (h4 : m * n * o * p = 16) : 
  (a * e * i * m) ^ 2 + (b * f * j * n) ^ 2 + (c * g * k * o) ^ 2 + (d * h * l * p) ^ 2 ≥ 1024 :=
by 
  sorry


end min_value_expression_l90_90867


namespace novel_pages_l90_90070

theorem novel_pages (x : ℕ)
  (h1 : x - ((1 / 6 : ℝ) * x + 10) = (5 / 6 : ℝ) * x - 10)
  (h2 : (5 / 6 : ℝ) * x - 10 - ((1 / 5 : ℝ) * ((5 / 6 : ℝ) * x - 10) + 20) = (2 / 3 : ℝ) * x - 28)
  (h3 : (2 / 3 : ℝ) * x - 28 - ((1 / 4 : ℝ) * ((2 / 3 : ℝ) * x - 28) + 25) = (1 / 2 : ℝ) * x - 46) :
  (1 / 2 : ℝ) * x - 46 = 80 → x = 252 :=
by
  sorry

end novel_pages_l90_90070


namespace outfit_choices_l90_90318

-- Define the numbers of shirts, pants, and hats.
def num_shirts : ℕ := 6
def num_pants : ℕ := 7
def num_hats : ℕ := 6

-- Define the number of colors and the constraints.
def num_colors : ℕ := 6

-- The total number of outfits without restrictions.
def total_outfits : ℕ := num_shirts * num_pants * num_hats

-- Number of outfits where all items are the same color.
def same_color_outfits : ℕ := num_colors

-- Number of outfits where the shirt and pants are the same color.
def same_shirt_pants_color_outfits : ℕ := num_colors + 1  -- accounting for the extra pair of pants

-- The total number of valid outfits calculated.
def valid_outfits : ℕ :=
  total_outfits - same_color_outfits - same_shirt_pants_color_outfits

-- The theorem statement asserting the correct answer.
theorem outfit_choices : valid_outfits = 239 := by
  sorry

end outfit_choices_l90_90318


namespace perfect_cube_divisor_l90_90127

theorem perfect_cube_divisor (a b : ℕ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : a^2 + 3*a*b + 3*b^2 - 1 ∣ a + b^3) :
  ∃ k > 1, ∃ m : ℕ, a^2 + 3*a*b + 3*b^2 - 1 = k^3 * m := 
sorry

end perfect_cube_divisor_l90_90127


namespace prime_eq_sum_of_two_squares_l90_90404

theorem prime_eq_sum_of_two_squares (p : ℕ) (hp_prime : Nat.Prime p) (hp_mod : p % 4 = 1) : 
  ∃ a b : ℤ, p = a^2 + b^2 := 
sorry

end prime_eq_sum_of_two_squares_l90_90404


namespace granola_bars_relation_l90_90090

theorem granola_bars_relation (x y z : ℕ) (h1 : z = x / (3 * y)) : z = x / (3 * y) :=
by {
    sorry
}

end granola_bars_relation_l90_90090


namespace abs_add_eq_abs_sub_implies_mul_eq_zero_l90_90372

variable {a b : ℝ}

theorem abs_add_eq_abs_sub_implies_mul_eq_zero (h : |a + b| = |a - b|) : a * b = 0 :=
sorry

end abs_add_eq_abs_sub_implies_mul_eq_zero_l90_90372


namespace area_is_25_l90_90094

noncomputable def area_of_square (x : ℝ) : ℝ :=
  let side1 := 5 * x - 20
  let side2 := 25 - 4 * x
  if h : side1 = side2 then 
    side1 * side1
  else 
    0

theorem area_is_25 (x : ℝ) (h_eq : 5 * x - 20 = 25 - 4 * x) : area_of_square x = 25 :=
by
  sorry

end area_is_25_l90_90094


namespace James_will_take_7_weeks_l90_90542

def pages_per_hour : ℕ := 5
def hours_per_day : ℕ := 4 - 1
def pages_per_day : ℕ := hours_per_day * pages_per_hour
def total_pages : ℕ := 735
def days_to_finish : ℕ := total_pages / pages_per_day
def weeks_to_finish : ℕ := days_to_finish / 7

theorem James_will_take_7_weeks :
  weeks_to_finish = 7 :=
by
  -- You can add the necessary proof steps here
  sorry

end James_will_take_7_weeks_l90_90542


namespace jesse_remaining_pages_l90_90480

theorem jesse_remaining_pages (pages_read : ℕ)
  (h1 : pages_read = 83)
  (h2 : pages_read = (1 / 3 : ℝ) * total_pages)
  : pages_remaining = 166 :=
  by 
    -- Here we would build the proof, skipped with sorry
    sorry

end jesse_remaining_pages_l90_90480


namespace paper_stars_per_bottle_l90_90794

theorem paper_stars_per_bottle (a b total_bottles : ℕ) (h1 : a = 33) (h2 : b = 307) (h3 : total_bottles = 4) :
  (a + b) / total_bottles = 85 :=
by
  sorry

end paper_stars_per_bottle_l90_90794


namespace andrey_boris_denis_eat_candies_l90_90203

def andrey_boris_condition (a b : ℕ) : Prop :=
  a = 4 ∧ b = 3

def andrey_denis_condition (a d : ℕ) : Prop :=
  a = 6 ∧ d = 7

def total_candies_condition (total : ℕ) : Prop :=
  total = 70

theorem andrey_boris_denis_eat_candies :
  ∃ (a b d : ℕ), andrey_boris_condition a b ∧ andrey_denis_condition a d ∧ 
                  (total_candies_condition (2 * (12 + 9 + 14)) ∧ 
                   2 * 12 = 24 ∧ 2 * 9 = 18 ∧ 2 * 14 = 28) →
                  (a = 24 ∧ b = 18 ∧ d = 28) :=
by
  sorry

end andrey_boris_denis_eat_candies_l90_90203


namespace dot_product_range_l90_90856

theorem dot_product_range (a b : ℝ) (θ : ℝ) (h1 : a = 8) (h2 : b = 12)
  (h3 : 30 * (Real.pi / 180) ≤ θ ∧ θ ≤ 60 * (Real.pi / 180)) :
  48 * Real.sqrt 3 ≤ a * b * Real.cos θ ∧ a * b * Real.cos θ ≤ 48 :=
by
  sorry

end dot_product_range_l90_90856


namespace average_annual_growth_rate_eq_l90_90422

-- Definition of variables based on given conditions
def sales_2021 := 298 -- in 10,000 units
def sales_2023 := 850 -- in 10,000 units
def years := 2

-- Problem statement in Lean 4
theorem average_annual_growth_rate_eq :
  sales_2021 * (1 + x) ^ years = sales_2023 :=
sorry

end average_annual_growth_rate_eq_l90_90422


namespace minimum_value_8_l90_90772

noncomputable def minimum_value (x : ℝ) : ℝ :=
  3 * x + 2 / x^5 + 3 / x

theorem minimum_value_8 (x : ℝ) (hx : x > 0) :
  ∃ y : ℝ, (∀ z > 0, minimum_value z ≥ y) ∧ (y = 8) :=
by
  sorry

end minimum_value_8_l90_90772


namespace frac_subtraction_l90_90730

theorem frac_subtraction : (18 / 42) - (3 / 8) = (3 / 56) := by
  -- Conditions
  have h1 : 18 / 42 = 3 / 7 := by sorry
  have h2 : 3 / 7 = 24 / 56 := by sorry
  have h3 : 3 / 8 = 21 / 56 := by sorry
  -- Proof using the conditions
  sorry

end frac_subtraction_l90_90730


namespace value_of_x_squared_plus_reciprocal_squared_l90_90497

theorem value_of_x_squared_plus_reciprocal_squared (x : ℝ) (hx : 0 < x) (h : x + 1/x = Real.sqrt 2020) : x^2 + 1/x^2 = 2018 :=
sorry

end value_of_x_squared_plus_reciprocal_squared_l90_90497


namespace greatest_ln_2_l90_90516

theorem greatest_ln_2 (x1 x2 x3 x4 : ℝ) (h1 : x1 = (Real.log 2) ^ 2) (h2 : x2 = Real.log (Real.log 2)) (h3 : x3 = Real.log (Real.sqrt 2)) (h4 : x4 = Real.log 2) 
  (h5 : Real.log 2 < 1) : 
  x4 = max x1 (max x2 (max x3 x4)) := by 
  sorry

end greatest_ln_2_l90_90516


namespace dice_sum_probability_l90_90737

theorem dice_sum_probability
  (a b c d : ℕ)
  (cond1 : 1 ≤ a ∧ a ≤ 6)
  (cond2 : 1 ≤ b ∧ b ≤ 6)
  (cond3 : 1 ≤ c ∧ c ≤ 6)
  (cond4 : 1 ≤ d ∧ d ≤ 6)
  (sum_cond : a + b + c + d = 5) :
  (∃ p, p = 1 / 324) :=
sorry

end dice_sum_probability_l90_90737


namespace canadian_ratio_correct_l90_90293

-- The total number of scientists
def total_scientists : ℕ := 70

-- Half of the scientists are from Europe
def european_scientists : ℕ := total_scientists / 2

-- The number of scientists from the USA
def usa_scientists : ℕ := 21

-- The number of Canadian scientists
def canadian_scientists : ℕ := total_scientists - european_scientists - usa_scientists

-- The ratio of the number of Canadian scientists to the total number of scientists
def canadian_ratio : ℚ := canadian_scientists / total_scientists

-- Prove that the ratio is 1:5
theorem canadian_ratio_correct : canadian_ratio = 1 / 5 :=
by
  sorry

end canadian_ratio_correct_l90_90293


namespace probability_of_factor_less_than_ten_is_half_l90_90588

-- Definitions for the factors and counts
def numFactors (n : ℕ) : ℕ :=
  let psa := 1;
  let psb := 2;
  let psc := 1;
  (psa + 1) * (psb + 1) * (psc + 1)

def factorsLessThanTen (n : ℕ) : List ℕ :=
  if n = 90 then [1, 2, 3, 5, 6, 9] else []

def probabilityLessThanTen (n : ℕ) : ℚ :=
  let totalFactors := numFactors n;
  let lessThanTenFactors := factorsLessThanTen n;
  let favorableOutcomes := lessThanTenFactors.length;
  favorableOutcomes / totalFactors

-- The proof statement
theorem probability_of_factor_less_than_ten_is_half :
  probabilityLessThanTen 90 = 1 / 2 := sorry

end probability_of_factor_less_than_ten_is_half_l90_90588


namespace ellipse_k_values_l90_90561

theorem ellipse_k_values (k : ℝ) :
  (∃ a b : ℝ, a = (k + 8) ∧ b = 9 ∧ 
  (b > a → (a * (1 - (1 / 2) ^ 2) = b - a) ∧ k = 4) ∧ 
  (a > b → (b * (1 - (1 / 2) ^ 2) = a - b) ∧ k = -5/4)) :=
sorry

end ellipse_k_values_l90_90561


namespace probability_of_not_red_l90_90501

-- Definitions based on conditions
def total_number_of_jelly_beans : ℕ := 7 + 9 + 10 + 12 + 5
def number_of_non_red_jelly_beans : ℕ := 9 + 10 + 12 + 5

-- Proving the probability
theorem probability_of_not_red : 
  (number_of_non_red_jelly_beans : ℚ) / total_number_of_jelly_beans = 36 / 43 :=
by sorry

end probability_of_not_red_l90_90501


namespace sample_size_is_150_l90_90068

theorem sample_size_is_150 
  (classes : ℕ) (students_per_class : ℕ) (selected_students : ℕ)
  (h1 : classes = 40) (h2 : students_per_class = 50) (h3 : selected_students = 150)
  : selected_students = 150 :=
sorry

end sample_size_is_150_l90_90068


namespace intersection_point_l90_90743

variable (x y : ℚ)

theorem intersection_point :
  (8 * x - 5 * y = 40) ∧ (10 * x + 2 * y = 14) → 
  (x = 25 / 11) ∧ (y = 48 / 11) :=
by
  sorry

end intersection_point_l90_90743


namespace largest_common_in_range_l90_90361

-- Definitions for the problem's conditions
def first_seq (n : ℕ) : ℕ := 3 + 8 * n
def second_seq (m : ℕ) : ℕ := 5 + 9 * m

-- Statement of the theorem we are proving
theorem largest_common_in_range : 
  ∃ n m : ℕ, first_seq n = second_seq m ∧ 1 ≤ first_seq n ∧ first_seq n ≤ 200 ∧ first_seq n = 131 := by
  sorry

end largest_common_in_range_l90_90361


namespace perfect_square_values_l90_90257

theorem perfect_square_values :
  ∀ n : ℕ, 0 < n → (∃ k : ℕ, (n^2 + 11 * n - 4) * n.factorial + 33 * 13^n + 4 = k^2) ↔ n = 1 ∨ n = 2 :=
by sorry

end perfect_square_values_l90_90257


namespace inequality_a_squared_plus_b_squared_l90_90078

variable (a b : ℝ)

theorem inequality_a_squared_plus_b_squared (h : a > b) : a^2 + b^2 > ab := 
sorry

end inequality_a_squared_plus_b_squared_l90_90078


namespace percentage_x_y_l90_90316

variable (x y P : ℝ)

theorem percentage_x_y 
  (h1 : 0.5 * (x - y) = (P / 100) * (x + y))
  (h2 : y = (1 / 9) * x) : 
  P = 40 :=
sorry

end percentage_x_y_l90_90316


namespace kyle_paper_delivery_l90_90802

-- Define the number of houses Kyle delivers to from Monday through Saturday
def housesDaily : ℕ := 100

-- Define the number of days from Monday to Saturday
def daysWeek : ℕ := 6

-- Define the adjustment for Sunday (10 fewer, 30 additional)
def sundayAdjust : ℕ := 30 - 10 + 100

-- Define the total number of papers delivered from Monday to Saturday
def papersMonToSat : ℕ := daysWeek * housesDaily

-- Define the total number of papers delivered on Sunday
def papersSunday : ℕ := sundayAdjust

-- Define the total number of papers delivered each week
def totalPapers : ℕ := papersMonToSat + papersSunday

-- The theorem we want to prove
theorem kyle_paper_delivery : totalPapers = 720 := by
  -- We are focusing only on the statement here.
  sorry

end kyle_paper_delivery_l90_90802


namespace percentage_milk_in_B_l90_90654

theorem percentage_milk_in_B :
  ∀ (A B C : ℕ),
  A = 1200 → B + C = A → B + 150 = C - 150 →
  (B:ℝ) / (A:ℝ) * 100 = 37.5 :=
by
  intros A B C hA hBC hE
  sorry

end percentage_milk_in_B_l90_90654


namespace not_p_or_not_q_implies_p_and_q_and_p_or_q_l90_90668

variable (p q : Prop)

theorem not_p_or_not_q_implies_p_and_q_and_p_or_q (h : ¬ (¬ p ∨ ¬ q)) : (p ∧ q) ∧ (p ∨ q) :=
sorry

end not_p_or_not_q_implies_p_and_q_and_p_or_q_l90_90668


namespace total_number_of_workers_l90_90988

variables (W N : ℕ)
variables (average_salary_workers average_salary_techs average_salary_non_techs : ℤ)
variables (num_techs total_salary total_salary_techs total_salary_non_techs : ℤ)

theorem total_number_of_workers (h1 : average_salary_workers = 8000)
                               (h2 : average_salary_techs = 14000)
                               (h3 : num_techs = 7)
                               (h4 : average_salary_non_techs = 6000)
                               (h5 : total_salary = W * 8000)
                               (h6 : total_salary_techs = 7 * 14000)
                               (h7 : total_salary_non_techs = N * 6000)
                               (h8 : total_salary = total_salary_techs + total_salary_non_techs)
                               (h9 : W = 7 + N) : 
                               W = 28 :=
sorry

end total_number_of_workers_l90_90988


namespace trigonometric_identity_proof_l90_90664

variable (α β : Real)

theorem trigonometric_identity_proof :
  4.28 * Real.sin (β / 2 - Real.pi / 2) ^ 2 - Real.cos (α - 3 * Real.pi / 2) ^ 2 = 
  Real.cos (α + β) * Real.cos (α - β) :=
by
  sorry

end trigonometric_identity_proof_l90_90664


namespace bill_due_in_months_l90_90382

theorem bill_due_in_months
  (TD : ℝ) (FV : ℝ) (R_annual : ℝ) (m : ℝ) 
  (h₀ : TD = 270)
  (h₁ : FV = 2520)
  (h₂ : R_annual = 16) :
  m = 9 :=
by
  sorry

end bill_due_in_months_l90_90382


namespace expand_product_l90_90722

theorem expand_product (x : ℝ) : (x + 3) * (x + 9) = x^2 + 12 * x + 27 := 
by 
  sorry

end expand_product_l90_90722


namespace least_x_value_l90_90925

theorem least_x_value : ∀ x : ℝ, (4 * x^2 + 7 * x + 3 = 5) → x = -2 ∨ x >= -2 := by 
    intro x
    intro h
    sorry

end least_x_value_l90_90925


namespace quadratic_inequality_solution_range_l90_90184

open Set Real

theorem quadratic_inequality_solution_range
  (a : ℝ) : (∃ (x1 x2 : ℤ), x1 ≠ x2 ∧ (∀ x : ℝ, x^2 - a * x + 2 * a < 0 ↔ ↑x1 < x ∧ x < ↑x2)) ↔ 
    (a ∈ Icc (-1 : ℝ) ((-1:ℝ)/3)) ∨ (a ∈ Ioo (25 / 3 : ℝ) 9) :=
sorry

end quadratic_inequality_solution_range_l90_90184


namespace total_value_of_pile_l90_90470

def value_of_pile (total_coins dimes : ℕ) (value_dime value_nickel : ℝ) : ℝ :=
  let nickels := total_coins - dimes
  let value_dimes := dimes * value_dime
  let value_nickels := nickels * value_nickel
  value_dimes + value_nickels

theorem total_value_of_pile :
  value_of_pile 50 14 0.10 0.05 = 3.20 := by
  sorry

end total_value_of_pile_l90_90470


namespace Mr_A_Mrs_A_are_normal_l90_90146

def is_knight (person : Type) : Prop := sorry
def is_liar (person : Type) : Prop := sorry
def is_normal (person : Type) : Prop := sorry

variable (Mr_A Mrs_A : Type)

axiom Mr_A_statement : is_normal Mrs_A → False
axiom Mrs_A_statement : is_normal Mr_A → False

theorem Mr_A_Mrs_A_are_normal :
  is_normal Mr_A ∧ is_normal Mrs_A :=
sorry

end Mr_A_Mrs_A_are_normal_l90_90146


namespace intersection_eq_l90_90044

def setM : Set ℝ := { x | x^2 - 2*x < 0 }
def setN : Set ℝ := { x | -1 ≤ x ∧ x ≤ 1 }

theorem intersection_eq : setM ∩ setN = { x | 0 < x ∧ x ≤ 1 } := sorry

end intersection_eq_l90_90044


namespace simplify_expression_l90_90742

theorem simplify_expression : 
  (1 - 1 / (2 + Real.sqrt 5) + 1 / (2 - Real.sqrt 5) = 1) :=
by
  sorry

end simplify_expression_l90_90742


namespace num_even_divisors_of_8_l90_90207

def factorial (n : Nat) : Nat :=
  match n with
  | 0     => 1
  | Nat.succ n' => Nat.succ n' * factorial n'

-- Define the prime factorization of 8!
def prime_factors_eight_factorial : Nat := 2^7 * 3^2 * 5 * 7

-- Definition of an even divisor of 8!
def is_even_divisor (d : Nat) : Prop :=
  d ∣ prime_factors_eight_factorial ∧ 2 ∣ d

-- Calculation of number of even divisors of 8!
def num_even_divisors_8! : Nat :=
  7 * 3 * 2 * 2

theorem num_even_divisors_of_8! :
  num_even_divisors_8! = 84 :=
sorry

end num_even_divisors_of_8_l90_90207


namespace solve_inequality_l90_90632

theorem solve_inequality (x : ℝ) (h : x ≠ -1) : (2 - x) / (x + 1) ≥ 0 ↔ -1 < x ∧ x ≤ 2 :=
by
  sorry

end solve_inequality_l90_90632


namespace horner_v4_at_2_l90_90520

def horner (a : List Int) (x : Int) : Int :=
  a.foldr (fun ai acc => ai + x * acc) 0

noncomputable def poly_coeffs : List Int := [1, -12, 60, -160, 240, -192, 64]

theorem horner_v4_at_2 : horner poly_coeffs 2 = 80 := by
  sorry

end horner_v4_at_2_l90_90520


namespace roots_relationship_l90_90131

variable {a b c : ℝ} (h : a ≠ 0)

theorem roots_relationship (x y : ℝ) :
  (x = (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a) ∨ x = (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)) →
  (y = (-b + Real.sqrt (b^2 - 4*a*c)) / 2 ∨ y = (-b - Real.sqrt (b^2 - 4*a*c)) / 2) →
  (x = y / a) :=
by
  sorry

end roots_relationship_l90_90131


namespace arithmetic_sequence_sum_l90_90887

variable (a : ℕ → ℝ)
variable (d : ℝ)

axiom a2_a5 : a 2 + a 5 = 4
axiom a6_a9 : a 6 + a 9 = 20

theorem arithmetic_sequence_sum : a 4 + a 7 = 12 := by
  sorry

end arithmetic_sequence_sum_l90_90887


namespace inequality_proof_l90_90908

theorem inequality_proof {x y : ℝ} (h : x^4 + y^4 ≥ 2) : |x^12 - y^12| + 2 * x^6 * y^6 ≥ 2 := 
by sorry

end inequality_proof_l90_90908


namespace part_a_part_b_l90_90605

def g (n : ℕ) : ℕ := (n.digits 10).prod

theorem part_a : ∀ n : ℕ, g n ≤ n :=
by
  -- Proof omitted
  sorry

theorem part_b : {n : ℕ | n^2 - 12*n + 36 = g n} = {4, 9} :=
by
  -- Proof omitted
  sorry

end part_a_part_b_l90_90605


namespace double_acute_angle_l90_90112

theorem double_acute_angle (α : ℝ) (h : 0 < α ∧ α < π / 2) : 0 < 2 * α ∧ 2 * α < π :=
by
  sorry

end double_acute_angle_l90_90112


namespace find_n_in_range_l90_90285

theorem find_n_in_range : ∃ n, 5 ≤ n ∧ n ≤ 10 ∧ n ≡ 10543 [MOD 7] ∧ n = 8 := 
by
  sorry

end find_n_in_range_l90_90285


namespace avg_temp_l90_90729

theorem avg_temp (M T W Th F : ℝ) (h1 : M = 41) (h2 : F = 33) (h3 : (T + W + Th + F) / 4 = 46) : 
  (M + T + W + Th) / 4 = 48 :=
by
  -- insert proof steps here
  sorry

end avg_temp_l90_90729


namespace experts_win_probability_l90_90249

noncomputable def probability_of_experts_winning (p : ℝ) (q : ℝ) (needed_expert_wins : ℕ) (needed_audience_wins : ℕ) : ℝ :=
  p ^ 4 + 4 * (p ^ 3 * q)

-- Probability values
def p : ℝ := 0.6
def q : ℝ := 1 - p

-- Number of wins needed
def needed_expert_wins : ℕ := 3
def needed_audience_wins : ℕ := 2

theorem experts_win_probability :
  probability_of_experts_winning p q needed_expert_wins needed_audience_wins = 0.4752 :=
by
  -- Proof would go here
  sorry

end experts_win_probability_l90_90249


namespace jenny_hours_left_l90_90061

theorem jenny_hours_left 
    (h_research : ℕ := 10)
    (h_proposal : ℕ := 2)
    (h_visual_aids : ℕ := 5)
    (h_editing : ℕ := 3)
    (h_total : ℕ := 25) :
    h_total - (h_research + h_proposal + h_visual_aids + h_editing) = 5 := by
  sorry

end jenny_hours_left_l90_90061


namespace percentage_less_l90_90076

theorem percentage_less (P T J : ℝ) (hT : T = 0.9375 * P) (hJ : J = 0.8 * T) : (P - J) / P * 100 = 25 :=
by
  sorry

end percentage_less_l90_90076


namespace parabola_min_distance_a_l90_90693

noncomputable def directrix_distance (P : Real × Real) (a : Real) : Real :=
abs (P.2 + 1 / (4 * a))

noncomputable def distance (P Q : Real × Real) : Real :=
Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem parabola_min_distance_a (a : Real) :
  (∀ (P : Real × Real), P.2 = a * P.1^2 → 
    distance P (2, 0) + directrix_distance P a = Real.sqrt 5) ↔ 
    a = 1 / 4 ∨ a = -1 / 4 :=
by
  sorry

end parabola_min_distance_a_l90_90693


namespace expressions_not_equal_l90_90145

theorem expressions_not_equal (x : ℝ) (hx : x > 0) : 
  3 * x^x ≠ 2 * x^x + x^(2 * x) ∧ 
  x^(3 * x) ≠ 2 * x^x + x^(2 * x) ∧ 
  (3 * x)^x ≠ 2 * x^x + x^(2 * x) ∧ 
  (3 * x)^(3 * x) ≠ 2 * x^x + x^(2 * x) :=
by 
  sorry

end expressions_not_equal_l90_90145


namespace solution_set_l90_90536

noncomputable def f : ℝ → ℝ := sorry

axiom f_cond1 : ∀ x : ℝ, f x + deriv f x > 1
axiom f_cond2 : f 0 = 4

theorem solution_set (x : ℝ) : e^x * f x > e^x + 3 ↔ x > 0 :=
by sorry

end solution_set_l90_90536


namespace downstream_speed_is_40_l90_90284

variable (Vu : ℝ) (Vs : ℝ) (Vd : ℝ)

theorem downstream_speed_is_40 (h1 : Vu = 26) (h2 : Vs = 33) :
  Vd = 40 :=
by
  sorry

end downstream_speed_is_40_l90_90284


namespace largest_root_polynomial_intersection_l90_90697

/-
Given a polynomial P(x) = x^6 - 15x^5 + 74x^4 - 130x^3 + a * x^2 + b * x
and a line L(x) = c * x - 24,
such that P(x) stays above L(x) except at three distinct values of x where they intersect,
and one of the intersections is a root of triple multiplicity.
Prove that the largest value of x for which P(x) = L(x) is 6.
-/
theorem largest_root_polynomial_intersection (a b c : ℝ) (P L : ℝ → ℝ) (x : ℝ) :
  P x = x^6 - 15*x^5 + 74*x^4 - 130*x^3 + a*x^2 + b*x →
  L x = c*x - 24 →
  (∀ x, P x ≥ L x) ∨ (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ P x1 = L x1 ∧ P x2 = L x2 ∧ P x3 = L x3 ∧
  (∃ x0 : ℝ, x1 = x0 ∧ x2 = x0 ∧ x3 = x0 ∧ ∃ k : ℕ, k = 3)) →
  x = 6 :=
sorry

end largest_root_polynomial_intersection_l90_90697


namespace quadratic_polynomial_l90_90646

noncomputable def p (x : ℝ) : ℝ := (14 * x^2 + 4 * x + 12) / 15

theorem quadratic_polynomial :
  p (-2) = 4 ∧ p 1 = 2 ∧ p 3 = 10 :=
by
  have : p (-2) = (14 * (-2 : ℝ) ^ 2 + 4 * (-2 : ℝ) + 12) / 15 := rfl
  have : p 1 = (14 * (1 : ℝ) ^ 2 + 4 * (1 : ℝ) + 12) / 15 := rfl
  have : p 3 = (14 * (3 : ℝ) ^ 2 + 4 * (3 : ℝ) + 12) / 15 := rfl
  -- You can directly state the equalities or keep track of the computation steps.
  sorry

end quadratic_polynomial_l90_90646


namespace caleb_puffs_to_mom_l90_90234

variable (initial_puffs : ℕ) (puffs_to_sister : ℕ) (puffs_to_grandmother : ℕ) (puffs_to_dog : ℕ)
variable (puffs_per_friend : ℕ) (friends : ℕ)

theorem caleb_puffs_to_mom
  (h1 : initial_puffs = 40) 
  (h2 : puffs_to_sister = 3)
  (h3 : puffs_to_grandmother = 5) 
  (h4 : puffs_to_dog = 2) 
  (h5 : puffs_per_friend = 9)
  (h6 : friends = 3)
  : initial_puffs - ( friends * puffs_per_friend + puffs_to_sister + puffs_to_grandmother + puffs_to_dog ) = 3 :=
by
  sorry

end caleb_puffs_to_mom_l90_90234


namespace length_at_4kg_length_increases_by_2_relationship_linear_length_at_12kg_l90_90117

noncomputable def spring_length (x : ℝ) : ℝ :=
  2 * x + 18

-- Problem (1)
theorem length_at_4kg : (spring_length 4) = 26 :=
  by
    -- The complete proof is omitted.
    sorry

-- Problem (2)
theorem length_increases_by_2 : ∀ (x y : ℝ), y = x + 1 → (spring_length y) = (spring_length x) + 2 :=
  by
    -- The complete proof is omitted.
    sorry

-- Problem (3)
theorem relationship_linear : ∃ (k b : ℝ), (∀ x, spring_length x = k * x + b) ∧ k = 2 ∧ b = 18 :=
  by
    -- The complete proof is omitted.
    sorry

-- Problem (4)
theorem length_at_12kg : (spring_length 12) = 42 :=
  by
    -- The complete proof is omitted.
    sorry

end length_at_4kg_length_increases_by_2_relationship_linear_length_at_12kg_l90_90117


namespace min_b_geometric_sequence_l90_90449

theorem min_b_geometric_sequence (a b c : ℝ) (h_geom : b^2 = a * c) (h_1_4 : (a = 1 ∨ b = 1 ∨ c = 1) ∧ (a = 4 ∨ b = 4 ∨ c = 4)) :
  b ≥ -2 ∧ (∃ b', b' < b → b' ≥ -2) :=
by {
  sorry -- Proof required
}

end min_b_geometric_sequence_l90_90449


namespace circle_equation_l90_90178

theorem circle_equation (a b r : ℝ) 
    (h₁ : b = -4 * a)
    (h₂ : abs (a + b - 1) / Real.sqrt 2 = r)
    (h₃ : (b + 2) / (a - 3) * (-1) = -1)
    (h₄ : a = 1)
    (h₅ : b = -4)
    (h₆ : r = 2 * Real.sqrt 2) :
    ∀ x y: ℝ, (x - 1) ^ 2 + (y + 4) ^ 2 = 8 := 
by
  intros
  sorry

end circle_equation_l90_90178


namespace product_of_numbers_l90_90352

theorem product_of_numbers (a b c m : ℚ) (h_sum : a + b + c = 240)
    (h_m_a : 6 * a = m) (h_m_b : m = b - 12) (h_m_c : m = c + 12) :
    a * b * c = 490108320 / 2197 :=
by 
  sorry

end product_of_numbers_l90_90352


namespace remainder_when_divided_by_19_l90_90511

theorem remainder_when_divided_by_19 {N : ℤ} (h : N % 342 = 47) : N % 19 = 9 :=
sorry

end remainder_when_divided_by_19_l90_90511


namespace weight_of_each_hardcover_book_l90_90652

theorem weight_of_each_hardcover_book
  (weight_limit : ℕ := 80)
  (hardcover_books : ℕ := 70)
  (textbooks : ℕ := 30)
  (knick_knacks : ℕ := 3)
  (textbook_weight : ℕ := 2)
  (knick_knack_weight : ℕ := 6)
  (over_weight : ℕ := 33)
  (total_weight : ℕ := hardcover_books * x + textbooks * textbook_weight + knick_knacks * knick_knack_weight)
  (weight_eq : total_weight = weight_limit + over_weight) :
  x = 1 / 2 :=
by {
  sorry
}

end weight_of_each_hardcover_book_l90_90652


namespace fraction_of_airing_time_spent_on_commercials_l90_90758

theorem fraction_of_airing_time_spent_on_commercials 
  (num_programs : ℕ) (minutes_per_program : ℕ) (total_commercial_time : ℕ) 
  (h1 : num_programs = 6) (h2 : minutes_per_program = 30) (h3 : total_commercial_time = 45) : 
  (total_commercial_time : ℚ) / (num_programs * minutes_per_program : ℚ) = 1 / 4 :=
by {
  -- The proof is omitted here as only the statement is required according to the instruction.
  sorry
}

end fraction_of_airing_time_spent_on_commercials_l90_90758


namespace compute_B_93_l90_90877

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![1, 0, 0], ![0, 0, -1], ![0, 1, 0]]

theorem compute_B_93 : B^93 = B := by
  sorry

end compute_B_93_l90_90877


namespace max_value_x_y3_z4_l90_90523

theorem max_value_x_y3_z4 (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (hxyz : x + y + z = 2) :
  x + y^3 + z^4 ≤ 2 :=
by
  sorry

end max_value_x_y3_z4_l90_90523


namespace probability_of_blue_ball_l90_90128

theorem probability_of_blue_ball 
(P_red P_yellow P_blue : ℝ) 
(h_red : P_red = 0.48)
(h_yellow : P_yellow = 0.35) 
(h_prob : P_red + P_yellow + P_blue = 1) 
: P_blue = 0.17 := 
sorry

end probability_of_blue_ball_l90_90128


namespace min_a_value_l90_90671

theorem min_a_value (a : ℝ) : 
  (∀ x : ℝ, (0 < x ∧ x ≤ 1/2) → x^2 + a * x + 1 ≥ 0) → a ≥ -5/2 := 
sorry

end min_a_value_l90_90671


namespace find_percentage_l90_90256

variable (X P : ℝ)

theorem find_percentage (h₁ : 0.20 * X = 400) (h₂ : (P / 100) * X = 2400) : P = 120 :=
by
  -- The proof is intentionally left out
  sorry

end find_percentage_l90_90256


namespace find_intersection_point_l90_90135

theorem find_intersection_point :
  ∃ (x y z : ℝ), 
    ((∃ t : ℝ, x = 1 + 2 * t ∧ y = 1 - t ∧ z = -2 + 3 * t) ∧ 
    (4 * x + 2 * y - z - 11 = 0)) ∧ 
    (x = 3 ∧ y = 0 ∧ z = 1) :=
by
  sorry

end find_intersection_point_l90_90135


namespace max_value_l90_90658

noncomputable def f (x y : ℝ) : ℝ := 8 * x ^ 2 + 9 * x * y + 18 * y ^ 2 + 2 * x + 3 * y
noncomputable def g (x y : ℝ) : Prop := 4 * x ^ 2 + 9 * y ^ 2 = 8

theorem max_value : ∃ x y : ℝ, g x y ∧ f x y = 26 :=
by
  sorry

end max_value_l90_90658


namespace simplify_expression_l90_90618

open Complex

theorem simplify_expression :
  ((4 + 6 * I) / (4 - 6 * I) * (4 - 6 * I) / (4 + 6 * I) + (4 - 6 * I) / (4 + 6 * I) * (4 + 6 * I) / (4 - 6 * I)) = 2 :=
by
  sorry

end simplify_expression_l90_90618


namespace solve_ineq_case1_solve_ineq_case2_l90_90211

theorem solve_ineq_case1 {a x : ℝ} (ha_pos : 0 < a) (ha_lt_one : a < 1) : 
  a^(x + 5) < a^(4 * x - 1) ↔ x < 2 :=
sorry

theorem solve_ineq_case2 {a x : ℝ} (ha_gt_one : a > 1) : 
  a^(x + 5) < a^(4 * x - 1) ↔ x > 2 :=
sorry

end solve_ineq_case1_solve_ineq_case2_l90_90211


namespace line_of_symmetry_l90_90600

-- Definitions of the circles and the line
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 9
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4 * x + 4 * y - 1 = 0
def line (x y : ℝ) : Prop := x - y - 2 = 0

-- The theorem stating the symmetry condition
theorem line_of_symmetry :
  ∀ (x y : ℝ), circle1 x y ↔ ∃ (x' y' : ℝ), line ((x + x') / 2) ((y + y') / 2) ∧ circle2 x' y' :=
sorry

end line_of_symmetry_l90_90600


namespace solve_fraction_l90_90180

theorem solve_fraction :
  (144^2 - 100^2) / 22 = 488 := 
by 
  sorry

end solve_fraction_l90_90180


namespace y_intercept_of_line_l90_90897

theorem y_intercept_of_line (m x y b : ℝ) (h1 : m = 4) (h2 : x = 50) (h3 : y = 300) (h4 : y = m * x + b) : b = 100 := by
  sorry

end y_intercept_of_line_l90_90897


namespace total_cost_of_tennis_balls_l90_90472

theorem total_cost_of_tennis_balls
  (packs : ℕ) (balls_per_pack : ℕ) (cost_per_ball : ℕ)
  (h1 : packs = 4) (h2 : balls_per_pack = 3) (h3 : cost_per_ball = 2) : 
  packs * balls_per_pack * cost_per_ball = 24 := by
  sorry

end total_cost_of_tennis_balls_l90_90472


namespace bus_capacity_fraction_l90_90064

theorem bus_capacity_fraction
  (capacity : ℕ)
  (x : ℚ)
  (return_fraction : ℚ)
  (total_people : ℕ)
  (capacity_eq : capacity = 200)
  (return_fraction_eq : return_fraction = 4/5)
  (total_people_eq : total_people = 310)
  (people_first_trip_eq : 200 * x + 200 * 4/5 = 310) :
  x = 3/4 :=
by
  sorry

end bus_capacity_fraction_l90_90064


namespace student_ticket_price_l90_90616

-- Define the conditions
variables (S T : ℝ)
def condition1 := 4 * S + 3 * T = 79
def condition2 := 12 * S + 10 * T = 246

-- Prove that the price of a student ticket is 9 dollars, given the equations above
theorem student_ticket_price (h1 : condition1 S T) (h2 : condition2 S T) : T = 9 :=
sorry

end student_ticket_price_l90_90616


namespace sum_of_possible_values_of_a_l90_90679

theorem sum_of_possible_values_of_a :
  (∀ r s : ℤ, r + s = a ∧ r * s = 3 * a) → ∃ a : ℤ, (a = 12) :=
by
  sorry

end sum_of_possible_values_of_a_l90_90679


namespace playground_total_l90_90875

def boys : ℕ := 44
def girls : ℕ := 53

theorem playground_total : boys + girls = 97 := by
  sorry

end playground_total_l90_90875


namespace infinite_triangles_with_conditions_l90_90786

theorem infinite_triangles_with_conditions :
  ∃ (A : ℝ) (B : ℝ) (C : ℝ), 
  (A > 0) ∧ (B > 0) ∧ (C > 0) ∧ (B - A = 2) ∧ (C = 4) ∧ 
  (Δ > 0) := sorry

end infinite_triangles_with_conditions_l90_90786


namespace find_s_l90_90720

def is_monic_cubic (p : Polynomial ℝ) : Prop :=
  p.degree = 3 ∧ p.leadingCoeff = 1

def has_roots (p : Polynomial ℝ) (roots : Set ℝ) : Prop :=
  ∀ x ∈ roots, p.eval x = 0

def poly_condition (f g : Polynomial ℝ) (s : ℝ) : Prop :=
  ∀ x : ℝ, f.eval x - g.eval x = 2 * s

theorem find_s (s : ℝ)
  (f g : Polynomial ℝ)
  (hf_monic : is_monic_cubic f)
  (hg_monic : is_monic_cubic g)
  (hf_roots : has_roots f {s + 2, s + 6})
  (hg_roots : has_roots g {s + 4, s + 10})
  (h_condition : poly_condition f g s) :
  s = 10.67 :=
sorry

end find_s_l90_90720


namespace sin_of_2000_deg_l90_90750

theorem sin_of_2000_deg (a : ℝ) (h : Real.tan (160 * Real.pi / 180) = a) : 
  Real.sin (2000 * Real.pi / 180) = -a / Real.sqrt (1 + a^2) := 
by
  sorry

end sin_of_2000_deg_l90_90750


namespace add_base_12_l90_90433

theorem add_base_12 :
  let a := 5*12^2 + 1*12^1 + 8*12^0
  let b := 2*12^2 + 7*12^1 + 6*12^0
  let result := 7*12^2 + 9*12^1 + 2*12^0
  a + b = result :=
by
  -- Placeholder for the actual proof
  sorry

end add_base_12_l90_90433


namespace smallest_three_digit_number_multiple_of_conditions_l90_90303

theorem smallest_three_digit_number_multiple_of_conditions :
  ∃ x : ℕ, 100 ≤ x ∧ x ≤ 999 ∧
  (x % 2 = 0) ∧ ((x + 1) % 3 = 0) ∧ ((x + 2) % 4 = 0) ∧ ((x + 3) % 5 = 0) ∧ ((x + 4) % 6 = 0) 
  ∧ x = 122 := 
by
  sorry

end smallest_three_digit_number_multiple_of_conditions_l90_90303


namespace average_weight_a_b_l90_90809

variables (A B C : ℝ)

theorem average_weight_a_b (h1 : (A + B + C) / 3 = 43)
                          (h2 : (B + C) / 2 = 43)
                          (h3 : B = 37) :
                          (A + B) / 2 = 40 :=
by
  sorry

end average_weight_a_b_l90_90809


namespace element_in_set_l90_90462

variable (A : Set ℕ) (a b : ℕ)
def condition : Prop := A = {a, b, 1}

theorem element_in_set (h : condition A a b) : 1 ∈ A :=
by sorry

end element_in_set_l90_90462


namespace polynomial_roots_fraction_sum_l90_90042

theorem polynomial_roots_fraction_sum (a b c : ℝ) 
  (h1 : a + b + c = 12) 
  (h2 : ab + ac + bc = 20) 
  (h3 : abc = 3) : 
  (1 / a^2) + (1 / b^2) + (1 / c^2) = 328 / 9 := 
by 
  sorry

end polynomial_roots_fraction_sum_l90_90042


namespace division_by_negative_divisor_l90_90351

theorem division_by_negative_divisor : 15 / (-3) = -5 :=
by sorry

end division_by_negative_divisor_l90_90351


namespace small_branches_per_branch_l90_90204

theorem small_branches_per_branch (x : ℕ) (h1 : 1 + x + x^2 = 57) : x = 7 :=
by {
  sorry
}

end small_branches_per_branch_l90_90204


namespace x_squared_plus_y_squared_l90_90998

theorem x_squared_plus_y_squared (x y : ℝ) (h1 : x * y = 3) (h2 : (x - y) ^ 2 = 9) : 
  x ^ 2 + y ^ 2 = 15 := sorry

end x_squared_plus_y_squared_l90_90998


namespace find_a_for_chord_length_l90_90377

theorem find_a_for_chord_length :
  ∀ a : ℝ, ((∃ x y : ℝ, (x - 1)^2 + (y - 1)^2 = 1 ∧ (2 * x - y + a = 0)) 
  → ((2 * 1 - 1 + a = 0) → a = -1)) :=
by
  sorry

end find_a_for_chord_length_l90_90377


namespace evaluate_expression_l90_90800

variable (a : ℝ)
variable (x : ℝ)

theorem evaluate_expression (h : x = a + 9) : x - a + 6 = 15 := by
  sorry

end evaluate_expression_l90_90800


namespace proof_minimum_value_l90_90235

noncomputable def minimum_value_inequality (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2 * b = 1) : Prop :=
  (1 / a + a / b) ≥ 1 + 2 * Real.sqrt 2

theorem proof_minimum_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2 * b = 1) : minimum_value_inequality a b h1 h2 h3 :=
  by
    sorry

end proof_minimum_value_l90_90235


namespace lindy_total_distance_l90_90581

-- Definitions derived from the conditions
def jack_speed : ℕ := 5
def christina_speed : ℕ := 7
def lindy_speed : ℕ := 12
def initial_distance : ℕ := 360

theorem lindy_total_distance :
  lindy_speed * (initial_distance / (jack_speed + christina_speed)) = 360 := by
  sorry

end lindy_total_distance_l90_90581


namespace Bennett_sales_l90_90510

-- Define the variables for the number of screens sold in each month.
variables (J F M : ℕ)

-- State the given conditions.
theorem Bennett_sales (h1: F = 2 * J) (h2: F = M / 4) (h3: M = 8800) :
  J + F + M = 12100 := by
sorry

end Bennett_sales_l90_90510


namespace find_pos_real_nums_l90_90820

theorem find_pos_real_nums (x y z a b c : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z):
  (x + y + z = a + b + c) ∧ (4 * x * y * z = a^2 * x + b^2 * y + c^2 * z + a * b * c) →
  (a = y + z - x ∧ b = z + x - y ∧ c = x + y - z) :=
by
  sorry

end find_pos_real_nums_l90_90820


namespace area_of_square_field_l90_90124

theorem area_of_square_field (side_length : ℕ) (h : side_length = 25) :
  side_length * side_length = 625 := by
  sorry

end area_of_square_field_l90_90124


namespace equation_solutions_l90_90667

theorem equation_solutions
  (a : ℝ) :
  (∃ x : ℝ, (1 < a ∧ a < 2) ∧ (x = (1 - a) / a ∨ x = -1)) ∨
  (a = 2 ∧ (∃ x : ℝ, x = -1 ∨ x = -1/2)) ∨
  (a > 2 ∧ (∃ x : ℝ, x = (1 - a) / a ∨ x = -1 ∨ x = 1 - a)) ∨
  (0 ≤ a ∧ a ≤ 1 ∧ (∃ x : ℝ, x = -1)) ∨
  (a < 0 ∧ (∃ x : ℝ, x = (1 - a) / a ∨ x = -1)) := sorry

end equation_solutions_l90_90667


namespace simplify_expr_l90_90636

theorem simplify_expr (x : ℝ) : (3 * x)^5 + (4 * x) * (x^4) = 247 * x^5 :=
by
  sorry

end simplify_expr_l90_90636


namespace picture_area_l90_90173

theorem picture_area (x y : ℕ) (hx : x > 1) (hy : y > 1) 
  (h : (2 * x + 5) * (y + 4) - x * y = 84) : x * y = 15 :=
by
  sorry

end picture_area_l90_90173


namespace find_f_neg_one_l90_90104

theorem find_f_neg_one (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_symm : ∀ x, f (4 - x) = -f x)
  (h_f3 : f 3 = 3) :
  f (-1) = 3 := 
sorry

end find_f_neg_one_l90_90104


namespace pyramid_volume_l90_90619

theorem pyramid_volume (S : ℝ) :
  ∃ (V : ℝ),
  (∀ (a b h : ℝ), S = a * b ∧
  h = a * (Real.tan (60 * (Real.pi / 180))) ∧
  h = b * (Real.tan (30 * (Real.pi / 180))) ∧
  V = (1/3) * S * h) →
  V = (S * Real.sqrt S) / 3 :=
by
  sorry

end pyramid_volume_l90_90619


namespace tom_caught_16_trout_l90_90317

theorem tom_caught_16_trout (melanie_trout : ℕ) (tom_caught_twice : melanie_trout * 2 = 16) : 
  2 * melanie_trout = 16 :=
by 
  sorry

end tom_caught_16_trout_l90_90317


namespace original_average_l90_90721

theorem original_average (A : ℝ) (h : (10 * A = 70)) : A = 7 :=
sorry

end original_average_l90_90721


namespace min_episodes_to_watch_l90_90890

theorem min_episodes_to_watch (T W H F Sa Su M trip_days total_episodes: ℕ)
  (hW: W = 1) (hTh: H = 1) (hF: F = 1) (hSa: Sa = 2) (hSu: Su = 2) (hMo: M = 0)
  (total_episodes_eq: total_episodes = 60)
  (trip_days_eq: trip_days = 17):
  total_episodes - ((4 * W + 2 * Sa + 1 * M) * (trip_days / 7) + (trip_days % 7) * (W + Sa + Su + Mo)) = 39 := 
by
  sorry

end min_episodes_to_watch_l90_90890


namespace stratified_sampling_girls_count_l90_90749

theorem stratified_sampling_girls_count :
  (boys girls sampleSize totalSample : ℕ) →
  boys = 36 →
  girls = 18 →
  sampleSize = 6 →
  totalSample = boys + girls →
  (sampleSize * girls) / totalSample = 2 :=
by
  intros boys girls sampleSize totalSample h_boys h_girls h_sampleSize h_totalSample
  sorry

end stratified_sampling_girls_count_l90_90749


namespace find_y_l90_90803

-- Definitions of angles and the given problem.
def angle_ABC : ℝ := 90
def angle_ABD (y : ℝ) : ℝ := 3 * y
def angle_DBC (y : ℝ) : ℝ := 2 * y

-- The theorem stating the problem
theorem find_y (y : ℝ) (h1 : angle_ABC = 90) (h2 : angle_ABD y + angle_DBC y = angle_ABC) : y = 18 :=
  by 
  sorry

end find_y_l90_90803


namespace probability_same_outcomes_l90_90050

-- Let us define the event space for a fair coin
inductive CoinTossOutcome
| H : CoinTossOutcome
| T : CoinTossOutcome

open CoinTossOutcome

-- Definition of an event where the outcomes are the same (HHH or TTT)
def same_outcomes (t1 t2 t3 : CoinTossOutcome) : Prop :=
  (t1 = H ∧ t2 = H ∧ t3 = H) ∨ (t1 = T ∧ t2 = T ∧ t3 = T)

-- Number of all possible outcomes for three coin tosses
def total_outcomes : ℕ := 2 ^ 3

-- Number of favorable outcomes where all outcomes are the same
def favorable_outcomes : ℕ := 2

-- Calculation of probability
def prob_same_outcomes : ℚ := favorable_outcomes / total_outcomes

-- The statement to be proved in Lean 4
theorem probability_same_outcomes : prob_same_outcomes = 1 / 4 := 
by sorry

end probability_same_outcomes_l90_90050


namespace calculate_fraction_pow_l90_90798

theorem calculate_fraction_pow :
  (0.5 ^ 4) / (0.05 ^ 3) = 500 := 
  sorry

end calculate_fraction_pow_l90_90798


namespace max_ab_sum_l90_90989

theorem max_ab_sum (a b: ℤ) (h1: a ≠ b) (h2: a * b = -132) (h3: a ≤ b): a + b = -1 :=
sorry

end max_ab_sum_l90_90989


namespace ana_additional_payment_l90_90593

theorem ana_additional_payment (A B L : ℝ) (h₁ : A < B) (h₂ : A < L) : 
  (A + (B + L - 2 * A) / 3 = ((A + B + L) / 3)) :=
by
  sorry

end ana_additional_payment_l90_90593


namespace eval_expr_l90_90004

theorem eval_expr : 8^3 + 8^3 + 8^3 + 8^3 - 2^6 * 2^3 = 1536 := by
  -- Proof will go here
  sorry

end eval_expr_l90_90004


namespace bill_due_in_months_l90_90448

noncomputable def true_discount_time (TD A R : ℝ) : ℝ :=
  let P := A - TD
  let T := TD / (P * R / 100)
  12 * T

theorem bill_due_in_months :
  ∀ (TD A R : ℝ), TD = 189 → A = 1764 → R = 16 →
  abs (true_discount_time TD A R - 10.224) < 1 :=
by
  intros TD A R hTD hA hR
  sorry

end bill_due_in_months_l90_90448


namespace minimal_value_expression_l90_90648

theorem minimal_value_expression (a b c : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : a + b + c = 1) :
  (a + (ab)^(1/3) + (abc)^(1/4)) ≥ (1/3 + 1/(3 * (3^(1/3))) + 1/(3 * (3^(1/4)))) :=
sorry

end minimal_value_expression_l90_90648


namespace sequence_arithmetic_l90_90945

theorem sequence_arithmetic (a : ℕ → Real)
    (h₁ : a 3 = 2)
    (h₂ : a 7 = 1)
    (h₃ : ∃ d, ∀ n, 1 / (1 + a (n + 1)) = 1 / (1 + a n) + d):
    a 11 = 1 / 2 := by
  sorry

end sequence_arithmetic_l90_90945


namespace determinant_evaluation_l90_90027

theorem determinant_evaluation (x z : ℝ) :
  (Matrix.det ![
    ![1, x, z],
    ![1, x + z, z],
    ![1, x, x + z]
  ]) = x * z - z * z := 
sorry

end determinant_evaluation_l90_90027


namespace find_radius_l90_90313

noncomputable def square_radius (r : ℝ) : Prop :=
  let s := (2 * r) / Real.sqrt 2  -- side length of the square derived from the radius
  let perimeter := 4 * s         -- perimeter of the square
  let area := Real.pi * r^2      -- area of the circumscribed circle
  perimeter = area               -- given condition

theorem find_radius (r : ℝ) (h : square_radius r) : r = (4 * Real.sqrt 2) / Real.pi :=
by
  sorry

end find_radius_l90_90313


namespace max_distance_eq_of_l1_l90_90029

noncomputable def equation_of_l1 (l1 l2 : ℝ → ℝ) (A B : ℝ × ℝ) : Prop :=
  A = (1, 3) ∧ B = (2, 4) ∧ -- Points A and B
  l1 A.1 = A.2 ∧ l2 B.1 = B.2 ∧ -- l1 passes through A and l2 passes through B
  (∀ (x : ℝ), l1 x - l2 x = 1) ∧ -- l1 and l2 are parallel (constant difference in y-values)
  (∃ (c : ℝ), ∀ (x : ℝ), l1 x = -x + c ∧ l2 x = -x + c + 1) -- distance maximized

theorem max_distance_eq_of_l1 : 
  ∃ (l1 l2 : ℝ → ℝ), equation_of_l1 l1 l2 (1, 3) (2, 4) ∧
  ∀ (x : ℝ), l1 x = -x + 4 := 
sorry

end max_distance_eq_of_l1_l90_90029


namespace circumscribed_circle_radius_l90_90445

variables (A B C : ℝ) (a b c : ℝ) (R : ℝ) (area : ℝ)

-- Given conditions
def sides_ratio := a / b = 7 / 5 ∧ b / c = 5 / 3
def triangle_area := area = 45 * Real.sqrt 3
def sides := (a, b, c)
def angles := (A, B, C)

-- Prove radius
theorem circumscribed_circle_radius 
  (h_ratio : sides_ratio a b c)
  (h_area : triangle_area area) :
  R = 14 :=
sorry

end circumscribed_circle_radius_l90_90445


namespace s_6_of_30_eq_146_over_175_l90_90950

def s (θ : ℚ) : ℚ := 1 / (2 - θ)

theorem s_6_of_30_eq_146_over_175 : s (s (s (s (s (s 30))))) = 146 / 175 := sorry

end s_6_of_30_eq_146_over_175_l90_90950


namespace tara_marbles_modulo_l90_90455

noncomputable def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem tara_marbles_modulo : 
  let B := 6  -- number of blue marbles
  let Y := 12  -- additional yellow marbles for balance
  let total_marbles := B + Y  -- total marbles
  let arrangements := binom (total_marbles + B) B
  let N := arrangements
  N % 1000 = 564 :=
by
  let B := 6  -- number of blue marbles
  let Y := 12  -- additional yellow marbles for balance
  let total_marbles := B + Y  -- total marbles
  let arrangements := binom (total_marbles + B) B
  let N := arrangements
  have : N % 1000 = 564 := sorry
  exact this

end tara_marbles_modulo_l90_90455


namespace sum_reciprocals_factors_12_l90_90894

theorem sum_reciprocals_factors_12 :
  (1:ℚ) + (1/2) + (1/3) + (1/4) + (1/6) + (1/12) = (7/3:ℚ) := 
by
  sorry

end sum_reciprocals_factors_12_l90_90894


namespace min_throws_to_same_sum_l90_90087

/-- Define the set of possible sums for four six-sided dice --/
def dice_sum_range := {s : ℕ | 4 ≤ s ∧ s ≤ 24}

/-- The total number of possible sums when rolling four six-sided dice --/
def num_possible_sums : ℕ := 24 - 4 + 1

/-- 
  The minimum number of throws required to ensure that the same sum appears at least twice 
  by the Pigeonhole principle.
--/
theorem min_throws_to_same_sum : num_possible_sums + 1 = 22 := by
  sorry

end min_throws_to_same_sum_l90_90087


namespace min_distance_exists_l90_90238

open Real

-- Define the distance formula function
noncomputable def distance (x : ℝ) : ℝ :=
sqrt ((x - 1) ^ 2 + (3 - 2 * x) ^ 2 + (3 * x - 3) ^ 2)

theorem min_distance_exists :
  ∃ (x : ℝ), distance x = sqrt (14 * x^2 - 32 * x + 19) ∧
               ∀ y, distance y ≥ (sqrt 35) / 7 :=
sorry

end min_distance_exists_l90_90238


namespace certain_number_N_l90_90062

theorem certain_number_N (G N : ℕ) (hG : G = 127)
  (h₁ : ∃ k : ℕ, N = G * k + 10)
  (h₂ : ∃ m : ℕ, 2045 = G * m + 13) :
  N = 2042 :=
sorry

end certain_number_N_l90_90062


namespace avg_of_second_largest_and_second_smallest_is_eight_l90_90398

theorem avg_of_second_largest_and_second_smallest_is_eight :
  ∀ (a b c d e : ℕ), 
  a + b + c + d + e = 40 → 
  a < b ∧ b < c ∧ c < d ∧ d < e →
  ((d + b) / 2 : ℕ) = 8 := 
by
  intro a b c d e hsum horder
  /- the proof goes here, but we use sorry to skip it -/
  sorry

end avg_of_second_largest_and_second_smallest_is_eight_l90_90398


namespace total_fish_l90_90644

theorem total_fish (x y : ℕ) : (19 - 2 * x) + (27 - 4 * y) = 46 - 2 * x - 4 * y :=
  by
    sorry

end total_fish_l90_90644


namespace average_rate_of_interest_l90_90057

theorem average_rate_of_interest (total_investment : ℝ) (rate1 rate2 average_rate : ℝ) (amount1 amount2 : ℝ)
  (H1 : total_investment = 6000)
  (H2 : rate1 = 0.03)
  (H3 : rate2 = 0.07)
  (H4 : average_rate = 0.042)
  (H5 : amount1 + amount2 = total_investment)
  (H6 : rate1 * amount1 = rate2 * amount2) :
  (rate1 * amount1 + rate2 * amount2) / total_investment = average_rate := 
sorry

end average_rate_of_interest_l90_90057


namespace find_time_when_velocity_is_one_l90_90428

-- Define the equation of motion
def equation_of_motion (t : ℝ) : ℝ := 7 * t^2 + 8

-- Define the velocity function as the derivative of the equation of motion
def velocity (t : ℝ) : ℝ := by
  let s := equation_of_motion t
  exact 14 * t  -- Since we calculated the derivative above

-- Statement of the problem to be proved
theorem find_time_when_velocity_is_one : 
  (velocity (1 / 14)) = 1 :=
by
  -- Placeholder for the proof
  sorry

end find_time_when_velocity_is_one_l90_90428


namespace a_in_M_l90_90214

def M : Set ℝ := { x | x ≤ 5 }
def a : ℝ := 2

theorem a_in_M : a ∈ M :=
by
  -- Proof omitted
  sorry

end a_in_M_l90_90214


namespace fractions_equal_l90_90443

theorem fractions_equal (x y z : ℝ) (hx1 : x ≠ 1) (hy1 : y ≠ 1) (hxy : x ≠ y)
  (h : (yz - x^2) / (1 - x) = (xz - y^2) / (1 - y)) : (yz - x^2) / (1 - x) = x + y + z ∧ (xz - y^2) / (1 - y) = x + y + z :=
sorry

end fractions_equal_l90_90443


namespace kevin_eggs_l90_90700

theorem kevin_eggs : 
  ∀ (bonnie george cheryl kevin : ℕ),
  bonnie = 13 → 
  george = 9 → 
  cheryl = 56 → 
  cheryl = bonnie + george + kevin + 29 →
  kevin = 5 :=
by
  intros bonnie george cheryl kevin h_bonnie h_george h_cheryl h_eqn
  subst h_bonnie
  subst h_george
  subst h_cheryl
  simp at h_eqn
  sorry

end kevin_eggs_l90_90700


namespace taxi_company_charges_l90_90554

theorem taxi_company_charges
  (X : ℝ)  -- charge for the first 1/5 of a mile
  (C : ℝ)  -- charge for each additional 1/5 of a mile
  (total_charge : ℝ)  -- total charge for an 8-mile ride
  (remaining_distance_miles : ℝ)  -- remaining miles after the first 1/5 mile
  (remaining_increments : ℝ)  -- remaining 1/5 mile increments
  (charge_increments : ℝ)  -- total charge for remaining increments
  (X_val : X = 2.50)
  (C_val : C = 0.40)
  (total_charge_val : total_charge = 18.10)
  (remaining_distance_miles_val : remaining_distance_miles = 7.8)
  (remaining_increments_val : remaining_increments = remaining_distance_miles * 5)
  (charge_increments_val : charge_increments = remaining_increments * C)
  (proof_1: charge_increments = 15.60)
  (proof_2: total_charge - charge_increments = X) : X = 2.50 := 
by
  sorry

end taxi_company_charges_l90_90554


namespace find_theta_l90_90436

theorem find_theta (Theta : ℕ) (h1 : 1 ≤ Theta ∧ Theta ≤ 9)
  (h2 : 294 / Theta = (30 + Theta) + 3 * Theta) : Theta = 6 :=
by sorry

end find_theta_l90_90436


namespace buckets_required_l90_90471

theorem buckets_required (C : ℕ) (h : C > 0) : 
  let original_buckets := 25
  let reduced_capacity := 2 / 5
  let total_capacity := original_buckets * C
  let new_buckets := total_capacity / ((2 / 5) * C)
  new_buckets = 63 := 
by
  sorry

end buckets_required_l90_90471


namespace geometric_sequence_sum_terms_l90_90833

noncomputable def geom_sequence_sum (a : ℕ → ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a 0 * (1 - q ^ n) / (1 - q)

theorem geometric_sequence_sum_terms
  (a : ℕ → ℝ) (q : ℝ)
  (h_q_nonzero : q ≠ 1)
  (S3_eq : geom_sequence_sum a q 3 = 8)
  (S6_eq : geom_sequence_sum a q 6 = 7)
  : a 6 * q ^ 6 + a 7 * q ^ 7 + a 8 * q ^ 8 = 1 / 8 := sorry

end geometric_sequence_sum_terms_l90_90833


namespace ladder_geometric_sequence_solution_l90_90063

-- A sequence {aₙ} is a 3rd-order ladder geometric sequence given by a_{n+3}^2 = a_n * a_{n+6} for any positive integer n
def ladder_geometric_3rd_order (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 3) ^ 2 = a n * a (n + 6)

-- Initial conditions
def initial_conditions (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ a 4 = 2

-- Main theorem to be proven in Lean 4
theorem ladder_geometric_sequence_solution :
  ∃ a : ℕ → ℝ, ladder_geometric_3rd_order a ∧ initial_conditions a ∧ a 10 = 8 :=
by
  sorry

end ladder_geometric_sequence_solution_l90_90063


namespace sum_of_coefficients_l90_90594

/-- If (2x - 1)^4 = a₄x^4 + a₃x^3 + a₂x^2 + a₁x + a₀, then the sum of the coefficients a₀ + a₁ + a₂ + a₃ + a₄ is 1. -/
theorem sum_of_coefficients :
  ∃ a₄ a₃ a₂ a₁ a₀ : ℝ, (2 * x - 1) ^ 4 = a₄ * x ^ 4 + a₃ * x ^ 3 + a₂ * x ^ 2 + a₁ * x + a₀ → 
  a₀ + a₁ + a₂ + a₃ + a₄ = 1 :=
by
  sorry

end sum_of_coefficients_l90_90594


namespace inequality_holds_l90_90766

theorem inequality_holds (a b : ℝ) : 
  a^2 + a * b + b^2 ≥ 3 * (a + b - 1) :=
sorry

end inequality_holds_l90_90766


namespace average_speed_stan_l90_90762

theorem average_speed_stan (d1 d2 : ℝ) (h1 h2 rest : ℝ) (total_distance total_time : ℝ) (avg_speed : ℝ) :
  d1 = 350 → 
  d2 = 400 → 
  h1 = 6 → 
  h2 = 7 → 
  rest = 0.5 → 
  total_distance = d1 + d2 → 
  total_time = h1 + h2 + rest → 
  avg_speed = total_distance / total_time → 
  avg_speed = 55.56 :=
by 
  intros h_d1 h_d2 h_h1 h_h2 h_rest h_total_distance h_total_time h_avg_speed
  sorry

end average_speed_stan_l90_90762


namespace original_cost_l90_90474

theorem original_cost (original_cost : ℝ) (h : 0.30 * original_cost = 588) : original_cost = 1960 :=
sorry

end original_cost_l90_90474


namespace find_S6_l90_90035

variable (a_n : ℕ → ℝ) -- Assume a_n gives the nth term of an arithmetic sequence.
variable (S_n : ℕ → ℝ) -- Assume S_n gives the sum of the first n terms of the sequence.

-- Conditions:
axiom S_2_eq : S_n 2 = 2
axiom S_4_eq : S_n 4 = 10

-- Define what it means to find S_6
theorem find_S6 : S_n 6 = 18 :=
by
  sorry

end find_S6_l90_90035


namespace value_of_p_l90_90956

noncomputable def p_value_condition (p q : ℝ) (h1 : p + q = 1) (h2 : p > 0) (h3 : q > 0) : Prop :=
  (9 * p^8 * q = 36 * p^7 * q^2)

theorem value_of_p (p q : ℝ) (h1 : p + q = 1) (h2 : p > 0) (h3 : q > 0) (h4 : p_value_condition p q h1 h2 h3) :
  p = 4 / 5 :=
by
  sorry

end value_of_p_l90_90956


namespace heads_at_least_twice_in_5_tosses_l90_90822

noncomputable def probability_at_least_two_heads (n : ℕ) (p : ℚ) : ℚ :=
1 - (n : ℚ) * p^(n : ℕ)

theorem heads_at_least_twice_in_5_tosses :
  probability_at_least_two_heads 5 (1/2) = 13/16 :=
by
  sorry

end heads_at_least_twice_in_5_tosses_l90_90822


namespace second_bounce_distance_correct_l90_90583

noncomputable def second_bounce_distance (R v g : ℝ) : ℝ := 2 * R - (2 * v / 3) * (Real.sqrt (R / g))

theorem second_bounce_distance_correct (R v g : ℝ) (hR : R > 0) (hv : v > 0) (hg : g > 0) :
  second_bounce_distance R v g = 2 * R - (2 * v / 3) * (Real.sqrt (R / g)) := 
by
  -- Placeholder for the proof
  sorry

end second_bounce_distance_correct_l90_90583


namespace max_projection_area_tetrahedron_l90_90009

-- Define the side length of the tetrahedron
variable (a : ℝ)

-- Define a theorem stating the maximum projection area of a tetrahedron
theorem max_projection_area_tetrahedron (h : a > 0) : 
  ∃ A, A = (a^2 / 2) :=
by
  -- Proof is omitted
  sorry

end max_projection_area_tetrahedron_l90_90009


namespace complex_quadrant_l90_90018

theorem complex_quadrant (z : ℂ) (h : (2 - I) * z = 1 + I) : 
  0 < z.re ∧ 0 < z.im := 
by 
  -- Proof will be provided here 
  sorry

end complex_quadrant_l90_90018


namespace smallest_n_value_l90_90939

-- Define the conditions as given in the problem
def num_birthdays := 365

-- Formulating the main statement
theorem smallest_n_value : ∃ (n : ℕ), (∀ (group_size : ℕ), group_size = 2 * n - 10 → group_size ≥ 3286) ∧ n = 1648 :=
by
  use 1648
  sorry

end smallest_n_value_l90_90939


namespace roman_remy_gallons_l90_90715

theorem roman_remy_gallons (R : ℕ) (Remy_uses : 3 * R + 1 = 25) :
  R + (3 * R + 1) = 33 :=
by
  sorry

end roman_remy_gallons_l90_90715


namespace jason_seashells_remaining_l90_90227

-- Define the initial number of seashells Jason found
def initial_seashells : ℕ := 49

-- Define the number of seashells Jason gave to Tim
def seashells_given_to_tim : ℕ := 13

-- Define the number of seashells Jason now has
def seashells_now : ℕ := initial_seashells - seashells_given_to_tim

-- The theorem to prove: 
theorem jason_seashells_remaining : seashells_now = 36 := 
by
  -- Proof steps will go here
  sorry

end jason_seashells_remaining_l90_90227


namespace min_sum_abc_l90_90023

theorem min_sum_abc (a b c : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_abc : a * b * c = 1020) : a + b + c = 33 :=
sorry

end min_sum_abc_l90_90023


namespace coupon_savings_inequalities_l90_90054

variable {P : ℝ} (p : ℝ) (hP : P = 150 + p) (hp_pos : p > 0)
variable (ha : 0.15 * P > 30) (hb : 0.15 * P > 0.20 * p)
variable (cA_saving : ℝ := 0.15 * P)
variable (cB_saving : ℝ := 30)
variable (cC_saving : ℝ := 0.20 * p)

theorem coupon_savings_inequalities (h1 : 0.15 * P - 30 > 0) (h2 : 0.15 * P - 0.20 * (P - 150) > 0) :
  let x := 200
  let y := 600
  y - x = 400 :=
by
  sorry

end coupon_savings_inequalities_l90_90054


namespace tan_ratio_l90_90653

theorem tan_ratio (p q : ℝ) 
  (h1: Real.sin (p + q) = 5 / 8)
  (h2: Real.sin (p - q) = 3 / 8) : Real.tan p / Real.tan q = 4 := 
by
  sorry

end tan_ratio_l90_90653


namespace class_gpa_l90_90924

theorem class_gpa (n : ℕ) (hn : n > 0) (gpa1 : ℝ := 30) (gpa2 : ℝ := 33) : 
    (gpa1 * (n:ℝ) + gpa2 * (2 * n : ℝ)) / (3 * n : ℝ) = 32 :=
by
  sorry

end class_gpa_l90_90924


namespace desk_chair_production_l90_90557

theorem desk_chair_production (x : ℝ) (h₁ : x > 0) (h₂ : 540 / x - 540 / (x + 2) = 3) : 
  ∃ x, 540 / x - 540 / (x + 2) = 3 := 
by
  sorry

end desk_chair_production_l90_90557


namespace count_consecutive_sets_sum_15_l90_90607

theorem count_consecutive_sets_sum_15 : 
  ∃ n : ℕ, 
    (n > 0 ∧
    ∃ a : ℕ, 
      (n ≥ 2 ∧ 
      ∃ s : (Finset ℕ), 
        (∀ x ∈ s, x ≥ 1) ∧ 
        (s.sum id = 15))
  ) → 
  n = 2 :=
  sorry

end count_consecutive_sets_sum_15_l90_90607


namespace factorize_expression_l90_90376

theorem factorize_expression (a : ℝ) : (a + 2) * (a - 2) - 3 * a = (a - 4) * (a + 1) :=
by
  sorry

end factorize_expression_l90_90376


namespace determine_g_l90_90323

noncomputable def g (x : ℝ) : ℝ := -4 * x^4 + 6 * x^3 - 9 * x^2 + 10 * x - 8

theorem determine_g (x : ℝ) : 
  4 * x^4 + 5 * x^2 - 2 * x + 7 + g x = 6 * x^3 - 4 * x^2 + 8 * x - 1 := by
  sorry

end determine_g_l90_90323


namespace fraction_equality_l90_90995

variables (x y : ℝ)

theorem fraction_equality (h : y / 2 = (2 * y - x) / 3) : y / x = 2 :=
sorry

end fraction_equality_l90_90995


namespace fifty_third_number_is_2_pow_53_l90_90406

theorem fifty_third_number_is_2_pow_53 :
  ∀ n : ℕ, (n = 53) → ∃ seq : ℕ → ℕ, (seq 1 = 2) ∧ (∀ k : ℕ, seq (k+1) = 2 * seq k) ∧ (seq n = 2 ^ 53) :=
  sorry

end fifty_third_number_is_2_pow_53_l90_90406


namespace number_of_toys_l90_90539

-- Definitions based on conditions
def selling_price : ℝ := 18900
def cost_price_per_toy : ℝ := 900
def gain_per_toy : ℝ := 3 * cost_price_per_toy

-- The number of toys sold
noncomputable def number_of_toys_sold (SP CP gain : ℝ) : ℝ :=
  (SP - gain) / CP

-- The theorem statement to prove
theorem number_of_toys (SP CP gain : ℝ) : number_of_toys_sold SP CP gain = 18 :=
by
  have h1: SP = 18900 := by sorry
  have h2: CP = 900 := by sorry
  have h3: gain = 3 * CP := by sorry
  -- Further steps to establish the proof
  sorry

end number_of_toys_l90_90539


namespace tiffany_initial_lives_l90_90435

theorem tiffany_initial_lives (x : ℕ) 
    (H1 : x - 14 + 27 = 56) : x = 43 :=
sorry

end tiffany_initial_lives_l90_90435


namespace prob_all_four_even_dice_l90_90960

noncomputable def probability_even (n : ℕ) : ℚ := (3 / 6)^n

theorem prob_all_four_even_dice : probability_even 4 = 1 / 16 := 
by
  sorry

end prob_all_four_even_dice_l90_90960


namespace area_BEIH_l90_90281

noncomputable def point (x y : ℝ) : ℝ × ℝ := (x, y)

noncomputable def area_quad (B E I H : ℝ × ℝ) : ℝ :=
  (1/2) * ((B.1 * E.2 + E.1 * I.2 + I.1 * H.2 + H.1 * B.2) - (B.2 * E.1 + E.2 * I.1 + I.2 * H.1 + H.2 * B.1))

theorem area_BEIH :
  let A : ℝ × ℝ := point 0 3
  let B : ℝ × ℝ := point 0 0
  let C : ℝ × ℝ := point 3 0
  let D : ℝ × ℝ := point 3 3
  let E : ℝ × ℝ := point 0 2
  let F : ℝ × ℝ := point 1 0
  let I : ℝ × ℝ := point (3/10) 2.1
  let H : ℝ × ℝ := point (3/4) (3/4)
  area_quad B E I H = 1.0125 :=
by
  sorry

end area_BEIH_l90_90281


namespace total_parents_surveyed_l90_90387

-- Define the given conditions
def percent_agree : ℝ := 0.20
def percent_disagree : ℝ := 0.80
def disagreeing_parents : ℕ := 640

-- Define the statement to prove
theorem total_parents_surveyed :
  ∃ (total_parents : ℕ), disagreeing_parents = (percent_disagree * total_parents) ∧ total_parents = 800 :=
by
  sorry

end total_parents_surveyed_l90_90387


namespace probability_painted_faces_l90_90744

theorem probability_painted_faces (total_cubes : ℕ) (corner_cubes : ℕ) (no_painted_face_cubes : ℕ) (successful_outcomes : ℕ) (total_outcomes : ℕ) 
  (probability : ℚ) : 
  total_cubes = 125 ∧ corner_cubes = 8 ∧ no_painted_face_cubes = 27 ∧ successful_outcomes = 216 ∧ total_outcomes = 7750 ∧ 
  probability = 72 / 2583 :=
by
  sorry

end probability_painted_faces_l90_90744


namespace nancy_potatoes_l90_90386

theorem nancy_potatoes (sandy_potatoes total_potatoes : ℕ) (h1 : sandy_potatoes = 7) (h2 : total_potatoes = 13) :
    total_potatoes - sandy_potatoes = 6 :=
by
  sorry

end nancy_potatoes_l90_90386


namespace shop_owner_pricing_l90_90961

theorem shop_owner_pricing (L C M S : ℝ)
  (h1 : C = 0.75 * L)
  (h2 : S = 1.3 * C)
  (h3 : S = 0.75 * M) : 
  M = 1.3 * L := 
sorry

end shop_owner_pricing_l90_90961


namespace planar_graph_edge_vertex_inequality_l90_90935

def planar_graph (G : Type _) : Prop := -- Placeholder for planar graph property
  sorry

variables {V E : ℕ}

theorem planar_graph_edge_vertex_inequality (G : Type _) (h : planar_graph G) :
  E ≤ 3 * V - 6 :=
sorry

end planar_graph_edge_vertex_inequality_l90_90935


namespace Rob_has_three_dimes_l90_90592

theorem Rob_has_three_dimes (quarters dimes nickels pennies : ℕ) 
                            (val_quarters val_nickels val_pennies : ℚ)
                            (total_amount : ℚ) :
  quarters = 7 →
  nickels = 5 →
  pennies = 12 →
  val_quarters = 0.25 →
  val_nickels = 0.05 →
  val_pennies = 0.01 →
  total_amount = 2.42 →
  (7 * 0.25 + 5 * 0.05 + 12 * 0.01 + dimes * 0.10 = total_amount) →
  dimes = 3 :=
by sorry

end Rob_has_three_dimes_l90_90592


namespace greatest_x_value_l90_90801

theorem greatest_x_value :
  ∃ x, (∀ y, (y ≠ 6) → (y^2 - y - 30) / (y - 6) = 2 / (y + 4) → y ≤ x) ∧ (x^2 - x - 30) / (x - 6) = 2 / (x + 4) ∧ x ≠ 6 ∧ x = -3 :=
sorry

end greatest_x_value_l90_90801


namespace mike_picked_12_pears_l90_90509

theorem mike_picked_12_pears
  (jason_pears : ℕ)
  (keith_pears : ℕ)
  (total_pears : ℕ)
  (H1 : jason_pears = 46)
  (H2 : keith_pears = 47)
  (H3 : total_pears = 105) :
  (total_pears - (jason_pears + keith_pears)) = 12 :=
by
  sorry

end mike_picked_12_pears_l90_90509


namespace role_of_scatter_plot_correct_l90_90575

-- Definitions for problem context
def role_of_scatter_plot (role : String) : Prop :=
  role = "Roughly judging whether variables are linearly related"

-- Problem and conditions
theorem role_of_scatter_plot_correct :
  role_of_scatter_plot "Roughly judging whether variables are linearly related" :=
by 
  sorry

end role_of_scatter_plot_correct_l90_90575


namespace five_coins_not_155_l90_90717

def coin_values : List ℕ := [5, 25, 50]

def can_sum_to (n : ℕ) (count : ℕ) : Prop :=
  ∃ (a b c : ℕ), a + b + c = count ∧ a * 5 + b * 25 + c * 50 = n

theorem five_coins_not_155 : ¬ can_sum_to 155 5 :=
  sorry

end five_coins_not_155_l90_90717


namespace sqrt_difference_l90_90330

theorem sqrt_difference : Real.sqrt 18 - Real.sqrt 8 = Real.sqrt 2 := 
by 
  sorry

end sqrt_difference_l90_90330


namespace neither_cable_nor_vcr_fraction_l90_90953

variable (T : ℕ) -- Let T be the total number of housing units

def cableTV_fraction : ℚ := 1 / 5
def VCR_fraction : ℚ := 1 / 10
def both_fraction_given_cable : ℚ := 1 / 4

theorem neither_cable_nor_vcr_fraction : 
  (T : ℚ) * (1 - ((1 / 5) + ((1 / 10) - ((1 / 4) * (1 / 5))))) = (T : ℚ) * (3 / 4) :=
by sorry

end neither_cable_nor_vcr_fraction_l90_90953


namespace cos_beta_l90_90174

variable (α β : ℝ)
variable (hα : 0 < α ∧ α < π / 2)
variable (hβ : 0 < β ∧ β < π / 2)
variable (h1 : Real.sin α = 3 / 5)
variable (h2 : Real.cos (α + β) = 5 / 13)

theorem cos_beta (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h1 : Real.sin α = 3 / 5) (h2 : Real.cos (α + β) = 5 / 13) : 
  Real.cos β = 56 / 65 := by
  sorry

end cos_beta_l90_90174


namespace money_weed_eating_l90_90490

-- Define the amounts and conditions
def money_mowing : ℕ := 68
def money_per_week : ℕ := 9
def weeks : ℕ := 9
def total_money : ℕ := money_per_week * weeks

-- Define the proof that the money made weed eating is 13 dollars
theorem money_weed_eating :
  total_money - money_mowing = 13 := sorry

end money_weed_eating_l90_90490


namespace sum_reciprocals_eq_three_l90_90364

-- Define nonzero real numbers x and y with their given condition
variables (x y : ℝ) (hx : x ≠ 0) (hy: y ≠ 0) (h : x + y = 3 * x * y)

-- State the theorem to prove the sum of reciprocals of x and y is 3
theorem sum_reciprocals_eq_three (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 3 * x * y) : (1 / x) + (1 / y) = 3 :=
sorry

end sum_reciprocals_eq_three_l90_90364


namespace combined_time_is_45_l90_90424

-- Definitions based on conditions
def Pulsar_time : ℕ := 10
def Polly_time : ℕ := 3 * Pulsar_time
def Petra_time : ℕ := (1 / 6 ) * Polly_time

-- Total combined time
def total_time : ℕ := Pulsar_time + Polly_time + Petra_time

-- Theorem to prove
theorem combined_time_is_45 : total_time = 45 := by
  sorry

end combined_time_is_45_l90_90424


namespace glass_bowls_sold_l90_90518

theorem glass_bowls_sold
  (BowlsBought : ℕ) (CostPricePerBowl SellingPricePerBowl : ℝ) (PercentageGain : ℝ)
  (CostPrice := BowlsBought * CostPricePerBowl)
  (SellingPrice : ℝ := (102 : ℝ) * SellingPricePerBowl)
  (gain := (SellingPrice - CostPrice) / CostPrice * 100) :
  PercentageGain = 8.050847457627118 →
  BowlsBought = 118 →
  CostPricePerBowl = 12 →
  SellingPricePerBowl = 15 →
  PercentageGain = gain →
  102 = 102 := by
  intro h1 h2 h3 h4 h5
  sorry

end glass_bowls_sold_l90_90518


namespace total_money_correct_l90_90862

def total_money_in_cents : ℕ :=
  let Cindy := 5 * 10 + 3 * 50
  let Eric := 3 * 25 + 2 * 100 + 1 * 50
  let Garrick := 8 * 5 + 7 * 1
  let Ivy := 60 * 1 + 5 * 25
  let TotalBeforeRemoval := Cindy + Eric + Garrick + Ivy
  let BeaumontRemoval := 2 * 10 + 3 * 5 + 10 * 1
  let EricRemoval := 1 * 25 + 1 * 50
  TotalBeforeRemoval - BeaumontRemoval - EricRemoval

theorem total_money_correct : total_money_in_cents = 637 := by
  sorry

end total_money_correct_l90_90862


namespace range_of_a2_div_a1_l90_90426

theorem range_of_a2_div_a1 (a_1 a_2 d : ℤ) : 
  1 ≤ a_1 ∧ a_1 ≤ 3 ∧ 
  a_2 = a_1 + d ∧ 
  6 ≤ 3 * a_1 + 2 * d ∧ 
  3 * a_1 + 2 * d ≤ 15 
  → (2 / 3 : ℚ) ≤ (a_2 : ℚ) / a_1 ∧ (a_2 : ℚ) / a_1 ≤ 5 :=
sorry

end range_of_a2_div_a1_l90_90426


namespace range_of_k_l90_90393

theorem range_of_k (k : ℝ) : (x^2 + k * y^2 = 2) ∧ (k > 0) ∧ (k < 1) ↔ (0 < k ∧ k < 1) :=
by
  sorry

end range_of_k_l90_90393


namespace total_books_to_read_l90_90871

theorem total_books_to_read (books_per_week : ℕ) (weeks : ℕ) (total_books : ℕ) 
  (h1 : books_per_week = 6) 
  (h2 : weeks = 5) 
  (h3 : total_books = books_per_week * weeks) : 
  total_books = 30 :=
by
  rw [h1, h2] at h3
  exact h3

end total_books_to_read_l90_90871


namespace class_size_l90_90835

theorem class_size
  (S_society : ℕ) (S_music : ℕ) (S_both : ℕ) (S : ℕ)
  (h_society : S_society = 25)
  (h_music : S_music = 32)
  (h_both : S_both = 27)
  (h_total : S = S_society + S_music - S_both) :
  S = 30 :=
by
  rw [h_society, h_music, h_both] at h_total
  exact h_total

end class_size_l90_90835


namespace uniq_increasing_seq_l90_90253

noncomputable def a (n : ℕ) : ℕ := n -- The correct sequence a_n = n

theorem uniq_increasing_seq (a : ℕ → ℕ)
  (h1 : a 2 = 2)
  (h2 : ∀ n m : ℕ, a (n * m) = a n * a m)
  (h_inc : ∀ n m : ℕ, n < m → a n < a m) : ∀ n : ℕ, a n = n := by
  -- Here we would place the proof, skipping it for now with sorry
  sorry

end uniq_increasing_seq_l90_90253


namespace rhombus_diagonal_length_l90_90491

theorem rhombus_diagonal_length (area d1 d2 : ℝ) (h₁ : area = 24) (h₂ : d1 = 8) (h₃ : area = (d1 * d2) / 2) : d2 = 6 := 
by sorry

end rhombus_diagonal_length_l90_90491


namespace football_team_total_players_l90_90429

/-- Let's denote the total number of players on the football team as P.
    We know that there are 31 throwers, and all of them are right-handed.
    The rest of the team is divided so one third are left-handed and the rest are right-handed.
    There are a total of 57 right-handed players on the team.
    Prove that the total number of players on the football team is 70. -/
theorem football_team_total_players 
  (P : ℕ) -- total number of players
  (T : ℕ := 31) -- number of throwers
  (L : ℕ) -- number of left-handed players
  (R : ℕ := 57) -- total number of right-handed players
  (H_all_throwers_rhs: ∀ x : ℕ, (x < P) → (x < T) → (x = T → x < R)) -- all throwers are right-handed
  (H_rest_division: ∀ x : ℕ, (x < P - T) → (x = L) → (x = 2 * L))
  : P = 70 :=
  sorry

end football_team_total_players_l90_90429


namespace marbles_end_of_day_l90_90157

theorem marbles_end_of_day :
  let initial_marbles := 40
  let lost_marbles_at_breakfast := 3
  let given_to_Susie_at_lunch := 5
  let new_marbles_from_mom := 12
  let returned_by_Susie := 2 * given_to_Susie_at_lunch
  initial_marbles - lost_marbles_at_breakfast - given_to_Susie_at_lunch + new_marbles_from_mom + returned_by_Susie = 54 :=
by
  let initial_marbles := 40
  let lost_marbles_at_breakfast := 3
  let given_to_Susie_at_lunch := 5
  let new_marbles_from_mom := 12
  let returned_by_Susie := 2 * given_to_Susie_at_lunch
  show initial_marbles - lost_marbles_at_breakfast - given_to_Susie_at_lunch + new_marbles_from_mom + returned_by_Susie = 54
  sorry

end marbles_end_of_day_l90_90157


namespace standard_deviation_bound_l90_90302

theorem standard_deviation_bound (mu sigma : ℝ) (h_mu : mu = 51) (h_ineq : mu - 3 * sigma > 44) : sigma < 7 / 3 :=
by
  sorry

end standard_deviation_bound_l90_90302


namespace number_of_diagonals_l90_90059

-- Define the regular pentagonal prism and its properties
def regular_pentagonal_prism : Type := sorry

-- Define what constitutes a diagonal in this context
def is_diagonal (p : regular_pentagonal_prism) (v1 v2 : Nat) : Prop :=
  sorry -- We need to detail what counts as a diagonal based on the conditions

-- Hypothesis on the structure specifying that there are 5 vertices on the top and 5 on the bottom
axiom vertices_on_top_and_bottom (p : regular_pentagonal_prism) : sorry -- We need the precise formalization

-- The main theorem
theorem number_of_diagonals (p : regular_pentagonal_prism) : ∃ n, n = 10 :=
  sorry

end number_of_diagonals_l90_90059


namespace five_digit_divisibility_l90_90348

-- Definitions of n and m
def n (a b c d e : ℕ) := 10000 * a + 1000 * b + 100 * c + 10 * d + e
def m (a b d e : ℕ) := 1000 * a + 100 * b + 10 * d + e

-- Condition that n is a five-digit number whose first digit is non-zero and n/m is an integer
theorem five_digit_divisibility (a b c d e : ℕ):
  1 <= a ∧ a <= 9 → 0 <= b ∧ b <= 9 → 0 <= c ∧ c <= 9 → 0 <= d ∧ d <= 9 → 0 <= e ∧ e <= 9 →
  m a b d e ∣ n a b c d e →
  ∃ x y : ℕ, a = x ∧ b = y ∧ c = 0 ∧ d = 0 ∧ e = 0 :=
by
  sorry

end five_digit_divisibility_l90_90348


namespace largest_prime_divisor_of_36_squared_plus_49_squared_l90_90906

theorem largest_prime_divisor_of_36_squared_plus_49_squared :
  Nat.gcd (36^2 + 49^2) 3697 = 3697 :=
by
  -- Since 3697 is prime, and the calculation shows 36^2 + 49^2 is 3697
  sorry

end largest_prime_divisor_of_36_squared_plus_49_squared_l90_90906


namespace Jill_age_l90_90757

theorem Jill_age 
  (G H I J : ℕ)
  (h1 : G = H - 4)
  (h2 : H = I + 5)
  (h3 : I + 2 = J)
  (h4 : G = 18) : 
  J = 19 := 
sorry

end Jill_age_l90_90757


namespace total_potatoes_sold_is_322kg_l90_90017

-- Define the given conditions
def bags_morning := 29
def bags_afternoon := 17
def weight_per_bag := 7

-- The theorem to prove the total kilograms sold is 322kg
theorem total_potatoes_sold_is_322kg : (bags_morning + bags_afternoon) * weight_per_bag = 322 :=
by
  sorry -- Placeholder for the actual proof

end total_potatoes_sold_is_322kg_l90_90017


namespace robert_birth_year_l90_90296

theorem robert_birth_year (n : ℕ) (h1 : (n + 1)^2 - n^2 = 89) : n = 44 ∧ n^2 = 1936 :=
by {
  sorry
}

end robert_birth_year_l90_90296


namespace percentage_of_original_solution_l90_90138

-- Define the problem and conditions
variable (P : ℝ)
variable (h1 : (0.5 * P + 0.5 * 60) = 55)

-- The theorem to prove
theorem percentage_of_original_solution : P = 50 :=
by
  -- Proof will go here
  sorry

end percentage_of_original_solution_l90_90138


namespace chocolates_sold_at_selling_price_l90_90601
noncomputable def chocolates_sold (C S : ℝ) (n : ℕ) : Prop :=
  (35 * C = n * S) ∧ ((S - C) / C * 100) = 66.67

theorem chocolates_sold_at_selling_price : ∃ n : ℕ, ∀ C S : ℝ,
  chocolates_sold C S n → n = 21 :=
by
  sorry

end chocolates_sold_at_selling_price_l90_90601


namespace purely_imaginary_iff_l90_90344

noncomputable def isPurelyImaginary (z : ℂ) : Prop :=
  z.re = 0

theorem purely_imaginary_iff (a : ℝ) :
  isPurelyImaginary (Complex.mk ((a * (a + 2)) / (a - 1)) (a ^ 2 + 2 * a - 3))
  ↔ a = 0 ∨ a = -2 := by
  sorry

end purely_imaginary_iff_l90_90344


namespace intersection_A_B_l90_90321

-- Definitions for sets A and B
def A : Set ℝ := { x | ∃ y : ℝ, x + y^2 = 1 }
def B : Set ℝ := { y | ∃ x : ℝ, y = x^2 - 1 }

-- The proof goal to show the intersection of sets A and B
theorem intersection_A_B : A ∩ B = { z | -1 ≤ z ∧ z ≤ 1 } :=
by
  sorry

end intersection_A_B_l90_90321


namespace possible_new_perimeters_l90_90481

theorem possible_new_perimeters
  (initial_tiles := 8)
  (initial_shape := "L")
  (initial_perimeter := 12)
  (additional_tiles := 2)
  (new_perimeters := [12, 14, 16]) :
  True := sorry

end possible_new_perimeters_l90_90481


namespace number_of_nintendo_games_to_give_away_l90_90047

-- Define the conditions
def initial_nintendo_games : ℕ := 20
def desired_nintendo_games_left : ℕ := 12

-- Define the proof problem as a Lean theorem
theorem number_of_nintendo_games_to_give_away :
  initial_nintendo_games - desired_nintendo_games_left = 8 :=
by
  sorry

end number_of_nintendo_games_to_give_away_l90_90047


namespace total_cartons_accepted_l90_90409

theorem total_cartons_accepted (total_cartons : ℕ) (customers : ℕ) (damaged_cartons_per_customer : ℕ) (initial_cartons_per_customer accepted_cartons_per_customer total_accepted_cartons : ℕ) :
    total_cartons = 400 →
    customers = 4 →
    damaged_cartons_per_customer = 60 →
    initial_cartons_per_customer = total_cartons / customers →
    accepted_cartons_per_customer = initial_cartons_per_customer - damaged_cartons_per_customer →
    total_accepted_cartons = accepted_cartons_per_customer * customers →
    total_accepted_cartons = 160 := 
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end total_cartons_accepted_l90_90409


namespace temperature_comparison_l90_90640

theorem temperature_comparison: ¬ (-3 > -0.3) :=
by
  sorry -- Proof goes here, skipped for now.

end temperature_comparison_l90_90640


namespace ones_digit_542_mul_3_is_6_l90_90628

/--
Given that the ones (units) digit of 542 is 2, prove that the ones digit of 542 multiplied by 3 is 6.
-/
theorem ones_digit_542_mul_3_is_6 (h: ∃ n : ℕ, 542 = 10 * n + 2) : (542 * 3) % 10 = 6 := 
by
  sorry

end ones_digit_542_mul_3_is_6_l90_90628


namespace primes_divisibility_l90_90907

theorem primes_divisibility
  (p1 p2 p3 p4 q1 q2 q3 q4 : ℕ)
  (hp1_lt_p2 : p1 < p2) (hp2_lt_p3 : p2 < p3) (hp3_lt_p4 : p3 < p4)
  (hq1_lt_q2 : q1 < q2) (hq2_lt_q3 : q2 < q3) (hq3_lt_q4 : q3 < q4)
  (hp4_minus_p1 : p4 - p1 = 8) (hq4_minus_q1 : q4 - q1 = 8)
  (hp1_gt_5 : 5 < p1) (hq1_gt_5 : 5 < q1) :
  30 ∣ (p1 - q1) :=
sorry

end primes_divisibility_l90_90907


namespace factorize_expression_l90_90291

theorem factorize_expression (x y : ℝ) : x^2 - 1 + 2 * x * y + y^2 = (x + y + 1) * (x + y - 1) :=
by sorry

end factorize_expression_l90_90291


namespace prime_factorization_sum_l90_90694

theorem prime_factorization_sum (x y : ℕ) (hx : x > 0) (hy : y > 0) (hxy : 13 * x^7 = 17 * y^11) : 
  a * e + b * f = 18 :=
by
  -- Let a and b be prime factors of x
  let a : ℕ := 17 -- prime factor found in the solution
  let e : ℕ := 1 -- exponent found for 17
  let b : ℕ := 0 -- no second prime factor
  let f : ℕ := 0 -- corresponding exponent

  sorry

end prime_factorization_sum_l90_90694


namespace planar_graph_edge_bound_l90_90024

structure Graph :=
  (V E : ℕ) -- vertices and edges

def planar_connected (G : Graph) : Prop := 
  sorry -- Planarity and connectivity conditions are complex to formalize

def num_faces (G : Graph) : ℕ :=
  sorry -- Number of faces based on V, E and planarity

theorem planar_graph_edge_bound (G : Graph) (h_planar : planar_connected G) 
  (euler : G.V - G.E + num_faces G = 2) 
  (face_bound : 2 * G.E ≥ 3 * num_faces G) : 
  G.E ≤ 3 * G.V - 6 :=
sorry

end planar_graph_edge_bound_l90_90024


namespace simplify_fraction_l90_90390

theorem simplify_fraction :
  (18 / 462) + (35 / 77) = 38 / 77 := 
by sorry

end simplify_fraction_l90_90390


namespace max_sum_of_positives_l90_90067

theorem max_sum_of_positives (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y + 1 / x + 1 / y = 5) : x + y ≤ 4 :=
sorry

end max_sum_of_positives_l90_90067


namespace Berry_Temperature_Friday_l90_90838

theorem Berry_Temperature_Friday (temps : Fin 6 → ℝ) (avg_temp : ℝ) (total_days : ℕ) (friday_temp : ℝ) :
  temps 0 = 99.1 → 
  temps 1 = 98.2 →
  temps 2 = 98.7 →
  temps 3 = 99.3 →
  temps 4 = 99.8 →
  temps 5 = 98.9 →
  avg_temp = 99 →
  total_days = 7 →
  friday_temp = (avg_temp * total_days) - (temps 0 + temps 1 + temps 2 + temps 3 + temps 4 + temps 5) →
  friday_temp = 99 :=
by 
  intros h0 h1 h2 h3 h4 h5 h_avg h_days h_friday
  sorry

end Berry_Temperature_Friday_l90_90838


namespace goose_eggs_count_l90_90286

theorem goose_eggs_count 
  (E : ℕ) 
  (hatch_rate : ℚ)
  (survive_first_month_rate : ℚ)
  (survive_first_year_rate : ℚ)
  (geese_survived_first_year : ℕ)
  (no_more_than_one_goose_per_egg : Prop) 
  (hatch_eq : hatch_rate = 2/3) 
  (survive_first_month_eq : survive_first_month_rate = 3/4) 
  (survive_first_year_eq : survive_first_year_rate = 2/5) 
  (geese_survived_eq : geese_survived_first_year = 130):
  E = 650 :=
by
  sorry

end goose_eggs_count_l90_90286


namespace geometric_sequence_product_l90_90727

-- Defining a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- Given data
def a := fun n => (4 : ℝ) * (2 : ℝ)^(n-4)

-- Main proof problem
theorem geometric_sequence_product (a : ℕ → ℝ) (h : is_geometric_sequence a) (h₁ : a 4 = 4) :
  a 2 * a 6 = 16 :=
by
  sorry

end geometric_sequence_product_l90_90727


namespace compute_x2_y2_l90_90959

theorem compute_x2_y2 (x y : ℝ) (h1 : 1 < x) (h2 : 1 < y)
  (h3 : (Real.log x / Real.log 4)^3 + (Real.log y / Real.log 5)^3 + 27 = 9 * (Real.log x / Real.log 4) * (Real.log y / Real.log 5)) :
  x^2 + y^2 = 189 := 
by sorry

end compute_x2_y2_l90_90959


namespace scientific_notation_of_great_wall_l90_90872

theorem scientific_notation_of_great_wall : 
  ∀ n : ℕ, (6700010 : ℝ) = 6.7 * 10^6 :=
by
  sorry

end scientific_notation_of_great_wall_l90_90872


namespace percentage_increase_l90_90388

theorem percentage_increase (d : ℝ) (v_current v_reduce v_increase t_reduce t_increase : ℝ) (h1 : d = 96)
  (h2 : v_current = 8) (h3 : v_reduce = v_current - 4) (h4 : t_reduce = d / v_reduce) 
  (h5 : t_increase = d / v_increase) (h6 : t_reduce = t_current + 16) (h7 : t_increase = t_current - 16) :
  (v_increase - v_current) / v_current * 100 = 50 := 
sorry

end percentage_increase_l90_90388


namespace problem_statement_l90_90345

noncomputable def min_expression_value (θ1 θ2 θ3 θ4 : ℝ) : ℝ :=
  (2 * (Real.sin θ1)^2 + 1 / (Real.sin θ1)^2) *
  (2 * (Real.sin θ2)^2 + 1 / (Real.sin θ2)^2) *
  (2 * (Real.sin θ3)^2 + 1 / (Real.sin θ3)^2) *
  (2 * (Real.sin θ4)^2 + 1 / (Real.sin θ4)^2)

theorem problem_statement (θ1 θ2 θ3 θ4 : ℝ) (h_pos: θ1 > 0 ∧ θ2 > 0 ∧ θ3 > 0 ∧ θ4 > 0) (h_sum: θ1 + θ2 + θ3 + θ4 = Real.pi) :
  min_expression_value θ1 θ2 θ3 θ4 = 81 :=
sorry

end problem_statement_l90_90345


namespace sindbad_can_identify_eight_genuine_dinars_l90_90879

/--
Sindbad has 11 visually identical dinars in his purse, one of which may be counterfeit and differs in weight from the genuine ones. Using a balance scale twice without weights, it's possible to identify at least 8 genuine dinars.
-/
theorem sindbad_can_identify_eight_genuine_dinars (dinars : Fin 11 → ℝ) (is_genuine : Fin 11 → Prop) :
  (∃! i, ¬ is_genuine i) → 
  (∃ S : Finset (Fin 11), S.card = 8 ∧ S ⊆ (Finset.univ : Finset (Fin 11)) ∧ ∀ i ∈ S, is_genuine i) :=
sorry

end sindbad_can_identify_eight_genuine_dinars_l90_90879


namespace contradiction_proof_l90_90848

theorem contradiction_proof (a b c d : ℝ) (h1 : a = 1) (h2 : b = 1) (h3 : c = 1) (h4 : d = 1) (h5 : a * c + b * d > 1) : ¬ (a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) :=
by 
  sorry

end contradiction_proof_l90_90848


namespace repeatingDecimal_as_fraction_l90_90056

def repeatingDecimal : ℚ := 0.136513513513

theorem repeatingDecimal_as_fraction : repeatingDecimal = 136377 / 999000 := 
by 
  sorry

end repeatingDecimal_as_fraction_l90_90056


namespace new_average_age_l90_90405

theorem new_average_age (avg_age : ℕ) (num_people : ℕ) (leaving_age : ℕ) (remaining_people : ℕ) :
  avg_age = 40 →
  num_people = 8 →
  leaving_age = 25 →
  remaining_people = 7 →
  (avg_age * num_people - leaving_age) / remaining_people = 42 :=
by
  sorry

end new_average_age_l90_90405


namespace input_for_output_16_l90_90219

theorem input_for_output_16 (x : ℝ) (y : ℝ) : 
  (y = (if x < 0 then (x + 1)^2 else (x - 1)^2)) → 
  y = 16 → 
  (x = 5 ∨ x = -5) :=
by sorry

end input_for_output_16_l90_90219


namespace even_binomial_coefficients_l90_90250

theorem even_binomial_coefficients (n : ℕ) (h_pos: 0 < n) : 
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ n - 1 → 2 ∣ Nat.choose n k) ↔ ∃ k : ℕ, n = 2^k :=
by
  sorry

end even_binomial_coefficients_l90_90250


namespace fixed_point_of_exponential_function_l90_90440

theorem fixed_point_of_exponential_function (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) : 
  ∃ p : ℝ × ℝ, p = (-2, 1) ∧ ∀ x : ℝ, (x, a^(x + 2)) = p → x = -2 ∧ a^(x + 2) = 1 :=
by
  sorry

end fixed_point_of_exponential_function_l90_90440


namespace three_star_five_l90_90450

-- Definitions based on conditions
def star (a b : ℕ) : ℕ := 2 * a^2 + 3 * a * b + 2 * b^2

-- Theorem statement to be proved
theorem three_star_five : star 3 5 = 113 := by
  sorry

end three_star_five_l90_90450


namespace greatest_int_lt_neg_31_div_6_l90_90745

theorem greatest_int_lt_neg_31_div_6 : ∃ (n : ℤ), n < -31 / 6 ∧ ∀ m : ℤ, m < -31 / 6 → m ≤ n := 
sorry

end greatest_int_lt_neg_31_div_6_l90_90745


namespace arithmetic_sequence_a6_l90_90119

theorem arithmetic_sequence_a6 (a : ℕ → ℤ) (h_arith : ∀ n, a (n+1) - a n = a 2 - a 1)
  (h_a1 : a 1 = 5) (h_a5 : a 5 = 1) : a 6 = 0 :=
by
  -- Definitions derived from conditions in the problem:
  -- 1. a : ℕ → ℤ : Sequence defined on ℕ with integer values.
  -- 2. h_arith : ∀ n, a (n+1) - a n = a 2 - a 1 : Arithmetic sequence property
  -- 3. h_a1 : a 1 = 5 : First term of the sequence is 5.
  -- 4. h_a5 : a 5 = 1 : Fifth term of the sequence is 1.
  sorry

end arithmetic_sequence_a6_l90_90119


namespace ratio_of_coeffs_l90_90825

theorem ratio_of_coeffs
  (a b c d e : ℝ) 
  (h_poly : ∀ x : ℝ, a * x^4 + b * x^3 + c * x^2 + d * x + e = 0 → x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4) : 
  d / e = 25 / 12 :=
by
  sorry

end ratio_of_coeffs_l90_90825


namespace stratified_sampling_total_sample_size_l90_90223

-- Definitions based on conditions
def pure_milk_brands : ℕ := 30
def yogurt_brands : ℕ := 10
def infant_formula_brands : ℕ := 35
def adult_milk_powder_brands : ℕ := 25
def sampled_infant_formula_brands : ℕ := 7

-- The goal is to prove that the total sample size n is 20.
theorem stratified_sampling_total_sample_size : 
  let total_brands := pure_milk_brands + yogurt_brands + infant_formula_brands + adult_milk_powder_brands
  let sampling_fraction := sampled_infant_formula_brands / infant_formula_brands
  let pure_milk_samples := pure_milk_brands * sampling_fraction
  let yogurt_samples := yogurt_brands * sampling_fraction
  let adult_milk_samples := adult_milk_powder_brands * sampling_fraction
  let n := pure_milk_samples + yogurt_samples + sampled_infant_formula_brands + adult_milk_samples
  n = 20 :=
by
  sorry

end stratified_sampling_total_sample_size_l90_90223


namespace johns_previous_salary_l90_90360

-- Conditions
def johns_new_salary : ℝ := 70
def percent_increase : ℝ := 0.16666666666666664

-- Statement
theorem johns_previous_salary :
  ∃ x : ℝ, x + percent_increase * x = johns_new_salary ∧ x = 60 :=
by
  sorry

end johns_previous_salary_l90_90360


namespace find_savings_l90_90358

noncomputable def savings (income expenditure : ℕ) : ℕ :=
  income - expenditure

theorem find_savings (I E : ℕ) (h_ratio : I = 9 * E) (h_income : I = 18000) : savings I E = 2000 :=
by
  sorry

end find_savings_l90_90358


namespace peter_initial_erasers_l90_90876

theorem peter_initial_erasers (E : ℕ) (h : E + 3 = 11) : E = 8 :=
by {
  sorry
}

end peter_initial_erasers_l90_90876


namespace inequality_proof_l90_90168

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    a + b + c ≤ (a^2 + b^2) / (2 * c) + (a^2 + c^2) / (2 * b) + (b^2 + c^2) / (2 * a) ∧ 
    (a^2 + b^2) / (2 * c) + (a^2 + c^2) / (2 * b) + (b^2 + c^2) / (2 * a) ≤ (a^3 / (b * c)) + (b^3 / (a * c)) + (c^3 / (a * b)) := 
by
  sorry

end inequality_proof_l90_90168


namespace people_who_speak_French_l90_90971

theorem people_who_speak_French (T L N B : ℕ) (hT : T = 25) (hL : L = 13) (hN : N = 6) (hB : B = 9) : 
  ∃ F : ℕ, F = 15 := 
by 
  sorry

end people_who_speak_French_l90_90971


namespace at_least_one_gt_one_l90_90938

theorem at_least_one_gt_one (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y > 2) : x > 1 ∨ y > 1 :=
sorry

end at_least_one_gt_one_l90_90938


namespace prime_of_form_a2_minus_1_l90_90987

theorem prime_of_form_a2_minus_1 (a : ℕ) (p : ℕ) (ha : a ≥ 2) (hp : p = a^2 - 1) (prime_p : Nat.Prime p) : p = 3 := 
by 
  sorry

end prime_of_form_a2_minus_1_l90_90987


namespace karl_savings_proof_l90_90171

-- Definitions based on the conditions
def original_price_per_notebook : ℝ := 3.00
def sale_discount : ℝ := 0.25
def extra_discount_threshold : ℝ := 10
def extra_discount_rate : ℝ := 0.05

-- The number of notebooks Karl could have purchased instead
def notebooks_purchased : ℝ := 12

-- The total savings calculation
noncomputable def total_savings : ℝ := 
  let original_total := notebooks_purchased * original_price_per_notebook
  let discounted_price_per_notebook := original_price_per_notebook * (1 - sale_discount)
  let extra_discount := if notebooks_purchased > extra_discount_threshold then discounted_price_per_notebook * extra_discount_rate else 0
  let total_price_after_discounts := notebooks_purchased * discounted_price_per_notebook - notebooks_purchased * extra_discount
  original_total - total_price_after_discounts

-- Formal statement to prove
theorem karl_savings_proof : total_savings = 10.35 := 
  sorry

end karl_savings_proof_l90_90171


namespace distance_between_foci_l90_90611

-- Let the hyperbola be defined by the equation xy = 4.
def hyperbola (x y : ℝ) : Prop := x * y = 4

-- Prove that the distance between the foci of this hyperbola is 8.
theorem distance_between_foci : ∀ (x y : ℝ), hyperbola x y → ∃ d, d = 8 :=
by {
    sorry
}

end distance_between_foci_l90_90611


namespace alyssa_total_spent_l90_90081

theorem alyssa_total_spent :
  let grapes := 12.08
  let cherries := 9.85
  grapes + cherries = 21.93 := by
  sorry

end alyssa_total_spent_l90_90081


namespace larry_wins_prob_l90_90400

def probability_larry_wins (pLarry pJulius : ℚ) : ℚ :=
  let r := (1 - pLarry) * (1 - pJulius)
  pLarry * (1 / (1 - r))

theorem larry_wins_prob : probability_larry_wins (2 / 3) (1 / 3) = 6 / 7 :=
by
  -- Definitions for probabilities
  let pLarry := 2 / 3
  let pJulius := 1 / 3
  have r := (1 - pLarry) * (1 - pJulius)
  have S := pLarry * (1 / (1 - r))
  -- Expected result
  have expected := 6 / 7
  -- Prove the result equals the expected
  sorry

end larry_wins_prob_l90_90400


namespace tom_sells_games_for_225_42_usd_l90_90782

theorem tom_sells_games_for_225_42_usd :
  let initial_usd := 200
  let usd_to_eur := 0.85
  let tripled_usd := initial_usd * 3
  let eur_value := tripled_usd * usd_to_eur
  let eur_to_jpy := 130
  let jpy_value := eur_value * eur_to_jpy
  let percent_sold := 0.40
  let sold_jpy_value := jpy_value * percent_sold
  let jpy_to_usd := 0.0085
  let sold_usd_value := sold_jpy_value * jpy_to_usd
  sold_usd_value = 225.42 :=
by
  sorry

end tom_sells_games_for_225_42_usd_l90_90782


namespace rectangle_pairs_l90_90239

theorem rectangle_pairs :
  {p : ℕ × ℕ | 0 < p.1 ∧ 0 < p.2 ∧ p.1 * p.2 = 18} = {(1, 18), (2, 9), (3, 6), (6, 3), (9, 2), (18, 1)} :=
by { sorry }

end rectangle_pairs_l90_90239


namespace sufficient_condition_implies_range_l90_90805

theorem sufficient_condition_implies_range {x m : ℝ} : (∀ x, 1 ≤ x ∧ x < 4 → x < m) → 4 ≤ m :=
by
  sorry

end sufficient_condition_implies_range_l90_90805


namespace t_shirt_sale_revenue_per_minute_l90_90842

theorem t_shirt_sale_revenue_per_minute (total_tshirts : ℕ) (total_minutes : ℕ)
  (black_tshirts white_tshirts : ℕ) (cost_black cost_white : ℕ) 
  (half_total_tshirts : total_tshirts = black_tshirts + white_tshirts)
  (equal_halves : black_tshirts = white_tshirts)
  (black_price : cost_black = 30) (white_price : cost_white = 25)
  (total_time : total_minutes = 25)
  (total_sold : total_tshirts = 200) :
  ((black_tshirts * cost_black) + (white_tshirts * cost_white)) / total_minutes = 220 := by
  sorry

end t_shirt_sale_revenue_per_minute_l90_90842


namespace prime_factors_sum_l90_90261

theorem prime_factors_sum (w x y z t : ℕ) 
  (h : 2^w * 3^x * 5^y * 7^z * 11^t = 107100) : 
  2 * w + 3 * x + 5 * y + 7 * z + 11 * t = 38 :=
sorry

end prime_factors_sum_l90_90261


namespace investment_change_l90_90777

theorem investment_change (x : ℝ) :
  (1 : ℝ) > (0 : ℝ) → 
  1.05 * x / x - 1 * 100 = 5 :=
by
  sorry

end investment_change_l90_90777


namespace repeating_decimal_value_l90_90999

def repeating_decimal : ℝ := 0.0000253253325333 -- Using repeating decimal as given in the conditions

theorem repeating_decimal_value :
  (10^7 - 10^5) * repeating_decimal = 253 / 990 :=
sorry

end repeating_decimal_value_l90_90999


namespace triangle_ratio_l90_90655

theorem triangle_ratio (A B C : ℝ) (a b c : ℝ)
  (hA : A = 2 * Real.pi / 3)
  (h_a : a = Real.sqrt 3 * c)
  (h_angle_sum : A + B + C = Real.pi)
  (h_law_of_sines : a / Real.sin A = c / Real.sin C) :
  b / c = 1 :=
sorry

end triangle_ratio_l90_90655


namespace inequality_example_l90_90946

theorem inequality_example (a b m : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_m_pos : 0 < m) (h_ba : b = 2) (h_aa : a = 1) :
  (b + m) / (a + m) < b / a :=
sorry

end inequality_example_l90_90946


namespace moles_ethane_and_hexachloroethane_l90_90476

-- Define the conditions
def balanced_eq (a b c d : ℕ) : Prop :=
  a * 6 = b ∧ d * 6 = c

-- The main theorem statement
theorem moles_ethane_and_hexachloroethane (moles_Cl2 : ℕ) :
  moles_Cl2 = 18 → balanced_eq 1 1 18 3 :=
by
  sorry

end moles_ethane_and_hexachloroethane_l90_90476


namespace francis_violin_count_l90_90304

theorem francis_violin_count :
  let ukuleles := 2
  let guitars := 4
  let ukulele_strings := 4
  let guitar_strings := 6
  let violin_strings := 4
  let total_strings := 40
  ∃ (violins: ℕ), violins = 2 := by
    sorry

end francis_violin_count_l90_90304


namespace sufficient_not_necessary_l90_90251

theorem sufficient_not_necessary (a b : ℝ) :
  (a^2 + b^2 = 0 → ab = 0) ∧ (ab = 0 → ¬(a^2 + b^2 = 0)) := 
by
  have h1 : (a^2 + b^2 = 0 → ab = 0) := sorry
  have h2 : (ab = 0 → ¬(a^2 + b^2 = 0)) := sorry
  exact ⟨h1, h2⟩

end sufficient_not_necessary_l90_90251


namespace polygon_sides_l90_90589

theorem polygon_sides (n : ℕ) : 
  ((n - 2) * 180 = 4 * 360) → n = 10 :=
by
  sorry

end polygon_sides_l90_90589


namespace root_interval_l90_90374

noncomputable def f (x : ℝ) : ℝ := 3^x + 3 * x - 8

theorem root_interval (h₁ : f 1 < 0) (h₂ : f 1.5 > 0) (h₃ : f 1.25 < 0) (h₄ : f 2 > 0) :
  ∃ x, 1.25 < x ∧ x < 1.5 ∧ f x = 0 :=
sorry

end root_interval_l90_90374


namespace puffy_muffy_total_weight_l90_90082

theorem puffy_muffy_total_weight (scruffy_weight muffy_weight puffy_weight : ℕ)
  (h1 : scruffy_weight = 12)
  (h2 : muffy_weight = scruffy_weight - 3)
  (h3 : puffy_weight = muffy_weight + 5) :
  puffy_weight + muffy_weight = 23 := by
  sorry

end puffy_muffy_total_weight_l90_90082


namespace total_students_l90_90401

theorem total_students (ratio_boys : ℕ) (ratio_girls : ℕ) (num_girls : ℕ) 
  (h_ratio : ratio_boys = 8) (h_ratio_girls : ratio_girls = 5) (h_num_girls : num_girls = 175) : 
  ratio_boys * (num_girls / ratio_girls) + num_girls = 455 :=
by
  sorry

end total_students_l90_90401


namespace ratio_of_container_volumes_l90_90071

-- Define the volumes of the first and second containers.
variables (A B : ℝ )

-- Hypotheses based on the problem conditions
-- First container is 4/5 full
variable (h1 : A * 4 / 5 = B * 2 / 3)

-- The statement to prove
theorem ratio_of_container_volumes : A / B = 5 / 6 :=
by
  sorry

end ratio_of_container_volumes_l90_90071


namespace total_students_in_Lansing_l90_90158

theorem total_students_in_Lansing :
  let num_schools_300 := 20
  let num_schools_350 := 30
  let num_schools_400 := 15
  let students_per_school_300 := 300
  let students_per_school_350 := 350
  let students_per_school_400 := 400
  (num_schools_300 * students_per_school_300 + num_schools_350 * students_per_school_350 + num_schools_400 * students_per_school_400 = 22500) := 
  sorry

end total_students_in_Lansing_l90_90158


namespace average_monthly_sales_booster_club_l90_90288

noncomputable def monthly_sales : List ℕ := [80, 100, 75, 95, 110, 180, 90, 115, 130, 200, 160, 140]

noncomputable def average_sales (sales : List ℕ) : ℝ :=
  (sales.foldr (λ x acc => x + acc) 0 : ℕ) / sales.length

theorem average_monthly_sales_booster_club : average_sales monthly_sales = 122.92 := by
  sorry

end average_monthly_sales_booster_club_l90_90288


namespace range_of_a_extrema_of_y_l90_90163

variable {a b c : ℝ}

def setA (a b c : ℝ) : Prop := a^2 - b * c - 8 * a + 7 = 0
def setB (a b c : ℝ) : Prop := b^2 + c^2 + b * c - b * a + b = 0

theorem range_of_a (h: ∃ a b c : ℝ, setA a b c ∧ setB a b c) : 1 ≤ a ∧ a ≤ 9 :=
sorry

theorem extrema_of_y (h: ∃ a b c : ℝ, setA a b c ∧ setB a b c) 
  (y : ℝ) 
  (hy1 : y = a * b + b * c + a * c)
  (hy2 : ∀ x y z : ℝ, setA x y z → setB x y z → y = x * y + y * z + x * z) : 
  y = 88 ∨ y = -56 :=
sorry

end range_of_a_extrema_of_y_l90_90163


namespace problem_1_problem_2_problem_3_l90_90077

-- Definition and proof state for problem 1
theorem problem_1 (a b m n : ℕ) (h₀ : a + b * Real.sqrt 3 = (m + n * Real.sqrt 3)^2) : 
  a = m^2 + 3 * n^2 ∧ b = 2 * m * n := by
  sorry

-- Definition and proof state for problem 2
theorem problem_2 (a m n : ℕ) (h₀ : a + 4 * Real.sqrt 3 = (m + n * Real.sqrt 3)^2) : 
  a = 13 ∨ a = 7 := by
  sorry

-- Definition and proof state for problem 3
theorem problem_3 : Real.sqrt (6 + 2 * Real.sqrt 5) = 1 + Real.sqrt 5 := by
  sorry

end problem_1_problem_2_problem_3_l90_90077


namespace range_of_a_l90_90000

theorem range_of_a (a : ℝ) (A : Set ℝ) (h : A = {x | a * x^2 - 3 * x + 1 = 0} ∧ ∃ (n : ℕ), 2 ^ n - 1 = 3) :
  a ∈ Set.Ioo (-(1:ℝ)/0) 0 ∪ Set.Ioo 0 (9 / 4) :=
sorry

end range_of_a_l90_90000


namespace max_squares_covered_by_card_l90_90630

noncomputable def card_coverage_max_squares (card_side : ℝ) (square_side : ℝ) : ℕ :=
  if card_side = 2 ∧ square_side = 1 then 9 else 0

theorem max_squares_covered_by_card : card_coverage_max_squares 2 1 = 9 := by
  sorry

end max_squares_covered_by_card_l90_90630


namespace range_of_x_l90_90010

theorem range_of_x (x y : ℝ) (h1 : 4 * x + y = 3) (h2 : -2 < y ∧ y ≤ 7) : 1 ≤ x ∧ x < 5 / 4 := 
  sorry

end range_of_x_l90_90010


namespace min_value_square_distance_l90_90442

theorem min_value_square_distance (x y : ℝ) (h : x^2 + y^2 - 4*x + 2 = 0) : 
  ∃ c, (∀ x y : ℝ, x^2 + y^2 - 4*x + 2 = 0 → x^2 + (y - 2)^2 ≥ c) ∧ c = 2 :=
sorry

end min_value_square_distance_l90_90442


namespace coefficient_x3_in_expansion_l90_90849

theorem coefficient_x3_in_expansion : 
  (∃ (r : ℕ), 5 - r / 2 = 3 ∧ 2 * Nat.choose 5 r = 10) :=
by 
  sorry

end coefficient_x3_in_expansion_l90_90849


namespace parabola_coefficients_l90_90072

theorem parabola_coefficients
    (vertex : (ℝ × ℝ))
    (passes_through : (ℝ × ℝ))
    (vertical_axis_of_symmetry : Prop)
    (hv : vertex = (2, -3))
    (hp : passes_through = (0, 1))
    (has_vertical_axis : vertical_axis_of_symmetry) :
    ∃ a b c : ℝ, ∀ x : ℝ, (x = 0 → (a * x^2 + b * x + c = 1)) ∧ (x = 2 → (a * x^2 + b * x + c = -3)) := sorry

end parabola_coefficients_l90_90072


namespace find_a_l90_90548

theorem find_a (a : ℤ) (A B : Set ℤ) (hA : A = {0, 1}) (hB : B = {-1, 0, a + 3}) (h : A ⊆ B) : a = -2 := by
  sorry

end find_a_l90_90548


namespace math_problem_l90_90980

noncomputable def log_base (base x : ℝ) : ℝ := Real.log x / Real.log base

theorem math_problem (a b c : ℝ) (h1 : ∃ k : ℤ, log_base c b = k)
  (h2 : log_base a (1 / b) > log_base a (Real.sqrt b) ∧ log_base a (Real.sqrt b) > log_base b (a^2)) :
  (∃ n : ℕ, n = 1 ∧ 
    ((1 / b > Real.sqrt b ∧ Real.sqrt b > a^2) ∨ 
    (Real.log b + log_base a a = 0) ∨ 
    (0 < a ∧ a < b ∧ b < 1) ∨ 
    (a * b = 1))) :=
by sorry

end math_problem_l90_90980


namespace probability_not_green_l90_90380

theorem probability_not_green :
  let red_balls := 6
  let yellow_balls := 3
  let black_balls := 4
  let green_balls := 5
  let total_balls := red_balls + yellow_balls + black_balls + green_balls
  let not_green_balls := red_balls + yellow_balls + black_balls
  total_balls = 18 ∧ not_green_balls = 13 → (not_green_balls : ℚ) / total_balls = 13 / 18 := 
by
  intros
  sorry

end probability_not_green_l90_90380


namespace sales_on_third_day_l90_90893

variable (a m : ℕ)

def first_day_sales : ℕ := a
def second_day_sales : ℕ := 3 * a - 3 * m
def third_day_sales : ℕ := (3 * a - 3 * m) + m

theorem sales_on_third_day 
  (a m : ℕ) : third_day_sales a m = 3 * a - 2 * m :=
by
  -- Assuming the conditions as our definitions:
  let fds := first_day_sales a
  let sds := second_day_sales a m
  let tds := third_day_sales a m

  -- Proof direction:
  show tds = 3 * a - 2 * m
  sorry

end sales_on_third_day_l90_90893


namespace complex_exponential_sum_l90_90573

theorem complex_exponential_sum (γ δ : ℝ) 
  (h : Complex.exp (Complex.I * γ) + Complex.exp (Complex.I * δ) = -1 / 2 + 5 / 4 * Complex.I) :
  Complex.exp (-Complex.I * γ) + Complex.exp (-Complex.I * δ) = -1 / 2 - 5 / 4 * Complex.I :=
by
  sorry

end complex_exponential_sum_l90_90573


namespace distinct_balls_boxes_l90_90230

theorem distinct_balls_boxes :
  (3^5 = 243) :=
by sorry

end distinct_balls_boxes_l90_90230


namespace find_a4_plus_a6_l90_90444

variable {a : ℕ → ℝ}

-- Geometric sequence definition
def is_geometric_seq (a : ℕ → ℝ) := ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- Conditions for the problem
axiom seq_geometric : is_geometric_seq a
axiom seq_positive : ∀ n : ℕ, n > 0 → a n > 0
axiom given_equation : a 3 * a 5 + 2 * a 4 * a 6 + a 5 * a 7 = 81

-- The problem to prove
theorem find_a4_plus_a6 : a 4 + a 6 = 9 :=
sorry

end find_a4_plus_a6_l90_90444


namespace range_of_2a_plus_b_l90_90028

theorem range_of_2a_plus_b (a b : ℝ) (ha : 1 < a ∧ a < 4) (hb : -2 < b ∧ b < 2) :
  0 < 2 * a + b ∧ 2 * a + b < 10 :=
sorry

end range_of_2a_plus_b_l90_90028


namespace appropriate_length_of_presentation_l90_90614

theorem appropriate_length_of_presentation (wpm : ℕ) (min_time min_words max_time max_words total_words : ℕ) 
  (h1 : total_words = 160) 
  (h2 : min_time = 45) 
  (h3 : min_words = min_time * wpm) 
  (h4 : max_time = 60) 
  (h5 : max_words = max_time * wpm) : 
  7200 ≤ 9400 ∧ 9400 ≤ 9600 :=
by 
  sorry

end appropriate_length_of_presentation_l90_90614


namespace max_x_value_l90_90226

theorem max_x_value (x y z : ℝ) (h1 : x + y + z = 7) (h2 : x * y + x * z + y * z = 12) : x ≤ 1 :=
by sorry

end max_x_value_l90_90226


namespace height_of_prism_l90_90673

-- Definitions based on conditions
def Volume : ℝ := 120
def edge1 : ℝ := 3
def edge2 : ℝ := 4
def BaseArea : ℝ := edge1 * edge2

-- Define the problem statement
theorem height_of_prism (h : ℝ) : (BaseArea * h / 2 = Volume) → (h = 20) :=
by
  intro h_value
  have Volume_equiv : h = 2 * Volume / BaseArea := sorry
  sorry

end height_of_prism_l90_90673


namespace bottle_ratio_l90_90423

theorem bottle_ratio (C1 C2 : ℝ)  
  (h1 : (C1 / 2) + (C2 / 4) = (C1 + C2) / 3) :
  C2 = 2 * C1 :=
sorry

end bottle_ratio_l90_90423


namespace bus_speed_l90_90580

theorem bus_speed (t : ℝ) (d : ℝ) (h : t = 42 / 60) (d_eq : d = 35) : d / t = 50 :=
by
  -- Assume
  sorry

end bus_speed_l90_90580


namespace simplify_expression_l90_90370

variable (x y : ℝ)

theorem simplify_expression:
  3*x^2 - 3*(2*x^2 + 4*y) + 2*(x^2 - y) = -x^2 - 14*y := 
by 
  sorry

end simplify_expression_l90_90370


namespace Ariella_has_more_savings_l90_90957

variable (Daniella_savings: ℝ) (Ariella_future_savings: ℝ) (interest_rate: ℝ) (time_years: ℝ)
variable (initial_Ariella_savings: ℝ)

-- Conditions
axiom h1 : Daniella_savings = 400
axiom h2 : Ariella_future_savings = 720
axiom h3 : interest_rate = 0.10
axiom h4 : time_years = 2

-- Assume simple interest formula for future savings
axiom simple_interest : Ariella_future_savings = initial_Ariella_savings * (1 + interest_rate * time_years)

-- Show the difference in savings
theorem Ariella_has_more_savings : initial_Ariella_savings - Daniella_savings = 200 :=
by sorry

end Ariella_has_more_savings_l90_90957


namespace sale_in_fifth_month_l90_90578

def sales_month_1 := 6635
def sales_month_2 := 6927
def sales_month_3 := 6855
def sales_month_4 := 7230
def sales_month_6 := 4791
def target_average := 6500
def number_of_months := 6

def total_sales := sales_month_1 + sales_month_2 + sales_month_3 + sales_month_4 + sales_month_6

theorem sale_in_fifth_month :
  (target_average * number_of_months) - total_sales = 6562 :=
by
  sorry

end sale_in_fifth_month_l90_90578


namespace geometric_sequence_sum_l90_90040

def geometric_sequence_props (a : ℕ → ℤ) (S : ℕ → ℤ) := 
  (∀ n, a n = a 1 * 2 ^ (n - 1)) ∧ 
  (∀ n, S n = (1 - 2^n) / (1 - 2)) ∧ 
  (a 5 - a 3 = 12) ∧ 
  (a 6 - a 4 = 24)

theorem geometric_sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ) 
  (h : geometric_sequence_props a S) :
  ∀ n, S n / a n = 2 - 2^(1 - n) :=
by
  sorry

end geometric_sequence_sum_l90_90040


namespace system_of_equations_solution_l90_90716

theorem system_of_equations_solution:
  ∀ (x y : ℝ), 
    x^2 + y^2 + x + y = 42 ∧ x * y = 15 → 
      (x = 3 ∧ y = 5) ∨ (x = 5 ∧ y = 3) ∨ 
      (x = (-9 + Real.sqrt 21) / 2 ∧ y = (-9 - Real.sqrt 21) / 2) ∨ 
      (x = (-9 - Real.sqrt 21) / 2 ∧ y = (-9 + Real.sqrt 21) / 2) := 
by
  sorry

end system_of_equations_solution_l90_90716


namespace algebraic_expression_value_l90_90421

-- Define the conditions 
variables (x y : ℝ)
def condition1 : Prop := x + y = 2
def condition2 : Prop := x - y = 4

-- State the main theorem
theorem algebraic_expression_value (h1 : condition1 x y) (h2 : condition2 x y) :
  1 + x^2 - y^2 = 9 :=
sorry

end algebraic_expression_value_l90_90421


namespace range_of_x_l90_90792

theorem range_of_x (x : ℝ) : (abs (x + 1) + abs (x - 5) = 6) ↔ (-1 ≤ x ∧ x ≤ 5) :=
by sorry

end range_of_x_l90_90792


namespace log_addition_property_l90_90756

noncomputable def logFunction (x : ℝ) : ℝ := Real.log x

theorem log_addition_property (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : logFunction (a * b) = 1) :
  logFunction (a^2) + logFunction (b^2) = 2 :=
by
  sorry

end log_addition_property_l90_90756


namespace total_students_is_88_l90_90247

def orchestra_students : Nat := 20
def band_students : Nat := 2 * orchestra_students
def choir_boys : Nat := 12
def choir_girls : Nat := 16
def choir_students : Nat := choir_boys + choir_girls

def total_students : Nat := orchestra_students + band_students + choir_students

theorem total_students_is_88 : total_students = 88 := by
  sorry

end total_students_is_88_l90_90247


namespace base_of_524_l90_90349

theorem base_of_524 : 
  ∀ (b : ℕ), (5 * b^2 + 2 * b + 4 = 340) → b = 8 :=
by
  intros b h
  sorry

end base_of_524_l90_90349


namespace range_of_a_l90_90007

theorem range_of_a (x a : ℝ) (p : 0 < x ∧ x < 1)
  (q : (x - a) * (x - (a + 2)) ≤ 0) (h : ∀ x, (0 < x ∧ x < 1) → (x - a) * (x - (a + 2)) ≤ 0) :
  -1 ≤ a ∧ a ≤ 0 :=
sorry

end range_of_a_l90_90007


namespace power_equality_l90_90574

theorem power_equality (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
sorry

end power_equality_l90_90574


namespace determine_location_with_coords_l90_90608

-- Define the conditions as a Lean structure
structure Location where
  longitude : ℝ
  latitude : ℝ

-- Define the specific location given in option ①
def location_118_40 : Location :=
  {longitude := 118, latitude := 40}

-- Define the theorem and its statement
theorem determine_location_with_coords :
  ∃ loc : Location, loc = location_118_40 := 
  by
  sorry -- Placeholder for the proof

end determine_location_with_coords_l90_90608


namespace money_made_l90_90761

def initial_amount : ℕ := 26
def final_amount : ℕ := 52

theorem money_made : (final_amount - initial_amount) = 26 :=
by sorry

end money_made_l90_90761


namespace forty_percent_of_number_l90_90452

theorem forty_percent_of_number (N : ℝ) (h : (1 / 4) * (1 / 3) * (2 / 5) * N = 17) : 0.4 * N = 204 :=
sorry

end forty_percent_of_number_l90_90452


namespace ladybugs_with_spots_l90_90759

theorem ladybugs_with_spots (total_ladybugs : ℕ) (ladybugs_without_spots : ℕ) : total_ladybugs = 67082 ∧ ladybugs_without_spots = 54912 → total_ladybugs - ladybugs_without_spots = 12170 := by
  sorry

end ladybugs_with_spots_l90_90759


namespace math_preference_related_to_gender_l90_90197

-- Definitions for conditions
def total_students : ℕ := 100
def male_students : ℕ := 55
def female_students : ℕ := total_students - male_students -- 45
def likes_math : ℕ := 40
def female_likes_math : ℕ := 20
def female_not_like_math : ℕ := female_students - female_likes_math -- 25
def male_likes_math : ℕ := likes_math - female_likes_math -- 20
def male_not_like_math : ℕ := male_students - male_likes_math -- 35

-- Calculate Chi-square
def chi_square (a b c d : ℕ) : Float :=
  let numerator := (total_students * (a * d - b * c)^2).toFloat
  let denominator := ((a + b) * (c + d) * (a + c) * (b + d)).toFloat
  numerator / denominator

def k_square : Float := chi_square 20 35 20 25 -- Calculate with given values

-- Prove the result
theorem math_preference_related_to_gender :
  k_square > 7.879 :=
by
  sorry

end math_preference_related_to_gender_l90_90197


namespace coordinates_of_D_l90_90477
-- Importing the necessary library

-- Defining the conditions as given in the problem
def AB : ℝ × ℝ := (5, 3)
def C : ℝ × ℝ := (-1, 3)
def CD : ℝ × ℝ := (2 * 5, 2 * 3)

-- The target proof statement
theorem coordinates_of_D :
  ∃ D : ℝ × ℝ, CD = D - C ∧ D = (9, -3) :=
by
  sorry

end coordinates_of_D_l90_90477


namespace digit_B_divisible_by_9_l90_90981

-- Defining the condition for B making 762B divisible by 9
theorem digit_B_divisible_by_9 (B : ℕ) : (15 + B) % 9 = 0 ↔ B = 3 := 
by
  sorry

end digit_B_divisible_by_9_l90_90981


namespace amy_books_l90_90815

theorem amy_books (maddie_books : ℕ) (luisa_books : ℕ) (amy_luisa_more_than_maddie : ℕ) (h1 : maddie_books = 15) (h2 : luisa_books = 18) (h3 : amy_luisa_more_than_maddie = maddie_books + 9) : ∃ (amy_books : ℕ), amy_books = amy_luisa_more_than_maddie - luisa_books ∧ amy_books = 6 :=
by
  have total_books := 24
  sorry

end amy_books_l90_90815


namespace range_of_b_over_a_l90_90532

noncomputable def f (a b x : ℝ) : ℝ := (a * x - b / x - 2 * a) * Real.exp x

noncomputable def f' (a b x : ℝ) : ℝ := (b / x^2 + a * x - b / x - a) * Real.exp x

theorem range_of_b_over_a (a b : ℝ) (h₀ : a > 0) (h₁ : ∃ x : ℝ, 1 < x ∧ f a b x + f' a b x = 0) : 
  -1 < b / a := sorry

end range_of_b_over_a_l90_90532


namespace shaded_total_area_l90_90123

theorem shaded_total_area:
  ∀ (r₁ r₂ r₃ : ℝ),
  π * r₁ ^ 2 = 100 * π →
  r₂ = r₁ / 2 →
  r₃ = r₂ / 2 →
  (1 / 2) * (π * r₁ ^ 2) + (1 / 2) * (π * r₂ ^ 2) + (1 / 2) * (π * r₃ ^ 2) = 65.625 * π :=
by
  intro r₁ r₂ r₃ h₁ h₂ h₃
  sorry

end shaded_total_area_l90_90123


namespace solution_inequality_l90_90322

variable (a x : ℝ)

theorem solution_inequality (h : ∀ x, |x - a| + |x + 4| ≥ 1) : a ≤ -5 ∨ a ≥ -3 := by
  sorry

end solution_inequality_l90_90322


namespace part1_part2_l90_90994

noncomputable def f (x a : ℝ) : ℝ := abs (2 * x - 5 * a) + abs (2 * x + 1)
noncomputable def g (x : ℝ) : ℝ := abs (x - 1) + 3

-- (1)
theorem part1 (x : ℝ) : abs (g x) < 8 → -4 < x ∧ x < 6 :=
by
  sorry

-- (2)
theorem part2 (a : ℝ) : (∀ x1 : ℝ, ∃ x2 : ℝ, f x1 a = g x2) → (a ≥ 0.4 ∨ a ≤ -0.8) :=
by
  sorry

end part1_part2_l90_90994


namespace difference_between_multiplication_and_subtraction_l90_90289

theorem difference_between_multiplication_and_subtraction (x : ℤ) (h1 : x = 11) :
  (3 * x) - (26 - x) = 18 := by
  sorry

end difference_between_multiplication_and_subtraction_l90_90289


namespace vlad_taller_than_sister_l90_90486

-- Definitions based on the conditions
def vlad_feet : ℕ := 6
def vlad_inches : ℕ := 3
def sister_feet : ℕ := 2
def sister_inches : ℕ := 10
def inches_per_foot : ℕ := 12

-- Derived values for heights in inches
def vlad_height_in_inches : ℕ := (vlad_feet * inches_per_foot) + vlad_inches
def sister_height_in_inches : ℕ := (sister_feet * inches_per_foot) + sister_inches

-- Lean 4 statement for the proof problem
theorem vlad_taller_than_sister : vlad_height_in_inches - sister_height_in_inches = 41 := 
by 
  sorry

end vlad_taller_than_sister_l90_90486


namespace hyperbola_equation_l90_90996

theorem hyperbola_equation 
  (h k a c : ℝ)
  (center_cond : (h, k) = (3, -1))
  (vertex_cond : a = abs (2 - (-1)))
  (focus_cond : c = abs (7 - (-1)))
  (b : ℝ)
  (b_square : c^2 = a^2 + b^2) :
  h + k + a + b = 5 + Real.sqrt 55 := 
by
  -- Prove that given the conditions, the value of h + k + a + b is 5 + √55.
  sorry

end hyperbola_equation_l90_90996


namespace linear_function_m_l90_90678

theorem linear_function_m (m : ℤ) (h₁ : |m| = 1) (h₂ : m + 1 ≠ 0) : m = 1 := by
  sorry

end linear_function_m_l90_90678


namespace xy_square_value_l90_90417

theorem xy_square_value (x y : ℝ) (h1 : x * (x + y) = 24) (h2 : y * (x + y) = 72) : (x + y)^2 = 96 :=
by
  sorry

end xy_square_value_l90_90417


namespace weight_of_new_person_l90_90274

def total_weight_increase (num_people : ℕ) (weight_increase_per_person : ℝ) : ℝ :=
  num_people * weight_increase_per_person

def new_person_weight (old_person_weight : ℝ) (total_weight_increase : ℝ) : ℝ :=
  old_person_weight + total_weight_increase

theorem weight_of_new_person :
  let old_person_weight := 50
  let num_people := 8
  let weight_increase_per_person := 2.5
  new_person_weight old_person_weight (total_weight_increase num_people weight_increase_per_person) = 70 := 
by
  sorry

end weight_of_new_person_l90_90274


namespace bookseller_original_cost_l90_90522

theorem bookseller_original_cost
  (x y z : ℝ)
  (h1 : 1.10 * x = 11.00)
  (h2 : 1.10 * y = 16.50)
  (h3 : 1.10 * z = 24.20) :
  x + y + z = 47.00 := by
  sorry

end bookseller_original_cost_l90_90522


namespace triangle_altitude_l90_90188

theorem triangle_altitude {A b h : ℝ} (hA : A = 720) (hb : b = 40) (hArea : A = 1 / 2 * b * h) : h = 36 :=
by
  sorry

end triangle_altitude_l90_90188


namespace eval_expression_at_a_l90_90559

theorem eval_expression_at_a (a : ℝ) (h : a = 1 / 2) : (2 * a⁻¹ + a⁻¹ / 2) / a = 10 :=
by
  sorry

end eval_expression_at_a_l90_90559


namespace james_total_payment_l90_90182

noncomputable def total_amount_paid : ℕ :=
  let dirt_bike_count := 3
  let off_road_vehicle_count := 4
  let atv_count := 2
  let moped_count := 5
  let scooter_count := 3
  let dirt_bike_cost := dirt_bike_count * 150
  let off_road_vehicle_cost := off_road_vehicle_count * 300
  let atv_cost := atv_count * 450
  let moped_cost := moped_count * 200
  let scooter_cost := scooter_count * 100
  let registration_dirt_bike := dirt_bike_count * 25
  let registration_off_road_vehicle := off_road_vehicle_count * 25
  let registration_atv := atv_count * 30
  let registration_moped := moped_count * 15
  let registration_scooter := scooter_count * 20
  let maintenance_dirt_bike := dirt_bike_count * 50
  let maintenance_off_road_vehicle := off_road_vehicle_count * 75
  let maintenance_atv := atv_count * 100
  let maintenance_moped := moped_count * 60
  let total_cost_of_vehicles := dirt_bike_cost + off_road_vehicle_cost + atv_cost + moped_cost + scooter_cost
  let total_registration_costs := registration_dirt_bike + registration_off_road_vehicle + registration_atv + registration_moped + registration_scooter
  let total_maintenance_costs := maintenance_dirt_bike + maintenance_off_road_vehicle + maintenance_atv + maintenance_moped
  total_cost_of_vehicles + total_registration_costs + total_maintenance_costs

theorem james_total_payment : total_amount_paid = 5170 := by
  -- The proof would be written here
  sorry

end james_total_payment_l90_90182


namespace Tim_sweets_are_multiple_of_4_l90_90468

-- Define the conditions
def sweets_are_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

-- Given definitions
def Peter_sweets : ℕ := 44
def largest_possible_number_per_tray : ℕ := 4

-- Define the proposition to be proven
theorem Tim_sweets_are_multiple_of_4 (O : ℕ) (h1 : sweets_are_divisible_by_4 Peter_sweets) (h2 : sweets_are_divisible_by_4 largest_possible_number_per_tray) :
  sweets_are_divisible_by_4 O :=
sorry

end Tim_sweets_are_multiple_of_4_l90_90468


namespace max_ab_l90_90949

theorem max_ab (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 40) : 
  ab ≤ 400 :=
sorry

end max_ab_l90_90949


namespace method_one_cost_eq_300_method_two_cost_eq_300_method_one_more_cost_effective_l90_90210

noncomputable def method_one_cost (x : ℕ) : ℕ := 120 + 10 * x

noncomputable def method_two_cost (x : ℕ) : ℕ := 15 * x

theorem method_one_cost_eq_300 (x : ℕ) : method_one_cost x = 300 ↔ x = 18 :=
by sorry

theorem method_two_cost_eq_300 (x : ℕ) : method_two_cost x = 300 ↔ x = 20 :=
by sorry

theorem method_one_more_cost_effective (x : ℕ) :
  x ≥ 40 → method_one_cost x < method_two_cost x :=
by sorry

end method_one_cost_eq_300_method_two_cost_eq_300_method_one_more_cost_effective_l90_90210


namespace timmy_needs_speed_l90_90478

variable (s1 s2 s3 : ℕ) (extra_speed : ℕ)

theorem timmy_needs_speed
  (h_s1 : s1 = 36)
  (h_s2 : s2 = 34)
  (h_s3 : s3 = 38)
  (h_extra_speed : extra_speed = 4) :
  (s1 + s2 + s3) / 3 + extra_speed = 40 := 
sorry

end timmy_needs_speed_l90_90478


namespace cone_height_l90_90912

theorem cone_height (r h : ℝ) (π : ℝ) (Hπ : Real.pi = π) (slant_height : ℝ) (lateral_area : ℝ) (base_area : ℝ) 
  (H1 : slant_height = 2) 
  (H2 : lateral_area = 2 * π * r) 
  (H3 : base_area = π * r^2) 
  (H4 : lateral_area = 4 * base_area) 
  (H5 : r^2 + h^2 = slant_height^2) 
  : h = π / 2 := by 
sorry

end cone_height_l90_90912


namespace remainder_count_l90_90045

theorem remainder_count (n : ℕ) (h : n > 5) : 
  ∃ l : List ℕ, l.length = 5 ∧ ∀ x ∈ l, x ∣ 42 ∧ x > 5 := 
sorry

end remainder_count_l90_90045


namespace normal_level_short_of_capacity_l90_90237

noncomputable def total_capacity (water_amount : ℕ) (percentage : ℝ) : ℝ :=
  water_amount / percentage

noncomputable def normal_level (water_amount : ℕ) : ℕ :=
  water_amount / 2

theorem normal_level_short_of_capacity (water_amount : ℕ) (percentage : ℝ) (capacity : ℝ) (normal : ℕ) : 
  water_amount = 30 ∧ percentage = 0.75 ∧ capacity = total_capacity water_amount percentage ∧ normal = normal_level water_amount →
  (capacity - ↑normal) = 25 :=
by
  intros h
  sorry

end normal_level_short_of_capacity_l90_90237


namespace joggers_difference_l90_90315

theorem joggers_difference (Tyson_joggers Alexander_joggers Christopher_joggers : ℕ) 
  (h1 : Alexander_joggers = Tyson_joggers + 22) 
  (h2 : Christopher_joggers = 20 * Tyson_joggers)
  (h3 : Christopher_joggers = 80) : 
  Christopher_joggers - Alexander_joggers = 54 :=
by 
  sorry

end joggers_difference_l90_90315


namespace problem_solution_l90_90752

noncomputable def otimes (a b : ℝ) : ℝ := (a^3) / b

theorem problem_solution :
  (otimes (otimes 2 3) 4) - (otimes 2 (otimes 3 4)) = (32/9) :=
by
  sorry

end problem_solution_l90_90752


namespace power_product_l90_90141

theorem power_product (m n : ℕ) (hm : 2 < m) (hn : 0 < n) : 
  (2^m - 1) * (2^n + 1) > 0 :=
by 
  sorry

end power_product_l90_90141


namespace find_line_equation_l90_90403

theorem find_line_equation (a b : ℝ) :
  (2 * a + 3 * b = 0 ∧ a * b < 0) ↔ (3 * a - 2 * b = 0 ∨ a - b + 1 = 0) :=
by
  sorry

end find_line_equation_l90_90403


namespace ratio_area_rectangle_to_square_l90_90641

variable (s : ℝ)
variable (area_square : ℝ := s^2)
variable (longer_side_rectangle : ℝ := 1.2 * s)
variable (shorter_side_rectangle : ℝ := 0.85 * s)
variable (area_rectangle : ℝ := longer_side_rectangle * shorter_side_rectangle)

theorem ratio_area_rectangle_to_square :
  area_rectangle / area_square = 51 / 50 := by
  sorry

end ratio_area_rectangle_to_square_l90_90641


namespace kamals_salary_change_l90_90129

theorem kamals_salary_change : 
  ∀ (S : ℝ), ((S * 0.5 * 1.3 * 0.8 - S) / S) * 100 = -48 :=
by
  intro S
  sorry

end kamals_salary_change_l90_90129


namespace alphabet_letter_count_l90_90691

def sequence_count : Nat :=
  let total_sequences := 2^7
  let sequences_per_letter := 1 + 7 -- 1 correct sequence + 7 single-bit alterations
  total_sequences / sequences_per_letter

theorem alphabet_letter_count : sequence_count = 16 :=
  by
    -- Proof placeholder
    sorry

end alphabet_letter_count_l90_90691


namespace dilation_image_l90_90120

theorem dilation_image :
  let z_0 := (1 : ℂ) + 2 * I
  let k := (2 : ℂ)
  let z_1 := (3 : ℂ) + I
  let z := z_0 + k * (z_1 - z_0)
  z = 5 :=
by
  sorry

end dilation_image_l90_90120


namespace range_of_a_l90_90427

-- Define the set A
def A (a x : ℝ) := 6 * x + a > 0

-- Theorem stating the range of a given the conditions
theorem range_of_a (a : ℝ) (h : ¬ A a 1) : a ≤ -6 :=
by
  -- Here we would provide the proof
  sorry

end range_of_a_l90_90427


namespace remaining_angle_obtuse_cut_by_right_is_acute_remaining_angle_straight_cut_by_acute_is_obtuse_l90_90706

-- Definitions of angle types
def obtuse_angle (θ : ℝ) := θ > 90 ∧ θ < 180
def right_angle (θ : ℝ) := θ = 90
def acute_angle (θ : ℝ) := θ > 0 ∧ θ < 90
def straight_angle (θ : ℝ) := θ = 180

-- Proposition 1: Remaining angle when an obtuse angle is cut by a right angle is acute
theorem remaining_angle_obtuse_cut_by_right_is_acute (θ : ℝ) (φ : ℝ) 
    (h1 : obtuse_angle θ) (h2 : right_angle φ) : acute_angle (θ - φ) :=
  sorry

-- Proposition 2: Remaining angle when a straight angle is cut by an acute angle is obtuse
theorem remaining_angle_straight_cut_by_acute_is_obtuse (α : ℝ) (β : ℝ) 
    (h1 : straight_angle α) (h2 : acute_angle β) : obtuse_angle (α - β) :=
  sorry

end remaining_angle_obtuse_cut_by_right_is_acute_remaining_angle_straight_cut_by_acute_is_obtuse_l90_90706


namespace A_inter_complement_B_eq_01_l90_90115

open Set

noncomputable def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x^2 - 2 * x < 0}
def B : Set ℝ := {x | x ≥ 1}
def complement_B : Set ℝ := U \ B

theorem A_inter_complement_B_eq_01 : A ∩ complement_B = (Set.Ioo 0 1) := 
by 
  sorry

end A_inter_complement_B_eq_01_l90_90115


namespace glass_pieces_same_color_l90_90932

theorem glass_pieces_same_color (r y b : ℕ) (h : r + y + b = 2002) :
  (∃ k : ℕ, ∀ n, n ≥ k → (r + y + b) = n ∧ (r = 0 ∨ y = 0 ∨ b = 0)) ∧
  (∀ (r1 y1 b1 r2 y2 b2 : ℕ),
    r1 + y1 + b1 = 2002 →
    r2 + y2 + b2 = 2002 →
    (∃ k : ℕ, ∀ n, n ≥ k → (r1 = 0 ∨ y1 = 0 ∨ b1 = 0)) →
    (∃ l : ℕ, ∀ m, m ≥ l → (r2 = 0 ∨ y2 = 0 ∨ b2 = 0)) →
    r1 = r2 ∧ y1 = y2 ∧ b1 = b2):=
by
  sorry

end glass_pieces_same_color_l90_90932


namespace milk_water_ratio_l90_90267

theorem milk_water_ratio (x y : ℝ) (h1 : 5 * x + 2 * y = 4 * x + 7 * y) :
  x / y = 5 :=
by 
  sorry

end milk_water_ratio_l90_90267


namespace non_upgraded_sensor_ratio_l90_90570

theorem non_upgraded_sensor_ratio 
  (N U S : ℕ) 
  (units : ℕ := 24) 
  (fraction_upgraded : ℚ := 1 / 7) 
  (fraction_non_upgraded : ℚ := 6 / 7)
  (h1 : U / S = fraction_upgraded)
  (h2 : units * N = (fraction_non_upgraded * S)) : 
  N / U = 1 / 4 := 
by 
  sorry

end non_upgraded_sensor_ratio_l90_90570


namespace sum_of_integers_l90_90683

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 8) (h3 : x * y = 180) : x + y = 28 := 
by
  sorry

end sum_of_integers_l90_90683


namespace composite_of_squares_l90_90229

theorem composite_of_squares (n : ℕ) (h1 : 8 * n + 1 = x^2) (h2 : 24 * n + 1 = y^2) (h3 : n > 1) : ∃ a b : ℕ, a ∣ (8 * n + 3) ∧ b ∣ (8 * n + 3) ∧ a ≠ 1 ∧ b ≠ 1 ∧ a ≠ (8 * n + 3) ∧ b ≠ (8 * n + 3) := by
  sorry

end composite_of_squares_l90_90229


namespace greatest_two_digit_product_12_l90_90507

-- Definition of a two-digit whole number
def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

-- Definition of the digit product condition
def digits_product (n : ℕ) (p : ℕ) : Prop := ∃ (d1 d2 : ℕ), d1 * d2 = p ∧ n = 10 * d1 + d2

-- The main theorem stating the greatest two-digit number whose digits multiply to 12 is 62
theorem greatest_two_digit_product_12 : ∀ (n : ℕ), is_two_digit (n) → digits_product (n) 12 → n <= 62 :=
by {
    sorry -- Proof of the theorem
}

end greatest_two_digit_product_12_l90_90507


namespace nancy_more_money_l90_90858

def jade_available := 1920
def giraffe_jade := 120
def elephant_jade := 2 * giraffe_jade
def giraffe_price := 150
def elephant_price := 350

def num_giraffes := jade_available / giraffe_jade
def num_elephants := jade_available / elephant_jade
def revenue_giraffes := num_giraffes * giraffe_price
def revenue_elephants := num_elephants * elephant_price

def revenue_difference := revenue_elephants - revenue_giraffes

theorem nancy_more_money : revenue_difference = 400 :=
by sorry

end nancy_more_money_l90_90858


namespace polar_to_cartesian_l90_90418

-- Definitions for the polar coordinates conversion
noncomputable def polar_to_cartesian_eq (C : ℝ → ℝ → Prop) :=
  ∀ (ρ θ : ℝ), (ρ^2 * (1 + 3 * (Real.sin θ)^2) = 4) → C (ρ * (Real.cos θ)) (ρ * (Real.sin θ))

-- Define the Cartesian equation
def cartesian_eq (x y : ℝ) : Prop :=
  (x^2 / 4 + y^2 = 1)

-- The main theorem
theorem polar_to_cartesian 
  (C : ℝ → ℝ → Prop)
  (h : polar_to_cartesian_eq C) :
  ∀ x y : ℝ, C x y ↔ cartesian_eq x y :=
by
  sorry

end polar_to_cartesian_l90_90418


namespace calculate_N_l90_90705

theorem calculate_N (h : (25 / 100) * N = (55 / 100) * 3010) : N = 6622 :=
by
  sorry

end calculate_N_l90_90705


namespace set_of_x_satisfying_inequality_l90_90914

theorem set_of_x_satisfying_inequality : 
  {x : ℝ | (x - 2)^2 < 9} = {x : ℝ | -1 < x ∧ x < 5} :=
by
  sorry

end set_of_x_satisfying_inequality_l90_90914


namespace capacity_of_first_bucket_is_3_l90_90140

variable (C : ℝ)

theorem capacity_of_first_bucket_is_3 
  (h1 : 48 / C = 48 / 3 - 4) : 
  C = 3 := 
  sorry

end capacity_of_first_bucket_is_3_l90_90140


namespace max_students_gcd_l90_90309

def numPens : Nat := 1802
def numPencils : Nat := 1203
def numErasers : Nat := 1508
def numNotebooks : Nat := 2400

theorem max_students_gcd : Nat.gcd (Nat.gcd (Nat.gcd numPens numPencils) numErasers) numNotebooks = 1 := by
  sorry

end max_students_gcd_l90_90309


namespace solve_for_m_l90_90974

theorem solve_for_m (x y m : ℤ) (h1 : x - 2 * y = -3) (h2 : 2 * x + 3 * y = m - 1) (h3 : x = -y) : m = 2 :=
by
  sorry

end solve_for_m_l90_90974


namespace perimeter_of_shaded_area_l90_90540

theorem perimeter_of_shaded_area (AB AD : ℝ) (h1 : AB = 14) (h2 : AD = 12) : 
  2 * AB + 2 * AD = 52 := 
by
  sorry

end perimeter_of_shaded_area_l90_90540


namespace domain_range_sum_l90_90853

def f (x : ℝ) : ℝ := -x^2 + 2 * x

theorem domain_range_sum (m n : ℝ) (hmn : ∀ x, m ≤ x ∧ x ≤ n → (f x = 3 * x)) : m + n = -1 :=
by
  sorry

end domain_range_sum_l90_90853


namespace seq_period_3_l90_90150

def seq (a : ℕ → ℚ) := ∀ n, 
  (0 ≤ a n ∧ a n < 1) ∧ (
  (0 ≤ a n ∧ a n < 1/2 → a (n+1) = 2 * a n) ∧ 
  (1/2 ≤ a n ∧ a n < 1 → a (n+1) = 2 * a n - 1))

theorem seq_period_3 (a : ℕ → ℚ) (h : seq a) (h1 : a 1 = 6 / 7) : 
  a 2016 = 3 / 7 := 
sorry

end seq_period_3_l90_90150


namespace mean_reciprocals_first_three_composites_l90_90746

theorem mean_reciprocals_first_three_composites :
  (1 / 4 + 1 / 6 + 1 / 8) / 3 = (13 : ℚ) / 72 := 
by
  sorry

end mean_reciprocals_first_three_composites_l90_90746


namespace greatest_possible_multiple_of_4_l90_90049

theorem greatest_possible_multiple_of_4 (x : ℕ) (h1 : x % 4 = 0) (h2 : x^2 < 400) : x ≤ 16 :=
by 
sorry

end greatest_possible_multiple_of_4_l90_90049


namespace dog_catches_fox_at_distance_l90_90677

def initial_distance : ℝ := 30
def dog_leap_distance : ℝ := 2
def fox_leap_distance : ℝ := 1
def dog_leaps_per_time_unit : ℝ := 2
def fox_leaps_per_time_unit : ℝ := 3

noncomputable def dog_speed : ℝ := dog_leaps_per_time_unit * dog_leap_distance
noncomputable def fox_speed : ℝ := fox_leaps_per_time_unit * fox_leap_distance
noncomputable def relative_speed : ℝ := dog_speed - fox_speed
noncomputable def time_to_catch := initial_distance / relative_speed
noncomputable def distance_dog_runs := time_to_catch * dog_speed

theorem dog_catches_fox_at_distance :
  distance_dog_runs = 120 :=
  by sorry

end dog_catches_fox_at_distance_l90_90677


namespace visitors_on_saturday_l90_90639

theorem visitors_on_saturday (S : ℕ) (h1 : S + (S + 40) = 440) : S = 200 := by
  sorry

end visitors_on_saturday_l90_90639


namespace right_triangle_side_length_l90_90252

theorem right_triangle_side_length (a c b : ℕ) (h1 : a = 3) (h2 : c = 5) (h3 : c^2 = a^2 + b^2) : b = 4 :=
sorry

end right_triangle_side_length_l90_90252


namespace caroline_socks_gift_l90_90615

theorem caroline_socks_gift :
  ∀ (initial lost donated_fraction purchased total received),
    initial = 40 →
    lost = 4 →
    donated_fraction = 2 / 3 →
    purchased = 10 →
    total = 25 →
    received = total - (initial - lost - donated_fraction * (initial - lost) + purchased) →
    received = 3 :=
by
  intros initial lost donated_fraction purchased total received
  intro h_initial h_lost h_donated_fraction h_purchased h_total h_received
  sorry

end caroline_socks_gift_l90_90615


namespace floor_neg_seven_over_four_l90_90704

theorem floor_neg_seven_over_four : (Int.floor (-7 / 4 : ℚ)) = -2 := 
by
  sorry

end floor_neg_seven_over_four_l90_90704


namespace john_gallons_of_gas_l90_90504

theorem john_gallons_of_gas
  (rental_cost : ℝ)
  (gas_cost_per_gallon : ℝ)
  (mile_cost : ℝ)
  (miles_driven : ℝ)
  (total_cost : ℝ)
  (rental_cost_val : rental_cost = 150)
  (gas_cost_per_gallon_val : gas_cost_per_gallon = 3.50)
  (mile_cost_val : mile_cost = 0.50)
  (miles_driven_val : miles_driven = 320)
  (total_cost_val : total_cost = 338) :
  ∃ gallons_of_gas : ℝ, gallons_of_gas = 8 :=
by
  sorry

end john_gallons_of_gas_l90_90504


namespace find_k_l90_90114

theorem find_k {k : ℚ} (h : (3 : ℚ)^3 + 7 * (3 : ℚ)^2 + k * (3 : ℚ) + 23 = 0) : k = -113 / 3 :=
by
  sorry

end find_k_l90_90114


namespace cubic_sum_l90_90396

theorem cubic_sum (a b c : ℝ) (h1 : a + b + c = 5) (h2 : a * b + b * c + c * a = 7) (h3 : a * b * c = 2) :
  a^3 + b^3 + c^3 = 26 :=
by
  sorry

end cubic_sum_l90_90396


namespace sequence_general_term_l90_90560

theorem sequence_general_term :
  ∀ (a : ℕ → ℝ), a 1 = 2 ^ (5 / 2) ∧ 
  (∀ n, a (n+1) = 4 * (4 * a n) ^ (1/4)) →
  ∀ n, a n = 2 ^ (10 / 3 * (1 - 1 / 4 ^ n)) :=
by
  intros a h1 h_rec
  sorry

end sequence_general_term_l90_90560


namespace find_multiple_l90_90367

/-- 
Given:
1. Hank Aaron hit 755 home runs.
2. Dave Winfield hit 465 home runs.
3. Hank Aaron has 175 fewer home runs than a certain multiple of the number that Dave Winfield has.

Prove:
The multiple of Dave Winfield's home runs that Hank Aaron's home runs are compared to is 2.
-/
def multiple_of_dave_hr (ha_hr dw_hr diff : ℕ) (m : ℕ) : Prop :=
  ha_hr + diff = m * dw_hr

theorem find_multiple :
  multiple_of_dave_hr 755 465 175 2 :=
by
  sorry

end find_multiple_l90_90367


namespace mary_total_nickels_l90_90415

theorem mary_total_nickels : (7 + 12 + 9 = 28) :=
by
  sorry

end mary_total_nickels_l90_90415


namespace triangle_probability_l90_90902

open Classical

theorem triangle_probability :
  let a := 5
  let b := 6
  let lengths := [1, 2, 6, 11]
  let valid_third_side x := 1 < x ∧ x < 11
  let valid_lengths := lengths.filter valid_third_side
  let probability := valid_lengths.length / lengths.length
  probability = 1 / 2 :=
by {
  sorry
}

end triangle_probability_l90_90902


namespace raman_profit_percentage_l90_90264

theorem raman_profit_percentage
  (cost1 weight1 rate1 : ℕ) (cost2 weight2 rate2 : ℕ) (total_cost_mix total_weight mixing_rate selling_rate profit profit_percentage : ℕ)
  (h_cost1 : cost1 = weight1 * rate1)
  (h_cost2 : cost2 = weight2 * rate2)
  (h_total_cost_mix : total_cost_mix = cost1 + cost2)
  (h_total_weight : total_weight = weight1 + weight2)
  (h_mixing_rate : mixing_rate = total_cost_mix / total_weight)
  (h_selling_price : selling_rate * total_weight = profit + total_cost_mix)
  (h_profit : profit = selling_rate * total_weight - total_cost_mix)
  (h_profit_percentage : profit_percentage = (profit * 100) / total_cost_mix)
  (h_weight1 : weight1 = 54)
  (h_rate1 : rate1 = 150)
  (h_weight2 : weight2 = 36)
  (h_rate2 : rate2 = 125)
  (h_selling_rate_value : selling_rate = 196) :
  profit_percentage = 40 :=
sorry

end raman_profit_percentage_l90_90264


namespace range_of_a_minus_b_l90_90337

theorem range_of_a_minus_b (a b : ℝ) (ha : 0 < a ∧ a < 1) (hb : 2 < b ∧ b < 4) : -4 < a - b ∧ a - b < -1 :=
by
  sorry

end range_of_a_minus_b_l90_90337


namespace max_intersections_cos_circle_l90_90248

theorem max_intersections_cos_circle :
  let circle := λ x y => (x - 4)^2 + y^2 = 25
  let cos_graph := λ x => (x, Real.cos x)
  ∀ x y, (circle x y ∧ y = Real.cos x) → (∃ (p : ℕ), p ≤ 8) := sorry

end max_intersections_cos_circle_l90_90248


namespace old_manufacturing_cost_l90_90430

theorem old_manufacturing_cost (P : ℝ) (h1 : 50 = 0.50 * P) : 0.60 * P = 60 :=
by
  sorry

end old_manufacturing_cost_l90_90430


namespace angle_A_measure_l90_90814

variable {a b c A : ℝ}

def vector_m (b c a : ℝ) : ℝ × ℝ := (b, c - a)
def vector_n (b c a : ℝ) : ℝ × ℝ := (b - c, c + a)

theorem angle_A_measure (h_perpendicular : (vector_m b c a).1 * (vector_n b c a).1 + (vector_m b c a).2 * (vector_n b c a).2 = 0) :
  A = 2 * π / 3 := sorry

end angle_A_measure_l90_90814


namespace total_blossoms_l90_90467

theorem total_blossoms (first second third : ℕ) (h1 : first = 2) (h2 : second = 2 * first) (h3 : third = 4 * second) : first + second + third = 22 :=
by
  sorry

end total_blossoms_l90_90467


namespace polygon_triangle_division_l90_90687

theorem polygon_triangle_division (n k : ℕ) (h : k * 3 = n * 3 - 6) : k ≥ n - 2 := sorry

end polygon_triangle_division_l90_90687


namespace ab_value_l90_90269

theorem ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a ^ 2 + b ^ 2 = 35) : a * b = 13 :=
by
  sorry

end ab_value_l90_90269


namespace zeros_not_adjacent_probability_l90_90928

def total_arrangements : ℕ := Nat.factorial 5

def adjacent_arrangements : ℕ := 2 * Nat.factorial 4

def probability_not_adjacent : ℚ := 
  1 - (adjacent_arrangements / total_arrangements)

theorem zeros_not_adjacent_probability :
  probability_not_adjacent = 0.6 := 
by 
  sorry

end zeros_not_adjacent_probability_l90_90928


namespace trader_loss_percent_l90_90277

theorem trader_loss_percent :
  let SP1 : ℝ := 404415
  let SP2 : ℝ := 404415
  let gain_percent : ℝ := 15 / 100
  let loss_percent : ℝ := 15 / 100
  let CP1 : ℝ := SP1 / (1 + gain_percent)
  let CP2 : ℝ := SP2 / (1 - loss_percent)
  let TCP : ℝ := CP1 + CP2
  let TSP : ℝ := SP1 + SP2
  let overall_loss : ℝ := TSP - TCP
  let overall_loss_percent : ℝ := (overall_loss / TCP) * 100
  overall_loss_percent = -2.25 := 
sorry

end trader_loss_percent_l90_90277


namespace smallest_n_is_1770_l90_90464

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 1000) + (n % 1000) / 100 + (n % 100) / 10 + (n % 10)

def is_smallest_n (n : ℕ) : Prop :=
  n = sum_of_digits n + 1755 ∧ (∀ m : ℕ, (m < n → m ≠ sum_of_digits m + 1755))

theorem smallest_n_is_1770 : is_smallest_n 1770 :=
sorry

end smallest_n_is_1770_l90_90464


namespace non_congruent_right_triangles_count_l90_90538

def isRightTriangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

def areaEqualsFourTimesPerimeter (a b c : ℕ) : Prop :=
  a * b = 8 * (a + b + c)

theorem non_congruent_right_triangles_count :
  {n : ℕ // ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ isRightTriangle a b c ∧ areaEqualsFourTimesPerimeter a b c ∧ n = 3} := sorry

end non_congruent_right_triangles_count_l90_90538


namespace stock_price_after_two_years_l90_90660

def initial_price : ℝ := 120

def first_year_increase (p : ℝ) : ℝ := p * 2

def second_year_decrease (p : ℝ) : ℝ := p * 0.30

def final_price (initial : ℝ) : ℝ :=
  let after_first_year := first_year_increase initial
  after_first_year - second_year_decrease after_first_year

theorem stock_price_after_two_years : final_price initial_price = 168 :=
by
  sorry

end stock_price_after_two_years_l90_90660


namespace find_tangent_value_l90_90213

noncomputable def tangent_value (a : ℝ) : Prop :=
  ∃ (x₀ y₀ : ℝ), y₀ = x₀ + 1 ∧ y₀ = Real.log (x₀ + a) ∧
  (1 / (x₀ + a) = 1)

theorem find_tangent_value : tangent_value 2 :=
  sorry

end find_tangent_value_l90_90213


namespace angle_with_same_terminal_side_l90_90738

-- Given conditions in the problem: angles to choose from
def angles : List ℕ := [60, 70, 100, 130]

-- Definition of the equivalence relation (angles having the same terminal side)
def same_terminal_side (θ α : ℕ) : Prop :=
  ∃ k : ℤ, θ = α + k * 360

-- Proof goal: 420° has the same terminal side as one of the angles in the list
theorem angle_with_same_terminal_side :
  ∃ α ∈ angles, same_terminal_side 420 α :=
sorry  -- proof not required

end angle_with_same_terminal_side_l90_90738


namespace water_consumption_eq_l90_90373

-- Define all conditions
variables (x : ℝ) (improvement : ℝ := 0.8) (water : ℝ := 80) (days_difference : ℝ := 5)

-- State the theorem
theorem water_consumption_eq (h : improvement = 0.8) (initial_water := 80) (difference := 5) : 
  initial_water / x - (initial_water * improvement) / x = difference :=
sorry

end water_consumption_eq_l90_90373


namespace polynomial_roots_absolute_sum_l90_90326

theorem polynomial_roots_absolute_sum (p q r : ℤ) (h1 : p + q + r = 0) (h2 : p * q + q * r + r * p = -2027) :
  |p| + |q| + |r| = 98 := 
sorry

end polynomial_roots_absolute_sum_l90_90326


namespace min_value_of_x_plus_y_l90_90790

theorem min_value_of_x_plus_y (x y : ℝ) (h1: y ≠ 0) (h2: 1 / y = (x - 1) / 2) : x + y ≥ 2 * Real.sqrt 2 := by
  sorry

end min_value_of_x_plus_y_l90_90790


namespace mary_has_more_money_than_marco_l90_90414

variable (Marco Mary : ℕ)

theorem mary_has_more_money_than_marco
    (h1 : Marco = 24)
    (h2 : Mary = 15)
    (h3 : ∃ maryNew : ℕ, maryNew = Mary + Marco / 2 - 5)
    (h4 : ∃ marcoNew : ℕ, marcoNew = Marco / 2) :
    ∃ diff : ℕ, diff = maryNew - marcoNew ∧ diff = 10 := 
by 
    sorry

end mary_has_more_money_than_marco_l90_90414


namespace no_such_arrangement_exists_l90_90037

theorem no_such_arrangement_exists :
  ¬ ∃ (f : ℕ → ℕ) (c : ℕ), 
    (∀ n, 1 ≤ f n ∧ f n ≤ 1331) ∧
    (∀ x y z, f (x + 11 * y + 121 * z) = c → f ((x+1) + 11 * y + 121 * z) = c + 8) ∧
    (∀ x y z, f (x + 11 * y + 121 * z) = c → f (x + 11 * (y+1) + 121 * z) = c + 9) :=
sorry

end no_such_arrangement_exists_l90_90037


namespace clock_correction_time_l90_90122

theorem clock_correction_time :
  let time_loss_per_day : ℝ := 15 / 60
  let days_elapsed : ℝ := 9 + 6 / 24
  let total_time_loss : ℝ := (15 / 1440) * (days_elapsed * 24)
  let correction : ℝ := total_time_loss * 60
  correction = 138.75 :=
by
  let time_loss_per_day : ℝ := 15 / 60
  let days_elapsed : ℝ := 9 + 6 / 24
  let total_time_loss : ℝ := (15 / 1440) * (days_elapsed * 24)
  let correction : ℝ := total_time_loss * 60
  have : correction = 138.75 := sorry
  exact this

end clock_correction_time_l90_90122


namespace exists_numbers_with_prime_sum_and_product_l90_90461

open Nat

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem exists_numbers_with_prime_sum_and_product :
  ∃ a b c : ℕ, is_prime (a + b + c) ∧ is_prime (a * b * c) :=
  by
    -- First import the prime definitions and variables.
    let a := 1
    let b := 1
    let c := 3
    have h1 : is_prime (a + b + c) := by sorry
    have h2 : is_prime (a * b * c) := by sorry
    exact ⟨a, b, c, h1, h2⟩

end exists_numbers_with_prime_sum_and_product_l90_90461


namespace fraction_meaningful_iff_l90_90492

theorem fraction_meaningful_iff (x : ℝ) : (∃ y, y = 1 / (x + 1)) ↔ x ≠ -1 :=
by
  sorry

end fraction_meaningful_iff_l90_90492


namespace abs_neg_three_l90_90375

noncomputable def abs_val (a : ℤ) : ℤ :=
  if a < 0 then -a else a

theorem abs_neg_three : abs_val (-3) = 3 :=
by
  sorry

end abs_neg_three_l90_90375


namespace area_of_triangle_with_medians_l90_90212

theorem area_of_triangle_with_medians
  (s_a s_b s_c : ℝ) :
  (∃ t : ℝ, t = (1 / 3 : ℝ) * ((s_a + s_b + s_c) * (s_b + s_c - s_a) * (s_a + s_c - s_b) * (s_a + s_b - s_c)).sqrt) :=
sorry

end area_of_triangle_with_medians_l90_90212


namespace differentiable_increasing_necessary_but_not_sufficient_l90_90692

variable {f : ℝ → ℝ}

theorem differentiable_increasing_necessary_but_not_sufficient (h_diff : ∀ x : ℝ, DifferentiableAt ℝ f x) :
  (∀ x : ℝ, 0 < deriv f x) → ∀ x : ℝ, MonotoneOn f (Set.univ : Set ℝ) ∧ ¬ (∀ x : ℝ, MonotoneOn f (Set.univ : Set ℝ) → ∀ x : ℝ, 0 < deriv f x) := 
sorry

end differentiable_increasing_necessary_but_not_sufficient_l90_90692


namespace even_function_a_eq_one_l90_90201

noncomputable def f (x a : ℝ) : ℝ := x * Real.log (x + Real.sqrt (a + x ^ 2))

theorem even_function_a_eq_one (a : ℝ) :
  (∀ x : ℝ, f x a = f (-x) a) → a = 1 :=
by
  sorry

end even_function_a_eq_one_l90_90201


namespace john_needs_total_planks_l90_90338

theorem john_needs_total_planks : 
  let large_planks := 12
  let small_planks := 17
  large_planks + small_planks = 29 :=
by
  sorry

end john_needs_total_planks_l90_90338


namespace both_pumps_drain_lake_l90_90886

theorem both_pumps_drain_lake (T : ℝ) (h₁ : 1 / 9 + 1 / 6 = 5 / 18) : 
  (5 / 18) * T = 1 → T = 18 / 5 := sorry

end both_pumps_drain_lake_l90_90886


namespace red_chairs_count_l90_90647

-- Given conditions
variables {R Y B : ℕ} -- Assuming the number of chairs are natural numbers

-- Main theorem statement
theorem red_chairs_count : 
  Y = 4 * R ∧ B = Y - 2 ∧ R + Y + B = 43 -> R = 5 :=
by
  sorry

end red_chairs_count_l90_90647


namespace solve_m_value_l90_90297

-- Definitions for conditions
def hyperbola_eq (m : ℝ) : Prop := ∀ x y : ℝ, 3 * m * x^2 - m * y^2 = 3
def has_focus (m : ℝ) : Prop := (∃ f1 f2 : ℝ, f1 = 0 ∧ f2 = 2)

-- Statement of the problem to prove
theorem solve_m_value (m : ℝ) (h_eq : hyperbola_eq m) (h_focus : has_focus m) : m = -1 :=
sorry

end solve_m_value_l90_90297


namespace cats_not_eating_cheese_or_tuna_l90_90498

-- Define the given conditions
variables (n C T B : ℕ)

-- State the problem in Lean
theorem cats_not_eating_cheese_or_tuna 
  (h_n : n = 100)  
  (h_C : C = 25)  
  (h_T : T = 70)  
  (h_B : B = 15)
  : n - (C - B + T - B + B) = 20 := 
by {
  -- Insert proof here
  sorry
}

end cats_not_eating_cheese_or_tuna_l90_90498


namespace smallest_rel_prime_l90_90026

theorem smallest_rel_prime (n : ℕ) (h : n > 1) (rel_prime : ∀ p ∈ [2, 3, 5, 7], ¬ p ∣ n) : n = 11 :=
by sorry

end smallest_rel_prime_l90_90026


namespace quadratic_has_single_solution_l90_90006

theorem quadratic_has_single_solution (k : ℚ) : 
  (∀ x : ℚ, 3 * x^2 - 7 * x + k = 0 → x = 7 / 6) ↔ k = 49 / 12 := 
by
  sorry

end quadratic_has_single_solution_l90_90006


namespace cakes_served_at_lunch_today_l90_90066

variable (L : ℕ)
variable (dinnerCakes : ℕ) (yesterdayCakes : ℕ) (totalCakes : ℕ)

theorem cakes_served_at_lunch_today :
  (dinnerCakes = 6) → (yesterdayCakes = 3) → (totalCakes = 14) → (L + dinnerCakes + yesterdayCakes = totalCakes) → L = 5 :=
by
  intros h_dinner h_yesterday h_total h_eq
  sorry

end cakes_served_at_lunch_today_l90_90066


namespace range_of_x_when_a_is_1_and_p_and_q_are_true_range_of_a_when_p_necessary_for_q_l90_90305

-- Define the propositions p and q
def p (x a : ℝ) := (x - a) * (x - 3 * a) < 0
def q (x : ℝ) := x^2 - 5 * x + 6 < 0

-- Question 1: When a = 1, if p ∧ q is true, determine the range of x
theorem range_of_x_when_a_is_1_and_p_and_q_are_true :
  ∀ x, p x 1 ∧ q x → 2 < x ∧ x < 3 :=
by
  sorry

-- Question 2: If p is a necessary but not sufficient condition for q, determine the range of a
theorem range_of_a_when_p_necessary_for_q :
  ∀ a, (∀ x, q x → p x a) ∧ ¬ (∀ x, p x a → q x) → 1 ≤ a ∧ a ≤ 2 :=
by
  sorry

end range_of_x_when_a_is_1_and_p_and_q_are_true_range_of_a_when_p_necessary_for_q_l90_90305


namespace solve_for_a_l90_90136

theorem solve_for_a (a : ℝ) : 
  (2 * a + 16 + 3 * a - 8) / 2 = 69 → a = 26 :=
by
  sorry

end solve_for_a_l90_90136


namespace evaluate_fraction_l90_90975

theorem evaluate_fraction : (1 / (2 + (1 / (3 + (1 / 4))))) = 13 / 30 :=
by
  sorry

end evaluate_fraction_l90_90975


namespace find_f_1002_l90_90245

noncomputable def f : ℕ → ℝ :=
  sorry

theorem find_f_1002 (f : ℕ → ℝ) 
  (h : ∀ a b n : ℕ, a + b = 2^n → f a + f b = n^2) :
  f 1002 = 21 :=
sorry

end find_f_1002_l90_90245


namespace hotdogs_total_l90_90947

theorem hotdogs_total:
  let e := 2.5
  let l := 2 * (e * 2)
  let m := 7
  let h := 1.5 * (e * 2)
  let z := 0.5
  (e * 2 + l + m + h + z) = 30 := 
by
  sorry

end hotdogs_total_l90_90947


namespace relay_race_solution_l90_90012

variable (Sadie_time : ℝ) (Sadie_speed : ℝ)
variable (Ariana_time : ℝ) (Ariana_speed : ℝ)
variable (Sarah_speed : ℝ)
variable (total_distance : ℝ)

def relay_race_time : Prop :=
  let Sadie_distance := Sadie_time * Sadie_speed
  let Ariana_distance := Ariana_time * Ariana_speed
  let Sarah_distance := total_distance - Sadie_distance - Ariana_distance
  let Sarah_time := Sarah_distance / Sarah_speed
  Sadie_time + Ariana_time + Sarah_time = 4.5

theorem relay_race_solution (h1: Sadie_time = 2) (h2: Sadie_speed = 3)
  (h3: Ariana_time = 0.5) (h4: Ariana_speed = 6)
  (h5: Sarah_speed = 4) (h6: total_distance = 17) :
  relay_race_time Sadie_time Sadie_speed Ariana_time Ariana_speed Sarah_speed total_distance :=
by
  sorry

end relay_race_solution_l90_90012


namespace min_omega_l90_90556

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + 1)
noncomputable def g (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * (x - 1) + 1)

def condition1 (ω : ℝ) : Prop := ω > 0
def condition2 (ω : ℝ) (x : ℝ) : Prop := g ω x = Real.sin (ω * x - ω + 1)
def condition3 (ω : ℝ) (k : ℤ) : Prop := ∃ k : ℤ, ω = 1 - k * Real.pi

theorem min_omega (ω : ℝ) (k : ℤ) (x : ℝ) : condition1 ω → condition2 ω x → condition3 ω k → ω = 1 :=
by
  intros h1 h2 h3
  sorry

end min_omega_l90_90556


namespace sum_roots_of_quadratic_eq_l90_90857

theorem sum_roots_of_quadratic_eq (a b c: ℝ) (x: ℝ) :
    (a = 1) →
    (b = -7) →
    (c = -9) →
    (x ^ 2 - 7 * x + 2 = 11) →
    (∃ r1 r2 : ℝ, x ^ 2 - 7 * x - 9 = 0 ∧ r1 + r2 = 7) :=
by
  sorry

end sum_roots_of_quadratic_eq_l90_90857


namespace intersection_setA_setB_l90_90489

def setA := {x : ℝ | |x| < 1}
def setB := {x : ℝ | x^2 - 2 * x ≤ 0}

theorem intersection_setA_setB :
  {x : ℝ | 0 ≤ x ∧ x < 1} = setA ∩ setB :=
by
  sorry

end intersection_setA_setB_l90_90489


namespace toll_for_18_wheel_truck_l90_90013

theorem toll_for_18_wheel_truck : 
  let x := 5 
  let w := 15 
  let y := 2 
  let T := 3.50 + 0.50 * (x - 2) + 0.10 * w + y 
  T = 8.50 := 
by 
  -- let x := 5 
  -- let w := 15 
  -- let y := 2 
  -- let T := 3.50 + 0.50 * (x - 2) + 0.10 * w + y 
  -- Note: the let statements within the brackets above
  sorry

end toll_for_18_wheel_truck_l90_90013


namespace smaller_than_negative_one_l90_90311

theorem smaller_than_negative_one :
  ∃ x ∈ ({0, -1/2, 1, -2} : Set ℝ), x < -1 ∧ x = -2 :=
by
  -- the proof part is skipped
  sorry

end smaller_than_negative_one_l90_90311


namespace complement_union_and_complement_intersect_l90_90661

-- Definitions of sets according to the problem conditions
def U : Set ℝ := Set.univ
def A : Set ℝ := { x | 3 ≤ x ∧ x < 7 }
def B : Set ℝ := { x | 2 < x ∧ x < 10 }

-- The correct answers derived in the solution
def complement_union_A_B : Set ℝ := { x | x ≤ 2 ∨ 10 ≤ x }
def complement_A_intersect_B : Set ℝ := { x | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10) }

-- Statement of the mathematically equivalent proof problem
theorem complement_union_and_complement_intersect:
  (Set.compl (A ∪ B) = complement_union_A_B) ∧ 
  ((Set.compl A) ∩ B = complement_A_intersect_B) :=
  by 
    sorry

end complement_union_and_complement_intersect_l90_90661


namespace average_student_headcount_is_10983_l90_90962

def student_headcount_fall_03_04 := 11500
def student_headcount_spring_03_04 := 10500
def student_headcount_fall_04_05 := 11600
def student_headcount_spring_04_05 := 10700
def student_headcount_fall_05_06 := 11300
def student_headcount_spring_05_06 := 10300 -- Assume value

def total_student_headcount :=
  student_headcount_fall_03_04 + student_headcount_spring_03_04 +
  student_headcount_fall_04_05 + student_headcount_spring_04_05 +
  student_headcount_fall_05_06 + student_headcount_spring_05_06

def average_student_headcount := total_student_headcount / 6

theorem average_student_headcount_is_10983 :
  average_student_headcount = 10983 :=
by -- Will prove the theorem
sorry

end average_student_headcount_is_10983_l90_90962


namespace point_in_fourth_quadrant_l90_90181

-- Definitions of the quadrants as provided in the conditions
def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0
def third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0
def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

-- Given point
def point : ℝ × ℝ := (1, -2)

-- Theorem statement
theorem point_in_fourth_quadrant : fourth_quadrant point.fst point.snd :=
sorry

end point_in_fourth_quadrant_l90_90181


namespace georgina_parrot_days_l90_90220

theorem georgina_parrot_days
  (total_phrases : ℕ)
  (phrases_per_week : ℕ)
  (initial_phrases : ℕ)
  (phrases_now : total_phrases = 17)
  (teaching_rate : phrases_per_week = 2)
  (initial_known : initial_phrases = 3) :
  (49 : ℕ) = (((17 - 3) / 2) * 7) :=
by
  -- proof will be here
  sorry

end georgina_parrot_days_l90_90220


namespace jennifer_money_left_l90_90689

variable (initial_amount : ℝ) (spent_sandwich_rate : ℝ) (spent_museum_rate : ℝ) (spent_book_rate : ℝ)

def money_left := initial_amount - (spent_sandwich_rate * initial_amount + spent_museum_rate * initial_amount + spent_book_rate * initial_amount)

theorem jennifer_money_left (h_initial : initial_amount = 150)
  (h_sandwich_rate : spent_sandwich_rate = 1/5)
  (h_museum_rate : spent_museum_rate = 1/6)
  (h_book_rate : spent_book_rate = 1/2) :
  money_left initial_amount spent_sandwich_rate spent_museum_rate spent_book_rate = 20 :=
by
  sorry

end jennifer_money_left_l90_90689


namespace number_of_bottle_caps_l90_90913

-- Condition: 7 bottle caps weigh exactly one ounce
def weight_of_7_caps : ℕ := 1 -- ounce

-- Condition: Josh's collection weighs 18 pounds, and 1 pound = 16 ounces
def weight_of_collection_pounds : ℕ := 18 -- pounds
def weight_of_pound_in_ounces : ℕ := 16 -- ounces per pound

-- Condition: Question translated to proof statement
theorem number_of_bottle_caps :
  (weight_of_collection_pounds * weight_of_pound_in_ounces * 7 = 2016) :=
by
  sorry

end number_of_bottle_caps_l90_90913


namespace intersection_of_A_and_B_union_of_A_and_B_complement_of_A_with_respect_to_U_l90_90684

open Set

universe u

def U : Set ℝ := univ

def A : Set ℝ := { x | x ≥ 1 ∨ x ≤ -3 }

def B : Set ℝ := { x | -4 < x ∧ x < 0 }

theorem intersection_of_A_and_B :
  A ∩ B = { x | -4 < x ∧ x ≤ -3 } :=
by sorry

theorem union_of_A_and_B :
  A ∪ B = { x | x < 0 ∨ x ≥ 1 } :=
by sorry

theorem complement_of_A_with_respect_to_U :
  U \ A = { x | -3 < x ∧ x < 1 } :=
by sorry

end intersection_of_A_and_B_union_of_A_and_B_complement_of_A_with_respect_to_U_l90_90684


namespace sufficient_conditions_for_x_sq_lt_one_l90_90840

theorem sufficient_conditions_for_x_sq_lt_one
  (x : ℝ) :
  (0 < x ∧ x < 1) ∨ (-1 < x ∧ x < 0) ∨ (-1 < x ∧ x < 1) → x^2 < 1 :=
by
  sorry

end sufficient_conditions_for_x_sq_lt_one_l90_90840


namespace problem1_problem2_l90_90487

variable (α : ℝ)

-- First problem statement
theorem problem1 (h : Real.tan α = 2) : 
  (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.sin α + 3 * Real.cos α) = 6 / 13 :=
by 
  sorry

-- Second problem statement
theorem problem2 (h : Real.tan α = 2) :
  3 * (Real.sin α)^2 + 3 * Real.sin α * Real.cos α - 2 * (Real.cos α)^2 = 16 / 5 :=
by 
  sorry

end problem1_problem2_l90_90487


namespace heaps_never_empty_l90_90389

-- Define initial conditions
def initial_heaps := (1993, 199, 19)

-- Allowed operations
def add_stones (a b c : ℕ) : (ℕ × ℕ × ℕ) :=
if a = 1993 then (a + b + c, b, c)
else if b = 199 then (a, b + a + c, c)
else (a, b, c + a + b)

def remove_stones (a b c : ℕ) : (ℕ × ℕ × ℕ) :=
if a = 1993 then (a - (b + c), b, c)
else if b = 199 then (a, b - (a + c), c)
else (a, b, c - (a + b))

-- The proof statement
theorem heaps_never_empty :
  ∀ a b c : ℕ, a = 1993 ∧ b = 199 ∧ c = 19 ∧ (∀ n : ℕ, (a + b + c) % 2 = 1) ∧ (a - (b + c) % 2 = 1) → ¬(a = 0 ∨ b = 0 ∨ c = 0) := 
by {
  sorry
}

end heaps_never_empty_l90_90389


namespace snack_cost_is_five_l90_90353

-- Define the cost of one ticket
def ticket_cost : ℕ := 18

-- Define the total number of people
def total_people : ℕ := 4

-- Define the total cost for tickets and snacks
def total_cost : ℕ := 92

-- Define the unknown cost of one set of snacks
def snack_cost := 92 - 4 * 18

-- Statement asserting that the cost of one set of snacks is $5
theorem snack_cost_is_five : snack_cost = 5 := by
  sorry

end snack_cost_is_five_l90_90353


namespace find_side_b_l90_90320

variables {a b c : ℝ} {B : ℝ}

theorem find_side_b 
  (area_triangle : (1 / 2) * a * c * (Real.sin B) = Real.sqrt 3) 
  (B_is_60_degrees : B = Real.pi / 3) 
  (relation_ac : a^2 + c^2 = 3 * a * c) : 
  b = 2 * Real.sqrt 2 := 
by 
  sorry

end find_side_b_l90_90320


namespace find_ABC_l90_90454

theorem find_ABC (A B C : ℕ) (h1 : A ≠ B) (h2 : B ≠ C) (h3 : A ≠ C) 
  (hA : A < 5) (hB : B < 5) (hC : C < 5) (h_nonzeroA : A ≠ 0) (h_nonzeroB : B ≠ 0) (h_nonzeroC : C ≠ 0)
  (h4 : B + C = 5) (h5 : A + 1 = C) (h6 : A + B = C) : A = 3 ∧ B = 1 ∧ C = 4 := 
by
  sorry

end find_ABC_l90_90454


namespace sum_last_two_digits_l90_90603

theorem sum_last_two_digits (a b : ℕ) (h₁ : a = 7) (h₂ : b = 13) : 
  (a^25 + b^25) % 100 = 0 :=
by
  sorry

end sum_last_two_digits_l90_90603


namespace total_coronavirus_cases_l90_90527

theorem total_coronavirus_cases (ny_cases ca_cases tx_cases : ℕ)
    (h_ny : ny_cases = 2000)
    (h_ca : ca_cases = ny_cases / 2)
    (h_tx : ca_cases = tx_cases + 400) :
    ny_cases + ca_cases + tx_cases = 3600 := by
  sorry

end total_coronavirus_cases_l90_90527


namespace abs_sum_sequence_l90_90075

def S (n : ℕ) : ℤ := n^2 - 4 * n

def a (n : ℕ) : ℤ := S n - S (n-1)

theorem abs_sum_sequence (h : ∀ n, S n = n^2 - 4 * n) :
  (|a 1| + |a 2| + |a 3| + |a 4| + |a 5| + |a 6| + |a 7| + |a 8| + |a 9| + |a 10|) = 68 :=
by
  sorry

end abs_sum_sequence_l90_90075


namespace remainder_of_x50_div_x_minus_1_cubed_l90_90816

theorem remainder_of_x50_div_x_minus_1_cubed :
  (x : ℝ) → (x ^ 50) % ((x - 1) ^ 3) = 1225 * x ^ 2 - 2400 * x + 1176 := 
by
  sorry

end remainder_of_x50_div_x_minus_1_cubed_l90_90816


namespace sum_is_24_l90_90060

-- Define the conditions
def A := 3
def B := 7 * A

-- Define the theorem to prove that the sum is 24
theorem sum_is_24 : A + B = 24 :=
by
  -- Adding sorry here since we're not required to provide the proof
  sorry

end sum_is_24_l90_90060


namespace conference_center_capacity_l90_90001

theorem conference_center_capacity (n_rooms : ℕ) (fraction_full : ℚ) (current_people : ℕ) (full_capacity : ℕ) (people_per_room : ℕ) 
  (h1 : n_rooms = 6) (h2 : fraction_full = 2/3) (h3 : current_people = 320) (h4 : current_people = fraction_full * full_capacity) 
  (h5 : people_per_room = full_capacity / n_rooms) : people_per_room = 80 :=
by
  -- The proof will go here.
  sorry

end conference_center_capacity_l90_90001


namespace tan_value_l90_90231

theorem tan_value (α : ℝ) (h1 : α ∈ (Set.Ioo (π/2) π)) (h2 : Real.sin α = 4/5) : Real.tan α = -4/3 :=
sorry

end tan_value_l90_90231


namespace auntie_em_can_park_l90_90048

noncomputable def parking_probability : ℚ :=
  let total_ways := (Nat.choose 20 5)
  let unfavorables := (Nat.choose 14 5)
  let probability_cannot_park := (unfavorables : ℚ) / total_ways
  1 - probability_cannot_park

theorem auntie_em_can_park :
  parking_probability = 964 / 1107 :=
by
  sorry

end auntie_em_can_park_l90_90048


namespace balance_scale_measurements_l90_90410

theorem balance_scale_measurements {a b c : ℕ}
    (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 11) :
    ∀ w : ℕ, 1 ≤ w ∧ w ≤ 11 → ∃ (x y z : ℤ), w = abs (x * a + y * b + z * c) :=
sorry

end balance_scale_measurements_l90_90410


namespace december_revenue_times_average_l90_90216

variable (D : ℝ) -- December's revenue
variable (N : ℝ) -- November's revenue
variable (J : ℝ) -- January's revenue

-- Conditions
def revenue_in_november : N = (2/5) * D := by sorry
def revenue_in_january : J = (1/2) * N := by sorry

-- Statement to be proved
theorem december_revenue_times_average :
  D = (10/3) * ((N + J) / 2) :=
by sorry

end december_revenue_times_average_l90_90216


namespace line_through_chord_with_midpoint_l90_90624

theorem line_through_chord_with_midpoint (x y : ℝ) :
  (∃ x1 y1 x2 y2 : ℝ,
    (x = x1 ∧ y = y1 ∨ x = x2 ∧ y = y2) ∧
    x = -1 ∧ y = 1 ∧
    x1^2 / 4 + y1^2 / 3 = 1 ∧
    x2^2 / 4 + y2^2 / 3 = 1) →
  3 * x - 4 * y + 7 = 0 :=
by
  sorry

end line_through_chord_with_midpoint_l90_90624


namespace log_sum_geometric_sequence_l90_90933

open Real

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ (n : ℕ), a n ≠ 0 ∧ a (n + 1) / a n = a 1 / a 0

theorem log_sum_geometric_sequence 
  (a : ℕ → ℝ) 
  (h_pos : ∀ n, 0 < a n) 
  (h_geo : geometric_sequence a) 
  (h_eq : a 10 * a 11 + a 9 * a 12 = 2 * exp 5) : 
  log (a 1) + log (a 2) + log (a 3) + log (a 4) + log (a 5) + 
  log (a 6) + log (a 7) + log (a 8) + log (a 9) + log (a 10) + 
  log (a 11) + log (a 12) + log (a 13) + log (a 14) + log (a 15) + 
  log (a 16) + log (a 17) + log (a 18) + log (a 19) + log (a 20) = 50 :=
sorry

end log_sum_geometric_sequence_l90_90933


namespace max_ab_under_constraint_l90_90885

theorem max_ab_under_constraint (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 3 * a + 2 * b = 1) : 
  ab ≤ (1 / 24) ∧ (ab = 1 / 24 ↔ a = 1 / 6 ∧ b = 1 / 4) :=
sorry

end max_ab_under_constraint_l90_90885


namespace shifted_parabola_passes_through_neg1_1_l90_90806

def original_parabola (x : ℝ) : ℝ := -(x + 1)^2 + 4

def shifted_parabola (x : ℝ) : ℝ := -(x - 1)^2 + 2

theorem shifted_parabola_passes_through_neg1_1 :
  shifted_parabola (-1) = 1 :=
by 
  -- Proof goes here
  sorry

end shifted_parabola_passes_through_neg1_1_l90_90806


namespace find_a6_l90_90331

-- Define the geometric sequence and the given terms
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

variables {a : ℕ → ℝ} (r : ℝ)

-- Given conditions
axiom a_2 : a 2 = 2
axiom a_10 : a 10 = 8
axiom geo_seq : geometric_sequence a

-- Statement to prove
theorem find_a6 : a 6 = 4 :=
sorry

end find_a6_l90_90331


namespace number_of_integer_values_x_floor_2_sqrt_x_eq_12_l90_90606

theorem number_of_integer_values_x_floor_2_sqrt_x_eq_12 :
  ∃! n : ℕ, n = 7 ∧ (∀ x : ℕ, (⌊2 * Real.sqrt x⌋ = 12 ↔ 36 ≤ x ∧ x < 43)) :=
by 
  sorry

end number_of_integer_values_x_floor_2_sqrt_x_eq_12_l90_90606


namespace first_year_with_sum_15_l90_90672

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 1000) + (n % 1000 / 100) + (n % 100 / 10) + (n % 10)

theorem first_year_with_sum_15 : ∃ y > 2100, sum_of_digits y = 15 :=
  sorry

end first_year_with_sum_15_l90_90672


namespace value_of_expression_l90_90751

theorem value_of_expression (a b c d : ℕ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) : 
  a * b / (c * d) = 180 :=
by
  sorry

end value_of_expression_l90_90751


namespace expand_expression_l90_90469

theorem expand_expression (x : ℝ) : 
  (x - 3) * (x + 3) * (x^2 + 5) = x^4 - 4 * x^2 - 45 := 
by
  sorry

end expand_expression_l90_90469


namespace chosen_number_l90_90109

theorem chosen_number (x : ℤ) (h : 2 * x - 138 = 110) : x = 124 :=
sorry

end chosen_number_l90_90109


namespace smallest_y_for_square_l90_90634

theorem smallest_y_for_square (y M : ℕ) (h1 : 2310 * y = M^2) (h2 : 2310 = 2 * 3 * 5 * 7 * 11) : y = 2310 :=
by sorry

end smallest_y_for_square_l90_90634


namespace area_of_rectangular_field_l90_90598

theorem area_of_rectangular_field (W D : ℝ) (hW : W = 15) (hD : D = 17) :
  ∃ L : ℝ, (W * L = 120) ∧ D^2 = L^2 + W^2 :=
by 
  use 8
  sorry

end area_of_rectangular_field_l90_90598


namespace no_positive_integer_solutions_l90_90620

theorem no_positive_integer_solutions (x y z : ℕ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) :
  x^3 + 2 * y^3 ≠ 4 * z^3 :=
by
  sorry

end no_positive_integer_solutions_l90_90620


namespace min_value_of_expression_l90_90134

noncomputable def min_expression_value (y : ℝ) (hy : y > 2) : ℝ :=
  (y^2 + y + 1) / Real.sqrt (y - 2)

theorem min_value_of_expression (y : ℝ) (hy : y > 2) :
  min_expression_value y hy = 3 * Real.sqrt 35 :=
sorry

end min_value_of_expression_l90_90134


namespace modulus_problem_l90_90775

theorem modulus_problem : (13 ^ 13 + 13) % 14 = 12 :=
by
  sorry

end modulus_problem_l90_90775


namespace projection_multiplier_l90_90778

noncomputable def a : ℝ × ℝ := (3, 6)
noncomputable def b : ℝ × ℝ := (-1, 0)

theorem projection_multiplier :
  let dot_product := a.1 * b.1 + a.2 * b.2
  let b_norm_sq := b.1 * b.1 + b.2 * b.2
  let proj := (dot_product / b_norm_sq) * 2
  (proj * b.1, proj * b.2) = (6, 0) :=
by 
  sorry

end projection_multiplier_l90_90778


namespace correct_division_result_l90_90108

theorem correct_division_result (x : ℝ) 
  (h : (x - 14) / 5 = 11) : (x - 5) / 7 = 64 / 7 :=
by
  sorry

end correct_division_result_l90_90108


namespace water_loss_per_jump_l90_90137

def pool_capacity : ℕ := 2000 -- in liters
def jump_limit : ℕ := 1000
def clean_threshold : ℝ := 0.80

theorem water_loss_per_jump :
  (pool_capacity * (1 - clean_threshold)) * 1000 / jump_limit = 400 :=
by
  -- We prove that the water lost per jump in mL is 400
  sorry

end water_loss_per_jump_l90_90137


namespace inequality_x2_gt_y2_plus_6_l90_90863

theorem inequality_x2_gt_y2_plus_6 (x y : ℝ) (h1 : x > y) (h2 : y > 3 / (x - y)) : x^2 > y^2 + 6 :=
sorry

end inequality_x2_gt_y2_plus_6_l90_90863


namespace min_frac_sum_l90_90811

noncomputable def min_value (x y : ℝ) : ℝ :=
  if (x + y = 1 ∧ x > 0 ∧ y > 0) then 1/x + 4/y else 0

theorem min_frac_sum (x y : ℝ) (h₁ : x + y = 1) (h₂: x > 0) (h₃: y > 0) : 
  min_value x y = 9 :=
sorry

end min_frac_sum_l90_90811


namespace ratio_expression_x_2y_l90_90585

theorem ratio_expression_x_2y :
  ∀ (x y : ℝ), x / (2 * y) = 27 → (7 * x + 6 * y) / (x - 2 * y) = 96 / 13 :=
by
  intros x y h
  sorry

end ratio_expression_x_2y_l90_90585


namespace find_M_plus_N_l90_90625

theorem find_M_plus_N (M N : ℕ) 
  (h1 : 5 / 7 = M / 63) 
  (h2 : 5 / 7 = 70 / N) : 
  M + N = 143 :=
by
  sorry

end find_M_plus_N_l90_90625


namespace distance_covered_at_40kmph_l90_90850

def total_distance : ℝ := 250
def speed_40 : ℝ := 40
def speed_60 : ℝ := 60
def total_time : ℝ := 5.2

theorem distance_covered_at_40kmph :
  ∃ (x : ℝ), (x / speed_40 + (total_distance - x) / speed_60 = total_time) ∧ x = 124 :=
  sorry

end distance_covered_at_40kmph_l90_90850


namespace hyperbola_equation_l90_90482

theorem hyperbola_equation {x y : ℝ} (h1 : x ^ 2 / 2 - y ^ 2 = 1) 
  (h2 : x = -2) (h3 : y = 2) : y ^ 2 / 2 - x ^ 2 / 4 = 1 :=
by sorry

end hyperbola_equation_l90_90482


namespace tan_double_angle_sum_l90_90732

theorem tan_double_angle_sum (α : ℝ) (h : Real.tan α = 3 / 2) :
  Real.tan (2 * α + Real.pi / 4) = -7 / 17 := 
sorry

end tan_double_angle_sum_l90_90732


namespace average_runs_per_game_l90_90977

-- Define the number of games
def games : ℕ := 6

-- Define the list of runs scored in each game
def runs : List ℕ := [1, 4, 4, 5, 5, 5]

-- The sum of the runs
def total_runs : ℕ := List.sum runs

-- The average runs per game
def avg_runs : ℚ := total_runs / games

-- The theorem to prove
theorem average_runs_per_game : avg_runs = 4 := by sorry

end average_runs_per_game_l90_90977


namespace lcm_48_180_value_l90_90484

def lcm_48_180 : ℕ := Nat.lcm 48 180

theorem lcm_48_180_value : lcm_48_180 = 720 :=
by
-- Proof not required, insert sorry
sorry

end lcm_48_180_value_l90_90484


namespace original_price_of_sarees_l90_90270

theorem original_price_of_sarees (P : ℝ) (h1 : 0.95 * 0.80 * P = 133) : P = 175 :=
sorry

end original_price_of_sarees_l90_90270


namespace isosceles_triangle_perimeter_l90_90021

-- Define the sides of the isosceles triangle
def side1 : ℝ := 4
def side2 : ℝ := 8

-- Hypothesis: The perimeter of an isosceles triangle with the given sides
-- Given condition
def is_isosceles_triangle (a b c : ℝ) : Prop := (a = b ∨ b = c ∨ c = a) ∧ a + b > c ∧ b + c > a ∧ c + a > b

-- Theorem statement
theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = side1 ∨ a = side2) (h2 : b = side1 ∨ b = side2) :
  ∃ p : ℝ, is_isosceles_triangle a b side2 ∧ p = a + b + side2 → p = 20 :=
sorry

end isosceles_triangle_perimeter_l90_90021


namespace distance_between_stations_l90_90852

-- Definitions based on conditions in step a):
def speed_train1 : ℝ := 20  -- speed of the first train in km/hr
def speed_train2 : ℝ := 25  -- speed of the second train in km/hr
def extra_distance : ℝ := 55  -- one train has traveled 55 km more

-- Definition of the proof problem
theorem distance_between_stations :
  ∃ D1 D2 T : ℝ, D1 = speed_train1 * T ∧ D2 = speed_train2 * T ∧ D2 = D1 + extra_distance ∧ D1 + D2 = 495 :=
by
  sorry

end distance_between_stations_l90_90852


namespace negation_of_proposition_l90_90555

theorem negation_of_proposition (p : Prop) : 
  (∀ x : ℝ, x ≥ 0 → x^2 - x + 1 ≥ 0) ↔ ¬(∃ x : ℝ, x ≥ 0 ∧ x^2 - x + 1 < 0) :=
by sorry

end negation_of_proposition_l90_90555


namespace range_of_a_l90_90587

noncomputable def f (a x : ℝ) : ℝ :=
  Real.exp x + x^2 + (3 * a + 2) * x

theorem range_of_a (a : ℝ) :
  (∃ x ∈ Set.Ioo (-1 : ℝ) 0, ∀ y ∈ Set.Ioo (-1 : ℝ) 0, f a x ≤ f a y) →
  a ∈ Set.Ioo (-1 : ℝ) (-1 / (3 * Real.exp 1)) :=
sorry

end range_of_a_l90_90587


namespace weight_box_plate_cups_l90_90776

theorem weight_box_plate_cups (b p c : ℝ) 
  (h₁ : b + 20 * p + 30 * c = 4.8)
  (h₂ : b + 40 * p + 50 * c = 8.4) : 
  b + 10 * p + 20 * c = 3 :=
sorry

end weight_box_plate_cups_l90_90776


namespace bathroom_visits_time_l90_90144

variable (t_8 : ℕ) (n8 : ℕ) (n6 : ℕ)

theorem bathroom_visits_time (h1 : t_8 = 20) (h2 : n8 = 8) (h3 : n6 = 6) :
  (t_8 / n8) * n6 = 15 := by
  sorry

end bathroom_visits_time_l90_90144


namespace number_of_oxygen_atoms_l90_90319

/-- Given a compound has 1 H, 1 Cl, and a certain number of O atoms and the molecular weight of the compound is 68 g/mol,
    prove that the number of O atoms is 2. -/
theorem number_of_oxygen_atoms (atomic_weight_H: ℝ) (atomic_weight_Cl: ℝ) (atomic_weight_O: ℝ) (molecular_weight: ℝ) (n : ℕ):
    atomic_weight_H = 1.0 →
    atomic_weight_Cl = 35.5 →
    atomic_weight_O = 16.0 →
    molecular_weight = 68.0 →
    molecular_weight = atomic_weight_H + atomic_weight_Cl + n * atomic_weight_O →
    n = 2 :=
by
  sorry

end number_of_oxygen_atoms_l90_90319


namespace arithmetic_sequence_twelfth_term_l90_90637

theorem arithmetic_sequence_twelfth_term :
  let a1 := (1 : ℚ) / 2;
  let a2 := (5 : ℚ) / 6;
  let d := a2 - a1;
  (a1 + 11 * d) = (25 : ℚ) / 6 :=
by
  let a1 := (1 : ℚ) / 2;
  let a2 := (5 : ℚ) / 6;
  let d := a2 - a1;
  exact sorry

end arithmetic_sequence_twelfth_term_l90_90637


namespace nonagon_diagonals_count_eq_27_l90_90919

theorem nonagon_diagonals_count_eq_27 :
  let vertices := 9
  let connections_per_vertex := vertices - 3
  let total_raw_count_diagonals := vertices * connections_per_vertex
  let distinct_diagonals := total_raw_count_diagonals / 2
  distinct_diagonals = 27 :=
by
  let vertices := 9
  let connections_per_vertex := vertices - 3
  let total_raw_count_diagonals := vertices * connections_per_vertex
  let distinct_diagonals := total_raw_count_diagonals / 2
  have : distinct_diagonals = 27 := sorry
  exact this

end nonagon_diagonals_count_eq_27_l90_90919


namespace thomas_blocks_total_l90_90898

theorem thomas_blocks_total 
  (first_stack : ℕ)
  (second_stack : ℕ)
  (third_stack : ℕ)
  (fourth_stack : ℕ)
  (fifth_stack : ℕ) 
  (h1 : first_stack = 7)
  (h2 : second_stack = first_stack + 3)
  (h3 : third_stack = second_stack - 6)
  (h4 : fourth_stack = third_stack + 10)
  (h5 : fifth_stack = 2 * second_stack) :
  first_stack + second_stack + third_stack + fourth_stack + fifth_stack = 55 :=
by
  sorry

end thomas_blocks_total_l90_90898


namespace problem_l90_90976

def isRightTriangle (a b c : ℝ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

def CannotFormRightTriangle (lst : List ℝ) : Prop :=
  ¬isRightTriangle lst.head! lst.tail.head! lst.tail.tail.head!

theorem problem :
  (¬isRightTriangle 3 4 5 ∧ ¬isRightTriangle 5 12 13 ∧ ¬isRightTriangle 2 3 (Real.sqrt 13)) ∧ CannotFormRightTriangle [4, 6, 8] :=
by
  sorry

end problem_l90_90976


namespace comparison_a_b_c_l90_90596

theorem comparison_a_b_c :
  let a := (1 / 2) ^ (1 / 3)
  let b := (1 / 3) ^ (1 / 2)
  let c := Real.log (3 / Real.pi)
  c < b ∧ b < a :=
by
  sorry

end comparison_a_b_c_l90_90596


namespace email_sequence_correct_l90_90183

theorem email_sequence_correct :
    ∀ (a b c d e f : Prop),
    (a → (e → (b → (c → (d → f))))) :=
by 
  sorry

end email_sequence_correct_l90_90183


namespace hyperbola_equation_l90_90106

-- Definitions for a given hyperbola
variables {a b : ℝ}
axiom a_pos : a > 0
axiom b_pos : b > 0

-- Definitions for the asymptote condition
axiom point_on_asymptote : (4 : ℝ) = (b / a) * 3

-- Definitions for the focal distance condition
axiom point_circle_intersect : (3 : ℝ)^2 + 4^2 = (a^2 + b^2)

-- The goal is to prove the hyperbola's specific equation
theorem hyperbola_equation : 
  (a^2 = 9 ∧ b^2 = 16) →
  (∃ a b : ℝ, (4 : ℝ)^2 + 3^2 = (a^2 + b^2) ∧ 
               (4 : ℝ) = (b / a) * 3 ∧ 
               ((a^2 = 9) ∧ (b^2 = 16)) ∧ (a > 0) ∧ (b > 0)) :=
sorry

end hyperbola_equation_l90_90106


namespace inequality_problem_l90_90781

theorem inequality_problem (a b c : ℝ) (h : a * b * c = 1) :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (c + a))) + (1 / (c^3 * (a + b))) ≥ (3 / 2) :=
by 
  sorry

end inequality_problem_l90_90781


namespace find_r_s_l90_90052

noncomputable def parabola_line_intersection (x y m : ℝ) : Prop :=
  y = x^2 + 5*x ∧ y + 6 = m*(x - 10)

theorem find_r_s (r s m : ℝ) (Q : ℝ × ℝ)
  (hq : Q = (10, -6))
  (h_parabola : ∀ x, ∃ y, y = x^2 + 5*x)
  (h_line : ∀ x, ∃ y, y + 6 = m*(x - 10)) :
  parabola_line_intersection x y m → (r < m ∧ m < s) ∧ (r + s = 50) :=
sorry

end find_r_s_l90_90052


namespace fare_range_l90_90530

noncomputable def fare (x : ℝ) : ℝ :=
  if x <= 3 then 8 else 8 + 1.5 * (x - 3)

theorem fare_range (x : ℝ) (hx : fare x = 16) : 8 ≤ x ∧ x < 9 :=
by
  sorry

end fare_range_l90_90530


namespace area_of_region_bounded_by_sec_and_csc_l90_90086

theorem area_of_region_bounded_by_sec_and_csc (x y : ℝ) :
  (∃ (x y : ℝ), x = 1 ∧ y = 1 ∧ 0 ≤ x ∧ 0 ≤ y) → 
  (∃ (area : ℝ), area = 1) :=
by 
  sorry

end area_of_region_bounded_by_sec_and_csc_l90_90086


namespace Einstein_sold_25_cans_of_soda_l90_90273

def sell_snacks_proof : Prop :=
  let pizza_price := 12
  let fries_price := 0.30
  let soda_price := 2
  let goal := 500
  let pizza_boxes := 15
  let fries_packs := 40
  let still_needed := 258
  let earned_from_pizza := pizza_boxes * pizza_price
  let earned_from_fries := fries_packs * fries_price
  let total_earned := earned_from_pizza + earned_from_fries
  let total_have := goal - still_needed
  let earned_from_soda := total_have - total_earned
  let cans_of_soda_sold := earned_from_soda / soda_price
  cans_of_soda_sold = 25

theorem Einstein_sold_25_cans_of_soda : sell_snacks_proof := by
  sorry

end Einstein_sold_25_cans_of_soda_l90_90273


namespace find_m_l90_90125

theorem find_m (x y m : ℤ) (h1 : x = 3) (h2 : y = 1) (h3 : x - m * y = 1) : m = 2 :=
by
  -- Proof goes here
  sorry

end find_m_l90_90125


namespace cos_angle_identity_l90_90869

theorem cos_angle_identity (α : ℝ) (h : Real.cos (π / 2 - α) = Real.sqrt 2 / 3) :
  Real.cos (π - 2 * α) = - (5 / 9) := by
sorry

end cos_angle_identity_l90_90869


namespace sequence_general_term_l90_90846

theorem sequence_general_term (n : ℕ) : 
  (2 * n - 1) / (2 ^ n) = a_n := 
sorry

end sequence_general_term_l90_90846


namespace find_value_of_N_l90_90845

theorem find_value_of_N (x N : ℝ) (hx : x = 1.5) (h : (3 + 2 * x)^5 = (N + 3 * x)^4) : N = 1.5 := by
  -- Here we will assume that the proof is filled in and correct.
  sorry

end find_value_of_N_l90_90845


namespace inequality_problem_l90_90808

noncomputable def nonneg_real := {x : ℝ // 0 ≤ x}

theorem inequality_problem (x y z : nonneg_real) (h : x.val * y.val + y.val * z.val + z.val * x.val = 1) :
  1 / (x.val + y.val) + 1 / (y.val + z.val) + 1 / (z.val + x.val) ≥ 5 / 2 :=
sorry

end inequality_problem_l90_90808


namespace value_of_p_l90_90829

theorem value_of_p (p q : ℤ) (h1 : p + q = 2010)
  (h2 : ∃ (x1 x2 : ℕ), x1 > 0 ∧ x2 > 0 ∧ 67 * x1^2 + p * x1 + q = 0 ∧ 67 * x2^2 + p * x2 + q = 0) : p = -2278 :=
by
  sorry

end value_of_p_l90_90829


namespace smallest_multiple_of_18_and_40_l90_90224

-- Define the conditions
def multiple_of_18 (n : ℕ) : Prop := n % 18 = 0
def multiple_of_40 (n : ℕ) : Prop := n % 40 = 0

-- Prove that the smallest number that meets the conditions is 360
theorem smallest_multiple_of_18_and_40 : ∃ n : ℕ, multiple_of_18 n ∧ multiple_of_40 n ∧ ∀ m : ℕ, (multiple_of_18 m ∧ multiple_of_40 m) → n ≤ m :=
  by
    let n := 360
    -- We have to prove that 360 is the smallest number that is a multiple of both 18 and 40
    sorry

end smallest_multiple_of_18_and_40_l90_90224


namespace percentage_of_total_spent_on_other_items_l90_90682

-- Definitions for the given problem conditions

def total_amount (a : ℝ) := a
def spent_on_clothing (a : ℝ) := 0.50 * a
def spent_on_food (a : ℝ) := 0.10 * a
def spent_on_other_items (a x_clothing x_food : ℝ) := a - x_clothing - x_food
def tax_on_clothing (x_clothing : ℝ) := 0.04 * x_clothing
def tax_on_food := 0
def tax_on_other_items (x_other_items : ℝ) := 0.08 * x_other_items
def total_tax (a : ℝ) := 0.052 * a

-- The theorem we need to prove
theorem percentage_of_total_spent_on_other_items (a x_clothing x_food x_other_items : ℝ)
    (h1 : x_clothing = spent_on_clothing a)
    (h2 : x_food = spent_on_food a)
    (h3 : x_other_items = spent_on_other_items a x_clothing x_food)
    (h4 : tax_on_clothing x_clothing + tax_on_food + tax_on_other_items x_other_items = total_tax a) :
    0.40 * a = x_other_items :=
sorry

end percentage_of_total_spent_on_other_items_l90_90682


namespace problem_statement_l90_90363

theorem problem_statement (x y : ℝ) (hx : 0 < x) (hy : 0 < y)
  (h : x - Real.sqrt x ≤ y - 1 / 4 ∧ y - 1 / 4 ≤ x + Real.sqrt x) :
  y - Real.sqrt y ≤ x - 1 / 4 ∧ x - 1 / 4 ≤ y + Real.sqrt y :=
sorry

end problem_statement_l90_90363


namespace lesser_solution_is_minus_15_l90_90079

noncomputable def lesser_solution : ℤ := -15

theorem lesser_solution_is_minus_15 :
  ∃ x y : ℤ, x^2 + 10 * x - 75 = 0 ∧ y^2 + 10 * y - 75 = 0 ∧ x < y ∧ x = lesser_solution :=
by 
  sorry

end lesser_solution_is_minus_15_l90_90079


namespace arithmetic_sequence_x_value_l90_90397

theorem arithmetic_sequence_x_value (x : ℝ) (a2 a1 d : ℝ)
  (h1 : a1 = 1 / 3)
  (h2 : a2 = x - 2)
  (h3 : d = 4 * x + 1 - a2)
  (h2_eq_d_a1 : a2 - a1 = d) : x = - (8 / 3) :=
by
  -- Proof yet to be completed
  sorry

end arithmetic_sequence_x_value_l90_90397


namespace line_through_point_intersecting_circle_eq_l90_90189

theorem line_through_point_intersecting_circle_eq :
  ∃ k l : ℝ, (x + 2*y + 9 = 0 ∨ 2*x - y + 3 = 0) ∧ 
    ∀ L : ℝ × ℝ,  
      (L = (-3, -3)) ∧ (x^2 + y^2 + 4*y - 21 = 0) → 
      (L = (-3,-3) → (x + 2*y + 9 = 0 ∨ 2*x - y + 3 = 0)) := 
sorry

end line_through_point_intersecting_circle_eq_l90_90189


namespace number_verification_l90_90832

def is_product_of_four_consecutive (n : ℕ) : Prop :=
  ∃ a : ℕ, n = a * (a + 1) * (a + 2) * (a + 3)

theorem number_verification (h1 : 1680 % 3 = 0) (h2 : ∃ a : ℕ, 1680 = a * (a + 1) * (a + 2) * (a + 3)) : 
  is_product_of_four_consecutive 1680 :=
by
  sorry

end number_verification_l90_90832


namespace selling_price_correct_l90_90698

noncomputable def discount1 (price : ℝ) : ℝ := price * 0.85
noncomputable def discount2 (price : ℝ) : ℝ := price * 0.90
noncomputable def discount3 (price : ℝ) : ℝ := price * 0.95

noncomputable def final_price (initial_price : ℝ) : ℝ :=
  discount3 (discount2 (discount1 initial_price))

theorem selling_price_correct : final_price 3600 = 2616.30 := by
  sorry

end selling_price_correct_l90_90698


namespace tim_movie_marathon_duration_l90_90446

-- Define the durations of each movie
def first_movie_duration : ℕ := 2

def second_movie_duration : ℕ := 
  first_movie_duration + (first_movie_duration / 2)

def combined_first_two_movies_duration : ℕ :=
  first_movie_duration + second_movie_duration

def last_movie_duration : ℕ := 
  combined_first_two_movies_duration - 1

-- Define the total movie marathon duration
def total_movie_marathon_duration : ℕ := 
  first_movie_duration + second_movie_duration + last_movie_duration

-- Problem statement to be proved
theorem tim_movie_marathon_duration : total_movie_marathon_duration = 9 := by
  sorry

end tim_movie_marathon_duration_l90_90446


namespace remainder_of_hx10_divided_by_hx_is_6_l90_90154

noncomputable def h (x : ℝ) : ℝ := x^5 + x^4 + x^3 + x^2 + x + 1

theorem remainder_of_hx10_divided_by_hx_is_6 : 
  let q := h (x ^ 10);
  q % h (x) = 6 := by
  sorry

end remainder_of_hx10_divided_by_hx_is_6_l90_90154


namespace find_stock_rate_l90_90292

theorem find_stock_rate (annual_income : ℝ) (investment_amount : ℝ) (R : ℝ) 
  (h1 : annual_income = 2000) (h2 : investment_amount = 6800) : 
  R = 2000 / 6800 :=
by
  sorry

end find_stock_rate_l90_90292


namespace range_of_x_when_a_eq_1_p_and_q_range_of_a_when_not_p_sufficient_for_not_q_l90_90217

-- Define the propositions
def p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := -x^2 + 5 * x - 6 ≥ 0

-- Question 1: Prove that for a = 1 and p ∧ q is true, the range of x is [2, 3)
theorem range_of_x_when_a_eq_1_p_and_q : 
  ∀ x : ℝ, p 1 x ∧ q x → 2 ≤ x ∧ x < 3 := 
by sorry

-- Question 2: Prove that if ¬p is a sufficient but not necessary condition for ¬q, 
-- then the range of a is (1, 2)
theorem range_of_a_when_not_p_sufficient_for_not_q :
  ∀ a : ℝ, (∀ x : ℝ, ¬p a x → ¬q x) ∧ (∃ x : ℝ, ¬(¬p a x → ¬q x)) → 1 < a ∧ a < 2 := 
by sorry

end range_of_x_when_a_eq_1_p_and_q_range_of_a_when_not_p_sufficient_for_not_q_l90_90217


namespace determine_percentage_of_yellow_in_darker_green_paint_l90_90354

noncomputable def percentage_of_yellow_in_darker_green_paint : Real :=
  let volume_light_green := 5
  let volume_darker_green := 1.66666666667
  let percentage_light_green := 0.20
  let final_percentage := 0.25
  let total_volume := volume_light_green + volume_darker_green
  let total_yellow_required := final_percentage * total_volume
  let yellow_in_light_green := percentage_light_green * volume_light_green
  (total_yellow_required - yellow_in_light_green) / volume_darker_green

theorem determine_percentage_of_yellow_in_darker_green_paint :
  percentage_of_yellow_in_darker_green_paint = 0.4 := by
  sorry

end determine_percentage_of_yellow_in_darker_green_paint_l90_90354


namespace carla_cream_volume_l90_90341

-- Definitions of the given conditions and problem
def watermelon_puree_volume : ℕ := 500
def servings_count : ℕ := 4
def volume_per_serving : ℕ := 150
def total_smoothies_volume := servings_count * volume_per_serving
def cream_volume := total_smoothies_volume - watermelon_puree_volume

-- Statement of the proposition we want to prove
theorem carla_cream_volume : cream_volume = 100 := by
  sorry

end carla_cream_volume_l90_90341


namespace complex_number_addition_identity_l90_90674

-- Definitions of the conditions
def imaginary_unit (i : ℂ) := i^2 = -1

def complex_fraction_decomposition (a b : ℝ) (i : ℂ) := 
  (1 + i) / (1 - i) = a + b * i

-- The statement of the problem
theorem complex_number_addition_identity :
  ∃ (a b : ℝ) (i : ℂ), imaginary_unit i ∧ complex_fraction_decomposition a b i ∧ (a + b = 1) :=
sorry

end complex_number_addition_identity_l90_90674


namespace hexagon_interior_angles_l90_90534

theorem hexagon_interior_angles
  (A B C D E F : ℝ)
  (hA : A = 90)
  (hB : B = 120)
  (hCD : C = D)
  (hE : E = 2 * C + 20)
  (hF : F = 60)
  (hsum : A + B + C + D + E + F = 720) :
  D = 107.5 := 
by
  -- formal proof required here
  sorry

end hexagon_interior_angles_l90_90534


namespace exists_common_ratio_of_geometric_progression_l90_90951

theorem exists_common_ratio_of_geometric_progression (a r : ℝ) (h_pos : 0 < r) 
(h_eq: a = a * r + a * r^2 + a * r^3) : ∃ r : ℝ, r^3 + r^2 + r - 1 = 0 :=
by sorry

end exists_common_ratio_of_geometric_progression_l90_90951


namespace age_solution_l90_90584

theorem age_solution (M S : ℕ) (h1 : M = S + 16) (h2 : M + 2 = 2 * (S + 2)) : S = 14 :=
by sorry

end age_solution_l90_90584


namespace cages_used_l90_90866

-- Define the initial conditions
def total_puppies : ℕ := 18
def puppies_sold : ℕ := 3
def puppies_per_cage : ℕ := 5

-- State the theorem to prove the number of cages used
theorem cages_used : (total_puppies - puppies_sold) / puppies_per_cage = 3 := by
  sorry

end cages_used_l90_90866


namespace trapezoid_area_l90_90943

theorem trapezoid_area (outer_triangle_area inner_triangle_area : ℝ) (congruent_trapezoids : ℕ) 
  (h1 : outer_triangle_area = 36) (h2 : inner_triangle_area = 4) (h3 : congruent_trapezoids = 3) :
  (outer_triangle_area - inner_triangle_area) / congruent_trapezoids = 32 / 3 :=
by sorry

end trapezoid_area_l90_90943


namespace solve_absolute_value_eq_l90_90610

theorem solve_absolute_value_eq (x : ℝ) : (|x - 3| = 5 - x) → x = 4 :=
by
  sorry

end solve_absolute_value_eq_l90_90610


namespace suresh_investment_correct_l90_90688

noncomputable def suresh_investment
  (ramesh_investment : ℝ)
  (total_profit : ℝ)
  (ramesh_profit_share : ℝ)
  : ℝ := sorry

theorem suresh_investment_correct
  (ramesh_investment : ℝ := 40000)
  (total_profit : ℝ := 19000)
  (ramesh_profit_share : ℝ := 11875)
  : suresh_investment ramesh_investment total_profit ramesh_profit_share = 24000 := sorry

end suresh_investment_correct_l90_90688


namespace gcd_50421_35343_l90_90685

theorem gcd_50421_35343 : Int.gcd 50421 35343 = 23 := by
  sorry

end gcd_50421_35343_l90_90685


namespace solution_set_of_inequality_l90_90342

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x + 6

theorem solution_set_of_inequality (m : ℝ) : 
  f (m + 3) > f (2 * m) ↔ (-1/3 : ℝ) < m ∧ m < 3 :=
by 
  sorry

end solution_set_of_inequality_l90_90342


namespace divides_quartic_sum_l90_90884

theorem divides_quartic_sum (a b c n : ℤ) (h1 : n ∣ (a + b + c)) (h2 : n ∣ (a^2 + b^2 + c^2)) : n ∣ (a^4 + b^4 + c^4) := 
sorry

end divides_quartic_sum_l90_90884


namespace number_of_shirts_in_first_batch_minimum_selling_price_per_shirt_l90_90586

-- Define the conditions
def first_batch_cost : ℝ := 13200
def second_batch_cost : ℝ := 28800
def unit_price_difference : ℝ := 10
def discount_rate : ℝ := 0.8
def profit_margin : ℝ := 1.25
def last_batch_count : ℕ := 50

-- Define the theorem for the first part
theorem number_of_shirts_in_first_batch (x : ℕ) (h₁ : first_batch_cost / x + unit_price_difference = second_batch_cost / (2 * x)) : x = 120 :=
sorry

-- Define the theorem for the second part
theorem minimum_selling_price_per_shirt (x : ℕ) (y : ℝ)
  (h₁ : first_batch_cost / x + unit_price_difference = second_batch_cost / (2 * x))
  (h₂ : x = 120)
  (h₃ : (3 * x - last_batch_count) * y + last_batch_count * discount_rate * y ≥ (first_batch_cost + second_batch_cost) * profit_margin) : y ≥ 150 :=
sorry

end number_of_shirts_in_first_batch_minimum_selling_price_per_shirt_l90_90586


namespace tan_of_angle_in_third_quadrant_l90_90308

open Real

theorem tan_of_angle_in_third_quadrant 
  (α : ℝ) 
  (h1 : sin (π + α) = 3/5) 
  (h2 : π < α ∧ α < 3 * π / 2) : 
  tan α = 3 / 4 :=
by
  sorry

end tan_of_angle_in_third_quadrant_l90_90308


namespace line_slope_through_origin_intersects_parabola_l90_90780

theorem line_slope_through_origin_intersects_parabola (k : ℝ) :
  (∃ x1 x2 : ℝ, 5 * (kx1) = 2 * x1 ^ 2 - 9 * x1 + 10 ∧ 5 * (kx2) = 2 * x2 ^ 2 - 9 * x2 + 10 ∧ x1 + x2 = 77) → k = 29 :=
by
  intro h
  sorry

end line_slope_through_origin_intersects_parabola_l90_90780


namespace food_for_elephants_l90_90714

theorem food_for_elephants (t : ℕ) : 
  (∀ (food_per_day : ℕ), (12 * food_per_day) * 1 = (1000 * food_per_day) * 600) →
  (∀ (food_per_day : ℕ), (t * food_per_day) * 1 = (100 * food_per_day) * d) →
  d = 500 * t :=
by
  sorry

end food_for_elephants_l90_90714


namespace original_gain_percentage_is_5_l90_90725

def costPrice : ℝ := 200
def newCostPrice : ℝ := costPrice * 0.95
def desiredProfitRatio : ℝ := 0.10
def newSellingPrice : ℝ := newCostPrice * (1 + desiredProfitRatio)
def originalSellingPrice : ℝ := newSellingPrice + 1

theorem original_gain_percentage_is_5 :
  ((originalSellingPrice - costPrice) / costPrice) * 100 = 5 :=
by 
  sorry

end original_gain_percentage_is_5_l90_90725


namespace probability_of_type_I_error_l90_90968

theorem probability_of_type_I_error 
  (K_squared : ℝ)
  (alpha : ℝ)
  (critical_val : ℝ)
  (h1 : K_squared = 4.05)
  (h2 : alpha = 0.05)
  (h3 : critical_val = 3.841)
  (h4 : 4.05 > 3.841) :
  alpha = 0.05 := 
sorry

end probability_of_type_I_error_l90_90968


namespace min_lines_to_separate_points_l90_90151

theorem min_lines_to_separate_points (m n : ℕ) (h_m : m = 8) (h_n : n = 8) : 
  (m - 1) + (n - 1) = 14 := by
  sorry

end min_lines_to_separate_points_l90_90151


namespace smallest_number_of_sparrows_in_each_flock_l90_90563

theorem smallest_number_of_sparrows_in_each_flock (P : ℕ) (H : 14 * P ≥ 182) : 
  ∃ S : ℕ, S = 14 ∧ S ∣ 182 ∧ (∃ P : ℕ, S ∣ (14 * P)) := 
by 
  sorry

end smallest_number_of_sparrows_in_each_flock_l90_90563


namespace danny_reaches_steve_house_in_31_minutes_l90_90582

theorem danny_reaches_steve_house_in_31_minutes:
  ∃ (t : ℝ), 2 * t - t = 15.5 * 2 ∧ t = 31 := sorry

end danny_reaches_steve_house_in_31_minutes_l90_90582


namespace solid_has_identical_views_is_sphere_or_cube_l90_90686

-- Define the conditions for orthographic projections being identical
def identical_views_in_orthographic_projections (solid : Type) : Prop :=
  sorry -- Assume the logic for checking identical orthographic projections is defined

-- Define the types for sphere and cube
structure Sphere : Type := 
  (radius : ℝ)

structure Cube : Type := 
  (side_length : ℝ)

-- The main statement to prove
theorem solid_has_identical_views_is_sphere_or_cube (solid : Type) 
  (h : identical_views_in_orthographic_projections solid) : 
  solid = Sphere ∨ solid = Cube :=
by 
  sorry -- The detailed proof is omitted

end solid_has_identical_views_is_sphere_or_cube_l90_90686


namespace common_difference_of_arithmetic_sequence_l90_90784

theorem common_difference_of_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) 
  (h1 : a 0 = 2) 
  (h2 : ∀ n, a (n+1) = a n + d)
  (h3 : a 9 = 20): 
  d = 2 := 
by
  sorry

end common_difference_of_arithmetic_sequence_l90_90784


namespace clarinet_fraction_l90_90147

theorem clarinet_fraction 
  (total_flutes total_clarinets total_trumpets total_pianists total_band: ℕ)
  (percent_flutes : ℚ) (fraction_trumpets fraction_pianists : ℚ)
  (total_persons_in_band: ℚ)
  (flutes_got_in : total_flutes = 20)
  (clarinets_got_in : total_clarinets = 30)
  (trumpets_got_in : total_trumpets = 60)
  (pianists_got_in : total_pianists = 20)
  (band_got_in : total_band = 53)
  (percent_flutes_got_in: percent_flutes = 0.8)
  (fraction_trumpets_got_in: fraction_trumpets = 1/3)
  (fraction_pianists_got_in: fraction_pianists = 1/10)
  (persons_in_band: total_persons_in_band = 53) :
  (15 / 30 : ℚ) = (1 / 2 : ℚ) := 
by
  sorry

end clarinet_fraction_l90_90147


namespace total_liquid_poured_out_l90_90854

noncomputable def capacity1 := 2
noncomputable def capacity2 := 6
noncomputable def percentAlcohol1 := 0.3
noncomputable def percentAlcohol2 := 0.4
noncomputable def totalCapacity := 10
noncomputable def finalConcentration := 0.3

theorem total_liquid_poured_out :
  capacity1 + capacity2 = 8 :=
by
  sorry

end total_liquid_poured_out_l90_90854


namespace is_isosceles_right_triangle_l90_90930

theorem is_isosceles_right_triangle 
  {a b c : ℝ}
  (h : |c^2 - a^2 - b^2| + (a - b)^2 = 0) : 
  a = b ∧ c^2 = a^2 + b^2 :=
sorry

end is_isosceles_right_triangle_l90_90930


namespace find_n_l90_90130

theorem find_n (x y n : ℝ) (h1 : 2 * x - 5 * y = 3 * n + 7) (h2 : x - 3 * y = 4) 
  (h3 : x = y):
  n = -1 / 3 := 
by 
  sorry

end find_n_l90_90130


namespace P_work_time_l90_90263

theorem P_work_time (T : ℝ) (hT : T > 0) : 
  (1 / T + 1 / 6 = 1 / 2.4) → T = 4 :=
by
  intros h
  sorry

end P_work_time_l90_90263


namespace complement_of_M_is_correct_l90_90865

def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x^2 - x > 0}
def complement_M : Set ℝ := {x | 0 ≤ x ∧ x ≤ 1}

theorem complement_of_M_is_correct :
  (U \ M) = complement_M :=
by
  sorry

end complement_of_M_is_correct_l90_90865


namespace possible_values_of_a_l90_90765

theorem possible_values_of_a :
  ∃ (a b c : ℤ), ∀ x : ℤ, (x - a) * (x - 5) + 1 = (x + b) * (x + c) → a = 3 ∨ a = 7 :=
by
  sorry

end possible_values_of_a_l90_90765


namespace calories_difference_l90_90804

def calories_burnt (hours : ℕ) : ℕ := 30 * hours

theorem calories_difference :
  calories_burnt 5 - calories_burnt 2 = 90 :=
by
  sorry

end calories_difference_l90_90804


namespace slope_of_line_l90_90934

theorem slope_of_line (x₁ y₁ x₂ y₂ : ℝ) (h₁ : 2/x₁ + 3/y₁ = 0) (h₂ : 2/x₂ + 3/y₂ = 0) (h_diff : x₁ ≠ x₂) : 
  (y₂ - y₁) / (x₂ - x₁) = -3/2 :=
sorry

end slope_of_line_l90_90934


namespace simplify_expression_l90_90747

theorem simplify_expression : 4 * (12 / 9) * (36 / -45) = -12 / 5 :=
by
  sorry

end simplify_expression_l90_90747


namespace sum_of_numbers_l90_90770

-- Definitions that come directly from the conditions
def product_condition (A B : ℕ) : Prop := A * B = 9375
def quotient_condition (A B : ℕ) : Prop := A / B = 15

-- Theorem that proves the sum of A and B is 400, based on the given conditions
theorem sum_of_numbers (A B : ℕ) (h1 : product_condition A B) (h2 : quotient_condition A B) : A + B = 400 :=
sorry

end sum_of_numbers_l90_90770


namespace cell_phone_plan_cost_l90_90143

theorem cell_phone_plan_cost:
  let base_cost : ℕ := 25
  let text_cost : ℕ := 8
  let extra_min_cost : ℕ := 12
  let texts_sent : ℕ := 150
  let hours_talked : ℕ := 27
  let extra_minutes := (hours_talked - 25) * 60
  let total_cost := (base_cost * 100) + (texts_sent * text_cost) + (extra_minutes * extra_min_cost)
  (total_cost = 5140) :=
by
  sorry

end cell_phone_plan_cost_l90_90143


namespace solve_for_x_and_y_l90_90156

theorem solve_for_x_and_y (x y : ℝ) (h1 : x + y = 15) (h2 : x - y = 5) : x = 10 ∧ y = 5 :=
by
  sorry

end solve_for_x_and_y_l90_90156


namespace mrs_hilt_initial_money_l90_90552

def initial_amount (pencil_cost candy_cost left_money : ℕ) := 
  pencil_cost + candy_cost + left_money

theorem mrs_hilt_initial_money :
  initial_amount 20 5 18 = 43 :=
by
  -- initial_amount 20 5 18 
  -- = 20 + 5 + 18
  -- = 25 + 18 
  -- = 43
  sorry

end mrs_hilt_initial_money_l90_90552


namespace value_of_hash_l90_90545

def hash (a b c d : ℝ) : ℝ := b^2 - 4 * a * c * d

theorem value_of_hash : hash 2 3 2 1 = -7 := by
  sorry

end value_of_hash_l90_90545


namespace fraction_to_decimal_l90_90629

theorem fraction_to_decimal :
  (7 : ℚ) / 16 = 0.4375 :=
by sorry

end fraction_to_decimal_l90_90629


namespace sum_eq_product_l90_90208

theorem sum_eq_product (a b c : ℝ) (h1 : 1 + b * c ≠ 0) (h2 : 1 + c * a ≠ 0) (h3 : 1 + a * b ≠ 0) :
  (b - c) / (1 + b * c) + (c - a) / (1 + c * a) + (a - b) / (1 + a * b) =
  ((b - c) * (c - a) * (a - b)) / ((1 + b * c) * (1 + c * a) * (1 + a * b)) :=
by
  sorry

end sum_eq_product_l90_90208


namespace theta_in_second_quadrant_l90_90769

theorem theta_in_second_quadrant (θ : ℝ) (h₁ : Real.sin θ > 0) (h₂ : Real.cos θ < 0) : 
  π / 2 < θ ∧ θ < π := 
sorry

end theta_in_second_quadrant_l90_90769


namespace liz_spent_total_l90_90268

-- Definitions:
def recipe_book_cost : ℕ := 6
def baking_dish_cost : ℕ := 2 * recipe_book_cost
def ingredient_cost : ℕ := 3
def number_of_ingredients : ℕ := 5
def apron_cost : ℕ := recipe_book_cost + 1

-- Total cost calculation:
def total_cost : ℕ :=
  recipe_book_cost + baking_dish_cost + (number_of_ingredients * ingredient_cost) + apron_cost

-- Theorem Statement:
theorem liz_spent_total : total_cost = 40 := by
  sorry

end liz_spent_total_l90_90268


namespace brenda_trays_l90_90097

-- Define main conditions
def cookies_per_tray : ℕ := 80
def cookies_per_box : ℕ := 60
def cost_per_box : ℕ := 350
def total_cost : ℕ := 1400  -- Using cents for calculation to avoid float numbers

-- State the problem
theorem brenda_trays :
  (total_cost / cost_per_box) * cookies_per_box / cookies_per_tray = 3 := 
by
  sorry

end brenda_trays_l90_90097


namespace range_of_a_l90_90034

noncomputable def f (x : ℝ) : ℝ := 2 * x + 1 / Real.exp x - Real.exp x

theorem range_of_a (a : ℝ) (h : f (a - 1) + f (2 * a ^ 2) ≤ 0) : 
  a ∈ Set.Iic (-1) ∪ Set.Ici (1 / 2) :=
sorry

end range_of_a_l90_90034


namespace right_square_pyramid_height_l90_90579

theorem right_square_pyramid_height :
  ∀ (h x : ℝ),
    let topBaseSide := 3
    let bottomBaseSide := 6
    let lateralArea := 4 * (1/2) * (topBaseSide + bottomBaseSide) * x
    let baseAreasSum := topBaseSide^2 + bottomBaseSide^2
    lateralArea = baseAreasSum →
    x = 5/2 →
    h = 2 :=
by
  intros h x topBaseSide bottomBaseSide lateralArea baseAreasSum lateralEq baseEq
  sorry

end right_square_pyramid_height_l90_90579


namespace sum_of_roots_l90_90558

open Polynomial

noncomputable def f (a b : ℝ) : Polynomial ℝ := Polynomial.C b + Polynomial.C a * X + X^2
noncomputable def g (c d : ℝ) : Polynomial ℝ := Polynomial.C d + Polynomial.C c * X + X^2

theorem sum_of_roots (a b c d : ℝ)
  (h1 : eval 1 (f a b) = eval 2 (g c d))
  (h2 : eval 1 (g c d) = eval 2 (f a b))
  (hf_roots : ∃ r1 r2 : ℝ, (f a b).roots = {r1, r2})
  (hg_roots : ∃ s1 s2 : ℝ, (g c d).roots = {s1, s2}) :
  (-(a + c) = 6) :=
sorry

end sum_of_roots_l90_90558


namespace initial_team_sizes_l90_90979

/-- 
On the first day of the sports competition, 1/6 of the boys' team and 1/7 of the girls' team 
did not meet the qualifying standards and were eliminated. During the rest of the competition, 
the same number of athletes from both teams were eliminated for not meeting the standards. 
By the end of the competition, a total of 48 boys and 50 girls did not meet the qualifying standards. 
Moreover, the number of girls who met the qualifying standards was twice the number of boys who did.
We are to prove the initial number of boys and girls in their respective teams.
-/

theorem initial_team_sizes (initial_boys initial_girls : ℕ) :
  (∃ (x : ℕ), 
    initial_boys = x + 48 ∧ 
    initial_girls = 2 * x + 50 ∧ 
    48 - (1 / 6 : ℚ) * (x + 48 : ℚ) = 50 - (1 / 7 : ℚ) * (2 * x + 50 : ℚ) ∧
    initial_girls - 2 * initial_boys = 98 - 2 * 72
  ) ↔ 
  initial_boys = 72 ∧ initial_girls = 98 := 
sorry

end initial_team_sizes_l90_90979


namespace find_sale_in_second_month_l90_90638

def sale_in_second_month (sale1 sale3 sale4 sale5 sale6 target_average : ℕ) (S : ℕ) : Prop :=
  sale1 + S + sale3 + sale4 + sale5 + sale6 = target_average * 6

theorem find_sale_in_second_month :
  sale_in_second_month 5420 6200 6350 6500 7070 6200 5660 :=
by
  sorry

end find_sale_in_second_month_l90_90638


namespace multiplications_in_three_hours_l90_90222

theorem multiplications_in_three_hours :
  let rate := 15000  -- multiplications per second
  let seconds_in_three_hours := 3 * 3600  -- seconds in three hours
  let total_multiplications := rate * seconds_in_three_hours
  total_multiplications = 162000000 :=
by
  let rate := 15000
  let seconds_in_three_hours := 3 * 3600
  let total_multiplications := rate * seconds_in_three_hours
  have h : total_multiplications = 162000000 := sorry
  exact h

end multiplications_in_three_hours_l90_90222


namespace right_rect_prism_volume_l90_90911

theorem right_rect_prism_volume (a b c : ℝ) 
  (h1 : a * b = 56) 
  (h2 : b * c = 63) 
  (h3 : a * c = 36) : 
  a * b * c = 504 := by
  sorry

end right_rect_prism_volume_l90_90911


namespace total_green_peaches_l90_90091

-- Define the known conditions
def baskets : ℕ := 7
def green_peaches_per_basket : ℕ := 2

-- State the problem and the proof goal
theorem total_green_peaches : baskets * green_peaches_per_basket = 14 := by
  -- Provide a proof here
  sorry

end total_green_peaches_l90_90091


namespace find_x_in_list_l90_90460

theorem find_x_in_list :
  ∃ x : ℕ, x > 0 ∧ x ≤ 120 ∧ (45 + 76 + 110 + x + x) / 5 = 2 * x ∧ x = 29 :=
by
  sorry

end find_x_in_list_l90_90460


namespace fraction_of_25_exact_value_l90_90813

-- Define the conditions
def eighty_percent_of_sixty : ℝ := 0.80 * 60
def smaller_by_twenty_eight (x : ℝ) : Prop := x * 25 = eighty_percent_of_sixty - 28

-- The proof problem
theorem fraction_of_25_exact_value (x : ℝ) : smaller_by_twenty_eight x → x = 4 / 5 := by
  intro h
  sorry

end fraction_of_25_exact_value_l90_90813


namespace rationalize_cube_root_sum_l90_90346

theorem rationalize_cube_root_sum :
  let a := (5 : ℝ)^(1/3)
  let b := (3 : ℝ)^(1/3)
  let numerator := a^2 + a * b + b^2
  let denom := a - b
  let fraction := 1 / denom * numerator
  let A := 25
  let B := 15
  let C := 9
  let D := 2
  A + B + C + D = 51 :=
by
  let a := (5 : ℝ)^(1/3)
  let b := (3 : ℝ)^(1/3)
  let numerator := a^2 + a * b + b^2
  let denom := a - b
  let fraction := 1 / denom * numerator
  let A := 25
  let B := 15
  let C := 9
  let D := 2
  have step1 : (a^3 = 5) := by sorry
  have step2 : (b^3 = 3) := by sorry
  have denom_eq : denom = 2 := by sorry
  have frac_simp : fraction = (A^(1/3) + B^(1/3) + C^(1/3)) / D := by sorry
  show A + B + C + D = 51
  sorry

end rationalize_cube_root_sum_l90_90346


namespace slope_of_CD_l90_90710

-- Given circle equations
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y - 15 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 16*x + 8*y + 48 = 0

-- Define the line whose slope needs to be found
def line (x y : ℝ) : Prop := 22*x - 12*y - 33 = 0

-- State the proof problem
theorem slope_of_CD : ∀ x y : ℝ, circle1 x y → circle2 x y → line x y ∧ (∃ m : ℝ, m = 11/6) :=
by sorry

end slope_of_CD_l90_90710


namespace largest_integer_solution_l90_90860

theorem largest_integer_solution (x : ℤ) (h : -x ≥ 2 * x + 3) : x ≤ -1 := sorry

end largest_integer_solution_l90_90860


namespace smallest_number_of_marbles_l90_90355

theorem smallest_number_of_marbles 
  (r w b bl n : ℕ) 
  (h : r + w + b + bl = n)
  (h1 : r * (r - 1) * (r - 2) * (r - 3) = 24 * w * b * (r * (r - 1) / 2))
  (h2 : r * (r - 1) * (r - 2) * (r - 3) = 24 * bl * b * (r * (r - 1) / 2))
  (h_no_neg : 4 ≤ r):
  n = 18 :=
sorry

end smallest_number_of_marbles_l90_90355


namespace age_problem_solution_l90_90783

namespace AgeProblem

variables (S M : ℕ) (k : ℕ)

-- Condition: The present age of the son is 22
def son_age (S : ℕ) := S = 22

-- Condition: The man is 24 years older than his son
def man_age (M S : ℕ) := M = S + 24

-- Condition: In two years, man's age will be a certain multiple of son's age
def age_multiple (M S k : ℕ) := M + 2 = k * (S + 2)

-- Question: The ratio of man's age to son's age in two years
def age_ratio (M S : ℕ) := (M + 2) / (S + 2)

theorem age_problem_solution (S M : ℕ) (k : ℕ) 
  (h1 : son_age S)
  (h2 : man_age M S)
  (h3 : age_multiple M S k)
  : age_ratio M S = 2 :=
by
  rw [son_age, man_age, age_multiple, age_ratio] at *
  sorry

end AgeProblem

end age_problem_solution_l90_90783


namespace angle_of_rotation_l90_90841

-- Definitions for the given conditions
def radius_large := 9 -- cm
def radius_medium := 3 -- cm
def radius_small := 1 -- cm
def speed := 1 -- cm/s

-- Definition of the angles calculations
noncomputable def rotations_per_revolution (R1 R2 : ℝ) : ℝ := R1 / R2
noncomputable def total_rotations (R1 R2 R3 : ℝ) : ℝ := 
  let rotations_medium := rotations_per_revolution R1 R2
  let net_rotations_medium := rotations_medium - 1
  net_rotations_medium * rotations_per_revolution R2 R3 + 1

-- Assertion to prove
theorem angle_of_rotation : 
  total_rotations radius_large radius_medium radius_small * 360 = 2520 :=
by 
  simp [total_rotations, rotations_per_revolution]
  exact sorry -- proof placeholder

end angle_of_rotation_l90_90841


namespace opposite_numbers_reciprocal_values_l90_90499

theorem opposite_numbers_reciprocal_values (a b m n : ℝ) (h₁ : a + b = 0) (h₂ : m * n = 1) : 5 * a + 5 * b - m * n = -1 :=
by sorry

end opposite_numbers_reciprocal_values_l90_90499


namespace no_positive_real_solutions_l90_90826

theorem no_positive_real_solutions 
  (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x^3 + y^3 + z^3 = x + y + z) (h2 : x^2 + y^2 + z^2 = x * y * z) :
  false :=
by sorry

end no_positive_real_solutions_l90_90826


namespace simplify_trig_expression_l90_90649

theorem simplify_trig_expression (α : ℝ) :
  (2 * Real.sin (Real.pi - α) + Real.sin (2 * α)) / (2 * Real.cos (α / 2) ^ 2) = 2 * Real.sin α :=
by
  sorry

end simplify_trig_expression_l90_90649


namespace tourism_revenue_scientific_notation_l90_90650

-- Define the conditions given in the problem.
def total_tourism_revenue := 12.41 * 10^9

-- Prove the scientific notation of the total tourism revenue.
theorem tourism_revenue_scientific_notation :
  total_tourism_revenue = 1.241 * 10^9 :=
sorry

end tourism_revenue_scientific_notation_l90_90650


namespace system_is_inconsistent_l90_90011

def system_of_equations (x1 x2 x3 : ℝ) : Prop :=
  (x1 + 4*x2 + 10*x3 = 1) ∧
  (0*x1 - 5*x2 - 13*x3 = -1.25) ∧
  (0*x1 + 0*x2 + 0*x3 = 1.25)

theorem system_is_inconsistent : 
  ∀ x1 x2 x3, ¬ system_of_equations x1 x2 x3 :=
by
  intro x1 x2 x3
  sorry

end system_is_inconsistent_l90_90011


namespace triangle_area_l90_90739

theorem triangle_area {a b m : ℝ} (h1 : a = 27) (h2 : b = 29) (h3 : m = 26) : 
  ∃ (area : ℝ), area = 270 :=
by
  sorry

end triangle_area_l90_90739


namespace harriet_forward_speed_proof_l90_90696

def harriet_forward_time : ℝ := 3 -- forward time in hours
def harriet_return_speed : ℝ := 150 -- return speed in km/h
def harriet_total_time : ℝ := 5 -- total trip time in hours

noncomputable def harriet_forward_speed : ℝ :=
  let distance := harriet_return_speed * (harriet_total_time - harriet_forward_time)
  distance / harriet_forward_time

theorem harriet_forward_speed_proof : harriet_forward_speed = 100 := by
  sorry

end harriet_forward_speed_proof_l90_90696


namespace canister_ratio_l90_90923

variable (C D : ℝ) -- Define capacities of canister C and canister D
variable (hC_half : 1/2 * C) -- Canister C is 1/2 full of water
variable (hD_third : 1/3 * D) -- Canister D is 1/3 full of water
variable (hD_after : 1/12 * D) -- Canister D contains 1/12 after pouring

theorem canister_ratio (h : 1/2 * C = 1/4 * D) : D / C = 2 :=
by
  sorry

end canister_ratio_l90_90923


namespace pears_sold_l90_90425

theorem pears_sold (m a total : ℕ) (h1 : a = 2 * m) (h2 : m = 120) (h3 : a = 240) : total = 360 :=
by
  sorry

end pears_sold_l90_90425


namespace sufficient_but_not_necessary_condition_l90_90948

theorem sufficient_but_not_necessary_condition (x y : ℝ) :
  (x > 1 ∧ y > 1) → (x + y > 2) ∧ ¬((x + y > 2) → (x > 1 ∧ y > 1)) := 
by
  sorry

end sufficient_but_not_necessary_condition_l90_90948


namespace sum_of_two_numbers_l90_90577

theorem sum_of_two_numbers :
  ∀ (A B : ℚ), (A - B = 8) → (1 / 4 * (A + B) = 6) → (A = 16) → (A + B = 24) :=
by
  intros A B h1 h2 h3
  sorry

end sum_of_two_numbers_l90_90577


namespace bob_pennies_l90_90883

theorem bob_pennies (a b : ℤ) (h1 : b + 1 = 4 * (a - 1)) (h2 : b - 1 = 3 * (a + 1)) : b = 31 :=
by
  sorry

end bob_pennies_l90_90883


namespace num_ballpoint_pens_l90_90243

-- Define the total number of school supplies
def total_school_supplies : ℕ := 60

-- Define the number of pencils
def num_pencils : ℕ := 5

-- Define the number of notebooks
def num_notebooks : ℕ := 10

-- Define the number of erasers
def num_erasers : ℕ := 32

-- Define the number of ballpoint pens and prove it equals 13
theorem num_ballpoint_pens : total_school_supplies - (num_pencils + num_notebooks + num_erasers) = 13 :=
by
sorry

end num_ballpoint_pens_l90_90243


namespace second_candidate_votes_l90_90900

theorem second_candidate_votes (total_votes : ℕ) (first_candidate_percentage : ℝ) (first_candidate_votes: ℕ)
    (h1 : total_votes = 2400)
    (h2 : first_candidate_percentage = 0.80)
    (h3 : first_candidate_votes = total_votes * first_candidate_percentage) :
    total_votes - first_candidate_votes = 480 := by
    sorry

end second_candidate_votes_l90_90900


namespace simplify_expression_l90_90859

theorem simplify_expression (a : ℝ) : a^2 * (-a)^4 = a^6 := by
  sorry

end simplify_expression_l90_90859


namespace Dave_won_tickets_l90_90369

theorem Dave_won_tickets :
  ∀ (tickets_toys tickets_clothes total_tickets : ℕ),
  (tickets_toys = 8) →
  (tickets_clothes = 18) →
  (tickets_clothes = tickets_toys + 10) →
  (total_tickets = tickets_toys + tickets_clothes) →
  total_tickets = 26 :=
by
  intros tickets_toys tickets_clothes total_tickets h1 h2 h3 h4
  have h5 : tickets_clothes = 8 + 10 := by sorry
  have h6 : tickets_clothes = 18 := by sorry
  have h7 : tickets_clothes = 18 := by sorry
  exact sorry

end Dave_won_tickets_l90_90369


namespace find_a_if_g_even_l90_90132

noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 2 then x - 1 else if -2 ≤ x ∧ x ≤ 0 then -1 else 0

noncomputable def g (x : ℝ) (a : ℝ) : ℝ := (f x) + a * x

theorem find_a_if_g_even (a : ℝ) : (∀ x : ℝ, f x + a * x = f (-x) + a * (-x)) → a = -1/2 :=
by
  intro h
  sorry

end find_a_if_g_even_l90_90132


namespace total_surface_area_space_l90_90621

theorem total_surface_area_space (h r1 : ℝ) (h_cond : h = 8) (r1_cond : r1 = 3) : 
  (2 * π * (r1 + 1) * h - 2 * π * r1 * h) = 16 * π := 
by
  sorry

end total_surface_area_space_l90_90621


namespace car_initial_speed_l90_90734

theorem car_initial_speed (s t : ℝ) (h₁ : t = 15 * s^2) (h₂ : t = 3) :
  s = (Real.sqrt 2) / 5 :=
by
  sorry

end car_initial_speed_l90_90734


namespace roots_of_eq_l90_90084

theorem roots_of_eq (x : ℝ) : (x - 1) * (x - 2) = 0 ↔ (x = 1 ∨ x = 2) := by
  sorry

end roots_of_eq_l90_90084


namespace black_balls_in_box_l90_90983

theorem black_balls_in_box (B : ℕ) (probability : ℚ) 
  (h1 : probability = 0.38095238095238093) 
  (h2 : B / (14 + B) = probability) : 
  B = 9 := by
  sorry

end black_balls_in_box_l90_90983


namespace square_ratio_short_to_long_side_l90_90760

theorem square_ratio_short_to_long_side (a b : ℝ) (h : a / b + 1 / 2 = b / (Real.sqrt (a^2 + b^2))) : (a / b)^2 = (3 - Real.sqrt 5) / 2 := by
  sorry

end square_ratio_short_to_long_side_l90_90760


namespace distance_to_mothers_house_l90_90058

theorem distance_to_mothers_house 
  (D : ℝ) 
  (h1 : (2 / 3) * D = 156.0) : 
  D = 234.0 := 
sorry

end distance_to_mothers_house_l90_90058


namespace geometric_sequence_sum_l90_90155

theorem geometric_sequence_sum (a : ℕ → ℤ) (r : ℤ) (h_geom : ∀ n, a (n + 1) = a n * r)
  (h1 : a 0 + a 1 + a 2 = 8)
  (h2 : a 3 + a 4 + a 5 = -4) :
  a 6 + a 7 + a 8 = 2 := 
sorry

end geometric_sequence_sum_l90_90155


namespace determine_counterfeit_coin_l90_90922

-- Definitions and conditions
def coin_weight (coin : ℕ) : ℕ :=
  match coin with
  | 1 => 1 -- 1-kopek coin weighs 1 gram
  | 2 => 2 -- 2-kopeks coin weighs 2 grams
  | 3 => 3 -- 3-kopeks coin weighs 3 grams
  | 5 => 5 -- 5-kopeks coin weighs 5 grams
  | _ => 0 -- Invalid coin denomination, should not happen

def is_counterfeit (coin : ℕ) (actual_weight : ℕ) : Prop :=
  coin_weight coin ≠ actual_weight

-- Statement of the problem to be proved
theorem determine_counterfeit_coin (coins : List (ℕ × ℕ)) :
   (∀ (coin: ℕ) (weight: ℕ) (h : (coin, weight) ∈ coins),
      coin_weight coin = weight ∨ is_counterfeit coin weight) →
   (∃ (counterfeit_coin: ℕ) (weight: ℕ),
      (counterfeit_coin, weight) ∈ coins ∧ is_counterfeit counterfeit_coin weight) :=
sorry

end determine_counterfeit_coin_l90_90922


namespace addition_of_two_odds_is_even_subtraction_of_two_odds_is_even_l90_90753

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1
def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

theorem addition_of_two_odds_is_even (a b : ℤ) (ha : is_odd a) (hb : is_odd b) : is_even (a + b) :=
sorry

theorem subtraction_of_two_odds_is_even (a b : ℤ) (ha : is_odd a) (hb : is_odd b) : is_even (a - b) :=
sorry

end addition_of_two_odds_is_even_subtraction_of_two_odds_is_even_l90_90753


namespace find_Y_l90_90701

theorem find_Y :
  ∃ Y : ℤ, (19 + Y / 151) * 151 = 2912 ∧ Y = 43 :=
by
  use 43
  sorry

end find_Y_l90_90701


namespace number_of_blocks_l90_90385

theorem number_of_blocks (total_amount : ℕ) (gift_worth : ℕ) (workers_per_block : ℕ) (h1 : total_amount = 4000) (h2 : gift_worth = 4) (h3 : workers_per_block = 100) :
  (total_amount / gift_worth) / workers_per_block = 10 :=
by
-- This part will be proven later, hence using sorry for now
sorry

end number_of_blocks_l90_90385


namespace marco_total_time_l90_90479

def marco_run_time (laps distance1 distance2 speed1 speed2 : ℕ ) : ℝ :=
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  laps * (time1 + time2)

theorem marco_total_time :
  marco_run_time 7 150 350 3 4 = 962.5 :=
by
  sorry

end marco_total_time_l90_90479


namespace find_other_subject_given_conditions_l90_90544

theorem find_other_subject_given_conditions :
  ∀ (P C M : ℕ),
  P = 65 →
  (P + C + M) / 3 = 85 →
  (P + M) / 2 = 90 →
  ∃ (S : ℕ), (P + S) / 2 = 70 ∧ S = C :=
by
  sorry

end find_other_subject_given_conditions_l90_90544


namespace total_weight_of_full_bucket_l90_90882

variable (a b x y : ℝ)

def bucket_weights :=
  (x + (1/3) * y = a) → (x + (3/4) * y = b) → (x + y = (16/5) * b - (11/5) * a)

theorem total_weight_of_full_bucket :
  bucket_weights a b x y :=
by
  intro h1 h2
  -- proof goes here, can be omitted as per instructions
  sorry

end total_weight_of_full_bucket_l90_90882


namespace two_pow_2014_mod_seven_l90_90508

theorem two_pow_2014_mod_seven : 
  ∃ r : ℕ, 2 ^ 2014 ≡ r [MOD 7] → r = 2 :=
sorry

end two_pow_2014_mod_seven_l90_90508


namespace ben_points_l90_90764

theorem ben_points (zach_points : ℝ) (total_points : ℝ) (ben_points : ℝ) 
  (h1 : zach_points = 42.0) 
  (h2 : total_points = 63) 
  (h3 : total_points = zach_points + ben_points) : 
  ben_points = 21 :=
by
  sorry

end ben_points_l90_90764


namespace range_of_x_l90_90868

theorem range_of_x (a x : ℝ) (h : 0 ≤ a ∧ a ≤ 4) : x^2 + a * x > 4 * x + a - 3 ↔ (x > 3 ∨ x < -1) := by
  sorry

end range_of_x_l90_90868


namespace not_in_range_g_zero_l90_90092

noncomputable def g (x: ℝ) : ℤ :=
  if x > -3 then ⌈2 / (x + 3)⌉
  else if x < -3 then ⌊2 / (x + 3)⌋
  else 0 -- g(x) is not defined at x = -3, this is a placeholder

theorem not_in_range_g_zero :
  ¬ (∃ x : ℝ, g x = 0) :=
sorry

end not_in_range_g_zero_l90_90092


namespace parallel_vectors_x_value_l90_90142

variable {x : ℝ}

theorem parallel_vectors_x_value (h : (1 / x) = (2 / -6)) : x = -3 := sorry

end parallel_vectors_x_value_l90_90142


namespace quadratic_equal_roots_l90_90300

theorem quadratic_equal_roots (k : ℝ) :
  (∀ x : ℝ, x^2 + k * x + 1 = 0 → x = -k / 2) ↔ (k = 2 ∨ k = -2) :=
by
  sorry

end quadratic_equal_roots_l90_90300


namespace divisor_is_679_l90_90604

noncomputable def x : ℕ := 8
noncomputable def y : ℕ := 9
noncomputable def z : ℝ := 549.7025036818851
noncomputable def p : ℕ := x^3
noncomputable def q : ℕ := y^3
noncomputable def r : ℕ := p * q

theorem divisor_is_679 (k : ℝ) (h : r / k = z) : k = 679 := by
  sorry

end divisor_is_679_l90_90604


namespace tan_sum_l90_90810

open Real

theorem tan_sum 
  (α β γ θ φ : ℝ)
  (h1 : tan θ = (sin α * cos γ - sin β * sin γ) / (cos α * cos γ - cos β * sin γ))
  (h2 : tan φ = (sin α * sin γ - sin β * cos γ) / (cos α * sin γ - cos β * cos γ)) : 
  tan (θ + φ) = tan (α + β) :=
by
  sorry

end tan_sum_l90_90810


namespace line_equation_through_point_line_equation_sum_of_intercepts_l90_90244

theorem line_equation_through_point (x y : ℝ) (h : y = 2 * x + 5)
  (hx : x = -2) (hy : y = 1) : 2 * x - y + 5 = 0 :=
by {
  sorry
}

theorem line_equation_sum_of_intercepts (x y : ℝ) (h : y = 2 * x + 6)
  (hx : x = -3) (hy : y = 3) : 2 * x - y + 6 = 0 :=
by {
  sorry
}

end line_equation_through_point_line_equation_sum_of_intercepts_l90_90244


namespace infinite_n_divisible_by_p_l90_90190

theorem infinite_n_divisible_by_p (p : ℕ) (hp : Nat.Prime p) : 
  ∃ᶠ n in Filter.atTop, p ∣ (2^n - n) :=
by
  sorry

end infinite_n_divisible_by_p_l90_90190


namespace average_people_per_hour_l90_90531

theorem average_people_per_hour (total_people : ℕ) (days : ℕ) (hours_per_day : ℕ) (total_hours : ℕ) (average_per_hour : ℕ) :
  total_people = 3000 ∧ days = 5 ∧ hours_per_day = 24 ∧ total_hours = days * hours_per_day ∧ average_per_hour = total_people / total_hours → 
  average_per_hour = 25 :=
by
  sorry

end average_people_per_hour_l90_90531


namespace smallest_nat_mod_5_6_7_l90_90807

theorem smallest_nat_mod_5_6_7 (n : ℕ) :
  n % 5 = 4 ∧ n % 6 = 5 ∧ n % 7 = 6 → n = 209 :=
sorry

end smallest_nat_mod_5_6_7_l90_90807


namespace men_in_second_group_l90_90793

theorem men_in_second_group (m w : ℝ) (x : ℝ) 
  (h1 : 3 * m + 8 * w = x * m + 2 * w) 
  (h2 : 2 * m + 2 * w = (3 / 7) * (3 * m + 8 * w)) : x = 6 :=
by
  sorry

end men_in_second_group_l90_90793


namespace no_integer_root_quadratic_trinomials_l90_90767

theorem no_integer_root_quadratic_trinomials :
  ¬ ∃ (a b c : ℤ),
    (∃ r1 r2 : ℤ, a * r1^2 + b * r1 + c = 0 ∧ a * r2^2 + b * r2 + c = 0 ∧ r1 ≠ r2) ∧
    (∃ s1 s2 : ℤ, (a + 1) * s1^2 + (b + 1) * s1 + (c + 1) = 0 ∧ (a + 1) * s2^2 + (b + 1) * s2 + (c + 1) = 0 ∧ s1 ≠ s2) :=
by
  sorry

end no_integer_root_quadratic_trinomials_l90_90767


namespace book_club_meeting_days_l90_90537

theorem book_club_meeting_days :
  Nat.lcm (Nat.lcm 5 6) (Nat.lcm 8 (Nat.lcm 9 10)) = 360 := 
by sorry

end book_club_meeting_days_l90_90537


namespace area_triangle_CIN_l90_90663

variables (A B C D M N I : Type*)

-- Definitions and assumptions
-- ABCD is a square
def is_square (ABCD : Type*) (side : ℝ) : Prop := sorry
-- M is the midpoint of AB
def midpoint_AB (M A B : Type*) : Prop := sorry
-- N is the midpoint of BC
def midpoint_BC (N B C : Type*) : Prop := sorry
-- Lines CM and DN intersect at I
def lines_intersect_at (C M D N I : Type*) : Prop := sorry

-- Goal
theorem area_triangle_CIN (ABCD : Type*) (side : ℝ) (M N C I : Type*) 
  (h1 : is_square ABCD side)
  (h2 : midpoint_AB M A B)
  (h3 : midpoint_BC N B C)
  (h4 : lines_intersect_at C M D N I) :
  sorry := sorry

end area_triangle_CIN_l90_90663


namespace average_temperature_Robertson_l90_90788

def temperatures : List ℝ := [18, 21, 19, 22, 20]

noncomputable def average (temps : List ℝ) : ℝ :=
  (temps.sum) / (temps.length)

theorem average_temperature_Robertson :
  average temperatures = 20.0 :=
by
  sorry

end average_temperature_Robertson_l90_90788


namespace cost_per_gift_l90_90643

theorem cost_per_gift (a b c : ℕ) (hc : c = 70) (ha : a = 3) (hb : b = 4) :
  c / (a + b) = 10 :=
by sorry

end cost_per_gift_l90_90643


namespace student_ticket_cost_l90_90915

theorem student_ticket_cost (cost_per_student_ticket : ℝ) :
  (12 * cost_per_student_ticket + 4 * 3 = 24) → cost_per_student_ticket = 1 :=
by
  intros h
  -- We should provide a complete proof here, but for illustration, we use sorry.
  sorry

end student_ticket_cost_l90_90915


namespace union_of_sets_l90_90797

open Set

variable (a b : ℕ)

noncomputable def M : Set ℕ := {3, 2 * a}
noncomputable def N : Set ℕ := {a, b}

theorem union_of_sets (h : M a ∩ N a b = {2}) : M a ∪ N a b = {1, 2, 3} :=
by
  -- skipped proof
  sorry

end union_of_sets_l90_90797


namespace softball_team_total_players_l90_90065

theorem softball_team_total_players 
  (M W : ℕ) 
  (h1 : W = M + 4)
  (h2 : (M : ℚ) / (W : ℚ) = 0.6666666666666666) :
  M + W = 20 :=
by sorry

end softball_team_total_players_l90_90065


namespace abs_sum_l90_90101

theorem abs_sum (a b c : ℚ) (h₁ : a = -1/4) (h₂ : b = -2) (h₃ : c = -11/4) :
  |a| + |b| - |c| = -1/2 :=
by {
  sorry
}

end abs_sum_l90_90101


namespace lana_eats_fewer_candies_l90_90944

-- Definitions based on conditions
def canEatNellie : ℕ := 12
def canEatJacob : ℕ := canEatNellie / 2
def candiesBeforeLanaCries : ℕ := 6 -- This is the derived answer for Lana
def initialCandies : ℕ := 30
def remainingCandies : ℕ := 3 * 3 -- After division, each gets 3 candies and they are 3 people

-- Statement to prove how many fewer candies Lana can eat compared to Jacob
theorem lana_eats_fewer_candies :
  canEatJacob = 6 → 
  (initialCandies - remainingCandies = 12 + canEatJacob + candiesBeforeLanaCries) →
  canEatJacob - candiesBeforeLanaCries = 3 :=
by
  intros hJacobEats hCandiesAte
  sorry

end lana_eats_fewer_candies_l90_90944


namespace M_inter_N_eq_interval_l90_90246

def M : Set ℝ := {x | x^2 - x ≤ 0}
def N : Set ℝ := {x | -1 < x ∧ x < 1}

theorem M_inter_N_eq_interval : (M ∩ N) = {x | 0 ≤ x ∧ x < 1} := 
  sorry

end M_inter_N_eq_interval_l90_90246


namespace m_range_and_simplification_l90_90160

theorem m_range_and_simplification (x y m : ℝ)
  (h1 : (3 * (x + 1) / 2) + y = 2)
  (h2 : 3 * x - m = 2 * y)
  (hx : x ≤ 1)
  (hy : y ≤ 1) :
  (-3 ≤ m) ∧ (m ≤ 5) ∧ (|x - 1| + |y - 1| + |m + 3| + |m - 5| - |x + y - 2| = 8) := 
by sorry

end m_range_and_simplification_l90_90160


namespace solve_n_l90_90282

/-
Define the condition for the problem.
Given condition: \(\frac{1}{n+1} + \frac{2}{n+1} + \frac{n}{n+1} = 4\)
-/

noncomputable def condition (n : ℚ) : Prop :=
  (1 / (n + 1) + 2 / (n + 1) + n / (n + 1)) = 4

/-
The theorem to prove: Value of \( n \) that satisfies the condition is \( n = -\frac{1}{3} \)
-/
theorem solve_n : ∃ n : ℚ, condition n ∧ n = -1 / 3 :=
by
  sorry

end solve_n_l90_90282


namespace sub_neg_four_l90_90073

theorem sub_neg_four : -3 - 1 = -4 :=
by
  sorry

end sub_neg_four_l90_90073


namespace smallest_value_n_l90_90927

def factorial_factors_of_five (n : ℕ) : ℕ :=
  n / 5 + n / 25 + n / 125 + n / 625 + n / 3125

theorem smallest_value_n
  (a b c m n : ℕ)
  (h1 : a + b + c = 2003)
  (h2 : a = 2 * b)
  (h3 : a.factorial * b.factorial * c.factorial = m * 10 ^ n)
  (h4 : ¬ (10 ∣ m)) :
  n = 400 :=
by
  sorry

end smallest_value_n_l90_90927


namespace points_satisfy_equation_l90_90102

theorem points_satisfy_equation (x y : ℝ) : 
  (2 * x^2 + 3 * x * y + y^2 + x = 1) ↔ (y = -x - 1) ∨ (y = -2 * x + 1) := by
  sorry

end points_satisfy_equation_l90_90102


namespace least_positive_integer_to_add_l90_90395

theorem least_positive_integer_to_add (n : ℕ) (h_start : n = 525) : ∃ k : ℕ, k > 0 ∧ (n + k) % 5 = 0 ∧ k = 4 :=
by {
  sorry
}

end least_positive_integer_to_add_l90_90395


namespace factorization_correct_l90_90356

theorem factorization_correct (m : ℤ) : m^2 - 1 = (m - 1) * (m + 1) :=
by {
  -- sorry, this is a place-holder for the proof.
  sorry
}

end factorization_correct_l90_90356


namespace shortest_chord_line_intersect_circle_l90_90553

-- Define the equation of the circle C
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 3 = 0

-- Define the fixed point
def fixed_point : ℝ × ℝ := (0, 1)

-- Define the center of the circle
def center : ℝ × ℝ := (1, 0)

-- Define the equation of the line l
def line_eq (x y : ℝ) : Prop := x - y + 1 = 0

-- The theorem that needs to be proven
theorem shortest_chord_line_intersect_circle :
  ∃ k : ℝ, ∀ x y : ℝ, (circle_eq x y ∧ y = k * x + 1) ↔ line_eq x y :=
by
  sorry

end shortest_chord_line_intersect_circle_l90_90553


namespace vanya_scores_not_100_l90_90795

-- Definitions for initial conditions
def score_r (M : ℕ) := M - 14
def score_p (M : ℕ) := M - 9
def score_m (M : ℕ) := M

-- Define the maximum score constraint
def max_score := 100

-- Main statement to be proved
theorem vanya_scores_not_100 (M : ℕ) 
  (hr : score_r M ≤ max_score) 
  (hp : score_p M ≤ max_score) 
  (hm : score_m M ≤ max_score) : 
  ¬(score_r M = max_score ∧ (score_p M = max_score ∨ score_m M = max_score)) ∧
  ¬(score_r M = max_score ∧ score_p M = max_score ∧ score_m M = max_score) :=
sorry

end vanya_scores_not_100_l90_90795


namespace digit_sum_of_product_l90_90789

def digits_after_multiplication (a b : ℕ) : ℕ :=
  let product := a * b
  let units_digit := product % 10
  let tens_digit := (product / 10) % 10
  tens_digit + units_digit

theorem digit_sum_of_product :
  digits_after_multiplication 707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707 909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909 = 9 :=
by 
  -- proof goes here
sorry

end digit_sum_of_product_l90_90789


namespace simplify_polynomials_l90_90206

theorem simplify_polynomials (x : ℝ) :
  (3 * x^2 + 8 * x - 5) - (2 * x^2 + 3 * x - 15) = x^2 + 5 * x + 10 :=
by 
  sorry

end simplify_polynomials_l90_90206


namespace average_of_new_set_l90_90186

theorem average_of_new_set (s : List ℝ) (h₁ : s.length = 10) (h₂ : (s.sum / 10) = 7) : 
  ((s.map (λ x => x * 12)).sum / 10) = 84 :=
by
  sorry

end average_of_new_set_l90_90186


namespace problem_statement_l90_90736

theorem problem_statement (x y z : ℝ) 
  (h1 : x + y + z = 6) 
  (h2 : x * y + y * z + z * x = 11) 
  (h3 : x * y * z = 6) : 
  x / (y * z) + y / (z * x) + z / (x * y) = 7 / 3 := 
sorry

end problem_statement_l90_90736


namespace max_group_size_l90_90966

theorem max_group_size 
  (students_class1 : ℕ) (students_class2 : ℕ) 
  (leftover_class1 : ℕ) (leftover_class2 : ℕ) 
  (h_class1 : students_class1 = 69) 
  (h_class2 : students_class2 = 86) 
  (h_leftover1 : leftover_class1 = 5) 
  (h_leftover2 : leftover_class2 = 6) : 
  Nat.gcd (students_class1 - leftover_class1) (students_class2 - leftover_class2) = 16 :=
by
  sorry

end max_group_size_l90_90966


namespace square_of_sum_possible_l90_90909

theorem square_of_sum_possible (a b c : ℝ) : 
  ∃ d : ℝ, d = (a + b + c)^2 :=
sorry

end square_of_sum_possible_l90_90909


namespace calories_in_250_grams_of_lemonade_l90_90215

theorem calories_in_250_grams_of_lemonade:
  ∀ (lemon_juice_grams sugar_grams water_grams total_grams: ℕ)
    (lemon_juice_cal_per_100 sugar_cal_per_100 total_cal: ℕ),
  lemon_juice_grams = 150 →
  sugar_grams = 150 →
  water_grams = 300 →
  total_grams = lemon_juice_grams + sugar_grams + water_grams →
  lemon_juice_cal_per_100 = 30 →
  sugar_cal_per_100 = 386 →
  total_cal = (lemon_juice_grams * lemon_juice_cal_per_100 / 100) + (sugar_grams * sugar_cal_per_100 / 100) →
  (250:ℕ) * total_cal / total_grams = 260 :=
by
  intros lemon_juice_grams sugar_grams water_grams total_grams lemon_juice_cal_per_100 sugar_cal_per_100 total_cal
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end calories_in_250_grams_of_lemonade_l90_90215


namespace cost_rose_bush_l90_90334

-- Define the constants
def total_roses := 6
def friend_roses := 2
def total_aloes := 2
def cost_aloe := 100
def total_spent_self := 500

-- Prove the cost of each rose bush
theorem cost_rose_bush : (total_spent_self - total_aloes * cost_aloe) / (total_roses - friend_roses) = 75 :=
by
  sorry

end cost_rose_bush_l90_90334


namespace ledi_age_10_in_years_l90_90964

-- Definitions of ages of Duoduo and Ledi
def duoduo_current_age : ℝ := 10
def years_ago : ℝ := 12.3
def sum_ages_years_ago : ℝ := 12

-- Function to calculate Ledi's current age
def ledi_current_age :=
  (sum_ages_years_ago + years_ago + years_ago) + (duoduo_current_age - years_ago)

-- Function to calculate years from now for Ledi to be 10 years old
def years_until_ledi_age_10 (ledi_age_now : ℝ) : ℝ :=
  10 - ledi_age_now

-- Main statement we need to prove
theorem ledi_age_10_in_years : years_until_ledi_age_10 ledi_current_age = 6.3 :=
by
  -- Proof goes here
  sorry

end ledi_age_10_in_years_l90_90964


namespace zoe_total_money_l90_90343

def numberOfPeople : ℕ := 6
def sodaCostPerBottle : ℝ := 0.5
def pizzaCostPerSlice : ℝ := 1.0

theorem zoe_total_money :
  numberOfPeople * sodaCostPerBottle + numberOfPeople * pizzaCostPerSlice = 9 := 
by
  sorry

end zoe_total_money_l90_90343


namespace infinite_grid_coloring_l90_90437

theorem infinite_grid_coloring (color : ℕ × ℕ → Fin 4)
  (h_coloring_condition : ∀ (i j : ℕ), color (i, j) ≠ color (i + 1, j) ∧
                                      color (i, j) ≠ color (i, j + 1) ∧
                                      color (i, j) ≠ color (i + 1, j + 1) ∧
                                      color (i + 1, j) ≠ color (i, j + 1)) :
  ∃ m : ℕ, ∃ a b : Fin 4, ∀ n : ℕ, color (m, n) = a ∨ color (m, n) = b :=
sorry

end infinite_grid_coloring_l90_90437


namespace solve_for_x_l90_90609

theorem solve_for_x (x : ℝ) (h : x / 5 + 3 = 4) : x = 5 :=
sorry

end solve_for_x_l90_90609


namespace smallest_stable_triangle_side_length_l90_90566

/-- The smallest possible side length that can appear in any stable triangle with side lengths that 
are multiples of 5, 80, and 112, respectively, is 20. -/
theorem smallest_stable_triangle_side_length {a b c : ℕ} 
  (hab : ∃ k₁, a = 5 * k₁) 
  (hbc : ∃ k₂, b = 80 * k₂) 
  (hac : ∃ k₃, c = 112 * k₃) 
  (abc_triangle : a + b > c ∧ a + c > b ∧ b + c > a) : 
  a = 20 ∨ b = 20 ∨ c = 20 :=
sorry

end smallest_stable_triangle_side_length_l90_90566


namespace value_of_Priyanka_l90_90719

-- Defining the context with the conditions
variables (X : ℕ) (Neha : ℕ) (Sonali Priyanka Sadaf Tanu : ℕ)
-- The conditions given in the problem
axiom h1 : Neha = X
axiom h2 : Sonali = 15
axiom h3 : Priyanka = 15
axiom h4 : Sadaf = Neha
axiom h5 : Tanu = Neha

-- Stating the theorem we need to prove
theorem value_of_Priyanka : Priyanka = 15 :=
by
  sorry

end value_of_Priyanka_l90_90719


namespace problem_statement_l90_90837

theorem problem_statement (p : ℝ) : 
  (∀ (q : ℝ), q > 0 → (3 * (p * q^2 + 2 * p^2 * q + 2 * q^2 + 5 * p * q)) / (p + q) > 3 * p^2 * q) 
  ↔ (0 ≤ p ∧ p ≤ 7.275) :=
sorry

end problem_statement_l90_90837


namespace prove_problem_l90_90105

noncomputable def proof_problem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 1) : Prop :=
  (1 + 1 / x) * (1 + 1 / y) ≥ 9

theorem prove_problem (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 1) : proof_problem x y hx hy h :=
  sorry

end prove_problem_l90_90105


namespace least_subtraction_for_divisibility_l90_90152

def original_number : ℕ := 5474827

def required_subtraction : ℕ := 7

theorem least_subtraction_for_divisibility :
  ∃ k : ℕ, (original_number - required_subtraction) = 12 * k :=
sorry

end least_subtraction_for_divisibility_l90_90152


namespace min_value_abs_b_minus_c_l90_90391

-- Define the problem conditions
def condition1 (a b c : ℝ) : Prop :=
  (a - 2 * b - 1)^2 + (a - c - Real.log c)^2 = 0

-- Define the theorem to be proved
theorem min_value_abs_b_minus_c {a b c : ℝ} (h : condition1 a b c) : |b - c| = 1 :=
sorry

end min_value_abs_b_minus_c_l90_90391


namespace bryan_total_books_and_magazines_l90_90836

-- Define the conditions
def books_per_shelf : ℕ := 23
def magazines_per_shelf : ℕ := 61
def bookshelves : ℕ := 29

-- Define the total books and magazines
def total_books : ℕ := books_per_shelf * bookshelves
def total_magazines : ℕ := magazines_per_shelf * bookshelves
def total_books_and_magazines : ℕ := total_books + total_magazines

-- The proof problem statement
theorem bryan_total_books_and_magazines : total_books_and_magazines = 2436 := 
by
  sorry

end bryan_total_books_and_magazines_l90_90836


namespace two_n_plus_m_is_36_l90_90179

theorem two_n_plus_m_is_36 (n m : ℤ) (h1 : 3 * n - m < 5) (h2 : n + m > 26) (h3 : 3 * m - 2 * n < 46) : 
  2 * n + m = 36 :=
sorry

end two_n_plus_m_is_36_l90_90179


namespace books_left_over_l90_90003

def total_books (box_count : ℕ) (books_per_box : ℤ) : ℤ :=
  box_count * books_per_box

theorem books_left_over
  (box_count : ℕ)
  (books_per_box : ℤ)
  (new_box_capacity : ℤ)
  (books_total : ℤ := total_books box_count books_per_box) :
  box_count = 1500 →
  books_per_box = 35 →
  new_box_capacity = 43 →
  books_total % new_box_capacity = 40 :=
by
  intros
  sorry

end books_left_over_l90_90003


namespace circle_equation_has_valid_k_l90_90111

theorem circle_equation_has_valid_k (k : ℝ) : (∃ a b r : ℝ, r > 0 ∧ ∀ x y : ℝ, (x - a) ^ 2 + (y - b) ^ 2 = r ^ 2) ↔ k < 5 / 4 := by
  sorry

end circle_equation_has_valid_k_l90_90111


namespace last_digit_base4_of_389_l90_90466

theorem last_digit_base4_of_389 : (389 % 4 = 1) :=
by sorry

end last_digit_base4_of_389_l90_90466


namespace problem_l90_90952

noncomputable def g (x : ℝ) : ℝ := 2 - 3 * x ^ 2

noncomputable def f (gx : ℝ) (x : ℝ) : ℝ := (2 - 3 * x ^ 2) / x ^ 2

theorem problem (x : ℝ) (hx : x ≠ 0) : f (g x) x = 3 / 2 :=
  sorry

end problem_l90_90952


namespace debra_probability_l90_90774

theorem debra_probability :
  let p_tail := 1 / 2
  let p_head := 1 / 2
  let p_THTHT := p_tail * p_head * p_tail * p_head * p_tail
  let P := (1 / 4) / (3 / 4)
  (p_THTHT * P) = 1 / 96 :=
by
  -- Definitions of p_tail, p_head, p_THTHT, and P
  let p_tail := 1 / 2
  let p_head := 1 / 2
  let p_THTHT := p_tail * p_head * p_tail * p_head * p_tail
  let P := (1 / 4) / (3 / 4)
  -- Placeholder for proof computation
  sorry

end debra_probability_l90_90774


namespace point_in_third_quadrant_l90_90368

def quadrant_of_point (x y : ℤ) : String :=
  if x > 0 ∧ y > 0 then "first"
  else if x < 0 ∧ y > 0 then "second"
  else if x < 0 ∧ y < 0 then "third"
  else if x > 0 ∧ y < 0 then "fourth"
  else "on_axis"

theorem point_in_third_quadrant : quadrant_of_point (-2) (-3) = "third" :=
  by sorry

end point_in_third_quadrant_l90_90368


namespace integer_combination_zero_l90_90763

theorem integer_combination_zero (a b c : ℤ) (h : a * Real.sqrt 2 + b * Real.sqrt 3 + c = 0) : 
  a = 0 ∧ b = 0 ∧ c = 0 :=
sorry

end integer_combination_zero_l90_90763


namespace solve_for_x_l90_90513

theorem solve_for_x (x : ℚ) :
  (3 + 1 / (2 + 1 / (3 + 3 / (4 + x)))) = 225 / 73 ↔ x = -647 / 177 :=
by sorry

end solve_for_x_l90_90513


namespace mn_sum_l90_90978

theorem mn_sum (M N : ℚ) (h1 : (4 : ℚ) / 7 = M / 63) (h2 : (4 : ℚ) / 7 = 84 / N) : M + N = 183 := sorry

end mn_sum_l90_90978


namespace team_A_wins_exactly_4_of_7_l90_90740

noncomputable def probability_team_A_wins_4_of_7 : ℚ :=
  (Nat.choose 7 4) * ((1/2)^4) * ((1/2)^3)

theorem team_A_wins_exactly_4_of_7 :
  probability_team_A_wins_4_of_7 = 35 / 128 := by
sorry

end team_A_wins_exactly_4_of_7_l90_90740


namespace correct_operation_l90_90083

theorem correct_operation (a b : ℝ) : 2 * a^2 * b - a^2 * b = a^2 * b :=
by
  sorry

end correct_operation_l90_90083


namespace similar_area_ratios_l90_90972

theorem similar_area_ratios (a₁ a₂ s₁ s₂ : ℝ) (h₁ : a₁ = s₁^2) (h₂ : a₂ = s₂^2) (h₃ : a₁ / a₂ = 1 / 9) (h₄ : s₁ = 4) : s₂ = 12 :=
by
  sorry

end similar_area_ratios_l90_90972


namespace complement_complement_l90_90099

theorem complement_complement (alpha : ℝ) (h : alpha = 35) : (90 - (90 - alpha)) = 35 := by
  -- proof goes here, but we write sorry to skip it
  sorry

end complement_complement_l90_90099


namespace triangle_angle_B_max_sin_A_plus_sin_C_l90_90910

theorem triangle_angle_B (a b c : ℝ) (A B C : ℝ) (h1 : (a - c) * Real.sin A + c * Real.sin C - b * Real.sin B = 0) 
  (h2 : a / Real.sin A = b / Real.sin B) (h3 : b / Real.sin B = c / Real.sin C) : 
  B = Real.arccos (1/2) := 
sorry

theorem max_sin_A_plus_sin_C (a b c : ℝ) (A B C : ℝ) (h1 : (a - c) * Real.sin A + c * Real.sin C - b * Real.sin B = 0) 
  (h2 : a / Real.sin A = b / Real.sin B) (h3 : b / Real.sin B = c / Real.sin C) 
  (hB : B = Real.arccos (1/2)) : 
  Real.sin A + Real.sin C = Real.sqrt 3 :=
sorry

end triangle_angle_B_max_sin_A_plus_sin_C_l90_90910


namespace Michelle_silver_beads_count_l90_90799

theorem Michelle_silver_beads_count 
  (total_beads : ℕ) (blue_beads : ℕ) (red_beads : ℕ) (white_beads : ℕ) (silver_beads : ℕ)
  (h1 : total_beads = 40)
  (h2 : blue_beads = 5)
  (h3 : red_beads = 2 * blue_beads)
  (h4 : white_beads = blue_beads + red_beads)
  (h5 : silver_beads = total_beads - (blue_beads + red_beads + white_beads)) :
  silver_beads = 10 :=
by
  sorry

end Michelle_silver_beads_count_l90_90799


namespace max_value_of_function_l90_90881

noncomputable def f (x : ℝ) (α : ℝ) := x ^ α

theorem max_value_of_function (α : ℝ)
  (h₁ : f 4 α = 2)
  : ∃ a : ℝ, 3 ≤ a ∧ a ≤ 5 ∧ (f (a - 3) (α) + f (5 - a) α = 2) := 
sorry

end max_value_of_function_l90_90881


namespace Stan_pays_magician_l90_90993

theorem Stan_pays_magician :
  let hours_per_day := 3
  let days_per_week := 7
  let weeks := 2
  let hourly_rate := 60
  let total_hours := hours_per_day * days_per_week * weeks
  let total_payment := hourly_rate * total_hours
  total_payment = 2520 := 
by 
  sorry

end Stan_pays_magician_l90_90993


namespace dividend_calculation_l90_90046

theorem dividend_calculation :
  let divisor := 12
  let quotient := 909809
  let dividend := divisor * quotient
  dividend = 10917708 :=
by
  let divisor := 12
  let quotient := 909809
  let dividend := divisor * quotient
  show dividend = 10917708
  sorry

end dividend_calculation_l90_90046


namespace tomatoes_picked_l90_90191

theorem tomatoes_picked (initial_tomatoes picked_tomatoes : ℕ)
  (h₀ : initial_tomatoes = 17)
  (h₁ : initial_tomatoes - picked_tomatoes = 8) :
  picked_tomatoes = 9 :=
by
  sorry

end tomatoes_picked_l90_90191


namespace model_x_computers_used_l90_90748

theorem model_x_computers_used
    (x_rate : ℝ)
    (y_rate : ℝ)
    (combined_rate : ℝ)
    (num_computers : ℝ) :
    x_rate = 1 / 72 →
    y_rate = 1 / 36 →
    combined_rate = num_computers * (x_rate + y_rate) →
    combined_rate = 1 →
    num_computers = 24 := by
  intros h1 h2 h3 h4
  sorry

end model_x_computers_used_l90_90748


namespace combined_flock_size_l90_90194

def original_ducks := 100
def killed_per_year := 20
def born_per_year := 30
def years_passed := 5
def another_flock := 150

theorem combined_flock_size :
  original_ducks + years_passed * (born_per_year - killed_per_year) + another_flock = 300 :=
by
  sorry

end combined_flock_size_l90_90194


namespace speed_conversion_l90_90242

theorem speed_conversion (s : ℚ) (h : s = 13 / 48) : 
  ((13 / 48) * 3.6 = 0.975) :=
by
  sorry

end speed_conversion_l90_90242


namespace unit_cubes_with_paint_l90_90254

/-- Conditions:
1. Cubes with each side one inch long are glued together to form a larger cube.
2. The larger cube's face is painted with red color and the entire assembly is taken apart.
3. 23 small cubes are found with no paints on them.
-/
theorem unit_cubes_with_paint (n : ℕ) (h1 : n^3 - (n - 2)^3 = 23) (h2 : n = 4) :
    n^3 - 23 = 41 :=
by
  sorry

end unit_cubes_with_paint_l90_90254


namespace martha_initial_crayons_l90_90547

theorem martha_initial_crayons : ∃ (x : ℕ), (x / 2 + 20 = 29) ∧ x = 18 :=
by
  sorry

end martha_initial_crayons_l90_90547


namespace total_people_l90_90116

-- Definitions of the given conditions
variable (I N B Ne T : ℕ)

-- These variables represent the given conditions
axiom h1 : I = 25
axiom h2 : N = 23
axiom h3 : B = 21
axiom h4 : Ne = 23

-- The theorem we want to prove
theorem total_people : T = 50 :=
by {
  sorry -- We denote the skipping of proof details.
}

end total_people_l90_90116


namespace selling_prices_max_profit_strategy_l90_90366

theorem selling_prices (x y : ℕ) (hx : y - x = 30) (hy : 2 * x + 3 * y = 740) : x = 130 ∧ y = 160 :=
by
  sorry

theorem max_profit_strategy (m : ℕ) (hm : 20 ≤ m ∧ m ≤ 80) 
(hcost : 90 * m + 110 * (80 - m) ≤ 8400) : m = 20 ∧ (80 - m) = 60 :=
by
  sorry

end selling_prices_max_profit_strategy_l90_90366


namespace roots_subtraction_l90_90412

theorem roots_subtraction (a b : ℝ) (h_roots : a * b = 20 ∧ a + b = 12) (h_order : a > b) : a - b = 8 :=
sorry

end roots_subtraction_l90_90412


namespace largest_possible_difference_l90_90768

theorem largest_possible_difference (A_est : ℕ) (B_est : ℕ) (A : ℝ) (B : ℝ)
(hA_est : A_est = 40000) (hB_est : B_est = 70000)
(hA_range : 36000 ≤ A ∧ A ≤ 44000)
(hB_range : 60870 ≤ B ∧ B ≤ 82353) :
  abs (B - A) = 46000 :=
by sorry

end largest_possible_difference_l90_90768


namespace equation_of_curve_E_equation_of_line_l_through_origin_intersecting_E_l90_90205

noncomputable def F₁ : ℝ × ℝ := (-Real.sqrt 3, 0)
noncomputable def F₂ : ℝ × ℝ := (Real.sqrt 3, 0)

def is_ellipse (P : ℝ × ℝ) : Prop :=
  Real.sqrt ((P.1 - F₁.1) ^ 2 + P.2 ^ 2) + Real.sqrt ((P.1 - F₂.1) ^ 2 + P.2 ^ 2) = 4

theorem equation_of_curve_E :
  ∀ P : ℝ × ℝ, is_ellipse P ↔ (P.1 ^ 2 / 4 + P.2 ^ 2 = 1) :=
sorry

def intersects_at_origin (C D : ℝ × ℝ) : Prop :=
  C.1 * D.1 + C.2 * D.2 = 0

theorem equation_of_line_l_through_origin_intersecting_E :
  ∀ (l : ℝ → ℝ) (C D : ℝ × ℝ),
    (l 0 = -2) →
    (∀ P : ℝ × ℝ, is_ellipse P ↔ (P.1, P.2) = (C.1, l C.1) ∨ (P.1, P.2) = (D.1, l D.1)) →
    intersects_at_origin C D →
    (∀ x, l x = 2 * x - 2) ∨ (∀ x, l x = -2 * x - 2) :=
sorry

end equation_of_curve_E_equation_of_line_l_through_origin_intersecting_E_l90_90205


namespace first_digit_base_9_of_y_l90_90965

def base_3_to_base_10 (n : Nat) : Nat := sorry
def base_10_to_base_9_first_digit (n : Nat) : Nat := sorry

theorem first_digit_base_9_of_y :
  let y := 11220022110022112221
  let base_10_y := base_3_to_base_10 y
  base_10_to_base_9_first_digit base_10_y = 4 :=
by
  let y := 11220022110022112221
  let base_10_y := base_3_to_base_10 y
  show base_10_to_base_9_first_digit base_10_y = 4
  sorry

end first_digit_base_9_of_y_l90_90965


namespace kite_diagonal_ratio_l90_90812

theorem kite_diagonal_ratio (a b x y : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a ≥ b)
  (hx1 : 0 ≤ x) (hx2 : x < a) (hy1 : 0 ≤ y) (hy2 : y < b)
  (orthogonal_diagonals : a^2 + y^2 = b^2 + x^2) :
  (a / b)^2 = 4 / 3 := 
sorry

end kite_diagonal_ratio_l90_90812


namespace sum_of_series_l90_90431

theorem sum_of_series : 
  (3 + 13 + 23 + 33 + 43) + (11 + 21 + 31 + 41 + 51) = 270 := by
  sorry

end sum_of_series_l90_90431


namespace smallest_number_increased_by_seven_divisible_by_37_47_53_l90_90088

theorem smallest_number_increased_by_seven_divisible_by_37_47_53 : 
  ∃ n : ℕ, (n + 7) % 37 = 0 ∧ (n + 7) % 47 = 0 ∧ (n + 7) % 53 = 0 ∧ n = 92160 :=
by
  sorry

end smallest_number_increased_by_seven_divisible_by_37_47_53_l90_90088


namespace remainder_of_x_pow_15_minus_1_div_x_plus_1_is_neg_2_l90_90272

theorem remainder_of_x_pow_15_minus_1_div_x_plus_1_is_neg_2 :
  (x^15 - 1) % (x + 1) = -2 := 
sorry

end remainder_of_x_pow_15_minus_1_div_x_plus_1_is_neg_2_l90_90272


namespace find_a_for_even_function_l90_90177

theorem find_a_for_even_function (a : ℝ) (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = (x + 1)*(x - a) ∧ f (-x) = f x) : a = 1 :=
sorry

end find_a_for_even_function_l90_90177


namespace calculation1_calculation2_calculation3_calculation4_l90_90014

-- Define the problem and conditions
theorem calculation1 : 9.5 * 101 = 959.5 := 
by 
  sorry

theorem calculation2 : 12.5 * 8.8 = 110 := 
by 
  sorry

theorem calculation3 : 38.4 * 187 - 15.4 * 384 + 3.3 * 16 = 1320 := 
by 
  sorry

theorem calculation4 : 5.29 * 73 + 52.9 * 2.7 = 529 := 
by 
  sorry

end calculation1_calculation2_calculation3_calculation4_l90_90014


namespace train_speed_ratio_l90_90896

theorem train_speed_ratio 
  (distance_2nd_train : ℕ)
  (time_2nd_train : ℕ)
  (speed_1st_train : ℚ)
  (H1 : distance_2nd_train = 400)
  (H2 : time_2nd_train = 4)
  (H3 : speed_1st_train = 87.5) :
  distance_2nd_train / time_2nd_train = 100 ∧ 
  (speed_1st_train / (distance_2nd_train / time_2nd_train)) = 7 / 8 :=
by
  sorry

end train_speed_ratio_l90_90896


namespace find_K_l90_90733

theorem find_K 
  (Z K : ℤ) 
  (hZ_range : 1000 < Z ∧ Z < 2000)
  (hZ_eq : Z = K^4)
  (hK_pos : K > 0) :
  K = 6 :=
by {
  sorry -- Proof to be filled in
}

end find_K_l90_90733


namespace general_formula_sum_of_b_l90_90669

variable {a : ℕ → ℕ} (b : ℕ → ℕ)
variable (n : ℕ)

def is_geometric_sequence (a : ℕ → ℕ) (q : ℕ) : Prop :=
  ∀ n : ℕ, a (n+2) = q * a (n+1)

def initial_conditions (a : ℕ → ℕ) : Prop :=
  a 1 + a 2 = 9 ∧ a 2 + a 3 = 18

theorem general_formula (q : ℕ) (h1 : is_geometric_sequence a q) (h2 : initial_conditions a) :
  a n = 3 * 2^(n - 1) :=
sorry

def b_n (a : ℕ → ℕ) (n : ℕ) : ℕ := a n + 2 * n

def sum_b (b : ℕ → ℕ) (n : ℕ) : ℕ :=
  (Finset.range n).sum b

theorem sum_of_b (h1 : ∀ m : ℕ, b m = a m + 2 * m) (h2 : initial_conditions a) :
  sum_b b n = 3 * 2^n + n * (n + 1) - 3 :=
sorry

end general_formula_sum_of_b_l90_90669


namespace condition_a_neither_necessary_nor_sufficient_for_b_l90_90526

theorem condition_a_neither_necessary_nor_sufficient_for_b {x y : ℝ} (h : ¬(x = 1 ∧ y = 2)) (k : ¬(x + y = 3)) : ¬((x ≠ 1 ∧ y ≠ 2) ↔ (x + y ≠ 3)) :=
by
  sorry

end condition_a_neither_necessary_nor_sufficient_for_b_l90_90526


namespace total_cost_of_digging_well_l90_90133

noncomputable def cost_of_digging (depth : ℝ) (diameter : ℝ) (cost_per_cubic_meter : ℝ) : ℝ :=
  let radius := diameter / 2
  let volume := Real.pi * (radius^2) * depth
  volume * cost_per_cubic_meter

theorem total_cost_of_digging_well :
  cost_of_digging 14 3 15 = 1484.4 :=
by
  sorry

end total_cost_of_digging_well_l90_90133


namespace find_m_range_l90_90280

variable {R : Type*} [LinearOrderedField R]
variable (f : R → R)
variable (m : R)

-- Define that the function f is monotonically increasing
def monotonically_increasing (f : R → R) : Prop :=
  ∀ ⦃x y : R⦄, x ≤ y → f x ≤ f y

-- Lean statement for the proof problem
theorem find_m_range (h1 : monotonically_increasing f) (h2 : f (2 * m - 3) > f (-m)) : m > 1 :=
by
  sorry

end find_m_range_l90_90280


namespace parallel_resistor_problem_l90_90126

theorem parallel_resistor_problem
  (x : ℝ)
  (r : ℝ := 2.2222222222222223)
  (y : ℝ := 5) : 
  (1 / r = 1 / x + 1 / y) → x = 4 :=
by sorry

end parallel_resistor_problem_l90_90126


namespace number_of_pages_in_each_chapter_l90_90032

variable (x : ℕ)  -- Variable for number of pages in each chapter

-- Definitions based on the problem conditions
def pages_read_before_4_o_clock := 10 * x
def pages_read_at_4_o_clock := 20
def pages_read_after_4_o_clock := 2 * x
def total_pages_read := pages_read_before_4_o_clock x + pages_read_at_4_o_clock + pages_read_after_4_o_clock x

-- The theorem statement
theorem number_of_pages_in_each_chapter (h : total_pages_read x = 500) : x = 40 :=
sorry

end number_of_pages_in_each_chapter_l90_90032


namespace ratio_of_arithmetic_sequences_l90_90283

-- Definitions for the conditions
variables {a_n b_n : ℕ → ℝ}
variables {S_n T_n : ℕ → ℝ}
variables (d_a d_b : ℝ)

-- Arithmetic sequences conditions
def is_arithmetic_sequence (u_n : ℕ → ℝ) (t : ℝ) (d : ℝ) : Prop :=
  ∀ (n : ℕ), u_n n = t + n * d

-- Sum of first n terms conditions
def sum_of_first_n_terms (u_n : ℕ → ℝ) (Sn : ℕ → ℝ) : Prop :=
  ∀ (n : ℕ), Sn n = n * (u_n 1 + u_n (n-1)) / 2

-- Main theorem statement
theorem ratio_of_arithmetic_sequences (h1 : is_arithmetic_sequence a_n (a_n 0) d_a)
                                     (h2 : is_arithmetic_sequence b_n (b_n 0) d_b)
                                     (h3 : sum_of_first_n_terms a_n S_n)
                                     (h4 : sum_of_first_n_terms b_n T_n)
                                     (h5 : ∀ n, (S_n n) / (T_n n) = (2 * n) / (3 * n + 1)) :
                                     ∀ n, (a_n n) / (b_n n) = (2 * n - 1) / (3 * n - 1) := sorry

end ratio_of_arithmetic_sequences_l90_90283


namespace minimize_sum_of_distances_l90_90196

theorem minimize_sum_of_distances (P : ℝ × ℝ) (A : ℝ × ℝ) (F : ℝ × ℝ) 
  (hP_on_parabola : P.2 ^ 2 = 2 * P.1)
  (hA : A = (3, 2)) 
  (hF : F = (1/2, 0)) : 
  |P - A| + |P - F| ≥ |(2, 2) - A| + |(2, 2) - F| :=
by sorry

end minimize_sum_of_distances_l90_90196


namespace common_number_is_eleven_l90_90362

theorem common_number_is_eleven 
  (a b c d e f g h i : ℝ)
  (H1 : (a + b + c + d + e) / 5 = 7)
  (H2 : (e + f + g + h + i) / 5 = 10)
  (H3 : (a + b + c + d + e + f + g + h + i) / 9 = 74 / 9) :
  e = 11 := 
sorry

end common_number_is_eleven_l90_90362


namespace expand_polynomial_l90_90878

theorem expand_polynomial :
  (3 * x^2 + 2 * x + 1) * (2 * x^2 + 3 * x + 4) = 6 * x^4 + 13 * x^3 + 20 * x^2 + 11 * x + 4 :=
by
  sorry

end expand_polynomial_l90_90878


namespace tourists_count_l90_90411

theorem tourists_count :
  ∃ (n : ℕ), (1 / 2 * n + 1 / 3 * n + 1 / 4 * n = 39) :=
by
  use 36
  sorry

end tourists_count_l90_90411


namespace nancy_pictures_left_l90_90920

-- Given conditions stated in the problem
def picturesZoo : Nat := 49
def picturesMuseum : Nat := 8
def picturesDeleted : Nat := 38

-- The statement of the problem, proving Nancy still has 19 pictures after deletions
theorem nancy_pictures_left : (picturesZoo + picturesMuseum) - picturesDeleted = 19 := by
  sorry

end nancy_pictures_left_l90_90920


namespace problem_solution_l90_90167

theorem problem_solution (m n p : ℝ) 
  (h1 : 1 * m + 4 * p - 2 = 0) 
  (h2 : 2 * 1 - 5 * p + n = 0) 
  (h3 : (m / (-4)) * (2 / 5) = -1) :
  n = -12 :=
sorry

end problem_solution_l90_90167


namespace average_ab_l90_90517

theorem average_ab {a b : ℝ} (h : (3 + 5 + 7 + a + b) / 5 = 15) : (a + b) / 2 = 30 :=
by
  sorry

end average_ab_l90_90517


namespace rate_grapes_l90_90970

/-- Given that Bruce purchased 8 kg of grapes at a rate G per kg, 8 kg of mangoes at the rate of 55 per kg, 
and paid a total of 1000 to the shopkeeper, prove that the rate per kg for the grapes (G) is 70. -/
theorem rate_grapes (G : ℝ) (h1 : 8 * G + 8 * 55 = 1000) : G = 70 :=
by 
  sorry

end rate_grapes_l90_90970


namespace billy_win_probability_l90_90463

-- Definitions of states and transition probabilities
def alice_step_prob_pos : ℚ := 1 / 2
def alice_step_prob_neg : ℚ := 1 / 2
def billy_step_prob_pos : ℚ := 2 / 3
def billy_step_prob_neg : ℚ := 1 / 3

-- Definitions of states in the Markov chain
inductive State
| S0 | S1 | Sm1 | S2 | Sm2 -- Alice's states
| T0 | T1 | Tm1 | T2 | Tm2 -- Billy's states

open State

-- The theorem statement: the probability that Billy wins the game
theorem billy_win_probability : 
  ∃ (P : State → ℚ), 
  P S0 = 11 / 19 ∧ P T0 = 14 / 19 ∧ 
  P S1 = 1 / 2 * P T0 ∧
  P Sm1 = 1 / 2 * P S0 + 1 / 2 ∧
  P T0 = 2 / 3 * P T1 + 1 / 3 * P Tm1 ∧
  P T1 = 2 / 3 + 1 / 3 * P S0 ∧
  P Tm1 = 2 / 3 * P T0 ∧
  P S2 = 0 ∧ P Sm2 = 1 ∧ P T2 = 1 ∧ P Tm2 = 0 := 
by 
  sorry

end billy_win_probability_l90_90463


namespace sin_cos_alpha_l90_90666

open Real

theorem sin_cos_alpha (α : ℝ) (h1 : sin (2 * α) = -sqrt 2 / 2) (h2 : α ∈ Set.Ioc (3 * π / 2) (2 * π)) :
  sin α + cos α = sqrt 2 / 2 :=
sorry

end sin_cos_alpha_l90_90666


namespace common_area_approximation_l90_90985

noncomputable def elliptical_domain (x y : ℝ) : Prop :=
  (x^2 / 3 + y^2 / 2) ≤ 1

noncomputable def circular_domain (x y : ℝ) : Prop :=
  (x^2 + y^2) ≤ 2

noncomputable def intersection_area : ℝ :=
  7.27

theorem common_area_approximation :
  ∃ area, 
    elliptical_domain x y ∧ circular_domain x y →
    abs (area - intersection_area) < 0.01 :=
sorry

end common_area_approximation_l90_90985


namespace h_inv_f_neg3_does_not_exist_real_l90_90958

noncomputable def h : ℝ → ℝ := sorry
noncomputable def f : ℝ → ℝ := sorry

theorem h_inv_f_neg3_does_not_exist_real (h_inv : ℝ → ℝ)
  (h_cond : ∀ (x : ℝ), f (h_inv (h x)) = 7 * x ^ 2 + 4) :
  ¬ ∃ x : ℝ, h_inv (f (-3)) = x :=
by 
  sorry

end h_inv_f_neg3_does_not_exist_real_l90_90958


namespace find_a7_a8_l90_90095

variable {R : Type*} [LinearOrderedField R]
variable {a : ℕ → R}

-- Conditions
def cond1 : a 1 + a 2 = 40 := sorry
def cond2 : a 3 + a 4 = 60 := sorry

-- Goal 
theorem find_a7_a8 : a 7 + a 8 = 135 := 
by 
  -- provide the actual proof here
  sorry

end find_a7_a8_l90_90095


namespace negative_sixty_represents_expenditure_l90_90314

def positive_represents_income (x : ℤ) : Prop := x > 0
def negative_represents_expenditure (x : ℤ) : Prop := x < 0

theorem negative_sixty_represents_expenditure :
  negative_represents_expenditure (-60) ∧ abs (-60) = 60 :=
by
  sorry

end negative_sixty_represents_expenditure_l90_90314


namespace sequence_formula_correct_l90_90402

-- Define the sequence S_n
def S (n : ℕ) : ℤ := n^2 - 2

-- Define the general term of the sequence a_n
def a (n : ℕ) : ℤ :=
  if n = 1 then -1 else 2 * n - 1

-- Theorem to prove that for the given S_n, the defined a_n is correct
theorem sequence_formula_correct (n : ℕ) (h : n > 0) : 
  a n = if n = 1 then -1 else S n - S (n - 1) :=
by sorry

end sequence_formula_correct_l90_90402


namespace intersection_of_P_and_Q_l90_90089

theorem intersection_of_P_and_Q (P Q : Set ℕ) (h1 : P = {1, 3, 6, 9}) (h2 : Q = {1, 2, 4, 6, 8}) :
  P ∩ Q = {1, 6} :=
by
  sorry

end intersection_of_P_and_Q_l90_90089


namespace find_y_l90_90754

theorem find_y (y : ℕ) : y = (12 ^ 3 * 6 ^ 4) / 432 → y = 5184 :=
by
  intro h
  rw [h]
  sorry

end find_y_l90_90754


namespace factorization_correct_l90_90036

theorem factorization_correct (x : ℝ) : x^2 - 9 = (x + 3) * (x - 3) :=
by 
  sorry

end factorization_correct_l90_90036


namespace sum_opposite_numbers_correct_opposite_sum_numbers_correct_l90_90963

def opposite (x : Int) : Int := -x

def sum_opposite_numbers (a b : Int) : Int := opposite a + opposite b

def opposite_sum_numbers (a b : Int) : Int := opposite (a + b)

theorem sum_opposite_numbers_correct (a b : Int) : sum_opposite_numbers (-6) 4 = 2 := 
by sorry

theorem opposite_sum_numbers_correct (a b : Int) : opposite_sum_numbers (-6) 4 = 2 := 
by sorry

end sum_opposite_numbers_correct_opposite_sum_numbers_correct_l90_90963


namespace circle_properties_l90_90942

theorem circle_properties (C : ℝ) (hC : C = 36) :
  let r := 18 / π
  let d := 36 / π
  let A := 324 / π
  2 * π * r = 36 ∧ d = 2 * r ∧ A = π * r^2 :=
by
  sorry

end circle_properties_l90_90942


namespace unique_solution_condition_l90_90986

variable (c d x : ℝ)

-- Define the equation
def equation : Prop := 4 * x - 7 + c = d * x + 3

-- Lean theorem for the proof problem
theorem unique_solution_condition :
  (∃! x, equation c d x) ↔ d ≠ 4 :=
sorry

end unique_solution_condition_l90_90986


namespace rational_expression_is_rational_l90_90328

theorem rational_expression_is_rational (a b c : ℚ) (h : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  ∃ r : ℚ, 
    r = Real.sqrt ((1 / (a - b)^2) + (1 / (b - c)^2) + (1 / (c - a)^2)) :=
sorry

end rational_expression_is_rational_l90_90328


namespace butterfly_cocoon_l90_90299

theorem butterfly_cocoon (c l : ℕ) (h1 : l + c = 120) (h2 : l = 3 * c) : c = 30 :=
by
  sorry

end butterfly_cocoon_l90_90299


namespace sin_690_degree_l90_90051

theorem sin_690_degree : Real.sin (690 * Real.pi / 180) = -1/2 :=
by
  sorry

end sin_690_degree_l90_90051


namespace integral_cos_square_div_one_plus_cos_minus_sin_squared_l90_90002

theorem integral_cos_square_div_one_plus_cos_minus_sin_squared:
  ∫ x in (-2 * Real.pi / 3 : Real)..0, (Real.cos x)^2 / (1 + Real.cos x - Real.sin x)^2 = (Real.sqrt 3) / 2 - Real.log 2 := 
by
  sorry

end integral_cos_square_div_one_plus_cos_minus_sin_squared_l90_90002


namespace number_of_students_with_type_B_l90_90817

theorem number_of_students_with_type_B
  (total_students : ℕ)
  (students_with_type_A : total_students ≠ 0 ∧ total_students ≠ 0 → 2 * total_students = 90)
  (students_with_type_B : 2 * total_students = 90) :
  2/5 * total_students = 18 :=
by
  sorry

end number_of_students_with_type_B_l90_90817


namespace apple_price_theorem_l90_90562

-- Given conditions
def apple_counts : List Nat := [20, 40, 60, 80, 100, 120, 140]

-- Helper function to calculate revenue for a given apple count.
def revenue (apples : Nat) (price_per_batch : Nat) (price_per_leftover : Nat) (batch_size : Nat) : Nat :=
  (apples / batch_size) * price_per_batch + (apples % batch_size) * price_per_leftover

-- Theorem stating that the price per 7 apples is 1 cent and 3 cents per leftover apple ensures equal revenue.
theorem apple_price_theorem : 
  ∀ seller ∈ apple_counts, 
  revenue seller 1 3 7 = 20 :=
by
  intros seller h_seller
  -- Proof will follow here
  sorry

end apple_price_theorem_l90_90562


namespace tangent_and_normal_are_correct_at_point_l90_90529

def point_on_curve (x y : ℝ) : Prop :=
  x^2 - 2*x*y + 3*y^2 - 2*y - 16 = 0

def tangent_line (x y : ℝ) : Prop :=
  2*x - 7*y + 19 = 0

def normal_line (x y : ℝ) : Prop :=
  7*x + 2*y - 13 = 0

theorem tangent_and_normal_are_correct_at_point
  (hx : point_on_curve 1 3) :
  tangent_line 1 3 ∧ normal_line 1 3 :=
by
  sorry

end tangent_and_normal_are_correct_at_point_l90_90529


namespace total_area_of_removed_triangles_l90_90921

theorem total_area_of_removed_triangles : 
  ∀ (side_length_of_square : ℝ) (hypotenuse_length_of_triangle : ℝ),
  side_length_of_square = 20 →
  hypotenuse_length_of_triangle = 10 →
  4 * (1/2 * (hypotenuse_length_of_triangle^2 / 2)) = 100 :=
by
  intros side_length_of_square hypotenuse_length_of_triangle h_side_length h_hypotenuse_length
  -- Proof would go here, but we add "sorry" to complete the statement
  sorry

end total_area_of_removed_triangles_l90_90921


namespace range_of_a_l90_90503

theorem range_of_a (a b c : ℝ) 
  (h1 : a^2 - b*c - 8*a + 7 = 0) 
  (h2 : b^2 + c^2 + b*c - 6*a + 6 = 0) :
  1 ≤ a ∧ a ≤ 9 :=
sorry

end range_of_a_l90_90503


namespace correct_sum_l90_90496

theorem correct_sum (a b c n : ℕ) (h_m_pos : 100 * a + 10 * b + c > 0) (h_n_pos : n > 0)
    (h_err_sum : 100 * a + 10 * c + b + n = 128) : 100 * a + 10 * b + c + n = 128 := 
by
  sorry

end correct_sum_l90_90496


namespace parallelogram_side_sum_l90_90039

variable (x y : ℚ)

theorem parallelogram_side_sum :
  4 * x - 1 = 10 →
  5 * y + 3 = 12 →
  x + y = 91 / 20 :=
by
  intros h1 h2
  sorry

end parallelogram_side_sum_l90_90039


namespace students_taking_neither_l90_90617

-- Definitions based on conditions
def total_students : ℕ := 60
def students_CS : ℕ := 40
def students_Elec : ℕ := 35
def students_both_CS_and_Elec : ℕ := 25

-- Lean statement to prove the number of students taking neither computer science nor electronics
theorem students_taking_neither : total_students - (students_CS + students_Elec - students_both_CS_and_Elec) = 10 :=
by
  sorry

end students_taking_neither_l90_90617


namespace edward_spent_money_l90_90038

-- Definitions based on the conditions
def books := 2
def cost_per_book := 3

-- Statement of the proof problem
theorem edward_spent_money : 
  (books * cost_per_book = 6) :=
by
  -- proof goes here
  sorry

end edward_spent_money_l90_90038


namespace calculate_land_tax_l90_90899

def plot_size : ℕ := 15
def cadastral_value_per_sotka : ℕ := 100000
def tax_rate : ℝ := 0.003

theorem calculate_land_tax :
  plot_size * cadastral_value_per_sotka * tax_rate = 4500 := 
by 
  sorry

end calculate_land_tax_l90_90899


namespace minimum_ab_l90_90612

theorem minimum_ab (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : ab = a + 4 * b + 5) : ab ≥ 25 :=
sorry

end minimum_ab_l90_90612


namespace cube_volume_l90_90172

theorem cube_volume (A : ℝ) (h : A = 24) : 
  ∃ V : ℝ, V = 8 :=
by
  sorry

end cube_volume_l90_90172


namespace average_age_of_others_when_youngest_was_born_l90_90416

noncomputable def average_age_when_youngest_was_born (total_people : ℕ) (average_age : ℕ) (youngest_age : ℕ) : ℚ :=
  let total_age := total_people * average_age
  let age_without_youngest := total_age - youngest_age
  age_without_youngest / (total_people - 1)

theorem average_age_of_others_when_youngest_was_born :
  average_age_when_youngest_was_born 7 30 7 = 33.833 :=
by
  sorry

end average_age_of_others_when_youngest_was_born_l90_90416


namespace find_bicycle_speed_l90_90821

-- Let's define the conditions first
def distance := 10  -- Distance in km
def time_diff := 1 / 3  -- Time difference in hours
def speed_of_bicycle (x : ℝ) := x
def speed_of_car (x : ℝ) := 2 * x

-- Prove the equation using the given conditions
theorem find_bicycle_speed (x : ℝ) (h : x ≠ 0) :
  (distance / speed_of_bicycle x) = (distance / speed_of_car x) + time_diff :=
by {
  sorry
}

end find_bicycle_speed_l90_90821


namespace range_of_m_l90_90260

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (m+1)*x^2 - m*x + m - 1 ≥ 0) ↔ m ≥ (2*Real.sqrt 3)/3 := by
  sorry

end range_of_m_l90_90260


namespace find_m_l90_90502

theorem find_m (m : ℝ) : (1 : ℝ) * (-4 : ℝ) + (2 : ℝ) * m = 0 → m = 2 :=
by
  sorry

end find_m_l90_90502


namespace cookie_cost_1_l90_90378

theorem cookie_cost_1 (C : ℝ) 
  (h1 : ∀ c, c > 0 → 1.2 * c = c + 0.2 * c)
  (h2 : 50 * (1.2 * C) = 60) :
  C = 1 :=
by
  sorry

end cookie_cost_1_l90_90378


namespace intersection_M_N_l90_90107

def M : Set ℤ := {-2, -1, 0, 1, 2}
def N : Set ℤ := {x | x^2 + 2*x - 3 ≤ 0}

theorem intersection_M_N :
  M ∩ N = {-2, -1, 0, 1} :=
by
  sorry

end intersection_M_N_l90_90107


namespace max_integer_value_l90_90306

theorem max_integer_value (x : ℝ) : 
  ∃ (n : ℤ), n = 15 ∧ ∀ x : ℝ, 
  (4 * x^2 + 8 * x + 19) / (4 * x^2 + 8 * x + 5) ≤ n :=
by
  sorry

end max_integer_value_l90_90306


namespace collinear_points_b_value_l90_90891

theorem collinear_points_b_value (b : ℝ)
    (h : let slope1 := (4 - (-6)) / ((2 * b + 1) - 4)
         let slope2 := (1 - (-6)) / ((-3 * b + 2) - 4)
         slope1 = slope2) :
    b = -1 / 44 :=
by
  have slope1 := (4 - (-6)) / ((2 * b + 1) - 4)
  have slope2 := (1 - (-6)) / ((-3 * b + 2) - 4)
  have := h
  sorry

end collinear_points_b_value_l90_90891


namespace webinar_end_time_correct_l90_90599

-- Define start time and duration as given conditions
def startTime : Nat := 3*60 + 15  -- 3:15 p.m. in minutes after noon
def duration : Nat := 350         -- duration of the webinar in minutes

-- Define the expected end time in minutes after noon (9:05 p.m. is 9*60 + 5 => 545 minutes after noon)
def endTimeExpected : Nat := 9*60 + 5

-- Statement to prove that the calculated end time matches the expected end time
theorem webinar_end_time_correct : startTime + duration = endTimeExpected :=
by
  sorry

end webinar_end_time_correct_l90_90599


namespace find_five_digit_number_l90_90916

theorem find_five_digit_number : 
  ∃ (A B C D E : ℕ), 
    (0 < A ∧ A ≤ 9) ∧ 
    (0 < B ∧ B ≤ 9) ∧ 
    (0 < C ∧ C ≤ 9) ∧ 
    (0 < D ∧ D ≤ 9) ∧ 
    (0 < E ∧ E ≤ 9) ∧ 
    (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E) ∧ 
    (B ≠ C ∧ B ≠ D ∧ B ≠ E) ∧ 
    (C ≠ D ∧ C ≠ E) ∧ 
    (D ≠ E) ∧ 
    (2016 = (10 * D + E) * A * B) ∧ 
    (¬ (10 * D + E) % 3 = 0) ∧ 
    (10^4 * A + 10^3 * B + 10^2 * C + 10 * D + E = 85132) :=
sorry

end find_five_digit_number_l90_90916


namespace train_length_l90_90030

theorem train_length (L S : ℝ) 
  (h1 : L = S * 40) 
  (h2 : L + 1800 = S * 120) : 
  L = 900 := 
by
  sorry

end train_length_l90_90030


namespace area_of_region_ABCDEFGHIJ_l90_90967

/-- 
  Given:
  1. Region ABCDEFGHIJ consists of 13 equal squares.
  2. Region ABCDEFGHIJ is inscribed in rectangle PQRS.
  3. Point A is on line PQ, B is on line QR, E is on line RS, and H is on line SP.
  4. PQ has length 28 and QR has length 26.

  Prove that the area of region ABCDEFGHIJ is 338 square units.
-/
theorem area_of_region_ABCDEFGHIJ 
  (squares : ℕ)             -- Number of squares in region ABCDEFGHIJ
  (len_PQ len_QR : ℕ)       -- Lengths of sides PQ and QR
  (area : ℕ)                 -- Area of region ABCDEFGHIJ
  (h1 : squares = 13)
  (h2 : len_PQ = 28)
  (h3 : len_QR = 26)
  : area = 338 :=
sorry

end area_of_region_ABCDEFGHIJ_l90_90967


namespace find_a4_l90_90307

def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ :=
  a + (n - 1) * d

def sum_first_n_terms (a d : ℤ) (n : ℕ) : ℤ :=
  n * a + (n * (n - 1) / 2) * d

theorem find_a4 (a d : ℤ)
    (h₁ : sum_first_n_terms a d 5 = 15)
    (h₂ : sum_first_n_terms a d 9 = 63) :
  arithmetic_sequence a d 4 = 5 :=
sorry

end find_a4_l90_90307


namespace wheat_grains_approximation_l90_90371

theorem wheat_grains_approximation :
  let total_grains : ℕ := 1536
  let wheat_per_sample : ℕ := 28
  let sample_size : ℕ := 224
  let wheat_estimate : ℕ := total_grains * wheat_per_sample / sample_size
  wheat_estimate = 169 := by
  sorry

end wheat_grains_approximation_l90_90371


namespace jamie_nickels_l90_90892

theorem jamie_nickels (x : ℕ) (hx : 5 * x + 10 * x + 25 * x = 1320) : x = 33 :=
sorry

end jamie_nickels_l90_90892


namespace abs_gt_two_l90_90711

theorem abs_gt_two (x : ℝ) : |x| > 2 → x > 2 ∨ x < -2 :=
by
  intros
  sorry

end abs_gt_two_l90_90711


namespace units_digit_pow_prod_l90_90571

theorem units_digit_pow_prod : 
  ((2 ^ 2023) * (5 ^ 2024) * (11 ^ 2025)) % 10 = 0 :=
by
  sorry

end units_digit_pow_prod_l90_90571


namespace quadratic_linear_common_solution_l90_90432

theorem quadratic_linear_common_solution
  (a x1 x2 d e : ℝ)
  (ha : a ≠ 0) (hx1x2 : x1 ≠ x2) (hd : d ≠ 0)
  (h_quad : ∀ x, a * (x - x1) * (x - x2) = 0 → x = x1 ∨ x = x2)
  (h_linear : d * x1 + e = 0)
  (h_combined : ∀ x, a * (x - x1) * (x - x2) + d * x + e = 0 → x = x1) :
  d = a * (x2 - x1) :=
by sorry

end quadratic_linear_common_solution_l90_90432


namespace solution_l90_90895

noncomputable def problem_statement (x y : ℝ) (hx : x > 1) (hy : y > 1) (h : (Real.log x / Real.log 3)^4 + (Real.log y / Real.log 5)^4 + 16 = 12 * (Real.log x / Real.log 3) * (Real.log y / Real.log 5)) : ℝ :=
  (x^2 * y^2)

theorem solution : ∀ x y : ℝ, x > 1 → y > 1 → (Real.log x / Real.log 3)^4 + (Real.log y / Real.log 5)^4 + 16 = 12 * (Real.log x / Real.log 3) * (Real.log y / Real.log 5) →
  (x^2 * y^2) = 225^(Real.sqrt 2) :=
by
  intros x y hx hy h
  sorry

end solution_l90_90895


namespace solution_set_of_inequality_l90_90659

theorem solution_set_of_inequality (a : ℝ) :
  (∀ x : ℝ, |x - 2| + |x + 3| ≥ a) ↔ a ≤ 5 :=
sorry

end solution_set_of_inequality_l90_90659


namespace cows_horses_ratio_l90_90033

theorem cows_horses_ratio (cows horses : ℕ) (h : cows = 21) (ratio : cows / horses = 7 / 2) : horses = 6 :=
sorry

end cows_horses_ratio_l90_90033


namespace marie_packs_construction_paper_l90_90904

theorem marie_packs_construction_paper (marie_glue_sticks : ℕ) (allison_glue_sticks : ℕ) (total_allison_items : ℕ)
    (glue_sticks_difference : allison_glue_sticks = marie_glue_sticks + 8)
    (marie_glue_sticks_count : marie_glue_sticks = 15)
    (total_items_allison : total_allison_items = 28)
    (marie_construction_paper_multiplier : ℕ)
    (construction_paper_ratio : marie_construction_paper_multiplier = 6) : 
    ∃ (marie_construction_paper_packs : ℕ), marie_construction_paper_packs = 30 := 
by
  sorry

end marie_packs_construction_paper_l90_90904


namespace acute_triangle_tangent_sum_range_l90_90724

theorem acute_triangle_tangent_sum_range
  (a b c : ℝ) (A B C : ℝ)
  (triangle_ABC_acute : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2)
  (opposite_sides : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (side_relation : b^2 - a^2 = a * c)
  (angle_relation : A + B + C = π)
  (angles_in_radians : 0 < A ∧ A < π)
  (angles_positive : A > 0 ∧ B > 0 ∧ C > 0) :
  1 < (1 / Real.tan A + 1 / Real.tan B) ∧ (1 / Real.tan A + 1 / Real.tan B) < 2 * Real.sqrt 3 / 3 :=
sorry 

end acute_triangle_tangent_sum_range_l90_90724


namespace fraction_of_students_with_buddy_l90_90844

theorem fraction_of_students_with_buddy (s n : ℕ) (h : n = 4 * s / 3) : 
  (n / 4 + s / 3) / (n + s : ℚ) = 2 / 7 :=
by
  sorry

end fraction_of_students_with_buddy_l90_90844


namespace smallest_n_value_l90_90827

theorem smallest_n_value (n : ℕ) (h : 15 * n - 2 ≡ 0 [MOD 11]) : n = 6 :=
sorry

end smallest_n_value_l90_90827


namespace find_f_2_l90_90645

noncomputable def f : ℝ → ℝ := sorry

axiom f_additive (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : f (x + y) = f x + f y
axiom f_8 : f 8 = 3

theorem find_f_2 : f 2 = 3 / 4 := 
by sorry

end find_f_2_l90_90645


namespace Amelia_wins_probability_correct_l90_90110

-- Define the probabilities
def probability_Amelia_heads := 1 / 3
def probability_Blaine_heads := 2 / 5

-- The infinite geometric series sum calculation for Amelia to win
def probability_Amelia_wins :=
  probability_Amelia_heads * (1 / (1 - (1 - probability_Amelia_heads) * (1 - probability_Blaine_heads)))

-- Given values p and q from the conditions
def p := 5
def q := 9

-- The correct answer $\frac{5}{9}$
def Amelia_wins_correct := 5 / 9

-- Prove that the probability calculation matches the given $\frac{5}{9}$, and find q - p
theorem Amelia_wins_probability_correct :
  probability_Amelia_wins = Amelia_wins_correct ∧ q - p = 4 := by
  sorry

end Amelia_wins_probability_correct_l90_90110


namespace average_of_w_x_z_l90_90488

theorem average_of_w_x_z (w x z y a : ℝ) (h1 : 2 / w + 2 / x + 2 / z = 2 / y)
  (h2 : w * x * z = y) (h3 : w + x + z = a) : (w + x + z) / 3 = a / 3 :=
by sorry

end average_of_w_x_z_l90_90488


namespace cases_needed_to_raise_funds_l90_90590

-- Define conditions as lemmas that will be used in the main theorem.
lemma packs_per_case : ℕ := 3
lemma muffins_per_pack : ℕ := 4
lemma muffin_price : ℕ := 2
lemma fundraising_goal : ℕ := 120

-- Calculate muffins per case
noncomputable def muffins_per_case : ℕ := packs_per_case * muffins_per_pack

-- Calculate money earned per case
noncomputable def money_per_case : ℕ := muffins_per_case * muffin_price

-- The main theorem to prove the number of cases needed
theorem cases_needed_to_raise_funds : 
  (fundraising_goal / money_per_case) = 5 :=
by
  sorry

end cases_needed_to_raise_funds_l90_90590


namespace largest_of_eight_consecutive_integers_l90_90209

theorem largest_of_eight_consecutive_integers (n : ℕ) 
  (h : 8 * n + 28 = 3652) : n + 7 = 460 := by 
  sorry

end largest_of_eight_consecutive_integers_l90_90209


namespace fixed_point_of_function_l90_90903

theorem fixed_point_of_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : (a^(2-2) - 3) = -2 :=
by
  sorry

end fixed_point_of_function_l90_90903


namespace min_b1_b2_l90_90595

-- Define the sequence recurrence relation
def sequence_recurrence (b : ℕ → ℕ) : Prop :=
  ∀ n ≥ 1, b (n + 2) = (b n + 2011) / (1 + b (n + 1))

-- Problem statement: Prove the minimum value of b₁ + b₂ is 2012
theorem min_b1_b2 (b : ℕ → ℕ) (h : ∀ n ≥ 1, 0 < b n) (rec : sequence_recurrence b) :
  b 1 + b 2 ≥ 2012 :=
sorry

end min_b1_b2_l90_90595


namespace students_brought_two_plants_l90_90218

theorem students_brought_two_plants 
  (a1 a2 a3 a4 a5 : ℕ) (p1 p2 p3 p4 p5 : ℕ)
  (h1 : a1 + a2 + a3 + a4 + a5 = 20)
  (h2 : a1 * p1 + a2 * p2 + a3 * p3 + a4 * p4 + a5 * p5 = 30)
  (h3 : p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p1 ≠ p5 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p2 ≠ p5 ∧
        p3 ≠ p4 ∧ p3 ≠ p5 ∧ p4 ≠ p5)
  : ∃ a : ℕ, a = 1 ∧ (∃ i : ℕ, p1 = 2 ∨ p2 = 2 ∨ p3 = 2 ∨ p4 = 2 ∨ p5 = 2) :=
sorry

end students_brought_two_plants_l90_90218


namespace first_trial_addition_amounts_l90_90785

-- Define the range and conditions for the biological agent addition amount.
def lower_bound : ℝ := 20
def upper_bound : ℝ := 30
def golden_ratio_method : ℝ := 0.618
def first_trial_addition_amount_1 : ℝ := lower_bound + (upper_bound - lower_bound) * golden_ratio_method
def first_trial_addition_amount_2 : ℝ := upper_bound - (upper_bound - lower_bound) * golden_ratio_method

-- Prove that the possible addition amounts for the first trial are 26.18g or 23.82g.
theorem first_trial_addition_amounts :
  (first_trial_addition_amount_1 = 26.18 ∨ first_trial_addition_amount_2 = 23.82) :=
by
  -- Placeholder for the proof.
  sorry

end first_trial_addition_amounts_l90_90785


namespace probability_of_rain_at_least_once_l90_90917

theorem probability_of_rain_at_least_once 
  (P_sat : ℝ) (P_sun : ℝ) (P_mon : ℝ)
  (h_sat : P_sat = 0.30)
  (h_sun : P_sun = 0.60)
  (h_mon : P_mon = 0.50) :
  (1 - (1 - P_sat) * (1 - P_sun) * (1 - P_mon)) * 100 = 86 :=
by
  rw [h_sat, h_sun, h_mon]
  sorry

end probability_of_rain_at_least_once_l90_90917


namespace right_triangle_integers_solutions_l90_90384

theorem right_triangle_integers_solutions :
  ∃ (a b c : ℕ), a ≤ b ∧ b ≤ c ∧ a^2 + b^2 = c^2 ∧ (a + b + c : ℕ) = (1 / 2 * a * b : ℚ) ∧
  ((a = 5 ∧ b = 12 ∧ c = 13) ∨ (a = 6 ∧ b = 8 ∧ c = 10)) :=
sorry

end right_triangle_integers_solutions_l90_90384


namespace find_number_l90_90495

theorem find_number (x : ℝ) (h : x / 5 = 30 + x / 6) : x = 900 :=
by
  -- Proof goes here
  sorry

end find_number_l90_90495


namespace range_of_a_l90_90731

noncomputable def range_a (a : ℝ) : Prop :=
  ∃ (x y : ℝ), 0 < x ∧ 0 < y ∧ (x^3 * Real.exp (y / x) = a * y^3)

theorem range_of_a (a : ℝ) : range_a a → a ≥ Real.exp 3 / 27 :=
by
  sorry

end range_of_a_l90_90731


namespace quadratic_min_value_l90_90198

theorem quadratic_min_value (k : ℝ) :
  (∀ x : ℝ, 3 ≤ x ∧ x ≤ 5 → y = (1/2) * (x - 1) ^ 2 + k) ∧
  (∀ y : ℝ, 3 ≤ y ∧ y ≤ 5 → y ≥ 3) → k = 1 :=
sorry

end quadratic_min_value_l90_90198


namespace right_triangle_area_l90_90016

open Real

theorem right_triangle_area
  (a : ℝ) 
  (h₁ : a > 0) 
  (h₂ : a < 24)
  (h₃ : 24^2 + a^2 = (48 - a)^2) : 
  1/2 * 24 * a = 216 :=
by
  -- This is just a statement, the proof is omitted
  sorry

end right_triangle_area_l90_90016


namespace distance_traveled_downstream_l90_90635

noncomputable def boat_speed_in_still_water : ℝ := 12
noncomputable def current_speed : ℝ := 4
noncomputable def travel_time_in_minutes : ℝ := 18
noncomputable def travel_time_in_hours : ℝ := travel_time_in_minutes / 60

theorem distance_traveled_downstream :
  let effective_speed := boat_speed_in_still_water + current_speed
  let distance := effective_speed * travel_time_in_hours
  distance = 4.8 := 
by
  sorry

end distance_traveled_downstream_l90_90635


namespace max_value_g_eq_3_in_interval_l90_90290

noncomputable def g (x : ℝ) : ℝ := 4 * x - x^4

theorem max_value_g_eq_3_in_interval : 
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 2 → g x ≤ 3) ∧ (∃ (x : ℝ), 0 ≤ x ∧ x ≤ 2 ∧ g x = 3) :=
by
  sorry

end max_value_g_eq_3_in_interval_l90_90290


namespace average_words_per_minute_l90_90851

theorem average_words_per_minute 
  (total_words : ℕ) 
  (total_hours : ℕ) 
  (h_words : total_words = 30000) 
  (h_hours : total_hours = 100) : 
  (total_words / total_hours / 60 = 5) := by
  sorry

end average_words_per_minute_l90_90851


namespace at_least_two_equal_l90_90626

theorem at_least_two_equal (x y z : ℝ) (h : (x - y) / (2 + x * y) + (y - z) / (2 + y * z) + (z - x) / (2 + z * x) = 0) : 
x = y ∨ y = z ∨ z = x := 
by
  sorry

end at_least_two_equal_l90_90626


namespace total_sum_of_ages_is_correct_l90_90506

-- Definition of conditions
def ageOfYoungestChild : Nat := 4
def intervals : Nat := 3

-- Total sum calculation
def sumOfAges (ageOfYoungestChild intervals : Nat) :=
  let Y := ageOfYoungestChild
  Y + (Y + intervals) + (Y + 2 * intervals) + (Y + 3 * intervals) + (Y + 4 * intervals)

theorem total_sum_of_ages_is_correct : sumOfAges 4 3 = 50 :=
by
  sorry

end total_sum_of_ages_is_correct_l90_90506


namespace points_above_y_eq_x_l90_90485

theorem points_above_y_eq_x (x y : ℝ) : (y > x) → (y, x) ∈ {p : ℝ × ℝ | p.2 < p.1} :=
by
  intro h
  sorry

end points_above_y_eq_x_l90_90485


namespace train_pass_bridge_in_approx_26_64_sec_l90_90500

noncomputable def L_train : ℝ := 240 -- Length of the train in meters
noncomputable def L_bridge : ℝ := 130 -- Length of the bridge in meters
noncomputable def Speed_train_kmh : ℝ := 50 -- Speed of the train in km/h
noncomputable def Speed_train_ms : ℝ := (Speed_train_kmh * 1000) / 3600 -- Speed of the train in m/s
noncomputable def Total_distance : ℝ := L_train + L_bridge -- Total distance to be covered by the train
noncomputable def Time : ℝ := Total_distance / Speed_train_ms -- Time to pass the bridge

theorem train_pass_bridge_in_approx_26_64_sec : |Time - 26.64| < 0.01 := by
  sorry

end train_pass_bridge_in_approx_26_64_sec_l90_90500


namespace range_alpha_minus_beta_l90_90787

theorem range_alpha_minus_beta (α β : ℝ) (h1 : -π ≤ α) (h2 : α ≤ β) (h3 : β ≤ π / 2) :
  - (3 * π) / 2 ≤ α - β ∧ α - β ≤ 0 :=
sorry

end range_alpha_minus_beta_l90_90787


namespace find_expression_value_l90_90019

theorem find_expression_value (x : ℝ) (h : x^2 - 3 * x - 1 = 0) : -3 * x^2 + 9 * x + 4 = 1 :=
by sorry

end find_expression_value_l90_90019


namespace average_percent_score_is_65_point_25_l90_90543

theorem average_percent_score_is_65_point_25 :
  let percent_score : List (ℕ × ℕ) := [(95, 10), (85, 20), (75, 40), (65, 50), (55, 60), (45, 15), (35, 5)]
  let total_students : ℕ := 200
  let total_score : ℕ := percent_score.foldl (fun acc p => acc + p.1 * p.2) 0
  (total_score : ℚ) / (total_students : ℚ) = 65.25 := by
{
  sorry
}

end average_percent_score_is_65_point_25_l90_90543


namespace minimum_omega_l90_90329

open Real

theorem minimum_omega (ω : ℕ) (h_ω_pos : ω > 0) :
  (∃ k : ℤ, ω * (π / 6) + (π / 6) = k * π + (π / 2)) → ω = 2 :=
by
  sorry

end minimum_omega_l90_90329


namespace eggs_supplied_l90_90262

-- Define the conditions
def daily_eggs_first_store (D : ℕ) : ℕ := 12 * D
def daily_eggs_second_store : ℕ := 30
def total_weekly_eggs (D : ℕ) : ℕ := 7 * (daily_eggs_first_store D + daily_eggs_second_store)

-- Statement: prove that if the total number of eggs supplied in a week is 630,
-- then Mark supplies 5 dozen eggs to the first store each day.
theorem eggs_supplied (D : ℕ) (h : total_weekly_eggs D = 630) : D = 5 :=
by
  sorry

end eggs_supplied_l90_90262


namespace expr_for_pos_x_min_value_l90_90195

section
variable {f : ℝ → ℝ}
variable {a : ℝ}

def even_func (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def func_def (f : ℝ → ℝ) (a : ℝ) : Prop := ∀ x, x ≤ 0 → f x = 4^(-x) - a * 2^(-x)

-- Assuming f is even and specified as in the problem for x ≤ 0
axiom ev_func : even_func f
axiom f_condition : 0 < a

theorem expr_for_pos_x (f : ℝ → ℝ) (a : ℝ) (h1 : even_func f) (h2 : func_def f a) : 
  ∀ x, 0 < x → f x = 4^x - a * 2^x :=
sorry -- this aims to prove the function's form for positive x.

theorem min_value (f : ℝ → ℝ) (a : ℝ) (h1 : even_func f) (h2 : func_def f a) :
  (0 < a ∧ a ≤ 2 → ∃ x, 0 < x ∧ f x = 1 - a) ∧
  (2 < a → ∃ x, 0 < x ∧ f x = -a^2 / 4) :=
sorry -- this aims to prove the minimum value on the interval (0, +∞).
end

end expr_for_pos_x_min_value_l90_90195


namespace point_is_in_second_quadrant_l90_90298

def in_second_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

theorem point_is_in_second_quadrant (x y : ℝ) (h₁ : x = -3) (h₂ : y = 2) :
  in_second_quadrant x y := 
by {
  sorry
}

end point_is_in_second_quadrant_l90_90298


namespace f_neg1_plus_f_2_l90_90379

def f (x : ℤ) : ℤ :=
  if x ≤ 0 then 4 * x else 2 * x

theorem f_neg1_plus_f_2 : f (-1) + f 2 = 0 := 
by
  -- Definition of f is provided above and conditions are met in that.
  sorry

end f_neg1_plus_f_2_l90_90379


namespace three_digit_even_two_odd_no_repetition_l90_90602

-- Define sets of digits
def digits : List ℕ := [0, 1, 3, 4, 5, 6]
def evens : List ℕ := [0, 4, 6]
def odds : List ℕ := [1, 3, 5]

noncomputable def total_valid_numbers : ℕ :=
  let choose_0 := 12 -- Given by A_{2}^{1} A_{3}^{2} = 12
  let without_0 := 36 -- Given by C_{2}^{1} * C_{3}^{2} * A_{3}^{3} = 36
  choose_0 + without_0

theorem three_digit_even_two_odd_no_repetition : total_valid_numbers = 48 :=
by
  -- Proof would be provided here
  sorry

end three_digit_even_two_odd_no_repetition_l90_90602


namespace complement_of_intersection_l90_90657

def U : Set ℕ := {0, 1, 2, 3}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {1, 2, 3}
def intersection : Set ℕ := M ∩ N
def complement : Set ℕ := U \ intersection

theorem complement_of_intersection (U M N : Set ℕ) :
  U = {0, 1, 2, 3} →
  M = {0, 1, 2} →
  N = {1, 2, 3} →
  (U \ (M ∩ N)) = {0, 3} := by
  intro hU hM hN
  simp [hU, hM, hN]
  sorry

end complement_of_intersection_l90_90657


namespace person_left_time_l90_90905

theorem person_left_time :
  ∃ (x y : ℚ), 
    0 ≤ x ∧ x < 1 ∧ 
    0 ≤ y ∧ y < 1 ∧ 
    (120 + 30 * x = 360 * y) ∧
    (360 * x = 150 + 30 * y) ∧
    (4 + x = 4 + 64 / 143) := 
by
  sorry

end person_left_time_l90_90905


namespace triangle_count_l90_90633

def count_triangles (smallest intermediate larger even_larger whole_structure : Nat) : Nat :=
  smallest + intermediate + larger + even_larger + whole_structure

theorem triangle_count :
  count_triangles 2 6 6 6 12 = 32 :=
by
  sorry

end triangle_count_l90_90633


namespace program_final_value_l90_90278

-- Define the program execution in a Lean function
def program_result (i : ℕ) (S : ℕ) : ℕ :=
  if i < 9 then S
  else program_result (i - 1) (S * i)

-- Initial conditions
def initial_i := 11
def initial_S := 1

-- The theorem to prove
theorem program_final_value : program_result initial_i initial_S = 990 := by
  sorry

end program_final_value_l90_90278


namespace find_n_l90_90831

theorem find_n 
  (n : ℕ) 
  (h_lcm : Nat.lcm n 16 = 48) 
  (h_gcf : Nat.gcd n 16 = 18) : 
  n = 54 := 
sorry

end find_n_l90_90831


namespace rainfall_second_week_l90_90053

theorem rainfall_second_week (x : ℝ) 
  (h1 : x + 1.5 * x = 25) :
  1.5 * x = 15 :=
by
  sorry

end rainfall_second_week_l90_90053


namespace solve_inequality_1_solve_inequality_2_l90_90458

-- Definitions based on given conditions
noncomputable def f (x : ℝ) : ℝ := abs (x + 1)

-- Lean statement for the first proof problem
theorem solve_inequality_1 :
  ∀ x : ℝ, f x ≤ 5 - f (x - 3) ↔ -2 ≤ x ∧ x ≤ 3 :=
by
  sorry

-- Lean statement for the second proof problem
theorem solve_inequality_2 (a : ℝ) :
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ 2 * f x + abs (x + a) ≤ x + 4) ↔ -2 ≤ a ∧ a ≤ 4 :=
by
  sorry

end solve_inequality_1_solve_inequality_2_l90_90458


namespace pool_capacity_l90_90990

variable (C : ℕ)

-- Conditions
def rate_first_valve := C / 120
def rate_second_valve := C / 120 + 50
def combined_rate := C / 48

-- Proof statement
theorem pool_capacity (C_pos : 0 < C) (h1 : rate_first_valve C + rate_second_valve C = combined_rate C) : C = 12000 := by
  sorry

end pool_capacity_l90_90990


namespace morgan_olivia_same_debt_l90_90005

theorem morgan_olivia_same_debt (t : ℝ) : 
  (200 * (1 + 0.12 * t) = 300 * (1 + 0.04 * t)) → 
  t = 25 / 3 :=
by
  sorry

end morgan_olivia_same_debt_l90_90005


namespace base_b_representation_1987_l90_90918

theorem base_b_representation_1987 (x y z b : ℕ) (h1 : x + y + z = 25) (h2 : x ≥ 1)
  (h3 : 1987 = x * b^2 + y * b + z) (h4 : 12 < b) (h5 : b < 45) :
  x = 5 ∧ y = 9 ∧ z = 11 ∧ b = 19 :=
sorry

end base_b_representation_1987_l90_90918


namespace solve_for_x_l90_90161

noncomputable def solve_equation (x : ℝ) : Prop := 
  (6 * x + 2) / (3 * x^2 + 6 * x - 4) = 3 * x / (3 * x - 2) ∧ x ≠ 2 / 3

theorem solve_for_x (x : ℝ) (h : solve_equation x) : x = (Real.sqrt 6) / 3 ∨ x = - (Real.sqrt 6) / 3 := 
  sorry

end solve_for_x_l90_90161


namespace units_digit_of_product_of_seven_consecutive_l90_90041

theorem units_digit_of_product_of_seven_consecutive (n : ℕ) : 
  ∃ d ∈ [n, n+1, n+2, n+3, n+4, n+5, n+6], d % 10 = 0 :=
by
  sorry

end units_digit_of_product_of_seven_consecutive_l90_90041


namespace x_ge_y_l90_90901

variable (a : ℝ)

def x : ℝ := 2 * a * (a + 3)
def y : ℝ := (a - 3) * (a + 3)

theorem x_ge_y : x a ≥ y a := 
by 
  sorry

end x_ge_y_l90_90901


namespace stream_speed_l90_90139

variable (v : ℝ)

def effective_speed_downstream (v : ℝ) : ℝ := 7.5 + v
def effective_speed_upstream (v : ℝ) : ℝ := 7.5 - v 

theorem stream_speed : (7.5 - v) / (7.5 + v) = 1 / 2 → v = 2.5 :=
by
  intro h
  -- Proof will be resolved here
  sorry

end stream_speed_l90_90139


namespace find_may_monday_l90_90473

noncomputable def weekday (day_of_month : ℕ) (first_day_weekday : ℕ) : ℕ :=
(day_of_month + first_day_weekday - 1) % 7

theorem find_may_monday (r n : ℕ) (condition1 : weekday r 5 = 5) (condition2 : weekday n 5 = 1) (condition3 : 15 < n ∧ n < 25) : 
  n = 20 :=
by
  -- Proof omitted.
  sorry

end find_may_monday_l90_90473


namespace adjacent_probability_is_2_over_7_l90_90325

variable (n : Nat := 5) -- number of student performances
variable (m : Nat := 2) -- number of teacher performances

/-- Total number of ways to insert two performances
    (ignoring adjacency constraints) into the program list. -/
def total_insertion_ways : Nat :=
  Fintype.card (Fin (n + m))

/-- Number of ways to insert two performances such that they are adjacent. -/
def adjacent_insertion_ways : Nat :=
  Fintype.card (Fin (n + 1))

/-- Probability that two specific performances are adjacent in a program list. -/
def adjacent_probability : ℚ :=
  adjacent_insertion_ways / total_insertion_ways

theorem adjacent_probability_is_2_over_7 :
  adjacent_probability = (2 : ℚ) / 7 := by
  sorry

end adjacent_probability_is_2_over_7_l90_90325


namespace train_cross_post_time_proof_l90_90569

noncomputable def train_cross_post_time (speed_kmh : ℝ) (length_m : ℝ) : ℝ :=
  let speed_ms := speed_kmh * 1000 / 3600
  length_m / speed_ms

theorem train_cross_post_time_proof : train_cross_post_time 40 190.0152 = 17.1 := by
  sorry

end train_cross_post_time_proof_l90_90569


namespace function_D_min_value_is_2_l90_90651

noncomputable def function_A (x : ℝ) : ℝ := x + 2
noncomputable def function_B (x : ℝ) : ℝ := Real.sin x + 2
noncomputable def function_C (x : ℝ) : ℝ := abs x + 2
noncomputable def function_D (x : ℝ) : ℝ := x^2 + 1

theorem function_D_min_value_is_2
  (x : ℝ) :
  ∃ x, function_D x = 2 := by
  sorry
 
end function_D_min_value_is_2_l90_90651


namespace additional_distance_if_faster_speed_l90_90823

-- Conditions
def speed_slow := 10 -- km/hr
def speed_fast := 15 -- km/hr
def actual_distance := 30 -- km

-- Question and answer
theorem additional_distance_if_faster_speed : (speed_fast * (actual_distance / speed_slow) - actual_distance) = 15 := by
  sorry

end additional_distance_if_faster_speed_l90_90823


namespace circle_condition_l90_90662

theorem circle_condition (k : ℝ) : (∃ x y : ℝ, x^2 + y^2 + 4 * x + 2 * y + 4 * k + 1 = 0) → (k < 1) :=
by
  sorry

end circle_condition_l90_90662


namespace circle_equation_focus_parabola_origin_l90_90984

noncomputable def parabola_focus (p : ℝ) (x y : ℝ) : Prop :=
  y^2 = 4 * p * x

def passes_through_origin (x y : ℝ) : Prop :=
  (0 - x)^2 + (0 - y)^2 = x^2 + y^2

theorem circle_equation_focus_parabola_origin :
  (∃ x y : ℝ, parabola_focus 1 x y ∧ passes_through_origin x y)
    → ∃ k : ℝ, (x^2 - 2 * x + y^2 = k) :=
sorry

end circle_equation_focus_parabola_origin_l90_90984


namespace triangle_side_length_condition_l90_90259

theorem triangle_side_length_condition (a : ℝ) (h₁ : a > 0) (h₂ : a + 2 > a + 5) (h₃ : a + 5 > a + 2) (h₄ : a + 2 + a + 5 > a) : a > 3 :=
by
  sorry

end triangle_side_length_condition_l90_90259


namespace range_of_m_l90_90929

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, |x - 1| + |x + m| ≤ 4) ↔ -5 ≤ m ∧ m ≤ 3 :=
by
  sorry

end range_of_m_l90_90929


namespace functional_equation_solution_l90_90192

def odd_integers (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem functional_equation_solution (f : ℤ → ℤ)
  (h_odd : ∀ x : ℤ, odd_integers (f x))
  (h_eq : ∀ x y : ℤ, 
    f (x + f x + y) + f (x - f x - y) = f (x + y) + f (x - y)) :
  ∃ (d k : ℤ) (ell : ℕ → ℤ), 
    (∀ i : ℕ, i < d → odd_integers (ell i)) ∧
    ∀ (m : ℤ) (i : ℕ), i < d → 
      f (m * d + i) = 2 * k * m * d + ell i :=
sorry

end functional_equation_solution_l90_90192


namespace max_min_values_of_g_l90_90276

noncomputable def g (x : ℝ) : ℝ := (Real.sin x)^8 + 8 * (Real.cos x)^8

theorem max_min_values_of_g :
  (∀ x : ℝ, g x ≤ 8) ∧ (∀ x : ℝ, g x ≥ 8 / 27) :=
by
  sorry

end max_min_values_of_g_l90_90276


namespace parabola_properties_l90_90055

theorem parabola_properties (p : ℝ) (h : p > 0) (F : ℝ × ℝ) (l : ℝ → ℝ) (A B : ℝ × ℝ) (M : ℝ × ℝ)
  (hp : p = 4) 
  (hF : F = (p / 2, 0)) 
  (hA : A.2^2 = 2 * p * A.1) 
  (hB : B.2^2 = 2 * p * B.1) 
  (hM : M = ((A.1 + B.1) / 2, 2)) 
  (hl : ∀ x, l x = 2 * x - 4) 
  : (p = 4) ∧ (l 0 = -4) ∧ (A ≠ B) → 
    (p = 4) ∧ (l 0 = -4) ∧ (A ≠ B) ∧ (|A.1 - B.1| + |A.2 - B.2| = 10) :=
by 
  sorry

end parabola_properties_l90_90055


namespace A_alone_time_l90_90453

theorem A_alone_time (x : ℕ) (h1 : 3 * x / 4  = 12) : x / 3 = 16 := by
  sorry

end A_alone_time_l90_90453


namespace coins_division_remainder_l90_90941

theorem coins_division_remainder :
  ∃ n : ℕ, (n % 8 = 6 ∧ n % 7 = 5 ∧ n % 9 = 0) :=
sorry

end coins_division_remainder_l90_90941


namespace total_amount_shared_l90_90456

theorem total_amount_shared (x y z : ℝ) (h1 : x = 1.25 * y) (h2 : y = 1.2 * z) (h3 : z = 400) :
  x + y + z = 1480 :=
by
  sorry

end total_amount_shared_l90_90456


namespace MrJones_pants_count_l90_90843

theorem MrJones_pants_count (P : ℕ) (h1 : 6 * P + P = 280) : P = 40 := by
  sorry

end MrJones_pants_count_l90_90843


namespace james_initial_marbles_l90_90475

theorem james_initial_marbles (m n : ℕ) (h1 : n = 4) (h2 : m / (n - 1) = 21) :
  m = 28 :=
by sorry

end james_initial_marbles_l90_90475


namespace projection_inequality_l90_90098

-- Define the problem with given Cartesian coordinate system, finite set of points in space, and their orthogonal projections
variable (O_xyz : Type) -- Cartesian coordinate system
variable (S : Finset O_xyz) -- finite set of points in space
variable (S_x S_y S_z : Finset O_xyz) -- sets of orthogonal projections onto the planes

-- Define the orthogonal projections (left as a comment here since detailed implementation is not specified)
-- (In Lean, actual definitions of orthogonal projections would follow mathematical and geometric definitions)

-- State the theorem to be proved
theorem projection_inequality :
  (Finset.card S) ^ 2 ≤ (Finset.card S_x) * (Finset.card S_y) * (Finset.card S_z) := 
sorry

end projection_inequality_l90_90098


namespace kate_change_l90_90232

def candyCost : ℝ := 0.54
def amountGiven : ℝ := 1.00
def change (amountGiven candyCost : ℝ) : ℝ := amountGiven - candyCost

theorem kate_change : change amountGiven candyCost = 0.46 := by
  sorry

end kate_change_l90_90232


namespace sufficient_but_not_necessary_l90_90187

def sequence_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n, a n < a (n + 1)

def abs_condition (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) > abs (a n)

theorem sufficient_but_not_necessary (a : ℕ → ℝ) :
  (abs_condition a → sequence_increasing a) ∧ ¬ (sequence_increasing a → abs_condition a) :=
by
  sorry

end sufficient_but_not_necessary_l90_90187


namespace solve_equation_l90_90438

theorem solve_equation (x : ℝ) : (⌊Real.sin x⌋:ℝ)^2 = Real.cos x ^ 2 - 1 ↔ ∃ n : ℤ, x = n * Real.pi := by
  sorry

end solve_equation_l90_90438


namespace no_nat_fun_satisfying_property_l90_90855

theorem no_nat_fun_satisfying_property :
  ¬ ∃ f : ℕ → ℕ, ∀ x : ℕ, f (f x) = x + 1 :=
by
  sorry

end no_nat_fun_satisfying_property_l90_90855


namespace union_sets_l90_90407

def A : Set ℝ := {x | -1 < x ∧ x < 1}
def B : Set ℝ := {x | 0 < x ∧ x < 2}

theorem union_sets : A ∪ B = {x | -1 < x ∧ x < 2} := by
  sorry

end union_sets_l90_90407


namespace trig_identity_proof_l90_90153

noncomputable def value_expr : ℝ :=
  (2 * Real.cos (10 * Real.pi / 180) - Real.sin (20 * Real.pi / 180)) / Real.sin (70 * Real.pi / 180)

theorem trig_identity_proof : value_expr = Real.sqrt 3 :=
by
  sorry

end trig_identity_proof_l90_90153


namespace polynomial_roots_l90_90279

theorem polynomial_roots :
  (∀ x : ℤ, (x^3 - 4*x^2 - 11*x + 24 = 0) ↔ (x = 4 ∨ x = 3 ∨ x = -1)) :=
sorry

end polynomial_roots_l90_90279


namespace ball_hits_ground_l90_90572

theorem ball_hits_ground 
  (y : ℝ → ℝ) 
  (height_eq : ∀ t, y t = -3 * t^2 - 6 * t + 90) :
  ∃ t : ℝ, y t = 0 ∧ t = 5.00 :=
by
  sorry

end ball_hits_ground_l90_90572


namespace mean_books_read_l90_90103

theorem mean_books_read :
  let readers1 := 4
  let books1 := 3
  let readers2 := 5
  let books2 := 5
  let readers3 := 2
  let books3 := 7
  let readers4 := 1
  let books4 := 10
  let total_readers := readers1 + readers2 + readers3 + readers4
  let total_books := (readers1 * books1) + (readers2 * books2) + (readers3 * books3) + (readers4 * books4)
  let mean_books := total_books / total_readers
  mean_books = 5.0833 :=
by
  sorry

end mean_books_read_l90_90103


namespace simplify_fraction_multiplication_l90_90565

theorem simplify_fraction_multiplication :
  (15/35) * (28/45) * (75/28) = 5/7 :=
by
  sorry

end simplify_fraction_multiplication_l90_90565


namespace Billy_weight_l90_90541

variables (Billy Brad Carl Dave Edgar : ℝ)

-- Conditions
def conditions :=
  Carl = 145 ∧
  Dave = Carl + 8 ∧
  Brad = Dave / 2 ∧
  Billy = Brad + 9 ∧
  Edgar = 3 * Dave ∧
  Edgar = Billy + 20

-- The statement to prove
theorem Billy_weight (Billy Brad Carl Dave Edgar : ℝ) (h : conditions Billy Brad Carl Dave Edgar) : Billy = 85.5 :=
by
  -- Proof would go here
  sorry

end Billy_weight_l90_90541


namespace solution_set_of_inequality_l90_90834

def fraction_inequality_solution : Set ℝ := {x : ℝ | -4 < x ∧ x < -1}

theorem solution_set_of_inequality (x : ℝ) :
  (2 - x) / (x + 4) > 1 ↔ -4 < x ∧ x < -1 := by
sorry

end solution_set_of_inequality_l90_90834


namespace exist_ints_a_b_l90_90680

theorem exist_ints_a_b (n : ℕ) : (∃ a b : ℤ, (n : ℤ) + a^2 = b^2) ↔ ¬ n % 4 = 2 := 
by
  sorry

end exist_ints_a_b_l90_90680


namespace calculate_percentage_passed_l90_90735

theorem calculate_percentage_passed (F_H F_E F_HE : ℝ) (h1 : F_H = 0.32) (h2 : F_E = 0.56) (h3 : F_HE = 0.12) :
  1 - (F_H + F_E - F_HE) = 0.24 := by
  sorry

end calculate_percentage_passed_l90_90735


namespace number_of_possible_x_values_l90_90726
noncomputable def triangle_sides_possible_values (x : ℕ) : Prop :=
  27 < x ∧ x < 63

theorem number_of_possible_x_values : 
  ∃ n, n = (62 - 28 + 1) ∧ ( ∀ x : ℕ, triangle_sides_possible_values x ↔ 28 ≤ x ∧ x ≤ 62) :=
sorry

end number_of_possible_x_values_l90_90726


namespace inequality_abc_l90_90991

theorem inequality_abc 
  (a b c : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_order : a ≥ b ∧ b ≥ c) 
  (h_sum : a + b + c ≤ 1) : 
  a^2 + 3 * b^2 + 5 * c^2 ≤ 1 := 
by sorry

end inequality_abc_l90_90991


namespace solve_for_s_l90_90294

-- Definition of the condition
def condition (s : ℝ) : Prop := (s - 60) / 3 = (6 - 3 * s) / 4

-- Theorem stating that if the condition holds, then s = 19.85
theorem solve_for_s (s : ℝ) : condition s → s = 19.85 := 
by {
  sorry -- Proof is skipped as per requirements
}

end solve_for_s_l90_90294


namespace integral_solutions_l90_90549

theorem integral_solutions (a b c : ℤ) (h : a^2 + b^2 + c^2 = a^2 * b^2) : a = 0 ∧ b = 0 ∧ c = 0 :=
sorry

end integral_solutions_l90_90549


namespace width_of_wall_l90_90093

-- Define the dimensions of a single brick.
def brick_length : ℝ := 25
def brick_width : ℝ := 11.25
def brick_height : ℝ := 6

-- Define the number of bricks.
def num_bricks : ℝ := 6800

-- Define the dimensions of the wall (length and height).
def wall_length : ℝ := 850
def wall_height : ℝ := 600

-- Prove that the width of the wall is 22.5 cm.
theorem width_of_wall : 
  (wall_length * wall_height * 22.5 = num_bricks * (brick_length * brick_width * brick_height)) :=
by
  sorry

end width_of_wall_l90_90093


namespace algebraic_expression_domain_l90_90439

theorem algebraic_expression_domain (x : ℝ) : (∃ y : ℝ, y = 1 / (x + 2)) ↔ (x ≠ -2) := 
sorry

end algebraic_expression_domain_l90_90439


namespace no_positive_divisor_of_2n2_square_l90_90020

theorem no_positive_divisor_of_2n2_square (n : ℕ) (hn : n > 0) : 
  ∀ d : ℕ, d > 0 → d ∣ 2 * n ^ 2 → ¬∃ x : ℕ, x ^ 2 = d ^ 2 * n ^ 2 + d ^ 3 := 
by
  sorry

end no_positive_divisor_of_2n2_square_l90_90020


namespace max_area_quadrilateral_l90_90310

theorem max_area_quadrilateral (a b c d : ℝ) (h1 : a = 1) (h2 : b = 4) (h3 : c = 7) (h4 : d = 8) : 
  ∃ A : ℝ, (A ≤ (1/2) * 1 * 8 + (1/2) * 4 * 7) ∧ (A = 18) :=
by
  sorry

end max_area_quadrilateral_l90_90310


namespace soda_cost_l90_90494

-- Definitions of the given conditions
def initial_amount : ℝ := 40
def cost_pizza : ℝ := 2.75
def cost_jeans : ℝ := 11.50
def quarters_left : ℝ := 97
def value_per_quarter : ℝ := 0.25

-- Calculate amount left in dollars
def amount_left : ℝ := quarters_left * value_per_quarter

-- Statement we want to prove: the cost of the soda
theorem soda_cost :
  initial_amount - amount_left - (cost_pizza + cost_jeans) = 1.5 :=
by
  sorry

end soda_cost_l90_90494


namespace honey_teas_l90_90255

-- Definitions corresponding to the conditions
def evening_cups := 2
def evening_servings_per_cup := 2
def morning_cups := 1
def morning_servings_per_cup := 1
def afternoon_cups := 1
def afternoon_servings_per_cup := 1
def servings_per_ounce := 6
def container_ounces := 16

-- Calculation for total servings of honey per day and total days until the container is empty
theorem honey_teas :
  (container_ounces * servings_per_ounce) / 
  (evening_cups * evening_servings_per_cup +
   morning_cups * morning_servings_per_cup +
   afternoon_cups * afternoon_servings_per_cup) = 16 :=
by
  sorry

end honey_teas_l90_90255


namespace simple_interest_time_period_l90_90550

theorem simple_interest_time_period 
  (P : ℝ) (R : ℝ := 4) (T : ℝ) (SI : ℝ := (2 / 5) * P) :
  SI = P * R * T / 100 → T = 10 :=
by {
  sorry
}

end simple_interest_time_period_l90_90550


namespace cos_sum_is_zero_l90_90839

theorem cos_sum_is_zero (x y z : ℝ) 
  (h1: Real.cos (2 * x) + 2 * Real.cos (2 * y) + 3 * Real.cos (2 * z) = 0) 
  (h2: Real.sin (2 * x) + 2 * Real.sin (2 * y) + 3 * Real.sin (2 * z) = 0) : 
  Real.cos (4 * x) + Real.cos (4 * y) + Real.cos (4 * z) = 0 := 
by 
  sorry

end cos_sum_is_zero_l90_90839


namespace total_marbles_l90_90312

variable (w o p : ℝ)

-- Conditions as hypothesis
axiom h1 : o + p = 10
axiom h2 : w + p = 12
axiom h3 : w + o = 5

theorem total_marbles : w + o + p = 13.5 :=
by
  sorry

end total_marbles_l90_90312


namespace factorize_polynomial_1_factorize_polynomial_2_triangle_shape_l90_90969

-- Proof for (1)
theorem factorize_polynomial_1 (a : ℝ) : 2*a^2 - 8*a + 8 = 2*(a - 2)^2 :=
by
  sorry

-- Proof for (2)
theorem factorize_polynomial_2 (x y : ℝ) : x^2 - y^2 + 3*x - 3*y = (x - y)*(x + y + 3) :=
by
  sorry

-- Proof for (3)
theorem triangle_shape (a b c : ℝ) (h : a^2 - ab - ac + bc = 0) : 
  (a = b ∨ a = c) :=
by
  sorry

end factorize_polynomial_1_factorize_polynomial_2_triangle_shape_l90_90969


namespace two_x_equals_y_l90_90681

theorem two_x_equals_y (x y : ℝ) (h1 : (x + y) / 3 = 1) (h2 : x + 2 * y = 5) : 2 * x = y := 
by
  sorry

end two_x_equals_y_l90_90681


namespace value_of_expression_l90_90199

theorem value_of_expression (m : ℝ) (α : ℝ) (h : m < 0) (h_M : M = (3 * m, -m)) :
  let sin_alpha := -m / (Real.sqrt 10 * -m)
  let cos_alpha := 3 * m / (Real.sqrt 10 * -m)
  (1 / (2 * sin_alpha * cos_alpha + cos_alpha^2) = 10 / 3) :=
by
  sorry

end value_of_expression_l90_90199


namespace proof_f_g_l90_90656

def f (x : ℝ) : ℝ := x^3 - 2*x^2 + 5*x - 1
def g (x : ℝ) : ℝ := 2*x + 3

theorem proof_f_g (x : ℝ) : f (g 2) - g (f 2) = 258 :=
by
  sorry

end proof_f_g_l90_90656


namespace impossible_to_save_one_minute_for_60kmh_l90_90169

theorem impossible_to_save_one_minute_for_60kmh (v : ℝ) (h : v = 60) :
  ¬ ∃ (new_v : ℝ), 1 / new_v = (1 / 60) - 1 :=
by
  sorry

end impossible_to_save_one_minute_for_60kmh_l90_90169


namespace series_sum_to_4_l90_90773

theorem series_sum_to_4 (x : ℝ) (hx : ∑' n : ℕ, (n + 1) * x^n = 4) : x = 1 / 2 := 
sorry

end series_sum_to_4_l90_90773


namespace total_course_selection_schemes_l90_90121

theorem total_course_selection_schemes :
  let PE_courses := 4
  let Art_courses := 4
  let total_courses := PE_courses + Art_courses
  let choose_two :=
    (Nat.choose PE_courses 1) * (Nat.choose Art_courses 1)
  let choose_three :=
    (Nat.choose PE_courses 2) * (Nat.choose Art_courses 1) +
    (Nat.choose PE_courses 1) * (Nat.choose Art_courses 2)
  total_courses = 8
  ∧ (choose_two + choose_three = 64) :=
by
  sorry

end total_course_selection_schemes_l90_90121


namespace skating_average_l90_90524

variable (minutesPerDay1 minutesPerDay2 : Nat)
variable (days1 days2 totalDays requiredAverage : Nat)

theorem skating_average :
  minutesPerDay1 = 80 →
  days1 = 6 →
  minutesPerDay2 = 100 →
  days2 = 2 →
  totalDays = 9 →
  requiredAverage = 95 →
  (minutesPerDay1 * days1 + minutesPerDay2 * days2 + x) / totalDays = requiredAverage →
  x = 175 :=
by
  intro h1 h2 h3 h4 h5 h6 h7
  sorry

end skating_average_l90_90524
