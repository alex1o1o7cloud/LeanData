import Mathlib

namespace incorrect_conclusion_intersection_l227_227540

theorem incorrect_conclusion_intersection :
  ∀ (x : ℝ), (0 = -2 * x + 4) → (x = 2) :=
by
  intro x h
  sorry

end incorrect_conclusion_intersection_l227_227540


namespace students_in_both_clubs_l227_227423

theorem students_in_both_clubs:
  ∀ (U D S : Finset ℕ ), (U.card = 300) → (D.card = 100) → (S.card = 140) → (D ∪ S).card = 210 → (D ∩ S).card = 30 := 
sorry

end students_in_both_clubs_l227_227423


namespace haley_seeds_total_l227_227813

-- Conditions
def seeds_in_big_garden : ℕ := 35
def small_gardens : ℕ := 7
def seeds_per_small_garden : ℕ := 3

-- Question rephrased as a problem with the correct answer
theorem haley_seeds_total : seeds_in_big_garden + small_gardens * seeds_per_small_garden = 56 := by
  sorry

end haley_seeds_total_l227_227813


namespace complement_A_in_U_l227_227445

def U : Set ℕ := {2, 3, 4}
def A : Set ℕ := {2, 3}

theorem complement_A_in_U : (U \ A) = {4} :=
by 
  sorry

end complement_A_in_U_l227_227445


namespace modulus_of_z_l227_227825

noncomputable def z : ℂ := (Complex.I / (1 + 2 * Complex.I))

theorem modulus_of_z : Complex.abs z = (Real.sqrt 5) / 5 := by
  sorry

end modulus_of_z_l227_227825


namespace license_plate_combinations_l227_227156

def num_choices_two_repeat_letters : ℕ :=
  (Nat.choose 26 2) * (Nat.choose 4 2) * (5 * 4)

theorem license_plate_combinations : num_choices_two_repeat_letters = 39000 := by
  sorry

end license_plate_combinations_l227_227156


namespace calculate_monthly_rent_l227_227858

theorem calculate_monthly_rent (P : ℝ) (R : ℝ) (T : ℝ) (M : ℝ) (rent : ℝ) :
  P = 12000 →
  R = 0.06 →
  T = 400 →
  M = 0.1 →
  rent = 103.70 :=
by
  intros hP hR hT hM
  sorry

end calculate_monthly_rent_l227_227858


namespace brownies_in_pan_l227_227969

theorem brownies_in_pan : 
    ∀ (pan_length pan_width brownie_length brownie_width : ℕ), 
    pan_length = 24 -> 
    pan_width = 20 -> 
    brownie_length = 3 -> 
    brownie_width = 2 -> 
    (pan_length * pan_width) / (brownie_length * brownie_width) = 80 := 
by
  intros pan_length pan_width brownie_length brownie_width h1 h2 h3 h4
  sorry

end brownies_in_pan_l227_227969


namespace minimum_candies_l227_227979

theorem minimum_candies (students : ℕ) (N : ℕ) (k : ℕ) : 
  students = 25 → 
  N = 25 * k → 
  (∀ n, 1 ≤ n → n ≤ students → ∃ m, n * k + m ≤ N) → 
  600 ≤ N := 
by
  intros hs hn hd
  sorry

end minimum_candies_l227_227979


namespace similar_triangles_x_value_l227_227650

theorem similar_triangles_x_value
  (x : ℝ)
  (h_similar : ∀ (AB BC DE EF : ℝ), AB / BC = DE / EF)
  (h_AB : AB = x)
  (h_BC : BC = 33)
  (h_DE : DE = 96)
  (h_EF : EF = 24) :
  x = 132 :=
by
  -- Proof steps will be here
  sorry

end similar_triangles_x_value_l227_227650


namespace inequality_hold_l227_227370

theorem inequality_hold {a b : ℝ} (h : a < b) : -3 * a > -3 * b :=
sorry

end inequality_hold_l227_227370


namespace largest_inscribed_parabola_area_l227_227932

noncomputable def maximum_parabolic_area_in_cone (r l : ℝ) : ℝ :=
  (l * r) / 2 * Real.sqrt 3

theorem largest_inscribed_parabola_area (r l : ℝ) : 
  ∃ t : ℝ, t = maximum_parabolic_area_in_cone r l :=
by
  let t_max := (l * r) / 2 * Real.sqrt 3
  use t_max
  sorry

end largest_inscribed_parabola_area_l227_227932


namespace molly_bike_miles_l227_227490

def total_miles_ridden (daily_miles years_riding days_per_year : ℕ) : ℕ :=
  daily_miles * years_riding * days_per_year

theorem molly_bike_miles :
  total_miles_ridden 3 3 365 = 3285 :=
by
  -- The definition and theorem are provided; the implementation will be done by the prover.
  sorry

end molly_bike_miles_l227_227490


namespace part1_part2_l227_227027

open Real

-- Definitions used in the proof
def p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0 ∧ a > 0
def q (x : ℝ) : Prop := abs (x - 1) ≤ 2 ∧ (x + 3) / (x - 2) ≥ 0

theorem part1 (x : ℝ) : (p 1 x ∧ q x) → 2 < x ∧ x ≤ 3 := by
  sorry

theorem part2 (a : ℝ) : (¬ (∃ x, p a x) → ¬ (∃ x, q x)) → a > 3 / 2 := by
  sorry

end part1_part2_l227_227027


namespace choir_members_l227_227616

theorem choir_members (n : ℕ) (h1 : n % 7 = 3) (h2 : n % 11 = 6) (h3 : 200 ≤ n ∧ n ≤ 300) :
  n = 220 :=
sorry

end choir_members_l227_227616


namespace smallest_yellow_marbles_l227_227617

def total_marbles (n : ℕ) := n

def blue_marbles (n : ℕ) := n / 3

def red_marbles (n : ℕ) := n / 4

def green_marbles := 6

def yellow_marbles (n : ℕ) := n - (blue_marbles n + red_marbles n + green_marbles)

theorem smallest_yellow_marbles (n : ℕ) (hn : n % 12 = 0) (blue : blue_marbles n = n / 3)
  (red : red_marbles n = n / 4) (green : green_marbles = 6) :
  yellow_marbles n = 4 ↔ n = 24 :=
by sorry

end smallest_yellow_marbles_l227_227617


namespace problem_solution_l227_227970

theorem problem_solution : (6 * 7 * 8 * 9 * 10) / (6 + 7 + 8 + 9 + 10) = 756 := by
  sorry

end problem_solution_l227_227970


namespace anna_has_4_twenty_cent_coins_l227_227159

theorem anna_has_4_twenty_cent_coins (x y : ℕ) (h1 : x + y = 15) (h2 : 59 - 3 * x = 24) : y = 4 :=
by {
  -- evidence based on the established conditions would be derived here
  sorry
}

end anna_has_4_twenty_cent_coins_l227_227159


namespace product_of_solutions_l227_227594

theorem product_of_solutions :
  ∀ x : ℝ, (x + 3) / (2 * x + 3) = (4 * x + 4) / (7 * x + 4) →
  (∀ x1 x2 : ℝ, (x1 ≠ x2) → (x = x1 ∨ x = x2) → x1 * x2 = 0) :=
by
  sorry

end product_of_solutions_l227_227594


namespace pass_rate_l227_227227

theorem pass_rate (total_students : ℕ) (students_not_passed : ℕ) (pass_rate : ℚ) :
  total_students = 500 → 
  students_not_passed = 40 → 
  pass_rate = (total_students - students_not_passed) / total_students * 100 →
  pass_rate = 92 :=
by 
  intros ht hs hpr 
  sorry

end pass_rate_l227_227227


namespace freshman_class_total_students_l227_227207

theorem freshman_class_total_students (N : ℕ) 
    (h1 : 90 ≤ N) 
    (h2 : 100 ≤ N)
    (h3 : 20 ≤ N) 
    (h4: (90 : ℝ) / N * (20 : ℝ) / 100 = (20 : ℝ) / N):
    N = 450 :=
  sorry

end freshman_class_total_students_l227_227207


namespace inequality_system_solution_l227_227946

theorem inequality_system_solution (x : ℝ) :
  (x + 7) / 3 ≤ x + 3 ∧ 2 * (x + 1) < x + 3 ↔ -1 ≤ x ∧ x < 1 :=
by
  sorry

end inequality_system_solution_l227_227946


namespace lucy_deposit_l227_227386

theorem lucy_deposit :
  ∃ D : ℝ, 
    let initial_balance := 65 
    let withdrawal := 4 
    let final_balance := 76 
    initial_balance + D - withdrawal = final_balance ∧ D = 15 :=
by
  -- sorry skips the proof
  sorry

end lucy_deposit_l227_227386


namespace equilateral_triangle_l227_227847

theorem equilateral_triangle 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : (a + b + c) * (b + c - a) = 3 * b * c) 
  (h2 : Real.sin A = 2 * Real.sin B * Real.cos C) 
  (h3 : a^2 = b^2 + c^2 - 2 * b * c * Real.cos A) 
  (h4 : b = c) : 
  A = 60 ∧ B = 60 ∧ C = 60 ∧ a = b ∧ b = c := 
sorry

end equilateral_triangle_l227_227847


namespace custom_mul_of_two_and_neg_three_l227_227561

-- Define the custom operation "*"
def custom.mul (a b : Int) : Int := a * b

-- The theorem to prove that 2 * (-3) using custom.mul equals -6
theorem custom_mul_of_two_and_neg_three : custom.mul 2 (-3) = -6 :=
by
  -- This is where the proof would go
  sorry

end custom_mul_of_two_and_neg_three_l227_227561


namespace cost_per_container_is_21_l227_227835

-- Define the given problem conditions as Lean statements.

--  Let w be the number of weeks represented by 210 days.
def number_of_weeks (days: ℕ) : ℕ := days / 7
def weeks : ℕ := number_of_weeks 210

-- Let p be the total pounds of litter used over the number of weeks.
def pounds_per_week : ℕ := 15
def total_litter_pounds (weeks: ℕ) : ℕ := weeks * pounds_per_week
def total_pounds : ℕ := total_litter_pounds weeks

-- Let c be the number of 45-pound containers needed for the total pounds of litter.
def pounds_per_container : ℕ := 45
def number_of_containers (total_pounds pounds_per_container: ℕ) : ℕ := total_pounds / pounds_per_container
def containers : ℕ := number_of_containers total_pounds pounds_per_container

-- Given the total cost, find the cost per container.
def total_cost : ℕ := 210
def cost_per_container (total_cost containers: ℕ) : ℕ := total_cost / containers
def cost : ℕ := cost_per_container total_cost containers

-- Prove that the cost per container is 21.
theorem cost_per_container_is_21 : cost = 21 := by
  sorry

end cost_per_container_is_21_l227_227835


namespace journey_speed_l227_227081

theorem journey_speed (v : ℚ) 
  (equal_distance : ∀ {d}, (d = 0.22) → ((0.66 / 3) = d))
  (total_distance : ∀ {d}, (d = 660 / 1000) → (660 / 1000 = 0.66))
  (total_time : ∀ {t} , (t = 11 / 60) → (11 / 60 = t)): 
  (0.22 / 2 + 0.22 / v + 0.22 / 6 = 11 / 60) → v = 1.2 := 
by 
  sorry

end journey_speed_l227_227081


namespace both_teams_joint_renovation_team_renovation_split_l227_227398

-- Problem setup for part 1
def renovation_total_length : ℕ := 2400
def teamA_daily_progress : ℕ := 30
def teamB_daily_progress : ℕ := 50
def combined_days_to_complete_renovation : ℕ := 30

theorem both_teams_joint_renovation (x : ℕ) :
  (teamA_daily_progress + teamB_daily_progress) * x = renovation_total_length → 
  x = combined_days_to_complete_renovation :=
by
  sorry

-- Problem setup for part 2
def total_renovation_days : ℕ := 60
def length_renovated_by_teamA : ℕ := 900
def length_renovated_by_teamB : ℕ := 1500

theorem team_renovation_split (a b : ℕ) :
  a / teamA_daily_progress + b / teamB_daily_progress = total_renovation_days ∧ 
  a + b = renovation_total_length → 
  a = length_renovated_by_teamA ∧ b = length_renovated_by_teamB :=
by
  sorry

end both_teams_joint_renovation_team_renovation_split_l227_227398


namespace unit_prices_purchase_plans_exchange_methods_l227_227181

theorem unit_prices (x r : ℝ) (hx : r = 2 * x) 
  (h_eq : (40/(2*r)) + 4 = 30/x) : 
  x = 2.5 ∧ r = 5 := sorry

theorem purchase_plans (x r : ℝ) (a b : ℕ)
  (hx : x = 2.5) (hr : r = 5) (h_eq : x * a + r * b = 200)
  (h_ge_20 : 20 ≤ a ∧ 20 ≤ b) (h_mult_10 : a % 10 = 0) :
  (a, b) = (20, 30) ∨ (a, b) = (30, 25) ∨ (a, b) = (40, 20) := sorry

theorem exchange_methods (a b t m : ℕ) 
  (hx : x = 2.5) (hr : r = 5) 
  (h_leq : 1 < m ∧ m < 10) 
  (h_eq : a + 2 * t = b + (m - t))
  (h_planA : (a = 20 ∧ b = 30) ∨ (a = 30 ∧ b = 25) ∨ (a = 40 ∧ b = 20)) :
  (m = 5 ∧ t = 5 ∧ b = 30) ∨
  (m = 8 ∧ t = 6 ∧ b = 25) ∨
  (m = 5 ∧ t = 0 ∧ b = 25) ∨
  (m = 8 ∧ t = 1 ∧ b = 20) := sorry

end unit_prices_purchase_plans_exchange_methods_l227_227181


namespace football_cost_l227_227742

theorem football_cost (cost_shorts cost_shoes money_have money_need : ℝ)
  (h_shorts : cost_shorts = 2.40)
  (h_shoes : cost_shoes = 11.85)
  (h_have : money_have = 10)
  (h_need : money_need = 8) :
  (money_have + money_need - (cost_shorts + cost_shoes) = 3.75) :=
by
  -- Proof goes here
  sorry

end football_cost_l227_227742


namespace martha_savings_l227_227572

-- Definitions based on conditions
def weekly_latte_spending : ℝ := 4.00 * 5
def weekly_iced_coffee_spending : ℝ := 2.00 * 3
def total_weekly_coffee_spending : ℝ := weekly_latte_spending + weekly_iced_coffee_spending
def annual_coffee_spending : ℝ := total_weekly_coffee_spending * 52
def savings_percentage : ℝ := 0.25

-- The theorem to be proven
theorem martha_savings : annual_coffee_spending * savings_percentage = 338.00 := by
  sorry

end martha_savings_l227_227572


namespace min_value_ratio_l227_227772

variable {α : Type*} [LinearOrderedField α]

theorem min_value_ratio (a : ℕ → α) (h1 : a 7 = a 6 + 2 * a 5) (h2 : ∃ m n : ℕ, a m * a n = 8 * a 1^2) :
  ∃ m n : ℕ, (1 / m + 4 / n = 11 / 6) :=
by
  sorry

end min_value_ratio_l227_227772


namespace distinct_ints_divisibility_l227_227289

theorem distinct_ints_divisibility
  (x y z : ℤ) 
  (h1 : x ≠ y) 
  (h2 : y ≠ z) 
  (h3 : z ≠ x) : 
  ∃ k : ℤ, (x - y) ^ 5 + (y - z) ^ 5 + (z - x) ^ 5 = 5 * (y - z) * (z - x) * (x - y) * k := 
by 
  sorry

end distinct_ints_divisibility_l227_227289


namespace lines_perpendicular_if_one_perpendicular_and_one_parallel_l227_227126

def Line : Type := sorry  -- Define the type representing lines
def Plane : Type := sorry  -- Define the type representing planes

def is_perpendicular_to_plane (a : Line) (α : Plane) : Prop := sorry  -- Definition for a line being perpendicular to a plane
def is_parallel_to_plane (b : Line) (α : Plane) : Prop := sorry  -- Definition for a line being parallel to a plane
def is_perpendicular (a b : Line) : Prop := sorry  -- Definition for a line being perpendicular to another line

theorem lines_perpendicular_if_one_perpendicular_and_one_parallel 
  (a b : Line) (α : Plane) 
  (h1 : is_perpendicular_to_plane a α) 
  (h2 : is_parallel_to_plane b α) : 
  is_perpendicular a b := 
sorry

end lines_perpendicular_if_one_perpendicular_and_one_parallel_l227_227126


namespace ellipse_find_m_l227_227015

theorem ellipse_find_m (a b m e : ℝ) 
  (h1 : a^2 = 4) 
  (h2 : b^2 = m)
  (h3 : e = 1/2) :
  m = 3 := 
by
  sorry

end ellipse_find_m_l227_227015


namespace distance_BC_l227_227640

theorem distance_BC (AB AC CD DA: ℝ) (hAB: AB = 50) (hAC: AC = 40) (hCD: CD = 25) (hDA: DA = 35):
  BC = 10 ∨ BC = 90 :=
by
  sorry

end distance_BC_l227_227640


namespace corina_problem_l227_227795

variable (P Q : ℝ)

theorem corina_problem (h1 : P + Q = 16) (h2 : P - Q = 4) : P = 10 :=
sorry

end corina_problem_l227_227795


namespace not_p_and_not_q_true_l227_227298

variable (p q: Prop)

theorem not_p_and_not_q_true (h1: ¬ (p ∧ q)) (h2: ¬ (p ∨ q)) : ¬ p ∧ ¬ q :=
by
  sorry

end not_p_and_not_q_true_l227_227298


namespace sin_product_identity_sin_cos_fraction_identity_l227_227482

-- First Proof Problem: Proving that the product of sines equals the given value
theorem sin_product_identity :
  (Real.sin (Real.pi * 6 / 180) * 
   Real.sin (Real.pi * 42 / 180) * 
   Real.sin (Real.pi * 66 / 180) * 
   Real.sin (Real.pi * 78 / 180)) = 
  (Real.sqrt 5 - 1) / 32 := 
by 
  sorry

-- Second Proof Problem: Given sin alpha and alpha in the second quadrant, proving the given fraction value
theorem sin_cos_fraction_identity (α : Real) 
  (h1 : π/2 < α ∧ α < π)
  (h2 : Real.sin α = Real.sqrt 15 / 4) :
  (Real.sin (α + Real.pi / 4)) / 
  (Real.sin (2 * α) + Real.cos (2 * α) + 1) = 
  -Real.sqrt 2 :=
by 
  sorry

end sin_product_identity_sin_cos_fraction_identity_l227_227482


namespace outer_boundary_diameter_l227_227629

theorem outer_boundary_diameter (statue_width garden_width path_width fountain_diameter : ℝ) 
  (h_statue : statue_width = 2) 
  (h_garden : garden_width = 10) 
  (h_path : path_width = 8) 
  (h_fountain : fountain_diameter = 12) : 
  2 * ((fountain_diameter / 2 + statue_width) + garden_width + path_width) = 52 :=
by
  sorry

end outer_boundary_diameter_l227_227629


namespace find_m_value_l227_227515

theorem find_m_value (f g : ℝ → ℝ) (m : ℝ)
  (h1 : ∀ (x : ℝ), f x = 4 * x^2 - 3 * x + 5)
  (h2 : ∀ (x : ℝ), g x = 2 * x^2 - m * x + 8)
  (h3 : f 5 - g 5 = 15) :
  m = -17 / 5 :=
by
  sorry

end find_m_value_l227_227515


namespace box_dimensions_sum_l227_227658

theorem box_dimensions_sum (A B C : ℝ)
  (h1 : A * B = 18)
  (h2 : A * C = 32)
  (h3 : B * C = 50) :
  A + B + C = 57.28 := 
sorry

end box_dimensions_sum_l227_227658


namespace Vanya_two_digit_number_l227_227692

-- Define the conditions as a mathematical property
theorem Vanya_two_digit_number:
  ∃ (m n : ℕ), 0 ≤ m ∧ m ≤ 9 ∧ 0 ≤ n ∧ n ≤ 9 ∧ (10 * n + m) ^ 2 = 4 * (10 * m + n) ∧ (10 * m + n) = 81 :=
by
  -- Remember to replace the proof with 'sorry'
  sorry

end Vanya_two_digit_number_l227_227692


namespace tank_fraction_l227_227431

theorem tank_fraction (x : ℚ) (h₁ : 48 * x + 8 = 48 * (9 / 10)) : x = 2 / 5 :=
by
  sorry

end tank_fraction_l227_227431


namespace not_inequality_neg_l227_227001

theorem not_inequality_neg (x y : ℝ) (h : x > y) : ¬ (-x > -y) :=
by {
  sorry
}

end not_inequality_neg_l227_227001


namespace sufficient_conditions_for_equation_l227_227812

theorem sufficient_conditions_for_equation 
  (a b c : ℤ) :
  (a = b ∧ b = c + 1) ∨ (a = c ∧ b - 1 = c) →
  a * (a - b) + b * (b - c) + c * (c - a) = 2 :=
by
  sorry

end sufficient_conditions_for_equation_l227_227812


namespace temperature_on_Friday_l227_227115

-- Define the temperatures for each day
variables (M T W Th F : ℕ)

-- Declare the given conditions as assumptions
axiom cond1 : (M + T + W + Th) / 4 = 48
axiom cond2 : (T + W + Th + F) / 4 = 46
axiom cond3 : M = 40

-- State the theorem
theorem temperature_on_Friday : F = 32 :=
by
  -- This is where the proof would go, but we'll use sorry to skip it
  sorry

end temperature_on_Friday_l227_227115


namespace first_player_win_boards_l227_227203

-- Define what it means for a player to guarantee a win
def first_player_guarantees_win (n m : ℕ) : Prop :=
  ¬(n % 2 = 1 ∧ m % 2 = 1)

-- The main theorem that matches the math proof problem
theorem first_player_win_boards : (first_player_guarantees_win 6 7) ∧
                                  (first_player_guarantees_win 6 8) ∧
                                  (first_player_guarantees_win 7 8) ∧
                                  (first_player_guarantees_win 8 8) ∧
                                  ¬(first_player_guarantees_win 7 7) := 
by 
sorry

end first_player_win_boards_l227_227203


namespace chessboard_disk_cover_l227_227741

noncomputable def chessboardCoveredSquares : ℕ :=
  let D : ℝ := 1 -- assuming D is a positive real number; actual value irrelevant as it gets cancelled in the comparison
  let grid_size : ℕ := 8
  let total_squares : ℕ := grid_size * grid_size
  let boundary_squares : ℕ := 28 -- pre-calculated in the insides steps
  let interior_squares : ℕ := total_squares - boundary_squares
  let non_covered_corners : ℕ := 4
  interior_squares - non_covered_corners

theorem chessboard_disk_cover : chessboardCoveredSquares = 32 := sorry

end chessboard_disk_cover_l227_227741


namespace simplify_nested_fourth_roots_l227_227459

variable (M : ℝ)
variable (hM : M > 1)

theorem simplify_nested_fourth_roots : 
  (M^(1/4) * (M^(1/4) * (M^(1/4) * M)^(1/4))^(1/4))^(1/4) = M^(21/64) := by
  sorry

end simplify_nested_fourth_roots_l227_227459


namespace laura_weekly_mileage_l227_227831

-- Define the core conditions

-- Distance to school per round trip (house <-> school)
def school_trip_distance : ℕ := 20

-- Number of trips to school per week
def school_trips_per_week : ℕ := 7

-- Distance to supermarket: 10 miles farther than school
def extra_distance_to_supermarket : ℕ := 10
def supermarket_trip_distance : ℕ := school_trip_distance + 2 * extra_distance_to_supermarket

-- Number of trips to supermarket per week
def supermarket_trips_per_week : ℕ := 2

-- Calculate the total weekly distance
def total_distance_per_week : ℕ := 
  (school_trips_per_week * school_trip_distance) +
  (supermarket_trips_per_week * supermarket_trip_distance)

-- Theorem to prove the total distance Laura drives per week
theorem laura_weekly_mileage :
  total_distance_per_week = 220 := by
  sorry

end laura_weekly_mileage_l227_227831


namespace min_geometric_ratio_l227_227786

theorem min_geometric_ratio (q : ℝ) (a : ℕ → ℝ) (h1 : ∀ n, a (n + 1) = a n * q)
(h2 : 1 < q) (h3 : q < 2) : q = 6 / 5 := by
  sorry

end min_geometric_ratio_l227_227786


namespace inequality_solution_l227_227469

theorem inequality_solution (x : ℝ) :
  (2 * x^2 + x < 6) ↔ (-2 < x ∧ x < 3 / 2) :=
by
  sorry

end inequality_solution_l227_227469


namespace find_k_in_expression_l227_227427

theorem find_k_in_expression :
  (2^1004 + 5^1005)^2 - (2^1004 - 5^1005)^2 = 20 * 10^1004 :=
by
  sorry

end find_k_in_expression_l227_227427


namespace convert_polar_to_rectangular_example_l227_227172

noncomputable def polar_to_rectangular (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

theorem convert_polar_to_rectangular_example :
  polar_to_rectangular 6 (5 * Real.pi / 2) = (0, 6) := by
  sorry

end convert_polar_to_rectangular_example_l227_227172


namespace dan_total_marbles_l227_227543

theorem dan_total_marbles (violet_marbles : ℕ) (red_marbles : ℕ) (h₁ : violet_marbles = 64) (h₂ : red_marbles = 14) : violet_marbles + red_marbles = 78 :=
sorry

end dan_total_marbles_l227_227543


namespace inequality_solution_l227_227887

theorem inequality_solution (x : ℝ) : 
  (2 * x) / (x + 2) ≤ 3 ↔ x ∈ Set.Iic (-6) ∪ Set.Ioi (-2) :=
by
  sorry

end inequality_solution_l227_227887


namespace solve_system_of_equations_l227_227989

theorem solve_system_of_equations (a b c x y z : ℝ)
  (h1 : a^3 + a^2 * x + a * y + z = 0)
  (h2 : b^3 + b^2 * x + b * y + z = 0)
  (h3 : c^3 + c^2 * x + c * y + z = 0) :
  x = -(a + b + c) ∧ y = ab + ac + bc ∧ z = -abc :=
by {
  sorry
}

end solve_system_of_equations_l227_227989


namespace negative_movement_south_l227_227583

noncomputable def movement_interpretation (x : ℤ) : String :=
if x > 0 then 
  "moving " ++ toString x ++ "m north"
else 
  "moving " ++ toString (-x) ++ "m south"

theorem negative_movement_south : movement_interpretation (-50) = "moving 50m south" := 
by 
  sorry

end negative_movement_south_l227_227583


namespace pictures_at_dolphin_show_l227_227684

def taken_before : Int := 28
def total_pictures_taken : Int := 44

theorem pictures_at_dolphin_show : total_pictures_taken - taken_before = 16 := by
  -- solution proof goes here
  sorry

end pictures_at_dolphin_show_l227_227684


namespace div_expr_l227_227723

namespace Proof

theorem div_expr (x : ℝ) (h : x = 3.242 * 10) : x / 100 = 0.3242 := by
  sorry

end Proof

end div_expr_l227_227723


namespace leaf_distance_after_11_gusts_l227_227403

def distance_traveled (gusts : ℕ) (swirls : ℕ) (forward_per_gust : ℕ) (backward_per_swirl : ℕ) : ℕ :=
  (gusts * forward_per_gust) - (swirls * backward_per_swirl)

theorem leaf_distance_after_11_gusts :
  ∀ (forward_per_gust backward_per_swirl : ℕ),
  forward_per_gust = 5 →
  backward_per_swirl = 2 →
  distance_traveled 11 11 forward_per_gust backward_per_swirl = 33 :=
by
  intros forward_per_gust backward_per_swirl hfg hbs
  rw [hfg, hbs]
  unfold distance_traveled
  sorry

end leaf_distance_after_11_gusts_l227_227403


namespace will_remaining_balance_l227_227570

theorem will_remaining_balance :
  ∀ (initial_money conversion_fee : ℝ) 
    (exchange_rate : ℝ)
    (sweater_cost tshirt_cost shoes_cost hat_cost socks_cost : ℝ)
    (shoes_refund_percentage : ℝ)
    (discount_percentage sales_tax_percentage : ℝ),
  initial_money = 74 →
  conversion_fee = 2 →
  exchange_rate = 1.5 →
  sweater_cost = 13.5 →
  tshirt_cost = 16.5 →
  shoes_cost = 45 →
  hat_cost = 7.5 →
  socks_cost = 6 →
  shoes_refund_percentage = 0.85 →
  discount_percentage = 0.10 →
  sales_tax_percentage = 0.05 →
  (initial_money - conversion_fee) * exchange_rate -
  ((sweater_cost + tshirt_cost + shoes_cost + hat_cost + socks_cost - shoes_cost * shoes_refund_percentage) *
   (1 - discount_percentage) * (1 + sales_tax_percentage)) /
  exchange_rate = 39.87 :=
by
  intros initial_money conversion_fee exchange_rate
        sweater_cost tshirt_cost shoes_cost hat_cost socks_cost
        shoes_refund_percentage discount_percentage sales_tax_percentage
        h1 h2 h3 h4 h5 h6 h7 h8 h9 h10
  sorry

end will_remaining_balance_l227_227570


namespace impossible_coins_l227_227236

theorem impossible_coins (p1 p2 : ℝ) (h1 : (1 - p1) * (1 - p2) = p1 * p2) (h2 : p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2) : false :=
  by sorry

end impossible_coins_l227_227236


namespace max_ab_min_inv_a_plus_4_div_b_l227_227013

theorem max_ab (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_sum : a + 4 * b = 4) : 
  ab ≤ 1 :=
by
  sorry

theorem min_inv_a_plus_4_div_b (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_sum : a + 4 * b = 4) :
  1 / a + 4 / b ≥ 25 / 4 :=
by
  sorry

end max_ab_min_inv_a_plus_4_div_b_l227_227013


namespace volume_remaining_proof_l227_227518

noncomputable def volume_remaining_part (v_original v_total_small : ℕ) : ℕ := v_original - v_total_small

def original_edge_length := 9
def small_edge_length := 3
def num_edges := 12

def volume_original := original_edge_length ^ 3
def volume_small := small_edge_length ^ 3
def volume_total_small := num_edges * volume_small

theorem volume_remaining_proof : volume_remaining_part volume_original volume_total_small = 405 := by
  sorry

end volume_remaining_proof_l227_227518


namespace minimal_colors_l227_227519

def complete_graph (n : ℕ) := Type

noncomputable def color_edges (G : complete_graph 2015) := ℕ → ℕ → ℕ

theorem minimal_colors (G : complete_graph 2015) (color : color_edges G) :
  (∀ {u v w : ℕ} (h1 : u ≠ v) (h2 : v ≠ w) (h3 : w ≠ u), color u v ≠ color v w ∧ color u v ≠ color u w ∧ color u w ≠ color v w) →
  ∃ C: ℕ, C = 2015 := 
sorry

end minimal_colors_l227_227519


namespace system_of_equations_property_l227_227171

theorem system_of_equations_property (a x y : ℝ)
  (h1 : x + y = 1 - a)
  (h2 : x - y = 3 * a + 5)
  (h3 : 0 < x)
  (h4 : 0 ≤ y) :
  (a = -5 / 3 → x = y) ∧ 
  (a = -2 → x + y = 5 + a) ∧ 
  (0 < x ∧ x ≤ 1 → 2 ≤ y ∧ y < 4) := 
by
  sorry

end system_of_equations_property_l227_227171


namespace doug_initial_marbles_l227_227891

theorem doug_initial_marbles 
  (ed_marbles : ℕ)
  (doug_marbles : ℕ)
  (lost_marbles : ℕ)
  (ed_condition : ed_marbles = doug_marbles + 5)
  (lost_condition : lost_marbles = 3)
  (ed_value : ed_marbles = 27) :
  doug_marbles + lost_marbles = 25 :=
by
  sorry

end doug_initial_marbles_l227_227891


namespace find_b1_over_b2_l227_227722

variable {a b k a1 a2 b1 b2 : ℝ}

-- Assuming a is inversely proportional to b
def inversely_proportional (a b : ℝ) (k : ℝ) : Prop :=
  a * b = k

-- Define that a_1 and a_2 are nonzero and their ratio is 3/4
def a1_a2_ratio (a1 a2 : ℝ) (ratio : ℝ) : Prop :=
  a1 / a2 = ratio

-- Define that b_1 and b_2 are nonzero
def nonzero (x : ℝ) : Prop :=
  x ≠ 0

theorem find_b1_over_b2 (a1 a2 b1 b2 : ℝ) (h1 : inversely_proportional a b k)
  (h2 : a1_a2_ratio a1 a2 (3 / 4))
  (h3 : nonzero a1) (h4 : nonzero a2) (h5 : nonzero b1) (h6 : nonzero b2) :
  b1 / b2 = 4 / 3 := 
sorry

end find_b1_over_b2_l227_227722


namespace bleaching_takes_3_hours_l227_227670

-- Define the total time and the relationship between dyeing and bleaching.
def total_time : ℕ := 9
def dyeing_takes_twice (H : ℕ) : Prop := 2 * H + H = total_time

-- Prove that bleaching takes 3 hours.
theorem bleaching_takes_3_hours : ∃ H : ℕ, dyeing_takes_twice H ∧ H = 3 := 
by 
  sorry

end bleaching_takes_3_hours_l227_227670


namespace moving_circle_trajectory_l227_227342

theorem moving_circle_trajectory (x y : ℝ) 
  (fixed_circle : x^2 + y^2 = 4): 
  (x^2 + y^2 = 9) ∨ (x^2 + y^2 = 1) :=
sorry

end moving_circle_trajectory_l227_227342


namespace hyperbola_asymptotes_l227_227262

theorem hyperbola_asymptotes (x y : ℝ) : 
  (x^2 / 4 - y^2 = 1) → (y = x / 2 ∨ y = -x / 2) :=
by
  sorry

end hyperbola_asymptotes_l227_227262


namespace determine_m_for_divisibility_by_11_l227_227558

def is_divisible_by_11 (n : ℤ) : Prop :=
  n % 11 = 0

def sum_digits_odd_pos : ℤ :=
  8 + 6 + 2 + 8

def sum_digits_even_pos (m : ℤ) : ℤ :=
  5 + m + 4

theorem determine_m_for_divisibility_by_11 :
  ∃ m : ℤ, is_divisible_by_11 (sum_digits_odd_pos - sum_digits_even_pos m) ∧ m = 4 := 
by
  sorry

end determine_m_for_divisibility_by_11_l227_227558


namespace LaurynCompanyEmployees_l227_227696

noncomputable def LaurynTotalEmployees (men women total : ℕ) : Prop :=
  men = 80 ∧ women = men + 20 ∧ total = men + women

theorem LaurynCompanyEmployees : ∃ total, ∀ men women, LaurynTotalEmployees men women total → total = 180 :=
by 
  sorry

end LaurynCompanyEmployees_l227_227696


namespace find_two_numbers_l227_227428

noncomputable def x := 5 + 2 * Real.sqrt 5
noncomputable def y := 5 - 2 * Real.sqrt 5

theorem find_two_numbers :
  (x * y = 5) ∧ (x + y = 10) :=
by {
  sorry
}

end find_two_numbers_l227_227428


namespace compare_quadratics_maximize_rectangle_area_l227_227422

-- (Ⅰ) Problem statement for comparing quadratic expressions
theorem compare_quadratics (x : ℝ) : (x + 1) * (x - 3) > (x + 2) * (x - 4) := by
  sorry

-- (Ⅱ) Problem statement for maximizing rectangular area with given perimeter
theorem maximize_rectangle_area (x y : ℝ) (h : 2 * (x + y) = 36) : 
  x = 9 ∧ y = 9 ∧ x * y = 81 := by
  sorry

end compare_quadratics_maximize_rectangle_area_l227_227422


namespace points_in_quadrant_I_l227_227381

theorem points_in_quadrant_I (x y : ℝ) :
  (y > 3 * x) ∧ (y > 5 - 2 * x) → (x > 0) ∧ (y > 0) := by
  sorry

end points_in_quadrant_I_l227_227381


namespace amount_b_l227_227933

-- Definitions of the conditions
variables (a b : ℚ) 

def condition1 : Prop := a + b = 1210
def condition2 : Prop := (2 / 3) * a = (1 / 2) * b

-- The theorem to prove
theorem amount_b (h₁ : condition1 a b) (h₂ : condition2 a b) : b = 691.43 :=
sorry

end amount_b_l227_227933


namespace simple_interest_amount_is_58_l227_227312

noncomputable def principal (CI : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  CI / ((1 + r / 100)^t - 1)

noncomputable def simple_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P * r * t / 100

theorem simple_interest_amount_is_58 (CI : ℝ) (r : ℝ) (t : ℝ) (P : ℝ) :
  CI = 59.45 -> r = 5 -> t = 2 -> P = principal CI r t ->
  simple_interest P r t = 58 :=
by
  sorry

end simple_interest_amount_is_58_l227_227312


namespace cost_of_schools_renovation_plans_and_min_funding_l227_227930

-- Define costs of Type A and Type B schools
def cost_A : ℝ := 60
def cost_B : ℝ := 85

-- Initial conditions given in the problem
axiom initial_condition_1 : cost_A + 2 * cost_B = 230
axiom initial_condition_2 : 2 * cost_A + cost_B = 205

-- Variables for number of Type A and Type B schools to renovate
variables (x : ℕ) (y : ℕ)
-- Total schools to renovate
axiom total_schools : x + y = 6

-- National and local finance constraints
axiom national_finance_max : 60 * x + 85 * y ≤ 380
axiom local_finance_min : 10 * x + 15 * y ≥ 70

-- Proving the cost of one Type A and one Type B school
theorem cost_of_schools : cost_A = 60 ∧ cost_B = 85 := 
by {
  sorry
}

-- Proving the number of renovation plans and the least funding plan
theorem renovation_plans_and_min_funding :
  ∃ x y, (x + y = 6) ∧ 
         (10 * x + 15 * y ≥ 70) ∧ 
         (60 * x + 85 * y ≤ 380) ∧ 
         (x = 2 ∧ y = 4 ∨ x = 3 ∧ y = 3 ∨ x = 4 ∧ y = 2) ∧ 
         (∀ (a b : ℕ), (a + b = 6) ∧ 
                       (10 * a + 15 * b ≥ 70) ∧ 
                       (60 * a + 85 * b ≤ 380) → 
                       60 * a + 85 * b ≥ 410) :=
by {
  sorry
}

end cost_of_schools_renovation_plans_and_min_funding_l227_227930


namespace shark_sightings_in_cape_may_l227_227693

theorem shark_sightings_in_cape_may (x : ℕ) (hx : x + (2 * x - 8) = 40) : 2 * x - 8 = 24 := 
by 
  sorry

end shark_sightings_in_cape_may_l227_227693


namespace circular_park_diameter_factor_l227_227820

theorem circular_park_diameter_factor (r : ℝ) :
  (π * (3 * r)^2) / (π * r^2) = 9 ∧ (2 * π * (3 * r)) / (2 * π * r) = 3 :=
by
  sorry

end circular_park_diameter_factor_l227_227820


namespace lily_pads_cover_entire_lake_l227_227775

/-- 
If a patch of lily pads doubles in size every day and takes 57 days to cover half the lake,
then it will take 58 days to cover the entire lake.
-/
theorem lily_pads_cover_entire_lake (days_to_half : ℕ) (h : days_to_half = 57) : (days_to_half + 1 = 58) := by
  sorry

end lily_pads_cover_entire_lake_l227_227775


namespace num_arithmetic_sequences_l227_227125

-- Definitions of the arithmetic sequence conditions
def is_arithmetic_sequence (a d n : ℕ) : Prop :=
  0 ≤ a ∧ 0 ≤ d ∧ n ≥ 3 ∧ 
  (∃ k : ℕ, k = 97 ∧ 
  (n * (2 * a + (n - 1) * d) = 2 * k ^ 2)) 

-- Prove that there are exactly 4 such sequences
theorem num_arithmetic_sequences : 
  ∃ (n : ℕ) (a d : ℕ), 
  is_arithmetic_sequence a d n ∧ 
  (n * (2 * a + (n - 1) * d) = 2 * 97^2) ∧ (
    (n = 97 ∧ ((a = 97 ∧ d = 0) ∨ (a = 49 ∧ d = 1) ∨ (a = 1 ∧ d = 2))) ∨
    (n = 97^2 ∧ a = 1 ∧ d = 0)
  ) :=
sorry

end num_arithmetic_sequences_l227_227125


namespace money_spent_on_video_games_l227_227199

theorem money_spent_on_video_games :
  let total_money := 50
  let fraction_books := 1 / 4
  let fraction_snacks := 2 / 5
  let fraction_apps := 1 / 5
  let spent_books := fraction_books * total_money
  let spent_snacks := fraction_snacks * total_money
  let spent_apps := fraction_apps * total_money
  let spent_other := spent_books + spent_snacks + spent_apps
  let spent_video_games := total_money - spent_other
  spent_video_games = 7.5 :=
by
  sorry

end money_spent_on_video_games_l227_227199


namespace minimum_g_a_l227_227339

noncomputable def f (x a : ℝ) : ℝ := x ^ 2 + 2 * a * x + 3

noncomputable def g (a : ℝ) : ℝ := 3 * a ^ 2 + 2 * a

theorem minimum_g_a : ∀ a : ℝ, a ≤ -1 → g a = 3 * a ^ 2 + 2 * a → g a ≥ 1 := by
  sorry

end minimum_g_a_l227_227339


namespace cos_shifted_eq_l227_227450

noncomputable def cos_shifted (theta : ℝ) (h1 : Real.cos theta = -12 / 13) (h2 : theta ∈ Set.Ioo Real.pi (3 / 2 * Real.pi)) : Real :=
  Real.cos (theta + Real.pi / 4)

theorem cos_shifted_eq (theta : ℝ) (h1 : Real.cos theta = -12 / 13) (h2 : theta ∈ Set.Ioo Real.pi (3 / 2 * Real.pi)) :
  cos_shifted theta h1 h2 = -7 * Real.sqrt 2 / 26 := 
by
  sorry

end cos_shifted_eq_l227_227450


namespace monotonically_increasing_sequence_b_bounds_l227_227475

theorem monotonically_increasing_sequence_b_bounds (b : ℝ) :
  (∀ n : ℕ, 0 < n → (n + 1)^2 + b * (n + 1) > n^2 + b * n) ↔ b > -3 :=
by
  sorry

end monotonically_increasing_sequence_b_bounds_l227_227475


namespace cortney_downloads_all_files_in_2_hours_l227_227270

theorem cortney_downloads_all_files_in_2_hours :
  let speed := 2 -- internet speed in megabits per minute
  let file1 := 80 -- file size in megabits
  let file2 := 90 -- file size in megabits
  let file3 := 70 -- file size in megabits
  let time1 := file1 / speed -- time to download first file in minutes
  let time2 := file2 / speed -- time to download second file in minutes
  let time3 := file3 / speed -- time to download third file in minutes
  let total_time_minutes := time1 + time2 + time3
  let total_time_hours := total_time_minutes / 60
  total_time_hours = 2 :=
by
  sorry

end cortney_downloads_all_files_in_2_hours_l227_227270


namespace meena_work_days_l227_227764

theorem meena_work_days (M : ℝ) : 1/5 + 1/M = 3/10 → M = 10 :=
by
  sorry

end meena_work_days_l227_227764


namespace unique_triple_l227_227796

theorem unique_triple (x y z : ℤ) (h₁ : x + y = z) (h₂ : y + z = x) (h₃ : z + x = y) :
  (x = 0) ∧ (y = 0) ∧ (z = 0) :=
sorry

end unique_triple_l227_227796


namespace repeating_decimal_fraction_form_l227_227687

noncomputable def repeating_decimal_rational := 2.71717171

theorem repeating_decimal_fraction_form : 
  repeating_decimal_rational = 269 / 99 ∧ (269 + 99 = 368) := 
by 
  sorry

end repeating_decimal_fraction_form_l227_227687


namespace baked_by_brier_correct_l227_227456

def baked_by_macadams : ℕ := 20
def baked_by_flannery : ℕ := 17
def total_baked : ℕ := 55

def baked_by_brier : ℕ := total_baked - (baked_by_macadams + baked_by_flannery)

-- Theorem statement
theorem baked_by_brier_correct : baked_by_brier = 18 := 
by
  -- proof will go here 
  sorry

end baked_by_brier_correct_l227_227456


namespace distance_walked_east_l227_227318

-- Definitions for distances
def s1 : ℕ := 25   -- distance walked south
def s2 : ℕ := 20   -- distance walked east
def s3 : ℕ := 25   -- distance walked north
def final_distance : ℕ := 35   -- final distance from the starting point

-- Proof problem: Prove that the distance walked east in the final step is as expected
theorem distance_walked_east (d : Real) :
  d = Real.sqrt (final_distance ^ 2 - s2 ^ 2) :=
sorry

end distance_walked_east_l227_227318


namespace complex_multiplication_l227_227114

-- Define the imaginary unit i
def i := Complex.I

-- Define the theorem we need to prove
theorem complex_multiplication : 
  (3 - 7 * i) * (-6 + 2 * i) = -4 + 48 * i := 
by 
  -- Proof is omitted
  sorry

end complex_multiplication_l227_227114


namespace Elberta_has_23_dollars_l227_227230

theorem Elberta_has_23_dollars (GrannySmith_has : ℕ := 72)
    (Anjou_has : ℕ := GrannySmith_has / 4)
    (Elberta_has : ℕ := Anjou_has + 5) : Elberta_has = 23 :=
by
  sorry

end Elberta_has_23_dollars_l227_227230


namespace hot_drinks_prediction_at_2_deg_l227_227866

-- Definition of the regression equation as a function
def regression_equation (x : ℝ) : ℝ :=
  -2.35 * x + 147.77

-- The statement to be proved
theorem hot_drinks_prediction_at_2_deg :
  abs (regression_equation 2 - 143) < 1 :=
sorry

end hot_drinks_prediction_at_2_deg_l227_227866


namespace polynomial_rewrite_l227_227971

theorem polynomial_rewrite (d : ℤ) (h : d ≠ 0) :
  let a := 20
  let b := 18
  let c := 18
  let e := 8
  (16 * d + 15 + 18 * d^2 + 5 * d^3) + (4 * d + 3 + 3 * d^3) = a * d + b + c * d^2 + e * d^3 ∧ a + b + c + e = 64 := 
by
  sorry

end polynomial_rewrite_l227_227971


namespace gasoline_used_by_car_l227_227120

noncomputable def total_gasoline_used (gasoline_per_km : ℝ) (duration_hours : ℝ) (speed_kmh : ℝ) : ℝ :=
  gasoline_per_km * duration_hours * speed_kmh

theorem gasoline_used_by_car :
  total_gasoline_used 0.14 (2 + 0.5) 93.6 = 32.76 := sorry

end gasoline_used_by_car_l227_227120


namespace words_with_mistakes_percentage_l227_227229

theorem words_with_mistakes_percentage (n x : ℕ) 
  (h1 : (x - 1 : ℝ) / n = 0.24)
  (h2 : (x - 1 : ℝ) / (n - 1) = 0.25) :
  (x : ℝ) / n * 100 = 28 := 
by 
  sorry

end words_with_mistakes_percentage_l227_227229


namespace radius_correct_l227_227808

open Real

noncomputable def radius_of_circle
  (tangent_length : ℝ) 
  (secant_internal_segment : ℝ) 
  (tangent_secant_perpendicular : Prop) : ℝ := sorry

theorem radius_correct
  (tangent_length : ℝ) 
  (secant_internal_segment : ℝ) 
  (tangent_secant_perpendicular : Prop)
  (h1 : tangent_length = 12) 
  (h2 : secant_internal_segment = 10) 
  (h3 : tangent_secant_perpendicular) : radius_of_circle tangent_length secant_internal_segment tangent_secant_perpendicular = 13 := 
sorry

end radius_correct_l227_227808


namespace first_product_of_digits_of_98_l227_227245

theorem first_product_of_digits_of_98 : (9 * 8 = 72) :=
by simp [mul_eq_mul_right_iff] -- This will handle the basic arithmetic automatically

end first_product_of_digits_of_98_l227_227245


namespace frequency_distribution_necessary_l227_227826

/-- Definition of the necessity to use Frequency Distribution to understand 
the proportion of first-year high school students in the city whose height 
falls within a certain range -/
def necessary_for_proportion (A B C D : Prop) : Prop := D

theorem frequency_distribution_necessary (A B C D : Prop) :
  necessary_for_proportion A B C D ↔ D :=
by
  sorry

end frequency_distribution_necessary_l227_227826


namespace premium_percentage_on_shares_l227_227443

theorem premium_percentage_on_shares
    (investment : ℕ)
    (share_price : ℕ)
    (premium_percentage : ℕ)
    (dividend_percentage : ℕ)
    (total_dividend : ℕ)
    (number_of_shares : ℕ)
    (investment_eq : investment = number_of_shares * (share_price + premium_percentage))
    (dividend_eq : total_dividend = number_of_shares * (share_price * dividend_percentage / 100))
    (investment_val : investment = 14400)
    (share_price_val : share_price = 100)
    (dividend_percentage_val : dividend_percentage = 5)
    (total_dividend_val : total_dividend = 600)
    (number_of_shares_val : number_of_shares = 600 / 5) :
    premium_percentage = 20 :=
by
  sorry

end premium_percentage_on_shares_l227_227443


namespace intersection_of_sets_l227_227308

theorem intersection_of_sets :
  let A := {-2, -1, 0, 1, 2}
  let B := {x | -2 < x ∧ x ≤ 2}
  A ∩ B = {-1, 0, 1, 2} :=
by
  sorry

end intersection_of_sets_l227_227308


namespace rectangular_floor_problem_possibilities_l227_227190

theorem rectangular_floor_problem_possibilities :
  ∃ (s : Finset (ℕ × ℕ)), 
    (∀ (p : ℕ × ℕ), p ∈ s → p.2 > p.1 ∧ p.2 % 3 = 0 ∧ (p.1 - 6) * (p.2 - 6) = 36) 
    ∧ s.card = 2 := 
sorry

end rectangular_floor_problem_possibilities_l227_227190


namespace fourth_competitor_jump_l227_227753

theorem fourth_competitor_jump :
  let first_jump := 22
  let second_jump := first_jump + 1
  let third_jump := second_jump - 2
  let fourth_jump := third_jump + 3
  fourth_jump = 24 := by
  sorry

end fourth_competitor_jump_l227_227753


namespace fraction_is_one_fourth_l227_227510

theorem fraction_is_one_fourth
  (f : ℚ)
  (m : ℕ)
  (h1 : (1 / 5) ^ m * f^2 = 1 / (10 ^ 4))
  (h2 : m = 4) : f = 1 / 4 := by
  sorry

end fraction_is_one_fourth_l227_227510


namespace simplify_2A_minus_B_twoA_minusB_value_when_a_neg2_b_1_twoA_minusB_independent_of_a_l227_227066

def A (a b : ℝ) := 2 * a^2 - 5 * a * b + 3 * b
def B (a b : ℝ) := 4 * a^2 + 6 * a * b + 8 * a

theorem simplify_2A_minus_B {a b : ℝ} :
  2 * A a b - B a b = -16 * a * b + 6 * b - 8 * a :=
by
  sorry

theorem twoA_minusB_value_when_a_neg2_b_1 :
  2 * A (-2) (1) - B (-2) (1) = 54 :=
by
  sorry

theorem twoA_minusB_independent_of_a {b : ℝ} :
  (∀ a : ℝ, 2 * A a b - B a b = 6 * b - 8 * a) → b = -1 / 2 :=
by
  sorry

end simplify_2A_minus_B_twoA_minusB_value_when_a_neg2_b_1_twoA_minusB_independent_of_a_l227_227066


namespace pencil_eraser_cost_l227_227546

variable (p e : ℕ)

theorem pencil_eraser_cost
  (h1 : 15 * p + 5 * e = 125)
  (h2 : p > e)
  (h3 : p > 0)
  (h4 : e > 0) :
  p + e = 11 :=
sorry

end pencil_eraser_cost_l227_227546


namespace fixed_monthly_fee_l227_227183

theorem fixed_monthly_fee (x y : ℝ) 
  (h1 : x + 20 * y = 15.20) 
  (h2 : x + 40 * y = 25.20) : 
  x = 5.20 := 
sorry

end fixed_monthly_fee_l227_227183


namespace minimum_candies_l227_227215

variables (c z : ℕ) (total_candies : ℕ)

def remaining_red_candies := (3 * c) / 5
def remaining_green_candies := (2 * z) / 5
def remaining_total_candies := remaining_red_candies + remaining_green_candies
def red_candies_fraction := remaining_red_candies * 8 = 3 * remaining_total_candies

theorem minimum_candies (h1 : 5 * c = 2 * z) (h2 : red_candies_fraction) :
  total_candies ≥ 35 := sorry

end minimum_candies_l227_227215


namespace field_trip_seniors_l227_227951

theorem field_trip_seniors (n : ℕ) 
  (h1 : n < 300) 
  (h2 : n % 17 = 15) 
  (h3 : n % 19 = 12) : 
  n = 202 :=
  sorry

end field_trip_seniors_l227_227951


namespace find_x_l227_227545

  -- Definition of the vectors
  def a (x : ℝ) : ℝ × ℝ := (x - 1, 2)
  def b : ℝ × ℝ := (2, 1)

  -- Condition that vectors are parallel
  def are_parallel (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.2 = v1.2 * v2.1

  -- Theorem statement
  theorem find_x (x : ℝ) (h : are_parallel (a x) b) : x = 5 :=
  sorry
  
end find_x_l227_227545


namespace probability_intersection_of_diagonals_hendecagon_l227_227844

-- Definition statements expressing the given conditions and required probability

def total_diagonals (n : ℕ) : ℕ := (Nat.choose n 2) - n

def ways_to_choose_2_diagonals (n : ℕ) : ℕ := Nat.choose (total_diagonals n) 2

def ways_sets_of_intersecting_diagonals (n : ℕ) : ℕ := Nat.choose n 4

def probability_intersection_lies_inside (n : ℕ) : ℚ :=
  ways_sets_of_intersecting_diagonals n / ways_to_choose_2_diagonals n

theorem probability_intersection_of_diagonals_hendecagon :
  probability_intersection_lies_inside 11 = 165 / 473 := 
by
  sorry

end probability_intersection_of_diagonals_hendecagon_l227_227844


namespace ratio_square_areas_l227_227348

theorem ratio_square_areas (r : ℝ) (h1 : r > 0) :
  let s1 := 2 * r / Real.sqrt 5
  let area1 := (s1) ^ 2
  let h := r * Real.sqrt 3
  let s2 := r
  let area2 := (s2) ^ 2
  area1 / area2 = 4 / 5 := by
  sorry

end ratio_square_areas_l227_227348


namespace bear_population_l227_227063

theorem bear_population (black_bears white_bears brown_bears total_bears : ℕ) 
(h1 : black_bears = 60)
(h2 : white_bears = black_bears / 2)
(h3 : brown_bears = black_bears + 40) :
total_bears = black_bears + white_bears + brown_bears :=
sorry

end bear_population_l227_227063


namespace derivative_y_l227_227414

noncomputable def y (x : ℝ) : ℝ :=
  (1 / 4) * Real.log ((x - 1) / (x + 1)) - (1 / 2) * Real.arctan x

theorem derivative_y (x : ℝ) : deriv y x = 1 / (x^4 - 1) :=
  sorry

end derivative_y_l227_227414


namespace number_of_numbers_l227_227978

theorem number_of_numbers (n S : ℕ) 
  (h1 : (S + 26) / n = 15)
  (h2 : (S + 36) / n = 16)
  : n = 10 :=
sorry

end number_of_numbers_l227_227978


namespace product_of_primes_impossible_l227_227367

theorem product_of_primes_impossible (q : ℕ) (hq1 : Nat.Prime q) (hq2 : q % 2 = 1) :
  ¬ ∀ i ∈ Finset.range (q-1), ∃ p1 p2 : ℕ, Nat.Prime p1 ∧ Nat.Prime p2 ∧ (i^2 + i + q = p1 * p2) :=
sorry

end product_of_primes_impossible_l227_227367


namespace min_value_expression_l227_227263

theorem min_value_expression (a b: ℝ) (h : 2 * a + b = 1) : (a - 1) ^ 2 + (b - 1) ^ 2 = 4 / 5 :=
sorry

end min_value_expression_l227_227263


namespace circle_passing_through_points_eq_l227_227405

theorem circle_passing_through_points_eq :
  let A := (-2, 1)
  let B := (9, 3)
  let C := (1, 7)
  let center := (7/2, 2)
  let radius_sq := 125 / 4
  ∀ x y : ℝ, (x - center.1)^2 + (y - center.2)^2 = radius_sq ↔ 
    (∃ t : ℝ, (x - center.1)^2 + (y - center.2)^2 = t^2) ∧
    ∀ P : ℝ × ℝ, P = A ∨ P = B ∨ P = C → (P.1 - center.1)^2 + (P.2 - center.2)^2 = radius_sq := by sorry

end circle_passing_through_points_eq_l227_227405


namespace problem_solution_exists_l227_227327

theorem problem_solution_exists (a b n : ℕ) (p : ℕ) [hp : Fact (Nat.Prime p)]
  (h : a > 0 ∧ b > 0 ∧ n > 0 ∧ a ^ 2013 + b ^ 2013 = p ^ n) :
  ∃ k : ℕ, a = 2^k ∧ b = 2^k ∧ n = 2013 * k + 1 ∧ p = 2 := by
  sorry

end problem_solution_exists_l227_227327


namespace bicycle_parking_income_l227_227388

theorem bicycle_parking_income (x : ℝ) (y : ℝ) 
    (h1 : 0 ≤ x ∧ x ≤ 2000)
    (h2 : y = 0.5 * x + 0.8 * (2000 - x)) : 
    y = -0.3 * x + 1600 := by
  sorry

end bicycle_parking_income_l227_227388


namespace red_button_probability_l227_227766

/-
Mathematical definitions derived from the problem:
Initial setup:
- Jar A has 6 red buttons and 10 blue buttons.
- Same number of red and blue buttons are removed. Jar A retains 3/4 of original buttons.
- Calculate the final number of red buttons in Jar A and B, and determine the probability both selected buttons are red.
-/
theorem red_button_probability :
  let initial_red := 6
  let initial_blue := 10
  let total_buttons := initial_red + initial_blue
  let removal_fraction := 3 / 4
  let final_buttons := (3 / 4 : ℚ) * total_buttons
  let removed_buttons := total_buttons - final_buttons
  let removed_each_color := removed_buttons / 2
  let final_red_A := initial_red - removed_each_color
  let final_red_B := removed_each_color
  let prob_red_A := final_red_A / final_buttons
  let prob_red_B := final_red_B / removed_buttons
  prob_red_A * prob_red_B = 1 / 6 :=
by
  sorry

end red_button_probability_l227_227766


namespace number_of_quadratic_PQ_equal_to_PR_l227_227435

noncomputable def P (x : ℝ) : ℝ := (x - 1) * (x - 2) * (x - 3) * (x - 4)

def is_quadratic (Q : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, Q = λ x => a * x^2 + b * x + c

theorem number_of_quadratic_PQ_equal_to_PR :
  let possible_Qx_fwds := 4^4
  let non_quadratic_cases := 6
  possible_Qx_fwds - non_quadratic_cases = 250 :=
by
  sorry

end number_of_quadratic_PQ_equal_to_PR_l227_227435


namespace field_length_l227_227856

theorem field_length (w l: ℕ) (hw1: l = 2 * w) (hw2: 8 * 8 = 64) (hw3: 64 = l * w / 2) : l = 16 := 
by
  sorry

end field_length_l227_227856


namespace quadratic_inequality_l227_227897

noncomputable def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_inequality (a b c : ℝ) 
  (h₁ : quadratic_function a b c 1 = quadratic_function a b c 3) 
  (h₂ : quadratic_function a b c 1 > quadratic_function a b c 4) : 
  a < 0 ∧ 4 * a + b = 0 :=
by
  sorry

end quadratic_inequality_l227_227897


namespace highest_lowest_difference_l227_227437

variable (x1 x2 x3 x4 x5 x_max x_min : ℝ)

theorem highest_lowest_difference (h1 : x1 + x2 + x3 + x4 + x5 - x_max = 37.84)
                                  (h2 : x1 + x2 + x3 + x4 + x5 - x_min = 38.64):
                                  x_max - x_min = 0.8 := 
by
  sorry

end highest_lowest_difference_l227_227437


namespace determine_min_bottles_l227_227967

-- Define the capacities and constraints
def mediumBottleCapacity : ℕ := 80
def largeBottleCapacity : ℕ := 1200
def additionalBottles : ℕ := 5

-- Define the minimum number of medium-sized bottles Jasmine needs to buy
def minimumMediumBottles (mediumCapacity largeCapacity extras : ℕ) : ℕ :=
  let requiredBottles := largeCapacity / mediumCapacity
  requiredBottles

theorem determine_min_bottles :
  minimumMediumBottles mediumBottleCapacity largeBottleCapacity additionalBottles = 15 :=
by
  sorry

end determine_min_bottles_l227_227967


namespace third_side_length_is_six_l227_227738

-- Defining the lengths of the sides of the triangle
def side1 : ℕ := 2
def side2 : ℕ := 6

-- Defining that the third side is an even number between 4 and 8
def is_even (x : ℕ) : Prop := x % 2 = 0
def valid_range (x : ℕ) : Prop := 4 < x ∧ x < 8

-- Stating the theorem
theorem third_side_length_is_six (x : ℕ) (h1 : is_even x) (h2 : valid_range x) : x = 6 :=
by
  sorry

end third_side_length_is_six_l227_227738


namespace number_of_valid_M_l227_227625

def base_4_representation (M : ℕ) :=
  let c_3 := (M / 256) % 4
  let c_2 := (M / 64) % 4
  let c_1 := (M / 16) % 4
  let c_0 := M % 4
  (256 * c_3) + (64 * c_2) + (16 * c_1) + (4 * c_0)

def base_7_representation (M : ℕ) :=
  let d_3 := (M / 343) % 7
  let d_2 := (M / 49) % 7
  let d_1 := (M / 7) % 7
  let d_0 := M % 7
  (343 * d_3) + (49 * d_2) + (7 * d_1) + d_0

def valid_M (M T : ℕ) :=
  1000 ≤ M ∧ M < 10000 ∧ 
  T = base_4_representation M + base_7_representation M ∧ 
  (T % 100) = ((3 * M) % 100)

theorem number_of_valid_M : 
  ∃ n : ℕ, n = 81 ∧ ∀ M T, valid_M M T → n = (81 : ℕ) :=
sorry

end number_of_valid_M_l227_227625


namespace cost_price_of_computer_table_l227_227094

/-- The owner of a furniture shop charges 20% more than the cost price. 
    Given that the customer paid Rs. 3000 for the computer table, 
    prove that the cost price of the computer table was Rs. 2500. -/
theorem cost_price_of_computer_table (CP SP : ℝ) (h1 : SP = CP + 0.20 * CP) (h2 : SP = 3000) : CP = 2500 :=
by {
  sorry
}

end cost_price_of_computer_table_l227_227094


namespace at_least_two_inequalities_hold_l227_227905

theorem at_least_two_inequalities_hold 
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : a + b + c ≥ a * b * c) : 
  (2 / a + 3 / b + 6 / c ≥ 6 ∨ 2 / b + 3 / c + 6 / a ≥ 6) ∨ (2 / b + 3 / c + 6 / a ≥ 6 ∨ 2 / c + 3 / a + 6 / b ≥ 6) ∨ (2 / c + 3 / a + 6 / b ≥ 6 ∨ 2 / a + 3 / b + 6 / c ≥ 6) := 
sorry

end at_least_two_inequalities_hold_l227_227905


namespace delta_value_l227_227657

theorem delta_value (Δ : ℤ) : 5 * (-3) = Δ - 3 → Δ = -12 :=
by
  sorry

end delta_value_l227_227657


namespace repeating_decimal_exceeds_finite_decimal_by_l227_227051

-- Definitions based on the problem conditions
def repeating_decimal := 8 / 11
def finite_decimal := 18 / 25

-- Statement to be proved
theorem repeating_decimal_exceeds_finite_decimal_by : 
  repeating_decimal - finite_decimal = 2 / 275 :=
by
  -- Skipping the proof itself with 'sorry'
  sorry

end repeating_decimal_exceeds_finite_decimal_by_l227_227051


namespace new_volume_l227_227250

theorem new_volume (l w h : ℝ) 
  (h1 : l * w * h = 4320)
  (h2 : l * w + w * h + l * h = 852)
  (h3 : l + w + h = 52) :
  (l + 4) * (w + 4) * (h + 4) = 8624 := sorry

end new_volume_l227_227250


namespace words_lost_due_to_prohibition_l227_227926

-- Define the conditions given in the problem.
def number_of_letters := 64
def forbidden_letter := 7
def total_one_letter_words := number_of_letters
def total_two_letter_words := number_of_letters * number_of_letters

-- Define the forbidden letter loss calculation.
def one_letter_words_lost := 1
def two_letter_words_lost := number_of_letters + number_of_letters - 1

-- Define the total words lost calculation.
def total_words_lost := one_letter_words_lost + two_letter_words_lost

-- State the theorem to prove the number of words lost is 128.
theorem words_lost_due_to_prohibition : total_words_lost = 128 :=
by sorry

end words_lost_due_to_prohibition_l227_227926


namespace ten_year_old_dog_is_64_human_years_l227_227119

namespace DogYears

-- Definition of the conditions
def first_year_in_human_years : ℕ := 15
def second_year_in_human_years : ℕ := 9
def subsequent_year_in_human_years : ℕ := 5

-- Definition of the total human years for a 10-year-old dog.
def dog_age_in_human_years (dog_age : ℕ) : ℕ :=
  if dog_age = 1 then first_year_in_human_years
  else if dog_age = 2 then first_year_in_human_years + second_year_in_human_years
  else first_year_in_human_years + second_year_in_human_years + (dog_age - 2) * subsequent_year_in_human_years

-- The statement to prove
theorem ten_year_old_dog_is_64_human_years : dog_age_in_human_years 10 = 64 :=
  by
    sorry

end DogYears

end ten_year_old_dog_is_64_human_years_l227_227119


namespace households_without_car_or_bike_l227_227037

/--
In a neighborhood having 90 households, some did not have either a car or a bike.
If 16 households had both a car and a bike and 44 had a car, and
there were 35 households with a bike only.
Prove that there are 11 households that did not have either a car or a bike.
-/
theorem households_without_car_or_bike
  (total_households : ℕ)
  (both_car_and_bike : ℕ)
  (car : ℕ)
  (bike_only : ℕ)
  (H1 : total_households = 90)
  (H2 : both_car_and_bike = 16)
  (H3 : car = 44)
  (H4 : bike_only = 35) :
  ∃ N : ℕ, N = total_households - (car - both_car_and_bike + bike_only + both_car_and_bike) ∧ N = 11 :=
by {
  sorry
}

end households_without_car_or_bike_l227_227037


namespace hearty_buys_red_packages_l227_227349

-- Define the conditions
def packages_of_blue := 3
def beads_per_package := 40
def total_beads := 320

-- Calculate the number of blue beads
def blue_beads := packages_of_blue * beads_per_package

-- Calculate the number of red beads
def red_beads := total_beads - blue_beads

-- Prove that the number of red packages is 5
theorem hearty_buys_red_packages : (red_beads / beads_per_package) = 5 := by
  sorry

end hearty_buys_red_packages_l227_227349


namespace part1_part2_l227_227533

noncomputable section

def f (x : ℝ) (a : ℝ) : ℝ := abs (x + 2 * a)

theorem part1 (a : ℝ) :
  (∀ x : ℝ, -4 < x ∧ x < 4 ↔ f x a < 4 - 2 * a) →
  a = 0 := 
sorry

theorem part2 (m : ℝ) :
  (∀ x : ℝ, f x 1 - f (-2 * x) 1 ≤ x + m) →
  2 ≤ m :=
sorry

end part1_part2_l227_227533


namespace point_below_line_l227_227362

theorem point_below_line (a : ℝ) (h : 2 * a - 3 > 3) : a > 3 :=
sorry

end point_below_line_l227_227362


namespace gcd_6724_13104_l227_227433

theorem gcd_6724_13104 : Int.gcd 6724 13104 = 8 := 
sorry

end gcd_6724_13104_l227_227433


namespace find_a_l227_227077

-- Define the sets A and B based on the conditions
def A (a : ℝ) : Set ℝ := {a ^ 2, a + 1, -3}
def B (a : ℝ) : Set ℝ := {a - 3, a ^ 2 + 1, 2 * a - 1}

-- Statement: Prove that a = -1 satisfies the condition A ∩ B = {-3}
theorem find_a (a : ℝ) (h : A a ∩ B a = {-3}) : a = -1 :=
by
  sorry

end find_a_l227_227077


namespace total_people_count_l227_227571

-- Definitions based on given conditions
def Cannoneers : ℕ := 63
def Women : ℕ := 2 * Cannoneers
def Men : ℕ := 2 * Women
def TotalPeople : ℕ := Women + Men

-- Lean statement to prove
theorem total_people_count : TotalPeople = 378 := by
  -- placeholders for proof steps
  sorry

end total_people_count_l227_227571


namespace more_red_peaches_than_green_l227_227500

-- Given conditions
def red_peaches : Nat := 17
def green_peaches : Nat := 16

-- Statement to prove
theorem more_red_peaches_than_green : red_peaches - green_peaches = 1 :=
by
  sorry

end more_red_peaches_than_green_l227_227500


namespace curve_focus_x_axis_l227_227324

theorem curve_focus_x_axis : 
    (x^2 - y^2 = 1)
    ∨ (∃ a b : ℝ, a ≠ 0 ∧ a ≠ b ∧ 
        (a*x^2 + b*y^2 = 1 → False)
    )
    ∨ (∃ a b : ℝ, a ≠ 0 ∧ a ≠ b ∧ 
        (b*y^2 - a*x^2 = 1 → False)
    )
    ∨ (∃ c : ℝ, c ≠ 0 ∧ 
        (y = c*x^2 → False)
    ) :=
sorry

end curve_focus_x_axis_l227_227324


namespace cyclist_problem_l227_227082

theorem cyclist_problem (MP NP : ℝ) (h1 : NP = MP + 30) (h2 : ∀ t : ℝ, t*MP = 10*t) 
  (h3 : ∀ t : ℝ, t*NP = 10*t) 
  (h4 : ∀ t : ℝ, t*MP = 42 → t*(MP + 30) = t*42 - 1/3) : 
  MP = 180 := 
sorry

end cyclist_problem_l227_227082


namespace find_S_l227_227751

variable (R S T c : ℝ)
variable (h1 : R = c * (S^2 / T^2))
variable (c_value : c = 8)
variable (h2 : R = 2) (h3 : T = 2) (h4 : S = 1)
variable (R_new : R = 50) (T_new : T = 5)

theorem find_S : S = 12.5 := by
  sorry

end find_S_l227_227751


namespace bars_cannot_form_triangle_l227_227522

theorem bars_cannot_form_triangle 
  (a b c : ℕ) (h1 : a = 4) (h2 : b = 5) (h3 : c = 10) : 
  ¬(a + b > c ∧ a + c > b ∧ b + c > a) :=
by 
  rw [h1, h2, h3]
  sorry

end bars_cannot_form_triangle_l227_227522


namespace combination_sum_l227_227333

theorem combination_sum : Nat.choose 10 3 + Nat.choose 10 4 = 330 := 
by
  sorry

end combination_sum_l227_227333


namespace simplify_expression_l227_227976

theorem simplify_expression (a b : ℚ) (ha : a = -1) (hb : b = 1/4) : 
  (a + 2 * b) ^ 2 + (a + 2 * b) * (a - 2 * b) = 1 := 
by 
  sorry

end simplify_expression_l227_227976


namespace six_digit_number_l227_227300

noncomputable def number_of_digits (N : ℕ) : ℕ := sorry

theorem six_digit_number :
  ∀ (N : ℕ),
    (N % 2020 = 0) ∧
    (∀ a b : ℕ, (a ≠ b ∧ N / 10^a % 10 ≠ N / 10^b % 10)) ∧
    (∀ a b : ℕ, (a ≠ b) → ((N / 10^a % 10 = N / 10^b % 10) -> (N % 2020 ≠ 0))) →
    number_of_digits N = 6 :=
sorry

end six_digit_number_l227_227300


namespace Arman_worked_last_week_l227_227276

variable (H : ℕ) -- hours worked last week
variable (wage_last_week wage_this_week : ℝ)
variable (hours_this_week worked_this_week two_weeks_earning : ℝ)
variable (worked_last_week : Prop)

-- Define assumptions based on the problem conditions
def condition1 : wage_last_week = 10 := by sorry
def condition2 : wage_this_week = 10.5 := by sorry
def condition3 : hours_this_week = 40 := by sorry
def condition4 : worked_this_week = wage_this_week * hours_this_week := by sorry
def condition5 : worked_this_week = 420 := by sorry -- 10.5 * 40
def condition6 : two_weeks_earning = wage_last_week * (H : ℝ) + worked_this_week := by sorry
def condition7 : two_weeks_earning = 770 := by sorry

-- Proof statement
theorem Arman_worked_last_week : worked_last_week := by
  have h1 : wage_last_week * (H : ℝ) + worked_this_week = two_weeks_earning := sorry
  have h2 : wage_last_week * (H : ℝ) + 420 = 770 := sorry
  have h3 : wage_last_week * (H : ℝ) = 350 := sorry
  have h4 : (10 : ℝ) * (H : ℝ) = 350 := sorry
  have h5 : H = 35 := sorry
  sorry

end Arman_worked_last_week_l227_227276


namespace problem_statement_l227_227023

noncomputable def omega : ℂ := sorry -- Definition placeholder for a specific nonreal root of x^4 = 1. 

theorem problem_statement (h1 : omega ^ 4 = 1) (h2 : omega ^ 2 = -1) : 
  (1 - omega + omega ^ 3) ^ 4 + (1 + omega - omega ^ 3) ^ 4 = -14 := 
sorry

end problem_statement_l227_227023


namespace value_of_k_l227_227123

theorem value_of_k (k : ℝ) : 
  (∃ p q : ℝ, p ≠ 0 ∧ q ≠ 0 ∧ p/q = 3/2 ∧ p + q = -10 ∧ p * q = k) → k = 24 :=
by 
  sorry

end value_of_k_l227_227123


namespace sufficient_conditions_for_x_squared_lt_one_l227_227332

variable (x : ℝ)

theorem sufficient_conditions_for_x_squared_lt_one :
  (∀ x, (0 < x ∧ x < 1) → (x^2 < 1)) ∧
  (∀ x, (-1 < x ∧ x < 0) → (x^2 < 1)) ∧
  (∀ x, (-1 < x ∧ x < 1) → (x^2 < 1)) :=
by
  sorry

end sufficient_conditions_for_x_squared_lt_one_l227_227332


namespace penny_exceeded_by_32_l227_227235

def bulk_price : ℤ := 5
def min_spend_before_tax : ℤ := 40
def tax_per_pound : ℤ := 1
def penny_payment : ℤ := 240

def total_cost_per_pound : ℤ := bulk_price + tax_per_pound

def min_pounds_for_min_spend : ℤ := min_spend_before_tax / bulk_price

def total_pounds_penny_bought : ℤ := penny_payment / total_cost_per_pound

def pounds_exceeded : ℤ := total_pounds_penny_bought - min_pounds_for_min_spend

theorem penny_exceeded_by_32 : pounds_exceeded = 32 := by
  sorry

end penny_exceeded_by_32_l227_227235


namespace number_of_rolls_l227_227369

theorem number_of_rolls (p : ℚ) (h : p = 1 / 9) : (2 : ℕ) = 2 :=
by 
  have h1 : 2 = 2 := rfl
  exact h1

end number_of_rolls_l227_227369


namespace equation_of_lamps_l227_227288

theorem equation_of_lamps (n k : ℕ) (N M : ℕ) (h1 : 0 < n) (h2 : 0 < k) (h3 : k ≥ n) (h4 : (k - n) % 2 = 0) : 
  N = 2^(k - n) * M := 
sorry

end equation_of_lamps_l227_227288


namespace glove_pair_probability_l227_227410

/-- 
A box contains 6 pairs of black gloves (i.e., 12 black gloves) and 4 pairs of beige gloves (i.e., 8 beige gloves).
We need to prove that the probability of drawing a matching pair of gloves is 47/95.
-/
theorem glove_pair_probability : 
  let total_gloves := 20
  let black_gloves := 12
  let beige_gloves := 8
  let P1_black := (black_gloves / total_gloves) * ((black_gloves - 1) / (total_gloves - 1))
  let P2_beige := (beige_gloves / total_gloves) * ((beige_gloves - 1) / (total_gloves - 1))
  let total_probability := P1_black + P2_beige
  total_probability = 47 / 95 :=
sorry

end glove_pair_probability_l227_227410


namespace purely_imaginary_m_no_m_in_fourth_quadrant_l227_227065

def z (m : ℝ) : ℂ := ⟨m^2 - 8 * m + 15, m^2 - 5 * m⟩

theorem purely_imaginary_m :
  (∀ m : ℝ, z m = ⟨0, m^2 - 5 * m⟩ ↔ m = 3) :=
by
  sorry

theorem no_m_in_fourth_quadrant :
  ¬ ∃ m : ℝ, (m^2 - 8 * m + 15 > 0) ∧ (m^2 - 5 * m < 0) :=
by
  sorry

end purely_imaginary_m_no_m_in_fourth_quadrant_l227_227065


namespace train_passes_jogger_in_37_seconds_l227_227862

-- Define the parameters
def jogger_speed_kmph : ℝ := 9
def train_speed_kmph : ℝ := 45
def headstart : ℝ := 250
def train_length : ℝ := 120

-- Convert speeds from km/h to m/s
noncomputable def jogger_speed_mps : ℝ := jogger_speed_kmph * 1000 / 3600
noncomputable def train_speed_mps : ℝ := train_speed_kmph * 1000 / 3600

-- Calculate relative speed in m/s
noncomputable def relative_speed : ℝ :=
  train_speed_mps - jogger_speed_mps

-- Calculate total distance to be covered in meters
def total_distance : ℝ :=
  headstart + train_length

-- Calculate time taken to pass the jogger in seconds
noncomputable def time_to_pass : ℝ :=
  total_distance / relative_speed

theorem train_passes_jogger_in_37_seconds :
  time_to_pass = 37 :=
by
  -- Proof would be here
  sorry

end train_passes_jogger_in_37_seconds_l227_227862


namespace two_pi_irrational_l227_227671

-- Assuming \(\pi\) is irrational as is commonly accepted
def irrational (x : ℝ) : Prop := ¬ (∃ a b : ℤ, b ≠ 0 ∧ x = a / b)

theorem two_pi_irrational : irrational (2 * Real.pi) := 
by 
  sorry

end two_pi_irrational_l227_227671


namespace smallest_k_674_l227_227497

theorem smallest_k_674 :
  ∀ (S : Finset ℕ), (S ⊆ Finset.range 2017) → (S.card = 674) → 
  ∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧ (672 < a - b) ∧ (a - b < 1344) ∨ (672 < b - a) ∧ (b - a < 1344) :=
by sorry

end smallest_k_674_l227_227497


namespace isosceles_triangle_l227_227683

-- Given: sides a, b, c of a triangle satisfying a specific condition
-- To Prove: the triangle is isosceles (has at least two equal sides)

theorem isosceles_triangle (a b c : ℝ)
  (h : (c - b) / a + (a - c) / b + (b - a) / c = 0) :
  (a = b ∨ b = c ∨ a = c) :=
sorry

end isosceles_triangle_l227_227683


namespace lcm_140_225_is_6300_l227_227320

def lcm_140_225 : ℕ := Nat.lcm 140 225

theorem lcm_140_225_is_6300 : lcm_140_225 = 6300 :=
by
  sorry

end lcm_140_225_is_6300_l227_227320


namespace symmetric_coords_l227_227117

-- Define the initial point and the line equation
def initial_point : ℝ × ℝ := (-1, 1)
def line_eq (x y : ℝ) : Prop := x - y - 1 = 0

-- Define what it means for one point to be symmetric to another point with respect to a line
def symmetric_point (p q : ℝ × ℝ) : Prop :=
  ∃ (m : ℝ), line_eq m p.1 ∧ line_eq m q.1 ∧ 
             p.1 + q.1 = 2 * m ∧
             p.2 + q.2 = 2 * m

-- The theorem we want to prove
theorem symmetric_coords : ∃ (symmetric : ℝ × ℝ), symmetric_point initial_point symmetric ∧ symmetric = (2, -2) :=
sorry

end symmetric_coords_l227_227117


namespace min_value_of_f_l227_227011

noncomputable def f (x : ℝ) : ℝ := |x - 1| + |x - 3| + Real.exp x

theorem min_value_of_f :
  ∃ x ∈ Set.Icc (Real.exp 0) (Real.exp 3), f x = 6 - 2 * Real.log 2 :=
sorry

end min_value_of_f_l227_227011


namespace minimum_value_expression_l227_227756

theorem minimum_value_expression (x : ℝ) (h : -3 < x ∧ x < 2) :
  ∃ y, y = (x^2 + 4 * x + 5) / (2 * x + 6) ∧ y = 3 / 4 :=
by
  sorry

end minimum_value_expression_l227_227756


namespace symmetric_line_proof_l227_227804

-- Define the given lines
def line_l (x y : ℝ) : Prop := 3 * x - 4 * y + 5 = 0
def axis_of_symmetry (x y : ℝ) : Prop := x + y = 0

-- Define the final symmetric line to be proved
def symmetric_line (x y : ℝ) : Prop := 4 * x - 3 * y - 5 = 0

-- State the theorem
theorem symmetric_line_proof (x y : ℝ) : 
  (line_l (-y) (-x)) → 
  axis_of_symmetry x y → 
  symmetric_line x y := 
sorry

end symmetric_line_proof_l227_227804


namespace sum_of_series_is_correct_l227_227160

noncomputable def geometric_series_sum_5_terms : ℚ :=
  let a := 1 / 4
  let r := 1 / 4
  let n := 5
  a * (1 - r^n) / (1 - r)

theorem sum_of_series_is_correct :
  geometric_series_sum_5_terms = 1023 / 3072 := by
  sorry

end sum_of_series_is_correct_l227_227160


namespace work_completion_days_l227_227765

theorem work_completion_days (a b c : ℝ) :
  (1/a) = 1/90 → (1/b) = 1/45 → (1/a + 1/b + 1/c) = 1/5 → c = 6 :=
by
  intros ha hb habc
  sorry

end work_completion_days_l227_227765


namespace book_total_pages_l227_227512

theorem book_total_pages (x : ℕ) (h1 : x * (3 / 5) * (3 / 8) = 36) : x = 120 := 
by
  -- Proof should be supplied here, but we only need the statement
  sorry

end book_total_pages_l227_227512


namespace joe_average_score_l227_227700

theorem joe_average_score (A B C : ℕ) (lowest_score : ℕ) (final_average : ℕ) :
  lowest_score = 45 ∧ final_average = 65 ∧ (A + B + C) / 3 = final_average →
  (A + B + C + lowest_score) / 4 = 60 := by
  sorry

end joe_average_score_l227_227700


namespace total_earnings_correct_l227_227319

-- Define the weekly earnings and the duration of the harvest.
def weekly_earnings : ℕ := 16
def harvest_duration : ℕ := 76

-- Theorems to state the problem requiring a proof.
theorem total_earnings_correct : (weekly_earnings * harvest_duration = 1216) := 
by
  sorry -- Proof is not required.

end total_earnings_correct_l227_227319


namespace john_weekly_earnings_increase_l227_227676

theorem john_weekly_earnings_increase (original_earnings new_earnings : ℕ) 
  (h₀ : original_earnings = 60) 
  (h₁ : new_earnings = 72) : 
  ((new_earnings - original_earnings) / original_earnings) * 100 = 20 :=
by
  sorry

end john_weekly_earnings_increase_l227_227676


namespace candies_count_l227_227439

theorem candies_count (x : ℚ) (h : x + 3 * x + 12 * x + 72 * x = 468) : x = 117 / 22 :=
by
  sorry

end candies_count_l227_227439


namespace general_formula_constant_c_value_l227_227883

-- Definition of an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) := ∀ n : ℕ, a (n + 1) - a n = d

-- Given sequence {a_n}
variables {a : ℕ → ℝ} (S : ℕ → ℝ) (d : ℝ)
-- Conditions
variables (h1 : a 3 * a 4 = 117) (h2 : a 2 + a 5 = 22) (hd_pos : d > 0)
-- Proof that the general formula for the sequence {a_n} is a_n = 4n - 3
theorem general_formula :
  (∀ n, a n = 4 * n - 3) :=
sorry

-- Given new sequence {b_n}
variables (b : ℕ → ℕ → ℝ) {c : ℝ} (hc : c ≠ 0)
-- New condition that bn is an arithmetic sequence
variables (h_b1 : b 1 = S 1 / (1 + c)) (h_b2 : b 2 = S 2 / (2 + c)) (h_b3 : b 3 = S 3 / (3 + c))
-- Proof that c = -1/2 is the correct constant
theorem constant_c_value :
  (c = -1 / 2) :=
sorry

end general_formula_constant_c_value_l227_227883


namespace C_is_a_liar_l227_227965

def is_knight_or_liar (P : Prop) : Prop :=
P = true ∨ P = false

variable (A B C : Prop)

-- A, B and C can only be true (knight) or false (liar)
axiom a1 : is_knight_or_liar A
axiom a2 : is_knight_or_liar B
axiom a3 : is_knight_or_liar C

-- A says "B is a liar", meaning if A is a knight, B is a liar, and if A is a liar, B is a knight
axiom a4 : A = true → B = false
axiom a5 : A = false → B = true

-- B says "A and C are of the same type", meaning if B is a knight, A and C are of the same type, otherwise they are not
axiom a6 : B = true → (A = C)
axiom a7 : B = false → (A ≠ C)

-- Prove that C is a liar
theorem C_is_a_liar : C = false :=
by
  sorry

end C_is_a_liar_l227_227965


namespace incorrect_conclusion_l227_227902

theorem incorrect_conclusion (p q : ℝ) (h1 : p < 0) (h2 : q < 0) : ¬ ∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x1 ∧ (x1 * |x1| + p * x1 + q = 0) ∧ (x2 * |x2| + p * x2 + q = 0) ∧ (x3 * |x3| + p * x3 + q = 0) :=
by
  sorry

end incorrect_conclusion_l227_227902


namespace calculate_expression_l227_227651

theorem calculate_expression :
  |(-1 : ℝ)| + Real.sqrt 9 - (1 - Real.sqrt 3)^0 - (1/2)^(-1 : ℝ) = 1 :=
by
  sorry

end calculate_expression_l227_227651


namespace remainder_of_x50_div_x_plus_1_cubed_l227_227635

theorem remainder_of_x50_div_x_plus_1_cubed (x : ℚ) : 
  (x ^ 50) % ((x + 1) ^ 3) = 1225 * x ^ 2 + 2450 * x + 1176 :=
by sorry

end remainder_of_x50_div_x_plus_1_cubed_l227_227635


namespace five_letter_words_with_one_consonant_l227_227493

theorem five_letter_words_with_one_consonant :
  let letters := ['A', 'B', 'C', 'D', 'E', 'F']
  let vowels := ['A', 'E']
  let consonants := ['B', 'C', 'D', 'F']
  let total_words := (letters.length : ℕ)^5
  let vowel_only_words := (vowels.length : ℕ)^5
  total_words - vowel_only_words = 7744 :=
by
  sorry

end five_letter_words_with_one_consonant_l227_227493


namespace area_transformed_region_l227_227974

-- Define the transformation matrix
def matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![3, 1], ![4, 3]]

-- Define the area of region T
def area_T := 6

-- The statement we want to prove: the area of T' is 30.
theorem area_transformed_region :
  let det := matrix.det
  area_T * det = 30 :=
by
  sorry

end area_transformed_region_l227_227974


namespace exists_root_between_l227_227903

-- Given definitions and conditions
variables (a b c : ℝ)
variables (ha : a ≠ 0)
variables (x1 x2 : ℝ)
variable (h1 : a * x1^2 + b * x1 + c = 0)    -- root of the first equation
variable (h2 : -a * x2^2 + b * x2 + c = 0)   -- root of the second equation

-- Proof statement
theorem exists_root_between (a b c : ℝ) (ha : a ≠ 0) (x1 x2 : ℝ)
    (h1 : a * x1^2 + b * x1 + c = 0) (h2 : -a * x2^2 + b * x2 + c = 0) :
    ∃ x3 : ℝ, 
      (x1 ≤ x3 ∧ x3 ≤ x2 ∨ x1 ≥ x3 ∧ x3 ≥ x2) ∧ 
      (1 / 2 * a * x3^2 + b * x3 + c = 0) :=
sorry

end exists_root_between_l227_227903


namespace divisible_by_120_l227_227226

theorem divisible_by_120 (n : ℕ) : 120 ∣ n * (n^2 - 1) * (n^2 - 5 * n + 26) := sorry

end divisible_by_120_l227_227226


namespace tan_neg4095_eq_one_l227_227569

theorem tan_neg4095_eq_one : Real.tan (Real.pi / 180 * -4095) = 1 := by
  sorry

end tan_neg4095_eq_one_l227_227569


namespace swimming_pool_water_remaining_l227_227636

theorem swimming_pool_water_remaining :
  let initial_water := 500 -- initial water in gallons
  let evaporation_rate := 1.5 -- water loss due to evaporation in gallons/day
  let leak_rate := 0.8 -- water loss due to leak in gallons/day
  let total_days := 20 -- total number of days

  let total_daily_loss := evaporation_rate + leak_rate -- total daily loss in gallons/day
  let total_loss := total_daily_loss * total_days -- total loss over the period in gallons
  let remaining_water := initial_water - total_loss -- remaining water after 20 days in gallons

  remaining_water = 454 :=
by
  sorry

end swimming_pool_water_remaining_l227_227636


namespace average_weight_l227_227472

/-- 
Given the following conditions:
1. (A + B) / 2 = 40
2. (B + C) / 2 = 41
3. B = 27
Prove that the average weight of a, b, and c is 45 kg.
-/
theorem average_weight (A B C : ℝ) 
  (h1 : (A + B) / 2 = 40)
  (h2 : (B + C) / 2 = 41)
  (h3 : B = 27): 
  (A + B + C) / 3 = 45 :=
by
  sorry

end average_weight_l227_227472


namespace simple_interest_sum_l227_227836

theorem simple_interest_sum :
  let P := 1750
  let CI := 4000 * ((1 + (10 / 100))^2) - 4000
  let SI := (1 / 2) * CI
  SI = (P * 8 * 3) / 100 
  :=
by
  -- Definitions
  let P := 1750
  let CI := 4000 * ((1 + 10 / 100)^2) - 4000
  let SI := (1 / 2) * CI
  
  -- Claim
  have : SI = (P * 8 * 3) / 100 := sorry

  exact this

end simple_interest_sum_l227_227836


namespace price_of_soda_l227_227329

theorem price_of_soda (regular_price_per_can : ℝ) (case_discount : ℝ) (bulk_discount : ℝ) (num_cases : ℕ) (num_cans : ℕ) :
  regular_price_per_can = 0.15 →
  case_discount = 0.12 →
  bulk_discount = 0.05 →
  num_cases = 3 →
  num_cans = 75 →
  (num_cans * ((regular_price_per_can * (1 - case_discount)) * (1 - bulk_discount))) = 9.405 :=
by
  intros h1 h2 h3 h4 h5
  -- normal price per can
  have hp1 : ℝ := regular_price_per_can
  -- price after case discount
  have hp2 : ℝ := hp1 * (1 - case_discount)
  -- price after bulk discount
  have hp3 : ℝ := hp2 * (1 - bulk_discount)
  -- total price
  have total_price : ℝ := num_cans * hp3
  -- goal
  sorry -- skip the proof, as only the statement is needed.

end price_of_soda_l227_227329


namespace min_value_fraction_l227_227059

theorem min_value_fraction (x y : ℝ) (hx : x > -1) (hy : y > 0) (hxy : x + 2 * y = 1) : 
  ∃ m, (∀ z, z = (1 / (x + 1) + 1 / y) → z ≥ m) ∧ m = (3 + 2 * Real.sqrt 2) / 2 :=
by
  sorry

end min_value_fraction_l227_227059


namespace q_is_20_percent_less_than_p_l227_227024

theorem q_is_20_percent_less_than_p (p q : ℝ) (h : p = 1.25 * q) : (q - p) / p * 100 = -20 := by
  sorry

end q_is_20_percent_less_than_p_l227_227024


namespace consecutive_page_sum_l227_227406

theorem consecutive_page_sum (n : ℕ) (h : n * (n + 1) * (n + 2) = 479160) : n + (n + 1) + (n + 2) = 234 :=
sorry

end consecutive_page_sum_l227_227406


namespace good_numbers_correct_l227_227137

noncomputable def good_numbers (n : ℕ) : ℝ :=
  1 / 2 * (8^n + 10^n) - 1

theorem good_numbers_correct (n : ℕ) : good_numbers n = 
  1 / 2 * (8^n + 10^n) - 1 := 
sorry

end good_numbers_correct_l227_227137


namespace arithmetic_geometric_sequence_problem_l227_227466

variable {n : ℕ}

def a (n : ℕ) : ℕ := 3 * n - 1
def b (n : ℕ) : ℕ := 2 ^ n
def S (n : ℕ) : ℕ := n * (2 + (2 + (n - 1) * (3 - 1))) / 2 -- sum of an arithmetic sequence
def T (n : ℕ) : ℕ := (3 * n - 4) * 2 ^ (n + 1) + 8

theorem arithmetic_geometric_sequence_problem :
  (a 1 = 2) ∧ (b 1 = 2) ∧ (a 4 + b 4 = 27) ∧ (S 4 - b 4 = 10) →
  (∀ n, T n = (3 * n - 4) * 2 ^ (n + 1) + 8) := sorry

end arithmetic_geometric_sequence_problem_l227_227466


namespace probability_correct_l227_227643

noncomputable def probability_B1_eq_5_given_WB : ℚ :=
  let P_B1_eq_5 : ℚ := 1 / 8
  let P_WB : ℚ := 1 / 5
  let P_WB_given_B1_eq_5 : ℚ := 1 / 16 + 369 / 2048
  (P_B1_eq_5 * P_WB_given_B1_eq_5) / P_WB

theorem probability_correct :
  probability_B1_eq_5_given_WB = 115 / 1024 :=
by
  sorry

end probability_correct_l227_227643


namespace last_four_digits_of_m_smallest_l227_227352

theorem last_four_digits_of_m_smallest (m : ℕ) (h1 : m > 0)
  (h2 : m % 6 = 0) (h3 : m % 8 = 0)
  (h4 : ∀ d, d ∈ (m.digits 10) → d = 2 ∨ d = 7)
  (h5 : 2 ∈ (m.digits 10)) (h6 : 7 ∈ (m.digits 10)) :
  (m % 10000) = 2722 :=
sorry

end last_four_digits_of_m_smallest_l227_227352


namespace at_least_one_not_less_than_two_l227_227777

theorem at_least_one_not_less_than_two
  (x y z : ℝ)
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  let a := x + 1 / y
  let b := y + 1 / z
  let c := z + 1 / x
  a >= 2 ∨ b >= 2 ∨ c >= 2 := 
sorry

end at_least_one_not_less_than_two_l227_227777


namespace sum_of_numbers_l227_227889

theorem sum_of_numbers (avg : ℝ) (num : ℕ) (h1 : avg = 5.2) (h2 : num = 8) : 
  (avg * num = 41.6) :=
by
  sorry

end sum_of_numbers_l227_227889


namespace perpendicular_vectors_vector_sum_norm_min_value_f_l227_227010

noncomputable def a (x : ℝ) : ℝ × ℝ :=
  (Real.cos (3*x/2), Real.sin (3*x/2))

noncomputable def b (x : ℝ) : ℝ × ℝ :=
  (Real.cos (x/2), -Real.sin (x/2))

noncomputable def f (x m : ℝ) : ℝ :=
  (a x).1 * (b x).1 + (a x).2 * (b x).2 - 2 * m * Real.sqrt ((a x).1 + (b x).1)^2 + ((a x).2 + (b x).2)^2

theorem perpendicular_vectors (x : ℝ) (h : 0 ≤ x ∧ x ≤ Real.pi / 2) :
  (a x).1 * (b x).1 + (a x).2 * (b x).2 = 0 ↔ x = Real.pi / 4 := sorry

theorem vector_sum_norm (x : ℝ) (h : 0 ≤ x ∧ x ≤ Real.pi / 2) :
  Real.sqrt ((a x).1 + (b x).1)^2 + ((a x).2 + (b x).2)^2 ≥ 1 ↔ 0 ≤ x ∧ x ≤ Real.pi / 3 := sorry

theorem min_value_f (m : ℝ) :
  (∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 → f x m ≥ -2) ↔ m = Real.sqrt 2 / 2 := sorry

end perpendicular_vectors_vector_sum_norm_min_value_f_l227_227010


namespace age_proof_l227_227537

-- Let's define the conditions first
variable (s f : ℕ) -- s: age of the son, f: age of the father

-- Conditions derived from the problem statement
def son_age_condition : Prop := s = 8 - 1
def father_age_condition : Prop := f = 5 * s

-- The goal is to prove that the father's age is 35
theorem age_proof (s f : ℕ) (h₁ : son_age_condition s) (h₂ : father_age_condition s f) : f = 35 :=
by sorry

end age_proof_l227_227537


namespace big_al_bananas_l227_227534

-- Define conditions for the arithmetic sequence and total consumption
theorem big_al_bananas (a : ℕ) : 
  (a + (a + 6) + (a + 12) + (a + 18) + (a + 24) = 100) → 
  (a + 24 = 32) :=
by
  sorry

end big_al_bananas_l227_227534


namespace bracelet_pairing_impossible_l227_227737

/--
Elizabeth has 100 different bracelets, and each day she wears three of them to school. 
Prove that it is impossible for any pair of bracelets to appear together on her wrist exactly once.
-/
theorem bracelet_pairing_impossible : 
  (∃ (bracelet_set : Finset (Finset (Fin 100))), 
    (∀ (a b : Fin 100), a ≠ b → ∃ t ∈ bracelet_set, {a, b} ⊆ t) ∧ (∀ t ∈ bracelet_set, t.card = 3) ∧ (bracelet_set.card * 3 / 2 ≠ 99)) :=
sorry

end bracelet_pairing_impossible_l227_227737


namespace find_product_of_two_numbers_l227_227100

theorem find_product_of_two_numbers (a b : ℚ) (h1 : a + b = 7) (h2 : a - b = 2) : 
  a * b = 11 + 1/4 := 
by 
  sorry

end find_product_of_two_numbers_l227_227100


namespace product_of_integer_with_100_l227_227292

theorem product_of_integer_with_100 (x : ℝ) (h : 10 * x = x + 37.89) : 100 * x = 421 :=
by
  -- insert the necessary steps to solve the problem
  sorry

end product_of_integer_with_100_l227_227292


namespace n_minus_two_is_square_of_natural_number_l227_227788

theorem n_minus_two_is_square_of_natural_number (n : ℕ) (h_n : n ≥ 3) (h_odd_m : Odd (1 / 2 * n * (n - 1))) :
  ∃ k : ℕ, n - 2 = k^2 := 
  by
  sorry

end n_minus_two_is_square_of_natural_number_l227_227788


namespace walking_speed_l227_227470

-- Define the constants and variables
def speed_there := 25 -- speed from village to post-office in kmph
def total_time := 5.8 -- total round trip time in hours
def distance := 20.0 -- distance to the post-office in km
 
-- Define the theorem that needs to be proved
theorem walking_speed :
  ∃ (speed_back : ℝ), speed_back = 4 := 
by
  sorry

end walking_speed_l227_227470


namespace seating_impossible_l227_227220

theorem seating_impossible (reps : Fin 54 → Fin 27) : 
  ¬ ∃ (s : Fin 54 → Fin 54),
    (∀ i : Fin 27, ∃ a b : Fin 54, a ≠ b ∧ s a = i ∧ s b = i ∧ (b - a ≡ 10 [MOD 54] ∨ a - b ≡ 10 [MOD 54])) :=
sorry

end seating_impossible_l227_227220


namespace intersection_M_N_l227_227050

open Set

noncomputable def M : Set ℝ := {x | x ≥ 2}

noncomputable def N : Set ℝ := {x | x^2 - 6*x + 5 < 0}

theorem intersection_M_N : M ∩ N = {x | 2 ≤ x ∧ x < 5} :=
by
  sorry

end intersection_M_N_l227_227050


namespace sum_3n_terms_l227_227906

variable {a_n : ℕ → ℝ} -- Definition of the sequence
variable {S : ℕ → ℝ} -- Definition of the sum function

-- Conditions
axiom sum_n_terms (n : ℕ) : S n = 3
axiom sum_2n_terms (n : ℕ) : S (2 * n) = 15

-- Question and correct answer
theorem sum_3n_terms (n : ℕ) : S (3 * n) = 63 := 
sorry -- Proof to be provided

end sum_3n_terms_l227_227906


namespace gcd_78_182_l227_227549

theorem gcd_78_182 : Nat.gcd 78 182 = 26 := 
by
  sorry

end gcd_78_182_l227_227549


namespace chloe_paid_per_dozen_l227_227487

-- Definitions based on conditions
def half_dozen_sale_price : ℕ := 30
def profit : ℕ := 500
def dozens_sold : ℕ := 50
def full_dozen_sale_price := 2 * half_dozen_sale_price
def total_revenue := dozens_sold * full_dozen_sale_price
def total_cost := total_revenue - profit

-- Proof problem
theorem chloe_paid_per_dozen : (total_cost / dozens_sold) = 50 :=
by
  sorry

end chloe_paid_per_dozen_l227_227487


namespace part_a_solutions_l227_227279

theorem part_a_solutions (x : ℝ) : (⌊x⌋^2 - x = -0.99) ↔ (x = 0.99 ∨ x = 1.99) :=
sorry

end part_a_solutions_l227_227279


namespace zack_traveled_to_18_countries_l227_227365

-- Defining the conditions
variables (countries_traveled_by_george countries_traveled_by_joseph 
           countries_traveled_by_patrick countries_traveled_by_zack : ℕ)

-- Set the conditions as per the problem statement
axiom george_traveled : countries_traveled_by_george = 6
axiom joseph_traveled : countries_traveled_by_joseph = countries_traveled_by_george / 2
axiom patrick_traveled : countries_traveled_by_patrick = 3 * countries_traveled_by_joseph
axiom zack_traveled : countries_traveled_by_zack = 2 * countries_traveled_by_patrick

-- The theorem to prove Zack traveled to 18 countries
theorem zack_traveled_to_18_countries : countries_traveled_by_zack = 18 :=
by
  -- Adding the proof here is unnecessary as per the instructions
  sorry

end zack_traveled_to_18_countries_l227_227365


namespace abs_lt_one_sufficient_not_necessary_l227_227309

theorem abs_lt_one_sufficient_not_necessary (x : ℝ) : (|x| < 1) -> (x < 1) ∧ ¬(x < 1 -> |x| < 1) :=
by
  sorry

end abs_lt_one_sufficient_not_necessary_l227_227309


namespace quadratic_function_symmetry_l227_227527

-- Define the quadratic function
def f (x : ℝ) (b c : ℝ) : ℝ := -x^2 + b * x + c

-- State the problem as a theorem
theorem quadratic_function_symmetry (b c : ℝ) (h_symm : ∀ x, f x b c = f (4 - x) b c) :
  f 2 b c > f 1 b c ∧ f 1 b c > f 4 b c :=
by
  -- Include a placeholder for the proof
  sorry

end quadratic_function_symmetry_l227_227527


namespace find_multiple_l227_227860

theorem find_multiple (x y m : ℕ) (h1 : y + x = 50) (h2 : y = m * x - 43) (h3 : y = 31) : m = 4 :=
by
  sorry

end find_multiple_l227_227860


namespace harry_morning_routine_l227_227688

variable (t1 t2 : ℕ)
variable (h1 : t1 = 15)
variable (h2 : t2 = 2 * t1)

theorem harry_morning_routine : t1 + t2 = 45 := by
  sorry

end harry_morning_routine_l227_227688


namespace product_of_roots_l227_227401

theorem product_of_roots (p q r : ℝ) (hp : 3*p^3 - 9*p^2 + 5*p - 15 = 0) 
  (hq : 3*q^3 - 9*q^2 + 5*q - 15 = 0) (hr : 3*r^3 - 9*r^2 + 5*r - 15 = 0) :
  p * q * r = 5 :=
sorry

end product_of_roots_l227_227401


namespace g_six_l227_227947

noncomputable def g : ℝ → ℝ := sorry

axiom g_func_eq (x y : ℝ) : g (x - y) = g x * g y
axiom g_nonzero (x : ℝ) : g x ≠ 0
axiom g_double (x : ℝ) : g (2 * x) = g x ^ 2
axiom g_value : g 6 = 1

theorem g_six : g 6 = 1 := by
  exact g_value

end g_six_l227_227947


namespace intersection_points_lie_on_circle_l227_227165

variables (u x y : ℝ)

theorem intersection_points_lie_on_circle :
  (∃ u : ℝ, 3 * u - 4 * y + 2 = 0 ∧ 2 * x - 3 * u * y - 4 = 0) →
  ∃ r : ℝ, (x^2 + y^2 = r^2) :=
by 
  sorry

end intersection_points_lie_on_circle_l227_227165


namespace even_natural_number_factors_count_l227_227782

def is_valid_factor (a b c : ℕ) : Prop := 
  1 ≤ a ∧ a ≤ 3 ∧ 
  0 ≤ b ∧ b ≤ 2 ∧ 
  0 ≤ c ∧ c ≤ 2 ∧ 
  a + b + c ≤ 4

noncomputable def count_valid_factors : ℕ :=
  Nat.card { x : ℕ × ℕ × ℕ // is_valid_factor x.1 x.2.1 x.2.2 }

theorem even_natural_number_factors_count : count_valid_factors = 15 := 
  sorry

end even_natural_number_factors_count_l227_227782


namespace volume_of_rock_correct_l227_227197

-- Define the initial conditions
def tank_length := 30
def tank_width := 20
def water_depth := 8
def water_level_rise := 4

-- Define the volume function for the rise in water level
def calculate_volume_of_rise (length: ℕ) (width: ℕ) (rise: ℕ) : ℕ :=
  length * width * rise

-- Define the target volume of the rock
def volume_of_rock := 2400

-- The theorem statement that the volume of the rock is 2400 cm³
theorem volume_of_rock_correct :
  calculate_volume_of_rise tank_length tank_width water_level_rise = volume_of_rock :=
by 
  sorry

end volume_of_rock_correct_l227_227197


namespace square_floor_tile_count_l227_227090

theorem square_floor_tile_count (n : ℕ) (h1 : 2 * n - 1 = 25) : n^2 = 169 :=
by
  sorry

end square_floor_tile_count_l227_227090


namespace marks_in_english_l227_227452

theorem marks_in_english :
  let m := 35             -- Marks in Mathematics
  let p := 52             -- Marks in Physics
  let c := 47             -- Marks in Chemistry
  let b := 55             -- Marks in Biology
  let n := 5              -- Number of subjects
  let avg := 46.8         -- Average marks
  let total_marks := avg * n
  total_marks - (m + p + c + b) = 45 := sorry

end marks_in_english_l227_227452


namespace problem_statement_l227_227633

noncomputable def f (x : ℝ) : ℝ := Real.sin x

noncomputable def g (x : ℝ) : ℝ := Real.sin (3 * (x - 1))

theorem problem_statement :
  (∀ x : ℝ, f (-x) = -f (x)) ∧ (∀ x1 x2 : ℝ, x1 + x2 = π / 2 → g x1 = g x2) :=
by 
  sorry

end problem_statement_l227_227633


namespace find_second_derivative_at_1_l227_227407

-- Define the function f(x) and its second derivative
noncomputable def f (x : ℝ) := x * Real.exp x
noncomputable def f'' (x : ℝ) := (x + 2) * Real.exp x

-- State the theorem to be proved
theorem find_second_derivative_at_1 : f'' 1 = 2 * Real.exp 1 := by
  sorry

end find_second_derivative_at_1_l227_227407


namespace non_working_games_count_l227_227499

-- Definitions based on conditions
def total_games : Nat := 15
def total_earnings : Nat := 30
def price_per_game : Nat := 5

-- Definition to be proved
def working_games : Nat := total_earnings / price_per_game
def non_working_games : Nat := total_games - working_games

-- Statement to be proved
theorem non_working_games_count : non_working_games = 9 :=
by
  sorry

end non_working_games_count_l227_227499


namespace solution_set_l227_227109

variable {f : ℝ → ℝ}

-- Define that f is an odd function
def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define that f is decreasing on positive reals
def decreasing_on_pos_reals (f : ℝ → ℝ) : Prop := ∀ x y, 0 < x → x < y → f y < f x

-- Given conditions
axiom f_odd : odd_function f
axiom f_decreasing : decreasing_on_pos_reals f
axiom f_at_two_zero : f 2 = 0

-- Main theorem statement
theorem solution_set : { x : ℝ | (x - 1) * f (x - 1) > 0 } = { x | x < -1 } ∪ { x | x > 3 } :=
sorry

end solution_set_l227_227109


namespace determine_n_l227_227085

open Function

noncomputable def coeff_3 (n : ℕ) : ℕ :=
  2^(n-2) * Nat.choose n 2

noncomputable def coeff_4 (n : ℕ) : ℕ :=
  2^(n-3) * Nat.choose n 3

theorem determine_n (n : ℕ) (b3_eq_2b4 : coeff_3 n = 2 * coeff_4 n) : n = 5 :=
  sorry

end determine_n_l227_227085


namespace complex_imaginary_condition_l227_227143

theorem complex_imaginary_condition (m : ℝ) : (∀ m : ℝ, (m^2 - 3*m - 4 = 0) → (m^2 - 5*m - 6) ≠ 0) ↔ (m ≠ -1 ∧ m ≠ 6) :=
by
  sorry

end complex_imaginary_condition_l227_227143


namespace value_of_m_l227_227586

theorem value_of_m (m : ℝ) : (∃ x : ℝ, x = 2 ∧ x^2 - m * x + 8 = 0) → m = 6 := by
  sorry

end value_of_m_l227_227586


namespace gcd_204_85_l227_227703

theorem gcd_204_85 : Nat.gcd 204 85 = 17 := 
by sorry

end gcd_204_85_l227_227703


namespace maximum_mass_difference_l227_227935

theorem maximum_mass_difference (m1 m2 : ℝ) (h1 : 19.7 ≤ m1 ∧ m1 ≤ 20.3) (h2 : 19.7 ≤ m2 ∧ m2 ≤ 20.3) :
  abs (m1 - m2) ≤ 0.6 :=
by
  sorry

end maximum_mass_difference_l227_227935


namespace greatest_sum_of_other_two_roots_l227_227458

noncomputable def polynomial (x : ℝ) (k : ℝ) : ℝ :=
  x^3 - k * x^2 + 20 * x - 15

theorem greatest_sum_of_other_two_roots (k x1 x2 : ℝ) (h : polynomial 3 k = 0) (hx : x1 * x2 = 5)
  (h_prod_sum : 3 * x1 + 3 * x2 + x1 * x2 = 20) : x1 + x2 = 5 :=
by
  sorry

end greatest_sum_of_other_two_roots_l227_227458


namespace rhombus_area_l227_227642

-- Define the lengths of the diagonals
def d1 : ℝ := 6
def d2 : ℝ := 8

-- Problem statement: The area of the rhombus
theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 6) (h2 : d2 = 8) : (1 / 2) * d1 * d2 = 24 := by
  -- The proof is not required, so we use sorry.
  sorry

end rhombus_area_l227_227642


namespace problem1_problem2_l227_227075

-- Define the sets P and Q
def set_P : Set ℝ := {x | 2 * x^2 - 5 * x - 3 < 0}
def set_Q (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 1}

-- Problem (1): P ∩ Q = Q implies a ∈ (-1/2, 2)
theorem problem1 (a : ℝ) : (set_Q a) ⊆ set_P → -1/2 < a ∧ a < 2 :=
by 
  sorry

-- Problem (2): P ∩ Q = ∅ implies a ∈ (-∞, -3/2] ∪ [3, ∞)
theorem problem2 (a : ℝ) : (set_Q a) ∩ set_P = ∅ → a ≤ -3/2 ∨ a ≥ 3 :=
by 
  sorry

end problem1_problem2_l227_227075


namespace commission_8000_l227_227996

variable (C k : ℝ)

def commission_5000 (C k : ℝ) : Prop := C + 5000 * k = 110
def commission_11000 (C k : ℝ) : Prop := C + 11000 * k = 230

theorem commission_8000 
  (h1 : commission_5000 C k) 
  (h2 : commission_11000 C k)
  : C + 8000 * k = 170 :=
sorry

end commission_8000_l227_227996


namespace triangle_inequalities_l227_227834

-- Definitions of the variables
variables {ABC : Triangle} {r : ℝ} {R : ℝ} {ρ_a ρ_b ρ_c : ℝ} {P_a P_b P_c : ℝ}

-- Problem statement based on given conditions and proof requirement
theorem triangle_inequalities (ABC : Triangle) (r : ℝ) (R : ℝ) (ρ_a ρ_b ρ_c : ℝ) (P_a P_b P_c : ℝ) :
  (3/2) * r ≤ ρ_a + ρ_b + ρ_c ∧ ρ_a + ρ_b + ρ_c ≤ (3/4) * R ∧ 4 * r ≤ P_a + P_b + P_c ∧ P_a + P_b + P_c ≤ 2 * R :=
  sorry

end triangle_inequalities_l227_227834


namespace calc_expression_l227_227169

theorem calc_expression : (3^500 + 4^501)^2 - (3^500 - 4^501)^2 = 16 * 10^500 :=
  sorry

end calc_expression_l227_227169


namespace smallest_n_divides_999_l227_227486

/-- 
Given \( 1 \leq n < 1000 \), \( n \) divides 999, and \( n+6 \) divides 99,
prove that the smallest possible value of \( n \) is 27.
 -/
theorem smallest_n_divides_999 (n : ℕ) 
  (h1 : 1 ≤ n) 
  (h2 : n < 1000) 
  (h3 : n ∣ 999) 
  (h4 : n + 6 ∣ 99) : 
  n = 27 :=
  sorry

end smallest_n_divides_999_l227_227486


namespace victor_score_l227_227686

-- Definitions based on the conditions
def max_marks : ℕ := 300
def percentage : ℕ := 80

-- Statement to be proved
theorem victor_score : (percentage * max_marks) / 100 = 240 := by
  sorry

end victor_score_l227_227686


namespace serenity_total_shoes_l227_227520

def pairs_of_shoes : ℕ := 3
def shoes_per_pair : ℕ := 2

theorem serenity_total_shoes : pairs_of_shoes * shoes_per_pair = 6 := by
  sorry

end serenity_total_shoes_l227_227520


namespace room_dimension_l227_227715

theorem room_dimension {a : ℝ} (h1 : a > 0) 
  (h2 : 4 = 2^2) 
  (h3 : 14 = 2 * (7)) 
  (h4 : 2 * a = 14) :
  (a + 2 * a - 2 = 19) :=
sorry

end room_dimension_l227_227715


namespace number_of_buses_required_l227_227507

def total_seats : ℕ := 28
def students_per_bus : ℝ := 14.0

theorem number_of_buses_required :
  (total_seats / students_per_bus) = 2 := 
by
  -- The actual proof is intentionally left out.
  sorry

end number_of_buses_required_l227_227507


namespace initial_passengers_is_350_l227_227208

variable (N : ℕ)

def initial_passengers (N : ℕ) : Prop :=
  let after_first_train := 9 * N / 10
  let after_second_train := 27 * N / 35
  let after_third_train := 108 * N / 175
  after_third_train = 216

theorem initial_passengers_is_350 : initial_passengers 350 := 
  sorry

end initial_passengers_is_350_l227_227208


namespace new_average_age_l227_227064

theorem new_average_age (avg_age : ℕ) (num_students : ℕ) (teacher_age : ℕ) (new_num_individuals : ℕ) (new_avg_age : ℕ) :
  avg_age = 15 ∧ num_students = 20 ∧ teacher_age = 36 ∧ new_num_individuals = 21 →
  new_avg_age = (num_students * avg_age + teacher_age) / new_num_individuals → new_avg_age = 16 :=
by
  intros
  sorry

end new_average_age_l227_227064


namespace common_ratio_geometric_progression_l227_227178

theorem common_ratio_geometric_progression {x y z r : ℝ} (h_diff1 : x ≠ y) (h_diff2 : y ≠ z) (h_diff3 : z ≠ x)
  (hx_nonzero : x ≠ 0) (hy_nonzero : y ≠ 0) (hz_nonzero : z ≠ 0)
  (h_gm_progression : ∃ r : ℝ, x * (y - z) = x * (y - z) * r ∧ z * (x - y) = (y * (z - x)) * r) : r^2 + r + 1 = 0 :=
sorry

end common_ratio_geometric_progression_l227_227178


namespace stops_away_pinedale_mall_from_yahya_house_l227_227174

-- Definitions based on problem conditions
def bus_speed_kmh : ℕ := 60
def stop_interval_minutes : ℕ := 5
def distance_to_mall_km : ℕ := 40

-- Definition of how many stops away is Pinedale mall from Yahya's house
def stops_to_mall : ℕ := distance_to_mall_km / (bus_speed_kmh / 60 * stop_interval_minutes)

-- Lean statement to prove the given conditions lead to the correct number of stops
theorem stops_away_pinedale_mall_from_yahya_house :
  stops_to_mall = 8 :=
by 
  -- This is a placeholder for the proof. 
  -- Actual proof steps would convert units and calculate as described in the problem.
  sorry

end stops_away_pinedale_mall_from_yahya_house_l227_227174


namespace helmet_price_for_given_profit_helmet_price_for_max_profit_l227_227353

section helmet_sales

-- Define the conditions
variable (original_price : ℝ := 80) (initial_sales : ℝ := 200) (cost_price : ℝ := 50) 
variable (price_reduction_unit : ℝ := 1) (additional_sales_per_reduction : ℝ := 10)
variable (minimum_price_reduction : ℝ := 10)

-- Profits
def profit (x : ℝ) : ℝ :=
  (original_price - x - cost_price) * (initial_sales + additional_sales_per_reduction * x)

-- Prove the selling price when profit is 5250 yuan
theorem helmet_price_for_given_profit (GDP : profit 15 = 5250) : (original_price - 15) = 65 :=
by
  sorry

-- Prove the price for maximum profit
theorem helmet_price_for_max_profit : 
  ∃ x, x = 10 ∧ (original_price - x = 70) ∧ (profit x = 6000) :=
by 
  sorry

end helmet_sales

end helmet_price_for_given_profit_helmet_price_for_max_profit_l227_227353


namespace find_a_parallel_lines_l227_227016

theorem find_a_parallel_lines (a : ℝ) (l1_parallel_l2 : x + a * y + 6 = 0 → (a - 1) * x + 2 * y + 3 * a = 0 → Parallel) : a = -1 :=
sorry

end find_a_parallel_lines_l227_227016


namespace joey_needs_figures_to_cover_cost_l227_227911

-- Definitions based on conditions
def cost_sneakers : ℕ := 92
def earnings_per_lawn : ℕ := 8
def lawns : ℕ := 3
def earnings_per_hour : ℕ := 5
def work_hours : ℕ := 10
def price_per_figure : ℕ := 9

-- Total earnings from mowing lawns
def earnings_lawns := lawns * earnings_per_lawn
-- Total earnings from job
def earnings_job := work_hours * earnings_per_hour
-- Total earnings from both
def total_earnings := earnings_lawns + earnings_job
-- Remaining amount to cover the cost
def remaining_amount := cost_sneakers - total_earnings

-- Correct answer based on the problem statement
def collectible_figures_needed := remaining_amount / price_per_figure

-- Lean 4 statement to prove the requirement
theorem joey_needs_figures_to_cover_cost :
  collectible_figures_needed = 2 := by
  sorry

end joey_needs_figures_to_cover_cost_l227_227911


namespace fractional_part_sum_leq_l227_227800

noncomputable def fractional_part (z : ℝ) : ℝ :=
  z - (⌊z⌋ : ℝ)

theorem fractional_part_sum_leq (x y : ℝ) :
  fractional_part (x + y) ≤ fractional_part x + fractional_part y :=
by
  sorry

end fractional_part_sum_leq_l227_227800


namespace x_intercept_of_line_l227_227582

def point1 := (10, 3)
def point2 := (-12, -8)

theorem x_intercept_of_line :
  let m := (point2.snd - point1.snd) / (point2.fst - point1.fst)
  let line_eq (x : ℝ) := m * (x - point1.fst) + point1.snd
  ∃ x : ℝ, line_eq x = 0 ∧ x = 4 :=
by
  sorry

end x_intercept_of_line_l227_227582


namespace total_pieces_l227_227265

def pieces_from_friend : ℕ := 123
def pieces_from_brother : ℕ := 136
def pieces_needed : ℕ := 117

theorem total_pieces :
  pieces_from_friend + pieces_from_brother + pieces_needed = 376 :=
by
  unfold pieces_from_friend pieces_from_brother pieces_needed
  sorry

end total_pieces_l227_227265


namespace parabola1_right_of_parabola2_l227_227275

def parabola1 (x : ℝ) : ℝ := x^2 - x + 3
def parabola2 (x : ℝ) : ℝ := x^2 + x + 5

theorem parabola1_right_of_parabola2 :
  ∃ x1 x2 : ℝ, x1 > x2 ∧ parabola1 x1 < parabola2 x2 :=
by
  sorry

end parabola1_right_of_parabola2_l227_227275


namespace oxygen_part_weight_l227_227468

def atomic_weight_N : ℝ := 14.01
def atomic_weight_O : ℝ := 16.00

def molecular_weight_N2O : ℝ := 2 * atomic_weight_N + atomic_weight_O
def given_molecular_weight : ℝ := 108

theorem oxygen_part_weight : molecular_weight_N2O = 44.02 → atomic_weight_O = 16.00 := by
  sorry

end oxygen_part_weight_l227_227468


namespace min_value_xyz_l227_227528

theorem min_value_xyz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 8) : 
  x + 2 * y + 4 * z ≥ 12 := sorry

end min_value_xyz_l227_227528


namespace parabola_directrix_l227_227724

theorem parabola_directrix (x y : ℝ) (h : y = x^2) : 4 * y + 1 = 0 := 
sorry

end parabola_directrix_l227_227724


namespace vertical_bisecting_line_of_circles_l227_227139

theorem vertical_bisecting_line_of_circles :
  ∀ (x y : ℝ),
  (x^2 + y^2 - 2 * x + 6 * y + 2 = 0 ∨ x^2 + y^2 + 4 * x - 2 * y - 4 = 0) →
  (4 * x + 3 * y + 5 = 0) :=
sorry

end vertical_bisecting_line_of_circles_l227_227139


namespace housewife_more_oil_l227_227708

theorem housewife_more_oil 
    (reduction_percent : ℝ := 10)
    (reduced_price : ℝ := 16)
    (budget : ℝ := 800)
    (approx_answer : ℝ := 5.01) :
    let P := reduced_price / (1 - reduction_percent / 100)
    let Q_original := budget / P
    let Q_reduced := budget / reduced_price
    let delta_Q := Q_reduced - Q_original
    abs (delta_Q - approx_answer) < 0.02 := 
by
  -- Let the goal be irrelevant to the proof because the proof isn't provided
  sorry

end housewife_more_oil_l227_227708


namespace find_n_l227_227039

noncomputable def objects_per_hour (n : ℕ) : ℕ := n

theorem find_n (n : ℕ) (h₁ : 1 + (2 / 3) + (1 / 3) + (1 / 3) = 7 / 3) 
  (h₂ : objects_per_hour n * 7 / 3 = 28) : n = 12 :=
by
  have total_hours := h₁ 
  have total_objects := h₂
  sorry

end find_n_l227_227039


namespace range_of_x_l227_227366

variable {p : ℝ} {x : ℝ}

theorem range_of_x (h : 0 ≤ p ∧ p ≤ 4) : x^2 + p * x > 4 * x + p - 3 ↔ (x ≤ -1 ∨ x ≥ 3) :=
sorry

end range_of_x_l227_227366


namespace age_ratio_l227_227400

theorem age_ratio (B_age : ℕ) (H1 : B_age = 34) (A_age : ℕ) (H2 : A_age = B_age + 4) :
  (A_age + 10) / (B_age - 10) = 2 :=
by
  sorry

end age_ratio_l227_227400


namespace local_min_4_l227_227556

def seq (n : ℕ) : ℝ := n^3 - 48 * n + 5

theorem local_min_4 (m : ℕ) (h1 : seq (m-1) > seq m) (h2 : seq (m+1) > seq m) : m = 4 :=
sorry

end local_min_4_l227_227556


namespace non_zero_real_value_l227_227188

theorem non_zero_real_value (y : ℝ) (hy : y ≠ 0) (h : (3 * y)^5 = (9 * y)^4) : y = 27 :=
sorry

end non_zero_real_value_l227_227188


namespace find_wall_width_l227_227900

-- Define the dimensions of the brick in meters
def brick_length : ℝ := 0.20
def brick_width : ℝ := 0.1325
def brick_height : ℝ := 0.08

-- Define the dimensions of the wall in meters
def wall_length : ℝ := 7
def wall_height : ℝ := 15.5
def number_of_bricks : ℝ := 4094.3396226415093

-- Volume of one brick
def brick_volume : ℝ := brick_length * brick_width * brick_height

-- Total volume of bricks used
def total_brick_volume : ℝ := number_of_bricks * brick_volume

-- Wall volume in terms of width W
def wall_volume (W : ℝ) : ℝ := wall_length * W * wall_height

-- The theorem we want to prove
theorem find_wall_width (W : ℝ) (h : wall_volume W = total_brick_volume) : W = 0.08 := by
  sorry

end find_wall_width_l227_227900


namespace calc_mod_residue_l227_227299

theorem calc_mod_residue :
  (245 * 15 - 18 * 8 + 5) % 17 = 0 := by
  sorry

end calc_mod_residue_l227_227299


namespace tan_quadruple_angle_l227_227852

theorem tan_quadruple_angle (θ : ℝ) (h : Real.tan θ = 3) : Real.tan (4 * θ) = -24 / 7 :=
sorry

end tan_quadruple_angle_l227_227852


namespace triangle_perimeter_l227_227420

theorem triangle_perimeter (x : ℕ) :
  (x = 6 ∨ x = 3) →
  ∃ (a b c : ℕ), (a = x ∧ (b = x ∨ c = x)) ∧ 
  (a + b + c = 9 ∨ a + b + c = 15 ∨ a + b + c = 18) :=
by
  intro h
  sorry

end triangle_perimeter_l227_227420


namespace trigonometric_identity_l227_227192

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) : 
  (2 * Real.sin α - Real.cos α) / (Real.sin α + 2 * Real.cos α) = 3 / 4 :=
by 
  sorry

end trigonometric_identity_l227_227192


namespace parabola_vertex_coordinates_l227_227243

theorem parabola_vertex_coordinates :
  ∀ x y : ℝ, y = 3 * (x - 7)^2 + 5 → (7, 5) = (7, 5) :=
by
  intros x y h
  exact rfl

end parabola_vertex_coordinates_l227_227243


namespace Angie_necessities_amount_l227_227711

noncomputable def Angie_salary : ℕ := 80
noncomputable def Angie_left_over : ℕ := 18
noncomputable def Angie_taxes : ℕ := 20
noncomputable def Angie_expenses : ℕ := Angie_salary - Angie_left_over
noncomputable def Angie_necessities : ℕ := Angie_expenses - Angie_taxes

theorem Angie_necessities_amount :
  Angie_necessities = 42 :=
by
  unfold Angie_necessities
  unfold Angie_expenses
  sorry

end Angie_necessities_amount_l227_227711


namespace kona_distance_proof_l227_227566

-- Defining the distances as constants
def distance_to_bakery : ℕ := 9
def distance_from_grandmother_to_home : ℕ := 27
def additional_trip_distance : ℕ := 6

-- Defining the variable for the distance from bakery to grandmother's house
def x : ℕ := 30

-- Main theorem to prove the distance
theorem kona_distance_proof :
  distance_to_bakery + x + distance_from_grandmother_to_home = 2 * x + additional_trip_distance :=
by
  sorry

end kona_distance_proof_l227_227566


namespace cos_pi_plus_alpha_l227_227893

-- Define the angle α and conditions given
variable (α : Real) (h1 : 0 < α) (h2 : α < π/2)

-- Given condition sine of α
variable (h3 : Real.sin α = 4/5)

-- Define the cosine identity to prove the assertion
theorem cos_pi_plus_alpha (h1 : 0 < α) (h2 : α < π/2) (h3 : Real.sin α = 4/5) :
  Real.cos (π + α) = -3/5 :=
sorry

end cos_pi_plus_alpha_l227_227893


namespace height_of_tree_l227_227653

-- Definitions based on conditions
def net_gain (hop: ℕ) (slip: ℕ) : ℕ := hop - slip

def total_distance (hours: ℕ) (net_gain: ℕ) (final_hop: ℕ) : ℕ :=
  hours * net_gain + final_hop

-- Conditions
def hop : ℕ := 3
def slip : ℕ := 2
def time : ℕ := 20

-- Deriving the net gain per hour
#eval net_gain hop slip  -- Evaluates to 1

-- Final height proof problem
theorem height_of_tree : total_distance 19 (net_gain hop slip) hop = 22 := by
  sorry  -- Proof to be filled in

end height_of_tree_l227_227653


namespace object_distance_traveled_l227_227453

theorem object_distance_traveled
  (t : ℕ) (v_mph : ℝ) (mile_to_feet : ℕ)
  (h_t : t = 2)
  (h_v : v_mph = 68.18181818181819)
  (h_mile : mile_to_feet = 5280) :
  ∃ d : ℝ, d = 200 :=
by {
  sorry
}

end object_distance_traveled_l227_227453


namespace john_plays_periods_l227_227938

theorem john_plays_periods
  (PointsPer4Minutes : ℕ := 7)
  (PeriodDurationMinutes : ℕ := 12)
  (TotalPoints : ℕ := 42) :
  (TotalPoints / PointsPer4Minutes) / (PeriodDurationMinutes / 4) = 2 := by
  sorry

end john_plays_periods_l227_227938


namespace least_xy_l227_227917

theorem least_xy (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 1 / (3 * y) = 1 / 9) : x * y = 108 :=
by
  sorry

end least_xy_l227_227917


namespace monthly_income_ratio_l227_227630

noncomputable def A_annual_income : ℝ := 571200
noncomputable def C_monthly_income : ℝ := 17000
noncomputable def B_monthly_income : ℝ := C_monthly_income * 1.12
noncomputable def A_monthly_income : ℝ := A_annual_income / 12

theorem monthly_income_ratio :
  (A_monthly_income / B_monthly_income) = 2.5 :=
by
  sorry

end monthly_income_ratio_l227_227630


namespace perpendicular_vectors_eq_l227_227261

theorem perpendicular_vectors_eq {x : ℝ} (h : (x - 5) * 2 + 3 * x = 0) : x = 2 :=
sorry

end perpendicular_vectors_eq_l227_227261


namespace product_of_roots_l227_227484

theorem product_of_roots (r : ℂ) (h1 : r^7 = 1) (h2 : r ≠ 1) :
    (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) * (r^6 - 1) = 7 :=
by sorry

end product_of_roots_l227_227484


namespace fraction_oil_is_correct_l227_227894

noncomputable def fraction_oil_third_bottle (C : ℚ) (oil1 : ℚ) (oil2 : ℚ) (water1 : ℚ) (water2 : ℚ) := 
  (oil1 + oil2) / (oil1 + oil2 + water1 + water2)

theorem fraction_oil_is_correct (C : ℚ) (hC : C > 0) :
  let oil1 := C / 2
  let oil2 := C / 2
  let water1 := C / 2
  let water2 := 3 * C / 4
  fraction_oil_third_bottle C oil1 oil2 water1 water2 = 4 / 9 := by
  sorry

end fraction_oil_is_correct_l227_227894


namespace purchase_price_of_grinder_l227_227710

theorem purchase_price_of_grinder (G : ℝ) (H : 0.95 * G + 8800 - (G + 8000) = 50) : G = 15000 := 
sorry

end purchase_price_of_grinder_l227_227710


namespace least_number_subtracted_to_divisible_by_10_l227_227436

def least_subtract_to_divisible_by_10 (n : ℕ) : ℕ :=
  let last_digit := n % 10
  10 - last_digit

theorem least_number_subtracted_to_divisible_by_10 (n : ℕ) : (n = 427751) → ((n - least_subtract_to_divisible_by_10 n) % 10 = 0) :=
by
  intros h
  sorry

end least_number_subtracted_to_divisible_by_10_l227_227436


namespace bears_total_l227_227110

-- Define the number of each type of bear
def brown_bears : ℕ := 15
def white_bears : ℕ := 24
def black_bears : ℕ := 27
def polar_bears : ℕ := 12
def grizzly_bears : ℕ := 18

-- Define the total number of bears
def total_bears : ℕ := brown_bears + white_bears + black_bears + polar_bears + grizzly_bears

-- The theorem stating the total number of bears is 96
theorem bears_total : total_bears = 96 :=
by
  -- The proof is omitted here
  sorry

end bears_total_l227_227110


namespace problem_statement_l227_227593

variable {a b c d : ℝ}

theorem problem_statement (h : a * d - b * c = 1) : 
  a^2 + b^2 + c^2 + d^2 + a * b + c * d ≠ 1 := 
sorry

end problem_statement_l227_227593


namespace solve_fractional_equation_l227_227460

theorem solve_fractional_equation (x : ℝ) (h : x ≠ 3) : (2 * x) / (x - 3) = 1 ↔ x = -3 :=
by
  sorry

end solve_fractional_equation_l227_227460


namespace no_point_satisfies_both_systems_l227_227176

theorem no_point_satisfies_both_systems (x y : ℝ) :
  (y < 3 ∧ x - y < 3 ∧ x + y < 4) ∧
  ((y - 3) * (x - y - 3) ≥ 0 ∧ (y - 3) * (x + y - 4) ≤ 0 ∧ (x - y - 3) * (x + y - 4) ≤ 0)
  → false :=
sorry

end no_point_satisfies_both_systems_l227_227176


namespace minimize_J_l227_227151

noncomputable def H (p q : ℝ) : ℝ := -3 * p * q + 4 * p * (1 - q) + 2 * (1 - p) * q - 5 * (1 - p) * (1 - q)

noncomputable def J (p : ℝ) : ℝ := max (H p 0) (H p 1)

theorem minimize_J : ∃ p : ℝ, 0 ≤ p ∧ p ≤ 1 ∧ (∀ p' : ℝ, 0 ≤ p' ∧ p' ≤ 1 → J p ≤ J p') ∧ p = 1 / 2 :=
by
  sorry

end minimize_J_l227_227151


namespace smallest_nineteen_multiple_l227_227249

theorem smallest_nineteen_multiple (n : ℕ) 
  (h₁ : 19 * n ≡ 5678 [MOD 11]) : n = 8 :=
by sorry

end smallest_nineteen_multiple_l227_227249


namespace largest_integer_less_than_100_with_remainder_4_l227_227295

theorem largest_integer_less_than_100_with_remainder_4 (k n : ℤ) (h1 : k = 7 * n + 4) (h2 : k < 100) : k ≤ 95 :=
sorry

end largest_integer_less_than_100_with_remainder_4_l227_227295


namespace set_equality_l227_227108

open Set

variable (A : Set ℕ)

theorem set_equality (h1 : {1, 3} ⊆ A) (h2 : {1, 3} ∪ A = {1, 3, 5}) : A = {1, 3, 5} :=
sorry

end set_equality_l227_227108


namespace prove_A_exactly_2_hits_prove_B_at_least_2_hits_prove_B_exactly_2_more_hits_A_l227_227973

noncomputable def probability_A_exactly_2_hits :=
  let p_A := 1/2
  let trials := 3
  (trials.choose 2) * (p_A ^ 2) * ((1 - p_A) ^ (trials - 2))

noncomputable def probability_B_at_least_2_hits :=
  let p_B := 2/3
  let trials := 3
  (trials.choose 2) * (p_B ^ 2) * ((1 - p_B) ^ (trials - 2)) + (trials.choose 3) * (p_B ^ 3)

noncomputable def probability_B_exactly_2_more_hits_A :=
  let p_A := 1/2
  let p_B := 2/3
  let trials := 3
  let B_2_A_0 := (trials.choose 2) * (p_B ^ 2) * (1 - p_B) * (trials.choose 0) * (p_A ^ 0) * ((1 - p_A) ^ trials)
  let B_3_A_1 := (trials.choose 3) * (p_B ^ 3) * (trials.choose 1) * (p_A ^ 1) * ((1 - p_A) ^ (trials - 1))
  B_2_A_0 + B_3_A_1

theorem prove_A_exactly_2_hits : probability_A_exactly_2_hits = 3/8 := sorry
theorem prove_B_at_least_2_hits : probability_B_at_least_2_hits = 20/27 := sorry
theorem prove_B_exactly_2_more_hits_A : probability_B_exactly_2_more_hits_A = 1/6 := sorry

end prove_A_exactly_2_hits_prove_B_at_least_2_hits_prove_B_exactly_2_more_hits_A_l227_227973


namespace min_possible_value_l227_227391

theorem min_possible_value (a b : ℤ) (h : a > b) :
  (∃ x : ℚ, x = (2 * a + 3 * b) / (a - 2 * b) ∧ (x + 1 / x = (2 : ℚ))) :=
sorry

end min_possible_value_l227_227391


namespace age_difference_l227_227952

variable (A B C : ℕ)

theorem age_difference (h : A + B = B + C + 12) : A - C = 12 := 
sorry

end age_difference_l227_227952


namespace sand_bucket_capacity_l227_227915

theorem sand_bucket_capacity
  (sandbox_depth : ℝ)
  (sandbox_width : ℝ)
  (sandbox_length : ℝ)
  (sand_weight_per_cubic_foot : ℝ)
  (water_per_4_trips : ℝ)
  (water_bottle_ounces : ℝ)
  (water_bottle_cost : ℝ)
  (tony_total_money : ℝ)
  (tony_change : ℝ)
  (tony's_bucket_capacity : ℝ) :
  sandbox_depth = 2 →
  sandbox_width = 4 →
  sandbox_length = 5 →
  sand_weight_per_cubic_foot = 3 →
  water_per_4_trips = 3 →
  water_bottle_ounces = 15 →
  water_bottle_cost = 2 →
  tony_total_money = 10 →
  tony_change = 4 →
  tony's_bucket_capacity = 2 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry -- skipping the proof as per instructions

end sand_bucket_capacity_l227_227915


namespace gcd_266_209_l227_227875

-- Definitions based on conditions
def a : ℕ := 266
def b : ℕ := 209

-- Theorem stating the GCD of a and b
theorem gcd_266_209 : Nat.gcd a b = 19 :=
by {
  -- Declare the specific integers as conditions
  let a := 266
  let b := 209
  -- Use the Euclidean algorithm (steps within the proof are not required)
  -- State that the conclusion is the GCD of a and b 
  sorry
}

end gcd_266_209_l227_227875


namespace limonia_largest_unachievable_l227_227029

noncomputable def largest_unachievable_amount (n : ℕ) : ℕ :=
  12 * n^2 + 14 * n - 1

theorem limonia_largest_unachievable (n : ℕ) :
  ∀ k, ¬ ∃ a b c d : ℕ, 
    k = a * (6 * n + 1) + b * (6 * n + 4) + c * (6 * n + 7) + d * (6 * n + 10) 
    → k = largest_unachievable_amount n :=
sorry

end limonia_largest_unachievable_l227_227029


namespace quadrilateral_perimeter_l227_227621

noncomputable def EG (FH : ℝ) : ℝ := Real.sqrt ((FH + 5) ^ 2 + FH ^ 2)

theorem quadrilateral_perimeter 
  (EF FH GH : ℝ) 
  (h1 : EF = 12)
  (h2 : FH = 7)
  (h3 : GH = FH) :
  EF + FH + GH + EG FH = 26 + Real.sqrt 193 :=
by
  rw [h1, h2, h3]
  sorry

end quadrilateral_perimeter_l227_227621


namespace expressionEquals243_l227_227945

noncomputable def calculateExpression : ℕ :=
  (1 / 3) * 9 * (1 / 27) * 81 * (1 / 243) * 729 * (1 / 2187) * 6561 *
  (1 / 19683) * 59049

theorem expressionEquals243 : calculateExpression = 243 := by
  sorry

end expressionEquals243_l227_227945


namespace fouad_age_l227_227096

theorem fouad_age (F : ℕ) (Ahmed_current_age : ℕ) (H : Ahmed_current_age = 11) (H2 : F + 4 = 2 * Ahmed_current_age) : F = 18 :=
by
  -- We do not need to write the proof steps, just a placeholder.
  sorry

end fouad_age_l227_227096


namespace jogger_distance_l227_227002

theorem jogger_distance 
(speed_jogger : ℝ := 9)
(speed_train : ℝ := 45)
(train_length : ℕ := 120)
(time_to_pass : ℕ := 38)
(relative_speed_mps : ℝ := (speed_train - speed_jogger) * (1 / 3.6))
(distance_covered : ℝ := (relative_speed_mps * time_to_pass))
(d : ℝ := distance_covered - train_length) :
d = 260 := sorry

end jogger_distance_l227_227002


namespace difference_is_divisible_by_p_l227_227789

-- Lean 4 statement equivalent to the math proof problem
theorem difference_is_divisible_by_p
  (a : ℕ → ℕ) (p : ℕ) (d : ℕ)
  (h_prime : Nat.Prime p)
  (h_prog : ∀ i j: ℕ, 1 ≤ i ∧ i ≤ p ∧ 1 ≤ j ∧ j ≤ p ∧ i < j → a j = a (i + 1) + (j - 1) * d)
  (h_a_gt_p : a 1 > p)
  (h_arith_prog_primes : ∀ i: ℕ, 1 ≤ i ∧ i ≤ p → Nat.Prime (a i)) :
  d % p = 0 := sorry

end difference_is_divisible_by_p_l227_227789


namespace train_speed_computed_l227_227878

noncomputable def train_speed_in_kmh (train_length : ℝ) (platform_length : ℝ) (time_in_seconds : ℝ) : ℝ :=
  let total_distance := train_length + platform_length
  let speed_mps := total_distance / time_in_seconds
  speed_mps * 3.6

theorem train_speed_computed :
  train_speed_in_kmh 250 50.024 15 = 72.006 := by
  sorry

end train_speed_computed_l227_227878


namespace integer_modulo_solution_l227_227990

theorem integer_modulo_solution (a : ℤ) : 
  (5 ∣ a^3 + 3 * a + 1) ↔ (a % 5 = 1 ∨ a % 5 = 2) := 
by
  exact sorry

end integer_modulo_solution_l227_227990


namespace arithmetic_sequence_ratio_l227_227955

-- Define conditions
def sum_ratios (A_n B_n : ℕ → ℚ) (n : ℕ) : Prop := (A_n n) / (B_n n) = (4 * n + 2) / (5 * n - 5)
def arithmetic_sequences (a_n b_n : ℕ → ℚ) : Prop :=
  ∃ A_n B_n : ℕ → ℚ,
    (∀ n, A_n n = n * (a_n 1) + (n * (n - 1) / 2) * (a_n 2 - a_n 1)) ∧
    (∀ n, B_n n = n * (b_n 1) + (n * (n - 1) / 2) * (b_n 2 - b_n 1)) ∧
    ∀ n, sum_ratios A_n B_n n

-- Theorem to be proven
theorem arithmetic_sequence_ratio
  (a_n b_n : ℕ → ℚ)
  (h : arithmetic_sequences a_n b_n) :
  (a_n 5 + a_n 13) / (b_n 5 + b_n 13) = 7 / 8 :=
sorry

end arithmetic_sequence_ratio_l227_227955


namespace square_binomial_unique_a_l227_227849

theorem square_binomial_unique_a (a : ℝ) : 
  (∃ r s : ℝ, (ax^2 - 8*x + 16) = (r*x + s)^2) ↔ a = 1 :=
by
  sorry

end square_binomial_unique_a_l227_227849


namespace sara_grew_4_onions_l227_227513

def onions_grown_by_sally : Nat := 5
def onions_grown_by_fred : Nat := 9
def total_onions_grown : Nat := 18

def onions_grown_by_sara : Nat :=
  total_onions_grown - (onions_grown_by_sally + onions_grown_by_fred)

theorem sara_grew_4_onions :
  onions_grown_by_sara = 4 :=
by
  sorry

end sara_grew_4_onions_l227_227513


namespace lines_parallel_if_perpendicular_to_same_plane_l227_227829

variables (m n : Line) (α : Plane)

-- Define conditions using Lean's logical constructs
def perpendicular_to_plane (l : Line) (p : Plane) : Prop := sorry -- This would define the condition
def parallel_lines (l1 l2 : Line) : Prop := sorry -- This would define the condition

-- The statement to prove
theorem lines_parallel_if_perpendicular_to_same_plane 
  (h1 : perpendicular_to_plane m α) 
  (h2 : perpendicular_to_plane n α) : 
  parallel_lines m n :=
sorry

end lines_parallel_if_perpendicular_to_same_plane_l227_227829


namespace auditorium_rows_l227_227476

noncomputable def rows_in_auditorium : Nat :=
  let class1 := 30
  let class2 := 26
  let condition1 := ∃ row : Nat, row < class1 ∧ ∀ students_per_row : Nat, students_per_row ≤ row 
  let condition2 := ∃ empty_rows : Nat, empty_rows ≥ 3 ∧ ∀ students : Nat, students = class2 - empty_rows
  29

theorem auditorium_rows (n : Nat) (class1 : Nat) (class2 : Nat) (c1 : class1 ≥ n) (c2 : class2 ≤ n - 3)
  : n = 29 :=
by
  sorry

end auditorium_rows_l227_227476


namespace cats_left_l227_227842

theorem cats_left (siamese_cats : ℕ) (house_cats : ℕ) (cats_sold : ℕ) (total_initial_cats : ℕ) (remaining_cats : ℕ) :
  siamese_cats = 15 → house_cats = 49 → cats_sold = 19 → total_initial_cats = siamese_cats + house_cats → remaining_cats = total_initial_cats - cats_sold → remaining_cats = 45 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2] at h4
  rw [h4, h3] at h5
  exact h5

end cats_left_l227_227842


namespace value_of_f_8_minus_f_4_l227_227665

noncomputable def f : ℝ → ℝ := sorry

theorem value_of_f_8_minus_f_4 :
  -- Conditions
  (∀ x, f (-x) = -f x) ∧              -- odd function
  (∀ x, f (x + 5) = f x) ∧            -- period of 5
  (f 1 = 1) ∧                         -- f(1) = 1
  (f 2 = 3) →                         -- f(2) = 3
  -- Goal
  f 8 - f 4 = -2 :=
sorry

end value_of_f_8_minus_f_4_l227_227665


namespace interval_of_defined_expression_l227_227402

theorem interval_of_defined_expression (x : ℝ) :
  (x > 2 ∧ x < 5) ↔ (x - 2 > 0 ∧ 5 - x > 0) :=
by
  sorry

end interval_of_defined_expression_l227_227402


namespace solve_inequality_l227_227993

open Set Real

theorem solve_inequality (x : ℝ) : { x : ℝ | x^2 - 4 * x > 12 } = {x : ℝ | x < -2} ∪ {x : ℝ | 6 < x} := 
sorry

end solve_inequality_l227_227993


namespace total_coins_are_correct_l227_227198

-- Define the initial number of coins
def initial_dimes : Nat := 2
def initial_quarters : Nat := 6
def initial_nickels : Nat := 5

-- Define the additional coins given by Linda's mother
def additional_dimes : Nat := 2
def additional_quarters : Nat := 10
def additional_nickels : Nat := 2 * initial_nickels

-- Calculate the total number of each type of coin
def total_dimes : Nat := initial_dimes + additional_dimes
def total_quarters : Nat := initial_quarters + additional_quarters
def total_nickels : Nat := initial_nickels + additional_nickels

-- Total number of coins
def total_coins : Nat := total_dimes + total_quarters + total_nickels

-- Theorem to prove the total number of coins is 35
theorem total_coins_are_correct : total_coins = 35 := by
  -- Skip the proof
  sorry

end total_coins_are_correct_l227_227198


namespace value_of_x_l227_227079

theorem value_of_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 :=
by
  sorry

end value_of_x_l227_227079


namespace translation_coordinates_l227_227287

theorem translation_coordinates :
  ∀ (x y : ℤ) (a : ℤ), 
  (x, y) = (3, -4) → a = 5 → (x - a, y) = (-2, -4) :=
by
  sorry

end translation_coordinates_l227_227287


namespace quadratic_ineq_real_solutions_l227_227907

theorem quadratic_ineq_real_solutions (d : ℝ) (h₀ : 0 < d) :
  (∀ x : ℝ, x^2 - 8 * x + d < 0 → 0 < d ∧ d < 16) :=
by
  sorry

end quadratic_ineq_real_solutions_l227_227907


namespace compute_pqr_l227_227071

theorem compute_pqr (p q r : ℤ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) 
  (h_sum : p + q + r = 26) (h_eq : (1 : ℚ) / ↑p + (1 : ℚ) / ↑q + (1 : ℚ) / ↑r + 360 / (p * q * r) = 1) : 
  p * q * r = 576 := 
sorry

end compute_pqr_l227_227071


namespace trigonometric_identity_l227_227286

theorem trigonometric_identity (α : ℝ) :
    1 - 1/4 * (Real.sin (2 * α)) ^ 2 + Real.cos (2 * α) = (Real.cos α) ^ 2 + (Real.cos α) ^ 4 :=
by
  sorry

end trigonometric_identity_l227_227286


namespace cost_price_is_3000_l227_227269

variable (CP SP : ℝ)

-- Condition: selling price (SP) is 20% more than the cost price (CP)
def sellingPrice : ℝ := CP + 0.20 * CP

-- Condition: selling price (SP) is Rs. 3600
axiom selling_price_eq : SP = 3600

-- Given the above conditions, prove that the cost price (CP) is Rs. 3000
theorem cost_price_is_3000 (h : sellingPrice CP = SP) : CP = 3000 := by
  sorry

end cost_price_is_3000_l227_227269


namespace num_paths_from_E_to_G_pass_through_F_l227_227648

-- Definitions for the positions on the grid.
def E := (0, 4)
def G := (5, 0)
def F := (3, 3)

-- Function to calculate the number of combinations.
def binom (n k: ℕ) : ℕ := Nat.choose n k

-- The mathematical statement to be proven.
theorem num_paths_from_E_to_G_pass_through_F :
  (binom 4 1) * (binom 5 2) = 40 :=
by
  -- Placeholder for the proof.
  sorry

end num_paths_from_E_to_G_pass_through_F_l227_227648


namespace number_of_adults_attending_concert_l227_227201

-- We have to define the constants and conditions first.
variable (A C : ℕ)
variable (h1 : A + C = 578)
variable (h2 : 2 * A + 3 / 2 * C = 985)

-- Now we state the theorem that given these conditions, A is equal to 236.

theorem number_of_adults_attending_concert : A = 236 :=
by sorry

end number_of_adults_attending_concert_l227_227201


namespace find_abc_l227_227193

open Real

noncomputable def abc_value (a b c : ℝ) : ℝ := a * b * c

theorem find_abc (a b c : ℝ)
  (h₁ : a - b = 3)
  (h₂ : a^2 + b^2 = 39)
  (h₃ : a + b + c = 10) :
  abc_value a b c = -150 + 15 * Real.sqrt 69 :=
by
  sorry

end find_abc_l227_227193


namespace carrie_pays_94_l227_227241

theorem carrie_pays_94 :
  ∀ (num_shirts num_pants num_jackets : ℕ) (cost_shirt cost_pants cost_jacket : ℕ),
  num_shirts = 4 →
  cost_shirt = 8 →
  num_pants = 2 →
  cost_pants = 18 →
  num_jackets = 2 →
  cost_jacket = 60 →
  (cost_shirt * num_shirts + cost_pants * num_pants + cost_jacket * num_jackets) / 2 = 94 :=
by
  intros num_shirts num_pants num_jackets cost_shirt cost_pants cost_jacket
  sorry

end carrie_pays_94_l227_227241


namespace remainder_when_dividing_by_2x_minus_4_l227_227536

def f (x : ℝ) := 4 * x^3 - 9 * x^2 + 12 * x - 14
def g (x : ℝ) := 2 * x - 4

theorem remainder_when_dividing_by_2x_minus_4 : f 2 = 6 := by
  sorry

end remainder_when_dividing_by_2x_minus_4_l227_227536


namespace monica_milk_l227_227132

theorem monica_milk (don_milk : ℚ) (rachel_fraction : ℚ) (monica_fraction : ℚ) (h_don : don_milk = 3 / 4)
  (h_rachel : rachel_fraction = 1 / 2) (h_monica : monica_fraction = 1 / 3) :
  monica_fraction * (rachel_fraction * don_milk) = 1 / 8 :=
by
  sorry

end monica_milk_l227_227132


namespace mittens_per_box_l227_227257

theorem mittens_per_box (boxes : ℕ) (scarves_per_box : ℕ) (total_clothing : ℕ) (h_boxes : boxes = 7) (h_scarves : scarves_per_box = 3) (h_total : total_clothing = 49) : 
  let total_scarves := boxes * scarves_per_box
  let total_mittens := total_clothing - total_scarves
  total_mittens / boxes = 4 :=
by
  sorry

end mittens_per_box_l227_227257


namespace max_groups_l227_227102

-- Define the conditions
def valid_eq (a b : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ (3 * a + b = 13)

-- The proof problem: No need for the proof body, just statement
theorem max_groups : ∃! (l : List (ℕ × ℕ)), (∀ ab ∈ l, valid_eq ab.fst ab.snd) ∧ l.length = 3 := sorry

end max_groups_l227_227102


namespace no_infinite_subset_of_natural_numbers_l227_227950

theorem no_infinite_subset_of_natural_numbers {
  S : Set ℕ 
} (hS_infinite : S.Infinite) :
  ¬ (∀ a b : ℕ, a ∈ S → b ∈ S → a^2 - a * b + b^2 ∣ (a * b)^2) :=
sorry

end no_infinite_subset_of_natural_numbers_l227_227950


namespace Zach_scored_more_l227_227954

theorem Zach_scored_more :
  let Zach := 42
  let Ben := 21
  Zach - Ben = 21 :=
by
  let Zach := 42
  let Ben := 21
  exact rfl

end Zach_scored_more_l227_227954


namespace solve_system_l227_227832

theorem solve_system (a b c : ℝ)
  (h1 : b + c = 10 - 4 * a)
  (h2 : a + c = -16 - 4 * b)
  (h3 : a + b = 9 - 4 * c) :
  2 * a + 2 * b + 2 * c = 1 :=
by
  sorry

end solve_system_l227_227832


namespace last_three_digits_of_7_pow_99_l227_227455

theorem last_three_digits_of_7_pow_99 : (7 ^ 99) % 1000 = 573 := 
by sorry

end last_three_digits_of_7_pow_99_l227_227455


namespace john_makes_money_l227_227600

-- Definitions of the conditions
def num_cars := 5
def time_first_3_cars := 3 * 40 -- 3 cars each take 40 minutes
def time_remaining_car := 40 * 3 / 2 -- Each remaining car takes 50% longer
def time_remaining_cars := 2 * time_remaining_car -- 2 remaining cars
def total_time_min := time_first_3_cars + time_remaining_cars
def total_time_hr := total_time_min / 60 -- Convert total time from minutes to hours
def rate_per_hour := 20

-- Theorem statement
theorem john_makes_money : total_time_hr * rate_per_hour = 80 := by
  sorry

end john_makes_money_l227_227600


namespace sum_digits_l227_227940

def repeat_pattern (d: ℕ) (n: ℕ) : ℕ :=
  let pattern := if d = 404 then 404 else if d = 707 then 707 else 0
  pattern * 10^(n / 3)

def N1 := repeat_pattern 404 101
def N2 := repeat_pattern 707 101
def P := N1 * N2

def thousands_digit (n: ℕ) : ℕ :=
  (n / 1000) % 10

def units_digit (n: ℕ) : ℕ :=
  n % 10

theorem sum_digits : thousands_digit P + units_digit P = 10 := by
  sorry

end sum_digits_l227_227940


namespace work_completed_by_a_l227_227816

theorem work_completed_by_a (a b : ℕ) (work_in_30_days : a + b = 4 * 30) (a_eq_3b : a = 3 * b) : (120 / a) = 40 :=
by
  -- Given a + b = 120 and a = 3 * b, prove that 120 / a = 40
  sorry

end work_completed_by_a_l227_227816


namespace sum_xyz_le_two_l227_227301

theorem sum_xyz_le_two (x y z : ℝ) (h : 2 * x + y^2 + z^2 ≤ 2) : x + y + z ≤ 2 :=
sorry

end sum_xyz_le_two_l227_227301


namespace cost_per_foot_l227_227417

theorem cost_per_foot (area : ℕ) (total_cost : ℕ) (side_length : ℕ) (perimeter : ℕ) (cost_per_foot : ℕ) :
  area = 289 → total_cost = 3944 → side_length = Nat.sqrt 289 → perimeter = 4 * 17 →
  cost_per_foot = total_cost / perimeter → cost_per_foot = 58 :=
by
  intros
  sorry

end cost_per_foot_l227_227417


namespace total_eggs_l227_227865

noncomputable def total_eggs_in_all_containers (n : ℕ) (f l : ℕ) : ℕ :=
  n * (f * l)

theorem total_eggs (f l : ℕ) :
  (f = 14 + 20 - 1) →
  (l = 3 + 2 - 1) →
  total_eggs_in_all_containers 28 f l = 3696 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end total_eggs_l227_227865


namespace find_value_simplify_expression_l227_227502

-- Define the first part of the problem
theorem find_value (α : ℝ) (h : Real.tan α = 1/3) : 
  (1 / (2 * Real.sin α * Real.cos α + Real.cos α ^ 2)) = 2 / 3 := 
  sorry

-- Define the second part of the problem
theorem simplify_expression (α : ℝ) (h : Real.tan α = 1/3) : 
  (Real.tan (π - α) * Real.cos (2 * π - α) * Real.sin (-α + 3 * π / 2)) / (Real.cos (-α - π) * Real.sin (-π - α)) = -1 := 
  sorry

end find_value_simplify_expression_l227_227502


namespace alpha_half_quadrant_l227_227524

theorem alpha_half_quadrant (k : ℤ) (α : ℝ)
  (h : 2 * k * Real.pi - Real.pi / 2 < α ∧ α < 2 * k * Real.pi) :
  (∃ n : ℤ, 2 * n * Real.pi - Real.pi / 4 < α / 2 ∧ α / 2 < 2 * n * Real.pi) ∨
  (∃ n : ℤ, (2 * n + 1) * Real.pi - Real.pi / 4 < α / 2 ∧ α / 2 < (2 * n + 1) * Real.pi) :=
sorry

end alpha_half_quadrant_l227_227524


namespace tournament_matches_l227_227240

theorem tournament_matches (n : ℕ) (total_matches : ℕ) (matches_three_withdrew : ℕ) (matches_after_withdraw : ℕ) :
  ∀ (x : ℕ), total_matches = (n * (n - 1) / 2) → matches_three_withdrew = 6 - x → matches_after_withdraw = total_matches - (3 * 2 - x) → 
  matches_after_withdraw = 50 → x = 1 :=
by
  intros
  sorry

end tournament_matches_l227_227240


namespace proof_inequality_l227_227620

noncomputable def proof_problem (x : ℝ) (Hx : x ∈ Set.Ioo (Real.exp (-1)) (1)) : Prop :=
  let a := Real.log x
  let b := (1 / 2) ^ (Real.log x)
  let c := Real.exp (Real.log x)
  b > c ∧ c > a

theorem proof_inequality {x : ℝ} (Hx : x ∈ Set.Ioo (Real.exp (-1)) (1)) :
  proof_problem x Hx :=
sorry

end proof_inequality_l227_227620


namespace bananas_in_each_bunch_l227_227559

theorem bananas_in_each_bunch (x: ℕ) : (6 * x + 5 * 7 = 83) → x = 8 :=
by
  intro h
  sorry

end bananas_in_each_bunch_l227_227559


namespace final_statement_l227_227719

variable (f : ℝ → ℝ)

-- Conditions
axiom even_function : ∀ x, f (x) = f (-x)
axiom periodic_minus_one : ∀ x, f (x + 1) = -f (x)
axiom increasing_on_neg_one_to_zero : ∀ x y, -1 ≤ x ∧ x < y ∧ y ≤ 0 → f (x) < f (y)

-- Statement
theorem final_statement :
  (∀ x, f (x + 2) = f (x)) ∧
  (¬ (∀ x, 0 ≤ x ∧ x ≤ 1 → f (x) < f (x + 1))) ∧
  (∀ x y, 1 ≤ x ∧ x < y ∧ y ≤ 2 → f (x) < f (y)) ∧
  (f (2) = f (0)) :=
by
  sorry

end final_statement_l227_227719


namespace range_of_a_l227_227624

noncomputable def is_even_function (f : ℝ → ℝ) : Prop :=
∀ x, f x = f (-x)

noncomputable def strictly_increasing_on_nonnegative (f : ℝ → ℝ) : Prop :=
∀ x1 x2, (0 ≤ x1 ∧ 0 ≤ x2 ∧ x1 ≠ x2 → (f x1 - f x2) / (x1 - x2) > 0)

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) (m n : ℝ) (h_even : is_even_function f)
  (h_strict : strictly_increasing_on_nonnegative f)
  (h_m : m = 1/2) (h_f : ∀ x, m ≤ x ∧ x ≤ n → f (a * x + 1) ≤ f 2) :
  a ≤ 2 :=
sorry

end range_of_a_l227_227624


namespace number_of_people_in_family_l227_227655

-- Define the conditions
def planned_spending : ℝ := 15
def savings_percentage : ℝ := 0.40
def cost_per_orange : ℝ := 1.5

-- Define the proof target: the number of people in the family
theorem number_of_people_in_family : 
  planned_spending * savings_percentage / cost_per_orange = 4 := 
by
  -- sorry to skip the proof; this is for statement only
  sorry

end number_of_people_in_family_l227_227655


namespace perfect_square_m_value_l227_227584

theorem perfect_square_m_value (m : ℤ) :
  (∃ a : ℤ, ∀ x : ℝ, (x^2 + (m : ℝ)*x + 1 : ℝ) = (x + (a : ℝ))^2) → m = 2 ∨ m = -2 :=
by
  sorry

end perfect_square_m_value_l227_227584


namespace longest_leg_of_smallest_triangle_l227_227008

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

end longest_leg_of_smallest_triangle_l227_227008


namespace original_number_l227_227052

theorem original_number (x : ℝ) (h : 1.47 * x = 1214.33) : x = 826.14 :=
sorry

end original_number_l227_227052


namespace valid_N_count_l227_227837

theorem valid_N_count : 
  (∃ n : ℕ, 0 < n ∧ (49 % (n + 3) = 0) ∧ (49 / (n + 3)) % 2 = 1) → 
  (∃ count : ℕ, count = 2) :=
sorry

end valid_N_count_l227_227837


namespace least_number_subtracted_l227_227372

theorem least_number_subtracted (n : ℕ) (h : n = 2361) : 
  ∃ k, (n - k) % 23 = 0 ∧ k = 15 := 
by
  sorry

end least_number_subtracted_l227_227372


namespace unique_infinite_sequence_l227_227874

-- Defining conditions for the infinite sequence of negative integers
variable (a : ℕ → ℤ)
  
-- Condition 1: Elements in sequence are negative integers
def sequence_negative : Prop :=
  ∀ n, a n < 0 

-- Condition 2: For every positive integer n, the first n elements taken modulo n have n distinct remainders
def distinct_mod_remainders (n : ℕ) : Prop :=
  ∀ i j, i < n → j < n → i ≠ j → (a i % n ≠ a j % n) 

-- The main theorem statement
theorem unique_infinite_sequence (a : ℕ → ℤ) 
  (h1 : sequence_negative a) 
  (h2 : ∀ n, distinct_mod_remainders a n) :
  ∀ k : ℤ, ∃! n, a n = k :=
sorry

end unique_infinite_sequence_l227_227874


namespace find_remainder_l227_227752

-- Definition of N based on given conditions
def N : ℕ := 44 * 432

-- Definition of next multiple of 432
def next_multiple_of_432 : ℕ := N + 432

-- Statement to prove the remainder when next_multiple_of_432 is divided by 39 is 12
theorem find_remainder : next_multiple_of_432 % 39 = 12 := 
by sorry

end find_remainder_l227_227752


namespace number_of_students_per_normal_class_l227_227478

theorem number_of_students_per_normal_class (total_students : ℕ) (percentage_moving : ℕ) (grade_levels : ℕ) (adv_class_size : ℕ) (additional_classes : ℕ) 
  (h1 : total_students = 1590) 
  (h2 : percentage_moving = 40) 
  (h3 : grade_levels = 3) 
  (h4 : adv_class_size = 20) 
  (h5 : additional_classes = 6) : 
  (total_students * percentage_moving / 100 / grade_levels - adv_class_size) / additional_classes = 32 :=
by
  sorry

end number_of_students_per_normal_class_l227_227478


namespace gcd_lcm_product_l227_227416

theorem gcd_lcm_product (a b : ℕ) (h1 : a = 24) (h2 : b = 60) :
  Nat.gcd a b * Nat.lcm a b = 1440 :=
by
  sorry

end gcd_lcm_product_l227_227416


namespace proof_F_4_f_5_l227_227557

def f (a : ℤ) : ℤ := a - 2

def F (a b : ℤ) : ℤ := a * b + b^2

theorem proof_F_4_f_5 :
  F 4 (f 5) = 21 := by
  sorry

end proof_F_4_f_5_l227_227557


namespace avg_b_c_is_45_l227_227694

-- Define the weights of a, b, and c
variables (a b c : ℝ)

-- Conditions given in the problem
def avg_a_b_c (a b c : ℝ) := (a + b + c) / 3 = 45
def avg_a_b (a b : ℝ) := (a + b) / 2 = 40
def weight_b (b : ℝ) := b = 35

-- Theorem statement
theorem avg_b_c_is_45 (a b c : ℝ) (h1 : avg_a_b_c a b c) (h2 : avg_a_b a b) (h3 : weight_b b) :
  (b + c) / 2 = 45 := by
  -- Proof omitted for brevity
  sorry

end avg_b_c_is_45_l227_227694


namespace quadratic_has_one_solution_implies_m_l227_227224

theorem quadratic_has_one_solution_implies_m (m : ℚ) :
  (∀ x : ℚ, 3 * x^2 - 7 * x + m = 0 → (b^2 - 4 * a * m = 0)) ↔ m = 49 / 12 :=
by
  sorry

end quadratic_has_one_solution_implies_m_l227_227224


namespace average_hidden_primes_l227_227446

theorem average_hidden_primes (x y z : ℕ) (hx : Nat.Prime x) (hy : Nat.Prime y) (hz : Nat.Prime z)
  (h_diff : x ≠ y ∧ y ≠ z ∧ x ≠ z) (h_sum : 44 + x = 59 + y ∧ 59 + y = 38 + z) :
  (x + y + z) / 3 = 14 := 
by
  sorry

end average_hidden_primes_l227_227446


namespace total_carrots_grown_l227_227146

theorem total_carrots_grown
  (Sandy_carrots : ℕ) (Sam_carrots : ℕ) (Sophie_carrots : ℕ) (Sara_carrots : ℕ)
  (h1 : Sandy_carrots = 6)
  (h2 : Sam_carrots = 3)
  (h3 : Sophie_carrots = 2 * Sam_carrots)
  (h4 : Sara_carrots = (Sandy_carrots + Sam_carrots + Sophie_carrots) - 5) :
  Sandy_carrots + Sam_carrots + Sophie_carrots + Sara_carrots = 25 :=
by sorry

end total_carrots_grown_l227_227146


namespace arithmetic_mean_of_a_and_b_is_sqrt3_l227_227654

theorem arithmetic_mean_of_a_and_b_is_sqrt3 :
  let a := (Real.sqrt 3 + Real.sqrt 2)
  let b := (Real.sqrt 3 - Real.sqrt 2)
  (a + b) / 2 = Real.sqrt 3 := 
by
  sorry

end arithmetic_mean_of_a_and_b_is_sqrt3_l227_227654


namespace number_of_combinations_l227_227133

-- Define the binomial coefficient (combinations) function
def C (n k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.choose n k else 0

-- Our main theorem statement
theorem number_of_combinations (n k m : ℕ) (h1 : 1 ≤ n) (h2 : m > 1) :
  let valid_combinations := C (n - (k - 1) * (m - 1)) k;
  let invalid_combinations := n - (k - 1) * m;
  valid_combinations - invalid_combinations = 
  C (n - (k - 1) * (m - 1)) k - (n - (k - 1) * m) := by
  let valid_combinations := C (n - (k - 1) * (m - 1)) k
  let invalid_combinations := n - (k - 1) * m
  sorry

end number_of_combinations_l227_227133


namespace difference_of_squares_l227_227246

theorem difference_of_squares : 
  let a := 625
  let b := 575
  (a^2 - b^2) = 60000 :=
by 
  let a := 625
  let b := 575
  sorry

end difference_of_squares_l227_227246


namespace remainder_is_4_l227_227179

-- Definitions based on the given conditions
def dividend := 132
def divisor := 16
def quotient := 8

-- The theorem we aim to prove, stating the remainder
theorem remainder_is_4 : dividend = divisor * quotient + 4 := sorry

end remainder_is_4_l227_227179


namespace lowest_total_points_l227_227622

-- Five girls and their respective positions
inductive Girl where
  | Fiona
  | Gertrude
  | Hannah
  | India
  | Janice
  deriving DecidableEq, Repr, Inhabited

open Girl

-- Initial position mapping
def initial_position : Girl → Nat
  | Fiona => 1
  | Gertrude => 2
  | Hannah => 3
  | India => 4
  | Janice => 5

-- Final position mapping
def final_position : Girl → Nat
  | Fiona => 3
  | Gertrude => 2
  | Hannah => 5
  | India => 1
  | Janice => 4

-- Define a function to calculate points for given initial and final positions
def points_awarded (g : Girl) : Nat :=
  initial_position g - final_position g

-- Define a function to calculate the total number of points
def total_points : Nat :=
  points_awarded Fiona + points_awarded Gertrude + points_awarded Hannah + points_awarded India + points_awarded Janice

theorem lowest_total_points : total_points = 5 :=
by
  -- Placeholder to skip the proof steps
  sorry

end lowest_total_points_l227_227622


namespace s_plus_t_l227_227877

def g (x : ℝ) : ℝ := 3 * x ^ 4 + 9 * x ^ 3 - 7 * x ^ 2 + 2 * x + 4
def h (x : ℝ) : ℝ := x ^ 2 + 2 * x - 1

noncomputable def s (x : ℝ) : ℝ := 3 * x ^ 2 + 3
noncomputable def t (x : ℝ) : ℝ := 3 * x + 6

theorem s_plus_t : s 1 + t (-1) = 9 := by
  sorry

end s_plus_t_l227_227877


namespace smallest_initial_number_sum_of_digits_l227_227681

theorem smallest_initial_number_sum_of_digits : ∃ (N : ℕ), 
  (0 ≤ N ∧ N < 1000) ∧ 
  ∃ (k : ℕ), 16 * N + 700 + 50 * k < 1000 ∧ 
  (N = 16) ∧ 
  (Nat.digits 10 N).sum = 7 := 
by
  sorry

end smallest_initial_number_sum_of_digits_l227_227681


namespace inequality_proof_l227_227870

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b + b * c + c * a ≥ 1) :
    (1 / a^2) + (1 / b^2) + (1 / c^2) ≥ (Real.sqrt 3) / (a * b * c) :=
by
  sorry

end inequality_proof_l227_227870


namespace girls_without_notebooks_l227_227599

noncomputable def girls_in_class : Nat := 20
noncomputable def students_with_notebooks : Nat := 25
noncomputable def boys_with_notebooks : Nat := 16

theorem girls_without_notebooks : 
  (girls_in_class - (students_with_notebooks - boys_with_notebooks)) = 11 := by
  sorry

end girls_without_notebooks_l227_227599


namespace closest_total_population_of_cities_l227_227463

theorem closest_total_population_of_cities 
    (n_cities : ℕ) (avg_population_lower avg_population_upper : ℕ)
    (h_lower : avg_population_lower = 3800) (h_upper : avg_population_upper = 4200) :
  (25:ℕ) * (4000:ℕ) = 100000 :=
by
  sorry

end closest_total_population_of_cities_l227_227463


namespace best_solved_completing_square_l227_227972

theorem best_solved_completing_square :
  ∀ (x : ℝ), x^2 - 2*x - 3 = 0 → (x - 1)^2 - 4 = 0 :=
sorry

end best_solved_completing_square_l227_227972


namespace simplify_expr1_simplify_expr2_l227_227374

-- Definition for the expression (2x - 3y)²
def expr1 (x y : ℝ) : ℝ := (2 * x - 3 * y) ^ 2

-- Theorem to prove that (2x - 3y)² = 4x² - 12xy + 9y²
theorem simplify_expr1 (x y : ℝ) : expr1 x y = 4 * (x ^ 2) - 12 * x * y + 9 * (y ^ 2) := 
sorry

-- Definition for the expression (x + y) * (x + y) * (x² + y²)
def expr2 (x y : ℝ) : ℝ := (x + y) * (x + y) * (x ^ 2 + y ^ 2)

-- Theorem to prove that (x + y) * (x + y) * (x² + y²) = x⁴ + 2x²y² + y⁴ + 2x³y + 2xy³
theorem simplify_expr2 (x y : ℝ) : expr2 x y = x ^ 4 + 2 * (x ^ 2) * (y ^ 2) + y ^ 4 + 2 * (x ^ 3) * y + 2 * x * (y ^ 3) := 
sorry

end simplify_expr1_simplify_expr2_l227_227374


namespace abs_sum_lt_ineq_l227_227912

theorem abs_sum_lt_ineq (x : ℝ) (a : ℝ) (h₀ : 0 < a) : 
  (∃ x : ℝ, |x - 4| + |x - 3| < a) ↔ (1 < a) :=
by
  sorry

end abs_sum_lt_ineq_l227_227912


namespace g_g_2_eq_394_l227_227871

def g (x : ℝ) : ℝ := 4 * x ^ 2 - 6

theorem g_g_2_eq_394 : g (g 2) = 394 :=
by
  sorry

end g_g_2_eq_394_l227_227871


namespace scientific_notation_14nm_l227_227175

theorem scientific_notation_14nm :
  0.000000014 = 1.4 * 10^(-8) := 
by 
  sorry

end scientific_notation_14nm_l227_227175


namespace difference_in_soda_bottles_l227_227517

def diet_soda_bottles : ℕ := 4
def regular_soda_bottles : ℕ := 83

theorem difference_in_soda_bottles :
  regular_soda_bottles - diet_soda_bottles = 79 :=
by
  sorry

end difference_in_soda_bottles_l227_227517


namespace distance_and_area_of_triangle_l227_227395

theorem distance_and_area_of_triangle :
  let p1 := (0, 6)
  let p2 := (8, 0)
  let origin := (0, 0)
  let distance := Real.sqrt ((8 - 0)^2 + (0 - 6)^2)
  let area := (1 / 2 : ℝ) * 8 * 6
  distance = 10 ∧ area = 24 :=
by
  let p1 := (0, 6)
  let p2 := (8, 0)
  let origin := (0, 0)
  let distance := Real.sqrt ((8 - 0)^2 + (0 - 6)^2)
  let area := (1 / 2 : ℝ) * 8 * 6
  have h_dist : distance = 10 := sorry
  have h_area : area = 24 := sorry
  exact ⟨h_dist, h_area⟩

end distance_and_area_of_triangle_l227_227395


namespace domain_of_sqrt_log_l227_227914

noncomputable def domain_of_function : Set ℝ := 
  {x : ℝ | (-Real.sqrt 2) ≤ x ∧ x < -1 ∨ 1 < x ∧ x ≤ Real.sqrt 2}

theorem domain_of_sqrt_log : ∀ x : ℝ, 
  (∃ y : ℝ, y = Real.sqrt (Real.log (x^2 - 1) / Real.log (1/2)) ∧ 
  y ≥ 0) ↔ x ∈ domain_of_function := 
by
  sorry

end domain_of_sqrt_log_l227_227914


namespace last_digit_B_l227_227615

theorem last_digit_B 
  (B : ℕ) 
  (h : ∀ n : ℕ, n % 10 = (B - 287)^2 % 10 → n % 10 = 4) :
  (B = 5 ∨ B = 9) :=
sorry

end last_digit_B_l227_227615


namespace find_a_l227_227283

variables {a b c : ℂ}

-- Given conditions
variables (h1 : a + b + c = 5) 
variables (h2 : a * b + b * c + c * a = 5) 
variables (h3 : a * b * c = 5)
variables (h4 : a.im = 0) -- a is real

theorem find_a : a = 4 :=
by
  sorry

end find_a_l227_227283


namespace preston_charges_5_dollars_l227_227111

def cost_per_sandwich (x : Real) : Prop :=
  let number_of_sandwiches := 18
  let delivery_fee := 20
  let tip_percentage := 0.10
  let total_received := 121
  let total_cost := number_of_sandwiches * x + delivery_fee
  let tip := tip_percentage * total_cost
  let final_amount := total_cost + tip
  final_amount = total_received

theorem preston_charges_5_dollars :
  ∀ x : Real, cost_per_sandwich x → x = 5 :=
by
  intros x h
  sorry

end preston_charges_5_dollars_l227_227111


namespace trig_expression_value_l227_227026

theorem trig_expression_value (α : ℝ) (h : Real.tan α = 2) :
  (2 * Real.sin α - Real.cos α) / (Real.sin α + 2 * Real.cos α) = 3 / 4 :=
by
  sorry

end trig_expression_value_l227_227026


namespace triangle_perimeter_l227_227857

-- Given conditions
def inradius : ℝ := 2.5
def area : ℝ := 40

-- The formula relating inradius, area, and perimeter
def perimeter_formula (r a p : ℝ) : Prop := a = r * p / 2

-- Prove the perimeter p of the triangle
theorem triangle_perimeter : ∃ (p : ℝ), perimeter_formula inradius area p ∧ p = 32 := by
  sorry

end triangle_perimeter_l227_227857


namespace sum_reciprocal_geo_seq_l227_227294

theorem sum_reciprocal_geo_seq {a_5 a_6 a_7 a_8 : ℝ}
  (h_sum : a_5 + a_6 + a_7 + a_8 = 15 / 8)
  (h_prod : a_6 * a_7 = -9 / 8) :
  (1 / a_5) + (1 / a_6) + (1 / a_7) + (1 / a_8) = -5 / 3 := by
  sorry

end sum_reciprocal_geo_seq_l227_227294


namespace total_fruits_is_78_l227_227818

def oranges_louis : Nat := 5
def apples_louis : Nat := 3

def oranges_samantha : Nat := 8
def apples_samantha : Nat := 7

def oranges_marley : Nat := 2 * oranges_louis
def apples_marley : Nat := 3 * apples_samantha

def oranges_edward : Nat := 3 * oranges_louis
def apples_edward : Nat := 3 * apples_louis

def total_fruits_louis : Nat := oranges_louis + apples_louis
def total_fruits_samantha : Nat := oranges_samantha + apples_samantha
def total_fruits_marley : Nat := oranges_marley + apples_marley
def total_fruits_edward : Nat := oranges_edward + apples_edward

def total_fruits_all : Nat :=
  total_fruits_louis + total_fruits_samantha + total_fruits_marley + total_fruits_edward

theorem total_fruits_is_78 : total_fruits_all = 78 := by
  sorry

end total_fruits_is_78_l227_227818


namespace fractional_equation_solution_l227_227859

theorem fractional_equation_solution (x : ℝ) (h₁ : x ≠ 0) : (1 / x = 2 / (x + 3)) → x = 3 := by
  sorry

end fractional_equation_solution_l227_227859


namespace initial_puppies_count_l227_227968

theorem initial_puppies_count (P : ℕ) (h1 : P - 2 + 3 = 8) : P = 7 :=
sorry

end initial_puppies_count_l227_227968


namespace proposition_equivalence_l227_227880

open Classical

theorem proposition_equivalence
  (p q : Prop) :
  ¬(p ∨ q) ↔ (¬p ∧ ¬q) :=
by sorry

end proposition_equivalence_l227_227880


namespace race_head_start_l227_227387

variables {Va Vb L H : ℝ}

theorem race_head_start
  (h1 : Va = 20 / 14 * Vb)
  (h2 : L / Va = (L - H) / Vb) : 
  H = 3 / 10 * L :=
by
  sorry

end race_head_start_l227_227387


namespace grading_combinations_l227_227608

/-- There are 12 students in the class. -/
def num_students : ℕ := 12

/-- There are 4 possible grades (A, B, C, and D). -/
def num_grades : ℕ := 4

/-- The total number of ways to assign grades. -/
theorem grading_combinations : (num_grades ^ num_students) = 16777216 := 
by
  sorry

end grading_combinations_l227_227608


namespace prob_select_math_books_l227_227660

theorem prob_select_math_books :
  let total_books := 5
  let math_books := 3
  let total_ways_select_2 := Nat.choose total_books 2
  let ways_select_2_math := Nat.choose math_books 2
  let probability := (ways_select_2_math : ℚ) / total_ways_select_2
  probability = 3 / 10 :=
by
  sorry

end prob_select_math_books_l227_227660


namespace lisa_phone_spending_l227_227727

variable (cost_phone : ℕ) (cost_contract_per_month : ℕ) (case_percentage : ℕ) (headphones_ratio : ℕ)

/-- Given the cost of the phone, the monthly contract cost, 
    the percentage cost of the case, and ratio cost of headphones,
    prove that the total spending in the first year is correct.
-/ 
theorem lisa_phone_spending 
    (h_cost_phone : cost_phone = 1000) 
    (h_cost_contract_per_month : cost_contract_per_month = 200) 
    (h_case_percentage : case_percentage = 20)
    (h_headphones_ratio : headphones_ratio = 2) :
    cost_phone + (cost_phone * case_percentage / 100) + 
    ((cost_phone * case_percentage / 100) / headphones_ratio) + 
    (cost_contract_per_month * 12) = 3700 :=
by
  sorry

end lisa_phone_spending_l227_227727


namespace num_two_digit_prime_with_units_digit_3_eq_6_l227_227886

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def units_digit_is_3 (n : ℕ) : Prop := n % 10 = 3

def two_digit_number (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

theorem num_two_digit_prime_with_units_digit_3_eq_6 :
  ∃ S : Finset ℕ, (∀ n ∈ S, is_prime n ∧ units_digit_is_3 n ∧ two_digit_number n) ∧ S.card = 6 :=
by
  sorry

end num_two_digit_prime_with_units_digit_3_eq_6_l227_227886


namespace number_of_members_after_four_years_l227_227668

theorem number_of_members_after_four_years (b : ℕ → ℕ) (initial_condition : b 0 = 21) 
    (yearly_update : ∀ k, b (k + 1) = 4 * b k - 9) : 
    b 4 = 4611 :=
    sorry

end number_of_members_after_four_years_l227_227668


namespace length_of_bridge_l227_227511

noncomputable def speed_kmhr_to_ms (v : ℕ) : ℝ := (v : ℝ) * (1000 / 3600)

noncomputable def distance_traveled (v : ℝ) (t : ℕ) : ℝ := v * (t : ℝ)

theorem length_of_bridge 
  (length_train : ℕ) -- 90 meters
  (speed_train_kmhr : ℕ) -- 45 km/hr
  (time_cross_bridge : ℕ) -- 30 seconds
  (conversion_factor : ℝ := 1000 / 3600) 
  : ℝ := 
  let speed_train_ms := speed_kmhr_to_ms speed_train_kmhr
  let total_distance := distance_traveled speed_train_ms time_cross_bridge
  total_distance - (length_train : ℝ)

example : length_of_bridge 90 45 30 = 285 := by
  sorry

end length_of_bridge_l227_227511


namespace hardcover_books_count_l227_227242

theorem hardcover_books_count
  (h p : ℕ)
  (h_plus_p_eq_10 : h + p = 10)
  (total_cost_eq_250 : 30 * h + 20 * p = 250) :
  h = 5 :=
by
  sorry

end hardcover_books_count_l227_227242


namespace problem_statement_l227_227020

theorem problem_statement {f : ℝ → ℝ}
  (Hodd : ∀ x, f (-x) = -f x)
  (Hdecreasing : ∀ x y, x < y → f x > f y)
  (a b : ℝ) (H : f a + f b > 0) : a + b < 0 :=
sorry

end problem_statement_l227_227020


namespace math_proof_problem_l227_227150

noncomputable def find_value (a b c : ℝ) : ℝ :=
  (a^3 + b^3 + c^3) / (a * b * c * (a * b + a * c + b * c))

theorem math_proof_problem (a b c : ℝ)
  (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : a + b + c = 0) (h5 : a * b + a * c + b * c ≠ 0) :
  find_value a b c = 3 :=
by 
  -- sorry is used as we are only asked to provide the theorem statement in Lean.
  sorry

end math_proof_problem_l227_227150


namespace gcd_20020_11011_l227_227194

theorem gcd_20020_11011 : Nat.gcd 20020 11011 = 1001 := 
by
  sorry

end gcd_20020_11011_l227_227194


namespace birth_year_l227_227548

theorem birth_year (x : ℤ) (h : 1850 < x^2 - 10 - x ∧ 1849 ≤ x^2 - 10 - x ∧ x^2 - 10 - x ≤ 1880) : 
x^2 - 10 - x ≠ 1849 ∧ x^2 - 10 - x ≠ 1855 ∧ x^2 - 10 - x ≠ 1862 ∧ x^2 - 10 - x ≠ 1871 ∧ x^2 - 10 - x ≠ 1880 := 
sorry

end birth_year_l227_227548


namespace range_of_m_l227_227451

def G (x y : ℤ) : ℤ :=
  if x ≥ y then x - y
  else y - x

theorem range_of_m (m : ℤ) :
  (∀ x, 0 < x → G x 1 > 4 → G (-1) x ≤ m) ↔ 9 ≤ m ∧ m < 10 :=
sorry

end range_of_m_l227_227451


namespace friend_spent_l227_227928

theorem friend_spent (x you friend total: ℝ) (h1 : total = you + friend) (h2 : friend = you + 3) (h3 : total = 11) : friend = 7 := by
  sorry

end friend_spent_l227_227928


namespace total_allocation_is_1800_l227_227811

-- Definitions from conditions.
def part_value (amount_food : ℕ) (ratio_food : ℕ) : ℕ :=
  amount_food / ratio_food

def total_parts (ratio_household : ℕ) (ratio_food : ℕ) (ratio_misc : ℕ) : ℕ :=
  ratio_household + ratio_food + ratio_misc

def total_amount (part_value : ℕ) (total_parts : ℕ) : ℕ :=
  part_value * total_parts

-- Given conditions
def ratio_household := 5
def ratio_food := 4
def ratio_misc := 1
def amount_food := 720

-- Prove the total allocation
theorem total_allocation_is_1800 
  (amount_food : ℕ := 720) 
  (ratio_household : ℕ := 5) 
  (ratio_food : ℕ := 4) 
  (ratio_misc : ℕ := 1) : 
  total_amount (part_value amount_food ratio_food) (total_parts ratio_household ratio_food ratio_misc) = 1800 :=
by
  sorry

end total_allocation_is_1800_l227_227811


namespace total_area_of_field_l227_227056

theorem total_area_of_field 
  (A_s : ℕ) 
  (h₁ : A_s = 315)
  (A_l : ℕ) 
  (h₂ : A_l - A_s = (1/5) * ((A_s + A_l) / 2)) : 
  A_s + A_l = 700 := 
  by 
    sorry

end total_area_of_field_l227_227056


namespace solve_inequality_l227_227757

theorem solve_inequality :
  ∀ x : ℝ, (3 * x^2 - 4 * x - 7 < 0) ↔ (-1 < x ∧ x < 7 / 3) :=
by
  sorry

end solve_inequality_l227_227757


namespace find_largest_n_l227_227910

theorem find_largest_n 
  (a : ℕ → ℕ) (b : ℕ → ℕ)
  (x y : ℕ)
  (h_a1 : a 1 = 1)
  (h_b1 : b 1 = 1)
  (h_arith_a : ∀ n : ℕ, a n = 1 + (n - 1) * x)
  (h_arith_b : ∀ n : ℕ, b n = 1 + (n - 1) * y)
  (h_order : x ≤ y)
  (h_product : ∃ n : ℕ, a n * b n = 4021) :
  ∃ n : ℕ, a n * b n = 4021 ∧ n ≤ 11 := 
by
  sorry

end find_largest_n_l227_227910


namespace melanie_attended_games_l227_227076

-- Define the total number of football games and the number of games missed by Melanie.
def total_games := 7
def missed_games := 4

-- Define what we need to prove: the number of games attended by Melanie.
theorem melanie_attended_games : total_games - missed_games = 3 := 
by
  sorry

end melanie_attended_games_l227_227076


namespace number_of_players_l227_227074

/-- Jane bought 600 minnows, each prize has 3 minnows, 15% of the players win a prize, 
and 240 minnows are left over. To find the total number of players -/
theorem number_of_players (total_minnows left_over_minnows minnows_per_prize prizes_win_percent : ℕ) 
(h1 : total_minnows = 600) 
(h2 : minnows_per_prize = 3)
(h3 : prizes_win_percent * 100 = 15)
(h4 : left_over_minnows = 240) : 
total_minnows - left_over_minnows = 360 → 
  360 / minnows_per_prize = 120 → 
  (prizes_win_percent * 100 / 100) * P = 120 → 
  P = 800 := 
by 
  sorry

end number_of_players_l227_227074


namespace boat_speed_in_still_water_l227_227049

/--
The speed of the stream is 6 kmph.
The boat can cover 48 km downstream or 32 km upstream in the same time.
We want to prove that the speed of the boat in still water is 30 kmph.
-/
theorem boat_speed_in_still_water (x : ℝ)
  (h1 : ∃ t : ℝ, t = 48 / (x + 6) ∧ t = 32 / (x - 6)) : x = 30 :=
by
  sorry

end boat_speed_in_still_water_l227_227049


namespace problem_l227_227419

theorem problem (a b : ℝ) (h₁ : a = -a) (h₂ : b = 1 / b) : a + b = 1 ∨ a + b = -1 :=
  sorry

end problem_l227_227419


namespace problem1_problem2_l227_227191

-- Problem 1
theorem problem1 : ∀ x : ℝ, 4 * x - 3 * (20 - x) + 4 = 0 → x = 8 :=
by
  intro x
  intro h
  sorry

-- Problem 2
theorem problem2 : ∀ x : ℝ, (2 * x + 1) / 3 = 1 - (x - 1) / 5 → x = 1 :=
by
  intro x
  intro h
  sorry

end problem1_problem2_l227_227191


namespace probability_red_ball_is_two_thirds_red_balls_taken_out_is_three_l227_227998

-- Definitions for the conditions
def total_balls : ℕ := 18
def initial_red_balls : ℕ := 12
def initial_white_balls : ℕ := 6
def probability_red_ball : ℚ := initial_red_balls / total_balls
def probability_white_ball_after_removal (x : ℕ) : ℚ := initial_white_balls / (total_balls - x)

-- Statement of the proof problem
theorem probability_red_ball_is_two_thirds : probability_red_ball = 2 / 3 := 
by sorry

theorem red_balls_taken_out_is_three : ∃ x : ℕ, probability_white_ball_after_removal x = 2 / 5 ∧ x = 3 := 
by sorry

end probability_red_ball_is_two_thirds_red_balls_taken_out_is_three_l227_227998


namespace ladder_base_distance_l227_227311

theorem ladder_base_distance (h l : ℝ) (h_eq : h = 13) (l_eq : l = 12) :
  ∃ b : ℝ, (h^2 = l^2 + b^2) ∧ b = 5 :=
by
  sorry

end ladder_base_distance_l227_227311


namespace range_of_ab_l227_227471

noncomputable def range_ab : Set ℝ := 
  { x | 4 ≤ x ∧ x ≤ 112 / 9 }

theorem range_of_ab (a b : ℝ) 
  (q : ℝ) (h1 : q ∈ (Set.Icc (1/3) 2)) 
  (h2 : ∃ m : ℝ, ∃ nq : ℕ, 
    (m * q ^ nq) * m ^ (2 - nq) = 1 ∧ 
    (m + m * q ^ nq) = a ∧ 
    (m * q + m * q ^ 2) = b):
  ab = (q + 1/q + q^2 + 1/q^2) → 
  (ab ∈ range_ab) := 
by 
  sorry

end range_of_ab_l227_227471


namespace allan_balloons_l227_227734

def initial_balloons : ℕ := 5
def additional_balloons : ℕ := 3
def total_balloons : ℕ := initial_balloons + additional_balloons

theorem allan_balloons :
  total_balloons = 8 :=
sorry

end allan_balloons_l227_227734


namespace prove_function_domain_l227_227581

def function_domain := {x : ℝ | (x + 4 ≥ 0 ∧ x ≠ 0)}

theorem prove_function_domain :
  function_domain = {x : ℝ | x ∈ (Set.Icc (-4:ℝ) 0).diff ({0}:Set ℝ) ∪ (Set.Ioi 0)} :=
by
  sorry

end prove_function_domain_l227_227581


namespace sum_of_two_smallest_l227_227413

variable (a b c d : ℕ)
variable (x : ℕ)

-- Four numbers a, b, c, d are in the ratio 3:5:7:9
def ratios := (a = 3 * x) ∧ (b = 5 * x) ∧ (c = 7 * x) ∧ (d = 9 * x)

-- The average of these numbers is 30
def average := (a + b + c + d) / 4 = 30

-- The theorem to prove the sum of the two smallest numbers (a and b) is 40
theorem sum_of_two_smallest (h1 : ratios a b c d x) (h2 : average a b c d) : a + b = 40 := by
  sorry

end sum_of_two_smallest_l227_227413


namespace polynomial_bound_l227_227218

noncomputable def P (a b c d : ℝ) (x : ℝ) : ℝ :=
  a * x^3 + b * x^2 + c * x + d

theorem polynomial_bound (a b c d : ℝ) (h : ∀ x : ℝ, |x| < 1 → |P a b c d x| ≤ 1) :
  |a| + |b| + |c| + |d| ≤ 7 :=
sorry

end polynomial_bound_l227_227218


namespace intersection_A_B_l227_227007

-- Definitions of sets A and B based on the given conditions
def A : Set ℕ := {4, 5, 6, 7}
def B : Set ℕ := {x | 3 ≤ x ∧ x < 6}

-- The theorem stating the proof problem
theorem intersection_A_B : A ∩ B = {4, 5} :=
by
  sorry

end intersection_A_B_l227_227007


namespace remainder_of_division_l227_227364

theorem remainder_of_division :
  ∃ R : ℕ, 176 = (19 * 9) + R ∧ R = 5 :=
by
  sorry

end remainder_of_division_l227_227364


namespace subtraction_addition_example_l227_227107

theorem subtraction_addition_example :
  1500000000000 - 877888888888 + 123456789012 = 745567900124 :=
by
  sorry

end subtraction_addition_example_l227_227107


namespace solve_system_of_equations_l227_227397

theorem solve_system_of_equations (x y : ℝ) :
    (5 * x * (1 + 1 / (x^2 + y^2)) = 12 ∧ 5 * y * (1 - 1 / (x^2 + y^2)) = 4) ↔
    (x = 2 ∧ y = 1) ∨ (x = 2 / 5 ∧ y = -(1 / 5)) :=
by
  sorry

end solve_system_of_equations_l227_227397


namespace common_real_root_pair_l227_227641

theorem common_real_root_pair (n : ℕ) (hn : n > 1) :
  ∃ x : ℝ, (∃ a b : ℤ, ((x^n + (a : ℝ) * x = 2008) ∧ (x^n + (b : ℝ) * x = 2009))) ↔
    ((a = 2007 ∧ b = 2008) ∨
     (a = (-1)^(n-1) - 2008 ∧ b = (-1)^(n-1) - 2009)) :=
by sorry

end common_real_root_pair_l227_227641


namespace crayon_colors_correct_l227_227238

-- The Lean code will define the conditions and the proof statement as follows:
noncomputable def crayon_problem := 
  let crayons_per_box := (160 / (5 * 4)) -- Total crayons / Total boxes
  let colors := (crayons_per_box / 2) -- Crayons per box / Crayons per color
  colors = 4

-- This is the theorem that needs to be proven:
theorem crayon_colors_correct : crayon_problem := by
  sorry

end crayon_colors_correct_l227_227238


namespace find_m_value_l227_227664

theorem find_m_value (m a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℝ)
  (h1 : (x + m)^9 = a_0 + a_1 * (x + 1) + a_2 * (x + 1)^2 + a_3 * (x + 1)^3 + 
  a_4 * (x + 1)^4 + a_5 * (x + 1)^5 + a_6 * (x + 1)^6 + a_7 * (x + 1)^7 + 
  a_8 * (x + 1)^8 + a_9 * (x + 1)^9)
  (h2 : a_0 - a_1 + a_2 - a_3 + a_4 - a_5 + a_6 - a_7 + a_8 - a_9 = 3^9) :
  m = 4 :=
by
  sorry

end find_m_value_l227_227664


namespace exact_time_between_9_10_l227_227964

theorem exact_time_between_9_10
  (t : ℝ)
  (h1 : 0 ≤ t ∧ t < 60)
  (h2 : |6 * (t + 5) - (270 + 0.5 * (t - 2))| = 180) :
  t = 10 + 3 / 4 :=
sorry

end exact_time_between_9_10_l227_227964


namespace find_n_l227_227508

theorem find_n (n : ℕ) (h : n * n.factorial + n.factorial = 720) : n = 5 :=
sorry

end find_n_l227_227508


namespace compute_9_times_one_seventh_pow_4_l227_227375

theorem compute_9_times_one_seventh_pow_4 :
  9 * (1 / 7) ^ 4 = 9 / 2401 :=
by
  -- The actual proof would go here.
  sorry

end compute_9_times_one_seventh_pow_4_l227_227375


namespace extreme_values_range_of_a_l227_227771

noncomputable def f (x : ℝ) := x^2 * Real.exp x
noncomputable def y (x : ℝ) (a : ℝ) := f x - a * x

theorem extreme_values :
  ∃ x_max x_min,
    (x_max = -2 ∧ f x_max = 4 / Real.exp 2) ∧
    (x_min = 0 ∧ f x_min = 0) := sorry

theorem range_of_a (a : ℝ) :
  (∃ x₁ x₂, x₁ ≠ 0 ∧ x₂ ≠ 0 ∧ y x₁ a = 0 ∧ y x₂ a = 0) ↔
  -1 / Real.exp 1 < a ∧ a < 0 := sorry

end extreme_values_range_of_a_l227_227771


namespace bucket_water_l227_227259

theorem bucket_water (oz1 oz2 oz3 oz4 oz5 total1 total2: ℕ) 
  (h1 : oz1 = 11)
  (h2 : oz2 = 13)
  (h3 : oz3 = 12)
  (h4 : oz4 = 16)
  (h5 : oz5 = 10)
  (h_total : total1 = oz1 + oz2 + oz3 + oz4 + oz5)
  (h_second_bucket : total2 = 39)
  : total1 - total2 = 23 :=
sorry

end bucket_water_l227_227259


namespace number_of_players_l227_227350

theorem number_of_players (S : ℕ) (h1 : S = 22) (h2 : ∀ (n : ℕ), S = n * 2) : ∃ n, n = 11 :=
by
  sorry

end number_of_players_l227_227350


namespace largest_y_l227_227638

def interior_angle (n : ℕ) : ℚ := (n - 2) * 180 / n

theorem largest_y (x y : ℕ) (hx : x ≥ y) (hy : y ≥ 3) 
  (h : (interior_angle x * 28) = (interior_angle y * 29)) :
  y = 57 :=
by
  sorry

end largest_y_l227_227638


namespace N_eq_P_l227_227922

def N : Set ℝ := {x | ∃ n : ℤ, x = (n : ℝ) / 2 - 1 / 3}
def P : Set ℝ := {x | ∃ p : ℤ, x = (p : ℝ) / 2 + 1 / 6}

theorem N_eq_P : N = P :=
  sorry

end N_eq_P_l227_227922


namespace probability_of_number_between_21_and_30_l227_227392

-- Define the success condition of forming a two-digit number between 21 and 30.
def successful_number (d1 d2 : Nat) : Prop :=
  let n1 := 10 * d1 + d2
  let n2 := 10 * d2 + d1
  (21 ≤ n1 ∧ n1 ≤ 30) ∨ (21 ≤ n2 ∧ n2 ≤ 30)

-- Calculate the probability of a successful outcome.
def probability_success (favorable total : Nat) : Nat :=
  favorable / total

-- The main theorem claiming the probability that Melinda forms a number between 21 and 30.
theorem probability_of_number_between_21_and_30 :
  let successful_counts := 10
  let total_possible := 36
  probability_success successful_counts total_possible = 5 / 18 :=
by
  sorry

end probability_of_number_between_21_and_30_l227_227392


namespace find_z_l227_227145

theorem find_z (x y z : ℝ) (h : 1 / (x + 1) + 1 / (y + 1) = 1 / z) :
  z = (x + 1) * (y + 1) / (x + y + 2) :=
sorry

end find_z_l227_227145


namespace cost_of_500_cookies_in_dollars_l227_227824

def cost_in_cents (cookies : Nat) (cost_per_cookie : Nat) : Nat :=
  cookies * cost_per_cookie

def cents_to_dollars (cents : Nat) : Nat :=
  cents / 100

theorem cost_of_500_cookies_in_dollars :
  cents_to_dollars (cost_in_cents 500 2) = 10
:= by
  sorry

end cost_of_500_cookies_in_dollars_l227_227824


namespace hyperbola_asymptote_m_l227_227328

def isAsymptote (x y : ℝ) (m : ℝ) : Prop :=
  y = m * x ∨ y = -m * x

theorem hyperbola_asymptote_m (m : ℝ) : 
  (∀ x y, (x^2 / 25 - y^2 / 16 = 1 → isAsymptote x y m)) ↔ m = 4 / 5 := 
by
  sorry

end hyperbola_asymptote_m_l227_227328


namespace lamps_on_after_n2_minus_1_lamps_on_after_n2_minus_n_plus_1_l227_227797

def lamps_on_again (n : ℕ) (steps : ℕ → Bool → Bool) : ∃ M : ℕ, ∀ s, (s ≥ M) → (n > 1 → ∀ i : ℕ, steps i true = true) := 
sorry

theorem lamps_on_after_n2_minus_1 (n : ℕ) (k : ℕ) (hk : n = 2^k) (steps : ℕ → Bool → Bool) : 
∀ s, (s ≥ n^2 - 1) → (n > 1 → ∀ i : ℕ, steps i true = true) := 
sorry

theorem lamps_on_after_n2_minus_n_plus_1 (n : ℕ) (k : ℕ) (hk : n = 2^k + 1) (steps : ℕ → Bool → Bool) : 
∀ s, (s ≥ n^2 - n + 1) → (n > 1 → ∀ i : ℕ, steps i true = true) := 
sorry

end lamps_on_after_n2_minus_1_lamps_on_after_n2_minus_n_plus_1_l227_227797


namespace inequality_proof_l227_227399

theorem inequality_proof
  (x y z : ℝ)
  (hx : x > y)
  (hy : y > 1)
  (hz : 1 > z)
  (hzpos : z > 0)
  (a : ℝ := (1 + x * z) / z)
  (b : ℝ := (1 + x * y) / x)
  (c : ℝ := (1 + y * z) / y) :
  a > b ∧ a > c :=
by
  sorry

end inequality_proof_l227_227399


namespace bottle_caps_total_l227_227432

-- Mathematical conditions
def x : ℕ := 18
def y : ℕ := 63

-- Statement to prove
theorem bottle_caps_total : x + y = 81 :=
by
  -- The proof is skipped as indicated by 'sorry'
  sorry

end bottle_caps_total_l227_227432


namespace find_chosen_number_l227_227025

theorem find_chosen_number (x : ℤ) (h : 2 * x - 138 = 106) : x = 122 :=
by
  sorry

end find_chosen_number_l227_227025


namespace seated_students_count_l227_227929

theorem seated_students_count :
  ∀ (S T standing_students total_attendees : ℕ),
    T = 30 →
    standing_students = 25 →
    total_attendees = 355 →
    total_attendees = S + T + standing_students →
    S = 300 :=
by
  intros S T standing_students total_attendees hT hStanding hTotalAttendees hEquation
  sorry

end seated_students_count_l227_227929


namespace sequence_2010_eq_4040099_l227_227421

def sequence_term (n : Nat) : Int :=
  if n % 2 = 0 then 
    (n^2 - 1 : Int) 
  else 
    -(n^2 - 1 : Int)

theorem sequence_2010_eq_4040099 : sequence_term 2010 = 4040099 := 
  by 
    sorry

end sequence_2010_eq_4040099_l227_227421


namespace range_of_m_l227_227225

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, x ≤ 3 → (x ≤ m → (x < y → y < m))) → m ≥ 3 := 
by
  sorry

end range_of_m_l227_227225


namespace problem_statement_l227_227068

theorem problem_statement (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  (1 / Real.sqrt (2011 + Real.sqrt (2011^2 - 1)) = Real.sqrt m - Real.sqrt n) →
  m + n = 2011 :=
sorry

end problem_statement_l227_227068


namespace perimeter_of_rectangle_l227_227184

theorem perimeter_of_rectangle (L W : ℝ) (h1 : L / W = 5 / 2) (h2 : L * W = 4000) : 2 * L + 2 * W = 280 :=
sorry

end perimeter_of_rectangle_l227_227184


namespace total_boys_in_groups_l227_227447

-- Definitions of number of groups
def total_groups : ℕ := 35
def groups_with_1_boy : ℕ := 10
def groups_with_at_least_2_boys : ℕ := 19
def groups_with_3_boys_twice_groups_with_3_girls (groups_with_3_boys groups_with_3_girls : ℕ) : Prop :=
  groups_with_3_boys = 2 * groups_with_3_girls

theorem total_boys_in_groups :
  ∃ (groups_with_3_girls groups_with_3_boys groups_with_1_girl_2_boys : ℕ),
    groups_with_1_boy + groups_with_at_least_2_boys + groups_with_3_girls = total_groups
    ∧ groups_with_3_boys_twice_groups_with_3_girls groups_with_3_boys groups_with_3_girls
    ∧ groups_with_1_girl_2_boys + groups_with_3_boys = groups_with_at_least_2_boys
    ∧ (groups_with_1_boy * 1 + groups_with_1_girl_2_boys * 2 + groups_with_3_boys * 3) = 60 :=
sorry

end total_boys_in_groups_l227_227447


namespace prob_divisible_by_5_l227_227575

theorem prob_divisible_by_5 (M: ℕ) (h1: 100 ≤ M ∧ M < 1000) (h2: M % 10 = 5): 
  (∃ (k: ℕ), M = 5 * k) :=
by
  sorry

end prob_divisible_by_5_l227_227575


namespace ned_defuse_time_l227_227103

theorem ned_defuse_time (flights_total time_per_flight bomb_time time_spent : ℕ) (h1 : flights_total = 20) (h2 : time_per_flight = 11) (h3 : bomb_time = 72) (h4 : time_spent = 165) :
  bomb_time - (flights_total * time_per_flight - time_spent) / time_per_flight * time_per_flight = 17 := by
  sorry

end ned_defuse_time_l227_227103


namespace domain_of_v_l227_227851

noncomputable def v (x : ℝ) : ℝ := 1 / (x ^ (1/3) + x^2 - 1)

theorem domain_of_v : ∀ x, x ≠ 1 → x ^ (1/3) + x^2 - 1 ≠ 0 :=
by
  sorry

end domain_of_v_l227_227851


namespace initial_people_in_elevator_l227_227627

theorem initial_people_in_elevator (W n : ℕ) (avg_initial_weight avg_new_weight new_person_weight : ℚ)
  (h1 : avg_initial_weight = 152)
  (h2 : avg_new_weight = 151)
  (h3 : new_person_weight = 145)
  (h4 : W = n * avg_initial_weight)
  (h5 : W + new_person_weight = (n + 1) * avg_new_weight) :
  n = 6 :=
by
  sorry

end initial_people_in_elevator_l227_227627


namespace number_of_cubes_with_at_least_two_faces_painted_is_56_l227_227254

def one_inch_cubes_with_at_least_two_faces_painted 
  (side_length : ℕ) (face_colors : ℕ) (cubes_per_side : ℕ) :=
  if side_length = 4 ∧ face_colors = 6 ∧ cubes_per_side = 1 then 56 else 0

theorem number_of_cubes_with_at_least_two_faces_painted_is_56 :
  one_inch_cubes_with_at_least_two_faces_painted 4 6 1 = 56 :=
by
  sorry

end number_of_cubes_with_at_least_two_faces_painted_is_56_l227_227254


namespace total_money_spent_l227_227697

theorem total_money_spent (emma_spent : ℤ) (elsa_spent : ℤ) (elizabeth_spent : ℤ) 
(emma_condition : emma_spent = 58) 
(elsa_condition : elsa_spent = 2 * emma_spent) 
(elizabeth_condition : elizabeth_spent = 4 * elsa_spent) 
:
emma_spent + elsa_spent + elizabeth_spent = 638 :=
by
  rw [emma_condition, elsa_condition, elizabeth_condition]
  norm_num
  sorry

end total_money_spent_l227_227697


namespace flower_beds_fraction_l227_227234

noncomputable def isosceles_right_triangle_area (leg : ℝ) : ℝ :=
  (1 / 2) * leg^2

noncomputable def fraction_of_yard_occupied_by_flower_beds : ℝ :=
  let leg_length := (25 - 15) / 2
  let area_of_one_triangle := isosceles_right_triangle_area leg_length
  let total_area_of_flower_beds := 2 * area_of_one_triangle
  let area_of_yard := 25 * 5
  total_area_of_flower_beds / area_of_yard

theorem flower_beds_fraction : 
  let leg_length := (25 - 15) / 2
  let area_of_one_triangle := isosceles_right_triangle_area leg_length
  let total_area_of_flower_beds := 2 * area_of_one_triangle
  let area_of_yard := 25 * 5
  total_area_of_flower_beds / area_of_yard = 1 / 5 :=
by
  sorry

end flower_beds_fraction_l227_227234


namespace grade_assignment_ways_l227_227448

theorem grade_assignment_ways : (4 ^ 12) = 16777216 :=
by
  -- mathematical proof
  sorry

end grade_assignment_ways_l227_227448


namespace y_work_time_l227_227717

noncomputable def total_work := 1 

noncomputable def work_rate_x := 1 / 40
noncomputable def work_x_in_8_days := 8 * work_rate_x
noncomputable def remaining_work := total_work - work_x_in_8_days

noncomputable def work_rate_y := remaining_work / 36

theorem y_work_time :
  (1 / work_rate_y) = 45 :=
by
  sorry

end y_work_time_l227_227717


namespace teacher_total_score_l227_227371

variable (written_score : ℕ)
variable (interview_score : ℕ)
variable (weight_written : ℝ)
variable (weight_interview : ℝ)

theorem teacher_total_score :
  (written_score = 80) → (interview_score = 60) → (weight_written = 0.6) → (weight_interview = 0.4) →
  (written_score * weight_written + interview_score * weight_interview = 72) :=
by
  sorry

end teacher_total_score_l227_227371


namespace find_temperature_l227_227118

theorem find_temperature 
  (temps : List ℤ)
  (h_len : temps.length = 8)
  (h_mean : (temps.sum / 8 : ℝ) = -0.5)
  (h_temps : temps = [-6, -3, x, -6, 2, 4, 3, 0]) : 
  x = 2 :=
by 
  sorry

end find_temperature_l227_227118


namespace lead_amount_in_mixture_l227_227579

theorem lead_amount_in_mixture 
  (W : ℝ) 
  (h_copper : 0.60 * W = 12) 
  (h_mixture_composition : (0.15 * W = 0.15 * W) ∧ (0.25 * W = 0.25 * W) ∧ (0.60 * W = 0.60 * W)) :
  (0.25 * W = 5) :=
by
  sorry

end lead_amount_in_mixture_l227_227579


namespace cube_volume_fourth_power_l227_227322

theorem cube_volume_fourth_power (s : ℝ) (h : 6 * s^2 = 864) : s^4 = 20736 :=
sorry

end cube_volume_fourth_power_l227_227322


namespace number_of_frames_bought_l227_227754

/- 
   Define the problem conditions:
   1. Each photograph frame costs 3 dollars.
   2. Sally paid with a 20 dollar bill.
   3. Sally got 11 dollars in change.
-/ 

def frame_cost : Int := 3
def initial_payment : Int := 20
def change_received : Int := 11

/- 
   Prove that the number of photograph frames Sally bought is 3.
-/

theorem number_of_frames_bought : (initial_payment - change_received) / frame_cost = 3 := 
by
  sorry

end number_of_frames_bought_l227_227754


namespace angle_F_measure_l227_227078

theorem angle_F_measure (D E F : ℝ) (h₁ : D = 80) (h₂ : E = 2 * F + 24) (h₃ : D + E + F = 180) : F = 76 / 3 :=
by
  sorry

end angle_F_measure_l227_227078


namespace moles_of_KOH_combined_l227_227129

theorem moles_of_KOH_combined 
  (moles_NH4Cl : ℕ)
  (moles_KCl : ℕ)
  (balanced_reaction : ℕ → ℕ → ℕ)
  (h_NH4Cl : moles_NH4Cl = 3)
  (h_KCl : moles_KCl = 3)
  (reaction_ratio : ∀ n, balanced_reaction n n = n) :
  balanced_reaction moles_NH4Cl moles_KCl = 3 * balanced_reaction 1 1 := 
by
  sorry

end moles_of_KOH_combined_l227_227129


namespace parsley_rows_l227_227069

-- Define the conditions laid out in the problem
def garden_rows : ℕ := 20
def plants_per_row : ℕ := 10
def rosemary_rows : ℕ := 2
def chives_planted : ℕ := 150

-- Define the target statement to prove
theorem parsley_rows :
  let total_plants := garden_rows * plants_per_row
  let remaining_rows := garden_rows - rosemary_rows
  let chives_rows := chives_planted / plants_per_row
  let parsley_rows := remaining_rows - chives_rows
  parsley_rows = 3 :=
by
  sorry

end parsley_rows_l227_227069


namespace range_of_a_same_solution_set_l227_227995

-- Define the inequality (x-2)(x-5) ≤ 0
def ineq1 (x : ℝ) : Prop :=
  (x - 2) * (x - 5) ≤ 0

-- Define the first inequality in the system (x-2)(x-5) ≤ 0
def ineq_system_1 (x : ℝ) : Prop :=
  (x - 2) * (x - 5) ≤ 0

-- Define the second inequality in the system x(x-a) ≥ 0
def ineq_system_2 (x a : ℝ) : Prop :=
  x * (x - a) ≥ 0

-- The final proof statement
theorem range_of_a_same_solution_set (a : ℝ) :
  (∀ x : ℝ, ineq_system_1 x ↔ ineq1 x) →
  (∀ x : ℝ, ineq_system_2 x a → ineq1 x) →
  a ≤ 2 :=
sorry

end range_of_a_same_solution_set_l227_227995


namespace simplify_expression_l227_227314

theorem simplify_expression (p : ℝ) : ((7 * p - 3) - 3 * p * 2) * 2 + (5 - 2 / 2) * (8 * p - 12) = 34 * p - 54 := by
  sorry

end simplify_expression_l227_227314


namespace compute_expression_l227_227321

theorem compute_expression : 45 * 1313 - 10 * 1313 = 45955 := by
  sorry

end compute_expression_l227_227321


namespace solution_fraction_l227_227043

-- Conditions and definition of x
def initial_quantity : ℝ := 1
def concentration_70 : ℝ := 0.70
def concentration_25 : ℝ := 0.25
def concentration_new : ℝ := 0.35

-- Definition of the fraction of the solution replaced
def x (fraction : ℝ) : Prop :=
  concentration_70 * initial_quantity - concentration_70 * fraction + concentration_25 * fraction = concentration_new * initial_quantity

-- The theorem we need to prove
theorem solution_fraction : ∃ (fraction : ℝ), x fraction ∧ fraction = 7 / 9 :=
by
  use 7 / 9
  simp [x]
  sorry  -- Proof steps would be filled here

end solution_fraction_l227_227043


namespace girls_ran_9_miles_l227_227896

def boys_laps : ℕ := 34
def additional_laps : ℕ := 20
def lap_distance : ℚ := 1 / 6

def girls_laps : ℕ := boys_laps + additional_laps
def girls_miles : ℚ := girls_laps * lap_distance

theorem girls_ran_9_miles : girls_miles = 9 := by
  sorry

end girls_ran_9_miles_l227_227896


namespace cannot_move_reach_goal_l227_227157

structure Point :=
(x : ℤ)
(y : ℤ)

def area (p1 p2 p3 : Point) : ℚ :=
  (1 / 2 : ℚ) * abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))

noncomputable def isTriangleAreaPreserved (initPos finalPos : Point) (helper1Init helper1Final helper2Init helper2Final : Point) : Prop :=
  area initPos helper1Init helper2Init = area finalPos helper1Final helper2Final

theorem cannot_move_reach_goal :
  ¬ ∃ (r₀ r₁ : Point) (a₀ a₁ : Point) (s₀ s₁ : Point),
    r₀ = ⟨0, 0⟩ ∧ r₁ = ⟨2, 2⟩ ∧
    a₀ = ⟨0, 1⟩ ∧ a₁ = ⟨0, 1⟩ ∧
    s₀ = ⟨1, 0⟩ ∧ s₁ = ⟨1, 0⟩ ∧
    isTriangleAreaPreserved r₀ r₁ a₀ a₁ s₀ s₁ :=
by sorry

end cannot_move_reach_goal_l227_227157


namespace ratio_areas_ACEF_ADC_l227_227384

-- Define the basic geometric setup
variables (A B C D E F : Point) 
variables (BC CD DE : ℝ) 
variable (α : ℝ)
variables (h1 : 0 < α) (h2 : α < 1/2) (h3 : CD = DE) (h4 : CD = α * BC) 

-- Assuming the given conditions, we want to prove the ratio of areas
noncomputable def ratio_areas (α : ℝ) : ℝ := 4 * (1 - α)

theorem ratio_areas_ACEF_ADC (h1 : 0 < α) (h2 : α < 1/2) (h3 : CD = DE) (h4 : CD = α * BC) :
  ratio_areas α = 4 * (1 - α) :=
sorry

end ratio_areas_ACEF_ADC_l227_227384


namespace area_of_region_l227_227799

def region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (abs p.1 - p.1)^2 + (abs p.2 - p.2)^2 ≤ 16 ∧ 2 * p.2 + p.1 ≤ 0}

noncomputable def area : ℝ := sorry

theorem area_of_region : area = 5 + Real.pi := by
  sorry

end area_of_region_l227_227799


namespace solve_quadratic_eq_l227_227101

theorem solve_quadratic_eq (x : ℝ) (h : (x + 5) ^ 2 = 16) : x = -1 ∨ x = -9 :=
sorry

end solve_quadratic_eq_l227_227101


namespace fraction_product_equals_l227_227925

def frac1 := 7 / 4
def frac2 := 8 / 14
def frac3 := 9 / 6
def frac4 := 10 / 25
def frac5 := 28 / 21
def frac6 := 15 / 45
def frac7 := 32 / 16
def frac8 := 50 / 100

theorem fraction_product_equals : 
  (frac1 * frac2 * frac3 * frac4 * frac5 * frac6 * frac7 * frac8) = (4 / 5) := 
by
  sorry

end fraction_product_equals_l227_227925


namespace mangoes_per_kg_l227_227377

theorem mangoes_per_kg (total_kg : ℕ) (sold_market_kg : ℕ) (sold_community_factor : ℚ) (remaining_mangoes : ℕ) (mangoes_per_kg : ℕ) :
  total_kg = 60 ∧ sold_market_kg = 20 ∧ sold_community_factor = 1/2 ∧ remaining_mangoes = 160 → mangoes_per_kg = 8 :=
  by
  sorry

end mangoes_per_kg_l227_227377


namespace probability_A_not_lose_l227_227317

-- Define the probabilities
def P_A_wins : ℝ := 0.30
def P_draw : ℝ := 0.25
def P_A_not_lose : ℝ := 0.55

-- Statement to prove
theorem probability_A_not_lose : P_A_wins + P_draw = P_A_not_lose :=
by 
  sorry

end probability_A_not_lose_l227_227317


namespace ratio_of_selling_prices_l227_227980

variable (CP : ℝ)
def SP1 : ℝ := CP * 1.6
def SP2 : ℝ := CP * 0.8

theorem ratio_of_selling_prices : SP2 / SP1 = 1 / 2 := 
by sorry

end ratio_of_selling_prices_l227_227980


namespace total_packs_l227_227272

theorem total_packs (cards_per_person : ℕ) (cards_per_pack : ℕ) (people_count : ℕ) (cards_per_person_eq : cards_per_person = 540) (cards_per_pack_eq : cards_per_pack = 20) (people_count_eq : people_count = 4) :
  (cards_per_person / cards_per_pack) * people_count = 108 :=
by
  sorry

end total_packs_l227_227272


namespace cd_e_value_l227_227248

theorem cd_e_value (a b c d e f : ℝ) 
  (h1 : a * b * c = 195) (h2 : b * c * d = 65) 
  (h3 : d * e * f = 250) (h4 : (a * f) / (c * d) = 0.75) :
  c * d * e = 1000 := 
by
  sorry

end cd_e_value_l227_227248


namespace balance_balls_l227_227479

variable {R Y B W : ℕ}

theorem balance_balls (h1 : 4 * R = 8 * B) 
                      (h2 : 3 * Y = 9 * B) 
                      (h3 : 5 * B = 3 * W) : 
    (2 * R + 4 * Y + 3 * W) = 21 * B :=
by 
  sorry

end balance_balls_l227_227479


namespace smallest_positive_multiple_of_17_with_condition_l227_227116

theorem smallest_positive_multiple_of_17_with_condition :
  ∃ k : ℕ, k > 0 ∧ (k % 17 = 0) ∧ (k - 3) % 101 = 0 ∧ k = 306 :=
by
  sorry

end smallest_positive_multiple_of_17_with_condition_l227_227116


namespace can_determine_counterfeit_coin_l227_227244

/-- 
Given 101 coins where 50 are counterfeit and each counterfeit coin 
differs by 1 gram from the genuine ones, prove that Petya can 
determine if a given coin is counterfeit with a single weighing 
using a balance scale.
-/
theorem can_determine_counterfeit_coin :
  ∃ (coins : Fin 101 → ℤ), 
    (∃ i : Fin 101, (1 ≤ i ∧ i ≤ 50 → coins i = 1) ∧ (51 ≤ i ∧ i ≤ 101 → coins i = 0)) →
    (∃ (b : ℤ), (0 < b → b ∣ 1) ∧ (¬(0 < b → b ∣ 1) → coins 101 = b)) :=
by
  sorry

end can_determine_counterfeit_coin_l227_227244


namespace solution_to_diameter_area_problem_l227_227186

def diameter_area_problem : Prop :=
  let radius := 4
  let area_of_shaded_region := 16 + 8 * Real.pi
  -- Definitions derived directly from conditions
  let circle_radius := radius
  let diameter1_perpendicular_to_diameter2 := True
  -- Conclusively prove the area of the shaded region
  ∀ (PQ RS : ℝ) (h1 : PQ = 2 * circle_radius) (h2 : RS = 2 * circle_radius) (h3 : diameter1_perpendicular_to_diameter2),
  ∃ (area : ℝ), area = area_of_shaded_region

-- This is just the statement, the proof part is omitted.
theorem solution_to_diameter_area_problem : diameter_area_problem :=
  sorry

end solution_to_diameter_area_problem_l227_227186


namespace max_third_side_length_l227_227538

theorem max_third_side_length (x : ℕ) (h1 : 28 + x > 47) (h2 : 47 + x > 28) (h3 : 28 + 47 > x) :
  x = 74 :=
sorry

end max_third_side_length_l227_227538


namespace complement_union_A_B_l227_227843

open Set

variable {U : Type*} [Preorder U] [BoundedOrder U]

def A : Set ℝ := {x | x < 1}
def B : Set ℝ := {x | x ≥ 2}

theorem complement_union_A_B :
  compl (A ∪ B) = {x : ℝ | 1 ≤ x ∧ x < 2} :=
by
  sorry

end complement_union_A_B_l227_227843


namespace rotted_tomatoes_is_correct_l227_227042

noncomputable def shipment_1 : ℕ := 1000
noncomputable def sold_Saturday : ℕ := 300
noncomputable def shipment_2 : ℕ := 2 * shipment_1
noncomputable def tomatoes_Tuesday : ℕ := 2500

-- Define remaining tomatoes after the first shipment accounting for Saturday's sales
noncomputable def remaining_tomatoes_1 : ℕ := shipment_1 - sold_Saturday

-- Define total tomatoes after second shipment arrives
noncomputable def total_tomatoes_after_second_shipment : ℕ := remaining_tomatoes_1 + shipment_2

-- Define the amount of tomatoes that rotted
noncomputable def rotted_tomatoes : ℕ :=
  total_tomatoes_after_second_shipment - tomatoes_Tuesday

theorem rotted_tomatoes_is_correct :
  rotted_tomatoes = 200 := by
  sorry

end rotted_tomatoes_is_correct_l227_227042


namespace increasing_m_range_l227_227637

noncomputable def f (x m : ℝ) : ℝ := x^2 + Real.log x - 2 * m * x

theorem increasing_m_range (m : ℝ) : 
  (∀ x > 0, (2 * x + 1 / x - 2 * m ≥ 0)) → m ≤ Real.sqrt 2 :=
by
  intros h
  -- Proof steps would go here
  sorry

end increasing_m_range_l227_227637


namespace rahul_share_of_payment_l227_227173

-- Definitions
def rahulWorkDays : ℕ := 3
def rajeshWorkDays : ℕ := 2
def totalPayment : ℤ := 355

-- Theorem statement
theorem rahul_share_of_payment :
  let rahulWorkRate := 1 / (rahulWorkDays : ℝ)
  let rajeshWorkRate := 1 / (rajeshWorkDays : ℝ)
  let combinedWorkRate := rahulWorkRate + rajeshWorkRate
  let rahulShareRatio := rahulWorkRate / combinedWorkRate
  let rahulShare := (totalPayment : ℝ) * rahulShareRatio
  rahulShare = 142 :=
by
  sorry

end rahul_share_of_payment_l227_227173


namespace b_finishes_remaining_work_in_5_days_l227_227565

theorem b_finishes_remaining_work_in_5_days :
  let A_work_rate := 1 / 4
  let B_work_rate := 1 / 14
  let combined_work_rate := A_work_rate + B_work_rate
  let work_completed_together := 2 * combined_work_rate
  let work_remaining := 1 - work_completed_together
  let days_b_to_finish := work_remaining / B_work_rate
  days_b_to_finish = 5 :=
by
  let A_work_rate := 1 / 4
  let B_work_rate := 1 / 14
  let combined_work_rate := A_work_rate + B_work_rate
  let work_completed_together := 2 * combined_work_rate
  let work_remaining := 1 - work_completed_together
  let days_b_to_finish := work_remaining / B_work_rate
  show days_b_to_finish = 5
  sorry

end b_finishes_remaining_work_in_5_days_l227_227565


namespace exponent_division_l227_227923

theorem exponent_division :
  (1000 ^ 7) / (10 ^ 17) = 10 ^ 4 := 
  sorry

end exponent_division_l227_227923


namespace triangle_inequality_range_l227_227845

theorem triangle_inequality_range {a b c : ℝ} (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  1 ≤ (a^2 + b^2 + c^2) / (a * b + b * c + c * a) ∧ (a^2 + b^2 + c^2) / (a * b + b * c + c * a) < 2 := 
by 
  sorry

end triangle_inequality_range_l227_227845


namespace intersection_of_A_and_B_l227_227793

def A : Set ℕ := {0, 2}
def B : Set ℕ := {1, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {2} :=
by
  sorry

end intersection_of_A_and_B_l227_227793


namespace ratio_of_liquid_p_to_q_initial_l227_227604

noncomputable def initial_ratio_of_p_to_q : ℚ :=
  let p := 20
  let q := 15
  p / q

theorem ratio_of_liquid_p_to_q_initial
  (p q : ℚ)
  (h1 : p + q = 35)
  (h2 : p / (q + 13) = 5 / 7) :
  p / q = 4 / 3 := by
  sorry

end ratio_of_liquid_p_to_q_initial_l227_227604


namespace compare_nsquare_pow2_pos_int_l227_227046

-- Proposition that captures the given properties of comparing n^2 and 2^n
theorem compare_nsquare_pow2_pos_int (n : ℕ) (hn : n > 0) : 
  (n = 1 → n^2 < 2^n) ∧
  (n = 2 → n^2 = 2^n) ∧
  (n = 3 → n^2 > 2^n) ∧
  (n = 4 → n^2 = 2^n) ∧
  (n ≥ 5 → n^2 < 2^n) :=
by
  sorry

end compare_nsquare_pow2_pos_int_l227_227046


namespace fraction_halfway_between_l227_227530

def halfway_fraction (a b : ℚ) : ℚ := (a + b) / 2

theorem fraction_halfway_between :
  halfway_fraction (3/4) (5/6) = 19/24 :=
sorry

end fraction_halfway_between_l227_227530


namespace daily_production_l227_227838

theorem daily_production (x : ℕ) (hx1 : 216 / x > 4)
  (hx2 : 3 * x + (x + 8) * ((216 / x) - 4) = 232) : 
  x = 24 := by
sorry

end daily_production_l227_227838


namespace pictures_hung_in_new_galleries_l227_227632

noncomputable def total_pencils_used : ℕ := 218
noncomputable def pencils_per_picture : ℕ := 5
noncomputable def pencils_per_exhibition : ℕ := 3

noncomputable def pictures_initial : ℕ := 9
noncomputable def galleries_requests : List ℕ := [4, 6, 8, 5, 7, 3, 9]
noncomputable def total_exhibitions : ℕ := 1 + galleries_requests.length

theorem pictures_hung_in_new_galleries :
  let total_pencils_for_signing := total_exhibitions * pencils_per_exhibition
  let total_pencils_for_drawing := total_pencils_used - total_pencils_for_signing
  let total_pictures_drawn := total_pencils_for_drawing / pencils_per_picture
  let pictures_in_new_galleries := total_pictures_drawn - pictures_initial
  pictures_in_new_galleries = 29 :=
by
  sorry

end pictures_hung_in_new_galleries_l227_227632


namespace gcd_8fact_11fact_9square_l227_227396

theorem gcd_8fact_11fact_9square : Nat.gcd (Nat.factorial 8) ((Nat.factorial 11) * 9^2) = 40320 := 
sorry

end gcd_8fact_11fact_9square_l227_227396


namespace x_axis_line_l227_227494

variable (A B C : ℝ)

theorem x_axis_line (h : ∀ x : ℝ, A * x + B * 0 + C = 0) : B ≠ 0 ∧ A = 0 ∧ C = 0 := by
  sorry

end x_axis_line_l227_227494


namespace S_10_minus_S_7_l227_227053

-- Define the first term and common difference of the arithmetic sequence
variables (a₁ d : ℕ)

-- Define the arithmetic sequence based on the first term and common difference
def arithmetic_sequence (n : ℕ) : ℕ := a₁ + (n - 1) * d

-- Define the sum of the first n terms of the arithmetic sequence
def sum_arithmetic_sequence (n : ℕ) : ℕ := n * (2 * a₁ + (n - 1) * d) / 2

-- Conditions given in the problem
axiom a_5_eq : a₁ + 4 * d = 8
axiom S_3_eq : sum_arithmetic_sequence a₁ 3 = 6

-- The goal: prove that S_10 - S_7 = 48
theorem S_10_minus_S_7 : sum_arithmetic_sequence a₁ 10 - sum_arithmetic_sequence a₁ 7 = 48 :=
sorry

end S_10_minus_S_7_l227_227053


namespace op_15_5_eq_33_l227_227341

def op (x y : ℕ) : ℕ :=
  2 * x + x / y

theorem op_15_5_eq_33 : op 15 5 = 33 := by
  sorry

end op_15_5_eq_33_l227_227341


namespace divisor_greater_2016_l227_227685

theorem divisor_greater_2016 (d : ℕ) (h : 2016 / d = 0) : d > 2016 :=
sorry

end divisor_greater_2016_l227_227685


namespace slope_of_tangent_at_4_l227_227736

def f (x : ℝ) : ℝ := x^3 - 7 * x^2 + 1

theorem slope_of_tangent_at_4 : (deriv f 4) = -8 := by
  sorry

end slope_of_tangent_at_4_l227_227736


namespace original_number_l227_227483

theorem original_number (x : ℝ) (h : x * 1.20 = 1080) : x = 900 :=
sorry

end original_number_l227_227483


namespace no_values_of_b_l227_227525

def f (b x : ℝ) := x^2 + b * x - 1

theorem no_values_of_b : ∀ b : ℝ, ∃ x : ℝ, f b x = 3 :=
by
  intro b
  use 0  -- example, needs actual computation
  sorry

end no_values_of_b_l227_227525


namespace amount_of_bill_l227_227868

noncomputable def TD : ℝ := 360
noncomputable def BD : ℝ := 418.9090909090909
noncomputable def FV (TD BD : ℝ) : ℝ := TD * BD / (BD - TD)

theorem amount_of_bill :
  FV TD BD = 2568 :=
by
  sorry

end amount_of_bill_l227_227868


namespace solve_x_from_equation_l227_227735

theorem solve_x_from_equation :
  ∀ (x : ℝ), x ≠ 0 ∧ (3 * x)^5 = (9 * x)^4 → x = 27 :=
by
  intro x
  rintro ⟨hx, h⟩
  sorry

end solve_x_from_equation_l227_227735


namespace ordered_quadruple_ellipse_l227_227649

noncomputable def ellipse_quadruple := 
  let f₁ : (ℝ × ℝ) := (1, 1)
  let f₂ : (ℝ × ℝ) := (1, 7)
  let p : (ℝ × ℝ) := (12, -1)
  let a := (5 / 2) * (Real.sqrt 5 + Real.sqrt 37)
  let b := (1 / 2) * Real.sqrt (1014 + 50 * Real.sqrt 185)
  let h := 1
  let k := 4
  (a, b, h, k)

theorem ordered_quadruple_ellipse :
  let e : (ℝ × ℝ × ℝ × ℝ) := θse_quadruple
  e = ((5 / 2 * (Real.sqrt 5 + Real.sqrt 37)), (1 / 2 * Real.sqrt (1014 + 50 * Real.sqrt 185)), 1, 4) :=
by
  sorry

end ordered_quadruple_ellipse_l227_227649


namespace probability_of_collinear_dots_in_5x5_grid_l227_227541

def collinear_dots_probability (total_dots chosen_dots collinear_sets : ℕ) : ℚ :=
  (collinear_sets : ℚ) / (Nat.choose total_dots chosen_dots)

theorem probability_of_collinear_dots_in_5x5_grid :
  collinear_dots_probability 25 4 12 = 12 / 12650 := by
  sorry

end probability_of_collinear_dots_in_5x5_grid_l227_227541


namespace transform_polynomial_l227_227644

variables {x y : ℝ}

theorem transform_polynomial (h : y = x - 1 / x) :
  (x^6 + x^5 - 5 * x^4 + 2 * x^3 - 5 * x^2 + x + 1 = 0) ↔ (x^2 * (y^2 + y - 3) = 0) :=
sorry

end transform_polynomial_l227_227644


namespace solids_with_triangular_front_view_l227_227924

-- Definitions based on given conditions
inductive Solid
  | TriangularPyramid
  | SquarePyramid
  | TriangularPrism
  | SquarePrism
  | Cone
  | Cylinder

def can_have_triangular_front_view : Solid → Prop
  | Solid.TriangularPyramid => true
  | Solid.SquarePyramid => true
  | Solid.TriangularPrism => true
  | Solid.SquarePrism => false
  | Solid.Cone => true
  | Solid.Cylinder => false

-- Theorem statement
theorem solids_with_triangular_front_view :
  {s : Solid | can_have_triangular_front_view s} = 
  {Solid.TriangularPyramid, Solid.SquarePyramid, Solid.TriangularPrism, Solid.Cone} :=
by
  sorry

end solids_with_triangular_front_view_l227_227924


namespace base_7_to_base_10_l227_227394

theorem base_7_to_base_10 (a b c d e : ℕ) (h : 23456 = e * 10000 + d * 1000 + c * 100 + b * 10 + a) :
  2 * 7^4 + 3 * 7^3 + 4 * 7^2 + 5 * 7^1 + 6 * 7^0 = 6068 :=
by
  sorry

end base_7_to_base_10_l227_227394


namespace min_perimeter_l227_227189

theorem min_perimeter (a b : ℕ) (h1 : b = 3 * a) (h2 : 3 * a + 8 * a = 11) (h3 : 2 * a + 12 * a = 14)
  : 2 * (15 + 11) = 52 := 
sorry

end min_perimeter_l227_227189


namespace smallest_x_value_l227_227682

open Real

theorem smallest_x_value (x : ℝ) 
  (h : x * abs x = 3 * x + 2) : 
  x = -2 ∨ (∀ y, y * abs y = 3 * y + 2 → y ≥ -2) := sorry

end smallest_x_value_l227_227682


namespace circle_C2_equation_line_l_equation_l227_227755

-- Proof problem 1: Finding the equation of C2
theorem circle_C2_equation (C1_center_x C1_center_y : ℝ) (A_x A_y : ℝ) 
  (C2_center_x : ℝ) (C1_radius : ℝ) :
  C1_center_x = 6 ∧ C1_center_y = 7 ∧ C1_radius = 5 →
  A_x = 2 ∧ A_y = 4 →
  C2_center_x = 6 →
  (∀ y : ℝ, ((y - C1_center_y = C1_radius + (C1_radius + (y - C1_center_y)))) →
    (x - C2_center_x)^2 + (y - C2_center_y)^2 = 1) :=
sorry

-- Proof problem 2: Finding the equation of the line l
theorem line_l_equation (O_x O_y A_x A_y : ℝ) 
  (C1_center_x C1_center_y : ℝ) 
  (A_BC_dist : ℝ) :
  O_x = 0 ∧ O_y = 0 →
  A_x = 2 ∧ A_y = 4 →
  C1_center_x = 6 ∧ C1_center_y = 7 →
  A_BC_dist = 2 * (25^(1 / 2)) →
  ((2 : ℝ)*x - y + 5 = 0 ∨ (2 : ℝ)*x - y - 15 = 0) :=
sorry

end circle_C2_equation_line_l_equation_l227_227755


namespace car_speed_is_80_l227_227810

theorem car_speed_is_80 : ∃ v : ℝ, (1 / v * 3600 = 45) ∧ (v = 80) :=
by
  sorry

end car_speed_is_80_l227_227810


namespace base7_sub_base5_to_base10_l227_227605

def base7to10 (n : Nat) : Nat :=
  match n with
  | 52403 => 5 * 7^4 + 2 * 7^3 + 4 * 7^2 + 0 * 7^1 + 3 * 7^0
  | _ => 0

def base5to10 (n : Nat) : Nat :=
  match n with
  | 20345 => 2 * 5^4 + 0 * 5^3 + 3 * 5^2 + 4 * 5^1 + 5 * 5^0
  | _ => 0

theorem base7_sub_base5_to_base10 :
  base7to10 52403 - base5to10 20345 = 11540 :=
by
  sorry

end base7_sub_base5_to_base10_l227_227605


namespace orange_pear_difference_l227_227937

theorem orange_pear_difference :
  let O1 := 37
  let O2 := 10
  let O3 := 2 * O2
  let P1 := 30
  let P2 := 3 * P1
  let P3 := P2 + 4
  (O1 + O2 + O3 - (P1 + P2 + P3)) = -147 := 
by
  sorry

end orange_pear_difference_l227_227937


namespace min_turns_for_route_l227_227730

-- Define the number of parallel and intersecting streets
def num_parallel_streets := 10
def num_intersecting_streets := 10

-- Define the grid as a product of these two numbers
def num_intersections := num_parallel_streets * num_intersecting_streets

-- Define the minimum number of turns necessary for a closed bus route passing through all intersections
def min_turns (grid_size : Nat) : Nat :=
  if grid_size = num_intersections then 20 else 0

-- The main theorem statement
theorem min_turns_for_route : min_turns num_intersections = 20 :=
  sorry

end min_turns_for_route_l227_227730


namespace largest_value_of_x_l227_227255

noncomputable def find_largest_x : ℝ :=
  let a := 10
  let b := 39
  let c := 18
  let discriminant := b^2 - 4 * a * c
  let sqrt_discriminant := Real.sqrt discriminant
  let x1 := (-b + sqrt_discriminant) / (2 * a)
  let x2 := (-b - sqrt_discriminant) / (2 * a)
  if x1 > x2 then x1 else x2

theorem largest_value_of_x :
  ∃ x : ℝ, 3 * (10 * x^2 + 11 * x + 12) = x * (10 * x - 45) ∧
  x = find_largest_x := by
  exists find_largest_x
  sorry

end largest_value_of_x_l227_227255


namespace number_under_35_sampled_l227_227331

-- Define the conditions
def total_employees : ℕ := 500
def employees_under_35 : ℕ := 125
def employees_35_to_49 : ℕ := 280
def employees_over_50 : ℕ := 95
def sample_size : ℕ := 100

-- Define the theorem stating the desired result
theorem number_under_35_sampled : (employees_under_35 * sample_size / total_employees) = 25 :=
by
  sorry

end number_under_35_sampled_l227_227331


namespace find_bc_l227_227098

theorem find_bc (b c : ℤ) (h : ∀ x : ℝ, x^2 + (b : ℝ) * x + (c : ℝ) = 0 ↔ x = 1 ∨ x = 2) :
  b = -3 ∧ c = 2 := by
  sorry

end find_bc_l227_227098


namespace find_income_of_deceased_l227_227805
noncomputable def income_of_deceased_member 
  (members_before : ℕ) (avg_income_before : ℕ) 
  (members_after : ℕ) (avg_income_after : ℕ) : ℕ :=
  (members_before * avg_income_before) - (members_after * avg_income_after)

theorem find_income_of_deceased 
  (members_before avg_income_before members_after avg_income_after : ℕ) :
  income_of_deceased_member 4 840 3 650 = 1410 :=
by
  -- Problem claims income_of_deceased_member = Income before - Income after
  sorry

end find_income_of_deceased_l227_227805


namespace total_games_played_l227_227267

def number_of_games (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem total_games_played :
  number_of_games 9 2 = 36 :=
by
  -- Proof to be filled in later
  sorry

end total_games_played_l227_227267


namespace no_pos_int_lt_2000_7_times_digits_sum_l227_227916

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem no_pos_int_lt_2000_7_times_digits_sum :
  ∀ n : ℕ, n < 2000 → n = 7 * sum_of_digits n → False :=
by
  intros n h1 h2
  sorry

end no_pos_int_lt_2000_7_times_digits_sum_l227_227916


namespace cubic_root_relation_l227_227058

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x - 1

theorem cubic_root_relation
  (x1 x2 x3 : ℝ)
  (hx1x2 : x1 < x2)
  (hx2x3 : x2 < 0)
  (hx3pos : 0 < x3)
  (hfx1 : f x1 = 0)
  (hfx2 : f x2 = 0)
  (hfx3 : f x3 = 0) :
  x3^2 - x2^2 = x3 - x1 :=
sorry

end cubic_root_relation_l227_227058


namespace smallest_nine_consecutive_sum_l227_227335

theorem smallest_nine_consecutive_sum (n : ℕ) (h : (n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5) + (n+6) + (n+7) + (n+8) = 2007)) : n = 219 :=
sorry

end smallest_nine_consecutive_sum_l227_227335


namespace length_of_football_field_l227_227913

theorem length_of_football_field :
  ∃ x : ℝ, (4 * x + 500 = 1172) ∧ x = 168 :=
by
  use 168
  simp
  sorry

end length_of_football_field_l227_227913


namespace find_B_plus_C_l227_227231

theorem find_B_plus_C 
(A B C : ℕ)
(h1 : A ≠ B)
(h2 : B ≠ C)
(h3 : C ≠ A)
(h4 : A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0)
(h5 : A < 5 ∧ B < 5 ∧ C < 5)
(h6 : 25 * A + 5 * B + C + 25 * B + 5 * C + A + 25 * C + 5 * A + B = 125 * A + 25 * A + 5 * A) : 
B + C = 4 * A := by
  sorry

end find_B_plus_C_l227_227231


namespace find_smaller_circle_radius_l227_227390

noncomputable def smaller_circle_radius (R : ℝ) : ℝ :=
  R / (Real.sqrt 2 - 1)

theorem find_smaller_circle_radius (R : ℝ) (x : ℝ) :
  (∀ (c1 c2 c3 c4 : ℝ),  c1 = c2 ∧ c2 = c3 ∧ c3 = c4 ∧ c4 = x
  ∧ c1 + c2 = 2 * c3 * Real.sqrt 2)
  → x = smaller_circle_radius R :=
by 
  intros h
  sorry

end find_smaller_circle_radius_l227_227390


namespace possible_values_of_a_l227_227699

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 - a*x + 5 else a / x

theorem possible_values_of_a (a : ℝ) :
  (∀ x y : ℝ, x ≤ y → f a x ≥ f a y) ↔ (2 ≤ a ∧ a ≤ 3) :=
by
  sorry

end possible_values_of_a_l227_227699


namespace remainder_2027_div_28_l227_227698

theorem remainder_2027_div_28 : 2027 % 28 = 3 :=
by
  sorry

end remainder_2027_div_28_l227_227698


namespace total_apples_bought_l227_227134

def apples_bought_by_Junhyeok := 7 * 16
def apples_bought_by_Jihyun := 6 * 25

theorem total_apples_bought : apples_bought_by_Junhyeok + apples_bought_by_Jihyun = 262 := by
  sorry

end total_apples_bought_l227_227134


namespace seating_arrangement_l227_227258

theorem seating_arrangement (boys girls : ℕ) (h_boys : boys = 4) (h_girls : girls = 3) : 
  (∃ n : ℕ, n = (boys.factorial * girls.factorial) + (girls.factorial * boys.factorial) ∧ n = 288) :=
by 
  sorry

end seating_arrangement_l227_227258


namespace remainder_sum_l227_227822

theorem remainder_sum (a b c d : ℕ) 
  (h_a : a % 30 = 15) 
  (h_b : b % 30 = 7) 
  (h_c : c % 30 = 22) 
  (h_d : d % 30 = 6) : 
  (a + b + c + d) % 30 = 20 := 
by
  sorry

end remainder_sum_l227_227822


namespace graph_passes_quadrants_l227_227999

theorem graph_passes_quadrants (a b : ℝ) (h_a : 1 < a) (h_b : -1 < b ∧ b < 0) : 
    ∀ x : ℝ, (0 < a^x + b ∧ x > 0) ∨ (a^x + b < 0 ∧ x < 0) ∨ (0 < x ∧ a^x + b = 0) → x ≠ 0 ∧ 0 < x :=
sorry

end graph_passes_quadrants_l227_227999


namespace solve_logarithmic_equation_l227_227003

noncomputable def log_base (a x : ℝ) : ℝ := Real.log x / Real.log a

theorem solve_logarithmic_equation (x : ℝ) (h_pos : x > 0) :
  log_base 8 x + log_base 4 (x^2) + log_base 2 (x^3) = 15 ↔ x = 2 ^ (45 / 13) :=
by
  have h1 : log_base 8 x = (1 / 3) * log_base 2 x :=
    by { sorry }
  have h2 : log_base 4 (x^2) = log_base 2 x :=
    by { sorry }
  have h3 : log_base 2 (x^3) = 3 * log_base 2 x :=
    by { sorry }
  have h4 : (1 / 3) * log_base 2 x + log_base 2 x + 3 * log_base 2 x = 15 ↔ log_base 2 x = 45 / 13 :=
    by { sorry }
  exact sorry

end solve_logarithmic_equation_l227_227003


namespace original_bananas_total_l227_227278

theorem original_bananas_total (willie_bananas : ℝ) (charles_bananas : ℝ) : willie_bananas = 48.0 → charles_bananas = 35.0 → willie_bananas + charles_bananas = 83.0 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num

end original_bananas_total_l227_227278


namespace find_k_l227_227614

theorem find_k (k : ℕ) (h_pos : k > 0) (h_coef : 15 * k^4 < 120) : k = 1 :=
sorry

end find_k_l227_227614


namespace min_value_z_l227_227773

theorem min_value_z (x y z : ℤ) (h1 : x + y + z = 100) (h2 : x < y) (h3 : y < 2 * z) : z ≥ 21 :=
sorry

end min_value_z_l227_227773


namespace problem_goal_l227_227277

-- Define the problem stating that there is a graph of points (x, y) satisfying the condition
def area_of_graph_satisfying_condition : Real :=
  let A := 2013
  -- Define the pairs (a, b) which are multiples of 2013
  let pairs := [(1, 2013), (3, 671), (11, 183), (33, 61)]
  -- Calculate the area of each region formed by pairs
  let area := pairs.length * 4
  area

-- Problem goal statement proving the area is equal to 16
theorem problem_goal : area_of_graph_satisfying_condition = 16 := by
  sorry

end problem_goal_l227_227277


namespace geometric_series_common_ratio_l227_227677

theorem geometric_series_common_ratio (a S r : ℝ) (h₁ : a = 400) (h₂ : S = 2500) 
  (h₃ : S = a / (1 - r)) : r = 21 / 25 := 
sorry

end geometric_series_common_ratio_l227_227677


namespace price_per_acre_is_1863_l227_227743

-- Define the conditions
def totalAcres : ℕ := 4
def numLots : ℕ := 9
def pricePerLot : ℤ := 828
def totalRevenue : ℤ := numLots * pricePerLot
def totalCost (P : ℤ) : ℤ := totalAcres * P

-- The proof problem: Prove that the price per acre P is 1863
theorem price_per_acre_is_1863 (P : ℤ) (h : totalCost P = totalRevenue) : P = 1863 :=
by
  sorry

end price_per_acre_is_1863_l227_227743


namespace seventy_three_days_after_monday_is_thursday_l227_227213

def day_of_week : Nat → String
| 0 => "Monday"
| 1 => "Tuesday"
| 2 => "Wednesday"
| 3 => "Thursday"
| 4 => "Friday"
| 5 => "Saturday"
| _ => "Sunday"

theorem seventy_three_days_after_monday_is_thursday :
  day_of_week (73 % 7) = "Thursday" :=
by
  sorry

end seventy_three_days_after_monday_is_thursday_l227_227213


namespace fraction_white_tulips_l227_227551

theorem fraction_white_tulips : 
  ∀ (total_tulips yellow_fraction red_fraction pink_fraction white_fraction : ℝ),
  total_tulips = 60 →
  yellow_fraction = 1 / 2 →
  red_fraction = 1 / 3 →
  pink_fraction = 1 / 4 →
  white_fraction = 
    ((total_tulips * (1 - yellow_fraction)) * (1 - red_fraction) * (1 - pink_fraction)) / total_tulips →
  white_fraction = 1 / 4 :=
by
  intros total_tulips yellow_fraction red_fraction pink_fraction white_fraction 
    h_total h_yellow h_red h_pink h_white
  sorry

end fraction_white_tulips_l227_227551


namespace jerry_wants_to_raise_average_l227_227960

theorem jerry_wants_to_raise_average :
  let average_first_3_tests := 85
  let total_first_3_tests := 3 * average_first_3_tests
  let score_fourth_test := 93
  let total_after_fourth_test := total_first_3_tests + score_fourth_test
  let new_average := (total_after_fourth_test : ℤ) / 4
  new_average - average_first_3_tests = 2 :=
by
  let average_first_3_tests := 85
  let total_first_3_tests := 3 * average_first_3_tests
  let score_fourth_test := 93
  let total_after_fourth_test := total_first_3_tests + score_fourth_test
  let new_average := (total_after_fourth_test : ℤ) / 4
  have h : new_average - average_first_3_tests = 2 := by
    sorry
  exact h

end jerry_wants_to_raise_average_l227_227960


namespace highest_score_l227_227223

theorem highest_score (H L : ℕ) (avg total46 total44 runs46 runs44 : ℕ)
  (h1 : H - L = 150)
  (h2 : avg = 61)
  (h3 : total46 = 46)
  (h4 : runs46 = avg * total46)
  (h5 : runs46 = 2806)
  (h6 : total44 = 44)
  (h7 : runs44 = 58 * total44)
  (h8 : runs44 = 2552)
  (h9 : runs46 - runs44 = H + L) :
  H = 202 := by
  sorry

end highest_score_l227_227223


namespace factor_expression_l227_227828

theorem factor_expression (x : ℝ) : 84 * x^7 - 297 * x^13 = 3 * x^7 * (28 - 99 * x^6) :=
by sorry

end factor_expression_l227_227828


namespace radian_measure_of_central_angle_l227_227733

-- Given conditions
variables (l r : ℝ)
variables (h1 : (1 / 2) * l * r = 1)
variables (h2 : 2 * r + l = 4)

-- The theorem to prove
theorem radian_measure_of_central_angle (l r : ℝ) (h1 : (1 / 2) * l * r = 1) (h2 : 2 * r + l = 4) : 
  l / r = 2 :=
by 
  -- Proof steps are not provided as per the requirement
  sorry

end radian_measure_of_central_angle_l227_227733


namespace lateral_surface_area_of_rotated_square_l227_227438

noncomputable def lateralSurfaceAreaOfRotatedSquare (side_length : ℝ) : ℝ :=
  2 * Real.pi * side_length * side_length

theorem lateral_surface_area_of_rotated_square :
  lateralSurfaceAreaOfRotatedSquare 1 = 2 * Real.pi :=
by
  sorry

end lateral_surface_area_of_rotated_square_l227_227438


namespace matt_skips_correctly_l227_227210

-- Definitions based on conditions
def skips_per_second := 3
def jumping_time_minutes := 10
def seconds_per_minute := 60
def total_jumping_seconds := jumping_time_minutes * seconds_per_minute
def expected_skips := total_jumping_seconds * skips_per_second

-- Proof statement
theorem matt_skips_correctly :
  expected_skips = 1800 :=
by
  sorry

end matt_skips_correctly_l227_227210


namespace find_d_l227_227794

open Real

theorem find_d (a b c d : ℝ) 
  (h : a^3 + b^3 + c^3 + a^2 + b^2 + 1 = d^2 + d + sqrt (a + b + c - 2 * d)) : 
  d = 1 ∨ d = -(4 / 3) :=
sorry

end find_d_l227_227794


namespace set_of_a_l227_227652

theorem set_of_a (a : ℝ) :
  (∃ x : ℝ, a * x ^ 2 + a * x + 1 = 0) → -- Set A contains elements
  (a ≠ 0 ∧ a ^ 2 - 4 * a = 0) →           -- Conditions a ≠ 0 and Δ = 0
  a = 4 := 
sorry

end set_of_a_l227_227652


namespace cards_ratio_l227_227313

variable (x : ℕ)

def partially_full_decks_cards := 3 * x
def full_decks_cards := 3 * 52
def total_cards_before := 200 + 34

theorem cards_ratio (h : 3 * x + full_decks_cards = total_cards_before) : x / 52 = 1 / 2 :=
by sorry

end cards_ratio_l227_227313


namespace find_a_l227_227767

-- Conditions: x = 5 is a solution to the equation 2x - a = -5
-- We need to prove that a = 15 under these conditions

theorem find_a (x a : ℤ) (h1 : x = 5) (h2 : 2 * x - a = -5) : a = 15 :=
by
  -- We are required to prove the statement, so we skip the proof part here
  sorry

end find_a_l227_227767


namespace add_base7_l227_227626

-- Define the two numbers in base 7 to be added.
def number1 : ℕ := 2 * 7 + 5
def number2 : ℕ := 5 * 7 + 4

-- Define the expected result in base 7.
def expected_sum : ℕ := 1 * 7^2 + 1 * 7 + 2

theorem add_base7 :
  let sum : ℕ := number1 + number2
  sum = expected_sum := sorry

end add_base7_l227_227626


namespace sequence_difference_l227_227601

theorem sequence_difference : 
  (∃ (a : ℕ → ℤ) (S : ℕ → ℤ), 
    (∀ n : ℕ, S n = n^2 + 2 * n) ∧ 
    (∀ n : ℕ, n > 0 → a n = S n - S (n - 1) ) ∧ 
    (a 4 - a 2 = 4)) :=
by
  sorry

end sequence_difference_l227_227601


namespace range_of_a_l227_227885

def A (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 5
def B (x : ℝ) (a : ℝ) : Prop := x > a

theorem range_of_a (a : ℝ) : (∀ x : ℝ, A x → B x a) → a < -2 :=
by
  sorry

end range_of_a_l227_227885


namespace f_1988_eq_1988_l227_227106

noncomputable def f (n : ℕ) : ℕ := sorry

axiom f_f_eq_add (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : f (f m + f n) = m + n

theorem f_1988_eq_1988 : f 1988 = 1988 := 
by
  sorry

end f_1988_eq_1988_l227_227106


namespace wonderland_cities_l227_227477

theorem wonderland_cities (V E B : ℕ) (hE : E = 45) (hB : B = 42) (h_connected : connected_graph) (h_simple : simple_graph) (h_bridges : count_bridges = 42) : V = 45 :=
sorry

end wonderland_cities_l227_227477


namespace simplify_expression_l227_227326

variable (a b : ℤ) -- Define variables a and b

theorem simplify_expression : 
  (35 * a + 70 * b + 15) + (15 * a + 54 * b + 5) - (20 * a + 85 * b + 10) =
  30 * a + 39 * b + 10 := 
by sorry

end simplify_expression_l227_227326


namespace sum_of_arithmetic_sequence_l227_227592

theorem sum_of_arithmetic_sequence
    (a : ℕ → ℤ)
    (S : ℕ → ℤ)
    (hS : ∀ n : ℕ, S n = n * a n)
    (h_condition : a 1 - a 5 - a 10 - a 15 + a 19 = 2) :
    S 19 = -38 :=
sorry

end sum_of_arithmetic_sequence_l227_227592


namespace gray_region_area_l227_227709

noncomputable def area_of_gray_region (C_center D_center : ℝ × ℝ) (C_radius D_radius : ℝ) :=
  let rect_area := 35
  let semicircle_C_area := (25 * Real.pi) / 2
  let quarter_circle_D_area := (16 * Real.pi) / 4
  rect_area - semicircle_C_area - quarter_circle_D_area

theorem gray_region_area :
  area_of_gray_region (5, 5) (12, 5) 5 4 = 35 - 16.5 * Real.pi :=
by
  simp [area_of_gray_region]
  sorry

end gray_region_area_l227_227709


namespace integer_solutions_of_system_l227_227994

theorem integer_solutions_of_system (x y z : ℤ) :
  x + y + z = 2 ∧ x^3 + y^3 + z^3 = -10 ↔ 
  (x = 3 ∧ y = 3 ∧ z = -4) ∨ 
  (x = 3 ∧ y = -4 ∧ z = 3) ∨ 
  (x = -4 ∧ y = 3 ∧ z = 3) :=
sorry

end integer_solutions_of_system_l227_227994


namespace time_after_10000_seconds_l227_227603

def time_add_seconds (h m s : Nat) (t : Nat) : (Nat × Nat × Nat) :=
  let total_seconds := h * 3600 + m * 60 + s + t
  let hours := (total_seconds / 3600) % 24
  let minutes := (total_seconds % 3600) / 60
  let seconds := (total_seconds % 3600) % 60
  (hours, minutes, seconds)

theorem time_after_10000_seconds :
  time_add_seconds 5 45 0 10000 = (8, 31, 40) :=
by
  sorry

end time_after_10000_seconds_l227_227603


namespace determine_a_l227_227769

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x < 1 then 2 * x + a else -x - 2 * a

theorem determine_a (a : ℝ) (h : a ≠ 0) (h_eq : f a (1 - a) = f a (1 + a)) : a = -3/4 :=
by
  sorry

end determine_a_l227_227769


namespace additional_time_proof_l227_227761

-- Given the charging rate of the battery and the additional time required to reach a percentage
noncomputable def charging_rate := 20 / 60
noncomputable def initial_time := 60
noncomputable def additional_time := 150

-- Define the total time required to reach a certain percentage
noncomputable def total_time := initial_time + additional_time

-- The proof statement to verify the additional time required beyond the initial 60 minutes
theorem additional_time_proof : total_time - initial_time = additional_time := sorry

end additional_time_proof_l227_227761


namespace geometric_series_sum_l227_227562

theorem geometric_series_sum :
  let a := (1 : ℝ) / 4
  let r := (1 : ℝ) / 4
  let n := 7
  let S_n := a * (1 - r^n) / (1 - r)
  S_n = 16383 / 49152 :=
by
  sorry

end geometric_series_sum_l227_227562


namespace jacket_purchase_price_l227_227498

theorem jacket_purchase_price (P S SP : ℝ)
  (h1 : S = P + 0.40 * S)
  (h2 : SP = 0.80 * S)
  (h3 : SP - P = 18) :
  P = 54 :=
by
  sorry

end jacket_purchase_price_l227_227498


namespace sheela_monthly_income_l227_227356

theorem sheela_monthly_income (d : ℝ) (p : ℝ) (income : ℝ) (h1 : d = 4500) (h2 : p = 0.28) (h3 : d = p * income) : 
  income = 16071.43 :=
by
  sorry

end sheela_monthly_income_l227_227356


namespace quadratic_roots_abs_eq_l227_227850

theorem quadratic_roots_abs_eq (x1 x2 m : ℝ) (h1 : x1 > 0) (h2 : x2 < 0) 
  (h_eq_roots : ∀ x, x^2 - (x1 + x2)*x + x1*x2 = 0) : 
  ∃ q : ℝ, q = x^2 - (1 - 4*m)/x + 2 := 
by
  sorry

end quadratic_roots_abs_eq_l227_227850


namespace sum_of_remainders_mod_53_l227_227768

theorem sum_of_remainders_mod_53 (x y z : ℕ) (h1 : x % 53 = 31) (h2 : y % 53 = 17) (h3 : z % 53 = 8) : 
  (x + y + z) % 53 = 3 :=
by {
  sorry
}

end sum_of_remainders_mod_53_l227_227768


namespace prop_sufficient_not_necessary_l227_227481

-- Let p and q be simple propositions.
variables (p q : Prop)

-- Define the statement to be proved: 
-- "either p or q is false" is a sufficient but not necessary condition 
-- for "not p is true".
theorem prop_sufficient_not_necessary (hpq : ¬(p ∧ q)) : ¬ p :=
sorry

end prop_sufficient_not_necessary_l227_227481


namespace problem_statement_l227_227142

-- Define function f(x) given parameter m
def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + m * x + 3

-- Define even function condition
def is_even (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)

-- Define the monotonic decreasing interval condition
def is_monotonically_decreasing (f : ℝ → ℝ) (I : Set ℝ) :=
 ∀ x y : ℝ, x ∈ I → y ∈ I → x < y → f x ≥ f y

theorem problem_statement :
  (∀ x : ℝ, f m x = f m (-x)) → is_monotonically_decreasing (f 0) {x | 0 < x} :=
by 
  sorry

end problem_statement_l227_227142


namespace cross_section_area_l227_227706

-- Definitions for the conditions stated in the problem
def frustum_height : ℝ := 6
def upper_base_side : ℝ := 4
def lower_base_side : ℝ := 8

-- The main statement to be proved
theorem cross_section_area :
  (exists (cross_section_area : ℝ),
    cross_section_area = 16 * Real.sqrt 6) :=
sorry

end cross_section_area_l227_227706


namespace length_CD_l227_227221

-- Definitions of the edge lengths provided in the problem
def edge_lengths : Set ℕ := {7, 13, 18, 27, 36, 41}

-- Assumption that AB = 41
def AB := 41
def BC : ℕ := 13
def AC : ℕ := 36

-- Main theorem to prove that CD = 13
theorem length_CD (AB BC AC : ℕ) (edges : Set ℕ) (hAB : AB = 41) (hedges : edges = edge_lengths) :
  ∃ (CD : ℕ), CD ∈ edges ∧ CD = 13 :=
by
  sorry

end length_CD_l227_227221


namespace buratino_correct_l227_227747

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

def valid_nine_digit_number (n : ℕ) : Prop :=
  n >= 10^8 ∧ n < 10^9 ∧ (∀ i j : ℕ, i < 9 ∧ j < 9 ∧ i ≠ j → ((n / 10^i) % 10 ≠ (n / 10^j) % 10)) ∧
  (∀ i : ℕ, i < 9 → (n / 10^i) % 10 ≠ 7)

def can_form_prime (n : ℕ) : Prop :=
  ∃ m : ℕ, valid_nine_digit_number n ∧ (m < 1000 ∧ is_prime m ∧
   (∃ erase_indices : List ℕ, erase_indices.length = 6 ∧ 
    ∀ i : ℕ, i ∈ erase_indices → i < 9 ∧ 
    (n % 10^(9 - i)) / 10^(3 - i) = m))

theorem buratino_correct : 
  ∀ n : ℕ, valid_nine_digit_number n → ¬ can_form_prime n :=
by
  sorry

end buratino_correct_l227_227747


namespace cassie_has_8_parrots_l227_227144

-- Define the conditions
def num_dogs : ℕ := 4
def nails_per_foot : ℕ := 4
def feet_per_dog : ℕ := 4
def nails_per_dog := nails_per_foot * feet_per_dog

def nails_total_dogs : ℕ := num_dogs * nails_per_dog

def claws_per_leg : ℕ := 3
def legs_per_parrot : ℕ := 2
def normal_claws_per_parrot := claws_per_leg * legs_per_parrot

def extra_toe_parrot_claws : ℕ := normal_claws_per_parrot + 1

def total_nails : ℕ := 113

-- Establishing the proof problem
theorem cassie_has_8_parrots : 
  ∃ (P : ℕ), (6 * (P - 1) + 7 = 49) ∧ P = 8 := by
  sorry

end cassie_has_8_parrots_l227_227144


namespace pentagonal_tiles_count_l227_227901

theorem pentagonal_tiles_count (a b : ℕ) (h1 : a + b = 30) (h2 : 3 * a + 5 * b = 120) : b = 15 :=
by
  sorry

end pentagonal_tiles_count_l227_227901


namespace incorrect_value_l227_227363

theorem incorrect_value:
  ∀ (n : ℕ) (initial_mean corrected_mean : ℚ) (correct_value incorrect_value : ℚ),
  n = 50 →
  initial_mean = 36 →
  corrected_mean = 36.5 →
  correct_value = 48 →
  incorrect_value = correct_value - (corrected_mean * n - initial_mean * n) →
  incorrect_value = 23 :=
by
  intros n initial_mean corrected_mean correct_value incorrect_value
  intros h1 h2 h3 h4 h5
  sorry

end incorrect_value_l227_227363


namespace area_of_quadrilateral_PQRS_l227_227504

noncomputable def calculate_area_of_quadrilateral_PQRS (PQ PR : ℝ) (PS_corrected : ℝ) : ℝ :=
  let area_ΔPQR := (1/2) * PQ * PR
  let RS := Real.sqrt (PR^2 - PQ^2)
  let area_ΔPRS := (1/2) * PR * RS
  area_ΔPQR + area_ΔPRS

theorem area_of_quadrilateral_PQRS :
  let PQ := 8
  let PR := 10
  let PS_corrected := Real.sqrt (PQ^2 + PR^2)
  calculate_area_of_quadrilateral_PQRS PQ PR PS_corrected = 70 := 
by
  sorry

end area_of_quadrilateral_PQRS_l227_227504


namespace evaluate_expression_l227_227758

theorem evaluate_expression : 3000^3 - 2999 * 3000^2 - 2999^2 * 3000 + 2999^3 = 5999 := by
  sorry

end evaluate_expression_l227_227758


namespace remainder_of_3_pow_100_plus_5_mod_8_l227_227542

theorem remainder_of_3_pow_100_plus_5_mod_8 :
  (3^100 + 5) % 8 = 6 := by
sorry

end remainder_of_3_pow_100_plus_5_mod_8_l227_227542


namespace CD_expression_l227_227936

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (A B C A1 B1 C1 D : V)
variables (a b c : V)

-- Given conditions
axiom AB_eq_a : A - B = a
axiom AC_eq_b : A - C = b
axiom AA1_eq_c : A - A1 = c
axiom midpoint_D : D = (1/2) • (B1 + C1)

-- We need to show
theorem CD_expression : C - D = (1/2) • a - (1/2) • b + c :=
sorry

end CD_expression_l227_227936


namespace determine_b_l227_227124

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := 1 / (3 * x + b)
noncomputable def f_inv (x : ℝ) : ℝ := (2 - 3 * x) / (3 * x)

theorem determine_b (b : ℝ) :
    (∀ x : ℝ, f_inv (f x b) = x) ↔ b = -3 :=
by
  sorry

end determine_b_l227_227124


namespace inequality_proof_l227_227373

variable (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
variable (h_ab_bc_ca : a * b + b * c + c * a = 1)

theorem inequality_proof :
  (3 / Real.sqrt (a^2 + 1)) + (4 / Real.sqrt (b^2 + 1)) + (12 / Real.sqrt (c^2 + 1)) < (39 / 2) :=
by
  sorry

end inequality_proof_l227_227373


namespace simplify_expression_l227_227330

theorem simplify_expression :
  (-2) ^ 2006 + (-1) ^ 3007 + 1 ^ 3010 - (-2) ^ 2007 = -2 ^ 2006 := 
sorry

end simplify_expression_l227_227330


namespace systematic_sampling_l227_227347

theorem systematic_sampling (N n : ℕ) (hN : N = 1650) (hn : n = 35) :
  let E := 5 
  let segments := 35 
  let individuals_per_segment := 47 
  1650 % 35 = E ∧ 
  (1650 - E) / 35 = individuals_per_segment :=
by 
  sorry

end systematic_sampling_l227_227347


namespace spencer_walk_distance_l227_227806

theorem spencer_walk_distance :
  let distance_house_library := 0.3
  let distance_library_post_office := 0.1
  let total_distance := 0.8
  (total_distance - (distance_house_library + distance_library_post_office)) = 0.4 :=
by
  sorry

end spencer_walk_distance_l227_227806


namespace find_x_l227_227091

theorem find_x (x : ℤ) (h : 5 * x - 28 = 232) : x = 52 :=
by
  sorry

end find_x_l227_227091


namespace inequality_solution_set_l227_227266

theorem inequality_solution_set 
  (a b : ℝ) 
  (h : ∀ x, x ∈ Set.Icc (-3) 1 → ax^2 + (a + b)*x + 2 > 0) : 
  a + b = -4/3 := 
sorry

end inequality_solution_set_l227_227266


namespace coffee_amount_l227_227953

theorem coffee_amount (total_mass : ℕ) (coffee_ratio : ℕ) (milk_ratio : ℕ) (h_total_mass : total_mass = 4400) (h_coffee_ratio : coffee_ratio = 2) (h_milk_ratio : milk_ratio = 9) : 
  total_mass * coffee_ratio / (coffee_ratio + milk_ratio) = 800 :=
by
  -- Placeholder for the proof
  sorry

end coffee_amount_l227_227953


namespace cost_of_article_l227_227379

theorem cost_of_article (C : ℝ) (G : ℝ)
    (h1 : G = 520 - C)
    (h2 : 1.08 * G = 580 - C) :
    C = 230 :=
by
    sorry

end cost_of_article_l227_227379


namespace average_age_of_population_l227_227606

theorem average_age_of_population
  (k : ℕ)
  (ratio_women_men : 7 * (k : ℕ) = 7 * (k : ℕ) + 5 * (k : ℕ) - 5 * (k : ℕ))
  (avg_age_women : ℝ := 38)
  (avg_age_men : ℝ := 36)
  : ( (7 * k * avg_age_women) + (5 * k * avg_age_men) ) / (12 * k) = 37 + (1 / 6) :=
by
  sorry

end average_age_of_population_l227_227606


namespace max_value_of_g_l227_227704

def g (n : ℕ) : ℕ :=
  if n < 15 then n + 15 else g (n - 7)

theorem max_value_of_g : ∃ m, ∀ n, g n ≤ m ∧ (∃ k, g k = m) :=
by
  use 29
  sorry

end max_value_of_g_l227_227704


namespace loss_percentage_initial_selling_l227_227185

theorem loss_percentage_initial_selling (CP SP' : ℝ) 
  (hCP : CP = 1250) 
  (hSP' : SP' = CP * 1.15) 
  (h_diff : SP' - 500 = 937.5) : 
  (CP - 937.5) / CP * 100 = 25 := 
by 
  sorry

end loss_percentage_initial_selling_l227_227185


namespace problem1_inner_problem2_inner_l227_227305

-- Problem 1
theorem problem1_inner {m n : ℤ} (hm : |m| = 5) (hn : |n| = 4) (opposite_signs : m * n < 0) :
  m^2 - m * n + n = 41 ∨ m^2 - m * n + n = 49 :=
sorry

-- Problem 2
theorem problem2_inner {a b c d x : ℝ} (opposite_ab : a + b = 0) (reciprocals_cd : c * d = 1) (hx : |x| = 5) (hx_pos : x > 0) :
  3 * (a + b) - 2 * (c * d) + x = 3 :=
sorry

end problem1_inner_problem2_inner_l227_227305


namespace sandwiches_bought_l227_227030

theorem sandwiches_bought (sandwich_cost soda_cost total_cost_sodas total_cost : ℝ)
  (h1 : sandwich_cost = 2.45)
  (h2 : soda_cost = 0.87)
  (h3 : total_cost_sodas = 4 * soda_cost)
  (h4 : total_cost = 8.38) :
  ∃ (S : ℕ), sandwich_cost * S + total_cost_sodas = total_cost ∧ S = 2 :=
by
  use 2
  simp [h1, h2, h3, h4]
  sorry

end sandwiches_bought_l227_227030


namespace count_square_free_integers_l227_227088

def square_free_in_range_2_to_199 : Nat :=
  91

theorem count_square_free_integers :
  ∃ n : Nat, n = 91 ∧
  ∀ m : Nat, 2 ≤ m ∧ m < 200 →
  (∀ k : Nat, k^2 ∣ m → k^2 = 1) :=
by
  -- The proof will be filled here
  sorry

end count_square_free_integers_l227_227088


namespace packs_of_red_balls_l227_227873

/-
Julia bought some packs of red balls, R packs.
Julia bought 10 packs of yellow balls.
Julia bought 8 packs of green balls.
There were 19 balls in each package.
Julia bought 399 balls in total.
The goal is to prove that the number of packs of red balls Julia bought, R, is equal to 3.
-/

theorem packs_of_red_balls (R : ℕ) (balls_per_pack : ℕ) (packs_yellow : ℕ) (packs_green : ℕ) (total_balls : ℕ) 
  (h1 : balls_per_pack = 19) (h2 : packs_yellow = 10) (h3 : packs_green = 8) (h4 : total_balls = 399) 
  (h5 : total_balls = R * balls_per_pack + (packs_yellow + packs_green) * balls_per_pack) : 
  R = 3 :=
by
  -- Proof goes here
  sorry

end packs_of_red_balls_l227_227873


namespace total_cakes_served_l227_227729

def weekday_cakes_lunch : Nat := 6 + 8 + 10
def weekday_cakes_dinner : Nat := 9 + 7 + 5 + 13
def weekday_cakes_total : Nat := weekday_cakes_lunch + weekday_cakes_dinner

def weekend_cakes_lunch : Nat := 2 * (6 + 8 + 10)
def weekend_cakes_dinner : Nat := 2 * (9 + 7 + 5 + 13)
def weekend_cakes_total : Nat := weekend_cakes_lunch + weekend_cakes_dinner

def total_weekday_cakes : Nat := 5 * weekday_cakes_total
def total_weekend_cakes : Nat := 2 * weekend_cakes_total

def total_week_cakes : Nat := total_weekday_cakes + total_weekend_cakes

theorem total_cakes_served : total_week_cakes = 522 := by
  sorry

end total_cakes_served_l227_227729


namespace age_problem_l227_227310

theorem age_problem (F : ℝ) (M : ℝ) (Y : ℝ)
  (hF : F = 40.00000000000001)
  (hM : M = (2/5) * F)
  (hY : M + Y = (1/2) * (F + Y)) :
  Y = 8.000000000000002 :=
sorry

end age_problem_l227_227310


namespace milo_cash_reward_l227_227202

theorem milo_cash_reward : 
  let three_2s := [2, 2, 2]
  let four_3s := [3, 3, 3, 3]
  let one_4 := [4]
  let one_5 := [5]
  let all_grades := three_2s ++ four_3s ++ one_4 ++ one_5
  let total_grades := all_grades.length
  let total_sum := all_grades.sum
  let average_grade := total_sum / total_grades
  5 * average_grade = 15 := by
  sorry

end milo_cash_reward_l227_227202


namespace mini_toy_height_difference_l227_227792

variables (H_standard H_toy H_mini_diff : ℝ)

def poodle_heights : Prop :=
  H_standard = 28 ∧ H_toy = 14 ∧ H_standard - 8 = H_mini_diff + H_toy

theorem mini_toy_height_difference (H_standard H_toy H_mini_diff: ℝ) (h: poodle_heights H_standard H_toy H_mini_diff) :
  H_mini_diff = 6 :=
by {
  sorry
}

end mini_toy_height_difference_l227_227792


namespace total_volume_of_pyramids_l227_227720

theorem total_volume_of_pyramids :
  let base := 40
  let height_base := 20
  let height_pyramid := 30
  let area_base := (1 / 2) * base * height_base
  let volume_pyramid := (1 / 3) * area_base * height_pyramid
  3 * volume_pyramid = 12000 :=
by 
  sorry

end total_volume_of_pyramids_l227_227720


namespace min_value_expr_l227_227725

theorem min_value_expr (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_sum : a + b = 1) : 
  (b / (3 * a)) + (3 / b) ≥ 5 := 
sorry

end min_value_expr_l227_227725


namespace Jane_age_l227_227505

theorem Jane_age (J A : ℕ) (h1 : J + A = 54) (h2 : J - A = 22) : A = 16 := 
by 
  sorry

end Jane_age_l227_227505


namespace A_iff_B_l227_227359

-- Define Proposition A: ab > b^2
def PropA (a b : ℝ) : Prop := a * b > b ^ 2

-- Define Proposition B: 1/b < 1/a < 0
def PropB (a b : ℝ) : Prop := 1 / b < 1 / a ∧ 1 / a < 0

theorem A_iff_B (a b : ℝ) : (PropA a b) ↔ (PropB a b) := sorry

end A_iff_B_l227_227359


namespace cookies_per_person_l227_227574

theorem cookies_per_person (total_cookies : ℕ) (num_people : ℕ) (h1 : total_cookies = 35) (h2 : num_people = 5) :
  total_cookies / num_people = 7 := 
by {
  sorry
}

end cookies_per_person_l227_227574


namespace smallest_positive_angle_same_terminal_side_l227_227919

theorem smallest_positive_angle_same_terminal_side 
  (k : ℤ) : ∃ α : ℝ, 0 < α ∧ α < 360 ∧ -2002 = α + k * 360 ∧ α = 158 :=
by
  sorry

end smallest_positive_angle_same_terminal_side_l227_227919


namespace clothing_discounted_to_fraction_of_original_price_l227_227853

-- Given conditions
variable (P : ℝ) (f : ℝ)

-- Price during first sale is fP, price during second sale is 0.5P
-- Price decreased by 40% from first sale to second sale
def price_decrease_condition : Prop :=
  f * P - (1/2) * P = 0.4 * (f * P)

-- The main theorem to prove
theorem clothing_discounted_to_fraction_of_original_price (h : price_decrease_condition P f) :
  f = 5/6 :=
sorry

end clothing_discounted_to_fraction_of_original_price_l227_227853


namespace empty_can_mass_l227_227740

-- Define the mass of the full can
def full_can_mass : ℕ := 35

-- Define the mass of the can with half the milk
def half_can_mass : ℕ := 18

-- The theorem stating the mass of the empty can
theorem empty_can_mass : full_can_mass - (2 * (full_can_mass - half_can_mass)) = 1 := by
  sorry

end empty_can_mass_l227_227740


namespace number_of_bowls_l227_227121

theorem number_of_bowls (n : ℕ) :
  (∀ (b : ℕ), b > 0) →
  (∀ (a : ℕ), ∃ (k : ℕ), true) →
  (8 * 12 = 96) →
  (6 * n = 96) →
  n = 16 :=
by
  intros h1 h2 h3 h4
  sorry

end number_of_bowls_l227_227121


namespace total_rope_in_inches_l227_227293

-- Definitions for conditions
def feet_last_week : ℕ := 6
def feet_less : ℕ := 4
def inches_per_foot : ℕ := 12

-- Condition: rope bought this week
def feet_this_week := feet_last_week - feet_less

-- Condition: total rope bought in feet
def total_feet := feet_last_week + feet_this_week

-- Condition: total rope bought in inches
def total_inches := total_feet * inches_per_foot

-- Theorem statement
theorem total_rope_in_inches : total_inches = 96 := by
  sorry

end total_rope_in_inches_l227_227293


namespace yogurt_price_l227_227361

theorem yogurt_price (x y : ℝ) (h1 : 4 * x + 4 * y = 14) (h2 : 2 * x + 8 * y = 13) : x = 2.5 :=
by
  sorry

end yogurt_price_l227_227361


namespace minimum_value_of_u_l227_227170

noncomputable def minimum_value_lemma (x y : ℝ) (hx : Real.sin x + Real.sin y = 1 / 3) : Prop :=
  ∃ m : ℝ, m = -1/9 ∧ ∀ z, z = Real.sin x + Real.cos x ^ 2 → z ≥ m

theorem minimum_value_of_u
  (x y : ℝ)
  (hx : Real.sin x + Real.sin y = 1 / 3) :
  ∃ m : ℝ, m = -1/9 ∧ ∀ z, z = Real.sin x + Real.cos x ^ 2 → z ≥ m :=
sorry

end minimum_value_of_u_l227_227170


namespace total_monkeys_is_correct_l227_227523

-- Define the parameters
variables (m n : ℕ)

-- Define the conditions as separate definitions
def monkeys_on_n_bicycles : ℕ := 3 * n
def monkeys_on_remaining_bicycles : ℕ := 5 * (m - n)

-- Define the total number of monkeys
def total_monkeys : ℕ := monkeys_on_n_bicycles n + monkeys_on_remaining_bicycles m n

-- State the theorem
theorem total_monkeys_is_correct : total_monkeys m n = 5 * m - 2 * n :=
by
  sorry

end total_monkeys_is_correct_l227_227523


namespace train_speed_l227_227672

theorem train_speed (distance_meters : ℕ) (time_seconds : ℕ) 
  (h_distance : distance_meters = 150) (h_time : time_seconds = 20) : 
  distance_meters / 1000 / (time_seconds / 3600) = 27 :=
by 
  have h1 : distance_meters = 150 := h_distance
  have h2 : time_seconds = 20 := h_time
  -- other intermediate steps would go here, but are omitted
  -- for now, we assume the final calculation is:
  sorry

end train_speed_l227_227672


namespace shelby_gold_stars_l227_227749

theorem shelby_gold_stars (stars_yesterday stars_today : ℕ) (h1 : stars_yesterday = 4) (h2 : stars_today = 3) :
  stars_yesterday + stars_today = 7 := 
by
  sorry

end shelby_gold_stars_l227_227749


namespace abigail_money_loss_l227_227881

theorem abigail_money_loss
  (initial_amount : ℕ)
  (spent_amount : ℕ)
  (remaining_amount : ℕ)
  (h1 : initial_amount = 11)
  (h2 : spent_amount = 2)
  (h3 : remaining_amount = 3) :
  initial_amount - spent_amount - remaining_amount = 6 :=
by sorry

end abigail_money_loss_l227_227881


namespace minimum_sum_sequence_l227_227112

noncomputable def a_n (n : ℕ) : ℤ := 2 * n - 49

noncomputable def S_n (n : ℕ) : ℤ := (n * (a_n 1 + a_n n)) / 2

theorem minimum_sum_sequence : ∃ n : ℕ, S_n n = (n - 24) * (n - 24) - 24 * 24 ∧ (∀ m : ℕ, S_n m ≥ S_n n) ∧ n = 24 := 
by {
  sorry -- Proof omitted
}

end minimum_sum_sequence_l227_227112


namespace tangent_line_to_parabola_parallel_l227_227966

theorem tangent_line_to_parabola_parallel (m : ℝ) :
  ∀ (x y : ℝ), (y = x^2) → (2*x - y + m = 0 → m = -1) :=
by
  sorry

end tangent_line_to_parabola_parallel_l227_227966


namespace triangle_strike_interval_l227_227473

/-- Jacob strikes the cymbals every 7 beats and the triangle every t beats.
    Given both are struck at the same time every 14 beats, this proves t = 2. -/
theorem triangle_strike_interval :
  ∃ t : ℕ, t ≠ 7 ∧ (∀ n : ℕ, (7 * n % t = 0) → ∃ k : ℕ, 7 * n = 14 * k) ∧ t = 2 :=
by
  use 2
  sorry

end triangle_strike_interval_l227_227473


namespace fraction_zero_iff_numerator_zero_l227_227164

variable (x : ℝ)

def numerator (x : ℝ) : ℝ := x - 5
def denominator (x : ℝ) : ℝ := 6 * x + 12

theorem fraction_zero_iff_numerator_zero (h_denominator_nonzero : denominator 5 ≠ 0) : 
  numerator x / denominator x = 0 ↔ x = 5 :=
by sorry

end fraction_zero_iff_numerator_zero_l227_227164


namespace boards_per_package_calculation_l227_227659

-- Defining the conditions
def total_boards : ℕ := 154
def num_packages : ℕ := 52

-- Defining the division of total_boards by num_packages within rationals
def boards_per_package : ℚ := total_boards / num_packages

-- Prove that the boards per package is mathematically equal to the total boards divided by the number of packages
theorem boards_per_package_calculation :
  boards_per_package = 154 / 52 := by
  sorry

end boards_per_package_calculation_l227_227659


namespace equation_solution_l227_227138

theorem equation_solution (x : ℝ) (h : x + 1/x = 2.5) : x^2 + 1/x^2 = 4.25 := 
by sorry

end equation_solution_l227_227138


namespace find_weight_per_square_inch_l227_227345

-- Define the TV dimensions and other given data
def bill_tv_width : ℕ := 48
def bill_tv_height : ℕ := 100
def bob_tv_width : ℕ := 70
def bob_tv_height : ℕ := 60
def weight_difference_pounds : ℕ := 150
def ounces_per_pound : ℕ := 16

-- Compute areas
def bill_tv_area := bill_tv_width * bill_tv_height
def bob_tv_area := bob_tv_width * bob_tv_height

-- Assume weight per square inch
def weight_per_square_inch : ℕ := 4

-- Total weight computation given in ounces
def bill_tv_weight := bill_tv_area * weight_per_square_inch
def bob_tv_weight := bob_tv_area * weight_per_square_inch
def weight_difference_ounces := weight_difference_pounds * ounces_per_pound

-- The theorem to prove
theorem find_weight_per_square_inch : 
  bill_tv_weight - bob_tv_weight = weight_difference_ounces → weight_per_square_inch = 4 :=
by
  intros
  /- Proof by computation -/
  sorry

end find_weight_per_square_inch_l227_227345


namespace incorrect_option_A_l227_227939

theorem incorrect_option_A (x y : ℝ) :
  ¬(5 * x + y / 2 = (5 * x + y) / 2) :=
by sorry

end incorrect_option_A_l227_227939


namespace x_intercept_is_2_l227_227409

noncomputable def x_intercept_of_line : ℝ :=
  by
  sorry -- This is where the proof would go

theorem x_intercept_is_2 :
  (∀ x y : ℝ, 5 * x - 2 * y - 10 = 0 → y = 0 → x = 2) :=
  by
  intro x y H_eq H_y0
  rw [H_y0] at H_eq
  simp at H_eq
  sorry -- This is where the proof would go

end x_intercept_is_2_l227_227409


namespace fourth_vertex_of_parallelogram_l227_227343

structure Point where
  x : ℤ
  y : ℤ

def midPoint (P Q : Point) : Point :=
  { x := (P.x + Q.x) / 2, y := (P.y + Q.y) / 2 }

def isMidpoint (M P Q : Point) : Prop :=
  M = midPoint P Q

theorem fourth_vertex_of_parallelogram (A B C D : Point)
  (hA : A = {x := -2, y := 1})
  (hB : B = {x := -1, y := 3})
  (hC : C = {x := 3, y := 4})
  (h1 : isMidpoint (midPoint A C) B D ∨
        isMidpoint (midPoint A B) C D ∨
        isMidpoint (midPoint B C) A D) :
  D = {x := 2, y := 2} ∨ D = {x := -6, y := 0} ∨ D = {x := 4, y := 6} := by
  sorry

end fourth_vertex_of_parallelogram_l227_227343


namespace smallest_diff_of_YZ_XY_l227_227941

theorem smallest_diff_of_YZ_XY (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 2509) (h4 : a + b > c) (h5 : b + c > a) (h6 : a + c > b) : b - a = 1 :=
by {
  sorry
}

end smallest_diff_of_YZ_XY_l227_227941


namespace rubber_band_problem_l227_227597

noncomputable def a : ℤ := 4
noncomputable def b : ℤ := 12
noncomputable def c : ℤ := 3
noncomputable def band_length := a * Real.pi + b * Real.sqrt c

theorem rubber_band_problem (r1 r2 d : ℝ) (h1 : r1 = 3) (h2 : r2 = 9) (h3 : d = 12) :
  let a := 4
  let b := 12
  let c := 3
  let band_length := a * Real.pi + b * Real.sqrt c
  a + b + c = 19 :=
by
  sorry

end rubber_band_problem_l227_227597


namespace area_of_circle_l227_227983

open Real

theorem area_of_circle :
  ∃ (A : ℝ), (∀ x y : ℝ, (x^2 + y^2 - 4 * x + 2 * y - 11 = 0) → A = 16 * π) :=
sorry

end area_of_circle_l227_227983


namespace candidate_votes_percentage_l227_227817

-- Conditions
variables {P : ℝ} 
variables (totalVotes : ℝ := 8000)
variables (differenceVotes : ℝ := 2400)

-- Proof Problem
theorem candidate_votes_percentage (h : ((P / 100) * totalVotes + ((P / 100) * totalVotes + differenceVotes) = totalVotes)) : P = 35 :=
by
  sorry

end candidate_votes_percentage_l227_227817


namespace sara_letters_ratio_l227_227848

variable (L_J : ℕ) (L_F : ℕ) (L_T : ℕ)

theorem sara_letters_ratio (hLJ : L_J = 6) (hLF : L_F = 9) (hLT : L_T = 33) : 
  (L_T - (L_J + L_F)) / L_J = 3 := by
  sorry

end sara_letters_ratio_l227_227848


namespace inverse_f_486_l227_227035

-- Define the function f with given properties.
def f : ℝ → ℝ := sorry

-- Condition 1: f(5) = 2
axiom f_at_5 : f 5 = 2

-- Condition 2: f(3x) = 3f(x) for all x
axiom f_scale : ∀ x, f (3 * x) = 3 * f x

-- Proposition: f⁻¹(486) = 1215
theorem inverse_f_486 : (∃ x, f x = 486) → ∀ x, f x = 486 → x = 1215 :=
by sorry

end inverse_f_486_l227_227035


namespace range_of_m_l227_227532

theorem range_of_m (x y m : ℝ) 
  (h1 : x - 2 * y = 1) 
  (h2 : 2 * x + y = 4 * m) 
  (h3 : x + 3 * y < 6) : 
  m < 7 / 4 := 
sorry

end range_of_m_l227_227532


namespace distance_to_city_center_l227_227595

theorem distance_to_city_center 
  (D : ℕ) 
  (H1 : D = 200 + 200 + D) 
  (H_total : 900 = 200 + 200 + D) : 
  D = 500 :=
by { sorry }

end distance_to_city_center_l227_227595


namespace gcd_lcm_sum_l227_227233

theorem gcd_lcm_sum (a b : ℕ) (h₁ : a = 120) (h₂ : b = 3507) :
  Nat.gcd a b + Nat.lcm a b = 140283 := by 
  sorry

end gcd_lcm_sum_l227_227233


namespace problem_statement_l227_227981

/-- For any positive integer n, given θ ∈ (0, π) and x ∈ ℂ such that 
x + 1/x = 2√2 cos θ - sin θ, it follows that x^n + 1/x^n = 2 cos (n α). -/
theorem problem_statement (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < π)
  (x : ℂ) (hx : x + 1/x = 2 * (2:ℝ).sqrt * θ.cos - θ.sin)
  (n : ℕ) (hn : 0 < n) : x^n + x⁻¹^n = 2 * θ.cos * n := 
  sorry

end problem_statement_l227_227981


namespace sum_coefficients_l227_227944

theorem sum_coefficients (a : ℤ) (f : ℤ → ℤ) :
  f x = (1 - 2 * x)^7 ∧ a_0 = f 0 ∧ a_1_plus_a_7 = f 1 - f 0 
→ a_1_plus_a_7 = -2 :=
by sorry

end sum_coefficients_l227_227944


namespace large_box_times_smaller_box_l227_227909

noncomputable def large_box_volume (width length height : ℕ) : ℕ := width * length * height

noncomputable def small_box_volume (width length height : ℕ) : ℕ := width * length * height

theorem large_box_times_smaller_box :
  let large_volume := large_box_volume 30 20 5
  let small_volume := small_box_volume 6 4 1
  large_volume / small_volume = 125 :=
by
  let large_volume := large_box_volume 30 20 5
  let small_volume := small_box_volume 6 4 1
  show large_volume / small_volume = 125
  sorry

end large_box_times_smaller_box_l227_227909


namespace find_x4_l227_227746

theorem find_x4 (x_1 x_2 : ℝ) (h1 : 0 < x_1) (h2 : x_1 < x_2) 
  (P : (ℝ × ℝ)) (Q : (ℝ × ℝ)) (hP : P = (2, Real.log 2)) 
  (hQ : Q = (500, Real.log 500)) 
  (R : (ℝ × ℝ)) (x_4 : ℝ) :
  R = ((x_1 + x_2) / 2, (Real.log x_1 + Real.log x_2) / 2) →
  Real.log x_4 = (Real.log x_1 + Real.log x_2) / 2 →
  x_4 = Real.sqrt 1000 :=
by 
  intro hR hT
  sorry

end find_x4_l227_227746


namespace no_square_from_vertices_of_equilateral_triangles_l227_227689

-- Definitions
def equilateral_triangle_grid (p : ℝ × ℝ) : Prop := 
  ∃ k l : ℤ, p.1 = k * (1 / 2) ∧ p.2 = l * (Real.sqrt 3 / 2)

def form_square_by_vertices (A B C D : ℝ × ℝ) : Prop := 
  (A.1 = 0 ∧ A.2 = 0) ∧ 
  (B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2 = (C.1 - A.1) ^ 2 + (C.2 - A.2) ^ 2 ∧ 
  (C.1 - B.1) ^ 2 + (C.2 - B.2) ^ 2 = (D.1 - B.1) ^ 2 + (D.2 - B.2) ^ 2 ∧ 
  (D.1 - C.1) ^ 2 + (D.2 - C.2) ^ 2 = (A.1 - D.1) ^ 2 + (A.2 - D.2) ^ 2
  
-- Problem Statement
theorem no_square_from_vertices_of_equilateral_triangles :
  ¬ ∃ (A B C D : ℝ × ℝ), 
    equilateral_triangle_grid A ∧ 
    equilateral_triangle_grid B ∧ 
    equilateral_triangle_grid C ∧ 
    equilateral_triangle_grid D ∧ 
    form_square_by_vertices A B C D :=
by
  sorry

end no_square_from_vertices_of_equilateral_triangles_l227_227689


namespace find_A_l227_227155

-- Given a three-digit number AB2 such that AB2 - 41 = 591
def valid_number (A B : ℕ) : Prop :=
  (A * 100) + (B * 10) + 2 - 41 = 591

-- We aim to prove that A = 6 given B = 2
theorem find_A (A : ℕ) (B : ℕ) (hB : B = 2) : A = 6 :=
  by
  have h : valid_number A B := by sorry
  sorry

end find_A_l227_227155


namespace smallest_n_not_divisible_by_10_l227_227931

theorem smallest_n_not_divisible_by_10 :
  ∃ n : ℕ, n > 2016 ∧ (1^n + 2^n + 3^n + 4^n) % 10 ≠ 0 ∧ n = 2020 := by
  sorry

end smallest_n_not_divisible_by_10_l227_227931


namespace remaining_unit_area_l227_227666

theorem remaining_unit_area
    (total_units : ℕ)
    (total_area : ℕ)
    (num_12x6_units : ℕ)
    (length_12x6_unit : ℕ)
    (width_12x6_unit : ℕ)
    (remaining_units_area : ℕ)
    (num_remaining_units : ℕ)
    (remaining_unit_area : ℕ) :
  total_units = 72 →
  total_area = 8640 →
  num_12x6_units = 30 →
  length_12x6_unit = 12 →
  width_12x6_unit = 6 →
  remaining_units_area = total_area - (num_12x6_units * length_12x6_unit * width_12x6_unit) →
  num_remaining_units = total_units - num_12x6_units →
  remaining_unit_area = remaining_units_area / num_remaining_units →
  remaining_unit_area = 154 :=
by
  intros h_total_units h_total_area h_num_12x6_units h_length_12x6_unit h_width_12x6_unit h_remaining_units_area h_num_remaining_units h_remaining_unit_area
  sorry

end remaining_unit_area_l227_227666


namespace find_numbers_with_conditions_l227_227904

theorem find_numbers_with_conditions (n : ℕ) (hn1 : n % 100 = 0) (hn2 : (n.divisors).card = 12) : 
  n = 200 ∨ n = 500 :=
by
  sorry

end find_numbers_with_conditions_l227_227904


namespace total_amount_spent_correct_l227_227041

noncomputable def total_amount_spent (mango_cost pineapple_cost cost_pineapple total_people : ℕ) : ℕ :=
  let pineapple_people := cost_pineapple / pineapple_cost
  let mango_people := total_people - pineapple_people
  let mango_cost_total := mango_people * mango_cost
  cost_pineapple + mango_cost_total

theorem total_amount_spent_correct :
  total_amount_spent 5 6 54 17 = 94 := by
  -- This is where the proof would go, but it's omitted per instructions
  sorry

end total_amount_spent_correct_l227_227041


namespace smallest_possible_sum_l227_227135

theorem smallest_possible_sum (E F G H : ℕ) (h1 : F > 0) (h2 : E + F + G = 3 * F) (h3 : F * G = 4 * F * F / 3) :
  E = 6 ∧ F = 9 ∧ G = 12 ∧ H = 16 ∧ E + F + G + H = 43 :=
by 
  sorry

end smallest_possible_sum_l227_227135


namespace find_s_for_g_l227_227153

def g (x : ℝ) (s : ℝ) : ℝ := 3*x^4 - 2*x^3 + 2*x^2 + x + s

theorem find_s_for_g (s : ℝ) : g (-1) s = 0 ↔ s = -6 :=
by
  sorry

end find_s_for_g_l227_227153


namespace rectangular_to_polar_coordinates_l227_227823

theorem rectangular_to_polar_coordinates :
  ∃ r θ, (r > 0) ∧ (0 ≤ θ ∧ θ < 2 * Real.pi) ∧ (r, θ) = (5, 7 * Real.pi / 4) :=
by
  sorry

end rectangular_to_polar_coordinates_l227_227823


namespace gcd_2048_2101_eq_1_l227_227303

theorem gcd_2048_2101_eq_1 : Int.gcd 2048 2101 = 1 := sorry

end gcd_2048_2101_eq_1_l227_227303


namespace brenda_bought_stones_l227_227863

-- Given Conditions
def n_bracelets : ℕ := 3
def n_stones_per_bracelet : ℕ := 12

-- Problem Statement: Prove Betty bought the correct number of stone-shaped stars
theorem brenda_bought_stones :
  let n_total_stones := n_bracelets * n_stones_per_bracelet
  n_total_stones = 36 := 
by 
  -- proof goes here, but we omit it with sorry
  sorry

end brenda_bought_stones_l227_227863


namespace prove_y_value_l227_227232

theorem prove_y_value (x y : ℤ) (h1 : x - y = 8) (h2 : x + y = 10) : y = 1 :=
by
  sorry

end prove_y_value_l227_227232


namespace greatest_t_solution_l227_227550

theorem greatest_t_solution :
  ∀ t : ℝ, t ≠ 8 ∧ t ≠ -5 →
  (t^2 - t - 40) / (t - 8) = 5 / (t + 5) →
  t ≤ -2 :=
by
  sorry

end greatest_t_solution_l227_227550


namespace number_multiplied_by_3_l227_227408

theorem number_multiplied_by_3 (k : ℕ) : 
  2^13 - 2^(13-2) = 3 * k → k = 2048 :=
by
  sorry

end number_multiplied_by_3_l227_227408


namespace homework_duration_reduction_l227_227418

theorem homework_duration_reduction (x : ℝ) (initial_duration final_duration : ℝ) (h_initial : initial_duration = 90) (h_final : final_duration = 60) : 
  90 * (1 - x)^2 = 60 :=
by
  sorry

end homework_duration_reduction_l227_227418


namespace mike_spent_on_new_tires_l227_227424

-- Define the given amounts
def amount_spent_on_speakers : ℝ := 118.54
def total_amount_spent_on_car_parts : ℝ := 224.87

-- Define the amount spent on new tires
def amount_spent_on_new_tires : ℝ := total_amount_spent_on_car_parts - amount_spent_on_speakers

-- The theorem we want to prove
theorem mike_spent_on_new_tires : amount_spent_on_new_tires = 106.33 :=
by
  -- the proof would go here
  sorry

end mike_spent_on_new_tires_l227_227424


namespace calculate_total_customers_l227_227273

theorem calculate_total_customers 
    (num_no_tip : ℕ) 
    (total_tip_amount : ℕ) 
    (tip_per_customer : ℕ) 
    (number_tipped_customers : ℕ) 
    (number_total_customers : ℕ)
    (h1 : num_no_tip = 5) 
    (h2 : total_tip_amount = 15) 
    (h3 : tip_per_customer = 3) 
    (h4 : number_tipped_customers = total_tip_amount / tip_per_customer) :
    number_total_customers = number_tipped_customers + num_no_tip := 
by {
    sorry
}

end calculate_total_customers_l227_227273


namespace circle_equation_l227_227514

theorem circle_equation (x y : ℝ) (h : ∀ x y : ℝ, x^2 + y^2 ≥ 64) :
  x^2 + y^2 - 64 = 0 ↔ x = 0 ∧ y = 0 :=
by
  sorry

end circle_equation_l227_227514


namespace mn_plus_one_unequal_pos_integers_l227_227089

theorem mn_plus_one_unequal_pos_integers (m n : ℕ) 
  (S : Finset ℕ) (h_card : S.card = m * n + 1) :
  (∃ (b : Fin (m + 1) → ℕ), (∀ i j : Fin (m + 1), i ≠ j → ¬(b i ∣ b j)) ∧ (∀ i : Fin (m + 1), b i ∈ S)) ∨ 
  (∃ (a : Fin (n + 1) → ℕ), (∀ i : Fin n, a i ∣ a (i + 1)) ∧ (∀ i : Fin (n + 1), a i ∈ S)) :=
sorry

end mn_plus_one_unequal_pos_integers_l227_227089


namespace fraction_zero_value_l227_227346

theorem fraction_zero_value (x : ℝ) (h : (3 - x) ≠ 0) : (x+2)/(3-x) = 0 ↔ x = -2 := by
  sorry

end fraction_zero_value_l227_227346


namespace sqrt_ax3_eq_negx_sqrt_ax_l227_227149

variable (a x : ℝ)
variable (ha : a < 0) (hx : x < 0)

theorem sqrt_ax3_eq_negx_sqrt_ax : Real.sqrt (a * x^3) = -x * Real.sqrt (a * x) := by
  sorry

end sqrt_ax3_eq_negx_sqrt_ax_l227_227149


namespace solve_system_l227_227271

theorem solve_system :
  ∃ x y z : ℝ, (8 * (x^3 + y^3 + z^3) = 73) ∧
              (2 * (x^2 + y^2 + z^2) = 3 * (x * y + y * z + z * x)) ∧
              (x * y * z = 1) ∧
              (x, y, z) = (1, 2, 0.5) ∨ (x, y, z) = (1, 0.5, 2) ∨
              (x, y, z) = (2, 1, 0.5) ∨ (x, y, z) = (2, 0.5, 1) ∨
              (x, y, z) = (0.5, 1, 2) ∨ (x, y, z) = (0.5, 2, 1) :=
by
  sorry

end solve_system_l227_227271


namespace symmetric_points_sum_l227_227136

theorem symmetric_points_sum (n m : ℤ) 
  (h₁ : (3 : ℤ) = m)
  (h₂ : n = (-5 : ℤ)) : 
  m + n = (-2 : ℤ) := 
by 
  sorry

end symmetric_points_sum_l227_227136


namespace max_correct_answers_l227_227984

theorem max_correct_answers (a b c : ℕ) (n : ℕ := 60) (p_correct : ℤ := 5) (p_blank : ℤ := 0) (p_incorrect : ℤ := -2) (S : ℤ := 150) :
        a + b + c = n ∧ p_correct * a + p_blank * b + p_incorrect * c = S → a ≤ 38 :=
by
  sorry

end max_correct_answers_l227_227984


namespace minimum_f_l227_227009

def f (x y : ℤ) : ℤ := |5 * x^2 + 11 * x * y - 5 * y^2|

theorem minimum_f (x y : ℤ) (h : x ≠ 0 ∨ y ≠ 0) : ∃ (m : ℤ), m = 5 ∧ ∀ (x y : ℤ), (x ≠ 0 ∨ y ≠ 0) → f x y ≥ m :=
by sorry

end minimum_f_l227_227009


namespace find_function_l227_227152

theorem find_function (f : ℤ → ℤ) :
  (∀ x y : ℤ, f (x + y) = f x + f y - 2023) →
  ∃ c : ℤ, ∀ x : ℤ, f x = c * x + 2023 :=
by
  intros h
  sorry

end find_function_l227_227152


namespace students_in_game_divisors_of_119_l227_227988

theorem students_in_game_divisors_of_119 (n : ℕ) (h1 : ∃ (k : ℕ), k * n = 119) :
  n = 7 ∨ n = 17 :=
sorry

end students_in_game_divisors_of_119_l227_227988


namespace min_value_arith_geo_seq_l227_227492

theorem min_value_arith_geo_seq (A B C D : ℕ) (h1 : 0 < A) (h2 : 0 < B) (h3 : 0 < C) (h4 : 0 < D)
  (h_arith : C - B = B - A) (h_geo : C * C = B * D) (h_frac : 4 * C = 7 * B) :
  A + B + C + D = 97 :=
sorry

end min_value_arith_geo_seq_l227_227492


namespace simplify_fraction_l227_227833

open Complex

theorem simplify_fraction :
  (3 + 3 * I) / (-1 + 3 * I) = -1.2 - 1.2 * I :=
by
  sorry

end simplify_fraction_l227_227833


namespace eval_expression_l227_227368

theorem eval_expression (a b : ℤ) (h1 : a = 3) (h2 : b = 2) :
  (a^3 + b)^2 - (a^3 - b)^2 = 216 := 
by 
  sorry

end eval_expression_l227_227368


namespace find_abc_l227_227237

noncomputable def f (x : ℝ) : ℝ := 1 + 2 * Real.cos x + 3 * Real.sin x

theorem find_abc (a b c : ℝ) : 
  (∀ x : ℝ, a * f x + b * f (x - c) = 1) →
  (∃ n : ℤ, a = 1 / 2 ∧ b = 1 / 2 ∧ c = (2 * n + 1) * Real.pi) :=
by
  sorry

end find_abc_l227_227237


namespace least_amount_of_money_l227_227280

variable (Money : Type) [LinearOrder Money]
variable (Anne Bo Coe Dan El : Money)

-- Conditions from the problem
axiom anne_less_than_bo : Anne < Bo
axiom dan_less_than_bo : Dan < Bo
axiom coe_less_than_anne : Coe < Anne
axiom coe_less_than_el : Coe < El
axiom coe_less_than_dan : Coe < Dan
axiom dan_less_than_anne : Dan < Anne

theorem least_amount_of_money : (∀ x, x = Anne ∨ x = Bo ∨ x = Coe ∨ x = Dan ∨ x = El → Coe < x) :=
by
  sorry

end least_amount_of_money_l227_227280


namespace composite_prime_fraction_l227_227340

theorem composite_prime_fraction :
  let P1 : ℕ := 4 * 6 * 8 * 9 * 10 * 12 * 14 * 15
  let P2 : ℕ := 16 * 18 * 20 * 21 * 22 * 24 * 25 * 26
  let first_prime : ℕ := 2
  let second_prime : ℕ := 3
  (P1 + first_prime) / (P2 + second_prime) =
    (4 * 6 * 8 * 9 * 10 * 12 * 14 * 15 + 2) / (16 * 18 * 20 * 21 * 22 * 24 * 25 * 26 + 3) := by
  sorry

end composite_prime_fraction_l227_227340


namespace range_of_k_l227_227264

noncomputable def h (x : ℝ) (k : ℝ) : ℝ := 2 * x - k / x + k / 3

theorem range_of_k (k : ℝ) :
  (∀ x : ℝ, 1 < x → 2 + k / x^2 > 0) ↔ k ≥ -2 :=
by
  sorry

end range_of_k_l227_227264


namespace ratio_of_term_to_difference_l227_227957

def arithmetic_progression_sum (n a d : ℕ) : ℕ := (n * (2 * a + (n - 1) * d)) / 2

theorem ratio_of_term_to_difference (a d : ℕ) 
  (h1: arithmetic_progression_sum 7 a d = arithmetic_progression_sum 3 a d + 20)
  (h2 : d ≠ 0) : a / d = 1 / 2 := 
by 
  sorry

end ratio_of_term_to_difference_l227_227957


namespace scientific_notation_216000_l227_227054

theorem scientific_notation_216000 : 216000 = 2.16 * 10^5 :=
by
  -- proof will be provided here
  sorry

end scientific_notation_216000_l227_227054


namespace max_ballpoint_pens_l227_227732

theorem max_ballpoint_pens (x y z : ℕ) (hx : x + y + z = 15)
  (hy : 10 * x + 40 * y + 60 * z = 500) (hz : x ≥ 1 ∧ y ≥ 1 ∧ z ≥ 1) :
  x ≤ 6 :=
sorry

end max_ballpoint_pens_l227_227732


namespace min_value_a_l227_227776

theorem min_value_a (a : ℝ) : (∀ x : ℝ, a < x → 2 * x + 2 / (x - a) ≥ 7) → a ≥ 3 / 2 :=
by
  sorry

end min_value_a_l227_227776


namespace distance_from_Q_to_EF_is_24_div_5_l227_227553

-- Define the configuration of the square and points
def E := (0, 8)
def F := (8, 8)
def G := (8, 0)
def H := (0, 0)
def N := (4, 0) -- Midpoint of GH
def r1 := 4 -- Radius of the circle centered at N
def r2 := 8 -- Radius of the circle centered at E

-- Definition of the first circle centered at N with radius r1
def circle1 (x y : ℝ) := (x - 4)^2 + y^2 = r1^2

-- Definition of the second circle centered at E with radius r2
def circle2 (x y : ℝ) := x^2 + (y - 8)^2 = r2^2

-- Define the intersection point Q, other than H
def Q := (32 / 5, 16 / 5) -- Found as an intersection point between circle1 and circle2

-- Define the distance from point Q to the line EF
def dist_to_EF := 8 - (Q.2) -- (Q.2 is the y-coordinate of Q)

-- The main statement to prove
theorem distance_from_Q_to_EF_is_24_div_5 : dist_to_EF = 24 / 5 := by
  sorry

end distance_from_Q_to_EF_is_24_div_5_l227_227553


namespace find_investment_period_l227_227503

variable (P : ℝ) (r : ℝ) (n : ℕ) (A : ℝ)

theorem find_investment_period (hP : P = 12000)
                               (hr : r = 0.10)
                               (hn : n = 2)
                               (hA : A = 13230) :
                               ∃ t : ℝ, A = P * (1 + r / n)^(n * t) ∧ t = 1 := 
by
  sorry

end find_investment_period_l227_227503


namespace burglar_goods_value_l227_227702

theorem burglar_goods_value (V : ℝ) (S : ℝ) (S_increased : ℝ) (S_total : ℝ) (h1 : S = V / 5000) (h2 : S_increased = 1.25 * S) (h3 : S_total = S_increased + 2) (h4 : S_total = 12) : V = 40000 := by
  sorry

end burglar_goods_value_l227_227702


namespace correct_equation_solves_time_l227_227661

noncomputable def solve_time_before_stop (t : ℝ) : Prop :=
  let total_trip_time := 4 -- total trip time in hours including stop
  let stop_time := 0.5 -- stop time in hours
  let total_distance := 180 -- total distance in km
  let speed_before_stop := 60 -- speed before stop in km/h
  let speed_after_stop := 80 -- speed after stop in km/h
  let time_after_stop := total_trip_time - stop_time - t -- time after the stop in hours
  speed_before_stop * t + speed_after_stop * time_after_stop = total_distance -- distance equation

-- The theorem states that the equation is valid for solving t
theorem correct_equation_solves_time :
  solve_time_before_stop t = (60 * t + 80 * (7/2 - t) = 180) :=
sorry -- proof not required

end correct_equation_solves_time_l227_227661


namespace percentage_increase_in_area_is_96_l227_227802

theorem percentage_increase_in_area_is_96 :
  let r₁ := 5
  let r₃ := 7
  let A (r : ℝ) := Real.pi * r^2
  ((A r₃ - A r₁) / A r₁) * 100 = 96 := by
  sorry

end percentage_increase_in_area_is_96_l227_227802


namespace sin_405_eq_sqrt2_div_2_l227_227429

theorem sin_405_eq_sqrt2_div_2 :
  Real.sin (405 * Real.pi / 180) = Real.sqrt 2 / 2 := by
  sorry

end sin_405_eq_sqrt2_div_2_l227_227429


namespace largest_possible_sum_l227_227252

def max_sum_pair_mult_48 : Prop :=
  ∃ (heartsuit clubsuit : ℕ), (heartsuit * clubsuit = 48) ∧ (heartsuit + clubsuit = 49) ∧ 
  (∀ (h c : ℕ), (h * c = 48) → (h + c ≤ 49))

theorem largest_possible_sum : max_sum_pair_mult_48 :=
  sorry

end largest_possible_sum_l227_227252


namespace island_length_l227_227290

theorem island_length (area width : ℝ) (h_area : area = 50) (h_width : width = 5) : 
  area / width = 10 := 
by
  sorry

end island_length_l227_227290


namespace sufficient_but_not_necessary_condition_l227_227578

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (a > 4 → a^2 > 16) ∧ (∃ a, (a < -4) ∧ (a^2 > 16)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l227_227578


namespace arithmetic_seq_sixth_term_l227_227780

theorem arithmetic_seq_sixth_term
  (a d : ℤ)
  (h1 : a + d = 14)
  (h2 : a + 3 * d = 32) : a + 5 * d = 50 := 
by
  sorry

end arithmetic_seq_sixth_term_l227_227780


namespace find_initial_maple_trees_l227_227560

def initial_maple_trees (final_maple_trees planted_maple_trees : ℕ) : ℕ :=
  final_maple_trees - planted_maple_trees

theorem find_initial_maple_trees : initial_maple_trees 11 9 = 2 := by
  sorry

end find_initial_maple_trees_l227_227560


namespace number_of_juniors_l227_227701

variables (J S x : ℕ)

theorem number_of_juniors (h1 : (2 / 5 : ℚ) * J = x)
                          (h2 : (1 / 4 : ℚ) * S = x)
                          (h3 : J + S = 30) :
  J = 11 :=
sorry

end number_of_juniors_l227_227701


namespace necessary_and_sufficient_condition_l227_227646

theorem necessary_and_sufficient_condition (a : ℝ) :
  (∀ x : ℝ, x^2 + a * x - 4 * a ≥ 0) ↔ (-16 ≤ a ∧ a ≤ 0) :=
sorry

end necessary_and_sufficient_condition_l227_227646


namespace initial_percentage_acid_l227_227585

theorem initial_percentage_acid (P : ℝ) (h1 : 27 * P / 100 = 18 * 60 / 100) : P = 40 :=
sorry

end initial_percentage_acid_l227_227585


namespace prove_f_2_eq_10_prove_f_4_eq_10_prove_f_prove_f_l227_227731

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom f_cond1 : ∀ x : ℝ, f x + (deriv g x) = 10
axiom f_cond2 : ∀ x : ℝ, f x - (deriv g (4 - x)) = 10
axiom g_even : ∀ x : ℝ, g x = g (-x)

theorem prove_f_2_eq_10 : f 2 = 10 := sorry
theorem prove_f_4_eq_10 : f 4 = 10 := sorry
theorem prove_f'_neg1_eq_f'_neg3 : deriv f (-1) = deriv f (-3) := sorry
theorem prove_f'_2023_ne_0 : deriv f 2023 ≠ 0 := sorry

end prove_f_2_eq_10_prove_f_4_eq_10_prove_f_prove_f_l227_227731


namespace min_value_x3_l227_227987

noncomputable def min_x3 (x1 x2 x3 : ℝ) : ℝ := -21 / 11

theorem min_value_x3 (x1 x2 x3 : ℝ) 
  (h1 : x1 + (1 / 2) * x2 + (1 / 3) * x3 = 1)
  (h2 : x1^2 + (1 / 2) * x2^2 + (1 / 3) * x3^2 = 3) 
  : x3 ≥ - (21 / 11) := 
by sorry

end min_value_x3_l227_227987


namespace count_books_in_row_on_tuesday_l227_227985

-- Define the given conditions
def tiles_count_monday : ℕ := 38
def books_count_monday : ℕ := 75
def total_count_tuesday : ℕ := 301
def tiles_count_tuesday := tiles_count_monday * 2

-- The Lean statement we need to prove
theorem count_books_in_row_on_tuesday (hcbooks : books_count_monday = 75) 
(hc1 : total_count_tuesday = 301) 
(hc2 : tiles_count_tuesday = tiles_count_monday * 2):
  (total_count_tuesday - tiles_count_tuesday) / books_count_monday = 3 :=
by
  sorry

end count_books_in_row_on_tuesday_l227_227985


namespace other_number_is_7_l227_227209

-- Given conditions
variable (a b : ℤ)
variable (h1 : 2 * a + 3 * b = 110)
variable (h2 : a = 32 ∨ b = 32)

-- The proof goal
theorem other_number_is_7 : (a = 7 ∧ b = 32) ∨ (a = 32 ∧ b = 7) :=
by
  sorry

end other_number_is_7_l227_227209


namespace rent_budget_l227_227918

variables (food_per_week : ℝ) (weekly_food_budget : ℝ) (video_streaming : ℝ)
          (cell_phone : ℝ) (savings : ℝ) (rent : ℝ)
          (total_spending : ℝ)

-- Conditions
def food_budget := food_per_week * 4 = weekly_food_budget
def video_streaming_budget := video_streaming = 30
def cell_phone_budget := cell_phone = 50
def savings_budget := savings = 0.1 * total_spending
def savings_amount := savings = 198

-- Prove
theorem rent_budget (h1 : food_budget food_per_week weekly_food_budget)
                    (h2 : video_streaming_budget video_streaming)
                    (h3 : cell_phone_budget cell_phone)
                    (h4 : savings_budget savings total_spending)
                    (h5 : savings_amount savings) :
  rent = 1500 :=
sorry

end rent_budget_l227_227918


namespace revenue_increase_l227_227140

theorem revenue_increase (n : ℕ) (C P : ℝ) 
  (h1 : n * P = 1.20 * C) : 
  (0.95 * n * P) = 1.14 * C :=
by
  sorry

end revenue_increase_l227_227140


namespace decimal_to_base13_185_l227_227526

theorem decimal_to_base13_185 : 
  ∀ n : ℕ, n = 185 → 
      ∃ a b c : ℕ, a * 13^2 + b * 13 + c = n ∧ 0 ≤ a ∧ a < 13 ∧ 0 ≤ b ∧ b < 13 ∧ 0 ≤ c ∧ c < 13 ∧ (a, b, c) = (1, 1, 3) := 
by
  intros n hn
  use 1, 1, 3
  sorry

end decimal_to_base13_185_l227_227526


namespace compute_value_l227_227304

def Δ (p q : ℕ) : ℕ := p^3 - q

theorem compute_value : Δ (5^Δ 2 7) (4^Δ 4 8) = 125 - 4^56 := by
  sorry

end compute_value_l227_227304


namespace tangent_line_equation_parallel_to_given_line_l227_227596

theorem tangent_line_equation_parallel_to_given_line :
  ∃ (x y : ℝ),  y = x^3 - 3 * x^2
    ∧ (3 * x^2 - 6 * x = -3)
    ∧ (y = -2)
    ∧ (3 * x + y - 1 = 0) :=
sorry

end tangent_line_equation_parallel_to_given_line_l227_227596


namespace solve_system_of_equations_l227_227004

theorem solve_system_of_equations 
  (a1 a2 a3 a4 : ℝ) (h_distinct : a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a2 ≠ a3 ∧ a2 ≠ a4 ∧ a3 ≠ a4)
  (x1 x2 x3 x4 : ℝ)
  (h1 : |a1 - a1| * x1 + |a1 - a2| * x2 + |a1 - a3| * x3 + |a1 - a4| * x4 = 1)
  (h2 : |a2 - a1| * x1 + |a2 - a2| * x2 + |a2 - a3| * x3 + |a2 - a4| * x4 = 1)
  (h3 : |a3 - a1| * x1 + |a3 - a2| * x2 + |a3 - a3| * x3 + |a3 - a4| * x4 = 1)
  (h4 : |a4 - a1| * x1 + |a4 - a2| * x2 + |a4 - a3| * x3 + |a4 - a4| * x4 = 1) :
  x1 = 1 / (a1 - a4) ∧ x2 = 0 ∧ x3 = 0 ∧ x4 = 1 / (a1 - a4) :=
sorry

end solve_system_of_equations_l227_227004


namespace organizingCommitteeWays_l227_227567

-- Define the problem context
def numberOfTeams : Nat := 5
def membersPerTeam : Nat := 8
def hostTeamSelection : Nat := 4
def otherTeamsSelection : Nat := 2

-- Define binomial coefficient
def binom (n k : Nat) : Nat := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Calculate the number of ways to select committee members
def totalCommitteeWays : Nat := numberOfTeams * 
                                 (binom membersPerTeam hostTeamSelection) * 
                                 ((binom membersPerTeam otherTeamsSelection) ^ (numberOfTeams - 1))

-- The theorem to prove
theorem organizingCommitteeWays : 
  totalCommitteeWays = 215134600 := 
    sorry

end organizingCommitteeWays_l227_227567


namespace largest_sum_of_ABC_l227_227062

-- Define the variables and the conditions
def A := 533
def B := 5
def C := 1

-- Define the product condition
def product_condition : Prop := (A * B * C = 2665)

-- Define the distinct positive integers condition
def distinct_positive_integers_condition : Prop := (A > 0 ∧ B > 0 ∧ C > 0 ∧ A ≠ B ∧ B ≠ C ∧ A ≠ C)

-- State the theorem
theorem largest_sum_of_ABC : product_condition → distinct_positive_integers_condition → A + B + C = 539 := by
  intros _ _
  sorry

end largest_sum_of_ABC_l227_227062


namespace race_duration_l227_227086

theorem race_duration 
  (lap_distance : ℕ) (laps : ℕ)
  (award_per_hundred_meters : ℝ) (earn_rate_per_minute : ℝ)
  (total_distance : ℕ) (total_award : ℝ) (duration : ℝ) :
  lap_distance = 100 →
  laps = 24 →
  award_per_hundred_meters = 3.5 →
  earn_rate_per_minute = 7 →
  total_distance = lap_distance * laps →
  total_award = (total_distance / 100) * award_per_hundred_meters →
  duration = total_award / earn_rate_per_minute →
  duration = 12 := 
by 
  intros;
  sorry

end race_duration_l227_227086


namespace interest_rate_supposed_to_be_invested_l227_227744

variable (P T : ℕ) (additional_interest interest_rate_15 interest_rate_R : ℚ)

def simple_interest (principal: ℚ) (time: ℚ) (rate: ℚ) : ℚ := (principal * time * rate) / 100

theorem interest_rate_supposed_to_be_invested :
  P = 15000 → T = 2 → additional_interest = 900 → interest_rate_15 = 15 →
  simple_interest P T interest_rate_15 = simple_interest P T interest_rate_R + additional_interest →
  interest_rate_R = 12 := by
  intros hP hT h_add h15 h_interest
  simp [simple_interest] at *
  sorry

end interest_rate_supposed_to_be_invested_l227_227744


namespace intersect_condition_l227_227084

theorem intersect_condition (m : ℕ) (h : m ≠ 0) : 
  (∃ x y : ℝ, (3 * x - 2 * y = 0) ∧ ((x - m)^2 + y^2 = 1)) → m = 1 :=
by 
  sorry

end intersect_condition_l227_227084


namespace minimum_value_of_quadratic_expression_l227_227867

def quadratic_expr (x y : ℝ) : ℝ := x^2 - x * y + y^2

def constraint (x y : ℝ) : Prop := x + y = 5

theorem minimum_value_of_quadratic_expression :
  ∃ m, ∀ x y, constraint x y → quadratic_expr x y ≥ m ∧ (∃ x y, constraint x y ∧ quadratic_expr x y = m) :=
sorry

end minimum_value_of_quadratic_expression_l227_227867


namespace berengere_contribution_l227_227750

noncomputable def exchange_rate : ℝ := (1.5 : ℝ)
noncomputable def pastry_cost_euros : ℝ := (8 : ℝ)
noncomputable def lucas_money_cad : ℝ := (10 : ℝ)
noncomputable def lucas_money_euros : ℝ := lucas_money_cad / exchange_rate

theorem berengere_contribution :
  pastry_cost_euros - lucas_money_euros = (4 / 3 : ℝ) :=
by
  sorry

end berengere_contribution_l227_227750


namespace solution_set_l227_227166

noncomputable def satisfies_equations (x y : ℝ) : Prop :=
  (x^2 + 3 * x * y = 12) ∧ (x * y = 16 + y^2 - x * y - x^2)

theorem solution_set :
  {p : ℝ × ℝ | satisfies_equations p.1 p.2} = {(4, 1), (-4, -1), (-4, 1), (4, -1)} :=
by sorry

end solution_set_l227_227166


namespace largest_digit_divisible_by_9_l227_227618

theorem largest_digit_divisible_by_9 : ∀ (B : ℕ), B < 10 → (∃ n : ℕ, 9 * n = 5 + B + 4 + 8 + 6 + 1) → B = 9 := by
  sorry

end largest_digit_divisible_by_9_l227_227618


namespace age_difference_l227_227864

variable (A B C : ℕ)

theorem age_difference (h1 : A + B > B + C) (h2 : C = A - 13) : (A + B) - (B + C) = 13 := by
  sorry

end age_difference_l227_227864


namespace range_of_x_l227_227576

theorem range_of_x (x : ℝ) (h1 : 1 / x < 4) (h2 : 1 / x > -2) : x > -(1 / 2) :=
sorry

end range_of_x_l227_227576


namespace rational_root_k_values_l227_227760

theorem rational_root_k_values (k : ℤ) :
  (∃ x : ℚ, x^2017 - x^2016 + x^2 + k * x + 1 = 0) ↔ (k = 0 ∨ k = -2) :=
by
  sorry

end rational_root_k_values_l227_227760


namespace find_given_number_l227_227162

theorem find_given_number (x : ℕ) : 10 * x + 2 = 3 * (x + 200000) → x = 85714 :=
by
  sorry

end find_given_number_l227_227162


namespace negation_proposition_l227_227908

theorem negation_proposition :
  (¬ (∀ x : ℝ, x ≥ 0)) ↔ (∃ x : ℝ, x < 0) :=
by
  sorry

end negation_proposition_l227_227908


namespace trig_identity_l227_227547

open Real

theorem trig_identity (theta : ℝ) (h : tan theta = 2) : 
  (sin (π / 2 + theta) - cos (π - theta)) / (sin (π / 2 - theta) - sin (π - theta)) = -2 :=
by
  sorry

end trig_identity_l227_227547


namespace isosceles_triangle_perimeter_l227_227669

theorem isosceles_triangle_perimeter :
  ∃ (a b c : ℕ), 
  (a = 3 ∧ b = 6 ∧ (c = 6 ∨ c = 3)) ∧ 
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) ∧
  (a + b + c = 15) :=
sorry

end isosceles_triangle_perimeter_l227_227669


namespace race_winner_l227_227048

-- Definitions and conditions based on the problem statement
def tortoise_speed : ℕ := 5  -- Tortoise speed in meters per minute
def hare_speed_1 : ℕ := 20  -- Hare initial speed in meters per minute
def hare_time_1 : ℕ := 3  -- Hare initial running time in minutes
def hare_speed_2 : ℕ := 10  -- Hare speed when going back in meters per minute
def hare_time_2 : ℕ := 2  -- Hare back running time in minutes
def hare_sleep_time : ℕ := 5  -- Hare sleeping time in minutes
def hare_speed_3 : ℕ := 25  -- Hare final speed in meters per minute
def track_length : ℕ := 130  -- Total length of the race track in meters

-- The problem statement
theorem race_winner :
  track_length / tortoise_speed > hare_time_1 + hare_time_2 + hare_sleep_time + (track_length - (hare_speed_1 * hare_time_1 - hare_speed_2 * hare_time_2)) / hare_speed_3 :=
sorry

end race_winner_l227_227048


namespace solution_set_l227_227158

theorem solution_set (x y : ℝ) : 
  x^5 - 10 * x^3 * y^2 + 5 * x * y^4 = 0 ↔ 
  x = 0 
  ∨ y = x / Real.sqrt (5 + 2 * Real.sqrt 5) 
  ∨ y = x / Real.sqrt (5 - 2 * Real.sqrt 5) 
  ∨ y = -x / Real.sqrt (5 + 2 * Real.sqrt 5) 
  ∨ y = -x / Real.sqrt (5 - 2 * Real.sqrt 5) := 
by
  sorry

end solution_set_l227_227158


namespace problem_inequality_l227_227028

theorem problem_inequality (n a b : ℕ) (h₁ : n ≥ 2) 
  (h₂ : ∀ m, 2^m ∣ 5^n - 3^n → m ≤ a) 
  (h₃ : ∀ m, 2^m ≤ n → m ≤ b) : a ≤ b + 3 :=
sorry

end problem_inequality_l227_227028


namespace estimated_white_balls_l227_227801

noncomputable def estimate_white_balls (total_balls draws white_draws : ℕ) : ℕ :=
  total_balls * white_draws / draws

theorem estimated_white_balls (total_balls draws white_draws : ℕ) (h1 : total_balls = 20)
  (h2 : draws = 100) (h3 : white_draws = 40) :
  estimate_white_balls total_balls draws white_draws = 8 := by
  sorry

end estimated_white_balls_l227_227801


namespace range_of_a_l227_227222

noncomputable def is_even (f : ℝ → ℝ) :=
  ∀ x : ℝ, f x = f (-x)

noncomputable def is_increasing_on_nonneg (f : ℝ → ℝ) :=
  ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y

theorem range_of_a
  {f : ℝ → ℝ}
  (hf_even : is_even f)
  (hf_increasing : is_increasing_on_nonneg f)
  (hf_inequality : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f (a * x + 1) ≤ f (x - 3)) :
  -1 ≤ a ∧ a ≤ 0 :=
sorry

end range_of_a_l227_227222


namespace mean_variance_transformation_l227_227019

variable (n : ℕ)
variable (x : Fin n → ℝ)
variable (mean_original variance_original : ℝ)
variable (meam_new variance_new : ℝ)
variable (offset : ℝ)

theorem mean_variance_transformation (hmean : mean_original = 2.8) (hvariance : variance_original = 3.6) 
  (hoffset : offset = 60) : 
  (mean_new = mean_original + offset) ∧ (variance_new = variance_original) :=
  sorry

end mean_variance_transformation_l227_227019


namespace odd_three_mn_l227_227415

theorem odd_three_mn (m n : ℕ) (hm : m % 2 = 1) (hn : n % 2 = 1) : (3 * m * n) % 2 = 1 :=
sorry

end odd_three_mn_l227_227415


namespace closest_perfect_square_to_314_l227_227055

theorem closest_perfect_square_to_314 :
  ∃ n : ℤ, n^2 = 324 ∧ ∀ m : ℤ, m^2 ≠ 324 → |m^2 - 314| > |324 - 314| :=
by
  sorry

end closest_perfect_square_to_314_l227_227055


namespace ratio_of_ages_l227_227302

theorem ratio_of_ages (a b : ℕ) (h1 : a - 10 = (b - 10) / 2) (h2 : a + b = 35) : a / gcd a b = 3 ∧ b / gcd a b = 4 :=
by
  sorry

end ratio_of_ages_l227_227302


namespace assisted_work_time_l227_227485

theorem assisted_work_time (a b c : ℝ) (ha : a = 1 / 11) (hb : b = 1 / 20) (hc : c = 1 / 55) :
  (1 / ((a + b) + (a + c) / 2)) = 8 :=
by
  sorry

end assisted_work_time_l227_227485


namespace nicky_cristina_race_l227_227809

theorem nicky_cristina_race :
  ∀ (head_start t : ℕ), ∀ (cristina_speed nicky_speed time_nicky_run : ℝ),
  head_start = 12 →
  cristina_speed = 5 →
  nicky_speed = 3 →
  ((cristina_speed * t) = (nicky_speed * t + nicky_speed * head_start)) →
  time_nicky_run = head_start + t →
  time_nicky_run = 30 :=
by
  intros
  sorry

end nicky_cristina_race_l227_227809


namespace student_A_recruit_as_pilot_exactly_one_student_pass_l227_227168

noncomputable def student_A_recruit_prob : ℝ :=
  1 * 0.5 * 0.6 * 1

theorem student_A_recruit_as_pilot :
  student_A_recruit_prob = 0.3 :=
by
  sorry

noncomputable def one_student_pass_reinspection : ℝ :=
  0.5 * (1 - 0.6) * (1 - 0.75) +
  (1 - 0.5) * 0.6 * (1 - 0.75) +
  (1 - 0.5) * (1 - 0.6) * 0.75

theorem exactly_one_student_pass :
  one_student_pass_reinspection = 0.275 :=
by
  sorry

end student_A_recruit_as_pilot_exactly_one_student_pass_l227_227168


namespace nancy_pictures_l227_227200

theorem nancy_pictures (z m b d : ℕ) (hz : z = 120) (hm : m = 75) (hb : b = 45) (hd : d = 93) :
  (z + m + b) - d = 147 :=
by {
  -- Theorem definition capturing the problem statement
  sorry
}

end nancy_pictures_l227_227200


namespace quotient_of_division_l227_227426

theorem quotient_of_division (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) 
  (h1 : dividend = 181) (h2 : divisor = 20) (h3 : remainder = 1) 
  (h4 : dividend = (divisor * quotient) + remainder) : quotient = 9 :=
by
  sorry -- proof goes here

end quotient_of_division_l227_227426


namespace expected_number_of_hits_l227_227783

variable (W : ℝ) (n : ℕ)
def expected_hits (W : ℝ) (n : ℕ) : ℝ := W * n

theorem expected_number_of_hits :
  W = 0.75 → n = 40 → expected_hits W n = 30 :=
by
  intros hW hn
  rw [hW, hn]
  norm_num
  sorry

end expected_number_of_hits_l227_227783


namespace max_robot_weight_l227_227509

-- Definitions of the given conditions
def standard_robot_weight : ℕ := 100
def battery_weight : ℕ := 20
def min_payload : ℕ := 10
def max_payload : ℕ := 25
def min_robot_weight_extra : ℕ := 5
def min_robot_weight : ℕ := standard_robot_weight + min_robot_weight_extra

-- Definition for total minimum weight of the robot
def min_total_weight : ℕ := min_robot_weight + battery_weight + min_payload

-- Proposition for the maximum weight condition
theorem max_robot_weight :
  2 * min_total_weight = 270 :=
by
  -- Insert proof here
  sorry

end max_robot_weight_l227_227509


namespace pieces_per_package_l227_227787

-- Definitions from conditions
def total_pieces_of_gum : ℕ := 486
def number_of_packages : ℕ := 27

-- Mathematical statement to prove
theorem pieces_per_package : total_pieces_of_gum / number_of_packages = 18 := sorry

end pieces_per_package_l227_227787


namespace mr_bird_speed_to_work_l227_227122

theorem mr_bird_speed_to_work (
  d t : ℝ
) (h1 : d = 45 * (t + 4 / 60)) 
  (h2 : d = 55 * (t - 2 / 60))
  (h3 : t = 29 / 60)
  (d_eq : d = 24.75) :
  (24.75 / (29 / 60)) = 51.207 := 
sorry

end mr_bird_speed_to_work_l227_227122


namespace total_oranges_picked_l227_227763

/-- Michaela needs 20 oranges to get full --/
def oranges_michaela_needs : ℕ := 20

/-- Cassandra needs twice as many oranges as Michaela to get full --/
def oranges_cassandra_needs : ℕ := 2 * oranges_michaela_needs

/-- After both have eaten until they are full, 30 oranges remain --/
def oranges_remaining : ℕ := 30

/-- The total number of oranges eaten by both Michaela and Cassandra --/
def oranges_eaten : ℕ := oranges_michaela_needs + oranges_cassandra_needs

/-- Prove that the total number of oranges picked from the farm is 90 --/
theorem total_oranges_picked : oranges_eaten + oranges_remaining = 90 := by
  sorry

end total_oranges_picked_l227_227763


namespace option_C_incorrect_l227_227656

def p (x y : ℝ) : ℝ := x^3 - 3 * x^2 * y + 3 * x * y^2 - y^3

theorem option_C_incorrect (x y : ℝ) : 
  ((x^3 - 3 * x^2 * y) - (3 * x * y^2 + y^3)) ≠ p x y := by
  sorry

end option_C_incorrect_l227_227656


namespace greatest_remainder_le_11_l227_227934

noncomputable def greatest_remainder (x : ℕ) : ℕ := x % 12

theorem greatest_remainder_le_11 (x : ℕ) (h : x % 12 ≠ 0) : greatest_remainder x = 11 :=
by
  sorry

end greatest_remainder_le_11_l227_227934


namespace olivia_wallet_final_amount_l227_227678

variable (initial_money : ℕ) (money_added : ℕ) (money_spent : ℕ)

theorem olivia_wallet_final_amount
  (h1 : initial_money = 100)
  (h2 : money_added = 148)
  (h3 : money_spent = 89) :
  initial_money + money_added - money_spent = 159 :=
  by 
    sorry

end olivia_wallet_final_amount_l227_227678


namespace quinn_free_donuts_l227_227045

-- Definitions based on conditions
def books_per_week : ℕ := 2
def weeks : ℕ := 10
def books_needed_for_donut : ℕ := 5

-- Calculation based on conditions
def total_books_read : ℕ := books_per_week * weeks
def free_donuts (total_books : ℕ) : ℕ := total_books / books_needed_for_donut

-- Proof statement
theorem quinn_free_donuts : free_donuts total_books_read = 4 := by
  sorry

end quinn_free_donuts_l227_227045


namespace melanie_total_plums_l227_227563

-- Define the initial conditions
def melaniePlums : Float := 7.0
def samGavePlums : Float := 3.0

-- State the theorem to prove
theorem melanie_total_plums : melaniePlums + samGavePlums = 10.0 := 
by
  sorry

end melanie_total_plums_l227_227563


namespace sqrt_sum_ge_two_l227_227645

theorem sqrt_sum_ge_two (a b c : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : a * b + b * c + c * a + 2 * a * b * c = 1) : 
  Real.sqrt a + Real.sqrt b + Real.sqrt c ≥ 2 := 
by
  sorry

end sqrt_sum_ge_two_l227_227645


namespace rowing_distance_l227_227691

theorem rowing_distance
  (v_still : ℝ) (v_current : ℝ) (time : ℝ)
  (h1 : v_still = 15) (h2 : v_current = 3) (h3 : time = 17.998560115190784) :
  (v_still + v_current) * 1000 / 3600 * time = 89.99280057595392 :=
by
  rw [h1, h2, h3] -- Apply the given conditions
  -- This will reduce to proving (15 + 3) * 1000 / 3600 * 17.998560115190784 = 89.99280057595392
  sorry

end rowing_distance_l227_227691


namespace min_abs_val_sum_l227_227888

theorem min_abs_val_sum : ∃ x : ℝ, (∀ y : ℝ, |y - 1| + |y - 2| + |y - 3| ≥ |x - 1| + |x - 2| + |x - 3|) ∧ |x - 1| + |x - 2| + |x - 3| = 1 :=
sorry

end min_abs_val_sum_l227_227888


namespace ellipse_equation_l227_227580

theorem ellipse_equation (b : Real) (c : Real)
  (h₁ : 0 < b ∧ b < 5) 
  (h₂ : 25 - b^2 = c^2)
  (h₃ : 5 + c = 2 * b) :
  ∃ (b : Real), (b^2 = 16) ∧ (∀ x y : Real, (x^2 / 25 + y^2 / b^2 = 1 ↔ x^2 / 25 + y^2 / 16 = 1)) := 
sorry

end ellipse_equation_l227_227580


namespace evening_to_morning_ratio_l227_227383

-- Definitions based on conditions
def morning_miles : ℕ := 2
def total_miles : ℕ := 12
def evening_miles : ℕ := total_miles - morning_miles

-- Lean statement to prove the ratio
theorem evening_to_morning_ratio : evening_miles / morning_miles = 5 := by
  -- we simply state the final ratio we want to prove
  sorry

end evening_to_morning_ratio_l227_227383


namespace find_some_number_l227_227798

theorem find_some_number (a : ℕ) (some_number : ℕ)
  (h1 : a = 105)
  (h2 : a ^ 3 = 21 * 35 * some_number * 35) :
  some_number = 21 :=
by
  sorry

end find_some_number_l227_227798


namespace count_five_digit_multiples_of_five_l227_227623

theorem count_five_digit_multiples_of_five : 
  ∃ (n : ℕ), n = 18000 ∧ (∀ x, 10000 ≤ x ∧ x ≤ 99999 ∧ x % 5 = 0 ↔ ∃ k, 10000 ≤ 5 * k ∧ 5 * k ≤ 99999) :=
by
  sorry

end count_five_digit_multiples_of_five_l227_227623


namespace solution_set_of_inequality_l227_227779

theorem solution_set_of_inequality
  (a b : ℝ)
  (h1 : a < 0) 
  (h2 : b / a = 1) :
  { x : ℝ | (x - 1) * (a * x + b) < 0 } = { x : ℝ | x < -1 } ∪ {x : ℝ | 1 < x} :=
by
  sorry

end solution_set_of_inequality_l227_227779


namespace beverage_distribution_l227_227872

theorem beverage_distribution (total_cans : ℕ) (number_of_children : ℕ) (hcans : total_cans = 5) (hchildren : number_of_children = 8) :
  (total_cans / number_of_children : ℚ) = 5 / 8 :=
by
  -- Given the conditions
  have htotal_cans : total_cans = 5 := hcans
  have hnumber_of_children : number_of_children = 8 := hchildren
  
  -- we need to show the beverage distribution
  rw [htotal_cans, hnumber_of_children]
  exact by norm_num

end beverage_distribution_l227_227872


namespace infinite_danish_numbers_l227_227465

-- Definitions translated from problem conditions
def is_danish (n : ℕ) : Prop :=
  ∃ k, n = 3 * k ∨ n = 2 * 4 ^ k

theorem infinite_danish_numbers :
  ∃ S : Set ℕ, Set.Infinite S ∧ ∀ n ∈ S, is_danish n ∧ is_danish (2^n + n) := sorry

end infinite_danish_numbers_l227_227465


namespace arc_length_problem_l227_227942

noncomputable def arc_length (r : ℝ) (theta : ℝ) : ℝ :=
  r * theta

theorem arc_length_problem :
  ∀ (r : ℝ) (theta_deg : ℝ), r = 1 ∧ theta_deg = 150 → 
  arc_length r (theta_deg * (Real.pi / 180)) = (5 * Real.pi / 6) :=
by
  intro r theta_deg h
  sorry

end arc_length_problem_l227_227942


namespace problem1_l227_227228

theorem problem1 (x y : ℝ) 
  (h1 : 3 * x + 2 * y = 10)
  (h2 : x / 2 - (y + 1) / 3 = 1) :
  x = 3 ∧ y = 1 / 2 := 
sorry

end problem1_l227_227228


namespace volume_of_box_l227_227182

noncomputable def volume_expression (y : ℝ) : ℝ :=
  (15 - 2 * y) * (12 - 2 * y) * y

theorem volume_of_box (y : ℝ) :
  volume_expression y = 4 * y^3 - 54 * y^2 + 180 * y :=
by
  sorry

end volume_of_box_l227_227182


namespace instantaneous_rate_of_change_at_0_l227_227568

noncomputable def f (x : ℝ) : ℝ := 2 * Real.exp (Real.sin x)

theorem instantaneous_rate_of_change_at_0 : (deriv f 0) = 2 :=
  by
  sorry

end instantaneous_rate_of_change_at_0_l227_227568


namespace sufficient_but_not_necessary_l227_227355

theorem sufficient_but_not_necessary (a : ℝ) :
  0 < a ∧ a < 1 → (∀ x : ℝ, ax^2 + 2 * a * x + 1 > 0) ∧ ¬ (∀ a, (∀ x : ℝ, ax^2 + 2 * a * x + 1 > 0) → 0 < a ∧ a < 1) :=
by
  sorry

end sufficient_but_not_necessary_l227_227355


namespace linear_equation_variables_l227_227073

theorem linear_equation_variables (m n : ℤ) (h1 : 3 * m - 2 * n = 1) (h2 : n - m = 1) : m = 0 ∧ n = 1 :=
by {
  sorry
}

end linear_equation_variables_l227_227073


namespace pyramid_volume_l227_227161

noncomputable def volume_of_pyramid 
  (ABCD : Type) 
  (rectangle : ABCD) 
  (DM_perpendicular : Prop) 
  (MA MC MB : ℕ) 
  (lengths : MA = 11 ∧ MC = 13 ∧ MB = 15 ∧ DM = 5) : ℝ :=
  80 * Real.sqrt 6

theorem pyramid_volume (ABCD : Type) 
    (rectangle : ABCD) 
    (DM_perpendicular : Prop) 
    (MA MC MB DM : ℕ)
    (lengths : MA = 11 ∧ MC = 13 ∧ MB = 15 ∧ DM = 5) 
  : volume_of_pyramid ABCD rectangle DM_perpendicular MA MC MB lengths = 80 * Real.sqrt 6 :=
  by {
    sorry
  }

end pyramid_volume_l227_227161


namespace part1_part2_part3_l227_227790

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - x + 1

theorem part1 (a : ℝ) (x : ℝ) (h : 0 < x) :
  (a ≤ 0 → (∀ x > 0, f a x < 0)) ∧
  (a > 0 → (∀ x ∈ Set.Ioo 0 a, f a x > 0) ∧ (∀ x ∈ Set.Ioi a, f a x < 0)) :=
sorry

theorem part2 {a : ℝ} : (∀ x > 0, f a x ≤ 0) → a = 1 :=
sorry

theorem part3 (n : ℕ) (h : 0 < n) :
  (1 + 1 / n : ℝ)^n < Real.exp 1 ∧ Real.exp 1 < (1 + 1 / n : ℝ)^(n + 1) :=
sorry

end part1_part2_part3_l227_227790


namespace ticket_cost_l227_227899

noncomputable def calculate_cost (x : ℝ) : ℝ :=
  6 * (1.1 * x) + 5 * (x / 2)

theorem ticket_cost (x : ℝ) (h : 4 * (1.1 * x) + 3 * (x / 2) = 28.80) : 
  calculate_cost x = 44.41 := by
  sorry

end ticket_cost_l227_227899


namespace base6_number_divisibility_l227_227204

/-- 
Given that 45x2 in base 6 converted to its decimal equivalent is 6x + 1046,
and it is divisible by 19. Prove that x = 5 given that x is a base-6 digit.
-/
theorem base6_number_divisibility (x : ℕ) (h1 : 0 ≤ x ∧ x ≤ 5) (h2 : (6 * x + 1046) % 19 = 0) : x = 5 :=
sorry

end base6_number_divisibility_l227_227204


namespace base12_mod_9_remainder_l227_227461

noncomputable def base12_to_base10 (n : ℕ) : ℕ :=
  1 * 12^3 + 7 * 12^2 + 3 * 12^1 + 2 * 12^0

theorem base12_mod_9_remainder : (base12_to_base10 1732) % 9 = 2 := by
  sorry

end base12_mod_9_remainder_l227_227461


namespace toy_discount_price_l227_227296

theorem toy_discount_price (original_price : ℝ) (discount_rate : ℝ) (price_after_first_discount : ℝ) (price_after_second_discount : ℝ) : 
  original_price = 200 → 
  discount_rate = 0.1 →
  price_after_first_discount = original_price * (1 - discount_rate) →
  price_after_second_discount = price_after_first_discount * (1 - discount_rate) →
  price_after_second_discount = 162 :=
by
  intros h1 h2 h3 h4
  sorry

end toy_discount_price_l227_227296


namespace ice_forms_inner_surface_in_winter_l227_227588

-- Definitions based on conditions
variable (humid_air_inside : Prop) 
variable (heat_transfer_inner_surface : Prop) 
variable (heat_transfer_outer_surface : Prop) 
variable (temp_inner_surface_below_freezing : Prop) 
variable (condensation_inner_surface_below_freezing : Prop)
variable (ice_formation_inner_surface : Prop)
variable (cold_dry_air_outside : Prop)
variable (no_significant_condensation_outside : Prop)

-- Proof of the theorem
theorem ice_forms_inner_surface_in_winter :
  humid_air_inside ∧
  heat_transfer_inner_surface ∧
  heat_transfer_outer_surface ∧
  (¬sufficient_heating → temp_inner_surface_below_freezing) ∧
  (condensation_inner_surface_below_freezing ↔ (temp_inner_surface_below_freezing ∧ humid_air_inside)) ∧
  (ice_formation_inner_surface ↔ (condensation_inner_surface_below_freezing ∧ temp_inner_surface_below_freezing)) ∧
  (cold_dry_air_outside → ¬ice_formation_outer_surface)
  → ice_formation_inner_surface :=
sorry

end ice_forms_inner_surface_in_winter_l227_227588


namespace fraction_equals_one_l227_227855

/-- Given the fraction (12-11+10-9+8-7+6-5+4-3+2-1) / (1-2+3-4+5-6+7-8+9-10+11),
    prove that its value is equal to 1. -/
theorem fraction_equals_one :
  (12 - 11 + 10 - 9 + 8 - 7 + 6 - 5 + 4 - 3 + 2 - 1) /
  (1 - 2 + 3 - 4 + 5 - 6 + 7 - 8 + 9 - 10 + 11) = 1 := by
  sorry

end fraction_equals_one_l227_227855


namespace simplify_expression_l227_227357

theorem simplify_expression : 
  (3.875 * (1 / 5) + (38 + 3 / 4) * 0.09 - 0.155 / 0.4) / 
  (2 + 1 / 6 + (((4.32 - 1.68 - (1 + 8 / 25)) * (5 / 11) - 2 / 7) / (1 + 9 / 35)) + (1 + 11 / 24))
  = 1 := sorry

end simplify_expression_l227_227357


namespace log_cut_problem_l227_227982

theorem log_cut_problem (x y : ℕ) (h1 : x + y = 30) (h2 : 3 * x + 4 * y = 100) :
  2 * x + 3 * y = 70 := by
  sorry

end log_cut_problem_l227_227982


namespace num_comics_bought_l227_227256

def initial_comic_books : ℕ := 14
def current_comic_books : ℕ := 13
def comic_books_sold (initial : ℕ) : ℕ := initial / 2
def comics_bought (initial current : ℕ) : ℕ :=
  current - (initial - comic_books_sold initial)

theorem num_comics_bought :
  comics_bought initial_comic_books current_comic_books = 6 :=
by
  sorry

end num_comics_bought_l227_227256


namespace percentage_divisible_by_7_l227_227610

-- Define the total integers and the condition for being divisible by 7
def total_ints := 140
def divisible_by_7 (n : ℕ) : Prop := n % 7 = 0

-- Calculate the number of integers between 1 and 140 that are divisible by 7
def count_divisible_by_7 : ℕ := Nat.succ (140 / 7)

-- The theorem to prove
theorem percentage_divisible_by_7 : (count_divisible_by_7 / total_ints : ℚ) * 100 = 14.29 := by
  sorry

end percentage_divisible_by_7_l227_227610


namespace james_nickels_count_l227_227552

-- Definitions
def total_cents : ℕ := 685
def more_nickels_than_quarters := 11

-- Variables representing the number of nickels and quarters
variables (n q : ℕ)

-- Conditions
axiom h1 : 5 * n + 25 * q = total_cents
axiom h2 : n = q + more_nickels_than_quarters

-- Theorem stating the number of nickels
theorem james_nickels_count : n = 32 := 
by
  -- Proof will go here, marked as "sorry" to complete the statement
  sorry

end james_nickels_count_l227_227552


namespace probability_of_selection_is_equal_l227_227251

-- Define the conditions of the problem
def total_students := 2004
def eliminated_students := 4
def remaining_students := total_students - eliminated_students -- 2000
def selected_students := 50
def k := remaining_students / selected_students -- 40

-- Define the probability calculation
def probability_selected := selected_students / remaining_students

-- The theorem stating that every student has a 1/40 probability of being selected
theorem probability_of_selection_is_equal :
  probability_selected = 1 / 40 :=
by
  -- insert proof logic here
  sorry

end probability_of_selection_is_equal_l227_227251


namespace martha_knits_hat_in_2_hours_l227_227986

-- Definitions based on given conditions
variables (H : ℝ)
def knit_times (H : ℝ) : ℝ := H + 3 + 2 + 3 + 6

def total_knitting_time (H : ℝ) : ℝ := 3 * knit_times H

-- The main statement to be proven
theorem martha_knits_hat_in_2_hours (H : ℝ) (h : total_knitting_time H = 48) : H = 2 := 
by
  sorry

end martha_knits_hat_in_2_hours_l227_227986


namespace expected_expenditure_l227_227047

-- Define the parameters and conditions
def b : ℝ := 0.8
def a : ℝ := 2
def e_condition (e : ℝ) : Prop := |e| < 0.5
def revenue : ℝ := 10

-- Define the expenditure function based on the conditions
def expenditure (x e : ℝ) : ℝ := b * x + a + e

-- The expected expenditure should not exceed 10.5
theorem expected_expenditure (e : ℝ) (h : e_condition e) : expenditure revenue e ≤ 10.5 :=
sorry

end expected_expenditure_l227_227047


namespace tensor_example_l227_227354
-- Import the necessary library

-- Define the binary operation ⊗
def tensor (a b : ℚ) : ℚ := (a + b) / (a - b)

-- State the main theorem
theorem tensor_example : tensor (tensor 8 6) 2 = 9 / 5 := by
  sorry

end tensor_example_l227_227354


namespace greatest_possible_sum_of_two_consecutive_integers_product_lt_1000_l227_227128

theorem greatest_possible_sum_of_two_consecutive_integers_product_lt_1000 : 
  ∃ n : ℤ, (n * (n + 1) < 1000) ∧ (n + (n + 1) = 63) :=
sorry

end greatest_possible_sum_of_two_consecutive_integers_product_lt_1000_l227_227128


namespace Isabella_exchange_l227_227260

/-
Conditions:
1. Isabella exchanged d U.S. dollars to receive (8/5)d Canadian dollars.
2. After spending 80 Canadian dollars, she had d + 20 Canadian dollars left.
3. Sum of the digits of d is 14.
-/

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.foldr (.+.) 0

theorem Isabella_exchange (d : ℕ) (h : (8 * d / 5) - 80 = d + 20) : sum_of_digits d = 14 :=
by sorry

end Isabella_exchange_l227_227260


namespace max_lcm_15_2_3_5_6_9_10_l227_227890

theorem max_lcm_15_2_3_5_6_9_10 : 
  max (max (max (max (max (Nat.lcm 15 2) (Nat.lcm 15 3)) (Nat.lcm 15 5)) (Nat.lcm 15 6)) (Nat.lcm 15 9)) (Nat.lcm 15 10) = 45 :=
by
  sorry

end max_lcm_15_2_3_5_6_9_10_l227_227890


namespace problem_statement_l227_227334

noncomputable def proposition_p (x : ℝ) : Prop := ∃ x0 : ℝ, x0 - 2 > 0
noncomputable def proposition_q (x : ℝ) : Prop := ∀ x : ℝ, (2:ℝ)^x > x^2

theorem problem_statement : ∃ (p q : Prop), (∃ x0 : ℝ, x0 - 2 > 0) ∧ (¬ (∀ x : ℝ, (2:ℝ)^x > x^2)) :=
by
  sorry

end problem_statement_l227_227334


namespace possible_value_of_b_l227_227442

-- Definition of the linear function
def linear_function (x : ℝ) (b : ℝ) : ℝ := -2 * x + b

-- Condition for the linear function to pass through the second, third, and fourth quadrants
def passes_second_third_fourth_quadrants (b : ℝ) : Prop :=
  b < 0

-- Lean 4 statement expressing the problem
theorem possible_value_of_b (b : ℝ) (h : passes_second_third_fourth_quadrants b) : b = -1 :=
  sorry

end possible_value_of_b_l227_227442


namespace min_value_exp_l227_227854

theorem min_value_exp (a b : ℝ) (h_condition : a - 3 * b + 6 = 0) : 
  ∃ (m : ℝ), m = 2^a + 1 / 8^b ∧ m ≥ (1 / 4) :=
by
  sorry

end min_value_exp_l227_227854


namespace no_square_cube_l227_227376

theorem no_square_cube (n : ℕ) (h : n > 0) : ¬ (∃ k : ℕ, k^2 = n * (n + 1) * (n + 2) * (n + 3)) ∧ ¬ (∃ l : ℕ, l^3 = n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end no_square_cube_l227_227376


namespace incorrect_statement_among_props_l227_227219

theorem incorrect_statement_among_props 
    (A: Prop := True)  -- Axioms in mathematics are accepted truths that do not require proof.
    (B: Prop := True)  -- A mathematical proof can proceed in different valid sequences depending on the approach and insights.
    (C: Prop := True)  -- All concepts utilized in a proof must be clearly defined before their use in arguments.
    (D: Prop := False) -- Logical deductions based on false premises can lead to valid conclusions.
    (E: Prop := True): -- Proof by contradiction only needs one assumption to be negated and shown to lead to a contradiction to be valid.
  ¬D := 
by sorry

end incorrect_statement_among_props_l227_227219


namespace integral_eq_exp_integral_eq_one_l227_227662

noncomputable
def y1 (τ : ℝ) (t : ℝ) (y : ℝ → ℝ) : Prop :=
  y τ = ∫ x in (0 : ℝ)..t, y x + 1

theorem integral_eq_exp (y : ℝ → ℝ) : 
  (∀ τ t, y1 τ t y) ↔ (∀ t, y t = Real.exp t) := 
  sorry

noncomputable
def y2 (t : ℝ) (y : ℝ → ℝ) : Prop :=
  ∫ x in (0 : ℝ)..t, y x * Real.sin (t - x) = 1 - Real.cos t

theorem integral_eq_one (y : ℝ → ℝ) : 
  (∀ t, y2 t y) ↔ (∀ t, y t = 1) :=
  sorry

end integral_eq_exp_integral_eq_one_l227_227662


namespace perfect_square_l227_227474

variables {n x k ℓ : ℕ}

theorem perfect_square (h1 : x^2 < n) (h2 : n < (x + 1)^2)
  (h3 : k = n - x^2) (h4 : ℓ = (x + 1)^2 - n) :
  ∃ m : ℕ, n - k * ℓ = m^2 :=
by
  sorry

end perfect_square_l227_227474


namespace umar_age_is_10_l227_227943

-- Define Ali's age
def Ali_age := 8

-- Define the age difference between Ali and Yusaf
def age_difference := 3

-- Define Yusaf's age based on the conditions
def Yusaf_age := Ali_age - age_difference

-- Define Umar's age which is twice Yusaf's age
def Umar_age := 2 * Yusaf_age

-- Prove that Umar's age is 10
theorem umar_age_is_10 : Umar_age = 10 :=
by
  -- Proof is skipped
  sorry

end umar_age_is_10_l227_227943


namespace domain_range_equal_l227_227869

noncomputable def f (a b x : ℝ) : ℝ := Real.sqrt (a * x^2 + b * x)

theorem domain_range_equal {a b : ℝ} (hb : b > 0) :
  (∀ y, ∃ x, f a b x = y) ↔ (a = -4 ∨ a = 0) :=
sorry

end domain_range_equal_l227_227869


namespace pythagorean_theorem_mod_3_l227_227591

theorem pythagorean_theorem_mod_3 {x y z : ℕ} (h : x^2 + y^2 = z^2) : x % 3 = 0 ∨ y % 3 = 0 ∨ z % 3 = 0 :=
by 
  sorry

end pythagorean_theorem_mod_3_l227_227591


namespace outfits_count_l227_227680

theorem outfits_count (shirts ties pants belts : ℕ) (h_shirts : shirts = 7) (h_ties : ties = 5) (h_pants : pants = 4) (h_belts : belts = 2) : 
  (shirts * pants * (ties + 1) * (belts + 1 + 1) = 504) :=
by
  rw [h_shirts, h_ties, h_pants, h_belts]
  sorry

end outfits_count_l227_227680


namespace trapezoid_median_l227_227663

theorem trapezoid_median 
  (h : ℝ)
  (triangle_base : ℝ := 24)
  (trapezoid_base1 : ℝ := 15)
  (trapezoid_base2 : ℝ := 33)
  (triangle_area_eq_trapezoid_area : (1 / 2) * triangle_base * h = ((trapezoid_base1 + trapezoid_base2) / 2) * h)
  : (trapezoid_base1 + trapezoid_base2) / 2 = 24 :=
by
  sorry

end trapezoid_median_l227_227663


namespace no_rational_solutions_l227_227997

theorem no_rational_solutions (a b c d : ℚ) (n : ℕ) :
  ¬ ((a + b * (Real.sqrt 2))^(2 * n) + (c + d * (Real.sqrt 2))^(2 * n) = 5 + 4 * (Real.sqrt 2)) :=
sorry

end no_rational_solutions_l227_227997


namespace range_of_a_l227_227147

-- Definitions for propositions
def p (a : ℝ) : Prop :=
  (1 - 4 * (a^2 - 6 * a) > 0) ∧ (a^2 - 6 * a < 0)

def q (a : ℝ) : Prop :=
  (a - 3)^2 - 4 ≥ 0

-- Proof statement
theorem range_of_a (a : ℝ) : (p a ∨ q a) ∧ ¬(p a ∧ q a) ↔ (a ≤ 0 ∨ 1 < a ∧ a < 5 ∨ a ≥ 6) :=
by 
  sorry

end range_of_a_l227_227147


namespace factory_workers_l227_227516

-- Define parameters based on given conditions
def sewing_factory_x : ℤ := 1995
def shoe_factory_y : ℤ := 1575

-- Conditions based on the problem setup
def shoe_factory_of_sewing_factory := (15 * sewing_factory_x) / 19 = shoe_factory_y
def shoe_factory_plan_exceed := (3 * shoe_factory_y) / 7 < 1000
def sewing_factory_plan_exceed := (3 * sewing_factory_x) / 5 > 1000

-- Theorem stating the problem's assertion
theorem factory_workers (x y : ℤ) 
  (h1 : (15 * x) / 19 = y)
  (h2 : (4 * y) / 7 < 1000)
  (h3 : (3 * x) / 5 > 1000) : 
  x = 1995 ∧ y = 1575 :=
sorry

end factory_workers_l227_227516


namespace minimum_n_for_i_pow_n_eq_neg_i_l227_227070

open Complex

theorem minimum_n_for_i_pow_n_eq_neg_i : ∃ (n : ℕ), 0 < n ∧ (i^n = -i) ∧ ∀ (m : ℕ), 0 < m ∧ (i^m = -i) → n ≤ m :=
by
  sorry

end minimum_n_for_i_pow_n_eq_neg_i_l227_227070


namespace scientific_notation_3050000_l227_227440

def scientific_notation (n : ℕ) : String :=
  "3.05 × 10^6"

theorem scientific_notation_3050000 :
  scientific_notation 3050000 = "3.05 × 10^6" :=
by
  sorry

end scientific_notation_3050000_l227_227440


namespace geometric_sequence_properties_l227_227034

-- Define the first term and common ratio
def first_term : ℕ := 12
def common_ratio : ℚ := 1/2

-- Define the formula for the n-th term of the geometric sequence
def nth_term (a : ℕ) (r : ℚ) (n : ℕ) := a * r^(n-1)

-- The 8th term in the sequence
def term_8 := nth_term first_term common_ratio 8

-- Half of the 8th term
def half_term_8 := (1/2) * term_8

-- Prove that the 8th term is 3/32 and half of the 8th term is 3/64
theorem geometric_sequence_properties : 
  (term_8 = (3/32)) ∧ (half_term_8 = (3/64)) := 
by 
  sorry

end geometric_sequence_properties_l227_227034


namespace f_zero_f_pos_f_decreasing_solve_inequality_l227_227449

open Real

noncomputable def f : ℝ → ℝ := sorry

axiom f_mul_add (m n : ℝ) : f m * f n = f (m + n)
axiom f_pos_neg (x : ℝ) : x < 0 → 1 < f x

theorem f_zero : f 0 = 1 :=
sorry

theorem f_pos (x : ℝ) : 0 < x → 0 < f x ∧ f x < 1 :=
sorry

theorem f_decreasing (x₁ x₂ : ℝ) : x₁ < x₂ → f x₁ > f x₂ :=
sorry

theorem solve_inequality (a x : ℝ) :
  f (x^2 - 3 * a * x + 1) * f (-3 * x + 6 * a + 1) ≥ 1 ↔
  (a > 1/3 ∧ 2 ≤ x ∧ x ≤ 3 * a + 1) ∨
  (a = 1/3 ∧ x = 2) ∨
  (a < 1/3 ∧ 3 * a + 1 ≤ x ∧ x ≤ 2) :=
sorry

end f_zero_f_pos_f_decreasing_solve_inequality_l227_227449


namespace find_f_minus_3_l227_227380

def rational_function (f : ℚ → ℚ) : Prop :=
  ∀ x : ℚ, x ≠ 0 → 4 * f (1 / x) + (3 * f x / x) = 2 * x^2

theorem find_f_minus_3 (f : ℚ → ℚ) (h : rational_function f) : 
  f (-3) = 494 / 117 :=
by
  sorry

end find_f_minus_3_l227_227380


namespace p_necessary_not_sufficient_for_p_and_q_l227_227815

-- Define statements p and q as propositions
variables (p q : Prop)

-- Prove that "p is true" is a necessary but not sufficient condition for "p ∧ q is true"
theorem p_necessary_not_sufficient_for_p_and_q : (p ∧ q → p) ∧ (p → ¬ (p ∧ q)) :=
by sorry

end p_necessary_not_sufficient_for_p_and_q_l227_227815


namespace cannot_fit_rectangle_l227_227830

theorem cannot_fit_rectangle 
  (w1 h1 : ℕ) (w2 h2 : ℕ) 
  (h1_pos : 0 < h1) (w1_pos : 0 < w1)
  (h2_pos : 0 < h2) (w2_pos : 0 < w2) :
  w1 = 5 → h1 = 6 → w2 = 3 → h2 = 8 →
  ¬(w2 ≤ w1 ∧ h2 ≤ h1) :=
by
  intros H1 W1 H2 W2
  sorry

end cannot_fit_rectangle_l227_227830


namespace smallest_number_of_coins_l227_227762

theorem smallest_number_of_coins : ∃ (n : ℕ), 
  n ≡ 2 [MOD 5] ∧ 
  n ≡ 1 [MOD 4] ∧ 
  n ≡ 0 [MOD 3] ∧ 
  n = 57 := 
by
  sorry

end smallest_number_of_coins_l227_227762


namespace alligators_hiding_correct_l227_227554

def total_alligators := 75
def not_hiding_alligators := 56

def hiding_alligators (total not_hiding : Nat) : Nat :=
  total - not_hiding

theorem alligators_hiding_correct : hiding_alligators total_alligators not_hiding_alligators = 19 := 
by
  sorry

end alligators_hiding_correct_l227_227554


namespace complement_intersection_l227_227167

def U : Set ℤ := {-1, 0, 1, 2}
def A : Set ℤ := {1, 2}
def B : Set ℤ := {0, 2}

theorem complement_intersection :
  (U \ A) ∩ B = {0} :=
  by
    sorry

end complement_intersection_l227_227167


namespace mileage_per_gallon_l227_227791

noncomputable def car_mileage (distance: ℝ) (gasoline: ℝ) : ℝ :=
  distance / gasoline

theorem mileage_per_gallon :
  car_mileage 190 4.75 = 40 :=
by
  -- proof omitted
  sorry

end mileage_per_gallon_l227_227791


namespace john_weekly_earnings_before_raise_l227_227521

theorem john_weekly_earnings_before_raise :
  ∀(x : ℝ), (70 = 1.0769 * x) → x = 64.99 :=
by
  intros x h
  sorry

end john_weekly_earnings_before_raise_l227_227521


namespace eleven_not_sum_of_two_primes_l227_227956

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem eleven_not_sum_of_two_primes :
  ¬ ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = 11 :=
by sorry

end eleven_not_sum_of_two_primes_l227_227956


namespace bucket_B_more_than_C_l227_227613

-- Define the number of pieces of fruit in bucket B as a constant
def B := 12

-- Define the number of pieces of fruit in bucket C as a constant
def C := 9

-- Define the number of pieces of fruit in bucket A based on B
def A := B + 4

-- Define the total number of pieces of fruit in all three buckets
def total_fruit := A + B + C

-- Prove that bucket B has 3 more pieces of fruit than bucket C
theorem bucket_B_more_than_C : B - C = 3 := by
  -- sorry is used to skip the proof
  sorry

end bucket_B_more_than_C_l227_227613


namespace sum_real_imag_parts_l227_227467

open Complex

theorem sum_real_imag_parts (z : ℂ) (i : ℂ) (i_property : i * i = -1) (z_eq : z * i = -1 + i) :
  (z.re + z.im = 2) :=
  sorry

end sum_real_imag_parts_l227_227467


namespace simplest_form_eq_a_l227_227501

theorem simplest_form_eq_a (a : ℝ) (h : a ≠ 1) : 1 - (1 / (1 + (a / (1 - a)))) = a :=
by sorry

end simplest_form_eq_a_l227_227501


namespace greatest_divisible_by_13_l227_227323

theorem greatest_divisible_by_13 (A B C : ℕ) (hA : 1 ≤ A ∧ A ≤ 9) (hB : 0 ≤ B ∧ B ≤ 9) (hC : 0 ≤ C ∧ C ≤ 9)
  (distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C) : (10000 * A + 1000 * B + 100 * C + 10 * B + A = 96769) 
  ↔ (10000 * A + 1000 * B + 100 * C + 10 * B + A) % 13 = 0 :=
sorry

end greatest_divisible_by_13_l227_227323


namespace factorize_xcube_minus_x_l227_227555

theorem factorize_xcube_minus_x (x : ℝ) : x^3 - x = x * (x + 1) * (x - 1) :=
by 
  sorry

end factorize_xcube_minus_x_l227_227555


namespace kristin_annual_income_l227_227196

theorem kristin_annual_income (p : ℝ) :
  ∃ A : ℝ, 
  (0.01 * p * 28000 + 0.01 * (p + 2) * (A - 28000) = (0.01 * (p + 0.25) * A)) ∧
  A = 32000 :=
by
  sorry

end kristin_annual_income_l227_227196


namespace full_price_ticket_revenue_correct_l227_227337

-- Define the constants and assumptions
variables (f t : ℕ) (p : ℝ)

-- Total number of tickets sold
def total_tickets := (f + t = 180)

-- Total revenue from ticket sales
def total_revenue := (f * p + t * (p / 3) = 2600)

-- Full price ticket revenue
def full_price_revenue := (f * p = 975)

-- The theorem combines the above conditions to prove the correct revenue from full-price tickets
theorem full_price_ticket_revenue_correct :
  total_tickets f t →
  total_revenue f t p →
  full_price_revenue f p :=
by
  sorry

end full_price_ticket_revenue_correct_l227_227337


namespace max_selection_no_five_times_l227_227667

theorem max_selection_no_five_times (S : Finset ℕ) (hS : S = Finset.Icc 1 2014) :
  ∃ n, n = 1665 ∧ 
  ∀ (a b : ℕ), a ∈ S → b ∈ S → (a = 5 * b ∨ b = 5 * a) → false :=
sorry

end max_selection_no_five_times_l227_227667


namespace max_pawns_l227_227097

def chessboard : Type := ℕ × ℕ -- Define a chessboard as a grid of positions (1,1) to (8,8)
def e4 : chessboard := (5, 4) -- Define the position e4
def symmetric_wrt_e4 (p1 p2 : chessboard) : Prop :=
  p1.1 + p2.1 = 10 ∧ p1.2 + p2.2 = 8 -- Symmetry condition relative to e4

def placed_on (pos : chessboard) : Prop := sorry -- placeholder for placement condition

theorem max_pawns (no_e4 : ¬ placed_on e4)
  (no_symmetric_pairs : ∀ p1 p2, symmetric_wrt_e4 p1 p2 → ¬ (placed_on p1 ∧ placed_on p2)) :
  ∃ max_pawns : ℕ, max_pawns = 39 :=
sorry

end max_pawns_l227_227097


namespace people_born_in_country_l227_227959

-- Define the conditions
def people_immigrated : ℕ := 16320
def new_people_total : ℕ := 106491

-- Define the statement to be proven
theorem people_born_in_country (people_born : ℕ) (h : people_born = new_people_total - people_immigrated) : 
    people_born = 90171 :=
  by
    -- This is where we would provide the proof, but we use sorry to skip the proof.
    sorry

end people_born_in_country_l227_227959


namespace range_of_a_l227_227315

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if h : x ≤ 1 then (a - 2) * x - 1 else Real.log x / Real.log a

theorem range_of_a (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ 2 < a ∧ a ≤ 3 :=
by {
  sorry
}

end range_of_a_l227_227315


namespace obtuse_triangle_count_l227_227083

-- Definitions based on conditions
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def is_arithmetic_sequence (a b c : ℕ) : Prop :=
  2 * b = a + c

def is_obtuse_triangle (a b c : ℕ) : Prop :=
  a * a + b * b < c * c ∨ b * b + c * c < a * a ∨ c * c + a * a < b * b

-- Main conjecture to prove
theorem obtuse_triangle_count :
  ∃ (n : ℕ), n = 157 ∧
    ∀ (a b c : ℕ), 
      a <= 50 ∧ b <= 50 ∧ c <= 50 ∧ 
      is_arithmetic_sequence a b c ∧ 
      is_triangle a b c ∧ 
      is_obtuse_triangle a b c → 
    true := sorry

end obtuse_triangle_count_l227_227083


namespace range_of_x_l227_227131

def y_function (x : ℝ) : ℝ := x

def y_translated (x : ℝ) : ℝ := x + 2

theorem range_of_x {x : ℝ} (h : y_translated x > 0) : x > -2 := 
by {
  sorry
}

end range_of_x_l227_227131


namespace range_of_m_l227_227307

open Set Real

-- Define over the real numbers ℝ
noncomputable def A : Set ℝ := { x : ℝ | x^2 - 2*x - 3 ≤ 0 }
noncomputable def B (m : ℝ) : Set ℝ := { x : ℝ | x^2 - 2*m*x + m^2 - 4 ≤ 0 }
noncomputable def CRB (m : ℝ) : Set ℝ := { x : ℝ | x < m - 2 ∨ x > m + 2 }

-- Main theorem statement
theorem range_of_m (m : ℝ) (h : A ⊆ CRB m) : m < -3 ∨ m > 5 :=
sorry

end range_of_m_l227_227307


namespace outOfPocketCost_l227_227092

noncomputable def visitCost : ℝ := 300
noncomputable def castCost : ℝ := 200
noncomputable def insuranceCoverage : ℝ := 0.60

theorem outOfPocketCost : (visitCost + castCost - (visitCost + castCost) * insuranceCoverage) = 200 := by
  sorry

end outOfPocketCost_l227_227092


namespace cylindrical_plane_l227_227716

open Set

-- Define a cylindrical coordinate point (r, θ, z)
structure CylindricalCoord where
  r : ℝ
  theta : ℝ
  z : ℝ

-- Condition 1: In cylindrical coordinates, z is the height
def height_in_cylindrical := λ coords : CylindricalCoord => coords.z 

-- Condition 2: z is constant c
variable (c : ℝ)

-- The theorem to be proven
theorem cylindrical_plane (c : ℝ) :
  {p : CylindricalCoord | p.z = c} = {q : CylindricalCoord | q.z = c} :=
by
  sorry

end cylindrical_plane_l227_227716


namespace complement_union_A_B_l227_227961

def U : Set ℕ := {x | x < 6}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {2, 4}

theorem complement_union_A_B :
  (U \ (A ∪ B)) = {0, 5} := by
  sorry

end complement_union_A_B_l227_227961


namespace marty_combinations_l227_227393

theorem marty_combinations : 
  ∃ n : ℕ, n = 5 * 4 ∧ n = 20 :=
by
  sorry

end marty_combinations_l227_227393


namespace find_particular_number_l227_227180

theorem find_particular_number (x : ℤ) (h : ((x / 23) - 67) * 2 = 102) : x = 2714 := 
by 
  sorry

end find_particular_number_l227_227180


namespace general_solution_of_differential_eq_l227_227412

noncomputable def y (x C : ℝ) : ℝ := x * (Real.exp (x ^ 2) + C)

theorem general_solution_of_differential_eq {x C : ℝ} (h : x ≠ 0) :
  let y' := (1 : ℝ) * (Real.exp (x ^ 2) + C) + x * (2 * x * Real.exp (x ^ 2))
  y' = (y x C / x) + 2 * x ^ 2 * Real.exp (x ^ 2) :=
by
  -- the proof goes here
  sorry

end general_solution_of_differential_eq_l227_227412


namespace length_of_FD_l227_227739

theorem length_of_FD
  (ABCD_is_square : ∀ (A B C D : ℝ), A = 8 ∧ B = 8 ∧ C = 8 ∧ D = 8)
  (E_midpoint_AD : ∀ (A D E : ℝ), E = (A + D) / 2)
  (F_on_BD : ∀ (B D F E : ℝ), B = 8 ∧ F = 3 ∧ D = 8 ∧ E = 4):
  ∃ (FD : ℝ), FD = 3 := by
  sorry

end length_of_FD_l227_227739


namespace remainder_4x_div_9_l227_227425

theorem remainder_4x_div_9 (x : ℕ) (k : ℤ) (h : x = 9 * k + 5) : (4 * x) % 9 = 2 := 
by sorry

end remainder_4x_div_9_l227_227425


namespace min_ab_min_a_b_max_two_a_one_b_min_one_a_sq_four_b_sq_l227_227705

variables (a b : ℝ)
variables (ha : a > 0) (hb : b > 0) (h : 4 * a + b = a * b)

theorem min_ab : 16 ≤ a * b :=
sorry

theorem min_a_b : 9 ≤ a + b :=
sorry

theorem max_two_a_one_b : 2 > (2 / a + 1 / b) :=
sorry

theorem min_one_a_sq_four_b_sq : 1 / 5 ≤ (1 / a^2 + 4 / b^2) :=
sorry

end min_ab_min_a_b_max_two_a_one_b_min_one_a_sq_four_b_sq_l227_227705


namespace john_school_year_hours_l227_227712

theorem john_school_year_hours (summer_earnings : ℝ) (summer_hours_per_week : ℝ) (summer_weeks : ℝ) (target_school_earnings : ℝ) (school_weeks : ℝ) :
  summer_earnings = 4000 → summer_hours_per_week = 40 → summer_weeks = 8 → target_school_earnings = 5000 → school_weeks = 25 →
  (target_school_earnings / (summer_earnings / (summer_hours_per_week * summer_weeks)) / school_weeks) = 16 :=
by
  sorry

end john_school_year_hours_l227_227712


namespace garden_perimeter_l227_227781

theorem garden_perimeter
  (a b : ℝ)
  (h1: a^2 + b^2 = 225)
  (h2: a * b = 54) :
  2 * (a + b) = 2 * Real.sqrt 333 :=
by
  sorry

end garden_perimeter_l227_227781


namespace min_area_quadrilateral_l227_227774

theorem min_area_quadrilateral
  (S_AOB S_COD : ℝ) (h₁ : S_AOB = 4) (h₂ : S_COD = 9) :
  ∃ S_BOC S_AOD, S_AOB + S_COD + S_BOC + S_AOD = 25 :=
by
  sorry

end min_area_quadrilateral_l227_227774


namespace intersection_S_T_l227_227358

def S := {x : ℝ | abs x < 5}
def T := {x : ℝ | (x + 7) * (x - 3) < 0}

theorem intersection_S_T : S ∩ T = {x : ℝ | -5 < x ∧ x < 3} :=
by
  sorry

end intersection_S_T_l227_227358


namespace jen_total_birds_l227_227177

theorem jen_total_birds (C D G : ℕ) (h1 : D = 150) (h2 : D = 4 * C + 10) (h3 : G = (D + C) / 2) :
  D + C + G = 277 := sorry

end jen_total_birds_l227_227177


namespace zoo_children_count_l227_227821

theorem zoo_children_count:
  ∀ (C : ℕ), 
  (10 * C + 16 * 10 = 220) → 
  C = 6 :=
by
  intro C
  intro h
  sorry

end zoo_children_count_l227_227821


namespace solve_for_k_l227_227714

def sameLine (p1 p2 p3 : ℝ × ℝ) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p1.1) = (p3.2 - p1.2) * (p2.1 - p1.1)

theorem solve_for_k :
  (sameLine (3, 10) (1, k) (-7, 2)) → k = 8.4 :=
by
  sorry

end solve_for_k_l227_227714


namespace number_of_students_l227_227247

theorem number_of_students (n : ℕ) :
  n < 50 ∧ n % 6 = 4 ∧ n % 4 = 2 → n = 10 ∨ n = 22 ∨ n = 34 := by
  -- Proof goes here
  sorry

end number_of_students_l227_227247


namespace segment_length_is_13_l227_227268

def point := (ℝ × ℝ)

def p1 : point := (2, 3)
def p2 : point := (7, 15)

noncomputable def distance (p1 p2 : point) : ℝ :=
  Real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem segment_length_is_13 : distance p1 p2 = 13 := by
  sorry

end segment_length_is_13_l227_227268


namespace find_m_l227_227214

theorem find_m (m x : ℝ) 
  (h1 : (m - 1) * x^2 + 5 * x + m^2 - 3 * m + 2 = 0) 
  (h2 : m^2 - 3 * m + 2 = 0)
  (h3 : m ≠ 1) : 
  m = 2 := 
sorry

end find_m_l227_227214


namespace trig_proof_l227_227040

variable {α a : ℝ}

theorem trig_proof (h₁ : (∃ a : ℝ, a < 0 ∧ (4 * a, -3 * a) = (4 * a, -3 * a)))
                    (h₂ : a < 0) :
  2 * Real.sin α + Real.cos α = 2 / 5 := 
sorry

end trig_proof_l227_227040


namespace solve_congruence_l227_227535

theorem solve_congruence (n : ℕ) (h₀ : 0 ≤ n ∧ n < 47) (h₁ : 13 * n ≡ 5 [MOD 47]) :
  n = 4 :=
sorry

end solve_congruence_l227_227535


namespace triangle_ABC_AC_l227_227316

-- Defining the relevant points and lengths in the triangle
variables {A B C D : Type} 
variables (AB CD : ℝ)
variables (AD BC AC : ℝ)

-- Given constants
axiom hAB : AB = 3
axiom hCD : CD = Real.sqrt 3
axiom hAD_BC : AD = BC

-- The final theorem statement that needs to be proved
theorem triangle_ABC_AC :
  (AD = BC) ∧ (CD = Real.sqrt 3) ∧ (AB = 3) → AC = Real.sqrt 7 :=
by
  intros h
  sorry

end triangle_ABC_AC_l227_227316


namespace susan_initial_amount_l227_227589

theorem susan_initial_amount :
  ∃ S: ℝ, (S - (1/5 * S + 1/4 * S + 120) = 1200) → S = 2400 :=
by
  sorry

end susan_initial_amount_l227_227589


namespace mary_total_nickels_l227_227587

-- Define the initial number of nickels Mary had
def mary_initial_nickels : ℕ := 7

-- Define the number of nickels her dad gave her
def mary_received_nickels : ℕ := 5

-- The goal is to prove the total number of nickels Mary has now is 12
theorem mary_total_nickels : mary_initial_nickels + mary_received_nickels = 12 :=
by
  sorry

end mary_total_nickels_l227_227587


namespace range_of_a_l227_227675

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, -2 ≤ x ∧ x ≤ 2 → 0 ≤ f x ∧ f x ≤ 4) →
  (∀ x, -2 ≤ x ∧ x ≤ 2 → ∃ x₀, -2 ≤ x₀ ∧ x₀ ≤ 2 ∧ (a * x₀ - 1 = f x)) →
  a ∈ Set.Iic (-5/2) ∪ Set.Ici (5/2) :=
sorry

end range_of_a_l227_227675


namespace pies_can_be_made_l227_227785

def total_apples : Nat := 51
def apples_handout : Nat := 41
def apples_per_pie : Nat := 5

theorem pies_can_be_made :
  ((total_apples - apples_handout) / apples_per_pie) = 2 := by
  sorry

end pies_can_be_made_l227_227785


namespace soda_mineral_cost_l227_227206

theorem soda_mineral_cost
  (x y : ℝ)
  (h1 : 2 * x + y = 7)
  (h2 : 4 * x + 3 * y = 16) :
  10 * x + 10 * y = 45 :=
  sorry

end soda_mineral_cost_l227_227206


namespace equation1_solution_equation2_solution_l227_227095

theorem equation1_solution : ∀ x : ℚ, x - 0.4 * x = 120 → x = 200 := by
  sorry

theorem equation2_solution : ∀ x : ℚ, 5 * x - 5/6 = 5/4 → x = 5/12 := by
  sorry

end equation1_solution_equation2_solution_l227_227095


namespace treadmill_time_saved_l227_227962

theorem treadmill_time_saved:
  let monday_speed := 6
  let tuesday_speed := 4
  let wednesday_speed := 5
  let thursday_speed := 6
  let friday_speed := 3
  let distance := 3 
  let daily_times : List ℚ := 
    [distance/monday_speed, distance/tuesday_speed, distance/wednesday_speed, distance/thursday_speed, distance/friday_speed]
  let total_time := (daily_times.map (λ t => t)).sum
  let total_distance := 5 * distance 
  let uniform_speed := 5 
  let uniform_time := total_distance / uniform_speed 
  let time_difference := total_time - uniform_time 
  let time_in_minutes := time_difference * 60 
  time_in_minutes = 21 := 
by 
  sorry

end treadmill_time_saved_l227_227962


namespace find_a_l227_227036

theorem find_a (a : ℝ) (h₁ : ¬ (a = 0)) (h_perp : (∀ x y : ℝ, (a * x + 1 = 0) 
  -> (a - 2) * x + y + a = 0 -> ∀ x₁ y₁, (a * x₁ + 1 = 0) -> y = y₁)) : a = 2 := 
by 
  sorry

end find_a_l227_227036


namespace not_square_of_expression_l227_227491

theorem not_square_of_expression (n : ℕ) (h : n > 0) : ¬ ∃ m : ℤ, m * m = 2 * n * n + 2 - n :=
by
  sorry

end not_square_of_expression_l227_227491


namespace cost_of_headphones_l227_227444

-- Define the constants for the problem
def bus_ticket_cost : ℕ := 11
def drinks_and_snacks_cost : ℕ := 3
def wifi_cost_per_hour : ℕ := 2
def trip_hours : ℕ := 3
def earnings_per_hour : ℕ := 12
def total_earnings := earnings_per_hour * trip_hours
def total_expenses_without_headphones := bus_ticket_cost + drinks_and_snacks_cost + (wifi_cost_per_hour * trip_hours)

-- Prove the cost of headphones, H, is $16 
theorem cost_of_headphones : total_earnings = total_expenses_without_headphones + 16 := by
  -- setup the goal
  sorry

end cost_of_headphones_l227_227444


namespace number_of_boxes_l227_227496

theorem number_of_boxes (eggs_per_box : ℕ) (total_eggs : ℕ) (h1 : eggs_per_box = 3) (h2 : total_eggs = 6) : total_eggs / eggs_per_box = 2 := by
  sorry

end number_of_boxes_l227_227496


namespace tan_alpha_plus_pi_over_4_sin_2alpha_expr_l227_227898

open Real

theorem tan_alpha_plus_pi_over_4 (α : ℝ) (h : tan α = 2) : tan (α + π / 4) = -3 :=
by
  sorry

theorem sin_2alpha_expr (α : ℝ) (h : tan α = 2) :
  (sin (2 * α)) / (sin (α) ^ 2 + sin (α) * cos (α)) = 2 / 3 :=
by
  sorry

end tan_alpha_plus_pi_over_4_sin_2alpha_expr_l227_227898


namespace speed_of_water_l227_227281

theorem speed_of_water (v : ℝ) (swim_speed_still_water : ℝ)
  (distance : ℝ) (time : ℝ)
  (h1 : swim_speed_still_water = 4) 
  (h2 : distance = 14) 
  (h3 : time = 7) 
  (h4 : 4 - v = distance / time) : 
  v = 2 := 
sorry

end speed_of_water_l227_227281


namespace grasshopper_jump_l227_227087

-- Definitions for the distances jumped
variables (G F M : ℕ)

-- Conditions given in the problem
def condition1 : Prop := G = F + 19
def condition2 : Prop := M = F - 12
def condition3 : Prop := M = 8

-- The theorem statement
theorem grasshopper_jump : condition1 G F ∧ condition2 F M ∧ condition3 M → G = 39 :=
by
  sorry

end grasshopper_jump_l227_227087


namespace find_A_minus_B_l227_227991

def A : ℤ := 3^7 + Nat.choose 7 2 * 3^5 + Nat.choose 7 4 * 3^3 + Nat.choose 7 6 * 3
def B : ℤ := Nat.choose 7 1 * 3^6 + Nat.choose 7 3 * 3^4 + Nat.choose 7 5 * 3^2 + 1

theorem find_A_minus_B : A - B = 128 := 
by
  -- Proof goes here
  sorry

end find_A_minus_B_l227_227991


namespace least_isosceles_triangles_cover_rectangle_l227_227217

-- Define the dimensions of the rectangle
def rectangle_height : ℕ := 10
def rectangle_width : ℕ := 100

-- Define the least number of isosceles right triangles needed to cover the rectangle
def least_number_of_triangles (h w : ℕ) : ℕ :=
  if h = rectangle_height ∧ w = rectangle_width then 11 else 0

-- The theorem statement
theorem least_isosceles_triangles_cover_rectangle :
  least_number_of_triangles rectangle_height rectangle_width = 11 :=
by
  -- skip the proof
  sorry

end least_isosceles_triangles_cover_rectangle_l227_227217


namespace total_wet_surface_area_is_correct_l227_227879

noncomputable def wet_surface_area (cistern_length cistern_width water_depth platform_length platform_width platform_height : ℝ) : ℝ :=
  let two_longer_walls := 2 * (cistern_length * water_depth)
  let two_shorter_walls := 2 * (cistern_width * water_depth)
  let area_walls := two_longer_walls + two_shorter_walls
  let area_bottom := cistern_length * cistern_width
  let submerged_height := water_depth - platform_height
  let two_longer_sides_platform := 2 * (platform_length * submerged_height)
  let two_shorter_sides_platform := 2 * (platform_width * submerged_height)
  let area_platform_sides := two_longer_sides_platform + two_shorter_sides_platform
  area_walls + area_bottom + area_platform_sides

theorem total_wet_surface_area_is_correct :
  wet_surface_area 8 4 1.25 1 0.5 0.75 = 63.5 :=
by
  -- The proof goes here
  sorry

end total_wet_surface_area_is_correct_l227_227879


namespace tom_watches_movies_total_duration_l227_227770

-- Define the running times for each movie
def M := 120
def A := M - 30
def B := A + 10
def D := 2 * B - 20

-- Define the number of times Tom watches each movie
def watch_B := 2
def watch_A := 3
def watch_M := 1
def watch_D := 4

-- Calculate the total time spent watching each movie
def total_time_B := watch_B * B
def total_time_A := watch_A * A
def total_time_M := watch_M * M
def total_time_D := watch_D * D

-- Calculate the total duration Tom spends watching these movies in a week
def total_duration := total_time_B + total_time_A + total_time_M + total_time_D

-- The statement to prove
theorem tom_watches_movies_total_duration :
  total_duration = 1310 := 
by
  sorry

end tom_watches_movies_total_duration_l227_227770


namespace solve_integer_divisibility_l227_227105

theorem solve_integer_divisibility :
  {n : ℕ | n < 589 ∧ 589 ∣ (n^2 + n + 1)} = {49, 216, 315, 482} :=
by
  sorry

end solve_integer_divisibility_l227_227105


namespace good_number_is_1008_l227_227404

-- Given conditions
def sum_1_to_2015 : ℕ := (2015 * (2015 + 1)) / 2
def sum_mod_2016 : ℕ := sum_1_to_2015 % 2016

-- The proof problem expressed in Lean
theorem good_number_is_1008 (x : ℕ) (h1 : sum_1_to_2015 = 2031120)
  (h2 : sum_mod_2016 = 1008) :
  x = 1008 ↔ (sum_1_to_2015 - x) % 2016 = 0 := by
  sorry

end good_number_is_1008_l227_227404


namespace martha_children_l227_227360

noncomputable def num_children (total_cakes : ℕ) (cakes_per_child : ℕ) : ℕ :=
  total_cakes / cakes_per_child

theorem martha_children : num_children 18 6 = 3 := by
  sorry

end martha_children_l227_227360


namespace sum_first_8_terms_of_geom_seq_l227_227430

-- Definitions: the sequence a_n, common ratio q, and the fact that specific terms form an arithmetic sequence.
def geom_seq (a : ℕ → ℕ) (a1 : ℕ) (q : ℕ) := ∀ n, a n = a1 * q^(n-1)
def arith_seq (b c d : ℕ) := 2 * b + (c - 2 * b) = d

-- Conditions
variables {a : ℕ → ℕ} {a1 : ℕ} {q : ℕ}
variables (h1 : geom_seq a a1 q) (h2 : q = 2)
variables (h3 : arith_seq (2 * a 4) (a 6) 48)

-- Goal: sum of the first 8 terms of the sequence equals 255
def sum_geometric_sequence (a1 : ℕ) (q : ℕ) (n : ℕ) := a1 * (1 - q^n) / (1 - q)

theorem sum_first_8_terms_of_geom_seq : 
  sum_geometric_sequence a1 q 8 = 255 :=
by
  sorry

end sum_first_8_terms_of_geom_seq_l227_227430


namespace total_charge_for_2_hours_l227_227539

theorem total_charge_for_2_hours (F A : ℕ) 
  (h1 : F = A + 40) 
  (h2 : F + 4 * A = 375) : 
  F + A = 174 :=
by 
  sorry

end total_charge_for_2_hours_l227_227539


namespace arithmetic_sequence_equal_sum_l227_227072

variable (a d : ℕ) -- defining first term and common difference as natural numbers
variable (n : ℕ) -- defining n as a natural number

noncomputable def sum_arithmetic_sequence (n: ℕ) (a d: ℕ): ℕ := (n * (2 * a + (n - 1) * d) ) / 2

theorem arithmetic_sequence_equal_sum (a d n : ℕ) :
  sum_arithmetic_sequence (10 * n) a d = sum_arithmetic_sequence (15 * n) a d - sum_arithmetic_sequence (10 * n) a d :=
by
  sorry

end arithmetic_sequence_equal_sum_l227_227072


namespace vertices_of_parabolas_is_parabola_l227_227389

theorem vertices_of_parabolas_is_parabola 
  (a c k : ℝ) (ha : 0 < a) (hc : 0 < c) (hk : 0 < k) :
  ∃ (f : ℝ → ℝ), (∀ t : ℝ, f t = (-k^2 / (4 * a)) * t^2 + c) ∧ 
  ∀ (pt : ℝ × ℝ), (∃ t : ℝ, pt = (-(k * t) / (2 * a), f t)) → 
  ∃ a' b' c', (∀ t : ℝ, pt.2 = a' * pt.1^2 + b' * pt.1 + c') ∧ (a < 0) :=
by sorry

end vertices_of_parabolas_is_parabola_l227_227389


namespace depth_of_first_hole_l227_227434

-- Conditions as definitions in Lean 4
def number_of_workers_first_hole : Nat := 45
def hours_worked_first_hole : Nat := 8

def number_of_workers_second_hole : Nat := 110  -- 45 existing workers + 65 extra workers
def hours_worked_second_hole : Nat := 6
def depth_second_hole : Nat := 55

-- The key assumption that work done (W) is proportional to the depth of the hole (D)
theorem depth_of_first_hole :
  let work_first_hole := number_of_workers_first_hole * hours_worked_first_hole
  let work_second_hole := number_of_workers_second_hole * hours_worked_second_hole
  let depth_first_hole := (work_first_hole * depth_second_hole) / work_second_hole
  depth_first_hole = 30 := sorry

end depth_of_first_hole_l227_227434


namespace div_problem_l227_227707

theorem div_problem : 150 / (6 / 3) = 75 := by
  sorry

end div_problem_l227_227707


namespace triangle_identity_l227_227211

theorem triangle_identity (a b c : ℝ) (B: ℝ) (hB: B = 120) :
    a^2 + a * c + c^2 - b^2 = 0 :=
by
  sorry

end triangle_identity_l227_227211


namespace calculate_c_l227_227814

-- Define the given equation as a hypothesis
theorem calculate_c (a b k c : ℝ) (h : (1 / (k * a) - 1 / (k * b) = 1 / c)) :
  c = k * a * b / (b - a) :=
by
  sorry

end calculate_c_l227_227814


namespace positive_integer_solution_l227_227728

/-- Given that x, y, and t are all equal to 1, and x + y + z + t = 10, we need to prove that z = 7. -/
theorem positive_integer_solution {x y z t : ℕ} (hx : x = 1) (hy : y = 1) (ht : t = 1) (h : x + y + z + t = 10) : z = 7 :=
by {
  -- We would provide the proof here, but for now, we use sorry
  sorry
}

end positive_integer_solution_l227_227728


namespace correct_weights_l227_227099

def weight (item : String) : Nat :=
  match item with
  | "Banana" => 140
  | "Pear" => 120
  | "Melon" => 1500
  | "Tomato" => 150
  | "Apple" => 170
  | _ => 0

theorem correct_weights :
  weight "Banana" = 140 ∧
  weight "Pear" = 120 ∧
  weight "Melon" = 1500 ∧
  weight "Tomato" = 150 ∧
  weight "Apple" = 170 ∧
  (weight "Melon" > weight "Pear") ∧
  (weight "Melon" < weight "Tomato") :=
by
  sorry

end correct_weights_l227_227099


namespace probability_at_least_one_boy_and_one_girl_l227_227544

noncomputable def mathematics_club_prob : ℚ :=
  let boys := 14
  let girls := 10
  let total_members := 24
  let total_committees := Nat.choose total_members 5
  let boys_committees := Nat.choose boys 5
  let girls_committees := Nat.choose girls 5
  let committees_with_at_least_one_boy_and_one_girl := total_committees - (boys_committees + girls_committees)
  let probability := (committees_with_at_least_one_boy_and_one_girl : ℚ) / (total_committees : ℚ)
  probability

theorem probability_at_least_one_boy_and_one_girl :
  mathematics_club_prob = (4025 : ℚ) / 4251 :=
by
  sorry

end probability_at_least_one_boy_and_one_girl_l227_227544


namespace clara_climbs_stone_blocks_l227_227021

-- Define the number of steps per level
def steps_per_level : Nat := 8

-- Define the number of blocks per step
def blocks_per_step : Nat := 3

-- Define the number of levels in the tower
def levels : Nat := 4

-- Define a function to compute the total number of blocks given the constants
def total_blocks (steps_per_level blocks_per_step levels : Nat) : Nat :=
  steps_per_level * blocks_per_step * levels

-- Statement of the theorem
theorem clara_climbs_stone_blocks :
  total_blocks steps_per_level blocks_per_step levels = 96 :=
by
  -- Lean requires 'sorry' as a placeholder for the proof.
  sorry

end clara_climbs_stone_blocks_l227_227021


namespace g_six_g_seven_l227_227673

noncomputable def g : ℝ → ℝ :=
sorry

axiom additivity : ∀ x y : ℝ, g (x + y) = g x + g y
axiom g_three : g 3 = 4

theorem g_six : g 6 = 8 :=
by {
  -- proof steps to be added by the prover
  sorry
}

theorem g_seven : g 7 = 28 / 3 :=
by {
  -- proof steps to be added by the prover
  sorry
}

end g_six_g_seven_l227_227673


namespace candy_seller_initial_candies_l227_227590

-- Given conditions
def num_clowns : ℕ := 4
def num_children : ℕ := 30
def candies_per_person : ℕ := 20
def candies_left : ℕ := 20

-- Question: What was the initial number of candies?
def total_people : ℕ := num_clowns + num_children
def total_candies_given_out : ℕ := total_people * candies_per_person
def initial_candies : ℕ := total_candies_given_out + candies_left

theorem candy_seller_initial_candies : initial_candies = 700 :=
by
  sorry

end candy_seller_initial_candies_l227_227590


namespace find_y_l227_227060

theorem find_y (y : ℚ) (h : 1/3 - 1/4 = 4/y) : y = 48 := sorry

end find_y_l227_227060


namespace range_of_a_l227_227441

noncomputable def has_two_distinct_real_roots (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a^x₁ = x₁ ∧ a^x₂ = x₂

theorem range_of_a (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : has_two_distinct_real_roots a) : 
  1 < a ∧ a < Real.exp (1 / Real.exp 1) :=
sorry

end range_of_a_l227_227441


namespace proof_P_l227_227759

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the complement of P in U
def CU_P : Set ℕ := {4, 5}

-- Define the set P as the difference between U and CU_P
def P : Set ℕ := U \ CU_P

-- Prove that P = {1, 2, 3}
theorem proof_P :
  P = {1, 2, 3} :=
by
  sorry

end proof_P_l227_227759


namespace relationship_among_neg_a_square_neg_a_cube_l227_227607

theorem relationship_among_neg_a_square_neg_a_cube (a : ℝ) (h : -1 < a ∧ a < 0) : (-a > a^2 ∧ a^2 > -a^3) :=
by
  sorry

end relationship_among_neg_a_square_neg_a_cube_l227_227607


namespace sqrt_three_squared_l227_227612

theorem sqrt_three_squared : (Real.sqrt 3) ^ 2 = 3 := by
  sorry

end sqrt_three_squared_l227_227612


namespace probability_of_two_red_shoes_is_0_1332_l227_227726

def num_red_shoes : ℕ := 4
def num_green_shoes : ℕ := 6
def total_shoes : ℕ := num_red_shoes + num_green_shoes

def probability_first_red_shoe : ℚ := num_red_shoes / total_shoes
def remaining_red_shoes_after_first_draw : ℕ := num_red_shoes - 1
def remaining_shoes_after_first_draw : ℕ := total_shoes - 1
def probability_second_red_shoe : ℚ := remaining_red_shoes_after_first_draw / remaining_shoes_after_first_draw

def probability_two_red_shoes : ℚ := probability_first_red_shoe * probability_second_red_shoe

theorem probability_of_two_red_shoes_is_0_1332 : probability_two_red_shoes = 1332 / 10000 :=
by
  sorry

end probability_of_two_red_shoes_is_0_1332_l227_227726


namespace compound_interest_second_year_l227_227647

variables {P r CI_2 CI_3 : ℝ}

-- Given conditions as definitions in Lean
def interest_rate : ℝ := 0.05
def year_3_interest : ℝ := 1260
def relation_between_CI2_and_CI3 (CI_2 CI_3 : ℝ) : Prop :=
  CI_3 = CI_2 * (1 + interest_rate)

-- The theorem to prove
theorem compound_interest_second_year :
  relation_between_CI2_and_CI3 CI_2 year_3_interest ∧
  r = interest_rate →
  CI_2 = 1200 := 
sorry

end compound_interest_second_year_l227_227647


namespace art_collection_total_cost_l227_227141

theorem art_collection_total_cost 
  (price_first_three : ℕ)
  (price_fourth : ℕ)
  (total_first_three : price_first_three * 3 = 45000)
  (price_fourth_cond : price_fourth = price_first_three + (price_first_three / 2)) :
  3 * price_first_three + price_fourth = 67500 :=
by
  sorry

end art_collection_total_cost_l227_227141


namespace new_bag_marbles_l227_227351

open Nat

theorem new_bag_marbles 
  (start_marbles : ℕ)
  (lost_marbles : ℕ)
  (given_marbles : ℕ)
  (received_back_marbles : ℕ)
  (end_marbles : ℕ)
  (h_start : start_marbles = 40)
  (h_lost : lost_marbles = 3)
  (h_given : given_marbles = 5)
  (h_received_back : received_back_marbles = 2 * given_marbles)
  (h_end : end_marbles = 54) :
  (end_marbles = (start_marbles - lost_marbles - given_marbles + received_back_marbles + new_bag) ∧ new_bag = 12) :=
by
  sorry

end new_bag_marbles_l227_227351


namespace chocolate_distribution_l227_227325

theorem chocolate_distribution (n : ℕ) 
  (h1 : 12 * 2 ≤ n * 2 ∨ n * 2 ≤ 12 * 2) 
  (h2 : ∃ d : ℚ, (12 / n) = d ∧ d * n = 12) : 
  n = 15 :=
by 
  sorry

end chocolate_distribution_l227_227325


namespace tan_alpha_value_tan_beta_value_sum_angles_l227_227921

open Real

noncomputable def tan_alpha (α : ℝ) : ℝ := sin α / cos α
noncomputable def tan_beta (β : ℝ) : ℝ := sin β / cos β

def conditions (α β : ℝ) :=
  α > 0 ∧ α < π / 2 ∧ β > 0 ∧ β < π / 2 ∧ 
  sin α = 1 / sqrt 10 ∧ tan β = 1 / 7

theorem tan_alpha_value (α β : ℝ) (h : conditions α β) : tan_alpha α = 1 / 3 := sorry

theorem tan_beta_value (α β : ℝ) (h : conditions α β) : tan_beta β = 1 / 7 := sorry

theorem sum_angles (α β : ℝ) (h : conditions α β) : 2 * α + β = π / 4 := sorry

end tan_alpha_value_tan_beta_value_sum_angles_l227_227921


namespace total_amount_collected_in_paise_total_amount_collected_in_rupees_l227_227127

-- Definitions and conditions
def num_members : ℕ := 96
def contribution_per_member : ℕ := 96
def total_paise_collected : ℕ := num_members * contribution_per_member
def total_rupees_collected : ℚ := total_paise_collected / 100

-- Theorem stating the total amount collected
theorem total_amount_collected_in_paise :
  total_paise_collected = 9216 := by sorry

theorem total_amount_collected_in_rupees :
  total_rupees_collected = 92.16 := by sorry

end total_amount_collected_in_paise_total_amount_collected_in_rupees_l227_227127


namespace origami_papers_total_l227_227631

-- Define the conditions as Lean definitions
def num_cousins : ℕ := 6
def papers_per_cousin : ℕ := 8

-- Define the total number of origami papers that Haley has to give away
def total_papers : ℕ := num_cousins * papers_per_cousin

-- Statement of the proof
theorem origami_papers_total : total_papers = 48 :=
by
  -- Skipping the proof for now
  sorry

end origami_papers_total_l227_227631


namespace total_hats_l227_227846

theorem total_hats (B G : ℕ) (cost_blue cost_green total_cost green_quantity : ℕ)
  (h1 : cost_blue = 6)
  (h2 : cost_green = 7)
  (h3 : total_cost = 530)
  (h4 : green_quantity = 20)
  (h5 : G = green_quantity)
  (h6 : total_cost = B * cost_blue + G * cost_green) :
  B + G = 85 :=
by
  sorry

end total_hats_l227_227846


namespace xiaofang_final_score_l227_227920

def removeHighestLowestScores (scores : List ℕ) : List ℕ :=
  let max_score := scores.maximum.getD 0
  let min_score := scores.minimum.getD 0
  scores.erase max_score |>.erase min_score

def average (scores : List ℕ) : ℚ :=
  (scores.sum : ℚ) / scores.length

theorem xiaofang_final_score :
  let scores := [95, 94, 91, 88, 91, 90, 94, 93, 91, 92]
  average (removeHighestLowestScores scores) = 92 := by
  sorry

end xiaofang_final_score_l227_227920


namespace factor_polynomial_l227_227163

theorem factor_polynomial : ∀ x : ℝ, 
  x^8 - 4 * x^6 + 6 * x^4 - 4 * x^2 + 1 = (x - 1)^4 * (x + 1)^4 :=
by
  intro x
  sorry

end factor_polynomial_l227_227163


namespace coefficient_x9_l227_227216

theorem coefficient_x9 (p : Polynomial ℚ) : 
  p = (1 + 3 * Polynomial.X - Polynomial.X^2)^5 →
  Polynomial.coeff p 9 = 15 := 
by
  intro h
  rw [h]
  -- additional lean tactics to prove the statement would go here
  sorry

end coefficient_x9_l227_227216


namespace convert_to_scientific_notation_l227_227104

theorem convert_to_scientific_notation :
  40.25 * 10^9 = 4.025 * 10^9 :=
by
  -- Sorry is used here to skip the proof
  sorry

end convert_to_scientific_notation_l227_227104


namespace ott_fraction_is_3_over_13_l227_227067

-- Defining the types and quantities involved
noncomputable def moes_original_money (amount_given: ℚ) := amount_given * 5
noncomputable def lokis_original_money (amount_given: ℚ) := amount_given * 5
noncomputable def nicks_original_money (amount_given: ℚ) := amount_given * 3

-- Total original money of the group (excluding Ott)
noncomputable def total_original_money (amount_given: ℚ) :=
  moes_original_money amount_given + lokis_original_money amount_given + nicks_original_money amount_given

-- Total money received by Ott
noncomputable def otts_received_money (amount_given: ℚ) := 3 * amount_given

-- Fraction of the group's total money Ott now has
noncomputable def otts_fraction_of_total_money (amount_given: ℚ) : ℚ :=
  otts_received_money amount_given / total_original_money amount_given

-- The theorem to be proved
theorem ott_fraction_is_3_over_13 :
  otts_fraction_of_total_money 1 = 3 / 13 :=
by
  -- The body of the proof is skipped with sorry
  sorry

end ott_fraction_is_3_over_13_l227_227067


namespace chairs_to_remove_l227_227839

/-- Given conditions:
1. Each row holds 13 chairs.
2. There are 169 chairs initially.
3. There are 95 expected attendees.

Task: 
Prove that the number of chairs to be removed to ensure complete rows and minimize empty seats is 65. -/
theorem chairs_to_remove (chairs_per_row total_chairs expected_attendees : ℕ)
  (h1 : chairs_per_row = 13)
  (h2 : total_chairs = 169)
  (h3 : expected_attendees = 95) :
  ∃ chairs_to_remove : ℕ, chairs_to_remove = 65 :=
by
  sorry -- proof omitted

end chairs_to_remove_l227_227839


namespace pascal_triangle_43rd_element_in_51_row_l227_227895

theorem pascal_triangle_43rd_element_in_51_row :
  (Nat.choose 50 42) = 10272278170 :=
  by
  -- proof construction here
  sorry

end pascal_triangle_43rd_element_in_51_row_l227_227895


namespace attendance_rate_correct_l227_227130

def total_students : ℕ := 50
def students_on_leave : ℕ := 2
def given_attendance_rate : ℝ := 96

theorem attendance_rate_correct :
  ((total_students - students_on_leave) / total_students * 100 : ℝ) = given_attendance_rate := sorry

end attendance_rate_correct_l227_227130


namespace rectangle_area_in_triangle_l227_227807

theorem rectangle_area_in_triangle (c k y : ℝ) (h1 : c > 0) (h2 : k > 0) (h3 : 0 < y) (h4 : y < k) : 
  ∃ A : ℝ, A = y * ((c * (k - y)) / k) := 
by
  sorry

end rectangle_area_in_triangle_l227_227807


namespace range_of_a_l227_227639

noncomputable def has_root_in_R (f : ℝ → ℝ) : Prop :=
∃ x : ℝ, f x = 0

theorem range_of_a (a : ℝ) (h : has_root_in_R (λ x => 4 * x + a * 2^x + a + 1)) : a ≤ 0 :=
sorry

end range_of_a_l227_227639


namespace Rachel_average_speed_l227_227411

noncomputable def total_distance : ℝ := 2 + 4 + 6

noncomputable def time_to_Alicia : ℝ := 2 / 3
noncomputable def time_to_Lisa : ℝ := 4 / 5
noncomputable def time_to_Nicholas : ℝ := 1 / 2

noncomputable def total_time : ℝ := (20 / 30) + (24 / 30) + (15 / 30)

noncomputable def average_speed : ℝ := total_distance / total_time

theorem Rachel_average_speed : average_speed = 360 / 59 :=
by
  sorry

end Rachel_average_speed_l227_227411


namespace tangent_sum_l227_227882

theorem tangent_sum :
  (Finset.sum (Finset.range 2019) (λ k => Real.tan ((k + 1) * Real.pi / 47) * Real.tan ((k + 2) * Real.pi / 47))) = -2021 :=
by
  -- proof will be completed here
  sorry

end tangent_sum_l227_227882


namespace perimeter_of_figure_l227_227948

-- Given conditions
def side_length : Nat := 2
def num_horizontal_segments : Nat := 16
def num_vertical_segments : Nat := 10

-- Define a function to calculate the perimeter based on the given conditions
def calculate_perimeter (side_length : Nat) (num_horizontal_segments : Nat) (num_vertical_segments : Nat) : Nat :=
  (num_horizontal_segments * side_length) + (num_vertical_segments * side_length)

-- Statement to be proved
theorem perimeter_of_figure : calculate_perimeter side_length num_horizontal_segments num_vertical_segments = 52 :=
by
  -- The proof would go here
  sorry

end perimeter_of_figure_l227_227948


namespace sum_of_digits_in_T_shape_35_l227_227205

-- Define the set of digits
def digits : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define the problem variables and conditions
theorem sum_of_digits_in_T_shape_35
  (a b c d e f g h : ℕ)
  (h1 : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧
        b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧
        c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧
        d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧
        e ≠ f ∧ e ≠ g ∧ e ≠ h ∧
        f ≠ g ∧ f ≠ h ∧
        g ≠ h)
  (h2 : a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧ 
        e ∈ digits ∧ f ∈ digits ∧ g ∈ digits ∧ h ∈ digits)
  (h3 : a + b + c + d = 26)
  (h4 : e + b + f + g + h = 20) :
  a + b + c + d + e + f + g + h = 35 := by
  sorry

end sum_of_digits_in_T_shape_35_l227_227205


namespace necessary_and_sufficient_condition_l227_227674

theorem necessary_and_sufficient_condition {a b : ℝ} :
  (a > b) ↔ (a^3 > b^3) := sorry

end necessary_and_sufficient_condition_l227_227674


namespace certain_fraction_ratio_l227_227564

theorem certain_fraction_ratio :
  ∃ x : ℚ,
    (2 / 5 : ℚ) / x = (0.46666666666666673 : ℚ) / (1 / 2) ∧ x = 3 / 7 :=
by sorry

end certain_fraction_ratio_l227_227564


namespace remainder_8_pow_2023_div_5_l227_227634

-- Definition for modulo operation
def mod_five (a : Nat) : Nat := a % 5

-- Key theorem to prove
theorem remainder_8_pow_2023_div_5 : mod_five (8 ^ 2023) = 2 :=
by
  sorry -- This is where the proof would go, but it's not required per the instructions

end remainder_8_pow_2023_div_5_l227_227634


namespace second_train_speed_l227_227338

noncomputable def speed_of_second_train (length1 length2 speed1 clearance_time : ℝ) : ℝ :=
  let total_distance := (length1 + length2) / 1000 -- convert meters to kilometers
  let time_in_hours := clearance_time / 3600 -- convert seconds to hours
  let relative_speed := total_distance / time_in_hours
  relative_speed - speed1

theorem second_train_speed : 
  speed_of_second_train 60 280 42 16.998640108791296 = 30.05 := 
by
  sorry

end second_train_speed_l227_227338


namespace polio_cases_in_1990_l227_227093

theorem polio_cases_in_1990 (c_1970 c_2000 : ℕ) (T : ℕ) (linear_decrease : ∀ t, c_1970 - (c_2000 * t) / T > 0):
  (c_1970 = 300000) → (c_2000 = 600) → (T = 30) → ∃ c_1990, c_1990 = 100400 :=
by
  intros
  sorry

end polio_cases_in_1990_l227_227093


namespace perfect_square_iff_n_eq_5_l227_227033

theorem perfect_square_iff_n_eq_5 (n : ℕ) (h_pos : 0 < n) :
  ∃ m : ℕ, n * 2^(n-1) + 1 = m^2 ↔ n = 5 := by
  sorry

end perfect_square_iff_n_eq_5_l227_227033


namespace Xiao_Ming_max_notebooks_l227_227721

-- Definitions of the given conditions
def total_yuan : ℝ := 30
def total_books : ℕ := 30
def notebook_cost : ℝ := 4
def exercise_book_cost : ℝ := 0.4

-- Definition of the variables used in the inequality
def x (max_notebooks : ℕ) : ℝ := max_notebooks
def exercise_books (max_notebooks : ℕ) : ℝ := total_books - x max_notebooks

-- Definition of the total cost inequality
def total_cost (max_notebooks : ℕ) : ℝ :=
  x max_notebooks * notebook_cost + exercise_books max_notebooks * exercise_book_cost

theorem Xiao_Ming_max_notebooks (max_notebooks : ℕ) : total_cost max_notebooks ≤ total_yuan → max_notebooks ≤ 5 :=
by
  -- Proof goes here
  sorry

end Xiao_Ming_max_notebooks_l227_227721


namespace proof_problem_l227_227239

def sum_of_sequence (S : ℕ → ℕ) (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, 0 < n → S n = n * (n + 1) + 2 ∧ S 1 = a 1 ∧ (∀ n, 1 < n → a n = S n - S (n - 1))

def general_term_a (a : ℕ → ℕ) : Prop :=
  a 1 = 4 ∧ (∀ n, 1 < n → a n = 2 * n)

def geometric_sequence (a : ℕ → ℕ) (b : ℕ → ℕ) : Prop :=
  ∀ k : ℕ, 0 < k → 
  a 2 = 4 ∧ a (k+2) = 2 * (k + 2) ∧ a (3 * k + 2) = 2 * (3 * k + 2) →
  b 1 = a 2 ∧ b 2 = a (k + 2) ∧ b 3 = a (3 * k + 2) ∧ 
  (∀ n, b n = 2^(n + 1))

theorem proof_problem :
  ∃ (a b S : ℕ → ℕ),
  sum_of_sequence S a ∧ general_term_a a ∧ geometric_sequence a b :=
sorry

end proof_problem_l227_227239


namespace correct_sentence_completion_l227_227506

-- Define the possible options
inductive Options
| A : Options  -- "However he was reminded frequently"
| B : Options  -- "No matter he was reminded frequently"
| C : Options  -- "However frequently he was reminded"
| D : Options  -- "No matter he was frequently reminded"

-- Define the correctness condition
def correct_option : Options := Options.C

-- Define the proof problem
theorem correct_sentence_completion (opt : Options) : opt = correct_option :=
by sorry

end correct_sentence_completion_l227_227506


namespace weight_lift_equality_l227_227464

-- Definitions based on conditions
def total_weight_25_pounds_lifted_times := 750
def total_weight_20_pounds_lifted_per_time (n : ℝ) := 60 * n

-- Statement of the proof problem
theorem weight_lift_equality : ∃ n, total_weight_20_pounds_lifted_per_time n = total_weight_25_pounds_lifted_times :=
  sorry

end weight_lift_equality_l227_227464


namespace car_catch_up_distance_l227_227018

-- Define the problem: Prove that Car A catches Car B at the specified location, given the conditions

theorem car_catch_up_distance
  (distance : ℝ := 300)
  (v_A v_B t_A : ℝ)
  (start_A_late : ℝ := 1) -- Car A leaves 1 hour later
  (arrive_A_early : ℝ := 1) -- Car A arrives 1 hour earlier
  (t : ℝ := 2) -- Time when Car A catches up with Car B
  (dist_eq_A : distance = v_A * t_A)
  (dist_eq_B : distance = v_B * (t_A + 2))
  (catch_up_time_eq : v_A * t = v_B * (t + 1)):
  (distance - v_A * t) = 150 := sorry

end car_catch_up_distance_l227_227018


namespace inscribed_rectangle_l227_227745

theorem inscribed_rectangle (b h : ℝ) : ∃ x : ℝ, 
  (∃ q : ℝ, x = q / 2) → 
  ∃ x : ℝ, 
    (∃ q : ℝ, q = 2 * x ∧ x = h * q / (2 * h + b)) :=
sorry

end inscribed_rectangle_l227_227745


namespace total_weight_new_group_l227_227992

variable (W : ℝ) -- Total weight of the original group of 20 people
variable (weights_old : List ℝ) 
variable (weights_new : List ℝ)

-- Given conditions
def five_weights_old : List ℝ := [40, 55, 60, 75, 80]
def average_weight_increase : ℝ := 2
def group_size : ℕ := 20
def num_replaced : ℕ := 5

-- Define theorem
theorem total_weight_new_group :
(W - five_weights_old.sum + group_size * average_weight_increase) -
(W - five_weights_old.sum) = weights_new.sum → 
weights_new.sum = 350 := 
by
  sorry

end total_weight_new_group_l227_227992


namespace range_of_a_l227_227531

open Set Real

def set_M (a : ℝ) : Set ℝ := { x | x * (x - a - 1) < 0 }
def set_N : Set ℝ := { x | x^2 - 2*x - 3 ≤ 0 }

theorem range_of_a (a : ℝ) : set_M a ⊆ set_N ↔ -2 ≤ a ∧ a ≤ 2 := by
  sorry

end range_of_a_l227_227531


namespace root_expression_value_l227_227876

theorem root_expression_value 
  (p q r s : ℝ)
  (h1 : p + q + r + s = 15)
  (h2 : p*q + p*r + p*s + q*r + q*s + r*s = 35)
  (h3 : p*q*r + p*q*s + q*r*s + p*r*s = 27)
  (h4 : p*q*r*s = 9)
  (h5 : ∀ x : ℝ, x^4 - 15*x^3 + 35*x^2 - 27*x + 9 = 0 → (x = p ∨ x = q ∨ x = r ∨ x = s)) :
  (p / (1 / p + q*r) + q / (1 / q + r*s) + r / (1 / r + s*p) + s / (1 / s + p*q) = 155 / 123) := 
sorry

end root_expression_value_l227_227876


namespace restaurant_bill_l227_227803

theorem restaurant_bill
    (t : ℝ)
    (h1 : ∀ k : ℝ, k = 9 * (t / 10 + 3)) :
    t = 270 :=
by
    sorry

end restaurant_bill_l227_227803


namespace flowers_to_embroider_l227_227017

-- Defining constants based on the problem conditions
def stitches_per_minute : ℕ := 4
def stitches_per_flower : ℕ := 60
def stitches_per_unicorn : ℕ := 180
def stitches_per_godzilla : ℕ := 800
def num_unicorns : ℕ := 3
def num_godzillas : ℕ := 1
def total_minutes : ℕ := 1085

-- Theorem statement to prove the number of flowers Carolyn wants to embroider
theorem flowers_to_embroider : 
  (total_minutes * stitches_per_minute - (num_godzillas * stitches_per_godzilla + num_unicorns * stitches_per_unicorn)) / stitches_per_flower = 50 :=
by
  sorry

end flowers_to_embroider_l227_227017


namespace weight_of_bowling_ball_l227_227573

-- Define weights of bowling ball and canoe
variable (b c : ℚ)

-- Problem conditions
def cond1 : Prop := (9 * b = 5 * c)
def cond2 : Prop := (4 * c = 120)

-- The statement to prove
theorem weight_of_bowling_ball (h1 : cond1 b c) (h2 : cond2 c) : b = 50 / 3 := sorry

end weight_of_bowling_ball_l227_227573


namespace function_positive_on_interval_l227_227014

theorem function_positive_on_interval (a : ℝ) : 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → (2 - a^2) * x + a > 0) ↔ 0 < a ∧ a < 2 :=
by
  sorry

end function_positive_on_interval_l227_227014


namespace R_depends_on_d_and_n_l227_227012

variable (n a d : ℕ)

noncomputable def s1 : ℕ := (n * (2 * a + (n - 1) * d)) / 2
noncomputable def s2 : ℕ := (2 * n * (2 * a + (2 * n - 1) * d)) / 2
noncomputable def s3 : ℕ := (3 * n * (2 * a + (3 * n - 1) * d)) / 2
noncomputable def R : ℕ := s3 n a d - s2 n a d - s1 n a d

theorem R_depends_on_d_and_n : R n a d = 2 * d * n^2 :=
by
  sorry

end R_depends_on_d_and_n_l227_227012


namespace binomial_expansion_a5_l227_227336

theorem binomial_expansion_a5 (x : ℝ) 
  (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℝ) 
  (h : (x - 1) ^ 8 = a_0 + a_1 * (1 + x) + a_2 * (1 + x) ^ 2 + a_3 * (1 + x) ^ 3 + a_4 * (1 + x) ^ 4 + a_5 * (1 + x) ^ 5 + a_6 * (1 + x) ^ 6 + a_7 * (1 + x) ^ 7 + a_8 * (1 + x) ^ 8) : 
  a_5 = -448 := 
sorry

end binomial_expansion_a5_l227_227336


namespace final_milk_concentration_l227_227529

theorem final_milk_concentration
  (initial_mixture_volume : ℝ)
  (initial_milk_volume : ℝ)
  (replacement_volume : ℝ)
  (replacements_count : ℕ)
  (final_milk_volume : ℝ) :
  initial_mixture_volume = 100 → 
  initial_milk_volume = 36 → 
  replacement_volume = 50 →
  replacements_count = 2 →
  final_milk_volume = 9 →
  (final_milk_volume / initial_mixture_volume * 100) = 9 :=
by
  sorry

end final_milk_concentration_l227_227529


namespace range_of_b_l227_227038

theorem range_of_b (b : ℝ) (h : Real.sqrt ((b-2)^2) = 2 - b) : b ≤ 2 :=
by {
  sorry
}

end range_of_b_l227_227038


namespace find_a5_l227_227718

open Nat

def increasing_seq (a : Nat → Nat) : Prop :=
  ∀ m n : Nat, m < n → a m < a n

theorem find_a5
  (a : Nat → Nat)
  (h1 : ∀ n : Nat, a (a n) = 3 * n)
  (h2 : increasing_seq a)
  (h3 : ∀ n : Nat, a n > 0) :
  a 5 = 8 :=
by
  sorry

end find_a5_l227_227718


namespace number_of_grandchildren_l227_227187

-- Definitions based on the conditions
def cards_per_grandkid := 2
def money_per_card := 80
def total_money_given_away := 480

-- Calculation of money each grandkid receives per year
def money_per_grandkid := cards_per_grandkid * money_per_card

-- The theorem we want to prove
theorem number_of_grandchildren :
  (total_money_given_away / money_per_grandkid) = 3 :=
by
  -- Placeholder for the proof
  sorry 

end number_of_grandchildren_l227_227187


namespace ratio_second_to_first_l227_227713

-- Define the given conditions and variables
variables 
  (total_water : ℕ := 1200)
  (neighborhood1_usage : ℕ := 150)
  (neighborhood4_usage : ℕ := 350)
  (x : ℕ) -- water usage by second neighborhood

-- Define the usage by third neighborhood in terms of the second neighborhood usage
def neighborhood3_usage := x + 100

-- Define remaining water usage after substracting neighborhood 4 usage
def remaining_water := total_water - neighborhood4_usage

-- The sum of water used by neighborhoods
def total_usage_neighborhoods := neighborhood1_usage + neighborhood3_usage x + x

theorem ratio_second_to_first (h : total_usage_neighborhoods x = remaining_water) :
  (x : ℚ) / neighborhood1_usage = 2 := 
by
  sorry

end ratio_second_to_first_l227_227713


namespace add_in_base14_l227_227344

-- Define symbols A, B, C, D in base 10 as they are used in the base 14 representation
def base14_A : ℕ := 10
def base14_B : ℕ := 11
def base14_C : ℕ := 12
def base14_D : ℕ := 13

-- Define the numbers given in base 14
def num1_base14 : ℕ := 9 * 14^2 + base14_C * 14 + 7
def num2_base14 : ℕ := 4 * 14^2 + base14_B * 14 + 3

-- Define the expected result in base 14
def result_base14 : ℕ := 1 * 14^2 + 0 * 14 + base14_A

-- The theorem statement that needs to be proven
theorem add_in_base14 : num1_base14 + num2_base14 = result_base14 := by
  sorry

end add_in_base14_l227_227344


namespace sum_of_solutions_eq_zero_l227_227022

theorem sum_of_solutions_eq_zero :
  let f (x : ℝ) := 2^|x| + 4 * |x|
  (∀ x : ℝ, f x = 20) →
  (∃ x₁ x₂ : ℝ, f x₁ = 20 ∧ f x₂ = 20 ∧ x₁ + x₂ = 0) :=
sorry

end sum_of_solutions_eq_zero_l227_227022


namespace students_present_each_day_l227_227080
open BigOperators

namespace Absenteeism

def absenteeism_rate : ℕ → ℝ 
| 0 => 14
| n+1 => absenteeism_rate n + 2

def present_rate (n : ℕ) : ℝ := 100 - absenteeism_rate n

theorem students_present_each_day :
  present_rate 0 = 86 ∧
  present_rate 1 = 84 ∧
  present_rate 2 = 82 ∧
  present_rate 3 = 80 ∧
  present_rate 4 = 78 := 
by
  -- Placeholder for the proof steps
  sorry

end Absenteeism

end students_present_each_day_l227_227080


namespace three_minus_pi_to_zero_l227_227840

theorem three_minus_pi_to_zero : (3 - Real.pi) ^ 0 = 1 := by
  -- proof goes here
  sorry

end three_minus_pi_to_zero_l227_227840


namespace sum_powers_l227_227253

theorem sum_powers :
  ∃ (α β γ : ℂ), α + β + γ = 2 ∧ α^2 + β^2 + γ^2 = 5 ∧ α^3 + β^3 + γ^3 = 8 ∧ α^5 + β^5 + γ^5 = 46.5 :=
by
  sorry

end sum_powers_l227_227253


namespace max_n_possible_l227_227032

theorem max_n_possible (k : ℕ) (h_k : k > 1) : ∃ n : ℕ, n = k - 1 :=
by
  sorry

end max_n_possible_l227_227032


namespace geometric_sequence_third_sixth_term_l227_227841

theorem geometric_sequence_third_sixth_term (a r : ℝ) 
  (h3 : a * r^2 = 18) 
  (h6 : a * r^5 = 162) : 
  a = 2 ∧ r = 3 := 
sorry

end geometric_sequence_third_sixth_term_l227_227841


namespace first_term_geometric_sequence_l227_227282

theorem first_term_geometric_sequence (a r : ℚ) 
    (h1 : a * r^2 = 8) 
    (h2 : a * r^4 = 27 / 4) : 
    a = 256 / 27 :=
by sorry

end first_term_geometric_sequence_l227_227282


namespace largest_constant_C_l227_227274

theorem largest_constant_C :
  ∃ C, C = 2 / Real.sqrt 3 ∧ ∀ (x y z : ℝ), x^2 + y^2 + z^2 + 1 ≥ C * (x + y + z) := sorry

end largest_constant_C_l227_227274


namespace roots_difference_squared_l227_227628

theorem roots_difference_squared
  {Φ ϕ : ℝ}
  (hΦ : Φ^2 - Φ - 2 = 0)
  (hϕ : ϕ^2 - ϕ - 2 = 0)
  (h_diff : Φ ≠ ϕ) :
  (Φ - ϕ)^2 = 9 :=
by sorry

end roots_difference_squared_l227_227628


namespace quadratic_other_root_l227_227619

theorem quadratic_other_root (k : ℝ) (h : ∀ x, x^2 - k*x - 4 = 0 → x = 2 ∨ x = -2) :
  ∀ x, x^2 - k*x - 4 = 0 → x = -2 :=
by
  sorry

end quadratic_other_root_l227_227619


namespace isosceles_triangle_if_perpendiculars_intersect_at_single_point_l227_227057

noncomputable def is_isosceles_triangle (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ c = a

theorem isosceles_triangle_if_perpendiculars_intersect_at_single_point
  (a b c : ℝ)
  (D E F P Q R H : Type)
  (intersection_point: P = Q ∧ Q = R ∧ P = R ∧ P = H) :
  is_isosceles_triangle a b c := 
sorry

end isosceles_triangle_if_perpendiculars_intersect_at_single_point_l227_227057


namespace find_smaller_number_l227_227892

theorem find_smaller_number (x : ℕ) (hx : x + 4 * x = 45) : x = 9 :=
by
  sorry

end find_smaller_number_l227_227892


namespace gear_angular_speed_proportion_l227_227195

theorem gear_angular_speed_proportion :
  ∀ (ω_A ω_B ω_C ω_D k: ℝ),
    30 * ω_A = k →
    45 * ω_B = k →
    50 * ω_C = k →
    60 * ω_D = k →
    ω_A / ω_B = 1 ∧
    ω_B / ω_C = 45 / 50 ∧
    ω_C / ω_D = 50 / 60 ∧
    ω_A / ω_D = 10 / 7.5 :=
  by
    -- proof goes here
    sorry

end gear_angular_speed_proportion_l227_227195


namespace average_weight_of_children_l227_227378

theorem average_weight_of_children
  (S_B S_G : ℕ)
  (avg_boys_weight : S_B = 8 * 160)
  (avg_girls_weight : S_G = 5 * 110) :
  (S_B + S_G) / 13 = 141 := 
by
  sorry

end average_weight_of_children_l227_227378


namespace proportion_correct_l227_227819

theorem proportion_correct (x y : ℝ) (h : 5 * y = 4 * x) : x / y = 5 / 4 :=
sorry

end proportion_correct_l227_227819


namespace ratio_of_fruit_salads_l227_227061

theorem ratio_of_fruit_salads 
  (salads_Alaya : ℕ) 
  (total_salads : ℕ) 
  (h1 : salads_Alaya = 200) 
  (h2 : total_salads = 600) : 
  (total_salads - salads_Alaya) / salads_Alaya = 2 :=
by 
  sorry

end ratio_of_fruit_salads_l227_227061


namespace OHaraTriple_example_l227_227000

def OHaraTriple (a b x : ℕ) : Prop :=
  (Nat.sqrt a + Nat.sqrt b = x)

theorem OHaraTriple_example : OHaraTriple 49 64 15 :=
by
  sorry

end OHaraTriple_example_l227_227000


namespace log_ab_a2_plus_log_ab_b2_eq_2_l227_227488

theorem log_ab_a2_plus_log_ab_b2_eq_2 (a b : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (h_distinct : a ≠ b) (h_a_gt_2 : a > 2) (h_b_gt_2 : b > 2) :
  Real.log (a^2) / Real.log (a * b) + Real.log (b^2) / Real.log (a * b) = 2 :=
by
  sorry

end log_ab_a2_plus_log_ab_b2_eq_2_l227_227488


namespace pencils_remaining_l227_227975

variable (initial_pencils : ℝ) (pencils_given : ℝ)

theorem pencils_remaining (h1 : initial_pencils = 56.0) 
                          (h2 : pencils_given = 9.5) 
                          : initial_pencils - pencils_given = 46.5 :=
by 
  sorry

end pencils_remaining_l227_227975


namespace trapezoid_fraction_l227_227005

theorem trapezoid_fraction 
  (shorter_base longer_base side_length : ℝ)
  (angle_adjacent : ℝ)
  (h1 : shorter_base = 120)
  (h2 : longer_base = 180)
  (h3 : side_length = 130)
  (h4 : angle_adjacent = 60) :
  ∃ fraction : ℝ, fraction = 1 / 2 :=
by
  sorry

end trapezoid_fraction_l227_227005


namespace find_two_digit_number_l227_227291

open Nat

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem find_two_digit_number :
  ∃ N : ℕ, 
    (10 ≤ N ∧ N < 100) ∧ 
    (N % 2 = 1) ∧ 
    (N % 9 = 0) ∧ 
    is_perfect_square ((N / 10) * (N % 10)) ∧ 
    N = 99 :=
by
  sorry

end find_two_digit_number_l227_227291


namespace profit_percentage_l227_227602

theorem profit_percentage (CP SP : ℝ) (hCP : CP = 500) (hSP : SP = 625) : 
  ((SP - CP) / CP) * 100 = 25 := 
by 
  sorry

end profit_percentage_l227_227602


namespace drawings_on_last_page_l227_227963

theorem drawings_on_last_page :
  let n_notebooks := 10 
  let p_pages := 50
  let d_original := 5
  let d_new := 8
  let total_drawings := n_notebooks * p_pages * d_original
  let total_pages_new := total_drawings / d_new
  let filled_complete_pages := 6 * p_pages
  let drawings_on_last_page := total_drawings - filled_complete_pages * d_new - 40 * d_new
  drawings_on_last_page == 4 :=
  sorry

end drawings_on_last_page_l227_227963


namespace find_years_l227_227748

variable (p m x : ℕ)

def two_years_ago := p - 2 = 2 * (m - 2)
def four_years_ago := p - 4 = 3 * (m - 4)
def ratio_in_x_years (x : ℕ) := (p + x) * 2 = (m + x) * 3

theorem find_years (h1 : two_years_ago p m) (h2 : four_years_ago p m) : ratio_in_x_years p m 2 :=
by
  sorry

end find_years_l227_227748


namespace tom_paid_1145_l227_227044

-- Define the quantities
def quantity_apples : ℕ := 8
def rate_apples : ℕ := 70
def quantity_mangoes : ℕ := 9
def rate_mangoes : ℕ := 65

-- Calculate costs
def cost_apples : ℕ := quantity_apples * rate_apples
def cost_mangoes : ℕ := quantity_mangoes * rate_mangoes

-- Calculate the total amount paid
def total_amount_paid : ℕ := cost_apples + cost_mangoes

-- The theorem to prove
theorem tom_paid_1145 :
  total_amount_paid = 1145 :=
by sorry

end tom_paid_1145_l227_227044


namespace roots_of_equation_l227_227212

def operation (a b : ℝ) : ℝ := a^2 * b + a * b - 1

theorem roots_of_equation :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ operation x₁ 1 = 0 ∧ operation x₂ 1 = 0 :=
by
  sorry

end roots_of_equation_l227_227212


namespace solve_inequality_system_l227_227297

theorem solve_inequality_system (x : ℝ) :
  (x + 2 < 3 * x) ∧ ((5 - x) / 2 + 1 < 0) → (x > 7) :=
by
  sorry

end solve_inequality_system_l227_227297


namespace correct_statement_l227_227784

-- Definitions as per conditions
def P1 : Prop := ∃ x : ℝ, x^2 = 64 ∧ abs x ^ 3 = 2
def P2 : Prop := ∀ x : ℝ, x = 0 → (¬∃ y, y * x = 1 ∧ -x = y)
def P3 : Prop := ∀ x y : ℝ, x + y = 0 → abs x / abs y = -1
def P4 : Prop := ∀ x a : ℝ, abs x + x = a → a > 0

-- The proof problem
theorem correct_statement : P1 ∧ ¬P2 ∧ ¬P3 ∧ ¬P4 := by
  sorry

end correct_statement_l227_227784


namespace value_of_f_at_1_over_16_l227_227462

noncomputable def f (x : ℝ) (α : ℝ) := x ^ α

theorem value_of_f_at_1_over_16 (α : ℝ) (h : f 4 α = 2) : f (1 / 16) α = 1 / 4 :=
by
  sorry

end value_of_f_at_1_over_16_l227_227462


namespace proportional_parts_middle_l227_227611

theorem proportional_parts_middle (x : ℚ) (hx : x + (1/2) * x + (1/4) * x = 120) : (1/2) * x = 240 / 7 :=
by
  sorry

end proportional_parts_middle_l227_227611


namespace discriminant_zero_no_harmonic_progression_l227_227454

theorem discriminant_zero_no_harmonic_progression (a b c : ℝ) 
    (h_disc : b^2 = 24 * a * c) : 
    ¬ (2 * (1 / b) = (1 / a) + (1 / c)) := 
sorry

end discriminant_zero_no_harmonic_progression_l227_227454


namespace tree_difference_l227_227148

-- Given constants
def Hassans_apple_trees : Nat := 1
def Hassans_orange_trees : Nat := 2

def Ahmeds_orange_trees : Nat := 8
def Ahmeds_apple_trees : Nat := 4 * Hassans_apple_trees

-- Total trees computations
def Ahmeds_total_trees : Nat := Ahmeds_apple_trees + Ahmeds_orange_trees
def Hassans_total_trees : Nat := Hassans_apple_trees + Hassans_orange_trees

-- Theorem to prove the difference in total trees
theorem tree_difference : Ahmeds_total_trees - Hassans_total_trees = 9 := by
  sorry

end tree_difference_l227_227148


namespace correct_system_of_equations_l227_227495

theorem correct_system_of_equations (x y : ℕ) :
  (7 * x + 7 = y) ∧ (9 * (x - 1) = y) ↔ 
  (y = 7 * x + 7) ∧ (y = 9 * (x - 1)) :=
by
  sorry

end correct_system_of_equations_l227_227495


namespace geometric_progression_arcsin_sin_l227_227285

noncomputable def least_positive_t : ℝ :=
  9 + 4 * Real.sqrt 5

theorem geometric_progression_arcsin_sin 
  (α : ℝ) 
  (hα1: 0 < α) 
  (hα2: α < Real.pi / 2) 
  (t : ℝ) 
  (h : ∀ (a b c d : ℝ), 
    a = Real.arcsin (Real.sin α) ∧ 
    b = Real.arcsin (Real.sin (3 * α)) ∧ 
    c = Real.arcsin (Real.sin (5 * α)) ∧ 
    d = Real.arcsin (Real.sin (t * α)) → 
    b / a = c / b ∧ c / b = d / c) : 
  t = least_positive_t :=
sorry

end geometric_progression_arcsin_sin_l227_227285


namespace solution_set_quadratic_inequality_l227_227958

theorem solution_set_quadratic_inequality :
  { x : ℝ | x^2 + 3 * x - 4 < 0 } = { x : ℝ | -4 < x ∧ x < 1 } :=
sorry

end solution_set_quadratic_inequality_l227_227958


namespace lesser_fraction_l227_227385

theorem lesser_fraction (x y : ℚ) (h1 : x + y = 13 / 14) (h2 : x * y = 1 / 8) : min x y = 163 / 625 :=
by sorry

end lesser_fraction_l227_227385


namespace regular_price_of_one_tire_l227_227113

theorem regular_price_of_one_tire
  (x : ℝ) -- Define the variable \( x \) as the regular price of one tire
  (h1 : 3 * x + 10 = 250) -- Set up the equation based on the condition

  : x = 80 := 
sorry

end regular_price_of_one_tire_l227_227113


namespace train_speed_is_64_98_kmph_l227_227284

noncomputable def train_length : ℝ := 200
noncomputable def bridge_length : ℝ := 180
noncomputable def passing_time : ℝ := 21.04615384615385
noncomputable def speed_in_kmph : ℝ := 3.6 * (train_length + bridge_length) / passing_time

theorem train_speed_is_64_98_kmph : abs (speed_in_kmph - 64.98) < 0.01 :=
by
  sorry

end train_speed_is_64_98_kmph_l227_227284


namespace x_value_unique_l227_227609

theorem x_value_unique (x : ℝ) (h : ∀ y : ℝ, 10 * x * y - 15 * y + 5 * x - 7 = 0) :
  x = 3 / 2 :=
sorry

end x_value_unique_l227_227609


namespace find_g_inv_neg_fifteen_sixtyfour_l227_227884

noncomputable def g (x : ℝ) : ℝ := (x^6 - 1) / 4

theorem find_g_inv_neg_fifteen_sixtyfour : g⁻¹ (-15/64) = 1/2 :=
by
  sorry  -- Proof is not required

end find_g_inv_neg_fifteen_sixtyfour_l227_227884


namespace length_of_BC_l227_227695

noncomputable def perimeter (a b c : ℝ) := a + b + c
noncomputable def area (b c : ℝ) (A : ℝ) := 0.5 * b * c * (Real.sin A)

theorem length_of_BC
  (a b c : ℝ)
  (h_perimeter : perimeter a b c = 20)
  (h_area : area b c (Real.pi / 3) = 10 * Real.sqrt 3) :
  a = 7 :=
by
  sorry

end length_of_BC_l227_227695


namespace elyse_passing_threshold_l227_227006

def total_questions : ℕ := 90
def programming_questions : ℕ := 20
def database_questions : ℕ := 35
def networking_questions : ℕ := 35
def programming_correct_rate : ℝ := 0.8
def database_correct_rate : ℝ := 0.5
def networking_correct_rate : ℝ := 0.7
def passing_percentage : ℝ := 0.65

theorem elyse_passing_threshold :
  let programming_correct := programming_correct_rate * programming_questions
  let database_correct := database_correct_rate * database_questions
  let networking_correct := networking_correct_rate * networking_questions
  let total_correct := programming_correct + database_correct + networking_correct
  let required_to_pass := passing_percentage * total_questions
  total_correct = required_to_pass → 0 = 0 :=
by
  intro _h
  sorry

end elyse_passing_threshold_l227_227006


namespace shaded_cells_product_l227_227977

def product_eq (a b c : ℕ) (p : ℕ) : Prop := a * b * c = p

theorem shaded_cells_product :
  ∃ (a₁₁ a₁₂ a₁₃ a₂₁ a₂₂ a₂₃ a₃₁ a₃₂ a₃₃ : ℕ),
    product_eq a₁₁ a₁₂ a₁₃ 12 ∧
    product_eq a₂₁ a₂₂ a₂₃ 112 ∧
    product_eq a₃₁ a₃₂ a₃₃ 216 ∧
    product_eq a₁₁ a₂₁ a₃₁ 12 ∧
    product_eq a₁₂ a₂₂ a₃₂ 12 ∧
    (a₁₁ * a₂₂ * a₃₃ = 3 * 2 * 5) :=
sorry

end shaded_cells_product_l227_227977


namespace dealership_sales_l227_227690

theorem dealership_sales :
  (∀ (n : ℕ), 3 * n ≤ 36 → 5 * n ≤ x) →
  (36 / 3) * 5 = 60 :=
by
  sorry

end dealership_sales_l227_227690


namespace infinite_points_inside_circle_l227_227778

theorem infinite_points_inside_circle:
  ∀ c : ℝ, c = 3 → ∀ x y : ℚ, 0 < x ∧ 0 < y  ∧ x^2 + y^2 < 9 → ∃ a b : ℚ, 0 < a ∧ 0 < b ∧ a^2 + b^2 < 9 :=
sorry

end infinite_points_inside_circle_l227_227778


namespace number_of_boys_l227_227679

theorem number_of_boys (total_students girls : ℕ) (h1 : total_students = 13) (h2 : girls = 6) :
  total_students - girls = 7 :=
by 
  -- We'll skip the proof as instructed
  sorry

end number_of_boys_l227_227679


namespace scientific_notation_of_448000_l227_227598

theorem scientific_notation_of_448000 :
  448000 = 4.48 * 10^5 :=
by 
  sorry

end scientific_notation_of_448000_l227_227598


namespace simon_project_score_l227_227480

-- Define the initial conditions
def num_students_before : Nat := 20
def num_students_total : Nat := 21
def avg_before : ℕ := 86
def avg_after : ℕ := 88

-- Calculate total score before Simon's addition
def total_score_before : ℕ := num_students_before * avg_before

-- Calculate total score after Simon's addition
def total_score_after : ℕ := num_students_total * avg_after

-- Definition to represent Simon's score
def simon_score : ℕ := total_score_after - total_score_before

-- Theorem that we need to prove
theorem simon_project_score : simon_score = 128 :=
by
  sorry

end simon_project_score_l227_227480


namespace problem_1_solution_set_problem_2_range_of_a_l227_227949

-- Define the function f(x)
def f (x a : ℝ) := |2 * x - a| + |x - 1|

-- Problem 1: Solution set of the inequality f(x) ≥ 2 when a = 3
theorem problem_1_solution_set :
  { x : ℝ | f x 3 ≥ 2 } = { x : ℝ | x ≤ 2/3 ∨ x ≥ 2 } :=
sorry

-- Problem 2: Range of a such that f(x) ≥ 5 - x for all x ∈ ℝ
theorem problem_2_range_of_a (a : ℝ) :
  (∀ x : ℝ, f x a ≥ 5 - x) ↔ a ≥ 6 :=
sorry

end problem_1_solution_set_problem_2_range_of_a_l227_227949


namespace product_probability_probability_one_l227_227489

def S : Set Int := {13, 57}

theorem product_probability (a b : Int) (h₁ : a ∈ S) (h₂ : b ∈ S) (h₃ : a ≠ b) : 
  (a * b > 15) := 
by 
  sorry

theorem probability_one : 
  (∃ a b : Int, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ a * b > 15) ∧ 
  (∀ a b : Int, a ∈ S ∧ b ∈ S ∧ a ≠ b → a * b > 15) :=
by 
  sorry

end product_probability_probability_one_l227_227489


namespace first_chapter_length_l227_227927

theorem first_chapter_length (total_pages : ℕ) (second_chapter_pages : ℕ) (third_chapter_pages : ℕ)
  (h : total_pages = 125) (h2 : second_chapter_pages = 35) (h3 : third_chapter_pages  = 24) :
  total_pages - second_chapter_pages - third_chapter_pages = 66 :=
by
  -- Construct the proof using the provided conditions
  sorry

end first_chapter_length_l227_227927


namespace triangle_area_l227_227031

-- Define a triangle as a structure with vertices A, B, and C, where the lengths AB, AC, and BC are provided
structure Triangle :=
  (A B C : ℝ)
  (AB AC BC : ℝ)
  (is_isosceles : AB = AC)
  (BC_length : BC = 20)
  (AB_length : AB = 26)

-- Define the length bisector and Pythagorean properties
def bisects_base (t : Triangle) : Prop :=
  ∃ D : ℝ, (t.B - D) = (D - t.C) ∧ 2 * D = t.B + t.C

def pythagorean_theorem_AD (t : Triangle) (D : ℝ) (AD : ℝ) : Prop :=
  t.AB^2 = AD^2 + (t.B - D)^2

-- State the problem as a theorem
theorem triangle_area (t : Triangle) (D : ℝ) (AD : ℝ) (h1 : bisects_base t) (h2 : pythagorean_theorem_AD t D AD) :
  AD = 24 ∧ (1 / 2) * t.BC * AD = 240 :=
sorry

end triangle_area_l227_227031


namespace sum_of_coordinates_l227_227577

-- Definitions of points and their coordinates
def pointC (x : ℝ) : ℝ × ℝ := (x, 8)
def pointD (x : ℝ) : ℝ × ℝ := (x, -8)

-- The goal is to prove that the sum of the four coordinate values of points C and D is 2x
theorem sum_of_coordinates (x : ℝ) :
  (pointC x).1 + (pointC x).2 + (pointD x).1 + (pointD x).2 = 2 * x :=
by
  sorry

end sum_of_coordinates_l227_227577


namespace ln_1_2_over_6_gt_e_l227_227457

theorem ln_1_2_over_6_gt_e :
  let x := 1.2
  let exp1 := x^6
  let exp2 := (1.44)^2 * 1.44
  let final_val := 2.0736 * 1.44
  final_val > 2.718 :=
by {
  sorry
}

end ln_1_2_over_6_gt_e_l227_227457


namespace identify_roles_l227_227382

-- Define the number of liars and truth-tellers
def num_liars : Nat := 1000
def num_truth_tellers : Nat := 1000

-- Define the properties of the individuals
def first_person_is_liar := true
def second_person_is_truth_teller := true

-- The main statement equivalent to the problem
theorem identify_roles : first_person_is_liar = true ∧ second_person_is_truth_teller = true := by
  sorry

end identify_roles_l227_227382


namespace max_sum_pyramid_on_hexagonal_face_l227_227306

structure hexagonal_prism :=
(faces_initial : ℕ)
(vertices_initial : ℕ)
(edges_initial : ℕ)

structure pyramid_added :=
(faces_total : ℕ)
(vertices_total : ℕ)
(edges_total : ℕ)
(total_sum : ℕ)

theorem max_sum_pyramid_on_hexagonal_face (h : hexagonal_prism) :
  (h = ⟨8, 12, 18⟩) →
  ∃ p : pyramid_added, 
    p = ⟨13, 13, 24, 50⟩ :=
by
  sorry

end max_sum_pyramid_on_hexagonal_face_l227_227306


namespace find_f_neg3_l227_227827

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom periodic_function : ∀ x : ℝ, f (x + 4) = f x
axiom sum_equation : f 1 + f 2 + f 3 + f 4 + f 5 = 6

theorem find_f_neg3 : f (-3) = 6 := by
  sorry

end find_f_neg3_l227_227827


namespace rick_iron_clothing_l227_227861

theorem rick_iron_clothing :
  let shirts_per_hour := 4
  let pants_per_hour := 3
  let jackets_per_hour := 2
  let hours_shirts := 3
  let hours_pants := 5
  let hours_jackets := 2
  let total_clothing := (shirts_per_hour * hours_shirts) + (pants_per_hour * hours_pants) + (jackets_per_hour * hours_jackets)
  total_clothing = 31 := by
  sorry

end rick_iron_clothing_l227_227861


namespace find_three_tuple_solutions_l227_227154

open Real

theorem find_three_tuple_solutions :
  (x y z : ℝ) → (x^2 + y^2 + 25 * z^2 = 6 * x * z + 8 * y * z)
  → (3 * x^2 + 2 * y^2 + z^2 = 240)
  → (x = 6 ∧ y = 8 ∧ z = 2) ∨ (x = -6 ∧ y = -8 ∧ z = -2) :=
by
  intro x y z
  intro h1 h2
  sorry

end find_three_tuple_solutions_l227_227154
