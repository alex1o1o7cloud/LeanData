import Mathlib

namespace NUMINAMATH_GPT_problem_1_problem_2_l1509_150982

-- Define the sets M and N as conditions and include a > 0 condition.
def M (a : ℝ) : Set ℝ := {x : ℝ | (x + a) * (x - 1) ≤ 0}
def N : Set ℝ := {x : ℝ | 4 * x ^ 2 - 4 * x - 3 < 0}

-- Problem 1: Prove that a = 2 given the set conditions.
theorem problem_1 (a : ℝ) (h_pos : a > 0) :
  M a ∪ N = {x : ℝ | -2 ≤ x ∧ x < 3 / 2} → a = 2 :=
sorry

-- Problem 2: Prove the range of a is 0 < a ≤ 1 / 2 given the set conditions.
theorem problem_2 (a : ℝ) (h_pos : a > 0) :
  N ∪ (compl (M a)) = Set.univ → 0 < a ∧ a ≤ 1 / 2 :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1509_150982


namespace NUMINAMATH_GPT_distance_between_cities_l1509_150972

noncomputable def speed_a : ℝ := 1 / 10
noncomputable def speed_b : ℝ := 1 / 15
noncomputable def time_to_meet : ℝ := 6
noncomputable def distance_diff : ℝ := 12

theorem distance_between_cities : 
  (time_to_meet * (speed_a + speed_b) = 60) →
  time_to_meet * speed_a - time_to_meet * speed_b = distance_diff →
  time_to_meet * (speed_a + speed_b) = 60 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_distance_between_cities_l1509_150972


namespace NUMINAMATH_GPT_well_depth_l1509_150949

theorem well_depth (e x a b c d : ℝ)
  (h1 : x = 2 * a + b)
  (h2 : x = 3 * b + c)
  (h3 : x = 4 * c + d)
  (h4 : x = 5 * d + e)
  (h5 : x = 6 * e + a) :
  x = 721 / 76 * e ∧
  a = 265 / 76 * e ∧
  b = 191 / 76 * e ∧
  c = 37 / 19 * e ∧
  d = 129 / 76 * e :=
sorry

end NUMINAMATH_GPT_well_depth_l1509_150949


namespace NUMINAMATH_GPT_remainder_of_largest_divided_by_second_smallest_l1509_150948

theorem remainder_of_largest_divided_by_second_smallest 
  (a b c : ℕ) (h1 : a = 10) (h2 : b = 11) (h3 : c = 12) :
  c % b = 1 :=
by
  -- We assume and/or prove the necessary statements here.
  -- The core of the proof uses existing facts or assumptions.
  -- We insert the proof strategy or intermediate steps here.
  
  sorry

end NUMINAMATH_GPT_remainder_of_largest_divided_by_second_smallest_l1509_150948


namespace NUMINAMATH_GPT_intersection_sets_l1509_150957

open Set

def A := {x : ℤ | abs x < 3}
def B := {x : ℤ | abs x > 1}

theorem intersection_sets :
  A ∩ B = {-2, 2} := by
  sorry

end NUMINAMATH_GPT_intersection_sets_l1509_150957


namespace NUMINAMATH_GPT_calculate_value_l1509_150959

-- Definition of the given values
def val1 : ℕ := 444
def val2 : ℕ := 44
def val3 : ℕ := 4

-- Theorem statement proving the value of the expression
theorem calculate_value : (val1 - val2 - val3) = 396 := 
by 
  sorry

end NUMINAMATH_GPT_calculate_value_l1509_150959


namespace NUMINAMATH_GPT_total_games_to_determine_winner_l1509_150922

-- Conditions: Initial number of teams in the preliminary round
def initial_teams : ℕ := 24

-- Condition: Preliminary round eliminates 50% of the teams
def preliminary_round_elimination (n : ℕ) : ℕ := n / 2

-- Function to compute the required games for any single elimination tournament
def single_elimination_games (teams : ℕ) : ℕ :=
  if teams = 0 then 0
  else teams - 1

-- Proof Statement: Total number of games to determine the winner
theorem total_games_to_determine_winner (n : ℕ) (h : n = 24) :
  preliminary_round_elimination n + single_elimination_games (preliminary_round_elimination n) = 23 :=
by
  sorry

end NUMINAMATH_GPT_total_games_to_determine_winner_l1509_150922


namespace NUMINAMATH_GPT_complex_root_product_value_l1509_150943

noncomputable def complex_root_product (r : ℂ) (h1 : r^7 = 1) (h2 : r ≠ 1) : ℂ :=
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) * (r^6 - 1)

theorem complex_root_product_value (r : ℂ) (h1 : r^7 = 1) (h2 : r ≠ 1) : complex_root_product r h1 h2 = 14 := 
  sorry

end NUMINAMATH_GPT_complex_root_product_value_l1509_150943


namespace NUMINAMATH_GPT_correct_operation_l1509_150937

theorem correct_operation :
  (3 * a^3 - 2 * a^3 = a^3) ∧ ¬(m - 4 * m = -3) ∧ ¬(a^2 * b - a * b^2 = 0) ∧ ¬(2 * x + 3 * x = 5 * x^2) :=
by
  sorry

end NUMINAMATH_GPT_correct_operation_l1509_150937


namespace NUMINAMATH_GPT_johns_contribution_l1509_150952

theorem johns_contribution (A : ℝ) (J : ℝ) : 
  (1.7 * A = 85) ∧ ((5 * A + J) / 6 = 85) → J = 260 := 
by
  sorry

end NUMINAMATH_GPT_johns_contribution_l1509_150952


namespace NUMINAMATH_GPT_find_number_eq_l1509_150966

theorem find_number_eq (x : ℝ) (h : (35 / 100) * x = (20 / 100) * 40) : x = 160 / 7 :=
by
  sorry

end NUMINAMATH_GPT_find_number_eq_l1509_150966


namespace NUMINAMATH_GPT_rahuls_share_l1509_150933

theorem rahuls_share (total_payment : ℝ) (rahul_days : ℝ) (rajesh_days : ℝ) (rahul_share : ℝ)
  (rahul_work_one_day : rahul_days > 0) (rajesh_work_one_day : rajesh_days > 0)
  (total_payment_eq : total_payment = 105) 
  (rahul_days_eq : rahul_days = 3) 
  (rajesh_days_eq : rajesh_days = 2) :
  rahul_share = 42 := 
by
  sorry

end NUMINAMATH_GPT_rahuls_share_l1509_150933


namespace NUMINAMATH_GPT_tan_add_pi_over_4_l1509_150927

variable {α : ℝ}

theorem tan_add_pi_over_4 (h : Real.tan (α - Real.pi / 4) = 1 / 4) : Real.tan (α + Real.pi / 4) = -4 :=
sorry

end NUMINAMATH_GPT_tan_add_pi_over_4_l1509_150927


namespace NUMINAMATH_GPT_parabola_line_intersect_solutions_count_l1509_150953

theorem parabola_line_intersect_solutions_count :
  ∃ b1 b2 : ℝ, (b1 ≠ b2 ∧ (b1^2 - b1 - 3 = 0) ∧ (b2^2 - b2 - 3 = 0)) :=
by
  sorry

end NUMINAMATH_GPT_parabola_line_intersect_solutions_count_l1509_150953


namespace NUMINAMATH_GPT_max_perimeter_convex_quadrilateral_l1509_150955

theorem max_perimeter_convex_quadrilateral :
  ∃ (AB BC AD CD AC BD : ℝ), 
    AB = 1 ∧ BC = 1 ∧
    AD ≤ 1 ∧ CD ≤ 1 ∧ AC ≤ 1 ∧ BD ≤ 1 ∧
    2 + 4 * Real.sin (Real.pi / 12) = 
      AB + BC + AD + CD :=
sorry

end NUMINAMATH_GPT_max_perimeter_convex_quadrilateral_l1509_150955


namespace NUMINAMATH_GPT_gratuity_is_four_l1509_150930

-- Define the prices and tip percentage (conditions)
def a : ℕ := 10
def b : ℕ := 13
def c : ℕ := 17
def p : ℚ := 0.1

-- Define the total bill and gratuity based on the given definitions
def total_bill : ℕ := a + b + c
def gratuity : ℚ := total_bill * p

-- Theorem (proof problem): Prove that the gratuity is $4
theorem gratuity_is_four : gratuity = 4 := by
  sorry

end NUMINAMATH_GPT_gratuity_is_four_l1509_150930


namespace NUMINAMATH_GPT_steps_to_get_down_empire_state_building_l1509_150971

theorem steps_to_get_down_empire_state_building (total_steps : ℕ) (steps_building_to_garden : ℕ) (steps_to_madison_square : ℕ) :
  total_steps = 991 -> steps_building_to_garden = 315 -> steps_to_madison_square = total_steps - steps_building_to_garden -> steps_to_madison_square = 676 :=
by
  intros
  subst_vars
  sorry

end NUMINAMATH_GPT_steps_to_get_down_empire_state_building_l1509_150971


namespace NUMINAMATH_GPT_b_contribution_l1509_150942

/-- A starts business with Rs. 3500.
    After 9 months, B joins as a partner.
    After a year, the profit is divided in the ratio 2:3.
    Prove that B's contribution to the capital is Rs. 21000. -/
theorem b_contribution (a_capital : ℕ) (months_a : ℕ) (b_time : ℕ) (profit_ratio_num : ℕ) (profit_ratio_den : ℕ)
  (h_a_capital : a_capital = 3500)
  (h_months_a : months_a = 12)
  (h_b_time : b_time = 3)
  (h_profit_ratio : profit_ratio_num = 2 ∧ profit_ratio_den = 3) :
  (21000 * b_time * profit_ratio_num) / (3 * profit_ratio_den) = 3500 * months_a := by
  sorry

end NUMINAMATH_GPT_b_contribution_l1509_150942


namespace NUMINAMATH_GPT_ratio_a_to_c_l1509_150920

-- Define the variables and ratios
variables (x y z a b c d : ℝ)

-- Define the conditions as given ratios
variables (h1 : a / b = 2 * x / (3 * y))
variables (h2 : b / c = z / (5 * z))
variables (h3 : a / d = 4 * x / (7 * y))
variables (h4 : d / c = 7 * y / (3 * z))

-- Statement to prove the ratio of a to c
theorem ratio_a_to_c (x y z a b c d : ℝ) 
  (h1 : a / b = 2 * x / (3 * y)) 
  (h2 : b / c = z / (5 * z)) 
  (h3 : a / d = 4 * x / (7 * y)) 
  (h4 : d / c = 7 * y / (3 * z)) : a / c = 2 * x / (15 * y) :=
sorry

end NUMINAMATH_GPT_ratio_a_to_c_l1509_150920


namespace NUMINAMATH_GPT_proof_problem_l1509_150989

-- Define the propositions as Lean terms
def prop1 : Prop := ∀ (l1 l2 : ℝ) (h1 : l1 ≠ 0 ∧ l2 ≠ 0), (l1 * l2 = -1) → (l1 ≠ l2)  -- Two perpendicular lines must intersect (incorrect definition)
def prop2 : Prop := ∀ (l : ℝ), ∃! (m : ℝ), (l * m = -1)  -- There is only one perpendicular line (incorrect definition)
def prop3 : Prop := (∀ (α β γ : ℝ), α = β → γ = 90 → α + γ = β + γ)  -- Equal corresponding angles when intersecting a third (incorrect definition)
def prop4 : Prop := ∀ (A B C : ℝ), (A = B ∧ B = C) → (A = C)  -- Transitive property of parallel lines

-- The statement that only one of these propositions is true, and it is the fourth one
theorem proof_problem (h1 : ¬ prop1) (h2 : ¬ prop2) (h3 : ¬ prop3) (h4 : prop4) : 
  ∃! (i : ℕ), i = 4 := 
by
  sorry

end NUMINAMATH_GPT_proof_problem_l1509_150989


namespace NUMINAMATH_GPT_total_amount_proof_l1509_150918

-- Define the relationships between x, y, and z in terms of the amounts received
variables (x y z : ℝ)

-- Given: For each rupee x gets, y gets 0.45 rupees and z gets 0.50 rupees
def relationship1 : Prop := ∀ (k : ℝ), y = 0.45 * k ∧ z = 0.50 * k ∧ x = k

-- Given: The share of y is Rs. 54
def condition1 : Prop := y = 54

-- The total amount x + y + z is Rs. 234
def total_amount (x y z : ℝ) : ℝ := x + y + z

-- Prove that the total amount is Rs. 234
theorem total_amount_proof (x y z : ℝ) (h1: relationship1 x y z) (h2: condition1 y) : total_amount x y z = 234 :=
sorry

end NUMINAMATH_GPT_total_amount_proof_l1509_150918


namespace NUMINAMATH_GPT_min_value_expr_l1509_150992

theorem min_value_expr (x y : ℝ) : ∃ (m : ℝ), (∀ (x y : ℝ), x^2 + x * y + y^2 ≥ m) ∧ m = 0 :=
by
  sorry

end NUMINAMATH_GPT_min_value_expr_l1509_150992


namespace NUMINAMATH_GPT_correlation_comparison_l1509_150998

/-- The data for variables x and y are (1, 3), (2, 5.3), (3, 6.9), (4, 9.1), and (5, 10.8) -/
def xy_data : List (Int × Float) := [(1, 3), (2, 5.3), (3, 6.9), (4, 9.1), (5, 10.8)]

/-- The data for variables U and V are (1, 12.7), (2, 10.2), (3, 7), (4, 3.6), and (5, 1) -/
def UV_data : List (Int × Float) := [(1, 12.7), (2, 10.2), (3, 7), (4, 3.6), (5, 1)]

/-- r1 is the linear correlation coefficient between y and x -/
noncomputable def r1 : Float := sorry

/-- r2 is the linear correlation coefficient between V and U -/
noncomputable def r2 : Float := sorry

/-- The problem is to prove that r2 < 0 < r1 given the data conditions -/
theorem correlation_comparison : r2 < 0 ∧ 0 < r1 := 
by 
  sorry

end NUMINAMATH_GPT_correlation_comparison_l1509_150998


namespace NUMINAMATH_GPT_puppy_weight_l1509_150967

variable (p s l r : ℝ)

theorem puppy_weight :
  p + s + l + r = 40 ∧ 
  p^2 + l^2 = 4 * s ∧ 
  p^2 + s^2 = l^2 → 
  p = Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_puppy_weight_l1509_150967


namespace NUMINAMATH_GPT_hounds_score_points_l1509_150917

theorem hounds_score_points (x y : ℕ) (h_total : x + y = 82) (h_margin : x - y = 18) : y = 32 :=
sorry

end NUMINAMATH_GPT_hounds_score_points_l1509_150917


namespace NUMINAMATH_GPT_cover_square_with_rectangles_l1509_150934

theorem cover_square_with_rectangles :
  ∃ n : ℕ, n = 24 ∧
  ∀ (rect_area : ℕ) (square_area : ℕ), rect_area = 2 * 3 → square_area = 12 * 12 → square_area / rect_area = n :=
by
  use 24
  sorry

end NUMINAMATH_GPT_cover_square_with_rectangles_l1509_150934


namespace NUMINAMATH_GPT_range_AD_dot_BC_l1509_150954

noncomputable def vector_dot_product_range (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 1) : ℝ :=
  let ab := 2
  let ac := 1
  let bc := ac - ab
  let ad := x * ac + (1 - x) * ab
  ad * bc

theorem range_AD_dot_BC : 
  ∃ (a b : ℝ), vector_dot_product_range x h1 h2 = a ∧ ∀ (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 1), a ≤ vector_dot_product_range x h1 h2 ∧ vector_dot_product_range x h1 h2 ≤ b :=
sorry

end NUMINAMATH_GPT_range_AD_dot_BC_l1509_150954


namespace NUMINAMATH_GPT_probability_red_white_green_probability_any_order_l1509_150956

-- Definitions based on the conditions
def total_balls := 28
def red_balls := 15
def white_balls := 9
def green_balls := 4

-- Part (a): Probability of first red, second white, third green
theorem probability_red_white_green : 
  (red_balls / total_balls) * (white_balls / (total_balls - 1)) * (green_balls / (total_balls - 2)) = 5 / 182 :=
by 
  sorry

-- Part (b): Probability of red, white, and green in any order
theorem probability_any_order :
  6 * ((red_balls / total_balls) * (white_balls / (total_balls - 1)) * (green_balls / (total_balls - 2))) = 15 / 91 :=
by
  sorry

end NUMINAMATH_GPT_probability_red_white_green_probability_any_order_l1509_150956


namespace NUMINAMATH_GPT_inscribed_regular_polygon_sides_l1509_150960

theorem inscribed_regular_polygon_sides (n : ℕ) (h_central_angle : 360 / n = 72) : n = 5 :=
by
  sorry

end NUMINAMATH_GPT_inscribed_regular_polygon_sides_l1509_150960


namespace NUMINAMATH_GPT_yolkino_to_palkino_distance_l1509_150911

theorem yolkino_to_palkino_distance 
  (n : ℕ) 
  (digit_sum : ℕ → ℕ) 
  (h1 : ∀ k : ℕ, k ≤ n → digit_sum k + digit_sum (n - k) = 13) : 
  n = 49 := 
by 
  sorry

end NUMINAMATH_GPT_yolkino_to_palkino_distance_l1509_150911


namespace NUMINAMATH_GPT_alice_has_ball_after_two_turns_l1509_150913

def prob_alice_keeps_ball : ℚ := (2/3 * 1/2) + (1/3 * 1/3)

theorem alice_has_ball_after_two_turns :
  prob_alice_keeps_ball = 4 / 9 :=
by
  -- This line is just a placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_alice_has_ball_after_two_turns_l1509_150913


namespace NUMINAMATH_GPT_temple_run_red_coins_l1509_150944

variables (x y z : ℕ)

theorem temple_run_red_coins :
  x + y + z = 2800 →
  x + 3 * y + 5 * z = 7800 →
  z = y + 200 →
  y = 700 := 
by 
  intro h1 h2 h3
  sorry

end NUMINAMATH_GPT_temple_run_red_coins_l1509_150944


namespace NUMINAMATH_GPT_no_such_real_x_exists_l1509_150963

theorem no_such_real_x_exists :
  ¬ ∃ (x : ℝ), ⌊ x ⌋ + ⌊ 2 * x ⌋ + ⌊ 4 * x ⌋ + ⌊ 8 * x ⌋ + ⌊ 16 * x ⌋ + ⌊ 32 * x ⌋ = 12345 := 
sorry

end NUMINAMATH_GPT_no_such_real_x_exists_l1509_150963


namespace NUMINAMATH_GPT_calculate_f_17_69_l1509_150902

noncomputable def f (x y: ℕ) : ℚ := sorry

axiom f_self : ∀ x, f x x = x
axiom f_symm : ∀ x y, f x y = f y x
axiom f_add : ∀ x y, (x + y) * f x y = y * f x (x + y)

theorem calculate_f_17_69 : f 17 69 = 73.3125 := sorry

end NUMINAMATH_GPT_calculate_f_17_69_l1509_150902


namespace NUMINAMATH_GPT_max_cars_with_ac_but_not_rs_l1509_150984

namespace CarProblem

variables (total_cars : ℕ) 
          (cars_without_ac : ℕ)
          (cars_with_rs : ℕ)
          (cars_with_ac : ℕ := total_cars - cars_without_ac)
          (cars_with_ac_and_rs : ℕ)
          (cars_with_ac_but_not_rs : ℕ := cars_with_ac - cars_with_ac_and_rs)

theorem max_cars_with_ac_but_not_rs 
        (h1 : total_cars = 100)
        (h2 : cars_without_ac = 37)
        (h3 : cars_with_rs ≥ 51)
        (h4 : cars_with_ac_and_rs = min cars_with_rs cars_with_ac) :
        cars_with_ac_but_not_rs = 12 := by
    sorry

end CarProblem

end NUMINAMATH_GPT_max_cars_with_ac_but_not_rs_l1509_150984


namespace NUMINAMATH_GPT_day_of_week_dec_26_l1509_150916

theorem day_of_week_dec_26 (nov_26_is_thu : true) : true :=
sorry

end NUMINAMATH_GPT_day_of_week_dec_26_l1509_150916


namespace NUMINAMATH_GPT_proof_problem_l1509_150900

theorem proof_problem (x : ℕ) (h : (x - 4) / 10 = 5) : (x - 5) / 7 = 7 :=
  sorry

end NUMINAMATH_GPT_proof_problem_l1509_150900


namespace NUMINAMATH_GPT_prize_distribution_l1509_150999

theorem prize_distribution (x y z : ℕ) (h₁ : 15000 * x + 10000 * y + 5000 * z = 1000000) (h₂ : 93 ≤ z - x) (h₃ : z - x < 96) :
  x + y + z = 147 :=
sorry

end NUMINAMATH_GPT_prize_distribution_l1509_150999


namespace NUMINAMATH_GPT_num_handshakes_l1509_150926

-- Definition of the conditions
def num_teams : Nat := 4
def women_per_team : Nat := 2
def total_women : Nat := num_teams * women_per_team
def handshakes_per_woman : Nat := total_women -1 - women_per_team

-- Statement of the problem to prove
theorem num_handshakes (h: total_women = 8) : (total_women * handshakes_per_woman) / 2 = 24 := 
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_num_handshakes_l1509_150926


namespace NUMINAMATH_GPT_multiply_72517_9999_l1509_150929

theorem multiply_72517_9999 : 72517 * 9999 = 725097483 :=
by
  sorry

end NUMINAMATH_GPT_multiply_72517_9999_l1509_150929


namespace NUMINAMATH_GPT_total_cost_family_visit_l1509_150935

/-
Conditions:
1. entrance_ticket_cost: $5 per person
2. attraction_ticket_cost_kid: $2 per kid
3. attraction_ticket_cost_parent: $4 per parent
4. family_discount_threshold: A family of 6 or more gets a 10% discount on entrance tickets
5. senior_discount: Senior citizens get a 50% discount on attraction tickets
6. family_composition: 4 children, 2 parents, and 1 grandmother
7. visit_attraction: The family plans to visit at least one attraction
-/

def entrance_ticket_cost : ℝ := 5
def attraction_ticket_cost_kid : ℝ := 2
def attraction_ticket_cost_parent : ℝ := 4
def family_discount_threshold : ℕ := 6
def family_discount_rate : ℝ := 0.10
def senior_discount_rate : ℝ := 0.50
def number_of_kids : ℕ := 4
def number_of_parents : ℕ := 2
def number_of_seniors : ℕ := 1

theorem total_cost_family_visit : 
  let total_entrance_fee := (number_of_kids + number_of_parents + number_of_seniors) * entrance_ticket_cost 
  let total_entrance_fee_discounted := total_entrance_fee * (1 - family_discount_rate)
  let total_attraction_fee_kids := number_of_kids * attraction_ticket_cost_kid
  let total_attraction_fee_parents := number_of_parents * attraction_ticket_cost_parent
  let total_attraction_fee_seniors := number_of_seniors * attraction_ticket_cost_parent * (1 - senior_discount_rate)
  let total_attraction_fee := total_attraction_fee_kids + total_attraction_fee_parents + total_attraction_fee_seniors
  (number_of_kids + number_of_parents + number_of_seniors ≥ family_discount_threshold) → 
  (total_entrance_fee_discounted + total_attraction_fee = 49.50) :=
by
  -- Assuming we calculate entrance fee and attraction fee correctly, state the theorem
  sorry

end NUMINAMATH_GPT_total_cost_family_visit_l1509_150935


namespace NUMINAMATH_GPT_solve_problem_l1509_150985

theorem solve_problem :
  ∃ (x y : ℝ), 7 * x + y = 19 ∧ x + 3 * y = 1 ∧ 2 * x + y = 5 :=
by
  sorry

end NUMINAMATH_GPT_solve_problem_l1509_150985


namespace NUMINAMATH_GPT_combined_gold_cost_l1509_150968

def gary_gold_weight : ℕ := 30
def gary_gold_cost_per_gram : ℕ := 15
def anna_gold_weight : ℕ := 50
def anna_gold_cost_per_gram : ℕ := 20

theorem combined_gold_cost : (gary_gold_weight * gary_gold_cost_per_gram) + (anna_gold_weight * anna_gold_cost_per_gram) = 1450 :=
by {
  sorry -- Proof goes here
}

end NUMINAMATH_GPT_combined_gold_cost_l1509_150968


namespace NUMINAMATH_GPT_total_crayons_lost_or_given_away_l1509_150908

/-
Paul gave 52 crayons to his friends.
Paul lost 535 crayons.
Paul had 492 crayons left.
Prove that the total number of crayons lost or given away is 587.
-/
theorem total_crayons_lost_or_given_away
  (crayons_given : ℕ)
  (crayons_lost : ℕ)
  (crayons_left : ℕ)
  (h_crayons_given : crayons_given = 52)
  (h_crayons_lost : crayons_lost = 535)
  (h_crayons_left : crayons_left = 492) :
  crayons_given + crayons_lost = 587 := 
sorry

end NUMINAMATH_GPT_total_crayons_lost_or_given_away_l1509_150908


namespace NUMINAMATH_GPT_angle_A_is_pi_over_3_minimum_value_AM_sq_div_S_l1509_150950

open Real

variable {A B C a b c : ℝ}
variable (AM BM MC : ℝ)

-- Conditions
axiom triangle_sides : b / (sin A + sin C) = (a - c) / (sin B - sin C)
axiom BM_MC_relation : BM = (1 / 2) * MC

-- Part 1: Measure of angle A
theorem angle_A_is_pi_over_3 (triangle_sides : b / (sin A + sin C) = (a - c) / (sin B - sin C)) : 
  A = π / 3 :=
by sorry

-- Part 2: Minimum value of |AM|^2 / S
noncomputable def area_triangle (a b c : ℝ) (A : ℝ) : ℝ := 1 / 2 * b * c * sin A

axiom condition_b_eq_2c : b = 2 * c

theorem minimum_value_AM_sq_div_S (AM BM MC : ℝ) (S : ℝ) (H : BM = (1 / 2) * MC) 
  (triangle_sides : b / (sin A + sin C) = (a - c) / (sin B - sin C)) 
  (area : S = area_triangle a b c A)
  (condition_b_eq_2c : b = 2 * c) : 
  (AM ^ 2) / S ≥ (8 * sqrt 3) / 9 :=
by sorry

end NUMINAMATH_GPT_angle_A_is_pi_over_3_minimum_value_AM_sq_div_S_l1509_150950


namespace NUMINAMATH_GPT_union_M_N_l1509_150980

-- Define the set M
def M : Set ℤ := {x | x^2 - x = 0}

-- Define the set N
def N : Set ℤ := {y | y^2 + y = 0}

-- Prove that the union of M and N is {-1, 0, 1}
theorem union_M_N :
  M ∪ N = {-1, 0, 1} :=
by
  sorry

end NUMINAMATH_GPT_union_M_N_l1509_150980


namespace NUMINAMATH_GPT_range_of_a_l1509_150923

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, |x - a| + |x - 1| ≤ 2) → (a > 3 ∨ a < -1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1509_150923


namespace NUMINAMATH_GPT_eccentricity_of_ellipse_l1509_150978

noncomputable def ellipse_eccentricity (a b c : ℝ) : ℝ := c / a

theorem eccentricity_of_ellipse:
  ∀ (a b : ℝ) (c : ℝ), 
    0 < b ∧ b < a ∧ a = 3 * c → 
    ellipse_eccentricity a b c = 1/3 := by
  intros a b c h
  let e := ellipse_eccentricity a b c
  have h1 : 0 < b := h.1
  have h2 : b < a := h.2.left
  have h3 : a = 3 * c := h.2.right
  simp [ellipse_eccentricity, h3]
  sorry

end NUMINAMATH_GPT_eccentricity_of_ellipse_l1509_150978


namespace NUMINAMATH_GPT_triangle_inequality_l1509_150940

variable {a b c : ℝ}
variable {x y z : ℝ}

theorem triangle_inequality (ha : a ≥ b) (hb : b ≥ c) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (hx_yz_sum : x + y + z = π) :
  bc + ca - ab < bc * Real.cos x + ca * Real.cos y + ab * Real.cos z ∧
  bc * Real.cos x + ca * Real.cos y + ab * Real.cos z ≤ (a^2 + b^2 + c^2) / 2 := sorry

end NUMINAMATH_GPT_triangle_inequality_l1509_150940


namespace NUMINAMATH_GPT_tree_original_height_l1509_150986

theorem tree_original_height (current_height_in: ℝ) (growth_percentage: ℝ)
  (h1: current_height_in = 180) (h2: growth_percentage = 0.50) :
  ∃ (original_height_ft: ℝ), original_height_ft = 10 :=
by
  have original_height_in := current_height_in / (1 + growth_percentage)
  have original_height_ft := original_height_in / 12
  use original_height_ft
  sorry

end NUMINAMATH_GPT_tree_original_height_l1509_150986


namespace NUMINAMATH_GPT_range_of_f3_l1509_150928

def f (a c x : ℝ) : ℝ := a * x^2 - c

theorem range_of_f3 (a c : ℝ)
  (h1 : -4 ≤ f a c 1 ∧ f a c 1 ≤ -1)
  (h2 : -1 ≤ f a c 2 ∧ f a c 2 ≤ 5) :
  -1 ≤ f a c 3 ∧ f a c 3 ≤ 20 := 
sorry

end NUMINAMATH_GPT_range_of_f3_l1509_150928


namespace NUMINAMATH_GPT_ratio_mets_to_redsox_l1509_150905

theorem ratio_mets_to_redsox (Y M R : ℕ)
  (h1 : Y / M = 3 / 2)
  (h2 : M = 96)
  (h3 : Y + M + R = 360) :
  M / R = 4 / 5 :=
by sorry

end NUMINAMATH_GPT_ratio_mets_to_redsox_l1509_150905


namespace NUMINAMATH_GPT_farm_owns_60_more_horses_than_cows_l1509_150939

-- Let x be the number of cows initially
-- The number of horses initially is 4x
-- After selling 15 horses and buying 15 cows, the ratio of horses to cows becomes 7:3

theorem farm_owns_60_more_horses_than_cows (x : ℕ) (h_pos : 0 < x)
  (h_ratio : (4 * x - 15) / (x + 15) = 7 / 3) :
  (4 * x - 15) - (x + 15) = 60 :=
by
  sorry

end NUMINAMATH_GPT_farm_owns_60_more_horses_than_cows_l1509_150939


namespace NUMINAMATH_GPT_min_value_expression_l1509_150994

theorem min_value_expression {x y : ℝ} : 
  2 * x + y - 3 = 0 →
  x + 2 * y - 1 = 0 →
  (5 * x^2 + 8 * x * y + 5 * y^2 - 14 * x - 10 * y + 30) / (4 - x^2 - 10 * x * y - 25 * y^2) ^ (7 / 2) = 5 / 32 :=
by
  sorry

end NUMINAMATH_GPT_min_value_expression_l1509_150994


namespace NUMINAMATH_GPT_muffin_price_proof_l1509_150921

noncomputable def price_per_muffin (s m t : ℕ) (contribution : ℕ) : ℕ :=
  contribution / (s + m + t)

theorem muffin_price_proof :
  ∀ (sasha_muffins melissa_muffins : ℕ) (h1 : sasha_muffins = 30) (h2 : melissa_muffins = 4 * sasha_muffins)
  (tiffany_muffins total_muffins : ℕ) (h3 : total_muffins = sasha_muffins + melissa_muffins)
  (h4 : tiffany_muffins = total_muffins / 2)
  (h5 : total_muffins = sasha_muffins + melissa_muffins + tiffany_muffins)
  (contribution : ℕ) (h6 : contribution = 900),
  price_per_muffin sasha_muffins melissa_muffins tiffany_muffins contribution = 4 :=
by
  intros sasha_muffins melissa_muffins h1 h2 tiffany_muffins total_muffins h3 h4 h5 contribution h6
  simp [price_per_muffin]
  sorry

end NUMINAMATH_GPT_muffin_price_proof_l1509_150921


namespace NUMINAMATH_GPT_length_of_bridge_is_230_l1509_150976

noncomputable def train_length : ℚ := 145
noncomputable def train_speed_kmh : ℚ := 45
noncomputable def time_to_cross_bridge : ℚ := 30
noncomputable def train_speed_ms : ℚ := (train_speed_kmh * 1000) / 3600
noncomputable def bridge_length : ℚ := (train_speed_ms * time_to_cross_bridge) - train_length

theorem length_of_bridge_is_230 :
  bridge_length = 230 :=
sorry

end NUMINAMATH_GPT_length_of_bridge_is_230_l1509_150976


namespace NUMINAMATH_GPT_binomial_identity_l1509_150995

-- Given:
variables {k n : ℕ}

-- Conditions:
axiom h₁ : 1 < k
axiom h₂ : 1 < n

-- Statement:
theorem binomial_identity (h₁ : 1 < k) (h₂ : 1 < n) : 
  k * Nat.choose n k = n * Nat.choose (n - 1) (k - 1) := 
sorry

end NUMINAMATH_GPT_binomial_identity_l1509_150995


namespace NUMINAMATH_GPT_min_value_of_expression_l1509_150983

theorem min_value_of_expression (a_n : ℕ → ℝ) (S_n : ℕ → ℝ)
    (h1 : ∀ n, S_n n = (4/3) * (a_n n - 1)) :
  ∃ (n : ℕ), (4^(n - 2) + 1) * (16 / a_n n + 1) = 4 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_expression_l1509_150983


namespace NUMINAMATH_GPT_find_a_l1509_150906

theorem find_a (x y a : ℤ) (h1 : a * x + y = 40) (h2 : 2 * x - y = 20) (h3 : 3 * y^2 = 48) : a = 3 :=
sorry

end NUMINAMATH_GPT_find_a_l1509_150906


namespace NUMINAMATH_GPT_negation_of_exists_proposition_l1509_150988

theorem negation_of_exists_proposition :
  ¬ (∃ x₀ : ℝ, x₀^2 - 1 < 0) ↔ ∀ x : ℝ, x^2 - 1 ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_exists_proposition_l1509_150988


namespace NUMINAMATH_GPT_digit_sum_2001_not_perfect_square_l1509_150975

theorem digit_sum_2001_not_perfect_square (n : ℕ) (h : (n.digits 10).sum = 2001) : ¬ ∃ k : ℕ, n = k * k := 
sorry

end NUMINAMATH_GPT_digit_sum_2001_not_perfect_square_l1509_150975


namespace NUMINAMATH_GPT_tim_youth_comparison_l1509_150970

theorem tim_youth_comparison :
  let Tim_age : ℕ := 5
  let Rommel_age : ℕ := 3 * Tim_age
  let Jenny_age : ℕ := Rommel_age + 2
  Jenny_age - Tim_age = 12 := 
by 
  sorry

end NUMINAMATH_GPT_tim_youth_comparison_l1509_150970


namespace NUMINAMATH_GPT_votes_cast_l1509_150974

theorem votes_cast (V : ℝ) (h1 : 0.35 * V + 2250 = 0.65 * V) : V = 7500 := 
by
  sorry

end NUMINAMATH_GPT_votes_cast_l1509_150974


namespace NUMINAMATH_GPT_investment_ratio_correct_l1509_150925

-- Constants representing the savings and investments
def weekly_savings_wife : ℕ := 100
def monthly_savings_husband : ℕ := 225
def weeks_in_month : ℕ := 4
def months_saving : ℕ := 4
def cost_per_share : ℕ := 50
def shares_bought : ℕ := 25

-- Derived quantities from the conditions
def total_savings_wife : ℕ := weekly_savings_wife * weeks_in_month * months_saving
def total_savings_husband : ℕ := monthly_savings_husband * months_saving
def total_savings : ℕ := total_savings_wife + total_savings_husband
def total_invested_in_stocks : ℕ := shares_bought * cost_per_share
def investment_ratio_nat : ℚ := (total_invested_in_stocks : ℚ) / (total_savings : ℚ)

-- Proof statement
theorem investment_ratio_correct : investment_ratio_nat = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_investment_ratio_correct_l1509_150925


namespace NUMINAMATH_GPT_rainfall_second_week_value_l1509_150997

-- Define the conditions
variables (rainfall_first_week rainfall_second_week : ℝ)
axiom condition1 : rainfall_first_week + rainfall_second_week = 30
axiom condition2 : rainfall_second_week = 1.5 * rainfall_first_week

-- Define the theorem we want to prove
theorem rainfall_second_week_value : rainfall_second_week = 18 := by
  sorry

end NUMINAMATH_GPT_rainfall_second_week_value_l1509_150997


namespace NUMINAMATH_GPT_electric_fan_wattage_l1509_150951

theorem electric_fan_wattage (hours_per_day : ℕ) (energy_per_month : ℝ) (days_per_month : ℕ) 
  (h1 : hours_per_day = 8) (h2 : energy_per_month = 18) (h3 : days_per_month = 30) : 
  (energy_per_month * 1000) / (days_per_month * hours_per_day) = 75 := 
by { 
  -- Placeholder for the proof
  sorry 
}

end NUMINAMATH_GPT_electric_fan_wattage_l1509_150951


namespace NUMINAMATH_GPT_total_games_played_l1509_150993

theorem total_games_played (won_games : ℕ) (won_ratio : ℕ) (lost_ratio : ℕ) (tied_ratio : ℕ) (total_games : ℕ) :
  won_games = 42 →
  won_ratio = 7 →
  lost_ratio = 4 →
  tied_ratio = 5 →
  total_games = won_games + lost_ratio * (won_games / won_ratio) + tied_ratio * (won_games / won_ratio) →
  total_games = 96 :=
by
  intros h_won h_won_ratio h_lost_ratio h_tied_ratio h_total
  sorry

end NUMINAMATH_GPT_total_games_played_l1509_150993


namespace NUMINAMATH_GPT_find_f_of_2_l1509_150907

def f (x : ℝ) (a b : ℝ) : ℝ := x^3 + a * x^2 + b * x + a^2

theorem find_f_of_2 (a b : ℝ)
  (h1 : 3 + 2 * a + b = 0)
  (h2 : 1 + a + b + a^2 = 10)
  (ha : a = 4)
  (hb : b = -11) :
  f 2 a b = 18 := by {
  -- We assume the values of a and b provided by the user as the correct pair.
  sorry
}

end NUMINAMATH_GPT_find_f_of_2_l1509_150907


namespace NUMINAMATH_GPT_ellipse_foci_x_axis_l1509_150903

theorem ellipse_foci_x_axis (m n : ℝ) (h_eq : ∀ (x y : ℝ), m * x^2 + n * y^2 = 1)
  (h_foci : ∃ (c : ℝ), c = 0 ∧ (c^2 = 1 - n/m)) : n > m ∧ m > 0 ∧ n > 0 :=
sorry

end NUMINAMATH_GPT_ellipse_foci_x_axis_l1509_150903


namespace NUMINAMATH_GPT_perfect_square_a_i_l1509_150981

theorem perfect_square_a_i (a : ℕ → ℕ)
  (h1 : a 1 = 1) 
  (h2 : a 2 = 1) 
  (h3 : ∀ n, a (n + 2) = 18 * a (n + 1) - a n) :
  ∀ i, ∃ k, 5 * (a i) ^ 2 - 1 = k ^ 2 :=
by
  -- The proof is missing the skipped definitions from the problem and solution context
  sorry

end NUMINAMATH_GPT_perfect_square_a_i_l1509_150981


namespace NUMINAMATH_GPT_evaluate_expression_l1509_150947

noncomputable def g : ℕ → ℕ := sorry
noncomputable def g_inv : ℕ → ℕ := sorry

axiom g_inverse : ∀ x, g (g_inv x) = x ∧ g_inv (g x) = x

axiom g_1_2 : g 1 = 2
axiom g_4_7 : g 4 = 7
axiom g_3_8 : g 3 = 8

theorem evaluate_expression :
  g_inv (g_inv 8 * g_inv 2) = 3 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1509_150947


namespace NUMINAMATH_GPT_common_ratio_of_geometric_seq_l1509_150941

variable {α : Type*} [Field α]

-- Definition of the geometric sequence
def geometric_seq (a : α) (q : α) (n : ℕ) : α := a * q ^ n

-- Sum of the first three terms of the geometric sequence
def sum_first_three_terms (a q: α) : α :=
  geometric_seq a q 0 + geometric_seq a q 1 + geometric_seq a q 2

theorem common_ratio_of_geometric_seq (a q : α) (h : sum_first_three_terms a q = 3 * a) : q = 1 ∨ q = -2 :=
sorry

end NUMINAMATH_GPT_common_ratio_of_geometric_seq_l1509_150941


namespace NUMINAMATH_GPT_lines_identical_pairs_count_l1509_150901

theorem lines_identical_pairs_count :
  (∃ a d : ℝ, (4 * x + a * y + d = 0 ∧ d * x - 3 * y + 15 = 0)) →
  (∃! n : ℕ, n = 2) := 
sorry

end NUMINAMATH_GPT_lines_identical_pairs_count_l1509_150901


namespace NUMINAMATH_GPT_a100_gt_2pow99_l1509_150946

theorem a100_gt_2pow99 (a : Fin 101 → ℕ) 
  (h_pos : ∀ i, a i > 0) 
  (h_initial : a 1 > a 0) 
  (h_rec : ∀ k, 2 ≤ k → a k = 3 * a (k - 1) - 2 * a (k - 2)) 
  : a 100 > 2 ^ 99 :=
by
  sorry

end NUMINAMATH_GPT_a100_gt_2pow99_l1509_150946


namespace NUMINAMATH_GPT_balloons_given_by_mom_l1509_150962

-- Definitions of the initial and total number of balloons
def initial_balloons := 26
def total_balloons := 60

-- Theorem: Proving the number of balloons Tommy's mom gave him
theorem balloons_given_by_mom : total_balloons - initial_balloons = 34 :=
by
  -- This proof is obvious from the setup, so we write sorry to skip the proof.
  sorry

end NUMINAMATH_GPT_balloons_given_by_mom_l1509_150962


namespace NUMINAMATH_GPT_evaluate_expression_l1509_150969

noncomputable def expression_equal : Prop :=
  let a := (11: ℝ)
  let b := (11 : ℝ)^((1 : ℝ) / 6)
  let c := (11 : ℝ)^((1 : ℝ) / 5)
  (b / c = a^(-((1 : ℝ) / 30)))

theorem evaluate_expression :
  expression_equal :=
sorry

end NUMINAMATH_GPT_evaluate_expression_l1509_150969


namespace NUMINAMATH_GPT_ratio_b_to_c_l1509_150979

variables (a b c d e f : ℝ)

theorem ratio_b_to_c 
  (h1 : a / b = 1 / 3)
  (h2 : c / d = 1 / 2)
  (h3 : d / e = 3)
  (h4 : e / f = 1 / 10)
  (h5 : a * b * c / (d * e * f) = 0.15) :
  b / c = 9 := 
sorry

end NUMINAMATH_GPT_ratio_b_to_c_l1509_150979


namespace NUMINAMATH_GPT_collinear_magnitude_a_perpendicular_magnitude_b_l1509_150915

noncomputable section

open Real

-- Defining the vectors a and b
def a (x : ℝ) : ℝ × ℝ := (1, x)
def b (x : ℝ) : ℝ × ℝ := (x, 3)

-- Defining the magnitude of a vector
def magnitude (v : ℝ × ℝ) : ℝ := sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Given conditions and respective proofs
theorem collinear_magnitude_a (x : ℝ) (h : 1 * 3 = x ^ 2) : magnitude (a x) = 2 :=
by sorry

theorem perpendicular_magnitude_b (x : ℝ) (h : 1 * x + x * 3 = 0) : magnitude (b x) = 3 :=
by sorry

end NUMINAMATH_GPT_collinear_magnitude_a_perpendicular_magnitude_b_l1509_150915


namespace NUMINAMATH_GPT_fifteenth_term_geometric_sequence_l1509_150932

theorem fifteenth_term_geometric_sequence :
  let a1 := 5
  let r := (1 : ℝ) / 2
  let fifteenth_term := a1 * r^(14 : ℕ)
  fifteenth_term = (5 : ℝ) / 16384 := by
sorry

end NUMINAMATH_GPT_fifteenth_term_geometric_sequence_l1509_150932


namespace NUMINAMATH_GPT_maria_threw_out_carrots_l1509_150910

theorem maria_threw_out_carrots (initially_picked: ℕ) (picked_next_day: ℕ) (total_now: ℕ) (carrots_thrown_out: ℕ) :
  initially_picked = 48 → 
  picked_next_day = 15 → 
  total_now = 52 → 
  (initially_picked + picked_next_day - total_now = carrots_thrown_out) → 
  carrots_thrown_out = 11 :=
by
  intros
  sorry

end NUMINAMATH_GPT_maria_threw_out_carrots_l1509_150910


namespace NUMINAMATH_GPT_thirty_six_forty_five_nine_eighteen_l1509_150987

theorem thirty_six_forty_five_nine_eighteen :
  18 * 36 + 45 * 18 - 9 * 18 = 1296 :=
by
  sorry

end NUMINAMATH_GPT_thirty_six_forty_five_nine_eighteen_l1509_150987


namespace NUMINAMATH_GPT_range_of_a_l1509_150973

-- Define the function f and its derivative f'
def f (a x : ℝ) : ℝ := x^3 + 3 * a * x^2 + 3 * (a + 2) * x + 1
def f_prime (a x : ℝ) : ℝ := 3 * x^2 + 6 * a * x + 3 * (a + 2)

-- We are given that for f to have both maximum and minimum values, f' must have two distinct roots
-- Thus we translate the mathematical condition to the discriminant of f' being greater than 0
def discriminant_greater_than_zero (a : ℝ) : Prop :=
  (6 * a)^2 - 4 * 3 * 3 * (a + 2) > 0

-- Finally, we want to prove that this simplifies to a condition on a
theorem range_of_a (a : ℝ) : discriminant_greater_than_zero a ↔ (a > 2 ∨ a < -1) :=
by
  -- Write the proof here
  sorry

end NUMINAMATH_GPT_range_of_a_l1509_150973


namespace NUMINAMATH_GPT_find_height_of_triangular_prism_l1509_150936

-- Define the conditions
def volume (V : ℝ) : Prop := V = 120
def base_side1 (a : ℝ) : Prop := a = 3
def base_side2 (b : ℝ) : Prop := b = 4

-- The final proof problem
theorem find_height_of_triangular_prism (V : ℝ) (a : ℝ) (b : ℝ) (h : ℝ) 
  (h1 : volume V) (h2 : base_side1 a) (h3 : base_side2 b) : h = 20 :=
by
  -- The actual proof goes here
  sorry

end NUMINAMATH_GPT_find_height_of_triangular_prism_l1509_150936


namespace NUMINAMATH_GPT_martha_total_butterflies_l1509_150958

variable (Yellow Blue Black : ℕ)

def butterfly_equations (Yellow Blue Black : ℕ) : Prop :=
  (Blue = 2 * Yellow) ∧ (Blue = 6) ∧ (Black = 10)

theorem martha_total_butterflies 
  (h : butterfly_equations Yellow Blue Black) : 
  (Yellow + Blue + Black = 19) :=
by
  sorry

end NUMINAMATH_GPT_martha_total_butterflies_l1509_150958


namespace NUMINAMATH_GPT_area_of_inscribed_square_l1509_150964

theorem area_of_inscribed_square (a : ℝ) : 
    ∃ S : ℝ, S = 3 * a^2 / (7 - 4 * Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_GPT_area_of_inscribed_square_l1509_150964


namespace NUMINAMATH_GPT_sum_of_roots_of_quadratic_l1509_150919

theorem sum_of_roots_of_quadratic :
  ∀ x : ℝ, x^2 + 2000*x - 2000 = 0 ->
  (∃ x1 x2 : ℝ, (x1 ≠ x2 ∧ x1^2 + 2000*x1 - 2000 = 0 ∧ x2^2 + 2000*x2 - 2000 = 0 ∧ x1 + x2 = -2000)) :=
sorry

end NUMINAMATH_GPT_sum_of_roots_of_quadratic_l1509_150919


namespace NUMINAMATH_GPT_original_number_is_7_l1509_150965

theorem original_number_is_7 (x : ℤ) (h : (((3 * (x + 3) + 3) - 3) / 3) = 10) : x = 7 :=
sorry

end NUMINAMATH_GPT_original_number_is_7_l1509_150965


namespace NUMINAMATH_GPT_cost_of_apples_l1509_150996

theorem cost_of_apples (price_per_six_pounds : ℕ) (pounds_to_buy : ℕ) (expected_cost : ℕ) :
  price_per_six_pounds = 5 → pounds_to_buy = 18 → (expected_cost = 15) → 
  (price_per_six_pounds / 6) * pounds_to_buy = expected_cost :=
by
  intro price_per_six_pounds_eq pounds_to_buy_eq expected_cost_eq
  rw [price_per_six_pounds_eq, pounds_to_buy_eq, expected_cost_eq]
  -- the actual proof would follow, using math steps similar to the solution but skipped here
  sorry

end NUMINAMATH_GPT_cost_of_apples_l1509_150996


namespace NUMINAMATH_GPT_total_tips_l1509_150914

def tips_per_customer := 2
def customers_friday := 28
def customers_saturday := 3 * customers_friday
def customers_sunday := 36

theorem total_tips : 
  (tips_per_customer * (customers_friday + customers_saturday + customers_sunday) = 296) :=
by
  sorry

end NUMINAMATH_GPT_total_tips_l1509_150914


namespace NUMINAMATH_GPT_find_n_l1509_150931

noncomputable def arctan_sum_eq_pi_over_2 (n : ℕ) : Prop :=
  Real.arctan (1 / 3) + Real.arctan (1 / 4) + Real.arctan (1 / 7) + Real.arctan (1 / n) = Real.pi / 2

theorem find_n (h : ∃ n, arctan_sum_eq_pi_over_2 n) : ∃ n, n = 54 := by
  obtain ⟨n, hn⟩ := h
  have H : 1 / 3 + 1 / 4 + 1 / 7 < 1 := by sorry
  sorry

end NUMINAMATH_GPT_find_n_l1509_150931


namespace NUMINAMATH_GPT_simplify_expression_l1509_150977

theorem simplify_expression :
  (18 / 17) * (13 / 24) * (68 / 39) = 1 := 
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1509_150977


namespace NUMINAMATH_GPT_log_equivalence_l1509_150938

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_equivalence (x : ℝ) (h : log_base 16 (x - 3) = 1 / 2) : log_base 256 (x + 1) = 3 / 8 :=
  sorry

end NUMINAMATH_GPT_log_equivalence_l1509_150938


namespace NUMINAMATH_GPT_total_pages_l1509_150990

-- Conditions
variables (B1 B2 : ℕ)
variable (h1 : (2 / 3 : ℚ) * B1 - (1 / 3 : ℚ) * B1 = 90)
variable (h2 : (3 / 4 : ℚ) * B2 - (1 / 4 : ℚ) * B2 = 120)

-- Theorem statement
theorem total_pages (B1 B2 : ℕ) (h1 : (2 / 3 : ℚ) * B1 - (1 / 3 : ℚ) * B1 = 90) (h2 : (3 / 4 : ℚ) * B2 - (1 / 4 : ℚ) * B2 = 120) :
  B1 + B2 = 510 :=
sorry

end NUMINAMATH_GPT_total_pages_l1509_150990


namespace NUMINAMATH_GPT_total_splash_width_l1509_150904

theorem total_splash_width :
  let pebble_splash := 1 / 4
  let rock_splash := 1 / 2
  let boulder_splash := 2
  let pebbles := 6
  let rocks := 3
  let boulders := 2
  let total_pebble_splash := pebbles * pebble_splash
  let total_rock_splash := rocks * rock_splash
  let total_boulder_splash := boulders * boulder_splash
  let total_splash := total_pebble_splash + total_rock_splash + total_boulder_splash
  total_splash = 7 := by
  sorry

end NUMINAMATH_GPT_total_splash_width_l1509_150904


namespace NUMINAMATH_GPT_ellipse_foci_distance_l1509_150924

noncomputable def distance_between_foci (a b : ℝ) : ℝ := 
  Real.sqrt (a^2 - b^2)

theorem ellipse_foci_distance :
  ∀ (a b : ℝ), a = 6 → b = 3 → distance_between_foci a b = 3 * Real.sqrt 3 :=
by
  intros a b h_a h_b
  rw [h_a, h_b]
  simp [distance_between_foci]
  sorry

end NUMINAMATH_GPT_ellipse_foci_distance_l1509_150924


namespace NUMINAMATH_GPT_billy_can_play_l1509_150961

-- Define the conditions
def total_songs : ℕ := 52
def songs_to_learn : ℕ := 28

-- Define the statement to be proved
theorem billy_can_play : total_songs - songs_to_learn = 24 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_billy_can_play_l1509_150961


namespace NUMINAMATH_GPT_xy_sum_l1509_150912

theorem xy_sum (x y : ℝ) (h : x^2 + y^2 = 12 * x - 8 * y + 20) : x + y = 12 + 2 * Real.sqrt 6 ∨ x + y = 12 - 2 * Real.sqrt 6 :=
by
  sorry

end NUMINAMATH_GPT_xy_sum_l1509_150912


namespace NUMINAMATH_GPT_taxi_ride_cost_l1509_150909

-- Definitions based on conditions
def base_fare : ℝ := 2.00
def cost_per_mile : ℝ := 0.30
def minimum_charge : ℝ := 5.00
def fare (miles : ℝ) : ℝ := base_fare + miles * cost_per_mile

-- Theorem statement reflecting the problem
theorem taxi_ride_cost (miles : ℝ) (h : miles < 4) : fare miles < minimum_charge → fare miles = minimum_charge :=
by
  sorry

end NUMINAMATH_GPT_taxi_ride_cost_l1509_150909


namespace NUMINAMATH_GPT_slope_angle_of_tangent_line_expx_at_0_l1509_150945

theorem slope_angle_of_tangent_line_expx_at_0 :
  let f := fun x : ℝ => Real.exp x 
  let f' := fun x : ℝ => Real.exp x
  ∀ x : ℝ, f' x = Real.exp x → 
  (∃ α : ℝ, Real.tan α = 1) →
  α = Real.pi / 4 :=
by
  intros f f' h_deriv h_slope
  sorry

end NUMINAMATH_GPT_slope_angle_of_tangent_line_expx_at_0_l1509_150945


namespace NUMINAMATH_GPT_product_of_real_roots_of_equation_l1509_150991

theorem product_of_real_roots_of_equation : 
  ∀ x : ℝ, (x^4 + (x - 4)^4 = 32) → x = 2 :=
sorry

end NUMINAMATH_GPT_product_of_real_roots_of_equation_l1509_150991
