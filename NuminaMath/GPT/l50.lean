import Mathlib

namespace two_numbers_max_product_l50_5064

theorem two_numbers_max_product :
  ∃ x y : ℝ, x - y = 4 ∧ x + y = 35 ∧ ∀ z w : ℝ, z - w = 4 → z + w = 35 → z * w ≤ x * y :=
by
  sorry

end two_numbers_max_product_l50_5064


namespace mean_equality_l50_5053

theorem mean_equality (x : ℤ) (h : (8 + 10 + 24) / 3 = (16 + x + 18) / 3) : x = 8 := by 
sorry

end mean_equality_l50_5053


namespace area_increase_percentage_l50_5009

-- Define the original dimensions l and w as non-zero real numbers
variables (l w : ℝ) (hl : l ≠ 0) (hw : w ≠ 0)

-- Define the new dimensions after increase
def new_length := 1.15 * l
def new_width := 1.25 * w

-- Define the original and new areas
def original_area := l * w
def new_area := new_length l * new_width w

-- The statement to prove
theorem area_increase_percentage :
  ((new_area l w - original_area l w) / original_area l w) * 100 = 43.75 :=
by
  sorry

end area_increase_percentage_l50_5009


namespace points_on_line_possible_l50_5097

theorem points_on_line_possible : ∃ n : ℕ, 9 * n - 8 = 82 :=
by
  sorry

end points_on_line_possible_l50_5097


namespace james_balloons_l50_5088

-- Definitions
def amy_balloons : ℕ := 513
def extra_balloons_james_has : ℕ := 709

-- Statement of the problem
theorem james_balloons : amy_balloons + extra_balloons_james_has = 1222 :=
by
  -- Placeholder for the actual proof
  sorry

end james_balloons_l50_5088


namespace total_students_l50_5043

def Varsity_students : ℕ := 1300
def Northwest_students : ℕ := 1400
def Central_students : ℕ := 1800
def Greenbriar_students : ℕ := 1650

theorem total_students : Varsity_students + Northwest_students + Central_students + Greenbriar_students = 6150 :=
by
  -- Proof is omitted, so we use sorry.
  sorry

end total_students_l50_5043


namespace probability_B_winning_l50_5062

def P_A : ℝ := 0.2
def P_D : ℝ := 0.5
def P_B : ℝ := 1 - (P_A + P_D)

theorem probability_B_winning : P_B = 0.3 :=
by
  -- Proof steps go here
  sorry

end probability_B_winning_l50_5062


namespace money_made_per_minute_l50_5094

theorem money_made_per_minute (total_tshirts : ℕ) (time_minutes : ℕ) (black_tshirt_price white_tshirt_price : ℕ) (num_black num_white : ℕ) :
  total_tshirts = 200 →
  time_minutes = 25 →
  black_tshirt_price = 30 →
  white_tshirt_price = 25 →
  num_black = total_tshirts / 2 →
  num_white = total_tshirts / 2 →
  (num_black * black_tshirt_price + num_white * white_tshirt_price) / time_minutes = 220 :=
by
  sorry

end money_made_per_minute_l50_5094


namespace total_balls_l50_5078

theorem total_balls (black_balls : ℕ) (prob_pick_black : ℚ) (total_balls : ℕ) :
  black_balls = 4 → prob_pick_black = 1 / 3 → total_balls = 12 :=
by
  intros h1 h2
  sorry

end total_balls_l50_5078


namespace probability_no_more_than_10_seconds_l50_5095

noncomputable def total_cycle_time : ℕ := 80
noncomputable def green_time : ℕ := 30
noncomputable def yellow_time : ℕ := 10
noncomputable def red_time : ℕ := 40
noncomputable def can_proceed : ℕ := green_time + yellow_time + yellow_time

theorem probability_no_more_than_10_seconds : 
  can_proceed / total_cycle_time = 5 / 8 := 
  sorry

end probability_no_more_than_10_seconds_l50_5095


namespace parabola_intersect_l50_5093

theorem parabola_intersect (b c m p q x1 x2 : ℝ)
  (h_intersect1 : x1^2 + b * x1 + c = 0)
  (h_intersect2 : x2^2 + b * x2 + c = 0)
  (h_order : m < x1)
  (h_middle : x1 < x2)
  (h_range : x2 < m + 1)
  (h_valm : p = m^2 + b * m + c)
  (h_valm1 : q = (m + 1)^2 + b * (m + 1) + c) :
  p < 1 / 4 ∧ q < 1 / 4 :=
sorry

end parabola_intersect_l50_5093


namespace unique_element_set_l50_5077

theorem unique_element_set (a : ℝ) : 
  (∃! x, (a - 1) * x^2 + 3 * x - 2 = 0) ↔ (a = 1 ∨ a = -1 / 8) :=
by sorry

end unique_element_set_l50_5077


namespace light_travel_50_years_l50_5074

theorem light_travel_50_years :
  let one_year_distance := 9460800000000 -- distance light travels in one year
  let fifty_years_distance := 50 * one_year_distance
  let scientific_notation_distance := 473.04 * 10^12
  fifty_years_distance = scientific_notation_distance :=
by
  sorry

end light_travel_50_years_l50_5074


namespace max_receptivity_compare_receptivity_l50_5008

-- Define the piecewise function f(x)
noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 10 then -0.1 * x^2 + 2.6 * x + 43
  else if 10 < x ∧ x <= 16 then 59
  else if 16 < x ∧ x <= 30 then -3 * x + 107
  else 0 -- To cover the case when x is outside the given ranges

-- Problem 1
theorem max_receptivity :
  f 10 = 59 ∧ ∀ x, 10 < x ∧ x ≤ 16 → f x = 59 :=
by
  sorry

-- Problem 2
theorem compare_receptivity :
  f 5 > f 20 :=
by
  sorry

end max_receptivity_compare_receptivity_l50_5008


namespace monthly_earnings_is_correct_l50_5036

-- Conditions as definitions

def first_floor_cost_per_room : ℕ := 15
def second_floor_cost_per_room : ℕ := 20
def first_floor_rooms : ℕ := 3
def second_floor_rooms : ℕ := 3
def third_floor_rooms : ℕ := 3
def occupied_third_floor_rooms : ℕ := 2

-- Calculated values from conditions
def third_floor_cost_per_room : ℕ := 2 * first_floor_cost_per_room

-- Total earnings on each floor
def first_floor_earnings : ℕ := first_floor_cost_per_room * first_floor_rooms
def second_floor_earnings : ℕ := second_floor_cost_per_room * second_floor_rooms
def third_floor_earnings : ℕ := third_floor_cost_per_room * occupied_third_floor_rooms

-- Total monthly earnings
def total_monthly_earnings : ℕ :=
  first_floor_earnings + second_floor_earnings + third_floor_earnings

theorem monthly_earnings_is_correct : total_monthly_earnings = 165 := by
  -- proof omitted
  sorry

end monthly_earnings_is_correct_l50_5036


namespace smallest_number_of_three_integers_l50_5022

theorem smallest_number_of_three_integers 
  (a b c : ℕ) 
  (hpos1 : 0 < a) (hpos2 : 0 < b) (hpos3 : 0 < c) 
  (hmean : (a + b + c) / 3 = 24)
  (hmed : b = 23)
  (hlargest : b + 4 = c) 
  : a = 22 :=
by
  sorry

end smallest_number_of_three_integers_l50_5022


namespace can_lid_boxes_count_l50_5073

theorem can_lid_boxes_count 
  (x y : ℕ) 
  (h1 : 3 * x + y + 14 = 75) : 
  x = 20 ∧ y = 1 :=
by 
  sorry

end can_lid_boxes_count_l50_5073


namespace num_students_above_120_l50_5011

noncomputable def class_size : ℤ := 60
noncomputable def mean_score : ℝ := 110
noncomputable def std_score : ℝ := sorry  -- We do not know σ explicitly
noncomputable def probability_100_to_110 : ℝ := 0.35

def normal_distribution (x : ℝ) : Prop :=
  sorry -- placeholder for the actual normal distribution formula N(110, σ^2)

theorem num_students_above_120 :
  ∃ (students_above_120 : ℤ),
  (class_size = 60) ∧
  (∀ score, normal_distribution score → (100 ≤ score ∧ score ≤ 110) → probability_100_to_110 = 0.35) →
  students_above_120 = 9 :=
sorry

end num_students_above_120_l50_5011


namespace correct_transformation_l50_5026

theorem correct_transformation (a b : ℝ) (hb : b ≠ 0): (a / b = 2 * a / (2 * b)) :=
by
  sorry

end correct_transformation_l50_5026


namespace multiplier_is_three_l50_5089

theorem multiplier_is_three (n m : ℝ) (h₁ : n = 3) (h₂ : 7 * n = m * n + 12) : m = 3 := 
by
  -- Skipping the proof using sorry
  sorry 

end multiplier_is_three_l50_5089


namespace value_of_a_l50_5024

noncomputable def f (a x : ℝ) : ℝ := a^x + Real.logb a (x^2 + 1)

theorem value_of_a (a : ℝ) (h : f a 1 + f a 2 = a^2 + a + 2) : a = Real.sqrt 10 :=
by
  sorry

end value_of_a_l50_5024


namespace Dhoni_spending_difference_l50_5047

-- Definitions
def RentPercent := 20
def LeftOverPercent := 61
def TotalSpendPercent := 100 - LeftOverPercent
def DishwasherPercent := TotalSpendPercent - RentPercent

-- Theorem statement
theorem Dhoni_spending_difference :
  DishwasherPercent = RentPercent - 1 := 
by
  sorry

end Dhoni_spending_difference_l50_5047


namespace concert_attendance_l50_5099

-- Define the given conditions
def buses : ℕ := 8
def students_per_bus : ℕ := 45

-- Statement of the problem
theorem concert_attendance :
  buses * students_per_bus = 360 :=
sorry

end concert_attendance_l50_5099


namespace Mark_average_speed_l50_5005

theorem Mark_average_speed 
  (start_time : ℝ) (end_time : ℝ) (distance : ℝ)
  (h1 : start_time = 8.5) (h2 : end_time = 14.75) (h3 : distance = 210) :
  distance / (end_time - start_time) = 33.6 :=
by 
  sorry

end Mark_average_speed_l50_5005


namespace max_ab_bc_cd_da_l50_5002

theorem max_ab_bc_cd_da (a b c d : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : 0 ≤ d) (h₄ : a + b + c + d = 200) :
  ab + bc + cd + da ≤ 10000 :=
sorry

end max_ab_bc_cd_da_l50_5002


namespace difference_between_relations_l50_5004

-- Definitions based on conditions
def functional_relationship 
  (f : α → β) (x : α) (y : β) : Prop :=
  f x = y

def correlation_relationship (X Y : Type) : Prop :=
  ∃ (X_rand : X → ℝ) (Y_rand : Y → ℝ), 
    ∀ (x : X), ∃ (y : Y), X_rand x ≠ Y_rand y

-- Theorem stating the problem
theorem difference_between_relations :
  (∀ (f : α → β) (x : α) (y : β), functional_relationship f x y) ∧ 
  (∀ (X Y : Type), correlation_relationship X Y) :=
sorry

end difference_between_relations_l50_5004


namespace number_of_ways_to_choose_water_polo_team_l50_5098

theorem number_of_ways_to_choose_water_polo_team :
  let total_members := 15
  let team_size := 7
  let goalie_positions := 1
  let player_choices := team_size - goalie_positions
  ∃ (total_ways : ℕ), 
  total_ways = total_members * Nat.choose (total_members - 1) player_choices ∧ 
  total_ways = 45045 :=
by
  let total_members := 15
  let team_size := 7
  let goalie_positions := 1
  let player_choices := team_size - goalie_positions
  have total_ways : ℕ := total_members * Nat.choose (total_members - 1) player_choices
  use total_ways
  sorry

end number_of_ways_to_choose_water_polo_team_l50_5098


namespace find_real_root_a_l50_5050

theorem find_real_root_a (a b c : ℂ) (ha : a.im = 0) (h1 : a + b + c = 5) (h2 : a * b + b * c + c * a = 7) (h3 : a * b * c = 3) : a = 1 :=
sorry

end find_real_root_a_l50_5050


namespace radius_range_l50_5035

noncomputable def circle_eq (x y r : ℝ) := x^2 + y^2 = r^2

def point_P_on_line_AB (m n : ℝ) := 4 * m + 3 * n - 24 = 0

def point_P_in_interval (m : ℝ) := 0 ≤ m ∧ m ≤ 6

theorem radius_range {r : ℝ} :
  (∀ (m n x y : ℝ), point_P_in_interval m →
     circle_eq x y r →
     circle_eq ((x + m) / 2) ((y + n) / 2) r → 
     point_P_on_line_AB m n ∧
     (4 * r ^ 2 ≤ (25 / 9) * m ^ 2 - (64 / 3) * m + 64 ∧
     (25 / 9) * m ^ 2 - (64 / 3) * m + 64 ≤ 36 * r ^ 2)) →
  (8 / 3 ≤ r ∧ r < 12 / 5) :=
sorry

end radius_range_l50_5035


namespace inverse_variation_l50_5060

theorem inverse_variation (a b : ℝ) (k : ℝ) (h1 : a * b = k) (h2 : 800 * 0.5 = k) (h3 : a = 1600) : b = 0.25 :=
by 
  sorry

end inverse_variation_l50_5060


namespace lambda_phi_relation_l50_5020

-- Define the context and conditions
variables (A B C D M N : Type) -- Points on the triangle with D being the midpoint of BC
variables (AB AC BC BN BM MN : ℝ) -- Lengths
variables (lambda phi : ℝ) -- Ratios given in the problem

-- Conditions
-- 1. M is a point on the median AD of triangle ABC
variable (h1 : M = D ∨ M = A ∨ M = D) -- Simplified condition stating M's location
-- 2. The line BM intersects the side AC at point N
variable (h2 : N = M ∧ N ≠ A ∧ N ≠ C) -- Defining the intersection point
-- 3. AB is tangent to the circumcircle of triangle NBC
variable (h3 : tangent AB (circumcircle N B C))
-- 4. BC = lambda BN
variable (h4 : BC = lambda * BN)
-- 5. BM = phi * MN
variable (h5 : BM = phi * MN)

-- Goal
theorem lambda_phi_relation : phi = lambda ^ 2 :=
sorry

end lambda_phi_relation_l50_5020


namespace quadratic_roots_range_l50_5057

theorem quadratic_roots_range (k : ℝ) : 
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (k * x₁^2 - 4 * x₁ + 1 = 0) ∧ (k * x₂^2 - 4 * x₂ + 1 = 0)) 
  ↔ (k < 4 ∧ k ≠ 0) := 
by
  sorry

end quadratic_roots_range_l50_5057


namespace number_of_true_propositions_is_two_l50_5017

def proposition1 (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (1 + x) = f (1 - x)

def proposition2 : Prop :=
∀ x : ℝ, 2 * Real.sin x * Real.cos (abs x) -- minimum period not 1
  -- We need to define proper periodicity which is complex; so here's a simplified representation
  ≠ 2 * Real.sin (x + 1) * Real.cos (abs (x + 1))

def increasing_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, a (n + 1) > a n

def proposition3 (k : ℝ) : Prop :=
∀ n : ℕ, n > 0 → increasing_sequence (fun n => n^2 + k * n + 2)

def condition (f : ℝ → ℝ) (k : ℝ) : Prop :=
proposition1 f ∧ proposition2 ∧ proposition3 k

theorem number_of_true_propositions_is_two (f : ℝ → ℝ) (k : ℝ) :
  condition f k → 2 = 2 :=
by
  sorry

end number_of_true_propositions_is_two_l50_5017


namespace zoe_bought_8_roses_l50_5086

-- Define the conditions
def each_flower_costs : ℕ := 3
def roses_bought (R : ℕ) : Prop := true
def daisies_bought : ℕ := 2
def total_spent : ℕ := 30

-- The main theorem to prove
theorem zoe_bought_8_roses (R : ℕ) (h1 : total_spent = 30) 
  (h2 : 3 * R + 3 * daisies_bought = total_spent) : R = 8 := by
  sorry

end zoe_bought_8_roses_l50_5086


namespace calculate_paint_area_l50_5087

def barn_length : ℕ := 12
def barn_width : ℕ := 15
def barn_height : ℕ := 6
def window_length : ℕ := 2
def window_width : ℕ := 2

def area_to_paint : ℕ := 796

theorem calculate_paint_area 
    (b_len : ℕ := barn_length) 
    (b_wid : ℕ := barn_width) 
    (b_hei : ℕ := barn_height) 
    (win_len : ℕ := window_length) 
    (win_wid : ℕ := window_width) : 
    b_len = 12 → 
    b_wid = 15 → 
    b_hei = 6 → 
    win_len = 2 → 
    win_wid = 2 →
    area_to_paint = 796 :=
by
  -- Here, the proof would be provided.
  -- This line is a placeholder (sorry) indicating that the proof is yet to be constructed.
  sorry

end calculate_paint_area_l50_5087


namespace stickers_left_correct_l50_5033

-- Define the initial number of stickers and number of stickers given away
def n_initial : ℝ := 39.0
def n_given_away : ℝ := 22.0

-- Proof statement: The number of stickers left at the end is 17.0
theorem stickers_left_correct : n_initial - n_given_away = 17.0 := by
  sorry

end stickers_left_correct_l50_5033


namespace age_difference_l50_5044

variable (a b c : ℕ)

theorem age_difference (h : a + b = b + c + 13) : a - c = 13 :=
by
  sorry

end age_difference_l50_5044


namespace min_value_frac_sum_l50_5070

theorem min_value_frac_sum (m n : ℝ) (h1 : 2 * m + n = 2) (h2 : m * n > 0) : 
  ∃ c : ℝ, c = 4 ∧ (∀ m n, 2 * m + n = 2 → m * n > 0 → (1 / m + 2 / n) ≥ c) :=
sorry

end min_value_frac_sum_l50_5070


namespace trajectory_proof_l50_5096

noncomputable def trajectory_eqn (x y : ℝ) : Prop :=
  (y + Real.sqrt 2) * (y - Real.sqrt 2) / (x * x) = -2

theorem trajectory_proof :
  ∀ (x y : ℝ), x ≠ 0 → trajectory_eqn x y → (y*y / 2 + x*x = 1) :=
by
  intros x y hx htrajectory
  sorry

end trajectory_proof_l50_5096


namespace slope_of_line_l50_5056

theorem slope_of_line : 
  ∀ (x1 y1 x2 y2 : ℝ), 
  x1 = 1 → y1 = 3 → x2 = 6 → y2 = -7 → 
  (x1 ≠ x2) → ((y2 - y1) / (x2 - x1) = -2) :=
by
  intros x1 y1 x2 y2 hx1 hy1 hx2 hy2 hx1_ne_x2
  rw [hx1, hy1, hx2, hy2]
  sorry

end slope_of_line_l50_5056


namespace route_Y_quicker_than_route_X_l50_5051

theorem route_Y_quicker_than_route_X :
    let dist_X := 9  -- distance of Route X in miles
    let speed_X := 45  -- speed of Route X in miles per hour
    let dist_Y := 8  -- total distance of Route Y in miles
    let normal_dist_Y := 6.5  -- normal speed distance of Route Y in miles
    let construction_dist_Y := 1.5  -- construction zone distance of Route Y in miles
    let normal_speed_Y := 50  -- normal speed of Route Y in miles per hour
    let construction_speed_Y := 25  -- construction zone speed of Route Y in miles per hour
    let time_X := (dist_X / speed_X) * 60  -- time for Route X in minutes
    let time_Y1 := (normal_dist_Y / normal_speed_Y) * 60  -- time for normal speed segment of Route Y in minutes
    let time_Y2 := (construction_dist_Y / construction_speed_Y) * 60  -- time for construction zone segment of Route Y in minutes
    let time_Y := time_Y1 + time_Y2  -- total time for Route Y in minutes
    time_X - time_Y = 0.6 :=  -- the difference in time between Route X and Route Y in minutes
by
  sorry

end route_Y_quicker_than_route_X_l50_5051


namespace three_divides_two_pow_n_plus_one_l50_5058

theorem three_divides_two_pow_n_plus_one (n : ℕ) (hn : n > 0) : 
  (3 ∣ 2^n + 1) ↔ Odd n := 
sorry

end three_divides_two_pow_n_plus_one_l50_5058


namespace cube_difference_l50_5041

variable (x : ℝ)

theorem cube_difference (h : x - 1/x = 5) : x^3 - 1/x^3 = 120 := by
  sorry

end cube_difference_l50_5041


namespace rectangular_prism_lateral_edge_length_l50_5072

-- Definition of the problem conditions
def is_rectangular_prism (v : ℕ) : Prop := v = 8
def sum_lateral_edges (l : ℕ) : ℕ := 4 * l

-- Theorem stating the problem to prove
theorem rectangular_prism_lateral_edge_length :
  ∀ (v l : ℕ), is_rectangular_prism v → sum_lateral_edges l = 56 → l = 14 :=
by
  intros v l h1 h2
  sorry

end rectangular_prism_lateral_edge_length_l50_5072


namespace find_other_number_l50_5039

theorem find_other_number (A B : ℕ) (H1 : Nat.lcm A B = 2310) (H2 : Nat.gcd A B = 30) (H3 : A = 770) : B = 90 :=
  by
  sorry

end find_other_number_l50_5039


namespace perimeter_equal_l50_5069

theorem perimeter_equal (x : ℕ) (hx : x = 4)
    (side_square : ℕ := x + 2) 
    (side_triangle : ℕ := 2 * x) 
    (perimeter_square : ℕ := 4 * side_square)
    (perimeter_triangle : ℕ := 3 * side_triangle) :
    perimeter_square = perimeter_triangle :=
by
    -- Given x = 4
    -- Calculate side lengths
    -- side_square = x + 2 = 4 + 2 = 6
    -- side_triangle = 2 * x = 2 * 4 = 8
    -- Calculate perimeters
    -- perimeter_square = 4 * side_square = 4 * 6 = 24
    -- perimeter_triangle = 3 * side_triangle = 3 * 8 = 24
    -- Therefore, perimeter_square = perimeter_triangle = 24
    sorry

end perimeter_equal_l50_5069


namespace Nicky_wait_time_l50_5010

theorem Nicky_wait_time (x : ℕ) (h1 : x + (4 * x + 14) = 114) : x = 20 :=
by {
  sorry
}

end Nicky_wait_time_l50_5010


namespace minimum_value_l50_5006

noncomputable def min_value_of_expression (a b c : ℝ) : ℝ :=
  1/a + 2/b + 4/c

theorem minimum_value (a b c : ℝ) (h₀ : c > 0) (h₁ : a ≠ 0) (h₂ : b ≠ 0)
    (h₃ : 4 * a^2 - 2 * a * b + b^2 - c = 0)
    (h₄ : ∀ x y, 4*x^2 - 2*x*y + y^2 - c = 0 → |2*x + y| ≤ |2*a + b|)
    : min_value_of_expression a b c = -1 :=
sorry

end minimum_value_l50_5006


namespace problem_statement_l50_5046

variable (p q : ℝ)

def condition := p ^ 2 / q ^ 3 = 4 / 5

theorem problem_statement (hpq : condition p q) : 11 / 7 + (2 * q ^ 3 - p ^ 2) / (2 * q ^ 3 + p ^ 2) = 2 :=
sorry

end problem_statement_l50_5046


namespace compare_negatives_l50_5055

theorem compare_negatives : -3 < -2 := 
by { sorry }

end compare_negatives_l50_5055


namespace dime_quarter_problem_l50_5061

theorem dime_quarter_problem :
  15 * 25 + 10 * 10 = 25 * 25 + 35 * 10 :=
by
  sorry

end dime_quarter_problem_l50_5061


namespace solve_custom_eq_l50_5042

-- Define the custom operation a * b = ab + a + b, we will use ∗ instead of * to avoid confusion with multiplication

def custom_op (a b : Nat) : Nat := a * b + a + b

-- State the problem in Lean 4
theorem solve_custom_eq (x : Nat) : custom_op 3 x = 27 → x = 6 :=
by
  sorry

end solve_custom_eq_l50_5042


namespace find_y_l50_5029

theorem find_y (x y : ℤ) (h1 : x + 2 * y = 100) (h2 : x = 50) : y = 25 :=
by
  sorry

end find_y_l50_5029


namespace Sherman_weekly_driving_time_l50_5027

theorem Sherman_weekly_driving_time (daily_commute : ℕ := 30) (weekend_drive : ℕ := 2) : 
  (5 * (2 * daily_commute) / 60 + 2 * weekend_drive) = 9 := 
by
  sorry

end Sherman_weekly_driving_time_l50_5027


namespace tickets_left_l50_5007

-- Definitions for the conditions given in the problem
def tickets_whack_a_mole : ℕ := 32
def tickets_skee_ball : ℕ := 25
def tickets_spent_on_hat : ℕ := 7

-- The main proof statement to verify
theorem tickets_left : tickets_whack_a_mole + tickets_skee_ball - tickets_spent_on_hat = 50 :=
by
  sorry

end tickets_left_l50_5007


namespace gcd_A_B_l50_5066

theorem gcd_A_B (a b : ℕ) (h1 : Nat.gcd a b = 1) (h2 : a > 0) (h3 : b > 0) : 
  Nat.gcd (8 * a + 3 * b) (3 * a + 2 * b) ≠ 1 → Nat.gcd (8 * a + 3 * b) (3 * a + 2 * b) = 7 :=
by
  sorry

end gcd_A_B_l50_5066


namespace solid_produces_quadrilateral_l50_5013

-- Define the solids and their properties
inductive Solid 
| cone 
| cylinder 
| sphere

-- Define the condition for a plane cut resulting in a quadrilateral cross-section
def can_produce_quadrilateral_cross_section (s : Solid) : Prop :=
  match s with
  | Solid.cone => False
  | Solid.cylinder => True
  | Solid.sphere => False

-- Theorem to prove that only a cylinder can produce a quadrilateral cross-section
theorem solid_produces_quadrilateral : 
  ∃ s : Solid, can_produce_quadrilateral_cross_section s :=
by
  existsi Solid.cylinder
  trivial

end solid_produces_quadrilateral_l50_5013


namespace darnel_difference_l50_5034

theorem darnel_difference (sprint_1 jog_1 sprint_2 jog_2 sprint_3 jog_3 : ℝ)
  (h_sprint_1 : sprint_1 = 0.8932)
  (h_jog_1 : jog_1 = 0.7683)
  (h_sprint_2 : sprint_2 = 0.9821)
  (h_jog_2 : jog_2 = 0.4356)
  (h_sprint_3 : sprint_3 = 1.2534)
  (h_jog_3 : jog_3 = 0.6549) :
  (sprint_1 + sprint_2 + sprint_3 - (jog_1 + jog_2 + jog_3)) = 1.2699 := by
  sorry

end darnel_difference_l50_5034


namespace power_difference_divisible_l50_5023

-- Define the variables and conditions
variables {a b c : ℤ} {n : ℕ}

-- Condition: a - b is divisible by c
def is_divisible (a b c : ℤ) : Prop := ∃ k : ℤ, a - b = k * c

-- Lean proof statement
theorem power_difference_divisible {a b c : ℤ} {n : ℕ} (h : is_divisible a b c) : c ∣ (a^n - b^n) :=
  sorry

end power_difference_divisible_l50_5023


namespace find_m_values_l50_5032

def is_solution (m : ℝ) : Prop :=
  let A : Set ℝ := {1, -2}
  let B : Set ℝ := {x | m * x + 1 = 0}
  B ⊆ A

theorem find_m_values :
  {m : ℝ | is_solution m} = {0, -1, 1 / 2} :=
by
  sorry

end find_m_values_l50_5032


namespace determine_a_l50_5091

def quadratic_condition (a : ℝ) (x : ℝ) : Prop := 
  abs (x^2 + 2 * a * x + 3 * a) ≤ 2

theorem determine_a : {a : ℝ | ∃! x : ℝ, quadratic_condition a x} = {1, 2} :=
sorry

end determine_a_l50_5091


namespace solution_for_x_l50_5016

theorem solution_for_x : ∀ (x : ℚ), (10 * x^2 + 9 * x - 2 = 0) ∧ (30 * x^2 + 59 * x - 6 = 0) → x = 1 / 5 :=
by
  sorry

end solution_for_x_l50_5016


namespace find_number_l50_5083

-- Definitions based on conditions
def condition (x : ℝ) : Prop := (x - 5) / 3 = 4

-- The target theorem to prove
theorem find_number (x : ℝ) (h : condition x) : x = 17 :=
sorry

end find_number_l50_5083


namespace frank_reading_days_l50_5018

-- Define the parameters
def pages_weekdays : ℚ := 5.7
def pages_weekends : ℚ := 9.5
def total_pages : ℚ := 576
def pages_per_week : ℚ := (pages_weekdays * 5) + (pages_weekends * 2)

-- Define the property to be proved
theorem frank_reading_days : 
  (total_pages / pages_per_week).floor * 7 + 
  (total_pages - (total_pages / pages_per_week).floor * pages_per_week) / pages_weekdays 
  = 85 := 
  by
    sorry

end frank_reading_days_l50_5018


namespace corveus_sleep_hours_l50_5001

-- Definition of the recommended hours of sleep per day
def recommended_sleep_per_day : ℕ := 6

-- Definition of the hours of sleep Corveus lacks per week
def lacking_sleep_per_week : ℕ := 14

-- Definition of days in a week
def days_in_week : ℕ := 7

-- Prove that Corveus sleeps 4 hours per day given the conditions
theorem corveus_sleep_hours :
  (recommended_sleep_per_day * days_in_week - lacking_sleep_per_week) / days_in_week = 4 :=
by
  -- The proof steps would go here
  sorry

end corveus_sleep_hours_l50_5001


namespace circle_area_percentage_decrease_l50_5031

theorem circle_area_percentage_decrease (r : ℝ) (A : ℝ := Real.pi * r^2) 
  (r' : ℝ := 0.5 * r) (A' : ℝ := Real.pi * (r')^2) :
  (A - A') / A * 100 = 75 := 
by
  sorry

end circle_area_percentage_decrease_l50_5031


namespace sequence_general_term_l50_5085

noncomputable def a₁ : ℕ → ℚ := sorry

variable (S : ℕ → ℚ)

axiom h₀ : a₁ 1 = -1
axiom h₁ : ∀ n : ℕ, a₁ (n + 1) = S n * S (n + 1)

theorem sequence_general_term (n : ℕ) : S n = -1 / n := by
  sorry

end sequence_general_term_l50_5085


namespace taxable_income_l50_5081

theorem taxable_income (tax_paid : ℚ) (state_tax_rate : ℚ) (months_resident : ℚ) (total_months : ℚ) (T : ℚ) :
  tax_paid = 1275 ∧ state_tax_rate = 0.04 ∧ months_resident = 9 ∧ total_months = 12 → 
  T = 42500 :=
by
  intros h
  sorry

end taxable_income_l50_5081


namespace suitableTempForPreservingBoth_l50_5082

-- Definitions for the temperature ranges of types A and B vegetables
def suitableTempRangeA := {t : ℝ | 3 ≤ t ∧ t ≤ 8}
def suitableTempRangeB := {t : ℝ | 5 ≤ t ∧ t ≤ 10}

-- The intersection of the suitable temperature ranges
def suitableTempRangeForBoth := {t : ℝ | 5 ≤ t ∧ t ≤ 8}

-- The theorem statement we need to prove
theorem suitableTempForPreservingBoth :
  suitableTempRangeForBoth = suitableTempRangeA ∩ suitableTempRangeB :=
sorry

end suitableTempForPreservingBoth_l50_5082


namespace sum_of_x_values_l50_5003

theorem sum_of_x_values (x : ℂ) (h₁ : x ≠ -3) (h₂ : 3 = (x^3 - 3 * x^2 - 10 * x) / (x + 3)) : x + (5 - x) = 5 :=
sorry

end sum_of_x_values_l50_5003


namespace coloring_time_saved_percentage_l50_5076

variable (n : ℕ := 10) -- number of pictures
variable (draw_time : ℝ := 2) -- time to draw each picture in hours
variable (total_time : ℝ := 34) -- total time spent on drawing and coloring in hours

/-- 
  Prove the percentage of time saved on coloring each picture compared to drawing 
  given the specified conditions.
-/
theorem coloring_time_saved_percentage (n : ℕ) (draw_time total_time : ℝ) 
  (h1 : draw_time > 0)
  (draw_total_time : draw_time * n = 20)
  (total_picture_time : draw_time * n + coloring_total_time = total_time) :
  (draw_time - (coloring_total_time / n)) / draw_time * 100 = 30 := 
by
  sorry

end coloring_time_saved_percentage_l50_5076


namespace perfect_squares_represented_as_diff_of_consecutive_cubes_l50_5030

theorem perfect_squares_represented_as_diff_of_consecutive_cubes : ∃ (count : ℕ), 
  count = 40 ∧ 
  ∀ n : ℕ, 
  (∃ a : ℕ, a^2 = ( ( n + 1 )^3 - n^3 ) ∧ a^2 < 20000) → count = 40 := by 
sorry

end perfect_squares_represented_as_diff_of_consecutive_cubes_l50_5030


namespace value_of_f_neg_4_l50_5067

noncomputable def f : ℝ → ℝ := λ x => if x ≥ 0 then Real.sqrt x else - (Real.sqrt (-x))

-- Definition that f is an odd function
def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

theorem value_of_f_neg_4 :
  isOddFunction f ∧ (∀ x, x ≥ 0 → f x = Real.sqrt x) → f (-4) = -2 := 
by
  sorry

end value_of_f_neg_4_l50_5067


namespace subcommittee_count_l50_5075

def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem subcommittee_count : 
  let R := 10
  let D := 4
  let subR := 4
  let subD := 2
  binomial R subR * binomial D subD = 1260 := 
by
  sorry

end subcommittee_count_l50_5075


namespace recurring_decimal_sum_l50_5021

noncomputable def x : ℚ := 1 / 3

noncomputable def y : ℚ := 14 / 999

noncomputable def z : ℚ := 5 / 9999

theorem recurring_decimal_sum :
  x + y + z = 3478 / 9999 := by
  sorry

end recurring_decimal_sum_l50_5021


namespace problem_l50_5015

noncomputable def f : ℝ → ℝ := sorry 

theorem problem
  (h_even : ∀ x : ℝ, f x = f (-x))
  (h_func : ∀ x : ℝ, f (2 + x) = -f (2 - x))
  (h_value : f (-3) = -2) :
  f 2007 = 2 :=
sorry

end problem_l50_5015


namespace value_of_expression_l50_5037

theorem value_of_expression
  (a b x y : ℝ)
  (h1 : a + b = 0)
  (h2 : x * y = 1) : 
  2 * (a + b) + (7 / 4) * (x * y) = 7 / 4 := 
sorry

end value_of_expression_l50_5037


namespace placements_for_nine_squares_l50_5025

-- Define the parameters and conditions of the problem
def countPlacements (n : ℕ) : ℕ := sorry

theorem placements_for_nine_squares : countPlacements 9 = 25 := sorry

end placements_for_nine_squares_l50_5025


namespace cannot_be_square_difference_l50_5063

def square_difference_formula (a b : ℝ) : ℝ := a^2 - b^2

def expression_A (x : ℝ) : ℝ := (x + 1) * (x - 1)
def expression_B (x : ℝ) : ℝ := (-x + 1) * (-x - 1)
def expression_C (x : ℝ) : ℝ := (x + 1) * (-x + 1)
def expression_D (x : ℝ) : ℝ := (x + 1) * (1 + x)

theorem cannot_be_square_difference (x : ℝ) : 
  ¬ (∃ a b, (x + 1) * (1 + x) = square_difference_formula a b) := 
sorry

end cannot_be_square_difference_l50_5063


namespace slices_per_banana_l50_5012

-- Define conditions
def yogurts : ℕ := 5
def slices_per_yogurt : ℕ := 8
def bananas : ℕ := 4
def total_slices_needed : ℕ := yogurts * slices_per_yogurt

-- Statement to prove
theorem slices_per_banana : total_slices_needed / bananas = 10 := by sorry

end slices_per_banana_l50_5012


namespace evaluate_expression_l50_5084

theorem evaluate_expression : 150 * (150 - 5) - (150 * 150 - 7) = -743 :=
by
  sorry

end evaluate_expression_l50_5084


namespace exists_m_n_l50_5014

theorem exists_m_n (p : ℕ) (hp : p > 10) [hp_prime : Fact (Nat.Prime p)] :
  ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ m + n < p ∧ (5^m * 7^n - 1) % p = 0 :=
sorry

end exists_m_n_l50_5014


namespace geometric_sequence_solution_l50_5038

-- Assume we have a type for real numbers
variable {R : Type} [LinearOrderedField R]

theorem geometric_sequence_solution (a b c : R)
  (h1 : -1 ≠ 0) (h2 : a ≠ 0) (h3 : b ≠ 0) (h4 : c ≠ 0) (h5 : -9 ≠ 0)
  (h : ∃ r : R, r ≠ 0 ∧ (a = r * -1) ∧ (b = r * a) ∧ (c = r * b) ∧ (-9 = r * c)) :
  b = -3 ∧ a * c = 9 := by
  sorry

end geometric_sequence_solution_l50_5038


namespace math_problem_proof_l50_5080

-- Define the fractions involved
def frac1 : ℚ := -49
def frac2 : ℚ := 4 / 7
def frac3 : ℚ := -8 / 7

-- The original expression
def original_expr : ℚ :=
  frac1 * frac2 - frac2 / frac3

-- Declare the theorem to be proved
theorem math_problem_proof : original_expr = -27.5 :=
by
  sorry

end math_problem_proof_l50_5080


namespace tan_2theta_l50_5068

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x + Real.cos x

theorem tan_2theta (θ : ℝ) (h : ∀ x, f x ≤ f θ) : Real.tan (2 * θ) = -4 / 3 := 
by 
  sorry

end tan_2theta_l50_5068


namespace find_altitude_to_hypotenuse_l50_5048

-- define the conditions
def area : ℝ := 540
def hypotenuse : ℝ := 36
def altitude : ℝ := 30

-- define the problem statement
theorem find_altitude_to_hypotenuse (A : ℝ) (c : ℝ) (h : ℝ) 
  (h_area : A = 540) (h_hypotenuse : c = 36) : h = 30 :=
by
  -- skipping the proof
  sorry

end find_altitude_to_hypotenuse_l50_5048


namespace range_of_function_l50_5040

noncomputable def function_range: Set ℝ :=
  { y | ∃ x, y = (1/2)^(x^2 - 2*x + 2) }

theorem range_of_function :
  function_range = {y | 0 < y ∧ y ≤ 1/2} :=
sorry

end range_of_function_l50_5040


namespace prove_smallest_solution_l50_5065

noncomputable def smallest_solution : ℝ :=
  if h : 0 ≤ (3 - Real.sqrt 17) / 2 then min ((3 - Real.sqrt 17) / 2) 1
  else (3 - Real.sqrt 17) / 2  -- Assumption as sqrt(17) > 3, so (3 - sqrt(17))/2 < 0

theorem prove_smallest_solution :
  ∃ x : ℝ, (x * |x| = 3 * x - 2) ∧ 
           (∀ y : ℝ, (y * |y| = 3 * y - 2) → x ≤ y) ∧
           x = (3 - Real.sqrt 17) / 2 :=
sorry

end prove_smallest_solution_l50_5065


namespace range_of_a_l50_5019

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x - 3| + |x - a| ≥ 3) ↔ a ≤ 0 ∨ a ≥ 6 :=
by
  sorry

end range_of_a_l50_5019


namespace product_469157_9999_l50_5059

theorem product_469157_9999 : 469157 * 9999 = 4690872843 := by
  -- computation and its proof would go here
  sorry

end product_469157_9999_l50_5059


namespace smallest_even_integer_l50_5092

theorem smallest_even_integer :
  ∃ (x : ℤ), |3 * x - 4| ≤ 20 ∧ (∀ (y : ℤ), |3 * y - 4| ≤ 20 → (2 ∣ y) → x ≤ y) ∧ (2 ∣ x) :=
by
  use -4
  sorry

end smallest_even_integer_l50_5092


namespace no_two_digit_factorization_2023_l50_5028

theorem no_two_digit_factorization_2023 :
  ¬ ∃ (a b : ℕ), 10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a * b = 2023 := 
by
  sorry

end no_two_digit_factorization_2023_l50_5028


namespace find_numbers_l50_5090

noncomputable def sum_nat (a b : ℕ) : ℕ :=
  (b - a + 1) * (a + b) / 2

theorem find_numbers : 
  {n : ℕ | ∃ (a b : ℕ), 1 ≤ a ∧ a < b ∧ b ≤ 9 ∧ n = 10 * a + b ∧ n = sum_nat a b} = {14, 26, 37, 48, 59} :=
by {
  sorry
}

end find_numbers_l50_5090


namespace perimeter_C_is_74_l50_5000

/-- Definitions of side lengths based on given perimeters -/
def side_length_A (p_A : ℕ) : ℕ :=
  p_A / 4

def side_length_B (p_B : ℕ) : ℕ :=
  p_B / 4

/-- Definition of side length of C in terms of side lengths of A and B -/
def side_length_C (s_A s_B : ℕ) : ℚ :=
  (s_A : ℚ) / 2 + 2 * (s_B : ℚ)

/-- Definition of perimeter in terms of side length -/
def perimeter (s : ℚ) : ℚ :=
  4 * s

/-- Theorem statement: the perimeter of square C is 74 -/
theorem perimeter_C_is_74 (p_A p_B : ℕ) (h₁ : p_A = 20) (h₂ : p_B = 32) :
  perimeter (side_length_C (side_length_A p_A) (side_length_B p_B)) = 74 := by
  sorry

end perimeter_C_is_74_l50_5000


namespace equation_holds_iff_b_eq_c_l50_5071

theorem equation_holds_iff_b_eq_c (a b c : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : c < 10) (h4 : a ≠ b) (h5 : a ≠ c) (h6 : b ≠ c) :
  (10 * a + b + 1) * (10 * a + c) = 100 * a * a + 100 * a + b + c ↔ b = c :=
by sorry

end equation_holds_iff_b_eq_c_l50_5071


namespace surface_area_of_sphere_l50_5052

theorem surface_area_of_sphere (V : ℝ) (hV : V = 72 * π) : 
  ∃ A : ℝ, A = 36 * π * (2^(2/3)) := by 
  sorry

end surface_area_of_sphere_l50_5052


namespace geometric_sequence_sum_S6_l50_5045

theorem geometric_sequence_sum_S6 (S : ℕ → ℝ) (S_2_eq_4 : S 2 = 4) (S_4_eq_16 : S 4 = 16) :
  S 6 = 52 :=
sorry

end geometric_sequence_sum_S6_l50_5045


namespace leon_required_score_l50_5049

noncomputable def leon_scores : List ℕ := [72, 68, 75, 81, 79]

theorem leon_required_score (n : ℕ) :
  (List.sum leon_scores + n) / (List.length leon_scores + 1) ≥ 80 ↔ n ≥ 105 :=
by sorry

end leon_required_score_l50_5049


namespace minimum_value_l50_5079

noncomputable def expr (x y : ℝ) := x^2 + x * y + y^2 - 3 * y

theorem minimum_value :
  ∃ x y : ℝ, expr x y = -3 ∧
  ∀ x' y' : ℝ, expr x' y' ≥ -3 :=
sorry

end minimum_value_l50_5079


namespace find_abc_integers_l50_5054

theorem find_abc_integers (a b c : ℕ) (h1 : 1 < a) (h2 : a < b) (h3 : b < c) 
(h4 : (a - 1) * (b - 1) * (c - 1) ∣ a * b * c - 1) : (a = 3 ∧ b = 5 ∧ c = 15) ∨ 
(a = 2 ∧ b = 4 ∧ c = 8) :=
sorry

end find_abc_integers_l50_5054
