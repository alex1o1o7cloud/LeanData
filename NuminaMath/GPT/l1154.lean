import Mathlib

namespace NUMINAMATH_GPT_dominoes_per_player_l1154_115488

-- Define the conditions
def total_dominoes : ℕ := 28
def number_of_players : ℕ := 4

-- The theorem
theorem dominoes_per_player : total_dominoes / number_of_players = 7 :=
by sorry

end NUMINAMATH_GPT_dominoes_per_player_l1154_115488


namespace NUMINAMATH_GPT_red_balloon_is_one_l1154_115400

open Nat

theorem red_balloon_is_one (R B : Nat) (h1 : R + B = 85) (h2 : R ≥ 1) (h3 : ∀ i j, i < R → j < R → i ≠ j → (i < B ∨ j < B)) : R = 1 :=
by
  sorry

end NUMINAMATH_GPT_red_balloon_is_one_l1154_115400


namespace NUMINAMATH_GPT_jackson_chairs_l1154_115443

theorem jackson_chairs (a b c d : ℕ) (h1 : a = 6) (h2 : b = 4) (h3 : c = 12) (h4 : d = 6) : a * b + c * d = 96 := 
by sorry

end NUMINAMATH_GPT_jackson_chairs_l1154_115443


namespace NUMINAMATH_GPT_no_common_points_l1154_115458

theorem no_common_points 
  (x x_o y y_o : ℝ) 
  (h_parabola : y^2 = 4 * x) 
  (h_inside : y_o^2 < 4 * x_o) : 
  ¬ ∃ (x y : ℝ), y * y_o = 2 * (x + x_o) ∧ y^2 = 4 * x :=
by
  sorry

end NUMINAMATH_GPT_no_common_points_l1154_115458


namespace NUMINAMATH_GPT_find_C_coordinates_l1154_115471

open Real

noncomputable def coordC (A B : ℝ × ℝ) : ℝ × ℝ :=
  let n := A.1
  let m := B.1
  let coord_n_y : ℝ := n
  let coord_m_y : ℝ := m
  let y_value (x : ℝ) : ℝ := sqrt 3 / x
  (sqrt 3 / 2, 2)

theorem find_C_coordinates :
  ∃ C : ℝ × ℝ, 
  (∃ A B : ℝ × ℝ, 
   A.2 = sqrt 3 / A.1 ∧
   B.2 = sqrt 3 / B.1 + 6 ∧
   A.2 + 6 = B.2 ∧
   B.2 > A.2 ∧ 
   (sqrt 3 / 2, 2) = coordC A B) ∧
   (sqrt 3 / 2, 2) = (C.1, C.2) :=
by
  sorry

end NUMINAMATH_GPT_find_C_coordinates_l1154_115471


namespace NUMINAMATH_GPT_ratio_red_to_green_apple_l1154_115492

def total_apples : ℕ := 496
def green_apples : ℕ := 124
def red_apples : ℕ := total_apples - green_apples

theorem ratio_red_to_green_apple :
  red_apples / green_apples = 93 / 31 :=
by
  sorry

end NUMINAMATH_GPT_ratio_red_to_green_apple_l1154_115492


namespace NUMINAMATH_GPT_value_of_a_set_of_x_l1154_115495

open Real

noncomputable def f (x a : ℝ) : ℝ := sin (x + π / 6) + sin (x - π / 6) + cos x + a

theorem value_of_a : ∀ a, (∀ x, f x a ≤ 1) → a = -1 :=
sorry

theorem set_of_x (a : ℝ) (k : ℤ) : a = -1 →
  {x : ℝ | f x a = 0} = {x | ∃ k : ℤ, x = 2 * k * π ∨ x = 2 * k * π + 2 * π / 3} :=
sorry

end NUMINAMATH_GPT_value_of_a_set_of_x_l1154_115495


namespace NUMINAMATH_GPT_man_overtime_hours_correctness_l1154_115473

def man_worked_overtime_hours (r h_r t : ℕ): ℕ :=
  let regular_pay := r * h_r
  let overtime_pay := t - regular_pay
  let overtime_rate := 2 * r
  overtime_pay / overtime_rate

theorem man_overtime_hours_correctness : man_worked_overtime_hours 3 40 186 = 11 := by
  sorry

end NUMINAMATH_GPT_man_overtime_hours_correctness_l1154_115473


namespace NUMINAMATH_GPT_smallest_lcm_for_80k_quadruples_l1154_115447

-- Declare the gcd and lcm functions for quadruples
def gcd_quad (a b c d : ℕ) : ℕ := Nat.gcd (Nat.gcd a b) (Nat.gcd c d)
def lcm_quad (a b c d : ℕ) : ℕ := Nat.lcm (Nat.lcm a b) (Nat.lcm c d)

-- Main statement we need to prove
theorem smallest_lcm_for_80k_quadruples :
  ∃ m : ℕ, (∃ (a b c d : ℕ), gcd_quad a b c d = 100 ∧ lcm_quad a b c d = m) ∧
    (∀ m', m' < m → ¬ (∃ (a' b' c' d' : ℕ), gcd_quad a' b' c' d' = 100 ∧ lcm_quad a' b' c' d' = m')) ∧
    m = 2250000 :=
sorry

end NUMINAMATH_GPT_smallest_lcm_for_80k_quadruples_l1154_115447


namespace NUMINAMATH_GPT_pyramid_new_volume_l1154_115459

-- Define constants
def V : ℝ := 100
def l : ℝ := 3
def w : ℝ := 2
def h : ℝ := 1.20

-- Define the theorem
theorem pyramid_new_volume : (l * w * h) * V = 720 := by
  sorry -- Proof is skipped

end NUMINAMATH_GPT_pyramid_new_volume_l1154_115459


namespace NUMINAMATH_GPT_parenthesis_removal_correctness_l1154_115420

theorem parenthesis_removal_correctness (x y z : ℝ) : 
  (x^2 - (x - y + 2 * z) ≠ x^2 - x + y - 2 * z) ∧
  (x - (-2 * x + 3 * y - 1) ≠ x + 2 * x - 3 * y + 1) ∧
  (3 * x + 2 * (x - 2 * y + 1) ≠ 3 * x + 2 * x - 4 * y + 2) ∧
  (-(x - 2) - 2 * (x^2 + 2) = -x + 2 - 2 * x^2 - 4) :=
by
  sorry

end NUMINAMATH_GPT_parenthesis_removal_correctness_l1154_115420


namespace NUMINAMATH_GPT_expression_evaluation_l1154_115494

theorem expression_evaluation : (50 + 12) ^ 2 - (12 ^ 2 + 50 ^ 2) = 1200 := 
by
  sorry

end NUMINAMATH_GPT_expression_evaluation_l1154_115494


namespace NUMINAMATH_GPT_radius_for_visibility_l1154_115467

def is_concentric (hex_center : ℝ × ℝ) (circle_center : ℝ × ℝ) : Prop :=
  hex_center = circle_center

def regular_hexagon (side_length : ℝ) : Prop :=
  side_length = 3

theorem radius_for_visibility
  (r : ℝ)
  (hex_center : ℝ × ℝ)
  (circle_center : ℝ × ℝ)
  (P_visible: ℝ)
  (prob_Four_sides_visible: ℝ ) :
  is_concentric hex_center circle_center →
  regular_hexagon 3 →
  prob_Four_sides_visible = 1 / 3 →
  P_visible = 4 →
  r = 2.6 :=
by sorry

end NUMINAMATH_GPT_radius_for_visibility_l1154_115467


namespace NUMINAMATH_GPT_vertex_on_x_axis_l1154_115442

theorem vertex_on_x_axis (m : ℝ) : 
  (∃ x : ℝ, x^2 - 8 * x + m = 0) ↔ m = 16 :=
by
  sorry

end NUMINAMATH_GPT_vertex_on_x_axis_l1154_115442


namespace NUMINAMATH_GPT_fully_loaded_truck_weight_l1154_115413

def empty_truck : ℕ := 12000
def weight_soda_crate : ℕ := 50
def num_soda_crates : ℕ := 20
def weight_dryer : ℕ := 3000
def num_dryers : ℕ := 3
def weight_fresh_produce : ℕ := 2 * (weight_soda_crate * num_soda_crates)

def total_loaded_truck_weight : ℕ :=
  empty_truck + (weight_soda_crate * num_soda_crates) + weight_fresh_produce + (weight_dryer * num_dryers)

theorem fully_loaded_truck_weight : total_loaded_truck_weight = 24000 := by
  sorry

end NUMINAMATH_GPT_fully_loaded_truck_weight_l1154_115413


namespace NUMINAMATH_GPT_increasing_on_interval_of_m_l1154_115417

def f (m x : ℝ) := 2 * x^3 - 3 * m * x^2 + 6 * x

theorem increasing_on_interval_of_m (m : ℝ) :
  (∀ x : ℝ, 2 < x → 6 * x^2 - 6 * m * x + 6 ≥ 0) → m ≤ 5 / 2 :=
sorry

end NUMINAMATH_GPT_increasing_on_interval_of_m_l1154_115417


namespace NUMINAMATH_GPT_difference_in_spectators_l1154_115477

-- Define the parameters given in the problem
def people_game_2 : ℕ := 80
def people_game_1 : ℕ := people_game_2 - 20
def people_game_3 : ℕ := people_game_2 + 15
def people_last_week : ℕ := 200

-- Total people who watched the games this week
def people_this_week : ℕ := people_game_1 + people_game_2 + people_game_3

-- Theorem statement: Prove the difference in people watching the games between this week and last week is 35.
theorem difference_in_spectators : people_this_week - people_last_week = 35 :=
  sorry

end NUMINAMATH_GPT_difference_in_spectators_l1154_115477


namespace NUMINAMATH_GPT_max_distance_proof_l1154_115405

-- Definitions for fuel consumption rates per 100 km
def fuel_consumption_U : Nat := 20 -- liters per 100 km
def fuel_consumption_V : Nat := 25 -- liters per 100 km
def fuel_consumption_W : Nat := 5  -- liters per 100 km
def fuel_consumption_X : Nat := 10 -- liters per 100 km

-- Definitions for total available fuel
def total_fuel : Nat := 50 -- liters

-- Distance calculation
def distance (fuel_consumption : Nat) (fuel : Nat) : Nat :=
  (fuel * 100) / fuel_consumption

-- Distances
def distance_U := distance fuel_consumption_U total_fuel
def distance_V := distance fuel_consumption_V total_fuel
def distance_W := distance fuel_consumption_W total_fuel
def distance_X := distance fuel_consumption_X total_fuel

-- Maximum total distance calculation
def maximum_total_distance : Nat :=
  distance_U + distance_V + distance_W + distance_X

-- The statement to be proved
theorem max_distance_proof :
  maximum_total_distance = 1950 := by
  sorry

end NUMINAMATH_GPT_max_distance_proof_l1154_115405


namespace NUMINAMATH_GPT_rooster_count_l1154_115482

theorem rooster_count (total_chickens hens roosters : ℕ) 
  (h1 : total_chickens = roosters + hens)
  (h2 : roosters = 2 * hens)
  (h3 : total_chickens = 9000) 
  : roosters = 6000 := 
by
  sorry

end NUMINAMATH_GPT_rooster_count_l1154_115482


namespace NUMINAMATH_GPT_apples_fallen_l1154_115412

theorem apples_fallen (H1 : ∃ ground_apples : ℕ, ground_apples = 10 + 3)
                      (H2 : ∃ tree_apples : ℕ, tree_apples = 5)
                      (H3 : ∃ total_apples : ℕ, total_apples = ground_apples ∧ total_apples = 10 + 3 + 5)
                      : ∃ fallen_apples : ℕ, fallen_apples = 13 :=
by
  sorry

end NUMINAMATH_GPT_apples_fallen_l1154_115412


namespace NUMINAMATH_GPT_sum_of_squares_positive_l1154_115485

theorem sum_of_squares_positive (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a * b * c < 0) : 
  (a^2 + b^2 > 0) ∧ (b^2 + c^2 > 0) ∧ (c^2 + a^2 > 0) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_positive_l1154_115485


namespace NUMINAMATH_GPT_trajectory_eq_l1154_115445

theorem trajectory_eq (M : Type) [MetricSpace M] : 
  (∀ (r x y : ℝ), (x + 2)^2 + y^2 = (r + 1)^2 ∧ |x - 1| = 1 → y^2 = -8 * x) :=
by sorry

end NUMINAMATH_GPT_trajectory_eq_l1154_115445


namespace NUMINAMATH_GPT_problem1_correct_problem2_correct_l1154_115466

noncomputable def problem1 : ℚ :=
  (1/2 - 5/9 + 7/12) * (-36)

theorem problem1_correct : problem1 = -19 := 
by 
  sorry

noncomputable def mixed_number (a : ℤ) (b : ℚ) : ℚ := a + b

noncomputable def problem2 : ℚ :=
  (mixed_number (-199) (24/25)) * 5

theorem problem2_correct : problem2 = -999 - 4/5 :=
by
  sorry

end NUMINAMATH_GPT_problem1_correct_problem2_correct_l1154_115466


namespace NUMINAMATH_GPT_range_of_a_l1154_115479

theorem range_of_a (a : ℝ) : 
  ((-1 + a) ^ 2 + (-1 - a) ^ 2 < 4) ↔ (-1 < a ∧ a < 1) := 
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1154_115479


namespace NUMINAMATH_GPT_greatest_value_y_l1154_115419

theorem greatest_value_y (y : ℝ) (hy : 11 = y^2 + 1/y^2) : y + 1/y ≤ Real.sqrt 13 :=
sorry

end NUMINAMATH_GPT_greatest_value_y_l1154_115419


namespace NUMINAMATH_GPT_sum_of_four_digit_integers_up_to_4999_l1154_115404

theorem sum_of_four_digit_integers_up_to_4999 : 
  let a := 1000
  let l := 4999
  let n := l - a + 1
  let S := (n / 2) * (a + l)
  S = 11998000 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_four_digit_integers_up_to_4999_l1154_115404


namespace NUMINAMATH_GPT_stewart_farm_sheep_count_l1154_115464

theorem stewart_farm_sheep_count
  (ratio : ℕ → ℕ → Prop)
  (S H : ℕ)
  (ratio_S_H : ratio S H)
  (one_sheep_seven_horses : ratio 1 7)
  (food_per_horse : ℕ)
  (total_food : ℕ)
  (food_per_horse_val : food_per_horse = 230)
  (total_food_val : total_food = 12880)
  (calc_horses : H = total_food / food_per_horse)
  (calc_sheep : S = H / 7) :
  S = 8 :=
by {
  /- Given the conditions, we need to show that S = 8 -/
  sorry
}

end NUMINAMATH_GPT_stewart_farm_sheep_count_l1154_115464


namespace NUMINAMATH_GPT_decreasing_y_as_x_increases_l1154_115422

theorem decreasing_y_as_x_increases :
  (∀ x1 x2, x1 < x2 → (-2 * x1 + 1) > (-2 * x2 + 1)) ∧
  ¬ (∀ x1 x2, x1 < x2 → (x1^2 + 1) > (x2^2 + 1)) ∧
  ¬ (∀ x1 x2, x1 < x2 → (-x1^2 + 1) > (-x2^2 + 1)) ∧
  ¬ (∀ x1 x2, x1 < x2 → (2 * x1 + 1) > (2 * x2 + 1)) :=
by
  sorry

end NUMINAMATH_GPT_decreasing_y_as_x_increases_l1154_115422


namespace NUMINAMATH_GPT_range_G_l1154_115480

noncomputable def G (x : ℝ) : ℝ := |x + 2| - 2 * |x - 2|

theorem range_G : Set.range G = Set.Icc (-8 : ℝ) 8 := sorry

end NUMINAMATH_GPT_range_G_l1154_115480


namespace NUMINAMATH_GPT_total_selection_methods_l1154_115461

-- Define the students and days
inductive Student
| S1 | S2 | S3 | S4 | S5

inductive Day
| Wednesday | Thursday | Friday | Saturday | Sunday

-- The condition where S1 cannot be on Saturday and S2 cannot be on Sunday
def valid_arrangement (arrangement : Day → Student) : Prop :=
  arrangement Day.Saturday ≠ Student.S1 ∧
  arrangement Day.Sunday ≠ Student.S2

-- The main statement
theorem total_selection_methods : ∃ (arrangement_count : ℕ), 
  arrangement_count = 78 ∧
  ∀ (arrangement : Day → Student), valid_arrangement arrangement → 
  arrangement_count = 78 :=
sorry

end NUMINAMATH_GPT_total_selection_methods_l1154_115461


namespace NUMINAMATH_GPT_servings_of_popcorn_l1154_115406

theorem servings_of_popcorn (popcorn_per_serving : ℕ) (jared_consumption : ℕ)
    (friend_consumption : ℕ) (num_friends : ℕ) :
    popcorn_per_serving = 30 →
    jared_consumption = 90 →
    friend_consumption = 60 →
    num_friends = 3 →
    (jared_consumption + num_friends * friend_consumption) / popcorn_per_serving = 9 := 
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_servings_of_popcorn_l1154_115406


namespace NUMINAMATH_GPT_value_of_a_l1154_115462

theorem value_of_a (a : ℝ) (h : 3 ∈ ({1, a, a - 2} : Set ℝ)) : a = 5 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_l1154_115462


namespace NUMINAMATH_GPT_volume_correct_l1154_115407

-- Define the structure and conditions
structure Point where
  x : ℝ
  y : ℝ

def is_on_circle (C : Point) (P : Point) : Prop :=
  (P.x - C.x)^2 + (P.y - C.y)^2 = 25

def volume_of_solid_of_revolution (P A B : Point) : ℝ := sorry

noncomputable def main : ℝ :=
  volume_of_solid_of_revolution {x := 2, y := -8} {x := 4.58, y := -1.98} {x := -3.14, y := -3.91}

theorem volume_correct :
  main = 672.1 := by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_volume_correct_l1154_115407


namespace NUMINAMATH_GPT_triangular_prism_skew_pair_count_l1154_115478

-- Definition of a triangular prism with 6 vertices and 15 lines through any two vertices
structure TriangularPrism :=
  (vertices : Fin 6)   -- 6 vertices
  (lines : Fin 15)     -- 15 lines through any two vertices

-- A function to check if two lines are skew lines 
-- (not intersecting and not parallel in three-dimensional space)
def is_skew (line1 line2 : Fin 15) : Prop := sorry

-- Function to count pairs of lines that are skew in a triangular prism
def count_skew_pairs (prism : TriangularPrism) : Nat := sorry

-- Theorem stating the number of skew pairs in a triangular prism is 36
theorem triangular_prism_skew_pair_count (prism : TriangularPrism) :
  count_skew_pairs prism = 36 := 
sorry

end NUMINAMATH_GPT_triangular_prism_skew_pair_count_l1154_115478


namespace NUMINAMATH_GPT_triangle_equilateral_l1154_115440

noncomputable def is_equilateral (a b c : ℝ) (A B C : ℝ) : Prop :=
  A = 60 ∧ B = 60 ∧ C = 60 ∧ a = b ∧ b = c

theorem triangle_equilateral (A B C a b c : ℝ) (hB : B = 60) (hb : b^2 = a * c) :
  is_equilateral a b c A B C :=
by
  sorry

end NUMINAMATH_GPT_triangle_equilateral_l1154_115440


namespace NUMINAMATH_GPT_michael_truck_meet_once_l1154_115418

noncomputable def meets_count (michael_speed : ℕ) (pail_distance : ℕ) (truck_speed : ℕ) (truck_stop_duration : ℕ) : ℕ :=
  if michael_speed = 4 ∧ pail_distance = 300 ∧ truck_speed = 8 ∧ truck_stop_duration = 45 then 1 else sorry

theorem michael_truck_meet_once :
  meets_count 4 300 8 45 = 1 :=
by simp [meets_count]

end NUMINAMATH_GPT_michael_truck_meet_once_l1154_115418


namespace NUMINAMATH_GPT_number_of_true_propositions_l1154_115439

def inverse_proposition (x y : ℝ) : Prop :=
  ¬(x + y = 0 → (x ≠ -y))

def contrapositive_proposition (a b : ℝ) : Prop :=
  (a^2 ≤ b^2) → (a ≤ b)

def negation_proposition (x : ℝ) : Prop :=
  (x ≤ -3) → ¬(x^2 + x - 6 > 0)

theorem number_of_true_propositions : 
  (∃ (x y : ℝ), inverse_proposition x y) ∧
  (∃ (a b : ℝ), contrapositive_proposition a b) ∧
  ¬(∃ (x : ℝ), negation_proposition x) → 
  2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_number_of_true_propositions_l1154_115439


namespace NUMINAMATH_GPT_no_four_consecutive_product_square_l1154_115448

/-- Prove that there do not exist four consecutive positive integers whose product is a perfect square. -/
theorem no_four_consecutive_product_square :
  ¬ ∃ (x : ℕ), ∃ (n : ℕ), n * n = x * (x + 1) * (x + 2) * (x + 3) :=
sorry

end NUMINAMATH_GPT_no_four_consecutive_product_square_l1154_115448


namespace NUMINAMATH_GPT_div_sqrt3_mul_inv_sqrt3_eq_one_l1154_115460

theorem div_sqrt3_mul_inv_sqrt3_eq_one :
  (3 / Real.sqrt 3) * (1 / Real.sqrt 3) = 1 :=
by
  sorry

end NUMINAMATH_GPT_div_sqrt3_mul_inv_sqrt3_eq_one_l1154_115460


namespace NUMINAMATH_GPT_evaluate_cyclotomic_sum_l1154_115487

theorem evaluate_cyclotomic_sum : 
  (Complex.I ^ 1520 + Complex.I ^ 1521 + Complex.I ^ 1522 + Complex.I ^ 1523 + Complex.I ^ 1524 = 2) :=
by sorry

end NUMINAMATH_GPT_evaluate_cyclotomic_sum_l1154_115487


namespace NUMINAMATH_GPT_quadratic_roots_sum_product_l1154_115444

theorem quadratic_roots_sum_product : 
  ∃ x1 x2 : ℝ, (x1^2 - 2*x1 - 4 = 0) ∧ (x2^2 - 2*x2 - 4 = 0) ∧ 
  (x1 ≠ x2) ∧ (x1 + x2 + x1 * x2 = -2) :=
sorry

end NUMINAMATH_GPT_quadratic_roots_sum_product_l1154_115444


namespace NUMINAMATH_GPT_total_hovering_time_is_24_hours_l1154_115468

-- Define the initial conditions
def mountain_time_day1 : ℕ := 3
def central_time_day1 : ℕ := 4
def eastern_time_day1 : ℕ := 2

-- Define the additional time hovered in each zone on the second day
def additional_time_per_zone_day2 : ℕ := 2

-- Calculate the total time spent on each day
def total_time_day1 : ℕ := mountain_time_day1 + central_time_day1 + eastern_time_day1
def total_additional_time_day2 : ℕ := 3 * additional_time_per_zone_day2 -- there are three zones
def total_time_day2 : ℕ := total_time_day1 + total_additional_time_day2

-- Calculate the total time over the two days
def total_time_two_days : ℕ := total_time_day1 + total_time_day2

-- Prove that the total time over the two days is 24 hours
theorem total_hovering_time_is_24_hours : total_time_two_days = 24 := by
  sorry

end NUMINAMATH_GPT_total_hovering_time_is_24_hours_l1154_115468


namespace NUMINAMATH_GPT_solve_abcd_l1154_115411

theorem solve_abcd : 
  (∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 → |4 * x^3 - d * x| ≤ 1) ∧ 
  (∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 → |4 * x^3 + a * x^2 + b * x + c| ≤ 1) →
  d = 3 ∧ b = -3 ∧ a = 0 ∧ c = 0 :=
by
  sorry

end NUMINAMATH_GPT_solve_abcd_l1154_115411


namespace NUMINAMATH_GPT_additional_land_cost_l1154_115451

noncomputable def initial_land := 300
noncomputable def final_land := 900
noncomputable def cost_per_square_meter := 20

theorem additional_land_cost : (final_land - initial_land) * cost_per_square_meter = 12000 :=
by
  -- Define the amount of additional land purchased
  let additional_land := final_land - initial_land
  -- Calculate the cost of the additional land            
  show additional_land * cost_per_square_meter = 12000
  sorry

end NUMINAMATH_GPT_additional_land_cost_l1154_115451


namespace NUMINAMATH_GPT_vector_expression_l1154_115441

-- Define the vectors a, b, and c
def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (1, -1)
def c : ℝ × ℝ := (-1, -2)

-- The target relationship
theorem vector_expression :
  c = (- (3 / 2) • a + (1 / 2) • b) :=
sorry

end NUMINAMATH_GPT_vector_expression_l1154_115441


namespace NUMINAMATH_GPT_diapers_per_pack_l1154_115416

def total_boxes := 30
def packs_per_box := 40
def price_per_diaper := 5
def total_revenue := 960000

def total_packs_per_week := total_boxes * packs_per_box
def total_diapers_sold := total_revenue / price_per_diaper

theorem diapers_per_pack :
  total_diapers_sold / total_packs_per_week = 160 :=
by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_diapers_per_pack_l1154_115416


namespace NUMINAMATH_GPT_salary_increase_percentage_l1154_115450

theorem salary_increase_percentage (old_salary new_salary : ℕ) (h1 : old_salary = 10000) (h2 : new_salary = 10200) : 
    ((new_salary - old_salary) / old_salary : ℚ) * 100 = 2 := 
by 
  sorry

end NUMINAMATH_GPT_salary_increase_percentage_l1154_115450


namespace NUMINAMATH_GPT_rope_length_l1154_115497

theorem rope_length (x S : ℝ) (H1 : x + 7 * S = 140)
(H2 : x - S = 20) : x = 35 := by
sorry

end NUMINAMATH_GPT_rope_length_l1154_115497


namespace NUMINAMATH_GPT_arithmetic_square_root_16_l1154_115452

theorem arithmetic_square_root_16 : Real.sqrt 16 = 4 := by
  sorry

end NUMINAMATH_GPT_arithmetic_square_root_16_l1154_115452


namespace NUMINAMATH_GPT_remainder_div_14_l1154_115410

variables (x k : ℕ)

theorem remainder_div_14 (h : x = 142 * k + 110) : x % 14 = 12 := by 
  sorry

end NUMINAMATH_GPT_remainder_div_14_l1154_115410


namespace NUMINAMATH_GPT_train_speed_l1154_115454

-- Definition of the problem
def train_length : ℝ := 350
def time_to_cross_man : ℝ := 4.5
def expected_speed : ℝ := 77.78

-- Theorem statement
theorem train_speed :
  train_length / time_to_cross_man = expected_speed :=
sorry

end NUMINAMATH_GPT_train_speed_l1154_115454


namespace NUMINAMATH_GPT_regression_line_fits_l1154_115429

variables {x y : ℝ}

def points := [(1, 2), (2, 5), (4, 7), (5, 10)]

def regression_line (x : ℝ) : ℝ := x + 3

theorem regression_line_fits :
  (∀ p ∈ points, regression_line p.1 = p.2) ∧ (regression_line 3 = 6) :=
by
  sorry

end NUMINAMATH_GPT_regression_line_fits_l1154_115429


namespace NUMINAMATH_GPT_inscribed_triangle_perimeter_geq_half_l1154_115438

theorem inscribed_triangle_perimeter_geq_half (a : ℝ) (s' : ℝ) (h_a_pos : a > 0) 
  (h_equilateral : ∀ (A B C : Type) (a b c : A), a = b ∧ b = c ∧ c = a) :
  2 * s' >= (3 * a) / 2 :=
by
  sorry

end NUMINAMATH_GPT_inscribed_triangle_perimeter_geq_half_l1154_115438


namespace NUMINAMATH_GPT_log_eighteen_fifteen_l1154_115463

noncomputable def log_base (b x : ℝ) := Real.log x / Real.log b

theorem log_eighteen_fifteen (a b : ℝ) (h1 : Real.log 2 = a) (h2 : Real.log 3 = b) :
  log_base 18 15 = (b - a + 1) / (a + 2 * b) :=
by sorry

end NUMINAMATH_GPT_log_eighteen_fifteen_l1154_115463


namespace NUMINAMATH_GPT_area_of_sector_l1154_115403

def radius : ℝ := 5
def central_angle : ℝ := 2

theorem area_of_sector : (1 / 2) * radius^2 * central_angle = 25 := by
  sorry

end NUMINAMATH_GPT_area_of_sector_l1154_115403


namespace NUMINAMATH_GPT_num_baskets_l1154_115421

axiom num_apples_each_basket : ℕ
axiom total_apples : ℕ

theorem num_baskets (h1 : num_apples_each_basket = 17) (h2 : total_apples = 629) : total_apples / num_apples_each_basket = 37 :=
  sorry

end NUMINAMATH_GPT_num_baskets_l1154_115421


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1154_115499

theorem solution_set_of_inequality :
  { x : ℝ | x^2 + x - 2 < 0 } = { x : ℝ | -2 < x ∧ x < 1 } :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1154_115499


namespace NUMINAMATH_GPT_thief_speed_is_43_75_l1154_115434

-- Given Information
def speed_owner : ℝ := 50
def time_head_start : ℝ := 0.5
def total_time_to_overtake : ℝ := 4

-- Question: What is the speed of the thief's car v?
theorem thief_speed_is_43_75 (v : ℝ) (hv : 4 * v = speed_owner * (total_time_to_overtake - time_head_start)) : v = 43.75 := 
by {
  -- The proof of this theorem is omitted as it is not required.
  sorry
}

end NUMINAMATH_GPT_thief_speed_is_43_75_l1154_115434


namespace NUMINAMATH_GPT_smallest_integer_solution_system_of_inequalities_solution_l1154_115453

-- Define the conditions and problem
variable (x : ℝ)

-- Part 1: Prove smallest integer solution for 5x + 15 > x - 1
theorem smallest_integer_solution :
  5 * x + 15 > x - 1 → x = -3 := sorry

-- Part 2: Prove solution set for system of inequalities
theorem system_of_inequalities_solution :
  (-3 * (x - 2) ≥ 4 - x) ∧ ((1 + 4 * x) / 3 > x - 1) → (-4 < x ∧ x ≤ 1) := sorry

end NUMINAMATH_GPT_smallest_integer_solution_system_of_inequalities_solution_l1154_115453


namespace NUMINAMATH_GPT_find_x_squared_perfect_square_l1154_115469

theorem find_x_squared_perfect_square (n m : ℕ) (h1 : 0 < n) (h2 : 0 < m) (h3 : n ≠ m)
  (h4 : n > m) (h5 : n % 2 ≠ m % 2) : 
  ∃ x : ℤ, x = 0 ∧ ∀ x, (x = 0) → ∃ k : ℕ, (x ^ (2 ^ n) - 1) / (x ^ (2 ^ m) - 1) = k^2 :=
sorry

end NUMINAMATH_GPT_find_x_squared_perfect_square_l1154_115469


namespace NUMINAMATH_GPT_find_percentage_l1154_115489

/-- 
Given some percentage P of 6,000, when subtracted from 1/10th of 6,000 (which is 600), 
the difference is 693. Prove that P equals 1.55.
-/
theorem find_percentage (P : ℝ) (h₁ : 6000 / 10 = 600) (h₂ : 600 - (P / 100) * 6000 = 693) : 
  P = 1.55 :=
  sorry

end NUMINAMATH_GPT_find_percentage_l1154_115489


namespace NUMINAMATH_GPT_marbles_total_is_260_l1154_115402

/-- Define the number of marbles in each jar. -/
def jar1 : ℕ := 80
def jar2 : ℕ := 2 * jar1
def jar3 : ℕ := jar1 / 4

/-- The total number of marbles Courtney has. -/
def total_marbles : ℕ := jar1 + jar2 + jar3

/-- Proving that the total number of marbles is 260. -/
theorem marbles_total_is_260 : total_marbles = 260 := 
by
  sorry

end NUMINAMATH_GPT_marbles_total_is_260_l1154_115402


namespace NUMINAMATH_GPT_quadratic_solutions_l1154_115456

theorem quadratic_solutions (x : ℝ) : x^2 - 2 * x = 0 ↔ x = 0 ∨ x = 2 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_solutions_l1154_115456


namespace NUMINAMATH_GPT_find_rectangle_area_l1154_115484

noncomputable def rectangle_area (a b : ℕ) : ℕ :=
  a * b

theorem find_rectangle_area (a b : ℕ) :
  (5 : ℚ) / 8 = (a : ℚ) / b ∧ (a + 6) * (b + 6) - a * b = 114 ∧ a + b = 13 →
  rectangle_area a b = 40 :=
by
  sorry

end NUMINAMATH_GPT_find_rectangle_area_l1154_115484


namespace NUMINAMATH_GPT_mary_paid_amount_l1154_115409

-- Definitions for the conditions:
def is_adult (person : String) : Prop := person = "Mary"
def children_count (n : ℕ) : Prop := n = 3
def ticket_cost_adult : ℕ := 2  -- $2 for adults
def ticket_cost_child : ℕ := 1  -- $1 for children
def change_received : ℕ := 15   -- $15 change

-- Mathematical proof to find the amount Mary paid given the conditions
theorem mary_paid_amount (person : String) (n : ℕ) 
  (h1 : is_adult person) (h2 : children_count n) :
  ticket_cost_adult + ticket_cost_child * n + change_received = 20 := 
by 
  -- Sorry as the proof is not required
  sorry

end NUMINAMATH_GPT_mary_paid_amount_l1154_115409


namespace NUMINAMATH_GPT_exists_marked_sum_of_three_l1154_115435

theorem exists_marked_sum_of_three (s : Finset ℕ) (h₀ : s.card = 22) (h₁ : ∀ x ∈ s, x ≤ 30) :
  ∃ a ∈ s, ∃ b ∈ s, ∃ c ∈ s, ∃ d ∈ s, a = b + c + d :=
by
  sorry

end NUMINAMATH_GPT_exists_marked_sum_of_three_l1154_115435


namespace NUMINAMATH_GPT_remainder_theorem_example_l1154_115415

def polynomial (x : ℝ) : ℝ := x^15 + 3

theorem remainder_theorem_example :
  polynomial (-2) = -32765 :=
by
  -- Substitute x = -2 in the polynomial and show the remainder is -32765
  sorry

end NUMINAMATH_GPT_remainder_theorem_example_l1154_115415


namespace NUMINAMATH_GPT_time_for_q_to_complete_work_alone_l1154_115449

theorem time_for_q_to_complete_work_alone (P Q : ℝ) (h1 : (1 / P) + (1 / Q) = 1 / 40) (h2 : (20 / P) + (12 / Q) = 1) : Q = 64 / 3 :=
by
  sorry

end NUMINAMATH_GPT_time_for_q_to_complete_work_alone_l1154_115449


namespace NUMINAMATH_GPT_expected_points_experts_over_100_games_probability_of_envelope_five_selected_l1154_115424

-- Game conditions and probabilities
def game_conditions (experts_points audience_points : ℕ) : Prop :=
  experts_points = 6 ∨ audience_points = 6

noncomputable def equal_teams := (1 : ℝ) / 2

-- Expected score of Experts over 100 games
noncomputable def expected_points_experts (games : ℕ) := 465

-- Probability that envelope number 5 is chosen in the next game
noncomputable def probability_envelope_five := (12 : ℝ) / 13

theorem expected_points_experts_over_100_games : 
  expected_points_experts 100 = 465 := 
sorry

theorem probability_of_envelope_five_selected : 
  probability_envelope_five = 0.715 := 
sorry

end NUMINAMATH_GPT_expected_points_experts_over_100_games_probability_of_envelope_five_selected_l1154_115424


namespace NUMINAMATH_GPT_no_nontrivial_solutions_l1154_115427

theorem no_nontrivial_solutions :
  ∀ (x y z t : ℤ), (¬(x = 0 ∧ y = 0 ∧ z = 0 ∧ t = 0)) → ¬(x^2 = 2 * y^2 ∧ x^4 + 3 * y^4 + 27 * z^4 = 9 * t^4) :=
by
  intros x y z t h_nontrivial h_eqs
  sorry

end NUMINAMATH_GPT_no_nontrivial_solutions_l1154_115427


namespace NUMINAMATH_GPT_sara_sent_letters_l1154_115486

theorem sara_sent_letters (J : ℕ)
  (h1 : 9 + 3 * J + J = 33) : J = 6 :=
by
  sorry

end NUMINAMATH_GPT_sara_sent_letters_l1154_115486


namespace NUMINAMATH_GPT_parallel_vectors_determine_t_l1154_115491

theorem parallel_vectors_determine_t (t : ℝ) (h : (t, -6) = (k * -3, k * 2)) : t = 9 :=
by
  sorry

end NUMINAMATH_GPT_parallel_vectors_determine_t_l1154_115491


namespace NUMINAMATH_GPT_Smith_gave_Randy_l1154_115423

theorem Smith_gave_Randy {original_money Randy_keeps gives_Sally Smith_gives : ℕ}
  (h1: original_money = 3000)
  (h2: Randy_keeps = 2000)
  (h3: gives_Sally = 1200)
  (h4: Randy_keeps + gives_Sally = original_money + Smith_gives) :
  Smith_gives = 200 :=
by
  sorry

end NUMINAMATH_GPT_Smith_gave_Randy_l1154_115423


namespace NUMINAMATH_GPT_neg_p_l1154_115483

-- Proposition p : For any x in ℝ, cos x ≤ 1
def p : Prop := ∀ (x : ℝ), Real.cos x ≤ 1

-- Negation of p: There exists an x₀ in ℝ such that cos x₀ > 1
theorem neg_p : ¬p ↔ (∃ (x₀ : ℝ), Real.cos x₀ > 1) := sorry

end NUMINAMATH_GPT_neg_p_l1154_115483


namespace NUMINAMATH_GPT_sector_area_l1154_115428

theorem sector_area (theta : ℝ) (d : ℝ) (r : ℝ := d / 2) (circle_area : ℝ := π * r^2) 
    (sector_area : ℝ := (theta / 360) * circle_area) : 
  theta = 120 → d = 6 → sector_area = 3 * π :=
by
  intro htheta hd
  sorry

end NUMINAMATH_GPT_sector_area_l1154_115428


namespace NUMINAMATH_GPT_polygon_largest_area_l1154_115474

-- Definition for the area calculation of each polygon based on given conditions
def area_A : ℝ := 3 * 1 + 2 * 0.5
def area_B : ℝ := 6 * 1
def area_C : ℝ := 4 * 1 + 3 * 0.5
def area_D : ℝ := 5 * 1 + 1 * 0.5
def area_E : ℝ := 7 * 1

-- Theorem stating the problem
theorem polygon_largest_area :
  area_E = max (max (max (max area_A area_B) area_C) area_D) area_E :=
by
  -- The proof steps would go here.
  sorry

end NUMINAMATH_GPT_polygon_largest_area_l1154_115474


namespace NUMINAMATH_GPT_remainder_sum_l1154_115430

-- Define the conditions given in the problem.
def remainder_13_mod_5 : ℕ := 3
def remainder_12_mod_5 : ℕ := 2
def remainder_11_mod_5 : ℕ := 1

theorem remainder_sum :
  ((13 ^ 6 + 12 ^ 7 + 11 ^ 8) % 5) = 3 := by
  sorry

end NUMINAMATH_GPT_remainder_sum_l1154_115430


namespace NUMINAMATH_GPT_trader_gain_pens_l1154_115446

theorem trader_gain_pens (C S : ℝ) (h1 : S = 1.25 * C) 
                         (h2 : 80 * S = 100 * C) : S - C = 0.25 * C :=
by
  have h3 : S = 1.25 * C := h1
  have h4 : 80 * S = 100 * C := h2
  sorry

end NUMINAMATH_GPT_trader_gain_pens_l1154_115446


namespace NUMINAMATH_GPT_correct_proposition_D_l1154_115408

theorem correct_proposition_D (a b : ℝ) (h1 : a < 0) (h2 : b < 0) : 
  (b / a) + (a / b) ≥ 2 := 
sorry

end NUMINAMATH_GPT_correct_proposition_D_l1154_115408


namespace NUMINAMATH_GPT_find_v1_l1154_115475

def u (x : ℝ) : ℝ := 4 * x - 9

def v (y : ℝ) : ℝ := y^2 + 4 * y - 5

theorem find_v1 : v 1 = 11.25 := by
  sorry

end NUMINAMATH_GPT_find_v1_l1154_115475


namespace NUMINAMATH_GPT_consecutive_squares_not_arithmetic_sequence_l1154_115472

theorem consecutive_squares_not_arithmetic_sequence (x y z w : ℕ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z ∧ 0 < w)
  (h_order: x < y ∧ y < z ∧ z < w) :
  ¬ (∃ d : ℕ, y^2 = x^2 + d ∧ z^2 = y^2 + d ∧ w^2 = z^2 + d) :=
sorry

end NUMINAMATH_GPT_consecutive_squares_not_arithmetic_sequence_l1154_115472


namespace NUMINAMATH_GPT_price_increase_problem_l1154_115496

variable (P P' x : ℝ)

theorem price_increase_problem
  (h1 : P' = P * (1 + x / 100))
  (h2 : P = P' * (1 - 23.076923076923077 / 100)) :
  x = 30 :=
by
  sorry

end NUMINAMATH_GPT_price_increase_problem_l1154_115496


namespace NUMINAMATH_GPT_geo_arith_sequences_sum_first_2n_terms_l1154_115401

variables (n : ℕ)

-- Given conditions in (a)
def common_ratio : ℕ := 3
def arithmetic_diff : ℕ := 2

-- The sequences provided in the solution (b)
def a_n (n : ℕ) : ℕ := common_ratio ^ n
def b_n (n : ℕ) : ℕ := 2 * n + 1

-- Sum formula for geometric series up to 2n terms
def S_2n (n : ℕ) : ℕ := (common_ratio^(2 * n + 1) - common_ratio) / 2 + 2 * n

theorem geo_arith_sequences :
  a_n n = common_ratio ^ n
  ∨ b_n n = 2 * n + 1 := sorry

theorem sum_first_2n_terms :
  S_2n n = (common_ratio^(2 * n + 1) - common_ratio) / 2 + 2 * n := sorry

end NUMINAMATH_GPT_geo_arith_sequences_sum_first_2n_terms_l1154_115401


namespace NUMINAMATH_GPT_largest_divisor_of_product_of_five_consecutive_integers_l1154_115431

theorem largest_divisor_of_product_of_five_consecutive_integers
  (a b c d e : ℤ) 
  (h: a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e):
  ∃ (n : ℤ), n = 60 ∧ n ∣ (a * b * c * d * e) :=
sorry

end NUMINAMATH_GPT_largest_divisor_of_product_of_five_consecutive_integers_l1154_115431


namespace NUMINAMATH_GPT_sequence_terms_distinct_l1154_115436

theorem sequence_terms_distinct (n m : ℕ) (hnm : n ≠ m) : 
  (n / (n + 1) : ℚ) ≠ (m / (m + 1) : ℚ) :=
sorry

end NUMINAMATH_GPT_sequence_terms_distinct_l1154_115436


namespace NUMINAMATH_GPT_domain_log_base_2_l1154_115465

theorem domain_log_base_2 (x : ℝ) : (1 - x > 0) ↔ (x < 1) := by
  sorry

end NUMINAMATH_GPT_domain_log_base_2_l1154_115465


namespace NUMINAMATH_GPT_cryptarithm_solutions_unique_l1154_115490

/- Definitions corresponding to the conditions -/
def is_valid_digit (d : Nat) : Prop := d < 10

def is_six_digit_number (n : Nat) : Prop := n >= 100000 ∧ n < 1000000

def matches_cryptarithm (abcdef bcdefa : Nat) : Prop := abcdef * 3 = bcdefa

/- Prove that the two identified solutions are valid and no other solutions exist -/
theorem cryptarithm_solutions_unique :
  ∀ (A B C D E F : Nat),
  is_valid_digit A → is_valid_digit B → is_valid_digit C →
  is_valid_digit D → is_valid_digit E → is_valid_digit F →
  let abcdef := 100000 * A + 10000 * B + 1000 * C + 100 * D + 10 * E + F
  let bcdefa := 100000 * B + 10000 * C + 1000 * D + 100 * E + 10 * F + A
  is_six_digit_number abcdef →
  is_six_digit_number bcdefa →
  matches_cryptarithm abcdef bcdefa →
  (abcdef = 142857 ∨ abcdef = 285714) :=
by
  intros A B C D E F A_valid B_valid C_valid D_valid E_valid F_valid abcdef bcdefa abcdef_six_digit bcdefa_six_digit cryptarithm_match
  sorry

end NUMINAMATH_GPT_cryptarithm_solutions_unique_l1154_115490


namespace NUMINAMATH_GPT_simplify_polynomial_l1154_115432

theorem simplify_polynomial (x : ℝ) : 
  (2 * x + 1) ^ 5 - 5 * (2 * x + 1) ^ 4 + 10 * (2 * x + 1) ^ 3 - 10 * (2 * x + 1) ^ 2 + 5 * (2 * x + 1) - 1 = 32 * x ^ 5 := 
by 
  sorry

end NUMINAMATH_GPT_simplify_polynomial_l1154_115432


namespace NUMINAMATH_GPT_solve_system_l1154_115493

theorem solve_system (x₁ x₂ x₃ : ℝ) (h₁ : 2 * x₁^2 / (1 + x₁^2) = x₂) (h₂ : 2 * x₂^2 / (1 + x₂^2) = x₃) (h₃ : 2 * x₃^2 / (1 + x₃^2) = x₁) :
  (x₁ = 0 ∧ x₂ = 0 ∧ x₃ = 0) ∨ (x₁ = 1 ∧ x₂ = 1 ∧ x₃ = 1) :=
sorry

end NUMINAMATH_GPT_solve_system_l1154_115493


namespace NUMINAMATH_GPT_max_side_length_is_11_l1154_115457

theorem max_side_length_is_11 (a b c : ℕ) (h_perm : a + b + c = 24) (h_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_ineq1 : a + b > c) (h_ineq2 : a + c > b) (h_ineq3 : b + c > a) (h_order : a < b ∧ b < c) : c = 11 :=
by
  sorry

end NUMINAMATH_GPT_max_side_length_is_11_l1154_115457


namespace NUMINAMATH_GPT_vacuum_tube_pins_and_holes_l1154_115414

theorem vacuum_tube_pins_and_holes :
  ∀ (pins holes : Finset ℕ), 
  pins = {1, 2, 3, 4, 5, 6, 7} →
  holes = {1, 2, 3, 4, 5, 6, 7} →
  (∃ (a : ℕ), ∀ k ∈ pins, ∃ b ∈ holes, (2 * k) % 7 = b) := by
  sorry

end NUMINAMATH_GPT_vacuum_tube_pins_and_holes_l1154_115414


namespace NUMINAMATH_GPT_functional_equation_zero_l1154_115437

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation_zero (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x + |y|) = f (|x|) + f (y)) :
  ∀ x : ℝ, f x = 0 :=
by
  sorry

end NUMINAMATH_GPT_functional_equation_zero_l1154_115437


namespace NUMINAMATH_GPT_winning_candidate_percentage_l1154_115425

theorem winning_candidate_percentage
  (votes_candidate1 : ℕ) (votes_candidate2 : ℕ) (votes_candidate3 : ℕ)
  (total_votes : ℕ) (winning_votes : ℕ) (percentage : ℚ)
  (h1 : votes_candidate1 = 1000)
  (h2 : votes_candidate2 = 2000)
  (h3 : votes_candidate3 = 4000)
  (h4 : total_votes = votes_candidate1 + votes_candidate2 + votes_candidate3)
  (h5 : winning_votes = votes_candidate3)
  (h6 : percentage = (winning_votes : ℚ) / total_votes * 100) :
  percentage = 57.14 := 
sorry

end NUMINAMATH_GPT_winning_candidate_percentage_l1154_115425


namespace NUMINAMATH_GPT_product_of_numbers_eq_zero_l1154_115433

theorem product_of_numbers_eq_zero (a b c : ℝ) 
  (h1 : a + b + c = 1) 
  (h2 : a^2 + b^2 + c^2 = 1) 
  (h3 : a^3 + b^3 + c^3 = 1) : 
  a * b * c = 0 := 
by
  sorry

end NUMINAMATH_GPT_product_of_numbers_eq_zero_l1154_115433


namespace NUMINAMATH_GPT_minimum_pie_pieces_l1154_115498

theorem minimum_pie_pieces (p q : ℕ) (h_coprime : Nat.gcd p q = 1) : 
  ∃ n, (∀ k, k = p ∨ k = q → (n ≠ 0 → n % k = 0)) ∧ n = p + q - 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_minimum_pie_pieces_l1154_115498


namespace NUMINAMATH_GPT_last_digit_two_power_2015_l1154_115470

/-- The last digit of powers of 2 cycles through 2, 4, 8, 6. Therefore, the last digit of 2^2015 is the same as 2^3, which is 8. -/
theorem last_digit_two_power_2015 : (2^2015) % 10 = 8 :=
by sorry

end NUMINAMATH_GPT_last_digit_two_power_2015_l1154_115470


namespace NUMINAMATH_GPT_g_at_4_l1154_115481

noncomputable def g : ℝ → ℝ := sorry

axiom g_condition : ∀ x : ℝ, g x + 3 * g (2 - x) = 4 * x^2

theorem g_at_4 : g 4 = -2 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_g_at_4_l1154_115481


namespace NUMINAMATH_GPT_breadth_of_hall_l1154_115455

/-- Given a hall of length 20 meters and a uniform verandah width of 2.5 meters,
    with a cost of Rs. 700 for flooring the verandah at Rs. 3.50 per square meter,
    prove that the breadth of the hall is 15 meters. -/
theorem breadth_of_hall (h_length : ℝ) (v_width : ℝ) (cost : ℝ) (rate : ℝ) (b : ℝ) :
  h_length = 20 ∧ v_width = 2.5 ∧ cost = 700 ∧ rate = 3.50 →
  25 * (b + 5) - 20 * b = 200 →
  b = 15 :=
by
  intros hc ha
  sorry

end NUMINAMATH_GPT_breadth_of_hall_l1154_115455


namespace NUMINAMATH_GPT_balls_picking_l1154_115476

theorem balls_picking (red_bag blue_bag : ℕ) (h_red : red_bag = 3) (h_blue : blue_bag = 5) : (red_bag * blue_bag = 15) :=
by
  sorry

end NUMINAMATH_GPT_balls_picking_l1154_115476


namespace NUMINAMATH_GPT_santino_fruit_total_l1154_115426

-- Definitions of the conditions
def numPapayaTrees : ℕ := 2
def numMangoTrees : ℕ := 3
def papayasPerTree : ℕ := 10
def mangosPerTree : ℕ := 20
def totalFruits (pTrees : ℕ) (pPerTree : ℕ) (mTrees : ℕ) (mPerTree : ℕ) : ℕ :=
  (pTrees * pPerTree) + (mTrees * mPerTree)

-- Theorem that states the total number of fruits is 80 given the conditions
theorem santino_fruit_total : totalFruits numPapayaTrees papayasPerTree numMangoTrees mangosPerTree = 80 := 
  sorry

end NUMINAMATH_GPT_santino_fruit_total_l1154_115426
