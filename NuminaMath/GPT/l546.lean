import Mathlib

namespace obtuse_vertex_angle_is_135_l546_54645

-- Define the obtuse scalene triangle with the given properties
variables {a b c : ℝ} (triangle : Triangle ℝ)
variables (φ : ℝ) (h_obtuse : φ > 90 ∧ φ < 180) (h_scalene : a ≠ b ∧ b ≠ c ∧ c ≠ a)
variables (h_side_relation : a^2 + b^2 = 2 * c^2) (h_sine_obtuse : Real.sin φ = Real.sqrt 2 / 2)

-- The measure of the obtuse vertex angle is 135 degrees
theorem obtuse_vertex_angle_is_135 :
  φ = 135 := by
  sorry

end obtuse_vertex_angle_is_135_l546_54645


namespace math_books_count_l546_54612

theorem math_books_count (M H : ℕ) :
  M + H = 90 →
  4 * M + 5 * H = 396 →
  H = 90 - M →
  M = 54 :=
by
  intro h1 h2 h3
  sorry

end math_books_count_l546_54612


namespace number_of_red_yarns_l546_54606

-- Definitions
def scarves_per_yarn : Nat := 3
def blue_yarns : Nat := 6
def yellow_yarns : Nat := 4
def total_scarves : Nat := 36

-- Theorem
theorem number_of_red_yarns (R : Nat) (H1 : scarves_per_yarn * blue_yarns + scarves_per_yarn * yellow_yarns + scarves_per_yarn * R = total_scarves) :
  R = 2 :=
by
  sorry

end number_of_red_yarns_l546_54606


namespace distance_scientific_notation_l546_54647

theorem distance_scientific_notation :
  55000000 = 5.5 * 10^7 :=
sorry

end distance_scientific_notation_l546_54647


namespace greater_segment_difference_l546_54672

theorem greater_segment_difference :
  ∀ (L1 L2 : ℝ), L1 = 7 ∧ L1^2 - L2^2 = 32 → L1 - L2 = 7 - Real.sqrt 17 :=
by
  intros L1 L2 h
  sorry

end greater_segment_difference_l546_54672


namespace distinct_units_digits_perfect_cube_l546_54688

theorem distinct_units_digits_perfect_cube : 
  (∀ (d : ℕ), d < 10 → ( ∃ (n : ℕ), (n % 10 = d) ∧ (∃ (m : ℕ), (m ^ 3 % 10 = d) ))) ↔
  (∃ (digits : Finset ℕ), digits = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ digits.card = 10) := 
sorry

end distinct_units_digits_perfect_cube_l546_54688


namespace broadcasting_methods_count_l546_54661

-- Defining the given conditions
def num_commercials : ℕ := 4 -- number of different commercial advertisements
def num_psa : ℕ := 2 -- number of different public service advertisements
def total_slots : ℕ := 6 -- total number of slots for commercials

-- The assertion we want to prove
theorem broadcasting_methods_count : 
  (num_psa * (total_slots - num_commercials - 1) * (num_commercials.factorial)) = 48 :=
by sorry

end broadcasting_methods_count_l546_54661


namespace odd_function_periodic_value_l546_54624

noncomputable def f : ℝ → ℝ := sorry  -- Define f

theorem odd_function_periodic_value:
  (∀ x, f (-x) = - f x) →  -- f is odd
  (∀ x, f (x + 3) = f x) → -- f has period 3
  f 1 = 2014 →            -- given f(1) = 2014
  f 2013 + f 2014 + f 2015 = 0 := by
  intros h_odd h_period h_f1
  sorry

end odd_function_periodic_value_l546_54624


namespace roger_total_distance_l546_54632

theorem roger_total_distance :
  let morning_ride_miles := 2
  let evening_ride_miles := 5 * morning_ride_miles
  let next_day_morning_ride_km := morning_ride_miles * 1.6
  let next_day_ride_km := 2 * next_day_morning_ride_km
  let next_day_ride_miles := next_day_ride_km / 1.6
  morning_ride_miles + evening_ride_miles + next_day_ride_miles = 16 :=
by
  sorry

end roger_total_distance_l546_54632


namespace graph_passes_through_point_l546_54616

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x-2) + 1

theorem graph_passes_through_point (a : ℝ) (h_pos : a > 0) (h_neq : a ≠ 1) : f a 2 = 2 :=
by
  sorry

end graph_passes_through_point_l546_54616


namespace divisible_by_91_l546_54658

theorem divisible_by_91 (n : ℕ) : 91 ∣ (5^n * (5^n + 1) - 6^n * (3^n + 2^n)) := 
by 
  sorry

end divisible_by_91_l546_54658


namespace shopkeeper_gain_l546_54679

noncomputable def gain_percent (cost_per_kg : ℝ) (claimed_weight : ℝ) (actual_weight : ℝ) : ℝ :=
  let gain := cost_per_kg - (actual_weight / claimed_weight) * cost_per_kg
  (gain / ((actual_weight / claimed_weight) * cost_per_kg)) * 100

theorem shopkeeper_gain (c : ℝ) (cw aw : ℝ) (h : c = 1) (hw : cw = 1) (ha : aw = 0.75) : 
  gain_percent c cw aw = 33.33 :=
by sorry

end shopkeeper_gain_l546_54679


namespace different_ways_to_eat_spaghetti_l546_54660

-- Define the conditions
def red_spaghetti := 5
def blue_spaghetti := 5
def total_spaghetti := 6

-- This is the proof statement
theorem different_ways_to_eat_spaghetti : 
  ∃ (ways : ℕ), ways = 62 ∧ 
  (∃ r b : ℕ, r ≤ red_spaghetti ∧ b ≤ blue_spaghetti ∧ r + b = total_spaghetti) := 
sorry

end different_ways_to_eat_spaghetti_l546_54660


namespace slices_left_for_tomorrow_is_four_l546_54625

def initial_slices : ℕ := 12
def lunch_slices : ℕ := initial_slices / 2
def remaining_slices_after_lunch : ℕ := initial_slices - lunch_slices
def dinner_slices : ℕ := remaining_slices_after_lunch / 3
def slices_left_for_tomorrow : ℕ := remaining_slices_after_lunch - dinner_slices

theorem slices_left_for_tomorrow_is_four : slices_left_for_tomorrow = 4 := by
  sorry

end slices_left_for_tomorrow_is_four_l546_54625


namespace sin_cos_eq_one_l546_54685

theorem sin_cos_eq_one (x : ℝ) (hx0 : 0 ≤ x) (hx2pi : x < 2 * Real.pi) :
  (Real.sin x - Real.cos x = 1) ↔ (x = Real.pi / 2 ∨ x = Real.pi) :=
by
  sorry

end sin_cos_eq_one_l546_54685


namespace fran_threw_away_80_pct_l546_54634

-- Definitions based on the conditions
def initial_votes_game_of_thrones := 10
def initial_votes_twilight := 12
def initial_votes_art_of_deal := 20
def altered_votes_twilight := initial_votes_twilight / 2
def new_total_votes := 2 * initial_votes_game_of_thrones

-- Theorem we are proving
theorem fran_threw_away_80_pct :
  ∃ x, x = 80 ∧
    new_total_votes = initial_votes_game_of_thrones + altered_votes_twilight + (initial_votes_art_of_deal * (1 - x / 100)) := by
  sorry

end fran_threw_away_80_pct_l546_54634


namespace susie_earnings_l546_54611

-- Define the constants and conditions
def price_per_slice : ℕ := 3
def price_per_whole_pizza : ℕ := 15
def slices_sold : ℕ := 24
def whole_pizzas_sold : ℕ := 3

-- Calculate earnings from slices and whole pizzas
def earnings_from_slices : ℕ := slices_sold * price_per_slice
def earnings_from_whole_pizzas : ℕ := whole_pizzas_sold * price_per_whole_pizza
def total_earnings : ℕ := earnings_from_slices + earnings_from_whole_pizzas

-- Prove that the total earnings are $117
theorem susie_earnings : total_earnings = 117 := by
  sorry

end susie_earnings_l546_54611


namespace range_of_u_l546_54649

def satisfies_condition (x y : ℝ) : Prop := (x^2 / 3) + y^2 = 1

def u (x y : ℝ) : ℝ := |2 * x + y - 4| + |3 - x - 2 * y|

theorem range_of_u {x y : ℝ} (h : satisfies_condition x y) : ∀ u, 1 ≤ u ∧ u ≤ 13 :=
sorry

end range_of_u_l546_54649


namespace pages_read_first_day_l546_54639

-- Alexa is reading a Nancy Drew mystery with 95 pages.
def total_pages : ℕ := 95

-- She read 58 pages the next day.
def pages_read_second_day : ℕ := 58

-- She has 19 pages left to read.
def pages_left_to_read : ℕ := 19

-- How many pages did she read on the first day?
theorem pages_read_first_day : total_pages - pages_read_second_day - pages_left_to_read = 18 := by
  -- Proof is omitted as instructed
  sorry

end pages_read_first_day_l546_54639


namespace pizzeria_large_pizzas_sold_l546_54610

theorem pizzeria_large_pizzas_sold (price_small price_large total_earnings num_small_pizzas num_large_pizzas : ℕ):
  price_small = 2 →
  price_large = 8 →
  total_earnings = 40 →
  num_small_pizzas = 8 →
  total_earnings = price_small * num_small_pizzas + price_large * num_large_pizzas →
  num_large_pizzas = 3 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end pizzeria_large_pizzas_sold_l546_54610


namespace rectangle_ratio_l546_54668

theorem rectangle_ratio (t a b : ℝ) (h₀ : b = 2 * a) (h₁ : (t + 2 * a) ^ 2 = 3 * t ^ 2) : b / a = 2 :=
by
  sorry

end rectangle_ratio_l546_54668


namespace points_on_line_l546_54682

theorem points_on_line (n : ℕ) : 9 * n - 8 = 82 → n = 10 :=
by
  sorry

end points_on_line_l546_54682


namespace find_missing_coordinates_l546_54691

def parallelogram_area (A B : ℝ × ℝ) (C D : ℝ × ℝ) : ℝ :=
  abs ((B.1 - A.1) * (D.2 - A.2))

theorem find_missing_coordinates :
  ∃ (x y : ℝ), (x, y) ≠ (4, 4) ∧ (x, y) ≠ (5, 9) ∧ (x, y) ≠ (8, 9) ∧
  parallelogram_area (4, 4) (5, 9) (8, 9) (x, y) = 5 :=
sorry

end find_missing_coordinates_l546_54691


namespace min_blocks_for_wall_l546_54669

-- Definitions based on conditions
def length_of_wall := 120
def height_of_wall := 6
def block_height := 1
def block_lengths := [1, 3]
def blocks_third_row := 3

-- Function to calculate the total blocks given the constraints from the conditions
noncomputable def min_blocks_needed : Nat := 164 + 80

-- Theorem assertion that the minimum number of blocks required is 244
theorem min_blocks_for_wall : min_blocks_needed = 244 := by
  -- The proof would go here
  sorry

end min_blocks_for_wall_l546_54669


namespace center_number_is_4_l546_54653

-- Define the numbers and the 3x3 grid
inductive Square
| center | top_middle | left_middle | right_middle | bottom_middle

-- Define the properties of the problem
def isConsecutiveAdjacent (a b : ℕ) : Prop := 
  (a + 1 = b ∨ a = b + 1)

-- The condition to check the sum of edge squares
def sum_edge_squares (grid : Square → ℕ) : Prop := 
  grid Square.top_middle + grid Square.left_middle + grid Square.right_middle + grid Square.bottom_middle = 28

-- The condition that the center square number is even
def even_center (grid : Square → ℕ) : Prop := 
  grid Square.center % 2 = 0

-- The main theorem statement
theorem center_number_is_4 (grid : Square → ℕ) :
  (∀ i j : Square, i ≠ j → isConsecutiveAdjacent (grid i) (grid j)) → 
  (grid Square.top_middle + grid Square.left_middle + grid Square.right_middle + grid Square.bottom_middle = 28) →
  (grid Square.center % 2 = 0) →
  grid Square.center = 4 :=
by sorry

end center_number_is_4_l546_54653


namespace range_of_t_l546_54656

noncomputable def f (t : ℝ) (x : ℝ) : ℝ :=
if x ≤ 0 then x^2 + 2 * t * x + t^2 else x + 1 / x + t

theorem range_of_t (t : ℝ) :
  (∀ x : ℝ, f t 0 ≤ f t x) ↔ (0 ≤ t ∧ t ≤ 2) :=
by sorry

end range_of_t_l546_54656


namespace lateral_area_cone_l546_54641

-- Define the cone problem with given conditions
def radius : ℝ := 5
def slant_height : ℝ := 10

-- Given these conditions, prove the lateral area is 50π
theorem lateral_area_cone (r : ℝ) (l : ℝ) (h_r : r = 5) (h_l : l = 10) : (1/2) * 2 * Real.pi * r * l = 50 * Real.pi :=
by 
  -- import useful mathematical tools
  sorry

end lateral_area_cone_l546_54641


namespace moskvich_halfway_from_zhiguli_to_b_l546_54696

-- Define the Moskvich's and Zhiguli's speeds as real numbers
variables (u v : ℝ)

-- Define the given conditions as named hypotheses
axiom speed_condition : u = v
axiom halfway_condition : u = (1 / 2) * (u + v) 

-- The mathematical statement we want to prove
theorem moskvich_halfway_from_zhiguli_to_b (speed_condition : u = v) (halfway_condition : u = (1 / 2) * (u + v)) : 
  ∃ t : ℝ, t = 2 := 
sorry -- Proof omitted

end moskvich_halfway_from_zhiguli_to_b_l546_54696


namespace evaluate_expression_l546_54654

noncomputable def expression (a : ℚ) : ℚ := 
  (a / (a - 1)) / ((a + 1) / (a^2 - 1)) - (1 - 2 * a)

theorem evaluate_expression (a : ℚ) (ha : a = -1/3) : expression a = -2 :=
by 
  rw [expression, ha]
  sorry

end evaluate_expression_l546_54654


namespace sandy_total_money_received_l546_54693

def sandy_saturday_half_dollars := 17
def sandy_sunday_half_dollars := 6
def half_dollar_value : ℝ := 0.50

theorem sandy_total_money_received :
  (sandy_saturday_half_dollars * half_dollar_value) +
  (sandy_sunday_half_dollars * half_dollar_value) = 11.50 :=
by
  sorry

end sandy_total_money_received_l546_54693


namespace part1_part2_l546_54605

-- Proof for part 1
theorem part1 (x : ℤ) : (x - 1 ∣ x - 3 ↔ (x = -1 ∨ x = 0 ∨ x = 2 ∨ x = 3)) :=
by sorry

-- Proof for part 2
theorem part2 (x : ℤ) : (x + 2 ∣ x^2 + 3 ↔ (x = -9 ∨ x = -3 ∨ x = -1 ∨ x = 5)) :=
by sorry

end part1_part2_l546_54605


namespace combined_weight_of_candles_l546_54651

theorem combined_weight_of_candles :
  (∀ (candles: ℕ), ∀ (weight_per_candle: ℕ),
    (weight_per_candle = 8 + 1) →
    (candles = 10 - 3) →
    (candles * weight_per_candle = 63)) :=
by
  intros
  sorry

end combined_weight_of_candles_l546_54651


namespace problem1_problem2_prob_dist_problem2_expectation_l546_54671

noncomputable def probability_A_wins_match_B_wins_once (pA pB : ℚ) : ℚ :=
  (pB * pA * pA) + (pA * pB * pA * pA)

theorem problem1 : probability_A_wins_match_B_wins_once (2/3) (1/3) = 20/81 :=
  by sorry

noncomputable def P_X (x : ℕ) (pA pB : ℚ) : ℚ :=
  match x with
  | 2 => pA^2 + pB^2
  | 3 => pB * pA^2 + pA * pB^2
  | 4 => (pA * pB * pA * pA) + (pB * pA * pB * pB)
  | 5 => (pB * pA * pB * pA) + (pA * pB * pA * pB)
  | _ => 0

theorem problem2_prob_dist : 
  P_X 2 (2/3) (1/3) = 5/9 ∧
  P_X 3 (2/3) (1/3) = 2/9 ∧
  P_X 4 (2/3) (1/3) = 10/81 ∧
  P_X 5 (2/3) (1/3) = 8/81 :=
  by sorry

noncomputable def E_X (pA pB : ℚ) : ℚ :=
  2 * (P_X 2 pA pB) + 3 * (P_X 3 pA pB) + 
  4 * (P_X 4 pA pB) + 5 * (P_X 5 pA pB)

theorem problem2_expectation : E_X (2/3) (1/3) = 224/81 :=
  by sorry

end problem1_problem2_prob_dist_problem2_expectation_l546_54671


namespace polygon_interior_angle_l546_54603

theorem polygon_interior_angle (n : ℕ) (h1 : ∀ (i : ℕ), i < n → (n - 2) * 180 / n = 140): n = 9 := 
sorry

end polygon_interior_angle_l546_54603


namespace obtuse_angle_between_line_and_plane_l546_54652

-- Define the problem conditions
def is_obtuse_angle (θ : ℝ) : Prop := θ > 90 ∧ θ < 180

-- Define what we are proving
theorem obtuse_angle_between_line_and_plane (θ : ℝ) (h1 : θ = angle_between_line_and_plane) :
  is_obtuse_angle θ :=
sorry

end obtuse_angle_between_line_and_plane_l546_54652


namespace circles_5_and_8_same_color_l546_54676

-- Define the circles and colors
inductive Color
  | red
  | yellow
  | blue

def circles : Nat := 8

-- Define the adjacency relationship (i.e., directly connected)
-- This is a placeholder. In practice, this would be defined based on the problem's diagram.
def directly_connected (c1 c2 : Nat) : Prop := sorry

-- Simulate painting circles with given constraints
def painted (c : Nat) : Color := sorry

-- Define the conditions
axiom paint_condition (c1 c2 : Nat) (h : directly_connected c1 c2) : painted c1 ≠ painted c2

-- The proof problem: show that circles 5 and 8 must be painted the same color
theorem circles_5_and_8_same_color : painted 5 = painted 8 := 
sorry

end circles_5_and_8_same_color_l546_54676


namespace remainder_of_division_l546_54619

theorem remainder_of_division (x y R : ℕ) 
  (h1 : y = 1782)
  (h2 : y - x = 1500)
  (h3 : y = 6 * x + R) :
  R = 90 :=
by
  sorry

end remainder_of_division_l546_54619


namespace a_divides_b_l546_54626

theorem a_divides_b (a b : ℕ) (h_pos : 0 < a ∧ 0 < b)
    (h : ∀ n : ℕ, a^n ∣ b^(n+1)) : a ∣ b :=
by
  sorry

end a_divides_b_l546_54626


namespace repeated_digit_in_mod_sequence_l546_54602

theorem repeated_digit_in_mod_sequence : 
  ∃ (x y : ℕ), x ≠ y ∧ (2^1970 % 9 = 4) ∧ 
  (∀ n : ℕ, n < 10 → n = 2^1970 % 9 → n = x ∨ n = y) :=
sorry

end repeated_digit_in_mod_sequence_l546_54602


namespace largest_multiple_of_15_less_than_500_l546_54692

theorem largest_multiple_of_15_less_than_500 : ∃ x : ℕ, x < 500 ∧ x % 15 = 0 ∧ (∀ y : ℕ, y < 500 ∧ y % 15 = 0 → y ≤ x) := sorry

end largest_multiple_of_15_less_than_500_l546_54692


namespace magic_8_ball_probability_l546_54640

def probability_positive (p_pos : ℚ) (questions : ℕ) (positive_responses : ℕ) : ℚ :=
  (Nat.choose questions positive_responses : ℚ) * (p_pos ^ positive_responses) * ((1 - p_pos) ^ (questions - positive_responses))

theorem magic_8_ball_probability :
  probability_positive (1/3) 7 3 = 560 / 2187 :=
by
  sorry

end magic_8_ball_probability_l546_54640


namespace cos_half_pi_plus_alpha_correct_l546_54659

noncomputable def cos_half_pi_plus_alpha
  (α : ℝ)
  (h1 : Real.tan α = -3/4)
  (h2 : α ∈ Set.Ioo (3 * Real.pi / 2) (2 * Real.pi)) : Real :=
  Real.cos (Real.pi / 2 + α)

theorem cos_half_pi_plus_alpha_correct
  (α : ℝ)
  (h1 : Real.tan α = -3/4)
  (h2 : α ∈ Set.Ioo (3 * Real.pi / 2) (2 * Real.pi)) :
  cos_half_pi_plus_alpha α h1 h2 = 3/5 := by
  sorry

end cos_half_pi_plus_alpha_correct_l546_54659


namespace lcm_of_two_numbers_l546_54638

theorem lcm_of_two_numbers (a b : ℕ) (h_ratio : a * 3 = b * 2) (h_sum : a + b = 30) : Nat.lcm a b = 18 :=
  sorry

end lcm_of_two_numbers_l546_54638


namespace factor_poly_PQ_sum_l546_54670

theorem factor_poly_PQ_sum (P Q : ℝ) (h : (∀ x : ℝ, (x^2 + 3 * x + 4) * (x^2 + -3 * x + 4) = x^4 + P * x^2 + Q)) : P + Q = 15 :=
by
  sorry

end factor_poly_PQ_sum_l546_54670


namespace closest_point_on_parabola_to_line_l546_54675

noncomputable def line := { P : ℝ × ℝ | 2 * P.1 - P.2 = 4 }
noncomputable def parabola := { P : ℝ × ℝ | P.2 = P.1^2 }

theorem closest_point_on_parabola_to_line : 
  ∃ P : ℝ × ℝ, P ∈ parabola ∧ 
  (∀ Q ∈ parabola, ∀ R ∈ line, dist P R ≤ dist Q R) ∧ 
  P = (1, 1) := 
sorry

end closest_point_on_parabola_to_line_l546_54675


namespace expression_equals_64_l546_54680

theorem expression_equals_64 :
  let a := 2^3 + 2^3
  let b := 2^3 * 2^3
  let c := (2^3)^3
  let d := 2^12 / 2^2
  b = 2^6 :=
by
  sorry

end expression_equals_64_l546_54680


namespace calculate_remaining_area_l546_54604

/-- In a rectangular plot of land ABCD, where AB = 20 meters and BC = 12 meters, 
    a triangular garden ABE is installed where AE = 15 meters and BE intersects AE at a perpendicular angle, 
    the area of the remaining part of the land which is not occupied by the garden is 150 square meters. -/
theorem calculate_remaining_area 
  (AB BC AE : ℝ) 
  (hAB : AB = 20) 
  (hBC : BC = 12) 
  (hAE : AE = 15)
  (h_perpendicular : true) : -- BE ⊥ AE implying right triangle ABE
  ∃ area_remaining : ℝ, area_remaining = 150 :=
by
  sorry

end calculate_remaining_area_l546_54604


namespace value_of_expression_l546_54636

theorem value_of_expression (x y : ℝ) (h₁ : x = 3) (h₂ : y = 4) :
  (x^3 + 3*y^3) / 9 = 24.33 :=
by
  sorry

end value_of_expression_l546_54636


namespace train_second_speed_20_l546_54607

variable (x v: ℕ)

theorem train_second_speed_20 
  (h1 : (x / 40) + (2 * x / v) = (6 * x / 48)) : 
  v = 20 := by 
  sorry

end train_second_speed_20_l546_54607


namespace max_value_of_expression_l546_54646

theorem max_value_of_expression
  (x y z : ℝ)
  (h : 9 * x^2 + 4 * y^2 + 25 * z^2 = 1) :
  11 * x + 3 * y + 8 * z ≤ 3.1925 :=
sorry

end max_value_of_expression_l546_54646


namespace equal_parts_count_l546_54664

def scale_length_in_inches : ℕ := (7 * 12) + 6
def part_length_in_inches : ℕ := 18
def number_of_parts (total_length part_length : ℕ) : ℕ := total_length / part_length

theorem equal_parts_count :
  number_of_parts scale_length_in_inches part_length_in_inches = 5 :=
by
  sorry

end equal_parts_count_l546_54664


namespace intersection_M_N_l546_54674

-- Definitions for the sets M and N based on the given conditions
def M : Set ℝ := { x | x < 2 }
def N : Set ℝ := { x | x^2 - 3 * x - 4 ≤ 0 }

-- The statement we need to prove
theorem intersection_M_N : M ∩ N = { x | -1 ≤ x ∧ x ≤ 2 } :=
by
  sorry

end intersection_M_N_l546_54674


namespace loaves_of_bread_can_bake_l546_54601

def total_flour_in_cupboard := 200
def total_flour_on_counter := 100
def total_flour_in_pantry := 100
def flour_per_loaf := 200

theorem loaves_of_bread_can_bake :
  (total_flour_in_cupboard + total_flour_on_counter + total_flour_in_pantry) / flour_per_loaf = 2 := by
  sorry

end loaves_of_bread_can_bake_l546_54601


namespace students_in_line_l546_54600

theorem students_in_line (n : ℕ) (h : 1 ≤ n ∧ n ≤ 130) : 
  n = 3 ∨ n = 43 ∨ n = 129 :=
by
  sorry

end students_in_line_l546_54600


namespace maggie_bouncy_balls_l546_54699

theorem maggie_bouncy_balls (yellow_packs green_pack_given green_pack_bought : ℝ)
    (balls_per_pack : ℝ)
    (hy : yellow_packs = 8.0)
    (hg_given : green_pack_given = 4.0)
    (hg_bought : green_pack_bought = 4.0)
    (hbp : balls_per_pack = 10.0) :
    (yellow_packs * balls_per_pack + green_pack_bought * balls_per_pack - green_pack_given * balls_per_pack = 80.0) :=
by
  sorry

end maggie_bouncy_balls_l546_54699


namespace chloe_boxes_of_clothing_l546_54644

theorem chloe_boxes_of_clothing (total_clothing pieces_per_box : ℕ) (h1 : total_clothing = 32) (h2 : pieces_per_box = 2 + 6) :
  ∃ B : ℕ, B = total_clothing / pieces_per_box ∧ B = 4 :=
by
  -- Proof can be filled in here
   sorry

end chloe_boxes_of_clothing_l546_54644


namespace roots_formula_l546_54635

theorem roots_formula (x₁ x₂ p : ℝ)
  (h₁ : x₁ + x₂ = 6 * p)
  (h₂ : x₁ * x₂ = p^2)
  (h₃ : ∀ x, x ^ 2 - 6 * p * x + p ^ 2 = 0 → x = x₁ ∨ x = x₂) :
  (1 / (x₁ + p) + 1 / (x₂ + p) = 1 / p) :=
by
  sorry

end roots_formula_l546_54635


namespace inequality_solution_l546_54695

theorem inequality_solution (x : ℝ) (hx : x ≥ 0) : (2021 * (x ^ 10) - 1 ≥ 2020 * x) ↔ (x = 1) :=
sorry

end inequality_solution_l546_54695


namespace find_missing_number_l546_54615

theorem find_missing_number (x : ℤ) : x + 64 = 16 → x = -48 := by
  intro h
  linarith

end find_missing_number_l546_54615


namespace find_function_expression_l546_54608

noncomputable def f (a b x : ℝ) : ℝ := 2 ^ (a * x + b)

theorem find_function_expression
  (a b : ℝ)
  (h1 : f a b 1 = 2)
  (h2 : ∃ g : ℝ → ℝ, (∀ x y : ℝ, f (-a) (-b) x = y ↔ f a b y = x) ∧ g (f a b 1) = 1) :
  ∃ (a b : ℝ), f a b x = 2 ^ (-x + 2) :=
by
  sorry

end find_function_expression_l546_54608


namespace sum_of_coordinates_B_l546_54662

theorem sum_of_coordinates_B 
  (M : ℝ × ℝ)
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (hM_def : M = (-3, 2))
  (hA_def : A = (-8, 5))
  (hM_midpoint : M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) :
  B.1 + B.2 = 1 := 
sorry

end sum_of_coordinates_B_l546_54662


namespace number_of_liars_l546_54690

theorem number_of_liars {n : ℕ} (h1 : n ≥ 1) (h2 : n ≤ 200) (h3 : ∃ k : ℕ, k < n ∧ k ≥ 1) : 
  (∃ l : ℕ, l = 199 ∨ l = 200) := 
sorry

end number_of_liars_l546_54690


namespace samara_tire_spending_l546_54633

theorem samara_tire_spending :
  ∀ (T : ℕ), 
    (2457 = 25 + 79 + T + 1886) → 
    T = 467 :=
by intros T h
   sorry

end samara_tire_spending_l546_54633


namespace correct_calculation_l546_54621

theorem correct_calculation (x : ℝ) : (-2 * x^2)^3 = -8 * x^6 :=
by sorry

end correct_calculation_l546_54621


namespace range_of_x1_f_x2_l546_54678

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x + 4 * Real.exp 1 else Real.exp x / x^2

theorem range_of_x1_f_x2:
  ∃ (x1 x2 : ℝ), x1 ≤ 0 ∧ 0 < x2 ∧ f x1 = f x2 ∧ -4 * (Real.exp 1)^2 ≤ x1 * f x2 ∧ x1 * f x2 ≤ 0 :=
sorry

end range_of_x1_f_x2_l546_54678


namespace number_of_palindromes_divisible_by_6_l546_54637

theorem number_of_palindromes_divisible_by_6 :
  let is_palindrome (n : ℕ) : Prop := n / 1000 = n % 10 ∧ (n / 100 % 10) = (n / 10 % 10)
  let valid_digits (n : ℕ) : Prop := n >= 1000 ∧ n < 10000
  let divisible_6 (n : ℕ) : Prop := n % 6 = 0
  (Finset.filter (λ n => is_palindrome n ∧ valid_digits n ∧ divisible_6 n) (Finset.range 10000)).card = 13 :=
by
  -- We define what it means to be a palindrome between 1000 and 10000
  let is_palindrome (n : ℕ) : Prop := n / 1000 = n % 10 ∧ n / 100 % 10 = n / 10 % 10
  
  -- We define a valid number between 1000 and 10000
  let valid_digits (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000
  
  -- We define what it means to be divisible by 6
  let divisible_6 (n : ℕ) : Prop := n % 6 = 0

  -- Filtering the range 10000 within valid four-digit palindromes and checking for multiples of 6
  exact sorry

end number_of_palindromes_divisible_by_6_l546_54637


namespace prism_edges_l546_54655

theorem prism_edges (V F E n : ℕ) (h1 : V + F + E = 44) (h2 : V = 2 * n) (h3 : F = n + 2) (h4 : E = 3 * n) : E = 21 := by
  sorry

end prism_edges_l546_54655


namespace length_of_bridge_l546_54643

/-- What is the length of a bridge (in meters), which a train 156 meters long and travelling at 45 km/h can cross in 40 seconds? -/
theorem length_of_bridge (train_length: ℕ) (train_speed_kmh: ℕ) (time_seconds: ℕ) (bridge_length: ℕ) :
  train_length = 156 →
  train_speed_kmh = 45 →
  time_seconds = 40 →
  bridge_length = 344 :=
by {
  sorry
}

end length_of_bridge_l546_54643


namespace find_y_l546_54620

theorem find_y : ∃ y : ℚ, y + 2/3 = 1/4 - (2/5) * 2 ∧ y = -511/420 :=
by
  sorry

end find_y_l546_54620


namespace find_dividend_l546_54657

noncomputable def divisor := (-14 : ℚ) / 3
noncomputable def quotient := (-286 : ℚ) / 5
noncomputable def remainder := (19 : ℚ) / 9
noncomputable def dividend := 269 + (2 / 45 : ℚ)

theorem find_dividend :
  dividend = (divisor * quotient) + remainder := by
  sorry

end find_dividend_l546_54657


namespace yard_length_calculation_l546_54694

theorem yard_length_calculation (n_trees : ℕ) (distance : ℕ) (h1 : n_trees = 26) (h2 : distance = 32) : (n_trees - 1) * distance = 800 :=
by
  -- This is where the proof would go.
  sorry

end yard_length_calculation_l546_54694


namespace value_of_x_minus_y_squared_l546_54618

theorem value_of_x_minus_y_squared (x y : ℝ) (hx : x^2 = 4) (hy : y^2 = 9) : 
  ((x - y)^2 = 1) ∨ ((x - y)^2 = 25) :=
sorry

end value_of_x_minus_y_squared_l546_54618


namespace middle_number_is_eight_l546_54642

theorem middle_number_is_eight
    (x y z : ℕ)
    (h1 : x + y = 14)
    (h2 : x + z = 20)
    (h3 : y + z = 22) :
    y = 8 := by
  sorry

end middle_number_is_eight_l546_54642


namespace number_of_men_l546_54666

theorem number_of_men (M : ℕ) (h : M * 40 = 20 * 68) : M = 34 :=
by
  sorry

end number_of_men_l546_54666


namespace harry_terry_difference_l546_54609

theorem harry_terry_difference :
  let H := 8 - (2 + 5)
  let T := 8 - 2 + 5
  H - T = -10 :=
by 
  sorry

end harry_terry_difference_l546_54609


namespace max_gcd_of_linear_combinations_l546_54681

theorem max_gcd_of_linear_combinations (a b c : ℕ) (h1 : a + b + c ≤ 3000000) (h2 : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  gcd (a * b + 1) (gcd (a * c + 1) (b * c + 1)) ≤ 998285 :=
sorry

end max_gcd_of_linear_combinations_l546_54681


namespace shirt_cost_l546_54622

variables (J S : ℝ)

theorem shirt_cost :
  (3 * J + 2 * S = 69) ∧
  (2 * J + 3 * S = 86) →
  S = 24 :=
by
  sorry

end shirt_cost_l546_54622


namespace complete_set_of_events_l546_54617

-- Define the range of numbers on a die
def die_range := {n : ℕ | 1 ≤ n ∧ n ≤ 6}

-- Define what an outcome is
def outcome := { p : ℕ × ℕ | p.1 ∈ die_range ∧ p.2 ∈ die_range }

-- The theorem stating the complete set of outcomes
theorem complete_set_of_events : outcome = { p : ℕ × ℕ | 1 ≤ p.1 ∧ p.1 ≤ 6 ∧ 1 ≤ p.2 ∧ p.2 ≤ 6 } :=
by sorry

end complete_set_of_events_l546_54617


namespace arithmetic_sequence_common_difference_l546_54677

theorem arithmetic_sequence_common_difference (a_n : ℕ → ℤ) (h1 : a_n 5 = 3) (h2 : a_n 6 = -2) : a_n 6 - a_n 5 = -5 :=
by
  sorry

end arithmetic_sequence_common_difference_l546_54677


namespace inverse_at_neg_two_l546_54686

def g (x : ℝ) : ℝ := 5 * x^3 - 3

theorem inverse_at_neg_two :
  g (-2) = -43 :=
by
  -- sorry here to skip the proof, as instructed.
  sorry

end inverse_at_neg_two_l546_54686


namespace symmetric_points_power_l546_54663

theorem symmetric_points_power 
  (a b : ℝ) 
  (h1 : 2 * a = 8) 
  (h2 : 2 = a + b) :
  a^b = 1/16 := 
by sorry

end symmetric_points_power_l546_54663


namespace jake_last_10_shots_l546_54623

-- conditions
variable (total_shots_initially : ℕ) (shots_made_initially : ℕ) (percentage_initial : ℝ)
variable (total_shots_finally : ℕ) (shots_made_finally : ℕ) (percentage_final : ℝ)

axiom initial_conditions : shots_made_initially = percentage_initial * total_shots_initially
axiom final_conditions : shots_made_finally = percentage_final * total_shots_finally
axiom shots_difference : total_shots_finally - total_shots_initially = 10

-- prove that Jake made 7 out of the last 10 shots
theorem jake_last_10_shots : total_shots_initially = 30 → 
                             percentage_initial = 0.60 →
                             total_shots_finally = 40 → 
                             percentage_final = 0.62 →
                             shots_made_finally - shots_made_initially = 7 :=
by
  -- proofs to be filled in
  sorry

end jake_last_10_shots_l546_54623


namespace trapezoid_perimeter_l546_54628

theorem trapezoid_perimeter
  (AB CD AD BC : ℝ)
  (h1 : AB = 5)
  (h2 : CD = 5)
  (h3 : AD = 16)
  (h4 : BC = 8) :
  AB + BC + CD + AD = 34 :=
by
  sorry

end trapezoid_perimeter_l546_54628


namespace exposed_surface_area_l546_54614

theorem exposed_surface_area (r h : ℝ) (π : ℝ) (sphere_surface_area : ℝ) (cylinder_lateral_surface_area : ℝ) 
  (cond1 : r = 10) (cond2 : h = 5) (cond3 : sphere_surface_area = 4 * π * r^2) 
  (cond4 : cylinder_lateral_surface_area = 2 * π * r * h) :
  let hemisphere_curved_surface_area := sphere_surface_area / 2
  let hemisphere_base_area := π * r^2
  let total_surface_area := hemisphere_curved_surface_area + hemisphere_base_area + cylinder_lateral_surface_area
  total_surface_area = 400 * π :=
by
  sorry

end exposed_surface_area_l546_54614


namespace last_four_digits_of_5_pow_2018_l546_54687

theorem last_four_digits_of_5_pow_2018 : 
  (5^2018) % 10000 = 5625 :=
by {
  sorry
}

end last_four_digits_of_5_pow_2018_l546_54687


namespace seashells_in_six_weeks_l546_54648

def jar_weekly_update (week : Nat) (jarA : Nat) (jarB : Nat) : Nat × Nat :=
  if week % 3 = 0 then (jarA / 2, jarB / 2)
  else (jarA + 20, jarB * 2)

def total_seashells_after_weeks (initialA : Nat) (initialB : Nat) (weeks : Nat) : Nat :=
  let rec update (w : Nat) (jA : Nat) (jB : Nat) :=
    match w with
    | 0 => jA + jB
    | n + 1 =>
      let (newA, newB) := jar_weekly_update n jA jB
      update n newA newB
  update weeks initialA initialB

theorem seashells_in_six_weeks :
  total_seashells_after_weeks 50 30 6 = 97 :=
sorry

end seashells_in_six_weeks_l546_54648


namespace total_clothing_l546_54667

def num_boxes : ℕ := 4
def scarves_per_box : ℕ := 2
def mittens_per_box : ℕ := 6

theorem total_clothing :
  num_boxes * scarves_per_box + num_boxes * mittens_per_box = 32 :=
by
  sorry

end total_clothing_l546_54667


namespace exsphere_identity_l546_54627

-- Given definitions for heights and radii
variables {h1 h2 h3 h4 r1 r2 r3 r4 : ℝ}

-- Definition of the relationship that needs to be proven
theorem exsphere_identity 
  (h1 h2 h3 h4 r1 r2 r3 r4 : ℝ) :
  2 * (1 / h1 + 1 / h2 + 1 / h3 + 1 / h4) = 1 / r1 + 1 / r2 + 1 / r3 + 1 / r4 := 
sorry

end exsphere_identity_l546_54627


namespace sum_of_multiples_of_4_between_34_and_135_l546_54631

theorem sum_of_multiples_of_4_between_34_and_135 :
  let first := 36
  let last := 132
  let n := (last - first) / 4 + 1
  let sum := n * (first + last) / 2
  sum = 2100 := 
by
  sorry

end sum_of_multiples_of_4_between_34_and_135_l546_54631


namespace Shara_shells_total_l546_54629

def initial_shells : ℕ := 20
def shells_per_day : ℕ := 5
def days : ℕ := 3
def shells_fourth_day : ℕ := 6

theorem Shara_shells_total : (initial_shells + (shells_per_day * days) + shells_fourth_day) = 41 := by
  sorry

end Shara_shells_total_l546_54629


namespace proof_problem_l546_54689

variable {R : Type*} [Field R] {x y z w N : R}

theorem proof_problem 
  (h1 : 4 * x * z + y * w = N)
  (h2 : x * w + y * z = 6)
  (h3 : (2 * x + y) * (2 * z + w) = 15) :
  N = 3 :=
by sorry

end proof_problem_l546_54689


namespace area_of_triangle_l546_54697

def triangle (α β γ : Type) : (α ≃ β) ≃ γ ≃ Prop := sorry

variables (α β γ : Type) (AB AC AM : ℝ)
variables (ha : AB = 9) (hb : AC = 17) (hc : AM = 12)

theorem area_of_triangle (α β γ : Type) (AB AC AM : ℝ)
  (ha : AB = 9) (hb : AC = 17) (hc : AM = 12) : 
  ∃ A : ℝ, A = 74 :=
sorry

end area_of_triangle_l546_54697


namespace betty_age_l546_54673

theorem betty_age (A M B : ℕ) (h1 : A = 2 * M) (h2 : A = 4 * B) (h3 : M = A - 22) : B = 11 :=
by
  sorry

end betty_age_l546_54673


namespace inverse_function_correct_l546_54613

noncomputable def f (x : ℝ) : ℝ := 3 - 7 * x

noncomputable def g (x : ℝ) : ℝ := (3 - x) / 7

theorem inverse_function_correct : ∀ x : ℝ, f (g x) = x ∧ g (f x) = x :=
by
  intro x
  sorry

end inverse_function_correct_l546_54613


namespace circle_area_is_323pi_l546_54698

-- Define points A and B
def A : ℝ × ℝ := (2, 9)
def B : ℝ × ℝ := (14, 7)

-- Define that points A and B lie on circle ω
def on_circle_omega (A B C : ℝ × ℝ) (r : ℝ) : Prop :=
  (A.1 - C.1) ^ 2 + (A.2 - C.2) ^ 2 = r ^ 2 ∧
  (B.1 - C.1) ^ 2 + (B.2 - C.2) ^ 2 = r ^ 2

-- Define the tangent lines intersect at a point on the x-axis
def tangents_intersect_on_x_axis (A B : ℝ × ℝ) (C : ℝ × ℝ) (ω : (ℝ × ℝ) → ℝ): Prop := 
  ∃ x : ℝ, (A.1 - C.1) ^ 2 + (A.2 - C.2) ^ 2 = (B.1 - C.1) ^ 2 + (B.2 - C.2) ^ 2 ∧
             C.2 = 0

-- Problem statement to prove
theorem circle_area_is_323pi (C : ℝ × ℝ) (radius : ℝ) (on_circle_omega : on_circle_omega A B C radius)
  (tangents_intersect_on_x_axis : tangents_intersect_on_x_axis A B C omega) :
  π * radius ^ 2 = 323 * π :=
sorry

end circle_area_is_323pi_l546_54698


namespace number_of_tangent_small_circles_l546_54665

-- Definitions from the conditions
def central_radius : ℝ := 2
def small_radius : ℝ := 1

-- The proof problem statement
theorem number_of_tangent_small_circles : 
  ∃ n : ℕ, (∀ i j k : ℕ, i ≠ j ∧ j ≠ k ∧ k ≠ i → 
    dist (3 * central_radius) (3 * small_radius) = 3) ∧ n = 3 :=
by
  sorry

end number_of_tangent_small_circles_l546_54665


namespace factorize_x_cube_minus_9x_l546_54650

-- Lean statement: Prove that x^3 - 9x = x(x+3)(x-3)
theorem factorize_x_cube_minus_9x (x : ℝ) : x^3 - 9 * x = x * (x + 3) * (x - 3) :=
by
  sorry

end factorize_x_cube_minus_9x_l546_54650


namespace unique_root_in_interval_l546_54684

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2 + x - 2

theorem unique_root_in_interval (n : ℤ) (h_root : ∃ x : ℝ, 1 < x ∧ x < 2 ∧ f x = 0) :
  n = 1 := 
sorry

end unique_root_in_interval_l546_54684


namespace find_xyz_sum_l546_54683

theorem find_xyz_sum
  (x y z : ℝ)
  (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h1 : x^2 + x * y + y^2 = 108)
  (h2 : y^2 + y * z + z^2 = 16)
  (h3 : z^2 + z * x + x^2 = 124) :
  x * y + y * z + z * x = 48 := 
  sorry

end find_xyz_sum_l546_54683


namespace tan_ratio_l546_54630

theorem tan_ratio (x : ℝ) (h : Real.tan (x + Real.pi / 4) = 2) : 
  Real.tan x / Real.tan (2 * x) = 4 / 9 :=
sorry

end tan_ratio_l546_54630
