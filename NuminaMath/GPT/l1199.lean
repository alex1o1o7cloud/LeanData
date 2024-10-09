import Mathlib

namespace constructible_iff_multiple_of_8_l1199_119989

def is_constructible_with_L_tetromino (m n : ℕ) : Prop :=
  ∃ (k : ℕ), 4 * k = m * n

theorem constructible_iff_multiple_of_8 (m n : ℕ) (hm : 1 < m) (hn : 1 < n) : 
  is_constructible_with_L_tetromino m n ↔ 8 ∣ m * n :=
sorry

end constructible_iff_multiple_of_8_l1199_119989


namespace samantha_total_cost_l1199_119927

noncomputable def daily_rental_rate : ℝ := 30
noncomputable def daily_rental_days : ℝ := 3
noncomputable def cost_per_mile : ℝ := 0.15
noncomputable def miles_driven : ℝ := 500

theorem samantha_total_cost :
  (daily_rental_rate * daily_rental_days) + (cost_per_mile * miles_driven) = 165 :=
by
  sorry

end samantha_total_cost_l1199_119927


namespace max_value_of_expression_l1199_119994

open Real

theorem max_value_of_expression (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 3) :
  a + sqrt (a * b) + (a * b * c) ^ (1 / 4) ≤ 10 / 3 := sorry

end max_value_of_expression_l1199_119994


namespace range_of_a_l1199_119997

variable {a x : ℝ}

theorem range_of_a (h_eq : 2 * (x + a) = x + 3) (h_ineq : 2 * x - 10 > 8 * a) : a < -1 / 3 := 
sorry

end range_of_a_l1199_119997


namespace identify_linear_equation_l1199_119929

def is_linear_equation (eq : String) : Prop := sorry

theorem identify_linear_equation :
  is_linear_equation "2x = 0" ∧ ¬is_linear_equation "x^2 - 4x = 3" ∧ ¬is_linear_equation "x + 2y = 1" ∧ ¬is_linear_equation "x - 1 = 1 / x" :=
by 
  sorry

end identify_linear_equation_l1199_119929


namespace cylinder_in_sphere_volume_difference_is_correct_l1199_119943

noncomputable def volume_difference (base_radius_cylinder : ℝ) (radius_sphere : ℝ) : ℝ :=
  let height_cylinder := Real.sqrt (radius_sphere^2 - base_radius_cylinder^2)
  let volume_sphere := (4 / 3) * Real.pi * radius_sphere^3
  let volume_cylinder := Real.pi * base_radius_cylinder^2 * height_cylinder
  volume_sphere - volume_cylinder

theorem cylinder_in_sphere_volume_difference_is_correct :
  volume_difference 4 7 = (1372 - 48 * Real.sqrt 33) / 3 * Real.pi :=
by
  sorry

end cylinder_in_sphere_volume_difference_is_correct_l1199_119943


namespace unique_solution_values_a_l1199_119980

theorem unique_solution_values_a (a : ℝ) : 
  (∃ x y : ℝ, |x| + |y - 1| = 1 ∧ y = a * x + 2012) ∧ 
  (∀ x1 y1 x2 y2 : ℝ, (|x1| + |y1 - 1| = 1 ∧ y1 = a * x1 + 2012) ∧ 
                      (|x2| + |y2 - 1| = 1 ∧ y2 = a * x2 + 2012) → 
                      (x1 = x2 ∧ y1 = y2)) ↔ 
  a = 2011 ∨ a = -2011 := 
sorry

end unique_solution_values_a_l1199_119980


namespace range_of_m_l1199_119987

theorem range_of_m (m : ℝ) :
  (∀ x1 x2 : ℝ, 0 ≤ x1 ∧ x1 ≤ 2 ∧ 0 ≤ x2 ∧ x2 ≤ 2 → |((x2^2 - m * x2) - (x1^2 - m * x1))| ≤ 9) →
  -5 / 2 ≤ m ∧ m ≤ 13 / 2 :=
sorry

end range_of_m_l1199_119987


namespace flight_duration_l1199_119991

noncomputable def departure_time_pst := 9 * 60 + 15 -- in minutes
noncomputable def arrival_time_est := 17 * 60 + 40 -- in minutes
noncomputable def time_difference := 3 * 60 -- in minutes

theorem flight_duration (h m : ℕ) 
  (h_cond : 0 < m ∧ m < 60) 
  (total_flight_time : (arrival_time_est - (departure_time_pst + time_difference)) = h * 60 + m) : 
  h + m = 30 :=
sorry

end flight_duration_l1199_119991


namespace calculate_expression_l1199_119920

theorem calculate_expression (x y : ℚ) (hx : x = 5 / 6) (hy : y = 6 / 5) : 
  (1 / 3) * (x ^ 8) * (y ^ 9) = 2 / 5 :=
by
  sorry

end calculate_expression_l1199_119920


namespace max_stickers_single_player_l1199_119964

noncomputable def max_stickers (num_players : ℕ) (average_stickers : ℕ) : ℕ :=
  let total_stickers := num_players * average_stickers
  let min_stickers_one_player := 1
  let min_stickers_others := (num_players - 1) * min_stickers_one_player
  total_stickers - min_stickers_others

theorem max_stickers_single_player : 
  ∀ (num_players average_stickers : ℕ), 
    num_players = 25 → 
    average_stickers = 4 →
    ∀ player_stickers : ℕ, player_stickers ≤ max_stickers num_players average_stickers → player_stickers = 76 :=
    by
      intro num_players average_stickers players_eq avg_eq player_stickers player_le_max
      sorry

end max_stickers_single_player_l1199_119964


namespace correct_transformation_l1199_119999

theorem correct_transformation (a b c : ℝ) (h : c ≠ 0) (h1 : a / c = b / c) : a = b :=
by 
  -- Actual proof would go here, but we use sorry for the scaffold.
  sorry

end correct_transformation_l1199_119999


namespace correct_overestimation_l1199_119983

theorem correct_overestimation (y : ℕ) : 
  25 * y + 4 * y = 29 * y := 
by 
  sorry

end correct_overestimation_l1199_119983


namespace clothing_price_l1199_119947

theorem clothing_price
  (total_spent : ℕ)
  (num_pieces : ℕ)
  (price_piece_1 : ℕ)
  (price_piece_2 : ℕ)
  (num_remaining_pieces : ℕ)
  (total_remaining_pieces_price : ℕ)
  (price_remaining_piece : ℕ) :
  total_spent = 610 →
  num_pieces = 7 →
  price_piece_1 = 49 →
  price_piece_2 = 81 →
  num_remaining_pieces = 5 →
  total_spent = price_piece_1 + price_piece_2 + total_remaining_pieces_price →
  total_remaining_pieces_price = price_remaining_piece * num_remaining_pieces →
  price_remaining_piece = 96 :=
by
  intros h_total_spent h_num_pieces h_price_piece_1 h_price_piece_2 h_num_remaining_pieces h_total_remaining_price h_price_remaining_piece
  sorry

end clothing_price_l1199_119947


namespace fiona_reaches_goal_l1199_119959

-- Define the set of lily pads
def pads : Finset ℕ := Finset.range 15

-- Define the start, predator, and goal pads
def start_pad : ℕ := 0
def predator_pads : Finset ℕ := {4, 8}
def goal_pad : ℕ := 13

-- Define the hop probabilities
def hop_next : ℚ := 1/3
def hop_two : ℚ := 1/3
def hop_back : ℚ := 1/3

-- Define the transition probabilities (excluding jumps to negative pads)
def transition (current next : ℕ) : ℚ :=
  if next = current + 1 ∨ next = current + 2 ∨ (next = current - 1 ∧ current > 0)
  then 1/3 else 0

-- Define the function to check if a pad is safe
def is_safe (pad : ℕ) : Prop := ¬ (pad ∈ predator_pads)

-- Define the probability that Fiona reaches pad 13 without landing on 4 or 8
noncomputable def probability_reach_13 : ℚ :=
  -- Function to recursively calculate the probability
  sorry

-- Statement to prove
theorem fiona_reaches_goal : probability_reach_13 = 16 / 177147 := 
sorry

end fiona_reaches_goal_l1199_119959


namespace walking_speed_10_mph_l1199_119934

theorem walking_speed_10_mph 
  (total_minutes : ℕ)
  (distance : ℕ)
  (rest_per_segment : ℕ)
  (rest_time : ℕ)
  (segments : ℕ)
  (walk_time : ℕ)
  (walk_time_hours : ℕ) :
  total_minutes = 328 → 
  distance = 50 → 
  rest_per_segment = 7 → 
  segments = 4 →
  rest_time = segments * rest_per_segment →
  walk_time = total_minutes - rest_time →
  walk_time_hours = walk_time / 60 →
  distance / walk_time_hours = 10 :=
by
  sorry

end walking_speed_10_mph_l1199_119934


namespace total_value_of_coins_l1199_119960

theorem total_value_of_coins (num_quarters num_nickels : ℕ) (val_quarter val_nickel : ℝ)
  (h_quarters : num_quarters = 8) (h_nickels : num_nickels = 13)
  (h_total_coins : num_quarters + num_nickels = 21) (h_val_quarter : val_quarter = 0.25)
  (h_val_nickel : val_nickel = 0.05) :
  num_quarters * val_quarter + num_nickels * val_nickel = 2.65 := 
sorry

end total_value_of_coins_l1199_119960


namespace find_positive_integer_k_l1199_119970

theorem find_positive_integer_k (p : ℕ) (hp : Prime p) (hp2 : Odd p) : 
  ∃ k : ℕ, k > 0 ∧ ∃ n : ℕ, n * n = k - p * k ∧ k = ((p + 1) * (p + 1)) / 4 :=
by
  sorry

end find_positive_integer_k_l1199_119970


namespace math_problem_l1199_119937

theorem math_problem (a b : ℝ) (h : a / (1 + a) + b / (1 + b) = 1) : 
  a / (1 + b^2) - b / (1 + a^2) = a - b := 
sorry

end math_problem_l1199_119937


namespace part1_part2_l1199_119998

noncomputable def f (x : ℝ) : ℝ := |3 * x + 2|

theorem part1 (x : ℝ): f x < 6 - |x - 2| ↔ (-3/2 < x ∧ x < 1) :=
by sorry

theorem part2 (a : ℝ) (m n : ℝ) (h₁ : 0 < m) (h₂ : 0 < n) (h₃ : m + n = 4) (h₄ : 0 < a) (h₅ : ∀ x, |x - a| - f x ≤ 1/m + 1/n) :
    0 < a ∧ a ≤ 1/3 :=
by sorry

end part1_part2_l1199_119998


namespace distance_car_to_stream_l1199_119919

theorem distance_car_to_stream (total_distance : ℝ) (stream_to_meadow : ℝ) (meadow_to_campsite : ℝ) (h1 : total_distance = 0.7) (h2 : stream_to_meadow = 0.4) (h3 : meadow_to_campsite = 0.1) :
  (total_distance - (stream_to_meadow + meadow_to_campsite) = 0.2) :=
by
  sorry

end distance_car_to_stream_l1199_119919


namespace determine_g_l1199_119957

noncomputable def g : ℝ → ℝ := sorry 

lemma g_functional_equation (x y : ℝ) : g (x * y) = g ((x^2 + y^2 + 1) / 3) + (x - y)^2 :=
sorry

lemma g_at_zero : g 0 = 1 :=
sorry

theorem determine_g (x : ℝ) : g x = 2 - 2 * x :=
sorry

end determine_g_l1199_119957


namespace evaluate_expression_l1199_119953

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := 
by 
  sorry

end evaluate_expression_l1199_119953


namespace no_such_abc_exists_l1199_119958

theorem no_such_abc_exists : ¬ ∃ (a b c : ℝ), ∀ (x y : ℝ),
  |x + a| + |x + y + b| + |y + c| > |x| + |x + y| + |y| :=
by
  sorry

end no_such_abc_exists_l1199_119958


namespace find_m_l1199_119996

theorem find_m 
  (m : ℝ)
  (h_pos : 0 < m)
  (asymptote_twice_angle : ∃ l : ℝ, l = 3 ∧ (x - l * y = 0 ∧ m * x^2 - y^2 = m)) :
  m = 3 :=
by
  sorry

end find_m_l1199_119996


namespace ratio_comparison_l1199_119977

-- Define the ratios in the standard and sport formulations
def ratio_flavor_corn_standard : ℚ := 1 / 12
def ratio_flavor_water_standard : ℚ := 1 / 30
def ratio_flavor_water_sport : ℚ := 1 / 60

-- Define the amounts of corn syrup and water in the sport formulation
def corn_syrup_sport : ℚ := 2
def water_sport : ℚ := 30

-- Calculate the amount of flavoring in the sport formulation
def flavoring_sport : ℚ := water_sport / 60

-- Calculate the ratio of flavoring to corn syrup in the sport formulation
def ratio_flavor_corn_sport : ℚ := flavoring_sport / corn_syrup_sport

-- Define the theorem to prove the ratio comparison
theorem ratio_comparison :
  (ratio_flavor_corn_sport / ratio_flavor_corn_standard) = 3 :=
by
  -- Using the given conditions and definitions, prove the theorem
  sorry

end ratio_comparison_l1199_119977


namespace coin_toss_min_n_l1199_119900

theorem coin_toss_min_n (n : ℕ) :
  (1 : ℝ) - (1 / (2 : ℝ)) ^ n ≥ 15 / 16 → n ≥ 4 :=
by
  sorry

end coin_toss_min_n_l1199_119900


namespace unique_solution_system_eqns_l1199_119995

theorem unique_solution_system_eqns (a b c : ℕ) :
  (a^3 - b^3 - c^3 = 3 * a * b * c) ∧ (a^2 = 2 * (b + c)) ↔ (a = 2 ∧ b = 1 ∧ c = 1) := by 
  sorry

end unique_solution_system_eqns_l1199_119995


namespace no_term_in_sequence_is_3_alpha_5_beta_l1199_119971

theorem no_term_in_sequence_is_3_alpha_5_beta :
  ∀ (v : ℕ → ℕ),
    v 0 = 0 →
    v 1 = 1 →
    (∀ n, 1 ≤ n → v (n + 1) = 8 * v n * v (n - 1)) →
    ∀ n, ∀ (α β : ℕ), α > 0 → β > 0 → v n ≠ 3^α * 5^β := by
  intros v h0 h1 recurrence n α β hα hβ
  sorry

end no_term_in_sequence_is_3_alpha_5_beta_l1199_119971


namespace hyperbola_asymptotes_l1199_119979

theorem hyperbola_asymptotes (x y : ℝ) :
  (x^2 / 4 - y^2 = 1) → (y = x / 2 ∨ y = -x / 2) :=
sorry

end hyperbola_asymptotes_l1199_119979


namespace hyperbola_asymptote_product_l1199_119952

theorem hyperbola_asymptote_product (k1 k2 : ℝ) (h1 : k1 = 1) (h2 : k2 = -1) :
  k1 * k2 = -1 :=
by
  rw [h1, h2]
  norm_num

end hyperbola_asymptote_product_l1199_119952


namespace find_y_l1199_119903

-- Define vectors as tuples
def vector_1 : ℝ × ℝ := (3, 4)
def vector_2 (y : ℝ) : ℝ × ℝ := (y, -5)

-- Define dot product
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Condition for orthogonality
def orthogonal (u v : ℝ × ℝ) : Prop :=
  dot_product u v = 0

-- The theorem we want to prove
theorem find_y (y : ℝ) :
  orthogonal vector_1 (vector_2 y) → y = (20 / 3) :=
by
  sorry

end find_y_l1199_119903


namespace unit_prices_max_books_l1199_119981

-- Definitions based on conditions 1 and 2
def unit_price_A (x : ℝ) : Prop :=
  x > 5 ∧ (1200 / x = 900 / (x - 5))

-- Definitions based on conditions 3, 4, and 5
def max_books_A (y : ℝ) : Prop :=
  0 ≤ y ∧ y ≤ 300 ∧ 0.9 * 20 * y + 15 * (300 - y) ≤ 5100

theorem unit_prices
  (x : ℝ)
  (h : unit_price_A x) :
  x = 20 ∧ x - 5 = 15 :=
sorry

theorem max_books
  (y : ℝ)
  (hy : max_books_A y) :
  y ≤ 200 :=
sorry

end unit_prices_max_books_l1199_119981


namespace cost_of_four_enchiladas_and_five_tacos_l1199_119932

-- Define the cost of an enchilada and a taco
variables (e t : ℝ)

-- Define the conditions given in the problem
def condition1 : Prop := e + 4 * t = 2.30
def condition2 : Prop := 4 * e + t = 3.10

-- Define the final cost of four enchiladas and five tacos
def cost : ℝ := 4 * e + 5 * t

-- State the theorem we need to prove
theorem cost_of_four_enchiladas_and_five_tacos 
  (h1 : condition1 e t) 
  (h2 : condition2 e t) : 
  cost e t = 4.73 := 
sorry

end cost_of_four_enchiladas_and_five_tacos_l1199_119932


namespace trains_time_to_clear_each_other_l1199_119902

noncomputable def relative_speed (v1 v2 : ℝ) : ℝ :=
  v1 + v2

noncomputable def speed_to_m_s (v_kmph : ℝ) : ℝ :=
  v_kmph * 1000 / 3600

noncomputable def total_length (l1 l2 : ℝ) : ℝ :=
  l1 + l2

theorem trains_time_to_clear_each_other :
  ∀ (l1 l2 : ℝ) (v1_kmph v2_kmph : ℝ),
    l1 = 100 → l2 = 280 →
    v1_kmph = 42 → v2_kmph = 30 →
    (total_length l1 l2) / (speed_to_m_s (relative_speed v1_kmph v2_kmph)) = 19 :=
by
  intros l1 l2 v1_kmph v2_kmph h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end trains_time_to_clear_each_other_l1199_119902


namespace probability_not_rel_prime_50_l1199_119908

theorem probability_not_rel_prime_50 : 
  let n := 50;
  let non_rel_primes_count := n - Nat.totient 50;
  let total_count := n;
  let probability := (non_rel_primes_count : ℚ) / (total_count : ℚ);
  probability = 3 / 5 :=
by
  sorry

end probability_not_rel_prime_50_l1199_119908


namespace cost_per_square_meter_l1199_119928

-- Definitions from conditions
def lawn_length : ℝ := 80
def lawn_breadth : ℝ := 50
def road_width : ℝ := 10
def total_cost : ℝ := 3600

-- Theorem to prove the cost per square meter of traveling the roads
theorem cost_per_square_meter :
  total_cost / 
  ((lawn_length * road_width) + (lawn_breadth * road_width) - (road_width * road_width)) = 3 := by
  sorry

end cost_per_square_meter_l1199_119928


namespace jake_total_distance_l1199_119910

noncomputable def jake_rate : ℝ := 4 -- Jake's walking rate in miles per hour
noncomputable def total_time : ℝ := 2 -- Jake's total walking time in hours
noncomputable def break_time : ℝ := 0.5 -- Jake's break time in hours

theorem jake_total_distance :
  jake_rate * (total_time - break_time) = 6 :=
by
  sorry

end jake_total_distance_l1199_119910


namespace m_plus_n_l1199_119990

theorem m_plus_n (m n : ℕ) (hm : 0 < m) (hn : 1 < n) (h : m ^ n = 2^25 * 3^40) : m + n = 209957 :=
  sorry

end m_plus_n_l1199_119990


namespace total_money_l1199_119975

-- Define the variables A, B, and C as real numbers.
variables (A B C : ℝ)

-- Define the conditions as hypotheses.
def conditions : Prop :=
  A + C = 300 ∧ B + C = 150 ∧ C = 50

-- State the theorem to prove the total amount of money A, B, and C have.
theorem total_money (h : conditions A B C) : A + B + C = 400 :=
by {
  -- This proof is currently omitted.
  sorry
}

end total_money_l1199_119975


namespace midpoint_set_of_segments_eq_circle_l1199_119963

-- Define the existence of skew perpendicular lines with given properties
variable (a d : ℝ)

-- Conditions: Distance between lines is a, segment length is d
-- The coordinates system configuration
-- Point on the first line: (x, 0, 0)
-- Point on the second line: (0, y, a)
def are_midpoints_of_segments_of_given_length
  (p : ℝ × ℝ × ℝ) : Prop :=
  ∃ (x y : ℝ), 
    p = (x / 2, y / 2, a / 2) ∧ 
    x^2 + y^2 = d^2 - a^2

-- Proof statement
theorem midpoint_set_of_segments_eq_circle :
  { p : ℝ × ℝ × ℝ | are_midpoints_of_segments_of_given_length a d p } =
  { p : ℝ × ℝ × ℝ | ∃ (r : ℝ), p = (r * (d^2 - a^2) / (2*d), r * (d^2 - a^2) / (2*d), a / 2)
    ∧ r^2 * (d^2 - a^2) = (d^2 - a^2) } :=
sorry

end midpoint_set_of_segments_eq_circle_l1199_119963


namespace price_of_jumbo_pumpkin_l1199_119967

theorem price_of_jumbo_pumpkin (total_pumpkins : ℕ) (total_revenue : ℝ)
  (regular_pumpkins : ℕ) (price_regular : ℝ)
  (sold_jumbo_pumpkins : ℕ) (revenue_jumbo : ℝ): 
  total_pumpkins = 80 →
  total_revenue = 395.00 →
  regular_pumpkins = 65 →
  price_regular = 4.00 →
  sold_jumbo_pumpkins = total_pumpkins - regular_pumpkins →
  revenue_jumbo = total_revenue - (price_regular * regular_pumpkins) →
  revenue_jumbo / sold_jumbo_pumpkins = 9.00 :=
by
  intro h_total_pumpkins
  intro h_total_revenue
  intro h_regular_pumpkins
  intro h_price_regular
  intro h_sold_jumbo_pumpkins
  intro h_revenue_jumbo
  sorry

end price_of_jumbo_pumpkin_l1199_119967


namespace point_on_parabola_touching_x_axis_l1199_119940

theorem point_on_parabola_touching_x_axis (a b c : ℤ) (h : ∃ r : ℤ, a * (r * r) + b * r + c = 0 ∧ (r * r) = 0) :
  ∃ (a' b' : ℤ), ∃ k : ℤ, (k * k) + a' * k + b' = 0 ∧ (k * k) = 0 :=
sorry

end point_on_parabola_touching_x_axis_l1199_119940


namespace sides_of_right_triangle_l1199_119941

theorem sides_of_right_triangle (r : ℝ) (a b c : ℝ) 
  (h_area : (2 / (2 / r)) * 2 = 2 * r) 
  (h_right : a^2 + b^2 = c^2) :
  (a = r ∧ b = (4 / 3) * r ∧ c = (5 / 3) * r) ∨
  (b = r ∧ a = (4 / 3) * r ∧ c = (5 / 3) * r) :=
sorry

end sides_of_right_triangle_l1199_119941


namespace expand_expression_l1199_119993

variable (x y : ℝ)

theorem expand_expression (x y : ℝ) : 12 * (3 * x + 4 - 2 * y) = 36 * x + 48 - 24 * y :=
by
  sorry

end expand_expression_l1199_119993


namespace decreasing_function_solution_set_l1199_119904

theorem decreasing_function_solution_set {f : ℝ → ℝ} (h : ∀ x y, x < y → f y < f x) :
  {x : ℝ | f 2 < f (2*x + 1)} = {x : ℝ | x < 1/2} :=
by
  sorry

end decreasing_function_solution_set_l1199_119904


namespace meaningful_sqrt_l1199_119918

theorem meaningful_sqrt (a : ℝ) (h : a ≥ 4) : a = 6 ↔ ∃ x ∈ ({-1, 0, 2, 6} : Set ℝ), x = 6 := 
by
  sorry

end meaningful_sqrt_l1199_119918


namespace hyperbola_equation_l1199_119914

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (asymptote_eq : ∀ x y, 3*x + 4*y = 0 → y = (-3/4) * x)
  (focus_eq : (0, 5) = (0, 5)) :
  ∃ a b : ℝ, a = 3 ∧ b = 4 ∧ (∀ y x, (y^2 / 9 - x^2 / 16 = 1)) :=
sorry

end hyperbola_equation_l1199_119914


namespace other_number_is_31_l1199_119935

namespace LucasProblem

-- Definitions of the integers a and b and the condition on their sum
variables (a b : ℤ)
axiom h_sum : 3 * a + 4 * b = 161
axiom h_one_is_17 : a = 17 ∨ b = 17

-- The theorem we need to prove
theorem other_number_is_31 (h_one_is_17 : a = 17 ∨ b = 17) : 
  (b = 17 → a = 31) ∧ (a = 17 → false) :=
by
  sorry

end LucasProblem

end other_number_is_31_l1199_119935


namespace part1_solution_set_part2_range_of_a_l1199_119956

def f (x a : ℝ) := |x - a^2| + |x - (2*a + 1)|

-- Part (1)
theorem part1_solution_set (x : ℝ) :
  ∀ x, f x 2 ≥ 4 ↔ x ≤ 3 / 2 ∨ x ≥ 11 / 2 := sorry

-- Part (2)
theorem part2_range_of_a (a : ℝ) :
  (∀ x, f x a ≥ 4) ↔ (a ≤ -1 ∨ a ≥ 3) := sorry

end part1_solution_set_part2_range_of_a_l1199_119956


namespace julia_played_more_kids_l1199_119915

variable (kidsPlayedMonday : Nat) (kidsPlayedTuesday : Nat)

theorem julia_played_more_kids :
  kidsPlayedMonday = 11 →
  kidsPlayedTuesday = 12 →
  kidsPlayedTuesday - kidsPlayedMonday = 1 :=
by
  intros hMonday hTuesday
  sorry

end julia_played_more_kids_l1199_119915


namespace emily_art_supplies_l1199_119962

theorem emily_art_supplies (total_spent skirts_cost skirt_quantity : ℕ) 
  (total_spent_eq : total_spent = 50) 
  (skirt_cost_eq : skirts_cost = 15) 
  (skirt_quantity_eq : skirt_quantity = 2) :
  total_spent - skirt_quantity * skirts_cost = 20 :=
by
  sorry

end emily_art_supplies_l1199_119962


namespace find_y_values_l1199_119984

theorem find_y_values (x : ℝ) (h1 : x^2 + 4 * ( (x + 1) / (x - 3) )^2 = 50)
  (y := ( (x - 3)^2 * (x + 4) ) / (2 * x - 4)) :
  y = -32 / 7 ∨ y = 2 :=
sorry

end find_y_values_l1199_119984


namespace bus_driver_earnings_l1199_119978

variables (rate : ℝ) (regular_hours overtime_hours : ℕ) (regular_rate overtime_rate : ℝ)

def calculate_regular_earnings (regular_rate : ℝ) (regular_hours : ℕ) : ℝ :=
  regular_rate * regular_hours

def calculate_overtime_earnings (overtime_rate : ℝ) (overtime_hours : ℕ) : ℝ :=
  overtime_rate * overtime_hours

def total_compensation (regular_rate overtime_rate : ℝ) (regular_hours overtime_hours : ℕ) : ℝ :=
  calculate_regular_earnings regular_rate regular_hours + calculate_overtime_earnings overtime_rate overtime_hours

theorem bus_driver_earnings :
  let regular_rate := 16
  let overtime_rate := regular_rate * 1.75
  let regular_hours := 40
  let total_hours := 44
  let overtime_hours := total_hours - regular_hours
  total_compensation regular_rate overtime_rate regular_hours overtime_hours = 752 :=
by
  sorry

end bus_driver_earnings_l1199_119978


namespace abhay_speed_l1199_119945

variables (A S : ℝ)

theorem abhay_speed (h1 : 24 / A = 24 / S + 2) (h2 : 24 / (2 * A) = 24 / S - 1) : A = 12 :=
by {
  sorry
}

end abhay_speed_l1199_119945


namespace cos_transformation_l1199_119930

variable {θ a : ℝ}

theorem cos_transformation (h : Real.sin (θ + π / 12) = a) :
  Real.cos (θ + 7 * π / 12) = -a := 
sorry

end cos_transformation_l1199_119930


namespace find_a_div_b_l1199_119939

theorem find_a_div_b (a b : ℝ) (h : a ≠ b) (h_eq : a / b + (a + 6 * b) / (b + 6 * a) = 3) : 
  a / b = (8 + Real.sqrt 46) / 6 ∨ a / b = (8 - Real.sqrt 46) / 6 :=
by 
  sorry

end find_a_div_b_l1199_119939


namespace men_in_first_group_l1199_119992

variable (M : ℕ) (daily_wage : ℝ)
variable (h1 : M * 10 * daily_wage = 1200)
variable (h2 : 9 * 6 * daily_wage = 1620)
variable (dw_eq : daily_wage = 30)

theorem men_in_first_group : M = 4 :=
by sorry

end men_in_first_group_l1199_119992


namespace max_value_quadratic_expression_l1199_119938

theorem max_value_quadratic_expression : ∃ x : ℝ, -3 * x^2 + 18 * x - 5 ≤ 22 ∧ ∀ y : ℝ, -3 * y^2 + 18 * y - 5 ≤ 22 := 
by 
  sorry

end max_value_quadratic_expression_l1199_119938


namespace polynomial_remainder_l1199_119922

theorem polynomial_remainder (x : ℤ) : 
  (2 * x + 3) ^ 504 % (x^2 - x + 1) = (16 * x + 5) :=
by
  sorry

end polynomial_remainder_l1199_119922


namespace seth_oranges_l1199_119925

def initial_boxes := 9
def boxes_given_to_mother := 1

def remaining_boxes_after_giving_to_mother := initial_boxes - boxes_given_to_mother
def boxes_given_away := remaining_boxes_after_giving_to_mother / 2
def boxes_left := remaining_boxes_after_giving_to_mother - boxes_given_away

theorem seth_oranges : boxes_left = 4 := by
  sorry

end seth_oranges_l1199_119925


namespace half_is_greater_than_third_by_one_sixth_l1199_119923

theorem half_is_greater_than_third_by_one_sixth : (0.5 : ℝ) - (1 / 3 : ℝ) = 1 / 6 := by
  sorry

end half_is_greater_than_third_by_one_sixth_l1199_119923


namespace kamal_average_marks_l1199_119951

theorem kamal_average_marks :
  (76 / 120) * 0.2 + 
  (60 / 110) * 0.25 + 
  (82 / 100) * 0.15 + 
  (67 / 90) * 0.2 + 
  (85 / 100) * 0.15 + 
  (78 / 95) * 0.05 = 0.70345 :=
by 
  sorry

end kamal_average_marks_l1199_119951


namespace katrina_tax_deduction_l1199_119909

variable (hourlyWage : ℚ) (taxRate : ℚ)

def wageInCents (wage : ℚ) : ℚ := wage * 100
def taxInCents (wageInCents : ℚ) (rate : ℚ) : ℚ := wageInCents * rate / 100

theorem katrina_tax_deduction : 
  hourlyWage = 25 ∧ taxRate = 2.5 → taxInCents (wageInCents hourlyWage) taxRate = 62.5 := 
by 
  sorry

end katrina_tax_deduction_l1199_119909


namespace determine_a_l1199_119972

theorem determine_a 
  (a : ℝ)
  (h : ∀ x : ℝ, |a * x - 2| < 3 ↔ - 5 / 3 < x ∧ x < 1 / 3) : 
  a = -3 := by 
  sorry

end determine_a_l1199_119972


namespace part1_part2_l1199_119905

theorem part1 (x : ℝ) (m : ℝ) :
  (∃ x, x^2 - 2*(m-1)*x + m^2 = 0) → (m ≤ 1 / 2) := 
  sorry

theorem part2 (x1 x2 : ℝ) (m : ℝ) :
  (x1^2 - 2*(m-1)*x1 + m^2 = 0) ∧ (x2^2 - 2*(m-1)*x2 + m^2 = 0) ∧ 
  (x1^2 + x2^2 = 8 - 3*x1*x2) → (m = -2 / 5) := 
  sorry

end part1_part2_l1199_119905


namespace sum_of_digits_3n_l1199_119949

def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem sum_of_digits_3n (n : ℕ) (hn1 : digit_sum n = 100) (hn2 : digit_sum (44 * n) = 800) : digit_sum (3 * n) = 300 := by
  sorry

end sum_of_digits_3n_l1199_119949


namespace geom_seq_308th_term_l1199_119974

def geom_seq (a : ℤ) (r : ℤ) (n : ℕ) : ℤ :=
  a * r ^ n

-- Given conditions
def a := 10
def r := -1

theorem geom_seq_308th_term : geom_seq a r 307 = -10 := by
  sorry

end geom_seq_308th_term_l1199_119974


namespace john_salary_april_l1199_119912

theorem john_salary_april 
  (initial_salary : ℤ)
  (raise_percentage : ℤ)
  (cut_percentage : ℤ)
  (bonus : ℤ)
  (february_salary : ℤ)
  (march_salary : ℤ)
  : initial_salary = 3000 →
    raise_percentage = 10 →
    cut_percentage = 15 →
    bonus = 500 →
    february_salary = initial_salary + (initial_salary * raise_percentage / 100) →
    march_salary = february_salary - (february_salary * cut_percentage / 100) →
    march_salary + bonus = 3305 :=
by
  intros
  sorry

end john_salary_april_l1199_119912


namespace pow_mod_eq_residue_l1199_119973

theorem pow_mod_eq_residue :
  (3 : ℤ)^(2048) % 11 = 5 :=
sorry

end pow_mod_eq_residue_l1199_119973


namespace maximum_value_x_2y_2z_l1199_119976

noncomputable def max_sum (x y z : ℝ) (h : x^2 + y^2 + z^2 = 25) : ℝ :=
  x + 2*y + 2*z

theorem maximum_value_x_2y_2z (x y z : ℝ) (h : x^2 + y^2 + z^2 = 25) : 
  max_sum x y z h ≤ 15 :=
sorry

end maximum_value_x_2y_2z_l1199_119976


namespace find_m_l1199_119921

-- Definitions based on conditions
def Point (α : Type) := α × α

def A : Point ℝ := (2, -3)
def B : Point ℝ := (4, 3)
def C (m : ℝ) : Point ℝ := (5, m)

-- The collinearity condition
def collinear (p1 p2 p3 : Point ℝ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (y2 - y1) * (x3 - x2) = (y3 - y2) * (x2 - x1)

-- The proof problem
theorem find_m (m : ℝ) : collinear A B (C m) → m = 6 :=
by
  sorry

end find_m_l1199_119921


namespace triplets_of_positive_integers_l1199_119916

/-- We want to determine all positive integer triplets (a, b, c) such that
    ab - c, bc - a, and ca - b are all powers of 2.
    A power of 2 is an integer of the form 2^n, where n is a non-negative integer.-/
theorem triplets_of_positive_integers (a b c : ℕ) (h1: 0 < a) (h2: 0 < b) (h3: 0 < c) :
  ((∃ k1 : ℕ, ab - c = 2^k1) ∧ (∃ k2 : ℕ, bc - a = 2^k2) ∧ (∃ k3 : ℕ, ca - b = 2^k3))
  ↔ (a = 2 ∧ b = 2 ∧ c = 2) ∨ (a = 3 ∧ b = 2 ∧ c = 2) ∨ (a = 2 ∧ b = 6 ∧ c = 11) ∨ (a = 3 ∧ b = 5 ∧ c = 7) :=
sorry

end triplets_of_positive_integers_l1199_119916


namespace golf_balls_count_l1199_119955

theorem golf_balls_count (dozen_count : ℕ) (balls_per_dozen : ℕ) (total_balls : ℕ) 
  (h1 : dozen_count = 13) 
  (h2 : balls_per_dozen = 12) 
  (h3 : total_balls = dozen_count * balls_per_dozen) : 
  total_balls = 156 := 
sorry

end golf_balls_count_l1199_119955


namespace monotonic_function_range_l1199_119936

theorem monotonic_function_range (a : ℝ) :
  (∀ x : ℝ, -3 * x^2 + 2 * a * x - 1 ≤ 0) → -Real.sqrt 3 ≤ a ∧ a ≤ Real.sqrt 3 :=
by
  sorry

end monotonic_function_range_l1199_119936


namespace wholesale_cost_per_bag_l1199_119933

theorem wholesale_cost_per_bag (W : ℝ) (h1 : 1.12 * W = 28) : W = 25 :=
sorry

end wholesale_cost_per_bag_l1199_119933


namespace inequality_holds_for_n_ge_0_l1199_119907

theorem inequality_holds_for_n_ge_0
  (n : ℤ)
  (h : n ≥ 0)
  (a b c x y z : ℝ)
  (Habc : 0 < a ∧ 0 < b ∧ 0 < c)
  (Hxyz : 0 < x ∧ 0 < y ∧ 0 < z)
  (Hmax : max a (max b (max c (max x (max y z)))) = a)
  (Hsum : a + b + c = x + y + z)
  (Hprod : a * b * c = x * y * z) : a^n + b^n + c^n ≥ x^n + y^n + z^n := 
sorry

end inequality_holds_for_n_ge_0_l1199_119907


namespace boat_speed_in_still_water_l1199_119946

theorem boat_speed_in_still_water :
  ∀ (V_b V_s : ℝ) (distance time : ℝ),
  V_s = 5 →
  time = 4 →
  distance = 84 →
  (distance / time) = V_b + V_s →
  V_b = 16 :=
by
  -- Given definitions and values
  intros V_b V_s distance time
  intro hV_s
  intro htime
  intro hdistance
  intro heq
  sorry -- Placeholder for the actual proof

end boat_speed_in_still_water_l1199_119946


namespace unique_positive_integer_divisibility_l1199_119966

theorem unique_positive_integer_divisibility (n : ℕ) (h : n > 0) : 
  (5^(n-1) + 3^(n-1)) ∣ (5^n + 3^n) ↔ n = 1 :=
by
  sorry

end unique_positive_integer_divisibility_l1199_119966


namespace muffin_banana_ratio_l1199_119986

variables (m b : ℝ)

theorem muffin_banana_ratio (h1 : 4 * m + 3 * b = x) 
                            (h2 : 2 * (4 * m + 3 * b) = 2 * m + 16 * b) : 
                            m / b = 5 / 3 :=
by sorry

end muffin_banana_ratio_l1199_119986


namespace garage_sale_items_count_l1199_119931

theorem garage_sale_items_count :
  (16 + 22) + 1 = 38 :=
by
  -- proof goes here
  sorry

end garage_sale_items_count_l1199_119931


namespace altitude_identity_l1199_119901

variable {a b c d : ℝ}

def is_right_triangle (A B C : ℝ) : Prop :=
  A^2 + B^2 = C^2

def right_angle_triangle (a b c : ℝ) : Prop := 
  a^2 + b^2 = c^2

def altitude_property (a b c d : ℝ) : Prop :=
  a * b = c * d

theorem altitude_identity (a b c d : ℝ) (h1: right_angle_triangle a b c) (h2: altitude_property a b c d) :
  1 / a^2 + 1 / b^2 = 1 / d^2 :=
sorry

end altitude_identity_l1199_119901


namespace find_k_l1199_119942

theorem find_k : ∃ k : ℕ, ∀ n : ℕ, n > 0 → (2^n + 11) % (2^k - 1) = 0 ↔ k = 4 :=
by
  sorry

end find_k_l1199_119942


namespace total_distance_after_fourth_bounce_l1199_119988

noncomputable def total_distance_traveled (initial_height : ℝ) (bounce_ratio : ℝ) (num_bounces : ℕ) : ℝ :=
  let fall_distances := (List.range (num_bounces + 1)).map (λ n => initial_height * bounce_ratio^n)
  let rise_distances := (List.range num_bounces).map (λ n => initial_height * bounce_ratio^(n+1))
  fall_distances.sum + rise_distances.sum

theorem total_distance_after_fourth_bounce :
  total_distance_traveled 25 (5/6 : ℝ) 4 = 154.42 :=
by
  sorry

end total_distance_after_fourth_bounce_l1199_119988


namespace polygon_vertices_product_at_least_2014_l1199_119911

theorem polygon_vertices_product_at_least_2014 :
  ∀ (vertices : Fin 90 → ℕ), 
    (∀ i, 1 ≤ vertices i ∧ vertices i ≤ 90) → 
    ∃ i, (vertices i) * (vertices ((i + 1) % 90)) ≥ 2014 :=
sorry

end polygon_vertices_product_at_least_2014_l1199_119911


namespace ratio_fraction_l1199_119961

theorem ratio_fraction (x : ℚ) : x = 2 / 9 ↔ (2 / 6) / x = (3 / 4) / (1 / 2) := by
  sorry

end ratio_fraction_l1199_119961


namespace base_conversion_subtraction_l1199_119969

def base6_to_base10 (n : ℕ) : ℕ :=
3 * (6^2) + 2 * (6^1) + 5 * (6^0)

def base5_to_base10 (m : ℕ) : ℕ :=
2 * (5^2) + 3 * (5^1) + 1 * (5^0)

theorem base_conversion_subtraction : 
  base6_to_base10 325 - base5_to_base10 231 = 59 :=
by
  sorry

end base_conversion_subtraction_l1199_119969


namespace eval_at_5_l1199_119906

def g (x : ℝ) : ℝ := 3 * x^4 - 8 * x^3 + 15 * x^2 - 10 * x - 75

theorem eval_at_5 : g 5 = 1125 := by
  sorry

end eval_at_5_l1199_119906


namespace total_amount_paid_l1199_119954

def price_grapes (kg: ℕ) (rate: ℕ) : ℕ := kg * rate
def price_mangoes (kg: ℕ) (rate: ℕ) : ℕ := kg * rate
def price_pineapple (kg: ℕ) (rate: ℕ) : ℕ := kg * rate
def price_kiwi (kg: ℕ) (rate: ℕ) : ℕ := kg * rate

theorem total_amount_paid :
  price_grapes 14 54 + price_mangoes 10 62 + price_pineapple 8 40 + price_kiwi 5 30 = 1846 :=
by
  sorry

end total_amount_paid_l1199_119954


namespace smallest_value_expression_l1199_119950

theorem smallest_value_expression (x y : ℝ) (hx : -5 ≤ x ∧ x ≤ -3) (hy : 3 ≤ y ∧ y ≤ 5) :
  ∃ m, m = y ∧ m = 3 :=
by
  sorry

end smallest_value_expression_l1199_119950


namespace cos_pi_over_2_plus_2theta_l1199_119985

theorem cos_pi_over_2_plus_2theta (θ : ℝ) (hcos : Real.cos θ = 1 / 3) (hθ : 0 < θ ∧ θ < Real.pi) :
    Real.cos (Real.pi / 2 + 2 * θ) = - (4 * Real.sqrt 2) / 9 := 
sorry

end cos_pi_over_2_plus_2theta_l1199_119985


namespace remainder_549547_div_7_l1199_119944

theorem remainder_549547_div_7 : 549547 % 7 = 5 :=
by
  sorry

end remainder_549547_div_7_l1199_119944


namespace problem_1_problem_2_problem_3_l1199_119982

variable (α : ℝ)
variable (tan_alpha_two : Real.tan α = 2)

theorem problem_1 : (2 * Real.cos α + 3 * Real.sin α) / (3 * Real.cos α + Real.sin α) = 8 / 5 :=
by
  sorry

theorem problem_2 : (Real.cos α ^ 2 + Real.sin α * Real.cos α) / (2 * Real.sin α * Real.cos α + Real.sin α ^ 2) = 3 / 8 :=
by
  sorry

theorem problem_3 : (Real.sin α ^ 2 - Real.sin α * Real.cos α + 2) = 12 / 5 :=
by
  sorry

end problem_1_problem_2_problem_3_l1199_119982


namespace ben_less_than_jack_l1199_119968

def jack_amount := 26
def total_amount := 50
def eric_ben_difference := 10

theorem ben_less_than_jack (E B J : ℕ) (h1 : E = B - eric_ben_difference) (h2 : J = jack_amount) (h3 : E + B + J = total_amount) :
  J - B = 9 :=
by sorry

end ben_less_than_jack_l1199_119968


namespace value_of_k_l1199_119948

theorem value_of_k (k : ℕ) : (∃ b : ℕ, x^2 - 20 * x + k = (x + b)^2) → k = 100 := by
  sorry

end value_of_k_l1199_119948


namespace class_students_l1199_119924

theorem class_students (A B : ℕ) 
  (h1 : A + B = 85) 
  (h2 : (3 * A) / 8 + (3 * B) / 5 = 42) : 
  A = 40 ∧ B = 45 :=
by
  sorry

end class_students_l1199_119924


namespace fraction_comparison_l1199_119917

theorem fraction_comparison : (5555553 / 5555557 : ℚ) > (6666664 / 6666669 : ℚ) :=
  sorry

end fraction_comparison_l1199_119917


namespace green_chips_correct_l1199_119926

-- Definitions
def total_chips : ℕ := 120
def blue_chips : ℕ := total_chips / 4
def red_chips : ℕ := total_chips * 20 / 100
def yellow_chips : ℕ := total_chips / 10
def non_green_chips : ℕ := blue_chips + red_chips + yellow_chips
def green_chips : ℕ := total_chips - non_green_chips

-- Statement to prove
theorem green_chips_correct : green_chips = 54 := by
  -- Proof would go here
  sorry

end green_chips_correct_l1199_119926


namespace xiaohong_money_l1199_119913

def cost_kg_pears (x : ℝ) := x

def cost_kg_apples (x : ℝ) := x + 1.1

theorem xiaohong_money (x : ℝ) (hx : 6 * x - 3 = 5 * (x + 1.1) - 4) : 6 * x - 3 = 24 :=
by sorry

end xiaohong_money_l1199_119913


namespace tan_prod_eq_sqrt_seven_l1199_119965

theorem tan_prod_eq_sqrt_seven : 
  let x := (Real.pi / 7) 
  let y := (2 * Real.pi / 7)
  let z := (3 * Real.pi / 7)
  Real.tan x * Real.tan y * Real.tan z = Real.sqrt 7 :=
by
  sorry

end tan_prod_eq_sqrt_seven_l1199_119965
