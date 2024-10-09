import Mathlib

namespace runner_speed_comparison_l2221_222172

theorem runner_speed_comparison
  (t1 t2 : ℕ → ℝ) -- function to map lap-time.
  (s v1 v2 : ℝ)  -- speed of runners v1 and v2 respectively, and the street distance s.
  (h1 : t1 1 < t2 1) -- first runner overtakes the second runner twice implying their lap-time comparison.
  (h2 : ∀ n, t1 (n + 1) = t1 n + t1 1) -- lap time consistency for runner 1
  (h3 : ∀ n, t2 (n + 1) = t2 n + t2 1) -- lap time consistency for runner 2
  (h4 : t1 3 < t2 2) -- first runner completes 3 laps faster than second runner completes 2 laps
   : 2 * v2 ≤ v1 := sorry

end runner_speed_comparison_l2221_222172


namespace domino_covering_impossible_odd_squares_l2221_222137

theorem domino_covering_impossible_odd_squares
  (board1 : ℕ) -- 24 squares
  (board2 : ℕ) -- 21 squares
  (board3 : ℕ) -- 23 squares
  (board4 : ℕ) -- 35 squares
  (board5 : ℕ) -- 63 squares
  (h1 : board1 = 24)
  (h2 : board2 = 21)
  (h3 : board3 = 23)
  (h4 : board4 = 35)
  (h5 : board5 = 63) :
  (board2 % 2 = 1) ∧ (board3 % 2 = 1) ∧ (board4 % 2 = 1) ∧ (board5 % 2 = 1) :=
by {
  sorry
}

end domino_covering_impossible_odd_squares_l2221_222137


namespace shoe_size_combination_l2221_222177

theorem shoe_size_combination (J A : ℕ) (hJ : J = 7) (hA : A = 2 * J) : J + A = 21 := by
  sorry

end shoe_size_combination_l2221_222177


namespace find_a_plus_b_l2221_222166

noncomputable def f (a b x : ℝ) := a ^ x + b

theorem find_a_plus_b (a b : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) 
  (dom1 : f a b (-2) = -2) (dom2 : f a b 0 = 0) :
  a + b = (Real.sqrt 3) / 3 - 3 :=
by
  unfold f at dom1 dom2
  sorry

end find_a_plus_b_l2221_222166


namespace relation_between_A_and_B_l2221_222146

-- Define the sets A and B
def A : Set ℤ := { x | ∃ k : ℕ, x = 7 * k + 3 }
def B : Set ℤ := { x | ∃ k : ℤ, x = 7 * k - 4 }

-- Prove the relationship between A and B
theorem relation_between_A_and_B : A ⊆ B :=
by
  sorry

end relation_between_A_and_B_l2221_222146


namespace gcd_54_180_l2221_222188

theorem gcd_54_180 : Nat.gcd 54 180 = 18 := by
  sorry

end gcd_54_180_l2221_222188


namespace find_g_five_l2221_222191

def g (a b c x : ℝ) : ℝ := a * x^7 + b * x^6 + c * x - 3

theorem find_g_five (a b c : ℝ) (h : g a b c (-5) = -3) : g a b c 5 = 31250 * b - 3 := 
sorry

end find_g_five_l2221_222191


namespace range_of_8x_plus_y_l2221_222192

theorem range_of_8x_plus_y (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_condition : 1 / x + 2 / y = 2) : 8 * x + y ≥ 9 :=
by
  sorry

end range_of_8x_plus_y_l2221_222192


namespace centrally_symmetric_equidecomposable_l2221_222168

-- Assume we have a type for Polyhedra
variable (Polyhedron : Type)

-- Conditions
variable (sameVolume : Polyhedron → Polyhedron → Prop)
variable (centrallySymmetricFaces : Polyhedron → Prop)
variable (equidecomposable : Polyhedron → Polyhedron → Prop)

-- Theorem statement
theorem centrally_symmetric_equidecomposable 
  (P Q : Polyhedron) 
  (h1 : sameVolume P Q) 
  (h2 : centrallySymmetricFaces P) 
  (h3 : centrallySymmetricFaces Q) :
  equidecomposable P Q := 
sorry

end centrally_symmetric_equidecomposable_l2221_222168


namespace directrix_of_parabola_l2221_222198

theorem directrix_of_parabola (y x : ℝ) (p : ℝ) (h₁ : y = 8 * x ^ 2) (h₂ : y = 4 * p * x) : 
  p = 2 ∧ (y = -p ↔ y = -2) :=
by
  sorry

end directrix_of_parabola_l2221_222198


namespace pebbles_divisibility_impossibility_l2221_222160

def initial_pebbles (K A P D : Nat) := K + A + P + D

theorem pebbles_divisibility_impossibility 
  (K A P D : Nat)
  (hK : K = 70)
  (hA : A = 30)
  (hP : P = 21)
  (hD : D = 45) :
  ¬ (∃ n : Nat, initial_pebbles K A P D = 4 * n) :=
by
  sorry

end pebbles_divisibility_impossibility_l2221_222160


namespace surface_area_three_dimensional_shape_l2221_222134

-- Define the edge length of the largest cube
def edge_length_large : ℕ := 5

-- Define the condition for dividing the edge of the attachment face of the large cube into five equal parts
def divided_into_parts (edge_length : ℕ) (parts : ℕ) : Prop :=
  parts = 5

-- Define the condition that the edge lengths of all three blocks are different
def edge_lengths_different (e1 e2 e3 : ℕ) : Prop :=
  e1 ≠ e2 ∧ e1 ≠ e3 ∧ e2 ≠ e3

-- Define the surface area formula for a cube
def surface_area (s : ℕ) : ℕ :=
  6 * s^2

-- State the problem as a theorem
theorem surface_area_three_dimensional_shape (e1 e2 e3 : ℕ) (h1 : e1 = edge_length_large)
    (h2 : divided_into_parts e1 5) (h3 : edge_lengths_different e1 e2 e3) : 
    surface_area e1 + (surface_area e2 + surface_area e3 - 4 * (e2 * e3)) = 270 :=
sorry

end surface_area_three_dimensional_shape_l2221_222134


namespace minimum_steps_to_catch_thief_l2221_222114

-- Definitions of positions A, B, C, D, etc., along the board
-- Assuming the positions and movement rules are predefined somewhere in the environment.
-- For a simple abstract model, we assume the following:
-- The positions are nodes in a graph, and each move is one step along the edges of this graph.

def Position : Type := String -- This can be refined to reflect the actual chessboard structure.
def neighbor (p1 p2 : Position) : Prop := sorry -- Predicate defining that p1 and p2 are neighbors.

-- Positions are predefined for simplicity.
def A : Position := "A"
def B : Position := "B"
def C : Position := "C"
def D : Position := "D"
def F : Position := "F"

-- Condition: policeman and thief take turns moving, starting with the policeman.
-- Initial positions of the policeman and the thief.
def policemanStart : Position := A
def thiefStart : Position := B

-- Statement: Prove that the policeman can catch the thief in a minimum of 4 moves.
theorem minimum_steps_to_catch_thief (policeman thief : Position) (turns : ℕ) :
  policeman = policemanStart →
  thief = thiefStart →
  (∀ t < turns, (neighbor policeman thief)) →
  (turns = 4) :=
sorry

end minimum_steps_to_catch_thief_l2221_222114


namespace find_n_in_geometric_sequence_l2221_222139

def geometric_sequence (an : ℕ → ℝ) (n : ℕ) : Prop :=
  ∃ q : ℝ, ∀ k : ℕ, an (k + 1) = an k * q

theorem find_n_in_geometric_sequence (a : ℕ → ℝ)
  (h1 : a 1 = 1)
  (h2 : ∃ q : ℝ, ∀ n, a (n + 1) = a n * q)
  (h3 : ∀ q : ℝ, a n = a 1 * a 2 * a 3 * a 4 * a 5) :
  n = 11 :=
sorry

end find_n_in_geometric_sequence_l2221_222139


namespace octal_addition_correct_l2221_222197

def octal_to_decimal (n : ℕ) : ℕ := 
  /- function to convert an octal number to decimal goes here -/
  sorry

def decimal_to_octal (n : ℕ) : ℕ :=
  /- function to convert a decimal number to octal goes here -/
  sorry

theorem octal_addition_correct :
  let a := 236 
  let b := 521
  let c := 74
  let sum_decimal := octal_to_decimal a + octal_to_decimal b + octal_to_decimal c
  decimal_to_octal sum_decimal = 1063 :=
by
  sorry

end octal_addition_correct_l2221_222197


namespace symmetric_parabola_equation_l2221_222180

theorem symmetric_parabola_equation (x y : ℝ) (h : y^2 = 2 * x) : (y^2 = -2 * (x + 2)) :=
by
  sorry

end symmetric_parabola_equation_l2221_222180


namespace sum_of_fourth_powers_l2221_222111

theorem sum_of_fourth_powers (n : ℤ) 
  (h : n * (n + 1) * (n + 2) = 12 * (n + (n + 1) + (n + 2))) : 
  (n^4 + (n + 1)^4 + (n + 2)^4) = 7793 := 
by 
  sorry

end sum_of_fourth_powers_l2221_222111


namespace determine_sum_of_digits_l2221_222170

theorem determine_sum_of_digits (x y : ℕ) (hx : x < 10) (hy : y < 10)
  (h : ∃ a b c d : ℕ, 
       a = 30 + x ∧ b = 10 * y + 4 ∧
       c = (a * (b % 10)) % 100 ∧ 
       d = (a * (b % 10)) / 100 ∧ 
       10 * d + c = 156) :
  x + y = 13 :=
by
  sorry

end determine_sum_of_digits_l2221_222170


namespace algebraic_expression_value_l2221_222184

theorem algebraic_expression_value (x : ℝ) (h : x = Real.sqrt 19 - 1) : x^2 + 2 * x + 2 = 20 := by
  sorry

end algebraic_expression_value_l2221_222184


namespace eggs_collected_l2221_222163

def total_eggs_collected (b1 e1 b2 e2 : ℕ) : ℕ :=
  b1 * e1 + b2 * e2

theorem eggs_collected :
  total_eggs_collected 450 36 405 42 = 33210 :=
by
  sorry

end eggs_collected_l2221_222163


namespace ratio_of_radii_l2221_222135

theorem ratio_of_radii 
  (a b : ℝ)
  (h1 : ∀ (a b : ℝ), π * b^2 - π * a^2 = 4 * π * a^2) : 
  a / b = 1 / Real.sqrt 5 :=
by
  sorry

end ratio_of_radii_l2221_222135


namespace solution_product_l2221_222171

theorem solution_product (p q : ℝ) (hpq : p ≠ q) (h1 : (x-3)*(3*x+18) = x^2-15*x+54) (hp : (x - p) * (x - q) = x^2 - 12 * x + 54) :
  (p + 2) * (q + 2) = -80 := sorry

end solution_product_l2221_222171


namespace car_speed_l2221_222104

theorem car_speed (distance time : ℝ) (h_distance : distance = 275) (h_time : time = 5) : (distance / time = 55) :=
by
  sorry

end car_speed_l2221_222104


namespace geometric_probability_l2221_222194

noncomputable def probability_point_within_rectangle (l w : ℝ) (A_rectangle A_circle : ℝ) : ℝ :=
  A_rectangle / A_circle

theorem geometric_probability (l w : ℝ) (r : ℝ) (A_rectangle : ℝ) (h_length : l = 4) 
  (h_width : w = 3) (h_radius : r = 2.5) (h_area_rectangle : A_rectangle = 12) :
  A_rectangle / (Real.pi * r^2) = 48 / (25 * Real.pi) :=
by
  sorry

end geometric_probability_l2221_222194


namespace geometric_sequence_condition_l2221_222100

theorem geometric_sequence_condition (A B q : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ)
  (hSn_def : ∀ n, S n = A * q^n + B) (hq_ne_zero : q ≠ 0) :
  (∀ n, a n = S n - S (n-1)) → (A = -B) ↔ (∀ n, a n = A * (q - 1) * q^(n-1)) := 
sorry

end geometric_sequence_condition_l2221_222100


namespace li_ming_estimated_weight_is_correct_l2221_222101

-- Define the regression equation as a function
def regression_equation (x : ℝ) : ℝ := 0.7 * x - 52

-- Define the height of Li Ming
def li_ming_height : ℝ := 180

-- The estimated weight according to the regression equation
def estimated_weight : ℝ := regression_equation li_ming_height

-- Theorem statement: Given the height, the weight should be 74
theorem li_ming_estimated_weight_is_correct : estimated_weight = 74 :=
by
  sorry

end li_ming_estimated_weight_is_correct_l2221_222101


namespace odd_function_f_a_zero_l2221_222196

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + (a + 1) * Real.cos x + x

theorem odd_function_f_a_zero (a : ℝ) (h_odd : ∀ x : ℝ, f a (-x) = -f a x) : f a a = 0 := 
sorry

end odd_function_f_a_zero_l2221_222196


namespace factorize_difference_of_squares_l2221_222120

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factorize_difference_of_squares_l2221_222120


namespace kangaroo_chase_l2221_222145

noncomputable def time_to_catch_up (jumps_baby: ℕ) (jumps_mother: ℕ) (time_period: ℕ) 
  (jump_dist_mother: ℕ) (jump_dist_reduction_factor: ℕ) 
  (initial_baby_jumps: ℕ): ℕ :=
  let jump_dist_baby := jump_dist_mother / jump_dist_reduction_factor
  let distance_mother := jumps_mother * jump_dist_mother
  let distance_baby := jumps_baby * jump_dist_baby
  let relative_velocity := distance_mother - distance_baby
  let initial_distance := initial_baby_jumps * jump_dist_baby
  (initial_distance / relative_velocity) * time_period

theorem kangaroo_chase :
 ∀ (jumps_baby: ℕ) (jumps_mother: ℕ) (time_period: ℕ) 
   (jump_dist_mother: ℕ) (jump_dist_reduction_factor: ℕ) 
   (initial_baby_jumps: ℕ),
  jumps_baby = 5 ∧ jumps_mother = 3 ∧ time_period = 2 ∧ 
  jump_dist_mother = 6 ∧ jump_dist_reduction_factor = 3 ∧ 
  initial_baby_jumps = 12 →
  time_to_catch_up jumps_baby jumps_mother time_period jump_dist_mother 
    jump_dist_reduction_factor initial_baby_jumps = 6 := 
by
  intros jumps_baby jumps_mother time_period jump_dist_mother 
    jump_dist_reduction_factor initial_baby_jumps _; sorry

end kangaroo_chase_l2221_222145


namespace first_player_wins_l2221_222159

-- Define the initial conditions
def initial_pieces : ℕ := 1
def final_pieces (m n : ℕ) : ℕ := m * n
def num_moves (pieces : ℕ) : ℕ := pieces - 1

-- Theorem statement: Given the initial dimensions and the game rules,
-- prove that the first player will win.
theorem first_player_wins (m n : ℕ) (h_m : m = 6) (h_n : n = 8) : 
  (num_moves (final_pieces m n)) % 2 = 0 → false :=
by
  -- The solution details and the proof will be here.
  sorry

end first_player_wins_l2221_222159


namespace transport_tax_to_be_paid_l2221_222183

noncomputable def engine_power : ℕ := 150
noncomputable def tax_rate : ℕ := 20
noncomputable def annual_tax : ℕ := engine_power * tax_rate
noncomputable def months_used : ℕ := 8
noncomputable def prorated_tax : ℕ := (months_used * annual_tax) / 12

theorem transport_tax_to_be_paid : prorated_tax = 2000 := 
by 
  -- sorry is used to skip the proof step
  sorry

end transport_tax_to_be_paid_l2221_222183


namespace mushrooms_on_log_l2221_222113

theorem mushrooms_on_log :
  ∃ (G : ℕ), ∃ (S : ℕ), S = 9 * G ∧ G + S = 30 ∧ G = 3 :=
by
  sorry

end mushrooms_on_log_l2221_222113


namespace books_given_away_l2221_222127

theorem books_given_away (original_books : ℝ) (books_left : ℝ) (books_given : ℝ) 
    (h1 : original_books = 54.0) 
    (h2 : books_left = 31) : 
    books_given = original_books - books_left → books_given = 23 :=
by
  sorry

end books_given_away_l2221_222127


namespace preferred_point_condition_l2221_222162

theorem preferred_point_condition (x y : ℝ) (h₁ : x^2 + y^2 ≤ 2008)
  (cond : ∀ x' y', (x'^2 + y'^2 ≤ 2008) → (x' ≤ x → y' ≥ y) → (x = x' ∧ y = y')) :
  x^2 + y^2 = 2008 ∧ x ≤ 0 ∧ y ≥ 0 :=
by
  sorry

end preferred_point_condition_l2221_222162


namespace person_before_you_taller_than_you_l2221_222186

-- Define the persons involved in the problem.
variable (Person : Type)
variable (Taller : Person → Person → Prop)
variable (P Q You : Person)

-- The conditions given in the problem.
axiom standing_queue : Taller P Q
axiom queue_structure : You = Q

-- The question we need to prove, which is the correct answer to the problem.
theorem person_before_you_taller_than_you : Taller P You :=
by
  sorry

end person_before_you_taller_than_you_l2221_222186


namespace regular_ducks_sold_l2221_222158

theorem regular_ducks_sold (R : ℕ) (h1 : 3 * R + 5 * 185 = 1588) : R = 221 :=
by {
  sorry
}

end regular_ducks_sold_l2221_222158


namespace least_prime_factor_of_11_pow_5_minus_11_pow_4_l2221_222190

theorem least_prime_factor_of_11_pow_5_minus_11_pow_4 : 
  Nat.minFac (11^5 - 11^4) = 2 := 
by sorry

end least_prime_factor_of_11_pow_5_minus_11_pow_4_l2221_222190


namespace toys_produced_per_day_l2221_222189

theorem toys_produced_per_day :
  (3400 / 5 = 680) :=
by
  sorry

end toys_produced_per_day_l2221_222189


namespace domain_of_function_l2221_222164

theorem domain_of_function :
  { x : ℝ | 0 ≤ 2 * x - 10 ∧ 2 * x - 10 ≠ 0 } = { x : ℝ | x > 5 } :=
by
  sorry

end domain_of_function_l2221_222164


namespace no_natural_numbers_condition_l2221_222130

theorem no_natural_numbers_condition :
  ¬ ∃ (a : Fin 2018 → ℕ), ∀ i : Fin 2018,
    ∃ k : ℕ, (a i) ^ 2018 + a ((i + 1) % 2018) = 5 ^ k :=
by sorry

end no_natural_numbers_condition_l2221_222130


namespace outdoor_section_area_l2221_222123

theorem outdoor_section_area :
  ∀ (width length : ℕ), width = 4 → length = 6 → (width * length = 24) :=
by
  sorry

end outdoor_section_area_l2221_222123


namespace find_constant_c_l2221_222138

def f: ℝ → ℝ := sorry

noncomputable def constant_c := 8

theorem find_constant_c (h : ∀ x : ℝ, f x + 3 * f (constant_c - x) = x) (h2 : f 2 = 2) : 
  constant_c = 8 :=
sorry

end find_constant_c_l2221_222138


namespace proportion_Q_to_R_l2221_222182

theorem proportion_Q_to_R (q r : ℕ) (h1 : 3 * q + 5 * r = 1000) (h2 : 4 * r - 2 * q = 250) : q = r :=
by sorry

end proportion_Q_to_R_l2221_222182


namespace daily_sales_profit_45_selling_price_for_1200_profit_l2221_222156

-- Definitions based on given conditions

def cost_price : ℤ := 30
def base_selling_price : ℤ := 40
def base_sales_volume : ℤ := 80
def price_increase_effect : ℤ := 2
def max_selling_price : ℤ := 55

-- Part (1): Prove that for a selling price of 45 yuan, the daily sales profit is 1050 yuan.
theorem daily_sales_profit_45 :
  let selling_price := 45
  let increase_in_price := selling_price - base_selling_price
  let decrease_in_volume := increase_in_price * price_increase_effect
  let new_sales_volume := base_sales_volume - decrease_in_volume
  let profit_per_item := selling_price - cost_price
  let daily_profit := profit_per_item * new_sales_volume
  daily_profit = 1050 := by sorry

-- Part (2): Prove that to achieve a daily profit of 1200 yuan, the selling price should be 50 yuan.
theorem selling_price_for_1200_profit :
  let target_profit := 1200
  ∃ (selling_price : ℤ), 
  let increase_in_price := selling_price - base_selling_price
  let decrease_in_volume := increase_in_price * price_increase_effect
  let new_sales_volume := base_sales_volume - decrease_in_volume
  let profit_per_item := selling_price - cost_price
  let daily_profit := profit_per_item * new_sales_volume
  daily_profit = target_profit ∧ selling_price ≤ max_selling_price ∧ selling_price = 50 := by sorry

end daily_sales_profit_45_selling_price_for_1200_profit_l2221_222156


namespace triangle_angle_A_eq_pi_div_3_triangle_area_l2221_222106

variable (A B C a b c : ℝ)
variable (S : ℝ)

-- First part: Proving A = π / 3
theorem triangle_angle_A_eq_pi_div_3 (h1 : a * Real.cos C + c * Real.cos A = 2 * b * Real.cos A)
                                      (h2 : a ≠ 0) (h3 : b ≠ 0) (h4 : c ≠ 0) (h5 : A > 0) (h6 : A < Real.pi) :
  A = Real.pi / 3 :=
sorry

-- Second part: Finding the area of the triangle
theorem triangle_area (h1 : a * Real.cos C + c * Real.cos A = 2 * b * Real.cos A)
                      (h2 : b + c = Real.sqrt 10) (h3 : a = 2) (h4 : A = Real.pi / 3) :
  S = Real.sqrt 3 / 2 :=
sorry

end triangle_angle_A_eq_pi_div_3_triangle_area_l2221_222106


namespace map_x_eq_3_and_y_eq_2_under_z_squared_to_uv_l2221_222167

theorem map_x_eq_3_and_y_eq_2_under_z_squared_to_uv :
  (∀ (z : ℂ), (z = 3 + I * z.im) → ((z^2).re = 9 - (9*z.im^2) / 36)) ∧
  (∀ (z : ℂ), (z = z.re + I * 2) → ((z^2).re = (4*z.re^2) / 16 - 4)) :=
by 
  sorry

end map_x_eq_3_and_y_eq_2_under_z_squared_to_uv_l2221_222167


namespace paving_cost_l2221_222155

theorem paving_cost (l w r : ℝ) (h_l : l = 5.5) (h_w : w = 4) (h_r : r = 700) :
  l * w * r = 15400 :=
by sorry

end paving_cost_l2221_222155


namespace janet_home_time_l2221_222165

def blocks_north := 3
def blocks_west := 7 * blocks_north
def blocks_south := blocks_north
def blocks_east := 2 * blocks_south -- Initially mistaken, recalculating needed
def remaining_blocks_west := blocks_west - blocks_east
def total_blocks_home := blocks_south + remaining_blocks_west
def walking_speed := 2 -- blocks per minute

theorem janet_home_time :
  (blocks_south + remaining_blocks_west) / walking_speed = 9 := by
  -- We assume that Lean can handle the arithmetic properly here.
  sorry

end janet_home_time_l2221_222165


namespace regular_polygon_sides_l2221_222142

theorem regular_polygon_sides (N : ℕ) (h : ∀ θ, θ = 140 → N * (180 -θ) = 360) : N = 9 :=
by
  sorry

end regular_polygon_sides_l2221_222142


namespace cos_300_eq_half_l2221_222148

theorem cos_300_eq_half : Real.cos (2 * π * (300 / 360)) = 1 / 2 :=
by
  sorry

end cos_300_eq_half_l2221_222148


namespace find_a1_l2221_222173

theorem find_a1 (a : ℕ → ℝ) (h_rec : ∀ n, a (n + 1) = a n / (2 * a n + 1)) (h_init : a 3 = 1 / 5) : a 1 = 1 := by
  sorry

end find_a1_l2221_222173


namespace largest_common_term_in_range_l2221_222122

def seq1 (n : ℕ) : ℕ := 5 + 9 * n
def seq2 (m : ℕ) : ℕ := 3 + 8 * m

theorem largest_common_term_in_range :
  ∃ (a : ℕ) (n m : ℕ), seq1 n = a ∧ seq2 m = a ∧ 1 ≤ a ∧ a ≤ 200 ∧ (∀ b, (∃ nf mf, seq1 nf = b ∧ seq2 mf = b ∧ 1 ≤ b ∧ b ≤ 200) → b ≤ a) :=
sorry

end largest_common_term_in_range_l2221_222122


namespace sandy_younger_than_molly_l2221_222140

variable (s m : ℕ)
variable (h_ratio : 7 * m = 9 * s)
variable (h_sandy : s = 56)

theorem sandy_younger_than_molly : 
  m - s = 16 := 
by
  sorry

end sandy_younger_than_molly_l2221_222140


namespace arithmetic_sequence_sum_geometric_sequence_ratio_l2221_222119

-- Definition of an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) :=
  ∀ n, a (n + 1) = a n + d

-- Definition of a geometric sequence
def is_geometric_sequence (a : ℕ → ℕ) (q : ℚ) :=
  ∀ n, a (n + 1) = a n * q
  
-- Prove the sum of the first n terms for an arithmetic sequence
theorem arithmetic_sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ) :
  a 1 = 3 ∧ (∀ n, S n = (n * (3 + a (n + 1) - 1)) / 2) ∧ is_arithmetic_sequence a 4 → 
  S n = 2 * n^2 + n :=
sorry

-- Prove the range of the common ratio for a geometric sequence
theorem geometric_sequence_ratio (a : ℕ → ℕ) (S : ℕ → ℚ) (q : ℚ) :
  a 1 = 3 ∧ is_geometric_sequence a q ∧ ∃ lim : ℚ, (∀ n, S n = (a 1 * (1 - q^n)) / (1 - q)) ∧ lim < 12 → 
  -1 < q ∧ q < 1 ∧ q ≠ 0 ∧ q < 3/4 :=
sorry

end arithmetic_sequence_sum_geometric_sequence_ratio_l2221_222119


namespace set_intersection_complement_eq_l2221_222126

open Set

variable (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)

theorem set_intersection_complement_eq {U : Set ℕ} {M : Set ℕ} {N : Set ℕ}
    (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2, 3}) (hN : N = {3, 4, 5}) :
    (U \ M) ∩ N = {4, 5} :=
by
  sorry

end set_intersection_complement_eq_l2221_222126


namespace inequality_solution_l2221_222154

theorem inequality_solution :
  {x : ℝ | ((x - 1) * (x - 4) * (x - 5)) / ((x - 2) * (x - 6) * (x - 7)) > 0} = 
  {x : ℝ | (1 < x ∧ x < 2) ∨ (4 < x ∧ x < 5) ∨ (6 < x ∧ x < 7)} :=
sorry

end inequality_solution_l2221_222154


namespace amount_paid_l2221_222105

-- Defining the conditions as constants
def cost_of_apple : ℝ := 0.75
def change_received : ℝ := 4.25

-- Stating the theorem that needs to be proved
theorem amount_paid (a : ℝ) : a = cost_of_apple + change_received :=
by
  sorry

end amount_paid_l2221_222105


namespace tan_alpha_equiv_l2221_222185

theorem tan_alpha_equiv (α : ℝ) (h : Real.tan α = 2) : 
    (2 * Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 1 := 
by 
  sorry

end tan_alpha_equiv_l2221_222185


namespace ball_speed_is_20_l2221_222141

def ball_flight_time : ℝ := 8
def collie_speed : ℝ := 5
def collie_catch_time : ℝ := 32

noncomputable def collie_distance : ℝ := collie_speed * collie_catch_time

theorem ball_speed_is_20 :
  collie_distance = ball_flight_time * 20 :=
by
  sorry

end ball_speed_is_20_l2221_222141


namespace tan_angle_addition_l2221_222195

theorem tan_angle_addition (x : ℝ) (h : Real.tan x = 2) : Real.tan (x + Real.pi / 3) = (5 * Real.sqrt 3 + 8) / -11 := by
  sorry

end tan_angle_addition_l2221_222195


namespace angle_conversion_l2221_222181

theorem angle_conversion : (1 : ℝ) * (π / 180) * (-225) = - (5 * π / 4) :=
by
  sorry

end angle_conversion_l2221_222181


namespace max_profit_l2221_222116

noncomputable def annual_profit (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 5 then 
    -0.5 * x^2 + 3.5 * x - 0.5 
  else if x > 5 then 
    17 - 2.5 * x 
  else 
    0

theorem max_profit :
  ∀ x : ℝ, (annual_profit 3.5 = 5.625) :=
by
  -- Proof omitted
  sorry

end max_profit_l2221_222116


namespace intersection_M_N_l2221_222149

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l2221_222149


namespace amount_each_girl_gets_l2221_222128

theorem amount_each_girl_gets
  (B G : ℕ) 
  (total_sum : ℝ)
  (amount_each_boy : ℝ)
  (sum_boys_girls : B + G = 100)
  (total_sum_distributed : total_sum = 312)
  (amount_boy : amount_each_boy = 3.60)
  (B_approx : B = 60) :
  (total_sum - amount_each_boy * B) / G = 2.40 := 
by 
  sorry

end amount_each_girl_gets_l2221_222128


namespace set_inter_and_complement_l2221_222102

def U : Set ℕ := {2, 3, 4, 5, 6, 7}
def A : Set ℕ := {4, 5, 7}
def B : Set ℕ := {4, 6}

theorem set_inter_and_complement :
  A ∩ (U \ B) = {5, 7} := by
  sorry

end set_inter_and_complement_l2221_222102


namespace mehki_age_l2221_222161

theorem mehki_age (Z J M : ℕ) (h1 : Z = 6) (h2 : J = Z - 4) (h3 : M = 2 * (J + Z)) : M = 16 := by
  sorry

end mehki_age_l2221_222161


namespace probability_y_greater_than_x_equals_3_4_l2221_222143

noncomputable def probability_y_greater_than_x : Real :=
  let total_area : Real := 1000 * 4034
  let triangle_area : Real := 0.5 * 1000 * (4034 - 1000)
  let rectangle_area : Real := 3034 * 4034
  let area_y_greater_than_x : Real := triangle_area + rectangle_area
  area_y_greater_than_x / total_area

theorem probability_y_greater_than_x_equals_3_4 :
  probability_y_greater_than_x = 3 / 4 :=
sorry

end probability_y_greater_than_x_equals_3_4_l2221_222143


namespace determine_omega_l2221_222193

noncomputable def f (ω : ℝ) (ϕ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + ϕ)

-- Conditions
variables (ω : ℝ) (ϕ : ℝ)
axiom omega_pos : ω > 0
axiom phi_bound : abs ϕ < Real.pi / 2
axiom symm_condition1 : ∀ x, f ω ϕ (Real.pi / 4 - x) = -f ω ϕ (Real.pi / 4 + x)
axiom symm_condition2 : ∀ x, f ω ϕ (-Real.pi / 2 - x) = f ω ϕ x
axiom monotonic_condition : ∀ x1 x2, 0 < x1 → x1 < x2 → x2 < Real.pi / 8 → f ω ϕ x1 < f ω ϕ x2

theorem determine_omega : ω = 1 ∨ ω = 5 :=
sorry

end determine_omega_l2221_222193


namespace cubes_with_even_red_faces_l2221_222125

theorem cubes_with_even_red_faces :
  let block_dimensions := (5, 5, 1)
  let painted_sides := 6
  let total_cubes := 25
  let cubes_with_2_red_faces := 16
  cubes_with_2_red_faces = 16 := by
  sorry

end cubes_with_even_red_faces_l2221_222125


namespace geometric_seq_sum_l2221_222110

-- Definitions of the conditions
def a (n : ℕ) : ℤ :=
  match n with
  | 0 => 1
  | _ => (-3)^(n - 1)

theorem geometric_seq_sum : 
  a 0 + |a 1| + a 2 + |a 3| + a 4 = 121 := by
  sorry

end geometric_seq_sum_l2221_222110


namespace find_exponent_l2221_222174

theorem find_exponent (n : ℕ) (some_number : ℕ) (h1 : n = 27) 
  (h2 : 2 ^ (2 * n) + 2 ^ (2 * n) + 2 ^ (2 * n) + 2 ^ (2 * n) = 4 ^ some_number) :
  some_number = 28 :=
by 
  sorry

end find_exponent_l2221_222174


namespace three_digit_numbers_div_by_17_l2221_222199

theorem three_digit_numbers_div_by_17 : ∃ n : ℕ, n = 53 ∧ 
  let min_k := Nat.ceil (100 / 17)
  let max_k := Nat.floor (999 / 17)
  min_k = 6 ∧ max_k = 58 ∧ (max_k - min_k + 1) = n :=
by
  sorry

end three_digit_numbers_div_by_17_l2221_222199


namespace symm_diff_complement_symm_diff_union_subset_symm_diff_inter_subset_l2221_222176

-- Definitions of sequences of events and symmetric difference
variable (A : ℕ → Set α) (B : ℕ → Set α)

-- Definition of symmetric difference
def symm_diff (S T : Set α) : Set α := (S \ T) ∪ (T \ S)

-- Theorems to be proven
theorem symm_diff_complement (A1 B1 : Set α) :
  symm_diff A1 B1 = symm_diff (Set.compl A1) (Set.compl B1) := sorry

theorem symm_diff_union_subset :
  symm_diff (⋃ n, A n) (⋃ n, B n) ⊆ ⋃ n, symm_diff (A n) (B n) := sorry

theorem symm_diff_inter_subset :
  symm_diff (⋂ n, A n) (⋂ n, B n) ⊆ ⋃ n, symm_diff (A n) (B n) := sorry

end symm_diff_complement_symm_diff_union_subset_symm_diff_inter_subset_l2221_222176


namespace correct_expansion_l2221_222187

variables {x y : ℝ}

theorem correct_expansion : 
  (-x + y)^2 = x^2 - 2 * x * y + y^2 := sorry

end correct_expansion_l2221_222187


namespace seven_lines_regions_l2221_222175

theorem seven_lines_regions (n : ℕ) (hn : n = 7) (h1 : ¬ ∃ l1 l2 : ℝ, l1 = l2) (h2 : ∀ l1 l2 l3 : ℝ, ¬ (l1 ≠ l2 ∧ l2 ≠ l3 ∧ l1 ≠ l3 ∧ (l1 = l2 ∧ l2 = l3))) :
  ∃ R : ℕ, R = 29 :=
by
  sorry

end seven_lines_regions_l2221_222175


namespace phone_not_answered_prob_l2221_222179

noncomputable def P_not_answered_within_4_rings : ℝ :=
  let P1 := 1 - 0.1
  let P2 := 1 - 0.3
  let P3 := 1 - 0.4
  let P4 := 1 - 0.1
  P1 * P2 * P3 * P4

theorem phone_not_answered_prob : 
  P_not_answered_within_4_rings = 0.3402 := 
by 
  -- The detailed steps and proof will be implemented here 
  sorry

end phone_not_answered_prob_l2221_222179


namespace scientific_notation_of_75500000_l2221_222131

theorem scientific_notation_of_75500000 :
  ∃ (a : ℝ) (n : ℤ), 75500000 = a * 10 ^ n ∧ a = 7.55 ∧ n = 7 :=
by {
  sorry
}

end scientific_notation_of_75500000_l2221_222131


namespace Carlos_gave_Rachel_21_blocks_l2221_222157

def initial_blocks : Nat := 58
def remaining_blocks : Nat := 37
def given_blocks : Nat := initial_blocks - remaining_blocks

theorem Carlos_gave_Rachel_21_blocks : given_blocks = 21 :=
by
  sorry

end Carlos_gave_Rachel_21_blocks_l2221_222157


namespace range_of_a_for_monotonic_increasing_f_l2221_222153

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - a * x - 2 * Real.log x

theorem range_of_a_for_monotonic_increasing_f (a : ℝ) (h : ∀ x : ℝ, 1 ≤ x → (x - a - 2 / x) ≥ 0) : a ≤ -1 :=
by {
  -- Placeholder for the detailed proof steps
  sorry
}

end range_of_a_for_monotonic_increasing_f_l2221_222153


namespace algebraic_expression_l2221_222129

theorem algebraic_expression (x : ℝ) (h : x - 1/x = 3) : x^2 + 1/x^2 = 11 := 
by
  sorry

end algebraic_expression_l2221_222129


namespace average_percent_score_l2221_222151

theorem average_percent_score :
    let students := 120
    let score_95 := 95 * 12
    let score_85 := 85 * 24
    let score_75 := 75 * 30
    let score_65 := 65 * 20
    let score_55 := 55 * 18
    let score_45 := 45 * 10
    let score_35 := 35 * 6
    let total_score := score_95 + score_85 + score_75 + score_65 + score_55 + score_45 + score_35
    (total_score.toFloat / students.toFloat) = 69.8333 :=
by
  sorry

end average_percent_score_l2221_222151


namespace average_episodes_per_year_l2221_222147

def num_years : Nat := 14
def seasons_15_episodes : Nat := 8
def episodes_per_season_15 : Nat := 15
def seasons_20_episodes : Nat := 4
def episodes_per_season_20 : Nat := 20
def seasons_12_episodes : Nat := 2
def episodes_per_season_12 : Nat := 12

theorem average_episodes_per_year :
  (seasons_15_episodes * episodes_per_season_15 +
   seasons_20_episodes * episodes_per_season_20 +
   seasons_12_episodes * episodes_per_season_12) / num_years = 16 := by
  sorry

end average_episodes_per_year_l2221_222147


namespace find_a1_l2221_222133

theorem find_a1 (a_1 : ℕ) (S : ℕ → ℕ) (S_formula : ∀ n : ℕ, S n = (a_1 * (3^n - 1)) / 2)
  (a_4_eq : (S 4) - (S 3) = 54) : a_1 = 2 :=
  sorry

end find_a1_l2221_222133


namespace Alfonso_daily_earnings_l2221_222103

-- Define the conditions given in the problem
def helmet_cost : ℕ := 340
def current_savings : ℕ := 40
def days_per_week : ℕ := 5
def weeks_to_work : ℕ := 10

-- Define the question as a property to prove
def daily_earnings : ℕ := 6

-- Prove that the daily earnings are $6 given the conditions
theorem Alfonso_daily_earnings :
  (helmet_cost - current_savings) / (days_per_week * weeks_to_work) = daily_earnings :=
by
  sorry

end Alfonso_daily_earnings_l2221_222103


namespace bsnt_value_l2221_222109

theorem bsnt_value (B S N T : ℝ) (hB : 0 < B) (hS : 0 < S) (hN : 0 < N) (hT : 0 < T)
    (h1 : Real.log (B * S) / Real.log 10 + Real.log (B * N) / Real.log 10 = 3)
    (h2 : Real.log (N * T) / Real.log 10 + Real.log (N * S) / Real.log 10 = 4)
    (h3 : Real.log (S * T) / Real.log 10 + Real.log (S * B) / Real.log 10 = 5) :
    B * S * N * T = 10000 :=
sorry

end bsnt_value_l2221_222109


namespace suff_not_nec_l2221_222178

theorem suff_not_nec (x : ℝ) : (x > 0 → x ≠ 0) ∧ (x ≠ 0 → ¬(x ≤ 0)) :=
by
  sorry

end suff_not_nec_l2221_222178


namespace initial_people_per_column_l2221_222136

theorem initial_people_per_column (P x : ℕ) (h1 : P = 16 * x) (h2 : P = 48 * 10) : x = 30 :=
by 
  sorry

end initial_people_per_column_l2221_222136


namespace angle_B_in_geometric_progression_l2221_222144

theorem angle_B_in_geometric_progression 
  {A B C a b c : ℝ} 
  (hSum : A + B + C = Real.pi)
  (hGeo : A = B / 2)
  (hGeo2 : C = 2 * B)
  (hSide : b^2 - a^2 = a * c)
  : B = 2 * Real.pi / 7 := 
by
  sorry

end angle_B_in_geometric_progression_l2221_222144


namespace inscribed_circle_radius_l2221_222169

variable (A p s r : ℝ)

-- Condition: Area is twice the perimeter
def twice_perimeter_condition : Prop := A = 2 * p

-- Condition: The formula connecting the area, inradius, and semiperimeter
def area_inradius_semiperimeter_relation : Prop := A = r * s

-- Condition: The perimeter is twice the semiperimeter
def perimeter_semiperimeter_relation : Prop := p = 2 * s

-- Prove the radius of the inscribed circle is 4
theorem inscribed_circle_radius (h1 : twice_perimeter_condition A p)
                                (h2 : area_inradius_semiperimeter_relation A r s)
                                (h3 : perimeter_semiperimeter_relation p s) :
  r = 4 :=
by
  sorry

end inscribed_circle_radius_l2221_222169


namespace summation_eq_16_implies_x_eq_3_over_4_l2221_222112

theorem summation_eq_16_implies_x_eq_3_over_4 (x : ℝ) (h : ∑' n : ℕ, (n + 1) * x^n = 16) : x = 3 / 4 :=
sorry

end summation_eq_16_implies_x_eq_3_over_4_l2221_222112


namespace max_notebooks_lucy_can_buy_l2221_222124

-- Definitions given in the conditions
def lucyMoney : ℕ := 2145
def notebookCost : ℕ := 230

-- Theorem to prove the number of notebooks Lucy can buy
theorem max_notebooks_lucy_can_buy : lucyMoney / notebookCost = 9 := 
by
  sorry

end max_notebooks_lucy_can_buy_l2221_222124


namespace no_possible_salary_distribution_l2221_222108

theorem no_possible_salary_distribution (x y z : ℕ) (h1 : x + y + z = 13) (h2 : x + 3 * y + 5 * z = 200) : false :=
by {
  -- Proof goes here
  sorry
}

end no_possible_salary_distribution_l2221_222108


namespace tan_double_angle_l2221_222132

theorem tan_double_angle (x : ℝ) (h : (Real.sqrt 3) * Real.cos x - Real.sin x = 0) : Real.tan (2 * x) = - (Real.sqrt 3) :=
by
  sorry

end tan_double_angle_l2221_222132


namespace verify_extending_points_l2221_222115

noncomputable def verify_P_and_Q (A B P Q : ℝ → ℝ → ℝ) : Prop := 
  let vector_relation_P := P = - (2/5) • A + (7/5) • B
  let vector_relation_Q := Q = - (1/4) • A + (5/4) • B 
  vector_relation_P ∧ vector_relation_Q

theorem verify_extending_points 
  (A B P Q : ℝ → ℝ → ℝ)
  (h1 : 7 • (P - A) = 2 • (B - P))
  (h2 : 5 • (Q - A) = 1 • (Q - B)) :
  verify_P_and_Q A B P Q := 
by
  sorry  

end verify_extending_points_l2221_222115


namespace num_perfect_square_factors_of_360_l2221_222152

theorem num_perfect_square_factors_of_360 : 
  ∃ (n : ℕ), n = 4 ∧ ∀ d : ℕ, d ∣ 360 → (∀ p e, p^e ∣ d → (p = 2 ∨ p = 3 ∨ p = 5) ∧ e % 2 = 0) :=
by
  sorry

end num_perfect_square_factors_of_360_l2221_222152


namespace towels_folded_in_one_hour_l2221_222107

theorem towels_folded_in_one_hour :
  let jane_rate := 12 * 5 -- Jane's rate in towels/hour
  let kyla_rate := 6 * 9  -- Kyla's rate in towels/hour
  let anthony_rate := 3 * 14 -- Anthony's rate in towels/hour
  let david_rate := 4 * 6 -- David's rate in towels/hour
  jane_rate + kyla_rate + anthony_rate + david_rate = 180 := 
by
  let jane_rate := 12 * 5
  let kyla_rate := 6 * 9
  let anthony_rate := 3 * 14
  let david_rate := 4 * 6
  show jane_rate + kyla_rate + anthony_rate + david_rate = 180
  sorry

end towels_folded_in_one_hour_l2221_222107


namespace graph_passes_through_quadrants_l2221_222117

-- Definitions based on the conditions
def linear_function (x : ℝ) : ℝ := -2 * x + 1

-- The property to be proven
theorem graph_passes_through_quadrants :
  (∃ x > 0, linear_function x > 0) ∧  -- Quadrant I
  (∃ x < 0, linear_function x > 0) ∧  -- Quadrant II
  (∃ x > 0, linear_function x < 0) := -- Quadrant IV
sorry

end graph_passes_through_quadrants_l2221_222117


namespace initially_had_8_l2221_222150

-- Define the number of puppies given away
def given_away : ℕ := 4

-- Define the number of puppies still with Sandy
def still_has : ℕ := 4

-- Define the total number of puppies initially
def initially_had (x y : ℕ) : ℕ := x + y

-- Prove that the number of puppies Sandy's dog had initially equals 8
theorem initially_had_8 : initially_had given_away still_has = 8 :=
by sorry

end initially_had_8_l2221_222150


namespace quadrilateral_is_rhombus_l2221_222118

theorem quadrilateral_is_rhombus (a b c d : ℝ)
  (h : a^2 + b^2 + c^2 + d^2 = ab + bc + cd + ad) : a = b ∧ b = c ∧ c = d :=
by
  sorry

end quadrilateral_is_rhombus_l2221_222118


namespace concert_tickets_l2221_222121

theorem concert_tickets (A C : ℕ) (h1 : C = 3 * A) (h2 : 7 * A + 3 * C = 6000) : A + C = 1500 :=
by {
  -- Proof omitted
  sorry
}

end concert_tickets_l2221_222121
