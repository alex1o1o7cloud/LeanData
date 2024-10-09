import Mathlib

namespace total_travel_time_l2289_228910

-- Defining the conditions
def car_travel_180_miles_in_4_hours : Prop :=
  180 / 4 = 45

def car_travel_135_miles_additional_time : Prop :=
  135 / 45 = 3

-- The main statement to be proved
theorem total_travel_time : car_travel_180_miles_in_4_hours ∧ car_travel_135_miles_additional_time → 4 + 3 = 7 := by
  sorry

end total_travel_time_l2289_228910


namespace A_and_B_together_finish_in_ten_days_l2289_228919

-- Definitions of conditions
def B_daily_work := 1 / 15
def A_daily_work := B_daily_work / 2
def combined_daily_work := A_daily_work + B_daily_work

-- The theorem to be proved
theorem A_and_B_together_finish_in_ten_days : 1 / combined_daily_work = 10 := 
  by 
    sorry

end A_and_B_together_finish_in_ten_days_l2289_228919


namespace inverse_proportion_points_l2289_228966

theorem inverse_proportion_points (x1 x2 x3 : ℝ) :
  (10 / x1 = -5) →
  (10 / x2 = 2) →
  (10 / x3 = 5) →
  x1 < x3 ∧ x3 < x2 :=
by sorry

end inverse_proportion_points_l2289_228966


namespace intersection_M_N_l2289_228978

def M : Set ℝ := { x | -4 < x ∧ x < 2 }

def N : Set ℝ := { x | x^2 - x - 6 < 0 }

theorem intersection_M_N : M ∩ N = { x | -2 < x ∧ x < 2 } := by
  sorry

end intersection_M_N_l2289_228978


namespace area_enclosed_curve_l2289_228969

-- The proof statement
theorem area_enclosed_curve (x y : ℝ) : (x^2 + y^2 = 2 * (|x| + |y|)) → 
  (area_of_enclosed_region = 2 * π + 8) :=
sorry

end area_enclosed_curve_l2289_228969


namespace arithmetic_sequence_a_find_p_q_find_c_minus_a_find_y_values_l2289_228995

-- Problem (a)
theorem arithmetic_sequence_a (x1 x2 x3 x4 x5 : ℕ) (h : (x1 = 2 ∧ x2 = 5 ∧ x3 = 10 ∧ x4 = 13 ∧ x5 = 15)) : 
  ∃ a b c, (a = 5 ∧ b = 10 ∧ c = 15 ∧ b - a = c - b ∧ b - a > 0) := 
sorry

-- Problem (b)
theorem find_p_q (p q : ℕ) (h : ∃ d, (7 - p = d ∧ q - 7 = d ∧ 13 - q = d)) : 
  p = 4 ∧ q = 10 :=
sorry

-- Problem (c)
theorem find_c_minus_a (a b c : ℕ) (h : ∃ d, (b - a = d ∧ c - b = d ∧ (a + 21) - c = d)) :
  c - a = 14 :=
sorry

-- Problem (d)
theorem find_y_values (y : ℤ) (h : ∃ d, ((2*y + 3) - (y - 6) = d ∧ (y*y + 2) - (2*y + 3) = d) ) :
  y = 5 ∨ y = -2 :=
sorry

end arithmetic_sequence_a_find_p_q_find_c_minus_a_find_y_values_l2289_228995


namespace line_product_l2289_228999

theorem line_product (b m : ℝ) (h1: b = -1) (h2: m = 2) : m * b = -2 :=
by
  rw [h1, h2]
  norm_num


end line_product_l2289_228999


namespace first_player_winning_strategy_l2289_228952

-- Defining the type for the positions on the chessboard
structure Position where
  x : Nat
  y : Nat
  deriving DecidableEq

-- Initial position C1
def C1 : Position := ⟨3, 1⟩

-- Winning position H8
def H8 : Position := ⟨8, 8⟩

-- Function to check if a position is a winning position
-- the target winning position is H8
def isWinningPosition (p : Position) : Bool :=
  p = H8

-- Function to determine the next possible positions
-- from the current position based on the allowed moves
def nextPositions (p : Position) : List Position :=
  (List.range (8 - p.x)).map (λ dx => ⟨p.x + dx + 1, p.y⟩) ++
  (List.range (8 - p.y)).map (λ dy => ⟨p.x, p.y + dy + 1⟩) ++
  (List.range (min (8 - p.x) (8 - p.y))).map (λ d => ⟨p.x + d + 1, p.y + d + 1⟩)

-- Statement of the problem: First player has a winning strategy from C1
theorem first_player_winning_strategy : 
  ∃ move : Position, move ∈ nextPositions C1 ∧
  ∀ next_move : Position, next_move ∈ nextPositions move → isWinningPosition next_move :=
sorry

end first_player_winning_strategy_l2289_228952


namespace train_passenger_count_l2289_228900

theorem train_passenger_count (P : ℕ) (total_passengers : ℕ) (r : ℕ)
  (h1 : r = 60)
  (h2 : total_passengers = P + r + 3 * (P + r))
  (h3 : total_passengers = 640) :
  P = 100 :=
by
  sorry

end train_passenger_count_l2289_228900


namespace regular_hexagon_interior_angle_measure_l2289_228944

theorem regular_hexagon_interior_angle_measure :
  let n := 6
  let sum_of_angles := (n - 2) * 180
  let measure_of_each_angle := sum_of_angles / n
  measure_of_each_angle = 120 :=
by
  sorry

end regular_hexagon_interior_angle_measure_l2289_228944


namespace largest_integer_le_zero_of_f_l2289_228905

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 6

theorem largest_integer_le_zero_of_f :
  ∃ x₀ : ℝ, (f x₀ = 0) ∧ 2 ≤ x₀ ∧ x₀ < 3 ∧ (∀ k : ℤ, k ≤ x₀ → k = 2 ∨ k < 2) :=
by
  sorry

end largest_integer_le_zero_of_f_l2289_228905


namespace female_lion_weight_l2289_228954

theorem female_lion_weight (male_weight : ℚ) (weight_difference : ℚ) (female_weight : ℚ) : 
  male_weight = 145/4 → 
  weight_difference = 47/10 → 
  male_weight = female_weight + weight_difference → 
  female_weight = 631/20 :=
by
  intros h₁ h₂ h₃
  sorry

end female_lion_weight_l2289_228954


namespace students_in_section_B_l2289_228926

variable (x : ℕ)

/-- There are 30 students in section A and the number of students in section B is x. The 
    average weight of section A is 40 kg, and the average weight of section B is 35 kg. 
    The average weight of the whole class is 38 kg. Prove that the number of students in
    section B is 20. -/
theorem students_in_section_B (h : 30 * 40 + x * 35 = 38 * (30 + x)) : x = 20 :=
  sorry

end students_in_section_B_l2289_228926


namespace exists_infinitely_many_n_odd_floor_l2289_228901

def even (n : ℤ) := ∃ k : ℤ, n = 2 * k
def odd (n : ℤ) := ∃ k : ℤ, n = 2 * k + 1

theorem exists_infinitely_many_n_odd_floor (α : ℝ) : 
  ∃ᶠ n in at_top, odd ⌊n^2 * α⌋ := sorry

end exists_infinitely_many_n_odd_floor_l2289_228901


namespace JordanRectangleWidth_l2289_228936

/-- Given that Carol's rectangle measures 15 inches by 24 inches,
and Jordan's rectangle is 8 inches long with equal area as Carol's rectangle,
prove that Jordan's rectangle is 45 inches wide. -/
theorem JordanRectangleWidth :
  ∃ W : ℝ, (15 * 24 = 8 * W) → W = 45 := by
  sorry

end JordanRectangleWidth_l2289_228936


namespace combined_selling_price_l2289_228990

theorem combined_selling_price 
  (cost_price1 cost_price2 cost_price3 : ℚ)
  (profit_percentage1 profit_percentage2 profit_percentage3 : ℚ)
  (h1 : cost_price1 = 1200) (h2 : profit_percentage1 = 0.4)
  (h3 : cost_price2 = 800)  (h4 : profit_percentage2 = 0.3)
  (h5 : cost_price3 = 600)  (h6 : profit_percentage3 = 0.5) : 
  cost_price1 * (1 + profit_percentage1) +
  cost_price2 * (1 + profit_percentage2) +
  cost_price3 * (1 + profit_percentage3) = 3620 := by 
  sorry

end combined_selling_price_l2289_228990


namespace inverse_89_mod_90_l2289_228973

theorem inverse_89_mod_90 : (89 * 89) % 90 = 1 := 
by
  -- Mathematical proof is skipped
  sorry

end inverse_89_mod_90_l2289_228973


namespace rachel_assembly_time_l2289_228906

theorem rachel_assembly_time :
  let chairs := 20
  let tables := 8
  let bookshelves := 5
  let time_per_chair := 6
  let time_per_table := 8
  let time_per_bookshelf := 12
  let total_chairs_time := chairs * time_per_chair
  let total_tables_time := tables * time_per_table
  let total_bookshelves_time := bookshelves * time_per_bookshelf
  total_chairs_time + total_tables_time + total_bookshelves_time = 244 := by
  sorry

end rachel_assembly_time_l2289_228906


namespace product_of_two_digit_numbers_5488_has_smaller_number_56_l2289_228942

theorem product_of_two_digit_numbers_5488_has_smaller_number_56 (a b : ℕ) (h_a2 : 10 ≤ a) (h_a3 : a < 100) (h_b2 : 10 ≤ b) (h_b3 : b < 100) (h_prod : a * b = 5488) : a = 56 ∨ b = 56 :=
by {
  sorry
}

end product_of_two_digit_numbers_5488_has_smaller_number_56_l2289_228942


namespace train_cross_pole_time_l2289_228986

variable (L : Real) (V : Real)

theorem train_cross_pole_time (hL : L = 110) (hV : V = 144) : 
  (110 / (144 * 1000 / 3600) = 2.75) := 
by
  sorry

end train_cross_pole_time_l2289_228986


namespace sum_of_interior_angles_octagon_l2289_228948

theorem sum_of_interior_angles_octagon : (8 - 2) * 180 = 1080 :=
by
  sorry

end sum_of_interior_angles_octagon_l2289_228948


namespace triangle_two_acute_angles_l2289_228970

theorem triangle_two_acute_angles (A B C : ℝ) (h_triangle : A + B + C = 180) (h_pos : A > 0 ∧ B > 0 ∧ C > 0)
  (h_acute_triangle: A < 90 ∨ B < 90 ∨ C < 90): A < 90 ∧ B < 90 ∨ A < 90 ∧ C < 90 ∨ B < 90 ∧ C < 90 :=
by
  sorry

end triangle_two_acute_angles_l2289_228970


namespace train_cross_bridge_in_56_seconds_l2289_228991

noncomputable def train_pass_time (length_train length_bridge : ℝ) (speed_train_kmh : ℝ) : ℝ :=
  let total_distance := length_train + length_bridge
  let speed_train_ms := speed_train_kmh * (1000 / 3600)
  total_distance / speed_train_ms

theorem train_cross_bridge_in_56_seconds :
  train_pass_time 560 140 45 = 56 :=
by
  -- The proof can be added here
  sorry

end train_cross_bridge_in_56_seconds_l2289_228991


namespace sufficient_and_necessary_condition_l2289_228917

theorem sufficient_and_necessary_condition (x : ℝ) :
  (x - 2) * (x + 2) > 0 ↔ x > 2 ∨ x < -2 :=
by sorry

end sufficient_and_necessary_condition_l2289_228917


namespace solve_for_y_l2289_228914

theorem solve_for_y : ∀ (y : ℚ), 2 * y + 3 * y = 500 - (4 * y + 6 * y) → y = 100 / 3 := by
  intros y h
  sorry

end solve_for_y_l2289_228914


namespace unique_function_property_l2289_228945

def f (n : Nat) : Nat := sorry

theorem unique_function_property :
  (∀ x y : ℕ+, x < y → f x < f y) ∧
  (∀ y x : ℕ+, f (y * f x) = x^2 * f (x * y)) →
  ∀ n : ℕ+, f n = n^2 :=
by
  intros h
  sorry

end unique_function_property_l2289_228945


namespace find_expression_value_l2289_228953

-- We declare our variables x and y
variables (x y : ℝ)

-- We state our conditions as hypotheses
def h1 : 3 * x + y = 5 := sorry
def h2 : x + 3 * y = 8 := sorry

-- We prove the given mathematical expression
theorem find_expression_value (h1 : 3 * x + y = 5) (h2 : x + 3 * y = 8) : 10 * x^2 + 19 * x * y + 10 * y^2 = 153 := 
by
  -- We intentionally skip the proof
  sorry

end find_expression_value_l2289_228953


namespace product_of_four_consecutive_integers_divisible_by_12_l2289_228957

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l2289_228957


namespace solve_inequalities_l2289_228984

theorem solve_inequalities (x : ℝ) :
  (2 * x + 1 < 3) ∧ ((x / 2) + ((1 - 3 * x) / 4) ≤ 1) → -3 ≤ x ∧ x < 1 := 
by
  sorry

end solve_inequalities_l2289_228984


namespace f_monotonic_non_overlapping_domains_domain_of_sum_l2289_228977

axiom f : ℝ → ℝ
axiom f_decreasing : ∀ x₁ x₂ : ℝ, -1 ≤ x₁ → x₁ ≤ 1 → -1 ≤ x₂ → x₂ ≤ 1 → x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) < 0

theorem f_monotonic : ∀ x₁ x₂ : ℝ, -1 ≤ x₁ → x₁ ≤ 1 → -1 ≤ x₂ → x₂ ≤ 1 → x₁ ≤ x₂ → f x₁ ≥ f x₂ := sorry

theorem non_overlapping_domains : ∀ c : ℝ, (c - 1 > c^2 + 1 → c > 2) ∧ (c^2 - 1 > c + 1 → c < -1) := sorry

theorem domain_of_sum : 
  ∀ c : ℝ,
  -1 ≤ c ∧ c ≤ 2 →
  (∃ a b : ℝ, 
    ((-1 ≤ c ∧ c ≤ 0) ∨ (1 ≤ c ∧ c ≤ 2) → a = c^2 - 1 ∧ b = c + 1) ∧ 
    (0 < c ∧ c < 1 → a = c - 1 ∧ b = c^2 + 1)
  ) := sorry

end f_monotonic_non_overlapping_domains_domain_of_sum_l2289_228977


namespace cubic_roots_geometric_progression_l2289_228961

theorem cubic_roots_geometric_progression 
  (a r : ℝ)
  (h_roots: 27 * a^3 * r^3 - 81 * a^2 * r^2 + 63 * a * r - 14 = 0)
  (h_sum: a + a * r + a * r^2 = 3)
  (h_product: a^3 * r^3 = 14 / 27)
  : (max (a^2) ((a * r^2)^2) - min (a^2) ((a * r^2)^2) = 5 / 3) := 
sorry

end cubic_roots_geometric_progression_l2289_228961


namespace greatest_number_of_consecutive_integers_whose_sum_is_36_l2289_228923

/-- 
Given that the sum of N consecutive integers starting from a is 36, 
prove that the greatest possible value of N is 72.
-/
theorem greatest_number_of_consecutive_integers_whose_sum_is_36 :
  ∀ (N a : ℤ), (N > 0) → (N * (2 * a + N - 1)) = 72 → N ≤ 72 := 
by
  intros N a hN h
  sorry

end greatest_number_of_consecutive_integers_whose_sum_is_36_l2289_228923


namespace games_attended_l2289_228946

theorem games_attended (games_this_month games_last_month games_next_month total_games : ℕ) 
  (h1 : games_this_month = 11) 
  (h2 : games_last_month = 17) 
  (h3 : games_next_month = 16) : 
  total_games = games_this_month + games_last_month + games_next_month → 
  total_games = 44 :=
by
  sorry

end games_attended_l2289_228946


namespace find_positive_integers_l2289_228903

noncomputable def is_power_of_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2 ^ k

theorem find_positive_integers (a b c : ℕ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_c : 0 < c)
  (hab_c : is_power_of_two (a * b - c))
  (hbc_a : is_power_of_two (b * c - a))
  (hca_b : is_power_of_two (c * a - b)) :
  (a = 2 ∧ b = 2 ∧ c = 2) ∨
  (a = 2 ∧ b = 2 ∧ c = 3) ∨
  (a = 3 ∧ b = 5 ∧ c = 7) ∨
  (a = 2 ∧ b = 6 ∧ c = 11) :=
sorry

end find_positive_integers_l2289_228903


namespace find_valid_pairs_l2289_228912

noncomputable def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ d : ℕ, d ∣ n → d = 1 ∨ d = n)

def distinct_two_digit_primes : List (ℕ × ℕ) :=
  [(13, 53), (19, 47), (23, 43), (29, 37)]

def average (p q : ℕ) : ℕ := (p + q) / 2

def number1 (p q : ℕ) : ℕ := 100 * p + q
def number2 (p q : ℕ) : ℕ := 100 * q + p

theorem find_valid_pairs (p q : ℕ)
  (hp : is_prime p) (hq : is_prime q)
  (hpq : p ≠ q)
  (havg : average p q ∣ number1 p q ∧ average p q ∣ number2 p q) :
  (p, q) ∈ distinct_two_digit_primes ∨ (q, p) ∈ distinct_two_digit_primes :=
sorry

end find_valid_pairs_l2289_228912


namespace action_figure_total_l2289_228927

variable (initial_figures : ℕ) (added_figures : ℕ)

theorem action_figure_total (h₁ : initial_figures = 8) (h₂ : added_figures = 2) : (initial_figures + added_figures) = 10 := by
  sorry

end action_figure_total_l2289_228927


namespace intersection_A_B_l2289_228933

-- Define the sets A and B
def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {x | x > 0}

-- The theorem we want to prove
theorem intersection_A_B : A ∩ B = {1} := 
by {
  sorry
}

end intersection_A_B_l2289_228933


namespace arithmetic_sequence_solution_l2289_228907

variable {a : ℕ → ℤ}  -- assuming our sequence is integer-valued for simplicity

-- a is an arithmetic sequence if there exists a common difference d such that 
-- ∀ n, a_{n+1} = a_n + d
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

-- sum of the terms from a₁ to a₁₀₁₇ is equal to zero
def sum_condition (a : ℕ → ℤ) : Prop :=
  (Finset.range 2017).sum a = 0

theorem arithmetic_sequence_solution (a : ℕ → ℤ) (h_arith : is_arithmetic_sequence a) (h_sum : sum_condition a) :
  a 3 + a 2013 = 0 :=
sorry

end arithmetic_sequence_solution_l2289_228907


namespace neg_univ_prop_l2289_228993

-- Translate the original mathematical statement to a Lean 4 statement.
theorem neg_univ_prop :
  (¬(∀ x : ℝ, x^2 ≠ x)) ↔ (∃ x : ℝ, x^2 = x) :=
by
  sorry

end neg_univ_prop_l2289_228993


namespace problem_statement_l2289_228904

variable {F : Type*} [Field F]

theorem problem_statement (m : F) (h : m + 1 / m = 6) : m^2 + 1 / m^2 + 4 = 38 :=
by
  sorry

end problem_statement_l2289_228904


namespace jungkook_has_smallest_collection_l2289_228974

-- Define the collections
def yoongi_collection : ℕ := 7
def jungkook_collection : ℕ := 6
def yuna_collection : ℕ := 9

-- State the theorem
theorem jungkook_has_smallest_collection : 
  jungkook_collection = min yoongi_collection (min jungkook_collection yuna_collection) := 
by
  sorry

end jungkook_has_smallest_collection_l2289_228974


namespace find_width_l2289_228922

variable (L W : ℕ)

def perimeter (L W : ℕ) : ℕ := 2 * L + 2 * W

theorem find_width (h1 : perimeter L W = 46) (h2 : W = L + 7) : W = 15 :=
sorry

end find_width_l2289_228922


namespace centroids_coincide_l2289_228921

noncomputable def centroid (A B C : ℂ) : ℂ :=
  (A + B + C) / 3

theorem centroids_coincide (A B C : ℂ) (k : ℝ) (C1 A1 B1 : ℂ)
  (h1 : C1 = k * (B - A) + A)
  (h2 : A1 = k * (C - B) + B)
  (h3 : B1 = k * (A - C) + C) :
  centroid A1 B1 C1 = centroid A B C := by
  sorry

end centroids_coincide_l2289_228921


namespace fraction_simplification_l2289_228964

theorem fraction_simplification :
  8 * (15 / 11) * (-25 / 40) = -15 / 11 :=
by
  sorry

end fraction_simplification_l2289_228964


namespace total_cost_of_backpack_and_pencil_case_l2289_228935

-- Definitions based on the given conditions
def pencil_case_price : ℕ := 8
def backpack_price : ℕ := 5 * pencil_case_price

-- Statement of the proof problem
theorem total_cost_of_backpack_and_pencil_case : 
  pencil_case_price + backpack_price = 48 :=
by
  -- Skip the proof
  sorry

end total_cost_of_backpack_and_pencil_case_l2289_228935


namespace largest_root_of_equation_l2289_228941

theorem largest_root_of_equation : ∃ (x : ℝ), (x - 37)^2 - 169 = 0 ∧ ∀ y, (y - 37)^2 - 169 = 0 → y ≤ x :=
by
  sorry

end largest_root_of_equation_l2289_228941


namespace minimum_value_quadratic_l2289_228958

noncomputable def quadratic (x : ℝ) : ℝ := x^2 - 4 * x + 5

theorem minimum_value_quadratic :
  ∀ x : ℝ, quadratic x ≥ 1 :=
by
  sorry

end minimum_value_quadratic_l2289_228958


namespace odd_base_divisibility_by_2_base_divisibility_by_m_l2289_228960

-- Part (a)
theorem odd_base_divisibility_by_2 (q : ℕ) :
  (∀ a : ℕ, (a * q) % 2 = 0 ↔ a % 2 = 0) → q % 2 = 1 := 
sorry

-- Part (b)
theorem base_divisibility_by_m (q m : ℕ) (h1 : m > 1) :
  (∀ a : ℕ, (a * q) % m = 0 ↔ a % m = 0) → ∃ k : ℕ, q = 1 + m * k ∧ k ≥ 1 :=
sorry

end odd_base_divisibility_by_2_base_divisibility_by_m_l2289_228960


namespace beginning_of_spring_period_and_day_l2289_228931

noncomputable def daysBetween : Nat := 46 -- Total days: Dec 21, 2004 to Feb 4, 2005

theorem beginning_of_spring_period_and_day :
  let total_days := daysBetween
  let segment := total_days / 9
  let day_within_segment := total_days % 9
  segment = 5 ∧ day_within_segment = 1 := by
sorry

end beginning_of_spring_period_and_day_l2289_228931


namespace apples_picked_l2289_228928

theorem apples_picked (n_a : ℕ) (k_a : ℕ) (total : ℕ) (m_a : ℕ) (h_n : n_a = 3) (h_k : k_a = 6) (h_t : total = 16) :
  m_a = total - (n_a + k_a) →
  m_a = 7 :=
by
  sorry

end apples_picked_l2289_228928


namespace nine_x_five_y_multiple_l2289_228997

theorem nine_x_five_y_multiple (x y : ℤ) (h : 2 * x + 3 * y ≡ 0 [ZMOD 17]) : 
  9 * x + 5 * y ≡ 0 [ZMOD 17] := 
by
  sorry

end nine_x_five_y_multiple_l2289_228997


namespace room_length_l2289_228988

theorem room_length (width : ℝ) (total_cost : ℝ) (cost_per_sqm : ℝ) (length : ℝ)
  (h_width : width = 3.75)
  (h_total_cost : total_cost = 28875)
  (h_cost_per_sqm : cost_per_sqm = 1400)
  (h_length : length = total_cost / cost_per_sqm / width) :
  length = 5.5 := by
  sorry

end room_length_l2289_228988


namespace find_number_l2289_228924

theorem find_number {x : ℤ} (h : x + 5 = 6) : x = 1 :=
sorry

end find_number_l2289_228924


namespace find_x_of_perpendicular_l2289_228963

-- Definitions based on the conditions in a)
def a (x : ℝ) : ℝ × ℝ := (x, x + 1)
def b : ℝ × ℝ := (1, 2)

-- The mathematical proof problem in Lean 4 statement: prove that the dot product is zero implies x = -2/3
theorem find_x_of_perpendicular (x : ℝ) (h : (a x).fst * b.fst + (a x).snd * b.snd = 0) : x = -2 / 3 := 
by
  sorry

end find_x_of_perpendicular_l2289_228963


namespace remaining_area_is_344_l2289_228956

def garden_length : ℕ := 20
def garden_width : ℕ := 18
def shed_side : ℕ := 4

def area_rectangle : ℕ := garden_length * garden_width
def area_shed : ℕ := shed_side * shed_side

def remaining_garden_area : ℕ := area_rectangle - area_shed

theorem remaining_area_is_344 : remaining_garden_area = 344 := by
  sorry

end remaining_area_is_344_l2289_228956


namespace tank_capacity_l2289_228943

theorem tank_capacity (x : ℝ) (h : 0.50 * x = 75) : x = 150 :=
by sorry

end tank_capacity_l2289_228943


namespace polygon_sides_l2289_228971

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 720) : n = 6 :=
by sorry

end polygon_sides_l2289_228971


namespace dice_sum_is_4_l2289_228965

-- Defining the sum of points obtained from two dice rolls
def sum_of_dice (a b : ℕ) : ℕ := a + b

-- The main theorem stating the condition we need to prove
theorem dice_sum_is_4 (a b : ℕ) (h : sum_of_dice a b = 4) :
  (a = 3 ∧ b = 1) ∨ (a = 1 ∧ b = 3) ∨ (a = 2 ∧ b = 2) :=
sorry

end dice_sum_is_4_l2289_228965


namespace pyramid_on_pentagonal_prism_l2289_228998

-- Define the structure of a pentagonal prism
structure PentagonalPrism where
  faces : ℕ
  vertices : ℕ
  edges : ℕ

-- Initial pentagonal prism properties
def initialPrism : PentagonalPrism := {
  faces := 7,
  vertices := 10,
  edges := 15
}

-- Assume we add a pyramid on top of one pentagonal face
def addPyramid (prism : PentagonalPrism) : PentagonalPrism := {
  faces := prism.faces - 1 + 5, -- 1 face covered, 5 new faces
  vertices := prism.vertices + 1, -- 1 new vertex
  edges := prism.edges + 5 -- 5 new edges
}

-- The resulting shape after adding the pyramid
def resultingShape : PentagonalPrism := addPyramid initialPrism

-- Calculating the sum of faces, vertices, and edges
def sumFacesVerticesEdges (shape : PentagonalPrism) : ℕ :=
  shape.faces + shape.vertices + shape.edges

-- Statement of the problem in Lean 4
theorem pyramid_on_pentagonal_prism : sumFacesVerticesEdges resultingShape = 42 := by
  sorry

end pyramid_on_pentagonal_prism_l2289_228998


namespace no_natural_number_exists_l2289_228994

theorem no_natural_number_exists 
  (n : ℕ) : ¬ ∃ x y : ℕ, (2 * n * (n + 1) * (n + 2) * (n + 3) + 12) = x^2 + y^2 := 
by sorry

end no_natural_number_exists_l2289_228994


namespace sum_of_two_rationals_negative_l2289_228985

theorem sum_of_two_rationals_negative (a b : ℚ) (h : a + b < 0) : a < 0 ∨ b < 0 := sorry

end sum_of_two_rationals_negative_l2289_228985


namespace f_odd_f_periodic_f_def_on_interval_problem_solution_l2289_228967

noncomputable def f : ℝ → ℝ := 
sorry

theorem f_odd (x : ℝ) : f (-x) = -f x := 
sorry

theorem f_periodic (x : ℝ) : f (x + 4) = f x := 
sorry

theorem f_def_on_interval (x : ℝ) (h : -2 < x ∧ x < 0) : f x = 2 ^ x :=
sorry

theorem problem_solution : f 2015 - f 2014 = 1 / 2 :=
sorry

end f_odd_f_periodic_f_def_on_interval_problem_solution_l2289_228967


namespace james_planted_60_percent_l2289_228902

theorem james_planted_60_percent :
  let total_trees := 2
  let plants_per_tree := 20
  let seeds_per_plant := 1
  let total_seeds := total_trees * plants_per_tree * seeds_per_plant
  let planted_trees := 24
  (planted_trees / total_seeds) * 100 = 60 := 
by
  sorry

end james_planted_60_percent_l2289_228902


namespace sets_equal_l2289_228930

theorem sets_equal :
  let M := {x | x^2 - 2 * x + 1 = 0}
  let N := {1}
  M = N :=
by
  sorry

end sets_equal_l2289_228930


namespace duration_of_each_movie_l2289_228913

-- define the conditions
def num_screens : ℕ := 6
def hours_open : ℕ := 8
def num_movies : ℕ := 24

-- define the total screening time
def total_screening_time : ℕ := num_screens * hours_open

-- define the expected duration of each movie
def movie_duration : ℕ := total_screening_time / num_movies

-- state the theorem
theorem duration_of_each_movie : movie_duration = 2 := by sorry

end duration_of_each_movie_l2289_228913


namespace ellipsoid_volume_div_pi_l2289_228929

noncomputable def ellipsoid_projection_min_area : ℝ := 9 * Real.pi
noncomputable def ellipsoid_projection_max_area : ℝ := 25 * Real.pi
noncomputable def ellipsoid_circle_projection_area : ℝ := 16 * Real.pi
noncomputable def ellipsoid_volume (a b c : ℝ) : ℝ := (4/3) * Real.pi * a * b * c

theorem ellipsoid_volume_div_pi (a b c : ℝ)
  (h_min : (a * b = 9))
  (h_max : (b * c = 25))
  (h_circle : (b = 4)) :
  ellipsoid_volume a b c / Real.pi = 75 := 
  by
    sorry

end ellipsoid_volume_div_pi_l2289_228929


namespace total_emails_in_april_is_675_l2289_228983

-- Define the conditions
def daily_emails : ℕ := 20
def additional_emails : ℕ := 5
def april_days : ℕ := 30
def half_april_days : ℕ := april_days / 2

-- Define the total number of emails received
def total_emails : ℕ :=
  (daily_emails * half_april_days) +
  ((daily_emails + additional_emails) * half_april_days)

-- Define the statement to be proven
theorem total_emails_in_april_is_675 : total_emails = 675 :=
  by
  sorry

end total_emails_in_april_is_675_l2289_228983


namespace hens_count_l2289_228982

theorem hens_count (H C : ℕ) (h1 : H + C = 46) (h2 : 2 * H + 4 * C = 140) : H = 22 :=
by
  sorry

end hens_count_l2289_228982


namespace madeline_biked_more_l2289_228920

def madeline_speed : ℕ := 12
def madeline_time : ℕ := 3
def max_speed : ℕ := 15
def max_time : ℕ := 2

theorem madeline_biked_more : (madeline_speed * madeline_time) - (max_speed * max_time) = 6 := 
by 
  sorry

end madeline_biked_more_l2289_228920


namespace simplify_powers_l2289_228908

-- Defining the multiplicative rule for powers
def power_mul (x : ℕ) (a b : ℕ) : ℕ := x^(a+b)

-- Proving that x^5 * x^6 = x^11
theorem simplify_powers (x : ℕ) : x^5 * x^6 = x^11 :=
by
  change x^5 * x^6 = x^(5 + 6)
  sorry

end simplify_powers_l2289_228908


namespace find_m_l2289_228951

def vector_parallel (a b : ℝ × ℝ) : Prop := 
  ∃ k : ℝ, b.1 = k * a.1 ∧ b.2 = k * a.2

theorem find_m
  (m : ℝ)
  (a : ℝ × ℝ := (m, 1))
  (b : ℝ × ℝ := (2, -1))
  (h : vector_parallel a (b.1 - a.1, b.2 - a.2)) :
  m = -2 :=
by
  sorry

end find_m_l2289_228951


namespace remainder_n_sq_plus_3n_5_mod_25_l2289_228932

theorem remainder_n_sq_plus_3n_5_mod_25 (k : ℤ) (n : ℤ) (h : n = 25 * k - 1) : 
  (n^2 + 3 * n + 5) % 25 = 3 := 
by
  sorry

end remainder_n_sq_plus_3n_5_mod_25_l2289_228932


namespace infinite_primes_dividing_expression_l2289_228955

theorem infinite_primes_dividing_expression (k : ℕ) (hk : k > 0) : 
  ∃ᶠ p in Filter.atTop, ∃ n : ℕ, p ∣ (2017^n + k) :=
sorry

end infinite_primes_dividing_expression_l2289_228955


namespace cost_of_candy_l2289_228981

theorem cost_of_candy (initial_amount remaining_amount : ℕ) (h_init : initial_amount = 4) (h_remaining : remaining_amount = 3) : initial_amount - remaining_amount = 1 :=
by
  sorry

end cost_of_candy_l2289_228981


namespace arithmetic_sequence_a4_l2289_228916

theorem arithmetic_sequence_a4 (a1 : ℤ) (S3 : ℤ) (h1 : a1 = 3) (h2 : S3 = 15) : 
  ∃ (a4 : ℤ), a4 = 9 :=
by
  sorry

end arithmetic_sequence_a4_l2289_228916


namespace no_adjacent_same_roll_probability_l2289_228950

noncomputable def probability_no_adjacent_same_roll : ℚ :=
  (1331 / 1728)

theorem no_adjacent_same_roll_probability :
  (probability_no_adjacent_same_roll = (1331 / 1728)) :=
by
  sorry

end no_adjacent_same_roll_probability_l2289_228950


namespace max_non_overlapping_squares_l2289_228937

theorem max_non_overlapping_squares (m n : ℕ) : 
  ∃ max_squares : ℕ, max_squares = m :=
by
  sorry

end max_non_overlapping_squares_l2289_228937


namespace simplify_and_evaluate_evaluate_at_zero_l2289_228949

noncomputable def simplified_expression (x : ℤ) : ℚ :=
  (1 - 1/(x-1)) / ((x^2 - 4*x + 4) / (x^2 - 1))

theorem simplify_and_evaluate (x : ℤ) (h : x ≠ 1 ∧ x ≠ 2 ∧ x ≠ -1) : 
  simplified_expression x = (x+1)/(x-2) :=
by
  sorry

theorem evaluate_at_zero : simplified_expression 0 = -1/2 :=
by
  sorry

end simplify_and_evaluate_evaluate_at_zero_l2289_228949


namespace calculate_expression_l2289_228918

variable (f g : ℝ → ℝ)

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)
def is_odd_function (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g x = - g (-x)

theorem calculate_expression 
  (hf : is_even_function f)
  (hg : is_odd_function g)
  (hfg : ∀ x : ℝ, f x - g x = x ^ 3 + x ^ 2 + 1) :
  f 1 + g 1 = 1 :=
  sorry

end calculate_expression_l2289_228918


namespace solve_r_l2289_228989

-- Definitions related to the problem
def satisfies_equation (r : ℝ) : Prop := ⌊r⌋ + 2 * r = 16

-- Theorem statement
theorem solve_r : ∃ (r : ℝ), satisfies_equation r ∧ r = 5.5 :=
by
  sorry

end solve_r_l2289_228989


namespace find_a_l2289_228909

theorem find_a (a b c : ℂ) (ha : a.im = 0)
  (h1 : a + b + c = 5)
  (h2 : a * b + b * c + c * a = 8)
  (h3 : a * b * c = 4) :
  a = 1 ∨ a = 2 :=
sorry

end find_a_l2289_228909


namespace triangle_condition_l2289_228940

noncomputable def f (x : ℝ) : ℝ :=
  (Real.sin x) * (Real.cos x) + (Real.sqrt 3) * (Real.cos x) ^ 2 - (Real.sqrt 3) / 2

theorem triangle_condition (a b c : ℝ) (h : b^2 + c^2 = a^2 + Real.sqrt 3 * b * c) : 
  f (Real.pi / 6) = Real.sqrt 3 / 2 := by
  sorry

end triangle_condition_l2289_228940


namespace Robert_has_taken_more_photos_l2289_228980

variables (C L R : ℕ) -- Claire's, Lisa's, and Robert's photos

-- Conditions definitions:
def ClairePhotos : Prop := C = 8
def LisaPhotos : Prop := L = 3 * C
def RobertPhotos : Prop := R > C

-- The proof problem statement:
theorem Robert_has_taken_more_photos (h1 : ClairePhotos C) (h2 : LisaPhotos C L) : RobertPhotos C R :=
by { sorry }

end Robert_has_taken_more_photos_l2289_228980


namespace horse_problem_l2289_228911

-- Definitions based on conditions:
def total_horses : ℕ := 100
def tiles_pulled_by_big_horse (x : ℕ) : ℕ := 3 * x
def tiles_pulled_by_small_horses (x : ℕ) : ℕ := (100 - x) / 3

-- The statement to prove:
theorem horse_problem (x : ℕ) : 
    tiles_pulled_by_big_horse x + tiles_pulled_by_small_horses x = 100 :=
sorry

end horse_problem_l2289_228911


namespace doubling_n_constant_C_l2289_228996

theorem doubling_n_constant_C (e n R r : ℝ) (h_pos_e : 0 < e) (h_pos_n : 0 < n) (h_pos_R : 0 < R) (h_pos_r : 0 < r)
  (C : ℝ) (hC : C = e^2 * n / (R + n * r^2)) :
  C = (2 * e^2 * n) / (R + 2 * n * r^2) := 
sorry

end doubling_n_constant_C_l2289_228996


namespace avg_of_7_consecutive_integers_l2289_228975

theorem avg_of_7_consecutive_integers (c d : ℝ) (h : d = (c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5)) / 6) :
  ((d + (d + 1) + (d + 2) + (d + 3) + (d + 4) + (d + 5) + (d + 6)) / 7) = c + 5.5 := by
  sorry

end avg_of_7_consecutive_integers_l2289_228975


namespace students_selected_milk_l2289_228968

noncomputable def selected_soda_percent : ℚ := 50 / 100
noncomputable def selected_milk_percent : ℚ := 30 / 100
noncomputable def selected_soda_count : ℕ := 90
noncomputable def selected_milk_count := selected_milk_percent / selected_soda_percent * selected_soda_count

theorem students_selected_milk :
    selected_milk_count = 54 :=
by
  sorry

end students_selected_milk_l2289_228968


namespace numeric_value_of_BAR_l2289_228979

variable (b a t c r : ℕ)

-- Conditions from the problem
axiom h1 : b + a + t = 6
axiom h2 : c + a + t = 8
axiom h3 : c + a + r = 12

-- Required to prove
theorem numeric_value_of_BAR : b + a + r = 10 :=
by
  -- Proof goes here
  sorry

end numeric_value_of_BAR_l2289_228979


namespace gcd_three_numbers_l2289_228934

theorem gcd_three_numbers (a b c : ℕ) (h1 : a = 72) (h2 : b = 120) (h3 : c = 168) :
  Nat.gcd (Nat.gcd a b) c = 24 :=
by
  rw [h1, h2, h3]
  exact sorry

end gcd_three_numbers_l2289_228934


namespace count_dracula_is_alive_l2289_228938

variable (P Q : Prop)
variable (h1 : P)          -- I am human
variable (h2 : P → Q)      -- If I am human, then Count Dracula is alive

theorem count_dracula_is_alive : Q :=
by
  sorry

end count_dracula_is_alive_l2289_228938


namespace asymptotes_of_hyperbola_l2289_228959

theorem asymptotes_of_hyperbola : 
  ∀ (x y : ℝ), 9 * y^2 - 25 * x^2 = 169 → (y = (5/3) * x ∨ y = -(5/3) * x) :=
by 
  sorry

end asymptotes_of_hyperbola_l2289_228959


namespace cooking_time_remaining_l2289_228925

def time_to_cook_remaining (n_total n_cooked t_per : ℕ) : ℕ := (n_total - n_cooked) * t_per

theorem cooking_time_remaining :
  ∀ (n_total n_cooked t_per : ℕ), n_total = 13 → n_cooked = 5 → t_per = 6 → time_to_cook_remaining n_total n_cooked t_per = 48 :=
by
  intros n_total n_cooked t_per h1 h2 h3
  rw [h1, h2, h3]
  exact rfl

end cooking_time_remaining_l2289_228925


namespace quadratic_real_roots_l2289_228976

theorem quadratic_real_roots (K : ℝ) :
  ∃ x : ℝ, K^2 * x^2 + (K^2 - 1) * x - 2 * K^2 = 0 :=
sorry

end quadratic_real_roots_l2289_228976


namespace travel_ways_l2289_228972

theorem travel_ways (buses : Nat) (trains : Nat) (boats : Nat) 
  (hb : buses = 5) (ht : trains = 6) (hb2 : boats = 2) : 
  buses + trains + boats = 13 := by
  sorry

end travel_ways_l2289_228972


namespace bathroom_area_l2289_228915

-- Definitions based on conditions
def totalHouseArea : ℝ := 1110
def numBedrooms : ℕ := 4
def bedroomArea : ℝ := 11 * 11
def kitchenArea : ℝ := 265
def numBathrooms : ℕ := 2

-- Mathematically equivalent proof problem
theorem bathroom_area :
  let livingArea := kitchenArea  -- living area is equal to kitchen area
  let totalRoomArea := numBedrooms * bedroomArea + kitchenArea + livingArea
  let remainingArea := totalHouseArea - totalRoomArea
  let bathroomArea := remainingArea / numBathrooms
  bathroomArea = 48 :=
by
  repeat { sorry }

end bathroom_area_l2289_228915


namespace smallest_positive_x_l2289_228939

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

theorem smallest_positive_x : ∃ x : ℕ, x > 0 ∧ is_palindrome (x + 6789) ∧ x = 218 := by
  sorry

end smallest_positive_x_l2289_228939


namespace karen_bonus_problem_l2289_228947

theorem karen_bonus_problem (n already_graded last_two target : ℕ) (h_already_graded : already_graded = 8)
  (h_last_two : last_two = 290) (h_target : target = 600) (max_score : ℕ)
  (h_max_score : max_score = 150) (required_avg : ℕ) (h_required_avg : required_avg = 75) :
  ∃ A : ℕ, (A = 70) ∧ (target = 600) ∧ (last_two = 290) ∧ (already_graded = 8) ∧
  (required_avg = 75) := by
  sorry

end karen_bonus_problem_l2289_228947


namespace find_f_zero_l2289_228962

variable (f : ℝ → ℝ)

def odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x, g (x + 1) = -g (-x + 1)

def even_function (g : ℝ → ℝ) : Prop :=
  ∀ x, g (x - 1) = g (-x - 1)

theorem find_f_zero
  (H1 : odd_function f)
  (H2 : even_function f)
  (H3 : f 4 = 6) :
  f 0 = -6 := by
  sorry

end find_f_zero_l2289_228962


namespace population_percentage_l2289_228992

theorem population_percentage (total_population : ℕ) (percentage : ℕ) (result : ℕ) :
  total_population = 25600 → percentage = 90 → result = (percentage * total_population) / 100 → result = 23040 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end population_percentage_l2289_228992


namespace selling_price_correct_l2289_228987

-- Define the conditions
def purchase_price : ℝ := 12000
def repair_costs : ℝ := 5000
def transportation_charges : ℝ := 1000
def profit_percentage : ℝ := 0.50

-- Calculate total cost
def total_cost : ℝ := purchase_price + repair_costs + transportation_charges

-- Define the selling price and the proof goal
def selling_price : ℝ := total_cost + (profit_percentage * total_cost)

-- Prove that the selling price equals Rs 27000
theorem selling_price_correct : selling_price = 27000 := 
by 
  -- Proof is not required, so we use sorry
  sorry

end selling_price_correct_l2289_228987
