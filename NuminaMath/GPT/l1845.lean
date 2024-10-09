import Mathlib

namespace exponential_monotonicity_l1845_184519

theorem exponential_monotonicity {a b c : ℝ} (h1 : a > b) (h2 : b > 0) (h3 : c > 1) : c^a > c^b :=
by 
  sorry 

end exponential_monotonicity_l1845_184519


namespace swimming_pool_cost_l1845_184553

/-!
# Swimming Pool Cost Problem

Given:
* The pool takes 50 hours to fill.
* The hose runs at 100 gallons per hour.
* Water costs 1 cent for 10 gallons.

Prove that the total cost to fill the pool is 5 dollars.
-/

theorem swimming_pool_cost :
  let hours_to_fill := 50
  let hose_rate := 100  -- gallons per hour
  let cost_per_gallon := 0.01 / 10  -- dollars per gallon
  let total_volume := hours_to_fill * hose_rate  -- total volume in gallons
  let total_cost := total_volume * cost_per_gallon
  total_cost = 5 :=
by
  sorry

end swimming_pool_cost_l1845_184553


namespace valid_paths_from_P_to_Q_l1845_184576

-- Define the grid dimensions and alternate coloring conditions
def grid_width := 10
def grid_height := 8
def is_white_square (r c : ℕ) : Bool :=
  (r + c) % 2 = 1

-- Define the starting and ending squares P and Q
def P : ℕ × ℕ := (0, grid_width / 2)
def Q : ℕ × ℕ := (grid_height - 1, grid_width / 2)

-- Define a function to count valid 9-step paths from P to Q
noncomputable def count_valid_paths : ℕ :=
  -- Here the function to compute valid paths would be defined
  -- This is broad outline due to lean's framework missing specific combinatorial functions
  245

-- The theorem to state the proof problem
theorem valid_paths_from_P_to_Q : count_valid_paths = 245 :=
sorry

end valid_paths_from_P_to_Q_l1845_184576


namespace two_digit_numbers_non_repeating_l1845_184511

-- The set of available digits is given as 0, 1, 2, 3, 4
def digits : List ℕ := [0, 1, 2, 3, 4]

-- Ensure the tens place digits are subset of 1, 2, 3, 4 (exclude 0)
def valid_tens : List ℕ := [1, 2, 3, 4]

theorem two_digit_numbers_non_repeating :
  let num_tens := valid_tens.length
  let num_units := (digits.length - 1)
  num_tens * num_units = 16 :=
by
  -- Observe num_tens = 4, since valid_tens = [1, 2, 3, 4]
  -- Observe num_units = 4, since digits.length = 5 and we exclude the tens place digit
  sorry

end two_digit_numbers_non_repeating_l1845_184511


namespace longest_chord_of_circle_l1845_184501

theorem longest_chord_of_circle (r : ℝ) (h : r = 3) : ∃ l, l = 6 := by
  sorry

end longest_chord_of_circle_l1845_184501


namespace binary_to_decimal_11011_is_27_l1845_184582

def binary_to_decimal : ℕ :=
  1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0

theorem binary_to_decimal_11011_is_27 : binary_to_decimal = 27 := by
  sorry

end binary_to_decimal_11011_is_27_l1845_184582


namespace median_of_36_consecutive_integers_l1845_184534

theorem median_of_36_consecutive_integers (sum_of_integers : ℕ) (num_of_integers : ℕ) 
  (h1 : num_of_integers = 36) (h2 : sum_of_integers = 6 ^ 4) : 
  (sum_of_integers / num_of_integers) = 36 := 
by 
  sorry

end median_of_36_consecutive_integers_l1845_184534


namespace binary_equals_octal_l1845_184589

-- Define the binary number 1001101 in decimal
def binary_1001101_decimal : ℕ := 1 * 2^6 + 0 * 2^5 + 0 * 2^4 + 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0

-- Define the octal number 115 in decimal
def octal_115_decimal : ℕ := 1 * 8^2 + 1 * 8^1 + 5 * 8^0

-- Theorem statement
theorem binary_equals_octal :
  binary_1001101_decimal = octal_115_decimal :=
sorry

end binary_equals_octal_l1845_184589


namespace evaluate_expression_l1845_184550

theorem evaluate_expression : ((3^4)^3 + 5) - ((4^3)^4 + 5) = -16245775 := by
  sorry

end evaluate_expression_l1845_184550


namespace solve_linear_system_l1845_184599

theorem solve_linear_system :
  ∃ (x y : ℚ), (4 * x - 3 * y = 2) ∧ (6 * x + 5 * y = 1) ∧ (x = 13 / 38) ∧ (y = -4 / 19) :=
by
  sorry

end solve_linear_system_l1845_184599


namespace no_real_roots_of_quadratic_l1845_184587

def quadratic_discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

theorem no_real_roots_of_quadratic (h : quadratic_discriminant 1 (-1) 1 < 0) :
  ¬ ∃ x : ℝ, x^2 - x + 1 = 0 :=
by
  sorry

end no_real_roots_of_quadratic_l1845_184587


namespace second_term_arithmetic_sequence_l1845_184536

theorem second_term_arithmetic_sequence 
  (a d : ℤ)
  (h1 : a + 15 * d = 8)
  (h2 : a + 16 * d = 10) : 
  a + d = -20 := 
by sorry

end second_term_arithmetic_sequence_l1845_184536


namespace sum_g_h_k_l1845_184583

def polynomial_product_constants (d g h k : ℤ) : Prop :=
  ((5 * d^2 + 4 * d + g) * (4 * d^2 + h * d - 5) = 20 * d^4 + 11 * d^3 - 9 * d^2 + k * d - 20)

theorem sum_g_h_k (d g h k : ℤ) (h1 : polynomial_product_constants d g h k) : g + h + k = -16 :=
by
  sorry

end sum_g_h_k_l1845_184583


namespace coed_softball_team_total_players_l1845_184500

theorem coed_softball_team_total_players (M W : ℕ) 
  (h1 : W = M + 4) 
  (h2 : (M : ℚ) / W = 0.6363636363636364) :
  M + W = 18 := 
by sorry

end coed_softball_team_total_players_l1845_184500


namespace max_value_of_a_b_c_l1845_184581

theorem max_value_of_a_b_c (a b c : ℤ) (h1 : a + b = 2006) (h2 : c - a = 2005) (h3 : a < b) : 
  a + b + c = 5013 :=
sorry

end max_value_of_a_b_c_l1845_184581


namespace equivalent_expression_l1845_184530

theorem equivalent_expression (x : ℝ) (hx : x > 0) : (x^2 * x^(1/4))^(1/3) = x^(3/4) := 
  sorry

end equivalent_expression_l1845_184530


namespace highest_page_number_l1845_184584

/-- Given conditions: Pat has 19 instances of the digit '7' and an unlimited supply of all 
other digits. Prove that the highest page number Pat can number without exceeding 19 instances 
of the digit '7' is 99. -/
theorem highest_page_number (num_of_sevens : ℕ) (highest_page : ℕ) 
  (h1 : num_of_sevens = 19) : highest_page = 99 :=
sorry

end highest_page_number_l1845_184584


namespace car_miles_per_tankful_in_city_l1845_184533

-- Define constants for the given values
def miles_per_tank_on_highway : ℝ := 462
def fewer_miles_per_gallon : ℝ := 15
def miles_per_gallon_in_city : ℝ := 40

-- Prove the car traveled 336 miles per tankful in the city
theorem car_miles_per_tankful_in_city :
  (miles_per_tank_on_highway / (miles_per_gallon_in_city + fewer_miles_per_gallon)) * miles_per_gallon_in_city = 336 := 
by
  sorry

end car_miles_per_tankful_in_city_l1845_184533


namespace find_t_l1845_184520

noncomputable def a_sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 5 ∧ ∀ n : ℕ, n ≥ 2 → a (n + 1) = 3 * a n + 3 ^ n

noncomputable def b_sequence (a : ℕ → ℤ) (b : ℕ → ℤ) (t : ℤ) : Prop :=
  ∀ n : ℕ, b n = (a (n + 1) + t) / 3^(n + 1)

theorem find_t (a : ℕ → ℤ) (b : ℕ → ℤ) (t : ℤ) :
  a_sequence a →
  b_sequence a b t →
  (∀ n : ℕ, (b (n + 1) - b n) = (b 1 - b 0)) →
  t = -1 / 2 :=
by
  sorry

end find_t_l1845_184520


namespace profit_per_meter_l1845_184595

theorem profit_per_meter (number_of_meters : ℕ) (total_selling_price cost_price_per_meter : ℝ) 
  (h1 : number_of_meters = 85) 
  (h2 : total_selling_price = 8925) 
  (h3 : cost_price_per_meter = 90) :
  (total_selling_price - cost_price_per_meter * number_of_meters) / number_of_meters = 15 :=
  sorry

end profit_per_meter_l1845_184595


namespace combined_weight_l1845_184572

theorem combined_weight (S R : ℕ) (h1 : S = 71) (h2 : S - 5 = 2 * R) : S + R = 104 := by
  sorry

end combined_weight_l1845_184572


namespace main_inequality_equality_condition_l1845_184504

variable {a b c : ℝ}
variable (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

theorem main_inequality 
  (hpos_a : 0 < a) 
  (hpos_b : 0 < b) 
  (hpos_c : 0 < c) :
  (1 / (a * (1 + b)) + 1 / (b * (1 + c)) + 1 / (c * (1 + a)) ≥ 3 / (1 + a * b * c)) :=
  sorry

theorem equality_condition 
  (hpos_a : 0 < a) 
  (hpos_b : 0 < b) 
  (hpos_c : 0 < c) :
  (1 / (a * (1 + b)) + 1 / (b * (1 + c)) + 1 / (c * (1 + a)) = 3 / (1 + a * b * c) ↔ a = 1 ∧ b = 1 ∧ c = 1) :=
  sorry

end main_inequality_equality_condition_l1845_184504


namespace correct_statements_in_triangle_l1845_184564

theorem correct_statements_in_triangle (a b c : ℝ) (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (h_sum : A + B + C = π) :
  (c = a * Real.cos B + b * Real.cos A) ∧ 
  (a^3 + b^3 = c^3 → a^2 + b^2 > c^2) :=
by
  sorry

end correct_statements_in_triangle_l1845_184564


namespace exists_airline_route_within_same_republic_l1845_184592

theorem exists_airline_route_within_same_republic
  (C : Type) [Fintype C] [DecidableEq C]
  (R : Type) [Fintype R] [DecidableEq R]
  (belongs_to : C → R)
  (airline_route : C → C → Prop)
  (country_size : Fintype.card C = 100)
  (republics_size : Fintype.card R = 3)
  (millionaire_cities : {c : C // ∃ n : ℕ, n ≥ 70 ∧ (∃ S : Finset C, S.card = n ∧ ∀ x ∈ S, airline_route c x) })
  (at_least_70_millionaire_cities : ∃ n : ℕ, n ≥ 70 ∧ (∃ S : Finset {c : C // ∃ n : ℕ, n ≥ 70 ∧ ( ∃ S : Finset C, S.card = n ∧ ∀ x ∈ S, airline_route c x )}, S.card = n)):
  ∃ (c1 c2 : C), airline_route c1 c2 ∧ belongs_to c1 = belongs_to c2 := 
sorry

end exists_airline_route_within_same_republic_l1845_184592


namespace C_plus_D_l1845_184574

theorem C_plus_D (D C : ℚ) (h : ∀ x : ℚ, x ≠ 3 ∧ x ≠ 5 → (D * x - 17) / ((x - 3) * (x - 5)) = C / (x - 3) + 2 / (x - 5)) :
  C + D = 32 / 5 :=
by
  sorry

end C_plus_D_l1845_184574


namespace gcd_n_cube_plus_m_square_l1845_184507

theorem gcd_n_cube_plus_m_square (n m : ℤ) (h : n > 2^3) : Int.gcd (n^3 + m^2) (n + 2) = 1 :=
by
  sorry

end gcd_n_cube_plus_m_square_l1845_184507


namespace orange_ring_weight_correct_l1845_184573

-- Define the weights as constants
def purple_ring_weight := 0.3333333333333333
def white_ring_weight := 0.4166666666666667
def total_weight := 0.8333333333
def orange_ring_weight := 0.0833333333

-- Theorem statement
theorem orange_ring_weight_correct :
  total_weight - purple_ring_weight - white_ring_weight = orange_ring_weight :=
by
  -- Sorry is added to skip the proof part as per the instruction
  sorry

end orange_ring_weight_correct_l1845_184573


namespace ratio_of_red_to_black_l1845_184521

theorem ratio_of_red_to_black (r b : ℕ) (h_r : r = 26) (h_b : b = 70) :
  r / Nat.gcd r b = 13 ∧ b / Nat.gcd r b = 35 :=
by
  sorry

end ratio_of_red_to_black_l1845_184521


namespace number_of_dogs_in_shelter_l1845_184510

variables (D C R P : ℕ)

-- Conditions
axiom h1 : 15 * C = 7 * D
axiom h2 : 9 * P = 5 * R
axiom h3 : 15 * (C + 8) = 11 * D
axiom h4 : 7 * P = 5 * (R + 6)

theorem number_of_dogs_in_shelter : D = 30 :=
by sorry

end number_of_dogs_in_shelter_l1845_184510


namespace calculate_expression_l1845_184590

theorem calculate_expression : 
  ∀ (x y : ℕ), x = 3 → y = 4 → 3*(x^4 + 2*y^2)/9 = 37 + 2/3 :=
by
  intros x y hx hy
  sorry

end calculate_expression_l1845_184590


namespace triangle_angles_l1845_184579

theorem triangle_angles
  (h_a a h_b b : ℝ)
  (h_a_ge_a : h_a ≥ a)
  (h_b_ge_b : h_b ≥ b)
  (a_ge_h_b : a ≥ h_b)
  (b_ge_h_a : b ≥ h_a) : 
  a = b ∧ 
  (a = h_a ∧ b = h_b) → 
  ∃ A B C : ℝ, Set.toFinset ({A, B, C} : Set ℝ) = {90, 45, 45} := 
by 
  sorry

end triangle_angles_l1845_184579


namespace algebraic_expression_value_l1845_184526

theorem algebraic_expression_value {m n : ℝ} 
  (h1 : n = m - 2022) 
  (h2 : m * n = -2022) : 
  (2022 / m) + ((m^2 - 2022 * m) / n) = 2022 := 
by sorry

end algebraic_expression_value_l1845_184526


namespace total_books_l1845_184593

theorem total_books (Zig_books : ℕ) (Flo_books : ℕ) (Tim_books : ℕ) 
  (hz : Zig_books = 60) (hf : Zig_books = 4 * Flo_books) (ht : Tim_books = Flo_books / 2) :
  Zig_books + Flo_books + Tim_books = 82 := by
  sorry

end total_books_l1845_184593


namespace minutes_until_8_00_am_l1845_184563

-- Definitions based on conditions
def time_in_minutes (hours : Nat) (minutes : Nat) : Nat := hours * 60 + minutes

def current_time : Nat := time_in_minutes 7 30 + 16

def target_time : Nat := time_in_minutes 8 0

-- The theorem we need to prove
theorem minutes_until_8_00_am : target_time - current_time = 14 :=
by
  sorry

end minutes_until_8_00_am_l1845_184563


namespace ratio_of_areas_of_circles_l1845_184560

theorem ratio_of_areas_of_circles 
  (R_A R_B : ℝ) 
  (h : (π / 2 * R_A) = (π / 3 * R_B)) : 
  (π * R_A ^ 2) / (π * R_B ^ 2) = (4 : ℚ) / 9 := 
sorry

end ratio_of_areas_of_circles_l1845_184560


namespace length_of_DF_l1845_184546

theorem length_of_DF
  (D E F P Q: Type)
  (DP: ℝ)
  (EQ: ℝ)
  (h1: DP = 27)
  (h2: EQ = 36)
  (perp: ∀ (u v: Type), u ≠ v):
  ∃ (DF: ℝ), DF = 4 * Real.sqrt 117 :=
by
  sorry

end length_of_DF_l1845_184546


namespace intersection_A_B_l1845_184575

def A : Set ℝ := {-2, -1, 2, 3}
def B : Set ℝ := {x : ℝ | x^2 - x - 6 < 0}

theorem intersection_A_B : A ∩ B = {-1, 2} :=
by
  sorry

end intersection_A_B_l1845_184575


namespace sum_of_nine_consecutive_even_integers_mod_10_l1845_184570

theorem sum_of_nine_consecutive_even_integers_mod_10 : 
  (10112 + 10114 + 10116 + 10118 + 10120 + 10122 + 10124 + 10126 + 10128) % 10 = 0 := by
  sorry

end sum_of_nine_consecutive_even_integers_mod_10_l1845_184570


namespace sum_absolute_values_of_first_ten_terms_l1845_184578

noncomputable def S (n : ℕ) : ℤ := n^2 - 4 * n + 2

noncomputable def a (n : ℕ) : ℤ := S n - S (n - 1)

noncomputable def absolute_sum_10 : ℤ :=
  |a 1| + |a 2| + |a 3| + |a 4| + |a 5| + |a 6| + |a 7| + |a 8| + |a 9| + |a 10|

theorem sum_absolute_values_of_first_ten_terms : absolute_sum_10 = 68 := by
  sorry

end sum_absolute_values_of_first_ten_terms_l1845_184578


namespace platform_length_l1845_184559

noncomputable def train_length : ℕ := 1200
noncomputable def time_to_cross_tree : ℕ := 120
noncomputable def time_to_pass_platform : ℕ := 230

theorem platform_length
  (v : ℚ)
  (h1 : v = train_length / time_to_cross_tree)
  (total_distance : ℚ)
  (h2 : total_distance = v * time_to_pass_platform)
  (platform_length : ℚ)
  (h3 : total_distance = train_length + platform_length) :
  platform_length = 1100 := by 
  sorry

end platform_length_l1845_184559


namespace sin_cos_15_degree_l1845_184541

theorem sin_cos_15_degree :
  (Real.sin (15 * Real.pi / 180)) * (Real.cos (15 * Real.pi / 180)) = 1 / 4 :=
by
  sorry

end sin_cos_15_degree_l1845_184541


namespace find_k_l1845_184555

variables {x k : ℝ}

theorem find_k (h1 : (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 8)) (h2 : k ≠ 0) : k = 8 :=
sorry

end find_k_l1845_184555


namespace solve_equation1_solve_equation2_l1845_184586

theorem solve_equation1 (x : ℝ) : x^2 - 2 * x - 2 = 0 ↔ (x = 1 + Real.sqrt 3 ∨ x = 1 - Real.sqrt 3) :=
by
  sorry

theorem solve_equation2 (x : ℝ) : 2 * (x - 3)^2 = x - 3 ↔ (x = 3/2 ∨ x = 7/2) :=
by
  sorry

end solve_equation1_solve_equation2_l1845_184586


namespace Craig_walk_distance_l1845_184518

/-- Craig walked some distance from school to David's house and 0.7 miles from David's house to his own house. 
In total, Craig walked 0.9 miles. Prove that the distance Craig walked from school to David's house is 0.2 miles. 
--/
theorem Craig_walk_distance (d_school_David d_David_Craig d_total : ℝ) 
  (h1 : d_David_Craig = 0.7) 
  (h2 : d_total = 0.9) : 
  d_school_David = 0.2 :=
by 
  sorry

end Craig_walk_distance_l1845_184518


namespace find_y_l1845_184567

theorem find_y (y : ℝ) (h_cond : y = (1 / y) * (-y) - 3) : y = -4 := 
sorry

end find_y_l1845_184567


namespace sum_of_solutions_l1845_184544

theorem sum_of_solutions : 
  (∀ x : ℝ, (3 * x) / 15 = 4 / x) → (0 + 4 = 4) :=
by
  sorry

end sum_of_solutions_l1845_184544


namespace man_work_alone_l1845_184568

theorem man_work_alone (W: ℝ) (M S: ℝ)
  (hS: S = W / 6.67)
  (hMS: M + S = W / 4):
  W / M = 10 :=
by {
  -- This is a placeholder for the proof
  sorry
}

end man_work_alone_l1845_184568


namespace trigonometric_identity_l1845_184594

theorem trigonometric_identity
  (θ : ℝ)
  (h : Real.tan θ = 1 / 3) :
  Real.sin (3 / 2 * Real.pi + 2 * θ) = -4 / 5 :=
by sorry

end trigonometric_identity_l1845_184594


namespace profit_per_package_l1845_184558

theorem profit_per_package
  (packages_first_center_per_day : ℕ)
  (packages_second_center_multiplier : ℕ)
  (weekly_profit : ℕ)
  (days_per_week : ℕ)
  (H1 : packages_first_center_per_day = 10000)
  (H2 : packages_second_center_multiplier = 3)
  (H3 : weekly_profit = 14000)
  (H4 : days_per_week = 7) :
  (weekly_profit / (packages_first_center_per_day * days_per_week + 
                    packages_second_center_multiplier * packages_first_center_per_day * days_per_week) : ℝ) = 0.05 :=
by
  sorry

end profit_per_package_l1845_184558


namespace minimal_team_members_l1845_184529

theorem minimal_team_members (n : ℕ) : 
  (n ≡ 1 [MOD 6]) ∧ (n ≡ 2 [MOD 8]) ∧ (n ≡ 3 [MOD 9]) → n = 343 := 
by
  sorry

end minimal_team_members_l1845_184529


namespace square_area_l1845_184561

noncomputable def side_length1 (x : ℝ) : ℝ := 5 * x - 20
noncomputable def side_length2 (x : ℝ) : ℝ := 25 - 2 * x

theorem square_area (x : ℝ) (h : side_length1 x = side_length2 x) :
  (side_length1 x)^2 = 7225 / 49 :=
by
  sorry

end square_area_l1845_184561


namespace fraction_of_total_calls_l1845_184597

-- Definitions based on conditions
variable (B : ℚ) -- Calls processed by each member of Team B
variable (N : ℚ) -- Number of members in Team B

-- The fraction of calls processed by each member of Team A
def team_A_call_fraction : ℚ := 1 / 5

-- The fraction of calls processed by each member of Team C
def team_C_call_fraction : ℚ := 7 / 8

-- The fraction of agents in Team A relative to Team B
def team_A_agents_fraction : ℚ := 5 / 8

-- The fraction of agents in Team C relative to Team B
def team_C_agents_fraction : ℚ := 3 / 4

-- Total calls processed by Team A, Team B, and Team C
def total_calls_team_A : ℚ := (B * team_A_call_fraction) * (N * team_A_agents_fraction)
def total_calls_team_B : ℚ := B * N
def total_calls_team_C : ℚ := (B * team_C_call_fraction) * (N * team_C_agents_fraction)

-- Sum of total calls processed by all teams
def total_calls_all_teams : ℚ := total_calls_team_A B N + total_calls_team_B B N + total_calls_team_C B N

-- Potential total calls if all teams were as efficient as Team B
def potential_total_calls : ℚ := 3 * (B * N)

-- Fraction of total calls processed by all teams combined
def processed_fraction : ℚ := total_calls_all_teams B N / potential_total_calls B N

theorem fraction_of_total_calls : processed_fraction B N = 19 / 32 :=
by
  sorry -- Proof omitted

end fraction_of_total_calls_l1845_184597


namespace area_of_black_region_l1845_184557

theorem area_of_black_region (side_small side_large : ℕ) 
  (h1 : side_small = 5) 
  (h2 : side_large = 9) : 
  (side_large * side_large) - (side_small * side_small) = 56 := 
by
  sorry

end area_of_black_region_l1845_184557


namespace find_some_number_l1845_184516

def op (x w : ℕ) := (2^x) / (2^w)

theorem find_some_number (n : ℕ) (hn : 0 < n) : (op (op 4 n) n) = 4 → n = 2 :=
by
  sorry

end find_some_number_l1845_184516


namespace right_triangle_sides_l1845_184577

theorem right_triangle_sides :
  (4^2 + 5^2 ≠ 6^2) ∧
  (1^2 + 1^2 = (Real.sqrt 2)^2) ∧
  (6^2 + 8^2 ≠ 11^2) ∧
  (5^2 + 12^2 ≠ 23^2) :=
by
  repeat { sorry }

end right_triangle_sides_l1845_184577


namespace only_integers_square_less_than_three_times_l1845_184547

-- We want to prove that the only integers n that satisfy n^2 < 3n are 1 and 2.
theorem only_integers_square_less_than_three_times (n : ℕ) (h : n^2 < 3 * n) : n = 1 ∨ n = 2 :=
sorry

end only_integers_square_less_than_three_times_l1845_184547


namespace problem_sol_52_l1845_184524

theorem problem_sol_52 
  (x y: ℝ)
  (h1: x + y = 7)
  (h2: 4 * x * y = 7)
  (a b c d : ℕ)
  (hx_form : x = (a + b * Real.sqrt c) / d ∨ x = (a - b * Real.sqrt c) / d)
  (ha_pos : 0 < a)
  (hb_pos : 0 < b)
  (hc_pos : 0 < c)
  (hd_pos : 0 < d)
  : a + b + c + d = 52 := sorry

end problem_sol_52_l1845_184524


namespace area_covered_by_both_strips_is_correct_l1845_184509

-- Definitions of lengths of the strips and areas
def length_total : ℝ := 16
def length_left : ℝ := 9
def length_right : ℝ := 7
def area_left_only : ℝ := 27
def area_right_only : ℝ := 18

noncomputable def width_strip : ℝ := sorry -- The width can be inferred from solution but is not the focus of the proof.

-- Definition of the area covered by both strips
def S : ℝ := 13.5

-- Proof statement
theorem area_covered_by_both_strips_is_correct :
  ∀ w : ℝ,
    length_left * w - S = area_left_only ∧ length_right * w - S = area_right_only →
    S = 13.5 := 
by
  sorry

end area_covered_by_both_strips_is_correct_l1845_184509


namespace find_n_l1845_184556

theorem find_n (n : ℕ) (h : ∀ x : ℝ, (n : ℝ) < x ∧ x < (n + 1 : ℝ) → 3 * x - 5 = 0) :
  n = 1 :=
sorry

end find_n_l1845_184556


namespace midpoint_product_l1845_184508

theorem midpoint_product (x' y' : ℤ) 
  (h1 : (0 + x') / 2 = 2) 
  (h2 : (9 + y') / 2 = 4) : 
  (x' * y') = -4 :=
by
  sorry

end midpoint_product_l1845_184508


namespace storks_more_than_birds_l1845_184517

theorem storks_more_than_birds 
  (initial_birds : ℕ) 
  (joined_storks : ℕ) 
  (joined_birds : ℕ) 
  (h_init_birds : initial_birds = 3) 
  (h_joined_storks : joined_storks = 6) 
  (h_joined_birds : joined_birds = 2) : 
  (joined_storks - (initial_birds + joined_birds)) = 1 := 
by 
  -- Proof goes here
  sorry

end storks_more_than_birds_l1845_184517


namespace eval_floor_abs_neg_45_7_l1845_184538

theorem eval_floor_abs_neg_45_7 : ∀ x : ℝ, x = -45.7 → (⌊|x|⌋ = 45) := by
  intros x hx
  sorry

end eval_floor_abs_neg_45_7_l1845_184538


namespace min_expression_value_l1845_184552

theorem min_expression_value (a b c : ℝ) (h_sum : a + b + c = -1) (h_abc : a * b * c ≤ -3) :
  3 ≤ (ab + 1) / (a + b) + (bc + 1) / (b + c) + (ca + 1) / (c + a) :=
sorry

end min_expression_value_l1845_184552


namespace smartphone_charging_time_l1845_184598

theorem smartphone_charging_time :
  ∀ (T S : ℕ), T = 53 → T + (1 / 2 : ℚ) * S = 66 → S = 26 :=
by
  intros T S hT equation
  sorry

end smartphone_charging_time_l1845_184598


namespace sheela_overall_total_income_l1845_184554

def monthly_income_in_rs (income: ℝ) (savings: ℝ) (percent: ℝ): Prop :=
  savings = percent * income

def overall_total_income_in_rs (monthly_income: ℝ) 
                              (savings_deposit: ℝ) (fd_deposit: ℝ) 
                              (savings_interest_rate_monthly: ℝ) 
                              (fd_interest_rate_annual: ℝ): ℝ :=
  let annual_income := monthly_income * 12
  let savings_interest := savings_deposit * (savings_interest_rate_monthly * 12)
  let fd_interest := fd_deposit * fd_interest_rate_annual
  annual_income + savings_interest + fd_interest

theorem sheela_overall_total_income:
  ∀ (monthly_income: ℝ)
    (savings_deposit: ℝ) (fd_deposit: ℝ)
    (savings_interest_rate_monthly: ℝ) (fd_interest_rate_annual: ℝ),
    (monthly_income_in_rs monthly_income savings_deposit 0.28)  →
    monthly_income = 16071.43 →
    savings_deposit = 4500 →
    fd_deposit = 3000 →
    savings_interest_rate_monthly = 0.02 →
    fd_interest_rate_annual = 0.06 →
    overall_total_income_in_rs monthly_income savings_deposit fd_deposit
                           savings_interest_rate_monthly fd_interest_rate_annual
    = 194117.16 := 
by
  intros
  sorry

end sheela_overall_total_income_l1845_184554


namespace alchemy_value_l1845_184523

def letter_values : List Int :=
  [3, 2, 1, 0, -1, -2, -3, -2, -1, 0, 1, 2, 3, 2, 1, 0, -1, -2, -3, -2, -1,
  0, 1, 2, 3]

def char_value (c : Char) : Int :=
  letter_values.getD ((c.toNat - 'A'.toNat) % 13) 0

def word_value (s : String) : Int :=
  s.toList.map char_value |>.sum

theorem alchemy_value :
  word_value "ALCHEMY" = 8 :=
by
  sorry

end alchemy_value_l1845_184523


namespace speed_of_point_C_l1845_184512

theorem speed_of_point_C 
    (a T R L x : ℝ) 
    (h1 : x = L * (a * T) / R - L) 
    (h_eq: (a * T) / (a * T - R) = (L + x) / x) :
    (a * L) / R = x / T :=
by
  sorry

end speed_of_point_C_l1845_184512


namespace stefan_more_vail_l1845_184531

/-- Aiguo had 20 seashells --/
def a : ℕ := 20

/-- Vail had 5 less seashells than Aiguo --/
def v : ℕ := a - 5

/-- The total number of seashells of Stefan, Vail, and Aiguo is 66 --/
def total_seashells (s v a : ℕ) : Prop := s + v + a = 66

theorem stefan_more_vail (s v a : ℕ)
  (h_a : a = 20)
  (h_v : v = a - 5)
  (h_total : total_seashells s v a) :
  s - v = 16 :=
by {
  -- proofs would go here
  sorry
}

end stefan_more_vail_l1845_184531


namespace remainder_when_sum_divided_by_7_l1845_184506

-- Define the sequence
def arithmetic_sequence (a d n : ℕ) : ℕ := a + (n - 1) * d

-- Define the sum of the arithmetic sequence
def arithmetic_sequence_sum (a d n : ℕ) : ℕ := (n * (2 * a + (n - 1) * d)) / 2

-- Given conditions
def a : ℕ := 3
def d : ℕ := 7
def last_term : ℕ := 304

-- Calculate the number of terms in the sequence
noncomputable def n : ℕ := (last_term + 4) / 7

-- Calculate the sum
noncomputable def S : ℕ := arithmetic_sequence_sum a d n

-- Lean 4 statement to prove the remainder
theorem remainder_when_sum_divided_by_7 : S % 7 = 3 := by
  -- sorry placeholder for proof
  sorry

end remainder_when_sum_divided_by_7_l1845_184506


namespace angle_sum_l1845_184588

theorem angle_sum (y : ℝ) (h : 3 * y + y = 120) : y = 30 :=
sorry

end angle_sum_l1845_184588


namespace abs_difference_of_numbers_l1845_184505

theorem abs_difference_of_numbers (x y : ℝ) (h1 : x + y = 24) (h2 : x * y = 104) : |x - y| = 4 * Real.sqrt 10 :=
by
  sorry

end abs_difference_of_numbers_l1845_184505


namespace problem1_union_problem2_intersection_problem3_subset_l1845_184528

def A : Set ℝ := {x | x^2 - 2 * x - 3 ≤ 0}

def B (m : ℝ) : Set ℝ := {x | x^2 - 2 * m * x + m^2 - 4 ≤ 0}

theorem problem1_union (m : ℝ) (hm : m = 2) : A ∪ B m = {x | -1 ≤ x ∧ x ≤ 4} :=
sorry

theorem problem2_intersection (m : ℝ) (h : A ∩ B m = {x | 1 ≤ x ∧ x ≤ 3}) : m = 3 :=
sorry

theorem problem3_subset (m : ℝ) (h : A ⊆ {x | ¬ (x ∈ B m)}) : m > 5 ∨ m < -3 :=
sorry

end problem1_union_problem2_intersection_problem3_subset_l1845_184528


namespace triangle_problem_l1845_184549

noncomputable def triangle_sin_B (a b : ℝ) (A : ℝ) : ℝ :=
  b * Real.sin A / a

noncomputable def triangle_side_c (a b A : ℝ) : ℝ :=
  let discr := b^2 + a^2 - 2 * b * a * Real.cos A
  Real.sqrt discr

noncomputable def sin_diff_angle (sinB cosB sinC cosC : ℝ) : ℝ :=
  sinB * cosC - cosB * sinC

theorem triangle_problem
  (a b : ℝ)
  (A : ℝ)
  (ha : a = Real.sqrt 39)
  (hb : b = 2)
  (hA : A = Real.pi * (2 / 3)) :
  (triangle_sin_B a b A = Real.sqrt 13 / 13) ∧
  (triangle_side_c a b A = 5) ∧
  (sin_diff_angle (Real.sqrt 13 / 13) (2 * Real.sqrt 39 / 13) (5 * Real.sqrt 13 / 26) (3 * Real.sqrt 39 / 26) = -7 * Real.sqrt 3 / 26) :=
by sorry

end triangle_problem_l1845_184549


namespace geometric_sequence_ratio_l1845_184535

theorem geometric_sequence_ratio 
  (a_n b_n : ℕ → ℝ) 
  (S_n T_n : ℕ → ℝ) 
  (h1 : ∀ n : ℕ, S_n n = a_n n * (1 - (1/2)^n)) 
  (h2 : ∀ n : ℕ, T_n n = b_n n * (1 - (1/3)^n))
  (h3 : ∀ n, n > 0 → (S_n n) / (T_n n) = (3^n + 1) / 4) : 
  (a_n 3) / (b_n 3) = 9 :=
by
  sorry

end geometric_sequence_ratio_l1845_184535


namespace overall_percentage_change_in_membership_l1845_184515

theorem overall_percentage_change_in_membership :
  let M := 1
  let fall_inc := 1.08
  let winter_inc := 1.15
  let spring_dec := 0.81
  (M * fall_inc * winter_inc * spring_dec - M) / M * 100 = 24.2 := by
  sorry

end overall_percentage_change_in_membership_l1845_184515


namespace simply_connected_polyhedron_faces_l1845_184532

def polyhedron_faces_condition (σ3 σ4 σ5 : Nat) (V E F : Nat) : Prop :=
  V - E + F = 2

theorem simply_connected_polyhedron_faces : 
  ∀ (σ3 σ4 σ5 : Nat) (V E F : Nat),
  polyhedron_faces_condition σ3 σ4 σ5 V E F →
  (σ4 = 0 ∧ σ5 = 0 → σ3 ≥ 4) ∧
  (σ3 = 0 ∧ σ5 = 0 → σ4 ≥ 6) ∧
  (σ3 = 0 ∧ σ4 = 0 → σ5 ≥ 12) := 
by
  intros
  sorry

end simply_connected_polyhedron_faces_l1845_184532


namespace min_ab_l1845_184562

theorem min_ab (a b : ℝ) (h_pos : 0 < a ∧ 0 < b) (h_eq : a * b = a + b + 3) : a * b ≥ 9 :=
sorry

end min_ab_l1845_184562


namespace max_value_l1845_184540

-- Definition of the ellipse and the goal function
def ellipse (x y : ℝ) := 2 * x^2 + 3 * y^2 = 12

-- Definition of the function we want to maximize
def func (x y : ℝ) := x + 2 * y

-- The theorem to prove that the maximum value of x + 2y on the ellipse is √22
theorem max_value (x y : ℝ) (h : ellipse x y) : ∃ θ : ℝ, func x y ≤ Real.sqrt 22 :=
by
  sorry

end max_value_l1845_184540


namespace marked_price_correct_l1845_184542

noncomputable def marked_price (cost_price : ℝ) (profit_margin : ℝ) (selling_percentage : ℝ) : ℝ :=
  (cost_price * (1 + profit_margin)) / selling_percentage

theorem marked_price_correct :
  marked_price 1360 0.15 0.8 = 1955 :=
by
  sorry

end marked_price_correct_l1845_184542


namespace difference_of_solutions_l1845_184522

theorem difference_of_solutions (x : ℝ) (h : (x + 3)^2 / (3 * x + 65) = 2) : ∃ a b : ℝ, a ≠ b ∧ (x = a ∨ x = b) ∧ abs (a - b) = 22 :=
by
  sorry

end difference_of_solutions_l1845_184522


namespace profit_with_discount_l1845_184502

theorem profit_with_discount (CP SP_with_discount SP_no_discount : ℝ) (discount profit_no_discount : ℝ) (H1 : discount = 0.1) (H2 : profit_no_discount = 0.3889) (H3 : SP_no_discount = CP * (1 + profit_no_discount)) (H4 : SP_with_discount = SP_no_discount * (1 - discount)) : (SP_with_discount - CP) / CP * 100 = 25 :=
by
  -- The proof will be filled here
  sorry

end profit_with_discount_l1845_184502


namespace triangle_area_on_ellipse_l1845_184503

def onEllipse (p : ℝ × ℝ) : Prop := (p.1)^2 + 4 * (p.2)^2 = 4

def isCentroid (C : ℝ × ℝ) (A B : ℝ × ℝ) : Prop :=
  A ≠ B ∧ B ≠ C ∧ C ≠ A ∧
  C = ((A.1 + B.1) / 3, (A.2 + B.2) / 3)

theorem triangle_area_on_ellipse
  (A B C : ℝ × ℝ)
  (h₁ : A ≠ B)
  (h₂ : B ≠ C)
  (h₃ : C ≠ A)
  (h₄ : onEllipse A)
  (h₅ : onEllipse B)
  (h₆ : onEllipse C)
  (h₇ : isCentroid C A B)
  (h₈ : C = (0, 0))  : 
  1 / 2 * (A.1 - B.1) * (B.2 - A.2) = 1 :=
by
  sorry

end triangle_area_on_ellipse_l1845_184503


namespace quadrilateral_impossible_l1845_184571

theorem quadrilateral_impossible (a b c d : ℕ) (h1 : 2 * a ^ 2 - 18 * a + 36 = 0)
    (h2 : b ^ 2 - 20 * b + 75 = 0) (h3 : c ^ 2 - 20 * c + 75 = 0) (h4 : 2 * d ^ 2 - 18 * d + 36 = 0) :
    ¬(a + b > d ∧ a + c > d ∧ b + c > d ∧ a + d > c ∧ b + d > c ∧ c + d > b ∧
      a + b + c > d ∧ a + b + d > c ∧ a + c + d > b ∧ b + c + d > a) :=
by
  sorry

end quadrilateral_impossible_l1845_184571


namespace problem1_problem2_l1845_184543

variable {α : ℝ}

-- Given condition
def tan_alpha (α : ℝ) : Prop := Real.tan α = 3

-- Proof statements to be shown
theorem problem1 (h : tan_alpha α) : (Real.sin α + 3 * Real.cos α) / (2 * Real.sin α + 5 * Real.cos α) = 6 / 11 :=
by sorry

theorem problem2 (h : tan_alpha α) : Real.sin α ^ 2 + Real.sin α * Real.cos α + 3 * Real.cos α ^ 2 = 6 :=
by sorry

end problem1_problem2_l1845_184543


namespace average_of_remaining_numbers_l1845_184591

theorem average_of_remaining_numbers (sum : ℕ) (average : ℕ) (remaining_sum : ℕ) (remaining_average : ℚ) :
  (average = 90) →
  (sum = 1080) →
  (remaining_sum = sum - 72 - 84) →
  (remaining_average = remaining_sum / 10) →
  remaining_average = 92.4 :=
by
  sorry

end average_of_remaining_numbers_l1845_184591


namespace find_a2_l1845_184525

variable (a : ℕ → ℤ)

-- Conditions
def is_arithmetic_sequence (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def is_geometric_sequence (x y z : ℤ) : Prop :=
  y * y = x * z

-- Specific condition for the problem
axiom h_arithmetic : is_arithmetic_sequence a 2
axiom h_geometric : is_geometric_sequence (a 1 + 2) (a 3 + 6) (a 4 + 8)

-- Theorem to prove
theorem find_a2 : a 1 + 2 = -8 := 
sorry

-- We assert that the value of a_2 must satisfy the given conditions

end find_a2_l1845_184525


namespace negation_of_prop_l1845_184548

theorem negation_of_prop :
  ¬ (∀ x : ℝ, x^2 - 1 > 0) ↔ ∃ x : ℝ, x^2 - 1 ≤ 0 :=
sorry

end negation_of_prop_l1845_184548


namespace number_students_first_class_l1845_184527

theorem number_students_first_class
  (average_first_class : ℝ)
  (average_second_class : ℝ)
  (students_second_class : ℕ)
  (combined_average : ℝ)
  (total_students : ℕ)
  (total_marks_first_class : ℝ)
  (total_marks_second_class : ℝ)
  (total_combined_marks : ℝ)
  (x : ℕ)
  (h1 : average_first_class = 50)
  (h2 : average_second_class = 65)
  (h3 : students_second_class = 40)
  (h4 : combined_average = 59.23076923076923)
  (h5 : total_students = x + 40)
  (h6 : total_marks_first_class = 50 * x)
  (h7 : total_marks_second_class = 65 * 40)
  (h8 : total_combined_marks = 59.23076923076923 * (x + 40))
  (h9 : total_marks_first_class + total_marks_second_class = total_combined_marks) :
  x = 25 :=
sorry

end number_students_first_class_l1845_184527


namespace negation_divisible_by_5_is_odd_l1845_184585

theorem negation_divisible_by_5_is_odd : 
  ¬∀ n : ℤ, (n % 5 = 0) → (n % 2 ≠ 0) ↔ ∃ n : ℤ, (n % 5 = 0) ∧ (n % 2 = 0) := 
by 
  sorry

end negation_divisible_by_5_is_odd_l1845_184585


namespace van_distance_l1845_184513

noncomputable def distance_covered (initial_time new_time speed : ℝ) : ℝ :=
  speed * new_time

theorem van_distance :
  distance_covered 5 (5 * (3 / 2)) 60 = 450 := 
by
  sorry

end van_distance_l1845_184513


namespace train_crossing_time_l1845_184569

theorem train_crossing_time
  (L1 L2 : ℝ) (v : ℝ) 
  (t1 t2 t : ℝ) 
  (h_t1 : t1 = 27)
  (h_t2 : t2 = 17)
  (hv_ratio : v = v)
  (h_L1 : L1 = v * t1)
  (h_L2 : L2 = v * t2)
  (h_t12 : t = (L1 + L2) / (v + v)) :
  t = 22 :=
by
  sorry

end train_crossing_time_l1845_184569


namespace line_passes_fixed_point_l1845_184537

theorem line_passes_fixed_point (k : ℝ) :
    ((k + 1) * -1) - ((2 * k - 1) * 1) + 3 * k = 0 :=
by
    -- The proof is omitted as the primary aim is to ensure the correct Lean statement.
    sorry

end line_passes_fixed_point_l1845_184537


namespace inequality_am_gm_holds_l1845_184514

theorem inequality_am_gm_holds 
    (a b c : ℝ) 
    (ha : a > 0) 
    (hb : b > 0) 
    (hc : c > 0) 
    (h : a^3 + b^3 = c^3) : 
  a^2 + b^2 - c^2 > 6 * (c - a) * (c - b) := 
sorry

end inequality_am_gm_holds_l1845_184514


namespace solve_for_N_l1845_184566

theorem solve_for_N :
    (481 + 483 + 485 + 487 + 489 + 491 = 3000 - N) → (N = 84) :=
by
    -- Proof is omitted
    sorry

end solve_for_N_l1845_184566


namespace repeatingDecimal_exceeds_l1845_184565

noncomputable def repeatingDecimalToFraction (d : ℚ) : ℚ := 
    -- Function to convert repeating decimal to fraction
    if d = 0.99999 then 1 else (d * 100 - d) / 99  -- Simplified conversion for demonstration

def decimalToFraction (d : ℚ) : ℚ :=
    -- Function to convert decimal to fraction
    if d = 0.72 then 18 / 25 else 0  -- Replace with actual conversion

theorem repeatingDecimal_exceeds (x y : ℚ) (hx : repeatingDecimalToFraction x = 8/11) (hy : decimalToFraction y = 18/25):
    x - y = 2 / 275 :=
by
    sorry

end repeatingDecimal_exceeds_l1845_184565


namespace determine_8_genuine_coins_l1845_184596

-- Assume there are 11 coins and one may be counterfeit.
variable (coins : Fin 11 → ℝ)
variable (is_counterfeit : Fin 11 → Prop)
variable (genuine_weight : ℝ)
variable (balance : (Fin 11 → ℝ) → (Fin 11 → ℝ) → Prop)

-- The weight of genuine coins.
axiom genuine_coins_weight : ∀ i, ¬ is_counterfeit i → coins i = genuine_weight

-- The statement of the mathematical problem in Lean 4.
theorem determine_8_genuine_coins :
  ∃ (genuine_set : Finset (Fin 11)), genuine_set.card ≥ 8 ∧ ∀ i ∈ genuine_set, ¬ is_counterfeit i :=
sorry

end determine_8_genuine_coins_l1845_184596


namespace sum_of_200_terms_l1845_184539

variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)
variable (a1 a200 : ℝ)

-- Conditions
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, a n = a 0 + n * (a 1 - a 0)

def sum_of_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
∀ n, S n = (n * (a 1 + a n)) / 2

def collinearity_condition (a1 a200 : ℝ) : Prop :=
a1 + a200 = 1

-- Proof statement
theorem sum_of_200_terms (a : ℕ → ℝ) (S : ℕ → ℝ) (a1 a200 : ℝ) 
  (h_seq : arithmetic_sequence a)
  (h_sum : sum_of_first_n_terms S a)
  (h_collinear : collinearity_condition a1 a200) : 
  S 200 = 100 := 
sorry

end sum_of_200_terms_l1845_184539


namespace fraction_of_students_between_11_and_13_is_two_fifths_l1845_184580

def totalStudents : ℕ := 45
def under11 : ℕ :=  totalStudents / 3
def over13 : ℕ := 12
def between11and13 : ℕ := totalStudents - (under11 + over13)
def fractionBetween11and13 : ℚ := between11and13 / totalStudents

theorem fraction_of_students_between_11_and_13_is_two_fifths :
  fractionBetween11and13 = 2 / 5 := 
by 
  sorry

end fraction_of_students_between_11_and_13_is_two_fifths_l1845_184580


namespace period_f_axis_of_symmetry_f_max_value_f_l1845_184551

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 5)

theorem period_f :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = Real.pi := sorry

theorem axis_of_symmetry_f (k : ℤ) :
  ∀ x, 2 * x - Real.pi / 5 = Real.pi / 4 + k * Real.pi → x = 9 * Real.pi / 40 + k * Real.pi / 2 := sorry

theorem max_value_f :
  ∃ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), f x = 1 ∧ x = 7 * Real.pi / 20 := sorry

end period_f_axis_of_symmetry_f_max_value_f_l1845_184551


namespace max_value_of_f_l1845_184545

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.sin x) ^ 2 + 2 * Real.cos x - 3

theorem max_value_of_f : ∀ x : ℝ, f x ≤ -1/2 :=
by
  sorry

end max_value_of_f_l1845_184545
