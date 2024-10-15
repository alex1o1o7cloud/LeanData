import Mathlib

namespace NUMINAMATH_GPT_sum_of_prime_factors_l1349_134998

theorem sum_of_prime_factors (n : ℕ) (h : n = 257040) : 
  (2 + 5 + 3 + 107 = 117) :=
by sorry

end NUMINAMATH_GPT_sum_of_prime_factors_l1349_134998


namespace NUMINAMATH_GPT_total_exercise_time_l1349_134985

-- Definitions based on given conditions
def javier_daily : ℕ := 50
def javier_days : ℕ := 7
def sanda_daily : ℕ := 90
def sanda_days : ℕ := 3

-- Proof problem to verify the total exercise time for both Javier and Sanda
theorem total_exercise_time : javier_daily * javier_days + sanda_daily * sanda_days = 620 := by
  sorry

end NUMINAMATH_GPT_total_exercise_time_l1349_134985


namespace NUMINAMATH_GPT_boat_speed_determination_l1349_134947

theorem boat_speed_determination :
  ∃ x : ℝ, 
    (∀ u d : ℝ, u = 170 / (x + 6) ∧ d = 170 / (x - 6))
    ∧ (u + d = 68)
    ∧ (x = 9) := 
by
  sorry

end NUMINAMATH_GPT_boat_speed_determination_l1349_134947


namespace NUMINAMATH_GPT_gcd_polynomial_l1349_134971

theorem gcd_polynomial {b : ℕ} (h : 570 ∣ b) : Nat.gcd (4*b^3 + 2*b^2 + 5*b + 95) b = 95 := 
sorry

end NUMINAMATH_GPT_gcd_polynomial_l1349_134971


namespace NUMINAMATH_GPT_polynomial_simplification_l1349_134965

variable (x : ℝ)

theorem polynomial_simplification :
  (3 * x^2 + 5 * x + 9) * (x + 2) - (x + 2) * (x^2 + 5 * x - 72) + (4 * x - 15) * (x + 2) * (x + 4) =
  6 * x^3 - 28 * x^2 - 59 * x + 42 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_simplification_l1349_134965


namespace NUMINAMATH_GPT_parabola_properties_l1349_134918

-- Definitions of the conditions
def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c
def point_A (a b c : ℝ) : Prop := parabola a b c (-1) = 0
def point_B (a b c m : ℝ) : Prop := parabola a b c m = 0
def opens_downwards (a : ℝ) : Prop := a < 0
def valid_m (m : ℝ) : Prop := 1 < m ∧ m < 2

-- Conclusion ①
def conclusion_1 (a b : ℝ) : Prop := b > 0

-- Conclusion ②
def conclusion_2 (a c : ℝ) : Prop := 3 * a + 2 * c < 0

-- Conclusion ③
def conclusion_3 (a b c x1 x2 y1 y2 : ℝ) : Prop :=
  x1 < x2 ∧ x1 + x2 > 1 ∧ parabola a b c x1 = y1 ∧ parabola a b c x2 = y2 → y1 > y2

-- Conclusion ④
def conclusion_4 (a b c : ℝ) : Prop :=
  a ≤ -1 → ∃ x1 x2 : ℝ, (a * x1^2 + b * x1 + c = 1) ∧ (a * x2^2 + b * x2 + c = 1) ∧ (x1 ≠ x2)

-- The theorem to prove
theorem parabola_properties (a b c m : ℝ) :
  (opens_downwards a) →
  (point_A a b c) →
  (point_B a b c m) →
  (valid_m m) →
  (conclusion_1 a b) ∧ (conclusion_2 a c → false) ∧ (∀ x1 x2 y1 y2, conclusion_3 a b c x1 x2 y1 y2) ∧ (conclusion_4 a b c) :=
by
  sorry

end NUMINAMATH_GPT_parabola_properties_l1349_134918


namespace NUMINAMATH_GPT_find_range_t_l1349_134936

noncomputable def f (x t : ℝ) : ℝ :=
  if x < t then -6 + Real.exp (x - 1) else x^2 - 4 * x

theorem find_range_t (f : ℝ → ℝ → ℝ)
  (h : ∀ t : ℝ, (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ f x₁ t = x₁ - 6 ∧ f x₂ t = x₂ - 6 ∧ f x₃ t = x₃ - 6)) :
  ∀ t : ℝ, 1 < t ∧ t ≤ 2 := sorry

end NUMINAMATH_GPT_find_range_t_l1349_134936


namespace NUMINAMATH_GPT_contrapositive_statement_l1349_134921

theorem contrapositive_statement {a b : ℤ} :
  (∀ a b : ℤ, (a % 2 = 1 ∧ b % 2 = 1) → (a + b) % 2 = 0) →
  (∀ a b : ℤ, ¬((a + b) % 2 = 0) → ¬(a % 2 = 1 ∧ b % 2 = 1)) :=
by 
  intros h a b
  sorry

end NUMINAMATH_GPT_contrapositive_statement_l1349_134921


namespace NUMINAMATH_GPT_tidal_power_station_location_l1349_134904

-- Define the conditions
def tidal_power_plants : ℕ := 9
def first_bidirectional_plant := 1980
def significant_bidirectional_plant_location : String := "Jiangxia"
def largest_bidirectional_plant : Prop := true

-- Assumptions based on conditions
axiom china_has_9_tidal_power_plants : tidal_power_plants = 9
axiom first_bidirectional_in_1980 : (first_bidirectional_plant = 1980) -> significant_bidirectional_plant_location = "Jiangxia"
axiom largest_bidirectional_in_world : largest_bidirectional_plant

-- Definition of the problem
theorem tidal_power_station_location : significant_bidirectional_plant_location = "Jiangxia" :=
by
  sorry

end NUMINAMATH_GPT_tidal_power_station_location_l1349_134904


namespace NUMINAMATH_GPT_no_integer_solution_l1349_134928

theorem no_integer_solution :
  ¬(∃ x : ℤ, 7 - 3 * (x^2 - 2) > 19) :=
by
  sorry

end NUMINAMATH_GPT_no_integer_solution_l1349_134928


namespace NUMINAMATH_GPT_sachin_age_l1349_134932

theorem sachin_age {Sachin_age Rahul_age : ℕ} (h1 : Sachin_age + 14 = Rahul_age) (h2 : Sachin_age * 9 = Rahul_age * 7) : Sachin_age = 49 := by
sorry

end NUMINAMATH_GPT_sachin_age_l1349_134932


namespace NUMINAMATH_GPT_smallest_b_value_l1349_134978

theorem smallest_b_value (a b : ℕ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a - b = 7) 
    (h₄ : (Nat.gcd ((a^3 + b^3) / (a + b)) (a^2 * b)) = 12) : b = 6 :=
by
    -- proof goes here
    sorry

end NUMINAMATH_GPT_smallest_b_value_l1349_134978


namespace NUMINAMATH_GPT_problem_solution_l1349_134906

def x : ℤ := -2 + 3
def y : ℤ := abs (-5)
def z : ℤ := 4 * (-1/4)

theorem problem_solution : x + y + z = 5 := 
by
  -- Definitions based on the problem statement
  have h1 : x = -2 + 3 := rfl
  have h2 : y = abs (-5) := rfl
  have h3 : z = 4 * (-1/4) := rfl
  
  -- Exact result required to be proved. Adding placeholder for steps.
  sorry

end NUMINAMATH_GPT_problem_solution_l1349_134906


namespace NUMINAMATH_GPT_probability_comparison_l1349_134972

variables (M N : ℕ) (m n : ℝ)
variable (h₁ : m > 10^6)
variable (h₂ : n ≤ 10^6)

theorem probability_comparison (h₃: 0 < M) (h₄: 0 < N):
  (m * M) / (m * M + n * N) > (M / (M + N)) :=
by
  have h₅: n / m < 1 := sorry
  have h₆: M > 0 := by linarith
  have h₇: 1 + (n / m) * (N / M) < 2 := sorry
  sorry

end NUMINAMATH_GPT_probability_comparison_l1349_134972


namespace NUMINAMATH_GPT_jane_paints_correct_area_l1349_134951

def height_of_wall : ℕ := 10
def length_of_wall : ℕ := 15
def width_of_door : ℕ := 3
def height_of_door : ℕ := 5

def area_of_wall := height_of_wall * length_of_wall
def area_of_door := width_of_door * height_of_door
def area_to_be_painted := area_of_wall - area_of_door

theorem jane_paints_correct_area : area_to_be_painted = 135 := by
  sorry

end NUMINAMATH_GPT_jane_paints_correct_area_l1349_134951


namespace NUMINAMATH_GPT_ants_meet_distance_is_half_total_l1349_134934

-- Definitions given in the problem
structure Tile :=
  (width : ℤ)
  (length : ℤ)

structure Ant :=
  (start_position : String)

-- Conditions from the problem
def tile : Tile := ⟨4, 6⟩
def maricota : Ant := ⟨"M"⟩
def nandinha : Ant := ⟨"N"⟩
def total_lengths := 14
def total_widths := 12

noncomputable
def calculate_total_distance (total_lengths : ℤ) (total_widths : ℤ) (tile : Tile) := 
  (total_lengths * tile.length) + (total_widths * tile.width)

-- Question stated as a theorem
theorem ants_meet_distance_is_half_total :
  calculate_total_distance total_lengths total_widths tile = 132 →
  (calculate_total_distance total_lengths total_widths tile) / 2 = 66 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_ants_meet_distance_is_half_total_l1349_134934


namespace NUMINAMATH_GPT_quadratic_function_integer_values_not_imply_integer_coefficients_l1349_134943

theorem quadratic_function_integer_values_not_imply_integer_coefficients :
  ∃ (a b c : ℚ), (∀ x : ℤ, ∃ y : ℤ, (a * (x : ℚ)^2 + b * (x : ℚ) + c = (y : ℚ))) ∧
    (¬ (∃ (a_int b_int c_int : ℤ), a = (a_int : ℚ) ∧ b = (b_int : ℚ) ∧ c = (c_int : ℚ))) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_function_integer_values_not_imply_integer_coefficients_l1349_134943


namespace NUMINAMATH_GPT_find_h_from_quadratic_l1349_134963

theorem find_h_from_quadratic (
  p q r : ℝ) (h₁ : ∀ x, p * x^2 + q * x + r = 7 * (x - 5)^2 + 14) :
  ∀ m k h, (∀ x, 5 * p * x^2 + 5 * q * x + 5 * r = m * (x - h)^2 + k) → h = 5 :=
by
  intros m k h h₂
  sorry

end NUMINAMATH_GPT_find_h_from_quadratic_l1349_134963


namespace NUMINAMATH_GPT_find_investment_amount_l1349_134960

noncomputable def brokerage_fee (market_value : ℚ) : ℚ := (1 / 4 / 100) * market_value

noncomputable def actual_cost (market_value : ℚ) : ℚ := market_value + brokerage_fee market_value

noncomputable def income_per_100_face_value (interest_rate : ℚ) : ℚ := (interest_rate / 100) * 100

noncomputable def investment_amount (income : ℚ) (actual_cost_per_100 : ℚ) (income_per_100 : ℚ) : ℚ :=
  (income * actual_cost_per_100) / income_per_100

theorem find_investment_amount :
  investment_amount 756 (actual_cost 124.75) (income_per_100_face_value 10.5) = 9483.65625 :=
sorry

end NUMINAMATH_GPT_find_investment_amount_l1349_134960


namespace NUMINAMATH_GPT_income_ratio_l1349_134942

theorem income_ratio (I1 I2 E1 E2 : ℕ) (h1 : I1 = 5000) (h2 : E1 / E2 = 3 / 2) (h3 : I1 - E1 = 2000) (h4 : I2 - E2 = 2000) : I1 / I2 = 5 / 4 :=
by
  /- Proof omitted -/
  sorry

end NUMINAMATH_GPT_income_ratio_l1349_134942


namespace NUMINAMATH_GPT_min_value_of_T_l1349_134941

noncomputable def T (x p : ℝ) : ℝ := |x - p| + |x - 15| + |x - (15 + p)|

theorem min_value_of_T (p : ℝ) (hp : 0 < p ∧ p < 15) :
  ∃ x, p ≤ x ∧ x ≤ 15 ∧ T x p = 15 :=
sorry

end NUMINAMATH_GPT_min_value_of_T_l1349_134941


namespace NUMINAMATH_GPT_system_of_equations_solution_l1349_134940

theorem system_of_equations_solution (x y : ℝ) 
  (h1 : x - 2 * y = 1)
  (h2 : 3 * x + 4 * y = 23) :
  x = 5 ∧ y = 2 :=
sorry

end NUMINAMATH_GPT_system_of_equations_solution_l1349_134940


namespace NUMINAMATH_GPT_range_of_m_l1349_134903

theorem range_of_m (m : ℝ) (x : ℝ) :
  (|1 - (x - 1) / 2| ≤ 3) →
  (x^2 - 2 * x + 1 - m^2 ≤ 0) →
  (m > 0) →
  (∃ (q_is_necessary_but_not_sufficient_for_p : Prop), q_is_necessary_but_not_sufficient_for_p →
  (m ≥ 8)) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1349_134903


namespace NUMINAMATH_GPT_rectangle_sides_l1349_134984

theorem rectangle_sides (x y : ℕ) :
  (2 * x + 2 * y = x * y) →
  x > 0 →
  y > 0 →
  (x = 3 ∧ y = 6) ∨ (x = 6 ∧ y = 3) ∨ (x = 4 ∧ y = 4) :=
by
  sorry

end NUMINAMATH_GPT_rectangle_sides_l1349_134984


namespace NUMINAMATH_GPT_common_point_of_geometric_progression_l1349_134999

theorem common_point_of_geometric_progression (a b c x y : ℝ) (r : ℝ) 
  (h1 : b = a * r) (h2 : c = a * r^2) 
  (h3 : a * x + b * y = c) : 
  x = 1 / 2 ∧ y = -1 / 2 := 
sorry

end NUMINAMATH_GPT_common_point_of_geometric_progression_l1349_134999


namespace NUMINAMATH_GPT_arithmetic_sequence_n_l1349_134944

theorem arithmetic_sequence_n 
  (a : ℕ → ℕ)
  (S : ℕ → ℕ)
  (a1 : a 1 = 1)
  (a3_plus_a5 : a 3 + a 5 = 14)
  (Sn_eq_100 : S n = 100) :
  n = 10 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_n_l1349_134944


namespace NUMINAMATH_GPT_first_tap_fill_time_l1349_134983

theorem first_tap_fill_time (T : ℚ) :
  (∀ (second_tap_empty_time : ℚ), second_tap_empty_time = 8) →
  (∀ (combined_fill_time : ℚ), combined_fill_time = 40 / 3) →
  (1/T - 1/8 = 3/40) →
  T = 5 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_first_tap_fill_time_l1349_134983


namespace NUMINAMATH_GPT_vasya_hits_ship_l1349_134914

theorem vasya_hits_ship (board_size : ℕ) (ship_length : ℕ) (shots : ℕ) : 
  board_size = 10 ∧ ship_length = 4 ∧ shots = 24 → ∃ strategy : Fin board_size × Fin board_size → Prop, 
  (∀ pos, strategy pos → pos.1 * board_size + pos.2 < shots) ∧ 
  ∀ (ship_pos : Fin board_size × Fin board_size) (horizontal : Bool), 
  ∃ shot_pos, strategy shot_pos ∧ 
  (if horizontal then 
    ship_pos.1 = shot_pos.1 ∧ ship_pos.2 ≤ shot_pos.2 ∧ shot_pos.2 < ship_pos.2 + ship_length 
  else 
    ship_pos.2 = shot_pos.2 ∧ ship_pos.1 ≤ shot_pos.1 ∧ shot_pos.1 < ship_pos.1 + ship_length) :=
sorry

end NUMINAMATH_GPT_vasya_hits_ship_l1349_134914


namespace NUMINAMATH_GPT_average_reading_time_l1349_134902

theorem average_reading_time (t_Emery t_Serena : ℕ) (h1 : t_Emery = 20) (h2 : t_Serena = 5 * t_Emery) : 
  (t_Emery + t_Serena) / 2 = 60 := 
by
  sorry

end NUMINAMATH_GPT_average_reading_time_l1349_134902


namespace NUMINAMATH_GPT_sequence_is_constant_l1349_134933

noncomputable def sequence_condition (a : ℕ → ℝ) :=
  a 1 = 1 ∧ ∀ m n : ℕ, m > 0 → n > 0 → |a n - a m| ≤ 2 * m * n / (m ^ 2 + n ^ 2)

theorem sequence_is_constant (a : ℕ → ℝ) 
  (h : sequence_condition a) :
  ∀ n : ℕ, n > 0 → a n = 1 :=
by
  sorry

end NUMINAMATH_GPT_sequence_is_constant_l1349_134933


namespace NUMINAMATH_GPT_fish_ratio_l1349_134931

variables (O R B : ℕ)
variables (h1 : O = B + 25)
variables (h2 : B = 75)
variables (h3 : (O + B + R) / 3 = 75)

theorem fish_ratio : R / O = 1 / 2 :=
sorry

end NUMINAMATH_GPT_fish_ratio_l1349_134931


namespace NUMINAMATH_GPT_percent_y_of_x_l1349_134990

theorem percent_y_of_x (x y : ℝ) (h : 0.60 * (x - y) = 0.30 * (x + y)) : y / x = 1 / 3 :=
by
  -- proof steps would be provided here
  sorry

end NUMINAMATH_GPT_percent_y_of_x_l1349_134990


namespace NUMINAMATH_GPT_number_of_roosters_l1349_134962

def chickens := 9000
def ratio_roosters_hens := 2 / 1

theorem number_of_roosters (h : ratio_roosters_hens = 2 / 1) (c : chickens = 9000) : ∃ r : ℕ, r = 6000 := 
by sorry

end NUMINAMATH_GPT_number_of_roosters_l1349_134962


namespace NUMINAMATH_GPT_minimum_percentage_increase_mean_l1349_134911

def mean (s : List ℤ) : ℚ :=
  (s.sum : ℚ) / s.length

theorem minimum_percentage_increase_mean (F : List ℤ) (p1 p2 : ℤ) (F' : List ℤ)
  (hF : F = [ -4, -1, 0, 6, 9 ])
  (hp1 : p1 = 2) (hp2 : p2 = 3)
  (hF' : F' = [p1, p2, 0, 6, 9])
  : (mean F' - mean F) / mean F * 100 = 100 := 
sorry

end NUMINAMATH_GPT_minimum_percentage_increase_mean_l1349_134911


namespace NUMINAMATH_GPT_remaining_number_is_divisible_by_divisor_l1349_134907

def initial_number : ℕ := 427398
def subtracted_number : ℕ := 8
def remaining_number : ℕ := initial_number - subtracted_number
def divisor : ℕ := 10

theorem remaining_number_is_divisible_by_divisor :
  remaining_number % divisor = 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_remaining_number_is_divisible_by_divisor_l1349_134907


namespace NUMINAMATH_GPT_jacob_dimes_l1349_134959

-- Definitions of the conditions
def mrs_hilt_total_cents : ℕ := 2 * 1 + 2 * 10 + 2 * 5
def jacob_base_cents : ℕ := 4 * 1 + 1 * 5
def difference : ℕ := 13

-- The proof problem: prove Jacob has 1 dime.
theorem jacob_dimes (d : ℕ) (h : mrs_hilt_total_cents - (jacob_base_cents + 10 * d) = difference) : d = 1 := by
  sorry

end NUMINAMATH_GPT_jacob_dimes_l1349_134959


namespace NUMINAMATH_GPT_find_n_from_remainders_l1349_134927

theorem find_n_from_remainders (a n : ℕ) (h1 : a^2 % n = 8) (h2 : a^3 % n = 25) : n = 113 := 
by 
  -- proof needed here
  sorry

end NUMINAMATH_GPT_find_n_from_remainders_l1349_134927


namespace NUMINAMATH_GPT_tony_total_puzzle_time_l1349_134923

def warm_up_puzzle_time : ℕ := 10
def number_of_puzzles : ℕ := 2
def multiplier : ℕ := 3
def time_per_puzzle : ℕ := warm_up_puzzle_time * multiplier
def total_time : ℕ := warm_up_puzzle_time + number_of_puzzles * time_per_puzzle

theorem tony_total_puzzle_time : total_time = 70 := 
by
  sorry

end NUMINAMATH_GPT_tony_total_puzzle_time_l1349_134923


namespace NUMINAMATH_GPT_initial_bananas_per_child_l1349_134986

theorem initial_bananas_per_child 
    (absent : ℕ) (present : ℕ) (total : ℕ) (x : ℕ) (B : ℕ)
    (h1 : absent = 305)
    (h2 : present = 305)
    (h3 : total = 610)
    (h4 : B = present * (x + 2))
    (h5 : B = total * x) : 
    x = 2 :=
by
  sorry

end NUMINAMATH_GPT_initial_bananas_per_child_l1349_134986


namespace NUMINAMATH_GPT_divisor_of_109_l1349_134958

theorem divisor_of_109 (d : ℕ) (h : 109 = 9 * d + 1) : d = 12 :=
sorry

end NUMINAMATH_GPT_divisor_of_109_l1349_134958


namespace NUMINAMATH_GPT_g_triple_apply_l1349_134961

noncomputable def g (x : ℝ) : ℝ :=
  if x < 10 then x^2 - 9 else x - 15

theorem g_triple_apply : g (g (g 20)) = 1 :=
by
  sorry

end NUMINAMATH_GPT_g_triple_apply_l1349_134961


namespace NUMINAMATH_GPT_necklaces_sold_correct_l1349_134909

-- Define the given constants and conditions
def necklace_price : ℕ := 25
def bracelet_price : ℕ := 15
def earring_price : ℕ := 10
def ensemble_price : ℕ := 45
def bracelets_sold : ℕ := 10
def earrings_sold : ℕ := 20
def ensembles_sold : ℕ := 2
def total_revenue : ℕ := 565

-- Define the equation to calculate the total revenue
def total_revenue_calculation (N : ℕ) : ℕ :=
  (necklace_price * N) + (bracelet_price * bracelets_sold) + (earring_price * earrings_sold) + (ensemble_price * ensembles_sold)

-- Define the proof problem
theorem necklaces_sold_correct : 
  ∃ N : ℕ, total_revenue_calculation N = total_revenue ∧ N = 5 := by
  sorry

end NUMINAMATH_GPT_necklaces_sold_correct_l1349_134909


namespace NUMINAMATH_GPT_find_a6_l1349_134996

theorem find_a6 (a : ℕ → ℚ) (h₁ : ∀ n, a (n + 1) = 2 * a n - 1) (h₂ : a 8 = 16) : a 6 = 19 / 4 :=
sorry

end NUMINAMATH_GPT_find_a6_l1349_134996


namespace NUMINAMATH_GPT_evaluate_expression_l1349_134957

-- Define the base value
def base := 3000

-- Define the exponential expression
def exp_value := base ^ base

-- Prove that base * exp_value equals base ^ (1 + base)
theorem evaluate_expression : base * exp_value = base ^ (1 + base) := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1349_134957


namespace NUMINAMATH_GPT_baking_time_one_batch_l1349_134956

theorem baking_time_one_batch (x : ℕ) (time_icing_per_batch : ℕ) (num_batches : ℕ) (total_time : ℕ)
  (h1 : num_batches = 4)
  (h2 : time_icing_per_batch = 30)
  (h3 : total_time = 200)
  (h4 : total_time = num_batches * x + num_batches * time_icing_per_batch) :
  x = 20 :=
by
  rw [h1, h2, h3] at h4
  sorry

end NUMINAMATH_GPT_baking_time_one_batch_l1349_134956


namespace NUMINAMATH_GPT_range_of_a_l1349_134993

noncomputable def f (x : ℝ) : ℝ := sorry -- f(x) is an odd and monotonically increasing function, to be defined later.

noncomputable def g (x a : ℝ) : ℝ :=
  f (x^2) + f (a - 2 * |x|)

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 x3 x4 : ℝ, g x1 a = 0 ∧ g x2 a = 0 ∧ g x3 a = 0 ∧ g x4 a = 0 ∧
    x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4) ↔
  0 < a ∧ a < 1 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1349_134993


namespace NUMINAMATH_GPT_find_opposite_pair_l1349_134916

def is_opposite (x y : ℤ) : Prop := x = -y

theorem find_opposite_pair :
  ¬is_opposite 4 4 ∧ ¬is_opposite 2 2 ∧ ¬is_opposite (-8) (-8) ∧ is_opposite 4 (-4) := 
by
  sorry

end NUMINAMATH_GPT_find_opposite_pair_l1349_134916


namespace NUMINAMATH_GPT_num_terms_in_expansion_eq_3_pow_20_l1349_134967

-- Define the expression 
def expr (x y : ℝ) := (1 + x + y) ^ 20

-- Statement of the problem
theorem num_terms_in_expansion_eq_3_pow_20 (x y : ℝ) : (3 : ℝ)^20 = (1 + x + y) ^ 20 :=
by sorry

end NUMINAMATH_GPT_num_terms_in_expansion_eq_3_pow_20_l1349_134967


namespace NUMINAMATH_GPT_trip_time_l1349_134975

theorem trip_time (T : ℝ) (x : ℝ) : 
  (150 / 4 = 50 / 30 + (x - 50) / 4 + (150 - x) / 30) → (T = 37.5) :=
by
  sorry

end NUMINAMATH_GPT_trip_time_l1349_134975


namespace NUMINAMATH_GPT_function_relation_l1349_134976

noncomputable def f (x : ℝ) : ℝ := x * Real.sin x

theorem function_relation:
  f (-Real.pi / 3) > f 1 ∧ f 1 > f (Real.pi / 5) :=
by 
  sorry

end NUMINAMATH_GPT_function_relation_l1349_134976


namespace NUMINAMATH_GPT_point_P_inside_circle_l1349_134922

theorem point_P_inside_circle
  (a b c : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : a > b)
  (e : ℝ)
  (h4 : e = 1 / 2)
  (x1 x2 : ℝ)
  (hx1 : a * x1 ^ 2 + b * x1 - c = 0)
  (hx2 : a * x2 ^ 2 + b * x2 - c = 0) :
  x1 ^ 2 + x2 ^ 2 < 2 :=
by
  sorry

end NUMINAMATH_GPT_point_P_inside_circle_l1349_134922


namespace NUMINAMATH_GPT_range_of_a_l1349_134920

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x^2 - a * x + 5 < 0) ↔ (a < -2 * Real.sqrt 5 ∨ a > 2 * Real.sqrt 5) := 
by 
  sorry

end NUMINAMATH_GPT_range_of_a_l1349_134920


namespace NUMINAMATH_GPT_negation_abs_val_statement_l1349_134987

theorem negation_abs_val_statement (x : ℝ) :
  ¬ (|x| ≤ 3 ∨ |x| > 5) ↔ (|x| > 3 ∧ |x| ≤ 5) :=
by sorry

end NUMINAMATH_GPT_negation_abs_val_statement_l1349_134987


namespace NUMINAMATH_GPT_system_of_equations_n_eq_1_l1349_134970

theorem system_of_equations_n_eq_1 {x y n : ℝ} 
  (h₁ : 5 * x - 4 * y = n) 
  (h₂ : 3 * x + 5 * y = 8)
  (h₃ : x = y) : 
  n = 1 := 
by
  sorry

end NUMINAMATH_GPT_system_of_equations_n_eq_1_l1349_134970


namespace NUMINAMATH_GPT_intersection_points_lie_on_ellipse_l1349_134938

theorem intersection_points_lie_on_ellipse (s : ℝ) : 
  ∃ (x y : ℝ), (2 * s * x - 3 * y - 4 * s = 0 ∧ x - 3 * s * y + 4 = 0) ∧ (x^2 / 16 + y^2 / 9 = 1) :=
sorry

end NUMINAMATH_GPT_intersection_points_lie_on_ellipse_l1349_134938


namespace NUMINAMATH_GPT_max_chord_length_l1349_134935

noncomputable def family_of_curves (θ x y : ℝ) := 
  2 * (2 * Real.sin θ - Real.cos θ + 3) * x^2 - (8 * Real.sin θ + Real.cos θ + 1) * y = 0

def line (x y : ℝ) := 2 * x = y

theorem max_chord_length :
  (∀ (θ : ℝ), ∀ (x y : ℝ), family_of_curves θ x y → line x y) → 
  ∃ (L : ℝ), L = 8 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_max_chord_length_l1349_134935


namespace NUMINAMATH_GPT_volcano_ash_height_l1349_134900

theorem volcano_ash_height (r d : ℝ) (h : r = 2700) (h₁ : 2 * r = 18 * d) : d = 300 :=
by
  sorry

end NUMINAMATH_GPT_volcano_ash_height_l1349_134900


namespace NUMINAMATH_GPT_inequality_proof_l1349_134912

variable (u v w : ℝ)

theorem inequality_proof (h1 : u > 0) (h2 : v > 0) (h3 : w > 0) (h4 : u + v + w + Real.sqrt (u * v * w) = 4) :
    Real.sqrt (u * v / w) + Real.sqrt (v * w / u) + Real.sqrt (w * u / v) ≥ u + v + w := 
  sorry

end NUMINAMATH_GPT_inequality_proof_l1349_134912


namespace NUMINAMATH_GPT_stephen_female_worker_ants_l1349_134973

-- Define the conditions
def stephen_ants : ℕ := 110
def worker_ants (total_ants : ℕ) : ℕ := total_ants / 2
def male_worker_ants (workers : ℕ) : ℕ := (20 / 100) * workers

-- Define the question and correct answer
def female_worker_ants (total_ants : ℕ) : ℕ :=
  let workers := worker_ants total_ants
  workers - male_worker_ants workers

-- The theorem to prove
theorem stephen_female_worker_ants : female_worker_ants stephen_ants = 44 :=
  by sorry -- Skip the proof for now

end NUMINAMATH_GPT_stephen_female_worker_ants_l1349_134973


namespace NUMINAMATH_GPT_elvins_first_month_bill_l1349_134995

-- Define the variables involved
variables (F C : ℝ)

-- State the given conditions
def condition1 : Prop := F + C = 48
def condition2 : Prop := F + 2 * C = 90

-- State the theorem we need to prove
theorem elvins_first_month_bill (F C : ℝ) (h1 : F + C = 48) (h2 : F + 2 * C = 90) : F + C = 48 :=
by sorry

end NUMINAMATH_GPT_elvins_first_month_bill_l1349_134995


namespace NUMINAMATH_GPT_man_age_twice_son_age_l1349_134910

theorem man_age_twice_son_age (S M X : ℕ) (h1 : S = 28) (h2 : M = S + 30) (h3 : M + X = 2 * (S + X)) : X = 2 :=
by
  sorry

end NUMINAMATH_GPT_man_age_twice_son_age_l1349_134910


namespace NUMINAMATH_GPT_simplify_expression_l1349_134929

variable {R : Type} [LinearOrderedField R]

theorem simplify_expression (x y z : R) (h : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) (h_sum : x + y + z = 3) :
  (1 / (y^2 + z^2 - x^2) + 1 / (x^2 + z^2 - y^2) + 1 / (x^2 + y^2 - z^2)) =
    3 / (-9 + 6 * y + 6 * z - 2 * y * z) :=
  sorry

end NUMINAMATH_GPT_simplify_expression_l1349_134929


namespace NUMINAMATH_GPT_range_of_m_l1349_134937

theorem range_of_m (m : ℝ) : 
  (¬ (∀ x : ℝ, x^2 + m * x + 1 = 0 → x > 0) → m ≥ -2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1349_134937


namespace NUMINAMATH_GPT_least_distance_fly_crawled_l1349_134969

noncomputable def leastDistance (baseRadius height startDist endDist : ℝ) : ℝ :=
  let C := 2 * Real.pi * baseRadius
  let slantHeight := Real.sqrt (baseRadius ^ 2 + height ^ 2)
  let theta := C / slantHeight
  let x1 := startDist * Real.cos 0
  let y1 := startDist * Real.sin 0
  let x2 := endDist * Real.cos (theta / 2)
  let y2 := endDist * Real.sin (theta / 2)
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem least_distance_fly_crawled (baseRadius height startDist endDist : ℝ) (h1 : baseRadius = 500) (h2 : height = 150 * Real.sqrt 7) (h3 : startDist = 150) (h4 : endDist = 300 * Real.sqrt 2) :
  leastDistance baseRadius height startDist endDist = 150 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_GPT_least_distance_fly_crawled_l1349_134969


namespace NUMINAMATH_GPT_balcony_more_than_orchestra_l1349_134968

theorem balcony_more_than_orchestra (O B : ℕ) 
  (h1 : O + B = 355) 
  (h2 : 12 * O + 8 * B = 3320) : 
  B - O = 115 :=
by 
  -- Sorry, this will skip the proof.
  sorry

end NUMINAMATH_GPT_balcony_more_than_orchestra_l1349_134968


namespace NUMINAMATH_GPT_order_of_magnitude_l1349_134966

theorem order_of_magnitude (a b : ℝ) (h1 : a > 0) (h2 : b < 0) (h3 : |a| < |b|) :
  -b > a ∧ a > -a ∧ -a > b := by
  sorry

end NUMINAMATH_GPT_order_of_magnitude_l1349_134966


namespace NUMINAMATH_GPT_roque_bike_time_l1349_134913

-- Definitions of conditions
def roque_walk_time_per_trip : ℕ := 2
def roque_walk_trips_per_week : ℕ := 3
def roque_bike_trips_per_week : ℕ := 2
def total_commuting_time_per_week : ℕ := 16

-- Statement of the problem to prove
theorem roque_bike_time (B : ℕ) :
  (roque_walk_time_per_trip * 2 * roque_walk_trips_per_week + roque_bike_trips_per_week * 2 * B = total_commuting_time_per_week) → 
  B = 1 :=
by
  sorry

end NUMINAMATH_GPT_roque_bike_time_l1349_134913


namespace NUMINAMATH_GPT_math_problem_l1349_134915

theorem math_problem (x y : ℝ) (h1 : x + Real.sin y = 2023) (h2 : x + 2023 * Real.cos y = 2022) (h3 : Real.pi / 2 ≤ y ∧ y ≤ Real.pi) :
  x + y = 2022 + Real.pi / 2 :=
sorry

end NUMINAMATH_GPT_math_problem_l1349_134915


namespace NUMINAMATH_GPT_largest_divisor_l1349_134930

theorem largest_divisor (n : ℕ) (hn : n > 0) (h : 360 ∣ n^3) :
  ∃ w : ℕ, w > 0 ∧ w ∣ n ∧ ∀ d : ℕ, (d > 0 ∧ d ∣ n) → d ≤ 30 := 
sorry

end NUMINAMATH_GPT_largest_divisor_l1349_134930


namespace NUMINAMATH_GPT_find_height_on_BC_l1349_134988

noncomputable def height_on_BC (a b : ℝ) (A B C : ℝ) : ℝ := b * (Real.sin C)

theorem find_height_on_BC (A B C a b h : ℝ)
  (h_a: a = Real.sqrt 3)
  (h_b: b = Real.sqrt 2)
  (h_cos: 1 + 2 * Real.cos (B + C) = 0)
  (h_A: A = Real.pi / 3)
  (h_B: B = Real.pi / 4)
  (h_C: C = 5 * Real.pi / 12)
  (h_h: h = height_on_BC a b A B C) :
  h = (Real.sqrt 3 + 1) / 2 :=
sorry

end NUMINAMATH_GPT_find_height_on_BC_l1349_134988


namespace NUMINAMATH_GPT_mary_potatoes_l1349_134939

theorem mary_potatoes (original new_except : ℕ) (h₁ : original = 25) (h₂ : new_except = 7) :
  original + new_except = 32 := by
  sorry

end NUMINAMATH_GPT_mary_potatoes_l1349_134939


namespace NUMINAMATH_GPT_used_more_brown_sugar_l1349_134948

-- Define the amounts of sugar used
def brown_sugar : ℝ := 0.62
def white_sugar : ℝ := 0.25

-- Define the statement to prove
theorem used_more_brown_sugar : brown_sugar - white_sugar = 0.37 :=
by
  sorry

end NUMINAMATH_GPT_used_more_brown_sugar_l1349_134948


namespace NUMINAMATH_GPT_average_age_of_town_l1349_134901

-- Definitions based on conditions
def ratio_of_women_to_men (nw nm : ℕ) : Prop := nw * 8 = nm * 9

def young_men (nm : ℕ) (n_young_men : ℕ) (average_age_young : ℕ) : Prop :=
  n_young_men = 40 ∧ average_age_young = 25

def remaining_men_average_age (nm n_young_men : ℕ) (average_age_remaining : ℕ) : Prop :=
  average_age_remaining = 35

def women_average_age (average_age_women : ℕ) : Prop :=
  average_age_women = 30

-- Complete problem statement we need to prove
theorem average_age_of_town (nw nm : ℕ) (total_avg_age : ℕ) :
  ratio_of_women_to_men nw nm →
  young_men nm 40 25 →
  remaining_men_average_age nm 40 35 →
  women_average_age 30 →
  total_avg_age = 32 * 17 + 6 :=
sorry

end NUMINAMATH_GPT_average_age_of_town_l1349_134901


namespace NUMINAMATH_GPT_recycling_points_l1349_134926

theorem recycling_points (chloe_recycled : ℤ) (friends_recycled : ℤ) (points_per_pound : ℤ) :
  chloe_recycled = 28 ∧ friends_recycled = 2 ∧ points_per_pound = 6 → (chloe_recycled + friends_recycled) / points_per_pound = 5 :=
by
  sorry

end NUMINAMATH_GPT_recycling_points_l1349_134926


namespace NUMINAMATH_GPT_foxes_wolves_bears_num_l1349_134982

-- Definitions and theorem statement
def num_hunters := 45
def num_rabbits := 2008
def rabbits_per_fox := 59
def rabbits_per_wolf := 41
def rabbits_per_bear := 40

theorem foxes_wolves_bears_num (x y z : ℤ) : 
  x + y + z = num_hunters → 
  rabbits_per_wolf * x + rabbits_per_fox * y + rabbits_per_bear * z = num_rabbits → 
  x = 18 ∧ y = 10 ∧ z = 17 :=
by 
  intro h1 h2 
  sorry

end NUMINAMATH_GPT_foxes_wolves_bears_num_l1349_134982


namespace NUMINAMATH_GPT_jillian_apartment_size_l1349_134980

theorem jillian_apartment_size :
  ∃ (s : ℝ), (1.20 * s = 720) ∧ s = 600 := by
sorry

end NUMINAMATH_GPT_jillian_apartment_size_l1349_134980


namespace NUMINAMATH_GPT_value_of_certain_number_l1349_134974

theorem value_of_certain_number (a b : ℕ) (h : 1 / 7 * 8 = 5) (h2 : 1 / 5 * b = 35) : b = 175 :=
by
  -- by assuming the conditions hold, we need to prove b = 175
  sorry

end NUMINAMATH_GPT_value_of_certain_number_l1349_134974


namespace NUMINAMATH_GPT_pyarelal_loss_l1349_134989

theorem pyarelal_loss (P : ℝ) (total_loss : ℝ) (h1 : total_loss = 670) (h2 : 1 / 9 * P + P = 10 / 9 * P):
  (9 / (1 + 9)) * total_loss = 603 :=
by
  sorry

end NUMINAMATH_GPT_pyarelal_loss_l1349_134989


namespace NUMINAMATH_GPT_proof_of_calculation_l1349_134991

theorem proof_of_calculation : (7^2 - 5^2)^4 = 331776 := by
  sorry

end NUMINAMATH_GPT_proof_of_calculation_l1349_134991


namespace NUMINAMATH_GPT_find_k_l1349_134949

def vector := ℝ × ℝ  -- Define a vector as a pair of real numbers

def dot_product (v1 v2 : vector) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def a (k : ℝ) : vector := (k, 3)
def b : vector := (1, 4)
def c : vector := (2, 1)
def linear_combination (k : ℝ) : vector := ((2 * k - 3), -6)

theorem find_k (k : ℝ) (h : dot_product (linear_combination k) c = 0) : k = 3 := by
  sorry

end NUMINAMATH_GPT_find_k_l1349_134949


namespace NUMINAMATH_GPT_math_and_english_scores_sum_l1349_134905

theorem math_and_english_scores_sum (M E : ℕ) (total_score : ℕ) :
  (∀ (H : ℕ), H = (50 + M + E) / 3 → 
   50 + M + E + H = total_score) → 
   total_score = 248 → 
   M + E = 136 :=
by
  intros h1 h2;
  sorry

end NUMINAMATH_GPT_math_and_english_scores_sum_l1349_134905


namespace NUMINAMATH_GPT_total_value_of_coins_l1349_134950

variables {p n : ℕ}

-- Ryan has 17 coins consisting of pennies and nickels
axiom coins_eq : p + n = 17

-- The number of pennies is equal to the number of nickels
axiom pennies_eq_nickels : p = n

-- Prove that the total value of Ryan's coins is 49 cents
theorem total_value_of_coins : (p * 1 + n * 5) = 49 :=
by sorry

end NUMINAMATH_GPT_total_value_of_coins_l1349_134950


namespace NUMINAMATH_GPT_remainder_product_div_10_l1349_134994

def unitsDigit (n : ℕ) : ℕ := n % 10

theorem remainder_product_div_10 :
  let a := 1734
  let b := 5389
  let c := 80607
  let p := a * b * c
  unitsDigit p = 2 := by
  sorry

end NUMINAMATH_GPT_remainder_product_div_10_l1349_134994


namespace NUMINAMATH_GPT_sum_of_reflected_coordinates_l1349_134955

noncomputable def sum_of_coordinates (C D : ℝ × ℝ) : ℝ :=
  C.1 + C.2 + D.1 + D.2

theorem sum_of_reflected_coordinates (y : ℝ) :
  let C := (3, y)
  let D := (3, -y)
  sum_of_coordinates C D = 6 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_reflected_coordinates_l1349_134955


namespace NUMINAMATH_GPT_total_earnings_correct_l1349_134953

-- Definitions for the conditions
def price_per_bracelet := 5
def price_for_two_bracelets := 8
def initial_bracelets := 30
def earnings_from_selling_at_5_each := 60

-- Variables to store intermediate calculations
def bracelets_sold_at_5_each := earnings_from_selling_at_5_each / price_per_bracelet
def remaining_bracelets := initial_bracelets - bracelets_sold_at_5_each
def pairs_sold_at_8_each := remaining_bracelets / 2
def earnings_from_pairs := pairs_sold_at_8_each * price_for_two_bracelets
def total_earnings := earnings_from_selling_at_5_each + earnings_from_pairs

-- The theorem stating that Zayne made $132 in total
theorem total_earnings_correct :
  total_earnings = 132 :=
sorry

end NUMINAMATH_GPT_total_earnings_correct_l1349_134953


namespace NUMINAMATH_GPT_find_other_digits_l1349_134952

def is_divisible_by_9 (n : ℕ) : Prop :=
  n % 9 = 0

def tens_digit (n : ℕ) : ℕ :=
  (n / 10) % 10

theorem find_other_digits (n : ℕ) (h : ℕ) :
  tens_digit n = h →
  h = 1 →
  is_divisible_by_9 n →
  ∃ m : ℕ, m < 9 ∧ n = 10 * ((n / 10) / 10) * 10 + h * 10 + m ∧ (∃ k : ℕ, k * 9 = h + m + (n / 100)) :=
sorry

end NUMINAMATH_GPT_find_other_digits_l1349_134952


namespace NUMINAMATH_GPT_orthogonal_planes_k_value_l1349_134946

theorem orthogonal_planes_k_value
  (k : ℝ)
  (h : 3 * (-1) + 1 * 1 + (-2) * k = 0) : 
  k = -1 :=
sorry

end NUMINAMATH_GPT_orthogonal_planes_k_value_l1349_134946


namespace NUMINAMATH_GPT_exponent_calculation_l1349_134945

theorem exponent_calculation :
  ((19 ^ 11) / (19 ^ 8) * (19 ^ 3) = 47015881) :=
by
  sorry

end NUMINAMATH_GPT_exponent_calculation_l1349_134945


namespace NUMINAMATH_GPT_fraction_planted_of_field_is_correct_l1349_134924

/-- Given a right triangle with legs 5 units and 12 units, and a small unplanted square S
at the right-angle vertex such that the shortest distance from S to the hypotenuse is 3 units,
prove that the fraction of the field that is planted is 52761/857430. -/
theorem fraction_planted_of_field_is_correct :
  let area_triangle := (5 * 12) / 2
  let area_square := (180 / 169) ^ 2
  let area_planted := area_triangle - area_square
  let fraction_planted := area_planted / area_triangle
  fraction_planted = 52761 / 857430 :=
sorry

end NUMINAMATH_GPT_fraction_planted_of_field_is_correct_l1349_134924


namespace NUMINAMATH_GPT_Megan_acorns_now_l1349_134977

def initial_acorns := 16
def given_away_acorns := 7
def remaining_acorns := initial_acorns - given_away_acorns

theorem Megan_acorns_now : remaining_acorns = 9 := by
  sorry

end NUMINAMATH_GPT_Megan_acorns_now_l1349_134977


namespace NUMINAMATH_GPT_girls_more_than_boys_l1349_134919

-- Defining the conditions
def ratio_boys_girls : Nat := 3 / 4
def total_students : Nat := 42

-- Defining the hypothesis based on conditions
theorem girls_more_than_boys : (total_students * ratio_boys_girls) / (3 + 4) * (4 - 3) = 6 := by
  sorry

end NUMINAMATH_GPT_girls_more_than_boys_l1349_134919


namespace NUMINAMATH_GPT_number_of_B_students_l1349_134908

/- Define the assumptions of the problem -/
variable (x : ℝ)  -- the number of students who earn a B

/- Express the number of students getting each grade in terms of x -/
def number_of_A (x : ℝ) := 0.6 * x
def number_of_C (x : ℝ) := 1.3 * x
def number_of_D (x : ℝ) := 0.8 * x
def total_students (x : ℝ) := number_of_A x + x + number_of_C x + number_of_D x

/- Prove that x = 14 for the total number of students being 50 -/
theorem number_of_B_students : total_students x = 50 → x = 14 :=
by 
  sorry

end NUMINAMATH_GPT_number_of_B_students_l1349_134908


namespace NUMINAMATH_GPT_find_P_l1349_134979

theorem find_P (P Q R S : ℕ) (h1 : P ≠ Q) (h2 : P ≠ R) (h3 : P ≠ S) (h4 : Q ≠ R) (h5 : Q ≠ S) (h6 : R ≠ S)
  (h7 : P > 0) (h8 : Q > 0) (h9 : R > 0) (h10 : S > 0)
  (hPQ : P * Q = 72) (hRS : R * S = 72) (hDiff : P - Q = R + S) : P = 12 :=
by
  sorry

end NUMINAMATH_GPT_find_P_l1349_134979


namespace NUMINAMATH_GPT_average_salary_l1349_134954

theorem average_salary (a b c d e : ℕ) (h₁ : a = 8000) (h₂ : b = 5000) (h₃ : c = 15000) (h₄ : d = 7000) (h₅ : e = 9000) :
  (a + b + c + d + e) / 5 = 9000 :=
by sorry

end NUMINAMATH_GPT_average_salary_l1349_134954


namespace NUMINAMATH_GPT_selling_price_before_clearance_l1349_134964

-- Define the cost price (CP)
def CP : ℝ := 100

-- Define the gain percent before the clearance sale
def gain_percent_before : ℝ := 0.35

-- Define the discount percent during the clearance sale
def discount_percent : ℝ := 0.10

-- Define the gain percent during the clearance sale
def gain_percent_sale : ℝ := 0.215

-- Calculate the selling price before the clearance sale (SP_before)
def SP_before : ℝ := CP * (1 + gain_percent_before)

-- Calculate the selling price during the clearance sale (SP_sale)
def SP_sale : ℝ := SP_before * (1 - discount_percent)

-- Proof statement in Lean 4
theorem selling_price_before_clearance : SP_before = 135 :=
by
  -- Place to fill in the proof later
  sorry

end NUMINAMATH_GPT_selling_price_before_clearance_l1349_134964


namespace NUMINAMATH_GPT_rosie_pie_count_l1349_134981

-- Conditions and definitions
def apples_per_pie (total_apples pies : ℕ) : ℕ := total_apples / pies

-- Theorem statement (mathematical proof problem)
theorem rosie_pie_count :
  ∀ (a p : ℕ), a = 12 → p = 3 → (36 : ℕ) / (apples_per_pie a p) = 9 :=
by
  intros a p ha hp
  rw [ha, hp]
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_rosie_pie_count_l1349_134981


namespace NUMINAMATH_GPT_neg_proposition_equiv_l1349_134925

theorem neg_proposition_equiv :
  (¬ ∀ n : ℕ, n^2 ≤ 2^n) ↔ (∃ n : ℕ, n^2 > 2^n) :=
by
  sorry

end NUMINAMATH_GPT_neg_proposition_equiv_l1349_134925


namespace NUMINAMATH_GPT_cos_value_given_sin_l1349_134992

theorem cos_value_given_sin (α : ℝ) (h : Real.sin (π / 6 + α) = 3 / 5) : 
  Real.cos (π / 3 - α) = 3 / 5 :=
sorry

end NUMINAMATH_GPT_cos_value_given_sin_l1349_134992


namespace NUMINAMATH_GPT_smallest_possible_area_square_l1349_134997

theorem smallest_possible_area_square : 
  ∃ (c : ℝ), (∀ (x y : ℝ), ((y = 3 * x - 20) ∨ (y = x^2)) ∧ 
      (10 * (9 + 4 * c) = ((c + 20) / Real.sqrt 10) ^ 2) ∧ 
      (c = 80) ∧ 
      (10 * (9 + 4 * c) = 3290)) :=
by {
  use 80,
  sorry
}

end NUMINAMATH_GPT_smallest_possible_area_square_l1349_134997


namespace NUMINAMATH_GPT_variance_is_stability_measure_l1349_134917

def stability_measure (yields : Fin 10 → ℝ) : Prop :=
  let mean := (yields 0 + yields 1 + yields 2 + yields 3 + yields 4 + yields 5 + yields 6 + yields 7 + yields 8 + yields 9) / 10
  let variance := 
    ((yields 0 - mean)^2 + (yields 1 - mean)^2 + (yields 2 - mean)^2 + (yields 3 - mean)^2 + 
     (yields 4 - mean)^2 + (yields 5 - mean)^2 + (yields 6 - mean)^2 + (yields 7 - mean)^2 + 
     (yields 8 - mean)^2 + (yields 9 - mean)^2) / 10
  true -- just a placeholder, would normally state that this is the appropriate measure

theorem variance_is_stability_measure (yields : Fin 10 → ℝ) : stability_measure yields :=
by 
  sorry

end NUMINAMATH_GPT_variance_is_stability_measure_l1349_134917
