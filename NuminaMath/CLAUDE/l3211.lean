import Mathlib

namespace stockholm_malmo_distance_l3211_321158

/-- The scale factor of the map, representing kilometers per centimeter. -/
def scale : ℝ := 10

/-- The distance between Stockholm and Malmo on the map, in centimeters. -/
def map_distance : ℝ := 112

/-- The actual distance between Stockholm and Malmo, in kilometers. -/
def actual_distance : ℝ := map_distance * scale

theorem stockholm_malmo_distance : actual_distance = 1120 := by
  sorry

end stockholm_malmo_distance_l3211_321158


namespace expression_equals_sum_l3211_321128

theorem expression_equals_sum (a b c : ℝ) (ha : a = 14) (hb : b = 19) (hc : c = 23) :
  (a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)) /
  (a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b)) = a + b + c := by
  sorry

#eval (14 : ℝ) + 19 + 23

end expression_equals_sum_l3211_321128


namespace big_bottles_count_l3211_321144

/-- The number of big bottles initially in storage -/
def big_bottles : ℕ := 14000

/-- The number of small bottles initially in storage -/
def small_bottles : ℕ := 6000

/-- The percentage of small bottles sold -/
def small_bottles_sold_percent : ℚ := 20 / 100

/-- The percentage of big bottles sold -/
def big_bottles_sold_percent : ℚ := 23 / 100

/-- The total number of bottles remaining in storage -/
def total_remaining : ℕ := 15580

theorem big_bottles_count :
  (small_bottles * (1 - small_bottles_sold_percent) : ℚ).floor +
  (big_bottles * (1 - big_bottles_sold_percent) : ℚ).floor = total_remaining := by
  sorry

end big_bottles_count_l3211_321144


namespace mass_o2_for_combustion_l3211_321157

/-- The mass of O2 gas required for complete combustion of C8H18 -/
theorem mass_o2_for_combustion (moles_c8h18 : ℝ) (molar_mass_o2 : ℝ) : 
  moles_c8h18 = 7 → molar_mass_o2 = 32 → 
  (25 / 2 * moles_c8h18 * molar_mass_o2 : ℝ) = 2800 := by
  sorry

#check mass_o2_for_combustion

end mass_o2_for_combustion_l3211_321157


namespace count_divisible_numbers_l3211_321192

theorem count_divisible_numbers : 
  (Finset.filter (fun n : ℕ => 
    n ≤ 10^10 ∧ 
    (∀ k : ℕ, k ∈ Finset.range 10 → k.succ ∣ n)
  ) (Finset.range (10^10 + 1))).card = 3968253 := by
  sorry

end count_divisible_numbers_l3211_321192


namespace game_probability_difference_l3211_321127

def p_heads : ℚ := 3/4
def p_tails : ℚ := 1/4

def p_win_game_c : ℚ := p_heads^4 + p_tails^4

def p_win_game_d : ℚ := p_heads^4 * p_tails + p_tails^4 * p_heads

theorem game_probability_difference :
  p_win_game_c - p_win_game_d = 61/256 := by sorry

end game_probability_difference_l3211_321127


namespace equal_star_set_eq_four_lines_l3211_321100

-- Define the operation ⋆
def star (a b : ℝ) : ℝ := a^3 * b - a * b^3

-- Define the set of points (x, y) where x ⋆ y = y ⋆ x
def equal_star_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | star p.1 p.2 = star p.2 p.1}

-- Define the four lines
def four_lines : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = 0 ∨ p.2 = 0 ∨ p.1 = p.2 ∨ p.1 + p.2 = 0}

-- Theorem stating the equivalence of the two sets
theorem equal_star_set_eq_four_lines :
  equal_star_set = four_lines := by sorry

end equal_star_set_eq_four_lines_l3211_321100


namespace first_tier_tax_percentage_l3211_321107

theorem first_tier_tax_percentage
  (first_tier_limit : ℝ)
  (second_tier_rate : ℝ)
  (car_price : ℝ)
  (total_tax : ℝ)
  (h1 : first_tier_limit = 11000)
  (h2 : second_tier_rate = 0.09)
  (h3 : car_price = 18000)
  (h4 : total_tax = 1950) :
  ∃ first_tier_rate : ℝ,
    first_tier_rate = 0.12 ∧
    total_tax = first_tier_rate * first_tier_limit +
                second_tier_rate * (car_price - first_tier_limit) := by
  sorry

end first_tier_tax_percentage_l3211_321107


namespace quadratic_root_values_l3211_321145

/-- Given that 1 - i is a root of a real-coefficient quadratic equation x² + ax + b = 0,
    prove that a = -2 and b = 2 -/
theorem quadratic_root_values (a b : ℝ) : 
  (Complex.I : ℂ)^2 = -1 →
  (1 - Complex.I)^2 + a*(1 - Complex.I) + b = 0 →
  a = -2 ∧ b = 2 := by
sorry

end quadratic_root_values_l3211_321145


namespace equation_graph_l3211_321156

/-- The set of points (x, y) satisfying (x+y)³ = x³ + y³ is equivalent to the union of three lines -/
theorem equation_graph (x y : ℝ) :
  (x + y)^3 = x^3 + y^3 ↔ (x = 0 ∨ y = 0 ∨ x + y = 0) :=
by sorry

end equation_graph_l3211_321156


namespace rogers_dimes_l3211_321134

/-- The number of dimes Roger initially collected -/
def initial_dimes : ℕ := 15

/-- The number of pennies Roger collected -/
def pennies : ℕ := 42

/-- The number of nickels Roger collected -/
def nickels : ℕ := 36

/-- The number of coins Roger had left after donating -/
def coins_left : ℕ := 27

/-- The number of coins Roger donated -/
def coins_donated : ℕ := 66

theorem rogers_dimes :
  initial_dimes = 15 ∧
  pennies + nickels + initial_dimes = coins_left + coins_donated :=
by sorry

end rogers_dimes_l3211_321134


namespace ceiling_floor_difference_l3211_321169

theorem ceiling_floor_difference : 
  ⌈(12 : ℚ) / 7 * (-29 : ℚ) / 3⌉ - ⌊(12 : ℚ) / 7 * ⌊(-29 : ℚ) / 3⌋⌋ = 2 := by
  sorry

end ceiling_floor_difference_l3211_321169


namespace line_tangent_to_circle_l3211_321187

/-- The line √3x - y + m = 0 is tangent to the circle x^2 + y^2 - 2y = 0 if and only if m = -1 or m = 3 -/
theorem line_tangent_to_circle (m : ℝ) : 
  (∀ x y : ℝ, (Real.sqrt 3 * x - y + m = 0) → (x^2 + y^2 - 2*y = 0) → 
   (∀ ε > 0, ∃ x' y' : ℝ, x' ≠ x ∨ y' ≠ y ∧ 
    (Real.sqrt 3 * x' - y' + m = 0) ∧ 
    (x'^2 + y'^2 - 2*y' ≠ 0) ∧
    ((x' - x)^2 + (y' - y)^2 < ε^2))) ↔ 
  (m = -1 ∨ m = 3) :=
sorry

end line_tangent_to_circle_l3211_321187


namespace number_grid_solution_l3211_321133

theorem number_grid_solution : 
  ∃ (a b c d : ℕ) (s : Finset ℕ),
    s = {1, 2, 3, 4, 5, 6, 7, 8} ∧
    a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ d ∈ s ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    a * b = c ∧
    c / b = d ∧
    a = d :=
by sorry

end number_grid_solution_l3211_321133


namespace candy_theorem_l3211_321191

def candy_problem (bars_per_friend : ℕ) (num_friends : ℕ) (spare_bars : ℕ) : ℕ :=
  bars_per_friend * num_friends + spare_bars

theorem candy_theorem (bars_per_friend : ℕ) (num_friends : ℕ) (spare_bars : ℕ) :
  candy_problem bars_per_friend num_friends spare_bars =
  bars_per_friend * num_friends + spare_bars :=
by
  sorry

#eval candy_problem 2 7 10

end candy_theorem_l3211_321191


namespace wire_cutting_l3211_321135

/-- Given a wire of length 28 cm, if one piece is 2.00001/5 times the length of the other,
    then the shorter piece is 20 cm long. -/
theorem wire_cutting (total_length : ℝ) (ratio : ℝ) (shorter_length : ℝ) : 
  total_length = 28 →
  ratio = 2.00001 / 5 →
  shorter_length + ratio * shorter_length = total_length →
  shorter_length = 20 := by
sorry

end wire_cutting_l3211_321135


namespace mans_speed_against_current_l3211_321176

/-- Given a man's speed with the current and the speed of the current, 
    calculate the man's speed against the current. -/
theorem mans_speed_against_current 
  (speed_with_current : ℝ) 
  (current_speed : ℝ) 
  (h1 : speed_with_current = 21) 
  (h2 : current_speed = 2.5) : 
  speed_with_current - 2 * current_speed = 16 :=
by sorry

end mans_speed_against_current_l3211_321176


namespace part_one_part_two_l3211_321182

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |a * x + 1|
def g (x : ℝ) : ℝ := |x + 1| + 2

-- Part I
theorem part_one :
  {x : ℝ | f (1/2) x < 2} = {x : ℝ | 0 < x ∧ x < 4/3} := by sorry

-- Part II
theorem part_two :
  (∀ x ∈ Set.Ioo 0 1, f a x ≤ g x) → -5 ≤ a ∧ a ≤ 3 := by sorry

end part_one_part_two_l3211_321182


namespace opposite_of_negative_two_l3211_321165

def opposite (x : ℝ) : ℝ := -x

theorem opposite_of_negative_two :
  opposite (-2) = 2 := by
  sorry

end opposite_of_negative_two_l3211_321165


namespace perpendicular_lines_condition_l3211_321124

/-- Two lines in the form A₁x + B₁y + C₁ = 0 and A₂x + B₂y + C₂ = 0 are perpendicular -/
def are_perpendicular (A₁ B₁ C₁ A₂ B₂ C₂ : ℝ) : Prop :=
  A₁ * A₂ + B₁ * B₂ = 0

/-- The theorem stating the necessary and sufficient condition for two lines to be perpendicular -/
theorem perpendicular_lines_condition
  (A₁ B₁ C₁ A₂ B₂ C₂ : ℝ) :
  (∃ x y : ℝ, A₁ * x + B₁ * y + C₁ = 0 ∧ A₂ * x + B₂ * y + C₂ = 0) →
  (are_perpendicular A₁ B₁ C₁ A₂ B₂ C₂ ↔ 
   ∀ x₁ y₁ x₂ y₂ : ℝ, 
   A₁ * x₁ + B₁ * y₁ + C₁ = 0 ∧ 
   A₁ * x₂ + B₁ * y₂ + C₁ = 0 ∧ 
   A₂ * x₁ + B₂ * y₁ + C₂ = 0 ∧ 
   A₂ * x₂ + B₂ * y₂ + C₂ = 0 →
   (x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁) ≠ 0 →
   ((x₂ - x₁) * (y₂ - y₁) = 0)) :=
by sorry

end perpendicular_lines_condition_l3211_321124


namespace price_difference_proof_l3211_321173

def shop_x_price : ℚ := 1.25
def shop_y_price : ℚ := 2.75
def num_copies : ℕ := 40

theorem price_difference_proof :
  (shop_y_price * num_copies) - (shop_x_price * num_copies) = 60 := by
  sorry

end price_difference_proof_l3211_321173


namespace factorization_implies_sum_l3211_321137

theorem factorization_implies_sum (C D : ℤ) :
  (∀ y : ℝ, 6 * y^2 - 31 * y + 35 = (C * y - 5) * (D * y - 7)) →
  C * D + C = 9 := by
  sorry

end factorization_implies_sum_l3211_321137


namespace digit_101_of_7_12_l3211_321110

/-- The decimal representation of 7/12 has a repeating sequence of 4 digits. -/
def decimal_7_12_period : ℕ := 4

/-- The first digit of the repeating sequence in the decimal representation of 7/12. -/
def first_digit_7_12 : ℕ := 5

/-- The 101st digit after the decimal point in the decimal representation of 7/12 is 5. -/
theorem digit_101_of_7_12 : 
  (101 % decimal_7_12_period = 1) → 
  (Nat.digitChar (first_digit_7_12) = '5') := by
sorry

end digit_101_of_7_12_l3211_321110


namespace defective_and_shipped_percentage_l3211_321120

/-- The percentage of defective units produced -/
def defective_rate : ℝ := 0.08

/-- The percentage of defective units shipped -/
def shipped_rate : ℝ := 0.05

/-- The percentage of units that are both defective and shipped -/
def defective_and_shipped_rate : ℝ := defective_rate * shipped_rate

theorem defective_and_shipped_percentage :
  defective_and_shipped_rate = 0.004 := by sorry

end defective_and_shipped_percentage_l3211_321120


namespace complex_equation_solution_l3211_321195

theorem complex_equation_solution (z : ℂ) : (Complex.I * z = 4 + 3 * Complex.I) → z = 3 - 4 * Complex.I := by
  sorry

end complex_equation_solution_l3211_321195


namespace single_point_condition_l3211_321174

/-- The equation represents a single point if and only if d equals 125/4 -/
theorem single_point_condition (d : ℝ) : 
  (∃! p : ℝ × ℝ, 3 * p.1^2 + 2 * p.2^2 + 9 * p.1 - 14 * p.2 + d = 0) ↔ 
  d = 125 / 4 := by
  sorry

end single_point_condition_l3211_321174


namespace parallel_vectors_k_value_l3211_321131

/-- Given two vectors in R³ that satisfy certain conditions, prove that k = -3/2 --/
theorem parallel_vectors_k_value (a b : ℝ × ℝ × ℝ) (k : ℝ) :
  a = (1, 2, 1) →
  b = (1, 2, 2) →
  ∃ (t : ℝ), t ≠ 0 ∧ (k • a + b) = t • (a - 2 • b) →
  k = -3/2 :=
by sorry

end parallel_vectors_k_value_l3211_321131


namespace system_solution_l3211_321168

theorem system_solution (x y a b : ℝ) : 
  x = 1 ∧ 
  y = -2 ∧ 
  3 * x + 2 * y = a ∧ 
  b * x - y = 5 → 
  b - a = 4 := by
sorry

end system_solution_l3211_321168


namespace thirty_switch_network_connections_l3211_321141

/-- Represents a network of switches with their connections. -/
structure SwitchNetwork where
  num_switches : ℕ
  connections_per_switch : ℕ
  no_multiple_connections : Bool

/-- Calculates the total number of connections in the network. -/
def total_connections (network : SwitchNetwork) : ℕ :=
  (network.num_switches * network.connections_per_switch) / 2

/-- Theorem stating that a network of 30 switches, each connected to 4 others,
    has 60 total connections. -/
theorem thirty_switch_network_connections :
  let network := SwitchNetwork.mk 30 4 true
  total_connections network = 60 := by
  sorry

end thirty_switch_network_connections_l3211_321141


namespace cube_set_closed_under_multiplication_cube_set_not_closed_under_addition_cube_set_not_closed_under_division_cube_set_not_closed_under_squaring_l3211_321198

def is_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

def cube_set : Set ℕ := {n : ℕ | is_cube n ∧ n > 0}

theorem cube_set_closed_under_multiplication (a b : ℕ) (ha : a ∈ cube_set) (hb : b ∈ cube_set) :
  (a * b) ∈ cube_set :=
sorry

theorem cube_set_not_closed_under_addition :
  ∃ a b : ℕ, a ∈ cube_set ∧ b ∈ cube_set ∧ (a + b) ∉ cube_set :=
sorry

theorem cube_set_not_closed_under_division :
  ∃ a b : ℕ, a ∈ cube_set ∧ b ∈ cube_set ∧ b ≠ 0 ∧ (a / b) ∉ cube_set :=
sorry

theorem cube_set_not_closed_under_squaring :
  ∃ a : ℕ, a ∈ cube_set ∧ (a^2) ∉ cube_set :=
sorry

end cube_set_closed_under_multiplication_cube_set_not_closed_under_addition_cube_set_not_closed_under_division_cube_set_not_closed_under_squaring_l3211_321198


namespace population_reaches_capacity_years_to_max_capacity_l3211_321122

/-- The maximum capacity of the realm in people -/
def max_capacity : ℕ := 35000 / 2

/-- The initial population in 2023 -/
def initial_population : ℕ := 500

/-- The population growth factor every 20 years -/
def growth_factor : ℕ := 2

/-- The population after n 20-year periods -/
def population (n : ℕ) : ℕ := initial_population * growth_factor ^ n

/-- The number of 20-year periods after which the population reaches or exceeds the maximum capacity -/
def periods_to_max_capacity : ℕ := 5

theorem population_reaches_capacity :
  population periods_to_max_capacity ≥ max_capacity ∧
  population (periods_to_max_capacity - 1) < max_capacity :=
sorry

theorem years_to_max_capacity : periods_to_max_capacity * 20 = 100 :=
sorry

end population_reaches_capacity_years_to_max_capacity_l3211_321122


namespace energy_drink_cost_l3211_321185

/-- The cost of an energy drink bottle given the sales and purchases of a basketball team. -/
theorem energy_drink_cost (cupcakes : ℕ) (cupcake_price : ℚ) 
  (cookies : ℕ) (cookie_price : ℚ)
  (basketballs : ℕ) (basketball_price : ℚ)
  (energy_drinks : ℕ) :
  cupcakes = 50 →
  cupcake_price = 2 →
  cookies = 40 →
  cookie_price = 1/2 →
  basketballs = 2 →
  basketball_price = 40 →
  energy_drinks = 20 →
  (cupcakes : ℚ) * cupcake_price + (cookies : ℚ) * cookie_price 
    - (basketballs : ℚ) * basketball_price = (energy_drinks : ℚ) * 2 :=
by sorry

end energy_drink_cost_l3211_321185


namespace parallel_vectors_t_value_l3211_321153

def vector_a : Fin 2 → ℝ := ![(-1), 3]
def vector_b (t : ℝ) : Fin 2 → ℝ := ![1, t]

def parallel (u v : Fin 2 → ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ ∀ i, u i = k * v i

theorem parallel_vectors_t_value :
  ∀ t : ℝ, parallel vector_a (vector_b t) → t = -3 :=
by sorry

end parallel_vectors_t_value_l3211_321153


namespace password_from_polynomial_factorization_password_for_given_values_l3211_321147

/-- Generates a password from the factors of x^3 - xy^2 --/
def generate_password (x y : ℕ) : ℕ :=
  x * 10000 + (x + y) * 100 + (x - y)

/-- The polynomial x^3 - xy^2 factors as x(x-y)(x+y) --/
theorem password_from_polynomial_factorization (x y : ℕ) :
  x^3 - x*y^2 = x * (x - y) * (x + y) :=
sorry

/-- The password generated from x^3 - xy^2 with x=18 and y=5 is 181323 --/
theorem password_for_given_values :
  generate_password 18 5 = 181323 :=
sorry

end password_from_polynomial_factorization_password_for_given_values_l3211_321147


namespace expansion_equality_l3211_321151

theorem expansion_equality (m n : ℝ) : (m + n) * (m - 2*n) = m^2 - m*n - 2*n^2 := by
  sorry

end expansion_equality_l3211_321151


namespace paper_clip_cost_l3211_321117

/-- The cost of Eldora's purchase -/
def eldora_cost : ℝ := 55.40

/-- The cost of Finn's purchase -/
def finn_cost : ℝ := 61.70

/-- The number of paper clip boxes Eldora bought -/
def eldora_clips : ℕ := 15

/-- The number of index card packages Eldora bought -/
def eldora_cards : ℕ := 7

/-- The number of paper clip boxes Finn bought -/
def finn_clips : ℕ := 12

/-- The number of index card packages Finn bought -/
def finn_cards : ℕ := 10

/-- The cost of one box of paper clips -/
noncomputable def clip_cost : ℝ := 1.835

theorem paper_clip_cost : 
  ∃ (card_cost : ℝ), 
    (eldora_clips : ℝ) * clip_cost + (eldora_cards : ℝ) * card_cost = eldora_cost ∧ 
    (finn_clips : ℝ) * clip_cost + (finn_cards : ℝ) * card_cost = finn_cost :=
by sorry

end paper_clip_cost_l3211_321117


namespace jump_data_mode_l3211_321160

def jump_data : List Nat := [160, 163, 160, 157, 160]

def mode (l : List Nat) : Nat :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem jump_data_mode :
  mode jump_data = 160 := by
  sorry

end jump_data_mode_l3211_321160


namespace min_cubes_in_prism_l3211_321170

/-- Given a rectangular prism built with N identical 1-cm cubes,
    where 420 cubes are hidden from a viewpoint showing three faces,
    the minimum possible value of N is 630. -/
theorem min_cubes_in_prism (N : ℕ) (l m n : ℕ) : 
  (l - 1) * (m - 1) * (n - 1) = 420 →
  N = l * m * n →
  (∀ l' m' n' : ℕ, (l' - 1) * (m' - 1) * (n' - 1) = 420 → l' * m' * n' ≥ N) →
  N = 630 := by
  sorry

end min_cubes_in_prism_l3211_321170


namespace product_of_roots_l3211_321111

theorem product_of_roots (x : ℝ) : 
  (x^2 + 2*x - 35 = 0) → 
  ∃ y : ℝ, (y^2 + 2*y - 35 = 0) ∧ (x * y = -35) :=
by sorry

end product_of_roots_l3211_321111


namespace equation_solution_l3211_321167

theorem equation_solution (x : ℝ) : 
  (1 / (Real.sqrt x + Real.sqrt (x - 2)) + 1 / (Real.sqrt (x + 2) + Real.sqrt x) = 1 / 4) → 
  x = 257 / 16 := by
sorry

end equation_solution_l3211_321167


namespace geometric_sequence_seventh_term_l3211_321103

theorem geometric_sequence_seventh_term (x : ℝ) (b : ℕ → ℝ) 
  (h1 : b 1 = Real.sin x ^ 2)
  (h2 : b 2 = Real.sin x * Real.cos x)
  (h3 : b 3 = (Real.cos x ^ 2) / (Real.sin x))
  (h_geom : ∀ n : ℕ, n ≥ 1 → b (n + 1) = (b 2 / b 1) * b n) :
  b 7 = Real.cos x + Real.sin x :=
sorry

end geometric_sequence_seventh_term_l3211_321103


namespace sin_cos_sum_2023_17_l3211_321112

theorem sin_cos_sum_2023_17 :
  Real.sin (2023 * π / 180) * Real.cos (17 * π / 180) +
  Real.cos (2023 * π / 180) * Real.sin (17 * π / 180) =
  -Real.sqrt 3 / 2 := by
  sorry

end sin_cos_sum_2023_17_l3211_321112


namespace min_faces_two_dice_l3211_321164

theorem min_faces_two_dice (a b : ℕ) : 
  a ≥ 8 → b ≥ 8 →  -- Both dice have at least 8 faces
  (∀ i j, 1 ≤ i ∧ i ≤ a ∧ 1 ≤ j ∧ j ≤ b) →  -- Each face has a distinct integer from 1 to the number of faces
  (Nat.card {(i, j) | 1 ≤ i ∧ i ≤ a ∧ 1 ≤ j ∧ j ≤ b ∧ i + j = 9} : ℚ) / (a * b : ℚ) = 
    (2/3) * ((Nat.card {(i, j) | 1 ≤ i ∧ i ≤ a ∧ 1 ≤ j ∧ j ≤ b ∧ i + j = 11} : ℚ) / (a * b : ℚ)) →  -- Probability condition for sum of 9 and 11
  (Nat.card {(i, j) | 1 ≤ i ∧ i ≤ a ∧ 1 ≤ j ∧ j ≤ b ∧ i + j = 14} : ℚ) / (a * b : ℚ) = 1/9 →  -- Probability condition for sum of 14
  a + b ≥ 22 ∧ ∀ c d, c ≥ 8 → d ≥ 8 → 
    (∀ i j, 1 ≤ i ∧ i ≤ c ∧ 1 ≤ j ∧ j ≤ d) →
    (Nat.card {(i, j) | 1 ≤ i ∧ i ≤ c ∧ 1 ≤ j ∧ j ≤ d ∧ i + j = 9} : ℚ) / (c * d : ℚ) = 
      (2/3) * ((Nat.card {(i, j) | 1 ≤ i ∧ i ≤ c ∧ 1 ≤ j ∧ j ≤ d ∧ i + j = 11} : ℚ) / (c * d : ℚ)) →
    (Nat.card {(i, j) | 1 ≤ i ∧ i ≤ c ∧ 1 ≤ j ∧ j ≤ d ∧ i + j = 14} : ℚ) / (c * d : ℚ) = 1/9 →
    c + d ≥ 22 :=
by sorry

end min_faces_two_dice_l3211_321164


namespace computer_price_proof_l3211_321163

/-- The original price of the computer in yuan -/
def original_price : ℝ := 5000

/-- The installment price of the computer -/
def installment_price (price : ℝ) : ℝ := 1.04 * price

/-- The cash price of the computer -/
def cash_price (price : ℝ) : ℝ := 0.9 * price

/-- Theorem stating that the original price satisfies the given conditions -/
theorem computer_price_proof : 
  installment_price original_price - cash_price original_price = 700 := by
  sorry


end computer_price_proof_l3211_321163


namespace savings_fraction_is_5_17_l3211_321186

/-- Represents the worker's savings scenario -/
structure WorkerSavings where
  monthly_pay : ℝ
  savings_fraction : ℝ
  savings_fraction_constant : Prop
  monthly_pay_constant : Prop
  all_savings_from_pay : Prop
  total_savings_eq_5times_unsaved : Prop

/-- Theorem stating that the savings fraction is 5/17 -/
theorem savings_fraction_is_5_17 (w : WorkerSavings) : w.savings_fraction = 5 / 17 :=
by sorry

end savings_fraction_is_5_17_l3211_321186


namespace sector_area_l3211_321199

/-- Given a sector with radius R and perimeter 4R, its area is R^2 -/
theorem sector_area (R : ℝ) (R_pos : R > 0) : 
  let perimeter := 4 * R
  let arc_length := perimeter - 2 * R
  let area := (1 / 2) * R * arc_length
  area = R^2 := by sorry

end sector_area_l3211_321199


namespace real_part_z_2017_l3211_321104

def z : ℂ := 1 + Complex.I

theorem real_part_z_2017 : (z^2017).re = 2^1008 := by sorry

end real_part_z_2017_l3211_321104


namespace M_intersect_N_eq_zero_one_l3211_321114

-- Define set M
def M : Set ℝ := {x | x^2 = x}

-- Define set N
def N : Set ℝ := {-1, 0, 1}

-- Theorem statement
theorem M_intersect_N_eq_zero_one : M ∩ N = {0, 1} := by sorry

end M_intersect_N_eq_zero_one_l3211_321114


namespace no_solutions_to_equation_l3211_321125

theorem no_solutions_to_equation :
  ¬ ∃ x : ℝ, (2 * x^2 - 10 * x) / (x^2 - 5 * x) = x - 3 :=
by sorry

end no_solutions_to_equation_l3211_321125


namespace correct_amount_to_return_l3211_321142

/-- Calculates the amount to be returned in rubles given an initial deposit in USD and an exchange rate. -/
def amount_to_return (initial_deposit : ℝ) (exchange_rate : ℝ) : ℝ :=
  initial_deposit * exchange_rate

/-- Theorem stating that given the specific initial deposit and exchange rate, the amount to be returned is 581,500 rubles. -/
theorem correct_amount_to_return :
  amount_to_return 10000 58.15 = 581500 := by
  sorry

end correct_amount_to_return_l3211_321142


namespace course_assessment_probabilities_l3211_321119

/-- Represents a student in the course -/
inductive Student := | A | B | C

/-- Represents the type of assessment -/
inductive AssessmentType := | Theory | Experimental

/-- The probability of a student passing a specific assessment type -/
def passProbability (s : Student) (t : AssessmentType) : ℝ :=
  match s, t with
  | Student.A, AssessmentType.Theory => 0.9
  | Student.B, AssessmentType.Theory => 0.8
  | Student.C, AssessmentType.Theory => 0.7
  | Student.A, AssessmentType.Experimental => 0.8
  | Student.B, AssessmentType.Experimental => 0.7
  | Student.C, AssessmentType.Experimental => 0.9

/-- The probability of at least two students passing the theory assessment -/
def atLeastTwoPassTheory : ℝ := sorry

/-- The probability of all three students passing both assessments -/
def allPassBoth : ℝ := sorry

theorem course_assessment_probabilities :
  (atLeastTwoPassTheory = 0.902) ∧ (allPassBoth = 0.254) := by sorry

end course_assessment_probabilities_l3211_321119


namespace purely_imaginary_z_l3211_321121

theorem purely_imaginary_z (z : ℂ) : 
  (∃ b : ℝ, z = b * Complex.I) →  -- z is purely imaginary
  (∃ r : ℝ, (z + 2) / (1 + Complex.I) = r) →  -- (z+2)/(1+i) is real
  z = -2 * Complex.I :=  -- z = -2i
by sorry

end purely_imaginary_z_l3211_321121


namespace negative_eighth_power_2009_times_eight_power_2009_l3211_321146

theorem negative_eighth_power_2009_times_eight_power_2009 :
  (-0.125)^2009 * 8^2009 = -1 := by
  sorry

end negative_eighth_power_2009_times_eight_power_2009_l3211_321146


namespace ratio_of_60_to_12_l3211_321139

theorem ratio_of_60_to_12 : 
  let a := 60
  let b := 12
  (a : ℚ) / b = 5 / 1 := by sorry

end ratio_of_60_to_12_l3211_321139


namespace impossible_arrangement_l3211_321161

/-- A table is a function from pairs of indices to natural numbers -/
def Table := Fin 10 → Fin 10 → ℕ

/-- Predicate to check if two cells are adjacent in the table -/
def adjacent (i j k l : Fin 10) : Prop :=
  (i = k ∧ j.val + 1 = l.val) ∨ (i = k ∧ l.val + 1 = j.val) ∨
  (j = l ∧ i.val + 1 = k.val) ∧ (j = l ∧ k.val + 1 = i.val)

/-- Predicate to check if a quadratic equation has two integer roots -/
def has_two_int_roots (a b : ℕ) : Prop :=
  ∃ x y : ℤ, x ≠ y ∧ x^2 - a*x + b = 0 ∧ y^2 - a*y + b = 0

theorem impossible_arrangement : ¬∃ (t : Table),
  (∀ i j : Fin 10, 51 ≤ t i j ∧ t i j ≤ 150) ∧
  (∀ i j k l : Fin 10, adjacent i j k l →
    has_two_int_roots (t i j) (t k l) ∨ has_two_int_roots (t k l) (t i j)) :=
sorry

end impossible_arrangement_l3211_321161


namespace find_divisor_l3211_321171

theorem find_divisor : 
  ∃ d : ℕ, d > 0 ∧ 136 = 9 * d + 1 :=
by
  -- The proof goes here
  sorry

end find_divisor_l3211_321171


namespace table_tennis_probabilities_l3211_321129

/-- Represents the probability of player A winning a serve -/
def p_win : ℝ := 0.6

/-- Probability that player A scores i points in two consecutive serves -/
def p_score (i : Fin 3) : ℝ :=
  match i with
  | 0 => (1 - p_win)^2
  | 1 => 2 * p_win * (1 - p_win)
  | 2 => p_win^2

/-- Theorem stating the probabilities of specific score situations in a table tennis game -/
theorem table_tennis_probabilities :
  let p_b_leads := p_score 0 * p_win + p_score 1 * (1 - p_win)
  let p_a_leads := p_score 1 * p_score 2 + p_score 2 * p_score 1 + p_score 2 * p_score 2
  (p_b_leads = 0.352) ∧ (p_a_leads = 0.3072) := by
  sorry


end table_tennis_probabilities_l3211_321129


namespace no_power_ending_222_l3211_321116

theorem no_power_ending_222 :
  ¬ ∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ ∃ (n : ℕ), x^y = 1000*n + 222 :=
sorry

end no_power_ending_222_l3211_321116


namespace clique_of_nine_l3211_321178

/-- Represents the relationship of knowing each other in a group of people -/
def Knows (n : ℕ) := Fin n → Fin n → Prop

/-- States that the 'Knows' relation is symmetric -/
def SymmetricKnows {n : ℕ} (knows : Knows n) :=
  ∀ i j : Fin n, knows i j → knows j i

/-- States that among any 3 people, at least two know each other -/
def AtLeastTwoKnowEachOther {n : ℕ} (knows : Knows n) :=
  ∀ i j k : Fin n, i ≠ j ∧ j ≠ k ∧ i ≠ k →
    knows i j ∨ knows j k ∨ knows i k

/-- Defines a clique of size 4 where everyone knows each other -/
def HasCliqueFour {n : ℕ} (knows : Knows n) :=
  ∃ i j k l : Fin n, i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ i ≠ k ∧ i ≠ l ∧ j ≠ l ∧
    knows i j ∧ knows i k ∧ knows i l ∧
    knows j k ∧ knows j l ∧
    knows k l

theorem clique_of_nine (knows : Knows 9) 
  (symm : SymmetricKnows knows) 
  (atleast_two : AtLeastTwoKnowEachOther knows) : 
  HasCliqueFour knows := by
  sorry

end clique_of_nine_l3211_321178


namespace fisherman_catch_l3211_321108

/-- The number of bass caught by the fisherman -/
def bass : ℕ := 32

/-- The number of trout caught by the fisherman -/
def trout : ℕ := bass / 4

/-- The number of bluegill caught by the fisherman -/
def bluegill : ℕ := 2 * bass

/-- The total number of fish caught by the fisherman -/
def total_fish : ℕ := 104

theorem fisherman_catch :
  bass + trout + bluegill = total_fish ∧
  trout = bass / 4 ∧
  bluegill = 2 * bass :=
sorry

end fisherman_catch_l3211_321108


namespace exact_three_correct_deliveries_probability_l3211_321126

def num_packages : ℕ := 5

def num_correct_deliveries : ℕ := 3

def total_permutations : ℕ := num_packages.factorial

def choose (n k : ℕ) : ℕ := (n.factorial) / (k.factorial * (n - k).factorial)

def num_ways_correct_deliveries : ℕ := choose num_packages num_correct_deliveries

def num_derangements_remaining : ℕ := 1

theorem exact_three_correct_deliveries_probability :
  (num_ways_correct_deliveries * num_derangements_remaining : ℚ) / total_permutations = 1 / 12 := by
  sorry

end exact_three_correct_deliveries_probability_l3211_321126


namespace equation_solutions_l3211_321193

theorem equation_solutions (x y n : ℕ+) : 
  (((x : ℝ)^2 + (y : ℝ)^2)^(n : ℝ) = ((x * y : ℝ)^2016)) ↔ 
  n ∈ ({1344, 1728, 1792, 1920, 1984} : Set ℕ+) :=
sorry

end equation_solutions_l3211_321193


namespace problem_statement_l3211_321166

theorem problem_statement (x y z a b c : ℝ) 
  (h1 : x/a + y/b + z/c = 5)
  (h2 : a/x + b/y + c/z = 6) :
  x^2/a^2 + y^2/b^2 + z^2/c^2 = 13 := by sorry

end problem_statement_l3211_321166


namespace sixth_term_is_three_l3211_321162

/-- An arithmetic sequence with 10 terms -/
def ArithmeticSequence := Fin 10 → ℝ

/-- The property that the sequence is arithmetic -/
def is_arithmetic (a : ArithmeticSequence) : Prop :=
  ∃ d : ℝ, ∀ i j : Fin 10, a j - a i = d * (j - i)

/-- The sum of even-numbered terms is 15 -/
def sum_even_terms_is_15 (a : ArithmeticSequence) : Prop :=
  a 1 + a 3 + a 5 + a 7 + a 9 = 15

theorem sixth_term_is_three
  (a : ArithmeticSequence)
  (h_arith : is_arithmetic a)
  (h_sum : sum_even_terms_is_15 a) :
  a 5 = 3 :=
sorry

end sixth_term_is_three_l3211_321162


namespace isosceles_obtuse_triangle_smallest_angle_l3211_321196

/-- 
Theorem: In an isosceles, obtuse triangle where one angle is 60% larger than a right angle, 
each of the two smallest angles measures 18°.
-/
theorem isosceles_obtuse_triangle_smallest_angle :
  ∀ (a b c : ℝ), 
  -- The triangle is isosceles
  a = b →
  -- The triangle is obtuse (one angle > 90°)
  c > 90 →
  -- One angle (c) is 60% larger than a right angle
  c = 90 * 1.6 →
  -- The sum of angles in a triangle is 180°
  a + b + c = 180 →
  -- Each of the two smallest angles (a and b) measures 18°
  a = 18 ∧ b = 18 := by
sorry

end isosceles_obtuse_triangle_smallest_angle_l3211_321196


namespace factoring_expression_l3211_321180

theorem factoring_expression (y : ℝ) : 3*y*(2*y+5) + 4*(2*y+5) = (3*y+4)*(2*y+5) := by
  sorry

end factoring_expression_l3211_321180


namespace coin_landing_probability_l3211_321189

/-- Represents the specially colored square -/
structure ColoredSquare where
  side_length : ℝ
  triangle_leg : ℝ
  diamond_side : ℝ

/-- Represents the circular coin -/
structure Coin where
  diameter : ℝ

/-- Calculates the probability of the coin landing on a black region -/
def black_region_probability (square : ColoredSquare) (coin : Coin) : ℝ :=
  sorry

/-- The main theorem to prove -/
theorem coin_landing_probability 
  (square : ColoredSquare)
  (coin : Coin)
  (h_square_side : square.side_length = 8)
  (h_triangle_leg : square.triangle_leg = 2)
  (h_diamond_side : square.diamond_side = 2 * Real.sqrt 2)
  (h_coin_diameter : coin.diameter = 1) :
  ∃ (a b : ℕ), 
    black_region_probability square coin = 1 / 196 * (a + b * Real.sqrt 2 + Real.pi) ∧
    a + b = 68 :=
  sorry

end coin_landing_probability_l3211_321189


namespace quadratic_solution_sum_l3211_321177

theorem quadratic_solution_sum (c d : ℝ) : 
  (∀ x, x^2 - 6*x + 11 = 23 ↔ x = c ∨ x = d) →
  c ≥ d →
  3*c + 2*d = 15 + Real.sqrt 21 := by
sorry

end quadratic_solution_sum_l3211_321177


namespace inequality_solution_set_l3211_321150

theorem inequality_solution_set (x : ℝ) :
  (x ∈ {x : ℝ | -6 * x^2 + 2 < x}) ↔ (x < -2/3 ∨ x > 1/2) :=
sorry

end inequality_solution_set_l3211_321150


namespace room_length_l3211_321159

/-- The length of a rectangular room with given width and area -/
theorem room_length (width : ℝ) (area : ℝ) (h1 : width = 20) (h2 : area = 80) :
  area / width = 4 := by sorry

end room_length_l3211_321159


namespace trajectory_equation_l3211_321152

theorem trajectory_equation (a b x y : ℝ) : 
  a^2 + b^2 = 100 →  -- Line segment length is 10
  x = a / 5 →        -- AM = 4MB implies x = a/(1+4)
  y = 4*b / 5 →      -- AM = 4MB implies y = 4b/(1+4)
  16*x^2 + y^2 = 64  -- Trajectory equation
:= by sorry

end trajectory_equation_l3211_321152


namespace characters_with_initial_D_l3211_321105

-- Define the total number of characters
def total_characters : ℕ := 60

-- Define the number of characters with initial A
def characters_A : ℕ := total_characters / 2

-- Define the number of characters with initial C
def characters_C : ℕ := characters_A / 2

-- Define the remaining characters (D and E)
def remaining_characters : ℕ := total_characters - characters_A - characters_C

-- Theorem stating the number of characters with initial D
theorem characters_with_initial_D : 
  ∃ (d e : ℕ), d = 2 * e ∧ d + e = remaining_characters ∧ d = 10 :=
sorry

end characters_with_initial_D_l3211_321105


namespace exactly_three_primes_probability_l3211_321172

-- Define a die as a type with 6 possible outcomes
def Die := Fin 6

-- Define a function to check if a number is prime (for a 6-sided die)
def isPrime (n : Die) : Bool :=
  n.val + 1 = 2 || n.val + 1 = 3 || n.val + 1 = 5

-- Define the probability of rolling a prime number on a single die
def probPrime : ℚ := 1/2

-- Define the number of dice
def numDice : ℕ := 6

-- Define the number of dice we want to show prime numbers
def targetPrimes : ℕ := 3

-- State the theorem
theorem exactly_three_primes_probability :
  (numDice.choose targetPrimes : ℚ) * probPrime^targetPrimes * (1 - probPrime)^(numDice - targetPrimes) = 5/16 := by
  sorry

end exactly_three_primes_probability_l3211_321172


namespace range_of_a_l3211_321138

-- Define the inequality function
def f (x a : ℝ) : ℝ := x^2 + (2-a)*x + 4-2*a

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (∀ x ≥ 2, f x a > 0) ↔ a < 3 :=
sorry

end range_of_a_l3211_321138


namespace servant_worked_nine_months_l3211_321106

/-- Represents the salary and work duration of a servant --/
structure ServantSalary where
  yearly_cash : ℕ  -- Yearly cash salary in Rupees
  turban_value : ℕ  -- Value of the turban in Rupees
  received_cash : ℕ  -- Cash received when leaving in Rupees
  months_worked : ℕ  -- Number of months worked

/-- Calculates the number of months a servant worked based on their salary structure --/
def calculate_months_worked (s : ServantSalary) : ℕ :=
  ((s.received_cash + s.turban_value) * 12) / (s.yearly_cash + s.turban_value)

/-- Theorem stating that under the given conditions, the servant worked for 9 months --/
theorem servant_worked_nine_months (s : ServantSalary) 
  (h1 : s.yearly_cash = 90)
  (h2 : s.turban_value = 90)
  (h3 : s.received_cash = 45) :
  calculate_months_worked s = 9 := by
  sorry

end servant_worked_nine_months_l3211_321106


namespace ellipse_eccentricity_l3211_321118

/-- The eccentricity of an ellipse with specific properties -/
theorem ellipse_eccentricity (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
  (h4 : b = (a + c) / 2) (h5 : b^2 = a^2 - c^2) : 
  let e := c / a
  0 < e ∧ e < 1 ∧ e = 3/5 := by sorry

end ellipse_eccentricity_l3211_321118


namespace absolute_value_equation_l3211_321175

theorem absolute_value_equation (x : ℝ) :
  |x - 25| + |x - 15| = |2*x - 40| → x = 20 := by
  sorry

end absolute_value_equation_l3211_321175


namespace common_root_quadratic_equations_l3211_321136

theorem common_root_quadratic_equations (a : ℝ) :
  (∃ x : ℝ, x^2 + a*x + 8 = 0 ∧ x^2 + x + a = 0) ↔ a = -6 := by
  sorry

end common_root_quadratic_equations_l3211_321136


namespace heloise_gave_ten_dogs_l3211_321140

/-- The number of dogs Heloise gave to Janet -/
def dogs_given_to_janet (total_pets : ℕ) (remaining_dogs : ℕ) : ℕ :=
  let dog_ratio := 10
  let cat_ratio := 17
  let total_ratio := dog_ratio + cat_ratio
  let pets_per_ratio := total_pets / total_ratio
  let original_dogs := dog_ratio * pets_per_ratio
  original_dogs - remaining_dogs

/-- Proof that Heloise gave 10 dogs to Janet -/
theorem heloise_gave_ten_dogs :
  dogs_given_to_janet 189 60 = 10 := by
  sorry

end heloise_gave_ten_dogs_l3211_321140


namespace function_monotonically_increasing_l3211_321102

/-- The function f(x) = x^2 - 2x + 8 is monotonically increasing on the interval (1, +∞) -/
theorem function_monotonically_increasing (x y : ℝ) : x > 1 → y > 1 → x < y →
  (x^2 - 2*x + 8) < (y^2 - 2*y + 8) := by sorry

end function_monotonically_increasing_l3211_321102


namespace cost_graph_two_segments_l3211_321179

/-- The cost function for pencils -/
def cost (n : ℕ) : ℚ :=
  if n ≤ 10 then 10 * n else 8 * n - 40

/-- The graph of the cost function consists of two connected linear segments -/
theorem cost_graph_two_segments :
  ∃ (a b : ℕ) (m₁ m₂ c₁ c₂ : ℚ),
    a < b ∧
    (∀ n, 1 ≤ n ∧ n ≤ a → cost n = m₁ * n + c₁) ∧
    (∀ n, b ≤ n ∧ n ≤ 20 → cost n = m₂ * n + c₂) ∧
    (m₁ * a + c₁ = m₂ * b + c₂) ∧
    m₁ ≠ m₂ :=
sorry

end cost_graph_two_segments_l3211_321179


namespace coefficient_a3_value_l3211_321149

theorem coefficient_a3_value (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, x^5 + 3*x^3 + 1 = a₀ + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + a₅*(x-1)^5) →
  a₃ = 13 := by
sorry

end coefficient_a3_value_l3211_321149


namespace parabola_one_intersection_l3211_321183

/-- A parabola that intersects the x-axis at exactly one point -/
def one_intersection_parabola (c : ℝ) : Prop :=
  ∃! x, x^2 + x + c = 0

/-- The theorem stating that the parabola y = x^2 + x + c intersects 
    the x-axis at exactly one point when c = 1/4 -/
theorem parabola_one_intersection :
  one_intersection_parabola (1/4 : ℝ) ∧ 
  ∀ c : ℝ, one_intersection_parabola c → c = 1/4 :=
sorry

end parabola_one_intersection_l3211_321183


namespace total_distance_swam_l3211_321181

/-- Represents the swimming styles -/
inductive SwimmingStyle
| Freestyle
| Butterfly

/-- Calculates the distance swam for a given style -/
def distance_swam (style : SwimmingStyle) (total_time : ℕ) : ℕ :=
  match style with
  | SwimmingStyle.Freestyle =>
    let cycle_time := 26  -- 20 minutes swimming + 6 minutes rest
    let cycles := total_time / cycle_time
    let distance_per_cycle := 500  -- 100 meters in 4 minutes, so 500 meters in 20 minutes
    cycles * distance_per_cycle
  | SwimmingStyle.Butterfly =>
    let cycle_time := 35  -- 30 minutes swimming + 5 minutes rest
    let cycles := total_time / cycle_time
    let distance_per_cycle := 429  -- 100 meters in 7 minutes, so approximately 429 meters in 30 minutes
    cycles * distance_per_cycle

theorem total_distance_swam :
  let freestyle_time := 90  -- 1 hour and 30 minutes in minutes
  let butterfly_time := 90  -- 1 hour and 30 minutes in minutes
  let freestyle_distance := distance_swam SwimmingStyle.Freestyle freestyle_time
  let butterfly_distance := distance_swam SwimmingStyle.Butterfly butterfly_time
  freestyle_distance + butterfly_distance = 2358 := by
  sorry


end total_distance_swam_l3211_321181


namespace no_real_solutions_for_log_equation_l3211_321143

theorem no_real_solutions_for_log_equation :
  ∀ (p q : ℝ), Real.log (p * q) = Real.log (p^2 + q^2 + 1) → False :=
sorry

end no_real_solutions_for_log_equation_l3211_321143


namespace smallest_square_partition_l3211_321109

theorem smallest_square_partition : ∃ (n : ℕ),
  n > 0 ∧
  (∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ a + b = 10 ∧ a ≥ 8 ∧ n^2 = a * 1^2 + b * 2^2) ∧
  (∀ (m : ℕ), m < n →
    ¬(∃ (c d : ℕ), c > 0 ∧ d > 0 ∧ c + d = 10 ∧ c ≥ 8 ∧ m^2 = c * 1^2 + d * 2^2)) ∧
  n = 4 :=
by sorry

end smallest_square_partition_l3211_321109


namespace kaeli_problems_per_day_l3211_321155

def marie_pascale_problems_per_day : ℕ := 4
def marie_pascale_total_problems : ℕ := 72
def kaeli_extra_problems : ℕ := 54

def days : ℕ := marie_pascale_total_problems / marie_pascale_problems_per_day

def kaeli_total_problems : ℕ := marie_pascale_total_problems + kaeli_extra_problems

theorem kaeli_problems_per_day : 
  kaeli_total_problems / days = 7 :=
sorry

end kaeli_problems_per_day_l3211_321155


namespace probability_r_successes_correct_l3211_321132

/-- The probability of exactly r successful shots by the time the nth shot is taken -/
def probability_r_successes (n r : ℕ) (p : ℝ) : ℝ :=
  Nat.choose (n - 1) (r - 1) * p ^ r * (1 - p) ^ (n - r)

/-- Theorem stating the probability of exactly r successful shots by the nth shot -/
theorem probability_r_successes_correct (n r : ℕ) (p : ℝ) 
    (h1 : 0 ≤ p) (h2 : p ≤ 1) (h3 : 1 ≤ r) (h4 : r ≤ n) : 
  probability_r_successes n r p = Nat.choose (n - 1) (r - 1) * p ^ r * (1 - p) ^ (n - r) :=
by sorry

end probability_r_successes_correct_l3211_321132


namespace gold_cube_side_length_l3211_321197

/-- Proves that a gold cube with given parameters has a side length of 6 cm -/
theorem gold_cube_side_length (L : ℝ) 
  (density : ℝ) (buy_price : ℝ) (sell_factor : ℝ) (profit : ℝ) :
  density = 19 →
  buy_price = 60 →
  sell_factor = 1.5 →
  profit = 123120 →
  profit = (sell_factor * buy_price * density * L^3) - (buy_price * density * L^3) →
  L = 6 :=
by sorry

end gold_cube_side_length_l3211_321197


namespace cubic_max_value_l3211_321184

/-- Given a cubic function with a known maximum value, prove the constant term --/
theorem cubic_max_value (m : ℝ) : 
  (∃ (x : ℝ), ∀ (t : ℝ), -t^3 + 3*t^2 + m ≤ -x^3 + 3*x^2 + m) ∧
  (∃ (x : ℝ), -x^3 + 3*x^2 + m = 10) →
  m = 6 := by
sorry

end cubic_max_value_l3211_321184


namespace kelly_apples_l3211_321190

/-- The number of apples Kelly initially has -/
def initial_apples : ℕ := 56

/-- The number of additional apples Kelly needs to pick -/
def apples_to_pick : ℕ := 49

/-- The total number of apples Kelly wants to have -/
def total_apples : ℕ := initial_apples + apples_to_pick

theorem kelly_apples : total_apples = 105 := by
  sorry

end kelly_apples_l3211_321190


namespace exponential_fraction_simplification_l3211_321188

theorem exponential_fraction_simplification :
  (3^1011 + 3^1009) / (3^1011 - 3^1009) = 5/4 := by
  sorry

end exponential_fraction_simplification_l3211_321188


namespace equidistant_points_in_quadrants_I_II_l3211_321154

/-- A point on the line 3x + 5y = 15 that is equidistant from the coordinate axes -/
def equidistant_point (x y : ℝ) : Prop :=
  3 * x + 5 * y = 15 ∧ (x = y ∨ x = -y)

/-- The point is in quadrant I -/
def in_quadrant_I (x y : ℝ) : Prop := x > 0 ∧ y > 0

/-- The point is in quadrant II -/
def in_quadrant_II (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- The point is in quadrant III -/
def in_quadrant_III (x y : ℝ) : Prop := x < 0 ∧ y < 0

/-- The point is in quadrant IV -/
def in_quadrant_IV (x y : ℝ) : Prop := x > 0 ∧ y < 0

theorem equidistant_points_in_quadrants_I_II :
  ∀ x y : ℝ, equidistant_point x y → (in_quadrant_I x y ∨ in_quadrant_II x y) ∧
  ¬(in_quadrant_III x y ∨ in_quadrant_IV x y) := by
  sorry

end equidistant_points_in_quadrants_I_II_l3211_321154


namespace last_bead_is_blue_l3211_321130

/-- Represents the colors of beads -/
inductive BeadColor
| Red
| Orange
| Yellow
| Green
| Blue
| Purple

/-- Represents the pattern of beads -/
def beadPattern : List BeadColor :=
  [BeadColor.Red, BeadColor.Orange, BeadColor.Yellow, BeadColor.Yellow,
   BeadColor.Green, BeadColor.Blue, BeadColor.Purple]

/-- The total number of beads in the bracelet -/
def totalBeads : Nat := 83

/-- Theorem stating that the last bead of the bracelet is blue -/
theorem last_bead_is_blue :
  (totalBeads % beadPattern.length) = 6 →
  beadPattern[(totalBeads - 1) % beadPattern.length] = BeadColor.Blue :=
by sorry

end last_bead_is_blue_l3211_321130


namespace remainder_properties_l3211_321194

theorem remainder_properties (a b n : ℤ) (hn : n ≠ 0) :
  (((a + b) % n = ((a % n + b % n) % n)) ∧
   ((a - b) % n = ((a % n - b % n) % n)) ∧
   ((a * b) % n = ((a % n * b % n) % n))) := by
  sorry

end remainder_properties_l3211_321194


namespace fraction_reducibility_fraction_reducibility_2_l3211_321113

theorem fraction_reducibility (n : ℤ) :
  (∃ k : ℤ, n = 3 * k - 1) ↔ 
    ∃ a b : ℤ, a ≠ 0 ∧ b ≠ 0 ∧ b ≠ 1 ∧ b * (n^2 + 2*n + 4) = a * (n^2 + n + 3) :=
sorry

theorem fraction_reducibility_2 (n : ℤ) :
  (∃ k : ℤ, n = 3 * k ∨ n = 3 * k + 1) ↔ 
    ∃ a b : ℤ, a ≠ 0 ∧ b ≠ 0 ∧ b ≠ 1 ∧ b * (n^3 - n^2 - 3*n) = a * (n^2 - n + 3) :=
sorry

end fraction_reducibility_fraction_reducibility_2_l3211_321113


namespace subset_implies_a_range_l3211_321148

open Set Real

-- Define set A
def A : Set ℝ := {x | x^2 - 4*x + 3 < 0}

-- Define set B
def B (a : ℝ) : Set ℝ := {x | 2^(1-x) + a ≤ 0 ∧ x^2 - 2*(a+7)*x + 5 ≤ 0}

-- Theorem statement
theorem subset_implies_a_range (a : ℝ) :
  A ⊆ B a → -4 ≤ a ∧ a ≤ -1 := by
  sorry

end subset_implies_a_range_l3211_321148


namespace complex_fraction_real_implies_a_negative_one_l3211_321101

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the condition that (a+i)/(1-i) is real
def is_real (a : ℝ) : Prop := ∃ (r : ℝ), (a + i) / (1 - i) = r

-- Theorem statement
theorem complex_fraction_real_implies_a_negative_one (a : ℝ) :
  is_real a → a = -1 := by sorry

end complex_fraction_real_implies_a_negative_one_l3211_321101


namespace mean_calculation_l3211_321115

theorem mean_calculation (x y : ℝ) : 
  (28 + x + 50 + 78 + 104) / 5 = 62 → 
  (48 + 62 + 98 + y + x) / 5 = (258 + y) / 5 := by
sorry

end mean_calculation_l3211_321115


namespace ben_win_probability_l3211_321123

theorem ben_win_probability (lose_prob : ℚ) (win_prob : ℚ) : 
  lose_prob = 5/8 → win_prob = 1 - lose_prob → win_prob = 3/8 := by
  sorry

end ben_win_probability_l3211_321123
