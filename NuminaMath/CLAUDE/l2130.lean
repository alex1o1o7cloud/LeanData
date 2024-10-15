import Mathlib

namespace NUMINAMATH_CALUDE_retailer_profit_is_twenty_percent_l2130_213099

/-- Calculates the percentage profit of a retailer given wholesale price, retail price, and discount percentage. -/
def calculate_percentage_profit (wholesale_price retail_price discount_percent : ℚ) : ℚ :=
  let discount := discount_percent * retail_price / 100
  let selling_price := retail_price - discount
  let profit := selling_price - wholesale_price
  (profit / wholesale_price) * 100

/-- Theorem stating that under the given conditions, the retailer's percentage profit is 20%. -/
theorem retailer_profit_is_twenty_percent :
  calculate_percentage_profit 99 132 10 = 20 := by
  sorry

end NUMINAMATH_CALUDE_retailer_profit_is_twenty_percent_l2130_213099


namespace NUMINAMATH_CALUDE_square_circles_l2130_213074

/-- A square in a plane -/
structure Square where
  vertices : Finset (ℝ × ℝ)
  is_square : vertices.card = 4

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Function to check if a circle's diameter has endpoints as vertices of the square -/
def is_valid_circle (s : Square) (c : Circle) : Prop :=
  ∃ (v1 v2 : ℝ × ℝ), v1 ∈ s.vertices ∧ v2 ∈ s.vertices ∧
    v1 ≠ v2 ∧
    c.center = ((v1.1 + v2.1) / 2, (v1.2 + v2.2) / 2) ∧
    c.radius = Real.sqrt ((v1.1 - v2.1)^2 + (v1.2 - v2.2)^2) / 2

/-- The main theorem -/
theorem square_circles (s : Square) :
  ∃! (circles : Finset Circle), circles.card = 2 ∧
    ∀ c ∈ circles, is_valid_circle s c ∧
    ∀ c, is_valid_circle s c → c ∈ circles :=
  sorry


end NUMINAMATH_CALUDE_square_circles_l2130_213074


namespace NUMINAMATH_CALUDE_green_notebook_cost_l2130_213052

theorem green_notebook_cost (total_cost black_cost pink_cost : ℕ) 
  (h1 : total_cost = 45)
  (h2 : black_cost = 15)
  (h3 : pink_cost = 10) :
  (total_cost - (black_cost + pink_cost)) / 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_green_notebook_cost_l2130_213052


namespace NUMINAMATH_CALUDE_cheryl_material_calculation_l2130_213006

/-- The amount of the second type of material Cheryl needed for her project -/
def second_material_amount : ℚ := 1 / 8

/-- The amount of the first type of material Cheryl bought -/
def first_material_amount : ℚ := 2 / 9

/-- The amount of material Cheryl had left after the project -/
def leftover_amount : ℚ := 4 / 18

/-- The total amount of material Cheryl used -/
def total_used : ℚ := 1 / 8

theorem cheryl_material_calculation :
  second_material_amount = 
    (first_material_amount + leftover_amount + total_used) - first_material_amount := by
  sorry

end NUMINAMATH_CALUDE_cheryl_material_calculation_l2130_213006


namespace NUMINAMATH_CALUDE_parallelogram_area_12_48_l2130_213083

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 12 cm and height 48 cm is 576 square centimeters -/
theorem parallelogram_area_12_48 : parallelogram_area 12 48 = 576 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_12_48_l2130_213083


namespace NUMINAMATH_CALUDE_divisor_problem_l2130_213077

theorem divisor_problem (a b : ℕ) (divisor : ℕ) : 
  (10 ≤ a ∧ a ≤ 99) →  -- a is a two-digit number
  (a = 10 * (a / 10) + (a % 10)) →  -- a is represented in decimal form
  (divisor > 0) →  -- divisor is positive
  (a % divisor = 0) →  -- a is divisible by divisor
  (∀ x y : ℕ, (10 ≤ x ∧ x ≤ 99) → (x % divisor = 0) → (x / 10) * (x % 10) ≤ (a / 10) * (a % 10)) →  -- greatest possible value of b × a
  ((a / 10) * (a % 10) = 35) →  -- b × a = 35
  divisor = 3 :=
by sorry

end NUMINAMATH_CALUDE_divisor_problem_l2130_213077


namespace NUMINAMATH_CALUDE_inscribed_rectangle_width_l2130_213092

/-- Given a right-angled triangle with legs a and b, and a rectangle inscribed
    such that its width d satisfies d(d - (a + b)) = 0, prove that d = a + b -/
theorem inscribed_rectangle_width (a b d : ℝ) (h : d * (d - (a + b)) = 0) :
  d = a + b := by
  sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_width_l2130_213092


namespace NUMINAMATH_CALUDE_divisible_by_thirteen_l2130_213080

theorem divisible_by_thirteen (n : ℕ) (h : n > 0) :
  ∃ m : ℤ, 4^(2*n - 1) + 3^(n + 1) = 13 * m := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_thirteen_l2130_213080


namespace NUMINAMATH_CALUDE_x_varies_as_three_sevenths_power_of_z_l2130_213009

/-- Given that x varies directly as the cube of y, and y varies directly as the seventh root of z,
    prove that x varies as the (3/7)th power of z. -/
theorem x_varies_as_three_sevenths_power_of_z 
  (x y z : ℝ) 
  (hxy : ∃ (k : ℝ), x = k * y^3) 
  (hyz : ∃ (j : ℝ), y = j * z^(1/7)) :
  ∃ (m : ℝ), x = m * z^(3/7) := by
sorry

end NUMINAMATH_CALUDE_x_varies_as_three_sevenths_power_of_z_l2130_213009


namespace NUMINAMATH_CALUDE_min_value_of_even_function_l2130_213088

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (a^2 - 1) * x - 3 * a

-- State the theorem
theorem min_value_of_even_function (a : ℝ) :
  (∀ x, f a x = f a (-x)) →  -- f is an even function
  (∀ x, x ∈ Set.Icc (4 * a + 2) (a^2 + 1) → f a x ∈ Set.range (f a)) →  -- domain of f is [4a+2, a^2+1]
  (∃ x, x ∈ Set.Icc (4 * a + 2) (a^2 + 1) ∧ f a x = -1) →  -- -1 is in the range of f
  (∀ x, x ∈ Set.Icc (4 * a + 2) (a^2 + 1) → f a x ≥ -1) →  -- -1 is the minimum value
  (∃ x, x ∈ Set.Icc (4 * a + 2) (a^2 + 1) ∧ f a x = -1)  -- the minimum value of f(x) is -1
  := by sorry

end NUMINAMATH_CALUDE_min_value_of_even_function_l2130_213088


namespace NUMINAMATH_CALUDE_initial_distance_proof_l2130_213042

/-- The initial distance between Fred and Sam -/
def initial_distance : ℝ := 35

/-- Fred's walking speed in miles per hour -/
def fred_speed : ℝ := 2

/-- Sam's walking speed in miles per hour -/
def sam_speed : ℝ := 5

/-- The distance Sam walks before they meet -/
def sam_distance : ℝ := 25

theorem initial_distance_proof :
  initial_distance = sam_distance + (sam_distance * fred_speed) / sam_speed :=
by sorry

end NUMINAMATH_CALUDE_initial_distance_proof_l2130_213042


namespace NUMINAMATH_CALUDE_min_buses_for_field_trip_l2130_213076

/-- Represents the number of passengers that can be transported by a combination of buses. -/
def transport_capacity (small medium large : ℕ) : ℕ :=
  30 * small + 48 * medium + 72 * large

/-- Represents the total number of buses used. -/
def total_buses (small medium large : ℕ) : ℕ :=
  small + medium + large

theorem min_buses_for_field_trip :
  ∃ (small medium large : ℕ),
    small ≤ 10 ∧
    medium ≤ 15 ∧
    large ≤ 5 ∧
    transport_capacity small medium large ≥ 1230 ∧
    total_buses small medium large = 25 ∧
    (∀ (s m l : ℕ),
      s ≤ 10 →
      m ≤ 15 →
      l ≤ 5 →
      transport_capacity s m l ≥ 1230 →
      total_buses s m l ≥ 25) :=
by sorry

end NUMINAMATH_CALUDE_min_buses_for_field_trip_l2130_213076


namespace NUMINAMATH_CALUDE_derivative_f_at_1_l2130_213065

def f (x : ℝ) : ℝ := x^2 + 2

theorem derivative_f_at_1 : 
  deriv f 1 = 2 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_1_l2130_213065


namespace NUMINAMATH_CALUDE_two_from_four_combinations_l2130_213030

theorem two_from_four_combinations : Nat.choose 4 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_two_from_four_combinations_l2130_213030


namespace NUMINAMATH_CALUDE_olivia_basketball_cards_l2130_213002

theorem olivia_basketball_cards 
  (basketball_price : ℕ)
  (baseball_decks : ℕ)
  (baseball_price : ℕ)
  (total_paid : ℕ)
  (change : ℕ)
  (h1 : basketball_price = 3)
  (h2 : baseball_decks = 5)
  (h3 : baseball_price = 4)
  (h4 : total_paid = 50)
  (h5 : change = 24) :
  ∃ (x : ℕ), x * basketball_price + baseball_decks * baseball_price = total_paid - change ∧ x = 2 :=
by sorry

end NUMINAMATH_CALUDE_olivia_basketball_cards_l2130_213002


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_l2130_213027

theorem square_plus_reciprocal_square (x : ℝ) (h : x + 1/x = 8) : x^2 + 1/x^2 = 62 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_l2130_213027


namespace NUMINAMATH_CALUDE_largest_n_with_prime_differences_l2130_213054

theorem largest_n_with_prime_differences : ∃ n : ℕ, 
  (n = 10) ∧ 
  (∀ m : ℕ, m > 10 → 
    ∃ p : ℕ, Prime p ∧ 2 < p ∧ p < m ∧ ¬(Prime (m - p))) ∧
  (∀ p : ℕ, Prime p → 2 < p → p < 10 → Prime (10 - p)) :=
sorry

end NUMINAMATH_CALUDE_largest_n_with_prime_differences_l2130_213054


namespace NUMINAMATH_CALUDE_circumcircle_area_of_special_triangle_l2130_213096

/-- Given a triangle ABC with sides a, b, c, area S, where a² + b² - c² = 4√3 * S and c = 1,
    the area of its circumcircle is π. -/
theorem circumcircle_area_of_special_triangle (a b c S : ℝ) : 
  a > 0 → b > 0 → c > 0 → S > 0 →
  a^2 + b^2 - c^2 = 4 * Real.sqrt 3 * S →
  c = 1 →
  ∃ (R : ℝ), R > 0 ∧ π * R^2 = π := by sorry

end NUMINAMATH_CALUDE_circumcircle_area_of_special_triangle_l2130_213096


namespace NUMINAMATH_CALUDE_solve_equation_l2130_213023

theorem solve_equation (x : ℝ) (h : 0.12 / x * 2 = 12) : x = 0.02 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2130_213023


namespace NUMINAMATH_CALUDE_composition_equation_solution_l2130_213089

theorem composition_equation_solution (c : ℝ) : 
  let r (x : ℝ) := 5 * x - 8
  let s (x : ℝ) := 4 * x - c
  r (s 3) = 17 → c = 7 := by
sorry

end NUMINAMATH_CALUDE_composition_equation_solution_l2130_213089


namespace NUMINAMATH_CALUDE_smallest_n_multiple_of_seven_l2130_213064

theorem smallest_n_multiple_of_seven (x y : ℤ) 
  (hx : 7 ∣ (x + 2)) 
  (hy : 7 ∣ (y - 2)) : 
  (∃ n : ℕ+, 7 ∣ (x^2 - x*y + y^2 + n) ∧ 
    ∀ m : ℕ+, 7 ∣ (x^2 - x*y + y^2 + m) → n ≤ m) ∧
  (7 ∣ (x^2 - x*y + y^2 + 2)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_multiple_of_seven_l2130_213064


namespace NUMINAMATH_CALUDE_letters_problem_l2130_213035

/-- The number of letters Greta's brother received -/
def brothers_letters : ℕ := sorry

/-- The number of letters Greta received -/
def gretas_letters : ℕ := sorry

/-- The number of letters Greta's mother received -/
def mothers_letters : ℕ := sorry

theorem letters_problem :
  (gretas_letters = brothers_letters + 10) ∧
  (mothers_letters = 2 * (gretas_letters + brothers_letters)) ∧
  (brothers_letters + gretas_letters + mothers_letters = 270) →
  brothers_letters = 40 := by
  sorry

end NUMINAMATH_CALUDE_letters_problem_l2130_213035


namespace NUMINAMATH_CALUDE_circle_tangent_to_x_axis_l2130_213047

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define what it means for a circle to be tangent to the x-axis
def tangentToXAxis (c : Circle) : Prop :=
  c.center.2 = c.radius

-- Define the equation of a circle
def circleEquation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

-- Theorem statement
theorem circle_tangent_to_x_axis :
  ∀ (c : Circle),
    c.center = (5, 4) →
    tangentToXAxis c →
    ∀ (x y : ℝ), circleEquation c x y ↔ (x - 5)^2 + (y - 4)^2 = 16 :=
by
  sorry


end NUMINAMATH_CALUDE_circle_tangent_to_x_axis_l2130_213047


namespace NUMINAMATH_CALUDE_zoo_animals_count_l2130_213060

theorem zoo_animals_count (penguins : ℕ) (polar_bears : ℕ) : 
  penguins = 21 → polar_bears = 2 * penguins → penguins + polar_bears = 63 := by
  sorry

end NUMINAMATH_CALUDE_zoo_animals_count_l2130_213060


namespace NUMINAMATH_CALUDE_mammaad_arrangements_l2130_213020

theorem mammaad_arrangements : 
  let total_letters : ℕ := 7
  let m_count : ℕ := 3
  let a_count : ℕ := 3
  let d_count : ℕ := 1
  (total_letters.factorial) / (m_count.factorial * a_count.factorial * d_count.factorial) = 140 := by
  sorry

end NUMINAMATH_CALUDE_mammaad_arrangements_l2130_213020


namespace NUMINAMATH_CALUDE_min_beta_delta_sum_l2130_213003

open Complex

/-- A complex-valued function g defined as g(z) = (5 + 3i)z^3 + βz + δ -/
def g (β δ : ℂ) (z : ℂ) : ℂ := (5 + 3*I)*z^3 + β*z + δ

/-- The theorem stating the minimum value of |β| + |δ| given the conditions -/
theorem min_beta_delta_sum :
  ∀ β δ : ℂ,
  (g β δ 1).im = 0 →
  (g β δ (-I)).im = 0 →
  (∃ β₀ δ₀ : ℂ, ∀ β δ : ℂ, (g β δ 1).im = 0 → (g β δ (-I)).im = 0 →
    Complex.abs β₀ + Complex.abs δ₀ ≤ Complex.abs β + Complex.abs δ) →
  ∃ β₀ δ₀ : ℂ, Complex.abs β₀ + Complex.abs δ₀ = Real.sqrt 73 :=
sorry

end NUMINAMATH_CALUDE_min_beta_delta_sum_l2130_213003


namespace NUMINAMATH_CALUDE_second_street_sales_l2130_213078

/-- Represents the sales data for a door-to-door salesman selling security systems. -/
structure SalesData where
  commission_per_sale : ℕ
  total_commission : ℕ
  first_street_sales : ℕ
  second_street_sales : ℕ
  fourth_street_sales : ℕ

/-- Theorem stating the number of security systems sold on the second street. -/
theorem second_street_sales (data : SalesData) : data.second_street_sales = 4 :=
  by
  have h1 : data.commission_per_sale = 25 := by sorry
  have h2 : data.total_commission = 175 := by sorry
  have h3 : data.first_street_sales = data.second_street_sales / 2 := by sorry
  have h4 : data.fourth_street_sales = 1 := by sorry
  have h5 : data.first_street_sales + data.second_street_sales + data.fourth_street_sales = 
            data.total_commission / data.commission_per_sale := by sorry
  sorry

end NUMINAMATH_CALUDE_second_street_sales_l2130_213078


namespace NUMINAMATH_CALUDE_game_outcomes_l2130_213097

/-- The game state -/
inductive GameState
| A (n : ℕ)  -- Player A's turn with current number n
| B (n : ℕ)  -- Player B's turn with current number n

/-- The possible outcomes of the game -/
inductive Outcome
| AWin  -- Player A wins
| BWin  -- Player B wins
| Draw  -- Neither player has a winning strategy

/-- Definition of a winning strategy for a player -/
def has_winning_strategy (player : GameState → Prop) (s : GameState) : Prop :=
  ∃ (strategy : GameState → ℕ), 
    ∀ (game : ℕ → GameState),
      game 0 = s →
      (∀ n, player (game n) → game (n + 1) = GameState.B (strategy (game n))) →
      (∃ m, game m = GameState.A 1990 ∨ game m = GameState.B 1)

/-- The main theorem about the game outcomes -/
theorem game_outcomes (n₀ : ℕ) : 
  (has_winning_strategy (λ s => ∃ n, s = GameState.A n) (GameState.A n₀) ↔ n₀ ≥ 8) ∧
  (has_winning_strategy (λ s => ∃ n, s = GameState.B n) (GameState.A n₀) ↔ n₀ ≤ 5) ∧
  (¬ has_winning_strategy (λ s => ∃ n, s = GameState.A n) (GameState.A n₀) ∧
   ¬ has_winning_strategy (λ s => ∃ n, s = GameState.B n) (GameState.A n₀) ↔ n₀ = 6 ∨ n₀ = 7) :=
sorry

end NUMINAMATH_CALUDE_game_outcomes_l2130_213097


namespace NUMINAMATH_CALUDE_age_difference_l2130_213018

theorem age_difference (a b c : ℕ) : 
  b = 18 →
  b = 2 * c →
  a + b + c = 47 →
  a = b + 2 :=
by sorry

end NUMINAMATH_CALUDE_age_difference_l2130_213018


namespace NUMINAMATH_CALUDE_last_l_replaced_by_p_l2130_213004

-- Define the alphabet size
def alphabet_size : ℕ := 26

-- Define the position of 'l' in the alphabet (1-indexed)
def l_position : ℕ := 12

-- Define the occurrence of the last 'l' in the message
def l_occurrence : ℕ := 2

-- Define the shift function
def shift (n : ℕ) : ℕ := 2^n

-- Define the function to calculate the new position
def new_position (start : ℕ) (shift : ℕ) : ℕ :=
  (start + shift - 1) % alphabet_size + 1

-- Define the position of 'p' in the alphabet (1-indexed)
def p_position : ℕ := 16

-- The theorem to prove
theorem last_l_replaced_by_p :
  new_position l_position (shift l_occurrence) = p_position := by
  sorry

end NUMINAMATH_CALUDE_last_l_replaced_by_p_l2130_213004


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_specific_proposition_l2130_213011

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) := by sorry

theorem negation_of_specific_proposition :
  (¬ ∃ x : ℝ, x^2 - 2 ≤ 0) ↔ (∀ x : ℝ, x^2 - 2 > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_specific_proposition_l2130_213011


namespace NUMINAMATH_CALUDE_determinant_expansion_second_column_l2130_213010

theorem determinant_expansion_second_column 
  (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) : 
  let M := ![![a₁, 3, b₁], ![a₂, 2, b₂], ![a₃, -2, b₃]]
  Matrix.det M = 3 * Matrix.det ![![a₂, b₂], ![a₃, b₃]] + 
                 2 * Matrix.det ![![a₁, b₁], ![a₃, b₃]] - 
                 2 * Matrix.det ![![a₁, b₁], ![a₂, b₂]] := by
  sorry

end NUMINAMATH_CALUDE_determinant_expansion_second_column_l2130_213010


namespace NUMINAMATH_CALUDE_part_I_part_II_l2130_213094

-- Define the statements p and q
def p (x : ℝ) : Prop := (x + 2) * (x - 6) ≤ 0
def q (m x : ℝ) : Prop := 2 - m ≤ x ∧ x ≤ 2 + m

-- Part I
theorem part_I (m : ℝ) (h1 : m > 0) (h2 : ∀ x, p x → q m x) : m ≥ 4 := by
  sorry

-- Part II
theorem part_II (x : ℝ) (h1 : ∀ x, p x ∨ q 5 x) (h2 : ¬∀ x, p x ∧ q 5 x) :
  x ∈ Set.Icc (-3 : ℝ) (-2) ∪ Set.Ioc 6 7 := by
  sorry

end NUMINAMATH_CALUDE_part_I_part_II_l2130_213094


namespace NUMINAMATH_CALUDE_binomial_multiply_three_l2130_213073

theorem binomial_multiply_three : 3 * Nat.choose 9 5 = 378 := by sorry

end NUMINAMATH_CALUDE_binomial_multiply_three_l2130_213073


namespace NUMINAMATH_CALUDE_probability_no_3x3_red_l2130_213098

/-- Represents a 4x4 grid where each cell can be colored red or blue -/
def Grid := Fin 4 → Fin 4 → Bool

/-- Checks if a 3x3 subgrid starting at (i, j) is all red -/
def has_red_3x3 (g : Grid) (i j : Fin 2) : Prop :=
  ∀ (x y : Fin 3), g (i + x) (j + y) = true

/-- A grid is valid if it doesn't contain a 3x3 red square -/
def is_valid_grid (g : Grid) : Prop :=
  ¬ ∃ (i j : Fin 2), has_red_3x3 g i j

/-- The probability of a single cell being red -/
def p_red : ℚ := 1/2

/-- The total number of possible 4x4 grids -/
def total_grids : ℕ := 2^16

/-- The number of valid 4x4 grids (without 3x3 red squares) -/
def valid_grids : ℕ := 65152

theorem probability_no_3x3_red : 
  (valid_grids : ℚ) / total_grids = 509 / 512 :=
sorry

end NUMINAMATH_CALUDE_probability_no_3x3_red_l2130_213098


namespace NUMINAMATH_CALUDE_ellipse_dot_product_l2130_213014

/-- An ellipse with given properties and a line intersecting it -/
structure EllipseWithLine where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0
  h_ecc : (1 - b^2 / a^2).sqrt = Real.sqrt 2 / 2
  h_point : b^2 = 1
  A : ℝ × ℝ
  B : ℝ × ℝ
  P : ℝ × ℝ
  h_A : A.1 = -a ∧ A.2 = 0
  h_B : B.1^2 / a^2 + B.2^2 / b^2 = 1
  h_P : P.1 = a
  h_collinear : ∃ (t : ℝ), B = A + t • (P - A)

/-- The dot product of OB and OP is 2 -/
theorem ellipse_dot_product (e : EllipseWithLine) : e.B.1 * e.P.1 + e.B.2 * e.P.2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_dot_product_l2130_213014


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l2130_213059

theorem repeating_decimal_sum (c d : ℕ) : 
  (c < 10 ∧ d < 10) →  -- c and d are single digits
  (5 : ℚ) / 13 = (c * 10 + d : ℚ) / 99 →  -- 0.cdcdc... = (c*10 + d) / 99
  c + d = 11 := by
sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l2130_213059


namespace NUMINAMATH_CALUDE_rectangle_area_perimeter_relation_l2130_213031

theorem rectangle_area_perimeter_relation (x : ℝ) : 
  let length : ℝ := 4 * x
  let width : ℝ := x + 7
  let area : ℝ := length * width
  let perimeter : ℝ := 2 * (length + width)
  (area = 2 * perimeter ∧ length > 0 ∧ width > 0) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_perimeter_relation_l2130_213031


namespace NUMINAMATH_CALUDE_max_sum_with_constraints_l2130_213081

theorem max_sum_with_constraints (a b : ℝ) 
  (h1 : 4 * a + 3 * b ≤ 10) 
  (h2 : 3 * a + 6 * b ≤ 12) : 
  a + b ≤ 22 / 7 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_with_constraints_l2130_213081


namespace NUMINAMATH_CALUDE_ps_length_is_eight_l2130_213072

/-- Triangle PQR with given side lengths and angle bisector PS -/
structure TrianglePQR where
  /-- Length of side PQ -/
  PQ : ℝ
  /-- Length of side QR -/
  QR : ℝ
  /-- Length of side PR -/
  PR : ℝ
  /-- PS is the angle bisector of ∠PQR -/
  PS_is_angle_bisector : Bool

/-- The theorem stating that PS = 8 in the given triangle -/
theorem ps_length_is_eight (t : TrianglePQR) 
  (h1 : t.PQ = 8)
  (h2 : t.QR = 15)
  (h3 : t.PR = 17)
  (h4 : t.PS_is_angle_bisector = true) :
  ∃ PS : ℝ, PS = 8 ∧ PS > 0 := by
  sorry


end NUMINAMATH_CALUDE_ps_length_is_eight_l2130_213072


namespace NUMINAMATH_CALUDE_right_triangle_third_side_l2130_213049

theorem right_triangle_third_side : ∀ a b c : ℝ,
  a > 0 → b > 0 → c > 0 →
  (a = 3 ∧ b = 4) ∨ (a = 3 ∧ c = 4) ∨ (b = 3 ∧ c = 4) →
  a^2 + b^2 = c^2 →
  c = 5 ∨ c = Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_l2130_213049


namespace NUMINAMATH_CALUDE_opposite_direction_speed_l2130_213061

/-- Given two people moving in opposite directions for 45 minutes,
    with one moving at 30 kmph and ending up 60 km apart,
    prove that the speed of the other person is 50 kmph. -/
theorem opposite_direction_speed 
  (riya_speed : ℝ) 
  (time : ℝ) 
  (total_distance : ℝ) 
  (h1 : riya_speed = 30) 
  (h2 : time = 45 / 60) 
  (h3 : total_distance = 60) : 
  ∃ (priya_speed : ℝ), priya_speed = 50 ∧ 
    riya_speed * time + priya_speed * time = total_distance :=
by sorry

end NUMINAMATH_CALUDE_opposite_direction_speed_l2130_213061


namespace NUMINAMATH_CALUDE_expected_rainfall_l2130_213013

/-- The expected value of total rainfall over 7 days given specific weather conditions --/
theorem expected_rainfall (p_sunny p_light p_heavy : ℝ) (r_light r_heavy : ℝ) (days : ℕ) : 
  p_sunny + p_light + p_heavy = 1 →
  p_sunny = 0.3 →
  p_light = 0.4 →
  p_heavy = 0.3 →
  r_light = 3 →
  r_heavy = 6 →
  days = 7 →
  days * (p_sunny * 0 + p_light * r_light + p_heavy * r_heavy) = 21 :=
by sorry

end NUMINAMATH_CALUDE_expected_rainfall_l2130_213013


namespace NUMINAMATH_CALUDE_robin_albums_l2130_213090

theorem robin_albums (total_pictures : ℕ) (pictures_per_album : ℕ) (h1 : total_pictures = 40) (h2 : pictures_per_album = 8) : total_pictures / pictures_per_album = 5 := by
  sorry

end NUMINAMATH_CALUDE_robin_albums_l2130_213090


namespace NUMINAMATH_CALUDE_kyle_practice_time_l2130_213055

/-- Kyle's daily basketball practice schedule -/
def KylePractice : Prop :=
  ∃ (total_time shooting_time running_time weightlifting_time : ℕ),
    -- Total practice time
    total_time = shooting_time + running_time + weightlifting_time
    -- Half time spent shooting
    ∧ 2 * shooting_time = total_time
    -- Running time is twice weightlifting time
    ∧ running_time = 2 * weightlifting_time
    -- Weightlifting time is 20 minutes
    ∧ weightlifting_time = 20
    -- Total time in hours is 2
    ∧ total_time = 120

/-- Theorem: Kyle's daily basketball practice is 2 hours -/
theorem kyle_practice_time : KylePractice := by
  sorry

end NUMINAMATH_CALUDE_kyle_practice_time_l2130_213055


namespace NUMINAMATH_CALUDE_height_difference_l2130_213034

/-- The height of the CN Tower in meters -/
def cn_tower_height : ℝ := 553

/-- The height of the Space Needle in meters -/
def space_needle_height : ℝ := 184

/-- Theorem stating the difference in height between the CN Tower and the Space Needle -/
theorem height_difference : cn_tower_height - space_needle_height = 369 := by
  sorry

end NUMINAMATH_CALUDE_height_difference_l2130_213034


namespace NUMINAMATH_CALUDE_car_speed_proof_l2130_213053

/-- Proves that a car traveling for two hours with an average speed of 60 km/h
    and a speed of 30 km/h in the second hour must have a speed of 90 km/h in the first hour. -/
theorem car_speed_proof (x : ℝ) :
  (x + 30) / 2 = 60 →
  x = 90 :=
by sorry

end NUMINAMATH_CALUDE_car_speed_proof_l2130_213053


namespace NUMINAMATH_CALUDE_matrix_product_result_l2130_213007

def odd_matrix (k : ℕ) : Matrix (Fin 2) (Fin 2) ℕ :=
  !![1, k; 0, 1]

def matrix_product : Matrix (Fin 2) (Fin 2) ℕ :=
  (List.range 50).foldl (λ acc i => acc * odd_matrix (2 * i + 1)) (odd_matrix 1)

theorem matrix_product_result :
  matrix_product = !![1, 2500; 0, 1] := by
  sorry

end NUMINAMATH_CALUDE_matrix_product_result_l2130_213007


namespace NUMINAMATH_CALUDE_art_to_maths_ratio_is_one_to_one_l2130_213019

/-- Represents the school supplies problem --/
structure SchoolSupplies where
  total_budget : ℕ
  maths_books : ℕ
  maths_book_price : ℕ
  science_books_diff : ℕ
  science_book_price : ℕ
  music_books_cost : ℕ

/-- The ratio of art books to maths books is 1:1 --/
def art_to_maths_ratio (s : SchoolSupplies) : Prop :=
  let total_spent := s.maths_books * s.maths_book_price + 
                     (s.maths_books + s.science_books_diff) * s.science_book_price + 
                     s.maths_books * s.maths_book_price + 
                     s.music_books_cost
  total_spent ≤ s.total_budget ∧ 
  (s.maths_books : ℚ) / s.maths_books = 1

/-- The main theorem stating that the ratio of art books to maths books is 1:1 --/
theorem art_to_maths_ratio_is_one_to_one (s : SchoolSupplies) 
  (h : s = { total_budget := 500,
             maths_books := 4,
             maths_book_price := 20,
             science_books_diff := 6,
             science_book_price := 10,
             music_books_cost := 160 }) : 
  art_to_maths_ratio s := by
  sorry


end NUMINAMATH_CALUDE_art_to_maths_ratio_is_one_to_one_l2130_213019


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2130_213024

theorem polynomial_simplification (y : ℝ) :
  (3 * y - 2) * (5 * y^12 + 3 * y^11 + 5 * y^9 + 3 * y^8) =
  15 * y^13 - y^12 + 3 * y^11 + 15 * y^10 - y^9 - 6 * y^8 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2130_213024


namespace NUMINAMATH_CALUDE_trumpington_band_size_l2130_213001

theorem trumpington_band_size (n : ℕ) : 
  (∃ k : ℕ, 20 * n = 26 * k + 4) → 
  20 * n < 1000 → 
  (∀ m : ℕ, (∃ j : ℕ, 20 * m = 26 * j + 4) → 20 * m < 1000 → 20 * m ≤ 20 * n) →
  20 * n = 940 :=
by sorry

end NUMINAMATH_CALUDE_trumpington_band_size_l2130_213001


namespace NUMINAMATH_CALUDE_total_tickets_used_l2130_213040

/-- The cost of the shooting game in tickets -/
def shooting_game_cost : ℕ := 5

/-- The cost of the carousel in tickets -/
def carousel_cost : ℕ := 3

/-- The number of times Jen played the shooting game -/
def jen_games : ℕ := 2

/-- The number of times Russel rode the carousel -/
def russel_rides : ℕ := 3

/-- Theorem stating the total number of tickets used -/
theorem total_tickets_used : 
  shooting_game_cost * jen_games + carousel_cost * russel_rides = 19 := by
  sorry

end NUMINAMATH_CALUDE_total_tickets_used_l2130_213040


namespace NUMINAMATH_CALUDE_min_degree_of_g_l2130_213057

/-- Given polynomials f, g, and h satisfying the equation 5f + 6g = h,
    where deg(f) = 10 and deg(h) = 11, the minimum possible degree of g is 11. -/
theorem min_degree_of_g (f g h : Polynomial ℝ)
  (eq : 5 • f + 6 • g = h)
  (deg_f : Polynomial.degree f = 10)
  (deg_h : Polynomial.degree h = 11) :
  Polynomial.degree g ≥ 11 ∧ ∃ (g' : Polynomial ℝ), Polynomial.degree g' = 11 ∧ 5 • f + 6 • g' = h :=
sorry

end NUMINAMATH_CALUDE_min_degree_of_g_l2130_213057


namespace NUMINAMATH_CALUDE_f_inequality_l2130_213051

/-- A function satisfying the given conditions -/
def f_conditions (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (x + 2) = f x) ∧
  (∀ x₁ x₂ : ℝ, 0 ≤ x₁ → x₁ < x₂ → x₂ ≤ 1 → f x₁ > f x₂) ∧
  (∀ x : ℝ, f (x + 1) = f (-x + 1))

theorem f_inequality (f : ℝ → ℝ) (h : f_conditions f) :
  f 5.5 < f 7.8 ∧ f 7.8 < f (-2) := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l2130_213051


namespace NUMINAMATH_CALUDE_inequality_holds_iff_a_in_range_l2130_213056

theorem inequality_holds_iff_a_in_range (a : ℝ) :
  (∀ x : ℝ, a * x / (x^2 + 4) < 1.5) ↔ -6 < a ∧ a < 6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_a_in_range_l2130_213056


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2130_213082

/-- A geometric sequence with common ratio q > 1 -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  q > 1 ∧ ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  GeometricSequence a q →
  (4 * (a 2005)^2 - 8 * (a 2005) + 3 = 0) →
  (4 * (a 2006)^2 - 8 * (a 2006) + 3 = 0) →
  a 2007 + a 2008 = 18 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2130_213082


namespace NUMINAMATH_CALUDE_prob_at_least_two_different_fruits_l2130_213036

def num_fruits : ℕ := 4
def num_meals : ℕ := 3

def prob_same_fruit_all_day : ℚ := (1 / num_fruits) ^ num_meals * num_fruits

theorem prob_at_least_two_different_fruits :
  1 - prob_same_fruit_all_day = 15/16 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_two_different_fruits_l2130_213036


namespace NUMINAMATH_CALUDE_height_matching_problem_sixteenth_answer_l2130_213093

/-- Represents a group of people with their height matches -/
structure HeightGroup :=
  (total : ℕ)
  (one_match : ℕ)
  (two_matches : ℕ)
  (three_matches : ℕ)
  (h_total : total = 16)
  (h_one : one_match = 6)
  (h_two : two_matches = 6)
  (h_three : three_matches = 3)

/-- The number of people accounted for by each match type -/
def accounted_for (g : HeightGroup) : ℕ :=
  g.one_match * 2 + g.two_matches * 3 + g.three_matches * 4

theorem height_matching_problem (g : HeightGroup) :
  accounted_for g = g.total + 3 :=
sorry

theorem sixteenth_answer (g : HeightGroup) :
  g.total - (g.one_match + g.two_matches + g.three_matches) = 1 ∧
  accounted_for g = g.total + 3 →
  3 = g.total - (accounted_for g - 3) :=
sorry

end NUMINAMATH_CALUDE_height_matching_problem_sixteenth_answer_l2130_213093


namespace NUMINAMATH_CALUDE_simplify_radicals_l2130_213000

theorem simplify_radicals : Real.sqrt (5 * 3) * Real.sqrt (3^3 * 5^3) = 225 := by
  sorry

end NUMINAMATH_CALUDE_simplify_radicals_l2130_213000


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l2130_213071

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |2 * x - 4| = x + 3 :=
by
  -- The unique solution is x = 7
  use 7
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l2130_213071


namespace NUMINAMATH_CALUDE_tangent_line_implies_a_eq_neg_one_l2130_213021

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x + 1/x - a * Real.log x

-- Define the tangent line
def tangent_line (x : ℝ) : ℝ := x + 1

-- Theorem statement
theorem tangent_line_implies_a_eq_neg_one (a : ℝ) :
  (∃ x₀ : ℝ, x₀ > 0 ∧ 
    f a x₀ = tangent_line x₀ ∧ 
    (deriv (f a)) x₀ = (deriv tangent_line) x₀) →
  a = -1 :=
sorry

end

end NUMINAMATH_CALUDE_tangent_line_implies_a_eq_neg_one_l2130_213021


namespace NUMINAMATH_CALUDE_min_people_with_both_hat_and_glove_l2130_213028

theorem min_people_with_both_hat_and_glove (n : ℕ) (gloves hats both : ℕ) : 
  n > 0 → 
  gloves = n / 3 →
  hats = 2 * n / 3 →
  gloves + hats - both = n →
  both ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_min_people_with_both_hat_and_glove_l2130_213028


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l2130_213032

/-- Given vectors a and b in R², find k such that a ⟂ (a + kb) -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (h1 : a = (-2, 1)) (h2 : b = (3, 2)) :
  ∃ k : ℝ, k = 5/4 ∧ a.1 * (a.1 + k * b.1) + a.2 * (a.2 + k * b.2) = 0 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l2130_213032


namespace NUMINAMATH_CALUDE_wallet_value_l2130_213067

def total_bills : ℕ := 12
def five_dollar_bills : ℕ := 4
def five_dollar_value : ℕ := 5
def ten_dollar_value : ℕ := 10

theorem wallet_value :
  (five_dollar_bills * five_dollar_value) +
  ((total_bills - five_dollar_bills) * ten_dollar_value) = 100 :=
by sorry

end NUMINAMATH_CALUDE_wallet_value_l2130_213067


namespace NUMINAMATH_CALUDE_function_increasing_implies_a_leq_one_l2130_213012

/-- Given a function f(x) = e^(|x-a|), where a is a constant,
    if f(x) is increasing on [1, +∞), then a ≤ 1 -/
theorem function_increasing_implies_a_leq_one (a : ℝ) :
  (∀ x y, 1 ≤ x ∧ x < y → (Real.exp (|x - a|) < Real.exp (|y - a|))) →
  a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_function_increasing_implies_a_leq_one_l2130_213012


namespace NUMINAMATH_CALUDE_share_division_l2130_213085

theorem share_division (total : ℕ) (a b c : ℚ) 
  (h_total : total = 427)
  (h_sum : a + b + c = total)
  (h_ratio : 3 * a = 4 * b ∧ 4 * b = 7 * c) : 
  c = 84 := by
  sorry

end NUMINAMATH_CALUDE_share_division_l2130_213085


namespace NUMINAMATH_CALUDE_find_multiple_l2130_213008

theorem find_multiple (x y m : ℝ) : 
  x + y = 8 → 
  y - m * x = 7 → 
  y - x = 7.5 → 
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_find_multiple_l2130_213008


namespace NUMINAMATH_CALUDE_mall_profit_analysis_l2130_213084

def average_daily_sales : ℝ := 20
def profit_per_shirt : ℝ := 40
def additional_sales_per_yuan : ℝ := 2

def daily_profit (x : ℝ) : ℝ :=
  (profit_per_shirt - x) * (average_daily_sales + additional_sales_per_yuan * x)

theorem mall_profit_analysis :
  ∃ (f : ℝ → ℝ),
    (∀ x, daily_profit x = f x) ∧
    (f x = -2 * x^2 + 60 * x + 800) ∧
    (∃ x_max, ∀ x, f x ≤ f x_max ∧ x_max = 15) ∧
    (∃ x1 x2, x1 ≠ x2 ∧ f x1 = 1200 ∧ f x2 = 1200 ∧ (x1 = 10 ∨ x1 = 20) ∧ (x2 = 10 ∨ x2 = 20)) :=
by sorry

end NUMINAMATH_CALUDE_mall_profit_analysis_l2130_213084


namespace NUMINAMATH_CALUDE_seventh_group_draw_l2130_213087

/-- Represents the systematic sampling method for a population -/
structure SystematicSampling where
  populationSize : Nat
  groupCount : Nat
  sampleSize : Nat
  firstDrawn : Nat

/-- Calculates the number drawn in a specific group -/
def SystematicSampling.numberDrawnInGroup (s : SystematicSampling) (groupNumber : Nat) : Nat :=
  let groupSize := s.populationSize / s.groupCount
  let baseNumber := (groupNumber - 1) * groupSize
  baseNumber + (s.firstDrawn + groupNumber - 1) % 10

theorem seventh_group_draw (s : SystematicSampling) 
  (h1 : s.populationSize = 100)
  (h2 : s.groupCount = 10)
  (h3 : s.sampleSize = 10)
  (h4 : s.firstDrawn = 6) :
  s.numberDrawnInGroup 7 = 63 := by
  sorry

#check seventh_group_draw

end NUMINAMATH_CALUDE_seventh_group_draw_l2130_213087


namespace NUMINAMATH_CALUDE_combined_weight_is_1170_l2130_213037

/-- The weight Tony can lift in "the curl" exercise -/
def curl_weight : ℝ := 90

/-- The weight Tony can lift in "the military press" exercise -/
def military_press_weight : ℝ := 2 * curl_weight

/-- The weight Tony can lift in "the squat" exercise -/
def squat_weight : ℝ := 5 * military_press_weight

/-- The weight Tony can lift in "the bench press" exercise -/
def bench_press_weight : ℝ := 1.5 * military_press_weight

/-- The combined weight Tony can lift in the squat and bench press exercises -/
def combined_weight : ℝ := squat_weight + bench_press_weight

theorem combined_weight_is_1170 : combined_weight = 1170 := by
  sorry

end NUMINAMATH_CALUDE_combined_weight_is_1170_l2130_213037


namespace NUMINAMATH_CALUDE_pirate_gold_distribution_l2130_213026

theorem pirate_gold_distribution (total : ℕ) (jack jimmy tom sanji : ℕ) : 
  total = 280 ∧ 
  jimmy = jack + 11 ∧ 
  tom = jack - 15 ∧ 
  sanji = jack + 20 ∧ 
  total = jack + jimmy + tom + sanji → 
  sanji = 86 := by
sorry

end NUMINAMATH_CALUDE_pirate_gold_distribution_l2130_213026


namespace NUMINAMATH_CALUDE_nina_money_theorem_l2130_213038

theorem nina_money_theorem (x : ℝ) (h1 : 10 * x = 14 * (x - 3)) : 10 * x = 105 := by
  sorry

end NUMINAMATH_CALUDE_nina_money_theorem_l2130_213038


namespace NUMINAMATH_CALUDE_f_neg_one_gt_f_two_l2130_213045

-- Define f as a function from real numbers to real numbers
variable (f : ℝ → ℝ)

-- Condition 1: y = f(x+1) is an even function
def is_even_shifted (f : ℝ → ℝ) : Prop :=
  ∀ x, f (1 + x) = f (1 - x)

-- Condition 2: f(x) is an increasing function on the interval [1, +∞)
def is_increasing_on_interval (f : ℝ → ℝ) : Prop :=
  ∀ x y, 1 ≤ x → x < y → f x < f y

-- Theorem statement
theorem f_neg_one_gt_f_two 
  (h1 : is_even_shifted f) 
  (h2 : is_increasing_on_interval f) : 
  f (-1) > f 2 := by
  sorry

end NUMINAMATH_CALUDE_f_neg_one_gt_f_two_l2130_213045


namespace NUMINAMATH_CALUDE_equation_solutions_count_l2130_213091

theorem equation_solutions_count : 
  ∃! (s : Finset ℝ), (∀ x ∈ s, (x^2 - 5)^2 = 36) ∧ s.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_count_l2130_213091


namespace NUMINAMATH_CALUDE_kanga_lands_on_84_l2130_213039

def jump_sequence (n : ℕ) : ℕ :=
  9 * n

def kanga_position (n : ℕ) (extra_jumps : ℕ) : ℕ :=
  jump_sequence n + 
  if extra_jumps ≤ 2 then 3 * extra_jumps
  else 6 + (extra_jumps - 2)

theorem kanga_lands_on_84 : 
  ∃ (n : ℕ) (extra_jumps : ℕ), 
    kanga_position n extra_jumps = 84 ∧ 
    kanga_position n extra_jumps ≠ 82 ∧
    kanga_position n extra_jumps ≠ 83 ∧
    kanga_position n extra_jumps ≠ 85 ∧
    kanga_position n extra_jumps ≠ 86 :=
by sorry

end NUMINAMATH_CALUDE_kanga_lands_on_84_l2130_213039


namespace NUMINAMATH_CALUDE_ribbon_gap_theorem_l2130_213058

theorem ribbon_gap_theorem (R : ℝ) (h : R > 0) :
  let original_length := 2 * Real.pi * R
  let new_length := original_length + 1
  let new_radius := R + (new_length / (2 * Real.pi) - R)
  new_radius - R = 1 / (2 * Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_ribbon_gap_theorem_l2130_213058


namespace NUMINAMATH_CALUDE_gcd_problem_l2130_213043

theorem gcd_problem (b : ℤ) (h : 345 ∣ b) :
  Nat.gcd (5*b^3 + 2*b^2 + 7*b + 69).natAbs b.natAbs = 69 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l2130_213043


namespace NUMINAMATH_CALUDE_root_of_cubic_polynomials_l2130_213062

theorem root_of_cubic_polynomials (a b c d k : ℂ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (hk1 : a * k^3 + b * k^2 + c * k + d = 0)
  (hk2 : b * k^3 + c * k^2 + d * k + a = 0) :
  k = 1 ∨ k = -1 ∨ k = Complex.I ∨ k = -Complex.I :=
by sorry

end NUMINAMATH_CALUDE_root_of_cubic_polynomials_l2130_213062


namespace NUMINAMATH_CALUDE_inequality_solution_l2130_213046

theorem inequality_solution (x : ℝ) :
  (x^2 + 2*x - 15) / (x + 5) < 0 ↔ -5 < x ∧ x < 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2130_213046


namespace NUMINAMATH_CALUDE_rth_term_is_8r_l2130_213075

-- Define the sum of n terms for the arithmetic progression
def S (n : ℕ) : ℕ := 5 * n + 4 * n^2 + 1

-- Define the r-th term of the arithmetic progression
def a (r : ℕ) : ℕ := S r - S (r - 1)

-- Theorem stating that the r-th term is equal to 8r
theorem rth_term_is_8r (r : ℕ) : a r = 8 * r := by
  sorry

end NUMINAMATH_CALUDE_rth_term_is_8r_l2130_213075


namespace NUMINAMATH_CALUDE_paint_one_third_square_l2130_213029

theorem paint_one_third_square (n : ℕ) (k : ℕ) : n = 18 ∧ k = 6 →
  Nat.choose n k = 18564 := by
  sorry

end NUMINAMATH_CALUDE_paint_one_third_square_l2130_213029


namespace NUMINAMATH_CALUDE_meeting_day_is_thursday_l2130_213079

-- Define the days of the week
inductive Day : Type
  | Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

-- Define a function to determine if Joãozinho lies on a given day
def lies_on_day (d : Day) : Prop :=
  d = Day.Tuesday ∨ d = Day.Thursday ∨ d = Day.Saturday

-- Define a function to get the next day
def next_day (d : Day) : Day :=
  match d with
  | Day.Sunday => Day.Monday
  | Day.Monday => Day.Tuesday
  | Day.Tuesday => Day.Wednesday
  | Day.Wednesday => Day.Thursday
  | Day.Thursday => Day.Friday
  | Day.Friday => Day.Saturday
  | Day.Saturday => Day.Sunday

-- Theorem statement
theorem meeting_day_is_thursday :
  ∀ (d : Day),
    lies_on_day d →
    (lies_on_day d → d ≠ Day.Saturday) →
    (lies_on_day d → next_day d ≠ Day.Wednesday) →
    d = Day.Thursday :=
by
  sorry


end NUMINAMATH_CALUDE_meeting_day_is_thursday_l2130_213079


namespace NUMINAMATH_CALUDE_age_determination_l2130_213095

def binary_sum (n : ℕ) : Prop :=
  ∃ (a b c d : Bool),
    n = (if a then 1 else 0) + 
        (if b then 2 else 0) + 
        (if c then 4 else 0) + 
        (if d then 8 else 0)

theorem age_determination (n : ℕ) (h : n < 16) : binary_sum n := by
  sorry

end NUMINAMATH_CALUDE_age_determination_l2130_213095


namespace NUMINAMATH_CALUDE_surface_area_unchanged_l2130_213086

/-- The surface area of a cube with corner cubes removed -/
def surface_area_with_corners_removed (cube_side_length : ℝ) (corner_side_length : ℝ) : ℝ :=
  6 * cube_side_length^2

/-- The theorem stating that the surface area remains unchanged -/
theorem surface_area_unchanged (cube_side_length : ℝ) (corner_side_length : ℝ) 
  (h1 : cube_side_length = 5) 
  (h2 : corner_side_length = 2) : 
  surface_area_with_corners_removed cube_side_length corner_side_length = 150 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_unchanged_l2130_213086


namespace NUMINAMATH_CALUDE_P_sufficient_not_necessary_for_Q_l2130_213016

theorem P_sufficient_not_necessary_for_Q :
  (∀ a : ℝ, a > 1 → (a - 1) * (a + 1) > 0) ∧
  (∃ a : ℝ, (a - 1) * (a + 1) > 0 ∧ ¬(a > 1)) := by
  sorry

end NUMINAMATH_CALUDE_P_sufficient_not_necessary_for_Q_l2130_213016


namespace NUMINAMATH_CALUDE_abc_sum_mod_11_l2130_213022

theorem abc_sum_mod_11 (a b c : Nat) : 
  a < 11 → b < 11 → c < 11 → a > 0 → b > 0 → c > 0 →
  (a * b * c) % 11 = 3 →
  (8 * c) % 11 = 5 →
  (a + 3 * b) % 11 = 10 →
  (a + b + c) % 11 = 0 := by
  sorry

end NUMINAMATH_CALUDE_abc_sum_mod_11_l2130_213022


namespace NUMINAMATH_CALUDE_inequalities_hold_l2130_213025

theorem inequalities_hold (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 2) :
  ab ≤ 1 ∧ Real.sqrt a + Real.sqrt b ≤ 2 ∧ a^2 + b^2 ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequalities_hold_l2130_213025


namespace NUMINAMATH_CALUDE_jacobs_graham_crackers_l2130_213069

/-- Represents the number of graham crackers needed for one s'more -/
def graham_crackers_per_smore : ℕ := 2

/-- Represents the number of marshmallows needed for one s'more -/
def marshmallows_per_smore : ℕ := 1

/-- Represents the number of marshmallows Jacob currently has -/
def current_marshmallows : ℕ := 6

/-- Represents the number of additional marshmallows Jacob needs to buy -/
def additional_marshmallows : ℕ := 18

/-- Theorem stating the number of graham crackers Jacob has -/
theorem jacobs_graham_crackers :
  (current_marshmallows + additional_marshmallows) * graham_crackers_per_smore = 48 := by
  sorry

end NUMINAMATH_CALUDE_jacobs_graham_crackers_l2130_213069


namespace NUMINAMATH_CALUDE_irene_age_is_46_l2130_213033

-- Define the ages as natural numbers
def eddie_age : ℕ := 92
def becky_age : ℕ := eddie_age / 4
def irene_age : ℕ := 2 * becky_age

-- Theorem statement
theorem irene_age_is_46 : irene_age = 46 := by
  sorry

end NUMINAMATH_CALUDE_irene_age_is_46_l2130_213033


namespace NUMINAMATH_CALUDE_april_order_proof_l2130_213048

/-- The number of cases of soda ordered in April -/
def april_cases : ℕ := sorry

/-- The number of cases of soda ordered in May -/
def may_cases : ℕ := 30

/-- The number of bottles per case -/
def bottles_per_case : ℕ := 20

/-- The total number of bottles ordered in April and May -/
def total_bottles : ℕ := 1000

theorem april_order_proof :
  april_cases = 20 ∧
  april_cases * bottles_per_case + may_cases * bottles_per_case = total_bottles :=
by sorry

end NUMINAMATH_CALUDE_april_order_proof_l2130_213048


namespace NUMINAMATH_CALUDE_parabola_h_value_l2130_213066

/-- Represents a parabola of the form y = a(x-h)^2 + k -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Checks if a point (x, y) lies on the parabola -/
def Parabola.contains (p : Parabola) (x y : ℝ) : Prop :=
  y = p.a * (x - p.h)^2 + p.k

theorem parabola_h_value (p : Parabola) :
  p.a < 0 →
  0 < p.h →
  p.h < 6 →
  p.contains 0 4 →
  p.contains 6 5 →
  p.h = 4 := by
  sorry

#check parabola_h_value

end NUMINAMATH_CALUDE_parabola_h_value_l2130_213066


namespace NUMINAMATH_CALUDE_division_problem_l2130_213070

theorem division_problem (dividend quotient remainder : ℕ) (divisor : ℕ) :
  dividend = 760 →
  quotient = 21 →
  remainder = 4 →
  dividend = divisor * quotient + remainder →
  divisor = 36 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l2130_213070


namespace NUMINAMATH_CALUDE_arctan_arcsin_arccos_sum_l2130_213005

theorem arctan_arcsin_arccos_sum : Real.arctan (Real.sqrt 3 / 3) + Real.arcsin (-1/2) + Real.arccos 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_arctan_arcsin_arccos_sum_l2130_213005


namespace NUMINAMATH_CALUDE_amy_money_left_l2130_213017

/-- Calculates the amount of money Amy had when she left the fair. -/
def money_left (initial_amount spent : ℕ) : ℕ :=
  initial_amount - spent

/-- Proves that Amy had $11 when she left the fair. -/
theorem amy_money_left :
  money_left 15 4 = 11 := by
  sorry

end NUMINAMATH_CALUDE_amy_money_left_l2130_213017


namespace NUMINAMATH_CALUDE_union_A_B_when_a_is_one_intersection_A_B_empty_iff_a_leq_neg_three_or_geq_three_l2130_213044

-- Define set A
def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < a + 2}

-- Define set B (domain of the function)
def B : Set ℝ := {x | -1 < x ∧ x < 2}

-- Theorem 1
theorem union_A_B_when_a_is_one :
  A 1 ∪ B = {x | -1 < x ∧ x < 3} := by sorry

-- Theorem 2
theorem intersection_A_B_empty_iff_a_leq_neg_three_or_geq_three (a : ℝ) :
  A a ∩ B = ∅ ↔ a ≤ -3 ∨ a ≥ 3 := by sorry

end NUMINAMATH_CALUDE_union_A_B_when_a_is_one_intersection_A_B_empty_iff_a_leq_neg_three_or_geq_three_l2130_213044


namespace NUMINAMATH_CALUDE_worker_y_fraction_l2130_213068

theorem worker_y_fraction (fx fy : ℝ) : 
  fx + fy = 1 →
  0.005 * fx + 0.008 * fy = 0.0074 →
  fy = 0.8 := by
sorry

end NUMINAMATH_CALUDE_worker_y_fraction_l2130_213068


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l2130_213050

/-- Given a geometric sequence of positive integers where the first term is 5 and the fourth term is 500,
    prove that the third term is equal to 5 * 100^(2/3). -/
theorem geometric_sequence_third_term :
  ∀ (seq : ℕ → ℕ),
    (∀ n, seq (n + 1) / seq n = seq 2 / seq 1) →  -- Geometric sequence condition
    seq 1 = 5 →                                   -- First term is 5
    seq 4 = 500 →                                 -- Fourth term is 500
    seq 3 = 5 * 100^(2/3) :=
by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_third_term_l2130_213050


namespace NUMINAMATH_CALUDE_sales_problem_l2130_213015

-- Define the sales revenue function
def sales_revenue (x : ℝ) : ℝ := 1000 * x

-- Define the sales cost function
def sales_cost (x : ℝ) : ℝ := 500 * x + 2000

-- State the theorem
theorem sales_problem :
  -- Condition 1: When x = 0, sales cost is 2000
  sales_cost 0 = 2000 ∧
  -- Condition 2: When x = 2, sales revenue is 2000 and sales cost is 3000
  sales_revenue 2 = 2000 ∧ sales_cost 2 = 3000 ∧
  -- Condition 3: Sales revenue is directly proportional to x (already satisfied by definition)
  -- Condition 4: Sales cost is a linear function of x (already satisfied by definition)
  -- Proof goals:
  -- 1. The functions satisfy all conditions (implicitly proved by the above)
  -- 2. Sales revenue equals sales cost at 4 tons
  (∃ x : ℝ, x = 4 ∧ sales_revenue x = sales_cost x) ∧
  -- 3. Profit at 10 tons is 3000 yuan
  sales_revenue 10 - sales_cost 10 = 3000 :=
by sorry

end NUMINAMATH_CALUDE_sales_problem_l2130_213015


namespace NUMINAMATH_CALUDE_fraction_equality_l2130_213063

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h : (x^2 + 4*x*y) / (y^2 - 4*x*y) = 3) :
  ∃ z : ℝ, (x^2 - 4*x*y) / (y^2 + 4*x*y) = z :=
by sorry

end NUMINAMATH_CALUDE_fraction_equality_l2130_213063


namespace NUMINAMATH_CALUDE_relationship_of_values_l2130_213041

/-- An odd function f defined on ℝ satisfying f(x) + xf'(x) < 0 for x < 0 -/
class OddDecreasingFunction (f : ℝ → ℝ) : Prop where
  odd : ∀ x, f (-x) = -f x
  decreasing : ∀ x < 0, f x + x * (deriv f x) < 0

/-- The main theorem stating the relationship between πf(π), (-2)f(-2), and f(1) -/
theorem relationship_of_values (f : ℝ → ℝ) [OddDecreasingFunction f] :
  π * f π > (-2) * f (-2) ∧ (-2) * f (-2) > f 1 := by
  sorry

end NUMINAMATH_CALUDE_relationship_of_values_l2130_213041
