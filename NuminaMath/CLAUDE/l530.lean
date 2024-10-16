import Mathlib

namespace NUMINAMATH_CALUDE_comet_orbit_equation_l530_53005

/-- Represents an elliptical orbit -/
structure EllipticalOrbit where
  perihelion : ℝ
  aphelion : ℝ

/-- The equation of an ellipse -/
def is_ellipse_equation (a b : ℝ) (eq : ℝ × ℝ → Prop) : Prop :=
  ∀ x y, eq (x, y) ↔ x^2 / a^2 + y^2 / b^2 = 1

/-- Theorem stating the equation of the comet's orbit -/
theorem comet_orbit_equation (orbit : EllipticalOrbit) 
  (h_perihelion : orbit.perihelion = 2)
  (h_aphelion : orbit.aphelion = 6) :
  (∃ eq : ℝ × ℝ → Prop, is_ellipse_equation 4 (12: ℝ).sqrt eq) ∨
  (∃ eq : ℝ × ℝ → Prop, is_ellipse_equation (12: ℝ).sqrt 4 eq) :=
sorry

end NUMINAMATH_CALUDE_comet_orbit_equation_l530_53005


namespace NUMINAMATH_CALUDE_monotonicity_depends_on_a_l530_53083

/-- The function f(x) = x³ + ax² + 1 where a is a real number -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 1

/-- The derivative of f(x) with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x

/-- Theorem stating that the monotonicity of f depends on the value of a -/
theorem monotonicity_depends_on_a :
  ∀ a : ℝ, ∃ x y : ℝ, x < y ∧
    ((f_derivative a x > 0 ∧ f_derivative a y < 0) ∨
     (f_derivative a x < 0 ∧ f_derivative a y > 0) ∨
     (∀ z : ℝ, f_derivative a z ≥ 0) ∨
     (∀ z : ℝ, f_derivative a z ≤ 0)) :=
by sorry

end NUMINAMATH_CALUDE_monotonicity_depends_on_a_l530_53083


namespace NUMINAMATH_CALUDE_not_perfect_power_probability_l530_53004

/-- A function that determines if a number is a perfect power --/
def isPerfectPower (n : ℕ) : Prop :=
  ∃ (x y : ℕ), y > 1 ∧ x^y = n

/-- The count of numbers from 1 to 200 that are perfect powers --/
def perfectPowerCount : ℕ := 22

/-- The probability of selecting a number that is not a perfect power --/
def probabilityNotPerfectPower : ℚ := 89 / 100

theorem not_perfect_power_probability :
  (200 - perfectPowerCount : ℚ) / 200 = probabilityNotPerfectPower :=
sorry

end NUMINAMATH_CALUDE_not_perfect_power_probability_l530_53004


namespace NUMINAMATH_CALUDE_fuel_distribution_l530_53035

def total_fuel : ℝ := 60

theorem fuel_distribution (second_third : ℝ) (final_third : ℝ) 
  (h1 : second_third = total_fuel / 3)
  (h2 : final_third = second_third / 2)
  (h3 : second_third + final_third + (total_fuel - second_third - final_third) = total_fuel) :
  total_fuel - second_third - final_third = 30 := by
sorry

end NUMINAMATH_CALUDE_fuel_distribution_l530_53035


namespace NUMINAMATH_CALUDE_difference_23rd_21st_triangular_l530_53017

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem difference_23rd_21st_triangular : 
  triangular_number 23 - triangular_number 21 = 45 := by
  sorry

end NUMINAMATH_CALUDE_difference_23rd_21st_triangular_l530_53017


namespace NUMINAMATH_CALUDE_desk_chair_relationship_l530_53067

def chair_heights : List ℝ := [37.0, 40.0, 42.0, 45.0]
def desk_heights : List ℝ := [70.0, 74.8, 78.0, 82.8]

def linear_function (x : ℝ) : ℝ := 1.6 * x + 10.8

theorem desk_chair_relationship :
  ∀ (i : Fin 4),
    linear_function (chair_heights.get i) = desk_heights.get i :=
by sorry

end NUMINAMATH_CALUDE_desk_chair_relationship_l530_53067


namespace NUMINAMATH_CALUDE_binary_expression_equals_expected_result_l530_53069

/-- Converts a list of binary digits to a natural number. -/
def binary_to_nat (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 2 * acc + d) 0

/-- Calculates the result of the given binary expression. -/
def binary_expression_result : Nat :=
  let a := binary_to_nat [1, 0, 1, 1, 0]
  let b := binary_to_nat [1, 0, 1, 0]
  let c := binary_to_nat [1, 1, 1, 0, 0]
  let d := binary_to_nat [1, 1, 1, 0]
  a + b - c + d

/-- The expected result in binary. -/
def expected_result : Nat :=
  binary_to_nat [0, 1, 1, 1, 0]

theorem binary_expression_equals_expected_result :
  binary_expression_result = expected_result := by
  sorry

end NUMINAMATH_CALUDE_binary_expression_equals_expected_result_l530_53069


namespace NUMINAMATH_CALUDE_positive_root_of_cubic_l530_53029

theorem positive_root_of_cubic (x : ℝ) : 
  x = 3 - Real.sqrt 3 → x > 0 ∧ x^3 - 4*x^2 - 2*x - Real.sqrt 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_root_of_cubic_l530_53029


namespace NUMINAMATH_CALUDE_diplomats_conference_l530_53068

theorem diplomats_conference (D : ℕ) : 
  (20 : ℕ) ≤ D ∧  -- Number of diplomats who spoke Japanese
  (32 : ℕ) ≤ D ∧  -- Number of diplomats who did not speak Russian
  (D - (20 + (D - 32) - (D / 10 : ℕ)) : ℤ) = (D / 5 : ℕ) ∧  -- 20% spoke neither Japanese nor Russian
  (D / 10 : ℕ) ≤ 20  -- 10% spoke both Japanese and Russian (this must be ≤ 20)
  → D = 40 := by
sorry

end NUMINAMATH_CALUDE_diplomats_conference_l530_53068


namespace NUMINAMATH_CALUDE_liars_guessing_game_theorem_l530_53025

/-- The liar's guessing game -/
structure LiarsGuessingGame where
  k : ℕ+  -- The number of consecutive answers where at least one must be truthful
  n : ℕ+  -- The maximum size of the final guessing set

/-- A winning strategy for player B -/
def has_winning_strategy (game : LiarsGuessingGame) : Prop :=
  ∀ N : ℕ+, ∃ (strategy : ℕ+ → Finset ℕ+), 
    (∀ x : ℕ+, x ≤ N → x ∈ strategy N) ∧
    (Finset.card (strategy N) ≤ game.n)

/-- Main theorem about the liar's guessing game -/
theorem liars_guessing_game_theorem (game : LiarsGuessingGame) :
  (game.n ≥ 2^(game.k : ℕ) → has_winning_strategy game) ∧
  (∃ k : ℕ+, ∃ n : ℕ+, n ≥ (1.99 : ℝ)^(k : ℕ) ∧ 
    ¬(has_winning_strategy ⟨k, n⟩)) := by
  sorry

end NUMINAMATH_CALUDE_liars_guessing_game_theorem_l530_53025


namespace NUMINAMATH_CALUDE_circle_covering_theorem_l530_53073

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a circle
structure Circle where
  center : Point
  radius : ℝ

-- Define a function to check if a point is inside or on a circle
def pointInCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 ≤ c.radius^2

-- Main theorem
theorem circle_covering_theorem 
  (points : Finset Point) 
  (outer_circle : Circle) 
  (h1 : outer_circle.radius = 2)
  (h2 : points.card = 15)
  (h3 : ∀ p ∈ points, pointInCircle p outer_circle) :
  ∃ (inner_circle : Circle), 
    inner_circle.radius = 1 ∧ 
    (∃ (subset : Finset Point), subset ⊆ points ∧ subset.card ≥ 3 ∧ 
      ∀ p ∈ subset, pointInCircle p inner_circle) := by
  sorry

end NUMINAMATH_CALUDE_circle_covering_theorem_l530_53073


namespace NUMINAMATH_CALUDE_jack_initial_money_l530_53021

def initial_bottles : ℕ := 4
def bottle_cost : ℚ := 2
def cheese_weight : ℚ := 1/2
def cheese_cost_per_pound : ℚ := 10
def remaining_money : ℚ := 71

theorem jack_initial_money :
  let total_bottles := initial_bottles + 2 * initial_bottles
  let water_cost := total_bottles * bottle_cost
  let cheese_cost := cheese_weight * cheese_cost_per_pound
  let total_spent := water_cost + cheese_cost
  total_spent + remaining_money = 100 := by sorry

end NUMINAMATH_CALUDE_jack_initial_money_l530_53021


namespace NUMINAMATH_CALUDE_johns_apartment_paint_area_l530_53022

/-- Represents the dimensions of a bedroom -/
structure BedroomDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the total area to be painted in a single bedroom -/
def area_to_paint (dim : BedroomDimensions) (unpainted_area : ℝ) : ℝ :=
  2 * (dim.length * dim.height + dim.width * dim.height) + 
  dim.length * dim.width - unpainted_area

/-- Theorem stating the total area to be painted in John's apartment -/
theorem johns_apartment_paint_area :
  let bedroom_dim : BedroomDimensions := ⟨15, 12, 10⟩
  let unpainted_area : ℝ := 70
  let num_bedrooms : ℕ := 2
  num_bedrooms * (area_to_paint bedroom_dim unpainted_area) = 1300 := by
  sorry


end NUMINAMATH_CALUDE_johns_apartment_paint_area_l530_53022


namespace NUMINAMATH_CALUDE_z_imaginary_and_fourth_quadrant_l530_53046

def z (m : ℝ) : ℂ := m * (m + 2) + (m^2 + m - 2) * Complex.I

theorem z_imaginary_and_fourth_quadrant (m : ℝ) :
  (z m = Complex.I * Complex.im (z m) → m = 0) ∧
  (Complex.re (z m) > 0 ∧ Complex.im (z m) < 0 → 0 < m ∧ m < 1) :=
sorry

end NUMINAMATH_CALUDE_z_imaginary_and_fourth_quadrant_l530_53046


namespace NUMINAMATH_CALUDE_integral_proof_l530_53061

open Real

noncomputable def f (x : ℝ) : ℝ := 
  -20/27 * ((1 + x^(3/4))^(1/5) / x^(3/20))^9

theorem integral_proof (x : ℝ) (h : x > 0) : 
  deriv f x = (((1 + x^(3/4))^4)^(1/5)) / (x^2 * x^(7/20)) :=
by sorry

end NUMINAMATH_CALUDE_integral_proof_l530_53061


namespace NUMINAMATH_CALUDE_polynomial_factorization_l530_53011

theorem polynomial_factorization (p q : ℕ) (n : ℕ) (a : ℤ) :
  Prime p ∧ Prime q ∧ p ≠ q ∧ n ≥ 3 →
  (∃ (g h : Polynomial ℤ),
    (Polynomial.degree g > 0) ∧
    (Polynomial.degree h > 0) ∧
    (X^n + a * X^(n-1) + (p * q : ℤ) = g * h)) ↔
  (a = (-1)^n * (p * q : ℤ) + 1 ∨ a = -(p * q : ℤ) - 1) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l530_53011


namespace NUMINAMATH_CALUDE_slope_angle_sqrt3_l530_53036

/-- The slope angle of the line y = √3x + 1 is 60° -/
theorem slope_angle_sqrt3 : 
  let l : ℝ → ℝ := λ x => Real.sqrt 3 * x + 1
  ∃ θ : ℝ, θ = 60 * π / 180 ∧ Real.tan θ = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_slope_angle_sqrt3_l530_53036


namespace NUMINAMATH_CALUDE_ab_value_l530_53093

theorem ab_value (a b : ℕ+) (h1 : a + b = 30) (h2 : 3 * a * b + 5 * a = 4 * b + 180) : a * b = 29 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l530_53093


namespace NUMINAMATH_CALUDE_parabola_symmetric_line_l530_53008

/-- The parabola that intersects the x-axis at only one point -/
def parabola (x y a : ℝ) : Prop := y = 2 * x^2 + 4 * x + 5 - a

/-- The condition that the parabola intersects the x-axis at only one point -/
def single_intersection (a : ℝ) : Prop := 4^2 - 4 * 2 * (5 - a) = 0

/-- The axis of symmetry of the parabola -/
def axis_of_symmetry : ℝ := -1

/-- The x-coordinate of point A -/
def point_A_x : ℝ := -1

/-- The y-coordinate of point A -/
def point_A_y : ℝ := 0

/-- The x-coordinate of point B -/
def point_B_x : ℝ := 0

/-- The y-coordinate of point B -/
def point_B_y : ℝ := 2

/-- The equation of the symmetric line -/
def symmetric_line (x y : ℝ) : Prop := y = -2 * x - 2

theorem parabola_symmetric_line :
  ∀ a : ℝ, single_intersection a →
  ∃ x y : ℝ, parabola x y a ∧ symmetric_line x y :=
by sorry

end NUMINAMATH_CALUDE_parabola_symmetric_line_l530_53008


namespace NUMINAMATH_CALUDE_lunch_cost_proof_l530_53034

theorem lunch_cost_proof (adam_cost rick_cost jose_cost : ℝ) :
  adam_cost = (2/3) * rick_cost →
  rick_cost = jose_cost →
  jose_cost = 45 →
  adam_cost + rick_cost + jose_cost = 120 := by
sorry

end NUMINAMATH_CALUDE_lunch_cost_proof_l530_53034


namespace NUMINAMATH_CALUDE_length_of_EF_l530_53066

/-- A rectangle intersecting a circle -/
structure RectangleIntersectingCircle where
  /-- Length of AB -/
  AB : ℝ
  /-- Length of BC -/
  BC : ℝ
  /-- Length of DE -/
  DE : ℝ
  /-- Length of EF -/
  EF : ℝ

/-- Theorem stating the length of EF in the given configuration -/
theorem length_of_EF (r : RectangleIntersectingCircle) 
  (h1 : r.AB = 4)
  (h2 : r.BC = 5)
  (h3 : r.DE = 3) :
  r.EF = 7 := by
  sorry

#check length_of_EF

end NUMINAMATH_CALUDE_length_of_EF_l530_53066


namespace NUMINAMATH_CALUDE_inverse_proportion_points_order_l530_53013

theorem inverse_proportion_points_order (x₁ x₂ x₃ : ℝ) :
  x₁ ≠ 0 ∧ x₂ ≠ 0 ∧ x₃ ≠ 0 →
  -4 / x₁ = -1 →
  -4 / x₂ = 3 →
  -4 / x₃ = 5 →
  x₂ < x₃ ∧ x₃ < x₁ :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_points_order_l530_53013


namespace NUMINAMATH_CALUDE_two_digit_number_patterns_l530_53088

theorem two_digit_number_patterns 
  (a m n : ℕ) 
  (ha : 0 < a ∧ a < 10) 
  (hm : 0 < m ∧ m < 10) 
  (hn : 0 < n ∧ n < 10) : 
  ((10 * a + 5) ^ 2 = 100 * a * (a + 1) + 25) ∧ 
  ((10 * m + n) * (10 * m + (10 - n)) = 100 * m * (m + 1) + n * (10 - n)) :=
by sorry

end NUMINAMATH_CALUDE_two_digit_number_patterns_l530_53088


namespace NUMINAMATH_CALUDE_function_equation_solution_l530_53030

theorem function_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, (f x + x * y) * f (x - 3 * y) + (f y + x * y) * f (3 * x - y) = (f (x + y))^2) →
  (∀ x : ℝ, f x = 0 ∨ f x = x^2) :=
by sorry

end NUMINAMATH_CALUDE_function_equation_solution_l530_53030


namespace NUMINAMATH_CALUDE_second_division_count_correct_l530_53055

/-- Represents the number of people in the second division of money -/
def second_division_count : ℕ → Prop := λ x =>
  x > 6 ∧ (90 : ℚ) / (x - 6 : ℚ) = 120 / x

/-- The theorem stating the condition for the correct number of people in the second division -/
theorem second_division_count_correct (x : ℕ) : 
  second_division_count x ↔ 
    (∃ (y : ℕ), y > 0 ∧ 
      (90 : ℚ) / y = (120 : ℚ) / (y + 6) ∧
      x = y + 6) :=
sorry

end NUMINAMATH_CALUDE_second_division_count_correct_l530_53055


namespace NUMINAMATH_CALUDE_allison_bought_28_items_l530_53033

/-- The number of craft supply items Allison bought -/
def allison_total (marie_glue : ℕ) (marie_paper : ℕ) (glue_diff : ℕ) (paper_ratio : ℕ) : ℕ :=
  (marie_glue + glue_diff) + (marie_paper / paper_ratio)

/-- Theorem stating the total number of craft supply items Allison bought -/
theorem allison_bought_28_items : allison_total 15 30 8 6 = 28 := by
  sorry

end NUMINAMATH_CALUDE_allison_bought_28_items_l530_53033


namespace NUMINAMATH_CALUDE_power_product_eq_product_of_powers_l530_53031

theorem power_product_eq_product_of_powers (a b : ℝ) : (a * b)^2 = a^2 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_power_product_eq_product_of_powers_l530_53031


namespace NUMINAMATH_CALUDE_shortest_distance_parabola_to_line_l530_53057

/-- The shortest distance between a point on the parabola y = x^2 - 9x + 25 
    and a point on the line y = x - 8 is 4√2. -/
theorem shortest_distance_parabola_to_line :
  let parabola := {(x, y) : ℝ × ℝ | y = x^2 - 9*x + 25}
  let line := {(x, y) : ℝ × ℝ | y = x - 8}
  ∃ (d : ℝ), d = 4 * Real.sqrt 2 ∧ 
    ∀ (A : ℝ × ℝ) (B : ℝ × ℝ), A ∈ parabola → B ∈ line → 
      Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ≥ d :=
by sorry

end NUMINAMATH_CALUDE_shortest_distance_parabola_to_line_l530_53057


namespace NUMINAMATH_CALUDE_expansion_coefficient_zero_l530_53020

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := sorry

-- Define the coefficient function for the expansion of (1 - 1/x)(1+x)^5
def coefficient (r : ℤ) : ℚ :=
  if r = 2 then binomial 5 2 - binomial 5 3
  else if r = 1 then binomial 5 1 - binomial 5 2
  else if r = 0 then 1 - binomial 5 1
  else if r = -1 then -1
  else if r = 3 then binomial 5 3 - binomial 5 4
  else if r = 4 then binomial 5 4 - binomial 5 5
  else if r = 5 then binomial 5 5
  else 0

theorem expansion_coefficient_zero :
  ∃ (r : ℤ), r ∈ Set.Icc (-1 : ℤ) 5 ∧ coefficient r = 0 ∧ r = 2 :=
by sorry

end NUMINAMATH_CALUDE_expansion_coefficient_zero_l530_53020


namespace NUMINAMATH_CALUDE_isosceles_triangle_areas_sum_l530_53095

/-- Given a 6-8-10 right triangle, prove that the sum of the areas of right isosceles triangles
    constructed on the two shorter sides is equal to the area of the right isosceles triangle
    constructed on the hypotenuse. -/
theorem isosceles_triangle_areas_sum (a b c : ℝ) (h1 : a = 6) (h2 : b = 8) (h3 : c = 10)
  (h4 : a^2 + b^2 = c^2) : (1/2 * a^2) + (1/2 * b^2) = 1/2 * c^2 := by
  sorry

#check isosceles_triangle_areas_sum

end NUMINAMATH_CALUDE_isosceles_triangle_areas_sum_l530_53095


namespace NUMINAMATH_CALUDE_sum_of_digits_of_square_ones_l530_53047

def ones (n : ℕ) : ℕ := 
  (10^n - 1) / 9

def sum_of_digits (m : ℕ) : ℕ :=
  if m = 0 then 0 else m % 10 + sum_of_digits (m / 10)

theorem sum_of_digits_of_square_ones (n : ℕ) : 
  sum_of_digits ((ones n)^2) = n^2 :=
sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_square_ones_l530_53047


namespace NUMINAMATH_CALUDE_probability_two_kings_or_at_least_two_aces_l530_53001

def standard_deck : ℕ := 52
def num_aces : ℕ := 4
def num_kings : ℕ := 4
def cards_drawn : ℕ := 3

def prob_two_kings : ℚ := (Nat.choose num_kings 2 * Nat.choose (standard_deck - num_kings) 1) / Nat.choose standard_deck cards_drawn

def prob_two_aces : ℚ := (Nat.choose num_aces 2 * Nat.choose (standard_deck - num_aces) 1) / Nat.choose standard_deck cards_drawn

def prob_three_aces : ℚ := Nat.choose num_aces 3 / Nat.choose standard_deck cards_drawn

def prob_at_least_two_aces : ℚ := prob_two_aces + prob_three_aces

theorem probability_two_kings_or_at_least_two_aces :
  prob_two_kings + prob_at_least_two_aces = 1090482 / 40711175 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_kings_or_at_least_two_aces_l530_53001


namespace NUMINAMATH_CALUDE_boys_to_girls_ratio_l530_53099

theorem boys_to_girls_ratio 
  (T : ℕ) -- Total number of students
  (B : ℕ) -- Number of boys
  (G : ℕ) -- Number of girls
  (h1 : T = B + G) -- Total is sum of boys and girls
  (h2 : 2 * G = 3 * (T / 4)) -- 2/3 of girls = 1/4 of total
  : B * 3 = G * 5 := by
sorry

end NUMINAMATH_CALUDE_boys_to_girls_ratio_l530_53099


namespace NUMINAMATH_CALUDE_central_angles_sum_l530_53018

theorem central_angles_sum (x : ℝ) : 
  (6 * x + (7 * x + 10) + (2 * x + 10) + x = 360) → x = 21.25 := by
  sorry

end NUMINAMATH_CALUDE_central_angles_sum_l530_53018


namespace NUMINAMATH_CALUDE_expected_games_is_correct_l530_53039

/-- Represents the state of the game --/
inductive GameState
| Ongoing : ℕ → ℕ → GameState  -- Number of wins for player A and B
| Finished : GameState

/-- The probability of player A winning in an odd-numbered game --/
def prob_A_odd : ℚ := 3/5

/-- The probability of player B winning in an even-numbered game --/
def prob_B_even : ℚ := 3/5

/-- Determines if the game is finished based on the number of wins --/
def is_finished (wins_A wins_B : ℕ) : Bool :=
  (wins_A ≥ wins_B + 2) ∨ (wins_B ≥ wins_A + 2)

/-- Calculates the expected number of games until the match ends --/
noncomputable def expected_games : ℚ :=
  25/6

/-- Theorem stating that the expected number of games is 25/6 --/
theorem expected_games_is_correct : expected_games = 25/6 := by
  sorry

end NUMINAMATH_CALUDE_expected_games_is_correct_l530_53039


namespace NUMINAMATH_CALUDE_tan_product_eighths_of_pi_l530_53082

theorem tan_product_eighths_of_pi : 
  Real.tan (π / 8) * Real.tan (3 * π / 8) * Real.tan (5 * π / 8) * Real.tan (7 * π / 8) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_eighths_of_pi_l530_53082


namespace NUMINAMATH_CALUDE_problem_solution_l530_53044

theorem problem_solution :
  let x : ℝ := 88 * (1 + 0.25)
  let y : ℝ := 150 * (1 - 0.40)
  let z : ℝ := 60 * (1 + 0.15)
  (x + y + z = 269) ∧
  ((x * y * z) ^ (x - y) = (683100 : ℝ) ^ 20) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l530_53044


namespace NUMINAMATH_CALUDE_polynomial_constant_term_l530_53074

/-- A polynomial of degree 4 with integer coefficients -/
structure Polynomial4 where
  p : ℤ
  q : ℤ
  r : ℤ
  s : ℤ

/-- The polynomial g(x) = x^4 + px^3 + qx^2 + rx + s -/
def g (poly : Polynomial4) (x : ℤ) : ℤ :=
  x^4 + poly.p * x^3 + poly.q * x^2 + poly.r * x + poly.s

/-- A polynomial has all negative integer roots -/
def has_all_negative_integer_roots (poly : Polynomial4) : Prop :=
  ∃ (a b c d : ℤ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    ∀ (x : ℤ), g poly x = (x + a) * (x + b) * (x + c) * (x + d)

theorem polynomial_constant_term (poly : Polynomial4) :
  has_all_negative_integer_roots poly →
  poly.p + poly.q + poly.r + poly.s = 8091 →
  poly.s = 8064 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_constant_term_l530_53074


namespace NUMINAMATH_CALUDE_equation_represents_line_l530_53002

/-- The equation (2x + 3y - 1)(-1) = 0 represents a single straight line in the Cartesian plane. -/
theorem equation_represents_line : ∃ (a b c : ℝ) (h : (a, b) ≠ (0, 0)),
  ∀ (x y : ℝ), (2*x + 3*y - 1)*(-1) = 0 ↔ a*x + b*y + c = 0 :=
sorry

end NUMINAMATH_CALUDE_equation_represents_line_l530_53002


namespace NUMINAMATH_CALUDE_neither_sufficient_nor_necessary_l530_53014

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem neither_sufficient_nor_necessary (a b : V) : 
  ¬(∀ a b : V, ‖a‖ = ‖b‖ → ‖a + b‖ = ‖a - b‖) ∧ 
  ¬(∀ a b : V, ‖a + b‖ = ‖a - b‖ → ‖a‖ = ‖b‖) := by
  sorry

end NUMINAMATH_CALUDE_neither_sufficient_nor_necessary_l530_53014


namespace NUMINAMATH_CALUDE_equal_area_rectangles_l530_53096

/-- Given two rectangles with equal area, where one rectangle has dimensions 5 by 24 inches
    and the other has a length of 2 inches, prove that the width of the second rectangle is 60 inches. -/
theorem equal_area_rectangles (carol_length carol_width jordan_length : ℕ)
    (jordan_width : ℕ) (h1 : carol_length = 5) (h2 : carol_width = 24) (h3 : jordan_length = 2)
    (h4 : carol_length * carol_width = jordan_length * jordan_width) :
    jordan_width = 60 := by
  sorry

end NUMINAMATH_CALUDE_equal_area_rectangles_l530_53096


namespace NUMINAMATH_CALUDE_ice_cream_combinations_l530_53024

theorem ice_cream_combinations (n : ℕ) (k : ℕ) : 
  n = 5 → k = 3 → Nat.choose (n + k - 1) (k - 1) = 21 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_combinations_l530_53024


namespace NUMINAMATH_CALUDE_not_both_perfect_squares_l530_53064

theorem not_both_perfect_squares (x y z t : ℕ+) 
  (h1 : x.val * y.val - z.val * t.val = x.val + y.val)
  (h2 : x.val + y.val = z.val + t.val) :
  ¬(∃ (a c : ℕ), x.val * y.val = a^2 ∧ z.val * t.val = c^2) :=
by sorry

end NUMINAMATH_CALUDE_not_both_perfect_squares_l530_53064


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l530_53042

theorem complex_fraction_sum : 
  (481 + 1/6 : ℚ) + (265 + 1/12 : ℚ) + (904 + 1/20 : ℚ) - 
  (184 + 29/30 : ℚ) - (160 + 41/42 : ℚ) - (703 + 55/56 : ℚ) = 
  603 + 3/8 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l530_53042


namespace NUMINAMATH_CALUDE_unique_integer_triangle_l530_53012

/-- A triangle with integer sides and an altitude --/
structure IntegerTriangle where
  a : ℕ  -- side BC
  b : ℕ  -- side CA
  c : ℕ  -- side AB
  h : ℕ  -- altitude AD
  bd : ℕ -- length of BD
  dc : ℕ -- length of DC

/-- The triangle satisfies the given conditions --/
def satisfies_conditions (t : IntegerTriangle) : Prop :=
  ∃ (n : ℕ), t.h = n ∧ t.a = n + 1 ∧ t.b = n + 2 ∧ t.c = n + 3 ∧
  t.a^2 = t.bd^2 + t.h^2 ∧
  t.c^2 = (t.bd + t.dc)^2 + t.h^2

/-- The theorem stating the existence and uniqueness of the triangle --/
theorem unique_integer_triangle :
  ∃! (t : IntegerTriangle), satisfies_conditions t ∧ 
    t.a = 14 ∧ t.b = 13 ∧ t.c = 15 ∧ t.h = 12 :=
by sorry

end NUMINAMATH_CALUDE_unique_integer_triangle_l530_53012


namespace NUMINAMATH_CALUDE_x_minus_y_equals_twelve_l530_53080

theorem x_minus_y_equals_twelve (x y : ℕ) : 
  3^x * 4^y = 531441 → x = 12 → x - y = 12 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_twelve_l530_53080


namespace NUMINAMATH_CALUDE_stacy_homework_problem_l530_53059

/-- Represents the number of homework problems assigned by Stacy. -/
def homework_problems : ℕ → ℕ → ℕ → ℕ 
  | true_false, free_response, multiple_choice => 
    true_false + free_response + multiple_choice

theorem stacy_homework_problem :
  ∃ (true_false free_response multiple_choice : ℕ),
    true_false = 6 ∧
    free_response = true_false + 7 ∧
    multiple_choice = 2 * free_response ∧
    homework_problems true_false free_response multiple_choice = 45 :=
by
  sorry

#check stacy_homework_problem

end NUMINAMATH_CALUDE_stacy_homework_problem_l530_53059


namespace NUMINAMATH_CALUDE_sock_ratio_is_two_elevenths_l530_53037

/-- Represents the sock order problem -/
structure SockOrder where
  blackPairs : ℕ
  bluePairs : ℕ
  blackPrice : ℝ
  bluePrice : ℝ

/-- The original sock order -/
def originalOrder : SockOrder :=
  { blackPairs := 6,
    bluePairs := 0,  -- This will be determined
    blackPrice := 0, -- This will be determined
    bluePrice := 0   -- This will be determined
  }

/-- The interchanged sock order -/
def interchangedOrder (o : SockOrder) : SockOrder :=
  { blackPairs := o.bluePairs,
    bluePairs := o.blackPairs,
    blackPrice := o.blackPrice,
    bluePrice := o.bluePrice
  }

/-- Calculate the total cost of a sock order -/
def totalCost (o : SockOrder) : ℝ :=
  o.blackPairs * o.blackPrice + o.bluePairs * o.bluePrice

/-- The theorem stating the ratio of black to blue socks -/
theorem sock_ratio_is_two_elevenths :
  ∃ (o : SockOrder),
    o.blackPairs = 6 ∧
    o.blackPrice = 2 * o.bluePrice ∧
    totalCost (interchangedOrder o) = 1.6 * totalCost o ∧
    o.blackPairs / o.bluePairs = 2 / 11 :=
  sorry

end NUMINAMATH_CALUDE_sock_ratio_is_two_elevenths_l530_53037


namespace NUMINAMATH_CALUDE_area_ratio_in_regular_octagon_l530_53009

structure RegularOctagon where
  vertices : Fin 8 → Point

structure EquilateralTriangle where
  vertices : Fin 3 → Point

def RegularOctagon.smallTriangles (octagon : RegularOctagon) : Fin 8 → EquilateralTriangle :=
  sorry

def RegularOctagon.largeTriangle (octagon : RegularOctagon) : EquilateralTriangle :=
  sorry

def area (triangle : EquilateralTriangle) : ℝ := sorry

theorem area_ratio_in_regular_octagon (octagon : RegularOctagon) :
  let smallTriangles := octagon.smallTriangles
  let largeTriangle := octagon.largeTriangle
  let triangleABJ := smallTriangles 0
  (area triangleABJ) / (area largeTriangle) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_in_regular_octagon_l530_53009


namespace NUMINAMATH_CALUDE_inscribed_triangle_area_ratio_l530_53038

theorem inscribed_triangle_area_ratio (s : ℝ) (h : s > 0) :
  let square_area := s^2
  let triangle_base := s
  let triangle_height := s / 2
  let triangle_area := (triangle_base * triangle_height) / 2
  triangle_area / square_area = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_inscribed_triangle_area_ratio_l530_53038


namespace NUMINAMATH_CALUDE_smallest_lcm_with_gcd_5_l530_53085

theorem smallest_lcm_with_gcd_5 :
  ∃ (m n : ℕ), 
    1000 ≤ m ∧ m < 10000 ∧
    1000 ≤ n ∧ n < 10000 ∧
    Nat.gcd m n = 5 ∧
    Nat.lcm m n = 201000 ∧
    ∀ (p q : ℕ), 
      1000 ≤ p ∧ p < 10000 ∧
      1000 ≤ q ∧ q < 10000 ∧
      Nat.gcd p q = 5 →
      Nat.lcm p q ≥ 201000 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_lcm_with_gcd_5_l530_53085


namespace NUMINAMATH_CALUDE_tangent_circles_m_values_l530_53006

-- Define the equations of the circles
def C₁ (x y m : ℝ) : Prop := x^2 + y^2 - 2*m*x + 4*y + m^2 - 5 = 0
def C₂ (x y m : ℝ) : Prop := x^2 + y^2 + 2*x - 2*m*y + m^2 - 3 = 0

-- Define the condition for circles being tangent
def are_tangent (m : ℝ) : Prop :=
  ∃ x y, C₁ x y m ∧ C₂ x y m ∧
  (∀ x' y', C₁ x' y' m ∧ C₂ x' y' m → (x', y') = (x, y))

-- Theorem statement
theorem tangent_circles_m_values :
  {m : ℝ | are_tangent m} = {-5, -2, -1, 2} :=
sorry

end NUMINAMATH_CALUDE_tangent_circles_m_values_l530_53006


namespace NUMINAMATH_CALUDE_smallest_three_digit_non_divisor_l530_53098

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem smallest_three_digit_non_divisor : 
  ∀ n : ℕ, is_three_digit n → (n - 1) ∣ factorial n → n ≥ 1004 :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_non_divisor_l530_53098


namespace NUMINAMATH_CALUDE_alpha_squared_greater_than_beta_squared_l530_53084

theorem alpha_squared_greater_than_beta_squared
  (α β : ℝ)
  (h1 : α ∈ Set.Icc (-π/2) (π/2))
  (h2 : β ∈ Set.Icc (-π/2) (π/2))
  (h3 : α * Real.sin α - β * Real.sin β > 0) :
  α^2 > β^2 := by
  sorry

end NUMINAMATH_CALUDE_alpha_squared_greater_than_beta_squared_l530_53084


namespace NUMINAMATH_CALUDE_arithmetic_sequence_max_sum_l530_53094

/-- An arithmetic sequence with positive first term and a_3/a_4 = 7/5 reaches maximum sum at n = 6 -/
theorem arithmetic_sequence_max_sum (a : ℕ → ℝ) (d : ℝ) :
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence
  a 1 > 0 →  -- positive first term
  a 3 / a 4 = 7 / 5 →  -- given ratio
  ∃ S : ℕ → ℝ, ∀ n, S n = n * (2 * a 1 + (n - 1) * d) / 2 ∧  -- sum formula
  (∀ m, S m ≤ S 6) :=  -- maximum sum at n = 6
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_max_sum_l530_53094


namespace NUMINAMATH_CALUDE_sum_cube_inequality_l530_53016

theorem sum_cube_inequality (x1 x2 x3 x4 : ℝ) 
  (h_pos1 : x1 > 0) (h_pos2 : x2 > 0) (h_pos3 : x3 > 0) (h_pos4 : x4 > 0)
  (h_cond1 : x1^3 + x3^3 + 3*x1*x3 = 1)
  (h_cond2 : x2 + x4 = 1) : 
  (x1 + 1/x1)^3 + (x2 + 1/x2)^3 + (x3 + 1/x3)^3 + (x4 + 1/x4)^3 ≥ 125/4 := by
sorry

end NUMINAMATH_CALUDE_sum_cube_inequality_l530_53016


namespace NUMINAMATH_CALUDE_tv_cash_savings_l530_53043

/-- Calculates the savings when buying a television with cash instead of an installment plan. -/
theorem tv_cash_savings (cash_price : ℕ) (down_payment : ℕ) (monthly_payment : ℕ) (num_months : ℕ) : 
  cash_price = 400 →
  down_payment = 120 →
  monthly_payment = 30 →
  num_months = 12 →
  down_payment + monthly_payment * num_months - cash_price = 80 := by
sorry

end NUMINAMATH_CALUDE_tv_cash_savings_l530_53043


namespace NUMINAMATH_CALUDE_coal_pile_remaining_l530_53052

theorem coal_pile_remaining (total : ℝ) (used : ℝ) (remaining : ℝ) : 
  used = (4 : ℝ) / 10 * total → remaining = (6 : ℝ) / 10 * total :=
by
  sorry

end NUMINAMATH_CALUDE_coal_pile_remaining_l530_53052


namespace NUMINAMATH_CALUDE_worker_completion_time_l530_53023

/-- Given workers A and B, where A can complete a job in 15 days, works for 5 days,
    and B finishes the remaining work in 18 days, prove that B can complete the
    entire job alone in 27 days. -/
theorem worker_completion_time
  (total_days_A : ℕ)
  (work_days_A : ℕ)
  (remaining_days_B : ℕ)
  (h1 : total_days_A = 15)
  (h2 : work_days_A = 5)
  (h3 : remaining_days_B = 18) :
  (total_days_A * remaining_days_B) / (total_days_A - work_days_A) = 27 := by
  sorry

end NUMINAMATH_CALUDE_worker_completion_time_l530_53023


namespace NUMINAMATH_CALUDE_price_reduction_proof_l530_53075

theorem price_reduction_proof (current_price : ℝ) (reduction_percentage : ℝ) (claimed_reduction : ℝ) : 
  current_price = 45 ∧ reduction_percentage = 0.1 ∧ claimed_reduction = 10 →
  (100 / (100 - reduction_percentage * 100) * current_price) - current_price ≠ claimed_reduction :=
by
  sorry

end NUMINAMATH_CALUDE_price_reduction_proof_l530_53075


namespace NUMINAMATH_CALUDE_oncoming_train_speed_l530_53050

/-- Given two trains passing each other, calculate the speed of the oncoming train -/
theorem oncoming_train_speed
  (v₁ : ℝ)  -- Speed of the passenger's train in km/h
  (l : ℝ)   -- Length of the oncoming train in meters
  (t : ℝ)   -- Time taken for the oncoming train to pass in seconds
  (h₁ : v₁ = 40)  -- The speed of the passenger's train is 40 km/h
  (h₂ : l = 75)   -- The length of the oncoming train is 75 meters
  (h₃ : t = 3)    -- The time taken to pass is 3 seconds
  : ∃ v₂ : ℝ, v₂ = 50 ∧ l / 1000 = (v₁ + v₂) * (t / 3600) :=
sorry

end NUMINAMATH_CALUDE_oncoming_train_speed_l530_53050


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_zero_sqrt_product_division_equals_three_sqrt_two_over_two_l530_53051

-- Problem 1
theorem sqrt_expression_equals_zero :
  Real.sqrt 18 - Real.sqrt 32 + Real.sqrt 2 = 0 := by sorry

-- Problem 2
theorem sqrt_product_division_equals_three_sqrt_two_over_two :
  Real.sqrt 12 * (Real.sqrt 3 / 2) / Real.sqrt 2 = 3 * Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_zero_sqrt_product_division_equals_three_sqrt_two_over_two_l530_53051


namespace NUMINAMATH_CALUDE_book_arrangement_count_l530_53092

def num_books : ℕ := 7
def num_identical_books : ℕ := 3

theorem book_arrangement_count : 
  (num_books.factorial) / (num_identical_books.factorial) = 840 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l530_53092


namespace NUMINAMATH_CALUDE_prime_square_mod_24_l530_53048

theorem prime_square_mod_24 (p : ℕ) (hp : Prime p) (hp_gt_3 : p > 3) :
  p^2 % 24 = 1 :=
sorry

end NUMINAMATH_CALUDE_prime_square_mod_24_l530_53048


namespace NUMINAMATH_CALUDE_dividend_calculation_l530_53065

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 50)
  (h2 : quotient = 70)
  (h3 : remainder = 20) :
  divisor * quotient + remainder = 3520 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l530_53065


namespace NUMINAMATH_CALUDE_greatest_four_digit_multiple_of_17_l530_53063

theorem greatest_four_digit_multiple_of_17 : 
  ∀ n : ℕ, n ≤ 9999 → n ≥ 1000 → n % 17 = 0 → n ≤ 9996 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_four_digit_multiple_of_17_l530_53063


namespace NUMINAMATH_CALUDE_sin_20_cos_40_plus_cos_20_sin_40_l530_53054

theorem sin_20_cos_40_plus_cos_20_sin_40 : 
  Real.sin (20 * π / 180) * Real.cos (40 * π / 180) + 
  Real.cos (20 * π / 180) * Real.sin (40 * π / 180) = 
  Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_20_cos_40_plus_cos_20_sin_40_l530_53054


namespace NUMINAMATH_CALUDE_log_equation_solution_l530_53077

theorem log_equation_solution : 
  ∃! x : ℝ, (1 : ℝ) + Real.log x = Real.log (1 + x) :=
by
  use (1 : ℝ) / 9
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l530_53077


namespace NUMINAMATH_CALUDE_complex_equation_solution_l530_53062

theorem complex_equation_solution (z : ℂ) :
  z + Complex.abs z = 2 + Complex.I → z = 3/4 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l530_53062


namespace NUMINAMATH_CALUDE_inequality_system_solution_range_l530_53071

theorem inequality_system_solution_range (a : ℝ) : 
  (∃ (s : Finset ℤ), s.card = 5 ∧ 
    (∀ x : ℤ, x ∈ s ↔ (x : ℝ) - a ≤ 0 ∧ 7 + 2 * (x : ℝ) > 1)) →
  2 ≤ a ∧ a < 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_range_l530_53071


namespace NUMINAMATH_CALUDE_isosceles_base_length_l530_53028

/-- Represents a triangle with a perimeter -/
structure Triangle where
  perimeter : ℝ

/-- Represents an equilateral triangle -/
structure EquilateralTriangle extends Triangle

/-- Represents an isosceles triangle -/
structure IsoscelesTriangle extends Triangle where
  base : ℝ
  leg : ℝ

/-- Theorem stating the length of the base of the isosceles triangle -/
theorem isosceles_base_length 
  (et : EquilateralTriangle) 
  (it : IsoscelesTriangle) 
  (h1 : et.perimeter = 60) 
  (h2 : it.perimeter = 45) 
  (h3 : it.leg = et.perimeter / 3) : 
  it.base = 5 := by sorry

end NUMINAMATH_CALUDE_isosceles_base_length_l530_53028


namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l530_53089

theorem least_positive_integer_with_remainders : ∃ n : ℕ, 
  n > 0 ∧ 
  n % 4 = 1 ∧ 
  n % 5 = 2 ∧ 
  n % 6 = 3 ∧
  (∀ m : ℕ, m > 0 ∧ m % 4 = 1 ∧ m % 5 = 2 ∧ m % 6 = 3 → m ≥ n) ∧
  n = 57 := by
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l530_53089


namespace NUMINAMATH_CALUDE_decimal_to_base5_l530_53015

theorem decimal_to_base5 :
  ∃ (a b c : ℕ), a < 5 ∧ b < 5 ∧ c < 5 ∧ 88 = c * 5^2 + b * 5^1 + a * 5^0 ∧ 
  (a = 3 ∧ b = 2 ∧ c = 3) := by
sorry

end NUMINAMATH_CALUDE_decimal_to_base5_l530_53015


namespace NUMINAMATH_CALUDE_system_solution_l530_53056

theorem system_solution (a b c m n k : ℚ) :
  (∃ x y : ℚ, a * x + b * y = c ∧ m * x - n * y = k ∧ x = -3 ∧ y = 4) →
  (∃ x y : ℚ, a * (x + y) + b * (x - y) = c ∧ m * (x + y) - n * (x - y) = k ∧ x = 1/2 ∧ y = -7/2) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l530_53056


namespace NUMINAMATH_CALUDE_logarithm_equation_solution_l530_53000

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem logarithm_equation_solution : 
  ∃ (x : ℝ), x > 0 ∧ 
  (log_base (Real.sqrt 3) x + log_base (Real.sqrt 3) x + log_base (3 ^ (1/6 : ℝ)) x + 
   log_base (Real.sqrt 3) x + log_base (Real.sqrt 3) x + log_base (Real.sqrt 3) x + 
   log_base (Real.sqrt 3) x + log_base (Real.sqrt 3) x = 36) ∧
  x = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_logarithm_equation_solution_l530_53000


namespace NUMINAMATH_CALUDE_bowling_ball_weight_proof_l530_53010

/-- The weight of one bowling ball in pounds -/
def bowling_ball_weight : ℝ := 18.75

/-- The weight of one canoe in pounds -/
def canoe_weight : ℝ := 30

theorem bowling_ball_weight_proof :
  (8 : ℝ) * bowling_ball_weight = (5 : ℝ) * canoe_weight ∧
  (4 : ℝ) * canoe_weight = 120 ∧
  bowling_ball_weight = 18.75 := by
  sorry

#check bowling_ball_weight_proof

end NUMINAMATH_CALUDE_bowling_ball_weight_proof_l530_53010


namespace NUMINAMATH_CALUDE_total_vehicles_in_yard_l530_53040

theorem total_vehicles_in_yard (num_trucks : ℕ) (num_tanks : ℕ) : 
  num_trucks = 20 → 
  num_tanks = 5 * num_trucks → 
  num_tanks + num_trucks = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_total_vehicles_in_yard_l530_53040


namespace NUMINAMATH_CALUDE_max_value_cubic_sum_l530_53019

theorem max_value_cubic_sum (x y : ℝ) 
  (h_pos_x : 0 < x) (h_pos_y : 0 < y) 
  (h_x_bound : x ≤ 2) (h_y_bound : y ≤ 3) 
  (h_sum : x + y = 3) : 
  (∀ a b : ℝ, 0 < a → 0 < b → a ≤ 2 → b ≤ 3 → a + b = 3 → 4*a^3 + b^3 ≤ 4*x^3 + y^3) → 
  4*x^3 + y^3 = 33 := by
sorry

end NUMINAMATH_CALUDE_max_value_cubic_sum_l530_53019


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l530_53072

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = 2*x - 3) ↔ x ≥ 3/2 := by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l530_53072


namespace NUMINAMATH_CALUDE_repaved_total_correct_l530_53026

/-- The total inches of road repaved by a construction company -/
def total_repaved (before_today : ℕ) (today : ℕ) : ℕ :=
  before_today + today

/-- Theorem stating that the total inches repaved is 4938 -/
theorem repaved_total_correct : total_repaved 4133 805 = 4938 := by
  sorry

end NUMINAMATH_CALUDE_repaved_total_correct_l530_53026


namespace NUMINAMATH_CALUDE_remainder_of_n_l530_53007

theorem remainder_of_n (n : ℕ) (h1 : n^2 % 7 = 1) (h2 : n^3 % 7 = 6) : n % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_n_l530_53007


namespace NUMINAMATH_CALUDE_number_division_problem_l530_53058

theorem number_division_problem :
  let sum := 3927 + 2873
  let diff := 3927 - 2873
  let quotient := 3 * diff
  ∀ (N r : ℕ), 
    N / sum = quotient ∧ 
    N % sum = r ∧ 
    r < sum →
    N = 21481600 + r :=
by sorry

end NUMINAMATH_CALUDE_number_division_problem_l530_53058


namespace NUMINAMATH_CALUDE_initial_average_height_l530_53041

theorem initial_average_height (n : ℕ) (wrong_height actual_height : ℝ) (actual_average : ℝ) :
  n = 35 ∧
  wrong_height = 166 ∧
  actual_height = 106 ∧
  actual_average = 181 →
  (n * actual_average + (wrong_height - actual_height)) / n = 182 + 5 / 7 :=
by sorry

end NUMINAMATH_CALUDE_initial_average_height_l530_53041


namespace NUMINAMATH_CALUDE_remainder_of_9876543210_mod_101_l530_53078

theorem remainder_of_9876543210_mod_101 : 9876543210 % 101 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_9876543210_mod_101_l530_53078


namespace NUMINAMATH_CALUDE_remaining_integers_l530_53049

theorem remaining_integers (T : Finset ℕ) : 
  T = Finset.range 100 → 
  (T.filter (λ x => x % 4 ≠ 0 ∧ x % 5 ≠ 0)).card = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_remaining_integers_l530_53049


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l530_53027

theorem quadratic_equation_properties (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (hab : a ≤ b) :
  let f : ℝ → ℝ := fun x ↦ x^2 + (a + b - 1 : ℝ) * x + (a * b - a - b : ℝ)
  -- The equation has two distinct real solutions
  ∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0 ∧
  -- If one solution is an integer, then both are non-positive integers and b < 2a
  (∃ z : ℤ, f (z : ℝ) = 0 → ∃ r s : ℤ, r ≤ 0 ∧ s ≤ 0 ∧ f (r : ℝ) = 0 ∧ f (s : ℝ) = 0 ∧ b < 2 * a) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l530_53027


namespace NUMINAMATH_CALUDE_pictures_deleted_l530_53086

theorem pictures_deleted (zoo_pics : ℕ) (museum_pics : ℕ) (remaining_pics : ℕ) : 
  zoo_pics = 49 → museum_pics = 8 → remaining_pics = 19 →
  zoo_pics + museum_pics - remaining_pics = 38 := by
  sorry

end NUMINAMATH_CALUDE_pictures_deleted_l530_53086


namespace NUMINAMATH_CALUDE_negation_equivalence_l530_53060

theorem negation_equivalence :
  (¬ ∃ x : ℝ, (x < 1 ∨ x^2 ≥ 4)) ↔ (∀ x : ℝ, (x ≥ 1 ∧ x^2 < 4)) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l530_53060


namespace NUMINAMATH_CALUDE_exists_square_farther_than_V_l530_53097

/-- Represents a square on the board --/
structure Square where
  x : Fin 19
  y : Fin 19

/-- Defines the movement of the dragon --/
def dragonMove (s : Square) : Set Square :=
  { t | (t.x = s.x + 4 ∧ t.y = s.y + 1) ∨
        (t.x = s.x + 4 ∧ t.y = s.y - 1) ∨
        (t.x = s.x - 4 ∧ t.y = s.y + 1) ∨
        (t.x = s.x - 4 ∧ t.y = s.y - 1) ∨
        (t.x = s.x + 1 ∧ t.y = s.y + 4) ∨
        (t.x = s.x + 1 ∧ t.y = s.y - 4) ∨
        (t.x = s.x - 1 ∧ t.y = s.y + 4) ∨
        (t.x = s.x - 1 ∧ t.y = s.y - 4) }

/-- Draconian distance between two squares --/
def draconianDistance (s t : Square) : ℕ :=
  sorry

/-- Corner square --/
def C : Square :=
  { x := 0, y := 0 }

/-- Diagonally adjacent square to C --/
def V : Square :=
  { x := 1, y := 1 }

/-- Main theorem --/
theorem exists_square_farther_than_V :
  ∃ X : Square, draconianDistance C X > draconianDistance C V :=
sorry

end NUMINAMATH_CALUDE_exists_square_farther_than_V_l530_53097


namespace NUMINAMATH_CALUDE_book_costs_18_l530_53081

-- Define the cost of the album
def album_cost : ℝ := 20

-- Define the discount percentage for the CD
def cd_discount_percentage : ℝ := 0.30

-- Define the cost difference between the book and CD
def book_cd_difference : ℝ := 4

-- Calculate the cost of the CD
def cd_cost : ℝ := album_cost * (1 - cd_discount_percentage)

-- Calculate the cost of the book
def book_cost : ℝ := cd_cost + book_cd_difference

-- Theorem to prove
theorem book_costs_18 : book_cost = 18 := by
  sorry

end NUMINAMATH_CALUDE_book_costs_18_l530_53081


namespace NUMINAMATH_CALUDE_grid_toothpicks_l530_53003

/-- Calculates the number of toothpicks required for a rectangular grid. -/
def toothpicks_in_grid (height width : ℕ) : ℕ :=
  (height + 1) * width + (width + 1) * height

/-- Theorem: A rectangular grid with height 15 and width 12 requires 387 toothpicks. -/
theorem grid_toothpicks : toothpicks_in_grid 15 12 = 387 := by
  sorry

#eval toothpicks_in_grid 15 12

end NUMINAMATH_CALUDE_grid_toothpicks_l530_53003


namespace NUMINAMATH_CALUDE_plane_distance_ratio_l530_53032

theorem plane_distance_ratio (total_distance bus_distance : ℝ) 
  (h1 : total_distance = 1800)
  (h2 : bus_distance = 720)
  : (total_distance - (2/3 * bus_distance + bus_distance)) / total_distance = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_plane_distance_ratio_l530_53032


namespace NUMINAMATH_CALUDE_water_evaporation_proof_l530_53070

-- Define the initial composition of solution y
def solution_y_composition : ℝ := 0.3

-- Define the initial amount of solution y
def initial_amount : ℝ := 6

-- Define the amount of solution y added after evaporation
def amount_added : ℝ := 2

-- Define the amount remaining after evaporation
def amount_remaining : ℝ := 4

-- Define the new composition of the solution
def new_composition : ℝ := 0.4

-- Define the amount of water evaporated
def water_evaporated : ℝ := 2

-- Theorem statement
theorem water_evaporation_proof :
  let initial_liquid_x := solution_y_composition * initial_amount
  let added_liquid_x := solution_y_composition * amount_added
  let total_liquid_x := initial_liquid_x + added_liquid_x
  let new_total_amount := total_liquid_x / new_composition
  new_total_amount = amount_remaining + amount_added →
  water_evaporated = amount_added :=
by sorry

end NUMINAMATH_CALUDE_water_evaporation_proof_l530_53070


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l530_53090

-- Define the function f
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_function_properties (a b c : ℝ) (ha : a ≠ 0)
  (h_no_roots : ∀ x : ℝ, f a b c x ≠ x) :
  (∀ x : ℝ, f a b c (f a b c x) ≠ x) ∧
  (a > 0 → ∀ x : ℝ, f a b c (f a b c x) > x) ∧
  (a + b + c = 0 → ∀ x : ℝ, f a b c (f a b c x) < x) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l530_53090


namespace NUMINAMATH_CALUDE_function_value_theorem_l530_53087

theorem function_value_theorem (f : ℝ → ℝ) (a : ℝ) 
  (h1 : ∀ x, f (x + 1) = x) 
  (h2 : f a = 8) : 
  a = 9 := by
  sorry

end NUMINAMATH_CALUDE_function_value_theorem_l530_53087


namespace NUMINAMATH_CALUDE_number_equality_l530_53053

theorem number_equality (x : ℝ) (h : 0.15 * x = 0.25 * 16 + 2) : x = 40 := by
  sorry

end NUMINAMATH_CALUDE_number_equality_l530_53053


namespace NUMINAMATH_CALUDE_marcie_coffee_cups_l530_53045

theorem marcie_coffee_cups (sandra_cups : ℕ) (total_cups : ℕ) (marcie_cups : ℕ) : 
  sandra_cups = 6 → total_cups = 8 → marcie_cups = total_cups - sandra_cups → marcie_cups = 2 := by
  sorry

end NUMINAMATH_CALUDE_marcie_coffee_cups_l530_53045


namespace NUMINAMATH_CALUDE_parallelogram_sum_l530_53091

/-- A parallelogram with sides measuring 6y-2, 4x+5, 12y-10, and 8x+1 has x + y = 7/3 -/
theorem parallelogram_sum (x y : ℚ) : 
  (6 * y - 2 : ℚ) = (12 * y - 10 : ℚ) →
  (4 * x + 5 : ℚ) = (8 * x + 1 : ℚ) →
  x + y = 7/3 := by sorry

end NUMINAMATH_CALUDE_parallelogram_sum_l530_53091


namespace NUMINAMATH_CALUDE_remainder_8_pow_305_mod_9_l530_53079

theorem remainder_8_pow_305_mod_9 : 8^305 % 9 = 8 := by sorry

end NUMINAMATH_CALUDE_remainder_8_pow_305_mod_9_l530_53079


namespace NUMINAMATH_CALUDE_probability_of_valid_pair_l530_53076

def is_odd (n : ℤ) : Prop := n % 2 ≠ 0

def is_divisible_by_5 (n : ℤ) : Prop := n % 5 = 0

def valid_pair (a b : ℤ) : Prop :=
  1 ≤ a ∧ a ≤ 20 ∧
  1 ≤ b ∧ b ≤ 20 ∧
  a ≠ b ∧
  is_odd (a * b) ∧
  is_divisible_by_5 (a + b)

def total_pairs : ℕ := 190

def valid_pairs : ℕ := 18

theorem probability_of_valid_pair :
  (valid_pairs : ℚ) / total_pairs = 9 / 95 :=
sorry

end NUMINAMATH_CALUDE_probability_of_valid_pair_l530_53076
