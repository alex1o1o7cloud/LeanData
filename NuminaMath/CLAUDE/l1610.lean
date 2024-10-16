import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_root_implies_m_l1610_161090

theorem quadratic_root_implies_m (m : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x + m = 0 ∧ x = -1) → m = -3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_m_l1610_161090


namespace NUMINAMATH_CALUDE_dans_music_store_spending_l1610_161088

def clarinet_cost : ℚ := 130.30
def songbook_cost : ℚ := 11.24

theorem dans_music_store_spending :
  clarinet_cost + songbook_cost = 141.54 := by sorry

end NUMINAMATH_CALUDE_dans_music_store_spending_l1610_161088


namespace NUMINAMATH_CALUDE_min_words_for_90_percent_l1610_161065

/-- The minimum number of words needed to achieve at least 90% on a vocabulary exam -/
theorem min_words_for_90_percent (total_words : ℕ) (min_percentage : ℚ) : 
  total_words = 600 → min_percentage = 90 / 100 → 
  ∃ (min_words : ℕ), min_words = 540 ∧ 
    (min_words : ℚ) / total_words ≥ min_percentage ∧
    ∀ (n : ℕ), n < min_words → (n : ℚ) / total_words < min_percentage :=
by sorry

end NUMINAMATH_CALUDE_min_words_for_90_percent_l1610_161065


namespace NUMINAMATH_CALUDE_original_number_proof_l1610_161057

theorem original_number_proof (x y : ℝ) : 
  x = 19 ∧ 8 * x + 3 * y = 203 → x + y = 36 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l1610_161057


namespace NUMINAMATH_CALUDE_train_speed_calculation_l1610_161033

/-- Calculates the speed of a train in km/hr given its length in meters and time to cross a pole in seconds. -/
def trainSpeed (length : Float) (time : Float) : Float :=
  (length / 1000) / (time / 3600)

/-- Theorem stating that a train with a length of 450.00000000000006 meters 
    crossing a pole in 27 seconds has a speed of 60 km/hr. -/
theorem train_speed_calculation :
  let length : Float := 450.00000000000006
  let time : Float := 27
  trainSpeed length time = 60 := by
  sorry

#eval trainSpeed 450.00000000000006 27

end NUMINAMATH_CALUDE_train_speed_calculation_l1610_161033


namespace NUMINAMATH_CALUDE_obtuse_angle_condition_l1610_161059

/-- Two-dimensional vector -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Dot product of two 2D vectors -/
def dot_product (v w : Vector2D) : ℝ := v.x * w.x + v.y * w.y

/-- Angle between two vectors is obtuse iff their dot product is negative -/
def is_obtuse_angle (v w : Vector2D) : Prop := dot_product v w < 0

/-- Two vectors are parallel iff their components are proportional -/
def are_parallel (v w : Vector2D) : Prop := v.x * w.y = v.y * w.x

theorem obtuse_angle_condition (x : ℝ) :
  let a : Vector2D := ⟨x, 2⟩
  let b : Vector2D := ⟨2, -5⟩
  is_obtuse_angle a b ∧ ¬are_parallel a b ↔ x < 5 ∧ x ≠ -4/5 := by
  sorry

end NUMINAMATH_CALUDE_obtuse_angle_condition_l1610_161059


namespace NUMINAMATH_CALUDE_function_properties_l1610_161086

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 2

theorem function_properties :
  ∀ (a : ℝ),
  (∀ (x : ℝ), -5 ≤ x ∧ x ≤ 5 → 
    (a = -1 → 
      (∀ (y : ℝ), -5 ≤ y ∧ y ≤ 5 → f a y ≤ 37) ∧
      (∃ (y : ℝ), -5 ≤ y ∧ y ≤ 5 ∧ f a y = 37) ∧
      (∀ (y : ℝ), -5 ≤ y ∧ y ≤ 5 → f a y ≥ 1) ∧
      (∃ (y : ℝ), -5 ≤ y ∧ y ≤ 5 ∧ f a y = 1)) ∧
    ((-5 < a ∧ a < 5) ↔ 
      (∃ (y z : ℝ), -5 ≤ y ∧ y < z ∧ z ≤ 5 ∧ f a y > f a z)) ∧
    ((-5 < a ∧ a < 0) → 
      (∀ (y : ℝ), -5 ≤ y ∧ y ≤ 5 → f a y ≤ 27 - 10*a) ∧
      (∃ (y : ℝ), -5 ≤ y ∧ y ≤ 5 ∧ f a y = 27 - 10*a)) ∧
    ((0 ≤ a ∧ a < 5) → 
      (∀ (y : ℝ), -5 ≤ y ∧ y ≤ 5 → f a y ≤ 27 + 10*a) ∧
      (∃ (y : ℝ), -5 ≤ y ∧ y ≤ 5 ∧ f a y = 27 + 10*a))) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1610_161086


namespace NUMINAMATH_CALUDE_perfect_square_sum_implies_divisible_by_eight_l1610_161036

theorem perfect_square_sum_implies_divisible_by_eight (a n : ℕ) (h1 : a > 0) (h2 : Even a) 
  (h3 : ∃ k : ℕ, k^2 = (a^(n+1) - 1) / (a - 1)) : 8 ∣ a := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_sum_implies_divisible_by_eight_l1610_161036


namespace NUMINAMATH_CALUDE_savings_proof_l1610_161007

/-- Calculates savings given income and expenditure ratio -/
def calculate_savings (income : ℕ) (income_ratio expenditure_ratio : ℕ) : ℕ :=
  income - (income * expenditure_ratio) / income_ratio

/-- Proves that given the specified conditions, the savings are 3400 -/
theorem savings_proof (income : ℕ) (income_ratio expenditure_ratio : ℕ) 
  (h1 : income = 17000)
  (h2 : income_ratio = 5)
  (h3 : expenditure_ratio = 4) :
  calculate_savings income income_ratio expenditure_ratio = 3400 := by
  sorry

#eval calculate_savings 17000 5 4

end NUMINAMATH_CALUDE_savings_proof_l1610_161007


namespace NUMINAMATH_CALUDE_largest_primary_divisor_l1610_161030

/-- A positive integer is prime if it has exactly two positive divisors. -/
def IsPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

/-- A positive integer is a primary divisor if for every positive divisor d,
    at least one of d - 1 or d + 1 is prime. -/
def IsPrimaryDivisor (n : ℕ) : Prop :=
  n > 0 ∧ ∀ d : ℕ, d ∣ n → IsPrime (d - 1) ∨ IsPrime (d + 1)

/-- 48 is the largest primary divisor number. -/
theorem largest_primary_divisor : ∀ n : ℕ, IsPrimaryDivisor n → n ≤ 48 :=
  sorry

#check largest_primary_divisor

end NUMINAMATH_CALUDE_largest_primary_divisor_l1610_161030


namespace NUMINAMATH_CALUDE_cos_alpha_plus_pi_third_l1610_161043

theorem cos_alpha_plus_pi_third (α β : Real) 
  (h1 : 0 < α ∧ α < π/2)
  (h2 : 2 * Real.sin β - Real.cos α = 1)
  (h3 : Real.sin α + 2 * Real.cos β = Real.sqrt 3) :
  Real.cos (α + π/3) = -1/4 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_plus_pi_third_l1610_161043


namespace NUMINAMATH_CALUDE_total_cotton_needed_l1610_161023

/-- The amount of cotton needed for one tee-shirt in feet -/
def cotton_per_shirt : ℝ := 4

/-- The number of tee-shirts to be made -/
def num_shirts : ℕ := 15

/-- Theorem stating the total amount of cotton needed -/
theorem total_cotton_needed : 
  cotton_per_shirt * (num_shirts : ℝ) = 60 := by sorry

end NUMINAMATH_CALUDE_total_cotton_needed_l1610_161023


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l1610_161045

theorem unique_solution_for_equation :
  ∃! (n k : ℕ), n > 0 ∧ k > 0 ∧ (n + 1)^n = 2 * n^k + 3 * n + 1 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l1610_161045


namespace NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l1610_161021

/-- The complex number z = (2-i)/i corresponds to a point in the third quadrant -/
theorem complex_number_in_third_quadrant :
  let i : ℂ := Complex.I
  let z : ℂ := (2 - i) / i
  (z.re < 0) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l1610_161021


namespace NUMINAMATH_CALUDE_factorization_of_5x_cubed_minus_125x_l1610_161072

theorem factorization_of_5x_cubed_minus_125x (x : ℝ) :
  5 * x^3 - 125 * x = 5 * x * (x + 5) * (x - 5) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_5x_cubed_minus_125x_l1610_161072


namespace NUMINAMATH_CALUDE_sum_of_squared_differences_zero_l1610_161040

theorem sum_of_squared_differences_zero (x y z : ℝ) :
  (x - 4)^2 + (y - 5)^2 + (z - 6)^2 = 0 → x + y + z = 15 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squared_differences_zero_l1610_161040


namespace NUMINAMATH_CALUDE_island_puzzle_l1610_161005

-- Define the types of residents
inductive Resident
| TruthTeller
| Liar

-- Define the statement made by K
def kStatement (k m : Resident) : Prop :=
  k = Resident.Liar ∨ m = Resident.Liar

-- Theorem to prove
theorem island_puzzle :
  ∃ (k m : Resident),
    (k = Resident.TruthTeller ∧ 
     m = Resident.Liar ∧
     (k = Resident.TruthTeller → kStatement k m) ∧
     (k = Resident.Liar → ¬kStatement k m)) :=
sorry

end NUMINAMATH_CALUDE_island_puzzle_l1610_161005


namespace NUMINAMATH_CALUDE_memory_efficiency_improvement_l1610_161074

theorem memory_efficiency_improvement (x : ℝ) (h : x > 0) :
  (100 / x) - (100 / (1.2 * x)) = 5 / 12 ↔
  (100 / x) - (100 / ((1 + 0.2) * x)) = 5 / 12 :=
by sorry

end NUMINAMATH_CALUDE_memory_efficiency_improvement_l1610_161074


namespace NUMINAMATH_CALUDE_range_of_2a_plus_3b_inequality_with_squared_sum_l1610_161076

-- Part 1
theorem range_of_2a_plus_3b (a b : ℝ) 
  (h1 : -1 ≤ a + b ∧ a + b ≤ 1) 
  (h2 : -1 ≤ a - b ∧ a - b ≤ 1) : 
  -3 ≤ 2*a + 3*b ∧ 2*a + 3*b ≤ 3 :=
sorry

-- Part 2
theorem inequality_with_squared_sum (a b c : ℝ) 
  (h : a^2 + b^2 + c^2 = 6) : 
  1 / (a^2 + 1) + 1 / (b^2 + 2) > 1/2 - 1 / (c^2 + 3) :=
sorry

end NUMINAMATH_CALUDE_range_of_2a_plus_3b_inequality_with_squared_sum_l1610_161076


namespace NUMINAMATH_CALUDE_inner_hexagon_area_l1610_161049

/-- Represents an equilateral triangle --/
structure EquilateralTriangle where
  sideLength : ℝ
  area : ℝ

/-- Represents the configuration of triangles in the problem --/
structure TriangleConfiguration where
  largeTriangle : EquilateralTriangle
  smallTriangles : List EquilateralTriangle
  innerHexagonArea : ℝ

/-- The given configuration satisfies the problem conditions --/
def satisfiesProblemConditions (config : TriangleConfiguration) : Prop :=
  config.smallTriangles.length = 6 ∧
  config.smallTriangles.map (λ t => t.area) = [1, 1, 9, 9, 16, 16]

/-- The theorem to be proved --/
theorem inner_hexagon_area 
  (config : TriangleConfiguration) 
  (h : satisfiesProblemConditions config) : 
  config.innerHexagonArea = 38 := by
  sorry

end NUMINAMATH_CALUDE_inner_hexagon_area_l1610_161049


namespace NUMINAMATH_CALUDE_sum_of_coefficients_abs_l1610_161095

theorem sum_of_coefficients_abs (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℤ) :
  (∀ x, (2 - x)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| = 665 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_abs_l1610_161095


namespace NUMINAMATH_CALUDE_probability_specific_case_l1610_161063

/-- The probability of drawing one green, one white, and one blue ball simultaneously -/
def probability_three_colors (green white blue : ℕ) : ℚ :=
  let total := green + white + blue
  let favorable := green * white * blue
  let total_combinations := (total * (total - 1) * (total - 2)) / 6
  (favorable : ℚ) / total_combinations

/-- Theorem stating the probability of drawing one green, one white, and one blue ball -/
theorem probability_specific_case : 
  probability_three_colors 12 10 8 = 24 / 101 := by
  sorry


end NUMINAMATH_CALUDE_probability_specific_case_l1610_161063


namespace NUMINAMATH_CALUDE_exchange_process_duration_l1610_161026

/-- Represents the maximum number of exchanges possible in the described process -/
def max_exchanges (n : ℕ) : ℕ := n - 1

/-- The number of children in the line -/
def total_children : ℕ := 20

/-- The theorem stating that the exchange process cannot continue for more than an hour -/
theorem exchange_process_duration : max_exchanges total_children < 60 := by
  sorry


end NUMINAMATH_CALUDE_exchange_process_duration_l1610_161026


namespace NUMINAMATH_CALUDE_expected_worth_unfair_coin_l1610_161079

/-- An unfair coin with given probabilities and payoffs -/
structure UnfairCoin where
  prob_heads : ℝ
  prob_tails : ℝ
  gain_heads : ℝ
  loss_tails : ℝ
  fixed_cost : ℝ

/-- The expected worth of a coin flip -/
def expected_worth (c : UnfairCoin) : ℝ :=
  c.prob_heads * c.gain_heads + c.prob_tails * (-c.loss_tails) - c.fixed_cost

/-- Theorem: The expected worth of the specific unfair coin is -1/3 -/
theorem expected_worth_unfair_coin :
  let c : UnfairCoin := {
    prob_heads := 1/3,
    prob_tails := 2/3,
    gain_heads := 6,
    loss_tails := 2,
    fixed_cost := 1
  }
  expected_worth c = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_expected_worth_unfair_coin_l1610_161079


namespace NUMINAMATH_CALUDE_correct_calculation_l1610_161017

theorem correct_calculation (x : ℤ) (h : 23 - x = 4) : 23 * x = 437 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1610_161017


namespace NUMINAMATH_CALUDE_pond_length_l1610_161097

/-- Given a rectangular field with length 112 m and width half of its length,
    and a square-shaped pond inside the field with an area 1/98 of the field's area,
    prove that the length of the pond is 8 meters. -/
theorem pond_length (field_length : ℝ) (field_width : ℝ) (pond_area : ℝ) :
  field_length = 112 →
  field_width = field_length / 2 →
  pond_area = (field_length * field_width) / 98 →
  Real.sqrt pond_area = 8 := by
  sorry

end NUMINAMATH_CALUDE_pond_length_l1610_161097


namespace NUMINAMATH_CALUDE_existence_implies_lower_bound_l1610_161077

theorem existence_implies_lower_bound (a : ℝ) :
  (∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ x^2 + a ≤ a*x - 3) → a ≥ 7 := by
  sorry

end NUMINAMATH_CALUDE_existence_implies_lower_bound_l1610_161077


namespace NUMINAMATH_CALUDE_solve_for_z_l1610_161087

theorem solve_for_z (x y z : ℝ) : 
  x^2 - 3*x + 6 = y - 10 → 
  y = 2*z → 
  x = -5 → 
  z = 28 := by
sorry

end NUMINAMATH_CALUDE_solve_for_z_l1610_161087


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1610_161078

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ (x : ℝ), x = 2 ∧ x^2 / a^2 = 1) →
  (∃ (c : ℝ), c^2 = a^2 + b^2 ∧ c / a = 3/2) →
  a^2 = 4 ∧ b^2 = 5 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1610_161078


namespace NUMINAMATH_CALUDE_triangle_altitude_circumradius_l1610_161014

/-- For any triangle with sides a, b, c, altitude ha from vertex A to side a,
    and circumradius R, the equation ha = bc / (2R) holds. -/
theorem triangle_altitude_circumradius (a b c ha R : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ ha > 0 ∧ R > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_altitude : ha = (2 * (a.sqrt * b.sqrt * c.sqrt + (a + b + c) / 2)) / a)
  (h_circumradius : R = (a * b * c) / (4 * (a.sqrt * b.sqrt * c.sqrt + (a + b + c) / 2))) :
  ha = b * c / (2 * R) := by
sorry

end NUMINAMATH_CALUDE_triangle_altitude_circumradius_l1610_161014


namespace NUMINAMATH_CALUDE_ellipse_chord_slope_l1610_161091

/-- The slope of a chord in an ellipse with given midpoint -/
theorem ellipse_chord_slope (x y : ℝ) :
  (x^2 / 16 + y^2 / 9 = 1) →  -- ellipse equation
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁^2 / 16 + y₁^2 / 9 = 1 ∧  -- endpoint 1 on ellipse
    x₂^2 / 16 + y₂^2 / 9 = 1 ∧  -- endpoint 2 on ellipse
    (x₁ + x₂) / 2 = -1 ∧        -- x-coordinate of midpoint
    (y₁ + y₂) / 2 = 2 ∧         -- y-coordinate of midpoint
    (y₂ - y₁) / (x₂ - x₁) = 9 / 32) -- slope of chord
  := by sorry

end NUMINAMATH_CALUDE_ellipse_chord_slope_l1610_161091


namespace NUMINAMATH_CALUDE_problem_solution_l1610_161042

theorem problem_solution (x y : ℚ) : 
  x = 51 → x^3*y - 2*x^2*y + x*y = 51000 → y = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1610_161042


namespace NUMINAMATH_CALUDE_exist_pouring_sequence_l1610_161037

/-- Represents the state of the three containers -/
structure ContainerState :=
  (a : ℕ) -- Volume in 10-liter container
  (b : ℕ) -- Volume in 7-liter container
  (c : ℕ) -- Volume in 4-liter container

/-- Represents a pouring action between containers -/
inductive PourAction
  | Pour10to7
  | Pour10to4
  | Pour7to10
  | Pour7to4
  | Pour4to10
  | Pour4to7

/-- Applies a pouring action to a container state -/
def applyAction (state : ContainerState) (action : PourAction) : ContainerState :=
  match action with
  | PourAction.Pour10to7 => sorry
  | PourAction.Pour10to4 => sorry
  | PourAction.Pour7to10 => sorry
  | PourAction.Pour7to4 => sorry
  | PourAction.Pour4to10 => sorry
  | PourAction.Pour4to7 => sorry

/-- Checks if a container state is valid -/
def isValidState (state : ContainerState) : Prop :=
  state.a ≤ 10 ∧ state.b ≤ 7 ∧ state.c ≤ 4 ∧ state.a + state.b + state.c = 10

/-- Theorem: There exists a sequence of pouring actions to reach the desired state -/
theorem exist_pouring_sequence :
  ∃ (actions : List PourAction),
    let finalState := actions.foldl applyAction ⟨10, 0, 0⟩
    isValidState finalState ∧ finalState = ⟨4, 2, 4⟩ :=
  sorry

end NUMINAMATH_CALUDE_exist_pouring_sequence_l1610_161037


namespace NUMINAMATH_CALUDE_point_on_curve_in_third_quadrant_l1610_161082

theorem point_on_curve_in_third_quadrant :
  ∀ a : ℝ, a < 0 → 3 * a^2 + (2 * a)^2 = 28 → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_point_on_curve_in_third_quadrant_l1610_161082


namespace NUMINAMATH_CALUDE_arthur_muffins_l1610_161085

theorem arthur_muffins (james_muffins : ℕ) (arthur_muffins : ℕ) 
  (h1 : james_muffins = 12 * arthur_muffins) 
  (h2 : james_muffins = 1380) : 
  arthur_muffins = 115 := by
sorry

end NUMINAMATH_CALUDE_arthur_muffins_l1610_161085


namespace NUMINAMATH_CALUDE_tile_arrangements_l1610_161073

/-- The number of distinguishable arrangements of tiles -/
def distinguishable_arrangements (red blue green yellow : ℕ) : ℕ :=
  Nat.factorial (red + blue + green + yellow) /
  (Nat.factorial red * Nat.factorial blue * Nat.factorial green * Nat.factorial yellow)

/-- Theorem stating that the number of distinguishable arrangements
    of 1 red tile, 2 blue tiles, 2 green tiles, and 4 yellow tiles is 3780 -/
theorem tile_arrangements :
  distinguishable_arrangements 1 2 2 4 = 3780 := by
  sorry

end NUMINAMATH_CALUDE_tile_arrangements_l1610_161073


namespace NUMINAMATH_CALUDE_tangent_line_and_decreasing_condition_l1610_161062

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 + (1-2*a)*x + a

-- State the theorem
theorem tangent_line_and_decreasing_condition (a : ℝ) :
  -- The tangent line at x = 1 has equation 2x + y - 2 = 0
  (∃ m b : ℝ, ∀ x y : ℝ, y = f a x → (x = 1 → y = m*x + b) ∧ m = -2 ∧ b = 2) ∧
  -- f is strictly decreasing on ℝ iff a ∈ (3-√6, 3+√6)
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ > f a x₂) ↔ (a > 3 - Real.sqrt 6 ∧ a < 3 + Real.sqrt 6) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_and_decreasing_condition_l1610_161062


namespace NUMINAMATH_CALUDE_runners_meet_time_l1610_161000

/-- The circumference of the circular track in meters -/
def track_length : ℝ := 600

/-- The speeds of the four runners in meters per second -/
def runner_speeds : List ℝ := [5.0, 5.5, 6.0, 6.5]

/-- The time in seconds for the runners to meet again -/
def meeting_time : ℝ := 1200

/-- Theorem stating that the given meeting time is the minimum time for the runners to meet again -/
theorem runners_meet_time : 
  meeting_time = (track_length / (runner_speeds[1] - runner_speeds[0])) ∧
  meeting_time = (track_length / (runner_speeds[2] - runner_speeds[1])) ∧
  meeting_time = (track_length / (runner_speeds[3] - runner_speeds[2])) ∧
  (∀ t : ℝ, t > 0 → t < meeting_time → 
    ∃ i j : Fin 4, i ≠ j ∧ 
    (runner_speeds[i] * t) % track_length ≠ (runner_speeds[j] * t) % track_length) :=
by sorry

end NUMINAMATH_CALUDE_runners_meet_time_l1610_161000


namespace NUMINAMATH_CALUDE_ellipse_line_segment_no_intersection_l1610_161069

theorem ellipse_line_segment_no_intersection (a : ℝ) :
  a > 0 →
  (∀ x y : ℝ, x^2 + (1/2) * y^2 = a^2 →
    ((2 ≤ x ∧ x ≤ 4 ∧ y = (3-1)/(4-2) * (x-2) + 1) → False)) →
  (0 < a ∧ a < 3 * Real.sqrt 2 / 2) ∨ (a > Real.sqrt 82 / 2) :=
sorry

end NUMINAMATH_CALUDE_ellipse_line_segment_no_intersection_l1610_161069


namespace NUMINAMATH_CALUDE_proposition_q_false_iff_a_lt_2_l1610_161089

theorem proposition_q_false_iff_a_lt_2 (a : ℝ) :
  (¬∀ x : ℝ, a * x^2 + 4 * x + a ≥ -2 * x^2 + 1) ↔ a < 2 :=
by sorry

end NUMINAMATH_CALUDE_proposition_q_false_iff_a_lt_2_l1610_161089


namespace NUMINAMATH_CALUDE_equation_identity_l1610_161041

theorem equation_identity (x : ℝ) : (2*x - 1)^3 = 5*x^3 + (3*x + 1)*(x^2 - x - 1) - 10*x^2 + 10*x := by
  sorry

end NUMINAMATH_CALUDE_equation_identity_l1610_161041


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1610_161004

theorem simplify_and_evaluate (a b : ℚ) (h1 : a = -2) (h2 : b = 1/3) :
  4 * (a^2 - 2*a*b) - (3*a^2 - 5*a*b + 1) = 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1610_161004


namespace NUMINAMATH_CALUDE_largest_divisor_of_consecutive_even_product_l1610_161019

theorem largest_divisor_of_consecutive_even_product : 
  ∃ (d : ℕ), d = 24 ∧ 
  (∀ (n : ℕ), n > 0 → d ∣ (2*n) * (2*n + 2) * (2*n + 4)) ∧
  (∀ (k : ℕ), k > d → ∃ (m : ℕ), m > 0 ∧ ¬(k ∣ (2*m) * (2*m + 2) * (2*m + 4))) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_consecutive_even_product_l1610_161019


namespace NUMINAMATH_CALUDE_bus_driver_compensation_l1610_161053

/-- Calculates the total compensation for a bus driver given their work hours and pay rates. -/
def calculate_compensation (regular_rate : ℚ) (overtime_multiplier : ℚ) (total_hours : ℕ) (regular_hours : ℕ) : ℚ :=
  let overtime_rate := regular_rate * (1 + overtime_multiplier)
  let regular_pay := regular_rate * regular_hours
  let overtime_pay := overtime_rate * (total_hours - regular_hours)
  regular_pay + overtime_pay

/-- Theorem stating that the bus driver's compensation for 60 hours of work is $1200. -/
theorem bus_driver_compensation :
  calculate_compensation 16 0.75 60 40 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_bus_driver_compensation_l1610_161053


namespace NUMINAMATH_CALUDE_advanced_ticket_price_is_14_50_l1610_161009

/-- The price of an advanced ticket for the Rhapsody Theater -/
def advanced_ticket_price (total_tickets : ℕ) (door_price : ℚ) (total_revenue : ℚ) (door_tickets : ℕ) : ℚ :=
  (total_revenue - door_price * door_tickets) / (total_tickets - door_tickets)

/-- Theorem stating that the advanced ticket price is $14.50 given the specific conditions -/
theorem advanced_ticket_price_is_14_50 :
  advanced_ticket_price 800 22 16640 672 = 14.5 := by
  sorry

#eval advanced_ticket_price 800 22 16640 672

end NUMINAMATH_CALUDE_advanced_ticket_price_is_14_50_l1610_161009


namespace NUMINAMATH_CALUDE_function_composition_l1610_161061

/-- Given a function f(x) = (x(x-2))/2, prove that f(x+2) = ((x+2)x)/2 -/
theorem function_composition (x : ℝ) : 
  let f : ℝ → ℝ := λ x => (x * (x - 2)) / 2
  f (x + 2) = ((x + 2) * x) / 2 := by
sorry

end NUMINAMATH_CALUDE_function_composition_l1610_161061


namespace NUMINAMATH_CALUDE_triangle_perimeter_from_quadratic_roots_l1610_161022

theorem triangle_perimeter_from_quadratic_roots :
  ∀ a b c : ℝ,
  (a^2 - 7*a + 10 = 0) →
  (b^2 - 7*b + 10 = 0) →
  (c^2 - 7*c + 10 = 0) →
  (a + b > c) → (b + c > a) → (c + a > b) →
  (a + b + c = 12 ∨ a + b + c = 6 ∨ a + b + c = 15) :=
by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_from_quadratic_roots_l1610_161022


namespace NUMINAMATH_CALUDE_humphrey_birds_l1610_161070

-- Define the number of bird watchers
def num_watchers : ℕ := 3

-- Define the average number of birds seen
def average_birds : ℕ := 9

-- Define the number of birds seen by Marcus
def marcus_birds : ℕ := 7

-- Define the number of birds seen by Darrel
def darrel_birds : ℕ := 9

-- Theorem to prove
theorem humphrey_birds :
  ∃ (humphrey_birds : ℕ),
    humphrey_birds = average_birds * num_watchers - marcus_birds - darrel_birds :=
by
  sorry

end NUMINAMATH_CALUDE_humphrey_birds_l1610_161070


namespace NUMINAMATH_CALUDE_quadratic_form_minimum_l1610_161080

theorem quadratic_form_minimum (x y : ℝ) : x^2 + 4*x*y + 5*y^2 - 8*x - 6*y ≥ -9/5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_minimum_l1610_161080


namespace NUMINAMATH_CALUDE_complement_union_of_sets_l1610_161027

open Set

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {3, 5}

theorem complement_union_of_sets : 
  (A ∪ B)ᶜ = {2, 4} :=
by sorry

end NUMINAMATH_CALUDE_complement_union_of_sets_l1610_161027


namespace NUMINAMATH_CALUDE_union_A_M_eq_real_union_B_complement_M_eq_B_l1610_161031

-- Define the sets A, B, and M
def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 9}
def B (b : ℝ) : Set ℝ := {x | 8 - b < x ∧ x < b}
def M : Set ℝ := {x | x < -1 ∨ x > 5}

-- Statement for the first part of the problem
theorem union_A_M_eq_real (a : ℝ) :
  A a ∪ M = Set.univ ↔ -4 ≤ a ∧ a ≤ -1 :=
sorry

-- Statement for the second part of the problem
theorem union_B_complement_M_eq_B (b : ℝ) :
  B b ∪ (Set.univ \ M) = B b ↔ b > 9 :=
sorry

end NUMINAMATH_CALUDE_union_A_M_eq_real_union_B_complement_M_eq_B_l1610_161031


namespace NUMINAMATH_CALUDE_factorial_ratio_l1610_161024

theorem factorial_ratio : Nat.factorial 5 / Nat.factorial (5 - 3) = 60 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l1610_161024


namespace NUMINAMATH_CALUDE_parabola_point_coordinates_l1610_161008

theorem parabola_point_coordinates :
  ∀ (x y : ℝ),
  (y = 2 * x^2) →                          -- M is on the parabola y = 2x^2
  (x > 0 ∧ y > 0) →                        -- M is in the first quadrant
  ((x - 0)^2 + (y - 1/8)^2 = (1/4)^2) →    -- Distance from M to focus is 1/4
  (x = Real.sqrt 2 / 8 ∧ y = 1/16) := by
sorry

end NUMINAMATH_CALUDE_parabola_point_coordinates_l1610_161008


namespace NUMINAMATH_CALUDE_position_vector_coefficients_l1610_161003

/-- Given points A and B, and points P and Q on line segment AB,
    prove that their position vectors have the specified coefficients. -/
theorem position_vector_coefficients
  (A B P Q : ℝ × ℝ) -- A, B, P, Q are points in 2D space
  (h_P : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • A + t • B) -- P is on AB
  (h_Q : ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ Q = (1 - s) • A + s • B) -- Q is on AB
  (h_P_ratio : (dist A P) / (dist P B) = 3 / 5) -- AP:PB = 3:5
  (h_Q_ratio : (dist A Q) / (dist Q B) = 4 / 3) -- AQ:QB = 4:3
  : (∃ t₁ u₁ : ℝ, P = t₁ • A + u₁ • B ∧ t₁ = 5/8 ∧ u₁ = 3/8) ∧
    (∃ t₂ u₂ : ℝ, Q = t₂ • A + u₂ • B ∧ t₂ = 3/7 ∧ u₂ = 4/7) :=
by sorry

end NUMINAMATH_CALUDE_position_vector_coefficients_l1610_161003


namespace NUMINAMATH_CALUDE_prove_x_value_l1610_161099

theorem prove_x_value (x a b c d : ℕ) 
  (h1 : x = a + 7)
  (h2 : a = b + 12)
  (h3 : b = c + 15)
  (h4 : c = d + 25)
  (h5 : d = 95) : x = 154 := by
  sorry

end NUMINAMATH_CALUDE_prove_x_value_l1610_161099


namespace NUMINAMATH_CALUDE_abc_product_one_l1610_161046

theorem abc_product_one (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (h_eq : a^2 + 1/b^2 = b^2 + 1/c^2 ∧ b^2 + 1/c^2 = c^2 + 1/a^2) :
  |a*b*c| = 1 := by
  sorry

end NUMINAMATH_CALUDE_abc_product_one_l1610_161046


namespace NUMINAMATH_CALUDE_complex_sum_theorem_l1610_161029

theorem complex_sum_theorem (a b c d : ℝ) (ω : ℂ) : 
  a ≠ -1 → b ≠ -1 → c ≠ -1 → d ≠ -1 →
  ω^3 = 1 →
  ω ≠ 1 →
  (1 / (a + ω)) + (1 / (b + ω)) + (1 / (c + ω)) + (1 / (d + ω)) = 3 / ω →
  (1 / (a + 1)) + (1 / (b + 1)) + (1 / (c + 1)) + (1 / (d + 1)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_theorem_l1610_161029


namespace NUMINAMATH_CALUDE_max_value_of_f_on_interval_l1610_161025

-- Define the function f(x) = x^3 - 3x^2
def f (x : ℝ) : ℝ := x^3 - 3*x^2

-- Define the interval [-2, 4]
def interval : Set ℝ := Set.Icc (-2) 4

-- State the theorem
theorem max_value_of_f_on_interval :
  ∃ (c : ℝ), c ∈ interval ∧ ∀ (x : ℝ), x ∈ interval → f x ≤ f c ∧ f c = 16 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_on_interval_l1610_161025


namespace NUMINAMATH_CALUDE_hiker_distance_l1610_161010

/-- Given a hiker's movements, calculate the final distance from the starting point -/
theorem hiker_distance (north east south west : ℝ) :
  north = 15 ∧ east = 8 ∧ south = 3 ∧ west = 4 →
  Real.sqrt ((north - south)^2 + (east - west)^2) = 4 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_hiker_distance_l1610_161010


namespace NUMINAMATH_CALUDE_calculate_total_cost_l1610_161011

/-- The cost of a single movie ticket in dollars -/
def movie_ticket_cost : ℕ := 30

/-- The number of movie tickets -/
def num_movie_tickets : ℕ := 8

/-- The number of football game tickets -/
def num_football_tickets : ℕ := 5

/-- The total cost of buying movie tickets and football game tickets -/
def total_cost : ℕ := 840

/-- Theorem stating the total cost of buying movie and football game tickets -/
theorem calculate_total_cost :
  (num_movie_tickets * movie_ticket_cost) + 
  (num_football_tickets * (num_movie_tickets * movie_ticket_cost / 2)) = total_cost :=
by sorry

end NUMINAMATH_CALUDE_calculate_total_cost_l1610_161011


namespace NUMINAMATH_CALUDE_smallest_angle_theorem_l1610_161096

open Real

theorem smallest_angle_theorem : 
  let θ : ℝ := 90
  ∀ φ : ℝ, φ > 0 → φ < θ → 
    cos (φ * π / 180) ≠ sin (50 * π / 180) + cos (32 * π / 180) - sin (22 * π / 180) - cos (16 * π / 180) →
    cos (θ * π / 180) = sin (50 * π / 180) + cos (32 * π / 180) - sin (22 * π / 180) - cos (16 * π / 180) :=
by sorry

end NUMINAMATH_CALUDE_smallest_angle_theorem_l1610_161096


namespace NUMINAMATH_CALUDE_total_molecular_weight_eq_1284_07_l1610_161020

/-- Atomic weights in g/mol -/
def atomic_weight (element : String) : ℝ :=
  match element with
  | "Ca" => 40.08
  | "O"  => 16.00
  | "H"  => 1.01
  | "Al" => 26.98
  | "S"  => 32.07
  | "K"  => 39.10
  | "N"  => 14.01
  | _    => 0

/-- Molecular weight of Ca(OH)2 in g/mol -/
def mw_calcium_hydroxide : ℝ :=
  atomic_weight "Ca" + 2 * (atomic_weight "O" + atomic_weight "H")

/-- Molecular weight of Al2(SO4)3 in g/mol -/
def mw_aluminum_sulfate : ℝ :=
  2 * atomic_weight "Al" + 3 * (atomic_weight "S" + 4 * atomic_weight "O")

/-- Molecular weight of KNO3 in g/mol -/
def mw_potassium_nitrate : ℝ :=
  atomic_weight "K" + atomic_weight "N" + 3 * atomic_weight "O"

/-- Total molecular weight of the mixture in grams -/
def total_molecular_weight : ℝ :=
  4 * mw_calcium_hydroxide + 2 * mw_aluminum_sulfate + 3 * mw_potassium_nitrate

theorem total_molecular_weight_eq_1284_07 :
  total_molecular_weight = 1284.07 := by
  sorry


end NUMINAMATH_CALUDE_total_molecular_weight_eq_1284_07_l1610_161020


namespace NUMINAMATH_CALUDE_cube_root_of_negative_eight_l1610_161068

theorem cube_root_of_negative_eight (x : ℝ) : x^3 = -8 → x = -2 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_negative_eight_l1610_161068


namespace NUMINAMATH_CALUDE_fraction_product_theorem_l1610_161039

theorem fraction_product_theorem : 
  (5 / 4 : ℚ) * (8 / 16 : ℚ) * (20 / 12 : ℚ) * (32 / 64 : ℚ) * 
  (50 / 20 : ℚ) * (40 / 80 : ℚ) * (70 / 28 : ℚ) * (48 / 96 : ℚ) = 625 / 768 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_theorem_l1610_161039


namespace NUMINAMATH_CALUDE_equation_solution_l1610_161075

theorem equation_solution : 
  let f (x : ℝ) := 1 / ((x - 1) * (x - 3)) + 1 / ((x - 3) * (x - 5)) + 1 / ((x - 5) * (x - 7))
  ∀ x : ℝ, f x = 1/8 ↔ x = -2 * Real.sqrt 14 ∨ x = 2 * Real.sqrt 14 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1610_161075


namespace NUMINAMATH_CALUDE_white_paint_calculation_l1610_161001

theorem white_paint_calculation (total_paint blue_paint : ℕ) 
  (h1 : total_paint = 6689)
  (h2 : blue_paint = 6029) :
  total_paint - blue_paint = 660 := by
  sorry

end NUMINAMATH_CALUDE_white_paint_calculation_l1610_161001


namespace NUMINAMATH_CALUDE_smallest_divisible_by_15_16_18_l1610_161006

theorem smallest_divisible_by_15_16_18 : ∃ (n : ℕ), n > 0 ∧ 15 ∣ n ∧ 16 ∣ n ∧ 18 ∣ n ∧ ∀ (m : ℕ), m > 0 → 15 ∣ m → 16 ∣ m → 18 ∣ m → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_15_16_18_l1610_161006


namespace NUMINAMATH_CALUDE_rectangle_property_l1610_161083

-- Define the rectangle's properties
def rectangle_length (x : ℝ) : ℝ := 4 * x
def rectangle_width (x : ℝ) : ℝ := x + 3

-- Define the area and perimeter functions
def area (x : ℝ) : ℝ := rectangle_length x * rectangle_width x
def perimeter (x : ℝ) : ℝ := 2 * (rectangle_length x + rectangle_width x)

-- State the theorem
theorem rectangle_property :
  ∃ x : ℝ, x > 0 ∧ area x = 3 * perimeter x ∧ Real.sqrt ((9 + Real.sqrt 153) / 4 - x) < 0.001 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_property_l1610_161083


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l1610_161048

theorem unique_solution_for_equation : ∃! (x y : ℕ), 1983 = 1982 * x - 1981 * y ∧ 1983 = 1982 * 31 * 5 - 1981 * (31 * 5 - 1) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l1610_161048


namespace NUMINAMATH_CALUDE_square_difference_given_linear_equations_l1610_161055

theorem square_difference_given_linear_equations (x y : ℝ) :
  (3 * x + 2 * y = 30) → (4 * x + 2 * y = 34) → x^2 - y^2 = -65 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_given_linear_equations_l1610_161055


namespace NUMINAMATH_CALUDE_line_passes_through_point_line_has_equal_intercepts_line_equation_is_correct_l1610_161047

/-- A line passing through point P(1,3) with equal x and y intercepts -/
def line_with_equal_intercepts : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 = 4}

theorem line_passes_through_point :
  (1, 3) ∈ line_with_equal_intercepts := by sorry

theorem line_has_equal_intercepts :
  ∃ a : ℝ, a ≠ 0 ∧ (a, 0) ∈ line_with_equal_intercepts ∧ (0, a) ∈ line_with_equal_intercepts := by sorry

theorem line_equation_is_correct :
  line_with_equal_intercepts = {p : ℝ × ℝ | p.1 + p.2 = 4} := by sorry

end NUMINAMATH_CALUDE_line_passes_through_point_line_has_equal_intercepts_line_equation_is_correct_l1610_161047


namespace NUMINAMATH_CALUDE_arrangement_probability_l1610_161093

def total_tiles : ℕ := 8
def x_tiles : ℕ := 5
def o_tiles : ℕ := 3

def specific_arrangement : List Char := ['X', 'X', 'X', 'O', 'O', 'X', 'O', 'X']

def probability_of_arrangement : ℚ := 1 / 56

theorem arrangement_probability :
  probability_of_arrangement = 1 / (total_tiles.choose x_tiles) :=
sorry

end NUMINAMATH_CALUDE_arrangement_probability_l1610_161093


namespace NUMINAMATH_CALUDE_smallest_number_of_eggs_l1610_161052

/-- The number of eggs in a full container -/
def full_container : ℕ := 12

/-- The number of containers with missing eggs -/
def containers_with_missing : ℕ := 2

/-- The number of eggs missing from each incomplete container -/
def eggs_missing_per_container : ℕ := 1

/-- The minimum number of eggs we're looking for -/
def min_eggs : ℕ := 106

theorem smallest_number_of_eggs :
  ∀ n : ℕ,
  n > 100 ∧
  ∃ c : ℕ, n = c * full_container - containers_with_missing * eggs_missing_per_container →
  n ≥ min_eggs :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_of_eggs_l1610_161052


namespace NUMINAMATH_CALUDE_three_solutions_imply_b_neg_c_zero_l1610_161094

theorem three_solutions_imply_b_neg_c_zero
  (f : ℝ → ℝ)
  (b c : ℝ)
  (h1 : ∀ x, f x = x^2 + b * |x| + c)
  (h2 : ∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f x = 0 ∧ f y = 0 ∧ f z = 0)
  (h3 : ∀ (w v : ℝ), f w = 0 ∧ f v = 0 ∧ w ≠ v → w = x ∨ w = y ∨ w = z ∨ v = x ∨ v = y ∨ v = z) :
  b < 0 ∧ c = 0 := by
sorry

end NUMINAMATH_CALUDE_three_solutions_imply_b_neg_c_zero_l1610_161094


namespace NUMINAMATH_CALUDE_max_value_of_function_l1610_161002

theorem max_value_of_function (f : ℝ → ℝ) (h : f = λ x => x + Real.sqrt 2 * Real.cos x) :
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi / 2 ∧
  ∀ (y : ℝ), 0 ≤ y ∧ y ≤ Real.pi / 2 → f y ≤ f x ∧
  f x = Real.pi / 4 + 1 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_function_l1610_161002


namespace NUMINAMATH_CALUDE_time_reduction_fraction_l1610_161098

theorem time_reduction_fraction (actual_speed : ℝ) (speed_increase : ℝ) : 
  actual_speed = 36.000000000000014 →
  speed_increase = 18 →
  (actual_speed / (actual_speed + speed_increase)) = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_time_reduction_fraction_l1610_161098


namespace NUMINAMATH_CALUDE_school_year_length_school_year_weeks_l1610_161081

theorem school_year_length 
  (num_children : ℕ) 
  (juice_boxes_per_child_per_day : ℕ) 
  (days_per_week : ℕ) 
  (total_juice_boxes : ℕ) : ℕ :=
  let juice_boxes_per_week := num_children * juice_boxes_per_child_per_day * days_per_week
  total_juice_boxes / juice_boxes_per_week

theorem school_year_weeks 
  (num_children : ℕ) 
  (juice_boxes_per_child_per_day : ℕ) 
  (days_per_week : ℕ) 
  (total_juice_boxes : ℕ) :
  school_year_length num_children juice_boxes_per_child_per_day days_per_week total_juice_boxes = 25 :=
by
  have h1 : num_children = 3 := by sorry
  have h2 : juice_boxes_per_child_per_day = 1 := by sorry
  have h3 : days_per_week = 5 := by sorry
  have h4 : total_juice_boxes = 375 := by sorry
  sorry

end NUMINAMATH_CALUDE_school_year_length_school_year_weeks_l1610_161081


namespace NUMINAMATH_CALUDE_units_digit_of_n_l1610_161054

/-- Given two natural numbers m and n, where mn = 17^6 and m has a units digit of 8,
    prove that the units digit of n is 2. -/
theorem units_digit_of_n (m n : ℕ) (h1 : m * n = 17^6) (h2 : m % 10 = 8) : n % 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_n_l1610_161054


namespace NUMINAMATH_CALUDE_statements_with_nonzero_solutions_l1610_161018

theorem statements_with_nonzero_solutions :
  ∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧
  (Real.sqrt (a^2 + b^2) = 3 * (a * b) ∨ Real.sqrt (a^2 + b^2) = 2 * (a * b)) ∧
  ¬(∃ (c d : ℝ), c ≠ 0 ∧ d ≠ 0 ∧
    (Real.sqrt (c^2 + d^2) = 2 * (c + d) ∨ Real.sqrt (c^2 + d^2) = (1/2) * (c + d))) :=
by
  sorry


end NUMINAMATH_CALUDE_statements_with_nonzero_solutions_l1610_161018


namespace NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_reciprocal_lt_one_l1610_161016

theorem x_gt_one_sufficient_not_necessary_for_reciprocal_lt_one :
  (∀ x : ℝ, x > 1 → 1 / x < 1) ∧
  (∃ x : ℝ, 1 / x < 1 ∧ x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_reciprocal_lt_one_l1610_161016


namespace NUMINAMATH_CALUDE_function_characterization_l1610_161058

theorem function_characterization (f : ℕ → ℕ) : 
  (∀ m n : ℕ, 2 * f (m * n) ≥ f (m^2 + n^2) - f m^2 - f n^2 ∧ 
               f (m^2 + n^2) - f m^2 - f n^2 ≥ 2 * f m * f n) → 
  (∀ n : ℕ, f n = n^2) := by
sorry

end NUMINAMATH_CALUDE_function_characterization_l1610_161058


namespace NUMINAMATH_CALUDE_jane_earnings_l1610_161071

def payment_per_bulb : ℚ := 0.50
def tulip_bulbs : ℕ := 20
def daffodil_bulbs : ℕ := 30

def iris_bulbs : ℕ := tulip_bulbs / 2
def crocus_bulbs : ℕ := daffodil_bulbs * 3

def total_bulbs : ℕ := tulip_bulbs + iris_bulbs + daffodil_bulbs + crocus_bulbs

def total_payment : ℚ := payment_per_bulb * total_bulbs

theorem jane_earnings : total_payment = 75 := by
  sorry

end NUMINAMATH_CALUDE_jane_earnings_l1610_161071


namespace NUMINAMATH_CALUDE_equation_solutions_l1610_161028

theorem equation_solutions : 
  {x : ℝ | (1 / (x^2 + 12*x - 9) + 1 / (x^2 + 3*x - 9) + 1 / (x^2 - 14*x - 9) = 0)} = {1, -9, 3, -3} := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1610_161028


namespace NUMINAMATH_CALUDE_max_people_in_line_l1610_161064

/-- Represents the state of the line at any given point -/
structure LineState where
  current : ℕ
  max : ℕ

/-- Updates the line state after people leave and join -/
def updateLine (state : LineState) (leave : ℕ) (join : ℕ) : LineState :=
  let remaining := state.current - min state.current leave
  let newCurrent := remaining + join
  { current := newCurrent, max := max state.max newCurrent }

/-- Repeats the process of people leaving and joining for a given number of times -/
def repeatProcess (initialState : LineState) (leave : ℕ) (join : ℕ) (times : ℕ) : LineState :=
  match times with
  | 0 => initialState
  | n + 1 => repeatProcess (updateLine initialState leave join) leave join n

/-- Calculates the final state after the entire process -/
def finalState (initial : ℕ) (leave : ℕ) (join : ℕ) (repetitions : ℕ) : LineState :=
  let initialState : LineState := { current := initial, max := initial }
  let afterRepetitions := repeatProcess initialState leave join repetitions
  let additionalJoin := afterRepetitions.current / 10  -- 10% rounded down
  updateLine afterRepetitions 0 additionalJoin

/-- Theorem stating that the maximum number of people in line is equal to the initial number -/
theorem max_people_in_line (initial : ℕ) (leave : ℕ) (join : ℕ) (repetitions : ℕ) 
    (h_initial : initial = 9) (h_leave : leave = 6) (h_join : join = 3) (h_repetitions : repetitions = 3) :
    (finalState initial leave join repetitions).max = initial := by
  sorry

end NUMINAMATH_CALUDE_max_people_in_line_l1610_161064


namespace NUMINAMATH_CALUDE_snail_distance_is_31_l1610_161066

def snail_path : List ℤ := [3, -5, 10, 2]

def distance (a b : ℤ) : ℕ := Int.natAbs (b - a)

def total_distance (path : List ℤ) : ℕ :=
  match path with
  | [] => 0
  | [_] => 0
  | x :: y :: rest => distance x y + total_distance (y :: rest)

theorem snail_distance_is_31 : total_distance snail_path = 31 := by
  sorry

end NUMINAMATH_CALUDE_snail_distance_is_31_l1610_161066


namespace NUMINAMATH_CALUDE_hyperbola_properties_l1610_161051

/-- An equilateral hyperbola with foci on the x-axis passing through (4, -2) -/
def equilateralHyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 = 12

theorem hyperbola_properties :
  -- The hyperbola is equilateral
  ∀ (x y : ℝ), equilateralHyperbola x y → x^2 - y^2 = 12 ∧
  -- The foci are on the x-axis (implied by the equation form)
  -- The hyperbola passes through the point (4, -2)
  equilateralHyperbola 4 (-2) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l1610_161051


namespace NUMINAMATH_CALUDE_fraction_value_l1610_161044

theorem fraction_value (p q : ℚ) (x : ℚ) 
  (h1 : p / q = 4 / 5)
  (h2 : x + (2 * q - p) / (2 * q + p) = 4) :
  x = 25 / 7 := by
sorry

end NUMINAMATH_CALUDE_fraction_value_l1610_161044


namespace NUMINAMATH_CALUDE_round_table_seating_l1610_161012

/-- The number of unique circular arrangements of n distinct objects -/
def circularArrangements (n : ℕ) : ℕ := Nat.factorial (n - 1)

/-- The number of people to be seated around the round table -/
def numberOfPeople : ℕ := 8

theorem round_table_seating :
  circularArrangements numberOfPeople = 5040 := by
  sorry

end NUMINAMATH_CALUDE_round_table_seating_l1610_161012


namespace NUMINAMATH_CALUDE_cos_180_deg_l1610_161038

-- Define cosine function for angles in degrees
noncomputable def cos_deg (θ : ℝ) : ℝ := 
  Real.cos (θ * Real.pi / 180)

-- Theorem statement
theorem cos_180_deg : cos_deg 180 = -1 := by
  sorry

end NUMINAMATH_CALUDE_cos_180_deg_l1610_161038


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1610_161092

theorem quadratic_inequality_solution (a b : ℝ) (ha : a ≠ 0) :
  (∀ x, (a * x^2 + b * x + 2 < 0) ↔ (x < -1/2 ∨ x > 1/3)) →
  (a - b) / a = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1610_161092


namespace NUMINAMATH_CALUDE_birds_on_fence_l1610_161032

theorem birds_on_fence (initial_birds : ℕ) (total_birds : ℕ) (new_birds : ℕ) : 
  initial_birds = 2 → total_birds = 6 → new_birds = total_birds - initial_birds → new_birds = 4 := by
  sorry

end NUMINAMATH_CALUDE_birds_on_fence_l1610_161032


namespace NUMINAMATH_CALUDE_kara_water_consumption_l1610_161050

/-- The amount of water Kara drinks with each medication dose in ounces -/
def water_per_dose : ℕ := 4

/-- The number of times Kara takes her medication per day -/
def doses_per_day : ℕ := 3

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of times Kara forgot to take her medication in the second week -/
def forgotten_doses : ℕ := 2

/-- The total amount of water Kara drank with her medication over two weeks -/
def total_water : ℕ := 
  (water_per_dose * doses_per_day * days_per_week) + 
  (water_per_dose * (doses_per_day * days_per_week - forgotten_doses))

theorem kara_water_consumption : total_water = 160 := by
  sorry

end NUMINAMATH_CALUDE_kara_water_consumption_l1610_161050


namespace NUMINAMATH_CALUDE_chris_birthday_money_l1610_161035

/-- Calculates the total amount of money Chris has after receiving birthday gifts -/
def total_money (initial_amount grandmother_gift aunt_uncle_gift parents_gift : ℕ) : ℕ :=
  initial_amount + grandmother_gift + aunt_uncle_gift + parents_gift

/-- Proves that Chris's total money after receiving gifts is correct -/
theorem chris_birthday_money :
  total_money 159 25 20 75 = 279 := by
  sorry

end NUMINAMATH_CALUDE_chris_birthday_money_l1610_161035


namespace NUMINAMATH_CALUDE_sum_even_minus_odd_product_equals_6401_l1610_161015

def sum_of_integers (a b : ℕ) : ℕ :=
  (b - a + 1) * (a + b) / 2

def count_even_integers (a b : ℕ) : ℕ :=
  (b - a) / 2 + 1

def product_of_odd_integers (a b : ℕ) : ℕ :=
  if ∃ n ∈ Finset.range (b - a + 1), Even (a + n) then 0 else 1

theorem sum_even_minus_odd_product_equals_6401 :
  sum_of_integers 100 150 + count_even_integers 100 150 - product_of_odd_integers 100 150 = 6401 := by
  sorry

end NUMINAMATH_CALUDE_sum_even_minus_odd_product_equals_6401_l1610_161015


namespace NUMINAMATH_CALUDE_base8_4532_to_decimal_l1610_161034

/-- Converts a base 8 digit to its decimal value -/
def base8_to_decimal (digit : Nat) : Nat :=
  if digit < 8 then digit else 0

/-- Converts a list of base 8 digits to a decimal number -/
def base8_list_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + base8_to_decimal d * (8 ^ i)) 0

theorem base8_4532_to_decimal :
  base8_list_to_decimal [2, 3, 5, 4] = 2394 := by
  sorry

end NUMINAMATH_CALUDE_base8_4532_to_decimal_l1610_161034


namespace NUMINAMATH_CALUDE_operations_are_finite_l1610_161067

/-- Represents a (2n+1)-gon with integers assigned to its vertices -/
structure Polygon (n : ℕ) where
  vertices : Fin (2*n+1) → ℤ
  sum_positive : 0 < (Finset.univ.sum vertices)

/-- Represents an operation on three consecutive vertices -/
def operation (p : Polygon n) (i : Fin (2*n+1)) : Polygon n :=
  sorry

/-- Predicate to check if an operation is valid (i.e., y < 0) -/
def is_valid_operation (p : Polygon n) (i : Fin (2*n+1)) : Prop :=
  sorry

/-- A sequence of operations -/
def operation_sequence (p : Polygon n) : List (Fin (2*n+1)) → Polygon n
  | [] => p
  | (i :: is) => operation_sequence (operation p i) is

/-- Theorem stating that any sequence of valid operations is finite -/
theorem operations_are_finite (n : ℕ) (p : Polygon n) :
  ∃ (N : ℕ), ∀ (seq : List (Fin (2*n+1))),
    (∀ i ∈ seq, is_valid_operation p i) →
    seq.length ≤ N :=
  sorry

end NUMINAMATH_CALUDE_operations_are_finite_l1610_161067


namespace NUMINAMATH_CALUDE_choose_president_and_vice_president_l1610_161060

/-- The number of candidates for class president -/
def president_candidates : ℕ := 3

/-- The number of candidates for vice president -/
def vice_president_candidates : ℕ := 5

/-- The total number of ways to choose one class president and one vice president -/
def total_ways : ℕ := president_candidates * vice_president_candidates

theorem choose_president_and_vice_president :
  total_ways = 15 :=
sorry

end NUMINAMATH_CALUDE_choose_president_and_vice_president_l1610_161060


namespace NUMINAMATH_CALUDE_total_questions_in_three_hours_l1610_161013

/-- The number of questions Bob creates in the first hour -/
def first_hour_questions : ℕ := 13

/-- Calculates the number of questions created in the second hour -/
def second_hour_questions : ℕ := 2 * first_hour_questions

/-- Calculates the number of questions created in the third hour -/
def third_hour_questions : ℕ := 2 * second_hour_questions

/-- Theorem: The total number of questions Bob creates in three hours is 91 -/
theorem total_questions_in_three_hours :
  first_hour_questions + second_hour_questions + third_hour_questions = 91 := by
  sorry

end NUMINAMATH_CALUDE_total_questions_in_three_hours_l1610_161013


namespace NUMINAMATH_CALUDE_f_iterated_property_l1610_161084

-- Define the function f
def f (p q : ℝ) (x : ℝ) : ℝ := x^2 + p*x + q

-- Define the iteration of f
def iterate_f (p q : ℝ) : ℕ → ℝ → ℝ
  | 0, x => x
  | n+1, x => iterate_f p q n (f p q x)

theorem f_iterated_property (p q : ℝ) 
  (h : ∀ x ∈ Set.Icc 1 3, |f p q x| ≤ 1/2) :
  iterate_f p q 2017 ((3 + Real.sqrt 7) / 2) = (3 - Real.sqrt 7) / 2 := by
  sorry

end NUMINAMATH_CALUDE_f_iterated_property_l1610_161084


namespace NUMINAMATH_CALUDE_inequalities_comparison_l1610_161056

theorem inequalities_comparison (a b : ℝ) : 
  (∃ a b : ℝ, a + b < 2) ∧ 
  (∀ a b : ℝ, a^2 + b^2 ≥ 2*a*b) ∧ 
  (∀ a b : ℝ, a*b ≤ ((a + b)/2)^2) ∧ 
  (∀ a b : ℝ, |a| + |b| ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_inequalities_comparison_l1610_161056
