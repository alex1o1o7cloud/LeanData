import Mathlib

namespace NUMINAMATH_CALUDE_smallest_sum_of_roots_l2292_229220

theorem smallest_sum_of_roots (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h1 : ∃ x : ℝ, x^2 + a*x + 3*b = 0)
  (h2 : ∃ x : ℝ, x^2 + 3*b*x + a = 0) :
  a + b ≥ 6.5 := by
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_roots_l2292_229220


namespace NUMINAMATH_CALUDE_village_population_equality_l2292_229254

/-- Represents the population change in a village over time. -/
structure VillagePopulation where
  initial : ℕ  -- Initial population
  rate : ℤ     -- Annual rate of change (positive for increase, negative for decrease)

/-- Calculates the population after a given number of years. -/
def population_after (v : VillagePopulation) (years : ℕ) : ℤ :=
  v.initial + v.rate * years

theorem village_population_equality (village_x village_y : VillagePopulation) 
  (h1 : village_x.initial = 78000)
  (h2 : village_x.rate = -1200)
  (h3 : village_y.initial = 42000)
  (h4 : population_after village_x 18 = population_after village_y 18) :
  village_y.rate = 800 := by
  sorry

#check village_population_equality

end NUMINAMATH_CALUDE_village_population_equality_l2292_229254


namespace NUMINAMATH_CALUDE_matrix_equation_l2292_229244

def A : Matrix (Fin 2) (Fin 2) ℝ := !![1, 2; 3, 4]

def B (a b c d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![a, b; c, d]

theorem matrix_equation (a b c d : ℝ) (h1 : A * B a b c d = B a b c d * A) 
  (h2 : 4 * b ≠ c) : (a - 2 * d) / (c - 4 * b) = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_matrix_equation_l2292_229244


namespace NUMINAMATH_CALUDE_line_hyperbola_intersection_range_l2292_229242

/-- The range of k values for which the line y = kx + 1 intersects the right branch of the hyperbola 3x^2 - y^2 = 3 at two distinct points -/
theorem line_hyperbola_intersection_range (k : ℝ) : 
  (∃ x₁ x₂ y₁ y₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧
    y₁ = k * x₁ + 1 ∧ y₂ = k * x₂ + 1 ∧
    3 * x₁^2 - y₁^2 = 3 ∧ 3 * x₂^2 - y₂^2 = 3) ↔
  -2 < k ∧ k < -Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_line_hyperbola_intersection_range_l2292_229242


namespace NUMINAMATH_CALUDE_smallest_n_cookies_l2292_229243

theorem smallest_n_cookies (n : ℕ) : (∀ m : ℕ, m > 0 → (15 * m - 3) % 7 ≠ 0) ∨ 
  ((15 * n - 3) % 7 = 0 ∧ n > 0 ∧ ∀ m : ℕ, 0 < m ∧ m < n → (15 * m - 3) % 7 ≠ 0) ↔ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_cookies_l2292_229243


namespace NUMINAMATH_CALUDE_newspaper_spend_l2292_229224

/-- The cost of a weekday newspaper edition -/
def weekday_cost : ℚ := 0.50

/-- The cost of a Sunday newspaper edition -/
def sunday_cost : ℚ := 2.00

/-- The number of weekday editions Hillary buys per week -/
def weekday_editions : ℕ := 3

/-- The number of weeks -/
def weeks : ℕ := 8

/-- Hillary's total newspaper spend over 8 weeks -/
def total_spend : ℚ := weeks * (weekday_editions * weekday_cost + sunday_cost)

theorem newspaper_spend : total_spend = 28 := by
  sorry

end NUMINAMATH_CALUDE_newspaper_spend_l2292_229224


namespace NUMINAMATH_CALUDE_inscribed_circle_segment_ratios_l2292_229280

/-- Given a triangle with sides in ratio 5:4:3, prove the ratios of segments divided by tangent points of inscribed circle -/
theorem inscribed_circle_segment_ratios (a b c : ℝ) (h : a / b = 5 / 4 ∧ b / c = 4 / 3) :
  let r := (a + b - c) / 2
  let s := (a + b + c) / 2
  (r / (s - b), r / (s - c), (s - c) / (s - b)) = (1 / 3, 1 / 2, 2 / 3) := by
  sorry


end NUMINAMATH_CALUDE_inscribed_circle_segment_ratios_l2292_229280


namespace NUMINAMATH_CALUDE_unemployment_rate_after_changes_l2292_229221

theorem unemployment_rate_after_changes (initial_unemployment : ℝ) : 
  initial_unemployment ≥ 0 ∧ initial_unemployment ≤ 100 →
  1.1 * initial_unemployment + 0.85 * (100 - initial_unemployment) = 100 →
  1.1 * initial_unemployment = 66 :=
by sorry

end NUMINAMATH_CALUDE_unemployment_rate_after_changes_l2292_229221


namespace NUMINAMATH_CALUDE_toott_permutations_eq_ten_l2292_229262

/-- The number of distinct permutations of the letters in "TOOTT" -/
def toott_permutations : ℕ :=
  Nat.factorial 5 / (Nat.factorial 3 * Nat.factorial 2)

/-- Theorem stating that the number of distinct permutations of "TOOTT" is 10 -/
theorem toott_permutations_eq_ten : toott_permutations = 10 := by
  sorry

end NUMINAMATH_CALUDE_toott_permutations_eq_ten_l2292_229262


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l2292_229298

theorem quadratic_real_roots_condition (a : ℝ) :
  (∃ x : ℝ, a * x^2 - 2*x + 1 = 0) ↔ (a ≤ 1 ∧ a ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l2292_229298


namespace NUMINAMATH_CALUDE_point_on_line_l2292_229251

/-- A point (x, y) lies on the line passing through (2, -4) and (8, 16) if and only if y = (10/3)x - 32/3 -/
theorem point_on_line (x y : ℝ) : 
  (y = (10/3)*x - 32/3) ↔ 
  (∃ t : ℝ, x = 2 + 6*t ∧ y = -4 + 20*t) :=
by sorry

end NUMINAMATH_CALUDE_point_on_line_l2292_229251


namespace NUMINAMATH_CALUDE_ratio_of_numbers_l2292_229236

theorem ratio_of_numbers (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) (h4 : a + b = 7 * (a - b)) :
  a / b = 2 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_numbers_l2292_229236


namespace NUMINAMATH_CALUDE_linear_function_point_relation_l2292_229282

/-- Given a linear function y = -x + 6 and two points A(-1, y₁) and B(2, y₂) on its graph, prove that y₁ > y₂ -/
theorem linear_function_point_relation (y₁ y₂ : ℝ) : 
  (∀ x : ℝ, -x + 6 = y₁ → x = -1) →  -- Point A(-1, y₁) is on the graph
  (∀ x : ℝ, -x + 6 = y₂ → x = 2) →   -- Point B(2, y₂) is on the graph
  y₁ > y₂ := by
sorry

end NUMINAMATH_CALUDE_linear_function_point_relation_l2292_229282


namespace NUMINAMATH_CALUDE_prob_A_wins_in_4_points_prob_game_ends_in_5_points_l2292_229267

/-- Represents the state of the game -/
inductive GameState
  | A_Serving
  | B_Serving

/-- Represents the outcome of a single point -/
inductive PointOutcome
  | A_Wins
  | B_Wins

/-- Game configuration -/
structure GameConfig where
  prob_A_wins_when_serving : ℝ
  prob_A_wins_when_B_serving : ℝ

/-- Game state after a certain number of points -/
structure GameSequence where
  state : GameState
  A_consecutive_wins : ℕ
  B_consecutive_wins : ℕ

/-- Function to determine if the game has ended -/
def is_game_over (seq : GameSequence) : Bool :=
  seq.A_consecutive_wins = 2 ∨ seq.B_consecutive_wins = 2

/-- Function to calculate the probability of a specific sequence of point outcomes -/
def sequence_probability (config : GameConfig) (outcomes : List PointOutcome) : ℝ :=
  sorry

/-- Theorem: Probability of player A winning a game that lasts for 4 points -/
theorem prob_A_wins_in_4_points (config : GameConfig) 
  (h1 : config.prob_A_wins_when_serving = 2/3)
  (h2 : config.prob_A_wins_when_B_serving = 1/4) :
  sequence_probability config [PointOutcome.B_Wins, PointOutcome.A_Wins, PointOutcome.A_Wins, PointOutcome.A_Wins] = 1/12 :=
sorry

/-- Theorem: Probability of the game ending after 5 points -/
theorem prob_game_ends_in_5_points (config : GameConfig)
  (h1 : config.prob_A_wins_when_serving = 2/3)
  (h2 : config.prob_A_wins_when_B_serving = 1/4) :
  (sequence_probability config [PointOutcome.B_Wins, PointOutcome.B_Wins, PointOutcome.A_Wins, PointOutcome.A_Wins, PointOutcome.A_Wins] +
   sequence_probability config [PointOutcome.A_Wins, PointOutcome.B_Wins, PointOutcome.A_Wins, PointOutcome.B_Wins, PointOutcome.B_Wins]) = 19/216 :=
sorry

end NUMINAMATH_CALUDE_prob_A_wins_in_4_points_prob_game_ends_in_5_points_l2292_229267


namespace NUMINAMATH_CALUDE_subcommittee_formation_count_l2292_229219

def senate_committee_ways (total_republicans : ℕ) (total_democrats : ℕ) 
  (subcommittee_republicans : ℕ) (subcommittee_democrats : ℕ) : ℕ :=
  Nat.choose total_republicans subcommittee_republicans * 
  Nat.choose total_democrats subcommittee_democrats

theorem subcommittee_formation_count : 
  senate_committee_ways 10 8 4 3 = 11760 := by
  sorry

end NUMINAMATH_CALUDE_subcommittee_formation_count_l2292_229219


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l2292_229200

/-- Given a train crossing a bridge, calculate the length of the bridge -/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : 
  train_length = 145 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600) * crossing_time - train_length = 230 :=
by
  sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l2292_229200


namespace NUMINAMATH_CALUDE_floor_sum_inequality_frac_sum_inequality_frac_periodic_l2292_229266

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- Define the fractional part function
noncomputable def frac (x : ℝ) : ℝ :=
  x - floor x

-- Theorem statements
theorem floor_sum_inequality (x y : ℝ) :
  floor x + floor y ≤ floor (x + y) := by sorry

theorem frac_sum_inequality (x y : ℝ) :
  frac x + frac y ≥ frac (x + y) := by sorry

theorem frac_periodic (x : ℝ) :
  frac (x + 1) = frac x := by sorry

end NUMINAMATH_CALUDE_floor_sum_inequality_frac_sum_inequality_frac_periodic_l2292_229266


namespace NUMINAMATH_CALUDE_gain_percentage_l2292_229241

/-- 
If the cost price of 50 articles equals the selling price of 46 articles, 
then the gain percentage is (1/11.5) * 100.
-/
theorem gain_percentage (C S : ℝ) (h : 50 * C = 46 * S) : 
  (S - C) / C * 100 = (1 / 11.5) * 100 := by
  sorry

end NUMINAMATH_CALUDE_gain_percentage_l2292_229241


namespace NUMINAMATH_CALUDE_piggy_bank_compartments_l2292_229253

/-- Given a piggy bank with an unknown number of compartments, prove that the number of compartments is 12 based on the given conditions. -/
theorem piggy_bank_compartments :
  ∀ (c : ℕ), -- c represents the number of compartments
  (∀ (i : ℕ), i < c → 2 = 2) → -- Each compartment initially has 2 pennies (this is a trivial condition in Lean)
  (∀ (i : ℕ), i < c → 6 = 6) → -- 6 pennies are added to each compartment (also trivial in Lean)
  (c * (2 + 6) = 96) →         -- Total pennies after adding is 96
  c = 12 := by
sorry


end NUMINAMATH_CALUDE_piggy_bank_compartments_l2292_229253


namespace NUMINAMATH_CALUDE_octagon_area_l2292_229247

/-- The area of a regular octagon inscribed in a square -/
theorem octagon_area (s : ℝ) (h : s = 4 + 2 * Real.sqrt 2) :
  let octagon_side := 2 * Real.sqrt 2
  let square_area := s^2
  let triangle_area := 2
  square_area - 4 * triangle_area = 16 + 8 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_octagon_area_l2292_229247


namespace NUMINAMATH_CALUDE_election_votes_l2292_229259

theorem election_votes (votes1 votes3 : ℕ) (winning_percentage : ℚ) 
  (h1 : votes1 = 1136)
  (h2 : votes3 = 11628)
  (h3 : winning_percentage = 55371428571428574 / 100000000000000000)
  (h4 : votes3 > votes1)
  (h5 : ↑votes3 = winning_percentage * ↑(votes1 + votes3 + votes2)) :
  ∃ votes2 : ℕ, votes2 = 8236 := by sorry

end NUMINAMATH_CALUDE_election_votes_l2292_229259


namespace NUMINAMATH_CALUDE_nearest_integer_to_3_plus_sqrt2_to_6_l2292_229234

theorem nearest_integer_to_3_plus_sqrt2_to_6 :
  ∃ n : ℤ, ∀ m : ℤ, |((3 : ℝ) + Real.sqrt 2)^6 - n| ≤ |((3 : ℝ) + Real.sqrt 2)^6 - m| ∧ n = 3707 :=
by sorry

end NUMINAMATH_CALUDE_nearest_integer_to_3_plus_sqrt2_to_6_l2292_229234


namespace NUMINAMATH_CALUDE_strawberry_sale_revenue_difference_l2292_229202

/-- Represents the sale of strawberries at a supermarket -/
structure StrawberrySale where
  pints_sold : ℕ
  sale_revenue : ℕ
  price_difference : ℕ

/-- Calculates the revenue difference between regular and sale prices -/
def revenue_difference (sale : StrawberrySale) : ℕ :=
  let sale_price := sale.sale_revenue / sale.pints_sold
  let regular_price := sale_price + sale.price_difference
  regular_price * sale.pints_sold - sale.sale_revenue

/-- Theorem stating the revenue difference for the given scenario -/
theorem strawberry_sale_revenue_difference :
  ∃ (sale : StrawberrySale),
    sale.pints_sold = 54 ∧
    sale.sale_revenue = 216 ∧
    sale.price_difference = 2 ∧
    revenue_difference sale = 108 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_sale_revenue_difference_l2292_229202


namespace NUMINAMATH_CALUDE_parabola_zeros_difference_l2292_229238

/-- Represents a quadratic function of the form ax² + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the y-coordinate for a given x-coordinate on the quadratic function -/
def QuadraticFunction.eval (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

/-- Represents the zeros of a quadratic function -/
structure QuadraticZeros where
  m : ℝ
  n : ℝ
  h_order : m > n

theorem parabola_zeros_difference (f : QuadraticFunction) (zeros : QuadraticZeros) :
  f.eval 1 = -3 →
  f.eval 3 = 9 →
  f.eval zeros.m = 0 →
  f.eval zeros.n = 0 →
  zeros.m - zeros.n = 2 := by
  sorry


end NUMINAMATH_CALUDE_parabola_zeros_difference_l2292_229238


namespace NUMINAMATH_CALUDE_remainder_theorem_l2292_229222

theorem remainder_theorem (P D Q R Q' R' : ℕ) (hD : D > 1) 
  (h1 : P = Q * D + R) (h2 : Q = (D - 1) * Q' + R') :
  P % (D * (D - 1)) = D * R' + R :=
by sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2292_229222


namespace NUMINAMATH_CALUDE_picnic_men_count_l2292_229205

/-- Given a picnic with 240 people, where there are 40 more men than women
    and 40 more adults than children, prove that there are 90 men. -/
theorem picnic_men_count :
  ∀ (men women adults children : ℕ),
    men + women + children = 240 →
    men = women + 40 →
    adults = children + 40 →
    men + women = adults →
    men = 90 := by
  sorry

end NUMINAMATH_CALUDE_picnic_men_count_l2292_229205


namespace NUMINAMATH_CALUDE_unique_n_satisfying_conditions_l2292_229258

theorem unique_n_satisfying_conditions : ∃! (n : ℕ), n ≥ 1 ∧
  ∃ (a b : ℕ+), 
    (∀ (p : ℕ), Prime p → ¬(p^3 ∣ (a.val^2 + b.val + 3))) ∧
    ((a.val * b.val + 3 * b.val + 8) : ℚ) / (a.val^2 + b.val + 3) = n ∧
    n = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_n_satisfying_conditions_l2292_229258


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l2292_229277

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a line in the form ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Function to check if a point is on a line
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Function to check if two lines are perpendicular
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

-- Theorem statement
theorem perpendicular_line_through_point :
  let givenLine : Line := { a := 2, b := 1, c := -5 }
  let point : Point := { x := 3, y := 0 }
  let perpendicularLine : Line := { a := 1, b := -2, c := 3 }
  perpendicular givenLine perpendicularLine ∧ 
  pointOnLine point perpendicularLine := by sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l2292_229277


namespace NUMINAMATH_CALUDE_bus_children_difference_l2292_229294

/-- Proves that the difference in children on the bus before and after a stop is 23 -/
theorem bus_children_difference (initial_count : Nat) (final_count : Nat)
    (h1 : initial_count = 41)
    (h2 : final_count = 18) :
    initial_count - final_count = 23 := by
  sorry

end NUMINAMATH_CALUDE_bus_children_difference_l2292_229294


namespace NUMINAMATH_CALUDE_weighted_mean_calculation_l2292_229255

def numbers : List ℝ := [16, 28, 45]
def weights : List ℝ := [2, 3, 5]

theorem weighted_mean_calculation :
  (List.sum (List.zipWith (· * ·) numbers weights)) / (List.sum weights) = 34.1 := by
  sorry

end NUMINAMATH_CALUDE_weighted_mean_calculation_l2292_229255


namespace NUMINAMATH_CALUDE_lines_perpendicular_iff_product_slopes_neg_one_l2292_229273

/-- Two lines y = k₁x + l₁ and y = k₂x + l₂, where k₁ ≠ 0 and k₂ ≠ 0, are perpendicular if and only if k₁k₂ = -1 -/
theorem lines_perpendicular_iff_product_slopes_neg_one
  (k₁ k₂ l₁ l₂ : ℝ) (hk₁ : k₁ ≠ 0) (hk₂ : k₂ ≠ 0) :
  (∃ (x y : ℝ), y = k₁ * x + l₁ ∧ y = k₂ * x + l₂) →
  (∀ (x₁ y₁ x₂ y₂ : ℝ),
    y₁ = k₁ * x₁ + l₁ →
    y₂ = k₂ * x₂ + l₂ →
    (x₂ - x₁) * (y₂ - y₁) = 0) ↔
  k₁ * k₂ = -1 := by
sorry

end NUMINAMATH_CALUDE_lines_perpendicular_iff_product_slopes_neg_one_l2292_229273


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2292_229215

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if c = 2√3, sin B = 2 sin A, and C = π/3, then a = 2, b = 4, and the area is 2√3 -/
theorem triangle_abc_properties (a b c A B C : ℝ) : 
  c = 2 * Real.sqrt 3 →
  Real.sin B = 2 * Real.sin A →
  C = π / 3 →
  (a = 2 ∧ b = 4 ∧ (1/2) * a * b * Real.sin C = 2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l2292_229215


namespace NUMINAMATH_CALUDE_equal_remainders_implies_m_zero_l2292_229230

-- Define the polynomials
def P₁ (m : ℝ) (y : ℝ) : ℝ := 29 * 42 * y^2 + m * y + 2
def P₂ (m : ℝ) (y : ℝ) : ℝ := y^2 + m * y + 2

-- Define the remainder functions
def R₁ (m : ℝ) : ℝ := P₁ m 1
def R₂ (m : ℝ) : ℝ := P₂ m (-1)

-- Theorem statement
theorem equal_remainders_implies_m_zero :
  ∀ m : ℝ, R₁ m = R₂ m → m = 0 :=
by sorry

end NUMINAMATH_CALUDE_equal_remainders_implies_m_zero_l2292_229230


namespace NUMINAMATH_CALUDE_remainder_sum_l2292_229231

theorem remainder_sum (a b : ℤ) (ha : a % 60 = 53) (hb : b % 45 = 17) : (a + b) % 15 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l2292_229231


namespace NUMINAMATH_CALUDE_sum_of_opposite_sign_integers_l2292_229248

theorem sum_of_opposite_sign_integers (a b : ℤ) : 
  (abs a = 3) → (abs b = 5) → (a * b < 0) → (a + b = -2 ∨ a + b = 2) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_opposite_sign_integers_l2292_229248


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2292_229239

/-- Given a hyperbola with foci F₁(-√5,0) and F₂(√5,0), and a point P on the hyperbola
    such that PF₁ · PF₂ = 0 and |PF₁| · |PF₂| = 2, the standard equation of the hyperbola
    is x²/4 - y² = 1. -/
theorem hyperbola_equation (F₁ F₂ P : ℝ × ℝ) : 
  F₁ = (-Real.sqrt 5, 0) →
  F₂ = (Real.sqrt 5, 0) →
  (P.1 - F₁.1) * (P.1 - F₂.1) + (P.2 - F₁.2) * (P.2 - F₂.2) = 0 →
  Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) * 
    Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = 2 →
  ∃ (x y : ℝ), x^2 / 4 - y^2 = 1 ∧ 
    (x, y) = P :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2292_229239


namespace NUMINAMATH_CALUDE_distance_between_complex_points_l2292_229246

theorem distance_between_complex_points :
  let z₁ : ℂ := 2 + 3*I
  let z₂ : ℂ := -2 + 2*I
  Complex.abs (z₁ - z₂) = Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_complex_points_l2292_229246


namespace NUMINAMATH_CALUDE_price_quantity_change_cost_difference_l2292_229269

theorem price_quantity_change (P Q : ℝ) (h1 : P > 0) (h2 : Q > 0) : 
  P * Q * 1.1 * 0.9 = P * Q * 0.99 := by
sorry

theorem cost_difference (P Q : ℝ) (h1 : P > 0) (h2 : Q > 0) : 
  P * Q * 1.1 * 0.9 - P * Q = P * Q * (-0.01) := by
sorry

end NUMINAMATH_CALUDE_price_quantity_change_cost_difference_l2292_229269


namespace NUMINAMATH_CALUDE_no_prime_roots_l2292_229252

/-- A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

/-- The quadratic equation x^2 - 108x + k = 0 -/
def quadraticEquation (x k : ℝ) : Prop := x^2 - 108*x + k = 0

/-- Both roots of the quadratic equation are prime numbers -/
def bothRootsPrime (k : ℝ) : Prop :=
  ∃ (p q : ℕ), isPrime p ∧ isPrime q ∧ 
    (∀ x : ℝ, quadraticEquation x k ↔ x = p ∨ x = q)

/-- There are no values of k for which both roots of the quadratic equation are prime -/
theorem no_prime_roots : ¬∃ k : ℝ, bothRootsPrime k := by sorry

end NUMINAMATH_CALUDE_no_prime_roots_l2292_229252


namespace NUMINAMATH_CALUDE_function_equation_solution_l2292_229218

theorem function_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, (f x * f y - f (x * y)) / 4 = x + y + 3) : 
  ∀ x : ℝ, f x = x + 4 := by
  sorry

end NUMINAMATH_CALUDE_function_equation_solution_l2292_229218


namespace NUMINAMATH_CALUDE_geometric_sequence_17th_term_l2292_229240

/-- Given a geometric sequence where a₅ = 5 and a₁₁ = 40, prove that a₁₇ = 320 -/
theorem geometric_sequence_17th_term (a : ℕ → ℝ) (h1 : ∀ n, a (n + 1) / a n = a 2 / a 1) 
    (h2 : a 5 = 5) (h3 : a 11 = 40) : a 17 = 320 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_17th_term_l2292_229240


namespace NUMINAMATH_CALUDE_l_shaped_region_perimeter_l2292_229257

/-- Represents an L-shaped region with a staircase pattern --/
structure LShapedRegion where
  width : ℝ
  height : ℝ
  unit_length : ℝ
  num_steps : ℕ

/-- Calculates the area of the L-shaped region --/
def area (r : LShapedRegion) : ℝ :=
  r.width * r.height - (r.num_steps * r.unit_length^2)

/-- Calculates the perimeter of the L-shaped region --/
def perimeter (r : LShapedRegion) : ℝ :=
  r.width + r.height + r.num_steps * r.unit_length + r.unit_length * (r.num_steps + 1)

/-- Theorem stating that an L-shaped region with specific properties has a perimeter of 39.4 meters --/
theorem l_shaped_region_perimeter :
  ∀ (r : LShapedRegion),
    r.width = 10 ∧
    r.unit_length = 1 ∧
    r.num_steps = 10 ∧
    area r = 72 →
    perimeter r = 39.4 := by
  sorry


end NUMINAMATH_CALUDE_l_shaped_region_perimeter_l2292_229257


namespace NUMINAMATH_CALUDE_parallelogram_vertices_l2292_229278

/-- A parallelogram with two known vertices and one side parallel to x-axis -/
structure Parallelogram where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  parallel_to_x_axis : Bool

/-- The other pair of opposite vertices of the parallelogram -/
def other_vertices (p : Parallelogram) : (ℝ × ℝ) × (ℝ × ℝ) :=
  sorry

theorem parallelogram_vertices (p : Parallelogram) 
  (h1 : p.v1 = (2, -3)) 
  (h2 : p.v2 = (8, 9)) 
  (h3 : p.parallel_to_x_axis = true) : 
  other_vertices p = ((5, -3), (5, 9)) := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_vertices_l2292_229278


namespace NUMINAMATH_CALUDE_f_extrema_on_interval_l2292_229207

def f (x : ℝ) : ℝ := x^3 + 2*x

theorem f_extrema_on_interval :
  let a := -1
  let b := 1
  ∃ (x_min x_max : ℝ),
    x_min ∈ [a, b] ∧
    x_max ∈ [a, b] ∧
    (∀ x ∈ [a, b], f x ≥ f x_min) ∧
    (∀ x ∈ [a, b], f x ≤ f x_max) ∧
    f x_min = -3 ∧
    f x_max = 3 :=
sorry

end NUMINAMATH_CALUDE_f_extrema_on_interval_l2292_229207


namespace NUMINAMATH_CALUDE_condition_A_not_necessary_nor_sufficient_l2292_229263

-- Define the conditions
def condition_A (θ : Real) (a : Real) : Prop := Real.sqrt (1 + Real.sin θ) = a
def condition_B (θ : Real) (a : Real) : Prop := Real.sin (θ / 2) + Real.cos (θ / 2) = a

-- Theorem statement
theorem condition_A_not_necessary_nor_sufficient :
  ¬(∀ θ a, condition_B θ a → condition_A θ a) ∧
  ¬(∀ θ a, condition_A θ a → condition_B θ a) := by
  sorry

end NUMINAMATH_CALUDE_condition_A_not_necessary_nor_sufficient_l2292_229263


namespace NUMINAMATH_CALUDE_pin_purchase_cost_l2292_229223

/-- The total cost of pins with a discount -/
def total_cost (num_pins : ℕ) (regular_price : ℚ) (discount_percent : ℚ) : ℚ :=
  num_pins * (regular_price * (1 - discount_percent / 100))

/-- Theorem stating the total cost of 10 pins with a 15% discount -/
theorem pin_purchase_cost :
  total_cost 10 20 15 = 170 := by
  sorry

end NUMINAMATH_CALUDE_pin_purchase_cost_l2292_229223


namespace NUMINAMATH_CALUDE_murtha_pebble_collection_l2292_229214

/-- The sum of an arithmetic sequence with n terms, starting from a, with a common difference of d -/
def arithmetic_sum (n : ℕ) (a : ℕ) (d : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

/-- The number of days Murtha collects pebbles -/
def days : ℕ := 15

/-- The number of pebbles Murtha collects on the first day -/
def initial_pebbles : ℕ := 2

/-- The daily increase in pebble collection -/
def daily_increase : ℕ := 1

theorem murtha_pebble_collection :
  arithmetic_sum days initial_pebbles daily_increase = 135 := by
  sorry

end NUMINAMATH_CALUDE_murtha_pebble_collection_l2292_229214


namespace NUMINAMATH_CALUDE_smallest_number_with_five_primes_including_even_l2292_229208

def is_prime (n : ℕ) : Prop := sorry

def has_five_different_prime_factors (n : ℕ) : Prop := sorry

def has_even_prime_factor (n : ℕ) : Prop := sorry

theorem smallest_number_with_five_primes_including_even :
  ∀ n : ℕ, 
    has_five_different_prime_factors n ∧ 
    has_even_prime_factor n → 
    n ≥ 2310 :=
sorry

end NUMINAMATH_CALUDE_smallest_number_with_five_primes_including_even_l2292_229208


namespace NUMINAMATH_CALUDE_complex_modulus_l2292_229217

/-- If z is a complex number satisfying (2+i)z = 5, then |z| = √5 -/
theorem complex_modulus (z : ℂ) (h : (2 + Complex.I) * z = 5) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l2292_229217


namespace NUMINAMATH_CALUDE_existence_of_constant_l2292_229225

theorem existence_of_constant : ∃ c : ℝ, c > 0 ∧
  ∀ a b n : ℕ, a > 0 → b > 0 → n > 0 →
  (∀ i j : ℕ, i ≤ n → j ≤ n → Nat.gcd (a + i) (b + j) > 1) →
  min a b > (c * n : ℝ) ^ (n / 2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_constant_l2292_229225


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_tangent_ratio_l2292_229285

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a circle -/
structure Circle :=
  (center : Point) (radius : ℝ)

/-- Represents an isosceles trapezoid -/
structure IsoscelesTrapezoid :=
  (A B C D : Point)

/-- Checks if a point lies on a given line segment -/
def pointOnSegment (P Q R : Point) : Prop :=
  sorry

/-- Checks if a line is tangent to a circle -/
def isTangent (P Q : Point) (circle : Circle) : Prop :=
  sorry

/-- Checks if a trapezoid is circumscribed around a circle -/
def isCircumscribed (trapezoid : IsoscelesTrapezoid) (circle : Circle) : Prop :=
  sorry

/-- Calculates the distance between two points -/
def distance (P Q : Point) : ℝ :=
  sorry

theorem isosceles_trapezoid_tangent_ratio 
  (trapezoid : IsoscelesTrapezoid) 
  (circle : Circle) 
  (P Q R S : Point) :
  isCircumscribed trapezoid circle →
  isTangent P S circle →
  pointOnSegment P Q R →
  pointOnSegment P S R →
  distance P Q / distance Q R = distance R S / distance S R :=
sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_tangent_ratio_l2292_229285


namespace NUMINAMATH_CALUDE_special_elements_in_100_l2292_229227

/-- Represents the number of elements in the nth group -/
def group_size (n : ℕ) : ℕ := n + 1

/-- Calculates the total number of elements up to and including the nth group -/
def total_elements (n : ℕ) : ℕ := n * (n + 3) / 2

/-- Represents the number of special elements in the first n groups -/
def special_elements (n : ℕ) : ℕ := n

theorem special_elements_in_100 :
  ∃ n : ℕ, total_elements n ≤ 100 ∧ total_elements (n + 1) > 100 ∧ special_elements n = 12 :=
sorry

end NUMINAMATH_CALUDE_special_elements_in_100_l2292_229227


namespace NUMINAMATH_CALUDE_constant_term_of_expansion_l2292_229268

/-- The constant term in the expansion of (x^2 - 2/x)^6 -/
def constantTerm : ℤ := 240

/-- The binomial expansion of (x^2 - 2/x)^6 -/
def expansion (x : ℚ) : ℚ := (x^2 - 2/x)^6

theorem constant_term_of_expansion :
  ∃ (f : ℚ → ℚ), (∀ x ≠ 0, f x = expansion x) ∧ 
  (∃ c : ℚ, ∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x| ∧ |x| < δ → |f x - c| < ε) ∧
  (c = constantTerm) := by
  sorry

end NUMINAMATH_CALUDE_constant_term_of_expansion_l2292_229268


namespace NUMINAMATH_CALUDE_village_population_equality_l2292_229271

/-- The rate at which Village X's population is decreasing per year -/
def decrease_rate : ℕ := sorry

/-- The initial population of Village X -/
def village_x_initial : ℕ := 74000

/-- The initial population of Village Y -/
def village_y_initial : ℕ := 42000

/-- The rate at which Village Y's population is increasing per year -/
def village_y_increase : ℕ := 800

/-- The number of years after which the populations will be equal -/
def years_until_equal : ℕ := 16

theorem village_population_equality :
  village_x_initial - years_until_equal * decrease_rate =
  village_y_initial + years_until_equal * village_y_increase →
  decrease_rate = 1200 := by
sorry

end NUMINAMATH_CALUDE_village_population_equality_l2292_229271


namespace NUMINAMATH_CALUDE_ratio_product_l2292_229292

theorem ratio_product (a b c d e f : ℝ) 
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : e / f = 1 / 2) :
  a * b * c / (d * e * f) = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ratio_product_l2292_229292


namespace NUMINAMATH_CALUDE_square_area_ratio_l2292_229265

theorem square_area_ratio (s₁ s₂ : ℝ) (h : s₁ = 2 * s₂) : s₁^2 / s₂^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l2292_229265


namespace NUMINAMATH_CALUDE_unique_five_digit_pair_l2292_229235

/-- Check if a number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- Check if each digit of b is exactly 1 greater than the corresponding digit of a -/
def digitsOneGreater (a b : ℕ) : Prop :=
  ∀ i : ℕ, i < 5 → (b / 10^i) % 10 = (a / 10^i) % 10 + 1

/-- The main theorem -/
theorem unique_five_digit_pair : 
  ∀ a b : ℕ, 
    10000 ≤ a ∧ a < 100000 ∧
    10000 ≤ b ∧ b < 100000 ∧
    isPerfectSquare a ∧
    isPerfectSquare b ∧
    b - a = 11111 ∧
    digitsOneGreater a b →
    a = 13225 ∧ b = 24336 :=
sorry

end NUMINAMATH_CALUDE_unique_five_digit_pair_l2292_229235


namespace NUMINAMATH_CALUDE_median_unchanged_l2292_229212

-- Define a type for scores
def Score := ℝ

-- Define a function to calculate the median of a list of scores
def median (scores : List Score) : Score :=
  sorry

-- Define a function to remove the highest and lowest scores
def removeExtremes (scores : List Score) : List Score :=
  sorry

-- Theorem statement
theorem median_unchanged (scores : List Score) (h : scores.length = 9) :
  median scores = median (removeExtremes scores) :=
sorry

end NUMINAMATH_CALUDE_median_unchanged_l2292_229212


namespace NUMINAMATH_CALUDE_isosceles_triangle_largest_angle_l2292_229275

theorem isosceles_triangle_largest_angle (α β γ : ℝ) :
  -- The triangle is isosceles with two angles equal to 50°
  α = β ∧ α = 50 ∧
  -- The sum of angles in a triangle is 180°
  α + β + γ = 180 →
  -- The largest angle is 80°
  max α (max β γ) = 80 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_largest_angle_l2292_229275


namespace NUMINAMATH_CALUDE_coefficient_of_x_l2292_229204

theorem coefficient_of_x (x : ℝ) : 
  let expansion := (1 + x) * (x - 2/x)^3
  ∃ (a b c d e : ℝ), expansion = a*x^4 + b*x^3 + c*x^2 + (-6)*x + e
  := by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_l2292_229204


namespace NUMINAMATH_CALUDE_wheel_on_semicircle_diameter_l2292_229233

theorem wheel_on_semicircle_diameter (r_wheel r_semicircle : ℝ) 
  (h_wheel : r_wheel = 8)
  (h_semicircle : r_semicircle = 25) :
  let untouched_length := 2 * (r_semicircle - (r_semicircle^2 - r_wheel^2).sqrt)
  untouched_length = 20 := by
sorry

end NUMINAMATH_CALUDE_wheel_on_semicircle_diameter_l2292_229233


namespace NUMINAMATH_CALUDE_B_power_difference_l2292_229276

def B : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 0, 2]

theorem B_power_difference :
  B^10 - 3 * B^9 = !![0, 4; 0, -1] := by sorry

end NUMINAMATH_CALUDE_B_power_difference_l2292_229276


namespace NUMINAMATH_CALUDE_zoo_bird_difference_l2292_229287

/-- Proves that in a zoo with 450 birds and where the number of birds is 5 times
    the number of all other animals, there are 360 more birds than non-bird animals. -/
theorem zoo_bird_difference (total_birds : ℕ) (bird_ratio : ℕ) 
    (h1 : total_birds = 450)
    (h2 : bird_ratio = 5)
    (h3 : total_birds = bird_ratio * (total_birds / bird_ratio)) :
  total_birds - (total_birds / bird_ratio) = 360 := by
  sorry

#eval 450 - (450 / 5)  -- This should evaluate to 360

end NUMINAMATH_CALUDE_zoo_bird_difference_l2292_229287


namespace NUMINAMATH_CALUDE_inequality_proof_l2292_229290

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a * b / (a + b)^2 + b * c / (b + c)^2 + c * a / (c + a)^2) + 
  3 * (a^2 + b^2 + c^2) / (a + b + c)^2 ≥ 7/4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2292_229290


namespace NUMINAMATH_CALUDE_arithmetic_problem_l2292_229213

theorem arithmetic_problem : 4 * (8 - 3)^2 - 2 * 7 = 86 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_problem_l2292_229213


namespace NUMINAMATH_CALUDE_function_characterization_l2292_229284

-- Define the property that f should satisfy
def SatisfiesProperty (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * f x + f y) = f x ^ 2 + y

-- State the theorem
theorem function_characterization :
  ∀ f : ℝ → ℝ, SatisfiesProperty f →
  (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = -x) :=
sorry

end NUMINAMATH_CALUDE_function_characterization_l2292_229284


namespace NUMINAMATH_CALUDE_abc_perfect_cube_l2292_229286

theorem abc_perfect_cube (a b c : ℤ) (h : (a : ℚ) / b + (b : ℚ) / c + (c : ℚ) / a = 3) :
  ∃ (n : ℤ), a * b * c = n^3 := by
sorry

end NUMINAMATH_CALUDE_abc_perfect_cube_l2292_229286


namespace NUMINAMATH_CALUDE_average_book_cost_l2292_229297

def initial_amount : ℕ := 236
def books_bought : ℕ := 6
def remaining_amount : ℕ := 14

theorem average_book_cost :
  (initial_amount - remaining_amount) / books_bought = 37 := by
  sorry

end NUMINAMATH_CALUDE_average_book_cost_l2292_229297


namespace NUMINAMATH_CALUDE_min_value_on_interval_l2292_229209

def f (x a : ℝ) : ℝ := 3 * x^4 - 8 * x^3 - 18 * x^2 + a

theorem min_value_on_interval (a : ℝ) :
  (∃ x ∈ Set.Icc (-1) 1, f x a = 6) ∧ 
  (∀ x ∈ Set.Icc (-1) 1, f x a ≤ 6) →
  (∃ x ∈ Set.Icc (-1) 1, f x a = -17) ∧ 
  (∀ x ∈ Set.Icc (-1) 1, f x a ≥ -17) := by
  sorry

end NUMINAMATH_CALUDE_min_value_on_interval_l2292_229209


namespace NUMINAMATH_CALUDE_water_pouring_game_score_l2292_229226

/-- Represents the players in the game -/
inductive Player
| Xiaoming
| Xiaolin

/-- Defines the scoring rules for the water pouring game -/
def score (overflowPlayer : Option Player) : Nat :=
  match overflowPlayer with
  | some Player.Xiaoming => 10
  | some Player.Xiaolin => 9
  | none => 3

/-- Represents a round in the game -/
structure Round where
  xiaomingPour : Nat
  xiaolinPour : Nat
  overflowPlayer : Option Player

/-- The three rounds of the game -/
def round1 : Round := ⟨5, 5, some Player.Xiaolin⟩
def round2 : Round := ⟨2, 7, none⟩
def round3 : Round := ⟨13, 0, some Player.Xiaoming⟩

/-- Calculates the total score for the given rounds -/
def totalScore (rounds : List Round) : Nat :=
  rounds.foldl (fun acc r => acc + score r.overflowPlayer) 0

/-- The main theorem to prove -/
theorem water_pouring_game_score :
  totalScore [round1, round2, round3] = 22 := by
  sorry

end NUMINAMATH_CALUDE_water_pouring_game_score_l2292_229226


namespace NUMINAMATH_CALUDE_rock_paper_scissors_probabilities_l2292_229249

/-- Represents the outcome of a single throw in Rock, Paper, Scissors -/
inductive Throw
  | Rock
  | Paper
  | Scissors

/-- Represents the result of a game between two players -/
inductive GameResult
  | Win
  | Lose
  | Tie

/-- Represents a player in the game -/
inductive Player
  | A
  | B
  | C

/-- The probability of winning, losing, or tying in a single throw -/
def singleThrowProb : ℚ := 1 / 3

/-- The probability of A winning against B with no more than two throws -/
def probAWinsTwoThrows : ℚ := 4 / 9

/-- The probability of C treating after two throws -/
def probCTreatsTwoThrows : ℚ := 2 / 9

/-- The probability of C treating after two throws on exactly two out of three independent days -/
def probCTreatsTwoDays : ℚ := 28 / 243

theorem rock_paper_scissors_probabilities :
  (probAWinsTwoThrows = 4 / 9) ∧
  (probCTreatsTwoThrows = 2 / 9) ∧
  (probCTreatsTwoDays = 28 / 243) := by
  sorry

end NUMINAMATH_CALUDE_rock_paper_scissors_probabilities_l2292_229249


namespace NUMINAMATH_CALUDE_function_value_sum_l2292_229289

/-- Given a function f(x) = a*sin(x) + b*x^3 + 4, where a and b are real numbers,
    prove that f(2016) + f(-2016) + f'(2017) - f'(-2017) = 8 -/
theorem function_value_sum (a b : ℝ) : 
  let f (x : ℝ) := a * Real.sin x + b * x^3 + 4
  let f' (x : ℝ) := a * Real.cos x + 3 * b * x^2
  f 2016 + f (-2016) + f' 2017 - f' (-2017) = 8 := by
  sorry

end NUMINAMATH_CALUDE_function_value_sum_l2292_229289


namespace NUMINAMATH_CALUDE_quadratic_solution_average_l2292_229274

/-- Given a quadratic equation 2x^2 - 6x + c = 0 with two real solutions and discriminant 12,
    prove that the average of the solutions is 1.5 -/
theorem quadratic_solution_average (c : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 2 * x₁^2 - 6 * x₁ + c = 0 ∧ 2 * x₂^2 - 6 * x₂ + c = 0) →
  ((-6)^2 - 4 * 2 * c = 12) →
  (∃ x₁ x₂ : ℝ, 2 * x₁^2 - 6 * x₁ + c = 0 ∧ 2 * x₂^2 - 6 * x₂ + c = 0 ∧ (x₁ + x₂) / 2 = 1.5) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_solution_average_l2292_229274


namespace NUMINAMATH_CALUDE_lego_problem_solution_l2292_229203

def lego_problem (initial_pieces : ℕ) : ℕ :=
  let castle_pieces := initial_pieces / 4
  let after_castle := initial_pieces - castle_pieces
  let spaceship_pieces := (after_castle * 2) / 5
  let after_spaceship := after_castle - spaceship_pieces
  let lost_after_building := (after_spaceship * 15) / 100
  let after_loss := after_spaceship - lost_after_building
  let town_pieces := after_loss / 2
  let after_town := after_loss - town_pieces
  let final_loss := (after_town * 10) / 100
  after_town - final_loss

theorem lego_problem_solution :
  lego_problem 500 = 85 := by sorry

end NUMINAMATH_CALUDE_lego_problem_solution_l2292_229203


namespace NUMINAMATH_CALUDE_exists_large_remainder_sum_l2292_229250

/-- Given positive integers N and a, generates a sequence of remainders by repeatedly dividing N by the last remainder, starting with a, until 0 is reached. -/
def remainderSequence (N a : ℕ+) : List ℕ :=
  sorry

/-- The theorem states that there exist positive integers N and a such that the sum of the remainder sequence is greater than 100N. -/
theorem exists_large_remainder_sum : ∃ N a : ℕ+, 
  (remainderSequence N a).sum > 100 * N.val := by
  sorry

end NUMINAMATH_CALUDE_exists_large_remainder_sum_l2292_229250


namespace NUMINAMATH_CALUDE_malt_shop_problem_l2292_229264

/-- Represents the number of ounces of chocolate syrup used per shake -/
def syrup_per_shake : ℕ := 4

/-- Represents the number of ounces of chocolate syrup used per cone -/
def syrup_per_cone : ℕ := 6

/-- Represents the number of shakes sold -/
def shakes_sold : ℕ := 2

/-- Represents the total number of ounces of chocolate syrup used -/
def total_syrup_used : ℕ := 14

/-- Represents the number of cones sold -/
def cones_sold : ℕ := 1

theorem malt_shop_problem :
  syrup_per_shake * shakes_sold + syrup_per_cone * cones_sold = total_syrup_used :=
by sorry

end NUMINAMATH_CALUDE_malt_shop_problem_l2292_229264


namespace NUMINAMATH_CALUDE_father_age_problem_l2292_229295

theorem father_age_problem (father_age son_age : ℕ) : 
  (father_age = 4 * son_age + 4) →
  (father_age + 4 = 2 * (son_age + 4) + 20) →
  father_age = 44 := by
  sorry

end NUMINAMATH_CALUDE_father_age_problem_l2292_229295


namespace NUMINAMATH_CALUDE_vasya_no_purchase_days_vasya_no_purchase_days_proof_l2292_229283

theorem vasya_no_purchase_days : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun x y z w =>
    x + y + z + w = 15 ∧
    9 * x + 4 * z = 30 ∧
    2 * y + z = 9 →
    w = 7

-- The proof is omitted
theorem vasya_no_purchase_days_proof : vasya_no_purchase_days 2 3 3 7 := by
  sorry

end NUMINAMATH_CALUDE_vasya_no_purchase_days_vasya_no_purchase_days_proof_l2292_229283


namespace NUMINAMATH_CALUDE_new_students_count_l2292_229232

/-- Represents the number of new students who joined the class -/
def new_students : ℕ := sorry

/-- The original average age of the class -/
def original_avg_age : ℕ := 40

/-- The average age of new students -/
def new_students_avg_age : ℕ := 32

/-- The decrease in average age after new students join -/
def avg_age_decrease : ℕ := 4

/-- The original number of students in the class -/
def original_class_size : ℕ := 18

theorem new_students_count :
  (original_class_size * original_avg_age + new_students * new_students_avg_age) / (original_class_size + new_students) = original_avg_age - avg_age_decrease ∧
  new_students = 18 :=
sorry

end NUMINAMATH_CALUDE_new_students_count_l2292_229232


namespace NUMINAMATH_CALUDE_sin_product_ninth_roots_l2292_229288

theorem sin_product_ninth_roots : 
  Real.sin (π / 9) * Real.sin (2 * π / 9) * Real.sin (4 * π / 9) = -1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sin_product_ninth_roots_l2292_229288


namespace NUMINAMATH_CALUDE_log_inequality_implication_l2292_229228

theorem log_inequality_implication (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (Real.log 3 / Real.log a < Real.log 3 / Real.log b) ∧
  (Real.log 3 / Real.log b < Real.log 3 / Real.log c) →
  ¬(a < b ∧ b < c) :=
by sorry

end NUMINAMATH_CALUDE_log_inequality_implication_l2292_229228


namespace NUMINAMATH_CALUDE_boa_constrictor_length_is_70_l2292_229279

/-- The length of the garden snake in inches -/
def garden_snake_length : ℕ := 10

/-- The factor by which the boa constrictor is longer than the garden snake -/
def boa_length_factor : ℕ := 7

/-- The length of the boa constrictor in inches -/
def boa_constrictor_length : ℕ := garden_snake_length * boa_length_factor

theorem boa_constrictor_length_is_70 : boa_constrictor_length = 70 := by
  sorry

end NUMINAMATH_CALUDE_boa_constrictor_length_is_70_l2292_229279


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2292_229211

-- Problem 1
theorem problem_1 (x y : ℝ) : (2*x - 3*y)^2 - (y + 3*x)*(3*x - y) = -5*x^2 - 12*x*y + 10*y^2 := by
  sorry

-- Problem 2
theorem problem_2 : (2+1)*(2^2+1)*(2^4+1)*(2^8+1) - 2^16 = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2292_229211


namespace NUMINAMATH_CALUDE_quadratic_roots_inequality_l2292_229237

/-- Given quadratic polynomials f(x) = x² + bx + c and g(x) = x² + px + q with roots m₁, m₂ and k₁, k₂ respectively,
    prove that f(k₁) + f(k₂) + g(m₁) + g(m₂) ≥ 0. -/
theorem quadratic_roots_inequality (b c p q m₁ m₂ k₁ k₂ : ℝ) :
  let f := fun x => x^2 + b*x + c
  let g := fun x => x^2 + p*x + q
  (f m₁ = 0) ∧ (f m₂ = 0) ∧ (g k₁ = 0) ∧ (g k₂ = 0) →
  f k₁ + f k₂ + g m₁ + g m₂ ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_inequality_l2292_229237


namespace NUMINAMATH_CALUDE_girls_at_game_l2292_229216

theorem girls_at_game (boys girls : ℕ) : 
  (boys : ℚ) / girls = 8 / 5 → 
  boys = girls + 18 → 
  girls = 30 := by
sorry

end NUMINAMATH_CALUDE_girls_at_game_l2292_229216


namespace NUMINAMATH_CALUDE_price_change_calculation_l2292_229245

theorem price_change_calculation :
  let original_price := 100
  let price_after_day1 := original_price * (1 - 0.12)
  let price_after_day2 := price_after_day1 * (1 - 0.10)
  let price_after_day3 := price_after_day2 * (1 - 0.08)
  let final_price := price_after_day3 * (1 + 0.05)
  (final_price / original_price) * 100 = 76.5072 := by
sorry

end NUMINAMATH_CALUDE_price_change_calculation_l2292_229245


namespace NUMINAMATH_CALUDE_min_chocolate_cookies_l2292_229261

theorem min_chocolate_cookies (chocolate_batch_size peanut_batch_size total_cookies : ℕ)
  (chocolate_ratio peanut_ratio : ℕ) :
  chocolate_batch_size = 5 →
  peanut_batch_size = 6 →
  chocolate_ratio = 3 →
  peanut_ratio = 2 →
  total_cookies = 94 →
  ∃ (chocolate_batches peanut_batches : ℕ),
    chocolate_batches * chocolate_batch_size + peanut_batches * peanut_batch_size = total_cookies ∧
    chocolate_batches * chocolate_batch_size * peanut_ratio = peanut_batches * peanut_batch_size * chocolate_ratio ∧
    chocolate_batches * chocolate_batch_size ≥ 60 ∧
    ∀ (c p : ℕ), c * chocolate_batch_size + p * peanut_batch_size = total_cookies →
      c * chocolate_batch_size * peanut_ratio = p * peanut_batch_size * chocolate_ratio →
      c * chocolate_batch_size ≥ chocolate_batches * chocolate_batch_size :=
by sorry

end NUMINAMATH_CALUDE_min_chocolate_cookies_l2292_229261


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l2292_229296

/-- Given plane vectors a and b, if a is parallel to 2b - a, then m = 9/2 -/
theorem parallel_vectors_m_value (a b : ℝ × ℝ) (m : ℝ) 
    (h1 : a = (4, 3))
    (h2 : b = (6, m))
    (h3 : ∃ (k : ℝ), a = k • (2 • b - a)) :
  m = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l2292_229296


namespace NUMINAMATH_CALUDE_remainder_problem_l2292_229256

theorem remainder_problem :
  (85^70 + 19^32)^16 ≡ 16 [ZMOD 21] := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2292_229256


namespace NUMINAMATH_CALUDE_expand_product_l2292_229229

theorem expand_product (x : ℝ) : (x + 3) * (x + 9) = x^2 + 12*x + 27 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l2292_229229


namespace NUMINAMATH_CALUDE_abs_reciprocal_of_neg_three_halves_l2292_229272

theorem abs_reciprocal_of_neg_three_halves :
  |(((-1 : ℚ) - (1 : ℚ) / (2 : ℚ))⁻¹)| = (2 : ℚ) / (3 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_abs_reciprocal_of_neg_three_halves_l2292_229272


namespace NUMINAMATH_CALUDE_product_nonzero_l2292_229270

theorem product_nonzero (n : ℤ) : n ≠ 5 → n ≠ 17 → n ≠ 257 → (n - 5) * (n - 17) * (n - 257) ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_product_nonzero_l2292_229270


namespace NUMINAMATH_CALUDE_problem_solution_l2292_229201

-- Define proposition p
def p : Prop := ∃ x₀ : ℝ, Real.log x₀ ≥ x₀ - 1

-- Define proposition q
def q : Prop := ∀ θ : ℝ, Real.sin θ + Real.cos θ < 1

-- Theorem to prove
theorem problem_solution :
  p ∧ ¬q := by sorry

end NUMINAMATH_CALUDE_problem_solution_l2292_229201


namespace NUMINAMATH_CALUDE_rachel_essay_pages_l2292_229206

/-- Rachel's essay writing problem -/
theorem rachel_essay_pages :
  let pages_per_30_min : ℕ := 1
  let research_time : ℕ := 45
  let editing_time : ℕ := 75
  let total_time : ℕ := 300
  let writing_time : ℕ := total_time - (research_time + editing_time)
  let pages_written : ℕ := writing_time / 30
  pages_written = 6 := by sorry

end NUMINAMATH_CALUDE_rachel_essay_pages_l2292_229206


namespace NUMINAMATH_CALUDE_original_number_proof_l2292_229210

theorem original_number_proof : 
  ∃ x : ℝ, (204 / x = 16) ∧ (x = 12.75) := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l2292_229210


namespace NUMINAMATH_CALUDE_conic_section_focal_distance_l2292_229299

theorem conic_section_focal_distance (a : ℝ) (h1 : a ≠ 0) :
  (∀ x y : ℝ, x^2 + a * y^2 + a^2 = 0 → 
    ∃ c : ℝ, c = 2 ∧ c^2 = a^2 - a) →
  a = (1 - Real.sqrt 17) / 2 := by
sorry

end NUMINAMATH_CALUDE_conic_section_focal_distance_l2292_229299


namespace NUMINAMATH_CALUDE_train_crossing_time_l2292_229281

/-- Proves that a train of length 500 m, traveling at 180 km/h, takes 10 seconds to cross an electric pole. -/
theorem train_crossing_time :
  let train_length : ℝ := 500  -- Length of the train in meters
  let train_speed_kmh : ℝ := 180  -- Speed of the train in km/h
  let train_speed_ms : ℝ := train_speed_kmh * 1000 / 3600  -- Speed in m/s
  let crossing_time : ℝ := train_length / train_speed_ms  -- Time to cross the pole
  crossing_time = 10 := by sorry

end NUMINAMATH_CALUDE_train_crossing_time_l2292_229281


namespace NUMINAMATH_CALUDE_ratio_invariance_l2292_229260

theorem ratio_invariance (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (x * y) / (y * x) = 1 → ∃ (r : ℝ), r ≠ 0 ∧ x / y = r :=
by sorry

end NUMINAMATH_CALUDE_ratio_invariance_l2292_229260


namespace NUMINAMATH_CALUDE_exists_return_steps_power_of_two_case_power_of_two_plus_one_case_l2292_229291

/-- Represents the state of a lamp (ON or OFF) -/
inductive LampState
| ON
| OFF

/-- Represents the configuration of n lamps -/
def LampConfig (n : ℕ) := Fin n → LampState

/-- Performs a single step of the lamp changing process -/
def step (n : ℕ) (config : LampConfig n) : LampConfig n :=
  sorry

/-- Checks if all lamps in the configuration are ON -/
def allOn (n : ℕ) (config : LampConfig n) : Prop :=
  sorry

/-- The initial configuration with all lamps ON -/
def initialConfig (n : ℕ) : LampConfig n :=
  sorry

/-- Theorem stating the existence of M(n) for any n > 1 -/
theorem exists_return_steps (n : ℕ) (h : n > 1) :
  ∃ M : ℕ, M > 0 ∧ allOn n ((step n)^[M] (initialConfig n)) :=
  sorry

/-- Theorem for the case when n is a power of 2 -/
theorem power_of_two_case (k : ℕ) :
  let n := 2^k
  allOn n ((step n)^[n^2 - 1] (initialConfig n)) :=
  sorry

/-- Theorem for the case when n is one more than a power of 2 -/
theorem power_of_two_plus_one_case (k : ℕ) :
  let n := 2^k + 1
  allOn n ((step n)^[n^2 - n + 1] (initialConfig n)) :=
  sorry

end NUMINAMATH_CALUDE_exists_return_steps_power_of_two_case_power_of_two_plus_one_case_l2292_229291


namespace NUMINAMATH_CALUDE_consistent_walnuts_dont_determine_oranges_l2292_229293

/-- Represents the state of trees in a park -/
structure ParkTrees where
  initial_walnuts : ℕ
  cut_walnuts : ℕ
  final_walnuts : ℕ

/-- Checks if the walnut tree information is consistent -/
def consistent_walnuts (park : ParkTrees) : Prop :=
  park.initial_walnuts - park.cut_walnuts = park.final_walnuts

/-- States that the number of orange trees cannot be determined -/
def orange_trees_undetermined (park : ParkTrees) : Prop :=
  ∀ n : ℕ, ∃ park' : ParkTrees, park'.initial_walnuts = park.initial_walnuts ∧
                                park'.cut_walnuts = park.cut_walnuts ∧
                                park'.final_walnuts = park.final_walnuts ∧
                                n ≠ 0  -- Assuming there's at least one orange tree

/-- Theorem stating that consistent walnut information doesn't determine orange tree count -/
theorem consistent_walnuts_dont_determine_oranges (park : ParkTrees) :
  consistent_walnuts park → orange_trees_undetermined park :=
by
  sorry

#check consistent_walnuts_dont_determine_oranges

end NUMINAMATH_CALUDE_consistent_walnuts_dont_determine_oranges_l2292_229293
