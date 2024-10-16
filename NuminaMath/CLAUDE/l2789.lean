import Mathlib

namespace NUMINAMATH_CALUDE_solution_set_characterization_l2789_278975

/-- A function f: ℝ → ℝ is even -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- The set of real numbers x where xf(x) > 0 -/
def SolutionSet (f : ℝ → ℝ) : Set ℝ := {x | x * f x > 0}

/-- The open interval (-∞, -1) ∪ (1, +∞) -/
def TargetSet : Set ℝ := {x | x < -1 ∨ x > 1}

theorem solution_set_characterization (f : ℝ → ℝ) 
  (h_even : IsEven f)
  (h_positive : ∀ x > 0, f x + x * (deriv f x) > 0)
  (h_zero : f 1 = 0) :
  SolutionSet f = TargetSet := by sorry

end NUMINAMATH_CALUDE_solution_set_characterization_l2789_278975


namespace NUMINAMATH_CALUDE_x_intercept_of_l_equation_of_l_l2789_278910

-- Define the lines l₁, l₂, and l₃
def l₁ (x y : ℝ) : Prop := x - y + 1 = 0
def l₂ (x y : ℝ) : Prop := 2*x + y - 4 = 0
def l₃ (x y : ℝ) : Prop := 4*x + 5*y - 12 = 0

-- Define the intersection point of l₁ and l₂
def intersection (x y : ℝ) : Prop := l₁ x y ∧ l₂ x y

-- Define line l
def l (x y : ℝ) : Prop := ∃ (ix iy : ℝ), intersection ix iy ∧ (y - iy) = ((x - ix) * (3 - iy)) / (3 - ix)

-- Theorem for part 1
theorem x_intercept_of_l : 
  (∃ (x y : ℝ), intersection x y) → 
  l 3 3 → 
  (∃ (x : ℝ), l x 0 ∧ x = -3) :=
sorry

-- Theorem for part 2
theorem equation_of_l :
  (∃ (x y : ℝ), intersection x y) →
  (∀ (x y : ℝ), l x y ↔ l₃ (x + a) (y + b)) →
  (∀ (x y : ℝ), l x y ↔ 4*x + 5*y - 14 = 0) :=
sorry

end NUMINAMATH_CALUDE_x_intercept_of_l_equation_of_l_l2789_278910


namespace NUMINAMATH_CALUDE_bingo_prize_calculation_l2789_278935

/-- The total prize money for the bingo night. -/
def total_prize_money : ℝ := 2400

/-- The amount received by each of the 10 winners after the first winner. -/
def winner_amount : ℝ := 160

theorem bingo_prize_calculation :
  let first_winner_share := total_prize_money / 3
  let remaining_after_first := total_prize_money - first_winner_share
  let each_winner_share := remaining_after_first / 10
  (each_winner_share = winner_amount) ∧ 
  (total_prize_money > 0) ∧
  (winner_amount > 0) :=
by sorry

end NUMINAMATH_CALUDE_bingo_prize_calculation_l2789_278935


namespace NUMINAMATH_CALUDE_tree_space_calculation_l2789_278907

/-- Given a road of length 166 feet where 16 trees are planted with 10 feet between each tree,
    prove that each tree occupies 1 square foot of sidewalk space. -/
theorem tree_space_calculation (road_length : ℝ) (num_trees : ℕ) (space_between : ℝ) : 
  road_length = 166 ∧ num_trees = 16 ∧ space_between = 10 → 
  (road_length - space_between * (num_trees - 1)) / num_trees = 1 :=
by sorry

end NUMINAMATH_CALUDE_tree_space_calculation_l2789_278907


namespace NUMINAMATH_CALUDE_apple_cost_proof_l2789_278957

theorem apple_cost_proof (original_price : ℝ) (price_increase : ℝ) (family_size : ℕ) (pounds_per_person : ℝ) : 
  original_price = 1.6 → 
  price_increase = 0.25 → 
  family_size = 4 → 
  pounds_per_person = 2 → 
  (original_price + original_price * price_increase) * (family_size : ℝ) * pounds_per_person = 16 := by
sorry

end NUMINAMATH_CALUDE_apple_cost_proof_l2789_278957


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l2789_278931

def a : Fin 2 → ℝ := ![(-1), 2]
def b (m : ℝ) : Fin 2 → ℝ := ![m, 1]

theorem perpendicular_vectors (m : ℝ) : 
  (∀ i : Fin 2, (a + b m) i * a i = 0) → m = 7 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l2789_278931


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l2789_278981

theorem quadratic_equation_solutions :
  let f : ℝ → ℝ := λ x ↦ x^2 - 25
  ∃ x₁ x₂ : ℝ, x₁ = 5 ∧ x₂ = -5 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l2789_278981


namespace NUMINAMATH_CALUDE_dividend_from_quotient_and_remainder_l2789_278901

theorem dividend_from_quotient_and_remainder :
  ∀ (dividend quotient remainder : ℕ),
    dividend = 23 * quotient + remainder →
    quotient = 38 →
    remainder = 7 →
    dividend = 881 := by
  sorry

end NUMINAMATH_CALUDE_dividend_from_quotient_and_remainder_l2789_278901


namespace NUMINAMATH_CALUDE_equation_d_is_linear_l2789_278953

/-- A linear equation in two variables is of the form ax + by = c, where a, b, and c are constants, and at least one of a or b is non-zero. --/
def IsLinearEquationInTwoVariables (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b c : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ ∀ x y, f x y ↔ a * x + b * y = c

/-- The equation x = y + 1 --/
def EquationD (x y : ℝ) : Prop := x = y + 1

theorem equation_d_is_linear : IsLinearEquationInTwoVariables EquationD := by
  sorry

#check equation_d_is_linear

end NUMINAMATH_CALUDE_equation_d_is_linear_l2789_278953


namespace NUMINAMATH_CALUDE_fraction_simplification_l2789_278920

theorem fraction_simplification (x : ℝ) : (2*x + 3)/4 + (5 - 4*x)/3 = (-10*x + 29)/12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2789_278920


namespace NUMINAMATH_CALUDE_max_teams_in_tournament_l2789_278958

/-- The number of players in each team -/
def players_per_team : ℕ := 3

/-- The maximum number of games that can be played -/
def max_games : ℕ := 200

/-- The number of games played between two teams -/
def games_between_teams : ℕ := players_per_team * players_per_team

/-- Calculates the total number of games for a given number of teams -/
def total_games (n : ℕ) : ℕ := games_between_teams * (n * (n - 1) / 2)

/-- The theorem stating the maximum number of teams that can participate -/
theorem max_teams_in_tournament : 
  ∃ (n : ℕ), n = 7 ∧ 
  total_games n ≤ max_games ∧ 
  ∀ (m : ℕ), m > n → total_games m > max_games :=
sorry

end NUMINAMATH_CALUDE_max_teams_in_tournament_l2789_278958


namespace NUMINAMATH_CALUDE_expected_value_binomial_li_expected_traffic_jams_l2789_278956

/-- The number of intersections Mr. Li passes through -/
def n : ℕ := 6

/-- The probability of a traffic jam at each intersection -/
def p : ℚ := 1/6

/-- The expected value of a binomial distribution is n * p -/
theorem expected_value_binomial (n : ℕ) (p : ℚ) :
  n * p = 1 → n = 6 ∧ p = 1/6 := by sorry

/-- The expected number of traffic jams Mr. Li encounters is 1 -/
theorem li_expected_traffic_jams :
  n * p = 1 := by sorry

end NUMINAMATH_CALUDE_expected_value_binomial_li_expected_traffic_jams_l2789_278956


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2789_278960

-- Problem 1
theorem problem_1 (x : ℝ) : x^2 * x^3 - x^5 = 0 := by sorry

-- Problem 2
theorem problem_2 (a : ℝ) : (a + 1)^2 + 2*a*(a - 1) = 3*a^2 + 1 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2789_278960


namespace NUMINAMATH_CALUDE_sin_cos_roots_l2789_278952

theorem sin_cos_roots (θ : Real) (a : Real) 
  (h1 : x^2 - 2 * Real.sqrt 2 * a * x + a = 0 ↔ x = Real.sin θ ∨ x = Real.cos θ)
  (h2 : -π/2 < θ ∧ θ < 0) : 
  a = -1/4 ∧ Real.sin θ - Real.cos θ = -Real.sqrt 6 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_roots_l2789_278952


namespace NUMINAMATH_CALUDE_wendy_bouquets_l2789_278972

/-- Represents the number of flowers of each type -/
structure FlowerCount where
  roses : ℕ
  lilies : ℕ
  daisies : ℕ

/-- Calculates the number of complete bouquets that can be made -/
def max_bouquets (initial : FlowerCount) (wilted : FlowerCount) (bouquet : FlowerCount) : ℕ :=
  let remaining : FlowerCount := ⟨
    initial.roses - wilted.roses,
    initial.lilies - wilted.lilies,
    initial.daisies - wilted.daisies
  ⟩
  min (remaining.roses / bouquet.roses)
      (min (remaining.lilies / bouquet.lilies)
           (remaining.daisies / bouquet.daisies))

/-- The main theorem stating that the maximum number of complete bouquets is 2 -/
theorem wendy_bouquets :
  let initial : FlowerCount := ⟨20, 15, 10⟩
  let wilted : FlowerCount := ⟨12, 8, 5⟩
  let bouquet : FlowerCount := ⟨3, 2, 1⟩
  max_bouquets initial wilted bouquet = 2 := by
  sorry


end NUMINAMATH_CALUDE_wendy_bouquets_l2789_278972


namespace NUMINAMATH_CALUDE_variance_equality_and_percentile_l2789_278945

-- Define the sequences x_i and y_i
def x : Fin 10 → ℝ := fun i => 2 * (i.val + 1)
def y : Fin 10 → ℝ := fun i => x i - 20

-- Define variance function
def variance (s : Fin 10 → ℝ) : ℝ := sorry

-- Define percentile function
def percentile (p : ℝ) (s : Fin 10 → ℝ) : ℝ := sorry

theorem variance_equality_and_percentile :
  (variance x = variance y) ∧ (percentile 0.3 y = -13) := by sorry

end NUMINAMATH_CALUDE_variance_equality_and_percentile_l2789_278945


namespace NUMINAMATH_CALUDE_floor_product_eq_square_l2789_278990

def floor (x : ℚ) : ℤ := Int.floor x

theorem floor_product_eq_square (x : ℤ) : 
  (floor (x / 2 : ℚ)) * (floor (x / 3 : ℚ)) * (floor (x / 4 : ℚ)) = x^2 ↔ x = 0 ∨ x = 24 :=
by sorry

end NUMINAMATH_CALUDE_floor_product_eq_square_l2789_278990


namespace NUMINAMATH_CALUDE_triangle_properties_l2789_278967

noncomputable section

/-- Triangle ABC with given properties -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The area of a triangle given two sides and the included angle -/
def area (t : Triangle) : ℝ := (1/2) * t.a * t.b * Real.sin t.C

theorem triangle_properties (t : Triangle) 
  (h1 : t.c = 2)
  (h2 : t.C = π/3) :
  (area t = Real.sqrt 3 → t.a = 2 ∧ t.b = 2) ∧
  (Real.sin t.B = 2 * Real.sin t.A → area t = 4 * Real.sqrt 3 / 3) := by
  sorry

end

end NUMINAMATH_CALUDE_triangle_properties_l2789_278967


namespace NUMINAMATH_CALUDE_triangle_inequalities_l2789_278902

theorem triangle_inequalities (a b c : ℝ) 
  (h1 : 0 ≤ a ∧ a ≤ 1) 
  (h2 : 0 ≤ b ∧ b ≤ 1) 
  (h3 : 0 ≤ c ∧ c ≤ 1) 
  (h4 : a + b + c = 2) : 
  (a * b * c + 28 / 27 ≥ a * b + b * c + c * a) ∧ 
  (a * b + b * c + c * a ≥ a * b * c + 1) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequalities_l2789_278902


namespace NUMINAMATH_CALUDE_circle_tangency_and_chord_properties_l2789_278968

-- Define the circle M
def circle_M (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

-- Define point P
def point_P (t : ℝ) : ℝ × ℝ := (-1, t)

-- Define circle N
def circle_N (x y a b : ℝ) : Prop := (x - a)^2 + (y - b)^2 = 1

-- Define the external tangency condition
def external_tangent (a b : ℝ) : Prop := (a - 2)^2 + b^2 = 9

-- Define the chord length condition
def chord_length (t : ℝ) : Prop := ∃ (k : ℝ), 8*k^2 + 6*t*k + t^2 - 1 = 0

-- Define the ST distance condition
def ST_distance (t : ℝ) : Prop := (t^2 + 8) / 16 = 9/16

theorem circle_tangency_and_chord_properties :
  ∀ t : ℝ,
  (∃ a b : ℝ, circle_N (-1) 1 a b ∧ external_tangent a b) →
  (chord_length t ∧ ST_distance t) →
  ((circle_N x y (-1) 0 ∨ circle_N x y (-2/5) (9/5)) ∧ (t = 1 ∨ t = -1)) :=
sorry

end NUMINAMATH_CALUDE_circle_tangency_and_chord_properties_l2789_278968


namespace NUMINAMATH_CALUDE_power_of_power_three_l2789_278928

theorem power_of_power_three : (3^3)^2 = 729 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_three_l2789_278928


namespace NUMINAMATH_CALUDE_quadratic_coefficient_sum_l2789_278974

/-- A quadratic function with roots at -3 and 5, and a minimum value of 36 -/
def quadratic (a b c : ℝ) : ℝ → ℝ := λ x => a * x^2 + b * x + c

theorem quadratic_coefficient_sum (a b c : ℝ) :
  (∀ x, quadratic a b c x ≥ 36) ∧ 
  quadratic a b c (-3) = 0 ∧ 
  quadratic a b c 5 = 0 →
  a + b + c = 36 := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_sum_l2789_278974


namespace NUMINAMATH_CALUDE_harrison_croissant_price_l2789_278955

/-- The price of a regular croissant that Harrison buys -/
def regular_croissant_price : ℝ := 3.50

/-- The price of an almond croissant that Harrison buys -/
def almond_croissant_price : ℝ := 5.50

/-- The total amount Harrison spends on croissants in a year -/
def total_spent : ℝ := 468

/-- The number of weeks in a year -/
def weeks_in_year : ℕ := 52

theorem harrison_croissant_price :
  regular_croissant_price * weeks_in_year + almond_croissant_price * weeks_in_year = total_spent :=
by sorry

end NUMINAMATH_CALUDE_harrison_croissant_price_l2789_278955


namespace NUMINAMATH_CALUDE_parabola_and_line_properties_l2789_278983

-- Define the parabola C: y² = 4x
def C (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus F(1, 0)
def F : ℝ × ℝ := (1, 0)

-- Define point K(-1, 0)
def K : ℝ × ℝ := (-1, 0)

-- Define the line l passing through K and intersecting C at A and B
def l (m : ℝ) (y : ℝ) : ℝ := m*y - 1

-- Define the symmetry of A and D with respect to the x-axis
def symmetric_x_axis (A D : ℝ × ℝ) : Prop :=
  A.1 = D.1 ∧ A.2 = -D.2

-- Define the dot product condition
def dot_product_condition (A B : ℝ × ℝ) : Prop :=
  (A.1 - F.1) * (B.1 - F.1) + (A.2 - F.2) * (B.2 - F.2) = 8/9

-- Main theorem
theorem parabola_and_line_properties
  (A B D : ℝ × ℝ)
  (m : ℝ)
  (h1 : C A.1 A.2)
  (h2 : C B.1 B.2)
  (h3 : A.1 = l m A.2)
  (h4 : B.1 = l m B.2)
  (h5 : symmetric_x_axis A D)
  (h6 : dot_product_condition A B) :
  (∃ (t : ℝ), F = (t * B.1 + (1 - t) * D.1, t * B.2 + (1 - t) * D.2)) ∧
  (∃ (a r : ℝ), a = 1/9 ∧ r = 2/3 ∧ ∀ (x y : ℝ), (x - a)^2 + y^2 = r^2 ↔ 
    (x - K.1)^2 + y^2 ≤ r^2 ∧ (x - B.1)^2 + (y - B.2)^2 ≤ r^2 ∧ (x - D.1)^2 + (y - D.2)^2 ≤ r^2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_and_line_properties_l2789_278983


namespace NUMINAMATH_CALUDE_polynomial_identity_l2789_278965

theorem polynomial_identity (x : ℝ) : 
  (x + 1)^4 + 4*(x + 1)^3 + 6*(x + 1)^2 + 4*(x + 1) + 1 = (x + 2)^4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_l2789_278965


namespace NUMINAMATH_CALUDE_correct_operation_l2789_278918

theorem correct_operation (a b : ℝ) : 3 * a^2 * b - 3 * b * a^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l2789_278918


namespace NUMINAMATH_CALUDE_volleyball_team_selection_l2789_278912

/-- The number of ways to choose k elements from a set of n elements -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The total number of players in the volleyball club -/
def total_players : ℕ := 18

/-- The number of quadruplets -/
def num_quadruplets : ℕ := 4

/-- The number of starters to be chosen -/
def num_starters : ℕ := 6

/-- The number of non-quadruplet players -/
def other_players : ℕ := total_players - num_quadruplets

theorem volleyball_team_selection :
  (binomial total_players num_starters) -
  (binomial other_players (num_starters - num_quadruplets)) -
  (binomial other_players num_starters) = 15470 := by sorry

end NUMINAMATH_CALUDE_volleyball_team_selection_l2789_278912


namespace NUMINAMATH_CALUDE_orange_count_l2789_278984

/-- Given a fruit farm that packs oranges, calculate the total number of oranges. -/
theorem orange_count (oranges_per_box : ℝ) (boxes_per_day : ℝ) 
  (h1 : oranges_per_box = 10.0) 
  (h2 : boxes_per_day = 2650.0) : 
  oranges_per_box * boxes_per_day = 26500.0 := by
  sorry

#check orange_count

end NUMINAMATH_CALUDE_orange_count_l2789_278984


namespace NUMINAMATH_CALUDE_same_color_probability_l2789_278925

def total_balls : ℕ := 8 + 5 + 3

def prob_blue : ℚ := 8 / total_balls
def prob_green : ℚ := 5 / total_balls
def prob_red : ℚ := 3 / total_balls

theorem same_color_probability : 
  prob_blue * prob_blue + prob_green * prob_green + prob_red * prob_red = 49 / 128 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l2789_278925


namespace NUMINAMATH_CALUDE_lydia_apple_eating_age_l2789_278903

/-- The age at which Lydia will first eat an apple from her tree -/
def apple_eating_age (tree_bearing_time years_since_planting current_age planting_age : ℕ) : ℕ :=
  planting_age + tree_bearing_time

/-- Proof that Lydia will be 11 when she first eats an apple from her tree -/
theorem lydia_apple_eating_age :
  let tree_bearing_time : ℕ := 7
  let planting_age : ℕ := 4
  let current_age : ℕ := 9
  apple_eating_age tree_bearing_time (current_age - planting_age) current_age planting_age = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_lydia_apple_eating_age_l2789_278903


namespace NUMINAMATH_CALUDE_spinster_count_spinster_count_proof_l2789_278976

theorem spinster_count : ℕ → ℕ → Prop :=
  fun spinsters cats =>
    (spinsters : ℚ) / (cats : ℚ) = 2 / 7 ∧
    cats = spinsters + 55 →
    spinsters = 22

-- The proof is omitted
theorem spinster_count_proof : spinster_count 22 77 := by sorry

end NUMINAMATH_CALUDE_spinster_count_spinster_count_proof_l2789_278976


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l2789_278909

-- Define the polynomial
def p (x : ℝ) : ℝ := x^3 - 18*x^2 + 91*x - 170

-- State the theorem
theorem partial_fraction_decomposition 
  (a b c : ℝ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_roots : p a = 0 ∧ p b = 0 ∧ p c = 0) 
  (D E F : ℝ) 
  (h_decomp : ∀ (s : ℝ), s ≠ a → s ≠ b → s ≠ c → 
    1 / (s^3 - 18*s^2 + 91*s - 170) = D / (s - a) + E / (s - b) + F / (s - c)) :
  D + E + F = 0 := by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l2789_278909


namespace NUMINAMATH_CALUDE_locus_is_circle_l2789_278966

/-- An ellipse with foci F₁ and F₂ -/
structure Ellipse (F₁ F₂ : ℝ × ℝ) where
  a : ℝ
  h : a > 0

/-- A point P on the ellipse -/
def PointOnEllipse (e : Ellipse F₁ F₂) (P : ℝ × ℝ) : Prop :=
  dist P F₁ + dist P F₂ = 2 * e.a

/-- The point Q extended from F₁P such that |PQ| = |PF₂| -/
def ExtendedPoint (P F₁ F₂ : ℝ × ℝ) (Q : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, t > 1 ∧ Q = F₁ + t • (P - F₁) ∧ dist P Q = dist P F₂

/-- The theorem stating that the locus of Q is a circle -/
theorem locus_is_circle (F₁ F₂ : ℝ × ℝ) (e : Ellipse F₁ F₂) :
  ∀ P Q : ℝ × ℝ, PointOnEllipse e P → ExtendedPoint P F₁ F₂ Q →
  ∃ center : ℝ × ℝ, ∃ radius : ℝ, dist Q center = radius :=
sorry

end NUMINAMATH_CALUDE_locus_is_circle_l2789_278966


namespace NUMINAMATH_CALUDE_meal_cost_l2789_278977

theorem meal_cost (total_cost : ℕ) (num_meals : ℕ) (h1 : total_cost = 21) (h2 : num_meals = 3) :
  total_cost / num_meals = 7 := by
sorry

end NUMINAMATH_CALUDE_meal_cost_l2789_278977


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l2789_278978

def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (2, -3)

theorem perpendicular_vectors (k : ℝ) : 
  (k • a - 2 • b) • a = 0 → k = -1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l2789_278978


namespace NUMINAMATH_CALUDE_mrs_hilt_books_read_l2789_278938

def chapters_per_book : ℕ := 17
def total_chapters_read : ℕ := 68

theorem mrs_hilt_books_read :
  total_chapters_read / chapters_per_book = 4 := by sorry

end NUMINAMATH_CALUDE_mrs_hilt_books_read_l2789_278938


namespace NUMINAMATH_CALUDE_tax_calculation_l2789_278917

/-- Calculates the annual income before tax given tax rates and differential savings -/
def annual_income_before_tax (original_rate new_rate : ℚ) (differential_savings : ℚ) : ℚ :=
  differential_savings / (original_rate - new_rate)

/-- Theorem stating that given the specified tax rates and differential savings, 
    the annual income before tax is $34,500 -/
theorem tax_calculation (original_rate new_rate differential_savings : ℚ) 
  (h1 : original_rate = 42 / 100)
  (h2 : new_rate = 28 / 100)
  (h3 : differential_savings = 4830) :
  annual_income_before_tax original_rate new_rate differential_savings = 34500 := by
  sorry

#eval annual_income_before_tax (42/100) (28/100) 4830

end NUMINAMATH_CALUDE_tax_calculation_l2789_278917


namespace NUMINAMATH_CALUDE_triangle_right_angle_l2789_278982

/-- If in a triangle ABC, sin(A+B) = sin(A-B), then the triangle ABC is a right triangle. -/
theorem triangle_right_angle (A B C : ℝ) (h_triangle : A + B + C = π) 
  (h_sin_eq : Real.sin (A + B) = Real.sin (A - B)) : 
  A = π / 2 ∨ B = π / 2 ∨ C = π / 2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_right_angle_l2789_278982


namespace NUMINAMATH_CALUDE_right_triangle_leg_length_l2789_278962

theorem right_triangle_leg_length : ∀ (a b c : ℝ),
  a = 8 →
  c = 17 →
  a^2 + b^2 = c^2 →
  b = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_leg_length_l2789_278962


namespace NUMINAMATH_CALUDE_banner_coverage_count_l2789_278932

/-- A banner is a 2x5 grid with one 1x1 square removed from one of the four corners -/
def Banner : Type := Unit

/-- The grid table dimensions -/
def grid_width : Nat := 18
def grid_height : Nat := 9

/-- The number of banners used to cover the grid -/
def num_banners : Nat := 18

/-- The number of squares in each banner -/
def squares_per_banner : Nat := 9

/-- The number of pairs of banners -/
def num_pairs : Nat := 9

theorem banner_coverage_count : 
  (2 ^ num_pairs : Nat) + (2 ^ num_pairs : Nat) = 1024 := by sorry

end NUMINAMATH_CALUDE_banner_coverage_count_l2789_278932


namespace NUMINAMATH_CALUDE_exists_function_satisfying_condition_l2789_278963

theorem exists_function_satisfying_condition :
  ∃ f : ℕ+ → ℕ+, ∀ n : ℕ+, (n : ℝ)^2 - 1 < (f (f n) : ℝ) ∧ (f (f n) : ℝ) < (n : ℝ)^2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_exists_function_satisfying_condition_l2789_278963


namespace NUMINAMATH_CALUDE_f_of_3_equals_0_l2789_278985

-- Define the function f
def f : ℝ → ℝ := fun x => (x - 1)^2 - 2*(x - 1)

-- State the theorem
theorem f_of_3_equals_0 : f 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_of_3_equals_0_l2789_278985


namespace NUMINAMATH_CALUDE_part_one_part_two_l2789_278926

-- Define the inequality
def inequality (a x : ℝ) : Prop := x^2 - (2*a + 1)*x + a*(a + 1) ≤ 0

-- Define the solution set A
def A (a : ℝ) : Set ℝ := {x | inequality a x}

-- Define set B
def B : Set ℝ := Set.Ioo (-2) 2

-- Part 1
theorem part_one : A 2 ∪ B = Set.Ioc (-2) 3 := by sorry

-- Part 2
theorem part_two : ∀ a : ℝ, A a ∩ B = ∅ ↔ a ≤ -3 ∨ a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2789_278926


namespace NUMINAMATH_CALUDE_brownie_division_l2789_278942

-- Define the dimensions of the pan
def pan_length : ℕ := 24
def pan_width : ℕ := 30

-- Define the dimensions of each brownie piece
def piece_length : ℕ := 3
def piece_width : ℕ := 4

-- Define the number of pieces
def num_pieces : ℕ := 60

-- Theorem statement
theorem brownie_division :
  pan_length * pan_width = num_pieces * piece_length * piece_width :=
by sorry

end NUMINAMATH_CALUDE_brownie_division_l2789_278942


namespace NUMINAMATH_CALUDE_f_increasing_condition_l2789_278941

/-- The quadratic function f(x) = 3x^2 - ax + 4 -/
def f (a : ℝ) (x : ℝ) : ℝ := 3 * x^2 - a * x + 4

/-- The derivative of f(x) with respect to x -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 6 * x - a

/-- The theorem stating the condition for f(x) to be increasing on [-5, +∞) -/
theorem f_increasing_condition (a : ℝ) : 
  (∀ x : ℝ, x ≥ -5 → (f_deriv a x ≥ 0)) ↔ a ≤ -30 := by sorry

end NUMINAMATH_CALUDE_f_increasing_condition_l2789_278941


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l2789_278900

theorem geometric_series_common_ratio 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h_positive : ∀ n, a n > 0) 
  (h_geometric : ∀ n, a (n + 1) = a n * q) 
  (h_sum : a 2 + a 4 = 3) 
  (h_product : a 3 * a 5 = 2) : 
  q = Real.sqrt ((3 * Real.sqrt 2 + 2) / 7) :=
sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l2789_278900


namespace NUMINAMATH_CALUDE_hyperbola_equilateral_triangle_l2789_278997

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x * y = 1

-- Define an equilateral triangle
def is_equilateral_triangle (P Q R : ℝ × ℝ) : Prop :=
  let d₁ := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  let d₂ := Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - R.2)^2)
  let d₃ := Real.sqrt ((R.1 - P.1)^2 + (R.2 - P.2)^2)
  d₁ = d₂ ∧ d₂ = d₃

-- Define the branches of the hyperbola
def on_branch_1 (x y : ℝ) : Prop := x > 0 ∧ y > 0 ∧ hyperbola x y
def on_branch_2 (x y : ℝ) : Prop := x < 0 ∧ y < 0 ∧ hyperbola x y

-- Theorem statement
theorem hyperbola_equilateral_triangle :
  ∀ (P Q R : ℝ × ℝ),
  hyperbola P.1 P.2 → hyperbola Q.1 Q.2 → hyperbola R.1 R.2 →
  is_equilateral_triangle P Q R →
  P = (-1, -1) →
  on_branch_2 P.1 P.2 →
  on_branch_1 Q.1 Q.2 →
  on_branch_1 R.1 R.2 →
  (¬(on_branch_1 P.1 P.2 ∧ on_branch_1 Q.1 Q.2 ∧ on_branch_1 R.1 R.2) ∧
   ¬(on_branch_2 P.1 P.2 ∧ on_branch_2 Q.1 Q.2 ∧ on_branch_2 R.1 R.2)) ∧
  ((Q = (2 - Real.sqrt 3, 2 + Real.sqrt 3) ∧ R = (2 + Real.sqrt 3, 2 - Real.sqrt 3)) ∨
   (Q = (2 + Real.sqrt 3, 2 - Real.sqrt 3) ∧ R = (2 - Real.sqrt 3, 2 + Real.sqrt 3))) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equilateral_triangle_l2789_278997


namespace NUMINAMATH_CALUDE_decagon_ratio_l2789_278943

/-- Represents a decagon with the properties described in the problem -/
structure Decagon :=
  (area : ℝ)
  (bisector_line : Set ℝ × Set ℝ)
  (below_area : ℝ)
  (triangle_base : ℝ)
  (xq : ℝ)
  (qy : ℝ)

/-- The theorem corresponding to the problem -/
theorem decagon_ratio (d : Decagon) : 
  d.area = 15 ∧ 
  d.below_area = 7.5 ∧ 
  d.triangle_base = 7 ∧ 
  d.xq + d.qy = 7 →
  d.xq / d.qy = 2 / 5 :=
by sorry

end NUMINAMATH_CALUDE_decagon_ratio_l2789_278943


namespace NUMINAMATH_CALUDE_journey_time_bounds_l2789_278906

/-- Represents the bus journey from Kimovsk to Moscow -/
structure BusJourney where
  speed : ℝ
  kimovsk_novomoskovsk : ℝ
  novomoskovsk_tula : ℝ
  tula_moscow : ℝ
  kimovsk_tula_time : ℝ
  novomoskovsk_moscow_time : ℝ

/-- The conditions of the bus journey -/
def journey_conditions (j : BusJourney) : Prop :=
  j.speed ≤ 60 ∧
  j.kimovsk_novomoskovsk = 35 ∧
  j.novomoskovsk_tula = 60 ∧
  j.tula_moscow = 200 ∧
  j.kimovsk_tula_time = 2 ∧
  j.novomoskovsk_moscow_time = 5

/-- The theorem stating the bounds of the total journey time -/
theorem journey_time_bounds (j : BusJourney) 
  (h : journey_conditions j) : 
  ∃ (t : ℝ), 5 + 7/12 ≤ t ∧ t ≤ 6 ∧ 
  t = (j.kimovsk_novomoskovsk + j.novomoskovsk_tula + j.tula_moscow) / j.speed :=
sorry

end NUMINAMATH_CALUDE_journey_time_bounds_l2789_278906


namespace NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_l2789_278950

/-- A quadratic function represented by its coefficients -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- Predicate for a quadratic function passing through the origin -/
def passes_through_origin (f : QuadraticFunction) : Prop :=
  f.a * 0^2 + f.b * 0 + f.c = 0

/-- Theorem stating that b = c = 0 is a sufficient condition -/
theorem sufficient_condition (f : QuadraticFunction) (h1 : f.b = 0) (h2 : f.c = 0) :
  passes_through_origin f := by sorry

/-- Theorem stating that b = c = 0 is not a necessary condition -/
theorem not_necessary_condition :
  ∃ f : QuadraticFunction, passes_through_origin f ∧ f.b ≠ 0 := by sorry

end NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_l2789_278950


namespace NUMINAMATH_CALUDE_apple_count_l2789_278927

/-- Given a box of fruit with apples and oranges, prove that the number of apples is 14 -/
theorem apple_count (total_oranges : ℕ) (removed_oranges : ℕ) (apple_percentage : ℚ) : 
  total_oranges = 26 →
  removed_oranges = 20 →
  apple_percentage = 70 / 100 →
  (∃ (apples : ℕ), 
    (apples : ℚ) / ((apples : ℚ) + (total_oranges - removed_oranges : ℚ)) = apple_percentage ∧
    apples = 14) :=
by sorry

end NUMINAMATH_CALUDE_apple_count_l2789_278927


namespace NUMINAMATH_CALUDE_saline_drip_rate_l2789_278940

/-- Proves that the saline drip makes 20 drops per minute given the treatment conditions -/
theorem saline_drip_rate (treatment_duration : ℕ) (drops_per_ml : ℚ) (total_volume : ℚ) :
  treatment_duration = 2 * 60 →  -- 2 hours in minutes
  drops_per_ml = 100 / 5 →       -- 100 drops per 5 ml
  total_volume = 120 →           -- 120 ml total volume
  (total_volume * drops_per_ml) / treatment_duration = 20 := by
  sorry

#check saline_drip_rate

end NUMINAMATH_CALUDE_saline_drip_rate_l2789_278940


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2789_278996

def f (x : ℝ) := x^2 - 2*x - 3

theorem quadratic_inequality (y₁ y₂ y₃ : ℝ) 
  (h₁ : f (-3) = y₁) 
  (h₂ : f (-2) = y₂) 
  (h₃ : f 2 = y₃) : 
  y₃ < y₂ ∧ y₂ < y₁ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2789_278996


namespace NUMINAMATH_CALUDE_restriction_surjective_l2789_278914

theorem restriction_surjective
  (f : Set.Ioc 0 1 → Set.Ioo 0 1)
  (hf_continuous : Continuous f)
  (hf_surjective : Function.Surjective f) :
  ∀ a ∈ Set.Ioo 0 1,
    Function.Surjective (fun x => f ⟨x, by sorry⟩ : Set.Ioo a 1 → Set.Ioo 0 1) :=
by sorry

end NUMINAMATH_CALUDE_restriction_surjective_l2789_278914


namespace NUMINAMATH_CALUDE_derivative_f_at_zero_l2789_278936

def f (x : ℝ) : ℝ := x^2 + 2*x

theorem derivative_f_at_zero : 
  deriv f 0 = 2 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_zero_l2789_278936


namespace NUMINAMATH_CALUDE_triangle_side_length_l2789_278994

theorem triangle_side_length 
  (x y z : ℝ) 
  (X Y Z : ℝ) 
  (h1 : y = 7)
  (h2 : z = 3)
  (h3 : Real.cos (Y - Z) = 7/8)
  (h4 : x > 0 ∧ y > 0 ∧ z > 0)
  (h5 : X + Y + Z = Real.pi)
  (h6 : x / Real.sin X = y / Real.sin Y)
  (h7 : y / Real.sin Y = z / Real.sin Z) :
  x = Real.sqrt 18.625 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2789_278994


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_values_l2789_278954

-- Define the coefficients of the two lines as functions of a
def line1_coeff (a : ℝ) : ℝ × ℝ := (1 - a, a)
def line2_coeff (a : ℝ) : ℝ × ℝ := (2*a + 3, a - 1)

-- Define the perpendicularity condition
def perpendicular (a : ℝ) : Prop :=
  (line1_coeff a).1 * (line2_coeff a).1 + (line1_coeff a).2 * (line2_coeff a).2 = 0

-- State the theorem
theorem perpendicular_lines_a_values :
  ∀ a : ℝ, perpendicular a → a = 1 ∨ a = -3 :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_values_l2789_278954


namespace NUMINAMATH_CALUDE_inscribed_square_perimeter_l2789_278951

/-- The perimeter of a square inscribed in a right triangle -/
theorem inscribed_square_perimeter (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  let x := a * b / (a + b)
  4 * x = 4 * a * b / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_perimeter_l2789_278951


namespace NUMINAMATH_CALUDE_amount_ratio_l2789_278934

theorem amount_ratio (total : ℕ) (r_amount : ℕ) : 
  total = 4000 →
  r_amount = 1600 →
  (r_amount : ℚ) / ((total - r_amount) : ℚ) = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_amount_ratio_l2789_278934


namespace NUMINAMATH_CALUDE_triangle_zero_sum_implies_zero_function_l2789_278986

/-- A function f: ℝ² → ℝ with the property that the sum of its values
    at the vertices of any equilateral triangle with side length 1 is zero. -/
def TriangleZeroSum (f : ℝ × ℝ → ℝ) : Prop :=
  ∀ A B C : ℝ × ℝ, 
    (dist A B = 1 ∧ dist B C = 1 ∧ dist C A = 1) → 
    f A + f B + f C = 0

/-- Theorem stating that any function with the TriangleZeroSum property
    is identically zero everywhere. -/
theorem triangle_zero_sum_implies_zero_function 
  (f : ℝ × ℝ → ℝ) (h : TriangleZeroSum f) : 
  ∀ x : ℝ × ℝ, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_zero_sum_implies_zero_function_l2789_278986


namespace NUMINAMATH_CALUDE_frank_reading_total_l2789_278998

theorem frank_reading_total (book1_pages_per_day book1_days book2_pages_per_day book2_days book3_pages_per_day book3_days : ℕ) :
  book1_pages_per_day = 22 →
  book1_days = 569 →
  book2_pages_per_day = 35 →
  book2_days = 315 →
  book3_pages_per_day = 18 →
  book3_days = 450 →
  book1_pages_per_day * book1_days + book2_pages_per_day * book2_days + book3_pages_per_day * book3_days = 31643 :=
by
  sorry

end NUMINAMATH_CALUDE_frank_reading_total_l2789_278998


namespace NUMINAMATH_CALUDE_cube_sum_magnitude_l2789_278905

theorem cube_sum_magnitude (w z : ℂ) 
  (h1 : Complex.abs (w + z) = 2)
  (h2 : Complex.abs (w^2 + z^2) = 20) :
  Complex.abs (w^3 + z^3) = 56 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_magnitude_l2789_278905


namespace NUMINAMATH_CALUDE_right_triangle_area_l2789_278959

/-- The area of a right-angled triangle with perpendicular sides of lengths √12 cm and √6 cm is 3√2 square centimeters. -/
theorem right_triangle_area : 
  let side1 : ℝ := Real.sqrt 12
  let side2 : ℝ := Real.sqrt 6
  (1 / 2 : ℝ) * side1 * side2 = 3 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2789_278959


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2789_278908

theorem polynomial_division_remainder :
  ∃ q : Polynomial ℝ, x^4 + 2*x^2 - 3 = (x^2 + 3*x + 2) * q + (-21*x - 21) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2789_278908


namespace NUMINAMATH_CALUDE_oil_barrel_difference_l2789_278979

theorem oil_barrel_difference :
  ∀ (a b : ℝ),
  a + b = 100 →
  (a + 15) = 4 * (b - 15) →
  a - b = 30 := by
sorry

end NUMINAMATH_CALUDE_oil_barrel_difference_l2789_278979


namespace NUMINAMATH_CALUDE_ratio_equality_l2789_278924

theorem ratio_equality : ∃ x : ℚ, (2 / 5 : ℚ) / (3 / 7 : ℚ) = x / (1 / 2 : ℚ) ∧ x = 7 / 15 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l2789_278924


namespace NUMINAMATH_CALUDE_digit_difference_in_base_d_l2789_278915

/-- Given digits A and B in base d > 8, if AB̄_d + AĀ_d = 194_d, then A_d - B_d = 5_d -/
theorem digit_difference_in_base_d (d A B : ℕ) (h1 : d > 8) 
  (h2 : A < d ∧ B < d) 
  (h3 : A * d + B + A * d + A = 1 * d * d + 9 * d + 4) : 
  A - B = 5 := by
  sorry

end NUMINAMATH_CALUDE_digit_difference_in_base_d_l2789_278915


namespace NUMINAMATH_CALUDE_bus_fare_payment_possible_l2789_278919

/-- Represents a person with their initial money and final payment -/
structure Person where
  initial_money : ℕ
  final_payment : ℕ

/-- Represents the bus fare payment scenario -/
def BusFareScenario (fare : ℕ) (people : List Person) : Prop :=
  (people.length = 3) ∧
  (∀ p ∈ people, p.final_payment = fare) ∧
  (∃ total : ℕ, total = people.foldl (λ sum person => sum + person.initial_money) 0) ∧
  (∃ payer : Person, payer ∈ people ∧ payer.initial_money ≥ 3 * fare)

/-- Theorem stating that it's possible to pay the bus fare -/
theorem bus_fare_payment_possible (fare : ℕ) (people : List Person) 
  (h : BusFareScenario fare people) : 
  ∃ (final_money : List ℕ), 
    final_money.length = people.length ∧ 
    final_money.sum = people.foldl (λ sum person => sum + person.initial_money) 0 - 3 * fare :=
sorry

end NUMINAMATH_CALUDE_bus_fare_payment_possible_l2789_278919


namespace NUMINAMATH_CALUDE_evas_harvest_l2789_278922

/-- Represents the dimensions of Eva's garden -/
structure GardenDimensions where
  length : ℕ
  width : ℕ

/-- Represents the planting and yield information -/
structure PlantingInfo where
  plantsPerSquareFoot : ℕ
  strawberriesPerPlant : ℕ

/-- Calculates the expected strawberry harvest given garden dimensions and planting information -/
def expectedHarvest (garden : GardenDimensions) (info : PlantingInfo) : ℕ :=
  garden.length * garden.width * info.plantsPerSquareFoot * info.strawberriesPerPlant

/-- Theorem stating that Eva's expected strawberry harvest is 3600 -/
theorem evas_harvest :
  let garden : GardenDimensions := { length := 10, width := 9 }
  let info : PlantingInfo := { plantsPerSquareFoot := 5, strawberriesPerPlant := 8 }
  expectedHarvest garden info = 3600 := by
  sorry


end NUMINAMATH_CALUDE_evas_harvest_l2789_278922


namespace NUMINAMATH_CALUDE_range_of_t_for_true_proposition_l2789_278989

theorem range_of_t_for_true_proposition (t : ℝ) :
  (∀ x : ℝ, x ≥ 1 → (x^2 + 2*x + t) / x > 0) ↔ t > -3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_t_for_true_proposition_l2789_278989


namespace NUMINAMATH_CALUDE_exponent_power_rule_l2789_278944

theorem exponent_power_rule (x : ℝ) : (x^3)^2 = x^6 := by
  sorry

end NUMINAMATH_CALUDE_exponent_power_rule_l2789_278944


namespace NUMINAMATH_CALUDE_max_regions_three_planes_is_eight_l2789_278993

/-- The maximum number of regions into which three planes can divide three-dimensional space -/
def max_regions_three_planes : ℕ := 8

/-- Theorem stating that the maximum number of regions into which three planes can divide three-dimensional space is 8 -/
theorem max_regions_three_planes_is_eight :
  max_regions_three_planes = 8 := by sorry

end NUMINAMATH_CALUDE_max_regions_three_planes_is_eight_l2789_278993


namespace NUMINAMATH_CALUDE_blue_section_probability_l2789_278969

def bernoulli_probability (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  Nat.choose n k * p^k * (1 - p)^(n - k)

theorem blue_section_probability : 
  bernoulli_probability 7 7 (2/7) = 128/823543 := by
  sorry

end NUMINAMATH_CALUDE_blue_section_probability_l2789_278969


namespace NUMINAMATH_CALUDE_sticker_distribution_l2789_278904

theorem sticker_distribution (total_stickers : ℕ) (num_friends : ℕ) 
  (h1 : total_stickers = 72) (h2 : num_friends = 9) :
  total_stickers / num_friends = 8 := by
sorry

end NUMINAMATH_CALUDE_sticker_distribution_l2789_278904


namespace NUMINAMATH_CALUDE_probability_of_zero_in_one_over_99999_l2789_278933

def decimal_expansion (n : ℕ) : List ℕ := 
  if n = 99999 then [0, 0, 0, 0, 1] else sorry

theorem probability_of_zero_in_one_over_99999 : 
  let expansion := decimal_expansion 99999
  let total_digits := expansion.length
  let zero_count := (expansion.filter (· = 0)).length
  (zero_count : ℚ) / total_digits = 4 / 5 := by sorry

end NUMINAMATH_CALUDE_probability_of_zero_in_one_over_99999_l2789_278933


namespace NUMINAMATH_CALUDE_frank_maze_time_l2789_278923

/-- Represents the maximum additional time Frank can spend in the current maze -/
def max_additional_time (current_time : ℕ) (previous_mazes : ℕ) (average_previous : ℕ) (max_average : ℕ) : ℕ :=
  max_average * (previous_mazes + 1) - (average_previous * previous_mazes + current_time)

/-- Theorem stating the maximum additional time Frank can spend in the maze -/
theorem frank_maze_time : max_additional_time 45 4 50 60 = 55 := by
  sorry

end NUMINAMATH_CALUDE_frank_maze_time_l2789_278923


namespace NUMINAMATH_CALUDE_sqrt_of_square_negative_eleven_l2789_278939

theorem sqrt_of_square_negative_eleven : Real.sqrt ((-11)^2) = 11 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_square_negative_eleven_l2789_278939


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l2789_278999

theorem rectangular_solid_surface_area (a b c : ℕ) : 
  Prime a ∧ Prime b ∧ Prime c →
  a * b * c = 399 →
  2 * (a * b + b * c + c * a) = 422 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l2789_278999


namespace NUMINAMATH_CALUDE_journey_possible_l2789_278961

/-- Represents a location along the route -/
structure Location :=
  (distance : ℝ)
  (from_quixajuba : Bool)

/-- Represents a person's state during the journey -/
structure PersonState :=
  (location : Location)
  (has_bicycle : Bool)

/-- Represents the state of the entire system at a given time -/
structure SystemState :=
  (time : ℝ)
  (person_a : PersonState)
  (person_b : PersonState)
  (person_c : PersonState)

/-- Defines the problem parameters -/
def problem_params : (ℝ × ℝ × ℝ) :=
  (24, 6, 18)  -- total_distance, walking_speed, biking_speed

/-- Defines a valid initial state -/
def initial_state : SystemState :=
  { time := 0,
    person_a := { location := { distance := 0, from_quixajuba := true }, has_bicycle := true },
    person_b := { location := { distance := 0, from_quixajuba := true }, has_bicycle := false },
    person_c := { location := { distance := 24, from_quixajuba := false }, has_bicycle := false } }

/-- Defines what it means for a system state to be valid -/
def is_valid_state (params : ℝ × ℝ × ℝ) (state : SystemState) : Prop :=
  let (total_distance, _, _) := params
  0 ≤ state.time ∧
  0 ≤ state.person_a.location.distance ∧ state.person_a.location.distance ≤ total_distance ∧
  0 ≤ state.person_b.location.distance ∧ state.person_b.location.distance ≤ total_distance ∧
  0 ≤ state.person_c.location.distance ∧ state.person_c.location.distance ≤ total_distance ∧
  (state.person_a.has_bicycle ∨ state.person_b.has_bicycle ∨ state.person_c.has_bicycle)

/-- Defines what it means for a system state to be a goal state -/
def is_goal_state (params : ℝ × ℝ × ℝ) (state : SystemState) : Prop :=
  let (total_distance, _, _) := params
  state.person_a.location.distance = total_distance ∧
  state.person_b.location.distance = total_distance ∧
  state.person_c.location.distance = 0 ∧
  state.time ≤ 160/60  -- 2 hours and 40 minutes in decimal hours

/-- The main theorem to be proved -/
theorem journey_possible (params : ℝ × ℝ × ℝ) (init : SystemState) :
  is_valid_state params init →
  ∃ (final : SystemState), is_valid_state params final ∧ is_goal_state params final :=
sorry

end NUMINAMATH_CALUDE_journey_possible_l2789_278961


namespace NUMINAMATH_CALUDE_sixth_term_is_729_l2789_278913

/-- Represents a geometric sequence of positive integers -/
structure GeometricSequence where
  first_term : ℕ
  common_ratio : ℕ
  is_positive : first_term > 0 ∧ common_ratio > 0

/-- Given a geometric sequence and a term number, compute the value of that term -/
def term_value (seq : GeometricSequence) (n : ℕ) : ℕ :=
  seq.first_term * seq.common_ratio ^ (n - 1)

/-- Theorem: In a geometric sequence of positive integers where the first term is 3
    and the fifth term is 243, the sixth term is 729 -/
theorem sixth_term_is_729 (seq : GeometricSequence) 
    (h1 : seq.first_term = 3)
    (h2 : term_value seq 5 = 243) :
  term_value seq 6 = 729 := by
  sorry

end NUMINAMATH_CALUDE_sixth_term_is_729_l2789_278913


namespace NUMINAMATH_CALUDE_parallelogram_bisector_l2789_278991

-- Define the parallelogram
def parallelogram : List (ℝ × ℝ) := [(5, 25), (5, 50), (14, 58), (14, 33)]

-- Define the property of the line
def divides_equally (m n : ℕ) : Prop :=
  let slope := m / n
  ∃ (b : ℝ), 
    (25 + b) / 5 = (58 - b) / 14 ∧ 
    (25 + b) / 5 = slope ∧
    (b > -25 ∧ b < 33)  -- Ensure the line intersects the parallelogram

-- Main theorem
theorem parallelogram_bisector :
  ∃ (m n : ℕ), 
    m.Coprime n ∧
    divides_equally m n ∧
    m = 71 ∧ n = 19 ∧
    m + n = 90 := by sorry

end NUMINAMATH_CALUDE_parallelogram_bisector_l2789_278991


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l2789_278921

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 3 * x - 5 * y = 10

-- Define y-intercept
def is_y_intercept (y : ℝ) : Prop :=
  line_equation 0 y

-- Theorem statement
theorem y_intercept_of_line :
  is_y_intercept (-2) :=
sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l2789_278921


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l2789_278947

def f (x : ℝ) : ℝ := x * (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5)

theorem f_derivative_at_zero : 
  deriv f 0 = -120 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l2789_278947


namespace NUMINAMATH_CALUDE_calculate_at_20at_l2789_278995

-- Define the @ operation (postfix)
def at_post (x : ℤ) : ℤ := 9 - x

-- Define the @ operation (prefix)
def at_pre (x : ℤ) : ℤ := x - 9

-- Theorem statement
theorem calculate_at_20at : at_pre (at_post 20) = -20 := by
  sorry

end NUMINAMATH_CALUDE_calculate_at_20at_l2789_278995


namespace NUMINAMATH_CALUDE_cars_time_passage_l2789_278964

/-- Given that a car comes down the road every 20 minutes, 
    prove that the time passed for 30 cars is 10 hours. -/
theorem cars_time_passage (interval : ℕ) (num_cars : ℕ) (hours_per_day : ℕ) :
  interval = 20 →
  num_cars = 30 →
  hours_per_day = 24 →
  (interval * num_cars) / 60 = 10 := by
  sorry

end NUMINAMATH_CALUDE_cars_time_passage_l2789_278964


namespace NUMINAMATH_CALUDE_kotelmel_triangle_area_error_l2789_278992

/-- The margin of error between Kotelmel's formula and the correct formula for the area of an equilateral triangle --/
theorem kotelmel_triangle_area_error :
  let a : ℝ := 1  -- We can use any positive real number for a
  let kotelmel_area := (1/3 + 1/10) * a^2
  let correct_area := (a^2 / 4) * Real.sqrt 3
  let error_percentage := |correct_area - kotelmel_area| / correct_area * 100
  ∃ ε > 0, error_percentage < 0.075 + ε ∧ error_percentage > 0.075 - ε :=
by sorry


end NUMINAMATH_CALUDE_kotelmel_triangle_area_error_l2789_278992


namespace NUMINAMATH_CALUDE_total_sleep_is_53_l2789_278988

/-- Represents Janna's sleep schedule and calculates total weekly sleep hours -/
def weekly_sleep_hours : ℝ :=
  let weekday_sleep := 7
  let weekend_sleep := 8
  let nap_hours := 0.5
  let friday_extra := 1
  
  -- Monday, Wednesday
  2 * weekday_sleep +
  -- Tuesday, Thursday (with naps)
  2 * (weekday_sleep + nap_hours) +
  -- Friday (with extra hour)
  (weekday_sleep + friday_extra) +
  -- Saturday, Sunday
  2 * weekend_sleep

/-- Theorem stating that Janna's total sleep hours in a week is 53 -/
theorem total_sleep_is_53 : weekly_sleep_hours = 53 := by
  sorry

end NUMINAMATH_CALUDE_total_sleep_is_53_l2789_278988


namespace NUMINAMATH_CALUDE_inscribed_sphere_volume_l2789_278949

/-- The volume of a sphere inscribed in a cube with edge length 8 feet -/
theorem inscribed_sphere_volume :
  let cube_edge : ℝ := 8
  let sphere_radius : ℝ := cube_edge / 2
  let sphere_volume : ℝ := (4 / 3) * Real.pi * sphere_radius ^ 3
  sphere_volume = (256 / 3) * Real.pi := by sorry

end NUMINAMATH_CALUDE_inscribed_sphere_volume_l2789_278949


namespace NUMINAMATH_CALUDE_no_valid_day_for_statements_l2789_278948

/-- Represents the days of the week -/
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents whether a statement is true or false on a given day -/
def Statement := Day → Prop

/-- The statement "I lied yesterday" -/
def LiedYesterday : Statement := fun d => 
  match d with
  | Day.Monday => false     -- Sunday's statement
  | Day.Tuesday => false    -- Monday's statement
  | Day.Wednesday => false  -- Tuesday's statement
  | Day.Thursday => false   -- Wednesday's statement
  | Day.Friday => false     -- Thursday's statement
  | Day.Saturday => false   -- Friday's statement
  | Day.Sunday => false     -- Saturday's statement

/-- The statement "I will lie tomorrow" -/
def WillLieTomorrow : Statement := fun d =>
  match d with
  | Day.Monday => false     -- Tuesday's statement
  | Day.Tuesday => false    -- Wednesday's statement
  | Day.Wednesday => false  -- Thursday's statement
  | Day.Thursday => false   -- Friday's statement
  | Day.Friday => false     -- Saturday's statement
  | Day.Saturday => false   -- Sunday's statement
  | Day.Sunday => false     -- Monday's statement

/-- Theorem stating that there is no day where both statements can be made without contradiction -/
theorem no_valid_day_for_statements : ¬∃ (d : Day), LiedYesterday d ∧ WillLieTomorrow d := by
  sorry


end NUMINAMATH_CALUDE_no_valid_day_for_statements_l2789_278948


namespace NUMINAMATH_CALUDE_power_sum_equals_eight_l2789_278973

theorem power_sum_equals_eight :
  (-2)^3 + (-2)^2 + (-2)^1 + 2^1 + 2^2 + 2^3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equals_eight_l2789_278973


namespace NUMINAMATH_CALUDE_inequality_proof_l2789_278971

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  b^2 / a + c^2 / b + a^2 / c ≥ a + b + c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2789_278971


namespace NUMINAMATH_CALUDE_copper_percentage_bounds_l2789_278916

/-- Represents an alloy composition -/
structure Alloy where
  nickel : ℝ
  copper : ℝ
  manganese : ℝ
  sum_to_one : nickel + copper + manganese = 1

/-- The three given alloys -/
def alloy1 : Alloy := ⟨0.3, 0.7, 0, by norm_num⟩
def alloy2 : Alloy := ⟨0, 0.1, 0.9, by norm_num⟩
def alloy3 : Alloy := ⟨0.15, 0.25, 0.6, by norm_num⟩

/-- The theorem stating the bounds on copper percentage in the new alloy -/
theorem copper_percentage_bounds (x₁ x₂ x₃ : ℝ) 
  (sum_to_one : x₁ + x₂ + x₃ = 1)
  (manganese_constraint : 0.9 * x₂ + 0.6 * x₃ = 0.4) :
  let copper_percentage := 0.7 * x₁ + 0.1 * x₂ + 0.25 * x₃
  0.4 ≤ copper_percentage ∧ copper_percentage ≤ 13/30 := by
  sorry


end NUMINAMATH_CALUDE_copper_percentage_bounds_l2789_278916


namespace NUMINAMATH_CALUDE_scientific_notation_14800_l2789_278929

theorem scientific_notation_14800 :
  14800 = 1.48 * (10 : ℝ)^4 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_14800_l2789_278929


namespace NUMINAMATH_CALUDE_edwin_alvin_age_difference_l2789_278980

/-- Represents the age difference between Edwin and Alvin -/
def ageDifference (edwinAge alvinAge : ℝ) : ℝ := edwinAge - alvinAge

/-- Theorem stating the age difference between Edwin and Alvin -/
theorem edwin_alvin_age_difference :
  ∃ (edwinAge alvinAge : ℝ),
    edwinAge > alvinAge ∧
    edwinAge + 2 = (1/3) * (alvinAge + 2) + 20 ∧
    edwinAge + alvinAge = 30.99999999 ∧
    ageDifference edwinAge alvinAge = 12 := by sorry

end NUMINAMATH_CALUDE_edwin_alvin_age_difference_l2789_278980


namespace NUMINAMATH_CALUDE_objective_function_has_only_minimum_l2789_278930

-- Define the variables and objective function
variable (x y : ℝ)
def z : ℝ → ℝ → ℝ := λ x y => 3*x + 5*y

-- Define the constraints
def constraint1 (x y : ℝ) : Prop := 6*x + 3*y < 15
def constraint2 (x y : ℝ) : Prop := y ≤ x + 1
def constraint3 (x y : ℝ) : Prop := x - 5*y ≤ 3

-- Theorem statement
theorem objective_function_has_only_minimum :
  ∃ (min : ℝ), ∀ (x y : ℝ), constraint1 x y → constraint2 x y → constraint3 x y →
    z x y ≥ min ∧ ¬∃ (max : ℝ), ∀ (x y : ℝ), constraint1 x y → constraint2 x y → constraint3 x y →
      z x y ≤ max :=
sorry

end NUMINAMATH_CALUDE_objective_function_has_only_minimum_l2789_278930


namespace NUMINAMATH_CALUDE_systematic_sample_interval_count_l2789_278937

/-- Calculates the number of sampled individuals within a given interval in a systematic sample. -/
def sampledInInterval (totalPopulation : ℕ) (sampleSize : ℕ) (intervalStart : ℕ) (intervalEnd : ℕ) : ℕ :=
  let groupDistance := totalPopulation / sampleSize
  (intervalEnd - intervalStart + 1) / groupDistance

/-- Theorem stating that for the given parameters, the number of sampled individuals in the interval [61, 140] is 4. -/
theorem systematic_sample_interval_count :
  sampledInInterval 840 42 61 140 = 4 := by
  sorry

#eval sampledInInterval 840 42 61 140

end NUMINAMATH_CALUDE_systematic_sample_interval_count_l2789_278937


namespace NUMINAMATH_CALUDE_prime_square_product_equality_l2789_278911

theorem prime_square_product_equality (p : ℕ) (x y : ℕ) (h_prime : Nat.Prime p) (h_p_gt_2 : p > 2)
  (h_x_range : x ∈ Finset.range ((p - 1) / 2 + 1) \ {0})
  (h_y_range : y ∈ Finset.range ((p - 1) / 2 + 1) \ {0})
  (h_square : ∃ k : ℕ, x * (p - x) * y * (p - y) = k^2) :
  x = y := by
sorry

end NUMINAMATH_CALUDE_prime_square_product_equality_l2789_278911


namespace NUMINAMATH_CALUDE_function_composition_equality_l2789_278987

theorem function_composition_equality (a b : ℝ) :
  (∀ x, (3 * ((a * x + b) : ℝ) - 4 = 4 * x + 5)) →
  a + b = 13 / 3 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_equality_l2789_278987


namespace NUMINAMATH_CALUDE_least_cookies_l2789_278946

theorem least_cookies (n : ℕ) : n = 59 ↔ 
  n > 0 ∧ 
  n % 6 = 5 ∧ 
  n % 8 = 3 ∧ 
  n % 9 = 6 ∧
  ∀ m : ℕ, m > 0 → m % 6 = 5 → m % 8 = 3 → m % 9 = 6 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_least_cookies_l2789_278946


namespace NUMINAMATH_CALUDE_third_term_value_l2789_278970

def S (n : ℕ) : ℤ := 2 * n^2 - 1

def a (n : ℕ) : ℤ := S n - S (n - 1)

theorem third_term_value : a 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_third_term_value_l2789_278970
