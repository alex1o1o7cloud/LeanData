import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volleyball_tournament_solution_l979_97902

/-- Represents the number of teams from city A -/
def n : ℕ := sorry

/-- Total number of games played -/
def total_games (n : ℕ) : ℕ := (n + 2*n).choose 2

/-- Number of games among teams from city A -/
def games_A (n : ℕ) : ℕ := n.choose 2

/-- Number of games among teams from city B -/
def games_B (n : ℕ) : ℕ := (2*n).choose 2

/-- Number of games between teams from A and B -/
def games_AB (n : ℕ) : ℕ := 2 * n^2

/-- Number of wins by teams from city A in inter-city games -/
def wins_A_inter (n : ℕ) (m : ℕ) : ℕ := m

/-- Number of wins by teams from city B in inter-city games -/
def wins_B_inter (n : ℕ) (m : ℕ) : ℕ := 2 * n^2 - m

/-- Total wins by teams from city A -/
def total_wins_A (n : ℕ) (m : ℕ) : ℕ := games_A n + wins_A_inter n m

/-- Total wins by teams from city B -/
def total_wins_B (n : ℕ) (m : ℕ) : ℕ := games_B n + wins_B_inter n m

/-- The main theorem stating that n = 5 is the only solution -/
theorem volleyball_tournament_solution :
  (∃ (m : ℕ), (3 * total_wins_A n m = 4 * total_wins_B n m) ∧ 
  (total_wins_A n m + total_wins_B n m = total_games n)) ↔ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volleyball_tournament_solution_l979_97902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_in_20_sided_polygon_l979_97976

/-- Represents a convex polygon with n sides --/
structure ConvexPolygon (n : ℕ) where
  -- The number of sides is at least 3
  sides_ge_three : n ≥ 3
  -- The angles are represented as a list of natural numbers
  angles : List ℕ
  -- The number of angles is equal to the number of sides
  angle_count : angles.length = n
  -- The sum of angles is (n-2) * 180
  angle_sum : angles.sum = (n - 2) * 180
  -- All angles are less than 180 degrees (convexity)
  all_angles_lt_180 : ∀ a ∈ angles, a < 180

/-- Checks if a list of natural numbers forms an increasing arithmetic sequence --/
def isIncreasingArithmeticSequence (l : List ℕ) : Prop :=
  ∃ d : ℕ, d > 0 ∧ ∀ i : ℕ, i + 1 < l.length → l[i+1]! - l[i]! = d

/-- The main theorem --/
theorem smallest_angle_in_20_sided_polygon 
  (p : ConvexPolygon 20) 
  (h_arithmetic : isIncreasingArithmeticSequence p.angles) :
  p.angles[0]! = 143 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_in_20_sided_polygon_l979_97976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vendor_apple_waste_percentage_l979_97925

/-- Calculates the percentage of apples thrown away by a vendor over two days -/
theorem vendor_apple_waste_percentage : (42 : ℝ) = 42 := by
  let initial_apples : ℝ := 100
  let first_day_sold_percent : ℝ := 30
  let first_day_waste_percent : ℝ := 20
  let second_day_sold_percent : ℝ := 50

  let first_day_remaining := initial_apples * (1 - first_day_sold_percent / 100)
  let first_day_waste := first_day_remaining * (first_day_waste_percent / 100)
  let second_day_start := first_day_remaining - first_day_waste
  let second_day_sold := second_day_start * (second_day_sold_percent / 100)
  let second_day_waste := second_day_start - second_day_sold

  let total_waste := first_day_waste + second_day_waste
  let waste_percentage := (total_waste / initial_apples) * 100

  sorry  -- Skip the proof for now

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vendor_apple_waste_percentage_l979_97925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_continuous_piecewise_function_sum_l979_97927

-- Define the piecewise function g(x)
noncomputable def g (c d : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then c * x + 2
  else if x ≥ -1 then 2 * x - 3
  else 3 * x - d

-- State the theorem
theorem continuous_piecewise_function_sum (c d : ℝ) :
  (Continuous (g c d)) → c + d = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_continuous_piecewise_function_sum_l979_97927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_is_pi_over_three_perimeter_is_twelve_l979_97948

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def isAcute (t : Triangle) : Prop :=
  0 < t.A ∧ t.A < Real.pi/2 ∧
  0 < t.B ∧ t.B < Real.pi/2 ∧
  0 < t.C ∧ t.C < Real.pi/2

def satisfiesCondition (t : Triangle) : Prop :=
  Real.sqrt 3 * t.c = 2 * t.a * Real.sin t.C

def hasAreaTwoSqrtThree (t : Triangle) : Prop :=
  1/2 * t.a * t.b * Real.sin t.C = 2 * Real.sqrt 3

def sideAIsFive (t : Triangle) : Prop :=
  t.a = 5

-- State the theorems
theorem angle_A_is_pi_over_three (t : Triangle) 
  (h1 : isAcute t) (h2 : satisfiesCondition t) : 
  t.A = Real.pi/3 := by sorry

theorem perimeter_is_twelve (t : Triangle) 
  (h1 : isAcute t) (h2 : satisfiesCondition t) 
  (h3 : hasAreaTwoSqrtThree t) (h4 : sideAIsFive t) : 
  t.a + t.b + t.c = 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_is_pi_over_three_perimeter_is_twelve_l979_97948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_parabola_l979_97962

/-- The curve described by the polar equation r = 6 tan θ sec θ is a parabola -/
theorem polar_to_parabola (r θ : ℝ) (x y : ℝ) :
  r = 6 * Real.tan θ * (1 / Real.cos θ) →
  x = r * Real.cos θ →
  y = r * Real.sin θ →
  x^2 = 6 * y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_parabola_l979_97962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_point_l979_97919

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := Real.exp (-5 * x) + 2

-- Define the point of tangency
def point : ℝ × ℝ := (0, 3)

-- Define the slope of the tangent line
def m : ℝ := -5

-- Define the equation of the tangent line
def tangent_line (x : ℝ) : ℝ := m * x + 3

-- Theorem statement
theorem tangent_line_at_point :
  (∀ x, tangent_line x = m * (x - point.1) + point.2) ∧
  (tangent_line point.1 = f point.1) ∧
  (m = (deriv f) point.1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_point_l979_97919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_m_l979_97935

noncomputable def f (x : ℝ) (φ : ℝ) := Real.sin (2 * x + φ)

theorem min_value_of_m (φ : ℝ) (h1 : |φ| < π/2) 
  (h2 : ∀ x, f (x + π/6) φ = f (-x + π/6) φ) 
  (h3 : ∃ x ∈ Set.Icc 0 (π/2), f x φ ≤ -1/2) : 
  (∀ m, (∃ x ∈ Set.Icc 0 (π/2), f x φ ≤ m) → m ≥ -1/2) ∧ 
  (∃ x ∈ Set.Icc 0 (π/2), f x φ = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_m_l979_97935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_approx_l979_97955

/-- An isosceles trapezoid with given dimensions -/
structure IsoscelesTrapezoid where
  leg : ℝ
  diagonal : ℝ
  longerBase : ℝ

/-- Calculate the area of the isosceles trapezoid -/
noncomputable def calculateArea (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem stating that the area of the specific trapezoid is approximately 1189.67 -/
theorem trapezoid_area_approx :
  let t : IsoscelesTrapezoid := { leg := 40, diagonal := 50, longerBase := 60 }
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |calculateArea t - 1189.67| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_approx_l979_97955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_q_change_l979_97946

/-- Given a function q defined in terms of w, d, and z, this theorem proves
    how q changes when w is quadrupled, d is doubled, and z is tripled. -/
theorem q_change (w d z : ℝ) (q : ℝ) (h : q = 5 * w / (4 * d * z^2)) :
  (5 * (4 * w) / (4 * (2 * d) * (3 * z)^2)) = (5 / 18) * q := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_q_change_l979_97946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l979_97953

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides

-- Define the given conditions
def given_condition (t : Triangle) : Prop :=
  Real.cos (2 * t.A) + 2 * (Real.sin t.B)^2 + 2 * (Real.sin t.C)^2 - 2 * Real.sqrt 3 * Real.sin t.B * Real.sin t.C = 1

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : given_condition t) 
  (h2 : t.b = Real.sqrt 3) 
  (h3 : t.c = 4) : 
  t.A = Real.pi / 6 ∧ Real.pi * (t.a / (2 * Real.sin t.A))^2 = 7 * Real.pi :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l979_97953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_properties_rectangle_opposite_sides_square_all_sides_equal_l979_97911

-- Define a quadrilateral
structure Quadrilateral where
  sides : Fin 4 → ℝ
  angles : Fin 4 → ℝ

-- Define a rectangle
structure Rectangle extends Quadrilateral where
  opposite_sides_equal : sides 0 = sides 2 ∧ sides 1 = sides 3

-- Define a square
structure Square extends Rectangle where
  all_sides_equal : sides 0 = sides 1 ∧ sides 1 = sides 2 ∧ sides 2 = sides 3

-- Theorem statements
theorem quadrilateral_properties (q : Quadrilateral) : 
  (Fintype.card (Fin 4) = 4) ∧ 
  (Fintype.card (Fin 4) = 4) :=
by
  apply And.intro
  · exact rfl
  · exact rfl

theorem rectangle_opposite_sides (r : Rectangle) : 
  r.sides 0 = r.sides 2 ∧ r.sides 1 = r.sides 3 :=
by
  exact r.opposite_sides_equal

theorem square_all_sides_equal (s : Square) : 
  s.sides 0 = s.sides 1 ∧ s.sides 1 = s.sides 2 ∧ s.sides 2 = s.sides 3 :=
by
  exact s.all_sides_equal

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_properties_rectangle_opposite_sides_square_all_sides_equal_l979_97911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_scurries_value_l979_97956

-- Define the number of people and horses
def n : ℕ := 8

-- Define a function to calculate the probability of scurrying home for each person
def scurry_prob (i : ℕ) : ℚ :=
  if i = 1 then 0
  else (i - 1 : ℚ) / ↑i

-- Define the expected number of people who scurry home
def expected_scurries : ℚ :=
  (Finset.range n).sum (λ i ↦ scurry_prob (i + 1))

-- Theorem statement
theorem expected_scurries_value :
  expected_scurries = 37 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_scurries_value_l979_97956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_exists_l979_97990

-- Define the plane
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

-- Define the two intersecting lines
variable (L1 L2 : Set V)

-- Define the intersection point of the two lines
variable (O : V)

-- Define the two moving points
variable (A B : ℝ → V)

-- State the theorem
theorem equidistant_point_exists 
  (h_intersect : L1 ∩ L2 = {O})
  (h_speed : ∃ (v : ℝ), ∀ (t : ℝ), ‖A t - O‖ = v * |t| ∧ ‖B t - O‖ = v * |t|)
  (h_on_lines : ∀ (t : ℝ), A t ∈ L1 ∧ B t ∈ L2) :
  ∃ (M : V), ∀ (t : ℝ), dist (A t) M = dist (B t) M :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_exists_l979_97990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_circle_centers_l979_97961

/-- Represents a circle in the real plane. -/
structure Circle (α : Type) where
  center : α × α
  radius : ℝ

/-- Represents that two circles have perpendicular common internal tangents. -/
def PerpendicularCommonInternalTangents (c1 c2 : Circle ℝ) : Prop := sorry

/-- Represents the length of a chord in a circle. -/
def ChordLength (c : Circle ℝ) : ℝ := sorry

/-- Represents the distance between the centers of two circles. -/
def DistanceBetweenCenters (c1 c2 : Circle ℝ) : ℝ := sorry

/-- Given two circles with mutually perpendicular common internal tangents and 
    chords connecting the points of tangency of lengths 3 and 5, 
    the distance between the centers of the circles is 4√2. -/
theorem distance_between_circle_centers 
  (circle1 circle2 : Circle ℝ) 
  (tangents_perpendicular : PerpendicularCommonInternalTangents circle1 circle2)
  (chord1_length : ChordLength circle1 = 3)
  (chord2_length : ChordLength circle2 = 5) :
  DistanceBetweenCenters circle1 circle2 = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_circle_centers_l979_97961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l979_97907

-- Define the points in the coordinate plane
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (8, 0)
def C : ℝ × ℝ := (15, 0)

-- Define the initial slopes of the lines
def slope_ℓA : ℝ := -1
def slope_ℓC : ℝ := 1

-- Define the function to calculate the area of the triangle formed by the rotating lines
noncomputable def triangle_area (θ : ℝ) : ℝ :=
  (1/2) * 8 * 31 * Real.tan θ * Real.sin θ

-- State the theorem
theorem max_triangle_area :
  ∃ θ : ℝ, ∀ φ : ℝ, triangle_area θ ≥ triangle_area φ ∧ triangle_area θ = 124 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l979_97907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_total_profit_optimal_selling_time_l979_97900

-- Define the problem parameters
def initial_cost : ℚ := 1020000
def annual_income : ℚ := 200000
def first_year_maintenance : ℚ := 10000
def maintenance_increase : ℚ := 10000
def max_years : ℕ := 20

-- Define the total profit function
noncomputable def total_profit (x : ℕ) : ℚ :=
  -1/2 * (x : ℚ)^2 + 19 * (x : ℚ) - 72

-- Define the average annual profit function
noncomputable def avg_annual_profit (x : ℕ) : ℚ :=
  total_profit x / (x : ℚ)

-- Theorem for the maximum total profit
theorem max_total_profit :
  ∃ (x : ℕ), x ≤ max_years ∧ total_profit x = 217/2 ∧
  ∀ (y : ℕ), y ≤ max_years → total_profit y ≤ total_profit x :=
by sorry

-- Theorem for the optimal selling time
theorem optimal_selling_time :
  ∃ (x : ℕ), x ≤ max_years ∧ x = 12 ∧
  ∀ (y : ℕ), y ≤ max_years → avg_annual_profit y ≤ avg_annual_profit x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_total_profit_optimal_selling_time_l979_97900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_option1_cheapest_l979_97945

-- Define constants
noncomputable def train_ticket_cost : ℝ := 200
noncomputable def berries_collected : ℝ := 5
noncomputable def market_berry_price : ℝ := 150
noncomputable def sugar_price : ℝ := 54
noncomputable def jam_yield : ℝ := 1.5
noncomputable def ready_made_jam_price : ℝ := 220

-- Define cost functions
noncomputable def cost_option1 : ℝ := 
  (train_ticket_cost / berries_collected + sugar_price) * jam_yield

noncomputable def cost_option2 : ℝ := 
  (market_berry_price + sugar_price) * jam_yield

noncomputable def cost_option3 : ℝ := 
  ready_made_jam_price * jam_yield

-- Theorem statement
theorem option1_cheapest : 
  cost_option1 < cost_option2 ∧ cost_option1 < cost_option3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_option1_cheapest_l979_97945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_2_f_nonnegative_iff_a_leq_2_l979_97910

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log (x - 1) - a * (x - 2)

-- Define the derivative of f
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := Real.log (x - 1) + x / (x - 1) - a

theorem tangent_line_at_2 (a : ℝ) :
  a = 2017 →
  ∃ m b : ℝ, m = -2015 ∧ b = 4030 ∧
  ∀ x y : ℝ, y = m * (x - 2) + f 2017 2 ↔ 2015 * x + y - 4030 = 0 :=
by sorry

theorem f_nonnegative_iff_a_leq_2 (a : ℝ) :
  (∀ x : ℝ, x ≥ 2 → f a x ≥ 0) ↔ a ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_2_f_nonnegative_iff_a_leq_2_l979_97910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l979_97934

theorem equation_solution : 
  {x : ℝ | (2 : ℝ)^(2*x) - 6*(2 : ℝ)^x + 8 = 0} = {1, 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l979_97934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_of_sequence_l979_97906

def sequence' (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | n + 1 =>
    if n % 2 = 0 then
      2 * sequence' n
    else
      2 + sequence' n

theorem divisibility_of_sequence' (p : ℕ) (hp : Nat.Prime p) (hp_gt_3 : p > 3) :
  ∃ S : Set ℕ, Set.Infinite S ∧ ∀ n ∈ S, (2^(2*p) - 1) / 3 ∣ sequence' n :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_of_sequence_l979_97906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_line_m_value_l979_97999

noncomputable section

-- Define the data points
def x_values : List ℝ := [2, 3, 4, 5]
def y_values (m : ℝ) : List ℝ := [15, m, 30, 35]

-- Define the regression line equation
def regression_line (x : ℝ) : ℝ := 7 * x + 0.5

-- Calculate the mean of x values
noncomputable def mean_x : ℝ := (x_values.sum) / (x_values.length : ℝ)

-- Calculate the mean of y values
noncomputable def mean_y (m : ℝ) : ℝ := (y_values m).sum / ((y_values m).length : ℝ)

-- Theorem stating that m = 20 satisfies the conditions
theorem regression_line_m_value :
  ∃ (m : ℝ), m = 20 ∧
  regression_line mean_x = mean_y m ∧
  y_values m = [15, m, 30, 35] :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_line_m_value_l979_97999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_suggestion_difference_l979_97964

def food_suggestions : List Nat := [408, 305, 137, 213, 137]

theorem suggestion_difference : 
  (List.maximum? food_suggestions).getD 0 - (List.minimum? food_suggestions).getD 0 = 271 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_suggestion_difference_l979_97964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_and_min_value_l979_97930

noncomputable section

-- Define the functions g and f
noncomputable def g (x : ℝ) : ℝ := x / Real.log x
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := g x - a * x

-- State the theorem
theorem monotonicity_and_min_value :
  -- g is increasing on (e, +∞)
  (∀ x y, Real.exp 1 < x ∧ x < y → g x < g y) ∧
  -- g is decreasing on (0,1) and (1,e)
  (∀ x y, 0 < x ∧ x < y ∧ y < 1 → g x > g y) ∧
  (∀ x y, 1 < x ∧ x < y ∧ y < Real.exp 1 → g x > g y) ∧
  -- If f is decreasing on (1, +∞), then the minimum value of a is 1/4
  (∀ a, (∀ x y, 1 < x ∧ x < y → f a x > f a y) → a ≥ 1/4) ∧
  (∃ a, a = 1/4 ∧ ∀ x y, 1 < x ∧ x < y → f a x > f a y) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_and_min_value_l979_97930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l979_97920

-- Define the ellipse M
def ellipse_M (x y : ℝ) : Prop := x^2/4 + y^2/2 = 1

-- Define the direction vector of line l
noncomputable def direction_vector : ℝ × ℝ := (1, Real.sqrt 2)

-- Define point A
noncomputable def point_A : ℝ × ℝ := (1, Real.sqrt 2)

-- State that point A lies on ellipse M
axiom A_on_M : ellipse_M point_A.1 point_A.2

-- Define the area of triangle ABC
noncomputable def triangle_area (B C : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem max_triangle_area :
  ∃ (B C : ℝ × ℝ), 
    ellipse_M B.1 B.2 ∧ 
    ellipse_M C.1 C.2 ∧ 
    (∃ (t : ℝ), B.1 - C.1 = t * direction_vector.1 ∧ B.2 - C.2 = t * direction_vector.2) ∧
    (∀ (B' C' : ℝ × ℝ), 
      ellipse_M B'.1 B'.2 → 
      ellipse_M C'.1 C'.2 → 
      (∃ (t' : ℝ), B'.1 - C'.1 = t' * direction_vector.1 ∧ B'.2 - C'.2 = t' * direction_vector.2) →
      triangle_area B' C' ≤ triangle_area B C) ∧
    triangle_area B C = Real.sqrt 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l979_97920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_line_l979_97915

/-- Definition of the circle C -/
def circleC (m : ℝ) (x y : ℝ) : Prop :=
  (x - 5)^2 + (y + 1)^2 = m ∧ m > 0

/-- Definition of the line L -/
def lineL (x y : ℝ) : Prop :=
  4*x + 3*y - 2 = 0

/-- Distance function from a point to the line -/
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  |4*x + 3*y - 2| / Real.sqrt (4^2 + 3^2)

/-- Theorem stating that if there's exactly one point on the circle
    with distance 1 to the line, then m = 4 -/
theorem circle_tangent_line (m : ℝ) :
  (∃! p : ℝ × ℝ, circleC m p.1 p.2 ∧ distance_to_line p.1 p.2 = 1) →
  m = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_line_l979_97915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l979_97983

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

-- State the theorem
theorem f_increasing_on_interval (a b : ℝ) 
  (h1 : 0 < a) (h2 : a < b) (h3 : b < Real.exp 1) : f a < f b := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l979_97983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_increase_by_one_fifth_l979_97932

theorem fraction_increase_by_one_fifth : ∃! (x y : ℕ+), 
  (Nat.Coprime x.val y.val) ∧ 
  ((x + 1 : ℚ) / (y + 1)) = 6/5 * (x / y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_increase_by_one_fifth_l979_97932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l979_97991

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  2 * t.a * Real.sin (t.C + Real.pi/6) = t.b + t.c ∧
  t.B = Real.pi/4 ∧
  t.b - t.a = Real.sqrt 2 - Real.sqrt 3

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : satisfies_conditions t) :
  t.A = Real.pi/3 ∧ 
  (1/2 * t.a * t.b * Real.sin t.C) = (3 + Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l979_97991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l979_97989

noncomputable def f (a x : ℝ) : ℝ := a * Real.sqrt (1 - x^2) + Real.sqrt (1 + x) + Real.sqrt (1 - x)

noncomputable def t (x : ℝ) : ℝ := Real.sqrt (1 + x) + Real.sqrt (1 - x)

noncomputable def h (a t : ℝ) : ℝ := (1/2) * a * t^2 + t - a

noncomputable def g (a : ℝ) : ℝ :=
  if a ≤ -Real.sqrt 2 / 2 then Real.sqrt 2
  else if a ≤ -1/2 then -a - 1 / (2*a)
  else a + 2

theorem f_properties (a : ℝ) (ha : a < 0) :
  (∀ x, x ∈ Set.Icc (-1) 1 → t x ∈ Set.Icc (Real.sqrt 2) 2) ∧
  (∀ x, x ∈ Set.Icc (-1) 1 → f a x = h a (t x)) ∧
  (∀ x, x ∈ Set.Icc (-1) 1 → f a x ≤ g a) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l979_97989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_theorem_l979_97942

-- Define the hyperbola and parabola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1
def parabola (b x y : ℝ) : Prop := y^2 = 4 * b * x

-- Define the focus of the parabola
def parabola_focus (b : ℝ) : ℝ × ℝ := (b, 0)

-- Define the foci of the hyperbola
noncomputable def hyperbola_foci (a b : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let c := Real.sqrt (a^2 + b^2)
  ((c, 0), (-c, 0))

-- Define the ratio condition
noncomputable def ratio_condition (a b : ℝ) : Prop :=
  let (f1, f2) := hyperbola_foci a b
  let focus := parabola_focus b
  (focus.1 - f2.1) / (f1.1 - focus.1) = 5 / 3

-- Define the eccentricity of the hyperbola
noncomputable def hyperbola_eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 + b^2 / a^2)

-- State the theorem
theorem hyperbola_eccentricity_theorem (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) 
  (h_ratio : ratio_condition a b) :
  hyperbola_eccentricity a b = 4 * Real.sqrt 15 / 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_theorem_l979_97942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_valid_pdf_l979_97975

-- Define the probability density function
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ -1.5 then 0
  else if x ≤ -0.5 then 0.5 * x + 0.75
  else if x ≤ 0.5 then 0.5
  else if x ≤ 1.5 then -0.5 * x + 0.75
  else 0

-- Theorem statement
theorem f_is_valid_pdf :
  (∫ x, f x) = 1 ∧
  (∀ x, f x ≥ 0) ∧
  (∀ x, x ≤ -1.5 → f x = 0) ∧
  (∀ x, -1.5 < x ∧ x ≤ -0.5 → f x = 0.5 * x + 0.75) ∧
  (∀ x, -0.5 < x ∧ x ≤ 0.5 → f x = 0.5) ∧
  (∀ x, 0.5 < x ∧ x ≤ 1.5 → f x = -0.5 * x + 0.75) ∧
  (∀ x, 1.5 < x → f x = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_valid_pdf_l979_97975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_diameter_DEF_l979_97959

/-- The diameter of the inscribed circle in a triangle --/
noncomputable def inscribed_circle_diameter (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  2 * Real.sqrt (s * (s - a) * (s - b) * (s - c)) / s

/-- Theorem: The diameter of the inscribed circle in triangle DEF --/
theorem inscribed_circle_diameter_DEF :
  inscribed_circle_diameter 13 8 10 = 2 * Real.sqrt (15.5 * 2.5 * 7.5 * 5.5) / 15.5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_diameter_DEF_l979_97959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_16_l979_97922

noncomputable section

-- Define the medians of the triangle
def median_a : ℝ := 5
def median_b : ℝ := 6
def median_c : ℝ := 5

-- Define the formula for the area of a triangle given its medians
noncomputable def triangle_area_from_medians (ma mb mc : ℝ) : ℝ :=
  let sm := (ma + mb + mc) / 2
  (4 / 3) * Real.sqrt (sm * (sm - ma) * (sm - mb) * (sm - mc))

-- Theorem statement
theorem triangle_area_is_16 :
  triangle_area_from_medians median_a median_b median_c = 16 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_16_l979_97922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l979_97979

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 
  (Real.sin (ω * x))^2 - (Real.cos (ω * x))^2 + 2 * (Real.sin (ω * x)) * (Real.cos (ω * x))

theorem function_properties (ω : ℝ) (h1 : 0 < ω) (h2 : ω < 4) 
  (h3 : ∀ x, f ω (x + 3 * π / 16) = f ω (3 * π / 16 - x)) :
  ω = 2 ∧ 
  ∀ x ∈ Set.Icc (5 * π / 48) (11 * π / 48), 
    f ω x ∈ Set.Icc (Real.sqrt 2 / 2) (Real.sqrt 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l979_97979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l979_97905

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1/5) * sin (x + π/3) + cos (x - π/6)

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), M = 6/5 ∧ ∀ (x : ℝ), f x ≤ M := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l979_97905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minesweeper_solutions_l979_97992

/-- Represents a cell in the Minesweeper grid -/
inductive Cell
| Empty : Cell
| Mine : Cell
| Number (n : Nat) : Cell

/-- Represents a Minesweeper grid -/
def MinesweeperGrid := List (List Cell)

/-- Checks if a given position is valid on the grid -/
def isValidPosition (grid : MinesweeperGrid) (row col : Nat) : Bool :=
  match grid with
  | [] => false
  | r::_ => row < grid.length && col < r.length

/-- Counts the number of mines in neighboring cells -/
def countAdjacentMines (grid : MinesweeperGrid) (row col : Nat) : Nat :=
  sorry

/-- Checks if the grid satisfies Minesweeper rules -/
def isValidGrid (grid : MinesweeperGrid) : Bool :=
  sorry

/-- Counts the total number of mines in the grid -/
def countTotalMines (grid : MinesweeperGrid) : Nat :=
  sorry

/-- Gets the size of the grid -/
def gridSize (grid : MinesweeperGrid) : Nat × Nat :=
  match grid with
  | [] => (0, 0)
  | r::_ => (grid.length, r.length)

/-- Theorem: There exist valid 9x6 Minesweeper grids with 7, 8, 9, or 10 mines -/
theorem minesweeper_solutions :
  ∃ (grid7 grid8 grid9 grid10 : MinesweeperGrid),
    (gridSize grid7).1 = 9 ∧ (gridSize grid7).2 = 6 ∧ isValidGrid grid7 ∧ countTotalMines grid7 = 7 ∧
    (gridSize grid8).1 = 9 ∧ (gridSize grid8).2 = 6 ∧ isValidGrid grid8 ∧ countTotalMines grid8 = 8 ∧
    (gridSize grid9).1 = 9 ∧ (gridSize grid9).2 = 6 ∧ isValidGrid grid9 ∧ countTotalMines grid9 = 9 ∧
    (gridSize grid10).1 = 9 ∧ (gridSize grid10).2 = 6 ∧ isValidGrid grid10 ∧ countTotalMines grid10 = 10 :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minesweeper_solutions_l979_97992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_of_f_in_interval_l979_97952

-- Define the function f(x) as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 10 + x - 3

-- State the theorem
theorem zero_of_f_in_interval (k : ℤ) :
  (∃ x : ℝ, x > ↑k ∧ x < ↑k + 1 ∧ f x = 0) → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_of_f_in_interval_l979_97952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_sum_l979_97936

theorem solution_sum (p q r s : ℕ+) (x y : ℝ) :
  x + y = 5 →
  2 * x * y = 5 →
  x = (p : ℝ) + (q : ℝ) * Real.sqrt (r : ℝ) / (s : ℝ) ∨
    x = (p : ℝ) - (q : ℝ) * Real.sqrt (r : ℝ) / (s : ℝ) →
  p + q + r + s = 23 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_sum_l979_97936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tower_lamps_l979_97904

/-- Represents the number of lamps on each floor of a tower -/
def LampSequence (n : ℕ) : ℕ → ℕ := sorry

/-- The total number of lamps in the tower -/
def TotalLamps (seq : ℕ → ℕ) : ℕ := (Finset.range 9).sum seq

theorem tower_lamps (n : ℕ) :
  (∀ k, LampSequence n (k + 1) = LampSequence n k + n) →
  LampSequence n 8 = 13 * LampSequence n 0 →
  TotalLamps (LampSequence n) = 126 →
  LampSequence n 8 = 26 := by
  sorry

#check tower_lamps

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tower_lamps_l979_97904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangement_theorem_l979_97982

/-- The number of ways to arrange 9 distinct objects in a row,
    with the two smallest at the ends and the largest in the middle -/
def arrangement_count : ℕ := 1440

/-- The number of distinct objects -/
def n : ℕ := 9

/-- The position of the middle object in a row of n objects -/
def middle_position (n : ℕ) : ℕ := (n + 1) / 2

/-- Factorial function -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem arrangement_theorem :
  arrangement_count = 2 * factorial (n - 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangement_theorem_l979_97982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_two_integer_solutions_l979_97913

-- Define the equation
def equation (a x : ℝ) : Prop :=
  |a - 3| * |x + 0.5| + 0.5 * |x + 2.5| + |a - x^2| = x^2 + x - 3 * |x + 0.5| + 2.5

-- Define the set of values for a
def valid_a_set : Set ℝ :=
  {0, 1, 4} ∪ Set.Icc 7 8 ∪ Set.Ioo 9 11

-- Define the property of having exactly two integer solutions
def has_two_integer_solutions (a : ℝ) : Prop :=
  ∃ x y : ℤ, x ≠ y ∧ equation a (x : ℝ) ∧ equation a (y : ℝ) ∧
  ∀ z : ℤ, equation a (z : ℝ) → (z = x ∨ z = y)

-- Theorem statement
theorem equation_two_integer_solutions :
  ∀ a : ℝ, has_two_integer_solutions a ↔ a ∈ valid_a_set :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_two_integer_solutions_l979_97913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stationery_count_l979_97969

theorem stationery_count (book_ratio pen_ratio books_bought : ℕ) : 
  book_ratio = 7 →
  pen_ratio = 3 →
  books_bought = 280 →
  (book_ratio * (books_bought / book_ratio) + pen_ratio * (books_bought / book_ratio)) = 400
  := by
    intros h1 h2 h3
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stationery_count_l979_97969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l979_97950

def M : Set ℝ := {-2, -1, 0, 1, 2}
def N : Set ℝ := {x | x + 2 ≥ x^2}

theorem intersection_M_N : M ∩ N = {-1, 0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l979_97950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_folded_paper_holes_l979_97937

/-- Represents a point on the paper -/
structure Point where
  x : Rat
  y : Rat

/-- Represents a hole on the paper -/
structure Hole where
  position : Point

/-- Represents the state of the paper -/
structure Paper where
  width : Rat
  height : Rat
  holes : List Hole

/-- Folds the paper from left to right -/
def foldLeftRight (p : Paper) : Paper :=
  { p with width := p.width / 2 }

/-- Folds the paper from top to bottom -/
def foldTopBottom (p : Paper) : Paper :=
  { p with height := p.height / 2 }

/-- Punches a hole at the given position -/
def punchHole (p : Paper) (pos : Point) : Paper :=
  { p with holes := Hole.mk pos :: p.holes }

/-- Unfolds the paper and reflects the holes -/
def unfold (p : Paper) : Paper :=
  let reflectedHoles := p.holes.bind (λ h => [
    h,
    { position := { x := h.position.x, y := p.height - h.position.y } },
    { position := { x := p.width - h.position.x, y := h.position.y } },
    { position := { x := p.width - h.position.x, y := p.height - h.position.y } }
  ])
  { width := p.width * 2,
    height := p.height * 2,
    holes := reflectedHoles }

theorem folded_paper_holes (initialPaper : Paper) :
  let foldedPaper := foldTopBottom (foldLeftRight initialPaper)
  let punchedPaper := punchHole (punchHole foldedPaper 
    { x := foldedPaper.width * 2 / 3, y := foldedPaper.height / 3 })
    { x := foldedPaper.width / 3, y := foldedPaper.height / 3 }
  let unfoldedPaper := unfold (unfold punchedPaper)
  unfoldedPaper.holes.length = 8 ∧ 
  (∀ h ∈ unfoldedPaper.holes, 
    (h.position.x ∈ [initialPaper.width / 3, initialPaper.width * 2 / 3]) ∧
    (h.position.y ∈ [initialPaper.height / 3, initialPaper.height * 2 / 3])) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_folded_paper_holes_l979_97937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_percentage_dried_grapes_l979_97958

/-- Calculates the percentage of water in dried grapes -/
noncomputable def water_percentage_in_dried_grapes (fresh_grape_weight : ℝ) (fresh_water_percentage : ℝ) (dried_grape_weight : ℝ) : ℝ :=
  let fresh_water_weight := fresh_grape_weight * fresh_water_percentage
  let fresh_dry_matter := fresh_grape_weight - fresh_water_weight
  let dried_water_weight := dried_grape_weight - fresh_dry_matter
  (dried_water_weight / dried_grape_weight) * 100

/-- Theorem stating that the percentage of water in dried grapes is 20% under given conditions -/
theorem water_percentage_dried_grapes :
  water_percentage_in_dried_grapes 10 0.9 1.25 = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_percentage_dried_grapes_l979_97958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l979_97997

open Set Real

-- Define the sets A and B
def A : Set ℝ := {x | |x - 2| < 3}
def B (a : ℝ) : Set ℝ := {x | (2 : ℝ)^x > (2 : ℝ)^a}

-- State the theorem
theorem range_of_a (a : ℝ) : A ∩ B a = A → a ∈ Iic (-1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l979_97997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_is_one_l979_97966

noncomputable def series_term (k : ℕ) : ℝ := (2 ^ (2 ^ k)) / ((4 ^ (2 ^ k)) - 4)

theorem series_sum_is_one :
  (∑' k, series_term k) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_is_one_l979_97966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alloy_composition_l979_97978

theorem alloy_composition (total_mass : ℝ) (x₁ x₂ x₃ x₄ : ℝ) : 
  total_mass = 25 →
  x₁ = 1.5 * x₂ →
  x₂ / x₃ = 3 / 4 →
  x₃ / x₄ = 5 / 6 →
  x₁ + x₂ + x₃ + x₄ = total_mass →
  abs (x₄ - 7.36) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alloy_composition_l979_97978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_fourth_term_a_general_formula_a_2023_l979_97947

/-- Sequence a_n defined by the given conditions -/
def a : ℕ → ℚ
  | 0 => 2  -- Adding a case for 0 to cover all natural numbers
  | 1 => 2
  | 2 => 1/2
  | 3 => 2/7
  | n+4 => 2 / (3*(n+4) - 2)

/-- The relation that holds for the sequence -/
axiom a_relation (n : ℕ) : 1 / a n + 1 / a (n+2) = 2 / a (n+1)

theorem a_fourth_term : a 4 = 1/5 := by
  -- Proof skipped
  sorry

theorem a_general_formula (n : ℕ) (h : n ≥ 4) : a n = 2 / (3*n - 2) := by
  -- Proof skipped
  sorry

-- Additional theorem to show the 2023rd term
theorem a_2023 : a 2023 = 2 / 6067 := by
  -- Proof skipped
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_fourth_term_a_general_formula_a_2023_l979_97947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_volume_approx_l979_97994

/-- The volume of a right rectangular prism with face areas 51, 52, and 53 square units -/
noncomputable def prism_volume : ℝ :=
  let face_area_1 : ℝ := 51
  let face_area_2 : ℝ := 52
  let face_area_3 : ℝ := 53
  Real.sqrt (face_area_1 * face_area_2 * face_area_3)

/-- The theorem stating that the volume of the prism is approximately 374 cubic units -/
theorem prism_volume_approx : ⌊prism_volume⌋₊ = 374 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_volume_approx_l979_97994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_l979_97916

/-- The fixed cost in ten thousand yuan -/
def fixed_cost : ℝ := 12

/-- The production cost per hundred units in ten thousand yuan -/
def production_cost_per_unit : ℝ := 10

/-- The total cost function in ten thousand yuan -/
def P (x : ℝ) : ℝ := fixed_cost + production_cost_per_unit * x

/-- The sales revenue function in ten thousand yuan -/
noncomputable def Q (x : ℝ) : ℝ := 
  if x ≤ 16 then -0.5 * x^2 + 22 * x else 224

/-- The profit function in ten thousand yuan -/
noncomputable def f (x : ℝ) : ℝ := Q x - P x

/-- Theorem stating the maximum profit and the corresponding production quantity -/
theorem max_profit :
  ∃ (max_profit : ℝ) (max_quantity : ℝ),
    max_profit = 60 ∧ 
    max_quantity = 12 ∧
    ∀ x, f x ≤ max_profit ∧
    f max_quantity = max_profit := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_l979_97916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_planes_properties_l979_97985

-- Define the structure for a 3D space
structure Space3D where
  Point : Type
  Vector : Type
  origin : Point
  vec : Point → Point → Vector
  add_vec : Point → Vector → Point
  -- Add more operations as needed

-- Define planes and lines
def Plane (S : Space3D) := S.Point → Prop
def Line (S : Space3D) := S.Point → Prop

-- Define perpendicularity and containment
def perpendicular (S : Space3D) (l : Line S) (p : Plane S) : Prop := sorry

def contained_in (S : Space3D) (l : Line S) (p : Plane S) : Prop := sorry

-- Define the concept of infinitely many lines
def infinitely_many_lines (S : Space3D) (p : Plane S) (prop : Line S → Prop) : Prop := sorry

-- Main theorem
theorem intersecting_planes_properties (S : Space3D) (α β : Plane S) 
  (h_intersect : ∃ (x : S.Point), α x ∧ β x) :
  (∀ (m : Line S), perpendicular S m α → 
    infinitely_many_lines S β (λ l ↦ perpendicular S l m)) ∧
  (∀ (m : Line S), contained_in S m α → 
    ∃ (l : Line S), contained_in S l β ∧ perpendicular S l m) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_planes_properties_l979_97985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_transitivity_counterexample_l979_97981

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between lines and between a line and a plane
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- Define two different lines and a plane
variable (m n : Line)
variable (α : Plane)

-- State that m and n are different
variable (m_neq_n : m ≠ n)

-- Define the theorem
theorem parallel_transitivity_counterexample :
  ¬(∀ (m n : Line) (α : Plane), m ≠ n → parallel_lines m n → parallel_line_plane m α → parallel_line_plane n α) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_transitivity_counterexample_l979_97981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_complex_expression_l979_97951

theorem max_value_complex_expression (z : ℂ) (h : Complex.abs z = 1) :
  Complex.abs (z^3 - 3*z + 2) ≤ 3 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_complex_expression_l979_97951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_plane_intersection_l979_97908

-- Define the basic structures
structure Line where

structure Plane where

-- Define the relationships between lines and planes
def parallel (l : Line) (p : Plane) : Prop := sorry

def skew (l1 l2 : Line) : Prop := sorry

def intersect (l : Line) (p : Plane) : Prop := sorry

def lies_within (l : Line) (p : Plane) : Prop := sorry

-- Define a point type
structure Point where

-- Define membership for points in lines and planes
def Point.mem_line (p : Point) (l : Line) : Prop := sorry

def Point.mem_plane (p : Point) (pl : Plane) : Prop := sorry

-- Define the theorem
theorem line_plane_intersection (a : Line) (α : Plane) 
  (h : ¬ parallel a α) : 
  (¬ ∀ l : Line, lies_within l α → skew l a) ∧ 
  (¬ ¬ ∃ l : Line, lies_within l α ∧ parallel l α) ∧
  (∃ point : Point, point.mem_line a ∧ point.mem_plane α) ∧
  (¬ ∀ l : Line, lies_within l α → intersect l α) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_plane_intersection_l979_97908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l979_97987

theorem trigonometric_equation_solution :
  ∃ α : ℝ, 0 ≤ α ∧ α ≤ Real.pi / 2 ∧
  Real.cos α - 4 * Real.sin α * Real.cos α = Real.sqrt 3 * Real.sin α ∧
  α = Real.pi / 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l979_97987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l979_97909

theorem problem_statement (m n : ℝ) (h1 : (2 : ℝ)^m = 6) (h2 : (3 : ℝ)^n = 6) : 
  (m + n > 4) ∧ (m * n > 4) ∧ ((m - 1)^2 + (n - 1)^2 > 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l979_97909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_point_C_l979_97960

/-- Represents a point on the complex plane -/
def ComplexPoint := ℂ

/-- Represents a square on the complex plane -/
structure ComplexSquare where
  A : ComplexPoint
  B : ComplexPoint
  C : ComplexPoint
  D : ComplexPoint

/-- Check if four points form a square in counterclockwise order -/
def isCounterclockwiseSquare (s : ComplexSquare) : Prop :=
  -- The definition of a counterclockwise square would go here
  sorry

theorem square_point_C (s : ComplexSquare) 
  (h1 : isCounterclockwiseSquare s)
  (h2 : s.A = (1 : ℂ) + 2*Complex.I)
  (h3 : s.B = (3 : ℂ) - 5*Complex.I) : 
  s.C = (10 : ℂ) - 3*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_point_C_l979_97960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_l979_97977

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 4 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x - Real.sin x ^ 4

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- State the theorem
theorem max_area_triangle (t : Triangle) (h1 : f t.A = 1) (h2 : (t.b^2 + t.c^2 + 2*t.b*t.c*Real.cos t.A)/4 = 7) :
  t.b * t.c * Real.sin t.A / 2 ≤ 7 * Real.sqrt 3 / 3 := by
  sorry

-- Note: The proof is omitted as per instructions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_l979_97977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hermione_study_hours_l979_97980

/-- Hermione's utility function -/
noncomputable def U (x y : ℝ) : ℝ := sorry

/-- The number of hours Hermione spends reading spells on Wednesday -/
def h : ℝ := sorry

/-- Hermione's utility is the same on both days -/
axiom utility_equal : U h (10 - h) = U (4 - h) (h + 2)

/-- h is between 0 and 10 inclusive -/
axiom h_bounds : 0 ≤ h ∧ h ≤ 10

theorem hermione_study_hours : h = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hermione_study_hours_l979_97980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_term_is_13_l979_97995

def fibonacci_like : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => fibonacci_like (n + 1) + fibonacci_like n

theorem seventh_term_is_13 : fibonacci_like 6 = 13 := by
  rw [fibonacci_like]
  rw [fibonacci_like]
  rw [fibonacci_like]
  rw [fibonacci_like]
  rw [fibonacci_like]
  rw [fibonacci_like]
  rfl

#eval fibonacci_like 6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_term_is_13_l979_97995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_quadrilateral_l979_97929

-- Define the fractional part function
noncomputable def frac (x : ℝ) : ℝ := x - ⌊x⌋

-- Define the quadrilateral
structure Quadrilateral where
  l : ℕ
  m : ℕ
  n : ℕ
  H : ℕ
  h1 : l > m
  h2 : m > n

-- Define the condition for the side lengths
def satisfies_condition (q : Quadrilateral) : Prop :=
  frac (3^q.l / 10000) = frac (3^q.m / 10000) ∧
  frac (3^q.m / 10000) = frac (3^q.n / 10000)

-- Define the perimeter of the quadrilateral
def perimeter (q : Quadrilateral) : ℕ := q.l + q.m + q.n + q.H

-- State the theorem
theorem min_perimeter_quadrilateral :
  ∀ q : Quadrilateral, satisfies_condition q →
  ∃ q_min : Quadrilateral, satisfies_condition q_min ∧
  perimeter q_min = 2004 ∧
  ∀ q' : Quadrilateral, satisfies_condition q' →
  perimeter q' ≥ perimeter q_min :=
by
  sorry

#check min_perimeter_quadrilateral

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_quadrilateral_l979_97929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l979_97957

/-- The function f(x) defined as sin(ωx) - √3 cos(ωx) -/
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x) - Real.sqrt 3 * Real.cos (ω * x)

/-- Theorem stating that given the conditions, ω must equal 1/3 -/
theorem omega_value (ω : ℝ) (x₁ x₂ : ℝ) (h_pos : ω > 0) 
  (h_f_x₁ : f ω x₁ = 2) (h_f_x₂ : f ω x₂ = 0) 
  (h_min_diff : ∀ y z : ℝ, f ω y = 2 → f ω z = 0 → |y - z| ≥ 3 * Real.pi / 2) :
  ω = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l979_97957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_ellipse_area_l979_97949

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  center : Point
  semiMajorAxis : ℝ
  semiMinorAxis : ℝ

/-- Checks if a point lies on the ellipse -/
def pointOnEllipse (e : Ellipse) (p : Point) : Prop :=
  (p.x - e.center.x)^2 / e.semiMajorAxis^2 + (p.y - e.center.y)^2 / e.semiMinorAxis^2 = 1

/-- Calculates the area of an ellipse -/
noncomputable def ellipseArea (e : Ellipse) : ℝ :=
  Real.pi * e.semiMajorAxis * e.semiMinorAxis

/-- Theorem: The area of the specific ellipse is 50π -/
theorem specific_ellipse_area :
  ∃ (e : Ellipse),
    e.center = Point.mk 2 3 ∧
    e.semiMajorAxis = 10 ∧
    e.semiMinorAxis = 5 ∧
    pointOnEllipse e (Point.mk (-8) 3) ∧
    pointOnEllipse e (Point.mk 12 3) ∧
    pointOnEllipse e (Point.mk 10 6) ∧
    ellipseArea e = 50 * Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_ellipse_area_l979_97949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l979_97996

-- Define the quadratic function f(x)
def f (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

-- Define the function g(x)
noncomputable def g (a : ℝ) (f : ℝ → ℝ) (x : ℝ) : ℝ := f (a^x)

-- Theorem statement
theorem quadratic_function_properties 
  (b c : ℝ) 
  (h1 : ∀ x, f b c x = f b c (2 - x)) 
  (h2 : f b c 3 = 0) 
  (h3 : ∃ a > 1, ∀ x ∈ Set.Icc (-1) 1, g a (f b c) x ≤ 5 ∧ ∃ x₀ ∈ Set.Icc (-1) 1, g a (f b c) x₀ = 5) :
  (∀ x, f b c x = x^2 - 2*x - 3) ∧ 
  (∃ a > 1, ∀ x ∈ Set.Icc (-1) 1, g a (f b c) x ≤ 5 ∧ ∃ x₀ ∈ Set.Icc (-1) 1, g a (f b c) x₀ = 5 ∧ a = 4) :=
by
  sorry

#check quadratic_function_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l979_97996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_solution_range_l979_97928

noncomputable def f (a : ℝ) (x : ℝ) := (2 - x) * Real.exp x - a * x - a

theorem f_solution_range (a : ℝ) :
  (∃ x y : ℕ+, x ≠ y ∧ 
    (∀ z : ℕ+, f a z > 0 ↔ (z = x ∨ z = y))) →
  a ∈ Set.Ioc (-1/4 * Real.exp 3) 0 :=
by
  sorry

#check f_solution_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_solution_range_l979_97928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_abc_value_l979_97940

theorem max_abc_value (a b c : ℕ) : 
  a ∈ Finset.range 11 → 
  b ∈ Finset.range 11 → 
  c ∈ Finset.range 11 → 
  a ≠ b → b ≠ c → a ≠ c → 
  (∀ x y z : ℕ, x ∈ Finset.range 11 → 
                y ∈ Finset.range 11 → 
                z ∈ Finset.range 11 → 
                x ≠ y → y ≠ z → x ≠ z → 
                x * y * z ≤ a * b * c) → 
  a * b * c = 990 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_abc_value_l979_97940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_values_for_50_l979_97974

/-- Represents the set of possible final values after the number elimination process -/
def FinalValues (n : ℕ) : Set ℕ :=
  {k : ℕ | k ≤ n ∧ k % 2 = 0}

/-- The elimination process preserves the property that the final value is in FinalValues -/
axiom elimination_preserves_final_values (n : ℕ) :
  ∀ a b : ℕ, a ≤ n → b ≤ n → (max a b - min a b) ∈ FinalValues n

/-- The main theorem: for n = 50, the final values are even numbers from 0 to 50 -/
theorem final_values_for_50 :
  FinalValues 50 = {k : ℕ | k ≤ 50 ∧ k % 2 = 0} := by
  sorry

#check final_values_for_50

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_values_for_50_l979_97974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_4_eq_5_point_5_l979_97963

noncomputable def f (x : ℝ) : ℝ := 4 / (3 - x)

noncomputable def f_inv (x : ℝ) : ℝ := (3 * x - 4) / x

noncomputable def g (x : ℝ) : ℝ := 1 / (f_inv x) + 5

theorem g_of_4_eq_5_point_5 : g 4 = 5.5 := by
  -- Expand the definition of g
  unfold g
  -- Expand the definition of f_inv
  unfold f_inv
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_4_eq_5_point_5_l979_97963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_DEF_l979_97944

-- Define the square PQRS
def PQRS_area : ℚ := 64

-- Define the side length of the smaller squares
def small_square_side : ℚ := 2

-- Define the triangle DEF
structure Triangle_DEF where
  DE : ℚ
  DF : ℚ
  EF : ℚ

-- Define the properties of the triangle
def isosceles_DEF (t : Triangle_DEF) : Prop := t.DE = t.DF

-- Define the midpoint T of side PQ
def T_midpoint (side_length : ℚ) : ℚ := side_length / 2

-- Define the folding property
def D_coincides_with_T (t : Triangle_DEF) (side_length : ℚ) : Prop :=
  t.DE = T_midpoint side_length

-- Theorem to prove
theorem area_of_DEF (t : Triangle_DEF) 
  (h1 : isosceles_DEF t) 
  (h2 : D_coincides_with_T t (PQRS_area.sqrt)) : 
  (1/2) * t.EF * (T_midpoint (PQRS_area.sqrt) - small_square_side) = 4 := by
  sorry

#check area_of_DEF

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_DEF_l979_97944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chi_squared_test_reject_null_hypothesis_batch_not_accepted_l979_97939

/-- Given parameters for the chi-squared test --/
structure ChiSquaredTestParams where
  n : ℕ                  -- sample size
  s_squared : ℝ          -- sample variance
  sigma_squared : ℝ      -- hypothesized population variance
  alpha : ℝ              -- significance level

/-- Calculate the observed chi-squared statistic --/
noncomputable def observed_chi_squared (params : ChiSquaredTestParams) : ℝ :=
  (params.n - 1 : ℝ) * params.s_squared / params.sigma_squared

/-- Calculate the critical chi-squared value using Wilson-Hilferty approximation --/
noncomputable def critical_chi_squared (params : ChiSquaredTestParams) : ℝ :=
  let k : ℝ := params.n - 1
  let z_alpha : ℝ := 2.326  -- Approximation for 99th percentile of standard normal distribution
  k * (1 - 2 / (9 * k) + z_alpha * Real.sqrt (2 / (9 * k))) ^ 3

/-- Predicate to represent that the variance does not significantly exceed the hypothesized value --/
def variance_not_significantly_exceeds (params : ChiSquaredTestParams) : Prop :=
  observed_chi_squared params ≤ critical_chi_squared params

/-- Theorem stating that the observed chi-squared statistic exceeds the critical value --/
theorem chi_squared_test_reject_null_hypothesis (params : ChiSquaredTestParams)
    (h_n : params.n = 121)
    (h_s_squared : params.s_squared = 0.3)
    (h_sigma_squared : params.sigma_squared = 0.2)
    (h_alpha : params.alpha = 0.01) :
    observed_chi_squared params > critical_chi_squared params := by
  sorry

/-- Corollary stating that the batch cannot be accepted --/
theorem batch_not_accepted (params : ChiSquaredTestParams)
    (h_n : params.n = 121)
    (h_s_squared : params.s_squared = 0.3)
    (h_sigma_squared : params.sigma_squared = 0.2)
    (h_alpha : params.alpha = 0.01) :
    ¬(variance_not_significantly_exceeds params) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chi_squared_test_reject_null_hypothesis_batch_not_accepted_l979_97939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_line_l979_97968

-- Define the points and their properties
def P : ℝ × ℝ := (0, -1)
def A (a : ℝ) : ℝ × ℝ := (a, 0)
def B (b : ℝ) : ℝ × ℝ := (0, b)
def M (a b : ℝ) : ℝ × ℝ := (-a, 2*b)

-- Define the vector operations
def vec (p q : ℝ × ℝ) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)
def dot (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the conditions
axiom cond1 (a b : ℝ) : vec (A a) (M a b) = (2 * (vec (A a) (B b)).1, 2 * (vec (A a) (B b)).2)
axiom cond2 (a b : ℝ) : dot (vec P (A a)) (vec (A a) (M a b)) = 0

-- Define the trajectory C and line l
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 2 * p.1^2}
def l (m : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = m * p.1 + (1/2)}

-- State the theorem
theorem trajectory_and_line :
  (∀ a b, M a b ∈ C) ∧
  (∃ m : ℝ, m = Real.sqrt 2/2 ∨ m = -Real.sqrt 2/2) ∧
  (∀ m, m = Real.sqrt 2/2 ∨ m = -Real.sqrt 2/2 →
    ∃ Q R : ℝ × ℝ,
      Q ∈ C ∧ R ∈ C ∧ Q ∈ l m ∧ R ∈ l m ∧
      (Q.1 - R.1)^2 + (Q.2 - R.2)^2 = 4 * (Q.1^2 + Q.2^2) ∧
      Q.1 * R.1 + Q.2 * R.2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_line_l979_97968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_axes_product_l979_97926

/-- Represents an ellipse with center P, major axis XY, minor axis WZ, and focus G. -/
structure Ellipse where
  P : ℝ × ℝ  -- Center
  X : ℝ × ℝ  -- One end of major axis
  Y : ℝ × ℝ  -- Other end of major axis
  W : ℝ × ℝ  -- One end of minor axis
  Z : ℝ × ℝ  -- Other end of minor axis
  G : ℝ × ℝ  -- Focus

/-- The distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The theorem about the product of axes of an ellipse -/
theorem ellipse_axes_product (e : Ellipse) 
    (h1 : distance e.P e.G = 9)
    (h2 : distance e.P e.W + distance e.P e.G - distance e.W e.G = 6) :
  (distance e.X e.Y) * (distance e.W e.Z) = 720 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_axes_product_l979_97926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_parameter_l979_97901

/-- Given two perpendicular lines l₁ and l₂, where l₁ is vertical and l₂ has a specific form,
    prove that the parameter a in their equations must be 2. -/
theorem perpendicular_lines_parameter (a : ℝ) : 
  (∃ l₁ l₂ : Set (ℝ × ℝ), 
    (∀ x y, (x, y) ∈ l₁ ↔ a * x + 1 = 0) ∧ 
    (∀ x y, (x, y) ∈ l₂ ↔ (a - 2) * x + y + a = 0) ∧
    a ≠ 0 ∧
    (∀ p q : ℝ × ℝ, p ∈ l₁ → q ∈ l₂ → (p.1 - q.1) * (p.2 - q.2) = 0)) →
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_parameter_l979_97901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_sin_c_l979_97972

theorem right_triangle_sin_c (A B C : ℝ) :
  -- Triangle ABC is a right triangle
  Real.sin B = 1 →
  -- Given sin A
  Real.sin A = 3/5 →
  -- Prove sin C
  Real.sin C = 4/5 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_sin_c_l979_97972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bead_arrangement_probability_l979_97912

/-- The number of red beads -/
def red_beads : ℕ := 4

/-- The number of white beads -/
def white_beads : ℕ := 3

/-- The number of blue beads -/
def blue_beads : ℕ := 2

/-- The total number of beads -/
def total_beads : ℕ := red_beads + white_beads + blue_beads

/-- The probability of arranging the beads with no two neighboring beads of the same color -/
def prob_no_adjacent_same_color : ℚ := 5 / 63

theorem bead_arrangement_probability :
  (∃ (valid_arrangements : ℕ) (total_arrangements : ℕ),
    (valid_arrangements : ℚ) / (total_arrangements : ℚ) = prob_no_adjacent_same_color) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bead_arrangement_probability_l979_97912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_x_value_l979_97971

noncomputable def f (x : ℝ) : ℝ := ((5*x - 20)/(4*x - 5))^2 + (5*x - 20)/(4*x - 5)

theorem largest_x_value :
  ∃ (max_x : ℝ), (f max_x = 20) ∧ 
  (∀ (x : ℝ), f x = 20 → x ≤ max_x) ∧
  (max_x = 9/5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_x_value_l979_97971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_x_intercept_l979_97921

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  focus1 : Point
  focus2 : Point

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: Given an ellipse with specific foci and one x-intercept, prove the other x-intercept -/
theorem ellipse_x_intercept (e : Ellipse) (x_intercept : Point) :
  e.focus1 = ⟨0, 2⟩ →
  e.focus2 = ⟨6, 0⟩ →
  x_intercept = ⟨0, 0⟩ →
  ∃ (other_x_intercept : Point),
    other_x_intercept.y = 0 ∧
    distance x_intercept e.focus1 + distance x_intercept e.focus2 =
      distance other_x_intercept e.focus1 + distance other_x_intercept e.focus2 ∧
    other_x_intercept.x = 48 / 7 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_x_intercept_l979_97921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_numbers_l979_97986

-- Define the range of numbers in the bag
def bag : Set ℕ := {n | 1 ≤ n ∧ n ≤ 30}

-- Define a composite number
def isComposite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

-- Define Chris's and Dana's numbers
variable (chris_number : ℕ)
variable (dana_number : ℕ)

-- Conditions
axiom chris_in_bag : chris_number ∈ bag
axiom dana_in_bag : dana_number ∈ bag
axiom different_numbers : chris_number ≠ dana_number
axiom chris_uncertainty : ∀ n ∈ bag, n ≠ chris_number → ¬(chris_number > n)
axiom dana_certainty : ∀ n ∈ bag, n ≠ dana_number → dana_number > n
axiom dana_composite : isComposite dana_number
axiom perfect_cube : ∃ m : ℕ, 50 * dana_number + chris_number = m^3

-- Theorem to prove
theorem sum_of_numbers : chris_number + dana_number = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_numbers_l979_97986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_S_to_T_l979_97938

-- Define the set 𝒯
def T : Set (ℝ × ℝ × ℝ) :=
  {p | p.1 ≥ 0 ∧ p.2.1 ≥ 0 ∧ p.2.2 ≥ 0 ∧ p.1 + p.2.1 + p.2.2 = 1}

-- Define the support relation
def supports (x y z a b c : ℝ) : Prop :=
  (x ≥ a ∧ y ≥ b) ∨ (x ≥ a ∧ z ≥ c) ∨ (y ≥ b ∧ z ≥ c)

-- Define the set 𝒮
def S : Set (ℝ × ℝ × ℝ) :=
  {p ∈ T | supports p.1 p.2.1 p.2.2 (1/4) (1/4) (1/4)}

-- State the theorem
theorem area_ratio_S_to_T : 
  (MeasureTheory.volume S) / (MeasureTheory.volume T) = 9/32 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_S_to_T_l979_97938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_of_f_l979_97924

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 15*x^2 - 33*x + 6

-- State the theorem
theorem monotonic_decreasing_interval_of_f :
  StrictMonoOn f (Set.Ioo (-1 : ℝ) 11) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_of_f_l979_97924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_mixture_zero_percent_acid_l979_97918

/-- Represents a mixture of acid and water -/
structure Mixture where
  acid : ℝ
  water : ℝ

/-- The percentage of acid in a mixture -/
noncomputable def acid_percentage (m : Mixture) : ℝ :=
  m.acid / (m.acid + m.water) * 100

/-- Adds acid to a mixture -/
def add_acid (m : Mixture) (amount : ℝ) : Mixture :=
  { acid := m.acid + amount, water := m.water }

theorem original_mixture_zero_percent_acid (m : Mixture) :
  acid_percentage (add_acid m 1) = 25 →
  acid_percentage (add_acid (add_acid m 1) 1) = 40 →
  acid_percentage m = 0 := by
  sorry

#check original_mixture_zero_percent_acid

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_mixture_zero_percent_acid_l979_97918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_s4_value_l979_97970

/-- s(n) is a function that returns the n-digit number formed by attaching
    the first n perfect squares in order. -/
def s (n : ℕ) : ℕ := sorry

/-- Examples of s(n) for specific values of n -/
axiom s1 : s 1 = 1
axiom s2 : s 2 = 14
axiom s3 : s 3 = 149
axiom s5 : s 5 = 1491625

/-- The number of digits in s(99) -/
axiom s99_digits : (Nat.digits 10 (s 99)).length = 355

/-- Theorem: s(4) is equal to 14916 -/
theorem s4_value : s 4 = 14916 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_s4_value_l979_97970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_lower_bound_inequality_holds_l979_97903

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.exp x
noncomputable def g (x : ℝ) : ℝ := Real.sin x + Real.cos x

-- Statement 1: For all real x, e^x ≥ x + 1
theorem exp_lower_bound (x : ℝ) : f x ≥ x + 1 := by sorry

-- Statement 2: For all real x ≥ 0 and real a ≤ 2, e^x + sin x + cos x - 2 - ax ≥ 0
theorem inequality_holds (x a : ℝ) (h1 : x ≥ 0) (h2 : a ≤ 2) : 
  f x + g x - 2 - a * x ≥ 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_lower_bound_inequality_holds_l979_97903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_values_satisfying_inequality_l979_97993

theorem least_values_satisfying_inequality (k m p : ℕ) : 
  (m > 0) →
  (0 ≤ p) →
  (p ≤ 9) →
  ((0.0000101 * (p : ℝ) + 0.0000101) * (10 : ℝ)^(k + m) > 10^6) →
  (∀ k' m' p' : ℕ, 
    (m' > 0) →
    (0 ≤ p') →
    (p' ≤ 9) →
    ((0.0000101 * (p' : ℝ) + 0.0000101) * (10 : ℝ)^(k' + m') > 10^6) →
    (k ≤ k' ∧ m ≤ m' ∧ p ≤ p')) →
  (k = 8 ∧ m = 1 ∧ p = 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_values_satisfying_inequality_l979_97993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_history_or_statistics_count_l979_97984

/-- Represents the number of students taking a subject or combination of subjects. -/
structure StudentCount where
  count : ℕ

instance : OfNat StudentCount n where
  ofNat := ⟨n⟩

instance : HAdd StudentCount StudentCount StudentCount where
  hAdd a b := ⟨a.count + b.count⟩

instance : HSub StudentCount StudentCount StudentCount where
  hSub a b := ⟨a.count - b.count⟩

theorem history_or_statistics_count
  (total : StudentCount)
  (history : StudentCount)
  (statistics : StudentCount)
  (history_not_statistics : StudentCount)
  (h_total : total = 90)
  (h_history : history = 36)
  (h_statistics : statistics = 30)
  (h_history_not_statistics : history_not_statistics = 29) :
  ∃ (history_or_statistics : StudentCount),
    history_or_statistics = history + statistics - (history - history_not_statistics) ∧
    history_or_statistics = 59 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_history_or_statistics_count_l979_97984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grammar_club_committee_probability_l979_97931

def total_members : ℕ := 24
def boys : ℕ := 12
def girls : ℕ := 12
def committee_size : ℕ := 5

theorem grammar_club_committee_probability :
  (Nat.choose total_members committee_size - 
   (Nat.choose boys committee_size + Nat.choose girls committee_size) : ℚ) / 
  Nat.choose total_members committee_size = 455 / 472 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grammar_club_committee_probability_l979_97931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_intersection_l979_97943

noncomputable section

-- Define the ellipse parameters
def a : ℝ := 2
def b : ℝ := Real.sqrt 2

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the line equation
def line (x y m : ℝ) : Prop := y = x + m

-- Define the right triangle condition
def is_right_triangle (A B Q : ℝ × ℝ) : Prop :=
  (A.1 - Q.1)^2 + (A.2 - Q.2)^2 + (B.1 - Q.1)^2 + (B.2 - Q.2)^2 = (A.1 - B.1)^2 + (A.2 - B.2)^2

theorem ellipse_and_line_intersection :
  (∀ x y, ellipse x y ↔ x^2 / 4 + y^2 / 2 = 1) ∧
  (∃ A B Q m,
    ellipse A.1 A.2 ∧
    ellipse B.1 B.2 ∧
    line A.1 A.2 m ∧
    line B.1 B.2 m ∧
    Q.1 = 0 ∧
    is_right_triangle A B Q →
    m = 3 * Real.sqrt 10 / 5 ∨ m = -3 * Real.sqrt 10 / 5) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_intersection_l979_97943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_focus_l979_97933

/-- A parabola with equation y^2 = 8x -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  eq_def : equation = fun x y => y^2 = 8*x

/-- A point on the parabola -/
structure PointOnParabola (p : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : p.equation x y

/-- The focus of the parabola -/
def focus : ℝ × ℝ := (2, 0)

/-- The distance between two points in ℝ² -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem distance_to_focus (p : Parabola) (A : PointOnParabola p) 
    (h : A.x = 4) : distance (A.x, A.y) focus = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_focus_l979_97933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_properties_l979_97973

/-- The curve C_n: x^2 - 2nx + y^2 = 0 (n = 1, 2, ...) -/
def C_n (n : ℕ) (x y : ℝ) : Prop := x^2 - 2*n*x + y^2 = 0

/-- The point P(-1, 0) -/
def P : ℝ × ℝ := (-1, 0)

/-- The slope of the tangent line l_n -/
noncomputable def k_n (n : ℕ) : ℝ := n / Real.sqrt (2*n + 1)

/-- The x-coordinate of the point of tangency P_n -/
noncomputable def x_n (n : ℕ) : ℝ := n / (n + 1)

/-- The y-coordinate of the point of tangency P_n -/
noncomputable def y_n (n : ℕ) : ℝ := n * Real.sqrt (2*n + 1) / (n + 1)

theorem tangent_line_properties :
  (∀ n : ℕ, k_n n > 0) ∧
  (k_n 2 = 2 * Real.sqrt 5 / 5) ∧
  (∀ n : ℕ, C_n n (x_n n) (y_n n)) ∧
  (∀ n : ℕ, Real.sqrt ((1 - x_n n) / (1 + x_n n)) < Real.sqrt 2 * Real.sin ((x_n n) / (y_n n))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_properties_l979_97973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_one_l979_97954

-- Define the function f(x)
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 2 * Real.log x - m / x

-- State the theorem
theorem tangent_slope_at_one (m : ℝ) :
  (deriv (f m)) 1 = 3 → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_one_l979_97954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_group_sample_is_44_l979_97917

/-- Represents a systematic sampling scenario -/
structure SystematicSampling where
  total_samples : ℕ
  num_groups : ℕ
  first_group_sample : ℕ
  target_group : ℕ
  h_positive : 0 < total_samples
  h_groups : 0 < num_groups
  h_divide : num_groups ∣ total_samples
  h_first_sample : first_group_sample ≤ total_samples / num_groups
  h_target : target_group ≤ num_groups

/-- Calculates the sample number for a given group in systematic sampling -/
def sample_number (s : SystematicSampling) : ℕ :=
  s.first_group_sample + (s.total_samples / s.num_groups) * (s.target_group - 1)

/-- Theorem: In the given systematic sampling scenario, the sample number for the fifth group is 44 -/
theorem fifth_group_sample_is_44 (s : SystematicSampling) 
  (h_total : s.total_samples = 81)
  (h_groups : s.num_groups = 9)
  (h_first : s.first_group_sample = 8)
  (h_target : s.target_group = 5) : 
  sample_number s = 44 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_group_sample_is_44_l979_97917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_5a_plus_2b_l979_97998

theorem cube_root_of_5a_plus_2b (a b : ℝ) : 
  (∃ k : ℤ, (2*a - 1) = (3 * k)^2) → 
  ((3*a + b).sqrt = 4) → 
  ∃ x : ℝ, x^3 = 5*a + 2*b ∧ x = 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_5a_plus_2b_l979_97998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_second_x_intercept_l979_97914

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  focus1 : Point
  focus2 : Point
  x_intercept : Point

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: The given ellipse configuration does not yield a second distinct x-intercept -/
theorem no_second_x_intercept (e : Ellipse) 
    (h1 : e.focus1 = ⟨0, 3⟩) 
    (h2 : e.focus2 = ⟨4, 0⟩) 
    (h3 : e.x_intercept = ⟨4, 0⟩) : 
  ¬∃ (p : Point), p.y = 0 ∧ p ≠ e.x_intercept ∧ 
    distance p e.focus1 + distance p e.focus2 = 
    distance e.x_intercept e.focus1 + distance e.x_intercept e.focus2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_second_x_intercept_l979_97914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l979_97988

open Real

-- Define the function f(x) = x - ln(x) - 2
noncomputable def f (x : ℝ) : ℝ := x - log x - 2

-- State the theorem
theorem root_in_interval : 
  ∃ x : ℝ, x ∈ Set.Ioo 3 4 ∧ f x = 0 ∧ 
  ∀ k : ℕ, k ≠ 3 → ¬∃ y : ℝ, y ∈ Set.Ioo (k : ℝ) ((k : ℝ) + 1) ∧ f y = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l979_97988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dreams_ratio_l979_97923

theorem dreams_ratio (dreams_per_day : ℕ) (days_per_year : ℕ) (total_dreams : ℕ) :
  dreams_per_day = 4 →
  days_per_year = 365 →
  total_dreams = 4380 →
  (total_dreams - dreams_per_day * days_per_year) / (dreams_per_day * days_per_year) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dreams_ratio_l979_97923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shyam_money_l979_97965

/-- Given the ratios of money between individuals and Ram's amount, calculate Shyam's amount --/
theorem shyam_money (ram_gopal_ratio : ℚ) (gopal_krishan_ratio : ℚ) (krishan_shyam_ratio : ℚ) 
  (ram_amount : ℕ) (h1 : ram_gopal_ratio = 7/17) (h2 : gopal_krishan_ratio = 7/17) 
  (h3 : krishan_shyam_ratio = 11/13) (h4 : ram_amount = 735) : ℕ :=
by
  sorry

#eval 2119 -- The expected result for Shyam's money

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shyam_money_l979_97965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_difference_example_l979_97941

/-- The difference in average speeds between two travelers -/
noncomputable def speed_difference (distance : ℝ) (time1 time2 : ℝ) : ℝ :=
  distance * (1 / time2 - 1 / time1)

/-- Theorem stating the difference in average speeds -/
theorem speed_difference_example :
  speed_difference 8 (40 / 60) (15 / 60) = 20 := by
  -- Unfold the definition of speed_difference
  unfold speed_difference
  -- Simplify the expression
  simp [mul_sub, mul_div]
  -- Perform numerical calculations
  norm_num
  -- Close the proof
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_difference_example_l979_97941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l979_97967

/-- Given a hyperbola with equation x²/b² - y²/a² = -1, if one of its asymptotes
    passes through the point (2,1), then its eccentricity is √5 -/
theorem hyperbola_eccentricity (a b : ℝ) (h : a > 0) (k : b > 0) :
  (∃ (x y : ℝ), x^2/b^2 - y^2/a^2 = -1) →
  (∃ (m : ℝ), m = a/b ∧ 1 = m * 2) →
  let c := Real.sqrt (a^2 + b^2)
  c/a = Real.sqrt 5 := by
  sorry

#check hyperbola_eccentricity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l979_97967
