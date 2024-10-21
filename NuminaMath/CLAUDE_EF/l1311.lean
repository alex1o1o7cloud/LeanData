import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_difference_approx_l1311_131147

noncomputable def regular_pentagon_area (side_length : ℝ) : ℝ := 
  (1/4) * Real.sqrt (5 * (5 + 2 * Real.sqrt 5)) * side_length^2

noncomputable def equilateral_triangle_area (side_length : ℝ) : ℝ := 
  (Real.sqrt 3 / 4) * side_length^2

noncomputable def regular_decagon_area (side_length : ℝ) : ℝ := 
  5 * side_length^2 * Real.sin (3 * Real.pi / 5)

noncomputable def region_R_area : ℝ := 
  regular_pentagon_area 1 + 15 * equilateral_triangle_area 1

noncomputable def region_S_area : ℝ := regular_decagon_area 2

theorem area_difference_approx : 
  ∃ ε > 0, abs (region_S_area - region_R_area - 3.16) < ε := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_difference_approx_l1311_131147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l1311_131100

/-- The inclination angle of a line with direction vector (-1, √3) is 120° -/
theorem line_inclination_angle : 
  ∀ (v : ℝ × ℝ), v = (-1, Real.sqrt 3) → 
  Real.arctan (v.2 / v.1) = 120 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l1311_131100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_product_equality_l1311_131116

/-- Triangle with incenter and excenter -/
structure TriangleWithCenters where
  A : ℝ × ℝ  -- Vertex A of the triangle
  B : ℝ × ℝ  -- Vertex B of the triangle
  C : ℝ × ℝ  -- Vertex C of the triangle
  O₃ : ℝ × ℝ  -- Incenter
  O' : ℝ × ℝ  -- Excenter opposite to vertex A

/-- Distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- Theorem: Product of distances from a vertex to incenter and excenter
    equals product of sides meeting at that vertex -/
theorem distance_product_equality (t : TriangleWithCenters) :
  distance t.C t.O₃ * distance t.C t.O' = distance t.A t.C * distance t.B t.C := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_product_equality_l1311_131116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_constant_l1311_131150

noncomputable section

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2 + 1

-- Define a point on the parabola
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : y = parabola x

-- Define a chord of the parabola
structure Chord where
  A : PointOnParabola
  B : PointOnParabola

-- Define the function s
noncomputable def s (chord : Chord) (C : ℝ × ℝ) : ℝ :=
  1 / ((chord.A.x - C.1)^2 + (chord.A.y - C.2)^2) +
  1 / ((chord.B.x - C.1)^2 + (chord.B.y - C.2)^2)

-- The theorem to prove
theorem parabola_chord_constant :
  ∃ d : ℝ, ∀ (chord : Chord), 
    (∃ m : ℝ, chord.A.y = m * chord.A.x + d ∧ chord.B.y = m * chord.B.x + d) →
    s chord (0, d) = 4/9 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_constant_l1311_131150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_max_min_values_l1311_131131

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 3 + Real.log x / Real.log 2

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := f (x^2) - (f x)^2

-- State the theorem
theorem g_max_min_values :
  ∀ x : ℝ, 1 ≤ x ∧ x ≤ 4 →
  (∀ y : ℝ, 1 ≤ y ∧ y ≤ 4 → g y ≤ -6) ∧
  (∃ z : ℝ, 1 ≤ z ∧ z ≤ 4 ∧ g z = -6) ∧
  (∀ y : ℝ, 1 ≤ y ∧ y ≤ 4 → g y ≥ -11) ∧
  (∃ w : ℝ, 1 ≤ w ∧ w ≤ 4 ∧ g w = -11) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_max_min_values_l1311_131131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l1311_131125

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Definition of the triangle ABC with given conditions -/
noncomputable def triangleABC : Triangle where
  a := 4
  b := Real.sqrt 5 -- placeholder value
  c := Real.sqrt 5 -- placeholder value
  A := 2 * Real.pi / 3
  B := Real.pi / 6 -- placeholder value
  C := Real.pi / 6 -- placeholder value

/-- D is the midpoint of BC -/
noncomputable def D (t : Triangle) : ℝ × ℝ := sorry

/-- Length of AD -/
noncomputable def lengthAD (t : Triangle) : ℝ := sorry

/-- Perimeter of a triangle -/
noncomputable def perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

/-- Theorem: The perimeter of triangle ABC is 4 + 2√5 -/
theorem triangle_perimeter :
  lengthAD triangleABC = Real.sqrt 2 →
  perimeter triangleABC = 4 + 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l1311_131125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_length_l1311_131165

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the length of a side
noncomputable def side_length (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Define the angle bisector
noncomputable def angle_bisector (t : Triangle) (a b : ℝ) : ℝ × ℝ :=
  sorry  -- The actual computation of the angle bisector is omitted

-- Theorem statement
theorem angle_bisector_length (t : Triangle) (a b : ℝ) (h1 : side_length t.B t.C = a) (h2 : side_length t.C t.A = b) :
  let D := angle_bisector t a b
  side_length t.C D = (2 * a * b * Real.cos ((Real.arccos ((a^2 + b^2 - (side_length t.A t.B)^2) / (2 * a * b))) / 2)) / (a + b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_length_l1311_131165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_bounds_l1311_131130

/-- The circle C₁ -/
def C₁ (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 2

/-- The line C₂ -/
def C₂ (x y : ℝ) : Prop := x + y + 2 = 0

/-- Distance from a point (x, y) to the line C₂ -/
noncomputable def dist_to_C₂ (x y : ℝ) : ℝ := |x + y + 2| / Real.sqrt 2

theorem distance_bounds :
  ∀ x y : ℝ, C₁ x y →
    (∃ x' y' : ℝ, C₁ x' y' ∧ dist_to_C₂ x' y' = 3 * Real.sqrt 2) ∧
    (∃ x' y' : ℝ, C₁ x' y' ∧ dist_to_C₂ x' y' = Real.sqrt 2) ∧
    (∀ x' y' : ℝ, C₁ x' y' → Real.sqrt 2 ≤ dist_to_C₂ x' y' ∧ dist_to_C₂ x' y' ≤ 3 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_bounds_l1311_131130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_exists_l1311_131189

/-- The sequence of natural numbers starting from 11 -/
def basicSequence : ℕ → ℕ
  | n => n + 10

/-- The given sequence as a function -/
def givenSequence : ℕ → ℕ
  | 1 => 111
  | 2 => 213
  | 3 => 141
  | 4 => 516
  | 5 => 171
  | 6 => 819
  | 7 => 202
  | 8 => 122
  | _ => 0  -- undefined for n > 8

/-- The transformation function -/
def f : ℕ → ℕ := sorry

/-- Theorem stating the existence of a transformation function -/
theorem transformation_exists :
  ∃ f : ℕ → ℕ, ∀ n : ℕ, n ≤ 8 → f (basicSequence n) = givenSequence n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_exists_l1311_131189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oliver_cherries_l1311_131118

def initial_cherries : ℕ → Prop := sorry
def remaining_cherries : ℕ → ℕ → Prop := sorry
def difference_cherries : ℕ → ℕ → Prop := sorry

theorem oliver_cherries (x : ℕ) :
  initial_cherries x →
  remaining_cherries x 6 →
  difference_cherries x 10 →
  x = 16 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oliver_cherries_l1311_131118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_of_quadratic_polynomial_l1311_131139

/-- A quadratic polynomial with real coefficients -/
def quadratic_polynomial (a b c : ℝ) : ℂ → ℂ := λ x ↦ a * x^2 + b * x + c

theorem root_of_quadratic_polynomial (a b c : ℝ) :
  a = 3 →
  b = -6 →
  (quadratic_polynomial a b c) (-1 - 2*Complex.I) = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_of_quadratic_polynomial_l1311_131139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_park_connections_eq_catalan_l1311_131178

/-- The number of ways to connect 2n gates in a circular park with n non-intersecting paths -/
def park_connections (n : ℕ) : ℚ :=
  (1 : ℚ) / (n + 1 : ℚ) * (Nat.choose (2 * n) n)

/-- Theorem stating that the number of ways to connect 2n gates in a circular park
    with n non-intersecting paths is equal to the nth Catalan number -/
theorem park_connections_eq_catalan (n : ℕ) :
  park_connections n = (1 : ℚ) / (n + 1 : ℚ) * (Nat.choose (2 * n) n) := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_park_connections_eq_catalan_l1311_131178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_measure_l1311_131179

noncomputable section

-- Define the radii of the circles
def r₁ : ℝ := 4
def r₂ : ℝ := 3
def r₃ : ℝ := 2

-- Define the ratio of shaded to unshaded area
def shade_ratio : ℝ := 5 / 8

-- Theorem statement
theorem angle_measure (θ : ℝ) 
  (h_shade : r₁^2 * θ + r₂^2 * (Real.pi - θ) + r₃^2 * θ = 
             shade_ratio * (r₁^2 * (Real.pi - θ) + r₂^2 * θ + r₃^2 * (Real.pi - θ))) :
  θ = 28 * Real.pi / 143 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_measure_l1311_131179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_amplitude_increases_period_l1311_131187

/-- Represents a simple pendulum -/
structure SimplePendulum where
  length : ℝ
  gravity : ℝ
  amplitude : ℝ

/-- The period of a simple pendulum for small angles -/
noncomputable def small_angle_period (p : SimplePendulum) : ℝ :=
  2 * Real.pi * Real.sqrt (p.length / p.gravity)

/-- States that the period is independent of mass -/
axiom period_mass_independent (p : SimplePendulum) (m1 m2 : ℝ) :
  small_angle_period p = small_angle_period p

/-- States that for larger amplitudes, the small angle approximation is not valid -/
axiom large_amplitude_approximation (θ : ℝ) :
  θ > 0 → abs (Real.sin θ - θ) > 0

/-- Theorem: Increasing the amplitude of a simple pendulum increases its period -/
theorem increasing_amplitude_increases_period (p : SimplePendulum) :
  ∀ ε > 0, ∃ δ > 0, ∀ p' : SimplePendulum,
    p'.length = p.length ∧ 
    p'.gravity = p.gravity ∧ 
    p'.amplitude > p.amplitude ∧ 
    p'.amplitude < p.amplitude + δ →
    small_angle_period p' > small_angle_period p + ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_amplitude_increases_period_l1311_131187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_correct_l1311_131105

/-- A right circular cone with two congruent spheres inside --/
structure ConeWithSpheres where
  base_radius : ℝ
  height : ℝ
  sphere_radius : ℝ
  is_right_circular : Bool
  spheres_tangent : Bool

/-- The specific cone described in the problem --/
noncomputable def problem_cone : ConeWithSpheres where
  base_radius := 4
  height := 10
  sphere_radius := 4 * Real.sqrt 29 / 7
  is_right_circular := true
  spheres_tangent := true

/-- Theorem stating that the sphere radius in the problem_cone is correct --/
theorem sphere_radius_correct (c : ConeWithSpheres) : 
  c = problem_cone → 
  c.sphere_radius = 4 * Real.sqrt 29 / 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_correct_l1311_131105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_of_curve_l1311_131160

noncomputable def f (x : ℝ) := 1 - Real.log (x^2 - 1)

theorem arc_length_of_curve (a b : ℝ) (ha : a = 3) (hb : b = 4) :
  ∫ x in a..b, Real.sqrt (1 + ((- 2 * x) / (x^2 - 1))^2) = 1 + 2 * Real.log (6/5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_of_curve_l1311_131160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_equation_is_general_solution_l1311_131137

-- Define the differential equation
def differential_equation (y : ℝ → ℝ) : Prop :=
  ∀ x, (deriv (deriv y)) x + 4 * y x = Real.sin (2 * x)

-- Define the general solution
noncomputable def general_solution (C₁ C₂ : ℝ) (x : ℝ) : ℝ :=
  C₁ * Real.cos (2 * x) + C₂ * Real.sin (2 * x) - (1/4) * x * Real.cos (2 * x)

-- Theorem statement
theorem solution_satisfies_equation :
  ∀ C₁ C₂ : ℝ, differential_equation (general_solution C₁ C₂) :=
by
  sorry

-- Theorem statement for general solution
theorem is_general_solution :
  ∀ y : ℝ → ℝ, differential_equation y →
  ∃ C₁ C₂ : ℝ, ∀ x : ℝ, y x = general_solution C₁ C₂ x :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_equation_is_general_solution_l1311_131137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_antonella_coin_value_l1311_131196

/-- Represents the types of Canadian coins in Antonella's purse -/
inductive Coin where
  | Loonie
  | Toonie
deriving BEq, Repr

/-- The value of a coin in dollars -/
def coin_value (c : Coin) : ℚ :=
  match c with
  | Coin.Loonie => 1
  | Coin.Toonie => 2

theorem antonella_coin_value :
  ∀ (coins : List Coin),
    coins.length = 10 →
    coins.count Coin.Toonie = 4 →
    (coins.map coin_value).sum - 3 = 11 →
    ∃ (other_coin : Coin), other_coin ≠ Coin.Toonie ∧ coin_value other_coin = 1 := by
  sorry

#eval coin_value Coin.Loonie
#eval coin_value Coin.Toonie

end NUMINAMATH_CALUDE_ERRORFEEDBACK_antonella_coin_value_l1311_131196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conditional_probability_problem_l1311_131193

-- Define the probabilities as real numbers between 0 and 1
noncomputable def prob_first : ℝ := 7/8
noncomputable def prob_second : ℝ := 6/8
noncomputable def prob_both : ℝ := 5/8

-- Define the conditional probability
noncomputable def conditional_prob : ℝ := prob_both / prob_first

-- Theorem statement
theorem conditional_probability_problem :
  0 ≤ prob_first ∧ prob_first ≤ 1 ∧
  0 ≤ prob_second ∧ prob_second ≤ 1 ∧
  0 ≤ prob_both ∧ prob_both ≤ 1 ∧
  prob_both ≤ min prob_first prob_second →
  conditional_prob = 5/7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_conditional_probability_problem_l1311_131193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_distance_for_given_fare_l1311_131154

/-- Represents the taxi fare structure and calculates the total fare for a given distance -/
noncomputable def taxiFare (distance : ℝ) : ℝ :=
  let baseFare := 8
  let midRate := 2.15
  let highRate := 2.85
  let fuelSurcharge := 1
  if distance ≤ 3 then
    baseFare + fuelSurcharge
  else if distance ≤ 8 then
    baseFare + midRate * (distance - 3) + fuelSurcharge
  else
    baseFare + midRate * 5 + highRate * (distance - 8) + fuelSurcharge

/-- Theorem stating that a fare of 31.15 yuan corresponds to a travel distance of 11.98 km -/
theorem taxi_distance_for_given_fare :
  ∃ (distance : ℝ), taxiFare distance = 31.15 ∧ distance = 11.98 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_distance_for_given_fare_l1311_131154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_chord_l1311_131176

/-- The circle equation -/
def circleEq (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 3 = 0

/-- The line equation -/
def lineEq (x y b : ℝ) : Prop := y = x + b

/-- The chord length is maximized when b = -1 -/
theorem longest_chord (b : ℝ) : 
  (∀ x y, circleEq x y → lineEq x y b → 
    ∀ b' x' y', circleEq x' y' → lineEq x' y' b' → 
      (x - x')^2 + (y - y')^2 ≤ (2*2)) → 
  b = -1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_chord_l1311_131176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1311_131198

theorem triangle_side_length (A B C : ℝ) (AB BC AC : ℝ) :
  Real.cos (3 * A - B) + Real.sin (A + B) = 1 →
  AB = 5 →
  BC = (5 * Real.sqrt 2) / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1311_131198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_150_degrees_l1311_131197

theorem cos_150_degrees : Real.cos (150 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_150_degrees_l1311_131197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_diagonal_length_l1311_131143

/-- Represents an ellipse in 2D space -/
structure Ellipse where
  center : ℝ × ℝ
  semi_major_axis : ℝ
  semi_minor_axis : ℝ

/-- Represents a rectangle in 2D space -/
structure Rectangle where
  center : ℝ × ℝ
  width : ℝ
  height : ℝ

/-- Checks if a rectangle is circumscribed around an ellipse -/
def isCircumscribed (r : Rectangle) (e : Ellipse) : Prop :=
  -- Definition of circumscription (simplified for this context)
  r.center = e.center ∧ r.width ≥ 2 * e.semi_major_axis ∧ r.height ≥ 2 * e.semi_minor_axis

/-- Calculates the length of the diagonal of a rectangle -/
noncomputable def diagonalLength (r : Rectangle) : ℝ :=
  Real.sqrt (r.width ^ 2 + r.height ^ 2)

/-- Theorem: The diagonal length of any rectangle circumscribed around a given ellipse is constant -/
theorem constant_diagonal_length (e : Ellipse) :
  ∀ r1 r2 : Rectangle, isCircumscribed r1 e → isCircumscribed r2 e →
  diagonalLength r1 = diagonalLength r2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_diagonal_length_l1311_131143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rubles_equation_solutions_l1311_131119

theorem rubles_equation_solutions (x y n : ℤ) : 
  (n * (x - 3) = y + 3 ∧ x + n = 3 * (y - n)) → 
  (n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 7) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rubles_equation_solutions_l1311_131119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_and_minimum_l1311_131190

/-- The ratio of coefficients of x³ and x⁴ in ((ax + 2b)⁶) is 4:3 -/
noncomputable def ratio_condition (a b : ℝ) : Prop :=
  (160 * a^3 * b^3) / (60 * a^4 * b^2) = 4/3

/-- F(a, b) = (b³ + 16) / a -/
noncomputable def F (a b : ℝ) : ℝ := (b^3 + 16) / a

theorem expansion_and_minimum (a b : ℝ) (ha : a > 0) (hb : b ≠ 0) 
  (h_ratio : ratio_condition a b) :
  (∃ c : ℝ, c = 20 ∧ 
    (∀ k, k ≠ 3 → Nat.choose 6 k * (2*b)^(6-k) ≤ c)) ∧ 
  (∀ x y, x > 0 → y ≠ 0 → F x y ≥ 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_and_minimum_l1311_131190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_team_right_handed_players_l1311_131157

theorem cricket_team_right_handed_players
  (total_players throwers right_handed : ℕ)
  (h1 : total_players = 120)
  (h2 : throwers = 70)
  (h3 : throwers ≤ total_players)
  (h4 : (total_players - throwers) % 5 = 0)
  (h5 : throwers ≤ right_handed) :
  right_handed = 90 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_team_right_handed_players_l1311_131157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_to_line_l1311_131199

-- Define the curve C
noncomputable def curve_C (θ : Real) : Real × Real :=
  (2 * Real.cos θ, Real.sin θ)

-- Define the line l
noncomputable def line_l (t : Real) : Real × Real :=
  (5 - 2 * Real.sqrt 3 * t, t)

-- Define the distance function between two points
noncomputable def distance (p q : Real × Real) : Real :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem min_distance_curve_to_line :
  ∃ (d : Real), d = Real.sqrt 13 / 13 ∧
  ∀ (θ t : Real), θ ∈ Set.Icc 0 Real.pi →
  distance (curve_C θ) (line_l t) ≥ d := by
  sorry

#check min_distance_curve_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_to_line_l1311_131199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_a_minus_pi_4_eq_half_implies_sin_2a_minus_cos_sq_a_eq_half_l1311_131186

theorem tan_a_minus_pi_4_eq_half_implies_sin_2a_minus_cos_sq_a_eq_half (a : ℝ) :
  Real.tan (a - Real.pi/4) = 1/2 → Real.sin (2*a) - (Real.cos a)^2 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_a_minus_pi_4_eq_half_implies_sin_2a_minus_cos_sq_a_eq_half_l1311_131186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_sum_converges_to_rational_l1311_131138

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib n + fib (n + 1)

-- Define our infinite sum
noncomputable def infiniteSum : ℝ := ∑' n, (fib n : ℝ) / 10^n

-- State the theorem
theorem infinite_sum_converges_to_rational : 
  infiniteSum = 10 / 89 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_sum_converges_to_rational_l1311_131138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_water_ratio_after_adding_water_l1311_131106

/-- Represents a mixture of milk and water -/
structure Mixture where
  total_volume : ℝ
  milk_ratio : ℝ
  water_ratio : ℝ

/-- Calculates the new ratio of milk to water after adding water to a mixture -/
noncomputable def new_ratio (m : Mixture) (added_water : ℝ) : ℝ × ℝ :=
  let initial_milk := m.total_volume * (m.milk_ratio / (m.milk_ratio + m.water_ratio))
  let initial_water := m.total_volume * (m.water_ratio / (m.milk_ratio + m.water_ratio))
  let new_water := initial_water + added_water
  (initial_milk, new_water)

theorem milk_water_ratio_after_adding_water
  (m : Mixture)
  (h1 : m.total_volume = 45)
  (h2 : m.milk_ratio = 4)
  (h3 : m.water_ratio = 1)
  (h4 : new_ratio m 9 = (36, 18)) :
  (2 : ℝ) / 1 = 36 / 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_water_ratio_after_adding_water_l1311_131106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_proper_and_largest_improper_fraction_l1311_131168

def number_set : Set ℕ := {2, 3, 4, 5}

def is_proper_fraction (n m : ℕ) : Prop := n < m

def is_improper_fraction (n m : ℕ) : Prop := n > m

theorem smallest_proper_and_largest_improper_fraction :
  (∀ a b, a ∈ number_set → b ∈ number_set → is_proper_fraction a b → (2 : ℚ) / 5 ≤ (a : ℚ) / b) ∧
  (∀ a b, a ∈ number_set → b ∈ number_set → is_improper_fraction a b → (a : ℚ) / b ≤ (5 : ℚ) / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_proper_and_largest_improper_fraction_l1311_131168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_condition_l1311_131169

/-- A point in the Cartesian coordinate plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Definition of an ellipse -/
def is_ellipse (P : Point → Prop) (F₁ F₂ : Point) (a : ℝ) : Prop :=
  ∀ p, P p ↔ distance p F₁ + distance p F₂ = 2*a ∧ 2*a > distance F₁ F₂

/-- The main theorem -/
theorem ellipse_condition (F₁ F₂ : Point) (a : ℝ) (ha : a > 0) :
  ¬(∀ P : Point → Prop, (∀ p, P p → distance p F₁ + distance p F₂ = 2*a) →
    is_ellipse P F₁ F₂ a) ∧
  (∀ P : Point → Prop, is_ellipse P F₁ F₂ a →
    ∀ p, P p → distance p F₁ + distance p F₂ = 2*a) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_condition_l1311_131169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_values_l1311_131127

noncomputable def f (a b x : ℝ) : ℝ := Real.log (abs ((2 * Real.exp 1) / (Real.exp 1 * x - 1) + a)) + b

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_values (a b : ℝ) :
  is_odd (f a b) → a = Real.exp 1 ∧ b = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_values_l1311_131127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_group_size_is_14_l1311_131140

/-- The number of men in the first group that can complete a piece of work in 25 days, 
    given that 20 men can complete the same work in 17.5 days. -/
def first_group_size : ℚ :=
  let days_first_group : ℚ := 25
  let days_second_group : ℚ := 35/2
  let size_second_group : ℚ := 20
  size_second_group * days_second_group / days_first_group

theorem first_group_size_is_14 : ⌊first_group_size⌋ = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_group_size_is_14_l1311_131140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_three_divisors_l1311_131117

theorem remainder_three_divisors : 
  {n : ℕ | n > 3 ∧ 73 % n = 3} = {5, 7, 10, 14, 35, 70} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_three_divisors_l1311_131117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identities_for_point_l1311_131122

theorem trig_identities_for_point (θ : Real) :
  let x : Real := -Real.sqrt 3
  let y : Real := Real.sqrt 6
  let r : Real := Real.sqrt (x^2 + y^2)
  (Real.sin θ = y / r) ∧ 
  (Real.cos θ = x / r) ∧ 
  (Real.tan θ = y / x) ∧
  (Real.sin (2*θ) = 2 * Real.sin θ * Real.cos θ) ∧
  (Real.cos (2*θ) = Real.cos θ^2 - Real.sin θ^2) ∧
  (Real.tan (2*θ) = Real.sin (2*θ) / Real.cos (2*θ)) →
  (Real.sin θ = Real.sqrt 6 / 3) ∧ 
  (Real.cos θ = -(Real.sqrt 3) / 3) ∧ 
  (Real.tan θ = -(Real.sqrt 2)) ∧
  (Real.sin (2*θ) = -(2 * Real.sqrt 2) / 3) ∧ 
  (Real.cos (2*θ) = -1 / 3) ∧ 
  (Real.tan (2*θ) = 2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identities_for_point_l1311_131122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_on_lined_paper_l1311_131161

/-- A regular n-gon can be placed on lined paper with all vertices on the lines. -/
def placeable_on_lined_paper (n : ℕ) : Prop :=
  ∃ (placement : ℝ × ℝ → ℝ × ℝ) (center : ℝ × ℝ) (radius : ℝ),
    ∀ i : Fin n, ∃ k : ℤ, 
      placement (center.1 + radius * Real.cos (2 * Real.pi * i / n), 
                 center.2 + radius * Real.sin (2 * Real.pi * i / n)) = (0, ↑k)

/-- The lines on the paper are parallel and evenly spaced. -/
axiom lines_parallel_and_evenly_spaced : 
  ∀ k : ℤ, ∃ line : ℝ × ℝ → Prop, 
    ∀ x y : ℝ, line (x, ↑k) ∧ (y = ↑k → line (x, y))

/-- Theorem: A regular n-gon can be placed on lined paper with all vertices on the lines 
    if and only if n is 3, 4, or 6. -/
theorem regular_polygon_on_lined_paper (n : ℕ) : 
  placeable_on_lined_paper n ↔ n = 3 ∨ n = 4 ∨ n = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_on_lined_paper_l1311_131161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_m_values_l1311_131159

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

/-- The equation of the ellipse -/
def is_ellipse (x y m : ℝ) : Prop :=
  x^2 / (10 - m) + y^2 / (m - 2) = 1

/-- The theorem stating the possible values of m -/
theorem ellipse_m_values :
  ∀ m : ℝ, 
    (∃ x y : ℝ, is_ellipse x y m) ∧ 
    (∃ a b : ℝ, a > b ∧ b > 0 ∧ eccentricity a b = 4) →
    m = 4 ∨ m = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_m_values_l1311_131159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_sqrt_3_l1311_131195

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the foci of a hyperbola -/
noncomputable def foci (h : Hyperbola) : Point × Point :=
  let c := Real.sqrt (h.a^2 + h.b^2)
  ⟨Point.mk (-c) 0, Point.mk c 0⟩

/-- Checks if two points are symmetric about the origin -/
def symmetric_about_origin (p q : Point) : Prop :=
  p.x = -q.x ∧ p.y = -q.y

/-- Checks if a point lies on the asymptote of a hyperbola -/
def on_asymptote (h : Hyperbola) (p : Point) : Prop :=
  p.y = (h.b / h.a) * p.x

/-- Calculates the area of a rectangle given its vertices -/
noncomputable def rectangle_area (p q r s : Point) : ℝ :=
  let width := |p.x - q.x|
  let height := |p.y - r.y|
  width * height

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + (h.b / h.a)^2)

theorem hyperbola_eccentricity_sqrt_3 
  (h : Hyperbola) (m n : Point) 
  (h_symmetric : symmetric_about_origin m n)
  (h_on_asymptote_m : on_asymptote h m)
  (h_on_asymptote_n : on_asymptote h n)
  (h_area : rectangle_area (foci h).1 (foci h).2 m n = 2 * Real.sqrt 6 * h.a^2) :
  eccentricity h = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_sqrt_3_l1311_131195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_range_theorem_l1311_131151

/-- The profit function for a production process -/
noncomputable def profit_function (x : ℝ) (t : ℝ) : ℝ := 100 * t * (5 * x + 1 - 3 / x)

/-- The theorem stating the range of x for which the profit is at least 3000 yuan after 2 hours -/
theorem profit_range_theorem (x : ℝ) (h1 : 1 ≤ x) (h2 : x ≤ 10) :
  profit_function x 2 ≥ 3000 ↔ 3 ≤ x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_range_theorem_l1311_131151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_diagonal_l1311_131135

-- Define the quadrilateral ABCD with center O
structure Quadrilateral :=
  (A B C D O : ℝ × ℝ)

-- Define the lengths of the sides and diagonals
noncomputable def side_lengths (q : Quadrilateral) : ℝ × ℝ × ℝ × ℝ × ℝ × ℝ := 
  let AB := ((q.A.1 - q.B.1)^2 + (q.A.2 - q.B.2)^2).sqrt
  let BC := ((q.B.1 - q.C.1)^2 + (q.B.2 - q.C.2)^2).sqrt
  let CD := ((q.C.1 - q.D.1)^2 + (q.C.2 - q.D.2)^2).sqrt
  let DA := ((q.D.1 - q.A.1)^2 + (q.D.2 - q.A.2)^2).sqrt
  let AC := ((q.A.1 - q.C.1)^2 + (q.A.2 - q.C.2)^2).sqrt
  let BD := ((q.B.1 - q.D.1)^2 + (q.B.2 - q.D.2)^2).sqrt
  (AB, BC, CD, DA, AC, BD)

-- Theorem statement
theorem quadrilateral_diagonal (q : Quadrilateral) : 
  let (AB, BC, CD, DA, AC, BD) := side_lengths q
  AB = 5 ∧ BC = 6 ∧ CD = 5 ∧ DA = 7 ∧ BD = 9 → AC = (271 : ℝ).sqrt := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_diagonal_l1311_131135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_triangle_distance_is_2_sqrt_5_l1311_131102

/-- The distance from the center of a sphere to the plane of a triangle -/
noncomputable def sphere_to_triangle_plane_distance (R : ℝ) (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  let K := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let r := K / s
  Real.sqrt (R^2 - r^2)

/-- Theorem stating the distance for the given sphere and triangle -/
theorem sphere_triangle_distance_is_2_sqrt_5 :
  sphere_to_triangle_plane_distance 6 15 15 24 = 2 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_triangle_distance_is_2_sqrt_5_l1311_131102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_intersection_bisection_when_parallel_l1311_131141

-- Define the line l: y = x + b
def line_l (b : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1 + b}

-- Define the ellipse C: x²/25 + y²/9 = 1
def ellipse_C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2/25 + p.2^2/9 = 1}

-- Define the condition that line l does not intersect ellipse C
def no_intersection (b : ℝ) : Prop := ∀ p, p ∈ line_l b → p ∉ ellipse_C

-- Define a point P on line l
def point_on_line (P : ℝ × ℝ) (b : ℝ) : Prop := P ∈ line_l b

-- Define tangent lines PM and PN
def tangent_lines (P M N : ℝ × ℝ) : Prop := 
  M ∈ ellipse_C ∧ N ∈ ellipse_C ∧ 
  (∀ p, p ∈ Set.Icc P M → p ∉ ellipse_C) ∧
  (∀ p, p ∈ Set.Icc P N → p ∉ ellipse_C)

-- Define the fixed point Q
noncomputable def fixed_point (b : ℝ) : ℝ × ℝ := (-25/b, 9/b)

-- Main theorem
theorem tangent_lines_intersection (b : ℝ) (P M N : ℝ × ℝ) :
  no_intersection b →
  point_on_line P b →
  tangent_lines P M N →
  ∃ Q : ℝ × ℝ, Q = fixed_point b ∧ Q ∈ Set.Icc M N :=
sorry

-- Theorem for bisection when MN is parallel to l
theorem bisection_when_parallel (b : ℝ) (P M N : ℝ × ℝ) :
  no_intersection b →
  point_on_line P b →
  tangent_lines P M N →
  (M.2 - N.2) / (M.1 - N.1) = 1 →  -- MN parallel to l
  let Q := fixed_point b
  Q.1 = (M.1 + N.1) / 2 ∧ Q.2 = (M.2 + N.2) / 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_intersection_bisection_when_parallel_l1311_131141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_21_equals_neg_one_l1311_131164

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log 2 / (x - 1)

-- State the theorem
theorem f_21_equals_neg_one :
  (∀ x > 0, f ((2 / x) + 1) = Real.log x) →
  f 21 = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_21_equals_neg_one_l1311_131164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_intersection_and_area_l1311_131146

/-- The curve C in polar coordinates -/
noncomputable def C (ρ θ : ℝ) : Prop := ρ * (Real.sin θ)^2 = 4 * Real.cos θ

/-- The line l₁ -/
def l₁ (θ : ℝ) : Prop := θ = Real.pi / 3

/-- The line l₂ -/
noncomputable def l₂ (ρ θ : ℝ) : Prop := ρ * Real.sin θ = 4 * Real.sqrt 3

/-- Point A is on curve C and line l₁, and is not the pole -/
def A (ρ θ : ℝ) : Prop := C ρ θ ∧ l₁ θ ∧ ρ ≠ 0

/-- Point B is on curve C and line l₂ -/
def B (ρ θ : ℝ) : Prop := C ρ θ ∧ l₂ ρ θ

/-- The area of a triangle given two sides and the included angle -/
noncomputable def triangleArea (a b γ : ℝ) : ℝ := (1 / 2) * a * b * Real.sin γ

theorem curve_intersection_and_area :
  ∃ (ρA θA ρB θB : ℝ),
    A ρA θA ∧ 
    B ρB θB ∧
    ρA = 8 / 3 ∧ 
    θA = Real.pi / 3 ∧
    ρB = 8 * Real.sqrt 3 ∧ 
    θB = Real.pi / 6 ∧
    triangleArea ρA ρB (θA - θB) = 16 * Real.sqrt 3 / 3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_intersection_and_area_l1311_131146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l1311_131155

-- Define the hyperbola C₁
def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the circle C₂
def circle_eq (a x y : ℝ) : Prop := x^2 + y^2 - 2*a*x + 3/4 * a^2 = 0

-- Define the asymptotes of C₁
def asymptote (a b x y : ℝ) : Prop := b*x + a*y = 0 ∨ b*x - a*y = 0

-- Define the eccentricity of a hyperbola
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2) / a

-- Main theorem
theorem hyperbola_eccentricity_range (a b : ℝ) :
  a > 0 ∧ b > 0 →
  (∃ x₁ y₁ x₂ y₂ : ℝ, x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧ 
    asymptote a b x₁ y₁ ∧ asymptote a b x₂ y₂ ∧
    circle_eq a x₁ y₁ ∧ circle_eq a x₂ y₂) →
  1 < eccentricity a b ∧ eccentricity a b < 2 * Real.sqrt 3 / 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l1311_131155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l1311_131172

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  -- Sum of angles is π
  angle_sum : A + B + C = Real.pi
  -- Side lengths are positive
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0
  -- Law of sines
  law_of_sines : a / Real.sin A = b / Real.sin B
  law_of_sines' : b / Real.sin B = c / Real.sin C

-- Define the theorem
theorem triangle_property (t : Triangle) 
  (h : 2 * t.a * Real.cos t.C + t.c = 2 * t.b) :
  t.A = Real.pi / 3 ∧ 
  (t.a = 1 → 2 < t.a + t.b + t.c ∧ t.a + t.b + t.c ≤ 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l1311_131172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cupcake_celebration_proof_l1311_131181

/-- Calculates the number of batches required for a cupcake celebration --/
def cupcake_batches_calculation (total_students : ℕ) (cupcakes_per_student : ℕ) 
  (cupcakes_per_batch : ℕ) (attendance_rate : ℚ) : ℕ :=
  let attending_students := (total_students : ℚ) * attendance_rate
  let total_cupcakes_needed := attending_students * (cupcakes_per_student : ℚ)
  let batches_needed := total_cupcakes_needed / (cupcakes_per_batch : ℚ)
  Int.ceil batches_needed |>.toNat

/-- Proves that 18 batches are required for the cupcake celebration --/
theorem cupcake_celebration_proof :
  cupcake_batches_calculation 150 3 20 (4/5) = 18 := by
  unfold cupcake_batches_calculation
  simp
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cupcake_celebration_proof_l1311_131181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_polar_points_l1311_131180

/-- The Euclidean distance between two points in polar coordinates -/
noncomputable def polar_distance (r₁ r₂ : ℝ) (θ₁ θ₂ : ℝ) : ℝ :=
  Real.sqrt (r₁^2 + r₂^2 - 2 * r₁ * r₂ * Real.cos (θ₁ - θ₂))

/-- Theorem: The distance between A(4, θ₁) and B(10, θ₂) in polar coordinates,
    where θ₁ - θ₂ = 2π/3, is 2√19 -/
theorem distance_between_polar_points :
  ∀ (θ₁ θ₂ : ℝ), θ₁ - θ₂ = 2 * Real.pi / 3 →
  polar_distance 4 10 θ₁ θ₂ = 2 * Real.sqrt 19 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_polar_points_l1311_131180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maximal_ellipse_in_M_l1311_131114

-- Define the region M
def M (p : ℝ) : Set (ℝ × ℝ) :=
  {xy : ℝ × ℝ | xy.1^2 - p^2 ≤ 2*p*xy.2 ∧ 2*p*xy.2 ≤ p^2 - xy.1^2}

-- State that M is convex
axiom M_is_convex (p : ℝ) : Convex ℝ (M p)

-- State that M is symmetric with respect to x-axis, y-axis, and origin
axiom M_symmetric_x (p : ℝ) : ∀ x y, (x, y) ∈ M p → (x, -y) ∈ M p
axiom M_symmetric_y (p : ℝ) : ∀ x y, (x, y) ∈ M p → (-x, y) ∈ M p
axiom M_symmetric_origin (p : ℝ) : ∀ x y, (x, y) ∈ M p → (-x, -y) ∈ M p

-- Define an ellipse
def Ellipse (a b : ℝ) : Set (ℝ × ℝ) :=
  {xy : ℝ × ℝ | xy.1^2 / a^2 + xy.2^2 / b^2 = 1}

-- Define the area of an ellipse
noncomputable def EllipseArea (a b : ℝ) : ℝ := Real.pi * a * b

-- Theorem: The ellipse with maximal area inscribed in M has the specified semi-axes and area
theorem maximal_ellipse_in_M (p : ℝ) (h : p > 0) :
  ∃ (a b : ℝ),
    (∀ (x y : ℝ), (x, y) ∈ Ellipse a b → (x, y) ∈ M p) ∧
    (∀ (a' b' : ℝ), (∀ (x y : ℝ), (x, y) ∈ Ellipse a' b' → (x, y) ∈ M p) →
      EllipseArea a' b' ≤ EllipseArea a b) ∧
    a = Real.sqrt (2/3) * p ∧
    b = Real.sqrt 2 / 3 * p ∧
    EllipseArea a b = 2 * Real.pi / (3 * Real.sqrt 3) * p^2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_maximal_ellipse_in_M_l1311_131114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_categorization_l1311_131126

-- Define the given numbers
noncomputable def numbers : List ℝ := [20/7, 0, -10, Real.pi, -2/3, 0.3, 1/2]

-- Define the sets
def positive_numbers : Set ℝ := {x | x > 0}
def integers : Set ℝ := {x | ∃ n : ℤ, x = n}
def fractions : Set ℝ := {x | ∃ a b : ℤ, b ≠ 0 ∧ x = a / b}
def non_negative_rationals : Set ℝ := {x | x ≥ 0 ∧ ∃ a b : ℤ, b ≠ 0 ∧ x = a / b}

-- Theorem statement
theorem correct_categorization :
  {20/7, Real.pi, 0.3, 1/2} ⊆ positive_numbers ∧
  {0, -10} ⊆ integers ∧
  {20/7, -2/3, 0.3, 1/2} ⊆ fractions ∧
  {20/7, 0, 0.3, 1/2} ⊆ non_negative_rationals :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_categorization_l1311_131126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measure_l1311_131133

open Real

theorem triangle_angle_measure (a b c A B C : ℝ) :
  (0 < A) → (A < π) → (0 < B) → (B < π) → (0 < C) → (C < π) →
  A + B + C = π →
  a / sin A = b / sin B →
  b / sin B = c / sin C →
  (Real.sqrt 2 * a - c) / b = cos C / cos B →
  B = π / 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measure_l1311_131133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_danny_wrappers_collection_l1311_131120

-- Define the types for bottle caps and wrappers
def BottleCaps := ℕ
def Wrappers := ℕ

-- Define the given conditions
def found_bottle_caps : ℕ := 22
def found_wrappers : ℕ := 30
def current_bottle_caps : ℕ := 17

-- Theorem stating that Danny has at least 30 wrappers in his collection now
theorem danny_wrappers_collection (initial_wrappers : ℕ) : 
  initial_wrappers + found_wrappers ≥ found_wrappers := by
  apply Nat.le_add_left

#check danny_wrappers_collection

end NUMINAMATH_CALUDE_ERRORFEEDBACK_danny_wrappers_collection_l1311_131120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sound_pressure_comparison_l1311_131107

-- Define the sound pressure level function
noncomputable def sound_pressure_level (p p₀ : ℝ) : ℝ := 20 * Real.log (p / p₀) / Real.log 10

-- Define the theorem
theorem sound_pressure_comparison 
  (p₀ p₁ p₂ p₃ : ℝ) 
  (h₀ : p₀ > 0)
  (h₁ : 60 ≤ sound_pressure_level p₁ p₀ ∧ sound_pressure_level p₁ p₀ ≤ 90)
  (h₂ : 50 ≤ sound_pressure_level p₂ p₀ ∧ sound_pressure_level p₂ p₀ ≤ 60)
  (h₃ : sound_pressure_level p₃ p₀ = 40) :
  p₁ ≥ p₂ ∧ p₃ = 100 * p₀ ∧ p₁ ≤ 100 * p₂ :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sound_pressure_comparison_l1311_131107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l1311_131136

-- Define the function f(x) as noncomputable due to the use of Real.sqrt
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x - 2 * Real.cos x

-- State the theorem
theorem f_max_value :
  ∃ (M : ℝ), (∀ x, f x ≤ M) ∧ (∃ x, f x = M) ∧ M = 4 := by
  -- We'll use 'sorry' to skip the proof for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l1311_131136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_partition_l1311_131174

/-- The set S of integers from 1 to 2022 -/
def S : Set Nat := Finset.range 2022

/-- A partition of S into n disjoint subsets -/
def Partition (n : Nat) : Type := 
  { p : Fin n → Set Nat // 
    (∀ i, p i ⊆ S) ∧ 
    (∀ i j, i ≠ j → p i ∩ p j = ∅) ∧
    (⋃ i, p i) = S }

/-- Condition (a): For all x, y in the subset, x ≠ y implies gcd(x, y) > 1 -/
def ConditionA (s : Set Nat) : Prop :=
  ∀ x y, x ∈ s → y ∈ s → x ≠ y → Nat.gcd x y > 1

/-- Condition (b): For all x, y in the subset, x ≠ y implies gcd(x, y) = 1 -/
def ConditionB (s : Set Nat) : Prop :=
  ∀ x y, x ∈ s → y ∈ s → x ≠ y → Nat.gcd x y = 1

/-- A valid partition satisfies either condition A or B for each subset -/
def ValidPartition (n : Nat) (p : Partition n) : Prop :=
  ∀ i, (ConditionA (p.val i) ∧ ¬ConditionB (p.val i)) ∨
       (ConditionB (p.val i) ∧ ¬ConditionA (p.val i))

/-- The main theorem: The smallest n for a valid partition is 14 -/
theorem smallest_valid_partition : 
  (∃ p : Partition 14, ValidPartition 14 p) ∧
  (∀ n < 14, ¬ ∃ p : Partition n, ValidPartition n p) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_partition_l1311_131174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_origin_l1311_131171

/-- The distance between two points in a 2D Cartesian coordinate system -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- Theorem: The distance between (-6, 8) and (0, 0) is 10 -/
theorem distance_to_origin : distance (-6) 8 0 0 = 10 := by
  -- Unfold the definition of distance
  unfold distance
  -- Simplify the expression
  simp
  -- The proof is complete, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_origin_l1311_131171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_specific_l1311_131183

theorem sin_double_angle_specific (α : Real) 
  (h1 : Real.cos α = 3/5) 
  (h2 : α ∈ Set.Ioo (-Real.pi/2) 0) : 
  Real.sin (2 * α) = -24/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_specific_l1311_131183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_shape_is_graph_l1311_131191

-- Define a function
def Function := ℝ → ℝ

-- Define a point in the Cartesian coordinate system
def CartesianPoint := ℝ × ℝ

-- Define the set of points representing a function
def FunctionPoints (f : Function) : Set CartesianPoint :=
  {p : CartesianPoint | ∃ x : ℝ, p = (x, f x)}

-- The theorem to prove
theorem function_shape_is_graph (f : Function) :
  ∃ shape : Set CartesianPoint, shape = FunctionPoints f ∧ 
  (∀ p : CartesianPoint, p ∈ shape ↔ ∃ x : ℝ, p = (x, f x)) := by
  exists FunctionPoints f
  constructor
  · rfl
  · intro p
    simp [FunctionPoints]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_shape_is_graph_l1311_131191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_total_score_probability_l1311_131121

-- Define the dartboard
noncomputable def outer_radius : ℝ := 8
noncomputable def inner_radius : ℝ := 5

-- Define the point values
def inner_points : List ℕ := [3, 4, 4]
def outer_points : List ℕ := [2, 3, 3]

-- Define the number of regions
def num_regions : ℕ := 3

-- Function to calculate area
noncomputable def circle_area (r : ℝ) : ℝ := Real.pi * r^2

-- Calculate probabilities
noncomputable def inner_prob : ℝ := circle_area inner_radius / circle_area outer_radius
noncomputable def outer_prob : ℝ := 1 - inner_prob

-- Function to double outer region scores
def double_outer_score (x : ℕ) : ℕ := 2 * x

-- Theorem statement
theorem odd_total_score_probability : 
  let inner_areas := List.replicate num_regions (inner_prob / num_regions)
  let outer_areas := List.replicate num_regions (outer_prob / num_regions)
  let inner_scores := inner_points
  let outer_scores := List.map double_outer_score outer_points
  let total_prob := (List.sum (List.map (fun p ↦ p * (1 - p)) (inner_areas ++ outer_areas))) * 2
  total_prob = 65 / 256 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_total_score_probability_l1311_131121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_cone_height_equality_l1311_131115

/-- Represents a cylinder with radius and height -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Represents a cone with radius and height -/
structure Cone where
  radius : ℝ
  height : ℝ

/-- Volume of a cylinder -/
noncomputable def cylinderVolume (c : Cylinder) : ℝ := Real.pi * c.radius^2 * c.height

/-- Volume of a cone -/
noncomputable def coneVolume (c : Cone) : ℝ := (1/3) * Real.pi * c.radius^2 * c.height

theorem cylinder_cone_height_equality (cyl : Cylinder) (con : Cone) 
    (h_radius : cyl.radius = con.radius)
    (h_volume_ratio : cylinderVolume cyl = 3 * coneVolume con) :
    cyl.height = con.height := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_cone_height_equality_l1311_131115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_proposition_l1311_131132

-- Define proposition p
def p (x : ℝ) : Prop := Real.sin x + 4 / Real.sin x ≥ 4

-- Define proposition q
def q (a : ℝ) : Prop := (a = -1) ↔ (∀ x y : ℝ, x - y + 5 = 0 ↔ (a - 1) * x + (a + 3) * y - 2 = 0)

-- Theorem to prove
theorem correct_proposition : (∃ x : ℝ, ¬(p x)) ∧ (q (-1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_proposition_l1311_131132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_magnitude_l1311_131129

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (x/2) * Real.cos (x/2) + (Real.cos (x/2))^2 - 1/2

-- Define the theorem
theorem angle_C_magnitude (A B C : ℝ) (a b c : ℝ) :
  f (B + C) = 1 →
  a = Real.sqrt 3 →
  b = 1 →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  a * Real.sin B = b * Real.sin A →
  a * Real.sin C = c * Real.sin A →
  b * Real.sin C = c * Real.sin B →
  C = π / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_magnitude_l1311_131129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_general_term_l1311_131108

theorem arithmetic_sequence_general_term 
  (a : ℕ → ℝ) 
  (h1 : ∀ n, Real.log (a (n + 1)) - Real.log (a n) = Real.log 3)
  (h2 : Real.log (a 1) + Real.log (a 2) + Real.log (a 3) = 6 * Real.log 3) :
  ∀ n, a n = 3^n := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_general_term_l1311_131108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_correct_l1311_131144

noncomputable def sequence_formula (n : ℕ) : ℝ :=
  if n = 1 then 1
  else (Finset.range (n - 1)).prod (λ k => ((4 : ℝ) ^ k - 1) ^ 2 - 1)

theorem sequence_formula_correct (a : ℕ → ℝ) :
  (∀ n, a n > 0) →
  (∀ n, Real.sqrt (a n * a (n + 1) + a n * a (n + 2)) = 
    4 * Real.sqrt (a n * a (n + 1) + (a (n + 1))^2) + 3 * Real.sqrt (a n * a (n + 1))) →
  a 1 = 1 →
  a 2 = 8 →
  ∀ n, a n = sequence_formula n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_correct_l1311_131144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_is_correct_l1311_131128

open Real

-- Define the integrand
noncomputable def f (x : ℝ) : ℝ := (x^3 - 6*x^2 + 14*x - 4) / ((x-2)*(x+2)^3)

-- Define the antiderivative
noncomputable def F (x : ℝ) : ℝ := (1/8) * log (abs (x-2)) + (7/8) * log (abs (x+2)) + (17*x + 18) / (2*(x+2)^2)

-- State the theorem
theorem integral_is_correct : 
  ∀ x : ℝ, x ≠ 2 ∧ x ≠ -2 → deriv F x = f x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_is_correct_l1311_131128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_count_in_sequence_l1311_131113

def sequence_div4 (n : ℕ) : ℚ :=
  (7200 : ℚ) / (4^n)

def is_integer (q : ℚ) : Prop :=
  ∃ (n : ℤ), q = n

theorem integer_count_in_sequence :
  (∃ (k : ℕ), k > 0 ∧
    (∀ (n : ℕ), n < k → is_integer (sequence_div4 n)) ∧
    ¬is_integer (sequence_div4 k)) ∧
  (∀ (m : ℕ), (∀ (n : ℕ), n < m → is_integer (sequence_div4 n)) ∧
    ¬is_integer (sequence_div4 m) → m = 3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_count_in_sequence_l1311_131113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_transformed_function_l1311_131188

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x) - Real.cos (2 * x)

noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (x - Real.pi / 3)

theorem symmetry_center_of_transformed_function :
  (∀ x : ℝ, f x = 2 * Real.sin (2 * x - Real.pi / 6)) →
  (∀ x : ℝ, g x = 2 * Real.sin (x - Real.pi / 3)) →
  (∀ x : ℝ, g (Real.pi / 3 + x) = g (Real.pi / 3 - x)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_transformed_function_l1311_131188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alex_uphill_speed_l1311_131145

/-- Represents Alex's bike journey with given parameters -/
structure BikeJourney where
  flat_speed : ℝ
  flat_time : ℝ
  uphill_time : ℝ
  downhill_speed : ℝ
  downhill_time : ℝ
  total_distance : ℝ
  walking_distance : ℝ

/-- Calculates the average uphill speed given a BikeJourney -/
noncomputable def average_uphill_speed (journey : BikeJourney) : ℝ :=
  let flat_distance := journey.flat_speed * journey.flat_time
  let downhill_distance := journey.downhill_speed * journey.downhill_time
  let biking_distance := journey.total_distance - journey.walking_distance
  let uphill_distance := biking_distance - flat_distance - downhill_distance
  uphill_distance / journey.uphill_time

/-- Theorem stating that Alex's average uphill speed is 12 miles per hour -/
theorem alex_uphill_speed :
  let journey : BikeJourney := {
    flat_speed := 20,
    flat_time := 4.5,
    uphill_time := 2.5,
    downhill_speed := 24,
    downhill_time := 1.5,
    total_distance := 164,
    walking_distance := 8
  }
  average_uphill_speed journey = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alex_uphill_speed_l1311_131145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_interval_l1311_131149

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (-x^2 - 2*x + 3)

def g (x : ℝ) : ℝ := -x^2 - 2*x + 3

def domain : Set ℝ := {x : ℝ | g x ≥ 0}

theorem f_increasing_interval :
  ∃ (a b : ℝ), a = -3 ∧ b = -1 ∧
  (∀ x y, x ∈ domain → y ∈ domain → a ≤ x → x < y → y ≤ b → f x < f y) ∧
  (∀ ε > 0, ∃ x y, x ∈ domain ∧ y ∈ domain ∧ x < a + ε ∧ y > a - ε ∧ f x ≥ f y) ∧
  (∀ ε > 0, ∃ x y, x ∈ domain ∧ y ∈ domain ∧ x < b + ε ∧ y > b - ε ∧ f x ≤ f y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_interval_l1311_131149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_points_satisfy_condition_l1311_131166

-- Define the plane as ℝ × ℝ
def Plane := ℝ × ℝ

-- Define the distance function between two points
noncomputable def distance (p1 p2 : Plane) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem no_points_satisfy_condition (A B : Plane) :
  ¬∃ P : Plane, distance P A + distance P B = (1/2) * distance A B := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_points_satisfy_condition_l1311_131166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_abs_even_but_not_conversely_l1311_131173

-- Define a real-valued function
variable (f : ℝ → ℝ)

-- Define symmetry about the origin
def symmetric_about_origin (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define an even function
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- Theorem stating the relationship
theorem symmetry_implies_abs_even_but_not_conversely :
  (symmetric_about_origin f → is_even (fun x ↦ |f x|)) ∧
  ¬(is_even (fun x ↦ |f x|) → symmetric_about_origin f) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_abs_even_but_not_conversely_l1311_131173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inverse_f_15_l1311_131103

-- Define the functions f and g
noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- State the given conditions
axiom g_has_inverse : Function.Bijective g
axiom f_g_relation : ∀ x : ℝ, Function.invFun f (g x) = x^4 - 1

-- State the theorem to be proved
theorem g_inverse_f_15 : Function.invFun g (f 15) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inverse_f_15_l1311_131103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correlation_implies_linear_relationship_l1311_131163

/-- Represents the correlation coefficient between two variables -/
def correlation_coefficient : ℝ → Prop := sorry

/-- Defines what it means for two variables to be linearly correlated -/
def linearly_correlated : Prop := sorry

/-- Defines what it means for a real number to be close to 1 -/
def close_to_one (r : ℝ) : Prop :=
  abs r > 0.7

theorem correlation_implies_linear_relationship (r : ℝ) :
  correlation_coefficient r →
  close_to_one r →
  linearly_correlated := by
  sorry

#check correlation_implies_linear_relationship

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correlation_implies_linear_relationship_l1311_131163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_intercept_of_line_l1311_131194

/-- The y-intercept of a line is the y-coordinate of the point where the line intersects the y-axis. -/
noncomputable def y_intercept (a b c : ℝ) : ℝ := -c / b

/-- The line equation in the form ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

theorem y_intercept_of_line :
  let l : Line := { a := 2, b := -3, c := 6 }
  y_intercept l.a l.b l.c = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_intercept_of_line_l1311_131194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_parabola_l1311_131142

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Distance from a point to a line y = k -/
def distanceToLine (p : Point) (k : ℝ) : ℝ :=
  |p.y - k|

/-- The fixed point (0, 3) -/
def A : Point :=
  { x := 0, y := 3 }

/-- Predicate to check if a point is on a parabola -/
def IsParabola (p : Point) : Prop :=
  ∃ (a b c : ℝ), p.y = a * p.x^2 + b * p.x + c

/-- Theorem: The locus of points equidistant from a fixed point (0, 3) 
    and a fixed line y = -1 is a parabola -/
theorem locus_is_parabola :
  ∀ p : Point, distance p A = distanceToLine p (-1) → IsParabola p := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_parabola_l1311_131142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_altitudes_is_2416_over_17_l1311_131158

/-- A triangle formed by a line and coordinate axes -/
structure AxisTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : a * b = c

/-- The sum of altitudes of the triangle -/
noncomputable def sumOfAltitudes (t : AxisTriangle) : ℝ :=
  t.a + t.b + (2 * t.c) / Real.sqrt (t.a^2 + t.b^2)

/-- The specific triangle formed by 15x + 8y = 120 -/
def specificTriangle : AxisTriangle :=
  { a := 8
    b := 15
    c := 120
    eq := by norm_num }

theorem sum_of_altitudes_is_2416_over_17 :
  sumOfAltitudes specificTriangle = 2416 / 17 := by
  sorry

#eval specificTriangle.a
#eval specificTriangle.b
#eval specificTriangle.c

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_altitudes_is_2416_over_17_l1311_131158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_proof_l1311_131104

-- Define the two given circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the point M
def point_M : ℝ × ℝ := (2, -2)

-- Define the equation of the circle we want to prove
def target_circle (x y : ℝ) : Prop := x^2 + y^2 - 3*x - 2 = 0

-- Theorem statement
theorem circle_equation_proof :
  ∀ (x y : ℝ),
  (∃ (k : ℝ), x^2 + y^2 - 6*x + k*(x^2 + y^2 - 4) = 0) ∧
  target_circle point_M.1 point_M.2 →
  target_circle x y :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_proof_l1311_131104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_inequality_l1311_131148

theorem trig_inequality (x : ℝ) (h1 : 0 < x) (h2 : x < π / 2) :
  (Real.tan x) ^ (Real.sin x) + (1 / Real.tan x) ^ (Real.cos x) ≥ 2 ∧
  ((Real.tan x) ^ (Real.sin x) + (1 / Real.tan x) ^ (Real.cos x) = 2 ↔ x = π / 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_inequality_l1311_131148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_score_difference_l1311_131192

def exam_scores : List ℝ := [60, 75, 82, 88, 92]
def percentages : List ℝ := [0.05, 0.35, 0.30, 0.15, 0.15]

def mean (scores : List ℝ) (percentages : List ℝ) : ℝ :=
  List.sum (List.zipWith (· * ·) scores percentages)

def median (scores : List ℝ) (percentages : List ℝ) : ℝ :=
  scores[2]!

theorem exam_score_difference :
  |mean exam_scores percentages - median exam_scores percentages| = 1.15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_score_difference_l1311_131192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_grid_arrangement_l1311_131177

/-- Represents a 3x3 grid with some fixed numbers and variables A, B, C, D --/
structure Grid where
  A : ℕ
  B : ℕ
  C : ℕ
  D : ℕ

/-- Checks if two numbers are adjacent in the grid --/
def adjacent (g : Grid) (x y : ℕ) : Prop :=
  (x = 1 ∧ y = 3) ∨ (x = 1 ∧ y = 5) ∨ (x = 1 ∧ y = 9) ∨
  (x = 3 ∧ y = 5) ∨ (x = 3 ∧ y = 7) ∨
  (x = 5 ∧ y = 2) ∨ (x = 5 ∧ y = 7) ∨
  (x = 9 ∧ y = 2) ∨
  (x = 2 ∧ y = 7) ∨
  (x = g.A ∧ y = 1) ∨ (x = g.A ∧ y = 3) ∨
  (x = g.B ∧ y = 3) ∨ (x = g.B ∧ y = g.C) ∨
  (x = g.C ∧ y = 5) ∨ (x = g.C ∧ y = 7) ∨
  (x = g.D ∧ y = 5) ∨ (x = g.D ∧ y = 9)

/-- The main theorem --/
theorem unique_grid_arrangement (g : Grid) : 
  (∀ x y, adjacent g x y → x + y < 12) →
  (g.A ≠ 1 ∧ g.A ≠ 3 ∧ g.A ≠ 5 ∧ g.A ≠ 7 ∧ g.A ≠ 9) →
  (g.B ≠ 1 ∧ g.B ≠ 3 ∧ g.B ≠ 5 ∧ g.B ≠ 7 ∧ g.B ≠ 9) →
  (g.C ≠ 1 ∧ g.C ≠ 3 ∧ g.C ≠ 5 ∧ g.C ≠ 7 ∧ g.C ≠ 9) →
  (g.D ≠ 1 ∧ g.D ≠ 3 ∧ g.D ≠ 5 ∧ g.D ≠ 7 ∧ g.D ≠ 9) →
  g.A = 8 ∧ g.B = 6 ∧ g.C = 4 ∧ g.D = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_grid_arrangement_l1311_131177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1311_131123

-- Define the triangle's side lengths
noncomputable def a : ℝ := 26
noncomputable def b : ℝ := 24
noncomputable def c : ℝ := 10

-- Define the semi-perimeter
noncomputable def s : ℝ := (a + b + c) / 2

-- State the theorem
theorem triangle_area : 
  Real.sqrt (s * (s - a) * (s - b) * (s - c)) = 120 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1311_131123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_law_of_sines_l1311_131101

-- Define a triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the circumcircle radius
noncomputable def circumradius (t : Triangle) : ℝ := sorry

-- Define the length of a side
noncomputable def side_length (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the angle at a vertex
noncomputable def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem law_of_sines (t : Triangle) :
  let R := circumradius t
  let AB := side_length t.C t.B
  let BC := side_length t.A t.C
  let AC := side_length t.A t.B
  let α := angle t.B t.A t.C
  let β := angle t.A t.B t.C
  let γ := angle t.A t.C t.B
  Real.sin α / BC = Real.sin β / AC ∧
  Real.sin β / AC = Real.sin γ / AB ∧
  Real.sin γ / AB = 1 / (2 * R) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_law_of_sines_l1311_131101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cells_for_no_coverage_l1311_131156

/-- Represents an L-shape piece on a grid -/
structure LShape where
  x : Nat
  y : Nat
  orientation : Nat
deriving Repr

/-- Checks if an L-shape is valid within a 4x4 grid -/
def is_valid_l_shape (l : LShape) : Bool :=
  match l.orientation with
  | 0 => l.x < 3 && l.y < 4
  | 1 => l.x < 4 && l.y < 3
  | 2 => l.x > 0 && l.y < 4
  | 3 => l.x < 4 && l.y > 0
  | _ => false

/-- Checks if two L-shapes overlap -/
def do_l_shapes_overlap (l1 l2 : LShape) : Bool :=
  sorry

/-- Checks if a set of L-shapes covers all colored cells without overlapping -/
def is_valid_coverage (colored_cells : Set (Nat × Nat)) (l_shapes : List LShape) : Bool :=
  sorry

/-- The main theorem -/
theorem min_cells_for_no_coverage :
  ∀ (colored_cells : Set (Nat × Nat)),
    (∀ (x y : Nat), x < 4 ∧ y < 4 → (x, y) ∈ colored_cells) →
    ¬∃ (l_shapes : List LShape),
      (∀ l, l ∈ l_shapes → is_valid_l_shape l) ∧
      (∀ l1 l2, l1 ∈ l_shapes → l2 ∈ l_shapes → l1 ≠ l2 → ¬do_l_shapes_overlap l1 l2) ∧
      is_valid_coverage colored_cells l_shapes :=
by
  sorry

#check min_cells_for_no_coverage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cells_for_no_coverage_l1311_131156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_annual_production_formula_l1311_131134

/-- Represents the annual production after x years, given initial production a and percentage increase p -/
noncomputable def annual_production (a : ℝ) (p : ℝ) (x : ℕ) : ℝ :=
  a * (1 + p / 100) ^ x

/-- Theorem stating that the annual production after x years is a(1+p%)^x -/
theorem annual_production_formula (a : ℝ) (p : ℝ) (x : ℕ) :
  annual_production a p x = a * (1 + p / 100) ^ x :=
by
  -- Unfold the definition of annual_production
  unfold annual_production
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_annual_production_formula_l1311_131134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_l1311_131111

/-- The time taken for two trains to cross each other -/
noncomputable def time_to_cross (length1 length2 speed1 speed2 : ℝ) : ℝ :=
  let total_length := length1 + length2
  let speed1_ms := speed1 * (5/18)  -- Convert km/hr to m/s
  let speed2_ms := speed2 * (5/18)  -- Convert km/hr to m/s
  let relative_speed := speed1_ms + speed2_ms
  total_length / relative_speed

/-- Theorem stating the time taken for two specific trains to cross each other -/
theorem trains_crossing_time :
  let length1 := (250 : ℝ)  -- Length of train 1 in meters
  let length2 := (120 : ℝ)  -- Length of train 2 in meters
  let speed1 := (80 : ℝ)    -- Speed of train 1 in km/hr
  let speed2 := (50 : ℝ)    -- Speed of train 2 in km/hr
  ∃ ε > 0, |time_to_cross length1 length2 speed1 speed2 - 10.25| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_l1311_131111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_40_l1311_131124

-- Define the lines
def line1 (x : ℝ) : ℝ := 3 * x - 6
def line2 (x : ℝ) : ℝ := -2 * x + 14

-- Define the intersection point of line1 and line2
def intersection : ℝ × ℝ :=
  let x := 4
  let y := line1 x
  (x, y)

-- Define the y-intercepts
def y_intercept1 : ℝ := line1 0
def y_intercept2 : ℝ := line2 0

-- Define the base of the triangle
def base : ℝ := y_intercept2 - y_intercept1

-- Define the height of the triangle
def triangle_height : ℝ := intersection.1

-- Theorem: The area of the triangle is 40 square units
theorem triangle_area_is_40 : (1/2 : ℝ) * base * triangle_height = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_40_l1311_131124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_removal_theorem_l1311_131167

/-- The number of ways to remove balls from a bag -/
def ballRemovalWays (n : ℕ) : ℚ :=
  (1 : ℚ) / n * (Nat.choose (2 * n - 2) (n - 1) : ℚ)

/-- Theorem stating the number of ways to remove balls from a bag -/
theorem ball_removal_theorem (n : ℕ) (h : n > 0) :
  ballRemovalWays n = (1 : ℚ) / n * (Nat.choose (2 * n - 2) (n - 1) : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_removal_theorem_l1311_131167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_perimeter_l1311_131170

/-- Represents the side lengths of the 9 squares -/
structure SquareSides where
  a : Fin 9 → ℕ

/-- Represents the dimensions of the rectangle -/
structure RectangleDimensions where
  length : ℕ
  width : ℕ

/-- Checks if the square side lengths satisfy the given relationships -/
def valid_square_sides (s : SquareSides) : Prop :=
  s.a 0 + s.a 1 = s.a 2 ∧
  s.a 0 + s.a 2 = s.a 3 ∧
  s.a 2 + s.a 3 = s.a 4 ∧
  s.a 3 + s.a 4 = s.a 5 ∧
  s.a 1 + s.a 2 + s.a 4 = s.a 6 ∧
  s.a 1 + s.a 6 = s.a 7 ∧
  s.a 0 + s.a 3 + s.a 5 = s.a 8 ∧
  s.a 5 + s.a 8 = s.a 6 + s.a 7

/-- Checks if the rectangle dimensions are valid -/
def valid_rectangle_dimensions (r : RectangleDimensions) (s : SquareSides) : Prop :=
  r.length = s.a 5 + s.a 8 ∧
  r.width = s.a 6 + s.a 7 ∧
  Nat.Coprime r.length r.width

/-- The main theorem -/
theorem rectangle_perimeter (s : SquareSides) (r : RectangleDimensions) :
  valid_square_sides s → valid_rectangle_dimensions r s →
  2 * (r.length + r.width) = 260 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_perimeter_l1311_131170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l1311_131152

/-- Parabola type representing x² = 4y --/
structure Parabola where
  x : ℝ
  y : ℝ
  eq : x^2 = 4*y

/-- Focus of the parabola --/
def focus : ℝ × ℝ := (0, 1)

/-- Point A --/
def point_A : ℝ × ℝ := (-1, 8)

/-- Distance between two points --/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem stating the minimum value of |PA| + |PF| --/
theorem min_distance_sum (P : Parabola) :
  ∃ (min_val : ℝ), min_val = 9 ∧
  ∀ (Q : Parabola), distance (Q.x, Q.y) point_A + distance (Q.x, Q.y) focus ≥ min_val := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l1311_131152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_one_half_l1311_131109

/-- The number of empty rooms -/
def num_rooms : ℕ := 2

/-- The number of people -/
def num_people : ℕ := 2

/-- The total number of possible arrangements -/
def total_arrangements : ℕ := num_rooms ^ num_people

/-- The number of arrangements where each person occupies a different room -/
def favorable_arrangements : ℕ := Nat.factorial num_people

/-- The probability that each person occupies a different room -/
noncomputable def probability_different_rooms : ℚ := 
  (favorable_arrangements : ℚ) / (total_arrangements : ℚ)

theorem probability_is_one_half : probability_different_rooms = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_one_half_l1311_131109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_keychain_arrangements_l1311_131162

/-- Represents the number of keys on the keychain -/
def total_keys : ℕ := 6

/-- Represents the number of distinct units to arrange (2 paired units + 2 single keys) -/
def distinct_units : ℕ := 4

/-- The number of distinct arrangements of keys on the keychain -/
def distinct_arrangements : ℕ := 3

/-- Theorem stating that the number of distinct arrangements is 3 -/
theorem keychain_arrangements :
  (Nat.factorial (distinct_units - 1)) / 2 = distinct_arrangements :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_keychain_arrangements_l1311_131162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_tax_difference_l1311_131182

/-- Proves that the difference between 8% and 7% sales tax on a $40 item is $0.4 -/
theorem sales_tax_difference 
  (item_price : ℝ)
  (tax_rate_1 : ℝ)
  (tax_rate_2 : ℝ)
  (h1 : item_price = 40)
  (h2 : tax_rate_1 = 0.08)
  (h3 : tax_rate_2 = 0.07) :
  item_price * (tax_rate_1 - tax_rate_2) = 0.4 := by
  sorry

#check sales_tax_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_tax_difference_l1311_131182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_band_arrangement_count_l1311_131184

/-- The number of possible row arrangements for a 96-member band with 6 to 18 members per row -/
theorem band_arrangement_count : 
  (Finset.filter (fun x => 6 ≤ x ∧ x ≤ 18 ∧ 96 % x = 0) (Finset.range 19)).card = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_band_arrangement_count_l1311_131184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_difference_factorization_l1311_131110

theorem cube_difference_factorization (a : ℝ) (ha : a ≠ 0) :
  a^3 - (1/a)^3 = (a - 1/a) * (a^2 + 1 + (1/a)^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_difference_factorization_l1311_131110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l1311_131153

/-- A parabola with equation y² = 2px passing through point (4, 4) -/
structure Parabola where
  p : ℝ
  eq : ∀ x y : ℝ, y^2 = 2 * p * x
  passes_through : (4 : ℝ)^2 = 2 * p * 4

/-- The directrix of the parabola -/
def directrix (para : Parabola) : ℝ → Prop :=
  λ x => x = -para.p

/-- The focus of the parabola -/
def focus (para : Parabola) : ℝ × ℝ :=
  (para.p, 0)

/-- The distance between two points in ℝ² -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_properties (para : Parabola) :
  (directrix para = λ x => x = -2) ∧
  (distance (4, 4) (focus para) = 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l1311_131153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_side_length_l1311_131175

/-- Represents an infinite sequence of equilateral triangles where each triangle
    is formed by joining the midpoints of the previous triangle's sides -/
noncomputable def TriangleSequence (s : ℝ) : ℕ → ℝ
  | 0 => s
  | n + 1 => TriangleSequence s n / 2

/-- The perimeter of the nth triangle in the sequence -/
noncomputable def Perimeter (s : ℝ) (n : ℕ) : ℝ := 3 * TriangleSequence s n

/-- The sum of perimeters of all triangles in the infinite sequence -/
noncomputable def SumOfPerimeters (s : ℝ) : ℝ := (3 * s) / (1 - (1/2))

theorem equilateral_triangle_side_length (s : ℝ) 
  (h : SumOfPerimeters s = 360) : s = 60 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_side_length_l1311_131175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_expression_equality_l1311_131112

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- State the theorem
theorem floor_expression_equality : 
  (floor 6.5) * (floor (2 / 3 : ℝ)) + (floor 2) * (7.2 : ℝ) + (floor 8.4) - (6 : ℝ) = 10.4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_expression_equality_l1311_131112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_lambda_value_l1311_131185

theorem max_lambda_value (a b c lambda : ℝ) :
  (0 < a ∧ a ≤ 1) →
  (0 < b ∧ b ≤ 1) →
  (0 < c ∧ c ≤ 1) →
  (∀ (x y z : ℝ), 0 < x ∧ x ≤ 1 → 0 < y ∧ y ≤ 1 → 0 < z ∧ z ≤ 1 →
    Real.sqrt 3 / Real.sqrt (x + y + z) ≥ 1 + lambda * (1 - x) * (1 - y) * (1 - z)) →
  lambda ≤ 64 / 27 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_lambda_value_l1311_131185
