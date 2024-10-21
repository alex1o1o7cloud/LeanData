import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_silver_cube_price_l1066_106691

/-- Calculates the selling price of a silver cube -/
noncomputable def silver_cube_selling_price (side_length : ℝ) (weight_per_cubic_inch : ℝ) 
  (price_per_ounce : ℝ) (markup_percentage : ℝ) : ℝ :=
  let volume := side_length ^ 3
  let weight := volume * weight_per_cubic_inch
  let value := weight * price_per_ounce
  value * (1 + markup_percentage / 100)

/-- Theorem: The selling price of Bob's silver cube is $4455 -/
theorem bob_silver_cube_price : 
  silver_cube_selling_price 3 6 25 10 = 4455 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval silver_cube_selling_price 3 6 25 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_silver_cube_price_l1066_106691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prop_3_prop_4_l1066_106662

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (perpendicular_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- Theorem for proposition 3
theorem prop_3 
  (m n : Line) (α β : Plane) 
  (h1 : perpendicular_plane m α) 
  (h2 : parallel m n) 
  (h3 : parallel_plane α β) : 
  perpendicular_plane n β :=
sorry

-- Theorem for proposition 4
theorem prop_4 
  (m n : Line) (α β : Plane) 
  (h1 : perpendicular m n) 
  (h2 : perpendicular_plane m α) 
  (h3 : perpendicular_plane n β) : 
  perpendicular_planes α β :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prop_3_prop_4_l1066_106662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_small_bag_price_l1066_106679

/-- Represents the price and balloon count for a bag of balloons -/
structure BalloonBag where
  price : ℚ
  count : ℕ

/-- Represents Mark's balloon purchase scenario -/
structure BalloonScenario where
  budget : ℚ
  mediumBag : BalloonBag
  largeBag : BalloonBag
  smallBagCount : ℕ
  totalBalloons : ℕ

/-- The specific scenario from the problem -/
def markScenario : BalloonScenario :=
  { budget := 24
  , mediumBag := { price := 6, count := 75 }
  , largeBag := { price := 12, count := 200 }
  , smallBagCount := 50
  , totalBalloons := 400
  }

/-- Calculate the price of small bags given the scenario -/
def calculateSmallBagPrice (scenario : BalloonScenario) : ℚ :=
  scenario.budget / (scenario.totalBalloons / scenario.smallBagCount : ℚ)

theorem small_bag_price :
  calculateSmallBagPrice markScenario = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_small_bag_price_l1066_106679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l1066_106642

-- Define the lines
def l1 (x y : ℝ) : Prop := 4 * x - 3 * y + 16 = 0
def l2 (x : ℝ) : Prop := x = -1

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the distance functions
noncomputable def d1 (x y : ℝ) : ℝ := abs (4 * x - 3 * y + 16) / Real.sqrt (4^2 + (-3)^2)
def d2 (x : ℝ) : ℝ := abs (x + 1)

-- Theorem statement
theorem min_distance_sum :
  ∃ (min_val : ℝ), min_val = 4 ∧
  ∀ (x y : ℝ), parabola x y → d1 x y + d2 x ≥ min_val := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l1066_106642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_g_strictly_decreasing_l1066_106648

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := Real.sin x
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := m * x - x^3 / 6

-- Theorem for the tangent line equation
theorem tangent_line_equation :
  ∃ (a b c : ℝ), a * (π/4) + b * (f (π/4)) + c = 0 ∧
  ∀ (x y : ℝ), y = f x → (a * x + b * y + c = 0 ↔ x - Real.sqrt 2 * y + 1 - π/4 = 0) := by
  sorry

-- Theorem for the intervals where g(x) is strictly decreasing
theorem g_strictly_decreasing (m : ℝ) :
  (m ≤ 0 → ∀ x y : ℝ, x < y → g m y < g m x) ∧
  (m > 0 → (∀ x y : ℝ, x < y ∧ y < -Real.sqrt (2*m) → g m y < g m x) ∧
           (∀ x y : ℝ, Real.sqrt (2*m) < x ∧ x < y → g m y < g m x)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_g_strictly_decreasing_l1066_106648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_table_tennis_probabilities_l1066_106634

/-- Represents a table tennis match between two players -/
structure TableTennisMatch where
  prob_a_scores_when_a_serves : ℝ
  prob_a_scores_when_b_serves : ℝ
  prob_game_ends_after_two_points : ℝ

/-- The probability of player A winning after four more points -/
def prob_a_wins_after_four_points (m : TableTennisMatch) : ℝ :=
  sorry

theorem table_tennis_probabilities (m : TableTennisMatch) 
  (h1 : m.prob_a_scores_when_b_serves = 2/5)
  (h2 : m.prob_game_ends_after_two_points = 7/15) :
  m.prob_a_scores_when_a_serves = 2/3 ∧ 
  prob_a_wins_after_four_points m = 32/225 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_table_tennis_probabilities_l1066_106634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_arithmetic_proofs_l1066_106668

theorem integer_arithmetic_proofs :
  (56 + (-18) + 37 = 75) ∧
  (12 - (-18) + (-7) - 15 = 8) ∧
  ((-2) + 3 + 19 - 11 = 9) ∧
  (0 - 4 + (-6) - (-8) = -2) := by
  constructor
  · ring
  · constructor
    · ring
    · constructor
      · ring
      · ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_arithmetic_proofs_l1066_106668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_two_digit_prime_factor_of_binom_250_125_l1066_106612

theorem largest_two_digit_prime_factor_of_binom_250_125 :
  (∃ (p : ℕ), Nat.Prime p ∧ 10 ≤ p ∧ p < 100 ∧ p ∣ Nat.choose 250 125 ∧
    ∀ (q : ℕ), Nat.Prime q → 10 ≤ q → q < 100 → q ∣ Nat.choose 250 125 → q ≤ p) ∧
  (83 ∣ Nat.choose 250 125) ∧ Nat.Prime 83 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_two_digit_prime_factor_of_binom_250_125_l1066_106612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solutions_l1066_106651

theorem functional_equation_solutions (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (f x * f y) + f (x + y) = f (x * y)) →
  ((∀ x : ℝ, f x = 0) ∨
   (∀ x : ℝ, f x = x - 1) ∨
   (∀ x : ℝ, f x = 1 - x)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solutions_l1066_106651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_coplanar_l1066_106623

/-- Given four points P, A, B, and C in ℝ³, if for any point O,
    OP = (1/4)OA + (1/4)OB + (1/2)OC, then P, A, B, and C are coplanar. -/
theorem points_coplanar (P A B C : Fin 3 → ℝ) :
  (∀ O : Fin 3 → ℝ, (O - P) = (1/4 : ℝ) • (O - A) + (1/4 : ℝ) • (O - B) + (1/2 : ℝ) • (O - C)) →
  ∃ (a b c d : ℝ), a • P + b • A + c • B + d • C = 0 ∧ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 ∨ d ≠ 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_coplanar_l1066_106623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_lower_bound_sum_cubes_roots_l1066_106646

-- Define a monic polynomial of degree n with real coefficients
def MonicPolynomial (n : ℕ) := Polynomial ℝ

-- Define the conditions on the coefficients
def CoefficientsCondition (p : MonicPolynomial n) : Prop :=
  ∃ a_n_2 : ℝ, 
    p.coeff (n-1) = -a_n_2 ∧ 
    p.coeff (n-2) = a_n_2 ∧ 
    p.coeff (n-3) = 2*a_n_2

-- Define the sum of cubes of roots
noncomputable def SumOfCubesOfRoots (p : MonicPolynomial n) : ℝ := 
  (p.roots.map (λ r => r^3)).sum

-- State the theorem
theorem greatest_lower_bound_sum_cubes_roots 
  (n : ℕ) (p : MonicPolynomial n) 
  (h : CoefficientsCondition p) : 
  |SumOfCubesOfRoots p| ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_lower_bound_sum_cubes_roots_l1066_106646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_distance_to_origin_l1066_106692

/-- A point on a parabola with a specific distance to its focus -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 4*x
  dist_to_focus : (x - 1)^2 + y^2 = 9

/-- The distance from a point to the origin -/
noncomputable def dist_to_origin (p : ParabolaPoint) : ℝ :=
  Real.sqrt (p.x^2 + p.y^2)

/-- Theorem stating that a point on the parabola y² = 4x with distance 3 to the focus
    has distance 2√3 to the origin -/
theorem parabola_point_distance_to_origin (p : ParabolaPoint) :
  dist_to_origin p = 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_distance_to_origin_l1066_106692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_b_for_real_roots_l1066_106658

theorem unique_b_for_real_roots : ∃! (b : ℝ), b > 0 ∧ 
  ∃ (a : ℝ), a > 0 ∧ 
  (∀ (a' : ℝ), a' > 0 → 
    (∀ (x : ℝ), x^3 - 2*a'*x^2 + b*x - 2*a' = 0 → x ∈ Set.univ) → 
    a ≤ a') ∧
  (∀ (x : ℝ), x^3 - 2*a*x^2 + b*x - 2*a = 0 → x ∈ Set.univ) ∧
  b = 9 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_b_for_real_roots_l1066_106658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_f_implies_product_one_l1066_106660

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := |Real.log x / Real.log 3|

-- State the theorem
theorem equal_f_implies_product_one 
  (a b : ℝ) 
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : a ≠ b) 
  (h4 : f a = f b) : 
  a * b = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_f_implies_product_one_l1066_106660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_collinearity_l1066_106656

-- Define the necessary types and structures
structure Point : Type :=
  (x : ℝ) (y : ℝ)

def Line : Type := Point → Point → Prop

-- Define the collinearity of three points
def AreCollinear (p q r : Point) : Prop :=
  ∃ (t : ℝ), q.x - p.x = t * (r.x - p.x) ∧ q.y - p.y = t * (r.y - p.y)

-- Define the intersection of two lines
def Intersect (l₁ l₂ : Line) (p : Point) : Prop :=
  l₁ p p ∧ l₂ p p

-- State the theorem
theorem intersection_collinearity 
  (A C I M N P Q T : Point) 
  (AP CQ MP NQ : Line) 
  (K : Point) 
  (h1 : Intersect AP CQ T) 
  (h2 : Intersect MP NQ K) : 
  AreCollinear T K I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_collinearity_l1066_106656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_solution_l1066_106626

/-- Given a function f(x) = 1 / (ax² + bx + c) where a, b, and c are non-zero constants,
    the inverse function f⁻¹(x) = k has solutions that are roots of ak³ + bk² + ck - 1 = 0 -/
theorem inverse_function_solution (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  let f := fun x ↦ 1 / (a * x^2 + b * x + c)
  ∀ k : ℝ, (∃ x : ℝ, f x = k) ↔ a * k^3 + b * k^2 + c * k - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_solution_l1066_106626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_coordinates_l1066_106602

noncomputable def ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

noncomputable def right_focus : ℝ × ℝ := (Real.sqrt 3, 0)

def line_through_focus (k : ℝ) (x y : ℝ) : Prop :=
  y = k * (x - (Real.sqrt 3))

def intersect_ellipse (A B : ℝ × ℝ) : Prop :=
  ellipse A.1 A.2 ∧ ellipse B.1 B.2

def distance_squared (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

def origin : ℝ × ℝ := (0, 0)

theorem ellipse_intersection_coordinates
  (k : ℝ) (A B : ℝ × ℝ) (hk : k ≠ 0)
  (hline : line_through_focus k A.1 A.2)
  (hintersect : intersect_ellipse A B)
  (hdist : distance_squared A origin = distance_squared A B) :
  A = (Real.sqrt 3 / 2, Real.sqrt 13 / 4) ∨
  A = (Real.sqrt 3 / 2, -Real.sqrt 13 / 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_coordinates_l1066_106602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_g_l1066_106637

/-- A polynomial with real coefficients -/
def MyPolynomial := ℝ → ℝ

/-- The degree of a polynomial -/
noncomputable def degree (p : MyPolynomial) : ℕ := sorry

/-- Given polynomial f(x) -/
def f : MyPolynomial := λ x => -9*x^5 + 4*x^3 + 2*x - 6

theorem degree_of_g (g : MyPolynomial) 
  (h : degree (λ x => f x + g x) = 2) : 
  degree g = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_g_l1066_106637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_palindrome_in_multiple_bases_l1066_106672

-- Helper function to convert a number to its digit representation in a given base
def to_digits (base : ℕ+) (n : ℕ+) : List ℕ :=
  sorry

-- Helper function to check if a list of digits is a palindrome
def is_palindrome (d : ℕ+) (digits : List ℕ) : Prop :=
  digits.length = d ∧ digits = digits.reverse

theorem palindrome_in_multiple_bases (K d : ℕ+) : 
  ∃ (n : ℕ+) (b : Fin K → ℕ+), 
    ∀ (i : Fin K), is_palindrome d (to_digits (b i) n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_palindrome_in_multiple_bases_l1066_106672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_extension_lengths_l1066_106621

/-- A trapezoid with given side lengths -/
structure Trapezoid where
  AD : ℝ
  BC : ℝ
  AB : ℝ
  CD : ℝ

/-- The point of intersection when lateral sides are extended -/
def ExtensionPoint (t : Trapezoid) : Point := sorry

/-- Length of extension MB -/
def ExtensionLengthMB (t : Trapezoid) : ℝ := sorry

/-- Length of extension MC -/
def ExtensionLengthMC (t : Trapezoid) : ℝ := sorry

/-- Theorem: For a trapezoid with given measurements, the extensions have specific lengths -/
theorem trapezoid_extension_lengths (t : Trapezoid) 
  (h1 : t.AD = 1.8) (h2 : t.BC = 1.2) (h3 : t.AB = 1.5) (h4 : t.CD = 1.2) : 
  ExtensionLengthMB t = 3 ∧ ExtensionLengthMC t = 2.4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_extension_lengths_l1066_106621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_diameter_approx_l1066_106698

/-- The number of complete revolutions a bicycle wheel makes in 1 km -/
def revolutions_per_km : ℝ := 624.4536030972898

/-- The length of 1 km in meters -/
def km_to_meters : ℝ := 1000

/-- The value of pi (π) -/
noncomputable def π : ℝ := Real.pi

/-- The diameter of the bicycle wheel in meters -/
noncomputable def wheel_diameter : ℝ :=
  (km_to_meters / revolutions_per_km) / π

theorem wheel_diameter_approx :
  |wheel_diameter - 0.5097| < 0.0001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_diameter_approx_l1066_106698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ordinate_of_D_max_at_three_halves_fifteen_fourths_is_max_l1066_106659

-- Define the parabola G
noncomputable def G (x : ℝ) : ℝ := -1/3 * x^2 + 3

-- Define points A and B
def A : ℝ × ℝ := (-3, 0)
def B : ℝ × ℝ := (0, 3)

-- Define the line AB
noncomputable def lineAB (x : ℝ) : ℝ := x + 3

-- Define the parabola H (parametrized by m)
noncomputable def H (m : ℝ) (x : ℝ) : ℝ := -1/3 * (x - m)^2 + m + 3

-- Define the y-coordinate of point D (as a function of m)
noncomputable def yD (m : ℝ) : ℝ := H m 0

theorem max_ordinate_of_D :
  ∃ (m : ℝ), ∀ (n : ℝ), yD m ≥ yD n ∧ yD m = 15/4 := by
  sorry

-- Prove that the maximum value occurs at m = 3/2
theorem max_at_three_halves :
  yD (3/2) = 15/4 := by
  sorry

-- Prove that 15/4 is indeed the maximum value
theorem fifteen_fourths_is_max :
  ∀ (n : ℝ), yD (3/2) ≥ yD n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ordinate_of_D_max_at_three_halves_fifteen_fourths_is_max_l1066_106659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_negative_23pi_over_6_l1066_106681

-- Define the function f
noncomputable def f (α : ℝ) : ℝ :=
  (2 * Real.sin (2 * Real.pi - α) * Real.cos (2 * Real.pi + α) - Real.cos (-α)) /
  (1 + Real.sin α ^ 2 + Real.sin (2 * Real.pi + α) - Real.cos (4 * Real.pi - α) ^ 2)

-- State the theorem
theorem f_value_at_negative_23pi_over_6 :
  f (-23 * Real.pi / 6) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_negative_23pi_over_6_l1066_106681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_relation_l1066_106630

/-- Definition of an ellipse with foci on x-axis or y-axis -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b
  h_a_ge_b : a ≥ b

/-- Eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a) ^ 2)

/-- Two ellipses where one passes through the vertices and foci of the other -/
structure EllipsePair where
  c1 : Ellipse
  c2 : Ellipse
  h_c2_passes : c2.a = c1.b ∧ c2.b = Real.sqrt (c1.a^2 - c1.b^2)

theorem eccentricity_relation (ep : EllipsePair) :
  let e1 := eccentricity ep.c1
  let e2 := eccentricity ep.c2
  e1^2 < 1/2 ∧ e1^2 + e2^2 > 1 := by
  sorry

#check eccentricity_relation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_relation_l1066_106630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l1066_106640

/-- The equation of a circle with center (1, -2) and tangent to the line x - 3y + 3 = 0 is (x-1)^2 + (y+2)^2 = 10 -/
theorem circle_equation (x y : ℝ) :
  let center : ℝ × ℝ := (1, -2)
  let line := {p : ℝ × ℝ | p.1 - 3*p.2 + 3 = 0}
  let distance (p : ℝ × ℝ) := |p.1 - 3*p.2 + 3| / Real.sqrt 10
  let is_tangent := distance center = Real.sqrt 10
  is_tangent → ((x - 1)^2 + (y + 2)^2 = 10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l1066_106640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_sum_of_cubes_l1066_106696

theorem cubic_equation_sum_of_cubes :
  ∀ u v w : ℝ,
  (∀ x : ℝ, (x - (20 : ℝ)^(1/3 : ℝ)) * (x - (70 : ℝ)^(1/3 : ℝ)) * (x - (170 : ℝ)^(1/3 : ℝ)) = 1/2 ↔ x = u ∨ x = v ∨ x = w) →
  u ≠ v ∧ u ≠ w ∧ v ≠ w →
  u^3 + v^3 + w^3 = 261.5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_sum_of_cubes_l1066_106696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l1066_106677

theorem function_property (f : ℕ → ℕ) 
  (h1 : ∀ n, f (f n) + f n = 2 * n + 3) 
  (h2 : f 0 = 1) : 
  ∀ n, f n = n + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l1066_106677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_sum_to_7_is_196_l1066_106667

def divisors_294 : List Nat := [2, 3, 7, 14, 21, 49, 42, 98, 147, 294]

def is_valid_arrangement (arr : List Nat) : Prop :=
  ∀ i, (i + 1) % arr.length < arr.length → 
    ∃ k > 1, k ∣ arr[i]! ∧ k ∣ arr[(i + 1) % arr.length]!

def adjacent_sum_to_7 (arr : List Nat) : Nat :=
  let i := arr.indexOf 7
  arr[(i - 1 + arr.length) % arr.length]! + arr[(i + 1) % arr.length]!

theorem adjacent_sum_to_7_is_196 :
  ∀ arr : List Nat, arr.Perm divisors_294 → is_valid_arrangement arr →
    adjacent_sum_to_7 arr = 196 := by
  sorry

#check adjacent_sum_to_7_is_196

end NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_sum_to_7_is_196_l1066_106667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1066_106650

/-- Given a train of length 120 meters that takes 5.999520038396929 seconds to cross an electric pole, its speed is approximately 72.004800384 km/hr. -/
theorem train_speed_calculation (train_length : ℝ) (time_to_cross : ℝ) (speed_kmh : ℝ) :
  train_length = 120 →
  time_to_cross = 5.999520038396929 →
  speed_kmh = (train_length / time_to_cross) * 3.6 →
  ‖speed_kmh - 72.004800384‖ < 1e-6 := by
  sorry

#check train_speed_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1066_106650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_circle_tangent_intersection_l1066_106661

-- Define the ellipse E
def E (x y : ℝ) : Prop := x^2 / 8 + y^2 / 4 = 1

-- Define the circle C
def C (x y : ℝ) : Prop := x^2 + y^2 = 8/3

-- Define the tangent line to C
def tangent_to_C (k m : ℝ) (x y : ℝ) : Prop := y = k * x + m

-- Define the perpendicularity condition
def perpendicular (x1 y1 x2 y2 : ℝ) : Prop := x1 * x2 + y1 * y2 = 0

-- Define the length of AB
noncomputable def length_AB (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Main theorem
theorem ellipse_circle_tangent_intersection :
  ∀ (k m : ℝ),
  ∃ (x1 y1 x2 y2 : ℝ),
  E x1 y1 ∧ E x2 y2 ∧
  tangent_to_C k m x1 y1 ∧ tangent_to_C k m x2 y2 ∧
  perpendicular x1 y1 x2 y2 ∧
  (4 * Real.sqrt 6 / 3 ≤ length_AB x1 y1 x2 y2 ∧ length_AB x1 y1 x2 y2 ≤ 2 * Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_circle_tangent_intersection_l1066_106661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_median_mean_equality_l1066_106627

noncomputable def numbers (y : ℝ) : List ℝ := [3, 7, 9, 19, y]

noncomputable def is_median (list : List ℝ) (m : ℝ) : Prop :=
  2 * (list.filter (λ x => x ≤ m)).length ≥ list.length ∧
  2 * (list.filter (λ x => x ≥ m)).length ≥ list.length

noncomputable def mean (list : List ℝ) : ℝ := list.sum / list.length

theorem unique_median_mean_equality :
  ∃! y : ℝ, is_median (numbers y) (mean (numbers y)) ∧ y = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_median_mean_equality_l1066_106627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l1066_106616

/-- The differential equation: x^2 y'' + 2x y' - 6y = 0 -/
def differential_equation (y : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → x^2 * (deriv^[2] y) x + 2*x * (deriv y) x - 6 * y x = 0

/-- The boundary condition: y(1) = 1 -/
def boundary_condition (y : ℝ → ℝ) : Prop :=
  y 1 = 1

/-- The boundedness condition: y(x) is bounded as x → 0 -/
def boundedness_condition (y : ℝ → ℝ) : Prop :=
  ∃ M : ℝ, ∀ x : ℝ, 0 < x → x < 1 → |y x| ≤ M

/-- The proposed solution: y(x) = x^2 -/
def proposed_solution : ℝ → ℝ := λ x ↦ x^2

theorem unique_solution :
  (differential_equation proposed_solution) ∧
  (boundary_condition proposed_solution) ∧
  (boundedness_condition proposed_solution) ∧
  (∀ y : ℝ → ℝ, differential_equation y → boundary_condition y → boundedness_condition y → y = proposed_solution) := by
  sorry

#check unique_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l1066_106616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_children_multiple_of_max_tables_l1066_106603

/-- Represents the number of children at the event -/
def num_children : ℕ := sorry

/-- Represents the number of adults at the event -/
def num_adults : ℕ := 12

/-- Represents the maximum number of tables that can be set up -/
def max_tables : ℕ := 4

/-- Theorem stating that the number of children must be a multiple of the maximum number of tables -/
theorem children_multiple_of_max_tables : 
  ∃ k : ℕ, num_children = k * max_tables ∧ 
  (num_adults + num_children) % max_tables = 0 := by
  sorry

#check children_multiple_of_max_tables

end NUMINAMATH_CALUDE_ERRORFEEDBACK_children_multiple_of_max_tables_l1066_106603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_arrangement_exists_l1066_106625

/-- A triangle in a plane --/
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

/-- An arrangement of triangles in a plane --/
def Arrangement := Fin 10 → Triangle

/-- Two triangles intersect --/
def intersect (t1 t2 : Triangle) : Prop := sorry

/-- A point is in a triangle --/
def point_in_triangle (p : ℝ × ℝ) (t : Triangle) : Prop := sorry

/-- The statement to be proved --/
theorem triangle_arrangement_exists : 
  ∃ (arr : Arrangement), 
    (∀ i j, i ≠ j → intersect (arr i) (arr j)) ∧ 
    (∀ i j k, i ≠ j → j ≠ k → i ≠ k → 
      ¬∃ (p : ℝ × ℝ), point_in_triangle p (arr i) ∧ 
                      point_in_triangle p (arr j) ∧ 
                      point_in_triangle p (arr k)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_arrangement_exists_l1066_106625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l1066_106635

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x + 3 / x - 4
  else if x < 0 then -((-x) + 3 / (-x) - 4)
  else 0

theorem f_increasing_on_interval :
  (∀ x, f (-x) = -f x) →  -- f is odd
  (∀ x > 0, f x = x + 3 / x - 4) →  -- definition for x > 0
  ∀ x y, Real.sqrt 3 < x → x < y → f x < f y :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l1066_106635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_probability_theorem_l1066_106605

def fair_dice_roll : Finset ℕ := Finset.range 6

def is_divisible_by_four (n : ℕ) : Prop := n % 4 = 0

def two_digit_number (a b : ℕ) : ℕ := 10 * a + b

theorem dice_probability_theorem :
  let outcomes := fair_dice_roll.product fair_dice_roll
  let favorable_outcomes := outcomes.filter (fun p => p.1 = 4 ∧ p.2 = 4)
  (favorable_outcomes.card : ℚ) / outcomes.card = 1 / 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_probability_theorem_l1066_106605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_for_star_equation_l1066_106669

-- Define the star operation
noncomputable def star (a b : ℝ) : ℝ := Real.sqrt (a + b) / Real.sqrt (a - b)

-- Theorem statement
theorem x_value_for_star_equation :
  ∀ x : ℝ, x > 30 → star x 30 = 8 → x = 650 / 21 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_for_star_equation_l1066_106669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_special_polygon_l1066_106604

/-- A simple closed polygon in 2D space -/
structure Polygon where
  vertices : List (ℝ × ℝ)
  is_simple_closed : Bool

/-- Represents a point in 2D space -/
def Point := ℝ × ℝ

/-- Checks if a point is interior to a polygon -/
def is_interior (p : Point) (poly : Polygon) : Bool :=
  sorry

/-- Checks if a point can see a side of the polygon -/
def can_see_side (p : Point) (side : Point × Point) (poly : Polygon) : Bool :=
  sorry

/-- The main theorem stating the existence of a polygon with the required properties -/
theorem exists_special_polygon : 
  ∃ (poly : Polygon), 
    (∀ (side1 side2 : Point × Point), 
      side1 ∈ (poly.vertices.zip (poly.vertices.tail!)) → 
      side2 ∈ (poly.vertices.zip (poly.vertices.tail!)) → 
      side1 ≠ side2 → 
      ∃ (p : Point), is_interior p poly ∧ can_see_side p side1 poly ∧ can_see_side p side2 poly) ∧
    (¬∃ (p : Point), is_interior p poly ∧ 
      ∀ (side : Point × Point), side ∈ (poly.vertices.zip (poly.vertices.tail!)) → 
        can_see_side p side poly) :=
  by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_special_polygon_l1066_106604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_units_for_profit_max_profit_during_epidemic_l1066_106695

-- Define the production cost function
noncomputable def Q (x : ℕ+) : ℝ := 5 + 135 / (x.val + 1 : ℝ)

-- Define the profit function without epidemic constraints
noncomputable def profit (x : ℕ+) : ℝ := (10 - Q x) * x.val - 1.8

-- Define the profit function with epidemic constraints
noncomputable def profit_epidemic (x : ℕ+) : ℝ := 
  if x.val ≤ 60 then (10 - Q x) * x.val - 1.8
  else (10 - Q x) * 60 - 0.01 * (x.val - 60) - 1.8

-- Theorem for minimum number of units for profitability
theorem min_units_for_profit : 
  ∀ x : ℕ+, x.val ≥ 63 → profit x > 0 ∧ ∀ y : ℕ+, y.val < 63 → profit y ≤ 0 :=
sorry

-- Theorem for maximum profit output level during epidemic
theorem max_profit_during_epidemic : 
  ∀ x : ℕ+, x.val ≤ 100 → profit_epidemic x ≤ profit_epidemic ⟨89, by norm_num⟩ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_units_for_profit_max_profit_during_epidemic_l1066_106695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pirate_treasure_probability_l1066_106653

theorem pirate_treasure_probability : 
  (Nat.choose 8 4 : ℚ) * (1/5 : ℚ)^4 * ((1 - (1/5 + 1/10)) : ℚ)^4 = 33614/1250000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pirate_treasure_probability_l1066_106653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_cubed_is_60_l1066_106654

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := 
  Nat.choose n k

-- Define the expression
noncomputable def expression (x : ℝ) : ℝ := 
  (2 * x - 1 / (2 * Real.sqrt x)) ^ 6

-- Theorem statement
theorem coefficient_of_x_cubed_is_60 :
  ∃ (c : ℝ), c = 60 ∧ 
    ∀ (x : ℝ), x > 0 → ∃ (r : ℝ), expression x = c * x^3 + x^4 * r := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_cubed_is_60_l1066_106654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_sequence_properties_l1066_106652

-- Define an exponential sequence
def is_exponential_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ r ≠ 1 ∧ a 1 = r ∧ ∀ n m : ℕ, a (n + m) = a n * a m

-- Define the sequence a_n
noncomputable def a_n (n : ℕ) : ℝ := 5 * 3^(n - 1)

-- Define the sequence b_n
noncomputable def b_n (n : ℕ) : ℝ := 4^n

-- Define the sequence c_n based on the recurrence relation
noncomputable def c_n : ℕ → ℝ
| 0 => 1/2  -- Added case for 0
| 1 => 1/2
| (n+2) => 1 / (2 * c_n (n+1) + 3)

-- Define the inverse sequence of c_n plus 1
noncomputable def inv_c_n_plus_1 (n : ℕ) : ℝ := 1 / c_n n + 1

-- Define the sequence d_n
noncomputable def d_n (a : ℕ) (n : ℕ) : ℝ := ((a + 1 : ℝ) / (a + 2 : ℝ)) ^ n

theorem exponential_sequence_properties :
  (¬ is_exponential_sequence a_n) ∧
  (is_exponential_sequence b_n) ∧
  (is_exponential_sequence inv_c_n_plus_1) ∧
  (∀ a : ℕ, ∀ u v w : ℕ, u < v → v < w →
    ¬(2 * d_n a v = d_n a u + d_n a w)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_sequence_properties_l1066_106652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2010_of_8_eq_8_l1066_106615

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := 
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Function f as defined in the problem -/
def f (n : ℕ) : ℕ := sumOfDigits (n^2 + 1)

/-- Recursive definition of f_k -/
def f_k : ℕ → ℕ → ℕ
  | 0, n => n
  | 1, n => f n
  | k+1, n => f (f_k k n)

theorem f_2010_of_8_eq_8 : f_k 2010 8 = 8 := by sorry

#eval f_k 2010 8  -- This will evaluate the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2010_of_8_eq_8_l1066_106615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sugar_water_mixture_concentration_l1066_106680

theorem sugar_water_mixture_concentration : 
  let solution1_weight : ℝ := 200
  let solution1_concentration : ℝ := 0.25
  let solution2_weight : ℝ := 300
  let solution2_sugar : ℝ := 60
  let total_weight := solution1_weight + solution2_weight
  let total_sugar := solution1_weight * solution1_concentration + solution2_sugar
  (total_sugar / total_weight) = 0.22 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sugar_water_mixture_concentration_l1066_106680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_T_l1066_106683

theorem value_of_T (x y : ℝ) (T : ℝ) 
  (h1 : (2 : ℝ)^x = 196) 
  (h2 : (7 : ℝ)^y = 196) 
  (h3 : T = 1/x + 1/y) : 
  T = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_T_l1066_106683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_trig_functions_l1066_106690

theorem symmetry_of_trig_functions (a : ℝ) : 
  (∀ x, Real.sin (2*x - π/3) = Real.cos (2*(2*a - x) + 2*π/3)) → 
  a = π/24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_trig_functions_l1066_106690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_approx_l1066_106649

/-- Represents a rhombus with given side length and one diagonal length -/
structure Rhombus where
  side : ℝ
  diagonal1 : ℝ

/-- Calculates the area of a rhombus -/
noncomputable def rhombusArea (r : Rhombus) : ℝ :=
  let diagonal2 := 2 * Real.sqrt (r.side^2 - (r.diagonal1/2)^2)
  (r.diagonal1 * diagonal2) / 2

/-- Theorem stating the area of a specific rhombus -/
theorem rhombus_area_approx :
  let r : Rhombus := { side := 25, diagonal1 := 10 }
  ∃ ε > 0, |rhombusArea r - 244.9| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_approx_l1066_106649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_segments_equal_l1066_106682

/-- A tetrahedron with right angles at one vertex -/
structure RightAngledTetrahedron where
  x : ℝ
  y : ℝ
  z : ℝ
  x_pos : 0 < x
  y_pos : 0 < y
  z_pos : 0 < z

/-- The length of a segment connecting midpoints of opposite edges -/
noncomputable def midpoint_segment_length (t : RightAngledTetrahedron) : ℝ :=
  (1 / 2) * Real.sqrt (t.x^2 + t.y^2 + t.z^2)

/-- Theorem: All segments connecting midpoints of opposite edges have equal length -/
theorem midpoint_segments_equal (t : RightAngledTetrahedron) :
  ∀ (s₁ s₂ : ℝ), s₁ = midpoint_segment_length t → s₂ = midpoint_segment_length t → s₁ = s₂ := by
  sorry

#check midpoint_segments_equal

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_segments_equal_l1066_106682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_example_l1066_106676

/-- The distance from a point to a line in 2D space -/
noncomputable def distancePointToLine (px py : ℝ) (a b c : ℝ) : ℝ :=
  |a * px + b * py + c| / Real.sqrt (a^2 + b^2)

/-- Theorem: The distance from (1,1) to the line x+y-1=0 is √2/2 -/
theorem distance_point_to_line_example : 
  distancePointToLine 1 1 1 1 (-1) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_example_l1066_106676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_l1066_106655

theorem number_of_subsets (M : Finset (Fin 4)) (N : Set (Finset (Fin 4))) :
  (M = {0, 1, 2, 3}) →
  (N = {p | p ⊆ M}) →
  Finset.card (Finset.powerset M) = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_l1066_106655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_P_on_same_line_l1066_106606

-- Define the circle
variable (r : ℝ) (h_r : r > 0)
def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = r^2}

-- Define points on the circle
def A (r : ℝ) : ℝ × ℝ := (-r, 0)
def B (r : ℝ) : ℝ × ℝ := (r, 0)
def O : ℝ × ℝ := (0, 0)

-- Fixed point M on the circle
variable (α : ℝ)
noncomputable def M (r α : ℝ) : ℝ × ℝ := (r * Real.cos α, r * Real.sin α)

-- Variable point Q on the circle
variable (β : ℝ)
noncomputable def Q (r β : ℝ) : ℝ × ℝ := (r * Real.cos β, r * Real.sin β)

-- Intersection point K
noncomputable def K (r α β : ℝ) : ℝ × ℝ := (r * Real.sin (α - β) / (Real.sin α - Real.sin β), 0)

-- Point P
noncomputable def P (r α β : ℝ) : ℝ × ℝ := (
  r * Real.sin (α - β) / (Real.sin α - Real.sin β),
  (Real.sin β / (Real.cos β - 1)) * (r * Real.sin (α - β) / (Real.sin α - Real.sin β) - r)
)

-- The line on which all P points lie
noncomputable def LocusLine (r α : ℝ) (p : ℝ × ℝ) : Prop :=
  p.1 + Real.tan (α / 2) * p.2 + r = 0

-- Theorem statement
theorem all_P_on_same_line (r α : ℝ) (h_r : r > 0) :
  ∀ β, P r α β ∈ {p : ℝ × ℝ | LocusLine r α p} :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_P_on_same_line_l1066_106606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2023_value_l1066_106688

def sequence_a : ℕ → ℚ
  | 0 => 4/5  -- Adding case for 0
  | 1 => 4/5
  | (n+2) => if sequence_a (n+1) < 1/2 then 2 * sequence_a (n+1) else 2 * sequence_a (n+1) - 1

theorem sequence_2023_value :
  sequence_a 2023 = 1/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2023_value_l1066_106688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lineGrowthLimit_l1066_106622

/-- The sum of the infinite series representing the growth of a line -/
noncomputable def lineGrowthSeries : ℝ := 2 + ∑' n, (1 / 5^n) * (1 + Real.sqrt 3)

/-- The theorem stating that the sum of the infinite series equals (10 + √3) / 4 -/
theorem lineGrowthLimit : lineGrowthSeries = (10 + Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lineGrowthLimit_l1066_106622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1066_106614

/-- Theorem about an ellipse with specific properties -/
theorem ellipse_properties (a b : ℝ) (h_ab : a > b) (h_b_pos : b > 0) :
  let e := Real.sqrt 3 / 2
  let area_OAB := a * b / 2
  let C := {(x, y) : ℝ × ℝ | x^2 / a^2 + y^2 / b^2 = 1}
  e = Real.sqrt (1 - b^2 / a^2) →
  area_OAB = 1 →
  (∃ (P : ℝ × ℝ), P ∈ C →
    let A := (a, 0)
    let B := (0, b)
    let M := (0, (P.2 * (P.1 - a)) / P.1)
    let N := ((P.1 * (P.2 - b)) / P.2, 0)
    C = {(x, y) : ℝ × ℝ | x^2 / 4 + y^2 = 1} ∧
    Real.sqrt ((A.1 - N.1)^2 + (A.2 - N.2)^2) * Real.sqrt ((B.1 - M.1)^2 + (B.2 - M.2)^2) = 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1066_106614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jordan_rice_purchase_l1066_106670

/-- Represents the cost and quantity of rice and lentils purchased by Jordan -/
structure Purchase where
  rice_price : ℚ
  lentil_price : ℚ
  total_pounds : ℚ
  total_cost : ℚ

/-- Calculates the amount of rice purchased given the conditions -/
def rice_amount (p : Purchase) : ℚ :=
  (p.total_cost - p.lentil_price * p.total_pounds) / (p.rice_price - p.lentil_price)

/-- Theorem stating that Jordan bought 15 pounds of rice -/
theorem jordan_rice_purchase :
  let p : Purchase := {
    rice_price := 120/100,
    lentil_price := 60/100,
    total_pounds := 30,
    total_cost := 2700/100
  }
  rice_amount p = 15 := by
  -- Proof goes here
  sorry

#eval rice_amount {
  rice_price := 120/100,
  lentil_price := 60/100,
  total_pounds := 30,
  total_cost := 2700/100
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jordan_rice_purchase_l1066_106670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_necessary_not_sufficient_l1066_106693

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

/-- The slope of the line x + y = 0 -/
def slope_line1 : ℝ := -1

/-- The slope of the line x - ay = 0 -/
noncomputable def slope_line2 (a : ℝ) : ℝ := 1 / a

/-- The condition a^2 = 1 -/
def condition (a : ℝ) : Prop := a^2 = 1

/-- The lines x + y = 0 and x - ay = 0 are perpendicular -/
def lines_perpendicular (a : ℝ) : Prop := perpendicular slope_line1 (slope_line2 a)

theorem condition_necessary_not_sufficient :
  (∀ a : ℝ, lines_perpendicular a → condition a) ∧
  ¬(∀ a : ℝ, condition a → lines_perpendicular a) :=
by
  constructor
  · intro a h
    -- Proof that the condition is necessary
    sorry
  · -- Proof that the condition is not sufficient
    sorry

#check condition_necessary_not_sufficient

end NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_necessary_not_sufficient_l1066_106693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_spherical_coords_equivalence_l1066_106644

noncomputable section

/-- Given spherical coordinates -/
def given_coords : ℝ × ℝ × ℝ := (5, 3 * Real.pi / 4, 9 * Real.pi / 5)

/-- Standard spherical coordinates -/
def standard_coords : ℝ × ℝ × ℝ := (5, 7 * Real.pi / 4, Real.pi / 5)

/-- Conditions for standard spherical coordinates -/
def is_standard_spherical (coords : ℝ × ℝ × ℝ) : Prop :=
  let (ρ, θ, φ) := coords
  ρ > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ 0 ≤ φ ∧ φ ≤ Real.pi

/-- Equivalence of spherical coordinates -/
def spherical_equiv (c1 c2 : ℝ × ℝ × ℝ) : Prop :=
  let (ρ1, θ1, φ1) := c1
  let (ρ2, θ2, φ2) := c2
  ρ1 = ρ2 ∧
  (θ1 % (2 * Real.pi) = θ2 % (2 * Real.pi) ∨
   θ1 % (2 * Real.pi) = (θ2 + Real.pi) % (2 * Real.pi)) ∧
  (φ1 = φ2 ∨ φ1 = Real.pi - φ2)

theorem standard_spherical_coords_equivalence :
  is_standard_spherical standard_coords ∧
  spherical_equiv given_coords standard_coords := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_spherical_coords_equivalence_l1066_106644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_polar_neg_three_neg_four_l1066_106609

noncomputable def rectangular_to_polar (x y : ℝ) : ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := if x > 0 then Real.arctan (y / x)
           else if x < 0 && y ≥ 0 then Real.pi + Real.arctan (y / x)
           else if x < 0 && y < 0 then -Real.pi + Real.arctan (y / x)
           else if x = 0 && y > 0 then Real.pi / 2
           else if x = 0 && y < 0 then -Real.pi / 2
           else 0  -- x = 0 and y = 0
  (r, θ)

theorem rectangular_to_polar_neg_three_neg_four :
  let (r, θ) := rectangular_to_polar (-3) (-4)
  r = 5 ∧ θ = Real.pi + Real.arctan (4/3) ∧ r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_polar_neg_three_neg_four_l1066_106609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bounds_of_S_l1066_106638

noncomputable def f (x : ℝ) := (3*x + 2) / (x + 3)

def S : Set ℝ := {y | ∃ x ≥ 1, f x = y}

theorem bounds_of_S :
  ∃ (M m : ℝ),
    (∀ y ∈ S, y ≤ M) ∧
    (∀ ε > 0, ∃ y ∈ S, y > M - ε) ∧
    (∀ y ∈ S, y ≥ m) ∧
    (∀ ε > 0, ∃ y ∈ S, y < m + ε) ∧
    M = 3 ∧
    m = 5/4 ∧
    m ∈ S ∧
    M ∉ S := by
  sorry

#check bounds_of_S

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bounds_of_S_l1066_106638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_special_case_l1066_106647

/-- Ellipse type -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h1 : a > b
  h2 : b > 0

/-- Point type -/
structure Point where
  x : ℝ
  y : ℝ

/-- Function to calculate eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- Theorem statement -/
theorem ellipse_eccentricity_special_case (e : Ellipse) 
  (F₁ F₂ A B D : Point) : 
  (F₁.x = -Real.sqrt (e.a^2 - e.b^2) ∧ F₁.y = 0) →  -- Left focus
  (F₂.x = Real.sqrt (e.a^2 - e.b^2) ∧ F₂.y = 0) →   -- Right focus
  (A.x = F₂.x ∧ B.x = F₂.x) →                       -- A and B on perpendicular line through F₂
  (A.y > 0 ∧ B.y < 0) →                             -- A above x-axis, B below
  (D.x = 0) →                                       -- D on y-axis
  (((A.y - D.y) / (A.x - D.x)) * ((B.y - F₁.y) / (B.x - F₁.x)) = -1) →  -- AD ⊥ F₁B
  eccentricity e = Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_special_case_l1066_106647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_identities_l1066_106699

-- Define a structure for a triangle's angles
structure TriangleAngles where
  α : Real
  β : Real
  γ : Real
  sum_to_pi : α + β + γ = Real.pi

-- Theorem statement
theorem triangle_angle_identities (t : TriangleAngles) :
  (Real.sin (2 * t.α) + Real.sin (2 * t.β) + Real.sin (2 * t.γ) = 4 * Real.sin t.α * Real.sin t.β * Real.sin t.γ) ∧
  (Real.tan (π/2 - t.α/2) + Real.tan (π/2 - t.β/2) + Real.tan (π/2 - t.γ/2) = 
   Real.tan (π/2 - t.α/2) * Real.tan (π/2 - t.β/2) * Real.tan (π/2 - t.γ/2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_identities_l1066_106699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wealth_ratio_is_seven_sixths_l1066_106665

/-- Represents the wealth distribution between two countries X and Y -/
structure WorldWealthDistribution where
  total_population : ℝ
  total_wealth : ℝ
  x_population_percentage : ℝ
  x_wealth_percentage : ℝ
  y_population_percentage : ℝ
  y_wealth_percentage : ℝ
  y_tax_rate : ℝ

/-- Calculates the ratio of wealth per citizen between countries X and Y -/
noncomputable def wealth_ratio (w : WorldWealthDistribution) : ℝ :=
  let x_wealth_per_citizen := (w.x_wealth_percentage * w.total_wealth) / (w.x_population_percentage * w.total_population)
  let y_wealth_per_citizen := ((1 - w.y_tax_rate) * w.y_wealth_percentage * w.total_wealth) / (w.y_population_percentage * w.total_population)
  x_wealth_per_citizen / y_wealth_per_citizen

/-- Theorem stating that the wealth ratio between countries X and Y is 7/6 -/
theorem wealth_ratio_is_seven_sixths (w : WorldWealthDistribution) 
  (h1 : w.x_population_percentage = 0.25)
  (h2 : w.x_wealth_percentage = 0.30)
  (h3 : w.y_population_percentage = 0.35)
  (h4 : w.y_wealth_percentage = 0.40)
  (h5 : w.y_tax_rate = 0.10) :
  wealth_ratio w = 7/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wealth_ratio_is_seven_sixths_l1066_106665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_untransformable_config_k_200_value_k_n_property_l1066_106673

/-- Represents a configuration of +1 and -1 on the vertices of a regular n-gon -/
def Configuration (n : ℕ) := Fin n → Bool

/-- Represents a transformation on a configuration -/
def Transformation (n : ℕ) := Configuration n → Configuration n

/-- The number of possible transformations for a given n -/
noncomputable def T (n : ℕ) : ℕ := sorry

/-- The number of distinct equivalence classes under the transformations -/
noncomputable def K (n : ℕ) : ℕ := 2^n / T n

theorem existence_of_untransformable_config (n : ℕ) (h : n > 2) :
  ∃ (c : Configuration n), ∀ (t : Transformation n), t c ≠ λ _ => true := by sorry

theorem k_200_value : K 200 = 2^80 := by sorry

-- Existential property of K(n)
theorem k_n_property (n : ℕ) :
  ∃ (S : Finset (Configuration n)), 
    (∀ (c1 c2 : Configuration n), c1 ∈ S ∧ c2 ∈ S → 
      (∀ (t : Transformation n), t c1 ≠ c2)) ∧ S.card = K n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_untransformable_config_k_200_value_k_n_property_l1066_106673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_l1066_106632

/-- The cosine function with angular frequency ω and phase φ -/
noncomputable def f (ω φ x : ℝ) : ℝ := Real.cos (ω * x + φ)

/-- The smallest positive period of a periodic function -/
noncomputable def smallest_positive_period (g : ℝ → ℝ) : ℝ := sorry

theorem min_omega_value (ω φ : ℝ) (h_ω : ω > 0) (h_φ : 0 < φ ∧ φ < Real.pi) :
  let T := smallest_positive_period (f ω φ)
  (f ω φ T = Real.sqrt 3 / 2) →
  (f ω φ (Real.pi / 9) = 0) →
  (∀ ω' > 0, (∃ φ' > 0, φ' < Real.pi ∧
    let T' := smallest_positive_period (f ω' φ')
    (f ω' φ' T' = Real.sqrt 3 / 2) ∧
    (f ω' φ' (Real.pi / 9) = 0)) → ω' ≥ ω) →
  ω = 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_l1066_106632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_sequence_general_term_l1066_106618

def a_sequence (n : ℕ) : ℚ :=
  if n = 0 then 1 else 3 / (a_sequence (n - 1) + 3)

theorem a_sequence_general_term (n : ℕ) : a_sequence n = 3 / (n + 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_sequence_general_term_l1066_106618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_workers_theorem_l1066_106687

/-- Calculates the number of additional workers needed to complete a project on time -/
def additional_workers_needed (total_days : ℕ) (initial_workers : ℕ) (days_passed : ℕ) (work_completed_percent : ℚ) : ℕ :=
  let remaining_days := total_days - days_passed
  let remaining_work_percent := 1 - work_completed_percent
  let daily_work_rate := work_completed_percent / days_passed
  let required_daily_rate := remaining_work_percent / remaining_days
  let total_workers_needed := (required_daily_rate / daily_work_rate) * initial_workers
  (total_workers_needed - initial_workers).ceil.toNat

theorem additional_workers_theorem (total_days initial_workers days_passed : ℕ) (work_completed_percent : ℚ) 
    (h1 : total_days = 50)
    (h2 : initial_workers = 70)
    (h3 : days_passed = 25)
    (h4 : work_completed_percent = 2/5) :
  additional_workers_needed total_days initial_workers days_passed work_completed_percent = 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_workers_theorem_l1066_106687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1066_106607

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) - Real.cos (2 * x)

theorem triangle_area (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < Real.pi / 2 →
  0 < B ∧ B < Real.pi / 2 →
  0 < C ∧ C < Real.pi / 2 →
  A + B + C = Real.pi →
  f A = 1 →
  b * c = 8 →
  (1 / 2) * b * c * Real.sin A = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1066_106607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_radius_formula_l1066_106675

-- Define a right triangular prism with pairwise perpendicular lateral edges
structure RightTriangularPrism where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c

-- Define the radius of the circumscribed sphere
noncomputable def circumscribed_sphere_radius (prism : RightTriangularPrism) : ℝ :=
  (1 / 2) * Real.sqrt (prism.a^2 + prism.b^2 + prism.c^2)

-- Theorem statement
theorem circumscribed_sphere_radius_formula (prism : RightTriangularPrism) :
  circumscribed_sphere_radius prism = (1 / 2) * Real.sqrt (prism.a^2 + prism.b^2 + prism.c^2) := by
  -- Unfold the definition of circumscribed_sphere_radius
  unfold circumscribed_sphere_radius
  -- The rest of the proof is trivial as it's just comparing identical expressions
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_radius_formula_l1066_106675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_to_156_l1066_106671

noncomputable def expression : ℝ := 3.76 * Real.sqrt 16.81 * (8.13 + 1.87)

def options : List ℝ := [150, 156, 160, 170, 180]

theorem closest_to_156 : 
  ∀ x ∈ options, |expression - 156| ≤ |expression - x| := by
  sorry

#eval options

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_to_156_l1066_106671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candle_height_ratio_theorem_l1066_106657

/-- The time (in hours) when the first candle is three times the height of the second candle -/
noncomputable def candle_height_ratio_time : ℝ := 40 / 11

theorem candle_height_ratio_theorem (h₁ h₂ : ℝ → ℝ) :
  (∀ t, h₁ t = 1 - t / 5) →  -- First candle height function
  (∀ t, h₂ t = 1 - t / 4) →  -- Second candle height function
  (h₁ candle_height_ratio_time = 3 * h₂ candle_height_ratio_time) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candle_height_ratio_theorem_l1066_106657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_triangle_area_l1066_106639

/-- The area of the region inside a circle but outside an equilateral triangle, 
    both centered at the same point. -/
theorem circle_triangle_area (r s : ℝ) : r = 3 → s = 6 → 
  (π * r^2) - (Real.sqrt 3 / 4 * s^2) = 9 * (π - Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_triangle_area_l1066_106639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_model_properties_l1066_106678

/-- Represents the sales model of a clothing store -/
structure SalesModel where
  initialSales : ℕ
  initialProfit : ℕ
  salesIncrease : ℕ
  priceReduction : ℝ

/-- Calculates the number of pieces sold after price reduction -/
def piecesAfterReduction (model : SalesModel) : ℝ :=
  model.initialSales + model.salesIncrease * model.priceReduction

/-- Represents the daily profit function -/
def dailyProfit (model : SalesModel) : ℝ :=
  (model.initialProfit - model.priceReduction) * (piecesAfterReduction model)

/-- Theorem stating the properties of the sales model -/
theorem sales_model_properties (model : SalesModel) 
  (h1 : model.initialSales = 20)
  (h2 : model.initialProfit = 60)
  (h3 : model.salesIncrease = 2) :
  (piecesAfterReduction model = 20 + 2 * model.priceReduction) ∧
  (∃ x y : ℝ, x ≠ y ∧ x ∈ ({20, 30} : Set ℝ) ∧ y ∈ ({20, 30} : Set ℝ) ∧ 
    dailyProfit {initialSales := 20, initialProfit := 60, salesIncrease := 2, priceReduction := x} = 2400 ∧
    dailyProfit {initialSales := 20, initialProfit := 60, salesIncrease := 2, priceReduction := y} = 2400) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_model_properties_l1066_106678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_a2015_a2016_l1066_106641

noncomputable def vector_a (k : ℤ) : ℝ × ℝ := 
  (Real.cos (k * Real.pi / 6), Real.sin (k * Real.pi / 6) + Real.cos (k * Real.pi / 6))

theorem dot_product_a2015_a2016 :
  let a2015 := vector_a 2015
  let a2016 := vector_a 2016
  (a2015.1 * a2016.1 + a2015.2 * a2016.2) = Real.sqrt 3 - 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_a2015_a2016_l1066_106641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_characterization_l1066_106619

/-- Given a complex number z defined in terms of a real number m, 
    this theorem characterizes when z is real, imaginary, pure imaginary, or zero. -/
theorem complex_number_characterization (m : ℝ) : 
  (∀ z : ℂ, z = (m^2 - 3*m : ℝ) + (m^2 - m - 6 : ℝ) * I →
    (z.im = 0 ↔ m = 3 ∨ m = -2) ∧ 
    (z.re ≠ 0 ∧ z.im ≠ 0 ↔ m ≠ 3 ∧ m ≠ -2) ∧
    (z.re = 0 ∧ z.im ≠ 0 ↔ m = 0) ∧
    (z = 0 ↔ m = 3)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_characterization_l1066_106619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_person_paint_time_l1066_106684

/-- Time it takes Kathleen to paint a room -/
def kathleen_time : ℚ := 2

/-- Time it takes Kathleen and the other person to paint both rooms together -/
def combined_time : ℚ := 2857142857142857 / 1000000000000000

/-- Calculates the time it takes the second person to paint a room -/
noncomputable def second_person_time (kt : ℚ) (ct : ℚ) : ℚ :=
  2 * ct / (2 - ct / kt)

theorem second_person_paint_time :
  second_person_time kathleen_time combined_time = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_person_paint_time_l1066_106684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Na_approx_l1066_106631

/-- The molar mass of sodium (Na) in g/mol -/
noncomputable def molar_mass_Na : ℝ := 22.99

/-- The molar mass of chlorine (Cl) in g/mol -/
noncomputable def molar_mass_Cl : ℝ := 35.45

/-- The molar mass of oxygen (O) in g/mol -/
noncomputable def molar_mass_O : ℝ := 16.00

/-- The molar mass of sodium hypochlorite (NaClO) in g/mol -/
noncomputable def molar_mass_NaClO : ℝ := molar_mass_Na + molar_mass_Cl + molar_mass_O

/-- The mass percentage of sodium (Na) in sodium hypochlorite (NaClO) -/
noncomputable def mass_percentage_Na : ℝ := (molar_mass_Na / molar_mass_NaClO) * 100

theorem mass_percentage_Na_approx :
  abs (mass_percentage_Na - 30.89) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Na_approx_l1066_106631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_union_equals_reals_l1066_106611

-- Define the real number a
variable (a : ℝ)

-- Define the domain of f
def domain (a : ℝ) (x : ℝ) : Prop := x < a ∨ x > a

-- Define the sets M and N
def M (a : ℝ) (f : ℝ → ℝ) : Set ℝ := {x | domain a x ∧ f x ≥ 0}
def N (a : ℝ) (f : ℝ → ℝ) : Set ℝ := {x | domain a x ∧ f x < 0}

-- The theorem to prove
theorem complement_union_equals_reals (a : ℝ) (f : ℝ → ℝ) : 
  (Set.univ \ M a f) ∪ (Set.univ \ N a f) = Set.univ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_union_equals_reals_l1066_106611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_term_coefficient_l1066_106663

theorem third_term_coefficient : 
  (5 : ℚ) / 2 = (Finset.range 6).sum (λ k => 
    (Nat.choose 5 k : ℚ) * (1/2)^k * (1 : ℚ)^(5-k)) * (1/2)^2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_term_coefficient_l1066_106663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_otimes_5_18_l1066_106674

/-- Custom binary operation ⊗ for positive integers -/
def otimes : ℕ+ → ℕ+ → ℕ+ := sorry

/-- First axiom of ⊗ operation -/
axiom otimes_axiom1 (a b : ℕ+) : otimes (a ^ 2 * b) b = a * otimes b b

/-- Second axiom of ⊗ operation -/
axiom otimes_axiom2 (a : ℕ+) : otimes (otimes a 1) a = otimes a 1

/-- Base case for ⊗ operation -/
axiom otimes_base : otimes 1 1 = 1

/-- Theorem to prove -/
theorem otimes_5_18 : otimes 5 18 = 8100 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_otimes_5_18_l1066_106674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_difference_is_ten_minutes_l1066_106600

-- Define the speeds for Minnie and Penny
noncomputable def minnie_flat_speed : ℝ := 25
noncomputable def minnie_downhill_speed : ℝ := 35
noncomputable def minnie_uphill_speed : ℝ := 10
noncomputable def penny_flat_speed : ℝ := 35
noncomputable def penny_downhill_speed : ℝ := 45
noncomputable def penny_uphill_speed : ℝ := 15

-- Define the distances for each segment
noncomputable def ab_distance : ℝ := 15
noncomputable def bd_distance_minnie : ℝ := 20
noncomputable def dc_distance : ℝ := 25
noncomputable def cb_distance : ℝ := 20
noncomputable def bd_distance_penny : ℝ := 15
noncomputable def da_distance : ℝ := 25

-- Calculate times for Minnie's route
noncomputable def minnie_ab_time : ℝ := ab_distance / minnie_uphill_speed
noncomputable def minnie_bd_time : ℝ := bd_distance_minnie / minnie_downhill_speed
noncomputable def minnie_dc_time : ℝ := dc_distance / minnie_flat_speed

-- Calculate times for Penny's route
noncomputable def penny_cb_time : ℝ := cb_distance / penny_uphill_speed
noncomputable def penny_bd_time : ℝ := bd_distance_penny / penny_downhill_speed
noncomputable def penny_da_time : ℝ := da_distance / penny_flat_speed

-- Calculate total times
noncomputable def minnie_total_time : ℝ := minnie_ab_time + minnie_bd_time + minnie_dc_time
noncomputable def penny_total_time : ℝ := penny_cb_time + penny_bd_time + penny_da_time

-- Theorem statement
theorem time_difference_is_ten_minutes :
  (minnie_total_time - penny_total_time) * 60 = 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_difference_is_ten_minutes_l1066_106600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_value_l1066_106613

theorem max_k_value (x₁ x₂ x₃ x₄ : ℝ) (h₁ : x₁ > x₂) (h₂ : x₂ > x₃) (h₃ : x₃ > x₄) (h₄ : x₄ > 0) :
  ∃ k_max : ℝ, k_max = 9 ∧ 
  (∀ k : ℝ, (Real.log (x₁/x₂) + Real.log (x₂/x₃) + Real.log (x₃/x₄) ≥ k * Real.log (x₁/x₄)) → k ≤ k_max) ∧
  (Real.log (x₁/x₂) + Real.log (x₂/x₃) + Real.log (x₃/x₄) ≥ k_max * Real.log (x₁/x₄)) :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_value_l1066_106613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_approx_l1066_106664

-- Define the speed of the train in km/hr
noncomputable def train_speed_kmh : ℝ := 72

-- Define the time taken to cross the pole in seconds
noncomputable def crossing_time : ℝ := 4.99960003199744

-- Define the conversion factor from km/hr to m/s
noncomputable def kmh_to_ms : ℝ := 1000 / 3600

-- Define the function to calculate the train length
noncomputable def train_length : ℝ := train_speed_kmh * kmh_to_ms * crossing_time

-- Theorem statement
theorem train_length_approx :
  ∃ ε > 0, |train_length - 99.992| < ε :=
by
  -- We'll use 0.001 as our epsilon
  use 0.001
  -- Split the goal into two parts
  constructor
  · -- Prove ε > 0
    norm_num
  · -- Prove |train_length - 99.992| < ε
    -- This part would require actual computation, which is complex in Lean
    -- For now, we'll use sorry to skip the proof
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_approx_l1066_106664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lindy_speed_is_ten_l1066_106601

/-- Represents the scenario with Jack, Christina, and Lindy --/
structure Scenario where
  distance : ℝ  -- Initial distance between Jack and Christina
  jack_speed : ℝ  -- Jack's walking speed
  christina_speed : ℝ  -- Christina's walking speed
  lindy_distance : ℝ  -- Total distance Lindy travels

/-- Calculates Lindy's speed given a scenario --/
noncomputable def lindy_speed (s : Scenario) : ℝ :=
  s.lindy_distance / (s.distance / (s.jack_speed + s.christina_speed))

/-- Theorem stating that Lindy's speed is 10 feet per second in the given scenario --/
theorem lindy_speed_is_ten :
  let s : Scenario := {
    distance := 150,
    jack_speed := 7,
    christina_speed := 8,
    lindy_distance := 100
  }
  lindy_speed s = 10 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lindy_speed_is_ten_l1066_106601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_implies_a_greater_than_two_l1066_106608

open Real

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := 2 * x^2 + b * x - a * log x

-- State the theorem
theorem f_negative_implies_a_greater_than_two :
  ∀ a : ℝ,
  (∀ b ∈ Set.Icc (-3) (-2),
    ∃ x ∈ Set.Ioo 1 (Real.exp 2),
      f a b x < 0) →
  a > 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_implies_a_greater_than_two_l1066_106608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_proof_l1066_106617

theorem triangle_proof (A B C : ℝ) (a b c : ℝ) :
  -- Given conditions
  0 < B ∧ B < π/2 →  -- B is an acute angle
  b = 2 →  -- Given condition
  -- Vectors are parallel
  2 * Real.sin B * (2 * Real.cos (B/2) ^ 2 - 1) + Real.sqrt 3 * Real.cos (2*B) = 0 →
  -- Theorem statements
  B = π/3 ∧
  (∀ S : ℝ, S = 1/2 * a * c * Real.sin B → S ≤ Real.sqrt 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_proof_l1066_106617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_comparison_theorem_l1066_106685

noncomputable def M : ℝ := (10^2000 + 1) / (10^2001 + 1)
noncomputable def N : ℝ := (10^2001 + 1) / (10^2002 + 1)
noncomputable def P : ℝ := (10^2000 + 9) / (10^2001 + 100)
noncomputable def Q : ℝ := (10^2001 + 9) / (10^2002 + 100)

theorem comparison_theorem : M > N ∧ P < Q := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_comparison_theorem_l1066_106685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_m_l1066_106624

def f (d : ℤ) (x : ℝ) : ℝ := 4 * x + d

theorem intersection_point_m (d : ℤ) (m : ℤ) :
  (∃ (f_inv : ℝ → ℝ), Function.RightInverse (f d) f_inv ∧ Function.LeftInverse (f d) f_inv) →
  f d 7 = m →
  f d m = 7 →
  m = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_m_l1066_106624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_conditional_probability_l1066_106629

-- Define the sample space
def Ω : Type := Fin 6 × Fin 6 × Fin 6

-- Define events A and B
def A : Set Ω := {ω | ω.1 ≠ ω.2.1 ∧ ω.1 ≠ ω.2.2 ∧ ω.2.1 ≠ ω.2.2}
def B : Set Ω := {ω | ω.1 = 2 ∨ ω.2.1 = 2 ∨ ω.2.2 = 2}

-- Define the probability measure
noncomputable def P : Set Ω → ℝ := sorry

-- State the theorem
theorem dice_conditional_probability :
  P (A ∩ B) / P B = 60 / 91 ∧ P (A ∩ B) / P A = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_conditional_probability_l1066_106629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_greater_z_greater_y_l1066_106686

theorem x_greater_z_greater_y 
  (x y z : ℝ) 
  (hx : x^2022 = Real.exp 1)
  (hy : (2022 : ℝ)^y = 2023)
  (hz : 2022*z = 2023) : 
  x > z ∧ z > y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_greater_z_greater_y_l1066_106686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_n_for_arithmetic_sequence_l1066_106628

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def sum_of_arithmetic_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (a 1 + a n) / 2

theorem max_n_for_arithmetic_sequence (a : ℕ → ℝ) (h_arith : arithmetic_sequence a) 
    (h_a9 : a 9 = 1) (h_S8 : sum_of_arithmetic_sequence a 8 = 0) :
    ∀ n : ℕ, sum_of_arithmetic_sequence a n ≠ 0 → n ≤ 9 :=
  by
    sorry

#check max_n_for_arithmetic_sequence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_n_for_arithmetic_sequence_l1066_106628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_points_count_l1066_106689

/-- A circle with center O and radius r -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ  -- ax + by + c = 0

/-- Three parallel tangents to the circle -/
structure ParallelTangents where
  line1 : Line
  line2 : Line
  line3 : Line

/-- Distances from circle center to tangents -/
noncomputable def tangent_distances (c : Circle) (t : ParallelTangents) : ℝ × ℝ × ℝ := by
  let d : ℝ := sorry  -- Distance to the nearest tangent
  exact (d, d + c.radius, d + 2 * c.radius)

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between a point and a line -/
noncomputable def distance_point_line (p : Point) (l : Line) : ℝ := sorry

/-- Distance between a point and a circle -/
noncomputable def distance_point_circle (p : Point) (c : Circle) : ℝ := sorry

/-- A point is equidistant from the circle and all tangents -/
def is_equidistant (p : Point) (c : Circle) (t : ParallelTangents) : Prop :=
  distance_point_circle p c = distance_point_line p t.line1 ∧
  distance_point_circle p c = distance_point_line p t.line2 ∧
  distance_point_circle p c = distance_point_line p t.line3

/-- The main theorem -/
theorem equidistant_points_count (c : Circle) (t : ParallelTangents) :
  ∃! (p1 p2 : Point), is_equidistant p1 c t ∧ is_equidistant p2 c t ∧ p1 ≠ p2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_points_count_l1066_106689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_ratio_l1066_106645

/-- Predicate for arithmetic sequence -/
def is_arithmetic_seq (x y z w : ℝ) : Prop :=
  y - x = z - y ∧ z - y = w - z

/-- Predicate for geometric sequence -/
def is_geometric_seq (x y z w v : ℝ) : Prop :=
  y / x = z / y ∧ z / y = w / z ∧ w / z = v / w

/-- Given an arithmetic sequence and a geometric sequence with specific properties, 
    prove that (a₁ - a₂) / b₂ = -1/2 -/
theorem sequence_ratio (a₁ a₂ b₁ b₂ b₃ : ℝ) : 
  (is_arithmetic_seq 1 a₁ a₂ 4) → 
  (is_geometric_seq 1 b₁ b₂ b₃ 4) →
  (a₁ - a₂) / b₂ = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_ratio_l1066_106645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1066_106666

noncomputable def f (x : ℝ) := Real.cos x * Real.sin (2 * x)

theorem f_properties :
  (∀ x, f (2 * Real.pi - x) + f x = 0) ∧
  (∀ x, f (Real.pi - x) = f x) ∧
  (∀ x, f (-x) = -f x) ∧
  (∀ x, f (x + 2 * Real.pi) = f x) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1066_106666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_f_two_zeros_l1066_106694

-- Define the function f(x) with parameter a
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * (x + 2)

-- Theorem for the monotonicity of f(x) when a = 1
theorem f_monotonicity :
  (∀ x₁ x₂, x₁ < x₂ ∧ x₂ < 0 → f 1 x₁ > f 1 x₂) ∧
  (∀ x₃ x₄, 0 < x₃ ∧ x₃ < x₄ → f 1 x₃ < f 1 x₄) :=
sorry

-- Theorem for the range of a when f(x) has two zeros
theorem f_two_zeros (a : ℝ) :
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) ↔ a > Real.exp (-1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_f_two_zeros_l1066_106694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interesting_numbers_perfect_square_product_l1066_106610

/-- A natural number is interesting if it can be factored into natural factors, each less than 30. -/
def is_interesting (n : ℕ) : Prop :=
  ∃ (factors : List ℕ), n = factors.prod ∧ ∀ f, f ∈ factors → f < 30 ∧ f > 1

/-- Given a set of 10,000 interesting numbers, there exist two whose product is a perfect square. -/
theorem interesting_numbers_perfect_square_product 
  (S : Finset ℕ) 
  (h_size : S.card = 10000) 
  (h_interesting : ∀ n, n ∈ S → is_interesting n) : 
  ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ ∃ k : ℕ, a * b = k ^ 2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interesting_numbers_perfect_square_product_l1066_106610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_properties_of_sine_function_l1066_106620

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin x

theorem properties_of_sine_function :
  let C := {(x, y) | ∃ (x : ℝ), y = f x}
  (∀ (x y : ℝ), (x, y) ∈ C ↔ (-x, y) ∈ C) ∧ 
  (∃ (a b : ℝ), ∀ (x y : ℝ), (x, y) ∈ C ↔ (2*a - x, 2*b - y) ∈ C) ∧
  (∀ (x₁ x₂ : ℝ), -π/2 < x₁ ∧ x₁ < x₂ ∧ x₂ < π/2 → f x₁ < f x₂) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_properties_of_sine_function_l1066_106620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_increasing_implies_a_range_l1066_106643

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then a^x else (6 - a) * x

theorem function_increasing_implies_a_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) > 0) →
  a ∈ Set.Icc 3 6 ∧ a ≠ 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_increasing_implies_a_range_l1066_106643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_decreasing_range_l1066_106697

/-- A function f : ℝ → ℝ is decreasing if for all x y, x < y implies f x > f y -/
def DecreasingOn (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

/-- The exponential function with base (1 - 2a) -/
noncomputable def f (a : ℝ) : ℝ → ℝ := fun x ↦ (1 - 2 * a) ^ x

theorem exponential_decreasing_range (a : ℝ) :
  DecreasingOn (f a) → 0 < a ∧ a < 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_decreasing_range_l1066_106697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_is_250_extra_interest_earned_l1066_106633

/-- Simple interest calculation -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  (principal * rate * time) / 100

/-- Proof that the difference in interest is always 250 -/
theorem interest_difference_is_250 (r : ℝ) : 
  simple_interest 2500 (r + 2) 5 - simple_interest 2500 r 5 = 250 := by
  sorry

/-- Main theorem stating the result -/
theorem extra_interest_earned : ∃ (amount : ℝ), 
  ∀ (r : ℝ), simple_interest 2500 (r + 2) 5 - simple_interest 2500 r 5 = amount ∧ amount = 250 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_is_250_extra_interest_earned_l1066_106633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_f_is_correct_stating_f_is_least_l1066_106636

/-- 
Given an integer n ≥ 3, f(n) is the least natural number such that 
every subset A ⊂ {1, 2, ..., n} with f(n) elements contains three pairwise coprime elements.
-/
def f (n : ℕ) : ℕ :=
  n / 2 + n / 3 - n / 6 + 1

/-- 
Theorem stating that f(n) is the correct function for the given property.
-/
theorem f_is_correct (n : ℕ) (h : n ≥ 3) :
  ∀ (A : Finset ℕ), A ⊆ Finset.range n → A.card = f n →
    ∃ (x y z : ℕ), x ∈ A ∧ y ∈ A ∧ z ∈ A ∧ 
      Nat.Coprime x y ∧ Nat.Coprime y z ∧ Nat.Coprime x z :=
by sorry

/-- 
Theorem stating that f(n) is the least natural number with the given property.
-/
theorem f_is_least (n : ℕ) (h : n ≥ 3) :
  ∀ (m : ℕ), m < f n →
    ∃ (A : Finset ℕ), A ⊆ Finset.range n ∧ A.card = m ∧
      ∀ (x y z : ℕ), x ∈ A → y ∈ A → z ∈ A →
        ¬(Nat.Coprime x y ∧ Nat.Coprime y z ∧ Nat.Coprime x z) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_f_is_correct_stating_f_is_least_l1066_106636
