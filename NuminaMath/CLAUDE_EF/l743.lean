import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_tile_2018x2020_with_5x8_l743_74348

theorem cannot_tile_2018x2020_with_5x8 : ¬ ∃ (tiling : Set (Nat × Nat × Nat × Nat)),
  (∀ (x y w h : Nat), (x, y, w, h) ∈ tiling → w = 5 ∧ h = 8) ∧
  (∀ (i j : Nat), i < 2018 ∧ j < 2020 →
    ∃! (x y w h : Nat), (x, y, w, h) ∈ tiling ∧ x ≤ i ∧ i < x + w ∧ y ≤ j ∧ j < y + h) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_tile_2018x2020_with_5x8_l743_74348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_LN_l743_74330

theorem right_triangle_LN (L M N : ℝ × ℝ) (sinN : ℝ) (LM : ℝ) :
  (N.1 - M.1)^2 + (N.2 - M.2)^2 = (L.1 - M.1)^2 + (L.2 - M.2)^2 →  -- Right angle at M
  (L.2 - M.2) * (N.1 - M.1) = (L.1 - M.1) * (N.2 - M.2) →  -- Right angle at M
  sinN = 4/5 →
  LM = 25 →
  sinN = LM / Real.sqrt ((L.1 - N.1)^2 + (L.2 - N.2)^2) →
  Real.sqrt ((L.1 - N.1)^2 + (L.2 - N.2)^2) = 31.25 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_LN_l743_74330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_cubed_l743_74324

/-- The expression to be simplified -/
def expression (x : ℝ) : ℝ :=
  5 * (x^2 - 2*x^3 + x) + 2 * (x + 3*x^3 - 2*x^2 + 2*x^5 + 2*x^3) - 7 * (1 + 2*x - 5*x^3 - x^2)

/-- The coefficient of x^3 in the simplified expression -/
theorem coefficient_of_x_cubed : 
  (deriv (deriv (deriv expression)) 0) / 6 = 35 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_cubed_l743_74324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_focus_l743_74313

/-- Parabola type -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Point type -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Check if a point lies on a parabola -/
def onParabola (p : Point) (c : Parabola) : Prop :=
  (p.y - c.k)^2 = 4 * c.a * (p.x - c.h)

/-- The focus of a parabola -/
def focus (c : Parabola) : Point :=
  ⟨c.h + c.a, c.k⟩

/-- Theorem: Distance from point A to the focus of parabola C -/
theorem distance_to_focus 
  (c : Parabola) 
  (p a b : Point) 
  (h1 : c.a = 1 ∧ c.h = 0 ∧ c.k = 0) 
  (h2 : p.x = -2 ∧ p.y = 0) 
  (h3 : onParabola a c ∧ onParabola b c) 
  (h4 : distance p a = (1/2) * distance a b) :
  distance a (focus c) = 5/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_focus_l743_74313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l743_74376

-- Define the function f(x) = e^x + x - 2
noncomputable def f (x : ℝ) : ℝ := Real.exp x + x - 2

-- Theorem statement
theorem root_in_interval : ∃ x ∈ Set.Ioo 0 1, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l743_74376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_specific_l743_74351

/-- The radius of a circle inscribed in a rhombus with diagonals d₁ and d₂ -/
noncomputable def inscribed_circle_radius (d₁ d₂ : ℝ) : ℝ :=
  (d₁ * d₂) / (4 * Real.sqrt ((d₁ / 2) ^ 2 + (d₂ / 2) ^ 2))

/-- Theorem: The radius of a circle inscribed in a rhombus with diagonals 12 and 30 is 30/√29 -/
theorem inscribed_circle_radius_specific : inscribed_circle_radius 12 30 = 30 / Real.sqrt 29 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_specific_l743_74351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_ratio_l743_74345

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y^2 = 2px -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Represents a line passing through a point at a given angle -/
structure Line where
  point : Point
  angle : ℝ

/-- Function to calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem stating the ratio of distances from intersection points to focus -/
theorem parabola_intersection_ratio 
  (C : Parabola) 
  (F : Point) 
  (l : Line) 
  (A B : Point) :
  F.x = C.p / 2 ∧ F.y = 0 ∧  -- Focus condition
  l.point = F ∧ l.angle = π / 3 ∧  -- Line condition
  A.y^2 = 2 * C.p * A.x ∧  -- A on parabola
  B.y^2 = 2 * C.p * B.x ∧  -- B on parabola
  A.y > 0 ∧ A.x > 0 ∧  -- A in first quadrant
  B.y < 0 ∧ B.x > 0  -- B in fourth quadrant
  →
  distance A F / distance B F = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_ratio_l743_74345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boy_swims_upstream_21km_l743_74368

/-- Represents the distance swam upstream by a boy given his swimming conditions -/
noncomputable def distance_upstream (downstream_distance : ℝ) (time : ℝ) (still_water_speed : ℝ) : ℝ :=
  let stream_speed := (downstream_distance / time - still_water_speed) / 2
  (still_water_speed - stream_speed) * time

/-- Theorem stating that under the given conditions, the boy swims 21 km upstream -/
theorem boy_swims_upstream_21km (downstream_distance : ℝ) (time : ℝ) (still_water_speed : ℝ)
  (h1 : downstream_distance = 91)
  (h2 : time = 7)
  (h3 : still_water_speed = 8) :
  distance_upstream downstream_distance time still_water_speed = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boy_swims_upstream_21km_l743_74368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_maximized_at_72_optimal_investment_correct_l743_74369

/-- Total profit function for the bicycle company -/
noncomputable def f (x : ℝ) : ℝ := 3 * Real.sqrt (2 * x) - (1/4) * x + 26

/-- The domain of the function -/
def is_in_domain (x : ℝ) : Prop := 40 ≤ x ∧ x ≤ 80

/-- Statement: The profit function reaches its maximum at x = 72 -/
theorem profit_maximized_at_72 :
  ∀ x : ℝ, is_in_domain x → f 72 ≥ f x := by
  sorry

/-- The optimal investment amounts -/
def optimal_investment : ℝ × ℝ := (72, 48)

/-- Statement: The optimal investment is 720,000 yuan in city A and 480,000 yuan in city B -/
theorem optimal_investment_correct :
  optimal_investment = (72, 48) := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_maximized_at_72_optimal_investment_correct_l743_74369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_three_integer_solutions_l743_74307

theorem quadratic_inequality_three_integer_solutions 
  (m : ℝ) : 
  (∃! (s : Finset ℤ), s.card = 3 ∧ ∀ x ∈ s, m * x^2 + (2 - m) * x - 2 > 0) ↔ 
  -1/2 < m ∧ m ≤ -2/5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_three_integer_solutions_l743_74307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_derivative_l743_74357

noncomputable section

open Real

-- Define the hyperbolic sine and cosine functions
def sh (x : ℝ) : ℝ := (exp x - exp (-x)) / 2
def ch (x : ℝ) : ℝ := (exp x + exp (-x)) / 2

-- Define the function y
def y (x : ℝ) : ℝ := (1/6) * log ((1 - sh (2*x)) / (2 + sh (2*x)))

-- State the theorem
theorem y_derivative (x : ℝ) : 
  deriv y x = ch (2*x) / (sh (2*x)^2 + sh (2*x) - 2) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_derivative_l743_74357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_simplification_l743_74329

theorem factorial_simplification : 
  (Nat.factorial 13) / (Nat.factorial 11 + 2 * Nat.factorial 10) = 132 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_simplification_l743_74329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_t_l743_74382

/-- Two circles in the Cartesian plane -/
structure TwoCircles where
  t : ℝ
  C₁ : Set (ℝ × ℝ) := {p | p.1^2 + (p.2 - t)^2 = 4}
  C₂ : Set (ℝ × ℝ) := {p | (p.1 - 2)^2 + p.2^2 = 14}

/-- Condition for a line to be tangent to a circle at a point -/
def IsTangentLine (m : ℝ × ℝ → ℝ) (C : Set (ℝ × ℝ)) (Q : ℝ × ℝ) : Prop :=
  Q ∈ C ∧ ∀ P ∈ C, P ≠ Q → m P ≠ m Q

/-- Tangent point conditions -/
def hasTangentPoint (tc : TwoCircles) : Prop :=
  ∃ (P Q : ℝ × ℝ), P ∈ tc.C₁ ∧ Q ∈ tc.C₂ ∧
    (∃ (m : ℝ × ℝ → ℝ), IsTangentLine m tc.C₂ Q) ∧
    (P.1^2 + P.2^2 = 2 * ((P.1 - Q.1)^2 + (P.2 - Q.2)^2))

/-- The main theorem -/
theorem range_of_t (tc : TwoCircles) (h : hasTangentPoint tc) :
    tc.t ∈ Set.Icc (-4 * Real.sqrt 3) (4 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_t_l743_74382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divides_g_product_l743_74300

def g : ℕ → ℕ
  | 0 => 0  -- Added case for 0
  | 1 => 0
  | 2 => 1
  | (n + 3) => g (n + 1) + g (n + 2)

theorem prime_divides_g_product {n : ℕ} (hn : n.Prime) (hn5 : n > 5) :
  n ∣ g n * (g n + 1) := by
  sorry

#eval g 6  -- This line is optional, just to test the function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divides_g_product_l743_74300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_and_ratio_l743_74390

noncomputable def f (x : ℝ) := Real.sin x + Real.sqrt 3 * Real.cos x + 1

theorem f_max_min_and_ratio (a b c : ℝ) :
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ 3 ∧ f x ≥ 2) ∧
  ((∀ x, a * f x + b * f (x - c) = 1) → b * Real.cos c / a = -1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_and_ratio_l743_74390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equality_proof_l743_74316

noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ 0 then -x else 3*x - 41

theorem g_equality_proof (a : ℝ) :
  a < 0 → (g (g (g 8)) = g (g (g a)) ↔ a = -58/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equality_proof_l743_74316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nearest_integer_to_x_plus_y_l743_74362

theorem nearest_integer_to_x_plus_y (x y : ℝ) 
  (h1 : |x| - y = 1)
  (h2 : |x| * y + x^2 = 2) : 
  ∃ (n : ℤ), n = 2 ∧ ∀ (m : ℤ), |↑m - (x + y)| ≥ |↑n - (x + y)| := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nearest_integer_to_x_plus_y_l743_74362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_powers_of_two_in_sequence_l743_74361

theorem infinite_powers_of_two_in_sequence :
  ∀ k : ℕ, ∃ n : ℕ, ∃ m : ℕ, 
    n ≥ 1 ∧ 
    (Int.floor (n * Real.sqrt 2) : ℤ) = 2^m ∧ 
    m ≥ k :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_powers_of_two_in_sequence_l743_74361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_M_l743_74392

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x + 3)^2 + y^2 = 100

-- Define point B
def point_B : ℝ × ℝ := (3, 0)

-- Define a point P on the circle
def point_P (x y : ℝ) : Prop := circle_C x y

-- Define point M as the intersection of perpendicular bisector of BP and CP
def point_M (x y : ℝ) (px py : ℝ) : Prop :=
  point_P px py ∧
  -- M is equidistant from B and P
  (x - 3)^2 + y^2 = (x - px)^2 + (y - py)^2 ∧
  -- M is on line CP
  (x + 3) * py = y * (px + 3)

-- Theorem statement
theorem trajectory_of_M (x y : ℝ) :
  (∃ px py : ℝ, point_M x y px py) →
  (x^2 / 25 + y^2 / 16 = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_M_l743_74392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_shorter_side_is_eight_l743_74347

/-- A rectangle with given area and perimeter -/
structure Rectangle where
  length : ℝ
  width : ℝ
  area_eq : length * width = 120
  perimeter_eq : 2 * (length + width) = 46

/-- The shorter side of a rectangle is the minimum of its length and width -/
noncomputable def Rectangle.shorter_side (r : Rectangle) : ℝ := min r.length r.width

/-- Theorem: For a rectangle with area 120 and perimeter 46, the shorter side is 8 -/
theorem rectangle_shorter_side_is_eight :
  ∀ (r : Rectangle), r.shorter_side = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_shorter_side_is_eight_l743_74347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_exponential_l743_74339

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a + 1) ^ x

theorem decreasing_exponential (a : ℝ) :
  (∀ x y : ℝ, x < y → f a y < f a x) → -1 < a ∧ a < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_exponential_l743_74339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_circle_l743_74379

/-- Given a line and a circle in the xy-plane, prove that under certain conditions, the radius of the circle has a specific value. -/
theorem intersection_line_circle (r : ℝ) (A B : ℝ × ℝ) :
  r > 0 →
  (∀ x y : ℝ, 3 * x - 4 * y - 1 = 0 ↔ (x, y) ∈ ({A, B} : Set (ℝ × ℝ))) →
  (∀ x y : ℝ, x^2 + y^2 = r^2 ↔ (x, y) ∈ ({A, B} : Set (ℝ × ℝ))) →
  ((A.1 * B.1 + A.2 * B.2) / (Real.sqrt ((A.1^2 + A.2^2) * (B.1^2 + B.2^2))) = 0) →
  r = Real.sqrt 2 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_circle_l743_74379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_one_two_l743_74341

-- Define the function f
noncomputable def f (x : ℝ) := Real.exp x + 2 * x - 6

-- State the theorem
theorem zero_in_interval_one_two :
  ∃ (z : ℝ), z ∈ Set.Ioo 1 2 ∧ f z = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_one_two_l743_74341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_quotient_formula_l743_74395

/-- Given positive real numbers x and y, where y = 95.99999999999636 and the remainder of x divided by y is 11.52,
    the quotient q can be expressed as q = (x - 11.52) / 95.99999999999636 -/
theorem division_quotient_formula (x y : ℝ) (h1 : y = 95.99999999999636) 
  (h2 : x - y * ⌊x / y⌋ = 11.52) (hx : x > 0) (hy : y > 0) : 
  ∃ q : ℝ, q = (x - 11.52) / 95.99999999999636 ∧ x = q * y + 11.52 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_quotient_formula_l743_74395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_bounded_diff_implies_a_range_l743_74354

/-- The function f(x) = a(x^2 - x - 1) / e^x -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (x^2 - x - 1) / Real.exp x

/-- Theorem stating the range of 'a' given the conditions -/
theorem f_bounded_diff_implies_a_range (a : ℝ) (h_a_pos : a > 0) :
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 4 → x₂ ∈ Set.Icc 0 4 → |f a x₁ - f a x₂| < 1) →
  0 < a ∧ a < Real.exp 3 / (5 + Real.exp 3) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_bounded_diff_implies_a_range_l743_74354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l743_74370

-- Define the points A, B, and C
def A : ℝ × ℝ := (0, 4)
def B : ℝ × ℝ := (-3, 0)
def C : ℝ × ℝ := (1, 1)

-- Define the distance function
def distance (p : ℝ × ℝ) (l : ℝ → ℝ) : ℝ :=
  sorry

-- Define the line AB
noncomputable def lineAB (x : ℝ) : ℝ :=
  (4/3) * x + 4

-- Define the altitude from B to AC
noncomputable def altitudeB (x : ℝ) : ℝ :=
  -(3/4) * x + 11/4

theorem triangle_ABC_properties :
  -- Part 1: Distance from C to AB is 13/5
  distance C lineAB = 13/5 ∧
  -- Part 2: Equation of altitude from B to AC
  ∀ x, altitudeB x = -(3/4) * x + 11/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l743_74370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quarter_circle_roll_distance_l743_74366

noncomputable def quarter_circle_roll (r : ℝ) : ℝ :=
  3 * r * Real.pi / 2

theorem quarter_circle_roll_distance :
  quarter_circle_roll (2 / Real.pi) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quarter_circle_roll_distance_l743_74366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_rotation_transformation_l743_74375

/-- Represents a 2D point --/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle defined by three points --/
structure Triangle where
  p1 : Point
  p2 : Point
  p3 : Point

/-- Applies a rotation to a point --/
noncomputable def rotate (center : Point) (angle : ℝ) (p : Point) : Point :=
  let dx := p.x - center.x
  let dy := p.y - center.y
  { x := center.x + dx * Real.cos angle - dy * Real.sin angle,
    y := center.y + dx * Real.sin angle + dy * Real.cos angle }

/-- Checks if two triangles are equal up to a small epsilon --/
def trianglesEqual (t1 t2 : Triangle) (ε : ℝ) : Prop :=
  (abs (t1.p1.x - t2.p1.x) < ε) ∧ (abs (t1.p1.y - t2.p1.y) < ε) ∧
  (abs (t1.p2.x - t2.p2.x) < ε) ∧ (abs (t1.p2.y - t2.p2.y) < ε) ∧
  (abs (t1.p3.x - t2.p3.x) < ε) ∧ (abs (t1.p3.y - t2.p3.y) < ε)

theorem triangle_rotation_transformation :
  let t1 := Triangle.mk (Point.mk 0 0) (Point.mk 0 10) (Point.mk 18 0)
  let t2 := Triangle.mk (Point.mk 30 26) (Point.mk 42 26) (Point.mk 30 12)
  let center := Point.mk 28 (-2)
  let angle := 90 * π / 180  -- 90 degrees in radians
  let rotated_t1 := Triangle.mk
    (rotate center angle t1.p1)
    (rotate center angle t1.p2)
    (rotate center angle t1.p3)
  trianglesEqual rotated_t1 t2 0.001 := by
  sorry

#eval 90 + 28 + (-2)  -- This should output 116

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_rotation_transformation_l743_74375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_p_l743_74328

/-- The universal set U --/
def U : Set ℕ := {1, 2, 3, 4}

/-- The set M defined by a quadratic equation --/
def M (p : ℝ) : Set ℝ := {x : ℝ | x^2 - 5*x + p = 0}

/-- Helper function to convert Set ℕ to Set ℝ --/
def setNatToReal (S : Set ℕ) : Set ℝ := {x : ℝ | ∃ n : ℕ, n ∈ S ∧ x = n}

/-- The theorem stating that p = 4 given the conditions --/
theorem find_p : 
  ∃ (p : ℝ), (M p)ᶜ ∩ (setNatToReal U) = setNatToReal {2, 3} ∧ p = 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_p_l743_74328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_5_l743_74301

-- Define the sales volume function
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a / (x - 4) + 10 * (x - 7)^2

-- Define the profit function
noncomputable def h (x : ℝ) (a : ℝ) : ℝ := (x - 4) * (f x a)

-- Theorem statement
theorem max_profit_at_5 (a : ℝ) :
  (∀ x, 4 < x → x < 7 → f x a ≥ 0) →  -- Ensure non-negative sales
  f 6 a = 15 →                        -- Given condition
  ∃ (x : ℝ), 4 < x ∧ x < 7 ∧
    ∀ y, 4 < y → y < 7 → h x a ≥ h y a ∧
    x = 5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_5_l743_74301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_wins_l743_74381

/-- Represents a player's move, either 1 or 2 -/
inductive Move : Type
| one : Move
| two : Move

/-- Represents the state of the game after each move -/
structure GameState :=
  (moves : List Move)
  (turn : Nat)

/-- Returns the opposite move -/
def oppositeMove (m : Move) : Move :=
  match m with
  | Move.one => Move.two
  | Move.two => Move.one

/-- Calculates the sum of digits in the game state -/
def sumOfDigits (state : GameState) : Nat :=
  state.moves.foldl (fun acc m => acc + match m with
    | Move.one => 1
    | Move.two => 2) 0

/-- The second player's strategy is to always play the opposite move -/
def secondPlayerStrategy (state : GameState) : Move :=
  match state.moves.getLast? with
  | some lastMove => oppositeMove lastMove
  | none => Move.one  -- This case should never happen in a valid game

/-- Theorem: The second player always wins with optimal play -/
theorem second_player_wins :
  ∀ (finalState : GameState),
  finalState.turn = 5 →
  (∀ (i : Nat), i < 5 → i % 2 = 1 →
    finalState.moves.get? i = some (secondPlayerStrategy { moves := finalState.moves.take i, turn := i })) →
  ¬(sumOfDigits finalState % 3 = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_wins_l743_74381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_time_to_reach_fishing_boat_l743_74303

-- Define the problem parameters
noncomputable def initial_distance : ℝ := 10
noncomputable def vessel_speed : ℝ := 21
noncomputable def fishing_boat_speed : ℝ := 9
noncomputable def angle_difference : ℝ := 60 * Real.pi / 180  -- Convert 60 degrees to radians

-- Define the function to calculate the time to reach the fishing boat
noncomputable def time_to_reach (t : ℝ) : ℝ :=
  (initial_distance^2 + (fishing_boat_speed * t)^2 - 
   2 * initial_distance * fishing_boat_speed * t * Real.cos angle_difference) / (vessel_speed^2)

-- Theorem statement
theorem shortest_time_to_reach_fishing_boat :
  ∃ t : ℝ, t > 0 ∧ time_to_reach t = t ∧ t = 5 / 12 := by
  sorry

#eval "Theorem stated successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_time_to_reach_fishing_boat_l743_74303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_increasing_condition_tangent_condition_l743_74309

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * x / (x + 1)

-- Part I
theorem monotonically_increasing_condition (a : ℝ) :
  (∀ x ∈ Set.Ioo 0 4, Monotone (f a)) ↔ a ≥ -4 := by sorry

-- Part II
theorem tangent_condition (a : ℝ) :
  (∃ x₀ > 0, f a x₀ = 2 * x₀ ∧ deriv (f a) x₀ = 2) ↔ a = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_increasing_condition_tangent_condition_l743_74309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_ratio_l743_74371

/-- The number of balls -/
def n : ℕ := 18

/-- The number of bins -/
def k : ℕ := 4

/-- The probability of tossing a ball into any specific bin -/
noncomputable def p_bin : ℚ := 1 / k

/-- The probability of the 6-2-5-5 distribution -/
noncomputable def p : ℚ := (4 * 3 * Nat.choose n 6 * Nat.choose 12 2 * Nat.choose 10 5) / k^n

/-- The probability of the 5-5-4-4 distribution -/
noncomputable def q : ℚ := (Nat.choose 4 2 * Nat.choose n 5 * Nat.choose 13 5 * Nat.choose 8 4) / k^n

/-- The main theorem stating the ratio of probabilities -/
theorem probability_ratio : p / q = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_ratio_l743_74371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_m_plus_one_mod_nine_l743_74394

/-- S(m) is the sum of the digits of a positive integer m -/
def S (m : ℕ+) : ℕ := sorry

/-- For a particular positive integer m, S(m) = 2080 -/
axiom exists_m : ∃ m : ℕ+, S m = 2080

theorem S_m_plus_one_mod_nine (m : ℕ+) (h : S m = 2080) : 
  S (m + 1) % 9 = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_m_plus_one_mod_nine_l743_74394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_distance_l743_74349

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 2 - y^2 / 8 = 1

-- Define the asymptotes
def asymptote1 (x y : ℝ) : Prop := y = 2 * x
def asymptote2 (x y : ℝ) : Prop := y = -2 * x

-- Define a point on the hyperbola
structure PointOnHyperbola where
  x : ℝ
  y : ℝ
  on_hyperbola : hyperbola x y

-- Define the distance function
noncomputable def distance (x y a b : ℝ) : ℝ := 
  Real.sqrt ((x - a)^2 + (y - b)^2)

-- The main theorem
theorem hyperbola_asymptote_distance 
  (P : PointOnHyperbola) 
  (h : ∃ (a b : ℝ), (asymptote1 a b ∧ distance P.x P.y a b = 1/5) ∨ 
                    (asymptote2 a b ∧ distance P.x P.y a b = 1/5)) :
  ∃ (c d : ℝ), (asymptote1 c d ∨ asymptote2 c d) ∧ 
               distance P.x P.y c d = 8 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_distance_l743_74349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_maximizes_sum_of_squared_sides_l743_74386

/-- A polygon inscribed in a circle -/
structure InscribedPolygon where
  vertices : List (ℝ × ℝ)
  inscribed : ∀ v ∈ vertices, (v.1^2 + v.2^2 = 1)  -- Assuming unit circle for simplicity

/-- The sum of squares of side lengths of a polygon -/
def sumOfSquaredSideLengths (p : InscribedPolygon) : ℝ := sorry

/-- An equilateral triangle inscribed in a circle -/
noncomputable def inscribedEquilateralTriangle : InscribedPolygon := sorry

/-- Theorem: The inscribed equilateral triangle maximizes the sum of squared side lengths -/
theorem equilateral_triangle_maximizes_sum_of_squared_sides :
  ∀ p : InscribedPolygon, sumOfSquaredSideLengths p ≤ sumOfSquaredSideLengths inscribedEquilateralTriangle := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_maximizes_sum_of_squared_sides_l743_74386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_and_g_minimum_l743_74326

noncomputable def f (x : ℝ) := 2 * (Real.sin x) ^ 4 + 2 * (Real.cos x) ^ 4 + (Real.cos (2 * x)) ^ 2 - 3

noncomputable def g (x : ℝ) := f (2 * x + Real.pi / 3)

theorem f_properties_and_g_minimum :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∀ (k : ℤ), ∀ (x : ℝ), f (k * Real.pi / 4 + x) = f (k * Real.pi / 4 - x)) ∧
  (∀ (k : ℤ), ∀ (x : ℝ), x ∈ Set.Icc (k * Real.pi / 2) (Real.pi / 4 + k * Real.pi / 2) → 
    ∀ (y : ℝ), y ∈ Set.Icc (k * Real.pi / 2) (Real.pi / 4 + k * Real.pi / 2) → x ≤ y → f y ≤ f x) ∧
  (∀ (x : ℝ), x ∈ Set.Icc (-Real.pi / 4) (Real.pi / 6) → g x ≥ -3 / 2) ∧
  g (Real.pi / 6) = -3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_and_g_minimum_l743_74326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_through_origin_l743_74359

noncomputable def f (x : ℝ) : ℝ := Real.exp (2 - x)

noncomputable def f' (x : ℝ) : ℝ := -Real.exp (2 - x)

theorem tangent_line_through_origin :
  ∃ (x₀ : ℝ), 
    (f x₀ = -f' x₀ * x₀) ∧ 
    (∀ (x y : ℝ), y = -Real.exp 3 * x ↔ y = f' x₀ * (x - x₀) + f x₀) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_through_origin_l743_74359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_squares_passed_l743_74365

/-- A circle on a unit square grid -/
structure GridCircle where
  radius : ℝ
  center : ℝ × ℝ
  not_touch_grid : ∀ (x y : ℤ), ((x : ℝ) - center.1)^2 + ((y : ℝ) - center.2)^2 ≠ radius^2
  not_pass_lattice : ∀ (x y : ℤ), ((x : ℝ) - center.1)^2 + ((y : ℝ) - center.2)^2 > radius^2

/-- The number of squares a circle passes through on a unit square grid -/
noncomputable def squares_passed (c : GridCircle) : ℕ :=
  sorry

/-- Theorem: A circle with radius 100 passes through at most 800 squares -/
theorem max_squares_passed (c : GridCircle) (h : c.radius = 100) :
  squares_passed c ≤ 800 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_squares_passed_l743_74365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_zero_condition_l743_74312

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then x - Real.exp x + 2
  else (1/3) * x^3 - 4*x + a

-- Theorem statement
theorem exactly_one_zero_condition (a : ℝ) :
  (∃! x, f a x = 0) → a > 16/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_zero_condition_l743_74312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_l743_74314

-- Define the curves C₁ and C₂
noncomputable def C₁ (α : ℝ) : ℝ × ℝ := (1 + Real.cos α, Real.sin α)

def C₂ (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

-- Define the ray
def ray (ρ θ : ℝ) : Prop := θ = Real.pi / 3 ∧ ρ ≥ 0

-- Define the intersection points
noncomputable def A : ℝ × ℝ := C₁ (Real.pi / 3)
noncomputable def B : ℝ × ℝ := (Real.sqrt 30 * Real.cos (Real.pi / 3) / Real.sqrt 5, Real.sqrt 30 * Real.sin (Real.pi / 3) / Real.sqrt 5)

-- Theorem statement
theorem distance_AB : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 30 / 5 - 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_l743_74314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partial_array_exists_l743_74327

/-- Represents a partially filled array -/
def PartialArray (m n : ℕ) := Fin m → Fin n → Option ℝ

/-- Sums the elements in a row of a partial array -/
def rowSum (arr : PartialArray m n) (i : Fin m) : ℝ :=
  (Finset.univ : Finset (Fin n)).sum (fun j => (arr i j).getD 0)

/-- Sums the elements in a column of a partial array -/
def colSum (arr : PartialArray m n) (j : Fin n) : ℝ :=
  (Finset.univ : Finset (Fin m)).sum (fun i => (arr i j).getD 0)

/-- The main theorem -/
theorem partial_array_exists (m n : ℕ) (A : Fin m → ℝ) (B : Fin n → ℝ)
    (hpos : ∀ i, A i > 0 ∧ ∀ j, B j > 0)
    (hsum : (Finset.univ : Finset (Fin m)).sum A = (Finset.univ : Finset (Fin n)).sum B) :
    ∃ (arr : PartialArray m n),
      (∀ i, rowSum arr i = A i) ∧
      (∀ j, colSum arr j = B j) ∧
      ((Finset.univ : Finset (Fin m × Fin n)).filter (fun (i, j) => (arr i j).isSome)).card < m + n :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_partial_array_exists_l743_74327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projective_iff_fractional_linear_l743_74332

noncomputable def P (a b c d : ℝ) (x : ℝ) : ℝ := (a * x + b) / (c * x + d)

def is_projective (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ x₃ x₄, (f x₁ - f x₂) * (f x₃ - f x₄) = (f x₁ - f x₃) * (f x₂ - f x₄)

theorem projective_iff_fractional_linear :
  ∀ (f : ℝ → ℝ), is_projective f ↔
  ∃ (a b c d : ℝ), (a * d - b * c ≠ 0) ∧ (∀ x, f x = P a b c d x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projective_iff_fractional_linear_l743_74332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_not_reachable_other_positions_reachable_l743_74389

-- Define the board positions
inductive Position
| S | P | Q | R | T | W

-- Define the possible directions
inductive Direction
| Up | Down | Left | Right

-- Define a move as a pair of directions
structure Move where
  first : Direction
  second : Direction

-- Function to check if two directions are perpendicular
def isPerpendicular (d1 d2 : Direction) : Prop :=
  (d1 = Direction.Up ∨ d1 = Direction.Down) ∧ (d2 = Direction.Left ∨ d2 = Direction.Right) ∨
  (d1 = Direction.Left ∨ d1 = Direction.Right) ∧ (d2 = Direction.Up ∨ d2 = Direction.Down)

-- Function to check if a move is valid
def isValidMove (m : Move) : Prop :=
  isPerpendicular m.first m.second

-- Define the applyMove function
def applyMove : Position → Move → Position
  | Position.S, _ => Position.S  -- Placeholder implementation
  | p, _ => p  -- Placeholder implementation for other positions

-- Function to check if a position is reachable from S
def isReachable (p : Position) : Prop :=
  ∃ (m : Move), isValidMove m ∧ (applyMove Position.S m = p)

-- Theorem to prove
theorem t_not_reachable : ¬ isReachable Position.T := by
  sorry

-- Additional theorem to show that other positions are reachable (for completeness)
theorem other_positions_reachable :
  isReachable Position.P ∧ isReachable Position.Q ∧ 
  isReachable Position.R ∧ isReachable Position.W := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_not_reachable_other_positions_reachable_l743_74389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l743_74358

open Real

/-- The function f(x) = x ln x + x^2 - ax + 2 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * log x + x^2 - a*x + 2

/-- Theorem stating the properties of the function f -/
theorem function_properties (a : ℝ) 
  (h1 : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) :
  (a ∈ Set.Ioi 3) ∧ 
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → f a x₁ = 0 → f a x₂ = 0 → x₁ * x₂ > 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l743_74358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_2_floor_x_range_l743_74304

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Theorem 1
theorem sqrt_2_floor : floor (Real.sqrt 2) = 1 := by sorry

-- Theorem 2
theorem x_range (x : ℝ) : floor (3 + Real.sqrt x) = 6 → 9 ≤ x ∧ x < 16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_2_floor_x_range_l743_74304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_positive_area_triangles_l743_74336

/-- A point in the coordinate plane with integer coordinates -/
structure Point where
  x : ℤ
  y : ℤ

/-- A triangle defined by three points -/
structure Triangle where
  p1 : Point
  p2 : Point
  p3 : Point

/-- The set of valid points in our grid -/
def validPoints : Set Point :=
  {p : Point | 1 ≤ p.x ∧ p.x ≤ 5 ∧ 1 ≤ p.y ∧ p.y ≤ 5}

/-- Predicate to check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

/-- Predicate to check if a triangle has positive area -/
def positiveArea (t : Triangle) : Prop :=
  ¬collinear t.p1 t.p2 t.p3

/-- The set of all triangles with vertices in validPoints -/
def allTriangles : Set Triangle :=
  {t : Triangle | t.p1 ∈ validPoints ∧ t.p2 ∈ validPoints ∧ t.p3 ∈ validPoints}

/-- The set of triangles with positive area -/
def positiveAreaTriangles : Set Triangle :=
  {t ∈ allTriangles | positiveArea t}

/-- Assume finiteness of the set of positive area triangles -/
instance : Fintype positiveAreaTriangles := sorry

/-- The main theorem stating the count of positive area triangles -/
theorem count_positive_area_triangles :
  Fintype.card positiveAreaTriangles = 2160 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_positive_area_triangles_l743_74336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_inequality_l743_74385

theorem cosine_inequality (x : ℝ) : 
  x ∈ Set.Icc 0 (2 * π) →
  (2 * Real.cos x ≤ |Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))| ∧ 
   |Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))| ≤ Real.sqrt 2) ↔
  x ∈ Set.Icc (π / 4) (7 * π / 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_inequality_l743_74385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_triangle_with_small_angle_l743_74350

-- Define a point in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a set of 6 points
def SixPoints : Finset Point := sorry

-- Define the property of no three points being collinear
def NoThreeCollinear (points : Finset Point) : Prop := sorry

-- Define a triangle formed by three points
structure Triangle where
  a : Point
  b : Point
  c : Point

-- Define the property of a triangle having an angle less than or equal to 30°
def HasAngleLeq30Deg (t : Triangle) : Prop := sorry

-- Theorem statement
theorem exists_triangle_with_small_angle 
  (h1 : SixPoints.card = 6)
  (h2 : NoThreeCollinear SixPoints) :
  ∃ (t : Triangle), (t.a ∈ SixPoints ∧ t.b ∈ SixPoints ∧ t.c ∈ SixPoints) ∧ HasAngleLeq30Deg t :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_triangle_with_small_angle_l743_74350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_intersect_l743_74302

/-- Definition of the first line -/
noncomputable def line1 (s : ℝ) : ℝ × ℝ := (2 + 3 * s, -4 * s)

/-- Definition of the second line -/
noncomputable def line2 (v : ℝ) : ℝ × ℝ := (6 + 5 * v, -10 + 3 * v)

/-- The intersection point -/
noncomputable def intersection_point : ℝ × ℝ := (242 / 29, -248 / 29)

/-- Theorem stating that the lines intersect at the given point -/
theorem lines_intersect : 
  ∃ (s v : ℝ), line1 s = line2 v ∧ line1 s = intersection_point := by
  -- We'll use s = 62/29 and v = 14/29 as found in the solution
  let s := 62 / 29
  let v := 14 / 29
  use s, v
  sorry  -- The actual proof is omitted for brevity


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_intersect_l743_74302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_proof_l743_74319

def hyperbola_equation (x y : ℝ) := 3/4 * y^2 - 1/3 * x^2 = 1

theorem hyperbola_proof :
  ∀ (a b : ℝ), a > 0 → b > 0 →
  (3 * a = 2 * b) →
  (4 / a^2 - 6 / b^2 = 1) →
  hyperbola_equation (Real.sqrt 6) 2 :=
by
  intros a b ha hb h1 h2
  simp [hyperbola_equation]
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_proof_l743_74319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_numbers_starting_with_four_l743_74338

/-- The count of numbers starting with 4 in the sequence of powers of 2 from 2^1 to 2^333 -/
def count_starting_with_four : ℕ := 33

/-- The number of digits in 2^333 -/
def digits_of_2_333 : ℕ := 101

/-- The first digit of 2^333 -/
def first_digit_of_2_333 : ℕ := 1

/-- The sequence of powers of 2 from 2^1 to 2^333 -/
def power_of_two_sequence : List ℕ := List.range 333 |>.map (fun n => 2^(n+1))

/-- Convert a natural number to a string -/
noncomputable def nat_to_string (n : ℕ) : String :=
  toString n

/-- Get the first character of a string -/
def first_char (s : String) : Char :=
  s.get 0

/-- Convert a character to a digit if possible -/
def char_to_digit (c : Char) : Option ℕ :=
  c.toNat - '0'.toNat

theorem count_numbers_starting_with_four :
  (power_of_two_sequence.filter (fun n => char_to_digit (first_char (nat_to_string n)) = some 4)).length = count_starting_with_four ∧
  (nat_to_string (2^333)).length = digits_of_2_333 ∧
  char_to_digit (first_char (nat_to_string (2^333))) = some first_digit_of_2_333 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_numbers_starting_with_four_l743_74338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse1_passes_through_A_and_B_ellipse2_passes_through_P_and_has_same_foci_l743_74333

-- Define the points
noncomputable def A : ℝ × ℝ := (-2, Real.sqrt 2)
noncomputable def B : ℝ × ℝ := (Real.sqrt 6, -1)
def P : ℝ × ℝ := (-3, 2)

-- Define the reference ellipse
def reference_ellipse (x y : ℝ) : Prop := x^2/9 + y^2/4 = 1

-- Define the first ellipse
def ellipse1 (x y : ℝ) : Prop := x^2/8 + y^2/4 = 1

-- Define the second ellipse
def ellipse2 (x y : ℝ) : Prop := x^2/15 + y^2/10 = 1

-- Theorem for the first ellipse
theorem ellipse1_passes_through_A_and_B :
  ellipse1 A.1 A.2 ∧ ellipse1 B.1 B.2 := by
  sorry

-- Theorem for the second ellipse
theorem ellipse2_passes_through_P_and_has_same_foci :
  ellipse2 P.1 P.2 ∧
  ∃ (c : ℝ), c^2 = 5 ∧
    ∀ (x y : ℝ), ellipse2 x y ↔ (x - c)^2 + y^2 + (x + c)^2 + y^2 = 2 * (15 + 10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse1_passes_through_A_and_B_ellipse2_passes_through_P_and_has_same_foci_l743_74333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_length_for_given_circles_l743_74383

/-- Two concentric circles with a chord in the smaller circle extended to the larger circle -/
structure ConcentricCirclesWithChord where
  /-- Center of both circles -/
  center : ℝ × ℝ
  /-- Radius of the smaller circle -/
  r₁ : ℝ
  /-- Radius of the larger circle -/
  r₂ : ℝ
  /-- Length of the chord in the smaller circle -/
  chord_length : ℝ
  /-- The chord is contained in the smaller circle -/
  chord_in_circle : chord_length ≤ 2 * r₁
  /-- The circles are concentric and the larger circle has a bigger radius -/
  concentric : r₁ < r₂

/-- The length of the intersection on the larger circle -/
noncomputable def intersection_length (c : ConcentricCirclesWithChord) : ℝ :=
  2 * Real.sqrt (c.r₂^2 - (c.r₁^2 - (c.chord_length / 2)^2))

/-- Theorem stating the length of the intersection on the larger circle -/
theorem intersection_length_for_given_circles :
  ∀ c : ConcentricCirclesWithChord,
  c.r₁ = 4 ∧ c.r₂ = 6 ∧ c.chord_length = 2 →
  intersection_length c = 2 * Real.sqrt 21 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_length_for_given_circles_l743_74383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_cube_l743_74320

theorem det_cube {n : Type*} [Fintype n] [DecidableEq n] 
  (M : Matrix n n ℝ) (h : Matrix.det M = 3) : 
  Matrix.det (M ^ 3) = 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_cube_l743_74320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l743_74388

noncomputable def f (x : ℝ) := Real.log (2 * x - x^2) / (x - 1)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x | 0 < x ∧ x < 1} ∪ {x | 1 < x ∧ x < 2} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l743_74388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l743_74344

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (2 + (Real.sqrt 2 / 2) * t, (Real.sqrt 2 / 2) * t)

-- Define the curve C
noncomputable def curve_C (θ : ℝ) : ℝ × ℝ := (4 * Real.cos θ, 2 * Real.sqrt 3 * Real.sin θ)

-- Define point P
def P : ℝ × ℝ := (2, 0)

-- Theorem statement
theorem intersection_distance_product : 
  ∃ (A B : ℝ × ℝ) (t₁ t₂ θ₁ θ₂ : ℝ),
    line_l t₁ = curve_C θ₁ ∧ 
    line_l t₂ = curve_C θ₂ ∧ 
    A = line_l t₁ ∧ 
    B = line_l t₂ ∧
    Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2) * Real.sqrt ((B.1 - P.1)^2 + (B.2 - P.2)^2) = 48/7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l743_74344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_exists_sin_geq_one_l743_74325

theorem negation_of_exists_sin_geq_one :
  (¬ ∃ x : ℝ, Real.sin x ≥ 1) ↔ (∀ x : ℝ, Real.sin x ≤ 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_exists_sin_geq_one_l743_74325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_correct_answers_satisfies_inequality_l743_74337

/-- Represents a quiz with a given number of questions, points for correct answers,
    points deducted for incorrect answers, and a target score. -/
structure Quiz where
  total_questions : ℕ
  points_correct : ℕ
  points_incorrect : ℕ
  target_score : ℕ

/-- Given a Quiz, determines the minimum number of correct answers needed to reach the target score. -/
def min_correct_answers (q : Quiz) : ℕ := 
  Nat.ceil ((q.target_score + q.total_questions * q.points_incorrect : ℚ) / (q.points_correct + q.points_incorrect : ℚ))

/-- Theorem stating that the minimum number of correct answers satisfies the inequality. -/
theorem min_correct_answers_satisfies_inequality (q : Quiz) 
  (h : q.total_questions = 25 ∧ q.points_correct = 5 ∧ q.points_incorrect = 1 ∧ q.target_score = 85) :
  let x := min_correct_answers q
  (5 : ℤ) * x - (25 - x) ≥ 85 := by
  sorry

#eval min_correct_answers {total_questions := 25, points_correct := 5, points_incorrect := 1, target_score := 85}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_correct_answers_satisfies_inequality_l743_74337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_expression_equality_l743_74399

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- State the theorem
theorem floor_expression_equality : 
  (floor 6.5) * (floor (2 / 3 : ℝ)) + (floor 2) * (7.2 : ℝ) + (floor 8.4) - (6.2 : ℝ) = 16.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_expression_equality_l743_74399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_quadruple_angle_l743_74372

theorem cos_quadruple_angle (θ : Real) (h : Real.cos θ = 1/3) : Real.cos (4*θ) = 17/81 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_quadruple_angle_l743_74372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_8_20_l743_74311

-- Define the clock time
noncomputable def hours : ℝ := 8
noncomputable def minutes : ℝ := 20

-- Define the angles of the clock hands
noncomputable def minute_hand_angle : ℝ := (minutes / 60) * 360
noncomputable def hour_hand_angle : ℝ := ((hours + minutes / 60) / 12) * 360

-- Define the acute angle between the hands
noncomputable def angle_between_hands : ℝ := 
  min (abs (hour_hand_angle - minute_hand_angle)) 
      (360 - abs (hour_hand_angle - minute_hand_angle))

-- Theorem statement
theorem clock_angle_at_8_20 : angle_between_hands = 130 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_8_20_l743_74311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_y_l743_74340

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 + Real.log x / Real.log 3

-- Define the function y
noncomputable def y (x : ℝ) : ℝ := (f x)^2 + f (x^2)

-- State the theorem
theorem max_value_of_y :
  ∃ (M : ℝ), M = 13 ∧
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 9 → y x ≤ M) ∧
  (∃ x : ℝ, 1 ≤ x ∧ x ≤ 9 ∧ y x = M) := by
  sorry

#check max_value_of_y

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_y_l743_74340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_missing_angle_measure_l743_74317

/-- Represents a regular polygon with some missing angles -/
structure RegularPolygon where
  n : ℕ                -- number of sides
  sumKnownAngles : ℝ   -- sum of known interior angles
  missingAngles : ℕ    -- number of missing angles

/-- The measure of each interior angle in a regular polygon -/
noncomputable def interiorAngleMeasure (p : RegularPolygon) : ℝ :=
  180 * (p.n - 2 : ℝ) / p.n

/-- The theorem to be proved -/
theorem missing_angle_measure (p : RegularPolygon) 
  (h1 : p.sumKnownAngles = 3240)
  (h2 : p.missingAngles = 2) :
  ⌊interiorAngleMeasure p⌋ = 166 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_missing_angle_measure_l743_74317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_intersection_l743_74318

-- Define necessary types and structures
structure Plane :=
  (x : ℝ) (y : ℝ)

def SegmentLength (A B : Plane) : ℝ := sorry

def IsoscelesTriangle (A B C : Plane) : Prop :=
  SegmentLength A B = SegmentLength B C

def Circle (O : Plane) (r : ℝ) : Set Plane := sorry

def InteriorTriangle (A B C : Plane) : Set Plane := sorry

def Segment (A B : Plane) : Set Plane := sorry

def Line (A B : Plane) : Set Plane := sorry

def Perpendicular (l1 l2 : Set Plane) : Prop := sorry

def InscribedCircle (A B C : Plane) (r : ℝ) : Prop :=
  ∃ (O : Plane), O ∈ InteriorTriangle A B C ∧ 
    Circle O r ⊆ InteriorTriangle A B C ∧
    (Circle O r ∩ Segment A B).Nonempty ∧
    (Circle O r ∩ Segment B C).Nonempty ∧
    (Circle O r ∩ Segment A C).Nonempty

def Altitude (A B C : Plane) : Set Plane :=
  { D | ∃ (D : Plane), D ∈ Line A C ∧ Perpendicular (Line B D) (Line A C) }

def CircleInscribed (A B C : Plane) : Set Plane := sorry

/-- Given an isosceles triangle ABC with inscribed circle, prove the length of EN --/
theorem inscribed_circle_intersection (A B C E D M N : Plane) : 
  IsoscelesTriangle A B C →
  SegmentLength A C = 4 * Real.sqrt 3 →
  InscribedCircle A B C 3 →
  E ∈ Line A E →
  E ∈ Altitude A B C →
  M ∈ Segment A E →
  N ∈ Segment A E →
  M ∈ CircleInscribed A B C →
  N ∈ CircleInscribed A B C →
  SegmentLength E D = 2 →
  SegmentLength E N = (1 + Real.sqrt 33) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_intersection_l743_74318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l743_74393

noncomputable def f (x : ℝ) : ℝ :=
  if x = 1 then 1 else 1 / abs (x - 1)

noncomputable def h (x b : ℝ) : ℝ :=
  (f x)^2 + b * (f x) + (1/2) * b^2 - 5/8

theorem arithmetic_sequence_sum (b : ℝ) :
  (∃ x₁ x₂ x₃ x₄ x₅ : ℝ,
    x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄ ∧ x₄ < x₅ ∧
    h x₁ b = 0 ∧ h x₂ b = 0 ∧ h x₃ b = 0 ∧ h x₄ b = 0 ∧ h x₅ b = 0 ∧
    ∃ d : ℝ, x₂ = x₁ + d ∧ x₃ = x₂ + d ∧ x₄ = x₃ + d ∧ x₅ = x₄ + d) →
  (∃ a d : ℝ, (Finset.sum (Finset.range 10) (λ i => a + i * d)) = 35) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l743_74393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sections_perpendicular_implies_right_prism_diagonals_equal_implies_right_prism_l743_74360

-- Define a prism
structure Prism where
  -- Add necessary fields for a prism
  mk :: -- This allows us to create a Prism without specifying fields for now

-- Define a right prism
def is_right_prism (p : Prism) : Prop :=
  -- Add definition for a right prism
  True -- Placeholder, replace with actual definition

-- Define sections through opposite lateral edges
def sections_perpendicular_to_base (p : Prism) : Prop :=
  -- Add definition for sections being perpendicular to base
  True -- Placeholder, replace with actual definition

-- Define diagonals of a prism
def diagonals_pairwise_equal (p : Prism) : Prop :=
  -- Add definition for diagonals being pairwise equal
  True -- Placeholder, replace with actual definition

-- Theorem 1: If sections through opposite lateral edges are perpendicular to base, then it's a right prism
theorem sections_perpendicular_implies_right_prism (p : Prism) :
  sections_perpendicular_to_base p → is_right_prism p :=
by
  sorry

-- Theorem 2: If diagonals are pairwise equal, then it's a right prism
theorem diagonals_equal_implies_right_prism (p : Prism) :
  diagonals_pairwise_equal p → is_right_prism p :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sections_perpendicular_implies_right_prism_diagonals_equal_implies_right_prism_l743_74360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prisoners_partition_exists_l743_74323

/-- A partition of 9 prisoners into groups of 3 for 5 days -/
def PrisonerPartition := List (List (Fin 9))

/-- Check if a partition is valid (15 groups of 3 prisoners each) -/
def isValidPartition (p : PrisonerPartition) : Prop :=
  p.length = 5 ∧ p.all (λ day => day.length = 3) ∧ 
  p.all (λ day => day.all (λ prisoner => prisoner < 9))

/-- Check if no pair of prisoners is in the same group more than once -/
def noPairRepeats (p : PrisonerPartition) : Prop :=
  ∀ i j, i < j → (p.filter (λ day => day.contains i ∧ day.contains j)).length ≤ 1

theorem prisoners_partition_exists : 
  ∃ p : PrisonerPartition, isValidPartition p ∧ noPairRepeats p := by
  sorry

#check prisoners_partition_exists

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prisoners_partition_exists_l743_74323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l743_74378

open Set Real

noncomputable def SolutionSet : Set ℝ :=
  Ioc (-6) (-2) ∪ Icc (-2) 1

noncomputable def f (x : ℝ) : ℝ :=
  (2 / (x + 2)) + (8 / (x + 6)) - 2

theorem inequality_solution :
  ∀ x : ℝ, f x ≥ 0 ↔ x ∈ SolutionSet := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l743_74378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l743_74343

theorem trigonometric_identity (x : ℝ) 
  (h1 : 0 < x) (h2 : x < Real.pi) (h3 : Real.sin x + Real.cos x = 7/13) : 
  (Real.sin x * Real.cos x = -60/169) ∧ 
  ((5*Real.sin x + 4*Real.cos x) / (15*Real.sin x - 7*Real.cos x) = 8/43) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l743_74343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l743_74363

noncomputable def a (n : ℕ) : ℝ := 2 * n - 1

noncomputable def b (n : ℕ) : ℝ := (4 * n - 1) / 3^(n - 1)

noncomputable def T (n : ℕ) : ℝ := (15 / 2) - (4 * n + 5) / (2 * 3^(n - 1))

theorem arithmetic_sequence_problem :
  (∀ n : ℕ, n ≥ 1 → (a (n + 1) - a n = a 2 - a 1)) ∧ 
  (a 2 * a 5 = 27 ∧ a 2 + a 5 = 12) ∧
  (∀ n : ℕ, n ≥ 1 → 3^(n - 1) * b n = n * a (n + 1) - (n - 1) * a n) ∧
  (∀ n : ℕ, n ≥ 1 → a n = 2 * n - 1) ∧
  (∀ n : ℕ, n ≥ 1 → b n = (4 * n - 1) / 3^(n - 1)) ∧
  (∀ n : ℕ, n ≥ 1 → T n = (15 / 2) - (4 * n + 5) / (2 * 3^(n - 1))) ∧
  (∀ n : ℕ, n ≥ 4 → T n ≥ 7) ∧
  (T 3 < 7) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l743_74363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_l743_74335

/-- The hyperbola equation -/
noncomputable def hyperbola_equation (x y : ℝ) : Prop := 9 * x^2 - 16 * y^2 = 1

/-- The distance from the center to a focus -/
noncomputable def focal_distance : ℝ := 5 / 12

/-- Theorem: The foci of the hyperbola 9x^2 - 16y^2 = 1 are at (0, ±5/12) -/
theorem hyperbola_foci :
  ∀ (x y : ℝ), hyperbola_equation x y →
  (x = 0 ∧ (y = focal_distance ∨ y = -focal_distance)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_l743_74335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_test_results_theorem_l743_74356

/-- Represents the test result of a boy, in seconds relative to the passing time -/
def TestResult := Float

/-- The number of boys in the group -/
def numBoys : Nat := 6

/-- The passing time for the 100-meter test in seconds -/
def passingTime : Float := 14

/-- The test results of the 6 boys -/
def testResults : List Float := [0.6, -1.1, 0, -0.2, 2, 0.5]

/-- Determines if a boy passed the test based on their result -/
def passed (result : Float) : Bool := result ≤ 0

/-- Counts the number of boys who did not pass the test -/
def countFailed (results : List Float) : Nat :=
  results.filter (fun r => ¬(passed r)) |>.length

/-- Calculates the fastest time among the test results -/
def fastestTime (results : List Float) : Float :=
  passingTime + results.foldl min 0

/-- Calculates the average score of the boys -/
def averageScore (results : List Float) : Float :=
  passingTime + (results.foldl (· + ·) 0) / numBoys.toFloat

theorem test_results_theorem :
  countFailed testResults = 3 ∧
  fastestTime testResults = 12.9 ∧
  averageScore testResults = 14.3 := by
  sorry

#eval countFailed testResults
#eval fastestTime testResults
#eval averageScore testResults

end NUMINAMATH_CALUDE_ERRORFEEDBACK_test_results_theorem_l743_74356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_reciprocal_l743_74310

noncomputable section

open Real

theorem tan_sum_reciprocal (x y : ℝ) 
  (h1 : sin x / cos y + sin y / cos x = 2)
  (h2 : cos x / sin y + cos y / sin x = 4)
  (h3 : sin x * sin y = cos x * cos y) :
  tan x / tan y + tan y / tan x = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_reciprocal_l743_74310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l743_74380

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Helper function to calculate area (marked as noncomputable)
noncomputable def area_of_triangle (t : Triangle) : ℝ :=
  Real.sqrt (t.a * t.a - (t.c / 2) * (t.c / 2)) * t.c / 4

-- Define the theorem
theorem triangle_theorem (t : Triangle) 
  (h1 : t.a * Real.cos t.B = t.b * Real.cos t.A) 
  (h2 : t.c = Real.sqrt 3) 
  (h3 : t.a = 6) : 
  t.A = t.B ∧ 
  area_of_triangle t = Real.sqrt 423 / 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l743_74380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paddle_up_river_l743_74352

noncomputable def paddleTime (distance : ℝ) (stillSpeed : ℝ) (currentSpeed : ℝ) : ℝ :=
  distance / (stillSpeed - currentSpeed)

noncomputable def totalPaddleTime (stillSpeed : ℝ) (section1 : ℝ × ℝ) (section2 : ℝ × ℝ) (section3 : ℝ × ℝ) : ℝ :=
  paddleTime section1.1 stillSpeed section1.2 +
  paddleTime section2.1 stillSpeed section2.2 +
  paddleTime section3.1 stillSpeed section3.2

theorem paddle_up_river (stillSpeed : ℝ) (section1 : ℝ × ℝ) (section2 : ℝ × ℝ) (section3 : ℝ × ℝ)
  (h1 : stillSpeed = 10)
  (h2 : section1 = (5, 4))
  (h3 : section2 = (8, 6))
  (h4 : section3 = (7, 3)) :
  totalPaddleTime stillSpeed section1 section2 section3 = 23 / 6 := by
  sorry

-- Remove the #eval line as it's not computable
-- #eval totalPaddleTime 10 (5, 4) (8, 6) (7, 3)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paddle_up_river_l743_74352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sine_problem_l743_74367

/-- The angle between three points in ℝ × ℝ -/
noncomputable def angle (A B C : ℝ × ℝ) : ℝ := sorry

/-- Given four equally spaced points on a line and a point Q satisfying certain angle conditions,
    prove that the sine of twice a specific angle is 3/5. -/
theorem angle_sine_problem (E F G H Q : ℝ × ℝ) : 
  (∃ k : ℝ, k > 0 ∧ F.1 - E.1 = k ∧ G.1 - F.1 = k ∧ H.1 - G.1 = k ∧ E.2 = F.2 ∧ F.2 = G.2 ∧ G.2 = H.2) →
  (Real.cos (angle E Q G) = 3/5) →
  (Real.cos (angle F Q H) = 1/5) →
  Real.sin (2 * angle F Q G) = 3/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sine_problem_l743_74367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequalities_for_ordered_reals_l743_74391

theorem inequalities_for_ordered_reals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x < y) :
  (3 : ℝ) ^ abs (x - y) > 1 ∧ x * abs x < y * abs y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequalities_for_ordered_reals_l743_74391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l743_74398

/-- An ellipse with center at the origin, foci on the x-axis, and eccentricity 1/2 -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a
  h_eccentricity : (a^2 - b^2).sqrt / a = 1/2

/-- The parabola x^2 = 8√3y -/
def parabola (x y : ℝ) : Prop := x^2 = 8 * Real.sqrt 3 * y

/-- The focus of the parabola x^2 = 8√3y -/
noncomputable def parabola_focus : ℝ × ℝ := (0, 2 * Real.sqrt 3)

theorem ellipse_equation (C : Ellipse) 
  (h_vertex : (C.a, 0) = parabola_focus) :
  ∀ x y, x^2 / 16 + y^2 / 12 = 1 ↔ 
    x^2 / C.a^2 + y^2 / C.b^2 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l743_74398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_expression_equals_zero_l743_74322

/-- Given a line segment divided into two parts where the sum of the lesser part
    and twice the greater part equals the whole segment, R is the ratio of the
    lesser part to the greater part. -/
noncomputable def R : ℝ :=
  let w : ℝ := 1  -- Lesser part (arbitrary non-zero value)
  let l : ℝ := 1  -- Greater part (equal to w based on the problem conditions)
  w / l

/-- The main theorem stating that R^[R^(R^2-R^(-1))-R^(-1)]-R^(-1) = 0 -/
theorem ratio_expression_equals_zero : R^(R^(R^2 - R⁻¹) - R⁻¹) - R⁻¹ = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_expression_equals_zero_l743_74322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l743_74374

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := (2 * x - 1) / (x + 2)

-- Define the point of tangency
def point : ℝ × ℝ := (-1, -3)

-- Define the slope of the tangent line at the point
def m : ℝ := 5

-- Theorem statement
theorem tangent_line_equation :
  let (x₀, y₀) := point
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (5 * x - y + 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l743_74374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_complex_to_line_l743_74342

/-- The distance from a point corresponding to a complex number to a line in the complex plane -/
theorem distance_complex_to_line :
  let z : ℂ := 2 / (1 - Complex.I)
  let p : ℝ × ℝ := (z.re, z.im)
  let d := |p.2 - p.1 - 1| / Real.sqrt 2
  d = Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_complex_to_line_l743_74342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_of_cubic_parabola_l743_74308

/-- The surface area of rotation for the curve y = (1/3)x³ around the x-axis from x = 0 to x = 1 -/
noncomputable def surfaceAreaOfRotation : ℝ :=
  2 * Real.pi * ∫ x in (0: ℝ)..1, (1/3 * x^3) * Real.sqrt (1 + (x^2)^2)

theorem surface_area_of_cubic_parabola : 
  surfaceAreaOfRotation = Real.pi/9 * (2 * Real.sqrt 2 - 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_of_cubic_parabola_l743_74308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_difference_l743_74355

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y - 10 = 0

-- Define the line equation
def line_eq (x y : ℝ) : Prop := x + y - 8 = 0

-- Define the distance function from a point to the line
noncomputable def distance_to_line (x y : ℝ) : ℝ := 
  |x + y - 8| / Real.sqrt 2

-- Theorem statement
theorem distance_difference : 
  ∃ (max_dist min_dist : ℝ),
    (∀ (x y : ℝ), circle_eq x y → distance_to_line x y ≤ max_dist) ∧
    (∃ (x y : ℝ), circle_eq x y ∧ distance_to_line x y = max_dist) ∧
    (∀ (x y : ℝ), circle_eq x y → distance_to_line x y ≥ min_dist) ∧
    (∃ (x y : ℝ), circle_eq x y ∧ distance_to_line x y = min_dist) ∧
    max_dist - min_dist = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_difference_l743_74355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_divisor_of_100_factorial_l743_74397

def factorial (n : ℕ) : ℕ := (Finset.range n).prod (λ i ↦ i + 1)

theorem smallest_divisor_of_100_factorial :
  ∀ k : ℕ, k > 100 → k < 102 → ¬(factorial 100 % k = 0) ∧ (factorial 100 % 102 = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_divisor_of_100_factorial_l743_74397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_students_walking_home_l743_74334

/-- Proves the number of students walking home given total students and proportions of bus riders and bikers -/
theorem students_walking_home 
  (total_students : ℕ) 
  (bus_ratio : ℚ)
  (bike_ratio : ℚ) :
  total_students = 500 →
  bus_ratio = 1 / 5 →
  bike_ratio = 45 / 100 →
  (total_students : ℚ) * (1 - bus_ratio - bike_ratio) = 175 := by
  sorry

-- Remove the #eval line as it's causing issues with universe levels

end NUMINAMATH_CALUDE_ERRORFEEDBACK_students_walking_home_l743_74334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_emily_investment_growth_l743_74373

/-- Calculates the final amount after compound interest --/
def compound_interest (principal : ℝ) (rate : ℝ) (periods : ℕ) : ℝ :=
  principal * (1 + rate) ^ periods

theorem emily_investment_growth :
  let principal := 2500
  let annual_rate := 0.04
  let years := 21
  let periods := years * 2
  let biannual_rate := annual_rate / 2
  abs (compound_interest principal biannual_rate periods - 5510.10) < 0.01 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_emily_investment_growth_l743_74373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_max_F_l743_74305

noncomputable def F (A B x : ℝ) : ℝ :=
  |Real.cos x^2 + 2 * Real.sin x * Real.cos x - Real.sin x^2 + A * x + B|

theorem minimize_max_F :
  ∃ (M : ℝ), M = Real.sqrt 2 ∧
  (∀ A B : ℝ, ∃ (M' : ℝ), (∀ x : ℝ, 0 ≤ x ∧ x ≤ 3*Real.pi/2 → F A B x ≤ M') ∧ M ≤ M') ∧
  (∃ A B : ℝ, ∀ x : ℝ, 0 ≤ x ∧ x ≤ 3*Real.pi/2 → F A B x ≤ M) ∧
  (∀ A B : ℝ, (∀ x : ℝ, 0 ≤ x ∧ x ≤ 3*Real.pi/2 → F A B x ≤ M) → A = 0 ∧ B = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_max_F_l743_74305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_deviation_10_gt_100_l743_74384

/-- A fair coin is tossed n times, resulting in m heads. -/
structure CoinToss (n : ℕ) where
  m : ℕ
  h_m_le_n : m ≤ n

/-- The frequency of heads in a coin toss. -/
def frequency (n : ℕ) (ct : CoinToss n) : ℚ := ct.m / n

/-- The deviation of the frequency from 0.5. -/
def deviation (n : ℕ) (ct : CoinToss n) : ℚ := frequency n ct - 1/2

/-- The absolute deviation of the frequency from 0.5. -/
def abs_deviation (n : ℕ) (ct : CoinToss n) : ℚ := |deviation n ct|

/-- The expected value of a random variable. -/
noncomputable def expected_value {α : Type*} (X : α → ℚ) : ℚ := sorry

/-- The statement that the expected absolute deviation for 10 tosses
    is greater than for 100 tosses. -/
theorem abs_deviation_10_gt_100 :
  expected_value (λ ct : CoinToss 10 => abs_deviation 10 ct) >
  expected_value (λ ct : CoinToss 100 => abs_deviation 100 ct) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_deviation_10_gt_100_l743_74384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_english_math_difference_l743_74377

/-- Represents the number of students passing only English -/
def E : ℕ := sorry

/-- Represents the number of students passing only Math -/
def M : ℕ := sorry

/-- Represents the number of students passing both English and Math -/
def B : ℕ := sorry

/-- The total number of students passing English is 30 -/
axiom total_english : E + B = 30

/-- The total number of students passing Math is 20 -/
axiom total_math : M + B = 20

/-- Some students pass both subjects -/
axiom both_subjects : B ≥ 0

/-- The difference between students passing only English and only Math is 10 -/
theorem english_math_difference : E - M = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_english_math_difference_l743_74377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_6880_l743_74331

theorem cube_root_6880 (h : (6.88 : ℝ) ^ (1/3 : ℝ) = 1.902) : 
  (6880 : ℝ) ^ (1/3 : ℝ) = 19.02 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_6880_l743_74331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l743_74321

-- Define the hyperbola
def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y = 0

-- Define the asymptote
def asymptote (a b x y : ℝ) : Prop := b*x - a*y = 0

-- Define the distance from focus to asymptote
def focus_asymptote_distance (a b : ℝ) : ℝ := b

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x₀ y₀ : ℝ), circle_eq x₀ y₀ ∧ asymptote a b x₀ y₀) →
  focus_asymptote_distance a b = 2 →
  ∀ (x y : ℝ), hyperbola a b x y ↔ x^2 / 16 - y^2 / 4 = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l743_74321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_condition_for_parallel_planes_l743_74315

structure Plane where
  -- Define a plane
  dummy : Unit

structure Line where
  -- Define a line
  dummy : Unit

def parallel (a b : Plane) : Prop :=
  -- Define parallel planes
  sorry

def parallel_line_plane (l : Line) (p : Plane) : Prop :=
  -- Define a line parallel to a plane
  sorry

def parallel_lines (l1 l2 : Line) : Prop :=
  -- Define parallel lines
  sorry

def intersecting_lines (l1 l2 : Line) : Prop :=
  -- Define intersecting lines
  sorry

def line_in_plane (l : Line) (p : Plane) : Prop :=
  -- Define a line lying in a plane
  sorry

theorem sufficient_condition_for_parallel_planes
  (α β : Plane) (m n l₁ l₂ : Line)
  (h1 : m ≠ n)
  (h2 : line_in_plane m α)
  (h3 : line_in_plane n α)
  (h4 : intersecting_lines l₁ l₂)
  (h5 : line_in_plane l₁ β)
  (h6 : line_in_plane l₂ β)
  (h7 : parallel_lines m l₁)
  (h8 : parallel_lines n l₂) :
  parallel α β :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_condition_for_parallel_planes_l743_74315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l743_74387

/-- Two arithmetic sequences -/
def a : ℕ → ℚ := sorry
def b : ℕ → ℚ := sorry

/-- Sum of the first n terms of sequence a -/
def S (n : ℕ) : ℚ := (n : ℚ) * (a 1 + a n) / 2

/-- Sum of the first n terms of sequence b -/
def T (n : ℕ) : ℚ := (n : ℚ) * (b 1 + b n) / 2

/-- The main theorem -/
theorem arithmetic_sequence_ratio (n : ℕ) (h : n > 0) :
  (∀ k : ℕ, k > 0 → S k / T k = ((7 : ℚ) * k + 1) / ((4 : ℚ) * k + 27)) →
  a 6 / b 6 = 78 / 71 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l743_74387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_line_to_circle_l743_74364

/-- The shortest distance between a point on the line y = x - 1 and a point on the circle x^2 + y^2 + 4x - 2y + 4 = 0 is 1. -/
theorem shortest_distance_line_to_circle : 
  let line := {p : ℝ × ℝ | p.2 = p.1 - 1}
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 + 4*p.1 - 2*p.2 + 4 = 0}
  ∃ (d : ℝ), d = 1 ∧ 
    (∀ (p₁ : ℝ × ℝ) (p₂ : ℝ × ℝ), p₁ ∈ line → p₂ ∈ circle → dist p₁ p₂ ≥ d) ∧
    (∃ (p₁ : ℝ × ℝ) (p₂ : ℝ × ℝ), p₁ ∈ line ∧ p₂ ∈ circle ∧ dist p₁ p₂ = d) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_line_to_circle_l743_74364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pi_div_3_irrational_sqrt4_rational_frac22_7_rational_repeating_decimal_rational_l743_74346

-- Define the given numbers
def sqrt4 : ℚ := 2
def frac22_7 : ℚ := 22 / 7
def repeating_decimal : ℚ := 101 / 99

-- State that π is irrational
axiom pi_irrational : Irrational Real.pi

-- Theorem to prove
theorem pi_div_3_irrational : Irrational (Real.pi / 3) := by
  sorry

-- Additional statements to show other numbers are rational
theorem sqrt4_rational : ∃ (a b : ℤ), sqrt4 = a / b := by
  sorry

theorem frac22_7_rational : ∃ (a b : ℤ), frac22_7 = a / b := by
  sorry

theorem repeating_decimal_rational : ∃ (a b : ℤ), repeating_decimal = a / b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pi_div_3_irrational_sqrt4_rational_frac22_7_rational_repeating_decimal_rational_l743_74346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l743_74396

noncomputable def f (x : ℝ) := Real.log x / Real.log 3 + x - 3

theorem root_in_interval :
  ∃ x : ℝ, 2 < x ∧ x < 3 ∧ f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l743_74396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_routes_for_robust_system_l743_74353

/-- Represents an airline route system -/
structure AirlineSystem where
  num_cities : ℕ
  num_companies : ℕ
  routes : List (ℕ × ℕ × ℕ)  -- (from_city, to_city, company)

/-- Checks if the system is connected even if one company ceases operations -/
def is_robust (system : AirlineSystem) : Prop :=
  ∀ (stopped_company : ℕ), 
    stopped_company < system.num_companies →
    ∀ (city1 city2 : ℕ), 
      city1 < system.num_cities → 
      city2 < system.num_cities →
      ∃ (path : List (ℕ × ℕ × ℕ)), 
        path ⊆ system.routes ∧
        path.all (λ (_, _, company) => company ≠ stopped_company) ∧
        path.head?.map Prod.fst = some city1 ∧
        path.getLast?.map Prod.fst = some city2

/-- The main theorem -/
theorem min_routes_for_robust_system :
  ∀ (system : AirlineSystem),
    system.num_cities = 15 →
    system.num_companies = 3 →
    is_robust system →
    system.routes.length ≥ 21 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_routes_for_robust_system_l743_74353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spinner_probability_l743_74306

def spinner_X : Finset ℕ := {1, 4, 5}
def spinner_Y : Finset ℕ := {1, 2, 3}
def spinner_Z : Finset ℕ := {2, 4, 6}

def is_even (n : ℕ) : Bool := n % 2 = 0

def total_outcomes : ℕ := (spinner_X.card) * (spinner_Y.card) * (spinner_Z.card)

def favorable_outcomes : ℕ := 
  (spinner_X.filter (fun x => is_even x)).card * (spinner_Y.filter (fun y => is_even y)).card * spinner_Z.card +
  (spinner_X.filter (fun x => !is_even x)).card * (spinner_Y.filter (fun y => !is_even y)).card * spinner_Z.card

theorem spinner_probability : 
  (favorable_outcomes : ℚ) / total_outcomes = 5 / 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spinner_probability_l743_74306
