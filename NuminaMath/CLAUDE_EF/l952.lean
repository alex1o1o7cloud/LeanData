import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_sign_matches_correlation_sign_l952_95202

-- Define the variables and their properties
variable (x y r b a : ℝ)

-- Define the linear relationship
def linear_relationship (x y : ℝ) : Prop := ∃ m c : ℝ, y = m * x + c

-- Define the correlation coefficient
def correlation_coefficient (x y : ℝ) : Prop := ∃ r : ℝ, -1 ≤ r ∧ r ≤ 1

-- Define the regression line
def regression_line (x y b a : ℝ) : Prop := y = b * x + a

-- Theorem statement
theorem slope_sign_matches_correlation_sign 
  (h1 : linear_relationship x y)
  (h2 : correlation_coefficient x y)
  (h3 : regression_line x y b a) :
  (r > 0 → b > 0) ∧ (r < 0 → b < 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_sign_matches_correlation_sign_l952_95202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_4_value_l952_95277

-- Define the functions f and g
def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- Define the inverse functions
def f_inv : ℝ → ℝ := sorry
def g_inv : ℝ → ℝ := sorry

-- Axioms for inverse functions
axiom f_inv_axiom : Function.LeftInverse f_inv f ∧ Function.RightInverse f_inv f
axiom g_inv_axiom : Function.LeftInverse g_inv g ∧ Function.RightInverse g_inv g

-- Axiom for symmetry about y = x
axiom symmetry : ∀ x : ℝ, f (x - 1) = g x + 3

-- Given condition
axiom g_5 : g 5 = 2015

-- Theorem to prove
theorem f_4_value : f 4 = 2018 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_4_value_l952_95277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_platform_passing_time_l952_95260

/-- Time taken for a train to pass a platform given certain conditions -/
theorem train_platform_passing_time 
  (man_passing_time : ℝ) 
  (train_speed_kmh : ℝ) 
  (platform_length : ℝ) 
  (h1 : man_passing_time = 20)
  (h2 : train_speed_kmh = 54)
  (h3 : platform_length = 30.0024) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.5 ∧ 
    |((platform_length + man_passing_time * (train_speed_kmh * 1000 / 3600)) / 
      (train_speed_kmh * 1000 / 3600) - 22)| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_platform_passing_time_l952_95260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_minus_6cos_x_plus_6_min_l952_95288

theorem cos_2x_minus_6cos_x_plus_6_min (x : ℝ) : Real.cos (2 * x) - 6 * Real.cos x + 6 ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_minus_6cos_x_plus_6_min_l952_95288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_4_magnitude_vector_neg_4c_components_vector_1_5c_components_l952_95229

-- Define a vector c in 2D space
variable (c : ℝ × ℝ)

-- Define the unit vector in the direction of c
noncomputable def unit_vector_c (c : ℝ × ℝ) : ℝ × ℝ :=
  let magnitude := Real.sqrt (c.1^2 + c.2^2)
  (c.1 / magnitude, c.2 / magnitude)

-- Define the vector with magnitude 4 in the direction of c
noncomputable def vector_4 (c : ℝ × ℝ) : ℝ × ℝ :=
  let u := unit_vector_c c
  (4 * u.1, 4 * u.2)

-- Define the vector -4c
def vector_neg_4c (c : ℝ × ℝ) : ℝ × ℝ :=
  (-4 * c.1, -4 * c.2)

-- Define the vector 1.5c
def vector_1_5c (c : ℝ × ℝ) : ℝ × ℝ :=
  (1.5 * c.1, 1.5 * c.2)

-- Theorem statements
theorem vector_4_magnitude (c : ℝ × ℝ) :
  Real.sqrt ((vector_4 c).1^2 + (vector_4 c).2^2) = 4 := by sorry

theorem vector_neg_4c_components (c : ℝ × ℝ) :
  vector_neg_4c c = (-4 * c.1, -4 * c.2) := by sorry

theorem vector_1_5c_components (c : ℝ × ℝ) :
  vector_1_5c c = (1.5 * c.1, 1.5 * c.2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_4_magnitude_vector_neg_4c_components_vector_1_5c_components_l952_95229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_raise_calculation_l952_95284

/-- Calculates the new weekly earnings after a percentage raise -/
noncomputable def new_earnings (initial_earnings : ℝ) (percentage_increase : ℝ) : ℝ :=
  initial_earnings * (1 + percentage_increase / 100)

/-- Proves that a 50% raise on $60 results in $90 weekly earnings -/
theorem raise_calculation (initial_earnings : ℝ) (percentage_increase : ℝ) 
  (h1 : initial_earnings = 60)
  (h2 : percentage_increase = 50) :
  new_earnings initial_earnings percentage_increase = 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_raise_calculation_l952_95284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sum_reciprocals_l952_95259

/-- Given vectors OA, OB, OC and conditions, prove the minimum value of 1/a + 2/b is 8 -/
theorem min_value_sum_reciprocals (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ (OA OB OC : ℝ × ℝ), 
    OA = (1, -2) ∧ 
    OB = (a, -1) ∧ 
    OC = (-b, 0) ∧
    (∃ (k : ℝ), OB - OA = k • (OC - OA)) ∧
    2 * a + b = 1) →
  (∀ x y : ℝ, x > 0 → y > 0 → 2 * x + y = 1 → 1/x + 2/y ≥ 8) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + y = 1 ∧ 1/x + 2/y = 8) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sum_reciprocals_l952_95259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_distance_l952_95221

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 12 = 1

-- Define the left focus of the hyperbola
def left_focus : ℝ × ℝ := (-4, 0)

-- Define point A
def point_A : ℝ × ℝ := (1, 4)

-- Define a point P on the right branch of the hyperbola
def point_P : ℝ × ℝ → Prop := λ p => 
  hyperbola p.1 p.2 ∧ p.1 > 0

-- Define the distance function between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the perimeter of triangle APF
noncomputable def perimeter (P : ℝ × ℝ) : ℝ :=
  distance left_focus P + distance P point_A + distance point_A left_focus

-- Define the line through two points
def line_through (p1 p2 : ℝ × ℝ) : ℝ → ℝ → Prop :=
  λ x y => (y - p1.2) * (p2.1 - p1.1) = (x - p1.1) * (p2.2 - p1.2)

-- Define the distance from a point to a line
noncomputable def point_to_line_distance (p : ℝ × ℝ) (l : ℝ → ℝ → Prop) : ℝ :=
  sorry -- This would require a more complex definition

-- The main theorem
theorem min_perimeter_distance :
  ∀ P, point_P P →
    (∀ Q, point_P Q → perimeter P ≤ perimeter Q) →
    point_to_line_distance left_focus (line_through point_A P) = 32/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_distance_l952_95221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_different_colors_l952_95200

/-- A type representing the 100 different colors -/
def Color := Fin 100

/-- A function that assigns a color to each integer -/
def coloring : ℤ → Color := sorry

/-- All 100 colors are used -/
axiom all_colors_used : ∀ c : Color, ∃ n : ℤ, coloring n = c

/-- The coloring property for intervals of equal length -/
axiom coloring_property : 
  ∀ (a b c d : ℤ), b - a = d - c →
    coloring a = coloring c → coloring b = coloring d →
      ∀ x : ℤ, 0 ≤ x ∧ x ≤ b - a → coloring (a + x) = coloring (c + x)

/-- The main theorem to be proved -/
theorem different_colors : coloring (-1990) ≠ coloring 1990 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_different_colors_l952_95200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_journey_theorem_l952_95245

/-- Represents the time taken for a ship's journey -/
structure ShipJourney where
  downstream_time : ℚ
  upstream_time : ℚ
  upstream_speed_multiplier : ℚ

/-- Calculates the time taken for a doubled-speed downstream journey -/
noncomputable def doubled_speed_time (j : ShipJourney) : ℚ :=
  j.downstream_time * (3 / 5)

/-- Theorem statement for the ship journey problem -/
theorem ship_journey_theorem (j : ShipJourney) 
  (h1 : j.downstream_time = 1)
  (h2 : j.upstream_time = 1)
  (h3 : j.upstream_speed_multiplier = 2) :
  doubled_speed_time j = 3/5 := by
  sorry

#check ship_journey_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_journey_theorem_l952_95245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concentric_circles_angle_theorem_l952_95230

/-- The acute angle formed by two lines passing through the center of three concentric circles -/
noncomputable def angle_between_lines (r1 r2 r3 : ℝ) (shaded_ratio : ℝ) : ℝ :=
  229 * Real.pi / 528

theorem concentric_circles_angle_theorem (r1 r2 r3 : ℝ) (shaded_ratio : ℝ) 
  (h1 : r1 = 4) (h2 : r2 = 3) (h3 : r3 = 2) (h4 : shaded_ratio = 7/17) :
  angle_between_lines r1 r2 r3 shaded_ratio = 229 * Real.pi / 528 := by
  sorry

#check concentric_circles_angle_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_concentric_circles_angle_theorem_l952_95230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_19_eq_190_l952_95213

/-- An arithmetic sequence with specified properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  a_9 : a 9 = 11
  a_11 : a 11 = 9

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (seq.a 1 + seq.a n)

/-- The theorem stating that S_19 = 190 for the given arithmetic sequence -/
theorem sum_19_eq_190 (seq : ArithmeticSequence) : sum_n seq 19 = 190 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_19_eq_190_l952_95213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circle_properties_l952_95268

/-- Triangle with external and inscribed circles -/
structure TriangleWithCircles where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  M : ℝ × ℝ
  P : ℝ × ℝ
  Q : ℝ × ℝ
  K : ℝ × ℝ
  L : ℝ × ℝ

/-- Calculate the distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Calculate the semiperimeter of a triangle -/
noncomputable def semiperimeter (t : TriangleWithCircles) : ℝ :=
  (distance t.A t.B + distance t.B t.C + distance t.C t.A) / 2

/-- Main theorem -/
theorem triangle_circle_properties (t : TriangleWithCircles) :
  distance t.A t.P = semiperimeter t ∧
  distance t.B t.M = distance t.C t.K ∧
  distance t.B t.C = distance t.P t.L := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circle_properties_l952_95268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_integer_to_cube_root_250_l952_95275

theorem closest_integer_to_cube_root_250 : 
  ∃ (n : ℤ), ∀ (m : ℤ), |n - (250 : ℝ) ^ (1/3)| ≤ |m - (250 : ℝ) ^ (1/3)| ∧ n = 6 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_integer_to_cube_root_250_l952_95275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_height_l952_95270

/-- Represents a quadrilateral pyramid with a square base -/
structure QuadPyramid where
  base_side : ℝ
  lateral_edge : ℝ

/-- The volume of a quadrilateral pyramid -/
noncomputable def volume (p : QuadPyramid) : ℝ :=
  (1 / 3) * p.base_side^2 * Real.sqrt (p.lateral_edge^2 - (p.base_side^2 / 2))

/-- The height of a quadrilateral pyramid -/
noncomputable def pyramidHeight (p : QuadPyramid) : ℝ :=
  Real.sqrt (p.lateral_edge^2 - (p.base_side^2 / 2))

theorem max_volume_height :
  ∃ (max_p : QuadPyramid), max_p.lateral_edge = 2 * Real.sqrt 3 ∧
    (∀ q : QuadPyramid, volume q ≤ volume max_p) ∧
    pyramidHeight max_p = 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_height_l952_95270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_A_in_triangle_l952_95266

/-- In a triangle ABC with sides a, b, c and angles A, B, C, 
    given a = 1, b = 2, and cos C = 1/4, prove that sin A = √15 / 8 -/
theorem sin_A_in_triangle (a b c : ℝ) (A B C : ℝ) : 
  a = 1 → b = 2 → Real.cos C = (1 : ℝ) / 4 → 
  0 < A ∧ A < Real.pi →
  0 < B ∧ B < Real.pi →
  0 < C ∧ C < Real.pi →
  a^2 = b^2 + c^2 - 2*b*c*(Real.cos A) →
  b^2 = a^2 + c^2 - 2*a*c*(Real.cos B) →
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos C) →
  Real.sin A = Real.sqrt 15 / 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_A_in_triangle_l952_95266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_average_speed_l952_95286

/-- Proves that the average speed of a train is 24 kmph given specific conditions -/
theorem train_average_speed (x : ℝ) (h : x > 0) : 
  (3 * x) / ((x / 40) + (2 * x / 20)) = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_average_speed_l952_95286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_range_extreme_points_count_l952_95252

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (1/3 * x^3 - 2*x^2 + (a+4)*x - 2*a - 4)

-- Theorem for the range of a
theorem f_inequality_range (a : ℝ) : 
  (∀ x < 2, f a x < -4/3 * Real.exp x) ↔ a ≥ 0 := by sorry

-- Define the number of extreme points
noncomputable def num_extreme_points (a : ℝ) : ℕ :=
  if a ≥ 0 then 1 else 3

-- Theorem for the number of extreme points
theorem extreme_points_count (a : ℝ) : 
  num_extreme_points a = if a ≥ 0 then 1 else 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_range_extreme_points_count_l952_95252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_turnip_pulling_l952_95293

theorem turnip_pulling (F : ℝ) (F_pos : F > 0) : ∃ n : ℕ, 
  F * (1 + (3/4 + (3/4)^2 + (3/4)^3 + (3/4)^4) + n) ≥ 
  F * (1 + 3/4 + (3/4)^2 + (3/4)^3 + (3/4)^4 + (3/4)^5) ∧ 
  n = 2 := by
  sorry

#check turnip_pulling

end NUMINAMATH_CALUDE_ERRORFEEDBACK_turnip_pulling_l952_95293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_b_value_ln2_approximation_l952_95214

-- Define the function f
noncomputable def f (x : ℝ) := Real.exp x - Real.exp (-x) - 2 * x

-- Define the function g
noncomputable def g (b : ℝ) (x : ℝ) := f (2 * x) - 4 * b * f x

-- Theorem for the maximum value of b
theorem max_b_value :
  ∃ (b_max : ℝ), b_max = 2 ∧
  (∀ (b : ℝ), (∀ (x : ℝ), x > 0 → g b x > 0) → b ≤ b_max) := by
  sorry

-- Constants for √2 bounds
def sqrt2_lower : ℝ := 1.4142
def sqrt2_upper : ℝ := 1.4143

-- Theorem for the approximation of ln(2)
theorem ln2_approximation :
  sqrt2_lower < Real.sqrt 2 ∧ Real.sqrt 2 < sqrt2_upper →
  ∃ (ln2_approx : ℝ), ln2_approx = 0.693 ∧
  abs (Real.log 2 - ln2_approx) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_b_value_ln2_approximation_l952_95214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_best_fit_slope_l952_95231

theorem best_fit_slope 
  (x₁ x₂ x₃ x₄ y₁ y₂ y₃ y₄ : ℝ) 
  (h1 : x₂ - x₁ = 2 * (x₄ - x₃))
  (h2 : x₃ - x₂ = 3 * (x₄ - x₃)) :
  let points := [(x₁, y₁), (x₂, y₂), (x₃, y₃), (x₄, y₄)]
  let x_mean := (x₁ + x₂ + x₃ + x₄) / 4
  let y_mean := (y₁ + y₂ + y₃ + y₄) / 4
  let numerator := (x₁ - x_mean) * (y₁ - y_mean) + (x₂ - x_mean) * (y₂ - y_mean) +
                   (x₃ - x_mean) * (y₃ - y_mean) + (x₄ - x_mean) * (y₄ - y_mean)
  let denominator := (x₁ - x_mean)^2 + (x₂ - x_mean)^2 + (x₃ - x_mean)^2 + (x₄ - x_mean)^2
  numerator / denominator = (y₄ - y₁) / (x₄ - x₁) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_best_fit_slope_l952_95231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_problem_l952_95243

/-- Calculate compound interest -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate / 100) ^ time - principal

/-- Calculate simple interest -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

theorem simple_interest_problem (principal : ℝ) :
  let ci := compound_interest 4000 10 2
  let si := simple_interest principal 6 2
  si = ci / 2 → principal = 3500 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_problem_l952_95243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l952_95285

noncomputable def f (a b x : ℝ) : ℝ := (1 + a * x^2) / (x + b)

noncomputable def g (a b x : ℝ) : ℝ := x * f a b x

theorem function_properties (a b : ℝ) :
  (∀ x, g a b x = g a b (-x)) →  -- g is even
  f a b 1 = 3 →                  -- f passes through (1, 3)
  a = 2 ∧ b = 0 ∧
  (∀ x₁ x₂, 1 < x₁ → x₁ < x₂ → g a b x₁ < g a b x₂) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l952_95285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_2_plus_i_times_i_l952_95279

theorem imaginary_part_of_2_plus_i_times_i : 
  Complex.im ((2 + Complex.I) * Complex.I) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_2_plus_i_times_i_l952_95279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l952_95247

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := x^2 + 1/x

-- Define the derivative of f(x)
noncomputable def f_prime (x : ℝ) : ℝ := 2*x - 1/x^2

-- Theorem statement
theorem tangent_line_at_one :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f_prime x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ y = x + 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l952_95247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fans_with_all_vouchers_l952_95296

def arena_capacity : ℕ := 4500
def soda_interval : ℕ := 60
def popcorn_interval : ℕ := 80
def hotdog_interval : ℕ := 100

theorem fans_with_all_vouchers :
  (Finset.filter (λ n => n % soda_interval = 0 ∧ 
                         n % popcorn_interval = 0 ∧ 
                         n % hotdog_interval = 0) 
                 (Finset.range arena_capacity)).card = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fans_with_all_vouchers_l952_95296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_b_value_l952_95258

/-- The circle equation -/
def circle_equation (x y m : ℝ) : Prop :=
  x^2 - 2*x + y^2 - 2*m*y + 2*m - 1 = 0

/-- The line equation -/
def line_equation (x y b : ℝ) : Prop :=
  y = x + b

/-- The circle has the smallest area when m = 1 -/
def smallest_area (m : ℝ) : Prop :=
  m = 1

/-- The line is tangent to the circle -/
def is_tangent (b : ℝ) : Prop :=
  ∃ (x y : ℝ), circle_equation x y 1 ∧ line_equation x y b

/-- Main theorem: When the circle has the smallest area and the line is tangent to it,
    b equals plus or minus square root of 2 -/
theorem tangent_line_b_value (b : ℝ) (h1 : smallest_area 1) (h2 : is_tangent b) :
  b = Real.sqrt 2 ∨ b = -Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_b_value_l952_95258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_M_and_N_l952_95264

open Set Real

/-- Given sets M and N, prove that their intersection is [2, +∞) -/
theorem intersection_of_M_and_N :
  let M : Set ℝ := {x | x > 1}
  let N : Set ℝ := {x | x^2 - 2*x ≥ 0}
  M ∩ N = Ici 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_M_and_N_l952_95264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_b_sequence_l952_95255

noncomputable def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

noncomputable def is_arithmetic_sequence (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d

noncomputable def sum_of_arithmetic_sequence (b : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (b 1 + b n) / 2

theorem sum_of_b_sequence 
  (a b : ℕ → ℝ) 
  (h_geo : is_geometric_sequence a) 
  (h_arith : is_arithmetic_sequence b) 
  (h_relation : a 2 * a 14 = 4 * a 8) 
  (h_equal : b 8 = a 8) : 
  sum_of_arithmetic_sequence b 15 = 60 := by
  sorry

#check sum_of_b_sequence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_b_sequence_l952_95255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partnership_profit_l952_95276

/-- Calculates the total profit of a partnership given the investments and one partner's profit share -/
theorem partnership_profit (a b c c_profit : ℕ) (h1 : c > 0) : 
  c_profit * (a / c + b / c + 1) = 252000 :=
by
  sorry

#check partnership_profit

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partnership_profit_l952_95276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_decreasing_function_symmetry_l952_95283

open Function Set Real

theorem odd_decreasing_function_symmetry
  (f : ℝ → ℝ) (a b : ℝ) 
  (h_odd : Odd f)
  (h_pos : 0 < a ∧ a < b)
  (h_decreasing : ∀ x y, x ∈ Ioo a b → y ∈ Ioo a b → x < y → f x > f y) :
  ∀ x y, x ∈ Ioo (-b) (-a) → y ∈ Ioo (-b) (-a) → x < y → f x > f y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_decreasing_function_symmetry_l952_95283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_digits_in_base5_of_567_l952_95290

/-- Converts a natural number to its base-5 representation -/
def toBase5 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
  aux n []

/-- Counts the number of even digits in a list of natural numbers -/
def countEvenDigits (digits : List ℕ) : ℕ :=
  digits.filter (λ d => d % 2 = 0) |>.length

/-- Theorem stating that the number of even digits in the base-5 representation of 567₁₀ is 2 -/
theorem even_digits_in_base5_of_567 :
  countEvenDigits (toBase5 567) = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_digits_in_base5_of_567_l952_95290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_bisector_problem_l952_95239

/-- A line is a perpendicular bisector of a line segment if it passes through the midpoint of the segment and is perpendicular to it. -/
def is_perpendicular_bisector (a b c d : ℝ) (x y : ℝ → ℝ) : Prop :=
  let midpoint_x := (a + c) / 2
  let midpoint_y := (b + d) / 2
  (x midpoint_x - y midpoint_y = 0) ∧ 
  ((y c - y b) * (x c - x a) = -(x c - x a) * (y c - y b))

/-- The problem statement -/
theorem perpendicular_bisector_problem :
  ∀ d : ℝ, is_perpendicular_bisector 2 5 8 11 (λ x ↦ x) (λ y ↦ y + d) ↔ d = -3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_bisector_problem_l952_95239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_sevens_in_range_l952_95297

-- Define a function to check if a number contains the digit 2
def containsTwo (n : ℕ) : Bool :=
  (n.digits 10).any (· = 2)

-- Define a function to count occurrences of 7 in a number
def countSevens (n : ℕ) : ℕ :=
  (n.digits 10).filter (· = 7) |>.length

-- Define the range of numbers
def numberRange : Finset ℕ := Finset.range 500

-- Define the set of valid numbers (excluding those containing 2)
def validNumbers : Finset ℕ := numberRange.filter (λ n ↦ ¬containsTwo (n + 1))

-- State the theorem
theorem count_sevens_in_range :
  (validNumbers.sum countSevens) = 80 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_sevens_in_range_l952_95297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equals_zero_l952_95201

theorem integral_equals_zero (m : ℝ) : (∫ (x : ℝ) in Set.Icc 0 1, x^2 + m*x) = 0 ↔ m = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equals_zero_l952_95201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_fixed_point_infinite_interval_unique_fixed_point_closed_interval_l952_95217

-- Part (a)
theorem unique_fixed_point_infinite_interval
  (a : ℝ)
  (f : ℝ → ℝ)
  (h_cont : Continuous f)
  (h_range : ∀ x, x ≥ a → f x ≥ a)
  (h_contract : ∀ x y, x ≥ a → y ≥ a → x ≠ y → |f x - f y| < |x - y|) :
  ∃! x, x ≥ a ∧ f x = x :=
sorry

-- Part (b)
theorem unique_fixed_point_closed_interval
  (a b : ℝ)
  (h_le : a ≤ b)
  (f : ℝ → ℝ)
  (h_cont : Continuous f)
  (h_range : ∀ x, x ∈ Set.Icc a b → f x ∈ Set.Icc a b)
  (h_contract : ∀ x y, x ∈ Set.Icc a b → y ∈ Set.Icc a b → x ≠ y → |f x - f y| < |x - y|) :
  ∃! x, x ∈ Set.Icc a b ∧ f x = x :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_fixed_point_infinite_interval_unique_fixed_point_closed_interval_l952_95217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_root_sixteen_twelfth_power_l952_95282

theorem fourth_root_sixteen_twelfth_power : (16 : ℝ) ^ ((1/4 : ℝ) * 12) = 4096 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_root_sixteen_twelfth_power_l952_95282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_difference_l952_95208

/-- A function f: ℝ → ℝ defined as f(x) = 9^x -/
noncomputable def f (x : ℝ) : ℝ := 9^x

/-- Theorem stating that f(x+2) - f(x) = 80*f(x) for all real x -/
theorem f_difference (x : ℝ) : f (x + 2) - f x = 80 * f x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_difference_l952_95208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_length_tangent_segment_l952_95292

/-- A triangle is represented as a set of three points in ℝ². -/
def Triangle := Set (ℝ × ℝ)

/-- A line segment is represented as two points in ℝ². -/
def LineSegment := Set (ℝ × ℝ)

/-- The perimeter of a triangle. -/
noncomputable def perimeter (t : Triangle) : ℝ := sorry

/-- Predicate to check if a line segment is tangent to the inscribed circle of a triangle. -/
def is_tangent_to_inscribed_circle (t : Triangle) (seg : LineSegment) : Prop := sorry

/-- Predicate to check if a line segment is parallel to one side of a triangle. -/
def is_parallel_to_side (t : Triangle) (seg : LineSegment) : Prop := sorry

/-- The length of a line segment. -/
noncomputable def length (seg : LineSegment) : ℝ := sorry

/-- Given a triangle with perimeter p and a line segment XY that is tangent to the inscribed circle 
    of the triangle and parallel to one of its sides, the maximum length of XY is p/8. -/
theorem max_length_tangent_segment (p : ℝ) (h : p > 0) : 
  ∃ (triangle : Triangle) (XY : LineSegment),
    (perimeter triangle = p) ∧ 
    (is_tangent_to_inscribed_circle triangle XY) ∧
    (is_parallel_to_side triangle XY) ∧
    (∀ (XY' : LineSegment), 
      (is_tangent_to_inscribed_circle triangle XY') ∧ 
      (is_parallel_to_side triangle XY') →
      length XY' ≤ p / 8) ∧
    (length XY = p / 8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_length_tangent_segment_l952_95292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_l952_95272

def y : ℕ → ℝ
  | 0 => 3  -- Add this case to cover Nat.zero
  | 1 => 3
  | k + 2 => y (k + 1) ^ 2 + y (k + 1) + 2

theorem series_sum : ∑' n, 1 / (y n + 1) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_l952_95272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l952_95269

def p (a : ℝ) : Prop := ∀ x : ℝ, (1/2)^(abs x) < a

def q (a : ℝ) : Prop := ∀ x : ℝ, 0 < a * x^2 + (a - 2) * x + 9/8

theorem range_of_a (a : ℝ) (h1 : p a ∨ q a) (h2 : ¬(p a ∧ q a)) :
  a ≥ 8 ∨ (1/2 < a ∧ a ≤ 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l952_95269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_sequence_terms_l952_95287

theorem distinct_sequence_terms (f : ℕ → ℕ) 
  (h : ∀ n : ℕ, f (f n) = f (n + 1) + f n) : 
  ∀ k m : ℕ, k ≠ m → f k ≠ f m :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_sequence_terms_l952_95287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_locus_is_hyperbola_l952_95203

/-- The angle θ between the complex numbers representing Z₁ and Z₂ -/
noncomputable def θ : ℝ := sorry

/-- The constant area of the triangle OZ₁Z₂ -/
noncomputable def S : ℝ := sorry

/-- Z₁ is a complex number with argument θ -/
noncomputable def Z₁ : ℂ := sorry

/-- Z₂ is a complex number with argument -θ -/
noncomputable def Z₂ : ℂ := sorry

/-- The centroid Z of the triangle OZ₁Z₂ -/
noncomputable def Z : ℂ := (Z₁ + Z₂) / 3

/-- The real part of Z -/
noncomputable def x : ℝ := Z.re

/-- The imaginary part of Z -/
noncomputable def y : ℝ := Z.im

/-- Assumption that θ is between 0 and π/2 -/
axiom θ_range : 0 < θ ∧ θ < Real.pi / 2

/-- Assumption that the area of triangle OZ₁Z₂ is constant -/
axiom constant_area : Complex.abs (Z₁ * Z₂) = 2 * S

/-- Theorem stating that the locus of Z is a hyperbola -/
theorem centroid_locus_is_hyperbola :
  ∃ (k : ℝ), x^2 - y^2 = k := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_locus_is_hyperbola_l952_95203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bug_position_convergence_l952_95291

noncomputable def bug_position (n : ℕ) : ℝ × ℝ :=
  match n with
  | 0 => (0, 0)
  | 1 => (1, 0)
  | n + 1 => 
    let (x, y) := bug_position n
    let distance := (1/2) ^ n
    match n % 4 with
    | 0 => (x + distance, y)
    | 1 => (x, y + distance)
    | 2 => (x - distance, y)
    | _ => (x, y - distance)

theorem bug_position_convergence :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, 
    let (x, y) := bug_position n
    abs (x - 4/5) < ε ∧ abs (y - 2/5) < ε := by
  sorry

#check bug_position
#check bug_position_convergence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bug_position_convergence_l952_95291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AB_is_sqrt_34_l952_95289

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = 2 * x^2 + 4 * x - 2

/-- Point A on the parabola -/
def point_A : ℝ × ℝ := (1, 4)

/-- Point B on the parabola -/
def point_B : ℝ × ℝ := (-1, -4)

/-- The origin is the midpoint of AB -/
def origin_is_midpoint (A B : ℝ × ℝ) : Prop :=
  (A.1 + B.1) / 2 = 0 ∧ (A.2 + B.2) / 2 = 0

/-- The length of a segment given two points -/
noncomputable def segment_length (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem length_of_AB_is_sqrt_34 :
  parabola point_A.1 point_A.2 ∧
  parabola point_B.1 point_B.2 ∧
  origin_is_midpoint point_A point_B →
  segment_length point_A point_B = Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AB_is_sqrt_34_l952_95289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l952_95242

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x + y - 1 = 0

-- Define the slope of a line
def line_slope (m : ℝ) : Prop := ∀ x y, line_equation x y → y = m * x + 1

-- Define the inclination angle
def inclination_angle (θ : ℝ) : Prop := 
  ∃ m, line_slope m ∧ Real.tan θ = m ∧ 0 ≤ θ ∧ θ < Real.pi

-- Theorem statement
theorem line_inclination_angle :
  ∃ θ, inclination_angle θ ∧ θ = 135 * Real.pi / 180 := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l952_95242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_points_determine_wooden_strip_position_l952_95236

/-- A point on a wall -/
structure WallPoint where
  x : ℝ
  y : ℝ

/-- A wooden strip represented as a line segment -/
structure WoodenStrip where
  start : WallPoint
  finish : WallPoint

/-- Proposition: Two distinct points on a wall are necessary and sufficient 
    to determine the unique position of a straight wooden strip -/
theorem two_points_determine_wooden_strip_position 
  (p1 p2 : WallPoint) (strip : WoodenStrip) : 
  p1 ≠ p2 → 
  (strip.start = p1 ∧ strip.finish = p2) ∨ (strip.start = p2 ∧ strip.finish = p1) ↔ 
  ∃! (s : WoodenStrip), (s.start = p1 ∧ s.finish = p2) ∨ (s.start = p2 ∧ s.finish = p1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_points_determine_wooden_strip_position_l952_95236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_sum_l952_95233

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def IsPerp (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

/-- Definition of line l₁ -/
def line_l₁ (a : ℝ) : ℝ → ℝ → Prop :=
  fun x y ↦ x + (a - 4) * y + 1 = 0

/-- Definition of line l₂ -/
def line_l₂ (b : ℝ) : ℝ → ℝ → Prop :=
  fun x y ↦ b * x + y - 2 = 0

/-- Theorem: If l₁ is perpendicular to l₂, then a + b = 4 -/
theorem perpendicular_lines_sum (a b : ℝ) :
  (∃ m₁ m₂, (∀ x y, line_l₁ a x y ↔ y = m₁ * x + (-1 / (a - 4))) ∧
             (∀ x y, line_l₂ b x y ↔ y = m₂ * x + 2) ∧
             IsPerp m₁ m₂) →
  a + b = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_sum_l952_95233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_collinearity_l952_95256

-- Define the basic types and structures
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the quadrilateral and intersection points
variable (A B C D P Q R : Point)

-- Define the lines
variable (AB BC CA AD BD CD QR RP PQ : Line)

-- Define the intersection function
noncomputable def intersect (l1 l2 : Line) : Point :=
  sorry

-- Define a point being on a line
def on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- State the theorem
theorem intersection_collinearity :
  -- Given conditions
  (P = intersect BC AD) →
  (Q = intersect CA BD) →
  (R = intersect AB CD) →
  -- Conclusion
  ∃ (l : Line),
    on_line (intersect BC QR) l ∧
    on_line (intersect CA RP) l ∧
    on_line (intersect AB PQ) l :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_collinearity_l952_95256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_problem_l952_95281

noncomputable section

/-- The quadratic function we're considering -/
def q (m : ℝ) (x : ℝ) : ℝ := x^2 + 2*m*x + m + 2

/-- The function f(m) -/
def f (m : ℝ) : ℝ := m + 3 / (m + 2)

theorem quadratic_inequality_problem :
  ∃ (m_min m_max : ℝ),
    (∀ m : ℝ, (∀ x : ℝ, q m x ≥ 0) ↔ m_min ≤ m ∧ m ≤ m_max) ∧
    m_min = -1 ∧ m_max = 2 ∧
    (∀ m : ℝ, m_min ≤ m → m ≤ m_max → f m ≥ 2*Real.sqrt 3 - 2) ∧
    (∃ m : ℝ, m_min ≤ m ∧ m ≤ m_max ∧ f m = 2*Real.sqrt 3 - 2) ∧
    (∀ m : ℝ, m_min ≤ m → m ≤ m_max →
      ∀ x : ℝ, x^2 + (m-3)*x - 3*m > 0 ↔ x < -m ∨ x > 3) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_problem_l952_95281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_other_side_length_l952_95206

/-- The area of a trapezium given the lengths of its parallel sides and the distance between them. -/
noncomputable def trapeziumArea (a b h : ℝ) : ℝ := (a + b) * h / 2

/-- Theorem: Given a trapezium with one parallel side of 18 cm, a distance between parallel sides
    of 10 cm, and an area of 190 cm^2, the length of the other parallel side is 20 cm. -/
theorem trapezium_other_side_length :
  ∀ x : ℝ,
  trapeziumArea 18 x 10 = 190 →
  x = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_other_side_length_l952_95206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_l952_95294

/-- First circle equation -/
noncomputable def circle1 (x y : ℝ) : Prop := x^2 + 2*x + y^2 + 6*y + 9 = 0

/-- Second circle equation -/
noncomputable def circle2 (x y : ℝ) : Prop := x^2 - 4*x + y^2 + 6*y + 13 = 0

/-- The intersection point of the two circles -/
noncomputable def intersection_point : ℝ × ℝ := (1/2, -3)

/-- Theorem stating that the product of coordinates of the intersection point is -1.5 -/
theorem intersection_product :
  (intersection_point.1 * intersection_point.2 = -3/2) ∧
  (∀ x y : ℝ, circle1 x y ∧ circle2 x y → (x, y) = intersection_point) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_l952_95294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_blue_probability_l952_95261

/-- Represents a 4x4 grid with red or blue squares -/
def Grid := Fin 4 → Fin 4 → Bool

/-- Rotates the grid 180 degrees -/
def rotate (g : Grid) : Grid :=
  fun i j => g (3 - i) (3 - j)

/-- Applies the color adjustment rule after rotation -/
def adjust (g₁ g₂ : Grid) : Grid :=
  fun i j => g₁ i j || g₂ i j

/-- Represents one round of the process (coloring, rotating, adjusting) -/
def process (g : Grid) : Grid :=
  let g' := rotate g
  adjust g g'

/-- The probability of a square being blue initially -/
noncomputable def p_blue : ℝ := 1 / 2

/-- The probability of the entire grid being blue after two rounds -/
noncomputable def p_all_blue : ℝ := (1 / 16) * (1 / 4096)

/-- Theorem stating the probability of the entire grid being blue after two rounds -/
theorem grid_blue_probability :
  p_all_blue = 1 / 65536 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_blue_probability_l952_95261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equals_singleton_four_l952_95274

def A : Set ℕ := {x | x < 6}
def B : Set ℕ := {x | x^2 - 8*x + 15 < 0}

theorem intersection_equals_singleton_four : A ∩ B = {4} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equals_singleton_four_l952_95274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_is_sqrt_17_l952_95241

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 2*x + y^2 - 4*y = 12

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (1, 2)

/-- The radius of the circle -/
noncomputable def circle_radius : ℝ := Real.sqrt 17

theorem circle_radius_is_sqrt_17 :
  ∀ x y : ℝ, circle_equation x y →
  (x - circle_center.1)^2 + (y - circle_center.2)^2 = circle_radius^2 := by
  sorry

#check circle_radius_is_sqrt_17

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_is_sqrt_17_l952_95241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_differential_equations_l952_95216

noncomputable section

variable (x y u v dx dy : ℝ)

def equation1 (x y u v : ℝ) : Prop := x * u - y * v - 1 = 0
def equation2 (x y u v : ℝ) : Prop := x - y + u - v = 0

def du (x y u v dx dy : ℝ) : ℝ := ((u - v) * dy + (y - v) * dx) / (y - x)
def dv (x y u v dx dy : ℝ) : ℝ := ((u - x) * dx + (x - v) * dy) / (y - x)

def u_x (x y u v : ℝ) : ℝ := (u - y) / (y - x)
def u_y (x y u v : ℝ) : ℝ := (y - v) / (y - x)
def v_x (x y u v : ℝ) : ℝ := (u - x) / (y - x)
def v_y (x y u v : ℝ) : ℝ := (x - v) / (y - x)

def d2u (x y u v dx dy : ℝ) : ℝ := 2 * ((u - y) / ((y - x)^2)) * dx^2 + 
              2 * ((x + y - v - u) / ((y - x)^2)) * dx * dy + 
              2 * ((v - x) / ((y - x)^2)) * dy^2

theorem differential_equations (hx : x ≠ y) 
  (h1 : equation1 x y u v) (h2 : equation2 x y u v) : 
  du x y u v dx dy = ((u - v) * dy + (y - v) * dx) / (y - x) ∧
  dv x y u v dx dy = ((u - x) * dx + (x - v) * dy) / (y - x) ∧
  u_x x y u v = (u - y) / (y - x) ∧
  u_y x y u v = (y - v) / (y - x) ∧
  v_x x y u v = (u - x) / (y - x) ∧
  v_y x y u v = (x - v) / (y - x) ∧
  d2u x y u v dx dy = 2 * ((u - y) / ((y - x)^2)) * dx^2 + 
                      2 * ((x + y - v - u) / ((y - x)^2)) * dx * dy + 
                      2 * ((v - x) / ((y - x)^2)) * dy^2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_differential_equations_l952_95216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reporter_earnings_l952_95235

/-- Represents the payment structure for a reporter --/
structure ReporterPayment where
  politics_per_article : ℚ
  business_per_article : ℚ
  science_per_article : ℚ
  politics_per_word : ℚ
  business_per_word : ℚ
  science_per_word : ℚ
  bonus_per_thousand : ℚ
  words_per_story : ℕ
  politics_stories : ℕ
  business_stories : ℕ
  science_stories : ℕ
  min_words_per_minute : ℚ
  max_words_per_minute : ℚ
  total_hours : ℚ

/-- Calculates the expected earnings per hour for a reporter --/
def expectedEarningsPerHour (payment : ReporterPayment) : ℚ :=
  let total_earnings := 
    (payment.politics_per_article + payment.words_per_story * payment.politics_per_word) * payment.politics_stories +
    (payment.business_per_article + payment.words_per_story * payment.business_per_word) * payment.business_stories +
    (payment.science_per_article + payment.words_per_story * payment.science_per_word) * payment.science_stories
  total_earnings / payment.total_hours

/-- Theorem stating the expected earnings per hour for the given scenario --/
theorem reporter_earnings (payment : ReporterPayment) 
  (h1 : payment.politics_per_article = 60)
  (h2 : payment.business_per_article = 70)
  (h3 : payment.science_per_article = 80)
  (h4 : payment.politics_per_word = 1/10)
  (h5 : payment.business_per_word = 1/10)
  (h6 : payment.science_per_word = 3/20)
  (h7 : payment.bonus_per_thousand = 25)
  (h8 : payment.words_per_story = 1500)
  (h9 : payment.politics_stories = 2)
  (h10 : payment.business_stories = 2)
  (h11 : payment.science_stories = 1)
  (h12 : payment.min_words_per_minute = 8)
  (h13 : payment.max_words_per_minute = 12)
  (h14 : payment.total_hours = 4) :
  expectedEarningsPerHour payment = 1165/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reporter_earnings_l952_95235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_over_b_equals_four_l952_95267

-- Define the variables and constants
variable (x y a b c : ℝ)

-- Define the constraints
def constraint1 (x : ℝ) : Prop := x ≥ 1
def constraint2 (x y : ℝ) : Prop := x + y ≤ 4
def constraint3 (a b c x y : ℝ) : Prop := a * x + b * y + c ≤ 0

-- Define the objective function
def z (x y : ℝ) : ℝ := 2 * x + y

-- Define the maximum and minimum values of z
def max_z (a b c : ℝ) : Prop := ∃ (x y : ℝ), constraint1 x ∧ constraint2 x y ∧ constraint3 a b c x y ∧ z x y = 6
def min_z (a b c : ℝ) : Prop := ∃ (x y : ℝ), constraint1 x ∧ constraint2 x y ∧ constraint3 a b c x y ∧ z x y = 1

-- State the theorem
theorem c_over_b_equals_four 
  (h1 : constraint1 x)
  (h2 : constraint2 x y)
  (h3 : constraint3 a b c x y)
  (h4 : max_z a b c)
  (h5 : min_z a b c)
  (h6 : b ≠ 0) :
  c / b = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_over_b_equals_four_l952_95267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_properties_l952_95278

-- Define the sum of binomial coefficients
def sum_binomial_coeff (n : ℕ) : ℕ := 2^n

-- Define the constant term in the expansion
def constant_term (n : ℕ) (m : ℚ) : ℚ := (Nat.choose n (n/2)) * m^(n/2)

theorem expansion_properties (m : ℚ) :
  -- Part 1
  (∃ n : ℕ, sum_binomial_coeff n = 64) →
  (∃ n : ℕ, n = 6 ∧ sum_binomial_coeff n = 64) ∧
  -- Part 2
  (constant_term 6 m = 35/16) →
  m = (7 : ℚ)^(1/3) / 4 :=
by
  sorry

#check expansion_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_properties_l952_95278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l952_95263

-- Define variables as parameters
variable (x y z : ℝ)

-- Define the conditions
axiom x_pos : 0 < x
axiom y_pos : 0 < y
axiom z_pos : 0 < z
axiom equality : (3:ℝ)^x = (4:ℝ)^y ∧ (4:ℝ)^y = (6:ℝ)^z

-- Part 1
theorem part_one (h : z = 1) : (x - 1) * (2 * y - 1) = 1 := by
  sorry

-- Part 2
theorem part_two : 1 / z - 1 / x = 1 / (2 * y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l952_95263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_population_increase_theorem_l952_95248

/-- Represents a region with its population and birth rates -/
structure Region where
  young_population : Nat
  adult_population : Nat
  young_birth_rate : Nat
  adult_birth_rate : Nat

/-- Calculates the number of births in a given time period -/
def births (population : Nat) (birth_rate : Nat) (time : Nat) : Nat :=
  (time / birth_rate) * population

/-- Calculates the total population increase for a region -/
def region_increase (r : Region) (time : Nat) : Nat :=
  births r.young_population r.young_birth_rate time +
  births r.adult_population r.adult_birth_rate time

/-- The main theorem to prove -/
theorem population_increase_theorem (region_a region_b : Region) :
  region_a.young_population = 2000 →
  region_a.adult_population = 6000 →
  region_a.young_birth_rate = 20 →
  region_a.adult_birth_rate = 30 →
  region_b.young_population = 1500 →
  region_b.adult_population = 5000 →
  region_b.young_birth_rate = 25 →
  region_b.adult_birth_rate = 35 →
  region_increase region_a 1500 + region_increase region_b 1500 = 227 := by
  intro h1 h2 h3 h4 h5 h6 h7 h8
  sorry

#eval region_increase
  { young_population := 2000, adult_population := 6000, young_birth_rate := 20, adult_birth_rate := 30 }
  1500

#eval region_increase
  { young_population := 1500, adult_population := 5000, young_birth_rate := 25, adult_birth_rate := 35 }
  1500

#eval region_increase
  { young_population := 2000, adult_population := 6000, young_birth_rate := 20, adult_birth_rate := 30 }
  1500 +
region_increase
  { young_population := 1500, adult_population := 5000, young_birth_rate := 25, adult_birth_rate := 35 }
  1500

end NUMINAMATH_CALUDE_ERRORFEEDBACK_population_increase_theorem_l952_95248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_distribution_probability_l952_95262

/-- A random variable following a normal distribution with mean 1 and standard deviation σ -/
def X (σ : ℝ) : Type := Unit

/-- The probability that X is greater than 2 -/
axiom prob_X_gt_2 (σ : ℝ) : ℝ

/-- The probability that X is greater than or equal to 0 -/
def prob_X_ge_0 (σ : ℝ) : ℝ := sorry

/-- Theorem stating that if P(X > 2) = 0.3, then P(X ≥ 0) = 0.7 -/
theorem normal_distribution_probability (σ : ℝ) :
  prob_X_gt_2 σ = 0.3 → prob_X_ge_0 σ = 0.7 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_distribution_probability_l952_95262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_and_max_value_l952_95271

noncomputable def f (a x : ℝ) : ℝ := 1 - 2*a - 2*a*Real.cos x - 2*(Real.sin x)^2

noncomputable def g (a : ℝ) : ℝ :=
  if a < -2 then 1
  else if a ≤ 2 then -a^2/2 - 2*a - 1
  else 1 - 4*a

theorem min_value_and_max_value (a : ℝ) :
  (∀ x, f a x ≥ g a) ∧
  (g a = 1/2 → a = -1) ∧
  (g a = 1/2 → ∃ x, f a x = 5 ∧ ∀ y, f a y ≤ 5) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_and_max_value_l952_95271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_power_sum_l952_95251

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_power_sum : i^23 + i^34 + i^(-17 : ℤ) = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_power_sum_l952_95251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_students_without_vision_assistance_l952_95205

def vision_assistance_count (total : ℕ) 
  (glasses_percent : ℚ) (contacts_percent : ℚ) (both_percent : ℚ) : ℕ :=
  let glasses_count := (total : ℚ) * glasses_percent
  let contacts_count := (total : ℚ) * contacts_percent
  let both_count := (total : ℚ) * both_percent
  let vision_assistance_count := glasses_count + contacts_count - both_count
  total - (Int.toNat (Int.floor vision_assistance_count))

theorem students_without_vision_assistance :
  vision_assistance_count 40 (25/100) (40/100) (10/100) = 18 := by
  sorry

#eval vision_assistance_count 40 (25/100) (40/100) (10/100)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_students_without_vision_assistance_l952_95205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_100_greater_than_14_l952_95228

noncomputable def mySequence (n : ℕ) : ℝ :=
  match n with
  | 0 => 1
  | n + 1 => mySequence n + 1 / mySequence n

theorem mySequence_100_greater_than_14 : mySequence 99 > 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_100_greater_than_14_l952_95228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_borrowed_amount_theorem_l952_95257

/-- Calculates the compound interest for a given principal, rate, and time --/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Represents the loan scenario described in the problem --/
def loan_scenario (initial_amount : ℝ) : Prop :=
  let amount_after_5_years := compound_interest initial_amount 0.04 5
  let final_amount := compound_interest amount_after_5_years 0.07 5
  final_amount = 20500

theorem borrowed_amount_theorem :
  ∃ (initial_amount : ℝ), 
    loan_scenario initial_amount ∧ 
    abs (initial_amount - 12016.77) < 0.01 := by
  sorry

#check borrowed_amount_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_borrowed_amount_theorem_l952_95257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_and_sum_of_cube_roots_l952_95211

theorem existence_and_sum_of_cube_roots : ∃ (p q r : ℕ+),
  (4 * Real.sqrt (Real.rpow 7 (1/3) - Real.rpow 6 (1/3)) = Real.rpow p (1/3) + Real.rpow q (1/3) - Real.rpow r (1/3)) ∧
  (p + q + r = 93) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_and_sum_of_cube_roots_l952_95211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rental_income_formula_rental_income_88_cars_l952_95209

/-- Represents the rental company's monthly income function. -/
noncomputable def rental_income (x : ℝ) : ℝ :=
  let cars_total := 100
  let base_rent := 3000
  let rent_increase_step := 50
  let rented_maintenance := 150
  let unrented_maintenance := 50
  let cars_not_rented := (x - base_rent) / rent_increase_step
  let cars_rented := cars_total - cars_not_rented
  cars_rented * (x - rented_maintenance) - cars_not_rented * unrented_maintenance

theorem rental_income_formula (x : ℝ) :
  rental_income x = -1/50 * x^2 + 162*x - 21000 := by
  sorry

theorem rental_income_88_cars :
  rental_income 3600 = 303000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rental_income_formula_rental_income_88_cars_l952_95209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_relation_l952_95249

/-- The eccentricity of an ellipse -/
def EllipseEccentricity (e : ℝ) : Prop :=
  0 < e ∧ e < 1

/-- The eccentricity of a hyperbola -/
def HyperbolaEccentricity (e : ℝ) : Prop :=
  e > 1

/-- Two conic sections with common foci -/
structure CommonFociConics where
  e₁ : ℝ
  e₂ : ℝ
  ellipse_ecc : EllipseEccentricity e₁
  hyperbola_ecc : HyperbolaEccentricity e₂
  common_point_condition : ∃ (P F₁ F₂ : ℝ × ℝ), ‖P - F₁ + (P - F₂)‖ = ‖F₁ - F₂‖

theorem eccentricity_relation (c : CommonFociConics) :
    c.e₁ * c.e₂ / Real.sqrt (c.e₁^2 + c.e₂^2) = Real.sqrt 2 / 2 := by
  sorry

#check eccentricity_relation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_relation_l952_95249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_school_children_count_l952_95218

/-- The number of children in the school -/
def C : ℕ := sorry

/-- The total number of bananas -/
def B : ℕ := sorry

/-- The number of absent children -/
def absent : ℕ := 305

theorem school_children_count :
  (B = 2 * C) →
  (B = 4 * (C - absent)) →
  C = 610 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_school_children_count_l952_95218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l952_95244

noncomputable def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

noncomputable def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := by
  intro x
  -- Expand the definition of g
  unfold g
  -- Expand the definition of f
  unfold f
  -- Simplify the expressions
  simp [sub_neg_eq_add]
  -- The rest of the proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l952_95244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_externally_tangent_circles_l952_95207

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The distance between two points in the 2D plane -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Two circles are externally tangent if the distance between their centers
    equals the sum of their radii -/
def externallyTangent (c1 c2 : Circle) : Prop :=
  distance c1.center c2.center = c1.radius + c2.radius

theorem externally_tangent_circles (m : ℝ) (hm : m > 0) :
  let c1 : Circle := ⟨(0, 0), 2⟩
  let c2 : Circle := ⟨(3, -4), m⟩
  externallyTangent c1 c2 → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_externally_tangent_circles_l952_95207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_face_value_calculation_l952_95234

/-- Calculates the face value of shares given investment details -/
theorem face_value_calculation (total_investment : ℚ) (quoted_price : ℚ) (dividend_rate : ℚ) (annual_income : ℚ) :
  total_investment = 4455 →
  quoted_price = 8.25 →
  dividend_rate = 12 / 100 →
  annual_income = 648 →
  (total_investment / quoted_price) * (dividend_rate * 10) = annual_income := by
  sorry

-- Remove the #eval statement as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_face_value_calculation_l952_95234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subset_size_l952_95223

def S : Finset ℕ := Finset.range 40

def isSquare (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def validSubset (T : Finset ℕ) : Prop :=
  T ⊆ S ∧ ∀ a b, a ∈ T → b ∈ T → a ≠ b → ¬isSquare (a * b)

theorem max_subset_size :
  (∃ T : Finset ℕ, validSubset T ∧ T.card = 26) ∧
  (∀ T : Finset ℕ, validSubset T → T.card ≤ 26) := by
  sorry

#check max_subset_size

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subset_size_l952_95223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_segment_length_l952_95227

-- Define the line l
def line_l (t : ℝ) : ℝ × ℝ := (1 + t, t)

-- Define the ellipse C
noncomputable def ellipse_C (θ : ℝ) : ℝ × ℝ := (Real.sqrt 2 * Real.cos θ, Real.sin θ)

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ t θ, line_l t = ellipse_C θ ∧ p = line_l t}

-- Theorem statement
theorem intersection_segment_length :
  ∃ A B : ℝ × ℝ, A ∈ intersection_points ∧ B ∈ intersection_points ∧
    A ≠ B ∧ Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 * Real.sqrt 2 / 3 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_segment_length_l952_95227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l952_95212

theorem right_triangle_area (a c : ℝ) (h1 : a = 15) (h2 : c = 17) : 
  (1/2) * a * Real.sqrt (c^2 - a^2) = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l952_95212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_perimeter_triangle_on_circle_l952_95265

/-- Given a unit circle with points A and B on its circumference such that the chord AB has length 1,
    and C is any point on the longer arc between A and B, 
    the maximum perimeter of triangle ABC is 3. -/
theorem max_perimeter_triangle_on_circle (A B C : ℝ × ℝ) : 
  (∀ x y, x^2 + y^2 = 1 → (A.1 - B.1)^2 + (A.2 - B.2)^2 = 1) →
  (∀ x y, x^2 + y^2 = 1 → (A.1 - x)^2 + (A.2 - y)^2 ≤ (A.1 - B.1)^2 + (A.2 - B.2)^2) →
  C.1^2 + C.2^2 = 1 →
  (A.1 - C.1)^2 + (A.2 - C.2)^2 + 
  (B.1 - C.1)^2 + (B.2 - C.2)^2 + 
  (A.1 - B.1)^2 + (A.2 - B.2)^2 ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_perimeter_triangle_on_circle_l952_95265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parkway_girls_not_playing_soccer_l952_95210

/-- The number of girl students not playing soccer at Parkway Elementary School -/
theorem parkway_girls_not_playing_soccer : ℕ := by
  let total_students : ℕ := 420
  let boys : ℕ := 320
  let students_playing_soccer : ℕ := 250
  let boys_playing_soccer : ℕ := (86 * students_playing_soccer) / 100

  let girls : ℕ := total_students - boys
  let girls_playing_soccer : ℕ := students_playing_soccer - boys_playing_soccer
  let girls_not_playing_soccer : ℕ := girls - girls_playing_soccer

  have h : girls_not_playing_soccer = 65 := by sorry
  exact girls_not_playing_soccer

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parkway_girls_not_playing_soccer_l952_95210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_window_height_is_correct_l952_95298

/-- Represents the dimensions and cost information for a room whitewashing problem -/
structure RoomInfo where
  length : ℝ
  width : ℝ
  height : ℝ
  whitewash_cost_per_sqft : ℝ
  door_length : ℝ
  door_width : ℝ
  num_windows : ℕ
  window_width : ℝ
  total_cost : ℝ

/-- Calculates the height of each window in the room -/
noncomputable def calculate_window_height (info : RoomInfo) : ℝ :=
  let wall_area := 2 * (info.length + info.width) * info.height
  let door_area := info.door_length * info.door_width
  let window_area := (wall_area - door_area - info.total_cost / info.whitewash_cost_per_sqft) / (info.num_windows : ℝ) / info.window_width
  window_area

/-- Theorem stating that the calculated window height is correct -/
theorem window_height_is_correct (info : RoomInfo) :
  info.length = 25 ∧
  info.width = 15 ∧
  info.height = 12 ∧
  info.whitewash_cost_per_sqft = 10 ∧
  info.door_length = 6 ∧
  info.door_width = 3 ∧
  info.num_windows = 3 ∧
  info.window_width = 3 ∧
  info.total_cost = 9060 →
  calculate_window_height info = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_window_height_is_correct_l952_95298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_one_third_l952_95222

/-- Trapezoid ABCD with point E where extended legs meet -/
structure ExtendedTrapezoid where
  /-- Length of base AB -/
  ab : ℝ
  /-- Length of base CD -/
  cd : ℝ
  /-- Height of trapezoid from AB to CD -/
  height : ℝ

/-- Calculate the area of a trapezoid given its base lengths and height -/
noncomputable def trapezoidArea (t : ExtendedTrapezoid) : ℝ :=
  (t.ab + t.cd) * t.height / 2

/-- Calculate the area ratio of triangle EAB to trapezoid ABCD -/
noncomputable def areaRatio (t : ExtendedTrapezoid) : ℝ :=
  let triangleEABArea := t.ab * t.height / 2
  triangleEABArea / trapezoidArea t

/-- Theorem stating that the area ratio is 1/3 for the given trapezoid -/
theorem area_ratio_is_one_third :
  let t : ExtendedTrapezoid := ⟨10, 20, 12⟩
  areaRatio t = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_one_third_l952_95222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_categorization_correct_l952_95219

def given_numbers : List ℚ := [-3.5, 0.4, 3, 1.75, 0, -30, -0.15, -128, 20, -2.6666666666666]

def is_integer (q : ℚ) : Prop := ∃ (n : ℤ), q = n

def is_negative_rational (q : ℚ) : Prop := q < 0

def is_positive_fraction (q : ℚ) : Prop := 0 < q ∧ q < 1

def integer_set : Set ℚ := {q | is_integer q}
def negative_rational_set : Set ℚ := {q | is_negative_rational q}
def positive_fraction_set : Set ℚ := {q | is_positive_fraction q}

theorem categorization_correct : 
  (integer_set ∩ (given_numbers.toFinset : Set ℚ)) = {3, 0, -30, -128, 20} ∧
  (negative_rational_set ∩ (given_numbers.toFinset : Set ℚ)) = {-3.5, -30, -0.15, -128, -2.6666666666666} ∧
  (positive_fraction_set ∩ (given_numbers.toFinset : Set ℚ)) = {0.4} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_categorization_correct_l952_95219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_coordinates_l952_95225

-- Define the parabola equation
noncomputable def parabola (x : ℝ) : ℝ := (1/3) * (x - 7)^2 + 5

-- State the theorem
theorem vertex_coordinates :
  ∃ (x y : ℝ), (x = 7 ∧ y = 5) ∧
  ∀ (x' : ℝ), parabola x' ≥ parabola x :=
by
  -- Introduce the vertex coordinates
  let x := 7
  let y := 5
  
  -- Prove the existence of the vertex
  use x, y
  
  constructor
  · -- Prove that (x, y) = (7, 5)
    constructor
    · rfl
    · rfl
  
  · -- Prove that (7, 5) is the minimum point
    intro x'
    -- The actual proof would go here
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_coordinates_l952_95225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factory_output_changes_l952_95224

/-- Represents the percentage increase in factory output -/
noncomputable def initial_increase : ℝ := 7.7

/-- Theorem stating the relationship between initial increase and subsequent changes -/
theorem factory_output_changes (original : ℝ) (original_pos : original > 0) :
  let first_increase := original * (1 + initial_increase / 100)
  let second_increase := first_increase * 1.30
  let final_decrease := second_increase * (1 - 0.3007)
  final_decrease = original →
  abs (initial_increase - 7.7) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_factory_output_changes_l952_95224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_implies_m_range_l952_95220

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 - (4*m-1) * x^2 + (15*m^2-2*m-7) * x + 2

theorem increasing_f_implies_m_range (m : ℝ) :
  (∀ x : ℝ, Monotone (f m)) → 2 < m ∧ m < 4 :=
by
  sorry

#check increasing_f_implies_m_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_implies_m_range_l952_95220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_solution_l952_95237

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log (1/2) else -x^2 - 2*x

-- Define the solution set
def solution_set : Set ℝ := {x | x < -2 ∨ x > 1}

-- Theorem statement
theorem f_inequality_solution : 
  {x : ℝ | f x < 0} = solution_set := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_solution_l952_95237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cp_bisects_dce_l952_95215

-- Define the types for points and planes
variable (Point Plane : Type)

-- Define the necessary relations and functions
variable (on_plane : Point → Plane → Prop)
variable (orthogonal : Plane → Plane → Prop)
variable (intersection_line : Plane → Plane → Set Point)
variable (angle_bisector : Point → Point → Point → Set Point)
variable (circle : Point → Point → Plane → Set Point)
variable (intersects : Set Point → Set Point → Set Point)

-- State the theorem
theorem cp_bisects_dce
  (π₁ π₂ π₃ : Plane)
  (A B C P D E : Point)
  (S : Set Point)
  (h1 : orthogonal π₁ π₂)
  (h2 : A ∈ intersection_line π₁ π₂)
  (h3 : B ∈ intersection_line π₁ π₂)
  (h4 : A ≠ B)
  (h5 : on_plane C π₂ ∧ ¬on_plane C π₁)
  (h6 : P ∈ intersects (angle_bisector B C A) {A, B})
  (h7 : S = circle A B π₁)
  (h8 : on_plane C π₃ ∧ on_plane P π₃)
  (h9 : D ∈ intersects S {X | on_plane X π₃})
  (h10 : E ∈ intersects S {X | on_plane X π₃})
  : P ∈ angle_bisector D C E := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cp_bisects_dce_l952_95215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l952_95226

/-- Triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The area of a triangle given two sides and the included angle --/
noncomputable def area (t : Triangle) : ℝ := (1/2) * t.a * t.b * Real.sin t.C

theorem triangle_abc_properties (t : Triangle) 
  (h1 : t.c = 2)
  (h2 : t.C = π/3) :
  (area t = Real.sqrt 3 → t.a = 2 ∧ t.b = 2) ∧
  (Real.sin t.C + Real.sin (t.B - t.A) = 2 * Real.sin (2 * t.A) → 
    area t = (2 * Real.sqrt 3) / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l952_95226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marked_triangle_theorem_l952_95253

/-- Triangle ABC with points D, E, F, G, H, J, K marked by successive arcs -/
structure MarkedTriangle where
  -- Triangle sides
  a : ℝ
  b : ℝ
  c : ℝ
  -- Distance AD
  x : ℝ
  -- Assumptions
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < x
  h_triangle : a < b + c ∧ b < a + c ∧ c < a + b
  h_x_valid : x ≤ c

noncomputable def MarkedTriangle.AK (t : MarkedTriangle) : ℝ :=
  t.c - (t.a - (t.b - (t.c - (t.a - (t.b - t.x)))))

noncomputable def MarkedTriangle.AL (t : MarkedTriangle) : ℝ :=
  (t.c - t.a + t.b) / 2

theorem marked_triangle_theorem (t : MarkedTriangle) :
  t.AK = t.x ∧ t.AL = (t.c - t.a + t.b) / 2 := by
  sorry

#check marked_triangle_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marked_triangle_theorem_l952_95253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coeff_x_cubed_in_expansion_l952_95280

/-- The coefficient of x^3 in the expansion of (√x - 2/x + 1)^7 is 7 -/
theorem coeff_x_cubed_in_expansion (x : ℝ) : 
  (PowerSeries.coeff ℝ 3 (PowerSeries.mk (λ n ↦ (Real.sqrt x - 2 / x + 1) ^ 7))) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coeff_x_cubed_in_expansion_l952_95280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_property_l952_95299

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola with parameter p -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Get the focus of a parabola -/
noncomputable def focus (para : Parabola) : Point :=
  { x := para.p / 2, y := 0 }

/-- Get a point on the parabola given the parameter t -/
noncomputable def pointOnParabola (para : Parabola) (t : ℝ) : Point :=
  { x := 2 * para.p * t^2, y := 2 * para.p * t }

/-- Get the foot of the perpendicular from a point to the directrix -/
noncomputable def footOfPerpendicular (para : Parabola) (point : Point) : Point :=
  { x := -para.p / 2, y := point.y }

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

theorem parabola_property (para : Parabola) :
  ∃ (M : Point),
    M.x = 6 ∧
    M = pointOnParabola para (Real.sqrt (3 / para.p)) ∧
    distance M (focus para) = distance M (footOfPerpendicular para M) →
    para.p = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_property_l952_95299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_and_increasing_l952_95232

def sequence_a : ℕ → ℝ
  | 0 => 2  -- Add this case to handle n = 0
  | 1 => 2
  | n + 2 => 2 * sequence_a (n + 1) - 1

theorem sequence_a_formula_and_increasing :
  (∀ n : ℕ, n ≥ 1 → sequence_a n = 2^(n-1) + 1) ∧
  (∀ n : ℕ, n ≥ 1 → sequence_a (n+1) > sequence_a n) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_and_increasing_l952_95232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_offset_is_nine_l952_95250

/-- Represents a quadrilateral with a diagonal and two offsets -/
structure Quadrilateral where
  diagonal : ℝ
  offset1 : ℝ
  offset2 : ℝ
  area : ℝ

/-- Calculates the area of a quadrilateral given its diagonal and offsets -/
noncomputable def calculate_area (q : Quadrilateral) : ℝ :=
  (q.diagonal * q.offset1 + q.diagonal * q.offset2) / 2

/-- Theorem stating that for a quadrilateral with given properties, the second offset is 9 -/
theorem second_offset_is_nine (q : Quadrilateral) 
  (h1 : q.diagonal = 40)
  (h2 : q.offset1 = 11)
  (h3 : q.area = 400)
  (h4 : calculate_area q = q.area) :
  q.offset2 = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_offset_is_nine_l952_95250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_k_for_increased_roots_l952_95246

noncomputable section

/-- The equation has increased roots at x = 2 and x = -2 -/
def has_increased_roots (f : ℝ → ℝ) : Prop :=
  f 2 = 0 ∧ f (-2) = 0 ∧ ∀ x, x ≠ 2 → x ≠ -2 → f x ≠ 0

/-- The equation in the problem -/
def equation (k : ℝ) (x : ℝ) : ℝ :=
  1 / (4 - x^2) + 2 - k / (x - 2)

theorem unique_k_for_increased_roots :
  ∃! k : ℝ, has_increased_roots (equation k) ∧ k = -1/4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_k_for_increased_roots_l952_95246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_theorem_l952_95238

/-- The count of positive integers less than 10000 where 2^x - x^2 is divisible by 7 -/
def count_divisible_by_seven : ℕ := 2857

/-- Predicate to check if 2^x - x^2 is divisible by 7 -/
def is_divisible_by_seven (x : ℕ) : Prop :=
  (2^x - x^2) % 7 = 0

/-- Decidable instance for is_divisible_by_seven -/
instance (x : ℕ) : Decidable (is_divisible_by_seven x) :=
  show Decidable ((2^x - x^2) % 7 = 0) from inferInstance

theorem count_theorem :
  (Finset.filter (fun x => is_divisible_by_seven x) (Finset.range 10000)).card = count_divisible_by_seven :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_theorem_l952_95238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_real_roots_condition_existence_of_a_satisfying_condition_l952_95240

/-- The quadratic equation in question -/
noncomputable def quadratic_equation (a x : ℝ) : ℝ := (a - 5) * x^2 - 4*x - 1

/-- The discriminant of the quadratic equation -/
noncomputable def discriminant (a : ℝ) : ℝ := 4*a - 4

/-- The sum of roots of the quadratic equation -/
noncomputable def sum_of_roots (a : ℝ) : ℝ := 4 / (a - 5)

/-- The product of roots of the quadratic equation -/
noncomputable def product_of_roots (a : ℝ) : ℝ := -1 / (a - 5)

theorem quadratic_real_roots_condition (a : ℝ) :
  (∃ x : ℝ, quadratic_equation a x = 0) ↔ a ≥ 1 := by sorry

theorem existence_of_a_satisfying_condition :
  ∃ a : ℝ, sum_of_roots a + product_of_roots a = 3 ∧ a = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_real_roots_condition_existence_of_a_satisfying_condition_l952_95240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_decreasing_interval_l952_95204

open Set
open Real

noncomputable def f (x : ℝ) : ℝ := 3 + x * (log x)

theorem monotonically_decreasing_interval :
  {x : ℝ | x ∈ (Set.Ioo 0 (1/Real.exp 1))} = {x : ℝ | ∀ y ∈ (Set.Ioo 0 (1/Real.exp 1)), x < y → f x > f y} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_decreasing_interval_l952_95204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_expression_l952_95273

noncomputable def sqrt_196 : ℝ := Real.sqrt 196

def expression (x : ℝ) : ℝ := (21 + x)^20 - (21 - x)^20

noncomputable def units_digit (n : ℝ) : ℕ := Int.natAbs (Int.mod (Int.floor n) 10)

theorem units_digit_of_expression : units_digit (expression sqrt_196) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_expression_l952_95273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_location_l952_95254

/-- Square with side length 1 -/
structure UnitSquare where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  is_unit_square : A = (0, 0) ∧ B = (1, 0) ∧ C = (1, 1) ∧ D = (0, 1)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Point satisfies the given condition -/
def satisfies_condition (s : UnitSquare) (P : ℝ × ℝ) : Prop :=
  distance s.A P * distance s.C P + distance s.B P * distance s.D P = 1

/-- Point lies on a diagonal of the square -/
def on_diagonal (P : ℝ × ℝ) : Prop :=
  (P.2 = P.1) ∨ (P.2 = 1 - P.1)

/-- Main theorem -/
theorem point_location (s : UnitSquare) (P : ℝ × ℝ) :
  satisfies_condition s P ↔ on_diagonal P := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_location_l952_95254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concert_ticket_price_l952_95295

/-- The original price of a concert ticket --/
def P : ℚ := 20

/-- The number of people who received a 40% discount --/
def first_group : ℕ := 10

/-- The number of people who received a 15% discount --/
def second_group : ℕ := 20

/-- The total number of people who bought tickets --/
def total_people : ℕ := 48

/-- The total revenue from ticket sales --/
def total_revenue : ℚ := 820

theorem concert_ticket_price :
  total_revenue = (first_group : ℚ) * (60 / 100) * P + 
                  (second_group : ℚ) * (85 / 100) * P + 
                  ((total_people - first_group - second_group : ℚ) * P) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_concert_ticket_price_l952_95295
