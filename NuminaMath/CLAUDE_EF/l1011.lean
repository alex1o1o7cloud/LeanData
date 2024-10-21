import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_slope_at_one_l1011_101110

-- Define the function f(x) = x + 2/x
noncomputable def f (x : ℝ) : ℝ := x + 2 / x

-- State the theorem
theorem min_value_of_f :
  (∀ x : ℝ, x > 0 → f x ≥ 2 * Real.sqrt 2) ∧
  (∃ x : ℝ, x > 0 ∧ f x = 2 * Real.sqrt 2) := by
  sorry

-- Define the derivative of f
noncomputable def f_derivative (x : ℝ) : ℝ := 1 - 2 / (x^2)

-- State that the slope at x = 1 is -1
theorem slope_at_one : f_derivative 1 = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_slope_at_one_l1011_101110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_pairs_property_l1011_101123

def sequenceA : ℕ → ℕ
  | 0 => 4
  | 1 => 11
  | (n + 2) => 3 * sequenceA (n + 1) - sequenceA n

theorem infinite_pairs_property :
  ∃ (seq : ℕ → ℕ), seq 0 = 4 ∧ seq 1 = 11 ∧
  (∀ n, seq (n + 2) = 3 * seq (n + 1) - seq n) ∧
  (∀ n, n ≥ 2 →
    (Nat.gcd (seq n) (seq (n + 1)) = 1) ∧
    (seq n ∣ seq (n + 1)^2 - 5) ∧
    (seq (n + 1) ∣ seq n^2 - 5) ∧
    (seq n < seq (n + 1))) :=
by
  use sequenceA
  constructor
  · rfl
  constructor
  · rfl
  constructor
  · intro n
    rfl
  · intro n hn
    sorry  -- Proof details omitted for brevity


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_pairs_property_l1011_101123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_true_l1011_101138

structure GeometricSpace where
  Line : Type
  Plane : Type
  contains : Plane → Line → Prop
  parallel : Plane → Plane → Prop
  parallel_line_plane : Line → Plane → Prop
  perpendicular : Plane → Plane → Prop
  perpendicular_line : Line → Line → Prop
  intersects : Line → Line → Prop
  line_in_plane : Plane → Line → Prop

variable (S : GeometricSpace)

def proposition1 : Prop :=
  ∀ (l : S.Line) (α : S.Plane),
    ∃ (m : S.Line), S.line_in_plane α m ∧ S.perpendicular_line l m

def proposition2 : Prop :=
  ∀ (a : S.Line) (β : S.Plane),
    S.parallel_line_plane a β →
    ∃ (l : S.Line), S.line_in_plane β l ∧ S.intersects a l

def proposition3 : Prop :=
  ∀ (α β : S.Plane) (a b : S.Line),
    S.parallel α β → S.contains α a → S.contains β b →
    ∃ (l : S.Line), S.perpendicular_line l a ∧ S.perpendicular_line l b

def proposition4 : Prop :=
  ∀ (α β : S.Plane) (a b c : S.Line),
    S.perpendicular α β → S.contains α a → S.contains β b →
    S.contains α c → S.contains β c →
    ¬S.perpendicular_line a c → ¬S.perpendicular_line a b

theorem exactly_two_true : 
  (proposition1 S ∧ proposition3 S) ∧ 
  (¬proposition2 S ∧ ¬proposition4 S) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_true_l1011_101138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_expression_no_quadratic_cubic_terms_l1011_101131

-- Problem 1
theorem constant_expression (x : ℝ) : (2*x+3)*(3*x+2)-6*x*(x+3)+5*x+16 = 22 := by sorry

-- Problem 2
theorem no_quadratic_cubic_terms (m n : ℝ) : 
  (∀ x, (x^2 + n*x + 3) * (x^2 - 3*x + m) = x^4 + (n-3)*x^3 + (m+3-3*n)*x^2 + (3*m-9)*x + 3*m) →
  m = 6 ∧ n = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_expression_no_quadratic_cubic_terms_l1011_101131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_properties_l1011_101188

-- Define the quadratic equation
def quadratic_eq (k x : ℝ) : Prop := 4 * k * x^2 - 4 * k * x + k + 1 = 0

-- Theorem statement
theorem quadratic_roots_properties :
  ∀ (k x₁ x₂ : ℝ), 
    k < 0 →
    quadratic_eq k x₁ →
    quadratic_eq k x₂ →
    (∀ k, k < 0) ∧
    (x₁^2 + x₂^2 = 4 → k = -1/7) ∧
    (∃ n : ℤ, (x₁ / x₂ + x₂ / x₁ - 2 : ℝ) = ↑n ↔ k ∈ ({-2, -3, -5} : Set ℝ)) :=
by
  intros k x₁ x₂ hk hx₁ hx₂
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_properties_l1011_101188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_math_department_absence_percentage_l1011_101108

theorem math_department_absence_percentage :
  let total_students : ℕ := 160
  let male_students : ℕ := 90
  let female_students : ℕ := 70
  let absent_male_ratio : ℚ := 1 / 5
  let absent_female_ratio : ℚ := 2 / 7
  let absent_students : ℚ := absent_male_ratio * male_students + absent_female_ratio * female_students
  let absence_percentage : ℚ := absent_students / total_students * 100
  absence_percentage = 23.75 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_math_department_absence_percentage_l1011_101108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_other_number_proof_l1011_101143

theorem other_number_proof (A B : ℕ) 
  (h1 : Nat.lcm A B = 2310)
  (h2 : Nat.gcd A B = 30)
  (h3 : A = 210) :
  B = 330 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_other_number_proof_l1011_101143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_larger_hexagon_l1011_101121

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Represents a hexagon -/
structure Hexagon where
  vertices : Fin 6 → Point

/-- The side length of the equilateral triangle -/
def triangle_side_length : ℝ := 2

/-- The area of a regular hexagon with side length s -/
noncomputable def hexagon_area (s : ℝ) : ℝ := 3 * Real.sqrt 3 / 2 * s^2

/-- The equilateral triangle ABC -/
noncomputable def triangle_ABC : Triangle := sorry

/-- The regular hexagon ABDE -/
noncomputable def hexagon_ABDE : Hexagon := sorry

/-- The regular hexagon BCHI -/
noncomputable def hexagon_BCHI : Hexagon := sorry

/-- The regular hexagon CAFG -/
noncomputable def hexagon_CAFG : Hexagon := sorry

/-- The larger hexagon DEFGHI -/
noncomputable def hexagon_DEFGHI : Hexagon := sorry

/-- Theorem: The area of the larger hexagon DEFGHI is 36√3 -/
theorem area_of_larger_hexagon :
  hexagon_area triangle_side_length * 3 = 36 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_larger_hexagon_l1011_101121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_sum_condition_l1011_101125

theorem sine_sum_condition (α β : ℝ) :
  (∃ k : ℤ, α + β = 2 * k * Real.pi + Real.pi / 6) →
  (Real.sin α * Real.cos β + Real.cos α * Real.sin β = 1 / 2) ∧
  ¬(∀ α β : ℝ, Real.sin α * Real.cos β + Real.cos α * Real.sin β = 1 / 2 → ∃ k : ℤ, α + β = 2 * k * Real.pi + Real.pi / 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_sum_condition_l1011_101125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_molecular_weight_from_moles_l1011_101154

/-- The molecular weight of a compound in g/mol -/
def molecular_weight (compound : Type) : ℝ := sorry

/-- The weight of a given number of moles of a compound in grams -/
def weight (compound : Type) (moles : ℝ) : ℝ := sorry

/-- A specific compound -/
def C : Type := sorry

theorem molecular_weight_from_moles :
  weight C 5 = 490 → molecular_weight C = 490 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_molecular_weight_from_moles_l1011_101154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_is_seven_l1011_101152

noncomputable def center : ℝ × ℝ := (-2, -3)
noncomputable def inside_point : ℝ × ℝ := (-2, 3)
noncomputable def outside_point : ℝ × ℝ := (6, -3)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem circle_radius_is_seven :
  ∃ (r : ℕ), r = 7 ∧
  distance center inside_point < r ∧
  r < distance center outside_point :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_is_seven_l1011_101152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_my_sequence_non_periodic_l1011_101112

def my_sequence (x : ℕ → ℝ) : Prop :=
  x 1 = 1 ∧ ∀ n : ℕ, n ≥ 1 → x (n + 1) = n * Real.sin (x n) + 1

theorem my_sequence_non_periodic (x : ℕ → ℝ) (h : my_sequence x) : 
  ¬ ∃ (T : ℕ) (m₀ : ℕ), T > 0 ∧ ∀ m ≥ m₀, x (m + T) = x m :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_my_sequence_non_periodic_l1011_101112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l1011_101124

/-- The equation (x-2)(x^2-4x+m)=0 has three roots that can serve as triangle sides -/
def has_triangle_roots (m : ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ : ℝ),
    (x₁ - 2) * (x₁^2 - 4*x₁ + m) = 0 ∧
    (x₂ - 2) * (x₂^2 - 4*x₂ + m) = 0 ∧
    (x₃ - 2) * (x₃^2 - 4*x₃ + m) = 0 ∧
    x₁ + x₂ > x₃ ∧ x₁ + x₃ > x₂ ∧ x₂ + x₃ > x₁ ∧
    x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0

/-- The range of m for which the equation has three roots forming a triangle -/
theorem m_range : ∀ m : ℝ, has_triangle_roots m ↔ 3 < m ∧ m ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l1011_101124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_on_parabola_directrix_l1011_101134

noncomputable def θ : ℝ := sorry

/-- The point A on the terminal side of angle θ -/
noncomputable def A : ℝ × ℝ := (-Real.sqrt 3, 1)

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = -1/4 * x^2

/-- The directrix equation -/
def directrix (y : ℝ) : Prop := y = 1

/-- Theorem statement -/
theorem cos_double_angle_on_parabola_directrix :
  A.1 = -Real.sqrt 3 ∧ 
  A.2 = 1 ∧
  parabola A.1 A.2 ∧
  directrix A.2 →
  Real.cos (2 * θ) = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_on_parabola_directrix_l1011_101134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marathon_total_time_l1011_101177

-- Define the speeds and times for each runner
noncomputable def dean_time : ℝ := 9

-- Micah runs 2/3 times as fast as Dean, so his time is 2/3 of Dean's
noncomputable def micah_time : ℝ := (2/3) * dean_time

-- Jake takes 1/3 times more time than Micah
noncomputable def jake_time : ℝ := micah_time + (1/3) * micah_time

-- Total time for all three runners
noncomputable def total_time : ℝ := dean_time + micah_time + jake_time

-- Theorem statement
theorem marathon_total_time : total_time = 23 := by
  -- Expand the definitions
  unfold total_time jake_time micah_time dean_time
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marathon_total_time_l1011_101177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_track_length_l1011_101130

/-- The length of a circular track given two runners' speeds and meeting time -/
theorem track_length (v1 v2 : ℝ) (t : ℝ) (h1 : v1 = 20) (h2 : v2 = 40) (h3 : t = 14.998800095992321) :
  ∃ L : ℝ, (abs (L - 249.98) < 0.01) ∧ L = (v1 + v2) * t / 3600 := by
  sorry

#check track_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_track_length_l1011_101130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1011_101109

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (3 - x) + 1 / (x - 4)

-- State the theorem
theorem range_of_f :
  ∀ x : ℝ, f x ∈ Set.range f ↔ x ≤ 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1011_101109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_symmetry_l1011_101184

-- Define the given functions as noncomputable
noncomputable def f (x φ : ℝ) : ℝ := Real.sin (2 * x + φ)
noncomputable def g (x φ : ℝ) : ℝ := Real.cos (2 * x + φ)

-- State the theorem
theorem cos_symmetry (φ : ℝ) :
  (∀ x : ℝ, f x φ ≤ f (π/6) φ) →
  ∀ x : ℝ, g (π/6 + x) φ = g (π/6 - x) φ :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_symmetry_l1011_101184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_of_collinear_vectors_l1011_101174

-- Define the vectors a and b
def a (l : ℝ) : Fin 2 → ℝ := ![2, l]
def b : Fin 2 → ℝ := ![-3, 1]

-- Define collinearity
def collinear (u v : Fin 2 → ℝ) : Prop :=
  ∃ k : ℝ, ∀ i : Fin 2, u i = k * v i

-- Define dot product
def dot_product (u v : Fin 2 → ℝ) : ℝ :=
  (u 0) * (v 0) + (u 1) * (v 1)

-- Theorem statement
theorem dot_product_of_collinear_vectors :
  ∀ l : ℝ, collinear (a l) b → dot_product (a l) b = -20/3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_of_collinear_vectors_l1011_101174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_estimate_majority_color_l1011_101115

/-- Represents the color of a ball -/
inductive BallColor
  | White
  | Black

/-- Represents a bag of balls -/
structure Bag where
  white_balls : ℕ
  black_balls : ℕ
  large_difference : white_balls ≠ black_balls

/-- Represents the result of drawing balls from the bag -/
structure DrawResult where
  total_draws : ℕ
  white_draws : ℕ

/-- The probability of drawing a white ball -/
noncomputable def white_probability (bag : Bag) : ℝ :=
  (bag.white_balls : ℝ) / ((bag.white_balls + bag.black_balls) : ℝ)

/-- The probability of drawing a black ball -/
noncomputable def black_probability (bag : Bag) : ℝ :=
  (bag.black_balls : ℝ) / ((bag.white_balls + bag.black_balls) : ℝ)

/-- The frequency of drawing white balls -/
noncomputable def white_frequency (result : DrawResult) : ℝ :=
  (result.white_draws : ℝ) / (result.total_draws : ℝ)

theorem estimate_majority_color (bag : Bag) (result : DrawResult) :
  result.total_draws = 10 →
  result.white_draws = 9 →
  white_frequency result = 0.9 →
  white_probability bag > black_probability bag :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_estimate_majority_color_l1011_101115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trees_original_height_l1011_101103

/-- Calculates the combined original height of three trees in feet, given their current heights in inches and growth percentages. -/
noncomputable def combined_original_height (height1 height2 height3 : ℝ) (growth1 growth2 growth3 : ℝ) : ℝ :=
  let original1 := height1 / (1 + growth1)
  let original2 := height2 / (1 + growth2)
  let original3 := height3 / (1 + growth3)
  (original1 + original2 + original3) / 12

/-- The combined original height of the three trees is approximately 37.81 feet. -/
theorem trees_original_height :
  let height1 := (240 : ℝ)
  let height2 := (300 : ℝ)
  let height3 := (180 : ℝ)
  let growth1 := (0.7 : ℝ)
  let growth2 := (0.5 : ℝ)
  let growth3 := (0.6 : ℝ)
  abs (combined_original_height height1 height2 height3 growth1 growth2 growth3 - 37.81) < 0.01 := by
  sorry

-- Remove the #eval statement as it's not necessary for the theorem and may cause issues with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trees_original_height_l1011_101103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decrease_interval_of_symmetric_to_g_l1011_101119

noncomputable def g (x : ℝ) : ℝ := Real.log x / Real.log (1/2)

def symmetric_to_g (f : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y ↔ g y = x

theorem decrease_interval_of_symmetric_to_g (f : ℝ → ℝ) 
  (h : symmetric_to_g f) : 
  ∀ x y, x < y → f y < f x :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decrease_interval_of_symmetric_to_g_l1011_101119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_double_intermediate_value_range_l1011_101163

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - (6/5) * x^2

-- Define the double intermediate value property
def is_double_intermediate_value (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x₁ x₂, a < x₁ ∧ x₁ < x₂ ∧ x₂ < b ∧
    (deriv f x₁ = (f b - f a) / (b - a)) ∧
    (deriv f x₂ = (f b - f a) / (b - a))

-- Theorem statement
theorem double_intermediate_value_range :
  ∀ t : ℝ, is_double_intermediate_value f 0 t ↔ (3/5 < t ∧ t < 6/5) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_double_intermediate_value_range_l1011_101163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_semicircle_area_60_degrees_l1011_101102

/-- The area of a figure formed by rotating a semicircle around one of its ends -/
noncomputable def rotated_semicircle_area (R : ℝ) (α : ℝ) : ℝ :=
  (α / (2 * Real.pi)) * (2 * R)^2 * Real.pi / 2

/-- Theorem: The area of a figure formed by rotating a semicircle of radius R 
    around one of its ends by an angle of 60° is equal to (2πR²)/3 -/
theorem rotated_semicircle_area_60_degrees (R : ℝ) (h : R > 0) :
  rotated_semicircle_area R (Real.pi / 3) = (2 * Real.pi * R^2) / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_semicircle_area_60_degrees_l1011_101102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_is_square_l1011_101150

-- Define the vertices of the quadrilateral
noncomputable def A : ℝ × ℝ := (4, 1 + Real.sqrt 2)
noncomputable def B : ℝ × ℝ := (1, 5 + Real.sqrt 2)
noncomputable def C : ℝ × ℝ := (-3, 2 + Real.sqrt 2)
noncomputable def D : ℝ × ℝ := (0, -2 + Real.sqrt 2)

-- Define vectors
noncomputable def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
noncomputable def DC : ℝ × ℝ := (C.1 - D.1, C.2 - D.2)
noncomputable def AC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)
noncomputable def BD : ℝ × ℝ := (D.1 - B.1, D.2 - B.2)

-- Define dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Theorem statement
theorem quadrilateral_is_square : 
  AB = DC ∧ 
  dot_product AC BD = 0 ∧ 
  dot_product AB AB = dot_product AC AC :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_is_square_l1011_101150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_equals_minus_one_minus_three_i_vector_AB_l1011_101101

-- Define the complex number z
noncomputable def z : ℂ := (-3 + Complex.I) / (Complex.I ^ 7)

-- Theorem 1: Prove that z equals -1 - 3i
theorem z_equals_minus_one_minus_three_i : z = -1 - 3 * Complex.I := by
  sorry

-- Define w
def w : ℂ := 2 - Complex.I

-- Theorem 2: Prove that w - z equals 3 + 2i
theorem vector_AB : w - z = 3 + 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_equals_minus_one_minus_three_i_vector_AB_l1011_101101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_eleven_l1011_101100

def sequence_a : ℕ → ℤ
  | 0 => 1  -- Adding case for 0
  | 1 => 1
  | 2 => 3
  | (n + 3) => (n + 4) * sequence_a (n + 2) - (n + 3) * sequence_a (n + 1)

theorem divisibility_by_eleven (n : ℕ) :
  11 ∣ sequence_a n ↔ n = 4 ∨ n = 8 ∨ n ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_eleven_l1011_101100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2016_at_pi_over_3_l1011_101148

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

noncomputable def f_n : ℕ → (ℝ → ℝ)
  | 0 => f
  | n + 1 => deriv (f_n n)

theorem f_2016_at_pi_over_3 : f_n 2016 (π / 3) = (Real.sqrt 3 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2016_at_pi_over_3_l1011_101148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_terms_in_sequence_l1011_101158

def sequenceterm (n : ℕ) : ℚ :=
  6400 / 2^n

theorem integer_terms_in_sequence : 
  {n : ℕ | ∃ k : ℤ, (sequenceterm n : ℚ) = k} = Finset.range 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_terms_in_sequence_l1011_101158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_order_exists_l1011_101127

-- Define the students
inductive Student : Type
  | A | B | C | D | E

-- Define the positions
def Position := Fin 5

-- Define the guesses made by each student
def guesses : Student → List (Student × Position)
  | Student.A => [(Student.B, ⟨2, by norm_num⟩), (Student.C, ⟨4, by norm_num⟩)]
  | Student.B => [(Student.E, ⟨3, by norm_num⟩), (Student.D, ⟨4, by norm_num⟩)]
  | Student.C => [(Student.A, ⟨0, by norm_num⟩), (Student.E, ⟨3, by norm_num⟩)]
  | Student.D => [(Student.C, ⟨0, by norm_num⟩), (Student.B, ⟨1, by norm_num⟩)]
  | Student.E => [(Student.A, ⟨2, by norm_num⟩), (Student.D, ⟨3, by norm_num⟩)]

-- Define a function to check if a guess is correct
def isCorrectGuess (order : Student → Position) (s : Student) (p : Position) : Prop :=
  order s = p

-- Define the theorem
theorem correct_order_exists :
  ∃ (order : Student → Position),
    (∀ s : Student, ∃ p : Position, order s = p) ∧
    (∀ s : Student, ∃ (g : Student × Position), g ∈ guesses s ∧ isCorrectGuess order g.1 g.2) ∧
    order Student.C = ⟨0, by norm_num⟩ ∧
    order Student.A = ⟨2, by norm_num⟩ ∧
    order Student.D = ⟨4, by norm_num⟩ :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_order_exists_l1011_101127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisors_360_not_divisible_by_3_l1011_101176

def divisors_not_divisible_by_three (n : ℕ) : ℕ :=
  (Finset.filter (fun d => d ∣ n ∧ ¬(3 ∣ d)) (Finset.range (n + 1))).card

theorem count_divisors_360_not_divisible_by_3 :
  divisors_not_divisible_by_three 360 = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisors_360_not_divisible_by_3_l1011_101176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l1011_101164

-- Define the curves C₁ and C₂
noncomputable def C₁ (α : ℝ) : ℝ × ℝ := (1 + Real.cos α, Real.sin α)

def C₂ (x y : ℝ) : Prop := x^2/3 + y^2 = 1

-- Define the ray θ = π/3
noncomputable def ray (ρ : ℝ) : ℝ × ℝ := (ρ * Real.cos (Real.pi/3), ρ * Real.sin (Real.pi/3))

-- Theorem statement
theorem intersection_distance :
  ∃ (ρ₁ ρ₂ : ℝ),
    (ray ρ₁).1^2 + (ray ρ₁).2^2 = 2 * (ray ρ₁).1 ∧  -- Intersection with C₁
    C₂ (ray ρ₂).1 (ray ρ₂).2 ∧                      -- Intersection with C₂
    |ρ₁ - ρ₂| = |Real.sqrt 30 / 5 - 1| :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l1011_101164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_s_l1011_101139

-- Define the function s(x) as noncomputable
noncomputable def s (x : ℝ) : ℝ := 1 / (2 - x)^2

-- State the theorem about the range of s(x)
theorem range_of_s :
  ∀ y : ℝ, y > 0 ↔ ∃ x : ℝ, x ≠ 2 ∧ s x = y :=
by
  sorry  -- Proof is omitted


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_s_l1011_101139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fabric_bolt_length_l1011_101169

/-- The length of a bolt of fabric -/
noncomputable def bolt_length (living_room_length living_room_width bedroom_length bedroom_width bolt_width remaining_area : ℝ) : ℝ :=
  (living_room_length * living_room_width + bedroom_length * bedroom_width + remaining_area) / bolt_width

/-- Theorem stating the length of the bolt of fabric -/
theorem fabric_bolt_length :
  bolt_length 4 6 2 4 16 160 = 12 := by
  -- Unfold the definition of bolt_length
  unfold bolt_length
  -- Simplify the arithmetic
  simp [mul_add, add_mul, add_assoc]
  -- Check that the equality holds
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fabric_bolt_length_l1011_101169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_power_less_than_500_l1011_101145

theorem greatest_power_less_than_500 (a b : ℕ) (h1 : b > 1) 
  (h2 : ∀ (x y : ℕ), y > 1 → x^y < 500 → a^b ≥ x^y) 
  (h3 : a^b < 500) : a + b = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_power_less_than_500_l1011_101145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_range_sin_cos_range_bounds_l1011_101183

theorem sin_cos_range (x : ℝ) (h : 0 < x ∧ x ≤ π/3) :
  ∃ y, y ∈ Set.Ioo 1 (2 + 2 * Real.sqrt 2) ∧
    ∃ x', 0 < x' ∧ x' ≤ π/3 ∧ y = Real.sin x' + Real.cos x' + Real.sin x' * Real.cos x' :=
by sorry

theorem sin_cos_range_bounds (y : ℝ) :
  (∃ x, 0 < x ∧ x ≤ π/3 ∧ y = Real.sin x + Real.cos x + Real.sin x * Real.cos x) →
  y ∈ Set.Icc 1 (2 + 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_range_sin_cos_range_bounds_l1011_101183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_two_l1011_101175

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the expression
noncomputable def expression : ℝ :=
  log10 4 + log10 9 + 2 * Real.sqrt ((log10 6)^2 - log10 36 + 1)

-- Theorem statement
theorem expression_equals_two : expression = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_two_l1011_101175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifty_above_median_l1011_101198

/-- Represents a group of people with their ages -/
structure AgeGroup where
  members : Finset ℕ
  ages : ℕ → ℕ

/-- The number of members in the group is 100 -/
def group_size (g : AgeGroup) : Prop := g.members.card = 100

/-- The average age of the group is 21 -/
def average_age (g : AgeGroup) : Prop := (g.members.sum g.ages) / g.members.card = 21

/-- The youngest member is 1 year old -/
def youngest_age (g : AgeGroup) : Prop := ∃ m ∈ g.members, g.ages m = 1

/-- The oldest member is 70 years old -/
def oldest_age (g : AgeGroup) : Prop := ∃ m ∈ g.members, g.ages m = 70

/-- The median age of the group -/
noncomputable def median_age (g : AgeGroup) : ℕ := sorry

/-- The main theorem: 50 members have an age greater than the median -/
theorem fifty_above_median (g : AgeGroup) 
  (h1 : group_size g) (h2 : average_age g) (h3 : youngest_age g) (h4 : oldest_age g) : 
  (g.members.filter (λ m => g.ages m > median_age g)).card = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifty_above_median_l1011_101198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_solution_correct_l1011_101179

/-- The smallest solution to the equation 3x/(x-3) + (3x^2 - 36)/x = 14 -/
noncomputable def smallest_solution : ℝ := (5 - Real.sqrt 241) / 6

/-- The equation in question -/
def equation (x : ℝ) : Prop :=
  x ≠ 0 ∧ x ≠ 3 ∧ 3 * x / (x - 3) + (3 * x^2 - 36) / x = 14

theorem smallest_solution_correct :
  equation smallest_solution ∧
  ∀ y, equation y → y ≤ smallest_solution := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_solution_correct_l1011_101179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equations_correctness_l1011_101107

-- Define the equations
noncomputable def equation1 : ℝ := Real.sqrt 2 + Real.sqrt 5
noncomputable def equation2 (a : ℝ) : ℝ := 5 * Real.sqrt a - 3 * Real.sqrt a
noncomputable def equation3 : ℝ := (Real.sqrt 8 + Real.sqrt 50) / 2
noncomputable def equation4 (a : ℝ) : ℝ := 2 * Real.sqrt (3 * a) + Real.sqrt (27 * a)

-- State the theorem
theorem equations_correctness :
  (∃ a : ℝ, equation2 a = 2 * Real.sqrt a) ∧
  (∃ a : ℝ, equation4 a = 5 * Real.sqrt (3 * a)) ∧
  (equation1 ≠ Real.sqrt 7) ∧
  (equation3 ≠ 7) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equations_correctness_l1011_101107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_outer_chips_l1011_101197

/-- Represents a circular chip in a 2D plane -/
structure Chip where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents an arrangement of chips -/
structure ChipArrangement where
  centerpieces : List Chip
  outer_chips : List Chip

/-- Checks if two chips are tangent -/
def are_tangent (c1 c2 : Chip) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

/-- Checks if the arrangement satisfies the tangency conditions -/
def valid_arrangement (arr : ChipArrangement) : Prop :=
  arr.centerpieces.length = 2 ∧
  ∀ c ∈ arr.outer_chips,
    (∃ c1 c2, c1 ∈ arr.outer_chips ∧ c2 ∈ arr.outer_chips ∧ c ≠ c1 ∧ c ≠ c2 ∧ c1 ≠ c2 ∧
      are_tangent c c1 ∧ are_tangent c c2) ∧
    (∀ cp ∈ arr.centerpieces, are_tangent c cp)

/-- The main theorem stating that the maximum number of outer chips is 12 -/
theorem max_outer_chips :
  ∀ arr : ChipArrangement,
    valid_arrangement arr →
    arr.outer_chips.length ≤ 12 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_outer_chips_l1011_101197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_complement_B_is_empty_l1011_101133

-- Define the sets A and B
def A : Set ℝ := {x | Real.sqrt (x - 2) ≤ 0}
def B : Set ℝ := {x | (10 : ℝ)^(x^2 - 2) = (10 : ℝ)^x}

-- State the theorem
theorem intersection_A_complement_B_is_empty :
  A ∩ (Set.univ \ B) = ∅ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_complement_B_is_empty_l1011_101133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mode_explains_decision_dealer_decision_based_on_mode_l1011_101196

/-- Represents the sales data for different types of packaged Taicha -/
def sales_data : List Nat := [15, 22, 18, 10]

/-- The type of packaged Taicha with the highest sales -/
def highest_sales_type : Nat := 1  -- Index 1 corresponds to type B (0-based indexing)

/-- Definition of the mode as the highest sales figure -/
def mode (data : List Nat) : Nat :=
  (data.maximum.getD 0)  -- Use getD to provide a default value of 0 if the list is empty

/-- Theorem stating that the mode (highest sales) corresponds to type B -/
theorem mode_explains_decision :
  mode sales_data = sales_data[highest_sales_type] := by
  sorry

/-- Theorem stating that the dealer's decision is based on the mode -/
theorem dealer_decision_based_on_mode :
  ∀ (index : Fin sales_data.length),
    sales_data[index] ≤ sales_data[highest_sales_type] := by
  sorry

#eval mode sales_data  -- This will print the result of the mode function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mode_explains_decision_dealer_decision_based_on_mode_l1011_101196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_angle_l1011_101189

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 3 * x + Real.sqrt 3 * y + 1 = 0

-- Define the slope of the line
noncomputable def line_slope : ℝ := -Real.sqrt 3

-- Define the slope angle in degrees
def slope_angle : ℝ := 120

-- Theorem statement
theorem line_slope_angle :
  slope_angle = 120 := by
  -- The proof would go here, but we'll use sorry for now
  sorry

#check line_slope_angle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_angle_l1011_101189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_is_semicircle_l1011_101142

/-- Curve C is defined by the parametric equations x = 2cos(θ) and y = 1 + 2sin(θ), 
    where θ ∈ [-π/2, π/2] -/
def curve_C : Set (ℝ × ℝ) :=
  {(x, y) | ∃ θ : ℝ, -Real.pi/2 ≤ θ ∧ θ ≤ Real.pi/2 ∧ x = 2 * Real.cos θ ∧ y = 1 + 2 * Real.sin θ}

/-- A semicircle with center (0, 1) and radius 2 -/
def semicircle : Set (ℝ × ℝ) :=
  {(x, y) | x^2 + (y-1)^2 = 4 ∧ 0 ≤ x ∧ x ≤ 2}

/-- Theorem stating that curve C represents a semicircle -/
theorem curve_C_is_semicircle : curve_C = semicircle := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_is_semicircle_l1011_101142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1011_101173

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 2) - 3 * x

theorem tangent_line_equation (x₀ y₀ : ℝ) (h₁ : x₀ = -1) (h₂ : y₀ = 3) (h₃ : f x₀ = y₀) :
  ∃ (a b c : ℝ), a * x₀ + b * y₀ + c = 0 ∧
                 (∀ x y, a * x + b * y + c = 0 ↔ y = (-a / b) * x - (c / b)) ∧
                 b ≠ 0 ∧
                 HasDerivAt f (-a / b) x₀ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1011_101173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_sum_l1011_101144

theorem existence_of_sum (A : Set ℕ) (m n : ℕ) 
  (h_A_infinite : Set.Infinite A)
  (h_m : m > 1)
  (h_n : n > 1)
  (h_coprime : Nat.Coprime m n)
  (h_prime_condition : ∀ p : ℕ, Nat.Prime p → ¬(p ∣ n) → 
    Set.Infinite {a ∈ A | ¬(p ∣ a)}) :
  ∃ S : Finset ℕ, S.toSet ⊆ A ∧ 
    ∃ sum : ℕ, sum = S.sum id ∧ 
      sum % m = 1 ∧ sum % n = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_sum_l1011_101144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_range_l1011_101180

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x + 1 else Real.exp (x * Real.log 2)

-- State the theorem
theorem f_inequality_range (x : ℝ) :
  f x + f (x - 1/2) > 1 ↔ x > -1/4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_range_l1011_101180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_knight_graph_existence_l1011_101129

/-- A knight relationship graph is a simple undirected graph where vertices represent knights
    and edges represent friendships. The absence of an edge represents enmity. -/
structure KnightGraph (n : ℕ) where
  friends : Fin n → Fin n → Bool
  symm : ∀ i j, friends i j = friends j i
  irrefl : ∀ i, friends i i = false

/-- Each knight has exactly three enemies -/
def has_three_enemies (G : KnightGraph n) (i : Fin n) : Prop :=
  (Finset.univ.filter (λ j => ¬G.friends i j)).card = 3

/-- The enemies of a knight's friends are also his enemies -/
def transitive_enmity (G : KnightGraph n) : Prop :=
  ∀ i j k, G.friends i j = true → G.friends j k = false → G.friends i k = false

/-- A valid knight graph satisfies all the conditions -/
def valid_knight_graph (G : KnightGraph n) : Prop :=
  (∀ i, has_three_enemies G i) ∧ transitive_enmity G

/-- The main theorem: A valid knight graph exists if and only if n = 4 or n = 6 -/
theorem knight_graph_existence (n : ℕ) :
  (∃ G : KnightGraph n, valid_knight_graph G) ↔ n = 4 ∨ n = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_knight_graph_existence_l1011_101129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_expressions_equality_l1011_101141

/-- Given three logarithmic expressions, we define them as functions of x -/
noncomputable def log_expr1 (x : ℝ) : ℝ := Real.log (x - 4) / Real.log (Real.sqrt (2*x - 8))
noncomputable def log_expr2 (x : ℝ) : ℝ := Real.log (5*x - 26) / Real.log ((x - 4)^2)
noncomputable def log_expr3 (x : ℝ) : ℝ := Real.log (2*x - 8) / Real.log (Real.sqrt (5*x - 26))

/-- The theorem states that x = 6 is the only value satisfying the conditions -/
theorem log_expressions_equality (x : ℝ) :
  (log_expr1 x = log_expr2 x ∧ log_expr3 x = log_expr1 x + 1) ∨
  (log_expr2 x = log_expr3 x ∧ log_expr1 x = log_expr2 x + 1) ∨
  (log_expr3 x = log_expr1 x ∧ log_expr2 x = log_expr3 x + 1) ↔
  x = 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_expressions_equality_l1011_101141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_point_with_modified_distances_l1011_101191

-- Define the triangle ABC
variable (A B C : EuclideanSpace ℝ (Fin 2))

-- Define point M and its distances to vertices
variable (M : EuclideanSpace ℝ (Fin 2))
variable (a b c : ℝ)

-- Distances from M to vertices
axiom dist_MA : ‖A - M‖ = a
axiom dist_MB : ‖B - M‖ = b
axiom dist_MC : ‖C - M‖ = c

-- The theorem to prove
theorem no_point_with_modified_distances :
  ¬∃ (N : EuclideanSpace ℝ (Fin 2)) (d : ℝ),
    d ≠ 0 ∧
    ‖A - N‖ = Real.sqrt (a^2 + d) ∧
    ‖B - N‖ = Real.sqrt (b^2 + d) ∧
    ‖C - N‖ = Real.sqrt (c^2 + d) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_point_with_modified_distances_l1011_101191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sam_catches_alice_l1011_101166

/-- The time (in minutes) it takes for Sam to catch up with Alice -/
noncomputable def catch_up_time (alice_speed sam_speed : ℝ) (initial_distance : ℝ) : ℝ :=
  initial_distance / (sam_speed - alice_speed) * 60

/-- Theorem stating that Sam catches up with Alice in 30 minutes -/
theorem sam_catches_alice :
  let alice_speed : ℝ := 10  -- Alice's speed in miles per hour
  let sam_speed : ℝ := 16    -- Sam's speed in miles per hour
  let initial_distance : ℝ := 3  -- Initial distance between Sam and Alice in miles
  catch_up_time alice_speed sam_speed initial_distance = 30 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sam_catches_alice_l1011_101166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_bound_for_curve_C_l1011_101193

noncomputable def curve_C (x y : ℝ) : Prop :=
  Real.sqrt ((x + 2)^2 + y^2) * Real.sqrt ((x - 2)^2 + y^2) = 16

def M : ℝ × ℝ := (-2, 0)
def N : ℝ × ℝ := (2, 0)

noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let (x₃, y₃) := C
  (1/2) * abs ((x₁ - x₃) * (y₂ - y₃) - (x₂ - x₃) * (y₁ - y₃))

theorem area_bound_for_curve_C :
  ∀ x y : ℝ, curve_C x y → triangle_area M N (x, y) ≤ 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_bound_for_curve_C_l1011_101193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_greater_than_two_thirds_l1011_101178

/-- Definition of an ellipse C with eccentricity e -/
structure Ellipse :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : c^2 = a^2 - b^2)

/-- Definition of a point on the ellipse -/
def PointOnEllipse (C : Ellipse) (x y : ℝ) : Prop :=
  x^2 / C.a^2 + y^2 / C.b^2 = 1

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (C : Ellipse) : ℝ := C.c / C.a

/-- IsoscelesObtuseTriangle is a proposition that we assume exists -/
axiom IsoscelesObtuseTriangle : ℝ × ℝ → ℝ × ℝ → ℝ × ℝ → Prop

/-- Theorem about the eccentricity of a specific ellipse -/
theorem eccentricity_greater_than_two_thirds (C : Ellipse) 
  (h4 : ∃ (N : ℝ × ℝ), PointOnEllipse C N.1 N.2 ∧ 
    IsoscelesObtuseTriangle (C.c, 0) (14 * C.a^2 / (9 * C.c), 0) N) :
  eccentricity C > 2/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_greater_than_two_thirds_l1011_101178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_range_theorem_l1011_101137

/-- The ellipse E -/
def E (x y : ℝ) : Prop := x^2 / 16 + y^2 / 12 = 1

/-- The hyperbola H -/
def H (x y : ℝ) : Prop := x^2 - y^2 = 2

/-- Point M on the major axis of E -/
def M (m : ℝ) : ℝ × ℝ := (m, 0)

/-- Any point P on E -/
def P : ℝ × ℝ → Prop := λ p => E p.1 p.2

/-- The distance between M and P -/
noncomputable def MP (m : ℝ) (p : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - m)^2 + p.2^2)

/-- The right vertex of E -/
def rightVertex : ℝ × ℝ := (4, 0)

theorem ellipse_range_theorem :
  (E 4 0) ∧ 
  (∀ x y, E x y ↔ H x y) →
  {m : ℝ | m ∈ Set.Icc (-4 : ℝ) 4 ∧
           ∀ p, P p → MP m p ≥ MP m rightVertex} = 
  Set.Icc (1 : ℝ) 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_range_theorem_l1011_101137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_stop_theorem_l1011_101132

/-- Calculates the stop time of a bus given its speeds with and without stoppages -/
noncomputable def bus_stop_time (speed_without_stops : ℝ) (speed_with_stops : ℝ) : ℝ :=
  let distance_diff := speed_without_stops - speed_with_stops
  let time_fraction := distance_diff / speed_without_stops
  time_fraction * 60

/-- Theorem: A bus with speed 54 km/h without stops and 45 km/h with stops, stops for 10 minutes per hour -/
theorem bus_stop_theorem :
  bus_stop_time 54 45 = 10 := by
  -- Unfold the definition of bus_stop_time
  unfold bus_stop_time
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_stop_theorem_l1011_101132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_pyramid_volume_l1011_101122

-- Define the pyramid
structure Pyramid :=
  (AB : ℝ)
  (AC : ℝ)
  (sin_BAC : ℝ)
  (lateral_angle : ℝ)

-- Define the theorem
theorem max_pyramid_volume 
  (p : Pyramid)
  (h_AB : p.AB = 5)
  (h_AC : p.AC = 8)
  (h_sin_BAC : p.sin_BAC = 3/5)
  (h_angle : p.lateral_angle ≤ π/3)
  (h_equal_angles : ∀ (edge : Fin 3), p.lateral_angle = p.lateral_angle) :
  ∃ (max_volume : ℝ), max_volume = 10 * Real.sqrt 51 ∧ 
  ∀ (volume : ℝ), volume ≤ max_volume := by
  sorry

#check max_pyramid_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_pyramid_volume_l1011_101122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_range_of_a_unique_zero_point_l1011_101120

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (2 * x) - a * x - 1

-- Statement for monotonicity
theorem monotonicity (a : ℝ) :
  (a ≤ 0 → StrictMono (f a)) ∧
  (a > 0 → ∀ x y, x < y → x < Real.log (a/2) / 2 → y < Real.log (a/2) / 2 → f a y < f a x) ∧
  (a > 0 → ∀ x y, x < y → x > Real.log (a/2) / 2 → y > Real.log (a/2) / 2 → f a x < f a y) :=
sorry

-- Statement for the range of a
theorem range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, x > 0 → f a x > 0) ↔ a ≤ 2 :=
sorry

-- Statement for the unique zero point
theorem unique_zero_point (a : ℝ) (x₀ : ℝ) :
  (x₀ > 0 ∧ f a x₀ = 0 ∧ ∀ x > 0, x ≠ x₀ → f a x ≠ 0) →
  x₀ < a - 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_range_of_a_unique_zero_point_l1011_101120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crank_slider_motion_l1011_101167

/-- Crank-slider mechanism -/
structure CrankSlider where
  OA : ℝ
  AB : ℝ
  MB : ℝ
  ω : ℝ

/-- Point on the connecting rod -/
structure Point where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Velocity components -/
structure Velocity where
  vx : ℝ → ℝ
  vy : ℝ → ℝ

def mechanism : CrankSlider :=
  { OA := 90
  , AB := 90
  , MB := 30
  , ω := 10 }

noncomputable def point_M (m : CrankSlider) : Point :=
  { x := λ t => 150 * Real.cos (m.ω * t)
  , y := λ t => 90 * Real.sin (m.ω * t) }

noncomputable def velocity_M (m : CrankSlider) : Velocity :=
  { vx := λ t => -1500 * Real.sin (10 * t)
  , vy := λ t => 900 * Real.cos (10 * t) }

theorem crank_slider_motion (m : CrankSlider) (t : ℝ) :
  m = mechanism →
  (point_M m).x t = 150 * Real.cos (m.ω * t) ∧
  (point_M m).y t = 90 * Real.sin (m.ω * t) ∧
  (velocity_M m).vx t = -1500 * Real.sin (10 * t) ∧
  (velocity_M m).vy t = 900 * Real.cos (10 * t) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_crank_slider_motion_l1011_101167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_zeros_is_negative_two_l1011_101187

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := (x - 3)^2 + 4

-- Define the transformations
def rotate_180 (f : ℝ → ℝ) (x : ℝ) : ℝ := -f x + 2 * f 3
def shift_left (f : ℝ → ℝ) (x : ℝ) : ℝ := f (x + 4)
def shift_down (f : ℝ → ℝ) (x : ℝ) : ℝ := f x - 3

-- Apply transformations to the original parabola
def transformed_parabola : ℝ → ℝ :=
  λ x => shift_down (shift_left (rotate_180 original_parabola)) x

-- Define the zeros of the transformed parabola
def zeros (f : ℝ → ℝ) : Set ℝ := {x : ℝ | f x = 0}

-- Theorem statement
theorem sum_of_zeros_is_negative_two :
  ∃ p q : ℝ, p ∈ zeros transformed_parabola ∧ 
             q ∈ zeros transformed_parabola ∧ 
             p + q = -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_zeros_is_negative_two_l1011_101187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hall_theorem_application_l1011_101118

open Matrix
open Finset

theorem hall_theorem_application (N : ℕ) (A : Matrix (Fin N) (Fin N) ℝ) 
  (h_nonneg : ∀ i j, 0 ≤ A i j) 
  (h_row_sum : ∀ i, (univ.sum (λ j ↦ A i j)) = 1)
  (h_col_sum : ∀ j, (univ.sum (λ i ↦ A i j)) = 1) :
  ∃ (f : Fin N → Fin N), Function.Injective f ∧ ∀ i, A i (f i) > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hall_theorem_application_l1011_101118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_half_angle_special_case_l1011_101161

theorem sin_half_angle_special_case (α : ℝ) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.cos α = (1 + Real.sqrt 5) / 4) : 
  Real.sin (α / 2) = (Real.sqrt 5 - 1) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_half_angle_special_case_l1011_101161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_CDE_l1011_101155

/-- Given a figure with rectangles OAED (6x6) and ABCD (24x18), where OC is the diagonal
    and E is the intersection of OC and ED, prove that the area of triangle CDE is 121.5 -/
theorem area_triangle_CDE (O A B C D E : ℝ × ℝ) : 
  O = (0, 0) →
  A = (6, 0) →
  B = (24, 0) →
  C = (24, 18) →
  D = (6, 18) →
  E.1 = 6 →
  E.2 ∈ Set.Icc 6 18 →
  (C.2 - O.2) / (C.1 - O.1) = (E.2 - O.2) / (E.1 - O.1) →
  (1/2) * (C.1 - E.1) * (C.2 - E.2) = 121.5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_CDE_l1011_101155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_4_5_l1011_101153

/-- Calculates the area of a triangle given its base and height -/
noncomputable def triangle_area (base height : ℝ) : ℝ := (base * height) / 2

/-- Theorem: The area of a triangle with base 4 and height 5 is 10 -/
theorem triangle_area_4_5 : triangle_area 4 5 = 10 := by
  -- Unfold the definition of triangle_area
  unfold triangle_area
  -- Simplify the arithmetic
  simp [mul_comm, div_eq_mul_inv]
  -- Check that 4 * 5 * (1/2) = 10
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_4_5_l1011_101153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_june_ride_time_to_cassandra_l1011_101149

/-- The time it takes June to ride to Cassandra's house -/
noncomputable def ride_time_to_cassandra (distance_to_julia : ℝ) (time_to_julia : ℝ) 
  (distance_to_cassandra : ℝ) (rest_time : ℝ) : ℝ :=
  let rate := distance_to_julia / time_to_julia
  let half_distance := distance_to_cassandra / 2
  let time_to_half := half_distance / rate
  2 * time_to_half + rest_time

/-- Theorem: Given the conditions, June takes 25 minutes to ride to Cassandra's house -/
theorem june_ride_time_to_cassandra : 
  ride_time_to_cassandra 1 4 5 5 = 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_june_ride_time_to_cassandra_l1011_101149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_three_equality_l1011_101104

theorem power_of_three_equality (x : ℝ) : (3 : ℝ)^7 * (3 : ℝ)^x = 81 → x = -3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_three_equality_l1011_101104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_constraint_l1011_101171

theorem max_value_constraint :
  ∀ (x y z : ℝ), 9*x^2 + 4*y^2 + 25*z^2 = 1 →
  (8*x + 3*y + 10*z) ≤ Real.sqrt 481 / 6 ∧
  ∃ (x₀ y₀ z₀ : ℝ), 9*x₀^2 + 4*y₀^2 + 25*z₀^2 = 1 ∧ 8*x₀ + 3*y₀ + 10*z₀ = Real.sqrt 481 / 6 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_constraint_l1011_101171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_sqrt_x_minus_3_l1011_101186

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 3)

-- State the theorem
theorem domain_of_sqrt_x_minus_3 :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≥ 3} :=
by
  -- The proof is omitted using 'sorry'
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_sqrt_x_minus_3_l1011_101186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_one_minus_x_squared_plus_x_l1011_101151

theorem integral_sqrt_one_minus_x_squared_plus_x : 
  ∫ x in (-1 : ℝ)..1, (Real.sqrt (1 - x^2) + x) = π / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_one_minus_x_squared_plus_x_l1011_101151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_exp_pi_third_i_l1011_101116

open Complex

-- Define Euler's formula
axiom euler_formula (x : ℝ) : exp (I * x) = cos x + I * sin x

-- Define the modulus of a complex number
noncomputable def modulus (z : ℂ) : ℝ := Real.sqrt (z.re ^ 2 + z.im ^ 2)

-- Theorem to prove
theorem modulus_exp_pi_third_i : modulus (exp ((π / 3) * I)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_exp_pi_third_i_l1011_101116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_center_location_l1011_101170

-- Define a 3D line
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

-- Define a 3D plane
structure Plane3D where
  normal : ℝ × ℝ × ℝ
  point : ℝ × ℝ × ℝ

-- Define a parallelepiped
structure Parallelepiped where
  vertices : List (ℝ × ℝ × ℝ)

def are_skew (l1 l2 : Line3D) : Prop :=
  sorry

noncomputable def equidistant_plane (l1 l2 : Line3D) : Plane3D :=
  sorry

noncomputable def intersection_point (p1 p2 p3 : Plane3D) : ℝ × ℝ × ℝ :=
  sorry

noncomputable def center (p : Parallelepiped) : ℝ × ℝ × ℝ :=
  sorry

theorem parallelepiped_center_location 
  (l1 l2 l3 : Line3D) 
  (h12 : are_skew l1 l2) 
  (h23 : are_skew l2 l3) 
  (h13 : are_skew l1 l3) 
  (p : Parallelepiped) 
  (hp : ∀ v ∈ p.vertices, v ∈ l1.point :: l2.point :: l3.point :: []) :
  center p = intersection_point 
    (equidistant_plane l1 l2) 
    (equidistant_plane l2 l3) 
    (equidistant_plane l1 l3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_center_location_l1011_101170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_18_10_l1011_101192

theorem binomial_18_10 (h1 : Nat.choose 16 7 = 11440) (h2 : Nat.choose 16 (16 - 9) = Nat.choose 16 7) :
  Nat.choose 18 10 = 43758 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_18_10_l1011_101192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l1011_101147

/-- Parabola C: y² = 4x -/
def parabola_C (x y : ℝ) : Prop := y^2 = 4*x

/-- Line l passing through (4, 0) -/
def line_l (t x y : ℝ) : Prop := x = t*y + 4

/-- Area of triangle ABO given y-coordinates of A and B -/
noncomputable def triangle_area (y₁ y₂ : ℝ) : ℝ :=
  let t := (y₁ + y₂) / 4
  2 * Real.sqrt (16 * (t^2 + 4))

theorem min_triangle_area :
  ∀ t : ℝ, ∃ y₁ y₂ : ℝ,
    parabola_C (t*y₁ + 4) y₁ ∧
    parabola_C (t*y₂ + 4) y₂ ∧
    line_l t (t*y₁ + 4) y₁ ∧
    line_l t (t*y₂ + 4) y₂ ∧
    triangle_area y₁ y₂ ≥ 16 :=
by sorry

#check min_triangle_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l1011_101147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_12769_l1011_101136

theorem largest_prime_factor_of_12769 : (Nat.factors 12769).maximum? = some 251 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_12769_l1011_101136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossiblePartition_l1011_101195

/-- Given a natural number n, returns the sequence of its digits -/
def digits (n : ℕ) : List ℕ := sorry

/-- Given a list of natural numbers, returns the concatenation of their digit sequences -/
def concatenateDigits (l : List ℕ) : List ℕ := sorry

theorem impossiblePartition (k : ℕ) : 
  ¬ ∃ (A B : Set ℕ), 
    (A ∪ B = Finset.range k.succ) ∧ 
    (A ∩ B = ∅) ∧
    (A ≠ ∅) ∧ (B ≠ ∅) ∧
    (∃ (orderA orderB : List ℕ), 
      (orderA.toFinset = A) ∧ 
      (orderB.toFinset = B) ∧ 
      (concatenateDigits orderA = concatenateDigits orderB)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossiblePartition_l1011_101195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shop_equation_system_l1011_101160

/-- Represents the number of rooms in the shop -/
def x : ℕ → ℕ := fun _ => 0

/-- Represents the number of guests in the shop -/
def y : ℕ → ℕ := fun _ => 0

/-- The system of equations describing the relationship between rooms and guests -/
def correct_system (x y : ℕ → ℕ) : Prop :=
  ∀ n, (7 * (x n) + 7 = y n) ∧ (9 * ((x n) - 1) = y n)

/-- Theorem stating that the given system of equations correctly describes the shop scenario -/
theorem shop_equation_system :
  (∀ x y : ℕ → ℕ, (∀ n, 7 * (x n) + 7 = y n) → ∃ z : ℕ, z = 7) →  
  (∀ x y : ℕ → ℕ, (∀ n, 9 * ((x n) - 1) = y n) → ∃ z : ℕ, z = 1) →  
  ∀ x y : ℕ → ℕ, correct_system x y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shop_equation_system_l1011_101160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l1011_101113

def a : ℕ → ℚ
  | 0 => 1
  | n + 1 => 2 * a n / (2 + a n)

theorem a_formula (n : ℕ) : a n = 2 / (n + 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l1011_101113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_implies_a_l1011_101105

noncomputable def binomial_expansion (a : ℝ) : ℝ → ℝ := fun x => (a / x - Real.sqrt (x / 2)) ^ 9

theorem binomial_coefficient_implies_a (a : ℝ) :
  (∃ c : ℝ, c = 9/4 ∧ ∀ x : ℝ, c * x^3 ∈ Set.range (fun x => binomial_expansion a x)) →
  a = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_implies_a_l1011_101105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_distinct_rhombuses_l1011_101190

/-- Represents a rhombus ABCD in 2D space -/
structure Rhombus where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- The distance between two points in 2D space -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The distance from a point to a line defined by two points -/
noncomputable def distanceToLine (p a b : ℝ × ℝ) : ℝ :=
  let v := (b.1 - a.1, b.2 - a.2)
  let w := (p.1 - a.1, p.2 - a.2)
  abs (v.1 * w.2 - v.2 * w.1) / distance a b

/-- Predicate to check if a given structure is a valid rhombus with the specified properties -/
def isValidRhombus (r : Rhombus) : Prop :=
  distance r.B r.D = 8 ∧
  distanceToLine r.B r.A r.D = 5 ∧
  distance r.A r.B = distance r.B r.C ∧
  distance r.B r.C = distance r.C r.D ∧
  distance r.C r.D = distance r.D r.A

/-- Theorem: There exist exactly two distinct rhombuses satisfying the given conditions -/
theorem two_distinct_rhombuses :
  ∃ (r1 r2 : Rhombus), isValidRhombus r1 ∧ isValidRhombus r2 ∧ r1 ≠ r2 ∧
  ∀ (r : Rhombus), isValidRhombus r → (r = r1 ∨ r = r2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_distinct_rhombuses_l1011_101190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l1011_101156

/-- Represents a hyperbola with semi-major axis a, semi-minor axis b, and focal distance c -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_c_ge_a : c ≥ a

/-- A point on the hyperbola -/
structure HyperbolaPoint (h : Hyperbola) where
  x : ℝ
  y : ℝ
  h_on_hyperbola : x^2 / h.a^2 - y^2 / h.b^2 = 1

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := h.c / h.a

/-- Calculate the angle between three points -/
noncomputable def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

theorem eccentricity_range (h : Hyperbola) (p : HyperbolaPoint h) 
  (h_sine_ratio : Real.sin (angle (p.x, p.y) (-h.c, 0) (h.c, 0)) / Real.sin (angle (p.x, p.y) (h.c, 0) (-h.c, 0)) = h.a / h.c) :
  1 < eccentricity h ∧ eccentricity h < Real.sqrt 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l1011_101156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_angles_is_180_l1011_101168

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ

structure Circle where
  center : Point
  radius : ℝ

-- Define Line as a type synonym for a pair of Points
def Line := Point × Point

-- Define the given conditions
noncomputable def intersectionPoints (c1 c2 : Circle) : Point × Point := sorry

noncomputable def commonTangent (c1 c2 : Circle) : Line := sorry

noncomputable def tangentPoints (l : Line) (c1 c2 : Circle) : Point × Point := sorry

-- Define the angle measure function
noncomputable def angleMeasure (p1 p2 p3 : Point) : ℝ := sorry

-- Theorem statement
theorem sum_of_angles_is_180 (c1 c2 : Circle) : 
  let (K, M) := intersectionPoints c1 c2
  let l := commonTangent c1 c2
  let (A, B) := tangentPoints l c1 c2
  angleMeasure A M B + angleMeasure A K B = 180 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_angles_is_180_l1011_101168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1011_101185

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  h_ab : ℝ
  h_bc : ℝ

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : Real.cos t.B + Real.sin ((t.A + t.C) / 2) = 0)
  (h2 : t.h_ab / t.h_bc = 3 / 5)
  (h3 : t.b = 7) :
  t.B = 2 * Real.pi / 3 ∧ 
  (1/2 * t.a * t.c * Real.sin t.B : ℝ) = 15 * Real.sqrt 3 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1011_101185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_cleverly_numbers_l1011_101140

def is_zero_cleverly_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧ 
  (n / 100) % 10 = 0 ∧
  9 * ((n / 1000) * 100 + (n % 100)) = n

theorem zero_cleverly_numbers :
  ∀ n : ℕ, is_zero_cleverly_number n ↔ n ∈ ({2025, 4050, 6075} : Set ℕ) :=
by sorry

#check zero_cleverly_numbers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_cleverly_numbers_l1011_101140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1011_101126

noncomputable def f (x : ℝ) := 2 * Real.sin x * Real.cos x

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x, f (x + π) = f x) ∧
  (∀ p, p > 0 → (∀ x, f (x + p) = f x) → p ≥ π) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1011_101126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asian_boy_ratio_l1011_101146

/-- Represents a class of students -/
structure StudentClass where
  prob_asian : ℝ  -- Probability of choosing an Asian student
  prob_boy : ℝ    -- Probability of choosing a boy

/-- Conditions for the class -/
def class_conditions (c : StudentClass) : Prop :=
  c.prob_asian = 3/4 * (1 - c.prob_asian) ∧
  c.prob_boy = 3/5 * (1 - c.prob_boy)

/-- Theorem: The ratio of Asian boys to total students is 9/56 -/
theorem asian_boy_ratio (c : StudentClass) (h : class_conditions c) :
  c.prob_asian * c.prob_boy = 9/56 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_asian_boy_ratio_l1011_101146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eq_g_shifted_l1011_101162

/-- The function f(x) = 2^(1-x) -/
noncomputable def f (x : ℝ) : ℝ := 2^(1-x)

/-- The function g(x) = 2^(-x) -/
noncomputable def g (x : ℝ) : ℝ := 2^(-x)

/-- Theorem stating that f(x) is equivalent to g(x-1) -/
theorem f_eq_g_shifted (x : ℝ) : f x = g (x - 1) := by
  -- Expand the definitions of f and g
  unfold f g
  -- Use properties of exponents to show the equality
  simp [Real.rpow_sub, Real.rpow_neg, Real.rpow_one]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eq_g_shifted_l1011_101162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_leak_emptying_time_l1011_101135

/-- Represents the time it takes to empty a cistern with a leak, given the fill times with and without the leak. -/
noncomputable def time_to_empty (fill_time_no_leak : ℝ) (fill_time_with_leak : ℝ) : ℝ :=
  (fill_time_no_leak * fill_time_with_leak) / (fill_time_with_leak - fill_time_no_leak)

/-- Theorem stating that for a cistern that fills in 4 hours without a leak and 5 hours with a leak,
    it will take 20 hours for the leak to empty the full cistern. -/
theorem cistern_leak_emptying_time :
  time_to_empty 4 5 = 20 := by
  -- Unfold the definition of time_to_empty
  unfold time_to_empty
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_leak_emptying_time_l1011_101135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l1011_101157

noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3) + 1

theorem g_properties :
  (∀ x, g x ≤ 2) ∧
  (∀ x ∈ Set.Icc (-5*Real.pi/12) (Real.pi/12), StrictMono g) ∧
  (∀ x, g (5*Real.pi/6 + x) = 2 - g (5*Real.pi/6 - x)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l1011_101157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_align_two_points_l1011_101165

/-- A point with integer coordinates on a plane -/
structure IntPoint where
  x : ℤ
  y : ℤ

/-- A configuration of four points on the plane -/
structure Configuration where
  points : Fin 4 → IntPoint

/-- A move is represented by three distinct indices: the point to move and the two points defining the move vector -/
structure Move where
  point_to_move : Fin 4
  vector_start : Fin 4
  vector_end : Fin 4
  distinct : point_to_move ≠ vector_start ∧ point_to_move ≠ vector_end ∧ vector_start ≠ vector_end

/-- Apply a move to a configuration -/
def apply_move (config : Configuration) (move : Move) : Configuration :=
  sorry

/-- Two points coincide if their coordinates are equal -/
def coincide (p1 p2 : IntPoint) : Prop :=
  p1.x = p2.x ∧ p1.y = p2.y

/-- Helper function to apply a list of moves -/
def apply_moves : Configuration → List Move → Configuration
  | config, [] => config
  | config, move :: moves => apply_moves (apply_move config move) moves

/-- The main theorem: it's always possible to make any two chosen points coincide -/
theorem align_two_points (config : Configuration) (i j : Fin 4) (h : i ≠ j) :
  ∃ (moves : List Move), coincide ((apply_moves config moves).points i) ((apply_moves config moves).points j) :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_align_two_points_l1011_101165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_g_l1011_101106

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℝ := Real.sqrt (x * (100 - x)) + Real.sqrt (x * (8 - x))

-- State the theorem
theorem max_value_of_g :
  ∃ (N : ℝ), N = g 8 ∧ N = Real.sqrt 736 ∧ ∀ x, 0 ≤ x ∧ x ≤ 8 → g x ≤ N :=
by
  -- We'll use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_g_l1011_101106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a1_value_l1011_101182

def seq (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → a (n + 2) = a n + a (n + 1)

theorem max_a1_value (a : ℕ → ℕ) (h1 : seq a) (h2 : ∀ n : ℕ, a n > 0) (h3 : a 5 = 60) :
  ∀ x : ℕ, (x > 0 ∧ seq (λ n => if n = 1 then x else a n)) → x ≤ 11 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a1_value_l1011_101182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1011_101114

theorem triangle_area : 
  ∀ (b c A B C : ℝ),
  b = 2 → B = π/6 → C = π/4 → 
  (1/2) * b * c * Real.sin A = Real.sqrt 3 + 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1011_101114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_angle_AXB_l1011_101128

noncomputable section

def O : ℝ × ℝ := (0, 0)
def P : ℝ × ℝ := (2, 1)
def A : ℝ × ℝ := (1, 7)
def B : ℝ × ℝ := (5, 1)

def X (l : ℝ) : ℝ × ℝ := (2*l, l)

def XA (l : ℝ) : ℝ × ℝ := (A.1 - (X l).1, A.2 - (X l).2)
def XB (l : ℝ) : ℝ × ℝ := (B.1 - (X l).1, B.2 - (X l).2)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

def angle_AXB (l : ℝ) : ℝ :=
  Real.arccos (dot_product (XA l) (XB l) / (Real.sqrt (dot_product (XA l) (XA l)) * Real.sqrt (dot_product (XB l) (XB l))))

theorem min_angle_AXB :
  ∃ l : ℝ, angle_AXB l = Real.arccos (-4 * Real.sqrt 17 / 17) ∧
  ∀ m : ℝ, dot_product (XA l) (XB l) ≤ dot_product (XA m) (XB m) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_angle_AXB_l1011_101128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_distance_line_l1011_101181

/-- A line passing through a point and intersecting positive x and y axes -/
structure IntersectingLine where
  slope : ℝ
  passesThrough : ℝ × ℝ := (1, 4)
  intersectsXAxis : ℝ → ℝ
  intersectsYAxis : ℝ → ℝ

/-- The sum of distances from origin to intersection points -/
def distanceSum (l : IntersectingLine) : ℝ :=
  abs (l.intersectsXAxis l.slope) + abs (l.intersectsYAxis l.slope)

/-- The theorem stating the equation of the line that minimizes the distance sum -/
theorem minimal_distance_line :
  ∃ (l : IntersectingLine),
    (∀ (l' : IntersectingLine), distanceSum l ≤ distanceSum l') ∧
    (λ (x y : ℝ) => 2 * x + y - 6 = 0) = (λ (x y : ℝ) => y - 4 = l.slope * (x - 1)) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_distance_line_l1011_101181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_cube_divisible_by_pq2r4_l1011_101117

theorem smallest_cube_divisible_by_pq2r4 (p q r : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r) 
  (hpq : p ≠ q) (hpr : p ≠ r) (hqr : q ≠ r) : 
  let pqr2_cube := (p * q * r^2)^3
  ∀ (m : ℕ), m^3 > 1 ∧ p * q^2 * r^4 ∣ m^3 → m^3 ≥ pqr2_cube := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_cube_divisible_by_pq2r4_l1011_101117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_equivalence_l1011_101111

-- Define the original equation
def original_equation (x : ℝ) : Prop :=
  2 * Real.sqrt x + 2 * x^(-(1/2 : ℝ)) = 5

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop :=
  4 * x^2 - 17 * x + 4 = 0

-- Theorem stating that the roots of both equations are the same
theorem roots_equivalence :
  ∀ x : ℝ, x > 0 → (original_equation x ↔ quadratic_equation x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_equivalence_l1011_101111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_path_length_of_specific_prism_l1011_101199

/-- Represents a rectangular prism with given dimensions -/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the path length of the center point on the top face when the prism is rolled -/
noncomputable def pathLengthOfCenterPoint (prism : RectangularPrism) : ℝ :=
  2 * Real.pi * prism.width

/-- Theorem: The path length of the center point on the top face of a 1x1x2 cm prism
    when rolled over its 1 cm edge until it returns to the top is 2π cm -/
theorem path_length_of_specific_prism :
  let prism : RectangularPrism := ⟨2, 1, 1⟩
  pathLengthOfCenterPoint prism = 2 * Real.pi :=
by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_path_length_of_specific_prism_l1011_101199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_triangle_area_l1011_101194

-- Define the vectors a and b
noncomputable def a (x : ℝ) : ℝ × ℝ := (2 * Real.sin x, Real.cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos x, 2 * Real.cos x)

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2 - 1

-- Define the monotonically increasing intervals
def is_monotone_increasing (f : ℝ → ℝ) (k : ℤ) : Prop :=
  ∀ x y, x ∈ Set.Icc (k * Real.pi - Real.pi / 3) (k * Real.pi + Real.pi / 6) →
         y ∈ Set.Icc (k * Real.pi - Real.pi / 3) (k * Real.pi + Real.pi / 6) →
         x ≤ y → f x ≤ f y

-- Theorem for monotonically increasing intervals
theorem f_monotone_increasing (k : ℤ) :
  is_monotone_increasing f k :=
sorry

-- Define a triangle
structure Triangle :=
  (A B C : ℝ)
  (a b c : ℝ)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b)

-- Theorem for the area of triangle ABC
theorem triangle_area (ABC : Triangle)
  (h_f : f ABC.B = 1)
  (h_b : ABC.b = Real.sqrt 3)
  (h_c : ABC.c = 2) :
  (1 / 2) * ABC.a * ABC.c * Real.sin ABC.B = Real.sqrt 3 / 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_triangle_area_l1011_101194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_and_ceiling_of_3_7_l1011_101159

noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

noncomputable def ceiling (x : ℝ) : ℤ :=
  -Int.floor (-x)

theorem floor_and_ceiling_of_3_7 :
  floor 3.7 = 3 ∧ ceiling 3.7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_and_ceiling_of_3_7_l1011_101159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_university_application_combinations_l1011_101172

theorem university_application_combinations (n m k : ℕ) (h1 : n = 7) (h2 : m = 5) (h3 : k = 2) : 
  (Nat.choose (n - 1) k) * (Nat.choose (n - k) (m - k)) = 150 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_university_application_combinations_l1011_101172
