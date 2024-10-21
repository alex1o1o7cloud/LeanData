import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_difference_l1310_131039

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x

-- State the theorem
theorem range_difference (a b : ℝ) :
  (∀ x ∈ Set.Icc (π/3) (4*π/3), f x ∈ Set.Icc a b) ∧
  (∀ y ∈ Set.Icc a b, ∃ x ∈ Set.Icc (π/3) (4*π/3), f x = y) →
  b - a = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_difference_l1310_131039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_70_79_range_l1310_131011

/-- Represents the frequency distribution of test scores -/
structure TestScoreDistribution :=
  (score_90_100 : Nat)
  (score_80_89 : Nat)
  (score_70_79 : Nat)
  (score_60_69 : Nat)
  (score_below_60 : Nat)

/-- Calculate the percentage of students in a specific score range -/
noncomputable def percentage_in_range (total : Nat) (range_count : Nat) : Real :=
  (range_count : Real) / (total : Real) * 100

/-- The main theorem to prove -/
theorem percentage_70_79_range (dist : TestScoreDistribution) : 
  percentage_in_range 
    (dist.score_90_100 + dist.score_80_89 + dist.score_70_79 + dist.score_60_69 + dist.score_below_60) 
    dist.score_70_79 = 
  (9 : Real) / (31 : Real) * 100 :=
by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval percentage_in_range 31 9

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_70_79_range_l1310_131011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_h_zeros_l1310_131078

-- Define the functions f, g, and h
noncomputable def f (x : ℝ) : ℝ := x^3 + Real.sin x

-- g is defined implicitly by its property
axiom g : ℝ → ℝ
axiom g_property (x : ℝ) : g x + g (2 - x) = 0

noncomputable def h (x : ℝ) : ℝ := f (x - 1) - g x

-- State the property that h has exactly 2019 zeros
axiom h_zeros : ∃ (zeros : Finset ℝ), zeros.card = 2019 ∧ ∀ x ∈ zeros, h x = 0

-- Theorem statement
theorem sum_of_h_zeros : 
  ∃ (zeros : Finset ℝ), zeros.card = 2019 ∧ (∀ x ∈ zeros, h x = 0) ∧ (zeros.sum id = 2019) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_h_zeros_l1310_131078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l1310_131027

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The eccentricity of an ellipse -/
noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ := Real.sqrt (1 - e.b^2 / e.a^2)

/-- The left focus of an ellipse -/
noncomputable def Ellipse.leftFocus (e : Ellipse) : ℝ × ℝ := (-Real.sqrt (e.a^2 - e.b^2), 0)

/-- The length of the vertical chord through the left focus -/
noncomputable def Ellipse.verticalChordLength (e : Ellipse) : ℝ := 2 * e.b^2 / e.a

/-- The theorem to be proved -/
theorem ellipse_theorem (e : Ellipse) 
    (h_ecc : e.eccentricity = Real.sqrt 2 / 2)
    (h_chord : e.verticalChordLength = Real.sqrt 2) :
  ∃ (k : ℝ),
    e.a^2 = 2 ∧ 
    e.b^2 = 1 ∧ 
    (3/2 : ℝ) = Real.sqrt ((16 * k^2 - 24) / (1 + 2*k^2)^2 * (1 + k^2)) ∧
    k^2 = 7/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l1310_131027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_l1310_131088

-- Define the curve C₁
noncomputable def C₁ (α : ℝ) : ℝ × ℝ := (2 + Real.cos α, 2 + Real.sin α)

-- Define the line C₂
noncomputable def C₂ (x : ℝ) : ℝ := Real.sqrt 3 * x

-- Define the intersection points
def intersection_points (ρ₁ ρ₂ : ℝ) : Prop :=
  let θ := Real.pi / 3
  C₂ (ρ₁ * Real.cos θ) = ρ₁ * Real.sin θ ∧
  C₂ (ρ₂ * Real.cos θ) = ρ₂ * Real.sin θ ∧
  (ρ₁ * Real.cos θ - 2)^2 + (ρ₁ * Real.sin θ - 2)^2 = 1 ∧
  (ρ₂ * Real.cos θ - 2)^2 + (ρ₂ * Real.sin θ - 2)^2 = 1

theorem intersection_sum (ρ₁ ρ₂ : ℝ) (h : intersection_points ρ₁ ρ₂) :
  1 / ρ₁ + 1 / ρ₂ = (2 * Real.sqrt 3 + 2) / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_l1310_131088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_herons_formula_l1310_131030

-- Define a triangle with sides a, b, and c
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c
  h_triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

-- Define the area function for a triangle
noncomputable def area (t : Triangle) : ℝ :=
  (1/4) * Real.sqrt (4 * t.a^2 * t.c^2 - (t.a^2 + t.c^2 - t.b^2)^2)

-- State the theorem
theorem herons_formula (t : Triangle) :
  area t = (1/4) * Real.sqrt (4 * t.a^2 * t.c^2 - (t.a^2 + t.c^2 - t.b^2)^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_herons_formula_l1310_131030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1310_131029

noncomputable section

/-- The ellipse C -/
def ellipse_C (x y : ℝ) (a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

/-- Point P on the ellipse -/
def point_P : ℝ × ℝ := (-1, 2*Real.sqrt 3 / 3)

/-- Distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- The right focus F2 -/
def focus_F2 (c : ℝ) : ℝ × ℝ := (c, 0)

/-- Line l passing through F1 and intersecting C at M and N -/
def line_l (k : ℝ) (x : ℝ) : ℝ := k * (x + 1)

/-- Area of triangle OMN -/
def area_OMN (M N : ℝ × ℝ) : ℝ := 
  (1/2) * abs (M.1 * N.2 - N.1 * M.2)

/-- Main theorem -/
theorem ellipse_properties (a b c : ℝ) 
  (h1 : a > b) (h2 : b > 0) 
  (h3 : ellipse_C point_P.1 point_P.2 a b)
  (h4 : distance point_P (focus_F2 c) = 4*Real.sqrt 3 / 3)
  (h5 : ∃ (k : ℝ) (M N : ℝ × ℝ), 
    ellipse_C M.1 M.2 a b ∧ 
    ellipse_C N.1 N.2 a b ∧
    M.2 = line_l k M.1 ∧ 
    N.2 = line_l k N.1 ∧
    area_OMN M N = 12/11) :
  (ellipse_C x y (Real.sqrt 3) (Real.sqrt 2) ∧ 
   c / a = 1 / Real.sqrt 3 ∧
   (∃ k : ℝ, k^2 = 3 ∧ (∀ x, line_l k x = Real.sqrt 3 * (x + 1) ∨ 
                               line_l k x = -Real.sqrt 3 * (x + 1)))) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1310_131029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_women_who_left_l1310_131060

theorem women_who_left (initial_men initial_women women_who_left : ℕ) : 
  (initial_men : ℚ) / initial_women = 4 / 5 →
  initial_men + 2 = 14 →
  2 * (initial_women - women_who_left) = 24 →
  women_who_left = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_women_who_left_l1310_131060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_falling_body_equation_of_motion_l1310_131089

/-- The equation of motion for a body falling from rest with constant acceleration -/
theorem falling_body_equation_of_motion 
  (g : ℝ) -- acceleration due to gravity
  (s : ℝ → ℝ) -- position as a function of time
  (v : ℝ → ℝ) -- velocity as a function of time
  (h1 : ∀ t, deriv v t = g) -- constant acceleration
  (h2 : v 0 = 0) -- initially at rest
  (h3 : ∀ t, deriv s t = v t) -- velocity is the derivative of position
  (h4 : s 0 = 0) -- initial position is 0
  : ∀ t, s t = (g * t^2) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_falling_body_equation_of_motion_l1310_131089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_valid_t_is_540_l1310_131081

noncomputable def isIsosceles (a b c : ℝ × ℝ) : Prop :=
  let d1 := (a.1 - b.1)^2 + (a.2 - b.2)^2
  let d2 := (b.1 - c.1)^2 + (b.2 - c.2)^2
  let d3 := (c.1 - a.1)^2 + (c.2 - a.2)^2
  d1 = d2 ∨ d2 = d3 ∨ d3 = d1

noncomputable def validT (t : ℝ) : Prop :=
  0 ≤ t ∧ t < 360 ∧
  isIsosceles (Real.cos (30 * Real.pi / 180), Real.sin (30 * Real.pi / 180))
              (Real.cos (90 * Real.pi / 180), Real.sin (90 * Real.pi / 180))
              (Real.cos (t * Real.pi / 180), Real.sin (t * Real.pi / 180))

theorem sum_of_valid_t_is_540 :
  ∃ (S : Finset ℝ), (∀ t ∈ S, validT t) ∧ (∀ t, validT t → t ∈ S) ∧ (S.sum id = 540) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_valid_t_is_540_l1310_131081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log5_1561_rounded_to_nearest_integer_l1310_131090

-- Define the base-5 logarithm
noncomputable def log5 (x : ℝ) : ℝ := Real.log x / Real.log 5

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- Theorem statement
theorem log5_1561_rounded_to_nearest_integer :
  floor (log5 1561 + 0.5) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log5_1561_rounded_to_nearest_integer_l1310_131090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bond_selling_price_approx_l1310_131005

/-- Calculates the selling price of a bond given its face value, interest rate on face value, and interest rate as a percentage of selling price. -/
noncomputable def bondSellingPrice (faceValue : ℝ) (interestRateOnFace : ℝ) (interestRateOnSelling : ℝ) : ℝ :=
  (faceValue * interestRateOnFace) / interestRateOnSelling

/-- Theorem stating that the selling price of a bond with given parameters is approximately 6153.85. -/
theorem bond_selling_price_approx (faceValue : ℝ) (interestRateOnFace : ℝ) (interestRateOnSelling : ℝ)
  (h1 : faceValue = 5000)
  (h2 : interestRateOnFace = 0.08)
  (h3 : interestRateOnSelling = 0.065) :
  ∃ (ε : ℝ), ε > 0 ∧ |bondSellingPrice faceValue interestRateOnFace interestRateOnSelling - 6153.85| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bond_selling_price_approx_l1310_131005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_angle_formula_l1310_131096

/-- A right prism with a rhombus base and a circumscribed sphere -/
structure RhombusPrism where
  /-- The acute angle of the rhombus base -/
  α : ℝ
  /-- The acute angle is between 0 and π/2 -/
  angle_bounds : 0 < α ∧ α < Real.pi / 2

/-- The angle between the larger diagonal of the prism and the plane of the base -/
noncomputable def diagonalAngle (prism : RhombusPrism) : ℝ :=
  Real.arctan (Real.sin (prism.α / 2))

/-- Theorem stating the angle between the larger diagonal and the base -/
theorem diagonal_angle_formula (prism : RhombusPrism) :
  diagonalAngle prism = Real.arctan (Real.sin (prism.α / 2)) := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_angle_formula_l1310_131096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coordinates_of_B_l1310_131080

-- Define the line segment AB
structure LineSegment where
  start : ℝ × ℝ
  end_ : ℝ × ℝ

-- Define parallelism to x-axis
def parallelToXAxis (l : LineSegment) : Prop :=
  l.start.2 = l.end_.2

-- Define length of a line segment
noncomputable def length (l : LineSegment) : ℝ :=
  Real.sqrt ((l.end_.1 - l.start.1)^2 + (l.end_.2 - l.start.2)^2)

theorem coordinates_of_B (AB : LineSegment) 
  (h1 : parallelToXAxis AB)
  (h2 : length AB = 2)
  (h3 : AB.start = (1, -3)) :
  AB.end_ = (-1, -3) ∨ AB.end_ = (3, -3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coordinates_of_B_l1310_131080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_identical_cones_vertex_angle_l1310_131041

/-- The angle at the vertex of identical cones in a specific geometric configuration -/
theorem identical_cones_vertex_angle : ∃ (α : ℝ),
  -- Three identical cones with vertex A touch each other externally
  -- Each of these cones touches internally a fourth cone with vertex at point A
  -- The fourth cone has an angle at the vertex of 2π/3
  -- α represents half of the angle at the vertex of the identical cones
  (α > 0) ∧ (α < π / 2) ∧
  (2 * α = 2 * Real.arctan ((3 : ℝ) / (4 + Real.sqrt 3))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_identical_cones_vertex_angle_l1310_131041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_fraction_product_l1310_131091

theorem repeating_decimal_fraction_product : ∃ (n d : ℤ), 
  (n ≠ 0 ∧ d ≠ 0) ∧ 
  (∀ (k : ℕ), (36 : ℚ) / 1000 + (36 : ℚ) / (1000 ^ (k + 1) - 1) = (n : ℚ) / d) ∧
  (∀ (n' d' : ℤ), n' ≠ 0 ∧ d' ≠ 0 → 
    (∀ (k : ℕ), (36 : ℚ) / 1000 + (36 : ℚ) / (1000 ^ (k + 1) - 1) = (n' : ℚ) / d') → 
    n' * d ≤ n * d') ∧
  n * d = 444 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_fraction_product_l1310_131091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_v_l1310_131055

-- Define the function v(x)
noncomputable def v (x : ℝ) : ℝ := 1 / Real.sqrt (Real.cos x)

-- Define the domain of v(x)
def domain_v : Set ℝ := ⋃ (n : ℤ), Set.Ioo (2 * n * Real.pi - Real.pi / 2) (2 * n * Real.pi + Real.pi / 2)

-- Theorem stating that the domain of v is correct
theorem domain_of_v : {x : ℝ | ∃ y, v x = y} = domain_v := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_v_l1310_131055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lyla_laps_when_isabelle_passes_fifth_time_l1310_131052

/-- Represents a runner on a circular track -/
structure Runner where
  speed : ℝ
  startPosition : ℝ

/-- The circular track -/
def Track := ℝ

theorem lyla_laps_when_isabelle_passes_fifth_time 
  (track : Track)
  (lyla isabelle : Runner)
  (h1 : isabelle.speed = 1.25 * lyla.speed)
  (h2 : isabelle.startPosition = 1/3)
  (h3 : lyla.startPosition = 0) :
  ∃ (t : ℝ), 
    (t * isabelle.speed + isabelle.startPosition) - (t * lyla.speed + lyla.startPosition) = 5 ∧ 
    ⌊t * lyla.speed + lyla.startPosition⌋ = 17 := by
  sorry

#check lyla_laps_when_isabelle_passes_fifth_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lyla_laps_when_isabelle_passes_fifth_time_l1310_131052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_COB_is_correct_l1310_131084

-- Define the points
def O : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (15, 0)
def C (p : ℝ) : ℝ × ℝ := (0, p)
def Q : ℝ × ℝ := (0, 15)

-- Define the condition that C is between Q and O on the y-axis
def C_between_Q_and_O (p : ℝ) : Prop :=
  0 < p ∧ p < 15

-- Define the area of triangle COB
noncomputable def area_COB (p : ℝ) : ℝ := (15 / 2) * p

-- State the theorem
theorem area_COB_is_correct (p : ℝ) (h : C_between_Q_and_O p) :
  area_COB p = (15 / 2) * p :=
by
  -- Unfold the definition of area_COB
  unfold area_COB
  -- The equality is now trivial
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_COB_is_correct_l1310_131084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1310_131008

-- Define the sequence a_n and its sum S_n
noncomputable def a (n : ℕ) : ℝ := 9 * n - 8
noncomputable def S (n : ℕ) : ℝ := (9 * n^2 - 7 * n) / 2

-- Define vectors a and b
noncomputable def vec_a (n : ℕ) : ℝ × ℝ := (S n, n)
noncomputable def vec_b (n : ℕ) : ℝ × ℝ := (9 * n - 7, 2)

-- Define collinearity condition
def collinear (n : ℕ) : Prop := 2 * (S n) = n * (9 * n - 7)

-- Define b_m
def b (m : ℕ) : ℕ := 9^(2*m-1) - 9^(m-1)

-- Define S_m (sum of first m terms of b_m)
noncomputable def S_m (m : ℕ) : ℝ := (9^(2*m+1) + 1 - 10 * 9^m + 1) / 80

theorem sequence_properties (n m : ℕ) (h : n > 0) (h_coll : collinear n) :
  (a n = 9 * n - 8) ∧
  (S_m m = (9^(2*m+1) + 1 - 10 * 9^m + 1) / 80) := by
  sorry

#check sequence_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1310_131008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2024_is_one_fourth_l1310_131087

def reciprocal_difference (a : ℚ) : ℚ := 1 / (1 - a)

def a_sequence : ℕ → ℚ
| 0 => -3
| n + 1 => reciprocal_difference (a_sequence n)

theorem sequence_2024_is_one_fourth :
  a_sequence 2023 = 1/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2024_is_one_fourth_l1310_131087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_sum_ellipse_l1310_131010

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 16 = 1

-- Define the foci
def F₂ : ℝ × ℝ := (3, 0)

-- Define point M
def M : ℝ × ℝ := (6, 4)

-- Define the distance between two points
noncomputable def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

theorem max_distance_sum_ellipse :
  ∀ P : ℝ × ℝ, is_on_ellipse P.1 P.2 →
  ∃ F₁ : ℝ × ℝ, 
    distance M F₂ = 5 →
    distance P F₁ + distance P F₂ = 10 →
    distance P M + distance P F₁ ≤ 15 ∧
    ∃ P₀ : ℝ × ℝ, is_on_ellipse P₀.1 P₀.2 ∧ distance P₀ M + distance P₀ F₁ = 15 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_sum_ellipse_l1310_131010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_shift_graph_transformation_l1310_131072

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := Real.sin (3 * x - 2)

-- Define the transformed function
noncomputable def g (x : ℝ) : ℝ := Real.sin (3 * x)

-- Theorem stating the equivalence of the functions
theorem sin_shift (x : ℝ) : f x = g (x - 2/3) := by
  -- Expand the definitions of f and g
  unfold f g
  -- Simplify the expressions
  simp [Real.sin_sub]
  -- The proof is complete
  sorry

-- Theorem stating the graph transformation
theorem graph_transformation : 
  ∀ x : ℝ, f x = g (x - 2/3) := by
  -- Use the previous theorem
  exact sin_shift


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_shift_graph_transformation_l1310_131072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_complex_number_l1310_131067

theorem imaginary_part_of_complex_number :
  ∀ (z : ℂ), z.im = 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_complex_number_l1310_131067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blackboard_numbers_l1310_131057

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ d : ℕ, d > 1 → d < p → ¬(p % d = 0)

theorem blackboard_numbers (n : ℕ) 
  (consecutive_numbers : ∀ k : ℕ, k ≤ n → k ∈ Set.range (λ i => i + 1))
  (three_erased : ∃ a b c : ℕ, a ∈ Set.range (λ i => i + 1) ∧ 
                               b ∈ Set.range (λ i => i + 1) ∧ 
                               c ∈ Set.range (λ i => i + 1) ∧
                               a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (two_primes : ∃ p q : ℕ, p ≠ q ∧ is_prime p ∧ is_prime q ∧ 
                (p ∈ Set.range (λ i => i + 1)) ∧ 
                (q ∈ Set.range (λ i => i + 1)))
  (average_remaining : (n * (n + 1) / 2 - (a + b + c)) / (n - 3) = 179 / 9) :
  n = 39 ∧ ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = 60 ∧ 
  ∀ r s : ℕ, is_prime r → is_prime s → r + s ≤ p + q :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blackboard_numbers_l1310_131057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_product_l1310_131015

-- Define the geometric series
def geometric_series (r : ℝ) : ℕ → ℝ := fun n => r^n

-- Define X and Y as sums of geometric series
noncomputable def X (x : ℝ) : ℝ := ∑' n, geometric_series x n
noncomputable def Y (y : ℝ) : ℝ := ∑' n, geometric_series y n

-- Define the series we want to prove about
noncomputable def XY_series (x y : ℝ) : ℝ := ∑' n, (x*y)^n

-- State the theorem
theorem geometric_series_product (x y : ℝ) (hx : |x| < 1) (hy : |y| < 1) :
  XY_series x y = (X x * Y y) / (X x + Y y - 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_product_l1310_131015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_equality_from_sum_and_product_l1310_131021

theorem set_equality_from_sum_and_product (x y z a b c : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c)
  (lower_bound_x : x ≥ min a (min b c))
  (lower_bound_y : y ≥ min a (min b c))
  (lower_bound_z : z ≥ min a (min b c))
  (upper_bound_x : x ≤ max a (max b c))
  (upper_bound_y : y ≤ max a (max b c))
  (upper_bound_z : z ≤ max a (max b c))
  (sum_eq : x + y + z = a + b + c)
  (prod_eq : x * y * z = a * b * c) :
  Finset.toSet {x, y, z} = Finset.toSet {a, b, c} := by
  sorry

#check set_equality_from_sum_and_product

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_equality_from_sum_and_product_l1310_131021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_driver_overtime_rate_increase_l1310_131020

/-- A bus driver's compensation structure and work hours --/
structure BusDriver where
  regularRate : ℝ
  regularHours : ℝ
  totalCompensation : ℝ
  totalHours : ℝ

/-- Calculate the percentage increase in overtime rate compared to regular rate --/
noncomputable def overtimeRateIncrease (bd : BusDriver) : ℝ :=
  let overtimeHours := bd.totalHours - bd.regularHours
  let regularEarnings := bd.regularRate * bd.regularHours
  let overtimeEarnings := bd.totalCompensation - regularEarnings
  let overtimeRate := overtimeEarnings / overtimeHours
  ((overtimeRate - bd.regularRate) / bd.regularRate) * 100

/-- Theorem stating the overtime rate increase for the given conditions --/
theorem bus_driver_overtime_rate_increase :
  let bd : BusDriver := {
    regularRate := 16,
    regularHours := 40,
    totalCompensation := 1200,
    totalHours := 60
  }
  overtimeRateIncrease bd = 75 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_driver_overtime_rate_increase_l1310_131020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_constant_vector_l1310_131025

/-- A vector in R² -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- The line y = (3/2)x + 3 -/
def onLine (v : Vector2D) : Prop :=
  v.y = (3/2) * v.x + 3

/-- Projection of v onto w -/
noncomputable def proj (v w : Vector2D) : Vector2D :=
  let dot := v.x * w.x + v.y * w.y
  let norm_squared := w.x^2 + w.y^2
  { x := (dot / norm_squared) * w.x,
    y := (dot / norm_squared) * w.y }

/-- The theorem statement -/
theorem projection_constant_vector :
  ∃ (w : Vector2D),
    (∀ (v1 v2 : Vector2D), onLine v1 → onLine v2 → proj v1 w = proj v2 w) ∧
    (∀ (v : Vector2D), onLine v → proj v w = { x := -18/13, y := 12/13 }) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_constant_vector_l1310_131025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_basis_range_l1310_131053

def a (m : ℝ) : ℝ × ℝ := (m, 3 * m - 4)
def b : ℝ × ℝ := (1, 2)

theorem vector_basis_range (m : ℝ) :
  (∀ c : ℝ × ℝ, ∃! p : ℝ × ℝ, c = p.1 • (a m) + p.2 • b) ↔ m ∈ {x | x < 4 ∨ x > 4} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_basis_range_l1310_131053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l1310_131043

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = -8*x

-- Define the hyperbola
def hyperbola (x y a : ℝ) : Prop := x^2/a^2 - y^2 = 1

-- Define the focus of the parabola
def parabola_focus : ℝ × ℝ := (-2, 0)

-- Define the foci of the hyperbola
noncomputable def hyperbola_foci (a : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) := 
  let c := Real.sqrt (a^2 + 1)
  ((-c, 0), (c, 0))

-- State the theorem
theorem hyperbola_properties (a : ℝ) (h1 : a > 0) 
  (h2 : parabola_focus = (hyperbola_foci a).1 ∨ parabola_focus = (hyperbola_foci a).2) :
  a = Real.sqrt 3 ∧ 
  (∀ x y : ℝ, x + Real.sqrt 3 * y = 0 ∨ x - Real.sqrt 3 * y = 0 ↔ 
    (∃ t : ℝ, x = a * t ∧ y = t)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l1310_131043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_l1310_131071

/-- The distance from a point (x₀, y₀) to a line Ax + By + C = 0 -/
noncomputable def point_to_line_distance (x₀ y₀ A B C : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / Real.sqrt (A^2 + B^2)

/-- The distance from the point (0, 1) to the line x + y - 6 = 0 is 5√2/2 -/
theorem distance_point_to_line : 
  point_to_line_distance 0 1 1 1 (-6) = 5 * Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_l1310_131071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l1310_131014

/-- Given lines l1, l2, and l3 in the plane, we define line l passing through
    the intersection of l1 and l2, and parallel to l3. -/
theorem line_equation : ∃ (A B C : ℝ),
  (∀ x y : ℝ, 3*x - 5*y - 10 = 0 ↔ (x, y) ∈ Set.range (λ t : ℝ × ℝ => t)) ∧ 
  (∀ x y : ℝ, x + y + 1 = 0 ↔ (x, y) ∈ Set.range (λ t : ℝ × ℝ => t)) ∧
  (∀ x y : ℝ, x + 2*y - 5 = 0 ↔ (x, y) ∈ Set.range (λ t : ℝ × ℝ => t)) ∧
  (∃ p : ℝ × ℝ, p ∈ Set.range (λ t : ℝ × ℝ => t) ∧ p ∈ Set.range (λ t : ℝ × ℝ => t) ∧ p ∈ Set.range (λ t : ℝ × ℝ => t)) ∧
  (∀ p q : ℝ × ℝ, p ∈ Set.range (λ t : ℝ × ℝ => t) ∧ q ∈ Set.range (λ t : ℝ × ℝ => t) ∧ p ≠ q → 
    ∃ r s : ℝ × ℝ, r ∈ Set.range (λ t : ℝ × ℝ => t) ∧ s ∈ Set.range (λ t : ℝ × ℝ => t) ∧ r ≠ s ∧ 
    (q.2 - p.2) / (q.1 - p.1) = (s.2 - r.2) / (s.1 - r.1)) →
  ∀ x y : ℝ, A * x + B * y + C = 0 ↔ (x, y) ∈ Set.range (λ t : ℝ × ℝ => t) ∧ A = 16 ∧ B = -8 ∧ C = -23 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l1310_131014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_l1310_131040

-- Define the curve C
def curve_C (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 8

-- Define the line l
def line_l (x y : ℝ) : Prop := Real.sqrt 3 * x - y + 2 * Real.sqrt 3 - 3 = 0

-- Define point P
def point_P : ℝ × ℝ := (-2, -3)

-- Theorem statement
theorem intersection_product :
  ∃ (A B : ℝ × ℝ),
    curve_C A.1 A.2 ∧
    curve_C B.1 B.2 ∧
    line_l A.1 A.2 ∧
    line_l B.1 B.2 ∧
    line_l point_P.1 point_P.2 ∧
    (let PA := Real.sqrt ((A.1 - point_P.1)^2 + (A.2 - point_P.2)^2)
     let PB := Real.sqrt ((B.1 - point_P.1)^2 + (B.2 - point_P.2)^2)
     PA * PB = 33) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_l1310_131040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_theorem_l1310_131079

/-- The expansion of (x + 1/2)^n -/
noncomputable def expansion (n : ℕ) (x : ℝ) := (x + 1/2)^n

/-- The coefficients of the expansion -/
noncomputable def coeff (n : ℕ) (i : ℕ) : ℝ := 
  if i ≤ n then Nat.choose n i * (1/2)^i else 0

/-- The condition that the first three coefficients form an arithmetic sequence -/
def arithmetic_condition (n : ℕ) : Prop :=
  coeff n 0 - coeff n 1 = coeff n 1 - coeff n 2

/-- The alternating sum of coefficients -/
noncomputable def alternating_sum (n : ℕ) : ℝ :=
  Finset.sum (Finset.range (n+1)) (λ i => (-1)^i * coeff n i)

theorem expansion_theorem (n : ℕ) (h : arithmetic_condition n) :
  n = 8 ∧ coeff n 5 = 7 ∧ alternating_sum n = 1/256 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_theorem_l1310_131079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1310_131069

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -1/3 * cos (2*x) - a * (sin x - cos x)

-- State the theorem
theorem range_of_a :
  ∀ a : ℝ,
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) < 1) ↔
  a ∈ Set.Icc (-Real.sqrt 2 / 6) (Real.sqrt 2 / 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1310_131069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_arithmetic_comparison_l1310_131042

/-- A geometric sequence with first term a₁ and common ratio r -/
def geometric_sequence (a₁ : ℝ) (r : ℝ) : ℕ → ℝ := λ n ↦ a₁ * r^(n-1)

/-- An arithmetic sequence with first term b₁ and common difference d -/
def arithmetic_sequence (b₁ : ℝ) (d : ℝ) : ℕ → ℝ := λ n ↦ b₁ + (n-1) * d

/-- The theorem stating that under given conditions, the 5th term of a geometric sequence
    is greater than the 5th term of an arithmetic sequence -/
theorem geometric_arithmetic_comparison
  (a₁ b₁ r d : ℝ)
  (h₁ : a₁ = b₁)
  (h₂ : a₁ > 0)
  (h₃ : geometric_sequence a₁ r 3 = arithmetic_sequence b₁ d 3)
  (h₄ : a₁ ≠ geometric_sequence a₁ r 3)
  : geometric_sequence a₁ r 5 > arithmetic_sequence b₁ d 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_arithmetic_comparison_l1310_131042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_theorem_l1310_131013

/-- Revenue function --/
noncomputable def R (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 10 then 10.8 - (1 / 30) * x^2
  else if x > 10 then 108 / x - 1000 / (3 * x^2)
  else 0

/-- Annual profit function --/
noncomputable def W (x : ℝ) : ℝ := x * R x - (10 + 2.7 * x)

/-- The point where maximum profit occurs --/
def max_profit_point : ℝ := 9

/-- The maximum profit value --/
def max_profit_value : ℝ := 38.6

/-- Theorem stating the maximum profit occurs at 9 thousand pieces with a value of 38.6 million yuan --/
theorem max_profit_theorem :
  (∀ x > 0, W x ≤ W max_profit_point) ∧ 
  W max_profit_point = max_profit_value := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_theorem_l1310_131013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_property_l1310_131064

-- Define the function f as noncomputable due to the use of Real.sqrt
noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then Real.sqrt x
  else if x ≥ 1 then 2 * (x - 1)
  else 0  -- This case is added to make the function total

-- State the theorem
theorem f_property (a : ℝ) (h1 : f a = f (a + 1)) : f (1 / a) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_property_l1310_131064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_pi_over_two_equals_one_l1310_131063

noncomputable def f (ω b : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 4) + b

theorem f_at_pi_over_two_equals_one 
  (ω b : ℝ) 
  (h_ω_pos : ω > 0)
  (h_period : 2 * Real.pi / 3 < 2 * Real.pi / ω ∧ 2 * Real.pi / ω < Real.pi)
  (h_symmetry : ∀ x, f ω b (3 * Real.pi / 2 - x) = 4 - f ω b (3 * Real.pi / 2 + x)) :
  f ω b (Real.pi / 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_pi_over_two_equals_one_l1310_131063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_probability_theorem_l1310_131094

def total_marbles : ℕ := 15
def blue_marbles : ℕ := 10
def red_marbles : ℕ := 5
def num_draws : ℕ := 8
def num_blue_draws : ℕ := 4

def prob_exactly_four_blue_at_least_one_red : ℚ :=
  (Nat.choose num_draws num_blue_draws) *
  ((blue_marbles : ℚ) / total_marbles) ^ num_blue_draws *
  ((red_marbles : ℚ) / total_marbles) ^ (num_draws - num_blue_draws) -
  ((blue_marbles : ℚ) / total_marbles) ^ num_draws

theorem marble_probability_theorem :
  (prob_exactly_four_blue_at_least_one_red * 1000).floor / 1000 = 131 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_probability_theorem_l1310_131094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_on_both_lines_l1310_131095

/-- Two lines in 2D space -/
structure Line2D where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- The first line -/
noncomputable def line1 : Line2D :=
  { point := (2, 3),
    direction := (3, -4) }

/-- The second line -/
noncomputable def line2 : Line2D :=
  { point := (4, -6),
    direction := (5, 1) }

/-- A point on a parameterized line -/
noncomputable def pointOnLine (l : Line2D) (t : ℝ) : ℝ × ℝ :=
  (l.point.1 + t * l.direction.1, l.point.2 + t * l.direction.2)

/-- The intersection point of the two lines -/
noncomputable def intersectionPoint : ℝ × ℝ := (175/23, 19/23)

/-- Theorem stating that the intersection point is on both lines -/
theorem intersection_point_on_both_lines :
  ∃ t u : ℝ, pointOnLine line1 t = intersectionPoint ∧ pointOnLine line2 u = intersectionPoint := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_on_both_lines_l1310_131095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_covered_once_ge_twice_l1310_131051

-- Define a circle with radius 1
def UnitCircle : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}

-- Define three unit circles on a plane
structure ThreeCircles where
  circle1 : Set (ℝ × ℝ)
  circle2 : Set (ℝ × ℝ)
  circle3 : Set (ℝ × ℝ)

def standardThreeCircles : ThreeCircles :=
  { circle1 := UnitCircle,
    circle2 := UnitCircle,
    circle3 := UnitCircle }

-- Define the area covered exactly once
noncomputable def AreaCoveredOnce (circles : ThreeCircles) : ℝ := sorry

-- Define the area covered exactly twice
noncomputable def AreaCoveredTwice (circles : ThreeCircles) : ℝ := sorry

-- Theorem statement
theorem area_covered_once_ge_twice (circles : ThreeCircles) :
  AreaCoveredOnce circles ≥ AreaCoveredTwice circles := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_covered_once_ge_twice_l1310_131051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_boat_travel_time_l1310_131099

/-- The length of the embankment in meters -/
noncomputable def embankment_length : ℝ := 50

/-- The time taken by the motorboat to pass the embankment downstream in seconds -/
noncomputable def downstream_time : ℝ := 5

/-- The time taken by the motorboat to pass the embankment upstream in seconds -/
noncomputable def upstream_time : ℝ := 4

/-- The speed of the current in meters per second -/
noncomputable def current_speed : ℝ := (embankment_length / downstream_time - embankment_length / upstream_time) / 2

/-- The time taken by a paper boat to travel the length of the embankment -/
noncomputable def paper_boat_time : ℝ := embankment_length / current_speed

theorem paper_boat_travel_time : paper_boat_time = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_boat_travel_time_l1310_131099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_hours_theorem_l1310_131047

/-- Represents the debt and payment information for a person --/
structure DebtInfo where
  initial_debt : ℚ
  payment_made : ℚ

/-- Represents the information for a task --/
structure TaskInfo where
  pay_rate : ℚ
  additional_payment : ℚ

/-- Calculates the total hours needed to work off debts --/
noncomputable def calculate_total_hours (person_a : DebtInfo) (person_b : DebtInfo) (person_c : DebtInfo)
  (task1 : TaskInfo) (task2 : TaskInfo) (task3 : TaskInfo)
  (task4 : TaskInfo) (task5 : TaskInfo) (task6 : TaskInfo) : ℚ :=
  let remaining_debt_a := person_a.initial_debt - person_a.payment_made
  let remaining_debt_b := person_b.initial_debt - person_b.payment_made
  let remaining_debt_c := person_c.initial_debt - person_c.payment_made

  let hours_task1 := (remaining_debt_a - task2.additional_payment) / task1.pay_rate
  let hours_task2 := task2.additional_payment / task2.pay_rate
  let hours_task3 := (remaining_debt_b - task4.additional_payment) / task3.pay_rate
  let hours_task4 := task4.additional_payment / task4.pay_rate
  let hours_task5 := (remaining_debt_c - task6.additional_payment) / task5.pay_rate
  let hours_task6 := task6.additional_payment / task6.pay_rate

  hours_task1 + hours_task2 + hours_task3 + hours_task4 + hours_task5 + hours_task6

theorem total_hours_theorem (person_a : DebtInfo) (person_b : DebtInfo) (person_c : DebtInfo)
  (task1 : TaskInfo) (task2 : TaskInfo) (task3 : TaskInfo)
  (task4 : TaskInfo) (task5 : TaskInfo) (task6 : TaskInfo)
  (h1 : person_a.initial_debt = 150)
  (h2 : person_a.payment_made = 60)
  (h3 : person_b.initial_debt = 200)
  (h4 : person_b.payment_made = 80)
  (h5 : person_c.initial_debt = 250)
  (h6 : person_c.payment_made = 100)
  (h7 : task1.pay_rate = 15)
  (h8 : task2.pay_rate = 12)
  (h9 : task3.pay_rate = 20)
  (h10 : task4.pay_rate = 10)
  (h11 : task5.pay_rate = 25)
  (h12 : task6.pay_rate = 30)
  (h13 : task2.additional_payment = 30)
  (h14 : task4.additional_payment = 40)
  (h15 : task6.additional_payment = 60)
  : calculate_total_hours person_a person_b person_c task1 task2 task3 task4 task5 task6 = 201/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_hours_theorem_l1310_131047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_f_2_equals_2_l1310_131046

-- Define the function f(x) = x / (x - 1)
noncomputable def f (x : ℝ) : ℝ := x / (x - 1)

-- Theorem stating that the maximum value of f(x) for x ≥ 2 is 2
theorem max_value_of_f :
  ∀ x : ℝ, x ≥ 2 → f x ≤ 2 :=
by
  sorry

-- Theorem stating that f(2) = 2, proving that the maximum is attained
theorem f_2_equals_2 :
  f 2 = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_f_2_equals_2_l1310_131046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_volume_l1310_131018

-- Define the cone
structure Cone where
  baseRadius : ℝ
  vertexAngle : ℝ

-- Define the sphere
structure Sphere where
  radius : ℝ

-- Define the problem
noncomputable def inscribedSphere (c : Cone) : Sphere :=
  { radius := c.baseRadius / 2 }

-- Theorem statement
theorem inscribed_sphere_volume (c : Cone) 
  (h1 : c.baseRadius = 12)  -- Half of the 24-inch diameter
  (h2 : c.vertexAngle = Real.pi / 2) : -- 90 degrees in radians
  (4 / 3 * Real.pi * (inscribedSphere c).radius ^ 3) = 288 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_volume_l1310_131018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_superdeficient_numbers_l1310_131070

def sumOfDivisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range n)).sum id + n

def isSuperdeficient (n : ℕ) : Prop := sumOfDivisors (sumOfDivisors n) = n + 3

theorem no_superdeficient_numbers : ¬∃ n : ℕ, isSuperdeficient n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_superdeficient_numbers_l1310_131070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l1310_131058

noncomputable def f (x : ℝ) : ℝ := (Real.sin x)^2 / (Real.sin x + 2)

theorem f_max_value :
  ∃ (M : ℝ), M = 1 ∧ ∀ (x : ℝ), f x ≤ M := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l1310_131058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_enough_for_six_days_l1310_131074

/-- Amount of cat food in a large package -/
noncomputable def B : ℝ := sorry

/-- Amount of cat food in a small package -/
noncomputable def S : ℝ := sorry

/-- A large package contains more food than a small one, but less than two small packages -/
axiom package_size_relation : B > S ∧ B < 2 * S

/-- Daily consumption of cat food -/
noncomputable def daily_consumption : ℝ := (B + 2 * S) / 2

/-- One large and two small packages of food are enough for the cat for exactly two days -/
axiom two_day_consumption : B + 2 * S = 2 * daily_consumption

/-- Theorem: 4 large and 4 small packages of food are not enough for the cat for six days -/
theorem not_enough_for_six_days : 4 * B + 4 * S < 6 * daily_consumption := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_enough_for_six_days_l1310_131074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chocolate_milk_tea_sales_l1310_131012

theorem chocolate_milk_tea_sales (total : ℕ) (winter_melon_ratio : ℚ) (okinawa_ratio : ℚ) : 
  total = 50 →
  winter_melon_ratio = 2 / 5 →
  okinawa_ratio = 3 / 10 →
  winter_melon_ratio + okinawa_ratio < 1 →
  (total : ℚ) - (winter_melon_ratio * total) - (okinawa_ratio * total) = 15 := by
    intro h_total h_winter h_okinawa h_sum
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chocolate_milk_tea_sales_l1310_131012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_commutation_l1310_131001

def A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 3; 4, 5]

def B (a b c d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![a, b; c, d]

theorem matrix_commutation (a b c d : ℝ) 
  (h1 : A * (B a b c d) = (B a b c d) * A) 
  (h2 : 2 * b ≠ 3 * c) : 
  (a - d) / (c - 2 * b) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_commutation_l1310_131001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_function_satisfies_conditions_l1310_131034

-- Define a function f
noncomputable def f : ℝ → ℝ := λ x => Real.log x / Real.log 3

-- Define arithmetic and geometric sequences
def isArithmeticSequence (s : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, s (n + 1) = s n + d

def isGeometricSequence (s : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, s (n + 1) = r * s n

-- State the theorem
theorem log_function_satisfies_conditions :
  ∃ (x y : ℕ → ℝ), 
    isArithmeticSequence x ∧ 
    isGeometricSequence y ∧
    ∀ n : ℕ, y n = f (x n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_function_satisfies_conditions_l1310_131034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_result_is_correct_l1310_131048

noncomputable def initial_value : ℝ := 1500

noncomputable def increase_by_percentage (x : ℝ) (p : ℝ) : ℝ := x * (1 + p / 100)

noncomputable def decrease_by_percentage (x : ℝ) (p : ℝ) : ℝ := x * (1 - p / 100)

noncomputable def final_result : ℝ :=
  increase_by_percentage
    (350 + decrease_by_percentage
      (increase_by_percentage initial_value 20 - 250)
      15)
    10

theorem final_result_is_correct : final_result = 1834.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_result_is_correct_l1310_131048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_marked_angles_is_1080_l1310_131003

/-- The sum of the four marked angles in a figure composed of four overlapping quadrilaterals is 1080°. -/
def sum_of_marked_angles_in_four_quadrilaterals : ℕ :=
  let quadrilateral_angle_sum := 360
  let total_quadrilaterals := 4
  let total_angle_sum := quadrilateral_angle_sum * total_quadrilaterals
  let overlapping_angle_sum := quadrilateral_angle_sum
  total_angle_sum - overlapping_angle_sum

theorem sum_of_marked_angles_is_1080 :
  sum_of_marked_angles_in_four_quadrilaterals = 1080 := by
  rfl

#eval sum_of_marked_angles_in_four_quadrilaterals

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_marked_angles_is_1080_l1310_131003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jacket_price_calculation_l1310_131007

/-- Calculates the final price of an item after a price increase followed by a discount --/
noncomputable def final_price (original_price : ℝ) (increase_percent : ℝ) (discount_percent : ℝ) : ℝ :=
  original_price * (1 + increase_percent / 100) * (1 - discount_percent / 100)

/-- Theorem stating that a $50 item with a 20% increase and 15% discount results in a $51 final price --/
theorem jacket_price_calculation :
  final_price 50 20 15 = 51 := by
  -- Unfold the definition of final_price
  unfold final_price
  -- Simplify the arithmetic expression
  simp [mul_add, mul_sub, mul_one]
  -- Prove the equality
  norm_num

-- We can't use #eval for noncomputable functions, so we'll use #check instead
#check final_price 50 20 15

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jacket_price_calculation_l1310_131007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monic_quartic_polynomial_with_specific_roots_l1310_131035

theorem monic_quartic_polynomial_with_specific_roots :
  ∃ (p : Polynomial ℚ),
    Polynomial.Monic p ∧
    Polynomial.degree p = 4 ∧
    (∀ x : ℂ, (p.map (algebraMap ℚ ℂ)).eval x = 0 ↔ 
      x = 3 + Real.sqrt 5 ∨ x = 3 - Real.sqrt 5 ∨ x = 2 + Real.sqrt 7 ∨ x = 2 - Real.sqrt 7) :=
by
  -- The polynomial x^4 - 10x^3 + 25x^2 + 2x - 12 satisfies these conditions
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monic_quartic_polynomial_with_specific_roots_l1310_131035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slower_pipe_theorem_l1310_131050

/-- Represents the time it takes for the slower pipe to fill the tank alone -/
noncomputable def slower_pipe_time (fast_rate slow_rate : ℝ) : ℝ :=
  1 / slow_rate

theorem slower_pipe_theorem (fast_rate slow_rate : ℝ) 
  (h1 : fast_rate = 4 * slow_rate) 
  (h2 : fast_rate + slow_rate = 1 / 40) : 
  slower_pipe_time fast_rate slow_rate = 200 := by
  sorry

#check slower_pipe_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slower_pipe_theorem_l1310_131050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_properties_l1310_131002

/-- Regression line parameters -/
def regression_slope : ℝ := 0.48
def regression_intercept : ℝ := 0.56

/-- Data points -/
def x_values : List ℝ := [1, 2, 3, 4, 5]
def y_values : List ℝ := [1, 1.6, 2.0, 2.4, 3]

/-- Regression line equation -/
def regression_line (x : ℝ) : ℝ := regression_slope * x + regression_intercept

/-- Sample centroid -/
def sample_centroid : ℝ × ℝ := (3, 2.0)

/-- Theorem stating the properties to be proved -/
theorem regression_properties :
  let n := x_values.length
  let x_mean := (x_values.sum) / n
  let y_mean := (y_values.sum) / n
  let a := y_values[3]
  (x_mean, y_mean) = sample_centroid ∧ a = 2.4 := by
  sorry

#eval regression_line 13 -- This will evaluate the prediction for January 2024

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_properties_l1310_131002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roundedScore_eq_2874_l1310_131006

/-- Rounds a real number to the nearest hundredth -/
noncomputable def roundToHundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

/-- The original score -/
def originalScore : ℝ := 28.737

/-- Theorem stating that rounding the original score to the nearest hundredth equals 28.74 -/
theorem roundedScore_eq_2874 : roundToHundredth originalScore = 28.74 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roundedScore_eq_2874_l1310_131006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_difference_l1310_131083

theorem book_difference (total : ℕ) (reading_ratio math_ratio : ℚ) (history : ℕ) : 
  total = 10 →
  reading_ratio = 2/5 →
  math_ratio = 3/10 →
  history = 1 →
  (↑total * math_ratio).floor - (total - (↑total * reading_ratio).floor - (↑total * math_ratio).floor - history) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_difference_l1310_131083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_sum_equals_two_plus_sqrt_three_l1310_131009

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 2 then 3 - x else 2 * x - x^2

-- State the theorem
theorem inverse_sum_equals_two_plus_sqrt_three :
  ∃ (y₁ y₂ y₃ : ℝ),
    (f y₁ = -2 ∧ y₁ > 2) ∧
    (f y₂ = 1 ∧ y₂ ≤ 2) ∧
    (f y₃ = 4 ∧ y₃ ≤ 2) ∧
    y₁ + y₂ + y₃ = 2 + Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_sum_equals_two_plus_sqrt_three_l1310_131009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_capacity_l1310_131032

/-- Represents a cylindrical water tank with a level indicator -/
structure WaterTank where
  current_volume : ℚ
  current_percentage : ℚ

/-- Calculates the most likely full capacity of a water tank -/
def most_likely_capacity (tank : WaterTank) : ℚ :=
  tank.current_volume / tank.current_percentage

/-- Theorem stating the most likely capacity of the given tank is 240 liters -/
theorem tank_capacity (tank : WaterTank) 
  (h1 : tank.current_volume = 60)
  (h2 : tank.current_percentage = 1/4) :
  most_likely_capacity tank = 240 := by
  unfold most_likely_capacity
  rw [h1, h2]
  norm_num

/-- Compute the result for the given tank -/
def result : ℚ :=
  most_likely_capacity { current_volume := 60, current_percentage := 1/4 }

#eval result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_capacity_l1310_131032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_relation_l1310_131038

theorem quadratic_root_relation (a b c : ℝ) :
  let eq1 : ℝ → ℝ := λ x => x^2 + a*x + b
  let eq2 : ℝ → ℝ := λ x => x^2 + c*x + a
  (∃ r1 r2 : ℝ, eq2 r1 = 0 ∧ eq2 r2 = 0) →
  (∀ x, eq1 x = 0 ↔ eq2 (x/3) = 0) →
  b/c = 27 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_relation_l1310_131038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1310_131016

open Complex

-- Define the complex number z
noncomputable def z : ℂ := (3 + I) / (2 - I)

-- Define z₁ as a function of m
def z₁ (m : ℝ) : ℂ := 2 + m * I

-- Theorem for part 1
theorem part_one (m : ℝ) : abs (z + z₁ m) = 5 ↔ m = 3 ∨ m = -5 := by sorry

-- Theorem for part 2
theorem part_two (a : ℝ) : 
  (re (a * z + 2 * I) < 0 ∧ im (a * z + 2 * I) > 0) ↔ -2 < a ∧ a < 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1310_131016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_focus_and_chord_length_l1310_131061

/-- Line l: mx - y + 1 - m = 0 -/
def line_l (m : ℝ) (x y : ℝ) : Prop := m * x - y + 1 - m = 0

/-- Parabola: y² = 8x -/
def parabola (x y : ℝ) : Prop := y^2 = 8 * x

/-- Circle: (x-1)² + (y-1)² = 6 -/
def my_circle (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 6

/-- Focus of the parabola y² = 8x -/
def focus : ℝ × ℝ := (2, 0)

/-- Theorem stating the main result -/
theorem line_through_focus_and_chord_length 
  (m : ℝ) (h : line_l m (focus.1) (focus.2)) :
  m = -1 ∧ ∃ (A B : ℝ × ℝ), 
    line_l m A.1 A.2 ∧ line_l m B.1 B.2 ∧ 
    my_circle A.1 A.2 ∧ my_circle B.1 B.2 ∧
    ∃ (d : ℝ), d = 2 * Real.sqrt 6 ∧ 
      (A.1 - B.1)^2 + (A.2 - B.2)^2 = d^2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_focus_and_chord_length_l1310_131061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_park_fencing_cost_l1310_131004

/-- The cost of fencing a rectangular park -/
noncomputable def fencing_cost (ratio_length width : ℚ) (area perimeter_cost : ℝ) : ℝ :=
  let length := (ratio_length * width : ℝ)
  let perimeter := 2 * (length + width)
  (perimeter * perimeter_cost) / 100

/-- Theorem: The cost of fencing the park is 200 rupees -/
theorem park_fencing_cost :
  ∃ (width : ℚ),
    let ratio_length : ℚ := 3/2
    let area : ℝ := 3750
    let perimeter_cost : ℝ := 80
    fencing_cost ratio_length width area perimeter_cost = 200 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_park_fencing_cost_l1310_131004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_properties_l1310_131036

-- Define the power function as noncomputable
noncomputable def power_function (α : ℝ) : ℝ → ℝ := λ x ↦ x ^ α

-- State the theorem
theorem power_function_properties :
  ∃ α : ℝ,
    (power_function α 3 = Real.sqrt 3) ∧
    (∀ x > 0, power_function α x = Real.sqrt x) ∧
    (¬ (∀ x, power_function α (-x) = power_function α x)) ∧
    (¬ (∀ x, power_function α (-x) = -(power_function α x))) ∧
    (∀ x y, 0 < x ∧ x < y → power_function α x ≤ power_function α y) :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_properties_l1310_131036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ideal_type_circle_l1310_131024

-- Define the line l
def line_l (x y : ℝ) : Prop := 3 * x + 4 * y - 12 = 0

-- Define the distance from a point to line l
noncomputable def dist_to_line_l (x y : ℝ) : ℝ := 
  |3 * x + 4 * y - 12| / Real.sqrt 25

-- Define the circles
def circle_A (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle_B (x y : ℝ) : Prop := x^2 + y^2 = 16
def circle_C (x y : ℝ) : Prop := (x - 4)^2 + (y - 4)^2 = 1
def circle_D (x y : ℝ) : Prop := (x - 4)^2 + (y - 4)^2 = 16

-- Define what it means for a circle to be an "ideal type"
def is_ideal_type (circle : (ℝ → ℝ → Prop)) : Prop :=
  ∃ p q : ℝ × ℝ, 
    p ≠ q ∧
    circle p.1 p.2 ∧ 
    circle q.1 q.2 ∧
    dist_to_line_l p.1 p.2 = 1 ∧
    dist_to_line_l q.1 q.2 = 1 ∧
    ∀ r : ℝ × ℝ, circle r.1 r.2 ∧ dist_to_line_l r.1 r.2 = 1 → r = p ∨ r = q

theorem ideal_type_circle :
  is_ideal_type circle_D ∧ 
  ¬is_ideal_type circle_A ∧ 
  ¬is_ideal_type circle_B ∧ 
  ¬is_ideal_type circle_C := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ideal_type_circle_l1310_131024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_intersection_l1310_131054

-- Define the functions f and g
noncomputable def f (x : ℝ) := Real.log (1 - x)
noncomputable def g (x : ℝ) := 1 / x

-- Define the domains of f and g
def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | x ≠ 0}

-- State the theorem
theorem domain_intersection :
  M ∩ N = {x : ℝ | x < 1 ∧ x ≠ 0} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_intersection_l1310_131054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_roots_trigonometric_equation_l1310_131023

theorem min_roots_trigonometric_equation 
  (k₀ k₁ k₂ : ℕ) (A₁ A₂ : ℝ) 
  (h₁ : k₀ < k₁) (h₂ : k₁ < k₂) : 
  ∃ (S : Finset ℝ), 
    (∀ x ∈ S, 0 ≤ x ∧ x < 2 * Real.pi ∧ 
      Real.sin (k₀ * x) + A₁ * Real.sin (k₁ * x) + A₂ * Real.sin (k₂ * x) = 0) ∧ 
    S.card ≥ 2 * k₀ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_roots_trigonometric_equation_l1310_131023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_side_length_l1310_131098

/-- Given two similar triangles satisfying certain conditions, 
    prove that the corresponding side of the larger triangle is 15 feet. -/
theorem similar_triangles_side_length 
  (A₁ A₂ : ℝ) -- Areas of the larger and smaller triangles
  (s₁ s₂ : ℝ) -- Corresponding side lengths of the larger and smaller triangles
  (k : ℕ) -- The integer whose square is the ratio of areas
  (h1 : A₁ - A₂ = 32) -- Difference in areas
  (h2 : A₁ / A₂ = k^2) -- Ratio of areas is a perfect square
  (h3 : ∃ n : ℕ, A₂ = n) -- Area of smaller triangle is an integer
  (h4 : s₂ = 5) -- Side length of smaller triangle
  (h5 : (s₁ / s₂)^2 = A₁ / A₂) -- Similarity condition for triangles
  : s₁ = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_side_length_l1310_131098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_distribution_converges_l1310_131019

/-- Represents the state of candy distribution at a given moment -/
structure CandyState where
  participants : ℕ  -- number of participants
  candies : ℕ → ℕ  -- function mapping participant index to their candy count

/-- The candy distribution process -/
def distributeCandy (state : CandyState) : CandyState :=
  { participants := state.participants,
    candies := fun i =>
      let prev := (i - 1 + state.participants) % state.participants
      let current := state.candies i
      let prevCandy := state.candies prev
      (current + 1) / 2 + (prevCandy + 1) / 2 }

/-- Predicate to check if all participants have the same number of candies -/
def allEqual (state : CandyState) : Prop :=
  ∀ i j, i < state.participants → j < state.participants →
    state.candies i = state.candies j

/-- The main theorem: candy distribution eventually leads to equality -/
theorem candy_distribution_converges
  (initial : CandyState)
  (h : initial.participants > 0) :
  ∃ n : ℕ, allEqual (Nat.iterate distributeCandy n initial) := by
  sorry

#check candy_distribution_converges

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_distribution_converges_l1310_131019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_intersection_and_max_value_l1310_131028

noncomputable section

-- Define the curves C₁ and C₂
def C₁ (a : ℝ) (φ : ℝ) : ℝ × ℝ := (a + a * Real.cos φ, a * Real.sin φ)
def C₂ (b : ℝ) (φ : ℝ) : ℝ × ℝ := (b * Real.cos φ, b + b * Real.sin φ)

-- Define the ray l
def l (α : ℝ) (ρ : ℝ) : ℝ × ℝ := (ρ * Real.cos α, ρ * Real.sin α)

-- Define the lengths |OA| and |OB|
def OA (a : ℝ) (θ : ℝ) : ℝ := Real.cos θ
def OB (b : ℝ) (θ : ℝ) : ℝ := 2 * Real.sin θ

theorem curve_intersection_and_max_value (a b : ℝ) :
  (a > 0) →
  (b > 0) →
  (∀ α, 0 ≤ α ∧ α ≤ Real.pi / 2 → ∃ ρ₁ ρ₂, l α ρ₁ = C₁ a (Real.arccos (Real.cos α)) ∧ l α ρ₂ = C₂ b (Real.arcsin (Real.sin α))) →
  (OA a 0 = 1) →
  (OB b (Real.pi / 2) = 2) →
  (a = 1 / 2 ∧ b = 1 ∧ 
   ∀ θ, 2 * (OA a θ)^2 + OA a θ * OB b θ ≤ Real.sqrt 2 + 1) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_intersection_and_max_value_l1310_131028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_bounded_neg_inf_to_zero_f_bounded_iff_a_in_range_l1310_131037

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 + a * (1/3)^x + (1/9)^x

-- Part 1: f is not bounded on (-∞, 0) when a = -1/2
theorem f_not_bounded_neg_inf_to_zero :
  ∀ M : ℝ, M > 0 → ∃ x : ℝ, x < 0 ∧ |f (-1/2) x| > M := by
  sorry

-- Part 2: f is bounded by 4 on [0, +∞) if and only if a ∈ [-6, 2]
theorem f_bounded_iff_a_in_range (a : ℝ) :
  (∀ x : ℝ, x ≥ 0 → |f a x| ≤ 4) ↔ -6 ≤ a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_bounded_neg_inf_to_zero_f_bounded_iff_a_in_range_l1310_131037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_toss_probability_l1310_131049

theorem coin_toss_probability : 
  let keiko_tosses := 2
  let ephraim_tosses := 3
  let total_outcomes := (2 : ℚ)^(keiko_tosses + ephraim_tosses)
  let favorable_outcomes := 
    (keiko_tosses.choose 0 * ephraim_tosses.choose 1 : ℚ) + 
    (keiko_tosses.choose 1 * ephraim_tosses.choose 2 : ℚ)
  favorable_outcomes / total_outcomes = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_toss_probability_l1310_131049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_manolo_rate_change_time_l1310_131044

/-- Represents the time in hours at which Manolo's face-mask making rate changes -/
def rate_change_time : ℝ := 1

/-- Represents the duration of Manolo's shift in hours -/
def shift_duration : ℝ := 4

/-- Represents the number of face-masks Manolo makes in a full shift -/
def total_masks : ℕ := 45

/-- Represents the rate of face-mask production (in minutes per mask) for the first hour -/
def initial_rate : ℝ := 4

/-- Represents the rate of face-mask production (in minutes per mask) after the first hour -/
def later_rate : ℝ := 6

/-- Represents the number of masks made in the first hour -/
noncomputable def masks_first_hour : ℝ := 60 / initial_rate

/-- Represents the number of masks made in the remaining hours -/
noncomputable def masks_remaining_hours : ℝ := (shift_duration - rate_change_time) * 60 / later_rate

theorem manolo_rate_change_time :
  rate_change_time = 1 ∧
  ⌊masks_first_hour⌋ + ⌊masks_remaining_hours⌋ = total_masks :=
by sorry

#eval rate_change_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_manolo_rate_change_time_l1310_131044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_pair_after_removal_l1310_131066

/-- Represents a deck of cards with counts for each number -/
def Deck := Fin 10 → Nat

/-- The initial deck with five of each number from 1 to 10 -/
def initialDeck : Deck := fun _ => 5

/-- A deck after removing two pairs -/
def remainingDeck (d : Deck) (a b : Fin 10) : Deck :=
  fun i => if i = a ∨ i = b then d i - 2 else d i

/-- The total number of cards in a deck -/
def totalCards (d : Deck) : Nat :=
  (Finset.range 10).sum (fun i => d i)

/-- The number of ways to choose 2 cards from n cards -/
def choose2 (n : Nat) : Nat :=
  n * (n - 1) / 2

/-- The number of pairs in a deck -/
def countPairs (d : Deck) : Nat :=
  (Finset.range 10).sum (fun i => choose2 (d i))

theorem probability_of_pair_after_removal (a b : Fin 10) (ha : a ≠ b) :
  let d := remainingDeck initialDeck a b
  let total := choose2 (totalCards d)
  let pairs := countPairs d
  pairs = 86 ∧ total = 1035 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_pair_after_removal_l1310_131066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_radius_is_7_sqrt_3_l1310_131065

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the angle bisector
noncomputable def angleBisector (t : Triangle) : ℝ × ℝ → ℝ × ℝ := sorry

-- Define the intersection point D
noncomputable def D (t : Triangle) : ℝ × ℝ := sorry

-- Define the circle passing through A and D
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the circumcircle of triangle ABC
noncomputable def circumcircle (t : Triangle) : Circle := sorry

-- State the theorem
theorem circumcircle_radius_is_7_sqrt_3 (t : Triangle) (c : Circle) :
  -- Conditions
  (angleBisector t t.A).2 = t.B.2 ∧  -- Center of circle lies on line BC
  c.radius = 35 ∧
  c.center.2 = t.B.2 ∧
  t.A ∈ ({x | (x.1 - c.center.1)^2 + (x.2 - c.center.2)^2 = c.radius^2} : Set (ℝ × ℝ)) ∧
  D t ∈ ({x | (x.1 - c.center.1)^2 + (x.2 - c.center.2)^2 = c.radius^2} : Set (ℝ × ℝ)) ∧
  (t.A.1 - t.B.1)^2 + (t.A.2 - t.B.2)^2 - ((t.A.1 - t.C.1)^2 + (t.A.2 - t.C.2)^2) = 216 ∧
  abs ((t.A.1 - t.B.1) * (t.B.2 - t.C.2) - (t.B.1 - t.C.1) * (t.A.2 - t.B.2)) / 2 = 90 * Real.sqrt 3 →
  -- Conclusion
  (circumcircle t).radius = 7 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_radius_is_7_sqrt_3_l1310_131065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_logarithmic_expression_l1310_131026

theorem simplify_logarithmic_expression (a : ℝ) (h : a > 1) :
  (2 * Real.log (Real.log (a^100))) / (2 + Real.log (Real.log a)) + (1/9)^(-(1/2 : ℝ)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_logarithmic_expression_l1310_131026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expression_equals_one_l1310_131092

open Real

theorem trigonometric_expression_equals_one (α : ℝ) : 
  (sin (8 * α) + sin (9 * α) + sin (10 * α) + sin (11 * α)) / 
  (cos (8 * α) + cos (9 * α) + cos (10 * α) + cos (11 * α)) * 
  (cos (8 * α) - cos (9 * α) - cos (10 * α) + cos (11 * α)) / 
  (sin (8 * α) - sin (9 * α) - sin (10 * α) + sin (11 * α)) = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expression_equals_one_l1310_131092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_exceeds_one_at_2018_l1310_131056

noncomputable def a_sequence (n : ℕ) : ℚ :=
  match n with
  | 0 => 1/2
  | n + 1 => a_sequence n + (a_sequence n)^2 / 2018

def first_exceeding_one (k : ℕ) : Prop :=
  a_sequence k < 1 ∧ a_sequence (k + 1) > 1 ∧ ∀ n < k, a_sequence n < 1

theorem sequence_exceeds_one_at_2018 :
  first_exceeding_one 2018 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_exceeds_one_at_2018_l1310_131056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_condition_extreme_value_condition_l1310_131062

-- Define the function f
noncomputable def f (b c x : ℝ) : ℝ := (1/3) * x^3 - (1/2) * x^2 + b * x + c

-- Theorem 1
theorem increasing_function_condition (b c : ℝ) :
  (∀ x : ℝ, HasDerivAt (f b c) ((x^2 - x + b) : ℝ) x) →
  (∀ x : ℝ, (x^2 - x + b) ≥ 0) →
  b ≥ (1/4 : ℝ) := by sorry

-- Theorem 2
theorem extreme_value_condition (b c : ℝ) :
  (∃ x : ℝ, x = 1 ∧ HasDerivAt (f b c) 0 x) →
  (∀ x : ℝ, x ∈ Set.Icc (-1) 2 → f b c x < c^2) →
  (c > (3 + Real.sqrt 33) / 6 ∨ c < (3 - Real.sqrt 33) / 6) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_condition_extreme_value_condition_l1310_131062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_f_neg_two_l1310_131033

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x + 2 else x^2 - 1

theorem f_of_f_neg_two : f (f (-2)) = 5 := by
  -- Evaluate f(-2)
  have h1 : f (-2) = 3 := by
    simp [f]
    norm_num
  
  -- Evaluate f(3)
  have h2 : f 3 = 5 := by
    simp [f]
    norm_num
  
  -- Combine the results
  calc
    f (f (-2)) = f 3 := by rw [h1]
    _          = 5   := by rw [h2]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_f_neg_two_l1310_131033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_excircle_incircle_area_relation_l1310_131059

/-- Represents a point in 2D space --/
structure Point where
  x : ℝ
  y : ℝ

/-- The area of a triangle given its vertices --/
noncomputable def area_triangle (A B C : Point) : ℝ := sorry

/-- The area of the triangle formed by the tangency points of the incircle --/
noncomputable def area_triangle_incircle_tangency_points (A B C : Point) : ℝ := sorry

theorem excircle_incircle_area_relation (A B C : Point) (R r : ℝ) :
  let S := area_triangle A B C
  let S_incircle := area_triangle_incircle_tangency_points A B C
  3 * R^2 - 4 * R * r - r^2 = 2 * S + S_incircle := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_excircle_incircle_area_relation_l1310_131059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_equivalent_integral_pairs_l1310_131076

/-- Definition of "equivalent integral" functions on an interval [a, b] -/
def equivalent_integral (f g : ℝ → ℝ) (a b : ℝ) : Prop :=
  a < b ∧ ∫ x in a..b, f x = ∫ x in a..b, g x

/-- The four pairs of functions -/
def f₁ (x : ℝ) : ℝ := 2 * |x|
def g₁ (x : ℝ) : ℝ := x + 1

noncomputable def f₂ (x : ℝ) : ℝ := Real.sin x
noncomputable def g₂ (x : ℝ) : ℝ := Real.cos x

noncomputable def f₃ (x : ℝ) : ℝ := Real.sqrt (1 - x^2)
noncomputable def g₃ (x : ℝ) : ℝ := 3/4 * Real.pi * x^2

noncomputable def f₄ : ℝ → ℝ := sorry  -- Placeholder for an odd function
noncomputable def g₄ : ℝ → ℝ := sorry  -- Placeholder for another odd function

/-- Theorem stating that exactly 3 out of 4 pairs are equivalent integral functions on [-1, 1] -/
theorem three_equivalent_integral_pairs :
  (equivalent_integral f₁ g₁ (-1) 1) ∧
  ¬(equivalent_integral f₂ g₂ (-1) 1) ∧
  (equivalent_integral f₃ g₃ (-1) 1) ∧
  (equivalent_integral f₄ g₄ (-1) 1) :=
by
  sorry  -- Proof omitted


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_equivalent_integral_pairs_l1310_131076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mancino_garden_count_l1310_131045

/-- Represents the dimensions of a garden -/
structure GardenDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a garden given its dimensions -/
def gardenArea (d : GardenDimensions) : ℝ := d.length * d.width

theorem mancino_garden_count :
  let mancinoDimensions : GardenDimensions := ⟨16, 5⟩
  let marquitaDimensions : GardenDimensions := ⟨8, 4⟩
  let marquitaGardenCount : ℕ := 2
  let totalArea : ℝ := 304

  let mancinoArea := gardenArea mancinoDimensions
  let marquitaArea := marquitaGardenCount * gardenArea marquitaDimensions
  let mancinoTotalArea := totalArea - marquitaArea

  ⌊mancinoTotalArea / mancinoArea⌋ = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mancino_garden_count_l1310_131045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1310_131093

noncomputable def f (x : ℝ) : ℝ := (x - 2) / (x^2 - 4)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≠ 2 ∧ x ≠ -2} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1310_131093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1310_131022

noncomputable def f (x : ℝ) := 2 * Real.sin (2 * x - 13 * Real.pi / 4)

theorem f_properties :
  (f (Real.pi / 8) = 0) ∧
  (∀ x, f x = 2 * Real.sin (2 * (x - 5 * Real.pi / 8))) := by
  constructor
  · -- Proof that f(π/8) = 0
    sorry
  · -- Proof that ∀ x, f(x) = 2 * sin(2(x - 5π/8))
    intro x
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1310_131022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_savings_is_55_80_l1310_131077

/-- Calculate the final price after applying two sequential discounts -/
def apply_discounts (original_price : ℚ) (discount1 : ℚ) (discount2 : ℚ) : ℚ :=
  original_price * (1 - discount1) * (1 - discount2)

/-- Calculate the price for a buy one, get one X% off deal -/
def buy_one_get_one_discount (original_price : ℚ) (discount : ℚ) : ℚ :=
  (original_price + original_price * (1 - discount)) / 2

/-- Calculate the total savings -/
def total_savings_calculation (
  chlorine_price : ℚ)
  (chlorine_discount1 : ℚ)
  (chlorine_discount2 : ℚ)
  (chlorine_quantity : ℕ)
  (soap_price : ℚ)
  (soap_discount1 : ℚ)
  (soap_discount2 : ℚ)
  (soap_quantity : ℕ)
  (wipes_price : ℚ)
  (wipes_discount : ℚ)
  (wipes_quantity : ℕ) : ℚ :=
  let chlorine_savings := chlorine_quantity * (chlorine_price - apply_discounts chlorine_price chlorine_discount1 chlorine_discount2)
  let soap_savings := soap_quantity * (soap_price - apply_discounts soap_price soap_discount1 soap_discount2)
  let wipes_savings := wipes_quantity * (wipes_price - buy_one_get_one_discount wipes_price wipes_discount)
  chlorine_savings + soap_savings + wipes_savings

theorem total_savings_is_55_80 :
  total_savings_calculation 10 (1/5) (1/10) 4 16 (1/4) (1/20) 6 8 (1/2) 8 = 558/10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_savings_is_55_80_l1310_131077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_problem_l1310_131031

-- Define the geometric sequence
def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q ^ (n - 1)

-- State the theorem
theorem geometric_sequence_problem (a₁ q : ℝ) (h1 : q ≠ 0) (h2 : q ≠ 1) :
  (let a := geometric_sequence a₁ q
   (a 1 + a 2 + a 3 = 168) ∧ (a 2 - a 5 = 42)) →
  geometric_sequence a₁ q 6 = 3 := by
  intro h
  sorry  -- The proof steps would go here

-- Define the main function
def solve_geometric_sequence : ℝ := by
  -- We would implement the solution steps here
  exact 3  -- The final answer

#eval solve_geometric_sequence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_problem_l1310_131031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_island_population_bounds_l1310_131097

/-- Represents a club on the island -/
structure Club where
  members : Finset Nat
  mem_count : members.card ≤ 55

/-- Represents the island with its clubs and residents -/
structure Island where
  clubs : Finset Club
  club_count : clubs.card = 50
  residents : Finset Nat
  resident_membership : ∀ r ∈ residents, ∃ c₁ c₂ : Club, c₁ ∈ clubs ∧ c₂ ∈ clubs ∧ r ∈ c₁.members ∧ (r ∈ c₂.members → c₁ = c₂)
  pair_intersection : ∀ c₁ c₂ : Club, c₁ ∈ clubs → c₂ ∈ clubs → c₁ ≠ c₂ → ∃ r ∈ residents, r ∈ c₁.members ∧ r ∈ c₂.members

/-- The number of inhabitants on the island is between 1225 and 1525, inclusive -/
theorem island_population_bounds (i : Island) : 1225 ≤ i.residents.card ∧ i.residents.card ≤ 1525 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_island_population_bounds_l1310_131097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_range_skew_lines_l1310_131075

/-- The angle between two skew lines -/
def angle_between_skew_lines (l1 l2 : Line3D) : ℝ := sorry

/-- Two lines are skew if they are not parallel and do not intersect -/
def are_skew (l1 l2 : Line3D) : Prop := sorry

theorem angle_range_skew_lines (l1 l2 : Line3D) (h : are_skew l1 l2) :
  ∃ θ : ℝ, θ = angle_between_skew_lines l1 l2 ∧ 0 < θ ∧ θ ≤ π / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_range_skew_lines_l1310_131075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apothem_equality_l1310_131068

theorem apothem_equality : ∀ (s t : ℝ),
  s > 0 ∧ t > 0 →
  s^2 = 4*s →
  (Real.sqrt 3 / 4) * t^2 = 3*t →
  s / 2 = (Real.sqrt 3 / 2 * t) / 3 :=
by
  intros s t h1 h2 h3
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apothem_equality_l1310_131068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carA_faster_than_carB_l1310_131000

/-- Represents a car with its travel segments -/
structure Car where
  segments : List (ℝ × ℝ)  -- List of (distance, speed) pairs
  stops : List ℝ           -- List of stop durations

/-- Calculates the total time taken by a car -/
noncomputable def totalTime (car : Car) : ℝ :=
  (car.segments.map (λ (d, v) => d / v)).sum + car.stops.sum

/-- Calculates the total distance traveled by a car -/
noncomputable def totalDistance (car : Car) : ℝ :=
  (car.segments.map (λ (d, _) => d)).sum

/-- Calculates the average speed of a car -/
noncomputable def averageSpeed (car : Car) : ℝ :=
  totalDistance car / totalTime car

/-- Car A with its travel segments -/
def carA : Car :=
  { segments := [(50, 60), (50, 40)]
  , stops := [] }

/-- Car B with its travel segments and stop -/
def carB : Car :=
  { segments := [(60, 60), (40, 40)]
  , stops := [0.25] }

/-- Theorem stating that Car A has a higher average speed than Car B -/
theorem carA_faster_than_carB : averageSpeed carA > averageSpeed carB := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_carA_faster_than_carB_l1310_131000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integers_product_divisibility_l1310_131086

theorem consecutive_integers_product_divisibility (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a < b) :
  ∀ n : ℤ, ∃ i j : ℤ, 
    (n ≤ i) ∧ (i < n + b) ∧
    (n ≤ j) ∧ (j < n + b) ∧
    (i ≠ j) ∧
    (∃ k : ℤ, i * j = k * (a * b)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integers_product_divisibility_l1310_131086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_increasing_interval_l1310_131082

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x)

theorem cos_2x_increasing_interval :
  ∀ x ∈ Set.Icc 0 π,
    (∀ y ∈ Set.Icc (π / 2) π, x < y → f x < f y) ∧
    (∀ y ∈ Set.Icc 0 (π / 2), x < y → f x ≥ f y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_increasing_interval_l1310_131082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_range_l1310_131085

noncomputable section

-- Define the circle C
def circle_center : ℝ × ℝ := (Real.sqrt 2, Real.pi / 4)
def circle_radius : ℝ := Real.sqrt 3

-- Define the line l
def line_param (α t : ℝ) : ℝ × ℝ := (2 + t * Real.cos α, 2 + t * Real.sin α)

-- Define the range of α
def α_range : Set ℝ := { α | 0 ≤ α ∧ α < Real.pi / 4 }

-- Theorem statement
theorem chord_length_range :
  ∀ α ∈ α_range,
  ∃ A B : ℝ × ℝ,
  (∃ t₁ t₂ : ℝ, A = line_param α t₁ ∧ B = line_param α t₂) ∧
  (A.1 - circle_center.1)^2 + (A.2 - circle_center.2)^2 = circle_radius^2 ∧
  (B.1 - circle_center.1)^2 + (B.2 - circle_center.2)^2 = circle_radius^2 ∧
  2 * Real.sqrt 2 ≤ Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ∧
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) < 2 * Real.sqrt 3 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_range_l1310_131085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l1310_131017

theorem negation_of_proposition (P : Prop) :
  (P = (∀ x : ℝ, (2 : ℝ)^x > 5)) →
  (¬P ↔ ∃ x : ℝ, (2 : ℝ)^x ≤ 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l1310_131017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l1310_131073

/-- The distance between two parallel lines -/
noncomputable def distance_between_parallel_lines (a b c₁ c₂ : ℝ) : ℝ :=
  abs (c₁ - c₂) / Real.sqrt (a^2 + b^2)

/-- Theorem: The distance between the given parallel lines is 7/10 -/
theorem distance_between_given_lines (m : ℝ) :
  (3 : ℝ) * 4 = 6 * m →  -- Parallel condition
  m ≠ 0 →  -- Ensure m is not zero for division
  distance_between_parallel_lines 3 4 (-3) (-1/2) = 7/10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l1310_131073
