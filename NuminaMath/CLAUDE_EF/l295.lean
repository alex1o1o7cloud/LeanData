import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l295_29538

theorem trigonometric_equation_solution (x : ℝ) (k : ℤ) : 
  (x = -Real.arccos ((Real.sqrt 13 - 1) / 4) + 2 * k * Real.pi ∨
   x = -Real.arccos ((1 - Real.sqrt 13) / 4) + 2 * k * Real.pi) →
  (Real.cos (5 * x) - Real.cos (7 * x)) / (Real.sin (4 * x) + Real.sin (2 * x)) = 2 * |Real.sin (2 * x)| :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l295_29538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_midpoints_l295_29569

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point
structure Point where
  x : ℝ
  y : ℝ

-- Define a chord
structure Chord where
  start : Point
  finish : Point  -- Changed 'end' to 'finish' to avoid keyword conflict

-- Define the problem
theorem locus_of_midpoints (c : Circle) (m : Point) :
  (m.x, m.y) ≠ c.center →
  (∃ r : ℝ, (m.x - c.center.1)^2 + (m.y - c.center.2)^2 < r^2) →
  ∀ p : Point,
    (∃ ch : Chord, 
      (ch.start.x - c.center.1)^2 + (ch.start.y - c.center.2)^2 = c.radius^2 ∧
      (ch.finish.x - c.center.1)^2 + (ch.finish.y - c.center.2)^2 = c.radius^2 ∧
      (m.x = (ch.start.x + ch.finish.x) / 2 ∧ m.y = (ch.start.y + ch.finish.y) / 2) ∧
      (p.x = (ch.start.x + ch.finish.x) / 2 ∧ p.y = (ch.start.y + ch.finish.y) / 2)) →
    (p.x - c.center.1)^2 + (p.y - c.center.2)^2 = 
      ((m.x - c.center.1)^2 + (m.y - c.center.2)^2) / 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_midpoints_l295_29569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_centroid_to_hypotenuse_l295_29508

theorem distance_from_centroid_to_hypotenuse 
  (S : ℝ) (α : ℝ) (h_S_pos : S > 0) (h_α_pos : 0 < α) (h_α_lt_pi_2 : α < π / 2) :
  ∃ (d : ℝ), d = (1 / 3) * Real.sqrt (S * Real.sin (2 * α)) := by
  sorry

#check distance_from_centroid_to_hypotenuse

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_centroid_to_hypotenuse_l295_29508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l295_29595

def is_geometric (a : ℕ → ℚ) (q : ℚ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

def contains_four_consecutive_terms (b : ℕ → ℚ) : Prop :=
  ∃ k : ℕ, (b k ∈ ({-53, -23, 19, 37, 82} : Set ℚ)) ∧
            (b (k + 1) ∈ ({-53, -23, 19, 37, 82} : Set ℚ)) ∧
            (b (k + 2) ∈ ({-53, -23, 19, 37, 82} : Set ℚ)) ∧
            (b (k + 3) ∈ ({-53, -23, 19, 37, 82} : Set ℚ))

theorem geometric_sequence_ratio
  (a : ℕ → ℚ) (q : ℚ) (b : ℕ → ℚ)
  (h1 : is_geometric a q)
  (h2 : |q| > 1)
  (h3 : ∀ n : ℕ, b n = a n + 1)
  (h4 : contains_four_consecutive_terms b) :
  q = -3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l295_29595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_general_equation_C_polar_equation_C_distance_AB_l295_29559

-- Define the curve C
noncomputable def curve_C (φ : ℝ) : ℝ × ℝ :=
  (2 * Real.cos φ, 2 + 2 * Real.sin φ)

-- Define points A and B
noncomputable def point_A : ℝ × ℝ := (Real.sqrt 3, 3)
noncomputable def point_B : ℝ × ℝ := (-Real.sqrt 3, 1)

-- Theorem for the general equation of curve C
theorem general_equation_C : 
  ∀ (x y : ℝ), (∃ φ, curve_C φ = (x, y)) → x^2 + (y - 2)^2 = 4 := by sorry

-- Theorem for the polar equation of curve C
theorem polar_equation_C :
  ∀ (ρ θ : ℝ), (∃ φ, curve_C φ = (ρ * Real.cos θ, ρ * Real.sin θ)) → ρ = 4 * Real.sin θ := by sorry

-- Theorem for the distance between A and B
theorem distance_AB :
  Real.sqrt ((point_A.1 - point_B.1)^2 + (point_A.2 - point_B.2)^2) = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_general_equation_C_polar_equation_C_distance_AB_l295_29559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_value_l295_29536

noncomputable def f (x : ℝ) : ℝ := Real.cos x - Real.sin x

theorem max_a_value (a : ℝ) :
  (∀ x y, x ∈ Set.Icc (-a) a → y ∈ Set.Icc (-a) a → x < y → f x > f y) →
  a ≤ 3 * π / 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_value_l295_29536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_cos_4x_plus_pi_third_l295_29563

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.cos (4 * x + Real.pi / 3)

-- State the theorem
theorem min_positive_period_of_cos_4x_plus_pi_third :
  ∃ T : ℝ, T > 0 ∧ (∀ x : ℝ, f (x + T) = f x) ∧
  (∀ T' : ℝ, T' > 0 ∧ (∀ x : ℝ, f (x + T') = f x) → T ≤ T') ∧
  T = Real.pi / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_cos_4x_plus_pi_third_l295_29563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l295_29542

/-- Given an ellipse described by the equation 9x^2 + 16y^2 = 144, 
    the distance between its foci is 2√7. -/
theorem ellipse_foci_distance : 
  ∀ (x y : ℝ), 9 * x^2 + 16 * y^2 = 144 → 
  ∃ (f₁ f₂ : ℝ × ℝ), 
    (f₁ ∈ Set.prod (Set.Icc (-4) 4) (Set.Icc (-3) 3)) ∧ 
    (f₂ ∈ Set.prod (Set.Icc (-4) 4) (Set.Icc (-3) 3)) ∧
    dist f₁ f₂ = 2 * Real.sqrt 7 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l295_29542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_youngest_sibling_age_l295_29510

theorem youngest_sibling_age (siblings : ℕ) (age_differences : List ℝ) (average_age : ℝ) :
  siblings = 10 ∧
  age_differences = [1, 2, 7, 11, 15, 18, 21, 28, 33] ∧
  average_age = 45 →
  ∃ (youngest_age : ℝ),
    youngest_age = 31.4 ∧
    average_age * (siblings : ℝ) = youngest_age * (siblings : ℝ) + age_differences.sum :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_youngest_sibling_age_l295_29510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_coordinate_magnitude_l295_29516

def T₁ : Fin 3 → ℝ := ![0, 1, 0]
def T₂ : Fin 3 → ℝ := ![0, 0, 0]
def T₃ : Fin 3 → ℝ := ![1, 0, 0]

def T₀ (z : ℝ) : Fin 3 → ℝ := ![0, 0, z]

def X (x y : ℝ) : Fin 3 → ℝ := ![x, y, 0]

noncomputable def distance (p q : Fin 3 → ℝ) : ℝ :=
  Real.sqrt (((p 0 - q 0)^2 + (p 1 - q 1)^2 + (p 2 - q 2)^2))

noncomputable def area (p q r : Fin 3 → ℝ) : ℝ :=
  (1/2) * abs (p 0 * (q 1 * r 2 - r 1 * q 2) - p 1 * (q 0 * r 2 - r 0 * q 2) + p 2 * (q 0 * r 1 - r 0 * q 1))

def constant_quantity (z x y : ℝ) : Prop :=
  ∃ (k : ℝ),
    distance (X x y) (T₀ z) * area T₁ T₂ T₃ = k ∧
    distance (X x y) T₁ * area (T₀ z) T₂ T₃ = k ∧
    distance (X x y) T₂ * area (T₀ z) T₁ T₃ = k ∧
    distance (X x y) T₃ * area (T₀ z) T₁ T₂ = k

theorem z_coordinate_magnitude (z : ℝ) :
  (T₀ z ≠ T₂) →
  (∃ x y, constant_quantity z x y) →
  abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_coordinate_magnitude_l295_29516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l295_29512

-- Define the function f(x) = 3x|x|
def f (x : ℝ) : ℝ := 3 * x * abs x

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (f (1 - a) + f (2 * a) < 0) ↔ (a < -1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l295_29512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_sqrt_e_l295_29571

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then -x + 1 else Real.log x

theorem f_composition_sqrt_e : f (f (Real.sqrt (Real.exp 1))) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_sqrt_e_l295_29571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersection_points_l295_29534

-- Define the circle C with center O and radius r
def circleSet (O : ℝ × ℝ) (r : ℝ) : Set (ℝ × ℝ) := {P | (P.1 - O.1)^2 + (P.2 - O.2)^2 = r^2}

-- Define the distance between two points
noncomputable def distance (P Q : ℝ × ℝ) : ℝ := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Theorem statement
theorem max_intersection_points (O P : ℝ × ℝ) (r : ℝ) :
  distance O P = 8 →
  (∃ (n : ℕ), n ≤ 2 ∧
    (∀ (m : ℕ), 
      (∃ (S : Finset (ℝ × ℝ)), S.card = m ∧ 
        (∀ Q ∈ S, Q ∈ circleSet O r ∧ distance P Q = 4)) →
      m ≤ n)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersection_points_l295_29534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cylinder_height_l295_29570

/-- The height of a right circular cylinder inscribed in a hemisphere -/
theorem inscribed_cylinder_height :
  ∀ (r_cylinder r_hemisphere : ℝ),
  r_cylinder > 0 →
  r_hemisphere > 0 →
  r_cylinder < r_hemisphere →
  ∃ (h : ℝ),
  h > 0 ∧
  h = Real.sqrt (r_hemisphere ^ 2 - r_cylinder ^ 2) ∧
  (r_cylinder = 3 ∧ r_hemisphere = 7 → h = Real.sqrt 40) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cylinder_height_l295_29570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_condition_l295_29541

/-- A line passing through the origin with angle of inclination α -/
def line (α : Real) : Set (Real × Real) :=
  {(x, y) | y = Real.tan α * x}

/-- The circle with equation x^2 + (y-2)^2 = 1 -/
def circle_C : Set (Real × Real) :=
  {(x, y) | x^2 + (y-2)^2 = 1}

/-- A line is tangent to a circle if it intersects the circle at exactly one point -/
def is_tangent (l : Set (Real × Real)) (c : Set (Real × Real)) : Prop :=
  ∃! p, p ∈ l ∩ c

theorem tangent_condition (α : Real) :
  (α = π/3 → is_tangent (line α) circle_C) ∧
  (∃ β, β ≠ π/3 ∧ is_tangent (line β) circle_C) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_condition_l295_29541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l295_29515

-- Define the universal set U
def U : Set ℤ := {0, 1, 2, 3}

-- Define set A
def A : Set ℤ := {x | 0 ≤ x ∧ x ≤ 2}

-- Define set B
def B : Set ℤ := {x | 1 ≤ x ∧ x ≤ 3}

-- Define the difference of two sets
def set_difference (X Y : Set ℤ) : Set ℤ := {x | x ∈ X ∧ x ∉ Y}

theorem set_operations :
  (U \ A = {3}) ∧
  ((U \ B) ∩ A = {0}) ∧
  ((set_difference A B) ∪ (set_difference B A) = {0, 3}) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l295_29515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_difference_value_l295_29532

-- Define the function f as noncomputable due to its dependence on Real.sin
noncomputable def f (x φ : Real) : Real := Real.sin (2 * x + φ)

-- State the theorem
theorem sine_difference_value 
  (φ : Real) 
  (x₁ x₂ : Real) 
  (h_φ : 0 < φ ∧ φ < Real.pi)
  (h_x : 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < Real.pi)
  (h_f_bound : ∀ x, f x φ ≤ |f (Real.pi/6) φ|)
  (h_f_values : f x₁ φ = -3/5 ∧ f x₂ φ = -3/5) :
  Real.sin (x₂ - x₁) = 4/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_difference_value_l295_29532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_of_OF_l295_29531

-- Define the triangle ABC and point P
variable (a b c p : ℝ)

-- Define the conditions
variable (ha : a ≠ 0)
variable (hb : b ≠ 0)
variable (hc : c ≠ 0)
variable (hp : p ≠ 0)
variable (hap : p < a)  -- Ensure P is between O and A

-- Define the points
def A (a : ℝ) : ℝ × ℝ := (0, a)
def B (b : ℝ) : ℝ × ℝ := (b, 0)
def C (c : ℝ) : ℝ × ℝ := (c, 0)
def P (p : ℝ) : ℝ × ℝ := (0, p)
def O : ℝ × ℝ := (0, 0)

-- Define the equation of line OF
def line_OF (b c p a : ℝ) (x y : ℝ) : Prop :=
  (1/b - 1/c) * x - (1/p - 1/a) * y = 0

-- Theorem statement
theorem equation_of_OF :
  ∃ (E F : ℝ × ℝ),
    (∃ (t : ℝ), E = (t * (C c).1, t * (C c).2 + (1 - t) * (A a).2)) ∧  -- E is on AC
    (∃ (s : ℝ), F = (s * (B b).1, s * (B b).2 + (1 - s) * (A a).2)) ∧  -- F is on AB
    (∃ (u : ℝ), E = (u * (B b).1 + (1 - u) * (P p).1, u * (B b).2 + (1 - u) * (P p).2)) ∧  -- E is on BP
    (∃ (v : ℝ), F = (v * (C c).1 + (1 - v) * (P p).1, v * (C c).2 + (1 - v) * (P p).2)) ∧  -- F is on CP
    (∀ x y : ℝ, (1/b - 1/c) * x + (1/p - 1/a) * y = 0 ↔ 
      ∃ (w : ℝ), (x, y) = (w * E.1, w * E.2)) →  -- Equation of OE
    (∀ x y : ℝ, line_OF b c p a x y ↔ 
      ∃ (z : ℝ), (x, y) = (z * F.1, z * F.2)) :=  -- Equation of OF
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_of_OF_l295_29531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meryll_multiple_choice_questions_l295_29529

/-- The number of multiple-choice questions Meryll wants to write -/
def M : ℚ := sorry

/-- The number of problem-solving questions Meryll wants to write -/
def P : ℚ := sorry

/-- The fraction of multiple-choice questions already written -/
def mc_written : ℚ := 2/5

/-- The fraction of problem-solving questions already written -/
def ps_written : ℚ := 1/3

/-- The total number of questions left to write -/
def questions_left : ℚ := 31

/-- The total number of problem-solving questions -/
def total_ps : ℚ := 15

theorem meryll_multiple_choice_questions :
  (P = total_ps) →
  ((1 - mc_written) * M + (1 - ps_written) * P = questions_left) →
  (M = 35) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meryll_multiple_choice_questions_l295_29529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_special_angle_l295_29547

theorem sin_minus_cos_special_angle (α : ℝ) 
  (h1 : Real.tan α = Real.sqrt 3 / 3) 
  (h2 : π < α ∧ α < 3 * π / 2) : 
  Real.sin α - Real.cos α = -1/2 + Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_special_angle_l295_29547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_courtyard_paving_stones_count_l295_29552

/-- Represents the dimensions of a trapezoid-shaped courtyard -/
structure Courtyard where
  length : ℝ
  side1 : ℝ
  side2 : ℝ
  height : ℝ

/-- Represents the dimensions of a paving stone consisting of two rectangles -/
structure PavingStone where
  rect1_length : ℝ
  rect1_width : ℝ
  rect2_length : ℝ
  rect2_width : ℝ

/-- Calculates the area of a trapezoid -/
noncomputable def trapezoidArea (c : Courtyard) : ℝ :=
  (c.side1 + c.side2) * c.height / 2

/-- Calculates the area of a paving stone -/
noncomputable def pavingStoneArea (p : PavingStone) : ℝ :=
  p.rect1_length * p.rect1_width + p.rect2_length * p.rect2_width

/-- Calculates the number of paving stones required to cover the courtyard -/
noncomputable def pavingStonesRequired (c : Courtyard) (p : PavingStone) : ℕ :=
  Int.natAbs (Int.ceil ((trapezoidArea c) / (pavingStoneArea p)))

theorem courtyard_paving_stones_count :
  let c : Courtyard := { length := 60, side1 := 16.5, side2 := 25, height := 12 }
  let p : PavingStone := { rect1_length := 2.5, rect1_width := 2, rect2_length := 1.5, rect2_width := 3 }
  pavingStonesRequired c p = 27 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_courtyard_paving_stones_count_l295_29552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_almond_walnut_ratio_l295_29535

/-- Given a mixture of almonds and walnuts, where the total weight is 140 pounds
    and the weight of almonds is 116.67 pounds, prove that the ratio of almonds
    to walnuts is 5:1. -/
theorem almond_walnut_ratio :
  ∀ (total_weight almond_weight : ℝ),
  total_weight = 140 →
  almond_weight = 116.67 →
  (almond_weight / (total_weight - almond_weight)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_almond_walnut_ratio_l295_29535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_ratio_theorem_l295_29591

theorem log_ratio_theorem (m n : ℝ) (hm : Real.log 2 = m) (hn : Real.log 3 = n) :
  (Real.log 12) / (Real.log 15) = (2*m + n) / (1 - m + n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_ratio_theorem_l295_29591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scrabble_champions_non_bearded_with_hair_l295_29555

def scrabble_champions (total_champions : ℕ) (women_percentage : ℚ) (bearded_percentage : ℚ) (bearded_bald_percentage : ℚ) (non_bearded_bald_percentage : ℚ) : ℕ := 
  let male_champions := total_champions - (women_percentage * ↑total_champions).floor
  let non_bearded_champions := male_champions - (bearded_percentage * ↑male_champions).floor
  (non_bearded_champions - (non_bearded_bald_percentage * ↑non_bearded_champions).floor).toNat

theorem scrabble_champions_non_bearded_with_hair :
  scrabble_champions 25 (3/5) (2/5) (3/5) (3/10) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scrabble_champions_non_bearded_with_hair_l295_29555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_perpendicular_lines_infinite_parallel_lines_l295_29584

-- Define a 3D space
structure Space3D where
  -- Add necessary fields here

-- Define a point in 3D space
structure Point3D where
  -- Add necessary fields here

-- Define a line in 3D space
structure Line3D where
  -- Add necessary fields here

-- Define a plane in 3D space
structure Plane3D where
  -- Add necessary fields here

-- Define perpendicularity between a line and another line
def perpendicular (l1 l2 : Line3D) : Prop :=
  sorry

-- Define parallelism between a line and a plane
def parallel (l : Line3D) (p : Plane3D) : Prop :=
  sorry

-- Define a set of lines passing through a point
def lines_through_point (p : Point3D) : Set Line3D :=
  sorry

-- Define a set of lines perpendicular to a given line
def perpendicular_lines (l : Line3D) : Set Line3D :=
  sorry

-- Define a set of lines parallel to a given plane
def parallel_lines (p : Plane3D) : Set Line3D :=
  sorry

-- Define a membership relation between Point3D and Plane3D
instance : Membership Point3D Plane3D where
  mem := fun _ _ => sorry

-- Theorem 1: Infinitely many lines pass through a point and are perpendicular to a given line
theorem infinite_perpendicular_lines 
  (s : Space3D) (p : Point3D) (l : Line3D) : 
  Set.Infinite (lines_through_point p ∩ perpendicular_lines l) :=
by sorry

-- Theorem 2: Infinitely many lines pass through a point outside a plane and are parallel to that plane
theorem infinite_parallel_lines 
  (s : Space3D) (p : Point3D) (pl : Plane3D) 
  (h : p ∉ pl) : 
  Set.Infinite (lines_through_point p ∩ parallel_lines pl) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_perpendicular_lines_infinite_parallel_lines_l295_29584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_equals_set_l295_29568

def U : Set ℤ := {x | -1 ≤ x ∧ x ≤ 5}
def A : Set ℤ := {1, 2, 5}
def B : Set ℤ := {x | 0 ≤ x ∧ x < 4}

theorem intersection_complement_equals_set : B ∩ (U \ A) = {0, 3} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_equals_set_l295_29568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_per_box_for_fine_arts_collection_l295_29582

/-- The cost per box for packaging a fine arts collection --/
theorem cost_per_box_for_fine_arts_collection : ℝ := by
  let box_volume : ℝ := 20 * 20 * 15
  let total_volume : ℝ := 3.06 * 1000000
  let total_cost : ℝ := 459
  let num_boxes : ℝ := total_volume / box_volume
  have h : total_cost / num_boxes = 0.90 := by sorry
  exact 0.90


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_per_box_for_fine_arts_collection_l295_29582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_even_l295_29520

noncomputable def g (x : ℝ) : ℝ := 4 / (3 * x^4 - 7)

theorem g_is_even : ∀ x : ℝ, g (-x) = g x := by
  intro x
  unfold g
  -- The rest of the proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_even_l295_29520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_and_distance_l295_29545

/-- Hyperbola C defined by x²/6 - y²/3 = 1 -/
def C : Set (ℝ × ℝ) :=
  {p | p.1^2 / 6 - p.2^2 / 3 = 1}

/-- The right focus of hyperbola C -/
def right_focus : ℝ × ℝ := (3, 0)

/-- The asymptote of hyperbola C -/
def asymptote (x y : ℝ) : Prop :=
  x + Real.sqrt 2 * y = 0 ∨ x - Real.sqrt 2 * y = 0

/-- The distance from a point to a line -/
noncomputable def distance_point_to_line (p : ℝ × ℝ) (l : ℝ → ℝ → Prop) : ℝ :=
  Real.sqrt 3

theorem hyperbola_focus_and_distance :
  right_focus ∈ C ∧
  distance_point_to_line right_focus asymptote = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_and_distance_l295_29545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_intersection_probability_l295_29592

/-- The probability that a random chord on the outer circle intersects the inner circle -/
theorem chord_intersection_probability (r₁ r₂ : ℝ) (h : 0 < r₁ ∧ r₁ < r₂) :
  (2 * Real.arcsin (r₁ / r₂)) / (2 * Real.pi) = (Real.arccos (r₁ / r₂)) / Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_intersection_probability_l295_29592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l295_29586

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.sin x)^2 + 2 * Real.sin x * Real.cos x

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T')) ∧
  (∀ (x y : ℝ), 3 * Real.pi / 8 ≤ x ∧ x < y ∧ y ≤ 7 * Real.pi / 8 → f y < f x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l295_29586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l295_29537

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def isAcute (t : Triangle) : Prop :=
  0 < t.A ∧ t.A < Real.pi/2 ∧
  0 < t.B ∧ t.B < Real.pi/2 ∧
  0 < t.C ∧ t.C < Real.pi/2

def satisfiesCondition (t : Triangle) : Prop :=
  1 + (Real.tan t.A / Real.tan t.B) = (2 * t.c) / (Real.sqrt 3 * t.b)

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : isAcute t) (h2 : satisfiesCondition t) : 
  t.A = Real.pi/6 ∧ 
  ∀ y, y = 2 * (Real.sin t.B)^2 - 2 * Real.sin t.B * Real.cos t.C → 
  1 < y ∧ y < 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l295_29537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_markup_rate_for_given_conditions_l295_29509

/-- Calculates the rate of markup on cost for a product with given selling price, profit percentage, and expense percentage. -/
noncomputable def rate_of_markup_on_cost (selling_price : ℝ) (profit_percentage : ℝ) (expense_percentage : ℝ) : ℝ :=
  let cost := selling_price * (1 - profit_percentage - expense_percentage)
  (selling_price - cost) / cost * 100

/-- The rate of markup on cost for a product with a selling price of $8.00, 20% profit, and 10% expenses is approximately 42.857%. -/
theorem markup_rate_for_given_conditions :
  let selling_price : ℝ := 8
  let profit_percentage : ℝ := 0.20
  let expense_percentage : ℝ := 0.10
  abs (rate_of_markup_on_cost selling_price profit_percentage expense_percentage - 42.857) < 0.001 := by
  sorry

-- Note: We can't use #eval for noncomputable functions, so we'll remove this line
-- #eval rate_of_markup_on_cost 8 0.20 0.10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_markup_rate_for_given_conditions_l295_29509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicularity_reasoning_l295_29526

/-- Represents a geometric object (circle or sphere) --/
inductive GeometricObject
| Circle : GeometricObject
| Sphere : GeometricObject

/-- Represents a tangent (line or plane) --/
inductive Tangent
| Line : Tangent
| Plane : Tangent

/-- Represents the property of perpendicularity between the line from center to tangent point
    and the tangent --/
def perpendicularProperty (obj : GeometricObject) (tang : Tangent) : Bool :=
  match obj, tang with
  | GeometricObject.Circle, Tangent.Line => true
  | GeometricObject.Sphere, Tangent.Plane => true
  | _, _ => false

/-- Represents the type of reasoning --/
inductive ReasoningType
| Inductive
| Deductive
| Analogical
| Other

/-- The theorem stating that the reasoning used is analogical --/
theorem tangent_perpendicularity_reasoning :
  (perpendicularProperty GeometricObject.Circle Tangent.Line →
   perpendicularProperty GeometricObject.Sphere Tangent.Plane) →
  ReasoningType.Analogical = 
    (let reasoning := 
      if perpendicularProperty GeometricObject.Circle Tangent.Line &&
         perpendicularProperty GeometricObject.Sphere Tangent.Plane
      then ReasoningType.Analogical
      else ReasoningType.Other
    reasoning) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicularity_reasoning_l295_29526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_geq_one_range_l295_29543

-- Define the piecewise function
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 else 2*x - 1

-- Define the set of x values that satisfy f(x) ≥ 1
def S : Set ℝ := {x : ℝ | f x ≥ 1}

-- Theorem statement
theorem f_geq_one_range : S = Set.Iic (-1) ∪ Set.Ici 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_geq_one_range_l295_29543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_slopes_is_neg_half_l295_29562

/-- The ellipse equation -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

/-- Definition of a point being a vertex of the ellipse -/
def is_vertex (x y : ℝ) : Prop :=
  (x = 0 ∧ y = 1) ∨ (x = 0 ∧ y = -1) ∨ (x = Real.sqrt 2 ∧ y = 0) ∨ (x = -Real.sqrt 2 ∧ y = 0)

/-- Theorem: Product of slopes of OA and OB is -1/2 -/
theorem product_of_slopes_is_neg_half
  (x₁ y₁ x₂ y₂ x y : ℝ)
  (hA : is_on_ellipse x₁ y₁)
  (hB : is_on_ellipse x₂ y₂)
  (hM : is_on_ellipse x y)
  (hA_not_vertex : ¬is_vertex x₁ y₁)
  (hB_not_vertex : ¬is_vertex x₂ y₂)
  (hM_not_vertex : ¬is_vertex x y)
  (h_exists_theta : ∃ θ : ℝ, 0 < θ ∧ θ < Real.pi/2 ∧ 
    x = x₁ * Real.cos θ + x₂ * Real.sin θ ∧
    y = y₁ * Real.cos θ + y₂ * Real.sin θ)
  (hx₁_nonzero : x₁ ≠ 0)
  (hx₂_nonzero : x₂ ≠ 0) :
  y₁ / x₁ * (y₂ / x₂) = -1/2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_slopes_is_neg_half_l295_29562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_one_l295_29507

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x + Real.sqrt x

-- State the theorem
theorem tangent_slope_at_one :
  deriv f 1 = 3/2 := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_one_l295_29507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_housing_units_without_tv_or_vcr_l295_29501

/-- Functions to represent different housing unit categories -/
noncomputable def number_with_cable_tv : ℝ → ℝ := sorry
noncomputable def number_with_vcr : ℝ → ℝ := sorry
noncomputable def number_with_both : ℝ → ℝ := sorry
noncomputable def number_without_either : ℝ → ℝ := sorry

/-- Theorem: 3/4 of housing units have neither cable TV nor VCR -/
theorem housing_units_without_tv_or_vcr (T : ℝ) 
  (h1 : T > 0) 
  (h2 : T * (1/5) = number_with_cable_tv T)
  (h3 : T * (1/10) = number_with_vcr T)
  (h4 : number_with_cable_tv T * (1/4) = number_with_both T) :
  T * (3/4) = number_without_either T :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_housing_units_without_tv_or_vcr_l295_29501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_shift_equivalence_l295_29599

/-- Proves that shifting the graph of y = cos(x/2) by 2π/3 units to the right
    results in the graph of y = cos(x/2 - π/3) -/
theorem cos_shift_equivalence (x : ℝ) :
  Real.cos (x/2 - π/3) = Real.cos ((x - 2*π/3)/2) :=
by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_shift_equivalence_l295_29599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_line_l295_29540

-- Define the curve
noncomputable def curve (x : ℝ) : ℝ := (x + 1) / (x - 1)

-- Define the derivative of the curve
noncomputable def curve_derivative (x : ℝ) : ℝ := -2 / ((x - 1)^2)

-- Theorem statement
theorem tangent_perpendicular_line (a : ℝ) :
  curve 3 = 2 →
  curve_derivative 3 = -1/2 →
  (curve_derivative 3) * (-a) = 1 →
  a = -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_line_l295_29540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_R_l295_29553

/-- Represents a rhombus ABCD -/
structure Rhombus where
  side_length : ℝ
  angle_B : ℝ

/-- Represents the region R inside the rhombus -/
def region_R (r : Rhombus) : Set (ℝ × ℝ) :=
  sorry -- Placeholder definition

/-- The area of a set in ℝ² -/
noncomputable def area : Set (ℝ × ℝ) → ℝ :=
  sorry -- Placeholder definition

/-- The theorem statement -/
theorem area_of_region_R (r : Rhombus) 
  (h1 : r.side_length = 3)
  (h2 : r.angle_B = 150 * π / 180) : 
  area (region_R r) = 9 * (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_R_l295_29553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Ba_in_BaF2_l295_29517

/-- The molar mass of Barium (Ba) in g/mol -/
noncomputable def molar_mass_Ba : ℝ := 137.33

/-- The molar mass of Fluorine (F) in g/mol -/
noncomputable def molar_mass_F : ℝ := 19.00

/-- The number of Ba atoms in BaF2 -/
def num_Ba_atoms : ℕ := 1

/-- The number of F atoms in BaF2 -/
def num_F_atoms : ℕ := 2

/-- The molar mass of BaF2 in g/mol -/
noncomputable def molar_mass_BaF2 : ℝ := molar_mass_Ba + num_F_atoms * molar_mass_F

/-- The mass percentage of Ba in BaF2 -/
noncomputable def mass_percentage_Ba : ℝ := (molar_mass_Ba / molar_mass_BaF2) * 100

/-- Theorem stating that the mass percentage of Ba in BaF2 is approximately 78.35% -/
theorem mass_percentage_Ba_in_BaF2 : 
  78.34 < mass_percentage_Ba ∧ mass_percentage_Ba < 78.36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Ba_in_BaF2_l295_29517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_final_position_l295_29585

/-- Represents the position of the triangle on the pentagon -/
inductive Position
| Top
| Right
| Bottom
| Left

/-- Represents a regular polygon -/
structure RegularPolygon where
  sides : ℕ
  sideLength : ℝ

/-- Calculates the internal angle of a regular polygon -/
noncomputable def internalAngle (p : RegularPolygon) : ℝ :=
  ((p.sides - 2) * 180) / p.sides

/-- Calculates the rotation angle when a polygon rolls around another -/
noncomputable def rotationAngle (rolling : RegularPolygon) (fixed : RegularPolygon) : ℝ :=
  360 - (internalAngle fixed + internalAngle rolling)

/-- Theorem stating the final position of the triangle -/
theorem triangle_final_position 
  (pentagon : RegularPolygon)
  (hexagon : RegularPolygon)
  (h_pentagon : pentagon.sides = 5)
  (h_hexagon : hexagon.sides = 6)
  (initial_position : Position)
  (h_initial : initial_position = Position.Bottom)
  (num_moves : ℕ)
  (h_moves : num_moves = 4) :
  let total_rotation := (num_moves * rotationAngle pentagon hexagon) % 360
  let final_position := 
    if total_rotation < 90 then Position.Bottom
    else if total_rotation < 180 then Position.Left
    else if total_rotation < 270 then Position.Top
    else Position.Right
  final_position = Position.Left := by
  sorry

#eval "Theorem statement compiled successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_final_position_l295_29585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_condition_l295_29598

def sequence_a : ℕ → ℝ
  | 0 => 1
  | (n + 1) => 2 * sequence_a n + 1

theorem geometric_sequence_condition (c : ℝ) :
  (∃ r : ℝ, ∀ n : ℕ, n ≥ 1 → (sequence_a (n + 1) + c) = r * (sequence_a n + c)) →
  c = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_condition_l295_29598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_radius_l295_29530

/-- Predicate to check if a circle with radius r is tangent to two concentric circles with radii r1 and r2. -/
def is_tangent_to_concentric_circles (r r1 r2 : ℝ) : Prop :=
  (r1 + r = r2 - r) ∨ (r1 - r = r2 - r)

/-- The radius of a circle tangent to two concentric circles with radii 3 and 5 is 1. -/
theorem tangent_circle_radius (r : ℝ) : 
  r > 0 → is_tangent_to_concentric_circles r 3 5 → r = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_radius_l295_29530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l295_29550

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x)

noncomputable def g (x : ℝ) : ℝ := f (x + Real.pi / 4)

theorem g_properties :
  (∀ x, g x = -Real.sin (2 * x)) ∧
  (∀ x, g x ≤ 1) ∧
  (∀ x, g (-x) = -g x) ∧
  (∀ x, g (x + Real.pi) = g x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l295_29550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_emma_distance_l295_29574

-- Define the race distances
noncomputable def swim_distance : ℝ := 2
noncomputable def bike_distance : ℝ := 40
noncomputable def run_distance : ℝ := 10

-- Define the total race distance
noncomputable def total_distance : ℝ := swim_distance + bike_distance + run_distance

-- Define Emma's progress as a fraction of the total distance
noncomputable def emma_progress : ℝ := 1 / 13

-- Theorem statement
theorem emma_distance : emma_progress * total_distance = 4 := by
  -- Unfold the definitions
  unfold emma_progress total_distance swim_distance bike_distance run_distance
  -- Perform the calculation
  simp [mul_add, mul_div_cancel']
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_emma_distance_l295_29574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_dihedral_angle_l295_29554

/-- Represents an ellipse in 2D space -/
structure Ellipse where
  a : ℝ  -- semi-major axis
  b : ℝ  -- semi-minor axis

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  normal : Point3D  -- normal vector to the plane
  d : ℝ             -- distance from origin

/-- Calculates the dihedral angle between two planes -/
noncomputable def dihedralAngle (p1 p2 : Plane) : ℝ := sorry

/-- Calculates the projection of a point onto a plane -/
noncomputable def projectOntoPlane (pt : Point3D) (pl : Plane) : Point3D := sorry

/-- Checks if a point is the focus of an ellipse -/
def isFocus (pt : Point3D) (e : Ellipse) : Prop := sorry

/-- Main theorem -/
theorem ellipse_dihedral_angle (e : Ellipse) 
  (majorAxis : Point3D × Point3D) 
  (minorAxis : Point3D × Point3D) 
  (foldPlane : Plane) 
  (projectionPlane : Plane) :
  e.a = 4 ∧ e.b = 2*Real.sqrt 3 ∧ 
  foldPlane.normal = ⟨0, 1, 0⟩ ∧
  (let projectedPoint := projectOntoPlane majorAxis.fst projectionPlane
   isFocus projectedPoint e) →
  dihedralAngle foldPlane projectionPlane = π/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_dihedral_angle_l295_29554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_l295_29567

/-- The value of k for which the line x = k intersects the parabola x = -3y^2 - 4y + 7 at exactly one point -/
def intersection_k : ℚ := 25 / 3

/-- The parabola equation -/
def parabola (y : ℚ) : ℚ := -3 * y^2 - 4 * y + 7

/-- The line equation -/
def line (k : ℚ) (x : ℚ) : Prop := x = k

theorem unique_intersection :
  ∃! k, ∃! y, parabola y = k ∧ line k (parabola y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_l295_29567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_exp_neg_x_squared_approx_l295_29573

/-- The definite integral of e^(-x²) from 0 to 1 is approximately 0.747 with an error less than 0.001 -/
theorem integral_exp_neg_x_squared_approx :
  ∃ (ε : ℝ), ε < 0.001 ∧ |∫ (x : ℝ) in Set.Icc 0 1, Real.exp (-x^2) - 0.747| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_exp_neg_x_squared_approx_l295_29573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_is_12000_l295_29578

-- Define the triangle
def triangle (A B C : ℝ × ℝ) : Prop :=
  let (xa, ya) := A
  let (xb, yb) := B
  let (xc, yc) := C
  (xb - xa)^2 + (yb - ya)^2 = 5000^2 ∧
  (xc - xa)^2 + (yc - ya)^2 = 4000^2 ∧
  (xb - xc)^2 + (yb - yc)^2 = (xb - xa)^2 + (yb - ya)^2 - ((xc - xa)^2 + (yc - ya)^2)

-- Define the distance function
noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  let (xp, yp) := P
  let (xq, yq) := Q
  Real.sqrt ((xq - xp)^2 + (yq - yp)^2)

-- Theorem statement
theorem total_distance_is_12000 (A B C : ℝ × ℝ) :
  triangle A B C →
  distance A B + distance B C + distance C A = 12000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_is_12000_l295_29578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l295_29577

/-- Triangle ABC with midpoint M on BC -/
structure TriangleABCM where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  M : ℝ × ℝ
  BC_midpoint : M = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)

/-- The length of a line segment between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem triangle_side_length (t : TriangleABCM) 
  (h1 : distance t.A t.B = 7)
  (h2 : distance t.A t.C = 6)
  (h3 : distance t.A t.M = 4) :
  distance t.B t.C = Real.sqrt 106 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l295_29577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_determination_l295_29564

noncomputable def f (A ω φ : ℝ) (x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

theorem function_determination (A ω φ : ℝ) (h1 : A > 0) (h2 : ω > 0) (h3 : |φ| < π/2)
  (h4 : f A ω φ 2 = Real.sqrt 2) (h5 : f A ω φ 6 = 0) :
  ∃ (A' ω' φ' : ℝ), A' = Real.sqrt 2 ∧ ω' = π/8 ∧ φ' = π/4 ∧
    ∀ x, f A ω φ x = f A' ω' φ' x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_determination_l295_29564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_semicircle_area_l295_29557

/-- The side length of the regular octagon -/
def side_length : ℝ := 3

/-- The area of a regular octagon with side length s -/
noncomputable def octagon_area (s : ℝ) : ℝ := 2 * (1 + Real.sqrt 2) * s^2

/-- The area of a semicircle with radius r -/
noncomputable def semicircle_area (r : ℝ) : ℝ := 0.5 * Real.pi * r^2

/-- The number of sides in an octagon -/
def num_sides : ℕ := 8

/-- Theorem stating the area of the region inside the octagon but outside all semicircles -/
theorem octagon_semicircle_area : 
  octagon_area side_length - num_sides * semicircle_area (side_length / 2) = 
  18 * (1 + Real.sqrt 2) - 9 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_semicircle_area_l295_29557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_strategy_exists_l295_29580

/-- Represents a game state with remaining numbers -/
def GameState := List ℕ

/-- Represents a player's move, crossing out 9 numbers -/
def Move := List ℕ

/-- Checks if a move is valid (contains exactly 9 distinct numbers from the game state) -/
def validMove (state : GameState) (move : Move) : Prop :=
  move.length = 9 ∧ move.toFinset ⊆ state.toFinset ∧ move.Nodup

/-- Applies a move to a game state, removing the crossed out numbers -/
def applyMove (state : GameState) (move : Move) : GameState :=
  state.filter (λ x => x ∉ move.toFinset)

/-- Represents a strategy for the first player -/
def Strategy := GameState → Move

/-- The game sequence from 1 to 101 -/
def initialState : GameState := List.range' 1 101

/-- Theorem stating that the first player can always score at least 55 points -/
theorem first_player_strategy_exists :
  ∃ (strategy : Strategy),
    ∀ (moves : List Move),
      moves.length = 10 →
      (∀ m ∈ moves, validMove (initialState) m) →
      let finalState := moves.foldl applyMove initialState
      (finalState.maximum?.getD 0 - finalState.minimum?.getD 0) ≥ 55 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_strategy_exists_l295_29580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_not_on_line_l_min_distance_Q_to_l_max_distance_Q_to_l_l295_29528

-- Define the line l
noncomputable def line_l (x : ℝ) : ℝ := Real.sqrt 3 * x - 1

-- Define the curve C
noncomputable def curve_C (θ : ℝ) : ℝ × ℝ := (Real.cos θ, 2 + Real.sin θ)

-- Define point P
noncomputable def point_P : ℝ × ℝ := (2 * Real.sqrt 3, 2)

-- Distance function from a point to line l
noncomputable def distance_to_l (x y : ℝ) : ℝ := 
  |Real.sqrt 3 * x - y - 1| / 2

theorem point_P_not_on_line_l : 
  line_l point_P.1 ≠ point_P.2 := by sorry

theorem min_distance_Q_to_l :
  ∃ (θ : ℝ), distance_to_l (curve_C θ).1 (curve_C θ).2 = 1/2 ∧
  ∀ (φ : ℝ), distance_to_l (curve_C φ).1 (curve_C φ).2 ≥ 1/2 := by sorry

theorem max_distance_Q_to_l :
  ∃ (θ : ℝ), distance_to_l (curve_C θ).1 (curve_C θ).2 = 5/2 ∧
  ∀ (φ : ℝ), distance_to_l (curve_C φ).1 (curve_C φ).2 ≤ 5/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_not_on_line_l_min_distance_Q_to_l_max_distance_Q_to_l_l295_29528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_length_pq_l295_29546

/-- A rhombus ABCD with diagonals AC and BD, and a point N on AB -/
structure RhombusWithPoint where
  /-- Length of diagonal AC -/
  ac : ℝ
  /-- Length of diagonal BD -/
  bd : ℝ
  /-- Point N on side AB -/
  n : ℝ × ℝ
  /-- Assertion that ABCD is a rhombus -/
  is_rhombus : ac > 0 ∧ bd > 0
  /-- Assertion that N is on AB -/
  n_on_ab : 0 ≤ n.1 ∧ n.1 ≤ ac

/-- The perpendicular distance from N to AC -/
noncomputable def dist_to_ac (r : RhombusWithPoint) : ℝ :=
  r.n.2

/-- The perpendicular distance from N to BD -/
noncomputable def dist_to_bd (r : RhombusWithPoint) : ℝ :=
  r.ac / 2 - r.n.1

/-- The length of PQ -/
noncomputable def length_pq (r : RhombusWithPoint) : ℝ :=
  Real.sqrt ((dist_to_ac r)^2 + (dist_to_bd r)^2)

/-- The theorem stating that the minimum length of PQ is 4 -/
theorem min_length_pq :
  ∃ (r : RhombusWithPoint), r.ac = 18 ∧ r.bd = 24 ∧
  ∀ (s : RhombusWithPoint), s.ac = 18 → s.bd = 24 →
  length_pq r ≤ length_pq s ∧ length_pq r = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_length_pq_l295_29546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_seeds_rounded_l295_29533

def apple_seeds : List ℕ := [2, 3, 5, 5, 6, 7, 7, 8, 25]

def average_seeds : ℚ :=
  (apple_seeds.sum : ℚ) / apple_seeds.length

theorem average_seeds_rounded :
  (average_seeds * 100).floor / 100 = 756 / 100 :=
by
  -- The proof goes here
  sorry

#eval (average_seeds * 100).floor / 100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_seeds_rounded_l295_29533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_squared_l295_29504

def expression (x : ℝ) : ℝ :=
  4 * (x^2 - 2*x^3 + 3*x) + 2 * (x + x^3 - 4*x^2 + 2*x^5 - 2*x^2) - 6 * (2 + x - 3*x^3 - 2*x^2)

theorem coefficient_of_x_squared :
  ∃ (a b c : ℝ), (fun x => expression x) = (fun x => a*x^2 + b*x + c) ∧ a = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_squared_l295_29504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_at_6_l295_29521

-- Define the arithmetic sequence
noncomputable def arithmetic_sequence (a₁ : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * 2

-- Define the sum of the first n terms
noncomputable def sum_n_terms (a₁ : ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (2 * a₁ + (n - 1) * 2) / 2

theorem min_sum_at_6 (a₁ : ℝ) :
  (arithmetic_sequence a₁ 5)^2 = (arithmetic_sequence a₁ 2) * (arithmetic_sequence a₁ 6) →
  ∃ (n : ℕ), ∀ (m : ℕ), sum_n_terms a₁ n ≤ sum_n_terms a₁ m ∧ n = 6 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_at_6_l295_29521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_relation_l295_29556

open Real

theorem tan_double_angle_relation (a b : ℝ) (x : ℝ) :
  (tan x = a / b) →
  (tan (2 * x) = (b + 1) / (a + b)) →
  (∀ y, 0 < y ∧ y < x → ¬(tan y = a / b ∧ tan (2 * y) = (b + 1) / (a + b))) →
  x = arctan (1 / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_relation_l295_29556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_from_polar_equation_l295_29518

/-- The area of the circle defined by r = 3 cos θ - 4 sin θ is 25π/4 -/
theorem circle_area_from_polar_equation : 
  let r (θ : ℝ) := 3 * Real.cos θ - 4 * Real.sin θ
  (∃ c : ℝ × ℝ, ∃ rad : ℝ, ∀ θ : ℝ, 
    (r θ * Real.cos θ - c.1)^2 + (r θ * Real.sin θ - c.2)^2 = rad^2) ∧
  Real.pi * (5/2)^2 = 25 * Real.pi / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_from_polar_equation_l295_29518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_specific_l295_29593

/-- The volume of a frustum of a cone -/
noncomputable def frustum_volume (r₁ r₂ l : ℝ) : ℝ :=
  let h := Real.sqrt (l^2 - (r₂ - r₁)^2)
  (1/3) * Real.pi * h * (r₁^2 + r₂^2 + r₁*r₂)

/-- Theorem: The volume of a frustum of a cone with bottom radius 1, top radius 2, and slant height 3 is (7π√8)/3 -/
theorem frustum_volume_specific : frustum_volume 1 2 3 = (7 * Real.pi * Real.sqrt 8) / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_specific_l295_29593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_inequality_l295_29511

theorem cyclic_inequality (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) 
  (h_not_zero : a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) : 
  (a * (b + c) - b * c) / (b^2 + c^2) + 
  (b * (c + a) - c * a) / (c^2 + a^2) + 
  (c * (a + b) - a * b) / (a^2 + b^2) ≥ 3/2 := by
  sorry

#check cyclic_inequality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_inequality_l295_29511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_statement1_is_true_l295_29524

-- Define a geometric body
structure GeometricBody where
  -- Add necessary fields (placeholder)
  dummy : Unit

-- Define a plane
structure Plane where
  -- Add necessary fields (placeholder)
  dummy : Unit

-- Define a prism
def isPrism (g : GeometricBody) : Prop :=
  -- Add definition of a prism (placeholder)
  True

-- Define a frustum
def isFrustum (g : GeometricBody) : Prop :=
  -- Add definition of a frustum (placeholder)
  True

-- Define a cone
def isCone (g : GeometricBody) : Prop :=
  -- Add definition of a cone (placeholder)
  True

-- Define a truncated cone
def isTruncatedCone (g : GeometricBody) : Prop :=
  -- Add definition of a truncated cone (placeholder)
  True

-- Define isParallel (placeholder)
def isParallel (f1 f2 : GeometricBody) : Prop := True

-- Define isTrapezoid (placeholder)
def isTrapezoid (f : GeometricBody) : Prop := True

-- Define isParallelogram (placeholder)
def isParallelogram (f : GeometricBody) : Prop := True

-- Define cutCone (placeholder)
def cutCone (g : GeometricBody) (p : Plane) : GeometricBody := g

-- Statement 1
def statement1 : Prop :=
  ∃ (g : GeometricBody) (p : Plane), 
    isPrism g ∧ 
    ∃ (g1 g2 : GeometricBody), 
      (g1 ≠ g2) ∧ 
      (¬ isPrism g1 ∨ ¬ isPrism g2)

-- Statement 2
def statement2 : Prop :=
  ∀ (g : GeometricBody),
    (∃ (f1 f2 : GeometricBody), f1 ≠ f2 ∧ isParallel f1 f2) ∧
    (∀ (f : GeometricBody), f ≠ f1 ∧ f ≠ f2 → isTrapezoid f) →
    isFrustum g
  where
    f1 : GeometricBody := ⟨()⟩
    f2 : GeometricBody := ⟨()⟩

-- Statement 3
def statement3 : Prop :=
  ∀ (g : GeometricBody) (p : Plane),
    isCone g →
    isTruncatedCone (cutCone g p)

-- Statement 4
def statement4 : Prop :=
  ∀ (g : GeometricBody),
    (∃ (f1 f2 : GeometricBody), f1 ≠ f2 ∧ isParallel f1 f2) ∧
    (∀ (f : GeometricBody), f ≠ f1 ∧ f ≠ f2 → isParallelogram f) →
    isPrism g
  where
    f1 : GeometricBody := ⟨()⟩
    f2 : GeometricBody := ⟨()⟩

-- Theorem stating that only statement1 is true
theorem only_statement1_is_true : 
  statement1 ∧ ¬statement2 ∧ ¬statement3 ∧ ¬statement4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_statement1_is_true_l295_29524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_waiting_time_probability_l295_29561

noncomputable def arrival_start : ℝ := 7 + 50/60
noncomputable def arrival_end : ℝ := 8 + 30/60
noncomputable def train_times : List ℝ := [7, 8, 8 + 30/60]

noncomputable def favorable_time (t : ℝ) : Bool :=
  (arrival_start ≤ t ∧ t ≤ 8) ∨ (8 + 20/60 ≤ t ∧ t ≤ arrival_end)

theorem waiting_time_probability :
  (∫ t in arrival_start..arrival_end, if favorable_time t then 1 else 0) /
  (arrival_end - arrival_start) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_waiting_time_probability_l295_29561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_sixth_minus_a_l295_29522

theorem cos_pi_sixth_minus_a (a : ℝ) 
  (h1 : 0 < a ∧ a < π / 6) 
  (h2 : Real.sin (a + π / 3) = 12 / 13) : 
  Real.cos (π / 6 - a) = 12 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_sixth_minus_a_l295_29522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_product_simplification_l295_29576

theorem logarithm_product_simplification (x y : ℝ) (hx : x > 0) (hy : y > 0) (hy1 : y ≠ 1) :
  (Real.log x / Real.log (y^4)) * (Real.log (y^3) / Real.log (x^7)) * (Real.log (x^2) / Real.log (y^5)) *
  (Real.log (y^5) / Real.log (x^2)) * (Real.log (x^7) / Real.log (y^3)) = (1/4) * (Real.log x / Real.log y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_product_simplification_l295_29576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100_equals_4_l295_29596

-- Define the sum of digits function
def s (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + s (n / 10)

-- Define the sequence a_n
def a : ℕ → ℕ
  | 0 => 2^20
  | (n+1) => s (a n)

-- Theorem statement
theorem a_100_equals_4 : a 100 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100_equals_4_l295_29596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_cos_vertex_l295_29539

/-- 
Given an isosceles triangle with base angles B and C, and vertex angle A,
if sin B = 4/5, then cos A = 7/25
-/
theorem isosceles_triangle_cos_vertex (A B C : Real) : 
  A + B + C = Real.pi → B = C → Real.sin B = 4/5 → Real.cos A = 7/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_cos_vertex_l295_29539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_laila_cycles_on_tuesday_l295_29590

-- Define the days of the week
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday
deriving Repr, DecidableEq

def Day.succ : Day → Day
  | Monday => Tuesday
  | Tuesday => Wednesday
  | Wednesday => Thursday
  | Thursday => Friday
  | Friday => Saturday
  | Saturday => Sunday
  | Sunday => Monday

def Day.pred : Day → Day
  | Monday => Sunday
  | Tuesday => Monday
  | Wednesday => Tuesday
  | Thursday => Wednesday
  | Friday => Thursday
  | Saturday => Friday
  | Sunday => Saturday

-- Define the sports
inductive Sport
  | Basketball
  | Golf
  | Running
  | Cycling
  | Tennis
deriving Repr, DecidableEq

-- Define the schedule
def schedule : Day → Sport := sorry

-- Conditions
axiom one_sport_per_day : ∀ d : Day, ∃! s : Sport, schedule d = s

axiom runs_three_days : ∃ d1 d2 d3 : Day, 
  d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3 ∧ 
  schedule d1 = Sport.Running ∧ 
  schedule d2 = Sport.Running ∧ 
  schedule d3 = Sport.Running

axiom no_consecutive_running : ∀ d : Day, 
  schedule d = Sport.Running → 
  schedule (Day.succ d) ≠ Sport.Running ∧ 
  schedule (Day.pred d) ≠ Sport.Running

axiom monday_basketball : schedule Day.Monday = Sport.Basketball

axiom wednesday_golf : schedule Day.Wednesday = Sport.Golf

axiom plays_cycling_and_tennis : ∃ d1 d2 : Day, 
  schedule d1 = Sport.Cycling ∧ 
  schedule d2 = Sport.Tennis

axiom no_cycling_after_running_or_tennis : ∀ d : Day, 
  (schedule d = Sport.Running ∨ schedule d = Sport.Tennis) → 
  schedule (Day.succ d) ≠ Sport.Cycling

-- Theorem to prove
theorem laila_cycles_on_tuesday : 
  schedule Day.Tuesday = Sport.Cycling := by
  sorry

#eval Day.Monday
#eval Sport.Basketball

end NUMINAMATH_CALUDE_ERRORFEEDBACK_laila_cycles_on_tuesday_l295_29590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l295_29575

noncomputable def ellipse_C (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

theorem ellipse_properties (a b : ℝ) :
  ellipse_C a b (3/2) (-Real.sqrt 6 / 2) →
  eccentricity a b = Real.sqrt 3 / 3 →
  (∀ x y : ℝ, ellipse_C a b x y ↔ 2 * x^2 / 9 + y^2 / 3 = 1) ∧
  (∀ x₁ y₁ x₂ y₂ : ℝ, 
    ellipse_C a b x₁ y₁ → 
    ellipse_C a b x₂ y₂ → 
    x₁ ≠ x₂ → 
    ¬(Real.sqrt ((x₁ - 1)^2 + y₁^2) = Real.sqrt ((x₂ - 1)^2 + y₂^2) ∧
      Real.sqrt ((x₁ - 1)^2 + y₁^2) = Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2))) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l295_29575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l295_29588

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := |Real.cos (2 * x) + Real.cos x|

-- State the theorem
theorem f_properties :
  -- f is an even function
  (∀ x, f x = f (-x)) ∧
  -- f is monotonically decreasing on [-5π/4, -π]
  (∀ x y, -5 * Real.pi / 4 ≤ x ∧ x ≤ y ∧ y ≤ -Real.pi → f y ≤ f x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l295_29588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_decreasing_range_l295_29566

noncomputable def f (a : ℝ) : ℝ → ℝ := fun x ↦ (a - 1) ^ x

theorem monotonically_decreasing_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) → 1 < a ∧ a < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_decreasing_range_l295_29566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_rectangle_l295_29505

-- Define the area function
noncomputable def S (x : ℝ) : ℝ := (1/2) * (23*x - x^2)

-- Define the domain of x
def x_domain (x : ℝ) : Prop := 3 ≤ x ∧ x ≤ 11

-- Define the relationship between x and y
def y_relation (x y : ℝ) : Prop := y = (23 - x) / 2

-- Theorem statement
theorem max_area_rectangle :
  ∃ (x : ℝ), x_domain x ∧
  (∀ (x' : ℝ), x_domain x' → S x ≥ S x') ∧
  S x = 66 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_rectangle_l295_29505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_removed_percentage_l295_29560

/-- Represents the dimensions of a rectangular prism -/
structure PrismDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular prism -/
noncomputable def prismVolume (d : PrismDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Calculates the volume of a cube -/
noncomputable def cubeVolume (side : ℝ) : ℝ :=
  side ^ 3

/-- Calculates the percentage of volume removed -/
noncomputable def percentageVolumeRemoved (prismDim : PrismDimensions) (cubeSide : ℝ) : ℝ :=
  let totalRemovedVolume := 8 * cubeVolume cubeSide
  let originalVolume := prismVolume prismDim
  (totalRemovedVolume / originalVolume) * 100

theorem volume_removed_percentage (prismDim : PrismDimensions) (cubeSide : ℝ) :
  prismDim.length = 18 ∧ prismDim.width = 12 ∧ prismDim.height = 10 ∧ cubeSide = 4 →
  percentageVolumeRemoved prismDim cubeSide = (512 / 2160) * 100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_removed_percentage_l295_29560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_fare_20km_l295_29565

/-- Taxi fare calculation function -/
noncomputable def taxiFare (distance : ℝ) : ℝ :=
  let initialFare := (10 : ℝ)
  let initialDistance := (3 : ℝ)
  let intermediateRate := (2 : ℝ)
  let longDistanceRate := (2.4 : ℝ)
  let longDistanceThreshold := (8 : ℝ)
  
  if distance ≤ initialDistance then
    initialFare
  else if distance ≤ longDistanceThreshold then
    initialFare + (distance - initialDistance) * intermediateRate
  else
    initialFare + (longDistanceThreshold - initialDistance) * intermediateRate +
      (distance - longDistanceThreshold) * longDistanceRate

/-- Theorem: The taxi fare for a 20 km trip is 48.8 yuan -/
theorem taxi_fare_20km : taxiFare 20 = 48.8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_fare_20km_l295_29565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_max_min_values_l295_29519

open Real

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 4 * cos x * cos (x - π/3) - 2

-- Define the interval
def interval : Set ℝ := Set.Icc (-π/6) (π/4)

-- Theorem for the smallest positive period
theorem smallest_positive_period :
  ∃ (T : ℝ), T > 0 ∧ T = π ∧ ∀ (x : ℝ), f (x + T) = f x ∧
  ∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T ≤ T' := by
  sorry

-- Theorem for maximum and minimum values
theorem max_min_values :
  (∃ (x : ℝ), x ∈ interval ∧ f x = 1 ∧ ∀ (y : ℝ), y ∈ interval → f y ≤ 1) ∧
  (∃ (x : ℝ), x ∈ interval ∧ f x = -2 ∧ ∀ (y : ℝ), y ∈ interval → f y ≥ -2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_max_min_values_l295_29519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_triangle_hexagon_equal_perimeter_l295_29544

/-- Given an equilateral triangle and a regular hexagon with equal perimeters,
    the ratio of the area of the triangle to the area of the hexagon is 2/3. -/
theorem area_ratio_triangle_hexagon_equal_perimeter :
  ∀ (a b : ℝ), a > 0 → b > 0 →
  (3 * a = 6 * b) →  -- Equal perimeters condition
  (Real.sqrt 3 / 4 * a^2) / ((3 * Real.sqrt 3 / 2) * b^2) = 2/3 :=
by
  intros a b ha hb hper
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_triangle_hexagon_equal_perimeter_l295_29544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_reciprocal_sequence_l295_29594

/-- The result of applying the sequence (square, square, reciprocal) n times to x -/
noncomputable def repeated_square_reciprocal (x : ℝ) : ℕ → ℝ
  | 0 => x
  | n + 1 => 1 / ((repeated_square_reciprocal x n) ^ 2) ^ 2

/-- The final result is equal to x^((-4)^n) -/
theorem square_reciprocal_sequence (x : ℝ) (hx : x ≠ 0) (n : ℕ) :
  repeated_square_reciprocal x n = x ^ ((-4 : ℤ) ^ n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_reciprocal_sequence_l295_29594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_plus_2pi_3_l295_29581

theorem cos_alpha_plus_2pi_3 (α : ℝ) 
  (h1 : Real.sin (α + π/3) + Real.sin α = -(4 * Real.sqrt 3)/5)
  (h2 : -π/2 < α ∧ α < 0) : 
  Real.cos (α + 2*π/3) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_plus_2pi_3_l295_29581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_products_l295_29597

open Real BigOperators

/-- Product of sine terms for odd denominator -/
noncomputable def sin_product_odd (n : ℕ) : ℝ := 
  ∏ k in Finset.range n, sin (π * (k + 1 : ℝ) / (2 * n + 1 : ℝ))

/-- Product of sine terms for even denominator -/
noncomputable def sin_product_even (n : ℕ) : ℝ := 
  ∏ k in Finset.range (n - 1), sin (π * (k + 1 : ℝ) / (2 * n : ℝ))

/-- Product of cosine terms for odd denominator -/
noncomputable def cos_product_odd (n : ℕ) : ℝ := 
  ∏ k in Finset.range n, cos (π * (k + 1 : ℝ) / (2 * n + 1 : ℝ))

/-- Product of cosine terms for even denominator -/
noncomputable def cos_product_even (n : ℕ) : ℝ := 
  ∏ k in Finset.range (n - 1), cos (π * (k + 1 : ℝ) / (2 * n : ℝ))

/-- Theorem stating the simplified forms of trigonometric products -/
theorem trigonometric_products (n : ℕ) (h : n > 0) :
  sin_product_odd n = sqrt (2 * n + 1 : ℝ) / 2^n ∧
  sin_product_even n = sqrt n / 2^(n-1) ∧
  cos_product_odd n = 1 / 2^n ∧
  cos_product_even n = sqrt n / 2^(n-1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_products_l295_29597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_abc_l295_29506

noncomputable def a : ℝ := Real.log 2 / Real.log (1/3)
noncomputable def b : ℝ := Real.log 3 / Real.log (1/2)
noncomputable def c : ℝ := (1/2) ^ (3/10 : ℝ)

theorem order_abc : b < a ∧ a < c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_abc_l295_29506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_after_2517_hours_l295_29572

/-- Represents days of the week -/
inductive DayOfWeek
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

/-- Represents time in 12-hour format -/
structure Time where
  hour : Nat
  minute : Nat
  isPM : Bool

/-- Represents a specific day and time -/
structure DayTime where
  day : DayOfWeek
  time : Time

/-- Adds a given number of hours to a DayTime -/
def addHours (start : DayTime) (hours : Nat) : DayTime :=
  sorry

/-- Theorem stating that 2517 hours after Monday 3:00 PM is Tuesday 9:00 PM -/
theorem time_after_2517_hours :
  let start : DayTime := ⟨DayOfWeek.Monday, ⟨3, 0, true⟩⟩
  let end_time : DayTime := addHours start 2517
  end_time = ⟨DayOfWeek.Tuesday, ⟨9, 0, true⟩⟩ :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_after_2517_hours_l295_29572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_to_plane_l295_29525

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between planes
variable (perpPlanes : Plane → Plane → Prop)

-- Define the intersection of planes
variable (intersectPlanes : Plane → Plane → Set Line)

-- Define the parallel relation between a line and a plane
variable (parallelLinePlane : Line → Plane → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perpLinePlane : Line → Plane → Prop)

-- Define the perpendicular relation between two lines
variable (perpLines : Line → Line → Prop)

-- Main theorem
theorem perpendicular_line_to_plane
  (α β : Plane) (a AB : Line)
  (h1 : perpPlanes α β)
  (h2 : AB ∈ intersectPlanes α β)
  (h3 : parallelLinePlane a α)
  (h4 : perpLines a AB) :
  perpLinePlane a β :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_to_plane_l295_29525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_slope_product_l295_29514

/-- Definition of an ellipse -/
def is_on_ellipse (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- Definition of symmetry about the origin -/
def symmetric_about_origin (x1 y1 x2 y2 : ℝ) : Prop :=
  x1 = -x2 ∧ y1 = -y2

/-- Slope of a line -/
noncomputable def line_slope (x1 y1 x2 y2 : ℝ) : ℝ :=
  (y2 - y1) / (x2 - x1)

/-- Theorem: Product of slopes for points on an ellipse -/
theorem ellipse_slope_product
  (a b x0 y0 m n : ℝ)
  (h_ellipse : a > b ∧ b > 0)
  (h_P : is_on_ellipse x0 y0 a b)
  (h_M : is_on_ellipse m n a b)
  (h_symmetry : symmetric_about_origin m n (-m) (-n)) :
  line_slope x0 y0 m n * line_slope x0 y0 (-m) (-n) = -b^2 / a^2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_slope_product_l295_29514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_area_formula_l295_29502

/-- The area of a regular octagon with side length a -/
noncomputable def regular_octagon_area (a : ℝ) : ℝ := 2 * a^2 * (1 + Real.sqrt 2)

/-- Theorem: The area of a regular octagon with side length a is 2a²(1+√2) -/
theorem regular_octagon_area_formula (a : ℝ) (h : a > 0) :
  regular_octagon_area a = 2 * a^2 * (1 + Real.sqrt 2) := by
  -- Unfold the definition of regular_octagon_area
  unfold regular_octagon_area
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_area_formula_l295_29502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_P_l295_29579

/-- The line l passing through point C(2,3) for all m ∈ ℝ -/
def line_l (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (2 * m + 1) * p.1 - (m - 1) * p.2 - m - 5 = 0}

/-- Point A is the projection of point B on line l -/
def is_projection (A B : ℝ × ℝ) (m : ℝ) : Prop :=
  A ∈ line_l m ∧ (B.1 - A.1) * (2 * m + 1) + (B.2 - A.2) * (m - 1) = 0

/-- The distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The theorem to be proved -/
theorem max_distance_to_P :
  ∀ m : ℝ, ∃ B : ℝ × ℝ,
    is_projection (-4, 1) B m →
    (∀ B' : ℝ × ℝ, is_projection (-4, 1) B' m →
      distance B' (3, -1) ≤ distance B (3, -1)) ∧
    distance B (3, -1) = 5 + Real.sqrt 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_P_l295_29579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_of_square_mod_9_l295_29549

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

def f (n : ℕ) : ℕ := sum_of_digits (n^2)

theorem sum_of_digits_of_square_mod_9 (n : ℕ) :
  ∃ k ∈ ({0, 1, 4, 7} : Finset ℕ), f n ≡ k [MOD 9] :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_of_square_mod_9_l295_29549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l295_29503

-- Define the function f
def f (k : ℝ) (x : ℝ) : ℝ := k * x^3 - 3 * (k + 1) * x^2 - 2 * k^2 + 4

-- State the theorem
theorem function_properties :
  ∃ (k : ℝ),
    (∀ x ∈ Set.Ioo 0 4, (deriv (f k)) x < 0) ∧
    (k = 1) ∧
    (∀ a : ℝ, (∀ t ∈ Set.Icc (-1) 1, ∃ x : ℝ, 2 * x^2 + 5 * x + a = f k t) →
      a ≤ -5/4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l295_29503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l295_29589

theorem log_equation_solution (b x : ℝ) (hb_pos : b > 0) (hb_neq_one : b ≠ 1) (hx_neq_one : x ≠ 1)
  (h_eq : (Real.log x) / (2 * Real.log b) + (Real.log b) / (3 * Real.log x) = 1) :
  x = b^(1 + Real.sqrt 3 / 3) ∨ x = b^(1 - Real.sqrt 3 / 3) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l295_29589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_inequality_holds_l295_29548

open Real MeasureTheory ProbabilityTheory

-- Define the function f(x) = x + 1/(x-1)
noncomputable def f (x : ℝ) : ℝ := x + 1 / (x - 1)

-- Define the set of a values that satisfy the inequality
def A : Set ℝ := {a : ℝ | ∀ x > 1, f x ≥ a}

-- Define the probability space
def Ω : Type := ℝ

-- Define the probability measure on [0, 5]
noncomputable def P : Measure ℝ := by sorry

theorem probability_inequality_holds :
  P A / P (Set.Icc 0 5) = 2/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_inequality_holds_l295_29548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_difference_l295_29583

/-- A linear function g satisfying g(x+2) - g(x) = 4 for all real x -/
noncomputable def g : ℝ → ℝ := sorry

/-- g is a linear function -/
axiom g_linear : IsLinearMap ℝ g

/-- g satisfies g(x+2) - g(x) = 4 for all real x -/
axiom g_property (x : ℝ) : g (x + 2) - g x = 4

/-- Theorem: g(2) - g(6) = -8 -/
theorem g_difference : g 2 - g 6 = -8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_difference_l295_29583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_range_l295_29523

-- Define the curves C₁ and C₂
noncomputable def C₁ (α : ℝ) : ℝ × ℝ := (1 + Real.cos α, Real.sin α)

noncomputable def C₂ (θ : ℝ) : ℝ := 2 * Real.sin θ / (Real.cos θ)^2

-- Define the ray l
def l (k : ℝ) (x : ℝ) : ℝ := k * x

-- Define the intersection points
noncomputable def A (k : ℝ) : ℝ × ℝ := (2 / Real.sqrt (1 + k^2), 2 * k / Real.sqrt (1 + k^2))

noncomputable def B (k : ℝ) : ℝ × ℝ := (2 * k, 2 * k^2)

-- Define the product |OA|⋅|OB|
noncomputable def product (k : ℝ) : ℝ := 
  Real.sqrt ((A k).1^2 + (A k).2^2) * Real.sqrt ((B k).1^2 + (B k).2^2)

-- State the theorem
theorem intersection_product_range :
  ∀ k : ℝ, 1 ≤ k → k < Real.sqrt 3 → 4 ≤ product k ∧ product k < 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_range_l295_29523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_real_imag_zero_l295_29500

theorem complex_sum_real_imag_zero (b : ℝ) :
  (((2 : ℂ) - b * Complex.I).re + ((2 : ℂ) - b * Complex.I).im = 0) →
  b = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_real_imag_zero_l295_29500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l295_29587

/-- Given a geometric sequence with first term a and common ratio q,
    returns the sum of the first n terms -/
noncomputable def geometricSum (a : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a else a * (1 - q^n) / (1 - q)

/-- Theorem: For a geometric sequence {a_n} with common ratio q,
    if a_2, 2a_5, 3a_8 form an arithmetic sequence,
    then 3S_3/S_6 = 9/4 or 3S_3/S_6 = 3/2, where S_n is the sum of the first n terms -/
theorem geometric_sequence_ratio (a : ℝ) (q : ℝ) :
  (4 * a * q^3 = a + 3 * a * q^6) →
  (3 * geometricSum a q 3) / (geometricSum a q 6) = 9/4 ∨
  (3 * geometricSum a q 3) / (geometricSum a q 6) = 3/2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l295_29587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_city_taxi_theorem_l295_29527

/-- Represents a city with a grid of streets -/
structure City where
  horizontal_streets : Nat
  vertical_streets : Nat

/-- Represents a block in the city -/
structure Block where
  i : Nat
  j : Nat

/-- Calculates the number of blocks in the city -/
def num_blocks (c : City) : Nat :=
  (c.horizontal_streets - 1) * (c.vertical_streets - 1)

/-- Calculates the Manhattan distance between two blocks -/
def manhattan_distance (b1 b2 : Block) : Nat :=
  (if b1.i ≥ b2.i then b1.i - b2.i else b2.i - b1.i) +
  (if b1.j ≥ b2.j then b1.j - b2.j else b2.j - b1.j)

/-- Calculates the minimum fare between two blocks -/
def min_fare (b1 b2 : Block) : Nat :=
  (manhattan_distance b1 b2) * 100 / 100

/-- Calculates the maximum fare between two blocks -/
def max_fare (b1 b2 : Block) : Nat :=
  ((manhattan_distance b1 b2) * 100 + 199) / 100

/-- Main theorem about the city and taxi fares -/
theorem city_taxi_theorem (c : City) (b1 b2 : Block) : 
  c.horizontal_streets = 7 ∧ 
  c.vertical_streets = 13 ∧ 
  b1.i = 4 ∧ b1.j = 2 ∧ 
  b2.i = 1 ∧ b2.j = 9 →
  num_blocks c = 72 ∧
  min_fare b1 b2 = 10 ∧
  max_fare b1 b2 = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_city_taxi_theorem_l295_29527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_sum_equality_l295_29551

theorem ceiling_sum_equality : ⌈Real.sqrt (16/9:ℝ)⌉ + ⌈(16/9:ℝ)⌉ + ⌈(16/9:ℝ)^2⌉ = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_sum_equality_l295_29551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_truth_l295_29558

-- Define the types for planes and lines
structure Plane : Type
structure Line : Type

-- Define the relations between planes and lines
def perpendicular : Plane ⊕ Line → Plane ⊕ Line → Prop := sorry
def parallel : Plane ⊕ Line → Plane ⊕ Line → Prop := sorry
def subset : Line → Plane → Prop := sorry
def intersect : Plane → Plane → Prop := sorry

-- Define the planes and lines
variable (α β : Plane)
variable (m n : Line)

-- Define the non-coincidence of planes and lines
axiom planes_non_coincident : α ≠ β
axiom lines_non_coincident : m ≠ n

-- Define the propositions
def proposition_1 (α β : Plane) (m n : Line) : Prop :=
  perpendicular (Sum.inr m) (Sum.inr n) →
  perpendicular (Sum.inr m) (Sum.inl α) →
  ¬subset n α →
  parallel (Sum.inr n) (Sum.inl α)

def proposition_2 (α β : Plane) (m n : Line) : Prop :=
  perpendicular (Sum.inl α) (Sum.inl β) →
  intersect α β →
  subset n α →
  perpendicular (Sum.inr n) (Sum.inr m) →
  perpendicular (Sum.inr n) (Sum.inl β)

def proposition_3 (α β : Plane) (m n : Line) : Prop :=
  perpendicular (Sum.inr m) (Sum.inr n) →
  parallel (Sum.inr m) (Sum.inl α) →
  parallel (Sum.inr n) (Sum.inl β) →
  perpendicular (Sum.inl α) (Sum.inl β)

def proposition_4 (α β : Plane) (m n : Line) : Prop :=
  subset n α →
  subset m β →
  intersect α β →
  ¬perpendicular (Sum.inl α) (Sum.inl β) →
  ¬perpendicular (Sum.inr n) (Sum.inr m)

-- Theorem to prove
theorem propositions_truth (α β : Plane) (m n : Line) :
  proposition_1 α β m n ∧
  proposition_2 α β m n ∧
  ¬proposition_3 α β m n ∧
  ¬proposition_4 α β m n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_truth_l295_29558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_det_zero_l295_29513

-- Define a right triangle
def RightTriangle (A B C : Real) : Prop :=
  0 < A ∧ 0 < B ∧ C = Real.pi / 2 ∧ A + B + C = Real.pi

-- Define the determinant function
noncomputable def det_matrix (A B : Real) : Real :=
  Real.tan A * Real.tan B - 1

-- Theorem statement
theorem right_triangle_det_zero {A B C : Real} (h : RightTriangle A B C) : 
  det_matrix A B = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_det_zero_l295_29513
