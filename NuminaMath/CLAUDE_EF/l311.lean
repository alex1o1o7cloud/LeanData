import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_range_l311_31105

def f_properties (f : ℝ → ℝ) : Prop :=
  (∃! x : ℝ, x ≤ 0 ∧ f x = 0 ∧ x = -3) ∧
  (∀ x ≤ 0, x * (deriv f x) < f x) ∧
  (∀ x, f x = f (-x))

theorem f_inequality_range (f : ℝ → ℝ) (hf : f_properties f) :
  {x : ℝ | f x / x ≤ 0} = Set.union (Set.Icc (-3) 0) (Set.Ici 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_range_l311_31105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_beta_value_l311_31116

theorem cos_beta_value (α β : Real) 
  (h1 : Real.sin α = Real.sqrt 5 / 5)
  (h2 : Real.sin (α - β) = -(Real.sqrt 10 / 10))
  (h3 : 0 < α ∧ α < Real.pi/2)
  (h4 : 0 < β ∧ β < Real.pi/2) :
  Real.cos β = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_beta_value_l311_31116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_salary_is_3000_l311_31156

-- Define the salaries of A and B
variable (salary_A salary_B : ℝ)

-- Define the conditions
axiom total_salary : salary_A + salary_B = 4000
axiom A_savings : salary_A * 0.05 = salary_B * 0.15

-- Theorem to prove
theorem A_salary_is_3000 : salary_A = 3000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_salary_is_3000_l311_31156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_ell_l311_31142

theorem existence_of_ell (k : ℕ) : ∃ ℓ : ℕ, 
  ∀ m n : ℕ, m > 0 → n > 0 → (Nat.gcd m ℓ = 1 ∧ Nat.gcd n ℓ = 1) → 
  (m^m ≡ n^n [MOD ℓ]) → (m ≡ n [MOD k]) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_ell_l311_31142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_divisor_of_cube_minus_self_for_even_l311_31130

theorem largest_divisor_of_cube_minus_self_for_even :
  (∀ n : ℤ, Even n → ∃ k : ℤ, n^3 - n = 6 * k) ∧
  (∀ m : ℤ, m > 6 → ∃ n : ℤ, Even n ∧ ¬∃ k : ℤ, n^3 - n = m * k) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_divisor_of_cube_minus_self_for_even_l311_31130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_2_to_log_2_transformation_l311_31123

-- Define the exponential function
noncomputable def exp_2 (x : ℝ) : ℝ := 2^x

-- Define the logarithmic function
noncomputable def log_2 (x : ℝ) : ℝ := Real.log x / Real.log 2

-- Define the transformation
def transform (f : ℝ → ℝ) (d : ℝ) : ℝ → ℝ := λ x ↦ f x - d

-- Statement of the theorem
theorem exp_2_to_log_2_transformation :
  ∀ x : ℝ, x > -1 →
  log_2 (x + 1) = (transform exp_2 1) ⁻¹ x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_2_to_log_2_transformation_l311_31123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_division_l311_31182

theorem equal_area_division (d : ℝ) : d = 2 ↔ 
  (1/2) * d * 3 = 6 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_division_l311_31182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_surface_area_l311_31151

noncomputable section

-- Define the points and constants
variable (A B C D O : EuclideanSpace ℝ (Fin 3))
variable (a : ℝ)

-- Define the conditions
def on_sphere (X : EuclideanSpace ℝ (Fin 3)) : Prop := sorry
def distance (X Y : EuclideanSpace ℝ (Fin 3)) : ℝ := sorry
def center_of_circumscribed_sphere (O A B C D : EuclideanSpace ℝ (Fin 3)) : Prop := sorry
def on_edge (O A D : EuclideanSpace ℝ (Fin 3)) : Prop := sorry

-- State the theorem
theorem sphere_surface_area 
  (h1 : on_sphere A ∧ on_sphere B ∧ on_sphere C ∧ on_sphere D)
  (h2 : distance A B = a)
  (h3 : distance B C = a)
  (h4 : distance A C = Real.sqrt 2 * a)
  (h5 : center_of_circumscribed_sphere O A B C D)
  (h6 : on_edge O A D)
  (h7 : distance D C = Real.sqrt 6 * a)
  : (4 : ℝ) * Real.pi * a^2 * 2 = 8 * Real.pi * a^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_surface_area_l311_31151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_m_range_l311_31115

-- Define the function f(x)
noncomputable def f (x : ℝ) (m : ℝ) : ℝ := Real.log x - m / x

-- Define the property of f being increasing on [1, 3]
def is_increasing_on_interval (m : ℝ) : Prop :=
  ∀ x y, 1 ≤ x → x ≤ y → y ≤ 3 → f x m ≤ f y m

-- Theorem statement
theorem f_increasing_m_range :
  {m : ℝ | is_increasing_on_interval m} = {m : ℝ | m ≥ -1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_m_range_l311_31115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extra_charge_count_l311_31136

/-- Represents an envelope with length and height -/
structure Envelope where
  length : ℚ
  height : ℚ

/-- Determines if an envelope requires an extra charge based on its dimensions -/
def requiresExtraCharge (e : Envelope) : Bool :=
  let ratio := e.length / e.height
  ratio < 14/10 || ratio > 24/10

/-- The set of envelopes to be checked -/
def envelopes : List Envelope := [
  { length := 5, height := 4 },
  { length := 10, height := 4 },
  { length := 4, height := 4 },
  { length := 12, height := 5 }
]

theorem extra_charge_count : 
  (envelopes.filter requiresExtraCharge).length = 3 := by
  sorry

#eval (envelopes.filter requiresExtraCharge).length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extra_charge_count_l311_31136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_4_lt_b_7_l311_31131

def b : ℕ → (ℕ → ℕ) → ℚ
  | 0, _ => 1  -- Adding a base case for 0
  | 1, α => 1 + 1 / α 1
  | n+1, α => 1 + 1 / (α 1 + 1 / b n (fun i => α (i+1)))

theorem b_4_lt_b_7 (α : ℕ → ℕ) (h : ∀ k, α k > 0) : b 4 α < b 7 α := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_4_lt_b_7_l311_31131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_theorem_l311_31110

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola -/
structure Parabola where
  vertex : Point
  focus : Point

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Check if a point is in the first quadrant -/
def isFirstQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- Check if a point is on the parabola -/
def isOnParabola (p : Point) (parab : Parabola) : Prop :=
  distance p parab.focus = p.y - parab.vertex.y + distance parab.vertex parab.focus

theorem parabola_point_theorem (parab : Parabola) (p : Point) :
  parab.vertex = ⟨0, 0⟩ →
  parab.focus = ⟨0, 2⟩ →
  isFirstQuadrant p →
  isOnParabola p parab →
  distance p parab.focus = 50 →
  p = ⟨8 * Real.sqrt 3, 48⟩ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_theorem_l311_31110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_x_cube_minus_x_square_l311_31197

-- Define the complex number i
def i : ℂ := Complex.I

-- Define x as given in the problem
noncomputable def x : ℂ := (1 + i * Real.sqrt 7) / 2

-- State the theorem
theorem inverse_x_cube_minus_x_square : 1 / (x^3 - x^2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_x_cube_minus_x_square_l311_31197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_economic_figure_calculation_l311_31177

-- Define the variables
variable (f p w : ℂ)

-- State the theorem
theorem economic_figure_calculation 
  (eq : f * p - w = (10000 : ℂ))
  (f_val : f = (5 : ℂ))
  (w_val : w = (5 : ℂ) + 125 * Complex.I) :
  p = (2001 : ℂ) + 25 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_economic_figure_calculation_l311_31177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_distance_to_parallel_line_l311_31174

/-- A line passing through a point and parallel to another line -/
structure ParallelLine where
  point : ℝ × ℝ
  parallel_to : ℝ → ℝ → ℝ → Prop

/-- The distance from a point to a line -/
noncomputable def distance_point_to_line (P : ℝ × ℝ) (l : ParallelLine) : ℝ :=
  sorry

/-- Theorem stating that the point (a, 2) with distance √5/5 to the line has a = 1 -/
theorem point_distance_to_parallel_line 
  (l : ParallelLine)
  (a : ℝ)
  (h1 : l.point = (2, 3))
  (h2 : l.parallel_to 1 (-2) 1)
  (h3 : a > 0)
  (h4 : distance_point_to_line (a, 2) l = Real.sqrt 5 / 5) :
  a = 1 := by
  sorry

#check point_distance_to_parallel_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_distance_to_parallel_line_l311_31174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_section_range_l311_31137

/-- The golden ratio -/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- Theorem: Given a range [1000, m] where a point x within this range divides it 
    according to the golden ratio, m must be either 2000 or 2618. -/
theorem golden_section_range (m : ℝ) : 
  (∃ x : ℝ, 1000 ≤ x ∧ x ≤ m ∧ (x - 1000) / (m - x) = φ) → 
  (m = 2000 ∨ m = 2618) := by
  sorry

#check golden_section_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_section_range_l311_31137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_area_triangle_ABC_l311_31143

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = -x^2 + 5*x + 2

/-- Point A -/
def A : ℝ × ℝ := (1, 0)

/-- Point B -/
def B : ℝ × ℝ := (5, 2)

/-- Point C -/
def C (p q : ℝ) : ℝ × ℝ := (p, q)

/-- Helper function to calculate the area of a triangle -/
noncomputable def area_triangle (P Q R : ℝ × ℝ) : ℝ :=
  (1/2) * abs (P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2))

/-- The theorem stating the largest possible area of triangle ABC -/
theorem largest_area_triangle_ABC :
  ∀ p q : ℝ,
  1 ≤ p → p ≤ 5 →
  parabola p q →
  parabola A.1 A.2 →
  parabola B.1 B.2 →
  (∀ p' q' : ℝ, 1 ≤ p' → p' ≤ 5 → parabola p' q' →
    area_triangle A B (C p' q') ≤ area_triangle A B (C p q)) →
  area_triangle A B (C p q) = 4.125 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_area_triangle_ABC_l311_31143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_osmotic_pressure_independent_of_protein_weight_l311_31198

/-- Represents the osmotic pressure of plasma -/
def osmoticPressure (proteinContent : ℝ) (naContent : ℝ) (clContent : ℝ) (proteinMolecularWeight : ℝ) : ℝ := sorry

/-- Represents the content of protein in plasma -/
def proteinContent : ℝ := sorry

/-- Represents the content of Na+ in plasma -/
def naContent : ℝ := sorry

/-- Represents the content of Cl- in plasma -/
def clContent : ℝ := sorry

/-- Represents the molecular weight of plasma protein -/
def proteinMolecularWeight : ℝ := sorry

/-- Theorem stating that osmotic pressure of plasma is independent of protein molecular weight -/
theorem osmotic_pressure_independent_of_protein_weight :
  ∀ (p₁ p₂ na cl : ℝ),
    osmoticPressure p₁ na cl proteinMolecularWeight =
    osmoticPressure p₂ na cl proteinMolecularWeight :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_osmotic_pressure_independent_of_protein_weight_l311_31198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l311_31188

theorem equation_solution (x : ℝ) : 
  (1 / 2 : ℝ) + 1 / x^2 = 7 / 8 → x = Real.sqrt (8 / 3) ∨ x = -Real.sqrt (8 / 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l311_31188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l311_31154

noncomputable def a : ℝ := Real.log 2 / Real.log 0.3
noncomputable def b : ℝ := Real.log 0.3 / Real.log 2
noncomputable def c : ℝ := (0.2 : ℝ) ^ (0.3 : ℝ)

theorem relationship_abc : b < a ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l311_31154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xiao_zhang_trip_time_l311_31139

/-- Represents the round trip of Xiao Zhang's mountain drive --/
structure MountainTrip where
  uphill_speed : ℝ
  downhill_speed : ℝ
  total_distance : ℝ
  return_trip_time_difference : ℝ

/-- Calculates the total time for a mountain trip --/
noncomputable def total_trip_time (trip : MountainTrip) : ℝ :=
  (trip.total_distance / 2) * (1 / trip.uphill_speed + 1 / trip.downhill_speed)

/-- Theorem stating that Xiao Zhang's trip took 7 hours --/
theorem xiao_zhang_trip_time :
  let trip := MountainTrip.mk 30 40 240 (1/6)
  total_trip_time trip = 7 := by
  sorry

#eval "Compilation successful!"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xiao_zhang_trip_time_l311_31139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_sphere_volume_ratio_l311_31104

/-- The ratio of the volume of a cylinder to the volume of a sphere,
    where the cylinder's base diameter and height are equal to the sphere's diameter -/
theorem cylinder_sphere_volume_ratio :
  ∀ (r : ℝ), r > 0 →
  (2 * Real.pi * r^3) / ((4/3) * Real.pi * r^3) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_sphere_volume_ratio_l311_31104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l311_31119

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos (2 * x)

theorem f_properties :
  (∃ (M : ℝ), ∀ (x : ℝ), f x ≤ M ∧ ∃ (x₀ : ℝ), f x₀ = M) ∧ -- maximum value exists
  (∀ (x : ℝ), f (-π/2 - x) = f (-π/2 + x)) ∧ -- symmetric about x = -π/2
  (∀ (x : ℝ), f (-x) = -f x) ∧ -- odd function
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x) ∧ -- periodic function
  ¬(∀ (x : ℝ), f (3*π/2 - x) = -f (3*π/2 + x)) -- not centrally symmetric about (3π/2, 0)
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l311_31119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_all_obtuse_tetrahedron_l311_31135

/-- A tetrahedron is represented by its four vertices in 3D space -/
def Tetrahedron := Fin 4 → ℝ × ℝ × ℝ

/-- An edge of a tetrahedron is represented by a pair of distinct vertex indices -/
def Edge := {p : Fin 4 × Fin 4 // p.1 ≠ p.2}

/-- A face of a tetrahedron is represented by three distinct vertex indices -/
def Face := {f : Fin 4 × Fin 4 × Fin 4 // f.1 ≠ f.2.1 ∧ f.1 ≠ f.2.2 ∧ f.2.1 ≠ f.2.2}

/-- The angle between two edges of a face at a given vertex -/
noncomputable def angle (t : Tetrahedron) (f : Face) (v : Fin 4) : ℝ := sorry

/-- Two faces are adjacent if they share an edge -/
def adjacent (f1 f2 : Face) : Prop := sorry

/-- An angle is obtuse if it's greater than π/2 -/
def is_obtuse (θ : ℝ) : Prop := θ > Real.pi / 2

/-- The main theorem: there is no tetrahedron where each pair of adjacent faces has an obtuse angle at their shared edge -/
theorem no_all_obtuse_tetrahedron :
  ¬ ∃ (t : Tetrahedron), ∀ (f1 f2 : Face), adjacent f1 f2 →
    ∃ (v : Fin 4), (v = f1.val.1 ∨ v = f1.val.2.1 ∨ v = f1.val.2.2) ∧
                   (v = f2.val.1 ∨ v = f2.val.2.1 ∨ v = f2.val.2.2) ∧
      (is_obtuse (angle t f1 v) ∨ is_obtuse (angle t f2 v)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_all_obtuse_tetrahedron_l311_31135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_triangle_area_relation_l311_31180

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a rectangle given two opposite corners -/
noncomputable def rectangleArea (p1 p2 : Point) : ℝ :=
  |p1.x - p2.x| * |p1.y - p2.y|

/-- Calculates the area of a triangle given three points -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ :=
  |p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y)| / 2

/-- The theorem to be proved -/
theorem rectangle_triangle_area_relation :
  let o : Point := ⟨0, 0⟩
  let q : Point := ⟨4, 2⟩
  let p : Point := ⟨4, 0⟩
  let t : Point := ⟨8, 0⟩
  rectangleArea o q = 2 * triangleArea p q t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_triangle_area_relation_l311_31180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distances_is_3root2_l311_31147

/-- Line l with parametric equations x = 3 + t and y = √5 + t -/
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (3 + t, Real.sqrt 5 + t)

/-- Circle C with equation x² + y² - 2√5y = 0 -/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2 * Real.sqrt 5 * y = 0

/-- Point P -/
noncomputable def point_P : ℝ × ℝ := (3, Real.sqrt 5)

/-- The sum of distances from P to the intersection points of l and C -/
noncomputable def sum_distances : ℝ := 3 * Real.sqrt 2

/-- Theorem stating that the sum of distances is 3√2 -/
theorem sum_distances_is_3root2 : sum_distances = 3 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distances_is_3root2_l311_31147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_on_square_l311_31168

/-- The area of a shape formed by an equilateral triangle on top of a square -/
theorem area_triangle_on_square (s : ℝ) (h : s = 4) :
  s^2 + (Real.sqrt 3 / 4) * s^2 = 16 + 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_on_square_l311_31168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l311_31103

/-- The function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x) + 2 * Real.sin x + 4 * Real.sqrt 3 * Real.cos x

/-- Theorem stating that the maximum value of f(x) is 17/2 -/
theorem f_max_value : ∀ x : ℝ, f x ≤ 17/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l311_31103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_petya_sequence_contains_five_l311_31108

theorem petya_sequence_contains_five (n : ℕ) : 
  ∃ k : ℕ+, ∀ m : ℕ, m ≥ k → (∃ d : ℕ, d ∈ Nat.digits 10 (n * 5^m) ∧ d = 5) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_petya_sequence_contains_five_l311_31108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_plus_three_pi_half_l311_31181

-- Define the function f
noncomputable def f (ω : ℝ) (φ : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (ω * x + φ)

-- State the theorem
theorem cos_alpha_plus_three_pi_half 
  (ω : ℝ) (φ : ℝ) (α : ℝ)
  (h_ω : ω > 0)
  (h_φ : -π/2 ≤ φ ∧ φ < π/2)
  (h_f0 : f ω φ 0 = -Real.sqrt 3 / 2)
  (h_sym : ∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f ω φ (x + T/2) = -f ω φ x ∧ T = π/2)
  (h_α : π/6 < α ∧ α < 2*π/3)
  (h_fα : f ω φ (α/2) = Real.sqrt 3 / 4) :
  Real.cos (α + 3*π/2) = (Real.sqrt 3 + Real.sqrt 15) / 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_plus_three_pi_half_l311_31181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chopped_cube_height_is_zero_l311_31107

/-- The height of a chopped cube after rotation -/
noncomputable def choppedCubeHeight (cubeSideLength : ℝ) : ℝ :=
  let diagonalLength := cubeSideLength * Real.sqrt 3
  let pyramidVolume := (1 / 6) * cubeSideLength ^ 3
  let cutFaceArea := (Real.sqrt 3 / 4) * (cubeSideLength * Real.sqrt 2) ^ 2
  let pyramidHeight := 3 * pyramidVolume / cutFaceArea
  cubeSideLength - pyramidHeight

/-- Theorem: The height of a chopped cube with side length 2 after rotation is 0 -/
theorem chopped_cube_height_is_zero :
  choppedCubeHeight 2 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chopped_cube_height_is_zero_l311_31107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_count_l311_31146

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + (y+5)^2 = 3

-- Define a line with equal x and y intercepts
def line_equal_intercepts (m b : ℝ) : Prop := 
  ∃ a : ℝ, (a ≠ 0 ∧ b = a ∧ m = -1) ∨ (a = 0 ∧ b = 0 ∧ m = -1)

-- Define a tangent line to the circle
def is_tangent_line (m b : ℝ) : Prop :=
  ∃ x y : ℝ, circle_C x y ∧ y = m * x + b ∧
  ∀ x' y' : ℝ, (y' = m * x' + b) → (x' = x ∧ y' = y ∨ ¬circle_C x' y')

-- Theorem statement
theorem tangent_lines_count :
  ∃! (lines : Finset (ℝ × ℝ)), 
    Finset.card lines = 4 ∧
    ∀ l ∈ lines, let (m, b) := l; is_tangent_line m b ∧ line_equal_intercepts m b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_count_l311_31146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_n_formula_l311_31101

noncomputable def f (x : ℝ) : ℝ := x / (x + 2)

noncomputable def f_n : ℕ → ℝ → ℝ
| 0, x => x
| 1, x => f x
| (n+2), x => f (f_n (n+1) x)

theorem f_n_formula (n : ℕ) (x : ℝ) (h1 : n ≥ 2) (h2 : x > 0) :
  f_n n x = x / ((2^n - 1) * x + 2^n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_n_formula_l311_31101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_l311_31165

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The distance from a point to a horizontal line -/
def distanceToHorizontalLine (p : Point) (y : ℝ) : ℝ :=
  |p.y - y|

/-- The equation of a parabola given its focus and directrix -/
def parabolaEquation (focus : Point) (directrix : ℝ) (p : Point) : Prop :=
  distance p focus = distanceToHorizontalLine p directrix

theorem trajectory_equation (P : Point) :
  (distanceToHorizontalLine P (-1) = distance P ⟨0, 3⟩ - 2) →
  (∃ (x y : ℝ), P = ⟨x, y⟩ ∧ x^2 = 12 * y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_l311_31165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_to_asymptote_distance_special_case_l311_31161

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := Real.sqrt (1 + h.b^2 / h.a^2)

/-- The distance from the center to a focus of a hyperbola -/
noncomputable def focal_distance (h : Hyperbola) : ℝ := h.a * eccentricity h

/-- The distance from a focus to an asymptote of a hyperbola -/
noncomputable def focus_to_asymptote_distance (h : Hyperbola) : ℝ := h.b^2 / h.a

theorem focus_to_asymptote_distance_special_case (h : Hyperbola)
  (h_real_axis : h.a = 1)
  (h_eccentricity : eccentricity h = Real.sqrt 5) :
  focus_to_asymptote_distance h = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_to_asymptote_distance_special_case_l311_31161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_january_salary_l311_31159

/-- Prove that given the average salaries for two sets of four months and the salary for May, 
    the salary for January can be determined. -/
theorem january_salary 
  (jan feb mar apr : ℝ)
  (avg_jan_to_apr : ℝ) 
  (avg_feb_to_may : ℝ) 
  (may_salary : ℝ) 
  (h1 : (jan + feb + mar + apr) / 4 = avg_jan_to_apr)
  (h2 : (feb + mar + apr + may_salary) / 4 = avg_feb_to_may)
  (h3 : avg_jan_to_apr = 8000)
  (h4 : avg_feb_to_may = 8200)
  (h5 : may_salary = 6500) :
  jan = 5700 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_january_salary_l311_31159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_function_single_peak_l311_31167

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle in 2D space -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Represents the path around the triangle -/
inductive PathSegment
  | AB
  | BC
  | CA

/-- Function to calculate the straight-line distance from a fixed point -/
def distanceFromStart (start : Point) (current : Point) : ℝ :=
  sorry

/-- Function to get the current point given a path segment and progress along it -/
def getCurrentPoint (t : Triangle) (seg : PathSegment) (progress : ℝ) : Point :=
  sorry

/-- Theorem stating that the distance function has a single peak -/
theorem distance_function_single_peak (t : Triangle) :
  ∃ peak : ℝ, 
    (∀ x, x < peak → (λ p => distanceFromStart t.A (getCurrentPoint t PathSegment.AB p)) x < 
                      (λ p => distanceFromStart t.A (getCurrentPoint t PathSegment.AB p)) peak) ∧
    (∀ x, peak < x → x ≤ 3 → (λ p => distanceFromStart t.A (getCurrentPoint t PathSegment.CA (x - 2))) x < 
                              (λ p => distanceFromStart t.A (getCurrentPoint t PathSegment.CA (x - 2))) peak) :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_function_single_peak_l311_31167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_ratio_l311_31191

theorem min_value_ratio (A B x : ℝ) (hA : A > 0) (hB : B > 0) (hx : x ≠ 0)
  (h1 : x^2 + 1/x^2 = A) (h2 : 3 * (x - 1/x) = B) :
  A / B ≥ 2 * Real.sqrt 2 / 3 ∧ ∃ y : ℝ, A / B = 2 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_ratio_l311_31191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_product_divisors_l311_31179

theorem prime_product_divisors (p q : ℕ) (n : ℕ) : 
  Nat.Prime p → Nat.Prime q → (Nat.divisors (p^n * q^6)).card = 28 → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_product_divisors_l311_31179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_positive_integer_with_remainder_one_l311_31111

theorem least_positive_integer_with_remainder_one (n : ℕ) : n = 10396 ↔ 
  (n > 1) ∧ 
  (∀ d : ℕ, d ∈ ({2, 3, 4, 5, 6, 7, 8, 9, 11} : Finset ℕ) → n % d = 1) ∧ 
  (∀ m : ℕ, m > 1 ∧ (∀ d : ℕ, d ∈ ({2, 3, 4, 5, 6, 7, 8, 9, 11} : Finset ℕ) → m % d = 1) → m ≥ n) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_positive_integer_with_remainder_one_l311_31111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_participation_schemes_l311_31187

/-- Represents a participation scheme for a competition with 6 people and 4 subjects. -/
structure ParticipationScheme where
  participants : Finset (Fin 6)
  assignment : Fin 6 → Fin 4
  valid_participants : participants.card = 4
  valid_assignment : ∀ i, i ∈ participants → assignment i < 4
  all_subjects_covered : ∀ j, j < 4 → ∃ i ∈ participants, assignment i = j
  english_restriction : ∀ i, i ∈ Finset.range 2 → assignment i ≠ 3

/-- The number of valid participation schemes. -/
def num_valid_schemes : ℕ := sorry

theorem count_participation_schemes :
  num_valid_schemes = 240 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_participation_schemes_l311_31187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_interval_of_f_l311_31149

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 6 / x - Real.log x / Real.log 2

-- State the theorem
theorem zero_point_interval_of_f :
  ∃ c ∈ Set.Ioo 3 4, f c = 0 :=
by
  -- Assume the conditions
  have h1 : Continuous f := sorry
  have h2 : StrictAnti f := sorry
  have h3 : f 3 > 0 := sorry
  have h4 : f 4 < 0 := sorry
  
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_interval_of_f_l311_31149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_cosine_relation_l311_31190

/-- Given a point Q in the first octant of 3D space, if the angles between the line OQ
    (where O is the origin) and the x-, y-, and z-axes are α, β, and γ respectively,
    and cos α = 1/4 and cos β = 1/3, then cos γ = √119 / 12. -/
theorem angle_cosine_relation (Q : ℝ × ℝ × ℝ) (α β γ : ℝ) :
  Q.1 > 0 ∧ Q.2.1 > 0 ∧ Q.2.2 > 0 →
  (Q.1 / Real.sqrt (Q.1^2 + Q.2.1^2 + Q.2.2^2) = Real.cos α) →
  (Q.2.1 / Real.sqrt (Q.1^2 + Q.2.1^2 + Q.2.2^2) = Real.cos β) →
  (Q.2.2 / Real.sqrt (Q.1^2 + Q.2.1^2 + Q.2.2^2) = Real.cos γ) →
  Real.cos α = 1/4 →
  Real.cos β = 1/3 →
  Real.cos γ = Real.sqrt 119 / 12 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_cosine_relation_l311_31190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sandys_savings_percentage_is_ten_percent_l311_31144

/-- The percentage of salary Sandy saved this year, given the conditions from the problem -/
noncomputable def sandys_savings_percentage : ℝ :=
  let last_year_salary : ℝ := 100 -- Arbitrary value for last year's salary
  let last_year_savings_rate : ℝ := 0.06
  let salary_increase_rate : ℝ := 0.10
  let savings_increase_rate : ℝ := 1.8333333333333331

  let this_year_salary : ℝ := last_year_salary * (1 + salary_increase_rate)
  let last_year_savings : ℝ := last_year_salary * last_year_savings_rate
  let this_year_savings : ℝ := last_year_savings * savings_increase_rate

  (this_year_savings / this_year_salary) * 100

/-- Proof that Sandy's savings percentage this year is 10% -/
theorem sandys_savings_percentage_is_ten_percent :
  sandys_savings_percentage = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sandys_savings_percentage_is_ten_percent_l311_31144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_plus_k_l311_31152

-- Define the function f
noncomputable def f (a k x : ℝ) : ℝ := a^x + k * a^(-x)

-- State the theorem
theorem range_of_a_plus_k (a k : ℝ) 
  (h_a_pos : a > 0) 
  (h_a_neq_1 : a ≠ 1) 
  (h_odd : ∀ x, f a k (-x) = -f a k x) 
  (h_decreasing : ∀ x y, x < y → f a k x > f a k y) :
  a + k ∈ Set.Ioo (-1) 0 := by
  sorry

#check range_of_a_plus_k

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_plus_k_l311_31152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l311_31129

theorem trig_problem (θ : ℝ) 
  (h1 : Real.sin θ + Real.cos θ = 1/5)
  (h2 : θ ∈ Set.Ioo 0 π) :
  Real.tan θ = -4/3 ∧ (1 - 2*Real.sin θ*Real.cos θ) / (Real.cos θ^2 - Real.sin θ^2) = -7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l311_31129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wednesday_harvest_l311_31193

/-- The amount of tomatoes harvested on Wednesday -/
def wednesday : ℝ := sorry

/-- The amount of tomatoes harvested on Thursday -/
def thursday : ℝ := sorry

/-- The amount of tomatoes harvested on Friday -/
def friday : ℝ := sorry

/-- Thursday's harvest is half of Wednesday's -/
axiom thursday_half_wednesday : thursday = wednesday / 2

/-- Total harvest over three days is 2000 kg -/
axiom total_harvest : wednesday + thursday + friday = 2000

/-- 700 kg remaining from Friday's harvest after giving away 700 kg -/
axiom friday_remainder : friday - 700 = 700

theorem wednesday_harvest : wednesday = 400 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wednesday_harvest_l311_31193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_general_term_l311_31185

def a : ℕ → ℤ
  | 0 => -1
  | n + 1 => 2 * a n + (3 * (n + 1) - 1) * 3^(n + 2)

theorem a_general_term (n : ℕ) :
  a n = 31 * 2^n + (3 * n - 10) * 3^(n + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_general_term_l311_31185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_average_age_l311_31166

theorem student_average_age 
  (num_students : ℕ) 
  (teacher_age : ℕ) 
  (total_average : ℚ) 
  (h1 : num_students = 30)
  (h2 : teacher_age = 45)
  (h3 : total_average = 15) :
  (num_students * total_average - teacher_age) / num_students = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_average_age_l311_31166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficiency_not_necessity_l311_31120

def is_geometric_sequence (s : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, s (n + 1) = r * s n

theorem sufficiency_not_necessity
  (a : ℕ → ℝ)
  (h1 : a 1 = 1)
  (h_p : ∀ n : ℕ, a (n + 1) + a n = 2^(n + 1) + 2^n)
  (h_q : is_geometric_sequence (λ n ↦ a n - 2^n)) :
  (∀ n : ℕ, a (n + 1) + a n = 2^(n + 1) + 2^n → is_geometric_sequence (λ n ↦ a n - 2^n)) ∧
  (∃ a : ℕ → ℝ, is_geometric_sequence (λ n ↦ a n - 2^n) ∧ ∃ n : ℕ, a (n + 1) + a n ≠ 2^(n + 1) + 2^n) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficiency_not_necessity_l311_31120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_of_eighth_terms_l311_31175

-- Define arithmetic sequences a_n and b_n
def a : ℕ → ℚ := sorry
def b : ℕ → ℚ := sorry

-- Define sum of first n terms for each sequence
def S (n : ℕ) : ℚ := (n : ℚ) / 2 * (a 1 + a n)
def T (n : ℕ) : ℚ := (n : ℚ) / 2 * (b 1 + b n)

-- State the theorem
theorem ratio_of_eighth_terms (h : ∀ n : ℕ, S n / T n = (7 * n + 3 : ℚ) / (n + 3 : ℚ)) :
  a 8 / b 8 = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_of_eighth_terms_l311_31175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_age_l311_31158

theorem teacher_age (num_students : ℕ) (student_avg_age : ℝ) (total_avg_age : ℝ) (teacher_age : ℝ) :
  num_students = 40 →
  student_avg_age = 15 →
  total_avg_age = 16 →
  (num_students : ℝ) * student_avg_age + teacher_age = (num_students + 1 : ℝ) * total_avg_age →
  teacher_age = 56 :=
by
  sorry

#check teacher_age

end NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_age_l311_31158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_star_result_l311_31186

def star (x y : ℝ) : ℝ := x * y + 4 * y - 3 * x

def nested_star : ℕ → ℝ
  | 0 => 2022
  | (n + 1) => star (nested_star n) (2023 - (n + 1))

theorem nested_star_result : nested_star 2022 = 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_star_result_l311_31186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_per_annum_is_twelve_l311_31125

/-- Represents the banker's gain in Rupees -/
noncomputable def bankers_gain : ℝ := 6.6

/-- Represents the true discount in Rupees -/
noncomputable def true_discount : ℝ := 55

/-- Represents the time period in years -/
noncomputable def time : ℝ := 1

/-- Calculates the banker's discount given the face value and rate -/
noncomputable def bankers_discount (face_value : ℝ) (rate : ℝ) : ℝ :=
  (face_value * rate * time) / 100

/-- Calculates the true discount given the face value and rate -/
noncomputable def calculate_true_discount (face_value : ℝ) (rate : ℝ) : ℝ :=
  (face_value * rate * time) / (100 + rate)

/-- Theorem stating that the percentage per annum is 12% -/
theorem percentage_per_annum_is_twelve : ∃ (face_value : ℝ),
  bankers_discount face_value 12 - calculate_true_discount face_value 12 = bankers_gain ∧
  calculate_true_discount face_value 12 = true_discount := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_per_annum_is_twelve_l311_31125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangements_A_not_first_count_arrangements_A_not_first_B_not_last_count_l311_31122

/-- The number of singers in the performance. -/
def n : ℕ := 5

/-- The factorial function. -/
def factorial (n : ℕ) : ℕ := Nat.factorial n

/-- The number of arrangements where Singer A is not the first to perform. -/
def arrangements_A_not_first : ℕ := n * factorial (n - 1)

/-- The number of arrangements where Singer A is not the first and Singer B is not the last to perform. -/
def arrangements_A_not_first_B_not_last : ℕ := 
  factorial n - 2 * factorial (n - 1) + factorial (n - 2)

/-- Theorem stating that the number of arrangements where Singer A is not the first to perform is 96. -/
theorem arrangements_A_not_first_count : arrangements_A_not_first = 96 := by
  -- Proof goes here
  sorry

/-- Theorem stating that the number of arrangements where Singer A is not the first and Singer B is not the last to perform is 78. -/
theorem arrangements_A_not_first_B_not_last_count : arrangements_A_not_first_B_not_last = 78 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangements_A_not_first_count_arrangements_A_not_first_B_not_last_count_l311_31122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_growth_l311_31102

def sequence_a (a₀ a₁ : ℕ+) : ℕ → ℕ
  | 0 => a₀
  | 1 => a₁
  | n + 2 => 3 * sequence_a a₀ a₁ (n + 1) - 2 * sequence_a a₀ a₁ n

theorem sequence_growth (a₀ a₁ : ℕ+) (h : a₁ > a₀) :
  sequence_a a₀ a₁ 100 > 2^99 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_growth_l311_31102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_passing_rate_is_70_percent_l311_31155

/-- Represents the percentage of students who answered a question correctly -/
def CorrectPercentage := Fin 101

/-- Represents the exam with 5 questions -/
structure Exam where
  q1 : CorrectPercentage
  q2 : CorrectPercentage
  q3 : CorrectPercentage
  q4 : CorrectPercentage
  q5 : CorrectPercentage

/-- Calculates the minimum passing rate for the exam -/
def minimumPassingRate (e : Exam) : ℚ :=
  sorry

/-- The given exam -/
def givenExam : Exam where
  q1 := ⟨81, by norm_num⟩
  q2 := ⟨91, by norm_num⟩
  q3 := ⟨85, by norm_num⟩
  q4 := ⟨79, by norm_num⟩
  q5 := ⟨74, by norm_num⟩

/-- Theorem stating that the minimum passing rate for the given exam is 70% -/
theorem minimum_passing_rate_is_70_percent :
  minimumPassingRate givenExam = 70 / 100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_passing_rate_is_70_percent_l311_31155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_length_P1P2_l311_31126

open Real

theorem max_length_P1P2 (θ : ℝ) (h1 : 0 ≤ θ) (h2 : θ < 2 * π) : 
  ∃ (max_length : ℝ), max_length = 3 * sqrt 2 ∧ 
    ∀ θ', 0 ≤ θ' ∧ θ' < 2 * π → 
      sqrt ((2 + sin θ' - cos θ')^2 + (2 - cos θ' - sin θ')^2) ≤ max_length :=
by
  -- Define max_length
  let max_length := 3 * sqrt 2

  -- Prove existence
  use max_length

  -- Split into two goals
  constructor

  -- Prove equality
  · rfl

  -- Prove inequality for all θ'
  · intro θ' hθ'
    -- The main proof goes here
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_length_P1P2_l311_31126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_are_coplanar_l311_31164

/-- Vector a in ℝ³ -/
def a : Fin 3 → ℝ := ![1, -2, 1]

/-- Vector b in ℝ³ -/
def b : Fin 3 → ℝ := ![3, 1, -2]

/-- Vector c in ℝ³ -/
def c : Fin 3 → ℝ := ![7, 14, -13]

/-- Theorem stating that vectors a, b, and c are coplanar -/
theorem vectors_are_coplanar : Matrix.det (Matrix.of ![a, b, c]) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_are_coplanar_l311_31164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_is_60_degrees_l311_31194

-- Define a triangle with side lengths a, b, and c
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c

-- Define the condition given in the problem
def satisfies_condition (t : Triangle) : Prop :=
  t.c^2 / (t.a + t.b) + t.a^2 / (t.b + t.c) = t.b

-- Define the angle B
noncomputable def angle_B (t : Triangle) : ℝ :=
  Real.arccos ((t.a^2 + t.c^2 - t.b^2) / (2 * t.a * t.c))

-- State the theorem
theorem angle_B_is_60_degrees (t : Triangle) (h : satisfies_condition t) :
  angle_B t = π / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_is_60_degrees_l311_31194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linda_mark_difference_l311_31157

/-- Represents the vacation cost-sharing scenario -/
structure VacationCosts where
  linda_paid : ℚ
  mark_paid : ℚ
  jane_paid : ℚ
  kyle_paid : ℚ
  total_people : ℕ

/-- Calculates the total amount paid by all participants -/
def total_paid (costs : VacationCosts) : ℚ :=
  costs.linda_paid + costs.mark_paid + costs.jane_paid + costs.kyle_paid

/-- Calculates the equal share each person should pay -/
def equal_share (costs : VacationCosts) : ℚ :=
  (total_paid costs) / costs.total_people

/-- Calculates the amount a person needs to pay or receive to equalize costs -/
def amount_to_equalize (paid : ℚ) (costs : VacationCosts) : ℚ :=
  equal_share costs - paid

/-- Theorem stating the difference between Linda's and Mark's equalization amounts -/
theorem linda_mark_difference (costs : VacationCosts) 
  (h1 : costs.linda_paid = 150)
  (h2 : costs.mark_paid = 180)
  (h3 : costs.jane_paid = 210)
  (h4 : costs.kyle_paid = 240)
  (h5 : costs.total_people = 4) :
  amount_to_equalize costs.linda_paid costs - amount_to_equalize costs.mark_paid costs = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linda_mark_difference_l311_31157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l311_31150

theorem sin_alpha_value (α : ℝ) 
  (h1 : π / 4 < α) 
  (h2 : α < π) 
  (h3 : Real.cos (α - π / 4) = 3 / 5) : 
  Real.sin α = 7 * Real.sqrt 2 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l311_31150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_open_interval_l311_31133

noncomputable def f (x : ℝ) := (1/3)^x - x

theorem zero_in_open_interval :
  ∃! z, z ∈ Set.Ioo 0 1 ∧ f z = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_open_interval_l311_31133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boris_behind_l311_31199

/-- Represents the race between Boris and Vasya -/
structure Race where
  distance : ℝ
  vasyaSpeed : ℝ
  borisSpeed : ℝ
  vasyaSlowdown : ℝ
  borisSlowdown : ℝ
  borisDelay : ℝ
  borisStop : ℝ

/-- Calculates the time taken by Vasya to complete the race -/
noncomputable def vasyaTime (race : Race) : ℝ :=
  (race.distance / 2) / race.vasyaSpeed + (race.distance / 2) / (race.vasyaSpeed * race.vasyaSlowdown)

/-- Calculates the time taken by Boris to complete the race -/
noncomputable def borisTime (race : Race) : ℝ :=
  race.borisDelay + (race.distance / 2) / race.borisSpeed + race.borisStop + 
  (race.distance / 2) / (race.borisSpeed * race.borisSlowdown)

/-- Theorem stating that Boris finishes at least one hour after Vasya -/
theorem boris_behind (race : Race) 
  (h1 : race.borisSpeed = 10 * race.vasyaSpeed)
  (h2 : race.borisDelay = 1)
  (h3 : race.vasyaSlowdown = 1/2)
  (h4 : race.borisStop = 4)
  (h5 : race.borisSlowdown = 1/2)
  (h6 : race.distance > 0)
  (h7 : race.vasyaSpeed > 0) :
  borisTime race - vasyaTime race ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boris_behind_l311_31199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_subset_with_difference_eight_l311_31127

def S : Set ℕ := {n | 1 ≤ n ∧ n ≤ 20}

theorem smallest_subset_with_difference_eight :
  ∃ n : ℕ, n = 9 ∧
  (∀ A : Finset ℕ, A.toSet ⊆ S → A.card ≥ n →
    ∃ x y : ℕ, x ∈ A ∧ y ∈ A ∧ x ≠ y ∧ (x - y = 8 ∨ y - x = 8)) ∧
  (∀ m : ℕ, m < n →
    ∃ B : Finset ℕ, B.toSet ⊆ S ∧ B.card = m ∧
    ∀ x y : ℕ, x ∈ B → y ∈ B → x ≠ y → x - y ≠ 8 ∧ y - x ≠ 8) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_subset_with_difference_eight_l311_31127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_approx_twelve_percent_l311_31132

/-- The percentage discount from the list price to achieve the desired profit -/
noncomputable def discount_percentage (cost_price list_price : ℝ) (profit_percentage : ℝ) : ℝ :=
  let desired_selling_price := cost_price * (1 + profit_percentage)
  (list_price - desired_selling_price) / list_price * 100

/-- Theorem stating that the discount percentage is approximately 12% -/
theorem discount_approx_twelve_percent :
  let cost_price := (47.50 : ℝ)
  let list_price := (67.47 : ℝ)
  let profit_percentage := (0.25 : ℝ)
  ∃ ε > 0, abs (discount_percentage cost_price list_price profit_percentage - 12) < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_approx_twelve_percent_l311_31132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_journey_distance_l311_31100

/-- Represents the battery discharge rate for an activity in units per hour -/
noncomputable def DischargeRate (hours : ℝ) : ℝ := 1 / hours

/-- Represents the total travel time in hours -/
noncomputable def TotalTime (videoRate : ℝ) (tetrisRate : ℝ) : ℝ :=
  2 / (videoRate + tetrisRate)

/-- Represents the total distance traveled by the train in km -/
noncomputable def TotalDistance (time : ℝ) (speed1 : ℝ) (speed2 : ℝ) : ℝ :=
  (time / 2) * speed1 + (time / 2) * speed2

theorem train_journey_distance :
  let videoRate := DischargeRate 3
  let tetrisRate := DischargeRate 5
  let time := TotalTime videoRate tetrisRate
  let distance := TotalDistance time 80 60
  ⌊distance⌋ = 257 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_journey_distance_l311_31100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_and_tangent_circles_l311_31178

/-- Given an equilateral triangle inscribed in a circle and three internally tangent circles, 
    prove that the radius of the externally tangent circle results in m' + n' = 65 --/
theorem inscribed_triangle_and_tangent_circles (P Q R S T : ℝ → ℝ → Prop) 
  (U : ℝ → ℝ → ℝ → Prop) (m' n' : ℕ) :
  (∀ x y z, U x y z → (x - y)^2 + (y - z)^2 + (z - x)^2 = 432) →  -- U is equilateral with side length 12√3
  (∀ x y, P x y ↔ (x^2 + y^2 = 144)) →  -- P has radius 12
  (∃ a b, U a b b ∧ (∀ x y, Q x y ↔ ((x - a)^2 + (y - b)^2 = 16))) →  -- Q has radius 4 and is tangent to P at a vertex of U
  (∃ c d e f, U c d e ∧ U c e f ∧ 
    (∀ x y, R x y ↔ ((x - d)^2 + (y - e)^2 = 9)) ∧ 
    (∀ x y, S x y ↔ ((x - e)^2 + (y - f)^2 = 9))) →  -- R and S have radius 3 and are tangent to P at other vertices of U
  (∀ x y, T x y ↔ (x^2 + y^2 = (m' / n' : ℚ)^2)) →  -- T has radius m'/n'
  (∃ x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄, 
    Q x₁ y₁ ∧ T x₁ y₁ ∧ R x₂ y₂ ∧ T x₂ y₂ ∧ S x₃ y₃ ∧ T x₃ y₃ ∧ P x₄ y₄ ∧ T x₄ y₄) →  -- Q, R, S are externally tangent to T, and T is internally tangent to P
  Nat.Coprime m' n' →
  m' + n' = 65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_and_tangent_circles_l311_31178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_sinusoid_multiple_l311_31183

theorem integer_sinusoid_multiple (a b n : ℕ) (θ : ℝ) 
  (h1 : a > b) 
  (h2 : 0 < θ) (h3 : θ < Real.pi / 2) 
  (h4 : Real.sin θ = (2 * a * b : ℝ) / ((a^2 + b^2) : ℝ)) : 
  ∃ k : ℤ, ((a^2 + b^2)^n : ℝ) * Real.sin θ = k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_sinusoid_multiple_l311_31183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_PAB_l311_31134

-- Define the curves and line
noncomputable def curve_C1 (α : ℝ) : ℝ × ℝ := (2 + Real.sqrt 7 * Real.cos α, Real.sqrt 7 * Real.sin α)

noncomputable def curve_C2 (θ : ℝ) : ℝ := 8 * Real.cos θ

noncomputable def line_l : ℝ := Real.pi / 3

-- Define the intersection points
noncomputable def point_A : ℝ × ℝ := sorry

noncomputable def point_B : ℝ × ℝ := sorry

-- Define a moving point P on C2
noncomputable def point_P (θ : ℝ) : ℝ × ℝ := 
  (curve_C2 θ * Real.cos θ, curve_C2 θ * Real.sin θ)

-- Helper function for triangle area
noncomputable def area_triangle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem max_area_triangle_PAB : 
  ∃ (θ : ℝ), ∀ (θ' : ℝ), 
    area_triangle point_A point_B (point_P θ) ≤ area_triangle point_A point_B (point_P θ') →
    area_triangle point_A point_B (point_P θ) = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_PAB_l311_31134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_proposition_l311_31176

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x ∈ Set.Icc 0 (π / 4) → Real.sin x < Real.cos x)) ↔
  (∃ x : ℝ, x ∈ Set.Icc 0 (π / 4) ∧ Real.sin x ≥ Real.cos x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_proposition_l311_31176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_three_l311_31189

theorem power_of_three (x : ℝ) (h : (3 : ℝ)^x = 5) : (3 : ℝ)^(x+2) = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_three_l311_31189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_values_for_m_l311_31117

theorem two_values_for_m : 
  ∃! (S : Finset ℕ), 
    (∀ m ∈ S, m > 0 ∧ ∃ k : ℕ, k > 0 ∧ 990 = k * (m^2 - 2)) ∧ 
    S.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_values_for_m_l311_31117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ellipse_to_point_l311_31140

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 / 8 + y^2 / 2 = 1

/-- The distance between two points -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

/-- The minimum distance between a point on the ellipse and A(1, 0) -/
theorem min_distance_ellipse_to_point :
  ∃ (x y : ℝ), ellipse x y ∧
  (∀ (x' y' : ℝ), ellipse x' y' →
    distance x y 1 0 ≤ distance x' y' 1 0) ∧
  distance x y 1 0 = Real.sqrt 15 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ellipse_to_point_l311_31140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_inverse_is_correct_l311_31192

noncomputable def f (x : ℝ) : ℝ := 4 * x + 5

noncomputable def g (x : ℝ) : ℝ := 3 * x - 4

noncomputable def h (x : ℝ) : ℝ := f (g x)

noncomputable def h_inverse (x : ℝ) : ℝ := (x + 11) / 12

theorem h_inverse_is_correct : Function.LeftInverse h_inverse h ∧ Function.RightInverse h_inverse h := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_inverse_is_correct_l311_31192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_vector_sum_l311_31163

/-- Circle C₁ with center at origin and radius 1 -/
def C₁ : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}

/-- Circle C₂ with center at (3, 4) and radius 1 -/
def C₂ : Set (ℝ × ℝ) := {p | (p.1 - 3)^2 + (p.2 - 4)^2 = 1}

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Statement of the problem -/
theorem range_of_vector_sum (A B : C₁) (P : C₂) (h : distance A B = Real.sqrt 3) :
  ∃ (x : ℝ), x ∈ Set.Icc 7 13 ∧ 
  (∀ (A' B' : C₁) (P' : C₂), distance A' B' = Real.sqrt 3 → 
    distance P' A' + distance P' B' ≤ x) ∧
  (∀ ε > 0, ∃ (A' B' : C₁) (P' : C₂), distance A' B' = Real.sqrt 3 ∧ 
    distance P' A' + distance P' B' > x - ε) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_vector_sum_l311_31163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_approx_l311_31141

noncomputable def smaller_square_perimeter : ℝ := 4
noncomputable def larger_square_area : ℝ := 16

noncomputable def smaller_square_side : ℝ := smaller_square_perimeter / 4
noncomputable def larger_square_side : ℝ := Real.sqrt larger_square_area

noncomputable def horizontal_distance : ℝ := larger_square_side + smaller_square_side
noncomputable def vertical_distance : ℝ := larger_square_side - smaller_square_side

noncomputable def distance_AB : ℝ := Real.sqrt (horizontal_distance ^ 2 + vertical_distance ^ 2)

theorem distance_AB_approx :
  (⌊distance_AB * 10 + 0.5⌋ : ℝ) / 10 = 5.8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_approx_l311_31141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l311_31153

noncomputable def f (x : ℝ) : ℝ := Real.sin (x + 7 * Real.pi / 4) + Real.cos (x - 3 * Real.pi / 4)

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ p = 2 * Real.pi ∧ ∀ (x : ℝ), f (x + p) = f x) ∧
  (∀ (x : ℝ), f ((-Real.pi / 4) - x) = f ((-Real.pi / 4) + x)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l311_31153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tate_education_duration_l311_31195

/-- Represents the duration of Tate's education -/
structure TateEducation where
  high_school : ℕ
  college : ℕ
  normal_high_school : ℕ

/-- The conditions of Tate's education -/
def TateConditions (e : TateEducation) : Prop :=
  e.normal_high_school = 4 ∧
  e.high_school = e.normal_high_school - 1 ∧
  e.college = 3 * e.high_school

/-- The theorem stating the total duration of Tate's education -/
theorem tate_education_duration (e : TateEducation) 
  (h : TateConditions e) : e.high_school + e.college = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tate_education_duration_l311_31195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l311_31171

noncomputable def a : ℝ × ℝ := (1, Real.sqrt 3)
def b (m : ℝ) : ℝ × ℝ := (2, m)

-- Helper function for dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Helper function for vector scaling
def scale (c : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (c * v.1, c * v.2)

-- Helper function for vector subtraction
def subtract (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 - w.1, v.2 - w.2)

-- Helper function for vector magnitude
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- Helper function for angle between vectors
noncomputable def angle (v w : ℝ × ℝ) : ℝ := 
  Real.arccos (dot_product v w / (magnitude v * magnitude w))

theorem vector_problem (m : ℝ) :
  ((dot_product (subtract (scale 3 a) (scale 2 (b m))) a = 0) → m = (4 * Real.sqrt 3) / 3) ∧
  ((angle a (b m) = 2 * π / 3) → m = Real.sqrt 3) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l311_31171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_tetrahedron_characterization_l311_31114

structure Pyramid where
  base : RegularTriangle
  apex : Point3D

class RegularTriangle where
  -- Define properties of a regular triangle

class Point3D where
  -- Define properties of a 3D point

def Pyramid.isRegularTetrahedron (p : Pyramid) : Prop :=
  sorry -- Definition of regular tetrahedron

def Pyramid.lateralEdgesEqual (p : Pyramid) : Prop :=
  sorry -- Definition of equal lateral edges

def Pyramid.lateralEdgeBaseAnglesEqual (p : Pyramid) : Prop :=
  sorry -- Definition of equal angles between lateral edges and base

theorem regular_tetrahedron_characterization (p : Pyramid) :
  p.isRegularTetrahedron ↔ (p.lateralEdgesEqual ∨ p.lateralEdgeBaseAnglesEqual) :=
by
  sorry -- Proof goes here

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_tetrahedron_characterization_l311_31114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_multiples_eq_45_l311_31173

/-- The count of positive integers not exceeding 150 that are multiples of 3 or 5 but not 6 -/
def count_multiples : ℕ :=
  (Finset.filter (λ n : ℕ ↦ n ≤ 150 ∧ (n % 3 = 0 ∨ n % 5 = 0) ∧ n % 6 ≠ 0) (Finset.range 151)).card

theorem count_multiples_eq_45 : count_multiples = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_multiples_eq_45_l311_31173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_at_minus_two_l311_31124

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the limit condition
def limit_condition (f : ℝ → ℝ) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x| ∧ |x| < δ → 
    |((f (-2 + x) - f (-2 - x)) / x) + 2| < ε

-- State the theorem
theorem derivative_at_minus_two (f : ℝ → ℝ) :
  limit_condition f → HasDerivAt f (-1) (-2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_at_minus_two_l311_31124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_is_half_l311_31118

/-- Represents a tetrahedron ABCD with specific properties --/
structure Tetrahedron where
  ab : ℝ
  cd : ℝ
  distance : ℝ
  angle : ℝ
  ab_eq : ab = 1
  cd_eq : cd = Real.sqrt 3
  distance_eq : distance = 2
  angle_eq : angle = π / 3

/-- Calculates the volume of the tetrahedron --/
noncomputable def tetrahedron_volume (t : Tetrahedron) : ℝ :=
  1 / 2

/-- Theorem stating that the volume of the tetrahedron with given properties is 1/2 --/
theorem tetrahedron_volume_is_half (t : Tetrahedron) :
  tetrahedron_volume t = 1 / 2 := by
  sorry

#check tetrahedron_volume_is_half

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_is_half_l311_31118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_transformation_l311_31128

/-- Represents a sequence of letters A and B -/
def Sequence := List Char

/-- Represents an operation on a sequence -/
inductive Operation
| insert : Char → Nat → Nat → Operation  -- insert n copies of a char at position i
| delete : Nat → Nat → Operation         -- delete n consecutive chars starting at position i

/-- Applies an operation to a sequence -/
def applyOperation (s : Sequence) (op : Operation) : Sequence :=
  match op with
  | Operation.insert c i n => (s.take i) ++ (List.replicate n c) ++ (s.drop i)
  | Operation.delete i n => (s.take i) ++ (s.drop (i + n))

/-- Checks if a sequence contains only A and B -/
def isValidSequence (s : Sequence) : Prop :=
  s.all (fun c => c = 'A' ∨ c = 'B')

theorem sequence_transformation (s1 s2 : Sequence) 
  (h1 : s1.length = 100)
  (h2 : s2.length = 100)
  (h3 : isValidSequence s1)
  (h4 : isValidSequence s2) :
  ∃ (ops : List Operation), 
    ops.length ≤ 100 ∧ 
    (ops.foldl applyOperation s1 = s2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_transformation_l311_31128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pirate_treasure_probability_l311_31121

def num_islands : ℕ := 8
def num_treasure_islands : ℕ := 4

def prob_treasure : ℚ := 1/3
def prob_trap : ℚ := 1/6
def prob_neither : ℚ := 1/2

theorem pirate_treasure_probability :
  (Nat.choose num_islands num_treasure_islands : ℚ) *
  prob_treasure ^ num_treasure_islands *
  prob_neither ^ (num_islands - num_treasure_islands) =
  35/648 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pirate_treasure_probability_l311_31121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_prime_or_power_of_two_l311_31172

theorem arithmetic_sequence_prime_or_power_of_two (n : ℕ) 
  (h_n : n > 6) 
  (coprime_seq : List ℕ) 
  (h_coprime : ∀ x ∈ coprime_seq, x < n ∧ Nat.Coprime x n) 
  (h_all : ∀ x < n, Nat.Coprime x n → x ∈ coprime_seq) 
  (h_arithmetic : ∃ d > 0, ∀ i, i + 1 < coprime_seq.length → 
    coprime_seq[i+1]! - coprime_seq[i]! = d) : 
  Nat.Prime n ∨ ∃ k : ℕ, n = 2^k := by
  sorry

#check arithmetic_sequence_prime_or_power_of_two

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_prime_or_power_of_two_l311_31172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l311_31170

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2) / Real.log 10

theorem f_monotone_decreasing :
  ∀ x y, x < y → x < 0 → y < 0 → f x ≥ f y :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l311_31170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_of_extrema_l311_31109

noncomputable def f (x : ℝ) := Real.cos (Real.pi / 2 * x - Real.pi / 6)

theorem dot_product_of_extrema (A B : ℝ × ℝ) :
  (∃ (x_A : ℝ), x_A > 0 ∧ A = (x_A, f x_A) ∧ 
    (∀ (x : ℝ), x > 0 → f x ≤ f x_A) ∧
    (∀ (y : ℝ), 0 < y ∧ y < x_A → f y < f x_A)) →
  (∃ (x_B : ℝ), x_B > A.1 ∧ B = (x_B, f x_B) ∧ 
    (∀ (x : ℝ), x > 0 → f x ≥ f x_B) ∧
    (∀ (y : ℝ), A.1 < y ∧ y < x_B → f y > f x_B)) →
  A.1 * B.1 + A.2 * B.2 = -2/9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_of_extrema_l311_31109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_rounded_to_nearest_dollar_l311_31160

noncomputable def purchase1 : ℝ := 2.49
noncomputable def purchase2 : ℝ := 7.23
noncomputable def purchase3 : ℝ := 4.88

noncomputable def round_to_nearest_dollar (x : ℝ) : ℤ :=
  round x

theorem total_rounded_to_nearest_dollar :
  round_to_nearest_dollar (purchase1 + purchase2 + purchase3) = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_rounded_to_nearest_dollar_l311_31160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_train_length_l311_31184

/-- Calculates the length of a train given the speeds of two trains, the time they take to clear each other, and the length of the other train. -/
noncomputable def calculate_train_length (speed1 speed2 : ℝ) (clear_time : ℝ) (other_train_length : ℝ) : ℝ :=
  let relative_speed := (speed1 + speed2) * (1000 / 3600)
  let total_distance := relative_speed * clear_time
  total_distance - other_train_length

/-- The length of the first train is 125 meters. -/
theorem first_train_length :
  let speed1 := (80 : ℝ)
  let speed2 := (65 : ℝ)
  let clear_time := (7.199424046076314 : ℝ)
  let second_train_length := (165 : ℝ)
  calculate_train_length speed1 speed2 clear_time second_train_length = 125 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_train_length_l311_31184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_to_plane_1_line_parallel_to_plane_2_l311_31196

-- Define the necessary structures
structure Point : Type
structure Line : Type
structure Plane : Type

-- Define the parallel and subset relations
axiom parallel : Line → Line → Prop
axiom parallel_line_plane : Line → Plane → Prop
axiom subset : Line → Plane → Prop
axiom intersect : Plane → Plane → Line

-- Theorem 1
theorem line_parallel_to_plane_1 
  (a b c : Line) (α : Plane) 
  (h1 : parallel a b) 
  (h2 : parallel b c) 
  (h3 : subset c α) 
  (h4 : ¬subset a α) : 
  parallel_line_plane a α :=
sorry

-- Theorem 2
theorem line_parallel_to_plane_2 
  (a b : Line) (α β : Plane)
  (h1 : intersect α β = b)
  (h2 : parallel a b)
  (h3 : subset a β) :
  parallel_line_plane a α :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_to_plane_1_line_parallel_to_plane_2_l311_31196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matt_problem_l311_31112

noncomputable def matt_sells_three_kittens (initial_cats : ℕ) (female_ratio : ℚ) 
  (kittens_per_female : ℕ) (remaining_kitten_ratio : ℚ) : ℕ :=
  let female_cats := (initial_cats : ℚ) * female_ratio
  let total_kittens := (female_cats * kittens_per_female).floor
  let total_cats := initial_cats + total_kittens
  let remaining_kittens := (total_cats : ℚ) * remaining_kitten_ratio
  (total_kittens - remaining_kittens.floor).natAbs

-- Main theorem
theorem matt_problem :
  matt_sells_three_kittens 6 (1/2) 7 (67/100) = 3 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matt_problem_l311_31112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_more_than_half_one_less_flag_l311_31106

/-- Represents the number of children with equal flags in both hands -/
def E : ℕ := sorry

/-- Represents the number of children with unequal flags in both hands -/
def N : ℕ := sorry

/-- Represents the total number of children -/
def T : ℕ := sorry

/-- Represents the number of children who initially had exactly one less flag in one hand -/
def k : ℕ := sorry

/-- The initial condition where E is one-fifth of N -/
axiom initial_condition : E = N / 5

/-- The condition after moving one flag, where E' is half of N' -/
axiom after_move_condition : E = N / 2

/-- The total number of children is the sum of those with equal and unequal flags -/
axiom total_children : T = E + N

/-- k is less than or equal to half of the total children -/
axiom k_constraint : k ≤ T / 2

/-- Theorem stating it's impossible for more than half of the children to initially have exactly one less flag in one hand -/
theorem not_more_than_half_one_less_flag : k ≤ T / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_more_than_half_one_less_flag_l311_31106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_roots_sum_l311_31162

theorem tan_roots_sum (α β : Real) (h1 : α ∈ Set.Ioo (-Real.pi/2) (Real.pi/2)) 
  (h2 : β ∈ Set.Ioo (-Real.pi/2) (Real.pi/2))
  (h3 : (Real.tan α)^2 + 3*Real.sqrt 3*(Real.tan α) + 4 = 0)
  (h4 : (Real.tan β)^2 + 3*Real.sqrt 3*(Real.tan β) + 4 = 0) :
  α + β = -2*Real.pi/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_roots_sum_l311_31162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_movie_ticket_discount_l311_31145

theorem movie_ticket_discount (evening_ticket_cost combo_cost combo_discount total_savings : ℝ) 
  (h1 : evening_ticket_cost = 10)
  (h2 : combo_cost = 10)
  (h3 : combo_discount = 0.5)
  (h4 : total_savings = 7)
  : (1 - (evening_ticket_cost + combo_cost * (1 - combo_discount) - total_savings - combo_cost * (1 - combo_discount)) / evening_ticket_cost) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_movie_ticket_discount_l311_31145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l311_31138

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 1 / (x + 2)

-- State the theorem
theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≠ -2} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l311_31138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_3920_simplification_l311_31169

theorem sqrt_3920_simplification : Real.sqrt 3920 = 28 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_3920_simplification_l311_31169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_constant_expression_l311_31113

/-- Definition of the ellipse E -/
def E : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 / 2 = 1}

/-- Definition of point P -/
noncomputable def P : ℝ × ℝ := (0, Real.sqrt 3)

/-- Definition of the dot product -/
def dot (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Definition of the expression to be proved constant -/
noncomputable def expr (M N : ℝ × ℝ) : ℝ :=
  dot M N - 7 * dot (M.1 - P.1, M.2 - P.2) (N.1 - P.1, N.2 - P.2)

theorem ellipse_constant_expression :
  ∀ M N : ℝ × ℝ,
  M ∈ E → N ∈ E →
  M ≠ (2, 0) → M ≠ (-2, 0) → M ≠ (0, Real.sqrt 2) → M ≠ (0, -Real.sqrt 2) →
  N ≠ (2, 0) → N ≠ (-2, 0) → N ≠ (0, Real.sqrt 2) → N ≠ (0, -Real.sqrt 2) →
  ∃ k : ℝ, N.2 - P.2 = k * (N.1 - P.1) →
  expr M N = -9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_constant_expression_l311_31113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_preston_sandwich_shop_revenue_l311_31148

/-- Calculate the total amount received by Preston after applying all discounts, service charge, and tip --/
theorem preston_sandwich_shop_revenue : 
  let sandwich_price : ℚ := 5
  let side_dish_price : ℚ := 3
  let drink_price : ℚ := 3/2
  let delivery_fee : ℚ := 20
  let service_charge_rate : ℚ := 1/20
  let service_charge_threshold : ℚ := 50
  let sandwiches_ordered : ℕ := 18
  let side_dishes_ordered : ℕ := 10
  let drinks_ordered : ℕ := 15
  let early_bird_discount_rate : ℚ := 1/10
  let group_discount_rate : ℚ := 1/5
  let voucher_discount_rate : ℚ := 3/20
  let tip_rate : ℚ := 1/10

  let sandwich_cost := sandwich_price * sandwiches_ordered
  let side_dish_cost := side_dish_price * side_dishes_ordered
  let drink_cost := drink_price * drinks_ordered

  let discounted_sandwich_cost := sandwich_cost * (1 - early_bird_discount_rate)
  let discounted_side_dish_cost := side_dish_cost * (1 - group_discount_rate)

  let total_before_voucher := discounted_sandwich_cost + discounted_side_dish_cost + drink_cost
  let discounted_total := total_before_voucher * (1 - voucher_discount_rate)

  let total_with_delivery := discounted_total + delivery_fee
  let service_charge := if total_with_delivery > service_charge_threshold then total_with_delivery * service_charge_rate else 0
  let total_with_service_charge := total_with_delivery + service_charge

  let tip := total_with_service_charge * tip_rate
  let final_total := total_with_service_charge + tip

  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/100 ∧ |final_total - 14827/100| < ε
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_preston_sandwich_shop_revenue_l311_31148
