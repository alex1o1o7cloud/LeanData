import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eggs_added_l994_99460

theorem eggs_added (initial_eggs : Float) (final_eggs : Float) (h1 : initial_eggs = 47.0) (h2 : final_eggs = 52) :
  final_eggs - initial_eggs = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eggs_added_l994_99460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_advance_probability_correct_l994_99491

/-- Represents the score for a single question -/
inductive QuestionScore
| Correct : QuestionScore
| Incorrect : QuestionScore

/-- Calculates the point value for a given question score -/
def pointValue (qs : QuestionScore) : Int :=
  match qs with
  | QuestionScore.Correct => 30
  | QuestionScore.Incorrect => -30

/-- Calculates the total score for three questions -/
def totalScore (q1 q2 q3 : QuestionScore) : Int :=
  pointValue q1 + pointValue q2 + pointValue q3

/-- The probability of answering a question correctly -/
noncomputable def correctProbability : ℝ := 0.8

/-- The probability of advancing to the next round -/
noncomputable def advanceProbability : ℝ := 112 / 125

theorem advance_probability_correct :
  advanceProbability =
    correctProbability ^ 3 +
    3 * correctProbability ^ 2 * (1 - correctProbability) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_advance_probability_correct_l994_99491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_l994_99471

def A : Set ℝ := {x | 3 * x - 7 > 0}

theorem complement_of_A : 
  {x : ℝ | x ≤ 7/3} = Aᶜ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_l994_99471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_acute_angles_l994_99421

/-- A tetrahedron with vertices A, B, C, and D. -/
structure Tetrahedron where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Face angle of a tetrahedron. -/
def faceAngle (t : Tetrahedron) : ℝ → ℝ → ℝ := sorry

/-- Dihedral angle of a tetrahedron. -/
def dihedralAngle (t : Tetrahedron) : ℝ → ℝ → ℝ := sorry

/-- An angle is acute if it's less than π/2. -/
def isAcute (angle : ℝ) : Prop := angle < Real.pi / 2

/-- Theorem: If all face angles of a tetrahedron are acute, then all its dihedral angles are acute. -/
theorem tetrahedron_acute_angles (t : Tetrahedron) :
  (∀ (x y : ℝ), isAcute (faceAngle t x y)) →
  (∀ (x y : ℝ), isAcute (dihedralAngle t x y)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_acute_angles_l994_99421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_necessarily_proportional_l994_99423

/-- Represents the number of triangles -/
def num_triangles : ℕ → ℕ := sorry

/-- Represents the number of small sticks required -/
def num_sticks : ℕ → ℕ := sorry

/-- States that the relationship between num_triangles and num_sticks is not necessarily proportional -/
theorem not_necessarily_proportional :
  ¬ (∀ k : ℝ, ∃ n : ℕ, ∀ m : ℕ, (num_triangles m : ℝ) = k * (num_sticks m : ℝ)) ∧
  ¬ (∀ k : ℝ, ∃ n : ℕ, ∀ m : ℕ, (num_triangles m : ℝ) * (num_sticks m : ℝ) = k) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_necessarily_proportional_l994_99423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pen_cost_calculation_l994_99427

/-- Calculate the final cost of pens with discounts and tax -/
theorem pen_cost_calculation (first_dozen_cost second_dozen_cost half_dozen_cost : ℚ)
  (first_dozen_discount second_dozen_discount : ℚ)
  (sales_tax : ℚ) :
  first_dozen_cost = 18 →
  second_dozen_cost = 16 →
  half_dozen_cost = 10 →
  first_dozen_discount = 1/10 →
  second_dozen_discount = 3/20 →
  sales_tax = 2/25 →
  let discounted_first_dozen := first_dozen_cost * (1 - first_dozen_discount)
  let discounted_second_dozen := second_dozen_cost * (1 - second_dozen_discount)
  let total_before_tax := discounted_first_dozen + discounted_second_dozen + half_dozen_cost
  let tax_amount := total_before_tax * sales_tax
  let final_cost := total_before_tax + tax_amount
  ⌊final_cost * 100⌋ / 100 = 11015/250 := by
  sorry

#check pen_cost_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pen_cost_calculation_l994_99427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_equivalence_l994_99449

theorem proposition_equivalence (a b c : ℝ) (hc : c ≠ 0) :
  (∀ a b, a < b → a * c^2 < b * c^2) ∧
  (∀ a b, a * c^2 < b * c^2 → a < b) ∧
  (∀ a b, a ≥ b → a * c^2 ≥ b * c^2) ∧
  (∀ a b, a * c^2 ≥ b * c^2 → a ≥ b) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_equivalence_l994_99449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_shaped_pgons_count_l994_99498

theorem star_shaped_pgons_count (p : ℕ) (h_prime : Nat.Prime p) (h_odd : Odd p) :
  ∃ (n : ℕ), ((1 : ℚ) / 2) * (((Nat.factorial (p - 1) + 1) / p : ℚ) + p - 4) = n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_shaped_pgons_count_l994_99498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_l994_99497

/-- The area of a parallelogram with base 30 cm and height 12 cm is 360 square centimeters. -/
theorem parallelogram_area : 30 * 12 = 360 := by
  -- Compute the result
  norm_num

#check parallelogram_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_l994_99497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cab_speed_fraction_cab_speed_is_five_sixths_l994_99424

/-- Proves that a cab walking at a reduced speed arrives 12 minutes late when its usual journey time is 60 minutes -/
theorem cab_speed_fraction (usual_time : ℚ) (delay : ℚ) (h1 : usual_time = 60) (h2 : delay = 12) :
  (usual_time / (usual_time + delay)) = 5 / 6 := by
  sorry

/-- The fraction of the cab's usual speed -/
def cab_speed_ratio (usual_time : ℚ) (delay : ℚ) : ℚ :=
  usual_time / (usual_time + delay)

/-- Proves that the cab is walking at 5/6 of its usual speed -/
theorem cab_speed_is_five_sixths (usual_time : ℚ) (delay : ℚ) 
  (h1 : usual_time = 60) (h2 : delay = 12) :
  cab_speed_ratio usual_time delay = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cab_speed_fraction_cab_speed_is_five_sixths_l994_99424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l994_99443

-- Define the arithmetic sequence
def a (n : ℕ) : ℤ := 31 - 2 * (n : ℤ)

-- Define the sum of the first n terms
def S (n : ℕ) : ℤ := n * (2 * 31 - ((n : ℤ) - 1) * 2) / 2

-- Define the sum of the first n terms of |a_n|
def T (n : ℕ) : ℤ :=
  if n ≤ 15 then 30 * (n : ℤ) - n^2
  else (n : ℤ)^2 - 30 * n + 450

theorem arithmetic_sequence_properties :
  (∀ n : ℕ, a n = 31 - 2 * (n : ℤ)) →
  a 1 = 29 →
  S 10 = S 20 →
  (∀ n : ℕ, S n = n * (2 * 31 - ((n : ℤ) - 1) * 2) / 2) →
  (∃ (n_max : ℕ), n_max = 15 ∧ 
    (∀ n : ℕ, S n ≤ S n_max) ∧
    S n_max = 225) ∧
  (∀ n : ℕ, T n = if n ≤ 15 then 30 * (n : ℤ) - n^2 else (n : ℤ)^2 - 30 * n + 450) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l994_99443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_is_linear_l994_99478

/-- Definition of a linear equation in one variable -/
def is_linear_equation (f : ℚ → ℚ) : Prop :=
  ∃ (a b : ℚ), a ≠ 0 ∧ ∀ x, f x = a * x + b

/-- The equation x - 2 = 1/3 -/
def equation (x : ℚ) : ℚ := x - 2 - 1/3

/-- Theorem: The equation x - 2 = 1/3 is a linear equation -/
theorem equation_is_linear : is_linear_equation equation := by
  use 1, -7/3
  constructor
  · exact one_ne_zero
  · intro x
    simp [equation]
    ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_is_linear_l994_99478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_E_and_max_area_l994_99440

noncomputable section

/-- The curve E formed by the locus of common points of two circles -/
def curve_E (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

/-- The point N on the y-axis -/
def N : ℝ × ℝ := (0, 2 * Real.sqrt 3)

/-- The area of triangle ABM given the x-coordinates of A and B -/
noncomputable def area_ABM (x₁ x₂ : ℝ) : ℝ :=
  Real.sqrt 3 / 2 * Real.sqrt ((x₁ + x₂)^2 - 4 * x₁ * x₂)

theorem curve_E_and_max_area :
  ∀ (r : ℝ), 0 < r → r < 4 →
  (∀ (x y : ℝ), (x + 1)^2 + y^2 = r^2 ∧ (x - 1)^2 + y^2 = (4 - r)^2 → curve_E x y) ∧
  (∀ (x₁ y₁ x₂ y₂ : ℝ), 
    curve_E x₁ y₁ → curve_E x₂ y₂ → 
    ∃ (k : ℝ), y₁ = k * x₁ + N.2 ∧ y₂ = k * x₂ + N.2 →
    area_ABM x₁ x₂ ≤ Real.sqrt 3 / 2) ∧
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    curve_E x₁ y₁ ∧ curve_E x₂ y₂ ∧
    ∃ (k : ℝ), y₁ = k * x₁ + N.2 ∧ y₂ = k * x₂ + N.2 ∧
    area_ABM x₁ x₂ = Real.sqrt 3 / 2) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_E_and_max_area_l994_99440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l994_99412

def f (x : ℝ) : ℝ := abs x

theorem problem_solution (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∀ x, f x + f (x - 1) ≤ 2 ↔ x ∈ Set.Icc (-1/2) (3/2)) ∧
  (∀ m, (∀ x, f (x - m) - abs (x + 2) ≤ 1/a + 1/b) ↔ m ∈ Set.Icc (-6) 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l994_99412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arctan_sum_of_cubic_roots_l994_99425

theorem arctan_sum_of_cubic_roots (x₁ x₂ x₃ : ℝ) : 
  x₁^3 - 16 * x₁ + 17 = 0 → 
  x₂^3 - 16 * x₂ + 17 = 0 → 
  x₃^3 - 16 * x₃ + 17 = 0 → 
  x₁ ≠ x₂ → x₁ ≠ x₃ → x₂ ≠ x₃ →
  Real.arctan x₁ + Real.arctan x₂ + Real.arctan x₃ = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arctan_sum_of_cubic_roots_l994_99425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_orthocenter_l994_99492

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the side length (marked as noncomputable due to Real.sqrt)
noncomputable def side_length (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the orthocenter (implementation not needed for the statement)
noncomputable def orthocenter (t : Triangle) : ℝ × ℝ :=
  sorry

-- Theorem statement
theorem distance_to_orthocenter (t : Triangle) :
  side_length t.A t.B = 2 →
  side_length t.A t.C = 5 →
  side_length t.B t.C = 6 →
  side_length t.B (orthocenter t) = 50 / Real.sqrt 39 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_orthocenter_l994_99492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_pyramid_sphere_radii_l994_99433

/-- A triangular pyramid with pairwise perpendicular lateral edges -/
structure TriangularPyramid where
  S : ℝ  -- Area of lateral face
  P : ℝ  -- Area of lateral face
  Q : ℝ  -- Area of lateral face
  S_pos : S > 0
  P_pos : P > 0
  Q_pos : Q > 0

/-- The radius of the inscribed sphere in the triangular pyramid -/
noncomputable def inscribed_sphere_radius (tp : TriangularPyramid) : ℝ :=
  Real.sqrt (2 * tp.S * tp.P * tp.Q) / (tp.S + tp.P + tp.Q + Real.sqrt (tp.S^2 + tp.P^2 + tp.Q^2))

/-- The radius of the sphere touching the base and extensions of lateral faces -/
noncomputable def touching_sphere_radius (tp : TriangularPyramid) : ℝ :=
  Real.sqrt (2 * tp.S * tp.P * tp.Q) / (tp.S + tp.P + tp.Q - Real.sqrt (tp.S^2 + tp.P^2 + tp.Q^2))

theorem triangular_pyramid_sphere_radii (tp : TriangularPyramid) : 
  (inscribed_sphere_radius tp = Real.sqrt (2 * tp.S * tp.P * tp.Q) / 
    (tp.S + tp.P + tp.Q + Real.sqrt (tp.S^2 + tp.P^2 + tp.Q^2))) ∧
  (touching_sphere_radius tp = Real.sqrt (2 * tp.S * tp.P * tp.Q) / 
    (tp.S + tp.P + tp.Q - Real.sqrt (tp.S^2 + tp.P^2 + tp.Q^2))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_pyramid_sphere_radii_l994_99433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_value_triangle_area_l994_99422

-- Define the lines L₁, L₂, and L
def L₁ (x y : ℝ) : Prop := x + y - 1 = 0
def L₂ (x y : ℝ) : Prop := 2 * x - y + 4 = 0
def L (a x y : ℝ) : Prop := a * x - y - 2 * a + 1 = 0

-- Define the intersection point P of L₁ and L₂
def P : ℝ × ℝ := (-1, 2)

-- Theorem 1: When L passes through P, a = -1/3
theorem intersection_value (a : ℝ) : 
  L a (P.1) (P.2) → a = -1/3 := by sorry

-- Helper function to calculate the area of a triangle
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ := sorry

-- Theorem 2: When L is perpendicular to L₁, the area of the triangle is 12
theorem triangle_area (a : ℝ) :
  (∀ x y : ℝ, L₁ x y → L a x y → x = y) →  -- L perpendicular to L₁
  (∃ A B C : ℝ × ℝ, 
    L₁ A.1 A.2 ∧ L₂ A.1 A.2 ∧
    L₁ B.1 B.2 ∧ L a B.1 B.2 ∧
    L₂ C.1 C.2 ∧ L a C.1 C.2 ∧
    area_triangle A B C = 12) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_value_triangle_area_l994_99422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_t_for_sequence_bound_l994_99403

noncomputable def a : ℕ → ℝ
  | 0 => 1
  | 1 => 1
  | (n + 2) => Real.sqrt ((n + 2 : ℝ) / 2 + a (n + 1) * a n)

theorem unique_t_for_sequence_bound :
  ∃! t : ℝ, ∀ n : ℕ, n > 0 → t * n < a n ∧ a n < t * n + 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_t_for_sequence_bound_l994_99403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kinetic_energy_constant_kinetic_energy_at_break_l994_99400

/-- A point object connected to a stationary cylinder by a rope on a frictionless surface -/
structure RopeSystem where
  m : ℝ        -- Mass of the object
  R : ℝ        -- Radius of the cylinder
  L₀ : ℝ       -- Initial length of the rope
  v₀ : ℝ       -- Initial velocity of the object
  T_max : ℝ    -- Maximum tension the rope can withstand
  h_m_pos : 0 < m
  h_R_pos : 0 < R
  h_L₀_pos : 0 < L₀
  h_v₀_pos : 0 < v₀
  h_T_max_pos : 0 < T_max

/-- The kinetic energy of the object at any point during its motion -/
noncomputable def kineticEnergy (s : RopeSystem) : ℝ := (1/2) * s.m * s.v₀^2

/-- The statement that the kinetic energy remains constant throughout the motion -/
theorem kinetic_energy_constant (s : RopeSystem) :
  kineticEnergy s = (1/2) * s.m * s.v₀^2 := by
  -- The proof is skipped for now
  sorry

/-- The main theorem: kinetic energy at rope breaking is mv₀²/2 -/
theorem kinetic_energy_at_break (s : RopeSystem) :
  kineticEnergy s = (1/2) * s.m * s.v₀^2 := by
  -- The proof is skipped for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kinetic_energy_constant_kinetic_energy_at_break_l994_99400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_to_decimal_conversion_l994_99481

/-- Converts a binary digit (0 or 1) to its decimal value -/
def binaryToDecimal (digit : Nat) : Nat :=
  if digit = 0 ∨ digit = 1 then digit else 0

/-- Calculates the decimal value of a binary digit at a given position -/
def binaryDigitValue (digit : Nat) (position : Nat) : Nat :=
  binaryToDecimal digit * 2^position

/-- The binary representation of the number as a list of digits (least significant first) -/
def binaryNumber : List Nat := [1, 0, 1, 1]

theorem binary_to_decimal_conversion :
  (List.sum (List.zipWith binaryDigitValue binaryNumber (List.range binaryNumber.length).reverse)) = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_to_decimal_conversion_l994_99481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_hyperbola_l994_99419

theorem point_on_hyperbola (s : ℝ) (h : s ≠ 0) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (((s + 2) / (s - 1))^2 / a^2) - (((s - 2) / (s + 1))^2 / b^2) = 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_hyperbola_l994_99419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_lemon_heads_unknown_l994_99496

/-- Represents the number of Lemon Heads in a package -/
def package_size : ℕ := 3

/-- Represents the number of Lemon Heads Patricia ate -/
def eaten_lemon_heads : ℕ := 15

/-- Represents the initial number of Lemon Heads Patricia had -/
def initial_lemon_heads : ℕ → ℕ := id

/-- Theorem stating the number of Lemon Heads Patricia has left -/
def lemon_heads_left (x : ℕ) : ℕ := x - eaten_lemon_heads

/-- Main theorem proving the number of Lemon Heads left is unknown -/
theorem remaining_lemon_heads_unknown : 
  ∃ (result : ℕ), ∃ (x : ℕ), result = lemon_heads_left x ∧ x ≥ eaten_lemon_heads := by
  sorry

#check remaining_lemon_heads_unknown

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_lemon_heads_unknown_l994_99496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l994_99456

noncomputable def f (x : ℝ) := (Real.log (5 - x) / Real.log 5) / Real.sqrt (x + 2)

theorem f_domain : ∀ x : ℝ, f x ≠ 0 ↔ -2 < x ∧ x < 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l994_99456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_line_equation_l994_99452

noncomputable def proj_vector (a b : ℝ) (x y : ℝ) : ℝ × ℝ :=
  let dot_product := a * x + b * y
  let norm_squared := a * a + b * b
  (dot_product / norm_squared * a, dot_product / norm_squared * b)

theorem projection_line_equation (x y : ℝ) :
  proj_vector 3 4 x y = (3, 4) → y = -3/4 * x + 25/4 := by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry

#check projection_line_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_line_equation_l994_99452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_funnel_height_l994_99479

noncomputable def cone_volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

theorem funnel_height :
  let r : ℝ := 4
  let v : ℝ := 200
  ∃ h : ℝ, v = cone_volume r h ∧ h = 375 / (8 * Real.pi) := by
  sorry

#check funnel_height

end NUMINAMATH_CALUDE_ERRORFEEDBACK_funnel_height_l994_99479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_is_10_over_11_l994_99459

def my_sequence (n : ℕ) : ℚ :=
  (-1)^(n+1) * (2*n) / (2*n + 1)

theorem fifth_term_is_10_over_11 : my_sequence 5 = 10 / 11 := by
  -- Unfold the definition of my_sequence
  unfold my_sequence
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_is_10_over_11_l994_99459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_and_zeros_l994_99429

noncomputable section

-- Define the functions
def f (x : ℝ) : ℝ := Real.exp x * (2 * x - 1)
def g (x : ℝ) : ℝ := x - 1
noncomputable def h (x : ℝ) : ℝ := f x / g x
def φ (a : ℝ) (x : ℝ) : ℝ := 2 * x * Real.exp x - Real.exp x - a * x + a

-- State the theorem
theorem monotonicity_and_zeros :
  (∀ x₁ x₂, x₁ < x₂ → x₂ < 0 → h x₁ < h x₂) ∧
  (∀ x₁ x₂, 1 < x₁ → x₁ < x₂ → x₂ < 3/2 → h x₁ > h x₂) ∧
  (∀ x₁ x₂, 3/2 < x₁ → x₁ < x₂ → h x₁ < h x₂) ∧
  (∀ a, (∃ x₁ x₂, x₁ ≠ x₂ ∧ φ a x₁ = 0 ∧ φ a x₂ = 0) ↔ 
    (0 < a ∧ a < 1) ∨ (a > 4 * Real.exp (3/2))) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_and_zeros_l994_99429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_after_removal_l994_99415

theorem arithmetic_mean_after_removal (initial_count : ℕ) (initial_mean : ℚ) 
  (removed_numbers : List ℚ) : 
  initial_count = 60 →
  initial_mean = 50 →
  removed_numbers = [40, 50, 55, 65] →
  let remaining_count := initial_count - removed_numbers.length
  let initial_sum := initial_count * initial_mean
  let removed_sum := removed_numbers.sum
  let remaining_sum := initial_sum - removed_sum
  let new_mean := remaining_sum / remaining_count
  (round new_mean : ℤ) = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_after_removal_l994_99415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_circle_sum_l994_99499

/-- Represents a region in the diagram --/
inductive Region
| Center
| AB
| AC
| BC
| A
| B
| C

/-- Represents a circle in the diagram --/
inductive Circle
| A
| B
| C

/-- The assignment of integers to regions --/
def Assignment := Region → Fin 7

/-- The set of regions that belong to a given circle --/
def circleRegions : Circle → List Region
  | Circle.A => [Region.Center, Region.AB, Region.AC, Region.A]
  | Circle.B => [Region.Center, Region.AB, Region.BC, Region.B]
  | Circle.C => [Region.Center, Region.AC, Region.BC, Region.C]

/-- The sum of numbers in a given circle for a given assignment --/
def circleSum (a : Assignment) (c : Circle) : Nat :=
  (circleRegions c).map (fun r => (a r).val) |>.sum

/-- An assignment is valid if it uses each number from 0 to 6 exactly once --/
def isValidAssignment (a : Assignment) : Prop :=
  ∀ n : Fin 7, ∃! r : Region, a r = n

/-- An assignment is balanced if all circles have the same sum --/
def isBalancedAssignment (a : Assignment) : Prop :=
  ∀ c1 c2 : Circle, circleSum a c1 = circleSum a c2

/-- The maximum possible sum for a valid and balanced assignment --/
theorem max_circle_sum :
  ∃ (a : Assignment), isValidAssignment a ∧ isBalancedAssignment a ∧
    ∀ (a' : Assignment), isValidAssignment a' ∧ isBalancedAssignment a' →
      circleSum a Circle.A ≥ circleSum a' Circle.A :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_circle_sum_l994_99499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rule_1_correct_rule_2_incorrect_l994_99428

-- Define the ⊕ operation
noncomputable def oplus (a b : ℝ) : ℝ := a + max a b

-- Theorem for Rule 1
theorem rule_1_correct : ∀ a b : ℝ, oplus a b = oplus b a := by
  sorry

-- Theorem for Rule 2
theorem rule_2_incorrect : ∃ a b c : ℝ, oplus a (oplus b c) ≠ oplus (oplus a b) c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rule_1_correct_rule_2_incorrect_l994_99428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fermat_prime_power_of_two_l994_99487

theorem fermat_prime_power_of_two (n : ℕ) :
  Nat.Prime (2^n + 1) → ∃ k : ℕ, n = 2^k :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fermat_prime_power_of_two_l994_99487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteenth_forty_first_415th_digit_l994_99406

/-- The decimal representation of a rational number -/
def decimal_representation (p q : ℕ) : ℕ → ℕ :=
  sorry

/-- The length of the repeating cycle in the decimal representation of a rational number -/
def cycle_length (p q : ℕ) : ℕ :=
  sorry

theorem fifteenth_forty_first_415th_digit :
  let p : ℕ := 15
  let q : ℕ := 41
  let n : ℕ := 415
  decimal_representation p q n = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteenth_forty_first_415th_digit_l994_99406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_power_product_l994_99401

theorem fraction_power_product : (1 / 3 : ℚ) ^ 5 * (2 / 5 : ℚ) ^ (-2 : ℤ) = 25 / 972 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_power_product_l994_99401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_to_asymptote_distance_l994_99446

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  a_pos : a > 0
  b_pos : b > 0

/-- The distance from a point (x, y) to the line y = mx + c -/
noncomputable def distanceToLine (x y m c : ℝ) : ℝ :=
  abs (y - m*x - c) / Real.sqrt (1 + m^2)

/-- Theorem: For a hyperbola with perpendicular asymptotes and vertex-to-asymptote distance of 1,
    the distance from a focus to an asymptote is √2 -/
theorem hyperbola_focus_to_asymptote_distance 
  (h : Hyperbola a b) 
  (perp_asymptotes : a = b) 
  (vertex_to_asymptote : distanceToLine a 0 1 0 = 1) : 
  distanceToLine (2*a) 0 1 0 = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_to_asymptote_distance_l994_99446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_processing_workshop_equation_l994_99426

theorem processing_workshop_equation :
  let total_workers : ℕ := 26
  let type_a_parts : ℕ := 2100
  let type_b_parts : ℕ := 1200
  let type_a_rate : ℕ := 30
  let type_b_rate : ℕ := 20
  ∀ x : ℝ, 0 < x → x < total_workers →
    (type_a_parts / (type_a_rate * x) = type_b_parts / (type_b_rate * (total_workers - x))) ↔
    (∃ division : ℝ → ℝ, division x = type_a_parts / (type_a_rate * x) ∧
                         division (total_workers - x) = type_b_parts / (type_b_rate * (total_workers - x))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_processing_workshop_equation_l994_99426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplified_expression_l994_99466

theorem simplified_expression :
  let x : ℝ := (Real.sqrt 3 - 1) ^ (1 - Real.sqrt 2) / (Real.sqrt 3 + 1) ^ (1 + Real.sqrt 2)
  let y : ℝ := 4 - 2 * Real.sqrt 3 * (1 / 2 ^ (1 + Real.sqrt 2))
  x = y ∧ 
  ∃ (a b c : ℕ), 
    y = a - b * Real.sqrt c ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    ∀ (p : ℕ), Nat.Prime p → ¬(c % (p * p) = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplified_expression_l994_99466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_is_geometric_l994_99417

/-- Given a sequence {a_n} with partial sums Sn satisfying lg(Sn + 1) = n,
    prove that {a_n} is a geometric sequence. -/
theorem sequence_is_geometric (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h : ∀ n : ℕ, Real.log (S n + 1) / Real.log 10 = n) : 
    ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_is_geometric_l994_99417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetrical_circle_equation_l994_99490

/-- Given two points P and Q, and a circle C, if line l is the perpendicular bisector of PQ,
    then the circle symmetrical to C with respect to l has the specified equation. -/
theorem symmetrical_circle_equation (a b : ℝ) : 
  let P : ℝ × ℝ := (a, b)
  let Q : ℝ × ℝ := (3 - b, 3 - a)
  let C : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.fst - 2)^2 + (p.snd - 3)^2 = 1}
  let l : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.fst + p.snd = 3}
  l = {p : ℝ × ℝ | (p.fst - (a + 3 - b) / 2)^2 + (p.snd - (b + 3 - a) / 2)^2 = 
           ((a - (3 - b))^2 + (b - (3 - a))^2) / 4} →
  {p : ℝ × ℝ | (p.fst - 1)^2 + (p.snd - 2)^2 = 1} = {p : ℝ × ℝ | ∃ q ∈ C, ∀ r ∈ l, 
    (p.fst - r.fst)^2 + (p.snd - r.snd)^2 = (q.fst - r.fst)^2 + (q.snd - r.snd)^2 ∧ 
    (p.fst + q.fst) / 2 = r.fst ∧ (p.snd + q.snd) / 2 = r.snd} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetrical_circle_equation_l994_99490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_l994_99475

noncomputable def f (x : ℝ) : ℤ := 
  ⌊x⌋ + ⌊2*x⌋ + ⌊4*x⌋ + ⌊8*x⌋ + ⌊16*x⌋ + ⌊32*x⌋

theorem no_solution : ∀ x : ℝ, f x ≠ 12345 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_l994_99475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_moment_of_inertia_l994_99431

/-- The moment of inertia of a right circular cylinder with respect to the diameter of its midsection -/
noncomputable def momentOfInertia (R H k : ℝ) : ℝ := k * Real.pi * H * R^2 * (H^2 / 3 + R^2 / 4)

/-- Theorem stating the moment of inertia of a right circular cylinder -/
theorem cylinder_moment_of_inertia (R H k : ℝ) (hR : R > 0) (hH : H > 0) (hk : k > 0) :
  momentOfInertia R H k = k * Real.pi * H * R^2 * (H^2 / 3 + R^2 / 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_moment_of_inertia_l994_99431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_g_property_l994_99438

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- Definition of function g in terms of f -/
noncomputable def g (f : ℝ → ℝ) (x : ℝ) : ℝ := (2 + f x) / f x

theorem odd_function_g_property (f : ℝ → ℝ) (hf : IsOdd f) :
  g f 2 = 3 → g f (-2) = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_g_property_l994_99438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_radius_regular_tetrahedron_l994_99432

/-- The radius of a sphere circumscribed around a regular tetrahedron -/
noncomputable def circumscribedSphereRadius (a : ℝ) : ℝ := (a * Real.sqrt 6) / 4

/-- This function represents the actual radius of the circumscribed sphere
    for a regular tetrahedron with edge length a. -/
noncomputable def radius_of_circumscribed_sphere_regular_tetrahedron (a : ℝ) : ℝ :=
  -- Its implementation is not provided as it's part of the problem to prove.
  sorry

/-- Theorem: The radius of a sphere circumscribed around a regular tetrahedron
    with edge length a is equal to (a * √6) / 4 -/
theorem circumscribed_sphere_radius_regular_tetrahedron (a : ℝ) (h : a > 0) :
  ∃ R : ℝ, R = circumscribedSphereRadius a ∧ 
  R = radius_of_circumscribed_sphere_regular_tetrahedron a :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_radius_regular_tetrahedron_l994_99432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_sum_theorem_l994_99430

theorem root_sum_theorem (p q : ℝ) : 
  (Complex.I : ℂ) ^ 2 = -1 →
  (2 : ℂ) + Complex.I = Complex.ofReal 2 + Complex.I →
  (Complex.ofReal 2 + Complex.I) ^ 3 + p * (Complex.ofReal 2 + Complex.I) + q = 0 →
  p + q = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_sum_theorem_l994_99430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_polygon_area_l994_99486

/-- Represents a square sheet of paper -/
structure Square where
  side : ℝ

/-- Represents a right triangle -/
structure RightTriangle where
  leg1 : ℝ
  leg2 : ℝ

/-- Represents the complex polygon formed by overlapping squares and a triangle -/
structure ComplexPolygon where
  squares : List Square
  triangle : RightTriangle

/-- Calculates the area of a square -/
noncomputable def squareArea (s : Square) : ℝ :=
  s.side * s.side

/-- Calculates the area of a right triangle -/
noncomputable def triangleArea (t : RightTriangle) : ℝ :=
  t.leg1 * t.leg2 / 2

/-- Theorem stating the area of the complex polygon -/
theorem complex_polygon_area 
  (cp : ComplexPolygon) 
  (h1 : cp.squares.length = 4)
  (h2 : ∀ s ∈ cp.squares, s.side = 8)
  (h3 : cp.triangle.leg1 = 8 ∧ cp.triangle.leg2 = 8) :
  (cp.squares.map squareArea).sum + triangleArea cp.triangle = 288 := by
  sorry

#check complex_polygon_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_polygon_area_l994_99486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fuel_vehicle_price_reduction_l994_99463

/-- Represents the price reduction scenario of a fuel vehicle over two months -/
theorem fuel_vehicle_price_reduction 
  (initial_price : ℝ) 
  (final_price : ℝ) 
  (reduction_rate : ℝ) 
  (h1 : initial_price = 23) 
  (h2 : final_price = 16) :
  final_price = initial_price * (1 - reduction_rate)^2 ↔ 
  23 * (1 - reduction_rate)^2 = 16 := by
  sorry

#check fuel_vehicle_price_reduction

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fuel_vehicle_price_reduction_l994_99463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_through_origin_l994_99402

noncomputable def point := ℝ × ℝ

noncomputable def distance (p1 p2 : point) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem total_distance_through_origin :
  let A : point := (-3, 5)
  let B : point := (5, -3)
  let C : point := (0, 0)
  distance A C + distance C B = 2 * Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_through_origin_l994_99402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eventual_five_l994_99436

/-- Definition of a slip operation -/
def slip (n : ℕ) (p : ℕ) : ℕ := (n + p^2) / p

/-- Predicate to check if a number is prime -/
def isPrime (p : ℕ) : Prop := Nat.Prime p

/-- Theorem: For any integer n ≥ 5, there exists a finite sequence of slip operations that transforms n into 5 -/
theorem eventual_five (n : ℕ) (h : n ≥ 5) : 
  ∃ (k : ℕ) (seq : ℕ → ℕ) (primes : ℕ → ℕ), 
    (∀ i, i < k → isPrime (primes i)) ∧
    (∀ i, i < k → primes i ∣ seq i) ∧
    (seq 0 = n) ∧
    (∀ i, i < k - 1 → seq (i + 1) = slip (seq i) (primes i)) ∧
    (seq (k - 1) = 5) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eventual_five_l994_99436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_exp_graph_l994_99489

-- Define the rotation function
def rotate90Clockwise (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, -p.1)

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := Real.exp x

-- Theorem statement
theorem rotated_exp_graph (x y : ℝ) :
  rotate90Clockwise (x, f x) = (y, -x) →
  y = Real.exp (-x) := by
  sorry

#check rotated_exp_graph

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_exp_graph_l994_99489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_leaf_area_l994_99447

/-- The area of a four-leaf shaped figure formed by semicircles on a square's sides -/
theorem four_leaf_area (a : ℝ) (h : a > 0) : 
  (4 * (π * (a / 2)^2 / 2) - a^2) = a^2 * (π / 2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_leaf_area_l994_99447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_range_l994_99454

/-- A line passing through (0, b) intersects the parabola y = x^2 at points A and B. 
    There exists another point C on the parabola, different from A and B, such that CA ⊥ CB. -/
theorem parabola_intersection_range (b : ℝ) : 
  (∃ (k : ℝ) (A B C : ℝ × ℝ),
    A ≠ B ∧ A ≠ C ∧ B ≠ C ∧
    (∀ x : ℝ, (k * x + b = x^2) ↔ (x, x^2) = A ∨ (x, x^2) = B) ∧
    (C.1, C.1^2) = C ∧
    ((A.2 - C.2) * (B.1 - C.1) = (B.2 - C.2) * (A.1 - C.1))) ↔ 
  b ≥ 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_range_l994_99454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_theorem_l994_99435

/-- Given a triangle with area A, side length a, and median length m to that side,
    prove that the cosine of the acute angle θ between the side and the median
    is equal to √13/7 when A = 24, a = 8, and m = 7. -/
theorem triangle_cosine_theorem (A a m : ℝ) (h_area : A = 24) (h_side : a = 8) (h_median : m = 7) :
  let θ := Real.arccos ((2 * A) / (a * m))
  Real.cos θ = Real.sqrt 13 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_theorem_l994_99435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_principals_in_period_l994_99473

/-- Represents a principal's term -/
structure Term where
  start : ℕ
  duration : ℕ
  end_year : ℕ := start + duration - 1

/-- Checks if two terms overlap -/
def overlap (t1 t2 : Term) : Prop :=
  (t1.start ≤ t2.start ∧ t2.start ≤ t1.end_year) ∨
  (t2.start ≤ t1.start ∧ t1.start ≤ t2.end_year)

/-- The problem statement -/
theorem max_principals_in_period :
  ∀ (terms : List Term),
    (∀ t ∈ terms, t.duration = 4) →
    (∀ t1 t2, t1 ∈ terms → t2 ∈ terms → t1 ≠ t2 → ¬(overlap t1 t2)) →
    (∀ t ∈ terms, t.start ≥ 1 ∧ t.end_year ≤ 10) →
    terms.length ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_principals_in_period_l994_99473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_RYZ_l994_99434

-- Define the square WXYZ
def square_WXYZ : Set (ℝ × ℝ) := sorry

-- Define the area of the square
def square_area : ℝ := 144

-- Define point P on side WY
def point_P : ℝ × ℝ := sorry

-- Define the ratio of WP to PY
def wp_py_ratio : ℚ := 2 / 1

-- Define point Q as midpoint of WP
def point_Q : ℝ × ℝ := sorry

-- Define point R as midpoint of ZP
def point_R : ℝ × ℝ := sorry

-- Define the area of quadrilateral WQRP
def area_WQRP : ℝ := 25

-- Define a function to calculate the area of a triangle
def area_of_triangle (A B C : ℝ × ℝ) : ℝ := sorry

-- Theorem to prove
theorem area_triangle_RYZ : 
  ∀ (W X Y Z P Q R : ℝ × ℝ),
  W ∈ square_WXYZ → X ∈ square_WXYZ → Y ∈ square_WXYZ → Z ∈ square_WXYZ →
  P = point_P → Q = point_Q → R = point_R →
  area_of_triangle R Y Z = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_RYZ_l994_99434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l994_99472

/-- The inclination angle of a line with equation x - y - 1 = 0 is 45° -/
theorem line_inclination_angle (l : Set (ℝ × ℝ)) : 
  (∀ p : ℝ × ℝ, p ∈ l ↔ p.1 - p.2 - 1 = 0) → 
  ∃ α : ℝ, α = 45 * (π / 180) ∧ (∀ p q : ℝ × ℝ, p ∈ l → q ∈ l → p ≠ q → 
    Real.arctan ((q.2 - p.2) / (q.1 - p.1)) = α) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l994_99472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_age_weight_not_directly_proportional_l994_99414

/-- Age of a human in years -/
def age : ℝ → ℝ := sorry

/-- Weight of a human in kilograms -/
def weight : ℝ → ℝ := sorry

/-- Proposition stating that age and weight are not necessarily directly proportional -/
theorem age_weight_not_directly_proportional :
  ¬ ∀ (k : ℝ), ∀ (t : ℝ), age t = k * weight t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_age_weight_not_directly_proportional_l994_99414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_transformation_l994_99418

theorem sin_transformation (x : ℝ) : 
  Real.sin (2 * (x - π / 3) - π / 3) = -Real.sin (2 * x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_transformation_l994_99418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limiting_ratio_sectors_to_circle_l994_99485

/-- The limiting ratio of the area of sectors outside a fixed circle but inside an inscribed
    equilateral triangle to the area of the circle, as the triangle's side length approaches infinity -/
theorem limiting_ratio_sectors_to_circle (r : ℝ) (hr : r > 0) :
  ∀ ε > 0, ∃ S : ℝ, ∀ s ≥ S,
    (Real.sqrt 3 / 4 * s^2 - Real.pi * r^2) / (Real.pi * r^2) > ε := by
  sorry

#check limiting_ratio_sectors_to_circle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limiting_ratio_sectors_to_circle_l994_99485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_projection_area_theorem_l994_99444

/-- A tetrahedron with two adjacent equilateral triangle faces -/
structure Tetrahedron where
  side_length : ℝ
  dihedral_angle : ℝ

/-- The maximum projection area of a rotating tetrahedron -/
noncomputable def max_projection_area (t : Tetrahedron) : ℝ := 
  (9 * Real.sqrt 3) / 4

/-- Theorem stating the maximum projection area of a specific tetrahedron -/
theorem max_projection_area_theorem (t : Tetrahedron) 
  (h1 : t.side_length = 3)
  (h2 : t.dihedral_angle = π / 6) : 
  max_projection_area t = (9 * Real.sqrt 3) / 4 := by
  sorry

#check max_projection_area_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_projection_area_theorem_l994_99444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_value_l994_99457

-- Define the power function
noncomputable def powerFunction (a : ℝ) (x : ℝ) : ℝ := x ^ a

-- State the theorem
theorem power_function_value (f : ℝ → ℝ) (a : ℝ) :
  (∀ x > 0, f x = powerFunction a x) →
  f 4 = 2 →
  f 16 = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_value_l994_99457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_to_line_l994_99474

-- Define the curve and line
noncomputable def curve (x : ℝ) : ℝ := x * Real.exp (-x)
def line (x : ℝ) : ℝ := x + 1

-- Define the derivative of the curve
noncomputable def curve_derivative (x : ℝ) : ℝ := (1 - x) * Real.exp (-x)

-- Theorem statement
theorem min_distance_curve_to_line :
  ∃ (p : ℝ × ℝ) (q : ℝ × ℝ),
    p.2 = curve p.1 ∧
    q.2 = line q.1 ∧
    (∀ (p' : ℝ × ℝ) (q' : ℝ × ℝ),
      p'.2 = curve p'.1 → q'.2 = line q'.1 →
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≤ Real.sqrt ((p'.1 - q'.1)^2 + (p'.2 - q'.2)^2)) ∧
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_to_line_l994_99474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flu_virus_diameter_scientific_notation_l994_99493

/-- Given the diameter of a new type of flu virus in meters, prove that it is equal to its scientific notation representation. -/
theorem flu_virus_diameter_scientific_notation :
  (0.000000815 : ℝ) = 8.15 * 10^(-7 : ℤ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flu_virus_diameter_scientific_notation_l994_99493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unreachable_set_properties_l994_99442

-- Define the set of points from which the squirrel cannot reach a point more than 1000 units away
def unreachable_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 < 5 ∧ 0 ≤ p.2 ∧ p.2 < 5}

-- Define the jumping rules
def can_jump (p q : ℝ × ℝ) : Prop :=
  (q.1 = p.1 - 5 ∧ q.2 = p.2 + 7) ∨ (q.1 = p.1 + 1 ∧ q.2 = p.2 - 1)

-- Define the first quadrant
def first_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 ≥ 0 ∧ p.2 ≥ 0

-- Define the distance from origin
noncomputable def distance_from_origin (p : ℝ × ℝ) : ℝ :=
  Real.sqrt (p.1^2 + p.2^2)

-- Theorem statement
theorem unreachable_set_properties :
  ∀ p ∈ unreachable_set,
    (∀ q, first_quadrant q → can_jump p q → distance_from_origin q ≤ 1000) ∧
    (MeasureTheory.volume unreachable_set = 25) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unreachable_set_properties_l994_99442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_radical_sum_l994_99410

theorem simplify_radical_sum : Real.sqrt (12 + 8 * Real.sqrt 3) + Real.sqrt (12 - 8 * Real.sqrt 3) = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_radical_sum_l994_99410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_line_proof_l994_99480

-- Define the point M
def M : ℝ × ℝ × ℝ := (-2, 0, 3)

-- Define the first line
def line1 (t : ℝ) : ℝ × ℝ × ℝ := (2 - t, 3, -2 + t)

-- Define the second line as the intersection of two planes
def line2 : Set (ℝ × ℝ × ℝ) :=
  {p | 2 * p.1 - 2 * p.2.1 - p.2.2 - 4 = 0 ∧ p.1 + 3 * p.2.1 + 2 * p.2.2 + 1 = 0}

-- Define the intersecting line
def intersecting_line (t : ℝ) : ℝ × ℝ × ℝ := (-2 + 13*t, -3*t, 3 - 12*t)

-- Theorem statement
theorem intersecting_line_proof :
  (∃ t : ℝ, intersecting_line t = M) ∧
  (∃ t₁ t₂ : ℝ, line1 t₁ ∈ Set.range intersecting_line) ∧
  (∃ p : ℝ × ℝ × ℝ, p ∈ line2 ∧ p ∈ Set.range intersecting_line) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_line_proof_l994_99480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_is_perfect_square_l994_99494

def A (k : ℕ) : ℕ := 10^(2*k+2) + 7 * ((10^(2*k+1) - 1) / 9) + 6

def B (k : ℕ) : ℕ := 3 * 10^(k+1) + 5 * ((10^k - 1) / 9) + 2

theorem difference_is_perfect_square (k : ℕ) :
  ∃ (n : ℕ), A k - B k = n^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_is_perfect_square_l994_99494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_expression_f_monotone_increasing_a_value_l994_99405

noncomputable def OA (a x : ℝ) : ℝ × ℝ := (a * Real.cos x ^ 2, 1)

noncomputable def OB (a x : ℝ) : ℝ × ℝ := (2, Real.sqrt 3 * a * Real.sin (2 * x - a))

noncomputable def f (a x : ℝ) : ℝ := (OA a x).1 * (OB a x).1 + (OA a x).2 * (OB a x).2

theorem f_expression (a x : ℝ) (h : a ≠ 0) :
  f a x = 2 * a * Real.sin (2 * x + π / 6) := by
  sorry

theorem f_monotone_increasing (a x : ℝ) (ha : a > 0) (k : ℤ) :
  MonotoneOn (f a) (Set.Icc (k * π - π / 3) (k * π + π / 6)) := by
  sorry

theorem a_value (a : ℝ) (h : a ≠ 0) :
  (∀ x ∈ Set.Icc 0 (π / 2), f a x ≤ 5) ∧ (∃ x ∈ Set.Icc 0 (π / 2), f a x = 5) →
  a = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_expression_f_monotone_increasing_a_value_l994_99405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_boggies_count_l994_99407

/-- Represents the number of boggies initially in the train -/
def n : ℕ := 12

/-- Length of each boggy in meters -/
def boggy_length : ℝ := 15

/-- Time taken to cross a telegraph post initially in seconds -/
def initial_time : ℝ := 9

/-- Time taken to cross a telegraph post after detaching one boggy in seconds -/
def final_time : ℝ := 8.25

/-- The speed of the train remains constant -/
axiom constant_speed : ℝ

theorem initial_boggies_count : n = 12 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_boggies_count_l994_99407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_bought_five_l994_99411

/-- The number of rabbits Anton sells each day -/
def total_rabbits : ℚ := 20

/-- The number of rabbits X buys on the first day -/
def x : ℚ := 5

/-- Condition: After Grandma #1 and Man #1 buy on day 1, and X buys, 7 rabbits are left -/
axiom day1_condition : 0.6 * total_rabbits - x = 7

/-- Condition: After Grandma #2 and Man #2 buy on day 2, and X buys twice as much, 0 rabbits are left -/
axiom day2_condition : 0.5 * total_rabbits - 2 * x = 0

/-- The theorem to be proved -/
theorem x_bought_five : x = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_bought_five_l994_99411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_five_consecutive_integers_not_square_l994_99468

theorem product_of_five_consecutive_integers_not_square (n : ℕ) :
  ∃ k : ℕ, n * (n + 1) * (n + 2) * (n + 3) * (n + 4) ≠ k^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_five_consecutive_integers_not_square_l994_99468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_of_sequence_l994_99470

/-- An arithmetic sequence with the given first four terms -/
structure ArithmeticSequence (x y : ℚ) where
  a1 : ℚ := x + 2 * y^2
  a2 : ℚ := x - 2 * y^2
  a3 : ℚ := x + 3 * y
  a4 : ℚ := x - 4 * y

/-- The fifth term of the arithmetic sequence -/
def fifthTerm (x y : ℚ) : ℚ := x - 7/2

/-- Theorem: The fifth term of the arithmetic sequence is x - 7/2 -/
theorem fifth_term_of_sequence (x y : ℚ) :
  ∃ (seq : ArithmeticSequence x y), fifthTerm x y = x - 7/2 := by
  sorry

#eval fifthTerm 5 (-1/2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_of_sequence_l994_99470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_b_c_relationship_l994_99462

-- Define the variables as noncomputable
noncomputable def a : ℝ := (4 : ℝ) ^ (1/2 : ℝ)
noncomputable def b : ℝ := (2 : ℝ) ^ (1/2 : ℝ)
noncomputable def c : ℝ := (1/2 : ℝ) ^ 4

-- State the theorem
theorem a_b_c_relationship : c < b ∧ b < a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_b_c_relationship_l994_99462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harry_potter_angular_momentum_l994_99477

/-- Calculates the angular momentum of a rotating person -/
noncomputable def angular_momentum (m : ℝ) (r : ℝ) (a : ℝ) : ℝ :=
  let I := m * r^2
  let ω := Real.sqrt (a / r)
  I * ω

theorem harry_potter_angular_momentum :
  let m : ℝ := 50.0  -- mass in kg
  let r : ℝ := 2.0   -- radius in m
  let g : ℝ := 9.8   -- gravitational acceleration in m/s²
  let a : ℝ := 5.0 * g  -- maximum acceleration in m/s²
  |angular_momentum m r a - 1000| < 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_harry_potter_angular_momentum_l994_99477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_4_value_l994_99416

def sequence_a : ℕ → ℚ
  | 0 => 1  -- Add this case for 0
  | 1 => 1
  | (n + 2) => 2 * sequence_a (n + 1) / (sequence_a (n + 1) + 2)

theorem a_4_value : sequence_a 4 = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_4_value_l994_99416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_q_trajectory_l994_99445

-- Define the semicircle
structure Semicircle where
  center : ℝ × ℝ
  radius : ℝ
  ab : ℝ × ℝ × ℝ × ℝ  -- coordinates of A and B

-- Define a point on the semicircle
def PointOnSemicircle (s : Semicircle) := { p : ℝ × ℝ // ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ Real.pi ∧ p = (s.center.1 + s.radius * Real.cos θ, s.center.2 + s.radius * Real.sin θ) }

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define point Q
noncomputable def Q (s : Semicircle) (p : PointOnSemicircle s) : ℝ × ℝ :=
  let c := s.center
  let (a, _, b, _) := s.ab
  let pa := distance p.val (a, s.center.2 - s.radius)
  let pb := distance p.val (b, s.center.2 - s.radius)
  let t := (Real.sqrt 2 - 1) / Real.sqrt 2
  (c.1 + t * (p.val.1 - c.1), c.2 + t * (p.val.2 - c.2))

-- Theorem statement
theorem q_trajectory (s : Semicircle) :
  ∀ p : PointOnSemicircle s,
    ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ Real.pi ∧
      Q s p = (s.center.1 + s.radius * (Real.sqrt 2 - 1) / Real.sqrt 2 * Real.cos θ,
               s.center.2 + s.radius * (Real.sqrt 2 - 1) / Real.sqrt 2 * Real.sin θ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_q_trajectory_l994_99445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_line_with_ratio_2_to_1_l994_99495

/-- Represents a conic section (ellipse or hyperbola) --/
structure ConicSection where
  a : ℝ
  b : ℝ
  isEllipse : Bool

/-- Defines the ellipse Γ₁ --/
noncomputable def Γ₁ : ConicSection := { a := 2, b := Real.sqrt 3, isEllipse := true }

/-- Defines the hyperbola Γ₂ --/
noncomputable def Γ₂ : ConicSection := { a := 1/2, b := Real.sqrt 3/2, isEllipse := false }

/-- Checks if a point (x, y) is on a conic section --/
def isOnConic (c : ConicSection) (x y : ℝ) : Prop :=
  if c.isEllipse then
    x^2 / c.a^2 + y^2 / c.b^2 = 1
  else
    x^2 / c.a^2 - y^2 / c.b^2 = 1

/-- Theorem: There exists a line through the focus (1,0) intersecting Γ₁ and Γ₂ such that |AB| = 2|CD| --/
theorem exists_line_with_ratio_2_to_1 :
  ∃ (m : ℝ),
    (isOnConic Γ₁ 1 (3/2)) ∧
    (isOnConic Γ₂ 1 (3/2)) ∧
    (∃ (y₁ y₂ y₃ y₄ : ℝ),
      isOnConic Γ₁ (m * y₁ + 1) y₁ ∧
      isOnConic Γ₁ (m * y₂ + 1) y₂ ∧
      isOnConic Γ₂ (m * y₃ + 1) y₃ ∧
      isOnConic Γ₂ (m * y₄ + 1) y₄ ∧
      |y₁ - y₂| = 2 * |y₃ - y₄|) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_line_with_ratio_2_to_1_l994_99495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_score_problem_l994_99461

def first_four_games : List ℕ := [17, 22, 19, 16]

theorem basketball_score_problem 
  (total_games : ℕ) 
  (first_four_avg : ℚ) 
  (final_avg : ℚ) :
  total_games = 8 →
  first_four_avg < 20 →
  final_avg > 21 →
  (List.sum first_four_games : ℚ) / 4 ≤ first_four_avg →
  ∃ (last_four : List ℕ),
    last_four.length = 4 ∧
    (List.sum (first_four_games ++ last_four) : ℚ) / total_games > final_avg ∧
    last_four.getLast? = some 29 ∧
    ∀ (x : ℕ), x < 29 → 
      (List.sum (first_four_games ++ (List.take 3 last_four ++ [x])) : ℚ) / total_games ≤ final_avg :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_score_problem_l994_99461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l994_99441

noncomputable def f (x : ℝ) := Real.cos (2 * x - Real.pi / 6) * Real.sin (2 * x) - 1 / 4

theorem f_properties :
  let period := Real.pi / 2
  ∀ k : ℤ,
    (∀ x : ℝ, f (x + period) = f x) ∧
    (∀ x y : ℝ, Real.pi / 6 + k * Real.pi / 2 ≤ x ∧ x < y ∧ y ≤ 5 * Real.pi / 12 + k * Real.pi / 2 → f y < f x) ∧
    (∀ x : ℝ, -Real.pi / 4 ≤ x ∧ x ≤ 0 → f x ≤ 1 / 4) ∧
    (∀ x : ℝ, -Real.pi / 4 ≤ x ∧ x ≤ 0 → f x ≥ -1 / 2) ∧
    (∃ x : ℝ, -Real.pi / 4 ≤ x ∧ x ≤ 0 ∧ f x = 1 / 4) ∧
    (∃ x : ℝ, -Real.pi / 4 ≤ x ∧ x ≤ 0 ∧ f x = -1 / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l994_99441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_books_remaining_correct_l994_99453

/-- Calculates the number of books remaining on a library shelf after two days of borrowing and returning. -/
noncomputable def booksRemaining (x a b c d : ℝ) : ℝ :=
  x - (a * b) - c + (d / 100) * (a * b)

/-- Theorem stating that the number of books remaining on the shelf after two days
    is equal to the initial number minus borrowed books plus returned books. -/
theorem books_remaining_correct (x a b c d : ℝ) :
  booksRemaining x a b c d = x - (a * b) - c + (d / 100) * (a * b) := by
  sorry

#check books_remaining_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_books_remaining_correct_l994_99453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_a_are_integers_l994_99404

def a : ℕ → ℚ
  | 0 => 2  -- Adding the case for 0 to address the missing case error
  | 1 => 1
  | 2 => 1
  | 3 => 997
  | (n + 4) => (1993 + a (n + 3) * a (n + 2)) / a (n + 1)

theorem all_a_are_integers : ∀ n : ℕ, ∃ k : ℤ, a n = k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_a_are_integers_l994_99404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_differential_f1_f2_l994_99409

noncomputable section

-- Function 1: z = 5x^2y^3
def f1 (x y : ℝ) : ℝ := 5 * x^2 * y^3

-- Function 2: z = arctg(x^2 + 3y)
noncomputable def f2 (x y : ℝ) : ℝ := Real.arctan (x^2 + 3*y)

-- Total differential of f1
def df1 (x y dx dy : ℝ) : ℝ := 5 * x * y^2 * (2 * y * dx + 3 * x * dy)

-- Total differential of f2
noncomputable def df2 (x y dx dy : ℝ) : ℝ := (2 * x * dx + 3 * dy) / (1 + (x^2 + 3*y)^2)

theorem total_differential_f1_f2 (x y dx dy : ℝ) :
  (deriv (fun x => f1 x y) x) * dx + (deriv (fun y => f1 x y) y) * dy = df1 x y dx dy ∧
  (deriv (fun x => f2 x y) x) * dx + (deriv (fun y => f2 x y) y) * dy = df2 x y dx dy := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_differential_f1_f2_l994_99409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_integer_ratios_l994_99469

/-- Given a triangle with integer side lengths, its semiperimeter is half the sum of its sides -/
def semiperimeter (a b c : ℕ) : ℚ :=
  (a + b + c : ℚ) / 2

/-- The area of a triangle can be calculated using Heron's formula -/
noncomputable def triangleArea (a b c : ℕ) : ℚ :=
  let s := semiperimeter a b c
  (s * (s - a) * (s - b) * (s - c)).sqrt

/-- The inradius of a triangle is the ratio of its area to its semiperimeter -/
noncomputable def inradius (a b c : ℕ) : ℚ :=
  triangleArea a b c / semiperimeter a b c

/-- The theorem states that there are infinitely many positive integers n
    such that the ratio of semiperimeter to inradius is an integer -/
theorem infinitely_many_integer_ratios :
  ∃ (f : ℕ → ℕ × ℕ × ℕ), Function.Injective f ∧
  ∀ n : ℕ, let (a, b, c) := f n; 
    (a > 0 ∧ b > 0 ∧ c > 0) ∧
    (a + b > c ∧ a + c > b ∧ b + c > a) ∧
    ∃ (m : ℕ), m > 0 ∧ semiperimeter a b c / inradius a b c = m := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_integer_ratios_l994_99469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_length_of_circles_l994_99458

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 10*x - 10*y = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 6*x - 2*y - 40 = 0

-- Define the common chord length
noncomputable def common_chord_length : ℝ := 2 * Real.sqrt 30

-- Theorem statement
theorem common_chord_length_of_circles :
  ∀ (x y : ℝ), circle1 x y ∧ circle2 x y →
  ∃ (a b : ℝ), (x - a)^2 + (y - b)^2 = common_chord_length^2 / 4 :=
by sorry

#check common_chord_length_of_circles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_length_of_circles_l994_99458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_computer_table_cost_price_l994_99455

/-- Given a furniture shop with the following pricing structure:
  * 45% markup on cost price
  * Rs. 200 flat assembly fee for computer tables
  * 8% sales tax
  * 15% loyalty discount
  
  And given that:
  * A customer bought a computer table and a chair for Rs. 3800 (including all fees and discounts)
  * The chair costs Rs. 1000 without any additional fees or discounts

  This theorem proves that the cost price of the computer table before the assembly fee 
  is approximately Rs. 1905.49 -/
theorem computer_table_cost_price 
  (markup : ℝ) 
  (assembly_fee : ℝ) 
  (sales_tax : ℝ) 
  (loyalty_discount : ℝ) 
  (total_paid : ℝ) 
  (chair_cost : ℝ) :
  markup = 0.45 →
  assembly_fee = 200 →
  sales_tax = 0.08 →
  loyalty_discount = 0.15 →
  total_paid = 3800 →
  chair_cost = 1000 →
  ∃ (cost_price : ℝ), 
    ((cost_price * (1 + markup) + assembly_fee) * (1 + sales_tax) * (1 - loyalty_discount) + 
     chair_cost * (1 + sales_tax)) = total_paid ∧ 
    (abs (cost_price - 1905.49) < 0.01) := by
  sorry

#check computer_table_cost_price

end NUMINAMATH_CALUDE_ERRORFEEDBACK_computer_table_cost_price_l994_99455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loss_percent_example_l994_99413

/-- Calculate the loss percent given the cost price and selling price -/
noncomputable def loss_percent (cost_price selling_price : ℝ) : ℝ :=
  ((cost_price - selling_price) / cost_price) * 100

/-- Proof that the loss percent is approximately 39.29% -/
theorem loss_percent_example : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |loss_percent 560 340 - 39.29| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_loss_percent_example_l994_99413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_value_l994_99488

def my_sequence (n : ℕ) : ℤ :=
  match n with
  | 0 => -1
  | n + 1 => 2 * my_sequence n - 3

theorem fifth_term_value : my_sequence 4 = -61 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_value_l994_99488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_four_ones_approx_l994_99482

/-- The probability of rolling exactly four 1's when rolling twelve 6-sided dice -/
def probability_four_ones : ℚ :=
  495 * (5^8 : ℚ) / (6^12 : ℚ)

/-- Theorem stating that the probability of rolling exactly four 1's
    when rolling twelve 6-sided dice is approximately 0.114 -/
theorem probability_four_ones_approx :
  abs ((probability_four_ones : ℝ) - 0.114) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_four_ones_approx_l994_99482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_temp_increase_per_century_l994_99465

/-- Given a total temperature change over a period of time, 
    calculate the temperature change per century. -/
noncomputable def temp_change_per_century (total_change : ℝ) (years : ℕ) : ℝ :=
  total_change / (years / 100 : ℝ)

/-- Theorem: The temperature increase per century is 3 units, 
    given that the total temperature change over 700 years is 21 units. -/
theorem temp_increase_per_century : 
  temp_change_per_century 21 700 = 3 := by
  -- Unfold the definition of temp_change_per_century
  unfold temp_change_per_century
  -- Simplify the arithmetic
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_temp_increase_per_century_l994_99465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_range_l994_99476

theorem sequence_inequality_range :
  (∀ n : ℕ+, ((-1 : ℝ) ^ (n.val + 2018 : ℕ)) < 2 + ((-1 : ℝ) ^ (n.val + 2017 : ℕ)) / n.val) →
  (∀ a : ℝ, (∀ n : ℕ+, a < 2 + ((-1 : ℝ) ^ (n.val + 2017 : ℕ)) / n.val) ↔ a ∈ Set.Icc (-2 : ℝ) (3/2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_range_l994_99476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ab_value_l994_99448

-- Define the line equation
def line_eq (k : ℝ) (x y : ℝ) : Prop := x + y = 2 * k

-- Define the circle equation
def circle_eq (k : ℝ) (x y : ℝ) : Prop := x^2 + y^2 = k^2 - 2*k + 3

-- Define the intersection point
def intersection_point (k : ℝ) (a b : ℝ) : Prop :=
  line_eq k a b ∧ circle_eq k a b

-- State the theorem
theorem max_ab_value :
  ∃ (max : ℝ), (∀ (k a b : ℝ), intersection_point k a b → a * b ≤ max) ∧ max = 9 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ab_value_l994_99448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_distance_inequality_triangle_distance_equality_l994_99408

-- Define a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define distances from a point to the sides of a triangle
noncomputable def distances (t : Triangle) (M : ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

-- Define inradius and circumradius of a triangle
noncomputable def inradius (t : Triangle) : ℝ := sorry
noncomputable def circumradius (t : Triangle) : ℝ := sorry

-- Define the theorem
theorem triangle_distance_inequality (t : Triangle) (M : ℝ × ℝ) :
  let (d_a, d_b, d_c) := distances t M
  let r := inradius t
  let R := circumradius t
  d_a^2 + d_b^2 + d_c^2 ≥ 12 * (r^4 / R^2) := by
  sorry

-- Define the equality case
theorem triangle_distance_equality (t : Triangle) :
  ∃ M : ℝ × ℝ, let (d_a, d_b, d_c) := distances t M
                let r := inradius t
                let R := circumradius t
                d_a^2 + d_b^2 + d_c^2 = 12 * (r^4 / R^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_distance_inequality_triangle_distance_equality_l994_99408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_painted_squares_l994_99464

/-- Represents a 4x4 board where each cell can be either painted or unpainted -/
def Board : Type := Fin 4 → Fin 4 → Bool

/-- Checks if three cells form an L-shape -/
def isLShape (a b c : Fin 4 × Fin 4) : Prop :=
  ((a.1 = b.1 ∧ a.2 = b.2 + 1 ∧ c.1 = b.1 + 1 ∧ c.2 = b.2) ∨
   (a.1 = b.1 ∧ a.2 = b.2 - 1 ∧ c.1 = b.1 + 1 ∧ c.2 = b.2) ∨
   (a.1 = b.1 ∧ a.2 = b.2 + 1 ∧ c.1 = b.1 - 1 ∧ c.2 = b.2) ∨
   (a.1 = b.1 ∧ a.2 = b.2 - 1 ∧ c.1 = b.1 - 1 ∧ c.2 = b.2))

/-- Checks if a board satisfies the L-shape condition -/
def satisfiesLShapeCondition (board : Board) : Prop :=
  ∀ a b c : Fin 4 × Fin 4, isLShape a b c →
    board a.1 a.2 ∨ board b.1 b.2 ∨ board c.1 c.2

/-- Counts the number of painted squares on a board -/
def countPaintedSquares (board : Board) : Nat :=
  (Finset.sum (Finset.univ : Finset (Fin 4)) fun i =>
    Finset.sum (Finset.univ : Finset (Fin 4)) fun j =>
      if board i j then 1 else 0)

/-- The main theorem stating that the minimum number of painted squares is 8 -/
theorem min_painted_squares :
  (∃ (board : Board), satisfiesLShapeCondition board ∧ countPaintedSquares board = 8) ∧
  (∀ (board : Board), satisfiesLShapeCondition board → countPaintedSquares board ≥ 8) := by
  sorry

#check min_painted_squares

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_painted_squares_l994_99464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_travel_distance_l994_99450

-- Define the rates and times
noncomputable def bike_rate : ℝ := 8
noncomputable def bike_time : ℝ := 1
noncomputable def jog_rate : ℝ := 6
noncomputable def jog_time : ℝ := 1/3

-- Define the total distance function
noncomputable def total_distance (bike_r bike_t jog_r jog_t : ℝ) : ℝ :=
  bike_r * bike_t + jog_r * jog_t

-- Theorem statement
theorem total_travel_distance :
  total_distance bike_rate bike_time jog_rate jog_time = 10 := by
  -- Unfold the definition of total_distance
  unfold total_distance
  -- Simplify the expression
  simp [bike_rate, bike_time, jog_rate, jog_time]
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_travel_distance_l994_99450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_boundary_distance_probability_l994_99420

/-- A rectangle with length 2 and width 1 -/
structure Rectangle :=
  (length : ℝ)
  (width : ℝ)
  (length_eq : length = 2)
  (width_eq : width = 1)

/-- A point on the boundary of the rectangle -/
structure BoundaryPoint (r : Rectangle) :=
  (x : ℝ)
  (y : ℝ)
  (on_boundary : (x = 0 ∨ x = r.length) ∨ (y = 0 ∨ y = r.width))

/-- The straight-line distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- The probability of an event occurring when choosing two random points on the boundary -/
noncomputable def probability (r : Rectangle) (event : BoundaryPoint r → BoundaryPoint r → Prop) : ℝ :=
  sorry

theorem rectangle_boundary_distance_probability (r : Rectangle) :
  probability r (fun p1 p2 => distance (p1.x, p1.y) (p2.x, p2.y) ≥ 1) = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_boundary_distance_probability_l994_99420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_magnitude_sum_collinear_value_l994_99483

-- Define the vectors a and b
noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin x, 3/2)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos x, -1)

-- Define the magnitude of the sum of vectors
noncomputable def mag_sum (x : ℝ) : ℝ := Real.sqrt ((Real.sin x + Real.cos x)^2 + (1/2)^2)

-- Define collinearity condition
def collinear (x : ℝ) : Prop := ∃ k : ℝ, a x = k • (b x)

-- Theorem statements
theorem max_magnitude_sum : 
  ∀ x : ℝ, mag_sum x ≤ 3/2 ∧ ∃ y : ℝ, mag_sum y = 3/2 := by sorry

theorem collinear_value : 
  ∀ x : ℝ, collinear x → 2 * (Real.cos x)^2 - Real.sin (2*x) = 20/13 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_magnitude_sum_collinear_value_l994_99483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_all_ones_quadratic_polynomial_l994_99484

/-- A natural number is composed solely of ones in its decimal representation -/
def is_all_ones (n : ℕ) : Prop := sorry

/-- A natural number is composed solely of ones in its decimal representation -/
def result_all_ones (n : ℕ) : Prop := sorry

/-- Quadratic polynomial with integer coefficients -/
def QuadraticPolynomial : Type := ℤ → ℤ → ℤ → ℕ → ℕ

theorem no_all_ones_quadratic_polynomial :
  ∀ (P : QuadraticPolynomial), ∃ (n : ℕ), is_all_ones n ∧ ¬(result_all_ones (P 0 1 2 n)) :=
by
  sorry

#check no_all_ones_quadratic_polynomial

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_all_ones_quadratic_polynomial_l994_99484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l994_99439

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin x, 1/2)

noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos x + Real.sin x, -1)

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem f_properties :
  (∃ T : ℝ, T > 0 ∧ (∀ x : ℝ, f (x + T) = f x) ∧
    (∀ S : ℝ, S > 0 → (∀ x : ℝ, f (x + S) = f x) → T ≤ S)) ∧
  (∃ x_max : ℝ, x_max ∈ Set.Icc (π/4) (π/2) ∧
    ∀ x ∈ Set.Icc (π/4) (π/2), f x ≤ f x_max) ∧
  (∃ x_min : ℝ, x_min ∈ Set.Icc (π/4) (π/2) ∧
    ∀ x ∈ Set.Icc (π/4) (π/2), f x_min ≤ f x) ∧
  f (π/3) = 1 ∧ f (π/2) = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l994_99439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l994_99467

-- Define the variables
variable (a b c x : ℝ)

-- State the theorem
theorem problem_solution :
  (∃ (s : ℝ), s^2 = a + 1 ∧ (s = 5 ∨ s = -5)) →
  ((-2)^3 = b) →
  (c = ⌊Real.sqrt 12⌋) →
  (x = Real.sqrt 12 - ⌊Real.sqrt 12⌋) →
  (a = 24 ∧ b = -8 ∧ c = 3 ∧ Real.sqrt ((Real.sqrt 12 + 3) - x) = Real.sqrt 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l994_99467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_satisfying_equation_l994_99437

open BigOperators Finset

theorem count_integers_satisfying_equation : 
  ∃ S : Finset ℤ, S.card = 13431 ∧ 
  ∀ n : ℤ, n ∈ S ↔ 1 + Int.floor ((120 : ℚ) * n / 121) = Int.ceil ((110 : ℚ) * n / 111) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_satisfying_equation_l994_99437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_right_angle_pyramid_l994_99451

/-- A pyramid with an equilateral triangle base and two right angles at the apex --/
structure RightAnglePyramid where
  -- The side length of the equilateral triangle base
  baseSideLength : ℝ
  -- The base is an equilateral triangle
  isEquilateralBase : baseSideLength > 0
  -- Two of the three vertex angles at the apex are right angles
  hasTwoRightAngles : True

/-- The volume of the RightAnglePyramid --/
noncomputable def volume (p : RightAnglePyramid) : ℝ :=
  (1 / 3) * ((Real.sqrt 3) / 4) * ((Real.sqrt 3) / 4)

/-- Theorem: The maximum volume of a RightAnglePyramid with base side length 1 is 1/16 --/
theorem max_volume_right_angle_pyramid :
  ∀ (p : RightAnglePyramid), p.baseSideLength = 1 → volume p = 1/16 :=
by
  sorry

#check max_volume_right_angle_pyramid

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_right_angle_pyramid_l994_99451
