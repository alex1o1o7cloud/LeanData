import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_is_72_l844_84450

-- Define the given parameters
def train_length : ℝ := 200
def bridge_length : ℝ := 132
def crossing_time : ℝ := 16.5986721062315

-- Define the function to calculate speed in km/hr
noncomputable def calculate_speed (train_length bridge_length crossing_time : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let speed_ms := total_distance / crossing_time
  speed_ms * 3.6

-- Theorem statement
theorem train_speed_is_72 :
  calculate_speed train_length bridge_length crossing_time = 72 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_is_72_l844_84450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ternary_to_base_nine_equality_l844_84466

def ternary_to_base_nine (ternary_digits : Fin 8 → Fin 3) : Fin 4 → Fin 9 :=
  fun _ => ⟨8, by norm_num⟩

def ternary_value (ternary_digits : Fin 8 → Fin 3) : ℕ :=
  (ternary_digits 0).val +
  (ternary_digits 1).val * 3 +
  (ternary_digits 2).val * 3^2 +
  (ternary_digits 3).val * 3^3 +
  (ternary_digits 4).val * 3^4 +
  (ternary_digits 5).val * 3^5 +
  (ternary_digits 6).val * 3^6 +
  (ternary_digits 7).val * 3^7

def base_nine_value (base_nine_digits : Fin 4 → Fin 9) : ℕ :=
  (base_nine_digits 0).val +
  (base_nine_digits 1).val * 9 +
  (base_nine_digits 2).val * 9^2 +
  (base_nine_digits 3).val * 9^3

axiom ternary_digits_are_two (ternary_digits : Fin 8 → Fin 3) :
  ∀ i : Fin 8, (ternary_digits i).val = 2

theorem ternary_to_base_nine_equality (ternary_digits : Fin 8 → Fin 3) :
  ternary_value ternary_digits = base_nine_value (ternary_to_base_nine ternary_digits) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ternary_to_base_nine_equality_l844_84466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_constant_l844_84402

def sequence_a : ℕ → ℝ := sorry
def sequence_b : ℕ → ℝ := sorry

axiom a_positive : ∀ n, 0 < sequence_a n
axiom b_positive : ∀ n, 0 < sequence_b n

axiom a_initial : sequence_a 0 = 1 ∧ sequence_a 0 ≥ sequence_a 1

axiom a_relation : ∀ n, n ≥ 1 → 
  sequence_a n * (sequence_b (n-1) + sequence_b (n+1)) = 
  sequence_a (n-1) * sequence_b (n-1) + sequence_a (n+1) * sequence_b (n+1)

axiom b_sum_bound : ∀ n, n ≥ 1 → 
  (Finset.range (n+1)).sum (λ i => sequence_b i) ≤ (n : ℝ) ^ (3/2)

theorem a_constant : ∀ n, sequence_a n = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_constant_l844_84402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dodecagon_diagonal_intersection_probability_l844_84449

/-- A regular dodecagon -/
structure RegularDodecagon where
  -- Add any necessary properties here

/-- The set of all diagonals in a regular dodecagon -/
def diagonals (d : RegularDodecagon) : Finset (Fin 12 × Fin 12) :=
  sorry

/-- The set of all pairs of diagonals in a regular dodecagon -/
def diagonal_pairs (d : RegularDodecagon) : Finset (Fin 12 × Fin 12 × Fin 12 × Fin 12) :=
  sorry

/-- Predicate to check if two diagonals intersect inside the dodecagon -/
def intersect_inside (d : RegularDodecagon) (diag1 diag2 : Fin 12 × Fin 12) : Prop :=
  sorry

/-- The set of all intersecting diagonal pairs inside the dodecagon -/
def intersecting_pairs (d : RegularDodecagon) : Finset (Fin 12 × Fin 12 × Fin 12 × Fin 12) :=
  sorry

theorem dodecagon_diagonal_intersection_probability (d : RegularDodecagon) :
  (Finset.card (intersecting_pairs d) : ℚ) /
  (Finset.card (diagonal_pairs d) : ℚ) = 495 / 1431 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dodecagon_diagonal_intersection_probability_l844_84449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_iff_a_in_range_l844_84409

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 1 then a^x else (4 - a/2)*x + 2

theorem f_monotone_iff_a_in_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ (4 ≤ a ∧ a < 8) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_iff_a_in_range_l844_84409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_damaged_cartons_per_customer_l844_84473

theorem damaged_cartons_per_customer 
  (total_cartons : ℕ) 
  (num_customers : ℕ) 
  (accepted_cartons : ℕ) 
  (h1 : total_cartons = 400)
  (h2 : num_customers = 4)
  (h3 : accepted_cartons = 160)
  (h4 : total_cartons % num_customers = 0) :
  (total_cartons - accepted_cartons) / num_customers = 60 := by
  sorry

#check damaged_cartons_per_customer

end NUMINAMATH_CALUDE_ERRORFEEDBACK_damaged_cartons_per_customer_l844_84473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_closest_to_18_l844_84479

noncomputable section

/-- The area of the shaded region in a rectangle with a semicircle removed -/
def shaded_area (rectangle_length : ℝ) (rectangle_width : ℝ) (semicircle_diameter : ℝ) : ℝ :=
  rectangle_length * rectangle_width - (Real.pi / 8) * semicircle_diameter ^ 2

/-- The whole number closest to a given real number -/
def closest_whole_number (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

theorem shaded_area_closest_to_18 :
  closest_whole_number (shaded_area 5 4 2) = 18 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_closest_to_18_l844_84479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_coord_negation_l844_84484

-- Define the spherical coordinates type
structure SphericalCoord where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

-- Define the rectangular coordinates type
structure RectangularCoord where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the conversion function from spherical to rectangular coordinates
noncomputable def sphericalToRectangular (s : SphericalCoord) : RectangularCoord :=
  { x := s.ρ * Real.sin s.φ * Real.cos s.θ,
    y := s.ρ * Real.sin s.φ * Real.sin s.θ,
    z := s.ρ * Real.cos s.φ }

-- Define the theorem
theorem spherical_coord_negation (s : SphericalCoord) 
  (h1 : s.ρ = 3)
  (h2 : s.θ = 5 * π / 6)
  (h3 : s.φ = π / 4) :
  let r := sphericalToRectangular s
  let new_s := SphericalCoord.mk 3 (7 * π / 6) (π / 4)
  sphericalToRectangular new_s = RectangularCoord.mk r.x (-r.y) r.z := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_coord_negation_l844_84484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_l844_84444

-- Define set A
def A : Set ℕ := {0, 1}

-- Define set B
def B : Set ℕ := {x : ℕ | 0 < x ∧ x < 3}

-- Theorem statement
theorem union_of_A_and_B : A ∪ B = {0, 1, 2} := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_l844_84444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_k_l844_84456

noncomputable def f (x : ℝ) : ℝ := 7 * x^3 - 3 / x + 5

noncomputable def g (k : ℝ) (x : ℝ) : ℝ := x^3 - k + 2 * x

theorem find_k : ∃ k : ℝ, f 3 - g k 3 = 5 ∧ k = -155 := by
  use -155
  constructor
  · -- Prove f 3 - g (-155) 3 = 5
    show f 3 - g (-155) 3 = 5
    calc
      f 3 - g (-155) 3
        = (7 * 3^3 - 3 / 3 + 5) - (3^3 - (-155) + 2 * 3) := by rfl
      _ = (189 - 1 + 5) - (27 + 155 + 6) := by norm_num
      _ = 193 - 188 := by norm_num
      _ = 5 := by norm_num
  · -- Prove k = -155
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_k_l844_84456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l844_84497

/-- Represents the number of days it takes to complete the work -/
noncomputable def completion_time (a_time b_time : ℝ) (a_leave_before : ℕ) : ℕ :=
  let combined_rate := 1 / a_time + 1 / b_time
  let t := (1 + (a_leave_before : ℝ) / b_time) / combined_rate
  ⌈t⌉.toNat

/-- Theorem stating that given the conditions, the work takes 8 days to complete -/
theorem work_completion_time :
  completion_time 10 20 5 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l844_84497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_statements_correct_l844_84452

noncomputable def x : ℂ := Complex.exp (Complex.I * Real.pi / 3)
noncomputable def y : ℂ := Complex.exp (-Complex.I * Real.pi / 3)

theorem all_statements_correct :
  (x^6 + y^6 = 2) ∧
  (x^12 + y^12 = 2) ∧
  (x^18 + y^18 = 2) ∧
  (x^24 + y^24 = 2) ∧
  (x^30 + y^30 = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_statements_correct_l844_84452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_negative_five_pi_twelve_l844_84433

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem f_value_at_negative_five_pi_twelve
  (ω φ : ℝ)
  (h_monotone : ∀ x₁ x₂, π/6 < x₁ ∧ x₁ < x₂ ∧ x₂ < 2*π/3 → f ω φ x₁ < f ω φ x₂)
  (h_symmetry_left : ∀ x, f ω φ (π/6 - x) = f ω φ (π/6 + x))
  (h_symmetry_right : ∀ x, f ω φ (2*π/3 - x) = f ω φ (2*π/3 + x)) :
  f ω φ (-5*π/12) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_negative_five_pi_twelve_l844_84433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_school_trip_student_count_l844_84468

-- Define number_of_students as a function
def number_of_students (classroom : ℕ) : ℕ := sorry

theorem school_trip_student_count 
  (num_classrooms : ℕ) 
  (seats_per_bus : ℕ) 
  (num_buses : ℕ) 
  (h1 : num_classrooms = 87)
  (h2 : seats_per_bus = 2)
  (h3 : num_buses = 29)
  (h4 : ∀ c1 c2 : ℕ, c1 < num_classrooms → c2 < num_classrooms → 
        (number_of_students c1 = number_of_students c2)) :
  (num_classrooms * seats_per_bus * num_buses : ℕ) = 5046 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_school_trip_student_count_l844_84468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modified_equilateral_triangle_area_ratio_l844_84413

/-- The ratio of the area of a removed isosceles triangle to the area of the remaining central hexagon in a modified equilateral triangle. -/
theorem modified_equilateral_triangle_area_ratio :
  let side_length : ℝ := 12
  let isosceles_side : ℝ := 5
  let equilateral_area : ℝ := (Real.sqrt 3 / 4) * side_length^2
  let isosceles_base : ℝ := 6  -- Assumption based on the problem
  let isosceles_height : ℝ := Real.sqrt (isosceles_side^2 - (isosceles_base / 2)^2)
  let isosceles_area : ℝ := (1 / 2) * isosceles_base * isosceles_height
  let hexagon_area : ℝ := equilateral_area - 3 * isosceles_area
  isosceles_area / hexagon_area = 12 / (36 * Real.sqrt 3 - 36) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_modified_equilateral_triangle_area_ratio_l844_84413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projections_distance_equals_radius_l844_84454

structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

structure Point :=
  (x : ℝ)
  (y : ℝ)

def isOnCircumference (c : Circle) (p : Point) : Prop :=
  (p.x - c.center.1)^2 + (p.y - c.center.2)^2 = c.radius^2

noncomputable def projectOnDiameter (c : Circle) (p : Point) (θ : ℝ) : Point :=
  { x := c.center.1 + (p.x - c.center.1) * Real.cos θ + (p.y - c.center.2) * Real.sin θ,
    y := c.center.2 - (p.x - c.center.1) * Real.sin θ + (p.y - c.center.2) * Real.cos θ }

def distance (p1 p2 : Point) : ℝ :=
  ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)^(1/2)

theorem projections_distance_equals_radius (c : Circle) (p : Point) :
  isOnCircumference c p →
  ∀ θ : ℝ, distance (projectOnDiameter c p θ) (projectOnDiameter c p (θ + π/2)) = c.radius :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projections_distance_equals_radius_l844_84454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_center_l844_84488

noncomputable def f (z : ℂ) : ℂ := ((-2 + Complex.I * Real.sqrt 2) * z + (-Real.sqrt 2 - 10 * Complex.I)) / 2

noncomputable def c : ℂ := -1.5 * Real.sqrt 2 - 1.11 * Complex.I

theorem rotation_center : f c = c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_center_l844_84488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_theorem_l844_84416

def arithmetic_sequence (m n : ℕ) : Prop :=
  let a₁ := (Nat.choose (5*m) (11-2*m)) - (Nat.choose (11-3*m) (2*m-2))
  let d : ℚ := (Nat.choose 5 3) * (5/2)^2 * (-2/5)^3
  let aₙ : ℚ := a₁ + (n-1) * d
  aₙ = 104 - 4*n

theorem arithmetic_sequence_theorem (m n : ℕ) :
  m > 0 →
  n = (77^77 - 15) % 19 →
  arithmetic_sequence m n :=
by
  sorry

#check arithmetic_sequence_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_theorem_l844_84416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_g_l844_84472

/-- The function g(x) = 4x - x^4 --/
def g (x : ℝ) : ℝ := 4 * x - x^4

/-- The fourth root of 4 --/
noncomputable def fourth_root_of_4 : ℝ := Real.rpow 4 (1/4)

theorem max_value_of_g :
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ fourth_root_of_4 ∧
  g x = 3 ∧
  ∀ (y : ℝ), 0 ≤ y ∧ y ≤ fourth_root_of_4 → g y ≤ g x :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_g_l844_84472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conference_collaboration_l844_84459

/-- Represents a mathematician at the conference -/
structure Mathematician :=
  (id : Nat)

/-- Represents the collaboration relationship between mathematicians -/
def HasCollaborated (m1 m2 : Mathematician) : Prop := sorry

/-- The set of all mathematicians at the conference -/
def Mathematicians : Finset Mathematician := sorry

/-- The set of collaborators for a given mathematician -/
def Collaborators (m : Mathematician) : Finset Mathematician := sorry

theorem conference_collaboration :
  (∀ m ∈ Mathematicians, (Collaborators m).card ≥ 1337) →
  Mathematicians.card = 2005 →
  ∃ (a b c d : Mathematician),
    a ∈ Mathematicians ∧ b ∈ Mathematicians ∧ c ∈ Mathematicians ∧ d ∈ Mathematicians ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    HasCollaborated a b ∧ HasCollaborated a c ∧ HasCollaborated a d ∧
    HasCollaborated b c ∧ HasCollaborated b d ∧ HasCollaborated c d :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_conference_collaboration_l844_84459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_with_limited_digits_l844_84427

def digits (n : ℕ) : List ℕ :=
  if n < 10 then [n]
  else (n % 10) :: digits (n / 10)

theorem multiple_with_limited_digits (k : ℕ) (hk : k > 1) :
  ∃ m : ℕ, m % k = 0 ∧ m < k^4 ∧ (digits m).toFinset.card ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_with_limited_digits_l844_84427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_solutions_l844_84437

/-- A complex number z is a valid solution if it forms an equilateral triangle with 0 and z^2 --/
def is_valid_solution (z : ℂ) : Prop :=
  z ≠ 0 ∧
  z ≠ z^2 ∧
  z^2 ≠ 0 ∧
  Complex.abs z = Complex.abs (z^2 - z) ∧
  Complex.abs z = Complex.abs z^2 ∧
  Complex.abs (z^2 - z) = Complex.abs z^2

/-- The set of all valid solutions --/
def valid_solutions : Set ℂ :=
  {z : ℂ | is_valid_solution z}

theorem exactly_two_solutions :
  ∃ (s : Finset ℂ), s.card = 2 ∧ ∀ z : ℂ, z ∈ s ↔ is_valid_solution z := by
  sorry

#check exactly_two_solutions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_solutions_l844_84437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l844_84419

-- Define the piecewise function f(x)
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 + 1 else 3 - x

-- Define the solution set
def solution_set : Set ℝ := { x | -2 ≤ x ∧ x ≤ 2 }

-- Theorem statement
theorem inequality_solution_set :
  ∀ x : ℝ, f x ≥ 2 * x^2 - 3 ↔ x ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l844_84419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_slope_sum_l844_84440

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in slope-intercept form -/
structure Line where
  m : ℝ
  c : ℝ

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := Real.sqrt (1 - e.b^2 / e.a^2)

/-- Check if a point lies on an ellipse -/
def on_ellipse (e : Ellipse) (p : Point) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- The slope of a line passing through two points -/
noncomputable def line_slope (p1 p2 : Point) : ℝ := (p2.y - p1.y) / (p2.x - p1.x)

theorem ellipse_intersection_slope_sum 
  (e : Ellipse) 
  (M P Q : Point) 
  (l : Line) :
  eccentricity e = Real.sqrt 3 / 2 →
  on_ellipse e M →
  M.x = 4 ∧ M.y = 1 →
  P.y = l.m * P.x + l.c →
  Q.y = l.m * Q.x + l.c →
  on_ellipse e P →
  on_ellipse e Q →
  l.m ≠ -3 →
  P ≠ Q →
  line_slope M P + line_slope M Q = 0 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_slope_sum_l844_84440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_plums_balance_one_pear_l844_84491

/-- The weight of an apple -/
def apple : ℝ := sorry

/-- The weight of a pear -/
def pear : ℝ := sorry

/-- The weight of a plum -/
def plum : ℝ := sorry

/-- 3 apples and 1 pear weigh as much as 10 plums -/
axiom balance1 : 3 * apple + pear = 10 * plum

/-- 1 apple and 6 plums balance 1 pear -/
axiom balance2 : apple + 6 * plum = pear

/-- Prove that 7 plums balance one pear -/
theorem seven_plums_balance_one_pear : 7 * plum = pear := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_plums_balance_one_pear_l844_84491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_cos_and_sin_l844_84418

theorem intersection_of_cos_and_sin (φ : Real) : 
  (0 ≤ φ) → (φ ≤ π) → 
  (Real.cos (π/3) = Real.sin (2*(π/3) + φ)) →
  φ = π/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_cos_and_sin_l844_84418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_multiple_of_eight_l844_84486

theorem largest_multiple_of_eight (n : ℤ) : 
  (∀ m : ℤ, m > n → -(8 * m) ≤ -150) → 
  (-(8 * n) > -150) → 
  n = 144 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_multiple_of_eight_l844_84486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_squared_sum_diff_l844_84492

theorem sin_squared_sum_diff (α : ℝ) : 
  Real.sin (α - π/6)^2 + Real.sin (α + π/6)^2 - Real.sin α^2 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_squared_sum_diff_l844_84492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_foci_l844_84434

/-- The hyperbola defined by xy = 2 -/
def hyperbola : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 * p.2 = 2}

/-- The foci of the hyperbola -/
def foci : Set (ℝ × ℝ) := {(2, 2), (-2, -2)}

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem distance_between_foci :
  ∀ (p q : ℝ × ℝ), p ∈ foci → q ∈ foci → p ≠ q → distance p q = 4 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_foci_l844_84434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_open_unit_interval_l844_84464

noncomputable def f (x : ℝ) : ℝ := x + 1/x

theorem f_decreasing_on_open_unit_interval :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → x₂ < 1 → f x₁ > f x₂ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_open_unit_interval_l844_84464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_for_equation_l844_84441

theorem unique_solution_for_equation : ∃! x : ℝ, 
  (x^2010 + 1) * (x^2008 + x^2006 + x^2004 + x^2 + 1) = 2010 * x^2009 := by
  -- We claim that x = 1 is the unique solution
  use 1
  constructor
  
  -- Prove that x = 1 satisfies the equation
  · simp
    -- The actual computation is complex, so we'll skip it
    sorry
  
  -- Prove uniqueness
  · intro y hy
    -- We'll outline the proof structure, but skip the details
    have h_positive : y > 0 := by
      -- Proof that y must be positive
      sorry
    
    -- Divide both sides by y^2009
    have h_eq : (y + 1/y^1005) * (y^1004 + y^1002 + y^1000 + 1/y^1002 + 1/y^1004) = 2010 := by
      -- Proof of this equality
      sorry
    
    -- Apply AM-GM inequality
    have h_amgm1 : y + 1/y^1005 ≥ 2 := by
      -- Proof using AM-GM
      sorry
    have h_amgm2 : y^1004 + y^1002 + y^1000 + 1/y^1002 + 1/y^1004 ≥ 1005 := by
      -- Proof using AM-GM
      sorry
    
    -- Conclude y = 1
    have h_eq_one : y = 1 := by
      -- Final step of the proof
      sorry
    
    exact h_eq_one

-- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_for_equation_l844_84441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_donkey_mule_bags_theorem_l844_84477

/-- The number of bags carried by the donkey -/
def donkey_bags : ℕ := 5

/-- The number of bags carried by the mule -/
def mule_bags : ℕ := 7

/-- The weight of each bag -/
def bag_weight : ℕ := 1  -- We define this as a constant instead of a variable

theorem donkey_mule_bags_theorem :
  -- Condition 1: Equal total weight
  donkey_bags * bag_weight = mule_bags * bag_weight ∧
  -- Condition 2: Donkey gives one bag to mule
  (mule_bags + 1) * bag_weight = 2 * (donkey_bags - 1) * bag_weight ∧
  -- Condition 3: Mule gives one bag to donkey
  (mule_bags - 1) * bag_weight = (donkey_bags + 1) * bag_weight :=
by
  -- We use 'sorry' to skip the proof for now
  sorry

#eval donkey_bags
#eval mule_bags

end NUMINAMATH_CALUDE_ERRORFEEDBACK_donkey_mule_bags_theorem_l844_84477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_chord_length_l844_84453

-- Define the line and circle
def my_line (x y : ℝ) : Prop := x - Real.sqrt 3 * y + 8 = 0
def my_circle (x y r : ℝ) : Prop := x^2 + y^2 = r^2

-- Define the theorem
theorem intersection_chord_length (r : ℝ) 
  (h1 : r > 0)
  (h2 : ∃ (A B : ℝ × ℝ), 
    my_line A.1 A.2 ∧ my_line B.1 B.2 ∧ 
    my_circle A.1 A.2 r ∧ my_circle B.1 B.2 r ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 6) :
  r = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_chord_length_l844_84453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_and_sum_l844_84446

def number (G H : Nat) : Nat :=
  12 * 10^9 + G * 10^8 + 5 * 10^7 + H * 10^6 + 4782

theorem divisibility_and_sum :
  ∃! (G H : Nat), G < 10 ∧ H < 10 ∧ 
  (number G H).mod 72 = 0 ∧
  (Finset.sum (Finset.filter (fun (p : Nat × Nat) => 
    p.1 < 10 ∧ p.2 < 10 ∧ (number p.1 p.2).mod 72 = 0) 
    (Finset.product (Finset.range 10) (Finset.range 10)))
    (fun p => p.1 * p.2)) = 12 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_and_sum_l844_84446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_q_factor_change_l844_84428

/-- The function q is defined as (5w / (4h(z^2))) + kx -/
noncomputable def q (w h z x k : ℝ) : ℝ := (5 * w) / (4 * h * (z ^ 2)) + k * x

/-- The theorem states that when w is quadrupled, h is doubled, z is tripled, and x is halved,
    while keeping k constant, the function q is multiplied by a factor of 2/9 -/
theorem q_factor_change (w h z x k : ℝ) (h_nonzero : h ≠ 0) (z_nonzero : z ≠ 0) :
  q (4 * w) (2 * h) (3 * z) (x / 2) k = (2 / 9) * q w h z x k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_q_factor_change_l844_84428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_n0_smallest_N_l844_84414

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 / (2 - x)

-- Define the sequence a_n
noncomputable def a : ℕ → ℝ
| 0 => 3  -- a_1 = 3 for part (1)
| n + 1 => f (a n)

-- Part (1)
theorem existence_of_n0 : ∃ n0 : ℕ, ∀ n ≥ n0, a (n + 1) > a n := by
  sorry

-- Part (2)
-- Define a new sequence a_m parameterized by m and initial value
noncomputable def a_m (m : ℕ) (a1 : ℝ) : ℕ → ℝ
| 0 => a1
| n + 1 => f (a_m m a1 n)

-- Conditions for part (2)
def condition_part2 (m : ℕ) (a1 : ℝ) : Prop :=
  m ≠ 1 ∧ 1 + 1 / (m : ℝ) < a1 ∧ a1 < (m : ℝ) / ((m : ℝ) - 1)

-- Theorem for part (2)
theorem smallest_N (m : ℕ) (a1 : ℝ) (h : condition_part2 m a1) :
  ∃ N : ℕ, (∀ n ≥ N, 0 < a_m m a1 n ∧ a_m m a1 n < 1) ∧
  (∀ N' < N, ∃ n ≥ N', ¬(0 < a_m m a1 n ∧ a_m m a1 n < 1)) ∧
  N = m + 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_n0_smallest_N_l844_84414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reduced_rates_fraction_l844_84417

theorem reduced_rates_fraction (hours_per_day : ℕ) (days_per_week : ℕ) 
  (weekday_reduced_hours : ℕ) (weekend_days : ℕ) :
  hours_per_day = 24 →
  days_per_week = 7 →
  weekday_reduced_hours = 12 →
  weekend_days = 2 →
  (weekday_reduced_hours * (days_per_week - weekend_days) + hours_per_day * weekend_days) / 
   (hours_per_day * days_per_week : ℚ) = 9 / 14 := by
  intros h1 h2 h3 h4
  simp [h1, h2, h3, h4]
  norm_num
  
#check reduced_rates_fraction

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reduced_rates_fraction_l844_84417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_fraction_with_20_percent_increase_l844_84490

theorem no_fraction_with_20_percent_increase : 
  ¬∃ (x y : ℕ+), 
    (Nat.Coprime x.val y.val) ∧ 
    ((x.val + 1 : ℚ) / (y.val + 1) = 6/5 * (x : ℚ) / y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_fraction_with_20_percent_increase_l844_84490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_inequality_l844_84451

noncomputable def f (x : ℝ) : ℝ := x - Real.log x + (2*x - 1) / x^2

theorem f_monotonicity_and_inequality :
  (∀ x ∈ Set.Ioo (0 : ℝ) 1 ∪ Set.Ioi (Real.sqrt 2), deriv f x > 0) ∧
  (∀ x ∈ Set.Ioo 1 (Real.sqrt 2), deriv f x < 0) ∧
  (∀ x ∈ Set.Icc 1 4, f x > deriv f x + 3/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_inequality_l844_84451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_sum_matrix_to_zero_matrix_l844_84482

/-- Represents an operation on a matrix -/
structure MatrixOperation (n : ℕ) where
  row : Fin n
  col_add : Fin n
  col_sub : Fin n

/-- Applies a single operation to a matrix -/
def apply_operation {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ) (op : MatrixOperation n) : Matrix (Fin n) (Fin n) ℝ :=
  sorry

/-- Checks if a matrix is the zero matrix -/
def is_zero_matrix {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ) : Prop :=
  ∀ i j, A i j = 0

/-- The main theorem -/
theorem zero_sum_matrix_to_zero_matrix {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ) :
  (∀ i, (Finset.univ.sum (λ j => A i j) = 0)) →
  (∀ j, (Finset.univ.sum (λ i => A i j) = 0)) →
  ∃ (ops : List (MatrixOperation n)), is_zero_matrix (ops.foldl apply_operation A) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_sum_matrix_to_zero_matrix_l844_84482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_sum_l844_84478

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^3 + Real.log (Real.sqrt (x^2 + 1) + x)

-- State the theorem
theorem function_value_sum (a b : ℝ) 
  (h : f (-a) + f (-b) - 3 = f a + f b + 3) : 
  f a + f b = -3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_sum_l844_84478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2015_value_l844_84489

noncomputable def sequence_a : ℕ → ℝ
  | 0 => Real.sqrt 3  -- Add case for 0
  | 1 => Real.sqrt 3
  | n + 2 => ⌊sequence_a (n + 1)⌋ + 1 / (sequence_a (n + 1) - ⌊sequence_a (n + 1)⌋)

theorem a_2015_value :
  sequence_a 2015 = 3021 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2015_value_l844_84489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_specific_triangle_l844_84494

/-- The radius of the inscribed circle of a triangle with side lengths a, b, and c. -/
noncomputable def inscribed_circle_radius (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  area / s

/-- Theorem: The radius of the inscribed circle in triangle ABC with side lengths 22, 12, and 14 is √10. -/
theorem inscribed_circle_radius_specific_triangle :
  inscribed_circle_radius 22 12 14 = Real.sqrt 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_specific_triangle_l844_84494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_approx_l844_84411

/-- The time (in seconds) it takes for two trains to completely pass by each other -/
noncomputable def train_passing_time (train_a_speed_kmph : ℝ) (train_b_speed_kmph : ℝ) 
                       (pole_crossing_time : ℝ) (platform_length : ℝ) : ℝ :=
  let train_a_speed_mps := train_a_speed_kmph * (5/18)
  let train_b_speed_mps := train_b_speed_kmph * (5/18)
  let train_a_length := train_a_speed_mps * pole_crossing_time
  let total_distance := train_a_length + platform_length
  let relative_speed := train_a_speed_mps + train_b_speed_mps
  total_distance / relative_speed

/-- Theorem stating that the train passing time is approximately 20.44 seconds -/
theorem train_passing_time_approx : 
  ∃ ε > 0, |train_passing_time 36 45 12 340 - 20.44| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_approx_l844_84411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_cost_increase_l844_84467

/-- The percent increase in the combined cost of a bicycle, helmet, and gloves --/
theorem combined_cost_increase (bicycle_cost helmet_cost gloves_cost : ℝ)
  (bicycle_increase helmet_increase gloves_increase : ℝ) :
  let original_total := bicycle_cost + helmet_cost + gloves_cost
  let new_bicycle_cost := bicycle_cost * (1 + bicycle_increase)
  let new_helmet_cost := helmet_cost * (1 + helmet_increase)
  let new_gloves_cost := gloves_cost * (1 + gloves_increase)
  let new_total := new_bicycle_cost + new_helmet_cost + new_gloves_cost
  let percent_increase := (new_total - original_total) / original_total * 100
  bicycle_cost = 200 →
  helmet_cost = 50 →
  gloves_cost = 30 →
  bicycle_increase = 0.1 →
  helmet_increase = 0.15 →
  gloves_increase = 0.2 →
  ∃ (ε : ℝ), abs (percent_increase - 11.96) < ε ∧ ε > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_cost_increase_l844_84467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_y_y_4_is_integer_smallest_n_for_integer_y_l844_84431

noncomputable def y : ℕ → ℝ
  | 0 => 0  -- Added case for 0
  | 1 => (4 : ℝ) ^ (1/4)
  | 2 => ((4 : ℝ) ^ (1/4)) ^ ((4 : ℝ) ^ (1/4))
  | n + 3 => (y (n + 2)) ^ ((4 : ℝ) ^ (1/4))

theorem smallest_integer_y (n : ℕ) : n < 4 → ¬(∃ m : ℤ, y n = m) := by
  sorry

theorem y_4_is_integer : ∃ m : ℤ, y 4 = m := by
  sorry

theorem smallest_n_for_integer_y : 
  (∃ n : ℕ, ∃ m : ℤ, y n = m) ∧ 
  (∀ k : ℕ, k < 4 → ¬(∃ m : ℤ, y k = m)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_y_y_4_is_integer_smallest_n_for_integer_y_l844_84431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_edge_sum_equals_24_plus_roots_l844_84476

/-- A rectangular solid with dimensions in geometric progression -/
structure GeometricProgressionSolid where
  a : ℝ
  r : ℝ
  volume_eq : a^3 = 432
  surface_area_eq : 2 * (a^2 / r + a^2 * r + a^2) = 396
  r_positive : r > 0

/-- The sum of the lengths of all edges of the rectangular solid -/
noncomputable def edge_sum (solid : GeometricProgressionSolid) : ℝ :=
  4 * (solid.a / solid.r + solid.a + solid.a * solid.r)

/-- Theorem stating that the sum of the lengths of all edges is 24(1 + ∛2 + ∛4) -/
theorem edge_sum_equals_24_plus_roots (solid : GeometricProgressionSolid) :
  edge_sum solid = 24 * (1 + Real.rpow 2 (1/3) + Real.rpow 4 (1/3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_edge_sum_equals_24_plus_roots_l844_84476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_angle_at_point_l844_84445

noncomputable section

/-- The curve function -/
def f (x : ℝ) : ℝ := (1/2) * x^2 - 2

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := x

/-- The point on the curve -/
def point : ℝ × ℝ := (1, -3/2)

/-- The angle of inclination of the tangent line -/
def angle_of_inclination : ℝ := Real.pi/4

theorem tangent_angle_at_point :
  let slope := f' point.fst
  Real.arctan slope = angle_of_inclination := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_angle_at_point_l844_84445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_theorem_l844_84487

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain (x : ℝ) : Prop := -Real.pi/2 < x ∧ x < Real.pi/2

-- State the theorem
theorem solution_set_theorem 
  (h1 : ∀ x, domain x → f x + f (-x) = 0)
  (h2 : ∀ x, 0 < x → x < Real.pi/2 → (deriv f x) * Real.cos x + f x * Real.sin x < 0) :
  {x | domain x ∧ f x < Real.sqrt 2 * f (Real.pi/4) * Real.cos x} = {x | Real.pi/4 < x ∧ x < Real.pi/2} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_theorem_l844_84487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_l844_84435

/-- Given a function f: ℝ → ℝ satisfying f(x+1) = x + 2x^2 for all x ∈ ℝ,
    prove that f(x) = 2x^2 - 3x + 1 for all x ∈ ℝ. -/
theorem function_composition (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x + 1) = x + 2 * x^2) :
  ∀ x : ℝ, f x = 2 * x^2 - 3 * x + 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_l844_84435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_worth_calculation_l844_84460

/-- Represents the total worth of a stock in Rupees -/
def stock_worth (x : ℝ) : Prop := true

/-- The profit percentage on the first part of the stock -/
def profit_percentage : ℝ := 0.20

/-- The loss percentage on the second part of the stock -/
def loss_percentage : ℝ := 0.10

/-- The fraction of stock sold at a profit -/
def profit_fraction : ℝ := 0.20

/-- The overall loss in Rupees -/
def overall_loss : ℝ := 500

theorem stock_worth_calculation (x : ℝ) (h : stock_worth x) :
  profit_fraction * profit_percentage * x - (1 - profit_fraction) * loss_percentage * x = overall_loss →
  x = 12500 := by
  sorry

#check stock_worth_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_worth_calculation_l844_84460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosA_value_angle_comparison_l844_84474

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the given conditions
noncomputable def given_triangle : Triangle where
  A := Real.arccos (Real.sqrt 6 / 3)  -- We define A using the expected result for cosA
  B := 2 * Real.arccos (Real.sqrt 6 / 3)  -- B = 2A
  C := Real.pi - (3 * Real.arccos (Real.sqrt 6 / 3))  -- A + B + C = π
  a := 3
  b := 2 * Real.sqrt 6
  c := Real.sqrt ((2 * Real.sqrt 6)^2 + 3^2 - 2 * (2 * Real.sqrt 6) * 3 * (Real.sqrt 6 / 3))  -- Using law of cosines

-- Theorem statements
theorem cosA_value (t : Triangle) (h : t = given_triangle) : 
  Real.cos t.A = Real.sqrt 6 / 3 := by sorry

theorem angle_comparison (t : Triangle) (h : t = given_triangle) :
  t.B < t.C := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosA_value_angle_comparison_l844_84474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_product_l844_84410

/-- Predicate for an equilateral triangle in 2D space --/
def IsEquilateralTriangle (p1 p2 p3 : ℝ × ℝ) : Prop :=
  let d12 := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
  let d23 := Real.sqrt ((p2.1 - p3.1)^2 + (p2.2 - p3.2)^2)
  let d31 := Real.sqrt ((p3.1 - p1.1)^2 + (p3.2 - p1.2)^2)
  d12 = d23 ∧ d23 = d31

theorem equilateral_triangle_product (a b : ℝ) : 
  IsEquilateralTriangle (0, 0) (a, 15) (b, 41) → a * b = -676 / 9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_product_l844_84410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_difference_specific_angles_l844_84415

theorem sine_difference_specific_angles (α β : Real) : 
  0 < α ∧ α < Real.pi/2 →  -- α is acute
  0 < β ∧ β < Real.pi/2 →  -- β is acute
  Real.sin α = 2 * Real.sin β →
  Real.cos α = 1/2 * Real.cos β →
  Real.sin (α - β) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_difference_specific_angles_l844_84415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_points_on_ellipse_l844_84448

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- The foci of the ellipse -/
noncomputable def foci : ℝ × ℝ × ℝ × ℝ := sorry

/-- The area of a triangle formed by a point on the ellipse and the foci -/
noncomputable def triangleArea (x y : ℝ) : ℝ := sorry

/-- The theorem stating that there are exactly two points on the ellipse
    forming triangles with the foci with area √3 -/
theorem two_points_on_ellipse :
  ∃! (s : Set (ℝ × ℝ)), 
    (∀ (x y : ℝ), (x, y) ∈ s ↔ ellipse x y ∧ triangleArea x y = Real.sqrt 3) ∧
    (∃ (l : List (ℝ × ℝ)), s = l.toFinset ∧ l.length = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_points_on_ellipse_l844_84448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l844_84429

/-- A triangle in a geometric figure -/
structure Triangle where
  area : ℝ

/-- A quadrilateral in a geometric figure -/
structure Quadrilateral where
  area : ℝ

/-- A figure composed of triangles and quadrilaterals -/
structure GeometricFigure where
  triangles : Finset Triangle
  quadrilaterals : Finset Quadrilateral

/-- Properties of the geometric figure -/
def ValidFigure (g : GeometricFigure) : Prop :=
  g.triangles.card = 4 ∧
  g.quadrilaterals.card = 3 ∧
  ∀ t ∈ g.triangles, t.area = 1

/-- Theorem: In a valid geometric figure, each quadrilateral has an area of √5 + 1 -/
theorem quadrilateral_area (g : GeometricFigure) (h : ValidFigure g) :
  ∀ q ∈ g.quadrilaterals, q.area = Real.sqrt 5 + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l844_84429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_statement_l844_84404

inductive CancerStatement
  | AIDSIncreasesRisk
  | CancerCellCharacteristics
  | NitritesAlterGenes
  | ContactIncreasesRisk

def isCorrectStatement (s : CancerStatement) : Bool :=
  match s with
  | .AIDSIncreasesRisk => true
  | .CancerCellCharacteristics => true
  | .NitritesAlterGenes => true
  | .ContactIncreasesRisk => false

theorem incorrect_statement :
  isCorrectStatement CancerStatement.ContactIncreasesRisk = false :=
by
  rfl

#eval isCorrectStatement CancerStatement.ContactIncreasesRisk

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_statement_l844_84404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_traffic_light_change_probability_l844_84471

/-- Represents the duration of each color in the traffic light cycle -/
structure TrafficLightCycle where
  green : ℕ
  yellow : ℕ
  red : ℕ

/-- Calculates the total cycle duration -/
def cycleDuration (cycle : TrafficLightCycle) : ℕ :=
  cycle.green + cycle.yellow + cycle.red

/-- Calculates the total duration of potential color change intervals -/
def changeIntervals (cycle : TrafficLightCycle) : ℕ :=
  3 * cycle.yellow -- Assuming color changes occur within the duration of the yellow light

/-- Theorem: Probability of observing a color change in a 5-second interval -/
theorem traffic_light_change_probability (cycle : TrafficLightCycle)
  (h1 : cycle.green = 45)
  (h2 : cycle.yellow = 5)
  (h3 : cycle.red = 50)
  (h4 : cycleDuration cycle = 100)
  (h5 : changeIntervals cycle = 15) :
  (changeIntervals cycle : ℚ) / (cycleDuration cycle : ℚ) = 3 / 20 := by
  sorry

-- Remove the #eval line as it's causing issues with universe levels

end NUMINAMATH_CALUDE_ERRORFEEDBACK_traffic_light_change_probability_l844_84471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_represents_three_lines_l844_84498

/-- A set is a line if it's the graph of a linear equation -/
def IsLine (s : Set (ℝ × ℝ)) : Prop :=
  ∃ (a b c : ℝ), s = {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0} ∧ (a ≠ 0 ∨ b ≠ 0)

/-- The equation represents three lines that do not all pass through a common point -/
theorem equation_represents_three_lines (x y : ℝ) : 
  (x + 2)^2 * (y - 1) * (x + y + 2) = (y + 2)^2 * (x - 1) * (x + y + 2) →
  ∃ (l₁ l₂ l₃ : Set (ℝ × ℝ)),
    (IsLine l₁ ∧ IsLine l₂ ∧ IsLine l₃) ∧
    (l₁ ∪ l₂ ∪ l₃).Subset {p : ℝ × ℝ | (p.1 + 2)^2 * (p.2 - 1) * (p.1 + p.2 + 2) = (p.2 + 2)^2 * (p.1 - 1) * (p.1 + p.2 + 2)} ∧
    ¬∃ (p : ℝ × ℝ), p ∈ l₁ ∩ l₂ ∩ l₃ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_represents_three_lines_l844_84498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_number_with_reversed_digits_l844_84457

def base_expansion (n : ℕ) (b : ℕ) : List ℕ :=
  sorry

def reverse_list (l : List ℕ) : List ℕ :=
  sorry

theorem exists_number_with_reversed_digits : ∃ N : ℕ,
  let base_5 := base_expansion N 5
  let base_7 := base_expansion N 7
  (base_5.length = 4) ∧
  (base_7 = reverse_list base_5) ∧
  (base_5.get? 1 = some 1) :=
sorry

#check exists_number_with_reversed_digits

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_number_with_reversed_digits_l844_84457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_year_after_2020_with_sum_5_l844_84499

def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

def nextYearWithSumOfDigits (startYear : ℕ) (targetSum : ℕ) : ℕ :=
  let rec find (year : ℕ) (fuel : ℕ) : ℕ :=
    if fuel = 0 then year
    else if sumOfDigits year = targetSum then year
    else find (year + 1) (fuel - 1)
  find (startYear + 1) 1000  -- Assuming we'll find the answer within 1000 years

theorem first_year_after_2020_with_sum_5 :
  nextYearWithSumOfDigits 2020 5 = 2021 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_year_after_2020_with_sum_5_l844_84499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_lines_fixed_point_l844_84426

/-- A line in 2D space defined by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The symmetry line y = x + 1 -/
def symmetry_line : Line := { slope := 1, intercept := 1 }

/-- Line l₁ with equation y - 2 = (k - 1)x -/
def l₁ (k : ℝ) : Line := { slope := k - 1, intercept := 2 }

/-- Checks if two lines are symmetric about the symmetry line -/
def symmetric_about (l1 l2 : Line) (s : Line) : Prop := sorry

/-- Checks if a line passes through a point -/
def passes_through (l : Line) (x y : ℝ) : Prop := y = l.slope * x + l.intercept

theorem symmetric_lines_fixed_point (k : ℝ) :
  ∃ l₂ : Line, symmetric_about (l₁ k) l₂ symmetry_line →
  passes_through l₂ 1 1 := by
  sorry

#check symmetric_lines_fixed_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_lines_fixed_point_l844_84426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_ratio_after_brownies_is_four_to_one_l844_84422

/-- Represents the weight of the care package at different stages --/
structure CarePackage where
  initialWeight : ℚ  -- Weight after first batch of jelly beans
  finalWeight : ℚ    -- Final weight after all items added
  weightRatio : ℚ    -- Ratio of weight after gummy worms to weight before

/-- Calculates the ratio of weight after adding brownies to weight before --/
noncomputable def weightRatioAfterBrownies (cp : CarePackage) : ℚ :=
  let weightBeforeGummyWorms := cp.finalWeight / cp.weightRatio
  let weightAfterBrownies := weightBeforeGummyWorms - cp.initialWeight
  (cp.initialWeight + weightAfterBrownies) / cp.initialWeight

/-- Theorem: The ratio of weight after adding brownies to weight before is 4:1 --/
theorem weight_ratio_after_brownies_is_four_to_one (cp : CarePackage)
    (h1 : cp.initialWeight = 2)
    (h2 : cp.finalWeight = 16)
    (h3 : cp.weightRatio = 2) :
    weightRatioAfterBrownies cp = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_ratio_after_brownies_is_four_to_one_l844_84422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_overlap_rotated_squares_l844_84480

/-- The area of overlap between two unit squares with one rotated -/
theorem area_overlap_rotated_squares (α : Real) 
  (h1 : 0 < α ∧ α < Real.pi / 2)
  (h2 : Real.cos α = 3 / 5) : 
  ∃ A, A = (1 : Real) / 3 ∧ 
  A = (1 - Real.sin α) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_overlap_rotated_squares_l844_84480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_rise_ratio_l844_84412

/-- Represents a right circular cone --/
structure Cone where
  radius : ℝ
  height : ℝ

/-- Represents a sphere --/
structure Sphere where
  radius : ℝ

/-- Calculates the volume of a cone --/
noncomputable def coneVolume (c : Cone) : ℝ := (1/3) * Real.pi * c.radius^2 * c.height

/-- Calculates the volume of a sphere --/
noncomputable def sphereVolume (s : Sphere) : ℝ := (4/3) * Real.pi * s.radius^3

/-- Theorem: The ratio of liquid level rise in two cones after dropping equal spheres --/
theorem liquid_rise_ratio 
  (cone1 cone2 : Cone) 
  (sphere : Sphere) 
  (h : coneVolume cone1 = coneVolume cone2) 
  (r1 : cone1.radius = 4) 
  (r2 : cone2.radius = 8) 
  (sr : sphere.radius = 2) :
  let rise1 := (coneVolume cone1 + sphereVolume sphere) / (Real.pi * cone1.radius^2) - cone1.height
  let rise2 := (coneVolume cone2 + sphereVolume sphere) / (Real.pi * cone2.radius^2) - cone2.height
  rise1 / rise2 = 4 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_rise_ratio_l844_84412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_journey_distance_squared_l844_84408

/-- Represents the distance traveled in a single step -/
noncomputable def step_distance (n : ℕ) : ℝ := n + n / 10

/-- Represents the net displacement after one cycle of 4 steps -/
noncomputable def cycle_displacement : ℝ × ℝ :=
  (step_distance 4 - step_distance 2, step_distance 1 - step_distance 3)

/-- The number of complete cycles in 1000 steps -/
def num_cycles : ℕ := 1000 / 4

/-- The theorem stating the square of the straight-line distance -/
theorem ant_journey_distance_squared :
  let (dx, dy) := cycle_displacement
  (num_cycles * dx)^2 + (num_cycles * dy)^2 = 605000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_journey_distance_squared_l844_84408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_complex_fraction_l844_84443

theorem modulus_of_complex_fraction : Complex.abs ((2 - Complex.I) / (1 + Complex.I)) = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_complex_fraction_l844_84443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_coefficient_l844_84425

theorem binomial_expansion_coefficient (n : ℕ) : 
  (∀ k : ℕ, k ≤ n → (n.choose k) * (2^(n-k)) * (((-1 : ℤ)^k).toNat) * (x^(2*(n-k)-k)) = 
             (n.choose (k+1)) * (2^(n-k-1)) * (((-1 : ℤ)^(k+1)).toNat) * (x^(2*(n-k-1)-(k+1)))) →
  (n.choose 2) * (2^(n-2)) * (((-1 : ℤ)^2).toNat) = 80 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_coefficient_l844_84425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_proof_l844_84401

-- Define the radius of the circles
def radius : ℝ := 1

-- Define the area of the shaded region
noncomputable def shaded_area : ℝ := 6 + Real.sqrt 3 - 2 * Real.pi

-- Theorem statement
theorem shaded_area_proof :
  let circles_radius : ℝ := radius
  let circles_tangent : Prop := True  -- Assumption that circles are tangent
  let common_tangents : Prop := True  -- Assumption that boundaries are common tangents
  shaded_area = 6 + Real.sqrt 3 - 2 * Real.pi :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_proof_l844_84401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_three_eighths_l844_84405

-- Define the function g
noncomputable def g : ℝ → ℝ := sorry

-- State the properties of g
axiom g_domain (x : ℝ) : 0 ≤ x ∧ x ≤ 1 → g x ∈ Set.Icc 0 1
axiom g_zero : g 0 = 0
axiom g_monotone {x y : ℝ} : 0 ≤ x → x < y → y ≤ 1 → g x ≤ g y
axiom g_complement {x : ℝ} : 0 ≤ x → x ≤ 1 → g (1 - x) = 1 - g x
axiom g_quarter {x : ℝ} : 0 ≤ x → x ≤ 1 → g (x / 4) = g x / 3

-- State the theorem to be proved
theorem g_three_eighths : g (3/8) = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_three_eighths_l844_84405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_proof_l844_84421

/-- Represents a circle in the 2D plane -/
structure Circle where
  a : ℚ
  b : ℚ
  c : ℚ
  equation : (x y : ℚ) → Prop := λ x y => x^2 + y^2 + a*x + b*y + c = 0

/-- The first given circle: x^2 + y^2 - 5x = 0 -/
def circle1 : Circle := { a := -5, b := 0, c := 0 }

/-- The second given circle: x^2 + y^2 = 2 -/
def circle2 : Circle := { a := 0, b := 0, c := -2 }

/-- The point M(2, -2) -/
def point_M : ℚ × ℚ := (2, -2)

/-- The equation of the circle we want to prove -/
def target_circle : Circle := { a := -15/4, b := 0, c := -1/2 }

theorem circle_equation_proof :
  (∀ (x y : ℚ), circle1.equation x y ∧ circle2.equation x y →
    target_circle.equation x y) ∧
  target_circle.equation point_M.1 point_M.2 := by
  sorry

#check circle_equation_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_proof_l844_84421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harry_bus_time_l844_84496

-- Define the variables
noncomputable def remaining_bus_time : ℝ := 25
noncomputable def total_travel_time : ℝ := 60

-- Define the function for walking time
noncomputable def walking_time (bus_time : ℝ) : ℝ := bus_time / 2

-- Theorem statement
theorem harry_bus_time :
  ∃ (x : ℝ),
    x + remaining_bus_time + walking_time (x + remaining_bus_time) = total_travel_time ∧
    x = 15 := by
  use 15
  constructor
  · simp [remaining_bus_time, total_travel_time, walking_time]
    norm_num
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_harry_bus_time_l844_84496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_danai_decorations_l844_84430

/-- The total number of decorations Danai puts up for Halloween -/
def total_decorations (skulls broomsticks spiderwebs pumpkins cauldrons lanterns_per_tree trees scarecrows_per_front_tree front_trees stickers windows new_decorations new_decorations_usage_rate earlier_decorations : ℕ) : ℕ :=
  skulls + broomsticks + spiderwebs + pumpkins + cauldrons + 
  (lanterns_per_tree * trees) + (scarecrows_per_front_tree * front_trees) + 
  (stickers / 2 / windows) * windows +
  (new_decorations * new_decorations_usage_rate / 100) + earlier_decorations

/-- Theorem stating that Danai will put up 102 decorations in total -/
theorem danai_decorations : 
  total_decorations 12 4 12 24 1 3 5 1 2 30 3 25 70 15 = 102 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_danai_decorations_l844_84430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_proof_l844_84400

theorem exponent_proof (x : ℝ) : (12 : ℝ)^x * (6^4 : ℝ) / 432 = 36 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_proof_l844_84400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_passing_time_l844_84481

/- Define the parameters for the trains -/
noncomputable def train_a_length : ℝ := 100  -- length in meters
noncomputable def train_b_length : ℝ := 150  -- length in meters
noncomputable def train_a_speed : ℝ := 60 * 1000 / 3600  -- speed in m/s
noncomputable def train_b_speed : ℝ := 75 * 1000 / 3600  -- speed in m/s

/- Theorem statement -/
theorem trains_passing_time :
  let total_length := train_a_length + train_b_length
  let relative_speed := train_a_speed + train_b_speed
  abs ((total_length / relative_speed) - 6.67) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_passing_time_l844_84481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_area_quadrilateral_l844_84469

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0

/-- Represents a parabola with parameter p -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- The semi-latus rectum of an ellipse -/
noncomputable def semiLatusRectum (e : Ellipse) : ℝ :=
  e.b^2 / e.a

theorem minimum_area_quadrilateral (e : Ellipse) (p : Parabola) :
  eccentricity e = 1/2 →
  e.b = 2 * Real.sqrt 3 →
  semiLatusRectum e = p.p / 2 →
  ∃ (area : ℝ),
    (∀ (k : ℝ), k ≠ 0 →
      let l1 := fun (x : ℝ) ↦ k * (x - 1)
      let l2 := fun (x : ℝ) ↦ -1/k * (x - 1)
      let quad_area := 16 * Real.sqrt ((k^2 + 2 + 1/k^2) * (2 * (k^2 + 2 + 1/k^2) + 1))
      quad_area ≥ area) ∧
    area = 96 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_area_quadrilateral_l844_84469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_product_equals_32_l844_84447

theorem fraction_product_equals_32 : 
  (1/2 : ℚ) * 4 * (1/8) * 16 * (1/32) * 64 * (1/128) * 256 * (1/512) * 1024 = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_product_equals_32_l844_84447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ticket_123456_is_lucky_l844_84439

/-- A function that represents a valid arithmetic expression using the digits of a ticket number -/
def ValidExpression (digits : List Nat) : Nat := sorry

/-- The ticket number -/
def ticket_number : List Nat := [1, 2, 3, 4, 5, 6]

/-- A ticket is lucky if there exists a valid expression using its digits that equals 100 -/
def is_lucky (ticket : List Nat) : Prop :=
  ∃ (expr : Nat), expr = ValidExpression ticket ∧ expr = 100

theorem ticket_123456_is_lucky : is_lucky ticket_number := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ticket_123456_is_lucky_l844_84439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_2012_l844_84485

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 2000 then 2 * Real.cos (Real.pi / 3 * x) else x - 12

-- State the theorem
theorem f_composition_2012 : f (f 2012) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_2012_l844_84485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_solution_l844_84406

theorem no_integer_solution : ¬∃ (x y : ℤ), (4^x.toNat + y = 34) ∧ (2^x.toNat - y = 20) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_solution_l844_84406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_n_l844_84470

/-- A graph representing the connections between circles. -/
structure Graph :=
  (V : Type) -- Vertices
  (E : V → V → Prop) -- Edges

/-- Predicate to check if two numbers are connected in the graph. -/
def connected (G : Graph) (a b : G.V) : Prop := G.E a b

/-- Predicate to check if a number arrangement is valid for a given n. -/
def valid_arrangement (G : Graph) (n : ℕ) (f : G.V → ℕ) : Prop :=
  ∀ a b : G.V,
    (¬connected G a b → (f a ^ 2 + f b ^ 2).gcd n = 1) ∧
    (connected G a b → ∃ d > 1, d ∣ (f a ^ 2 + f b ^ 2) ∧ d ∣ n)

/-- The graph does not contain a cycle of length 3. -/
def no_triangle (G : Graph) : Prop :=
  ∀ a b c : G.V, ¬(G.E a b ∧ G.E b c ∧ G.E c a)

/-- The main theorem stating that 65 is the smallest n for which a valid arrangement exists. -/
theorem smallest_valid_n (G : Graph) (h : no_triangle G) :
  (∃ n : ℕ, ∃ f : G.V → ℕ, valid_arrangement G n f) →
  (∀ m : ℕ, m < 65 → ¬∃ f : G.V → ℕ, valid_arrangement G m f) ∧
  (∃ f : G.V → ℕ, valid_arrangement G 65 f) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_n_l844_84470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bromine_atom_count_l844_84493

/-- Represents the number of atoms of an element in a compound -/
structure AtomCount where
  count : ℕ

/-- Represents the molecular weight in g/mol -/
structure MolecularWeight where
  weight : ℝ

/-- Represents the atomic weight in g/mol -/
structure AtomicWeight where
  weight : ℝ

/-- The atomic weight of hydrogen -/
def hydrogen_weight : AtomicWeight := ⟨1⟩

/-- The atomic weight of oxygen -/
def oxygen_weight : AtomicWeight := ⟨16⟩

/-- The atomic weight of bromine -/
def bromine_weight : AtomicWeight := ⟨80⟩

/-- The number of hydrogen atoms in the compound -/
def hydrogen_count : AtomCount := ⟨1⟩

/-- The number of oxygen atoms in the compound -/
def oxygen_count : AtomCount := ⟨3⟩

/-- The molecular weight of the compound -/
def compound_weight : MolecularWeight := ⟨129⟩

instance : HMul AtomCount AtomicWeight ℝ where
  hMul a w := a.count * w.weight

theorem bromine_atom_count :
  ∃ (br_count : AtomCount),
    br_count * bromine_weight = 
      compound_weight.weight - (hydrogen_count * hydrogen_weight + oxygen_count * oxygen_weight) ∧
    br_count.count = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bromine_atom_count_l844_84493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_problem_l844_84432

/-- Proof that B invested $200 given the conditions of the investment problem -/
theorem investment_problem (a_investment : ℝ) (a_months b_months : ℕ) (total_profit : ℝ) (a_share : ℝ) :
  a_investment = 100 ∧ 
  a_months = 12 ∧ 
  b_months = 6 ∧ 
  total_profit = 100 ∧ 
  a_share = 50 →
  ∃ b_investment : ℝ, 
    b_investment = 200 ∧ 
    (a_investment * (a_months : ℝ)) / (a_investment * (a_months : ℝ) + b_investment * (b_months : ℝ)) = a_share / total_profit :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_problem_l844_84432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l844_84462

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 1/2 - Real.log x

theorem f_properties (a : ℝ) :
  (∀ x > 0, f a x = a * x^2 - 1/2 - Real.log x) →
  (∃ x₀ > 0, x₀ = 2 ∧ (deriv (f a)) x₀ = -2/3) →
  (∀ x ∈ Set.Ioo 0 (Real.sqrt 12), (deriv (f a)) x < 0) ∧
  (∀ x ∈ Set.Ioi (Real.sqrt 12), (deriv (f a)) x > 0) ∧
  (∀ x > 1, f a x > 1/x - Real.exp (1-x)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l844_84462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_r_plus_s_equals_54_l844_84465

theorem r_plus_s_equals_54 (p q r s t u : ℕ) 
  (h_order : p < q ∧ q < r ∧ r < s ∧ s < t ∧ t < u)
  (h_sums : Finset.toSet {p + q, p + r, p + s, p + t, p + u, 
                          q + r, q + s, q + t, q + u, 
                          r + s, r + t, r + u, 
                          s + t, s + u, 
                          t + u} = 
            {25, 30, 38, 41, 49, 52, 54, 63, 68, 76, 79, 90, 95, 103, 117}) :
  r + s = 54 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_r_plus_s_equals_54_l844_84465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_sequences_count_l844_84403

/-- Counts the number of valid sequences of zeros and ones -/
def count_valid_sequences (n : ℕ) : ℕ :=
  let count_consecutive (k : ℕ) : ℕ := if k = n then 0 else n - k + 1
  (List.range (n + 1)).map count_consecutive |>.sum

/-- The length of the sequences -/
def sequence_length : ℕ := 15

/-- The number of zeros in an all-zero sequence -/
def all_zeros : ℕ := 6

/-- The number of ones in an all-one sequence -/
def all_ones : ℕ := 9

theorem valid_sequences_count :
  count_valid_sequences sequence_length - 
  (count_valid_sequences all_zeros + count_valid_sequences all_ones - 2) = 269 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_sequences_count_l844_84403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_sum_less_than_cotangent_sum_in_obtuse_triangle_l844_84458

-- Define a structure for a triangle
structure Triangle where
  α : Real
  β : Real
  γ : Real
  sum_to_pi : α + β + γ = Real.pi
  positive : 0 < α ∧ 0 < β ∧ 0 < γ

-- Define an obtuse triangle
def ObtuseTriangle (t : Triangle) : Prop :=
  Real.pi / 2 < t.α ∨ Real.pi / 2 < t.β ∨ Real.pi / 2 < t.γ

-- State the theorem
theorem tangent_sum_less_than_cotangent_sum_in_obtuse_triangle (t : Triangle) 
  (h : ObtuseTriangle t) : 
  Real.tan t.α + Real.tan t.β + Real.tan t.γ < 
  (Real.tan t.α)⁻¹ + (Real.tan t.β)⁻¹ + (Real.tan t.γ)⁻¹ :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_sum_less_than_cotangent_sum_in_obtuse_triangle_l844_84458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_when_a_zero_f_unique_zero_iff_a_positive_l844_84420

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - 1 / x - (a + 1) * Real.log x

theorem f_max_when_a_zero :
  ∃ (max : ℝ), max = -1 ∧ ∀ x, x > 0 → f 0 x ≤ max :=
sorry

theorem f_unique_zero_iff_a_positive :
  ∀ a : ℝ, (∃! x, x > 0 ∧ f a x = 0) ↔ a > 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_when_a_zero_f_unique_zero_iff_a_positive_l844_84420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_ab_is_two_min_value_is_two_l844_84423

-- Define the slopes of the two lines
noncomputable def slope1 (a : ℝ) := -a^2
noncomputable def slope2 (a b : ℝ) := b / (a^2 + 1)

-- State the theorem
theorem min_abs_ab_is_two (a b : ℝ) : 
  (slope1 a * slope2 a b = -1) → -- Perpendicularity condition
  (∀ x y : ℝ, |x * y| ≥ 2) →     -- Minimum value condition
  (∃ x y : ℝ, x * y = 2) →       -- Achievability of minimum
  (|a * b| ≥ 2 ∧ ∃ a b : ℝ, |a * b| = 2) := by
  sorry

-- Additional theorem to show the minimum value is indeed 2
theorem min_value_is_two :
  ∃ a b : ℝ, slope1 a * slope2 a b = -1 ∧ |a * b| = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_ab_is_two_min_value_is_two_l844_84423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_roots_l844_84407

/-- A polynomial with integer coefficients -/
def IntPolynomial := Polynomial ℤ

/-- Predicate to check if a number is a three-digit positive integer -/
def isThreeDigitPositive (n : ℤ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- Theorem: If P is a polynomial with integer coefficients and P(n) is a three-digit 
    positive integer for n = 1, 2, 3, ..., 1998, then P has no integer roots -/
theorem no_integer_roots (P : IntPolynomial) 
  (h : ∀ n : ℤ, 1 ≤ n ∧ n ≤ 1998 → isThreeDigitPositive (P.eval n)) : 
  ∀ m : ℤ, P.eval m ≠ 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_roots_l844_84407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_for_integer_series_l844_84424

noncomputable def a : ℝ := Real.pi / 2016

noncomputable def series (n : ℕ) : ℝ :=
  2 * (Finset.sum (Finset.range n) (fun k => Real.cos ((k + 1)^2 * a) * Real.sin ((k + 1) * a)))

theorem smallest_integer_for_integer_series :
  ∀ m : ℕ, m > 0 → m < 72 → ¬(∃ k : ℤ, series m = k) ∧
  ∃ k : ℤ, series 72 = k :=
by sorry

#check smallest_integer_for_integer_series

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_for_integer_series_l844_84424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_profit_is_1600_l844_84455

/-- Calculates the total profit given investments and one partner's share -/
def calculate_total_profit (x_investment y_investment x_share : ℚ) : ℚ :=
  let total_investment := x_investment + y_investment
  let x_ratio := x_investment / total_investment
  let y_ratio := y_investment / total_investment
  let y_share := (y_ratio * x_share) / x_ratio
  x_share + y_share

/-- Theorem: Given the investments and x's share, the total profit is 1600 -/
theorem total_profit_is_1600 (x_investment y_investment x_share : ℚ) 
  (h1 : x_investment = 5000)
  (h2 : y_investment = 15000)
  (h3 : x_share = 400) :
  calculate_total_profit x_investment y_investment x_share = 1600 := by
  sorry

#eval calculate_total_profit 5000 15000 400

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_profit_is_1600_l844_84455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_value_f_m_range_l844_84438

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log ((1 - a*x) / (x - 1)) / Real.log (1/2)

-- Theorem 1: If f is odd, then a = -1
theorem a_value (h : ∀ x, f a x = -f a (-x)) : a = -1 := by sorry

-- Define the function f with a = -1
noncomputable def f' (x : ℝ) : ℝ := Real.log ((x + 1) / (x - 1)) / Real.log (1/2)

-- Theorem 2: f' is monotonically increasing in (1, +∞)
theorem f'_increasing : ∀ x y, 1 < x → x < y → f' x < f' y := by sorry

-- Theorem 3: If f'(x) > (1/2)ˣ + m for all x in [3, 4], then m < -9/8
theorem m_range (h : ∀ x, 3 ≤ x → x ≤ 4 → f' x > (1/2)^x + m) : m < -9/8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_value_f_m_range_l844_84438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_zero_iff_a_eq_half_l844_84495

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*x + a*(Real.exp (x - 1) + Real.exp (-x + 1))

/-- The theorem stating that f(x) has a unique zero point if and only if a = 1/2 -/
theorem unique_zero_iff_a_eq_half (a : ℝ) :
  (∃! x, f a x = 0) ↔ a = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_zero_iff_a_eq_half_l844_84495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_transitive_parallel_complementary_angle_sum_parallel_transitive_l844_84442

-- Define the perpendicular and parallel relations
def perpendicular (a b : Line) : Prop := sorry
def parallel (a b : Line) : Prop := sorry

-- Define a type for angles
structure Angle where
  value : ℝ

-- Define a function to calculate the sum of two angles
def angleSum (α β : Angle) : Angle :=
  ⟨α.value + β.value⟩

-- Define what it means for two angles to be complementary
def complementary (α β : Angle) : Prop :=
  angleSum α β = ⟨180⟩

-- Statement 1
theorem perpendicular_transitive_parallel (a b c : Line) :
  perpendicular a b → parallel b c → perpendicular a c := by sorry

-- Statement 2
theorem complementary_angle_sum (α β : Angle) :
  angleSum α β = ⟨180⟩ → complementary α β := by sorry

-- Statement 3
theorem parallel_transitive (a b c : Line) :
  parallel a b → parallel b c → parallel a c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_transitive_parallel_complementary_angle_sum_parallel_transitive_l844_84442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l844_84436

def A : Set ℝ := {x | x > 3}
def B : Set ℝ := {x | (x - 1) * (x - 4) < 0}

theorem intersection_of_A_and_B : A ∩ B = Set.Ioo 3 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l844_84436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_equation_m_range_l844_84483

theorem cos_equation_m_range :
  ∀ m : ℝ, (∃ x : ℝ, Real.cos (2 * x) - 2 * Real.cos x = m - 1) ↔ -1/2 ≤ m ∧ m ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_equation_m_range_l844_84483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joe_possible_lists_l844_84461

/-- Represents the set of balls numbered 1 to 12 -/
def BallSet : Finset Nat := Finset.range 12

/-- Represents the operation of adding 5 to a ball's number and wrapping around -/
def addFiveModTwelve (n : Nat) : Nat :=
  ((n + 4) % 12) + 1

/-- Represents a single draw from the ball set -/
def singleDraw : Finset Nat := BallSet.image addFiveModTwelve

/-- Represents the list of three numbers generated by Joe -/
def joesList : Finset (Nat × Nat × Nat) :=
  singleDraw.product singleDraw |>.product singleDraw |>.image (fun ((a, b), c) => (a, b, c))

theorem joe_possible_lists :
  Finset.card joesList = 12^3 := by
  sorry

#eval Finset.card joesList

end NUMINAMATH_CALUDE_ERRORFEEDBACK_joe_possible_lists_l844_84461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_swimmer_speed_calculation_l844_84475

/-- Represents the speed of a swimmer in still water and the speed of the stream -/
structure SwimmerSpeeds where
  swimmer : ℝ
  stream : ℝ

/-- Given downstream and upstream distances and time, calculates the swimmer's speed in still water -/
noncomputable def calculate_swimmer_speed (downstream_distance upstream_distance time : ℝ) : ℝ :=
  (downstream_distance / time + upstream_distance / time) / 2

theorem swimmer_speed_calculation (downstream_distance upstream_distance time : ℝ) 
  (h1 : downstream_distance = 16)
  (h2 : upstream_distance = 10)
  (h3 : time = 2) :
  calculate_swimmer_speed downstream_distance upstream_distance time = 6.5 := by
  sorry

#check swimmer_speed_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_swimmer_speed_calculation_l844_84475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_password_A_seventh_week_l844_84463

/-- Represents the probability of using password A in week n -/
def P (n : ℕ) : ℚ :=
  1/4 + 3/4 * (-1/3)^(n-1)

/-- The conditions of the password selection process -/
axiom password_conditions : 
  ∃ (A B C D : String),
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    ∀ (week : ℕ), week > 0 → 
      ∃ (used : String),
        (used = A ∨ used = B ∨ used = C ∨ used = D) ∧
        (∀ (prev_week : ℕ), prev_week = week - 1 →
          ∃ (prev_used : String),
            (prev_used = A ∨ prev_used = B ∨ prev_used = C ∨ prev_used = D) ∧
            prev_used ≠ used ∧
            (∀ (s : String), s ≠ prev_used → (1 : ℚ)/3 = (if used = s then 1 else 0)))

/-- Password A is used in the first week -/
axiom A_used_first_week : P 1 = 1

/-- The main theorem: probability of password A being used in the seventh week -/
theorem password_A_seventh_week : P 7 = 61/243 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_password_A_seventh_week_l844_84463
