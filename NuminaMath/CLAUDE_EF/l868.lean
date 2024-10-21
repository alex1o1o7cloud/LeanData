import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l868_86818

noncomputable def f (x : ℝ) : ℝ := (x + 1)^2 / Real.sqrt (x + 2)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x > -2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l868_86818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perpendicular_bisector_theorem_l868_86847

/-- Given a triangle ABC with perpendicular bisectors intersecting BC at M and N,
    prove that if MN = BC, then the angle BAC (α) is either 60° or 120°. -/
theorem triangle_perpendicular_bisector_theorem 
  (A B C M N : EuclideanSpace ℝ (Fin 2)) 
  (α : ℝ) 
  (hAB : ‖B - A‖ = 2 * c)
  (hAC : ‖C - A‖ = 2 * b)
  (hAM : ‖M - A‖ = b / Real.cos α)
  (hAN : ‖N - A‖ = c / Real.cos α)
  (hMN : ‖N - M‖ = ‖C - B‖)
  (b c : ℝ) :
  α = Real.pi / 3 ∨ α = 2 * Real.pi / 3 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perpendicular_bisector_theorem_l868_86847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_M_equals_two_min_value_theorem_neg_six_min_value_theorem_ten_thirds_two_solutions_theorem_l868_86834

noncomputable def f (a x : ℝ) : ℝ := Real.cos x ^ 2 + a * Real.sin x - a / 4 - 1 / 2

noncomputable def M (a : ℝ) : ℝ :=
  if a ≤ 0 then -a / 4 + 1 / 2
  else if a < 2 then a ^ 2 / 4 - a / 4 + 1 / 2
  else 3 * a / 4 - 1 / 2

theorem max_value_theorem (a : ℝ) :
  ∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 → f a x ≤ M a := by
  sorry

theorem M_equals_two (a : ℝ) :
  M a = 2 ↔ a = -6 ∨ a = 10 / 3 := by
  sorry

theorem min_value_theorem_neg_six :
  ∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 → f (-6) x ≥ -5 := by
  sorry

theorem min_value_theorem_ten_thirds :
  ∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 → f (10 / 3) x ≥ -1 / 3 := by
  sorry

theorem two_solutions_theorem (a : ℝ) :
  (∃ x₁ x₂, 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ < 2 * Real.pi ∧
    f a x₁ = (1 + a) * Real.sin x₁ ∧
    f a x₂ = (1 + a) * Real.sin x₂) ↔
  (a > -6 ∧ a < 2) ∨ a = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_M_equals_two_min_value_theorem_neg_six_min_value_theorem_ten_thirds_two_solutions_theorem_l868_86834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_f_specific_value_cos_value_when_f_equals_2root6_div_5_l868_86824

/-- An angle in the third quadrant -/
noncomputable def α : ℝ := sorry

/-- Definition of the function f -/
noncomputable def f (x : ℝ) : ℝ := (Real.sin (3 * Real.pi / 2 - x) * Real.cos (Real.pi / 2 - x) * Real.tan (-x + Real.pi)) / (Real.sin (Real.pi / 2 + x) * Real.tan (2 * Real.pi - x))

theorem f_simplification :
  f α = -Real.sin α := by sorry

theorem f_specific_value :
  f (-32 * Real.pi / 3) = Real.sqrt 3 / 2 := by sorry

theorem cos_value_when_f_equals_2root6_div_5 :
  f α = 2 * Real.sqrt 6 / 5 →
  Real.cos (Real.pi + α) = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_f_specific_value_cos_value_when_f_equals_2root6_div_5_l868_86824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_unique_x_for_G_l868_86843

-- Define the function G as noncomputable
noncomputable def G (a b c d : ℝ) : ℝ := a^b + c / d

-- State the theorem
theorem exists_unique_x_for_G :
  ∃! x : ℝ, G 3 x 48 8 = 310 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_unique_x_for_G_l868_86843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_students_wearing_other_colors_l868_86825

theorem students_wearing_other_colors 
  (total_students : ℕ) 
  (blue_percent : ℚ) 
  (red_percent : ℚ) 
  (green_percent : ℚ) 
  (h1 : total_students = 900)
  (h2 : blue_percent = 44/100)
  (h3 : red_percent = 28/100)
  (h4 : green_percent = 10/100) :
  (total_students : ℚ) * (1 - (blue_percent + red_percent + green_percent)) = 162 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_students_wearing_other_colors_l868_86825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_sphere_radius_l868_86826

-- Define the Sphere structure
structure Sphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ

-- Define helper functions
def spheres_touch_each_other (s1 s2 s3 s4 : Sphere) : Prop := sorry
def spheres_touch_base (s1 s2 s3 s4 : Sphere) : Prop := sorry
def spheres_touch_lateral_surface (s1 s2 s3 s4 : Sphere) : Prop := sorry
def sphere_touches_lateral_surface (s : Sphere) : Prop := sorry
def sphere_touches_all (s1 s2 s3 s4 s5 : Sphere) : Prop := sorry

theorem fifth_sphere_radius (h : ℝ) (r : ℝ) (x : ℝ) :
  h = 7 ∧ r = 7 ∧ 
  (∃ (s₁ s₂ s₃ s₄ s₅ : Sphere), 
    s₁.radius = s₂.radius ∧ s₁.radius = s₃.radius ∧ s₁.radius = s₄.radius ∧
    s₅.radius = x ∧
    spheres_touch_each_other s₁ s₂ s₃ s₄ ∧
    spheres_touch_base s₁ s₂ s₃ s₄ ∧
    spheres_touch_lateral_surface s₁ s₂ s₃ s₄ ∧
    sphere_touches_lateral_surface s₅ ∧
    sphere_touches_all s₅ s₁ s₂ s₃ s₄) →
  x = 2 * Real.sqrt 2 - 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_sphere_radius_l868_86826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_electric_bicycle_inspection_probability_l868_86892

theorem electric_bicycle_inspection_probability :
  let num_neighborhoods : ℕ := 3
  let num_staff : ℕ := 4
  let total_assignments : ℕ := num_neighborhoods ^ num_staff
  let favorable_assignments : ℕ := 
    num_neighborhoods * (Nat.choose num_staff 2 * 1 + num_staff * 1) * 2
  (favorable_assignments : ℚ) / total_assignments = 14 / 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_electric_bicycle_inspection_probability_l868_86892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l868_86854

noncomputable def f (x : ℝ) := Real.sin (x + Real.pi/4) * Real.cos (x + Real.pi/4) + Real.cos x ^ 2

theorem f_properties :
  ∃ (T : ℝ),
    (∀ x, f (x + T) = f x) ∧
    (∀ k : ℤ, StrictMonoOn f (Set.Icc (k * Real.pi) (k * Real.pi + Real.pi/2))) ∧
    (∀ (a b c : ℝ) (A B C : ℝ),
      0 < a ∧ 0 < b ∧ 0 < c ∧
      0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi ∧
      A + B + C = Real.pi ∧
      a * Real.sin B = b * Real.sin A ∧
      b * Real.sin C = c * Real.sin B ∧
      f (A/2) = 1 ∧
      a = 2 →
      (1/2 * b * c * Real.sin A) ≤ Real.sqrt 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l868_86854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_percentage_change_l868_86842

/-- Given positive real numbers x and y that are inversely proportional,
    prove that when x increases by p%, y decreases by (100p)/(100+p)% -/
theorem inverse_proportion_percentage_change 
  (x y p : ℝ) 
  (h_pos_x : x > 0) 
  (h_pos_y : y > 0) 
  (h_inverse_prop : ∃ k, x * y = k ∧ k > 0) 
  (h_p_pos : p > 0) : 
  (y - y * (100 / (100 + p))) / y = p / (100 + p) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_percentage_change_l868_86842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_photo_distribution_equation_l868_86897

/-- Represents the number of students in the class -/
def x : ℕ := sorry

/-- The total number of photos given out -/
def total_photos : ℕ := 2550

/-- Theorem stating that the equation x(x-1) = 2550 correctly represents the photo distribution -/
theorem photo_distribution_equation : x * (x - 1) = total_photos := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_photo_distribution_equation_l868_86897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_perimeter_achieves_lower_bound_l868_86806

-- Define the space we're working in
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [CompleteSpace V]

-- Define the given points and line
variable (A B : V) (l : Set V)

-- Define the function to calculate the perimeter of a triangle
def triangle_perimeter (C : V) : ℝ := ‖A - B‖ + ‖B - C‖ + ‖C - A‖

-- State the theorem
theorem smallest_perimeter (A B : V) (l : Set V) :
  ∀ C ∈ l, triangle_perimeter A B C ≥ 2 * ‖A - B‖ :=
by
  sorry

-- Define the existence of a point that achieves the lower bound
theorem achieves_lower_bound (A B : V) (l : Set V) :
  ∃ C ∈ l, triangle_perimeter A B C = 2 * ‖A - B‖ :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_perimeter_achieves_lower_bound_l868_86806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l868_86813

noncomputable def f (x : ℝ) := (Real.sqrt (4 - x^2)) / x

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ∈ Set.Icc (-2) 0 ∪ Set.Ioc 0 2} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l868_86813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_distance_l868_86885

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- Check if a point is on the hyperbola -/
def isOnHyperbola (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 / (6 : ℝ) - p.y^2 / (3 : ℝ) = 1

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: For the given hyperbola, if a line through the left focus intersects
    the left branch at A and B, and |AF₂| + |BF₂| = 2|AB|, then |AB| = 4√6 -/
theorem hyperbola_intersection_distance
  (h : Hyperbola)
  (F₁ F₂ A B : Point)
  (h_eq : h.a^2 = 6 ∧ h.b^2 = 3)
  (h_on_A : isOnHyperbola h A)
  (h_on_B : isOnHyperbola h B)
  (h_left_branch : A.x < 0 ∧ B.x < 0)
  (h_line : ∃ (t : ℝ), A = Point.mk (F₁.x + t * (B.x - F₁.x)) (F₁.y + t * (B.y - F₁.y)))
  (h_sum : distance A F₂ + distance B F₂ = 2 * distance A B) :
  distance A B = 4 * Real.sqrt 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_distance_l868_86885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_false_negation_equivalence_non_parallel_no_intersection_l868_86875

-- Proposition ①
theorem contrapositive_false : 
  ¬(∀ m : ℝ, ∀ a b : ℝ, a^2*m ≤ b^2*m → a ≤ b) := by sorry

-- Proposition ②
theorem negation_equivalence :
  (¬∃ α β : ℝ, Real.tan (α + β) = Real.tan α + Real.tan β) ↔
  (∀ α β : ℝ, Real.tan (α + β) ≠ Real.tan α + Real.tan β) := by sorry

-- Definitions used in the statement
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

structure Line where
  direction : Point
  point : Point

def Line.parallel (a b : Line) : Prop :=
  ∃ k : ℝ, a.direction = Point.mk (k * b.direction.x) (k * b.direction.y) (k * b.direction.z)

def Point.in_line (p : Point) (l : Line) : Prop :=
  ∃ t : ℝ, p = Point.mk 
    (l.point.x + t * l.direction.x)
    (l.point.y + t * l.direction.y)
    (l.point.z + t * l.direction.z)

-- Proposition ③
theorem non_parallel_no_intersection :
  ∃ (a b : Line), ¬(Line.parallel a b) ∧ (∀ p : Point, ¬(Point.in_line p a ∧ Point.in_line p b)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_false_negation_equivalence_non_parallel_no_intersection_l868_86875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_factorization_l868_86891

theorem quadratic_factorization (a b c : ℤ) : 
  (∀ x : ℝ, x^2 + 19*x + 88 = (x + a)*(x + b)) →
  (∀ x : ℝ, x^2 - 23*x + 132 = (x - b)*(x - c)) →
  a + b + c = 31 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_factorization_l868_86891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_value_l868_86884

-- Define the geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the theorem
theorem angle_value (a : ℕ → ℝ) (α : ℝ) :
  geometric_sequence a →
  (a 1)^2 - 2*(a 1)*Real.sin α - Real.sqrt 3*Real.sin α = 0 →
  (a 8)^2 - 2*(a 8)*Real.sin α - Real.sqrt 3*Real.sin α = 0 →
  (a 1 + a 8)^2 = 2*(a 3)*(a 6) + 6 →
  0 < α →
  α < Real.pi/2 →
  α = Real.pi/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_value_l868_86884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_symmetry_l868_86881

def is_symmetric_axis (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

theorem sine_symmetry (ω φ : ℝ) (h_ω : ω > 0) :
  is_symmetric_axis (λ x ↦ Real.sin (ω * x + φ)) (π / 3) ∧
  is_symmetric_axis (λ x ↦ Real.sin (ω * x + φ)) (5 * π / 6) →
  φ = -π / 6 ∨ ∃ k : ℤ, φ = k * π - π / 6 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_symmetry_l868_86881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_givenCurve_is_limacon_l868_86846

/-- A polar curve is a limaçon if it can be expressed in the form r = a + b cos(θ) where a and b are constants and |a| < |b|. -/
def IsLimacon (r : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), (∀ θ, r θ = a + b * Real.cos θ) ∧ |a| < |b|

/-- The given polar equation r = 1 + 2cos(θ) -/
noncomputable def givenCurve (θ : ℝ) : ℝ := 1 + 2 * Real.cos θ

/-- Theorem: The curve defined by r = 1 + 2cos(θ) is a limaçon -/
theorem givenCurve_is_limacon : IsLimacon givenCurve := by
  use 1, 2
  constructor
  · intro θ
    rfl
  · norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_givenCurve_is_limacon_l868_86846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_cost_in_inr_l868_86863

/-- Exchange rate from Euro to Egyptian pounds -/
noncomputable def euro_to_egp : ℚ := 9

/-- Exchange rate from Euro to Indian Rupees -/
noncomputable def euro_to_inr : ℚ := 90

/-- Cost of the book in Egyptian pounds -/
noncomputable def book_cost_egp : ℚ := 540

/-- Function to convert Egyptian pounds to Indian Rupees -/
noncomputable def egp_to_inr (amount : ℚ) : ℚ :=
  amount * (euro_to_inr / euro_to_egp)

theorem book_cost_in_inr :
  egp_to_inr book_cost_egp = 5400 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_cost_in_inr_l868_86863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_parallel_implies_a_equals_three_l868_86808

-- Define the curve
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - Real.log (x + 1)

-- Define the tangent line at (0,0)
def tangent_slope (a : ℝ) : ℝ := a - 1

-- Define the line 2x-y-6=0
def line_slope : ℝ := 2

-- Theorem statement
theorem tangent_parallel_implies_a_equals_three (a : ℝ) :
  tangent_slope a = line_slope → a = 3 := by
  intro h
  have : a - 1 = 2 := h
  linarith


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_parallel_implies_a_equals_three_l868_86808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_increasing_fee_part2_no_loss_l868_86861

-- Define the processing fee function
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1/2) * Real.log (2*x + 1) - m*x

-- Part 1: Prove that f is increasing for x ∈ (0,99.5] when m = 1/200
theorem part1_increasing_fee (x : ℝ) (hx : x ∈ Set.Ioc 0 99.5) :
  Monotone (fun x => f (1/200) x) := by
  sorry

-- Part 2: Prove the range of m for which the enterprise doesn't incur losses
theorem part2_no_loss (x : ℝ) (hx : x ∈ Set.Icc 10 20) (m : ℝ) (hm : m ∈ Set.Ioc 0 ((Real.log 41 - 2)/40)) :
  f m x ≥ (1/20) * x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_increasing_fee_part2_no_loss_l868_86861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subgroup_order_divides_group_order_l868_86867

/-- Given a finite cyclic group of order n and its subgroup A, 
    if A is closed under addition modulo n, 
    then the order of A divides n. -/
theorem subgroup_order_divides_group_order (n : ℕ) (A : Finset ℕ) : 
  (∀ x ∈ A, x ≤ n) →
  (∀ x y, x ∈ A → y ∈ A → (x + y) % n ∈ A) →
  ∃ m : ℕ, m * A.card = n :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_subgroup_order_divides_group_order_l868_86867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l868_86862

noncomputable def a : ℝ × ℝ := (1, Real.sqrt 3)
def b : ℝ × ℝ := (-2, 0)

theorem vector_properties : 
  let norm_diff := Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)
  let angle := Real.arccos ((a.1 * (a.1 - b.1) + a.2 * (a.2 - b.2)) / 
    (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)))
  let norm_diff_t (t : ℝ) := Real.sqrt ((a.1 - t*b.1)^2 + (a.2 - t*b.2)^2)
  ∃ (t : ℝ),
    norm_diff = 2 * Real.sqrt 3 ∧ 
    angle = π / 6 ∧
    (∀ s, norm_diff_t s ≥ Real.sqrt 3) ∧
    norm_diff_t t = Real.sqrt 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l868_86862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_log_function_is_exp_minus_one_l868_86823

-- Define the logarithm function
noncomputable def log_function (x : ℝ) : ℝ := Real.log (x + 1)

-- Define the property of 90-degree counterclockwise rotation around the origin
def rotated_90_ccw (f g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x = y ↔ g (-y) = x

-- State the theorem
theorem rotated_log_function_is_exp_minus_one :
  ∃ f : ℝ → ℝ, rotated_90_ccw f log_function ∧ (∀ x : ℝ, f x = Real.exp (-Real.log 10 * x) - 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_log_function_is_exp_minus_one_l868_86823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_line_intercept_l868_86872

/-- The curve family parameterized by m -/
def curve_family (m : ℝ) (x y : ℝ) : Prop :=
  4 * x^2 + 5 * y^2 - 8 * m * x - 20 * m * y + 24 * m^2 - 20 = 0

/-- The line y = 2x + b -/
def line (b : ℝ) (x y : ℝ) : Prop :=
  y = 2 * x + b

/-- The length of the segment intercepted by the curve on the line -/
noncomputable def intercept_length : ℝ :=
  5/3 * Real.sqrt 5

theorem curve_line_intercept :
  ∃! b : ℝ, 
    (∀ m x y : ℝ, curve_family m x y → line b x y → 
      ∃ x₁ y₁ x₂ y₂ : ℝ, 
        curve_family m x₁ y₁ ∧ curve_family m x₂ y₂ ∧
        line b x₁ y₁ ∧ line b x₂ y₂ ∧
        Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = intercept_length) ∧
    (b = 2 ∨ b = -2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_line_intercept_l868_86872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l868_86887

noncomputable section

open Real

theorem trig_identity (θ : ℝ) : 
  (sin θ + (1 / sin θ))^3 + (cos θ + (1 / cos θ))^3 - sin θ^2 - cos θ^2 = 
  10 + 3 * (sin θ + cos θ + (1 / sin θ) + (1 / cos θ) + (cos θ / sin θ) + (sin θ / cos θ) + (cos θ / sin θ)^3 + (sin θ / cos θ)^3) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l868_86887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_theorem_l868_86814

theorem inequality_theorem {n : ℕ+} (α : ℝ) (x : Fin n → ℝ) 
  (h_α : α ≤ 1) 
  (h_x_dec : ∀ i j, i < j → x i ≥ x j) 
  (h_x_pos : ∀ i, x i > 0) 
  (h_x_le_one : x 0 ≤ 1) : 
  (1 + (Finset.univ.sum x))^α ≤ 1 + (Finset.univ.sum (λ i => ((Fin.val i + 1 : ℕ)^(α - 1)) * (x i)^α)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_theorem_l868_86814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_sequence_condition_l868_86858

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (2*a - 1)*x + 4 else a^x

-- Define the sequence a_n
noncomputable def a_n (a : ℝ) (n : ℕ) : ℝ := f a n

-- State the theorem
theorem increasing_sequence_condition (a : ℝ) :
  (∀ n : ℕ, a_n a n < a_n a (n + 1)) ↔ a > 3 := by
  sorry

#check increasing_sequence_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_sequence_condition_l868_86858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_scaling_l868_86871

open MeasureTheory

variable (g : ℝ → ℝ)
variable (a b : ℝ)

noncomputable def area_under_curve (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
  ∫ x in a..b, |f x|

theorem area_scaling (h : area_under_curve g a b = 15) :
  area_under_curve (fun x ↦ 4 * g (2 * (x - 1))) a b = 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_scaling_l868_86871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_necklace_beads_count_l868_86829

/-- Represents the number of beads in a necklace -/
structure Necklace where
  purple : ℕ
  blue : ℕ
  green : ℕ

/-- The total number of beads in the necklace -/
def total_beads (n : Necklace) : ℕ :=
  n.purple + n.blue + n.green

/-- Theorem stating the total number of beads in the necklace is 46 -/
theorem necklace_beads_count :
  ∀ (n : Necklace),
  n.purple = 7 →
  n.blue = 2 * n.purple →
  n.green = n.blue + 11 →
  total_beads n = 46 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_necklace_beads_count_l868_86829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_values_l868_86816

noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ 0 then -x else x^2

theorem alpha_values (α : ℝ) (h : f α = 9) : α = -9 ∨ α = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_values_l868_86816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_millet_exceeds_half_on_third_day_l868_86873

/-- Represents the state of the bird feeder on a given day -/
structure BirdFeeder where
  day : ℕ
  milletProportion : ℚ
  totalSeeds : ℚ

/-- Calculates the next day's bird feeder state -/
def nextDay (bf : BirdFeeder) : BirdFeeder :=
  let newMilletProportion := min (bf.milletProportion + 1/20) (1/2)
  let remainingMillet := bf.totalSeeds * bf.milletProportion * (7/10)
  let addedMillet := newMilletProportion
  { day := bf.day + 1,
    milletProportion := (remainingMillet + addedMillet) / (remainingMillet + 1),
    totalSeeds := remainingMillet + 1 }

/-- The initial state of the bird feeder -/
def initialFeeder : BirdFeeder :=
  { day := 1, milletProportion := 1/4, totalSeeds := 1 }

/-- Theorem stating that on the third day, the millet proportion exceeds 1/2 -/
theorem millet_exceeds_half_on_third_day :
  (nextDay (nextDay initialFeeder)).milletProportion > 1/2 := by
  sorry

#eval (nextDay (nextDay initialFeeder)).milletProportion

end NUMINAMATH_CALUDE_ERRORFEEDBACK_millet_exceeds_half_on_third_day_l868_86873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_sin_is_valid_extension_l868_86898

noncomputable def extended_sin (x : ℝ) : ℝ :=
  let k := ⌊(2 * x) / Real.pi⌋
  Real.sin (x - k * Real.pi / 2)

theorem extended_sin_is_valid_extension :
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), extended_sin x = Real.sin x) ∧
  (∀ x : ℝ, extended_sin (x + Real.pi / 2) = extended_sin x) ∧
  (∀ x : ℝ, ∃ k : ℤ, x ∈ Set.Ico (k * Real.pi / 2) ((k + 1) * Real.pi / 2) ∧
    extended_sin x = Real.sin (x - k * Real.pi / 2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_sin_is_valid_extension_l868_86898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_cost_time_plan1_more_cost_effective_l868_86828

/-- Represents a phone plan with monthly fee, call time limit, and overtime fee -/
structure PhonePlan where
  monthlyFee : ℚ
  callTimeLimit : ℚ
  overtimeFee : ℚ

/-- Calculates the cost of a phone plan given the call time -/
def planCost (plan : PhonePlan) (callTime : ℚ) : ℚ :=
  plan.monthlyFee + max 0 (callTime - plan.callTimeLimit) * plan.overtimeFee

/-- The two phone plans provided by the operator -/
def plan1 : PhonePlan := ⟨20, 180, 1/10⟩
def plan2 : PhonePlan := ⟨40, 480, 1/5⟩

theorem equal_cost_time :
  ∃ t : ℚ, t ≤ 480 ∧ planCost plan1 t = planCost plan2 t ∧ t = 380 := by
  sorry

theorem plan1_more_cost_effective :
  ∀ t : ℚ, t > 580 → planCost plan1 t < planCost plan2 t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_cost_time_plan1_more_cost_effective_l868_86828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_negative_three_l868_86874

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 else Real.log (x + 1)

-- State the theorem
theorem f_composition_negative_three : f (f (-3)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_negative_three_l868_86874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_condition_increasing_condition_l868_86835

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * Real.log (1 - x)

-- Theorem 1: If f(x) has an extremum at x = -1, then a = -1/2
theorem extremum_condition (a : ℝ) :
  (∃ (ε : ℝ), ε > 0 ∧ ∀ x ∈ Set.Ioo (-1 - ε) (-1 + ε), f a (-1) ≤ f a x) →
  a = -1/2 := by
sorry

-- Theorem 2: If f(x) is increasing on [-3, -2], then a ≤ -1/6
theorem increasing_condition (a : ℝ) :
  (∀ x y, x ∈ Set.Icc (-3) (-2) → y ∈ Set.Icc (-3) (-2) → x < y → f a x < f a y) →
  a ≤ -1/6 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_condition_increasing_condition_l868_86835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_decreasing_range_l868_86893

-- Define the exponential function as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2 - a) ^ x

-- State the theorem
theorem exponential_decreasing_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ > f a x₂) ↔ a ∈ Set.Ioo 1 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_decreasing_range_l868_86893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_table_seating_arrangement_l868_86877

def number_of_arrangements (n : ℕ) (adjacent_pair : ℕ) : ℕ :=
  2 * Nat.factorial (n - 2)

theorem round_table_seating_arrangement :
  number_of_arrangements 6 1 = 48 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_table_seating_arrangement_l868_86877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_is_hyperbola_with_foci_on_y_axis_l868_86811

-- Define the angle q in the third quadrant
noncomputable def q : Real := sorry

-- Define the condition for q being in the third quadrant
axiom q_in_third_quadrant : Real.pi < q ∧ q < 3 * Real.pi / 2

-- Define the equation of the curve
def curve_equation (x y : Real) : Prop :=
  x^2 + y^2 * Real.sin q = Real.cos q

-- State the theorem
theorem curve_is_hyperbola_with_foci_on_y_axis :
  ∃ (a b : Real), a > 0 ∧ b > 0 ∧
  ∀ (x y : Real), curve_equation x y ↔ 
    (y/a)^2 - (x/b)^2 = 1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_is_hyperbola_with_foci_on_y_axis_l868_86811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_living_room_flooring_cost_l868_86845

/-- Calculates the cost of flooring a rectangular room with a triangular decorative portion excluded -/
noncomputable def flooringCost (roomLength roomWidth triangleBase triangleHeight costPerSqMeter : ℝ) : ℝ :=
  let roomArea := roomLength * roomWidth
  let triangleArea := (triangleBase * triangleHeight) / 2
  let flooringArea := roomArea - triangleArea
  flooringArea * costPerSqMeter

/-- Theorem stating the cost of flooring the living room -/
theorem living_room_flooring_cost :
  flooringCost 10 7 4 3 900 = 57600 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_living_room_flooring_cost_l868_86845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_billboard_diagonal_l868_86896

/-- A rectangular billboard with given area and perimeter -/
structure Billboard where
  length : ℝ
  width : ℝ
  area_eq : length * width = 117
  perimeter_eq : 2 * (length + width) = 44

/-- The length of the diagonal of a billboard -/
noncomputable def diagonal (b : Billboard) : ℝ :=
  Real.sqrt (b.length ^ 2 + b.width ^ 2)

/-- Theorem stating that the diagonal of the billboard is 5√10 -/
theorem billboard_diagonal :
  ∀ b : Billboard, diagonal b = 5 * Real.sqrt 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_billboard_diagonal_l868_86896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slant_asymptote_and_coefficient_sum_l868_86804

/-- The rational function -/
noncomputable def f (x : ℝ) : ℝ := (5 * x^2 - 4 * x + 9) / (x - 5)

/-- The slant asymptote function -/
def g (x : ℝ) : ℝ := 5 * x + 21

/-- Theorem stating the slant asymptote and the sum of its coefficients -/
theorem slant_asymptote_and_coefficient_sum :
  (∀ ε > 0, ∃ N : ℝ, ∀ x > N, |f x - g x| < ε) ∧
  (let m := 5; let b := 21; m + b = 26) := by
  sorry

#check slant_asymptote_and_coefficient_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slant_asymptote_and_coefficient_sum_l868_86804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radius_C_polar_equation_C_l868_86844

-- Define the circle C
def circle_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (ρ, θ) := p; ρ > 0 ∧ ∃ (r : ℝ), r > 0 ∧ (ρ * Real.cos θ - 1)^2 + (ρ * Real.sin θ)^2 = r^2}

-- Define the point P
noncomputable def point_P : ℝ × ℝ := (Real.sqrt 3, Real.pi / 6)

-- State that P is on circle C
axiom P_on_C : point_P ∈ circle_C

-- State that the center of C is at (1, 0)
axiom center_C : (1, 0) ∈ circle_C

-- Theorem for the radius of circle C
theorem radius_C : ∃ (r : ℝ), r = 1 ∧ ∀ (p : ℝ × ℝ), p ∈ circle_C →
  let (ρ, θ) := p
  (ρ * Real.cos θ - 1)^2 + (ρ * Real.sin θ)^2 = r^2 := by
  sorry

-- Theorem for the polar equation of circle C
theorem polar_equation_C : ∀ (p : ℝ × ℝ), p ∈ circle_C ↔
  let (ρ, θ) := p
  ρ = 2 * Real.cos θ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_radius_C_polar_equation_C_l868_86844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_through_tunnel_l868_86878

/-- Calculates the speed of a train passing through a tunnel. -/
theorem train_speed_through_tunnel 
  (train_length : Real) 
  (tunnel_length : Real) 
  (time_minutes : Real) : Real := by
  have h1 : train_length = 0.1 := by sorry  -- 100 meters in km
  have h2 : tunnel_length = 3.5 := by sorry -- in km
  have h3 : time_minutes = 3 := by sorry    -- in minutes
  have h4 : (tunnel_length + train_length) / (time_minutes / 60) = 72 := by sorry
  exact 72

#check train_speed_through_tunnel

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_through_tunnel_l868_86878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_license_plate_sampling_l868_86850

/-- A function that represents systematic sampling of license plates -/
def systematicSample (n : ℕ) (k : ℕ) : Prop :=
  ∀ x : ℕ, x % n = k → x ∈ (Set.range (λ i ↦ i * n + k))

/-- Theorem stating that selecting license plates ending in 5 is a form of systematic sampling -/
theorem license_plate_sampling :
  systematicSample 10 5 := by
  sorry

#check license_plate_sampling

end NUMINAMATH_CALUDE_ERRORFEEDBACK_license_plate_sampling_l868_86850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_changes_theorem_l868_86809

noncomputable def calculate_final_price (initial_price : ℝ) (increase : ℝ) (discount1 : ℝ) (discount2 : ℝ) (tax : ℝ) : ℝ :=
  let price_after_increase := initial_price * (1 + increase)
  let price_after_discount1 := price_after_increase * (1 - discount1)
  let price_after_discount2 := price_after_discount1 * (1 - discount2)
  price_after_discount2 * (1 + tax)

noncomputable def calculate_percentage_gain (initial_price : ℝ) (final_price : ℝ) : ℝ :=
  (final_price - initial_price) / initial_price * 100

theorem price_changes_theorem (initial_price : ℝ) 
  (increase_A : ℝ) (discount1_A : ℝ) (discount2_A : ℝ)
  (increase_B : ℝ) (discount1_B : ℝ) (discount2_B : ℝ)
  (increase_C : ℝ) (discount1_C : ℝ) (discount2_C : ℝ)
  (tax : ℝ) :
  initial_price > 0 →
  increase_A = 0.315 →
  discount1_A = 0.125 →
  discount2_A = 0.173 →
  increase_B = 0.278 →
  discount1_B = 0.096 →
  discount2_B = 0.147 →
  increase_C = 0.331 →
  discount1_C = 0.117 →
  discount2_C = 0.139 →
  tax = 0.075 →
  let final_price_A := calculate_final_price initial_price increase_A discount1_A discount2_A tax
  let final_price_B := calculate_final_price initial_price increase_B discount1_B discount2_B tax
  let final_price_C := calculate_final_price initial_price increase_C discount1_C discount2_C tax
  let gain_A := calculate_percentage_gain initial_price final_price_A
  let gain_B := calculate_percentage_gain initial_price final_price_B
  let gain_C := calculate_percentage_gain initial_price final_price_C
  (abs (gain_A - 2.2821) < 0.0001) ∧ (abs (gain_B - 5.939) < 0.0001) ∧ (abs (gain_C - 8.7696) < 0.0001) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_changes_theorem_l868_86809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l868_86801

-- Define the hyperbola
def hyperbola (x y a b : ℝ) : Prop := y^2 / a^2 - x^2 / b^2 = 1

-- Define the vector conditions
def vector_conditions (F₁ F₂ O M P : ℝ × ℝ) (l : ℝ) : Prop :=
  F₂.1 - O.1 = M.1 - P.1 ∧ 
  F₂.2 - O.2 = M.2 - P.2 ∧
  F₁.1 - M.1 = l * ((F₁.1 - P.1) / Real.sqrt ((F₁.1 - P.1)^2 + (F₁.2 - P.2)^2) + 
                    (F₁.1 - O.1) / Real.sqrt ((F₁.1 - O.1)^2 + (F₁.2 - O.2)^2)) ∧
  F₁.2 - M.2 = l * ((F₁.2 - P.2) / Real.sqrt ((F₁.1 - P.1)^2 + (F₁.2 - P.2)^2) + 
                    (F₁.2 - O.2) / Real.sqrt ((F₁.1 - O.1)^2 + (F₁.2 - O.2)^2)) ∧
  l > 0

-- Define the theorem
theorem hyperbola_properties 
  (a b : ℝ) 
  (F₁ F₂ O M P : ℝ × ℝ) 
  (l : ℝ) 
  (h1 : ∀ x y, hyperbola x y a b)
  (h2 : vector_conditions F₁ F₂ O M P l)
  (h3 : hyperbola (Real.sqrt 3) 2 a b) :
  (∃ e : ℝ, e = 2 ∧ 
   (∀ x y, y^2 / 3 - x^2 / 9 = 1 ↔ hyperbola x y a b) ∧
   (∃ k : ℝ, k = 1 ∨ k = -1) ∧ ∀ x y, x = k * y + 3 → hyperbola x y a b) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l868_86801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_and_median_l868_86860

/-- A right triangle with given side lengths -/
structure RightTriangle where
  ab : ℝ
  bc : ℝ
  bac_is_right : ab > 0 ∧ bc > 0

/-- The length of the hypotenuse in a right triangle -/
noncomputable def hypotenuse (t : RightTriangle) : ℝ :=
  Real.sqrt (t.ab ^ 2 + t.bc ^ 2)

/-- The length of the median from the right angle to the hypotenuse in a right triangle -/
noncomputable def median_to_hypotenuse (t : RightTriangle) : ℝ :=
  Real.sqrt ((t.ab ^ 2 + t.bc ^ 2) / 4 + (t.ab * t.bc) ^ 2 / (4 * (t.ab ^ 2 + t.bc ^ 2)))

theorem right_triangle_hypotenuse_and_median
  (t : RightTriangle)
  (h_ab : t.ab = 6)
  (h_bc : t.bc = 10) :
  hypotenuse t = Real.sqrt 136 ∧ median_to_hypotenuse t = Real.sqrt 61 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_and_median_l868_86860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_l868_86840

-- Define the function f(x) = x^(1/2)
noncomputable def f (x : ℝ) : ℝ := Real.sqrt x

-- State the theorem
theorem f_nonnegative : ∀ x : ℝ, x > 0 → f x ≥ 0 := by
  intro x hx
  unfold f
  apply Real.sqrt_nonneg


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_l868_86840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_inequality_implies_expression_equality_l868_86859

theorem sqrt_inequality_implies_expression_equality (x : ℝ) :
  Real.sqrt 2 * x > Real.sqrt 3 * x + 1 →
  (x + 2) ^ (1/3) - abs (x + 3) = 2 * x + 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_inequality_implies_expression_equality_l868_86859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_odd_even_l868_86836

/-- A partition of positive integers into two sets -/
structure IntegerPartition :=
  (R : Set ℕ)
  (B : Set ℕ)
  (partition : R ∪ B = Set.univ ∧ R ∩ B = ∅)
  (R_infinite : Set.Infinite R)
  (B_infinite : Set.Infinite B)
  (R_closed : ∀ (S : Finset ℕ), S.card = 2023 → (∀ n ∈ S, n ∈ R) → (S.sum id) ∈ R)
  (B_closed : ∀ (S : Finset ℕ), S.card = 2023 → (∀ n ∈ S, n ∈ B) → (S.sum id) ∈ B)

/-- The main theorem -/
theorem partition_odd_even (p : IntegerPartition) :
  (∀ n : ℕ, n > 0 → (Odd n ↔ n ∈ p.R)) ∧ (∀ n : ℕ, n > 0 → (Even n ↔ n ∈ p.B)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_odd_even_l868_86836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_edge_length_approx_220_l868_86888

-- Define the parameters of the pyramids
noncomputable def hexagon_side : ℝ := 8
noncomputable def pentagon_side : ℝ := 5
noncomputable def first_pyramid_height : ℝ := 15
noncomputable def second_pyramid_height : ℝ := 10

-- Define the compound structure
structure CompoundPyramids where
  hexagon_base : ℝ
  pentagon_base : ℝ
  first_height : ℝ
  second_height : ℝ

-- Define the function to calculate the total edge length
noncomputable def total_edge_length (p : CompoundPyramids) : ℝ :=
  let hexagon_inradius := p.hexagon_base / 2 / Real.tan (Real.pi / 6)
  let pentagon_inradius := p.pentagon_base / 2 / Real.tan (Real.pi / 5)
  let hexagon_slant := Real.sqrt (p.first_height ^ 2 + hexagon_inradius ^ 2)
  let pentagon_slant := Real.sqrt (p.second_height ^ 2 + pentagon_inradius ^ 2)
  6 * p.hexagon_base + 6 * hexagon_slant + 5 * p.pentagon_base + 5 * pentagon_slant

-- Theorem statement
theorem total_edge_length_approx_220 :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧
  abs (total_edge_length (CompoundPyramids.mk hexagon_side pentagon_side first_pyramid_height second_pyramid_height) - 220) < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_edge_length_approx_220_l868_86888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_properties_l868_86856

-- Define the quadratic equation and its properties
def quadratic_equation (a x : ℝ) : Prop := a * x^2 + x + 1 = 0

-- Define the existence of two real roots
def has_two_real_roots (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation a x₁ ∧ quadratic_equation a x₂

-- Theorem statement
theorem quadratic_properties (a : ℝ) (h_a : a > 0) (h_roots : has_two_real_roots a) :
  let x₁ := Classical.choose h_roots
  let x₂ := Classical.choose (Classical.choose_spec h_roots)
  (1 + x₁) * (1 + x₂) = 1 ∧
  x₁ < -1 ∧ x₂ < -1 ∧
  (∀ r ∈ Set.Icc (1/10 : ℝ) 10, x₁ / x₂ = r → a ≤ 1/4) ∧
  (∃ r ∈ Set.Icc (1/10 : ℝ) 10, x₁ / x₂ = r ∧ a = 1/4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_properties_l868_86856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_translation_is_cos_plus_two_l868_86865

noncomputable section

def f (x : ℝ) : ℝ := Real.sin x

def translation (p : ℝ × ℝ) : ℝ × ℝ := (p.1 - Real.pi / 2, p.2 + 2)

def g (x : ℝ) : ℝ := Real.cos x + 2

theorem sin_translation_is_cos_plus_two :
  ∀ x : ℝ, g x = (translation (x, f x)).2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_translation_is_cos_plus_two_l868_86865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_charity_distribution_result_l868_86820

/-- Represents the charity organization's box distribution problem -/
def charity_distribution (initial_boxes : ℕ) (food_cost : ℚ) (supplies_cost : ℚ) 
  (shipping_cost : ℚ) (discount_rate : ℚ) (discount_threshold : ℕ) 
  (donation_multiplier : ℕ) (exchange_rate : ℚ) : ℕ :=
  let box_cost := food_cost + supplies_cost + shipping_cost
  let initial_cost := initial_boxes * box_cost
  let discounted_supplies_cost := supplies_cost * (1 - discount_rate)
  let discounted_box_cost := food_cost + discounted_supplies_cost + shipping_cost
  let discounted_total_cost := initial_boxes * discounted_box_cost
  let donation := discounted_total_cost * donation_multiplier * exchange_rate
  let additional_boxes := (donation / discounted_box_cost).floor.toNat
  initial_boxes + additional_boxes

/-- Theorem stating that given the specified conditions, the total number of boxes packed is 2319 -/
theorem charity_distribution_result : 
  charity_distribution 400 80 165 7 (1/10) 200 4 (6/5) = 2319 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_charity_distribution_result_l868_86820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_is_sqrt3_over_3_pi_l868_86889

-- Define the cone properties
noncomputable def slant_height : ℝ := 2
def lateral_surface_is_semicircle : Prop := True  -- We assume this as given

-- Define the volume of the cone
noncomputable def cone_volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

-- Theorem statement
theorem cone_volume_is_sqrt3_over_3_pi :
  ∃ (r h : ℝ), 
    r^2 + h^2 = slant_height^2 ∧
    2 * Real.pi * r = 2 * Real.pi ∧
    cone_volume r h = (Real.sqrt 3 / 3) * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_is_sqrt3_over_3_pi_l868_86889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_algorithm_not_infinite_l868_86899

/-- Represents an algorithm --/
structure Algorithm where
  steps : List String
  solves_problem : Bool

/-- Definition of algorithm characteristics --/
def has_finiteness (a : Algorithm) : Prop := a.steps.length < (Nat.succ 0)
def has_definiteness (a : Algorithm) : Prop := ∀ s ∈ a.steps, s ≠ ""
def has_effectiveness (a : Algorithm) : Prop := a.solves_problem = true
def has_infiniteness (a : Algorithm) : Prop := ¬(has_finiteness a)

/-- Theorem stating that an algorithm does not possess infiniteness --/
theorem algorithm_not_infinite (a : Algorithm) : ¬(has_infiniteness a) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_algorithm_not_infinite_l868_86899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_min_value_in_interval_l868_86868

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + a

-- Define the interval
def I : Set ℝ := Set.Icc (-2) 2

-- Theorem for the tangent line equation
theorem tangent_line_equation (a : ℝ) (h : a = -10) :
  ∃ y : ℝ, y = f a 2 ∧ (9 : ℝ) * 2 - y - 6 = 0 := by
  sorry

-- Theorem for the minimum value
theorem min_value_in_interval (a : ℝ) (h : ∃ x ∈ I, f a x = 20 ∧ ∀ y ∈ I, f a y ≤ 20) :
  ∃ x ∈ I, f a x = -7 ∧ ∀ y ∈ I, f a y ≥ -7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_min_value_in_interval_l868_86868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_implies_x_value_l868_86876

/-- Given two vectors v and w in ℝ², prove that if the projection of v onto w is (4, -1), then the x-coordinate of v is 21/4. -/
theorem projection_implies_x_value (v w : ℝ × ℝ) (h : v.2 = 4 ∧ w = (8, -2)) :
  ((v.1 * 8 - 8) / 68) • w = (4, -1) → v.1 = 21 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_implies_x_value_l868_86876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statements_l868_86852

theorem correct_statements :
  (Real.sqrt ((-2)^2) = 2) ∧
  ((1 / (Real.sqrt 3 + Real.sqrt 2)) = Real.sqrt 3 - Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statements_l868_86852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_magnitude_squared_l868_86831

def point_A : Fin 3 → ℝ := ![3, 7, -4]

def point_B (A : Fin 3 → ℝ) : Fin 3 → ℝ := ![A 0, 0, A 2]

def vector_OB (B : Fin 3 → ℝ) : Fin 3 → ℝ := B

def magnitude_squared (v : Fin 3 → ℝ) : ℝ :=
  (v 0) * (v 0) + (v 1) * (v 1) + (v 2) * (v 2)

theorem projection_magnitude_squared :
  magnitude_squared (vector_OB (point_B point_A)) = 25 := by
  simp [magnitude_squared, vector_OB, point_B, point_A]
  norm_num

#eval magnitude_squared (vector_OB (point_B point_A))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_magnitude_squared_l868_86831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_100_l868_86819

/-- Annual profit function -/
noncomputable def L (x : ℝ) : ℝ :=
  if x < 80 then
    -(1/3) * x^2 + 40 * x - 250
  else
    -(x + 10000 / x) + 1200

/-- Theorem stating the maximum profit occurs at x = 100 with a value of 1000 -/
theorem max_profit_at_100 :
  (∀ x > 0, L x ≤ L 100) ∧ L 100 = 1000 := by
  sorry

#check max_profit_at_100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_100_l868_86819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_in_second_quadrant_l868_86869

-- Define the point P
noncomputable def P : ℝ × ℝ := (Real.sin 2, Real.cos 2)

-- Define the property of being in the second quadrant
def is_in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

-- Theorem statement
theorem terminal_side_in_second_quadrant :
  let (x, y) := P
  is_in_second_quadrant x y := by
  -- Unfold the definition of P
  unfold P
  -- Split into two subgoals
  apply And.intro
  -- Prove Real.cos 2 < 0
  · sorry
  -- Prove Real.sin 2 > 0
  · sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_in_second_quadrant_l868_86869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_data_l868_86817

noncomputable def data : List ℝ := [47, 48, 51, 54, 55]

noncomputable def mean (xs : List ℝ) : ℝ :=
  (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  (xs.map (fun x => (x - mean xs)^2)).sum / xs.length

theorem variance_of_data : variance data = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_data_l868_86817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sinusoidal_parameters_l868_86870

/-- A sinusoidal function with angular frequency ω and phase shift φ -/
noncomputable def sinusoidalFunction (ω φ : ℝ) : ℝ → ℝ := fun x ↦ Real.sin (ω * x + φ)

/-- The period of a sinusoidal function -/
noncomputable def period (ω : ℝ) : ℝ := 2 * Real.pi / ω

theorem sinusoidal_parameters (f : ℝ → ℝ) (ω φ : ℝ) 
  (h1 : f = sinusoidalFunction ω φ)
  (h2 : period ω = 8)
  (h3 : ∀ x, f x ≤ f 1) :
  ω = Real.pi / 4 ∧ φ = Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sinusoidal_parameters_l868_86870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_product_l868_86815

theorem tan_product (α β : ℝ) 
  (h1 : Real.cos (α + β) = 1/3) 
  (h2 : Real.cos (α - β) = 1/5) : 
  Real.tan α * Real.tan β = -1/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_product_l868_86815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_theorem_l868_86812

def triangle_area (a c : ℝ) (C : ℝ) : Set ℝ :=
  let b₁ := (3 + Real.sqrt 5) / 2
  let b₂ := (3 - Real.sqrt 5) / 2
  let area₁ := (1/2) * a * b₁ * Real.sin C
  let area₂ := (1/2) * a * b₂ * Real.sin C
  {area₁, area₂}

theorem triangle_area_theorem :
  triangle_area 3 (Real.sqrt 7) (π/3) = {3 * Real.sqrt 3 / 4, 3 * Real.sqrt 3 / 2} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_theorem_l868_86812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_one_is_local_max_l868_86894

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x - x^2

-- State the theorem
theorem x_one_is_local_max : 
  ∃ δ > 0, ∀ x : ℝ, x ≠ 1 → |x - 1| < δ → f x ≤ f 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_one_is_local_max_l868_86894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_middle_term_l868_86886

theorem arithmetic_sequence_middle_term 
  (a₁ a₅ e : ℝ) 
  (h₁ : a₁ = 9) 
  (h₂ : a₅ = 63) 
  (h₃ : e = (a₁ + a₅) / 2) : 
  e = 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_middle_term_l868_86886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_l868_86853

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.log (-x^2 + x + 6)

-- Define the domain of f(x)
def domain : Set ℝ := {x : ℝ | -2 < x ∧ x < 3}

-- Theorem statement
theorem monotonic_decreasing_interval :
  ∃ (a b : ℝ), a = 1/2 ∧ b = 3 ∧
  (∀ x y, x ∈ domain → y ∈ domain → a < x → x < y → y < b → f y < f x) ∧
  (∀ x y, x ∈ domain → y ∈ domain → x < y → y ≤ a → f x ≤ f y) ∧
  (∀ x y, x ∈ domain → y ∈ domain → b ≤ x → x < y → f x ≤ f y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_l868_86853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l868_86827

noncomputable def f (x : ℝ) : ℝ := (2 * x^2) / (x^2 + 4)

theorem range_of_f :
  Set.range f = Set.Icc 0 2 \ {2} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l868_86827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_perimeter_l868_86802

/-- Represents a trapezoid ABCD with specific properties -/
structure Trapezoid where
  AB : ℝ
  CD : ℝ
  height : ℝ
  ab_cd_parallel : AB ≠ CD → True  -- AB and CD are parallel
  ab_cd_equal : AB = CD  -- AB and CD are equal
  ad_bc_perpendicular : True  -- AD and BC are perpendicular to AB and CD

/-- The perimeter of the trapezoid ABCD -/
noncomputable def perimeter (t : Trapezoid) : ℝ :=
  2 * (t.AB + t.height * Real.sqrt 2)

/-- Theorem stating the perimeter of the specific trapezoid -/
theorem trapezoid_perimeter : 
  ∀ (t : Trapezoid), 
  t.AB = 10 ∧ t.CD = 18 ∧ t.height = 4 → 
  perimeter t = 28 + 8 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_perimeter_l868_86802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l868_86832

theorem triangle_side_length (A B C : Real) (a b c : Real) :
  A = Real.pi/4 →  -- 45°
  B = Real.pi/3 →  -- 60°
  a = 10 →
  (Real.sin A) * b = (Real.sin B) * a →
  b = 5 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l868_86832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_horizontal_probability_l868_86895

/-- Represents a point on a 10x10 grid -/
structure GridPoint where
  x : Fin 10
  y : Fin 10
deriving Fintype, DecidableEq

/-- The set of all points on the 10x10 grid -/
def gridPoints : Finset GridPoint := Finset.univ

theorem vertical_horizontal_probability (P : GridPoint) :
  let Q := gridPoints.erase P
  (Q.filter (λ q => q.x = P.x ∨ q.y = P.y)).card / Q.card = 2 / 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_horizontal_probability_l868_86895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inhabitant_is_resident_l868_86883

/-- The type representing words in the English language -/
def Word : Type := String

/-- The meaning of a word -/
def meaning (w : Word) : Word := sorry

/-- Axiom: The meaning of "inhabitant" is "resident" -/
axiom inhabitant_meaning : meaning "inhabitant" = "resident"

/-- Theorem: The meaning of "inhabitant" is equivalent to "resident" -/
theorem inhabitant_is_resident : meaning "inhabitant" = "resident" := by
  exact inhabitant_meaning


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inhabitant_is_resident_l868_86883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_proof_l868_86855

theorem quadratic_function_proof (a b c : ℝ) : 
  let f := fun (x : ℝ) => a * x^2 + b * x + c
  (f 0 = f 4) →
  (∃ r₁ r₂ : ℝ, a * r₁^2 + b * r₁ + c = 0 ∧ 
                a * r₂^2 + b * r₂ + c = 0 ∧ 
                r₁^2 + r₂^2 = 10) →
  (f 0 = 3) →
  (∀ x, f x = x^2 - 4*x + 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_proof_l868_86855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_curves_l868_86880

/-- The area enclosed by the curves y = x^2, y = x, and y = 3x -/
noncomputable def enclosedArea : ℝ := ∫ x in (0)..(1), (x - x^2) + ∫ x in (1)..(3), (3*x - x^2)

/-- Theorem stating that the area enclosed by the curves y = x^2, y = x, and y = 3x is 10/3 -/
theorem area_enclosed_by_curves : enclosedArea = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_curves_l868_86880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l868_86879

-- Define a triangle ABC
structure Triangle where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  a : ℝ  -- Side opposite to A
  b : ℝ  -- Side opposite to B
  c : ℝ  -- Side opposite to C

-- Define the area function
noncomputable def area (t : Triangle) : ℝ := 
  1/2 * t.a * t.c * Real.sin t.B

-- Define the theorem
theorem triangle_properties (abc : Triangle) 
  (h1 : (Real.cos abc.A - 2 * Real.cos abc.C) / Real.cos abc.B = (2 * abc.c - abc.a) / abc.b)
  (h2 : Real.cos abc.B = 1/4)
  (h3 : abc.b = 2) :
  Real.sin abc.C / Real.sin abc.A = 2 ∧ 
  area abc = Real.sqrt 15 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l868_86879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l868_86803

/-- The side length of the larger hexagon -/
noncomputable def large_hexagon_side : ℝ := 3

/-- The area of a regular hexagon with side length s -/
noncomputable def hexagon_area (s : ℝ) : ℝ := (3 * Real.sqrt 3 / 2) * s^2

/-- The area of a semicircle with radius r -/
noncomputable def semicircle_area (r : ℝ) : ℝ := Real.pi * r^2 / 2

/-- The number of semicircles in the configuration -/
def num_semicircles : ℕ := 6

/-- The theorem stating the area of the shaded region -/
theorem shaded_area_calculation :
  let larger_hexagon_area := hexagon_area large_hexagon_side
  let semicircle_radius := large_hexagon_side / 2
  let total_semicircle_area := num_semicircles * semicircle_area semicircle_radius
  let smaller_hexagon_area := hexagon_area semicircle_radius
  let shaded_area := larger_hexagon_area - total_semicircle_area - smaller_hexagon_area
  shaded_area = (81 * Real.sqrt 3) / 8 - (27 * Real.pi) / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l868_86803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_toy_organization_time_l868_86810

/-- The time it takes to organize toys given the following conditions:
  * There are 50 toys to be organized
  * 4 toys are put into the box every 45 seconds
  * 3 toys are taken out of the box every 45 seconds
  * We want to find the time in minutes to put all 50 toys into the box
-/
theorem toy_organization_time
  (total_toys : ℕ)
  (toys_in_per_cycle : ℕ)
  (toys_out_per_cycle : ℕ)
  (cycle_duration : ℕ)
  (h1 : total_toys = 50)
  (h2 : toys_in_per_cycle = 4)
  (h3 : toys_out_per_cycle = 3)
  (h4 : cycle_duration = 45) :
  (((total_toys - 3) * cycle_duration) / (toys_in_per_cycle - toys_out_per_cycle) + cycle_duration) / 60 = 36 :=
by
  sorry

#check toy_organization_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_toy_organization_time_l868_86810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nearest_integer_to_five_to_sixth_l868_86807

theorem nearest_integer_to_five_to_sixth (x : ℝ) : 
  x = (3 + 2)^6 → ∃ n : ℤ, n = 9794 ∧ ∀ m : ℤ, |x - n| ≤ |x - m| :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nearest_integer_to_five_to_sixth_l868_86807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_f_l868_86830

-- Define the function f(x) = (x-3)e^x
noncomputable def f (x : ℝ) : ℝ := (x - 3) * Real.exp x

-- State the theorem
theorem monotonic_increasing_interval_f :
  StrictMonoOn f (Set.Ioi 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_f_l868_86830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_plus_beta_eq_three_pi_over_four_l868_86805

theorem alpha_plus_beta_eq_three_pi_over_four 
  (α β : Real) 
  (h1 : 0 < α ∧ α < π/2) 
  (h2 : 0 < β ∧ β < π/2) 
  (h3 : Real.cos α = Real.sqrt 5 / 5) 
  (h4 : Real.sin β = 3 * Real.sqrt 10 / 10) : 
  α + β = 3*π/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_plus_beta_eq_three_pi_over_four_l868_86805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_ratio_l868_86822

/-- Triangle ABC with side lengths -/
structure Triangle :=
  (AB BC AC : ℝ)

/-- Point M on side AC of triangle ABC -/
structure PointOnSide (t : Triangle) :=
  (AM MC : ℝ)
  (sum_eq : AM + MC = t.AC)

/-- Incircle of a triangle -/
structure Incircle (t : Triangle) :=
  (radius : ℝ)

/-- Main theorem -/
theorem incircle_ratio (t : Triangle) (m : PointOnSide t) 
  (h1 : t.AB = 14) (h2 : t.BC = 15) (h3 : t.AC = 17)
  (h4 : Incircle ⟨t.AB, m.AM, t.AB + m.AM - t.AC⟩ = Incircle ⟨t.BC, m.MC, t.BC + m.MC - t.AC⟩) :
  m.AM / m.MC = 36 / 83 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_ratio_l868_86822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cat_food_percentage_theorem_l868_86851

/-- Represents the food distribution for dogs and cats -/
structure PetFood where
  numDogs : ℕ
  numCats : ℕ
  dogFood : ℚ
  catFood : ℚ

/-- The percentage of food a single cat receives -/
noncomputable def catFoodPercentage (pf : PetFood) : ℚ :=
  (pf.catFood / (pf.numDogs * pf.dogFood + pf.numCats * pf.catFood)) * 100

/-- Theorem stating that a cat receives 3.125% of the total food under given conditions -/
theorem cat_food_percentage_theorem (pf : PetFood) 
  (h1 : pf.numDogs = 7)
  (h2 : pf.numCats = 4)
  (h3 : pf.numCats * pf.catFood = pf.dogFood) :
  catFoodPercentage pf = 25/8 := by
  sorry

#eval (25 : ℚ) / 8

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cat_food_percentage_theorem_l868_86851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_difference_l868_86864

/-- The line on which point P lies -/
def line (x y : ℝ) : Prop := 2 * x - y - 4 = 0

/-- Point A -/
def A : ℝ × ℝ := (4, -1)

/-- Point B -/
def B : ℝ × ℝ := (3, 4)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- The point P that maximizes the distance difference -/
def P : ℝ × ℝ := (5, 6)

theorem max_distance_difference :
  line P.1 P.2 ∧
  ∀ Q : ℝ × ℝ, line Q.1 Q.2 →
    |distance P A - distance P B| ≥ |distance Q A - distance Q B| := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_difference_l868_86864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cucumber_weight_after_evaporation_l868_86837

theorem cucumber_weight_after_evaporation 
  (initial_weight : ℝ) 
  (initial_water_percent : ℝ) 
  (final_water_percent : ℝ) 
  (h1 : initial_weight = 100) 
  (h2 : initial_water_percent = 99) 
  (h3 : final_water_percent = 96) : 
  (initial_weight * (1 - initial_water_percent / 100)) / (1 - final_water_percent / 100) = 25 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cucumber_weight_after_evaporation_l868_86837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_symmetry_l868_86857

/-- Given a point (a, b) and the line x + y = 0, 
    the point symmetric to (a, b) with respect to this line is (-a, -b). -/
theorem point_symmetry (a b : ℝ) : 
  let original : ℝ × ℝ := (a, b)
  let symmetry_line : Set (ℝ × ℝ) := {p | p.1 + p.2 = 0}
  let symmetric : ℝ × ℝ := (-a, -b)
  ∃ (mid : ℝ × ℝ), mid ∈ symmetry_line ∧ 
    (mid.1 - original.1 = symmetric.1 - mid.1) ∧
    (mid.2 - original.2 = symmetric.2 - mid.2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_symmetry_l868_86857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_increasing_by_more_than_one_and_by_one_l868_86833

noncomputable def a (n : ℕ+) : ℕ := Int.toNat ⌊(((n - 1) ^ 2 + n ^ 2 : ℝ).sqrt)⌋

theorem infinitely_many_increasing_by_more_than_one_and_by_one :
  (∃ S : Set ℕ+, Set.Infinite S ∧ ∀ m ∈ S, a (m + 1) - a m > 1) ∧
  (∃ T : Set ℕ+, Set.Infinite T ∧ ∀ m ∈ T, a (m + 1) - a m = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_increasing_by_more_than_one_and_by_one_l868_86833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_sin_le_one_l868_86800

theorem negation_of_sin_le_one :
  (¬ ∀ x : ℝ, Real.sin x ≤ 1) ↔ (∃ x₀ : ℝ, Real.sin x₀ > 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_sin_le_one_l868_86800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_m_values_l868_86882

/-- An ellipse with equation mx^2 + 4y^2 = 1 and eccentricity √2/2 has m equal to 2 or 8 -/
theorem ellipse_eccentricity_m_values (m : ℝ) :
  (∃ x y : ℝ, m * x^2 + 4 * y^2 = 1) →  -- ellipse equation
  (∃ a c : ℝ, a > 0 ∧ c > 0 ∧ c / a = Real.sqrt 2 / 2) →  -- eccentricity condition
  (m = 2 ∨ m = 8) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_m_values_l868_86882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_percentage_approx_5_105_l868_86839

/-- Calculates the percentage of total spending paid in taxes given spending percentages, tax rates, and discounts -/
noncomputable def tax_percentage_of_total_spending (clothing_percent food_percent other_percent : ℝ)
  (clothing_tax_rate food_tax_rate other_tax_rate : ℝ)
  (clothing_discount other_discount : ℝ) : ℝ :=
  let clothing_amount := clothing_percent
  let food_amount := food_percent
  let other_amount := other_percent
  let clothing_discounted := clothing_amount * (1 - clothing_discount)
  let other_discounted := other_amount * (1 - other_discount)
  let total_discounted := clothing_discounted + food_amount + other_discounted
  let clothing_tax := clothing_discounted * clothing_tax_rate
  let food_tax := food_amount * food_tax_rate
  let other_tax := other_discounted * other_tax_rate
  let total_tax := clothing_tax + food_tax + other_tax
  (total_tax / total_discounted) * 100

/-- The percentage of total spending paid in taxes is approximately 5.105% -/
theorem tax_percentage_approx_5_105 :
  ∃ ε > 0, |tax_percentage_of_total_spending 0.4 0.25 0.35 0.05 0.03 0.07 0.1 0.15 - 5.105| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_percentage_approx_5_105_l868_86839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_arrangement_theorem_l868_86838

def number_of_arrangements (total_books : ℕ) (arabic_books : ℕ) (german_books : ℕ) (spanish_books : ℕ) : ℕ :=
  sorry -- Definition to be filled later

theorem book_arrangement_theorem :
  let total_books := 10
  let arabic_books := 2
  let german_books := 4
  let spanish_books := 4
  let indistinguishable_german_books := 2
  number_of_arrangements total_books arabic_books german_books spanish_books = 576 := by
  sorry

#check book_arrangement_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_arrangement_theorem_l868_86838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_sum_equals_sqrt_pi_over_two_l868_86841

theorem infinite_sum_equals_sqrt_pi_over_two :
  ∑' (n : ℕ+), (n : ℝ) / ((n : ℝ)^4 + 4*(n : ℝ)^2 + 1) = Real.sqrt π / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_sum_equals_sqrt_pi_over_two_l868_86841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_binomial_expansion_l868_86821

/-- The constant term in the binomial expansion of (x - 2/x)^6 is -160 -/
theorem constant_term_binomial_expansion :
  let f : ℝ → ℝ := λ x => (x - 2/x)^6
  ∃ c : ℝ, c = -160 ∧ 
    ∀ x : ℝ, x ≠ 0 → (∃ p : Polynomial ℝ, (∀ y, f y = p.eval y) ∧ p.coeff 0 = c) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_binomial_expansion_l868_86821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_proof_slope_range_proof_u_range_proof_l868_86866

-- Define the points A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (2, 0)

-- Define the condition for point P on curve C
def on_curve_C (P : ℝ × ℝ) : Prop :=
  (P.1 + 2) * (P.1 - 2) + P.2 * P.2 = -3

-- Define the equation of curve C
def curve_C_equation (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

-- Define the point M
def M : ℝ × ℝ := (0, -2)

-- Define the condition for line l intersecting curve C
def line_intersects_C (k : ℝ) : Prop :=
  ∃ (x y : ℝ), curve_C_equation x y ∧ y = k * x - 2

-- Define the range of slope k
noncomputable def slope_range (k : ℝ) : Prop :=
  k ≤ -Real.sqrt 3 ∨ k ≥ Real.sqrt 3

-- Define u in terms of x and y
noncomputable def u (x y : ℝ) : ℝ :=
  (y + 2) / (x - 1)

-- Define the range of u
def u_range (u : ℝ) : Prop :=
  u ≤ -(3/4)

theorem curve_C_proof :
  ∀ (P : ℝ × ℝ), on_curve_C P ↔ curve_C_equation P.1 P.2 :=
by sorry

theorem slope_range_proof :
  ∀ (k : ℝ), line_intersects_C k ↔ slope_range k :=
by sorry

theorem u_range_proof :
  ∀ (x y : ℝ), curve_C_equation x y → u_range (u x y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_proof_slope_range_proof_u_range_proof_l868_86866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inversion_of_line_or_circle_l868_86849

-- Define the basic geometric objects
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

structure Plane where
  normal : Point3D
  d : ℝ

structure Sphere where
  center : Point3D
  radius : ℝ

-- Define geometric objects as sets of points
def Line : Type := Set Point3D
def Circle : Type := Set Point3D

-- Define predicates for geometric objects
def IsLine (s : Set Point3D) : Prop := sorry
def IsCircle (s : Set Point3D) : Prop := sorry
def IsPlane (s : Set Point3D) : Prop := sorry
def IsSphere (s : Set Point3D) : Prop := sorry

-- Define inversion as a function on Point3D
noncomputable def inversion (center : Point3D) (radius : ℝ) : Point3D → Point3D := sorry

-- Define the image of a set under inversion
def inversionImage (center : Point3D) (radius : ℝ) (s : Set Point3D) : Set Point3D :=
  {p | ∃ q ∈ s, inversion center radius q = p}

-- Axioms based on the given conditions
axiom line_is_intersection_of_planes (l : Line) : 
  ∃ (p1 p2 : Plane), l = {x : Point3D | x ∈ (Set.inter {y : Point3D | sorry} {z : Point3D | sorry})}

axiom circle_is_intersection_of_sphere_and_plane (c : Circle) :
  ∃ (s : Sphere) (p : Plane), c = {x : Point3D | x ∈ (Set.inter {y : Point3D | sorry} {z : Point3D | sorry})}

axiom inversion_transforms_plane_or_sphere (center : Point3D) (radius : ℝ) :
  (∀ (p : Set Point3D), IsPlane p → ∃ (s : Set Point3D), (IsPlane s ∨ IsSphere s) ∧ inversionImage center radius p = s) ∧
  (∀ (s : Set Point3D), IsSphere s → ∃ (t : Set Point3D), (IsPlane t ∨ IsSphere t) ∧ inversionImage center radius s = t)

-- The main theorem
theorem inversion_of_line_or_circle (center : Point3D) (radius : ℝ) (s : Set Point3D) 
  (h : IsLine s ∨ IsCircle s) : 
  IsLine (inversionImage center radius s) ∨ IsCircle (inversionImage center radius s) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inversion_of_line_or_circle_l868_86849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_product_sets_l868_86848

/-- A set of six real numbers where each number is the product of two others from the set -/
def ProductSet : Type := { s : Finset ℝ // s.card = 6 ∧ ∀ x ∈ s, ∃ y z, y ∈ s ∧ z ∈ s ∧ x = y * z ∧ y ≠ x ∧ z ≠ x }

/-- There exist infinitely many distinct ProductSets -/
theorem infinitely_many_product_sets : ∀ n : ℕ, ∃ S : Finset ProductSet, S.card > n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_product_sets_l868_86848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_in_list_l868_86890

noncomputable def is_integer (x : ℝ) : Prop := ∃ n : ℤ, x = n

noncomputable def nth_root (n : ℕ) (a : ℝ) : ℝ := Real.rpow a (1 / n)

noncomputable def count_integers (f : ℕ → ℝ) : ℕ := Nat.card {n : ℕ | is_integer (f n)}

theorem count_integers_in_list : 
  count_integers (λ k => nth_root k 32768) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_in_list_l868_86890
