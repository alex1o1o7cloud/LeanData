import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_theorem_l5_573

/-- Given three positive integers with HCF 23 and LCM having additional factors 11, 12, and 13,
    the largest of these numbers is 23 * 11 * 12 * 13 -/
theorem largest_number_theorem (a b c : ℕ+) : 
  (Nat.gcd a.val b.val = 23 ∧ Nat.gcd (Nat.gcd a.val b.val) c.val = 23) → 
  (∃ (x y z : ℕ+), Nat.lcm (Nat.lcm a.val b.val) c.val = 23 * 11 * 12 * 13 * x.val * y.val * z.val) →
  (max a.val (max b.val c.val) = 23 * 11 * 12 * 13) := by
  sorry

#check largest_number_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_theorem_l5_573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boa_constrictor_length_l5_548

/-- Given information about Albert's snakes, prove the length of the boa constrictor --/
theorem boa_constrictor_length 
  (num_snakes : ℝ) 
  (garden_snake_length : ℝ) 
  (boa_shortness_factor : ℝ) 
  (h1 : num_snakes = 2.0)
  (h2 : garden_snake_length = 10.0)
  (h3 : boa_shortness_factor = 7.0) :
  abs ((garden_snake_length / boa_shortness_factor) - 1.43) < 0.001 := by
  sorry

#check boa_constrictor_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boa_constrictor_length_l5_548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_is_fixed_point_l5_512

/-- The function f(z) as defined in the problem -/
noncomputable def f (z : ℂ) : ℂ := ((-1 + Complex.I * Real.sqrt 3) * z + (-3 * Real.sqrt 3 - 12 * Complex.I)) / 2

/-- The fixed point of f -/
noncomputable def c : ℂ := 2.25 + 2.25 * Real.sqrt 3 - 3 * Complex.I

/-- Theorem stating that c is a fixed point of f -/
theorem c_is_fixed_point : f c = c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_is_fixed_point_l5_512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l5_568

/-- A quadratic function with vertex at (-1, 1) and y-intercept at (0, 2) -/
def f (x : ℝ) : ℝ := x^2 + 2*x + 2

theorem quadratic_function_properties :
  (∀ x, f x = x^2 + 2*x + 2) ∧
  (f (-1) = 1) ∧
  (f 0 = 2) ∧
  (f 8 = 82) ∧
  (∀ y, y < 1 → ∀ x, f x ≠ y) :=
by
  sorry

#check quadratic_function_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l5_568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_log_function_l5_501

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (-a * x)

theorem odd_log_function (a : ℝ) :
  (∀ x, f a x = -(f a (-x))) → (a = 1 ∨ a = -1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_log_function_l5_501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_geometric_series_ratio_l5_500

/-- For an infinite geometric series with first term a and sum S, 
    the common ratio r is (S - a) / S -/
noncomputable def geometric_series_ratio (a : ℝ) (S : ℝ) : ℝ := (S - a) / S

theorem infinite_geometric_series_ratio :
  let a : ℝ := 500
  let S : ℝ := 3000
  geometric_series_ratio a S = 5/6 := by
  -- Unfold the definition of geometric_series_ratio
  unfold geometric_series_ratio
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_geometric_series_ratio_l5_500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_theorem_l5_537

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

-- Define the external point P
def P : ℝ × ℝ := (3, 4)

-- Define a line passing through P and intersecting the circle
def line_intersects_circle (A B : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, circle_eq (A.1 + t * (B.1 - A.1)) (A.2 + t * (B.2 - A.2)) ∧
           circle_eq B.1 B.2 ∧
           (A ≠ B) ∧
           (∃ s : ℝ, P.1 = A.1 + s * (B.1 - A.1) ∧ P.2 = A.2 + s * (B.2 - A.2))

-- Theorem statement
theorem dot_product_theorem (A B : ℝ × ℝ) :
  line_intersects_circle A B →
  (P.1 - A.1) * (P.1 - B.1) + (P.2 - A.2) * (P.2 - B.2) = 16 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_theorem_l5_537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_albaszu_new_productivity_l5_519

/-- Represents the productivity of the Albaszu machine -/
structure AlbaszuMachine where
  initial_productivity : ℕ
  repair_factor : ℚ
  initial_hours : ℕ
  new_hours : ℕ
  new_worker1_productivity : ℚ
  new_worker2_productivity : ℚ
  diminishing_return_factor : ℚ

/-- Calculates the new productivity of the Albaszu machine -/
noncomputable def calculate_new_productivity (machine : AlbaszuMachine) : ℚ :=
  let base_productivity := machine.initial_productivity * machine.repair_factor
  let hour_adjusted_productivity := base_productivity * (machine.new_hours / machine.initial_hours)
  let new_workers_productivity := base_productivity * (machine.new_worker1_productivity + machine.new_worker2_productivity)
  let total_productivity := hour_adjusted_productivity + new_workers_productivity
  total_productivity * (1 - machine.diminishing_return_factor)

/-- Theorem stating the new productivity of the Albaszu machine -/
theorem albaszu_new_productivity :
  let machine := AlbaszuMachine.mk 10 (3/2) 8 10 (4/5) (3/5) (1/10)
  Int.floor (calculate_new_productivity machine) = 35 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_albaszu_new_productivity_l5_519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_difference_l5_503

/-- Computes the compound interest balance -/
def compoundInterest (principal : ℝ) (rate : ℝ) (periods : ℕ) : ℝ :=
  principal * (1 + rate) ^ periods

/-- Computes the positive difference between two balances -/
def balanceDifference (balance1 : ℝ) (balance2 : ℝ) : ℝ :=
  abs (balance1 - balance2)

theorem compound_interest_difference :
  let angelaPrincipal := 9000
  let angelaRate := 0.05
  let angelaPeriods := 20
  let bobPrincipal := 11000
  let bobRate := 0.03
  let bobPeriods := 40
  let angelaBalance := compoundInterest angelaPrincipal angelaRate angelaPeriods
  let bobBalance := compoundInterest bobPrincipal bobRate bobPeriods
  let difference := balanceDifference bobBalance angelaBalance
  ⌊difference⌋ = 12002 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_difference_l5_503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_distances_to_foci_l5_566

-- Define the ellipse
def is_on_ellipse (P : ℝ × ℝ) : Prop :=
  P.1^2 + 3 * P.2^2 = 12

-- Define the foci
def are_foci (F1 F2 : ℝ × ℝ) : Prop :=
  ∃ c : ℝ, F1 = (c, 0) ∧ F2 = (-c, 0) ∧ c^2 = 8

-- Define the vector from a point to another
def vec (A B : ℝ × ℝ) : ℝ × ℝ :=
  (B.1 - A.1, B.2 - A.2)

-- Define the magnitude of a vector
noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

-- Theorem statement
theorem min_sum_of_distances_to_foci :
  ∀ (P : ℝ × ℝ) (F1 F2 : ℝ × ℝ),
    is_on_ellipse P →
    are_foci F1 F2 →
    ∀ Q : ℝ × ℝ, is_on_ellipse Q →
      magnitude (vec P F1 + vec P F2) ≤ magnitude (vec Q F1 + vec Q F2) ∧
      (∃ R : ℝ × ℝ, is_on_ellipse R ∧ magnitude (vec R F1 + vec R F2) = 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_distances_to_foci_l5_566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_range_for_parallelogram_area_l5_507

/-- Given two vectors α and β on a plane with magnitudes at most 1 and
    forming a parallelogram of area 1/2, the angle between them is
    between π/6 and 5π/6 inclusive. -/
theorem angle_range_for_parallelogram_area (α β : ℝ × ℝ) 
    (h_norm_α : ‖α‖ ≤ 1)
    (h_norm_β : ‖β‖ ≤ 1)
    (h_area : ‖α.1 * β.2 - α.2 * β.1‖ / 2 = 1/2) :
    let θ := Real.arccos ((α.1 * β.1 + α.2 * β.2) / (‖α‖ * ‖β‖))
    π/6 ≤ θ ∧ θ ≤ 5*π/6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_range_for_parallelogram_area_l5_507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_AB_on_AC_l5_515

noncomputable def A : ℝ × ℝ := (1, 1)
noncomputable def B : ℝ × ℝ := (-3, 3)
noncomputable def C : ℝ × ℝ := (4, 2)

noncomputable def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
noncomputable def AC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

theorem projection_AB_on_AC : 
  dot_product AB AC / magnitude AC = -Real.sqrt 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_AB_on_AC_l5_515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_condition_implies_trapezoid_or_parallelogram_l5_560

/-- A quadrilateral with diagonals -/
structure QuadrilateralWithDiagonals where
  /-- The areas of the four triangles formed by the diagonals -/
  triangle_areas : Fin 4 → ℝ
  /-- The condition that the square of one triangle's area equals the product of adjacent areas -/
  area_condition : ∃ i : Fin 4, (triangle_areas i) ^ 2 = (triangle_areas (i + 1)) * (triangle_areas (i - 1))

/-- The property of being a trapezoid -/
def is_trapezoid (q : QuadrilateralWithDiagonals) : Prop :=
  sorry -- Definition of trapezoid property

/-- The property of being a parallelogram -/
def is_parallelogram (q : QuadrilateralWithDiagonals) : Prop :=
  sorry -- Definition of parallelogram property

/-- The property of being either a trapezoid or a parallelogram -/
def is_trapezoid_or_parallelogram (q : QuadrilateralWithDiagonals) : Prop :=
  is_trapezoid q ∨ is_parallelogram q

/-- The main theorem -/
theorem quadrilateral_area_condition_implies_trapezoid_or_parallelogram 
  (q : QuadrilateralWithDiagonals) : is_trapezoid_or_parallelogram q := by
  sorry -- Proof to be completed


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_condition_implies_trapezoid_or_parallelogram_l5_560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l5_569

/-- Circle C₁ with equation x² + y² + 2x + 8y - 8 = 0 -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 8*y - 8 = 0

/-- Circle C₂ with equation x² + y² - 4x - 4y - 2 = 0 -/
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y - 2 = 0

/-- The center of C₁ -/
def center_C₁ : ℝ × ℝ := (-1, -4)

/-- The center of C₂ -/
def center_C₂ : ℝ × ℝ := (2, 2)

/-- The radius of C₁ -/
def radius_C₁ : ℝ := 5

/-- The radius of C₂ -/
noncomputable def radius_C₂ : ℝ := Real.sqrt 10

/-- The distance between the centers of C₁ and C₂ -/
noncomputable def distance_between_centers : ℝ := 3 * Real.sqrt 5

/-- Theorem: The circles C₁ and C₂ are intersecting -/
theorem circles_intersect : 
  radius_C₁ - radius_C₂ < distance_between_centers ∧ 
  distance_between_centers < radius_C₁ + radius_C₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l5_569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cave_entry_possible_l5_559

/-- Represents the state of a switch (Up or Down) -/
inductive SwitchState
| Up
| Down

/-- Represents the state of all four switches -/
def SwitchConfiguration := Fin 4 → SwitchState

/-- Checks if all switches are in the same state -/
def allSameState (config : SwitchConfiguration) : Prop :=
  ∀ i j : Fin 4, config i = config j

/-- Represents a single attempt to change switch states -/
def Attempt := SwitchConfiguration → SwitchConfiguration

/-- Represents a sequence of attempts -/
def AttemptSequence := List Attempt

/-- Theorem stating that it's possible to enter the cave within 10 attempts -/
theorem cave_entry_possible :
  ∃ (attempts : AttemptSequence),
    (attempts.length ≤ 10) ∧
    (∀ (initial_config : SwitchConfiguration),
      ∃ (final_config : SwitchConfiguration),
        (final_config = List.foldl (fun config attempt => attempt config) initial_config attempts) ∧
        (allSameState final_config)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cave_entry_possible_l5_559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_not_sufficient_l5_508

/-- Represents a sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- Checks if a sequence is geometric -/
def IsGeometric (s : Sequence) : Prop :=
  ∃ (r : ℝ) (a₀ : ℝ), a₀ ≠ 0 ∧ ∀ n, s n = a₀ * r^n

/-- The sequence a_n = n * (-2)^n -/
noncomputable def a : Sequence :=
  λ n => n * (-2)^n

/-- A general sequence b_n -/
noncomputable def b : Sequence :=
  λ n => n  -- We define b_n = n as per the problem statement

/-- The ratio sequence a_n / b_n -/
noncomputable def ratio : Sequence :=
  λ n => (a n) / (b n)

theorem necessary_not_sufficient :
  (∀ n, b n = n) →
  (IsGeometric ratio →
    ∀ k : ℝ, k ≠ 0 →
      IsGeometric (λ n => (a n) / (k * n))) ∧
  ¬(IsGeometric ratio →
    ∀ k : ℝ, k ≠ 0 →
      IsGeometric (λ n => (a n) / (k * n)) →
        ∀ n, b n = n) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_not_sufficient_l5_508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_elliptical_cone_volume_l5_543

/-- Volume of an elliptical cone -/
noncomputable def volume_elliptical_cone (a b h : ℝ) : ℝ := (1/3) * Real.pi * a * b * h

/-- The problem statement -/
theorem elliptical_cone_volume :
  let a : ℝ := 4  -- semi-major axis
  let b : ℝ := 3  -- semi-minor axis
  let h : ℝ := 6  -- height
  volume_elliptical_cone a b h = 24 * Real.pi :=
by
  -- Unfold the definition and simplify
  unfold volume_elliptical_cone
  simp [Real.pi]
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_elliptical_cone_volume_l5_543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_decomposition_and_nonexistence_l5_534

open Matrix Real

-- Define the 2x2 real matrix type
abbrev Mat2 := Matrix (Fin 2) (Fin 2) ℝ

-- Define the matrix [0 1; 1 0]
noncomputable def special_matrix : Mat2 := !![0, 1; 1, 0]

theorem matrix_decomposition_and_nonexistence :
  (∀ A : Mat2, ∃ B C : Mat2, A = B • B + C • C) ∧
  (¬ ∃ B C : Mat2, special_matrix = B • B + C • C ∧ B • C = C • B) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_decomposition_and_nonexistence_l5_534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_inverse_proof_l5_565

theorem matrix_inverse_proof : 
  let N : Matrix (Fin 3) (Fin 3) ℚ := !![2, 5/14, 0; 3/7, 1, 0; 0, 0, 1/2]
  let A : Matrix (Fin 3) (Fin 3) ℚ := !![-4, 5, 0; 6, -8, 0; 0, 0, 2]
  N * A = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_inverse_proof_l5_565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_average_score_l5_513

theorem cricket_average_score 
  (avg_10 : ℝ) 
  (avg_25 : ℝ) 
  (h1 : avg_10 = 60) 
  (h2 : avg_25 = 66) : 
  (avg_25 * 25 - avg_10 * 10) / 15 = 70 := by
  have total_10 : ℝ := avg_10 * 10
  have total_25 : ℝ := avg_25 * 25
  have total_15 : ℝ := total_25 - total_10
  have avg_15 : ℝ := total_15 / 15
  
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_average_score_l5_513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l5_563

/-- Given a hyperbola and a line that intersects it, prove that the eccentricity of the hyperbola is greater than √10. -/
theorem hyperbola_eccentricity_range (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ∧ y = 3*x) →
  Real.sqrt (1 + (b/a)^2) > Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l5_563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_canal_cross_section_area_l5_594

/-- The area of a trapezoidal cross-section of a canal -/
noncomputable def canal_area (top_width bottom_width depth : ℝ) : ℝ :=
  (1 / 2) * (top_width + bottom_width) * depth

/-- Theorem: The area of a trapezoidal cross-section of a canal with
    top width 6 m, bottom width 4 m, and depth 257.25 m is 1286.25 square meters -/
theorem canal_cross_section_area :
  canal_area 6 4 257.25 = 1286.25 := by
  -- Unfold the definition of canal_area
  unfold canal_area
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num
  -- QED
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_canal_cross_section_area_l5_594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_bound_difference_l5_572

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_differentiable : Differentiable ℝ f
axiom f_zero : f 0 = 0
axiom f_one : f 1 = 1
axiom f_derivative_bound : ∀ x : ℝ, |deriv f x| ≤ 2

-- Define the integral bounds
noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

-- Define the integral bounds property
axiom integral_bounds : Set.Ioo a b = {y | ∃ g : ℝ → ℝ, (∀ x, f x = g x) ∧ y = ∫ x in (0:ℝ)..1, g x}

-- State the theorem
theorem integral_bound_difference : b - a = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_bound_difference_l5_572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_ellipse_l5_574

/-- The circle in the problem -/
def problem_circle (x y : ℝ) : Prop := x^2 + (y - 6)^2 = 2

/-- The ellipse in the problem -/
def problem_ellipse (x y : ℝ) : Prop := x^2 / 10 + y^2 = 1

/-- The distance between two points -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

theorem max_distance_circle_ellipse :
  ∃ (x1 y1 x2 y2 : ℝ),
    problem_circle x1 y1 ∧ problem_ellipse x2 y2 ∧
    (∀ (a b c d : ℝ), problem_circle a b → problem_ellipse c d →
      distance x1 y1 x2 y2 ≥ distance a b c d) ∧
    distance x1 y1 x2 y2 = 6 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_ellipse_l5_574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_temperature_decrease_time_l5_558

def temperature_problem (initial_temp final_temp decrease_rate : ℚ) : ℚ :=
  (final_temp - initial_temp) / (-decrease_rate)

theorem temperature_decrease_time :
  temperature_problem (-5) (-25) 5 = 4 := by
  -- Unfold the definition of temperature_problem
  unfold temperature_problem
  -- Simplify the arithmetic
  simp [sub_eq_add_neg, neg_div]
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_temperature_decrease_time_l5_558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_polar_curve_l5_538

/-- The length of the arc of the curve r = a cos³(φ/3) --/
noncomputable def arcLength (a : ℝ) : ℝ := (3 * Real.pi * a) / 2

theorem arc_length_polar_curve (a : ℝ) (ha : a > 0) :
  2 * ∫ φ in (0)..(3 * Real.pi / 2),
    Real.sqrt ((a * Real.cos (φ / 3) ^ 3) ^ 2 +
      (- a * Real.cos (φ / 3) ^ 2 * Real.sin (φ / 3)) ^ 2) = arcLength a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_polar_curve_l5_538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PF1F2_l5_530

/-- The hyperbola Γ defined by x^2 - y^2/3 = 1 -/
def Γ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 - p.2^2/3 = 1}

/-- The foci of the hyperbola Γ -/
def foci : (ℝ × ℝ) × (ℝ × ℝ) :=
  ((-2, 0), (2, 0))

/-- A point P on Γ such that |OP| = 2 -/
noncomputable def P : ℝ × ℝ :=
  (Real.sqrt 7 / 2, 3 / 2)

/-- The area of triangle PF₁F₂ is 3 -/
theorem area_of_triangle_PF1F2 :
  let F₁ := foci.1
  let F₂ := foci.2
  P ∈ Γ ∧ Real.sqrt (P.1^2 + P.2^2) = 2 →
  abs ((P.1 * (F₁.2 - F₂.2) + F₁.1 * (F₂.2 - P.2) + F₂.1 * (P.2 - F₁.2)) / 2) = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PF1F2_l5_530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l5_576

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 6*x - 8*y = 0

-- Define the line equation
def line_equation (x y : ℝ) : Prop := y = -x

-- Define the region
def region (x y : ℝ) : Prop := circle_equation x y ∧ y ≥ -x

-- Theorem statement
theorem area_of_region : 
  ∃ A : ℝ, A = 12.5 * Real.pi ∧ 
  (∀ ε > 0, ∃ n : ℕ, ∀ partition : List (ℝ × ℝ),
    (∀ p ∈ partition, region p.1 p.2) →
    partition.length = n →
    |A - (partition.map (λ p => p.1 * p.2)).sum| < ε) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l5_576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cardinality_of_B_l5_547

def A : Finset ℕ := {1, 2, 3, 4, 5}

def B : Finset (ℕ × ℕ) := A.product A |>.filter (fun p => (p.1 - p.2) ∈ A)

theorem cardinality_of_B : Finset.card B = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cardinality_of_B_l5_547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2014_equals_one_l5_551

def sequence_a : ℕ → ℚ
  | 0 => 0  -- Add this case for n = 0
  | 1 => 0
  | n + 2 => (sequence_a (n + 1) - 2) / ((5 * sequence_a (n + 1) / 4) - 2)

theorem a_2014_equals_one : sequence_a 2014 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2014_equals_one_l5_551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_workers_needed_l5_545

/-- Represents the building project with its parameters and completion status -/
structure BuildingProject where
  totalDays : ℕ
  completedDays : ℕ
  initialWorkers : ℕ
  completedPercentage : ℚ
  deriving Repr

/-- Calculates the minimum number of workers needed to complete the project on time -/
def minWorkersNeeded (project : BuildingProject) : ℕ :=
  sorry

/-- Theorem stating that for the given project parameters, 5 workers are needed -/
theorem five_workers_needed (project : BuildingProject) 
  (h1 : project.totalDays = 40)
  (h2 : project.completedDays = 10)
  (h3 : project.initialWorkers = 10)
  (h4 : project.completedPercentage = 40)
  : minWorkersNeeded project = 5 := by
  sorry

#eval BuildingProject.mk 40 10 10 40

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_workers_needed_l5_545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_liter_measurement_possible_l5_539

/-- Represents the state of the two buckets -/
structure BucketState where
  big : ℕ  -- Amount of water in the 10-liter bucket
  small : ℕ -- Amount of water in the 6-liter bucket

/-- Represents the possible actions that can be performed -/
inductive BucketAction
  | FillBig
  | FillSmall
  | EmptyBig
  | EmptySmall
  | PourBigToSmall
  | PourSmallToBig

/-- Applies an action to a given bucket state -/
def applyAction (state : BucketState) (action : BucketAction) : BucketState :=
  match action with
  | BucketAction.FillBig => { big := 10, small := state.small }
  | BucketAction.FillSmall => { big := state.big, small := 6 }
  | BucketAction.EmptyBig => { big := 0, small := state.small }
  | BucketAction.EmptySmall => { big := state.big, small := 0 }
  | BucketAction.PourBigToSmall =>
      let amount := min (10 - state.big) state.small
      { big := state.big + amount, small := state.small - amount }
  | BucketAction.PourSmallToBig =>
      let amount := min state.small (10 - state.big)
      { big := state.big + amount, small := state.small - amount }

/-- Checks if the given sequence of actions results in 8 liters in the big bucket -/
def isValidSolution (actions : List BucketAction) : Prop :=
  let finalState := actions.foldl applyAction { big := 0, small := 0 }
  finalState.big = 8

/-- Theorem stating that there exists a sequence of actions that results in 8 liters in the big bucket -/
theorem eight_liter_measurement_possible : ∃ (actions : List BucketAction), isValidSolution actions := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_liter_measurement_possible_l5_539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phase_shift_of_f_l5_522

noncomputable def f (x : ℝ) := 3 * Real.sin (4 * x - Real.pi / 2)

theorem phase_shift_of_f :
  ∃ (shift : ℝ), ∀ (x : ℝ), f (x + shift) = 3 * Real.sin (4 * x) ∧ shift = Real.pi / 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phase_shift_of_f_l5_522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l5_583

noncomputable section

open Real

theorem triangle_property (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  (a + c) * (a - c) + b * (b - a) = 0 →
  (C = π / 3) ∧ (∃ (x : ℝ), x = sin A + sin B ∧ x ≤ sqrt 3 ∧ ∀ (y : ℝ), y = sin A + sin B → y ≤ x) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l5_583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_property_l5_505

def has_sum_property (S : Set ℕ) : Prop :=
  ∃ a b c, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a + b = c

def partition_satisfies_property (T : Set ℕ) : Prop :=
  ∀ A B : Set ℕ, A ∪ B = T → A ∩ B = ∅ → 
    has_sum_property A ∨ has_sum_property B

def T (n : ℕ) : Set ℕ := {i | 2 ≤ i ∧ i ≤ n}

theorem smallest_n_with_property :
  (∀ n < 7, ¬ partition_satisfies_property (T n)) ∧
  partition_satisfies_property (T 7) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_property_l5_505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_path_length_l5_557

noncomputable section

/-- The length of the circular path around the pond -/
noncomputable def path_length : ℝ := 3000

/-- Buratino's speed -/
noncomputable def speed_buratino : ℝ := 1

/-- Duremar's speed -/
noncomputable def speed_duremar : ℝ := 1 / 3

/-- Karabas-Barabas's speed -/
noncomputable def speed_karabas : ℝ := 1 / 4

/-- Distance between Buratino's meetings with Duremar and Karabas-Barabas -/
noncomputable def distance_between_meetings : ℝ := 150

theorem circular_path_length :
  speed_buratino = 3 * speed_duremar ∧
  speed_buratino = 4 * speed_karabas ∧
  distance_between_meetings = 150 →
  path_length = 3000 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_path_length_l5_557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unlimited_plane_symmetry_half_plane_symmetry_quadrant_symmetry_l5_555

-- Define the types for our geometric objects
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

structure Plane where
  points : Set Point
  lines : Set Line

structure HalfPlane where
  points : Set Point
  boundary : Line

structure Quadrant where
  points : Set Point
  xAxis : Line
  yAxis : Line

-- Define the concept of symmetry
def isAxisOfSymmetry (p : Plane) (l : Line) : Prop :=
  ∀ (point : Point), point ∈ p.points → (reflect point l) ∈ p.points
where
  reflect : Point → Line → Point := sorry

def isCenterOfSymmetry (p : Plane) (c : Point) : Prop :=
  ∀ (point : Point), point ∈ p.points → (centralReflect point c) ∈ p.points
where
  centralReflect : Point → Point → Point := sorry

def isPerpendicularTo (l1 l2 : Line) : Prop := sorry

-- State the theorems
theorem unlimited_plane_symmetry (p : Plane) :
  (∀ (l : Line), l ∈ p.lines → isAxisOfSymmetry p l) ∧
  (∀ (c : Point), c ∈ p.points → isCenterOfSymmetry p c) := by sorry

theorem half_plane_symmetry (hp : HalfPlane) (l : Line) :
  isPerpendicularTo l hp.boundary → isAxisOfSymmetry ⟨hp.points, {l, hp.boundary}⟩ l := by sorry

theorem quadrant_symmetry (q : Quadrant) :
  isAxisOfSymmetry ⟨q.points, {q.xAxis, q.yAxis}⟩ q.xAxis ∧
  isAxisOfSymmetry ⟨q.points, {q.xAxis, q.yAxis}⟩ q.yAxis := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unlimited_plane_symmetry_half_plane_symmetry_quadrant_symmetry_l5_555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_traces_ellipse_l5_581

/-- A complex number tracing a circle centered at the origin with radius 3 -/
def CircleTrace (z : ℂ) : Prop := Complex.abs z = 3

/-- The transformation applied to z -/
noncomputable def Transform (z : ℂ) : ℂ := z + (1 / z) + 2

/-- Definition of an ellipse in the complex plane -/
def IsEllipse (f : ℂ → ℂ) : Prop :=
  ∃ a b c d e : ℝ, a > 0 ∧ b > 0 ∧
    ∀ z : ℂ, (Complex.re (f z) - c)^2 / a + (Complex.im (f z) - d)^2 / b = e

/-- Theorem: The transformation traces an ellipse -/
theorem transform_traces_ellipse :
  ∀ z : ℂ, CircleTrace z → IsEllipse Transform := by
  sorry

#check transform_traces_ellipse

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_traces_ellipse_l5_581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coat_price_calculation_l5_570

def original_price : ℝ := 200
def initial_discount_rate : ℝ := 0.30
def additional_discount : ℝ := 10
def sales_tax_rate : ℝ := 0.10

theorem coat_price_calculation : 
  (original_price * (1 - initial_discount_rate) - additional_discount) * (1 + sales_tax_rate) = 143 := by
  -- Calculation steps
  have price_after_initial_discount := original_price * (1 - initial_discount_rate)
  have price_after_additional_discount := price_after_initial_discount - additional_discount
  have final_price := price_after_additional_discount * (1 + sales_tax_rate)
  -- Proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coat_price_calculation_l5_570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_result_l5_542

/-- Represents a complex number -/
structure MyComplex where
  re : ℝ
  im : ℝ

/-- Represents a transformation on complex numbers -/
structure Transformation where
  apply : MyComplex → MyComplex

/-- 60° counter-clockwise rotation around the origin -/
noncomputable def rotation60 : Transformation := {
  apply := λ z ↦ { 
    re := z.re * (1/2) - z.im * (Real.sqrt 3 / 2),
    im := z.re * (Real.sqrt 3 / 2) + z.im * (1/2)
  }
}

/-- Dilation with scale factor 2 centered at the origin -/
def dilation2 : Transformation := {
  apply := λ z ↦ { re := 2 * z.re, im := 2 * z.im }
}

/-- Composition of two transformations -/
def compose (t1 t2 : Transformation) : Transformation := {
  apply := λ z ↦ t2.apply (t1.apply z)
}

theorem transformation_result : 
  let initial : MyComplex := { re := -1, im := -2 }
  let final : MyComplex := { re := 2 * Real.sqrt 3 - 1, im := -(2 + Real.sqrt 3) }
  let combined_transformation := compose rotation60 dilation2
  combined_transformation.apply initial = final := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_result_l5_542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_properties_l5_554

/-- Given an expression (√x + 1/(3x²))^n, this theorem proves that if the ratio of
    the binomial coefficient of the fifth term to that of the third term is 14:3,
    then n = 10 and the constant term in the expansion is 5. -/
theorem expansion_properties (x : ℝ) (n : ℕ) :
  (Nat.choose n 4 : ℚ) / (Nat.choose n 2 : ℚ) = 14 / 3 →
  n = 10 ∧ 
  (∃ (r : ℕ), r = 2 ∧ (Nat.choose 10 r : ℚ) * (1 / 3 ^ r) = 5) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_properties_l5_554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_implies_equality_l5_531

theorem power_equality_implies_equality (a b x : ℕ) 
  (ha : a > 0) (hb : b > 0) (hx : x > 0)
  (h : (x : ℝ)^(a+b) = (a : ℝ)^b * b) : a = x ∧ b = x^x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_implies_equality_l5_531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_surface_area_approx_l5_571

/-- The total surface area of a cone with given radius and slant height -/
noncomputable def total_surface_area (radius : ℝ) (slant_height_feet : ℝ) : ℝ :=
  let slant_height_meters := slant_height_feet * 0.3048
  Real.pi * radius * (slant_height_meters + radius)

/-- Theorem stating the total surface area of a cone with specific measurements -/
theorem cone_surface_area_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ 
  |total_surface_area 28 98.5 - 5107.876| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_surface_area_approx_l5_571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisors_of_50_factorial_l5_580

/-- The number of prime divisors of 50! -/
def num_prime_divisors_50_factorial : ℕ := 15

/-- 50 factorial -/
def factorial_50 : ℕ := Nat.factorial 50

theorem prime_divisors_of_50_factorial :
  (Finset.filter (fun p ↦ Nat.Prime p ∧ factorial_50 % p = 0) (Finset.range (factorial_50 + 1))).card = num_prime_divisors_50_factorial :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisors_of_50_factorial_l5_580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_solutions_l5_504

open Real

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := -3 * cos (Real.pi * x / 2)

-- Define the domain
def D : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}

-- Define the set of solutions
def S : Set ℝ := {x ∈ D | g (g (g x)) = g x}

-- Theorem statement
theorem three_solutions : ∃ (f : Finset ℝ), f.card = 3 ∧ ∀ x ∈ f, x ∈ S := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_solutions_l5_504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_comparison_l5_579

/-- Represents a number in a given base as a list of digits -/
def BaseNumber (base : ℕ) := List ℕ

/-- Converts a BaseNumber to its decimal representation -/
def to_decimal (base : ℕ) (num : BaseNumber base) : ℕ := sorry

/-- Theorem: For positive integers a, b, n > 1, and non-zero digits x_0, ..., x_n,
    A_{n-1}/A_{n} < B_{n-1}/B_{n} if and only if a > b -/
theorem base_comparison 
  (a b n : ℕ) 
  (ha : a > 1) (hb : b > 1) (hn : n > 1)
  (digits : List ℕ)
  (hdigits : digits.length = n + 1)
  (hx_n : digits.head? = some (digits.head!))
  (hx_n_1 : digits.tail.head? = some (digits.tail.head!))
  (hx_n_nonzero : digits.head! ≠ 0)
  (hx_n_1_nonzero : digits.tail.head! ≠ 0)
  (A_n_1 : BaseNumber a := digits.tail)
  (A_n : BaseNumber a := digits)
  (B_n_1 : BaseNumber b := digits.tail)
  (B_n : BaseNumber b := digits)
  : (to_decimal a A_n_1 * to_decimal b B_n < to_decimal a A_n * to_decimal b B_n_1) ↔ a > b := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_comparison_l5_579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_surface_area_volume_l5_589

theorem prism_surface_area_volume (x : ℝ) (x_pos : x > 0) : 
  (2 * (Real.log x / Real.log 5 * Real.log x / Real.log 6 + 
        Real.log x / Real.log 5 * Real.log x / Real.log 7 + 
        Real.log x / Real.log 6 * Real.log x / Real.log 7) = 
   2 * (Real.log x / Real.log 5 * Real.log x / Real.log 6 * Real.log x / Real.log 7)) → 
  x = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_surface_area_volume_l5_589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ticket_cost_difference_l5_526

/-- Proves that the difference between the total cost of adult tickets and children's tickets is $50 -/
theorem ticket_cost_difference 
  (num_adults : ℕ) (num_children : ℕ) (adult_ticket_price : ℕ) (child_ticket_price : ℕ) : 
  num_adults = 9 →
  num_children = 7 →
  adult_ticket_price = 11 →
  child_ticket_price = 7 →
  num_adults * adult_ticket_price - num_children * child_ticket_price = 50 := by
  sorry

#check ticket_cost_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ticket_cost_difference_l5_526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_equation_l5_535

-- Define the custom operation
noncomputable def circle_slash (a b : ℝ) : ℝ := (Real.sqrt (3 * a + b)) ^ 3

-- Theorem statement
theorem solve_equation (x : ℝ) : circle_slash 7 x = 64 → x = -5 := by
  intro h
  -- The proof steps would go here
  sorry

#check solve_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_equation_l5_535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_plus_beta_l5_533

theorem cos_alpha_plus_beta (α β : Real) : 
  (0 < α) → (α < π / 2) →  -- α is in the first quadrant
  (π < β) → (β < 3 * π / 2) →  -- β is in the third quadrant
  Real.sin (α + π / 3) = 3 / 5 →
  Real.cos (β - π / 3) = -12 / 13 →
  (Real.cos (α + β) = 33 / 65 ∨ Real.cos (α + β) = 63 / 65) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_plus_beta_l5_533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_arrangement_l5_544

/-- Represents the gender of a schoolchild -/
inductive Gender
| Boy
| Girl

/-- Represents a seating arrangement of schoolchildren on a bench -/
def Arrangement := List Gender

/-- Checks if there's an even number of schoolchildren between every two boys -/
def evenBetweenBoys (arr : Arrangement) : Prop :=
  ∀ i j, i < j → arr.get? i = some Gender.Boy → arr.get? j = some Gender.Boy →
    ∃ k, Even k ∧ j - i - 1 = k

/-- Checks if there's an odd number of schoolchildren between every two girls -/
def oddBetweenGirls (arr : Arrangement) : Prop :=
  ∀ i j, i < j → arr.get? i = some Gender.Girl → arr.get? j = some Gender.Girl →
    ∃ k, Odd k ∧ j - i - 1 = k

/-- Theorem: It's impossible to arrange 10 schoolchildren satisfying both conditions -/
theorem no_valid_arrangement :
  ¬∃ (arr : Arrangement), arr.length = 10 ∧ evenBetweenBoys arr ∧ oddBetweenGirls arr := by
  sorry

#check no_valid_arrangement

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_arrangement_l5_544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_special_property_l5_561

/-- Given a triangle ABC with side lengths a, b, c and angles A, B, C,
    prove that if a^2 + c^2 = b^2 + √2ac, then angle B is π/4 and
    the maximum value of √2cos(A) + cos(C) is 1 -/
theorem triangle_special_property (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = Real.pi →
  a^2 + c^2 = b^2 + Real.sqrt 2 * a * c →
  B = Real.pi / 4 ∧ (∃ (x : ℝ), Real.sqrt 2 * Real.cos A + Real.cos C ≤ x ∧ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_special_property_l5_561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_inequality_l5_585

theorem absolute_value_inequality (x : ℝ) : 2 ≤ |x - 3| ∧ |x - 3| ≤ 8 ↔ x ∈ Set.Icc (-5) 1 ∪ Set.Icc 5 11 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_inequality_l5_585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_exponential_equation_l5_510

theorem unique_solution_exponential_equation :
  ∃! x : ℚ, (10 : ℝ)^(x : ℝ) * (1000 : ℝ)^((2*x) : ℝ) = (10000 : ℝ)^(3 : ℝ) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_exponential_equation_l5_510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_intersection_and_angle_l5_597

-- Define the lines
def line_n (x y : ℝ) : Prop := y - 2*x = 1
def line_p (x y : ℝ) : Prop := 3*x + y = 6

-- Define the intersection point
noncomputable def intersection_point : ℝ × ℝ := (1, 3)

-- Define the angle between the lines
noncomputable def angle_between_lines : ℝ := Real.pi / 4

-- Theorem statement
theorem lines_intersection_and_angle :
  (∀ x y : ℝ, line_n x y ∧ line_p x y ↔ (x, y) = intersection_point) ∧
  (let m₁ := (2 : ℝ); let m₂ := (-3 : ℝ); Real.arctan ((m₁ - m₂) / (1 + m₁ * m₂)) = angle_between_lines) := by
  sorry

#check lines_intersection_and_angle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_intersection_and_angle_l5_597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_average_theorem_l5_588

/-- The overall class average on a test, given the performance of three groups -/
def class_average (group1_percent : ℝ) (group1_score : ℝ) 
                  (group2_percent : ℝ) (group2_score : ℝ) 
                  (group3_percent : ℝ) (group3_score : ℝ) : ℝ :=
  group1_percent * group1_score + group2_percent * group2_score + group3_percent * group3_score

/-- Rounds a real number to the nearest integer -/
noncomputable def round_to_nearest (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

theorem class_average_theorem :
  let group1_percent : ℝ := 0.15
  let group1_score : ℝ := 100
  let group2_percent : ℝ := 0.50
  let group2_score : ℝ := 78
  let group3_percent : ℝ := 1 - group1_percent - group2_percent
  let group3_score : ℝ := 63
  round_to_nearest (class_average group1_percent group1_score 
                                  group2_percent group2_score 
                                  group3_percent group3_score) = 76 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_average_theorem_l5_588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_theorem_l5_599

/-- Given a parabola y² = 12x with focus F(3, 0), prove that the circle with center F
    and diameter AB, where A and B are the intersection points of the parabola with a line
    perpendicular to the x-axis passing through F, has the equation (x-3)² + y² = 36 -/
theorem parabola_circle_theorem (x y : ℝ) :
  let parabola := fun x y => y^2 = 12*x
  let focus := (3, 0)
  let perpendicular_line := fun x => x = 3
  let A := (3, 6)
  let B := (3, -6)
  parabola x y ∧ perpendicular_line x →
  (x - 3)^2 + y^2 = 36 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_theorem_l5_599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_odd_integers_l5_567

theorem max_ratio_odd_integers (x y : ℕ) : 
  x % 2 = 1 → 
  y % 2 = 1 → 
  x > 0 → 
  y > 0 → 
  (x + y) / 2 = 55 → 
  ∀ a b : ℕ, a % 2 = 1 → b % 2 = 1 → a > 0 → b > 0 → (a + b) / 2 = 55 → 
  (a : ℚ) / b ≤ (x : ℚ) / y →
  (x : ℚ) / y ≤ 9 := by
  sorry

#check max_ratio_odd_integers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_odd_integers_l5_567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_op_plus_ob_l5_550

open Real

/-- The origin point O -/
def O : ℝ × ℝ := (0, 0)

/-- Point A -/
def A : ℝ × ℝ := (2, 0)

/-- Point B -/
def B : ℝ × ℝ := (0, 2)

/-- Distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Vector addition in ℝ² -/
def vectorAdd (p q : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + q.1, p.2 + q.2)

/-- Magnitude of a vector in ℝ² -/
noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  distance v O

/-- The set of all points P satisfying |AP| = 1 -/
def P : Set (ℝ × ℝ) :=
  {p | distance p A = 1}

/-- Theorem: The maximum value of |OP + OB| for P satisfying |AP| = 1 is 2√2 + 1 -/
theorem max_op_plus_ob : 
  ∃ (max : ℝ), max = 2 * Real.sqrt 2 + 1 ∧ 
  ∀ (p : ℝ × ℝ), p ∈ P → magnitude (vectorAdd p B) ≤ max := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_op_plus_ob_l5_550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_is_13_l5_598

def S : Set ℂ := {z | ∃ x y : ℝ, z = x + y * Complex.I ∧ 1/2 ≤ x ∧ x ≤ Real.sqrt 2 / 2}

theorem smallest_m_is_13 : 
  (∃ m : ℕ+, (∀ n : ℕ+, n ≥ m → ∃ z ∈ S, z^(n:ℂ) = 1) ∧ 
   (∀ k : ℕ+, k < m → ¬(∀ n : ℕ+, n ≥ k → ∃ z ∈ S, z^(n:ℂ) = 1))) ∧ 
  (∀ n : ℕ+, n ≥ 13 → ∃ z ∈ S, z^(n:ℂ) = 1) ∧
  ¬(∀ n : ℕ+, n ≥ 12 → ∃ z ∈ S, z^(n:ℂ) = 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_is_13_l5_598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_squared_l5_511

-- Define f as a function from ℝ to ℝ
def f : ℝ → ℝ := sorry

-- Define the domain of a function
def domain (f : ℝ → ℝ) : Set ℝ := {x | ∃ y, f x = y}

theorem domain_of_f_squared :
  (domain f = Set.Icc 1 4) →
  (domain (fun x ↦ f (x^2)) = Set.Icc (-2) (-1) ∪ Set.Icc 1 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_squared_l5_511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_A_travel_time_is_8_4_l5_529

/-- Represents the travel scenario of two cars --/
structure TravelScenario where
  speedRatioA : ℚ
  speedRatioB : ℚ
  initialDelayHours : ℚ
  speedIncreaseTime : ℚ
  speedIncreaseFactor : ℚ

/-- Calculates the travel time of Car A given a travel scenario --/
noncomputable def carATravelTime (scenario : TravelScenario) : ℚ :=
  let t := scenario.speedIncreaseTime
  let x := (scenario.initialDelayHours * scenario.speedRatioB + t * scenario.speedRatioA) /
           (scenario.speedRatioB - scenario.speedRatioA * (scenario.speedIncreaseFactor - 1))
  x + scenario.initialDelayHours

/-- The main theorem stating that Car A's travel time is 8.4 hours --/
theorem car_A_travel_time_is_8_4 :
  let scenario : TravelScenario := {
    speedRatioA := 4
    speedRatioB := 3
    initialDelayHours := 6
    speedIncreaseTime := 6
    speedIncreaseFactor := 2
  }
  carATravelTime scenario = 42/5 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_A_travel_time_is_8_4_l5_529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_garden_length_l5_587

/-- Represents a rectangular garden with a fence on three sides and a wall on the fourth side. -/
structure Garden where
  fence_length : ℝ
  perpendicular_side : ℝ

/-- Calculates the area of the garden. -/
noncomputable def garden_area (g : Garden) : ℝ :=
  g.perpendicular_side * (g.fence_length - 2 * g.perpendicular_side)

/-- Finds the maximum area of the garden. -/
noncomputable def max_area (g : Garden) : ℝ :=
  (g.fence_length / 4) * (g.fence_length / 2)

/-- Theorem stating that the length of the side parallel to the wall that maximizes the garden area is 75 feet. -/
theorem optimal_garden_length (g : Garden) (h : g.fence_length = 150) :
    g.fence_length - 2 * (g.fence_length / 4) = 75 := by
  rw [h]
  norm_num

#check optimal_garden_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_garden_length_l5_587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conference_solution_l5_578

/-- Represents the number of clinics in the city -/
def n : ℕ := sorry

/-- Represents the number of doctors invited to the conference -/
def N : ℕ := sorry

/-- Each clinic invited 4 doctors -/
axiom four_doctors_per_clinic : 4 * n = 2 * N

/-- Each combination of two clinics had exactly one representative -/
axiom one_rep_per_combination : N = n * (n - 1) / 2

/-- The number of clinics is 5 and the number of doctors is 10 -/
theorem conference_solution : n = 5 ∧ N = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conference_solution_l5_578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_theorem_l5_527

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define external tangency
def externally_tangent (c1 c2 : Circle) : Prop :=
  distance c1.center c2.center = c1.radius + c2.radius

-- Define internal tangency
def internally_tangent (c1 c2 : Circle) : Prop :=
  distance c1.center c2.center = |c1.radius - c2.radius|

theorem circle_tangency_theorem (c1 c2 : Circle) :
  (externally_tangent c1 c2 → distance c1.center c2.center = c1.radius + c2.radius) ∧
  (internally_tangent c1 c2 → distance c1.center c2.center = |c1.radius - c2.radius|) ∧
  (distance c1.center c2.center = c1.radius + c2.radius → externally_tangent c1 c2) ∧
  (distance c1.center c2.center = |c1.radius - c2.radius| → internally_tangent c1 c2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_theorem_l5_527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radio_cost_price_l5_541

/-- The cost price of a radio given selling price, overhead expenses, and profit percentage. -/
noncomputable def cost_price (selling_price : ℝ) (overhead : ℝ) (profit_percent : ℝ) : ℝ :=
  (selling_price - overhead) / (1 + profit_percent / 100)

/-- Theorem stating that the cost price of the radio is approximately 228.41 -/
theorem radio_cost_price :
  let selling_price : ℝ := 350
  let overhead : ℝ := 15
  let profit_percent : ℝ := 45.833333333333314
  abs (cost_price selling_price overhead profit_percent - 228.41) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_radio_cost_price_l5_541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_F_l5_525

-- Define the function f on the interval (0,1)
noncomputable def f : {x : ℝ | 0 < x ∧ x < 1} → ℝ := sorry

-- Define the constant a
def a : ℝ := sorry

-- Assumptions on a
axiom a_bounds : -1/2 < a ∧ a < 1/2
axiom a_nonzero : a ≠ 0

-- Define F(x) in terms of f
noncomputable def F (x : ℝ) : ℝ := 
  f ⟨x + a, sorry⟩ + f ⟨x - a, sorry⟩

-- Theorem stating the domain of F
theorem domain_of_F :
  (∀ x : ℝ, -1/2 < a → a < 0 → (-a < x ∧ x < 1 + a) → F x = F x) ∧
  (∀ x : ℝ, 0 < a → a < 1/2 → (a < x ∧ x < 1 - a) → F x = F x) :=
by
  constructor
  · intro x h1 h2 h3
    rfl
  · intro x h1 h2 h3
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_F_l5_525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_theorem_l5_532

/-- Two lines in a plane -/
structure Lines where
  a : ℝ
  l₁ : ℝ → ℝ → Prop
  l₂ : ℝ → ℝ → Prop
  h₁ : ∀ x y, l₁ x y ↔ 2 * a * x + y - 1 = 0
  h₂ : ∀ x y, l₂ x y ↔ a * x + (a - 1) * y + 1 = 0

/-- Perpendicularity of two lines -/
def perpendicular (L : Lines) : Prop :=
  (2 * L.a) * (L.a / (L.a - 1)) = 1

/-- Parallelism of two lines -/
def parallel (L : Lines) : Prop :=
  2 * L.a = L.a / (L.a - 1)

/-- Distance between two parallel lines -/
noncomputable def distance (L : Lines) : ℝ :=
  |3 + 1| / Real.sqrt (3^2 + (-1)^2)

theorem lines_theorem (L : Lines) :
  (perpendicular L → L.a = -1 ∨ L.a = 1/2) ∧
  (parallel L ∧ L.a ≠ 0 → L.a = 3/2) ∧
  (parallel L ∧ L.a = 3/2 → distance L = 2 * Real.sqrt 10 / 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_theorem_l5_532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_eq_12_l5_521

def digits : List Nat := [2, 4, 7, 8]

def is_valid_two_digit_number (n : Nat) : Bool :=
  n ≥ 10 && n < 100 &&
  digits.any (fun d1 =>
    digits.any (fun d2 =>
      d1 ≠ d2 && n = d1 * 10 + d2))

def count_valid_numbers : Nat :=
  (List.filter is_valid_two_digit_number (List.range 100)).length

theorem count_valid_numbers_eq_12 : count_valid_numbers = 12 := by
  sorry

#eval count_valid_numbers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_eq_12_l5_521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lizzy_shipping_cost_l5_564

/-- Given the total weight of fish, weight per crate, and total shipping cost,
    calculate the shipping cost per crate. -/
noncomputable def shipping_cost_per_crate (total_weight : ℝ) (weight_per_crate : ℝ) (total_cost : ℝ) : ℝ :=
  total_cost / (total_weight / weight_per_crate)

/-- Theorem: The shipping cost per crate for Lizzy's fish shipment is $1.50. -/
theorem lizzy_shipping_cost :
  shipping_cost_per_crate 540 30 27 = 1.50 := by
  -- Unfold the definition of shipping_cost_per_crate
  unfold shipping_cost_per_crate
  -- Simplify the expression
  simp
  -- The proof is completed with 'sorry' as requested
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lizzy_shipping_cost_l5_564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_sum_theorem_l5_514

/-- Represents a cube with integers on its faces -/
structure Cube where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  d : ℕ+
  e : ℕ+
  f : ℕ+

/-- Sum of vertex labels for a cube -/
def vertexSum (cube : Cube) : ℚ :=
  (cube.a.val * cube.b.val * cube.c.val +
   cube.a.val * cube.e.val * cube.c.val +
   cube.a.val * cube.b.val * cube.f.val +
   cube.a.val * cube.e.val * cube.f.val +
   cube.d.val * cube.b.val * cube.c.val +
   cube.d.val * cube.e.val * cube.c.val +
   cube.d.val * cube.b.val * cube.f.val +
   cube.d.val * cube.e.val * cube.f.val) / 2

/-- Sum of face values for a cube -/
def faceSum (cube : Cube) : ℕ :=
  cube.a.val + cube.b.val + cube.c.val + cube.d.val + cube.e.val + cube.f.val

/-- Theorem stating that if the vertex sum is 1083, the face sum is 60 -/
theorem cube_sum_theorem (cube : Cube) :
  vertexSum cube = 1083 → faceSum cube = 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_sum_theorem_l5_514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l5_595

open Real

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The sine law for a triangle -/
axiom sine_law (t : Triangle) : t.a / (sin t.A) = t.b / (sin t.B)

/-- The cosine law for a triangle -/
axiom cosine_law (t : Triangle) : t.c^2 = t.a^2 + t.b^2 - 2 * t.a * t.b * cos t.C

/-- The area formula for a triangle -/
noncomputable def triangle_area (t : Triangle) : ℝ := 1/2 * t.a * t.c * sin t.B

theorem triangle_properties (t : Triangle) 
  (h1 : (sin t.C - sin t.A) * t.a = (sin t.C - sin t.B) * (t.c + t.b))
  (h2 : 2 * Real.sqrt 3 / 3 = t.a / (2 * sin t.A)) -- Radius of circumscribed circle
  (h3 : t.a + t.c = 4) :
  t.B = π/3 ∧ triangle_area t = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l5_595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_op_example_l5_556

-- Define the * operation
noncomputable def star_op (a b : ℝ) : ℝ :=
  if a > b ∧ a ≠ 0 then a ^ b
  else if a < b ∧ a ≠ 0 then a ^ (-b)
  else 0  -- undefined case

-- Theorem statement
theorem star_op_example : star_op 2 (-4) * star_op (-4) (-2) = 1 := by
  -- Unfold the definition of star_op
  unfold star_op
  -- Simplify the if-then-else expressions
  simp
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_op_example_l5_556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_250_between_integers_l5_524

theorem log_250_between_integers : ∃ (a b : ℤ), 
  (Real.log 250 > a) ∧ 
  (Real.log 250 < b) ∧ 
  (b - a = 1) ∧ 
  (a + b = 5) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_250_between_integers_l5_524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_proof_l5_577

-- Define the side length of the hexagon
def side_length : ℝ := 3

-- Define the area of the hexagon
noncomputable def hexagon_area : ℝ := 6 * (Real.sqrt 3 / 4 * side_length^2)

-- Define p and q
def p : ℕ := 729
def q : ℕ := 27

-- Theorem to prove
theorem hexagon_area_proof :
  hexagon_area = Real.sqrt p + Real.sqrt q ∧ p + q = 756 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_proof_l5_577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sine_relation_l5_509

theorem cosine_sine_relation (a : ℝ) :
  (Real.cos a + Real.sin a) / (Real.cos a - Real.sin a) = 2 →
  Real.cos a ^ 2 + Real.sin a * Real.cos a = 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sine_relation_l5_509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_properties_l5_552

/-- Represents a trapezium with given dimensions -/
structure Trapezium where
  side1 : ℝ
  side2 : ℝ
  height : ℝ

/-- Calculates the area of a trapezium -/
noncomputable def area (t : Trapezium) : ℝ := (t.side1 + t.side2) * t.height / 2

/-- Theorem about the area and longer side of a specific trapezium -/
theorem trapezium_properties (t : Trapezium) 
    (h1 : t.side1 = 30)
    (h2 : t.side2 = 12)
    (h3 : t.height = 16) :
    area t = 336 ∧ max t.side1 t.side2 = 30 := by
  sorry

#check trapezium_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_properties_l5_552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isothermal_work_theorem_l5_523

/-- Represents the properties and processes of an ideal monatomic gas -/
structure IdealMonatomicGas where
  /-- The number of moles of the gas -/
  moles : ℝ
  /-- Work done during isobaric heating (in Joules) -/
  isobaric_work : ℝ
  /-- Molar heat capacity at constant pressure (in J/(mol·K)) -/
  C_p : ℝ
  /-- Universal gas constant (in J/(mol·K)) -/
  R : ℝ

/-- Theorem stating the work done during isothermal heating -/
theorem isothermal_work_theorem (gas : IdealMonatomicGas) 
  (h1 : gas.moles = 1)
  (h2 : gas.isobaric_work = 40)
  (h3 : gas.C_p = 5/2 * gas.R) :
  ∃ (isothermal_work : ℝ), isothermal_work = 100 := by
  let isobaric_heat := gas.C_p * (gas.isobaric_work / gas.R)
  let isothermal_work := isobaric_heat
  use isothermal_work
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isothermal_work_theorem_l5_523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_at_3_l5_518

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x - 1

-- Theorem statement
theorem inverse_f_at_3 (a : ℝ) :
  (f a 1 = 1) → (∃ y, f a y = 3 ∧ y = 2) :=
by
  intro h
  -- Proof steps would go here
  sorry

#check inverse_f_at_3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_at_3_l5_518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_in_first_and_third_quadrants_l5_528

/-- Inverse proportion function -/
noncomputable def inverse_proportion (k : ℝ) (x : ℝ) : ℝ := (2 - k) / x

/-- Predicate to check if a point (x, y) is in the first or third quadrant -/
def in_first_or_third_quadrant (x y : ℝ) : Prop := (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0)

/-- Theorem stating the condition for the inverse proportion function to be in the first and third quadrants -/
theorem inverse_proportion_in_first_and_third_quadrants (k : ℝ) :
  (∀ x : ℝ, x ≠ 0 → in_first_or_third_quadrant x (inverse_proportion k x)) ↔ k < 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_in_first_and_third_quadrants_l5_528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tan_half_sum_l5_575

open Real

theorem triangle_tan_half_sum (A B C : ℝ) (h : A + B + C = π) :
  tan (A/2) * tan (B/2) + tan (B/2) * tan (C/2) + tan (A/2) * tan (C/2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tan_half_sum_l5_575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l5_520

-- Define the function representing the curve
noncomputable def f (x : ℝ) : ℝ := x * (3 * Real.log x + 1)

-- Define the derivative of the function
noncomputable def f' (x : ℝ) : ℝ := 3 * Real.log x + 4

-- Theorem statement
theorem tangent_line_at_one :
  let p : ℝ × ℝ := (1, 1)
  let m : ℝ := f' p.fst
  let tangent_line (x : ℝ) : ℝ := m * (x - p.fst) + p.snd
  ∀ x, tangent_line x = 4 * x - 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l5_520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adam_score_from_judge_III_l5_549

/-- Represents a judge's score for a competitor -/
def Score := Fin 5

/-- Represents a competitor -/
inductive Competitor
| Adam
| Berta
| Clara
| David
| Emil

/-- Represents a judge -/
inductive Judge
| I
| II
| III

/-- The scoring function -/
def scoring : Judge → Competitor → Score := sorry

theorem adam_score_from_judge_III :
  (∀ (j : Judge) (c1 c2 : Competitor), c1 ≠ c2 → scoring j c1 ≠ scoring j c2) →
  (scoring Judge.I Competitor.Adam = ⟨2, by norm_num⟩) →
  (scoring Judge.I Competitor.Berta = ⟨0, by norm_num⟩) →
  (scoring Judge.II Competitor.Berta = ⟨2, by norm_num⟩) →
  (scoring Judge.II Competitor.Clara = ⟨0, by norm_num⟩) →
  ((scoring Judge.I Competitor.Adam).val + (scoring Judge.II Competitor.Adam).val + (scoring Judge.III Competitor.Adam).val = 7) →
  ((scoring Judge.I Competitor.Berta).val + (scoring Judge.II Competitor.Berta).val + (scoring Judge.III Competitor.Berta).val = 5) →
  ((scoring Judge.I Competitor.Clara).val + (scoring Judge.II Competitor.Clara).val + (scoring Judge.III Competitor.Clara).val = 3) →
  ((scoring Judge.I Competitor.David).val + (scoring Judge.II Competitor.David).val + (scoring Judge.III Competitor.David).val = 4) →
  ((scoring Judge.I Competitor.Emil).val + (scoring Judge.II Competitor.Emil).val + (scoring Judge.III Competitor.Emil).val = 11) →
  scoring Judge.III Competitor.Adam = ⟨1, by norm_num⟩ := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_adam_score_from_judge_III_l5_549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integers_sum_divides_sum_of_squares_l5_590

theorem consecutive_integers_sum_divides_sum_of_squares (n : ℕ) (h : n = 1 ∨ n = 12) :
  (10 * n^2 + 90 * n + 285) % (10 * n + 45) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integers_sum_divides_sum_of_squares_l5_590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_will_be_late_l5_502

/-- Represents the cyclist's journey --/
structure Journey where
  total_distance : ℚ
  available_time : ℚ
  segment1_distance : ℚ
  segment1_speed : ℚ
  segment2_distance : ℚ
  segment2_speed : ℚ
  segment3_distance : ℚ
  segment3_speed : ℚ

/-- Calculates the total time required for the journey --/
def total_time (j : Journey) : ℚ :=
  j.segment1_distance / j.segment1_speed +
  j.segment2_distance / j.segment2_speed +
  j.segment3_distance / j.segment3_speed

/-- Theorem stating that the cyclist will be late --/
theorem cyclist_will_be_late (j : Journey) 
  (h1 : j.total_distance = 12)
  (h2 : j.available_time = 3/2)
  (h3 : j.segment1_distance = 4)
  (h4 : j.segment1_speed = 4)
  (h5 : j.segment2_distance = 4)
  (h6 : j.segment2_speed = 12)
  (h7 : j.segment3_distance = 4)
  (h8 : j.segment3_speed = 8) :
  total_time j > j.available_time :=
by
  -- Unfold the definition of total_time
  unfold total_time
  
  -- Simplify the expression
  simp [h3, h4, h5, h6, h7, h8]
  
  -- Perform the calculation
  norm_num
  
  -- Compare with available time
  rw [h2]
  norm_num
  
  -- The proof is complete
  done

#check cyclist_will_be_late

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_will_be_late_l5_502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_iff_two_or_three_l5_553

/-- The number formed by 1 followed by n digits of 4 -/
def num_with_fours (n : ℕ) : ℕ := 
  1 * 10^n + 4 * (10^n - 1) / 9

/-- A number is a perfect square if there exists an integer whose square equals the number -/
def is_perfect_square (m : ℕ) : Prop :=
  ∃ k : ℕ, k^2 = m

/-- Theorem: The number 144...4 (with n digits of 4) is a perfect square if and only if n = 2 or n = 3 -/
theorem perfect_square_iff_two_or_three (n : ℕ) :
  is_perfect_square (num_with_fours n) ↔ n = 2 ∨ n = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_iff_two_or_three_l5_553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_pi_plus_alpha_l5_593

theorem tan_half_pi_plus_alpha (α : ℝ) 
  (h1 : Real.cos (Real.pi + α) = -(1/3)) 
  (h2 : α ∈ Set.Ioo (3*Real.pi/2) (2*Real.pi)) : 
  Real.tan (Real.pi/2 + α) = Real.sqrt 2/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_pi_plus_alpha_l5_593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_of_three_l5_546

/-- Represents a cube with numbers on its faces -/
structure NumberedCube where
  faces : Fin 6 → Nat
  sum_is_21 : (List.range 6).sum = 21
  valid_numbers : ∀ i : Fin 6, faces i ∈ List.range 6

/-- The sum of numbers on four lateral faces in a roll -/
def lateral_sum (c : NumberedCube) (top bottom : Fin 6) : Nat :=
  21 - (c.faces top + c.faces bottom)

/-- Function to get the opposite face -/
def opposite_face : Fin 6 → Fin 6 := 
  fun i => match i with
  | 0 => 5
  | 1 => 4
  | 2 => 3
  | 3 => 2
  | 4 => 1
  | 5 => 0

theorem opposite_of_three (c : NumberedCube) 
  (h1 : ∃ top1 bottom1, lateral_sum c top1 bottom1 = 12)
  (h2 : ∃ top2 bottom2, lateral_sum c top2 bottom2 = 15)
  (h3 : ∃ face, c.faces face = 3) :
  ∃ opposite, c.faces opposite = 6 ∧ 
    (∀ i : Fin 6, c.faces i = 3 → c.faces (opposite_face i) = 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_of_three_l5_546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_currency_representation_theorem_l5_586

theorem currency_representation_theorem :
  let solutions := {(x, y) : ℕ × ℕ | 7 * x + 9 * y = 997}
  ∃ f : Fin 16 → ℕ × ℕ,
    (∀ k : Fin 16, f k = (1 + 9 * k.val, 110 - 7 * k.val)) ∧
    (∀ s, s ∈ solutions ↔ ∃ k : Fin 16, f k = s) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_currency_representation_theorem_l5_586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_closed_form_l5_562

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 0 -- Adding a case for 0 to cover all natural numbers
  | 1 => 2 * Real.sqrt 3
  | n + 2 => 8 * sequence_a (n + 1) / (4 - (sequence_a (n + 1))^2)

theorem sequence_a_closed_form (n : ℕ) :
  n ≥ 1 → sequence_a n = 2 * Real.tan (π / (3 * 2^(n-1))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_closed_form_l5_562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_wins_l5_506

/-- Represents a player in the game -/
inductive Player
| First
| Second

/-- Represents the state of the game board -/
structure GameState where
  digits : List Nat
  currentPlayer : Player

/-- Checks if a number is divisible by 11 -/
def isDivisibleBy11 (n : Nat) : Bool :=
  n % 11 = 0

/-- Checks if the game is over (i.e., if there's a number divisible by 11) -/
def isGameOver (state : GameState) : Bool :=
  let numbers := state.digits.reverse.inits.map (fun l => l.foldl (fun acc d => acc * 10 + d) 0)
  numbers.any isDivisibleBy11

/-- Represents a move in the game -/
def Move := Nat

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  { digits := move :: state.digits,
    currentPlayer := match state.currentPlayer with
      | Player.First => Player.Second
      | Player.Second => Player.First }

/-- Checks if a player has a winning strategy -/
def hasWinningStrategy (player : Player) (depth : Nat) (state : GameState) : Prop :=
  state.currentPlayer = player ∧
  (depth > 0 →
    ∃ (move : Move), ¬isGameOver (applyMove state move) ∧
      ∀ (opponentMove : Move), hasWinningStrategy player (depth - 1) (applyMove (applyMove state move) opponentMove))

/-- The main theorem: the second player has a winning strategy -/
theorem second_player_wins : 
  ∀ (depth : Nat) (state : GameState), state.currentPlayer = Player.Second → 
    hasWinningStrategy Player.Second depth state := by
  sorry

#check second_player_wins

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_wins_l5_506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_opposite_sides_m_range_l5_516

/-- Given points (3,1) and (-4,6) are on opposite sides of the line 3x-2y+m=0,
    prove that the range of values for m is -7 < m < 24. -/
theorem line_opposite_sides_m_range :
  ∀ m : ℝ,
  (let f : ℝ × ℝ → ℝ := λ (x, y) ↦ 3 * x - 2 * y + m
   (f (3, 1) * f (-4, 6) < 0)) ↔
  -7 < m ∧ m < 24 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_opposite_sides_m_range_l5_516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_aaron_in_middle_l5_517

-- Define the set of friends
inductive Friend : Type
| Aaron : Friend
| Darren : Friend
| Karen : Friend
| Maren : Friend
| Sharon : Friend

-- Define the car positions
def CarPosition : Type := Fin 5

-- Define the seating arrangement
def Seating : Type := Friend → CarPosition

-- Define the conditions of the seating arrangement
def ValidSeating (s : Seating) : Prop :=
  -- Aaron sat directly in front of Maren
  (s Friend.Aaron).val + 1 = (s Friend.Maren).val ∧
  -- Darren is seated two cars ahead of Maren
  (s Friend.Darren).val + 2 = (s Friend.Maren).val ∧
  -- No one sat directly between Darren and Maren
  ∀ f : Friend, (s f).val ≠ (s Friend.Maren).val - 1 ∨ f = Friend.Aaron ∧
  -- Karen sat directly behind Sharon
  (s Friend.Karen).val = (s Friend.Sharon).val + 1 ∧
  -- Each car seats only one person
  ∀ f1 f2 : Friend, f1 ≠ f2 → s f1 ≠ s f2

-- The theorem to prove
theorem aaron_in_middle (s : Seating) (h : ValidSeating s) : 
  (s Friend.Aaron).val = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_aaron_in_middle_l5_517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_three_l5_596

/-- The sum of the series (3n+2)/m^n from n=1 to infinity -/
noncomputable def seriesSum (m : ℝ) := ∑' n, (3 * n + 2) / m^n

theorem series_sum_equals_three (m : ℝ) (hm : m > 1) (hsum : seriesSum m = 2) : m = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_three_l5_596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bullet_speed_maintains_distance_l5_591

/-- Represents the scenario of a bullet hitting two blocks on a horizontal surface -/
structure BulletBlockScenario where
  M : ℝ  -- Mass of each block
  m : ℝ  -- Mass of the bullet
  S : ℝ  -- Initial and final distance between blocks
  μ : ℝ  -- Coefficient of friction
  g : ℝ  -- Acceleration due to gravity

/-- The required speed of the bullet -/
noncomputable def requiredBulletSpeed (scenario : BulletBlockScenario) : ℝ :=
  (2 * scenario.M / scenario.m) * Real.sqrt (scenario.μ * scenario.g * scenario.S)

/-- Function to calculate the final distance between blocks -/
noncomputable def FinalDistance (scenario : BulletBlockScenario) (v : ℝ) : ℝ :=
  scenario.S -- Placeholder implementation

/-- Theorem stating that the calculated bullet speed results in the final distance remaining S -/
theorem bullet_speed_maintains_distance (scenario : BulletBlockScenario) 
  (hm : scenario.m < scenario.M)  -- Assumption: m << M
  (hS : scenario.S > 0)  -- Assumption: Distance is positive
  (hμ : scenario.μ > 0)  -- Assumption: Coefficient of friction is positive
  (hg : scenario.g > 0)  -- Assumption: Gravity is positive
  : 
  let v := requiredBulletSpeed scenario
  FinalDistance scenario v = scenario.S :=
by
  sorry  -- Proof to be implemented


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bullet_speed_maintains_distance_l5_591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_values_l5_582

noncomputable def g (x : ℝ) : ℝ :=
  if x < 2 then 2 * x - 1 else 10 - 3 * x

theorem g_values : g (-4) = -9 ∧ g 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_values_l5_582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_600_plus_tan_240_l5_540

theorem sin_600_plus_tan_240 : Real.sin (600 * Real.pi / 180) + Real.tan (240 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_600_plus_tan_240_l5_540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplified_car_ratio_l5_592

theorem simplified_car_ratio (red_cars black_cars : ℕ) 
  (h1 : red_cars = 26) (h2 : black_cars = 70) :
  (red_cars / Nat.gcd red_cars black_cars, black_cars / Nat.gcd red_cars black_cars) = (13, 35) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplified_car_ratio_l5_592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_example_l5_584

/-- The area of a triangle given its vertices -/
noncomputable def triangleArea (A B C : ℝ × ℝ) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  (1/2) * abs ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

/-- Theorem: The area of a triangle with vertices A(2, 2), B(8, 2), and C(5, 10) is 24 square units -/
theorem triangle_area_example : triangleArea (2, 2) (8, 2) (5, 10) = 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_example_l5_584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_fill_time_is_48_minutes_l5_536

/-- Represents the time it takes to fill the pool with both valves open -/
noncomputable def fill_time (pool_capacity : ℝ) (first_valve_time : ℝ) (valve_difference : ℝ) : ℝ :=
  let first_valve_rate := pool_capacity / first_valve_time
  let second_valve_rate := first_valve_rate + valve_difference
  let combined_rate := first_valve_rate + second_valve_rate
  pool_capacity / combined_rate

/-- Theorem stating that it takes 48 minutes to fill the pool with both valves open -/
theorem pool_fill_time_is_48_minutes :
  fill_time 12000 120 50 = 48 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval fill_time 12000 120 50

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_fill_time_is_48_minutes_l5_536
