import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_root_l1316_131613

open Real

/-- The function f(x) = a^2 * ln(x) - x^2 + ax -/
noncomputable def f (a : ℝ) (x : ℝ) := a^2 * log x - x^2 + a*x

theorem f_monotonicity_and_root (a : ℝ) (h : a > 0) :
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < a → f a x₁ > f a x₂) ∧ 
  (∀ x₁ x₂, a < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂) ∧
  (∃ x, 1 < x ∧ x < exp 1 ∧ f a x = 0) ↔ 
  (1 < a ∧ a < (sqrt 5 - 1) / 2 * exp 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_root_l1316_131613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sphere_volume_in_prism_l1316_131632

/-- The volume of a sphere -/
noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

/-- The maximum volume of a sphere enclosed in a right triangular prism -/
theorem max_sphere_volume_in_prism (ab bc aa₁ : ℝ) (h_ab : ab = 6) (h_bc : bc = 8) (h_aa₁ : aa₁ = 3) :
  ∃ (v : ℝ), v = sphere_volume (aa₁ / 2) ∧ v = (9 * Real.pi) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sphere_volume_in_prism_l1316_131632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_from_volume_l1316_131610

/-- The volume of a cylinder with hemispheres at both ends -/
noncomputable def cylinderWithHemispheres (r : ℝ) (h : ℝ) : ℝ :=
  Real.pi * r^2 * h + (4/3) * Real.pi * r^3

/-- The length of a line segment given the volume of its 4-unit neighborhood -/
theorem length_from_volume (V : ℝ) : 
  V = cylinderWithHemispheres 4 18 → 384 * Real.pi = V := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_from_volume_l1316_131610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_coordinates_l1316_131616

/-- Given a parabola with equation y = (1/8)x^2, its focus coordinates are (0, 2) -/
theorem parabola_focus_coordinates (x y : ℝ) :
  y = (1/8) * x^2 → (0, 2) = (0, 1/4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_coordinates_l1316_131616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l1316_131625

/-- A function g defined as g(x) = sin(ω(x + π/2)), where ω > 0 and ω is an odd integer -/
noncomputable def g (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * (x + Real.pi / 2))

/-- Conditions on ω -/
def is_valid_ω (ω : ℝ) : Prop := ω > 0 ∧ ∃ k : ℤ, ω = 2 * ↑k + 1

theorem g_properties (ω : ℝ) (h : is_valid_ω ω) :
  (∀ x, g ω x = g ω (-x)) ∧  -- g is even
  g ω (-Real.pi / 2) = 0 ∧   -- g(-π/2) = 0
  (ω = 5 → ∃ x₁ x₂ x₃, 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ ≤ Real.pi / 2 ∧
    g ω x₁ = 0 ∧ g ω x₂ = 0 ∧ g ω x₃ = 0 ∧
    ∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ g ω x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃) ∧
  (∀ ω', is_valid_ω ω' →
    (∀ x₁ x₂, 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ Real.pi / 5 → g ω' x₁ > g ω' x₂) →
    ω' ≤ 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l1316_131625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_specific_vectors_l1316_131640

noncomputable def angle_between (a b : ℝ × ℝ) : ℝ :=
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))

theorem angle_between_specific_vectors :
  let a : ℝ × ℝ := (1, Real.sqrt 3)
  let b : ℝ × ℝ := (Real.sqrt 3 + 1, Real.sqrt 3 - 1)
  angle_between a b = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_specific_vectors_l1316_131640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_efficiency_increase_sakshi_tanya_l1316_131682

/-- Represents the efficiency of a worker in terms of work completed per day -/
noncomputable def efficiency (days : ℝ) : ℝ := 1 / days

/-- Calculates the percentage increase between two values -/
noncomputable def percentage_increase (original : ℝ) (new : ℝ) : ℝ :=
  ((new - original) / original) * 100

theorem efficiency_increase_sakshi_tanya : 
  let sakshi_days : ℝ := 12
  let tanya_days : ℝ := 10
  let sakshi_efficiency := efficiency sakshi_days
  let tanya_efficiency := efficiency tanya_days
  percentage_increase sakshi_efficiency tanya_efficiency = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_efficiency_increase_sakshi_tanya_l1316_131682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_sum_diff_magnitude_implies_orthogonal_l1316_131645

def planar_vector := ℝ × ℝ

def dot_product (a b : planar_vector) : ℝ :=
  a.1 * b.1 + a.2 * b.2

noncomputable def magnitude (v : planar_vector) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

theorem equal_sum_diff_magnitude_implies_orthogonal (a b : planar_vector) :
  magnitude (a.1 + b.1, a.2 + b.2) = magnitude (a.1 - b.1, a.2 - b.2) →
  dot_product a b = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_sum_diff_magnitude_implies_orthogonal_l1316_131645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_prime_factors_l1316_131602

/-- Definition of the sequence a_n -/
def a (n : ℕ) (a₀ : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => a₀
  | n + 2 => a₀ * a (n + 1) a₀ - a n a₀

/-- The main theorem -/
theorem infinitely_many_prime_factors (a₀ : ℕ) (ha₀ : a₀ > 1) :
  ∃ (S : Set ℕ), (Set.Infinite S) ∧ (∀ p ∈ S, Nat.Prime p ∧ ∃ n, p ∣ a n a₀) := by
  sorry

#check infinitely_many_prime_factors

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_prime_factors_l1316_131602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_junior_score_is_90_l1316_131675

noncomputable def junior_score (n : ℝ) (junior_percent : ℝ) (senior_percent : ℝ) 
  (total_average : ℝ) (senior_average : ℝ) : ℝ :=
  (total_average - senior_percent * senior_average) / junior_percent

theorem junior_score_is_90 (n : ℝ) (junior_percent : ℝ) (senior_percent : ℝ) 
  (total_average : ℝ) (senior_average : ℝ) :
  junior_percent = 0.2 →
  senior_percent = 0.8 →
  junior_percent + senior_percent = 1 →
  total_average = 78 →
  senior_average = 75 →
  (junior_percent * n * (junior_score n junior_percent senior_percent total_average senior_average) + 
   senior_percent * n * senior_average) / n = total_average →
  junior_score n junior_percent senior_percent total_average senior_average = 90 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_junior_score_is_90_l1316_131675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expr1_eq_one_expr2_eq_sqrt_six_l1316_131656

open Real

-- Define the expressions
noncomputable def expr1 : ℝ := (sin (27 * π / 180) + cos (45 * π / 180) * sin (18 * π / 180)) / 
                 (cos (27 * π / 180) - sin (45 * π / 180) * sin (18 * π / 180))

noncomputable def expr2 : ℝ := 2 * sin (50 * π / 180) + sin (10 * π / 180) * 
                 (1 + Real.sqrt 3 * tan (10 * π / 180)) * 
                 Real.sqrt (2 * sin (80 * π / 180) ^ 2)

-- State the theorems
theorem expr1_eq_one : expr1 = 1 := by sorry

theorem expr2_eq_sqrt_six : expr2 = Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expr1_eq_one_expr2_eq_sqrt_six_l1316_131656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1316_131661

-- Define the vector product operation
def vector_product (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 * b.1, a.2 * b.2)

-- Define the vectors m and n
noncomputable def m : ℝ × ℝ := (2, 1/2)
noncomputable def n : ℝ × ℝ := (Real.pi/3, 0)

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  let p : ℝ × ℝ := (x, Real.sin x)
  (vector_product m p + n).2

-- Theorem statement
theorem range_of_f :
  Set.range f = Set.Icc (-1/2 : ℝ) (1/2 : ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1316_131661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_fourth_l1316_131635

theorem tan_alpha_plus_pi_fourth (α : ℝ) 
  (h1 : α ∈ Set.Ioo (π / 2) π) 
  (h2 : Real.sin α = 3 / 5) : 
  Real.tan (α + π / 4) = 1 / 7 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_fourth_l1316_131635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_origin_l1316_131677

-- Define the curve
def on_curve (x y : ℝ) : Prop := x^4 + y^4 = 1

-- Define the distance function from a point to the origin
noncomputable def distance_to_origin (x y : ℝ) : ℝ := Real.sqrt (x^2 + y^2)

-- Theorem statement
theorem max_distance_to_origin :
  ∃ (max_dist : ℝ), max_dist = Real.sqrt 2 ∧
  ∀ (x y : ℝ), on_curve x y →
  distance_to_origin x y ≤ max_dist :=
by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_origin_l1316_131677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_road_trip_speed_l1316_131624

/-- Calculates the constant speed of a road trip given the total distance, total time,
    break frequency, break duration, and hotel search time. -/
noncomputable def calculateSpeed (totalDistance : ℝ) (totalTime : ℝ) (breakFrequency : ℝ) 
                   (breakDuration : ℝ) (hotelSearchTime : ℝ) : ℝ :=
  let numBreaks := totalTime / breakFrequency
  let totalBreakTime := numBreaks * breakDuration
  let drivingTime := totalTime - totalBreakTime - hotelSearchTime
  totalDistance / drivingTime

/-- Theorem stating that the constant speed for the given road trip conditions
    is approximately 62.7 miles per hour. -/
theorem road_trip_speed : 
  let totalDistance : ℝ := 2790
  let totalTime : ℝ := 50
  let breakFrequency : ℝ := 5
  let breakDuration : ℝ := 0.5
  let hotelSearchTime : ℝ := 0.5
  abs (calculateSpeed totalDistance totalTime breakFrequency breakDuration hotelSearchTime - 62.7) < 0.1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_road_trip_speed_l1316_131624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_range_l1316_131636

theorem triangle_area_range (a b c : ℝ) (A B C : ℝ) :
  b^2 + c^2 = a^2 - b*c →
  b * Real.sin A = 4 * Real.sin B →
  Real.log b + Real.log c ≥ 1 - 2 * Real.cos (B + C) →
  A = 2 * Real.pi / 3 ∧
  (let S := (1/2) * b * c * Real.sin A;
   Real.sqrt 3 / 4 ≤ S ∧ S ≤ 4 * Real.sqrt 3 / 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_range_l1316_131636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_C₂_symmetry_l1316_131606

noncomputable def C₂ (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 3)

theorem C₂_symmetry : 
  ∀ (x : ℝ), C₂ (5 * Real.pi / 12 + x) = C₂ (5 * Real.pi / 12 - x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_C₂_symmetry_l1316_131606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_quadrilateral_from_right_triangles_l1316_131641

/-- The area of a quadrilateral formed by two right triangles with given side lengths -/
theorem area_of_quadrilateral_from_right_triangles 
  (PQ PR PS RS : ℝ) 
  (h_pq : PQ = 8) 
  (h_pr : PR = 10) 
  (h_ps : PS = 10) 
  (h_rs : RS = 6)
  (h_pqr_right : PQ ^ 2 + QR ^ 2 = PR ^ 2) 
  (h_prs_right : PR ^ 2 + RS ^ 2 = PS ^ 2) : 
  (1 / 2 * PQ * PR) + (1 / 2 * PR * RS) = 70 := by
  sorry

#check area_of_quadrilateral_from_right_triangles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_quadrilateral_from_right_triangles_l1316_131641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_with_sine_line_through_point_with_double_angle_line_through_point_with_intercept_sum_l1316_131607

-- Define a line type
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : (x y : ℝ) → a * x + b * y + c = 0

-- Define a point type
structure Point where
  x : ℝ
  y : ℝ

-- Function to check if a point is on a line
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Function to calculate sine of angle of inclination
noncomputable def sineOfInclination (l : Line) : ℝ :=
  l.b / Real.sqrt (l.a^2 + l.b^2)

-- Function to calculate angle of inclination
noncomputable def angleOfInclination (l : Line) : ℝ :=
  Real.arctan (l.b / l.a)

-- Function to calculate sum of intercepts
noncomputable def sumOfIntercepts (l : Line) : ℝ :=
  -l.c/l.a - l.c/l.b

-- Theorem 1
theorem line_through_point_with_sine (l : Line) :
  pointOnLine ⟨-4, 0⟩ l ∧ sineOfInclination l = Real.sqrt 10 / 10 →
  (l.a = 1 ∧ l.b = -3 ∧ l.c = 4) ∨ (l.a = 1 ∧ l.b = 3 ∧ l.c = 4) := by
  sorry

-- Theorem 2
theorem line_through_point_with_double_angle (l : Line) :
  pointOnLine ⟨-1, -3⟩ l ∧ angleOfInclination l = 2 * Real.arctan 3 →
  l.a = 3 ∧ l.b = 4 ∧ l.c = 15 := by
  sorry

-- Theorem 3
theorem line_through_point_with_intercept_sum (l : Line) :
  pointOnLine ⟨-3, 4⟩ l ∧ sumOfIntercepts l = 12 →
  (l.a = 1 ∧ l.b = 3 ∧ l.c = 9) ∨ (l.a = 4 ∧ l.b = -1 ∧ l.c = 16) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_with_sine_line_through_point_with_double_angle_line_through_point_with_intercept_sum_l1316_131607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_x_over_3_period_l1316_131644

/-- The period of sin(x/3) is 6π -/
theorem sin_x_over_3_period : ∃ (p : ℝ), p > 0 ∧ 
  (∀ x : ℝ, Real.sin (x / 3) = Real.sin ((x + p) / 3)) ∧ 
  (∀ q : ℝ, 0 < q → q < p → 
    ∃ x : ℝ, Real.sin (x / 3) ≠ Real.sin ((x + q) / 3)) ∧ 
  p = 6 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_x_over_3_period_l1316_131644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_doubled_cone_volume_is_240pi_l1316_131673

noncomputable def cone_volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

noncomputable def d : ℝ := 12

noncomputable def h : ℝ := 10

noncomputable def r : ℝ := d / 2

theorem doubled_cone_volume_is_240pi :
  2 * cone_volume r h = 240 * Real.pi := by
  -- Unfold definitions
  unfold cone_volume r d
  -- Simplify
  simp [Real.pi]
  -- The rest of the proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_doubled_cone_volume_is_240pi_l1316_131673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_eigenvalues_l1316_131601

/-- Given a 2x2 matrix M with real entries that transforms (3, -1) to (3, 5),
    prove that its eigenvalues are -1 and 4. -/
theorem matrix_eigenvalues (a b : ℝ) : 
  let M : Matrix (Fin 2) (Fin 2) ℝ := !![2, a; b, 1]
  (M.mulVec ![3, -1] = ![3, 5]) →
  (∃ (v : Fin 2 → ℝ), v ≠ 0 ∧ M.mulVec v = (-1) • v) ∧
  (∃ (w : Fin 2 → ℝ), w ≠ 0 ∧ M.mulVec w = (4 : ℝ) • w) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_eigenvalues_l1316_131601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_roots_l1316_131666

theorem quadratic_equation_roots (a : ℝ) : 
  (a ≠ 0 ∧ 4 - 4*a > 0 ∧ a ∈ ({0, -1, 1, 2} : Set ℝ)) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_roots_l1316_131666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_problem_l1316_131648

-- Define the quadratic function
def f (a b : ℝ) (x : ℝ) := a * x^2 - b * x - 1

-- State the theorem
theorem quadratic_inequality_problem :
  ∀ a b : ℝ,
  (∀ x : ℝ, x ∈ Set.Icc (-1/2) (-1/3) → f a b x ≥ 0) →
  (∀ x : ℝ, x ∉ Set.Icc (-1/2) (-1/3) → f a b x < 0) →
  (a = -6 ∧ b = 5) ∧
  (∀ x : ℝ, x^2 - b*x - a < 0 ↔ x ∈ Set.Ioo 2 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_problem_l1316_131648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_in_cylinder_l1316_131628

-- Define the parameters
def cone_radius : ℝ := 15
def cone_height : ℝ := 24
def cylinder_radius : ℝ := 18

-- Define the volume of the cone
noncomputable def cone_volume : ℝ := (1/3) * Real.pi * cone_radius^2 * cone_height

-- Define the height of water in the cylinder
noncomputable def cylinder_water_height : ℝ := cone_volume / (Real.pi * cylinder_radius^2)

-- Theorem to prove
theorem water_height_in_cylinder :
  cylinder_water_height = 25/3 := by
  -- Expand the definitions
  unfold cylinder_water_height cone_volume
  -- Simplify the expression
  simp [cone_radius, cone_height, cylinder_radius]
  -- The proof itself
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_in_cylinder_l1316_131628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_parity_of_2020_naturals_l1316_131609

theorem product_parity_of_2020_naturals (nums : Fin 2020 → ℕ) 
  (h_sum_odd : Odd (Finset.sum Finset.univ (fun i => nums i))) : 
  Even (Finset.prod Finset.univ (fun i => nums i)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_parity_of_2020_naturals_l1316_131609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_at_zero_l1316_131604

noncomputable section

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

axiom h : ∀ x, f x = x^2 + 2*x*(f' 1)
axiom h' : ∀ x, HasDerivAt f (f' x) x

theorem derivative_at_zero : f' 0 = -4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_at_zero_l1316_131604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_binomial_fraction_is_integer_l1316_131634

theorem gcd_binomial_fraction_is_integer (m n : ℤ) (h1 : n ≥ m) (h2 : m ≥ 1) :
  ∃ k : ℤ, (Nat.gcd m.natAbs n.natAbs : ℚ) / n * Nat.choose n.natAbs m.natAbs = k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_binomial_fraction_is_integer_l1316_131634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_tangent_to_truncated_cone_l1316_131686

/-- Predicate to represent that a sphere is tangent to a truncated cone -/
def IsTangentSphere (r : ℝ) (r₁ : ℝ) (r₂ : ℝ) : Prop :=
  ∃ (h : ℝ), 
    h > 0 ∧ 
    r₁ > r₂ ∧ 
    r = (r₁ - r₂) * Real.sqrt ((h / (r₁ - r₂))^2 + 1) / 2

/-- The radius of a sphere tangent to a truncated cone -/
theorem sphere_radius_tangent_to_truncated_cone 
  (r₁ : ℝ) 
  (r₂ : ℝ) 
  (h : r₁ = 20 ∧ r₂ = 5) : 
  ∃ (r : ℝ), r = (15 * Real.sqrt 2) / 2 ∧ 
  r > 0 ∧ 
  r < r₁ ∧ 
  r < r₂ ∧
  IsTangentSphere r r₁ r₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_tangent_to_truncated_cone_l1316_131686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equals_one_sufficient_not_necessary_l1316_131698

noncomputable def f (x a : ℝ) : ℝ := Real.log (Real.sqrt (x^2 + a^2) - x)

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f x + f (-x) = 0

theorem a_equals_one_sufficient_not_necessary :
  (∃ a : ℝ, a ≠ 1 ∧ is_odd_function (f · a)) ∧
  (is_odd_function (f · 1)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equals_one_sufficient_not_necessary_l1316_131698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alexandra_magazines_l1316_131653

/-- The number of magazines Alexandra has after her purchases and the dog incident --/
def total_magazines (friday_magazines saturday_magazines sunday_multiplier chewed_magazines : ℕ) : ℕ :=
  friday_magazines + saturday_magazines + (sunday_multiplier * friday_magazines) - chewed_magazines

/-- Theorem stating that Alexandra has 48 magazines --/
theorem alexandra_magazines : 
  total_magazines 8 12 4 4 = 48 := by
  -- Unfold the definition of total_magazines
  unfold total_magazines
  -- Perform the arithmetic
  norm_num

#eval total_magazines 8 12 4 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alexandra_magazines_l1316_131653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pentagon_area_l1316_131651

/-- Pentagon with specific side lengths -/
structure Pentagon where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  side5 : ℝ

/-- Right triangle with two given sides -/
structure RightTriangle where
  base : ℝ
  height : ℝ

/-- Trapezoid with two bases and height -/
structure Trapezoid where
  base1 : ℝ
  base2 : ℝ
  height : ℝ

/-- Area calculation for the pentagon -/
noncomputable def pentagon_area (p : Pentagon) (t : RightTriangle) (z : Trapezoid) : ℝ :=
  (1/2 * t.base * t.height) + (1/2 * (z.base1 + z.base2) * z.height)

/-- Main theorem: Area of the specific pentagon -/
theorem specific_pentagon_area :
  let p : Pentagon := { side1 := 18, side2 := 25, side3 := 30, side4 := 28, side5 := 25 }
  let t : RightTriangle := { base := 18, height := 25 }
  let z : Trapezoid := { base1 := 28, base2 := 30, height := 25 }
  pentagon_area p t z = 950 := by
  sorry

#check specific_pentagon_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pentagon_area_l1316_131651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_divisible_by_5_or_2_l1316_131639

def digits : Finset ℕ := {1, 2, 3, 4, 5}

def is_divisible_by_5_or_2 (n : ℕ) : Bool :=
  n % 5 = 0 ∨ n % 2 = 0

def last_digit (n : ℕ) : ℕ :=
  n % 10

theorem probability_divisible_by_5_or_2 :
  (Finset.filter (λ d : ℕ => is_divisible_by_5_or_2 (last_digit d)) digits).card / digits.card = 3 / 5 := by
  -- Evaluate the filter
  simp [is_divisible_by_5_or_2, last_digit, digits]
  -- The rest of the proof
  sorry

#eval (Finset.filter (λ d : ℕ => is_divisible_by_5_or_2 (last_digit d)) digits).card
#eval digits.card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_divisible_by_5_or_2_l1316_131639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_plus_pi_fourth_monotonicity_interval_l1316_131672

-- Define the angle α and point P
noncomputable def α : ℝ := sorry
noncomputable def P : ℝ × ℝ := sorry

-- Part 1
theorem sin_alpha_plus_pi_fourth (h : P = (-3, 4)) :
  Real.sin (α + π/4) = Real.sqrt 2/10 := by sorry

-- Part 2
noncomputable def f (x : ℝ) : ℝ := Real.sin (x + α) + Real.cos x

theorem monotonicity_interval (h1 : P = (-3, Real.sqrt 3)) (h2 : α ∈ Set.Ioo 0 (2*π)) :
  ∃ (k : ℤ), StrictMonoOn f (Set.Icc (5*π/6 + 2*k*π) (11*π/6 + 2*k*π)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_plus_pi_fourth_monotonicity_interval_l1316_131672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l1316_131627

-- Define a quadratic function
def isQuadratic (f : ℝ → ℝ) : Prop := ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c

-- Define the theorem
theorem quadratic_function_properties (f : ℝ → ℝ) 
  (h_quad : isQuadratic f)
  (h_f_neg_one : f (-1) = 2)
  (h_f_prime_zero : (deriv f) 0 = 0)
  (h_integral : ∫ x in Set.Icc 0 1, f x = -2) :
  (∀ x, f x = 6 * x^2 - 4) ∧ 
  (∀ x ∈ Set.Icc (-1) 1, f x ≥ -4) ∧
  (∀ x ∈ Set.Icc (-1) 1, f x ≤ 2) ∧
  (∃ x ∈ Set.Icc (-1) 1, f x = -4) ∧
  (∃ x ∈ Set.Icc (-1) 1, f x = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l1316_131627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_coordinates_l1316_131618

noncomputable section

-- Define the curve C
def curve_C (θ : Real) : Real × Real :=
  (2 * Real.cos θ, Real.sqrt 3 * Real.sin θ)

-- Define the condition for θ
def θ_condition (θ : Real) : Prop :=
  Real.pi ≤ θ ∧ θ ≤ 2 * Real.pi

-- Define a point P on the curve
def point_P (θ : Real) : Real × Real :=
  curve_C θ

-- Define the angle of inclination of line OP
def angle_OP (P : Real × Real) : Real :=
  Real.arctan (P.2 / P.1)

-- State the theorem
theorem point_P_coordinates :
  ∃ θ : Real, θ_condition θ ∧ 
  angle_OP (point_P θ) = Real.pi / 3 ∧
  point_P θ = (-2 * Real.sqrt 5 / 5, -2 * Real.sqrt 15 / 5) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_coordinates_l1316_131618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_choir_configurations_l1316_131630

theorem choir_configurations (n : ℕ) : 
  (n = 90) → 
  (Finset.filter (λ x ↦ 6 ≤ x ∧ x ≤ 15 ∧ n % x = 0) (Finset.range 16)).card = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_choir_configurations_l1316_131630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_police_can_catch_criminal_l1316_131687

/-- Represents a position on the grid --/
structure Position where
  avenue : Nat
  street : Int

/-- Represents the state of the chase --/
structure ChaseState where
  criminal : Position
  officer1 : Position
  officer2 : Position
  time : ℝ

/-- The speed of the police officers --/
def officerSpeed : ℝ := 1

/-- The maximum speed of the criminal --/
def criminalMaxSpeed : ℝ := 10 * officerSpeed

/-- The number of avenues --/
def numAvenues : Nat := 10

/-- The maximum initial distance between the criminal and officers --/
def maxInitialDistance : Nat := 100

/-- A strategy is a function that takes the current state and returns the next positions for the officers --/
def Strategy := ChaseState → (Position × Position)

/-- Predicate to check if the criminal is caught --/
def isCaught (state : ChaseState) : Prop :=
  state.criminal.avenue = state.officer1.avenue ∨
  state.criminal.avenue = state.officer2.avenue ∨
  state.criminal.street = state.officer1.street ∨
  state.criminal.street = state.officer2.street

/-- Execute the strategy for a given time --/
def executeStrategy (strategy : Strategy) (initial_state : ChaseState) (t : ℝ) : ChaseState :=
  sorry -- Implementation details omitted for brevity

/-- The main theorem to prove --/
theorem police_can_catch_criminal :
  ∃ (strategy : Strategy), ∀ (initial_state : ChaseState),
    initial_state.criminal.avenue ≤ numAvenues ∧
    (|initial_state.criminal.street - initial_state.officer1.street| ≤ maxInitialDistance ∨
     |initial_state.criminal.street - initial_state.officer2.street| ≤ maxInitialDistance) →
    ∃ (t : ℝ), isCaught (executeStrategy strategy initial_state t) := by
  sorry -- Proof omitted


end NUMINAMATH_CALUDE_ERRORFEEDBACK_police_can_catch_criminal_l1316_131687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l1316_131642

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 - x + 1 / (x - 2 + a)

-- State the theorem
theorem function_property (a : ℝ) :
  (∀ x₁ x₂ : ℝ, 1 < x₁ ∧ x₁ < x₂ ∧ x₂ < 4 →
    (f a x₁ - f a x₂) / (x₁ - x₂) < -1) →
  a = -2 ∨ a = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l1316_131642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2017_equals_1_l1316_131662

def sequenceA (n : ℕ) : ℤ :=
  match n with
  | 0 => 1
  | 1 => 2
  | n + 2 => sequenceA (n + 1) - sequenceA n

theorem sequence_2017_equals_1 : sequenceA 2016 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2017_equals_1_l1316_131662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arcsin_arccos_equation_solution_l1316_131658

theorem arcsin_arccos_equation_solution :
  ∀ x : ℝ, x ∈ Set.Icc (-1 : ℝ) 1 →
    (Real.arcsin x + Real.arcsin (2*x) = Real.arccos x) ↔ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arcsin_arccos_equation_solution_l1316_131658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_family_financial_analysis_l1316_131646

/-- Represents the data for a family's monthly financial investment and income. -/
structure FamilyData where
  x : ℝ  -- monthly financial investment
  y : ℝ  -- monthly income

/-- Represents the statistical data for a group of families. -/
structure GroupData where
  n : ℕ              -- number of families
  sum_x : ℝ          -- sum of x values
  sum_y : ℝ          -- sum of y values
  sum_xy : ℝ         -- sum of x*y values
  sum_x_sq : ℝ       -- sum of x^2 values

/-- Calculates the slope of the regression line. -/
noncomputable def calculateSlope (data : GroupData) : ℝ :=
  let mean_x := data.sum_x / data.n
  let mean_y := data.sum_y / data.n
  (data.sum_xy - data.n * mean_x * mean_y) / (data.sum_x_sq - data.n * mean_x^2)

/-- Calculates the y-intercept of the regression line. -/
noncomputable def calculateIntercept (data : GroupData) (slope : ℝ) : ℝ :=
  let mean_x := data.sum_x / data.n
  let mean_y := data.sum_y / data.n
  mean_y - slope * mean_x

/-- Predicts the y value for a given x using the regression equation. -/
def predict (slope : ℝ) (intercept : ℝ) (x : ℝ) : ℝ :=
  slope * x + intercept

/-- Main theorem about the regression analysis of the family financial data. -/
theorem family_financial_analysis (data : GroupData)
    (h_n : data.n = 5)
    (h_sum_x : data.sum_x = 40)
    (h_sum_y : data.sum_y = 100)
    (h_sum_xy : data.sum_xy = 821)
    (h_sum_x_sq : data.sum_x_sq = 330) :
    let slope := calculateSlope data
    let intercept := calculateIntercept data slope
    (slope = 2.1 ∧
     intercept = 3.2 ∧
     slope > 0 ∧
     predict slope intercept 5 = 13.7) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_family_financial_analysis_l1316_131646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_at_sum_l1316_131691

noncomputable def f (x : ℝ) (φ : ℝ) := 4 * Real.sin (2 * x + φ)

theorem function_value_at_sum (φ : ℝ) (x₁ x₂ : ℝ) :
  |φ| < π / 2 →
  f (π / 12) φ = f (-π / 12) φ →
  x₁ ∈ Set.Ioo (-7 * π / 6) (-5 * π / 12) →
  x₂ ∈ Set.Ioo (-7 * π / 6) (-5 * π / 12) →
  x₁ ≠ x₂ →
  f x₁ φ = f x₂ φ →
  f (x₁ + x₂) φ = 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_at_sum_l1316_131691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_l1316_131676

/-- The distance from the point (0, 0) to the line 3x + 4y - 25 = 0 is 5 -/
theorem distance_point_to_line : 
  let point : ℝ × ℝ := (0, 0)
  let line := {(x, y) : ℝ × ℝ | 3 * x + 4 * y - 25 = 0}
  Real.sqrt ((3 * point.1 + 4 * point.2 - 25) ^ 2 / (3^2 + 4^2)) = 5 := by
  sorry

#check distance_point_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_l1316_131676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_chord_length_l1316_131664

/-- The circle with equation x^2 + y^2 - 6x = 0 -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 6*p.1 = 0}

/-- The point through which all lines must pass -/
def FixedPoint : ℝ × ℝ := (1, 2)

/-- A line passing through the FixedPoint -/
def LineThroughFixedPoint (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 - FixedPoint.2 = m * (p.1 - FixedPoint.1)}

/-- The length of the chord intercepted by the circle and a line -/
noncomputable def ChordLength (m : ℝ) : ℝ :=
  2 * Real.sqrt (9 - (2 * m / (1 + m^2))^2)

/-- The minimum chord length theorem -/
theorem min_chord_length :
  ∃ (min_length : ℝ), min_length = 2 ∧
  ∀ (m : ℝ), ChordLength m ≥ min_length := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_chord_length_l1316_131664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_passes_through_intersections_P_minimizes_distances_l1316_131681

-- Define the curve
def curve (x : ℝ) : ℝ := x^2 - 8*x + 2

-- Define the circle
def circle_equation (x y : ℝ) : Prop := (x - 4)^2 + (y - 1.5)^2 = 16.25

-- Define the line
def line (x y : ℝ) : Prop := x - y - 6 = 0

-- Define the center of the circle
noncomputable def C : ℝ × ℝ := (4, 1.5)

-- Define point D
noncomputable def D : ℝ × ℝ := (2, 1/2)

-- Define point P
noncomputable def P : ℝ × ℝ := (163/33, -35/33)

-- Theorem 1: The circle passes through the intersection points
theorem circle_passes_through_intersections :
  (∃ x, circle_equation x 0 ∧ curve x = 0) ∧
  (∃ y, circle_equation 0 y ∧ y = curve 0) := by
  sorry

-- Theorem 2: P minimizes the sum of distances
theorem P_minimizes_distances :
  ∀ x y, line x y →
    Real.sqrt ((x - C.1)^2 + (y - C.2)^2) +
    Real.sqrt ((x - D.1)^2 + (y - D.2)^2) ≥
    Real.sqrt ((P.1 - C.1)^2 + (P.2 - C.2)^2) +
    Real.sqrt ((P.1 - D.1)^2 + (P.2 - D.2)^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_passes_through_intersections_P_minimizes_distances_l1316_131681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hat_numbers_sum_l1316_131617

/-- Represents a four-digit perfect square number with tens digit 0 and non-zero units digit -/
structure FourDigitPerfectSquare where
  value : Nat
  is_four_digit : value ≥ 1000 ∧ value < 10000
  is_perfect_square : ∃ n, value = n * n
  tens_digit_zero : (value / 10) % 10 = 0
  units_digit_nonzero : value % 10 ≠ 0

/-- The set of all valid FourDigitPerfectSquare numbers -/
def ValidNumbers : Set FourDigitPerfectSquare := sorry

/-- Represents the hat numbers for A, B, and C -/
structure HatNumbers where
  a : FourDigitPerfectSquare
  b : FourDigitPerfectSquare
  c : FourDigitPerfectSquare
  in_valid_set : a ∈ ValidNumbers ∧ b ∈ ValidNumbers ∧ c ∈ ValidNumbers
  b_c_same_units : b.value % 10 = c.value % 10
  b_c_can_deduce : ∀ x y, x ∈ ValidNumbers → y ∈ ValidNumbers → x.value % 10 = y.value % 10 → x = b ∧ y = c
  a_can_deduce : ∀ x, x ∈ ValidNumbers → (∀ y z, y ∈ ValidNumbers → z ∈ ValidNumbers → y.value % 10 = z.value % 10 → y = b ∧ z = c) → x = a
  a_even_units : a.value % 2 = 0

theorem hat_numbers_sum (hn : HatNumbers) : hn.a.value + hn.b.value + hn.c.value = 14612 := by
  sorry

#eval 2704 + 2304 + 9604  -- To verify the sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hat_numbers_sum_l1316_131617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_x_gt_1_min_a_x_lt_1_l1316_131668

noncomputable def f (x : ℝ) : ℝ := 4 * x + 1 / (x - 1)

-- Theorem 1: Minimum value of f(x) when x > 1 is 8
theorem min_value_x_gt_1 : 
  ∀ x > 1, f x ≥ 8 ∧ ∃ x₀ > 1, f x₀ = 8 := by
  sorry

-- Theorem 2: Minimum value of a such that f(x) ≤ a when x < 1 is 0
theorem min_a_x_lt_1 : 
  (∃ a, ∀ x < 1, f x ≤ a) ∧ 
  (∀ a' < 0, ∃ x < 1, f x > a') := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_x_gt_1_min_a_x_lt_1_l1316_131668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_card_stack_arrangements_l1316_131696

/-- Represents a stack of cards -/
def CardStack (n : ℕ) := Fin (2 * n + 1) → ℕ

/-- Operation 1: Move cards from top to bottom -/
def moveTopToBottom (s : CardStack n) (k : ℕ) : CardStack n :=
  sorry

/-- Operation 2: Insert top n cards into gaps of bottom n+1 cards -/
def insertTopIntoBottom (s : CardStack n) : CardStack n :=
  sorry

/-- Predicate to check if a stack can be obtained from the initial stack -/
def isReachable (n : ℕ) (s : CardStack n) : Prop :=
  sorry

/-- Theorem: The number of reachable card arrangements is at most 2n(2n+1) -/
theorem card_stack_arrangements (n : ℕ) :
  ∃ (C : Finset (CardStack n)), (∀ s ∈ C, isReachable n s) ∧ C.card ≤ 2 * n * (2 * n + 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_card_stack_arrangements_l1316_131696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1316_131697

-- Define the function f(x) as noncomputable due to the use of Real.sqrt
noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.sqrt 3 * Real.cos x

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), (∀ x, f x ≤ M) ∧ (∃ x, f x = M) ∧ (M = 2) := by
  -- The proof is skipped using 'sorry'
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1316_131697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_symmetry_l1316_131652

noncomputable def f (x : ℝ) : ℝ := Real.tan (4 * x + Real.pi / 3) + 2

theorem center_of_symmetry (k : ℤ) :
  let x := -Real.pi/12 + k*Real.pi/8
  let y := 2
  ∃ (ε : ℝ), ε > 0 ∧ ∀ (t : ℝ), 0 < |t| ∧ |t| < ε →
    f (x + t) - y = -(f (x - t) - y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_symmetry_l1316_131652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_cosine_inequality_l1316_131615

theorem negation_of_cosine_inequality :
  (¬ ∀ x : ℝ, Real.cos x ≤ 1) ↔ (∃ x : ℝ, Real.cos x = 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_cosine_inequality_l1316_131615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_numbers_sets_not_equal_l1316_131694

theorem distinct_numbers_sets_not_equal (S : Finset ℝ) (h : S.card = 10) :
  let vasya_set := {x | ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ x = (a - b)^2}
  let petya_set := {x | ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ x = |a^2 - b^2|}
  vasya_set ≠ petya_set :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_numbers_sets_not_equal_l1316_131694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_sum_minus_products_l1316_131633

theorem square_sum_minus_products (x : ℝ) : 
  (x + 19)^2 + (x + 20)^2 + (x + 21)^2 - (x + 19)*(x + 20) - (x + 20)*(x + 21) - (x + 19)*(x + 21) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_sum_minus_products_l1316_131633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_ln_x_ln_one_minus_x_l1316_131699

theorem integral_ln_x_ln_one_minus_x : 
  ∫ (x : ℝ) in Set.Ioo 0 1, Real.log x * Real.log (1 - x) = 2 - Real.pi^2 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_ln_x_ln_one_minus_x_l1316_131699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_ngon_area_formula_l1316_131605

/-- A convex n-hedral angle cutting out a spherical n-gon from a sphere -/
structure SphericalNGon where
  n : ℕ
  R : ℝ
  σ : ℝ
  h_n : n ≥ 3
  h_R : R > 0
  h_σ : σ > 0

/-- The area of a spherical n-gon -/
noncomputable def sphericalNGonArea (S : SphericalNGon) : ℝ :=
  S.R^2 * (S.σ - (S.n - 2 : ℝ) * Real.pi)

/-- Theorem: The area of a spherical n-gon is R^2(σ - (n-2)π) -/
theorem spherical_ngon_area_formula (S : SphericalNGon) :
  sphericalNGonArea S = S.R^2 * (S.σ - (S.n - 2 : ℝ) * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_ngon_area_formula_l1316_131605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_chords_theorem_l1316_131600

/-- A type representing a point on the circle -/
def Point := Fin (2^500)

/-- A type representing a chord between two points -/
structure Chord where
  start : Point
  finish : Point

/-- The sum of the labels of the endpoints of a chord -/
def chord_sum (c : Chord) : ℕ := c.start.val + c.finish.val

/-- A predicate to check if two chords are disjoint -/
def disjoint (c1 c2 : Chord) : Prop :=
  c1.start ≠ c2.start ∧ c1.start ≠ c2.finish ∧
  c1.finish ≠ c2.start ∧ c1.finish ≠ c2.finish

theorem circle_chords_theorem :
  ∃ (chords : Finset Chord),
    chords.card = 100 ∧
    (∀ c1 c2, c1 ∈ chords → c2 ∈ chords → c1 ≠ c2 → disjoint c1 c2) ∧
    (∃ k : ℕ, ∀ c ∈ chords, chord_sum c = k) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_chords_theorem_l1316_131600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_solutions_1800_l1316_131622

def sum_of_fours_and_fives (n m : ℕ) : ℕ := 4 * n + 5 * m

theorem count_solutions_1800 : 
  (Finset.filter (fun p : ℕ × ℕ => sum_of_fours_and_fives p.1 p.2 = 1800) (Finset.product (Finset.range 451) (Finset.range 361))).card = 91 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_solutions_1800_l1316_131622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_sum_greater_than_sqrt_two_l1316_131665

noncomputable def f (x : ℝ) : ℝ := Real.log x

def g (a x : ℝ) : ℝ := x^2 + a

noncomputable def F (a x : ℝ) : ℝ := f x - g a x

theorem zeros_sum_greater_than_sqrt_two (a : ℝ) (x₁ x₂ : ℝ) 
  (h1 : x₁ < x₂) 
  (h2 : F a x₁ = 0) 
  (h3 : F a x₂ = 0) : 
  x₁ + x₂ > Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_sum_greater_than_sqrt_two_l1316_131665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_l1316_131693

/-- Calculates the future value of an investment with compound interest -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) (frequency : ℕ) : ℝ :=
  principal * (1 + rate / (frequency : ℝ)) ^ ((frequency : ℝ) * (time : ℝ))

/-- The problem statement -/
theorem investment_growth : 
  let principal := (5000 : ℝ)
  let rate := (0.10 : ℝ)
  let time := (2 : ℕ)
  let frequency := (1 : ℕ)
  compound_interest principal rate time frequency = 6050 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_l1316_131693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_quadrants_and_terminal_sides_l1316_131679

-- Define a function to determine the quadrant of an angle
noncomputable def quadrant (angle : ℝ) : ℕ :=
  let normalized_angle := angle % 360
  if 0 ≤ normalized_angle && normalized_angle < 90 then 1
  else if 90 ≤ normalized_angle && normalized_angle < 180 then 2
  else if 180 ≤ normalized_angle && normalized_angle < 270 then 3
  else 4

-- Define a function to generate the set of angles with the same terminal side
def same_terminal_side (angle : ℝ) : Set ℝ :=
  {α : ℝ | ∃ k : ℤ, α = angle + k * 360}

theorem angle_quadrants_and_terminal_sides :
  (quadrant 606 = 3) ∧
  (quadrant (-950) = 2) ∧
  (same_terminal_side (-457) = {α : ℝ | ∃ k : ℤ, α = -457 + k * 360}) ∧
  (quadrant (-457 + 360) = 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_quadrants_and_terminal_sides_l1316_131679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_not_less_than_two_l1316_131623

theorem at_least_one_not_less_than_two 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  ∃ x ∈ ({a + 1/b, b + 1/c, c + 1/a} : Set ℝ), x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_not_less_than_two_l1316_131623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_beautiful_ratio_l1316_131684

/-- The beautiful ratio of an isosceles triangle -/
noncomputable def beautiful_ratio (base leg : ℝ) : ℝ := base / leg

theorem isosceles_triangle_beautiful_ratio :
  ∀ (base leg : ℝ),
  base > 0 ∧ leg > 0 →
  base + 2 * leg = 20 →
  (base = 8 ∨ leg = 8) →
  beautiful_ratio base leg = 4/3 ∨ beautiful_ratio base leg = 1/2 :=
by
  intro base leg h1 h2 h3
  sorry

#check isosceles_triangle_beautiful_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_beautiful_ratio_l1316_131684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_multiple_l1316_131650

theorem smallest_multiple : ∃ x : ℕ+, x = 8 ∧ 
  (∀ y : ℕ+, (450 * y.val) % 720 = 0 → x.val ≤ y.val) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_multiple_l1316_131650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_with_special_angles_and_median_l1316_131655

theorem triangle_area_with_special_angles_and_median 
  (A B C : ℝ) (a b c : ℝ) (AM : ℝ) :
  -- Conditions
  Real.sin A = Real.sin B ∧ 
  Real.sin A = -Real.cos C ∧ 
  AM = Real.sqrt 7 ∧
  -- Triangle inequality
  a + b > c ∧ b + c > a ∧ c + a > b ∧
  -- Positive side lengths
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- Conclusion
  (1/2 : ℝ) * a * b * Real.sin C = Real.sqrt 3 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_with_special_angles_and_median_l1316_131655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_properties_l1316_131608

/-- Given n = 14, prove that the binomial coefficients of the 9th, 10th, and 11th terms
    in the expansion of (√x + 3x)^n form an arithmetic sequence, and that the only
    rational terms in the expansion are x^7, 164x^6, and 91x^5. -/
theorem expansion_properties (x : ℝ) : 
  let n : ℕ := 14
  let coeff (k : ℕ) := Nat.choose n k
  let is_rational (r : ℕ) := r % 6 = 0 ∧ r ≤ n
  -- Binomial coefficients form an arithmetic sequence
  coeff 8 + coeff 10 = 2 * coeff 9 ∧ 
  -- Only rational terms
  (∀ r, is_rational r ↔ r = 0 ∨ r = 6 ∨ r = 12) ∧
  -- Coefficients of rational terms
  coeff 0 = 1 ∧ coeff 6 = 164 ∧ coeff 12 = 91 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_properties_l1316_131608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lightning_distance_estimate_l1316_131690

/-- The speed of sound in feet per second -/
def speed_of_sound : ℚ := 1100

/-- The number of feet in a kilometer -/
def feet_per_km : ℚ := 3280

/-- The time between lightning flash and thunder in seconds -/
def time_delay : ℚ := 15

/-- Rounds a rational number to the nearest quarter -/
def round_to_nearest_quarter (x : ℚ) : ℚ :=
  (⌊x * 4 + 1/2⌋ / 4)

/-- The distance between Linus and the lightning strike in kilometers -/
def distance_km : ℚ :=
  (speed_of_sound * time_delay) / feet_per_km

theorem lightning_distance_estimate :
  round_to_nearest_quarter distance_km = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lightning_distance_estimate_l1316_131690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_periodic_l1316_131669

noncomputable def f (x : ℝ) := Real.sin x * Real.cos (2 * x)

theorem f_odd_and_periodic :
  (∀ x, f (-x) = -f x) ∧ 
  (∃ T > 0, ∀ x, f (x + T) = f x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_periodic_l1316_131669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l1316_131678

def mySequence : List ℕ := [2, 5, 11, 20, 32, 47]

def differences (s : List ℕ) : List ℕ :=
  List.zipWith (·-·) (s.tail) s

def arithmetic_progression (l : List ℕ) : Prop :=
  ∀ i, i + 2 < l.length → l[i+1]! - l[i]! = l[i]! - l[i-1]! + (l[1]! - l[0]!)

theorem sequence_property : 
  arithmetic_progression (differences mySequence) ∧ 
  mySequence[4]! = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l1316_131678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_syllogism_arrangement_l1316_131688

/-- Represents a student -/
structure Student where
  name : String

/-- Represents a class in the second year of high school -/
structure HighSchoolClass where
  name : String

/-- Predicate to check if a student is in a specific class -/
def is_in_class (s : Student) (c : HighSchoolClass) : Prop := sorry

/-- Predicate to check if a student is an only child -/
def is_only_child (s : Student) : Prop := sorry

/-- The specific class mentioned in the problem -/
def class_21 : HighSchoolClass := { name := "Class 21 of the second year of high school" }

/-- The specific student mentioned in the problem -/
def an_mengyi : Student := { name := "An Mengyi" }

theorem syllogism_arrangement :
  (∀ (s : Student), is_in_class s class_21 → is_only_child s) ∧ 
  is_in_class an_mengyi class_21 ∧
  is_only_child an_mengyi :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_syllogism_arrangement_l1316_131688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tea_box_solution_l1316_131692

/-- Represents the number of cups of tea that can be brewed from one tea bag -/
inductive CupsPerBag
  | two
  | three

/-- Represents a person who brewed tea -/
structure TeaBrewer where
  name : String
  cups_brewed : ℕ

/-- Represents a box of tea bags -/
structure TeaBox where
  total_bags : ℕ
  brewers : List TeaBrewer

/-- Checks if the given number of tea bags can brew the specified number of cups -/
def can_brew (bags : ℕ) (cups : ℕ) : Prop :=
  ∃ (two three : ℕ), two + three = bags ∧ 2 * two + 3 * three = cups

theorem tea_box_solution (box : TeaBox) : 
  box.brewers.length = 2 ∧ 
  (∀ b ∈ box.brewers, can_brew (box.total_bags / 2) b.cups_brewed) ∧
  (∃ b ∈ box.brewers, b.cups_brewed = 57) ∧
  (∃ b ∈ box.brewers, b.cups_brewed = 83) →
  box.total_bags = 56 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tea_box_solution_l1316_131692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_daily_sales_extrema_l1316_131612

/-- Sales volume function -/
noncomputable def g (t : ℝ) : ℝ := 80 - 2*t

/-- Price function -/
noncomputable def f (t : ℝ) : ℝ := 20 - (1/2) * |t - 10|

/-- Daily sales function -/
noncomputable def y (t : ℝ) : ℝ := g t * f t

theorem daily_sales_extrema :
  (∀ t : ℝ, 0 ≤ t ∧ t ≤ 20 → y t ≤ 1225) ∧
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 20 ∧ y t = 1225) ∧
  (∀ t : ℝ, 0 ≤ t ∧ t ≤ 20 → 600 ≤ y t) ∧
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 20 ∧ y t = 600) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_daily_sales_extrema_l1316_131612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_specific_points_l1316_131680

/-- The slope angle of a line passing through two points -/
noncomputable def slope_angle (x1 y1 x2 y2 : ℝ) : ℝ := 
  Real.arctan ((y2 - y1) / (x2 - x1)) * (180 / Real.pi)

/-- Theorem: The slope angle of the line passing through (2,3) and (4,5) is 45° -/
theorem slope_angle_specific_points : 
  slope_angle 2 3 4 5 = 45 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_specific_points_l1316_131680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integer_inequality_l1316_131619

theorem consecutive_integer_inequality (k : ℕ) : 
  ∃ d ∈ Finset.range 20, 
    d + k * 20 > 0 ∧ 
    (d + k * 20) % 20 = 15 ∧ 
    ∀ n : ℕ, n > 0 → 
      n * Real.sqrt (d + k * 20 : ℝ) * 
      (n * Real.sqrt (d + k * 20 : ℝ) - ⌊n * Real.sqrt (d + k * 20 : ℝ)⌋) > 
      (5 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integer_inequality_l1316_131619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_values_l1316_131611

/-- The function f(x) with parameters a and b -/
def f (a b x : ℝ) : ℝ := x^3 - a*x^2 - b*x + a^2

/-- Theorem stating that if f(x) has an extremum of 10 at x = 1, then a = -4 and b = 11 -/
theorem extremum_values (a b : ℝ) :
  (∃ (ε : ℝ → ℝ), (∀ x, x ≠ 1 → |x - 1| < ε x → |f a b x - f a b 1| ≤ |f a b x - 10|) ∧
                   (∀ δ > 0, ∃ x, |x - 1| < δ ∧ f a b x = f a b 1)) →
  f a b 1 = 10 →
  a = -4 ∧ b = 11 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_values_l1316_131611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_perfect_square_sum_subset_l1316_131621

/-- The set A of positive integers of the form 3 * 15^n for n ∈ ℕ -/
def A : Set ℕ := {x | ∃ n : ℕ, x = 3 * 15^n}

/-- A function that checks if a natural number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

/-- Main theorem: No finite subset of A sums to a perfect square -/
theorem no_perfect_square_sum_subset (S : Finset ℕ) (hS : ∀ x ∈ S, x ∈ A) : 
  ¬ isPerfectSquare (S.sum id) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_perfect_square_sum_subset_l1316_131621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_distance_l1316_131689

-- Define the distances
def AD : ℝ := 4000
def AB : ℝ := 4500

-- Define the function to calculate BD using Pythagorean theorem
noncomputable def BD : ℝ := Real.sqrt (AB^2 - AD^2)

-- Define the total distance function
noncomputable def totalDistance : ℝ := AB + BD + AD

-- Theorem statement
theorem trip_distance : totalDistance = 8500 + 50 * Real.sqrt 1700 := by
  -- Unfold the definitions
  unfold totalDistance BD
  -- Simplify the expression
  simp [AD, AB]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_distance_l1316_131689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_implies_q_l1316_131660

/-- The area of a triangle given the coordinates of its vertices --/
noncomputable def triangleArea (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1/2) * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

/-- Theorem stating that if the area of triangle ABC is 36, where A(3, 15), B(15, 0), and C(0, q), then q = 12.75 --/
theorem area_implies_q (q : ℝ) : triangleArea 3 15 15 0 0 q = 36 → q = 12.75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_implies_q_l1316_131660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_n_l1316_131695

def discrete_random_variable (n : ℕ+) : Type :=
  Fin n → ℝ

def equally_likely {n : ℕ+} (X : discrete_random_variable n) : Prop :=
  ∀ i : Fin n, X i = 1 / n

theorem find_n {n : ℕ+} (X : discrete_random_variable n) 
  (h1 : equally_likely X) 
  (h2 : (X 0 + X 1 + X 2) = 1 / 5) : 
  n = 15 := by
  sorry

#check find_n

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_n_l1316_131695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_prism_surface_area_l1316_131674

/-- A right prism with a square base -/
structure RightPrism where
  basePerimeter : ℝ
  height : ℝ

/-- The surface area of a right prism with a square base -/
noncomputable def surfaceArea (prism : RightPrism) : ℝ :=
  let baseSideLength := prism.basePerimeter / 4
  let baseArea := baseSideLength * baseSideLength
  let lateralArea := prism.basePerimeter * prism.height
  2 * baseArea + lateralArea

/-- Theorem: The surface area of a right prism with a square base,
    perimeter 4, and height 2 is 10 -/
theorem right_prism_surface_area :
  ∀ (prism : RightPrism),
    prism.basePerimeter = 4 ∧ prism.height = 2 →
    surfaceArea prism = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_prism_surface_area_l1316_131674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bottle_game_player1_win_prob_l1316_131647

/-- Represents a turn-based game with two players -/
structure Game where
  player1_win_prob : ℚ
  player2_win_prob : ℚ

/-- Calculates the probability of player 1 winning the game -/
noncomputable def player1_win_probability (g : Game) : ℚ :=
  g.player1_win_prob / (1 - (1 - g.player1_win_prob) * (1 - g.player2_win_prob))

/-- The specific game described in the problem -/
def bottle_game : Game :=
  { player1_win_prob := 2/3
  , player2_win_prob := 1/2 }

theorem bottle_game_player1_win_prob :
  player1_win_probability bottle_game = 4/5 := by
  sorry

-- Use #eval only for computable expressions
example : bottle_game.player1_win_prob = 2/3 := rfl
example : bottle_game.player2_win_prob = 1/2 := rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bottle_game_player1_win_prob_l1316_131647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_locus_l1316_131629

theorem circle_tangency_locus :
  let C1 := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}
  let C3 := {p : ℝ × ℝ | (p.1 - 3)^2 + p.2^2 = 25}
  let locus := {p : ℝ × ℝ | ∃ r : ℝ,
    (∀ q ∈ C1, (q.1 - p.1)^2 + (q.2 - p.2)^2 ≥ r^2 ∧ ∃ q₀ ∈ C1, (q₀.1 - p.1)^2 + (q₀.2 - p.2)^2 = r^2) ∧
    (∀ q ∈ C3, (q.1 - p.1)^2 + (q.2 - p.2)^2 ≤ r^2 ∧ ∃ q₁ ∈ C3, (q₁.1 - p.1)^2 + (q₁.2 - p.2)^2 = r^2)}
  ∀ p ∈ locus, 3 * p.1^2 + 4 * p.2^2 - 14 * p.1 - 49 = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_locus_l1316_131629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_statements_imply_negation_l1316_131603

-- Define the propositions r and s
variable (r s : Prop)

-- Define the four statements
def statement1 (r s : Prop) : Prop := ¬r ∧ ¬s
def statement2 (r s : Prop) : Prop := ¬r ∧ s
def statement3 (r s : Prop) : Prop := r ∧ ¬s
def statement4 (r s : Prop) : Prop := r ∧ s

-- Define the negation of "r and s are both false"
def negation (r s : Prop) : Prop := ¬(¬r ∧ ¬s)

-- Define a function that checks if a statement implies the negation
def implies_negation (statement negation : Prop) : Prop := statement → negation

-- Theorem: Exactly 3 out of 4 statements imply the negation
theorem three_statements_imply_negation (r s : Prop) :
  (implies_negation (statement1 r s) (negation r s) → False) ∧
  (implies_negation (statement2 r s) (negation r s)) ∧
  (implies_negation (statement3 r s) (negation r s)) ∧
  (implies_negation (statement4 r s) (negation r s)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_statements_imply_negation_l1316_131603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_5a_l1316_131614

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  let x' := x - 2 * (⌊x / 2⌋ : ℝ)
  if -1 ≤ x' ∧ x' < 0 then x' + a
  else if 0 ≤ x' ∧ x' < 1 then |2/5 - x'|
  else 0  -- This case should never occur due to the periodicity

theorem f_value_at_5a (a : ℝ) :
  (∀ x, f (x + 2) a = f x a) →  -- Period of 2
  f (-5/2) a = f (9/2) a →      -- Given condition
  f (5 * a) a = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_5a_l1316_131614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_l1316_131671

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 3)

theorem function_symmetry (ω : ℝ) (h1 : ω > 0) (h2 : ∀ x, f ω (x + Real.pi) = f ω x) :
  ∀ x, f ω (Real.pi / 6 - x) = f ω (Real.pi / 6 + x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_l1316_131671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_tangent_sum_min_value_is_2_sqrt_2_l1316_131638

open Real

theorem min_value_tangent_sum (α β : ℝ) 
  (h1 : 0 < α ∧ α < π/2) 
  (h2 : 0 < β ∧ β < π/2) 
  (h3 : sin α * cos β - 2 * cos α * sin β = 0) : 
  ∀ x y, 0 < x ∧ x < π/2 ∧ 0 < y ∧ y < π/2 ∧ sin x * cos y - 2 * cos x * sin y = 0 →
  tan (2*π + α) + tan (π/2 - β) ≤ tan (2*π + x) + tan (π/2 - y) :=
by sorry

theorem min_value_is_2_sqrt_2 (α β : ℝ) 
  (h1 : 0 < α ∧ α < π/2) 
  (h2 : 0 < β ∧ β < π/2) 
  (h3 : sin α * cos β - 2 * cos α * sin β = 0) : 
  tan (2*π + α) + tan (π/2 - β) = 2 * sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_tangent_sum_min_value_is_2_sqrt_2_l1316_131638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_radii_relation_l1316_131663

/-- A function representing the area of a triangle given its inradius and exradii. -/
noncomputable def area_of_triangle (r r₁ r₂ r₃ : ℝ) : ℝ :=
  Real.sqrt (r * r₁ * r₂ * r₃)

/-- For a triangle with area Q, inradius r, and exradii r₁, r₂, r₃, 
    the equation Q² = r r₁ r₂ r₃ holds. -/
theorem triangle_area_radii_relation (Q r r₁ r₂ r₃ : ℝ) 
    (hQ : Q > 0) 
    (hr : r > 0) 
    (hr₁ : r₁ > 0) 
    (hr₂ : r₂ > 0) 
    (hr₃ : r₃ > 0) 
    (h_area : Q = area_of_triangle r r₁ r₂ r₃) : 
  Q^2 = r * r₁ * r₂ * r₃ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_radii_relation_l1316_131663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equality_l1316_131683

-- Define the complex expression
noncomputable def complex_expression : ℂ := (1 - Complex.I * Real.sqrt 3) / ((Real.sqrt 3 + Complex.I)^2)

-- State the theorem
theorem complex_equality : complex_expression = -1/4 - (Real.sqrt 3)/4 * Complex.I := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equality_l1316_131683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1316_131649

-- Define set M
def M : Set ℝ := {x | x^2 + x - 6 < 0}

-- Define set N
def N : Set ℝ := {x | |x - 1| ≤ 2}

-- Theorem statement
theorem intersection_M_N :
  M ∩ N = Set.Icc (-1 : ℝ) 2 ∩ Set.Ioo (-1 : ℝ) 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1316_131649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_required_sleep_for_average_score_l1316_131643

/-- Represents a test result -/
structure TestResult where
  sleep : ℝ
  study : ℝ
  score : ℝ

/-- Calculates the required sleep for the second test -/
noncomputable def calculateRequiredSleep (test1 : TestResult) (study2 : ℝ) (targetAvg : ℝ) : ℝ :=
  let score2 := 2 * targetAvg - test1.score
  (test1.score * test1.sleep * study2) / (score2 * test1.study)

theorem required_sleep_for_average_score
  (test1 : TestResult)
  (study2 : ℝ)
  (targetAvg : ℝ)
  (h1 : test1.sleep = 6)
  (h2 : test1.study = 3)
  (h3 : test1.score = 60)
  (h4 : study2 = 5)
  (h5 : targetAvg = 75) :
  calculateRequiredSleep test1 study2 targetAvg = 2.4 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval calculateRequiredSleep ⟨6, 3, 60⟩ 5 75

end NUMINAMATH_CALUDE_ERRORFEEDBACK_required_sleep_for_average_score_l1316_131643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_beta_value_l1316_131659

theorem cos_beta_value (α β : Real) : 
  α ∈ Set.Ioo 0 π → 
  β ∈ Set.Ioo 0 π → 
  Real.sin (α + β) = 5/13 → 
  Real.tan (α/2) = 1/2 → 
  Real.cos β = -16/65 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_beta_value_l1316_131659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_headcount_rounded_l1316_131654

def fall_headcount_03_04 : ℕ := 11500
def fall_headcount_04_05 : ℕ := 11600
def fall_headcount_05_06 : ℕ := 11300

def average_headcount : ℚ :=
  (fall_headcount_03_04 + fall_headcount_04_05 + fall_headcount_05_06) / 3

theorem average_headcount_rounded : 
  Int.floor (average_headcount + 1/2) = 11467 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_headcount_rounded_l1316_131654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reaction_moles_equality_l1316_131657

/-- Represents the number of moles of a substance -/
def Moles : Type := ℝ

/-- Represents the reaction between HCH3CO2 and NaOH -/
def ReactionEquation (hch3co2 : Moles) (naoh : Moles) (h2o : Moles) : Prop :=
  hch3co2 = naoh ∧ hch3co2 = h2o

/-- Theorem: Given 1 mole of HCH3CO2 reacting to produce 1 mole of H2O, 
    the number of moles of NaOH combined is equal to 1 mole -/
theorem reaction_moles_equality 
  (hch3co2 : Moles) 
  (naoh : Moles) 
  (h2o : Moles) 
  (h_hch3co2 : hch3co2 = (1 : ℝ)) 
  (h_h2o : h2o = (1 : ℝ)) 
  (h_reaction : ReactionEquation hch3co2 naoh h2o) : 
  naoh = (1 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reaction_moles_equality_l1316_131657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_goods_train_speed_l1316_131631

/-- Proves that the speed of the goods train is 36 km/h given the problem conditions -/
theorem goods_train_speed (express_speed : ℝ) (time_difference : ℝ) (catch_up_time : ℝ) 
  (h1 : express_speed = 90)
  (h2 : time_difference = 6)
  (h3 : catch_up_time = 4) : 
  (express_speed * catch_up_time) / (catch_up_time + time_difference) = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_goods_train_speed_l1316_131631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_theorem_l1316_131620

/-- Calculates the time for a train to cross a man and exit a tunnel --/
noncomputable def train_crossing_time (train_length : ℝ) (initial_train_speed : ℝ) 
  (man_speed : ℝ) (tunnel_length : ℝ) (tunnel_train_speed : ℝ) : ℝ :=
  let relative_speed := initial_train_speed + man_speed
  let time_to_cross_man := train_length / relative_speed
  let time_in_tunnel := (train_length + tunnel_length) / tunnel_train_speed
  time_to_cross_man + time_in_tunnel

/-- The time for the train to cross the man and exit the tunnel is approximately 152.24 seconds --/
theorem train_crossing_theorem :
  ∃ ε > 0, |train_crossing_time 1200 (81000/3600) (5000/3600) 500 (60000/3600) - 152.24| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_theorem_l1316_131620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_point_l1316_131637

theorem angle_terminal_side_point (θ : ℝ) (t : ℝ) : 
  (Real.sin θ = t / Real.sqrt (4 + t^2) ∧ 
   Real.cos θ = -2 / Real.sqrt (4 + t^2) ∧
   Real.sin θ + Real.cos θ = Real.sqrt 5 / 5) →
  t = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_point_l1316_131637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_special_case_l1316_131670

/-- Represents a parabola in 2D space -/
structure Parabola where
  focus : ℝ × ℝ

/-- Represents an ellipse in 2D space -/
structure Ellipse where
  center : ℝ × ℝ
  semi_major_axis : ℝ
  semi_minor_axis : ℝ

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (e.semi_major_axis^2 - e.semi_minor_axis^2) / e.semi_major_axis

/-- Predicate to check if two points are the foci of an ellipse -/
def are_foci (e : Ellipse) (f1 f2 : ℝ × ℝ) : Prop := sorry

/-- Predicate to check if a point is on a parabola -/
def on_parabola (p : Parabola) (point : ℝ × ℝ) : Prop := sorry

/-- Function to count the number of intersections between a parabola and an ellipse -/
def num_intersections (p : Parabola) (e : Ellipse) : ℕ := sorry

/-- Theorem stating the eccentricity of an ellipse given specific conditions -/
theorem ellipse_eccentricity_special_case (p : Parabola) (e : Ellipse) 
  (h1 : p.focus = e.center)
  (h2 : ∃ (f1 f2 : ℝ × ℝ), are_foci e f1 f2 ∧ on_parabola p f1 ∧ on_parabola p f2)
  (h3 : num_intersections p e = 3) :
  eccentricity e = 2 * Real.sqrt 5 / 5 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_special_case_l1316_131670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1316_131685

noncomputable section

def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

def directrix : ℝ → Prop :=
  λ x ↦ x = -1

def origin : ℝ × ℝ :=
  (0, 0)

noncomputable def triangle_area (A B : ℝ × ℝ) : ℝ :=
  abs ((A.1 * B.2 - A.2 * B.1) / 2)

def eccentricity (a c : ℝ) : ℝ :=
  c / a

theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (A B : ℝ × ℝ) (hA : directrix A.1) (hB : directrix B.1)
  (h_area : triangle_area A B = 2 * Real.sqrt 3)
  (h_asymptote : ∃ (x y : ℝ), hyperbola a b x y ∧ 
    ((x, y) = A ∨ (x, y) = B)) :
  eccentricity a (Real.sqrt (a^2 + b^2)) = Real.sqrt 13 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1316_131685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_umbrellas_l1316_131626

/-- The number of umbrellas in John's house -/
def umbrellas_in_house : ℕ := 2

/-- The total number of umbrellas John owns -/
def total_umbrellas : ℕ := umbrellas_in_house + 1

/-- The cost of each umbrella in dollars -/
def umbrella_cost : ℕ := 8

/-- The total cost of all umbrellas in dollars -/
def total_cost : ℕ := 24

theorem johns_umbrellas : umbrellas_in_house = 2 := by
  rfl

#check johns_umbrellas

end NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_umbrellas_l1316_131626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_pipe_fill_time_l1316_131667

/-- Represents the time taken to fill a pool using two pipes -/
noncomputable def fill_time (t1 t2 : ℝ) : ℝ :=
  1 / (1 / t1 + 1 / t2)

/-- Theorem stating that two pipes filling a pool in 8 and 12 hours respectively
    will fill the pool in 4.8 hours when used together -/
theorem two_pipe_fill_time :
  fill_time 8 12 = 4.8 := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_pipe_fill_time_l1316_131667
