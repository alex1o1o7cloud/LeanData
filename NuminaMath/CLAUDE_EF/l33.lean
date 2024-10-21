import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_batch_size_l33_3369

/-- Represents the cost function for shipping and storage fees -/
def cost_function (x : ℕ) : ℝ := 4 + 4 * (x : ℝ)

/-- Represents the total number of desks to be purchased -/
def total_desks : ℕ := 36

/-- Represents the available funds for shipping and storage -/
def available_funds : ℝ := 48

/-- Theorem stating that there exists a positive integer x ≤ 36 such that the cost is within the available funds -/
theorem exists_valid_batch_size :
  ∃ x : ℕ, 0 < x ∧ x ≤ total_desks ∧ cost_function x ≤ available_funds :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_batch_size_l33_3369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l33_3347

/-- A function f(x) with parameter ω -/
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 
  Real.sin (ω * x)^2 + Real.sin (ω * x) * Real.cos (ω * x) - 1

/-- The theorem stating the range of ω given the conditions -/
theorem omega_range (ω : ℝ) (h1 : ω > 0) 
  (h2 : ∃ (s : Finset ℝ), s.card = 4 ∧ (∀ x ∈ s, 0 < x ∧ x < Real.pi / 2 ∧ f ω x = 0) ∧
                           (∀ x, 0 < x → x < Real.pi / 2 → f ω x = 0 → x ∈ s)) :
  3 < ω ∧ ω ≤ 9 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l33_3347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triple_sum_value_l33_3381

/-- The sum of 1 / (3^a * 4^b * 6^c) over all positive integer triples (a, b, c) where 1 ≤ a < b < c -/
noncomputable def triple_sum : ℝ :=
  ∑' (a : ℕ), ∑' (b : ℕ), ∑' (c : ℕ), if 1 ≤ a ∧ a < b ∧ b < c then 1 / (3^a * 4^b * 6^c) else 0

/-- The triple sum equals 1/1265 -/
theorem triple_sum_value : triple_sum = 1 / 1265 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triple_sum_value_l33_3381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_time_typing_cost_is_five_l33_3374

/-- Represents a typing service with a fixed cost for first-time typing and revisions. -/
structure TypingService where
  firstTypeCost : ℚ
  revisionCost : ℚ

/-- Represents a manuscript with a total number of pages and revision information. -/
structure Manuscript where
  totalPages : ℕ
  revisedOnce : ℕ
  revisedTwice : ℕ

/-- Calculates the total cost for typing and revising a manuscript. -/
def totalCost (service : TypingService) (manuscript : Manuscript) : ℚ :=
  service.firstTypeCost * manuscript.totalPages +
  service.revisionCost * (manuscript.revisedOnce + 2 * manuscript.revisedTwice)

/-- Theorem stating that the first-time typing cost is $5 given the problem conditions. -/
theorem first_time_typing_cost_is_five :
  ∃ (service : TypingService) (manuscript : Manuscript),
    service.revisionCost = 4 ∧
    manuscript.totalPages = 100 ∧
    manuscript.revisedOnce = 30 ∧
    manuscript.revisedTwice = 20 ∧
    totalCost service manuscript = 780 ∧
    service.firstTypeCost = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_time_typing_cost_is_five_l33_3374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_third_quadrant_l33_3338

noncomputable def z : ℂ := (2 + Complex.I) / (Complex.I^5 - 1)

theorem z_in_third_quadrant : 
  (z.re < 0) ∧ (z.im < 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_third_quadrant_l33_3338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_formula_l33_3359

/-- A regular quadrilateral pyramid -/
structure RegularQuadrilateralPyramid where
  /-- Distance from the center of the base to a lateral face -/
  a : ℝ
  /-- Distance from the center of the base to a lateral edge -/
  b : ℝ
  /-- Assumption that a and b are positive -/
  a_pos : a > 0
  b_pos : b > 0
  /-- Assumption that 2a² > b² (needed for the expression under the square root to be non-negative) -/
  two_a_sq_gt_b_sq : 2 * a^2 > b^2

/-- The dihedral angle at the base of a regular quadrilateral pyramid -/
noncomputable def dihedralAngle (p : RegularQuadrilateralPyramid) : ℝ :=
  Real.arccos ((Real.sqrt (2 * p.a^2 - p.b^2)) / p.b)

/-- Theorem: The dihedral angle at the base of a regular quadrilateral pyramid
    is equal to arccos((√(2a² - b²))/b) -/
theorem dihedral_angle_formula (p : RegularQuadrilateralPyramid) :
  dihedralAngle p = Real.arccos ((Real.sqrt (2 * p.a^2 - p.b^2)) / p.b) := by
  -- The proof is skipped for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_formula_l33_3359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parking_lot_vehicles_l33_3380

/-- A type representing vehicles --/
structure Vehicle where
  mk :: -- Constructor

/-- The set of vehicles in the parking lot --/
def set_of_vehicles : Set Vehicle := sorry

/-- Predicate indicating if a vehicle has a spare tire --/
def has_spare_tire : Vehicle → Prop := sorry

/-- Predicate indicating if a vehicle is four-wheeled --/
def is_four_wheeled : Vehicle → Prop := sorry

/-- Theorem about the vehicles in the parking lot --/
theorem parking_lot_vehicles (num_vehicles : ℕ) (total_tires : ℕ) :
  num_vehicles = 30 →
  total_tires = 150 →
  (∀ v, v ∈ set_of_vehicles → has_spare_tire v) →
  (∀ v, v ∈ set_of_vehicles → is_four_wheeled v) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parking_lot_vehicles_l33_3380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_expression_l33_3315

theorem largest_prime_factor_of_expression : 
  (Nat.factors (12^3 + 15^4 - 6^5)).maximum? = some 97 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_expression_l33_3315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l33_3322

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1) - x

-- State the theorem
theorem f_properties :
  ∀ x : ℝ, x > -1 →
    (∀ y : ℝ, y > 0 → (deriv f) y < 0) ∧
    (1 - 1 / (x + 1) ≤ Real.log (x + 1) ∧ Real.log (x + 1) ≤ x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l33_3322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_isosceles_triangles_l33_3308

-- Define the lines
def line1 (k : ℝ) (x y : ℝ) : Prop := x - 2*y = -k + 6
def line2 (k : ℝ) (x y : ℝ) : Prop := x + 3*y = 4*k + 1

-- Define the intersection point
def intersection (k : ℝ) : ℝ × ℝ := (k + 4, k - 1)

-- Define the fourth quadrant
def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

-- Define point A
def point_A : ℝ × ℝ := (2, 0)

-- Define point O (origin)
def point_O : ℝ × ℝ := (0, 0)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define an isosceles triangle
def isosceles_triangle (p1 p2 p3 : ℝ × ℝ) : Prop :=
  distance p1 p2 = distance p1 p3 ∨ distance p1 p2 = distance p2 p3 ∨ distance p1 p3 = distance p2 p3

-- Theorem statement
theorem intersection_and_isosceles_triangles :
  ∀ k : ℕ,
  (∀ x y : ℝ, line1 k x y ∧ line2 k x y → fourth_quadrant x y) →
  ((-4 : ℝ) < k ∧ k < 1) ∧
  (k = 0 →
    (isosceles_triangle point_O point_A (1, -5/2) ∧ line1 k 1 (-5/2)) ∧
    (isosceles_triangle point_O point_A (2, -2) ∧ line1 k 2 (-2)) ∧
    (isosceles_triangle point_O point_A (18/5, -6/5) ∧ line1 k (18/5) (-6/5))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_isosceles_triangles_l33_3308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l33_3345

noncomputable section

variable (A B C a b c : ℝ)

-- Define the triangle ABC
def triangle_ABC (A B C a b c : ℝ) : Prop :=
  0 < A ∧ A < Real.pi ∧
  0 < B ∧ B < Real.pi ∧
  0 < C ∧ C < Real.pi ∧
  A + B + C = Real.pi ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a = 2 * Real.sin C ∧
  b = 2 * Real.sin A ∧
  c = 2 * Real.sin B

-- Define the given conditions
def given_conditions (A a c : ℝ) : Prop :=
  2 * (Real.cos A)^2 - 2 * Real.sqrt 3 * Real.sin A * Real.cos A = -1 ∧
  a = 2 * Real.sqrt 3 ∧
  c = 2

-- Theorem statement
theorem triangle_problem (A B C a b c : ℝ) 
  (h_triangle : triangle_ABC A B C a b c)
  (h_conditions : given_conditions A a c) :
  (1/2 * b * c * Real.sin A = 2 * Real.sqrt 3) ∧
  ((b - 2*c) / (a * Real.cos (Real.pi/3 + C)) = 2) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l33_3345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l33_3350

open Real

-- Define the point through which the terminal side of angle α passes
noncomputable def point : ℝ × ℝ := (2 * sin (π / 3), -2 * cos (π / 3))

-- Theorem statement
theorem sin_alpha_value (α : ℝ) : 
  (∃ r : ℝ, r > 0 ∧ r * cos α = point.1 ∧ r * sin α = point.2) → 
  sin α = -1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l33_3350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_davids_english_marks_l33_3378

theorem davids_english_marks :
  let math_marks : ℕ := 35
  let physics_marks : ℕ := 42
  let chemistry_marks : ℕ := 57
  let biology_marks : ℕ := 55
  let total_subjects : ℕ := 5
  let average_marks : ℕ := 45
  let total_marks : ℕ := average_marks * total_subjects
  let english_marks : ℕ := total_marks - (math_marks + physics_marks + chemistry_marks + biology_marks)
  english_marks = 36 := by
  -- Proof goes here
  sorry

#check davids_english_marks

end NUMINAMATH_CALUDE_ERRORFEEDBACK_davids_english_marks_l33_3378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l33_3367

noncomputable def f (x : ℝ) : ℝ := 
  (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5) * (x - 2) * (x - 1) / 
  ((x - 2) * (x - 4) * (x - 2))

theorem solution_set : 
  {x : ℝ | f x = 1 ∧ x ≠ 2 ∧ x ≠ 4} = {0, 3, 5} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l33_3367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_height_of_specific_cone_l33_3373

/-- A right cone with given base diameter and slant height -/
structure RightCone where
  base_diameter : ℝ
  slant_height : ℝ

/-- The height of a right cone -/
noncomputable def cone_height (c : RightCone) : ℝ :=
  Real.sqrt (c.slant_height ^ 2 - (c.base_diameter / 2) ^ 2)

/-- Theorem stating that a right cone with base diameter 8 and slant height 5 has height 3 -/
theorem height_of_specific_cone :
  let c : RightCone := { base_diameter := 8, slant_height := 5 }
  cone_height c = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_height_of_specific_cone_l33_3373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_after_multiplication_l33_3382

theorem ratio_after_multiplication (x y : ℝ) (h : x / y = 7 / 5) :
  (x * y) / (y * x) = 1 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_after_multiplication_l33_3382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l33_3335

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * (Real.sin x) ^ 2 - Real.sin (2 * x - Real.pi / 3)

theorem f_properties :
  -- The smallest positive period of f is π
  (∃ T > 0, ∀ x, f (x + T) = f x) ∧ (∀ T > 0, (∀ x, f (x + T) = f x) → T ≥ Real.pi) ∧
  -- If f(α/2) = 1/2 + √3 for α ∈ (0, π), then sin(α) = 1/2
  (∀ α, 0 < α ∧ α < Real.pi → f (α / 2) = 1 / 2 + Real.sqrt 3 → Real.sin α = 1 / 2) ∧
  -- The maximum value of f(x) for x ∈ [-π/2, 0] is √3 + 1
  (∀ x, -Real.pi / 2 ≤ x ∧ x ≤ 0 → f x ≤ Real.sqrt 3 + 1) ∧
  (∃ x, -Real.pi / 2 ≤ x ∧ x ≤ 0 ∧ f x = Real.sqrt 3 + 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l33_3335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_distance_proof_l33_3330

-- Define the curve C₁
noncomputable def C₁ (α : ℝ) : ℝ × ℝ := (2 + 2 * Real.cos α, 2 * Real.sin α)

-- Define the relation between M and P
def M_P_relation (M P : ℝ × ℝ) : Prop := M = (2 * P.1, 2 * P.2)

-- Define the curve C₂
def C₂ (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1 ∧ 0 < y ∧ y ≤ 1

-- Define points A and B
noncomputable def A : ℝ × ℝ := (2, 2 * Real.sqrt 3 / 2)
noncomputable def B : ℝ × ℝ := (1, Real.sqrt 3 / 2)

-- Theorem statement
theorem curve_and_distance_proof :
  (∀ α ∈ Set.Ioo 0 π, ∃ P, M_P_relation (C₁ α) P ∧ C₂ P.1 P.2) ∧
  (Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_distance_proof_l33_3330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_difference_is_pi_over_two_l33_3302

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x) - 2 * (Real.cos x) ^ 2

noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (6 * x - Real.pi / 6)

theorem x_difference_is_pi_over_two (x1 x2 : ℝ) :
  g x1 * g x2 = -4 → |x1 - x2| = Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_difference_is_pi_over_two_l33_3302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roger_price_is_correct_cookie_pricing_theorem_l33_3366

/-- Represents the cookie production and pricing scenario --/
structure CookieScenario where
  art_small_base : ℚ
  art_large_base : ℚ
  art_height : ℚ
  art_cookies_per_batch : ℕ
  art_price_per_cookie : ℚ
  roger_cookies_per_batch : ℕ
  total_dough : ℚ

/-- Calculates the price Roger should charge per cookie to match Art's earnings --/
noncomputable def roger_price_per_cookie (scenario : CookieScenario) : ℚ :=
  (scenario.art_cookies_per_batch * scenario.art_price_per_cookie) / scenario.roger_cookies_per_batch

/-- Theorem stating that Roger's price per cookie should be 100/3 cents --/
theorem roger_price_is_correct (scenario : CookieScenario) 
  (h1 : scenario.art_small_base = 4)
  (h2 : scenario.art_large_base = 6)
  (h3 : scenario.art_height = 4)
  (h4 : scenario.art_cookies_per_batch = 10)
  (h5 : scenario.art_price_per_cookie = 50)
  (h6 : scenario.roger_cookies_per_batch = 15)
  (h7 : scenario.total_dough = (scenario.art_small_base + scenario.art_large_base) / 2 * scenario.art_height * scenario.art_cookies_per_batch) :
  roger_price_per_cookie scenario = 100 / 3 := by
  sorry

/-- Main theorem proving the correctness of Roger's cookie pricing --/
theorem cookie_pricing_theorem :
  ∃ (scenario : CookieScenario), roger_price_per_cookie scenario = 100 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roger_price_is_correct_cookie_pricing_theorem_l33_3366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_is_sqrt_10_l33_3340

-- Define the curve C in polar coordinates
noncomputable def curve_C (θ : Real) : Real := 2 * Real.cos θ + 4 * Real.sin θ

-- Define the polar angles for lines l₁ and l₂
noncomputable def θ₁ : Real := Real.pi / 2
noncomputable def θ₂ : Real := Real.pi / 4

-- Define the points A and B
noncomputable def ρ_A : Real := curve_C θ₁
noncomputable def ρ_B : Real := curve_C θ₂

-- State the theorem
theorem length_AB_is_sqrt_10 :
  Real.sqrt (ρ_A^2 + ρ_B^2 - 2 * ρ_A * ρ_B * Real.cos (θ₁ - θ₂)) = Real.sqrt 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_is_sqrt_10_l33_3340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_equation_solutions_l33_3385

theorem absolute_value_equation_solutions :
  ∃! (s : Finset ℝ), (∀ x ∈ s, |x - 2| = |x - 3| + |x - 4|) ∧ s.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_equation_solutions_l33_3385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frozen_yogurt_price_is_5_l33_3336

/-- The price of a pint of frozen yogurt -/
def frozen_yogurt_price : ℝ := sorry

/-- The price of a pack of chewing gum -/
def chewing_gum_price : ℝ := sorry

/-- The price of a tray of jumbo shrimp -/
def jumbo_shrimp_price : ℝ := sorry

/-- The total cost of Shannon's purchase -/
def total_cost : ℝ := sorry

/-- Theorem stating the price of a pint of frozen yogurt -/
theorem frozen_yogurt_price_is_5 
  (h1 : total_cost = 55)
  (h2 : jumbo_shrimp_price = 5)
  (h3 : chewing_gum_price = frozen_yogurt_price / 2)
  (h4 : 5 * frozen_yogurt_price + 2 * chewing_gum_price + 5 * jumbo_shrimp_price = total_cost) :
  frozen_yogurt_price = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frozen_yogurt_price_is_5_l33_3336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_center_P_l33_3348

/-- Given two circles and a third circle P, prove the locus of P's center -/
theorem locus_of_center_P (a : ℝ) :
  -- Circle 1
  (∃ x y : ℝ, x^2 + y^2 - a*x + 2*y + 1 = 0) →
  -- Circle 2
  (∃ x y : ℝ, x^2 + y^2 = 1) →
  -- Symmetry condition
  (∃ x y x' y' m : ℝ, x^2 + y^2 - a*x + 2*y + 1 = 0 ∧ 
          x'^2 + y'^2 = 1 ∧
          y = x - 1 ∧ y' = x' - 1 ∧ 
          m = (x + x')/2 ∧ m = (y + y')/2) →
  -- Circle P passes through C(-a, a)
  (∃ x y r : ℝ, (x + a)^2 + (y - a)^2 = r^2) →
  -- Circle P is tangent to y-axis
  (∃ x y r : ℝ, x^2 = r^2) →
  -- Locus of center P
  (∃ x y : ℝ, y^2 + 4*x - 4*y + 8 = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_center_P_l33_3348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_cover_theorem_l33_3314

/-- Represents a rectangle with width and length -/
structure Rectangle where
  width : ℝ
  length : ℝ
  width_pos : 0 < width
  length_pos : 0 < length
  width_le_length : width ≤ length

/-- The area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.length

/-- The diagonal of a rectangle -/
noncomputable def Rectangle.diagonal (r : Rectangle) : ℝ := Real.sqrt (r.width^2 + r.length^2)

/-- Predicate to check if two copies of one rectangle can cover another -/
def can_cover (r1 r2 : Rectangle) : Prop :=
  2 * r1.width ≥ r2.width ∧ 2 * r1.length ≥ r2.length

/-- Main theorem statement -/
theorem rectangle_cover_theorem (P Q : Rectangle) 
  (h_area : P.area = Q.area)
  (h_diag : P.diagonal > Q.diagonal)
  (h_P_covers_Q : can_cover P Q) :
  can_cover Q P := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_cover_theorem_l33_3314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l33_3321

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in the 2D plane represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point lies on a circle -/
def Circle.containsPoint (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- Checks if a point lies on a line -/
def Line.containsPoint (l : Line) (p : ℝ × ℝ) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

/-- Checks if a line is tangent to a circle -/
def Line.isTangentTo (l : Line) (c : Circle) : Prop :=
  ∃ p : ℝ × ℝ, l.containsPoint p ∧ c.containsPoint p ∧
  ∀ q : ℝ × ℝ, l.containsPoint q ∧ c.containsPoint q → q = p

theorem tangent_line_equation :
  let c : Circle := ⟨(0, 0), Real.sqrt 5⟩
  let l : Line := ⟨2, -1, -5⟩
  let m : ℝ × ℝ := (2, -1)
  l.containsPoint m ∧ l.isTangentTo c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l33_3321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chemical_plant_storage_cost_l33_3393

/-- Economic Order Quantity Model -/
structure EOQModel where
  A : ℝ  -- Annual demand
  B : ℝ  -- Annual storage cost per unit
  C : ℝ  -- Cost per order

/-- Annual storage cost function -/
noncomputable def T (model : EOQModel) (x : ℝ) : ℝ :=
  (model.B * x) / 2 + (model.A * model.C) / x

/-- The chemical plant's EOQ model -/
def chemicalPlantModel : EOQModel :=
  { A := 6000
    B := 120
    C := 2500 }

theorem chemical_plant_storage_cost :
  T chemicalPlantModel 300 = 68000 ∧
  (∃ x : ℝ, x > 0 ∧ T chemicalPlantModel x = 60000 ∧
    ∀ y : ℝ, y > 0 → T chemicalPlantModel y ≥ T chemicalPlantModel x) ∧
  (let x := Real.sqrt ((2 * chemicalPlantModel.A * chemicalPlantModel.C) / chemicalPlantModel.B)
   T chemicalPlantModel x = 60000 ∧ x = 500) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chemical_plant_storage_cost_l33_3393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l33_3362

def sequenceA (n : ℕ) : ℕ := 2^n + 1

theorem sequence_formula (n : ℕ) : sequenceA n = 2^n + 1 := by
  rfl

#eval sequenceA 1  -- Should output 3
#eval sequenceA 2  -- Should output 5
#eval sequenceA 3  -- Should output 9
#eval sequenceA 4  -- Should output 17
#eval sequenceA 5  -- Should output 33

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l33_3362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l33_3329

/-- Statement p: The solution set of a^x > 1 is {x | x < 0} -/
def statement_p (a : ℝ) : Prop :=
  ∀ x, a^x > 1 ↔ x < 0

/-- Statement q: The domain of y = √(ax^2 - x + a) is ℝ -/
def statement_q (a : ℝ) : Prop :=
  ∀ x, a * x^2 - x + a ≥ 0

/-- Either p or q is true, but not both -/
def xor_p_q (a : ℝ) : Prop :=
  (statement_p a ∨ statement_q a) ∧ ¬(statement_p a ∧ statement_q a)

theorem range_of_a :
  ∀ a : ℝ, xor_p_q a ↔ a ∈ Set.union (Set.Ioo 0 (1/2)) (Set.Ici 1) :=
by
  sorry

#check range_of_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l33_3329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_zero_iff_f_even_l33_3364

/-- The function f(x) = ln|x-a| -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (abs (x - a))

/-- Definition of an even function -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- Theorem: a=0 is necessary and sufficient for f to be even -/
theorem a_zero_iff_f_even (a : ℝ) :
  a = 0 ↔ IsEven (f a) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_zero_iff_f_even_l33_3364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_matrix_not_invertible_l33_3384

noncomputable def projection_matrix (v : Fin 2 → ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  let norm_v := Real.sqrt (v 0 ^ 2 + v 1 ^ 2)
  let u := fun i => v i / norm_v
  Matrix.of fun i j => u i * u j

theorem projection_matrix_not_invertible :
  let v : Fin 2 → ℝ := fun i => if i = 0 then 1 else 5
  let Q := projection_matrix (fun i => 4 * v i)
  ¬IsUnit Q := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_matrix_not_invertible_l33_3384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixty_seventh_digit_of_one_seventeenth_l33_3337

/-- The decimal representation of 1/17 as a list of digits in the repeating part -/
def oneSeventeenthDecimal : List Nat := [0, 5, 8, 8, 2, 3, 5, 2, 9, 4, 1, 1, 7, 6, 4, 7]

/-- The length of the repeating part in the decimal representation of 1/17 -/
def cycleLength : Nat := 16

/-- The 67th digit after the decimal point in the decimal representation of 1/17 is 8 -/
theorem sixty_seventh_digit_of_one_seventeenth : 
  (oneSeventeenthDecimal.get! ((67 - 1) % cycleLength)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixty_seventh_digit_of_one_seventeenth_l33_3337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_of_primes_in_list_l33_3360

def number_list : List Nat := [24, 25, 29, 31, 33]

def is_prime (n : Nat) : Bool :=
  n > 1 && (Nat.factorial (n - 1) + 1) % n == 1

def primes_in_list (lst : List Nat) : List Nat :=
  lst.filter is_prime

theorem arithmetic_mean_of_primes_in_list :
  let primes := primes_in_list number_list
  (primes.sum / primes.length : ℚ) = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_of_primes_in_list_l33_3360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decrease_interval_l33_3387

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := sin x - (1/2) * x

-- State the theorem
theorem monotonic_decrease_interval :
  ∀ x y, x ∈ Set.Ioo (π/3 : ℝ) π → y ∈ Set.Ioo (π/3 : ℝ) π → 
  x < y → f x > f y :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decrease_interval_l33_3387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l33_3356

/-- Simple interest calculation -/
noncomputable def simple_interest (principal rate time : ℝ) : ℝ :=
  principal * rate * time / 100

theorem interest_rate_calculation (principal interest time : ℝ) 
  (h_principal : principal = 1600)
  (h_interest : interest = 200)
  (h_time : time = 4) :
  ∃ (rate : ℝ), simple_interest principal rate time = interest ∧ rate = 3.125 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l33_3356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_point_and_solutions_l33_3327

noncomputable def f (c b x : ℝ) : ℝ := c * Real.log x + (1/2) * x^2 + b * x

theorem extreme_point_and_solutions 
  (c b : ℝ) 
  (hc : c ≠ 0) 
  (h_extreme : ∃ b', f c b' 1 = 0 ∧ ∀ x, x > 0 → (deriv (f c b')) 1 = 0) :
  (∃ b', f c b' 1 = 0 ∧ ∀ x, x > 0 → (deriv (f c b')) x ≤ 0) →
  (c > 1 ∧ 
   (∀ x, (0 < x ∧ x < 1 ∨ x > c) → (deriv (f c b)) x > 0) ∧
   (∀ x, (1 < x ∧ x < c) → (deriv (f c b)) x < 0)) ∧
  (∃! b₁ b₂, b₁ < b₂ ∧ f c b b₁ = 0 ∧ f c b b₂ = 0) ↔
  (-1/2 < c ∧ c < 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_point_and_solutions_l33_3327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_is_sqrt_5_l33_3390

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : a > 0 ∧ b > 0

/-- Represents a circle with center (c, 0) and radius 2a -/
structure Circle where
  c : ℝ
  a : ℝ

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- Theorem: The eccentricity of a hyperbola is sqrt(5) given specific conditions -/
theorem hyperbola_eccentricity_is_sqrt_5 (h : Hyperbola) (c : Circle) 
  (h_tangent : c.c^2 = 5 * h.a^2) : eccentricity h = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_is_sqrt_5_l33_3390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l33_3334

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x) + 1

-- State the theorem
theorem range_of_x (x : ℝ) : 
  f (2 * x - 1) + f (4 - x^2) > 2 → x > -1 ∧ x < 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l33_3334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_three_primes_l33_3325

/-- The number of sides on each die -/
def num_sides : ℕ := 12

/-- The number of dice rolled -/
def num_dice : ℕ := 7

/-- The number of dice that should show a prime number -/
def target_primes : ℕ := 3

/-- The number of prime numbers on each die -/
def primes_per_die : ℕ := 5

/-- The probability of exactly three dice showing a prime number when rolling 7 fair 12-sided dice -/
theorem prob_three_primes : 
  (Nat.choose num_dice target_primes : ℚ) * 
  ((primes_per_die : ℚ) / (num_sides : ℚ)) ^ target_primes * 
  ((num_sides - primes_per_die : ℚ) / (num_sides : ℚ)) ^ (num_dice - target_primes) = 
  105035 / 373248 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_three_primes_l33_3325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_distances_l33_3300

/-- The possible distances between two points on a number line, given their distances from the origin -/
theorem possible_distances (a b : ℝ) (ha : |a| = 3) (hb : |b| = 9) :
  |a - b| = 6 ∨ |a - b| = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_distances_l33_3300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_not_divisible_by_four_l33_3377

def a : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => a (n + 1) * a n + 1

theorem a_not_divisible_by_four (n : ℕ) : 
  ¬(4 ∣ a n) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_not_divisible_by_four_l33_3377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dealer_profit_percentage_l33_3397

/-- Calculates the dealer's profit percentage given the purchase and sale information -/
theorem dealer_profit_percentage 
  (purchase_quantity : ℕ) 
  (purchase_amount : ℚ) 
  (sale_quantity : ℕ) 
  (sale_amount : ℚ) 
  (h1 : purchase_quantity = 15)
  (h2 : purchase_amount = 25)
  (h3 : sale_quantity = 12)
  (h4 : sale_amount = 33)
  : (sale_amount / sale_quantity - purchase_amount / purchase_quantity) / (purchase_amount / purchase_quantity) * 100 = 65 := by
  sorry

#check dealer_profit_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dealer_profit_percentage_l33_3397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_B_pow_2021_eq_B_l33_3318

/-- The matrix B as defined in the problem -/
noncomputable def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![1/2, 0, -(Real.sqrt 3)/2],
    ![0, -1, 0],
    ![(Real.sqrt 3)/2, 0, 1/2]]

/-- Theorem stating that B^2021 = B -/
theorem B_pow_2021_eq_B : B^2021 = B := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_B_pow_2021_eq_B_l33_3318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_probability_probability_four_heads_l33_3319

/-- The probability of getting heads in a single flip of the biased coin -/
noncomputable def h : ℝ := 1 / 2

/-- The number of flips -/
def n : ℕ := 5

/-- The condition that the probability of getting heads exactly twice
    is equal to the probability of getting heads exactly three times -/
theorem equal_probability : Nat.choose n 2 * h^2 * (1 - h)^3 = Nat.choose n 3 * h^3 * (1 - h)^2 := by
  sorry

/-- The probability of getting heads exactly four times out of five flips -/
theorem probability_four_heads :
  Nat.choose n 4 * h^4 * (1 - h)^1 = 5 / 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_probability_probability_four_heads_l33_3319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_fold_f_of_2_l33_3394

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then -x^2 - 2*x else x + 7

-- State the theorem
theorem five_fold_f_of_2 : f (f (f (f (f 2)))) = -41 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_fold_f_of_2_l33_3394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_l33_3386

theorem exponential_inequality (x : ℝ) : (0.2 : ℝ)^x < 25 → x > -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_l33_3386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_OB_OA_l33_3361

/-- Curve C1 in polar coordinates -/
def C1 (ρ θ : ℝ) : Prop := ρ^2 * (3 * Real.sin θ^2 + 1) = 4

/-- Curve C2 in polar coordinates -/
def C2 (ρ θ : ℝ) : Prop := ρ = 4 * Real.cos θ

/-- Ray l in polar coordinates -/
def l (ρ θ α : ℝ) : Prop := ρ ≥ 0 ∧ θ = α

/-- The ratio of distances from origin to points B and A -/
noncomputable def ratio (α : ℝ) : ℝ :=
  Real.sqrt ((4 - 4 * Real.sin α^2) * (3 * Real.sin α^2 + 1))

theorem max_ratio_OB_OA :
  ∃ α : ℝ, ∀ β : ℝ, ratio α ≥ ratio β ∧ ratio α = 4 * Real.sqrt 3 / 3 := by
  sorry

#check max_ratio_OB_OA

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_OB_OA_l33_3361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_david_swim_time_theorem_l33_3389

def combined_swim_time (freestyle_time backstroke_base_diff butterfly_base_diff breaststroke_base_diff
                        backstroke_wind_effect butterfly_current_effect breaststroke_external_effect : ℤ) : ℤ :=
  let backstroke_time := freestyle_time + backstroke_base_diff + backstroke_wind_effect
  let butterfly_time := freestyle_time + backstroke_base_diff + butterfly_base_diff + butterfly_current_effect
  let breaststroke_time := freestyle_time + backstroke_base_diff + butterfly_base_diff + breaststroke_base_diff - breaststroke_external_effect
  freestyle_time + backstroke_time + butterfly_time + breaststroke_time

theorem david_swim_time_theorem :
  combined_swim_time 48 4 3 2 2 3 1 = 216 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_david_swim_time_theorem_l33_3389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_point_implies_a_eq_two_min_distance_is_one_F_above_negative_F_implies_a_leq_two_l33_3349

noncomputable def f (x : ℝ) : ℝ := Real.exp x + Real.sin x

def g (a : ℝ) (x : ℝ) : ℝ := a * x

noncomputable def F (a : ℝ) (x : ℝ) : ℝ := f x - g a x

theorem extremum_point_implies_a_eq_two (a : ℝ) :
  (∀ ε > 0, ∃ δ > 0, ∀ x, |x| < δ → F a x ≥ F a 0) →
  (∀ ε > 0, ∃ δ > 0, ∀ x, |x| < δ → F a x ≤ F a 0) →
  a = 2 := by sorry

theorem min_distance_is_one (x₁ x₂ : ℝ) :
  x₁ > 0 → x₂ > 0 → f x₁ = g 1 x₂ →
  |x₂ - x₁| ≥ 1 := by sorry

theorem F_above_negative_F_implies_a_leq_two (a : ℝ) :
  (∀ x ≥ 0, F a x ≥ F a (-x)) →
  a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_point_implies_a_eq_two_min_distance_is_one_F_above_negative_F_implies_a_leq_two_l33_3349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_intersection_points_l33_3396

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^2 + x * Real.sin x + Real.cos x

-- Part 1: Tangent line
theorem tangent_line : 
  ∃ (a b : ℝ), f a = b ∧ (deriv f) a = 0 ∧ a = 0 ∧ b = 1 :=
sorry

-- Part 2: Intersection points
theorem intersection_points :
  ∀ (b : ℝ), (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = b ∧ f x₂ = b) ↔ b > 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_intersection_points_l33_3396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cot_half_angle_l33_3357

theorem cot_half_angle (x : ℝ) (h : 2 * Real.sin x = 1 + Real.cos x) (h_cos : Real.cos x ≠ -1) :
  Real.tan (π/2 - x/2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cot_half_angle_l33_3357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coplanarity_condition_l33_3358

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the points and vectors
variable (O P A B C : V)

-- Define the condition for coplanarity
def coplanar (O P A B C : V) : Prop :=
  ∃ (a b c : ℝ), a + b + c = 1 ∧ P - O = a • (A - O) + b • (B - O) + c • (C - O)

-- State the theorem
theorem coplanarity_condition (h : A + B + C - 3 • O = 3 • (P - O)) :
  coplanar O P A B C :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coplanarity_condition_l33_3358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sin_plus_sqrt2_cos_l33_3351

theorem max_value_sin_plus_sqrt2_cos :
  (∀ x : ℝ, Real.sin x + Real.sqrt 2 * Real.cos x ≤ Real.sqrt 3) ∧
  (∃ y : ℝ, Real.sin y + Real.sqrt 2 * Real.cos y = Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sin_plus_sqrt2_cos_l33_3351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_B_power_five_l33_3355

def B : Matrix (Fin 2) (Fin 2) ℝ := !![1, 1; 4, 5]

theorem B_power_five :
  B^5 = 1169 • B - 204 • (1 : Matrix (Fin 2) (Fin 2) ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_B_power_five_l33_3355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_value_of_integral_l33_3379

open Real MeasureTheory

noncomputable def I (t : ℝ) : ℝ :=
  ∫ x in Set.Icc 0 1, |x * Real.exp (-x) - t * x|

theorem minimum_value_of_integral (t : ℝ) (h : 1/Real.exp 1 < t ∧ t < 1) :
  ∃ (min_value : ℝ), min_value = 1 + 2 / Real.exp 1 - (2 + 1/2) * Real.exp (-1 / Real.sqrt 2) ∧
    ∀ (s : ℝ), 1/Real.exp 1 < s ∧ s < 1 → I s ≥ min_value :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_value_of_integral_l33_3379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_a_plus_b_l33_3323

-- Define the variables
variable (a b : ℝ)

-- Define the conditions
axiom abs_a : |a| = 2
axiom b_def : b = -1

-- State the theorem
theorem abs_a_plus_b : |a + b| = 1 ∨ |a + b| = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_a_plus_b_l33_3323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mirror_country_transfers_l33_3354

-- Define the type for cities
def City : Type := ℕ

-- Define the type for countries (graphs)
def Country := City → City → Prop

-- Define the mirror country (complement graph)
def mirror_country (c : Country) : Country :=
  λ a b ↦ ¬ c a b

-- Define the property of requiring at least two transfers
def at_least_two_transfers (c : Country) (a b : City) : Prop :=
  ¬ c a b ∧ ∀ x, ¬(c a x ∧ c x b)

-- Define the property of at most two transfers
def at_most_two_transfers (c : Country) : Prop :=
  ∀ a b, ∃ x, c a x ∨ (∃ y, c a y ∧ c y x) ∧ (c x b ∨ (∃ z, c x z ∧ c z b))

-- Theorem statement
theorem mirror_country_transfers (c : Country) (a b : City) :
  at_least_two_transfers c a b → at_most_two_transfers (mirror_country c) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mirror_country_transfers_l33_3354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_reduction_fraction_l33_3331

def initial_speed : ℝ := 24.000000000000007
def speed_increase : ℝ := 12

theorem time_reduction_fraction (d : ℝ) (h : d > 0) : 
  1 - (d / initial_speed) / (d / (initial_speed + speed_increase)) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_reduction_fraction_l33_3331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_perimeter_equals_22_l33_3328

noncomputable def triangle_area (a b c : ℝ) : ℝ := 
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem rectangle_perimeter_equals_22 (triangle_side1 triangle_side2 triangle_side3 rectangle_width : ℝ) 
  (h1 : triangle_side1 = 5)
  (h2 : triangle_side2 = 12)
  (h3 : triangle_side3 = 13)
  (h4 : rectangle_width = 5)
  (h5 : triangle_area triangle_side1 triangle_side2 triangle_side3 = rectangle_width * (triangle_area triangle_side1 triangle_side2 triangle_side3 / rectangle_width)) :
  2 * (rectangle_width + triangle_area triangle_side1 triangle_side2 triangle_side3 / rectangle_width) = 22 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_perimeter_equals_22_l33_3328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_analytic_geometry_essence_l33_3371

/-- Represents a chapter in a mathematics textbook -/
structure Chapter where
  title : String
  content : String

/-- Represents the essence of a mathematical field -/
structure MathematicalEssence where
  description : String

/-- Analytic Geometry as represented in a textbook -/
def AnalyticGeometry : Type :=
  {chapters : List Chapter // chapters.length ≥ 2 ∧ 
    (∃ c1 c2, c1 ∈ chapters ∧ c2 ∈ chapters ∧ 
     c1.title = "Lines on the Coordinate Plane" ∧ c2.title = "Conic Sections")}

/-- The theorem stating the essence of analytic geometry -/
theorem analytic_geometry_essence (ag : AnalyticGeometry) : 
  ∃ (essence : MathematicalEssence), 
    essence.description = "to study the geometric properties of figures using algebraic methods" := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_analytic_geometry_essence_l33_3371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_principal_is_250_l33_3339

/-- Represents a simple interest loan -/
structure SimpleLoan where
  principal : ℚ
  rate : ℚ
  time : ℚ
  interest : ℚ

/-- Calculates the interest for a simple interest loan -/
def calculateInterest (loan : SimpleLoan) : ℚ :=
  (loan.principal * loan.rate * loan.time) / 100

/-- Theorem stating the principal amount of the loan -/
theorem loan_principal_is_250 :
  ∃ (loan : SimpleLoan),
    loan.rate = 4 ∧
    loan.time = 8 ∧
    loan.interest = loan.principal - 170 ∧
    calculateInterest loan = loan.interest ∧
    loan.principal = 250 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_principal_is_250_l33_3339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_difference_l33_3343

theorem sin_cos_difference (θ : Real) (h1 : Real.sin θ + Real.cos θ = 3/4) (h2 : 0 < θ ∧ θ < Real.pi) : 
  Real.sin θ - Real.cos θ = Real.sqrt 23 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_difference_l33_3343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_values_l33_3316

/-- The circle equation -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y + 1 = 0

/-- The line equation -/
def line_eq (x y b : ℝ) : Prop := 3*x + 4*y = b

/-- The line is tangent to the circle -/
def is_tangent (b : ℝ) : Prop := ∃ x y : ℝ, circle_eq x y ∧ line_eq x y b

theorem tangent_line_values :
  ∀ b : ℝ, is_tangent b → b = 2 ∨ b = 12 := by
  sorry

#check tangent_line_values

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_values_l33_3316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l33_3372

/-- The distance between the foci of an ellipse with semi-major axis a and semi-minor axis b -/
noncomputable def distance_between_foci (a b : ℝ) : ℝ := 2 * Real.sqrt (a^2 - b^2)

/-- Theorem: For an ellipse with semi-major axis 8 and semi-minor axis 3, 
    the distance between the foci is 2√55 -/
theorem ellipse_foci_distance : 
  distance_between_foci 8 3 = 2 * Real.sqrt 55 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l33_3372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l33_3324

theorem triangle_area (A B C : ℝ × ℝ) : 
  let angleB : ℝ := 30 * π / 180
  let sideAB : ℝ := 2 * Real.sqrt 3
  let sideAC : ℝ := 2
  (1/2) * sideAB * sideAC * Real.sin angleB = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l33_3324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_bounds_l33_3375

/-- The parabola function -/
def parabola (x : ℝ) : ℝ := x^2 + 4

/-- The line function -/
def line (s : ℝ) : ℝ := s

/-- The x-coordinate of the intersection point -/
noncomputable def intersection_x (s : ℝ) : ℝ := Real.sqrt (s - 4)

/-- The area of the triangle -/
noncomputable def triangle_area (s : ℝ) : ℝ := (s - 4) * Real.sqrt (s - 4)

/-- Theorem: If the area of the triangle is between 16 and 128 inclusive, then s is in [8, 132] -/
theorem triangle_area_bounds (s : ℝ) :
  (16 ≤ triangle_area s ∧ triangle_area s ≤ 128) → (8 ≤ s ∧ s ≤ 132) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_bounds_l33_3375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_and_real_coefficients_l33_3317

/-- A quadratic polynomial with real coefficients -/
def quadratic_polynomial (a b c : ℝ) : ℂ → ℂ := λ x ↦ a * x^2 + b * x + c

theorem quadratic_root_and_real_coefficients :
  ∃ (a b c : ℝ), 
    a = 3 ∧
    quadratic_polynomial a b c (4 + 2*Complex.I) = 0 ∧
    quadratic_polynomial a b c = quadratic_polynomial 3 (-24) 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_and_real_coefficients_l33_3317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_decomposition_l33_3344

theorem factorial_decomposition (n : ℕ) : 
  2^7 * 3^3 * n = Nat.factorial 10 → n = 175 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_decomposition_l33_3344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_table_seating_l33_3320

theorem circular_table_seating (n : ℕ) (k : ℕ) (h1 : n = 9) (h2 : k = 8) :
  (n.choose 1) * Nat.factorial (k - 1) = 45360 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_table_seating_l33_3320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l33_3311

-- Define the side lengths, area, and circumradius
variable (a b c : ℝ)
variable (area : ℝ)
variable (circumradius : ℝ)

-- Define s and t
noncomputable def s : ℝ := Real.sqrt a + Real.sqrt b + Real.sqrt c
noncomputable def t : ℝ := 1/a + 1/b + 1/c

-- State the theorem
theorem triangle_inequality (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : area = 1/4) (h5 : circumradius = 1) :
  s < t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l33_3311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_cities_l33_3310

/-- The distance between City A and City B -/
noncomputable def distance_AB : ℝ := 432

/-- The speed of Xiao Li in km/h -/
noncomputable def speed_XiaoLi : ℝ := 72

/-- The speed of Lao Li in km/h -/
noncomputable def speed_LaoLi : ℝ := 108

/-- The time difference of arrival in hours -/
noncomputable def time_difference : ℝ := 1/4

/-- The theorem stating the distance between City A and City B -/
theorem distance_between_cities : 
  ∃ (distance_AC distance_BC : ℝ),
    distance_AC = (3/5) * distance_BC ∧
    distance_AC + distance_BC = distance_AB ∧
    distance_AC / speed_XiaoLi = (distance_BC - speed_LaoLi * time_difference) / speed_LaoLi :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_cities_l33_3310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_bound_l33_3301

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

def sequence_property (a : ℕ → ℕ) : Prop :=
  a 1 > 1 ∧ ∀ n : ℕ, n ≥ 1 → floor ((a n + 1 : ℝ) / a (n + 1)) = floor ((a (n + 1) + 1 : ℝ) / a (n + 2))

theorem floor_bound (a : ℕ → ℕ) (h : sequence_property a) :
  ∀ n : ℕ, n ≥ 1 → floor ((a n + 1 : ℝ) / a (n + 1)) ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_bound_l33_3301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_double_exists_and_triple_exists_l33_3307

def begins_with_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2 * 10^k + (n - 2 * 10^k)

def move_two_to_end (n : ℕ) : ℕ :=
  let k := (Nat.log 10 n).succ
  (n - 2 * 10^(k-1)) * 10 + 2

theorem no_double_exists_and_triple_exists : 
  (¬ ∃ n : ℕ, begins_with_two n ∧ move_two_to_end n = 2 * n) ∧ 
  (∃ n : ℕ, begins_with_two n ∧ move_two_to_end n = 3 * n ∧ n = 285714) :=
by
  sorry

#eval move_two_to_end 285714

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_double_exists_and_triple_exists_l33_3307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l33_3398

noncomputable def x : ℕ → ℝ
  | 0 => 1  -- Define for 0 to avoid missing case
  | 1 => 1
  | n + 1 => x n * Real.sqrt (1 / (2 * (Real.sqrt (1 - x n ^ 2) + 1)))

noncomputable def y : ℕ → ℝ
  | 0 => 0  -- Define for 0 to avoid missing case
  | 1 => 0
  | n + 1 => Real.sqrt ((y n + 1) / 2)

theorem sequence_properties :
  (∀ n : ℕ, n ≥ 1 → x n * x (n + 1) + y n * y (n + 1) = y (n + 1)) ∧
  (∀ n : ℕ, n ≥ 1 → x n + y n ≤ Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l33_3398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithmic_property_l33_3305

theorem logarithmic_property (a : ℝ) (h_a : a > 0 ∧ a ≠ 1) :
  ∀ (x y : ℝ), x > 0 → y > 0 → Real.log (x * y) = Real.log x + Real.log y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithmic_property_l33_3305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_image_fixed_point_l33_3370

theorem function_image_fixed_point 
  {S : Type} [Finite S] 
  (A : Set (S → S)) 
  (f : S → S)
  (hf : f ∈ A)
  (T : Set S)
  (hT : T = Set.range f)
  (h : ∀ g ∈ A, g ≠ f → (f ∘ g ∘ f) ≠ (g ∘ f ∘ g)) :
  f '' T = T := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_image_fixed_point_l33_3370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_approx_l33_3304

/-- The speed of a train crossing a bridge -/
noncomputable def train_speed (train_length bridge_length : ℝ) (crossing_time : ℝ) : ℝ :=
  (train_length + bridge_length) / crossing_time

/-- Theorem: The speed of the train is approximately 10.91 m/s -/
theorem train_speed_approx :
  let train_length : ℝ := 120
  let bridge_length : ℝ := 480
  let crossing_time : ℝ := 55
  abs (train_speed train_length bridge_length crossing_time - 10.91) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_approx_l33_3304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_exponential_function_base_constraint_l33_3392

-- Define the function f(x) = (a-1)^x
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a - 1) ^ x

-- State the theorem
theorem decreasing_exponential_function_base_constraint (a : ℝ) :
  (∀ x y : ℝ, x < y → f a y < f a x) → 1 < a ∧ a < 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_exponential_function_base_constraint_l33_3392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odot_calculation_l33_3376

-- Define the custom operation as noncomputable
noncomputable def odot (a b : ℝ) : ℝ := a + (3 * a) / (2 * b)

-- State the theorem
theorem odot_calculation : odot (odot 5 3) 4 = 10.3125 := by
  -- Unfold the definition of odot
  unfold odot
  -- Perform the calculation
  norm_num
  -- Close the proof
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odot_calculation_l33_3376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_cosine_inequality_l33_3352

theorem negation_of_cosine_inequality :
  (¬ ∀ x : ℝ, Real.cos x ≤ 1) ↔ (∃ x : ℝ, Real.cos x > 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_cosine_inequality_l33_3352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_original_cost_l33_3332

/-- The original cost of the car -/
def original_cost : ℝ := 433500

/-- The cost of repairs -/
def repair_cost : ℝ := 13000

/-- The selling price of the car -/
def selling_price : ℝ := 60900

/-- The profit percentage -/
def profit_percent : ℝ := 10.727272727272727

/-- Theorem stating the original cost of the car -/
theorem car_original_cost : 
  (selling_price - (original_cost + repair_cost)) / (original_cost + repair_cost) * 100 = profit_percent := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_original_cost_l33_3332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_when_m_2_intersection_implies_m_3_subset_complement_implies_m_range_l33_3346

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*m*x + m^2 - 4 ≤ 0}

-- Theorems to prove
theorem union_when_m_2 : A ∪ B 2 = Set.Icc (-1) 4 := by sorry

theorem intersection_implies_m_3 : A ∩ B 3 = Set.Icc 1 3 → 3 = 3 := by sorry

theorem subset_complement_implies_m_range (m : ℝ) :
  A ⊆ (Set.univ \ B m) → m > 5 ∨ m < -3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_when_m_2_intersection_implies_m_3_subset_complement_implies_m_range_l33_3346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_approx_36_seconds_l33_3391

/-- The time taken for two trains to cross each other -/
noncomputable def crossing_time (length1 length2 speed1 speed2 : ℝ) : ℝ :=
  (length1 + length2) / (speed1 + speed2) * 3600 / 1000

/-- Theorem stating that the crossing time for the given train scenario is approximately 36 seconds -/
theorem train_crossing_time_approx_36_seconds :
  let length1 : ℝ := 200  -- Length of train 1 in meters
  let length2 : ℝ := 800  -- Length of train 2 in meters
  let speed1 : ℝ := 60    -- Speed of train 1 in km/hr
  let speed2 : ℝ := 40    -- Speed of train 2 in km/hr
  abs (crossing_time length1 length2 speed1 speed2 - 36) < 0.5
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_approx_36_seconds_l33_3391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_root_implies_a_bound_l33_3368

/-- The function f(x) defined in the problem -/
noncomputable def f (a x : ℝ) : ℝ := (2 : ℝ)^(2*x) + (2 : ℝ)^x * a + a + 1

/-- Theorem stating that if f has a root, then a ≤ 2 - 2√2 -/
theorem f_root_implies_a_bound (a : ℝ) :
  (∃ x : ℝ, f a x = 0) → a ≤ 2 - 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_root_implies_a_bound_l33_3368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_remaining_square_l33_3341

/-- Represents a square cut from the cake -/
structure Square where
  x : ℝ  -- x-coordinate of the bottom-left corner
  y : ℝ  -- y-coordinate of the bottom-left corner
  side : ℝ  -- side length of the square
  h_pos : side > 0  -- side length is positive
  h_x : 0 ≤ x ∧ x + side ≤ 3  -- x-coordinate is within the cake
  h_y : 0 ≤ y ∧ y + side ≤ 3  -- y-coordinate is within the cake

/-- The cake after Karlsson's cuts -/
def RemainingCake := List Square

theorem largest_remaining_square (karlsson_cuts : List Square) 
  (h_karlsson : karlsson_cuts.length = 4 ∧ ∀ s, s ∈ karlsson_cuts → s.side = 1) :
  ∃ (little_boy_square : Square), 
    little_boy_square.side = 1/3 ∧ 
    ∀ cut, cut ∈ karlsson_cuts → 
      ¬ (little_boy_square.x < cut.x + cut.side ∧ 
         cut.x < little_boy_square.x + little_boy_square.side ∧
         little_boy_square.y < cut.y + cut.side ∧
         cut.y < little_boy_square.y + little_boy_square.side) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_remaining_square_l33_3341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l33_3399

-- Define the ellipse C
def ellipse (x y : ℝ) (a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the conditions
theorem ellipse_properties (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : b = Real.sqrt 3)
  (h4 : ellipse 1 (-3/2) a b) :
  -- Part 1: Standard equation
  (∀ x y : ℝ, ellipse x y a b ↔ x^2 / 4 + y^2 / 3 = 1) ∧
  -- Part 2: Fixed point property
  (∀ x1 y1 x2 y2 k : ℝ, 
    ellipse x1 y1 a b → 
    ellipse x2 y2 a b → 
    x1 ≠ 2 → x1 ≠ -2 → 
    x2 ≠ 2 → x2 ≠ -2 → 
    k ≠ 0 →
    y1 = k * (x1 - 2) →
    y2 = 2 * k * (x2 + 2) →
    ∃ t : ℝ, y2 - y1 = t * (x2 - x1) ∧ -2/3 = x1 + t * (x2 - x1)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l33_3399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_a_values_values_satisfy_parallel_l33_3312

-- Define the slopes of the two lines
noncomputable def slope1 (a : ℝ) : ℝ := a
noncomputable def slope2 (a : ℝ) : ℝ := 3 / (a + 2)

-- Define the condition for parallel lines
def are_parallel (a : ℝ) : Prop := slope1 a = slope2 a

-- Theorem statement
theorem parallel_lines_a_values :
  ∀ a : ℝ, are_parallel a → a = 1 ∨ a = -3 := by
  sorry

-- Additional theorem to show that the values 1 and -3 indeed satisfy the parallel condition
theorem values_satisfy_parallel :
  are_parallel 1 ∧ are_parallel (-3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_a_values_values_satisfy_parallel_l33_3312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_condition_min_m_condition_min_m_value_l33_3365

-- Define the function f
noncomputable def f (a m : ℝ) (x : ℝ) : ℝ := Real.log x + (1/2) * a * x^2 - x - m

-- Theorem for part I
theorem increasing_f_condition (a m : ℝ) :
  (∀ x > 0, ∀ y > 0, x < y → f a m x < f a m y) → a ≥ 1/4 := by
  sorry

-- Theorem for part II
theorem min_m_condition (a m : ℝ) (h1 : a < 0) :
  (∀ x > 0, f a m x < 0) → m ≥ -1 := by
  sorry

-- Theorem for the minimum value of m in part II
theorem min_m_value (a : ℝ) (h1 : a < 0) :
  ∃ m : ℤ, (∀ x > 0, f a m x < 0) ∧ (∀ m' : ℤ, (∀ x > 0, f a m' x < 0) → m ≤ m') := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_condition_min_m_condition_min_m_value_l33_3365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_distinct_values_of_sin_k_pi_over_sqrt_3_l33_3326

theorem infinite_distinct_values_of_sin_k_pi_over_sqrt_3 :
  ∀ n : ℕ, ∃ k : ℕ, k > n ∧
    ∀ m : ℕ, m < k → Real.sin (m * Real.pi / Real.sqrt 3) ≠ Real.sin (k * Real.pi / Real.sqrt 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_distinct_values_of_sin_k_pi_over_sqrt_3_l33_3326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_c_l33_3353

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

noncomputable def frac (x : ℝ) : ℝ := x - (floor x)

theorem find_c : ∃ c : ℝ, 
  (3 * (floor c : ℝ)^2 + 21 * (floor c : ℝ) - 54 = 0) ∧ 
  (4 * (frac c)^2 - 12 * (frac c) + 5 = 0) ∧ 
  (c = 2.5 ∨ c = -8.5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_c_l33_3353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_trig_inequalities_l33_3395

/-- In an acute triangle with angles A, B, and C, where A > B, 
    prove the following trigonometric inequalities. -/
theorem acute_triangle_trig_inequalities 
  (A B C : ℝ) 
  (h_acute : A + B + C = π) 
  (h_positive : 0 < A ∧ 0 < B ∧ 0 < C) 
  (h_less_than_pi_half : A < π/2 ∧ B < π/2 ∧ C < π/2) 
  (h_A_gt_B : A > B) : 
  (Real.sin A > Real.sin B) ∧ 
  (Real.cos A < Real.cos B) ∧ 
  (Real.sin A + Real.sin B > Real.cos A + Real.cos B) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_trig_inequalities_l33_3395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l33_3363

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^2 - x * Real.log x - 2

-- State the theorem
theorem f_properties :
  -- f is strictly increasing on (0, +∞)
  (∀ x y, 0 < x ∧ x < y → f x < f y) ∧
  -- For any interval [a,b] ⊆ [1/2, +∞), if the range of f(x) on [a,b] is [k(a+2), k(b+2)],
  -- then k ∈ (1, (9+2ln2)/10]
  (∀ a b k, 1/2 ≤ a ∧ a < b ∧
    (∃ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b ∧ 
      f x = k * (x + 2) ∧ f y = k * (y + 2) ∧
      (∀ z, a ≤ z ∧ z ≤ b → k * (z + 2) ≤ f z ∧ f z ≤ k * (z + 2))) →
    1 < k ∧ k ≤ (9 + 2 * Real.log 2) / 10) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l33_3363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_increasing_iff_l33_3303

/-- A geometric sequence with first term a₁ and common ratio q -/
def geometric_sequence (a₁ q : ℝ) : ℕ → ℝ :=
  λ n ↦ a₁ * q ^ (n - 1)

/-- A sequence is increasing -/
def is_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n, a n < a (n + 1)

theorem geometric_sequence_increasing_iff (a₁ q : ℝ) (h : a₁ < 0) :
  is_increasing (geometric_sequence a₁ q) ↔ 0 < q ∧ q < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_increasing_iff_l33_3303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_theorem_l33_3313

-- Define the function f as noncomputable
noncomputable def f (t : ℝ) : ℝ := t / (1 - t)

-- State the theorem
theorem inverse_function_theorem (x y : ℝ) (h : y = f x) (h1 : x ≠ 1) :
  x = -(f (-y)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_theorem_l33_3313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_area_l33_3342

/-- A right triangle with an inscribed square -/
structure InscribedSquareTriangle where
  /-- Length of LP -/
  lp : ℝ
  /-- Length of NS -/
  ns : ℝ
  /-- LP is positive -/
  lp_pos : 0 < lp
  /-- NS is positive -/
  ns_pos : 0 < ns

/-- The area of the inscribed square -/
noncomputable def square_area (t : InscribedSquareTriangle) : ℝ :=
  Real.sqrt (t.lp * t.ns)

/-- Theorem stating the area of the inscribed square -/
theorem inscribed_square_area (t : InscribedSquareTriangle) 
    (h1 : t.lp = 32) (h2 : t.ns = 64) : 
    square_area t = 2048 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_area_l33_3342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_eq_sin_sin_unique_solution_l33_3388

open Real

theorem sin_eq_sin_sin_unique_solution :
  ∃! x : ℝ, 0 ≤ x ∧ x ≤ Real.arcsin 0.99 ∧ sin x = sin (sin x) :=
by
  sorry

-- Assumptions
axiom sin_gt_x : ∀ θ : ℝ, 0 < θ → θ < π / 2 → sin θ > θ

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_eq_sin_sin_unique_solution_l33_3388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l33_3309

theorem relationship_abc (a b c : ℝ) : 
  a = Real.log 2 → b = Real.log (1/2) / Real.log 3 → c = 2^(6/10) → b < a ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l33_3309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_definition_l33_3306

/-- Represents an athlete -/
structure Athlete where
  age : ℕ

/-- Represents a group of athletes -/
def AthleteGroup := List Athlete

/-- Represents a sample of ages -/
def AgeSample := List ℕ

/-- Define the population -/
def population : AthleteGroup := []

/-- Define the sample -/
def sample : AgeSample := []

theorem sample_definition (h1 : population.length = 1000) (h2 : sample.length = 100) :
  sample = (population.take 100).map Athlete.age := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_definition_l33_3306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_bounds_l33_3383

/-- The curve C in the xy-plane -/
def curve (x y : ℝ) : Prop := x^2 / 4 + y^2 / 9 = 1

/-- The line l in the xy-plane -/
def line (x y : ℝ) : Prop := 2*x + y - 6 = 0

/-- The distance from a point (x, y) to the line l -/
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  |2*x + y - 6| / Real.sqrt 5

/-- Theorem stating the maximum and minimum distances from curve C to line l -/
theorem distance_bounds :
  (∃ x y : ℝ, curve x y ∧ distance_to_line x y = 11 * Real.sqrt 5 / 5) ∧
  (∃ x y : ℝ, curve x y ∧ distance_to_line x y = Real.sqrt 5 / 5) ∧
  (∀ x y : ℝ, curve x y → Real.sqrt 5 / 5 ≤ distance_to_line x y ∧ distance_to_line x y ≤ 11 * Real.sqrt 5 / 5) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_bounds_l33_3383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_rent_is_4100_l33_3333

/-- Represents the car rental company's problem --/
structure CarRentalProblem where
  totalCars : ℕ
  initialRent : ℝ
  rentIncrement : ℝ
  carsUnrentedPerIncrement : ℝ
  rentedCarMaintenance : ℝ
  unrentedCarMaintenance : ℝ

/-- Calculates the monthly revenue for a given rent --/
noncomputable def monthlyRevenue (p : CarRentalProblem) (rent : ℝ) : ℝ :=
  let carsRented := p.totalCars - (rent - p.initialRent) / p.rentIncrement * p.carsUnrentedPerIncrement
  let carsUnrented := p.totalCars - carsRented
  rent * carsRented - p.rentedCarMaintenance * carsRented - p.unrentedCarMaintenance * carsUnrented

/-- Theorem stating that 4100 yuan maximizes the monthly revenue --/
theorem optimal_rent_is_4100 (p : CarRentalProblem) 
    (h1 : p.totalCars = 100)
    (h2 : p.initialRent = 3000)
    (h3 : p.rentIncrement = 50)
    (h4 : p.carsUnrentedPerIncrement = 1)
    (h5 : p.rentedCarMaintenance = 150)
    (h6 : p.unrentedCarMaintenance = 50) :
    ∀ rent, monthlyRevenue p 4100 ≥ monthlyRevenue p rent := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_rent_is_4100_l33_3333
