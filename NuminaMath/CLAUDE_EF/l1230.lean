import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l1230_123090

/-- Given an ellipse with semi-major axis a and semi-minor axis b, 
    prove that if there exists a point P on the ellipse such that PF₁ ⊥ PF₂, 
    then the eccentricity e of the ellipse satisfies √2/2 ≤ e < 1 -/
theorem ellipse_eccentricity_range (a b : ℝ) (h₁ : a > b) (h₂ : b > 0) :
  ∃ (x y : ℝ), 
    x^2 / a^2 + y^2 / b^2 = 1 ∧ 
    (x + a * Real.sqrt (1 - b^2/a^2))^2 + y^2 = 
    (x - a * Real.sqrt (1 - b^2/a^2))^2 + y^2 →
    let e := Real.sqrt (1 - b^2/a^2)
    Real.sqrt 2 / 2 ≤ e ∧ e < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l1230_123090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_radius_for_specific_hole_l1230_123086

/-- Represents the properties of a spherical hole in ice --/
structure IceHole where
  surface_diameter : ℝ
  depth : ℝ

/-- Calculates the radius of a ball that would create the given ice hole --/
noncomputable def ball_radius (hole : IceHole) : ℝ :=
  let surface_radius := hole.surface_diameter / 2
  let x := (hole.depth * hole.depth + surface_radius * surface_radius - hole.depth * hole.depth) / (2 * hole.depth)
  x + hole.depth

/-- Theorem stating that a ball creating a hole with 30 cm diameter and 10 cm depth has a radius of 16.25 cm --/
theorem ball_radius_for_specific_hole :
  let hole : IceHole := { surface_diameter := 30, depth := 10 }
  ball_radius hole = 16.25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_radius_for_specific_hole_l1230_123086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_equal_opposite_angles_l1230_123083

/-- A tetrahedral angle is a solid angle formed by four planes intersecting at a point. -/
structure TetrahedralAngle where
  vertex : EuclideanSpace ℝ (Fin 3)
  faces : Fin 4 → Subspace ℝ (EuclideanSpace ℝ (Fin 3))
  -- Additional conditions to ensure it forms a valid tetrahedral angle

/-- A sphere in 3D Euclidean space. -/
structure Sphere where
  center : EuclideanSpace ℝ (Fin 3)
  radius : ℝ

/-- A sphere is inscribed in a tetrahedral angle if it touches all four faces of the angle. -/
def isInscribed (s : Sphere) (t : TetrahedralAngle) : Prop :=
  ∀ i : Fin 4, ∃ p : EuclideanSpace ℝ (Fin 3), p ∈ t.faces i ∧ dist p s.center = s.radius

/-- The planar angle between two faces of a tetrahedral angle. -/
noncomputable def planarAngle (t : TetrahedralAngle) (i j : Fin 4) : ℝ :=
  sorry -- Definition of planar angle between faces i and j

/-- Theorem: If a sphere can be inscribed in a tetrahedral angle, then the sums of opposite planar angles are equal. -/
theorem inscribed_sphere_equal_opposite_angles (t : TetrahedralAngle) (s : Sphere) :
  isInscribed s t →
  planarAngle t 0 1 + planarAngle t 2 3 = planarAngle t 0 2 + planarAngle t 1 3 :=
by
  sorry -- Proof goes here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_equal_opposite_angles_l1230_123083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_two_irrational_in_set_zero_point_three_rational_neg_sqrt_nine_rational_zero_rational_sqrt_two_irrational_l1230_123010

-- Define the set of numbers given in the problem
def problem_set : Set ℝ := {0.3, -Real.sqrt 9, 0, Real.sqrt 2}

-- Theorem stating that √2 is the only irrational number in the set
theorem sqrt_two_irrational_in_set :
  ∃! x, x ∈ problem_set ∧ Irrational x :=
by
  sorry

-- Helper theorems for individual numbers
theorem zero_point_three_rational : ¬Irrational (0.3 : ℝ) :=
by
  sorry

theorem neg_sqrt_nine_rational : ¬Irrational (-Real.sqrt 9 : ℝ) :=
by
  sorry

theorem zero_rational : ¬Irrational (0 : ℝ) :=
by
  sorry

theorem sqrt_two_irrational : Irrational (Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_two_irrational_in_set_zero_point_three_rational_neg_sqrt_nine_rational_zero_rational_sqrt_two_irrational_l1230_123010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_solution_property_l1230_123071

/-- 
Given a quadratic equation 3x^2 + 6x + p = 0, this theorem states that the values of p 
for which the positive difference between its solutions equals the sum of the first 
solution squared and twice the square of the second solution are 6 and 18.
-/
theorem quadratic_equation_solution_property (p : ℝ) : 
  (∃ a b : ℝ, (3 * a^2 + 6 * a + p = 0) ∧ 
               (3 * b^2 + 6 * b + p = 0) ∧ 
               (abs (a - b) = a^2 + 2 * b^2)) ↔ 
  (p = 6 ∨ p = 18) := by
  sorry

#check quadratic_equation_solution_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_solution_property_l1230_123071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_widgets_sold_15_days_l1230_123094

/-- Calculates the total number of widgets sold over a given number of days -/
def total_widgets (days : ℕ) : ℕ :=
  let first_day := 2
  let subsequent_days := (Finset.sum (Finset.range days) (fun i => 3 * (i + 1))) - 3
  first_day + subsequent_days

/-- Theorem stating that the total number of widgets sold over 15 days is 359 -/
theorem widgets_sold_15_days : total_widgets 15 = 359 := by
  -- Unfold the definition of total_widgets
  unfold total_widgets
  -- Simplify the sum
  simp [Finset.sum_range]
  -- Evaluate the arithmetic
  norm_num
  -- QED
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_widgets_sold_15_days_l1230_123094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_characterization_l1230_123055

noncomputable def f (x : ℝ) := Real.exp x

theorem exponential_inequality_characterization (k : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) < |k| * (f x₁ + f x₂)) ↔
  k ∈ Set.Iic (-1/2) ∪ Set.Ici (1/2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_characterization_l1230_123055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_40_terms_l1230_123085

-- Define the sequence a_n
def a : ℕ → ℚ
  | 0 => 1  -- Added case for 0
  | 1 => 1
  | 2 => 3
  | (n + 3) => a (n + 1) / a (n + 2)

-- Define the sum of the first n terms
def S (n : ℕ) : ℚ :=
  Finset.sum (Finset.range n) (λ i => a (i + 1))

-- Theorem statement
theorem sum_of_40_terms :
  S 40 = 60 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_40_terms_l1230_123085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_is_zero_l1230_123064

/-- An arithmetic sequence with first four terms a, y, b, and 3y -/
structure ArithmeticSequence (α : Type*) [LinearOrderedField α] where
  a : α
  y : α
  b : α

/-- The ratio of a to b in the arithmetic sequence -/
noncomputable def ratio (seq : ArithmeticSequence ℝ) : ℝ :=
  seq.a / seq.b

/-- Theorem: The ratio of a to b in the specified arithmetic sequence is 0 -/
theorem ratio_is_zero (seq : ArithmeticSequence ℝ) 
  (h1 : seq.y - seq.a = seq.b - seq.y)  -- Arithmetic sequence condition
  (h2 : seq.b - seq.y = 3 * seq.y - seq.b)  -- Arithmetic sequence condition
  : ratio seq = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_is_zero_l1230_123064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_problem_l1230_123075

-- Define the solution set
def solution_set (a b : ℝ) : Set ℝ := {x : ℝ | |x + a| ≤ b}

-- State the theorem
theorem inequality_problem (a b : ℝ) 
  (h : solution_set a b = Set.Icc (-6) 2) : 
  a = 2 ∧ b = 4 ∧ 
  ∀ y z : ℝ, |a * y + z| < 1/3 → |y - b * z| < 1/6 → |z| < 2/27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_problem_l1230_123075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_special_property_l1230_123042

/-- Given a triangle ABC where sides a, b, c and angles A, B, C form arithmetic sequences,
    and the area is √3/2, prove that side b equals (3 + √3) / 3. -/
theorem triangle_special_property (a b c A B C : ℝ) : 
  -- Sides form an arithmetic sequence
  2 * b = a + c →
  -- Angles form an arithmetic sequence
  2 * B = A + C →
  -- Sum of angles in a triangle is π
  A + B + C = Real.pi →
  -- Area of the triangle is √3/2
  (1 / 2) * a * c * Real.sin B = Real.sqrt 3 / 2 →
  -- Side b equals (3 + √3) / 3
  b = (3 + Real.sqrt 3) / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_special_property_l1230_123042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_in_truncated_cone_l1230_123056

/-- A truncated cone with a tangent sphere -/
structure TruncatedConeWithSphere where
  bottom_radius : ℝ
  top_radius : ℝ
  height : ℝ
  sphere_radius : ℝ

/-- Predicate to represent that the sphere is tangent to the truncated cone -/
def IsTangent (sphere_radius top_radius bottom_radius height : ℝ) : Prop :=
  ∃ (x y : ℝ),
    x^2 + y^2 = sphere_radius^2 ∧
    (height - y)^2 + (bottom_radius - x)^2 = sphere_radius^2 ∧
    y^2 + (top_radius - x)^2 = sphere_radius^2

/-- The theorem stating the radius of the tangent sphere in a truncated cone -/
theorem sphere_radius_in_truncated_cone (cone : TruncatedConeWithSphere)
  (h_bottom : cone.bottom_radius = 20)
  (h_top : cone.top_radius = 5)
  (h_height : cone.height = 15)
  (h_tangent : IsTangent cone.sphere_radius cone.top_radius cone.bottom_radius cone.height) :
  cone.sphere_radius = 7.5 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_in_truncated_cone_l1230_123056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diver_descent_rate_l1230_123008

/-- The rate of descent for a diver given depth and time -/
noncomputable def rate_of_descent (depth : ℝ) (time : ℝ) : ℝ :=
  depth / time

/-- Theorem stating that for a depth of 3600 feet and time of 60 minutes, 
    the rate of descent is 60 feet per minute -/
theorem diver_descent_rate : 
  rate_of_descent 3600 60 = 60 := by
  -- Unfold the definition of rate_of_descent
  unfold rate_of_descent
  -- Simplify the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diver_descent_rate_l1230_123008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_riccati_solution_l1230_123034

noncomputable section

open Real

/-- The Riccati equation --/
def riccati_eq (y : ℝ → ℝ) (x : ℝ) : Prop :=
  deriv y x - y x ^ 2 + 2 * exp x * y x = exp (2 * x) + exp x

/-- The particular solution --/
def particular_solution (x : ℝ) : ℝ := exp x

/-- The general solution --/
def general_solution (C : ℝ) (x : ℝ) : ℝ := exp x + 1 / (C - x)

/-- Theorem stating that if the particular solution satisfies the Riccati equation,
    then the general solution also satisfies the equation --/
theorem riccati_solution (C : ℝ) :
  (∀ x, riccati_eq particular_solution x) →
  (∀ x, riccati_eq (general_solution C) x) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_riccati_solution_l1230_123034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bookshelf_probabilities_l1230_123065

/-- Represents a bookshelf with a number of English and Chinese books -/
structure Bookshelf where
  english : ℕ
  chinese : ℕ

/-- Calculates the probability of drawing two English books in a row without replacement -/
def prob_two_english_without_replacement (shelf : Bookshelf) : ℚ :=
  (shelf.english : ℚ) / (shelf.english + shelf.chinese) *
  (shelf.english - 1) / (shelf.english + shelf.chinese - 1)

/-- Calculates the probability of drawing two English books after transferring two books from one shelf to another -/
def prob_two_english_after_transfer (shelf_from shelf_to : Bookshelf) : ℚ :=
  let total_from := shelf_from.english + shelf_from.chinese
  let c2_2 := (shelf_from.english * (shelf_from.english - 1)) / 2
  let c3_2 := (shelf_from.chinese * (shelf_from.chinese - 1)) / 2
  let c2_1_c3_1 := shelf_from.english * shelf_from.chinese
  let c5_2 := (total_from * (total_from - 1)) / 2
  let c8_2 := 28

  (c2_2 / c5_2) * ((shelf_to.english + 2) * (shelf_to.english + 1) / c8_2) +
  (c3_2 / c5_2) * (shelf_to.english * (shelf_to.english - 1) / c8_2) +
  (c2_1_c3_1 / c5_2) * ((shelf_to.english + 1) * shelf_to.english / c8_2)

theorem bookshelf_probabilities 
  (shelf_a : Bookshelf) 
  (shelf_b : Bookshelf) 
  (h1 : shelf_a = ⟨4, 2⟩) 
  (h2 : shelf_b = ⟨2, 3⟩) : 
  prob_two_english_without_replacement shelf_a = 2/5 ∧ 
  prob_two_english_after_transfer shelf_b shelf_a = 93/280 := by
  sorry

#eval prob_two_english_without_replacement ⟨4, 2⟩
#eval prob_two_english_after_transfer ⟨2, 3⟩ ⟨4, 2⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bookshelf_probabilities_l1230_123065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gary_overtime_multiplier_l1230_123027

/-- Calculates the overtime pay multiplier given the regular hourly rate, 
    total hours worked, and total paycheck. -/
noncomputable def overtime_multiplier (regular_rate : ℝ) (total_hours : ℝ) (total_pay : ℝ) : ℝ :=
  let regular_hours := 40
  let overtime_hours := total_hours - regular_hours
  let regular_pay := regular_rate * regular_hours
  let overtime_pay := total_pay - regular_pay
  let overtime_rate := overtime_pay / overtime_hours
  overtime_rate / regular_rate

theorem gary_overtime_multiplier :
  overtime_multiplier 12 52 696 = 1.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gary_overtime_multiplier_l1230_123027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_k_l1230_123066

/-- The function f(x) = ax + b -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b

/-- Recursive definition of f_n -/
def f_n (a b : ℝ) : ℕ → (ℝ → ℝ)
  | 0 => λ x => x  -- Base case for n = 0
  | 1 => f a b
  | n + 1 => f a b ∘ f_n a b n

/-- The theorem to be proved -/
theorem find_k (a b : ℝ) (k : ℕ) :
  (2 * a + b = -2) →
  (f_n a b k = λ x ↦ -243 * x + 244) →
  k = 5 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_k_l1230_123066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sleeping_bag_wholesale_cost_l1230_123025

/-- The wholesale cost of a sleeping bag, given the selling price and profit margin. -/
noncomputable def wholesale_cost (selling_price : ℝ) (profit_margin : ℝ) : ℝ :=
  selling_price / (1 + profit_margin)

/-- Theorem stating that the wholesale cost of a sleeping bag is approximately $23.93 -/
theorem sleeping_bag_wholesale_cost :
  let selling_price : ℝ := 28
  let profit_margin : ℝ := 0.17
  ∃ ε > 0, |wholesale_cost selling_price profit_margin - 23.93| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sleeping_bag_wholesale_cost_l1230_123025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_third_of_recipe_sugar_l1230_123026

/-- Represents a mixed number as a pair of integers (whole, numerator, denominator) -/
structure MixedNumber where
  whole : Int
  numerator : Int
  denominator : Nat
  denominator_pos : denominator > 0

def MixedNumber.toRational (m : MixedNumber) : ℚ :=
  m.whole + (m.numerator : ℚ) / m.denominator

def oneThird : ℚ := 1 / 3

theorem one_third_of_recipe_sugar (original : MixedNumber) (result : MixedNumber) : 
  original.whole = 7 ∧ original.numerator = 2 ∧ original.denominator = 3 →
  result.whole = 2 ∧ result.numerator = 5 ∧ result.denominator = 9 →
  oneThird * original.toRational = result.toRational := by
  sorry

#check one_third_of_recipe_sugar

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_third_of_recipe_sugar_l1230_123026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_work_completion_time_l1230_123020

/-- A worker who can complete a job in a given number of days -/
structure Worker where
  days_to_complete : ℚ

/-- The amount of work completed by a worker in a given number of days -/
def work_completed (w : Worker) (days : ℚ) : ℚ :=
  days / w.days_to_complete

theorem remaining_work_completion_time 
  (a b : Worker)
  (a_total_days : ℚ)
  (a_worked_days : ℚ)
  (b_completes_remaining : Bool) :
  a.days_to_complete = 15 →
  a_worked_days = 5 →
  b.days_to_complete = 27 →
  b_completes_remaining = true →
  (1 - work_completed a a_worked_days) / (1 / b.days_to_complete) = 18 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_work_completion_time_l1230_123020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_ratio_l1230_123038

/-- Represents a tetrahedron ABCD with specific properties -/
structure Tetrahedron where
  a : ℝ  -- Length of edge AB
  b : ℝ  -- Length of edge CD
  d : ℝ  -- Distance between skew lines AB and CD
  ω : ℝ  -- Angle between AB and CD
  k : ℝ  -- Ratio of distances of plane ε from AB and CD

/-- Calculates the ratio of volumes of two solids formed by plane ε in the tetrahedron -/
noncomputable def volume_ratio (t : Tetrahedron) : ℝ :=
  (t.a / t.b) * t.k^2 * ((t.k + 3) / (3 * t.k + 1)) * Real.cos t.ω

/-- Theorem stating the volume ratio in the tetrahedron -/
theorem tetrahedron_volume_ratio (t : Tetrahedron) :
  volume_ratio t = (t.a / t.b) * t.k^2 * ((t.k + 3) / (3 * t.k + 1)) * Real.cos t.ω :=
by
  -- Unfold the definition of volume_ratio
  unfold volume_ratio
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_ratio_l1230_123038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1230_123024

/-- Triangle ABC with given properties -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  area : ℝ
  angle_sum : A + B + C = π
  sine_law : a / (Real.sin A) = b / (Real.sin B)
  cosine_law : c^2 = a^2 + b^2 - 2*a*b*(Real.cos C)
  area_formula : area = (1/2) * a * b * (Real.sin C)

/-- Main theorem about the triangle -/
theorem triangle_properties (abc : Triangle) 
  (h1 : abc.C = π/3)
  (h2 : abc.b = 5)
  (h3 : abc.area = 10 * Real.sqrt 3) :
  abc.a = 8 ∧ 
  abc.c = 7 ∧ 
  Real.sin (abc.A + π/6) = 13/14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1230_123024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_square_rectangle_perimeter_l1230_123017

/-- Represents a rectangle composed of six squares -/
structure SixSquareRectangle where
  top_left_area : ℝ
  bottom_left_area : ℝ

/-- Calculates the perimeter of a SixSquareRectangle -/
noncomputable def perimeter (rect : SixSquareRectangle) : ℝ :=
  let top_left_side := (rect.top_left_area).sqrt
  let bottom_left_side := (rect.bottom_left_area).sqrt
  let height := top_left_side + bottom_left_side
  let middle_square_side := top_left_side - bottom_left_side
  let top_right_side := top_left_side + middle_square_side
  let width := top_left_side + top_right_side
  2 * (height + width)

theorem six_square_rectangle_perimeter :
  ∀ (rect : SixSquareRectangle),
    rect.top_left_area = 36 →
    rect.bottom_left_area = 25 →
    perimeter rect = 48 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_square_rectangle_perimeter_l1230_123017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monomial_and_polynomial_properties_l1230_123063

open MvPolynomial

/-- The coefficient of a monomial -/
def coefficient (m : MvPolynomial (Fin n) ℚ) : ℚ :=
  sorry

/-- The degree of a monomial -/
def monomial_degree (m : MvPolynomial (Fin n) ℚ) : ℕ :=
  sorry

/-- The degree of a polynomial -/
def polynomial_degree (p : MvPolynomial (Fin n) ℚ) : ℕ :=
  sorry

/-- The monomial -2x²yz² -/
def m : MvPolynomial (Fin 3) ℚ :=
  sorry

/-- The polynomial 2-a-ab-2a²b -/
def p : MvPolynomial (Fin 2) ℚ :=
  sorry

theorem monomial_and_polynomial_properties :
  (coefficient m = -2) ∧
  (monomial_degree m = 5) ∧
  (polynomial_degree p = 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monomial_and_polynomial_properties_l1230_123063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_main_theorem_l1230_123006

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- The distance from the center to a focus of an ellipse -/
noncomputable def Ellipse.focal_distance (e : Ellipse) : ℝ :=
  Real.sqrt (e.a^2 - e.b^2)

/-- The condition that the foci and one endpoint of the minor axis form an equilateral triangle -/
noncomputable def Ellipse.equilateral_triangle_condition (e : Ellipse) : Prop :=
  e.b = Real.sqrt 3 * e.focal_distance

/-- The shortest distance from a focus to the ellipse -/
noncomputable def Ellipse.shortest_focus_distance (e : Ellipse) : ℝ :=
  e.b - e.focal_distance

theorem ellipse_equation (e : Ellipse) 
  (h_equilateral : e.equilateral_triangle_condition)
  (h_shortest_dist : e.shortest_focus_distance = Real.sqrt 3) :
  e.a = 2 * Real.sqrt 3 ∧ e.b = 3 := by
  sorry

theorem main_theorem (e : Ellipse) 
  (h_equilateral : e.equilateral_triangle_condition)
  (h_shortest_dist : e.shortest_focus_distance = Real.sqrt 3) :
  ∀ (x y : ℝ), x^2 / 12 + y^2 / 9 = 1 ↔ x^2 / e.a^2 + y^2 / e.b^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_main_theorem_l1230_123006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_ellipse_major_axis_l1230_123031

/-- The ellipse passing through (√5, √3) with the same foci as x²/25 + y²/9 = 1 -/
def special_ellipse : Set (ℝ × ℝ) :=
  {(x, y) | ∃ (a : ℝ), x^2 / a^2 + y^2 / (a^2 - 16) = 1 ∧ (Real.sqrt 5, Real.sqrt 3) ∈ {(x, y) | x^2 / a^2 + y^2 / (a^2 - 16) = 1}}

/-- The length of the major axis of the special ellipse -/
noncomputable def major_axis_length : ℝ := 4 * Real.sqrt 5

/-- Theorem stating that the length of the major axis of the special ellipse is 4√5 -/
theorem special_ellipse_major_axis :
  ∀ (x y : ℝ), (x, y) ∈ special_ellipse →
  ∃ (a : ℝ), x^2 / a^2 + y^2 / (a^2 - 16) = 1 ∧ a = major_axis_length / 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_ellipse_major_axis_l1230_123031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sum_acute_angle_l1230_123068

theorem sin_cos_sum_acute_angle (θ : ℝ) (a b : ℝ) 
  (h_acute : 0 < θ ∧ θ < π / 2)
  (h_sin_2θ : Real.sin (2 * θ) = a)
  (h_cos_2θ : Real.cos (2 * θ) = b) :
  Real.sin θ + Real.cos θ = Real.sqrt ((1 + b) / 2) + Real.sqrt ((1 - b) / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sum_acute_angle_l1230_123068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_theorem_l1230_123000

/-- The hyperbola C: x^2 - y^2/3 = 1 -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

/-- The line l: y = k(x - 3) passing through (3,0) with slope k -/
def line (k x : ℝ) : ℝ := k * (x - 3)

/-- The right focus of the hyperbola -/
def right_focus : ℝ × ℝ := (2, 0)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem: If a line with slope k passing through (3,0) intersects the right branch
    of the hyperbola x^2 - y^2/3 = 1 at points A and B, and the sum of distances
    from A and B to the right focus is 16, then k = ±1 -/
theorem intersection_distance_theorem (k : ℝ) :
  (∃ (xA yA xB yB : ℝ),
    hyperbola xA yA ∧ hyperbola xB yB ∧
    yA = line k xA ∧ yB = line k xB ∧
    xA ≠ xB ∧
    distance (xA, yA) right_focus + distance (xB, yB) right_focus = 16) →
  k = 1 ∨ k = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_theorem_l1230_123000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_balloons_l1230_123087

theorem tom_balloons (initial_balloons : ℕ) (fraction_given : ℚ) : 
  initial_balloons = 30 → 
  fraction_given = 2/3 → 
  initial_balloons - (fraction_given * ↑initial_balloons).floor = 10 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_balloons_l1230_123087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_tangent_slope_l1230_123041

/-- The equation of a curve and its tangent line at (0, 0) -/
noncomputable def Curve (a : ℝ) (x : ℝ) : ℝ := a * x - Real.log (x + 1)

/-- The slope of the tangent line at (0, 0) -/
def TangentSlope : ℝ := 2

/-- Theorem: Given the curve y = ax - ln(x + 1) and its tangent line y = 2x at (0, 0), a = 3 -/
theorem curve_tangent_slope (a : ℝ) : 
  (∀ x, Curve a x = a * x - Real.log (x + 1)) →
  (TangentSlope = 2) →
  a = 3 := by
  sorry

#check curve_tangent_slope

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_tangent_slope_l1230_123041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dig_holes_theorem_l1230_123003

/-- The rate at which workers dig holes -/
noncomputable def digRate (workers : ℝ) (holes : ℝ) (hours : ℝ) : ℝ :=
  holes / (workers * hours)

/-- The number of holes dug given a rate, number of workers, and time -/
noncomputable def holesDug (rate : ℝ) (workers : ℝ) (hours : ℝ) : ℝ :=
  rate * workers * hours

theorem dig_holes_theorem (initial_workers : ℝ) (initial_holes : ℝ) (initial_hours : ℝ)
    (final_workers : ℝ) (final_hours : ℝ) :
    initial_workers = 1.5 →
    initial_holes = 1.5 →
    initial_hours = 1.5 →
    final_workers = 2 →
    final_hours = 2 →
    holesDug (digRate initial_workers initial_holes initial_hours) final_workers final_hours = 8/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dig_holes_theorem_l1230_123003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_points_theorem_l1230_123048

noncomputable section

-- Define the unit circle
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the distance between two points
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

-- Define the angle between two points and the origin
noncomputable def angle_AOB (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.arccos ((x₁ * x₂ + y₁ * y₂) / (Real.sqrt (x₁^2 + y₁^2) * Real.sqrt (x₂^2 + y₂^2)))

-- Define the function to be maximized
def f (y₁ y₂ : ℝ) : ℝ := |y₁ + 2| + |y₂ + 2|

theorem circle_points_theorem (x₁ y₁ x₂ y₂ : ℝ) 
  (h₁ : unit_circle x₁ y₁) 
  (h₂ : unit_circle x₂ y₂) 
  (h₃ : distance x₁ y₁ x₂ y₂ = 1) :
  (angle_AOB x₁ y₁ x₂ y₂ = π / 3) ∧ 
  (∃ (max : ℝ), max = 4 + Real.sqrt 3 ∧ ∀ (y₁' y₂' : ℝ), f y₁' y₂' ≤ max) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_points_theorem_l1230_123048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_decreasing_function_implies_a_range_l1230_123018

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 - a*x) * Real.exp x

-- Define the derivative of f(x)
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := (x^2 + (2 - a)*x - a) * Real.exp x

-- State the theorem
theorem not_decreasing_function_implies_a_range (a : ℝ) :
  (∃ x ∈ Set.Icc (-1 : ℝ) 1, f' a x > 0) → a < 3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_decreasing_function_implies_a_range_l1230_123018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mark_walking_distance_l1230_123097

/-- The distance walked in miles, given a rate in miles per minute and a time in minutes -/
def distance_walked (rate : ℚ) (time : ℚ) : ℚ := rate * time

/-- Rounds a rational number to the nearest tenth -/
noncomputable def round_to_tenth (x : ℚ) : ℚ := 
  (x * 10).floor / 10 + if (x * 10 - (x * 10).floor ≥ 1/2) then 1/10 else 0

theorem mark_walking_distance : 
  let rate : ℚ := 1 / 18
  let time : ℚ := 15
  round_to_tenth (distance_walked rate time) = 8 / 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mark_walking_distance_l1230_123097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_woman_work_days_is_225_l1230_123007

/-- The number of days required for one woman to complete a work, given that:
    - 10 men and 15 women together can complete the work in 6 days
    - One man alone can complete the work in 100 days -/
def womanWorkDays : ℚ := 225

/-- Proof that the number of days required for one woman to complete the work is 225 -/
theorem woman_work_days_is_225 : womanWorkDays = 225 := by
  -- Define the work rate of one man
  let manRate : ℚ := 1 / 100

  -- Define the total work as 1
  let totalWork : ℚ := 1

  -- Define the equation for 10 men and 15 women completing the work in 6 days
  have h1 : (10 * manRate + 15 * (1 / womanWorkDays)) * 6 = totalWork := by
    -- Proof of this equation
    sorry

  -- Solve for womanWorkDays
  have h2 : womanWorkDays = 225 := by
    -- Algebraic manipulation to solve the equation
    sorry

  -- Conclude the proof
  exact h2

#eval womanWorkDays -- This will output 225

end NUMINAMATH_CALUDE_ERRORFEEDBACK_woman_work_days_is_225_l1230_123007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_property_distribution_l1230_123079

def distribute (total : ℝ) (fixed : ℝ) (fraction : ℝ) (remaining : ℝ) : ℝ :=
  fixed + fraction * remaining

def remaining_after_distribution (total : ℝ) (fixed : ℝ) (fraction : ℝ) : ℝ :=
  (total - fixed) * (1 - fraction)

theorem property_distribution (total : ℝ) (n : ℕ) : 
  total = 81000 ∧ n = 9 →
  ∀ i : ℕ, i ≤ n →
    distribute total (1000 * (i : ℝ)) (1/10) (remaining_after_distribution total (1000 * ((i-1) : ℝ)) (1/10)) = 
    distribute total (1000 * ((i+1) : ℝ)) (1/10) (remaining_after_distribution total (1000 * (i : ℝ)) (1/10)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_property_distribution_l1230_123079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_diagonal_intersection_l1230_123050

/-- A rectangle in a 2D plane --/
structure Rectangle where
  vertex1 : ℝ × ℝ
  vertex2 : ℝ × ℝ

/-- The intersection point of the diagonals of a rectangle --/
noncomputable def diagonalIntersection (rect : Rectangle) : ℝ × ℝ :=
  ((rect.vertex1.1 + rect.vertex2.1) / 2, (rect.vertex1.2 + rect.vertex2.2) / 2)

/-- Theorem: The diagonal intersection of a rectangle with vertices (2, -3) and (14, 9) is (8, 3) --/
theorem rectangle_diagonal_intersection :
  let rect : Rectangle := { vertex1 := (2, -3), vertex2 := (14, 9) }
  diagonalIntersection rect = (8, 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_diagonal_intersection_l1230_123050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_requires_conditional_structure_l1230_123043

/-- Represents the possible solution types for the equation ax + b = 0 -/
inductive SolutionType
  | Unique (x : ℚ)
  | Infinite
  | None

/-- Solves the equation ax + b = 0 -/
noncomputable def solveEquation (a b : ℚ) : SolutionType :=
  if a = 0 then
    if b = 0 then SolutionType.Infinite
    else SolutionType.None
  else
    SolutionType.Unique (-b / a)

/-- Theorem stating that the solution to ax + b = 0 requires a conditional structure -/
theorem solution_requires_conditional_structure (a b : ℚ) :
  (∃ x, a * x + b = 0) ↔ (solveEquation a b ≠ SolutionType.None) := by
  sorry

#check solution_requires_conditional_structure

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_requires_conditional_structure_l1230_123043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_negative_one_two_zeros_condition_sum_of_zeros_positive_l1230_123070

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * (x + 1)) / Real.exp x + (1 / 2) * x^2

theorem tangent_line_at_negative_one (a : ℝ) :
  a = 1 →
  ∃ m b : ℝ, m = Real.exp 1 - 1 ∧ b = Real.exp 1 - 1/2 ∧
  ∀ x : ℝ, f a (-1) + m * (x + 1) = m * x + b := by sorry

theorem two_zeros_condition (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) ↔ a < 0 := by sorry

theorem sum_of_zeros_positive (a : ℝ) (x₁ x₂ : ℝ) :
  x₁ ≠ x₂ → f a x₁ = 0 → f a x₂ = 0 → x₁ + x₂ > 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_negative_one_two_zeros_condition_sum_of_zeros_positive_l1230_123070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_conditions_l1230_123021

def has_144_divisors (n : ℕ) : Prop :=
  (Finset.filter (λ d ↦ d ∣ n) (Finset.range (n + 1))).card = 144

def has_ten_consecutive_divisors (n : ℕ) : Prop :=
  ∃ k : ℕ, ∀ i : ℕ, i < 10 → (k + i) ∣ n

theorem smallest_n_with_conditions :
  ∀ n : ℕ, n > 0 →
    (has_144_divisors n ∧ has_ten_consecutive_divisors n) →
    n ≥ 110880 :=
by
  sorry

#check smallest_n_with_conditions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_conditions_l1230_123021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_b_c_relationship_l1230_123049

/-- The definite integral of x^2 from 0 to 2 -/
noncomputable def a : ℝ := ∫ x in (0:ℝ)..2, x^2

/-- The definite integral of e^x from 0 to 2 -/
noncomputable def b : ℝ := ∫ x in (0:ℝ)..2, Real.exp x

/-- The definite integral of sin(x) from 0 to 2 -/
noncomputable def c : ℝ := ∫ x in (0:ℝ)..2, Real.sin x

/-- The theorem stating the relationship between a, b, and c -/
theorem a_b_c_relationship : c < a ∧ a < b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_b_c_relationship_l1230_123049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_s_value_l1230_123058

/-- Triangle DEF with given vertices -/
structure Triangle where
  D : ℝ × ℝ := (-2, 6)
  E : ℝ × ℝ := (2, -4)
  F : ℝ × ℝ := (6, -4)

/-- Vertical line intersecting the triangle -/
def vertical_line (s : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = s}

/-- Point of intersection between a line and the vertical line -/
noncomputable def intersection_point (line_start line_end : ℝ × ℝ) (s : ℝ) : ℝ × ℝ :=
  let m := (line_end.2 - line_start.2) / (line_end.1 - line_start.1)
  let b := line_start.2 - m * line_start.1
  (s, m * s + b)

/-- Area of a triangle given three points -/
noncomputable def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((p2.1 - p1.1) * (p3.2 - p1.2) - (p3.1 - p1.1) * (p2.2 - p1.2))

/-- The main theorem -/
theorem intersection_s_value (t : Triangle) (s : ℝ) : 
  let V := intersection_point t.D t.E s
  let W := intersection_point t.D t.F s
  triangle_area t.D V W = 20 → s = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_s_value_l1230_123058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_red_ball_location_l1230_123047

-- Define the types of boxes
inductive Box
| Gold
| Silver
| Copper

-- Define the propositions
def p : Prop := ∃! (b : Box), b = Box.Gold
def q : Prop := ¬∃! (b : Box), b = Box.Silver
def r : Prop := ¬∃! (b : Box), b = Box.Gold

-- Define the theorem
theorem red_ball_location (h1 : ∃! (b : Box), b = Box.Gold ∨ b = Box.Silver ∨ b = Box.Copper) 
                          (h2 : ∃! (prop : Prop), prop = p ∨ prop = q ∨ prop = r) :
  ∃! (b : Box), b = Box.Silver := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_red_ball_location_l1230_123047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_on_ellipse_l1230_123052

theorem max_value_on_ellipse :
  ∃ (M : ℝ), M = Real.sqrt 22 ∧
  (∀ x y : ℝ, 2*x^2 + 3*y^2 = 12 → x + 2*y ≤ M) ∧
  (∃ x y : ℝ, 2*x^2 + 3*y^2 = 12 ∧ x + 2*y = M) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_on_ellipse_l1230_123052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_product_minus_one_is_2003_times_square_l1230_123074

def sequence_a : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | (n + 2) => 2005 * sequence_a (n + 1) - sequence_a n

theorem adjacent_product_minus_one_is_2003_times_square :
  ∀ n : ℕ, ∃ k : ℤ, sequence_a (n + 1) * sequence_a n - 1 = 2003 * k^2 := by
  sorry

#eval sequence_a 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_product_minus_one_is_2003_times_square_l1230_123074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_hyperbola_l1230_123002

/-- The equation of a hyperbola -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 / 2 = 1

/-- A point on the hyperbola -/
noncomputable def point_on_hyperbola : ℝ × ℝ := (Real.sqrt 2, Real.sqrt 2)

/-- The equation of the tangent line -/
def tangent_line (x y : ℝ) : Prop := 2 * x - y - Real.sqrt 2 = 0

/-- Theorem: The equation of the tangent line to the hyperbola at the given point -/
theorem tangent_line_to_hyperbola :
  let (x₀, y₀) := point_on_hyperbola
  hyperbola x₀ y₀ → 
  ∀ x y, tangent_line x y ↔ 
    (y - y₀ = ((2 * x₀) / y₀) * (x - x₀)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_hyperbola_l1230_123002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_is_circle_center_l1230_123089

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - x + y - 1 = 0

/-- The center of the circle -/
noncomputable def circle_center : ℝ × ℝ := (1/2, -1/2)

/-- Theorem stating that the given point is the center of the circle -/
theorem is_circle_center :
  ∀ (x y : ℝ), circle_equation x y ↔ (x - circle_center.1)^2 + (y - circle_center.2)^2 = 3/2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_is_circle_center_l1230_123089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_polynomial_sum_l1230_123069

noncomputable def z : ℂ := (1/2 : ℝ) + (Complex.I * (Real.sqrt 3 / 2))

theorem complex_polynomial_sum : 
  z + 2*z^2 + 3*z^3 + 4*z^4 + 5*z^5 + 6*z^6 = 3 - Complex.I * (3 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_polynomial_sum_l1230_123069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l1230_123091

noncomputable def circle_C2 (x y : ℝ) : Prop := (x - 5)^2 + y^2 = 9

def distToLine (x y : ℝ) : ℝ := |x + 2|

noncomputable def distToCircle (x y : ℝ) : ℝ := ((x - 5)^2 + y^2 - 9).sqrt

theorem parabola_properties :
  ∀ x y : ℝ, y^2 = 20 * x →
    (∀ x' y' : ℝ, circle_C2 x' y' → (x - x')^2 + (y - y')^2 > 0) ∧
    distToLine x y = distToCircle x y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l1230_123091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bela_always_wins_l1230_123054

/-- The game state, representing the current position of the game -/
structure GameState where
  n : ℕ
  chosen : Set ℝ

/-- A valid move in the game -/
def ValidMove (state : GameState) (x : ℝ) : Prop :=
  x ∈ Set.Icc 0 (state.n : ℝ) ∧
  ∀ y ∈ state.chosen, 1 < |x - y| ∧ |x - y| < 3

/-- The winning strategy for Bela -/
def BelaWinningStrategy (n : ℕ) : Prop :=
  n > 10 →
  ∃ (strategy : GameState → ℝ),
    ∀ (state : GameState),
      ValidMove state (strategy state) ∧
      ¬∃ (x : ℝ), ValidMove (GameState.mk state.n (state.chosen ∪ {strategy state})) x

/-- Theorem stating that Bela always wins -/
theorem bela_always_wins :
  ∀ n : ℕ, BelaWinningStrategy n := by
  sorry

#check bela_always_wins

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bela_always_wins_l1230_123054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_mpg_is_fifteen_l1230_123076

/-- The combined rate of miles per gallon for two cars -/
noncomputable def combinedMilesPerGallon (ray_mpg : ℝ) (tom_mpg : ℝ) (ray_miles : ℝ) (tom_miles : ℝ) : ℝ :=
  (ray_miles + tom_miles) / (ray_miles / ray_mpg + tom_miles / tom_mpg)

/-- Theorem: The combined rate of miles per gallon for Ray and Tom's cars is 15 -/
theorem combined_mpg_is_fifteen :
  let ray_mpg : ℝ := 50
  let tom_mpg : ℝ := 8
  let ray_miles : ℝ := 200
  let tom_miles : ℝ := 160
  combinedMilesPerGallon ray_mpg tom_mpg ray_miles tom_miles = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_mpg_is_fifteen_l1230_123076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_of_logarithmic_function_l1230_123093

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 3 + Real.log (2 * x + 3) / Real.log a

theorem fixed_point_of_logarithmic_function (a : ℝ) (ha : a > 0 ∧ a ≠ 1) :
  f a (-1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_of_logarithmic_function_l1230_123093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_place_ties_l1230_123040

/-- Represents a hockey team's performance -/
structure HockeyTeam where
  wins : ℕ
  ties : ℕ

/-- Calculates the points for a hockey team -/
def points (team : HockeyTeam) : ℕ := 2 * team.wins + team.ties

/-- The playoff teams in the hockey league -/
def playoff_teams : Vector HockeyTeam 3 := 
  ⟨[{wins := 12, ties := 0},  -- First-place team (ties unknown)
    {wins := 13, ties := 1},  -- Second-place team
    {wins := 8,  ties := 10}],-- Elsa's team
   rfl⟩

/-- The average points of the playoff teams -/
def average_points : ℚ := 27

theorem first_place_ties : 
  ∃ t : ℕ, 
    playoff_teams.get 0 = {wins := 12, ties := t} ∧ 
    (playoff_teams.map points).toList.sum / 3 = average_points := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_place_ties_l1230_123040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_upper_bound_l1230_123013

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - Real.exp x

-- State the theorem
theorem tangent_line_and_upper_bound :
  (HasDerivAt (f 1) (1 - Real.exp 1) 1) ∧
  (∀ a : ℝ, a ∈ Set.Icc 1 (Real.exp 1 + 1) → ∀ x : ℝ, f a x ≤ x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_upper_bound_l1230_123013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_equation_range_l1230_123078

theorem sine_equation_range (m : ℝ) :
  (∃ x : ℝ, Real.sin x = 1 - m) ↔ 0 ≤ m ∧ m ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_equation_range_l1230_123078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_2022_mod_1000_l1230_123028

/-- Sequence of polynomials P_n(x) -/
def P : ℕ → (ℤ → ℤ)
  | 0 => λ _ => 1  -- Define P_0 to handle the Nat.zero case
  | 1 => λ x => x + 1
  | n + 2 => λ x => ((P (n + 1) x + 1)^5 - (P (n + 1) (-x) + 1)^5) / 2

/-- The main theorem stating that P_2022(1) ≡ 616 (mod 1000) -/
theorem P_2022_mod_1000 : P 2022 1 % 1000 = 616 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_2022_mod_1000_l1230_123028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_for_shifted_sine_l1230_123053

theorem min_omega_for_shifted_sine (ω : ℝ) (h1 : ω > 0) :
  (∀ x : ℝ, Real.sin (ω * (x + π / 3)) = Real.sin (ω * x)) →
  (∀ ω' > 0, (∀ x : ℝ, Real.sin (ω' * (x + π / 3)) = Real.sin (ω' * x)) → ω' ≥ ω) →
  ω = 6 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_for_shifted_sine_l1230_123053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_and_line_equations_l1230_123098

noncomputable section

-- Define the hyperbola C
def hyperbola (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 3

-- Define the line passing through the focus with slope √3
def focus_line (c : ℝ) (x y : ℝ) : Prop := y = Real.sqrt 3 * (x - c)

-- Define the line l
def line_l (k m : ℝ) (x y : ℝ) : Prop := y = k * x + m

-- Define the area of triangle AOB
def triangle_area (a b : ℝ) : ℝ := 3 * Real.sqrt 2

theorem hyperbola_and_line_equations :
  ∀ (a b c : ℝ),
  (a > 0) →
  (b > 0) →
  (∃ (x y : ℝ), hyperbola a b x y ∧ circle_O x y) →
  (∃ (x y : ℝ), focus_line c x y ∧ circle_O x y) →
  (∃ (k m x y : ℝ),
    k < 0 ∧
    m > 0 ∧
    line_l k m x y ∧
    circle_O x y ∧
    x > 0 ∧
    y > 0 ∧
    (∃ (x1 y1 x2 y2 : ℝ),
      hyperbola a b x1 y1 ∧
      hyperbola a b x2 y2 ∧
      line_l k m x1 y1 ∧
      line_l k m x2 y2 ∧
      triangle_area x1 y1 = triangle_area x2 y2)) →
  (a = Real.sqrt 3 ∧ b = 1 ∧ k = -1 ∧ m = Real.sqrt 6) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_and_line_equations_l1230_123098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_overall_ranking_l1230_123039

/-- Represents a judge's ranking of participants -/
def Ranking (α : Type*) := α → α → Bool

/-- Condition: No three judges have conflicting preferences for any three participants -/
def NoConflictingPreferences (α : Type*) (rankings : Finset (Ranking α)) : Prop :=
  ∀ (a b c : α) (j₁ j₂ j₃ : Ranking α),
    j₁ ∈ rankings → j₂ ∈ rankings → j₃ ∈ rankings →
    ¬(j₁ a b ∧ j₁ b c ∧ j₂ b c ∧ j₂ c a ∧ j₃ c a ∧ j₃ a b)

/-- A total ordering of participants -/
def OverallRanking (α : Type*) := α → α → Bool

/-- The overall ranking respects majority preference -/
def RespectsMajority (α : Type*) (rankings : Finset (Ranking α)) (overall : OverallRanking α) : Prop :=
  ∀ a b : α, overall a b = true →
    (rankings.filter (λ j => j a b = true)).card ≥ rankings.card / 2

/-- Main theorem: Given the conditions, there exists a valid overall ranking -/
theorem exists_valid_overall_ranking
  {α : Type*} [Fintype α] (rankings : Finset (Ranking α)) 
  (h_count : rankings.card = 100)
  (h_no_conflicts : NoConflictingPreferences α rankings) :
  ∃ (overall : OverallRanking α), 
    (∀ a b : α, (overall a b = true ∧ overall b a = false) ∨ (overall b a = true ∧ overall a b = false)) ∧
    (∀ a b c : α, overall a b = true → overall b c = true → overall a c = true) ∧
    RespectsMajority α rankings overall :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_overall_ranking_l1230_123039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_card_set_theorem_l1230_123004

def valid_card_set (a b c d : ℕ) : Prop :=
  a ≤ b ∧ b ≤ c ∧ c ≤ d ∧
  ((Finset.filter (· = 9) {a + b, a + c, a + d, b + c, b + d, c + d}).card = 2) ∧
  ((Finset.filter (· < 9) {a + b, a + c, a + d, b + c, b + d, c + d}).card = 2) ∧
  ((Finset.filter (· > 9) {a + b, a + c, a + d, b + c, b + d, c + d}).card = 2)

theorem card_set_theorem :
  ∀ a b c d : ℕ,
    valid_card_set a b c d ↔
      (a, b, c, d) ∈ ({(1, 2, 7, 8), (1, 3, 6, 8), (1, 4, 5, 8),
                      (2, 3, 6, 7), (2, 4, 5, 7), (3, 4, 5, 6)} : Set (ℕ × ℕ × ℕ × ℕ)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_card_set_theorem_l1230_123004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_functions_l1230_123096

-- Define the interval (0,1)
def openInterval : Set ℝ := {x : ℝ | 0 < x ∧ x < 1}

-- Define the functions
noncomputable def f1 : ℝ → ℝ := λ x ↦ Real.sqrt x
noncomputable def f2 : ℝ → ℝ := λ x ↦ Real.log (x + 1) / Real.log (1/2)
def f3 : ℝ → ℝ := λ x ↦ |x - 1|
noncomputable def f4 : ℝ → ℝ := λ x ↦ 2^(x + 1)

-- Define monotonically decreasing function
def MonoDecreasing (f : ℝ → ℝ) (S : Set ℝ) :=
  ∀ x y, x ∈ S → y ∈ S → x < y → f y < f x

-- Theorem statement
theorem monotonic_decreasing_functions :
  (MonoDecreasing f2 openInterval) ∧
  (MonoDecreasing f3 openInterval) ∧
  ¬(MonoDecreasing f1 openInterval) ∧
  ¬(MonoDecreasing f4 openInterval) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_functions_l1230_123096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bijective_increasing_bounds_l1230_123067

-- Define the type for bijective increasing functions from [0,1] to [0,1]
def BijIncreasing := {f : ℝ → ℝ // Function.Bijective f ∧ Monotone f ∧ Set.range f = Set.Icc 0 1}

-- State the theorem
theorem bijective_increasing_bounds (a : ℝ) (h : a ∈ Set.Ioo 0 1) :
  ∃ (m M : ℝ), m = a ∧ M = a + 1 ∧
  (∀ (f : BijIncreasing), m ≤ f.val a + (Function.invFun f.val) a) ∧
  (∀ (f : BijIncreasing), f.val a + (Function.invFun f.val) a ≤ M) ∧
  (∃ (f g : BijIncreasing), f.val a + (Function.invFun f.val) a = m ∧
                            g.val a + (Function.invFun f.val) a = M) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bijective_increasing_bounds_l1230_123067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_is_one_third_l1230_123044

/-- The sum of the series ∑(2^k / (8^k - 1)) for k from 1 to infinity -/
noncomputable def series_sum : ℝ := ∑' k, (2^k : ℝ) / ((8^k : ℝ) - 1)

/-- Theorem stating that the sum of the series is equal to 1/3 -/
theorem series_sum_is_one_third : series_sum = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_is_one_third_l1230_123044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_k_range_l1230_123046

/-- An ellipse with semi-major axis a, semi-minor axis b, and right focus at (3,0) -/
structure Ellipse (a b : ℝ) : Type where
  h1 : a > b
  h2 : b > 0
  h3 : (3 : ℝ) = Real.sqrt (a^2 - b^2)

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse a b) : ℝ := Real.sqrt (a^2 - b^2) / a

theorem ellipse_equation (e : Ellipse a b) (h : eccentricity e = Real.sqrt 3 / 2) :
  a^2 = 12 ∧ b^2 = 3 := by sorry

theorem k_range (e : Ellipse a b) 
  (h : Real.sqrt 2 / 2 < eccentricity e ∧ eccentricity e ≤ Real.sqrt 3 / 2) 
  (k : ℝ) 
  (hk : ∃ A B : ℝ × ℝ, 
    A.2 = k * A.1 ∧ 
    B.2 = k * B.1 ∧ 
    A.1^2 / a^2 + A.2^2 / b^2 = 1 ∧ 
    B.1^2 / a^2 + B.2^2 / b^2 = 1 ∧
    let M := ((A.1 + 3) / 2, A.2 / 2)
    let N := ((B.1 + 3) / 2, B.2 / 2)
    (0 : ℝ) * (M.1 - N.1) = (M.2 - N.2) * (M.1 + N.1)) :
  k ≤ -Real.sqrt 2 / 4 ∨ k ≥ Real.sqrt 2 / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_k_range_l1230_123046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_non_r_fibonacci_l1230_123060

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Lucas sequence -/
def lucas : ℕ → ℕ
  | 0 => 2
  | 1 => 1
  | (n + 2) => lucas (n + 1) + lucas n

/-- α value for Fibonacci and Lucas sequences -/
noncomputable def α : ℝ := (1 + Real.sqrt 5) / 2

/-- β value for Fibonacci and Lucas sequences -/
noncomputable def β : ℝ := (1 - Real.sqrt 5) / 2

/-- r-Fibonacci number -/
def is_r_fibonacci (n : ℕ) (r : ℕ) : Prop :=
  ∃ (coeffs : Finset ℕ), coeffs.card = r ∧ n = (coeffs.sum (λ i ↦ fib i))

/-- Set of positive integers that are not r-Fibonacci for any r in {1, 2, 3, 4, 5} -/
def non_r_fibonacci : Set ℕ :=
  {n : ℕ | n > 0 ∧ ∀ r ∈ Finset.range 5, ¬is_r_fibonacci n (r + 1)}

/-- The main theorem -/
theorem infinite_non_r_fibonacci : Set.Infinite non_r_fibonacci := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_non_r_fibonacci_l1230_123060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_fuel_usage_l1230_123029

/-- Represents the characteristics and journey of a car --/
structure Car where
  speed : ℚ  -- Speed in miles per hour
  fuelEfficiency : ℚ  -- Miles per gallon
  tankCapacity : ℚ  -- Capacity of the fuel tank in gallons
  travelTime : ℚ  -- Travel time in hours

/-- Calculates the fraction of a full tank used given a car's characteristics and journey --/
def fractionOfTankUsed (c : Car) : ℚ :=
  (c.speed * c.travelTime) / (c.fuelEfficiency * c.tankCapacity)

/-- Theorem stating that a car with given characteristics traveling for 5 hours uses 5/6 of its tank --/
theorem car_fuel_usage :
  let c : Car := {
    speed := 50,
    fuelEfficiency := 30,
    tankCapacity := 10,
    travelTime := 5
  }
  fractionOfTankUsed c = 5/6 := by
  sorry

#eval fractionOfTankUsed { speed := 50, fuelEfficiency := 30, tankCapacity := 10, travelTime := 5 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_fuel_usage_l1230_123029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sphere_radius_for_pyramid_pyramid_fits_in_sphere_l1230_123032

/-- The smallest radius of a sphere that can contain a regular quadrilateral pyramid -/
noncomputable def smallest_sphere_radius (base_edge : ℝ) (apothem : ℝ) : ℝ :=
  (base_edge / 2) * Real.sqrt 2

/-- Theorem: The smallest radius of a sphere that can contain a regular quadrilateral pyramid
    with a base edge of 14 and an apothem of 12 is 7√2 -/
theorem smallest_sphere_radius_for_pyramid :
  smallest_sphere_radius 14 12 = 7 * Real.sqrt 2 := by
  sorry

/-- The height of the pyramid -/
noncomputable def pyramid_height (base_edge : ℝ) (apothem : ℝ) : ℝ :=
  Real.sqrt (apothem^2 - (base_edge / 2)^2)

/-- Theorem: The height of the pyramid is less than or equal to the radius of the sphere -/
theorem pyramid_fits_in_sphere (base_edge : ℝ) (apothem : ℝ) :
  pyramid_height base_edge apothem ≤ smallest_sphere_radius base_edge apothem := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sphere_radius_for_pyramid_pyramid_fits_in_sphere_l1230_123032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_at_two_l1230_123015

-- Define the polynomial f(x)
noncomputable def f (x : ℝ) : ℝ := x^4 + 2*x^3 + 3*x^2 + 4*x + 5

-- Define the roots of f(x)
variable (u v w z : ℝ)

-- Assume that u, v, w, z are the roots of f(x)
axiom roots_of_f : f u = 0 ∧ f v = 0 ∧ f w = 0 ∧ f z = 0

-- Define h(x) as a polynomial with leading coefficient 1
-- and roots that are reciprocals of u, v, w, z
noncomputable def h (x : ℝ) : ℝ := (x - 1/u) * (x - 1/v) * (x - 1/w) * (x - 1/z)

-- State the theorem to be proved
theorem h_at_two (u v w z : ℝ) :
  h u v w z 2 = (2*u - 1) * (2*v - 1) * (2*w - 1) * (2*z - 1) / 5 :=
by sorry

-- Additional lemma to show that the product of roots of f is 5
lemma product_of_roots : u * v * w * z = 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_at_two_l1230_123015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_has_four_altitudes_l1230_123088

/-- A parallelogram is a quadrilateral with opposite sides parallel. -/
structure Parallelogram where
  -- We don't need to define the full structure, just declare it exists
  -- as we're only interested in its altitudes

/-- An altitude of a parallelogram is a line segment from a vertex perpendicular to the opposite side or its extension. -/
def Altitude (p : Parallelogram) : Type := sorry

/-- The number of altitudes in a parallelogram -/
def numAltitudes (p : Parallelogram) : ℕ := sorry

/-- Theorem: A parallelogram has exactly 4 altitudes -/
theorem parallelogram_has_four_altitudes (p : Parallelogram) :
  numAltitudes p = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_has_four_altitudes_l1230_123088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_cos_cubed_sin_squared_l1230_123062

theorem integral_cos_cubed_sin_squared (x : Real) :
  (deriv fun x => (Real.sin x ^ 3 / 3) - (Real.sin x ^ 5 / 5)) x = Real.cos x ^ 3 * Real.sin x ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_cos_cubed_sin_squared_l1230_123062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_a_range_l1230_123059

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := x^2 + Real.log x
def g (a x : ℝ) : ℝ := 2*x^2 - a*x

-- Define the symmetry condition
def symmetry_condition (a : ℝ) : Prop :=
  ∃ x : ℝ, x > 0 ∧ f x = g a (-x)

-- State the theorem
theorem symmetry_implies_a_range :
  ∀ a : ℝ, symmetry_condition a → a ≤ -1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_a_range_l1230_123059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shop_owner_profit_l1230_123023

/-- Represents a shop owner's pricing strategy -/
structure ShopOwner where
  buy_cheat : ℚ  -- Percentage of cheating while buying
  sell_cheat : ℚ -- Percentage of cheating while selling

/-- Calculates the percentage profit for a shop owner given their pricing strategy -/
noncomputable def calculate_profit (shop : ShopOwner) : ℚ :=
  let buy_amount := 1 + shop.buy_cheat
  let sell_amount := 1 - shop.sell_cheat
  let sell_price := 1 / sell_amount
  let revenue := sell_price * buy_amount
  (revenue - 1) * 100

/-- Theorem stating that the shop owner's profit is 43.75% -/
theorem shop_owner_profit :
  let shop := ShopOwner.mk (15/100) (20/100)
  calculate_profit shop = 175/4 := by
  sorry

#eval (175 : ℚ) / 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shop_owner_profit_l1230_123023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1230_123022

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x - 2)^2 + (y - 3)^2 = 1

-- Define point A
def point_A : ℝ × ℝ := (3, 5)

-- Define the tangent line equations
def tangent_line_1 (x : ℝ) : Prop := x = 3
def tangent_line_2 (x y : ℝ) : Prop := 3*x - 4*y + 11 = 0

-- Theorem statement
theorem tangent_line_equation :
  ∃ (x y : ℝ), (tangent_line_1 x ∨ tangent_line_2 x y) ∧
  (∀ (a b : ℝ), my_circle a b → ((a - 3)^2 + (b - 5)^2 ≥ (a - x)^2 + (b - y)^2)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1230_123022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1230_123082

-- Define the function f as noncomputable due to Real.sqrt
noncomputable def f (θ : Real) : Real := Real.sqrt 3 * Real.sin θ + Real.cos θ

-- Define the theorem
theorem f_properties :
  -- Part 1
  (∀ θ : Real, 0 ≤ θ ∧ θ ≤ π ∧ 
   Real.cos θ = Real.sqrt 3 / 2 ∧ Real.sin θ = 1 / 2 → 
   f θ = Real.sqrt 3) ∧
  -- Part 2
  (∃ θ : Real, 0 ≤ θ ∧ θ ≤ π ∧ f θ = 2 ∧ θ = π / 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1230_123082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimized_at_critical_point_l1230_123009

/-- The function f(x) = (x-a)^2 + (x-b)^2 + kx -/
noncomputable def f (a b k x : ℝ) : ℝ := (x - a)^2 + (x - b)^2 + k*x

/-- The critical point of f -/
noncomputable def critical_point (a b k : ℝ) : ℝ := (a + b - k/2) / 2

theorem f_minimized_at_critical_point (a b k : ℝ) :
  ∀ x : ℝ, f a b k (critical_point a b k) ≤ f a b k x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimized_at_critical_point_l1230_123009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_vector_l1230_123030

def a : ℝ × ℝ := (1, 3)
def b : ℝ × ℝ := (-1, 2)

theorem projection_vector (a b : ℝ × ℝ) : 
  let proj := ((a.1 * b.1 + a.2 * b.2) / (b.1^2 + b.2^2)) • b
  a = (1, 3) → b = (-1, 2) → proj = (-1, 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_vector_l1230_123030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_collinear_vectors_l1230_123061

/-- Given a triangle ABC with sides a and b, if vectors (a, √3b) and (cosA, sinB) are collinear,
    then the area of the triangle is 3√3/2 when a = √7 and b = 2. -/
theorem triangle_area_collinear_vectors (a b c : ℝ) (A B C : ℝ) :
  a = Real.sqrt 7 →
  b = 2 →
  a * Real.sin B = Real.sqrt 3 * b * Real.cos A →
  (1 / 2) * b * c * Real.sin A = (3 * Real.sqrt 3) / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_collinear_vectors_l1230_123061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_values_l1230_123045

theorem sin_cos_values (x : ℝ) 
  (h1 : Real.sin (x + π) + Real.cos (x - π) = 1/2) 
  (h2 : 0 < x ∧ x < π) : 
  Real.sin x * Real.cos x = -3/8 ∧ Real.sin x - Real.cos x = Real.sqrt 7 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_values_l1230_123045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integral_root_l1230_123019

theorem no_integral_root (f : Polynomial ℤ) (k : ℕ) 
  (h : ∀ i ∈ Finset.range k, ¬ (k : ℤ) ∣ f.eval i) : 
  ∀ n : ℤ, f.eval n ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integral_root_l1230_123019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_problem_l1230_123092

noncomputable def star (A B : ℝ) : ℝ := (A + B) / 3

theorem star_problem : star (star 2 10) 5 = 3 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_problem_l1230_123092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_roots_range_l1230_123099

/-- The function g(x) obtained from f(x) after translation and expansion -/
noncomputable def g (x : ℝ) : ℝ := Real.sin (x - Real.pi/6) + 1/2

/-- The theorem stating the range of k for which g(x) - k = 0 has two different real roots -/
theorem g_roots_range :
  ∀ k : ℝ, (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ ∈ Set.Icc 0 Real.pi ∧ x₂ ∈ Set.Icc 0 Real.pi ∧ 
    g x₁ = k ∧ g x₂ = k) ↔ k ∈ Set.Icc 1 (3/2) :=
by
  sorry

#check g_roots_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_roots_range_l1230_123099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_a_value_l1230_123014

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- Two lines are coincident if and only if they have the same slope and y-intercept -/
axiom coincident_lines {m b₁ b₂ : ℝ} :
  (∀ x y : ℝ, y = m * x + b₁ ↔ y = m * x + b₂) ↔ b₁ = b₂

/-- The slope-intercept form of a line ax + by + c = 0 is y = (-a/b)x - (c/b) -/
noncomputable def slope_intercept_form (a b c : ℝ) (hb : b ≠ 0) :
  ℝ × ℝ := (-a / b, -c / b)

theorem parallel_lines_a_value :
  ∀ a : ℝ,
  (∀ x y : ℝ, a * x + 2 * y + 6 = 0 ↔ x + (a - 1) * y + (a^2 - 1) = 0) →
  (∃ x y : ℝ, a * x + 2 * y + 6 ≠ 0 ∧ x + (a - 1) * y + (a^2 - 1) = 0) →
  a = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_a_value_l1230_123014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_disease_study_results_l1230_123011

-- Define the constants and variables
variable (s : ℕ) -- sample size parameter
variable (t p q : ℝ) -- cost and probability parameters

-- Define the contingency table
noncomputable def under_50_mild : ℝ := (5/6) * s
noncomputable def under_50_severe : ℝ := (1/6) * s
noncomputable def over_50_mild : ℝ := (2/3) * s
noncomputable def over_50_severe : ℝ := (4/3) * s

-- Define the chi-square statistic
noncomputable def chi_square (s : ℕ) : ℝ := (2 * s) / 3

-- Define the average cost for Team A
noncomputable def team_a_cost (t p : ℝ) : ℝ := -3 * t * p^2 + 6 * t

-- Define the average cost for Team B
noncomputable def team_b_cost (t q : ℝ) : ℝ := 6 * t * q^3 - 9 * t * q^2 + 6 * t

-- Theorem statement
theorem disease_study_results (h1 : 0 < p ∧ p < 1) (h2 : 0 < q ∧ q < 1) (h3 : p < q) :
  (∃ (s : ℕ), s ≥ 12 ∧ chi_square s > 6.635) ∧
  (∀ (t : ℝ), t > 0 → team_b_cost t q < team_a_cost t p) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_disease_study_results_l1230_123011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1230_123072

/-- Calculates the speed of a train given its length and time to cross an electric pole. -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  length / time

/-- Theorem stating that a train with given parameters has a specific speed. -/
theorem train_speed_calculation :
  let train_length : ℝ := 500
  let crossing_time : ℝ := 20
  train_speed train_length crossing_time = 25 := by
  -- Unfold the definition of train_speed
  unfold train_speed
  -- Simplify the expression
  simp
  -- Perform the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1230_123072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_of_f_l1230_123051

/-- The function for which we want to find the vertical asymptote -/
noncomputable def f (x : ℝ) : ℝ := (2 * x + 3) / (4 * x - 9)

/-- Theorem stating that the vertical asymptote of f occurs at x = 9/4 -/
theorem vertical_asymptote_of_f :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 9/4| ∧ |x - 9/4| < δ → |f x| > 1/ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_of_f_l1230_123051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_B_value_cos_B_plus_pi_over_4_l1230_123033

-- Define the triangle ABC
variable (A B C : ℝ) -- Angles
variable (a b c : ℝ) -- Sides

-- Define the conditions
axiom triangle_abc : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi
axiom side_relation : c = (Real.sqrt 5 / 2) * b

-- Part 1
theorem cos_B_value (h : C = 2 * B) : Real.cos B = Real.sqrt 5 / 4 := by
  sorry

-- Part 2
-- Define dot product condition
axiom dot_product_equality : a * b * Real.cos C = b * c * Real.cos A

theorem cos_B_plus_pi_over_4 : Real.cos (B + Real.pi/4) = -(Real.sqrt 2 / 10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_B_value_cos_B_plus_pi_over_4_l1230_123033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_ab_l1230_123080

/-- Definition of the sequence x_n -/
def x : ℕ → ℕ → ℕ → ℕ
  | a, b, 0 => a
  | a, b, 1 => b
  | a, b, (n+2) => x a b n + x a b (n+1)

/-- Theorem stating the minimum value of a + b -/
theorem min_sum_ab (a b : ℕ) :
  (∃ n : ℕ, x a b n = 1000) → (a + b ≥ 10 ∧ ∃ a' b' : ℕ, a' + b' = 10 ∧ ∃ n : ℕ, x a' b' n = 1000) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_ab_l1230_123080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l1230_123012

-- Define the set of available digits
def available_digits : Finset Nat := {0, 1, 2, 3, 7}

-- Define a function to check if a number is a valid five-digit even number without repeating digits
def is_valid (n : Nat) : Bool :=
  n ≥ 10000 ∧ n < 100000 ∧ 
  n % 2 = 0 ∧
  (Finset.card (Finset.filter (λ d => d ∈ Nat.digits 10 n) available_digits) = 5)

-- Theorem statement
theorem count_valid_numbers : 
  Finset.card (Finset.filter (λ n => is_valid n) (Finset.range 100000)) = 42 := by
  sorry

#eval Finset.card (Finset.filter (λ n => is_valid n) (Finset.range 100000))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l1230_123012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_starting_number_of_Y_l1230_123081

def set_X : Set ℕ := {n | 1 ≤ n ∧ n ≤ 12}
def set_Y (start : ℕ) : Set ℕ := {n | start ≤ n ∧ n ≤ 20}

theorem starting_number_of_Y : ∃ (start : ℕ),
  Finset.card (Finset.range (20 - start + 1)) = 12 ∧
  Finset.card (Finset.range 12 ∩ Finset.range (20 - start + 1)) = 12 ∧
  start = 9 := by
  sorry

#eval Finset.card (Finset.range 12 ∩ Finset.range 12)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_starting_number_of_Y_l1230_123081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l1230_123073

/-- The asymptotes of the hyperbola x²/4 - y²/3 = 1 are y = ±(√3/2)x -/
theorem hyperbola_asymptotes :
  let hyperbola := {p : ℝ × ℝ | p.1^2/4 - p.2^2/3 = 1}
  let asymptote₁ := {p : ℝ × ℝ | p.2 = (Real.sqrt 3 / 2) * p.1}
  let asymptote₂ := {p : ℝ × ℝ | p.2 = -(Real.sqrt 3 / 2) * p.1}
  (asymptote₁ ∪ asymptote₂) = {p : ℝ × ℝ | ∃ t : ℝ, t ≠ 0 ∧ p ∈ hyperbola ∧ 
    (t * p.1, t * p.2) ∈ hyperbola} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l1230_123073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1230_123057

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) 
  (h1 : b = 3) 
  (h2 : c = 1) 
  (h3 : A = 2 * B) 
  (h4 : 0 < A ∧ A < π) 
  (h5 : Real.cos A = (b^2 + c^2 - a^2) / (2 * b * c)) : 
  a = 2 * Real.sqrt 3 ∧ 
  Real.sin (A + π/4) = (4 - Real.sqrt 2) / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1230_123057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_differ_by_three_l1230_123036

/-- A standard 6-sided die -/
def StandardDie : Finset ℕ := Finset.range 6

/-- The set of all possible outcomes when rolling a standard die twice -/
def TwoRolls : Finset (ℕ × ℕ) := StandardDie.product StandardDie

/-- The condition for two rolls to differ by 3 -/
def DifferBy3 (roll : ℕ × ℕ) : Prop := 
  (roll.1 + 3 = roll.2) ∨ (roll.2 + 3 = roll.1)

/-- The set of all outcomes where the rolls differ by 3 -/
def FavorableOutcomes : Finset (ℕ × ℕ) := 
  TwoRolls.filter (fun roll => (roll.1 + 3 = roll.2) ∨ (roll.2 + 3 = roll.1))

theorem probability_differ_by_three : 
  (FavorableOutcomes.card : ℚ) / TwoRolls.card = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_differ_by_three_l1230_123036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_modulus_z_equation_solution_l1230_123084

open Complex

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Define the complex number z
noncomputable def z : ℂ := ((1 + i)^2 + 3*(1 - i)) / (2 + i)

-- Theorem for the modulus of z
theorem z_modulus : Complex.abs z = Real.sqrt 2 := by sorry

-- Theorem for the existence of a and b
theorem z_equation_solution :
  ∃ (a b : ℝ), z^2 + a • z + b • (1 : ℂ) = 1 + i ∧ a = -3 ∧ b = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_modulus_z_equation_solution_l1230_123084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_runners_separate_theorem_l1230_123095

/-- Represents a runner on a circular track -/
structure Runner where
  speed : ℝ
  initialPosition : ℝ

/-- Represents the state of the race at a given time -/
noncomputable def RaceState (trackLength : ℝ) (runners : Fin 3 → Runner) (t : ℝ) : Fin 3 → ℝ :=
  fun i => (runners i).initialPosition + (runners i).speed * t

/-- Calculates the shortest distance between two points on a circular track -/
noncomputable def circularDistance (trackLength : ℝ) (a b : ℝ) : ℝ :=
  min (abs (a - b)) (trackLength - abs (a - b))

/-- The main theorem stating that there exists a time when all runners are at least L apart -/
theorem runners_separate_theorem
  (L : ℝ)
  (runners : Fin 3 → Runner)
  (h_positive_L : L > 0)
  (h_distinct_speeds : ∀ i j, i ≠ j → (runners i).speed ≠ (runners j).speed)
  (h_same_start : ∀ i j, (runners i).initialPosition = (runners j).initialPosition) :
  ∃ t : ℝ, ∀ i j, i ≠ j →
    circularDistance (3 * L) (RaceState (3 * L) runners t i) (RaceState (3 * L) runners t j) ≥ L :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_runners_separate_theorem_l1230_123095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_product_equality_l1230_123035

-- Define a Point type
structure Point where
  x : ℝ
  y : ℝ

-- Define a Circle type
structure Circle where
  center : Point
  radius : ℝ

-- Define a function to calculate the distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Define membership for Point in Circle
def pointInCircle (p : Point) (c : Circle) : Prop :=
  distance p c.center = c.radius

-- Define the theorem
theorem circle_intersection_product_equality 
  (circle1 circle2 circle3 : Circle)
  (A1 A2 B1 B2 C1 C2 : Point)
  (h1 : pointInCircle A1 circle1 ∧ pointInCircle A1 circle2)
  (h2 : pointInCircle A2 circle1 ∧ pointInCircle A2 circle2)
  (h3 : pointInCircle B1 circle2 ∧ pointInCircle B1 circle3)
  (h4 : pointInCircle B2 circle2 ∧ pointInCircle B2 circle3)
  (h5 : pointInCircle C1 circle3 ∧ pointInCircle C1 circle1)
  (h6 : pointInCircle C2 circle3 ∧ pointInCircle C2 circle1) :
  distance A1 B2 * distance B1 C2 * distance C1 A2 = 
  distance A2 B1 * distance B2 C1 * distance C2 A1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_product_equality_l1230_123035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_valid_student_counts_l1230_123016

def is_valid_student_count (s : ℕ) : Bool :=
  130 ≤ s && s ≤ 220 && (s - 2) % 8 = 0

def valid_student_counts : List ℕ :=
  (List.range (220 - 130 + 1)).map (λ x => x + 130) |>.filter is_valid_student_count

theorem sum_of_valid_student_counts :
  valid_student_counts.sum = 2088 := by
  sorry

#eval valid_student_counts.sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_valid_student_counts_l1230_123016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_distance_theorem_l1230_123077

/-- Represents a journey with two legs at different speeds -/
structure Journey where
  totalTime : ℚ
  speed1 : ℚ
  speed2 : ℚ

/-- Calculates the total distance of a journey -/
def totalDistance (j : Journey) : ℚ :=
  (j.speed1 * j.totalTime / 2) + (j.speed2 * j.totalTime / 2)

theorem journey_distance_theorem (j : Journey) 
  (h1 : j.totalTime = 10)
  (h2 : j.speed1 = 21)
  (h3 : j.speed2 = 24) :
  totalDistance j = 224 := by
  sorry

#eval totalDistance { totalTime := 10, speed1 := 21, speed2 := 24 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_distance_theorem_l1230_123077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_subtended_angles_l1230_123005

-- Define a circle
def Circle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}

-- Define a triangle inscribed in a circle
def InscribedTriangle (c : Set (ℝ × ℝ)) : Set ((ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) := 
  {t | t.1 ∈ c ∧ t.2.1 ∈ c ∧ t.2.2 ∈ c}

-- Define the angle subtended by an arc
noncomputable def AngleSubtendedByArc (t : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) (a : ℝ) : ℝ := sorry

-- Theorem statement
theorem sum_of_subtended_angles (t : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) (h : t ∈ InscribedTriangle Circle) :
  ∃ a b c : ℝ,
    AngleSubtendedByArc t a + AngleSubtendedByArc t b + AngleSubtendedByArc t c = 180 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_subtended_angles_l1230_123005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_sum_equals_twelve_l1230_123001

-- Define the common logarithm (base 10)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem power_sum_equals_twelve (m n : ℝ) (h1 : lg 3 = m) (h2 : lg 4 = n) : 
  (10 : ℝ)^(m + n) = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_sum_equals_twelve_l1230_123001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculator_multiplication_product_representation_l1230_123037

/-- Represents a number in scientific notation as used by the calculator -/
structure CalcNumber where
  coefficient : ℚ
  exponent : ℤ

/-- Converts a CalcNumber to a rational number -/
def toRational (n : CalcNumber) : ℚ := n.coefficient * (10 : ℚ) ^ n.exponent

/-- Multiplies two CalcNumbers -/
def multiply (a b : CalcNumber) : CalcNumber :=
  { coefficient := a.coefficient * b.coefficient,
    exponent := a.exponent + b.exponent }

theorem calculator_multiplication (a b : CalcNumber) :
  toRational (multiply a b) = toRational a * toRational b := by
  sorry

theorem product_representation :
  let a := CalcNumber.mk 2 3
  let b := CalcNumber.mk 3 2
  toRational (multiply a b) = 6 * (10 : ℚ) ^ 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculator_multiplication_product_representation_l1230_123037
