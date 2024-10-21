import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solutions_l105_10543

noncomputable def system_of_equations (x y : ℝ) : Prop :=
  (3 : ℝ)^y * (9 : ℝ)^x = 81 ∧ Real.log ((y + x)^2) - Real.log x = 2 * Real.log 3

theorem system_solutions :
  ∃! (s : Set (ℝ × ℝ)), s = {(1, 2), (16, -28)} ∧
    ∀ (x y : ℝ), (x > 0 ∧ y + x ≠ 0 ∧ system_of_equations x y) ↔ (x, y) ∈ s :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solutions_l105_10543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_nonagon_interior_angle_proof_l105_10537

/-- The measure of one interior angle of a regular nonagon is 140 degrees. -/
noncomputable def regular_nonagon_interior_angle : ℝ :=
  140

/-- A regular nonagon has 9 sides. -/
def regular_nonagon_sides : ℕ := 9

/-- The sum of interior angles of a polygon with n sides is (n-2) * 180 degrees. -/
noncomputable def polygon_interior_angle_sum (n : ℕ) : ℝ := (n - 2 : ℝ) * 180

/-- The measure of one interior angle of a regular polygon is the sum of interior angles divided by the number of sides. -/
noncomputable def regular_polygon_interior_angle (n : ℕ) : ℝ := 
  (polygon_interior_angle_sum n) / n

/-- Proof that the measure of one interior angle of a regular nonagon is 140 degrees. -/
theorem regular_nonagon_interior_angle_proof : 
  regular_polygon_interior_angle regular_nonagon_sides = regular_nonagon_interior_angle :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_nonagon_interior_angle_proof_l105_10537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_sin_cos_product_l105_10584

-- Define the function f(x) = sin x cos x
noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos x

-- State the theorem
theorem min_positive_period_sin_cos_product (T : ℝ) :
  (∀ x, f (x + T) = f x) ∧ 
  (∀ S, 0 < S → S < T → ∃ x, f (x + S) ≠ f x) →
  T = π :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_sin_cos_product_l105_10584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_k_subset_prime_sum_l105_10526

def A : Finset Nat := Finset.range 16

def is_prime (n : Nat) : Prop := Nat.Prime n

def subset_has_prime_sum_pair (S : Finset Nat) : Prop :=
  ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ is_prime (a^2 + b^2)

theorem least_k_subset_prime_sum : 
  (∀ S : Finset Nat, S ⊆ A → S.card = 9 → subset_has_prime_sum_pair S) ∧
  (∀ k : Nat, k < 9 → ∃ S : Finset Nat, S ⊆ A ∧ S.card = k ∧ ¬subset_has_prime_sum_pair S) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_k_subset_prime_sum_l105_10526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l105_10520

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + 1) - a * x

-- State the theorem
theorem f_properties (a : ℝ) :
  (f a 1 = Real.log 2 - 1) →
  (∃ m : ℝ, m = -1/2 ∧ HasDerivAt (f a) m 1) ∧
  (a > 0 → (∀ x : ℝ, x > -1 ∧ x < (1-a)/a → HasDerivAt (f a) (1/(x+1) - a) x ∧ 1/(x+1) - a > 0) ∧
           (∀ x : ℝ, x > (1-a)/a → HasDerivAt (f a) (1/(x+1) - a) x ∧ 1/(x+1) - a < 0)) ∧
  (∀ x : ℝ, x > 0 → (1 + 1/x)^x < 3) ∧
  (∃ x : ℝ, x > 0 ∧ (1 + 1/x)^x > 2) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l105_10520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_approaches_circle_l105_10566

-- Define the ellipse parameters
noncomputable def ellipse_equation (x y a : ℝ) : Prop :=
  x^2 / (4*a) + y^2 / (a^2 - 1) = 1

-- Define the eccentricity of the ellipse
noncomputable def eccentricity (a : ℝ) : ℝ :=
  Real.sqrt (1 - (a^2 - 1) / (4*a))

-- Theorem statement
theorem ellipse_approaches_circle (a₁ a₂ : ℝ) 
  (h₁ : 1 < a₁) (h₂ : a₁ < a₂) (h₃ : a₂ < 2 + Real.sqrt 5) :
  eccentricity a₂ < eccentricity a₁ := by
  sorry

-- Additional lemma to show that eccentricity decreases as 'a' increases
lemma eccentricity_decreases (a : ℝ) (h : 1 < a ∧ a < 2 + Real.sqrt 5) :
  ∀ ε > 0, ∃ δ > 0, ∀ a' : ℝ, a < a' ∧ a' - a < δ → eccentricity a' < eccentricity a + ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_approaches_circle_l105_10566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loss_percentage_example_l105_10508

/-- Given a cost price and a selling price, calculate the percentage of loss. -/
noncomputable def loss_percentage (cost_price selling_price : ℚ) : ℚ :=
  ((cost_price - selling_price) / cost_price) * 100

/-- Theorem: The loss percentage for an item with cost price 1300 and selling price 1040 is 20%. -/
theorem loss_percentage_example : loss_percentage 1300 1040 = 20 := by
  -- Unfold the definition of loss_percentage
  unfold loss_percentage
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_loss_percentage_example_l105_10508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_corner_path_length_theorem_l105_10586

/-- Represents the dimensions of the rectangular box --/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a path through the corners of the box --/
structure CornerPath where
  length : ℝ
  visits_all_corners : Bool
  starts_and_ends_same_corner : Bool

/-- The maximum length of a valid corner path in the box --/
noncomputable def max_corner_path_length (box : BoxDimensions) : ℝ :=
  4 * Real.sqrt 6 + 2 * Real.sqrt 5

/-- Theorem stating the maximum length of a valid corner path --/
theorem max_corner_path_length_theorem (box : BoxDimensions) :
  box.length = 1 ∧ box.width = Real.sqrt 2 ∧ box.height = Real.sqrt 3 →
  ∀ path : CornerPath,
    path.visits_all_corners ∧ path.starts_and_ends_same_corner →
    path.length ≤ max_corner_path_length box :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_corner_path_length_theorem_l105_10586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcenter_property_l105_10590

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the circumcenter
noncomputable def circumcenter (t : Triangle) : ℝ × ℝ := sorry

-- Define vector operations
def vec_add (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
def vec_scale (a : ℝ) (u : ℝ × ℝ) : ℝ × ℝ := (a * u.1, a * u.2)

-- State the theorem
theorem circumcenter_property (t : Triangle) (x y : ℝ) :
  let O := circumcenter t
  let AB := vec_add t.B (vec_scale (-1) t.A)
  let AC := vec_add t.C (vec_scale (-1) t.A)
  let AO := vec_add O (vec_scale (-1) t.A)
  -- Given conditions
  (AB.1^2 + AB.2^2 = 64) →  -- AB = 8
  (AC.1^2 + AC.2^2 = 144) →  -- AC = 12
  (AB.1 * AC.1 + AB.2 * AC.2 = 48) →  -- ∠A = π/3
  (AO = vec_add (vec_scale x AB) (vec_scale y AC)) →
  -- Conclusion
  (6 * x + 9 * y = 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcenter_property_l105_10590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l105_10546

-- Define the equation of the circle
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y - 6 = 0

-- Define the center of the circle
def circle_center : ℝ × ℝ := (-1, 2)

-- Define the radius of the circle
noncomputable def circle_radius : ℝ := Real.sqrt 11

-- Theorem statement
theorem circle_properties :
  ∀ (x y : ℝ), circle_equation x y ↔ 
  (x - circle_center.1)^2 + (y - circle_center.2)^2 = circle_radius^2 := by
  sorry

#check circle_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l105_10546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_h_l105_10519

noncomputable def h (x : ℝ) : ℝ := (5*x - 2) / (2*x - 10)

theorem domain_of_h :
  Set.range h = {y | ∃ x : ℝ, x ≠ 5 ∧ y = h x} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_h_l105_10519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_one_comprehensive_survey_l105_10512

/-- Represents a type of survey. -/
inductive SurveyType
  | Comprehensive
  | Sample

/-- Represents a survey option. -/
structure SurveyOption where
  description : String
  type : SurveyType

/-- Determines if a survey option is suitable for a comprehensive survey. -/
def isSuitableForComprehensiveSurvey (option : SurveyOption) : Prop :=
  option.type = SurveyType.Comprehensive

/-- The list of survey options given in the problem. -/
def surveyOptions : List SurveyOption :=
  [
    { description := "Survey on the number of waste batteries discarded in the city every day", type := SurveyType.Sample },
    { description := "Survey on the quality of ice cream in the cold drink market", type := SurveyType.Sample },
    { description := "Survey on the current mental health status of middle school students nationwide", type := SurveyType.Sample },
    { description := "Survey on the components of the first large civil helicopter in China", type := SurveyType.Comprehensive }
  ]

/-- Theorem stating that only one survey option is suitable for a comprehensive survey. -/
theorem only_one_comprehensive_survey :
  ∃! option, option ∈ surveyOptions ∧ isSuitableForComprehensiveSurvey option :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_one_comprehensive_survey_l105_10512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_work_alone_days_l105_10506

noncomputable def combined_rate : ℝ := 1 / 40

noncomputable def a_rate : ℝ := 1 / 20

noncomputable def work_done_together : ℝ := 10 * combined_rate

noncomputable def remaining_work : ℝ := 1 - work_done_together

theorem a_work_alone_days : 
  ∃ (days : ℝ), days * a_rate = remaining_work ∧ days = 15 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_work_alone_days_l105_10506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_odd_coprime_squares_exist_l105_10502

theorem infinite_odd_coprime_squares_exist : 
  ∃ (m n : ℕ → ℕ) (r : ℕ → ℕ), 
    (∀ k, Odd (m k)) ∧ 
    (∀ k, m k < m (k + 1) ∧ n k < n (k + 1)) ∧ 
    (∀ k, Nat.Coprime (m k) (n k)) ∧ 
    (∀ k, (m k)^4 - 2*(n k)^4 = (r k)^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_odd_coprime_squares_exist_l105_10502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_inequality_l105_10521

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the isosceles property
def isIsosceles (t : Triangle) : Prop :=
  dist t.A t.B = dist t.B t.C

-- Define the angle at vertex B
noncomputable def angleBMeasure (t : Triangle) : ℝ :=
  Real.arccos ((dist t.A t.B ^ 2 + dist t.B t.C ^ 2 - dist t.A t.C ^ 2) / (2 * dist t.A t.B * dist t.B t.C))

-- Define the lengths of sides
def sideLength (p q : ℝ × ℝ) : ℝ :=
  dist p q

-- Theorem statement
theorem isosceles_triangle_inequality (t : Triangle) 
  (h1 : isIsosceles t) 
  (h2 : angleBMeasure t = 20 * π / 180) : 
  3 * sideLength t.A t.C > sideLength t.A t.B ∧ 
  2 * sideLength t.A t.C < sideLength t.A t.B := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_inequality_l105_10521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l105_10525

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (4 - x) / (x + 1)

-- State the theorem
theorem range_of_x (x : ℝ) : 
  (∃ y : ℝ, f x = y) ↔ (x ≤ 4 ∧ x ≠ -1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l105_10525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_Q_to_C2_l105_10509

noncomputable section

-- Define the Cartesian coordinate system
variable (x y t m k : ℝ)

-- Define line l1
def l1 (t k : ℝ) : ℝ × ℝ := (t - Real.sqrt 3, k * t)

-- Define line l2
def l2 (m k : ℝ) : ℝ × ℝ := (Real.sqrt 3 - m, m / (3 * k))

-- Define curve C1
def C1 (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1 ∧ y ≠ 0

-- Define line C2 in polar coordinates
def C2_polar (p θ : ℝ) : Prop := p * Real.sin (θ + Real.pi / 4) = 4 * Real.sqrt 2

-- Define line C2 in Cartesian coordinates
def C2_cartesian (x y : ℝ) : Prop := x + y = 8

-- Define a point Q on curve C1
def Q (α : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos α, Real.sin α)

-- Define the distance function between a point and line C2
def distance_to_C2 (x y : ℝ) : ℝ :=
  |x + y - 8| / Real.sqrt 2

-- State the theorem for the minimum distance
theorem min_distance_Q_to_C2 :
  ∃ (α : ℝ), ∀ (β : ℝ),
    distance_to_C2 (Q α).1 (Q α).2 ≤ distance_to_C2 (Q β).1 (Q β).2 ∧
    distance_to_C2 (Q α).1 (Q α).2 = 3 * Real.sqrt 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_Q_to_C2_l105_10509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_determinant_of_matrix_l105_10542

-- Define the vectors v and w
def v : Fin 3 → ℝ := ![3, -2, 2]
def w : Fin 3 → ℝ := ![-1, 4, 1]

-- Define the cross product of v and w
def v_cross_w : Fin 3 → ℝ := ![
  v 1 * w 2 - v 2 * w 1,
  v 2 * w 0 - v 0 * w 2,
  v 0 * w 1 - v 1 * w 0
]

-- Theorem statement
theorem max_determinant_of_matrix (u : Fin 3 → ℝ) (h : Finset.univ.sum (fun i => u i ^ 2) = 1) :
  (∀ (u' : Fin 3 → ℝ), Finset.univ.sum (fun i => u' i ^ 2) = 1 → 
    |Matrix.det (Matrix.of ![u', v, w])| ≤ |Matrix.det (Matrix.of ![u, v, w])|) →
  |Matrix.det (Matrix.of ![u, v, w])| = Real.sqrt 345 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_determinant_of_matrix_l105_10542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_is_three_l105_10552

/-- A circle with radius 1 passing through (2,3) -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ := 1
  passes_through : center.1^2 + center.2^2 = (2 - center.1)^2 + (3 - center.2)^2

/-- The line 3x-4y-4=0 -/
def line (x y : ℝ) : Prop := 3*x - 4*y - 4 = 0

/-- Distance from a point to the line -/
noncomputable def distanceToLine (p : ℝ × ℝ) : ℝ :=
  |3*p.1 - 4*p.2 - 4| / Real.sqrt (3^2 + 4^2)

/-- The maximum distance from the center of the circle to the line is 3 -/
theorem max_distance_is_three :
  ∃ (c : Circle), ∀ (other : Circle), distanceToLine c.center ≤ 3 ∧ 
  distanceToLine other.center ≤ distanceToLine c.center := by
  sorry

#check max_distance_is_three

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_is_three_l105_10552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equals_half_e_minus_one_l105_10567

open Real MeasureTheory

noncomputable def f (x : ℝ) : ℝ := 
  exp (sqrt ((2 - x) / (2 + x))) / ((2 + x) * sqrt (4 - x^2))

theorem integral_equals_half_e_minus_one : 
  ∫ x in (Set.Icc 0 2), f x = (exp 1 - 1) / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equals_half_e_minus_one_l105_10567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l105_10513

noncomputable def f (x : ℝ) := (3 * x) / (x + 1)

theorem f_properties :
  let a := (2 : ℝ)
  let b := (5 : ℝ)
  (∀ x y, x ∈ Set.Icc a b → y ∈ Set.Icc a b → x < y → f x < f y) ∧
  (∀ x, x ∈ Set.Icc a b → f x ≥ f a) ∧
  (∀ x, x ∈ Set.Icc a b → f x ≤ f b) ∧
  f a = 2 ∧
  f b = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l105_10513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_l105_10575

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem function_symmetry 
  (ω φ : ℝ) 
  (h_ω_pos : ω > 0) 
  (h_φ_bound : |φ| < π/2) 
  (h_period : ∀ x, f ω φ (x + π) = f ω φ x) 
  (h_odd : ∀ x, f ω φ (x + π/6) = -f ω φ (-x - π/6)) :
  ∀ x, f ω φ (5*π/12 + x) = f ω φ (5*π/12 - x) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_l105_10575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_difference_existence_of_eleven_l105_10560

theorem smallest_positive_difference : 
  ∀ (k l : ℕ), (36:ℤ)^k - (5:ℤ)^l ≥ 11 ∨ (5:ℤ)^l - (36:ℤ)^k ≥ 11 := by
  sorry

theorem existence_of_eleven :
  ∃ (k l : ℕ), (36:ℤ)^k - (5:ℤ)^l = 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_difference_existence_of_eleven_l105_10560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eldest_child_age_l105_10596

theorem eldest_child_age (ages : Fin 4 → ℕ) : 
  (∀ i : Fin 3, ages i.succ = ages i + 4) →
  (Finset.sum Finset.univ ages) = 48 →
  ages 3 = 18 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eldest_child_age_l105_10596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_rectangle_area_is_340_l105_10534

/-- Represents a rectangle divided into four smaller rectangles -/
structure DividedRectangle where
  total_perimeter : ℝ
  area1 : ℝ
  area2 : ℝ
  area3 : ℝ

/-- Calculates the area of the fourth rectangle -/
noncomputable def fourth_rectangle_area (r : DividedRectangle) : ℝ :=
  (r.total_perimeter / 2)^2 - r.area1 - r.area2 - r.area3

/-- Theorem stating that the area of the fourth rectangle is 340 -/
theorem fourth_rectangle_area_is_340 (r : DividedRectangle) 
  (h1 : r.total_perimeter = 40)
  (h2 : r.area1 = 20)
  (h3 : r.area2 = 25)
  (h4 : r.area3 = 15) : 
  fourth_rectangle_area r = 340 := by
  sorry

#check fourth_rectangle_area_is_340

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_rectangle_area_is_340_l105_10534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l105_10532

-- Define the hyperbola
structure Hyperbola where
  a : ℝ
  b : ℝ

-- Define the parabola
def Parabola := {p : ℝ × ℝ | p.1^2 = 4 * Real.sqrt 7 * p.2}

-- Define the asymptote condition
def asymptote_condition (h : Hyperbola) : Prop :=
  h.a / h.b = Real.sqrt 3 / 2

-- Define the set of foci for a hyperbola
def set_of_foci (h : Hyperbola) : Set (ℝ × ℝ) :=
  {f | (f.1 - h.a)^2 / h.a^2 - (f.2 - h.b)^2 / h.b^2 = 1}

-- Define the focus-directrix condition
def focus_directrix_condition (h : Hyperbola) : Prop :=
  ∃ (f : ℝ × ℝ), f.2 = -Real.sqrt 7 ∧ f ∈ set_of_foci h

-- State the theorem
theorem hyperbola_equation (h : Hyperbola) 
  (asymp_cond : asymptote_condition h) 
  (focus_dir_cond : focus_directrix_condition h) :
  ∀ (x y : ℝ), y^2 / 3 - x^2 / 4 = 1 ↔ (x, y) ∈ {p | p.1^2 / h.a^2 - p.2^2 / h.b^2 = 1} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l105_10532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_to_parabola_l105_10529

/-- The parabola defined by x = y^2/3 -/
noncomputable def parabola (y : ℝ) : ℝ := y^2 / 3

/-- The point we're measuring distance from -/
def point : ℝ × ℝ := (5, 10)

/-- The distance function between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The shortest distance from the point to the parabola -/
theorem shortest_distance_to_parabola :
  ∃ y : ℝ, ∀ z : ℝ, distance point (parabola y, y) ≤ distance point (parabola z, z) ∧
  distance point (parabola y, y) = Real.sqrt 53 := by
  sorry

#check shortest_distance_to_parabola

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_to_parabola_l105_10529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_like_terms_power_l105_10574

theorem like_terms_power (a b : ℝ) (m n : ℤ) : 
  (∃ (k : ℝ), 3 * a^2 * b^(n-1) = k * (-1/2) * a^(m+3) * b^2) → 
  (m : ℝ)^(n : ℝ) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_like_terms_power_l105_10574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_is_two_l105_10511

/-- The distance from a point in polar coordinates to a line in polar form --/
noncomputable def distance_polar_point_to_line (r : ℝ) (θ : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  let x := r * Real.cos θ
  let y := r * Real.sin θ
  let line_eq := λ (x y : ℝ) => Real.sqrt 3 * x - y + 8
  |line_eq x y| / Real.sqrt (3^2 + 1^2)

/-- The theorem stating that the distance from (2, 5π/6) to ρsin(θ - π/3) = 4 is 2 --/
theorem distance_to_line_is_two :
  distance_polar_point_to_line 2 (5 * Real.pi / 6) (Real.pi / 3) 4 = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_is_two_l105_10511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_9_is_99_l105_10535

/-- An arithmetic sequence with specific conditions -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_1_4_7 : a 1 + a 4 + a 7 = 39
  sum_3_6_9 : a 3 + a 6 + a 9 = 27

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

/-- Theorem: The sum of the first 9 terms of the specified arithmetic sequence is 99 -/
theorem sum_9_is_99 (seq : ArithmeticSequence) : sum_n seq 9 = 99 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_9_is_99_l105_10535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_F_satisfies_conditions_l105_10591

noncomputable def F (x : ℝ) : ℝ := (4/3) * x^3 - 9/x - 35

-- Theorem statement
theorem F_satisfies_conditions :
  (F 3 = -2) ∧ 
  (∀ x : ℝ, x ≠ 0 → deriv F x = 4 * x^2 + 9 / x^2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_F_satisfies_conditions_l105_10591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spectral_density_from_correlation_function_l105_10563

/-- The correlation function of a stationary random function -/
noncomputable def correlation_function (τ : ℝ) : ℝ := Real.exp (-abs τ) * Real.cos τ

/-- The spectral density of a stationary random function -/
noncomputable def spectral_density (ω : ℝ) : ℝ := (2 + ω^2) / (Real.pi * (1 + (1 - ω)^2) * (1 + (1 + ω)^2))

/-- Theorem stating the relationship between the correlation function and spectral density -/
theorem spectral_density_from_correlation_function :
  ∀ ω : ℝ, spectral_density ω = 
    (1 / (2 * Real.pi)) * ∫ (τ : ℝ), correlation_function τ * Complex.exp (-Complex.I * ω * τ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_spectral_density_from_correlation_function_l105_10563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slant_asymptote_sum_l105_10504

/-- The rational function f(x) = (3x^2 + 5x - 11) / (x - 4) -/
noncomputable def f (x : ℝ) : ℝ := (3 * x^2 + 5 * x - 11) / (x - 4)

/-- The slope of the slant asymptote of f(x) -/
def m : ℝ := 3

/-- The y-intercept of the slant asymptote of f(x) -/
def b : ℝ := 17

/-- Theorem: The sum of the slope and y-intercept of the slant asymptote of f(x) is 20 -/
theorem slant_asymptote_sum : m + b = 20 := by
  rw [m, b]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slant_asymptote_sum_l105_10504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_and_range_l105_10593

-- Define the complex number z
variable (z : ℂ)

-- Define the real number m
variable (m : ℝ)

-- Define z₁ in terms of z and m
noncomputable def z₁ (z : ℂ) (m : ℝ) : ℂ := z / (m + Complex.I ^ 2023)

-- State the theorem
theorem complex_number_and_range :
  (z - Complex.I ∈ Set.range (Complex.ofReal)) →
  ((z - 3 * Complex.I) / (-2 - Complex.I) ∈ {w : ℂ | w.re = 0}) →
  (z₁ z m).re < 0 →
  (z₁ z m).im > 0 →
  (z = 1 + Complex.I) ∧ (-1 < m ∧ m < 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_and_range_l105_10593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_points_of_f_l105_10541

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x - Real.pi / 6)

theorem symmetry_points_of_f :
  ∃ (α₁ α₂ : ℝ), α₁ ∈ Set.Ioo 0 Real.pi ∧ α₂ ∈ Set.Ioo 0 Real.pi ∧
  (∀ x : ℝ, f (α₁ + x) = f (α₁ - x)) ∧
  (∀ x : ℝ, f (α₂ + x) = f (α₂ - x)) ∧
  α₁ = Real.pi / 3 ∧ α₂ = 5 * Real.pi / 6 ∧
  (∀ α : ℝ, α ∈ Set.Ioo 0 Real.pi → (∀ x : ℝ, f (α + x) = f (α - x)) → (α = α₁ ∨ α = α₂)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_points_of_f_l105_10541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_degree_of_q_l105_10556

/-- The denominator of our rational function -/
def p (x : ℝ) : ℝ := 2*x^5 + x^4 - 7*x^2 + 1

/-- A general polynomial function -/
noncomputable def q (x : ℝ) : ℝ := sorry

/-- Our rational function -/
noncomputable def f (x : ℝ) : ℝ := q x / p x

/-- Definition of having a horizontal asymptote -/
def has_horizontal_asymptote (f : ℝ → ℝ) : Prop :=
  ∃ L : ℝ, ∀ ε > 0, ∃ M : ℝ, ∀ x > M, |f x - L| < ε

/-- The theorem stating the largest possible degree of q(x) -/
theorem largest_degree_of_q :
  has_horizontal_asymptote f →
  ∃ n : ℕ, (∀ m : ℕ, (∀ x : ℝ, ∃ k : ℝ, q x = k * x^m) → m ≤ n) ∧ n = 5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_degree_of_q_l105_10556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2alpha_value_l105_10524

theorem cos_2alpha_value (α : ℝ) (h1 : α ∈ Set.Ioo 0 π) (h2 : Real.sin α + Real.cos α = 1/2) : 
  Real.cos (2 * α) = -Real.sqrt 7 / 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2alpha_value_l105_10524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_remainder_l105_10503

-- Define the polynomial and divisor
def f (x : ℝ) : ℝ := 3*x^7 - 2*x^5 + 5*x^3 - 6
def g (x : ℝ) : ℝ := x^2 + 3*x + 2

-- Define the quotient and remainder
noncomputable def q (x : ℝ) : ℝ := sorry
def r (x : ℝ) : ℝ := 354*x + 342

-- State the theorem
theorem polynomial_division_remainder :
  ∀ x, f x = g x * q x + r x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_remainder_l105_10503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_factor_less_than_9_of_72_l105_10562

def factors (n : ℕ) : Finset ℕ :=
  Finset.filter (λ x => n % x = 0) (Finset.range (n + 1))

def factors_less_than (n k : ℕ) : Finset ℕ :=
  Finset.filter (λ x => x < k) (factors n)

theorem probability_factor_less_than_9_of_72 :
  (Finset.card (factors_less_than 72 9) : ℚ) / (Finset.card (factors 72) : ℚ) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_factor_less_than_9_of_72_l105_10562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_passing_time_theorem_l105_10583

-- Define the given parameters
noncomputable def train1_speed : ℝ := 60 -- km/hr
noncomputable def train1_crossing_time : ℝ := 7 -- seconds
noncomputable def train2_speed : ℝ := 80 -- km/hr

-- Define the unknown length of Train 2
variable (L2 : ℝ)

-- Define the calculated values
noncomputable def train1_speed_mps : ℝ := train1_speed * 1000 / 3600
noncomputable def train2_speed_mps : ℝ := train2_speed * 1000 / 3600
noncomputable def train1_length : ℝ := train1_speed_mps * train1_crossing_time
noncomputable def relative_speed : ℝ := train1_speed_mps + train2_speed_mps

-- State the theorem
theorem passing_time_theorem :
  (train1_length + L2) / relative_speed = (116.69 + L2) / 38.89 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_passing_time_theorem_l105_10583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_g_50_52_l105_10533

def g (x : ℤ) : ℤ := x^3 - 2*x^2 + x + 2023

theorem gcd_g_50_52 : Int.gcd (g 50) (g 52) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_g_50_52_l105_10533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l105_10559

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.sin x / (1 + Real.cos x)

-- State the theorem
theorem f_properties :
  -- 1. The period of f(x) is 2π
  (∀ x, f (x + 2 * Real.pi) = f x) ∧
  -- 2. f(x) is monotonically increasing on (0, π)
  (∀ x y, 0 < x ∧ x < y ∧ y < Real.pi → f x < f y) ∧
  -- 3. The center of symmetry of f(x) is (kπ, 0), where k ∈ ℤ
  (∀ k : ℤ, ∀ x, f (k * Real.pi + x) = -f (k * Real.pi - x)) ∧
  -- 4. f(x) is NOT monotonically decreasing on (π, 2π)
  ¬(∀ x y, Real.pi < x ∧ x < y ∧ y < 2 * Real.pi → f x > f y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l105_10559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_13245_4999991_l105_10523

noncomputable def round_to_nearest (x : ℝ) : ℤ :=
  if x - ⌊x⌋ < 0.5 then ⌊x⌋ else ⌈x⌉

theorem round_13245_4999991 :
  round_to_nearest 13245.4999991 = 13245 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_13245_4999991_l105_10523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_inequality_l105_10505

noncomputable def circle_area (r : ℝ) : ℝ := Real.pi * r^2

noncomputable def circle_circumference (r : ℝ) : ℝ := 2 * Real.pi * r

theorem circle_area_inequality (r R : ℝ) (hr : r = 5) (hR : R = 10) :
  let mean_circumference := (circle_circumference r + circle_circumference R) / 2
  let mean_radius := mean_circumference / (2 * Real.pi)
  let mean_area := (circle_area r + circle_area R) / 2
  circle_area mean_radius ≠ mean_area ∧
  (abs (circle_area mean_radius - mean_area) / mean_area) * 100 = 10 := by
  sorry

#check circle_area_inequality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_inequality_l105_10505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_and_maximum_l105_10569

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.log (x + m) - x

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := Real.log (x + 1) - x - x^2

-- Define HasExtremeValue
def HasExtremeValue (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ ε > 0, ∀ y, |y - x| < ε → f y ≤ f x ∨ f y ≥ f x

theorem extreme_value_and_maximum (m : ℝ) :
  (∃ (x : ℝ), HasExtremeValue (f m) x) →
  (∃ (x : ℝ), x = 0 ∧ HasExtremeValue (f m) x) →
  m = 1 ∧
  ∀ x ≥ -1/2, g x ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_and_maximum_l105_10569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_properties_l105_10538

-- Define the curve C in polar coordinates
noncomputable def C (θ : ℝ) : ℝ × ℝ := 
  (2 * (Real.sin θ + Real.cos θ + 3/2) * Real.cos θ, 
   2 * (Real.sin θ + Real.cos θ + 3/2) * Real.sin θ)

-- Define the parametric form of C
noncomputable def C_param (θ : ℝ) : ℝ × ℝ := 
  (1 + 2 * Real.cos θ, 1 + 2 * Real.sin θ)

-- Define the function to maximize
def f (x y : ℝ) : ℝ := 3 * x + 4 * y

theorem curve_C_properties :
  (∀ θ, C θ = C_param θ) ∧
  (∃ M, M = 17 ∧ ∀ θ, f (C_param θ).1 (C_param θ).2 ≤ M) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_properties_l105_10538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_equals_A_l105_10545

def A : Set ℝ := {x | 2 - (x + 3) / (x + 1) ≥ 0}

def B (a : ℝ) : Set ℝ := {x | (x - a - 1) * (x - 2 * a) < 0}

theorem union_equals_A (a : ℝ) (h : a < 1) :
  A ∪ B a = A ↔ a ∈ Set.Iic (-2) ∪ Set.Icc (1/2) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_equals_A_l105_10545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_of_cones_l105_10576

-- Define the cones
def cone_C : ℝ × ℝ := (20, 10)  -- (height, radius)
def cone_D : ℝ × ℝ := (10, 20)  -- (height, radius)

-- Define the volume of a cone
noncomputable def cone_volume (h r : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

-- Theorem statement
theorem volume_ratio_of_cones :
  let v_C := cone_volume cone_C.1 cone_C.2
  let v_D := cone_volume cone_D.1 cone_D.2
  v_C / v_D = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_of_cones_l105_10576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_for_all_sin_less_than_x_l105_10564

theorem negation_of_for_all_sin_less_than_x :
  (¬ ∀ x : ℝ, Real.sin x < x) ↔ (∃ x : ℝ, Real.sin x ≥ x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_for_all_sin_less_than_x_l105_10564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_10_value_l105_10598

def b : ℕ → ℕ
  | 0 => 4  -- We define b 0 to be 4 to match b 1 in the original problem
  | 1 => 5  -- This matches b 2 in the original problem
  | n + 2 => b (n + 1) * b n

theorem b_10_value : b 9 = 2560000000000000000000000000000000 := by
  -- We use b 9 here because Lean's natural numbers start from 0
  sorry

#eval b 9  -- This will evaluate b 9, which corresponds to b_10 in the original problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_10_value_l105_10598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_theater_ticket_difference_l105_10518

theorem theater_ticket_difference : ∃ (orchestra_tickets balcony_tickets : ℕ),
  let orchestra_price := 12
  let balcony_price := 8
  let total_tickets := 380
  let total_revenue := 3320
  orchestra_tickets + balcony_tickets = total_tickets ∧
  orchestra_price * orchestra_tickets + balcony_price * balcony_tickets = total_revenue ∧
  balcony_tickets - orchestra_tickets = 240
  := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_theater_ticket_difference_l105_10518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_ellipse_max_area_triangle_l105_10570

/-- Helper function to calculate the area of a triangle -/
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ :=
sorry

/-- The trajectory of point M on a line segment AB with given properties forms an ellipse -/
theorem trajectory_is_ellipse (A B M : ℝ × ℝ) :
  (∃ x₀ y₀ : ℝ, A = (x₀, 0) ∧ B = (0, y₀)) →  -- A and B on axes
  ‖A - B‖ = 3 →  -- AB has length 3
  M ∈ Set.Icc A B →  -- M is on AB
  ‖A - M‖ = 2 * ‖M - B‖ →  -- AM = 2MB
  ∃ x y : ℝ, M = (x, y) ∧ x^2 + y^2/4 = 1 :=
sorry

/-- The maximum area of triangle NEF under given conditions -/
theorem max_area_triangle (C : Set (ℝ × ℝ)) (E F N : ℝ × ℝ) (l : Set (ℝ × ℝ)) :
  C = {(x, y) : ℝ × ℝ | x^2 + y^2/4 = 1} →  -- C is the ellipse
  (0, 1) ∈ l →  -- l passes through (0,1)
  E ∈ C ∧ E ∈ l →  -- E is on C and l
  F ∈ C ∧ F ∈ l →  -- F is on C and l
  E ≠ F →  -- E and F are distinct
  N ∈ C →  -- N is on C
  N ≠ E ∧ N ≠ F →  -- N is different from E and F
  (∀ N' : ℝ × ℝ, N' ∈ C ∧ N' ≠ E ∧ N' ≠ F → area_triangle N' E F ≤ area_triangle N E F) →
  area_triangle N E F = 3 * Real.sqrt 3 / 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_ellipse_max_area_triangle_l105_10570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l105_10507

-- Define the function f(x) = x / (x + 1)
noncomputable def f (x : ℝ) : ℝ := x / (x + 1)

-- Define the open interval (0, +∞)
def openIntervalPositive : Set ℝ := { x : ℝ | 0 < x }

-- Theorem statement
theorem f_monotone_increasing : 
  StrictMonoOn f openIntervalPositive := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l105_10507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_diagonal_l105_10549

theorem quadrilateral_diagonal (a d b c o : ℝ) : 
  abs (a - d) = 4 →
  abs (b - c) = 7 →
  abs (o - c) = 6 →
  abs (o - a) = 4 →
  abs (b - d) = 8 →
  abs (a - c) = 2 * Real.sqrt 889 / 7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_diagonal_l105_10549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_munificence_monic_cubic_l105_10594

/-- Definition of munificence for a polynomial -/
noncomputable def munificence (p : ℝ → ℝ) : ℝ :=
  ⨆ (x : ℝ) (h : x ∈ Set.Icc (-1) 1), |p x|

/-- A monic cubic polynomial -/
def monic_cubic (b c d : ℝ) (x : ℝ) : ℝ :=
  x^3 + b*x^2 + c*x + d

/-- The smallest possible munificence of a monic cubic polynomial is 1 -/
theorem smallest_munificence_monic_cubic :
  (∀ b c d : ℝ, munificence (monic_cubic b c d) ≥ 1) ∧
  (∃ b c d : ℝ, munificence (monic_cubic b c d) = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_munificence_monic_cubic_l105_10594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l105_10565

/-- A quadratic function satisfying specific conditions -/
def f (b c x : ℝ) : ℝ := x^2 + b*x + c

/-- The function is non-negative for all sine values -/
axiom h1 (b c : ℝ) : ∀ α : ℝ, f b c (Real.sin α) ≥ 0

/-- The function is non-positive for all values of the form 2 + cos β -/
axiom h2 (b c : ℝ) : ∀ β : ℝ, f b c (2 + Real.cos β) ≤ 0

/-- The maximum value of f(sin α) is 8 -/
axiom h3 (b c : ℝ) : ∃ α : ℝ, f b c (Real.sin α) = 8 ∧ ∀ β : ℝ, f b c (Real.sin β) ≤ 8

theorem f_properties (b c : ℝ) :
  f b c 1 = 0 ∧ c ≥ 3 ∧ ∀ x : ℝ, f b c x = x^2 - 4*x + 3 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l105_10565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reservoir_capacity_l105_10516

theorem reservoir_capacity 
  (initial_percentage : Real) 
  (storm_deposit : Real) 
  (final_percentage : Real) : Real := by
  have h1 : initial_percentage = 55.00000000000001 / 100 := by sorry
  have h2 : storm_deposit = 120000000000 := by sorry
  have h3 : final_percentage = 85 / 100 := by sorry
  have h4 : ∀ capacity : Real,
    final_percentage * capacity - initial_percentage * capacity = storm_deposit →
    capacity = 400000000000 := by sorry
  exact 400000000000


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reservoir_capacity_l105_10516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_max_eq_3sum_min_l105_10595

open BigOperators Finset Nat

theorem sum_max_eq_3sum_min (n : ℕ) (h : n ≥ 3) :
  let M := Finset.range n
  let subsets := Finset.powerset M
  let three_elem_subsets := subsets.filter (λ s ↦ s.card = 3)
  ∑ s in three_elem_subsets, s.max' (by sorry) = 3 * ∑ s in three_elem_subsets, s.min' (by sorry) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_max_eq_3sum_min_l105_10595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l105_10515

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x * Real.exp x + 2

-- Define the derivative of f
noncomputable def f_deriv (x : ℝ) : ℝ := Real.exp x + x * Real.exp x

-- Theorem statement
theorem tangent_line_equation :
  let x₀ : ℝ := 0
  let y₀ : ℝ := f x₀
  let m : ℝ := f_deriv x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (x - y + 2 = 0) :=
by
  -- Introduce the variables
  intro x y
  -- Unfold the definitions
  simp [f, f_deriv]
  -- Simplify the expressions
  norm_num
  -- The proof is omitted
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l105_10515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_for_composite_polynomial_l105_10501

def is_composite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n

theorem smallest_a_for_composite_polynomial :
  ∃ (a : ℕ), a > 0 ∧ (∀ x : ℤ, is_composite (Int.natAbs (x^4 + a^2 + 16))) ∧
  (∀ b : ℕ, b > 0 ∧ b < a → ∃ x : ℤ, ¬is_composite (Int.natAbs (x^4 + b^2 + 16))) ∧
  a = 8 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_for_composite_polynomial_l105_10501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_expected_triangle_area_on_unit_circle_l105_10579

/-- 
The expected area of a triangle formed by three points chosen uniformly 
at random on the unit circle.
-/
noncomputable def expectedTriangleArea : ℝ := 3 / (2 * Real.pi)

/-- 
Theorem stating that the expected area of a triangle formed by three points 
chosen uniformly at random on the unit circle is 3/(2π).
-/
theorem expected_triangle_area_on_unit_circle : 
  expectedTriangleArea = 3 / (2 * Real.pi) := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_expected_triangle_area_on_unit_circle_l105_10579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l105_10587

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt ((x - 1) / (x - 2))

-- State the theorem
theorem f_domain : 
  ∀ x : ℝ, (x > 2 ∨ x ≤ 1) ↔ (∃ y : ℝ, f x = y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l105_10587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_product_of_b_values_l105_10548

theorem square_product_of_b_values (b₁ b₂ : ℝ) : 
  (∃ b : ℝ, (Set.Icc (-1) 4).prod (Set.Icc 2 b) = (Set.Icc 0 5).prod (Set.Icc 0 5)) →
  b₁ * b₂ = -21 ∧ 
  (∃ b : ℝ, (Set.Icc (-1) 4).prod (Set.Icc b 2) = (Set.Icc 0 5).prod (Set.Icc 0 5)) →
  b₁ * b₂ = -21 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_product_of_b_values_l105_10548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_match_overs_l105_10568

/-- Represents a cricket match with given conditions -/
structure CricketMatch where
  max_runs : Nat
  balls_per_over : Nat
  max_runs_per_ball : Nat

/-- Calculates the number of overs in a cricket match -/
def calculate_overs (m : CricketMatch) : Nat :=
  sorry

/-- Theorem stating that for a match with 663 max runs, 6 balls per over, 
    and 6 max runs per ball, the number of overs is 19 -/
theorem cricket_match_overs : 
  let m : CricketMatch := { max_runs := 663, balls_per_over := 6, max_runs_per_ball := 6 }
  calculate_overs m = 19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_match_overs_l105_10568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_prime_arithmetic_sequence_l105_10571

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- A function that checks if a list of natural numbers forms an arithmetic sequence -/
def isArithmeticSequence (list : List ℕ) (d : ℕ) : Prop :=
  list.length > 1 ∧ ∀ i, i + 1 < list.length → list.get! (i + 1) - list.get! i = d

theorem unique_prime_arithmetic_sequence :
  ∃! list : List ℕ,
    list.length = 5 ∧
    (∀ n, n ∈ list → isPrime n) ∧
    isArithmeticSequence list 6 ∧
    list.sum = 85 := by
  sorry

#eval [5, 11, 17, 23, 29].sum  -- To verify the sum is indeed 85

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_prime_arithmetic_sequence_l105_10571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_functions_l105_10558

-- Define the functions
def A : ℝ → ℝ := λ x => -(x^2) + 3

noncomputable def B : ℝ → ℝ := λ x => 
  if x ≤ 3 then -2 
  else if x > 3 ∧ x ≤ 4 then x - 3 
  else 3

def C : ℝ → ℝ := λ x => x

noncomputable def D : ℝ → ℝ := λ x => 
  if x^2 ≤ 9 then Real.sqrt (9 - x^2) else 0

def E : ℝ → ℝ := λ x => x^3 + x

-- Theorem statement
theorem inverse_functions : 
  (∃ f : ℝ → ℝ, Function.LeftInverse f C ∧ Function.RightInverse f C) ∧ 
  (∃ f : ℝ → ℝ, Function.LeftInverse f E ∧ Function.RightInverse f E) ∧ 
  (¬∃ f : ℝ → ℝ, Function.LeftInverse f A ∧ Function.RightInverse f A) ∧ 
  (¬∃ f : ℝ → ℝ, Function.LeftInverse f B ∧ Function.RightInverse f B) ∧ 
  (¬∃ f : ℝ → ℝ, Function.LeftInverse f D ∧ Function.RightInverse f D) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_functions_l105_10558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vector_combination_l105_10550

/-- Given two vectors in a plane with specific magnitudes and angle between them,
    prove that the angle between their linear combination and one of the vectors is π/6 -/
theorem angle_between_vector_combination (a b : ℝ × ℝ) : 
  let angle_between (v w : ℝ × ℝ) := Real.arccos ((v.1 * w.1 + v.2 * w.2) / (((v.1^2 + v.2^2) * (w.1^2 + w.2^2))^(1/2)))
  (angle_between a b = π/3) → 
  ((a.1^2 + a.2^2)^(1/2) = 1) →
  ((b.1^2 + b.2^2)^(1/2) = 1/2) →
  (angle_between (a.1 + 2*b.1, a.2 + 2*b.2) b = π/6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vector_combination_l105_10550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_AC_length_l105_10554

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the angles of the triangle
noncomputable def angle_A (t : Triangle) : ℝ := 30 * Real.pi / 180
noncomputable def angle_C (t : Triangle) : ℝ := 120 * Real.pi / 180

-- Define the length of side AB
noncomputable def length_AB (t : Triangle) : ℝ := 6 * Real.sqrt 3

-- State the theorem
theorem AC_length (t : Triangle) : 
  let AC := Real.sqrt ((t.A.1 - t.C.1)^2 + (t.A.2 - t.C.2)^2)
  AC = 6 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_AC_length_l105_10554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l105_10599

theorem problem_solution : 
  ((Real.sqrt 27 - Real.sqrt 12) * Real.sqrt (1/3) = 1) ∧
  ((Real.sqrt 2023 + 1) * (Real.sqrt 2023 - 1) + Real.sqrt 8 / Real.sqrt 2 = 2024) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l105_10599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_positive_l105_10581

noncomputable def expr1 (a b m : ℝ) : ℝ := (b + m) / (a + m) - b / a

noncomputable def expr2 (n : ℕ+) : ℝ := Real.sqrt (n + 3) + Real.sqrt n - (Real.sqrt (n + 2) + Real.sqrt (n + 1))

def expr3 (a b : ℝ) : ℝ := 2 * (a^2 + b^2) - (a + b)^2

noncomputable def expr4 (x : ℝ) : ℝ := (x^2 + 3) / Real.sqrt (x^2 + 2) - 2

theorem exactly_two_positive :
  ∃ (a b m : ℝ) (n : ℕ+) (x : ℝ),
    a > b ∧ b > 0 ∧ m > 0 ∧
    (expr1 a b m > 0) ∧
    (expr2 n ≤ 0) ∧
    (expr3 a b ≥ 0) ∧
    (expr4 x > 0) :=
by
  sorry

#check exactly_two_positive

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_positive_l105_10581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_tangent_properties_l105_10577

-- Define the function f
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := b * Real.exp x + x

-- Define the tangent line equation
def tangentLine (a : ℝ) (x y : ℝ) : Prop := a * x - y + 1 = 0

-- Main theorem
theorem curve_and_tangent_properties :
  ∀ (a b : ℝ),
  (∀ x y : ℝ, x = 0 → y = f b x → tangentLine a x y) →
  (a = 2 ∧ b = 1) ∧
  (∀ m : ℝ, (∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → 
    f b x₁ - f b x₂ < (x₁ - x₂) * (m * x₁ + m * x₂ + 1)) → 
    m ≤ Real.exp 1 / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_tangent_properties_l105_10577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivatives_of_functions_l105_10557

noncomputable section

-- Function 1
def f1 (x : ℝ) : ℝ := (1 + Real.cos x) / (1 - Real.cos x)

-- Function 2
def f2 (x : ℝ) : ℝ := Real.sin x - Real.cos x

-- Function 3
def f3 (x : ℝ) : ℝ := x^3 + 3*x^2 - 1

theorem derivatives_of_functions :
  (∀ x, deriv f1 x = (-2 * Real.sin x) / ((1 - Real.cos x)^2)) ∧
  (∀ x, deriv f2 x = Real.cos x + Real.sin x) ∧
  (∀ x, deriv f3 x = 3*x^2 + 6*x) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivatives_of_functions_l105_10557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_alloy_chromium_percentage_l105_10551

-- Define the composition of each alloy
def alloy1 : Float × Float := (0.6, 0.4)  -- (aluminum, chromium)
def alloy2 : Float × Float := (0.1, 0.9)  -- (chromium, titanium)
def alloy3 : Float × Float × Float := (0.2, 0.5, 0.3)  -- (aluminum, chromium, titanium)

-- Define the target titanium percentage in the new alloy
def target_titanium : Float := 0.45

-- Define the possible chromium percentages in the new alloy
def possible_chromium_percentages : Set Float := {0.25, 0.40}

-- Theorem statement
theorem new_alloy_chromium_percentage :
  ∃ (x y z : Float),
    x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧
    x + y + z = 1 ∧
    x * alloy2.snd + z * alloy3.2.2 = target_titanium →
    (x * alloy1.snd + y * alloy2.fst + z * alloy3.2.1) ∈ possible_chromium_percentages :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_alloy_chromium_percentage_l105_10551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_red_after_transfer_is_17_30_l105_10585

-- Define the contents of bag A and bag B
def bagA : Finset (Fin 5) := Finset.univ
def bagB : Finset (Fin 5) := Finset.univ

-- Define the color distributions in bag A and bag B
def redA : Finset (Fin 5) := {0, 1}
def whiteA : Finset (Fin 5) := {2, 3}
def blackA : Finset (Fin 5) := {4}
def redB : Finset (Fin 5) := {0, 1, 2}
def whiteB : Finset (Fin 5) := {3}
def blackB : Finset (Fin 5) := {4}

-- Define the probability of drawing a red ball from bag B after transfer
def prob_red_after_transfer : ℚ := 17/30

-- Theorem statement
theorem prob_red_after_transfer_is_17_30 :
  prob_red_after_transfer = 17/30 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_red_after_transfer_is_17_30_l105_10585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_point_l105_10500

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

theorem inverse_function_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∃ g : ℝ → ℝ, Function.LeftInverse g (f a) ∧ g 3 = -1) → a = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_point_l105_10500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_equivalence_l105_10517

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x > 0 ∧ (2 : ℝ)^x > (3 : ℝ)^x) ↔ (∀ x : ℝ, x > 0 → (2 : ℝ)^x ≤ (3 : ℝ)^x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_equivalence_l105_10517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_haley_recycling_bags_l105_10539

/-- Given a total number of cans, a percentage to be bagged, and the number of cans per bag,
    calculate the number of full bags used. -/
def calculate_bags (total_cans : ℕ) (percentage_bagged : ℚ) (cans_per_bag : ℕ) : ℕ :=
  ((total_cans : ℚ) * percentage_bagged / cans_per_bag).floor.toNat

/-- Prove that given 560 total cans, with 80% to be bagged and 35 cans per bag,
    the number of full bags used is 12. -/
theorem haley_recycling_bags : calculate_bags 560 (4/5) 35 = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_haley_recycling_bags_l105_10539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_pi_12_l105_10572

theorem sin_cos_pi_12 : 
  Real.sin (π / 12) - Real.cos (π / 12) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_pi_12_l105_10572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_intersection_ratio_l105_10588

open Real

theorem sine_intersection_ratio : 
  ∃ (p q : ℕ), 
    (p < q) ∧ 
    (Nat.Coprime p q) ∧
    (∀ x : ℝ, sin x = sin (π/3) → 
      (x ∈ Set.Icc 0 (2*π) → 
        (min (x - 0) ((2*π) - x)) / (max (x - 0) ((2*π) - x)) = p / q)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_intersection_ratio_l105_10588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_time_l105_10530

/-- Calculates the total time for a round trip given jogging and walking speeds and distance -/
noncomputable def total_time (jog_speed walk_speed distance : ℝ) : ℝ :=
  distance / jog_speed + distance / walk_speed

/-- Theorem: The total time for a 3-mile round trip with jogging speed 2 mph and walking speed 4 mph is 2.25 hours -/
theorem round_trip_time : total_time 2 4 3 = 2.25 := by
  -- Unfold the definition of total_time
  unfold total_time
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_time_l105_10530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_period_l105_10597

/-- Given a cosine function with positive constants a, b, c, and d,
    if it completes 5 periods in the interval from -2π to 3π, then b = 2 -/
theorem cosine_period (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (∀ x, ∃ y, y = a * Real.cos (b * x + c) + d) →
  (3 * Real.pi - (-2 * Real.pi) = 5 * (2 * Real.pi / b)) →
  b = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_period_l105_10597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rainfall_on_wednesday_l105_10528

theorem rainfall_on_wednesday 
  (monday_rain : ℚ) 
  (tuesday_rain : ℚ) 
  (total_rain : ℚ) 
  (h1 : monday_rain = 1/6)
  (h2 : tuesday_rain = 5/12)
  (h3 : total_rain = 2/3) :
  total_rain - (monday_rain + tuesday_rain) = 1/12 := by
  sorry

#eval (2/3 : ℚ) - ((1/6 : ℚ) + (5/12 : ℚ))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rainfall_on_wednesday_l105_10528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rocket_average_speed_l105_10540

theorem rocket_average_speed 
  (soar_time : ℝ) (soar_speed : ℝ) (plummet_distance : ℝ) (plummet_time : ℝ)
  (h1 : soar_time = 12)
  (h2 : soar_speed = 150)
  (h3 : plummet_distance = 600)
  (h4 : plummet_time = 3) :
  (soar_speed * soar_time + plummet_distance) / (soar_time + plummet_time) = 160 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rocket_average_speed_l105_10540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_statement_l105_10522

-- Define the basic geometric entities
structure Line
structure Plane

-- Define geometric relationships
def parallel (l1 l2 : Line) : Prop := sorry
def skew (l1 l2 : Line) : Prop := sorry
def have_common_point (l1 l2 : Line) : Prop := sorry

-- Theorem to prove
theorem incorrect_statement :
  ¬(∀ l1 l2 : Line, ¬(have_common_point l1 l2) → parallel l1 l2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_statement_l105_10522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_increasing_and_bounded_x_squared_floor_l105_10547

noncomputable def x : ℕ → ℝ
  | 0 => 1  -- Adding a case for 0 to cover all natural numbers
  | 1 => 1
  | n + 1 => x n / n + n / x n

theorem x_increasing_and_bounded (n : ℕ) (hn : n ≥ 1) :
  x n ≤ x (n + 1) ∧ Real.sqrt n ≤ x n ∧ x n ≤ Real.sqrt (n + 1) := by
  sorry

theorem x_squared_floor (n : ℕ) (hn : n ≥ 1) :
  ⌊(x n)^2⌋ = n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_increasing_and_bounded_x_squared_floor_l105_10547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_center_distance_l105_10531

/-- The distance traveled by the center of a ball rolling along a semicircular racetrack -/
noncomputable def centerDistance (ballDiameter : ℝ) (trackRadii : List ℝ) : ℝ :=
  let ballRadius := ballDiameter / 2
  let adjustedRadii := trackRadii.map (λ r => if r > ballRadius then r - ballRadius else r + ballRadius)
  (adjustedRadii.sum) * Real.pi

/-- Theorem stating the distance traveled by the center of the ball -/
theorem ball_center_distance :
  let ballDiameter : ℝ := 5
  let trackRadii : List ℝ := [120, 70, 90, 70, 120]
  centerDistance ballDiameter trackRadii = 467.5 * Real.pi := by
  sorry

#check ball_center_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_center_distance_l105_10531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_fraction_not_equal_to_red_ratio_l105_10555

theorem exists_fraction_not_equal_to_red_ratio (p : ℕ) (h_p : Nat.Prime p) (h_p_odd : p % 2 = 1) :
  let N := (p^3 - p) / 4 - 1
  ∃ a : ℕ, 0 < a ∧ a < p ∧
    ∀ (coloring : Fin N → Bool) (n : ℕ),
      n ≤ N →
        (Finset.filter (fun i => coloring ⟨i, by sorry⟩) (Finset.range n)).card / n ≠ a / p :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_fraction_not_equal_to_red_ratio_l105_10555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_passes_through_point_equation_represents_circle_circle_equation_distance_l105_10561

-- Define the center of the circle
def center : ℝ × ℝ := (8, -3)

-- Define the point that the circle passes through
def point : ℝ × ℝ := (5, 1)

-- Define the equation of the circle
def circle_equation (x y : ℝ) : Prop :=
  (x - center.1)^2 + (y - center.2)^2 = 25

-- Theorem statement
theorem circle_passes_through_point :
  circle_equation point.1 point.2 := by sorry

-- Theorem to prove that the equation represents a circle with the given center
theorem equation_represents_circle (x y : ℝ) :
  circle_equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = 25 := by sorry

-- Helper theorem to connect the circle equation with the distance formula
theorem circle_equation_distance (x y : ℝ) :
  circle_equation x y ↔ ((x - center.1)^2 + (y - center.2)^2 : ℝ) = 5^2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_passes_through_point_equation_represents_circle_circle_equation_distance_l105_10561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_enclosing_sphere_radius_l105_10592

/-- The radius of a corner sphere -/
def corner_sphere_radius : ℝ := 2

/-- The side length of the cube formed by the centers of the corner spheres -/
def cube_side_length : ℝ := 2 * corner_sphere_radius

/-- The space diagonal of the cube formed by the centers of the corner spheres -/
noncomputable def cube_space_diagonal : ℝ := cube_side_length * Real.sqrt 3

/-- The radius of the smallest enclosing sphere -/
noncomputable def enclosing_sphere_radius : ℝ := (cube_space_diagonal / 2) + corner_sphere_radius

theorem smallest_enclosing_sphere_radius :
  enclosing_sphere_radius = 2 * Real.sqrt 3 + 2 := by
  -- Proof steps would go here
  sorry

#eval corner_sphere_radius
#eval cube_side_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_enclosing_sphere_radius_l105_10592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tanner_september_savings_l105_10536

/-- Represents the amount of money Tanner saved in September -/
def september_savings : ℤ := sorry

/-- Represents the amount of money Tanner saved in October -/
def october_savings : ℤ := 48

/-- Represents the amount of money Tanner saved in November -/
def november_savings : ℤ := 25

/-- Represents the amount of money Tanner spent on a video game -/
def video_game_cost : ℤ := 49

/-- Represents the amount of money Tanner is left with -/
def remaining_money : ℤ := 41

/-- Theorem stating that Tanner saved $17 in September -/
theorem tanner_september_savings : 
  september_savings + october_savings + november_savings - video_game_cost = remaining_money →
  september_savings = 17 := by
  intro h
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tanner_september_savings_l105_10536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_lateral_faces_regular_pyramid_l105_10573

/-- The dihedral angle between adjacent lateral faces of a regular n-sided pyramid -/
noncomputable def dihedralAngleLateralFaces (n : ℕ) (β : ℝ) : ℝ :=
  2 * Real.arccos (Real.sin β * Real.sin (Real.pi / n))

/-- Theorem: The dihedral angle between adjacent lateral faces of a regular n-sided pyramid -/
theorem dihedral_angle_lateral_faces_regular_pyramid
  (n : ℕ) (β : ℝ) (h_n : n ≥ 3) (h_β : 0 < β ∧ β < Real.pi / 2) :
  dihedralAngleLateralFaces n β =
    2 * Real.arccos (Real.sin β * Real.sin (Real.pi / n)) := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_lateral_faces_regular_pyramid_l105_10573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interval_scaling_theorem_l105_10510

/-- Given an interval [a,b] of real numbers that does not contain integers,
    there exists a positive real number N such that [Na,Nb] does not contain integers
    and has a length greater than 1/6. -/
theorem interval_scaling_theorem (a b : ℝ) 
    (h_interval : a < b)
    (h_no_int : ∀ n : ℤ, (n : ℝ) ∉ Set.Icc a b) :
    ∃ N : ℝ, N > 0 ∧ 
    (∀ n : ℤ, (n : ℝ) ∉ Set.Icc (N * a) (N * b)) ∧
    (N * b - N * a > 1/6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interval_scaling_theorem_l105_10510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carbon_mass_percentage_in_CCl4_l105_10527

-- Define the atomic masses and molecule composition
noncomputable def atomic_mass_C : ℝ := 12.01
noncomputable def atomic_mass_Cl : ℝ := 35.45
def carbon_atoms_in_CCl4 : ℕ := 1
def chlorine_atoms_in_CCl4 : ℕ := 4

-- Define the molar mass of CCl4
noncomputable def molar_mass_CCl4 : ℝ :=
  atomic_mass_C * (carbon_atoms_in_CCl4 : ℝ) + atomic_mass_Cl * (chlorine_atoms_in_CCl4 : ℝ)

-- Define the mass percentage calculation
noncomputable def mass_percentage (element_mass : ℝ) (total_mass : ℝ) : ℝ :=
  (element_mass / total_mass) * 100

-- Theorem statement
theorem carbon_mass_percentage_in_CCl4 :
  abs (mass_percentage (atomic_mass_C * (carbon_atoms_in_CCl4 : ℝ)) molar_mass_CCl4 - 7.81) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_carbon_mass_percentage_in_CCl4_l105_10527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_negation_l105_10553

theorem triangle_angle_negation :
  ¬(∀ (angle_C angle_A angle_B : ℝ), 
    angle_C = 90 → (angle_A < 90 ∧ angle_B < 90)) ↔
  (∀ (angle_C angle_A angle_B : ℝ), 
    angle_C ≠ 90 → ¬(angle_A < 90 ∧ angle_B < 90)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_negation_l105_10553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_distance_l105_10544

/-- The hyperbola equation x^2 - y^2/2 = 1 -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/2 = 1

/-- A line passing through a point (a, 0) and having slope m -/
def line (x y a m : ℝ) : Prop := y = m * (x - a)

/-- The right focus of the hyperbola -/
noncomputable def right_focus : ℝ := Real.sqrt 6

/-- The distance between two points on a line intersecting the hyperbola -/
noncomputable def intersection_distance (a m : ℝ) : ℝ := sorry

/-- The number of lines through the right focus that intersect the hyperbola with a given distance -/
def num_lines_with_distance (d : ℝ) : ℕ := sorry

theorem hyperbola_intersection_distance :
  ∃ (lambda : ℝ), num_lines_with_distance lambda = 3 ∧ lambda = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_distance_l105_10544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l105_10514

theorem equation_solution :
  ∃ x : ℝ, 3 * (4 : ℝ)^x + (1/3) * (9 : ℝ)^(x+2) = 6 * (4 : ℝ)^(x+1) - (1/2) * (9 : ℝ)^(x+1) ∧ x = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l105_10514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_tournament_theorem_l105_10578

/-- Represents a player in the chess tournament -/
structure Player where
  id : Nat
  deriving Ord, DecidableEq

/-- Represents the outcome of a game -/
inductive GameResult
  | Win
  | Loss
  | Tie
  deriving DecidableEq

/-- Represents a chess tournament -/
structure ChessTournament where
  players : Finset Player
  games : Player → Player → Bool → GameResult
  score : Player → Rat

/-- The theorem statement for the chess tournament problem -/
theorem chess_tournament_theorem (t : ChessTournament) 
  (h1 : t.players.card > 1)
  (h2 : ∀ p1 p2 : Player, p1 ∈ t.players → p2 ∈ t.players → p1 ≠ p2 → 
    (t.games p1 p2 true = GameResult.Win ∨ t.games p1 p2 true = GameResult.Loss ∨ t.games p1 p2 true = GameResult.Tie) ∧
    (t.games p1 p2 false = GameResult.Win ∨ t.games p1 p2 false = GameResult.Loss ∨ t.games p1 p2 false = GameResult.Tie))
  (h3 : ∀ p1 p2 : Player, p1 ∈ t.players → p2 ∈ t.players → 
    t.games p1 p2 true = GameResult.Win ↔ t.games p2 p1 false = GameResult.Loss)
  (h4 : ∀ p : Player, p ∈ t.players → 
    t.score p = (t.players.sum fun q => 
      match t.games p q true with
      | GameResult.Win => 1
      | GameResult.Tie => 1/2
      | GameResult.Loss => 0
    ) + (t.players.sum fun q => 
      match t.games p q false with
      | GameResult.Win => 1
      | GameResult.Tie => 1/2
      | GameResult.Loss => 0
    ))
  (h5 : ∀ p1 p2 : Player, p1 ∈ t.players → p2 ∈ t.players → t.score p1 = t.score p2) :
  (∃ p1 p2 : Player, p1 ∈ t.players ∧ p2 ∈ t.players ∧ p1 ≠ p2 ∧
    (t.players.filter (fun q => t.games p1 q true = GameResult.Tie ∨ t.games p1 q false = GameResult.Tie)).card =
    (t.players.filter (fun q => t.games p2 q true = GameResult.Tie ∨ t.games p2 q false = GameResult.Tie)).card) ∧
  (∃ p1 p2 : Player, p1 ∈ t.players ∧ p2 ∈ t.players ∧ p1 ≠ p2 ∧
    (t.players.filter (fun q => t.games p1 q true = GameResult.Loss)).card =
    (t.players.filter (fun q => t.games p2 q true = GameResult.Loss)).card) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_tournament_theorem_l105_10578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_theorem_l105_10580

-- Define the condition that cx^2 ≥ ln(1 + x^2) for all real x
def condition (c : ℝ) : Prop :=
  ∀ x : ℝ, c * x^2 ≥ Real.log (1 + x^2)

-- Define the area between the curves
noncomputable def area (c : ℝ) : ℝ :=
  2 * ∫ x in (0 : ℝ)..1, (c * x^2 - Real.log (1 + x^2))

-- Theorem statement
theorem area_theorem (c : ℝ) (h : condition c) : 
  area c = 4 ↔ c = 3 * Real.log 2 + 3 * Real.pi / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_theorem_l105_10580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curvilinear_triangle_area_theorem_l105_10582

/-- The area of a curvilinear triangle formed by three equal arcs of circles -/
noncomputable def curvilinear_triangle_area (R : ℝ) : ℝ :=
  R^2 * (2 * Real.sqrt 3 - Real.pi) / 2

/-- Theorem: The area of a curvilinear triangle formed by three equal arcs of circles,
    each with radius R and touching each other pairwise, is R^2 * (2√3 - π) / 2 -/
theorem curvilinear_triangle_area_theorem (R : ℝ) (R_pos : R > 0) :
  curvilinear_triangle_area R = R^2 * (2 * Real.sqrt 3 - Real.pi) / 2 := by
  -- Unfold the definition of curvilinear_triangle_area
  unfold curvilinear_triangle_area
  -- The rest of the proof would go here
  sorry

#check curvilinear_triangle_area_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curvilinear_triangle_area_theorem_l105_10582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pollutant_after_10h_time_to_reduce_by_third_l105_10589

-- Define the relationship between pollutant content and time
noncomputable def pollutant_content (P₀ k t : ℝ) : ℝ := P₀ * Real.exp (-k * t)

-- Define the condition that 10% of pollutant was eliminated in first 5h
def elimination_condition (P₀ k : ℝ) : Prop :=
  pollutant_content P₀ k 5 = 0.9 * P₀

-- Theorem for the first question
theorem pollutant_after_10h (P₀ k : ℝ) (h₁ : P₀ > 0) (h₂ : k > 0) 
  (h₃ : elimination_condition P₀ k) :
  pollutant_content P₀ k 10 = 0.81 * P₀ := by sorry

-- Function to calculate the time needed to reduce pollutant by a certain fraction
noncomputable def time_to_reduce (P₀ k fraction : ℝ) : ℝ :=
  -(1 / k) * Real.log (fraction)

-- Theorem for the second question
theorem time_to_reduce_by_third (P₀ k : ℝ) (h₁ : P₀ > 0) (h₂ : k > 0) 
  (h₃ : elimination_condition P₀ k) :
  ⌈time_to_reduce P₀ k (2/3)⌉ = 21 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pollutant_after_10h_time_to_reduce_by_third_l105_10589
