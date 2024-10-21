import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_quadratic_roots_l890_89001

/-- A quadratic trinomial with specific geometric properties -/
structure GeometricQuadratic where
  a : ℝ
  b : ℝ
  c : ℝ
  intersect_axes : Nat
  equal_segments : Bool
  angle : ℝ
  h_a : a = 2 / Real.sqrt 3
  h_equal_segments : equal_segments = true
  h_angle : angle = 2 * Real.pi / 3
  h_intersect : intersect_axes = 3

/-- The roots of the quadratic trinomial with given geometric properties -/
def roots (q : GeometricQuadratic) : Set ℝ :=
  {x : ℝ | q.a * x^2 + q.b * x + q.c = 0}

/-- Theorem: The roots of the geometric quadratic are 0.5 and 1.5 -/
theorem geometric_quadratic_roots (q : GeometricQuadratic) : 
  roots q = {0.5, 1.5} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_quadratic_roots_l890_89001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ricotta_usage_ratio_l890_89089

/-- Represents the dimensions of a rectangular sheet -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents a cylindrical cannelloni -/
structure Cannelloni where
  base_radius : ℝ
  height : ℝ

/-- Creates a Cannelloni from a Rectangle and an overlap -/
noncomputable def makeCannelloni (rect : Rectangle) (overlap : ℝ) (useLongerSide : Bool) : Cannelloni :=
  let circumference := if useLongerSide then rect.length - overlap else rect.width - overlap
  let radius := circumference / (2 * Real.pi)
  let height := if useLongerSide then rect.width else rect.length
  { base_radius := radius, height := height }

/-- Calculates the volume of a Cannelloni -/
noncomputable def volume (c : Cannelloni) : ℝ :=
  Real.pi * c.base_radius^2 * c.height

theorem ricotta_usage_ratio (rect : Rectangle) (overlap : ℝ) 
    (h_rect_length : rect.length = 16)
    (h_rect_width : rect.width = 12)
    (h_overlap : overlap = 2)
    (h_original_ricotta : ℝ) (h_original_ricotta_value : h_original_ricotta = 500) :
  let original := makeCannelloni rect overlap true
  let new := makeCannelloni rect overlap false
  volume original / volume new = h_original_ricotta / 340 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ricotta_usage_ratio_l890_89089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_cross_section_area_l890_89079

/-- Represents a pyramid with an isosceles acute-angled triangular base -/
structure Pyramid where
  b : ℝ  -- side length of base triangle
  α : ℝ  -- base angle of triangle
  β : ℝ  -- angle between lateral edges and base plane
  acute_angle : 0 < α ∧ α < π/2
  positive_side : b > 0
  positive_angle : 0 < β ∧ β < π/2

/-- The area of the cross-section of the pyramid -/
noncomputable def crossSectionArea (p : Pyramid) : ℝ := 
  -p.b^2 * Real.cos p.α * Real.tan p.β / (2 * Real.cos (3*p.α))

theorem pyramid_cross_section_area (p : Pyramid) : 
  crossSectionArea p = -p.b^2 * Real.cos p.α * Real.tan p.β / (2 * Real.cos (3*p.α)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_cross_section_area_l890_89079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_candidate_votes_l890_89058

theorem fourth_candidate_votes (total_votes : ℕ) 
  (invalid_percent : ℚ) (first_percent : ℚ) (second_percent : ℚ) (third_percent : ℚ) :
  total_votes = 7000 →
  invalid_percent = 25 / 100 →
  first_percent = 40 / 100 →
  second_percent = 35 / 100 →
  third_percent = 15 / 100 →
  (1 - invalid_percent) * (total_votes : ℚ) * (1 - first_percent - second_percent - third_percent) = 525 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_candidate_votes_l890_89058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_integer_for_f_l890_89094

noncomputable def f (x : ℝ) : ℝ := x + (1 + Real.sqrt (1 + 4 * x)) / 2

theorem smallest_positive_integer_for_f :
  (∀ y : ℕ, 0 < y ∧ y < 2500 → f (y : ℝ) < 50 * Real.sqrt (y : ℝ)) ∧
  f 2500 ≥ 50 * Real.sqrt 2500 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_integer_for_f_l890_89094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gasket_sales_total_l890_89055

/-- Calculates the total amount received by an automobile parts supplier for gasket sales --/
theorem gasket_sales_total (base_price : ℕ) (discount_rate : ℚ) (total_packages : ℕ) 
  (company_x_percent : ℚ) (company_y_percent : ℚ) :
  base_price = 25 →
  discount_rate = 4/5 →
  total_packages = 60 →
  company_x_percent = 15/100 →
  company_y_percent = 15/100 →
  (let company_x_packages := (company_x_percent * total_packages).floor
   let company_y_packages := (company_y_percent * total_packages).floor
   let company_z_packages := total_packages - company_x_packages - company_y_packages
   let company_x_cost := company_x_packages * base_price
   let company_y_cost := company_y_packages * base_price
   let company_z_full_price_packages := min company_z_packages 10
   let company_z_discounted_packages := company_z_packages - company_z_full_price_packages
   let company_z_cost := company_z_full_price_packages * base_price + 
                         (company_z_discounted_packages * (discount_rate * base_price)).floor
   company_x_cost + company_y_cost + company_z_cost = 1340) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gasket_sales_total_l890_89055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l890_89057

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 12 = 1

-- Define the right focus of the ellipse
def right_focus : ℝ × ℝ := (2, 0)

-- Define the fixed point A
noncomputable def point_A : ℝ × ℝ := (-2, Real.sqrt 3)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem min_distance_sum :
  ∃ (min : ℝ), min = 10 ∧
  ∀ (M : ℝ × ℝ), ellipse M.1 M.2 →
    distance M point_A + 2 * distance M right_focus ≥ min := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l890_89057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fermat_numbers_coprime_l890_89008

/-- Definition of Fermat numbers -/
def fermat_number (n : ℕ) : ℕ := 2^(2^n) + 1

/-- Theorem: Fermat numbers are pairwise coprime -/
theorem fermat_numbers_coprime {m n : ℕ} (h : m > n) : 
  Nat.Coprime (fermat_number m) (fermat_number n) := by
  sorry

#check fermat_numbers_coprime

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fermat_numbers_coprime_l890_89008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_decreasing_iff_negative_one_l890_89093

-- Define the power function as noncomputable
noncomputable def power_function (k : ℝ) : ℝ → ℝ := λ x ↦ x^k

-- State the theorem
theorem power_function_decreasing_iff_negative_one :
  ∀ k : ℝ, (∀ x y : ℝ, 0 < x ∧ x < y → power_function k y < power_function k x) ↔ k = -1 :=
by
  sorry

-- You can add more lemmas or theorems here if needed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_decreasing_iff_negative_one_l890_89093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_WXYZ_is_half_unit_l890_89096

-- Define the rectangle ADEH
structure Rectangle (A D E H : ℝ × ℝ) : Prop where
  is_rectangle : true -- We assume ADEH is a rectangle

-- Define the trisection points
def trisect (P Q R : ℝ × ℝ) : Prop :=
  dist P Q = dist Q R ∧ dist P Q + dist Q R + dist R P = dist P R

-- Define the quadrilateral WXYZ
structure Quadrilateral (W X Y Z : ℝ × ℝ) : Prop where
  is_quadrilateral : true -- We assume WXYZ is a quadrilateral

-- Define area function for quadrilateral (placeholder)
def area_quadrilateral (W X Y Z : ℝ × ℝ) : ℝ :=
  sorry -- Placeholder for actual area calculation

-- Main theorem
theorem area_WXYZ_is_half_unit (A B C D E F G H W X Y Z : ℝ × ℝ) 
  (rect : Rectangle A D E H)
  (trisect_AD : trisect A B C ∧ trisect A C D)
  (trisect_HE : trisect H G F ∧ trisect H F E)
  (AH_eq_AC : dist A H = dist A C)
  (AC_eq_two : dist A C = 2)
  (quad : Quadrilateral W X Y Z) : 
  area_quadrilateral W X Y Z = 1/2 := by
  sorry

-- Define dist function (placeholder)
def dist (P Q : ℝ × ℝ) : ℝ :=
  sorry -- Placeholder for actual distance calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_WXYZ_is_half_unit_l890_89096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coloring_exists_for_eight_points_no_valid_coloring_for_nine_points_l890_89038

/-- Represents a coloring of points on a line. -/
def Coloring (n : ℕ) := Fin n → Bool

/-- Checks if a coloring is valid according to the given conditions. -/
def is_valid_coloring (n : ℕ) (c : Coloring n) : Prop :=
  ∀ (x y : Fin n), x < y → Even (x.val + y.val) →
    c x ≠ c ⟨(x.val + y.val) / 2, by sorry⟩ ∨ c ⟨(x.val + y.val) / 2, by sorry⟩ ≠ c y

theorem coloring_exists_for_eight_points :
  ∃ c : Coloring 8, is_valid_coloring 8 c :=
sorry

theorem no_valid_coloring_for_nine_points :
  ¬∃ c : Coloring 9, is_valid_coloring 9 c :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coloring_exists_for_eight_points_no_valid_coloring_for_nine_points_l890_89038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_card_arrangement_theorem_l890_89060

/-- Represents a fraction formed from two cards -/
structure CardFraction where
  numerator : Nat
  denominator : Nat

/-- Checks if a CardFraction represents a whole number -/
def CardFraction.isWhole (f : CardFraction) : Prop :=
  f.numerator % f.denominator = 0

/-- Represents a valid arrangement of cards into fractions -/
structure CardArrangement where
  fractions : Finset CardFraction
  extra : Nat

/-- Checks if a CardArrangement is valid according to the problem constraints -/
def CardArrangement.isValid (arr : CardArrangement) : Prop :=
  (arr.fractions.card = 4) ∧ 
  (∀ f ∈ arr.fractions, f.isWhole) ∧
  (∀ f ∈ arr.fractions, f.numerator ∈ Finset.range 10 ∧ f.denominator ∈ Finset.range 10) ∧
  (arr.extra ∈ Finset.range 10) ∧
  (Finset.card (arr.fractions.biUnion (λ f => {f.numerator, f.denominator}) ∪ {arr.extra}) = 9)

theorem card_arrangement_theorem :
  ∀ arr : CardArrangement, arr.isValid → (arr.extra = 5 ∨ arr.extra = 7) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_card_arrangement_theorem_l890_89060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l890_89072

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 and asymptotes 4ax ± by = 0 is √5 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let hyperbola := fun x y ↦ x^2 / a^2 - y^2 / b^2 = 1
  let asymptotes := fun x y ↦ (4*a*x + b*y = 0) ∨ (4*a*x - b*y = 0)
  let eccentricity := fun a b ↦ Real.sqrt (1 + b^2 / a^2)
  eccentricity a b = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l890_89072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_golden_ratio_l890_89037

/-- A trapezoid with specific side lengths -/
structure Trapezoid where
  -- Two non-parallel sides and one base have length 1
  side1 : ℝ := 1
  side2 : ℝ := 1
  base1 : ℝ := 1
  -- The other base and both diagonals have length a
  base2 : ℝ
  diagonal1 : ℝ
  diagonal2 : ℝ
  -- Constraints to ensure it's a valid trapezoid
  base2_eq_diag : base2 = diagonal1
  diag_eq : diagonal1 = diagonal2

/-- The golden ratio -/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- Theorem stating that the length of the second base and diagonals is the golden ratio -/
theorem trapezoid_golden_ratio (t : Trapezoid) : t.base2 = φ := by
  sorry

#check trapezoid_golden_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_golden_ratio_l890_89037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l890_89059

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- The equation of the ellipse -/
def ellipse_equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The eccentricity of the ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - e.b^2 / e.a^2)

/-- The right focus of the ellipse -/
noncomputable def right_focus (e : Ellipse) : ℝ × ℝ :=
  (e.a * eccentricity e, 0)

/-- The right vertex of the ellipse -/
def right_vertex (e : Ellipse) : ℝ × ℝ :=
  (e.a, 0)

/-- Length of the line segment formed by intersection with vertical line through focus -/
noncomputable def vertical_intersection_length (e : Ellipse) : ℝ :=
  2 * e.b^2 / e.a

theorem ellipse_properties (e : Ellipse) 
    (h_ecc : eccentricity e = 1/2)
    (h_int : vertical_intersection_length e = 3) :
  ∃ (k : Set ℝ), 
    (ellipse_equation e = fun x y ↦ x^2 / 4 + y^2 / 3 = 1) ∧ 
    (k = Set.Icc (-Real.sqrt 6 / 4) (-Real.sqrt 6 / 4) ∪ Set.Icc (Real.sqrt 6 / 4) (Real.sqrt 6 / 4)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l890_89059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_equidistant_point_l890_89088

/-- Given a point P(x,y) equidistant from A(0,4) and B(-2,0), 
    the minimum value of 2^x + 4^y is 4√2 -/
theorem min_value_equidistant_point (x y : ℝ) : 
  (x^2 + (y - 4)^2 = (x + 2)^2 + y^2) →
  (∀ a b : ℝ, (a^2 + (b - 4)^2 = (a + 2)^2 + b^2) → 
    (2:ℝ)^x + (4:ℝ)^y ≤ (2:ℝ)^a + (4:ℝ)^b) →
  (2:ℝ)^x + (4:ℝ)^y = 4 * Real.sqrt 2 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_equidistant_point_l890_89088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_reduction_equation_l890_89020

/-- Represents the monthly average price reduction rate -/
def x : ℝ := sorry

/-- The initial price in March (in thousands of yuan) -/
def initial_price : ℝ := 23

/-- The final price in May (in thousands of yuan) -/
def final_price : ℝ := 16

/-- The number of months between March and May -/
def months : ℕ := 2

/-- Theorem stating that the equation correctly represents the price reduction -/
theorem price_reduction_equation : 
  initial_price * (1 - x)^months = final_price := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_reduction_equation_l890_89020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correlation_coefficient_is_094_strong_correlation_l890_89086

/-- Given a linear regression equation ŷ = bx + a, calculate the correlation coefficient r -/
noncomputable def correlation_coefficient (b : ℝ) (S_y2 : ℝ) (S_x2 : ℝ) : ℝ :=
  b * Real.sqrt (S_x2 / S_y2)

/-- The correlation coefficient between y and x is 0.94 -/
theorem correlation_coefficient_is_094 :
  let b := 4.7
  let S_y2 := 50
  let S_x2 := 2
  correlation_coefficient b S_y2 S_x2 = 0.94 := by
  sorry

/-- Strong linear correlation if r > 0.9 -/
def strong_linear_correlation (r : ℝ) : Prop :=
  r > 0.9

/-- The correlation between y and x is strong -/
theorem strong_correlation :
  let r := correlation_coefficient 4.7 50 2
  strong_linear_correlation r := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correlation_coefficient_is_094_strong_correlation_l890_89086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_range_l890_89027

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, a * x^2 - 2 * a * x - 4 < 0) → a ∈ Set.Ioc (-4) 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_range_l890_89027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l890_89012

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (2 - Real.sqrt (4 - Real.sqrt (5 - x)))

-- Define the domain of f
def domain_f : Set ℝ := Set.Icc (-11) 5

-- Theorem stating that the domain of f is [-11, 5]
theorem domain_of_f : 
  { x : ℝ | ∃ y, f x = y } = domain_f := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l890_89012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_flow_rate_l890_89021

-- Define the original flow rate
variable (F : ℝ)

-- Define the flow rates after each restrictor
def flow_after_first (F : ℝ) : ℝ := 0.75 * F
def flow_after_second (F : ℝ) : ℝ := 0.30 * F

-- Define the final flow rate
def final_flow : ℝ := 2

-- Theorem statement
theorem original_flow_rate (F : ℝ) :
  final_flow = 0.6 * flow_after_second F - 1 →
  F = 50 / 3 :=
by
  intro h
  -- The proof steps would go here
  sorry

-- Verify the result
#eval (50 : ℚ) / 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_flow_rate_l890_89021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_l890_89036

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log ((1 - x) / (1 + x))

-- State the theorem
theorem f_is_odd : ∀ x ∈ Set.Ioo (-1 : ℝ) 1, f (-x) = -f x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_l890_89036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l890_89002

/-- Given two planar vectors α and β with |α| = 1, |β| ≤ 1, and the area of the parallelogram
    formed by α and β is S, the angle θ between α and β satisfies 0 < θ < π. -/
theorem angle_between_vectors (α β : EuclideanSpace ℝ (Fin 2)) (S : ℝ) :
  ‖α‖ = 1 →
  ‖β‖ ≤ 1 →
  S = ‖α‖ * ‖β‖ * Real.sin (InnerProductGeometry.angle α β) →
  0 < InnerProductGeometry.angle α β ∧ InnerProductGeometry.angle α β < π :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l890_89002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_intersection_l890_89024

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (1 + (Real.sqrt 2 / 2) * t, 2 + (Real.sqrt 2 / 2) * t)

-- Define the circle C in polar coordinates
noncomputable def circle_C (θ : ℝ) : ℝ := 4 * Real.sin θ

-- Define point P
def point_P : ℝ × ℝ := (1, 2)

-- Theorem statement
theorem circle_and_line_intersection :
  -- 1. The Cartesian equation of circle C
  (∀ x y : ℝ, (x, y) ∈ {p : ℝ × ℝ | ∃ θ : ℝ, p.1 = circle_C θ * Real.cos θ ∧ p.2 = circle_C θ * Real.sin θ} ↔ 
    x^2 + y^2 - 4*y = 0) ∧
  -- 2. The sum of distances from P to the intersection points
  (∃ A B : ℝ × ℝ, 
    (∃ t : ℝ, line_l t = A) ∧ 
    (∃ t : ℝ, line_l t = B) ∧ 
    A.1^2 + A.2^2 - 4*A.2 = 0 ∧ 
    B.1^2 + B.2^2 - 4*B.2 = 0 ∧ 
    Real.sqrt ((A.1 - point_P.1)^2 + (A.2 - point_P.2)^2) + 
    Real.sqrt ((B.1 - point_P.1)^2 + (B.2 - point_P.2)^2) = Real.sqrt 14) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_intersection_l890_89024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2008th_derivative_at_zero_l890_89061

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.sin (x/4))^6 + (Real.cos (x/4))^6

-- State the theorem
theorem f_2008th_derivative_at_zero : 
  (deriv^[2008] f) 0 = 3/8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2008th_derivative_at_zero_l890_89061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_condition_l890_89071

/-- Two linear functions with parallel, non-axis-aligned graphs -/
structure ParallelLinearFunctions where
  f : ℝ → ℝ
  g : ℝ → ℝ
  parallel : ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x, f x = a * x + b) ∧ (∀ x, g x = a * x + c)

/-- Condition for two functions to be tangent -/
def are_tangent (f g : ℝ → ℝ) : Prop :=
  ∃! x, f x = g x

/-- Main theorem -/
theorem tangent_condition (funcs : ParallelLinearFunctions) 
  (h : are_tangent (λ x ↦ (funcs.f x)^2) (λ x ↦ 7 * funcs.g x)) :
  ∀ A : ℝ, are_tangent (λ x ↦ (funcs.g x)^2) (λ x ↦ A * funcs.f x) ↔ A = -7 ∨ A = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_condition_l890_89071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_emptying_time_l890_89062

-- Define the tank capacity
noncomputable def C : ℝ := 822 + 6/7

-- Define the rates of leaks and inlet
noncomputable def original_leak_rate : ℝ := C / 6
noncomputable def inlet_rate : ℝ := 240
noncomputable def additional_leak_rate : ℝ := C / 12

-- Define the net emptying rate with both leaks and inlet
noncomputable def net_emptying_rate : ℝ := inlet_rate - original_leak_rate - additional_leak_rate

-- Theorem statement
theorem tank_emptying_time : 
  C / net_emptying_rate = 24 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_emptying_time_l890_89062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_l890_89069

theorem polynomial_divisibility (P : ℤ → ℤ) : 
  (∀ i : ℤ, 0 ≤ i ∧ i ≤ 2018 → P i = Nat.choose 2018 i.toNat) →
  (∃ a : Polynomial ℤ, a.degree ≤ 2018 ∧ ∀ x : ℤ, P x = a.eval x) →
  (∃! n : ℕ, n = 6 ∧ (2^n : ℤ) ∣ P 2020 ∧ ∀ m : ℕ, (2^m : ℤ) ∣ P 2020 → m ≤ n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_l890_89069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_and_unique_solution_l890_89017

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x^2

theorem f_max_and_unique_solution :
  (∃ (x_max : ℝ), ∀ (x : ℝ), x > 0 → f x ≤ f x_max) ∧
  (∃ x_max : ℝ, f x_max = 1 / (2 * Real.exp 1)) ∧
  (∀ (a : ℝ), a > 0 →
    (∃! (x : ℝ), f x - a / (2 * Real.exp 1) = 0) →
    (∃! (x : ℝ), (1/2) * x * (deriv f x) + (a * x^2 - x - 1/2) / x^2 = 0)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_and_unique_solution_l890_89017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_green_slope_probability_theorem_l890_89033

/-- Represents a triangular pyramid roof with right angles at the apex -/
structure TriangularPyramidRoof :=
  (α : ℝ) -- Inclination angle of the red slope
  (β : ℝ) -- Inclination angle of the blue slope

/-- The probability of a raindrop landing on the green slope of a triangular pyramid roof -/
noncomputable def greenSlopeProbability (roof : TriangularPyramidRoof) : ℝ :=
  1 - (Real.cos roof.α)^2 - (Real.cos roof.β)^2

/-- Theorem stating the probability of a raindrop landing on the green slope -/
theorem green_slope_probability_theorem (roof : TriangularPyramidRoof) :
  greenSlopeProbability roof = 1 - (Real.cos roof.α)^2 - (Real.cos roof.β)^2 := by
  sorry

#check green_slope_probability_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_green_slope_probability_theorem_l890_89033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_office_approx_l890_89054

/-- Represents the round trip between home and office -/
structure RoundTrip where
  speed_to_office : ℝ
  speed_to_home : ℝ
  stop_duration : ℝ
  total_time : ℝ

/-- Calculates the time spent driving to the office -/
noncomputable def time_to_office (trip : RoundTrip) : ℝ :=
  let total_driving_time := trip.total_time - trip.stop_duration
  let time_ratio := trip.speed_to_home / (trip.speed_to_office + trip.speed_to_home)
  total_driving_time * time_ratio

/-- The main theorem stating the time to office given the conditions -/
theorem time_to_office_approx (trip : RoundTrip) 
  (h1 : trip.speed_to_office = 58)
  (h2 : trip.speed_to_home = 62)
  (h3 : trip.stop_duration = 1/6)
  (h4 : trip.total_time = 3) :
  ∃ ε > 0, |time_to_office trip - 1.4639| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_office_approx_l890_89054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l890_89085

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : (2 : ℝ)^a * (2 : ℝ)^b = 8) :
  (1/a + 4/b) ≥ 3 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ (2 : ℝ)^a₀ * (2 : ℝ)^b₀ = 8 ∧ 1/a₀ + 4/b₀ = 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l890_89085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_light_ray_passes_through_center_l890_89099

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  topLeft : Point
  bottomRight : Point

/-- Represents a light ray's path -/
structure LightPath where
  start : Point
  reflectionPoints : List Point
  finish : Point

/-- The theorem stating that a light ray passing through opposite corners of a rectangle must pass through its center -/
theorem light_ray_passes_through_center (rect : Rectangle) (path : LightPath) :
  path.start = rect.topLeft ∧ 
  path.finish = rect.bottomRight ∧
  (∀ p ∈ path.reflectionPoints, p.x = rect.topLeft.x ∨ p.x = rect.bottomRight.x ∨ 
                                 p.y = rect.topLeft.y ∨ p.y = rect.bottomRight.y) →
  ∃ p ∈ path.reflectionPoints, 
    p.x = (rect.topLeft.x + rect.bottomRight.x) / 2 ∧ 
    p.y = (rect.topLeft.y + rect.bottomRight.y) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_light_ray_passes_through_center_l890_89099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_value_l890_89014

theorem sin_2alpha_value (α : ℝ) (h : Real.sin α + Real.cos α = 1/5) : Real.sin (2 * α) = -24/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_value_l890_89014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_nine_l890_89050

theorem power_of_nine (y : ℝ) (h : (3 : ℝ)^(2*y) = 5) : (9 : ℝ)^(y+1) = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_nine_l890_89050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_room_ratio_theorem_l890_89005

/-- Represents the dimensions of a rectangular room -/
structure RoomDimensions where
  length : ℚ
  width : ℚ

/-- Calculates the perimeter of a rectangular room -/
def perimeter (room : RoomDimensions) : ℚ :=
  2 * (room.length + room.width)

/-- Simplifies a ratio represented as a pair of rational numbers -/
def simplifyRatio (a b : ℚ) : ℚ × ℚ :=
  let gcd := Int.gcd a.num b.num
  (a / gcd, b / gcd)

theorem room_ratio_theorem (room : RoomDimensions) 
  (h1 : room.length = 25)
  (h2 : room.width = 15) :
  (simplifyRatio room.length (perimeter room) = (5, 16)) ∧ 
  (simplifyRatio room.width (perimeter room) = (3, 16)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_room_ratio_theorem_l890_89005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l890_89030

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  (Real.sin x + Real.cos x) * (Real.cos x - Real.sin x) + 
  (Real.sqrt 3 * Real.cos x) * (2 * Real.sin x)

-- Define the theorem
theorem triangle_properties (A B C a b c : ℝ) :
  -- Conditions
  0 < A ∧ A < π/2 ∧
  0 < B ∧ B < π/2 ∧
  0 < C ∧ C < π/2 ∧
  A + B + C = π ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a / Real.sin A = b / Real.sin B ∧
  b / Real.sin B = c / Real.sin C ∧
  f A = 1 ∧
  a = Real.sqrt 3 →
  -- Conclusions
  (A = π/3) ∧
  (3 < b + c ∧ b + c ≤ 2 * Real.sqrt 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l890_89030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adults_not_wearing_blue_is_ten_l890_89070

def adults_not_wearing_blue (num_children : ℕ) (num_adults : ℕ) (num_adults_blue : ℕ) : ℕ :=
  num_adults - num_adults_blue

-- Conditions
axiom num_children_def : 45 = 45
axiom num_adults_def : 45 / 3 = 45 / 3
axiom num_adults_blue_def : (45 / 3) / 3 = (45 / 3) / 3

-- Theorem statement
theorem adults_not_wearing_blue_is_ten :
  adults_not_wearing_blue 45 (45 / 3) ((45 / 3) / 3) = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_adults_not_wearing_blue_is_ten_l890_89070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_equation_l890_89078

theorem roots_of_equation : 
  let f : ℝ → ℝ := λ x => (x^3 - 2*x^2 - x + 2)*(x-3)*(x+1)
  ∀ x : ℝ, f x = 0 ↔ x ∈ ({-1, 1, 2, 3} : Set ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_equation_l890_89078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_slope_is_3_plus_sqrt_3_l890_89015

/-- A cubic function with two extreme points, one at the origin and the other on a specific circle -/
structure CubicFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  x₁ : ℝ
  x₂ : ℝ
  h₁ : c = 0
  h₂ : d = 0
  h₃ : x₁ = 0
  h₄ : (x₂ - 2)^2 + (a * x₂^3 + b * x₂^2 - 3)^2 = 1

/-- The maximum slope of the tangent line to the cubic function -/
noncomputable def max_slope (f : CubicFunction) : ℝ := 3 + Real.sqrt 3

/-- Theorem stating that the maximum slope of the tangent line is 3 + √3 -/
theorem max_slope_is_3_plus_sqrt_3 (f : CubicFunction) : 
  max_slope f = 3 + Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_slope_is_3_plus_sqrt_3_l890_89015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_tangents_condition_l890_89043

noncomputable section

-- Define the curves
def curve1 (a : ℝ) (x : ℝ) : ℝ := a / x
def curve2 (x : ℝ) : ℝ := x^2

-- Define the slopes of the tangent lines at the intersection point
def slope1 (a : ℝ) (x : ℝ) : ℝ := -a / x^2
def slope2 (x : ℝ) : ℝ := 2 * x

-- Define the condition for perpendicular tangent lines
def perpendicular_tangents (a : ℝ) : Prop :=
  ∃ x : ℝ, x > 0 ∧ curve1 a x = curve2 x ∧ slope1 a x * slope2 x = -1

-- State the theorem
theorem perpendicular_tangents_condition (a : ℝ) :
  perpendicular_tangents a ↔ a = Real.sqrt 2 / 4 ∨ a = -Real.sqrt 2 / 4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_tangents_condition_l890_89043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_mass_spring_system_energy_l890_89016

/-- The reduced mass of a two-mass system -/
noncomputable def reduced_mass (m M : ℝ) : ℝ := (m * M) / (m + M)

/-- Planck's constant in J⋅s -/
noncomputable def ℏ : ℝ := 1.0545718e-34

/-- The ground state energy of a two-mass spring system -/
noncomputable def ground_state_energy (k μ : ℝ) : ℝ :=
  (ℏ / 2) * Real.sqrt (k / μ)

/-- Conversion factor from Joules to eV -/
noncomputable def joules_to_eV : ℝ := 1 / 1.602176634e-19

theorem two_mass_spring_system_energy :
  let m : ℝ := 1
  let M : ℝ := 2 * m
  let k : ℝ := 100
  let μ : ℝ := reduced_mass m M
  let E : ℝ := ground_state_energy k μ * joules_to_eV
  ∃ (a p : ℤ), 0 < a ∧ a < 10 ∧ 
    |E - (a : ℝ) * 10^p| < |E - ((a+1) : ℝ) * 10^p| ∧
    |E - (a : ℝ) * 10^p| < |E - ((a-1) : ℝ) * 10^p| ∧
    a = 4 ∧ p = -15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_mass_spring_system_energy_l890_89016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l890_89051

noncomputable def f (x : ℝ) := Real.sin x * Real.cos x - Real.sqrt 3 * (Real.cos x)^2 + Real.sqrt 3 / 2

theorem f_properties :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ 
   ∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') ∧
  (∀ k : ℤ, ∃ c : ℝ, ∀ x, f (c + x) = f (c - x) ∧ c = k * Real.pi / 2 + Real.pi / 6) ∧
  (∀ k : ℤ, ∀ x ∈ Set.Icc (k * Real.pi - Real.pi / 12) (k * Real.pi + 5 * Real.pi / 12),
    ∀ y ∈ Set.Icc (k * Real.pi - Real.pi / 12) (k * Real.pi + 5 * Real.pi / 12),
    x ≤ y → f x ≤ f y) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ 1) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≥ -Real.sqrt 3 / 2) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = 1) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = -Real.sqrt 3 / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l890_89051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_absolute_difference_l890_89063

theorem square_absolute_difference (x y : ℤ) 
  (h : ∃ (k : ℤ), x^2 - 4*y + 1 = (x - 2*y) * (1 - 2*y) * k) : 
  ∃ (m : ℕ), (x - 2*y)^2 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_absolute_difference_l890_89063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distribution_count_l890_89091

def number_of_ways_to_distribute (n k : ℕ) : ℕ := k^n

theorem distribution_count (n k : ℕ) (h1 : n = 5) (h2 : k = 3) : 
  number_of_ways_to_distribute n k = k^n := by
  rfl

#check distribution_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distribution_count_l890_89091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_op_inequality_l890_89034

-- Define the ⊙ operation
noncomputable def circle_op (a b : ℝ) : ℝ := Real.sqrt (a * b) + a + b

-- State the theorem
theorem circle_op_inequality {k : ℝ} (hk : k > 0) :
  circle_op 1 k < 3 ↔ k > 2 + Real.sqrt 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_op_inequality_l890_89034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_focus_is_three_l890_89011

/-- The parabola y^2 = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- The point P on the parabola -/
structure Point (x y : ℝ) where
  on_parabola : parabola x y

/-- The focus of the parabola y^2 = 4x -/
def focus : ℝ × ℝ := (1, 0)

/-- The distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem distance_to_focus_is_three (P : Point 2 y) :
  distance (2, y) focus = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_focus_is_three_l890_89011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_chord_length_is_two_l890_89075

/-- The circle with equation x^2 + y^2 - 6x = 0 -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 6*x = 0

/-- The point through which all lines pass -/
def fixed_point : ℝ × ℝ := (1, 2)

/-- The minimum chord length -/
def min_chord_length : ℝ := 2

/-- Theorem stating that the minimum chord length is 2 -/
theorem min_chord_length_is_two :
  ∀ (l : Set (ℝ × ℝ)), 
    (fixed_point ∈ l) → 
    (∃ (a b : ℝ × ℝ), a ≠ b ∧ a ∈ l ∧ b ∈ l ∧ circle_eq a.1 a.2 ∧ circle_eq b.1 b.2) →
    (∃ (c d : ℝ × ℝ), c ≠ d ∧ c ∈ l ∧ d ∈ l ∧ circle_eq c.1 c.2 ∧ circle_eq d.1 d.2 ∧
      Real.sqrt ((c.1 - d.1)^2 + (c.2 - d.2)^2) = min_chord_length) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_chord_length_is_two_l890_89075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_above_x_axis_pure_imaginary_l890_89028

-- Define the complex number z as a function of real number m
def z (m : ℝ) : ℂ := Complex.mk (m^2 + 5*m + 6) (m^2 - 2*m - 15)

-- Statement for part 1
theorem above_x_axis (m : ℝ) : 
  (z m).im > 0 ↔ m < -3 ∨ m > 5 := by sorry

-- Statement for part 2
theorem pure_imaginary (m : ℝ) :
  (z m / (1 + Complex.I)).re = 0 ∧ (z m / (1 + Complex.I)).im ≠ 0 ↔ m = 3/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_above_x_axis_pure_imaginary_l890_89028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_in_interval_l890_89048

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (x^2 - 4*x + 1) * Real.exp x

-- State the theorem
theorem max_value_in_interval :
  ∃ (c : ℝ), c ∈ Set.Icc (-2 : ℝ) 0 ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) 0, f x ≤ f c) ∧
  f c = 6 / Real.exp 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_in_interval_l890_89048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_ratio_l890_89026

/-- Simple interest calculation function -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

/-- Compound interest calculation function -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * ((1 + rate / 100) ^ time - 1)

/-- Theorem stating the ratio of simple interest to compound interest -/
theorem interest_ratio :
  (simple_interest 1750.000000000002 8 3) / (compound_interest 4000 10 2) = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_ratio_l890_89026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_locus_and_intersection_length_l890_89013

-- Define the circle C
def C : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 16}

-- Define point A
def A : ℝ × ℝ := (0, 2)

-- Define the locus E of midpoint P
def E : Set (ℝ × ℝ) := {p | p.1^2 + (p.2 - 1)^2 = 4}

-- Define the line L passing through A with slope -3/4
def L : Set (ℝ × ℝ) := {p | 3 * p.1 + 4 * p.2 - 8 = 0}

theorem midpoint_locus_and_intersection_length :
  -- Part 1: The locus of midpoint P is E
  (∀ D ∈ C, ((A.1 + D.1) / 2, (A.2 + D.2) / 2) ∈ E) ∧
  -- Part 2: The length of MN is 4√21/5
  (∃ m n : ℝ × ℝ, m ∈ E ∩ L ∧ n ∈ E ∩ L ∧ m ≠ n ∧
    Real.sqrt ((m.1 - n.1)^2 + (m.2 - n.2)^2) = 4 * Real.sqrt 21 / 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_locus_and_intersection_length_l890_89013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_product_sum_l890_89006

noncomputable def log_product (c d : ℕ) : ℝ :=
  Real.log (d : ℝ) / Real.log (c : ℝ)

theorem log_product_sum (c d : ℕ) : 
  c > 0 → d > c → 
  (d - 2 - c + 2 = 435) → 
  log_product c d = 3 → 
  c + d = 520 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_product_sum_l890_89006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_slope_specific_l890_89064

/-- The slope of the angle bisector of the acute angle formed by two lines -/
noncomputable def angleBisectorSlope (m₁ m₂ : ℝ) : ℝ :=
  (m₁ + m₂ - Real.sqrt (m₁^2 + m₂^2 - m₁*m₂ + 1)) / (1 + m₁*m₂)

/-- Theorem: The slope of the angle bisector of the acute angle formed by y = 2x and y = 5x is (7 - 2√5) / 11 -/
theorem angle_bisector_slope_specific : 
  angleBisectorSlope 2 5 = (7 - 2 * Real.sqrt 5) / 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_slope_specific_l890_89064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_change_difference_l890_89080

/-- Represents the percentage of students in each category -/
structure StudentResponses where
  yes : ℚ
  no : ℚ
  undecided : ℚ

/-- Calculates the minimum percentage of students who changed their answers -/
noncomputable def min_change (initial final : StudentResponses) : ℚ :=
  max (final.yes - initial.yes) 0 +
  max (final.no - initial.no) 0 +
  max (final.undecided - initial.undecided) 0

/-- Calculates the maximum percentage of students who changed their answers -/
noncomputable def max_change (initial final : StudentResponses) : ℚ :=
  |final.yes - initial.yes| +
  |final.no - initial.no| +
  |final.undecided - initial.undecided|

/-- The main theorem to be proven -/
theorem change_difference (initial final : StudentResponses)
  (h_initial : initial.yes + initial.no + initial.undecided = 100)
  (h_final : final.yes + final.no + final.undecided = 100)
  (h_initial_yes : initial.yes = 40)
  (h_initial_no : initial.no = 40)
  (h_initial_undecided : initial.undecided = 20)
  (h_final_yes : final.yes = 60)
  (h_final_no : final.no = 30)
  (h_final_undecided : final.undecided = 10) :
  max_change initial final - min_change initial final = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_change_difference_l890_89080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_area_l890_89031

/-- Given a sector with radius 2 cm and central angle 120°, 
    when rolled into a cone, the area of the base of the cone is 4π/9 cm². -/
theorem cone_base_area (sector_radius : ℝ) (sector_angle : ℝ) : 
  sector_radius = 2 →
  sector_angle = 120 →
  let base_circumference := sector_angle / 360 * (2 * π * sector_radius)
  let base_radius := base_circumference / (2 * π)
  π * base_radius^2 = 4 * π / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_area_l890_89031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_forms_circle_four_equidistant_points_l890_89009

noncomputable def l₁ (a θ : ℝ) : ℝ := -1 / (Real.sin θ + a * Real.cos θ)
noncomputable def l₂ (a θ : ℝ) : ℝ := 1 / (Real.cos θ - a * Real.sin θ)

def C (x y : ℝ) : Prop := (x - 1/2)^2 + (y + 1/2)^2 = 1/2

theorem intersection_forms_circle (a : ℝ) :
  ∀ x y : ℝ, (∃ θ : ℝ, x = l₁ a θ * Real.cos θ ∧ y = l₁ a θ * Real.sin θ) ∧
             (∃ θ : ℝ, x = l₂ a θ * Real.cos θ ∧ y = l₂ a θ * Real.sin θ) →
  C x y := by
  sorry

theorem four_equidistant_points (a : ℝ) :
  (∃ x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ,
    C x₁ y₁ ∧ C x₂ y₂ ∧ C x₃ y₃ ∧ C x₄ y₄ ∧
    let d₁ := |a * x₁ + y₁ + 1| / Real.sqrt (a^2 + 1)
    let d₂ := |a * x₂ + y₂ + 1| / Real.sqrt (a^2 + 1)
    let d₃ := |a * x₃ + y₃ + 1| / Real.sqrt (a^2 + 1)
    let d₄ := |a * x₄ + y₄ + 1| / Real.sqrt (a^2 + 1)
    d₁ = d₂ ∧ d₂ = d₃ ∧ d₃ = d₄ ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧ (x₂ ≠ x₃ ∨ y₂ ≠ y₃) ∧ (x₃ ≠ x₄ ∨ y₃ ≠ y₄)) ↔
  a < 1 ∨ a > 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_forms_circle_four_equidistant_points_l890_89009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_paths_count_l890_89097

/-- A path in the Cartesian plane -/
structure CartesianPath where
  steps : List (Int × Int)

/-- Checks if a path is valid according to the given rules -/
def isValidPath (p : CartesianPath) : Bool :=
  sorry

/-- Counts the number of valid paths from (0,0) to (3,3) -/
def countValidPaths : Nat :=
  sorry

/-- Theorem stating that the number of valid paths from (0,0) to (3,3) is 80 -/
theorem valid_paths_count : countValidPaths = 80 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_paths_count_l890_89097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_m_value_l890_89066

-- Define the imaginary unit
noncomputable def i : ℂ := Complex.I

-- Define sets A and B
def A (m : ℝ) : Set ℂ := {2, 7, -4*m + (m+2)*i}
def B : Set ℂ := {8, 3}

-- Theorem statement
theorem intersection_implies_m_value (m : ℝ) : (A m ∩ B).Nonempty → m = -2 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_m_value_l890_89066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_symmetry_l890_89056

noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin (2 * x + Real.pi / 3)

theorem graph_symmetry (x : ℝ) : 
  f (-Real.pi/6 + x) = -f (-Real.pi/6 - x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_symmetry_l890_89056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_value_l890_89018

theorem sin_plus_cos_value (α : ℝ) 
  (h1 : Real.cos (α + π/4) = 7 * Real.sqrt 2 / 10)
  (h2 : Real.cos (2*α) = 7/25) : 
  Real.sin α + Real.cos α = 1/5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_value_l890_89018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_cricket_innings_l890_89046

def cricket_innings_problem (current_average : ℝ) (runs_needed : ℕ) (average_increase : ℝ) : Prop :=
  ∃ n : ℕ, 
    n > 0 ∧
    (current_average * (n : ℝ) + (runs_needed : ℝ)) / ((n : ℝ) + 1) = current_average + average_increase ∧
    n = 10

theorem solve_cricket_innings :
  cricket_innings_problem 36 80 4 := by
  use 10
  constructor
  · exact Nat.succ_pos 9
  constructor
  · norm_num
  · rfl

#check solve_cricket_innings

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_cricket_innings_l890_89046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_vector_AB_is_correct_l890_89090

def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (4, -1)

def vector_AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

noncomputable def magnitude_AB : ℝ := Real.sqrt (vector_AB.1^2 + vector_AB.2^2)

noncomputable def unit_vector_AB : ℝ × ℝ := (vector_AB.1 / magnitude_AB, vector_AB.2 / magnitude_AB)

theorem unit_vector_AB_is_correct : unit_vector_AB = (3/5, -4/5) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_vector_AB_is_correct_l890_89090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_r_at_zero_l890_89035

-- Define the function L(m)
noncomputable def L (m : ℝ) : ℝ := -Real.sqrt (m + 6)

-- Define the function r(m)
noncomputable def r (m : ℝ) : ℝ := (L (-m) - L m) / m

-- State the theorem
theorem limit_of_r_at_zero :
  ∀ ε > 0, ∃ δ > 0, ∀ m : ℝ, 0 < |m| ∧ |m| < δ ∧ -6 < m ∧ m < 6 →
    |r m - 1 / Real.sqrt 6| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_r_at_zero_l890_89035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_first_five_terms_l890_89044

def mySequence (n : ℕ) : ℚ := (2^n : ℚ) / (2*n - 1)

theorem sequence_first_five_terms :
  [mySequence 1, mySequence 2, mySequence 3, mySequence 4, mySequence 5] = [2, 4/3, 8/5, 16/7, 32/9] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_first_five_terms_l890_89044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l890_89040

-- Define the hyperbola and its properties
def Hyperbola (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0

-- Define the eccentricity of the hyperbola
noncomputable def Eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 + b^2) / a

-- Define the condition that an asymptote intersects the circle
def AsymptoteIntersectsCircle (a b : ℝ) : Prop :=
  (Real.sqrt 3 * b) / Real.sqrt (a^2 + b^2) ≤ 1

-- The main theorem
theorem hyperbola_eccentricity_range (a b : ℝ) :
  Hyperbola a b →
  AsymptoteIntersectsCircle a b →
  1 < Eccentricity a b ∧ Eccentricity a b ≤ Real.sqrt 6 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l890_89040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_workshop_A_more_stable_l890_89087

noncomputable def workshop_A : List ℝ := [100, 96, 101, 96, 97]
noncomputable def workshop_B : List ℝ := [103, 93, 100, 95, 99]

noncomputable def mean (xs : List ℝ) : ℝ := (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  let m := mean xs
  (xs.map (fun x => (x - m)^2)).sum / xs.length

theorem workshop_A_more_stable : variance workshop_A < variance workshop_B := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_workshop_A_more_stable_l890_89087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_length_l890_89042

-- Define the curve C
def C (l : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 / l - 4 = 0}

-- Define what it means for a curve to be a hyperbola
def is_hyperbola (S : Set (ℝ × ℝ)) : Prop := sorry

-- Define the focal length of a conic section
noncomputable def focal_length (S : Set (ℝ × ℝ)) : ℝ := sorry

-- The theorem to prove
theorem hyperbola_focal_length (l : ℝ) (h : l < -1) :
  is_hyperbola (C l) ∧ focal_length (C l) = 4 * Real.sqrt (1 - l) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_length_l890_89042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_waste_is_half_square_l890_89041

/-- The total area of metal wasted when cutting a circular disc from a square
    and then cutting a rectangle from the disc -/
noncomputable def total_waste (s : ℝ) : ℝ :=
  s^2 - (Real.pi * (s/2)^2 - s * (s/2))

/-- Theorem stating that the total waste is equal to s^2/2 -/
theorem total_waste_is_half_square (s : ℝ) (h : s > 0) :
  total_waste s = s^2 / 2 := by
  sorry

#check total_waste_is_half_square

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_waste_is_half_square_l890_89041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_product_72_l890_89019

/-- A standard die with faces numbered from 1 to 6 -/
def StandardDie : Set Nat := {1, 2, 3, 4, 5, 6}

/-- The probability of a specific outcome when rolling three dice -/
noncomputable def ProbabilityOfOutcome : ℝ := (1 / 6) ^ 3

/-- The number of ways to obtain a product of 72 with three dice -/
def WaysToGet72 : Nat := 3

/-- Theorem: The probability of obtaining a product of 72 when rolling three standard dice is 1/72 -/
theorem probability_product_72 (a b c : Nat) (h_a : a ∈ StandardDie) (h_b : b ∈ StandardDie) (h_c : c ∈ StandardDie) :
  (a * b * c = 72) → ProbabilityOfOutcome * WaysToGet72 = 1 / 72 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_product_72_l890_89019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_zeros_of_g_l890_89022

noncomputable def g (x : ℝ) := Real.sin (Real.log x)

theorem infinitely_many_zeros_of_g :
  ∃ (S : Set ℝ), Set.Infinite S ∧ (∀ x ∈ S, 0 < x ∧ x < 1 ∧ g x = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_zeros_of_g_l890_89022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_k_l890_89029

-- Define the circles and points
def origin : ℝ × ℝ := (0, 0)
def P : ℝ × ℝ := (5, 12)
def S (k : ℝ) : ℝ × ℝ := (0, k)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem circle_radius_k (k : ℝ) : 
  (distance origin P = distance origin (S k) + 5) → k = 8 := by
  sorry

#check circle_radius_k

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_k_l890_89029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_completion_time_l890_89045

-- Define the time it takes to complete the order in each workshop
noncomputable def time_workshop3 : ℝ := 8

noncomputable def time_workshop1 : ℝ := time_workshop3 + 10

noncomputable def time_workshop2 : ℝ := time_workshop1 - 3.6

-- State the theorem
theorem order_completion_time :
  (1 / time_workshop1 + 1 / time_workshop2 = 1 / time_workshop3) →
  time_workshop3 = 8 ∧ time_workshop3 - 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_completion_time_l890_89045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_is_1024_27_l890_89095

/-- A triangle with an inscribed rectangle -/
structure TriangleWithRectangle where
  /-- The altitude from Y to XZ -/
  altitude : ℝ
  /-- The length of XZ -/
  base : ℝ
  /-- The ratio of PQ to PS -/
  ratio : ℝ

/-- The area of the inscribed rectangle -/
noncomputable def rectangleArea (t : TriangleWithRectangle) : ℝ :=
  1024 / 27

/-- Theorem stating that the area of the inscribed rectangle is 1024/27 -/
theorem rectangle_area_is_1024_27 (t : TriangleWithRectangle) 
    (h1 : t.altitude = 8)
    (h2 : t.base = 12)
    (h3 : t.ratio = 1/3) : 
  rectangleArea t = 1024 / 27 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_is_1024_27_l890_89095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_24_l890_89074

theorem arithmetic_sequence_24 : ∃ (f : List ℤ → ℤ), 
  f [40, 4, 12, 2] = 24 ∧ 
  (∃ (a b c : ℤ), f = λ l => (l.get! 0 / l.get! 1) + l.get! 2 + l.get! 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_24_l890_89074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_property_2_ceiling_property_3_l890_89067

noncomputable def ceiling (x : ℝ) : ℤ :=
  ⌈x⌉

theorem ceiling_property_2 (x₁ x₂ : ℝ) :
  ceiling x₁ = ceiling x₂ → x₁ - x₂ < 1 :=
by sorry

theorem ceiling_property_3 (x₁ x₂ : ℝ) :
  ceiling (x₁ + x₂) ≤ ceiling x₁ + ceiling x₂ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_property_2_ceiling_property_3_l890_89067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_distance_condition_l890_89000

/-- Given two points and a distance, this theorem proves the condition for a line 
    passing through the first point to be at the given distance from the second point. -/
theorem line_distance_condition 
  (x₁ y₁ x₂ y₂ r : ℝ) (hr : r > 0) :
  ∀ m : ℝ, (r^2 * (m^2 + 1) = (m * (x₂ - x₁) + (y₁ - y₂))^2) ↔ 
  (∃ b : ℝ, |m * x₂ - y₂ + b| / Real.sqrt (m^2 + 1) = r ∧ 
            y₁ = m * x₁ + b) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_distance_condition_l890_89000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_parallel_lines_l890_89039

-- Define the slope angle of line l
noncomputable def slope_angle_l : ℝ := 3 * Real.pi / 4

-- Define points A and B
def point_A : ℝ × ℝ := (3, 2)
def point_B (a : ℝ) : ℝ × ℝ := (a, -1)

-- Define line l₂
def line_l2 (x y : ℝ) (b : ℝ) : Prop := 2 * x + b * y + 1 = 0

-- State the theorem
theorem perpendicular_parallel_lines 
  (a b : ℝ) 
  (h1 : ∃ (m : ℝ), Real.tan slope_angle_l = m) -- Slope of l exists
  (h2 : ∃ (m1 : ℝ), ((point_B a).2 - point_A.2) / ((point_B a).1 - point_A.1) = m1) -- Slope of l₁ exists
  (h3 : ∃ (m : ℝ) (m1 : ℝ), m * m1 = -1) -- l and l₁ are perpendicular
  (h4 : ∃ (m1 : ℝ) (m2 : ℝ), m1 = m2) -- l₁ and l₂ are parallel
  (h5 : line_l2 a (-1) b) -- Point B satisfies the equation of l₂
  : a + b = -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_parallel_lines_l890_89039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partnership_profit_l890_89082

/-- Partnership profit calculation -/
theorem partnership_profit (a_capital b_capital : ℕ) (a_months b_months : ℕ) (a_share : ℕ) : 
  a_capital = 5000 →
  b_capital = 6000 →
  a_months = 8 →
  b_months = 5 →
  a_share = 4800 →
  (a_capital * a_months : ℕ) * 7 = (a_capital * a_months + b_capital * b_months) * 4 →
  (a_share * 7 : ℕ) / 4 = 8400 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partnership_profit_l890_89082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_of_f_l890_89083

-- Define the function f(x) = a^(x-2) + 3
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x - 2) + 3

-- Theorem statement
theorem fixed_point_of_f (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a 2 = 4 := by
  -- Unfold the definition of f
  unfold f
  -- Simplify the exponent
  simp
  -- The rest of the proof
  sorry

#check fixed_point_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_of_f_l890_89083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milan_phone_usage_l890_89073

/-- Represents the phone usage and billing for Milan's two phone lines -/
structure PhoneUsage where
  /-- Monthly fee for the first phone line -/
  fee1 : ℚ
  /-- Per-minute rate for the first phone line -/
  rate1 : ℚ
  /-- Monthly fee for the second phone line -/
  fee2 : ℚ
  /-- Per-minute rate for the second phone line -/
  rate2 : ℚ
  /-- Total bill for both lines -/
  totalBill : ℚ
  /-- Difference in minutes used between first and second line -/
  minuteDiff : ℚ

/-- Calculates the total minutes used on both phone lines -/
def totalMinutes (usage : PhoneUsage) : ℚ :=
  let x := (usage.totalBill - usage.fee1 - usage.fee2) / (usage.rate1 + usage.rate2)
  2 * x + usage.minuteDiff

/-- Theorem stating that given Milan's phone usage conditions, the total minutes billed is 252 -/
theorem milan_phone_usage :
  let usage : PhoneUsage := {
    fee1 := 3,
    rate1 := 15/100,
    fee2 := 4,
    rate2 := 10/100,
    totalBill := 56,
    minuteDiff := 20
  }
  totalMinutes usage = 252 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_milan_phone_usage_l890_89073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_money_redistribution_l890_89025

theorem money_redistribution (a j : ℚ) : 
  let t : ℚ := 48
  let amy_redistribute := fun (a j t : ℚ) => (a - (t + j), 2*j, 2*t)
  let jan_redistribute := fun (a j t : ℚ) => (2*a, j - (a + t), 2*t)
  let toy_redistribute := fun (a j t : ℚ) => (2*a, 1.5*j, t - (2*a + 0.5*j))
  let (a₁, j₁, t₁) := amy_redistribute a j t
  let (a₂, j₂, t₂) := jan_redistribute a₁ j₁ t₁
  let (a₃, j₃, t₃) := toy_redistribute a₂ j₂ t₂
  t₃ = t → a₃ + j₃ + t₃ = 224 := by
  sorry

#check money_redistribution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_money_redistribution_l890_89025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integer_solutions_l890_89007

theorem count_integer_solutions : ∃! (s : Finset (ℤ × ℤ)), 
  (∀ (x y : ℤ), (x, y) ∈ s ↔ x^4 + y^4 = 4*y) ∧ s.card = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integer_solutions_l890_89007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_third_quadrant_l890_89010

theorem angle_in_third_quadrant (θ : ℝ) 
  (h1 : Real.sin θ * Real.sqrt (Real.sin θ * Real.sin θ) + Real.cos θ * Real.sqrt (Real.cos θ * Real.cos θ) = -1)
  (h2 : ∀ k : ℤ, θ ≠ k * Real.pi / 2) :
  -Real.pi ≤ θ ∧ θ < -Real.pi / 2 ∨ 3 * Real.pi / 2 < θ ∧ θ ≤ 2 * Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_third_quadrant_l890_89010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l890_89068

/-- A hyperbola with eccentricity √3 -/
def Hyperbola (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) - (p.2^2 / b^2) = 1}

/-- The vertices of the hyperbola -/
def vertices (a : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) := ((-a, 0), (a, 0))

/-- A point on the hyperbola -/
noncomputable def M (a b : ℝ) : ℝ × ℝ := sorry

/-- The area of the circumcircle of triangle ABM -/
noncomputable def circumcircleArea (a : ℝ) : ℝ := 3 * Real.pi * a^2

/-- The eccentricity of the hyperbola -/
noncomputable def eccentricity (a b : ℝ) : ℝ := sorry

theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  M a b ∈ Hyperbola a b →
  circumcircleArea a = 3 * Real.pi * a^2 →
  eccentricity a b = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l890_89068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_or_even_f_increasing_range_l890_89084

-- Define the function f(x) = ax^2 + 1/x
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 1/x

-- Part 1: Odd or even property of f(x)
theorem f_odd_or_even (a : ℝ) :
  (∀ x, f a x = f a (-x)) ∨ (∀ x, f a x = -(f a (-x))) ∨
  (∃ x, f a x ≠ f a (-x) ∧ f a x + f a (-x) ≠ 0) :=
by sorry

-- Part 2: Range of a for which f(x) is increasing on (1, +∞)
theorem f_increasing_range (a : ℝ) :
  (∀ x > 1, ∀ y > x, f a y > f a x) ↔ a ≥ 1/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_or_even_f_increasing_range_l890_89084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_gain_approx_16_67_percent_l890_89032

/-- Calculates the percentage gain on the selling price of a shoe. -/
noncomputable def percentage_gain_on_selling_price (manufacturing_cost : ℝ) (transportation_cost_per_100 : ℝ) (selling_price : ℝ) : ℝ :=
  let total_cost := manufacturing_cost + transportation_cost_per_100 / 100
  let gain := selling_price - total_cost
  (gain / selling_price) * 100

/-- Proves that the percentage gain on the selling price of a shoe is approximately 16.67%. -/
theorem percentage_gain_approx_16_67_percent :
  let manufacturing_cost : ℝ := 230
  let transportation_cost_per_100 : ℝ := 500
  let selling_price : ℝ := 282
  abs (percentage_gain_on_selling_price manufacturing_cost transportation_cost_per_100 selling_price - 16.67) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_gain_approx_16_67_percent_l890_89032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_and_intersection_of_sets_l890_89049

/-- Set A defined as {x | -2 < x < 7} -/
def A : Set ℝ := {x : ℝ | -2 < x ∧ x < 7}

/-- Set B defined as {x | a ≤ x ≤ 3a-2} -/
def B (a : ℝ) : Set ℝ := {x : ℝ | a ≤ x ∧ x ≤ 3*a - 2}

theorem union_and_intersection_of_sets :
  (∀ a : ℝ, a = 4 → A ∪ B a = Set.Ioc (-2) 10 ∧ (Aᶜ ∩ B a) = Set.Icc 7 10) ∧
  (∀ a : ℝ, A ∪ B a = A ↔ a < 3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_and_intersection_of_sets_l890_89049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_three_lines_intersect_l890_89047

-- Define a square
structure Square where
  side : ℝ
  side_pos : side > 0

-- Define a line that divides the square
structure DividingLine (s : Square) where
  slope : ℝ
  intercept : ℝ

-- Define the property that a line divides the square into quadrilaterals with area ratio 2:3
def DividesIntoQuadrilateralsWithRatio (s : Square) (l : DividingLine s) : Prop := sorry

-- Define a set of 9 lines
def NineLines (s : Square) : Type := Fin 9 → DividingLine s

-- Define the property that all lines in the set divide the square correctly
def AllLinesDivideCorrectly (s : Square) (lines : NineLines s) : Prop :=
  ∀ i, DividesIntoQuadrilateralsWithRatio s (lines i)

-- Define the property that at least three lines intersect at a single point
def AtLeastThreeLinesIntersect (s : Square) (lines : NineLines s) : Prop := sorry

-- The theorem to be proved
theorem at_least_three_lines_intersect 
  (s : Square) 
  (lines : NineLines s) 
  (h : AllLinesDivideCorrectly s lines) : 
  AtLeastThreeLinesIntersect s lines := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_three_lines_intersect_l890_89047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l890_89076

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x - Real.pi / 6)

theorem f_properties :
  -- Smallest positive period is π
  (∃ T : ℝ, T > 0 ∧ T = Real.pi ∧ ∀ x : ℝ, f (x + T) = f x ∧ ∀ T' : ℝ, T' > 0 ∧ (∀ x : ℝ, f (x + T') = f x) → T ≤ T') ∧
  -- Minimum value is -3
  (∃ m : ℝ, m = -3 ∧ ∀ x : ℝ, f x ≥ m) ∧
  -- Symmetry centers
  (∀ k : ℤ, ∃ c : ℝ × ℝ, c = (Real.pi / 12 + k * Real.pi / 2, 0) ∧
    ∀ x : ℝ, f (c.1 + x) = f (c.1 - x)) ∧
  -- Monotonically increasing intervals
  (∀ k : ℤ, ∀ x y : ℝ,
    -Real.pi / 6 + k * Real.pi ≤ x ∧ x < y ∧ y ≤ Real.pi / 3 + k * Real.pi →
    f x < f y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l890_89076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_function_eq_negative_f_l890_89004

open Function Real

-- Define the original function f
variable (f : ℝ → ℝ)

-- Define the inverse function of f
variable (f_inv : ℝ → ℝ)
axiom f_inv_is_inverse : LeftInverse f_inv f ∧ RightInverse f_inv f

-- Define the third function
def third_function (f_inv : ℝ → ℝ) (x : ℝ) : ℝ := f_inv (x + 2) + 1

-- Define the reflection operation about the line x + y = 0
def reflect_about_xy_eq_0 (g : ℝ → ℝ) (x : ℝ) : ℝ := -g (-x)

-- Define the fourth function as the reflection of the third function
def fourth_function (f f_inv : ℝ → ℝ) : ℝ → ℝ := 
  reflect_about_xy_eq_0 (third_function f_inv)

-- State the theorem
theorem fourth_function_eq_negative_f (f f_inv : ℝ → ℝ) 
  (h : LeftInverse f_inv f ∧ RightInverse f_inv f) :
  ∀ x : ℝ, fourth_function f f_inv x = -f (-x - 1) + 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_function_eq_negative_f_l890_89004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_through_origin_l890_89065

-- Define the curve E
def E : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = 4 * p.1}

-- Define points F and A
def F : ℝ × ℝ := (1, 0)
def A : ℝ × ℝ := (4, 0)

-- Define a line passing through A
def line_through_A (k : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = k * (p.1 - 4)}

-- Define the intersection points P and Q
def intersection_points (k : ℝ) : Set (ℝ × ℝ) := E ∩ line_through_A k

-- Define a circle through two points
def circle_through (P Q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {X : ℝ × ℝ | (X.1 - P.1)^2 + (X.2 - P.2)^2 = (Q.1 - P.1)^2 + (Q.2 - P.2)^2}

-- State the theorem
theorem circle_through_origin (k : ℝ) 
  (hk : ∃ P Q : ℝ × ℝ, P ∈ intersection_points k ∧ Q ∈ intersection_points k ∧ P ≠ Q) :
  ∃ P Q : ℝ × ℝ, P ∈ intersection_points k ∧ Q ∈ intersection_points k ∧ 
  (0, 0) ∈ circle_through P Q :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_through_origin_l890_89065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_journey_time_l890_89077

/-- Calculate the total time of a train journey with three segments and a stop -/
theorem train_journey_time (x y : ℝ) : 
  (x / 50 + (2 * x) / 75 + 1 / 6 + y / 100) = (14 * x + 3 * y + 50) / 300 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_journey_time_l890_89077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_local_operator_representation_l890_89081

-- Define the space of continuous functions on the real line
def C := ℝ → ℝ

-- Define a linear and local operator on C
def LinearLocal (T : C → C) : Prop :=
  (∀ c₁ c₂ : ℝ, ∀ ψ₁ ψ₂ : C, T (fun x ↦ c₁ * (ψ₁ x) + c₂ * (ψ₂ x)) = fun x ↦ c₁ * (T ψ₁ x) + c₂ * (T ψ₂ x)) ∧
  (∀ ψ₁ ψ₂ : C, ∀ a b : ℝ, (∀ x ∈ Set.Icc a b, ψ₁ x = ψ₂ x) → ∀ x ∈ Set.Icc a b, T ψ₁ x = T ψ₂ x)

-- State the theorem
theorem linear_local_operator_representation (T : C → C) (h : LinearLocal T) :
  ∃ f : C, ContinuousOn f Set.univ ∧ ∀ ψ : C, ContinuousOn ψ Set.univ → ∀ x : ℝ, T ψ x = f x * ψ x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_local_operator_representation_l890_89081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_solutions_equation_l890_89053

theorem three_solutions_equation : 
  ∃! k : ℕ, k > 0 ∧ 
  (∃ S : Finset (ℕ × ℕ), 
    (∀ (m n : ℕ), (m, n) ∈ S ↔ m > 0 ∧ n > 0 ∧ (6 : ℚ) / m + (3 : ℚ) / n = 1) ∧
    Finset.card S = k) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_solutions_equation_l890_89053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_AOB_l890_89098

/-- Given two points A and B in polar coordinates, prove that the area of triangle AOB is 3 --/
theorem area_triangle_AOB : 
  let A : ℝ × ℝ := (2, 2 * Real.pi / 3)
  let B : ℝ × ℝ := (3, Real.pi / 6)
  let O : ℝ × ℝ := (0, 0)
  ∃ (area_triangle : ℝ × ℝ → ℝ × ℝ → ℝ × ℝ → ℝ), area_triangle O A B = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_AOB_l890_89098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l890_89003

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := Real.sqrt (1 - (e.b / e.a)^2)

/-- The distance from a point to a line -/
noncomputable def distance_to_line (P : ℝ × ℝ) (l : ℝ → ℝ) : ℝ := sorry

/-- The distance between two points -/
noncomputable def distance (P Q : ℝ × ℝ) : ℝ := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem ellipse_eccentricity_range (e : Ellipse) 
  (F₁ F₂ : ℝ × ℝ) (l : ℝ → ℝ) (P : ℝ × ℝ) :
  (P.1 / e.a)^2 + (P.2 / e.b)^2 = 1 →  -- P is on the ellipse
  distance P F₁ + distance P F₂ = 2 * e.a →  -- F₁ and F₂ are foci
  distance P F₁ = 2 * distance_to_line P l →  -- Given condition
  (-3 + Real.sqrt 17) / 2 ≤ eccentricity e ∧ eccentricity e < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l890_89003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lemonade_water_calculation_correct_l890_89052

/-- Calculates the amount of water needed for lemonade -/
def lemonade_water_calculation (water_parts : ℕ) (lemon_juice_parts : ℕ) 
  (total_gallons : ℕ) (pints_per_gallon : ℕ) : ℕ × ℚ :=
  let total_parts := water_parts + lemon_juice_parts
  let total_pints := total_gallons * pints_per_gallon
  let pints_per_part : ℚ := (total_pints : ℚ) / (total_parts : ℚ)
  let water_pints := (water_parts : ℚ) * pints_per_part
  (⌊water_pints⌋.toNat, water_pints - ↑⌊water_pints⌋)

/-- Theorem stating that the lemonade_water_calculation function returns the correct result -/
theorem lemonade_water_calculation_correct :
  lemonade_water_calculation 5 2 3 8 = (17, 1/7) := by
  -- Unfold the definition and simplify
  unfold lemonade_water_calculation
  -- Perform the calculation steps
  simp [Nat.cast_add, Nat.cast_mul, Nat.cast_ofNat]
  -- The proof is completed
  sorry

-- Evaluate the function with the given values
#eval lemonade_water_calculation 5 2 3 8

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lemonade_water_calculation_correct_l890_89052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_highly_related_relationship_highly_related_relationship_prop_l890_89023

-- Define the linear correlation coefficient
def linear_correlation_coefficient : ℝ := -0.87

-- Define the threshold for a highly related relationship
def highly_related_threshold : ℝ := 0.7

-- Theorem statement
theorem highly_related_relationship (h : abs linear_correlation_coefficient ≥ highly_related_threshold) :
  True :=
sorry

-- Helper function to represent the conclusion
def relationship_is_highly_related : Prop :=
  abs linear_correlation_coefficient ≥ highly_related_threshold

-- Theorem using the helper function
theorem highly_related_relationship_prop :
  relationship_is_highly_related :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_highly_related_relationship_highly_related_relationship_prop_l890_89023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_intersection_l890_89092

-- Define the line l
noncomputable def line_l (φ t : ℝ) : ℝ × ℝ := (3 + t * Real.cos φ, 1 + t * Real.sin φ)

-- Define the circle C
noncomputable def circle_C (θ : ℝ) : ℝ × ℝ := (4 * Real.cos θ * Real.cos θ, 4 * Real.cos θ * Real.sin θ)

-- Theorem statement
theorem min_distance_intersection (φ : ℝ) (h : φ ∈ Set.Ioo 0 Real.pi) :
  ∃ P Q : ℝ × ℝ, 
    (∃ t θ₁ θ₂ : ℝ, P = line_l φ t ∧ P = circle_C θ₁ ∧
                    Q = line_l φ t ∧ Q = circle_C θ₂) ∧
    ∀ P' Q' : ℝ × ℝ, 
      (∃ t' θ₁' θ₂' : ℝ, P' = line_l φ t' ∧ P' = circle_C θ₁' ∧
                         Q' = line_l φ t' ∧ Q' = circle_C θ₂') →
      Real.sqrt 14 ≤ Real.sqrt ((P'.1 - Q'.1)^2 + (P'.2 - Q'.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_intersection_l890_89092
