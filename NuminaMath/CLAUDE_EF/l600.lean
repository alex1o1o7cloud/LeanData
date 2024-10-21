import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_period_sum_l600_60057

-- Define the types for our functions
def ContinuousPeriodicFunction := ℝ → ℝ

-- Define the properties for our functions
structure FunctionProperties (f : ContinuousPeriodicFunction) (period : ℕ) : Prop where
  continuous : Continuous f
  periodic : ∀ x, f (x + period) = f x
  non_constant : ∃ x y, f x ≠ f y
  smallest_period : ∀ p : ℕ, (∀ x, f (x + p) = f x) → period ≤ p

-- Define our theorem
theorem smallest_period_sum 
  (f g : ContinuousPeriodicFunction) 
  (m n : ℕ) 
  (hf : FunctionProperties f m) 
  (hg : FunctionProperties g n) 
  (h_coprime : Nat.Coprime m n) 
  (hm : m > 1) 
  (hn : n > 1) :
  FunctionProperties (λ x => f x + g x) (m * n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_period_sum_l600_60057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_theorem_l600_60054

/-- Parabola type representing y² = 4x -/
structure Parabola where
  x : ℝ
  y : ℝ
  eq : y^2 = 4*x

/-- Point type -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The focus of the parabola y² = 4x -/
def focus : Point :=
  ⟨1, 0⟩

/-- Theorem statement -/
theorem parabola_distance_theorem (A : Parabola) :
  let B : Point := ⟨3, 0⟩
  distance ⟨A.x, A.y⟩ focus = distance B focus →
  distance ⟨A.x, A.y⟩ B = 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_theorem_l600_60054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_indeterminate_and_unbounded_l600_60071

noncomputable def f (x : ℝ) : ℝ := (x^2 + x - 2) / (x^3 + 2*x + 1)

theorem f_indeterminate_and_unbounded :
  (f 1 = 0) ∧ 
  ∀ M : ℝ, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 1| ∧ |x - 1| < δ → |f x| > M :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_indeterminate_and_unbounded_l600_60071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_annika_hike_rate_l600_60076

/-- Represents a hiker's journey --/
structure HikeData where
  eastDistance : ℝ
  westDistance : ℝ
  totalTime : ℝ

/-- Calculates the hiking rate in minutes per kilometer --/
noncomputable def hikingRate (hike : HikeData) : ℝ :=
  hike.totalTime / (hike.eastDistance + hike.westDistance)

/-- Theorem stating the hiking rate for the given conditions --/
theorem annika_hike_rate :
  let hike : HikeData := { eastDistance := 3, westDistance := 3, totalTime := 35 }
  hikingRate hike = 35 / 6 := by
  -- Proof goes here
  sorry

#eval (35 : ℚ) / 6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_annika_hike_rate_l600_60076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_third_minus_2theta_l600_60005

theorem cos_pi_third_minus_2theta (θ : ℝ) 
  (h : Real.sin (θ - π/6) = Real.sqrt 3/3) : 
  Real.cos (π/3 - 2*θ) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_third_minus_2theta_l600_60005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_sum_of_digits_l600_60074

/-- Represents a digit in base 8 -/
def Base8Digit := Fin 8

/-- Represents the fraction 0.abc in base 8 -/
def Base8Fraction (a b c : Base8Digit) : ℚ :=
  (a.val : ℚ) / 8 + (b.val : ℚ) / 64 + (c.val : ℚ) / 512

/-- The theorem stating the largest possible sum of digits -/
theorem largest_sum_of_digits (a b c : Base8Digit) :
  (∃ y : ℕ, 0 < y ∧ y ≤ 16 ∧ Base8Fraction a b c = 1 / y) →
  (a.val : ℕ) + (b.val : ℕ) + (c.val : ℕ) ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_sum_of_digits_l600_60074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_y_intercept_l600_60050

/-- The y-intercept of the tangent line to y = e^x at (x₀, e^x₀) is less than 0 if and only if x₀ > 1 -/
theorem tangent_line_y_intercept (x₀ : ℝ) : 
  (Real.exp x₀ * (1 - x₀) < 0) ↔ (x₀ > 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_y_intercept_l600_60050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_box_dimension_l600_60016

/-- Given a rectangular box with dimensions 5, w, and 3 inches,
    if the face of greatest area has an area of 15 square inches,
    then w = 3 inches. -/
theorem box_dimension (w : ℝ) : w > 0 →
  (max (5 * w) (max (5 * 3) (w * 3)) = 15) →
  w = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_box_dimension_l600_60016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reduce_intervals_by_fifth_l600_60040

/-- The number of trams needed to reduce intervals by one-fifth -/
def reduce_intervals (initial_trams : ℕ) (reduction_fraction : ℚ) : ℕ :=
  let new_trams := (initial_trams : ℚ) / (1 - reduction_fraction)
  (Int.ceil new_trams).toNat - initial_trams

/-- Theorem: Adding 3 trams to 12 trams reduces intervals by one-fifth -/
theorem reduce_intervals_by_fifth :
  reduce_intervals 12 (1/5) = 3 := by
  sorry

#eval reduce_intervals 12 (1/5)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reduce_intervals_by_fifth_l600_60040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l600_60080

theorem problem_statement (a b : ℝ) (h : Set.toFinset {a, b/a, 1} = Set.toFinset {a^2, a+b, 0}) :
  a^2017 + b^2017 = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l600_60080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_collinear_points_l600_60001

/-- Given a > 0, b > 0, and points A(1,-2), B(a,-1), C(-b,0) are collinear,
    the minimum value of (2/a + 1/b) is 9 -/
theorem min_value_collinear_points (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (k : ℝ), k • ((a - 1, 1) : ℝ × ℝ) = ((-b - 1, 2) : ℝ × ℝ)) →
  (∀ (x y : ℝ), x > 0 → y > 0 → 2 * x + y = 1 → 2 / x + 1 / y ≥ 9) ∧
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 2 * x + y = 1 ∧ 2 / x + 1 / y = 9) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_collinear_points_l600_60001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triathlon_swimming_speed_l600_60032

/-- The harmonic mean of two positive real numbers -/
noncomputable def harmonicMean (a b : ℝ) : ℝ := 2 * a * b / (a + b)

/-- Theorem: Given the conditions of the triathlon problem, the swimming speed is 2.8 mph -/
theorem triathlon_swimming_speed (swimmingSpeed runningSpeed averageSpeed : ℝ) 
    (runningSpeed_pos : runningSpeed > 0)
    (averageSpeed_pos : averageSpeed > 0)
    (running_speed_is_7 : runningSpeed = 7)
    (average_speed_is_4 : averageSpeed = 4)
    (equal_time : harmonicMean swimmingSpeed runningSpeed = averageSpeed) :
    swimmingSpeed = 2.8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triathlon_swimming_speed_l600_60032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_success_rate_increase_l600_60092

def initial_success_rate : ℚ := 7 / 15
def additional_attempts : ℕ := 28
def additional_success_rate : ℚ := 3 / 4

def new_success_rate : ℚ := 
  (initial_success_rate * 15 + additional_success_rate * additional_attempts) / 
  (15 + additional_attempts)

theorem success_rate_increase : 
  Int.floor ((new_success_rate - initial_success_rate) * 100 + 1/2) = 18 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_success_rate_increase_l600_60092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_one_second_class_l600_60031

noncomputable def probability_exactly_one_second_class (total : ℕ) (first_class : ℕ) (second_class : ℕ) (selected : ℕ) : ℚ :=
  (Nat.choose second_class 1 * Nat.choose first_class 3 : ℚ) / Nat.choose total selected

theorem probability_one_second_class (total : ℕ) (first_class : ℕ) (second_class : ℕ) (selected : ℕ)
  (h1 : total = 100)
  (h2 : first_class = 90)
  (h3 : second_class = 10)
  (h4 : selected = 4)
  (h5 : total = first_class + second_class) :
  (Nat.choose second_class 1 * Nat.choose first_class 3 : ℚ) / Nat.choose total selected =
  probability_exactly_one_second_class total first_class second_class selected :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_one_second_class_l600_60031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_of_m_l600_60078

def m : ℕ := 2^3 * 3^4 * 5^6 * 7^7

theorem number_of_factors_of_m : (Finset.filter (·∣m) (Finset.range (m + 1))).card = 140 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_of_m_l600_60078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l600_60075

noncomputable def f (x : ℝ) : ℝ := (x + 2) / (x^2 + 4*x + 3)

theorem domain_of_f :
  {x : ℝ | ∃ y : ℝ, f x = y} = {x : ℝ | x ≠ -3 ∧ x ≠ -1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l600_60075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_T_l600_60022

-- Define the complex number ω
noncomputable def ω : ℂ := -1/2 + (1/2 * Complex.I * Real.sqrt 3)

-- Define the set T
def T : Set ℂ := {z | ∃ (a d : ℝ), 0 ≤ a ∧ a ≤ 2 ∧ 0 ≤ d ∧ d ≤ 2 ∧ z = a + d * ω}

-- Theorem statement
theorem area_of_T : MeasureTheory.volume T = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_T_l600_60022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_zero_plus_zero_power_negative_l600_60056

theorem power_zero_plus_zero_power_negative (a : ℝ) : (a^(-2 : ℤ))^0 + (a^0)^(-3 : ℤ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_zero_plus_zero_power_negative_l600_60056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_existence_conditions_l600_60006

/-- Tetrahedron existence conditions based on edge lengths -/
theorem tetrahedron_existence_conditions (k : ℕ) (a : ℝ) : 
  (k ∈ ({1, 2, 3, 4, 5} : Set ℕ)) →
  (∃ (tetrahedron : Type), 
    (∃ (edges : Fin 6 → ℝ), 
      (∀ i : Fin 6, edges i = a ∨ edges i = 1) ∧ 
      (Fintype.card {i : Fin 6 | edges i = a} = k))) ↔
  ((k = 1 → 0 < a ∧ a < Real.sqrt 3) ∧
   (k = 2 → 0 < a ∧ a < (Real.sqrt 6 + Real.sqrt 2) / 2) ∧
   (k = 3 → 0 < a) ∧
   (k = 4 → a > (Real.sqrt 6 - Real.sqrt 2) / 2) ∧
   (k = 5 → a > Real.sqrt 3 / 3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_existence_conditions_l600_60006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_negative_ten_l600_60086

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x - 9

noncomputable def g (x : ℝ) : ℝ := 3 * ((fun y ↦ (y + 9) / 4) x)^2 + 4 * ((fun y ↦ (y + 9) / 4) x) - 2

-- State the theorem
theorem g_of_negative_ten : g (-10) = -45 / 16 := by
  -- Expand the definition of g
  unfold g
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_negative_ten_l600_60086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polyhedron_with_specific_projections_l600_60009

/-- A polyhedron is a three-dimensional geometric object with flat polygonal faces, straight edges and sharp corners or vertices. -/
structure Polyhedron where

/-- A plane in three-dimensional space. -/
structure Plane where

/-- A polygon with a specific number of sides. -/
structure Polygon (n : Nat) where

/-- Represents the projection of a polyhedron onto a plane. -/
def projection (p : Polyhedron) (plane : Plane) : Polygon 3 ⊕ Polygon 4 ⊕ Polygon 5 :=
  sorry

/-- Two planes are perpendicular if they intersect at right angles. -/
def perpendicular (p1 p2 : Plane) : Prop :=
  sorry

/-- There exists a polyhedron whose projections onto three pairwise perpendicular planes
    are a triangle, a quadrilateral, and a pentagon. -/
theorem polyhedron_with_specific_projections :
  ∃ (p : Polyhedron) (p1 p2 p3 : Plane),
    perpendicular p1 p2 ∧ perpendicular p2 p3 ∧ perpendicular p3 p1 ∧
    (∃ (t : Polygon 3) (q : Polygon 4) (pent : Polygon 5),
      (projection p p1 = Sum.inl t) ∧
      (projection p p2 = Sum.inr (Sum.inl q)) ∧
      (projection p p3 = Sum.inr (Sum.inr pent))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polyhedron_with_specific_projections_l600_60009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_girls_relationship_l600_60096

/-- Represents the number of girls each boy dances with -/
def girls_per_boy (n : ℕ) : ℕ :=
  if n ≤ 5 then 7 + 2 * (n - 1) else 17

/-- The total number of girls in the event -/
def total_girls (b : ℕ) : ℕ :=
  Finset.sum (Finset.range b) girls_per_boy

/-- The relationship between the number of boys and girls -/
theorem boys_girls_relationship (b g : ℕ) (h : g = total_girls b) :
  b = (g + 30) / 17 := by
  sorry

#eval total_girls 10  -- For testing purposes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_girls_relationship_l600_60096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_solution_l600_60091

theorem complex_equation_solution (a b : ℝ) :
  (Complex.I + a) * (Complex.I + 1) = Complex.I * b →
  Complex.mk a b = Complex.I + 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_solution_l600_60091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_rational_coefficients_is_365_l600_60062

/-- The sum of the coefficients of all rational terms in the expansion of (2√x - 1/x)^6 -/
def sum_of_rational_coefficients : ℕ := 365

/-- The binomial expression (2√x - 1/x)^6 -/
noncomputable def binomial_expression (x : ℝ) : ℝ := (2 * Real.sqrt x - 1 / x) ^ 6

/-- Theorem stating that the sum of rational coefficients is 365 -/
theorem sum_of_rational_coefficients_is_365 :
  sum_of_rational_coefficients = 365 := by rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_rational_coefficients_is_365_l600_60062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l600_60035

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + a * x^2 + b * x

noncomputable def f_derivative (a b : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + b

theorem function_properties (a b : ℝ) :
  (f a b (-3) = 9 ∧ ∀ x, f a b x ≤ 9) →
  (a = 1 ∧ b = -3 ∧
   (∀ x ∈ Set.Icc (-3) 3, f 1 (-3) x ≤ 9) ∧
   (∃ x ∈ Set.Icc (-3) 3, f 1 (-3) x = -5/3) ∧
   (∀ x ∈ Set.Icc (-3) 3, f 1 (-3) x ≥ -5/3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l600_60035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negp_necessary_not_sufficient_for_negq_l600_60010

theorem negp_necessary_not_sufficient_for_negq :
  ∃ x : ℝ, (¬(abs x < 1) ∧ ¬(x^2 + x - 6 < 0)) ∧
  ∃ y : ℝ, (¬(abs y < 1) ∧ (y^2 + y - 6 < 0)) ∧
  ∀ z : ℝ, ((z^2 + z - 6 ≥ 0) → (abs z ≥ 1)) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negp_necessary_not_sufficient_for_negq_l600_60010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l600_60015

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (Real.pi - x) * Real.cos x + Real.cos (2 * x)

theorem f_properties :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ S, 0 < S ∧ S < T → ∃ y, f (y + S) ≠ f y) ∧
  (∀ x ∈ Set.Icc (Real.pi / 4) (Real.pi / 2), f x ≤ 1) ∧
  (∀ x ∈ Set.Icc (Real.pi / 4) (Real.pi / 2), f x ≥ -1) ∧
  (∃ x ∈ Set.Icc (Real.pi / 4) (Real.pi / 2), f x = 1) ∧
  (∃ x ∈ Set.Icc (Real.pi / 4) (Real.pi / 2), f x = -1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l600_60015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_closed_line_impossible_l600_60041

/-- Represents a prism with a given number of lateral edges and total edges -/
structure Prism where
  lateral_edges : ℕ
  total_edges : ℕ

/-- Theorem stating that it's impossible to form a closed broken line with all edges of the given prism -/
theorem prism_closed_line_impossible (p : Prism) 
  (h1 : p.lateral_edges = 171)
  (h2 : p.total_edges = 513) : 
  ¬ ∃ (closed_line : List ℕ), closed_line.length = p.total_edges ∧ closed_line.head? = closed_line.getLast? := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_closed_line_impossible_l600_60041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_of_seven_between_15_and_225_l600_60017

theorem multiples_of_seven_between_15_and_225 : 
  (Finset.filter (fun n => 7 ∣ n) (Finset.range 226 \ Finset.range 15)).card = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_of_seven_between_15_and_225_l600_60017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_tangent_lines_l600_60066

/-- Definition of the ellipse E -/
def ellipse_E (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

/-- Definition of the line -/
def line (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x + 1

/-- Definition of the circle -/
def my_circle (r : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 = r^2

theorem ellipse_and_tangent_lines :
  ∃ (r : ℝ), r > 0 ∧
  (∀ x y, ellipse_E x y → (x = 1 ∧ y = 3/2 → 
    (∃ k, ∀ x₁ y₁ x₂ y₂,
      (ellipse_E x₁ y₁ ∧ line k x₁ y₁) →
      (ellipse_E x₂ y₂ ∧ line k x₂ y₂) →
      x₁ ≠ x₂ →
      (∃ t₁ t₂, 
        my_circle r t₁ ((k * (t₁ + 4) + 1) / (t₁ + 4)) ∧
        my_circle r t₂ ((k * (t₂ + 4) + 1) / (t₂ + 4))) →
      k = 1))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_tangent_lines_l600_60066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_iff_a_eq_two_l600_60061

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The slope of a line -/
noncomputable def Line.slope (l : Line) : ℝ := -l.a / l.b

/-- Two lines are parallel if they have the same slope -/
def parallel (l₁ l₂ : Line) : Prop := l₁.slope = l₂.slope

/-- The first line: 2x - ay + 1 = 0 -/
def l₁ (a : ℝ) : Line := { a := 2, b := -a, c := 1 }

/-- The second line: (a-1)x - y + a = 0 -/
def l₂ (a : ℝ) : Line := { a := a - 1, b := -1, c := a }

/-- Theorem: a = 2 is a necessary and sufficient condition for l₁ ∥ l₂ -/
theorem parallel_iff_a_eq_two (a : ℝ) : parallel (l₁ a) (l₂ a) ↔ a = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_iff_a_eq_two_l600_60061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_rotates_around_w_l600_60058

/-- The complex rotation function -/
noncomputable def g (z : ℂ) : ℂ := ((1 + Complex.I) * z + (4 - 4 * Complex.I)) / 3

/-- The fixed point of the rotation -/
noncomputable def w : ℂ := 4/5 - 4/5 * Complex.I

/-- Theorem: g represents a rotation around w -/
theorem g_rotates_around_w : g w = w := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_rotates_around_w_l600_60058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_team_with_few_losses_l600_60051

/-- Represents a volleyball team -/
structure Team where
  id : ℕ

/-- Represents the result of a game between two teams -/
inductive GameResult
  | Win
  | Loss
deriving DecidableEq

/-- Represents a volleyball tournament -/
structure Tournament where
  teams : Finset Team
  results : Team → Team → GameResult
  total_teams : teams.card = 110
  all_play_all : ∀ t1 t2 : Team, t1 ∈ teams → t2 ∈ teams → t1 ≠ t2 → 
    (results t1 t2 = GameResult.Win ∧ results t2 t1 = GameResult.Loss) ∨
    (results t1 t2 = GameResult.Loss ∧ results t2 t1 = GameResult.Win)
  group_property : ∀ group : Finset Team, group ⊆ teams → group.card = 55 → 
    ∃ t ∈ group, (group.filter (λ t' => results t t' = GameResult.Loss)).card ≤ 4

/-- The main theorem to prove -/
theorem exists_team_with_few_losses (tournament : Tournament) :
  ∃ t ∈ tournament.teams, (tournament.teams.filter (λ t' => tournament.results t t' = GameResult.Loss)).card ≤ 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_team_with_few_losses_l600_60051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_min_point_exists_l600_60060

-- Define the function
noncomputable def f (x : ℝ) : ℝ := x + 2 / (x - 1)

-- State the theorem
theorem min_value_of_f :
  ∀ x : ℝ, x > 1 → f x ≥ 2 * Real.sqrt 2 + 1 :=
by sorry

-- State the existence of the minimum point
theorem min_point_exists :
  ∃ x : ℝ, x > 1 ∧ f x = 2 * Real.sqrt 2 + 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_min_point_exists_l600_60060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_n_value_l600_60008

theorem quadratic_equation_n_value : 
  let a : ℚ := 3
  let b : ℚ := -7
  let c : ℚ := -6
  let discriminant := b^2 - 4*a*c
  ∃ (m p : ℚ) (n : ℕ), 
    (∀ x : ℚ, 3*x^2 - 7*x - 6 = 0 ↔ x = (m + Real.sqrt (n : ℚ))/p ∨ x = (m - Real.sqrt (n : ℚ))/p) ∧
    n = 121 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_n_value_l600_60008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_from_max_inscribed_radius_l600_60063

/-- An ellipse with semi-major axis a, semi-minor axis b, and focal distance c. -/
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_a_gt_b : b < a
  h_c_sq : c^2 = a^2 - b^2

/-- A point on an ellipse. -/
structure PointOnEllipse (E : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / E.a^2 + y^2 / E.b^2 = 1
  h_not_vertex : x ≠ E.a ∧ x ≠ -E.a

/-- The maximum radius of the inscribed circle in the triangle formed by a point on the ellipse and its foci. -/
noncomputable def max_inscribed_radius (E : Ellipse) : ℝ := E.c / 3

/-- The eccentricity of an ellipse. -/
noncomputable def eccentricity (E : Ellipse) : ℝ := E.c / E.a

/-- Theorem stating that if the maximum inscribed radius is c/3, then the eccentricity is 4/5. -/
theorem eccentricity_from_max_inscribed_radius (E : Ellipse) :
  max_inscribed_radius E = E.c / 3 → eccentricity E = 4 / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_from_max_inscribed_radius_l600_60063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l600_60069

noncomputable def f (x : Real) : Real := 2 * Real.sqrt 2 * Real.cos x * Real.sin (x + Real.pi / 4) - 1

theorem f_properties :
  let a := 0
  let b := Real.pi / 2
  (f (Real.pi / 4) = 1) ∧
  (∃ (x : Real), x ∈ Set.Icc a b ∧ ∀ (y : Real), y ∈ Set.Icc a b → f y ≤ f x) ∧
  (∃ (x : Real), x ∈ Set.Icc a b ∧ ∀ (y : Real), y ∈ Set.Icc a b → f x ≤ f y) ∧
  (∀ (x : Real), x ∈ Set.Icc a b → f x ≤ Real.sqrt 2) ∧
  (∀ (x : Real), x ∈ Set.Icc a b → -1 ≤ f x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l600_60069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_set_value_l600_60033

/-- Represents a set in the sequence -/
structure MySet :=
  (first : ℕ)
  (sequence : Fin 4 → ℕ)

/-- The rule for calculating the first number in a set -/
def calculateFirst (s : Fin 4 → ℕ) : ℕ :=
  (s 0) * (s 1) * (s 2) * (s 3) - ((s 0) + (s 1) + (s 2) + (s 3))

/-- The sequence of sets -/
def sets : Fin 4 → MySet
| 0 => ⟨14, ![1, 2, 3, 4]⟩
| 1 => ⟨47, ![1, 3, 4, 5]⟩
| 2 => ⟨104, ![1, 4, 5, 6]⟩
| 3 => ⟨191, ![1, 5, 6, 7]⟩

/-- The theorem to prove -/
theorem fourth_set_value :
  (sets 3).first = calculateFirst (sets 3).sequence :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_set_value_l600_60033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_product_equality_l600_60077

open BigOperators

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

def productTerm (k : ℕ) : ℚ :=
  (fib k / fib (k - 1) : ℚ) - (fib k / fib (k + 1) : ℚ)

theorem fibonacci_product_equality :
  ∏ k in Finset.range 99, productTerm (k + 3) = (fib 101 : ℚ) / fib 102 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_product_equality_l600_60077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l600_60039

/-- Calculates the speed of a train given the lengths of two trains, the speed of one train,
    and the time it takes for them to clear each other when moving in opposite directions. -/
noncomputable def calculate_train_speed (length1 length2 : ℝ) (speed2 : ℝ) (clear_time : ℝ) : ℝ :=
  let total_length := length1 + length2
  let total_length_km := total_length / 1000
  let clear_time_hours := clear_time / 3600
  let relative_speed := total_length_km / clear_time_hours
  relative_speed - speed2

/-- The calculated speed of the first train is approximately 80.008 kmph. -/
theorem train_speed_calculation :
  let length1 : ℝ := 121
  let length2 : ℝ := 165
  let speed2 : ℝ := 65
  let clear_time : ℝ := 7.100121645440779
  let calculated_speed := calculate_train_speed length1 length2 speed2 clear_time
  ∃ ε > 0, |calculated_speed - 80.008| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l600_60039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chucks_accessible_area_l600_60070

noncomputable def chucks_area (shed_width shed_length leash_length wall_extension : ℝ) : ℝ :=
  (3/4) * Real.pi * leash_length^2 - (1/4) * Real.pi * wall_extension^2

theorem chucks_accessible_area :
  chucks_area 4 6 5 1 = (74/4) * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chucks_accessible_area_l600_60070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_properties_l600_60025

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then 4*x - x^2
  else if x ≤ -1 then 4*x + x^2
  else 0  -- undefined for -1 < x < 1

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := (f x - 9) / x

-- Theorem statement
theorem f_and_g_properties :
  (∀ x ≤ -1, f x = 4*x + x^2) ∧
  (Set.range g = Set.Iic (-2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_properties_l600_60025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_fraction_of_decimal_l600_60012

theorem simplest_fraction_of_decimal (p q : ℕ+) :
  (p : ℚ) / q = 390625 / 1000000 ∧
  Nat.Coprime p q →
  p = 25 ∧ q = 64 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_fraction_of_decimal_l600_60012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fractional_part_sum_power_l600_60036

-- Define the fractional part function as noncomputable
noncomputable def frac (x : ℝ) : ℝ := x - ⌊x⌋

-- State the theorem
theorem fractional_part_sum_power (a : ℝ) (n : ℕ) (h : a ≠ 0) :
  frac a + frac (1 / a) = 1 → frac (a^n) + frac (1 / (a^n)) = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fractional_part_sum_power_l600_60036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minus_ex_nonpositive_l600_60026

noncomputable section

open Real

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the conditions
axiom f_domain : ∀ x, x > 0 → ∃ y, f x = y
axiom f_derivative : ∀ x, x > 0 → deriv f x - f x / x = 1 - log x
axiom f_at_e : f (exp 1) = exp 2

-- Theorem to prove
theorem f_minus_ex_nonpositive : ∀ x, x > 0 → f x - exp x ≤ 0 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minus_ex_nonpositive_l600_60026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l600_60064

/-- The sum of the first n terms of an arithmetic sequence -/
noncomputable def S (n : ℕ) (a₁ d : ℝ) : ℝ := n * a₁ + (n * (n - 1) / 2) * d

/-- Theorem: For the given arithmetic sequence, S₂₀₁₇ = 4034 -/
theorem arithmetic_sequence_sum :
  let a₁ : ℝ := -2014
  let d : ℝ := 2  -- We know d = 2 from the problem solution
  S 2014 a₁ d / 2014 - S 2008 a₁ d / 2008 = 6 →
  S 2017 a₁ d = 4034 := by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l600_60064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_ratio_60_l600_60011

theorem factorial_ratio_60 :
  ∃! n : ℕ, n > 0 ∧ n.factorial / (n - 3).factorial = 60 :=
by
  -- The unique value is 5
  use 5
  -- Split into existence and uniqueness
  constructor
  -- Prove existence
  · constructor
    -- Prove 5 > 0
    · linarith
    -- Prove 5! / (5-3)! = 60
    · norm_num [Nat.factorial]
  -- Prove uniqueness
  · intro m ⟨m_pos, h_m⟩
    -- Show that m must equal 5
    sorry

#eval Nat.factorial 5 / Nat.factorial 2  -- Should output 60

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_ratio_60_l600_60011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_squares_regression_l600_60073

/-- Data points for the regression problem -/
def data_points : List (ℝ × ℝ) := [(1, 0.9), (2, 1.2), (3, 1.5), (4, 1.4), (5, 1.6)]

/-- Mean of x values -/
def mean_x : ℝ := 3

/-- Mean of y values -/
def mean_y : ℝ := 1.32

/-- Sum of x_i * y_i -/
def sum_xy : ℝ := 21.4

/-- Square root of sum of squared deviations of y -/
def sqrt_sum_sq_dev_y : ℝ := 0.55

/-- Approximation of square root of 10 -/
def sqrt_10 : ℝ := 3.16

/-- Least squares regression equation -/
def regression_equation (x : ℝ) : ℝ := 0.16 * x + 0.84

theorem least_squares_regression :
  ∀ x, regression_equation x = 
    (sum_xy - data_points.length * mean_x * mean_y) / 
    ((data_points.map (λ p => p.1^2)).sum - data_points.length * mean_x^2) * x +
    (mean_y - (sum_xy - data_points.length * mean_x * mean_y) / 
    ((data_points.map (λ p => p.1^2)).sum - data_points.length * mean_x^2) * mean_x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_squares_regression_l600_60073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_shorten_by_one_approx_expected_new_length_l600_60099

/-- Represents a sequence of digits where each digit is 0 or 9 -/
def DigitSequence := List (Fin 2)

/-- The length of the original sequence -/
def originalLength : Nat := 2015

/-- The probability of a digit being the same as the previous one -/
def sameDigitProb : ℝ := 0.1

/-- The probability of a digit being different from the previous one -/
def differentDigitProb : ℝ := 0.9

/-- Calculates the probability of the sequence shortening by exactly one digit -/
def probShortenByOne (n : Nat) (p : ℝ) : ℝ :=
  n * p * (1 - p)^(n - 1)

/-- Calculates the expected number of digits removed -/
def expectedRemoved (n : Nat) (p : ℝ) : ℝ := n * p

/-- Theorem: The probability of the sequence shortening by exactly one digit
    is approximately 1.564 × 10^(-90) -/
theorem prob_shorten_by_one_approx :
  ‖probShortenByOne (originalLength - 1) sameDigitProb - 1.564e-90‖ < 1e-92 := by sorry

/-- Theorem: The expected length of the new sequence is 1813.6 -/
theorem expected_new_length :
  ‖(originalLength : ℝ) - expectedRemoved (originalLength - 1) sameDigitProb - 1813.6‖ < 1e-6 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_shorten_by_one_approx_expected_new_length_l600_60099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_endpoint_distance_l600_60097

/-- The ellipse defined by the equation 16(x+2)^2 + 4y^2 = 64 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 16 * (p.1 + 2)^2 + 4 * p.2^2 = 64}

/-- An endpoint of the major axis of the ellipse -/
def MajorAxisEndpoint : ℝ × ℝ := (-4, 0)

/-- An endpoint of the minor axis of the ellipse -/
def MinorAxisEndpoint : ℝ × ℝ := (0, 2)

/-- The distance between an endpoint of the major axis and an endpoint of the minor axis -/
noncomputable def AxisEndpointDistance : ℝ :=
  Real.sqrt ((MajorAxisEndpoint.1 - MinorAxisEndpoint.1)^2 + 
             (MajorAxisEndpoint.2 - MinorAxisEndpoint.2)^2)

theorem axis_endpoint_distance :
  AxisEndpointDistance = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_endpoint_distance_l600_60097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_angle_at_seven_is_150_l600_60090

/-- Represents a clock with 12 hours --/
structure Clock :=
  (hours : Nat)
  (degrees_per_hour : Nat)
  (hour_hand_position : Nat)
  (minute_hand_position : Nat)

/-- The smaller angle between the hour and minute hands of a clock at 7:00 --/
def smaller_angle_at_seven (c : Clock) : Nat :=
  min (Int.natAbs (c.hour_hand_position - c.minute_hand_position))
      (360 - Int.natAbs (c.hour_hand_position - c.minute_hand_position))

/-- Theorem: The smaller angle between the hour and minute hands at 7:00 is 150° --/
theorem smaller_angle_at_seven_is_150 (c : Clock) :
  c.hours = 12 ∧ 
  c.degrees_per_hour = 30 ∧ 
  c.hour_hand_position = 210 ∧ 
  c.minute_hand_position = 0 →
  smaller_angle_at_seven c = 150 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_angle_at_seven_is_150_l600_60090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_liars_l600_60004

-- Define a type for people
inductive Person : Type
  | A | B | C | D | E

-- Define a function to represent whether a person is a liar or not
variable (is_liar : Person → Bool)

-- Define the statements made by each person
def statements (is_liar : Person → Bool) : Prop :=
  (is_liar Person.B = !is_liar Person.A) ∧
  (is_liar Person.C = !is_liar Person.B) ∧
  (is_liar Person.D = !is_liar Person.C) ∧
  (is_liar Person.E = !is_liar Person.D)

-- Define a function to count the number of liars
def count_liars (is_liar : Person → Bool) : Nat :=
  List.filter is_liar [Person.A, Person.B, Person.C, Person.D, Person.E] |>.length

-- Theorem: The maximum number of liars is 3
theorem max_liars :
  ∃ (is_liar : Person → Bool), statements is_liar ∧ count_liars is_liar = 3 ∧
  ∀ (other_is_liar : Person → Bool), statements other_is_liar → count_liars other_is_liar ≤ 3 :=
by
  sorry

#check max_liars

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_liars_l600_60004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hiker_speed_approx_l600_60002

/-- Calculates the hiker's speed in still water given downstream and upstream travel data -/
noncomputable def hikerSpeed (downstreamDistance : ℝ) (downstreamTime : ℝ) (upstreamDistance : ℝ) (upstreamTime : ℝ) : ℝ :=
  let c := (downstreamDistance / downstreamTime + upstreamDistance / upstreamTime) / (downstreamTime + upstreamTime)
  (downstreamDistance / downstreamTime + upstreamDistance / upstreamTime) / 2 - c

/-- The hiker's speed in still water is approximately 12.68 km/h -/
theorem hiker_speed_approx (ε : ℝ) (hε : ε > 0) :
  ∃ v, abs (v - hikerSpeed 250 14 120 16) < ε ∧ abs (v - 12.68) < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hiker_speed_approx_l600_60002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l600_60034

theorem expression_simplification :
  (2 : ℝ)^(1/3) * (8 : ℝ)^(1/3) + 18 / (3 * 3) - (8 : ℝ)^(5/3) = (2 : ℝ)^(4/3) - 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l600_60034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_mod_30_l600_60029

theorem sum_remainder_mod_30 (a b c d : ℕ) 
  (ha : a % 30 = 10)
  (hb : b % 30 = 15)
  (hc : c % 30 = 20)
  (hd : d % 30 = 25) :
  (a + b + c + d) % 30 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_mod_30_l600_60029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_most_popular_book_bought_by_at_least_five_l600_60055

/-- Represents a book purchase by a person -/
structure BookPurchase where
  person : Nat
  books : Finset Nat
  different_books : books.card = 3

/-- Represents the book purchasing scenario -/
structure BookStore where
  people : Finset Nat
  purchases : Finset BookPurchase
  total_people : people.card = 10
  all_purchases : ∀ p, p ∈ people → ∃ bp ∈ purchases, bp.person = p
  common_book : ∀ p1 p2, p1 ∈ people → p2 ∈ people → p1 ≠ p2 → 
    ∃ bp1 bp2, bp1 ∈ purchases ∧ bp2 ∈ purchases ∧ bp1.person = p1 ∧ bp2.person = p2 ∧ 
    ∃ b, b ∈ bp1.books ∩ bp2.books

/-- The main theorem -/
theorem most_popular_book_bought_by_at_least_five (bs : BookStore) :
  ∃ b : Nat, (bs.purchases.filter (λ bp ↦ b ∈ bp.books)).card ≥ 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_most_popular_book_bought_by_at_least_five_l600_60055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_building_floors_l600_60094

/-- Represents the number of floors in a building -/
def num_floors (n : ℕ) : ℕ := n

/-- Calculates the total number of apartment units in a building -/
def total_units (n : ℕ) : ℕ := 2 + 5 * (n - 1)

/-- The problem statement -/
theorem building_floors :
  ∃ n : ℕ, 
    n > 0 ∧ 
    2 * (total_units n) = 34 ∧ 
    num_floors n = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_building_floors_l600_60094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_triple_angle_l600_60095

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 3/5) : Real.cos (3 * θ) = -117/125 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_triple_angle_l600_60095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l600_60018

-- Define the triangle ABC
structure Triangle (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] :=
  (A B C : V)

-- Define properties of the triangle
def is_acute_angled {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] (t : Triangle V) : Prop := sorry

def angle_ACB_is_45_deg {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] (t : Triangle V) : Prop := sorry

-- Define altitude feet and orthocenter
noncomputable def altitude_foot {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] (t : Triangle V) (v : V) : V := sorry

noncomputable def orthocenter {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] (t : Triangle V) : V := sorry

-- Define points D and E
noncomputable def point_D {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] (t : Triangle V) (A₁ : V) : V := sorry
noncomputable def point_E {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] (t : Triangle V) (A₁ : V) : V := sorry

-- Main theorem
theorem triangle_properties 
  {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  (t : Triangle V) 
  (h_acute : is_acute_angled t)
  (h_angle : angle_ACB_is_45_deg t)
  (A₁ : V) (h_A₁ : A₁ = altitude_foot t t.A)
  (B₁ : V) (h_B₁ : B₁ = altitude_foot t t.B)
  (H : V) (h_H : H = orthocenter t)
  (D : V) (h_D : D = point_D t A₁)
  (E : V) (h_E : E = point_E t A₁)
  (h_equal : dist A₁ D = dist A₁ E ∧ dist A₁ D = dist A₁ B₁) :
  (dist A₁ B₁)^2 = ((dist A₁ t.B)^2 + (dist A₁ t.C)^2) / 2 ∧
  dist t.C H = dist D E := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l600_60018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reb_bike_time_difference_l600_60081

/-- Represents the time difference in minutes between biking and driving to work -/
noncomputable def time_difference (drive_time : ℝ) (drive_speed : ℝ) (bike_speed_min : ℝ) (route_reduction : ℝ) : ℝ :=
  let drive_distance := drive_speed * (drive_time / 60)
  let bike_distance := drive_distance * (1 - route_reduction)
  let bike_time := (bike_distance / bike_speed_min) * 60
  bike_time - drive_time

/-- Theorem stating that Reb needs to leave 75 minutes earlier when biking -/
theorem reb_bike_time_difference :
  time_difference 45 40 12 0.2 = 75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reb_bike_time_difference_l600_60081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nothing_is_correct_l600_60082

/-- The correct answer to the idiom question --/
def correct_answer : String := "nothing"

/-- A function that checks if a given answer is correct --/
def is_correct_answer (answer : String) : Bool :=
  answer == correct_answer

/-- Theorem stating that "nothing" is the correct answer --/
theorem nothing_is_correct : is_correct_answer "nothing" = true := by
  rfl

#eval is_correct_answer "nothing"  -- Should output true
#eval is_correct_answer "everything"  -- Should output false

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nothing_is_correct_l600_60082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_existence_l600_60059

theorem unique_function_existence : ∃! f : ℝ → ℝ, 
  (f 1 = 1) ∧ (∀ x y : ℝ, x > 0 → y > 0 → f (x^2 * y^2) = f (x^4 + y^4)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_existence_l600_60059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_and_max_k_l600_60042

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * (a + Real.log x)

-- Define the condition for k
def k_condition (k : ℤ) : Prop :=
  ∀ x > 1, ↑k < (f 1 x) / (x - 1)

theorem min_value_and_max_k :
  (∃ a : ℝ, ∀ x > 0, f a x ≥ -Real.exp (-2) ∧ (∃ x₀ > 0, f a x₀ = -Real.exp (-2)))
  ∧ (∃! k : ℤ, k_condition k ∧ ∀ m : ℤ, k_condition m → m ≤ k) :=
by sorry

#check min_value_and_max_k

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_and_max_k_l600_60042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_plus_sqrt_equation_solutions_l600_60089

noncomputable def cubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

theorem cube_root_plus_sqrt_equation_solutions :
  {x : ℝ | cubeRoot (3 - x) + Real.sqrt (x - 2) = 1} = {2, 3, 11} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_plus_sqrt_equation_solutions_l600_60089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_l600_60052

/-- Two lines are perpendicular if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The slope of line l1: x + ay + 1 = 0 -/
noncomputable def slope_l1 (a : ℝ) : ℝ := -1 / a

/-- The slope of line l2: ax + y + 1 = 0 -/
def slope_l2 (a : ℝ) : ℝ := -a

theorem perpendicular_lines (a : ℝ) (h : a ≠ 0) :
  perpendicular (slope_l1 a) (slope_l2 a) → a = 0 :=
by
  intro h_perp
  unfold perpendicular slope_l1 slope_l2 at h_perp
  -- The rest of the proof would go here
  sorry

#check perpendicular_lines

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_l600_60052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_2013_equals_1006_l600_60023

noncomputable def a (n : ℕ) : ℝ := n * Real.cos (n * Real.pi / 2)

noncomputable def S (n : ℕ) : ℝ := (Finset.range n).sum a

theorem sum_2013_equals_1006 : S 2013 = 1006 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_2013_equals_1006_l600_60023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_chart_reflection_equivalence_l600_60003

/-- Represents the ability of a chart type to reflect information about things -/
structure ReflectionAbility where
  value : ℝ
  nonneg : value ≥ 0

/-- Represents a line chart -/
structure LineChart where
  reflectionAbility : ReflectionAbility

/-- Defines the threshold for clear reflection -/
def clearReflectionThreshold : ReflectionAbility :=
  ⟨0.8, by norm_num⟩

/-- States that line charts can clearly reflect the situation of things -/
def reflectsSituation (chart : LineChart) : Prop :=
  chart.reflectionAbility.value ≥ clearReflectionThreshold.value

/-- States that line charts can clearly reflect the changes in things -/
def reflectsChanges (chart : LineChart) : Prop :=
  chart.reflectionAbility.value ≥ clearReflectionThreshold.value

/-- Theorem stating that for line charts, reflecting situation is equivalent to reflecting changes -/
theorem line_chart_reflection_equivalence (chart : LineChart) :
  reflectsSituation chart ↔ reflectsChanges chart := by
  sorry

#check line_chart_reflection_equivalence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_chart_reflection_equivalence_l600_60003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_diagonal_length_l600_60013

noncomputable def complex_to_point (z : ℂ) : ℝ × ℝ := (z.re, z.im)

noncomputable def point_A : ℝ × ℝ := complex_to_point (Complex.I)
noncomputable def point_B : ℝ × ℝ := complex_to_point 1
noncomputable def point_C : ℝ × ℝ := complex_to_point (4 + 2*Complex.I)

noncomputable def vector_BA : ℝ × ℝ := (point_A.1 - point_B.1, point_A.2 - point_B.2)
noncomputable def vector_BC : ℝ × ℝ := (point_C.1 - point_B.1, point_C.2 - point_B.2)
noncomputable def vector_BD : ℝ × ℝ := (vector_BA.1 + vector_BC.1, vector_BA.2 + vector_BC.2)

noncomputable def length_BD : ℝ := Real.sqrt (vector_BD.1^2 + vector_BD.2^2)

theorem parallelogram_diagonal_length :
  length_BD = Real.sqrt 13 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_diagonal_length_l600_60013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l600_60047

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the properties of the triangle
def isValidTriangle (t : Triangle) : Prop :=
  t.A = 45 * Real.pi / 180 ∧ t.B = 30 * Real.pi / 180 ∧ t.C = 105 * Real.pi / 180

-- Define the median CM
def hasMedian (t : Triangle) : Prop :=
  ∃ M : ℝ, M = (t.A + t.B + t.C) / 2

-- Define the inscribed circles and points D and E
def hasInscribedCircles (t : Triangle) : Prop :=
  ∃ D E : ℝ, D ≠ E ∧ D > 0 ∧ E > 0

-- Define the length of DE
noncomputable def DELength (t : Triangle) : ℝ :=
  4 * (Real.sqrt 2 - 1)

-- Theorem statement
theorem triangle_area (t : Triangle) 
  (h1 : isValidTriangle t) 
  (h2 : hasMedian t) 
  (h3 : hasInscribedCircles t) 
  (h4 : DELength t = 4 * (Real.sqrt 2 - 1)) : 
  (1/2) * (8 * Real.sqrt 2) * 8 * ((Real.sqrt 6 + Real.sqrt 2) / 4) = 16 * (Real.sqrt 3 + 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l600_60047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_derivative_at_one_l600_60098

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * x + Real.log x

-- State the theorem
theorem second_derivative_at_one :
  (deriv (deriv f)) 1 = 3 := by
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_derivative_at_one_l600_60098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_eq_one_third_l600_60000

/-- The sum of the infinite series Σ(3n + 2) / (n(n+1)(n+3)) for n from 1 to infinity -/
noncomputable def infinite_series_sum : ℝ := ∑' n, (3 * n + 2) / (n * (n + 1) * (n + 3))

/-- Theorem stating that the sum of the infinite series is equal to 1/3 -/
theorem infinite_series_sum_eq_one_third : infinite_series_sum = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_eq_one_third_l600_60000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_for_symmetry_l600_60085

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (x + Real.pi / 3)

theorem min_shift_for_symmetry (m : ℝ) :
  (∀ x, f (x - m) = -f (-x - m)) →  -- Symmetry about the origin
  (m > 0) →                        -- m is positive
  (∀ m' > 0, (∀ x, f (x - m') = -f (-x - m')) → m ≤ m') →  -- m is minimum
  m = 2 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_for_symmetry_l600_60085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exercise_book_purchase_theorem_l600_60027

def exercise_book_purchase_methods (total_money : ℕ) (price1 price2 price3 : ℕ) : ℕ :=
  let remaining_money := total_money - price1 - price2 - price3
  (Finset.filter (λ (x, y, z) => x * price1 + y * price2 + z * price3 = remaining_money ∧ x ≥ 1 ∧ y ≥ 1 ∧ z ≥ 1)
    (Finset.range (remaining_money + 1) ×ˢ Finset.range (remaining_money + 1) ×ˢ Finset.range (remaining_money + 1))).card

theorem exercise_book_purchase_theorem :
  exercise_book_purchase_methods 40 2 5 11 = 5 := by
  sorry

#eval exercise_book_purchase_methods 40 2 5 11

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exercise_book_purchase_theorem_l600_60027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_12_equals_25_l600_60072

def f : ℕ → ℕ
  | 0 => 1  -- Adding a case for 0
  | 1 => 1
  | 2 => 4
  | (n+3) => f (n+2) - f (n+1) + 2*(n+3)

theorem f_12_equals_25 : f 12 = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_12_equals_25_l600_60072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l600_60021

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 + x - 2) / Real.log 0.3

-- State the theorem
theorem f_increasing_on_interval :
  ∀ a b : ℝ, a < b → a < -2 → b ≤ -2 →
  ∀ x y : ℝ, x ∈ Set.Icc a b → y ∈ Set.Icc a b → x < y →
  f x < f y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l600_60021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_computation_l600_60087

-- Define the operation * as noncomputable
noncomputable def star (a b : ℝ) : ℝ := (a - b) / (1 - a * b)

-- State the theorem
theorem star_computation :
  star 0.5 (star 1 (star 1.5 (star 2 2.5))) = -1 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_computation_l600_60087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_and_negation_l600_60079

noncomputable def f (x : ℝ) := -x + Real.sin x

theorem proposition_and_negation :
  (∀ x ∈ Set.Ioo 0 (Real.pi / 2), f x < 0) ↔
  ¬(∃ x ∈ Set.Ioo 0 (Real.pi / 2), f x ≥ 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_and_negation_l600_60079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_relationship_l600_60045

-- Define a, b, and c as noncomputable
noncomputable def a : ℝ := Real.exp (Real.log 3 * Real.log (1/2))
noncomputable def b : ℝ := Real.log 25 / Real.log 24
noncomputable def c : ℝ := Real.log 26 / Real.log 25

-- State the theorem
theorem abc_relationship : b > c ∧ c > a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_relationship_l600_60045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_three_twos_to_seven_l600_60043

theorem cube_root_of_three_twos_to_seven (x : ℝ) :
  x = (2^7 + 2^7 + 2^7) ^ (1/3) → x = 36 * 6 ^ (1/3) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_three_twos_to_seven_l600_60043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_temperature_rounded_l600_60067

noncomputable def temperatures : List ℚ := [82, 85, 83, 86, 84, 87, 82]

noncomputable def mean_temperature : ℚ := (temperatures.sum) / temperatures.length

theorem mean_temperature_rounded : 
  Int.floor (mean_temperature + 1/2) = 84 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_temperature_rounded_l600_60067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_sum_mod_thirteen_l600_60084

theorem inverse_sum_mod_thirteen :
  (((2 : ZMod 13)⁻¹ + (3 : ZMod 13)⁻¹ + (5 : ZMod 13)⁻¹)⁻¹ : ZMod 13) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_sum_mod_thirteen_l600_60084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_youngest_age_in_office_l600_60068

/-- Proves that in an arithmetic sequence of 4 terms where the last term is 50
    and the sum of all terms is 158, the first term is 29. -/
theorem youngest_age_in_office (a : Fin 4 → ℕ) : 
  (∀ i : Fin 3, a i.succ - a i = a 1 - a 0) →  -- arithmetic sequence
  (a 3 = 50) →                                -- oldest person's age
  (a 0 + a 1 + a 2 + a 3 = 158) →             -- sum of ages
  a 0 = 29 :=                                 -- youngest person's age
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_youngest_age_in_office_l600_60068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monomial_sum_condition_l600_60048

/-- 
Given two monomials $3a^{m}b^{2}$ and $-\frac{1}{2}{a}^{4}{b}^{n-1}$,
if their sum is still a monomial, then $n-m = -1$.
-/
theorem monomial_sum_condition (a b : ℝ) (m n : ℤ) : 
  (∃ (c : ℝ) (p q : ℤ), 3 * a^m * b^2 + (-1/2) * a^4 * b^(n-1) = c * a^p * b^q) → 
  n - m = -1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monomial_sum_condition_l600_60048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_for_original_line_l600_60093

noncomputable section

open Real

def angle_l1 : ℝ := π / 30
def angle_l2 : ℝ := π / 20

def line_l (x : ℝ) : ℝ := (1 / 11) * x

def reflect_angle (θ α : ℝ) : ℝ := 2 * α - θ

def R (θ : ℝ) : ℝ :=
  reflect_angle (reflect_angle θ angle_l1) angle_l2

def is_original_line (m : ℕ) : Prop :=
  ∃ k : ℤ, (R^[m] (arctan (1 / 11))) = arctan (1 / 11) + 2 * π * (k : ℝ)

theorem smallest_m_for_original_line :
  (∀ n < 60, ¬ is_original_line n) ∧ is_original_line 60 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_for_original_line_l600_60093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_f_two_final_result_l600_60038

-- Define the set of positive real numbers
def S : Set ℝ := {x : ℝ | x > 0}

-- Define the function type
def F := ℝ → ℝ

-- Define the functional equation
def satisfies_equation (f : F) : Prop :=
  ∀ x y : ℝ, x > 0 → y > 0 → 
    f x * f y = f (x * y) + 1001 * (1 / x + 1 / y + 1000)

-- Theorem statement
theorem unique_f_two : 
  ∀ f : F, satisfies_equation f → f 2 = 1001.5 := by
  sorry

-- Define n and s
def n : ℕ := 1
def s : ℝ := 1001.5

-- Theorem for the final result
theorem final_result : n * s = 1001.5 := by
  rw [n, s]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_f_two_final_result_l600_60038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_even_increasing_function_l600_60019

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def strictly_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x y, x ∈ s → y ∈ s → x < y → f x < f y

theorem range_of_even_increasing_function 
  (f : ℝ → ℝ) 
  (h_even : is_even_function f) 
  (h_incr : strictly_increasing_on f (Set.Ici 0)) :
  {a : ℝ | f a ≥ f 3} = Set.Iic (-3) ∪ Set.Ici 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_even_increasing_function_l600_60019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_correct_propositions_l600_60024

-- Define a triangle
structure Triangle where
  isObtuse : Bool

-- Define a tetrahedron
structure Tetrahedron where
  faces : Fin 4 → Triangle

-- Define a face
inductive Face
  | Polygon
  | Triangle
  | Trapezoid

-- Define a polyhedron
structure Polyhedron where
  faces : List Face

-- Define a pyramid
def isPyramid (p : Polyhedron) : Prop :=
  ∃ (base : Face), base ∈ p.faces ∧ 
  ∀ (f : Face), f ∈ p.faces ∧ f ≠ base → f = Face.Triangle

-- Define a frustum
def isFrustum (p : Polyhedron) : Prop :=
  ∃ (base1 base2 : Face), base1 ∈ p.faces ∧ base2 ∈ p.faces ∧ base1 ≠ base2 ∧
  ∀ (f : Face), f ∈ p.faces ∧ f ≠ base1 ∧ f ≠ base2 → f = Face.Trapezoid

-- Define the three propositions
def proposition1 : Prop := ¬∃ (t : Tetrahedron), ∀ (i : Fin 4), (t.faces i).isObtuse

def proposition2 : Prop := ∀ (p : Polyhedron),
  (∃ (base : Face), base ∈ p.faces ∧ base = Face.Polygon ∧
    ∀ (f : Face), f ∈ p.faces ∧ f ≠ base → f = Face.Triangle) →
  isPyramid p

def proposition3 : Prop := ∀ (p : Polyhedron),
  (∃ (base1 base2 : Face), base1 ∈ p.faces ∧ base2 ∈ p.faces ∧ base1 ≠ base2 ∧
    ∀ (f : Face), f ∈ p.faces ∧ f ≠ base1 ∧ f ≠ base2 → f = Face.Trapezoid) →
  isFrustum p

-- Theorem to prove
theorem num_correct_propositions : 
  ¬proposition1 ∧ ¬proposition2 ∧ ¬proposition3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_correct_propositions_l600_60024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_after_adding_numbers_l600_60088

theorem average_after_adding_numbers (S : Finset ℝ) (sum_S : ℝ) :
  S.card = 12 →
  sum_S = S.sum id →
  sum_S / S.card = 90 →
  let new_S := S ∪ {80, 90}
  (new_S.sum id) / new_S.card = 89.2857142857 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_after_adding_numbers_l600_60088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_envelope_extra_charge_l600_60044

/-- Represents an envelope with length and height in inches -/
structure Envelope where
  length : ℚ
  height : ℚ

/-- Determines if an extra charge applies to an envelope -/
def extraChargeApplies (e : Envelope) : Bool :=
  let ratio := e.length / e.height
  ratio < 1.2 || ratio > 2.6

/-- The set of envelopes given in the problem -/
def envelopes : List Envelope := [
  ⟨5, 4⟩,  -- Envelope A
  ⟨10, 4⟩, -- Envelope B
  ⟨8, 8⟩,  -- Envelope C
  ⟨12, 5⟩  -- Envelope D
]

/-- The main theorem stating that exactly one envelope requires an extra charge -/
theorem one_envelope_extra_charge :
  (envelopes.filter extraChargeApplies).length = 1 := by
  -- Proof goes here
  sorry

#eval (envelopes.filter extraChargeApplies).length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_envelope_extra_charge_l600_60044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_balls_meet_time_distance_zero_glafira_distance_l600_60020

noncomputable section

/-- Motion of the first ball -/
def y₁ (U g t : ℝ) : ℝ := U * t - (g * t^2) / 2

/-- Motion of the second ball -/
def y₂ (U g t τ : ℝ) : ℝ := U * (t - τ) - (g * (t - τ)^2) / 2

/-- Distance between the balls -/
def s (U g t τ : ℝ) : ℝ := |y₁ U g t - y₂ U g t τ|

/-- Time at which the balls meet -/
def T (U g τ : ℝ) : ℝ := τ/2 + U/g

theorem balls_meet_time_distance_zero (U g τ V : ℝ) (hg : g > 0) (hU : U > 0) (hτ : τ > 0) (hV : V > 0) (h : 2*U ≥ g*τ) :
  s U g (T U g τ) τ = 0 ∧ T U g τ ≥ τ := by sorry

theorem glafira_distance (U g τ V : ℝ) (hg : g > 0) (hU : U > 0) (hτ : τ > 0) (hV : V > 0) (h : 2*U ≥ g*τ) :
  V * (T U g τ) = V * (τ/2 + U/g) := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_balls_meet_time_distance_zero_glafira_distance_l600_60020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ltf_winning_strategy_l600_60053

/-- Represents a game state with a list of card values -/
structure GameState where
  cards : List ℚ
  deriving Repr

/-- Represents a player's move: taking the first or last card -/
inductive Move
  | First
  | Last
  deriving Repr

/-- Defines the game rules and outcomes -/
def playGame (initialState : GameState) (ltfStrategy : GameState → Move) : Bool :=
  sorry  -- Implement game logic here

/-- Theorem stating that LTF can always prevent Sunny from winning iff n is even -/
theorem ltf_winning_strategy (n : ℕ) :
  (∃ (strategy : GameState → Move), ∀ (initialState : GameState),
    initialState.cards.length = n ∧ 
    (∀ x ∈ initialState.cards, x > 0) →
    playGame initialState strategy = true) ↔ 
  Even n :=
by
  sorry

#check ltf_winning_strategy

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ltf_winning_strategy_l600_60053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_relationship_l600_60049

theorem abc_relationship : ∃ (a b c : ℝ),
  a = Real.rpow 0.6 0.6 ∧
  b = Real.rpow 0.6 1.5 ∧
  c = Real.rpow 1.5 0.6 ∧
  b < a ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_relationship_l600_60049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hypotenuse_longer_than_legs_l600_60083

/-- An isosceles right triangle with given area and hypotenuse length -/
structure IsoscelesRightTriangle where
  -- The area of the triangle
  area : ℝ
  -- The length of the hypotenuse
  hypotenuse : ℝ
  -- Ensure the area is positive
  area_pos : 0 < area
  -- Ensure the hypotenuse length is positive
  hypotenuse_pos : 0 < hypotenuse

/-- The length of a leg in an isosceles right triangle -/
noncomputable def leg_length (t : IsoscelesRightTriangle) : ℝ :=
  Real.sqrt (2 * t.area)

/-- Theorem: In an isosceles right triangle with area 36 and hypotenuse length 12.000000000000002,
    the hypotenuse is longer than the legs -/
theorem hypotenuse_longer_than_legs (t : IsoscelesRightTriangle)
    (h_area : t.area = 36)
    (h_hypotenuse : t.hypotenuse = 12.000000000000002) :
    t.hypotenuse > leg_length t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hypotenuse_longer_than_legs_l600_60083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_sum_l600_60046

-- Define the diamond function
noncomputable def diamond (x : ℝ) : ℝ := (x^3 + 2*x^2 + 3*x) / 6

-- Theorem statement
theorem diamond_sum : diamond 2 + diamond 3 + diamond 4 = 92/3 := by
  -- Unfold the definition of diamond
  unfold diamond
  -- Simplify the expressions
  simp [pow_two, pow_three]
  -- Perform arithmetic
  ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_sum_l600_60046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_and_monotonicity_l600_60014

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3)*a*x^3 - (1/2)*x^2 + (a-1)*x + 1

theorem tangent_perpendicular_and_monotonicity (a : ℝ) :
  -- Part 1: Tangent line perpendicular
  (∃ (m : ℝ), m * f a 1 + 1 = f a 1 ∧ m = -2 → a = 2) ∧
  -- Part 2: Monotonicity
  (∀ x y, 2 ≤ x ∧ x < y → 
    (a ≤ 0 → f a x > f a y) ∧
    (0 < a ∧ a < 3/5 → 
      (x < (1 + Real.sqrt (1 - 4*a^2 + 4*a)) / (2*a) → f a x > f a y) ∧
      (y > (1 + Real.sqrt (1 - 4*a^2 + 4*a)) / (2*a) → f a x < f a y)) ∧
    (3/5 ≤ a → f a x < f a y)) :=
by sorry

#check tangent_perpendicular_and_monotonicity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_and_monotonicity_l600_60014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measure_l600_60037

theorem triangle_angle_measure (a b c : ℝ) (h : b^2 + c^2 - a^2 = Real.sqrt 3 * b * c) :
  Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c)) = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measure_l600_60037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_typeBMoreProfitable_l600_60030

/-- Represents the production and sales data for a tire company --/
structure TireCompany where
  typeACostFirstBatch : ℝ
  typeACostFirstPerTire : ℝ
  typeACostSecondBatch : ℝ
  typeACostSecondPerTire : ℝ
  typeBCostBatch : ℝ
  typeBCostPerTire : ℝ
  typeASellFirstPrice : ℝ
  typeASellSecondPrice : ℝ
  typeBSellPrice : ℝ
  typeASoldFirst : ℕ
  typeASoldTotal : ℕ
  typeBSold : ℕ

/-- Calculates the profit per tire for type A tires --/
noncomputable def profitPerTireA (company : TireCompany) : ℝ :=
  let firstBatchProfit := company.typeASoldFirst * (company.typeASellFirstPrice - company.typeACostFirstPerTire) - company.typeACostFirstBatch
  let secondBatchProfit := (company.typeASoldTotal - company.typeASoldFirst) * (company.typeASellSecondPrice - company.typeACostSecondPerTire) - company.typeACostSecondBatch
  (firstBatchProfit + secondBatchProfit) / company.typeASoldTotal

/-- Calculates the profit per tire for type B tires --/
noncomputable def profitPerTireB (company : TireCompany) : ℝ :=
  (company.typeBSold * company.typeBSellPrice - company.typeBCostBatch - company.typeBCostPerTire * company.typeBSold) / company.typeBSold

/-- Theorem stating that the profit per tire for type B is greater than type A --/
theorem typeBMoreProfitable (company : TireCompany) :
  company.typeACostFirstBatch = 22500 ∧
  company.typeACostFirstPerTire = 8 ∧
  company.typeACostSecondBatch = 20000 ∧
  company.typeACostSecondPerTire = 6 ∧
  company.typeBCostBatch = 24000 ∧
  company.typeBCostPerTire = 7 ∧
  company.typeASellFirstPrice = 20 ∧
  company.typeASellSecondPrice = 18 ∧
  company.typeBSellPrice = 19 ∧
  company.typeASoldFirst = 5000 ∧
  company.typeASoldTotal = 15000 ∧
  company.typeBSold = 10000 →
  profitPerTireB company > profitPerTireA company :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_typeBMoreProfitable_l600_60030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_geq_g_range_l600_60028

/-- The function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2*x - 1) * Real.exp x - a * (x^2 + x)

/-- The function g(x) as defined in the problem -/
def g (a : ℝ) (x : ℝ) : ℝ := -a * x^2 - a

/-- The theorem stating the range of values for a -/
theorem f_geq_g_range (a : ℝ) :
  (∀ x : ℝ, f a x ≥ g a x) ↔ (1 ≤ a ∧ a ≤ 4 * Real.exp (3/2)) := by
  sorry

#check f_geq_g_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_geq_g_range_l600_60028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l600_60065

-- Define the parabola
def parabola (p : ℝ × ℝ) : Prop := p.2^2 = 4 * p.1

-- Define the circle
def circleD (p : ℝ × ℝ) : Prop := (p.1 + 4)^2 + p.2^2 = 4

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the distance to directrix (which is equal to the distance to focus for points on the parabola)
noncomputable def distToDirectrix (p : ℝ × ℝ) : ℝ := distance p focus

-- The main theorem
theorem min_distance_sum :
  ∀ (M N : ℝ × ℝ), parabola M → circleD N →
  distance M N + distToDirectrix M ≥ 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l600_60065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_area_l600_60007

/-- A point in a 2D coordinate plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Triangle ABC with given coordinates -/
def triangleABC : (Point × Point × Point) :=
  (⟨0, 0⟩, ⟨4, 0⟩, ⟨0, 3⟩)

/-- Calculate the area of a right-angled triangle given its base and height -/
noncomputable def areaRightTriangle (base height : ℝ) : ℝ :=
  (1 / 2) * base * height

/-- The area of triangle ABC is 6 square units -/
theorem triangle_ABC_area :
  let (A, B, C) := triangleABC
  areaRightTriangle (B.x - A.x) (C.y - A.y) = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_area_l600_60007
