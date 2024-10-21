import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_minus_one_integral_l805_80537

open MeasureTheory Interval Set Real

theorem absolute_value_minus_one_integral : 
  ∫ x in (-1)..1, (|x| - 1) = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_minus_one_integral_l805_80537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_workers_completion_time_l805_80523

/-- The time taken for three workers to complete a job together, given their individual completion times -/
noncomputable def combined_work_time (t₁ t₂ t₃ : ℝ) : ℝ :=
  1 / (1/t₁ + 1/t₂ + 1/t₃)

theorem workers_completion_time :
  combined_work_time 12 15 20 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_workers_completion_time_l805_80523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l805_80515

-- Define the curve C
noncomputable def curve_C (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 6*y + 1 = 0

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ :=
  (3 + (1/2)*t, 3 + (Real.sqrt 3 / 2)*t)

-- Define point P
def point_P : ℝ × ℝ := (3, 3)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem intersection_distance_sum :
  ∃ (t1 t2 : ℝ),
    curve_C (line_l t1).1 (line_l t1).2 ∧
    curve_C (line_l t2).1 (line_l t2).2 ∧
    t1 ≠ t2 ∧
    distance point_P (line_l t1) + distance point_P (line_l t2) = 2 * Real.sqrt 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l805_80515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_side_parabola1_same_side_parabola2_same_side_parabola3_l805_80597

/-- Determine if two points are on the same side of a parabola --/
def same_side_parabola (a b c : ℝ) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  let f := λ x => a * x^2 + b * x + c
  (f x₁ - y₁) * (f x₂ - y₂) > 0

/-- The points A(-1,-1) and B(0,2) lie on the same side of the parabola y = 2x^2 + 4x --/
theorem same_side_parabola1 : same_side_parabola 2 4 0 (-1) (-1) 0 2 := by
  sorry

/-- The points A(-1,-1) and B(0,2) lie on the same side of the parabola y = -x^2 + 2x - 1 --/
theorem same_side_parabola2 : same_side_parabola (-1) 2 (-1) (-1) (-1) 0 2 := by
  sorry

/-- The points A(-1,-1) and B(0,2) lie on the same side of the parabola y = -x^2 + 3 --/
theorem same_side_parabola3 : same_side_parabola (-1) 0 3 (-1) (-1) 0 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_side_parabola1_same_side_parabola2_same_side_parabola3_l805_80597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l805_80532

/-- The function f(x) = (4x + a) / (x^2 + 1) is an odd function -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (4 * x + a) / (x^2 + 1)

theorem odd_function_properties (a : ℝ) (h : ∀ x, f a x = -f a (-x)) :
  /- a = 0 -/
  a = 0 ∧
  /- f(x) is monotonically decreasing on [1, +∞) for x > 0 -/
  (∀ x y, 1 ≤ x ∧ x < y → f 0 x > f 0 y) ∧
  /- The minimum value of m such that |f(x₁) - f(x₂)| ≤ m for all x₁, x₂ ∈ ℝ is 4 -/
  (∀ m, (∀ x₁ x₂, |f 0 x₁ - f 0 x₂| ≤ m) ↔ 4 ≤ m) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l805_80532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_age_of_union_l805_80546

-- Define the groups and their properties
structure MyGroup where
  size : ℕ
  total_age : ℕ

-- Define the average age function
def avg_age (g : MyGroup) : ℚ := g.total_age / g.size

-- State the theorem
theorem average_age_of_union
  (A B C D : MyGroup)
  (h_disjoint : A.size + B.size + C.size + D.size > 0)
  (h_A : avg_age A = 40)
  (h_B : avg_age B = 30)
  (h_C : avg_age C = 45)
  (h_D : avg_age D = 35)
  (h_AB : avg_age ⟨A.size + B.size, A.total_age + B.total_age⟩ = 37)
  (h_AC : avg_age ⟨A.size + C.size, A.total_age + C.total_age⟩ = 42)
  (h_AD : avg_age ⟨A.size + D.size, A.total_age + D.total_age⟩ = 39)
  (h_BC : avg_age ⟨B.size + C.size, B.total_age + C.total_age⟩ = 40)
  (h_BD : avg_age ⟨B.size + D.size, B.total_age + D.total_age⟩ = 37)
  (h_CD : avg_age ⟨C.size + D.size, C.total_age + D.total_age⟩ = 43)
  : avg_age ⟨A.size + B.size + C.size + D.size,
             A.total_age + B.total_age + C.total_age + D.total_age⟩ = 89/2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_age_of_union_l805_80546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l805_80595

noncomputable def f (x : ℝ) : ℝ := Real.cos (x / 2 + Real.pi / 5)

theorem f_properties :
  (∀ x, f (16 * Real.pi / 5 - x) = f x) ∧
  (∀ x, f ((x + 3 * Real.pi / 5) + 3 * Real.pi / 5) = -f (x + 3 * Real.pi / 5)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l805_80595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_octagon_area_theorem_l805_80529

/-- The ratio of the area of a circle inscribed in a regular octagon 
    (touching the midpoints of the octagon's sides) to the area of the octagon -/
noncomputable def circle_octagon_area_ratio : ℝ := (Real.sqrt 2) / 2 * Real.pi

/-- The numerator under the square root in the area ratio expression -/
def a : ℕ := 2

/-- The denominator in the area ratio expression -/
def b : ℕ := 2

theorem circle_octagon_area_theorem :
  circle_octagon_area_ratio = (Real.sqrt a) / b * Real.pi ∧ a * b = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_octagon_area_theorem_l805_80529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l805_80591

/-- Parabola with focus on x-axis passing through (2, 2√2) -/
def parabola_C (x y : ℝ) : Prop := y^2 = 4*x

/-- Point on parabola -/
axiom point_on_parabola : parabola_C 2 (2*Real.sqrt 2)

/-- Focus of parabola -/
def focus : ℝ × ℝ := (1, 0)

/-- Line passing through focus -/
def line_through_focus (k : ℝ) (x y : ℝ) : Prop := x = k*y + 1

/-- Intersection points of line and parabola -/
def intersection_points (k : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ x y, p = (x, y) ∧ parabola_C x y ∧ line_through_focus k x y}

/-- Area of triangle OAB -/
noncomputable def triangle_area (k : ℝ) : ℝ := 2 * Real.sqrt (k^2 + 1)

/-- Theorem: Minimum area of triangle OAB is 2 -/
theorem min_triangle_area :
  ∃ min_area : ℝ, min_area = 2 ∧ ∀ k : ℝ, triangle_area k ≥ min_area := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l805_80591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_quadratic_inequality_l805_80509

theorem min_value_quadratic_inequality (a : ℝ) (x₁ x₂ : ℝ) 
  (ha : a > 0)
  (hx : ∀ x, x^2 - 4*a*x + 3*a^2 < 0 ↔ (x = x₁ ∨ x = x₂)) :
  ∃ (min : ℝ), min = (4 * Real.sqrt 3) / 3 ∧
    ∀ y, y = x₁ + x₂ + a / (x₁ * x₂) → y ≥ min :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_quadratic_inequality_l805_80509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_value_l805_80584

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.cos (Real.pi * x) else 2 / x

theorem f_composition_value : f (f (4/3)) = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_value_l805_80584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l805_80550

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (3 * x + 8) / (x - 4)

-- Define the range of f
def range_f : Set ℝ := {y | ∃ x, f x = y ∧ x ≠ 4}

-- Theorem statement
theorem range_of_f : range_f = Set.Iio 3 ∪ Set.Ioi 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l805_80550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_properties_l805_80552

-- Define the set S of all non-zero real numbers
def S : Set ℝ := {x : ℝ | x ≠ 0}

-- Define the binary operation *
def star (a b : ℝ) : ℝ := a * b + 1

-- Theorem statement
theorem star_properties :
  -- 1. * is commutative over S
  (∀ (a b : ℝ), a ∈ S → b ∈ S → star a b = star b a) ∧
  -- 2. * is not associative over S
  (∃ (a b c : ℝ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ star a (star b c) ≠ star (star a b) c) ∧
  -- 3. 1 is not an identity element for * in S
  (∃ (a : ℝ), a ∈ S ∧ (star a 1 ≠ a ∨ star 1 a ≠ a)) ∧
  -- 4. Not every element of S has an inverse for *
  (∃ (a : ℝ), a ∈ S ∧ ∀ (b : ℝ), b ∈ S → star a b ≠ 1) ∧
  -- 5. 1/(a-1) is not an inverse for * of the element a in S
  (∃ (a : ℝ), a ∈ S ∧ a ≠ 1 ∧ star a (1 / (a - 1)) ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_properties_l805_80552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_point_on_line_with_sum_of_distances_l805_80560

-- Define the necessary types and functions
variable (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define a line in the vector space
def Line (p q : V) : Set V := {x : V | ∃ t : ℝ, x = p + t • (q - p)}

-- Define the distance between two points
def distance (p q : V) : ℝ := ‖p - q‖

-- State the theorem
theorem exists_point_on_line_with_sum_of_distances
  (e : Set V)
  (A P : V)
  (l : ℝ)
  (h1 : A ∈ e)
  (h2 : P ∉ e)
  (h3 : l > 0)
  (h4 : ∃ p q : V, e = Line V p q) :
  ∃ X : V, X ∈ e ∧ distance V A X + distance V X P = l :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_point_on_line_with_sum_of_distances_l805_80560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_side_length_of_triangle_ABO_l805_80581

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The parabola y = -x² -/
def onParabola (p : Point) : Prop :=
  p.y = -p.x^2

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The origin point (0, 0) -/
def origin : Point :=
  ⟨0, 0⟩

/-- Triangle ABO is isosceles right with AB as hypotenuse -/
def isIsoscelesRightTriangle (a b : Point) : Prop :=
  distance a origin = distance b origin ∧
  distance a origin^2 + distance b origin^2 = distance a b^2

theorem side_length_of_triangle_ABO :
  ∀ a b : Point,
  onParabola a →
  onParabola b →
  isIsoscelesRightTriangle a b →
  distance a origin = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_side_length_of_triangle_ABO_l805_80581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_ab_equals_2016_l805_80520

-- Define the function f
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (63 * Real.exp x) / a - b / (32 * Real.exp x)

-- Define the property of being an odd function
def is_odd_function (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

-- Theorem statement
theorem odd_function_implies_ab_equals_2016 (a b : ℝ) (h : is_odd_function (f a b)) :
  a * b = 2016 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_ab_equals_2016_l805_80520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_trinomials_with_common_non_integer_root_are_identical_l805_80573

-- Define a quadratic trinomial with integer coefficients
structure QuadraticTrinomial where
  b : Int
  c : Int

-- Define a function to represent the trinomial as a function of x
def evaluate (q : QuadraticTrinomial) (x : ℝ) : ℝ :=
  x^2 + q.b * x + q.c

-- Define what it means for a real number to be a root of a quadratic trinomial
def is_root (q : QuadraticTrinomial) (r : ℝ) : Prop :=
  evaluate q r = 0

-- State the theorem
theorem quadratic_trinomials_with_common_non_integer_root_are_identical
  (q1 q2 : QuadraticTrinomial) (α : ℝ) :
  (¬ ∃ (n : ℤ), α = n) →  -- α is not an integer
  (is_root q1 α) →  -- α is a root of q1
  (is_root q2 α) →  -- α is a root of q2
  q1 = q2  -- The trinomials are identical
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_trinomials_with_common_non_integer_root_are_identical_l805_80573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cot_thirty_degrees_l805_80525

theorem cot_thirty_degrees : Real.tan (π / 6) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cot_thirty_degrees_l805_80525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_implies_m_range_l805_80504

/-- The function f(x) = 4^x - m * 2^(x+1) -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (4 : ℝ)^x - m * (2 : ℝ)^(x+1)

/-- Theorem stating that if there exists x₀ such that f(-x₀) = -f(x₀), then m ≥ 1/2 -/
theorem f_symmetry_implies_m_range (m : ℝ) :
  (∃ x₀ : ℝ, f m (-x₀) = -(f m x₀)) → m ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_implies_m_range_l805_80504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_phi_forms_cone_l805_80554

-- Define spherical coordinates
structure SphericalCoord where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

-- Define a constant c
variable (c : ℝ)

-- Define the set of points satisfying φ = c
def ConstantPhiSet (c : ℝ) : Set SphericalCoord :=
  {p : SphericalCoord | p.φ = c}

-- Define a cone (simplified definition for this problem)
def Cone (c : ℝ) : Set SphericalCoord :=
  {p : SphericalCoord | p.φ = c}

-- Theorem statement
theorem constant_phi_forms_cone (c : ℝ) : ConstantPhiSet c = Cone c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_phi_forms_cone_l805_80554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_food_lasts_seventeen_more_days_l805_80562

/-- The number of additional days food lasts after more men join -/
noncomputable def additional_days (initial_men : ℕ) (initial_days : ℕ) (days_before_join : ℕ) (additional_men : ℝ) : ℝ :=
  let total_food := (initial_men : ℝ) * initial_days
  let remaining_food := total_food - ((initial_men : ℝ) * days_before_join)
  let total_men_after := (initial_men : ℝ) + additional_men
  remaining_food / total_men_after

/-- Theorem stating that the food lasts approximately 17 more days after additional men join -/
theorem food_lasts_seventeen_more_days :
  let result := additional_days 760 22 2 134.11764705882354
  ⌊result⌋ = 17 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_food_lasts_seventeen_more_days_l805_80562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_zero_l805_80565

-- Define the angle α
noncomputable def α : Real := Real.arctan (-2/1)

-- Define the point P
def P : (Real × Real) := (1, -2)

-- Theorem statement
theorem expression_equals_zero :
  (Real.sin α + 2 * Real.cos α) / (Real.sin α - Real.cos α) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_zero_l805_80565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_munificence_is_two_l805_80518

/-- The polynomial p(x) = x^2 - 2x - 1 -/
def p (x : ℝ) : ℝ := x^2 - 2*x - 1

/-- The absolute value of p(x) -/
def abs_p (x : ℝ) : ℝ := |p x|

/-- The interval [-1, 1] -/
def I : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}

/-- The munificence of p(x) over the interval I -/
noncomputable def munificence : ℝ := sSup (abs_p '' I)

/-- Theorem: The munificence of p(x) over the interval [-1, 1] is 2 -/
theorem munificence_is_two : munificence = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_munificence_is_two_l805_80518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_travel_solution_l805_80531

/-- Represents the scenario of Xiao Chen and Xiao Wang's travel to their company headquarters -/
structure TravelScenario where
  x : ℝ  -- Average speed of Xiao Chen's taxi in km/h
  y : ℝ  -- Distance from office to headquarters in km

/-- The travel time for Xiao Chen in hours -/
noncomputable def xiao_chen_time (s : TravelScenario) : ℝ := 50 / 60

/-- The travel time for Xiao Wang in hours -/
noncomputable def xiao_wang_time (s : TravelScenario) : ℝ := 40 / 60

/-- The conditions of the problem are satisfied -/
def satisfies_conditions (s : TravelScenario) : Prop :=
  s.x * xiao_chen_time s = s.y ∧
  (s.x - 6) * xiao_wang_time s = s.y - 10

/-- The theorem stating the solution to the problem -/
theorem travel_solution :
  ∃ s : TravelScenario, satisfies_conditions s ∧ s.x = 36 ∧ s.y = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_travel_solution_l805_80531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_derivative_l805_80594

open Real

/-- The volume of a sphere as a function of its radius -/
noncomputable def sphereVolume (R : ℝ) : ℝ := (4 / 3) * π * R^3

/-- The surface area of a sphere as a function of its radius -/
noncomputable def sphereSurfaceArea (R : ℝ) : ℝ := 4 * π * R^2

theorem sphere_volume_derivative (R : ℝ) (h : R > 0) :
  deriv sphereVolume R = sphereSurfaceArea R := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_derivative_l805_80594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_f_min_value_when_n_3_l805_80587

noncomputable def f (n : ℕ+) (x : ℝ) : ℝ := (Real.sin x) ^ (n : ℝ) + (Real.cos x) ^ (n : ℝ)

theorem f_symmetry (n : ℕ+) :
  ∀ x, f n (Real.pi / 2 - x) = f n x := by
  sorry

theorem f_min_value_when_n_3 :
  ∃ x₀ ∈ Set.Icc 0 (Real.pi / 2), f 3 x₀ = Real.sqrt 2 / 2 ∧
  ∀ x ∈ Set.Icc 0 (Real.pi / 2), f 3 x₀ ≤ f 3 x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_f_min_value_when_n_3_l805_80587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_population_net_change_approx_negative_twelve_net_change_exact_l805_80577

/-- Represents the population change over four years -/
noncomputable def population_change (initial : ℝ) : ℝ :=
  initial * (5/4) * (5/4) * (3/4) * (3/4)

/-- The net change in population as a percentage -/
noncomputable def net_change_percentage (initial : ℝ) : ℝ :=
  (population_change initial / initial - 1) * 100

/-- Theorem stating that the net change in population is approximately -12% -/
theorem population_net_change_approx_negative_twelve (initial : ℝ) (h : initial > 0) :
  ∃ ε > 0, |net_change_percentage initial + 12| < ε :=
by
  -- The proof is omitted for now
  sorry

#eval (225 : ℚ) / 256 -- This will compute the exact fraction

/-- Lemma to show the exact calculation of the net change -/
lemma net_change_exact_calculation :
  (225 : ℚ) / 256 - 1 = -31 / 256 :=
by
  -- The proof is straightforward arithmetic
  ring

/-- Theorem to show that the net change is exactly -31/256 -/
theorem net_change_exact (initial : ℝ) (h : initial > 0) :
  net_change_percentage initial = -31 * 100 / 256 :=
by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_population_net_change_approx_negative_twelve_net_change_exact_l805_80577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_form_example_l805_80539

/-- Predicate to check if a rational function is in its simplest form -/
def IsSimplestForm (num : ℝ → ℝ → ℝ) (den : ℝ → ℝ → ℝ) : Prop :=
  ∀ x y, (∀ (f : ℝ → ℝ → ℝ), num x y = f x y * den x y → f = λ _ _ => 1)

/-- The expression (x^2 + y^2) / (x + y) is in its simplest form -/
theorem simplest_form_example :
  IsSimplestForm (λ x y => x^2 + y^2) (λ x y => x + y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_form_example_l805_80539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_m_for_all_p_l805_80593

/-- Two sequences of real numbers -/
noncomputable def a : ℕ → ℝ := sorry
noncomputable def g : ℕ → ℝ := sorry

/-- a is an arithmetic sequence -/
axiom a_arithmetic : ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- g is a geometric sequence -/
axiom g_geometric : ∃ r : ℝ, ∀ n : ℕ, g (n + 1) = g n * r

/-- g is non-constant -/
axiom g_nonconstant : ∃ n m : ℕ, g n ≠ g m

/-- Initial conditions -/
axiom a1_eq_g1 : a 1 = g 1
axiom a1_neq_0 : a 1 ≠ 0
axiom a2_eq_g2 : a 2 = g 2
axiom a10_eq_g3 : a 10 = g 3

/-- Main theorem -/
theorem exists_m_for_all_p : ∀ p : ℕ, p > 0 → ∃ m : ℕ, m > 0 ∧ g p = a m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_m_for_all_p_l805_80593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_in_fourth_quadrant_l805_80556

noncomputable def complex_number : ℂ := (Complex.I * 3 - 2) / (Complex.I - 1) * Complex.I

theorem complex_number_in_fourth_quadrant :
  Real.sign complex_number.re = -1 ∧ Real.sign complex_number.im = 1 :=
by
  -- Simplify the complex number
  have h : complex_number = -5/2 + Complex.I/2 := by
    -- Proof of simplification goes here
    sorry
  
  -- Check the sign of the real part
  have h_re : Real.sign complex_number.re = -1 := by
    rw [h]
    -- Proof that the real part is negative goes here
    sorry
  
  -- Check the sign of the imaginary part
  have h_im : Real.sign complex_number.im = 1 := by
    rw [h]
    -- Proof that the imaginary part is positive goes here
    sorry
  
  -- Combine the results
  exact ⟨h_re, h_im⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_in_fourth_quadrant_l805_80556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_attendance_l805_80534

/-- Represents the days of the week --/
inductive Day
  | Mon
  | Tues
  | Wed
  | Thurs
  | Fri
  | Sat

/-- Represents a team member --/
inductive Member
  | Alice
  | Bob
  | Cara
  | Dave
  | Eve

/-- Represents the availability of a member on a given day --/
def isAvailable (m : Member) (d : Day) : Bool :=
  match m, d with
  | Member.Alice, Day.Mon => false
  | Member.Alice, Day.Tues => true
  | Member.Alice, Day.Wed => true
  | Member.Alice, Day.Thurs => false
  | Member.Alice, Day.Fri => false
  | Member.Alice, Day.Sat => true
  | Member.Bob, Day.Mon => true
  | Member.Bob, Day.Tues => false
  | Member.Bob, Day.Wed => false
  | Member.Bob, Day.Thurs => true
  | Member.Bob, Day.Fri => true
  | Member.Bob, Day.Sat => false
  | Member.Cara, Day.Mon => false
  | Member.Cara, Day.Tues => false
  | Member.Cara, Day.Wed => true
  | Member.Cara, Day.Thurs => true
  | Member.Cara, Day.Fri => true
  | Member.Cara, Day.Sat => false
  | Member.Dave, Day.Mon => true
  | Member.Dave, Day.Tues => true
  | Member.Dave, Day.Wed => true
  | Member.Dave, Day.Thurs => false
  | Member.Dave, Day.Fri => true
  | Member.Dave, Day.Sat => true
  | Member.Eve, Day.Mon => false
  | Member.Eve, Day.Tues => false
  | Member.Eve, Day.Wed => false
  | Member.Eve, Day.Thurs => true
  | Member.Eve, Day.Fri => true
  | Member.Eve, Day.Sat => false

/-- Counts the number of available members for a given day --/
def countAvailable (d : Day) : Nat :=
  (List.filter (fun m => isAvailable m d) [Member.Alice, Member.Bob, Member.Cara, Member.Dave, Member.Eve]).length

/-- Theorem: Wednesday and Friday have the maximum number of attendees --/
theorem max_attendance : 
  (countAvailable Day.Wed = countAvailable Day.Fri) ∧ 
  (∀ d : Day, countAvailable d ≤ countAvailable Day.Wed) := by
  sorry

#eval countAvailable Day.Mon
#eval countAvailable Day.Tues
#eval countAvailable Day.Wed
#eval countAvailable Day.Thurs
#eval countAvailable Day.Fri
#eval countAvailable Day.Sat

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_attendance_l805_80534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_floor_fraction_l805_80538

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

theorem sum_floor_fraction (n : ℕ+) :
  (∑' k : ℕ, floor ((n : ℝ) + 2^k / 2^(k+1))) = n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_floor_fraction_l805_80538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_squared_difference_equals_sqrt3_over_2_l805_80527

theorem cos_squared_difference_equals_sqrt3_over_2 :
  (Real.cos (π / 12))^2 - (Real.cos (5 * π / 12))^2 = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_squared_difference_equals_sqrt3_over_2_l805_80527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_value_l805_80516

/-- The function f(x) = x ln x -/
noncomputable def f (x : ℝ) := x * Real.log x

/-- The line l(x) = (k-2)x - k + 1 -/
def l (k : ℤ) (x : ℝ) := (k - 2 : ℝ) * x - k + 1

/-- The statement that f(x) is always above l(x) for x > 1 -/
def above_line (k : ℤ) : Prop := ∀ x > 1, f x > l k x

/-- The maximum value of k for which f(x) is always above l(x) when x > 1 -/
theorem max_k_value : ∀ k : ℤ, above_line k → k ≤ 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_value_l805_80516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l805_80507

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 + Real.log x / Real.log 10 + 9 * Real.log 10 / Real.log x

-- State the theorem
theorem max_value_of_f :
  ∀ x : ℝ, 0 < x → x < 1 → f x ≤ -5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l805_80507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_value_proof_l805_80503

/-- The initial market value of a machine that depreciates by 25% annually --/
noncomputable def initial_market_value : ℝ :=
  4000 / (0.75 ^ 2)

/-- The market value after two years of 25% annual depreciation --/
noncomputable def market_value_after_two_years (initial_value : ℝ) : ℝ :=
  initial_value * (0.75 ^ 2)

theorem machine_value_proof :
  ∀ ε > 0, |market_value_after_two_years initial_market_value - 4000| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_value_proof_l805_80503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_problem_l805_80514

theorem angle_problem (α β : Real) 
  (h1 : 0 < α ∧ α < π/2)
  (h2 : π/2 < β ∧ β < π)
  (h3 : Real.cos β = -1/3)
  (h4 : Real.sin (α + β) = 7/9) :
  Real.tan (β/2) = Real.sqrt 2 ∧ Real.sin α = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_problem_l805_80514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_selfie_ratio_l805_80592

/-- Given the total number of photos and the difference between this year and last year,
    calculates the ratio of selfies taken last year to this year. -/
theorem selfie_ratio (total : ℕ) (difference : ℕ) (h1 : total = 2430) (h2 : difference = 630) :
  (total - difference) / 2 = 10 / 17 := by
  sorry

#check selfie_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_selfie_ratio_l805_80592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_property_l805_80505

noncomputable def projection (v : ℝ × ℝ) (u : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := v.1 * u.1 + v.2 * u.2
  let magnitude_squared := u.1 * u.1 + u.2 * u.2
  let scalar := dot_product / magnitude_squared
  (scalar * u.1, scalar * u.2)

theorem projection_property (P : (ℝ × ℝ) → (ℝ × ℝ)) :
  P (3, 3) = (45/10, 9/10) →
  P (-3, 3) = (-30/13, -6/13) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_property_l805_80505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_minutes_run_l805_80568

/-- Represents the number of students in the eighth grade -/
def e : ℕ := sorry

/-- Average minutes run per day by sixth graders -/
def sixth_avg : ℚ := 14

/-- Average minutes run per day by seventh graders -/
def seventh_avg : ℚ := 15

/-- Average minutes run per day by eighth graders -/
def eighth_avg : ℚ := 8

/-- Number of sixth grade students -/
def sixth_count : ℕ := 6 * e

/-- Number of seventh grade students -/
def seventh_count : ℕ := 2 * e

/-- Number of eighth grade students -/
def eighth_count : ℕ := e

/-- Total number of students -/
def total_students : ℕ := sixth_count + seventh_count + eighth_count

/-- Total minutes run by all students -/
def total_minutes : ℚ := sixth_avg * (sixth_count : ℚ) + seventh_avg * (seventh_count : ℚ) + eighth_avg * (eighth_count : ℚ)

/-- Theorem stating the average minutes run by all students -/
theorem average_minutes_run : total_minutes / (total_students : ℚ) = 122 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_minutes_run_l805_80568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_circumcenter_to_line_l805_80588

noncomputable section

-- Define the points A, B, C
def A : ℝ × ℝ := (0, 1)
def B : ℝ × ℝ := (-Real.sqrt 3, 0)
def C : ℝ × ℝ := (-Real.sqrt 3, 2)

-- Define the line y = -√3x
def line (x : ℝ) : ℝ := -(Real.sqrt 3) * x

-- Function to calculate the circumcenter of a triangle
def circumcenter (p1 p2 p3 : ℝ × ℝ) : ℝ × ℝ := sorry

-- Function to calculate the distance from a point to a line y = mx + b
def distancePointToLine (p : ℝ × ℝ) (m b : ℝ) : ℝ := sorry

theorem distance_circumcenter_to_line :
  let cc := circumcenter A B C
  distancePointToLine cc (-(Real.sqrt 3)) 0 = 1/2 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_circumcenter_to_line_l805_80588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_l805_80549

-- Define the central angle in degrees
def α : ℝ := 60

-- Define the arc length
noncomputable def l : ℝ := 6 * Real.pi

-- Define the area of the sector
noncomputable def S : ℝ := 54 * Real.pi

-- Theorem statement
theorem sector_area :
  α = 60 ∧ l = 6 * Real.pi → S = 54 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_l805_80549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_point_implies_original_point_l805_80521

def InversePassesThrough (f : ℝ → ℝ) (x y : ℝ) : Prop :=
  ∃ (f_inv : ℝ → ℝ), Function.RightInverse f_inv f ∧ f_inv x = y

theorem inverse_point_implies_original_point (f : ℝ → ℝ) :
  InversePassesThrough f 1 5 → f 5 = 1 := by
  intro h
  rcases h with ⟨f_inv, right_inv, point_cond⟩
  have : f (f_inv 1) = 1 := by
    exact (right_inv 1)
  rw [point_cond] at this
  exact this


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_point_implies_original_point_l805_80521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_speed_minimizes_cost_l805_80566

/-- Represents the speed of the taxi in km/h -/
noncomputable def speed : ℝ → ℝ := id

/-- Represents the fuel cost per hour as a function of speed -/
noncomputable def fuel_cost (v : ℝ) : ℝ := (v^3) / 64000

/-- Represents the total cost as a function of speed -/
noncomputable def total_cost (v : ℝ) : ℝ := (fuel_cost v + 12) * (160 / v)

/-- Theorem stating that the speed minimizing total cost is 40∛36 km/h -/
theorem optimal_speed_minimizes_cost : 
  ∃ (v : ℝ), v = 40 * (36^(1/3)) ∧ 
  ∀ (u : ℝ), u > 0 → total_cost v ≤ total_cost u := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_speed_minimizes_cost_l805_80566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tic_tac_toe_perimeter_l805_80551

/-- A rectangular strip with width and length -/
structure Strip where
  width : ℝ
  length : ℝ

/-- The polygon formed by four strips in a tic-tac-toe pattern -/
structure TicTacToePolygon where
  strips : Finset Strip
  is_valid : strips.card = 4 ∧ 
             (∀ s ∈ strips, s.width = 4 ∧ s.length = 16) ∧
             (∃ v₁ v₂ h₁ h₂, v₁ ∈ strips ∧ v₂ ∈ strips ∧ h₁ ∈ strips ∧ h₂ ∈ strips ∧ 
                             v₁ ≠ v₂ ∧ h₁ ≠ h₂)

/-- The perimeter of the TicTacToePolygon -/
def perimeter (p : TicTacToePolygon) : ℝ :=
  2 * (16 + 16) + 4 * 4

/-- Theorem: The perimeter of the TicTacToePolygon is 80 inches -/
theorem tic_tac_toe_perimeter (p : TicTacToePolygon) : perimeter p = 80 := by
  unfold perimeter
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tic_tac_toe_perimeter_l805_80551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l805_80589

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * x^2

-- Define the derivative of f (m(x) in the problem)
noncomputable def m (a : ℝ) (x : ℝ) : ℝ := (1 / x) + 2 * a * x

-- Define the function g
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x - a * x^2 + a * x

-- Theorem for part I
theorem part_one (a : ℝ) : (deriv (m a)) 1 = 3 → a = 2 := by sorry

-- Theorem for part II
theorem part_two (a : ℝ) : 
  (∀ x > 0, Monotone (g a)) → a ≥ 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l805_80589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_pages_and_reading_time_l805_80544

/-- Represents a book with a given number of chapters and pages -/
structure Book where
  chapters : ℕ
  total_pages : ℕ

/-- Represents a reader with a maximum number of pages they can read per day -/
structure Reader where
  max_pages_per_day : ℕ

/-- Calculates the number of pages in each chapter given the first chapter's page count -/
def chapter_pages (first_chapter_pages : ℕ) (chapter_number : ℕ) : ℕ :=
  first_chapter_pages + 3 * (chapter_number - 1)

/-- Calculates the total number of pages in the book given the first chapter's page count -/
def total_pages (book : Book) (first_chapter_pages : ℕ) : ℕ :=
  (List.range book.chapters).map (fun i => chapter_pages first_chapter_pages (i + 1)) |>.sum

/-- Calculates the number of days needed to read the book -/
def days_to_read (book : Book) (reader : Reader) : ℕ :=
  (book.total_pages + reader.max_pages_per_day - 1) / reader.max_pages_per_day

theorem book_pages_and_reading_time (book : Book) (reader : Reader) 
    (h1 : book.chapters = 5)
    (h2 : book.total_pages = 95)
    (h3 : reader.max_pages_per_day = 10) :
    ∃ (first_chapter_pages : ℕ),
      first_chapter_pages = 13 ∧
      total_pages book first_chapter_pages = book.total_pages ∧
      days_to_read book reader = 10 := by
  sorry

#eval total_pages { chapters := 5, total_pages := 95 } 13
#eval days_to_read { chapters := 5, total_pages := 95 } { max_pages_per_day := 10 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_pages_and_reading_time_l805_80544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_tangent_sine_l805_80540

theorem angle_sum_tangent_sine (θ φ : ℝ) (h1 : 0 < θ ∧ θ < π / 2) (h2 : 0 < φ ∧ φ < π / 2)
  (h3 : Real.tan θ = 3 / 4) (h4 : Real.sin φ = 3 / 5) :
  θ + φ = Real.arctan (24 / 7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_tangent_sine_l805_80540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_characterization_of_special_sets_l805_80569

theorem characterization_of_special_sets (X : Set ℝ) :
  X.Nonempty ∧ X.Finite ∧ (∀ x ∈ X, x + |x| ∈ X) →
  (X ⊆ Set.Iic 0) ∧ (0 ∈ X) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_characterization_of_special_sets_l805_80569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_primes_satisfy_property_l805_80570

open Complex BigOperators

/-- A function that checks if a given set of complex numbers are equally spaced on the unit circle -/
def are_equally_spaced {n : ℕ} (z : Fin n → ℂ) : Prop :=
  ∀ i j : Fin n, ∃ k : ℕ, z j = z i * exp (2 * Real.pi * I * (k : ℝ) / n)

/-- The property that we want to prove holds for exactly two prime numbers -/
def property (n : ℕ) : Prop :=
  Nat.Prime n ∧
  ∀ z : Fin n → ℂ, 
    (∀ i : Fin n, abs (z i) = 1) →
    (∑ i, z i) = 0 →
    are_equally_spaced z

/-- Theorem stating that exactly two primes satisfy the property -/
theorem exactly_two_primes_satisfy_property : 
  ∃! (s : Finset ℕ), s.card = 2 ∧ ∀ n ∈ s, property n :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_primes_satisfy_property_l805_80570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_setA_is_pythagorean_triple_l805_80510

-- Define the sets of numbers
noncomputable def setA : List ℝ := [1, 2, Real.sqrt 5]
def setB : List ℝ := [6, 8, 9]
noncomputable def setC : List ℝ := [Real.sqrt 3, Real.sqrt 2, 5]
def setD : List ℝ := [3^2, 4^2, 5^2]

-- Function to check if a set of three numbers satisfies the Pythagorean theorem
def isPythagoreanTriple (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

-- Theorem stating that only setA satisfies the Pythagorean theorem
theorem only_setA_is_pythagorean_triple :
  (∃ (a b c : ℝ), a ∈ setA ∧ b ∈ setA ∧ c ∈ setA ∧ isPythagoreanTriple a b c) ∧
  (¬∃ (a b c : ℝ), a ∈ setB ∧ b ∈ setB ∧ c ∈ setB ∧ isPythagoreanTriple a b c) ∧
  (¬∃ (a b c : ℝ), a ∈ setC ∧ b ∈ setC ∧ c ∈ setC ∧ isPythagoreanTriple a b c) ∧
  (¬∃ (a b c : ℝ), a ∈ setD ∧ b ∈ setD ∧ c ∈ setD ∧ isPythagoreanTriple a b c) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_setA_is_pythagorean_triple_l805_80510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_through_origin_l805_80536

-- Define the curve function
noncomputable def f (x : ℝ) : ℝ := Real.exp (x - 1) + x

-- Define the derivative of the curve function
noncomputable def f' (x : ℝ) : ℝ := Real.exp (x - 1) + 1

-- Theorem statement
theorem tangent_line_through_origin : 
  ∃ (m : ℝ), 
    (f m = Real.exp (m - 1) + m) ∧ 
    ((f m) / m = f' m) ∧ 
    (0 = f m - (f' m) * m) ∧ 
    (f' m = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_through_origin_l805_80536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l805_80506

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (4 - 2*x) + Real.sqrt x

-- State the theorem
theorem range_of_f :
  let S := {y | ∃ x, 0 ≤ x ∧ x ≤ 2 ∧ f x = y}
  S = {y | Real.sqrt 2 ≤ y ∧ y ≤ Real.sqrt 6} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l805_80506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_two_digit_primes_sum_10_l805_80557

/-- A function that returns true if a number is prime, false otherwise -/
def isPrime (n : Nat) : Bool :=
  n > 1 && (Nat.factors n).length == 1

/-- A function that returns the sum of digits of a natural number -/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- A function that returns true if a number is two-digit, false otherwise -/
def isTwoDigit (n : Nat) : Bool :=
  10 ≤ n && n ≤ 99

/-- The count of two-digit prime numbers whose digits sum to 10 is 4 -/
theorem count_two_digit_primes_sum_10 :
  (Finset.filter (fun n => isPrime n && isTwoDigit n && sumOfDigits n = 10) (Finset.range 100)).card = 4 := by
  sorry

#eval (Finset.filter (fun n => isPrime n && isTwoDigit n && sumOfDigits n = 10) (Finset.range 100)).card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_two_digit_primes_sum_10_l805_80557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_ticket_cost_is_four_l805_80528

/-- Represents the cost of tickets and sales data for an event -/
structure TicketSales where
  generalAdmissionCost : ℚ
  totalTicketsSold : ℕ
  totalRevenue : ℚ
  generalAdmissionSold : ℕ

/-- Calculates the cost of a student ticket given ticket sales data -/
noncomputable def studentTicketCost (sales : TicketSales) : ℚ :=
  (sales.totalRevenue - sales.generalAdmissionCost * sales.generalAdmissionSold) /
  (sales.totalTicketsSold - sales.generalAdmissionSold)

/-- Theorem stating that the student ticket cost is 4 dollars for the given sales data -/
theorem student_ticket_cost_is_four :
  let sales : TicketSales := {
    generalAdmissionCost := 6,
    totalTicketsSold := 525,
    totalRevenue := 2876,
    generalAdmissionSold := 388
  }
  studentTicketCost sales = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_ticket_cost_is_four_l805_80528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_transformation_l805_80583

theorem matrix_transformation (a b c d : ℝ) : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![a, b; c, d]
  let M : Matrix (Fin 2) (Fin 2) ℝ := !![2, 0; 0, 3]
  M • A = !![2*a, 2*b; 3*c, 3*d] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_transformation_l805_80583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_inverse_five_l805_80576

-- Define the function g as noncomputable
noncomputable def g (x : ℝ) : ℝ := 25 / (4 + 5 * x)

-- State the theorem
theorem inverse_g_inverse_five : (Function.invFun g 5)⁻¹ = 5 := by
  -- The proof is omitted using 'sorry'
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_inverse_five_l805_80576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_sum_is_8_l805_80579

/-- An arithmetic sequence with common difference d and first term a1 -/
noncomputable def arithmeticSequence (d : ℝ) (a1 : ℝ) (n : ℕ) : ℝ := a1 + d * (n - 1)

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def arithmeticSum (d : ℝ) (a1 : ℝ) (n : ℕ) : ℝ :=
  n * a1 + n * (n - 1) * d / 2

theorem smallest_positive_sum_is_8 (d : ℝ) (a1 : ℝ) (h1 : d > 0)
    (h2 : arithmeticSequence d a1 7 = 3 * arithmeticSequence d a1 5) :
    (∀ k < 8, arithmeticSum d a1 k ≤ 0) ∧
    arithmeticSum d a1 8 > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_sum_is_8_l805_80579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_marked_points_l805_80575

-- Define the circle and starting point
def Circle : Type := Unit
def S : Circle := ()

-- Define the positions of Mario and Luigi as functions of time
noncomputable def luigi_position (t : ℝ) : ℝ := (Real.pi * t) / 3
noncomputable def mario_position (t : ℝ) : ℝ := Real.pi * t

-- Define Princess Daisy's position as the midpoint between Mario and Luigi
noncomputable def daisy_position (t : ℝ) : ℝ × ℝ :=
  (((Real.cos (luigi_position t) + Real.cos (mario_position t)) / 2),
   ((Real.sin (luigi_position t) + Real.sin (mario_position t)) / 2))

-- Define the property of a point being marked by Princess Daisy
def is_marked (p : ℝ × ℝ) : Prop :=
  ∃ t₁ t₂, 0 ≤ t₁ ∧ t₁ < t₂ ∧ t₂ ≤ 6 ∧ daisy_position t₁ = daisy_position t₂ ∧ daisy_position t₁ = p

-- The main theorem
theorem distinct_marked_points :
  ∃ (marked_points : Finset (ℝ × ℝ)),
    (∀ p ∈ marked_points, is_marked p) ∧
    (∀ p, is_marked p → p ∈ marked_points ∨ p = daisy_position 0) ∧
    (Finset.card marked_points = 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_marked_points_l805_80575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_valid_paths_l805_80561

/-- Represents a point on the grid --/
structure Point where
  x : ℕ
  y : ℕ

/-- Represents a move on the grid --/
inductive Move
  | Right
  | Down

/-- Represents a path on the grid --/
def GridPath := List Move

/-- The starting point A --/
def A : Point := ⟨0, 2⟩

/-- The intermediate point B --/
def B : Point := ⟨1, 1⟩

/-- The ending point C --/
def C : Point := ⟨2, 0⟩

/-- Function to check if a path is valid --/
def isValidPath (p : GridPath) : Bool :=
  sorry

/-- Function to check if a path passes through B --/
def passesThrough (p : GridPath) (point : Point) : Bool :=
  sorry

/-- Function to count valid paths from A to C passing through B --/
def countValidPaths : ℕ :=
  sorry

/-- Theorem stating that there are exactly 4 valid paths from A to C passing through B --/
theorem four_valid_paths : countValidPaths = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_valid_paths_l805_80561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_value_max_area_l805_80500

noncomputable section

variable (a b c A B C : ℝ)

-- Define the triangle ABC
def is_triangle (a b c A B C : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi ∧
  A + B + C = Real.pi

-- Define the condition given in the problem
def condition (a b c A B C : ℝ) : Prop :=
  c * Real.tan C = Real.sqrt 3 * (a * Real.cos B + b * Real.cos A)

-- Theorem 1: If the condition holds, then C = π/3
theorem angle_C_value (h : is_triangle a b c A B C) (cond : condition a b c A B C) : 
  C = Real.pi / 3 := by sorry

-- Theorem 2: If c = 2√3, the maximum area of the triangle is 3√3
theorem max_area (h : is_triangle a b c A B C) (h_c : c = 2 * Real.sqrt 3) :
  (∀ a' b' c' A' B' C', is_triangle a' b' c' A' B' C' → c' = 2 * Real.sqrt 3 → 
    1/2 * a * c * Real.sin B ≥ 1/2 * a' * c' * Real.sin B') ∧
  1/2 * a * c * Real.sin B = 3 * Real.sqrt 3 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_value_max_area_l805_80500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_inscribed_kite_sum_a_b_is_239_l805_80548

/-- A kite inscribed in a circle --/
structure InscribedKite :=
  (O : Point) -- Center of the circle
  (radius : ℝ) -- Radius of the circle
  (P : Point) -- Point where diagonals meet
  (OP_integer : ℤ) -- OP distance as an integer

/-- The minimum area of an inscribed kite --/
noncomputable def min_kite_area (k : InscribedKite) : ℝ := 120 * Real.sqrt 119

/-- Theorem: The minimum area of a kite inscribed in a circle of radius 60 --/
theorem min_area_inscribed_kite (k : InscribedKite) 
  (h1 : k.radius = 60) 
  (h2 : k.OP_integer < 60) :
  min_kite_area k = 120 * Real.sqrt 119 := by
  sorry

/-- The sum of a and b in the minimum area expression a√b --/
def sum_a_b : ℕ := 239

/-- Theorem: The sum of a and b is 239 --/
theorem sum_a_b_is_239 : sum_a_b = 239 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_inscribed_kite_sum_a_b_is_239_l805_80548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_inscribed_area_l805_80598

-- Define the curve C in polar coordinates
noncomputable def curve_C (θ : ℝ) : ℝ := 4 * Real.cos θ

-- Define the line l in parametric form
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (5 + (Real.sqrt 3 / 2) * t, (1 / 2) * t)

-- Define the Cartesian equation of curve C
def cartesian_C (x y : ℝ) : Prop := x^2 + y^2 = 4*x

-- Define the general equation of line l
def general_l (x y : ℝ) : Prop := x - Real.sqrt 3 * y - 5 = 0

-- Define a simple area function for rectangles
def rectangle_area (width height : ℝ) : ℝ := width * height

-- Theorem statement
theorem rectangle_inscribed_area : 
  ∃ (P Q : ℝ × ℝ) (width height : ℝ),
    (cartesian_C P.1 P.2 ∧ general_l P.1 P.2) ∧ 
    (cartesian_C Q.1 Q.2 ∧ general_l Q.1 Q.2) ∧
    (P ≠ Q) ∧
    (rectangle_area width height = 3 * Real.sqrt 7) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_inscribed_area_l805_80598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_planes_parallel_to_same_plane_are_parallel_lines_perpendicular_to_same_plane_are_parallel_l805_80596

-- Define a 3D space
variable (Point : Type) [NormedAddCommGroup Point] [InnerProductSpace ℝ Point] [FiniteDimensional ℝ Point]
variable [Fact (finrank ℝ Point = 3)]

-- Define planes and lines in the 3D space
variable (Plane Line : Set Point)

-- Define the parallel relation for planes
variable (parallel_planes : Plane → Plane → Prop)

-- Define the perpendicular relation for lines and planes
variable (perpendicular_line_plane : Line → Plane → Prop)

-- Theorem 1: Two planes parallel to the same plane are parallel
theorem planes_parallel_to_same_plane_are_parallel
  (P Q R : Plane)
  (h1 : parallel_planes P R)
  (h2 : parallel_planes Q R) :
  parallel_planes P Q :=
sorry

-- Theorem 2: Two lines perpendicular to the same plane are parallel
theorem lines_perpendicular_to_same_plane_are_parallel
  (l m : Line) (P : Plane)
  (h1 : perpendicular_line_plane l P)
  (h2 : perpendicular_line_plane m P) :
  ∃ (parallel_lines : Line → Line → Prop), parallel_lines l m :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_planes_parallel_to_same_plane_are_parallel_lines_perpendicular_to_same_plane_are_parallel_l805_80596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_degreeSum_nPointedStar_l805_80574

/-- An n-pointed star is created from a convex n-sided polygon by extending every third side -/
structure NPointedStar (n : ℕ) where
  h : n ≥ 6

/-- The degree-sum of the interior angles at the n points of an n-pointed star -/
def degreeSumInteriorAngles (n : ℕ) (star : NPointedStar n) : ℝ :=
  180 * (n - 2)

/-- Theorem: The degree-sum of the interior angles at the n points of an n-pointed star is 180°(n-2) -/
theorem degreeSum_nPointedStar (n : ℕ) (star : NPointedStar n) :
  degreeSumInteriorAngles n star = 180 * (n - 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_degreeSum_nPointedStar_l805_80574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_square_side_is_optimal_l805_80571

/-- The minimum side length of a square containing 5 non-overlapping unit circles -/
noncomputable def min_square_side : ℝ := 2 * Real.sqrt 2 + 2

/-- A configuration of 5 circles in a square -/
structure CircleConfiguration where
  square_side : ℝ
  centers : Fin 5 → ℝ × ℝ

/-- Predicate for a valid configuration of 5 non-overlapping unit circles in a square -/
def is_valid_configuration (config : CircleConfiguration) : Prop :=
  -- All centers are within the square
  (∀ i, (config.centers i).1 ≥ 1 ∧ (config.centers i).1 ≤ config.square_side - 1) ∧
  (∀ i, (config.centers i).2 ≥ 1 ∧ (config.centers i).2 ≤ config.square_side - 1) ∧
  -- No two circles overlap
  (∀ i j, i ≠ j → 
    ((config.centers i).1 - (config.centers j).1)^2 + 
    ((config.centers i).2 - (config.centers j).2)^2 ≥ 4)

theorem min_square_side_is_optimal :
  (∀ config : CircleConfiguration, is_valid_configuration config → config.square_side ≥ min_square_side) ∧
  (∃ config : CircleConfiguration, is_valid_configuration config ∧ config.square_side = min_square_side) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_square_side_is_optimal_l805_80571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l805_80535

theorem division_problem (x y : ℕ) 
  (hx : x > 0)
  (hy : y > 0)
  (h1 : x % y = 9)
  (h2 : (x : ℝ) / y = 96.12) : 
  y = 75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l805_80535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifthDegreeMonomial_is_valid_l805_80522

/-- A monomial is represented by its exponents for x and y. -/
structure Monomial where
  x : ℕ
  y : ℕ

/-- The degree of a monomial is the sum of its exponents. -/
def degree (m : Monomial) : ℕ := m.x + m.y

/-- A fifth-degree monomial containing x and y. -/
def fifthDegreeMonomial : Monomial := { x := 2, y := 3 }

/-- Theorem: The monomial x^2y^3 is a valid fifth-degree monomial containing x and y. -/
theorem fifthDegreeMonomial_is_valid :
  degree fifthDegreeMonomial = 5 ∧
  fifthDegreeMonomial.x > 0 ∧
  fifthDegreeMonomial.y > 0 := by
  sorry

#eval degree fifthDegreeMonomial

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifthDegreeMonomial_is_valid_l805_80522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_b_unique_l805_80555

def a : Fin 3 → ℝ := ![2, 1, 5]

theorem vector_b_unique (b : Fin 3 → ℝ) 
  (h1 : (a 0) * (b 0) + (a 1) * (b 1) + (a 2) * (b 2) = 11)
  (h2 : ((a 1) * (b 2) - (a 2) * (b 1), (a 2) * (b 0) - (a 0) * (b 2), (a 0) * (b 1) - (a 1) * (b 0)) = (-13, -9, 7)) :
  b = ![(-1), 3, 2] := by
  sorry

#check vector_b_unique

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_b_unique_l805_80555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_plus_17pi_12_l805_80511

theorem cos_alpha_plus_17pi_12 (α : ℝ) (h : Real.sin (α - π/12) = 1/3) :
  Real.cos (α + 17*π/12) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_plus_17pi_12_l805_80511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_in_S_l805_80513

noncomputable def S : Set (ℝ × ℝ) := {p : ℝ × ℝ | let (x, y) := p; x > 0 ∧ y > 0 ∧ Real.log (x^3 + (1/3)*y^3 + 1/9) = Real.log x + Real.log y}

theorem unique_solution_in_S : ∃! p : ℝ × ℝ, p ∈ S := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_in_S_l805_80513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_angle_range_l805_80541

noncomputable def f (x : ℝ) : ℝ := (1/2) * (x^3 - 1/x)

theorem tangent_angle_range :
  ∀ x : ℝ, x ≠ 0 →
  ∃ α : ℝ, 0 ≤ α ∧ α < Real.pi ∧
  Real.tan α = (deriv f) x ∧
  Real.pi/3 ≤ α ∧ α < Real.pi/2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_angle_range_l805_80541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_is_one_a_range_for_f_less_than_two_min_value_property_l805_80580

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + a| + |x|

-- Part I
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x ≥ 2} = Set.Iic (-1/2) ∪ Set.Ici (3/2) := by sorry

-- Part II
theorem a_range_for_f_less_than_two :
  (∃ x : ℝ, f a x < 2) ↔ a ∈ Set.Ioo (-2) 2 := by sorry

-- Additional theorem to show the minimum value property
theorem min_value_property (a : ℝ) :
  (∃ x : ℝ, f a x < 2) → (∃ x : ℝ, ∀ y : ℝ, f a x ≤ f a y) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_is_one_a_range_for_f_less_than_two_min_value_property_l805_80580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_point_of_f_l805_80563

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (1 / Real.exp 1) ^ (2 * x - 1) - 1

-- Theorem stating that (1/2, 0) is the unique point that f(x) always passes through
theorem unique_point_of_f :
  ∃! p : ℝ × ℝ, p.1 = 1/2 ∧ p.2 = 0 ∧ f p.1 = p.2 := by
  sorry

#check unique_point_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_point_of_f_l805_80563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_transformation_sequence_l805_80501

/-- Represents the transformation operation -/
def transform (n : ℕ) : ℕ × ℕ → Prop :=
  fun (a, b) => a + b = n ∧ a > 0 ∧ b > 0

/-- Represents a valid sequence of transformations -/
def valid_sequence : List ℕ → Prop
  | [] => False
  | [_] => True
  | (n::m::rest) => ∃ (a b : ℕ), transform n (a, b) ∧ (a = m ∨ b = m) ∧ valid_sequence (m::rest)

theorem exists_transformation_sequence :
  ∃ (seq : List ℕ), seq.head? = some 22 ∧ seq.getLast? = some 2001 ∧ valid_sequence seq := by
  sorry

#check exists_transformation_sequence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_transformation_sequence_l805_80501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_tetrahedron_condition1_regular_tetrahedron_condition2_circumscribed_sphere_height_ratio_point_trajectory_is_ellipse_l805_80564

-- Define a tetrahedron
structure Tetrahedron :=
  (P A B C : EuclideanSpace ℝ (Fin 3))

-- Define properties of the tetrahedron
def is_equilateral_base (t : Tetrahedron) : Prop :=
  ‖t.A - t.B‖ = ‖t.B - t.C‖ ∧ ‖t.B - t.C‖ = ‖t.C - t.A‖

def equal_dihedral_angles (t : Tetrahedron) : Prop :=
  sorry -- Definition of equal dihedral angles between lateral faces and base

def is_regular (t : Tetrahedron) : Prop :=
  ‖t.P - t.A‖ = ‖t.P - t.B‖ ∧ ‖t.P - t.B‖ = ‖t.P - t.C‖ ∧
  ‖t.P - t.C‖ = ‖t.A - t.B‖ ∧ ‖t.A - t.B‖ = ‖t.B - t.C‖ ∧
  ‖t.B - t.C‖ = ‖t.C - t.A‖

def projection_is_orthocenter (t : Tetrahedron) : Prop :=
  sorry -- Definition of projection of A on PBC being the orthocenter of PBC

def all_edges_equal (t : Tetrahedron) : Prop :=
  ‖t.P - t.A‖ = ‖t.P - t.B‖ ∧ ‖t.P - t.B‖ = ‖t.P - t.C‖ ∧
  ‖t.P - t.C‖ = ‖t.A - t.B‖ ∧ ‖t.A - t.B‖ = ‖t.B - t.C‖ ∧
  ‖t.B - t.C‖ = ‖t.C - t.A‖

noncomputable def circumscribed_sphere_radius (t : Tetrahedron) : ℝ :=
  sorry -- Definition of circumscribed sphere radius

noncomputable def tetrahedron_height (t : Tetrahedron) : ℝ :=
  sorry -- Definition of tetrahedron height

def point_on_face (t : Tetrahedron) (M : EuclideanSpace ℝ (Fin 3)) : Prop :=
  sorry -- Definition of point M on face PAB

noncomputable def distance_to_face (t : Tetrahedron) (M : EuclideanSpace ℝ (Fin 3)) : ℝ :=
  sorry -- Definition of distance from M to face ABC

noncomputable def distance_to_point (t : Tetrahedron) (M : EuclideanSpace ℝ (Fin 3)) : ℝ :=
  sorry -- Definition of distance from M to point P

def is_on_ellipse (t : Tetrahedron) (M : EuclideanSpace ℝ (Fin 3)) : Prop :=
  sorry -- Definition of M being on an ellipse

-- Theorem statements
theorem regular_tetrahedron_condition1 (t : Tetrahedron) :
  is_equilateral_base t → equal_dihedral_angles t → is_regular t :=
by sorry

theorem regular_tetrahedron_condition2 (t : Tetrahedron) :
  is_equilateral_base t → projection_is_orthocenter t → is_regular t :=
by sorry

theorem circumscribed_sphere_height_ratio (t : Tetrahedron) :
  all_edges_equal t → circumscribed_sphere_radius t / tetrahedron_height t = 3 / 4 :=
by sorry

theorem point_trajectory_is_ellipse (t : Tetrahedron) (M : EuclideanSpace ℝ (Fin 3)) :
  all_edges_equal t →
  point_on_face t M →
  distance_to_face t M = distance_to_point t M →
  is_on_ellipse t M :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_tetrahedron_condition1_regular_tetrahedron_condition2_circumscribed_sphere_height_ratio_point_trajectory_is_ellipse_l805_80564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_months_to_target_calculable_james_weight_loss_problem_l805_80545

/-- Represents the weight loss scenario for James -/
structure WeightLossScenario where
  initialWeight : ℝ
  currentWeight : ℝ
  targetWeight : ℝ
  monthsSinceDietStart : ℕ
  monthlyWeightLossRate : ℝ

/-- Calculates the number of months needed to reach the target weight -/
noncomputable def monthsToReachTarget (scenario : WeightLossScenario) : ℝ :=
  (scenario.currentWeight - scenario.targetWeight) / scenario.monthlyWeightLossRate

/-- Theorem stating that the number of months to reach the target weight can be calculated -/
theorem months_to_target_calculable (scenario : WeightLossScenario) 
  (h1 : scenario.currentWeight = 198)
  (h2 : scenario.targetWeight = 190)
  (h3 : scenario.monthsSinceDietStart = 12)
  (h4 : scenario.monthlyWeightLossRate > 0)
  (h5 : scenario.initialWeight > scenario.currentWeight) :
  ∃ (m : ℝ), m = monthsToReachTarget scenario :=
by
  sorry

/-- Main theorem representing the problem -/
theorem james_weight_loss_problem 
  (initialWeight : ℝ)
  (h1 : initialWeight > 198) :
  ∃ (scenario : WeightLossScenario) (months : ℝ),
    scenario.initialWeight = initialWeight ∧
    scenario.currentWeight = 198 ∧
    scenario.targetWeight = 190 ∧
    scenario.monthsSinceDietStart = 12 ∧
    scenario.monthlyWeightLossRate = (initialWeight - 198) / 12 ∧
    months = monthsToReachTarget scenario :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_months_to_target_calculable_james_weight_loss_problem_l805_80545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l805_80533

noncomputable def f (x a : ℝ) : ℝ := (1/3) * x^3 + x^2 - 3*x + a

theorem function_properties (a : ℝ) 
  (h_min : ∃ x_min ∈ Set.Icc (-2) 2, ∀ x ∈ Set.Icc (-2) 2, f x a ≥ f x_min a ∧ f x_min a = 2) :
  (∀ x ∈ Set.Ioo (-3) 1, ∀ y ∈ Set.Ioo (-3) 1, x < y → f x a > f y a) ∧
  (∃ x_max ∈ Set.Icc (-2) 2, ∀ x ∈ Set.Icc (-2) 2, f x a ≤ f x_max a ∧ f x_max a = 11) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l805_80533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_juice_amount_l805_80542

/-- Represents a fruit drink composition -/
structure FruitDrink where
  total : ℚ
  grapefruit_percent : ℚ
  lemon_percent : ℚ
  orange_ounces : ℚ

/-- Calculates the amount of orange juice in the drink -/
def calculate_orange_juice (drink : FruitDrink) : ℚ :=
  drink.total - (drink.grapefruit_percent / 100 * drink.total) - (drink.lemon_percent / 100 * drink.total)

/-- Theorem stating the amount of orange juice in the specific drink composition -/
theorem orange_juice_amount (drink : FruitDrink) 
  (h1 : drink.total = 50)
  (h2 : drink.grapefruit_percent = 25)
  (h3 : drink.lemon_percent = 35) :
  calculate_orange_juice drink = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_juice_amount_l805_80542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_implies_a_equals_two_l805_80572

/-- The function representing the curve y = ln(x + a) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + a)

/-- The derivative of f with respect to x -/
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := 1 / (x + a)

theorem tangent_line_implies_a_equals_two (a : ℝ) :
  (∃ x₀ : ℝ, f a x₀ = x₀ + 1 ∧ f' a x₀ = 1) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_implies_a_equals_two_l805_80572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tabs_closed_fraction_l805_80508

theorem tabs_closed_fraction (initial_tabs : ℕ) (final_tabs : ℕ) : 
  initial_tabs = 400 → final_tabs = 90 → 
  ∃ x : ℚ, 0 ≤ x ∧ x ≤ 1 ∧ 
  (initial_tabs * (1 - x) * (1 - 2/5) * (1/2) = final_tabs) ∧
  x = 1/4 := by
  sorry

#check tabs_closed_fraction

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tabs_closed_fraction_l805_80508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_constant_sum_distances_l805_80590

/-- The locus of points with constant sum of distances to two fixed points -/
theorem locus_constant_sum_distances (F₁ F₂ M : ℝ × ℝ) (d : ℝ) :
  F₁ = (-1, 0) →
  F₂ = (1, 0) →
  d = 2 →
  (Set.Icc (-1 : ℝ) 1).prod {0} =
    {M | dist M F₁ + dist M F₂ = d} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_constant_sum_distances_l805_80590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_power_4_l805_80526

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := 
  Nat.choose n k

-- Define the expression
def expr (x : ℚ) : ℚ := x * (x - 2/x)^7

-- Theorem statement
theorem coefficient_of_x_power_4 : 
  (∃ (f : ℚ → ℚ), (∀ x, expr x = f x) ∧ 
   (∃ (a : ℚ), ∀ x, f x = a * x^4 + x^5 * (λ x => x / (1 + x)) x)) → 
  (∃ (a : ℚ), a = 84) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_power_4_l805_80526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_expansion_l805_80599

theorem polynomial_expansion (x : ℝ) (h : x ≠ 0) :
  (x^10 - 4*x^3 + 2/x - 8) * (3*x^5) = 3*x^15 - 12*x^8 - 24*x^5 + 6*x^4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_expansion_l805_80599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_parallel_to_polar_axis_l805_80530

-- Define the polar coordinate system
def PolarPoint := ℝ × ℝ

-- Define the equation of the line
noncomputable def LineEquation (p : PolarPoint) : Prop :=
  p.1 * Real.sin p.2 = 1

-- Define the point (1, π/2)
noncomputable def GivenPoint : PolarPoint := (1, Real.pi / 2)

-- Theorem statement
theorem line_through_point_parallel_to_polar_axis :
  -- The line passes through the given point
  LineEquation GivenPoint ∧
  -- For any two points on the line, their x-coordinates are equal (parallel to polar axis)
  ∀ (p1 p2 : PolarPoint), LineEquation p1 → LineEquation p2 → p1.1 * Real.cos p1.2 = p2.1 * Real.cos p2.2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_parallel_to_polar_axis_l805_80530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l805_80517

open Real

-- Define the function f(x) on the open interval (0, π)
noncomputable def f (x : ℝ) : ℝ := (1/2) * x - Real.sin x

-- State the theorem
theorem f_min_value :
  ∃ (x : ℝ), x ∈ Set.Ioo 0 π ∧ 
  (∀ (y : ℝ), y ∈ Set.Ioo 0 π → f y ≥ f x) ∧
  f x = π/6 - Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l805_80517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_revenue_minimum_value_l805_80502

noncomputable def P (x : ℝ) : ℝ := 5 + 10 / x

noncomputable def Q (x : ℝ) : ℝ := -abs (x - 20) + 100

noncomputable def f (x : ℝ) : ℝ := P x * Q x

theorem revenue_minimum_value (x : ℝ) (h : 1 ≤ x ∧ x ≤ 30) : 
  f x ≥ 441 ∧ ∃ y : ℝ, 1 ≤ y ∧ y ≤ 30 ∧ f y = 441 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_revenue_minimum_value_l805_80502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_lambda_range_l805_80586

/-- An arithmetic sequence with sum of first n terms S_n -/
def S (n : ℕ) : ℝ := sorry

/-- The condition that there are precisely three positive integers n that satisfy n·S_n ≤ λ -/
def precisely_three_integers (S : ℕ → ℝ) (lambda : ℝ) : Prop :=
  ∃ (n1 n2 n3 : ℕ), n1 < n2 ∧ n2 < n3 ∧
    (∀ (n : ℕ), n > 0 → n * S n ≤ lambda ↔ n = n1 ∨ n = n2 ∨ n = n3)

theorem arithmetic_sequence_lambda_range (S : ℕ → ℝ) (lambda : ℝ) 
    (h1 : S 6 = -9)
    (h2 : S 8 = 4)
    (h3 : precisely_three_integers S lambda) :
  lambda ∈ Set.Icc (-54) (-81/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_lambda_range_l805_80586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_difference_l805_80582

/-- The parabola y^2 = x -/
def Parabola : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = p.1}

/-- Point M -/
def M : ℝ × ℝ := (2, -1)

/-- Points on the parabola -/
def PointOnParabola (P : ℝ × ℝ) : Prop := P ∈ Parabola

/-- Arithmetic sequence condition for y-coordinates -/
def ArithmeticSequence (P₁ P₂ P₃ P₄ : ℝ × ℝ) : Prop :=
  ∃ d : ℝ, P₂.2 - P₁.2 = d ∧ P₃.2 - P₂.2 = d ∧ P₄.2 - P₃.2 = d

/-- Distance between two points -/
noncomputable def Distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

/-- Theorem statement -/
theorem parabola_chord_difference (P₁ Q₁ P₂ Q₂ P₃ Q₃ P₄ Q₄ : ℝ × ℝ) 
  (h₁ : PointOnParabola P₁) (h₂ : PointOnParabola Q₁)
  (h₃ : PointOnParabola P₂) (h₄ : PointOnParabola Q₂)
  (h₅ : PointOnParabola P₃) (h₆ : PointOnParabola Q₃)
  (h₇ : PointOnParabola P₄) (h₈ : PointOnParabola Q₄)
  (h₉ : ArithmeticSequence P₁ P₂ P₃ P₄) :
  (Distance P₁ M / Distance M Q₁ - Distance P₂ M / Distance M Q₂) -
  (Distance P₃ M / Distance M Q₃ - Distance P₄ M / Distance M Q₄) > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_difference_l805_80582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equation_implies_affine_l805_80567

/-- A function satisfying the given functional equation is affine. -/
theorem function_equation_implies_affine (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x^3) - f (y^3) = (x^2 + x*y + y^2) * (f x - f y)) :
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equation_implies_affine_l805_80567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_first_two_terms_l805_80519

def sequence_b (b₁ b₂ : ℕ+) : ℕ → ℕ+
  | 0 => b₁
  | 1 => b₂
  | (n + 2) => ⟨(sequence_b b₁ b₂ n + 2017) / (1 + sequence_b b₁ b₂ (n + 1)), sorry⟩

theorem min_sum_first_two_terms :
  ∃ (b₁ b₂ : ℕ+), (∀ n, (sequence_b b₁ b₂ (n + 2) : ℕ) * (1 + sequence_b b₁ b₂ (n + 1)) = sequence_b b₁ b₂ n + 2017) ∧
  (∀ c₁ c₂ : ℕ+, (∀ n, (sequence_b c₁ c₂ (n + 2) : ℕ) * (1 + sequence_b c₁ c₂ (n + 1)) = sequence_b c₁ c₂ n + 2017) →
  (b₁ : ℕ) + (b₂ : ℕ) ≤ (c₁ : ℕ) + (c₂ : ℕ)) ∧
  (b₁ : ℕ) + (b₂ : ℕ) = 2018 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_first_two_terms_l805_80519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_area_of_triangle_l805_80585

theorem circumcircle_area_of_triangle (A B C : ℝ) (a b c : ℝ) :
  (Real.sqrt 3 * Real.sin B + 2 * (Real.cos (B / 2))^2 = 3) →
  ((Real.cos B) / b + (Real.cos C) / c = (Real.sin A * Real.sin B) / (6 * Real.sin C)) →
  (∃ (R : ℝ), R > 0 ∧ Real.pi * R^2 = 16 * Real.pi) :=
by
  intro h1 h2
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_area_of_triangle_l805_80585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_lines_count_l805_80547

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Calculates the distance from a point to a line -/
noncomputable def distanceToLine (p : Point) (l : Line) : ℝ :=
  abs (l.a * p.x + l.b * p.y + l.c) / Real.sqrt (l.a^2 + l.b^2)

/-- Counts the number of lines equidistant from two points -/
def countEquidistantLines (A B : Point) (m : ℝ) : ℕ :=
  sorry

theorem equidistant_lines_count 
  (A B : Point) 
  (h1 : A.x = 1 ∧ A.y = 2) 
  (h2 : B.x = 5 ∧ B.y = -1) 
  (m : ℝ) 
  (h3 : m > 0) :
  (m < 2.5 → countEquidistantLines A B m = 4) ∧
  (m = 2.5 → countEquidistantLines A B m = 3) ∧
  (m > 2.5 → countEquidistantLines A B m = 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_lines_count_l805_80547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l805_80558

-- Define the quadratic function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x + a

-- Define the function g(x)
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := Real.rpow 2 x + a

-- Theorem statement
theorem range_of_g (a : ℝ) :
  (∀ y : ℝ, y ≤ 3/4 → ∃ x : ℝ, f a x = y) →
  (∀ y : ℝ, y > -1/4 → ∃ x : ℝ, g a x = y) ∧
  (∀ y : ℝ, (∃ x : ℝ, g a x = y) → y > -1/4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l805_80558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_platform_l805_80553

noncomputable def train_length : ℝ := 720
noncomputable def platform_length : ℝ := 780
noncomputable def train_speed_kmh : ℝ := 90

noncomputable def total_distance : ℝ := train_length + platform_length

noncomputable def train_speed_ms : ℝ := train_speed_kmh * 1000 / 3600

noncomputable def time_to_pass : ℝ := total_distance / train_speed_ms

theorem train_passing_platform : time_to_pass = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_platform_l805_80553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_inequality_l805_80512

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 4 * x - (1/2) * x^2 - a * log x

-- State the theorem
theorem extreme_points_inequality (a : ℝ) (x₁ x₂ : ℝ) :
  x₁ > 0 ∧ x₂ > 0 ∧ x₁ < x₂ ∧
  (∃ ε > 0, ∀ x, x₁ - ε < x ∧ x < x₁ + ε → f a x ≤ f a x₁) ∧
  (∃ ε > 0, ∀ x, x₂ - ε < x ∧ x < x₂ + ε → f a x ≤ f a x₂) →
  f a x₁ + f a x₂ < 7 + exp 1 - log x₁ - log x₂ :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_inequality_l805_80512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_angle_is_arctan_two_l805_80578

/-- A regular truncated triangular pyramid with an inscribed sphere. -/
structure TruncatedPyramid where
  /-- The radius of the inscribed sphere -/
  R : ℝ
  /-- The side length of the lower base -/
  a : ℝ
  /-- The side length of the upper base -/
  b : ℝ
  /-- The radius is positive -/
  R_pos : R > 0
  /-- The side lengths are positive -/
  a_pos : a > 0
  b_pos : b > 0
  /-- The upper base is smaller than the lower base -/
  b_lt_a : b < a
  /-- The ratio of sphere surface area to pyramid surface area is π : 6√3 -/
  area_ratio : (4 * π * R^2) / ((a^2 * Real.sqrt 3) / 4 + (b^2 * Real.sqrt 3) / 4 + 3 * ((a + b)^2 / (4 * Real.sqrt 3))) = π / (6 * Real.sqrt 3)

/-- The angle between the lateral face and the base plane of the pyramid -/
noncomputable def lateral_angle (p : TruncatedPyramid) : ℝ :=
  Real.arctan 2

theorem lateral_angle_is_arctan_two (p : TruncatedPyramid) :
  lateral_angle p = Real.arctan 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_angle_is_arctan_two_l805_80578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_function_with_smallest_period_pi_l805_80543

noncomputable def f_A (x : ℝ) := Real.sin x + Real.cos x
noncomputable def f_B (x : ℝ) := Real.sin x ^ 2 - Real.sqrt 3 * Real.cos x ^ 2
noncomputable def f_C (x : ℝ) := Real.cos (abs x)
noncomputable def f_D (x : ℝ) := 3 * Real.sin (x / 2) * Real.cos (x / 2)

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def smallest_positive_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  is_periodic f p ∧ p > 0 ∧ ∀ q, 0 < q ∧ q < p → ¬ is_periodic f q

theorem periodic_function_with_smallest_period_pi :
  smallest_positive_period f_B π ∧
  ¬ smallest_positive_period f_A π ∧
  ¬ smallest_positive_period f_C π ∧
  ¬ smallest_positive_period f_D π :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_function_with_smallest_period_pi_l805_80543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_root_range_l805_80559

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (3 * x - 1) - a * x + a

-- State the theorem
theorem unique_root_range (a : ℝ) :
  a < 1 →
  (∃! (x₀ : ℤ), f a (x₀ : ℝ) ≤ 0) →
  a ∈ Set.Icc (2 / Real.exp 1) 1 ∧ a ≠ 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_root_range_l805_80559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farm_animal_ratio_l805_80524

/-- Proves that the initial ratio of horses to cows is 3:1 given the problem conditions --/
theorem farm_animal_ratio 
  (H C : ℕ) -- H: initial number of horses, C: initial number of cows
  (h1 : (H - 15) / (C + 15) = 5 / 3) -- ratio after transaction
  (h2 : H - 15 = C + 15 + 30) -- difference after transaction
  : H / C = 3 / 1 := by
  sorry

#check farm_animal_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_farm_animal_ratio_l805_80524
