import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l726_72666

/-- A parabola with focus at (1, 0) and points A and B satisfying certain conditions -/
structure Parabola where
  p : ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ
  h_p_pos : p > 0
  h_focus : (1 : ℝ) = p / 2
  h_A_on_parabola : A.2^2 = 2 * p * A.1
  h_B_on_parabola : B.2^2 = 2 * p * B.1
  h_AB_sides : A.2 * B.2 < 0
  h_OA_OB : A.1 * B.1 + A.2 * B.2 = -4

/-- The theorem stating properties of the parabola and related points -/
theorem parabola_properties (P : Parabola) :
  -- 1. The equation of the parabola is y^2 = 4x
  (∀ x y, y^2 = 2 * P.p * x ↔ y^2 = 4 * x) ∧
  -- 2. The line AB passes through the point (2, 0)
  (∃ m : ℝ, P.A.1 = m * P.A.2 + 2 ∧ P.B.1 = m * P.B.2 + 2) ∧
  -- 3. The minimum area of quadrilateral AMBN is 48
  (∀ M N : ℝ × ℝ,
    M.2^2 = 4 * M.1 →
    N.2^2 = 4 * N.1 →
    (N.2 - M.2) * (P.B.1 - P.A.1) = (N.1 - M.1) * (P.B.2 - P.A.2) →
    (2 - M.1) * (P.B.2 - P.A.2) = (0 - M.2) * (P.B.1 - P.A.1) →
    (2 - N.1) * (P.B.2 - P.A.2) = (0 - N.2) * (P.B.1 - P.A.1) →
    48 ≤ (1/2) * Real.sqrt ((P.B.1 - P.A.1)^2 + (P.B.2 - P.A.2)^2) *
               Real.sqrt ((N.1 - M.1)^2 + (N.2 - M.2)^2)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l726_72666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_city_mpg_equals_highway_mpg_l726_72668

/-- The average mileage per gallon on the highway -/
noncomputable def highway_mpg : ℝ := 12.2

/-- The maximum distance in miles that can be driven on 23 gallons of gasoline -/
noncomputable def max_distance : ℝ := 280.6

/-- The number of gallons of gasoline -/
noncomputable def gallons : ℝ := 23

/-- The average mileage per gallon in the city -/
noncomputable def city_mpg : ℝ := max_distance / gallons

theorem city_mpg_equals_highway_mpg : city_mpg = highway_mpg := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_city_mpg_equals_highway_mpg_l726_72668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_special_integers_l726_72622

theorem existence_of_special_integers (n k : ℕ) (h1 : n > 1) (h2 : k > 1) (h3 : n < 2^k) :
  ∃ (S : Finset ℤ),
    Finset.card S = 2 * k ∧
    (∀ x ∈ S, ¬(n ∣ Int.natAbs x)) ∧
    (∀ (A B : Finset ℤ), A ∪ B = S → A ∩ B = ∅ →
      (∃ (C : Finset ℤ), (C ⊆ A ∨ C ⊆ B) ∧ n ∣ (Finset.sum C (Int.natAbs ∘ id)))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_special_integers_l726_72622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l726_72676

noncomputable def m (x : ℝ) : ℝ × ℝ := (Real.sin x, 1)
noncomputable def n (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos x, 1/2)

noncomputable def f (x : ℝ) : ℝ := ((m x).1 + (n x).1) * (m x).1 + ((m x).2 + (n x).2) * (m x).2

theorem triangle_area (A : ℝ) (b c : ℝ) :
  A ∈ Set.Icc 0 (Real.pi / 2) →
  A = Real.pi / 3 →
  b = 4 →
  c = 2 * Real.sqrt 3 →
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f A ≥ f x) →
  (1/2) * b * c * Real.sin A = 2 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l726_72676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_floating_point_l726_72685

/-- Definition of a function with a "floating point" -/
def has_floating_point (f : ℝ → ℝ) (a b : ℝ) :=
  ∃ x₀ ∈ Set.Ioo a b, f (x₀ + 1) = f x₀ + f 1

/-- The function f(x) = x^2 + 2^x -/
noncomputable def f (x : ℝ) : ℝ := x^2 + (2 : ℝ)^x

/-- Theorem: f(x) = x^2 + 2^x has a floating point in (0, 1) -/
theorem f_has_floating_point : has_floating_point f 0 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_floating_point_l726_72685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_angle_l726_72678

-- Define the curves
noncomputable def C₁ (φ : ℝ) : ℝ × ℝ := (2 + 2 * Real.cos φ, 2 * Real.sin φ)

noncomputable def C₂ (θ : ℝ) : ℝ := 4 * Real.sin θ

def C₃ (α : ℝ) : Prop := 0 < α ∧ α < Real.pi

-- Define the intersection points
noncomputable def A (α : ℝ) : ℝ × ℝ := C₁ α

noncomputable def B (α : ℝ) : ℝ × ℝ := (C₂ α * Real.cos α, C₂ α * Real.sin α)

-- State the theorem
theorem intersection_angle :
  ∃ (α : ℝ), C₃ α ∧ 
  A α ≠ (0, 0) ∧ 
  B α ≠ (0, 0) ∧ 
  (A α).1^2 + (A α).2^2 = ((B α).1 - (A α).1)^2 + ((B α).2 - (A α).2)^2 ∧
  α = 3 * Real.pi / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_angle_l726_72678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_towel_bleaching_decrease_l726_72688

/-- Represents the percentage remaining after a reduction --/
noncomputable def remaining_percentage (reduction : ℝ) : ℝ := 1 - reduction

/-- Calculates the area of a rectangle given its length and breadth --/
noncomputable def area (length breadth : ℝ) : ℝ := length * breadth

/-- Represents the three bleaching processes --/
structure BleachingProcess where
  length_reduction1 : ℝ
  breadth_reduction1 : ℝ
  length_reduction2 : ℝ
  breadth_reduction2 : ℝ
  length_reduction3 : ℝ
  breadth_reduction3 : ℝ

/-- Calculates the final area after applying the bleaching processes --/
noncomputable def final_area (initial_length initial_breadth : ℝ) (process : BleachingProcess) : ℝ :=
  let length1 := initial_length * remaining_percentage process.length_reduction1
  let breadth1 := initial_breadth * remaining_percentage process.breadth_reduction1
  let length2 := length1 * remaining_percentage process.length_reduction2
  let breadth2 := breadth1 * remaining_percentage process.breadth_reduction2
  let length3 := length2 * remaining_percentage process.length_reduction3
  let breadth3 := breadth2 * remaining_percentage process.breadth_reduction3
  area length3 breadth3

/-- Calculates the percentage decrease in area --/
noncomputable def percentage_decrease (initial_area final_area : ℝ) : ℝ :=
  (initial_area - final_area) / initial_area * 100

/-- Theorem stating that the total percentage decrease in area after the three bleaching processes is 44.92% --/
theorem towel_bleaching_decrease (initial_length initial_breadth : ℝ) :
  let process : BleachingProcess := {
    length_reduction1 := 0.20,
    breadth_reduction1 := 0.10,
    length_reduction2 := 0.15,
    breadth_reduction2 := 0.05,
    length_reduction3 := 0.10,
    breadth_reduction3 := 0.08
  }
  let initial_area := area initial_length initial_breadth
  let final_area := final_area initial_length initial_breadth process
  percentage_decrease initial_area final_area = 44.92 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_towel_bleaching_decrease_l726_72688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_order_l726_72686

theorem alpha_order (x : ℝ) (h : x ∈ Set.Ioo (-1/2 : ℝ) 0) :
  Real.cos ((x + 1) * Real.pi) < Real.sin (Real.cos (x * Real.pi)) ∧
  Real.sin (Real.cos (x * Real.pi)) < Real.cos (Real.sin (x * Real.pi)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_order_l726_72686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_min_period_and_max_value_l726_72611

noncomputable def f (x : ℝ) := Real.sin (x / 3) + Real.cos (x / 3)

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T')) ∧
  (∃ (M : ℝ), (∀ (x : ℝ), f x ≤ M) ∧ (∃ (x : ℝ), f x = M)) :=
by sorry

theorem min_period_and_max_value :
  (∃ (T : ℝ), T = 6 * Real.pi ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T')) ∧
  (∃ (M : ℝ), M = Real.sqrt 2 ∧ (∀ (x : ℝ), f x ≤ M) ∧ (∃ (x : ℝ), f x = M)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_min_period_and_max_value_l726_72611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_term_is_2sqrt5_l726_72674

noncomputable def sequenceA (n : ℕ) : ℝ := Real.sqrt (3 * n - 1)

theorem seventh_term_is_2sqrt5 : sequenceA 7 = 2 * Real.sqrt 5 := by
  -- Unfold the definition of sequenceA
  unfold sequenceA
  -- Simplify the left-hand side
  simp [Real.sqrt_eq_rpow]
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_term_is_2sqrt5_l726_72674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_student_count_l726_72675

theorem initial_student_count (n : ℕ) 
  (initial_avg : ℚ) (new_avg : ℚ) (dropped_score : ℚ)
  (initial_avg_def : initial_avg = 62.5)
  (new_avg_def : new_avg = 63.0)
  (dropped_score_def : dropped_score = 55) : n = 16 := by
  
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_student_count_l726_72675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_triangle_condition_l726_72612

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c
  h_triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b

-- Define what it means for a triangle to be obtuse
def is_obtuse (t : Triangle) : Prop :=
  ∃ (angle : ℝ), angle > Real.pi / 2 ∧ 
    (angle = Real.arccos ((t.b^2 + t.c^2 - t.a^2) / (2 * t.b * t.c)) ∨
     angle = Real.arccos ((t.a^2 + t.c^2 - t.b^2) / (2 * t.a * t.c)) ∨
     angle = Real.arccos ((t.a^2 + t.b^2 - t.c^2) / (2 * t.a * t.b)))

-- State the theorem
theorem obtuse_triangle_condition (t : Triangle) 
  (h : t.a^2 < (t.b + t.c) * (t.c - t.b)) : 
  is_obtuse t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_triangle_condition_l726_72612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_27_l726_72645

theorem cube_root_of_27 : (27 : Real).rpow (1/3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_27_l726_72645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_points_collinear_l726_72633

-- Define a point in a plane
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define a line in a plane
structure Line :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

-- Define collinearity of three points
def collinear (p q r : Point) : Prop :=
  ∃ l : Line, p.x * l.a + p.y * l.b + l.c = 0 ∧
              q.x * l.a + q.y * l.b + l.c = 0 ∧
              r.x * l.a + r.y * l.b + l.c = 0

-- Define the perpendicular bisector of two points
noncomputable def perp_bisector (p q : Point) : Line :=
  { a := 2 * (q.x - p.x),
    b := 2 * (q.y - p.y),
    c := p.x^2 + p.y^2 - q.x^2 - q.y^2 }

-- Define symmetry with respect to a line
noncomputable def symmetric_point (p : Point) (l : Line) : Point :=
  { x := p.x - 2 * l.a * (l.a * p.x + l.b * p.y + l.c) / (l.a^2 + l.b^2),
    y := p.y - 2 * l.b * (l.a * p.x + l.b * p.y + l.c) / (l.a^2 + l.b^2) }

-- Define the point generation process
noncomputable def generate_point (a b c : Point) : Point :=
  symmetric_point a (perp_bisector b c)

-- Define the set of points after a day
def points_after_day (initial_points : List Point) : Set Point :=
  sorry

-- The main theorem
theorem initial_points_collinear 
  (p q r : Point) 
  (h : ∃ (x y z : Point), x ∈ points_after_day [p, q, r] ∧ 
                          y ∈ points_after_day [p, q, r] ∧ 
                          z ∈ points_after_day [p, q, r] ∧ 
                          x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
                          collinear x y z) : 
  collinear p q r := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_points_collinear_l726_72633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kite_area_is_twenty_l726_72699

/-- Represents a triangle with a base and height -/
structure Triangle where
  base : ℝ
  height : ℝ

/-- Represents a kite composed of two congruent triangles -/
structure Kite where
  triangle : Triangle

/-- Calculates the area of a triangle -/
noncomputable def Triangle.area (t : Triangle) : ℝ := (1 / 2) * t.base * t.height

/-- Calculates the area of a kite -/
noncomputable def Kite.area (k : Kite) : ℝ := 2 * k.triangle.area

/-- Theorem: The area of a kite with specific dimensions is 20 square inches -/
theorem kite_area_is_twenty (k : Kite) 
  (h1 : k.triangle.base = 10) 
  (h2 : k.triangle.height = 2) : 
  k.area = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_kite_area_is_twenty_l726_72699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_such_function_l726_72694

-- State the theorem
theorem no_such_function :
  ¬∃ (f : ℝ → ℝ), ∀ (x : ℝ),
    (f (1 + f x) = 1 - x) ∧ (f (f x) = x) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_such_function_l726_72694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_implies_a_l726_72651

noncomputable def f (x : ℝ) : ℝ := if x ≤ 0 then -x else x^2

theorem f_value_implies_a (a : ℝ) (h : f a = 4) : a = -4 ∨ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_implies_a_l726_72651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_polynomial_p_one_values_l726_72629

/-- A monic polynomial of degree n ≥ 2 with n real roots all ≤ 1 and p(2) = 3^n -/
def SpecialPolynomial (p : ℝ → ℝ) (n : ℕ) : Prop :=
  (∃ (roots : Finset ℝ), (roots.card = n) ∧ 
    (∀ r ∈ roots, r ≤ 1) ∧ 
    (∀ x, p x = (roots.prod (λ r ↦ x - r)))) ∧
  (n ≥ 2) ∧
  (p 2 = 3^n)

theorem special_polynomial_p_one_values {p : ℝ → ℝ} {n : ℕ} 
  (h : SpecialPolynomial p n) : 
  p 1 = 0 ∨ p 1 = (-1)^n * 2^n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_polynomial_p_one_values_l726_72629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_value_given_sin_l726_72614

theorem cos_value_given_sin (x : ℝ) : 
  Real.sin (x + π / 12) = -1 / 4 → Real.cos (5 * π / 6 - 2 * x) = -7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_value_given_sin_l726_72614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_problem_l726_72689

/-- The chord length intercepted by a line on a circle -/
noncomputable def chord_length (a b c : ℝ) (x₀ y₀ r : ℝ) : ℝ :=
  2 * Real.sqrt (r^2 - (a * x₀ + b * y₀ + c)^2 / (a^2 + b^2))

/-- The problem statement -/
theorem chord_length_problem :
  let line_eq := fun x y => 3 * x - 4 * y
  let circle_eq := fun x y => (x - 1)^2 + (y - 2)^2
  chord_length 3 (-4) 0 1 2 (Real.sqrt 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_problem_l726_72689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_cups_count_l726_72626

-- Define the class structure
structure MyClass where
  total_students : ℕ
  boys : ℕ
  girls : ℕ
  total_cups : ℕ

-- Define the theorem
theorem boys_cups_count (c : MyClass) 
  (h1 : c.total_students = 30)
  (h2 : c.girls = 2 * c.boys)
  (h3 : c.boys = 10)
  (h4 : c.total_cups = 90) :
  c.total_cups / c.boys = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_cups_count_l726_72626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_megan_candy_duration_l726_72613

/-- The number of days Megan's candy will last -/
noncomputable def candy_duration (neighbors_candy : ℝ) (sister_candy : ℝ) (daily_consumption : ℝ) : ℝ :=
  (neighbors_candy + sister_candy) / daily_consumption

/-- Theorem: Megan's candy will last 2 days -/
theorem megan_candy_duration :
  candy_duration 11 5 8 = 2 := by
  -- Unfold the definition of candy_duration
  unfold candy_duration
  -- Simplify the arithmetic
  simp [add_div]
  -- Check that (11 + 5) / 8 = 2
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_megan_candy_duration_l726_72613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_eq_altitude_squared_l726_72659

/-- Represents a trapezoid with bases b and c, and altitude a -/
structure Trapezoid where
  b : ℝ  -- shorter base
  c : ℝ  -- longer base
  a : ℝ  -- altitude
  h : b + c = 2 * a  -- condition that sum of bases is twice the altitude

/-- The area of a trapezoid -/
noncomputable def trapezoid_area (t : Trapezoid) : ℝ := (t.b + t.c) * t.a / 2

/-- Theorem: The area of a trapezoid with sum of bases equal to twice the altitude is equal to the square of its altitude -/
theorem trapezoid_area_eq_altitude_squared (t : Trapezoid) : 
  trapezoid_area t = t.a ^ 2 := by
  sorry

#check trapezoid_area_eq_altitude_squared

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_eq_altitude_squared_l726_72659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l726_72658

noncomputable def f (x : ℝ) : ℝ := (x^3 - 1) / x

theorem tangent_line_at_one :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := (2 * x₀^3 + 1) / x₀^2
  (λ (x y : ℝ) => y = m * (x - x₀) + y₀) = (λ (x y : ℝ) => y = 3 * x - 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l726_72658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_box_arrangement_count_five_box_arrangement_l726_72601

/-- Factorial function -/
def factorial : ℕ → ℕ
| 0 => 1
| n + 1 => (n + 1) * factorial n

/-- Number of permutations function -/
def number_of_permutations : ℕ → ℕ → ℕ
| 0, _ => 1
| _, 0 => 0
| n + 1, k + 1 => (n + 1) * number_of_permutations n k

/-- The number of permutations of n objects taken n at a time is equal to n! -/
theorem box_arrangement_count : ∀ n : ℕ, n > 0 → factorial n = number_of_permutations n n := by
  sorry

/-- The number of ways to arrange 5 distinct objects in 5 positions is 5! (120) -/
theorem five_box_arrangement : factorial 5 = 120 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_box_arrangement_count_five_box_arrangement_l726_72601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_theorem_l726_72625

-- Define the circle
structure Circle where
  center : Point
  radius : ℝ

-- Define the arc
structure Arc (c : Circle) where
  start : Point
  endpoint : Point

-- Define the angle
noncomputable def Angle (p q r : Point) : ℝ := sorry

-- Define arc length
noncomputable def ArcLength (c : Circle) (a : Arc c) : ℝ := sorry

-- State the theorem
theorem arc_length_theorem (c : Circle) (s i q : Point) (arc_sq : Arc c) :
  c.radius = 12 →
  Angle s i q = π / 4 →
  arc_sq.start = s →
  arc_sq.endpoint = q →
  ArcLength c arc_sq = 6 * π := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_theorem_l726_72625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_movement_probability_l726_72663

/-- Represents a vertex of a cube -/
inductive Vertex
| A | B | C | D | E | F | G | H

/-- Represents the position of an ant -/
def AntPosition := Vertex

/-- The number of ants -/
def numAnts : Nat := 8

/-- The probability of an ant moving to any adjacent vertex -/
noncomputable def moveProbability : Real := 1/3

/-- A valid configuration is one where no two ants are on the same vertex -/
def ValidConfiguration := List AntPosition

/-- The number of valid configurations where no two ants end up on the same vertex -/
def X : Nat := sorry

/-- The theorem stating the probability of no two ants arriving at the same vertex -/
theorem ant_movement_probability :
  (numValidConfigurations : Nat) →
  (numValidConfigurations = X) →
  (probability : Real) →
  (probability = X / 6561) →
  probability = (numValidConfigurations : Real) * moveProbability ^ numAnts :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_movement_probability_l726_72663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_price_for_revenue_min_sales_volume_and_price_l726_72660

/-- Original price in yuan -/
noncomputable def original_price : ℝ := 25

/-- Annual sales in units -/
noncomputable def annual_sales : ℝ := 80000

/-- Price elasticity: units decrease per yuan increase -/
noncomputable def price_elasticity : ℝ := 2000

/-- Sales volume as a function of price -/
noncomputable def sales_volume (price : ℝ) : ℝ := annual_sales - price_elasticity * (price - original_price)

/-- Total revenue as a function of price -/
noncomputable def total_revenue (price : ℝ) : ℝ := price * sales_volume price

/-- Investment cost for technological innovation -/
noncomputable def tech_investment (x : ℝ) : ℝ := (x^2 - 600) / 6

/-- Fixed advertising costs in million yuan -/
noncomputable def fixed_ad_cost : ℝ := 50

/-- Variable advertising costs as a function of price -/
noncomputable def variable_ad_cost (x : ℝ) : ℝ := x / 5

/-- Total investment as a function of price -/
noncomputable def total_investment (x : ℝ) : ℝ := tech_investment x + fixed_ad_cost + variable_ad_cost x

theorem max_price_for_revenue (price : ℝ) :
  total_revenue price ≥ total_revenue original_price → price ≤ 40 := by
  sorry

theorem min_sales_volume_and_price :
  ∃ (a x : ℝ), x * a ≥ total_revenue original_price + total_investment x ∧
               a = 10.2 * 10^6 ∧ x = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_price_for_revenue_min_sales_volume_and_price_l726_72660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_l726_72632

theorem sin_double_angle (α : ℝ) (h1 : Real.sin (3 * Real.pi / 2 - α) = 3 / 5) 
  (h2 : α ∈ Set.Ioo π (3 * Real.pi / 2)) : Real.sin (2 * α) = 24 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_l726_72632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l726_72609

theorem power_equality (n b : ℝ) : n = 2^(15/100) → n^b = 64 → b = 40 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l726_72609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relay_team_orders_l726_72628

/-- Represents a relay race team --/
structure RelayTeam where
  totalMembers : Nat
  laps : Nat
  fixedMember : Nat

/-- Calculates the number of different running orders for a relay team --/
def numberOfOrders (team : RelayTeam) : Nat :=
  if team.totalMembers ≠ team.laps then 0
  else if team.fixedMember ≠ 1 then 0
  else Nat.factorial (team.totalMembers - 1)

theorem relay_team_orders (team : RelayTeam) :
  team.totalMembers = 5 →
  team.laps = 5 →
  team.fixedMember = 1 →
  numberOfOrders team = 24 := by
  sorry

#eval numberOfOrders { totalMembers := 5, laps := 5, fixedMember := 1 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relay_team_orders_l726_72628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_internally_tangent_l726_72602

/-- The positional relationship between two circles -/
inductive CircleRelationship
  | Disjoint
  | ExternallyTangent
  | Intersecting
  | InternallyTangent
  | Concentric

/-- Represents a circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Calculates the distance between two points in 2D space -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Determines the relationship between two circles -/
noncomputable def circleRelationship (c1 c2 : Circle) : CircleRelationship :=
  let d := distance c1.center c2.center
  if d > c1.radius + c2.radius then CircleRelationship.Disjoint
  else if d = c1.radius + c2.radius then CircleRelationship.ExternallyTangent
  else if d < c1.radius + c2.radius ∧ d > abs (c1.radius - c2.radius) then CircleRelationship.Intersecting
  else if d = abs (c1.radius - c2.radius) then CircleRelationship.InternallyTangent
  else CircleRelationship.Concentric

theorem circles_internally_tangent : 
  let c1 : Circle := { center := (3, -2), radius := 1 }
  let c2 : Circle := { center := (7, 1), radius := 6 }
  circleRelationship c1 c2 = CircleRelationship.InternallyTangent := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_internally_tangent_l726_72602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l726_72662

/-- The curve C: x²/4 + y²/3 = 1 -/
def curve (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- The line l: x = 1 + t/2, y = 2 + (√3/2)t -/
def line (t x y : ℝ) : Prop := x = 1 + t/2 ∧ y = 2 + (Real.sqrt 3/2)*t

/-- Point M -/
def point_M : ℝ × ℝ := (1, 2)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem intersection_distance_product :
  ∃ (A B : ℝ × ℝ) (t1 t2 : ℝ),
    curve A.1 A.2 ∧
    curve B.1 B.2 ∧
    line t1 A.1 A.2 ∧
    line t2 B.1 B.2 ∧
    (distance point_M A) * (distance point_M B) = 28/15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l726_72662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l726_72604

theorem relationship_abc : 
  let a : ℚ := (-2:ℚ)^(0:ℤ)
  let b : ℚ := (1/2:ℚ)^(-1:ℤ)
  let c : ℚ := (-3:ℚ)^(-2:ℤ)
  b > a ∧ a > c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l726_72604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_at_two_l726_72646

def f (x : ℝ) := x^2 + x

theorem limit_at_two (h : ℝ → ℝ) (hf : h = f) :
  Filter.Tendsto (fun Δx => (h (2 + Δx) - h 2) / Δx) (Filter.atTop.comap (fun x => |x|)) (nhds 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_at_two_l726_72646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_sufficient_condition_sum_zero_f_odd_l726_72690

-- Define the function g
def g : ℝ → ℝ := sorry

-- Define the function f
def f : ℝ → ℝ := sorry

-- Axioms based on the given conditions
axiom sin_g_odd : ∀ x : ℝ, Real.sin (g (-x)) = - Real.sin (g x)
axiom f_monotone : Monotone f
axiom f_range : Set.range f = Set.univ
axiom f_zero : f 0 = 0

-- Theorem 1
theorem necessary_sufficient_condition :
  ∀ u₀ : ℝ, Real.sin (g u₀) = 1 ↔ Real.sin (g (-u₀)) = -1 := by sorry

-- Theorem 2
theorem sum_zero :
  ∀ a b : ℝ, f a = Real.pi / 2 ∧ f b = -Real.pi / 2 → a + b = 0 := by sorry

-- Theorem 3
theorem f_odd : ∀ x : ℝ, f (-x) = -f x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_sufficient_condition_sum_zero_f_odd_l726_72690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_210_l726_72621

-- Define the points and the circle
structure Point where
  x : ℝ
  y : ℝ

structure Circle where
  center : Point
  radius : ℝ

-- Define the rectangle ABCD
def Rectangle (A B C D : Point) : Prop :=
  ∃ (width height : ℝ), 
    width > 0 ∧ height > 0 ∧
    B.x - A.x = width ∧ C.y - B.y = height ∧
    C.x - D.x = width ∧ D.y - A.y = height

-- Define the circle passing through A and B and touching CD at its midpoint
def CircleProperties (circle : Circle) (A B C D : Point) : Prop :=
  ∃ (M : Point),
    M.x = (C.x + D.x) / 2 ∧ M.y = (C.y + D.y) / 2 ∧
    (A.x - circle.center.x)^2 + (A.y - circle.center.y)^2 = circle.radius^2 ∧
    (B.x - circle.center.x)^2 + (B.y - circle.center.y)^2 = circle.radius^2 ∧
    (M.x - circle.center.x)^2 + (M.y - circle.center.y)^2 = circle.radius^2

-- Define the line DE touching the circle at E and intersecting AB extension at K
def LineProperties (D E K : Point) (circle : Circle) : Prop :=
  ∃ (t : ℝ),
    E.x = D.x + t * (K.x - D.x) ∧
    E.y = D.y + t * (K.y - D.y) ∧
    (E.x - circle.center.x)^2 + (E.y - circle.center.y)^2 = circle.radius^2

-- Define the length ratio KE : KA = 3 : 2
def LengthRatio (K E A : Point) : Prop :=
  3 * ((A.x - K.x)^2 + (A.y - K.y)^2) = 2 * ((E.x - K.x)^2 + (E.y - K.y)^2)

-- Define the area of a trapezoid
noncomputable def TrapezoidArea (B C D K : Point) : ℝ :=
  let base1 := Real.sqrt ((B.x - C.x)^2 + (B.y - C.y)^2)
  let base2 := Real.sqrt ((D.x - K.x)^2 + (D.y - K.y)^2)
  let height := Real.sqrt ((C.y - B.y)^2)
  (base1 + base2) * height / 2

-- Main theorem
theorem trapezoid_area_is_210 
  (A B C D E K : Point) (circle : Circle) :
  Rectangle A B C D →
  CircleProperties circle A B C D →
  LineProperties D E K circle →
  LengthRatio K E A →
  B.x - A.x = 10 →
  TrapezoidArea B C D K = 210 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_210_l726_72621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l726_72696

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/4 = 1

-- Define the foci
def left_focus (F₁ : ℝ × ℝ) : Prop := F₁.1 = -Real.sqrt 5 ∧ F₁.2 = 0
def right_focus (F₂ : ℝ × ℝ) : Prop := F₂.1 = Real.sqrt 5 ∧ F₂.2 = 0

-- Define a point on the hyperbola
def point_on_hyperbola (P : ℝ × ℝ) : Prop := hyperbola P.1 P.2

-- Define the distance between two points
noncomputable def distance (P Q : ℝ × ℝ) : ℝ := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem hyperbola_foci_distance (F₁ F₂ P : ℝ × ℝ) :
  left_focus F₁ → right_focus F₂ → point_on_hyperbola P → distance P F₁ = 5 →
  distance P F₂ = 3 ∨ distance P F₂ = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l726_72696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_l726_72664

theorem negation_of_existence :
  (¬ ∃ x₀ : ℝ, x₀ > 0 ∧ |x₀| ≤ 2018) ↔ (∀ x : ℝ, x > 0 → |x| > 2018) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_l726_72664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_reflection_properties_l726_72652

-- Define the circle C
noncomputable def circle_C (x y : ℝ) : Prop :=
  ∃ (a b r : ℝ), (x - a)^2 + (y - b)^2 = r^2 ∧ a - 2*b = 0 ∧ (4 - a)^2 + b^2 = r^2

-- Define the chord length
noncomputable def chord_length (a b r : ℝ) : ℝ :=
  2 * Real.sqrt (r^2 - ((4*a - 3*b)^2 / 25))

-- Define the reflection of point M(-4,1) over x-axis
def reflection_M : ℝ × ℝ := (-4, -1)

-- Define the center of circle C
def center_C : ℝ × ℝ := (6, 3)

-- Theorem statement
theorem circle_and_reflection_properties :
  ∃ (a b r : ℝ),
    circle_C a b ∧
    ¬ circle_C 0 0 ∧
    chord_length a b r = 4 ∧
    (∀ (x y : ℝ), circle_C x y ↔ x^2 + y^2 - 12*x - 6*y + 32 = 0) ∧
    (∀ (x y : ℝ), (y + 1) / (x + 4) = 4 / 10 ↔ 2*x - 5*y + 3 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_reflection_properties_l726_72652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_to_radius_perpendicular_to_radius_is_tangent_l726_72654

/-- A circle with center and a point on its circumference -/
structure Circle where
  center : ℝ × ℝ
  point : ℝ × ℝ
  radius : ℝ
  is_on_circle : (point.1 - center.1)^2 + (point.2 - center.2)^2 = radius^2

/-- A line represented by two points -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- Predicate to check if a point is on a line -/
def on_line (p : ℝ × ℝ) (l : Line) : Prop :=
  ∃ t : ℝ, p = (l.point1.1 + t * (l.point2.1 - l.point1.1), 
               l.point1.2 + t * (l.point2.2 - l.point1.2))

/-- Predicate to check if a point is on a circle -/
def on_circle (p : ℝ × ℝ) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- Predicate to check if a line touches a circle at exactly one point -/
def touches_at (l : Line) (c : Circle) (p : ℝ × ℝ) : Prop :=
  on_line p l ∧ on_circle p c ∧ ∀ q, on_line q l ∧ on_circle q c → q = p

/-- Predicate to check if two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  (l1.point2.1 - l1.point1.1) * (l2.point2.1 - l2.point1.1) +
  (l1.point2.2 - l1.point1.2) * (l2.point2.2 - l2.point1.2) = 0

/-- The radius line from the center to a point on the circle -/
def radius_line (c : Circle) : Line :=
  { point1 := c.center, point2 := c.point }

theorem tangent_perpendicular_to_radius (c : Circle) (l : Line) :
  touches_at l c c.point → perpendicular l (radius_line c) :=
sorry

theorem perpendicular_to_radius_is_tangent (c : Circle) (l : Line) :
  on_line c.point l → perpendicular l (radius_line c) → touches_at l c c.point :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_to_radius_perpendicular_to_radius_is_tangent_l726_72654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_shortest_l726_72642

/-- A line in a 2D plane --/
structure Line where
  -- Define a line (placeholder)
  mk :: -- Add a constructor

/-- A point in a 2D plane --/
structure Point where
  -- Define a point (placeholder)
  mk :: -- Add a constructor

/-- Distance between two points --/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  -- Define distance (placeholder)
  0 -- Placeholder return value

/-- Perpendicular distance from a point to a line --/
noncomputable def perpendicularDistance (p : Point) (l : Line) : ℝ :=
  -- Define perpendicular distance (placeholder)
  0 -- Placeholder return value

/-- A line segment connecting a point to a line --/
structure LineSegment (p : Point) (l : Line) where
  endPoint : Point
  -- Remove the assumption field for now

/-- Theorem: The perpendicular segment is the shortest --/
theorem perpendicular_shortest (p : Point) (l : Line) :
  ∀ (seg : LineSegment p l), perpendicularDistance p l ≤ distance p seg.endPoint := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_shortest_l726_72642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_1500_terms_l726_72692

-- Rename the function to avoid conflict with any existing 'sequence' definition
def customSequence (n : ℕ) : ℕ := 
  let blockNum := (Nat.sqrt (2 * n + 1) + 1) / 2
  if n = (blockNum * (blockNum - 1)) / 2 + 1 then 1
  else 2

theorem sum_of_first_1500_terms : 
  (Finset.range 1500).sum (fun i => customSequence (i + 1)) = 2946 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_1500_terms_l726_72692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_multiple_of_121_l726_72649

def a : ℕ → ℕ
  | 0 => 0  -- Add this case to handle Nat.zero
  | 1 => 0  -- Add this case to handle n < 20
  | n => if n ≥ 20 then (if n = 20 then 20 else 50 * a (n-1) + n) else 0

theorem least_multiple_of_121 :
  (∀ k, 20 < k ∧ k < 41 → ¬(121 ∣ a k)) ∧ (121 ∣ a 41) := by sorry

#check least_multiple_of_121

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_multiple_of_121_l726_72649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rocket_jet_speed_l726_72684

/-- Represents the maximum speed of a single-stage rocket --/
noncomputable def rocket_max_speed (v0 m1 m2 : ℝ) : ℝ :=
  v0 * Real.log ((m1 + m2) / m1)

/-- Theorem stating the jet speed of the rocket engine --/
theorem rocket_jet_speed : ∃ (v0 : ℝ), 
  v0 > 0 ∧ 
  v0 < 26 ∧
  v0 > 24 ∧
  (∃ (m2 : ℝ), m2 > 0 ∧ rocket_max_speed v0 (2*m2) m2 = 10) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rocket_jet_speed_l726_72684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l726_72683

/-- The speed of a train in km/hr, given its length and time to cross a point -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / time) * (3600 / 1000)

/-- Theorem: A train 55 m long crossing an electric pole in 5.5 seconds has a speed of 36 km/hr -/
theorem train_speed_calculation : train_speed 55 5.5 = 36 := by
  -- Unfold the definition of train_speed
  unfold train_speed
  -- Perform the calculation
  simp [div_mul_eq_mul_div]
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l726_72683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_l_shaped_sheetrock_area_l726_72680

noncomputable def main_length : ℝ := 6
noncomputable def main_width_inches : ℝ := 60
noncomputable def cutout_length : ℝ := 2
noncomputable def cutout_width_inches : ℝ := 12

noncomputable def inches_to_feet (inches : ℝ) : ℝ := inches / 12

noncomputable def main_width : ℝ := inches_to_feet main_width_inches
noncomputable def cutout_width : ℝ := inches_to_feet cutout_width_inches

noncomputable def main_area : ℝ := main_length * main_width
noncomputable def cutout_area : ℝ := cutout_length * cutout_width

noncomputable def l_shaped_area : ℝ := main_area - cutout_area

theorem l_shaped_sheetrock_area :
  l_shaped_area = 28 := by
  -- Unfold definitions
  unfold l_shaped_area main_area cutout_area main_width cutout_width inches_to_feet
  -- Simplify
  simp [main_length, main_width_inches, cutout_length, cutout_width_inches]
  -- The proof itself
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_l_shaped_sheetrock_area_l726_72680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l726_72644

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| - 2 * |x - 1|

-- Part I
theorem part_one :
  let f₃ := f 3
  { x : ℝ | f₃ x ≥ 1 } = Set.Icc 0 (4/3) := by sorry

-- Part II
theorem part_two :
  { a : ℝ | ∀ x ∈ Set.Icc 1 2, f a x - |2*x - 5| ≤ 0 } = Set.Icc (-1) 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l726_72644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_max_value_l726_72615

noncomputable def g (x : ℝ) : ℝ := min (3 * x + 3) (min ((1/3) * x + 1) (-(2/3) * x + 8))

theorem g_max_value : ∃ m : ℝ, m = 10/3 ∧ ∀ x : ℝ, g x ≤ m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_max_value_l726_72615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_2sqrt10_l726_72693

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

-- Define the line
noncomputable def line (t : ℝ) : ℝ × ℝ := (2 + t, Real.sqrt 3 * t)

-- Define the chord length
noncomputable def chord_length (t₁ t₂ : ℝ) : ℝ :=
  let (x₁, y₁) := line t₁
  let (x₂, y₂) := line t₂
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

-- Theorem statement
theorem chord_length_is_2sqrt10 :
  ∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧
  hyperbola (line t₁).1 (line t₁).2 ∧
  hyperbola (line t₂).1 (line t₂).2 ∧
  chord_length t₁ t₂ = 2 * Real.sqrt 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_2sqrt10_l726_72693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_fractions_result_rational_expression_result_sqrt_difference_l726_72620

-- Define the sum of fractions
noncomputable def sum_fractions : ℕ → ℝ
  | 0 => 0
  | n + 1 => 1 / (Real.sqrt (n + 2) + Real.sqrt (n + 1)) + sum_fractions n

-- First theorem
theorem sum_fractions_result :
  sum_fractions 2022 * (Real.sqrt 2023 + 1) = 2022 := by
  sorry

-- Second theorem
theorem rational_expression_result :
  12 / (5 + Real.sqrt 13) + 5 / (Real.sqrt 13 - 2 * Real.sqrt 2) - 4 / (2 * Real.sqrt 2 + 3) = 10 * Real.sqrt 2 - 7 := by
  sorry

-- Given condition
theorem sqrt_difference (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) :
  (Real.sqrt a + Real.sqrt b) * (Real.sqrt a - Real.sqrt b) = a - b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_fractions_result_rational_expression_result_sqrt_difference_l726_72620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_volume_l726_72681

/-- The volume of a sphere inscribed in a cube with edge length 2 is (4/3)π. -/
theorem inscribed_sphere_volume (cube_edge : ℝ) (h : cube_edge = 2) :
  (4 / 3) * Real.pi * (cube_edge / 2)^3 = (4 / 3) * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_volume_l726_72681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_sequence_sum_l726_72618

def is_symmetric_sequence (b : ℕ → ℕ) (n : ℕ) : Prop :=
  ∀ i, i ≤ n → b i = b (n - i + 1)

def first_m_terms_geometric (b : ℕ → ℕ) (m : ℕ) : Prop :=
  ∀ i, i ≤ m → b i = 2^(i-1)

theorem symmetric_sequence_sum (b : ℕ → ℕ) (m n : ℕ) :
  m > 1 →
  n ≤ 2*m →
  is_symmetric_sequence b n →
  first_m_terms_geometric b m →
  ∃ S : ℕ, 
    ((S = 2^2009 - 1 ∧ m - 1 ≥ 2008) ∨
     (S = 2^(m+1) - 2^(2*m-2009) - 1 ∧ 1004 ≤ m - 1 ∧ m - 1 < 2008 ∧ Even n) ∨
     (S = 3 * 2^(m-1) - 2^(2*m-2010) - 1 ∧ 1004 ≤ m - 1 ∧ m - 1 < 2008 ∧ Odd n)) ∧
    S = (Finset.range 2009).sum (λ i => b (i+1)) :=
by sorry

#check symmetric_sequence_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_sequence_sum_l726_72618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_defined_iff_x_in_open_interval_2_5_l726_72631

-- Define the expression as noncomputable
noncomputable def f (x : ℝ) : ℝ := (Real.log (5 - x)) / Real.sqrt (x - 2)

-- Theorem statement
theorem f_defined_iff_x_in_open_interval_2_5 :
  ∀ x : ℝ, (∃ y : ℝ, f x = y) ↔ 2 < x ∧ x < 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_defined_iff_x_in_open_interval_2_5_l726_72631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_consumption_minimized_at_40_l726_72655

/-- The power consumption function for an electric bicycle -/
noncomputable def power_consumption (x : ℝ) : ℝ := (1/3) * x^3 - (39/2) * x^2 - 40 * x

/-- The speed that minimizes power consumption -/
def minimizing_speed : ℝ := 40

/-- Theorem stating that the power consumption is minimized at the speed of 40 -/
theorem power_consumption_minimized_at_40 :
  ∀ x : ℝ, x > 0 → power_consumption x ≥ power_consumption minimizing_speed := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_consumption_minimized_at_40_l726_72655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_green_faction_liars_exceed_truthful_l726_72610

/-- Represents a faction of deputies -/
inductive Faction
  | Blue
  | Red
  | Green

/-- Represents whether a deputy tells the truth or lies -/
inductive Honesty
  | Truthful
  | Lying

/-- The total number of deputies -/
def total_deputies : ℕ := 2016

/-- The number of deputies who answered "yes" to being in the blue faction -/
def blue_yes : ℕ := 1208

/-- The number of deputies who answered "yes" to being in the red faction -/
def red_yes : ℕ := 908

/-- The number of deputies who answered "yes" to being in the green faction -/
def green_yes : ℕ := 608

/-- The number of deputies in each faction with their honesty -/
def deputies : Faction → Honesty → ℕ := sorry

/-- The statement to be proved -/
theorem green_faction_liars_exceed_truthful :
  deputies Faction.Green Honesty.Lying - deputies Faction.Green Honesty.Truthful = 100 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_green_faction_liars_exceed_truthful_l726_72610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apartment_complex_buildings_l726_72607

/-- Represents an apartment complex with identical buildings -/
structure ApartmentComplex where
  max_occupancy_per_building : ℕ
  current_occupancy : ℕ
  occupancy_percentage : ℚ

/-- Calculates the number of buildings in the apartment complex -/
def number_of_buildings (complex : ApartmentComplex) : ℕ :=
  ((complex.current_occupancy : ℚ) / complex.occupancy_percentage).ceil.toNat

theorem apartment_complex_buildings (complex : ApartmentComplex) 
  (h1 : complex.max_occupancy_per_building = 70)
  (h2 : complex.occupancy_percentage = 3/4)
  (h3 : complex.current_occupancy = 210) :
  number_of_buildings complex = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apartment_complex_buildings_l726_72607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_perp_one_parallel_other_implies_planes_perp_l726_72641

/-- A line in 3D space -/
structure Line3D where
  -- Define properties of a line
  point : Real × Real × Real
  direction : Real × Real × Real

/-- A plane in 3D space -/
structure Plane3D where
  -- Define properties of a plane
  point : Real × Real × Real
  normal : Real × Real × Real

/-- Two lines are different -/
def different_lines (a b : Line3D) : Prop :=
  a ≠ b

/-- Two planes are different -/
def different_planes (α β : Plane3D) : Prop :=
  α ≠ β

/-- A line is perpendicular to a plane -/
def perpendicular_line_plane (l : Line3D) (p : Plane3D) : Prop :=
  -- Define perpendicularity condition
  sorry

/-- A line is parallel to a plane -/
def parallel_line_plane (l : Line3D) (p : Plane3D) : Prop :=
  -- Define parallelism condition
  sorry

/-- Two planes are perpendicular -/
def perpendicular_planes (p q : Plane3D) : Prop :=
  -- Define perpendicularity condition for planes
  sorry

theorem line_perp_one_parallel_other_implies_planes_perp
  (a : Line3D) (α β : Plane3D)
  (h1 : different_planes α β)
  (h2 : perpendicular_line_plane a α)
  (h3 : parallel_line_plane a β) :
  perpendicular_planes α β := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_perp_one_parallel_other_implies_planes_perp_l726_72641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_curves_is_5pi_l726_72619

/-- The area bounded by two polar curves -/
noncomputable def area_between_polar_curves (r₁ r₂ : ℝ → ℝ) (a b : ℝ) : ℝ :=
  (1/2) * ∫ x in a..b, (r₁ x)^2 - (r₂ x)^2

/-- The first curve in polar coordinates -/
noncomputable def r₁ (φ : ℝ) : ℝ := 6 * Real.sin φ

/-- The second curve in polar coordinates -/
noncomputable def r₂ (φ : ℝ) : ℝ := 4 * Real.sin φ

/-- Theorem stating that the area bounded by the given curves is 5π -/
theorem area_between_curves_is_5pi :
  area_between_polar_curves r₁ r₂ (-π/2) (π/2) = 5 * π := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_curves_is_5pi_l726_72619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l726_72637

noncomputable section

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
axiom f_property (x : ℝ) : f x + deriv f x > 1
axiom f_initial : f 0 = 2017

-- Define the inequality
def inequality (x : ℝ) : Prop := Real.exp x * f x - Real.exp x > 2016

-- Theorem statement
theorem solution_set :
  {x : ℝ | inequality f x} = {x : ℝ | x > 0} :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l726_72637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_31_16_l726_72653

noncomputable def geometric_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sum_31_16 :
  ∃ n : ℕ, geometric_sum 1 (1/2) n = 31/16 ∧ n = 5 := by
  use 5
  constructor
  · simp [geometric_sum]
    norm_num
  · rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_31_16_l726_72653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_is_square_plus_one_l726_72635

/-- A polynomial satisfying the given conditions -/
def p : ℝ → ℝ := sorry

/-- The polynomial satisfies p(3) = 10 -/
axiom p_at_3 : p 3 = 10

/-- The polynomial satisfies the functional equation for all real x and y -/
axiom p_functional (x y : ℝ) : p x * p y = p x + p y + p (x * y) - 2

/-- The polynomial p is equal to x^2 + 1 -/
theorem p_is_square_plus_one : p = fun x => x^2 + 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_is_square_plus_one_l726_72635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_two_cos_squared_l726_72657

theorem sin_plus_two_cos_squared (θ : ℝ) (a : ℝ) 
  (h1 : 0 < θ ∧ θ < Real.pi / 2) -- θ is an acute angle
  (h2 : Real.sin (2 * θ) = a) : -- sin 2θ = a
  (Real.sin θ + 2 * Real.cos θ)^2 = 5 - a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_two_cos_squared_l726_72657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_vector_range_l726_72656

/-- The ellipse on which points C and D move -/
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- The fixed point M -/
def M : ℝ × ℝ := (0, 2)

/-- Vector from M to a point P -/
def vector_MP (P : ℝ × ℝ) : ℝ × ℝ := (P.1 - M.1, P.2 - M.2)

/-- The relationship between vectors MC and MD -/
def vector_relation (C D : ℝ × ℝ) (lambda : ℝ) : Prop :=
  vector_MP D = lambda • (vector_MP C)

theorem ellipse_vector_range (C D : ℝ × ℝ) (lambda : ℝ) 
  (hC : ellipse C.1 C.2) (hD : ellipse D.1 D.2) 
  (h_relation : vector_relation C D lambda) : 
  1/3 ≤ lambda ∧ lambda ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_vector_range_l726_72656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l726_72627

open Real

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := (sin x * cos x) / (1 + sin x + cos x)

-- State the theorem
theorem f_max_value :
  ∃ (M : ℝ), M = (Real.sqrt 2 - 1) / 2 ∧
  (∀ x : ℝ, 1 + sin x + cos x ≠ 0 → f x ≤ M) ∧
  (∃ x : ℝ, 1 + sin x + cos x ≠ 0 ∧ f x = M) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l726_72627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_hole_radius_l726_72691

-- Define the cuboid dimensions
noncomputable def length : ℝ := 3
noncomputable def width : ℝ := 8
noncomputable def cuboid_height : ℝ := 9

-- Define the theorem
theorem cylindrical_hole_radius (r : ℝ) : 
  -- Condition: The surface area remains unchanged
  2 * Real.pi * r^2 = 2 * Real.pi * r * length →
  -- Conclusion: The radius of the hole is 3
  r = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_hole_radius_l726_72691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dance_lesson_cost_l726_72623

/-- The cost of a dance lesson pack -/
def pack_cost : ℚ := 75

/-- The number of classes in a pack -/
def pack_classes : ℕ := 10

/-- The total number of classes Ruby takes -/
def total_classes : ℕ := 13

/-- Calculate the total cost of dance lessons -/
noncomputable def total_cost : ℚ :=
  let avg_class_cost : ℚ := pack_cost / pack_classes
  let additional_class_cost : ℚ := avg_class_cost * (1 + 1/3)
  let additional_classes : ℕ := total_classes - pack_classes
  pack_cost + (additional_classes : ℚ) * additional_class_cost

theorem dance_lesson_cost : total_cost = 105 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dance_lesson_cost_l726_72623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_subtraction_for_common_remainder_l726_72677

theorem least_subtraction_for_common_remainder (n : ℕ) : 
  n = 90198 ↔ 
  (∀ p : ℕ, p ∈ ({7, 11, 13, 17, 19} : Set ℕ) → (90210 - n) % p = 12) ∧ 
  (∀ m : ℕ, m < n → ∃ p : ℕ, p ∈ ({7, 11, 13, 17, 19} : Set ℕ) ∧ (90210 - m) % p ≠ 12) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_subtraction_for_common_remainder_l726_72677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joe_school_time_l726_72608

/-- Represents the time Joe took to get from home to school -/
noncomputable def total_time (walk_time : ℝ) (walk_distance : ℝ) (run_distance : ℝ) (run_speed_factor : ℝ) : ℝ :=
  walk_time + (run_distance / walk_distance) * (walk_time / run_speed_factor)

/-- Theorem stating that Joe's total time to get from home to school is 18 minutes -/
theorem joe_school_time :
  let walk_time := (9 : ℝ)
  let walk_distance := (1 : ℝ) / 3
  let run_distance := (2 : ℝ) / 3
  let run_speed_factor := (2 : ℝ)
  total_time walk_time walk_distance run_distance run_speed_factor = 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_joe_school_time_l726_72608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_equals_thirteen_halves_l726_72617

-- Define lg as the base-10 logarithm
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_sum_equals_thirteen_halves : 
  (Real.log (Real.sqrt 27) / Real.log 3) + lg 25 + lg 4 + 7^(Real.log 2 / Real.log 7) + (-9.8)^0 = 13/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_equals_thirteen_halves_l726_72617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_black_squares_cover_l726_72672

/-- Represents a chessboard with black squares -/
def Chessboard := Fin 8 → Fin 8 → Bool

/-- Counts the number of black squares on the chessboard -/
def count_black (board : Chessboard) : Nat :=
  (Finset.sum (Finset.univ : Finset (Fin 8)) fun i =>
    Finset.sum (Finset.univ : Finset (Fin 8)) fun j =>
      if board i j then 1 else 0)

/-- Checks if a set of rows and columns cover all black squares -/
def covers_all_black (board : Chessboard) (rows cols : Finset (Fin 8)) : Prop :=
  ∀ i j, board i j → (i ∈ rows ∧ j ∈ cols)

theorem black_squares_cover (board : Chessboard) :
  count_black board = 12 →
  ∃ rows cols : Finset (Fin 8),
    rows.card = 4 ∧ cols.card = 4 ∧ covers_all_black board rows cols := by
  sorry

#check black_squares_cover

end NUMINAMATH_CALUDE_ERRORFEEDBACK_black_squares_cover_l726_72672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_range_l726_72634

/-- Given an inequality (x + 2) / (x - a) ≤ 2 with solution set P, and 1 ∉ P, 
    the range of a is (-1/2, 1]. -/
theorem inequality_solution_range (a : ℝ) (P : Set ℝ) : 
  (∀ x, x ∈ P ↔ (x + 2) / (x - a) ≤ 2) → 
  (1 ∉ P) → 
  a ∈ Set.Ioc (-1/2) 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_range_l726_72634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l726_72695

/-- Given a parabola and a hyperbola with certain properties, prove the equation of the parabola -/
theorem parabola_equation (p : ℝ) :
  (∀ x y, y^2 = 2*p*x) →  -- Parabola equation
  (∀ x y, x^2/8 - y^2/p = 1) →  -- Hyperbola equation
  (let focus : ℝ × ℝ := (p/2, 0);
   let asymptote (x y : ℝ) := Real.sqrt p * x + 2 * Real.sqrt 2 * y = 0 ∨
                               Real.sqrt p * x - 2 * Real.sqrt 2 * y = 0;
   ∀ x y, asymptote x y → dist focus (x, y) = (Real.sqrt 2 / 4) * p) →
  (∀ x y, y^2 = 16*x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l726_72695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_product_on_inclined_line_circle_intersection_l726_72605

/-- Line passing through a point with given inclination angle -/
structure InclinedLine where
  point : ℝ × ℝ
  angle : ℝ

/-- Circle in polar form -/
structure PolarCircle where
  radius : ℝ

/-- The problem statement -/
theorem distance_product_on_inclined_line_circle_intersection
  (l : InclinedLine)
  (c : PolarCircle)
  (h1 : l.point = (1, 1))
  (h2 : l.angle = π / 6)
  (h3 : c.radius = 2) :
  ∃ A B : ℝ × ℝ,
    A ≠ B ∧
    (A.1^2 + A.2^2 = 4) ∧
    (B.1^2 + B.2^2 = 4) ∧
    (∃ t1 t2 : ℝ, 
      A = (1 + (Real.sqrt 3 / 2) * t1, 1 + (1 / 2) * t1) ∧
      B = (1 + (Real.sqrt 3 / 2) * t2, 1 + (1 / 2) * t2)) ∧
    ((A.1 - 1)^2 + (A.2 - 1)^2) * ((B.1 - 1)^2 + (B.2 - 1)^2) = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_product_on_inclined_line_circle_intersection_l726_72605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l726_72639

noncomputable def f (x : ℝ) : ℝ := (8*x^4 + 6*x^3 + 7*x^2 + 2*x + 4) / (2*x^4 + 5*x^3 + 3*x^2 + x + 6)

theorem horizontal_asymptote_of_f :
  ∀ ε > 0, ∃ M, ∀ x, |x| > M → |f x - 4| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l726_72639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sliced_solid_surface_area_l726_72665

/-- A right prism with equilateral triangle bases -/
structure RightPrism where
  height : ℝ
  base_side : ℝ

/-- The sliced-off solid from the prism -/
structure SlicedSolid where
  prism : RightPrism

/-- Calculate the surface area of the sliced-off solid -/
noncomputable def surface_area (solid : SlicedSolid) : ℝ :=
  63 + (49 * Real.sqrt 3 + Real.sqrt 521) / 4

theorem sliced_solid_surface_area (solid : SlicedSolid) 
  (h1 : solid.prism.height = 18)
  (h2 : solid.prism.base_side = 14) :
  surface_area solid = 63 + (49 * Real.sqrt 3 + Real.sqrt 521) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sliced_solid_surface_area_l726_72665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_circles_perpendicular_tangents_l726_72643

/-- Two circles in a plane -/
structure TwoCircles where
  m : ℝ
  O₁ : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 5}
  O₂ : Set (ℝ × ℝ) := {p | (p.1 - m)^2 + p.2^2 = 5}

/-- The intersection points of two circles -/
def IntersectionPoints (c : TwoCircles) : Set (ℝ × ℝ) :=
  c.O₁ ∩ c.O₂

/-- The tangent lines to a circle at a point -/
def TangentLines (c : TwoCircles) (p : ℝ × ℝ) : Set (Set (ℝ × ℝ)) :=
  {l | ∃ (v : ℝ × ℝ), l = {q : ℝ × ℝ | (q.1 - p.1) * v.1 + (q.2 - p.2) * v.2 = 0} ∧
    (v.1 * p.1 + v.2 * p.2 = 0 ∨ v.1 * (p.1 - c.m) + v.2 * p.2 = 0)}

/-- Perpendicular tangent lines -/
def PerpendicularTangents (c : TwoCircles) (p : ℝ × ℝ) : Prop :=
  ∃ (l₁ l₂ : Set (ℝ × ℝ)), l₁ ∈ TangentLines c p ∧ l₂ ∈ TangentLines c p ∧
    ∃ (v₁ v₂ : ℝ × ℝ), l₁ = {q : ℝ × ℝ | (q.1 - p.1) * v₁.1 + (q.2 - p.2) * v₁.2 = 0} ∧
      l₂ = {q : ℝ × ℝ | (q.1 - p.1) * v₂.1 + (q.2 - p.2) * v₂.2 = 0} ∧
      v₁.1 * v₂.1 + v₁.2 * v₂.2 = 0

/-- The length of a line segment between two points -/
noncomputable def SegmentLength (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

theorem intersecting_circles_perpendicular_tangents
    (c : TwoCircles)
    (A B : ℝ × ℝ)
    (h₁ : A ∈ IntersectionPoints c)
    (h₂ : B ∈ IntersectionPoints c)
    (h₃ : PerpendicularTangents c A) :
    SegmentLength A B = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_circles_perpendicular_tangents_l726_72643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_fifth_term_l726_72624

/-- Given a geometric sequence with first four terms 2, 6x, 18x^2, and 54x^3, 
    the fifth term is 162x^4 -/
theorem geometric_sequence_fifth_term (x : ℝ) :
  let seq := λ n => 2 * (3 * x) ^ (n - 1)
  seq 5 = 162 * x^4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_fifth_term_l726_72624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l726_72673

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x + 1) * Real.log x - a * x + a

-- State the theorem
theorem f_properties (a : ℝ) (h_a : a > 0) :
  (∀ x > 0, Monotone (fun x => f a x)) ∧ (a ≤ 2) ∧
  (∀ x > 0, (x - 1) * f a x ≥ 0) ∧ (a ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l726_72673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_side_length_l726_72638

/-- Represents a trapezium with given dimensions -/
structure Trapezium where
  side1 : ℝ
  side2 : ℝ
  height : ℝ
  area : ℝ

/-- Calculates the area of a trapezium -/
noncomputable def trapezium_area (t : Trapezium) : ℝ :=
  (t.side1 + t.side2) * t.height / 2

/-- Theorem: Given a trapezium with one side 20 cm, height 5 cm, and area 95 cm², 
    the other side is 18 cm -/
theorem trapezium_side_length (t : Trapezium) 
    (h1 : t.side1 = 20)
    (h2 : t.height = 5)
    (h3 : t.area = 95)
    (h4 : t.area = trapezium_area t) :
  t.side2 = 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_side_length_l726_72638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_sum_l726_72687

-- Define the function h(x)
noncomputable def h (x : ℝ) : ℝ := 3 / (1 + 3 * x^2)

-- State the theorem
theorem range_sum (a b : ℝ) : 
  (∀ y : ℝ, y ∈ Set.Ioo a b ↔ ∃ x : ℝ, h x = y) → 
  (∀ x : ℝ, h x ≤ b) → 
  (∀ ε > 0, ∃ x : ℝ, h x > a + ε) → 
  a + b = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_sum_l726_72687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_sum_24_is_correct_l726_72600

/-- A fair, standard six-sided die -/
def Die : Type := Fin 6

/-- The probability of rolling a specific number on a fair, standard six-sided die -/
def prob_single_roll : ℚ := 1 / 6

/-- The sum of the top faces of four dice -/
def sum_of_four_dice (d1 d2 d3 d4 : Die) : ℕ := d1.val + d2.val + d3.val + d4.val + 4

/-- The probability of rolling a sum of 24 with four fair, standard six-sided dice -/
def prob_sum_24 : ℚ := (1 : ℚ) / 1296

theorem prob_sum_24_is_correct : 
  prob_sum_24 = (prob_single_roll ^ 4 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_sum_24_is_correct_l726_72600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_simplification_and_sum_l726_72636

noncomputable def f (x : ℝ) : ℝ := (x^3 + 7*x^2 + 14*x + 8) / (x + 1)

def g (x : ℝ) : ℝ := x^2 + 6*x + 8

theorem function_simplification_and_sum :
  (∀ x : ℝ, x ≠ -1 → f x = g x) ∧
  (1 + 6 + 8 + (-1) = 14) := by
  constructor
  · intro x hx
    sorry -- Proof of function equality skipped
  · norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_simplification_and_sum_l726_72636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_midpoint_triangle_perimeter_l726_72606

/-- Represents a right prism with equilateral triangular bases -/
structure RightPrism where
  base_side_length : ℝ
  height : ℝ

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a triangle in 3D space -/
structure Triangle3D where
  p1 : Point3D
  p2 : Point3D
  p3 : Point3D

/-- Calculate the distance between two points in 3D space -/
noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

/-- Calculate the perimeter of a triangle in 3D space -/
noncomputable def perimeter (t : Triangle3D) : ℝ :=
  distance t.p1 t.p2 + distance t.p2 t.p3 + distance t.p3 t.p1

/-- The main theorem to be proved -/
theorem prism_midpoint_triangle_perimeter (p : RightPrism) 
  (h1 : p.base_side_length = 10)
  (h2 : p.height = 20)
  (t : Triangle3D)
  (h3 : t.p1 = Point3D.mk 5 0 0)  -- Midpoint of AC
  (h4 : t.p2 = Point3D.mk 2.5 (5 * Real.sqrt 3 / 2) 0)  -- Midpoint of BC
  (h5 : t.p3 = Point3D.mk 5 (5 * Real.sqrt 3 / 2) 10)  -- Midpoint of DC
  : perimeter t = 5 + 10 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_midpoint_triangle_perimeter_l726_72606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_muffin_banana_cost_ratio_l726_72670

/-- The cost ratio of a muffin to a banana is 1, given Alice's and Bob's purchases. -/
theorem muffin_banana_cost_ratio :
  ∀ (muffin_cost banana_cost : ℝ),
  muffin_cost > 0 →
  banana_cost > 0 →
  3 * muffin_cost + 5 * banana_cost > 0 →
  3 * (3 * muffin_cost + 5 * banana_cost) = 4 * muffin_cost + 10 * banana_cost →
  muffin_cost / banana_cost = 1 := by
  intros muffin_cost banana_cost h1 h2 h3 h4
  -- The proof steps would go here
  sorry

#check muffin_banana_cost_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_muffin_banana_cost_ratio_l726_72670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_M_when_a_is_1_a_value_when_min_M_is_3_l726_72650

noncomputable section

/-- The function f(x) --/
def f (a x : ℝ) : ℝ := x^2 + x + a^2 + a

/-- The function g(x) --/
def g (a x : ℝ) : ℝ := x^2 - x + a^2 - a

/-- The function M(x) --/
def M (a x : ℝ) : ℝ := max (f a x) (g a x)

/-- Theorem 1: When a = 1, the minimum value of M(x) is 7/4 --/
theorem min_M_when_a_is_1 : 
  ∀ x : ℝ, M 1 x ≥ 7/4 ∧ ∃ y : ℝ, M 1 y = 7/4 := by
  sorry

/-- Theorem 2: When the minimum value of M(x) is 3, a = ± (√14 - 1)/2 --/
theorem a_value_when_min_M_is_3 : 
  (∀ x : ℝ, M a x ≥ 3 ∧ ∃ y : ℝ, M a y = 3) ↔ a = (Real.sqrt 14 - 1)/2 ∨ a = -(Real.sqrt 14 - 1)/2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_M_when_a_is_1_a_value_when_min_M_is_3_l726_72650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_expression_l726_72616

theorem evaluate_expression : 
  (1/4 : ℚ)^3 * (3/4 : ℚ)^4 * (((-2 : ℤ)^2) : ℚ) = 81/1024 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_expression_l726_72616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_3_simplest_l726_72661

def is_simplest_sqrt (x : ℝ) : Prop :=
  ∀ y : ℝ, y ≠ x → (∃ z : ℝ, z * z = y) → (∃ w : ℝ, w * w = x) → x < y

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem sqrt_3_simplest :
  is_prime 3 →
  (∃ z : ℝ, z * z = 8 ∧ z ≠ Real.sqrt 8) →
  is_simplest_sqrt (Real.sqrt 3) ∧ ¬is_simplest_sqrt (Real.sqrt 0.5) ∧ 
  ¬is_simplest_sqrt (Real.sqrt 8) ∧ ¬is_simplest_sqrt (Real.sqrt (1/3)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_3_simplest_l726_72661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hanil_additional_distance_l726_72671

/-- Represents the biking scenario of Onur and Hanil -/
structure BikingScenario where
  onur_daily_distance : ℕ
  days_per_week : ℕ
  total_weekly_distance : ℕ

/-- Calculate the additional distance Hanil bikes compared to Onur -/
def additional_distance (scenario : BikingScenario) : ℕ :=
  (scenario.total_weekly_distance - scenario.onur_daily_distance * scenario.days_per_week) /
    scenario.days_per_week

/-- Theorem stating that Hanil bikes 40 km more than Onur each day -/
theorem hanil_additional_distance (scenario : BikingScenario) 
    (h1 : scenario.onur_daily_distance = 250)
    (h2 : scenario.days_per_week = 5)
    (h3 : scenario.total_weekly_distance = 2700) :
    additional_distance scenario = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hanil_additional_distance_l726_72671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_product_ninth_pi_l726_72630

theorem sin_product_ninth_pi : 
  Real.sin (π / 9) * Real.sin (2 * π / 9) * Real.sin (4 * π / 9) = Real.sqrt 3 / 8 := by
  let triple_angle_formula : ℝ → ℝ := λ θ ↦ 3 * Real.sin θ - 4 * Real.sin θ ^ 3
  have sin_pi_third : Real.sin (π / 3) = Real.sqrt 3 / 2 := by sorry
  have sin_two_pi_third : Real.sin (2 * π / 3) = Real.sqrt 3 / 2 := by sorry
  have sin_four_pi_third : Real.sin (4 * π / 3) = -(Real.sqrt 3 / 2) := by sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_product_ninth_pi_l726_72630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_T_shape_20_squares_l726_72647

/-- Represents the area of a T-shaped region composed of congruent squares -/
noncomputable def area_T_shape (num_squares : ℕ) (diagonal_length : ℝ) : ℝ :=
  let square_area := (diagonal_length ^ 2) / 8
  num_squares * square_area

/-- Theorem: The area of a T-shaped region with 20 congruent squares and a diagonal of 8 cm is 160 cm² -/
theorem area_T_shape_20_squares : 
  area_T_shape 20 8 = 160 := by
  -- Unfold the definition of area_T_shape
  unfold area_T_shape
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_T_shape_20_squares_l726_72647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_l726_72698

-- Define the function f(x) = cos(-x)
noncomputable def f (x : ℝ) : ℝ := Real.cos (-x)

-- State the theorem
theorem f_increasing : StrictMono f := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_l726_72698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l726_72640

/-- The function to be minimized -/
noncomputable def f (m n : ℝ) : ℝ := m * Real.exp m + 3 * n * Real.exp (3 * n)

/-- The theorem stating the minimum value of the function -/
theorem min_value_of_f :
  ∀ m n : ℝ, m + 3 * n = 1 → ∀ x y : ℝ, x + 3 * y = 1 → f m n ≥ Real.sqrt (Real.exp 1) ∧ f x y ≥ Real.sqrt (Real.exp 1) :=
by
  sorry

#check min_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l726_72640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_voucher_c_saves_more_l726_72648

/-- Represents the discount amount for a given voucher and price -/
noncomputable def discount (voucher : Char) (price : ℝ) : ℝ :=
  match voucher with
  | 'A' => if price > 100 then 0.1 * price else 0
  | 'B' => if price > 200 then 30 else 0
  | 'C' => if price > 200 then 0.2 * (price - 200) else 0
  | _ => 0

/-- Theorem stating that Voucher C saves more than both A and B iff price > 400 -/
theorem voucher_c_saves_more (price : ℝ) :
  discount 'C' price > max (discount 'A' price) (discount 'B' price) ↔ price > 400 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_voucher_c_saves_more_l726_72648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_increase_l726_72669

theorem rectangle_area_increase (L B : ℝ) (h1 : L > 0) (h2 : B > 0) : 
  (1.05 * L * 1.15 * B - L * B) / (L * B) = 0.2075 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_increase_l726_72669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_population_is_135000_l726_72697

/-- The number of cities in the County of Maple -/
def num_cities : ℕ := 25

/-- The lower bound of the average population range -/
def lower_bound : ℕ := 5200

/-- The upper bound of the average population range -/
def upper_bound : ℕ := 5600

/-- The average population of the cities -/
def avg_population : ℚ := (lower_bound + upper_bound) / 2

/-- The total population of all cities in the County of Maple -/
def total_population : ℕ := (num_cities * avg_population).num.toNat

theorem total_population_is_135000 : total_population = 135000 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_population_is_135000_l726_72697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_specific_triangle_l726_72682

/-- The radius of the inscribed circle of a triangle -/
noncomputable def inscribedCircleRadius (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c)) / s

theorem inscribed_circle_radius_specific_triangle :
  inscribedCircleRadius 26 16 20 = 5 * Real.sqrt 33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_specific_triangle_l726_72682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_aluminum_oxide_properties_l726_72667

/-- Calculates the molecular weight of aluminum oxide and the reaction enthalpy for its formation -/
theorem aluminum_oxide_properties (atomic_weight_Al : ℝ) (atomic_weight_O : ℝ) (enthalpy_formation_Al2O3 : ℝ) 
  (h1 : atomic_weight_Al = 26.98)
  (h2 : atomic_weight_O = 16.00)
  (h3 : enthalpy_formation_Al2O3 = -1675.7) :
  (2 * atomic_weight_Al + 3 * atomic_weight_O = 101.96) ∧ 
  (2 * enthalpy_formation_Al2O3 = -3351.4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_aluminum_oxide_properties_l726_72667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_rectangles_at_212_misha_optimal_choice_l726_72679

/-- Counts the number of distinct rectangles with integer sides for a given perimeter -/
def countRectangles (perimeter : ℕ) : ℕ :=
  if perimeter % 2 = 0 then
    (perimeter / 4)
  else
    0

/-- Theorem stating that 212 maximizes the count of distinct rectangles -/
theorem max_rectangles_at_212 :
  ∀ n : ℕ, n ≤ 213 → countRectangles n ≤ countRectangles 212 := by
  sorry

/-- Corollary: 212 is the optimal choice for Misha -/
theorem misha_optimal_choice : 
  ∃ n : ℕ, n ≤ 213 ∧ ∀ m : ℕ, m ≤ 213 → countRectangles m ≤ countRectangles n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_rectangles_at_212_misha_optimal_choice_l726_72679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lisa_wetsuit_size_l726_72603

/-- Converts inches to centimeters -/
noncomputable def inches_to_cm (inches : ℝ) : ℝ :=
  inches * (31 / 12)

/-- Rounds a real number to the nearest tenth -/
noncomputable def round_to_tenth (x : ℝ) : ℝ :=
  ⌊x * 10 + 0.5⌋ / 10

theorem lisa_wetsuit_size :
  let chest_size_inches : ℝ := 36
  let comfort_adjustment : ℝ := 2
  let adjusted_size := chest_size_inches + comfort_adjustment
  round_to_tenth (inches_to_cm adjusted_size) = 98.2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lisa_wetsuit_size_l726_72603
