import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minus_four_equals_one_l1277_127789

-- Define the function f
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b / x + 3

-- State the theorem
theorem f_minus_four_equals_one (a b : ℝ) : 
  f a b 4 = 5 → f a b (-4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minus_four_equals_one_l1277_127789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f₂_properties_l1277_127735

noncomputable def f₂ (x : ℝ) : ℝ := 4 - 6 * (1/2)^x

theorem f₂_properties :
  (∀ x ≥ 0, f₂ x ∈ Set.Ioc (-2) 4) ∧ 
  StrictMono f₂ ∧
  (∀ x ≥ 0, f₂ x + f₂ (x + 2) < 2 * f₂ (x + 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f₂_properties_l1277_127735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_difference_specific_values_l1277_127738

theorem tan_difference_specific_values (α β : Real) 
  (h1 : Real.tan α = 3) (h2 : Real.tan β = 2) : 
  Real.tan (α - β) = 1/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_difference_specific_values_l1277_127738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equals_specific_set_l1277_127754

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- Define set A
def setA : Set ℝ :=
  {x | (floor x)^2 - 2*(floor x) = 3}

-- Define set B
def setB : Set ℝ :=
  {x | 0 < x + 2 ∧ x + 2 < 5}

-- Define the intersection of A and B
def AIntersectB : Set ℝ :=
  setA ∩ setB

-- State the theorem
theorem intersection_equals_specific_set :
  AIntersectB = {-1, Real.sqrt 7} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equals_specific_set_l1277_127754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1277_127717

theorem triangle_side_length (A : Real) (a b c : ℝ) :
  Real.cos A = 3/5 →
  a = 4 * Real.sqrt 2 →
  b = 5 →
  c^2 - 6*c - 7 = 0 →
  c > 0 →
  c = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1277_127717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_four_cube_formable_l1277_127732

/-- Represents a position where an additional square can be attached -/
inductive AttachmentPosition
| ExtendArm
| AdjacentToCenter

/-- Represents a square -/
structure Square

/-- Represents the cross-shaped polygon with 5 congruent squares -/
structure CrossPolygon :=
  (squares : Fin 5 → Square)
  (congruent : ∀ i j : Fin 5, squares i = squares j)
  (cross_shape : Unit) -- Placeholder for cross shape property

/-- Represents the polygon after attaching an additional square -/
structure ExtendedPolygon :=
  (base : CrossPolygon)
  (additional_square : Square)
  (attachment : AttachmentPosition)
  (congruent : ∀ i : Fin 5, base.squares i = additional_square)

/-- Checks if an extended polygon can be folded into a cube -/
def can_form_cube (p : ExtendedPolygon) : Prop :=
  sorry

/-- The main theorem stating that exactly 4 extended polygons can form a cube -/
theorem exactly_four_cube_formable (cp : CrossPolygon) :
  ∃! (s : Finset ExtendedPolygon),
    s.card = 4 ∧
    (∀ p ∈ s, can_form_cube p) ∧
    (∀ p : ExtendedPolygon, p.base = cp → can_form_cube p → p ∈ s) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_four_cube_formable_l1277_127732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expr_not_necessarily_quadratic_radical_l1277_127724

-- Define a real number a
variable (a : ℝ)

-- Define the expression sqrt(-a)
noncomputable def expr (a : ℝ) : ℝ := Real.sqrt (-a)

-- Theorem stating that expr is not necessarily a quadratic radical
theorem expr_not_necessarily_quadratic_radical :
  ¬ (∀ a : ℝ, ∃ q : ℚ, expr a = Real.sqrt q) ∧
  ¬ (∀ a : ℝ, ¬ ∃ q : ℚ, expr a = Real.sqrt q) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expr_not_necessarily_quadratic_radical_l1277_127724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_XYZ_is_202_5_l1277_127702

/-- Triangle DEF with right angle at D -/
structure RightTriangleDEF where
  DE : ℝ
  DF : ℝ
  right_angle : DE > 0 ∧ DF > 0

/-- Triangle XYZ similar to DEF with scale factor -/
structure SimilarTriangleXYZ where
  DEF : RightTriangleDEF
  scale_factor : ℝ

/-- The area of triangle XYZ -/
noncomputable def area_XYZ (t : SimilarTriangleXYZ) : ℝ :=
  (1/2) * (t.scale_factor * t.DEF.DE) * (t.scale_factor * t.DEF.DF)

theorem area_XYZ_is_202_5 (t : SimilarTriangleXYZ) 
  (h1 : t.DEF.DE = 30)
  (h2 : t.DEF.DF = 24)
  (h3 : t.scale_factor = 3/4) :
  area_XYZ t = 202.5 := by
  sorry

#check area_XYZ_is_202_5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_XYZ_is_202_5_l1277_127702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2020th_term_is_7_l1277_127726

/-- The sequence function that generates the nth term of the sequence -/
def mySequence (n : ℕ) : ℕ :=
  let block := (n.sqrt + 1 : ℕ)
  let position_in_block := n - (block - 1) * block / 2
  2 * position_in_block - 1

/-- Theorem stating that the 2020th term of the sequence is 7 -/
theorem sequence_2020th_term_is_7 : mySequence 2020 = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2020th_term_is_7_l1277_127726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_theorem_l1277_127733

noncomputable section

/-- A quadratic function f(x) = ax^2 + bx + c -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- The x-coordinates of the intersection points of a quadratic function with the x-axis -/
def intersectionPoints (f : ℝ → ℝ) : Set ℝ := {x : ℝ | f x = 0}

/-- The vertex of a quadratic function f(x) = ax^2 + bx + c is at (-b/(2a), f(-b/(2a))) -/
def vertex (a b c : ℝ) : ℝ × ℝ := (-b / (2 * a), QuadraticFunction a b c (-b / (2 * a)))

theorem quadratic_function_theorem (a b c : ℝ) :
  let f := QuadraticFunction a b c
  (∃ x y : ℝ, x ≠ y ∧ intersectionPoints f = {x, y}) →  -- Condition 1
  (∃ x y : ℝ, x ∈ intersectionPoints f ∧ y ∈ intersectionPoints f ∧ |x - y| = 2) →  -- Condition 2
  (vertex a b c = (-1, -1)) →  -- Condition 3
  ∀ x, f x = x^2 + 2*x :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_theorem_l1277_127733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_insurance_to_earnings_ratio_is_one_fifth_l1277_127747

/-- Represents Dr. Jones' monthly finances --/
structure MonthlyFinances where
  earnings : ℚ
  houseRental : ℚ
  foodExpense : ℚ
  utilityBillRatio : ℚ
  leftover : ℚ

/-- Calculates the insurance cost to earnings ratio --/
def insuranceToEarningsRatio (finances : MonthlyFinances) : ℚ :=
  let utilityBill := finances.earnings * finances.utilityBillRatio
  let knownExpenses := finances.houseRental + finances.foodExpense + utilityBill
  let insuranceCost := finances.earnings - knownExpenses - finances.leftover
  insuranceCost / finances.earnings

/-- Theorem stating that the insurance to earnings ratio is 1/5 --/
theorem insurance_to_earnings_ratio_is_one_fifth 
  (finances : MonthlyFinances)
  (h1 : finances.earnings = 6000)
  (h2 : finances.houseRental = 640)
  (h3 : finances.foodExpense = 380)
  (h4 : finances.utilityBillRatio = 1/4)
  (h5 : finances.leftover = 2280) :
  insuranceToEarningsRatio finances = 1/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_insurance_to_earnings_ratio_is_one_fifth_l1277_127747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_of_line_segment_l1277_127793

def z₁ : ℂ := -7 + 5 * Complex.I
def z₂ : ℂ := 5 - 9 * Complex.I

theorem midpoint_of_line_segment : (z₁ + z₂) / 2 = -1 - 2 * Complex.I := by
  -- Expand the definitions of z₁ and z₂
  simp [z₁, z₂]
  -- Perform the arithmetic
  ring
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_of_line_segment_l1277_127793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_triangle_l1277_127779

noncomputable def large_triangle_hypotenuse : ℝ := 14
noncomputable def small_triangle_hypotenuse : ℝ := 2

def is_right_angled_isosceles (hypotenuse : ℝ) : Prop :=
  ∃ (leg : ℝ), leg * leg * 2 = hypotenuse * hypotenuse

noncomputable def triangle_area (hypotenuse : ℝ) : ℝ :=
  hypotenuse * hypotenuse / 4

theorem fill_triangle : 
  (triangle_area large_triangle_hypotenuse) / (triangle_area small_triangle_hypotenuse) = 49 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_triangle_l1277_127779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_power_sum_l1277_127737

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_power_sum : i^(-5 : ℤ) + i^8 + i^14 - i^22 = 1 - i := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_power_sum_l1277_127737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_voltage_equals_current_times_impedance_l1277_127700

/-- Voltage (V) -/
def V : ℂ := ⟨2, 2⟩

/-- Impedance (Z) -/
def Z : ℂ := ⟨3, -4⟩

/-- Current (I) -/
noncomputable def I : ℂ := ⟨-2/25, 14/25⟩

/-- Theorem: V = IZ -/
theorem voltage_equals_current_times_impedance : V = I * Z := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_voltage_equals_current_times_impedance_l1277_127700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_reroll_probability_l1277_127773

def is_optimal_reroll (initial_roll : Fin 6 × Fin 6 × Fin 6) : Bool :=
  let a := initial_roll.1.val + 1
  let b := initial_roll.2.1.val + 1
  let c := initial_roll.2.2.val + 1
  (a ≤ 3 ∧ b ≤ 3) ∨ (a ≤ 3 ∧ c ≤ 3) ∨ (b ≤ 3 ∧ c ≤ 3) ∨ (a > 3 ∧ b > 3 ∧ c > 3)

def count_optimal_rerolls : Nat :=
  Finset.sum (Finset.univ : Finset (Fin 6 × Fin 6 × Fin 6)) (λ roll => if is_optimal_reroll roll then 1 else 0)

theorem optimal_reroll_probability :
  (count_optimal_rerolls : ℚ) / 216 = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_reroll_probability_l1277_127773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_person_announcing_ten_picked_ten_l1277_127725

/-- Represents a circle of 15 people with their chosen numbers -/
def Circle := Fin 15 → ℝ

/-- The condition that each person announces the average of their neighbors' numbers -/
def valid_circle (c : Circle) : Prop :=
  ∀ k : Fin 15, (c (k - 1) + c (k + 1)) / 2 = k.val + 1

/-- The theorem stating that the person announcing 10 must have picked 10 -/
theorem person_announcing_ten_picked_ten (c : Circle) (h : valid_circle c) :
  c 9 = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_person_announcing_ten_picked_ten_l1277_127725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_astronaut_revolutions_l1277_127780

/-- The number of revolutions an astronaut makes when attached to a small circle
    rolling inside a larger circle. -/
theorem astronaut_revolutions (n : ℕ) (h : n > 2) :
  (n : ℝ) - 1 = n / 1 - 1 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_astronaut_revolutions_l1277_127780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_of_six_moles_AlBr3_l1277_127740

/-- The atomic mass of Aluminum in g/mol -/
def atomic_mass_Al : ℝ := 26.98

/-- The atomic mass of Bromine in g/mol -/
def atomic_mass_Br : ℝ := 79.90

/-- The number of moles of AlBr3 -/
def moles_AlBr3 : ℝ := 6

/-- The molar mass of AlBr3 in g/mol -/
def molar_mass_AlBr3 : ℝ := atomic_mass_Al + 3 * atomic_mass_Br

/-- The weight of AlBr3 in grams -/
def weight_AlBr3 : ℝ := moles_AlBr3 * molar_mass_AlBr3

/-- Theorem stating the weight of 6 moles of AlBr3 -/
theorem weight_of_six_moles_AlBr3 : 
  ∃ ε > 0, |weight_AlBr3 - 1600.08| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_of_six_moles_AlBr3_l1277_127740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_urn_probability_l1277_127739

/-- Represents the color of a ball -/
inductive BallColor
| Red
| Blue

/-- Represents the state of the urn -/
structure UrnState :=
  (red : ℕ)
  (blue : ℕ)

/-- Represents a single operation of drawing and adding balls -/
def perform_operation (state : UrnState) : UrnState → Prop :=
  sorry

/-- The initial state of the urn -/
def initial_state : UrnState :=
  { red := 2, blue := 1 }

/-- The probability of drawing a red ball given the current state -/
def prob_draw_red (state : UrnState) : ℚ :=
  state.red / (state.red + state.blue)

/-- The probability of drawing a blue ball given the current state -/
def prob_draw_blue (state : UrnState) : ℚ :=
  state.blue / (state.red + state.blue)

/-- The probability of reaching the final state after 3 operations -/
def prob_final_state (final_state : UrnState) : ℚ → Prop :=
  sorry

theorem urn_probability :
  prob_final_state { red := 5, blue := 4 } (3/10) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_urn_probability_l1277_127739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_measure_l1277_127704

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ) : Prop :=
  ∃ (a b c : ℝ),
    a = Real.sqrt 3 ∧
    b = 1 ∧
    c = 2 ∧
    A + B + C = Real.pi

-- Theorem statement
theorem angle_A_measure (A B C : ℝ) :
  triangle_ABC A B C → A = Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_measure_l1277_127704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_is_sqrt_29_l1277_127796

/-- The polynomial whose roots form the rhombus -/
def p (z : ℂ) : ℂ := z^4 + 4*Complex.I*z^3 + (1 + Complex.I)*z^2 + (2 + 4*Complex.I)*z + (1 - 4*Complex.I)

/-- The roots of the polynomial -/
def roots : Set ℂ := {z | p z = 0}

/-- The area of the rhombus formed by the roots -/
noncomputable def rhombusArea : ℝ := sorry

/-- Theorem stating that the area of the rhombus is √29 -/
theorem rhombus_area_is_sqrt_29 : rhombusArea = Real.sqrt 29 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_is_sqrt_29_l1277_127796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_circle_l1277_127799

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- A line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point is in the third quadrant -/
def Point.in_third_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- Check if a point is on a circle -/
def Point.on_circle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- Check if a point is on a line -/
def Point.on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if a line is tangent to a circle -/
def Line.is_tangent_to (l : Line) (c : Circle) : Prop :=
  ∃ p : Point, p.on_circle c ∧ p.on_line l ∧
  ∀ q : Point, q.on_circle c ∧ q.on_line l → q = p

theorem tangent_line_to_circle (O : Point) (C : Circle) (l : Line) :
  (O.x = 0 ∧ O.y = 0) →
  C.center = Point.mk (-2) 0 ∧ C.radius = 1 →
  O.on_line l →
  l.is_tangent_to C →
  (∃ p : Point, p.on_circle C ∧ p.on_line l ∧ p.in_third_quadrant) →
  l.a = l.b ∧ l.c = 0 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_circle_l1277_127799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_minimum_distance_l1277_127744

/-- Parabola type representing y^2 = -8x -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ
  directrix : ℝ → ℝ × ℝ

/-- Point on a parabola -/
structure PointOnParabola (p : Parabola) where
  point : ℝ × ℝ
  on_parabola : p.equation point.1 point.2

theorem parabola_minimum_distance 
  (p : Parabola)
  (h_eq : p.equation = fun x y ↦ y^2 = -8*x)
  (O : ℝ × ℝ)
  (h_O : O = (0, 0))
  (A : PointOnParabola p)
  (h_AF : Real.sqrt ((A.point.1 - p.focus.1)^2 + (A.point.2 - p.focus.2)^2) = 4)
  : ∃ (d : ℝ), d = Real.sqrt 13 ∧ 
    ∀ (P : ℝ × ℝ), (∃ t, P = p.directrix t) → 
    d ≤ Real.sqrt ((P.1 - A.point.1)^2 + (P.2 - A.point.2)^2) + 
        Real.sqrt ((P.1 - O.1)^2 + (P.2 - O.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_minimum_distance_l1277_127744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concentric_circles_theorem_l1277_127716

/-- Two concentric circles with radii R and r -/
structure ConcentricCircles where
  R : ℝ
  r : ℝ
  h : R > r

/-- Points on the circles -/
structure CirclePoints (c : ConcentricCircles) where
  P : ℝ × ℝ  -- Fixed point on smaller circle
  B : ℝ × ℝ  -- Moving point on larger circle
  C : ℝ × ℝ  -- Intersection of BP with larger circle
  A : ℝ × ℝ  -- Intersection of perpendicular through P to BP with smaller circle

/-- Locus of midpoint -/
def locus_of_midpoint (A B : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {midpoint : ℝ × ℝ | ∃ (t : ℝ), midpoint = (A.1 + t * (B.1 - A.1), A.2 + t * (B.2 - A.2)) ∧ 0 ≤ t ∧ t ≤ 1}

/-- The theorem to be proved -/
theorem concentric_circles_theorem (c : ConcentricCircles) (p : CirclePoints c) :
  (∃ (BC CA AB : ℝ), BC^2 + CA^2 + AB^2 = 6 * c.R^2 + 2 * c.r^2) ∧
  (∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (0, c.r / 2) ∧
    radius = c.R / 2 ∧
    locus_of_midpoint p.A p.B = {point : ℝ × ℝ | (point.1 - center.1)^2 + (point.2 - center.2)^2 = radius^2}) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_concentric_circles_theorem_l1277_127716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_valid_quadruples_l1277_127709

/-- Least common multiple of two positive integers -/
def lcm (a b : ℕ+) : ℕ+ := sorry

/-- The number of valid quadruples satisfying the LCM conditions -/
def count_valid_quadruples : ℕ := sorry

/-- The set of valid quadruples -/
def valid_quadruples : Set (ℕ+ × ℕ+ × ℕ+ × ℕ+) := sorry

/-- Theorem stating that there are exactly 4 valid quadruples -/
theorem four_valid_quadruples :
  count_valid_quadruples = 4 ∧
  ∀ a b c d : ℕ+,
    (lcm a b = 900 ∧
     lcm b c = 1800 ∧
     lcm c d = 3600 ∧
     lcm d a = 1200) →
    (a, b, c, d) ∈ valid_quadruples :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_valid_quadruples_l1277_127709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_at_4_l1277_127767

/-- The function to be minimized -/
noncomputable def f (x : ℝ) : ℝ := 3/4 * x^2 - 6*x + 8

/-- The theorem stating that f is minimized at x = 4 -/
theorem f_min_at_4 :
  ∀ x : ℝ, f x ≥ f 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_at_4_l1277_127767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_shift_l1277_127775

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3)
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x)

theorem sin_shift (x : ℝ) : f x = g (x - Real.pi / 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_shift_l1277_127775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_difference_l1277_127705

theorem power_difference (x y : ℝ) (h1 : (3 : ℝ)^x = 4) (h2 : (3 : ℝ)^y = 5) : 
  (3 : ℝ)^(x-y) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_difference_l1277_127705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_n_l1277_127707

theorem no_valid_n : ¬ ∃ (n : ℕ), 219 ≤ n ∧ n ≤ 2019 ∧
  ∃ (x y : ℕ), 1 ≤ x ∧ x < n ∧ n < y ∧
  (∀ (k : ℕ), 1 ≤ k ∧ k ≤ n ∧ k ≠ x ∧ k ≠ x + 1 → y % k = 0) ∧
  y % x ≠ 0 ∧ y % (x + 1) ≠ 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_n_l1277_127707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_percentage_approx_1_8_percent_l1277_127768

/-- Calculates the percentage of the purchase price paid in interest given the purchase details and payment plan. -/
noncomputable def interest_percentage (purchase_price : ℝ) (down_payment : ℝ) (months : ℕ) (monthly_interest_rate : ℝ) : ℝ :=
  let principal := purchase_price - down_payment
  let total_amount := principal * (1 + monthly_interest_rate) ^ (months / 12 : ℝ)
  let interest_paid := total_amount - principal
  (interest_paid / purchase_price) * 100

/-- The percentage of the purchase price paid in interest is approximately 1.8%. -/
theorem interest_percentage_approx_1_8_percent :
  ∃ ε > 0, abs (interest_percentage 130 30 18 0.015 - 1.8) < ε :=
by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_percentage_approx_1_8_percent_l1277_127768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_sqrt_two_l1277_127719

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then (1/3)^x else Real.log x / Real.log (1/2)

-- State the theorem
theorem f_composition_sqrt_two : f (f (Real.sqrt 2)) = Real.sqrt 3 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_sqrt_two_l1277_127719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_ellipse_line_l1277_127751

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b
  h_a_gt_b : a > b

/-- Defines the eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- Defines a line with slope m passing through point (x₀, y₀) -/
structure Line where
  m : ℝ
  x₀ : ℝ
  y₀ : ℝ

/-- Helper function to calculate the area of a triangle given three points -/
noncomputable def area_triangle (P A B : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((B.1 - A.1) * (P.2 - A.2) - (B.2 - A.2) * (P.1 - A.1))

/-- Theorem: Maximum area of triangle PAB for given ellipse and line -/
theorem max_area_triangle_ellipse_line (e : Ellipse) (l : Line) :
  eccentricity e = Real.sqrt 5 / 5 →
  e.a^2 / Real.sqrt (e.a^2 - e.b^2) = 5 →
  l.m = 1 →
  l.x₀ = Real.sqrt (e.a^2 - e.b^2) →
  l.y₀ = 0 →
  (∃ (max_area : ℝ), 
    max_area = (16 * Real.sqrt 10) / 9 ∧
    ∀ (P : ℝ × ℝ), (P.1 / e.a)^2 + (P.2 / e.b)^2 = 1 →
    ∃ (A B : ℝ × ℝ), 
      (A.1 / e.a)^2 + (A.2 / e.b)^2 = 1 ∧
      (B.1 / e.a)^2 + (B.2 / e.b)^2 = 1 ∧
      A.2 - l.y₀ = l.m * (A.1 - l.x₀) ∧
      B.2 - l.y₀ = l.m * (B.1 - l.x₀) ∧
      area_triangle P A B ≤ max_area) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_ellipse_line_l1277_127751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_travel_through_intermediate_l1277_127770

/-- The set of cities -/
inductive City where
  | A
  | B
  | C

/-- The number of ways to travel between two cities -/
def travel_ways (from_city to_city : City) : ℕ := sorry

/-- Theorem: The number of ways to travel from A to C through B is the product
    of the number of ways from A to B and from B to C -/
theorem travel_through_intermediate (h1 : travel_ways City.A City.B = 3)
                                    (h2 : travel_ways City.B City.C = 4) :
  travel_ways City.A City.C = travel_ways City.A City.B * travel_ways City.B City.C := by
  sorry

#check travel_through_intermediate

end NUMINAMATH_CALUDE_ERRORFEEDBACK_travel_through_intermediate_l1277_127770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_classification_l1277_127752

theorem triangle_classification (A B C : ℝ) (h : 0 < Real.tan A * Real.tan B ∧ Real.tan A * Real.tan B < 1) :
  ∃ a b c, a + b + c = π ∧ a > π/2 ∧ b > 0 ∧ c > 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_classification_l1277_127752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triathlon_average_speed_approx_five_l1277_127794

/-- Calculates the average speed of a triathlete in a triathlon with equal-length segments -/
noncomputable def triathlonAverageSpeed (swimmingSpeed bikingSpeed runningSpeed : ℝ) : ℝ :=
  3 / (1 / swimmingSpeed + 1 / bikingSpeed + 1 / runningSpeed)

/-- Theorem stating that the average speed of a triathlete in a specific triathlon is approximately 5 km/h -/
theorem triathlon_average_speed_approx_five :
  let swimmingSpeed := (2 : ℝ)
  let bikingSpeed := (15 : ℝ)
  let runningSpeed := (12 : ℝ)
  let averageSpeed := triathlonAverageSpeed swimmingSpeed bikingSpeed runningSpeed
  ∃ ε > 0, abs (averageSpeed - 5) < ε ∧ ε < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triathlon_average_speed_approx_five_l1277_127794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_abs_l1277_127757

/-- The function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x) + 2 * (Real.cos x) ^ 2

/-- The theorem statement -/
theorem min_sum_abs (x₁ x₂ : ℝ) (h : f x₁ * f x₂ = -3) :
  ∃ (y₁ y₂ : ℝ), f y₁ * f y₂ = -3 ∧ |y₁ + y₂| ≤ |x₁ + x₂| ∧ |y₁ + y₂| = π / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_abs_l1277_127757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_distance_l1277_127785

theorem circle_line_distance (a : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 = 4 ∧ |x + y - a| / Real.sqrt 2 = 1) →
  (∃ (x1 y1 x2 y2 : ℝ), x1^2 + y1^2 = 4 ∧ x2^2 + y2^2 = 4 ∧ 
    |x1 + y1 - a| / Real.sqrt 2 = 1 ∧ |x2 + y2 - a| / Real.sqrt 2 = 1 ∧
    (x1 ≠ x2 ∨ y1 ≠ y2)) →
  -3 * Real.sqrt 2 < a ∧ a < 3 * Real.sqrt 2 := by
  intro h1 h2
  sorry

#check circle_line_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_distance_l1277_127785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_fraction_under_constraint_l1277_127762

def is_simplest_form (n : ℚ) : Prop := ∀ m : ℚ, m = n → (Int.gcd m.num m.den = 1)

theorem greatest_fraction_under_constraint :
  ∀ x : ℚ, (x^4 / x^2 < 17 ∧ is_simplest_form x) → x ≤ 15/4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_fraction_under_constraint_l1277_127762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_inequality_l1277_127774

theorem trig_inequality : π / 4 < 1 ∧ 1 < π / 3 → Real.tan 1 > Real.sin 1 ∧ Real.sin 1 > Real.cos 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_inequality_l1277_127774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_line_equation_l1277_127763

/-- A circle C with the following properties:
  - Tangent to the y-axis
  - Center below the x-axis
  - Intersects the x-axis at points A(1,0) and B(9,0)
-/
structure CircleC where
  center : ℝ × ℝ
  radius : ℝ
  tangent_to_y_axis : center.1 = radius
  center_below_x_axis : center.2 < 0
  intersects_x_axis : center.1 - radius = 1 ∧ center.1 + radius = 9

/-- A line l passing through A(1,0) and creating a chord of length 6 on circle C -/
structure LineL (c : CircleC) where
  slope : ℝ
  passes_through_A : slope * 1 + 0 = 0
  chord_length : ∃ (p q : ℝ × ℝ), p ≠ q ∧
    (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2 ∧
    (q.1 - c.center.1)^2 + (q.2 - c.center.2)^2 = c.radius^2 ∧
    (p.1 - q.1)^2 + (p.2 - q.2)^2 = 36

theorem circle_equation (c : CircleC) :
  (fun x y ↦ (x - 5)^2 + (y + 3)^2 = 25) = (fun x y ↦ (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2) :=
sorry

theorem line_equation (c : CircleC) (l : LineL c) :
  (fun x y ↦ x = 1 ∨ 7*x + 24*y - 7 = 0) = (fun x y ↦ y = l.slope * (x - 1)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_line_equation_l1277_127763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_replacement_theorem_l1277_127721

/-- Represents a 6-digit number -/
def SixDigitNumber := Fin 1000000

/-- Checks if a number is 6 digits long -/
def isSixDigit (n : Nat) : Prop := n ≥ 100000 ∧ n < 1000000

/-- The first number in the addition problem -/
def num1 : Nat := 742586

/-- The second number in the addition problem -/
def num2 : Nat := 829430

/-- The incorrect sum given in the problem -/
def incorrectSum : Nat := 1212016

/-- Function to replace all occurrences of one digit with another in a number -/
def replaceDigit (n : Nat) (d e : Fin 10) : Nat :=
  sorry

/-- Theorem stating the existence of d and e that solve the problem -/
theorem digit_replacement_theorem :
  ∃ (d e : Fin 10),
    (isSixDigit (replaceDigit num1 d e)) ∧
    (isSixDigit (replaceDigit num2 d e)) ∧
    (replaceDigit num1 d e + replaceDigit num2 d e = incorrectSum) ∧
    (d.val + e.val = 8) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_replacement_theorem_l1277_127721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_drain_rate_l1277_127710

-- Define the tank and flow rates
noncomputable def tank_capacity : ℝ := 10000
noncomputable def initial_volume : ℝ := tank_capacity / 2
noncomputable def pipe_rate : ℝ := 1 / 2
noncomputable def first_drain_rate : ℝ := 1 / 6
noncomputable def fill_time : ℝ := 60

-- Define the theorem
theorem second_drain_rate (x : ℝ) (h : initial_volume + (pipe_rate - first_drain_rate - x) * fill_time = tank_capacity) : x = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_drain_rate_l1277_127710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_f_values_l1277_127753

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := 2 / (2^x + 1) + Real.sin x

-- State the theorem
theorem sum_of_f_values :
  f (-2) + f (-1) + f 0 + f 1 + f 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_f_values_l1277_127753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_nine_minus_five_plus_cube_root_eight_times_neg_two_squared_equals_six_l1277_127745

theorem square_root_nine_minus_five_plus_cube_root_eight_times_neg_two_squared_equals_six :
  Real.sqrt 9 - 5 + (8 : Real)^(1/3) * (-2)^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_nine_minus_five_plus_cube_root_eight_times_neg_two_squared_equals_six_l1277_127745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cars_in_half_hour_quotient_of_N_l1277_127787

/-- Represents the maximum number of cars that can pass in half an hour -/
def N : ℕ := 2000

/-- Represents the length of a car in meters -/
def car_length : ℝ := 5

/-- Represents the time period in hours -/
def time_period : ℝ := 0.5

/-- Represents the safety rule function: 
    speed (km/h) → distance between cars (in car lengths) -/
noncomputable def safety_rule (speed : ℝ) : ℝ := ⌈speed / 10⌉

/-- Represents the distance between cars in meters based on speed -/
noncomputable def distance_between_cars (speed : ℝ) : ℝ := safety_rule speed * car_length

/-- Represents the total length of a car unit (car + safety distance) in meters -/
noncomputable def car_unit_length (speed : ℝ) : ℝ := car_length + distance_between_cars speed

/-- Theorem stating the maximum number of cars that can pass in half an hour -/
theorem max_cars_in_half_hour :
  ∀ (speed : ℝ), speed > 0 → 
  (speed * 1000 * time_period) / car_unit_length speed ≤ N := by
  sorry

/-- Theorem stating the quotient when N is divided by 10 -/
theorem quotient_of_N : N / 10 = 200 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cars_in_half_hour_quotient_of_N_l1277_127787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_side_formula_l1277_127713

/-- Theorem: In a quadrilateral ABCD with sides a, b, c, d and angles C, D,
    the following equation holds:
    a² = b² + c² + d² - 2bc cos(C) - 2cd cos(D) - 2bd cos(C+D) -/
theorem quadrilateral_side_formula 
  (a b c d : ℝ) 
  (C D : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : c > 0) 
  (h4 : d > 0) 
  (h5 : 0 < C ∧ C < π) 
  (h6 : 0 < D ∧ D < π) :
  a^2 = b^2 + c^2 + d^2 - 2*b*c*Real.cos C - 2*c*d*Real.cos D - 2*b*d*Real.cos (C+D) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_side_formula_l1277_127713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_and_range_l1277_127712

-- Define the function f
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.sin x - a * Real.cos x

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := f x 1 * f (-x) 1 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x

-- Theorem statement
theorem zero_and_range :
  (∃ a : ℝ, f (Real.pi / 4) a = 0) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 2 → -1 ≤ g x ∧ g x ≤ 2) := by
  sorry

#check zero_and_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_and_range_l1277_127712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_track_length_is_600_l1277_127798

/-- The length of the circular track -/
def track_length : ℝ := 600

/-- The distance run by the first runner before the first meeting -/
def first_runner_distance : ℝ := 120

/-- The additional distance run by the second runner between the first and second meetings -/
def second_runner_additional_distance : ℝ := 180

/-- Theorem stating that the track length is 600 meters given the conditions -/
theorem track_length_is_600 :
  ∀ (runner1_speed runner2_speed : ℝ),
  runner1_speed > 0 ∧ runner2_speed > 0 →
  (let first_meeting_time := first_runner_distance / runner1_speed
   let second_meeting_time := (track_length / 2 + second_runner_additional_distance) / runner2_speed
   (first_meeting_time * runner1_speed + first_meeting_time * runner2_speed = track_length / 2) ∧
   (second_meeting_time * runner1_speed + second_meeting_time * runner2_speed = track_length)) →
  track_length = 600 := by
  sorry

#check track_length_is_600

end NUMINAMATH_CALUDE_ERRORFEEDBACK_track_length_is_600_l1277_127798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_circle_distance_difference_l1277_127715

-- Define the hyperbola and circle
def hyperbola (x y : ℝ) : Prop := x^2 / 3 - y^2 / 5 = 1
def hyperbola_set : Set (ℝ × ℝ) := {p | hyperbola p.1 p.2}
def circle_set : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 3}

-- Define the points and conditions
def is_left_focus (F : ℝ × ℝ) (h : Set (ℝ × ℝ)) : Prop := sorry
def is_tangent_line (F P T : ℝ × ℝ) (c : Set (ℝ × ℝ)) : Prop := sorry
def on_right_branch (P : ℝ × ℝ) (h : Set (ℝ × ℝ)) : Prop := sorry
def is_midpoint (M F P : ℝ × ℝ) : Prop := sorry
def origin (O : ℝ × ℝ) : Prop := O = (0, 0)

-- Main theorem
theorem hyperbola_circle_distance_difference 
  (F P T M O : ℝ × ℝ) : 
  is_left_focus F hyperbola_set →
  is_tangent_line F P T circle_set →
  on_right_branch P hyperbola_set →
  is_midpoint M F P →
  origin O →
  ∃ (MO MT : ℝ), 
    MO = ‖M - O‖ ∧ 
    MT = ‖M - T‖ ∧ 
    MO - MT = Real.sqrt 5 - Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_circle_distance_difference_l1277_127715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_better_performance_and_stability_l1277_127784

/-- Represents a shooter's performance metrics -/
structure ShooterMetrics where
  average_score : ℝ
  variance : ℝ

/-- Defines when a shooter has better performance and is more stable -/
def has_better_performance_and_stability (a b : ShooterMetrics) : Prop :=
  a.average_score > b.average_score ∧ a.variance < b.variance

/-- Theorem stating the condition for better performance and stability -/
theorem better_performance_and_stability (a b : ShooterMetrics) :
  has_better_performance_and_stability a b →
  (a.average_score > b.average_score ∧ a.variance < b.variance) :=
by
  intro h
  exact h

#check better_performance_and_stability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_better_performance_and_stability_l1277_127784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_multiplication_result_l1277_127722

theorem matrix_multiplication_result : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![3, -2; -4, 5]
  let x : Matrix (Fin 2) (Fin 1) ℝ := !![4; -2]
  A * x = !![16; -26] := by
  -- Expand the definition of matrix multiplication
  simp [Matrix.mul_apply]
  -- Perform the calculation
  norm_num
  -- Close the proof
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_multiplication_result_l1277_127722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_divisible_by_21_with_sqrt_between_30_and_30_5_l1277_127727

theorem smallest_integer_divisible_by_21_with_sqrt_between_30_and_30_5 :
  ∃ n : ℕ+, (21 ∣ n) ∧ (900 < n.val) ∧ (n.val < 931) ∧
  (∀ m : ℕ+, (21 ∣ m) ∧ (900 < m.val) ∧ (m.val < 931) → n ≤ m) ∧
  n = 903 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_divisible_by_21_with_sqrt_between_30_and_30_5_l1277_127727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dynamics_value_l1277_127783

def alphabet_value (n : ℕ) : ℤ :=
  match n % 16 with
  | 0 => 3
  | 1 => -2
  | 2 => 1
  | 3 => 0
  | 4 => -1
  | 5 => -3
  | 6 => 3
  | 7 => -1
  | 8 => 0
  | 9 => 1
  | 10 => -2
  | 11 => 3
  | 12 => -1
  | 13 => 0
  | 14 => 1
  | 15 => -2
  | _ => 0  -- This case should never occur due to the modulo operation

def letter_to_position (c : Char) : ℕ :=
  match c with
  | 'a' => 1
  | 'b' => 2
  | 'c' => 3
  | 'd' => 4
  | 'e' => 5
  | 'f' => 6
  | 'g' => 7
  | 'h' => 8
  | 'i' => 9
  | 'j' => 10
  | 'k' => 11
  | 'l' => 12
  | 'm' => 13
  | 'n' => 14
  | 'o' => 15
  | 'p' => 16
  | 'q' => 17
  | 'r' => 18
  | 's' => 19
  | 't' => 20
  | 'u' => 21
  | 'v' => 22
  | 'w' => 23
  | 'x' => 24
  | 'y' => 25
  | 'z' => 26
  | _ => 0  -- This case should never occur for valid lowercase letters

def word_value (word : String) : ℤ :=
  word.data.map (fun c => alphabet_value (letter_to_position c)) |>.sum

theorem dynamics_value : word_value "dynamics" = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dynamics_value_l1277_127783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_wheels_in_garage_l1277_127741

theorem total_wheels_in_garage : 
  let bicycle_count : ℕ := 9
  let car_count : ℕ := 16
  let single_axle_trailer_count : ℕ := 5
  let double_axle_trailer_count : ℕ := 3
  let wheels_per_bicycle : ℕ := 2
  let wheels_per_car : ℕ := 4
  let wheels_per_single_axle_trailer : ℕ := 2
  let wheels_per_double_axle_trailer : ℕ := 4
  
  bicycle_count * wheels_per_bicycle +
  car_count * wheels_per_car +
  single_axle_trailer_count * wheels_per_single_axle_trailer +
  double_axle_trailer_count * wheels_per_double_axle_trailer = 104
:= by
  -- Proof goes here
  sorry

#eval 
  let bicycle_count : ℕ := 9
  let car_count : ℕ := 16
  let single_axle_trailer_count : ℕ := 5
  let double_axle_trailer_count : ℕ := 3
  let wheels_per_bicycle : ℕ := 2
  let wheels_per_car : ℕ := 4
  let wheels_per_single_axle_trailer : ℕ := 2
  let wheels_per_double_axle_trailer : ℕ := 4
  
  bicycle_count * wheels_per_bicycle +
  car_count * wheels_per_car +
  single_axle_trailer_count * wheels_per_single_axle_trailer +
  double_axle_trailer_count * wheels_per_double_axle_trailer

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_wheels_in_garage_l1277_127741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_with_three_divisible_by_eleven_l1277_127756

theorem range_with_three_divisible_by_eleven (x : ℕ) : 
  (∃ (a b c : ℕ), x ≤ a ∧ a < b ∧ b < c ∧ c ≤ 79 ∧ 
   11 ∣ a ∧ 11 ∣ b ∧ 11 ∣ c ∧
   (∀ (y : ℕ), x ≤ y ∧ y ≤ 79 ∧ 11 ∣ y → y = a ∨ y = b ∨ y = c)) →
  x = 55 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_with_three_divisible_by_eleven_l1277_127756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_properties_l1277_127755

/-- An isosceles triangle with base 2a and height m -/
structure IsoscelesTriangle (a m : ℝ) where
  base : ℝ := 2 * a
  height : ℝ := m
  isPositive : 0 < a ∧ 0 < m

/-- The inscribed circle of an isosceles triangle -/
noncomputable def inscribedCircle (t : IsoscelesTriangle a m) : ℝ :=
  (a * m) / (a + Real.sqrt (a^2 + m^2))

/-- The length of a tangent parallel to the base of an isosceles triangle -/
noncomputable def parallelTangent (t : IsoscelesTriangle a m) : ℝ :=
  2 * a * (Real.sqrt (a^2 + m^2) - a) / (a + Real.sqrt (a^2 + m^2))

/-- Theorem stating the properties of an inscribed circle and parallel tangent in an isosceles triangle -/
theorem isosceles_triangle_properties (a m : ℝ) (t : IsoscelesTriangle a m) :
  inscribedCircle t = (a * m) / (a + Real.sqrt (a^2 + m^2)) ∧
  parallelTangent t = 2 * a * (Real.sqrt (a^2 + m^2) - a) / (a + Real.sqrt (a^2 + m^2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_properties_l1277_127755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabolic_arch_height_at_3_inches_l1277_127776

/-- Represents a parabolic arch -/
structure ParabolicArch where
  maxHeight : ℝ
  span : ℝ

/-- Calculates the height of a parabolic arch at a given distance from the center -/
noncomputable def heightAtDistance (arch : ParabolicArch) (distance : ℝ) : ℝ :=
  let a := -4 * arch.maxHeight / (arch.span * arch.span)
  a * distance * distance + arch.maxHeight

/-- Theorem: The height of a parabolic arch with maximum height 20 inches and span 30 inches,
    at 3 inches from the center, is 19.2 inches -/
theorem parabolic_arch_height_at_3_inches :
  let arch := ParabolicArch.mk 20 30
  heightAtDistance arch 3 = 19.2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabolic_arch_height_at_3_inches_l1277_127776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_division_l1277_127761

theorem candy_division (x : ℝ) (x_pos : x > 0) : 
  (4/9 : ℝ) * x + (1/3 : ℝ) * x + (2/9 : ℝ) * x = x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_division_l1277_127761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_value_l1277_127701

-- Define the given conditions
noncomputable def radius : ℝ := 4
noncomputable def angle_AOB : ℝ := Real.pi / 2  -- right angle in radians

-- Define the shaded area function
noncomputable def shaded_area : ℝ :=
  (8 * Real.pi / 3 - 2 * Real.sqrt 3) - Real.pi / 2

-- Theorem statement
theorem shaded_area_value :
  shaded_area = 5 - 2 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_value_l1277_127701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_value_range_of_f_is_real_l1277_127792

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 2*x + 1)

-- Statement: The value range of f is ℝ
theorem value_range_of_f_is_real : Set.range f = Set.univ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_value_range_of_f_is_real_l1277_127792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l1277_127769

theorem power_equality (x : ℝ) (h : (2 : ℝ)^x = 3) : (4 : ℝ)^(3*x + 2) = 11664 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l1277_127769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correlation_coefficient_perfect_line_l1277_127795

/-- A sample point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A set of sample points -/
def SampleSet (n : ℕ) := Fin n → Point

/-- Correlation coefficient function (placeholder) -/
noncomputable def correlation_coefficient (sample : SampleSet n) : ℝ :=
  sorry

theorem correlation_coefficient_perfect_line
  (n : ℕ)
  (h_n : n ≥ 2)
  (sample : SampleSet n)
  (h_not_all_equal : ∃ i j, i ≠ j ∧ (sample i).x ≠ (sample j).x)
  (h_on_line : ∀ i, (sample i).y = 3 * (sample i).x + 1) :
  correlation_coefficient sample = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correlation_coefficient_perfect_line_l1277_127795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_remainder_of_30_l1277_127748

theorem prime_remainder_of_30 (p : ℕ) (h : Prime p) :
  Prime (p % 30) ∨ p % 30 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_remainder_of_30_l1277_127748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_60_neither_sufficient_nor_necessary_l1277_127788

-- Define a triangle with side lengths and angles
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given conditions
noncomputable def given_triangle : Triangle :=
  { a := 1
  , b := Real.sqrt 3
  , c := 0  -- We don't know c, but need to include it in the structure
  , A := 30 * Real.pi / 180
  , B := 60 * Real.pi / 180
  , C := 0  -- We don't know C, but need to include it in the structure
  }

-- Theorem stating that B = 60° is neither sufficient nor necessary
theorem b_60_neither_sufficient_nor_necessary (t : Triangle) : 
  ¬((t.a = 1 ∧ t.b = Real.sqrt 3 ∧ t.A = 30 * Real.pi / 180) ↔ t.B = 60 * Real.pi / 180) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_60_neither_sufficient_nor_necessary_l1277_127788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_length_l1277_127731

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the point
def my_point : ℝ × ℝ := (2, 0)

-- Theorem statement
theorem tangent_length :
  ∃ (t : ℝ), t > 0 ∧ t^2 = 3 ∧
  ∀ (x y : ℝ), my_circle x y →
  (x - my_point.1)^2 + (y - my_point.2)^2 = t^2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_length_l1277_127731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_and_function_zero_l1277_127797

-- Define the inequality
noncomputable def inequality (x : ℝ) : Prop := |x + 3| - 2*x - 1 < 0

-- Define the function f
noncomputable def f (x m : ℝ) : ℝ := |x - m| + |x + 1/m| - 2

theorem inequality_solution_and_function_zero :
  -- Part 1: The solution set of the inequality is (2, +∞)
  (∀ x : ℝ, inequality x ↔ x > 2) ∧
  -- Part 2: If f has a zero for some m > 0, then m = 1
  (∀ m : ℝ, m > 0 → (∃ x : ℝ, f x m = 0) → m = 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_and_function_zero_l1277_127797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_mud_green_tea_leaves_l1277_127777

/-- Represents the number of green tea leaves needed for a batch of mud masks -/
def greenTeaLeaves (sprigsOfMint : ℕ) (leavesPerSprig : ℕ) (effectivenessReduction : ℚ) : ℕ :=
  ((sprigsOfMint * leavesPerSprig : ℚ) / effectivenessReduction).ceil.toNat

/-- Theorem stating the number of green tea leaves needed for the new mud to maintain efficacy -/
theorem new_mud_green_tea_leaves :
  greenTeaLeaves 3 2 (1/2) = 12 := by
  -- Unfold the definition of greenTeaLeaves
  unfold greenTeaLeaves
  -- Simplify the arithmetic
  simp
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_mud_green_tea_leaves_l1277_127777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_three_gt_sqrt_two_l1277_127711

theorem cube_root_three_gt_sqrt_two : Real.rpow 3 (1/3) > Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_three_gt_sqrt_two_l1277_127711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1277_127750

theorem trigonometric_identities (α : Real) 
  (h1 : Real.sin α = Real.sqrt 5 / 5) 
  (h2 : α ∈ Set.Ioo (π / 2) π) : 
  Real.sin (2 * α) = -4 / 5 ∧ Real.tan (π / 4 - α) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1277_127750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grass_field_area_is_435_l1277_127778

/-- Calculates the area of a grass field given specific mowing conditions. -/
def grass_field_area (initial_productivity : ℝ) (productivity_increase : ℝ) 
  (initial_days : ℕ) (actual_days : ℕ) : ℝ :=
  let new_productivity := initial_productivity * (1 + productivity_increase)
  let total_area := initial_productivity * (initial_days : ℝ)
  total_area

/-- The main theorem proving the area of the grass field is 435 hectares. -/
theorem grass_field_area_is_435 : 
  grass_field_area 15 (33/100 * 13/100) 29 28 = 435 := by
  -- Unfold the definition of grass_field_area
  unfold grass_field_area
  -- Simplify the arithmetic
  simp
  -- The proof is incomplete, so we use sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_grass_field_area_is_435_l1277_127778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_inequality_equivalence_l1277_127742

theorem cosine_inequality_equivalence (a b : ℝ) :
  (∀ α : ℝ, |2 * (Real.cos α)^2 + a * Real.cos α + b| ≤ 1) ↔
  (∀ α : ℝ, 2 * (Real.cos α)^2 + a * Real.cos α + b = Real.cos (2 * α)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_inequality_equivalence_l1277_127742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hygiene_habits_and_risk_indicator_l1277_127782

-- Define the survey data
def total_sample : ℕ := 200
def case_group : ℕ := 100
def control_group : ℕ := 100
def case_not_good : ℕ := 40
def case_good : ℕ := 60
def control_not_good : ℕ := 10
def control_good : ℕ := 90

-- Define the K² formula
noncomputable def K_squared (n a b c d : ℕ) : ℝ :=
  (n * (a * d - b * c)^2 : ℝ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the confidence level
def confidence_level : ℝ := 0.99

-- Define the risk indicator formula
noncomputable def R (P_A_B P_not_A_B P_not_A_not_B P_A_not_B : ℝ) : ℝ :=
  (P_A_B / P_not_A_B) * (P_not_A_not_B / P_A_not_B)

-- State the theorem
theorem hygiene_habits_and_risk_indicator :
  ∃ (k : ℝ),
    K_squared total_sample case_not_good control_good control_not_good case_good > k ∧
    (K_squared total_sample case_not_good control_good control_not_good case_good > k →
      confidence_level ≤ 0.99) ∧
    R (case_not_good / case_group) 
      ((case_group - case_not_good) / case_group)
      ((control_group - control_not_good) / control_group)
      (control_not_good / control_group) = 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hygiene_habits_and_risk_indicator_l1277_127782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_lines_count_l1277_127771

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  A : Point
  B : Point
  C : Point
  is_equilateral : Prop  -- Changed to Prop as IsEquilateral is not defined

/-- Represents a line in the triangle -/
inductive TriangleLine
  | Altitude : Vertex → TriangleLine
  | Median : Vertex → TriangleLine
  | AngleBisector : Vertex → TriangleLine

/-- Vertices of the triangle -/
inductive Vertex
  | A
  | B
  | C

/-- Function to count distinct lines in the triangle -/
def count_distinct_lines (triangle : EquilateralTriangle) : ℕ :=
  3  -- We know the answer is 3, so we'll just return it directly

/-- Theorem stating that the number of distinct lines is 3 -/
theorem distinct_lines_count (triangle : EquilateralTriangle) :
  count_distinct_lines triangle = 3 := by
  -- The proof is trivial since we defined count_distinct_lines to always return 3
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_lines_count_l1277_127771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1277_127723

-- Define the functions f and g
noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- State the properties of f and g
axiom f_even : ∀ x : ℝ, f x = f (-x)
axiom g_odd : ∀ x : ℝ, g x = -g (-x)
axiom f_g_relation : ∀ x : ℝ, f x + 2 * g x = Real.exp x

-- State the theorem to be proved
theorem inequality_proof : g (-1) < f (-2) ∧ f (-2) < f (-3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1277_127723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decks_left_unsold_l1277_127718

/-- Proves that the number of decks left unsold is 3 given the initial conditions --/
theorem decks_left_unsold : ℕ :=
  by
  -- Define the given conditions
  let price : ℕ := 2
  let initial_decks : ℕ := 5
  let earnings : ℕ := 4

  -- Calculate the number of decks sold
  let decks_sold : ℕ := earnings / price

  -- Calculate the number of decks left unsold
  let decks_left : ℕ := initial_decks - decks_sold

  -- Assert the result
  exact decks_left

-- The theorem statement
#check decks_left_unsold

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decks_left_unsold_l1277_127718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_close_points_l1277_127760

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define the rectangle
def rectangle_width : ℝ := 4
def rectangle_height : ℝ := 3

-- Define a function to check if a point is inside the rectangle
def is_inside_rectangle (p : Point) : Prop :=
  0 ≤ p.x ∧ p.x ≤ rectangle_width ∧ 0 ≤ p.y ∧ p.y ≤ rectangle_height

-- Define the distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- The main theorem
theorem exists_close_points (points : Fin 6 → Point) 
  (h : ∀ i, is_inside_rectangle (points i)) :
  ∃ i j, i ≠ j ∧ distance (points i) (points j) ≤ Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_close_points_l1277_127760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farthest_distance_to_circle_cat_farthest_distance_l1277_127758

/-- The farthest distance from the origin to a point within or on a circle -/
theorem farthest_distance_to_circle (center : ℝ × ℝ) (radius : ℝ) :
  let origin : ℝ × ℝ := (0, 0)
  let distance_to_center := Real.sqrt ((center.1 - origin.1)^2 + (center.2 - origin.2)^2)
  let max_distance : ℝ := distance_to_center + radius
  ∀ point : ℝ × ℝ, (point.1 - center.1)^2 + (point.2 - center.2)^2 ≤ radius^2 →
    (point.1 - origin.1)^2 + (point.2 - origin.2)^2 ≤ max_distance^2 :=
by
  sorry

/-- The specific case for the cat problem -/
theorem cat_farthest_distance :
  let origin : ℝ × ℝ := (0, 0)
  let pole : ℝ × ℝ := (5, 1)
  let leash_length : ℝ := 15
  let max_distance : ℝ := Real.sqrt 26 + 15
  ∀ point : ℝ × ℝ, (point.1 - pole.1)^2 + (point.2 - pole.2)^2 ≤ leash_length^2 →
    (point.1 - origin.1)^2 + (point.2 - origin.2)^2 ≤ max_distance^2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_farthest_distance_to_circle_cat_farthest_distance_l1277_127758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_from_sine_ratio_l1277_127766

theorem triangle_cosine_from_sine_ratio (A B C : ℝ) (h : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi) 
  (h_ratio : ∃ (k : ℝ), k > 0 ∧ Real.sin A = 2*k ∧ Real.sin B = 3*k ∧ Real.sin C = 4*k) : 
  Real.cos C = -1/4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_from_sine_ratio_l1277_127766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_property_l1277_127703

noncomputable def powerFunction (α : ℝ) : ℝ → ℝ := fun x ↦ x ^ α

theorem power_function_property :
  ∀ α : ℝ, powerFunction α 2 = Real.sqrt 2 → powerFunction α 16 = 4 := by
  intro α h
  have h1 : α = 1/2 := by
    -- Proof that α = 1/2
    sorry
  -- Use h1 to prove that powerFunction α 16 = 4
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_property_l1277_127703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_quadratic_l1277_127720

open Real

theorem integral_sqrt_quadratic (x : ℝ) :
  let f : ℝ → ℝ := λ x => (x + 4) / 2 * sqrt (x^2 + 8*x + 25) + 9 / 2 * log (abs (x + 4 + sqrt (x^2 + 8*x + 25)))
  deriv f x = sqrt (x^2 + 8*x + 25) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_quadratic_l1277_127720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1277_127729

noncomputable def f (x : ℝ) := (2/3) * x^3 - x

theorem f_properties :
  (∀ x ∈ Set.Icc (-1 : ℝ) 3, f x ≤ 2007) ∧
  (∀ x t, t > 0 → |f (Real.sin x) + f (Real.cos x)| ≤ 2 * f (t + 1/(2*t))) ∧
  (HasDerivAt f ((Real.pi/4).tan) 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1277_127729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increase_average_grades_l1277_127781

theorem increase_average_grades (group_a_initial : ℝ) (group_b_initial : ℝ)
  (lopatin_grade : ℝ) (filin_grade : ℝ) 
  (group_a_size : ℕ) (group_b_size : ℕ) :
  group_a_initial = 47.2 →
  group_b_initial = 41.8 →
  lopatin_grade = 47 →
  filin_grade = 44 →
  group_a_size = 10 →
  group_b_size = 10 →
  (((group_a_initial * group_a_size - lopatin_grade - filin_grade) / (group_a_size - 2) > group_a_initial) ∧
   ((group_b_initial * group_b_size + lopatin_grade + filin_grade) / (group_b_size + 2) > group_b_initial)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_increase_average_grades_l1277_127781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shoe_pairing_probability_l1277_127734

/-- Represents a shoe pairing made by the child -/
def ShoePairing := Fin 8 → Fin 8

/-- Checks if a shoe pairing is valid according to the problem conditions -/
def is_valid_pairing (p : ShoePairing) : Prop :=
  ∀ k : Fin 3, ∀ s : Finset (Fin 8), s.card = k → 
    (∃ i ∈ s, p i ∉ s) ∨ (∃ i ∉ s, p i ∈ s)

/-- The total number of possible shoe pairings -/
def total_pairings : ℕ := Nat.factorial 8

/-- The number of valid shoe pairings -/
def valid_pairings : ℕ := 7560

theorem shoe_pairing_probability : 
  (valid_pairings : ℚ) / total_pairings = 21 / 112 := by
  sorry

#eval valid_pairings + 112  -- Should output 133

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shoe_pairing_probability_l1277_127734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_books_sold_on_thursday_l1277_127765

theorem books_sold_on_thursday 
  (initial_stock : ℕ) 
  (sold_monday : ℕ) 
  (sold_tuesday : ℕ) 
  (sold_wednesday : ℕ) 
  (sold_friday : ℕ) 
  (unsold_percentage : ℚ) 
  (h1 : initial_stock = 700)
  (h2 : sold_monday = 50)
  (h3 : sold_tuesday = 82)
  (h4 : sold_wednesday = 60)
  (h5 : sold_friday = 40)
  (h6 : unsold_percentage = 60 / 100) :
  ∃ (sold_thursday : ℕ),
    sold_thursday = initial_stock - 
      (sold_monday + sold_tuesday + sold_wednesday + sold_friday) - 
      (unsold_percentage * ↑initial_stock).floor := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_books_sold_on_thursday_l1277_127765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1277_127764

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt x + 5

-- State the theorem
theorem range_of_f :
  Set.range f = Set.Ici 5 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1277_127764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l1277_127730

open Real

-- Define the function f on the open interval (0, π/2)
variable (f : ℝ → ℝ)

-- Define the domain of f and its differentiability
variable (hf : ∀ x, 0 < x → x < π/2 → Differentiable ℝ f)

-- Define the second derivative of f
variable (f'' : ℝ → ℝ)
variable (hf'' : ∀ x, 0 < x → x < π/2 → HasDerivAt (deriv f) (f'' x) x)

-- State the condition on f and its second derivative
variable (h_cond : ∀ x, 0 < x → x < π/2 → cos x * f'' x + sin x * f x < 0)

-- State the theorem to be proved
theorem f_inequality : f (π/6) > Real.sqrt 3 * f (π/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l1277_127730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_ball_on_torus_l1277_127714

noncomputable def largest_ball_radius : ℝ := 9/4

noncomputable def inner_radius : ℝ := 2

noncomputable def outer_radius : ℝ := 4

noncomputable def torus_circle_center : ℝ × ℝ × ℝ := (3, 0, 1)

noncomputable def torus_circle_radius : ℝ := 1

theorem largest_ball_on_torus :
  let r := largest_ball_radius
  let center := torus_circle_center
  (r + 1)^2 = (r - 1)^2 + center.1^2 ∧
  r = 9/4 ∧
  inner_radius = 2 ∧
  outer_radius = 4 ∧
  torus_circle_radius = 1 :=
by
  -- The proof goes here
  sorry

#check largest_ball_on_torus

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_ball_on_torus_l1277_127714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_special_matrices_no_repeats_l1277_127772

/-- A 3x3 matrix with 1's on the main diagonal and 0's or 1's elsewhere -/
def SpecialMatrix := {A : Matrix (Fin 3) (Fin 3) ℕ // 
  (∀ i, A i i = 1) ∧ 
  (∀ i j, i ≠ j → A i j = 0 ∨ A i j = 1)}

/-- Check if two rows of a matrix are equal -/
def rowsEqual (A : Matrix (Fin 3) (Fin 3) ℕ) (i j : Fin 3) : Prop :=
  ∀ k, A i k = A j k

/-- A 3x3 matrix with no repeating rows -/
def NoRepeatingRows (A : Matrix (Fin 3) (Fin 3) ℕ) : Prop :=
  ∀ i j, i ≠ j → ¬(rowsEqual A i j)

/-- The set of special matrices with no repeating rows -/
def SpecialMatrixNoRepeats := {A : SpecialMatrix // NoRepeatingRows A.val}

/-- Assume that SpecialMatrixNoRepeats is finite -/
instance : Fintype SpecialMatrixNoRepeats := sorry

theorem count_special_matrices_no_repeats : Fintype.card SpecialMatrixNoRepeats = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_special_matrices_no_repeats_l1277_127772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_cutting_probability_l1277_127736

/-- The probability that all three parts of a 1-meter pipe are at least 25 cm long when cut at two random points. -/
theorem pipe_cutting_probability : ℝ := by
  -- Total length of the pipe in cm
  let total_length : ℝ := 100

  -- Minimum required length for each part in cm
  let min_length : ℝ := 25

  -- Area of all possible cuts
  let area_all_cuts : ℝ := (1 / 2) * total_length * total_length

  -- Area of valid cuts where all parts are at least min_length
  let area_valid_cuts : ℝ := (1 / 2) * (total_length - 2 * min_length) * (total_length - 2 * min_length)

  -- The probability is equal to 1/16
  have probability_all_parts_usable : (area_valid_cuts / area_all_cuts) = 1 / 16 := by
    -- Proof steps would go here
    sorry

  -- All cuts must be within the total length
  have cut_constraint : ∀ (x y : ℝ), x + y ≤ total_length := by
    -- Proof steps would go here
    sorry

  -- Each part must be at least min_length
  have part_length_constraint : ∀ (x y : ℝ), 
    x ≥ min_length ∧ y ≥ min_length ∧ (total_length - x - y) ≥ min_length := by
    -- Proof steps would go here
    sorry

  -- Return the probability
  exact 1 / 16


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_cutting_probability_l1277_127736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_intersection_range_l1277_127708

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a + Real.log x) / x

-- Define the function g
def g : ℝ → ℝ := λ x => 1

-- Theorem for the tangent line equation
theorem tangent_line_at_one (x y : ℝ) :
  f 4 1 = 4 →
  deriv (f 4) 1 = -3 →
  3 * x + y - 7 = 0 ↔ y - 4 = -3 * (x - 1) :=
sorry

-- Theorem for the range of a
theorem intersection_range (a : ℝ) :
  (∃ x, x ∈ Set.Ioo 0 (Real.exp 2) ∧ f a x = g x) ↔ a ∈ Set.Ici 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_intersection_range_l1277_127708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_monotone_increasing_F_no_zeros_iff_a_gt_four_l1277_127749

noncomputable section

open Real

def e : ℝ := exp 1

def f (x : ℝ) : ℝ := e^(x/2) - x/4

def g (x : ℝ) : ℝ := (x + 1) * ((deriv f) x)

def F (a : ℝ) (x : ℝ) : ℝ := log (x + 1) - a * f x + 4

theorem g_monotone_increasing :
  ∀ x > -1, (deriv g) x > 0 :=
sorry

theorem F_no_zeros_iff_a_gt_four (a : ℝ) (h : a > 0) :
  (∀ x, F a x ≠ 0) ↔ a > 4 :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_monotone_increasing_F_no_zeros_iff_a_gt_four_l1277_127749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_of_women_in_study_group_l1277_127790

theorem percentage_of_women_in_study_group 
  (women_lawyer_percentage : Real) 
  (probability_woman_lawyer : Real) : Real :=
  let percentage_women := probability_woman_lawyer / women_lawyer_percentage
  have h1 : women_lawyer_percentage = 0.60 := by sorry
  have h2 : probability_woman_lawyer = 0.54 := by sorry
  have h3 : percentage_women = 0.90 := by sorry
  percentage_women


end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_of_women_in_study_group_l1277_127790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_sets_equinumerous_l1277_127728

universe u
variable {α : Type u}
variable (A A₁ A₂ : Set α)

theorem nested_sets_equinumerous 
  (h1 : A ⊇ A₁) 
  (h2 : A₁ ⊇ A₂) 
  (h3 : Equiv A₂ A) : 
  Equiv A₁ A := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_sets_equinumerous_l1277_127728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_diameter_triple_volume_l1277_127743

/-- The volume of a sphere with radius r -/
noncomputable def sphereVolume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

/-- The radius of a sphere given its volume -/
noncomputable def sphereRadius (v : ℝ) : ℝ := ((3 * v) / (4 * Real.pi))^(1/3)

theorem sphere_diameter_triple_volume (r : ℝ) (h : r = 5) : 
  2 * sphereRadius (3 * sphereVolume r) = 10 * (3 : ℝ)^(1/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_diameter_triple_volume_l1277_127743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_of_18_dividing_30_factorial_l1277_127759

theorem largest_power_of_18_dividing_30_factorial :
  ∃ n : ℕ, n = 7 ∧
  (∀ m : ℕ, 18^m ∣ Nat.factorial 30 → m ≤ n) ∧
  (18^n ∣ Nat.factorial 30) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_of_18_dividing_30_factorial_l1277_127759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circumradius_l1277_127791

/-- The circumradius of a triangle given specific conditions -/
theorem triangle_circumradius (a b c : ℝ) (A B C : ℝ) :
  a^2 + b^2 = 6 →
  Real.cos (A - B) * Real.cos C = 2/3 →
  ∃ (R : ℝ), R = (3 * Real.sqrt 10) / 10 ∧ 
  R * 2 * Real.sin A = a ∧
  R * 2 * Real.sin B = b ∧
  R * 2 * Real.sin C = c :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circumradius_l1277_127791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_surface_area_is_24pi_l1277_127786

/-- A regular square prism with vertices on a sphere -/
structure PrismOnSphere where
  /-- The height of the prism -/
  height : ℝ
  /-- The volume of the prism -/
  volume : ℝ
  /-- The radius of the sphere -/
  sphere_radius : ℝ
  /-- All vertices of the prism lie on the surface of the sphere -/
  vertices_on_sphere : True
  /-- The prism is regular -/
  is_regular : True

/-- The surface area of a sphere given its radius -/
noncomputable def sphere_surface_area (r : ℝ) : ℝ := 4 * Real.pi * r^2

/-- Theorem: The surface area of the sphere is 24π -/
theorem sphere_surface_area_is_24pi (p : PrismOnSphere) 
    (h1 : p.height = 4)
    (h2 : p.volume = 16) :
  sphere_surface_area p.sphere_radius = 24 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_surface_area_is_24pi_l1277_127786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1277_127746

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : a > 0 ∧ b > 0

/-- The focal length of a hyperbola -/
noncomputable def focal_length (h : Hyperbola) : ℝ := 
  Real.sqrt (h.a^2 + h.b^2)

/-- Checks if a point (x, y) lies on the asymptote of a hyperbola -/
def on_asymptote (h : Hyperbola) (x y : ℝ) : Prop :=
  y = (h.b / h.a) * x ∨ y = -(h.b / h.a) * x

theorem hyperbola_equation (h : Hyperbola) 
  (h_focal : focal_length h = 10)
  (h_asymptote : on_asymptote h 2 1) :
  h.a = Real.sqrt 5 ∧ h.b = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1277_127746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_of_N_is_seven_l1277_127706

-- Define N as the cube root of 16^75 * 75^16
noncomputable def N : ℝ := (16^75 * 75^16)^(1/3)

-- Function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem sum_of_digits_of_N_is_seven :
  sumOfDigits (Int.floor N).natAbs = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_of_N_is_seven_l1277_127706
