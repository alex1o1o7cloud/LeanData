import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_seq_l1317_131768

def is_arithmetic_sequence (x a b y : ℝ) : Prop :=
  ∃ r : ℝ, a = x + r ∧ b = a + r ∧ y = b + r

def is_geometric_sequence (x c d y : ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ c = x * r ∧ d = c * r ∧ y = d * r

theorem min_value_seq (x y a b c d : ℝ) : 
  x > 0 → y > 0 → 
  is_arithmetic_sequence x a b y → 
  is_geometric_sequence x c d y → 
  (∀ a b c d, (a + b)^2 / (c * d) ≥ 4) ∧ 
  (∃ a b c d, (a + b)^2 / (c * d) = 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_seq_l1317_131768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_men_in_second_group_l1317_131737

/-- Represents the daily work done by a boy -/
def B : ℝ := sorry

/-- Represents the daily work done by a man -/
def M : ℝ := sorry

/-- The number of men in the second group -/
def x : ℕ := sorry

/-- The daily work done by a man is twice that of a boy -/
axiom man_work_twice_boy : M = 2 * B

/-- The total work done by the first group in 5 days -/
def total_work_1 : ℝ := (12 * M + 16 * B) * 5

/-- The total work done by the second group in 4 days -/
def total_work_2 : ℝ := (x * M + 24 * B) * 4

/-- Both groups complete the same total work -/
axiom equal_total_work : total_work_1 = total_work_2

theorem men_in_second_group : x = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_men_in_second_group_l1317_131737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_inequality_solution_set_l1317_131707

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- State the theorem
theorem floor_inequality_solution_set :
  {x : ℝ | (floor x)^2 - 3*(floor x) - 10 ≤ 0} = Set.Ico (-2 : ℝ) 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_inequality_solution_set_l1317_131707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_graph_l1317_131791

noncomputable section

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := Real.sin (a * x)
def g (a : ℝ) (x : ℝ) : ℝ := Real.sin (a * x + Real.pi / 3)

-- State the theorem
theorem shift_graph (a : ℝ) (h1 : a > 0) (h2 : ∀ x, f a x = f a (x + Real.pi)) :
  ∀ x, g a x = f a (x + Real.pi / 6) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_graph_l1317_131791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_functions_with_smaller_period_l1317_131711

/-- The period of a trigonometric function -/
noncomputable def TrigPeriod (f : ℝ → ℝ) : ℝ := sorry

/-- A function is periodic with period p if f(x + p) = f(x) for all x -/
def IsPeriodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem trig_functions_with_smaller_period :
  (TrigPeriod Real.sin = 2 * Real.pi) ∧
  (TrigPeriod Real.cos = 2 * Real.pi) ∧
  (TrigPeriod Real.tan < 2 * Real.pi) ∧
  (TrigPeriod Real.arctan < 2 * Real.pi) ∧
  (∀ f : ℝ → ℝ, (f = Real.sin ∨ f = Real.cos ∨ f = Real.tan ∨ f = Real.arctan) →
    IsPeriodic f (TrigPeriod f)) := by
  sorry

#check trig_functions_with_smaller_period

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_functions_with_smaller_period_l1317_131711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_inequality_g_l1317_131765

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := |x - 1| + |x + 1|
noncomputable def g (x : ℝ) : ℝ := |x + 3/2| + |x - 3/2|

-- Theorem 1: Solution set of f(x) ≤ x + 2
theorem solution_set_f (x : ℝ) : f x ≤ x + 2 ↔ 0 ≤ x ∧ x ≤ 2 := by
  sorry

-- Theorem 2: Inequality involving g(x)
theorem inequality_g (x a : ℝ) (ha : a ≠ 0) : 
  (|a + 1| - |2*a - 1|) / |a| ≤ g x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_inequality_g_l1317_131765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_interest_scaling_specific_investment_interest_l1317_131787

/-- Calculates the interest earned on an investment using simple interest formula -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem investment_interest_scaling (initial_principal : ℝ) (initial_interest : ℝ) (new_principal : ℝ) :
  initial_principal > 0 →
  initial_interest > 0 →
  new_principal > 0 →
  (let rate := initial_interest / initial_principal
   simple_interest initial_principal rate 1 = initial_interest) →
  simple_interest new_principal (initial_interest / initial_principal) 1 = (new_principal / initial_principal) * initial_interest :=
by
  sorry

/-- Specific case for the given problem -/
theorem specific_investment_interest :
  let initial_principal := 5000
  let initial_interest := 250
  let new_principal := 20000
  let rate := initial_interest / initial_principal
  (simple_interest initial_principal rate 1 = initial_interest) →
  simple_interest new_principal rate 1 = 1000 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_interest_scaling_specific_investment_interest_l1317_131787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_ellipse_hyperbola_l1317_131796

-- Define the parabola C
def parabola_C (x y : ℝ) : Prop := y^2 = 4*x

-- Define the point P on parabola C
def point_P : ℝ × ℝ := (1, 2)  -- We choose m = 2 as it satisfies the condition

-- Define the ellipse C'
def ellipse_C' (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := 3*x^2 - y^2/2 = 1

-- Define the focus of parabola C
def focus_C : ℝ × ℝ := (1, 0)

-- Theorem statement
theorem parabola_ellipse_hyperbola :
  ∀ x y : ℝ,
  (parabola_C x y ∧ ellipse_C' x y) →
  (∃ A B : ℝ × ℝ, 
    parabola_C A.1 A.2 ∧ parabola_C B.1 B.2 ∧
    ellipse_C' A.1 A.2 ∧ ellipse_C' B.1 B.2 ∧
    (∀ x y : ℝ, hyperbola x y ↔ 
      (∃ k : ℝ, y = k*x ∨ y = -k*x) ∧
      hyperbola point_P.1 point_P.2)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_ellipse_hyperbola_l1317_131796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_conical_tent_volume_l1317_131715

/-- The volume of a cone with radius r and height h is (1/3) * π * r^2 * h -/
theorem cone_volume (r h : ℝ) (hr : r > 0) (hh : h > 0) :
  (1 / 3 : ℝ) * Real.pi * r^2 * h = (1 / 3 : ℝ) * Real.pi * r^2 * h := by
  rfl

/-- The volume of a conical tent with diameter 20 feet and height 10 feet is (1000/3)π cubic feet -/
theorem conical_tent_volume :
  (1 / 3 : ℝ) * Real.pi * 10^2 * 10 = (1000 / 3 : ℝ) * Real.pi := by
  have h1 : (1 / 3 : ℝ) * Real.pi * 10^2 * 10 = (1 / 3 : ℝ) * Real.pi * 1000 := by ring
  have h2 : (1 / 3 : ℝ) * Real.pi * 1000 = (1000 / 3 : ℝ) * Real.pi := by ring
  rw [h1, h2]

#check conical_tent_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_conical_tent_volume_l1317_131715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_club_reorganization_theorem_l1317_131704

/-- Represents a club organization in a school -/
structure ClubOrganization where
  num_clubs : ℕ
  children_per_club : ℕ
  n : ℕ → ℕ

/-- Theorem stating the possibility of reorganizing clubs while preserving participation counts -/
theorem club_reorganization_theorem 
  (initial : ClubOrganization) 
  (h_initial_clubs : initial.num_clubs = 30)
  (h_initial_children : initial.children_per_club = 40) :
  ∃ (final : ClubOrganization), 
    final.num_clubs = 40 ∧ 
    final.children_per_club = 30 ∧ 
    (∀ i, i ≤ 30 → initial.n i = final.n i) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_club_reorganization_theorem_l1317_131704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_cyclic_six_digit_number_l1317_131780

def is_permutation (a b : ℕ) : Prop :=
  ∃ (f : ℕ → ℕ), Function.Bijective f ∧ (∀ d, d ∈ a.digits 10 ↔ f d ∈ b.digits 10)

def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n ≤ 999999

theorem unique_cyclic_six_digit_number :
  ∃! (x : ℕ), is_six_digit x ∧
    (∀ n ∈ ({2, 3, 4, 5, 6} : Finset ℕ), is_permutation x (n * x)) ∧
    x = 142857 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_cyclic_six_digit_number_l1317_131780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_min_max_values_m_range_l1317_131788

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x - Real.sqrt 3 * Real.cos (2 * x) + 1

-- Theorem for the smallest positive period
theorem smallest_positive_period : ∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') ∧ T = π := by
  sorry

-- Theorem for minimum and maximum values in the given interval
theorem min_max_values : 
  (∀ x ∈ Set.Icc (π/4) (π/2), f x ≥ 2) ∧ 
  (∃ x ∈ Set.Icc (π/4) (π/2), f x = 2) ∧
  (∀ x ∈ Set.Icc (π/4) (π/2), f x ≤ 3) ∧
  (∃ x ∈ Set.Icc (π/4) (π/2), f x = 3) := by
  sorry

-- Theorem for the range of m
theorem m_range : 
  (∀ m : ℝ, (∀ x ∈ Set.Icc (π/4) (π/2), (f x - m)^2 < 4) ↔ (1 < m ∧ m < 4)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_min_max_values_m_range_l1317_131788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_derived_shapes_l1317_131781

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  E : Point3D
  F : Point3D
  G : Point3D
  H : Point3D
  edgeLength : ℝ

/-- Given a cube, returns the midpoint of edge EF -/
noncomputable def midpointEF (c : Cube) : Point3D :=
  { x := (c.E.x + c.F.x) / 2
  , y := (c.E.y + c.F.y) / 2
  , z := (c.E.z + c.F.z) / 2 }

/-- Calculates the area of a triangle given three points -/
noncomputable def triangleArea (p1 p2 p3 : Point3D) : ℝ := sorry

/-- Calculates the volume of a pyramid given its base area and height -/
noncomputable def pyramidVolume (baseArea height : ℝ) : ℝ :=
  (1 / 3) * baseArea * height

/-- Main theorem about the cube and derived shapes -/
theorem cube_derived_shapes (c : Cube) (h : c.edgeLength = 6) :
  let M := midpointEF c
  let areaAMH := triangleArea c.A M c.H
  let volumeAMHE := pyramidVolume (triangleArea c.E c.H M) c.edgeLength
  let heightAMHE := volumeAMHE / ((1 / 3) * areaAMH)
  areaAMH = 9 * Real.sqrt 6 ∧
  volumeAMHE = 18 ∧
  heightAMHE = Real.sqrt 6 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_derived_shapes_l1317_131781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_length_l1317_131733

/-- The length of the major axis of an ellipse with given foci and tangent line -/
theorem ellipse_major_axis_length : ∃ (major_axis_length : ℝ), major_axis_length = 85 := by
  -- Define the foci of the ellipse
  let f1 : ℝ × ℝ := (11, 30)
  let f2 : ℝ × ℝ := (51, 65)
  
  -- Define the y-coordinate of the tangent line
  let tangent_y : ℝ := 10

  -- Define the reflection of f1 across the tangent line
  let f1_reflected : ℝ × ℝ := (f1.1, 2 * tangent_y - f1.2)

  -- Calculate the distance between f1_reflected and f2
  let major_axis_length : ℝ := Real.sqrt ((f2.1 - f1_reflected.1)^2 + (f2.2 - f1_reflected.2)^2)

  -- Assert that the major axis length is 85
  use major_axis_length
  sorry -- Skip the proof for now

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_length_l1317_131733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_when_a_is_2_range_of_a_when_f_strictly_increasing_l1317_131721

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (2*a - 1)*x - 3

-- Theorem for part (1)
theorem range_of_f_when_a_is_2 :
  ∀ x ∈ Set.Icc (-2 : ℝ) 3, 
  -21/4 ≤ f 2 x ∧ f 2 x ≤ 15 :=
sorry

-- Theorem for part (2)
theorem range_of_a_when_f_strictly_increasing :
  (∀ x ∈ Set.Icc (-1 : ℝ) 3, StrictMono (fun x => f a x)) →
  a ≥ 3/2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_when_a_is_2_range_of_a_when_f_strictly_increasing_l1317_131721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_classification_l1317_131764

variable {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α] [CompleteSpace α]

def trajectory (F₁ F₂ : α) (a : ℝ) : Set α :=
  {P : α | dist P F₁ + dist P F₂ = 2 * a}

/-- Predicate to check if a set is a line segment --/
def IsLineSegment (s : Set α) : Prop := sorry

/-- Predicate to check if a set is an ellipse --/
def IsEllipse (s : Set α) : Prop := sorry

theorem trajectory_classification (F₁ F₂ : α) (a : ℝ) (h : a ≥ 0) :
  (trajectory F₁ F₂ a = {F₁} ∧ F₁ = F₂ ↔ a = 0) ∧
  (trajectory F₁ F₂ a = ∅ ↔ 0 ≤ a ∧ a < dist F₁ F₂) ∧
  (∃ (l : Set α), IsLineSegment l ∧ trajectory F₁ F₂ a = l ↔ a = dist F₁ F₂) ∧
  (∃ (e : Set α), IsEllipse e ∧ trajectory F₁ F₂ a = e ↔ a > dist F₁ F₂) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_classification_l1317_131764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_temperature_conversion_l1317_131762

/-- Converts Fahrenheit to Celsius -/
noncomputable def fahrenheit_to_celsius (F : ℝ) : ℝ := 5/9 * (F - 32)

/-- Converts Celsius to Kelvin -/
noncomputable def celsius_to_kelvin (C : ℝ) : ℝ := C + 273.15

theorem temperature_conversion (F : ℝ) (h : F = 86) :
  let C := fahrenheit_to_celsius F
  let K := celsius_to_kelvin C
  C = 30 ∧ K = 303.15 := by
  sorry

-- Remove the #eval statements as they're causing issues with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_temperature_conversion_l1317_131762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_midpoint_with_odd_double_coords_l1317_131731

/-- A point in the xy-plane with integer coordinates -/
structure IntPoint where
  x : ℤ
  y : ℤ

/-- A point in the xy-plane with rational coordinates -/
structure RatPoint where
  x : ℚ
  y : ℚ

/-- Predicate to check if a point is between two other points -/
def between (p q r : RatPoint) : Prop :=
  ∃ t : ℚ, 0 < t ∧ t < 1 ∧
    q.x = p.x + t * (r.x - p.x) ∧
    q.y = p.y + t * (r.y - p.y)

/-- Convert IntPoint to RatPoint -/
def intToRat (p : IntPoint) : RatPoint :=
  ⟨↑p.x, ↑p.y⟩

/-- The main theorem -/
theorem exists_midpoint_with_odd_double_coords
  (points : Fin 1993 → IntPoint)
  (distinct : ∀ i j, i ≠ j → points i ≠ points j)
  (no_int_between : ∀ i : Fin 1992, ¬∃ q : IntPoint, between (intToRat (points i)) (intToRat q) (intToRat (points (i + 1))))
  (cyclic : points 0 = points 1992) :
  ∃ i : Fin 1992, ∃ q : RatPoint,
    between (intToRat (points i)) q (intToRat (points (i + 1))) ∧
    Odd (2 * ⌊q.x⌋) ∧ Odd (2 * ⌊q.y⌋) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_midpoint_with_odd_double_coords_l1317_131731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_10_10_12_l1317_131784

/-- The area of a triangle with side lengths 10, 10, and 12 is 48. -/
theorem triangle_area_10_10_12 : ∃ (A B C : ℝ × ℝ), 
  let d := (λ (p q : ℝ × ℝ) ↦ ((p.1 - q.1)^2 + (p.2 - q.2)^2).sqrt)
  d A B = 12 ∧ 
  d B C = 10 ∧ 
  d C A = 10 ∧ 
  (1/2 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))) = 48 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_10_10_12_l1317_131784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_cos_and_shifted_sin_l1317_131756

theorem intersection_of_cos_and_shifted_sin (φ : ℝ) : 
  (0 ≤ φ) → (φ < π) → 
  (Real.cos (π/3) = Real.sin (2*(π/3) + φ)) → 
  φ = π/6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_cos_and_shifted_sin_l1317_131756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_point_distance_l1317_131730

-- Define the hyperbola equation
def is_on_hyperbola (x y : ℝ) : Prop :=
  x^2 / 16 - y^2 / 36 = 1

-- Define the distance from a point to the focus
noncomputable def distance_to_focus (x y : ℝ) : ℝ :=
  ((x - 2 * Real.sqrt 13)^2 + y^2).sqrt

-- Theorem statement
theorem hyperbola_point_distance (x y : ℝ) :
  is_on_hyperbola x y → distance_to_focus x y = 9 → x^2 + y^2 = 133 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_point_distance_l1317_131730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equivalent_condition_and_m_value_l1317_131777

-- Define the propositions and conditions
def p (a : ℝ) : Prop := (3 : ℝ)^a ≤ 9
def q (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 3*(3-a)*x + 9 ≥ 0
def r (a : ℝ) : Prop := p a ∧ q a
def t (a m : ℝ) : Prop := a < m ∨ a > m + 1/2

-- Define the theorem
theorem equivalent_condition_and_m_value :
  (∃ a : ℝ, r a) ∧ 
  (∃ m : ℕ, (∀ a : ℝ, ¬(t a m) → r a) ∧ 
             ¬(∀ a : ℝ, r a → ¬(t a m)) ∧
             (∀ a : ℝ, r a → 1 ≤ a ∧ a ≤ 2) ∧
             m = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equivalent_condition_and_m_value_l1317_131777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_T_100_l1317_131797

def sequence_a : ℕ → ℚ
  | 0 => 1/2
  | n + 1 => 1 / (2 - sequence_a n)

def product_T (n : ℕ) : ℚ :=
  Finset.prod (Finset.range n) (fun i => sequence_a i)

theorem product_T_100 : product_T 100 = 1/101 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_T_100_l1317_131797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_2018_equals_3_l1317_131782

def b : ℕ → ℚ
  | 0 => 1/2  -- Add this case for 0
  | 1 => 1/2
  | n + 1 => (1 + b n) / (1 - b n)

theorem b_2018_equals_3 : b 2018 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_2018_equals_3_l1317_131782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l1317_131734

/-- The area of a trapezium with given parallel sides and height -/
noncomputable def trapeziumArea (a b h : ℝ) : ℝ := (a + b) * h / 2

/-- Theorem: The area of a trapezium with parallel sides 12 cm and 16 cm, 
    and height 14 cm, is 196 cm² -/
theorem trapezium_area_example : 
  trapeziumArea 12 16 14 = 196 := by
  -- Unfold the definition of trapeziumArea
  unfold trapeziumArea
  -- Simplify the arithmetic
  simp [add_mul, mul_div_right_comm]
  -- Evaluate the expression
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l1317_131734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_problem_l1317_131739

theorem train_length_problem (speed : ℝ) (passing_time : ℝ) :
  speed = 60 ∧ passing_time = 10 →
  (2 * speed * (passing_time / 3600)) / 2 = 1/6 := by
  intro h
  have speed_eq : speed = 60 := h.left
  have time_eq : passing_time = 10 := h.right
  
  -- Calculate relative speed
  let relative_speed := 2 * speed
  
  -- Calculate total distance
  let distance := relative_speed * (passing_time / 3600)
  
  -- Calculate length of each train
  let train_length := distance / 2
  
  -- Prove that train_length = 1/6
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_problem_l1317_131739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_half_plus_alpha_times_tan_pi_plus_alpha_l1317_131763

theorem sin_pi_half_plus_alpha_times_tan_pi_plus_alpha 
  (α : ℝ) 
  (h1 : α ∈ Set.Ioo 0 π) 
  (h2 : Real.cos α = -15/17) : 
  Real.sin (π/2 + α) * Real.tan (π + α) = 8/17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_half_plus_alpha_times_tan_pi_plus_alpha_l1317_131763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tournament_rankings_l1317_131783

/-- Represents a team in the tournament --/
inductive Team : Type
  | A | B | C | D
deriving DecidableEq, Fintype

/-- Represents a possible ranking sequence of teams --/
def RankingSequence := List Team

/-- Represents the result of a match between two teams --/
structure MatchResult where
  winner : Team
  loser : Team

/-- Represents the results of Saturday's matches --/
structure SaturdayResults where
  match1 : MatchResult
  match2 : MatchResult

/-- Represents the possible outcomes of Sunday's games, including the potential tiebreaker --/
structure SundayResults where
  firstPlace : Team
  secondPlace : Team
  thirdPlace : Team
  fourthPlace : Team
  tiebreaker : Bool

/-- The main theorem stating the number of possible ranking sequences --/
theorem tournament_rankings (saturday : SaturdayResults) : 
  ∃ (sequences : List RankingSequence), 
    (sequences.length = 32) ∧ 
    (∀ seq : RankingSequence, seq ∈ sequences → seq.length = 4) ∧
    (∀ seq : RankingSequence, seq ∈ sequences → seq.toFinset = Finset.univ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tournament_rankings_l1317_131783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_outside_AB_theorem_l1317_131750

/-- Represents a circle with a center point and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Calculates the area of a circle -/
noncomputable def circleArea (c : Circle) : ℝ := Real.pi * c.radius^2

/-- Checks if two circles are tangent to each other -/
def areTangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

/-- Represents the configuration of three circles A, B, and C -/
structure CircleConfiguration where
  A : Circle
  B : Circle
  C : Circle
  A_radius_is_1 : A.radius = 1
  B_radius_is_1 : B.radius = 1
  C_radius_is_2 : C.radius = 2
  A_B_tangent : areTangent A B
  C_A_tangent : areTangent C A
  C_A_tangent_not_midpoint : C.center ≠ (A.center.1 + B.center.1, A.center.2 + B.center.2) / 2

/-- Calculates the area inside circle C but outside circles A and B -/
noncomputable def areaOutsideAB (config : CircleConfiguration) : ℝ :=
  circleArea config.C - circleArea config.A - circleArea config.B + sorry

/-- The main theorem stating that the area can be expressed as a function of π -/
theorem area_outside_AB_theorem (config : CircleConfiguration) :
  ∃ (k : ℚ), areaOutsideAB config = k * Real.pi + 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_outside_AB_theorem_l1317_131750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_satisfying_number_l1317_131754

def has_product_triple (S : Set ℕ) : Prop :=
  ∃ a b c, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b = c

def satisfies_condition (n : ℕ) : Prop :=
  ∀ A B : Set ℕ, 
    A ∪ B = Finset.range (n + 1) ∧
    A ∩ B = ∅ →
    has_product_triple A ∨ has_product_triple B

theorem smallest_satisfying_number : 
  satisfies_condition 96 ∧ 
  ∀ m : ℕ, m < 96 → ¬(satisfies_condition m) := by
  sorry

#check smallest_satisfying_number

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_satisfying_number_l1317_131754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_x_coordinates_l1317_131716

def Y : ℝ × ℝ := (0, 0)
def Z : ℝ × ℝ := (42, 0)
def W : ℝ × ℝ := (136, 76)
def V : ℝ × ℝ := (138, 78)

def area_XYZ : ℝ := 504
def area_XWV : ℝ := 1512

def area_triangle (A B C : ℝ × ℝ) : ℝ := sorry

theorem sum_of_x_coordinates :
  ∃ (X : ℝ × ℝ) (x₁ x₂ x₃ x₄ : ℝ),
    (area_triangle Y Z X = area_XYZ) ∧
    (area_triangle W V X = area_XWV) ∧
    (x₁ + x₂ + x₃ + x₄ = 232) ∧
    (X.1 = x₁ ∨ X.1 = x₂ ∨ X.1 = x₃ ∨ X.1 = x₄) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_x_coordinates_l1317_131716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_calculation_l1317_131703

noncomputable section

open Real

theorem triangle_side_calculation (A B C : ℝ) (a b c : ℝ) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Side a is opposite to angle A, b to B, c to C
  -- a = 2
  a = 2 →
  -- cos C = -1/4
  Real.cos C = -1/4 →
  -- 3 sin A = 2 sin B
  3 * Real.sin A = 2 * Real.sin B →
  -- Law of sines holds
  Real.sin A / a = Real.sin B / b →
  -- Law of cosines holds
  c^2 = a^2 + b^2 - 2*a*b*Real.cos C →
  -- Conclusion: c = 4
  c = 4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_calculation_l1317_131703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_valid_fold_points_l1317_131769

/-- An isosceles right triangle with hypotenuse length 36 -/
structure IsoscelesRightTriangle where
  /-- The length of the hypotenuse -/
  hypotenuse : ℝ
  /-- The hypotenuse is 36 units long -/
  hypotenuse_eq : hypotenuse = 36

/-- A fold point for the triangle -/
def FoldPoint (t : IsoscelesRightTriangle) := ℝ × ℝ

/-- The set of all valid fold points for the triangle -/
def ValidFoldPoints (t : IsoscelesRightTriangle) : Set (FoldPoint t) :=
  {p : FoldPoint t | True}  -- Placeholder condition, replace with actual conditions

/-- The area of the set of valid fold points -/
noncomputable def AreaOfValidFoldPoints (t : IsoscelesRightTriangle) : ℝ :=
  -- Placeholder for area calculation
  81 * Real.pi - 162 * Real.sqrt 2

theorem area_of_valid_fold_points (t : IsoscelesRightTriangle) :
  AreaOfValidFoldPoints t = 81 * Real.pi - 162 * Real.sqrt 2 := by
  -- Proof goes here
  sorry

#eval 81 + 162 + 2  -- Should output 245

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_valid_fold_points_l1317_131769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_coefficients_l1317_131770

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_coefficients (a b c : ℝ) :
  a > 0 →
  (∀ x : ℝ, x ∈ ({1, 2, 3} : Set ℝ) → |f a b c x| = 4) →
  ((a = 8 ∧ b = -32 ∧ c = 28) ∨
   (a = 4 ∧ b = -20 ∧ c = 20) ∨
   (a = 4 ∧ b = -12 ∧ c = 4)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_coefficients_l1317_131770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_set_l1317_131778

theorem quadratic_inequality_solution_set (b c : ℝ) :
  (∀ x : ℝ, x^2 - b*x + c < 0 ↔ -3 < x ∧ x < 2) →
  b + c = -7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_set_l1317_131778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_proof_l1317_131725

theorem angle_sum_proof (α β : Real) : 
  α ∈ Set.Ioo 0 π → 
  β ∈ Set.Ioo 0 π → 
  Real.cos α = Real.sqrt 10 / 10 → 
  Real.cos β = Real.sqrt 5 / 5 → 
  α + β = 3 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_proof_l1317_131725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_undefined_values_l1317_131749

-- Define the expression as noncomputable
noncomputable def f (b : ℝ) : ℝ := (b + 5) / (b^2 - 9)

-- State the theorem
theorem undefined_values (b : ℝ) : 
  ¬ (∃ y, f b = y) ↔ b = -3 ∨ b = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_undefined_values_l1317_131749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_cost_is_30_l1317_131727

/-- Represents the coffee consumption and cost for a family --/
structure CoffeeConsumption where
  frenchRoastPerDonut : ℚ
  colombianRoastPerDonut : ℚ
  frenchRoastPotSize : ℚ
  frenchRoastPotCost : ℚ
  colombianRoastPotSize : ℚ
  colombianRoastPotCost : ℚ
  momDonuts : ℕ
  dadDonuts : ℕ
  sisterDonuts : ℕ

/-- Calculates the total cost of coffee for the family --/
def totalCoffeeCost (c : CoffeeConsumption) : ℚ :=
  let frenchRoastOz := c.frenchRoastPerDonut * c.momDonuts
  let colombianRoastOz := c.colombianRoastPerDonut * (c.dadDonuts + c.sisterDonuts)
  let frenchRoastPots := (frenchRoastOz / c.frenchRoastPotSize).ceil
  let colombianRoastPots := (colombianRoastOz / c.colombianRoastPotSize).ceil
  frenchRoastPots * c.frenchRoastPotCost + colombianRoastPots * c.colombianRoastPotCost

/-- Theorem stating that the total coffee cost for the given consumption is $30 --/
theorem coffee_cost_is_30 (c : CoffeeConsumption)
  (h1 : c.frenchRoastPerDonut = 2)
  (h2 : c.colombianRoastPerDonut = 3)
  (h3 : c.frenchRoastPotSize = 12)
  (h4 : c.frenchRoastPotCost = 3)
  (h5 : c.colombianRoastPotSize = 15)
  (h6 : c.colombianRoastPotCost = 4)
  (h7 : c.momDonuts = 8)
  (h8 : c.dadDonuts = 12)
  (h9 : c.sisterDonuts = 16) :
  totalCoffeeCost c = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_cost_is_30_l1317_131727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_triangle_area_15_20_l1317_131773

/-- A rhombus with given diagonal lengths -/
structure Rhombus where
  diagonal1 : ℝ
  diagonal2 : ℝ

/-- The area of a rhombus -/
noncomputable def rhombus_area (r : Rhombus) : ℝ := (r.diagonal1 * r.diagonal2) / 2

/-- The area of one of the two equal triangles that constitute the rhombus -/
noncomputable def triangle_area (r : Rhombus) : ℝ := rhombus_area r / 2

theorem rhombus_triangle_area_15_20 :
  let r : Rhombus := { diagonal1 := 15, diagonal2 := 20 }
  triangle_area r = 75 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_triangle_area_15_20_l1317_131773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_inequality_l1317_131759

theorem line_through_point_inequality (a b : ℝ) (θ : ℝ) 
  (h : Real.cos θ / a + Real.sin θ / b = 1) : 
  1 / a^2 + 1 / b^2 ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_inequality_l1317_131759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_symmetric_scanning_codes_l1317_131705

/-- A symmetric scanning code on a 9x9 grid -/
structure SymmetricScanningCode where
  grid : Fin 9 → Fin 9 → Bool
  at_least_one_of_each_color : (∃ i j, grid i j = true) ∧ (∃ i j, grid i j = false)
  symmetric : ∀ i j, grid i j = grid (8 - i) j
                  ∧ grid i j = grid i (8 - j)
                  ∧ grid i j = grid j i
                  ∧ grid i j = grid (8 - j) (8 - i)

/-- The number of symmetric regions in a 9x9 grid -/
def num_symmetric_regions : Nat := 13

/-- The number of valid symmetric scanning codes -/
def num_valid_codes : Nat := 2^num_symmetric_regions - 2

/-- Fintype instance for SymmetricScanningCode -/
instance : Fintype SymmetricScanningCode := sorry

/-- The theorem stating the count of symmetric scanning codes -/
theorem count_symmetric_scanning_codes :
  Fintype.card SymmetricScanningCode = num_valid_codes :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_symmetric_scanning_codes_l1317_131705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_f_implies_b_range_l1317_131740

-- Define the function f
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := -1/2 * (x - 2)^2 + b * Real.log (x + 2)

-- State the theorem
theorem decreasing_f_implies_b_range (b : ℝ) :
  (∀ x > 1, ∀ y > x, f b y < f b x) → b ≤ -3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_f_implies_b_range_l1317_131740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_M_coordinates_l1317_131728

noncomputable def cylindrical_to_cartesian (ρ θ z : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.cos θ, ρ * Real.sin θ, z)

theorem point_M_coordinates :
  let (x, y, z) := cylindrical_to_cartesian (Real.sqrt 2) ((5 * Real.pi) / 4) (Real.sqrt 2)
  x = -1 ∧ y = -1 ∧ z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_M_coordinates_l1317_131728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_restricted_odd_quadratic_restricted_odd_exponential_restricted_odd_complex_l1317_131794

-- Definition of restricted odd function
def is_restricted_odd (f : ℝ → ℝ) : Prop :=
  ∃ x, f (-x) = -f x

-- Part 1
theorem restricted_odd_quadratic :
  is_restricted_odd (λ x ↦ x^2 + 2*x - 4) := by sorry

-- Part 2
theorem restricted_odd_exponential (m : ℝ) :
  (∃ x ∈ Set.Icc (-1 : ℝ) 2, (λ x ↦ 2^x + m) (-x) = -(2^x + m)) ↔
  m ∈ Set.Icc (-17/8 : ℝ) (-1) := by sorry

-- Part 3
theorem restricted_odd_complex (m : ℝ) :
  (∃ x : ℝ, (λ x ↦ 4^x - m * 2^(x+1) + m^2 - 3) (-x) = -(4^x - m * 2^(x+1) + m^2 - 3)) ↔
  m ∈ Set.Icc (1 - Real.sqrt 3) (2 * Real.sqrt 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_restricted_odd_quadratic_restricted_odd_exponential_restricted_odd_complex_l1317_131794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_interval_is_15_minutes_l1317_131786

/-- Represents the time interval between bus departures in minutes -/
def bus_interval : ℝ → Prop := λ x => x > 0

/-- The time (in minutes) it takes for a bus to overtake the boat -/
def overtake_time : ℝ := 30

/-- The time (in minutes) it takes for a bus to approach from the opposite direction -/
def approach_time : ℝ := 10

/-- The speed of the boat relative to the bus in the same direction -/
noncomputable def relative_speed_same_direction (x : ℝ) : ℝ := (overtake_time - x) / overtake_time

/-- The speed of the boat relative to the bus in the opposite direction -/
noncomputable def relative_speed_opposite_direction (x : ℝ) : ℝ := (x - approach_time) / approach_time

/-- Theorem stating that the bus interval is 15 minutes -/
theorem bus_interval_is_15_minutes :
  ∀ x : ℝ, bus_interval x → x = 15 := by
  intro x h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_interval_is_15_minutes_l1317_131786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_euclidean_geometry_statements_l1317_131714

-- Define the basic geometric objects
variable (Point Line Plane : Type)

-- Define the geometric relations
variable (lies_on : Point → Line → Prop)
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (in_plane : Line → Plane → Prop)

-- Axioms of Euclidean geometry
axiom parallel_unique (p : Point) (l : Line) (π : Plane) :
  ¬(lies_on p l) → in_plane l π → ∃! m : Line, parallel m l ∧ lies_on p m ∧ in_plane m π

axiom perpendicular_parallel (a b c : Line) (π : Plane) :
  in_plane a π → in_plane b π → in_plane c π →
  perpendicular a b → perpendicular b c → parallel a c

-- Theorem to prove
theorem euclidean_geometry_statements :
  (∃ p l π, ¬(lies_on p l) → in_plane l π → ∃! m : Line, parallel m l ∧ lies_on p m ∧ in_plane m π) ∧
  (∃ a b c π, in_plane a π → in_plane b π → in_plane c π →
              perpendicular a b → perpendicular b c → parallel a c) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_euclidean_geometry_statements_l1317_131714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_disjoint_l1317_131771

noncomputable section

-- Define the points P and C
def P : ℝ × ℝ := (1, -5)
def C_polar : ℝ × ℝ := (4, Real.pi / 2)

-- Define the line l
def l_inclination : ℝ := Real.pi / 3
def l_parametric (t : ℝ) : ℝ × ℝ := (1 + (1/2) * t, -5 + (Real.sqrt 3 / 2) * t)

-- Define the circle C
def C_radius : ℝ := 4
def C_polar_equation (θ : ℝ) : ℝ := 8 * Real.sin θ

-- Theorem statement
theorem line_circle_disjoint : 
  ∀ t θ : ℝ, (l_parametric t).1^2 + (l_parametric t).2^2 > C_radius^2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_disjoint_l1317_131771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graduation_ceremony_chairs_l1317_131723

theorem graduation_ceremony_chairs (graduates : ℕ) (teachers : ℕ) 
  (h1 : graduates = 75)
  (h2 : teachers = 25) : ℕ := by
  let parents := graduates * 2
  let additional_family := (graduates * 3 + 9) / 10  -- Rounded up 30%
  let administrators := (teachers + 4) / 5 * 2  -- Rounded up groups of 5
  have : graduates + parents + additional_family + teachers + administrators = 283 := by
    sorry
  exact 283


end NUMINAMATH_CALUDE_ERRORFEEDBACK_graduation_ceremony_chairs_l1317_131723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plan_b_rate_is_approximately_0_08_l1317_131785

/-- Represents the charge for a phone call under Plan A -/
noncomputable def plan_a_charge (duration : ℝ) : ℝ :=
  0.60 + max 0 (duration - 5) * 0.06

/-- Represents the charge for a phone call under Plan B -/
noncomputable def plan_b_charge (rate : ℝ) (duration : ℝ) : ℝ :=
  rate * duration

/-- The duration at which both plans charge the same amount -/
def breakeven_duration : ℝ := 14.999999999999996

/-- Theorem stating that the per-minute charge for Plan B is approximately $0.08 -/
theorem plan_b_rate_is_approximately_0_08 :
  ∃ (rate : ℝ), 
    plan_a_charge breakeven_duration = plan_b_charge rate breakeven_duration ∧ 
    |rate - 0.08| < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plan_b_rate_is_approximately_0_08_l1317_131785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1317_131752

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c
  h_angle_sum : A + B + C = π

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.c + t.b * Real.cos (2 * t.A) = 2 * t.a * Real.cos t.A * Real.cos t.B)
  (h2 : t.A ≤ t.B) :
  -- Part 1
  t.A = π / 3 ∧
  -- Part 2
  (∃ (area : ℝ), 
    area = Real.sqrt 3 / 3 ∧ 
    (∀ (other_area : ℝ), 
      (∃ (AD : ℝ), AD = 1 ∧ 
        other_area = (1/2) * t.b * t.c * Real.sin t.A) → 
      area ≤ other_area)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1317_131752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_iff_a_eq_one_l1317_131798

noncomputable def left_side (x a : ℝ) : ℝ := (3 : ℝ) ^ (x^2 + 6*a*x + 9*a^2)

def right_side (x a : ℝ) : ℝ := a*x^2 + 6*a^2*x + 9*a^3 + a^2 - 4*a + 4

def equation (x a : ℝ) : Prop := left_side x a = right_side x a

theorem unique_solution_iff_a_eq_one :
  ∃! a, ∃! x, equation x a ↔ a = 1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_iff_a_eq_one_l1317_131798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1317_131772

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  A = π/4 →
  Real.sin A + Real.sin (B - C) = 2 * Real.sqrt 2 * Real.sin (2 * C) →
  1/2 * b * c * Real.sin A = 1 →
  a^2 = b^2 + c^2 - 2*b*c*(Real.cos A) →
  a = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1317_131772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_interesting_pairs_l1317_131748

/-- A cell on the grid -/
structure Cell where
  x : Fin 7
  y : Fin 7

/-- A pair of cells -/
structure CellPair where
  cell1 : Cell
  cell2 : Cell

/-- Predicate to check if two cells are neighbors -/
def isNeighbor (c1 c2 : Cell) : Prop :=
  (c1.x = c2.x ∧ c1.y.val + 1 = c2.y.val) ∨
  (c1.x = c2.x ∧ c1.y.val = c2.y.val + 1) ∨
  (c1.x.val + 1 = c2.x.val ∧ c1.y = c2.y) ∨
  (c1.x.val = c2.x.val + 1 ∧ c1.y = c2.y)

/-- Predicate to check if a pair of cells is interesting -/
def isInteresting (markedCells : Finset Cell) (pair : CellPair) : Prop :=
  isNeighbor pair.cell1 pair.cell2 ∧ (pair.cell1 ∈ markedCells ∨ pair.cell2 ∈ markedCells)

/-- Theorem: The maximum number of interesting pairs on a 7x7 grid with 14 marked cells is 55 -/
theorem max_interesting_pairs :
  ∀ (markedCells : Finset Cell),
    markedCells.card = 14 →
    (∃ (interestingPairs : Finset CellPair),
      (∀ pair ∈ interestingPairs, isInteresting markedCells pair) ∧
      interestingPairs.card ≤ 55) ∧
    (∃ (optimalMarkedCells : Finset Cell) (optimalInterestingPairs : Finset CellPair),
      optimalMarkedCells.card = 14 ∧
      (∀ pair ∈ optimalInterestingPairs, isInteresting optimalMarkedCells pair) ∧
      optimalInterestingPairs.card = 55) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_interesting_pairs_l1317_131748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xyz_sum_product_bounds_l1317_131793

theorem xyz_sum_product_bounds (x y z : ℝ) (h : 3 * (x + y + z) = x^2 + y^2 + z^2) :
  ∃ (N n : ℝ), (∀ t : ℝ, n ≤ x*y + x*z + y*z ∧ x*y + x*z + y*z ≤ N) ∧ N + 15 * n = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xyz_sum_product_bounds_l1317_131793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_jump_theorem_l1317_131795

/-- Represents the position of a frog on a line -/
structure FrogPosition where
  pos : ℝ

/-- Represents a frog jump -/
structure FrogJump where
  length : ℝ

/-- Represents the state of three frogs on a line -/
structure FrogState where
  frog1 : FrogPosition
  frog2 : FrogPosition
  frog3 : FrogPosition

/-- Checks if a frog jump is valid (lands at the midpoint between the other two frogs) -/
def isValidJump (state : FrogState) (jump : FrogJump) (jumpingFrog : Nat) : Prop :=
  match jumpingFrog with
  | 1 => jump.length = (state.frog2.pos + state.frog3.pos) / 2 - state.frog1.pos
  | 2 => jump.length = (state.frog1.pos + state.frog3.pos) / 2 - state.frog2.pos
  | 3 => jump.length = (state.frog1.pos + state.frog2.pos) / 2 - state.frog3.pos
  | _ => False

/-- The main theorem about the frog jumps -/
theorem frog_jump_theorem (initialState : FrogState) 
  (jump1 jump2 jump3 : FrogJump) (order : Fin 3 → Nat) :
  (∃ i j : Fin 3, i ≠ j ∧ 
    (jump1.length = 60 ∧ jump2.length = 60) ∨ 
    (jump1.length = 60 ∧ jump3.length = 60) ∨ 
    (jump2.length = 60 ∧ jump3.length = 60)) →
  (isValidJump initialState jump1 (order 0) ∧
   isValidJump 
     ⟨⟨initialState.frog1.pos + (if order 0 = 1 then jump1.length else 0)⟩,
      ⟨initialState.frog2.pos + (if order 0 = 2 then jump1.length else 0)⟩,
      ⟨initialState.frog3.pos + (if order 0 = 3 then jump1.length else 0)⟩⟩ 
     jump2 (order 1) ∧
   isValidJump 
     ⟨⟨initialState.frog1.pos + (if order 0 = 1 then jump1.length else 0) + 
                                 (if order 1 = 1 then jump2.length else 0)⟩,
      ⟨initialState.frog2.pos + (if order 0 = 2 then jump1.length else 0) + 
                                 (if order 1 = 2 then jump2.length else 0)⟩,
      ⟨initialState.frog3.pos + (if order 0 = 3 then jump1.length else 0) + 
                                 (if order 1 = 3 then jump2.length else 0)⟩⟩ 
     jump3 (order 2)) →
  ((jump1.length = 30 ∨ jump1.length = 120) ∨
   (jump2.length = 30 ∨ jump2.length = 120) ∨
   (jump3.length = 30 ∨ jump3.length = 120)) ∧
  (max (abs (initialState.frog1.pos - initialState.frog2.pos))
       (max (abs (initialState.frog2.pos - initialState.frog3.pos))
            (abs (initialState.frog3.pos - initialState.frog1.pos))) = 100 ∨
   max (abs (initialState.frog1.pos - initialState.frog2.pos))
       (max (abs (initialState.frog2.pos - initialState.frog3.pos))
            (abs (initialState.frog3.pos - initialState.frog1.pos))) = 160) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_jump_theorem_l1317_131795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_perpendicular_lines_l1317_131779

/-- The intersection point of a line and its perpendicular line passing through a given point. -/
theorem intersection_point_perpendicular_lines (m : ℚ) (b : ℚ) (x₀ y₀ : ℚ) :
  let line1 := λ x : ℚ => m * x + b
  let perpendicular_slope := -1 / m
  let line2 := λ x : ℚ => perpendicular_slope * (x - x₀) + y₀
  let x_intersect := (y₀ - b + m * x₀) / (m - perpendicular_slope)
  let y_intersect := line1 x_intersect
  m = -3 ∧ b = 4 ∧ x₀ = 3 ∧ y₀ = 2 →
  (x_intersect, y_intersect) = (9/10, 13/10) := by
  sorry

#check intersection_point_perpendicular_lines

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_perpendicular_lines_l1317_131779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_and_minimum_m_l1317_131761

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + Real.log (a * x + 1) - (3 * a / 2) * x + 1

theorem monotonicity_and_minimum_m :
  (∀ x > 0, HasDerivAt (f 0) x (1 / x)) ∧
  (∀ a > 0, ∀ x ∈ Set.Ioo (0 : ℝ) (1 / a), HasDerivAt (f a) x ((1 / x) + (a / (a * x + 1)) - (3 * a / 2))) ∧
  (∀ a > 0, ∀ x > 1 / a, HasDerivAt (f a) x ((1 / x) + (a / (a * x + 1)) - (3 * a / 2))) ∧
  (∀ a < 0, ∀ x ∈ Set.Ioo (0 : ℝ) (-2 / (3 * a)), HasDerivAt (f a) x ((1 / x) + (a / (a * x + 1)) - (3 * a / 2))) ∧
  (∀ a < 0, ∀ x ∈ Set.Ioo (-2 / (3 * a)) (-1 / a), HasDerivAt (f a) x ((1 / x) + (a / (a * x + 1)) - (3 * a / 2))) ∧
  (∀ x > 0, x * Real.exp (x - 1/2) + Real.log (2/3) ≥ f (2/3) x) ∧
  (∃ x > 0, x * Real.exp (x - 1/2) + Real.log (2/3) = f (2/3) x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_and_minimum_m_l1317_131761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_like_terms_power_l1317_131722

theorem like_terms_power (x y : ℝ) (m n : ℕ) :
  (∀ x y, 4 * x^4 * y^(m-2) = -x^(n+1) * y^2) →
  (n - m : ℤ)^2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_like_terms_power_l1317_131722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_middle_term_l1317_131709

def is_arithmetic_sequence (seq : List ℤ) : Prop :=
  seq.length = 3 ∧ 
  ∃ d : ℤ, seq.get? 1 = some ((seq.get? 0).getD 0 + d) ∧ 
           seq.get? 2 = some ((seq.get? 1).getD 0 + d)

theorem arithmetic_sequence_middle_term :
  ∀ y : ℤ, 
    (is_arithmetic_sequence [3^3, y, 3^5]) → 
    y = 135 :=
by
  intro y h
  sorry

#eval 3^3
#eval 3^5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_middle_term_l1317_131709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_ratio_l1317_131701

-- Define the triangle ABC
variable (A B C : EuclideanSpace ℝ (Fin 2))

-- Define points D and E on AB and AC respectively
variable (D : EuclideanSpace ℝ (Fin 2))
variable (E : EuclideanSpace ℝ (Fin 2))

-- Define the angle bisector AT and its intersection F with DE
variable (T : EuclideanSpace ℝ (Fin 2))
variable (F : EuclideanSpace ℝ (Fin 2))

-- Define the conditions
variable (h1 : D = (3/4) • A + (1/4) • B)
variable (h2 : E = (2/3) • A + (1/3) • C)
variable (h3 : T = (3/5) • B + (2/5) • C)
variable (h4 : F = (5/18) • T + (13/18) • A)
variable (h5 : F = (12/18) • D + (6/18) • E)

-- State the theorem
theorem angle_bisector_ratio :
  ‖F - A‖ / ‖T - A‖ = 5 / 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_ratio_l1317_131701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mn_value_l1317_131758

theorem mn_value (m : ℤ) (n : ℕ) (h : |m + n| + (m + 2)^2 = 0) : m^n = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mn_value_l1317_131758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_four_pairs_l1317_131724

def g (n : ℕ) : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => p.1^2 + 2*p.2^2 = n) (Finset.product (Finset.range (n+1)) (Finset.range (n+1)))).card

theorem smallest_n_with_four_pairs :
  ∀ n : ℕ, n < 20 → g n ≠ 4 ∧ g 20 = 4 :=
by
  sorry

#eval g 20

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_four_pairs_l1317_131724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_is_93_l1317_131738

def my_sequence (n : ℕ) : ℕ :=
  match n with
  | 0 => 3  -- We use 0-based indexing here
  | n + 1 => my_sequence n + 2 * (n + 1)

theorem tenth_term_is_93 : my_sequence 9 = 93 := by
  rw [my_sequence]
  simp
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_is_93_l1317_131738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_s_point_implies_a_value_l1317_131741

open Real

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 1
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := log (a * x)

-- Define the derivatives of f and g
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 2 * a * x
noncomputable def g_deriv (a : ℝ) (x : ℝ) : ℝ := 1 / x

-- Theorem statement
theorem s_point_implies_a_value (a : ℝ) (h_a : a > 0) :
  (∃ x₀ : ℝ, x₀ > 0 ∧ f a x₀ = g a x₀ ∧ f_deriv a x₀ = g_deriv a x₀) →
  a = 2 / exp 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_s_point_implies_a_value_l1317_131741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_of_sum_diff_primes_l1317_131766

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem max_product_of_sum_diff_primes (m n : ℤ) :
  (Int.natAbs (m + n)) < 100 →
  (Int.natAbs (m - n)) < 100 →
  is_prime (Int.natAbs (m + n)) →
  is_prime (Int.natAbs (m - n)) →
  m * n ≤ 2350 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_of_sum_diff_primes_l1317_131766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_increasing_and_bounded_l1317_131792

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 1
  | (n + 1) => (1 / 2) * sequence_a n * (4 - sequence_a n)

theorem sequence_a_increasing_and_bounded :
  ∀ n : ℕ, 0 < sequence_a n ∧ sequence_a n < sequence_a (n + 1) ∧ sequence_a (n + 1) < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_increasing_and_bounded_l1317_131792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_distance_l1317_131790

/-- Represents a hyperbola -/
structure Hyperbola where
  center : ℝ × ℝ
  a : ℝ
  b : ℝ

/-- Check if a point lies on the hyperbola -/
def Hyperbola.contains (h : Hyperbola) (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  let (cx, cy) := h.center
  (x - cx)^2 / h.a^2 - (y - cy)^2 / h.b^2 = 1

/-- The distance between the foci of a hyperbola -/
noncomputable def Hyperbola.focalDistance (h : Hyperbola) : ℝ :=
  2 * Real.sqrt (h.a^2 + h.b^2)

/-- Theorem about the focal distance of a specific hyperbola -/
theorem hyperbola_focal_distance :
  ∃ (h : Hyperbola),
    (∀ x y, y = 2*x + 3 ∨ y = -2*x + 7 → 
      (x - h.center.1)^2 / h.a^2 - (y - h.center.2)^2 / h.b^2 = 0) ∧
    h.contains (4, 5) ∧
    h.focalDistance = 3 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_distance_l1317_131790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1317_131760

theorem problem_solution : 
  ∃ (x y : ℝ), 
    (Real.sqrt 48 / Real.sqrt 3 + Real.sqrt (1/2) * Real.sqrt 12 - Real.sqrt 24 = 4 - Real.sqrt 6) ∧
    (3*x - 2*y = 3 ∧ x + 4*y = 1) ∧
    (x = 1 ∧ y = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1317_131760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_g_piecewise_l1317_131713

-- Define the piecewise function g(x)
noncomputable def g (x : ℝ) : ℝ :=
  if x < -1 then x^2 - 4
  else if x < 2 then -2*x - 1
  else x - 2

-- State the theorem about |g(x)|
theorem abs_g_piecewise (x : ℝ) :
  abs (g x) = 
    if x < -1 then x^2 - 4
    else if x < 2 then 2*x + 1
    else x - 2 :=
by sorry

-- Additional lemmas to show the behavior in specific intervals
lemma g_on_first_interval (x : ℝ) (h : -4 ≤ x ∧ x < -1) :
  g x = x^2 - 4 :=
by sorry

lemma g_on_second_interval (x : ℝ) (h : -1 ≤ x ∧ x < 2) :
  g x = -2*x - 1 :=
by sorry

lemma g_on_third_interval (x : ℝ) (h : 2 ≤ x ∧ x ≤ 4) :
  g x = x - 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_g_piecewise_l1317_131713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pythagoras_city_schools_l1317_131710

/-- Represents a student in the math contest --/
structure Student where
  score : ℕ
  rank : ℕ

/-- Represents a school in the city of Pythagoras --/
structure School where
  team : Fin 3 → Student

theorem pythagoras_city_schools 
  (students : List Student)
  (schools : List School)
  (david ellen frank : Student)
  (h1 : ∀ s1 s2 : Student, s1 ∈ students → s2 ∈ students → s1 ≠ s2 → s1.score ≠ s2.score)
  (h2 : david ∈ students ∧ ellen ∈ students ∧ frank ∈ students)
  (h3 : ∃ school ∈ schools, ∃ i j k : Fin 3, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
        school.team i = david ∧ school.team j = ellen ∧ school.team k = frank)
  (h4 : david.rank = (students.length + 1) / 2)
  (h5 : ∀ s ∈ students, s.rank ≤ david.rank → s.score ≤ david.score)
  (h6 : ellen.rank = 29)
  (h7 : frank.rank = 50)
  (h8 : ∀ school ∈ schools, ∀ i : Fin 3, (school.team i).score ≤ david.score)
  : schools.length = 19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pythagoras_city_schools_l1317_131710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_bottle_capacity_l1317_131753

/-- The volume of the first cup in milliliters -/
def cup1_volume : ℚ := 250

/-- The number of times the first cup is poured -/
def cup1_pours : ℕ := 20

/-- The volume of the second cup in milliliters -/
def cup2_volume : ℚ := 600

/-- The number of times the second cup is poured -/
def cup2_pours : ℕ := 13

/-- Conversion factor from milliliters to liters -/
def ml_to_l : ℚ := 1000

/-- The total volume of the water bottle in liters -/
noncomputable def bottle_volume : ℚ := 
  ((cup1_volume * cup1_pours + cup2_volume * cup2_pours) / ml_to_l)

theorem water_bottle_capacity : bottle_volume = 12.8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_bottle_capacity_l1317_131753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interphase_processes_l1317_131799

/-- Represents the phases of mitosis in higher plant cells -/
inductive MitosisPhase
  | Interphase
  | DivisionPhase

/-- Represents cellular processes during mitosis -/
inductive CellularProcess
  | PrepareActiveSubstances
  | ReplicateDNA
  | SynthesizeProteins

/-- Defines the characteristics of the interphase in mitosis -/
def interphaseCharacteristics : List CellularProcess :=
  [CellularProcess.PrepareActiveSubstances, CellularProcess.ReplicateDNA, CellularProcess.SynthesizeProteins]

/-- Represents the occurrence of a process in a phase -/
def occurs_in (process : CellularProcess) (phase : MitosisPhase) : Prop := sorry

/-- Theorem stating that the interphase in mitosis involves specific cellular processes -/
theorem interphase_processes (phase : MitosisPhase) :
  phase = MitosisPhase.Interphase →
  ∀ process, process ∈ interphaseCharacteristics →
  occurs_in process phase :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interphase_processes_l1317_131799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_B_to_center_squared_l1317_131776

-- Define the circle and points
def Circle : Type := ℝ × ℝ → Prop
def center : ℝ × ℝ := (0, 0)
noncomputable def radius : ℝ := Real.sqrt 72

variable (A B C : ℝ × ℝ)

-- Define the circle equation
def on_circle (p : ℝ × ℝ) : Prop :=
  (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2

-- Define the conditions
axiom A_on_circle : on_circle A
axiom C_on_circle : on_circle C
axiom AB_length : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 8^2
axiom BC_length : (C.1 - B.1)^2 + (C.2 - B.2)^2 = 3^2
axiom ABC_right_angle : (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) = 0

-- Theorem to prove
theorem distance_B_to_center_squared :
  (B.1 - center.1)^2 + (B.2 - center.2)^2 = 61 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_B_to_center_squared_l1317_131776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_four_points_with_odd_distances_l1317_131702

-- Define a point in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a function to calculate the distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Define a predicate to check if a number is an odd integer
def isOddInteger (n : ℝ) : Prop :=
  ∃ k : ℤ, n = 2 * k + 1

-- Theorem statement
theorem no_four_points_with_odd_distances :
  ¬ ∃ (A B C D : Point),
    (isOddInteger (distance A B)) ∧
    (isOddInteger (distance A C)) ∧
    (isOddInteger (distance A D)) ∧
    (isOddInteger (distance B C)) ∧
    (isOddInteger (distance B D)) ∧
    (isOddInteger (distance C D)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_four_points_with_odd_distances_l1317_131702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1317_131744

def M : Set ℕ := {1, 2, 3, 4, 5}

def N : Set ℕ := {x : ℕ | 2 ≤ x ∧ x ≤ 4}

theorem intersection_M_N : M ∩ N = {3, 4} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1317_131744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_relationship_l1317_131751

-- Define the function
noncomputable def f (x : ℝ) : ℝ := -1/2 * x

-- Define the theorem
theorem point_relationship (x₁ y₁ x₂ y₂ : ℝ) 
  (h1 : f x₁ = y₁) 
  (h2 : f x₂ = y₂) 
  (h3 : y₁ < y₂) : 
  x₁ > x₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_relationship_l1317_131751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1317_131755

/-- Given an ellipse with the equation x²/a² + y²/b² = 1 where a > b > 0,
    vertices B₁(0,-b), B₂(0,b), A(a,0), focus F(c,0), and B₁F ⊥ AB₂,
    prove that the eccentricity of the ellipse is (√5 - 1)/2 -/
theorem ellipse_eccentricity (a b c : ℝ) (h1 : a > b) (h2 : b > 0) :
  let e := Real.sqrt ((a^2 - b^2) / a^2)
  (∀ x y, x^2/a^2 + y^2/b^2 = 1) →
  ((0 : ℝ) - b)^2 + c^2 = (a - 0)^2 + b^2 →
  e = (Real.sqrt 5 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1317_131755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_depth_notation_l1317_131726

/-- Represents the altitude in meters, where positive values are above sea level
    and negative values are below sea level. -/
def Altitude : Type := ℤ

/-- Converts a depth below sea level to its corresponding altitude -/
def depthToAltitude (depth : ℕ) : Altitude := -(depth : ℤ)

/-- Converts a height above sea level to its corresponding altitude -/
def heightToAltitude (height : ℕ) : Altitude := (height : ℤ)

theorem depth_notation (depth : ℕ) (height : ℕ) :
  heightToAltitude height = (height : ℤ) →
  depthToAltitude depth = -(depth : ℤ) :=
by
  intro h
  rfl

#check depth_notation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_depth_notation_l1317_131726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1317_131732

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC exists
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) ∧ (A + B + C = π) →
  -- Sides a, b, c are opposite to angles A, B, C respectively
  (a / Real.sin A = b / Real.sin B) ∧ (b / Real.sin B = c / Real.sin C) →
  -- Vector (2a, 1) is parallel to vector (2b-c, cos C)
  2 * a * Real.cos C = 1 * (2 * b - c) →
  -- A = π/3
  A = π / 3 →
  -- Conclusions
  (Real.sin A = Real.sqrt 3 / 2) ∧
  (∀ x, (-2 * Real.cos (2 * C)) / (1 + Real.tan C) + 1 = x → -1 < x ∧ x ≤ Real.sqrt 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1317_131732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_solution_tan_sec_equation_l1317_131719

open Real

theorem smallest_positive_solution_tan_sec_equation :
  ∃ (x : ℝ), x > 0 ∧ 
  (∀ (y : ℝ), y > 0 → tan (4*y) + tan (5*y) = 1 / cos (5*y) → x ≤ y) ∧
  tan (4*x) + tan (5*x) = 1 / cos (5*x) ∧
  x = π / 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_solution_tan_sec_equation_l1317_131719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_sufficient_not_necessary_l1317_131767

-- Define the exponential function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (3 * a - 1) ^ x

-- Define what it means for f to be decreasing
def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

-- Define propositions p and q
def p (a : ℝ) : Prop := is_decreasing (f a)
def q (a : ℝ) : Prop := a > 1/3

-- Theorem stating that p is sufficient but not necessary for q
theorem p_sufficient_not_necessary :
  (∀ a, p a → q a) ∧ ¬(∀ a, q a → p a) := by
  sorry

-- Additional lemmas to support the main theorem
lemma p_implies_q : ∀ a, p a → q a := by
  sorry

lemma q_not_implies_p : ¬(∀ a, q a → p a) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_sufficient_not_necessary_l1317_131767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coprime_and_multiple_existence_l1317_131775

theorem coprime_and_multiple_existence (n : ℕ) (S : Finset ℕ) : 
  S.card = n + 1 → (∀ x, x ∈ S → x ≤ 2*n) →
  ∃ x y z, x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ (Nat.gcd x y = 1) ∧ (z % x = 0 ∨ z % y = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coprime_and_multiple_existence_l1317_131775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_difference_l1317_131743

theorem min_abs_difference (x y : ℕ) (h : x * y - 5 * x + 6 * y = 119) :
  ∃ (a b : ℕ), a * b - 5 * a + 6 * b = 119 ∧
  ∀ (c d : ℕ), c * d - 5 * c + 6 * d = 119 →
  |Int.ofNat a - Int.ofNat b| ≤ |Int.ofNat c - Int.ofNat d| ∧
  |Int.ofNat a - Int.ofNat b| = 77 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_difference_l1317_131743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_volume_correct_l1317_131735

/-- The volume of the intersecting region of two identical cones -/
noncomputable def intersectionVolume (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 1 then
    (Real.pi / 12) * x * (3 - (9/2) * x + (7/4) * x^2)
  else if 1 < x ∧ x ≤ 2 then
    (Real.pi / 6) * (1 - (3/2) * x + (3/4) * x^2 - (1/8) * x^3)
  else
    0

theorem intersection_volume_correct (x : ℝ) (h : 0 ≤ x ∧ x ≤ 2) :
  intersectionVolume x = 
    if 0 ≤ x ∧ x ≤ 1 then
      (Real.pi / 12) * x * (3 - (9/2) * x + (7/4) * x^2)
    else
      (Real.pi / 6) * (1 - (3/2) * x + (3/4) * x^2 - (1/8) * x^3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_volume_correct_l1317_131735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_is_twenty_l1317_131718

/-- An isosceles triangle with specific side lengths -/
structure IsoscelesTriangle where
  -- Side lengths
  ab : ℝ
  bc : ℝ
  ac : ℝ
  -- Isosceles property
  ac_eq_ab : ac = ab
  -- Angle equality
  angle_acb_eq_angle_abc : True  -- We can't directly represent angles in this simple structure

/-- The perimeter of the isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ :=
  t.ab + t.bc + t.ac

/-- Theorem: The perimeter of the specific isosceles triangle is 20 -/
theorem perimeter_is_twenty (t : IsoscelesTriangle) 
  (h1 : t.ab = 6) 
  (h2 : t.bc = 8) : perimeter t = 20 := by
  sorry

#check perimeter_is_twenty

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_is_twenty_l1317_131718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odot_calculation_l1317_131712

-- Define the odot operation as noncomputable
noncomputable def odot (a b : ℝ) : ℝ := a^3 / b^2

-- Theorem statement
theorem odot_calculation :
  (odot (odot 2 4) 6) - (odot 2 (odot 4 6)) = -81/32 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odot_calculation_l1317_131712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_and_equation_solution_range_l1317_131706

theorem exponential_inequality_and_equation_solution_range :
  (∀ x : ℝ, x > 0 → Real.exp x > 1/2 * x^2 + x + 1) ∧
  (∀ a : ℝ, (∃ x : ℝ, 0 < x ∧ x < Real.pi ∧ (Real.exp x - 1) / x = a * Real.sin x + 1) ↔ a > 1/2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_and_equation_solution_range_l1317_131706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_bounds_l1317_131742

/-- The area of the triangle cut out from the largest circle inscribed in an isosceles triangle -/
noncomputable def S (α : Real) : Real :=
  let β := α / 2
  if β ≤ Real.pi/4 then
    (2 * Real.sin β * (1 - Real.sin β)) ^ 2
  else
    (Real.cos β / (1 + Real.sin β)) ^ 2

/-- The maximum and minimum values of S(α) for 60° ≤ α ≤ 120° -/
theorem S_bounds :
  (∀ α, Real.pi/3 ≤ α ∧ α ≤ 2*Real.pi/3 → S α ≤ 1/4) ∧
  (∀ α, Real.pi/3 ≤ α ∧ α ≤ 2*Real.pi/3 → S α ≥ 7 - 4*Real.sqrt 3) ∧
  (∃ α, Real.pi/3 ≤ α ∧ α ≤ 2*Real.pi/3 ∧ S α = 1/4) ∧
  (∃ α, Real.pi/3 ≤ α ∧ α ≤ 2*Real.pi/3 ∧ S α = 7 - 4*Real.sqrt 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_bounds_l1317_131742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_time_math_science_history_l1317_131747

/-- Represents the total hours Ruth spends in school per day -/
def school_hours_per_day : ℝ := 8

/-- Represents the number of school days per week -/
def school_days_per_week : ℕ := 5

/-- Represents the percentage of time spent in math class on MWF -/
def math_percent_mwf : ℝ := 0.25

/-- Represents the percentage of time spent in science class on MWF -/
def science_percent_mwf : ℝ := 0.15

/-- Represents the percentage of time spent in math class on TT -/
def math_percent_tt : ℝ := 0.20

/-- Represents the percentage of time spent in science class on TT -/
def science_percent_tt : ℝ := 0.35

/-- Represents the percentage of time spent in history class on TT -/
def history_percent_tt : ℝ := 0.15

/-- Represents the number of MWF days per week -/
def mwf_days : ℕ := 3

/-- Represents the number of TT days per week -/
def tt_days : ℕ := 2

theorem total_time_math_science_history : 
  let total_hours := 
    (math_percent_mwf * school_hours_per_day * mwf_days + 
     math_percent_tt * school_hours_per_day * tt_days) +
    (science_percent_mwf * school_hours_per_day * mwf_days + 
     science_percent_tt * school_hours_per_day * tt_days) +
    (history_percent_tt * school_hours_per_day * tt_days)
  total_hours = 20.8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_time_math_science_history_l1317_131747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_implies_k_range_l1317_131720

/-- A function that represents (2x + k) / (x - 2) -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (2 * x + k) / (x - 2)

/-- The theorem stating that if f is monotonically increasing on (3, +∞),
    then k is in the interval (-∞, -4) -/
theorem f_increasing_implies_k_range (k : ℝ) :
  (∀ x y : ℝ, 3 < x ∧ x < y → f k x < f k y) →
  k < -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_implies_k_range_l1317_131720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AD_is_30_l1317_131708

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

theorem distance_AD_is_30 (A B C D : Point)
  (h1 : B.y = A.y) -- B is due east of A
  (h2 : C.x = B.x) -- C is due north of B
  (h3 : distance A C = 15 * Real.sqrt 2) -- Distance AC = 15√2
  (h4 : Real.arctan ((C.y - A.y) / (C.x - A.x)) = π / 4) -- ∠BAC = 45°
  (h5 : distance C D = 30) -- D is 30 meters from C
  (h6 : Real.arctan ((D.y - C.y) / (C.x - D.x)) = π / 4) -- D is northwest of C at 45°
  : distance A D = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AD_is_30_l1317_131708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_constant_condition_general_term_correct_l1317_131700

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Sequence a_n -/
noncomputable def a (x y : ℝ) : ℕ → ℝ
  | 0 => x
  | 1 => y
  | (n + 2) => (a x y (n + 1) * a x y n + 1) / (a x y (n + 1) + a x y n)

/-- General term of a_n -/
noncomputable def a_general (x y : ℝ) (n : ℕ) : ℝ :=
  ((x + 1)^(fib (n - 2)) * (y + 1)^(fib (n - 1)) + (x + 1)^(fib (n - 2)) * (y - 1)^(fib (n - 1))) /
  ((x + 1)^(fib (n - 2)) * (y + 1)^(fib (n - 1)) - (x - 1)^(fib (n - 2)) * (y - 1)^(fib (n - 1)))

theorem sequence_constant_condition (x y : ℝ) :
  (∃ n₀ : ℕ, ∀ n ≥ n₀, a x y n = a x y n₀) ↔ (abs x = 1 ∧ y ≠ -x) :=
by sorry

theorem general_term_correct (x y : ℝ) (n : ℕ) :
  a x y n = a_general x y n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_constant_condition_general_term_correct_l1317_131700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_median_difference_l1317_131736

/-- Represents the distribution of scores on an exam -/
structure ScoreDistribution where
  score70 : ℚ
  score80 : ℚ
  score85 : ℚ
  score90 : ℚ
  score95 : ℚ
  sum_to_100 : score70 + score80 + score85 + score90 + score95 = 100

/-- Calculates the mean score given a score distribution -/
def meanScore (d : ScoreDistribution) : ℚ :=
  (70 * d.score70 + 80 * d.score80 + 85 * d.score85 + 90 * d.score90 + 95 * d.score95) / 100

/-- Determines the median score given a score distribution -/
def medianScore (d : ScoreDistribution) : ℚ :=
  if d.score70 > 50 then 70
  else if d.score70 + d.score80 > 50 then 80
  else if d.score70 + d.score80 + d.score85 > 50 then 85
  else if d.score70 + d.score80 + d.score85 + d.score90 > 50 then 90
  else 95

/-- Theorem stating that the difference between mean and median is 4 for the given distribution -/
theorem mean_median_difference (d : ScoreDistribution) 
  (h1 : d.score70 = 15)
  (h2 : d.score80 = 35)
  (h3 : d.score85 = 10)
  (h4 : d.score90 = 25)
  (h5 : d.score95 = 15) :
  meanScore d - medianScore d = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_median_difference_l1317_131736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_slope_of_pq_l1317_131774

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 8 + y^2 / 2 = 1

-- Define point A
def point_A : ℝ × ℝ := (2, 1)

-- Define the property that a point is on the ellipse
def on_ellipse (p : ℝ × ℝ) : Prop := ellipse_C p.1 p.2

-- Define the angle bisector property
def angle_bisector_perpendicular (p q : ℝ × ℝ) : Prop :=
  (p.2 - point_A.2) / (p.1 - point_A.1) = -(q.2 - point_A.2) / (q.1 - point_A.1)

-- Define the slope of a line through two points
noncomputable def line_slope (p q : ℝ × ℝ) : ℝ := (q.2 - p.2) / (q.1 - p.1)

-- Theorem statement
theorem constant_slope_of_pq :
  ∀ p q : ℝ × ℝ,
  p ≠ q →
  on_ellipse p → on_ellipse q →
  angle_bisector_perpendicular p q →
  line_slope p q = 1/2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_slope_of_pq_l1317_131774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_in_fourth_quadrant_l1317_131757

theorem complex_in_fourth_quadrant (m : ℝ) : 
  (Complex.mk (m^2 - 8*m + 15) (m^2 - 5*m - 14)).re > 0 ∧ 
  (Complex.mk (m^2 - 8*m + 15) (m^2 - 5*m - 14)).im < 0 ↔ 
  ((-2 < m ∧ m < 3) ∨ (5 < m ∧ m < 7)) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_in_fourth_quadrant_l1317_131757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_f_l1317_131729

-- Define the function f as noncomputable due to its dependency on Real.pi
noncomputable def f (x : ℝ) : ℝ := (2 * Real.pi * x)^2

-- State the theorem
theorem derivative_of_f (x : ℝ) : 
  deriv f x = 8 * Real.pi^2 * x := by
  -- The proof is omitted and replaced with 'sorry'
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_f_l1317_131729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_not_three_sin_A_value_l1317_131745

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the condition a^2 + c^2 - b^2 = mac
def satisfiesCondition (t : Triangle) (m : ℝ) : Prop :=
  t.a^2 + t.c^2 - t.b^2 = m * t.a * t.c

-- Theorem 1: m cannot be equal to 3
theorem m_not_three (t : Triangle) (h : satisfiesCondition t 3) : False := by
  sorry

-- Theorem 2: If m = -1, b = 2√7, and c = 4, then sin A = √21/14
theorem sin_A_value (t : Triangle) 
  (h1 : satisfiesCondition t (-1)) 
  (h2 : t.b = 2 * Real.sqrt 7) 
  (h3 : t.c = 4) : 
  Real.sin (Real.arccos ((t.b^2 - t.a^2 - t.c^2) / (-2 * t.a * t.c))) = Real.sqrt 21 / 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_not_three_sin_A_value_l1317_131745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_digits_of_fraction_l1317_131746

theorem decimal_digits_of_fraction : ∃ (n : ℕ), 
  (5^7 : ℚ) / (8^3 * 125^2) = (n : ℚ) / 10^6 ∧ 
  1/10 ≤ (n : ℚ) / 10^6 ∧ 
  (n : ℚ) / 10^6 < 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_digits_of_fraction_l1317_131746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_min_omega_satisfies_l1317_131789

-- Define the function f(x)
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 6)

-- State the theorem
theorem min_omega_value (ω : ℝ) (h1 : ω > 0) 
  (h2 : ∀ x : ℝ, f ω x = f ω (Real.pi / 6 - x)) : ω ≥ 4 := by
  sorry

-- Define the minimum value of ω
def min_omega : ℝ := 4

-- Prove that min_omega satisfies the conditions
theorem min_omega_satisfies (h : min_omega > 0) 
  (h2 : ∀ x : ℝ, f min_omega x = f min_omega (Real.pi / 6 - x)) : 
  ∀ ω : ℝ, ω > 0 → (∀ x : ℝ, f ω x = f ω (Real.pi / 6 - x)) → ω ≥ min_omega := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_min_omega_satisfies_l1317_131789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_in_cylinder_l1317_131717

-- Define the cone's dimensions
def cone_radius : ℝ := 8
def cone_height : ℝ := 24

-- Define the cylinder's radius
def cylinder_radius : ℝ := 16

-- Function to calculate cone volume
noncomputable def cone_volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

-- Function to calculate cylinder volume
noncomputable def cylinder_volume (r h : ℝ) : ℝ := Real.pi * r^2 * h

-- Theorem stating that the height of water in the cylinder is 2 cm
theorem water_height_in_cylinder : 
  ∃ (h : ℝ), h = 2 ∧ 
  cone_volume cone_radius cone_height = cylinder_volume cylinder_radius h := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_in_cylinder_l1317_131717
